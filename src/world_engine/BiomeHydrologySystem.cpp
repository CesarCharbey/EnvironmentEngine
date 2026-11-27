// src/world_engine/BiomeHydrologySystem.cpp
#include <algorithm>
#include <cmath>
#include <limits>

#include <imgui.h>
#include <glad/glad.h>
#include <entt/entt.hpp>
#include <glm/glm.hpp>

#include "contracts/ISystemModule.hpp"
#include "contracts/WorldContracts.hpp"
#include "contracts/GpuBindings.hpp"
#include "core/SystemRegistry.hpp"
#include "core/gpu/ShaderManager.hpp"

namespace env::world {
namespace {

// Miroir CPU léger du bruit du shader pour estimer un heightRange serré sans stall GPU
glm::vec2 Hash22Cpu(glm::vec2 p, int seed) {
    p += static_cast<float>(seed) * 0.1337f;
    glm::vec3 p3 = glm::fract(glm::vec3(p.x, p.y, p.x) * glm::vec3(0.1031f, 0.1030f, 0.0973f));
    p3 += glm::dot(p3, glm::vec3(p3.y, p3.z, p3.x) + 33.33f);
    return glm::fract(glm::vec2(p3.x + p3.y, p3.x + p3.z) * glm::vec2(p3.z, p3.y)) * 2.0f - 1.0f;
}

float PerlinCpu(glm::vec2 p, int seed) {
    glm::vec2 i = glm::floor(p);
    glm::vec2 f = glm::fract(p);
    glm::vec2 u = f * f * (3.0f - 2.0f * f);
    float a = glm::dot(Hash22Cpu(i + glm::vec2(0.0f, 0.0f), seed), f - glm::vec2(0.0f, 0.0f));
    float b = glm::dot(Hash22Cpu(i + glm::vec2(1.0f, 0.0f), seed), f - glm::vec2(1.0f, 0.0f));
    float c = glm::dot(Hash22Cpu(i + glm::vec2(0.0f, 1.0f), seed), f - glm::vec2(0.0f, 1.0f));
    float d = glm::dot(Hash22Cpu(i + glm::vec2(1.0f, 1.0f), seed), f - glm::vec2(1.0f, 1.0f));
    return glm::mix(glm::mix(a, b, u.x), glm::mix(c, d, u.x), u.y);
}

class BiomeHydrologySystem final : public ISystemModule {
public:
    void Init(EngineContext& ctx) override {
        computeTerrain_ = &ctx.shaders.LoadCompute("world.terrain_fbm", "world/terrain_fbm.comp");
    }

    void Update(EngineContext& ctx) override {
        ctx.environment.waterTime.x = oceanLevel_;
        ctx.environment.waterTime.y = swampLevel_;
    }

    void Render(EngineContext& ctx) override {
        if (!computeTerrain_ || !computeTerrain_->IsValid()) return;

        bool computeIssued = false;
        auto view = ctx.registry.view<const ChunkCoordComponent, ChunkStateComponent, ChunkGpuResources>(
            entt::exclude<MockChunkTag>);

        int chunksProcessed = 0;

        for (auto entity : view) {
            if (chunksProcessed >= maxChunksPerFrame_) break;

            auto& state  = view.get<ChunkStateComponent>(entity);
            auto& gpuRes = view.get<ChunkGpuResources>(entity);
            const auto& coord = view.get<const ChunkCoordComponent>(entity);

            if (state.state == ChunkState::Evicting || gpuRes.heightNormalTex == 0) continue;

            // Régénération progressive : on ne vide jamais readyMask, on compare generation
            const bool needsGen = ((state.readyMask & ChunkReady::HeightNormal) == 0)
                               || (state.generation < targetGeneration_);

            if (needsGen) {
                computeTerrain_->Use();

                const GLuint id = computeTerrain_->Id();
                glProgramUniform2i(id, 0,  coord.coord.x, coord.coord.y);
                glProgramUniform1f(id, 1,  continentScale_);
                glProgramUniform1f(id, 2,  mountainScale_);
                glProgramUniform1f(id, 3,  heightAmplitude_);
                glProgramUniform1i(id, 4,  octaves_);
                glProgramUniform1f(id, 5,  lacunarity_);
                glProgramUniform1f(id, 6,  gain_);
                glProgramUniform1f(id, 7,  domainWarp_);
                glProgramUniform1i(id, 8,  seed_);
                glProgramUniform1f(id, 9,  moistureScale_);
                glProgramUniform1f(id, 10, oceanLevel_);
                glProgramUniform1f(id, 11, swampLevel_);
                glProgramUniform1f(id, 12, beachMaxHeight_);
                glProgramUniform1f(id, 13, highMountainHeight_);
                glProgramUniform1f(id, 14, riverWidthMax_ * 0.5f);
                glProgramUniform1f(id, 15, riverDepth_);
                glProgramUniform1f(id, 16, bankWidth_);
                glProgramUniform1f(id, 17, marshSlopeThreshold_);

                glBindImageTexture(env::bind::image::HeightNormalOut, gpuRes.heightNormalTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
                glBindImageTexture(env::bind::image::BiomeHydroOut,   gpuRes.biomeHydroTex,   0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

                glDispatchCompute(kChunkTexResolution / 16, kChunkTexResolution / 16, 1);

                // Estimation serrée du min/max réel de ce chunk (5x5 échantillons CPU + marge)
                gpuRes.heightRange = EstimateChunkHeightRange(coord.coord);

                state.readyMask |= (ChunkReady::HeightNormal | ChunkReady::BiomeHydro);
                state.generation = targetGeneration_;
                if (state.state == ChunkState::Requested) {
                    state.state = ChunkState::GpuReady;
                }

                computeIssued = true;
                chunksProcessed++;
            }
        }

        if (computeIssued) {
            glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        }
    }

    void Shutdown(EngineContext&) override {}

    void OnGui(EngineContext&) override {
        bool changed = false;

        ImGui::SeparatorText("Niveaux d'eau (GlobalEnvironmentUBO)");
        changed |= ImGui::SliderFloat("Niveau ocean (waterTime.x)", &oceanLevel_, -50.0f, 100.0f, "%.2f m");
        changed |= ImGui::SliderFloat("Niveau marais (waterTime.y)", &swampLevel_, -20.0f, 100.0f, "%.2f m");

        ImGui::SeparatorText("Terrain (Infinigen Warp + Voronoi)");
        changed |= ImGui::InputInt("Seed", &seed_);
        changed |= ImGui::SliderFloat("Echelle continentale", &continentScale_, 500.0f, 50000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        changed |= ImGui::SliderFloat("Echelle montagnes", &mountainScale_, 50.0f, 5000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        changed |= ImGui::SliderFloat("Amplitude verticale", &heightAmplitude_, 10.0f, 2000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        changed |= ImGui::SliderInt("Octaves", &octaves_, 1, 12);
        changed |= ImGui::SliderFloat("Lacunarite", &lacunarity_, 1.5f, 3.0f);
        changed |= ImGui::SliderFloat("Gain", &gain_, 0.2f, 0.8f);
        changed |= ImGui::SliderFloat("Domain warp", &domainWarp_, 0.0f, 500.0f, "%.0f m");

        ImGui::SeparatorText("Hydrologie meso");
        changed |= ImGui::DragFloatRange2("Largeur riviere", &riverWidthMin_, &riverWidthMax_, 0.5f, 1.0f, 200.0f, "%.1f m");
        changed |= ImGui::SliderFloat("Profondeur du lit", &riverDepth_, 0.5f, 30.0f, "%.1f m");
        changed |= ImGui::SliderFloat("Largeur des berges", &bankWidth_, 1.0f, 100.0f, "%.1f m");
        changed |= ImGui::SliderFloat("Seuil de marais (pente)", &marshSlopeThreshold_, 0.0f, 0.2f, "%.3f");

        ImGui::SeparatorText("Biomes");
        changed |= ImGui::SliderFloat("Echelle humidite", &moistureScale_, 1000.0f, 100000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        changed |= ImGui::SliderFloat("Hauteur plage max", &beachMaxHeight_, 0.0f, 20.0f, "%.1f m");
        changed |= ImGui::SliderFloat("Hauteur haute montagne", &highMountainHeight_, 100.0f, 3000.0f, "%.0f m");

        ImGui::SeparatorText("Budget");
        ImGui::SliderInt("Chunks generes max / frame", &maxChunksPerFrame_, 1, 16);

        if (changed || ImGui::Button("Forcer regeneration")) {
            targetGeneration_++;
        }
    }

private:
    float EvalBaseHeightCpu(glm::vec2 worldPos) const {
        float continent = PerlinCpu(worldPos / continentScale_, seed_) * 0.5f + 0.5f;
        float mountainMask = glm::smoothstep(0.35f, 0.78f, continent);
        float baseElevation = std::pow(continent, 1.8f) * heightAmplitude_;

        glm::vec2 warpedPos = worldPos;
        if (domainWarp_ > 0.001f) {
            glm::vec2 q = worldPos / mountainScale_;
            warpedPos += glm::vec2(
                PerlinCpu(q + glm::vec2(5.2f, 1.3f), seed_),
                PerlinCpu(q + glm::vec2(1.7f, 9.2f), seed_)
            ) * domainWarp_;
        }

        glm::vec2 freq = warpedPos / mountainScale_;
        float amplitude = heightAmplitude_ * 0.45f;
        float detailHeight = 0.0f;

        // 4 octaves suffisent pour une borne précise sur une grille 5x5
        const int sampleOctaves = std::min(octaves_, 4);
        for (int i = 0; i < sampleOctaves; ++i) {
            float n = PerlinCpu(freq, seed_);
            float ridgedN = 1.0f - std::abs(n);
            ridgedN = (ridgedN * ridgedN) * 2.0f - 1.0f;
            detailHeight += glm::mix(n * 0.25f, ridgedN, mountainMask) * amplitude;
            freq *= lacunarity_;
            amplitude *= gain_;
        }

        return baseElevation + detailHeight - (heightAmplitude_ * 0.35f);
    }

    glm::vec2 EstimateChunkHeightRange(glm::ivec2 chunkCoord) const {
        float minH = std::numeric_limits<float>::max();
        float maxH = std::numeric_limits<float>::lowest();

        const glm::vec2 origin = glm::vec2(chunkCoord) * kChunkSizeMeters;
        constexpr int kGrid = 5;
        for (int y = 0; y < kGrid; ++y) {
            for (int x = 0; x < kGrid; ++x) {
                glm::vec2 p = origin + glm::vec2(x, y) * (kChunkSizeMeters / float(kGrid - 1));
                float h = EvalBaseHeightCpu(p);
                minH = std::min(minH, h);
                maxH = std::max(maxH, h);
            }
        }

        // Marge de sécurité pour les hautes octaves, le Voronoi et le creusement de rivière
        const float detailMargin = heightAmplitude_ * 0.08f + 8.0f;
        return glm::vec2(minH - detailMargin - riverDepth_, maxH + detailMargin);
    }

    gpu::ShaderProgram* computeTerrain_   = nullptr;
    std::uint32_t       targetGeneration_ = 1;

    float oceanLevel_          = 0.0f;
    float swampLevel_          = 8.4f;
    int   seed_                = 1337;
    float continentScale_      = 6000.0f;
    float mountainScale_       = 900.0f;
    float heightAmplitude_     = 500.0f;
    int   octaves_             = 7;
    float lacunarity_          = 2.0f;
    float gain_                = 0.5f;
    float domainWarp_          = 120.0f;
    float riverWidthMin_       = 4.0f;
    float riverWidthMax_       = 40.0f;
    float riverDepth_          = 12.0f;
    float bankWidth_           = 25.0f;
    float marshSlopeThreshold_ = 0.04f;
    float moistureScale_       = 3000.0f;
    float beachMaxHeight_      = 3.0f;
    float highMountainHeight_  = 350.0f;
    int   maxChunksPerFrame_   = 4;
};

}  // namespace
REGISTER_SYSTEM_ORDERED(BiomeHydrologySystem, PassStage::WorldCompute, 0);
}  // namespace env::world