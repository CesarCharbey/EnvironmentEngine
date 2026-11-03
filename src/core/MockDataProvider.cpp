// =============================================================================
//  src/core/MockDataProvider.cpp
//  Les fonctions analytiques ci-dessous sont des MIRES DE TEST (colline, sinus,
//  disque) destinées à valider la plomberie. Elles ne préjugent en rien des
//  algorithmes de génération de Dev 1.
// =============================================================================
#include "core/MockDataProvider.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <vector>

#include <imgui.h>

#include "contracts/GpuBindings.hpp"
#include "contracts/WorldContracts.hpp"
#include "core/RenderBlackboard.hpp"
#include "core/gpu/ShaderManager.hpp"

namespace env::core {
namespace {

constexpr std::string_view kProducer = "MockDataProvider";
constexpr float            kPi       = 3.14159265358979f;
constexpr float            kSize     = kChunkSizeMeters;

struct PreviewVertex {
    glm::vec3 position;
    glm::vec3 normal;
};

float SmoothStep(float edge0, float edge1, float x) {
    const float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

/// Mire analytique : colline gaussienne + vallée creusée le long d'une rivière sinusoïdale
/// + plateau marécageux circulaire. Coordonnées locales au chunk (m).
class AnalyticTerrain {
public:
    explicit AnalyticTerrain(const MockDataProvider::TerrainParams& p) : p_(p) {}

    float RiverPhase(float x) const { return 2.0f * kPi * p_.riverFrequency * x / kSize + 0.6f; }
    float RiverCenterZ(float x) const { return 0.5f * kSize + p_.riverAmplitude * std::sin(RiverPhase(x)); }

    /// Distance signée approchée au lit de la rivière (< 0 dans le lit).
    float RiverSdf(float x, float z) const {
        const float slope = p_.riverAmplitude * (2.0f * kPi * p_.riverFrequency / kSize) * std::cos(RiverPhase(x));
        const float dist  = std::fabs(z - RiverCenterZ(x)) / std::sqrt(1.0f + slope * slope);
        return dist - p_.riverHalfWidth;
    }

    glm::vec2 MarshCenter() const { return {0.22f * kSize, 0.80f * kSize}; }
    float     MarshSdf(float x, float z) const { return glm::length(glm::vec2(x, z) - MarshCenter()) - p_.marshRadius; }

    float Height(float x, float z) const {
        float h = 14.0f + 0.035f * x + 4.0f * std::sin(x * 0.031f) * std::cos(z * 0.027f) +
                  2.0f * std::sin((x + z) * 0.071f);
        const glm::vec2 hillCenter(0.70f * kSize, 0.28f * kSize);
        const glm::vec2 d = glm::vec2(x, z) - hillCenter;
        h += p_.hillHeight * std::exp(-glm::dot(d, d) / (2.0f * p_.hillRadius * p_.hillRadius));

        const float sdf    = RiverSdf(x, z);
        const float valley = 1.0f - SmoothStep(0.0f, 6.0f * p_.riverHalfWidth, std::max(sdf, 0.0f));
        const float bed    = SmoothStep(0.0f, -p_.riverHalfWidth, sdf);
        h -= p_.valleyDepth * valley + 2.5f * bed;

        const float marsh = 1.0f - SmoothStep(-4.0f, 18.0f, MarshSdf(x, z));
        return h + (p_.marshLevel - 0.6f - h) * marsh;
    }

    glm::vec3 Normal(float x, float z) const {
        constexpr float e = 0.5f;
        const float     l = Height(x - e, z), r = Height(x + e, z);
        const float     b = Height(x, z - e), f = Height(x, z + e);
        return glm::normalize(glm::vec3(l - r, 2.0f * e, b - f));
    }

    float Moisture(float x, float z) const {
        const float river = 0.75f * std::exp(-std::max(RiverSdf(x, z), 0.0f) / 45.0f);
        const float marsh = 1.0f - SmoothStep(-10.0f, 40.0f, MarshSdf(x, z));
        return std::clamp(0.12f + std::max(river, marsh), 0.0f, 1.0f);
    }

    static BiomeType Biome(float height, float riverSdf, float marshSdf, float moisture) {
        if (height < 1.0f) return BiomeType::Ocean;
        if (marshSdf < 0.0f) return BiomeType::Swamp;
        if (height > 52.0f) return BiomeType::HighMountain;
        if (riverSdf < 3.5f) return BiomeType::Beach;
        if (moisture > 0.62f) return BiomeType::Jungle;
        if (moisture > 0.42f) return BiomeType::Forest;
        if (moisture > 0.22f) return BiomeType::Plains;
        return BiomeType::Desert;
    }

private:
    const MockDataProvider::TerrainParams& p_;
};

struct XorShift32 {
    std::uint32_t state;
    float         Next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return static_cast<float>(state & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
    }
};

}  // namespace

void MockDataProvider::Init(gpu::ShaderManager& shaders) {
    program_ = &shaders.LoadGraphics("core.mock_preview", "core_debug/mock_preview.vert", "core_debug/mock_preview.frag");
    BuildPreviewMeshes();
}

void MockDataProvider::Shutdown(entt::registry& registry, RenderBlackboard& blackboard) {
    Destroy(registry, blackboard);
    terrainVao_.Destroy();
    cubeVao_.Destroy();
    terrainVertices_.Destroy();
    terrainIndices_.Destroy();
    cubeVertices_.Destroy();
    cubeIndices_.Destroy();
    program_ = nullptr;
}

void MockDataProvider::BuildPreviewMeshes() {
    // --- Grille de terrain : positions xz dans [0,1], la hauteur vient de heightNormalTex ---
    const int res = std::clamp(gridResolution, 8, 512);
    std::vector<PreviewVertex> vertices;
    vertices.reserve(static_cast<std::size_t>((res + 1) * (res + 1)));
    for (int j = 0; j <= res; ++j)
        for (int i = 0; i <= res; ++i)
            vertices.push_back({glm::vec3(static_cast<float>(i) / res, 0.0f, static_cast<float>(j) / res), glm::vec3(0, 1, 0)});
    std::vector<std::uint32_t> indices;
    indices.reserve(static_cast<std::size_t>(res * res * 6));
    for (int j = 0; j < res; ++j)
        for (int i = 0; i < res; ++i) {
            const auto a = static_cast<std::uint32_t>(j * (res + 1) + i);
            const auto b = a + 1, c = a + static_cast<std::uint32_t>(res + 1), d = c + 1;
            indices.insert(indices.end(), {a, c, b, b, c, d});
        }
    terrainVertices_.CreateFrom(std::span<const PreviewVertex>(vertices), 0, "Mock.TerrainVertices");
    terrainIndices_.CreateFrom(std::span<const std::uint32_t>(indices), 0, "Mock.TerrainIndices");
    terrainIndexCount_   = static_cast<GLsizei>(indices.size());
    builtGridResolution_ = gridResolution;

    terrainVao_.Create("Mock.TerrainVAO");
    terrainVao_.SetVertexBuffer(0, terrainVertices_, 0, sizeof(PreviewVertex));
    terrainVao_.SetElementBuffer(terrainIndices_);
    terrainVao_.SetAttribute(0, 0, 3, GL_FLOAT, offsetof(PreviewVertex, position));
    terrainVao_.SetAttribute(1, 0, 3, GL_FLOAT, offsetof(PreviewVertex, normal));

    if (cubeVao_.IsValid()) return;

    // --- Cube unitaire [-0.5, 0.5]^3 : 24 sommets, 36 indices (DrawElementsIndirectCommand::count = 36) ---
    struct Face {
        glm::vec3 n, u, v;
    };
    const Face faces[6] = {{{1, 0, 0}, {0, 0, -1}, {0, 1, 0}}, {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},
                           {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}}, {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
                           {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},  {{0, 0, -1}, {-1, 0, 0}, {0, 1, 0}}};
    std::vector<PreviewVertex> cubeVerts;
    std::vector<std::uint32_t> cubeIdx;
    for (const Face& f : faces) {
        const auto base = static_cast<std::uint32_t>(cubeVerts.size());
        const glm::vec3 c = f.n * 0.5f;
        cubeVerts.push_back({c + (-f.u - f.v) * 0.5f, f.n});
        cubeVerts.push_back({c + (f.u - f.v) * 0.5f, f.n});
        cubeVerts.push_back({c + (f.u + f.v) * 0.5f, f.n});
        cubeVerts.push_back({c + (-f.u + f.v) * 0.5f, f.n});
        cubeIdx.insert(cubeIdx.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    }
    cubeVertices_.CreateFrom(std::span<const PreviewVertex>(cubeVerts), 0, "Mock.CubeVertices");
    cubeIndices_.CreateFrom(std::span<const std::uint32_t>(cubeIdx), 0, "Mock.CubeIndices");
    cubeVao_.Create("Mock.CubeVAO");
    cubeVao_.SetVertexBuffer(0, cubeVertices_, 0, sizeof(PreviewVertex));
    cubeVao_.SetElementBuffer(cubeIndices_);
    cubeVao_.SetAttribute(0, 0, 3, GL_FLOAT, offsetof(PreviewVertex, position));
    cubeVao_.SetAttribute(1, 0, 3, GL_FLOAT, offsetof(PreviewVertex, normal));
}

void MockDataProvider::Generate(EngineContext& ctx) {
    const auto start = std::chrono::steady_clock::now();
    constexpr int N  = static_cast<int>(kChunkTexResolution);
    const AnalyticTerrain terrain(params);

    // --- Textures du chunk : texel (i, j) <-> local (x, z), centre de texel (contrat) ---
    std::vector<glm::vec4> heightNormal(static_cast<std::size_t>(N * N));
    std::vector<glm::vec4> biomeHydro(static_cast<std::size_t>(N * N));
    float hMin = 1e9f, hMax = -1e9f;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            const glm::vec2 p        = ChunkTexelCenterLocal(static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(j));
            const float     h        = terrain.Height(p.x, p.y);
            const glm::vec3 n        = terrain.Normal(p.x, p.y);
            const float     riverSdf = terrain.RiverSdf(p.x, p.y);
            const float     marshSdf = terrain.MarshSdf(p.x, p.y);
            const float     moisture = terrain.Moisture(p.x, p.y);
            const BiomeType biome    = AnalyticTerrain::Biome(h, riverSdf, marshSdf, moisture);
            const auto      idx      = static_cast<std::size_t>(j * N + i);
            heightNormal[idx] = glm::vec4(h, n.x, n.y, n.z);
            biomeHydro[idx]   = glm::vec4(static_cast<float>(biome), std::clamp(riverSdf, -kRiverSdfFar, kRiverSdfFar),
                                          std::clamp(marshSdf, -kRiverSdfFar, kRiverSdfFar), moisture);
            hMin = std::min(hMin, h);
            hMax = std::max(hMax, h);
        }
    }

    gpu::Texture2D::Desc desc;
    desc.width          = N;
    desc.height         = N;
    desc.internalFormat = GL_RGBA32F;
    desc.levels         = 1;
    desc.debugName      = "Mock.HeightNormal";
    heightNormal_.Create(desc);
    heightNormal_.Upload(heightNormal.data(), GL_RGBA, GL_FLOAT);

    desc.internalFormat = GL_RGBA16F;
    desc.debugName      = "Mock.BiomeHydro";
    biomeHydro_.Create(desc);
    biomeHydro_.Upload(biomeHydro.data(), GL_RGBA, GL_FLOAT);

    // --- Instances : 4 archétypes, triés par archétype, une commande indirecte chacun ---
    const auto  total     = static_cast<std::uint32_t>(std::clamp(params.instanceCount, 0, static_cast<int>(kFoliageMaxInstancesPerChunk)));
    const float shares[4] = {0.15f, 0.40f, 0.30f, 0.15f};  // Grass, Shrub, Tree, Rock
    std::vector<FoliageInstance>        instances;
    DrawElementsIndirectCommand         commands[kFoliageArchetypeCount];
    XorShift32                          rng{params.seed == 0 ? 1u : params.seed};
    std::uint32_t                       assigned = 0;
    instances.reserve(total);
    for (std::uint32_t a = 0; a < kFoliageArchetypeCount; ++a) {
        const std::uint32_t wanted = (a + 1 == kFoliageArchetypeCount)
                                         ? total - assigned
                                         : static_cast<std::uint32_t>(std::lround(static_cast<float>(total) * shares[a]));
        assigned += wanted;
        commands[a].count        = 36;
        commands[a].firstIndex   = 0;
        commands[a].baseVertex   = 0;
        commands[a].baseInstance = static_cast<std::uint32_t>(instances.size());
        for (std::uint32_t k = 0; k < wanted; ++k) {
            for (int attempt = 0; attempt < 24; ++attempt) {
                const float x = rng.Next() * kSize, z = rng.Next() * kSize;
                const float h = terrain.Height(x, z);
                const bool  dry = terrain.RiverSdf(x, z) > (a == 3 ? 0.5f : 2.5f) && terrain.MarshSdf(x, z) > 1.5f;
                if (!dry || (a == 2 && h > 50.0f)) continue;
                FoliageInstance inst;
                inst.positionScale = glm::vec4(x, h, z, 0.7f + 0.6f * rng.Next());
                inst.params        = glm::vec4(rng.Next() * 2.0f * kPi, static_cast<float>(a), std::floor(rng.Next() * 4.0f), rng.Next());
                instances.push_back(inst);
                break;
            }
        }
        commands[a].instanceCount = static_cast<std::uint32_t>(instances.size()) - commands[a].baseInstance;
        instanceCounts_[a]        = commands[a].instanceCount;
    }
    instances_.Create(static_cast<GLsizeiptr>(std::max<std::size_t>(instances.size(), 1) * sizeof(FoliageInstance)),
                      GL_DYNAMIC_STORAGE_BIT, instances.empty() ? nullptr : instances.data(), "Mock.FoliageInstances");
    indirect_.Create(sizeof(commands), GL_DYNAMIC_STORAGE_BIT, commands, "Mock.FoliageIndirectCommands");

    // --- Entité ECS : exactement les composants que produira ChunkStreamingSystem (Dev 1) ---
    entt::registry& registry = ctx.registry;
    entity_                  = registry.create();
    registry.emplace<ChunkCoordComponent>(entity_, ChunkCoordComponent{glm::ivec2(0, 0), 0});
    registry.emplace<ChunkStateComponent>(entity_, ChunkStateComponent{ChunkState::GpuReady, ChunkReady::All, 1u, ctx.time.time});
    ChunkGpuResources resources;
    resources.heightNormalTex          = heightNormal_.Id();
    resources.biomeHydroTex            = biomeHydro_.Id();
    resources.foliageIndirectCmdBuffer = indirect_.Id();
    resources.foliageInstanceSSBO      = instances_.Id();
    resources.foliageInstanceCapacity  = static_cast<std::uint32_t>(instances.size());
    resources.foliageCommandCount      = kFoliageArchetypeCount;
    resources.heightRange              = glm::vec2(hMin, hMax);
    registry.emplace<ChunkGpuResources>(entity_, resources);
    registry.emplace<MockChunkTag>(entity_);

    ctx.blackboard.PublishTexture(bb::WorldFocusHeightNormal, heightNormal_, kProducer);
    ctx.blackboard.PublishTexture(bb::WorldFocusBiomeHydro, biomeHydro_, kProducer);

    generatedHeightRange_ = glm::vec2(hMin, hMax);
    heightRange           = generatedHeightRange_;
    generated_            = true;
    generationMs_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::printf("[MockDataProvider] Chunk mock (0,0) genere en %.1f ms : %zu instances, hauteur [%.1f, %.1f] m\n",
                static_cast<double>(generationMs_), instances.size(), static_cast<double>(hMin), static_cast<double>(hMax));
}

void MockDataProvider::Destroy(entt::registry& registry, RenderBlackboard& blackboard) {
    // Contrat : retirer la vue ECS AVANT de détruire les objets GL qu'elle référence.
    if (entity_ != entt::null && registry.valid(entity_)) registry.destroy(entity_);
    entity_ = entt::null;
    blackboard.UnpublishIfProducer(bb::WorldFocusHeightNormal, kProducer);
    blackboard.UnpublishIfProducer(bb::WorldFocusBiomeHydro, kProducer);
    heightNormal_.Destroy();
    biomeHydro_.Destroy();
    instances_.Destroy();
    indirect_.Destroy();
    generated_ = false;
}

void MockDataProvider::Update(EngineContext& ctx) {
    if (useMockWorld && (!generated_ || regenerateRequested_)) {
        if (generated_) Destroy(ctx.registry, ctx.blackboard);
        Generate(ctx);
        regenerateRequested_ = false;
    } else if (!useMockWorld && generated_) {
        Destroy(ctx.registry, ctx.blackboard);
    }
}

void MockDataProvider::Render(EngineContext& ctx) {
    chunksDrawn_ = 0;
    if (!drawPreview || program_ == nullptr || !program_->IsValid()) return;
    if (builtGridResolution_ != gridResolution) BuildPreviewMeshes();

    const gpu::ShaderProgram& p = *program_;
    p.Set("uViewMode", static_cast<int>(viewMode));
    p.Set("uHeightRange", heightRange);
    p.Set("uSdfRange", sdfRange);
    p.Use();

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);  // Reverse-Z
    glDepthMask(GL_TRUE);
    glDisable(GL_CULL_FACE);
    if (wireframe) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    const GLuint neutralBiomeHydro = ctx.blackboard.FallbackTexture(bb::Fallback::NeutralBiomeHydro).id;
    const GLuint zeroBuffer        = ctx.blackboard.FallbackBuffer().id;

    // Même vue ECS que la G-pass de Dev 2 : mock ET vrais chunks de Dev 1.
    auto view = ctx.registry.view<const ChunkCoordComponent, const ChunkStateComponent, const ChunkGpuResources>();
    view.each([&](const ChunkCoordComponent& coord, const ChunkStateComponent& state, const ChunkGpuResources& res) {
        if (state.state == ChunkState::Evicting) return;
        if ((state.readyMask & ChunkReady::HeightNormal) == 0 || res.heightNormalTex == 0) return;

        p.Set("uChunkOrigin", ChunkOriginRelative(coord.coord, ctx.renderOriginChunk));
        glBindTextureUnit(bind::tex::HeightNormal, res.heightNormalTex);
        const bool hasBiome = (state.readyMask & ChunkReady::BiomeHydro) != 0 && res.biomeHydroTex != 0;
        glBindTextureUnit(bind::tex::BiomeHydro, hasBiome ? res.biomeHydroTex : neutralBiomeHydro);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::ssbo::FoliageInstances, zeroBuffer);

        if (drawTerrain) {
            p.Set("uDrawMode", 0);
            terrainVao_.Bind();
            glDrawElements(GL_TRIANGLES, terrainIndexCount_, GL_UNSIGNED_INT, nullptr);
        }

        const bool hasFoliage = drawInstances && (state.readyMask & ChunkReady::Foliage) != 0 &&
                                res.foliageInstanceSSBO != 0 && res.foliageIndirectCmdBuffer != 0 &&
                                res.foliageCommandCount > 0;
        if (hasFoliage) {
            p.Set("uDrawMode", 1);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::ssbo::FoliageInstances, res.foliageInstanceSSBO);
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, res.foliageIndirectCmdBuffer);
            cubeVao_.Bind();
            glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, nullptr,
                                        static_cast<GLsizei>(res.foliageCommandCount), 0);
        }
        ++chunksDrawn_;
    });

    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
    glBindVertexArray(0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void MockDataProvider::OnGui(EngineContext& ctx) {
    (void)ctx;
    ImGui::Checkbox("Use Mock World", &useMockWorld);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Cree une entite chunk (0,0) conforme a WorldContracts.hpp\n(MockChunkTag) + publie world.focus.*");
    ImGui::SameLine();
    if (ImGui::Button("Regenerer")) regenerateRequested_ = true;

    if (generated_) {
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "Chunk mock actif - genere en %.1f ms", static_cast<double>(generationMs_));
        ImGui::Text("Instances : herbe %u, buissons %u, arbres %u, rochers %u", instanceCounts_[0], instanceCounts_[1],
                    instanceCounts_[2], instanceCounts_[3]);
        ImGui::Text("Hauteur : [%.1f, %.1f] m", static_cast<double>(generatedHeightRange_.x),
                    static_cast<double>(generatedHeightRange_.y));
    } else {
        ImGui::TextDisabled("Chunk mock inactif");
    }

    ImGui::SeparatorText("Apercu de debug (tous les chunks prets)");
    ImGui::Checkbox("Draw Debug Preview", &drawPreview);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Dev 2 : decocher quand votre passe ecrit scene.color elle-meme");
    ImGui::SameLine();
    ImGui::Checkbox("Terrain", &drawTerrain);
    ImGui::SameLine();
    ImGui::Checkbox("Instances", &drawInstances);
    ImGui::SameLine();
    ImGui::Checkbox("Wireframe", &wireframe);

    static const char* kModes[] = {"Clay", "Height", "Normal", "Biome", "River SDF", "Marsh SDF", "Moisture", "Chunk Grid"};
    int mode = static_cast<int>(viewMode);
    if (ImGui::Combo("View Mode", &mode, kModes, IM_ARRAYSIZE(kModes))) viewMode = static_cast<ViewMode>(mode);
    ImGui::DragFloat2("Plage hauteur (m)", &heightRange.x, 0.5f);
    ImGui::SameLine();
    if (ImGui::SmallButton("Auto")) heightRange = generatedHeightRange_;
    ImGui::SliderFloat("Plage SDF (m)", &sdfRange, 1.0f, 200.0f, "%.0f");
    static const int   kGridValues[] = {32, 64, 128, 256};
    static const char* kGridNames[]  = {"32", "64", "128", "256"};
    int gridIndex = 2;
    for (int i = 0; i < 4; ++i)
        if (kGridValues[i] == gridResolution) gridIndex = i;
    if (ImGui::Combo("Resolution grille", &gridIndex, kGridNames, 4)) gridResolution = kGridValues[gridIndex];
    ImGui::Text("Chunks dessines : %u", chunksDrawn_);

    ImGui::SeparatorText("Mire analytique (regeneration au relachement)");
    auto regenOnRelease = [this] {
        if (ImGui::IsItemDeactivatedAfterEdit()) regenerateRequested_ = true;
    };
    ImGui::SliderFloat("Hauteur colline", &params.hillHeight, 0.0f, 150.0f, "%.0f m");
    regenOnRelease();
    ImGui::SliderFloat("Rayon colline", &params.hillRadius, 10.0f, 150.0f, "%.0f m");
    regenOnRelease();
    ImGui::SliderFloat("Amplitude riviere", &params.riverAmplitude, 0.0f, 100.0f, "%.0f m");
    regenOnRelease();
    ImGui::SliderFloat("Frequence riviere", &params.riverFrequency, 0.25f, 4.0f, "%.2f");
    regenOnRelease();
    ImGui::SliderFloat("Demi-largeur riviere", &params.riverHalfWidth, 1.0f, 30.0f, "%.1f m");
    regenOnRelease();
    ImGui::SliderFloat("Profondeur vallee", &params.valleyDepth, 0.0f, 40.0f, "%.1f m");
    regenOnRelease();
    ImGui::SliderFloat("Rayon marais", &params.marshRadius, 0.0f, 100.0f, "%.0f m");
    regenOnRelease();
    ImGui::SliderInt("Nombre d'instances", &params.instanceCount, 0, 20000);
    regenOnRelease();
    ImGui::InputScalar("Seed", ImGuiDataType_U32, &params.seed);
    regenOnRelease();
}

}  // namespace env::core
