// =============================================================================
//  src/world_engine/BiomeHydrologySystem.cpp                        [DEV 1]
//  Stage : WorldCompute (ordre 0)
//
//  Rôle : génération GPU des chunks demandés : terrain procédural, biomes et
//  hydrologie multi-échelle (ADR-005) — macro (graphe de drainage), méso (champs
//  de distance signée par chunk), micro (consommé par le scatter et par l'eau).
//  Unique écrivain des niveaux d'eau du GlobalEnvironmentUBO.
//
//  Coquille vide : toute la logique est à écrire par Dev 1.
//  RÈGLE : ce fichier n'inclut jamais src/render_engine/ (vérifié au build).
// =============================================================================
#include <imgui.h>
#include <glad/glad.h>
#include <entt/entt.hpp>

#include "contracts/ISystemModule.hpp"
#include "contracts/WorldContracts.hpp"
#include "contracts/GpuBindings.hpp"
#include "core/SystemRegistry.hpp"
#include "core/gpu/ShaderManager.hpp"

namespace env::world {
namespace {

gpu::ShaderProgram* computeTerrain_ = nullptr;

void TodoBanner(const char* scope) {
    ImGui::TextColored(ImVec4(1.0f, 0.62f, 0.15f, 1.0f), "TODO Dev 1");
    ImGui::SameLine();
    ImGui::TextDisabled("%s", scope);
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
        // Si le shader n'a pas pu compiler (erreur de syntaxe), on annule
        if (!computeTerrain_ || !computeTerrain_->IsValid()) return;

        bool computeIssued = false;

        // On cherche tous les chunks demandés par le système de streaming
        auto view = ctx.registry.view<const ChunkCoordComponent, ChunkStateComponent, ChunkGpuResources>();

        int chunksProcessed = 0;

        for (auto entity : view) {
            // Limite de budget par frame pour ne pas faire ramer le GPU
            if (chunksProcessed >= maxChunksPerFrame_) break;

            auto& state = view.get<ChunkStateComponent>(entity);
            const auto& coord = view.get<const ChunkCoordComponent>(entity);
            const auto& gpuRes = view.get<ChunkGpuResources>(entity);

            // On regarde les masques (Bits) du contrat. 
            // Si la hauteur n'est pas encore calculée, ET que le streaming nous a donné une texture vierge :
            if ((state.readyMask & ChunkReady::HeightNormal) == 0 && gpuRes.heightNormalTex != 0) {
                
                // 1. Activation du Shader
                computeTerrain_->Use();
                
                // 2. Envoi des variables au Shader
                glm::dvec3 originWorld = ChunkOriginWorld(coord.coord);
                glProgramUniform2f(computeTerrain_->Id(), 0, static_cast<float>(originWorld.x), static_cast<float>(originWorld.z));
                glProgramUniform1f(computeTerrain_->Id(), 1, continentScale_);
                glProgramUniform1f(computeTerrain_->Id(), 2, heightAmplitude_);
                glProgramUniform1i(computeTerrain_->Id(), 3, octaves_);
                glProgramUniform1f(computeTerrain_->Id(), 4, lacunarity_);
                glProgramUniform1f(computeTerrain_->Id(), 5, gain_);
                glProgramUniform1i(computeTerrain_->Id(), 6, seed_);
                glProgramUniform1f(computeTerrain_->Id(), 7, moistureScale_);

                // 3. On lie la texture du chunk comme image de destination
                glBindImageTexture(env::bind::image::HeightNormalOut, gpuRes.heightNormalTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
                glBindImageTexture(env::bind::image::HeightNormalOut, gpuRes.heightNormalTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
                glBindImageTexture(env::bind::image::BiomeHydroOut, gpuRes.biomeHydroTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);


                // 4. Exécution ! (256/16 = 16 groupes de travail par axe)
                glDispatchCompute(kChunkTexResolution / 16, kChunkTexResolution / 16, 1);
                
                computeIssued = true;
                chunksProcessed++;

                // 5. On valide que le travail est fait (Bits HeightNormal ET BiomeHydro)
                state.readyMask |= (ChunkReady::HeightNormal | ChunkReady::BiomeHydro);
                
                // Pour que notre collègue l'affiche immédiatement (Jalon 1), on le passe en GpuReady
                if (state.state == ChunkState::Requested) {
                    state.state = ChunkState::GpuReady;
                }
            }
        }

        // 6. Règle absolue (ADR) : Si on a écrit sur la carte graphique, il faut mettre une barrière mémoire
        // pour empêcher le moteur de rendu de lire la texture tant que l'écriture n'est pas finie.
        if (computeIssued) {
            glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        }
    }

    void Shutdown(EngineContext&) override {

    }

    void OnGui(EngineContext&) override {
        TodoBanner("terrain procedural, biomes, hydrologie macro -> meso -> micro (ADR-005)");

        ImGui::SeparatorText("Niveaux d'eau (destines au GlobalEnvironmentUBO)");
        ImGui::SliderFloat("Niveau ocean (waterTime.x)", &oceanLevel_, -50.0f, 100.0f, "%.2f m");
        ImGui::SliderFloat("Niveau marais (waterTime.y)", &swampLevel_, -20.0f, 100.0f, "%.2f m");

        ImGui::SeparatorText("Terrain");
        ImGui::InputInt("Seed", &seed_);
        ImGui::SliderFloat("Echelle continentale", &continentScale_, 500.0f, 50000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Echelle montagnes", &mountainScale_, 50.0f, 5000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Amplitude verticale", &heightAmplitude_, 10.0f, 2000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderInt("Octaves", &octaves_, 1, 12);
        ImGui::SliderFloat("Lacunarite", &lacunarity_, 1.5f, 3.0f);
        ImGui::SliderFloat("Gain", &gain_, 0.2f, 0.8f);
        ImGui::SliderFloat("Domain warp", &domainWarp_, 0.0f, 500.0f, "%.0f m");

        ImGui::SeparatorText("Hydrologie macro");
        ImGui::SliderFloat("Taille cellule macro", &macroCellSize_, 512.0f, 16384.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Debit minimal riviere", &minRiverFlow_, 0.0f, 1.0f);
        ImGui::SliderFloat("Force des meandres", &meanderStrength_, 0.0f, 1.0f);

        ImGui::SeparatorText("Hydrologie meso");
        ImGui::DragFloatRange2("Largeur riviere", &riverWidthMin_, &riverWidthMax_, 0.5f, 1.0f, 200.0f, "%.1f m");
        ImGui::SliderFloat("Profondeur du lit", &riverDepth_, 0.5f, 20.0f, "%.1f m");
        ImGui::SliderFloat("Largeur des berges", &bankWidth_, 1.0f, 100.0f, "%.1f m");
        ImGui::SliderFloat("Seuil de marais (pente)", &marshSlopeThreshold_, 0.0f, 0.2f, "%.3f");

        ImGui::SeparatorText("Biomes");
        ImGui::SliderFloat("Echelle temperature", &temperatureScale_, 1000.0f, 100000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Echelle humidite", &moistureScale_, 1000.0f, 100000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Gradient thermique", &lapseRate_, 0.0f, 12.0f, "%.1f C / 1000 m");
        ImGui::SliderFloat("Hauteur plage max", &beachMaxHeight_, 0.0f, 20.0f, "%.1f m");
        ImGui::SliderFloat("Hauteur haute montagne", &highMountainHeight_, 100.0f, 3000.0f, "%.0f m");

        ImGui::SeparatorText("Budget");
        ImGui::SliderInt("Chunks generes max / frame", &maxChunksPerFrame_, 1, 16);
    }

private:
    // Variables liées aux sliders d'OnGui (aucune logique branchée).
    float oceanLevel_          = 0.0f;
    float swampLevel_          = 8.4f;
    int   seed_                = 1337;
    float continentScale_      = 8000.0f;
    float mountainScale_       = 900.0f;
    float heightAmplitude_     = 350.0f;
    int   octaves_             = 7;
    float lacunarity_          = 2.0f;
    float gain_                = 0.5f;
    float domainWarp_          = 80.0f;
    float macroCellSize_       = 4096.0f;
    float minRiverFlow_        = 0.15f;
    float meanderStrength_     = 0.5f;
    float riverWidthMin_       = 4.0f;
    float riverWidthMax_       = 60.0f;
    float riverDepth_          = 3.0f;
    float bankWidth_           = 18.0f;
    float marshSlopeThreshold_ = 0.02f;
    float temperatureScale_    = 20000.0f;
    float moistureScale_       = 12000.0f;
    float lapseRate_           = 6.5f;
    float beachMaxHeight_      = 3.0f;
    float highMountainHeight_  = 900.0f;
    int   maxChunksPerFrame_   = 2;
};

}  // namespace
REGISTER_SYSTEM_ORDERED(BiomeHydrologySystem, PassStage::WorldCompute, 0);
}  // namespace env::world
