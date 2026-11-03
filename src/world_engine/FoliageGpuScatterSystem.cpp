// =============================================================================
//  src/world_engine/FoliageGpuScatterSystem.cpp                     [DEV 1]
//  Stage : WorldCompute (ordre 10, après BiomeHydrologySystem)
//
//  Rôle : placement GPU de la végétation par chunk (ADR-001). Produit, pour chaque
//  chunk, le tableau d'instances de végétation et une commande de tirage indirect
//  par archétype ; Dev 2 dessine ensuite le chunk entier en un seul tirage
//  indirect multiple.
//
//  Coquille vide : toute la logique est à écrire par Dev 1.
//  RÈGLE : ce fichier n'inclut jamais src/render_engine/ (vérifié au build).
// =============================================================================
#include <array>
#include <cstddef>
#include <string>
#include <string_view>

#include <imgui.h>

#include "contracts/ISystemModule.hpp"
#include "contracts/WorldContracts.hpp"
#include "core/SystemRegistry.hpp"

namespace env::world {
namespace {

void TodoBanner(const char* scope) {
    ImGui::TextColored(ImVec4(1.0f, 0.62f, 0.15f, 1.0f), "TODO Dev 1");
    ImGui::SameLine();
    ImGui::TextDisabled("%s", scope);
}

class FoliageGpuScatterSystem final : public ISystemModule {
public:
    void Init(EngineContext&) override {
        // [DEV 1 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : shader compute de scatter à créer dans shaders/world/ ; slots du contrat
        //            SSBO_FoliageInstances, SSBO_FoliageIndirectCmds, SSBO_FoliageCounters,
        //            UBO_FoliageParams, TEX_HeightNormal, TEX_BiomeHydro.
        // Sorties  : programme compute chargé par le ShaderManager, UBO de paramètres, buffer de compteurs
        //            atomiques publié sous la clé world.foliage.counters.
        // Objectif : préparer une fois les ressources communes à tous les chunks.
    }

    void Update(EngineContext&) override {
        // [DEV 1 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : entités chunk (hors MockChunkTag) dont readyMask contient HeightNormal et BiomeHydro
        //            mais pas encore Foliage, et dont ChunkGpuResources référence un buffer d'instances et
        //            un buffer de commandes ; budget de chunks par frame ; paramètres du panneau.
        // Sorties  : la liste des chunks à traiter cette frame ; quand les paramètres changent, le bit
        //            Foliage retiré et le compteur generation incrémenté pour les chunks concernés.
        // Objectif : sélection côté CPU / ECS uniquement, avec un LOD de densité selon la distance.
    }

    void Render(EngineContext&) override {
        // [DEV 1 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : textures hauteur + normale et biome + hydrologie de chaque chunk retenu, matrice de
        //            densité biome x archétype, pente maximale, marge de berge, plage d'échelle, graine.
        // Sorties  : le tableau FoliageInstance du chunk (position locale au chunk, échelle, lacet,
        //            archétype, variante, aléa), partitionné par archétype ; une commande
        //            DrawElementsIndirectCommand par archétype — nombre d'indices du mesh de l'archétype
        //            convenu avec Dev 2, nombre d'instances rempli par compteur atomique, instance de base
        //            égale au début de la partition ; bit Foliage ajouté à readyMask.
        // Objectif : placement GPU par cellules (jitter ou Poisson), rejet selon la pente, la distance
        //            signée à la rivière et au marais, densité pondérée par biome ; barrière mémoire couvrant
        //            les buffers de stockage et les commandes indirectes avant le tirage de Dev 2.
    }

    void Shutdown(EngineContext&) override {
        // [DEV 1 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : ressources créées par Init.
        // Sorties  : UBO et buffer de compteurs libérés, clé world.foliage.counters retirée.
        // Objectif : libération propre avant la destruction du contexte OpenGL.
    }

    void OnGui(EngineContext&) override {
        TodoBanner("scatter GPU : instances + commandes indirectes par archetype (ADR-001)");

        ImGui::SeparatorText("Densite");
        ImGui::SliderFloat("Densite globale", &globalDensity_, 0.0f, 50.0f, "%.2f inst / 100 m2");
        ImGui::SliderInt("Instances max / chunk", &maxInstancesPerChunk_, 0, static_cast<int>(kFoliageMaxInstancesPerChunk));
        if (ImGui::BeginTable("##density", 1 + static_cast<int>(kFoliageArchetypeCount),
                              ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableSetupColumn("Biome");
            for (const std::string_view name : kFoliageArchetypeNames) ImGui::TableSetupColumn(std::string(name).c_str());
            ImGui::TableHeadersRow();
            for (std::size_t b = 0; b < kBiomeCount; ++b) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(kBiomeNames[b].data(), kBiomeNames[b].data() + kBiomeNames[b].size());
                for (std::size_t a = 0; a < kFoliageArchetypeCount; ++a) {
                    ImGui::TableNextColumn();
                    ImGui::PushID(static_cast<int>(b * kFoliageArchetypeCount + a));
                    ImGui::SetNextItemWidth(-1.0f);
                    ImGui::DragFloat("##d", &densityMatrix_[b][a], 0.01f, 0.0f, 4.0f, "%.2f");
                    ImGui::PopID();
                }
            }
            ImGui::EndTable();
        }

        ImGui::SeparatorText("Placement");
        ImGui::SliderFloat("Pente max", &slopeLimitDeg_, 0.0f, 90.0f, "%.0f deg");
        ImGui::SliderFloat("Marge berges", &riverBankMargin_, 0.0f, 20.0f, "%.1f m");
        ImGui::DragFloatRange2("Echelle", &minScale_, &maxScale_, 0.01f, 0.1f, 4.0f, "%.2f");
        ImGui::SliderFloat("Distance LOD densite", &lodDistance_, 32.0f, 2048.0f, "%.0f m");
        ImGui::InputInt("Seed scatter", &seed_);
        ImGui::SliderInt("Chunks traites max / frame", &maxChunksPerFrame_, 1, 16);
    }

private:
    // Variables liées aux sliders d'OnGui (aucune logique branchée).
    float globalDensity_        = 6.0f;
    int   maxInstancesPerChunk_ = 32768;
    float slopeLimitDeg_        = 38.0f;
    float riverBankMargin_      = 2.0f;
    float minScale_             = 0.7f;
    float maxScale_             = 1.3f;
    float lodDistance_          = 512.0f;
    int   seed_                 = 4242;
    int   maxChunksPerFrame_    = 2;
    // Densité relative par biome (lignes) et archétype (colonnes : herbe, buisson, arbre, rocher).
    std::array<std::array<float, kFoliageArchetypeCount>, kBiomeCount> densityMatrix_{{
        {0.0f, 0.0f, 0.0f, 0.0f},    // Ocean
        {0.1f, 0.05f, 0.0f, 0.3f},   // Beach
        {0.05f, 0.15f, 0.0f, 0.6f},  // Desert
        {1.0f, 0.3f, 0.05f, 0.1f},   // Plains
        {0.6f, 0.6f, 1.0f, 0.15f},   // Forest
        {0.8f, 1.0f, 1.0f, 0.05f},   // Jungle
        {0.7f, 0.5f, 0.3f, 0.0f},    // Swamp
        {0.1f, 0.05f, 0.0f, 1.0f},   // HighMountain
    }};
};

}  // namespace
REGISTER_SYSTEM_ORDERED(FoliageGpuScatterSystem, PassStage::WorldCompute, 10);
}  // namespace env::world
