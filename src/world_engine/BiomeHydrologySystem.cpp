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

#include "contracts/ISystemModule.hpp"
#include "core/SystemRegistry.hpp"

namespace env::world {
namespace {

void TodoBanner(const char* scope) {
    ImGui::TextColored(ImVec4(1.0f, 0.62f, 0.15f, 1.0f), "TODO Dev 1");
    ImGui::SameLine();
    ImGui::TextDisabled("%s", scope);
}

class BiomeHydrologySystem final : public ISystemModule {
public:
    void Init(EngineContext&) override {
        // [DEV 1 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : shaders compute à créer dans shaders/world/ ; slots du contrat UBO_WorldChunkParams,
        //            UBO_BiomeHydroParams, IMG_HeightNormalOut, IMG_BiomeHydroOut, SSBO_HydroGraphNodes,
        //            SSBO_HydroGraphEdges.
        // Sorties  : programmes compute chargés par le ShaderManager (terrain, macro-hydrologie,
        //            méso-hydrologie et biomes), UBO de paramètres, buffers du macro-graphe, carte macro.
        // Objectif : préparer une fois les ressources réutilisées par toutes les générations de chunks.
    }

    void Update(EngineContext&) override {
        // [DEV 1 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : entités chunk à l'état Requested (hors MockChunkTag), budget de chunks par frame,
        //            position de la caméra, paramètres du panneau (dont les niveaux d'eau).
        // Sorties  : waterTime.x (niveau de l'océan) et waterTime.y (niveau du marais) écrits dans le
        //            GlobalEnvironmentUBO — ce système en est l'unique écrivain ; un jeu de ressources du
        //            réservoir attribué (composant ChunkGpuResources) et l'état Generating pour chaque chunk
        //            retenu ; la liste des chunks à générer cette frame ; les cellules du macro-graphe proches
        //            de la caméra mises à jour et publiées sous world.hydro.graphNodes,
        //            world.hydro.graphEdges et world.hydro.macroMap.
        // Objectif : sélection côté CPU / ECS uniquement. Graphe de drainage déterministe calculable
        //            localement à partir de la graine (bassins versants, directions d'écoulement, débits,
        //            confluences, embouchures), mis en cache autour de la caméra.
    }

    void Render(EngineContext&) override {
        // [DEV 1 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : chunks retenus par Update, leurs coordonnées entières (précision illimitée), le
        //            macro-graphe, les paramètres de terrain, d'hydrologie et de biomes.
        // Sorties  : heightNormalTex (R = hauteur absolue en mètres, GBA = normale monde unitaire) ;
        //            biomeHydroTex (R = identifiant de biome entier, G = distance signée à la rivière en
        //            mètres, négative dans le lit, B = distance signée au marais, A = humidité dans [0, 1],
        //            valeur kRiverSdfFar loin de toute eau) ; heightRange du chunk ; readyMask complété des
        //            bits HeightNormal et BiomeHydro ; état GpuReady.
        // Objectif : 1) terrain procédural évalué en coordonnées monde (bruit fractal, déformation de
        //            domaine, érosion approchée) pour des coutures continues entre chunks ;
        //            2) rastérisation des segments macro traversant le chunk en distances signées, creusement
        //            du terrain cohérent avec le lit ; 3) classification des biomes (température, humidité,
        //            altitude, proximité de l'eau). Respecter la convention de centre de texel du contrat et
        //            insérer une barrière mémoire entre les étapes dépendantes.
    }

    void Shutdown(EngineContext&) override {
        // [DEV 1 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : ressources créées par Init.
        // Sorties  : buffers du macro-graphe, carte macro et UBO libérés ; clés world.hydro.* retirées.
        // Objectif : libération propre avant la destruction du contexte OpenGL.
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
