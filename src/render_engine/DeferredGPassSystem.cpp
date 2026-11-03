// =============================================================================
//  src/render_engine/DeferredGPassSystem.cpp                        [DEV 2]
//  Stage : GBufferPass
//
//  Rôle : remplissage du G-Buffer du pipeline différé (ADR-004) à partir du
//  terrain et de la végétation de tous les chunks prêts (mock inclus). Ce système
//  possède le G-Buffer et le publie dans le blackboard (clés gbuffer.*).
//  La résolution de l'éclairage est faite par DeferredLightingSystem.
//
//  Coquille vide : toute la logique est à écrire par Dev 2.
//  RÈGLE : ce fichier n'inclut jamais src/world_engine/ (vérifié au build).
// =============================================================================
#include <imgui.h>

#include "contracts/ISystemModule.hpp"
#include "core/SystemRegistry.hpp"

namespace env::render {
namespace {

void TodoBanner(const char* scope) {
    ImGui::TextColored(ImVec4(0.35f, 0.80f, 1.0f, 1.0f), "TODO Dev 2");
    ImGui::SameLine();
    ImGui::TextDisabled("%s", scope);
}

class DeferredGPassSystem final : public ISystemModule {
public:
    void Init(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : taille du viewport fournie par EngineContext, formats du contrat (table des clés).
        // Sorties  : un G-Buffer multi-attachements possédé par ce système et publié dans le blackboard :
        //            gbuffer.albedo (SRGB8_ALPHA8 : albédo + occlusion), gbuffer.normal (RGBA16F : normale
        //            monde), gbuffer.material (RGBA8 : rugosité, métal, humidité / neige, identifiant de
        //            matériau), gbuffer.velocity (RG16F : mouvement en NDC), gbuffer.depth (DEPTH32F,
        //            reverse-Z) ; programmes du terrain et de la végétation (shaders/render/) ; une grille de
        //            terrain par niveau de détail ; les meshes des archétypes de végétation regroupés dans un
        //            buffer commun, dont le nombre d'indices et les décalages sont convenus avec Dev 1.
        // Objectif : préparer les ressources du remplissage du G-Buffer.
    }

    void Update(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : taille du viewport, échelle de résolution du panneau.
        // Sorties  : G-Buffer réalloué quand la taille cible change, puis toutes les clés gbuffer.*
        //            republiées (les identifiants des textures changent à chaque réallocation).
        // Objectif : travail côté CPU uniquement (taille, éventuelle liste de chunks visibles par
        //            intersection du frustum avec l'AABB déduite de heightRange).
    }

    void Render(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : entités chunk portant ChunkCoordComponent, ChunkStateComponent et ChunkGpuResources
        //            dont readyMask contient HeightNormal et dont l'état n'est pas Evicting (le chunk mock
        //            porte les mêmes composants) ; origine de chaque chunk relative à l'origine de rendu ;
        //            textures du chunk sur les unités TEX_HeightNormal et TEX_BiomeHydro ; instances de
        //            végétation sur SSBO_FoliageInstances et commandes indirectes du chunk ; vent et temps
        //            lus dans le GlobalEnvironmentUBO.
        // Sorties  : G-Buffer rempli (couleurs effacées puis profondeur effacée à 0, convention reverse-Z).
        // Objectif : passe terrain (grille déplacée par la hauteur, niveau de détail selon la distance,
        //            matériaux par biome, berges humides d'après la distance signée à la rivière) puis passe
        //            végétation en un seul tirage indirect multiple par chunk (option : culling GPU préalable
        //            vers SSBO_FoliageVisible). Le core réinitialise l'état et la cible de rendu ensuite.
    }

    void Shutdown(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : ressources créées par Init.
        // Sorties  : clés gbuffer.* retirées si ce système en est le producteur ; G-Buffer, maillages et
        //            buffers libérés.
        // Objectif : libération propre avant la destruction du contexte OpenGL.
    }

    void OnGui(EngineContext&) override {
        TodoBanner("G-pass terrain + vegetation indirecte -> G-Buffer (ADR-004)");
        ImGui::SliderFloat("Echelle de resolution", &resolutionScale_, 0.25f, 1.0f, "%.2f");
        ImGui::Checkbox("Terrain", &drawTerrain_);
        ImGui::SameLine();
        ImGui::Checkbox("Vegetation", &drawFoliage_);
        ImGui::SameLine();
        ImGui::Checkbox("Wireframe", &wireframe_);
        ImGui::SliderInt("Resolution grille terrain", &terrainGridResolution_, 16, 256);
        ImGui::DragFloat3("Distances LOD terrain", lodDistances_, 4.0f, 16.0f, 8192.0f, "%.0f m");
        ImGui::SliderFloat("Distance vegetation", &foliageDrawDistance_, 32.0f, 4096.0f, "%.0f m");
    }

private:
    // Variables liées aux sliders d'OnGui (aucune logique branchée).
    float resolutionScale_       = 1.0f;
    bool  drawTerrain_           = true;
    bool  drawFoliage_           = true;
    bool  wireframe_             = false;
    int   terrainGridResolution_ = 128;
    float lodDistances_[3]       = {256.0f, 768.0f, 2048.0f};
    float foliageDrawDistance_   = 600.0f;
};

}  // namespace
REGISTER_SYSTEM(DeferredGPassSystem, PassStage::GBufferPass);
}  // namespace env::render
