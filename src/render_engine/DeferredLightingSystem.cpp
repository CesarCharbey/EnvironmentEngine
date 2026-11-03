// =============================================================================
//  src/render_engine/DeferredLightingSystem.cpp                     [DEV 2]
//  Stage : LightingWater (ordre 0, avant WaterOceanRiverSystem)
//
//  Rôle : résolution plein écran de l'éclairage PBR différé (ADR-004) : soleil,
//  lune, IBL et ombre des nuages, écrite dans scene.color. S'exécute après le
//  stage AtmosphereIBL afin de disposer de l'IBL et des LUTs d'atmosphère.
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

class DeferredLightingSystem final : public ISystemModule {
public:
    void Init(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : shaders de résolution à créer dans shaders/render/ (triangle plein écran + fragment
        //            de résolution).
        // Sorties  : programme de résolution chargé par le ShaderManager et objet de sommets vide pour le
        //            triangle plein écran.
        // Objectif : préparer les ressources de la passe de résolution.
    }

    void Update(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : état de publication des entrées dans le blackboard (publiée ou fallback).
        // Sorties  : choix de la variante de résolution à utiliser cette frame.
        // Objectif : travail côté CPU uniquement (par exemple ignorer l'IBL tant qu'elle est en fallback).
    }

    void Render(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : via le blackboard (fallback automatique si une passe amont n'existe pas encore) :
        //            gbuffer.albedo, gbuffer.normal, gbuffer.material, gbuffer.depth, atmo.transmittanceLUT,
        //            ibl.irradiance, ibl.prefiltered, ibl.brdfLUT, clouds.shadowMap, volumetrics.godRays,
        //            liés sur les unités TEX_* correspondantes du contrat ; soleil, lune (phase et
        //            éclairement), météo et matrices inverses lus dans le GlobalEnvironmentUBO.
        // Sorties  : scene.color (HDR linéaire), cible déjà liée par le core.
        // Objectif : BRDF spéculaire GGX et diffuse ; soleil atténué par la transmittance ; lune comme
        //            lumière principale la nuit selon sa phase ; IBL en split-sum ; ombre des nuages ;
        //            humidité et neige de surface ; ciel là où la profondeur vaut 0 (reverse-Z). Une fois
        //            cette passe active, décocher « Draw Debug Preview » dans le panneau Mock World.
    }

    void Shutdown(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : ressources créées par Init.
        // Sorties  : objet de sommets libéré.
        // Objectif : libération propre avant la destruction du contexte OpenGL.
    }

    void OnGui(EngineContext&) override {
        TodoBanner("resolution PBR differee (soleil, lune, IBL, ombre des nuages) -> scene.color");
        ImGui::SliderFloat("Intensite soleil", &sunScale_, 0.0f, 4.0f);
        ImGui::SliderFloat("Intensite lune", &moonScale_, 0.0f, 4.0f);
        ImGui::SliderFloat("Intensite IBL", &iblScale_, 0.0f, 4.0f);
        ImGui::Checkbox("Occlusion speculaire", &specularOcclusion_);
        static const char* kViews[] = {"Lit", "Albedo", "Normal", "Roughness", "Metal", "AO", "Depth", "IBL only", "Sun only"};
        ImGui::Combo("Vue de debug", &debugView_, kViews, IM_ARRAYSIZE(kViews));
    }

private:
    // Variables liées aux sliders d'OnGui (aucune logique branchée).
    float sunScale_          = 1.0f;
    float moonScale_         = 1.0f;
    float iblScale_          = 1.0f;
    bool  specularOcclusion_ = true;
    int   debugView_         = 0;
};

}  // namespace
REGISTER_SYSTEM_ORDERED(DeferredLightingSystem, PassStage::LightingWater, 0);
}  // namespace env::render
