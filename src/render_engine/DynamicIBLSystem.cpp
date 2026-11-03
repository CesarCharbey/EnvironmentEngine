// =============================================================================
//  src/render_engine/DynamicIBLSystem.cpp                           [DEV 2]
//  Stage : AtmosphereIBL (ordre 10, après CelestialAtmosphereSystem)
//
//  Rôle : éclairage par image dynamique, réparti sur plusieurs frames
//  (time-slicing, ADR-004), avec cubemaps en double tampon : on écrit dans le
//  tampon arrière, on publie le tampon avant, l'échange a lieu en fin de cycle.
//  Tant que rien n'est publié, les consommateurs reçoivent la cubemap grise de
//  fallback du blackboard.
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

class DynamicIBLSystem final : public ISystemModule {
public:
    void Init(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : résolutions du panneau ; shaders à créer dans shaders/render/ ; slots du contrat
        //            UBO_IblParams, IMG_IblOut, SSBO_IblShCoefficients.
        // Sorties  : deux jeux de cubemaps (avant / arrière) : environnement RGBA16F avec chaîne de mips
        //            (mip = rugosité), irradiance 32 x 32 (ou harmoniques sphériques d'ordre 2) ; LUT BRDF
        //            RG16F 512 x 512 calculée une seule fois ; cible de capture attachant une face à la fois ;
        //            programmes de capture, de convolution et de préfiltrage.
        // Objectif : préparer les ressources de l'IBL dynamique.
    }

    void Update(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : durée de la frame, direction du soleil lue dans le GlobalEnvironmentUBO, paramètres
        //            du panneau (intervalle, seuil angulaire, gel).
        // Sorties  : décision de démarrer un nouveau cycle de mise à jour.
        // Objectif : travail côté CPU uniquement. Démarrer un cycle quand l'intervalle est écoulé, quand
        //            le soleil a tourné de plus que le seuil depuis le dernier cycle, ou sur demande ;
        //            jamais pendant un cycle en cours ni quand l'IBL est gelée.
    }

    void Render(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : atmo.skyViewLUT et atmo.transmittanceLUT (blackboard), soleil et lune
        //            (GlobalEnvironmentUBO), budget d'étapes par frame.
        // Sorties  : en fin de cycle, échange avant / arrière puis publication de ibl.environment,
        //            ibl.irradiance, ibl.prefiltered et ibl.brdfLUT.
        // Objectif : machine à états time-sliced exécutant au plus N étapes par frame : capture du ciel
        //            dans chacune des 6 faces, convolution de l'irradiance, préfiltrage GGX à raison d'un mip
        //            par étape (rugosité = mip / (nombre de mips - 1)), puis publication. Ne jamais exposer
        //            une cubemap partiellement mise à jour ; barrière mémoire entre étapes dépendantes.
    }

    void Shutdown(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : ressources créées par Init.
        // Sorties  : clés ibl.* retirées, cubemaps, LUT et cible de capture libérées.
        // Objectif : libération propre avant la destruction du contexte OpenGL.
    }

    void OnGui(EngineContext&) override {
        TodoBanner("capture du ciel, irradiance, prefiltrage GGX, BRDF LUT (time-slicing)");
        ImGui::SliderInt("Resolution cubemap", &environmentResolution_, 64, 512);
        ImGui::SliderInt("Resolution irradiance", &irradianceResolution_, 8, 64);
        ImGui::SliderInt("Mips prefiltres", &prefilterMips_, 1, 9);
        ImGui::SliderInt("Etapes / frame", &stepsPerFrame_, 1, 16);
        ImGui::SliderFloat("Intervalle de mise a jour", &updateInterval_, 0.1f, 30.0f, "%.1f s");
        ImGui::SliderFloat("Seuil angulaire soleil", &sunAngleThresholdDeg_, 0.1f, 20.0f, "%.1f deg");
        ImGui::Checkbox("Geler", &freeze_);
    }

private:
    // Variables liées aux sliders d'OnGui (aucune logique branchée).
    int   environmentResolution_ = 256;
    int   irradianceResolution_  = 32;
    int   prefilterMips_         = 6;
    int   stepsPerFrame_         = 2;
    float updateInterval_        = 2.0f;
    float sunAngleThresholdDeg_  = 1.0f;
    bool  freeze_                = false;
};

}  // namespace
REGISTER_SYSTEM_ORDERED(DynamicIBLSystem, PassStage::AtmosphereIBL, 10);
}  // namespace env::render
