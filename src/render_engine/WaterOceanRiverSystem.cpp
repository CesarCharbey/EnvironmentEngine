// =============================================================================
//  src/render_engine/WaterOceanRiverSystem.cpp                      [DEV 2]
//  Stage : LightingWater (ordre 10, après DeferredLightingSystem)
//
//  Rôle : rendu de l'eau en forward par-dessus scene.color éclairée : océan FFT,
//  rivières et marais décrits par les champs de distance signée des chunks
//  (ADR-005). Les niveaux d'eau sont écrits par BiomeHydrologySystem (Dev 1) et
//  ne sont que lus ici.
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

class WaterOceanRiverSystem final : public ISystemModule {
public:
    void Init(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : shaders à créer dans shaders/render/ ; slots du contrat UBO_WaterParams,
        //            SSBO_OceanSpectrum, IMG_WaterOut, TEX_OceanDisplacement, TEX_OceanNormal.
        // Sorties  : spectre initial de l'océan, textures de déplacement et de normale par cascade, copie
        //            de scene.color destinée à la réfraction, maillage de surface (grille projetée ou
        //            clipmap), programmes du spectre, de la FFT et de la surface.
        // Objectif : préparer les ressources de l'eau.
    }

    void Update(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : niveaux d'eau waterTime.x (océan) et waterTime.y (marais), vent, lus dans le
        //            GlobalEnvironmentUBO en lecture seule ; paramètres du panneau.
        // Sorties  : indicateur de régénération du spectre quand le vent ou ses paramètres changent.
        // Objectif : travail côté CPU uniquement.
    }

    void Render(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : gbuffer.depth, ibl.prefiltered, atmo.transmittanceLUT, world.hydro.graphNodes
        //            (blackboard) ; entités chunk dont readyMask contient BiomeHydro (biomeHydroTex :
        //            distance signée à la rivière en G, au marais en B ; heightNormalTex pour la hauteur du
        //            terrain) ; temps de la frame.
        // Sorties  : water.oceanDisplacement et water.oceanNormal publiées ; surfaces d'eau composées en
        //            forward dans scene.color (profondeur en lecture seule, mélange activé).
        // Objectif : océan FFT (spectre de Phillips ou JONSWAP piloté par le vent, évolution temporelle,
        //            FFT inverse par cascade, écume) au niveau de l'océan ; rivières dont la surface dérive du
        //            terrain et de la distance signée, écoulement suivant le gradient de cette distance et le
        //            macro-graphe ; marais au niveau du marais ; absorption et diffusion, réfraction à partir
        //            d'une copie préalable de scene.color, réflexions écran.
    }

    void Shutdown(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : ressources créées par Init.
        // Sorties  : clés water.* retirées, textures, buffers et maillages libérés.
        // Objectif : libération propre avant la destruction du contexte OpenGL.
    }

    void OnGui(EngineContext&) override {
        TodoBanner("ocean FFT, rivieres et marais par SDF (ADR-005), shading de l'eau");

        ImGui::SeparatorText("Ocean FFT");
        ImGui::Checkbox("Ocean", &oceanEnabled_);
        ImGui::SliderInt("Resolution FFT", &fftResolution_, 64, 512);
        ImGui::DragFloat3("Cascades (m)", cascadeSizes_, 1.0f, 1.0f, 4096.0f, "%.0f");
        ImGui::SliderFloat("Amplitude", &amplitude_, 0.0f, 10.0f);
        ImGui::SliderFloat("Fetch", &fetchKm_, 1.0f, 1000.0f, "%.0f km", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Choppiness", &choppiness_, 0.0f, 3.0f);
        ImGui::SliderFloat("Seuil d'ecume", &foamThreshold_, 0.0f, 1.0f);
        ImGui::SliderFloat("Decroissance d'ecume", &foamDecay_, 0.0f, 1.0f);

        ImGui::SeparatorText("Rivieres / marais");
        ImGui::Checkbox("Rivieres", &riversEnabled_);
        ImGui::SameLine();
        ImGui::Checkbox("Marais", &swampsEnabled_);
        ImGui::SliderFloat("Vitesse d'ecoulement", &flowSpeed_, 0.0f, 5.0f, "%.2f m/s");
        ImGui::SliderFloat("Force des normales", &normalStrength_, 0.0f, 2.0f);
        ImGui::SliderFloat("Douceur des berges", &bankSoftness_, 0.1f, 10.0f, "%.1f m");
        ImGui::SliderFloat("Turbidite du marais", &murkiness_, 0.0f, 1.0f);

        ImGui::SeparatorText("Shading");
        ImGui::ColorEdit3("Absorption", absorption_);
        ImGui::ColorEdit3("Diffusion", scattering_);
        ImGui::SliderFloat("Refraction", &refractionStrength_, 0.0f, 1.0f);
        ImGui::Checkbox("Reflexions ecran (SSR)", &ssr_);
    }

private:
    // Variables liées aux sliders d'OnGui (aucune logique branchée).
    bool  oceanEnabled_       = true;
    int   fftResolution_      = 256;
    float cascadeSizes_[3]    = {250.0f, 37.0f, 7.0f};
    float amplitude_          = 1.0f;
    float fetchKm_            = 100.0f;
    float choppiness_         = 1.2f;
    float foamThreshold_      = 0.6f;
    float foamDecay_          = 0.9f;
    bool  riversEnabled_      = true;
    bool  swampsEnabled_      = true;
    float flowSpeed_          = 1.2f;
    float normalStrength_     = 0.8f;
    float bankSoftness_       = 1.5f;
    float murkiness_          = 0.7f;
    float absorption_[3]      = {0.45f, 0.09f, 0.06f};
    float scattering_[3]      = {0.02f, 0.10f, 0.12f};
    float refractionStrength_ = 0.3f;
    bool  ssr_                = true;
};

}  // namespace
REGISTER_SYSTEM_ORDERED(WaterOceanRiverSystem, PassStage::LightingWater, 10);
}  // namespace env::render
