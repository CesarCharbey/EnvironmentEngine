// =============================================================================
//  src/render_engine/VolumetricGodRaysSystem.cpp                    [DEV 2]
//  Stage : VolumetricsPost
//
//  Rôle : nuages volumétriques, carte d'ombre des nuages, god rays, composition
//  dans scene.color, puis exposition et tonemapping vers scene.display. Dès que
//  scene.display est publiée, le Viewport du core l'affiche à la place de
//  scene.color.
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

class VolumetricGodRaysSystem final : public ISystemModule {
public:
    void Init(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : shaders à créer dans shaders/render/ ; slots du contrat UBO_VolumetricParams,
        //            UBO_PostParams, IMG_VolumetricOut, SSBO_ExposureHistogram.
        // Sorties  : bruits 3D de forme (RGBA8 128^3) et de détail (RGBA8 32^3) publiés sous
        //            clouds.shapeNoise et clouds.detailNoise ; carte météo publiée sous clouds.weatherMap ;
        //            cible des nuages et son historique temporel ; carte d'ombre R16F ; cible des god rays en
        //            demi-résolution ; cible d'affichage tonemappée ; programmes associés.
        // Objectif : préparer les ressources volumétriques et de post-traitement.
    }

    void Update(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : taille du viewport, échelle de résolution du panneau, couverture et densité nuageuses
        //            (weatherExtra), vent (windDirSpeed) lus dans le GlobalEnvironmentUBO.
        // Sorties  : cibles réallouées quand leur taille change ; décalage de vent accumulé en double
        //            précision (monde infini).
        // Objectif : travail côté CPU uniquement.
    }

    void Render(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : scene.depth ou gbuffer.depth, atmo.transmittanceLUT, clouds.shapeNoise,
        //            clouds.detailNoise, clouds.weatherMap (blackboard) ; soleil, lune, vent et matrice
        //            de la frame précédente (GlobalEnvironmentUBO) ; indicateur de recalage de l'origine
        //            de rendu fourni par EngineContext.
        // Sorties  : clouds.shadowMap et volumetrics.godRays publiées ; nuages et god rays composés dans
        //            scene.color ; image finale tonemappée publiée sous scene.display.
        // Objectif : 1) raymarching des nuages dans la couche [bas, haut] (pas primaires et pas de lumière,
        //            Henyey-Greenstein à double lobe, effet powder, silver lining, jitter et reprojection
        //            temporelle réinitialisée lors d'un recalage d'origine) ; 2) ombre des nuages vue du
        //            soleil ; 3) god rays radiaux à l'écran ou volumétriques ; 4) composition ;
        //            5) exposition (manuelle ou par histogramme) puis tonemapping.
    }

    void Shutdown(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : ressources créées par Init.
        // Sorties  : clés clouds.*, volumetrics.godRays et scene.display retirées ; textures et cibles
        //            libérées.
        // Objectif : libération propre avant la destruction du contexte OpenGL.
    }

    void OnGui(EngineContext&) override {
        TodoBanner("raymarching des nuages, ombre des nuages, god rays, exposition / tonemap");

        ImGui::SeparatorText("Nuages volumetriques");
        ImGui::Checkbox("Nuages", &cloudsEnabled_);
        ImGui::DragFloatRange2("Couche (m)", &cloudBottom_, &cloudTop_, 10.0f, 200.0f, 12000.0f, "%.0f m");
        static const char* kScales[] = {"Pleine", "Demi", "Quart"};
        ImGui::Combo("Resolution", &renderScaleIndex_, kScales, IM_ARRAYSIZE(kScales));
        ImGui::SliderInt("Pas primaires", &primarySteps_, 8, 256);
        ImGui::SliderInt("Pas lumiere", &lightSteps_, 1, 16);
        ImGui::Checkbox("Reprojection temporelle", &temporal_);
        ImGui::SliderFloat("Echelle bruit de forme", &shapeScale_, 1000.0f, 50000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Echelle bruit de detail", &detailScale_, 50.0f, 5000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Force du detail", &detailStrength_, 0.0f, 1.0f);
        ImGui::SliderFloat("Extinction", &extinction_, 0.001f, 0.2f, "%.4f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("HG avant (g)", &hgForward_, 0.0f, 0.99f);
        ImGui::SliderFloat("HG arriere (g)", &hgBackward_, -0.99f, 0.0f);
        ImGui::SliderFloat("Silver lining", &silverIntensity_, 0.0f, 5.0f);
        ImGui::SliderFloat("Powder", &powder_, 0.0f, 4.0f);
        ImGui::SliderFloat("Influence du vent", &windInfluence_, 0.0f, 4.0f);

        ImGui::SeparatorText("God rays");
        ImGui::Checkbox("God rays", &godRaysEnabled_);
        static const char* kModes[] = {"Radial ecran", "Volumetrique (ombre des nuages)"};
        ImGui::Combo("Methode", &godRayMode_, kModes, IM_ARRAYSIZE(kModes));
        ImGui::SliderInt("Echantillons", &godRaySamples_, 8, 256);
        ImGui::SliderFloat("Densite", &godRayDensity_, 0.0f, 2.0f);
        ImGui::SliderFloat("Decroissance", &godRayDecay_, 0.8f, 1.0f, "%.3f");
        ImGui::SliderFloat("Poids", &godRayWeight_, 0.0f, 2.0f);
        ImGui::SliderFloat("Exposition god rays", &godRayExposure_, 0.0f, 2.0f);

        ImGui::SeparatorText("Post (scene.display)");
        ImGui::SliderFloat("Exposition (EV)", &exposureEv_, -10.0f, 10.0f, "%.2f EV");
        ImGui::Checkbox("Exposition automatique", &autoExposure_);
        static const char* kTonemappers[] = {"ACES", "AgX", "Reinhard", "Khronos PBR Neutral"};
        ImGui::Combo("Tonemapper", &tonemapper_, kTonemappers, IM_ARRAYSIZE(kTonemappers));
        ImGui::SliderFloat("Point blanc", &whitePoint_, 1.0f, 32.0f);
    }

private:
    // Variables liées aux sliders d'OnGui (aucune logique branchée).
    bool  cloudsEnabled_    = true;
    float cloudBottom_      = 1500.0f;
    float cloudTop_         = 4500.0f;
    int   renderScaleIndex_ = 1;
    int   primarySteps_     = 64;
    int   lightSteps_       = 6;
    bool  temporal_         = true;
    float shapeScale_       = 12000.0f;
    float detailScale_      = 900.0f;
    float detailStrength_   = 0.35f;
    float extinction_       = 0.04f;
    float hgForward_        = 0.6f;
    float hgBackward_       = -0.3f;
    float silverIntensity_  = 1.5f;
    float powder_           = 2.0f;
    float windInfluence_    = 1.0f;
    bool  godRaysEnabled_   = true;
    int   godRayMode_       = 0;
    int   godRaySamples_    = 64;
    float godRayDensity_    = 0.9f;
    float godRayDecay_      = 0.96f;
    float godRayWeight_     = 0.4f;
    float godRayExposure_   = 0.3f;
    float exposureEv_       = 0.0f;
    bool  autoExposure_     = true;
    int   tonemapper_       = 1;
    float whitePoint_       = 8.0f;
};

}  // namespace
REGISTER_SYSTEM(VolumetricGodRaysSystem, PassStage::VolumetricsPost);
}  // namespace env::render
