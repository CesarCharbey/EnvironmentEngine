// =============================================================================
//  src/render_engine/VolumetricGodRaysSystem.cpp                    [DEV 2]
//  Stage : VolumetricsPost
//
//  Rôle : nuages volumétriques, carte d'ombre des nuages, god rays, composition
//  dans scene.color, puis exposition et tonemapping vers scene.display. Dès que
//  scene.display est publiée, le Viewport du core l'affiche à la place de
//  scene.color.
//
//  Implémenté : exposition + tonemapping de scene.color (HDR) vers scene.display (tonemap.vert/.frag),
//  après l'eau. Exposition automatique provisoire (physique, LightingShared.hpp) ou manuelle.
//  À faire : nuages, ombre des nuages, god rays (jalon 3).
//  RÈGLE : ce fichier n'inclut jamais src/world_engine/ (vérifié au build).
// =============================================================================
#include <imgui.h>
#include <glad/glad.h>

#include <algorithm>

#include "contracts/ISystemModule.hpp"
#include "contracts/WorldContracts.hpp"
#include "contracts/GpuBindings.hpp"
#include "core/SystemRegistry.hpp"
#include "core/RenderBlackboard.hpp"
#include "core/gpu/ShaderManager.hpp"
#include "core/gpu/Framebuffer.hpp"
#include "core/gpu/GpuBuffer.hpp"
#include "render_engine/LightingShared.hpp"

namespace env::render {
namespace {

void TodoBanner(const char* scope) {
    ImGui::TextColored(ImVec4(0.35f, 0.80f, 1.0f, 1.0f), "TODO Dev 2");
    ImGui::SameLine();
    ImGui::TextDisabled("%s", scope);
}

class VolumetricGodRaysSystem final : public ISystemModule {
public:
    void Init(EngineContext& ctx) override {
        tonemapProg_ = &ctx.shaders.LoadGraphics("render.tonemap", "render/tonemap.vert", "render/tonemap.frag");
        emptyVao_.Create("post.emptyVao");
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : shaders à créer dans shaders/render/ ; slots du contrat UBO_VolumetricParams,
        //            UBO_PostParams, IMG_VolumetricOut, SSBO_ExposureHistogram.
        // Sorties  : bruits 3D de forme (RGBA8 128^3) et de détail (RGBA8 32^3) publiés sous
        //            clouds.shapeNoise et clouds.detailNoise ; carte météo publiée sous clouds.weatherMap ;
        //            cible des nuages et son historique temporel ; carte d'ombre R16F ; cible des god rays en
        //            demi-résolution ; cible d'affichage tonemappée ; programmes associés.
        // Objectif : préparer les ressources volumétriques et de post-traitement.
    }

    void Update(EngineContext& ctx) override {
        // Cible d'affichage à la taille de scene.color (republiée après chaque réallocation).
        const GLsizei w = std::max(ctx.viewportSize.x, 1), h = std::max(ctx.viewportSize.y, 1);
        if (!display_.IsValid()) {
            gpu::Framebuffer::Desc desc;
            desc.width     = w;
            desc.height    = h;
            desc.colors    = {{GL_RGBA8, {}, "scene.display"}};
            desc.debugName = "post.display";
            display_.Create(desc);
            displayDirty_ = true;
        } else if (display_.Resize(w, h)) {
            displayDirty_ = true;
        }
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : taille du viewport, échelle de résolution du panneau, couverture et densité nuageuses
        //            (weatherExtra), vent (windDirSpeed) lus dans le GlobalEnvironmentUBO.
        // Sorties  : cibles réallouées quand leur taille change ; décalage de vent accumulé en double
        //            précision (monde infini).
        // Objectif : travail côté CPU uniquement.
    }

    void Render(EngineContext& ctx) override {
        LightingShared& sh = SharedLighting();
        const std::uint64_t frame = ctx.time.frameIndex;
        sh.tonemapFrame = frame;  // l'éclairage écrira du HDR dès la frame suivante

        // scene.color n'est HDR que si l'éclairage l'a écrite ainsi cette frame (sinon : repli déjà encodé
        // par l'éclairage, ou éclairage désactivé) -> on n'affiche pas scene.display.
        if (!SceneColorIsHdr(frame) || !tonemapProg_->IsValid()) {
            ctx.blackboard.UnpublishIfProducer(env::bb::SceneDisplay, kTonemapProducer);
            displayDirty_ = true;
            return;
        }

        display_.Bind();
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        ctx.blackboard.BindTexture(env::bb::SceneColor, env::bind::tex::SceneColor);
        ctx.blackboard.BindTexture(env::bb::AtmoTransmittanceLUT, env::bind::tex::AtmoTransmittance);
        ctx.blackboard.BindTexture(env::bb::AtmoSkyViewLUT, env::bind::tex::AtmoSkyView);
        const gpu::ShaderProgram& p = *tonemapProg_;
        p.Set("uAtmoAvailable", sh.atmoAvailable);
        p.Set("uSunScale",      sh.sunScale);
        p.Set("uMoonScale",     sh.moonScale);
        SetExposureUniforms(p);
        p.Set("uDataView",      sh.dataView);
        p.Set("uAutoExposure",  autoExposure_);
        p.Set("uManualEv100",   exposureEv_);
        p.Set("uTonemapper",    tonemapper_);
        p.Set("uWhitePoint",    whitePoint_);
        p.Use();
        emptyVao_.Bind();
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);

        if (displayDirty_) {
            ctx.blackboard.PublishTexture(env::bb::SceneDisplay, display_.Color(0), kTonemapProducer);
            displayDirty_ = false;
        }
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

    void Shutdown(EngineContext& ctx) override {
        ctx.blackboard.UnpublishIfProducer(env::bb::SceneDisplay, kTonemapProducer);
        display_.Destroy();
        emptyVao_.Destroy();
        tonemapProg_ = nullptr;
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : ressources créées par Init.
        // Sorties  : clés clouds.*, volumetrics.godRays et scene.display retirées ; textures et cibles
        //            libérées.
        // Objectif : libération propre avant la destruction du contexte OpenGL.
    }

    void OnGui(EngineContext&) override {
        TodoBanner("raymarching des nuages, ombre des nuages, god rays");

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
        LightingShared& sh = SharedLighting();
        ImGui::Checkbox("Exposition automatique", &autoExposure_);
        ImGui::BeginDisabled(!autoExposure_);
        ImGui::SliderFloat("Compensation", &sh.exposureCompensationEv, -4.0f, 4.0f, "%+.1f EV");
        ImGui::SliderFloat("Adaptation a l'obscurite", &sh.nightAdaptation, 0.0f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("Part de l'obscurite compensee par l'exposition (sous EV 0).\n"
                              "0 = nuit noire, 1 = nuit ramenee au gris moyen (jour bleute).");
        ImGui::EndDisabled();
        ImGui::BeginDisabled(autoExposure_);
        ImGui::SliderFloat("Exposition (EV100)", &exposureEv_, -6.0f, 18.0f, "%.2f EV");
        ImGui::EndDisabled();
        ImGui::Checkbox("Virage Moonlight", &sh.moonlightShift);
        ImGui::SetItemTooltip("Basse lumiere : leger virage bleu-argente (garde >= 75 %% de la saturation).");
        static const char* kTonemappers[] = {"ACES", "AgX", "Reinhard", "Khronos PBR Neutral"};
        ImGui::Combo("Tonemapper", &tonemapper_, kTonemappers, IM_ARRAYSIZE(kTonemappers));
        ImGui::BeginDisabled(tonemapper_ != 2);
        ImGui::SliderFloat("Point blanc", &whitePoint_, 1.0f, 32.0f);
        ImGui::EndDisabled();
    }

private:
    gpu::ShaderProgram* tonemapProg_  = nullptr;
    gpu::VertexArray    emptyVao_;
    gpu::Framebuffer    display_;
    bool                displayDirty_ = true;  ///< scene.display à (re)publier

    // Variables liées aux sliders d'OnGui (nuages et god rays : aucune logique branchée).
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
    float exposureEv_       = 13.0f;  // EV100 manuel (journée ensoleillée ≈ 15)
    bool  autoExposure_     = true;
    int   tonemapper_       = 0;      // ACES : même rendu qu'avant la passe de tonemap
    float whitePoint_       = 8.0f;
};

}  // namespace
REGISTER_SYSTEM(VolumetricGodRaysSystem, PassStage::VolumetricsPost);
}  // namespace env::render
