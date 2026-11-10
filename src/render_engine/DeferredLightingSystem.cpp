// =============================================================================
//  src/render_engine/DeferredLightingSystem.cpp                     [DEV 2]
//  Stage : LightingWater (ordre 0, avant WaterOceanRiverSystem)
//
//  Rôle : résolution plein écran de l'éclairage PBR différé (ADR-004) : soleil,
//  lune, IBL et ombre des nuages, écrite dans scene.color. S'exécute après le
//  stage AtmosphereIBL afin de disposer de l'IBL et des LUTs d'atmosphère.
//
//  Deux modes, choisis chaque frame selon le blackboard :
//    * G-Buffer publié  : résolution complète (ciel + géométrie), depth test coupé.
//    * G-Buffer absent  : "ciel seul". Depth test GL_EQUAL à 0 contre scene.depth :
//      seuls les pixels où rien n'a été dessiné (ciel, Reverse-Z) reçoivent l'atmosphère,
//      l'aperçu mock du core reste visible par-dessus.
//
//  Entrées : gbuffer.*, atmo.*, ibl.*, clouds.shadowMap, GlobalEnvironmentUBO,
//            UBO_AtmosphereParams (lié par CelestialAtmosphereSystem dans le stage AtmosphereIBL ;
//            le binding indexé persiste, et le shader vérifie sa cohérence).
//  Sortie  : scene.color (cible déjà liée par le core).
//  RÈGLE : ce fichier n'inclut jamais src/world_engine/ (vérifié au build).
// =============================================================================
#include <imgui.h>
#include <glad/glad.h>

#include <string_view>

#include "contracts/ISystemModule.hpp"
#include "contracts/WorldContracts.hpp"
#include "contracts/GpuBindings.hpp"
#include "core/SystemRegistry.hpp"
#include "core/RenderBlackboard.hpp"
#include "core/gpu/ShaderManager.hpp"
#include "core/gpu/GpuBuffer.hpp"

namespace env::render {
namespace {

void StatusLine(const char* label, bool ok, const char* whenMissing) {
    ImGui::TextColored(ok ? ImVec4(0.45f, 0.85f, 0.45f, 1.0f) : ImVec4(0.95f, 0.70f, 0.30f, 1.0f),
                       "%s %s", ok ? "[ok]" : "[--]", label);
    if (!ok) {
        ImGui::SameLine();
        ImGui::TextDisabled("%s", whenMissing);
    }
}

class DeferredLightingSystem final : public ISystemModule {
public:
    void Init(EngineContext& ctx) override {
        program_ = &ctx.shaders.LoadGraphics("render.deferred_lighting",
                                             "render/deferred_lighting.vert",
                                             "render/deferred_lighting.frag");
        emptyVao_.Create("deferred.emptyVao");
    }

    void Update(EngineContext& ctx) override {
        const core::RenderBlackboard& bb = ctx.blackboard;
        gbufferAvailable_ = bb.IsPublished(env::bb::GBufferDepth) && bb.IsPublished(env::bb::GBufferAlbedo) &&
                            bb.IsPublished(env::bb::GBufferNormal) && bb.IsPublished(env::bb::GBufferMaterial);
        atmoAvailable_ = bb.IsPublished(env::bb::AtmoSkyViewLUT) && bb.IsPublished(env::bb::AtmoTransmittanceLUT);
        iblAvailable_  = bb.IsPublished(env::bb::IblIrradiance) && bb.IsPublished(env::bb::IblPrefiltered) &&
                         bb.IsPublished(env::bb::IblBrdfLUT);
        cloudShadowAvailable_  = bb.IsPublished(env::bb::CloudShadowMap);
        // Valeurs physiques (ciel ~1e4 cd/m²) : sans passe de tonemap, scene.color saturerait en blanc.
        applyFallbackExposure_ = !bb.IsPublished(env::bb::SceneDisplay);
    }

    void Render(EngineContext& ctx) override {
        if (!program_ || !program_->IsValid()) return;
        // Ni G-Buffer ni atmosphère : rien d'utile à dessiner, on laisse l'aperçu mock intact.
        if (!gbufferAvailable_ && !atmoAvailable_) return;

        const core::RenderBlackboard& bb = ctx.blackboard;
        namespace tex = env::bind::tex;
        bb.BindTexture(env::bb::GBufferAlbedo,          tex::GBufferAlbedo);
        bb.BindTexture(env::bb::GBufferNormal,          tex::GBufferNormal);
        bb.BindTexture(env::bb::GBufferMaterial,        tex::GBufferMaterial);
        bb.BindTexture(env::bb::GBufferDepth,           tex::GBufferDepth);
        bb.BindTexture(env::bb::AtmoTransmittanceLUT,   tex::AtmoTransmittance);
        bb.BindTexture(env::bb::AtmoMultiScatteringLUT, tex::AtmoMultiScattering);  // réservé (aerial perspective)
        bb.BindTexture(env::bb::AtmoSkyViewLUT,         tex::AtmoSkyView);
        bb.BindTexture(env::bb::IblIrradiance,          tex::IblIrradiance);
        bb.BindTexture(env::bb::IblPrefiltered,         tex::IblPrefiltered);
        bb.BindTexture(env::bb::IblBrdfLUT,             tex::IblBrdfLut);
        bb.BindTexture(env::bb::CloudShadowMap,         tex::CloudShadow);

        const gpu::ShaderProgram& p = *program_;
        p.Set("uGBufferAvailable",      gbufferAvailable_);
        p.Set("uAtmoAvailable",         atmoAvailable_);
        p.Set("uIblAvailable",          iblAvailable_);
        p.Set("uCloudShadowAvailable",  cloudShadowAvailable_);
        p.Set("uApplyFallbackExposure", applyFallbackExposure_);
        p.Set("uSpecularOcclusion",     specularOcclusion_);
        p.Set("uSunScale",              sunScale_);
        p.Set("uMoonScale",             moonScale_);
        p.Set("uIblScale",              iblScale_);
        p.Set("uDebugView",             debugView_);
        p.Set("uExposureCompensation",  exposureCompensationEv_);
        p.Set("uFogHeightFalloff",      fogHeightFalloff_);
        p.Set("uCloudLayerAltitude",    cloudLayerAltitude_);
        p.Set("uCloudShadowExtent",     cloudShadowExtent_);

        glDepthMask(GL_FALSE);
        if (gbufferAvailable_) {
            glDisable(GL_DEPTH_TEST);
        } else {
            // Le triangle est émis à z = 0 (infini en Reverse-Z) : ne passe que là où scene.depth vaut encore 0.
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_EQUAL);
        }

        p.Use();
        emptyVao_.Bind();
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // Restauration (le core remet aussi l'état à zéro avant chaque système).
        glBindVertexArray(0);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_GREATER);
        glDepthMask(GL_TRUE);
    }

    void Shutdown(EngineContext&) override {
        emptyVao_.Destroy();
        program_ = nullptr;
    }

    void OnGui(EngineContext&) override {
        ImGui::SeparatorText("Entrees");
        StatusLine("G-Buffer", gbufferAvailable_, "mode ciel seul (apercu mock conserve)");
        StatusLine("Atmosphere (LUTs)", atmoAvailable_, "ciel noir");
        StatusLine("IBL", iblAvailable_, "repli sur la sky-view");
        StatusLine("Ombre des nuages", cloudShadowAvailable_, "pas d'ombre");
        StatusLine("scene.display (tonemap)", !applyFallbackExposure_, "exposition + ACES provisoires");

        ImGui::SeparatorText("Eclairage");
        ImGui::SliderFloat("Intensite soleil", &sunScale_, 0.0f, 4.0f);
        ImGui::SliderFloat("Intensite lune", &moonScale_, 0.0f, 4.0f);
        ImGui::SliderFloat("Intensite IBL", &iblScale_, 0.0f, 4.0f);
        ImGui::Checkbox("Occlusion speculaire", &specularOcclusion_);
        static const char* kViews[] = {"Lit", "Albedo", "Normal", "Roughness", "Metal", "AO", "Depth", "IBL only", "Sun only"};
        ImGui::Combo("Vue de debug", &debugView_, kViews, IM_ARRAYSIZE(kViews));

        ImGui::SeparatorText("Brouillard de hauteur");
        ImGui::TextDisabled("Densite au sol : panneau Celestial (Brouillard)");
        ImGui::SliderFloat("Decroissance verticale", &fogHeightFalloff_, 0.0005f, 0.05f, "%.4f /m",
                           ImGuiSliderFlags_Logarithmic);

        ImGui::SeparatorText("Ombre des nuages (convention provisoire)");
        ImGui::SliderFloat("Altitude de la couche", &cloudLayerAltitude_, 500.0f, 8000.0f, "%.0f m");
        ImGui::SliderFloat("Emprise de la carte", &cloudShadowExtent_, 1000.0f, 50000.0f, "%.0f m",
                           ImGuiSliderFlags_Logarithmic);

        ImGui::BeginDisabled(!applyFallbackExposure_);
        ImGui::SeparatorText("Exposition provisoire");
        ImGui::SliderFloat("Compensation", &exposureCompensationEv_, -4.0f, 4.0f, "%+.1f EV");
        ImGui::EndDisabled();
    }

private:
    gpu::ShaderProgram* program_ = nullptr;
    gpu::VertexArray    emptyVao_;

    // État du blackboard (Update)
    bool gbufferAvailable_      = false;
    bool atmoAvailable_         = false;
    bool iblAvailable_          = false;
    bool cloudShadowAvailable_  = false;
    bool applyFallbackExposure_ = true;

    // Paramètres (OnGui)
    float sunScale_               = 1.0f;
    float moonScale_              = 1.0f;
    float iblScale_               = 1.0f;
    bool  specularOcclusion_      = true;
    int   debugView_              = 0;
    float exposureCompensationEv_ = 1.0f;
    float fogHeightFalloff_       = 0.005f;   // hauteur d'échelle 200 m
    float cloudLayerAltitude_     = 2000.0f;
    float cloudShadowExtent_      = 10000.0f;
};

}  // namespace
REGISTER_SYSTEM_ORDERED(DeferredLightingSystem, PassStage::LightingWater, 0);
}  // namespace env::render
