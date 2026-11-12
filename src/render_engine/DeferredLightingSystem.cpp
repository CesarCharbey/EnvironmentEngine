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
#include <glm/glm.hpp>
#include <entt/entt.hpp>

#include <cmath>
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
        voxelAvailable_        = bb.IsPublished(env::bb::VoxelGrid);  // VoxelIlluminationSystem (ADR-008)
        // Valeurs physiques (ciel ~1e4 cd/m²) : sans passe de tonemap, scene.color saturerait en blanc.
        applyFallbackExposure_ = !bb.IsPublished(env::bb::SceneDisplay);

        // Ombres du relief : la carte world.focus.heightNormal ne dit pas QUEL chunk elle couvre.
        // On retrouve l'entité chunk qui porte la même texture pour connaître son origine exacte
        // (fonctionne pour le mock comme pour les vrais chunks de Dev 1).
        focusAvailable_ = false;
        if (bb.IsPublished(env::bb::WorldFocusHeightNormal)) {
            const GLuint focusId = bb.GetTexture(env::bb::WorldFocusHeightNormal).id;
            auto view = ctx.registry.view<const ChunkCoordComponent, const ChunkStateComponent, const ChunkGpuResources>();
            for (const entt::entity e : view) {
                const auto& [coord, state, res] = view.get(e);
                if (state.state == ChunkState::Evicting || res.heightNormalTex != focusId) continue;
                focusChunkOrigin_ = ChunkOriginRelative(coord.coord, ctx.renderOriginChunk);
                focusAvailable_   = true;
                break;
            }
        }
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
        // Lecture seule de la carte de hauteur du chunk focus (slot Dev 1 existant) : ombres du relief.
        bb.BindTexture(env::bb::WorldFocusHeightNormal, tex::HeightNormal);
        // VXGI : grille publiée ce frame ; son placement est dans UBO_VoxelParams (lié par le producteur).
        bb.BindTexture(env::bb::VoxelGrid, tex::VoxelGrid);

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
        p.Set("uVoxelAvailable",        voxelAvailable_);
        p.Set("uVoxelShadows",          voxelShadows_);
        p.Set("uVoxelGI",               voxelGI_);
        p.Set("uGiIntensity",           giIntensity_);
        p.Set("uVoxelShadowSoftness",   std::tan(glm::radians(voxelShadowSoftnessDeg_)));
        p.Set("uExposureCompensation",  exposureCompensationEv_);
        p.Set("uNightAdaptation",       nightAdaptation_);
        p.Set("uMoonlightShift",        moonlightShift_);
        p.Set("uStarIntensity",         starIntensity_);
        p.Set("uTerrainShadows",        terrainShadows_);
        p.Set("uContactShadows",        contactShadows_);
        p.Set("uFocusAvailable",        focusAvailable_);
        p.Set("uFocusChunkOrigin",      focusChunkOrigin_);
        p.Set("uShadowPenumbra",        glm::radians(shadowPenumbraDeg_));
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
        StatusLine("Chunk focus (ombres du relief)", focusAvailable_, "pas d'ombre du relief");
        StatusLine("Grille VXGI (voxel.grid)", voxelAvailable_, "ni ombre lointaine ni GI");
        StatusLine("scene.display (tonemap)", !applyFallbackExposure_, "exposition + ACES provisoires");

        ImGui::SeparatorText("Eclairage");
        ImGui::SliderFloat("Intensite soleil", &sunScale_, 0.0f, 4.0f);
        ImGui::SliderFloat("Intensite lune", &moonScale_, 0.0f, 4.0f);
        ImGui::SliderFloat("Intensite IBL", &iblScale_, 0.0f, 4.0f);
        ImGui::Checkbox("Occlusion speculaire", &specularOcclusion_);
        static const char* kViews[] = {"Lit", "Albedo", "Normal", "Roughness", "Metal", "AO", "Depth", "IBL only",
                                       "Sun only", "Shadows", "VXAO", "VXGI (indirect)"};
        ImGui::Combo("Vue de debug", &debugView_, kViews, IM_ARRAYSIZE(kViews));

        ImGui::SeparatorText("VXGI (ADR-008)");
        ImGui::Checkbox("Ombres lointaines (cone)", &voxelShadows_);
        ImGui::SameLine();
        ImGui::Checkbox("VXAO + GI", &voxelGI_);
        ImGui::SliderFloat("Intensite GI", &giIntensity_, 0.0f, 4.0f);
        ImGui::SetItemTooltip("Multiplicateur du rebond de lumiere des voxels (1 = physique).");
        ImGui::SliderFloat("Penombre lointaine", &voxelShadowSoftnessDeg_, 0.5f, 10.0f, "%.1f deg");
        ImGui::SetItemTooltip("Demi-angle du cone d'ombre : la penombre s'elargit avec la distance a l'occultant.");

        ImGui::SeparatorText("Ombres portees");
        ImGui::Checkbox("Relief (heightfield)", &terrainShadows_);
        ImGui::SameLine();
        ImGui::Checkbox("Contact (ecran)", &contactShadows_);
        ImGui::SliderFloat("Penombre du relief", &shadowPenumbraDeg_, 0.1f, 5.0f, "%.2f deg");
        ImGui::SetItemTooltip("Demi-angle de la penombre. Soleil reel : 0,27 deg (ombres nettes).");

        ImGui::SeparatorText("Nuit");
        ImGui::SliderFloat("Etoiles", &starIntensity_, 0.0f, 4.0f);
        ImGui::Checkbox("Virage Moonlight", &moonlightShift_);
        ImGui::SetItemTooltip("Basse lumiere : leger virage bleu-argente (garde >= 75 %% de la saturation).");

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
        ImGui::SliderFloat("Adaptation a l'obscurite", &nightAdaptation_, 0.0f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("Part de l'obscurite compensee par l'exposition (sous EV 0).\n"
                              "0 = nuit noire, 1 = nuit ramenee au gris moyen (jour bleute).");
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
    bool voxelAvailable_        = false;
    bool applyFallbackExposure_ = true;
    bool      focusAvailable_   = false;
    glm::vec3 focusChunkOrigin_{0.0f};

    // Paramètres (OnGui)
    float sunScale_               = 1.0f;
    float moonScale_              = 1.0f;
    float iblScale_               = 1.0f;
    bool  specularOcclusion_      = true;
    int   debugView_              = 0;
    float exposureCompensationEv_ = 1.0f;
    float nightAdaptation_        = 0.6f;
    bool  moonlightShift_         = true;
    float starIntensity_          = 1.0f;
    bool  terrainShadows_         = true;
    bool  contactShadows_         = true;
    float shadowPenumbraDeg_      = 1.0f;
    bool  voxelShadows_           = true;
    bool  voxelGI_                = true;
    float giIntensity_            = 1.0f;
    float voxelShadowSoftnessDeg_ = 2.3f;   // tan ≈ 0,04
    float fogHeightFalloff_       = 0.005f;   // hauteur d'échelle 200 m
    float cloudLayerAltitude_     = 2000.0f;
    float cloudShadowExtent_      = 10000.0f;
};

}  // namespace
REGISTER_SYSTEM_ORDERED(DeferredLightingSystem, PassStage::LightingWater, 0);
}  // namespace env::render
