// =============================================================================
//  src/render_engine/DeferredLightingSystem.cpp                     [DEV 2]
//  Stage : LightingWater (ordre 0, avant WaterOceanRiverSystem)
//
//  Rôle : résolution plein écran de l'éclairage PBR différé (ADR-004) : soleil,
//  lune, IBL et ombre des nuages, écrite dans scene.color. S'exécute après le
//  stage AtmosphereIBL afin de disposer de l'IBL et des LUTs d'atmosphère.
//
//  Les réglages partagés avec l'eau et le tonemap (intensités, ombres, brouillard, exposition) et l'état
//  de la frame (entrées disponibles) vivent dans LightingShared.hpp ; ce système en est le propriétaire.
//  Sortie HDR linéaire tant que la passe de tonemap tourne ; sinon repli LDR (exposition provisoire).
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
#include <cstdint>
#include <string_view>

#include "contracts/ISystemModule.hpp"
#include "contracts/WorldContracts.hpp"
#include "contracts/GpuBindings.hpp"
#include "core/SystemRegistry.hpp"
#include "core/RenderBlackboard.hpp"
#include "core/gpu/ShaderManager.hpp"
#include "core/gpu/GpuBuffer.hpp"
#include "render_engine/LightingShared.hpp"

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
        LightingShared&               sh = SharedLighting();
        const core::RenderBlackboard& bb = ctx.blackboard;
        sh.gbufferAvailable = bb.IsPublished(env::bb::GBufferDepth) && bb.IsPublished(env::bb::GBufferAlbedo) &&
                              bb.IsPublished(env::bb::GBufferNormal) && bb.IsPublished(env::bb::GBufferMaterial);
        sh.atmoAvailable = bb.IsPublished(env::bb::AtmoSkyViewLUT) && bb.IsPublished(env::bb::AtmoTransmittanceLUT);
        sh.iblAvailable  = bb.IsPublished(env::bb::IblIrradiance) && bb.IsPublished(env::bb::IblPrefiltered) &&
                           bb.IsPublished(env::bb::IblBrdfLUT);
        sh.cloudShadowAvailable = bb.IsPublished(env::bb::CloudShadowMap);
        sh.voxelAvailable       = bb.IsPublished(env::bb::VoxelGrid);  // VoxelIlluminationSystem (ADR-008)
        sh.dataView             = (debugView_ >= 1 && debugView_ <= 6) || debugView_ == 9 || debugView_ == 10;

        // Sortie HDR si la passe de tonemap (VolumetricGodRaysSystem -> scene.display) a tourné à la frame
        // précédente ; sinon (système désactivé) repli : exposition provisoire + ACES directement ici.
        const std::uint64_t frame = ctx.time.frameIndex;
        writeHdr_ = sh.tonemapFrame != ~std::uint64_t{0} && sh.tonemapFrame + 1 >= frame;

        // Ombres du relief : la carte world.focus.heightNormal ne dit pas QUEL chunk elle couvre.
        // On retrouve l'entité chunk qui porte la même texture pour connaître son origine exacte
        // (fonctionne pour le mock comme pour les vrais chunks de Dev 1).
        sh.focusAvailable = false;
        if (bb.IsPublished(env::bb::WorldFocusHeightNormal)) {
            const GLuint focusId = bb.GetTexture(env::bb::WorldFocusHeightNormal).id;
            auto view = ctx.registry.view<const ChunkCoordComponent, const ChunkStateComponent, const ChunkGpuResources>();
            for (const entt::entity e : view) {
                const auto& [coord, state, res] = view.get(e);
                if (state.state == ChunkState::Evicting || res.heightNormalTex != focusId) continue;
                sh.focusChunkOrigin = ChunkOriginRelative(coord.coord, ctx.renderOriginChunk);
                sh.focusAvailable   = true;
                break;
            }
        }
    }

    void Render(EngineContext& ctx) override {
        if (!program_ || !program_->IsValid()) return;
        LightingShared& sh = SharedLighting();
        // Ni G-Buffer ni atmosphère : rien d'utile à dessiner, on laisse l'aperçu mock intact.
        if (!sh.gbufferAvailable && !sh.atmoAvailable) return;

        const core::RenderBlackboard& bb = ctx.blackboard;
        namespace tex = env::bind::tex;
        bb.BindTexture(env::bb::GBufferAlbedo,   tex::GBufferAlbedo);
        bb.BindTexture(env::bb::GBufferNormal,   tex::GBufferNormal);
        bb.BindTexture(env::bb::GBufferMaterial, tex::GBufferMaterial);
        BindSurfaceLightingResources(bb);

        const gpu::ShaderProgram& p = *program_;
        SetSurfaceLightingUniforms(p);
        SetExposureUniforms(p);
        p.Set("uApplyFallbackExposure", !writeHdr_);
        p.Set("uDebugView",             debugView_);
        p.Set("uStarIntensity",         starIntensity_);
        if (writeHdr_) sh.hdrFrame = ctx.time.frameIndex;

        glDepthMask(GL_FALSE);
        if (sh.gbufferAvailable) {
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
        LightingShared& sh = SharedLighting();
        ImGui::SeparatorText("Entrees");
        StatusLine("G-Buffer", sh.gbufferAvailable, "mode ciel seul (apercu mock conserve)");
        StatusLine("Atmosphere (LUTs)", sh.atmoAvailable, "ciel noir");
        StatusLine("IBL", sh.iblAvailable, "repli sur la sky-view");
        StatusLine("Ombre des nuages", sh.cloudShadowAvailable, "pas d'ombre");
        StatusLine("Chunk focus (ombres du relief)", sh.focusAvailable, "pas d'ombre du relief");
        StatusLine("Grille VXGI (voxel.grid)", sh.voxelAvailable, "ni ombre lointaine ni GI");
        StatusLine("Tonemap (scene.display)", writeHdr_, "exposition + ACES provisoires ici, eau desactivee");

        ImGui::SeparatorText("Eclairage");
        ImGui::SliderFloat("Intensite soleil", &sh.sunScale, 0.0f, 4.0f);
        ImGui::SliderFloat("Intensite lune", &sh.moonScale, 0.0f, 4.0f);
        ImGui::SliderFloat("Intensite IBL", &sh.iblScale, 0.0f, 4.0f);
        ImGui::Checkbox("Occlusion speculaire", &sh.specularOcclusion);
        static const char* kViews[] = {"Lit", "Albedo", "Normal", "Roughness", "Metal", "AO", "Depth", "IBL only",
                                       "Sun only", "Shadows", "VXAO", "VXGI (indirect)"};
        ImGui::Combo("Vue de debug", &debugView_, kViews, IM_ARRAYSIZE(kViews));

        ImGui::SeparatorText("VXGI (ADR-008)");
        ImGui::Checkbox("Ombres lointaines (cone)", &sh.voxelShadows);
        ImGui::SameLine();
        ImGui::Checkbox("VXAO + GI", &sh.voxelGI);
        ImGui::SliderFloat("Intensite GI", &sh.giIntensity, 0.0f, 4.0f);
        ImGui::SetItemTooltip("Multiplicateur du rebond de lumiere des voxels (1 = physique).");
        ImGui::SliderFloat("Penombre lointaine", &sh.voxelShadowSoftnessDeg, 0.5f, 10.0f, "%.1f deg");
        ImGui::SetItemTooltip("Demi-angle du cone d'ombre : la penombre s'elargit avec la distance a l'occultant.");

        ImGui::SeparatorText("Ombres portees");
        ImGui::Checkbox("Relief (heightfield)", &sh.terrainShadows);
        ImGui::SameLine();
        ImGui::Checkbox("Contact (ecran)", &sh.contactShadows);
        ImGui::SliderFloat("Penombre du relief", &sh.shadowPenumbraDeg, 0.1f, 5.0f, "%.2f deg");
        ImGui::SetItemTooltip("Demi-angle de la penombre. Soleil reel : 0,27 deg (ombres nettes).");

        ImGui::SeparatorText("Nuit");
        ImGui::SliderFloat("Etoiles", &starIntensity_, 0.0f, 4.0f);

        ImGui::SeparatorText("Brouillard de hauteur");
        ImGui::TextDisabled("Densite au sol : panneau Celestial (Brouillard)");
        ImGui::SliderFloat("Decroissance verticale", &sh.fogHeightFalloff, 0.0005f, 0.05f, "%.4f /m",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::Checkbox("Ombre volumetrique", &sh.fogShadows);
        ImGui::SetItemTooltip("L'air a l'ombre du relief ne diffuse plus le soleil (grille VXGI) : "
                              "pas de voile aveuglant devant les montagnes a contre-jour.");

        ImGui::SeparatorText("Ombre des nuages (convention provisoire)");
        ImGui::SliderFloat("Altitude de la couche", &sh.cloudLayerAltitude, 500.0f, 8000.0f, "%.0f m");
        ImGui::SliderFloat("Emprise de la carte", &sh.cloudShadowExtent, 1000.0f, 50000.0f, "%.0f m",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::TextDisabled("Exposition : panneau VolumetricGodRays");
    }

private:
    gpu::ShaderProgram* program_ = nullptr;
    gpu::VertexArray    emptyVao_;
    bool                writeHdr_ = false;  ///< scene.color en HDR (tonemap actif) ou repli LDR

    // Paramètres propres à la résolution différée (OnGui) ; les autres sont dans LightingShared.
    int   debugView_     = 0;
    float starIntensity_ = 1.0f;
};

}  // namespace
REGISTER_SYSTEM_ORDERED(DeferredLightingSystem, PassStage::LightingWater, 0);
}  // namespace env::render
