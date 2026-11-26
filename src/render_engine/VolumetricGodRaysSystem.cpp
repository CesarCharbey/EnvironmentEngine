// =============================================================================
//  src/render_engine/VolumetricGodRaysSystem.cpp                    [DEV 2]
//  Stage : VolumetricsPost
//
//  Rôle : nuages volumétriques, carte d'ombre des nuages, god rays, composition
//  dans scene.color, puis exposition et tonemapping vers scene.display. Dès que
//  scene.display est publiée, le Viewport du core l'affiche à la place de
//  scene.color.
//
//  Implémenté :
//    * Nuages volumétriques (jalon 3, 1re partie) :
//      - bruits 3D de forme (Perlin-Worley 128³) et de détail (Worley 32³), générés une fois au premier
//        Render() (cloud_noise.comp), publiés sous clouds.shapeNoise / clouds.detailNoise ;
//      - carte météo en fenêtre glissante (512², 40 km par défaut, centre calé sur un texel, fronts
//        advectés par le vent en coordonnées absolues) recalculée chaque frame (cloud_weather_map.comp),
//        publiée sous clouds.weatherMap ;
//      - raymarching en résolution réduite (volumetric_clouds.comp) vers IMG_VolumetricOut ;
//      - composition dans la passe de tonemap (scène × transmittance + luminance des nuages).
//      Vent accumulé en double (monde infini) ; les décalages des bruits sont réduits modulo leur
//      période côté CPU : aucune perte de précision loin de l'origine, pas de saut au recalage.
//    * Exposition + tonemapping de scene.color (HDR) vers scene.display (tonemap.vert/.frag), après l'eau.
//    * Ombre des nuages au sol (cloud_shadow.comp -> clouds.shadowMap, R16F 512², 20 km calés sur un texel),
//      même modèle de densité (cloud_density.glsl) ; lue par CloudShadow() (surface_lighting.glsl) à la
//      frame suivante, avec l'origine, l'astre et la base de couche stockés dans LightingShared.
//    * Reprojection temporelle (cloud_temporal.glsl, cloud_reproject.comp) : 1 pixel sur 2×2 (ou 4×4)
//      raymarché par frame (Bayer), mélangé à son historique ; les autres reprojetés (ping-pong de deux
//      cibles, prevViewProj_ + décalage d'origine de rendu) et bornés par les pixels frais voisins.
//      Historique lu sur le slot TEX_GodRays pendant les passes des nuages seulement (le tonemap y lie
//      ensuite la vraie texture des god rays).
//    * God rays volumétriques (volumetric_godrays.comp, demi-résolution, publiés sous volumetrics.godRays) :
//      lumière de l'astre dominant diffusée par le brouillard de hauteur, ombrée par les nuages et le relief
//      (VXGI) ; composés dans le tonemap (suréchantillonnage bilatéral). L'éclairage leur laisse le premier
//      tronçon du rayon (LightingShared::godRayDistance) : pas de double compte avec son brouillard.
//  RÈGLE : ce fichier n'inclut jamais src/world_engine/ (vérifié au build).
// =============================================================================
#include <imgui.h>
#include <glad/glad.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "contracts/ISystemModule.hpp"
#include "contracts/WorldContracts.hpp"
#include "contracts/GpuBindings.hpp"
#include "core/SystemRegistry.hpp"
#include "core/RenderBlackboard.hpp"
#include "core/gpu/ShaderManager.hpp"
#include "core/gpu/Framebuffer.hpp"
#include "core/gpu/GpuBuffer.hpp"
#include "core/gpu/Texture2D.hpp"
#include "core/gpu/Texture3D.hpp"
#include "render_engine/LightingShared.hpp"

namespace env::render {
namespace {

constexpr std::string_view kProducer      = "VolumetricGodRaysSystem";
constexpr GLsizei          kShapeSize      = 128;
constexpr GLsizei          kDetailSize     = 32;
constexpr GLsizei          kWeatherSize    = 512;
constexpr GLsizei          kShadowSize     = 1024;
constexpr int              kFrontPeriodCells = 128;  ///< période des fronts = 128 cellules de base

/// x mod period dans [0, period), en double (positions absolues d'un monde infini).
double WrapPositive(double x, double period) { return x - std::floor(x / period) * period; }

float WrappedDelta2D(glm::vec2 a, glm::vec2 b, float period) {
    glm::vec2 d = glm::abs(a - b);
    if (period > 0.0f) d = glm::min(d, glm::vec2(period) - d);
    return glm::length(d);
}

class VolumetricGodRaysSystem final : public ISystemModule {
public:
    void Init(EngineContext& ctx) override {
        tonemapProg_ = &ctx.shaders.LoadGraphics("render.tonemap", "render/tonemap.vert", "render/tonemap.frag");
        emptyVao_.Create("post.emptyVao");
        noiseProg_   = &ctx.shaders.LoadCompute("render.cloud_noise", "render/cloud_noise.comp");
        weatherProg_ = &ctx.shaders.LoadCompute("render.cloud_weather_map", "render/cloud_weather_map.comp");
        cloudsProg_  = &ctx.shaders.LoadCompute("render.volumetric_clouds", "render/volumetric_clouds.comp");
        shadowProg_  = &ctx.shaders.LoadCompute("render.cloud_shadow", "render/cloud_shadow.comp");
        reprojectProg_ = &ctx.shaders.LoadCompute("render.cloud_reproject", "render/cloud_reproject.comp");
        godRaysProg_   = &ctx.shaders.LoadCompute("render.volumetric_godrays", "render/volumetric_godrays.comp");
        cloudUbo_.Create(sizeof(CloudParamsUbo), GL_DYNAMIC_STORAGE_BIT, nullptr, "clouds.paramsUbo");
        surfaceLightingUbo_.Create(sizeof(SurfaceLightingUbo), GL_DYNAMIC_STORAGE_BIT, nullptr,
                                   "godrays.surfaceLightingUbo");
        // Les textures sont allouées au premier Render() (Init n'émet aucune commande GPU).
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

        // --- Vent : décalage accumulé en double (m) ---
        windDeltaXZ_ = glm::vec2(0.0f);
        const glm::vec2 wind  = glm::vec2(ctx.environment.windDirSpeed);
        const double    speed = static_cast<double>(ctx.environment.windDirSpeed.z) * windInfluence_;
        if (glm::length(wind) > 1e-4f) {
            const glm::vec2 d = glm::normalize(wind);
            const double    dx = d.x * speed * ctx.time.dt;
            const double    dz = d.y * speed * ctx.time.dt;
            windDirXZ_ = d;
            windDeltaXZ_   = glm::vec2(static_cast<float>(dx), static_cast<float>(dz));
            windOffset_[0] += dx;
            windOffset_[1] += dz;
        }

        // --- Fenêtre glissante de la carte météo, centrée sur la caméra et calée sur un texel ---
        const double originX = static_cast<double>(ctx.renderOriginChunk.x) * kChunkSizeMeters;
        const double originZ = static_cast<double>(ctx.renderOriginChunk.y) * kChunkSizeMeters;
        const double camX    = originX + ctx.environment.cameraPosition.x;
        const double camZ    = originZ + ctx.environment.cameraPosition.z;
        const double texel   = static_cast<double>(weatherMapExtent_) / kWeatherSize;
        const double centerX = std::floor(camX / texel + 0.5) * texel;
        const double centerZ = std::floor(camZ / texel + 0.5) * texel;
        weatherOriginRel_ = glm::vec2(static_cast<float>(centerX - originX), static_cast<float>(centerZ - originZ));
        // Fronts : coordonnées absolues advectées par le vent, modulo la période du bruit.
        frontPeriod_ = static_cast<double>(frontScale_) * kFrontPeriodCells;
        weatherCenterMod_ = glm::vec2(static_cast<float>(WrapPositive(centerX - windOffset_[0], frontPeriod_)),
                                      static_cast<float>(WrapPositive(centerZ - windOffset_[1], frontPeriod_)));
        // Bruits 3D : (origine de rendu - vent) modulo leur répétition ; le détail dérive un peu plus vite.
        const double ds = shapeScale_, dd = detailScale_;
        shapeOffset_  = glm::vec3(static_cast<float>(WrapPositive(originX - windOffset_[0], ds)), 0.0f,
                                  static_cast<float>(WrapPositive(originZ - windOffset_[1], ds)));
        detailOffset_ = glm::vec3(static_cast<float>(WrapPositive(originX - 1.3 * windOffset_[0], dd)),
                                  static_cast<float>(WrapPositive(-0.05 * ctx.time.time * speed, dd)),
                                  static_cast<float>(WrapPositive(originZ - 1.3 * windOffset_[1], dd)));

        // Cirrus 2.5D : décalage en double précision modulo la période (vent un peu plus rapide en haute altitude).
        const double dc = cirrusScale_;
        cirrusOffset_ = glm::vec2(static_cast<float>(WrapPositive(originX - 1.6 * windOffset_[0], dc)),
                                  static_cast<float>(WrapPositive(originZ - 1.6 * windOffset_[1], dc)));

        // --- Éclairs internes (orage : forte pluie + couverture, ou bouton manuel) ---
        if (hasPrevFrame_ && ctx.renderOriginShifted && lightningTimer_ > 0.0f) {
            const glm::vec2 shift = glm::vec2(lastOriginChunk_ - ctx.renderOriginChunk) * kChunkSizeMeters;
            lightningPosRel_.x += shift.x;
            lightningPosRel_.z += shift.y;
        }
        lightningIntensity_ = 0.0f;
        if (lightningEnabled_) {
            const float dt    = static_cast<float>(ctx.time.dt);
            const bool  storm = ctx.environment.weather.x > 0.45f && ctx.environment.weatherExtra.x > 0.55f;
            auto NextRand = [&]() {
                lightningRng_ = lightningRng_ * 1664525u + 1013904223u;
                return static_cast<float>(lightningRng_ >> 8u) * (1.0f / 16777216.0f);
            };
            if (lightningTimer_ <= 0.0f) {
                if (storm) lightningCooldown_ -= dt;
                if (triggerLightning_ || (storm && lightningCooldown_ <= 0.0f)) {
                    triggerLightning_ = false;
                    // Position autour de la caméra (2 à 10 km), au cœur de la couche nuageuse.
                    const float angle = NextRand() * 6.2831853f;
                    const float dist  = 2000.0f + NextRand() * 8000.0f;
                    const float alt   = cloudBottom_ + (0.35f + 0.35f * NextRand()) * (cloudTop_ - cloudBottom_);
                    lightningPosRel_  = glm::vec3(ctx.environment.cameraPosition.x + std::cos(angle) * dist, alt,
                                                  ctx.environment.cameraPosition.z + std::sin(angle) * dist);
                    lightningTimer_    = 0.38f;
                    lightningCooldown_ = 2.0f + NextRand() * 5.0f;
                }
            } else {
                lightningTimer_ = std::max(0.0f, lightningTimer_ - dt);
                // Double flash réaliste sur 0,38 s (stable avec la reprojection 2×2).
                const float u  = 1.0f - lightningTimer_ / 0.38f;
                const float p1 = std::exp(-std::pow((u - 0.12f) / 0.08f, 2.0f));
                const float p2 = 0.85f * std::exp(-std::pow((u - 0.42f) / 0.14f, 2.0f));
                lightningIntensity_ = p1 + p2;  // enveloppe normalisée (mise à l'échelle dans le shader)
            }
        }

        // --- Ombre des nuages : carte centrée sur la caméra, calée sur un texel (coordonnées absolues) ---
        {
            const double st = static_cast<double>(cloudShadowExtent_) / kShadowSize;
            const double sx = std::floor(camX / st + 0.5) * st, sz = std::floor(camZ / st + 0.5) * st;
            shadowOriginRel_ = glm::vec2(static_cast<float>(sx - originX), static_cast<float>(sz - originZ));
            const glm::vec3 sun  = glm::normalize(glm::vec3(ctx.environment.sunDirIlluminance));
            const glm::vec3 moon = glm::normalize(glm::vec3(ctx.environment.moonDirPhase));
            shadowLightUp_ = sun.y > -0.02f || moon.y > 0.0f;
            shadowLight_   = sun.y > -0.02f ? sun : moon;  // même choix d'astre que le raymarching
        }

        // --- Cible des nuages (résolution réduite), allouée / redimensionnée ici comme le G-Buffer ---
        const int div = 1 << std::clamp(renderScaleIndex_, 0, 2);
        const glm::ivec2 cs((w + div - 1) / div, (h + div - 1) / div);
        if (cs != cloudTargetSize_) {
            // Deux cibles en ping-pong : la courante (écrite, lue par le tonemap) et l'historique (frame
            // précédente). RGBA16F, linéaire, GL_CLAMP_TO_EDGE.
            gpu::Texture2D::Desc d;
            d.width          = cs.x;
            d.height         = cs.y;
            d.internalFormat = GL_RGBA16F;
            d.debugName      = "clouds.targetA";
            cloudTargets_[0].Create(d);
            d.debugName      = "clouds.targetB";
            cloudTargets_[1].Create(d);
            cloudTargetSize_ = cs;
            historyValid_    = false;  // redimensionnement / changement d'échelle
        }

        // --- God rays : cibles ping-pong en demi-résolution (republiées après réallocation) ---
        const glm::ivec2 gs((w + 1) / 2, (h + 1) / 2);
        if (gs != godRaysSize_) {
            gpu::Texture2D::Desc d;
            d.width          = gs.x;
            d.height         = gs.y;
            d.internalFormat = GL_RGBA16F;  // rgb luminance diffusée, a transmittance (ciel)
            d.debugName      = "volumetrics.godRaysA";
            godRaysTargets_[0].Create(d);
            d.debugName      = "volumetrics.godRaysB";
            godRaysTargets_[1].Create(d);
            godRaysSize_         = gs;
            godRaysHistoryValid_ = false;
            ctx.blackboard.PublishTexture(env::bb::VolumetricGodRays, godRaysTargets_[godRaysCurrent_], kProducer);
        }
        // Distance prise en charge par les god rays : l'éclairage n'y diffuse plus l'astre dominant dans son
        // brouillard (pas de double compte). 0 sans tonemap (repli LDR de l'éclairage) ou god rays coupés.
        {
            LightingShared& lsh = SharedLighting();
            const bool tonemapRuns = lsh.tonemapFrame != ~std::uint64_t{0} && lsh.tonemapFrame + 1 >= ctx.time.frameIndex;
            godRaysActive_ = godRaysEnabled_ && tonemapRuns && godRaysProg_ && godRaysProg_->IsValid();
            lsh.godRayDistance = godRaysActive_ ? godRayMaxDistance_ : 0.0f;
        }

        // --- Reprojection temporelle : matrices et origine de la frame précédente, invalidations ---
        const glm::vec3 sunNow   = glm::normalize(glm::vec3(ctx.environment.sunDirIlluminance));
        const glm::vec3 moonNow  = glm::normalize(glm::vec3(ctx.environment.moonDirPhase));
        const glm::vec3 lightNow = sunNow.y > -0.02f ? sunNow : moonNow;
        const glm::vec4 wxNow(ctx.environment.weatherExtra.x, ctx.environment.weatherExtra.y, ctx.environment.weather.x,
                              ctx.environment.weather.y);
        const int  block    = temporal_ ? (temporalPattern_ == 0 ? 2 : 4) : 1;
        const bool jump     = !hasPrevFrame_ || glm::dot(lightNow, prevLight_) < std::cos(glm::radians(1.0f)) ||
                              glm::any(glm::greaterThan(glm::abs(wxNow - prevWeather_), glm::vec4(0.08f))) ||
                              block != prevBlock_;
        if (jump) {
            historyValid_        = false;
            godRaysHistoryValid_ = false;
        }
        prevViewProj_     = hasPrevFrame_ ? lastViewProj_ : ctx.environment.viewProj;
        prevOriginChunk_  = hasPrevFrame_ ? lastOriginChunk_ : ctx.renderOriginChunk;
        lastViewProj_     = ctx.environment.viewProj;
        lastOriginChunk_  = ctx.renderOriginChunk;
        prevLight_        = lightNow;
        prevWeather_      = wxNow;
        prevBlock_        = block;
        hasPrevFrame_     = true;
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

        // --- Nuages : bruits (une fois), carte météo, raymarching ---
        const bool clouds = cloudsEnabled_ && RenderClouds(ctx);
        if (!clouds) {
            SharedLighting().cloudShadowValid = false;
            SharedLighting().clouds.valid     = false;  // la capture de l'IBL revient au ciel seul
        }

        // --- God rays volumétriques (demi-résolution) ---
        const bool godRays = godRaysActive_ && RenderGodRays(ctx);

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
        p.Set("uCloudsAvailable", clouds);
        p.Set("uCloudBottom",   cloudBottom_);
        p.Set("uGodRaysAvailable", godRays);
        ctx.blackboard.BindTexture(env::bb::SceneDepth, env::bind::tex::SceneDepth);
        // Le slot TEX_GodRays porte l'historique des nuages pendant leurs passes ; ici, les god rays.
        if (godRays) godRaysTargets_[godRaysCurrent_].BindUnit(env::bind::tex::GodRays);
        if (clouds)  cloudTargets_[current_].BindUnit(env::bind::tex::CloudWeatherMap);
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
        SharedLighting().cloudShadowValid = false;
        SharedLighting().clouds.valid     = false;
        for (std::string_view key : {env::bb::CloudShapeNoise, env::bb::CloudDetailNoise, env::bb::CloudWeatherMap,
                                     env::bb::CloudShadowMap, env::bb::VolumetricGodRays})
            ctx.blackboard.UnpublishIfProducer(key, kProducer);
        display_.Destroy();
        emptyVao_.Destroy();
        shapeNoise_.Destroy();
        detailNoise_.Destroy();
        weatherMap_.Destroy();
        cloudTargets_[0].Destroy();
        cloudTargets_[1].Destroy();
        godRaysTargets_[0].Destroy();
        godRaysTargets_[1].Destroy();
        SharedLighting().godRayDistance = 0.0f;
        shadowMap_.Destroy();
        cloudUbo_.Destroy();
        surfaceLightingUbo_.Destroy();
        tonemapProg_ = noiseProg_ = weatherProg_ = cloudsProg_ = shadowProg_ = reprojectProg_ = godRaysProg_ = nullptr;
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : ressources créées par Init.
        // Sorties  : clés clouds.*, volumetrics.godRays et scene.display retirées ; textures et cibles
        //            libérées.
        // Objectif : libération propre avant la destruction du contexte OpenGL.
    }

    void OnGui(EngineContext&) override {
        ImGui::SeparatorText("Nuages volumetriques");
        ImGui::Checkbox("Nuages", &cloudsEnabled_);
        ImGui::TextDisabled("Couverture, densite, pluie : panneau Celestial");
        ImGui::DragFloatRange2("Couche (m)", &cloudBottom_, &cloudTop_, 10.0f, 200.0f, 12000.0f, "%.0f m");
        cloudTop_ = std::max(cloudTop_, cloudBottom_ + 100.0f);
        static const char* kScales[] = {"Pleine", "Demi", "Quart"};
        ImGui::Combo("Resolution", &renderScaleIndex_, kScales, IM_ARRAYSIZE(kScales));
        ImGui::Text("Cible : %d x %d", cloudTargetSize_.x, cloudTargetSize_.y);
        ImGui::SliderInt("Pas primaires", &primarySteps_, 8, 256);
        ImGui::SliderInt("Pas lumiere", &lightSteps_, 1, 16);
        ImGui::SliderFloat("Distance max", &maxDistance_, 5000.0f, 150000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Fondu lointain", &fadeDistance_, 5000.0f, 200000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Distance caracteristique a laquelle les nuages se fondent dans le ciel.");
        ImGui::Checkbox("Reprojection temporelle", &temporal_);
        ImGui::SetItemTooltip("Un pixel par bloc est raymarche a chaque frame, les autres sont reprojetes.");
        ImGui::BeginDisabled(!temporal_);
        static const char* kPatterns[] = {"2 x 2 (1/4 par frame)", "4 x 4 (1/16 par frame)"};
        ImGui::Combo("Motif", &temporalPattern_, kPatterns, IM_ARRAYSIZE(kPatterns));
        ImGui::EndDisabled();
        ImGui::SliderFloat("Echelle bruit de forme", &shapeScale_, 1000.0f, 50000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Echelle bruit de detail", &detailScale_, 50.0f, 5000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Force du detail", &detailStrength_, 0.0f, 1.0f);
        ImGui::SliderFloat("Curl (volutes)", &curlStrength_, 0.0f, 1.0f);
        ImGui::SliderFloat("Inclinaison au vent", &skew_, 0.0f, 3000.0f, "%.0f m");
        ImGui::SliderFloat("Extinction", &extinction_, 0.001f, 0.2f, "%.4f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("HG avant (g)", &hgForward_, 0.0f, 0.99f);
        ImGui::SliderFloat("HG arriere (g)", &hgBackward_, -0.99f, 0.0f);
        ImGui::SliderFloat("Silver lining", &silverIntensity_, 0.0f, 5.0f);
        ImGui::SliderFloat("In-scattering (bords sombres)", &powder_, 0.0f, 2.0f);
        ImGui::SetItemTooltip("Probabilite d'in-scattering de Nubis 2017 (remplace le powder de 2015) : 2 = complet.");
        ImGui::SliderFloat("Influence du vent", &windInfluence_, 0.0f, 10.0f);
        ImGui::SetItemTooltip("Multiplicateur du vent au sol pour l'advection des nuages et des fronts.");
        ImGui::SeparatorText("Cirrus 2.5D (haute altitude)");
        ImGui::SliderFloat("Couverture cirrus", &cirrusCoverage_, 0.0f, 1.0f);
        ImGui::SliderFloat("Type cirrus", &cirrusType_, 0.0f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("0 = stries (streaky), 0.5 = filaments (wispy), 1 = moutonne (round).");
        ImGui::SliderFloat("Altitude cirrus", &cirrusAltitude_, 5000.0f, 12000.0f, "%.0f m");
        ImGui::SliderFloat("Echelle cirrus", &cirrusScale_, 8000.0f, 60000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        ImGui::Checkbox("Eclairs internes (orage)", &lightningEnabled_);
        ImGui::SameLine();
        if (ImGui::Button("Declencher un eclair")) triggerLightning_ = true;

        ImGui::SeparatorText("Ombre des nuages au sol (clouds.shadowMap)");
        ImGui::Checkbox("Ombres des nuages", &cloudShadowsEnabled_);
        ImGui::BeginDisabled(!cloudShadowsEnabled_);
        ImGui::SliderFloat("Opacite maximale", &cloudShadowOpacity_, 0.0f, 1.0f);
        ImGui::SetItemTooltip("1 = un nuage epais coupe entierement le soleil direct (physique).");
        ImGui::SliderFloat("Epaisseur optique", &cloudShadowStrength_, 0.05f, 4.0f, "x%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Emprise de l'ombre", &cloudShadowExtent_, 4000.0f, 80000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Cote de la carte (1024 texels) centree sur la camera ; au-dela, pas d'ombre.");
        if (shadowMap_.IsValid()) {
            ImGui::Image(static_cast<ImTextureID>(static_cast<std::uintptr_t>(shadowMap_.Id())), ImVec2(160, 160));
            ImGui::SetItemTooltip("Transmittance vers l'astre dominant (blanc = plein soleil), camera au centre.");
        }
        ImGui::EndDisabled();

        ImGui::SeparatorText("Carte meteo (fenetre glissante)");
        ImGui::SliderFloat("Emprise", &weatherMapExtent_, 10000.0f, 160000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        ImGui::Checkbox("Fronts localises", &localVariation_);
        ImGui::SetItemTooltip("Decoche : meteo uniforme (seuls les curseurs Celestial comptent).");
        ImGui::BeginDisabled(!localVariation_);
        ImGui::SliderFloat("Echelle des fronts", &frontScale_, 2000.0f, 60000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        ImGui::EndDisabled();
        ImGui::SliderFloat("Taille des nuages", &cloudSize_, 500.0f, 10000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Cellule des empreintes de nuages dans la carte meteo (Nubis : la carte place les nuages).");
        if (weatherMap_.IsValid()) {
            ImGui::Image(static_cast<ImTextureID>(static_cast<std::uintptr_t>(weatherMap_.Id())), ImVec2(160, 160));
            ImGui::SetItemTooltip("R couverture, G pluie, B type, A densite (camera au centre).");
        }

        ImGui::SeparatorText("God rays");
        ImGui::Checkbox("God rays", &godRaysEnabled_);
        ImGui::SetItemTooltip("Lumiere de l'astre diffusee par le brouillard, ombree par les nuages et le relief.\n"
                              "Milieu : brouillard de hauteur (densite : panneau Celestial, decroissance : Lighting).");
        ImGui::BeginDisabled(!godRaysEnabled_);
        ImGui::SliderFloat("Intensite (diffusion)", &godRayIntensity_, 0.0f, 8.0f, "x%.2f");
        ImGui::SliderFloat("Distance de marche", &godRayMaxDistance_, 300.0f, 6000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Anisotropie (g)", &godRayAnisotropy_, 0.0f, 0.95f, "%.2f");
        ImGui::SliderInt("Pas", &godRaySteps_, 8, 96);
        ImGui::SliderFloat("Rayons dans le ciel", &godRaySkyShafts_, 0.0f, 1.0f);
        ImGui::SetItemTooltip("Rayons crepusculaires : la brume a l'ombre des nuages ne diffuse plus le soleil.");
        ImGui::EndDisabled();

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
    /// Ressources des nuages (premier Render), génération des bruits, carte météo et raymarching.
    /// Retourne true si la cible des nuages est prête à être composée.
    bool RenderClouds(EngineContext& ctx) {
        if (!noiseProg_->IsValid() || !weatherProg_->IsValid() || !cloudsProg_->IsValid()) return false;
        const core::RenderBlackboard& bb = ctx.blackboard;
        if (!noiseReady_) {
            GenerateNoise();
            gpu::Texture2D::Desc d;
            d.width = d.height = kWeatherSize;
            d.internalFormat   = GL_RGBA8;
            d.sampler          = gpu::SamplerDesc{};  // linéaire, GL_CLAMP_TO_EDGE
            d.debugName        = "clouds.weatherMap";
            weatherMap_.Create(d);
            ctx.blackboard.PublishTexture(env::bb::CloudShapeNoise, shapeNoise_, kProducer);
            ctx.blackboard.PublishTexture(env::bb::CloudDetailNoise, detailNoise_, kProducer);
            ctx.blackboard.PublishTexture(env::bb::CloudWeatherMap, weatherMap_, kProducer);
            // Ombre des nuages : R16F, linéaire, GL_CLAMP_TO_EDGE, initialisée à 1 (plein soleil).
            gpu::Texture2D::Desc sd;
            sd.width = sd.height = kShadowSize;
            sd.internalFormat    = GL_R16F;
            sd.sampler           = gpu::SamplerDesc{};
            sd.debugName         = "clouds.shadowMap";
            shadowMap_.Create(sd);
            const float one[4] = {1.0f, 1.0f, 1.0f, 1.0f};
            shadowMap_.Clear(GL_RGBA, GL_FLOAT, one);
            ctx.blackboard.PublishTexture(env::bb::CloudShadowMap, shadowMap_, kProducer);
            noiseReady_ = true;
        }
        if (!cloudTargets_[0].IsValid() || !reprojectProg_->IsValid()) return false;

        // --- Passe 1 : carte météo glissante (recalculée uniquement si dérive / paramètres / shader changent) ---
        bool weatherUpdated = false;
        {
            const std::uint32_t wGen = weatherProg_->Generation();
            const glm::vec3 wEnv(ctx.environment.weatherExtra.x, ctx.environment.weatherExtra.y, ctx.environment.weather.x);
            const float     wTol = 0.02f * (weatherMapExtent_ / static_cast<float>(kWeatherSize));
            const bool dirty = weatherMapDirty_ || wGen != lastWeatherGen_ ||
                               std::abs(weatherMapExtent_ - lastWeatherExtent_) > 1e-2f ||
                               std::abs(frontScale_ - lastFrontScale_) > 1e-2f ||
                               std::abs(cloudSize_ - lastCloudSize_) > 1e-2f ||
                               std::abs(static_cast<float>(frontPeriod_) - lastFrontPeriod_) > 1.0f ||
                               localVariation_ != lastLocalVariation_ ||
                               glm::any(glm::greaterThan(glm::abs(wEnv - lastWeatherEnv_), glm::vec3(1e-4f))) ||
                               WrappedDelta2D(weatherCenterMod_, lastWeatherCenterMod_, static_cast<float>(frontPeriod_)) > wTol;
            if (dirty) {
                const gpu::ShaderProgram& p = *weatherProg_;
                p.Set("uCenterMod", weatherCenterMod_);
                p.Set("uPeriod", static_cast<float>(frontPeriod_));
                p.Set("uExtent", weatherMapExtent_);
                p.Set("uFrontScale", frontScale_);
                p.Set("uLocalVariation", localVariation_);
                // Empreintes : nombre entier de cellules sur la période (bruit périodique, sans couture).
                const int cells = std::max(1, static_cast<int>(std::lround(frontPeriod_ / cloudSize_)));
                p.Set("uCloudCells", cells);
                weatherMap_.BindImage(env::bind::image::VolumetricOut, GL_WRITE_ONLY);
                p.Use();
                glDispatchCompute(kWeatherSize / 8, kWeatherSize / 8, 1);
                glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

                weatherMapDirty_      = false;
                lastWeatherGen_       = wGen;
                lastWeatherCenterMod_ = weatherCenterMod_;
                lastWeatherExtent_    = weatherMapExtent_;
                lastFrontScale_       = frontScale_;
                lastCloudSize_        = cloudSize_;
                lastFrontPeriod_      = static_cast<float>(frontPeriod_);
                lastLocalVariation_   = localVariation_;
                lastWeatherEnv_       = wEnv;
                weatherUpdated        = true;
            }
        }
        // Mêmes paramètres pour toutes les passes de cette frame, et pour la capture de l'IBL de la suivante
        // (DynamicIBLSystem, stage AtmosphereIBL : il lit la carte météo et les bruits publiés ici).
        cloudParams_ = MakeCloudParams(ctx.renderOriginChunk);
        SharedLighting().clouds = cloudParams_;
        cloudUbo_.UploadObject(BuildCloudParamsUbo(cloudParams_, cloudParams_.originChunk));
        cloudUbo_.BindBase(GL_UNIFORM_BUFFER, env::bind::ubo::VolumetricParams);

        // Ressources du modèle de densité, communes aux passes 2 (ombre) et 3 (raymarching).
        shapeNoise_.BindUnit(env::bind::tex::CloudShapeNoise);
        detailNoise_.BindUnit(env::bind::tex::CloudDetailNoise);
        weatherMap_.BindUnit(env::bind::tex::CloudWeatherMap);

        // --- Passe 2 : carte d'ombre des nuages au sol (lue par l'éclairage à la frame suivante) ---
        RenderCloudShadow(ctx, weatherUpdated);

        // --- Passe 3 : raymarching en résolution réduite ---
        {
            const LightingShared& sh = SharedLighting();
            bb.BindTexture(env::bb::GBufferDepth, env::bind::tex::GBufferDepth);
            bb.BindTexture(env::bb::AtmoTransmittanceLUT, env::bind::tex::AtmoTransmittance);
            bb.BindTexture(env::bb::AtmoSkyViewLUT, env::bind::tex::AtmoSkyView);
            const gpu::ShaderProgram& p = *cloudsProg_;
            p.Set("uAtmoAvailable", sh.atmoAvailable);
            p.Set("uGBufferAvailable", sh.gbufferAvailable);
            p.Set("uPrimarySteps", primarySteps_);
            p.Set("uLightSteps", lightSteps_);
            // Ping-pong : la cible de la frame précédente devient l'historique.
            current_ ^= 1;
            const int        block = historyValid_ ? prevBlock_ : 1;  // historique invalide : tout raymarcher
            const glm::ivec2 offset = BayerOffset(block, ctx.time.frameIndex);
            SetTemporalUniforms(p, ctx, block, offset);
            cloudTargets_[current_].BindImage(env::bind::image::VolumetricOut, GL_WRITE_ONLY);
            cloudTargets_[current_ ^ 1].BindUnit(env::bind::tex::GodRays);  // historique (cf. cloud_temporal.glsl)
            p.Use();
            const glm::ivec2 cells = (cloudTargetSize_ + block - 1) / block;
            glDispatchCompute(static_cast<GLuint>((cells.x + 7) / 8), static_cast<GLuint>((cells.y + 7) / 8), 1);
            // Pixels frais relus par imageLoad dans la résolution.
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

            // --- Passe 4 : résolution temporelle (pixels non raymarchés) ---
            if (block > 1) {
                const gpu::ShaderProgram& r = *reprojectProg_;
                SetTemporalUniforms(r, ctx, block, offset);
                r.Set("uAtmoAvailable", sh.atmoAvailable);
                cloudTargets_[current_].BindImage(env::bind::image::VolumetricOut, GL_READ_WRITE);
                r.Use();
                glDispatchCompute(static_cast<GLuint>((cloudTargetSize_.x + 7) / 8),
                                  static_cast<GLuint>((cloudTargetSize_.y + 7) / 8), 1);
            }
            // Lu par imageLoad dans la passe de tonemap, puis comme historique (texture) à la frame suivante.
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
            historyValid_ = true;
        }
        return true;
    }

    /// God rays : marche dans le brouillard ombré par les nuages (clouds.shadowMap) et le relief (VXGI).
    bool RenderGodRays(EngineContext& ctx) {
        if (!godRaysTargets_[0].IsValid()) return false;
        const core::RenderBlackboard& bb = ctx.blackboard;
        BindSurfaceLightingResources(bb);  // ombre des nuages, grille VXGI, transmittance, ciel
        bb.BindTexture(env::bb::SceneDepth, env::bind::tex::SceneDepth);
        godRaysCurrent_ ^= 1;
        godRaysTargets_[godRaysCurrent_ ^ 1].BindUnit(env::bind::tex::GodRays);
        const gpu::ShaderProgram& p = *godRaysProg_;
        surfaceLightingUbo_.UploadObject(BuildSurfaceLightingUbo(ctx.renderOriginChunk));
        surfaceLightingUbo_.BindBase(GL_UNIFORM_BUFFER, env::bind::ubo::PostParams);
        p.Set("uAtmoAvailable", SharedLighting().atmoAvailable);
        p.Set("uSteps", godRaySteps_);
        p.Set("uMaxDistance", godRayMaxDistance_);
        p.Set("uIntensity", godRayIntensity_);
        p.Set("uAnisotropy", godRayAnisotropy_);
        p.Set("uSkyShafts", godRaySkyShafts_);
        p.Set("uPrevViewProj", prevViewProj_);
        p.Set("uOriginShift", glm::vec2(ctx.renderOriginChunk - prevOriginChunk_) * kChunkSizeMeters);
        p.Set("uHistoryValid", godRaysHistoryValid_);
        godRaysTargets_[godRaysCurrent_].BindImage(env::bind::image::VolumetricOut, GL_WRITE_ONLY);
        p.Use();
        glDispatchCompute(static_cast<GLuint>((godRaysSize_.x + 7) / 8), static_cast<GLuint>((godRaysSize_.y + 7) / 8), 1);
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);  // lue par texelFetch dans le tonemap
        godRaysHistoryValid_ = true;
        ctx.blackboard.PublishTexture(env::bb::VolumetricGodRays, godRaysTargets_[godRaysCurrent_], kProducer);
        return true;
    }

    /// Pixel raymarché dans chaque bloc à la frame `frame` (matrices de Bayer : les B² positions sont
    /// visitées en B² frames, dans un ordre qui disperse les pixels frais successifs).
    static glm::ivec2 BayerOffset(int block, std::uint64_t frame) {
        if (block <= 1) return glm::ivec2(0);
        static constexpr int kBayer2[4][2]  = {{0, 0}, {1, 1}, {1, 0}, {0, 1}};
        static constexpr int kBayer4[16][2] = {{0, 0}, {2, 2}, {2, 0}, {0, 2}, {1, 1}, {3, 3}, {3, 1}, {1, 3},
                                               {1, 0}, {3, 2}, {3, 0}, {1, 2}, {0, 1}, {2, 3}, {2, 1}, {0, 3}};
        if (block == 2) {
            const auto& o = kBayer2[frame % 4];
            return {o[0], o[1]};
        }
        const auto& o = kBayer4[frame % 16];
        return {o[0], o[1]};
    }

    /// Uniforms de cloud_temporal.glsl.
    void SetTemporalUniforms(const gpu::ShaderProgram& p, const EngineContext& ctx, int block, glm::ivec2 offset) const {
        p.Set("uBlock", block);
        p.Set("uBlockOffset", offset);
        p.Set("uHistoryValid", historyValid_);
        p.Set("uPrevViewProj", prevViewProj_);
        // Origine courante - origine précédente (m) : exacte, différence entière de chunks.
        p.Set("uOriginShift", glm::vec2(ctx.renderOriginChunk - prevOriginChunk_) * kChunkSizeMeters);
        p.Set("uWindDelta", windDeltaXZ_);
    }

    /// Paramètres des nuages de cette frame (uniforms de cloud_density.glsl et cloud_march.glsl).
    CloudParams MakeCloudParams(glm::ivec2 renderOriginChunk) const {
        CloudParams c;
        c.valid            = true;
        c.bottom           = cloudBottom_;
        c.top              = cloudTop_;
        c.shapeScale       = shapeScale_;
        c.detailScale      = detailScale_;
        c.detailStrength   = detailStrength_;
        c.curl             = curlStrength_;
        c.shapeOffset      = shapeOffset_;
        c.detailOffset     = detailOffset_;
        c.weatherOriginRel = weatherOriginRel_;
        c.weatherExtent    = weatherMapExtent_;
        c.skew             = windDirXZ_ * skew_;
        c.extinction       = extinction_;
        c.hgForward        = hgForward_;
        c.hgBackward       = hgBackward_;
        c.silver           = silverIntensity_;
        c.powder           = powder_;
        c.maxDistance      = maxDistance_;
        c.fadeDistance     = fadeDistance_;
        c.cirrusAltitude   = cirrusAltitude_;
        c.cirrusCoverage   = cirrusCoverage_;
        c.cirrusType       = cirrusType_;
        c.cirrusScale      = cirrusScale_;
        c.cirrusOffset     = cirrusOffset_;
        c.lightningPosRel    = lightningPosRel_;
        c.lightningIntensity = lightningIntensity_;
        c.originChunk      = renderOriginChunk;
        return c;
    }

    /// Carte d'ombre : astre dominant (soleil, sinon lune) ; sans astre levé ou ombres coupées, la carte
    /// est déclarée invalide et l'éclairage n'ombre pas (CloudShadow() renvoie 1).
    void RenderCloudShadow(EngineContext& ctx, bool weatherUpdated) {
        LightingShared& sh = SharedLighting();
        if (!cloudShadowsEnabled_ || !shadowProg_->IsValid() || !shadowLightUp_) {
            sh.cloudShadowValid = false;
            return;
        }
        const std::uint32_t sGen = shadowProg_->Generation();
        const float         sTol = 0.02f * (cloudShadowExtent_ / static_cast<float>(kShadowSize));
        const glm::vec2     skew = windDirXZ_ * skew_;
        const bool dirty = shadowMapDirty_ || weatherUpdated || !sh.cloudShadowValid || ctx.renderOriginShifted ||
                           sGen != lastShadowGen_ ||
                           glm::length(shadowOriginRel_ - lastShadowOriginRel_) > 1e-2f ||
                           glm::length(shadowLight_ - lastShadowLight_) > 1e-5f ||
                           WrappedDelta2D(glm::vec2(shapeOffset_.x, shapeOffset_.z),
                                          glm::vec2(lastShapeOffset_.x, lastShapeOffset_.z), shapeScale_) > sTol ||
                           std::abs(cloudBottom_ - lastShadowBottom_) > 1e-2f ||
                           std::abs(cloudTop_ - lastShadowTop_) > 1e-2f ||
                           std::abs(shapeScale_ - lastShadowShapeScale_) > 1e-2f ||
                           glm::length(skew - lastShadowSkew_) > 1e-2f ||
                           std::abs(extinction_ - lastShadowExtinction_) > 1e-5f ||
                           std::abs(cloudShadowExtent_ - lastShadowExtent_) > 1e-2f ||
                           std::abs(cloudShadowStrength_ - lastShadowStrength_) > 1e-4f ||
                           std::abs(cloudShadowOpacity_ - lastShadowOpacity_) > 1e-4f;
        if (!dirty) return;

        const gpu::ShaderProgram& p = *shadowProg_;
        p.Set("uShadowOrigin", shadowOriginRel_);
        p.Set("uShadowExtent", cloudShadowExtent_);
        p.Set("uLightDir", shadowLight_);
        p.Set("uShadowStrength", cloudShadowStrength_);
        p.Set("uShadowOpacity", cloudShadowOpacity_);
        shadowMap_.BindImage(env::bind::image::VolumetricOut, GL_WRITE_ONLY);
        p.Use();
        glDispatchCompute(kShadowSize / 8, kShadowSize / 8, 1);
        // Lue par texture() dans l'éclairage (frame suivante).
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

        // Tout ce qu'il faut pour relire la carte exactement comme elle a été calculée.
        sh.cloudShadowValid       = true;
        sh.cloudLayerAltitude     = cloudBottom_;
        sh.cloudShadowExtent      = cloudShadowExtent_;
        sh.cloudShadowOrigin      = shadowOriginRel_;
        sh.cloudShadowOriginChunk = ctx.renderOriginChunk;
        sh.cloudShadowLight       = shadowLight_;

        shadowMapDirty_       = false;
        lastShadowGen_        = sGen;
        lastShadowOriginRel_  = shadowOriginRel_;
        lastShadowLight_      = shadowLight_;
        lastShapeOffset_      = shapeOffset_;
        lastShadowBottom_     = cloudBottom_;
        lastShadowTop_        = cloudTop_;
        lastShadowShapeScale_ = shapeScale_;
        lastShadowSkew_       = skew;
        lastShadowExtinction_ = extinction_;
        lastShadowExtent_     = cloudShadowExtent_;
        lastShadowStrength_   = cloudShadowStrength_;
        lastShadowOpacity_    = cloudShadowOpacity_;
    }

    /// Bruits 3D périodiques, une seule fois (quelques ms). Mips par le pilote : RGBA8 128³ généré une fois.
    void GenerateNoise() {
        gpu::Texture3D::Desc d;
        d.width = d.height = d.depth = kShapeSize;
        d.internalFormat = GL_RGBA8;
        d.levels         = 0;
        d.sampler        = gpu::SamplerDesc::LinearMipmapped(1.0f);
        d.debugName      = "clouds.shapeNoise";
        shapeNoise_.Create(d);
        d.width = d.height = d.depth = kDetailSize;
        d.debugName = "clouds.detailNoise";
        detailNoise_.Create(d);

        noiseProg_->Use();
        noiseProg_->Set("uPass", 0);
        shapeNoise_.BindImage(env::bind::image::VolumetricOut, GL_WRITE_ONLY, 0, GL_TRUE, 0, GL_RGBA8);
        glDispatchCompute(kShapeSize / 4, kShapeSize / 4, kShapeSize / 4);
        noiseProg_->Set("uPass", 1);
        detailNoise_.BindImage(env::bind::image::VolumetricOut, GL_WRITE_ONLY, 0, GL_TRUE, 0, GL_RGBA8);
        glDispatchCompute(kDetailSize / 4, kDetailSize / 4, kDetailSize / 4);
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT);
        shapeNoise_.GenerateMipmaps();
        detailNoise_.GenerateMipmaps();
    }

    gpu::ShaderProgram* tonemapProg_  = nullptr;
    gpu::VertexArray    emptyVao_;
    gpu::Framebuffer    display_;
    gpu::GpuBuffer      cloudUbo_;
    gpu::GpuBuffer      surfaceLightingUbo_;
    bool                displayDirty_ = true;  ///< scene.display à (re)publier

    // Nuages
    gpu::ShaderProgram* noiseProg_   = nullptr;
    gpu::ShaderProgram* weatherProg_ = nullptr;
    gpu::ShaderProgram* cloudsProg_  = nullptr;
    gpu::ShaderProgram* shadowProg_  = nullptr;
    gpu::Texture2D      shadowMap_;
    glm::vec2           shadowOriginRel_{0.0f};
    glm::vec3           shadowLight_{0.0f, 1.0f, 0.0f};
    bool                shadowLightUp_       = false;
    bool                cloudShadowsEnabled_ = true;
    float               cloudShadowExtent_   = 14000.0f;
    float               cloudShadowStrength_ = 1.0f;
    float               cloudShadowOpacity_  = 1.0f;
    bool                weatherMapDirty_     = true;
    std::uint32_t       lastWeatherGen_      = 0;
    glm::vec2           lastWeatherCenterMod_{0.0f};
    float               lastWeatherExtent_   = -1.0f;
    float               lastFrontScale_      = -1.0f;
    float               lastCloudSize_       = -1.0f;
    float               lastFrontPeriod_     = -1.0f;
    bool                lastLocalVariation_  = true;
    glm::vec3           lastWeatherEnv_{-1.0f};
    bool                shadowMapDirty_      = true;
    std::uint32_t       lastShadowGen_       = 0;
    glm::vec2           lastShadowOriginRel_{0.0f};
    glm::vec3           lastShadowLight_{0.0f, 1.0f, 0.0f};
    glm::vec3           lastShapeOffset_{0.0f};
    float               lastShadowBottom_    = -1.0f;
    float               lastShadowTop_       = -1.0f;
    float               lastShadowShapeScale_= -1.0f;
    glm::vec2           lastShadowSkew_{0.0f};
    float               lastShadowExtinction_= -1.0f;
    float               lastShadowExtent_    = -1.0f;
    float               lastShadowStrength_  = -1.0f;
    float               lastShadowOpacity_   = -1.0f;
    gpu::Texture3D      shapeNoise_;
    gpu::Texture3D      detailNoise_;
    gpu::Texture2D      weatherMap_;
    gpu::Texture2D      cloudTargets_[2];       ///< ping-pong : courante / historique
    int                 current_ = 0;
    glm::ivec2          cloudTargetSize_{0};
    gpu::ShaderProgram* reprojectProg_ = nullptr;
    gpu::ShaderProgram* godRaysProg_   = nullptr;
    gpu::Texture2D      godRaysTargets_[2];
    int                 godRaysCurrent_      = 0;
    glm::ivec2          godRaysSize_{0};
    bool                godRaysActive_       = false;
    bool                godRaysHistoryValid_ = false;
    bool                historyValid_  = false;
    bool                hasPrevFrame_  = false;
    int                 temporalPattern_ = 0;   ///< 0 = 2×2, 1 = 4×4
    int                 prevBlock_       = 1;   ///< taille de bloc de la frame courante (fixée dans Update)
    glm::mat4           prevViewProj_{1.0f};
    glm::mat4           lastViewProj_{1.0f};
    glm::ivec2          prevOriginChunk_{0};
    glm::ivec2          lastOriginChunk_{0};
    glm::vec3           prevLight_{0.0f, 1.0f, 0.0f};
    glm::vec4           prevWeather_{0.0f};
    CloudParams         cloudParams_;
    bool                noiseReady_ = false;
    double              windOffset_[2] = {0.0, 0.0};  ///< m, accumulé (double : monde infini)
    double              frontPeriod_   = 1.0;
    glm::vec2           weatherOriginRel_{0.0f};
    glm::vec2           weatherCenterMod_{0.0f};
    glm::vec3           shapeOffset_{0.0f};
    glm::vec3           detailOffset_{0.0f};
    float               weatherMapExtent_ = 64000.0f;
    float               frontScale_       = 15000.0f;
    bool                localVariation_   = true;
    float               maxDistance_      = 60000.0f;
    float               fadeDistance_     = 45000.0f;
    float               cloudSize_        = 2500.0f;
    float               curlStrength_     = 0.35f;
    float               skew_             = 500.0f;
    glm::vec2           windDirXZ_{1.0f, 0.0f};
    glm::vec2           windDeltaXZ_{0.0f};

    // Variables liées aux sliders d'OnGui (god rays : aucune logique branchée).
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
    float cirrusAltitude_   = 7500.0f;
    float cirrusCoverage_   = 0.35f;
    float cirrusType_       = 0.5f;     ///< 0 = streaky, 0.5 = wispy, 1 = round
    float cirrusScale_      = 24000.0f;
    glm::vec2 cirrusOffset_{0.0f};
    bool lightningEnabled_   = true;
    bool triggerLightning_   = false;
    float lightningTimer_     = 0.0f;   ///< temps restant sur l'éclair en cours (s)
    float lightningCooldown_  = 3.0f;   ///< attente avant le prochain éclair (s)
    glm::vec3 lightningPosRel_{0.0f};
    float lightningIntensity_ = 0.0f;
    std::uint32_t lightningRng_= 1337u;
    bool  godRaysEnabled_   = true;
    float godRayIntensity_  = 1.0f;
    float godRayMaxDistance_= 2500.0f;
    float godRayAnisotropy_ = 0.8f;
    int   godRaySteps_      = 32;
    float godRaySkyShafts_  = 1.0f;
    float exposureEv_       = 13.0f;  // EV100 manuel (journée ensoleillée ≈ 15)
    bool  autoExposure_     = true;
    int   tonemapper_       = 0;      // ACES : même rendu qu'avant la passe de tonemap
    float whitePoint_       = 8.0f;
};

}  // namespace
REGISTER_SYSTEM(VolumetricGodRaysSystem, PassStage::VolumetricsPost);
}  // namespace env::render
