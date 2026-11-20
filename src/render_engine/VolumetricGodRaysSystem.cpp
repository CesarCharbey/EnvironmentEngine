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
//  À faire : reprojection temporelle, ombre des nuages (clouds.shadowMap), god rays.
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
constexpr int              kFrontPeriodCells = 128;  ///< période des fronts = 128 cellules de base

/// x mod period dans [0, period), en double (positions absolues d'un monde infini).
double WrapPositive(double x, double period) { return x - std::floor(x / period) * period; }

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
        noiseProg_   = &ctx.shaders.LoadCompute("render.cloud_noise", "render/cloud_noise.comp");
        weatherProg_ = &ctx.shaders.LoadCompute("render.cloud_weather_map", "render/cloud_weather_map.comp");
        cloudsProg_  = &ctx.shaders.LoadCompute("render.volumetric_clouds", "render/volumetric_clouds.comp");
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
        const glm::vec2 wind  = glm::vec2(ctx.environment.windDirSpeed);
        const double    speed = static_cast<double>(ctx.environment.windDirSpeed.z) * windInfluence_;
        if (glm::length(wind) > 1e-4f) {
            const glm::vec2 d = glm::normalize(wind);
            windDirXZ_ = d;
            windOffset_[0] += d.x * speed * ctx.time.dt;
            windOffset_[1] += d.y * speed * ctx.time.dt;
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

        // --- Cible des nuages (résolution réduite), allouée / redimensionnée ici comme le G-Buffer ---
        const int div = 1 << std::clamp(renderScaleIndex_, 0, 2);
        const glm::ivec2 cs((w + div - 1) / div, (h + div - 1) / div);
        if (cs != cloudTargetSize_) {
            gpu::Texture2D::Desc d;
            d.width          = cs.x;
            d.height         = cs.y;
            d.internalFormat = GL_RGBA16F;
            d.debugName      = "clouds.target";
            cloudTarget_.Create(d);
            cloudTargetSize_ = cs;
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

        // --- Nuages : bruits (une fois), carte météo, raymarching ---
        const bool clouds = cloudsEnabled_ && RenderClouds(ctx);

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
        if (clouds) {
            ctx.blackboard.BindTexture(env::bb::SceneDepth, env::bind::tex::SceneDepth);
            glBindImageTexture(env::bind::image::VolumetricOut, cloudTarget_.Id(), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
        }
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
        for (std::string_view key : {env::bb::CloudShapeNoise, env::bb::CloudDetailNoise, env::bb::CloudWeatherMap})
            ctx.blackboard.UnpublishIfProducer(key, kProducer);
        display_.Destroy();
        emptyVao_.Destroy();
        shapeNoise_.Destroy();
        detailNoise_.Destroy();
        weatherMap_.Destroy();
        cloudTarget_.Destroy();
        tonemapProg_ = noiseProg_ = weatherProg_ = cloudsProg_ = nullptr;
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : ressources créées par Init.
        // Sorties  : clés clouds.*, volumetrics.godRays et scene.display retirées ; textures et cibles
        //            libérées.
        // Objectif : libération propre avant la destruction du contexte OpenGL.
    }

    void OnGui(EngineContext&) override {
        TodoBanner("reprojection temporelle, ombre des nuages, god rays");

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
        ImGui::BeginDisabled();
        ImGui::Checkbox("Reprojection temporelle (a venir)", &temporal_);
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
            noiseReady_ = true;
        }
        if (!cloudTarget_.IsValid()) return false;

        // --- Passe 1 : carte météo glissante ---
        {
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
        }

        // --- Passe 2 : raymarching en résolution réduite ---
        {
            const LightingShared& sh = SharedLighting();
            bb.BindTexture(env::bb::GBufferDepth, env::bind::tex::GBufferDepth);
            bb.BindTexture(env::bb::AtmoTransmittanceLUT, env::bind::tex::AtmoTransmittance);
            bb.BindTexture(env::bb::AtmoSkyViewLUT, env::bind::tex::AtmoSkyView);
            bb.BindTexture(env::bb::IblIrradiance, env::bind::tex::IblIrradiance);
            shapeNoise_.BindUnit(env::bind::tex::CloudShapeNoise);
            detailNoise_.BindUnit(env::bind::tex::CloudDetailNoise);
            weatherMap_.BindUnit(env::bind::tex::CloudWeatherMap);
            const gpu::ShaderProgram& p = *cloudsProg_;
            p.Set("uAtmoAvailable", sh.atmoAvailable);
            p.Set("uGBufferAvailable", sh.gbufferAvailable);
            p.Set("uIblAvailable", sh.iblAvailable);
            p.Set("uCloudBottom", cloudBottom_);
            p.Set("uCloudTop", cloudTop_);
            p.Set("uPrimarySteps", primarySteps_);
            p.Set("uLightSteps", lightSteps_);
            p.Set("uShapeScale", shapeScale_);
            p.Set("uDetailScale", detailScale_);
            p.Set("uDetailStrength", detailStrength_);
            p.Set("uExtinction", extinction_);
            p.Set("uHgForward", hgForward_);
            p.Set("uHgBackward", hgBackward_);
            p.Set("uSilver", silverIntensity_);
            p.Set("uPowder", powder_);
            p.Set("uShapeOffset", shapeOffset_);
            p.Set("uDetailOffset", detailOffset_);
            p.Set("uWeatherOriginRel", weatherOriginRel_);
            p.Set("uWeatherExtent", weatherMapExtent_);
            p.Set("uMaxDistance", maxDistance_);
            p.Set("uFadeDistance", fadeDistance_);
            p.Set("uSkew", windDirXZ_ * skew_);
            p.Set("uCurlStrength", curlStrength_);
            cloudTarget_.BindImage(env::bind::image::VolumetricOut, GL_WRITE_ONLY);
            p.Use();
            glDispatchCompute(static_cast<GLuint>((cloudTargetSize_.x + 7) / 8),
                              static_cast<GLuint>((cloudTargetSize_.y + 7) / 8), 1);
            // Lu ensuite par imageLoad dans la passe de tonemap.
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        }
        return true;
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
    bool                displayDirty_ = true;  ///< scene.display à (re)publier

    // Nuages
    gpu::ShaderProgram* noiseProg_   = nullptr;
    gpu::ShaderProgram* weatherProg_ = nullptr;
    gpu::ShaderProgram* cloudsProg_  = nullptr;
    gpu::Texture3D      shapeNoise_;
    gpu::Texture3D      detailNoise_;
    gpu::Texture2D      weatherMap_;
    gpu::Texture2D      cloudTarget_;
    glm::ivec2          cloudTargetSize_{0};
    bool                noiseReady_ = false;
    double              windOffset_[2] = {0.0, 0.0};  ///< m, accumulé (double : monde infini)
    double              frontPeriod_   = 1.0;
    glm::vec2           weatherOriginRel_{0.0f};
    glm::vec2           weatherCenterMod_{0.0f};
    glm::vec3           shapeOffset_{0.0f};
    glm::vec3           detailOffset_{0.0f};
    float               weatherMapExtent_ = 40000.0f;
    float               frontScale_       = 15000.0f;
    bool                localVariation_   = true;
    float               maxDistance_      = 60000.0f;
    float               fadeDistance_     = 45000.0f;
    float               cloudSize_        = 2500.0f;
    float               curlStrength_     = 0.35f;
    float               skew_             = 500.0f;
    glm::vec2           windDirXZ_{1.0f, 0.0f};

    // Variables liées aux sliders d'OnGui (god rays : aucune logique branchée).
    bool  cloudsEnabled_    = true;
    float cloudBottom_      = 1500.0f;
    float cloudTop_         = 4500.0f;
    int   renderScaleIndex_ = 1;
    int   primarySteps_     = 64;
    int   lightSteps_       = 6;
    bool  temporal_         = true;
    float shapeScale_       = 9000.0f;
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
