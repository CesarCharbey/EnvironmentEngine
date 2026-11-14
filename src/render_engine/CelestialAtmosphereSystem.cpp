// =============================================================================
//  src/render_engine/CelestialAtmosphereSystem.cpp                  [DEV 2]
//  Stage : AtmosphereIBL (ordre 0, avant DynamicIBLSystem)
//
//  Rôle : astres, phases lunaires et météo temps réel ; atmosphère physique par
//  LUTs. Unique écrivain des champs soleil, lune, vent et météo du
//  GlobalEnvironmentUBO (AGENTS.md section 8). Tant que ce système ne les écrit
//  pas, les valeurs par défaut posées par le core restent en place.
//
//  Coquille vide : toute la logique est à écrire par Dev 2.
//  RÈGLE : ce fichier n'inclut jamais src/world_engine/ (vérifié au build).
// =============================================================================
#include <imgui.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string_view>

#include "contracts/ISystemModule.hpp"
#include "contracts/WorldContracts.hpp"
#include "contracts/GpuBindings.hpp"
#include "core/SystemRegistry.hpp"
#include "core/RenderBlackboard.hpp"
#include "core/gpu/ShaderManager.hpp"
#include "core/gpu/Texture2D.hpp"
#include "core/gpu/GpuBuffer.hpp"

namespace env::render {
namespace {

// Structure std140 envoyée sur le slot UBO_AtmosphereParams (slot 9)
struct alignas(16) AtmosphereParamsUBO {
    glm::vec4 radiiAndScaleHeights; // x = Rb (km), y = Rt (km), z = HR (km), w = HM (km)
    glm::vec4 rayleighScattering;   // xyz = beta_R (1/km), w = mieAnisotropy g
    glm::vec4 mieScatteringAndExt;  // xyz = beta_M_sca (1/km), w = beta_M_ext (1/km)
    glm::vec4 ozoneAbsorption;      // xyz = beta_O3 (1/km), w = ozoneEnabled (0 ou 1)
    glm::vec4 groundAlbedoAndTurb;  // xyz = groundAlbedo, w = turbidityFactor (météo)
    glm::vec4 nightSky;             // x = intensité du fond de ciel nocturne (étoiles + airglow), yzw = 0
};
static_assert(sizeof(AtmosphereParamsUBO) % 16 == 0, "AtmosphereParamsUBO must be std140 aligned");

// Conversion Azimut (deg, 0 = -Z Nord, 90 = +X Est) & Élévation (deg) -> Direction unitaire
glm::vec3 AzimuthElevationToDir(float azimuthDeg, float elevationDeg) {
    const float az = glm::radians(azimuthDeg);
    const float el = glm::radians(elevationDeg);
    const float cosEl = std::cos(el);
    return glm::normalize(glm::vec3(
        cosEl * std::sin(az),
        std::sin(el),
        -cosEl * std::cos(az)
    ));
}

// Conversion inverse Direction -> (Azimut deg, Élévation deg)
void DirToAzimuthElevation(const glm::vec3& dir, float& outAzimuthDeg, float& outElevationDeg) {
    outElevationDeg = glm::degrees(std::asin( std::clamp(dir.y, -1.0f, 1.0f) ));
    float az = glm::degrees(std::atan2(dir.x, -dir.z));
    if (az < 0.0f) az += 360.0f;
    outAzimuthDeg = az;
}

void TodoBanner(const char* scope) {
    ImGui::TextColored(ImVec4(0.35f, 0.80f, 1.0f, 1.0f), "TODO Dev 2");
    ImGui::SameLine();
    ImGui::TextDisabled("%s", scope);
}

constexpr std::string_view kProducer = "CelestialAtmosphereSystem";

const char* MoonPhaseName(float phase) {
    static const char* kNames[8] = {"nouvelle lune", "premier croissant", "premier quartier", "gibbeuse croissante",
                                    "pleine lune",   "gibbeuse decroissante", "dernier quartier", "dernier croissant"};
    const int index = static_cast<int>(std::floor(phase * 8.0f + 0.5f)) % 8;
    return kNames[index];
}

class CelestialAtmosphereSystem final : public ISystemModule {
public:
    void Init(EngineContext& ctx) override {
        // Entrées  : shaders compute à créer dans shaders/render/ ; slots du contrat UBO_AtmosphereParams,
        //            IMG_AtmoLutOut.
        // Sorties  : LUT de transmittance (RGBA16F 256 x 64), LUT de multi-diffusion (RGBA16F 32 x 32),
        //            LUT sky-view (RGBA16F 192 x 108), programmes compute, UBO de paramètres atmosphériques.
        // Chargement des 3 Compute Shaders (Hillaire 2020)
        transmittanceProg_   = &ctx.shaders.LoadCompute("atmo.transmittance",   "render/atmo_transmittance.comp");
        multiScatteringProg_ = &ctx.shaders.LoadCompute("atmo.multiscattering", "render/atmo_multiscattering.comp");
        skyViewProg_         = &ctx.shaders.LoadCompute("atmo.skyview",         "render/atmo_skyview.comp");

        // Allocation des 3 LUTs aux résolutions exactes (Texture2D::Desc, sampler par défaut = linéaire + clamp)
        transmittanceLUT_.Create({.width = 256, .height = 64,  .internalFormat = GL_RGBA16F, .levels = 1,
                                  .sampler = {}, .debugName = "atmo.transmittanceLUT"});
        multiScatteringLUT_.Create({.width = 32, .height = 32, .internalFormat = GL_RGBA16F, .levels = 1,
                                    .sampler = {}, .debugName = "atmo.multiScatteringLUT"});
        // Sky-view : u = azimut absolu / 2π -> GL_REPEAT en S pour éviter une couture au nord.
        gpu::SamplerDesc skyViewSampler{};
        skyViewSampler.wrapS = GL_REPEAT;
        skyViewLUT_.Create({.width = 192, .height = 108, .internalFormat = GL_RGBA16F, .levels = 1,
                            .sampler = skyViewSampler, .debugName = "atmo.skyViewLUT"});

        // Allocation de l'UBO des paramètres atmosphériques (UBO_AtmosphereParams = slot 9)
        // Signature : Create(size, flags, data, debugName) -> data = nullptr
        atmoUbo_.Create(sizeof(AtmosphereParamsUBO), GL_DYNAMIC_STORAGE_BIT, nullptr, "atmo.paramsUBO");

        // Publication dans le RenderBlackboard (3e argument = nom du producteur)
        ctx.blackboard.PublishTexture(env::bb::AtmoTransmittanceLUT,   transmittanceLUT_,   kProducer);
        ctx.blackboard.PublishTexture(env::bb::AtmoMultiScatteringLUT, multiScatteringLUT_, kProducer);
        ctx.blackboard.PublishTexture(env::bb::AtmoSkyViewLUT,         skyViewLUT_,         kProducer);

        atmoParamsDirty_ = true;
    }

    void Update(EngineContext& ctx) override {
        // Entrées  : temps de la frame ; paramètres du panneau (heure, jour, latitude, astres, vent,
        //            météo).
        // Sorties  : champs du GlobalEnvironmentUBO dont ce système est l'unique écrivain :
        //            sunDirIlluminance (direction vers le soleil + éclairement en lux), sunColorAngular,
        //            moonDirPhase (direction vers la lune + phase dans [0, 1), 0 = nouvelle, 0,5 = pleine),
        //            moonParams (éclairement, rayon angulaire, lumière cendrée), windDirSpeed, windGust,
        //            weather (pluie, neige, tempête de sable, humidité), weatherExtra (couverture et densité
        //            nuageuses, brouillard, humidité de surface).
        const float dt = ctx.time.dt;

        if (autoAdvance_) {
            timeOfDayHours_ += (dt * timeScale_) / 3600.0f;
            while (timeOfDayHours_ >= 24.0f) {
                timeOfDayHours_ -= 24.0f;
                dayOfYear_ = (dayOfYear_ % 365) + 1;
            }
        }

        if (useAstronomicalModel_) {
            // Déclinaison solaire (formule de Cooper)
            const float gamma = 2.0f * glm::pi<float>() * (static_cast<float>(dayOfYear_) + 10.0f) / 365.0f;
            const float declRad = glm::radians(-23.44f) * std::cos(gamma);
            const float latRad  = glm::radians(latitudeDeg_);

            // Angle horaire solaire (0 rad à midi solaire = 12h00)
            const float hourAngleRad = glm::radians(15.0f * (timeOfDayHours_ - 12.0f));

            const float sinEl = std::sin(latRad) * std::sin(declRad)
                              + std::cos(latRad) * std::cos(declRad) * std::cos(hourAngleRad);

            // Repère : Nord = -Z, Est = +X, Sud = +Z, Ouest = -X
            const float dirX = -std::cos(declRad) * std::sin(hourAngleRad);
            const float dirZ =  std::sin(latRad) * std::cos(declRad) * std::cos(hourAngleRad)
                              - std::cos(latRad) * std::sin(declRad);
            const glm::vec3 sunDir = glm::normalize(glm::vec3(dirX, sinEl, dirZ));
            DirToAzimuthElevation(sunDir, sunAzimuth_, sunElevation_);

            // Cycle lunaire synodique (29.53 jours). lunarAgeOffsetDays_ = âge de la lune au jour 0
            // (époque arbitraire) : sans lui, le jour 172 tombait sur un dernier croissant, levé avec le
            // soleil et donc invisible la nuit.
            moonPhase_ = std::fmod((static_cast<float>(dayOfYear_) + timeOfDayHours_ / 24.0f + lunarAgeOffsetDays_)
                                       / 29.53059f, 1.0f);
            const float moonHourAngle = hourAngleRad - moonPhase_ * 2.0f * glm::pi<float>();
            const float moonDecl      = declRad * std::cos(moonPhase_ * 2.0f * glm::pi<float>())
                                      + glm::radians(5.14f) * std::sin(moonPhase_ * 2.0f * glm::pi<float>());

            const float mSinEl = std::sin(latRad) * std::sin(moonDecl)
                               + std::cos(latRad) * std::cos(moonDecl) * std::cos(moonHourAngle);
            const float mDirX  = -std::cos(moonDecl) * std::sin(moonHourAngle);
            const float mDirZ  =  std::sin(latRad) * std::cos(moonDecl) * std::cos(moonHourAngle)
                               - std::cos(latRad) * std::sin(moonDecl);
            const glm::vec3 moonDir = glm::normalize(glm::vec3(mDirX, mSinEl, mDirZ));
            DirToAzimuthElevation(moonDir, moonAzimuth_, moonElevation_);
        }

        // Machine à états Météo : accumulation et séchage de l'humidité au sol, en deux phases.
        // Sous kPuddleThreshold (même seuil que terrain_wetness.glsl) : film d'eau en surface, qui se
        // forme et sèche vite. Au-dessus : eau stockée dans les flaques, qui se remplissent
        // progressivement et s'évaporent lentement (elles survivent au sol déjà sec en surface).
        constexpr float kPuddleThreshold = 0.35f;
        const bool      puddlePhase      = surfaceWetness_ > kPuddleThreshold;
        if (rain_ > 0.01f) {
            surfaceWetness_ = std::min(1.0f, surfaceWetness_ + rain_ * (puddlePhase ? 0.04f : 0.15f) * dt);
        } else {
            const float sunHeat = std::max(0.0f, std::sin(glm::radians(sunElevation_)));
            const float dryRate = (0.01f + 0.02f * sunHeat + 0.002f * windSpeed_) * (1.0f - humidity_ * 0.7f);
            surfaceWetness_ = std::max(0.0f, surfaceWetness_ - dryRate * (puddlePhase ? 0.5f : 1.5f) * dt);
        }

        const float effectiveCoverage = std::max(cloudCoverage_, std::max(rain_, snow_) * 0.85f);

        // Écriture dans ctx.environment
        const glm::vec3 sunDir  = AzimuthElevationToDir(sunAzimuth_,  sunElevation_);
        const glm::vec3 moonDir = AzimuthElevationToDir(moonAzimuth_, moonElevation_);

        // Azimut vent (0 = -Z, 90 = +X) -> vecteur 2D horizontal (X, Z) unitaire
        const float windRad = glm::radians(windDirectionDeg_);
        const glm::vec2 windDirXZ(std::sin(windRad), -std::cos(windRad));

        // Phase lunaire : 0 = nouvelle lune (facteur 0), 0.5 = pleine lune (facteur 1)
        const float phaseFactor = 0.5f * (1.0f - std::cos(moonPhase_ * 2.0f * glm::pi<float>()));
        const float effectiveMoonLux = moonIlluminance_ * phaseFactor;

        // (variable nommée "ubo" et non "env" : "env" masquerait le namespace env::)
        GlobalEnvironmentUBO& ubo = ctx.environment;
        ubo.sunDirIlluminance = glm::vec4(sunDir, sunIlluminance_);
        ubo.sunColorAngular   = glm::vec4(sunColor_[0], sunColor_[1], sunColor_[2], glm::radians(sunAngularRadiusDeg_));
        ubo.moonDirPhase      = glm::vec4(moonDir, moonPhase_);
        ubo.moonParams        = glm::vec4(effectiveMoonLux, glm::radians(moonAngularRadiusDeg_), earthshine_, 0.0f);
        // Layout exact de WorldContracts.hpp : xy = dir horizontale (X, Z), z = vitesse, w = turbulence
        ubo.windDirSpeed      = glm::vec4(windDirXZ.x, windDirXZ.y, windSpeed_, windTurbulence_);
        // Layout exact de WorldContracts.hpp : x = fréquence rafales, y = force rafales, zw = 0
        ubo.windGust          = glm::vec4(gustFrequency_, gustStrength_, 0.0f, 0.0f);
        ubo.weather           = glm::vec4(rain_, snow_, sandstorm_, humidity_);
        ubo.weatherExtra      = glm::vec4(effectiveCoverage, cloudDensity_, fogDensity_, surfaceWetness_);
    }

    void Render(EngineContext&) override {
        // Entrées  : paramètres atmosphériques du panneau, directions du soleil et de la lune.
        // Sorties  : LUTs publiées sous atmo.transmittanceLUT, atmo.multiScatteringLUT et atmo.skyViewLUT.
        // Mise à jour de l'UBO atmosphérique (avec turbidité liée à la météo)
        // Turbidité = multiplicateur de la densité d'aérosols (Mie). 1 = atmosphère claire de référence
        // (Hillaire 2020). Le brouillard au sol (fogDensity_) n'en fait PAS partie : c'est un brouillard
        // de hauteur local, appliqué par la passe d'éclairage, pas une propriété de toute l'atmosphère.
        const float weatherTurbidity = mieTurbidity_ * (1.0f + rain_ * 2.0f + sandstorm_ * 8.0f);

        AtmosphereParamsUBO params{};
        params.radiiAndScaleHeights = glm::vec4(
            planetRadiusKm_,
            planetRadiusKm_ + atmosphereHeightKm_,
            rayleighScaleHeightKm_,
            mieScaleHeightKm_
        );
        // Coefficients standards Terre (Hillaire 2020) en km^-1
        params.rayleighScattering  = glm::vec4(0.005802f, 0.013558f, 0.033100f, mieAnisotropy_);
        params.mieScatteringAndExt = glm::vec4(0.003996f * weatherTurbidity,
                                               0.003996f * weatherTurbidity,
                                               0.003996f * weatherTurbidity,
                                               0.004440f * weatherTurbidity);
        params.ozoneAbsorption     = glm::vec4(0.000650f, 0.001881f, 0.000085f, ozone_ ? 1.0f : 0.0f);
        params.groundAlbedoAndTurb = glm::vec4(groundAlbedo_[0], groundAlbedo_[1], groundAlbedo_[2], weatherTurbidity);
        // Intensité du fond de ciel nocturne ; la teinte (dégradé bleu nuit) est dans NightSkyColor (GLSL).
        params.nightSky = glm::vec4(nightSkyIntensity_, 0.0f, 0.0f, 0.0f);

        atmoUbo_.UploadObject(params);
        atmoUbo_.BindBase(GL_UNIFORM_BUFFER, env::bind::ubo::AtmosphereParams);

        // Un rechargement de shader (F5) incrémente Generation() : il faut alors recalculer les LUTs statiques.
        const std::uint32_t transGen = transmittanceProg_->Generation();
        const std::uint32_t msGen    = multiScatteringProg_->Generation();
        if (transGen != lastTransmittanceGen_ || msGen != lastMultiScatteringGen_) {
            atmoParamsDirty_ = true;
        }

        // Passe 1 & 2 : Transmittance (256x64) et Multi-Scattering (32x32)
        // Recalculées si un paramètre d'atmosphère ou la turbidité météo a changé
        if (atmoParamsDirty_ || std::abs(weatherTurbidity - lastTurbidity_) > 1e-3f) {
            if (transmittanceProg_->IsValid()) {
                transmittanceLUT_.BindImage(env::bind::image::AtmoLutOut, GL_WRITE_ONLY);
                transmittanceProg_->Dispatch(gpu::ShaderProgram::GroupCount(256, 8), gpu::ShaderProgram::GroupCount(64, 8));
                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
            }

            if (multiScatteringProg_->IsValid()) {
                transmittanceLUT_.BindUnit(env::bind::tex::AtmoTransmittance);
                multiScatteringLUT_.BindImage(env::bind::image::AtmoLutOut, GL_WRITE_ONLY);
                multiScatteringProg_->Dispatch(gpu::ShaderProgram::GroupCount(32, 8), gpu::ShaderProgram::GroupCount(32, 8));
                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
            }

            lastTurbidity_          = weatherTurbidity;
            lastTransmittanceGen_   = transGen;
            lastMultiScatteringGen_ = msGen;
            atmoParamsDirty_        = false;
        }

        // Passe 3 : Sky-View LUT (192x108), recalculée chaque frame
        if (skyViewProg_->IsValid()) {
            transmittanceLUT_.BindUnit(env::bind::tex::AtmoTransmittance);
            multiScatteringLUT_.BindUnit(env::bind::tex::AtmoMultiScattering);
            skyViewLUT_.BindImage(env::bind::image::AtmoLutOut, GL_WRITE_ONLY);
            // 192 / 8 = 24, 108 / 8 = 13.5 -> 14 groupes en Y
            skyViewProg_->Dispatch(gpu::ShaderProgram::GroupCount(192, 8), gpu::ShaderProgram::GroupCount(108, 8));
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
        }
    }

    void Shutdown(EngineContext& ctx) override {
        // Entrées  : ressources créées par Init.
        // Sorties  : clés atmo.* retirées, LUTs et UBO libérés.
        ctx.blackboard.UnpublishIfProducer(env::bb::AtmoTransmittanceLUT,   kProducer);
        ctx.blackboard.UnpublishIfProducer(env::bb::AtmoMultiScatteringLUT, kProducer);
        ctx.blackboard.UnpublishIfProducer(env::bb::AtmoSkyViewLUT,         kProducer);
        transmittanceLUT_.Destroy();
        multiScatteringLUT_.Destroy();
        skyViewLUT_.Destroy();
        atmoUbo_.Destroy();
    }

    void OnGui(EngineContext&) override {
        TodoBanner("modele astronomique, phases lunaires, meteo, LUTs d'atmosphere");

        ImGui::SeparatorText("Temps");
        ImGui::Checkbox("Modele astronomique", &useAstronomicalModel_);
        ImGui::SliderFloat("Heure", &timeOfDayHours_, 0.0f, 24.0f, "%.2f h");
        ImGui::SliderInt("Jour de l'annee", &dayOfYear_, 1, 365);
        ImGui::SliderFloat("Latitude", &latitudeDeg_, -90.0f, 90.0f, "%.1f deg");
        ImGui::Checkbox("Avance automatique", &autoAdvance_);
        ImGui::SliderFloat("x temps reel", &timeScale_, 1.0f, 3600.0f, "%.0f", ImGuiSliderFlags_Logarithmic);

        ImGui::SeparatorText("Soleil");
        ImGui::SliderFloat("Azimut soleil", &sunAzimuth_, 0.0f, 360.0f, "%.1f deg");
        ImGui::SliderFloat("Elevation soleil", &sunElevation_, -20.0f, 90.0f, "%.1f deg");
        ImGui::SliderFloat("Eclairement soleil", &sunIlluminance_, 0.0f, 130000.0f, "%.0f lux", ImGuiSliderFlags_Logarithmic);
        ImGui::ColorEdit3("Couleur soleil", sunColor_);
        ImGui::SliderFloat("Rayon angulaire soleil", &sunAngularRadiusDeg_, 0.05f, 5.0f, "%.3f deg");

        ImGui::SeparatorText("Lune");
        ImGui::SliderFloat("Azimut lune", &moonAzimuth_, 0.0f, 360.0f, "%.1f deg");
        ImGui::SliderFloat("Elevation lune", &moonElevation_, -20.0f, 90.0f, "%.1f deg");
        ImGui::SliderFloat("Phase", &moonPhase_, 0.0f, 0.999f, "%.3f");
        ImGui::SameLine();
        ImGui::TextDisabled("%s", MoonPhaseName(moonPhase_));
        ImGui::SliderFloat("Age de la lune au jour 0", &lunarAgeOffsetDays_, 0.0f, 29.53f, "%.1f j");
        ImGui::SetItemTooltip("Modele astronomique : fixe le cycle lunaire (29,53 j).\n"
                              "Pleine lune (phase 0,5) = visible toute la nuit ;\n"
                              "proche de la nouvelle lune = levee avec le soleil, visible de jour.");
        ImGui::SliderFloat("Eclairement lune", &moonIlluminance_, 0.0f, 1.0f, "%.3f lux");
        ImGui::SliderFloat("Rayon angulaire lune", &moonAngularRadiusDeg_, 0.05f, 5.0f, "%.3f deg");
        ImGui::SliderFloat("Lumiere cendree", &earthshine_, 0.0f, 0.2f, "%.3f");

        ImGui::SeparatorText("Vent");
        ImGui::SliderFloat("Direction du vent", &windDirectionDeg_, 0.0f, 360.0f, "%.0f deg");
        ImGui::SliderFloat("Vitesse", &windSpeed_, 0.0f, 40.0f, "%.1f m/s");
        ImGui::SliderFloat("Turbulence", &windTurbulence_, 0.0f, 1.0f);
        ImGui::SliderFloat("Frequence rafales", &gustFrequency_, 0.0f, 2.0f, "%.2f Hz");
        ImGui::SliderFloat("Force rafales", &gustStrength_, 0.0f, 1.0f);

        ImGui::SeparatorText("Meteo");
        ImGui::SliderFloat("Pluie", &rain_, 0.0f, 1.0f);
        ImGui::SliderFloat("Neige", &snow_, 0.0f, 1.0f);
        ImGui::SliderFloat("Tempete de sable", &sandstorm_, 0.0f, 1.0f);
        ImGui::SliderFloat("Humidite", &humidity_, 0.0f, 1.0f);
        ImGui::SliderFloat("Couverture nuageuse", &cloudCoverage_, 0.0f, 1.0f);
        ImGui::SliderFloat("Densite nuageuse", &cloudDensity_, 0.0f, 1.0f);
        ImGui::SliderFloat("Brouillard", &fogDensity_, 0.0f, 0.01f, "%.5f /m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Humidite de surface", &surfaceWetness_, 0.0f, 1.0f);

        // Tout changement ici invalide les LUTs de transmittance / multi-diffusion.
        ImGui::SeparatorText("Atmosphere");
        atmoParamsDirty_ |= ImGui::SliderFloat("Rayon planete", &planetRadiusKm_, 1000.0f, 10000.0f, "%.0f km");
        atmoParamsDirty_ |= ImGui::SliderFloat("Epaisseur atmosphere", &atmosphereHeightKm_, 10.0f, 200.0f, "%.0f km");
        atmoParamsDirty_ |= ImGui::SliderFloat("Hauteur d'echelle Rayleigh", &rayleighScaleHeightKm_, 1.0f, 20.0f, "%.1f km");
        atmoParamsDirty_ |= ImGui::SliderFloat("Hauteur d'echelle Mie", &mieScaleHeightKm_, 0.1f, 5.0f, "%.2f km");
        ImGui::SliderFloat("Turbidite (aerosols Mie)", &mieTurbidity_, 0.1f, 10.0f, "x%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Quantite de particules (poussiere, gouttelettes, pollution).\n"
                              "1 = air clair de reference. Plus haut = ciel plus blanc, halo solaire plus large.");
        ImGui::SliderFloat("Anisotropie Mie (g)", &mieAnisotropy_, 0.0f, 0.99f);
        atmoParamsDirty_ |= ImGui::Checkbox("Ozone", &ozone_);
        atmoParamsDirty_ |= ImGui::ColorEdit3("Albedo du sol", groundAlbedo_);
        ImGui::SliderFloat("Ciel nocturne", &nightSkyIntensity_, 0.0f, 0.3f, "%.4f", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Fond de ciel : etoiles + airglow (+ pollution lumineuse), degrade bleu nuit.\n"
                              "Luminance au zenith = 0,176 x intensite (cd/m2).\n"
                              "Site tres sombre ~0,001 ; peri-urbain ~0,02 ; ville ~0,1.");
    }

private:
    // Ressources GPU (possédées par ce système, publiées dans le blackboard).
    gpu::ShaderProgram* transmittanceProg_   = nullptr;
    gpu::ShaderProgram* multiScatteringProg_ = nullptr;
    gpu::ShaderProgram* skyViewProg_         = nullptr;
    gpu::Texture2D      transmittanceLUT_;
    gpu::Texture2D      multiScatteringLUT_;
    gpu::Texture2D      skyViewLUT_;
    gpu::GpuBuffer      atmoUbo_;
    bool                atmoParamsDirty_        = true;
    float               lastTurbidity_          = -1.0f;
    std::uint32_t       lastTransmittanceGen_   = 0;
    std::uint32_t       lastMultiScatteringGen_ = 0;

    // Variables liées aux sliders d'OnGui (aucune logique branchée).
    bool  useAstronomicalModel_  = false;
    float timeOfDayHours_        = 14.0f;
    int   dayOfYear_             = 172;
    float latitudeDeg_           = 45.0f;
    bool  autoAdvance_           = false;
    float timeScale_             = 60.0f;
    float sunAzimuth_            = 52.0f;
    float sunElevation_          = 47.0f;
    float sunIlluminance_        = 100000.0f;
    float sunColor_[3]           = {1.0f, 0.96f, 0.90f};
    float sunAngularRadiusDeg_   = 0.2665f;
    float moonAzimuth_           = 230.0f;
    float moonElevation_         = 35.0f;
    float moonPhase_             = 0.5f;
    float moonIlluminance_       = 0.25f;
    float moonAngularRadiusDeg_  = 0.259f;
    float earthshine_            = 0.02f;
    float lunarAgeOffsetDays_    = 19.3f;    // jour 172 (défaut) ≈ pleine lune
    float nightSkyIntensity_     = 0.02f;    // ciel péri-urbain : zénith ≈ 0,0035 cd/m²
    float windDirectionDeg_      = 106.7f;
    float windSpeed_             = 4.0f;
    float windTurbulence_        = 0.3f;
    float gustFrequency_         = 0.2f;
    float gustStrength_          = 0.35f;
    float rain_                  = 0.0f;
    float snow_                  = 0.0f;
    float sandstorm_             = 0.0f;
    float humidity_              = 0.5f;
    float cloudCoverage_         = 0.35f;
    float cloudDensity_          = 0.5f;
    float fogDensity_            = 0.0006f;
    float surfaceWetness_        = 0.0f;
    float planetRadiusKm_        = 6360.0f;
    float atmosphereHeightKm_    = 100.0f;
    float rayleighScaleHeightKm_ = 8.0f;
    float mieScaleHeightKm_      = 1.2f;
    float mieAnisotropy_         = 0.8f;
    float mieTurbidity_          = 1.0f;
    bool  ozone_                 = true;
    float groundAlbedo_[3]       = {0.3f, 0.3f, 0.3f};
};

}  // namespace
REGISTER_SYSTEM_ORDERED(CelestialAtmosphereSystem, PassStage::AtmosphereIBL, 0);
}  // namespace env::render
