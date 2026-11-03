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

#include "contracts/ISystemModule.hpp"
#include "core/SystemRegistry.hpp"

namespace env::render {
namespace {

void TodoBanner(const char* scope) {
    ImGui::TextColored(ImVec4(0.35f, 0.80f, 1.0f, 1.0f), "TODO Dev 2");
    ImGui::SameLine();
    ImGui::TextDisabled("%s", scope);
}

class CelestialAtmosphereSystem final : public ISystemModule {
public:
    void Init(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : shaders compute à créer dans shaders/render/ ; slots du contrat UBO_AtmosphereParams,
        //            IMG_AtmoLutOut.
        // Sorties  : LUT de transmittance (RGBA16F 256 x 64), LUT de multi-diffusion (RGBA16F 32 x 32),
        //            LUT sky-view (RGBA16F 192 x 108), programmes compute, UBO de paramètres atmosphériques.
        // Objectif : préparer les ressources de l'atmosphère.
    }

    void Update(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : temps de la frame ; paramètres du panneau (heure, jour, latitude, astres, vent,
        //            météo).
        // Sorties  : champs du GlobalEnvironmentUBO dont ce système est l'unique écrivain :
        //            sunDirIlluminance (direction vers le soleil + éclairement en lux), sunColorAngular,
        //            moonDirPhase (direction vers la lune + phase dans [0, 1), 0 = nouvelle, 0,5 = pleine),
        //            moonParams (éclairement, rayon angulaire, lumière cendrée), windDirSpeed, windGust,
        //            weather (pluie, neige, tempête de sable, humidité), weatherExtra (couverture et densité
        //            nuageuses, brouillard, humidité de surface).
        // Objectif : modèle astronomique (position du soleil et de la lune selon l'heure, le jour et la
        //            latitude, cycle synodique d'environ 29,53 jours) ; conversion azimut / élévation en
        //            direction selon la convention du contrat (azimut 0 = -Z, 90 = +X, Y vers le haut) ;
        //            machine à états météo (transitions, accumulation puis séchage de l'humidité de surface).
    }

    void Render(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : paramètres atmosphériques du panneau, directions du soleil et de la lune.
        // Sorties  : LUTs publiées sous atmo.transmittanceLUT, atmo.multiScatteringLUT et atmo.skyViewLUT.
        // Objectif : transmittance et multi-diffusion recalculées seulement quand les paramètres
        //            atmosphériques changent ; sky-view recalculée chaque frame ; barrière mémoire avant
        //            toute lecture par les passes suivantes (IBL, éclairage, nuages).
    }

    void Shutdown(EngineContext&) override {
        // [DEV 2 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : ressources créées par Init.
        // Sorties  : clés atmo.* retirées, LUTs et UBO libérés.
        // Objectif : libération propre avant la destruction du contexte OpenGL.
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

        ImGui::SeparatorText("Atmosphere");
        ImGui::SliderFloat("Rayon planete", &planetRadiusKm_, 1000.0f, 10000.0f, "%.0f km");
        ImGui::SliderFloat("Epaisseur atmosphere", &atmosphereHeightKm_, 10.0f, 200.0f, "%.0f km");
        ImGui::SliderFloat("Hauteur d'echelle Rayleigh", &rayleighScaleHeightKm_, 1.0f, 20.0f, "%.1f km");
        ImGui::SliderFloat("Hauteur d'echelle Mie", &mieScaleHeightKm_, 0.1f, 5.0f, "%.2f km");
        ImGui::SliderFloat("Anisotropie Mie (g)", &mieAnisotropy_, 0.0f, 0.99f);
        ImGui::Checkbox("Ozone", &ozone_);
        ImGui::ColorEdit3("Albedo du sol", groundAlbedo_);
    }

private:
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
    bool  ozone_                 = true;
    float groundAlbedo_[3]       = {0.3f, 0.3f, 0.3f};
};

}  // namespace
REGISTER_SYSTEM_ORDERED(CelestialAtmosphereSystem, PassStage::AtmosphereIBL, 0);
}  // namespace env::render
