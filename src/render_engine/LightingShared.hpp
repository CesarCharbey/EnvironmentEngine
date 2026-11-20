#pragma once
// =============================================================================
//  src/render_engine/LightingShared.hpp                             [DEV 2]
//
//  Réglages et état d'éclairage partagés entre les passes du Render Engine qui
//  éclairent des surfaces avec la même physique (shaders/render/surface_lighting.glsl) :
//    * DeferredLightingSystem : propriétaire des réglages (son panneau) et de l'état de la frame ;
//    * WaterOceanRiverSystem  : éclaire l'eau en forward avec exactement les mêmes paramètres ;
//    * VolumetricGodRaysSystem : exposition + tonemap vers scene.display (réglages d'exposition).
//
//  Interne au territoire Dev 2 (aucun contrat) : jamais inclus hors de src/render_engine/.
// =============================================================================
#include <cmath>
#include <cstdint>
#include <string_view>

#include <glad/glad.h>
#include <glm/glm.hpp>

#include "contracts/GpuBindings.hpp"
#include "contracts/WorldContracts.hpp"
#include "core/RenderBlackboard.hpp"
#include "core/gpu/ShaderManager.hpp"

namespace env::render {

/// Producteur de scene.display (passe de tonemap).
inline constexpr std::string_view kTonemapProducer = "VolumetricGodRaysSystem";

/// Paramètres des nuages tels qu'ils ont servi à la frame de calcul de la carte météo (VolumetricGodRaysSystem),
/// relus par la capture de l'IBL (DynamicIBLSystem) à la frame suivante : uniforms de cloud_density.glsl et
/// cloud_march.glsl. Les positions sont relatives à `originChunk` (recalées par SetCloudUniforms).
struct CloudParams {
    bool       valid = false;            ///< nuages actifs, bruits et carte météo publiés
    float      bottom = 1500.0f, top = 4500.0f;
    float      shapeScale = 9000.0f, detailScale = 900.0f, detailStrength = 0.35f, curl = 0.35f;
    glm::vec3  shapeOffset{0.0f}, detailOffset{0.0f};
    glm::vec2  weatherOriginRel{0.0f};
    float      weatherExtent = 40000.0f;
    glm::vec2  skew{0.0f};
    float      extinction = 0.04f, hgForward = 0.6f, hgBackward = -0.3f, silver = 1.5f, powder = 2.0f;
    float      maxDistance = 60000.0f, fadeDistance = 45000.0f;
    glm::ivec2 originChunk{0};
};

struct LightingShared {
    // --- Réglages (panneau de DeferredLightingSystem) ---
    float sunScale               = 1.0f;
    float moonScale              = 1.0f;
    float iblScale               = 1.0f;
    bool  specularOcclusion      = true;
    bool  terrainShadows         = true;
    bool  contactShadows         = true;
    float shadowPenumbraDeg      = 1.0f;
    bool  voxelShadows           = true;
    bool  voxelGI                = true;
    float giIntensity            = 1.0f;
    float voxelShadowSoftnessDeg = 2.3f;     // tan ≈ 0,04
    float fogHeightFalloff       = 0.005f;   // hauteur d'échelle 200 m
    bool  fogShadows             = true;     // ombre volumétrique du brouillard (VXGI)

    // --- Ombre des nuages : écrit par VolumetricGodRaysSystem avec la carte (lue à la frame suivante) ---
    bool       cloudShadowValid    = false;            ///< carte calculée (nuages et ombres actifs)
    float      cloudLayerAltitude  = 1500.0f;          ///< m : base de la couche (plan de projection)
    float      cloudShadowExtent   = 20000.0f;         ///< m
    glm::vec2  cloudShadowOrigin{0.0f};                ///< m : centre, relatif à cloudShadowOriginChunk
    glm::ivec2 cloudShadowOriginChunk{0};              ///< origine de rendu de la frame de calcul
    glm::vec3  cloudShadowLight{0.0f, 1.0f, 0.0f};     ///< astre pour lequel la carte a été calculée

    // --- God rays : distance de marche (0 = inactifs), écrite par VolumetricGodRaysSystem (Update) ---
    float godRayDistance = 0.0f;

    // --- Nuages : écrit par VolumetricGodRaysSystem avec la carte météo, lu par la capture de l'IBL ---
    CloudParams clouds;

    // --- Réglages d'exposition (panneau de VolumetricGodRaysSystem) ---
    float exposureCompensationEv = 1.0f;
    float nightAdaptation        = 0.6f;
    bool  moonlightShift         = true;

    // --- État de la frame (DeferredLightingSystem::Update / Render) ---
    bool      gbufferAvailable     = false;
    bool      atmoAvailable        = false;
    bool      iblAvailable         = false;
    bool      cloudShadowAvailable = false;
    bool      voxelAvailable       = false;
    bool      focusAvailable       = false;
    glm::vec3 focusChunkOrigin{0.0f};
    bool      dataView             = false;  ///< vue de debug « données brutes » : pas d'exposition

    // --- Synchronisation éclairage HDR <-> tonemap (index de frame) ---
    /// Dernière frame où la passe de tonemap a tourné (son Render).
    std::uint64_t tonemapFrame = ~std::uint64_t{0};
    /// Dernière frame où l'éclairage a écrit scene.color en HDR (sinon : repli LDR encodé).
    std::uint64_t hdrFrame     = ~std::uint64_t{0};
};

/// Instance unique, partagée par les systèmes du Render Engine.
inline LightingShared& SharedLighting() {
    static LightingShared shared;
    return shared;
}

/// true si l'éclairage a écrit scene.color en HDR linéaire pendant la frame `frame`.
inline bool SceneColorIsHdr(std::uint64_t frame) { return SharedLighting().hdrFrame == frame; }

/// Lie les ressources lues par surface_lighting.glsl (blackboard -> slots du contrat).
inline void BindSurfaceLightingResources(const core::RenderBlackboard& bb) {
    namespace tex = env::bind::tex;
    bb.BindTexture(env::bb::GBufferDepth,           tex::GBufferDepth);
    bb.BindTexture(env::bb::AtmoTransmittanceLUT,   tex::AtmoTransmittance);
    bb.BindTexture(env::bb::AtmoMultiScatteringLUT, tex::AtmoMultiScattering);
    bb.BindTexture(env::bb::AtmoSkyViewLUT,         tex::AtmoSkyView);
    bb.BindTexture(env::bb::IblIrradiance,          tex::IblIrradiance);
    bb.BindTexture(env::bb::IblPrefiltered,         tex::IblPrefiltered);
    bb.BindTexture(env::bb::IblBrdfLUT,             tex::IblBrdfLut);
    bb.BindTexture(env::bb::CloudShadowMap,         tex::CloudShadow);
    // Lecture seule de la carte de hauteur du chunk focus (slot Dev 1 existant) : ombres du relief.
    bb.BindTexture(env::bb::WorldFocusHeightNormal, tex::HeightNormal);
    // VXGI : grille publiée ce frame ; son placement est dans UBO_VoxelParams (lié par le producteur).
    bb.BindTexture(env::bb::VoxelGrid,              tex::VoxelGrid);
}

/// Renseigne les uniforms de surface_lighting.glsl (et uAtmoAvailable de sky_lighting.glsl).
/// renderOriginChunk : origine de rendu de la frame courante (recalage de l'origine de l'ombre des nuages).
inline void SetSurfaceLightingUniforms(const gpu::ShaderProgram& p, glm::ivec2 renderOriginChunk) {
    const LightingShared& s = SharedLighting();
    p.Set("uAtmoAvailable",        s.atmoAvailable);
    p.Set("uGBufferAvailable",     s.gbufferAvailable);
    p.Set("uIblAvailable",         s.iblAvailable);
    p.Set("uCloudShadowAvailable", s.cloudShadowAvailable);
    p.Set("uSpecularOcclusion",    s.specularOcclusion);
    p.Set("uSunScale",             s.sunScale);
    p.Set("uMoonScale",            s.moonScale);
    p.Set("uIblScale",             s.iblScale);
    p.Set("uVoxelAvailable",       s.voxelAvailable);
    p.Set("uVoxelShadows",         s.voxelShadows);
    p.Set("uVoxelGI",              s.voxelGI);
    p.Set("uGiIntensity",          s.giIntensity);
    p.Set("uVoxelShadowSoftness",  std::tan(glm::radians(s.voxelShadowSoftnessDeg)));
    p.Set("uTerrainShadows",       s.terrainShadows);
    p.Set("uContactShadows",       s.contactShadows);
    p.Set("uFocusAvailable",       s.focusAvailable);
    p.Set("uFocusChunkOrigin",     s.focusChunkOrigin);
    p.Set("uShadowPenumbra",       glm::radians(s.shadowPenumbraDeg));
    p.Set("uFogHeightFalloff",     s.fogHeightFalloff);
    p.Set("uFogShadows",           s.fogShadows);
    p.Set("uGodRayDistance",       s.godRayDistance);
    p.Set("uCloudLayerAltitude",   s.cloudLayerAltitude);
    p.Set("uCloudShadowExtent",    s.cloudShadowExtent);
    // Origine exprimée dans le repère de rendu de la frame courante (exact : différence entière de chunks).
    const glm::vec2 shift = glm::vec2(s.cloudShadowOriginChunk - renderOriginChunk) * kChunkSizeMeters;
    p.Set("uCloudShadowOrigin",    s.cloudShadowOrigin + shift);
    p.Set("uCloudShadowLight",     s.cloudShadowLight);
}

/// Uniforms de cloud_density.glsl et cloud_march.glsl. renderOriginChunk : origine de rendu de la frame qui
/// dessine ; les positions de `c` (relatives à c.originChunk) y sont recalées exactement (différence entière).
inline void SetCloudUniforms(const gpu::ShaderProgram& p, const CloudParams& c, glm::ivec2 renderOriginChunk) {
    // Position relative à l'origine de calcul = position relative à l'origine courante - shift.
    const glm::vec2 shift = glm::vec2(c.originChunk - renderOriginChunk) * kChunkSizeMeters;
    p.Set("uCloudBottom",      c.bottom);
    p.Set("uCloudTop",         c.top);
    p.Set("uShapeScale",       c.shapeScale);
    p.Set("uDetailScale",      c.detailScale);
    p.Set("uDetailStrength",   c.detailStrength);
    p.Set("uShapeOffset",      c.shapeOffset - glm::vec3(shift.x, 0.0f, shift.y));
    p.Set("uDetailOffset",     c.detailOffset - glm::vec3(shift.x, 0.0f, shift.y));
    p.Set("uWeatherOriginRel", c.weatherOriginRel + shift);
    p.Set("uWeatherExtent",    c.weatherExtent);
    p.Set("uSkew",             c.skew);
    p.Set("uCurlStrength",     c.curl);
    p.Set("uExtinction",       c.extinction);
    p.Set("uHgForward",        c.hgForward);
    p.Set("uHgBackward",       c.hgBackward);
    p.Set("uSilver",           c.silver);
    p.Set("uPowder",           c.powder);
    p.Set("uMaxDistance",      c.maxDistance);
    p.Set("uFadeDistance",     c.fadeDistance);
}

/// Renseigne les uniforms d'exposure.glsl.
inline void SetExposureUniforms(const gpu::ShaderProgram& p) {
    const LightingShared& s = SharedLighting();
    p.Set("uExposureCompensation", s.exposureCompensationEv);
    p.Set("uNightAdaptation",      s.nightAdaptation);
    p.Set("uMoonlightShift",       s.moonlightShift);
}

}  // namespace env::render
