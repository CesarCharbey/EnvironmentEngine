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
#include "core/gpu/GpuBuffer.hpp"
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
    // Cirrus 2.5D (Nubis Evolved 2022)
    float      cirrusAltitude = 7500.0f;
    float      cirrusCoverage = 0.35f;
    float      cirrusType     = 0.5f;    ///< 0 = streaky, 0.5 = wispy, 1 = round (cirrocumulus)
    float      cirrusScale    = 24000.0f;
    glm::vec2  cirrusOffset{0.0f};
    // Éclairs internes (Nubis Evolved 2022)
    glm::vec3  lightningPosRel{0.0f};
    float      lightningIntensity = 0.0f;
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

/// Miroir std140 de SurfaceLightingBlock (shaders/render/surface_lighting.glsl), UBO_PostParams.
struct alignas(16) SurfaceLightingUbo {
    glm::ivec4 flags0{0};            ///< x=gbufferAvailable, y=iblAvailable, z=cloudShadowAvailable, w=specularOcclusion
    glm::ivec4 flags1{0};            ///< x=voxelAvailable, y=voxelShadows, z=voxelGI, w=terrainShadows
    glm::ivec4 flags2{0};            ///< x=contactShadows, y=focusAvailable, z=fogShadows, w=atmoAvailable
    glm::vec4  scales{0.0f};         ///< x=sunScale, y=moonScale, z=iblScale, w=giIntensity
    glm::vec4  shadowParams{0.0f};   ///< x=voxelShadowSoftness, y=shadowPenumbra, z=fogHeightFalloff, w=godRayDistance
    glm::vec4  focusChunkOrigin{0.0f};  ///< xyz=focusChunkOrigin, w=0
    glm::vec4  cloudShadow0{0.0f};   ///< xy=cloudShadowOrigin + shift, z=cloudLayerAltitude, w=cloudShadowExtent
    glm::vec4  cloudShadow1{0.0f};   ///< xyz=cloudShadowLight, w=0
};
static_assert(sizeof(SurfaceLightingUbo) == 8 * 16, "SurfaceLightingUbo: layout std140");

/// Construit l'UBO d'éclairage de surface pour la frame courante.
inline SurfaceLightingUbo BuildSurfaceLightingUbo(glm::ivec2 renderOriginChunk) {
    const LightingShared& s     = SharedLighting();
    const glm::vec2       shift = glm::vec2(s.cloudShadowOriginChunk - renderOriginChunk) * kChunkSizeMeters;
    SurfaceLightingUbo    u;
    u.flags0 = {s.gbufferAvailable ? 1 : 0, s.iblAvailable ? 1 : 0, s.cloudShadowAvailable ? 1 : 0,
                s.specularOcclusion ? 1 : 0};
    u.flags1 = {s.voxelAvailable ? 1 : 0, s.voxelShadows ? 1 : 0, s.voxelGI ? 1 : 0, s.terrainShadows ? 1 : 0};
    u.flags2 = {s.contactShadows ? 1 : 0, s.focusAvailable ? 1 : 0, s.fogShadows ? 1 : 0, s.atmoAvailable ? 1 : 0};
    u.scales = {s.sunScale, s.moonScale, s.iblScale, s.giIntensity};
    u.shadowParams = {std::tan(glm::radians(s.voxelShadowSoftnessDeg)), glm::radians(s.shadowPenumbraDeg),
                      s.fogHeightFalloff, s.godRayDistance};
    u.focusChunkOrigin = glm::vec4(s.focusChunkOrigin, 0.0f);
    u.cloudShadow0     = glm::vec4(s.cloudShadowOrigin + shift, s.cloudLayerAltitude, s.cloudShadowExtent);
    u.cloudShadow1     = glm::vec4(s.cloudShadowLight, 0.0f);
    return u;
}

/// Miroir std140 de CloudParamsBlock (shaders/render/cloud_density.glsl), UBO_VolumetricParams.
struct alignas(16) CloudParamsUbo {
    glm::vec4 layerAndScales{0.0f};      ///< x=bottom, y=top, z=shapeScale, w=detailScale
    glm::vec4 detailAndSkew{0.0f};       ///< x=detailStrength, y=curl, z=skew.x, w=skew.y
    glm::vec4 shapeOffset{0.0f};         ///< xyz=shapeOffset - shift, w=extinction
    glm::vec4 detailOffset{0.0f};        ///< xyz=detailOffset - shift, w=weatherExtent
    glm::vec4 weatherAndPhase{0.0f};     ///< xy=weatherOriginRel + shift, z=hgForward, w=hgBackward
    glm::vec4 lightingAndDist{0.0f};     ///< x=silver, y=powder, z=maxDistance, w=fadeDistance
    glm::vec4 cirrus0{0.0f};             ///< x=cirrusAltitude, y=cirrusCoverage, z=cirrusType, w=cirrusScale
    glm::vec4 cirrusAndLightning{0.0f};  ///< xy=cirrusOffset - shift, z=lightningIntensity, w=0
    glm::vec4 lightningPos{0.0f};        ///< xyz=lightningPosRel + shift, w=0
};
static_assert(sizeof(CloudParamsUbo) == 9 * 16, "CloudParamsUbo: layout std140");

/// Construit l'UBO des nuages, recalé sur l'origine de rendu `renderOriginChunk`.
inline CloudParamsUbo BuildCloudParamsUbo(const CloudParams& c, glm::ivec2 renderOriginChunk) {
    const glm::vec2 shift = glm::vec2(c.originChunk - renderOriginChunk) * kChunkSizeMeters;
    CloudParamsUbo  u;
    u.layerAndScales     = {c.bottom, c.top, c.shapeScale, c.detailScale};
    u.detailAndSkew      = {c.detailStrength, c.curl, c.skew.x, c.skew.y};
    u.shapeOffset        = glm::vec4(c.shapeOffset - glm::vec3(shift.x, 0.0f, shift.y), c.extinction);
    u.detailOffset       = glm::vec4(c.detailOffset - glm::vec3(shift.x, 0.0f, shift.y), c.weatherExtent);
    u.weatherAndPhase    = {c.weatherOriginRel.x + shift.x, c.weatherOriginRel.y + shift.y, c.hgForward, c.hgBackward};
    u.lightingAndDist    = {c.silver, c.powder, c.maxDistance, c.fadeDistance};
    u.cirrus0            = {c.cirrusAltitude, c.cirrusCoverage, c.cirrusType, c.cirrusScale};
    u.cirrusAndLightning = {c.cirrusOffset.x - shift.x, c.cirrusOffset.y - shift.y, c.lightningIntensity, 0.0f};
    u.lightningPos       = glm::vec4(c.lightningPosRel + glm::vec3(shift.x, 0.0f, shift.y), 0.0f);
    return u;
}

/// Renseigne les uniforms d'exposure.glsl.
inline void SetExposureUniforms(const gpu::ShaderProgram& p) {
    const LightingShared& s = SharedLighting();
    p.Set("uExposureCompensation", s.exposureCompensationEv);
    p.Set("uNightAdaptation",      s.nightAdaptation);
    p.Set("uMoonlightShift",       s.moonlightShift);
}

}  // namespace env::render
