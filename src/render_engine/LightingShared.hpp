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
    float cloudLayerAltitude     = 2000.0f;
    float cloudShadowExtent      = 10000.0f;

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
inline void SetSurfaceLightingUniforms(const gpu::ShaderProgram& p) {
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
    p.Set("uCloudLayerAltitude",   s.cloudLayerAltitude);
    p.Set("uCloudShadowExtent",    s.cloudShadowExtent);
}

/// Renseigne les uniforms d'exposure.glsl.
inline void SetExposureUniforms(const gpu::ShaderProgram& p) {
    const LightingShared& s = SharedLighting();
    p.Set("uExposureCompensation", s.exposureCompensationEv);
    p.Set("uNightAdaptation",      s.nightAdaptation);
    p.Set("uMoonlightShift",       s.moonlightShift);
}

}  // namespace env::render
