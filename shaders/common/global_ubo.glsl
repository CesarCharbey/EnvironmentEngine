// =============================================================================
//  shaders/common/global_ubo.glsl
//  CONTRAT FIGE - Miroir EXACT (std140) de env::GlobalEnvironmentUBO
//  (include/contracts/WorldContracts.hpp). Toute modification = ADR.
//
//  Usage :  #include "common/global_ubo.glsl"
//  Les bindings (UBO_/SSBO_/TEX_/IMG_*) et constantes (CHUNK_SIZE_METERS,
//  BIOME_*, FOLIAGE_*, ...) sont injectes par ShaderManager apres #version.
//
//  Conventions (ADR-007) :
//    * positions relatives a l'origine de rendu flottante (uEnv.originChunk.xy)
//    * Reverse-Z, plan lointain infini : profondeur 1 = near, 0 = infini (ciel)
//    * Y vers le haut, metres
// =============================================================================
#ifndef ENV_GLOBAL_UBO_GLSL
#define ENV_GLOBAL_UBO_GLSL

#ifndef ENV_SHADER_PREAMBLE
#error "global_ubo.glsl doit etre charge via ShaderManager (preambule des bindings absent)"
#endif

layout(std140, binding = UBO_GlobalEnvironment) uniform GlobalEnvironmentBlock {
    // --- Camera [Core] ---
    mat4  view;
    mat4  proj;
    mat4  viewProj;
    mat4  invView;
    mat4  invProj;
    mat4  invViewProj;
    mat4  prevViewProj;
    vec4  cameraPosition;     // xyz = position camera (relative origine), w = 1
    vec4  cameraParams;       // x = near, y = distance de rendu, z = tan(fovY/2), w = aspect
    vec4  viewportSize;       // xy = pixels, zw = 1 / pixels
    ivec4 originChunk;        // xy = chunk (X, Z) de l'origine de rendu, z = generation
    // --- Soleil / Lune [CelestialAtmosphereSystem] ---
    vec4  sunDirIlluminance;  // xyz = direction VERS le soleil, w = lux
    vec4  sunColorAngular;    // rgb = couleur lineaire, a = rayon angulaire (rad)
    vec4  moonDirPhase;       // xyz = direction VERS la lune, w = phase [0,1) (0.5 = pleine)
    vec4  moonParams;         // x = lux, y = rayon angulaire (rad), z = lumiere cendree
    // --- Vent / Meteo [CelestialAtmosphereSystem] ---
    vec4  windDirSpeed;       // xy = direction (X, Z), z = vitesse (m/s), w = turbulence
    vec4  windGust;           // x = frequence (Hz), y = force
    vec4  weather;            // x = pluie, y = neige, z = tempete de sable, w = humidite
    vec4  weatherExtra;       // x = couverture nuageuse, y = densite nuageuse, z = brouillard (1/m), w = humidite de surface
    // --- Eau [BiomeHydrologySystem] / Temps [Core] ---
    vec4  waterTime;          // x = niveau ocean, y = niveau marais, z = temps (s), w = dt (s)
    uvec4 frameInfo;          // x = frame, y = vue de debug globale, z = flags (bit0 origine recalee, bit1 mock)
} uEnv;

// -----------------------------------------------------------------------------
// Accesseurs
// -----------------------------------------------------------------------------
vec3  envCameraPosition()  { return uEnv.cameraPosition.xyz; }
vec3  envSunDirection()    { return uEnv.sunDirIlluminance.xyz; }
float envSunIlluminance()  { return uEnv.sunDirIlluminance.w; }
vec3  envMoonDirection()   { return uEnv.moonDirPhase.xyz; }
float envMoonPhase()       { return uEnv.moonDirPhase.w; }
vec2  envWindDirection()   { return uEnv.windDirSpeed.xy; }
float envWindSpeed()       { return uEnv.windDirSpeed.z; }
float envRain()            { return uEnv.weather.x; }
float envSnow()            { return uEnv.weather.y; }
float envSandstorm()       { return uEnv.weather.z; }
float envHumidity()        { return uEnv.weather.w; }
float envOceanLevel()      { return uEnv.waterTime.x; }
float envSwampLevel()      { return uEnv.waterTime.y; }
float envTime()            { return uEnv.waterTime.z; }
float envDeltaTime()       { return uEnv.waterTime.w; }

// -----------------------------------------------------------------------------
// Reverse-Z (glClipControl GL_ZERO_TO_ONE) : NDC.z == profondeur
// -----------------------------------------------------------------------------
bool  envIsSky(float depth)       { return depth <= 0.0; }
/// Distance le long de -Z vue (m). Invalide pour depth == 0 (infini).
float envLinearDepth(float depth) { return uEnv.cameraParams.x / max(depth, 1e-7); }

/// Position vue depuis uv [0,1] et profondeur (depth > 0).
vec3 envViewPosFromDepth(vec2 uv, float depth) {
    vec4 p = uEnv.invProj * vec4(uv * 2.0 - 1.0, depth, 1.0);
    return p.xyz / p.w;
}

/// Position monde RELATIVE a l'origine de rendu depuis uv [0,1] et profondeur (depth > 0).
vec3 envWorldPosFromDepth(vec2 uv, float depth) {
    vec4 p = uEnv.invViewProj * vec4(uv * 2.0 - 1.0, depth, 1.0);
    return p.xyz / p.w;
}

/// Direction monde (unitaire) du rayon camera passant par uv [0,1]. Valable aussi pour le ciel.
vec3 envViewRayWorld(vec2 uv) {
    vec4 nearPoint = uEnv.invProj * vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    return normalize(mat3(uEnv.invView) * (nearPoint.xyz / nearPoint.w));
}

// -----------------------------------------------------------------------------
// Monde infini (ADR-007)
// -----------------------------------------------------------------------------
/// Coin d'un chunk relatif a l'origine de rendu (exact : difference entiere).
vec3 envChunkOriginRelative(ivec2 chunkCoord) {
    vec2 d = vec2(chunkCoord - uEnv.originChunk.xy) * CHUNK_SIZE_METERS;
    return vec3(d.x, 0.0, d.y);
}

// -----------------------------------------------------------------------------
// Miroirs GLSL des structures GPU du contrat (std430)
// -----------------------------------------------------------------------------
struct FoliageInstance {
    vec4 positionScale;  // xyz = position locale au chunk (m), w = echelle
    vec4 params;         // x = yaw, y = archetype (FOLIAGE_*), z = variante, w = alea
};

struct DrawElementsIndirectCommand {
    uint count;
    uint instanceCount;
    uint firstIndex;
    int  baseVertex;
    uint baseInstance;
};

// -----------------------------------------------------------------------------
// Palettes de debug partagees (inspecteur du blackboard, apercu mock)
// -----------------------------------------------------------------------------
/// Colormap Turbo (approximation polynomiale, Google 2019). x dans [0, 1].
vec3 envTurbo(float x) {
    const vec4 kRed4   = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kGreen4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 kBlue4  = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kRed2   = vec2(-152.94239396, 59.28637943);
    const vec2 kGreen2 = vec2(4.27729857, 2.82956604);
    const vec2 kBlue2  = vec2(-89.90310912, 27.34824973);
    x = clamp(x, 0.0, 1.0);
    vec4 v4 = vec4(1.0, x, x * x, x * x * x);
    vec2 v2 = v4.zw * v4.z;
    return vec3(dot(v4, kRed4) + dot(v2, kRed2), dot(v4, kGreen4) + dot(v2, kGreen2), dot(v4, kBlue4) + dot(v2, kBlue2));
}

/// Couleur de debug d'un biome (BIOME_*). Magenta si invalide.
vec3 envBiomeDebugColor(int biome) {
    if (biome == BIOME_OCEAN)         return vec3(0.05, 0.20, 0.45);
    if (biome == BIOME_BEACH)         return vec3(0.93, 0.85, 0.60);
    if (biome == BIOME_DESERT)        return vec3(0.85, 0.65, 0.30);
    if (biome == BIOME_PLAINS)        return vec3(0.55, 0.75, 0.30);
    if (biome == BIOME_FOREST)        return vec3(0.13, 0.45, 0.15);
    if (biome == BIOME_JUNGLE)        return vec3(0.02, 0.60, 0.35);
    if (biome == BIOME_SWAMP)         return vec3(0.30, 0.35, 0.20);
    if (biome == BIOME_HIGH_MOUNTAIN) return vec3(0.92, 0.92, 0.95);
    return vec3(1.0, 0.0, 1.0);
}

/// Colormap divergente pour SDF : bleu a l'interieur (< 0), rouge a l'exterieur, blanc sur la frontiere.
vec3 envDivergingSdf(float sdf, float range) {
    float t = clamp(sdf / max(range, 1e-4), -1.0, 1.0);
    vec3 inside  = vec3(0.10, 0.30, 0.90);
    vec3 outside = vec3(0.85, 0.20, 0.10);
    vec3 c = t < 0.0 ? mix(vec3(1.0), inside, -t) : mix(vec3(1.0), outside, t);
    float edge = 1.0 - smoothstep(0.0, 0.03, abs(t));
    return mix(c, vec3(0.0), edge * 0.85);
}

#endif // ENV_GLOBAL_UBO_GLSL
