//  shaders/render/ocean_common.glsl                                    [DEV 2]
//  Paramètres de l'eau (UBO_WaterParams, rempli par WaterOceanRiverSystem) et disposition du buffer
//  du spectre de l'océan (SSBO_OceanSpectrum) partagés par les compute shaders de la FFT et les shaders
//  de surface de l'eau.
//
//  Cartes de l'océan (texture 2D array RGBA16F N × N, une texture pour toutes les cascades, liée sur
//  TEX_OceanDisplacement en lecture et IMG_WaterOut en écriture) :
//    * couche c (0..2)     : xyz = déplacement (m) (x et z déjà multipliés par la « choppiness »),
//                            w = écume accumulée [0, 1] ;
//    * couche 3 + c        : x = ∂h/∂x, y = ∂h/∂z, z = λ ∂Dx/∂x, w = λ ∂Dz/∂z (normale du maillage déplacé).
#ifndef ENV_OCEAN_COMMON_GLSL
#define ENV_OCEAN_COMMON_GLSL

#define OCEAN_CASCADES 3
const float PI_OCEAN = 3.14159265358979;

layout(std140, binding = UBO_WaterParams) uniform WaterParamsBlock {
    ivec4 oceanInfo;       // x = N (résolution FFT), y = log2(N)
    vec4  cascadeSize;     // xyz = côté de chaque cascade (m)
    vec4  cutLow;          // xyz = nombre d'onde minimal de la cascade (rad/m)
    vec4  cutHigh;         // xyz = nombre d'onde maximal (exclu)
    vec4  oceanMisc;       // x = choppiness λ, y = seuil d'écume (jacobien), z = demi-vie de l'écume (s), w = temps (s)
    vec4  absorption;      // rgb = coefficient d'absorption de l'eau claire (1/m), w = force de réfraction
    vec4  scattering;      // rgb = coefficient de diffusion (1/m), w = turbidité du marais [0, 1]
    vec4  riverParams;     // x = vitesse d'écoulement (m/s), y = force des normales, z = douceur des berges (m), w = SSR (0/1)
    vec4  swampColor;      // rgb = absorption ajoutée par la turbidité (1/m), w = diffusion ajoutée (1/m)
    vec4  originModX;      // xyz = (X monde de l'origine de rendu) mod côté de la cascade (m), calculé en double
    vec4  originModZ;      // xyz = idem en Z : les cartes restent fixes quand l'origine se recale (ADR-007)
    vec4  slopeVariance;   // xyz = variance des pentes de chaque cascade (intégrale de k² S(k) du spectre)
} uWater;

#define uOceanN            (uWater.oceanInfo.x)
#define uOceanLogN         (uWater.oceanInfo.y)
#define uOceanCascadeSize  (uWater.cascadeSize)
#define uOceanCutLow       (uWater.cutLow)
#define uOceanCutHigh      (uWater.cutHigh)

layout(std430, binding = SSBO_OceanSpectrum) buffer OceanSpectrumBuffer {
    // [0, 3N²)            : h0 par cascade : vec4(h0(k), conj(h0(-k)))
    // [3N², 3N² + 6N²)    : résultat de la passe sur les lignes (2 vec4 par élément : 4 champs complexes)
    vec4 spectrum[];
};

// Coordonnée de lecture de la cascade c pour une position xz relative à l'origine de rendu.
vec3 OceanUv(vec2 xzRelative, int c, int layerOffset) {
    float L = uWater.cascadeSize[c];
    return vec3((xzRelative + vec2(uWater.originModX[c], uWater.originModZ[c])) / L, float(c + layerOffset));
}

int OceanSpectrumIndex(ivec2 n, int cascade) { return (cascade * uOceanN + n.y) * uOceanN + n.x; }
int OceanRowsIndex(ivec2 n, int cascade) {
    return OCEAN_CASCADES * uOceanN * uOceanN + 2 * OceanSpectrumIndex(n, cascade);
}

// Hachage entier 3D -> 3 × 32 bits (pcg3d, Jarzynski & Olano 2020).
uvec3 OceanHash3(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

#endif  // ENV_OCEAN_COMMON_GLSL
