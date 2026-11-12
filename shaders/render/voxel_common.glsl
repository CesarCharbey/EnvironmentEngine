//  shaders/render/voxel_common.glsl                                    [DEV 2]
//  VXGI (ADR-008) : placement de la grille de voxels et cone tracing, partagés par la voxélisation
//  (voxel_inject.comp) et la résolution différée (deferred_lighting.frag).
//
//  Grille : Texture3D RGBA16F 256 × 64 × 256, voxels cubiques de 8 m, chaîne de mips complète.
//    rgb = radiance sortante (cd/m²) PRÉ-MULTIPLIÉE par l'opacité, a = opacité [0, 1].
//    La pré-multiplication rend la moyenne des mips correcte (un voxel vide n'assombrit pas la lumière).
#ifndef ENV_VOXEL_COMMON_GLSL
#define ENV_VOXEL_COMMON_GLSL

layout(std140, binding = UBO_VoxelParams) uniform VoxelParamsBlock {
    vec4 uGridMin;        // xyz = coin min de la grille courante (m, relatif à l'origine de rendu), w = taille de voxel (m)
    vec4 uGridExtent;     // xyz = emprise (m), w = 1 si la grille courante est valide
    vec4 uGridInvExtent;  // xyz = 1 / emprise, w = LOD max exploitable (l'axe Y de 64 voxels s'épuise au LOD 6)
    vec4 uPrevGridMin;    // xyz = coin min de la grille PRÉCÉDENTE (m, relatif à l'origine courante), w = 1 si valide
} uVoxel;

// Grille échantillonnée : la grille publiée (éclairage) ou la grille de la frame précédente (injection).
layout(binding = TEX_VoxelGrid) uniform sampler3D uVoxelGrid;

float VoxelSize() { return uVoxel.uGridMin.w; }

vec3 VoxelUvw(vec3 p, vec3 gridMin) { return (p - gridMin) * uVoxel.uGridInvExtent.xyz; }

bool VoxelUvwInside(vec3 uvw) {
    return all(greaterThanEqual(uvw, vec3(0.0))) && all(lessThanEqual(uvw, vec3(1.0)));
}

// Visibilité [0, 1] le long d'un cône (origine, direction, tangente du demi-angle d'ouverture).
// Le diamètre du cône grandit avec la distance : on échantillonne un mip de plus en plus grossier,
// d'où une pénombre qui s'élargit naturellement avec la distance à l'occultant.
// La marche s'arrête à la sortie de la grille (au-delà, on ne sait pas : visible).
float VoxelConeVisibility(vec3 gridMin, vec3 origin, vec3 dir, float tanHalf, float startDist, float maxDist,
                          int maxSteps) {
    float voxel  = VoxelSize();
    float maxLod = uVoxel.uGridInvExtent.w;
    float occ    = 0.0;
    float t      = startDist;
    for (int i = 0; i < maxSteps; ++i) {
        if (t > maxDist || occ > 0.97) break;
        float diam = max(voxel, 2.0 * tanHalf * t);
        vec3  uvw  = VoxelUvw(origin + dir * t, gridMin);
        if (!VoxelUvwInside(uvw)) break;
        float lod     = min(log2(diam / voxel), maxLod);
        float a       = clamp(textureLod(uVoxelGrid, uvw, lod).a, 0.0, 0.999);
        float stepLen = max(diam * 0.6, voxel * 0.75);
        // Correction d'opacité : a vaut pour une traversée de longueur diam, le pas est plus court.
        a    = 1.0 - pow(1.0 - a, stepLen / diam);
        occ += (1.0 - occ) * a;
        t   += stepLen;
    }
    return 1.0 - occ;
}

// Radiance accumulée le long d'un cône (composition avant -> arrière, radiance pré-multipliée) :
// rgb = radiance reçue des voxels (cd/m²), a = occlusion totale. Le reste (1 - a) voit le ciel.
vec4 VoxelConeRadiance(vec3 gridMin, vec3 origin, vec3 dir, float tanHalf, float startDist, float maxDist,
                       int maxSteps) {
    float voxel  = VoxelSize();
    float maxLod = uVoxel.uGridInvExtent.w;
    vec4  acc    = vec4(0.0);
    float t      = startDist;
    for (int i = 0; i < maxSteps; ++i) {
        if (t > maxDist || acc.a > 0.97) break;
        float diam = max(voxel, 2.0 * tanHalf * t);
        vec3  uvw  = VoxelUvw(origin + dir * t, gridMin);
        if (!VoxelUvwInside(uvw)) break;
        float lod     = min(log2(diam / voxel), maxLod);
        vec4  s       = textureLod(uVoxelGrid, uvw, lod);
        float a       = clamp(s.a, 0.0, 0.999);
        float stepLen = max(diam * 0.8, voxel);
        float aStep   = 1.0 - pow(1.0 - a, stepLen / diam);
        vec3  rgb     = s.rgb * (aStep / max(a, 1e-4));  // radiance pré-multipliée par l'opacité corrigée
        acc.rgb += (1.0 - acc.a) * rgb;
        acc.a   += (1.0 - acc.a) * aStep;
        t       += stepLen;
    }
    return acc;
}

#endif  // ENV_VOXEL_COMMON_GLSL
