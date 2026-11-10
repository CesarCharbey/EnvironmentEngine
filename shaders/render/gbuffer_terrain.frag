#version 460 core
//  shaders/render/gbuffer_terrain.frag                                 [DEV 2]
//  Matériau de terrain procédural (sans textures) -> G-Buffer.
//    * Albédo de sol par biome, fondu bilinéaire entre texels (l'id se lit avec texelFetch).
//    * Roche selon la pente (N.y), lits de rivière / marais boueux et mouillés.
//    * Sorties (contrat gbuffer.*) : 0 albédo sRGB + AO, 1 normale monde, 2 matériau, 3 vitesse NDC.
#include "common/global_ubo.glsl"

in VS_OUT {
    vec3 worldPos;
    vec2 uv;
    vec4 clipCurr;
    vec4 clipPrev;
} fsIn;

layout(binding = TEX_HeightNormal) uniform sampler2D uHeightNormal;
layout(binding = TEX_BiomeHydro)   uniform sampler2D uBiomeHydro;

layout(location = 0) out vec4 oAlbedo;    // SRGB8_ALPHA8 : rgb albédo (linéaire, encodé par GL_FRAMEBUFFER_SRGB), a AO
layout(location = 1) out vec4 oNormal;    // RGBA16F : xyz normale monde
layout(location = 2) out vec4 oMaterial;  // RGBA8 : r rugosité, g métal, b humidité de surface, a id matériau / 255
layout(location = 3) out vec2 oVelocity;  // RG16F : déplacement NDC (courant - précédent)

// Albédo linéaire (rgb) et rugosité (a) du sol nu par biome.
vec4 BiomeGround(int biome) {
    if (biome == BIOME_OCEAN)         return vec4(0.20, 0.18, 0.13, 0.85);  // fond sableux
    if (biome == BIOME_BEACH)         return vec4(0.45, 0.40, 0.28, 0.85);
    if (biome == BIOME_DESERT)        return vec4(0.50, 0.36, 0.20, 0.85);
    if (biome == BIOME_PLAINS)        return vec4(0.12, 0.20, 0.06, 0.90);
    if (biome == BIOME_FOREST)        return vec4(0.07, 0.12, 0.04, 0.90);
    if (biome == BIOME_JUNGLE)        return vec4(0.05, 0.13, 0.04, 0.90);
    if (biome == BIOME_SWAMP)         return vec4(0.09, 0.10, 0.05, 0.85);
    if (biome == BIOME_HIGH_MOUNTAIN) return vec4(0.30, 0.29, 0.28, 0.85);  // éboulis (la neige vient de la météo)
    return vec4(0.12, 0.20, 0.06, 0.90);
}

int BiomeAt(ivec2 texel) {
    ivec2 size = textureSize(uBiomeHydro, 0);
    return int(texelFetch(uBiomeHydro, clamp(texel, ivec2(0), size - 1), 0).r + 0.5);
}

// Fondu bilinéaire des matériaux des 4 texels voisins (évite les transitions en escalier de 1 m).
vec4 BlendedGround(vec2 uv) {
    vec2  st = uv * vec2(textureSize(uBiomeHydro, 0)) - 0.5;  // centres de texels (contrat)
    ivec2 i0 = ivec2(floor(st));
    vec2  f  = fract(st);
    vec4 a = BiomeGround(BiomeAt(i0));
    vec4 b = BiomeGround(BiomeAt(i0 + ivec2(1, 0)));
    vec4 c = BiomeGround(BiomeAt(i0 + ivec2(0, 1)));
    vec4 d = BiomeGround(BiomeAt(i0 + ivec2(1, 1)));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

void main() {
    vec3 N  = normalize(textureLod(uHeightNormal, fsIn.uv, 0.0).gba);  // par pixel : plus net qu'interpolé
    vec4 bh = texture(uBiomeHydro, fsIn.uv);
    float riverSdf = bh.g;
    float marshSdf = bh.b;
    float moisture = clamp(bh.a, 0.0, 1.0);
    int   biome    = BiomeAt(ivec2(clamp(fsIn.uv, 0.0, 0.9999) * vec2(textureSize(uBiomeHydro, 0))));

    // --- Sol du biome, assombri par l'humidité du sol ---
    vec4  ground    = BlendedGround(fsIn.uv);
    vec3  albedo    = ground.rgb * mix(1.05, 0.80, moisture);
    float roughness = ground.a;

    // --- "Triplanar" simplifié : la roche affleure dans les pentes ---
    float slope = 1.0 - N.y;
    float rock  = smoothstep(0.25, 0.45, slope);
    vec3  rockAlbedo = vec3(0.20, 0.19, 0.18) * (0.9 + 0.1 * sin(fsIn.worldPos.y * 0.7));  // strates
    albedo    = mix(albedo, rockAlbedo, rock);
    roughness = mix(roughness, 0.80, rock);

    // --- Hydrologie : lit boueux et berges mouillées (lu comme humidité de surface par l'éclairage) ---
    float waterSdf = min(riverSdf, marshSdf);
    float bed      = 1.0 - smoothstep(-0.5, 1.0, waterSdf);
    float wetness  = 1.0 - smoothstep(0.0, 4.0, waterSdf);
    albedo = mix(albedo, vec3(0.10, 0.09, 0.07), bed);

    vec2 ndcCurr = fsIn.clipCurr.xy / fsIn.clipCurr.w;
    vec2 ndcPrev = fsIn.clipPrev.xy / fsIn.clipPrev.w;

    oAlbedo   = vec4(albedo, 1.0);
    oNormal   = vec4(N, 0.0);
    oMaterial = vec4(roughness, 0.0, wetness, float(biome) / 255.0);
    oVelocity = ndcCurr - ndcPrev;
}
