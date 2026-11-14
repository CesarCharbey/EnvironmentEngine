//  shaders/render/terrain_material.glsl                                [DEV 2]
//  Matériau de terrain procédural (sans textures), partagé par la G-pass (gbuffer_terrain.frag)
//  et la voxélisation VXGI (voxel_inject.comp) : le rebond de lumière indirecte a ainsi exactement
//  la couleur du sol rendu.
//    * Albédo de sol par biome, fondu bilinéaire entre texels (l'id se lit avec texelFetch).
//    * Roche selon la pente (N.y), lits de rivière / marais boueux et mouillés.
//    * Humidité de surface : berges (SDF) + pluie (uEnv.weatherExtra.w, accumulée par
//      CelestialAtmosphereSystem). Réponse dépendant de la porosité (Lagarde 2013) : un sol poreux
//      boit l'eau (s'assombrit et se sature, reste assez mat), la roche se couvre d'un film brillant.
//      Les flaques et les ondes de pluie (détail sous le voxel) sont dans terrain_wetness.glsl (G-pass).
//    * Neige selon l'altitude, la pente et la neige météo (material_params.glsl), par-dessus le reste.
//  Le shader qui inclut ce fichier doit avoir inclus "common/global_ubo.glsl".
#ifndef ENV_TERRAIN_MATERIAL_GLSL
#define ENV_TERRAIN_MATERIAL_GLSL

#include "render/material_params.glsl"

// Porosité du sol nu par biome : 1 = terre qui boit l'eau, faible = surface dure (film d'eau).
float BiomePorosity(int biome) {
    if (biome == BIOME_OCEAN || biome == BIOME_BEACH || biome == BIOME_DESERT) return 0.9;  // sable
    if (biome == BIOME_HIGH_MOUNTAIN) return 0.5;                                            // éboulis
    return 1.0;
}

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

int BiomeAt(sampler2D biomeHydro, ivec2 texel) {
    ivec2 size = textureSize(biomeHydro, 0);
    return int(texelFetch(biomeHydro, clamp(texel, ivec2(0), size - 1), 0).r + 0.5);
}

// Fondu bilinéaire des matériaux des 4 texels voisins (évite les transitions en escalier de 1 m).
vec4 BlendedGround(sampler2D biomeHydro, vec2 uv) {
    vec2  st = uv * vec2(textureSize(biomeHydro, 0)) - 0.5;  // centres de texels (contrat)
    ivec2 i0 = ivec2(floor(st));
    vec2  f  = fract(st);
    vec4 a = BiomeGround(BiomeAt(biomeHydro, i0));
    vec4 b = BiomeGround(BiomeAt(biomeHydro, i0 + ivec2(1, 0)));
    vec4 c = BiomeGround(BiomeAt(biomeHydro, i0 + ivec2(0, 1)));
    vec4 d = BiomeGround(BiomeAt(biomeHydro, i0 + ivec2(1, 1)));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

struct TerrainMaterial {
    vec3  albedo;     // linéaire
    float roughness;  // perceptuelle
    float wetness;    // humidité de surface [0, 1] (berges + pluie) ; gbuffer.material.b
    float porosity;   // [0, 1] : 1 = terre, ~0,25 = roche
    float snow;       // couverture neigeuse [0, 1]
    int   biome;      // id BIOME_* du texel
};

float TerrainLuminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

// uv : UV du chunk [0, 1] ; N : normale monde ; worldY : altitude (m), pour les strates de roche.
TerrainMaterial EvaluateTerrainMaterial(sampler2D biomeHydro, vec2 uv, vec3 N, float worldY) {
    vec4  bh       = textureLod(biomeHydro, uv, 0.0);
    float riverSdf = bh.g;
    float marshSdf = bh.b;
    float moisture = clamp(bh.a, 0.0, 1.0);

    TerrainMaterial m;
    m.biome = BiomeAt(biomeHydro, ivec2(clamp(uv, 0.0, 0.9999) * vec2(textureSize(biomeHydro, 0))));

    // --- Sol du biome, assombri par l'humidité du sol ---
    vec4 ground = BlendedGround(biomeHydro, uv);
    m.albedo    = ground.rgb * mix(1.05, 0.80, moisture);
    m.roughness = ground.a;

    // --- "Triplanar" simplifié : la roche affleure dans les pentes ---
    float slope = 1.0 - N.y;
    float rock  = smoothstep(0.25, 0.45, slope);
    vec3  rockAlbedo = vec3(0.20, 0.19, 0.18) * (0.9 + 0.1 * sin(worldY * 0.7));  // strates
    m.albedo    = mix(m.albedo, rockAlbedo, rock);
    m.roughness = mix(m.roughness, 0.80, rock);

    // --- Hydrologie : lit boueux et berges mouillées ---
    float waterSdf = min(riverSdf, marshSdf);
    float bed      = 1.0 - smoothstep(-0.5, 1.0, waterSdf);
    m.albedo       = mix(m.albedo, vec3(0.10, 0.09, 0.07), bed);
    m.porosity     = mix(mix(BiomePorosity(m.biome), 0.25, rock), 0.6, bed);  // vase : à demi saturée

    // --- Humidité : berges (capillarité) + pluie. La pluie mouille vite la surface (saturée dès 45 %
    //     de l'humidité météo) ; au-delà, l'eau s'accumule en flaques (terrain_wetness.glsl). ---
    float bankWet = 1.0 - smoothstep(0.0, 4.0, waterSdf);
    float rainWet = smoothstep(0.0, 0.45, clamp(uEnv.weatherExtra.w, 0.0, 1.0));
    m.wetness     = max(bankWet, rainWet);

    // Réponse PBR (Lagarde 2013) : l'eau remplit les pores -> moins de diffusion vers l'extérieur,
    // albédo assombri (×0,5 pour une terre saturée) et plus saturé ; rugosité vers un film d'eau, d'autant
    // plus lisse que la surface est dure (une terre poreuse reste satinée). Le F0 de l'eau est appliqué par
    // l'éclairage (gbuffer.material.b).
    float soak  = m.wetness * m.porosity;
    vec3  sat   = max(mix(vec3(TerrainLuminance(m.albedo)), m.albedo, 1.0 + 0.35 * soak), vec3(0.0));
    m.albedo    = sat * mix(1.0, 0.5, soak);
    m.roughness = mix(m.roughness, mix(0.10, 0.38, m.porosity), m.wetness);

    // --- Neige : recouvre le sol (et son humidité), pas le lit des rivières ni des marais ---
    m.snow      = SnowCoverage(worldY, N.y) * (1.0 - bed);
    m.albedo    = mix(m.albedo, kSnowAlbedo, m.snow);
    m.roughness = mix(m.roughness, kSnowRoughness, m.snow);
    m.wetness  *= 1.0 - m.snow;
    return m;
}

#endif  // ENV_TERRAIN_MATERIAL_GLSL
