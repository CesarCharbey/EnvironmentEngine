//  shaders/render/terrain_material.glsl                                [DEV 2]
//  Matériau de terrain procédural (sans textures), partagé par la G-pass (gbuffer_terrain.frag)
//  et la voxélisation VXGI (voxel_inject.comp) : le rebond de lumière indirecte a ainsi exactement
//  la couleur du sol rendu.
//    * Albédo de sol par biome, fondu bilinéaire entre texels (l'id se lit avec texelFetch).
//    * Roche selon la pente (N.y), lits de rivière / marais boueux et mouillés.
#ifndef ENV_TERRAIN_MATERIAL_GLSL
#define ENV_TERRAIN_MATERIAL_GLSL

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
    float wetness;    // humidité de surface (lits et berges), lue par l'éclairage
    int   biome;      // id BIOME_* du texel
};

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
    m.wetness      = 1.0 - smoothstep(0.0, 4.0, waterSdf);
    m.albedo       = mix(m.albedo, vec3(0.10, 0.09, 0.07), bed);
    return m;
}

#endif  // ENV_TERRAIN_MATERIAL_GLSL
