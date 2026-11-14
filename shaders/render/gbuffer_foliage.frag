#version 460 core
//  shaders/render/gbuffer_foliage.frag                                 [DEV 2]
//  Matériau de végétation basique -> G-Buffer (mêmes sorties que gbuffer_terrain.frag).
#include "common/global_ubo.glsl"
#include "render/material_params.glsl"

in VS_OUT {
    vec3 normal;
    float worldY;
    vec4 clipCurr;
    vec4 clipPrev;
    flat int   archetype;
    flat float variation;
} fsIn;

layout(location = 0) out vec4 oAlbedo;
layout(location = 1) out vec4 oNormal;
layout(location = 2) out vec4 oMaterial;
layout(location = 3) out vec2 oVelocity;

void main() {
    // Albédos linéaires par archétype, légère variation par instance.
    vec3  albedo    = vec3(0.10, 0.22, 0.05);  // herbe
    float roughness = 0.9;
    float bendUp    = 0.5;                      // végétation : normale ramenée vers +Y (éclairage plus doux)
    if (fsIn.archetype == FOLIAGE_SHRUB) albedo = vec3(0.08, 0.17, 0.05);
    if (fsIn.archetype == FOLIAGE_TREE)  albedo = vec3(0.04, 0.11, 0.04);
    if (fsIn.archetype == FOLIAGE_ROCK)  { albedo = vec3(0.25, 0.24, 0.22); roughness = 0.8; bendUp = 0.0; }
    albedo *= 0.8 + 0.4 * fsIn.variation;

    vec3 N = normalize(mix(normalize(fsIn.normal), vec3(0.0, 1.0, 0.0), bendUp));

    // Pluie : feuilles et rochers mouillés. Les feuilles (cuticule cireuse) ne boivent pas l'eau : elles
    // foncent peu et deviennent brillantes ; le rocher se comporte comme la roche du terrain.
    float wet      = smoothstep(0.0, 0.45, clamp(uEnv.weatherExtra.w, 0.0, 1.0));
    float porosity = fsIn.archetype == FOLIAGE_ROCK ? 0.25 : 0.15;
    albedo   *= mix(1.0, 0.5, wet * porosity);
    roughness = mix(roughness, fsIn.archetype == FOLIAGE_ROCK ? 0.15 : 0.35, wet);

    // Neige : posée sur les faces tournées vers le ciel (normale géométrique), même limite que le sol.
    float snow = SnowCoverage(fsIn.worldY, normalize(fsIn.normal).y);
    albedo    = mix(albedo, kSnowAlbedo, snow);
    roughness = mix(roughness, kSnowRoughness, snow);
    wet      *= 1.0 - snow;

    vec2 ndcCurr = fsIn.clipCurr.xy / fsIn.clipCurr.w;
    vec2 ndcPrev = fsIn.clipPrev.xy / fsIn.clipPrev.w;

    oAlbedo   = vec4(albedo, 1.0);
    oNormal   = vec4(N, 0.0);
    oMaterial = vec4(roughness, 0.0, wet, float(16 + fsIn.archetype) / 255.0);
    oVelocity = ndcCurr - ndcPrev;
}
