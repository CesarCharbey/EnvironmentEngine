#version 460 core
// =============================================================================
//  shaders/core_debug/mock_preview.frag  (core, gele)
//  Rendu "argile" + vues en fausses couleurs des donnees du contrat :
//    0 Clay | 1 Height | 2 Normal | 3 Biome | 4 River SDF | 5 Marsh SDF
//    6 Moisture | 7 Chunk Grid
//  Sortie encodee pour l'affichage (gamma 2.2) : ce n'est PAS le pipeline PBR.
// =============================================================================
#include "common/global_ubo.glsl"

in VS_OUT {
    vec3 worldPos;
    vec3 normal;
    vec2 chunkUV;
    flat int archetype;
} fsIn;

layout(binding = TEX_BiomeHydro) uniform sampler2D uBiomeHydro;

uniform int   uViewMode;
uniform vec2  uHeightRange;  // min / max pour la vue Height (m)
uniform float uSdfRange;     // demi-plage des vues SDF (m)

layout(location = 0) out vec4 oColor;

// Doit correspondre a kSceneClear (Application.cpp) une fois encode en gamma.
const vec3 kFogColorDisplay = vec3(0.62, 0.70, 0.80);

vec3 clayLighting(vec3 albedo, vec3 n) {
    vec3  sunDir  = normalize(envSunDirection());
    float ndl     = max(dot(n, sunDir), 0.0);
    float sunUp   = smoothstep(-0.05, 0.08, sunDir.y);
    float hemi    = 0.5 + 0.5 * n.y;
    vec3  ambient = mix(vec3(0.30, 0.26, 0.22), vec3(0.55, 0.65, 0.80), hemi) * 0.45;
    return albedo * (ambient + vec3(1.0, 0.95, 0.86) * ndl * 0.95 * sunUp);
}

vec3 archetypeClay(int archetype) {
    if (archetype == FOLIAGE_GRASS) return vec3(0.62, 0.70, 0.45);
    if (archetype == FOLIAGE_SHRUB) return vec3(0.55, 0.62, 0.42);
    if (archetype == FOLIAGE_TREE)  return vec3(0.42, 0.52, 0.36);
    if (archetype == FOLIAGE_ROCK)  return vec3(0.60, 0.58, 0.56);
    return vec3(0.8);
}

float isoLine(float value, float stepSize) {
    float f = value / stepSize;
    return smoothstep(0.0, 1.0, abs(fract(f - 0.5) - 0.5) / max(fwidth(f), 1e-4));
}

void main() {
    vec3  n          = normalize(fsIn.normal);
    vec4  biomeHydro = texture(uBiomeHydro, fsIn.chunkUV);
    int   biome      = int(texelFetch(uBiomeHydro, ivec2(clamp(fsIn.chunkUV, 0.0, 0.9999) * vec2(textureSize(uBiomeHydro, 0))), 0).r + 0.5);
    float riverSdf   = biomeHydro.g;
    float marshSdf   = biomeHydro.b;
    float moisture   = biomeHydro.a;
    bool  isInstance = fsIn.archetype >= 0;
    float height     = fsIn.worldPos.y;  // l'origine de rendu est a y = 0 : hauteur absolue

    vec3 color;
    if (uViewMode == 0) {
        vec3 albedo = isInstance ? archetypeClay(fsIn.archetype) : vec3(0.70, 0.62, 0.54);
        if (!isInstance) {
            float river = 1.0 - smoothstep(-1.0, 1.0, riverSdf);
            float marsh = 1.0 - smoothstep(-1.0, 1.0, marshSdf);
            albedo = mix(albedo, vec3(0.36, 0.40, 0.28), marsh * 0.85);
            albedo = mix(albedo, vec3(0.16, 0.36, 0.58), river);
            albedo *= mix(0.82, 1.0, isoLine(height, 5.0));  // isolignes tous les 5 m
        }
        color = clayLighting(albedo, n);
    } else if (uViewMode == 1) {
        color = envTurbo((height - uHeightRange.x) / max(uHeightRange.y - uHeightRange.x, 1e-3));
        color *= mix(0.6, 1.0, isoLine(height, 5.0));
    } else if (uViewMode == 2) {
        color = n * 0.5 + 0.5;
    } else if (uViewMode == 3) {
        color = isInstance ? archetypeClay(fsIn.archetype) : envBiomeDebugColor(biome);
    } else if (uViewMode == 4) {
        color = envDivergingSdf(riverSdf, uSdfRange) * mix(0.7, 1.0, isoLine(riverSdf, 5.0));
    } else if (uViewMode == 5) {
        color = envDivergingSdf(marshSdf, uSdfRange) * mix(0.7, 1.0, isoLine(marshSdf, 5.0));
    } else if (uViewMode == 6) {
        color = mix(vec3(0.80, 0.62, 0.38), vec3(0.08, 0.30, 0.90), clamp(moisture, 0.0, 1.0));
    } else {
        vec2  cell    = floor(fsIn.chunkUV * 16.0);
        float checker = mod(cell.x + cell.y, 2.0);
        vec2  border  = abs(fract(fsIn.chunkUV) - 0.5);
        float edge    = step(0.495, max(border.x, border.y));
        color = mix(vec3(0.85, 0.85, 0.80), vec3(0.60, 0.62, 0.68), checker);
        color = mix(color, vec3(1.0, 0.2, 0.1), edge);
    }

    // Leger ombrage des vues fausses couleurs pour garder la lecture du relief.
    if (uViewMode != 0 && uViewMode != 2)
        color *= 0.65 + 0.35 * max(dot(n, normalize(envSunDirection())), 0.0);

    // Brouillard de distance (densite pilotee par uEnv.weatherExtra.z).
    float dist = length(fsIn.worldPos - envCameraPosition());
    float fog  = 1.0 - exp(-dist * max(uEnv.weatherExtra.z, 0.0));
    color = mix(color, pow(kFogColorDisplay, vec3(2.2)), fog);

    oColor = vec4(pow(max(color, vec3(0.0)), vec3(1.0 / 2.2)), 1.0);
}
