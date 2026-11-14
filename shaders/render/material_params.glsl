//  shaders/render/material_params.glsl                                 [DEV 2]
//  Paramètres de matériaux partagés (UBO_MaterialParams), remplis par DeferredGPassSystem à chaque frame
//  et lus par la G-pass (terrain, végétation) et la voxélisation VXGI : le sol voxélisé a exactement la
//  même neige que le sol rendu.
//  Le shader qui inclut ce fichier doit avoir inclus "common/global_ubo.glsl".
#ifndef ENV_MATERIAL_PARAMS_GLSL
#define ENV_MATERIAL_PARAMS_GLSL

layout(std140, binding = UBO_MaterialParams) uniform MaterialParamsBlock {
    vec4 snowLine;  // x = altitude des premières neiges (m), y = altitude atteinte à neige maximale (m),
                    // z = demi-largeur de la transition (m), w = libre
} uMaterial;

// Couverture neigeuse [0, 1] d'une surface d'altitude `altitude` (m) et de normale verticale Ny.
// La limite des neiges descend de snowLine.x à snowLine.y quand la neige météo (uEnv.weather.y) monte.
// Les replats la retiennent plus bas que les pentes, ce qui dessine la limite selon le relief (sans
// bruit ni texture) ; au-delà de ~50° de pente, elle glisse.
float SnowCoverage(float altitude, float Ny) {
    float s = clamp(envSnow(), 0.0, 1.0);
    if (s <= 0.0) return 0.0;
    float T    = max(uMaterial.snowLine.z, 0.1);
    float line = mix(uMaterial.snowLine.x, uMaterial.snowLine.y, s);
    float h    = altitude + (Ny - 0.85) * 2.0 * T;
    return smoothstep(line - T, line + T, h) * smoothstep(0.55, 0.8, Ny) * smoothstep(0.0, 0.15, s);
}

// Matériau de la neige fraîche (albédo linéaire, rugosité perceptuelle).
const vec3  kSnowAlbedo    = vec3(0.85, 0.87, 0.90);
const float kSnowRoughness = 0.5;

#endif  // ENV_MATERIAL_PARAMS_GLSL
