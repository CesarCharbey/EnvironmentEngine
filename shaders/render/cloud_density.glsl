//  shaders/render/cloud_density.glsl                                   [DEV 2]
//  Modèle de densité des nuages (Nubis 2017), partagé par le raymarching (volumetric_clouds.comp) et la
//  carte d'ombre au sol (cloud_shadow.comp) : les ombres ont exactement la forme des nuages affichés.
//  Uniforms renseignés côté C++ par VolumetricGodRaysSystem::SetDensityUniforms.
//  Le shader qui inclut ce fichier doit avoir inclus "common/global_ubo.glsl".
#ifndef ENV_CLOUD_DENSITY_GLSL
#define ENV_CLOUD_DENSITY_GLSL

layout(binding = TEX_CloudShapeNoise)  uniform sampler3D uShapeNoise;
layout(binding = TEX_CloudDetailNoise) uniform sampler3D uDetailNoise;
layout(binding = TEX_CloudWeatherMap)  uniform sampler2D uWeatherMap;

uniform float uCloudBottom;       // m (altitude)
uniform float uCloudTop;          // m
uniform float uShapeScale;        // m : côté d'une répétition du bruit de forme
uniform float uDetailScale;       // m
uniform float uDetailStrength;    // [0, 1]
uniform vec3  uShapeOffset;       // m : (origine de rendu absolue - vent) mod uShapeScale (double côté C++)
uniform vec3  uDetailOffset;      // m : idem mod uDetailScale
uniform vec2  uWeatherOriginRel;  // m : centre de la fenêtre météo, relatif à l'origine de rendu
uniform float uWeatherExtent;     // m
uniform vec2  uSkew;              // m : décalage horizontal au sommet de la couche (vent, Nubis)
uniform float uCurlStrength;      // torsion du détail (en périodes du bruit de détail)

float Remap(float v, float a, float b, float c, float d) { return c + (v - a) / max(b - a, 1e-5) * (d - c); }

// Météo au point (xz relatif à l'origine de rendu). Hors de la fenêtre, la carte est lue en miroir :
// les nuages continuent jusqu'à l'horizon sans coupure ni retour à une couverture uniforme (qui laissait
// apparaître la répétition du bruit 3D) ; au-delà de 20 km, la symétrie ne se remarque pas.
vec4 SampleWeather(vec2 xz) {
    vec2 uv = (xz - uWeatherOriginRel) / uWeatherExtent + 0.5;
    uv = 1.0 - abs(mod(uv, 2.0) - 1.0);
    return textureLod(uWeatherMap, uv, 0.0);
}

// Gradient de densité selon la hauteur et le type (Nubis 2017) : remap(h, a, b, 0, 1) × remap(h, c, d, 1, 0),
// points interpolés entre stratus (0), stratocumulus (0,5) et cumulus (1). La décroissance progressive
// vers le haut, combinée à l'érosion par la couverture, arrondit les sommets (dômes, pas d'enclume).
float DensityHeightGradient(float h, float type) {
    const vec4 kStratus       = vec4(0.00, 0.08, 0.15, 0.28);
    const vec4 kStratocumulus = vec4(0.00, 0.12, 0.30, 0.60);
    const vec4 kCumulus       = vec4(0.00, 0.08, 0.40, 1.00);
    vec4 g = type < 0.5 ? mix(kStratus, kStratocumulus, type * 2.0) : mix(kStratocumulus, kCumulus, type * 2.0 - 1.0);
    return clamp(Remap(h, g.x, g.y, 0.0, 1.0), 0.0, 1.0) * clamp(Remap(h, g.z, g.w, 1.0, 0.0), 0.0, 1.0);
}

// Densité [0, 1+] ; `detail` = false pour les marches secondaires (lumière), plus économes.
float CloudDensity(vec3 pRel, float alt, bool detail) {
    float h = (alt - uCloudBottom) / (uCloudTop - uCloudBottom);
    if (h <= 0.0 || h >= 1.0) return 0.0;
    vec3 p = vec3(pRel.x, alt, pRel.z);
    p.xz += h * uSkew;  // le vent incline les nuages avec l'altitude

    vec4 w = SampleWeather(p.xz);
    float coverage = w.r;
    if (coverage <= 0.01) return 0.0;
    // Enclume (Nubis) : seulement pour les cellules orageuses (forte pluie + cumulus), le haut s'étale.
    float anvil = smoothstep(0.5, 1.0, w.g) * smoothstep(0.7, 1.0, w.b);
    coverage = pow(coverage, clamp(Remap(h, 0.7, 0.8, 1.0, mix(1.0, 0.5, anvil)), mix(1.0, 0.5, anvil), 1.0));

    vec4  shape = textureLod(uShapeNoise, (p + uShapeOffset) / uShapeScale, 0.0);
    float fbm   = shape.g * 0.625 + shape.b * 0.25 + shape.a * 0.125;
    float base  = Remap(shape.r, fbm - 1.0, 1.0, 0.0, 1.0);
    base  = smoothstep(0.30, 0.98, base);  // égalisation (répartition mesurée de la texture)
    base *= DensityHeightGradient(h, w.b);
    base  = clamp(Remap(base, 1.0 - coverage, 1.0, 0.0, 1.0), 0.0, 1.0) * coverage;
    if (base <= 0.0) return 0.0;

    if (detail && uDetailStrength > 0.0) {
        // Curl : torsion des coordonnées du détail, plus forte à la base (volutes, bords effilochés).
        vec3 pd   = (p + uDetailOffset) / uDetailScale;
        vec3 curl = textureLod(uDetailNoise, pd * 0.25, 0.0).gba * 2.0 - 1.0;
        pd += curl * uCurlStrength * (1.0 - h);
        float dfb   = textureLod(uDetailNoise, pd, 0.0).r;
        float erode = mix(dfb, 1.0 - dfb, clamp(h * 10.0, 0.0, 1.0));  // volutes en bas, bourgeons en haut
        base = clamp(Remap(base, erode * uDetailStrength * 0.4, 1.0, 0.0, 1.0), 0.0, 1.0);
    }
    return base * mix(0.6, 1.6, w.a);
}

#endif  // ENV_CLOUD_DENSITY_GLSL
