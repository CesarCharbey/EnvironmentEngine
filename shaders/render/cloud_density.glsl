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

layout(std140, binding = UBO_VolumetricParams) uniform CloudParamsBlock {
    vec4 layerAndScales;
    vec4 detailAndSkew;
    vec4 shapeOffset;
    vec4 detailOffset;
    vec4 weatherAndPhase;
    vec4 lightingAndDist;
    vec4 cirrus0;
    vec4 cirrusAndLightning;
    vec4 lightningPos;
} uCloud;

#define uCloudBottom        (uCloud.layerAndScales.x)
#define uCloudTop           (uCloud.layerAndScales.y)
#define uShapeScale         (uCloud.layerAndScales.z)
#define uDetailScale        (uCloud.layerAndScales.w)
#define uDetailStrength     (uCloud.detailAndSkew.x)
#define uCurlStrength       (uCloud.detailAndSkew.y)
#define uSkew               (uCloud.detailAndSkew.zw)
#define uShapeOffset        (uCloud.shapeOffset.xyz)
#define uExtinction         (uCloud.shapeOffset.w)
#define uDetailOffset       (uCloud.detailOffset.xyz)
#define uWeatherExtent      (uCloud.detailOffset.w)
#define uWeatherOriginRel   (uCloud.weatherAndPhase.xy)
#define uHgForward          (uCloud.weatherAndPhase.z)
#define uHgBackward         (uCloud.weatherAndPhase.w)
#define uSilver             (uCloud.lightingAndDist.x)
#define uPowder             (uCloud.lightingAndDist.y)
#define uMaxDistance        (uCloud.lightingAndDist.z)
#define uFadeDistance       (uCloud.lightingAndDist.w)
#define uCirrusAltitude     (uCloud.cirrus0.x)
#define uCirrusCoverage     (uCloud.cirrus0.y)
#define uCirrusType         (uCloud.cirrus0.z)
#define uCirrusScale        (uCloud.cirrus0.w)
#define uCirrusOffset       (uCloud.cirrusAndLightning.xy)
#define uLightningIntensity (uCloud.cirrusAndLightning.z)
#define uLightningPos       (uCloud.lightningPos.xyz)

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
    const vec4 kStratus       = vec4(0.00, 0.14, 0.24, 0.40);
    const vec4 kStratocumulus = vec4(0.00, 0.18, 0.38, 0.68);
    const vec4 kCumulus       = vec4(0.00, 0.16, 0.45, 1.00);
    vec4 g = type < 0.5 ? mix(kStratus, kStratocumulus, type * 2.0) : mix(kStratocumulus, kCumulus, type * 2.0 - 1.0);
    return clamp(Remap(h, g.x, g.y, 0.0, 1.0), 0.0, 1.0) * clamp(Remap(h, g.z, g.w, 1.0, 0.0), 0.0, 1.0);
}

// Densité [0, 1+] avec sortie du profil dimensionnel (Nubis Evolved 2022) ;
// `detail` = false pour les marches secondaires (lumière), plus économes ;
// `stepDt` > 0 adapte le LOD du détail à la taille du pas (supprime le dithering en fort détail).
float CloudDensity(vec3 pRel, float alt, bool detail, float stepDt, out float dimProfile) {
    dimProfile = 0.0;
    float h = (alt - uCloudBottom) / (uCloudTop - uCloudBottom);
    if (h <= 0.0 || h >= 1.0) return 0.0;
    vec3 p = vec3(pRel.x, alt, pRel.z);
    p.xz -= h * uSkew;  // le vent incline le sommet des nuages vers l'aval

    vec4 w = SampleWeather(p.xz);
    float coverage = w.r;
    if (coverage <= 0.01) return 0.0;
    // Enclume (Nubis) : seulement pour les cellules orageuses (forte pluie + cumulus), le haut s'étale.
    float anvil = smoothstep(0.5, 1.0, w.g) * smoothstep(0.7, 1.0, w.b);
    coverage = pow(coverage, clamp(Remap(h, 0.7, 0.8, 1.0, mix(1.0, 0.5, anvil)), mix(1.0, 0.5, anvil), 1.0));

    // Modulation macro incommensurable (rotation 37° + échelle 2,37×) : casse toute répétition 3D sous
    // forte couverture et fait onduler naturellement la hauteur de base de chaque nuage.
    const mat2 kMacroRot = mat2(0.7986355, -0.6018150, 0.6018150, 0.7986355);
    vec3  pShape   = p + uShapeOffset;
    vec2  macroXZ  = kMacroRot * (pShape.xz / (uShapeScale * 2.37));
    float macroMod = textureLod(uShapeNoise, vec3(macroXZ.x, h * 0.35, macroXZ.y), 1.0).r;

    float baseLift = clamp(0.18 * (1.0 - coverage) + 0.14 * (macroMod - 0.5) + 0.05 * w.b, 0.0, 0.28);
    float hLocal   = (h - baseLift) / max(1.0 - baseLift, 0.2);
    if (hLocal <= 0.0 || hLocal >= 1.0) return 0.0;

    // Profil dimensionnel (Nubis Evolved 2022) : early-out avant de lire le bruit 3D 128^3.
    float heightGrad = DensityHeightGradient(hLocal, w.b);
    dimProfile = heightGrad * coverage;
    if (dimProfile <= 0.01) return 0.0;

    vec2  warpXZ = vec2(macroMod - 0.5, w.b - 0.5) * (0.14 * uShapeScale);
    vec3  sp     = vec3(pShape.x + warpXZ.x, (alt - uCloudBottom) * 1.65 + uShapeOffset.y, pShape.z + warpXZ.y) / uShapeScale;
    vec4  shape  = textureLod(uShapeNoise, sp, 0.0);
    float fbm    = shape.g * 0.625 + shape.b * 0.25 + shape.a * 0.125;
    float base   = Remap(shape.r, fbm - 1.0, 1.0, 0.0, 1.0);
    base  = smoothstep(0.28, 0.98, base) * mix(0.80, 1.20, macroMod);
    base *= heightGrad;
    base  = clamp(Remap(base, 1.0 - coverage, 1.0, 0.0, 1.0), 0.0, 1.0) * coverage;
    if (base <= 0.0) return 0.0;

    if (detail && uDetailStrength > 0.0) {
        // Filtrage Nyquist selon le pas de marche : atténue le détail sous-échantillonné au loin (zéro dithering).
        float detailFade = stepDt > 0.0 ? (1.0 - smoothstep(0.08 * uDetailScale, 0.28 * uDetailScale, stepDt)) : 1.0;
        if (detailFade > 0.01) {
            float detailLod = stepDt > 0.0 ? clamp(log2(max(stepDt / max(uDetailScale * 0.055, 1.0), 1.0)), 0.0, 3.0) : 0.0;
            vec3 pd   = (p + uDetailOffset) / uDetailScale;
            vec3 curl = textureLod(uDetailNoise, pd * 0.25, min(detailLod, 2.0)).gba * 2.0 - 1.0;
            pd += curl * uCurlStrength * (1.0 - hLocal) * detailFade;
            float dfb   = textureLod(uDetailNoise, pd, detailLod).r;
            float erode = mix(dfb, 1.0 - dfb, clamp(hLocal * 8.0, 0.0, 1.0));  // volutes en bas, bourgeons en haut
            base = clamp(Remap(base, erode * uDetailStrength * detailFade * 0.35, 1.0, 0.0, 1.0), 0.0, 1.0);
        }
    }
    return base * mix(0.6, 1.6, w.a);
}

// Surcharges sans stepDt / sans profil dimensionnel (cloud_shadow.comp, marche de lumière, IBL).
float CloudDensity(vec3 pRel, float alt, bool detail, out float dimProfile) {
    return CloudDensity(pRel, alt, detail, 0.0, dimProfile);
}

float CloudDensity(vec3 pRel, float alt, bool detail) {
    float unusedProfile;
    return CloudDensity(pRel, alt, detail, 0.0, unusedProfile);
}

#endif  // ENV_CLOUD_DENSITY_GLSL
