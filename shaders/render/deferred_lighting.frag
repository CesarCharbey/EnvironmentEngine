#version 460 core
//  shaders/render/deferred_lighting.frag                               [DEV 2]
//  Résolution différée PBR -> scene.color (HDR linéaire, cd/m²).
//
//  * Ciel (depth == 0) : sky-view LUT (Hillaire 2020) + disque solaire (assombrissement centre-bord)
//    + disque lunaire éclairé comme une sphère (phase) + lumière cendrée.
//  * Géométrie : Cook-Torrance GGX (D_GGX, V_SmithGGXCorrelated, F_Schlick) + Lambert, soleil et lune
//    atténués par la transmittance atmosphérique, l'ombre de la planète et l'ombre des nuages ;
//    ambiant IBL split-sum (ou repli sur la sky-view) ; météo de surface ; brouillard de hauteur.
//  * Unités physiques : éclairements en lux, luminances en cd/m².
//    Tant que scene.display n'est pas publiée (pas encore de passe de tonemap), une exposition
//    automatique provisoire + ACES encode directement scene.color pour le Viewport.
#include "common/global_ubo.glsl"
#include "render/atmo_common.glsl"

in vec2 vUv;
layout(location = 0) out vec4 oColor;

layout(binding = TEX_GBufferAlbedo)     uniform sampler2D   uGAlbedo;    // SRGB8_ALPHA8 : rgb albédo, a AO
layout(binding = TEX_GBufferNormal)     uniform sampler2D   uGNormal;    // RGBA16F : xyz normale monde
layout(binding = TEX_GBufferMaterial)   uniform sampler2D   uGMaterial;  // RGBA8 : r rugosité, g métal, b humidité, a id
layout(binding = TEX_GBufferDepth)      uniform sampler2D   uGDepth;     // DEPTH32F reverse-Z
layout(binding = TEX_AtmoTransmittance) uniform sampler2D   uTransmittanceLUT;
layout(binding = TEX_AtmoSkyView)       uniform sampler2D   uSkyViewLUT;
layout(binding = TEX_IblIrradiance)     uniform samplerCube uIblIrradiance;   // convention : radiance diffuse (E / π)
layout(binding = TEX_IblPrefiltered)    uniform samplerCube uIblPrefiltered;  // mip = rugosité perceptuelle
layout(binding = TEX_IblBrdfLut)        uniform sampler2D   uIblBrdfLut;      // (NoV, rugosité) -> (scale, bias)
layout(binding = TEX_CloudShadow)       uniform sampler2D   uCloudShadow;     // transmittance des nuages

uniform bool  uGBufferAvailable;
uniform bool  uAtmoAvailable;
uniform bool  uIblAvailable;
uniform bool  uCloudShadowAvailable;
uniform bool  uApplyFallbackExposure;
uniform bool  uSpecularOcclusion;
uniform float uSunScale;
uniform float uMoonScale;
uniform float uIblScale;
uniform int   uDebugView;              // 0 Lit, 1 Albedo, 2 Normal, 3 Roughness, 4 Metal, 5 AO, 6 Depth, 7 IBL only, 8 Sun only
uniform float uExposureCompensation;   // EV ajoutés à l'exposition automatique provisoire
uniform float uFogHeightFalloff;       // 1/m : décroissance verticale du brouillard
uniform float uCloudLayerAltitude;     // m : plan de projection de l'ombre des nuages
uniform float uCloudShadowExtent;      // m : emprise de clouds.shadowMap, centrée sur la caméra

const float kHalfMax = 65000.0;  // borne du RGBA16F de scene.color

// =============================================================================
// Utilitaires
// =============================================================================
float Luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

vec3 LinearToSrgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

// Angle (rad) entre deux directions unitaires, précis pour les petits angles (disques des astres).
float AngleBetween(vec3 a, vec3 b) { return 2.0 * asin(clamp(length(a - b) * 0.5, 0.0, 1.0)); }

// Taille angulaire d'un pixel (rad) : sert d'anticrénelage aux bords des disques, sans dérivées.
float PixelAngle() { return 2.0 * uEnv.cameraParams.z * uEnv.viewportSize.w; }

// =============================================================================
// Atmosphère
// =============================================================================
bool AtmoValid() {
    // L'UBO AtmosphereParams est lié par CelestialAtmosphereSystem : on vérifie qu'il est cohérent.
    return uAtmoAvailable && uAtmo.uRadiiAndScaleHeights.x > 0.0
        && uAtmo.uRadiiAndScaleHeights.y > uAtmo.uRadiiAndScaleHeights.x;
}

// Luminance du ciel (cd/m², sans les disques) dans la direction dir.
vec3 SkyLuminance(vec3 dir) {
    if (!AtmoValid()) return vec3(0.0);
    return texture(uSkyViewLUT, SkyViewDirToUV(dir, AtmoViewRadiusKm(), uAtmo.uRadiiAndScaleHeights.x)).rgb;
}

// Idem, direction remontée au-dessus de l'horizon : sous l'horizon la LUT ne contient que l'air
// entre la caméra et le sol, pas le sol lui-même. Utilisé pour l'ambiant et le brouillard.
vec3 SkyLuminanceAbove(vec3 dir) {
    dir.y = max(dir.y, 0.03);
    return SkyLuminance(normalize(dir));
}

// Transmittance atmosphère × ombre de la planète, d'un point d'altitude altM (m) vers un astre.
vec3 LightTransmittance(float altM, vec3 lightDir) {
    if (!AtmoValid()) return vec3(smoothstep(-0.02, 0.02, lightDir.y));
    float Rb = uAtmo.uRadiiAndScaleHeights.x;
    float Rt = uAtmo.uRadiiAndScaleHeights.y;
    float r  = clamp(Rb + max(altM * 0.001, PLANET_RADIUS_OFFSET), Rb, Rt);
    vec3 pos = vec3(0.0, r, 0.0);  // repère local planète : +Y = zénith
    return texture(uTransmittanceLUT, TransmittanceParamsToUV(r, lightDir.y, Rb, Rt)).rgb
         * PlanetShadow(pos, lightDir, Rb);
}

// Éclairements au sommet de l'atmosphère (lux). La lune renvoie la lumière du soleil.
vec3 SunIlluminanceTOA()  { return uEnv.sunColorAngular.rgb * uEnv.sunDirIlluminance.w; }
vec3 MoonIlluminanceTOA() { return uEnv.sunColorAngular.rgb * uEnv.moonParams.x; }

// =============================================================================
// Disques des astres
// =============================================================================
// Soleil : luminance = E / Ω, assombrissement centre-bord I(μ) = 1 - u (1 - μ), normalisé
// pour conserver l'éclairement total (moyenne sur le disque = 1 - u / 3).
vec3 SunDisk(vec3 rayDir) {
    vec3  s    = normalize(envSunDirection());
    float angR = max(uEnv.sunColorAngular.a, 1e-4);
    float x    = AngleBetween(rayDir, s) / angR;  // 0 au centre, 1 au bord
    float aa   = PixelAngle() / angR;
    float coverage = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, x);
    if (coverage <= 0.0) return vec3(0.0);

    const float u = 0.6;
    float mu = sqrt(max(0.0, 1.0 - x * x));
    float limb = (1.0 - u * (1.0 - mu)) / (1.0 - u / 3.0);
    float omega = 2.0 * PI * (1.0 - cos(angR));

    return SunIlluminanceTOA() / omega * limb * coverage
         * LightTransmittance(envCameraPosition().y, s);
}

// Lune : sphère lambertienne. La direction du soleil vue depuis la lune est reconstruite à partir
// de la phase (élongation ψ = 2π·phase : 0 = nouvelle lune, π = pleine lune), dans le plan
// lune-soleil réel : le terminateur est donc orienté vers le vrai soleil ET respecte la phase,
// même quand les sliders azimut / phase sont réglés indépendamment.
vec3 MoonDisk(vec3 rayDir) {
    vec3  m    = normalize(envMoonDirection());
    float angR = max(uEnv.moonParams.y, 1e-4);
    float x    = AngleBetween(rayDir, m) / angR;
    float aa   = PixelAngle() / angR;
    float coverage = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, x);
    if (coverage <= 0.0) return vec3(0.0);

    // Repère tangent du disque
    vec3 refUp = abs(m.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(refUp, m));
    vec3 B = cross(m, T);
    vec3 perp = rayDir - m * dot(rayDir, m);
    vec2 q = vec2(dot(perp, T), dot(perp, B)) / sin(angR);  // position sur le disque, |q| <= 1
    float q2 = min(dot(q, q), 1.0);
    vec3 n = normalize(q.x * T + q.y * B - sqrt(1.0 - q2) * m);  // normale de l'hémisphère visible

    float psi = 2.0 * PI * envMoonPhase();
    vec3 s = normalize(envSunDirection());
    vec3 t = s - m * dot(s, m);
    t = dot(t, t) > 1e-8 ? normalize(t) : T;
    vec3 sunFromMoon = cos(psi) * m + sin(psi) * t;
    float ndl = max(dot(n, sunFromMoon), 0.0);

    // moonParams.x = éclairement lunaire au sol (phase incluse côté CPU, facteur 0.5 (1 - cos ψ)).
    // On remonte à l'éclairement de pleine lune, réparti sur le disque (sphère lambertienne vue
    // pleine : <n·l> = 2/3). La lumière cendrée ajoute une fraction de cette luminance partout.
    float phaseFactor = 0.5 * (1.0 - cos(psi));
    float eFull = uEnv.moonParams.x / max(phaseFactor, 1e-4);
    float omega = 2.0 * PI * (1.0 - cos(angR));
    float shade = ndl * 1.5 + uEnv.moonParams.z;

    return uEnv.sunColorAngular.rgb * (eFull / omega) * shade * coverage
         * LightTransmittance(envCameraPosition().y, m);
}

vec3 ShadeSky(vec3 rayDir) {
    return SkyLuminance(rayDir) + SunDisk(rayDir) * uSunScale + MoonDisk(rayDir) * uMoonScale;
}

// =============================================================================
// BRDF
// =============================================================================
float D_GGX(float NoH, float a) {
    float a2 = a * a;
    float f = (NoH * a2 - NoH) * NoH + 1.0;
    return a2 / (PI * f * f);
}

float V_SmithGGXCorrelated(float NoV, float NoL, float a) {
    float a2 = a * a;
    float gv = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    float gl = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-5);
}

vec3 F_Schlick(vec3 f0, float VoH) {
    float f = pow(1.0 - VoH, 5.0);
    return f0 + (vec3(1.0) - f0) * f;
}

// Approximation analytique de la LUT split-sum (Karis 2014, mobile) : utilisée sans IBL.
vec2 EnvBrdfApprox(float NoV, float roughness) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
    return vec2(-1.04, 1.04) * a004 + r.zw;
}

struct Surface {
    vec3  albedo;
    vec3  diffuseColor;
    vec3  f0;
    float roughness;  // perceptuelle
    float alpha;      // roughness²
    float metal;
    float ao;
    vec3  N;
};

// Luminance réfléchie (cd/m²) pour un astre d'éclairement E (lux) dans la direction L.
vec3 DirectLight(Surface s, vec3 V, vec3 L, vec3 E) {
    float NoL = dot(s.N, L);
    if (NoL <= 0.0) return vec3(0.0);
    vec3  H   = normalize(V + L);
    float NoV = max(dot(s.N, V), 1e-4);
    float NoH = clamp(dot(s.N, H), 0.0, 1.0);
    float VoH = clamp(dot(V, H), 0.0, 1.0);

    vec3 F = F_Schlick(s.f0, VoH);
    vec3 specular = D_GGX(NoH, s.alpha) * V_SmithGGXCorrelated(NoV, NoL, s.alpha) * F;
    vec3 diffuse  = (vec3(1.0) - F) * s.diffuseColor / PI;
    return (diffuse + specular) * E * NoL;
}

// Ombre des nuages. Convention PROVISOIRE (à aligner avec VolumetricGodRaysSystem quand il publiera
// clouds.shadowMap) : projection le long du soleil sur un plan d'altitude uCloudLayerAltitude, carte
// centrée sur la caméra, emprise uCloudShadowExtent mètres.
float CloudShadow(vec3 P, vec3 L) {
    if (!uCloudShadowAvailable) return 1.0;
    float t = (uCloudLayerAltitude - P.y) / max(L.y, 0.05);
    vec3 pc = P + L * max(t, 0.0);
    vec2 uv = (pc.xz - envCameraPosition().xz) / uCloudShadowExtent + 0.5;
    return texture(uCloudShadow, uv).r;
}

// Éclairage ambiant : radiance diffuse (E/π) et radiance spéculaire déjà pondérée par la BRDF.
void Ambient(Surface s, vec3 V, out vec3 diffuse, out vec3 specular) {
    float NoV = max(dot(s.N, V), 1e-4);
    vec3  R   = reflect(-V, s.N);

    if (uIblAvailable) {
        // Split-sum (Karis 2013)
        vec3 irradiance = texture(uIblIrradiance, s.N).rgb;
        float maxLod = float(textureQueryLevels(uIblPrefiltered) - 1);
        vec3 prefiltered = textureLod(uIblPrefiltered, R, s.roughness * maxLod).rgb;
        vec2 brdf = texture(uIblBrdfLut, vec2(NoV, s.roughness)).rg;
        diffuse  = irradiance * s.diffuseColor;
        specular = prefiltered * (s.f0 * brdf.x + brdf.y);
    } else {
        // Repli physique sur la sky-view (mêmes unités cd/m² que le ciel affiché).
        // Diffus : ciel vu par la normale (direction entre N et le zénith ≈ moyenne pondérée par cos)
        // + rebond du sol (albédo de l'atmosphère) pour la partie de l'hémisphère tournée vers le bas.
        vec3  up     = vec3(0.0, 1.0, 0.0);
        float skyVis = 0.5 + 0.5 * s.N.y;
        vec3  skyL   = SkyLuminanceAbove(normalize(s.N + up));
        vec3  sunDir = normalize(envSunDirection());
        vec3  groundE = SunIlluminanceTOA() * LightTransmittance(0.0, sunDir) * max(sunDir.y, 0.0) * uSunScale
                      + PI * SkyLuminanceAbove(up);
        vec3  groundL = AtmoValid() ? uAtmo.uGroundAlbedoAndTurb.xyz * groundE / PI : vec3(0.0);
        vec3  diffuseRadiance = skyL * skyVis + groundL * (1.0 - skyVis);

        // Spéculaire : ciel dans la direction réfléchie, flouté vers le diffus quand la rugosité monte.
        vec3 specL = mix(SkyLuminanceAbove(R), diffuseRadiance, s.alpha);
        vec2 ab = EnvBrdfApprox(NoV, s.roughness);
        diffuse  = diffuseRadiance * s.diffuseColor;
        specular = specL * (s.f0 * ab.x + ab.y);
    }

    diffuse *= s.ao;
    if (uSpecularOcclusion) {
        // Lagarde & de Rousiers 2014 (Frostbite)
        float so = clamp(pow(NoV + s.ao, exp2(-16.0 * s.alpha - 1.0)) - 1.0 + s.ao, 0.0, 1.0);
        specular *= so;
    } else {
        specular *= s.ao;
    }
}

// Brouillard de hauteur : densité σ(y) = σ0 exp(-k y), intégrée analytiquement le long du rayon.
vec3 ApplyHeightFog(vec3 color, vec3 P, vec3 rayDir, float dist) {
    float sigma0 = uEnv.weatherExtra.z;  // 1/m au niveau y = 0
    if (sigma0 <= 0.0) return color;
    float k    = max(uFogHeightFalloff, 1e-5);
    float camY = envCameraPosition().y;
    float kd   = k * (P.y - camY);
    float shape = abs(kd) > 1e-4 ? (1.0 - exp(-kd)) / kd : 1.0;
    float opticalDepth = sigma0 * exp(clamp(-k * camY, -80.0, 80.0)) * dist * shape;
    float fog = 1.0 - exp(-max(opticalDepth, 0.0));
    return mix(color, SkyLuminanceAbove(rayDir), fog);
}

// =============================================================================
// Exposition provisoire (tant que scene.display n'existe pas)
// =============================================================================
float FallbackExposure() {
    // Luminance moyenne estimée sur le ciel : zénith + 4 directions à 20° d'élévation.
    vec3 acc = SkyLuminance(vec3(0.0, 1.0, 0.0));
    const float ce = 0.9397, se = 0.3420;  // cos / sin 20°
    for (int i = 0; i < 4; ++i) {
        float az = (float(i) + 0.5) * 0.5 * PI;
        acc += SkyLuminance(vec3(ce * sin(az), se, -ce * cos(az)));
    }
    float lAvg = Luminance(acc / 5.0);
    // EV100 (S = 100, K = 12.5), borné : nuit ≈ -2, plein soleil ≈ 15.
    float ev100 = clamp(log2(max(lAvg, 1e-6) * 100.0 / 12.5), -2.0, 16.0) - uExposureCompensation;
    return 1.0 / (1.2 * exp2(ev100));
}

vec3 AcesFilmic(vec3 x) {  // Narkowicz 2015
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

// =============================================================================
void main() {
    float depth = uGBufferAvailable ? texture(uGDepth, vUv).r : 0.0;
    bool  isDataView = uDebugView >= 1 && uDebugView <= 6;
    vec3  rayDir = envViewRayWorld(vUv);
    vec3  color;

    if (envIsSky(depth)) {
        color = isDataView ? vec3(0.0) : (uDebugView == 0 ? ShadeSky(rayDir) : SkyLuminance(rayDir));
    } else {
        // --- Lecture du G-Buffer ---
        vec3 P = envWorldPosFromDepth(vUv, depth);  // relative à l'origine de rendu (y absolu)
        vec3 V = normalize(envCameraPosition() - P);
        vec4 gA = texture(uGAlbedo, vUv);           // sRGB -> linéaire par le sampler
        vec4 gM = texture(uGMaterial, vUv);

        Surface s;
        s.albedo    = gA.rgb;
        s.ao        = gA.a;
        s.N         = normalize(texture(uGNormal, vUv).xyz);
        s.roughness = gM.r;
        s.metal     = gM.g;

        // --- Météo de surface ---
        float snow = clamp(envSnow() * smoothstep(0.5, 0.9, s.N.y), 0.0, 1.0);
        float wet  = clamp(max(gM.b, uEnv.weatherExtra.w), 0.0, 1.0) * (1.0 - snow);
        float porous = 1.0 - s.metal;
        s.albedo    *= mix(1.0, 0.55, wet * porous);        // un matériau poreux mouillé s'assombrit
        s.roughness  = mix(s.roughness, 0.05, wet);         // film d'eau quasi miroir
        s.albedo     = mix(s.albedo, vec3(0.9), snow);
        s.roughness  = mix(s.roughness, 0.6, snow);
        s.metal      = mix(s.metal, 0.0, snow);

        s.roughness    = clamp(s.roughness, 0.045, 1.0);
        s.alpha        = s.roughness * s.roughness;
        s.diffuseColor = s.albedo * (1.0 - s.metal);
        s.f0           = mix(vec3(0.04), s.albedo, s.metal);
        s.f0           = mix(s.f0, vec3(0.02), wet * porous);  // F0 de l'eau

        // --- Lumières directes ---
        vec3 sunDir  = normalize(envSunDirection());
        vec3 moonDir = normalize(envMoonDirection());
        float cloud  = CloudShadow(P, sunDir);
        vec3 sunE  = SunIlluminanceTOA()  * LightTransmittance(P.y, sunDir)  * cloud * uSunScale;
        vec3 moonE = MoonIlluminanceTOA() * LightTransmittance(P.y, moonDir) * CloudShadow(P, moonDir) * uMoonScale;
        vec3 sunL  = DirectLight(s, V, sunDir, sunE);
        vec3 moonL = DirectLight(s, V, moonDir, moonE);

        // --- Ambiant ---
        vec3 ambD, ambS;
        Ambient(s, V, ambD, ambS);
        vec3 ambient = (ambD + ambS) * uIblScale;

        float dist = length(envCameraPosition() - P);
        switch (uDebugView) {
            case 1:  color = gA.rgb; break;
            case 2:  color = s.N * 0.5 + 0.5; break;
            case 3:  color = vec3(gM.r); break;
            case 4:  color = vec3(gM.g); break;
            case 5:  color = vec3(gA.a); break;
            case 6:  color = envTurbo(envLinearDepth(depth) / 1000.0); break;
            case 7:  color = ambient; break;
            case 8:  color = sunL; break;
            default: color = ApplyHeightFog(sunL + moonL + ambient, P, rayDir, dist); break;
        }
    }

    if (isDataView) {
        // Données brutes : pas d'exposition. Encodées pour le Viewport s'il affiche scene.color tel quel.
        oColor = vec4(uApplyFallbackExposure ? LinearToSrgb(color) : color, 1.0);
    } else if (uApplyFallbackExposure) {
        oColor = vec4(LinearToSrgb(AcesFilmic(color * FallbackExposure())), 1.0);
    } else {
        oColor = vec4(min(color, vec3(kHalfMax)), 1.0);  // le disque solaire dépasse le RGBA16F
    }
}
