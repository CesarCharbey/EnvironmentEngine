//  shaders/render/surface_lighting.glsl                                [DEV 2]
//  Éclairage d'une surface opaque ou de l'eau, partagé par la résolution différée
//  (deferred_lighting.frag) et le rendu de l'eau (water_surface.frag) : même BRDF, mêmes ombres
//  (relief, contact, cône VXGI, nuages), même ambiant (IBL split-sum / VXGI) et même brouillard.
//  Les uniforms sont renseignés côté C++ par SetSurfaceLightingUniforms (LightingShared.hpp).
//
//  Le shader qui inclut ce fichier doit avoir inclus "common/global_ubo.glsl",
//  "render/atmo_common.glsl", "render/sky_lighting.glsl" et "render/voxel_common.glsl".
#ifndef ENV_SURFACE_LIGHTING_GLSL
#define ENV_SURFACE_LIGHTING_GLSL

#include "render/color_utils.glsl"

layout(binding = TEX_GBufferDepth)      uniform sampler2D   uGDepth;     // DEPTH32F reverse-Z
layout(binding = TEX_IblIrradiance)     uniform samplerCube uIblIrradiance;   // convention : radiance diffuse (E / π)
layout(binding = TEX_IblPrefiltered)    uniform samplerCube uIblPrefiltered;  // mip = rugosité perceptuelle
layout(binding = TEX_IblBrdfLut)        uniform sampler2D   uIblBrdfLut;      // (NoV, rugosité) -> (scale, bias)
layout(binding = TEX_CloudShadow)       uniform sampler2D   uCloudShadow;     // transmittance des nuages
layout(binding = TEX_HeightNormal)      uniform sampler2D   uFocusHeight;     // world.focus.heightNormal (R = hauteur)

uniform bool  uGBufferAvailable;
uniform bool  uIblAvailable;
uniform bool  uCloudShadowAvailable;
uniform bool  uSpecularOcclusion;
uniform float uSunScale;
uniform float uMoonScale;
uniform float uIblScale;
// VXGI (ADR-008)
uniform bool  uVoxelAvailable;         // voxel.grid publiée (VoxelIlluminationSystem)
uniform bool  uVoxelShadows;           // ombre lointaine par cône vers le soleil / la lune
uniform bool  uVoxelGI;                // VXAO + rebond diffus (5 cônes)
uniform float uGiIntensity;            // multiplicateur de la lumière indirecte des voxels
uniform float uVoxelShadowSoftness;    // tangente du demi-angle du cône d'ombre (pénombre)
// Ombres portées
uniform bool  uTerrainShadows;         // heightfield du chunk focus
uniform bool  uContactShadows;         // espace écran (gbuffer.depth)
uniform bool  uFocusAvailable;         // world.focus.heightNormal publié ET chunk retrouvé dans l'ECS
uniform vec3  uFocusChunkOrigin;       // coin du chunk focus relatif à l'origine de rendu (m)
uniform float uShadowPenumbra;         // demi-angle (rad) de la pénombre des ombres du relief
uniform float uFogHeightFalloff;       // 1/m : décroissance verticale du brouillard
uniform bool  uFogShadows;             // ombre volumétrique du brouillard (grille VXGI)
uniform float uCloudLayerAltitude;     // m : plan de projection de l'ombre des nuages
uniform float uCloudShadowExtent;      // m : emprise de clouds.shadowMap, centrée sur la caméra

const float kHalfMax = 65000.0;  // borne du RGBA16F de scene.color

// =============================================================================
// Utilitaires
// =============================================================================

// Angle (rad) entre deux directions unitaires, précis pour les petits angles (disques des astres).
float AngleBetween(vec3 a, vec3 b) { return 2.0 * asin(clamp(length(a - b) * 0.5, 0.0, 1.0)); }

// Taille angulaire d'un pixel (rad) : sert d'anticrénelage aux bords des disques, sans dérivées.
float PixelAngle() { return 2.0 * uEnv.cameraParams.z * uEnv.viewportSize.w; }


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

// =============================================================================
// Ombres portées : relief (heightfield du chunk focus) + contact (espace écran)
// =============================================================================
// Bruit par pixel stable (Jimenez 2014) : décale les pas de marche pour casser les bandes.
float InterleavedGradientNoise(vec2 pixel) {
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

// a) Ombre douce du relief : marche le long de L dans la carte de hauteur du chunk focus
//    (world.focus.heightNormal, R = hauteur absolue). À chaque pas, la "hauteur libre" du rayon
//    au-dessus du terrain divisée par la distance donne l'angle sous lequel le relief masque l'astre ;
//    un smoothstep sur ±uShadowPenumbra produit la pénombre. Hors du chunk : on ne sait pas -> éclairé.
float HeightfieldShadow(vec3 P, vec3 L, float noise) {
    if (!uFocusAvailable || !uTerrainShadows || L.y <= 0.0) return 1.0;
    const int   kSteps   = 20;
    const float kMaxDist = 400.0;   // m (diagonale d'un chunk ≈ 362 m)
    const float kBias    = 0.35;    // m : écart grille rendue / carte bilinéaire
    float vis = 1.0;
    for (int i = 0; i < kSteps; ++i) {
        float f  = (float(i) + noise) / float(kSteps);
        float t  = 1.0 + kMaxDist * f * f;                 // pas resserrés près du point (contact)
        vec3  p  = P + L * t;
        vec2  uv = (p.xz - uFocusChunkOrigin.xz) / CHUNK_SIZE_METERS;
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) break;
        float h  = textureLod(uFocusHeight, uv, 0.0).r;
        float clearance = (p.y + kBias - h) / t;           // tangente de l'angle libre
        vis = min(vis, smoothstep(-uShadowPenumbra, uShadowPenumbra, clearance));
        if (vis <= 0.001) break;
    }
    return vis;
}

// b) Ombre de contact : 8 pas courts en espace écran contre gbuffer.depth (petits reliefs, pied de la
//    végétation, objets non présents dans la carte de hauteur). Longueur et épaisseur croissent avec
//    la distance pour rester à l'échelle de quelques pixels.
float ContactShadow(vec3 P, vec3 N, vec3 L, float dist, float noise) {
    if (!uGBufferAvailable || !uContactShadows) return 1.0;
    const int kSteps = 8;
    float len       = clamp(dist * 0.02, 0.4, 6.0);
    float thickness = clamp(dist * 0.01, 0.25, 3.0);
    // Anti-acné : le maillage rendu (grille de 2 m) et la normale lisse de la carte de hauteur divergent
    // de quelques cm, et un pixel couvre plus de terrain avec la distance -> départ décalé le long de la
    // normale et seuil de détection proportionnels à la distance.
    float bias = 0.05 + 0.004 * dist;
    vec3  start = P + N * bias;
    for (int i = 0; i < kSteps; ++i) {
        float t    = len * (float(i) + 0.5 + 0.5 * noise) / float(kSteps);
        vec4  clip = uEnv.viewProj * vec4(start + L * t, 1.0);
        if (clip.w <= 0.0) break;
        vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) break;
        float sceneDepth = textureLod(uGDepth, uv, 0.0).r;
        if (envIsSky(sceneDepth)) continue;
        float diff = clip.w - envLinearDepth(sceneDepth);  // > 0 : la géométrie est devant le rayon
        if (diff > bias && diff < thickness) {
            vec2 edge = min(uv, 1.0 - uv);                  // fondu près des bords de l'écran
            return 1.0 - smoothstep(0.0, 0.05, min(edge.x, edge.y));
        }
    }
    return 1.0;
}

// worldPos : position reconstruite du G-Buffer ; normal : normale monde (décalages anti-acné).
float ComputeTerrainAndContactShadow(vec3 worldPos, vec3 normal, vec3 lightDir, vec2 screenUv) {
    float noise = InterleavedGradientNoise(screenUv * uEnv.viewportSize.xy);
    float dist  = length(envCameraPosition() - worldPos);
    return min(HeightfieldShadow(worldPos + normal * 0.15, lightDir, noise),
               ContactShadow(worldPos, normal, lightDir, dist, noise));
}

// =============================================================================
// VXGI (ADR-008) : cone tracing dans voxel.grid
// =============================================================================
bool VoxelGridUsable(vec3 P) {
    return uVoxelAvailable && uVoxel.uGridExtent.w > 0.5 && VoxelUvwInside(VoxelUvw(P, uVoxel.uGridMin.xyz));
}

// Ombre LOINTAINE : cône fin vers l'astre, à partir d'un voxel et demi (en deçà, la grille de 8 m
// s'auto-occulterait ; l'ombre proche est assurée par le heightfield + le contact écran).
float VoxelShadow(vec3 P, vec3 N, vec3 L) {
    if (!uVoxelShadows || !VoxelGridUsable(P)) return 1.0;
    float voxel = VoxelSize();
    return VoxelConeVisibility(uVoxel.uGridMin.xyz, P + N * (0.5 * voxel), L, uVoxelShadowSoftness,
                               1.5 * voxel, 2000.0, 40);
}

// VXAO + rebond diffus : 5 cônes (normale + 4 inclinés de 60°, ouverture 60° -> tan 30° = 0.577),
// pondérés par le cosinus. Chaque cône ramène la lumière des voxels rencontrés ; la part non occultée
// voit le ciel. Résultat : radiance diffuse entrante moyenne (E/π) et visibilité du ciel (VXAO).
vec3  gVxgiRadiance = vec3(0.0);
float gVxao         = 1.0;

bool VoxelDiffuse(vec3 P, vec3 N, out vec3 radiance, out float visibility) {
    radiance   = vec3(0.0);
    visibility = 1.0;
    if (!uVoxelGI || !VoxelGridUsable(P)) return false;

    vec3 T = normalize(cross(abs(N.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0), N));
    vec3 B = cross(N, T);
    const float kTanHalf = 0.577;
    const float kSinTilt = 0.866, kCosTilt = 0.5;  // 60°
    vec3  dirs[5]    = vec3[5](N, N * kCosTilt + T * kSinTilt, N * kCosTilt - T * kSinTilt,
                               N * kCosTilt + B * kSinTilt, N * kCosTilt - B * kSinTilt);
    const float w[5] = float[5](1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0);  // ∝ cos, somme = 1

    float voxel  = VoxelSize();
    vec3  origin = P + N * voxel;  // hors du voxel de surface (sinon auto-occlusion)
    visibility = 0.0;
    for (int i = 0; i < 5; ++i) {
        vec4 c = VoxelConeRadiance(uVoxel.uGridMin.xyz, origin, dirs[i], kTanHalf, 0.5 * voxel, 256.0, 8);
        radiance   += w[i] * (c.rgb * uGiIntensity + (1.0 - c.a) * SkyLuminanceAbove(dirs[i]));
        visibility += w[i] * (1.0 - c.a);
    }
    return true;
}

// Éclairage ambiant : radiance diffuse (E/π) et radiance spéculaire déjà pondérée par la BRDF.
void Ambient(Surface s, vec3 P, vec3 V, out vec3 diffuse, out vec3 specular) {
    float NoV = max(dot(s.N, V), 1e-4);
    vec3  R   = reflect(-V, s.N);

    // VXGI disponible : il remplace le diffus (IBL ou repli sky-view) et occulte le spéculaire.
    vec3  vxRadiance;
    float vxVisibility;
    bool  hasVoxelGI = VoxelDiffuse(P, s.N, vxRadiance, vxVisibility);
    gVxgiRadiance = vxRadiance;
    gVxao         = vxVisibility;

    if (uIblAvailable) {
        // Split-sum (Karis 2013)
        vec3 irradiance = texture(uIblIrradiance, s.N).rgb;
        float maxLod = float(textureQueryLevels(uIblPrefiltered) - 1);
        vec3 prefiltered = textureLod(uIblPrefiltered, R, s.roughness * maxLod).rgb;
        vec2 brdf = texture(uIblBrdfLut, vec2(NoV, s.roughness)).rg;
        diffuse  = (hasVoxelGI ? vxRadiance : irradiance) * s.diffuseColor;
        specular = prefiltered * (s.f0 * brdf.x + brdf.y) * (hasVoxelGI ? vxVisibility : 1.0);
    } else if (hasVoxelGI) {
        // VXGI : rebond coloré des voxels + ciel non occulté ; le spéculaire (ciel réfléchi) est occulté
        // par la visibilité du ciel (VXAO), faute de cône de réflexion dédié.
        vec3 specL = mix(SkyLuminanceAbove(R) * vxVisibility, vxRadiance, s.alpha);
        vec2 ab = EnvBrdfApprox(NoV, s.roughness);
        diffuse  = vxRadiance * s.diffuseColor;
        specular = specL * (s.f0 * ab.x + ab.y);
    } else {
        // Repli physique sur la sky-view (mêmes unités cd/m² que le ciel affiché).
        // Ambiant hémisphérique doux : ciel (moyenne + direction de la normale) au-dessus, rebond du sol
        // (albédo de l'atmosphère éclairé par le soleil, la lune et le ciel) en dessous, mélangés par N.y.
        // C'est lui qui détaille les zones à l'ombre, de jour comme sous la lune.
        vec3  up      = vec3(0.0, 1.0, 0.0);
        float skyVis  = 0.5 + 0.5 * s.N.y;
        vec3  skyAvg  = SkyAverage();
        vec3  skyL    = 0.5 * (SkyLuminanceAbove(normalize(s.N + up)) + skyAvg);
        vec3  sunDir  = normalize(envSunDirection());
        vec3  moonDir = normalize(envMoonDirection());
        vec3  groundE = SunIlluminanceTOA()  * LightTransmittance(0.0, sunDir,  SunAngularRadius())  * max(sunDir.y, 0.0)  * uSunScale
                      + MoonIlluminanceTOA() * LightTransmittance(0.0, moonDir, MoonAngularRadius()) * max(moonDir.y, 0.0) * uMoonScale
                      + PI * skyAvg;
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

float PhaseHenyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1e-4, 1.0 + g2 - 2.0 * g * cosTheta), 1.5));
}

// Luminance diffusée par le brouillard lui-même (milieu optiquement épais, albédo de diffusion ≈ 1) :
// soleil et lune atténués (phase de Henyey-Greenstein, gouttelettes -> diffusion vers l'avant), rendus
// séparément (sunTerm, moonTerm) pour qu'ApplyHeightFog les multiplie par leur visibilité moyenne le long
// du rayon (ombre volumétrique) ; retourne le ciel moyen, ambiant isotrope. Surtout PAS la luminance du ciel à l'horizon (sky-view) : elle intègre des centaines de km
// d'atmosphère et la lueur crépusculaire, ce qui délavait tout le terrain (flash du crépuscule).
// Jamais les disques du soleil ni de la lune non plus.
vec3 FogInscatter(vec3 P, vec3 rayDir, out vec3 sunTerm, out vec3 moonTerm) {
    const float g = 0.6;
    vec3 sunDir  = normalize(envSunDirection());
    vec3 moonDir = normalize(envMoonDirection());
    sunTerm  = SunIlluminanceTOA() * LightTransmittance(P.y, sunDir, SunAngularRadius())
             * PhaseHenyeyGreenstein(dot(rayDir, sunDir), g) * uSunScale;
    moonTerm = MoonIlluminanceTOA() * LightTransmittance(P.y, moonDir, MoonAngularRadius())
             * PhaseHenyeyGreenstein(dot(rayDir, moonDir), g) * uMoonScale;
    return SkyAverage();
}

// Épaisseur optique du brouillard σ(y) = σ0 exp(-k y), de la caméra jusqu'à la distance t le long du
// rayon de direction verticale dirY (intégrale analytique).
float FogOpticalDepth(float t, float dirY) {
    float sigma0 = uEnv.weatherExtra.z;  // 1/m au niveau y = 0
    float k      = max(uFogHeightFalloff, 1e-5);
    float camY   = envCameraPosition().y;
    float kd     = k * dirY * t;
    float shape  = abs(kd) > 1e-4 ? (1.0 - exp(-kd)) / kd : 1.0;
    return sigma0 * exp(clamp(-k * camY, -80.0, 80.0)) * t * shape;
}

// Ombre VOLUMÉTRIQUE du brouillard : visibilité moyenne de l'astre L le long du rayon caméra -> point
// visé, chaque tronçon pondéré par ce qu'il apporte à la diffusion (T(début) - T(fin)). Sans elle, l'air
// situé dans l'ombre d'un relief diffusait le soleil en plein : voile aveuglant devant les montagnes à
// contre-jour (la phase g = 0,6 concentre la lumière vers l'avant). Cône de demi-ouverture 0,12 rad
// dans la grille VXGI ; hors grille, l'air est considéré éclairé. 4 points × 20 pas : écart moyen
// < 0,5/255 contre une référence à 12 points × 48 pas (cône 0,08), pour ~1/10 du coût.
const int kFogShadowSamples = 4;
float FogLightVisibility(vec3 rayDir, float dist, vec3 L) {
    if (!uFogShadows || !uVoxelAvailable || uVoxel.uGridExtent.w < 0.5) return 1.0;
    vec3  cam   = envCameraPosition();
    float voxel = VoxelSize();
    float Tprev = 1.0, acc = 0.0, wsum = 0.0;
    for (int i = 0; i < kFogShadowSamples; ++i) {
        float T1 = exp(-FogOpticalDepth(dist * float(i + 1) / float(kFogShadowSamples), rayDir.y));
        float w  = Tprev - T1;
        Tprev    = T1;
        if (w <= 0.0) continue;
        vec3 p = cam + rayDir * (dist * (float(i) + 0.5) / float(kFogShadowSamples));
        acc  += w * VoxelConeVisibility(uVoxel.uGridMin.xyz, p, L, 0.12, 0.5 * voxel, 1500.0, 20);
        wsum += w;
    }
    return wsum > 0.0 ? acc / wsum : 1.0;
}

// Brouillard de hauteur : densité σ(y) = σ0 exp(-k y), intégrée analytiquement le long du rayon.
// sunShadow / moonShadow : visibilité de l'astre AU POINT VISÉ (0 si la face lui tourne le dos).
vec3 ApplyHeightFog(vec3 color, vec3 P, vec3 rayDir, float dist, float sunShadow, float moonShadow) {
    if (uEnv.weatherExtra.z <= 0.0) return color;
    float fog = 1.0 - exp(-max(FogOpticalDepth(dist, rayDir.y), 0.0));
    if (fog < 1e-4) return color;

    vec3 sunTerm, moonTerm;
    vec3 ambient = FogInscatter(P, rayDir, sunTerm, moonTerm);
    // Ombre volumétrique pour l'astre dominant (le soleil tant qu'il est levé, sinon la lune). Calculée
    // seulement si le point visé n'est pas en plein soleil (ombré ou face dos à l'astre) : devant une
    // surface éclairée, l'air l'est aussi en pratique, et c'est devant les reliefs à l'ombre que le voile
    // fuyait. Ignorée aussi quand elle ne changerait pas le pixel de plus de 3 %.
    bool  sunUp      = envSunDirection().y > -0.02;
    vec3  mainTerm   = sunUp ? sunTerm : moonTerm;
    float surfShadow = sunUp ? sunShadow : moonShadow;
    float vis        = 1.0;
    if (surfShadow < 0.99 && fog > 0.002 &&
        fog * Luminance(mainTerm) > 0.03 * Luminance(mix(color, ambient, fog)))
        vis = FogLightVisibility(rayDir, dist, normalize(sunUp ? envSunDirection() : envMoonDirection()));
    vec3 inscatter = ambient + (sunUp ? sunTerm * vis + moonTerm : sunTerm + moonTerm * vis);
    return mix(color, inscatter, fog);
}


// =============================================================================
// Anti-aliasing spéculaire géométrique (Tokuyoshi & Kaplanyan 2019) : élargit α² selon la variation
// de la normale entre pixels voisins. Sans lui, une surface quasi miroir (eau, sol mouillé, α ≈ 0.0025)
// fait scintiller le reflet du soleil dès que la normale ou l'astre bouge d'un rien.
const float kSpecAaSigma2 = 0.15915494;  // 1 / (2π)
const float kSpecAaKappa  = 0.18;

#endif  // ENV_SURFACE_LIGHTING_GLSL
