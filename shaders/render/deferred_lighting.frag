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
#include "render/sky_lighting.glsl"   // ciel, transmittance, éclairements (partagés avec la voxélisation)
#include "render/voxel_common.glsl"   // VXGI : placement de voxel.grid + cone tracing (ADR-008)

in vec2 vUv;
layout(location = 0) out vec4 oColor;

layout(binding = TEX_GBufferAlbedo)     uniform sampler2D   uGAlbedo;    // SRGB8_ALPHA8 : rgb albédo, a AO
layout(binding = TEX_GBufferNormal)     uniform sampler2D   uGNormal;    // RGBA16F : xyz normale monde
layout(binding = TEX_GBufferMaterial)   uniform sampler2D   uGMaterial;  // RGBA8 : r rugosité, g métal, b humidité, a id
layout(binding = TEX_GBufferDepth)      uniform sampler2D   uGDepth;     // DEPTH32F reverse-Z
layout(binding = TEX_IblIrradiance)     uniform samplerCube uIblIrradiance;   // convention : radiance diffuse (E / π)
layout(binding = TEX_IblPrefiltered)    uniform samplerCube uIblPrefiltered;  // mip = rugosité perceptuelle
layout(binding = TEX_IblBrdfLut)        uniform sampler2D   uIblBrdfLut;      // (NoV, rugosité) -> (scale, bias)
layout(binding = TEX_CloudShadow)       uniform sampler2D   uCloudShadow;     // transmittance des nuages
layout(binding = TEX_HeightNormal)      uniform sampler2D   uFocusHeight;     // world.focus.heightNormal (R = hauteur)

uniform bool  uGBufferAvailable;
uniform bool  uIblAvailable;
uniform bool  uCloudShadowAvailable;
uniform bool  uApplyFallbackExposure;
uniform bool  uSpecularOcclusion;
uniform float uSunScale;
uniform float uMoonScale;
uniform float uIblScale;
uniform int   uDebugView;              // 0 Lit, 1 Albedo, 2 Normal, 3 Roughness, 4 Metal, 5 AO, 6 Depth, 7 IBL only,
                                       // 8 Sun only, 9 Shadows, 10 VXAO, 11 VXGI (indirect)
// VXGI (ADR-008)
uniform bool  uVoxelAvailable;         // voxel.grid publiée (VoxelIlluminationSystem)
uniform bool  uVoxelShadows;           // ombre lointaine par cône vers le soleil / la lune
uniform bool  uVoxelGI;                // VXAO + rebond diffus (5 cônes)
uniform float uGiIntensity;            // multiplicateur de la lumière indirecte des voxels
uniform float uVoxelShadowSoftness;    // tangente du demi-angle du cône d'ombre (pénombre)
uniform float uExposureCompensation;   // EV ajoutés à l'exposition automatique provisoire
uniform float uNightAdaptation;        // [0, 1] part de l'obscurité compensée par l'exposition (EV < 0)
uniform bool  uMoonlightShift;         // léger virage bleu-argenté en basse lumière (garde >= 75 % de saturation)
uniform float uStarIntensity;          // multiplicateur des étoiles procédurales
// Ombres portées
uniform bool  uTerrainShadows;         // heightfield du chunk focus
uniform bool  uContactShadows;         // espace écran (gbuffer.depth)
uniform bool  uFocusAvailable;         // world.focus.heightNormal publié ET chunk retrouvé dans l'ECS
uniform vec3  uFocusChunkOrigin;       // coin du chunk focus relatif à l'origine de rendu (m)
uniform float uShadowPenumbra;         // demi-angle (rad) de la pénombre des ombres du relief
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

    // Transmittance évaluée dans la direction du pixel : l'horizon découpe le disque au pixel près.
    return SunIlluminanceTOA() / omega * limb * coverage
         * LightTransmittance(envCameraPosition().y, rayDir, 0.0);
}

// Lune : sphère lambertienne. La direction du soleil vue depuis la lune est reconstruite à partir
// de la phase (élongation ψ = 2π·phase : 0 = nouvelle lune, π = pleine lune), dans le plan
// lune-soleil réel : le terminateur est donc orienté vers le vrai soleil ET respecte la phase,
// même quand les sliders azimut / phase sont réglés indépendamment.
vec3 MoonDisk(vec3 rayDir, out float outPeakLuminance) {
    outPeakLuminance = 0.0;
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

    // Disque argenté : moitié moins teinté que la lumière qu'il diffuse (reste lisible comme "blanc lunaire").
    vec3 diskTint = mix(vec3(1.0), MOONLIGHT_TINT, 0.5);
    vec3 radianceScale = diskTint * (eFull / omega)
                       * LightTransmittance(envCameraPosition().y, rayDir, 0.0);
    outPeakLuminance = Luminance(radianceScale) * 1.5;  // partie éclairée la plus brillante
    return radianceScale * shade * coverage;
}

// Contribution du disque lunaire du pixel courant, gardée à part pour l'exposition provisoire.
vec3  gMoonDisk = vec3(0.0);
float gMoonPeak = 0.0;

// --- Étoiles procédurales -------------------------------------------------------------------------
// Grille 3D de cellules sur la direction monde (dir × kCells) : la sphère unité traverse les cellules,
// chaque cellule tirée au sort porte une étoile à une position fixe. Uniquement fonction de rayDir :
// le champ d'étoiles est parfaitement stable (seul un scintillement de ±10 % est animé).
float Hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

vec3 Hash33(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}

vec3 StarField(vec3 dir) {
    const float kCells   = 220.0;  // ~0,26° par cellule
    const float kDensity = 0.012;  // fraction des cellules qui portent une étoile
    vec3  sp   = dir * kCells;
    vec3  base = floor(sp - 0.5);
    float size = max(PixelAngle() * 0.75, 2e-4);  // étoile ≈ 1 pixel, sans crénelage
    vec3  acc  = vec3(0.0);
    for (int i = 0; i < 8; ++i) {  // 2 × 2 × 2 cellules voisines : pas d'étoile coupée en bord de cellule
        vec3 c = base + vec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        if (Hash13(c) > kDensity) continue;
        vec3  h       = Hash33(c + 17.0);
        vec3  starDir = normalize(c + 0.15 + 0.7 * h);
        float ang     = AngleBetween(dir, starDir);
        // Distribution en loi de puissance : beaucoup d'étoiles faibles, quelques brillantes (cd/m² au pic).
        float peak    = 0.004 + 0.12 * pow(h.x, 10.0);
        float twinkle = 0.9 + 0.1 * sin(envTime() * (2.0 + 4.0 * h.y) + h.z * 60.0);
        vec3  tint    = mix(vec3(0.75, 0.85, 1.15), vec3(1.15, 0.95, 0.75), h.z);  // bleutées -> orangées
        acc += tint * peak * twinkle * exp(-(ang * ang) / (size * size));
    }
    return acc;
}

vec3 ShadeSky(vec3 rayDir) {
    gMoonDisk = MoonDisk(rayDir, gMoonPeak) * uMoonScale;
    gMoonPeak *= uMoonScale;
    vec3 sun  = SunDisk(rayDir) * uSunScale;
    vec3 sky  = SkyLuminance(rayDir) + sun + gMoonDisk;

    // Étoiles : apparaissent quand le soleil passe sous -4° (pleinement sous -10°), hors disques,
    // atténuées par l'atmosphère (plus ternes vers l'horizon, absentes sous l'horizon).
    float sunY     = normalize(envSunDirection()).y;
    float starFade = 1.0 - smoothstep(-0.1736, -0.0698, sunY);  // sin(-10°), sin(-4°)
    if (starFade > 0.0 && uStarIntensity > 0.0 && Luminance(gMoonDisk + sun) <= 0.0) {
        sky += StarField(rayDir) * uStarIntensity * starFade
             * LightTransmittance(envCameraPosition().y, rayDir, 0.0);
    }
    return sky;
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
// soleil et lune atténués (phase de Henyey-Greenstein, gouttelettes -> diffusion vers l'avant)
// + ciel moyen en ambiant isotrope. Surtout PAS la luminance du ciel à l'horizon : elle intègre des
// centaines de km d'atmosphère et la lueur crépusculaire, ce qui délavait tout le terrain.
vec3 FogInscatter(vec3 P, vec3 rayDir) {
    const float g = 0.6;
    vec3 sunDir  = normalize(envSunDirection());
    vec3 moonDir = normalize(envMoonDirection());
    vec3 direct  = SunIlluminanceTOA() * LightTransmittance(P.y, sunDir, SunAngularRadius())
                 * PhaseHenyeyGreenstein(dot(rayDir, sunDir), g) * uSunScale
                 + MoonIlluminanceTOA() * LightTransmittance(P.y, moonDir, MoonAngularRadius())
                 * PhaseHenyeyGreenstein(dot(rayDir, moonDir), g) * uMoonScale;

    return direct + SkyAverage();
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
    return mix(color, FogInscatter(P, rayDir), fog);
}

// =============================================================================
// Exposition provisoire (tant que scene.display n'existe pas)
// =============================================================================
// Éclairement (lux) d'une surface horizontale à l'altitude de la caméra : soleil + lune directs
// + ciel intégré sur l'hémisphère (quadrature : zénith + 2 couronnes de 6 directions, pondérées
// par angle solide × cosinus). C'est ce qui éclaire réellement le terrain : l'exposition suit donc
// le crépuscule de façon continue, y compris quand seule la lueur de l'horizon reste.
void HorizontalIlluminance(out vec3 eDirect, out vec3 eSky) {
    float camAlt  = envCameraPosition().y;
    vec3  sunDir  = normalize(envSunDirection());
    vec3  moonDir = normalize(envMoonDirection());
    eDirect = SunIlluminanceTOA()  * LightTransmittance(camAlt, sunDir,  SunAngularRadius())  * max(sunDir.y, 0.0)  * uSunScale
            + MoonIlluminanceTOA() * LightTransmittance(camAlt, moonDir, MoonAngularRadius()) * max(moonDir.y, 0.0) * uMoonScale;

    // Bandes : [0°, 30°] de zénith (Ω = 0.84 sr), [30°, 65°] (2.79 sr), [65°, 90°] (2.66 sr)
    const float wCap = 0.84 * 1.0;
    const float w45  = 2.79 / 6.0 * 0.707;
    const float w12  = 2.66 / 6.0 * 0.208;
    vec3  sky  = SkyLuminance(vec3(0.0, 1.0, 0.0)) * wCap;
    for (int i = 0; i < 6; ++i) {
        float az = (float(i) + 0.5) * (2.0 * PI / 6.0);
        vec2  h  = vec2(sin(az), -cos(az));
        sky += SkyLuminance(vec3(0.707 * h.x, 0.707, 0.707 * h.y)) * w45;
        sky += SkyLuminance(vec3(0.978 * h.x, 0.208, 0.978 * h.y)) * w12;
    }
    eSky = PI * sky / (wCap + 6.0 * w45 + 6.0 * w12);
}

const float kSkyKeyWeight = 0.5;

float FallbackExposure() {
    // Luminance de référence = la plus forte entre :
    //  * une surface grise à 18 % sous l'éclairement total (de jour, le soleil domine) ;
    //  * la luminance moyenne du ciel (au crépuscule et la nuit, c'est le ciel qui remplit l'image).
    // Les deux termes sont continus : pas de saut d'exposition au coucher ni au lever.
    vec3 eDirect, eSky;
    HorizontalIlluminance(eDirect, eSky);
    // Le ciel ne compte qu'à moitié : la nuit, on expose pour le paysage (et non pour le ciel), ce qui
    // laisse le ciel bleu nuit plus lumineux que le sol au lieu de tout ramener au même gris.
    float lAvg = max(0.18 * Luminance(eDirect + eSky) / PI, kSkyKeyWeight * Luminance(eSky) / PI);
    // EV100 (S = 100, K = 12.5). Sous EV 0 (crépuscule avancé, nuit), l'œil ne s'adapte que
    // partiellement : on ne compense qu'une fraction uNightAdaptation de l'obscurité. Le reste
    // de l'obscurité est conservé -> contraste et noirs réels, sans voile gris.
    float ev100 = log2(max(lAvg, 1e-8) * 100.0 / 12.5);
    if (ev100 < 0.0) ev100 *= uNightAdaptation;
    ev100 = clamp(ev100, -10.0, 16.0) - uExposureCompensation;
    return 1.0 / (1.2 * exp2(ev100));
}

// Virage "Moonlight" (remplace l'effet Purkinje complet, trop gris) : en basse lumière, on tire au plus
// 25 % de la couleur vers sa propre luminance teintée bleu-argenté. Herbe, roche et arbres gardent au
// moins 75 % de leur saturation et restent distinguables la nuit.
vec3 MoonlightShift(vec3 c) {
    float Y     = Luminance(c);
    float night = 0.04 / (0.04 + Y);  // 0 de jour, -> 1 sous ~0,04 cd/m²
    return mix(c, Y * MOONLIGHT_TINT, 0.25 * night);
}

vec3 AcesFilmic(vec3 x) {  // Narkowicz 2015
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

// =============================================================================
// Anti-aliasing spéculaire géométrique (Tokuyoshi & Kaplanyan 2019) : élargit α² selon la variation
// de la normale entre pixels voisins. Sans lui, une surface quasi miroir (eau, sol mouillé, α ≈ 0.0025)
// fait scintiller le reflet du soleil dès que la normale ou l'astre bouge d'un rien.
const float kSpecAaSigma2 = 0.15915494;  // 1 / (2π)
const float kSpecAaKappa  = 0.18;

void main() {
    float depth = uGBufferAvailable ? texture(uGDepth, vUv).r : 0.0;

    // Dérivées en flux de contrôle uniforme (avant toute branche) : triangle plein écran.
    vec3  gNormalRaw   = texture(uGNormal, vUv).xyz;
    vec3  dndx         = dFdx(gNormalRaw);
    vec3  dndy         = dFdy(gNormalRaw);
    float kernelAlpha2 = min(2.0 * kSpecAaSigma2 * (dot(dndx, dndx) + dot(dndy, dndy)), kSpecAaKappa);
    bool  isDataView = (uDebugView >= 1 && uDebugView <= 6) || uDebugView == 9 || uDebugView == 10;
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
        s.N         = normalize(gNormalRaw);
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
        s.alpha        = sqrt(clamp(s.roughness * s.roughness * s.roughness * s.roughness + kernelAlpha2, 0.0, 1.0));
        s.roughness    = sqrt(s.alpha);  // rugosité perceptuelle filtrée (mip IBL, LUT BRDF)
        s.diffuseColor = s.albedo * (1.0 - s.metal);
        s.f0           = mix(vec3(0.04), s.albedo, s.metal);
        s.f0           = mix(s.f0, vec3(0.02), wet * porous);  // F0 de l'eau

        // --- Lumières directes ---
        vec3 sunDir  = normalize(envSunDirection());
        vec3 moonDir = normalize(envMoonDirection());
        // Ombres portées = min(proche, lointaine) :
        //  * proche : heightfield 1 m du chunk focus + contact écran (arbres, petits reliefs) ;
        //  * lointaine : cône dans voxel.grid (ADR-008), tous chunks confondus, jusqu'à ~1 km.
        // Évaluées seulement si l'astre est levé ET éclaire la face (N·L > 0) : sinon la lumière directe
        // est nulle de toute façon. La lune n'a son cône voxel que la nuit (de jour, elle est négligeable).
        bool  sunLit     = sunDir.y  > -0.01 && dot(s.N, sunDir)  > 0.0;
        bool  moonLit    = moonDir.y > -0.01 && dot(s.N, moonDir) > 0.0;
        float sunShadow  = sunLit  ? min(ComputeTerrainAndContactShadow(P, s.N, sunDir, vUv),
                                         VoxelShadow(P, s.N, sunDir)) : 0.0;
        float moonShadow = moonLit ? ComputeTerrainAndContactShadow(P, s.N, moonDir, vUv) : 0.0;
        if (moonLit && sunDir.y < 0.05) moonShadow = min(moonShadow, VoxelShadow(P, s.N, moonDir));
        vec3 sunE  = SunIlluminanceTOA()  * LightTransmittance(P.y, sunDir, SunAngularRadius())
                   * CloudShadow(P, sunDir) * sunShadow * uSunScale;
        vec3 moonE = MoonIlluminanceTOA() * LightTransmittance(P.y, moonDir, MoonAngularRadius())
                   * CloudShadow(P, moonDir) * moonShadow * uMoonScale;
        vec3 sunL  = DirectLight(s, V, sunDir, sunE);
        vec3 moonL = DirectLight(s, V, moonDir, moonE);

        // --- Ambiant ---
        vec3 ambD, ambS;
        Ambient(s, P, V, ambD, ambS);
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
            case 9:  color = vec3(sunShadow, moonShadow, sunShadow); break;  // blanc = éclairé, vert = lune seule
            case 10: color = vec3(gVxao); break;                             // VXAO : 1 = ciel entièrement visible
            case 11: color = gVxgiRadiance; break;                           // lumière indirecte reçue (E/π)
            default: color = ApplyHeightFog(sunL + moonL + ambient, P, rayDir, dist); break;
        }
    }

    if (isDataView) {
        // Données brutes : pas d'exposition. Encodées pour le Viewport s'il affiche scene.color tel quel.
        oColor = vec4(uApplyFallbackExposure ? LinearToSrgb(color) : color, 1.0);
    } else if (uApplyFallbackExposure) {
        float exposure = FallbackExposure();
        vec3  moon = gMoonDisk;
        vec3  rest = color - moon;
        if (uMoonlightShift) rest = MoonlightShift(rest);
        // Adaptation locale quand l'œil regarde la lune : le disque (~4000 cd/m²) et la lumière cendrée
        // sont des milliers de fois plus lumineux que le paysage nocturne ; avec l'exposition globale ils
        // saturent tous deux en blanc et la phase disparaît. On plafonne donc l'exposition du disque pour
        // que sa partie éclairée reste juste sous le blanc. De jour (exposition faible) rien ne change.
        float moonExposure = min(exposure, 0.8 / max(gMoonPeak, 1e-6));
        oColor = vec4(LinearToSrgb(AcesFilmic(rest * exposure + moon * moonExposure)), 1.0);
    } else {
        oColor = vec4(min(color, vec3(kHalfMax)), 1.0);  // le disque solaire dépasse le RGBA16F
    }
}
