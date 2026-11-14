#version 460 core
//  shaders/render/deferred_lighting.frag                               [DEV 2]
//  Résolution différée PBR -> scene.color (HDR linéaire, cd/m²).
//
//  * Ciel (depth == 0) : sky-view LUT (Hillaire 2020) + disque solaire (assombrissement centre-bord)
//    + disque lunaire éclairé comme une sphère (phase) + lumière cendrée.
//  * Géométrie : Cook-Torrance GGX (D_GGX, V_SmithGGXCorrelated, F_Schlick) + Lambert, soleil et lune
//    atténués par la transmittance atmosphérique, l'ombre de la planète et l'ombre des nuages ;
//    ambiant IBL split-sum (ou repli sur la sky-view) ; météo de surface ; brouillard de hauteur.
//  * Unités physiques : éclairements en lux, luminances en cd/m². L'exposition et le tonemap sont faits
//    plus loin (tonemap.frag, VolumetricGodRaysSystem -> scene.display), après l'eau. Repli si cette passe
//    ne tourne pas : exposition automatique provisoire + ACES directement dans scene.color.
#include "common/global_ubo.glsl"
#include "render/atmo_common.glsl"
#include "render/sky_lighting.glsl"   // ciel, transmittance, éclairements (partagés avec la voxélisation)
#include "render/voxel_common.glsl"   // VXGI : placement de voxel.grid + cone tracing (ADR-008)
#include "render/surface_lighting.glsl"  // BRDF, ombres, ambiant, brouillard (partagés avec l'eau)
#include "render/exposure.glsl"          // exposition provisoire (plafond du disque lunaire, repli LDR)

in vec2 vUv;
layout(location = 0) out vec4 oColor;


layout(binding = TEX_GBufferAlbedo)     uniform sampler2D   uGAlbedo;    // SRGB8_ALPHA8 : rgb albédo, a AO
layout(binding = TEX_GBufferNormal)     uniform sampler2D   uGNormal;    // RGBA16F : xyz normale monde
layout(binding = TEX_GBufferMaterial)   uniform sampler2D   uGMaterial;  // RGBA8 : r rugosité, g métal, b humidité, a id

uniform bool  uApplyFallbackExposure;  // repli : pas de passe de tonemap active -> scene.color encodée ici
uniform int   uDebugView;              // 0 Lit, 1 Albedo, 2 Normal, 3 Roughness, 4 Metal, 5 AO, 6 Depth, 7 IBL only,
                                       // 8 Sun only, 9 Shadows, 10 VXAO, 11 VXGI (indirect)
uniform float uStarIntensity;          // multiplicateur des étoiles procédurales

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
        // L'humidité (albédo assombri, rugosité, flaques, ondes) et la neige sont déjà dans le G-Buffer :
        // la G-pass les calcule par matériau (terrain_material.glsl, terrain_wetness.glsl,
        // gbuffer_foliage.frag). Il ne reste ici que le F0 de l'eau.
        float wet    = clamp(gM.b, 0.0, 1.0);
        float porous = 1.0 - s.metal;

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
            default: color = ApplyHeightFog(sunL + moonL + ambient, P, rayDir, dist, sunShadow, moonShadow); break;
        }
    }

    if (isDataView) {
        // Données brutes : pas d'exposition (la passe de tonemap les laisse aussi telles quelles).
        oColor = vec4(uApplyFallbackExposure ? LinearToSrgb(color) : color, 1.0);
        return;
    }

    // Adaptation locale quand l'œil regarde la lune : le disque (~4000 cd/m²) et la lumière cendrée sont
    // des milliers de fois plus lumineux que le paysage nocturne ; avec l'exposition globale ils
    // saturent tous deux en blanc et la phase disparaît. On plafonne donc l'exposition du disque pour
    // que sa partie éclairée reste juste sous le blanc. De jour (exposition faible) rien ne change.
    // Sortie HDR : le disque est pré-multiplié par le rapport des deux expositions, la passe de tonemap
    // appliquant ensuite l'exposition globale.
    vec3  moon = gMoonDisk;
    vec3  rest = color - moon;
    float exposure = 1.0, moonExposure = 1.0;
    if (uApplyFallbackExposure || gMoonPeak > 0.0) {
        exposure     = FallbackExposure();
        moonExposure = min(exposure, 0.8 / max(gMoonPeak, 1e-6));
    }
    if (uApplyFallbackExposure) {
        if (uMoonlightShift) rest = MoonlightShift(rest);
        oColor = vec4(LinearToSrgb(AcesFilmic(rest * exposure + moon * moonExposure)), 1.0);
    } else {
        // le disque solaire dépasse le RGBA16F
        oColor = vec4(min(rest + moon * (moonExposure / exposure), vec3(kHalfMax)), 1.0);
    }
}
