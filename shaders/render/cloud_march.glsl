//  shaders/render/cloud_march.glsl                                     [DEV 2]
//  Marche des nuages le long d'un rayon (segment dans la coque, éclairage, intégration), partagée par le
//  raymarching affiché (volumetric_clouds.comp) et la capture de l'IBL (ibl_capture.comp) : l'éclairage
//  ambiant et les reflets voient exactement les mêmes nuages.
//    * Éclairage : marche secondaire en cône vers l'astre dominant + échantillon lointain (ombre propre),
//      Henyey-Greenstein à deux lobes, diffusion multiple approchée par octaves (Wrenninge 2013) +
//      transmission diffuse des nuages épais, probabilité d'in-scattering (Nubis 2017), liseré argenté.
//    * Intégration de Beer-Lambert à conservation d'énergie (Hillaire 2015) le long du rayon.
//  Le shader qui inclut ce fichier doit avoir inclus "common/global_ubo.glsl", "render/atmo_common.glsl",
//  "render/sky_lighting.glsl" et "render/cloud_density.glsl".
#ifndef ENV_CLOUD_MARCH_GLSL
#define ENV_CLOUD_MARCH_GLSL

// Cône de 6 directions (fixes) autour du rayon de lumière : comme Nubis, la marche secondaire s'élargit
// en cône, ce qui lisse l'ombre propre obtenue avec peu d'échantillons.
const vec3 kCone[6] = vec3[6](vec3( 0.38,  0.92,  0.12), vec3(-0.65, -0.21,  0.73), vec3( 0.18, -0.83, -0.53),
                              vec3( 0.86, -0.31,  0.40), vec3(-0.34,  0.48, -0.81), vec3(-0.79,  0.16, -0.59));

// Intersections d'un rayon avec une sphère centrée à l'origine : (t0, t1), ou (-1, -1).
vec2 RaySphere(vec3 p, vec3 d, float R) {
    float b = dot(p, d);
    float c = dot(p, p) - R * R;
    float disc = b * b - c;
    if (disc < 0.0) return vec2(-1.0);
    float s = sqrt(disc);
    return vec2(-b - s, -b + s);
}

float HenyeyGreenstein(float cosT, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1e-4, 1.0 + g2 - 2.0 * g * cosT), 1.5));
}

float InterleavedGradientNoise(vec2 pixel) {
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

float CloudPlanetRadius() { return (AtmoValid() ? uAtmo.uRadiiAndScaleHeights.x : 6360.0) * 1000.0; }

// Segment [tA, tB] du rayon dans la coque [bas, haut] ; le sol de la planète l'arrête.
// geomDist : distance de la géométrie opaque (1e30 : ciel). Relief DEVANT la couche (montagne sur le
// ciel) : on marche quand même (behindRelief), la composition cache ces nuages au pixel près ; relief
// DANS la couche : la marche s'arrête dessus. false : rien à marcher.
bool CloudSegment(vec3 cam, vec3 dir, float geomDist, out float tA, out float tB, out bool behindRelief) {
    float Rm  = CloudPlanetRadius();
    vec3  pc  = vec3(0.0, cam.y + Rm, 0.0);  // caméra dans le repère centré planète
    vec2  top = RaySphere(pc, dir, Rm + uCloudTop);
    vec2  bot = RaySphere(pc, dir, Rm + uCloudBottom);
    vec2  gnd = RaySphere(pc, dir, Rm);
    tA = max(top.x, 0.0);
    tB = top.y;
    if (bot.y > 0.0) {
        if (bot.x > tA) tB = min(tB, bot.x);  // la caméra est au-dessus de la base : on s'arrête en y entrant
        else            tA = max(tA, bot.y);  // la caméra est sous la base : on commence en sortant
    }
    if (gnd.x > 0.0) tB = min(tB, gnd.x);
    behindRelief = geomDist < tA;
    if (!behindRelief) tB = min(tB, geomDist);
    tB = min(tB, tA + uMaxDistance);
    return top.y > 0.0 && tB > tA;
}

// Densité 2D des cirrus (Nubis Evolved 2022) : mélange de 3 motifs (streaky, wispy, round)
// synthétisés depuis uShapeNoise et uDetailNoise sans nouveau slot de texture.
float SampleCirrusDensity(vec2 xz) {
    vec2 uv = (xz + uCirrusOffset) / max(uCirrusScale, 1.0);

    // Mélange des 3 motifs selon uCirrusType (0 -> 0.5 -> 1), en n'évaluant que ceux utilisés.
    float pattern;
    if (uCirrusType < 0.5) {
        vec2  windDir   = length(uSkew) > 1e-3 ? normalize(uSkew) : vec2(1.0, 0.0);
        vec2  windPerp  = vec2(-windDir.y, windDir.x);
        vec2  uvStretch = vec2(dot(uv, windDir) * 0.35, dot(uv, windPerp) * 2.2);
        float sShape    = textureLod(uShapeNoise, vec3(uvStretch, 0.25), 0.0).r;
        float sFine     = textureLod(uDetailNoise, vec3(uvStretch * 3.0, 0.5), 0.0).r;
        float streaky   = clamp(sShape * 0.75 + sFine * 0.25, 0.0, 1.0);
        if (uCirrusType <= 0.001) {
            pattern = streaky;
        } else {
            vec2  curl  = textureLod(uDetailNoise, vec3(uv * 2.0, 0.35), 0.0).gb * 2.0 - 1.0;
            float wispy = textureLod(uShapeNoise, vec3(uv * 1.4 + curl * 0.12, 0.65), 0.0).r;
            pattern = mix(streaky, wispy, uCirrusType * 2.0);
        }
    } else {
        vec4  rTex  = textureLod(uShapeNoise, vec3(uv * 2.2, 0.85), 0.0);
        float round = clamp(rTex.g * 0.7 + rTex.b * 0.3, 0.0, 1.0);
        if (uCirrusType >= 0.999) {
            pattern = round;
        } else {
            vec2  curl  = textureLod(uDetailNoise, vec3(uv * 2.0, 0.35), 0.0).gb * 2.0 - 1.0;
            float wispy = textureLod(uShapeNoise, vec3(uv * 1.4 + curl * 0.12, 0.65), 0.0).r;
            pattern = mix(wispy, round, uCirrusType * 2.0 - 1.0);
        }
    }

    // Découpe par la couverture locale (modulation douce par la météo globale).
    float cov = clamp(uCirrusCoverage, 0.0, 1.0);
    return clamp(Remap(pattern, 1.0 - cov * 0.85, 1.0, 0.0, 1.0), 0.0, 1.0) * cov;
}

// Sous-couche Cirrus 2.5D : une seule intersection sphérique à uCirrusAltitude,
// éclairage rasant au coucher de soleil (haute altitude) et 1 échantillon d'auto-ombrage.
vec4 EvaluateCirrus(vec3 cam, vec3 dir, vec3 L, vec3 lightTOA, float angR, vec3 skyAmb) {
    if (uCirrusCoverage <= 0.001 || dir.y <= 0.005) return vec4(0.0, 0.0, 0.0, 1.0);
    float Rm = CloudPlanetRadius();
    vec3  pc = vec3(0.0, cam.y + Rm, 0.0);
    vec2  hit = RaySphere(pc, dir, Rm + uCirrusAltitude);
    float tC  = hit.x > 0.0 ? hit.x : hit.y;
    if (tC <= 0.0) return vec4(0.0, 0.0, 0.0, 1.0);

    vec3  pC  = cam + dir * tC;
    float den = SampleCirrusDensity(pC.xz);
    if (den <= 0.001) return vec4(0.0, 0.0, 0.0, 1.0);

    // Auto-ombrage 2.5D léger : 1 seul échantillon décalé vers l'astre.
    float denL = SampleCirrusDensity(pC.xz + L.xz * 450.0);
    float od   = denL * 1.4;

    // Éclairage à 7500 m : reste illuminé en doré/rose plus longtemps au coucher du soleil.
    float cosT  = dot(dir, L);
    float phase = mix(HenyeyGreenstein(cosT, 0.65), 1.0 / (4.0 * PI), 0.45);
    vec3  E     = lightTOA * LightTransmittance(uCirrusAltitude, L, angR);
    vec3  col   = E * exp(-od) * phase * 1.8 + skyAmb * 0.65;

    // Opacité fine (les cirrus sont semi-transparents) + fondu lointain vers le ciel.
    float alpha = clamp(1.0 - exp(-den * 1.15), 0.0, 0.85);
    float fade  = exp(-tC / max(uFadeDistance * 1.8, 1.0));
    vec3  Lcir  = mix(SkyLuminance(dir) * alpha, col * alpha, fade);
    return vec4(Lcir, 1.0 - alpha);
}

// Marche de [tA, tB] : vec4(luminance diffusée (cd/m²), transmittance). jitter ∈ [0, 1) : décalage du
// premier pas ; detail : bruit de détail + curl (et pour les 2 premiers pas de lumière) ; skyAmb :
// radiance ambiante du ciel (E/π au zénith).
vec4 MarchClouds(vec3 cam, vec3 dir, float tA, float tB, int steps, int lightSteps, bool detail, float jitter,
                 vec3 skyAmb) {
    float Rm = CloudPlanetRadius();
    vec3  pc = vec3(0.0, cam.y + Rm, 0.0);
    // Astre dominant : le soleil tant qu'il éclaire, sinon la lune.
    vec3  sunDir  = normalize(envSunDirection());
    vec3  moonDir = normalize(envMoonDirection());
    bool  sunMain = sunDir.y > -0.08;
    vec3  L       = sunMain ? sunDir : moonDir;
    vec3  lightTOA = sunMain ? SunIlluminanceTOA() : MoonIlluminanceTOA();
    float angR    = sunMain ? SunAngularRadius() : MoonAngularRadius();
    float cosT    = dot(dir, L);
    float phase   = mix(HenyeyGreenstein(cosT, uHgForward), HenyeyGreenstein(cosT, uHgBackward), 0.3);
    float phase1  = mix(HenyeyGreenstein(cosT, uHgForward * 0.5),  HenyeyGreenstein(cosT, uHgBackward * 0.5),  0.3);
    float phase2  = mix(HenyeyGreenstein(cosT, uHgForward * 0.25), HenyeyGreenstein(cosT, uHgBackward * 0.25), 0.3);
    float silver  = uSilver * 0.1 * HenyeyGreenstein(cosT, 0.95);

    steps        = max(steps, 4);
    float thick  = uCloudTop - uCloudBottom;

    // Taille de pas adaptative selon la distance (Nubis Evolved 2022) :
    // pas fin de près (netteté des détails), pas croissant au loin (moins de pas vers l'horizon).
    // stepScale adapte automatiquement les constantes pour behindRelief (32 pas) et l'IBL (20 pas).
    const float kNearStep   = 35.0;     // m : pas fin près de la caméra (~47 m à 1,5 km)
    const float kFarOffset  = 250.0;    // m : augmentation de pas à kAdjustDist
    const float kAdjustDist = 25000.0;  // m : distance de référence
    float stepScale = 64.0 / float(steps);

    float dt0 = (kNearStep + (kFarOffset * tA) / kAdjustDist) * stepScale;
    float t   = tA + dt0 * jitter;
    vec3  Lsum = vec3(0.0);
    float T    = 1.0;
    float tWeighted = 0.0, wSum = 0.0;
    int   zeroCount = 0;

    for (int i = 0; i < steps; ++i) {
        if (t >= tB || T < 0.01) break;  // opaque ou fin de segment : arrêt anticipé
        float dt     = (kNearStep + (kFarOffset * t) / kAdjustDist) * stepScale;
        float stepDt = min(dt, tB - t);  // évite de dépasser tB sur le dernier pas

        vec3  p   = cam + dir * t;
        vec3  pp  = pc + dir * t;
        float alt = length(pp) - Rm;
        float dimProfile = 0.0;
        float den = CloudDensity(p, alt, detail, dimProfile);
        if (den > 0.0) {
            if (zeroCount >= 2) {
                t = max(tA, t - dt);
                zeroCount = 0;
                continue;
            }
            zeroCount = 0;
            float sigma = den * uExtinction;
            vec3  up    = pp / max(length(pp), 1.0);

            // Ombre propre : marche en cône vers l'astre (pas croissants), puis un échantillon lointain pour
            // l'ombre des grands nuages voisins (Nubis 2017).
            float od    = 0.0;
            float stepL = 0.3 * thick / float(max(lightSteps, 1));
            float tl    = 0.0;
            for (int j = 0; j < lightSteps; ++j) {
                float tm   = tl + 0.5 * stepL;
                vec3  off  = L * tm + kCone[j % 6] * (0.25 * tm);
                float qAlt = length(pp + off) - Rm;
                od += CloudDensity(p + off, qAlt, detail && j < 2) * uExtinction * stepL;
                tl += stepL;
                stepL *= 1.6;
            }
            {
                vec3  off  = L * (tl + thick);
                float qAlt = length(pp + off) - Rm;
                od += CloudDensity(p + off, qAlt, false) * uExtinction * thick * 0.5;
            }
            // Éclairement de l'astre à cette altitude (transmittance atmosphérique, ombre de la planète).
            vec3 E = lightTOA * LightTransmittance(alt, L, angR);

            // Probabilité d'in-scattering (Nubis 2017, remplace le « powder » de 2015) : peu de lumière
            // diffusée plusieurs fois arrive sur les bords peu denses et à la base des nuages.
            float hFrac    = clamp((alt - uCloudBottom) / thick, 0.0, 1.0);
            float depthP   = 0.05 + pow(clamp(den, 0.0, 1.0), Remap(hFrac, 0.3, 0.85, 0.5, 2.0));
            float vertP    = pow(clamp(Remap(hFrac, 0.07, 0.14, 0.1, 1.0), 0.1, 1.0), 0.8);
            float inscatter = mix(1.0, depthP * vertP, clamp(uPowder * 0.5, 0.0, 1.0));

            // Diffusion simple + diffusion multiple approchée (octaves de Wrenninge : fonctions de phase
            // précalculées hors boucle) ; les octaves multiples subissent l'in-scattering.
            float ms = phase * exp(-od) + silver * exp(-od) * (1.0 - clamp(den, 0.0, 1.0));
            ms += (0.5 * exp(-od * 0.25) * phase1 + 0.25 * exp(-od * 0.0625) * phase2) * inscatter;
            // Transmission diffuse des nuages épais (approximation de diffusion, 1 / (1 + 0,75 τ (1 - g))).
            ms += (1.0 - exp(-0.5 * od)) / (1.0 + 0.75 * (1.0 - 0.85) * od) * (1.0 / PI) * inscatter;

            // Ambiant du ciel (Nubis Evolved 2022) : occlusion par le profil dimensionnel (cœur dense
            // plus sombre, bords plus lumineux), combinée à la hauteur et à l'atténuation en profondeur.
            // skyAmb reste le ciel clair (jamais l'IBL) pour éviter toute boucle de rétroaction.
            float ambAO    = pow(1.0 - clamp(dimProfile, 0.0, 1.0), 0.5);
            float ambDepth = 0.25 + 0.75 * exp(-0.04 * od);
            vec3 S = sigma * (E * ms + skyAmb * mix(0.35, 1.0, hFrac) * ambDepth * ambAO);

            // Lueur interne d'éclair (Nubis Evolved 2022) : rayon fini (zéro fuite au loin)
            // et intensité calée sur la luminance du ciel (cohérente avec l'auto-exposition jour/nuit).
            if (uLightningIntensity > 0.001) {
                const float kLightningRadius = 4500.0;  // m : portée maximale stricte de l'éclair
                float dL = length(p - uLightningPos);
                if (dL < kLightningRadius) {
                    float falloff    = 1.0 - dL / kLightningRadius;
                    float atten      = falloff * falloff * falloff;  // décroissance douce -> 0 exact au bord
                    float ambLum     = dot(skyAmb, vec3(0.2126, 0.7152, 0.0722));
                    float flashScale = max(ambLum * 8.0, 0.6);       // adapté au jour comme à la nuit
                    float hGrad      = mix(0.35, 1.0, hFrac);
                    float glow       = uLightningIntensity * flashScale * hGrad * atten
                                     * (0.35 + 0.65 * clamp(dimProfile, 0.0, 1.0));
                    S += sigma * vec3(0.72, 0.84, 1.0) * glow;
                }
            }

            // Intégration à conservation d'énergie sur le pas (utilise stepDt).
            float Tstep = exp(-sigma * stepDt);
            Lsum += T * (S - S * Tstep) / max(sigma, 1e-7);
            tWeighted += T * (1.0 - Tstep) * t;
            wSum      += T * (1.0 - Tstep);
            T *= Tstep;
            t += dt;
        } else {
            ++zeroCount;
            t += (zeroCount >= 2) ? (2.0 * dt) : dt;
        }
    }

    // Marche arrêtée sous 1 % de transmittance : le nuage est opaque.
    if (T < 0.01) T = 0.0;

    // Perspective aérienne des nuages volumétriques bas.
    // Sous forte couverture, on rabat le ciel de fondu vers skyAmb (sans halo solaire de Mie)
    // pour éviter que le soleil couchant ne perce artificiellement un plafond opaque à l'horizon.
    float dCloud   = wSum > 0.0 ? tWeighted / wSum : tA;
    float fade     = exp(-dCloud / max(uFadeDistance, 1.0));
    float overcast = smoothstep(0.4, 0.9, uEnv.weatherExtra.x) * (1.0 - T);
    vec3  farSky   = mix(SkyLuminance(dir), skyAmb, overcast);
    Lsum = mix(farSky * (1.0 - T), Lsum, fade);

    // Sous-couche Cirrus 2.5D en haute altitude (derrière les nuages bas, uniquement si T > 0).
    if (T > 0.01 && uCirrusCoverage > 0.001) {
        vec4 cir = EvaluateCirrus(cam, dir, L, lightTOA, angR, skyAmb);
        Lsum += T * cir.rgb;
        T    *= cir.a;
    }
    return vec4(min(Lsum, vec3(65000.0)), T);
}

#endif  // ENV_CLOUD_MARCH_GLSL
