//  shaders/render/sky_lighting.glsl                                    [DEV 2]
//  Éclairage par le ciel, le soleil et la lune, partagé par la résolution différée
//  (deferred_lighting.frag) et la voxélisation VXGI (voxel_inject.comp) : les deux passes
//  évaluent la même physique (Hillaire 2020) et produisent des luminances dans les mêmes unités.
//
//  Le shader qui inclut ce fichier doit avoir inclus "common/global_ubo.glsl" et
//  "render/atmo_common.glsl", et le C++ doit renseigner l'uniform uAtmoAvailable.
#ifndef ENV_SKY_LIGHTING_GLSL
#define ENV_SKY_LIGHTING_GLSL

layout(binding = TEX_AtmoTransmittance) uniform sampler2D uTransmittanceLUT;
layout(binding = TEX_AtmoSkyView)       uniform sampler2D uSkyViewLUT;

uniform bool uAtmoAvailable;  // atmo.* publiées par CelestialAtmosphereSystem

bool AtmoValid() {
    // L'UBO AtmosphereParams est lié par CelestialAtmosphereSystem : on vérifie qu'il est cohérent.
    return uAtmoAvailable && uAtmo.uRadiiAndScaleHeights.x > 0.0
        && uAtmo.uRadiiAndScaleHeights.y > uAtmo.uRadiiAndScaleHeights.x;
}

// Luminance du ciel (cd/m², sans les disques) dans la direction dir.
vec3 SkyLuminance(vec3 dir) {
    if (!AtmoValid()) return vec3(0.0);
    return textureLod(uSkyViewLUT, SkyViewDirToUV(dir, AtmoViewRadiusKm(), uAtmo.uRadiiAndScaleHeights.x), 0.0).rgb;
}

// Idem, direction remontée au-dessus de l'horizon : sous l'horizon la LUT ne contient que l'air
// entre la caméra et le sol, pas le sol lui-même. Utilisé pour l'ambiant et le brouillard.
vec3 SkyLuminanceAbove(vec3 dir) {
    dir.y = max(dir.y, 0.03);
    return SkyLuminance(normalize(dir));
}

// Ciel moyen (cd/m²) : zénith + 4 directions à 15° d'élévation (cos 0.966, sin 0.259).
vec3 SkyAverage() {
    vec3 ring = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        float az = (float(i) + 0.5) * 0.5 * PI;
        ring += SkyLuminance(vec3(0.966 * sin(az), 0.259, -0.966 * cos(az)));
    }
    return 0.5 * SkyLuminance(vec3(0.0, 1.0, 0.0)) + 0.125 * ring;
}

// Transmittance atmosphère × ombre de la planète, d'un point d'altitude altM (m) vers un astre de
// rayon angulaire angR (rad). Repère local planète : +Y = zénith, donc mu = lightDir.y.
// Calculs par altitude (et non par rayon r ≈ 6360 km) : stables au mètre près.
vec3 LightTransmittance(float altM, vec3 lightDir, float angR) {
    if (!AtmoValid()) return vec3(smoothstep(-0.02, 0.02, lightDir.y));
    float Rb  = uAtmo.uRadiiAndScaleHeights.x;
    float Rt  = uAtmo.uRadiiAndScaleHeights.y;
    float hKm = clamp(altM * 0.001, 0.0, Rt - Rb);
    return textureLod(uTransmittanceLUT, TransmittanceParamsToUVAlt(hKm, lightDir.y, Rb, Rt), 0.0).rgb
         * PlanetShadowAlt(hKm, lightDir.y, Rb, angR);
}

float SunAngularRadius()  { return uEnv.sunColorAngular.a; }
float MoonAngularRadius() { return uEnv.moonParams.y; }

// Éclairements au sommet de l'atmosphère (lux). Clair de lune : teinte bleu-argenté (luminance conservée).
vec3 SunIlluminanceTOA()  { return uEnv.sunColorAngular.rgb * uEnv.sunDirIlluminance.w; }
vec3 MoonIlluminanceTOA() { return MOONLIGHT_TINT * uEnv.moonParams.x; }

#endif  // ENV_SKY_LIGHTING_GLSL
