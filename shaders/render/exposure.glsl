//  shaders/render/exposure.glsl                                        [DEV 2]
//  Exposition automatique provisoire (physique : éclairement horizontal + ciel) et tonemap ACES,
//  utilisés par la passe de tonemap (tonemap.frag) et, pour le plafond d'exposition du disque lunaire,
//  par la résolution différée. À remplacer par l'histogramme de luminance (SSBO_ExposureHistogram).
//
//  Le shader qui inclut ce fichier doit avoir inclus "common/global_ubo.glsl",
//  "render/atmo_common.glsl" et "render/sky_lighting.glsl", et déclarer uSunScale / uMoonScale
//  (surface_lighting.glsl) avant de l'inclure.
#ifndef ENV_EXPOSURE_GLSL
#define ENV_EXPOSURE_GLSL

#include "render/color_utils.glsl"

uniform float uExposureCompensation;   // EV ajoutés à l'exposition automatique provisoire
uniform float uNightAdaptation;        // [0, 1] part de l'obscurité compensée par l'exposition (EV < 0)
uniform bool  uMoonlightShift;         // léger virage bleu-argenté en basse lumière (garde >= 75 % de saturation)

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


#endif  // ENV_EXPOSURE_GLSL
