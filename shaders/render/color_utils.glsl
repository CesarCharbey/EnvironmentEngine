//  shaders/render/color_utils.glsl                                     [DEV 2]
//  Utilitaires de couleur partagés (luminance photométrique, encodage sRGB).
#ifndef ENV_COLOR_UTILS_GLSL
#define ENV_COLOR_UTILS_GLSL

float Luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

vec3 LinearToSrgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

#endif  // ENV_COLOR_UTILS_GLSL
