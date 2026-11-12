//  shaders/render/ibl_common.glsl                                      [DEV 2]
//  Outils partagés par les passes de l'IBL dynamique (DynamicIBLSystem) : directions des texels
//  d'une cubemap, angle solide d'un texel, échantillonnage d'importance GGX (Hammersley).
#ifndef ENV_IBL_COMMON_GLSL
#define ENV_IBL_COMMON_GLSL

const float kIblPi = 3.14159265358979323846;

// Coordonnées de face [-1, 1] du centre du texel (x, y) d'une face de taille size.
vec2 CubeFaceUv(ivec2 texel, int size) {
    return (vec2(texel) + 0.5) / float(size) * 2.0 - 1.0;
}

// Direction monde (non normalisée) d'un point (u, v) de la face, convention OpenGL :
// faces 0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z ; v croît vers le BAS de la face (coordonnée t).
vec3 CubeFaceDir(int face, vec2 uv) {
    switch (face) {
        case 0:  return vec3( 1.0, -uv.y, -uv.x);
        case 1:  return vec3(-1.0, -uv.y,  uv.x);
        case 2:  return vec3( uv.x,  1.0,  uv.y);
        case 3:  return vec3( uv.x, -1.0, -uv.y);
        case 4:  return vec3( uv.x, -uv.y,  1.0);
        default: return vec3(-uv.x, -uv.y, -1.0);
    }
}

vec3 CubeTexelDir(int face, ivec2 texel, int size) {
    return normalize(CubeFaceDir(face, CubeFaceUv(texel, size)));
}

// Angle solide (sr) du texel centré en uv, de côté 2 / size en coordonnées de face.
float CubeTexelSolidAngle(vec2 uv, int size) {
    float d = 2.0 / float(size);
    float r = 1.0 + dot(uv, uv);
    return d * d / (r * sqrt(r));
}

// Suite de Hammersley : déterministe -> la même IBL d'un cycle à l'autre (aucun bruit temporel).
float RadicalInverseVdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

vec2 Hammersley(uint i, uint count) {
    return vec2(float(i) / float(count), RadicalInverseVdC(i));
}

// Demi-vecteur H tiré selon D_GGX(H) (N·H), dans le repère de la normale N. alpha = rugosité².
vec3 ImportanceSampleGGX(vec2 xi, float alpha, vec3 N) {
    float a2       = alpha * alpha;
    float phi      = 2.0 * kIblPi * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    vec3  h        = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 T  = normalize(cross(up, N));
    vec3 B  = cross(N, T);
    return normalize(T * h.x + B * h.y + N * h.z);
}

float IblD_GGX(float NoH, float alpha) {
    float a2 = alpha * alpha;
    float f  = (NoH * a2 - NoH) * NoH + 1.0;
    return a2 / (kIblPi * f * f);
}

#endif  // ENV_IBL_COMMON_GLSL
