#version 460 core
//  shaders/render/tonemap.frag                                         [DEV 2]
//  Exposition + tonemapping : scene.color (HDR linéaire, cd/m²) -> scene.display (RGBA8, sRGB encodé
//  à la main : le Viewport affiche la texture telle quelle).
//  Exposition (tonemap.vert) automatique PROVISOIRE (exposure.glsl : éclairement physique de la scène,
//  sans lecture de l'image) en attendant l'histogramme de luminance, ou manuelle (EV100).
#include "common/global_ubo.glsl"
#include "render/atmo_common.glsl"
#include "render/sky_lighting.glsl"

uniform float uSunScale;   // mêmes multiplicateurs que l'éclairage (HorizontalIlluminance)
uniform float uMoonScale;
#include "render/exposure.glsl"

in vec2 vUv;
flat in float vExposure;  // calculée par tonemap.vert
layout(location = 0) out vec4 oColor;

layout(binding = TEX_SceneColor)    uniform sampler2D uSceneColor;
layout(binding = TEX_SceneDepth)    uniform sampler2D uSceneDepth;  // opaque + eau (WaterOceanRiverSystem)
// Nuages (volumetric_clouds.comp, résolution réduite) : rgb = luminance, a = transmittance.
// Liée sur TEX_CloudWeatherMap (inutilisée par le tonemap) pour bénéficier du filtrage bilinéaire matériel.
layout(binding = TEX_CloudWeatherMap) uniform sampler2D uCloudsTex;

uniform bool  uCloudsAvailable;
uniform bool  uGodRaysAvailable;
// God rays (volumetric_godrays.comp, demi-résolution) : rgb luminance diffusée par le brouillard (géométrie),
// a facteur des rayons crépusculaires appliqué à la sky-view (ciel seulement ; 1 sur la géométrie).
layout(binding = TEX_GodRays) uniform sampler2D uGodRays;
uniform float uCloudBottom;  // m : base de la couche (rejet du relief situé devant les nuages)

uniform bool  uDataView;      // vue de debug « données brutes » : ni exposition ni tonemap
uniform int   uTonemapper;    // 0 ACES, 1 AgX, 2 Reinhard étendu, 3 Khronos PBR Neutral
uniform float uWhitePoint;    // Reinhard étendu : valeur qui devient blanche

// AgX (Sobotka 2023), approximation polynomiale de la courbe (Wrensch 2023), look neutre.
vec3 AgX(vec3 c) {
    const mat3 kIn  = mat3(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                           0.0784335999999992, 0.878468636469772, 0.0784336,
                           0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    const mat3 kOut = mat3(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                           -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                           -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
    const float kMinEv = -12.47393, kMaxEv = 4.026069;
    c = kIn * max(c, vec3(1e-10));
    c = clamp((log2(c) - kMinEv) / (kMaxEv - kMinEv), 0.0, 1.0);
    vec3 x2 = c * c, x4 = x2 * x2;
    c = 15.5 * x4 * x2 - 40.14 * x4 * c + 31.96 * x4 - 6.868 * x2 * c + 0.4298 * x2 + 0.1191 * c - 0.00232;
    return clamp(kOut * c, 0.0, 1.0);  // déjà proche de l'affichage : on le traite comme linéaire sRGB
}

vec3 ReinhardExtended(vec3 c, float white) {
    float L  = Luminance(c);
    float Ld = L * (1.0 + L / (white * white)) / (1.0 + L);
    return clamp(c * (Ld / max(L, 1e-6)), 0.0, 1.0);
}

// Khronos PBR Neutral (2024) : conserve les teintes et les couleurs de base jusqu'à ~0,8.
vec3 PbrNeutral(vec3 c) {
    const float kStart = 0.8 - 0.04, kDesat = 0.15;
    float x = min(c.r, min(c.g, c.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    c -= offset;
    float peak = max(c.r, max(c.g, c.b));
    if (peak < kStart) return c;
    float d = 1.0 - kStart;
    float newPeak = 1.0 - d * d / (peak + d - kStart);
    c *= newPeak / peak;
    float g = 1.0 - 1.0 / (kDesat * (peak - newPeak) + 1.0);
    return mix(c, vec3(newPeak), g);
}

// Le relief (ou l'eau) situé devant la base des nuages les cache, au pixel près.
bool GeometryInFrontOfClouds(vec2 uv, float d) {
    if (envIsSky(d)) return false;
    vec3  cam  = envCameraPosition();
    if (cam.y >= uCloudBottom) return false;  // caméra dans / au-dessus de la couche : la marche a borné
    vec3  dir  = envViewRayWorld(uv);
    float Rm   = (AtmoValid() ? uAtmo.uRadiiAndScaleHeights.x : 6360.0) * 1000.0;
    vec3  pc   = vec3(0.0, cam.y + Rm, 0.0);
    float R    = Rm + uCloudBottom;
    float b    = dot(pc, dir), c = dot(pc, pc) - R * R;
    float tIn  = -b + sqrt(max(b * b - c, 0.0));  // sortie de la sphère de base (caméra dedans)
    return length(envWorldPosFromDepth(uv, d) - cam) < tIn;
}

float LinearSceneDepth(vec2 uv) {
    float d = textureLod(uSceneDepth, uv, 0.0).r;
    return envIsSky(d) ? 1.0e7 : envLinearDepth(d);
}

// Suréchantillonnage bilatéral en croix (5 taps) des god rays : poids spatial × ressemblance de profondeur
// (pas de halo autour des montagnes), et lissage du jitter IGN de la marche.
vec4 GodRaysAt(vec2 uv, float centerDepth) {
    ivec2 size = textureSize(uGodRays, 0);
    vec2  st   = uv * vec2(size) - 0.5;
    ivec2 c    = ivec2(floor(st + 0.5));
    float z0   = envIsSky(centerDepth) ? 1.0e7 : envLinearDepth(centerDepth);
    vec4  acc  = vec4(0.0);
    float wsum = 0.0;
    const ivec2 kOffsets[5] = ivec2[5](ivec2(0, 0), ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));
    for (int i = 0; i < 5; ++i) {
        ivec2 q  = clamp(c + kOffsets[i], ivec2(0), size - 1);
        vec2  dq = vec2(q) - st;
        float zq = LinearSceneDepth((vec2(q) + 0.5) / vec2(size));
        float dz = abs(zq - z0) / max(min(z0, zq), 1e-3);
        float w  = exp(-dot(dq, dq) * 0.9 - min(dz, 4.0) * 6.0) + 1e-5;
        acc  += texelFetch(uGodRays, q, 0) * w;
        wsum += w;
    }
    return acc / wsum;
}

void main() {
    vec3 hdr = texture(uSceneColor, vUv).rgb;
    if (uDataView) {
        oColor = vec4(LinearToSrgb(hdr), 1.0);
        return;
    }
    float sceneDepth = textureLod(uSceneDepth, vUv, 0.0).r;
    bool  skyPixel   = envIsSky(sceneDepth);
    vec4  gr         = uGodRaysAvailable ? GodRaysAt(vUv, sceneDepth) : vec4(0.0, 0.0, 0.0, 1.0);
    if (skyPixel) hdr *= gr.a;  // rayons crépusculaires dans la brume (sky-view seule, avant les nuages)
    if (uCloudsAvailable && !GeometryInFrontOfClouds(vUv, sceneDepth)) {
        vec4 cl = textureLod(uCloudsTex, vUv, 0.0);
        hdr = hdr * cl.a + cl.rgb;
    }
    hdr += gr.rgb;  // soleil diffusé par le brouillard, ombré par les nuages et le relief (géométrie)
    if (uMoonlightShift) hdr = MoonlightShift(hdr);
    vec3 c = hdr * vExposure;
    if      (uTonemapper == 1) c = AgX(c);
    else if (uTonemapper == 2) c = ReinhardExtended(c, uWhitePoint);
    else if (uTonemapper == 3) c = PbrNeutral(c);
    else                       c = AcesFilmic(c);
    oColor = vec4(LinearToSrgb(c), 1.0);
}
