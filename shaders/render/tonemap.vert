#version 460 core
//  shaders/render/tonemap.vert                                         [DEV 2]
//  Triangle plein écran (gl_VertexID, VAO vide). L'exposition automatique ne dépend que de l'UBO et des
//  LUTs d'atmosphère : elle est calculée ici, une fois par sommet, au lieu d'une fois par pixel.
#include "common/global_ubo.glsl"
#include "render/atmo_common.glsl"
#include "render/sky_lighting.glsl"

uniform float uSunScale;
uniform float uMoonScale;
#include "render/exposure.glsl"

uniform bool  uAutoExposure;
uniform float uManualEv100;

out vec2 vUv;
flat out float vExposure;

void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));  // (0,0) (2,0) (0,2)
    vUv = p;
    vExposure = uAutoExposure ? FallbackExposure() : 1.0 / (1.2 * exp2(uManualEv100));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
