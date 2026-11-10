#version 460 core
//  shaders/render/gbuffer_terrain.vert                                 [DEV 2]
//  Grille de terrain d'un chunk déplacée par heightNormalTex (R = hauteur absolue en m).
//  aGrid.xz : position locale (m) dans [0, CHUNK_SIZE_METERS] ; aGrid.y : 1 = sommet de jupe.
//  Les jupes (skirts) descendent de uSkirtDepth sous le bord du chunk : elles masquent les fentes
//  entre chunks voisins (chaque chunk ne connaît que ses texels, les bords ne coïncident pas exactement).
#include "common/global_ubo.glsl"

layout(location = 0) in vec3 aGrid;

layout(binding = TEX_HeightNormal) uniform sampler2D uHeightNormal;

uniform vec3  uChunkOffset;  // coin du chunk relatif à l'origine de rendu (m)
uniform float uSkirtDepth;   // m

out VS_OUT {
    vec3 worldPos;   // relatif à l'origine de rendu (y absolu)
    vec2 uv;         // UV du chunk [0, 1]
    vec4 clipCurr;
    vec4 clipPrev;
} vsOut;

void main() {
    vec2  uv = aGrid.xz / CHUNK_SIZE_METERS;
    float h  = textureLod(uHeightNormal, uv, 0.0).r;
    vec3  pos = uChunkOffset + vec3(aGrid.x, h - aGrid.y * uSkirtDepth, aGrid.z);

    vsOut.worldPos = pos;
    vsOut.uv       = uv;
    vsOut.clipCurr = uEnv.viewProj * vec4(pos, 1.0);
    vsOut.clipPrev = uEnv.prevViewProj * vec4(pos, 1.0);
    gl_Position    = vsOut.clipCurr;
}
