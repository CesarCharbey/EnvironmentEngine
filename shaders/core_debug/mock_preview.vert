#version 460 core
// =============================================================================
//  shaders/core_debug/mock_preview.vert  (core, gele)
//  Apercu de debug de TOUS les chunks prets (mock ET vrais chunks de Dev 1).
//  Exemple de reference pour Dev 2 :
//    * terrain : grille [0,1]^2 deplacee par heightNormalTex (TEX_HeightNormal)
//    * vegetation : glMultiDrawElementsIndirect, instances lues dans le SSBO
//      SSBO_FoliageInstances a l'index gl_BaseInstance + gl_InstanceID
// =============================================================================
#include "common/global_ubo.glsl"

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

layout(binding = TEX_HeightNormal) uniform sampler2D uHeightNormal;

layout(std430, binding = SSBO_FoliageInstances) readonly buffer FoliageInstanceBuffer {
    FoliageInstance instances[];
};

uniform int  uDrawMode;     // 0 = terrain, 1 = instances de vegetation (indirect)
uniform vec3 uChunkOrigin;  // coin du chunk relatif a l'origine de rendu (m)

out VS_OUT {
    vec3 worldPos;          // relatif a l'origine de rendu
    vec3 normal;
    vec2 chunkUV;
    flat int archetype;     // -1 pour le terrain
} vsOut;

void main() {
    vec3 position;
    vec3 normal;
    vec2 uv;
    int  archetype = -1;

    if (uDrawMode == 0) {
        uv = aPosition.xz;
        vec4 hn  = textureLod(uHeightNormal, uv, 0.0);
        position = uChunkOrigin + vec3(uv.x * CHUNK_SIZE_METERS, hn.r, uv.y * CHUNK_SIZE_METERS);
        normal   = normalize(hn.gba);
    } else {
        FoliageInstance inst = instances[gl_BaseInstance + gl_InstanceID];
        archetype = int(inst.params.y + 0.5);
        // Boite de debug par archetype (largeur, hauteur, profondeur) en metres.
        vec3 extent = vec3(0.35, 0.6, 0.35);                         // FOLIAGE_GRASS
        if (archetype == FOLIAGE_SHRUB) extent = vec3(1.4, 1.2, 1.4);
        if (archetype == FOLIAGE_TREE)  extent = vec3(1.2, 7.0, 1.2);
        if (archetype == FOLIAGE_ROCK)  extent = vec3(2.2, 0.9, 1.6);
        float scale = inst.positionScale.w;
        float c = cos(inst.params.x);
        float s = sin(inst.params.x);
        vec3 local = aPosition * extent * scale;
        local.y += 0.5 * extent.y * scale;  // pose la boite sur le sol
        vec3 rotated = vec3(c * local.x - s * local.z, local.y, s * local.x + c * local.z);
        normal   = vec3(c * aNormal.x - s * aNormal.z, aNormal.y, s * aNormal.x + c * aNormal.z);
        position = uChunkOrigin + inst.positionScale.xyz + rotated;
        uv       = inst.positionScale.xz / CHUNK_SIZE_METERS;
    }

    vsOut.worldPos  = position;
    vsOut.normal    = normal;
    vsOut.chunkUV   = uv;
    vsOut.archetype = archetype;
    gl_Position = uEnv.viewProj * vec4(position, 1.0);
}
