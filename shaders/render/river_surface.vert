#version 460 core
//  shaders/render/river_surface.vert                                   [DEV 2]
//  Rivières et marais d'un chunk : grille de 128 × 128 quads (2 m) dont chaque sommet lit sa surface
//  précalculée (river_surface.comp, SSBO_Dev2Scratch). Marais : niveau global uEnv.waterTime.y là où la
//  SDF du marais est proche. Sans eau à proximité, le sommet est enfoui 10 m sous le terrain : le test
//  de profondeur le cache, et le fragment shader rejette ce qui dépasserait hors de l'eau.
#include "common/global_ubo.glsl"

layout(location = 0) in vec2 aGrid;  // indices de sommet (i, j) dans [0, 128]

layout(std430, binding = SSBO_Dev2Scratch) readonly buffer RiverSurfaceBuffer {
    vec4 riverData[];
};

uniform uint uSlotBase;     // premier sommet du chunk dans le buffer
uniform vec3 uChunkOffset;  // coin du chunk relatif à l'origine de rendu (m)
uniform bool uRivers;
uniform bool uSwamps;

out VS_OUT {
    vec3  worldPos;
    vec2  x0;
    vec2  flow;
    float waterSdf;
    float marsh;
    float valid;
} vsOut;

const float kNoRiver    = -1.0e4;
// m : bande où la géométrie porte encore le niveau du marais ; le contour exact (SDF = 0) est découpé
// par pixel dans le fragment shader sur la SDF interpolée (bord lisse, pas les dents de la grille de 2 m).
const float kMarshReach = 6.0;

void main() {
    ivec2 v    = ivec2(aGrid + 0.5);
    vec4  d    = riverData[uSlotBase + uint(v.y * 129 + v.x)];
    vec2  sdfs = unpackHalf2x16(floatBitsToUint(d.z));  // (rivière, marais)
    vec2  flow = unpackHalf2x16(floatBitsToUint(d.w));

    float riverLevel = uRivers && d.x > kNoRiver * 0.5 ? d.x : kNoRiver;
    // +2 cm : le marais du mock est aplani exactement au niveau d'eau par défaut (pas de z-fighting).
    float marshLevel = uSwamps && sdfs.y < kMarshReach ? envSwampLevel() + 0.02 : kNoRiver;
    float level      = max(riverLevel, marshLevel);
    bool  hasWater   = level > kNoRiver * 0.5;
    float marsh      = marshLevel >= riverLevel && hasWater ? 1.0 : 0.0;

    vec3 P = uChunkOffset + vec3(float(v.x) * 2.0, hasWater ? level : d.y - 10.0, float(v.y) * 2.0);
    vsOut.worldPos = P;
    vsOut.x0       = P.xz;
    vsOut.flow     = marsh > 0.5 ? vec2(0.0) : flow;
    vsOut.waterSdf = min(uRivers ? sdfs.x : 1.0e4, uSwamps ? sdfs.y : 1.0e4);
    vsOut.marsh    = marsh;
    vsOut.valid    = hasWater ? 1.0 : 0.0;
    gl_Position    = uEnv.viewProj * vec4(P, 1.0);
}
