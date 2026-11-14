#version 460 core
//  shaders/render/gbuffer_foliage.vert                                 [DEV 2]
//  Végétation instanciée (glMultiDrawElementsIndirect, une commande par archétype).
//  Aucun vertex buffer : l'element buffer du VAO est l'identité (0, 1, 2, ...), donc gl_VertexID est
//  l'index du sommet. Forme procédurale de 36 sommets (DrawElementsIndirectCommand::count = 36, comme
//  le mock) : boîte [-0.5, 0.5]³ effilée vers le haut, dimensionnée par archétype.
//  Instance : instances[gl_BaseInstance + gl_InstanceID] (ADR-001).
#include "common/global_ubo.glsl"

layout(std430, binding = SSBO_FoliageInstances) readonly buffer FoliageInstanceBuffer {
    FoliageInstance instances[];
};

uniform vec3  uChunkOffset;   // coin du chunk relatif à l'origine de rendu (m)
uniform float uWindStrength;  // multiplicateur global de l'ondulation

out VS_OUT {
    vec3 normal;
    float worldY;          // altitude (m) : neige
    vec4 clipCurr;
    vec4 clipPrev;
    flat int   archetype;
    flat float variation;  // aléa de l'instance [0, 1)
} vsOut;

// Faces de la boîte : normale n, axes u, v avec u × v = n (sommets CCW vus de l'extérieur).
const vec3 kN[6] = vec3[6](vec3( 1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0), vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1));
const vec3 kU[6] = vec3[6](vec3( 0, 0,-1), vec3( 0, 0, 1), vec3(1, 0, 0), vec3(1,  0, 0), vec3(1, 0, 0), vec3(-1, 0, 0));
const vec3 kV[6] = vec3[6](vec3( 0, 1, 0), vec3( 0, 1, 0), vec3(0, 0,-1), vec3(0,  0, 1), vec3(0, 1, 0), vec3( 0, 1, 0));
const int  kCorner[6]  = int[6](0, 1, 2, 0, 2, 3);
const vec2 kCornerUV[4] = vec2[4](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, 1));

void main() {
    FoliageInstance inst = instances[gl_BaseInstance + gl_InstanceID];
    int   archetype = int(inst.params.y + 0.5);
    float scale     = inst.positionScale.w;

    // --- Sommet de la boîte unitaire ---
    int  vid  = gl_VertexID % 36;
    int  face = vid / 6;
    vec2 cs   = kCornerUV[kCorner[vid % 6]];
    vec3 p    = 0.5 * kN[face] + 0.5 * (kU[face] * cs.x + kV[face] * cs.y);
    vec3 n    = kN[face];

    // --- Forme par archétype : dimensions (m) et effilement du sommet ---
    vec3  extent = vec3(0.35, 0.6, 0.35);  // herbe
    float taper  = 0.15;
    float stiffness = 0.08;                // ondulation par m/s de vent et par m de hauteur
    if (archetype == FOLIAGE_SHRUB) { extent = vec3(1.4, 1.2, 1.4); taper = 0.55; stiffness = 0.03;  }
    if (archetype == FOLIAGE_TREE)  { extent = vec3(2.2, 7.0, 2.2); taper = 0.10; stiffness = 0.012; }  // conifère
    if (archetype == FOLIAGE_ROCK)  { extent = vec3(2.2, 0.9, 1.6); taper = 0.70; stiffness = 0.0;   }

    float h01 = p.y + 0.5;                  // 0 au sol, 1 au sommet
    p.xz *= mix(1.0, taper, h01);
    if (abs(n.y) < 0.5) {                   // les flancs d'une forme effilée regardent un peu vers le haut
        n.y = (1.0 - taper) * 0.5 * max(extent.x, extent.z) / extent.y;
        n = normalize(n);
    }

    vec3 local = p * extent * scale;
    local.y += 0.5 * extent.y * scale;      // pose la forme sur le sol

    // --- Orientation (yaw) ---
    float c = cos(inst.params.x), s = sin(inst.params.x);
    local = vec3(c * local.x - s * local.z, local.y, s * local.x + c * local.z);
    n     = vec3(c * n.x - s * n.z, n.y, s * n.x + c * n.z);

    vec3 basePos = uChunkOffset + inst.positionScale.xyz;
    vec3 pos     = basePos + local;

    // --- Vent : flexion proportionnelle à h01² (pied fixe), oscillation déphasée par instance ---
    float phase = inst.params.w * 6.2831853 + dot(basePos.xz, vec2(0.05, 0.07));
    float gust  = 1.0 + uEnv.windGust.y * sin(envTime() * 6.2831853 * uEnv.windGust.x + phase * 0.5);
    float sway  = 0.65 + 0.35 * sin(envTime() * (1.3 + uEnv.windDirSpeed.w * 2.0) + phase);
    float bend  = envWindSpeed() * stiffness * uWindStrength * extent.y * scale * h01 * h01 * sway * gust;
    vec2  windDir = envWindDirection();
    pos.xz += windDir * bend;
    // Correction (approchée) du temps précédent : on réutilise le même décalage, la vitesse NDC
    // ne capture donc que le mouvement de caméra pour la végétation.

    vsOut.normal    = n;
    vsOut.worldY    = pos.y;
    vsOut.archetype = archetype;
    vsOut.variation = inst.params.w;
    vsOut.clipCurr  = uEnv.viewProj * vec4(pos, 1.0);
    vsOut.clipPrev  = uEnv.prevViewProj * vec4(pos, 1.0);
    gl_Position     = vsOut.clipCurr;
}
