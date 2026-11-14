#version 460 core
//  shaders/render/ocean_surface.vert                                   [DEV 2]
//  Océan : grille projetée (Johanson 2004). Une grille régulière en espace écran (légèrement plus large
//  que l'écran pour que les sommets déplacés ne découvrent pas les bords) est projetée sur le plan
//  y = niveau de l'océan (uEnv.waterTime.x), puis déplacée par les cascades FFT. La densité des sommets
//  suit donc l'écran, sans couture ni niveau de détail à gérer.
//  Anti-crénelage du maillage : chaque cascade est lue au mip correspondant à l'empreinte d'un sommet et
//  s'efface quand ses vagues sont plus courtes que deux sommets (elles restent dans la normale par pixel).
#include "common/global_ubo.glsl"
#include "render/ocean_common.glsl"

layout(location = 0) in vec2 aGrid;  // [0, 1]²

layout(binding = TEX_OceanDisplacement) uniform sampler2DArray uOceanMaps;

uniform float uGridMargin;    // extension au-delà de l'écran (fraction du demi-écran)
uniform vec2  uGridCells;     // cellules de la grille (x, y)
uniform float uMaxDistance;   // m : au-delà, la grille est plaquée sur l'horizon

out VS_OUT {
    vec3  worldPos;    // position déplacée (relative à l'origine de rendu)
    vec2  x0;          // position xz NON déplacée : coordonnée de lecture des cartes
    vec2  flow;        // m/s (rivières)
    float waterSdf;    // m : distance au bord de l'eau (rivières / marais), < 0 dans l'eau
    float marsh;       // [0, 1] : part de marais (turbidité)
    float valid;       // 1 = sommet porteur d'eau
} vsOut;

void main() {
    vec2  ndc   = (aGrid * 2.0 - 1.0) * (1.0 + uGridMargin);
    vec4  nearP = uEnv.invViewProj * vec4(ndc, 1.0, 1.0);  // Reverse-Z : z = 1 au plan proche
    vec3  cam   = envCameraPosition();
    vec3  dir   = normalize(nearP.xyz / nearP.w - cam);
    float level = envOceanLevel();
    float h     = cam.y - level;  // > 0 : caméra au-dessus de l'eau

    // Intersection rayon / plan, bornée à uMaxDistance (rayons vers l'horizon ou au-dessus).
    float minSlope = abs(h) / uMaxDistance;
    float dy = h >= 0.0 ? min(dir.y, -minSlope) : max(dir.y, minSlope);
    float t  = abs(h) / max(abs(dy), 1e-6);
    vec3  d2 = normalize(vec3(dir.x, 0.0, dir.z)) * sqrt(max(1.0 - dy * dy, 0.0));
    vec3  p0 = vec3(cam.x + d2.x * t, level, cam.z + d2.z * t);

    // Empreinte (m) d'une cellule de la grille sur l'eau, étirée par l'incidence rasante.
    float cellAngle = 2.0 * (1.0 + uGridMargin) / uGridCells.y * uEnv.cameraParams.z;
    float footprint = t * cellAngle / max(abs(dy), 0.02);

    vec3 disp = vec3(0.0);
    for (int c = 0; c < OCEAN_CASCADES; ++c) {
        float L      = uOceanCascadeSize[c];
        float texel  = L / float(uOceanN);
        float lod    = log2(max(footprint / texel, 1.0));
        float weight = 1.0 - smoothstep(0.25 * L, 0.5 * L, footprint);
        if (weight <= 0.0) continue;
        disp += textureLod(uOceanMaps, OceanUv(p0.xz, c, 0), lod).xyz * weight;
    }
    disp *= 1.0 - smoothstep(0.5 * uMaxDistance, uMaxDistance, t);

    vec3 P = p0 + disp;
    vsOut.worldPos = P;
    vsOut.x0       = p0.xz;
    vsOut.flow     = vec2(0.0);
    vsOut.waterSdf = -1.0e4;
    vsOut.marsh    = 0.0;
    vsOut.valid    = 1.0;
    gl_Position    = uEnv.viewProj * vec4(P, 1.0);
}
