#version 460 core
//  shaders/render/gbuffer_terrain.frag                                 [DEV 2]
//  Matériau de terrain procédural (sans textures) -> G-Buffer.
//    * Albédo de sol par biome, fondu bilinéaire entre texels (l'id se lit avec texelFetch).
//    * Roche selon la pente (N.y), lits de rivière / marais boueux et mouillés.
//    * Sorties (contrat gbuffer.*) : 0 albédo sRGB + AO, 1 normale monde, 2 matériau, 3 vitesse NDC.
#include "common/global_ubo.glsl"
#include "render/terrain_material.glsl"

in VS_OUT {
    vec3 worldPos;
    vec2 uv;
    vec4 clipCurr;
    vec4 clipPrev;
} fsIn;

layout(binding = TEX_HeightNormal) uniform sampler2D uHeightNormal;
layout(binding = TEX_BiomeHydro)   uniform sampler2D uBiomeHydro;

layout(location = 0) out vec4 oAlbedo;    // SRGB8_ALPHA8 : rgb albédo (linéaire, encodé par GL_FRAMEBUFFER_SRGB), a AO
layout(location = 1) out vec4 oNormal;    // RGBA16F : xyz normale monde
layout(location = 2) out vec4 oMaterial;  // RGBA8 : r rugosité, g métal, b humidité de surface, a id matériau / 255
layout(location = 3) out vec2 oVelocity;  // RG16F : déplacement NDC (courant - précédent)

void main() {
    vec3 N = normalize(textureLod(uHeightNormal, fsIn.uv, 0.0).gba);  // par pixel : plus net qu'interpolé
    // Matériau partagé avec la voxélisation VXGI (render/terrain_material.glsl).
    TerrainMaterial m = EvaluateTerrainMaterial(uBiomeHydro, fsIn.uv, N, fsIn.worldPos.y);

    vec2 ndcCurr = fsIn.clipCurr.xy / fsIn.clipCurr.w;
    vec2 ndcPrev = fsIn.clipPrev.xy / fsIn.clipPrev.w;

    oAlbedo   = vec4(m.albedo, 1.0);
    oNormal   = vec4(N, 0.0);
    oMaterial = vec4(m.roughness, 0.0, m.wetness, float(m.biome) / 255.0);
    oVelocity = ndcCurr - ndcPrev;
}
