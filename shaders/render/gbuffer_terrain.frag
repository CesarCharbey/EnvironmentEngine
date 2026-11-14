#version 460 core
//  shaders/render/gbuffer_terrain.frag                                 [DEV 2]
//  Matériau de terrain procédural (sans textures) -> G-Buffer.
//    * Albédo de sol par biome, fondu bilinéaire entre texels (l'id se lit avec texelFetch).
//    * Roche selon la pente (N.y), lits de rivière / marais boueux et mouillés.
//    * Pluie : sol mouillé (terrain_material.glsl), flaques et impacts de gouttes (terrain_wetness.glsl).
//    * Sorties (contrat gbuffer.*) : 0 albédo sRGB + AO, 1 normale monde, 2 matériau, 3 vitesse NDC.
#include "common/global_ubo.glsl"
#include "render/terrain_material.glsl"
#include "render/terrain_wetness.glsl"

in VS_OUT {
    vec3 worldPos;
    vec2 uv;
    vec4 clipCurr;
    vec4 clipPrev;
} fsIn;

layout(binding = TEX_HeightNormal) uniform sampler2D uHeightNormal;
layout(binding = TEX_BiomeHydro)   uniform sampler2D uBiomeHydro;

uniform vec3 uChunkOffset;  // coin du chunk relatif à l'origine de rendu (m), partagé avec le vertex shader

layout(location = 0) out vec4 oAlbedo;    // SRGB8_ALPHA8 : rgb albédo (linéaire, encodé par GL_FRAMEBUFFER_SRGB), a AO
layout(location = 1) out vec4 oNormal;    // RGBA16F : xyz normale monde
layout(location = 2) out vec4 oMaterial;  // RGBA8 : r rugosité, g métal, b humidité de surface, a id matériau / 255
layout(location = 3) out vec2 oVelocity;  // RG16F : déplacement NDC (courant - précédent)

void main() {
    // Empreinte d'un pixel au sol (m), calculée avant tout branchement (dérivées).
    float footprint = length(fwidth(fsIn.worldPos.xz));

    vec4 hn = textureLod(uHeightNormal, fsIn.uv, 0.0);
    vec3 N  = normalize(hn.gba);  // par pixel : plus net qu'interpolé
    // Matériau partagé avec la voxélisation VXGI (render/terrain_material.glsl).
    TerrainMaterial m = EvaluateTerrainMaterial(uBiomeHydro, fsIn.uv, N, fsIn.worldPos.y);

    // Flaques + impacts de pluie (sans effet, et sans coût, par temps sec).
    WetSurface ws = WetSurface(N, m.roughness, m.wetness, m.albedo);
    if (uEnv.weatherExtra.w > 0.35) {
        ivec2 chunk = ivec2(round(uChunkOffset.xz / CHUNK_SIZE_METERS)) + uEnv.originChunk.xy;
        vec4 bh = textureLod(uBiomeHydro, fsIn.uv, 0.0);
        ws = ApplyPuddlesAndRipples(ws, uHeightNormal, chunk, fsIn.uv, hn.r, m.biome, min(bh.g, bh.b), footprint);
    }

    vec2 ndcCurr = fsIn.clipCurr.xy / fsIn.clipCurr.w;
    vec2 ndcPrev = fsIn.clipPrev.xy / fsIn.clipPrev.w;

    oAlbedo   = vec4(ws.albedo, 1.0);
    oNormal   = vec4(ws.N, 0.0);
    oMaterial = vec4(ws.roughness, 0.0, ws.wetness, float(m.biome) / 255.0);
    oVelocity = ndcCurr - ndcPrev;
}
