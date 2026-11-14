#version 460 core
//  shaders/render/water_depth.frag                                     [DEV 2]
//  Recopie gbuffer.depth dans scene.depth (triangle plein écran, écriture de couleur coupée) : l'eau est
//  ensuite testée et écrite contre la vraie profondeur opaque (scene.depth contient sinon l'aperçu du
//  mock). Après l'eau, scene.depth = profondeur finale (opaque + eau) pour les passes suivantes.
#include "common/global_ubo.glsl"

in vec2 vUv;
layout(binding = TEX_GBufferDepth) uniform sampler2D uGDepth;

void main() {
    gl_FragDepth = textureLod(uGDepth, vUv, 0.0).r;
}
