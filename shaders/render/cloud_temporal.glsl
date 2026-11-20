//  shaders/render/cloud_temporal.glsl                                  [DEV 2]
//  Reprojection temporelle des nuages (Schneider 2015) : un pixel sur B×B de la cible en demi-résolution
//  est raymarché chaque frame (motif de Bayer indexé par la frame), les autres sont reprojetés depuis
//  l'historique. Partagé par le raymarching (volumetric_clouds.comp : mélange du pixel frais avec son
//  historique) et la résolution (cloud_reproject.comp : pixels non raymarchés).
//
//  Historique : cible de la frame précédente, lue en bilinéaire sur le slot TEX_GodRays (même système,
//  demi-résolution, inutilisé tant que les god rays n'existent pas : pas de nouveau slot au contrat).
//  Le shader qui inclut ce fichier doit avoir inclus "common/global_ubo.glsl", "render/atmo_common.glsl"
//  et "render/cloud_density.glsl".
#ifndef ENV_CLOUD_TEMPORAL_GLSL
#define ENV_CLOUD_TEMPORAL_GLSL

layout(binding = TEX_GodRays) uniform sampler2D uCloudHistory;  // rgb luminance, a transmittance

uniform int   uBlock;          // côté du motif (1 = tout est raymarché : pas d'historique cette frame)
uniform ivec2 uBlockOffset;    // pixel raymarché dans chaque bloc cette frame (motif de Bayer)
uniform bool  uHistoryValid;   // false : premier rendu, redimensionnement, saut d'heure / de météo
uniform mat4  uPrevViewProj;   // viewProj de la frame précédente (repère de SON origine de rendu)
uniform vec2  uOriginShift;    // m : (origine courante - origine précédente) en xz

// Distance représentative des nuages le long du rayon : entrée dans la couche (parallaxe). -1 si le rayon
// ne la rencontre pas (direction traitée comme à l'infini).
float CloudLayerDistance(vec3 dir) {
    float Rm = (AtmoValid() ? uAtmo.uRadiiAndScaleHeights.x : 6360.0) * 1000.0;
    vec3  pc = vec3(0.0, envCameraPosition().y + Rm, 0.0);
    float hb = Rm + uCloudBottom, ht = Rm + uCloudTop;
    float b  = dot(pc, dir);
    float r2 = dot(pc, pc);
    float h  = length(pc);
    if (h < hb) {                              // sous la couche : sortie de la sphère de base
        return -b + sqrt(max(b * b - (r2 - hb * hb), 0.0));
    }
    if (h <= ht) return 0.5 * (uCloudTop - uCloudBottom);  // dans la couche
    float disc = b * b - (r2 - ht * ht);       // au-dessus : entrée par le sommet
    return disc >= 0.0 && -b - sqrt(disc) > 0.0 ? -b - sqrt(disc) : -1.0;
}

// UV de la frame précédente pour le rayon de direction `dir` (repère courant).
vec2 ReprojectCloudUv(vec3 dir) {
    float d = CloudLayerDistance(dir);
    vec4  clip;
    if (d > 0.0) {
        vec3 X = envCameraPosition() + dir * d;
        X.xz += uOriginShift;  // positions relatives à l'origine de rendu de la frame précédente
        clip = uPrevViewProj * vec4(X, 1.0);
    } else {
        clip = uPrevViewProj * vec4(dir, 0.0);  // point à l'infini : rotation seule
    }
    if (clip.w <= 1e-6) return vec2(-1.0);
    return clip.xy / clip.w * 0.5 + 0.5;
}

bool UvInside(vec2 uv) { return all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0))); }

#endif  // ENV_CLOUD_TEMPORAL_GLSL
