//  shaders/render/terrain_wetness.glsl                                 [DEV 2]
//  Flaques et impacts de pluie (ripples) du terrain, évalués dans la G-pass (gbuffer_terrain.frag) :
//  ils modifient directement la normale, la rugosité et l'humidité écrites dans le G-Buffer.
//  L'éclairage n'a rien à faire de plus, et aucune passe ni texture n'est ajoutée.
//
//  * Niveau d'eau : dérivé de l'humidité météo uEnv.weatherExtra.w. En dessous de 0,35 la surface
//    est seulement mouillée (terrain_material.glsl) ; de 0,35 à 1, l'eau s'accumule en flaques.
//  * Où ? Creux plats : cavité locale du heightfield (hauteur moyenne à ±4 m au-dessus du point),
//    pente faible, bruit de valeur à 2 octaves pour des contours naturels. Le sable draine.
//  * Ondes : deux couches de cellules de 25 cm portant chacune au plus une goutte (anneau amorti qui
//    s'élargit), densité et amplitude selon la pluie (uEnv.weather.x). Le rayon max reste dans la
//    cellule : une seule cellule lue par couche (2 hachages par pixel).
//  * Anti-crénelage : chaque détail s'efface quand un pixel couvre plus que sa taille (empreinte
//    fwidth) ; les ondes non résolues deviennent de la rugosité.
//  * Le bruit utilise un réseau ENTIER absolu (chunk × cellules + cellule locale) : continu entre
//    chunks, stable quand l'origine de rendu se recale, sans perte de précision loin de l'origine.
//
//  Le shader qui inclut ce fichier doit avoir inclus "common/global_ubo.glsl".
#ifndef ENV_TERRAIN_WETNESS_GLSL
#define ENV_TERRAIN_WETNESS_GLSL

// Hachage entier 3D -> 3 × 32 bits (pcg3d, Jarzynski & Olano 2020).
uvec3 WetHash3(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}
vec3  WetHash01(ivec2 c, uint seed) { return vec3(WetHash3(uvec3(uvec2(c), seed)) >> 8u) * (1.0 / 16777216.0); }

// Bruit de valeur sur un réseau de `cells` cellules par chunk (puissance de 2), lissé (smoothstep).
float WetValueNoise(ivec2 chunk, vec2 chunkUv, int cells, uint seed) {
    vec2  p  = chunkUv * float(cells);
    vec2  fl = floor(p);
    vec2  f  = p - fl;
    ivec2 i  = chunk * cells + ivec2(fl);
    vec2  u  = f * f * (3.0 - 2.0 * f);
    float a = WetHash01(i, seed).x,               b = WetHash01(i + ivec2(1, 0), seed).x;
    float c = WetHash01(i + ivec2(0, 1), seed).x, d = WetHash01(i + ivec2(1, 1), seed).x;
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Une couche d'impacts : gradient (d hauteur / d xz, sans unité) des anneaux de la cellule courante.
vec2 RippleLayer(ivec2 chunk, vec2 chunkUv, int cells, uint seed, float time, float rain) {
    vec2  p    = chunkUv * float(cells);
    vec2  fl   = floor(p);
    ivec2 cell = chunk * cells + ivec2(fl);
    vec3  h    = WetHash01(cell, seed);
    vec3  h2   = WetHash01(cell, seed + 7u);
    if (h2.x > rain) return vec2(0.0);          // densité de gouttes ∝ pluie

    const float kPeriod = 0.9;                  // s : durée de vie d'un anneau
    float age = fract(time / kPeriod + h.z);    // 0 -> 1
    vec2  d   = (p - fl - 0.5) - (h.xy - 0.5) * 0.16;  // centre décalé de ±8 % de cellule
    float r   = length(d);
    float R   = 0.04 + 0.34 * age;              // rayon max 0,38 + 0,08 < 0,5 : reste dans la cellule
    float x   = r - R;
    const float kWidth = 0.07, kWave = 2.0 * 3.14159265 / 0.11;
    float env = (1.0 - age) * (1.0 - age) * exp(-(x * x) / (kWidth * kWidth));
    float dh  = env * cos(kWave * x) * (0.6 + 0.4 * h2.y);
    return r > 1e-4 ? (d / r) * dh : vec2(0.0);
}

struct WetSurface {
    vec3  N;
    float roughness;
    float wetness;
    vec3  albedo;
};

// chunk : coordonnée absolue du chunk ; chunkUv : UV dans le chunk ; footprint : m couverts par un
// pixel (fwidth de la position monde, calculée AVANT tout branchement) ; h : hauteur du point (m).
// waterSdf : min(SDF rivière, SDF marais) ; les flaques s'arrêtent avant le lit (rendu par l'eau).
// snow : couverture neigeuse ; la neige recouvre les flaques.
WetSurface ApplyPuddlesAndRipples(WetSurface s, sampler2D heightNormal, ivec2 chunk, vec2 chunkUv, float h,
                                  int biome, float waterSdf, float snow, float footprint) {
    float waterLevel = smoothstep(0.35, 1.0, clamp(uEnv.weatherExtra.w, 0.0, 1.0));
    float flatness   = smoothstep(0.93, 0.985, s.N.y) * smoothstep(1.0, 3.0, waterSdf) * (1.0 - snow);
    if (waterLevel <= 0.0 || flatness <= 0.0) return s;

    // Cavité : de combien le point est sous la moyenne de son voisinage (> 0 dans un creux).
    float du   = 4.0 / CHUNK_SIZE_METERS;
    float hAvg = 0.25 * (textureLod(heightNormal, chunkUv + vec2(du, 0.0), 0.0).r
                       + textureLod(heightNormal, chunkUv - vec2(du, 0.0), 0.0).r
                       + textureLod(heightNormal, chunkUv + vec2(0.0, du), 0.0).r
                       + textureLod(heightNormal, chunkUv - vec2(0.0, du), 0.0).r);
    float cavity = clamp((hAvg - h) / 0.3, -1.0, 1.0);

    // Contours : octave de 8 m + octave de 2 m (estompée vers sa moyenne quand elle n'est plus résolue).
    float fine  = mix(WetValueNoise(chunk, chunkUv, 128, 11u), 0.5, smoothstep(0.25, 1.0, footprint));
    float noise = 0.68 * WetValueNoise(chunk, chunkUv, 32, 3u) + 0.32 * fine;
    bool  sandy = biome == BIOME_DESERT || biome == BIOME_BEACH;
    float mask  = noise + 0.45 * cavity - (sandy ? 0.35 : 0.0);

    // Le seuil baisse avec le niveau d'eau : les flaques naissent dans les creux et s'étendent
    // (au plus ~la moitié du sol plat sous une pluie maximale).
    float threshold = mix(0.92, 0.42, waterLevel);
    float puddle    = smoothstep(threshold, threshold + 0.06, mask) * flatness;
    if (puddle <= 0.0) return s;

    // Eau stagnante : miroir horizontal, fond un peu plus sombre (absorption), mouillé à 100 %.
    s.N         = normalize(mix(s.N, vec3(0.0, 1.0, 0.0), puddle));
    s.albedo   *= mix(1.0, 0.8, puddle);
    s.wetness   = max(s.wetness, puddle);

    // Ondes de pluie : résolues de près (normale), rugosité équivalente au loin.
    float rain     = clamp(uEnv.weather.x, 0.0, 1.0);
    float resolved = 1.0 - smoothstep(0.015, 0.05, footprint);  // cellule de 25 cm
    float puddleRoughness = 0.02;
    if (rain > 0.0) {
        if (resolved > 0.0) {
            float t = envTime();
            vec2  g = RippleLayer(chunk, chunkUv, 1024, 21u, t, rain)
                    + RippleLayer(chunk, chunkUv + 0.5 / 1024.0, 1024, 37u, t + 0.45, rain * 0.8);
            float amp = 0.35 * resolved * puddle;
            s.N = normalize(s.N - vec3(g.x, 0.0, g.y) * amp);
        }
        puddleRoughness = mix(0.02, 0.10 * rain + 0.02, 1.0 - resolved);
    }
    s.roughness = mix(s.roughness, puddleRoughness, puddle);
    return s;
}

#endif  // ENV_TERRAIN_WETNESS_GLSL
