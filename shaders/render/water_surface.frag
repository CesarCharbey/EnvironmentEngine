#version 460 core
//  shaders/render/water_surface.frag                                   [DEV 2]
//  Shading de l'eau (océan, rivières, marais), en forward par-dessus scene.color (HDR), avec la même
//  physique que les surfaces opaques (surface_lighting.glsl : BRDF, ombres, ambiant, brouillard).
//    * Normale : cascades FFT (océan) ; rivières et marais : les mêmes cartes de vaguelettes advectées le
//      long de l'écoulement (flow map à deux phases, Vlachos 2010) ; impacts de pluie (terrain_wetness).
//    * Réflexion : SSR (marche écran contre gbuffer.depth, raffinement binaire) ; en repli, cône VXGI
//      (relief hors écran) puis ciel préfiltré. Soleil et lune : GGX (anti-crénelage spéculaire).
//    * Transmission : réfraction de la copie de scene.color, absorption de Beer-Lambert sur le trajet de
//      la lumière (descente jusqu'au fond + remontée vers l'œil), diffusion simple du soleil et du ciel.
//      Turbidité propre aux marais. Écume (jacobien de l'océan) éclairée en diffus.
//    * Berges : réflexion et diffusion s'effacent quand l'épaisseur d'eau tend vers 0 (pas d'arête dure).
//    * Brouillard de hauteur de la scène, puis sortie HDR (la passe de tonemap expose).
#include "common/global_ubo.glsl"
#include "render/atmo_common.glsl"
#include "render/sky_lighting.glsl"
#include "render/voxel_common.glsl"
#include "render/surface_lighting.glsl"
#include "render/ocean_common.glsl"
#include "render/terrain_wetness.glsl"   // RippleLayer : impacts de pluie

// Test de profondeur AVANT le shader : sans cette ligne, la simple présence de `discard` (rivières) fait
// renoncer le pilote au test anticipé, et toute l'eau cachée sous le relief (océan sous l'île, surfaces
// enfouies des rivières) était entièrement ombrée pour rien (×10 sur le coût des surfaces).
// Contrepartie : un fragment rejeté écrirait quand même sa profondeur -> la passe des rivières (la seule
// qui rejette) est dessinée sans écriture de profondeur (WaterOceanRiverSystem).
layout(early_fragment_tests) in;

in VS_OUT {
    vec3  worldPos;
    vec2  x0;
    vec2  flow;
    float waterSdf;
    float marsh;
    float valid;
} fsIn;

layout(location = 0) out vec4 oColor;

layout(binding = TEX_OceanDisplacement) uniform sampler2DArray uOceanMaps;
// Copie de scene.color prise juste avant l'eau (la cible courante ne peut pas être lue) : c'est bien
// « la couleur de la scène », d'où le slot SceneColor.
layout(binding = TEX_SceneColor) uniform sampler2D uSceneCopy;

uniform int uKind;  // 0 = océan, 1 = rivières / marais

const float kWaterF0 = 0.02;

// --- Pentes des cartes de l'océan ---------------------------------------------------------------
// Somme des cascades [c0, c1] : (∂h/∂x, ∂h/∂z, λ∂Dx/∂x, λ∂Dz/∂z), filtrage matériel (mips + anisotrope).
vec4 OceanSlopes(vec2 xz, int c0, int c1) {
    vec4 s = vec4(0.0);
    for (int c = c0; c <= c1; ++c) s += texture(uOceanMaps, OceanUv(xz, c, OCEAN_CASCADES));
    return s;
}

vec3 SlopesToNormal(vec4 s, float strength) {
    vec2 g = s.xy * strength / max(vec2(1.0) + s.zw * strength, vec2(0.2));
    return normalize(vec3(-g.x, 1.0, -g.y));
}

float OceanFoam(vec2 xz) {
    float f = 0.0;
    // Moutons portés par les grandes vagues : la cascade fine ne déferle pas, l'intermédiaire à peine
    // (sinon un semis uniforme de petits points blancs).
    f += texture(uOceanMaps, OceanUv(xz, 0, 0)).w;
    f += 0.3 * texture(uOceanMaps, OceanUv(xz, 1, 0)).w;
    return clamp(f, 0.0, 1.0);
}

// Variance des pentes perdue par le filtrage (mips) des cascades sous l'empreinte d'un pixel : elle
// devient de la rugosité (esprit LEAN / Toksvig). Au loin et en incidence rasante, le scintillement
// sous-échantillonné laisse place à la traînée floue du soleil, comme sur une vraie mer.
float OceanFilteredSlopeVariance(float footprint, int c0, int c1) {
    float v = 0.0;
    for (int c = c0; c <= c1; ++c) {
        float L     = uWater.cascadeSize[c];
        float texel = L / float(uOceanN);
        v += uWater.slopeVariance[c] * smoothstep(texel, 0.25 * L, footprint);
    }
    return v;
}

// --- Réflexions en espace écran --------------------------------------------------------------------
// Marche dans l'espace clip (interpolation correcte en perspective) contre gbuffer.depth, puis
// raffinement binaire. rgb = radiance réfléchie, a = confiance (0 = raté : repli sur le cône / le ciel).
// Pas FIXES (aucun bruit par pixel) : sans accumulation temporelle, un décalage aléatoire fait attraper ou
// manquer les détails fins d'un pixel à l'autre (scintillement) ; le raffinement binaire corrige la
// position des impacts trouvés. Longueur du rayon proportionnelle à la distance de l'eau.
vec4 TraceScreenReflection(vec3 P, vec3 R, float distToCam) {
    const int kSteps  = 40;
    const int kRefine = 6;
    float maxDist = clamp(2.0 * distToCam + 40.0, 60.0, 600.0);
    vec4  c0 = uEnv.viewProj * vec4(P, 1.0);
    vec4  c1 = uEnv.viewProj * vec4(P + R * maxDist, 1.0);
    if (c1.w < 0.05) c1 = mix(c0, c1, (c0.w - 0.05) / max(c0.w - c1.w, 1e-6));  // ne pas passer derrière l'œil

    float fPrev = 0.0;
    for (int i = 0; i < kSteps; ++i) {
        float f  = (float(i) + 1.0) / float(kSteps);
        f = pow(f, 1.6);  // pas resserrés près du départ (les reflets proches sont les plus visibles)
        vec4  c  = mix(c0, c1, f);
        vec2  uv = c.xy / c.w * 0.5 + 0.5;
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) break;
        float d = textureLod(uGDepth, uv, 0.0).r;
        if (!envIsSky(d)) {
            // > 0 : le rayon est passé derrière la géométrie. Tolérance d'épaisseur serrée : un rayon qui
            // passe 1 m derrière un rocher ne le touche pas (sinon semis de faux reflets clairs).
            float behind    = c.w - envLinearDepth(d);
            float thickness = max(0.5, 0.025 * c.w);
            if (behind > 0.0 && behind < thickness) {
                float lo = fPrev, hi = f;
                for (int r = 0; r < kRefine; ++r) {
                    float m  = 0.5 * (lo + hi);
                    vec4  cm = mix(c0, c1, m);
                    vec2  um = cm.xy / cm.w * 0.5 + 0.5;
                    float dm = textureLod(uGDepth, um, 0.0).r;
                    if (!envIsSky(dm) && cm.w > envLinearDepth(dm)) hi = m; else lo = m;
                }
                vec4 ch = mix(c0, c1, hi);
                uv = ch.xy / ch.w * 0.5 + 0.5;
                vec2  edge = min(uv, 1.0 - uv);
                float conf = smoothstep(0.0, 0.08, min(edge.x, edge.y)) * (1.0 - smoothstep(0.6, 1.0, hi))
                           * (1.0 - smoothstep(0.5, 1.0, behind / thickness));
                return vec4(textureLod(uSceneCopy, uv, 0.0).rgb, conf);
            }
        }
        fPrev = f;
    }
    return vec4(0.0);
}

// Ombre lointaine (cône VXGI) allégée pour l'eau : 16 pas au lieu de 40, cône un peu plus large. Sur
// l'eau, l'ombre ne module que le reflet du soleil et la diffusion : pénombre plus douce invisible.
float WaterVoxelShadow(vec3 P, vec3 L) {
    if (!uVoxelShadows || !VoxelGridUsable(P)) return 1.0;
    float voxel = VoxelSize();
    return VoxelConeVisibility(uVoxel.uGridMin.xyz, P + vec3(0.0, 0.5 * voxel, 0.0), L,
                               max(uVoxelShadowSoftness, 0.06), 1.5 * voxel, 2000.0, 16);
}

// Coefficients du milieu (1/m) : eau claire + turbidité (marais), sédiments en rivière.
void WaterMedium(out vec3 sigmaA, out vec3 sigmaS) {
    sigmaA = uWater.absorption.rgb;
    sigmaS = uWater.scattering.rgb;
    if (uKind == 1) {
        float murk = uWater.scattering.w * fsIn.marsh;
        sigmaS *= mix(2.0, 1.0, fsIn.marsh);           // rivière : un peu de matière en suspension
        sigmaA += uWater.swampColor.rgb * murk;
        sigmaS += vec3(uWater.swampColor.w) * murk;
    }
}

void main() {
    // --- Grandeurs à dérivées (avant tout branchement / rejet) ---
    vec3  P         = fsIn.worldPos;
    vec2  screenUv  = gl_FragCoord.xy * uEnv.viewportSize.zw;
    float footprint = length(fwidth(fsIn.x0));
    float footMax   = max(length(dFdx(fsIn.x0)), length(dFdy(fsIn.x0)));  // grand axe (incidence rasante)
    float t         = envTime();

    // --- Normale ---
    vec3 N;
    float foam = 0.0;
    if (uKind == 0) {
        N    = SlopesToNormal(OceanSlopes(fsIn.x0, 0, OCEAN_CASCADES - 1), 1.0);
        foam = OceanFoam(fsIn.x0);
    } else {
        // Flow map : deux phases décalées d'une demi-période, mélangées pour masquer la remise à zéro.
        const float kPeriod = 2.0;
        vec2  flow  = fsIn.flow * uWater.riverParams.x;
        float ph0   = fract(t / kPeriod), ph1 = fract(t / kPeriod + 0.5);
        float blend = abs(2.0 * ph0 - 1.0);
        vec4  s0 = OceanSlopes(fsIn.x0 - flow * ph0 * kPeriod, 1, 2);
        vec4  s1 = OceanSlopes(fsIn.x0 - flow * ph1 * kPeriod + vec2(3.7, 1.9), 1, 2);
        float speed    = length(flow);
        float strength = uWater.riverParams.y * mix(0.35, 1.0, smoothstep(0.1, 1.5, speed)) * mix(1.0, 0.4, fsIn.marsh);
        N = SlopesToNormal(mix(s0, s1, blend), strength);
    }
    // Impacts de pluie (réseau absolu du monde, effacés selon l'empreinte du pixel).
    float rain = clamp(uEnv.weather.x, 0.0, 1.0);
    if (rain > 0.0) {
        float resolved = 1.0 - smoothstep(0.015, 0.05, footprint);
        if (resolved > 0.0) {
            vec2  rel   = P.xz / CHUNK_SIZE_METERS;
            ivec2 chunk = ivec2(floor(rel)) + uEnv.originChunk.xy;
            vec2  cuv   = fract(rel);
            vec2  g = RippleLayer(chunk, cuv, 1024, 21u, t, rain) + RippleLayer(chunk, cuv + 0.5 / 1024.0, 1024, 37u, t + 0.45, rain * 0.8);
            N = normalize(N - vec3(g.x, 0.0, g.y) * 0.5 * resolved);
        }
    }
    // Anti-crénelage spéculaire (Tokuyoshi & Kaplanyan 2019) : variation de la normale entre pixels.
    vec3  dndx    = dFdx(N), dndy = dFdy(N);
    float kernel2 = min(2.0 * kSpecAaSigma2 * (dot(dndx, dndx) + dot(dndy, dndy)), kSpecAaKappa);

    // --- Rejet hors de l'eau (rivières / marais) ---
    if (uKind == 1 && (fsIn.valid < 0.5 || fsIn.waterSdf > uWater.riverParams.z)) discard;

    vec3  cam    = envCameraPosition();
    vec3  toCam  = cam - P;
    float dist   = length(toCam);
    vec3  V      = toCam / dist;
    vec3  rayDir = -V;
    if (dot(N, V) < 0.02) N = normalize(N + V * (0.02 - dot(N, V)));  // vagues vues de dos (rasant)
    float NoV = max(dot(N, V), 1e-4);

    // Rugosité : miroir, élargi par la pluie non résolue, la variance des pentes filtrée (α² ≈ 2σ²) et
    // l'anti-crénelage.
    float roughness = mix(0.02, 0.10 * rain + 0.02, smoothstep(0.015, 0.05, footprint));
    float lostVar   = uKind == 0 ? OceanFilteredSlopeVariance(footMax, 0, OCEAN_CASCADES - 1)
                                 : OceanFilteredSlopeVariance(footMax, 1, 2) * uWater.riverParams.y * uWater.riverParams.y;
    float alpha     = sqrt(clamp(roughness * roughness * roughness * roughness + 2.0 * lostVar + kernel2, 0.0, 1.0));
    roughness       = sqrt(alpha);

    Surface s;
    s.albedo = vec3(0.0); s.diffuseColor = vec3(0.0); s.f0 = vec3(kWaterF0);
    s.roughness = roughness; s.alpha = alpha; s.metal = 0.0; s.ao = 1.0; s.N = N;

    // --- Astres : éclairements et ombres (mêmes fonctions que le sol) ---
    vec3  sunDir  = normalize(envSunDirection());
    vec3  moonDir = normalize(envMoonDirection());
    float noise   = InterleavedGradientNoise(gl_FragCoord.xy);
    vec3  up      = vec3(0.0, 1.0, 0.0);
    float sunShadow  = 0.0, moonShadow = 0.0;
    // Cône lointain seulement si le relief proche n'occulte pas déjà l'astre.
    if (sunDir.y > -0.01) {
        sunShadow = HeightfieldShadow(P + up * 0.15, sunDir, noise);
        if (sunShadow > 0.0) sunShadow = min(sunShadow, WaterVoxelShadow(P, sunDir));
    }
    if (moonDir.y > -0.01 && sunDir.y < 0.05) {
        moonShadow = HeightfieldShadow(P + up * 0.15, moonDir, noise);
        if (moonShadow > 0.0) moonShadow = min(moonShadow, WaterVoxelShadow(P, moonDir));
    }
    vec3 sunE  = SunIlluminanceTOA()  * LightTransmittance(P.y, sunDir, SunAngularRadius())
               * CloudShadow(P, sunDir) * uSunScale;
    vec3 moonE = MoonIlluminanceTOA() * LightTransmittance(P.y, moonDir, MoonAngularRadius())
               * CloudShadow(P, moonDir) * uMoonScale;
    vec3 specDirect = DirectLight(s, V, sunDir, sunE * sunShadow) + DirectLight(s, V, moonDir, moonE * moonShadow);

    // --- Réflexion de l'environnement ---
    vec3 R = reflect(rayDir, N);
    R.y = abs(R.y) + 1e-3;  // vaguelettes : jamais de réflexion vers le bas
    R = normalize(R);
    vec3 env;
    if (uIblAvailable) {
        float maxLod = float(textureQueryLevels(uIblPrefiltered) - 1);
        env = textureLod(uIblPrefiltered, R, roughness * maxLod).rgb;
    } else {
        env = SkyLuminanceAbove(R);
    }
    // 1) SSR jusqu'à 400 m, pour les rayons proches de l'horizon : au-delà de la distance, ou vers le haut
    //    du ciel, ils ne traversent presque que du ciel à l'écran (coût sans gain).
    float ssrConf = 0.0;
    vec3  ssrRgb  = vec3(0.0);
    if (uWater.riverParams.w > 0.5 && uGBufferAvailable && dist < 400.0) {
        // Rayon SSR selon une normale adoucie : des rayons cohérents d'un pixel à l'autre donnent un reflet
        // stable (miroir ondulé) au lieu d'impacts épars ; la rugosité fine reste dans le spéculaire.
        vec3 Rs = reflect(rayDir, normalize(mix(N, up, 0.6)));
        Rs.y = abs(Rs.y) + 1e-3;
        Rs = normalize(Rs);
        if (Rs.y < 0.6) {
            vec4 ssr = TraceScreenReflection(P, Rs, dist);
            ssrRgb  = ssr.rgb;
            ssrConf = ssr.a * (1.0 - smoothstep(300.0, 400.0, dist)) * (1.0 - smoothstep(0.45, 0.6, Rs.y));
        }
    }
    // 2) Relief hors écran (collines, rives lointaines) : cône VXGI, inutile si le SSR a trouvé le reflet.
    if (ssrConf < 0.98 && VoxelGridUsable(P)) {
        float voxel = VoxelSize();
        vec4  cone  = VoxelConeRadiance(uVoxel.uGridMin.xyz, P, R, 0.08, voxel, 1500.0, 14);
        env = cone.rgb * uGiIntensity + (1.0 - cone.a) * env;
    }
    env = mix(env, ssrRgb, ssrConf);
    env *= uIblScale;
    vec2 brdf     = uIblAvailable ? texture(uIblBrdfLut, vec2(NoV, roughness)).rg : EnvBrdfApprox(NoV, roughness);
    vec3 specEnvW = vec3(kWaterF0) * brdf.x + brdf.y;  // part réfléchie (split-sum)

    // --- Transmission : fond réfracté + milieu ---
    float gd     = uGBufferAvailable ? textureLod(uGDepth, screenUv, 0.0).r : 0.0;
    vec3  bottomP = envIsSky(gd) ? cam + rayDir * 1.0e5 : envWorldPosFromDepth(screenUv, gd);
    float thick  = max(length(bottomP - cam) - dist, 0.0);  // eau traversée le long du regard (m)

    vec2  offset = N.xz * uWater.absorption.w * 0.06 * clamp(thick * 0.5, 0.0, 1.0) / max(1.0, dist * 0.02);
    vec2  uvR    = screenUv + offset;
    float gdR    = uGBufferAvailable ? textureLod(uGDepth, uvR, 0.0).r : 0.0;
    vec3  bottomR = envIsSky(gdR) ? cam + rayDir * 1.0e5 : envWorldPosFromDepth(uvR, gdR);
    if (length(bottomR - cam) < dist) { uvR = screenUv; bottomR = bottomP; }  // objet devant l'eau : pas de réfraction
    float thickR = max(length(bottomR - cam) - dist, 0.0);
    vec3  bottom = textureLod(uSceneCopy, uvR, 0.0).rgb;

    vec3 sigmaA, sigmaS;
    WaterMedium(sigmaA, sigmaS);
    vec3  sigmaT = sigmaA + sigmaS;
    float depthV = max(P.y - bottomR.y, 0.0);                   // profondeur verticale du fond
    vec3  Tview  = exp(-sigmaT * min(thickR, 1.0e3));
    vec3  Tdown  = exp(-sigmaT * min(depthV, 1.0e3) / max(sunDir.y, 0.3));
    // Diffusion simple : soleil (transmis par la surface) + ciel, diffusés vers l'œil (phase ~ isotrope,
    // la lumière s'atténue aussi en descendant : facteur 1/2 pour un milieu épais).
    vec3 skyIrr  = (uIblAvailable ? texture(uIblIrradiance, up).rgb * PI : PI * SkyAverage()) * uIblScale;
    vec3 inLight = (sunE * sunShadow * max(sunDir.y, 0.0) + moonE * moonShadow * max(moonDir.y, 0.0)) * (1.0 - kWaterF0)
                 + skyIrr;
    vec3 scatter = sigmaS / max(sigmaT, vec3(1e-4)) * inLight / (4.0 * PI) * 0.5 * (1.0 - Tview);
    vec3 under   = bottom * Tview * Tdown + scatter;

    // Berges : l'eau s'efface quand son épaisseur tend vers 0.
    float shore = smoothstep(0.0, max(0.25 * uWater.riverParams.z, 0.05), thickR);
    vec3  color = under * (1.0 - specEnvW * shore) + (env * specEnvW + specDirect) * shore;

    // Écume : diffuseur blanc (océan).
    if (foam > 0.0) {
        vec3 foamL = 0.85 * (sunE * sunShadow * max(dot(N, sunDir), 0.0) + moonE * moonShadow * max(dot(N, moonDir), 0.0)
                           + skyIrr) / PI;
        color = mix(color, foamL, foam * 0.9);
    }

    color = ApplyHeightFog(color, P, rayDir, dist, sunShadow, moonShadow);
    oColor = vec4(min(color, vec3(kHalfMax)), 1.0);
}
