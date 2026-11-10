//  shaders/render/atmo_common.glsl                                    [DEV 2]
//  Fonctions physiques partagées (Hillaire 2020, "A Scalable and Production Ready
//  Sky and Atmosphere Rendering Technique").
//
//  Unités : distances en km (rayons, hauteurs d'échelle, coefficients en 1/km).
//  Repère local planète : centre en (0, 0, 0), la caméra est en (0, r, 0), +Y = zénith.
//  Les directions monde (Y vers le haut) sont utilisées telles quelles (approximation
//  "monde plat" locale, valable à l'échelle d'une zone jouable).
//
//  Luminances stockées en unités physiques (cd/m², éclairements en lux) : le jour
//  (~1e4) comme la nuit (~1e-3) restent dans la plage normale du RGBA16F.
//  Le tonemapping / l'exposition (VolumetricsPost) ramènera ces valeurs à l'écran.
#ifndef ENV_ATMO_COMMON_GLSL
#define ENV_ATMO_COMMON_GLSL

#include "common/global_ubo.glsl"

layout(std140, binding = UBO_AtmosphereParams) uniform AtmosphereParamsBlock {
    vec4 uRadiiAndScaleHeights; // x = Rb, y = Rt, z = HR, w = HM (en km)
    vec4 uRayleighScattering;   // xyz = beta_R, w = mieG
    vec4 uMieScatteringAndExt;  // xyz = beta_M_sca, w = beta_M_ext
    vec4 uOzoneAbsorption;      // xyz = beta_O3, w = ozoneEnabled
    vec4 uGroundAlbedoAndTurb;  // xyz = groundAlbedo, w = turbidity
    vec4 uNightSky;             // x = intensité du fond de ciel nocturne (voir NightSkyColor), yzw = 0
} uAtmo;

const float PI = 3.14159265358979323846;

// --- Direction artistique de la nuit ("nuit cinématique jouable") ---------------------------------
// Clair de lune bleu-argenté froid, normalisé en luminance (0.7255) : l'éclairement en lux est conservé,
// seule la teinte change (au lieu de la couleur chaude du soleil réfléchi).
const vec3 MOONLIGHT_TINT = vec3(0.55, 0.75, 1.0) / 0.7255;

// Fond de ciel nocturne (étoiles diffuses + airglow) : bleu nuit profond au zénith, bleu acier doux à
// l'horizon. intensity = uAtmo.uNightSky.x (luminance zénithale ≈ 0.176 × intensity cd/m²).
vec3 NightSkyColor(vec3 dir, float intensity) {
    const vec3 kZenith  = vec3(0.08, 0.18, 0.42);
    const vec3 kHorizon = vec3(0.15, 0.28, 0.50);
    return mix(kHorizon, kZenith, sqrt(clamp(dir.y, 0.0, 1.0))) * intensity;
}

// Décalage (km) qui évite l'auto-intersection avec le sol (Hillaire : PLANET_RADIUS_OFFSET).
const float PLANET_RADIUS_OFFSET = 0.01;

// Résolutions des LUTs : DOIVENT correspondre aux allocations de CelestialAtmosphereSystem.cpp.
const vec2 TRANSMITTANCE_LUT_RES = vec2(256.0, 64.0);
const vec2 MULTISCAT_LUT_RES     = vec2(32.0, 32.0);
const vec2 SKYVIEW_LUT_RES       = vec2(192.0, 108.0);

// -----------------------------------------------------------------------------
// Géométrie
// -----------------------------------------------------------------------------

// Intersection rayon (origine p, direction d unitaire) avec une sphère centrée en (0,0,0) de rayon R
// Retourne la distance positive la plus proche, ou -1.0 si aucune intersection devant le rayon
float RaySphereIntersectNearest(vec3 p, vec3 d, float R) {
    float b = dot(p, d);
    float c = dot(p, p) - R * R;
    float disc = b * b - c;
    if (disc < 0.0) return -1.0;
    float s = sqrt(disc);
    float t0 = -b - s;
    float t1 = -b + s;
    if (t0 >= 0.0) return t0;
    return (t1 >= 0.0) ? t1 : -1.0;
}

// Fraction visible d'un astre de rayon angulaire angR au-dessus de l'horizon géométrique de la
// planète, vu depuis l'altitude hKm (mu = cos de l'angle zénithal de l'astre). 0 = couché, 1 = levé.
// La transmittance LUT ne couvre que les directions au-dessus de l'horizon : ce test est
// indispensable pour qu'un astre couché n'éclaire plus ni l'atmosphère ni le sol.
//  * Test analytique (mu comparé au cosinus de l'horizon) plutôt qu'une intersection rayon/sphère :
//    sqrt(h (2 Rb + h)) évite l'annulation catastrophique de r² - Rb² en float près du sol
//    (r ≈ 6360 km n'a qu'une précision de ~0,8 m).
//  * Transition douce sur le diamètre apparent de l'astre : le disque se couche progressivement.
float PlanetShadowAlt(float hKm, float mu, float Rb, float angR) {
    float h = max(hKm, 0.0);
    float horizonMu = -sqrt(h * (2.0 * Rb + h)) / (Rb + h);
    float w = max(sin(angR), 1e-4);
    return smoothstep(horizonMu - w, horizonMu + w, mu);
}

// Variante pour un point du raymarching (repère local planète, km).
float PlanetShadow(vec3 pos, vec3 lightDir, float Rb) {
    float r = length(pos);
    return PlanetShadowAlt(r - Rb, dot(pos / r, lightDir), Rb, 0.0);
}

// Rayon (km, depuis le centre de la planète) du point de vue courant.
// L'origine de rendu flottante ne décale que X/Z : cameraPosition.y est une altitude absolue (m).
float AtmoViewRadiusKm() {
    float Rb = uAtmo.uRadiiAndScaleHeights.x;
    float Rt = uAtmo.uRadiiAndScaleHeights.y;
    float hKm = max(uEnv.cameraPosition.y * 0.001, PLANET_RADIUS_OFFSET);
    return Rb + min(hKm, Rt - Rb - PLANET_RADIUS_OFFSET);
}

// -----------------------------------------------------------------------------
// Milieu participant
// -----------------------------------------------------------------------------

// Évalue la diffusion (Rayleigh, Mie) et l'extinction totale à une altitude h (km)
void SampleMediumKm(float hKm, out vec3 sigScaR, out vec3 sigScaM, out vec3 sigExt) {
    float h = max(0.0, hKm);
    float densityR = exp(-h / uAtmo.uRadiiAndScaleHeights.z);
    float densityM = exp(-h / uAtmo.uRadiiAndScaleHeights.w);
    // Couche d'ozone centrée vers 25 km, largeur 15 km
    float densityO = max(0.0, 1.0 - abs(h - 25.0) / 15.0) * uAtmo.uOzoneAbsorption.w;

    sigScaR = uAtmo.uRayleighScattering.xyz * densityR;
    sigScaM = uAtmo.uMieScatteringAndExt.xyz * densityM;
    vec3 extM = vec3(uAtmo.uMieScatteringAndExt.w) * densityM;
    vec3 extO = uAtmo.uOzoneAbsorption.xyz * densityO;

    sigExt = sigScaR + extM + extO;
}

float PhaseRayleigh(float cosTheta) {
    return (3.0 / (16.0 * PI)) * (1.0 + cosTheta * cosTheta);
}

float PhaseCornetteShanksMie(float cosTheta, float g) {
    float g2 = g * g;
    float num = 3.0 * (1.0 - g2) * (1.0 + cosTheta * cosTheta);
    float den = (8.0 * PI) * (2.0 + g2) * pow(max(1e-4, 1.0 + g2 - 2.0 * g * cosTheta), 1.5);
    return num / den;
}

// -----------------------------------------------------------------------------
// Sub-UV : fait correspondre [0, 1] aux CENTRES du premier et du dernier texel,
// pour que les valeurs extrêmes (zénith, nadir, sol, sommet) soient exactes.
// -----------------------------------------------------------------------------
float UnitToSubUV(float u, float res) { return (u * (res - 1.0) + 0.5) / res; }
float SubUVToUnit(float u, float res) { return (u * res - 0.5) / (res - 1.0); }

// -----------------------------------------------------------------------------
// Transmittance LUT (Bruneton 2017 / Hillaire 2020) : uv <-> (r, mu)
// -----------------------------------------------------------------------------
// Version par altitude hKm, numériquement stable près du sol (utilisée pour l'éclairage des surfaces).
vec2 TransmittanceParamsToUVAlt(float hKm, float mu, float Rb, float Rt) {
    float h    = clamp(hKm, 0.0, Rt - Rb);
    float r    = Rb + h;
    float H2   = (Rt - Rb) * (Rt + Rb);         // Rt² - Rb²
    float rho2 = h * (2.0 * Rb + h);            // r² - Rb²
    float H    = sqrt(H2);
    float rho  = sqrt(rho2);
    float dMin = (Rt - Rb) - h;
    float dMax = rho + H;
    // Distance au sommet de l'atmosphère : d = -r mu + sqrt(r² mu² + Rt² - r²)
    float X = H2 - rho2;                        // Rt² - r² >= 0
    float s = sqrt(max(0.0, r * r * mu * mu + X));
    float d = mu >= 0.0 ? X / max(s + r * mu, 1e-6)   // forme sans annulation
                        : s - r * mu;
    float u = (d - dMin) / max(1e-5, dMax - dMin);
    float v = rho / max(1e-5, H);
    return vec2(clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0));
}

vec2 TransmittanceParamsToUV(float r, float mu, float Rb, float Rt) {
    return TransmittanceParamsToUVAlt(r - Rb, mu, Rb, Rt);
}

void UVToTransmittanceParams(vec2 uv, float Rb, float Rt, out float r, out float mu) {
    float H = sqrt(max(0.0, Rt * Rt - Rb * Rb));
    float rho = H * uv.y;
    r = sqrt(rho * rho + Rb * Rb);
    float dMin = Rt - r;
    float dMax = rho + H;
    float d = dMin + uv.x * (dMax - dMin);
    mu = (d == 0.0) ? 1.0 : ((H * H - rho * rho - d * d) / (2.0 * r * d));
    mu = clamp(mu, -1.0, 1.0);
}

// -----------------------------------------------------------------------------
// Multi-scattering LUT : u = cos(zénith soleil) * 0.5 + 0.5, v = altitude normalisée
// -----------------------------------------------------------------------------
vec2 MultiScatteringParamsToUV(float r, float muS, float Rb, float Rt) {
    vec2 uv = clamp(vec2(muS * 0.5 + 0.5, (r - Rb) / (Rt - Rb)), 0.0, 1.0);
    return vec2(UnitToSubUV(uv.x, MULTISCAT_LUT_RES.x), UnitToSubUV(uv.y, MULTISCAT_LUT_RES.y));
}

// -----------------------------------------------------------------------------
// Sky-View LUT (Hillaire 2020, section 5.3)
//   v ∈ [0, 0.5] : zénith -> horizon ; v ∈ [0.5, 1] : horizon -> nadir.
//   Compression quadratique autour de l'horizon (là où le ciel varie le plus).
//   L'horizon dépend de l'altitude r : il descend sous l'horizontale de beta quand on monte.
//   u = azimut absolu / 2π (0 = -Z nord, 0.25 = +X est), échantillonné en GL_REPEAT.
//   (Hillaire paramètre l'azimut relativement au soleil ; ici on garde l'azimut absolu
//    parce que la LUT contient aussi la lune.)
// -----------------------------------------------------------------------------
void SkyViewHorizon(float r, float Rb, out float beta, out float zenithHorizonAngle) {
    float vHorizon = sqrt(max(0.0, r * r - Rb * Rb));
    beta = acos(clamp(vHorizon / r, -1.0, 1.0));  // angle de l'horizon sous l'horizontale
    zenithHorizonAngle = PI - beta;
}

// uv (centre de texel) -> direction de vue monde
vec3 UVToSkyViewDir(vec2 uv, float r, float Rb) {
    float beta, zenithHorizonAngle;
    SkyViewHorizon(r, Rb, beta, zenithHorizonAngle);

    float v = SubUVToUnit(uv.y, SKYVIEW_LUT_RES.y);
    float zenithAngle;
    if (v < 0.5) {
        float c = 1.0 - 2.0 * v;
        zenithAngle = zenithHorizonAngle * (1.0 - c * c);
    } else {
        float c = 2.0 * v - 1.0;
        zenithAngle = zenithHorizonAngle + beta * c * c;
    }

    float azimuth = uv.x * 2.0 * PI;
    float s = sin(zenithAngle);
    return vec3(s * sin(azimuth), cos(zenithAngle), -s * cos(azimuth));
}

// direction de vue monde (unitaire) -> uv pour échantillonner atmo.skyViewLUT.
// À utiliser par toute passe qui lit le ciel (DeferredLighting, IBL, eau...) avec r = AtmoViewRadiusKm().
vec2 SkyViewDirToUV(vec3 viewDir, float r, float Rb) {
    float beta, zenithHorizonAngle;
    SkyViewHorizon(r, Rb, beta, zenithHorizonAngle);

    float zenithAngle = acos(clamp(viewDir.y, -1.0, 1.0));
    float v;
    if (zenithAngle < zenithHorizonAngle) {
        float c = zenithAngle / zenithHorizonAngle;
        v = 0.5 * (1.0 - sqrt(max(0.0, 1.0 - c)));
    } else {
        float c = (zenithAngle - zenithHorizonAngle) / max(beta, 1e-5);
        v = 0.5 + 0.5 * sqrt(clamp(c, 0.0, 1.0));
    }

    float azimuth = atan(viewDir.x, -viewDir.z);
    if (azimuth < 0.0) azimuth += 2.0 * PI;
    return vec2(azimuth / (2.0 * PI), UnitToSubUV(v, SKYVIEW_LUT_RES.y));
}

#endif  // ENV_ATMO_COMMON_GLSL
