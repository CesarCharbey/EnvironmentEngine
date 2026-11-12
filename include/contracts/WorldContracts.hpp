#pragma once
// =============================================================================
//  include/contracts/WorldContracts.hpp
//  CONTRAT FIGÉ — Données échangées entre World Engine (Dev 1) et Render Engine (Dev 2).
//  Toute modification = nouvel ADR dans DECISIONS.md + accord des deux devs.
//
//  Contenu :
//    1. Constantes du monde (chunks, origine flottante, végétation, physique)
//    2. Biomes et archétypes de végétation
//    3. GlobalEnvironmentUBO (std140, miroir EXACT de shaders/common/global_ubo.glsl)
//    4. Structures GPU POD (FoliageInstance, DrawElementsIndirectCommand)
//    5. Composants ECS EnTT
//    6. Helpers de coordonnées (monde infini, origine de rendu flottante)
//    7. Table des clés du RenderBlackboard (+ fallbacks)
//
//  Règles : uniquement des POD / constexpr / fonctions mathématiques pures.
//  Aucune dépendance vers OpenGL, src/core ou un dossier métier.
// =============================================================================
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace env {

// =============================================================================
// 1. Constantes du monde
// =============================================================================
/// Taille d'un chunk en mètres (axes X et Z).
inline constexpr float kChunkSizeMeters = 256.0f;
/// Résolution (texels par côté) de heightNormalTex et biomeHydroTex.
/// Le centre du texel (i, j) est en ((i + 0.5) / N, (j + 0.5) / N) * kChunkSizeMeters (local X, Z).
inline constexpr std::uint32_t kChunkTexResolution = 256;
/// L'origine de rendu se recale sur une grille de kOriginRebaseChunks chunks (8 x 256 m = 2048 m).
inline constexpr std::int32_t kOriginRebaseChunks = 8;
/// Nombre d'archétypes de végétation = nombre de DrawElementsIndirectCommand par chunk.
inline constexpr std::uint32_t kFoliageArchetypeCount = 4;
/// Capacité maximale d'instances de végétation par chunk.
inline constexpr std::uint32_t kFoliageMaxInstancesPerChunk = 65536;
/// Rayon de promotion physique autour de la caméra / des entités actives (ADR-006).
inline constexpr float kPhysicsPromotionRadius = 20.0f;
/// Valeur des SDF de biomeHydroTex signifiant « aucune rivière / aucun marais à proximité ».
inline constexpr float kRiverSdfFar = 1000.0f;
/// Identifiant Jolt invalide (== JPH::BodyID::cInvalidBodyID).
inline constexpr std::uint32_t kInvalidBodyId = 0xFFFFFFFFu;

// =============================================================================
// 2. Biomes et archétypes
// =============================================================================
enum class BiomeType : std::uint8_t {
    Ocean        = 0,
    Beach        = 1,
    Desert       = 2,
    Plains       = 3,
    Forest       = 4,
    Jungle       = 5,
    Swamp        = 6,
    HighMountain = 7,
    Count
};
inline constexpr std::size_t kBiomeCount = static_cast<std::size_t>(BiomeType::Count);
inline constexpr std::array<std::string_view, kBiomeCount> kBiomeNames = {
    "Ocean", "Beach", "Desert", "Plains", "Forest", "Jungle", "Swamp", "HighMountain"};

constexpr std::string_view BiomeName(BiomeType biome) noexcept {
    const auto i = static_cast<std::size_t>(biome);
    return i < kBiomeCount ? kBiomeNames[i] : std::string_view{"Invalid"};
}

enum class FoliageArchetype : std::uint32_t { Grass = 0, Shrub = 1, Tree = 2, Rock = 3, Count };
inline constexpr std::array<std::string_view, kFoliageArchetypeCount> kFoliageArchetypeNames = {
    "Grass", "Shrub", "Tree", "Rock"};
static_assert(static_cast<std::uint32_t>(FoliageArchetype::Count) == kFoliageArchetypeCount);

// =============================================================================
// 3. GlobalEnvironmentUBO — std140, binding env::bind::ubo::GlobalEnvironment
//    MIROIR EXACT de shaders/common/global_ubo.glsl (bloc GlobalEnvironmentBlock).
//    Uniquement des vec4 / mat4 / ivec4 / uvec4 : layout C++ == layout std140.
//    NE JAMAIS réordonner ni insérer au milieu : ajouter en fin + ADR.
//    Écrivains (un champ = un écrivain, AGENTS.md section 8) :
//      [Core]      caméra, viewport, originChunk, waterTime.zw, frameInfo
//      [Celestial] soleil, lune, vent, météo           (Dev 2)
//      [BiomeHydro] waterTime.xy (niveaux d'eau)       (Dev 1)
// =============================================================================
struct alignas(16) GlobalEnvironmentUBO {
    // --- Caméra [Core] — positions relatives à l'origine de rendu (ADR-007) ---
    glm::mat4 view{1.0f};
    glm::mat4 proj{1.0f};          ///< Reverse-Z, plan lointain infini, clip [0, 1]
    glm::mat4 viewProj{1.0f};
    glm::mat4 invView{1.0f};
    glm::mat4 invProj{1.0f};
    glm::mat4 invViewProj{1.0f};
    glm::mat4 prevViewProj{1.0f};  ///< viewProj de la frame précédente (TAA / vecteurs de mouvement)
    glm::vec4 cameraPosition{0.0f};  ///< xyz = position caméra (m, relative à l'origine), w = 1
    glm::vec4 cameraParams{0.0f};    ///< x = near, y = distance de rendu, z = tan(fovY/2), w = aspect
    glm::vec4 viewportSize{0.0f};    ///< xy = taille (px), zw = 1 / taille
    glm::ivec4 originChunk{0};       ///< xy = chunk (X, Z) de l'origine de rendu, z = génération, w = 0

    // --- Soleil [CelestialAtmosphereSystem] ---
    glm::vec4 sunDirIlluminance{0.0f};  ///< xyz = direction VERS le soleil (unitaire), w = éclairement (lux)
    glm::vec4 sunColorAngular{0.0f};    ///< rgb = couleur linéaire, a = rayon angulaire (rad)

    // --- Lune [CelestialAtmosphereSystem] ---
    glm::vec4 moonDirPhase{0.0f};  ///< xyz = direction VERS la lune, w = phase [0,1) (0 nouvelle, 0.5 pleine)
    glm::vec4 moonParams{0.0f};    ///< x = éclairement (lux), y = rayon angulaire (rad), z = lumière cendrée, w = 0

    // --- Vent [CelestialAtmosphereSystem] ---
    glm::vec4 windDirSpeed{0.0f};  ///< xy = direction horizontale (X, Z) unitaire, z = vitesse (m/s), w = turbulence [0,1]
    glm::vec4 windGust{0.0f};      ///< x = fréquence des rafales (Hz), y = force des rafales, zw = réservé

    // --- Météo [CelestialAtmosphereSystem] ---
    glm::vec4 weather{0.0f};       ///< x = pluie, y = neige, z = tempête de sable, w = humidité (tous [0,1])
    glm::vec4 weatherExtra{0.0f};  ///< x = couverture nuageuse, y = densité nuageuse, z = densité de brouillard (1/m), w = humidité de surface

    // --- Eau & temps ---
    glm::vec4 waterTime{0.0f};     ///< x = niveau océan (m) [BiomeHydro], y = niveau marais (m) [BiomeHydro], z = temps (s) [Core], w = dt (s) [Core]
    glm::uvec4 frameInfo{0u};      ///< x = index de frame, y = vue de debug globale, z = flags (bit0 origine recalée, bit1 mock actif), w = réservé
};

static_assert(sizeof(GlobalEnvironmentUBO) == 672, "GlobalEnvironmentUBO: taille std140 modifiee (voir global_ubo.glsl)");
static_assert(sizeof(GlobalEnvironmentUBO) % 16 == 0, "GlobalEnvironmentUBO: taille non multiple de 16");
static_assert(offsetof(GlobalEnvironmentUBO, view) == 0);
static_assert(offsetof(GlobalEnvironmentUBO, proj) == 64);
static_assert(offsetof(GlobalEnvironmentUBO, viewProj) == 128);
static_assert(offsetof(GlobalEnvironmentUBO, invView) == 192);
static_assert(offsetof(GlobalEnvironmentUBO, invProj) == 256);
static_assert(offsetof(GlobalEnvironmentUBO, invViewProj) == 320);
static_assert(offsetof(GlobalEnvironmentUBO, prevViewProj) == 384);
static_assert(offsetof(GlobalEnvironmentUBO, cameraPosition) == 448);
static_assert(offsetof(GlobalEnvironmentUBO, cameraParams) == 464);
static_assert(offsetof(GlobalEnvironmentUBO, viewportSize) == 480);
static_assert(offsetof(GlobalEnvironmentUBO, originChunk) == 496);
static_assert(offsetof(GlobalEnvironmentUBO, sunDirIlluminance) == 512);
static_assert(offsetof(GlobalEnvironmentUBO, sunColorAngular) == 528);
static_assert(offsetof(GlobalEnvironmentUBO, moonDirPhase) == 544);
static_assert(offsetof(GlobalEnvironmentUBO, moonParams) == 560);
static_assert(offsetof(GlobalEnvironmentUBO, windDirSpeed) == 576);
static_assert(offsetof(GlobalEnvironmentUBO, windGust) == 592);
static_assert(offsetof(GlobalEnvironmentUBO, weather) == 608);
static_assert(offsetof(GlobalEnvironmentUBO, weatherExtra) == 624);
static_assert(offsetof(GlobalEnvironmentUBO, waterTime) == 640);
static_assert(offsetof(GlobalEnvironmentUBO, frameInfo) == 656);

// =============================================================================
// 4. Structures GPU POD (std430) — miroirs dans shaders/common/global_ubo.glsl
// =============================================================================
/// Commande de glMultiDrawElementsIndirect. Ordre et taille imposés par OpenGL.
struct DrawElementsIndirectCommand {
    std::uint32_t count         = 0;  ///< nombre d'indices du mesh de l'archétype
    std::uint32_t instanceCount = 0;  ///< rempli par le scatter (compteur atomique)
    std::uint32_t firstIndex    = 0;
    std::int32_t  baseVertex    = 0;
    std::uint32_t baseInstance  = 0;  ///< offset de l'archétype dans le SSBO d'instances
};
static_assert(sizeof(DrawElementsIndirectCommand) == 20, "DrawElementsIndirectCommand: 5 x 32 bits imposes par GL");

/// Instance de végétation (ADR-001). Lue par instances[gl_BaseInstance + gl_InstanceID].
struct alignas(16) FoliageInstance {
    glm::vec4 positionScale{0.0f};  ///< xyz = position LOCALE au chunk (m, y = hauteur absolue), w = échelle uniforme
    glm::vec4 params{0.0f};         ///< x = yaw (rad), y = archétype (float entier), z = variante, w = aléa [0,1)
};
static_assert(sizeof(FoliageInstance) == 32, "FoliageInstance: 2 x vec4 (std430)");

// =============================================================================
// 5. Composants ECS EnTT
// =============================================================================
/// Coordonnées entières du chunk sur la grille infinie (X, Z) + niveau de détail.
struct ChunkCoordComponent {
    glm::ivec2   coord{0, 0};
    std::uint8_t lod = 0;
};

enum class ChunkState : std::uint8_t {
    Requested,   ///< créé par ChunkStreamingSystem, rien d'alloué
    Generating,  ///< ressources allouées, génération GPU en cours (peut durer plusieurs frames)
    GpuReady,    ///< au moins HeightNormal prêt : dessinable
    Evicting     ///< sortie de la zone : ne plus dessiner, libérer à la frame suivante
};

/// Bits de ChunkStateComponent::readyMask (miroir GLSL : CHUNK_READY_*).
namespace ChunkReady {
inline constexpr std::uint32_t HeightNormal    = 1u << 0;  ///< heightNormalTex valide
inline constexpr std::uint32_t BiomeHydro      = 1u << 1;  ///< biomeHydroTex valide
inline constexpr std::uint32_t Foliage         = 1u << 2;  ///< SSBO d'instances + buffer indirect valides
inline constexpr std::uint32_t PhysicsCollider = 1u << 3;  ///< collider Jolt du terrain construit
inline constexpr std::uint32_t All             = HeightNormal | BiomeHydro | Foliage;
}  // namespace ChunkReady

struct ChunkStateComponent {
    ChunkState    state      = ChunkState::Requested;
    std::uint32_t readyMask  = 0;    ///< combinaison de ChunkReady::*
    std::uint32_t generation = 0;    ///< incrémenté à chaque régénération (invalidation de caches)
    double        lastTouchedTime = 0.0;  ///< dernière fois dans la zone désirée (éviction LRU)
};

/// Vue NON-PROPRIÉTAIRE des ressources GPU d'un chunk (identifiants OpenGL bruts).
/// Le producteur (Dev 1, ou MockDataProvider) possède les objets RAII et retire ce
/// composant AVANT de les détruire. Un id à 0 = ressource absente (voir readyMask).
struct ChunkGpuResources {
    std::uint32_t heightNormalTex          = 0;  ///< GL_TEXTURE_2D RGBA32F : R = hauteur (m), GBA = normale monde
    std::uint32_t biomeHydroTex            = 0;  ///< GL_TEXTURE_2D RGBA16F : R = biome, G = SDF rivière, B = SDF marais, A = humidité
    std::uint32_t foliageIndirectCmdBuffer = 0;  ///< GL buffer : DrawElementsIndirectCommand[foliageCommandCount]
    std::uint32_t foliageInstanceSSBO      = 0;  ///< GL buffer : FoliageInstance[foliageInstanceCapacity]
    std::uint32_t foliageInstanceCapacity  = 0;
    std::uint32_t foliageCommandCount      = 0;  ///< normalement kFoliageArchetypeCount
    glm::vec2     heightRange{0.0f, 0.0f};       ///< min / max de hauteur (AABB de culling)
};

/// Tag : entité chunk générée par le MockDataProvider (core), à exclure des vues de streaming.
struct MockChunkTag {};

/// Transformation monde (double précision : monde infini). Convertir en relatif via WorldToRenderRelative.
struct TransformComponent {
    glm::dvec3 worldPosition{0.0};
    glm::quat  rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3  scale{1.0f};
};

/// Tag : l'entité peut recevoir un body Jolt lorsqu'elle entre dans le rayon de promotion (ADR-006).
struct PhysicsPromotableTag {};

/// Body Jolt d'une entité promue. objectLayer = env::core::PhysicsLayers::*.
struct PhysicsBodyComponent {
    std::uint32_t bodyId      = kInvalidBodyId;
    std::uint16_t objectLayer = 0;
};

// =============================================================================
// 6. Helpers de coordonnées (ADR-007)
// =============================================================================
namespace detail {
constexpr std::int32_t FloorDiv(std::int32_t value, std::int32_t divisor) noexcept {
    const std::int32_t q = value / divisor;
    return (value % divisor != 0 && ((value < 0) != (divisor < 0))) ? q - 1 : q;
}
}  // namespace detail

/// Chunk contenant une position monde absolue.
inline glm::ivec2 WorldToChunk(const glm::dvec3& world) noexcept {
    return {static_cast<std::int32_t>(std::floor(world.x / static_cast<double>(kChunkSizeMeters))),
            static_cast<std::int32_t>(std::floor(world.z / static_cast<double>(kChunkSizeMeters)))};
}

/// Coin (X min, Z min) d'un chunk en coordonnées monde absolues.
inline glm::dvec3 ChunkOriginWorld(glm::ivec2 chunk) noexcept {
    return {static_cast<double>(chunk.x) * kChunkSizeMeters, 0.0, static_cast<double>(chunk.y) * kChunkSizeMeters};
}

/// Origine de rendu pour une caméra donnée : chunk aligné sur la grille kOriginRebaseChunks.
inline glm::ivec2 ComputeRenderOriginChunk(const glm::dvec3& cameraWorld) noexcept {
    const glm::ivec2 c = WorldToChunk(cameraWorld);
    return {detail::FloorDiv(c.x, kOriginRebaseChunks) * kOriginRebaseChunks,
            detail::FloorDiv(c.y, kOriginRebaseChunks) * kOriginRebaseChunks};
}

/// Position monde absolue de l'origine de rendu.
inline glm::dvec3 RenderOriginWorld(glm::ivec2 originChunk) noexcept { return ChunkOriginWorld(originChunk); }

/// Coin d'un chunk relatif à l'origine de rendu (exact : différence entière).
inline glm::vec3 ChunkOriginRelative(glm::ivec2 chunk, glm::ivec2 originChunk) noexcept {
    const glm::ivec2 d = chunk - originChunk;
    return {static_cast<float>(d.x) * kChunkSizeMeters, 0.0f, static_cast<float>(d.y) * kChunkSizeMeters};
}

/// Position monde absolue -> position GPU / Jolt relative à l'origine de rendu.
inline glm::vec3 WorldToRenderRelative(const glm::dvec3& world, glm::ivec2 originChunk) noexcept {
    return glm::vec3(world - RenderOriginWorld(originChunk));
}

/// Centre du texel (i, j) dans le repère local du chunk (X, Z), en mètres.
constexpr glm::vec2 ChunkTexelCenterLocal(std::uint32_t i, std::uint32_t j) noexcept {
    return {(static_cast<float>(i) + 0.5f) / static_cast<float>(kChunkTexResolution) * kChunkSizeMeters,
            (static_cast<float>(j) + 0.5f) / static_cast<float>(kChunkTexResolution) * kChunkSizeMeters};
}

// =============================================================================
// 7. Table des clés du RenderBlackboard (AGENTS.md section 6)
//    Le blackboard sert automatiquement le fallback déclaré ici tant qu'une clé
//    n'est pas publiée par son producteur.
// =============================================================================
namespace bb {

enum class Kind : std::uint8_t { Texture2D, Texture3D, Cubemap, Buffer };

enum class Fallback : std::uint8_t {
    Checker,            ///< damier magenta 64x64 RGBA8 ("texture manquante")
    Black,              ///< RGBA8 = 0
    White,              ///< RGBA8 = 1
    FlatNormalTS,       ///< RGBA8 (0.5, 0.5, 1, 1) normale tangente plate
    FlatNormalWS,       ///< RGBA16F (0, 1, 0, 0) normale monde vers le haut
    FlatHeightNormal,   ///< RGBA32F (h = 0, n = 0,1,0)
    NeutralBiomeHydro,  ///< RGBA16F (Plains, kRiverSdfFar, kRiverSdfFar, 0.5)
    Black3D,            ///< 3D RGBA8 = 0
    White3D,            ///< 3D RGBA8 = 1
    GreyCube,           ///< cubemap RGBA16F gris 0.5
    ZeroBuffer,         ///< buffer de 64 Kio remis à zéro chaque frame (indirect = rien dessiné)
    Count
};

enum class Producer : std::uint8_t { Core, Dev1, Dev2 };

struct KeyInfo {
    std::string_view key;
    Kind             kind;
    Fallback         fallback;
    Producer         producer;
    std::string_view format;
    std::string_view description;
};

// --- Clés ---------------------------------------------------------------------
inline constexpr std::string_view SceneColor             = "scene.color";
inline constexpr std::string_view SceneDepth             = "scene.depth";
inline constexpr std::string_view SceneDisplay           = "scene.display";
inline constexpr std::string_view FrameGlobalUBO         = "frame.globalUBO";
inline constexpr std::string_view WorldFocusHeightNormal = "world.focus.heightNormal";
inline constexpr std::string_view WorldFocusBiomeHydro   = "world.focus.biomeHydro";
inline constexpr std::string_view WorldHydroMacroMap     = "world.hydro.macroMap";
inline constexpr std::string_view WorldHydroGraphNodes   = "world.hydro.graphNodes";
inline constexpr std::string_view WorldHydroGraphEdges   = "world.hydro.graphEdges";
inline constexpr std::string_view WorldFoliageCounters   = "world.foliage.counters";
inline constexpr std::string_view GBufferAlbedo          = "gbuffer.albedo";
inline constexpr std::string_view GBufferNormal          = "gbuffer.normal";
inline constexpr std::string_view GBufferMaterial        = "gbuffer.material";
inline constexpr std::string_view GBufferVelocity        = "gbuffer.velocity";
inline constexpr std::string_view GBufferDepth           = "gbuffer.depth";
inline constexpr std::string_view AtmoTransmittanceLUT   = "atmo.transmittanceLUT";
inline constexpr std::string_view AtmoMultiScatteringLUT = "atmo.multiScatteringLUT";
inline constexpr std::string_view AtmoSkyViewLUT         = "atmo.skyViewLUT";
inline constexpr std::string_view IblEnvironment         = "ibl.environment";
inline constexpr std::string_view IblIrradiance          = "ibl.irradiance";
inline constexpr std::string_view IblPrefiltered         = "ibl.prefiltered";
inline constexpr std::string_view IblBrdfLUT             = "ibl.brdfLUT";
inline constexpr std::string_view CloudShapeNoise        = "clouds.shapeNoise";
inline constexpr std::string_view CloudDetailNoise       = "clouds.detailNoise";
inline constexpr std::string_view CloudWeatherMap        = "clouds.weatherMap";
inline constexpr std::string_view CloudShadowMap         = "clouds.shadowMap";
inline constexpr std::string_view VolumetricGodRays      = "volumetrics.godRays";
inline constexpr std::string_view WaterOceanDisplacement = "water.oceanDisplacement";
inline constexpr std::string_view WaterOceanNormal       = "water.oceanNormal";
inline constexpr std::string_view VoxelGrid              = "voxel.grid";

inline constexpr KeyInfo kKeys[] = {
    // key                     kind                 fallback                     producer          format                         description
    {SceneColor,             Kind::Texture2D, Fallback::Black,             Producer::Core, "RGBA16F",            "HDR lineaire, taille viewport, cible par defaut des passes"},
    {SceneDepth,             Kind::Texture2D, Fallback::Black,             Producer::Core, "DEPTH32F",           "Reverse-Z (0 = infini), attachee avec scene.color"},
    {SceneDisplay,           Kind::Texture2D, Fallback::Black,             Producer::Dev2, "RGBA8 / RGBA16F",    "Image finale tonemappee, affichee dans le Viewport si publiee"},
    {FrameGlobalUBO,         Kind::Buffer,    Fallback::ZeroBuffer,        Producer::Core, "std140, 672 o",      "GlobalEnvironmentUBO de la frame"},
    {WorldFocusHeightNormal, Kind::Texture2D, Fallback::FlatHeightNormal,  Producer::Dev1, "RGBA32F 256^2",      "heightNormalTex du chunk sous la camera"},
    {WorldFocusBiomeHydro,   Kind::Texture2D, Fallback::NeutralBiomeHydro, Producer::Dev1, "RGBA16F 256^2",      "biomeHydroTex du chunk sous la camera"},
    {WorldHydroMacroMap,     Kind::Texture2D, Fallback::Black,             Producer::Dev1, "RGBA16F",            "Carte macro : bassins, debit, distance a la cote"},
    {WorldHydroGraphNodes,   Kind::Buffer,    Fallback::ZeroBuffer,        Producer::Dev1, "SSBO",               "Noeuds du macro-graphe hydrologique"},
    {WorldHydroGraphEdges,   Kind::Buffer,    Fallback::ZeroBuffer,        Producer::Dev1, "SSBO",               "Aretes du macro-graphe hydrologique"},
    {WorldFoliageCounters,   Kind::Buffer,    Fallback::ZeroBuffer,        Producer::Dev1, "SSBO uint[]",        "Compteurs atomiques du scatter"},
    {GBufferAlbedo,          Kind::Texture2D, Fallback::Checker,           Producer::Dev2, "SRGB8_ALPHA8",       "rgb albedo, a occlusion"},
    {GBufferNormal,          Kind::Texture2D, Fallback::FlatNormalWS,      Producer::Dev2, "RGBA16F",            "xyz normale monde, w libre"},
    {GBufferMaterial,        Kind::Texture2D, Fallback::Black,             Producer::Dev2, "RGBA8",              "r rugosite, g metal, b humidite/neige, a id materiau"},
    {GBufferVelocity,        Kind::Texture2D, Fallback::Black,             Producer::Dev2, "RG16F",              "Vecteurs de mouvement (NDC)"},
    {GBufferDepth,           Kind::Texture2D, Fallback::Black,             Producer::Dev2, "DEPTH32F",           "Profondeur reverse-Z du G-Buffer"},
    {AtmoTransmittanceLUT,   Kind::Texture2D, Fallback::White,             Producer::Dev2, "RGBA16F 256x64",     "LUT de transmittance atmospherique"},
    {AtmoMultiScatteringLUT, Kind::Texture2D, Fallback::Black,             Producer::Dev2, "RGBA16F 32x32",      "LUT de multi-diffusion"},
    {AtmoSkyViewLUT,         Kind::Texture2D, Fallback::Black,             Producer::Dev2, "RGBA16F 192x108",    "LUT sky-view"},
    {IblEnvironment,         Kind::Cubemap,   Fallback::GreyCube,          Producer::Dev2, "RGBA16F cube",       "Capture du ciel (time-sliced, double buffer)"},
    {IblIrradiance,          Kind::Cubemap,   Fallback::GreyCube,          Producer::Dev2, "RGBA16F cube 32^2",  "Irradiance diffuse"},
    {IblPrefiltered,         Kind::Cubemap,   Fallback::GreyCube,          Producer::Dev2, "RGBA16F cube + mips","Speculaire prefiltre, mip = rugosite"},
    {IblBrdfLUT,             Kind::Texture2D, Fallback::White,             Producer::Dev2, "RG16F 512^2",        "LUT BRDF split-sum"},
    {CloudShapeNoise,        Kind::Texture3D, Fallback::White3D,           Producer::Dev2, "RGBA8 128^3",        "Bruit de forme des nuages"},
    {CloudDetailNoise,       Kind::Texture3D, Fallback::White3D,           Producer::Dev2, "RGBA8 32^3",         "Bruit de detail des nuages"},
    {CloudWeatherMap,        Kind::Texture2D, Fallback::Black,             Producer::Dev2, "RGBA8",              "Couverture / type / precipitation"},
    {CloudShadowMap,         Kind::Texture2D, Fallback::White,             Producer::Dev2, "R16F",               "Transmittance des nuages vue du soleil"},
    {VolumetricGodRays,      Kind::Texture2D, Fallback::Black,             Producer::Dev2, "RGBA16F demi-res",   "God rays"},
    {WaterOceanDisplacement, Kind::Texture2D, Fallback::Black,             Producer::Dev2, "RGBA16F",            "Deplacement FFT de l'ocean"},
    {WaterOceanNormal,       Kind::Texture2D, Fallback::FlatNormalWS,      Producer::Dev2, "RGBA16F",            "Normale monde de l'ocean"},
    {VoxelGrid,              Kind::Texture3D, Fallback::Black3D,           Producer::Dev2, "RGBA16F 256x64x256", "Grille VXGI (ADR-008) : rgb radiance premultipliee par l'opacite, a opacite"},
};

constexpr const KeyInfo* FindKey(std::string_view key) noexcept {
    for (const KeyInfo& info : kKeys)
        if (info.key == key) return &info;
    return nullptr;
}

constexpr std::string_view ProducerName(Producer producer) noexcept {
    switch (producer) {
        case Producer::Core: return "Core";
        case Producer::Dev1: return "Dev 1";
        case Producer::Dev2: return "Dev 2";
    }
    return "?";
}

namespace detail {
constexpr bool KeysUnique() noexcept {
    constexpr std::size_t n = sizeof(kKeys) / sizeof(kKeys[0]);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
            if (kKeys[i].key == kKeys[j].key) return false;
    return true;
}
}  // namespace detail
static_assert(detail::KeysUnique(), "RenderBlackboard: cle declaree deux fois dans bb::kKeys");

}  // namespace bb
}  // namespace env
