#pragma once
// =============================================================================
//  include/contracts/GpuBindings.hpp
//  CONTRAT FIGÉ — Points de binding OpenGL partagés par Core, Dev 1 et Dev 2.
//  Toute modification = nouvel ADR dans DECISIONS.md + accord des deux devs.
//
//  Source unique de vérité :
//    * C++  : env::bind::ubo::GlobalEnvironment, env::bind::tex::HeightNormal, ...
//    * GLSL : UBO_GlobalEnvironment, TEX_HeightNormal, SSBO_FoliageInstances,
//             IMG_HeightNormalOut, ... — #define injectés automatiquement par
//             ShaderManager juste après la ligne #version de chaque shader.
//
//  Plages réservées par propriétaire (vérifiées à la compilation) :
//    UBO      : Core 0-1  | Dev 1 2-7  | Dev 2 8-15
//    SSBO     : Core 0-1  | Dev 1 2-7  | Dev 2 8-15
//    Textures : Core 0-5  | Dev 1 6-9  | Dev 2 10-47 (ADR-008 ; <= 32 samplers ACTIFS par étage)
//    Images   : Dev 1 0-2 | Core 3     | Dev 2 4-7   (GL_MAX_IMAGE_UNITS = 8 sur NVIDIA)
//               IMG_AtmoLutOut sert aussi à écrire voxel.grid (ADR-008 : passes jamais simultanées)
//
//  Une collision de slot ou un slot hors de la plage de son propriétaire est
//  une ERREUR DE COMPILATION (static_assert en bas de fichier).
//
//  Pour ajouter un slot : ajouter une ligne X(Nom, Slot, Owner, "description")
//  dans la bonne liste, puis mettre à jour la table de AGENTS.md (section 7).
// =============================================================================
#include <cstddef>
#include <cstdint>
#include <string_view>

// -----------------------------------------------------------------------------
// X-macros : X(Nom, Slot, Proprietaire, Description)
// -----------------------------------------------------------------------------
#define ENV_UBO_BINDINGS(X)                                                                         \
    X(GlobalEnvironment,  0, Core, "GlobalEnvironmentUBO : camera, soleil, lune, vent, meteo, eau") \
    X(CoreDebug,          1, Core, "Reserve : parametres debug / mock")                             \
    X(WorldChunkParams,   2, Dev1, "Parametres de generation de chunk (bruit, seed, LOD)")          \
    X(BiomeHydroParams,   3, Dev1, "Seuils de biomes + parametres d'hydrologie")                    \
    X(FoliageParams,      4, Dev1, "Densites / archetypes de vegetation")                           \
    X(MaterialParams,     8, Dev2, "Materiaux PBR terrain / vegetation")                            \
    X(AtmosphereParams,   9, Dev2, "Diffusion atmospherique (Rayleigh, Mie, ozone)")                \
    X(IblParams,         10, Dev2, "Parametres de l'IBL dynamique")                                 \
    X(VolumetricParams,  11, Dev2, "Nuages volumetriques + god rays")                               \
    X(WaterParams,       12, Dev2, "Ocean FFT, rivieres, marais")                                   \
    X(PostParams,        13, Dev2, "Exposition, tonemapping")                                      \
    X(VoxelParams,       14, Dev2, "Placement de la grille VXGI (ADR-008)")

#define ENV_SSBO_BINDINGS(X)                                                                        \
    X(CoreDebugLines,       0, Core, "Reserve : lignes de debug")                                   \
    X(CoreScratch,          1, Core, "Reserve : scratch core")                                      \
    X(FoliageInstances,     2, Dev1, "FoliageInstance[] (ecrit par le scatter, lu par la G-pass)")  \
    X(FoliageIndirectCmds,  3, Dev1, "DrawElementsIndirectCommand[] (un par archetype)")            \
    X(FoliageCounters,      4, Dev1, "Compteurs atomiques du scatter")                              \
    X(HydroGraphNodes,      5, Dev1, "Noeuds du macro-graphe hydrologique")                         \
    X(HydroGraphEdges,      6, Dev1, "Aretes du macro-graphe hydrologique")                         \
    X(ChunkTable,           7, Dev1, "Table des chunks residents (coord -> slot)")                  \
    X(LightList,            8, Dev2, "Lumieres ponctuelles (eclairs, feux)")                        \
    X(ExposureHistogram,    9, Dev2, "Histogramme de luminance / exposition auto")                  \
    X(IblShCoefficients,   10, Dev2, "Harmoniques spheriques d'irradiance (SH9)")                   \
    X(OceanSpectrum,       11, Dev2, "Spectre de l'ocean FFT")                                      \
    X(FoliageVisible,      12, Dev2, "Instances visibles apres culling GPU")                        \
    X(Dev2Scratch,         13, Dev2, "Scratch Dev 2")

#define ENV_TEXTURE_BINDINGS(X)                                                                     \
    X(CoreFallback,          0, Core, "Unite neutre / fallback")                                    \
    X(CoreInspect2D,         1, Core, "Inspecteur du blackboard (sampler2D)")                       \
    X(CoreInspect3D,         2, Core, "Inspecteur du blackboard (sampler3D)")                       \
    X(CoreInspectCube,       3, Core, "Inspecteur du blackboard (samplerCube)")                     \
    X(SceneColor,            4, Core, "scene.color RGBA16F HDR")                                    \
    X(SceneDepth,            5, Core, "scene.depth DEPTH32F reverse-Z")                             \
    X(HeightNormal,          6, Dev1, "Chunk heightNormalTex RGBA32F")                              \
    X(BiomeHydro,            7, Dev1, "Chunk biomeHydroTex RGBA16F")                                \
    X(HydroMacro,            8, Dev1, "Carte macro d'hydrologie")                                   \
    X(WorldNoise,            9, Dev1, "Bruit / LUT precalcules du generateur")                      \
    X(GBufferAlbedo,        10, Dev2, "G-Buffer albedo RGBA8 sRGB")                                 \
    X(GBufferNormal,        11, Dev2, "G-Buffer normale RGBA16F")                                   \
    X(GBufferMaterial,      12, Dev2, "G-Buffer materiau RGBA8")                                    \
    X(GBufferVelocity,      13, Dev2, "G-Buffer vitesse RG16F")                                     \
    X(GBufferDepth,         14, Dev2, "G-Buffer profondeur DEPTH32F")                               \
    X(AtmoTransmittance,    15, Dev2, "LUT de transmittance")                                       \
    X(AtmoMultiScattering,  16, Dev2, "LUT de multi-diffusion")                                     \
    X(AtmoSkyView,          17, Dev2, "LUT sky-view")                                               \
    X(IblEnvironment,       18, Dev2, "Cubemap d'environnement")                                    \
    X(IblIrradiance,        19, Dev2, "Cubemap d'irradiance")                                       \
    X(IblPrefiltered,       20, Dev2, "Cubemap speculaire prefiltree")                              \
    X(IblBrdfLut,           21, Dev2, "LUT BRDF")                                                   \
    X(CloudShapeNoise,      22, Dev2, "Bruit 3D de forme des nuages")                               \
    X(CloudDetailNoise,     23, Dev2, "Bruit 3D de detail des nuages")                              \
    X(CloudWeatherMap,      24, Dev2, "Carte meteo des nuages")                                     \
    X(CloudShadow,          25, Dev2, "Ombre des nuages (vue soleil)")                              \
    X(GodRays,              26, Dev2, "God rays demi-resolution")                                   \
    X(OceanDisplacement,    27, Dev2, "Deplacement FFT de l'ocean")                                 \
    X(OceanNormal,          28, Dev2, "Normale de l'ocean")                                         \
    X(FoliageAlbedoAtlas,   29, Dev2, "Atlas d'albedo de la vegetation")                            \
    X(TerrainMaterialArray, 30, Dev2, "Texture array des materiaux de terrain")                     \
    X(BlueNoise,            31, Dev2, "Bruit bleu (jitter temporel)")                               \
    X(VoxelGrid,            32, Dev2, "Grille VXGI 3D RGBA16F (ADR-008)")

#define ENV_IMAGE_BINDINGS(X)                                                                       \
    X(HeightNormalOut,  0, Dev1, "Ecriture compute heightNormalTex (rgba32f)")                      \
    X(BiomeHydroOut,    1, Dev1, "Ecriture compute biomeHydroTex (rgba16f)")                        \
    X(WorldScratchOut,  2, Dev1, "Scratch compute Dev 1")                                           \
    X(CoreScratchOut,   3, Core, "Reserve core")                                                    \
    X(AtmoLutOut,       4, Dev2, "Ecriture des LUTs d'atmosphere")                                  \
    X(IblOut,           5, Dev2, "Ecriture des faces / mips IBL")                                   \
    X(VolumetricOut,    6, Dev2, "Ecriture nuages / god rays")                                      \
    X(WaterOut,         7, Dev2, "Ecriture FFT / eau")

namespace env::bind {

enum class Owner : std::uint8_t { Core, Dev1, Dev2 };

constexpr std::string_view OwnerName(Owner owner) noexcept {
    switch (owner) {
        case Owner::Core: return "Core";
        case Owner::Dev1: return "Dev 1";
        case Owner::Dev2: return "Dev 2";
    }
    return "?";
}

struct SlotInfo {
    std::string_view name;
    std::uint32_t    slot;
    Owner            owner;
    std::string_view description;
};

struct SlotRange {
    Owner         owner;
    std::uint32_t first;
    std::uint32_t last;
};

#define ENV_BIND_DECLARE(Name, Slot, OwnerTag, Desc) inline constexpr std::uint32_t Name = Slot;
#define ENV_BIND_ROW(Name, Slot, OwnerTag, Desc) SlotInfo{#Name, Slot, Owner::OwnerTag, Desc},

// --- Constantes C++ ----------------------------------------------------------
namespace ubo   { ENV_UBO_BINDINGS(ENV_BIND_DECLARE) }
namespace ssbo  { ENV_SSBO_BINDINGS(ENV_BIND_DECLARE) }
namespace tex   { ENV_TEXTURE_BINDINGS(ENV_BIND_DECLARE) }
namespace image { ENV_IMAGE_BINDINGS(ENV_BIND_DECLARE) }

// --- Tables (UI "GPU Bindings", génération des #define GLSL) ------------------
inline constexpr SlotInfo kUboTable[]     = { ENV_UBO_BINDINGS(ENV_BIND_ROW) };
inline constexpr SlotInfo kSsboTable[]    = { ENV_SSBO_BINDINGS(ENV_BIND_ROW) };
inline constexpr SlotInfo kTextureTable[] = { ENV_TEXTURE_BINDINGS(ENV_BIND_ROW) };
inline constexpr SlotInfo kImageTable[]   = { ENV_IMAGE_BINDINGS(ENV_BIND_ROW) };

#undef ENV_BIND_DECLARE
#undef ENV_BIND_ROW

inline constexpr SlotRange kUboRanges[]     = {{Owner::Core, 0, 1}, {Owner::Dev1, 2, 7}, {Owner::Dev2, 8, 15}};
inline constexpr SlotRange kSsboRanges[]    = {{Owner::Core, 0, 1}, {Owner::Dev1, 2, 7}, {Owner::Dev2, 8, 15}};
inline constexpr SlotRange kTextureRanges[] = {{Owner::Core, 0, 5}, {Owner::Dev1, 6, 9}, {Owner::Dev2, 10, 47}};
inline constexpr SlotRange kImageRanges[]   = {{Owner::Dev1, 0, 2}, {Owner::Core, 3, 3}, {Owner::Dev2, 4, 7}};

// Préfixes des #define générés côté GLSL.
inline constexpr std::string_view kGlslPrefixUbo     = "UBO_";
inline constexpr std::string_view kGlslPrefixSsbo    = "SSBO_";
inline constexpr std::string_view kGlslPrefixTexture = "TEX_";
inline constexpr std::string_view kGlslPrefixImage   = "IMG_";

namespace detail {

template <std::size_t N>
constexpr bool SlotsUnique(const SlotInfo (&table)[N]) noexcept {
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i + 1; j < N; ++j)
            if (table[i].slot == table[j].slot) return false;
    return true;
}

template <std::size_t N, std::size_t M>
constexpr bool SlotsInOwnerRanges(const SlotInfo (&table)[N], const SlotRange (&ranges)[M]) noexcept {
    for (std::size_t i = 0; i < N; ++i) {
        bool ok = false;
        for (std::size_t r = 0; r < M; ++r)
            if (ranges[r].owner == table[i].owner && table[i].slot >= ranges[r].first &&
                table[i].slot <= ranges[r].last)
                ok = true;
        if (!ok) return false;
    }
    return true;
}

template <std::size_t N>
constexpr std::uint32_t MaxSlot(const SlotInfo (&table)[N]) noexcept {
    std::uint32_t m = 0;
    for (std::size_t i = 0; i < N; ++i) m = table[i].slot > m ? table[i].slot : m;
    return m;
}

}  // namespace detail

static_assert(detail::SlotsUnique(kUboTable), "GpuBindings: collision de slot UBO");
static_assert(detail::SlotsUnique(kSsboTable), "GpuBindings: collision de slot SSBO");
static_assert(detail::SlotsUnique(kTextureTable), "GpuBindings: collision d'unite de texture");
static_assert(detail::SlotsUnique(kImageTable), "GpuBindings: collision d'unite d'image");

static_assert(detail::SlotsInOwnerRanges(kUboTable, kUboRanges), "GpuBindings: slot UBO hors de la plage de son proprietaire");
static_assert(detail::SlotsInOwnerRanges(kSsboTable, kSsboRanges), "GpuBindings: slot SSBO hors de la plage de son proprietaire");
static_assert(detail::SlotsInOwnerRanges(kTextureTable, kTextureRanges), "GpuBindings: unite de texture hors de la plage de son proprietaire");
static_assert(detail::SlotsInOwnerRanges(kImageTable, kImageRanges), "GpuBindings: unite d'image hors de la plage de son proprietaire");

// Limites minimales garanties sur le matériel visé (NVIDIA / AMD / Intel récents).
static_assert(detail::MaxSlot(kImageTable) < 8, "GL_MAX_IMAGE_UNITS vaut 8 sur NVIDIA");
// GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS >= 80 en GL 4.6 ; le nombre de samplers ACTIFS par étage reste <= 32.
static_assert(detail::MaxSlot(kTextureTable) < 48, "Garder les unites de texture sous 48 (ADR-008)");
static_assert(detail::MaxSlot(kSsboTable) < 16, "Garder les bindings SSBO sous 16");
static_assert(detail::MaxSlot(kUboTable) < 16, "Garder les bindings UBO sous 16");

}  // namespace env::bind
