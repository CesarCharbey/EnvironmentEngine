// =============================================================================
//  src/render_engine/VoxelIlluminationSystem.cpp                    [DEV 2]
//  Stage : LightingWater (ordre -10, avant DeferredLightingSystem) — ADR-008
//
//  Rôle : voxélisation VXGI centrée sur la caméra. Produit voxel.grid (Texture3D RGBA16F
//  256 × 64 × 256, voxels de 8 m, mips) : rgb = radiance pré-multipliée par l'opacité, a = opacité.
//  DeferredLightingSystem y trace des cônes : ombres lointaines, VXAO, rebond diffus coloré.
//
//  * Update (CPU) : placement de la grille par pas entiers de 8 m (origine stockée en voxels
//    ABSOLUS : exacte et insensible au recalage de l'origine de rendu, ADR-007) ; emprise
//    verticale calée sur le relief (heightRange des chunks) ; liste des chunks qui recouvrent
//    la grille.
//  * Render (GPU) : DOUBLE TAMPON. La grille de la frame est entièrement reconstruite (effacement,
//    injection terrain puis végétation, mips) pendant que la grille précédente sert à ombrer la
//    lumière injectée. Reconstruire ~1 M de voxels utiles coûte une fraction de milliseconde : ni
//    retard d'éclairage quand le soleil bouge vite, ni état torique à maintenir.
//  * Végétation DÉTERMINISTE : épaisseur optique accumulée par atomiques dans un tampon d'entiers
//    (SSBO_Dev2Scratch, auto-nettoyé), puis résolue une fois par voxel. La grille est identique d'une
//    frame à l'autre à scène égale : pas de scintillement des ombres voxel ni du VXAO.
//  * Écriture via IMG_AtmoLutOut (réutilisée, ADR-008), lecture de la grille précédente via
//    TEX_VoxelGrid, placement publié dans UBO_VoxelParams.
//
//  RÈGLE : ce fichier n'inclut jamais src/world_engine/ (vérifié au build).
// =============================================================================
#include <imgui.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <entt/entt.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include "contracts/ISystemModule.hpp"
#include "contracts/WorldContracts.hpp"
#include "contracts/GpuBindings.hpp"
#include "core/SystemRegistry.hpp"
#include "core/RenderBlackboard.hpp"
#include "core/gpu/ShaderManager.hpp"
#include "core/gpu/Texture3D.hpp"
#include "core/gpu/GpuBuffer.hpp"

namespace env::render {
namespace {

constexpr std::string_view kProducer = "VoxelIlluminationSystem";

constexpr int   kDimX      = 256;
constexpr int   kDimY      = 64;
constexpr int   kDimZ      = 256;
constexpr float kVoxelSize = 8.0f;  // m, voxels cubiques
constexpr int   kVoxelsPerChunk  = static_cast<int>(kChunkSizeMeters / kVoxelSize);  // 32
constexpr int   kColumnsPerChunk = kVoxelsPerChunk * kVoxelsPerChunk;
static_assert(static_cast<float>(kVoxelsPerChunk) * kVoxelSize == kChunkSizeMeters,
              "La taille de voxel doit diviser exactement la taille d'un chunk");
/// LOD max exploité par le cone tracing : au LOD 6, l'axe Y (64 voxels) est réduit à 1.
constexpr float kMaxConeLod = 5.0f;
/// Marges verticales (m) : sous le point le plus bas, au-dessus du plus haut (arbres).
constexpr float kMarginBelow = 2.0f * kVoxelSize;
constexpr float kMarginAbove = 8.0f * kVoxelSize;
/// Hauteur max d'une plante voxélisée au-dessus du point le plus haut du chunk (m).
constexpr float kMaxFoliageHeight = kMarginAbove;

/// UBO_VoxelParams (std140) — miroir de VoxelParamsBlock (shaders/render/voxel_common.glsl).
struct alignas(16) VoxelParamsUBO {
    glm::vec4 gridMin;        // xyz coin min (m, relatif origine de rendu), w taille de voxel
    glm::vec4 gridExtent;     // xyz emprise (m), w = 1 si valide
    glm::vec4 gridInvExtent;  // xyz 1 / emprise, w LOD max
    glm::vec4 prevGridMin;    // xyz coin min de la grille précédente (relatif à l'origine COURANTE), w valide
};
static_assert(sizeof(VoxelParamsUBO) == 64, "VoxelParamsUBO: layout std140");

int FloorToVoxel(float meters) { return static_cast<int>(std::floor(meters / kVoxelSize)); }

class VoxelIlluminationSystem final : public ISystemModule {
public:
    void Init(EngineContext& ctx) override {
        program_    = &ctx.shaders.LoadCompute("render.voxel_inject", "render/voxel_inject.comp");
        mipProgram_ = &ctx.shaders.LoadCompute("render.voxel_mip", "render/voxel_mip.comp");

        gpu::Texture3D::Desc desc;
        desc.width             = kDimX;
        desc.height            = kDimY;
        desc.depth             = kDimZ;
        desc.internalFormat    = GL_RGBA16F;
        desc.levels            = 0;  // chaîne de mips complète (cone tracing)
        desc.sampler           = gpu::SamplerDesc{};
        desc.sampler.minFilter = GL_LINEAR_MIPMAP_LINEAR;  // CLAMP_TO_EDGE : les cônes s'arrêtent au bord
        static constexpr const char* kNames[2] = {"voxel.gridA", "voxel.gridB"};
        for (int i = 0; i < 2; ++i) {
            desc.debugName = kNames[i];
            grids_[i].Create(desc);
            grids_[i].Clear(GL_RGBA, GL_FLOAT, nullptr);
        }
        ubo_.Create(sizeof(VoxelParamsUBO), GL_DYNAMIC_STORAGE_BIT, nullptr, "voxel.paramsUBO");
        // Un uint par voxel (16 Mio). Nul au départ ; la passe de résolution le remet à zéro chaque frame.
        foliageAccum_.Create(static_cast<GLsizeiptr>(kDimX) * kDimY * kDimZ * sizeof(std::uint32_t),
                             GL_DYNAMIC_STORAGE_BIT, nullptr, "voxel.foliageAccum");
        foliageAccum_.ClearZero();
    }

    void Update(EngineContext& ctx) override {
        const core::RenderBlackboard& bb = ctx.blackboard;
        atmoAvailable_ = bb.IsPublished(env::bb::AtmoSkyViewLUT) && bb.IsPublished(env::bb::AtmoTransmittanceLUT);

        // --- XZ : grille centrée sur la caméra, par pas entiers de voxel ---
        const glm::vec3 cam(ctx.environment.cameraPosition);
        const int relX = FloorToVoxel(cam.x) - kDimX / 2;
        const int relZ = FloorToVoxel(cam.z) - kDimZ / 2;
        const float minX = static_cast<float>(relX) * kVoxelSize, maxX = minX + kDimX * kVoxelSize;
        const float minZ = static_cast<float>(relZ) * kVoxelSize, maxZ = minZ + kDimZ * kVoxelSize;

        // --- Chunks qui recouvrent la grille + plage de hauteurs connue ---
        chunks_.clear();
        float hMin = std::numeric_limits<float>::max(), hMax = std::numeric_limits<float>::lowest();
        auto view = ctx.registry.view<const ChunkCoordComponent, const ChunkStateComponent, const ChunkGpuResources>();
        for (const entt::entity e : view) {
            const auto& [coord, state, res] = view.get(e);
            if (state.state == ChunkState::Evicting) continue;
            if ((state.readyMask & ChunkReady::HeightNormal) == 0 || res.heightNormalTex == 0) continue;
            const glm::vec3 offset = ChunkOriginRelative(coord.coord, ctx.renderOriginChunk);
            if (offset.x >= maxX || offset.x + kChunkSizeMeters <= minX || offset.z >= maxZ ||
                offset.z + kChunkSizeMeters <= minZ)
                continue;
            if (res.heightRange.x != 0.0f || res.heightRange.y != 0.0f) {  // (0, 0) = inconnu
                hMin = std::min(hMin, res.heightRange.x);
                hMax = std::max(hMax, res.heightRange.y);
            }
            chunks_.push_back({e, offset, res.heightRange});
        }

        // --- Y : emprise de 512 m calée sur le relief (et non sur y = 0) ---
        // Relief plus haut que la grille : on garde la tranche autour de la caméra.
        const float height = kDimY * kVoxelSize;
        float bottom;
        if (hMin <= hMax) {
            bottom = hMin - kMarginBelow;
            if (hMax + kMarginAbove > bottom + height)
                bottom = std::clamp(cam.y - 0.5f * height, hMin - kMarginBelow, hMax + kMarginAbove - height);
        } else {
            bottom = cam.y - 0.5f * height;
        }
        const int relY = FloorToVoxel(bottom);

        // Origine absolue (voxels) : exacte, indépendante de l'origine de rendu.
        const glm::ivec3 originVox(ctx.renderOriginChunk.x * kVoxelsPerChunk, 0, ctx.renderOriginChunk.y * kVoxelsPerChunk);
        currentAbs_ = glm::ivec3(relX, relY, relZ) + originVox;

        const glm::vec3 extent(kDimX * kVoxelSize, kDimY * kVoxelSize, kDimZ * kVoxelSize);
        params_.gridMin       = glm::vec4(glm::vec3(currentAbs_ - originVox) * kVoxelSize, kVoxelSize);
        params_.gridExtent    = glm::vec4(extent, 1.0f);
        params_.gridInvExtent = glm::vec4(1.0f / extent, kMaxConeLod);
        params_.prevGridMin   = glm::vec4(glm::vec3(prevAbs_ - originVox) * kVoxelSize, prevValid_ ? 1.0f : 0.0f);
    }

    void Render(EngineContext& ctx) override {
        if (!program_ || !program_->IsValid() || !mipProgram_ || !mipProgram_->IsValid()) {
            ctx.blackboard.UnpublishIfProducer(env::bb::VoxelGrid, kProducer);  // l'éclairage retombe sur son repli
            prevValid_ = false;
            return;
        }

        gpu::Texture3D& grid = grids_[current_];
        gpu::Texture3D& prev = grids_[current_ ^ 1];

        ubo_.UploadObject(params_);
        ubo_.BindBase(GL_UNIFORM_BUFFER, env::bind::ubo::VoxelParams);

        grid.Clear(GL_RGBA, GL_FLOAT, nullptr);  // niveau 0 ; les mips sont régénérés à la fin
        prev.BindUnit(env::bind::tex::VoxelGrid);
        ctx.blackboard.BindTexture(env::bb::AtmoTransmittanceLUT, env::bind::tex::AtmoTransmittance);
        ctx.blackboard.BindTexture(env::bb::AtmoSkyViewLUT, env::bind::tex::AtmoSkyView);
        grid.BindImage(env::bind::image::AtmoLutOut, GL_READ_WRITE, 0, GL_TRUE, 0, GL_RGBA16F);

        const gpu::ShaderProgram& p = *program_;
        p.Set("uAtmoAvailable", atmoAvailable_);
        p.Set("uFoliageDensity", foliageDensity_);
        const GLuint neutralBiome = ctx.blackboard.FallbackTexture(bb::Fallback::NeutralBiomeHydro).id;

        // --- Passe 0 : terrain (une colonne de voxels par thread) ---
        p.Set("uPass", 0);
        chunksInjected_ = 0;
        for (const ChunkItem& item : chunks_) {
            const ChunkGpuResources* res   = Resources(ctx, item);
            const auto*              state = ctx.registry.try_get<ChunkStateComponent>(item.entity);
            if (!res || !state) continue;
            const bool hasBiome = (state->readyMask & ChunkReady::BiomeHydro) != 0 && res->biomeHydroTex != 0;
            glBindTextureUnit(env::bind::tex::HeightNormal, res->heightNormalTex);
            glBindTextureUnit(env::bind::tex::BiomeHydro, hasBiome ? res->biomeHydroTex : neutralBiome);
            p.Set("uChunkOffset", item.offset);
            p.Dispatch(gpu::ShaderProgram::GroupCount(kColumnsPerChunk, 64));
            ++chunksInjected_;
        }
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);  // la résolution de la végétation relit le terrain

        // --- Végétation : passe 1 (accumulation, une instance par thread) puis 2 (résolution, un voxel de
        //     la tranche Y par thread). Mêmes chunks et même tranche pour les deux passes : chaque entrée
        //     accumulée est résolue puis remise à zéro dans la même frame. ---
        instancesDispatched_ = 0;
        foliageItems_.clear();
        if (injectFoliage_) {
            for (const ChunkItem& item : chunks_) {
                const ChunkGpuResources* res   = Resources(ctx, item);
                const auto*              state = ctx.registry.try_get<ChunkStateComponent>(item.entity);
                if (!res || !state || (state->readyMask & ChunkReady::Foliage) == 0) continue;
                if (res->foliageInstanceSSBO == 0 || res->foliageIndirectCmdBuffer == 0 ||
                    res->foliageInstanceCapacity == 0 || res->foliageCommandCount == 0)
                    continue;
                foliageItems_.push_back({res, item.offset, FoliageYRange(item.heightRange)});
            }
        }
        if (!foliageItems_.empty()) {
            foliageAccum_.BindBase(GL_SHADER_STORAGE_BUFFER, env::bind::ssbo::Dev2Scratch);
            p.Set("uPass", 1);
            for (const FoliageItem& f : foliageItems_) {
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, env::bind::ssbo::FoliageInstances, f.res->foliageInstanceSSBO);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, env::bind::ssbo::FoliageIndirectCmds,
                                 f.res->foliageIndirectCmdBuffer);
                p.Set("uChunkOffset", f.offset);
                p.Set("uFoliageYRange", f.yRange);
                p.Set("uInstanceCapacity", f.res->foliageInstanceCapacity);
                p.Set("uCommandCount", f.res->foliageCommandCount);
                p.Dispatch(gpu::ShaderProgram::GroupCount(f.res->foliageInstanceCapacity, 64));
                instancesDispatched_ += f.res->foliageInstanceCapacity;
            }
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);  // sommes atomiques visibles pour la résolution

            p.Set("uPass", 2);
            for (const FoliageItem& f : foliageItems_) {
                p.Set("uChunkOffset", f.offset);
                p.Set("uFoliageYRange", f.yRange);
                const auto layers = static_cast<std::uint32_t>(f.yRange.y - f.yRange.x + 1);
                p.Dispatch(gpu::ShaderProgram::GroupCount(kColumnsPerChunk * layers, 64));
            }
        }

        // Écritures image du niveau 0 visibles pour les mips (texelFetch) puis le cone tracing ; remise à
        // zéro du tampon de végétation visible pour les atomiques de la frame suivante.
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT |
                        GL_SHADER_STORAGE_BARRIER_BIT);
        GenerateMips(grid);

        ctx.blackboard.PublishTexture(env::bb::VoxelGrid, grid, kProducer);
        prevAbs_   = currentAbs_;
        prevValid_ = true;
        current_ ^= 1;
    }

    void Shutdown(EngineContext& ctx) override {
        ctx.blackboard.UnpublishIfProducer(env::bb::VoxelGrid, kProducer);
        grids_[0].Destroy();
        grids_[1].Destroy();
        ubo_.Destroy();
        foliageAccum_.Destroy();
        program_ = nullptr;
    }

    void OnGui(EngineContext& ctx) override {
        ImGui::Text("Grille : %d x %d x %d voxels de %.0f m  (%.0f x %.0f x %.0f m)", kDimX, kDimY, kDimZ, kVoxelSize,
                    kDimX * kVoxelSize, kDimY * kVoxelSize, kDimZ * kVoxelSize);
        ImGui::Text("Origine (voxels absolus) : %d, %d, %d", currentAbs_.x, currentAbs_.y, currentAbs_.z);
        ImGui::Text("Altitudes couvertes : %.0f m -> %.0f m", currentAbs_.y * kVoxelSize,
                    (currentAbs_.y + kDimY) * kVoxelSize);
        ImGui::Text("Chunks injectes : %d   instances traitees : %u", chunksInjected_, instancesDispatched_);
        ImGui::SeparatorText("Vegetation");
        ImGui::Checkbox("Injecter la vegetation", &injectFoliage_);
        ImGui::SliderFloat("Densite (opacite)", &foliageDensity_, 0.0f, 4.0f);
        ImGui::SetItemTooltip("Opacite d'une plante dans un voxel de 8 m. Plus haut = forets plus sombres.");
        if (ImGui::Button("Inspecter voxel.grid")) ctx.blackboard.Inspect(env::bb::VoxelGrid);
        ImGui::SetItemTooltip("Panneau Render Blackboard : coupe de la grille (canal A = opacite).");
    }

private:
    struct ChunkItem {
        entt::entity entity = entt::null;
        glm::vec3    offset{0.0f};
        glm::vec2    heightRange{0.0f};  // (0, 0) = inconnu
    };
    struct FoliageItem {
        const ChunkGpuResources* res = nullptr;
        glm::vec3                offset{0.0f};
        glm::ivec2               yRange{0};
    };

    /// Tranche Y (voxels de la grille courante) où la végétation du chunk peut se trouver.
    glm::ivec2 FoliageYRange(const glm::vec2& heightRange) const {
        if (heightRange.x == 0.0f && heightRange.y == 0.0f) return {0, kDimY - 1};
        const float gridY0 = params_.gridMin.y;
        const int   lo     = FloorToVoxel(heightRange.x - gridY0);
        const int   hi     = FloorToVoxel(heightRange.y + kMaxFoliageHeight - gridY0);
        return {std::clamp(lo, 0, kDimY - 1), std::clamp(hi, 0, kDimY - 1)};
    }

    /// Mips par compute (voxel_mip.comp) : glGenerateTextureMipmap sur une texture 3D RGBA16F coûte
    /// ~30 ms sur NVIDIA (chemin lent du pilote, mesuré), contre quelques µs ici.
    void GenerateMips(gpu::Texture3D& grid) {
        grid.BindUnit(env::bind::tex::VoxelGrid);  // lecture du niveau N-1 (texelFetch)
        for (GLint level = 1; level < grid.Levels(); ++level) {
            grid.BindImage(env::bind::image::AtmoLutOut, GL_WRITE_ONLY, level, GL_TRUE, 0, GL_RGBA16F);
            mipProgram_->Set("uSrcLevel", static_cast<int>(level - 1));
            mipProgram_->Dispatch(gpu::ShaderProgram::GroupCount(static_cast<GLuint>(grid.LevelWidth(level)), 4),
                                  gpu::ShaderProgram::GroupCount(static_cast<GLuint>(grid.LevelHeight(level)), 4),
                                  gpu::ShaderProgram::GroupCount(static_cast<GLuint>(grid.LevelDepth(level)), 4));
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
        }
    }

    /// Ressources du chunk, ou nullptr si l'entité a disparu entre Update et Render.
    static const ChunkGpuResources* Resources(EngineContext& ctx, const ChunkItem& item) {
        if (!ctx.registry.valid(item.entity)) return nullptr;
        return ctx.registry.try_get<ChunkGpuResources>(item.entity);
    }

    gpu::ShaderProgram*           program_    = nullptr;
    gpu::ShaderProgram*           mipProgram_ = nullptr;
    std::array<gpu::Texture3D, 2> grids_;
    gpu::GpuBuffer                ubo_;
    gpu::GpuBuffer                foliageAccum_;  // épaisseur optique de la végétation (SSBO_Dev2Scratch)
    int                           current_ = 0;

    std::vector<ChunkItem>   chunks_;
    std::vector<FoliageItem> foliageItems_;
    VoxelParamsUBO         params_{};
    glm::ivec3             currentAbs_{0};
    glm::ivec3             prevAbs_{0};
    bool                   prevValid_     = false;
    bool                   atmoAvailable_ = false;
    int                    chunksInjected_      = 0;
    std::uint32_t          instancesDispatched_ = 0;

    // Paramètres (OnGui)
    bool  injectFoliage_  = true;
    float foliageDensity_ = 1.0f;
};

}  // namespace
REGISTER_SYSTEM_ORDERED(VoxelIlluminationSystem, PassStage::LightingWater, -10);
}  // namespace env::render
