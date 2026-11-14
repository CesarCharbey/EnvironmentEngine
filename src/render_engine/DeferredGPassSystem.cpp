// =============================================================================
//  src/render_engine/DeferredGPassSystem.cpp                        [DEV 2]
//  Stage : GBufferPass
//
//  Rôle : remplissage du G-Buffer du pipeline différé (ADR-004) à partir du
//  terrain et de la végétation de tous les chunks prêts (mock inclus). Ce système
//  possède le G-Buffer et le publie dans le blackboard (clés gbuffer.*).
//  La résolution de l'éclairage est faite par DeferredLightingSystem.
//
//  * Update : taille du G-Buffer (+ republication), liste des chunks visibles
//             (frustum vs AABB issue de heightRange) et niveau de détail par distance.
//  * Render : passe terrain (grilles LOD déplacées par heightNormalTex, jupes anti-fentes)
//             puis passe végétation (un glMultiDrawElementsIndirect par chunk).
//
//  Végétation : l'element buffer est l'identité (0..N-1) ; gbuffer_foliage.vert génère la forme
//  depuis gl_VertexID. Les commandes indirectes de Dev 1 doivent donc garder count = 36,
//  firstIndex = 0, baseVertex = 0 (comme le mock) tant que les vrais meshes d'archétypes ne sont pas
//  convenus entre Dev 1 et Dev 2.
//
//  RÈGLE : ce fichier n'inclut jamais src/world_engine/ (vérifié au build).
// =============================================================================
#include <imgui.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <entt/entt.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "contracts/ISystemModule.hpp"
#include "contracts/WorldContracts.hpp"
#include "contracts/GpuBindings.hpp"
#include "core/SystemRegistry.hpp"
#include "core/RenderBlackboard.hpp"
#include "core/gpu/ShaderManager.hpp"
#include "core/gpu/Framebuffer.hpp"
#include "core/gpu/GpuBuffer.hpp"

namespace env::render {
namespace {

constexpr std::string_view kProducer = "DeferredGPassSystem";
constexpr int              kLodCount = 4;
/// Capacité de l'element buffer identité de la végétation (>= DrawElementsIndirectCommand::count).
constexpr std::uint32_t    kFoliageIndexCapacity = 1024;
/// Marge verticale (m) ajoutée à l'AABB d'un chunk pour la végétation (arbres).
constexpr float            kFoliageHeightMargin = 15.0f;

/// Miroir std140 de MaterialParamsBlock (shaders/render/material_params.glsl), UBO_MaterialParams.
struct MaterialParamsUbo {
    glm::vec4 snowLine{0.0f};  ///< x = premières neiges (m), y = neige maximale (m), z = demi-transition (m)
};
static_assert(sizeof(MaterialParamsUbo) == 16);

/// Grille de terrain d'un niveau de détail : res x res quads sur [0, kChunkSizeMeters]² + jupes.
struct TerrainGrid {
    gpu::GpuBuffer   vertices;
    gpu::GpuBuffer   indices;
    gpu::VertexArray vao;
    GLsizei          indexCount = 0;
    int              resolution = 0;
};

/// Sommet : x, z = position locale (m) ; y = 1 pour un sommet de jupe (abaissé dans le shader).
void BuildTerrainGrid(TerrainGrid& grid, int res, const char* debugName) {
    res = std::max(res, 2);
    const float step = kChunkSizeMeters / static_cast<float>(res);
    std::vector<glm::vec3>     v;
    std::vector<std::uint32_t> idx;
    v.reserve(static_cast<std::size_t>((res + 1) * (res + 1) + 4 * (res + 1)));
    idx.reserve(static_cast<std::size_t>(res * res * 6 + 4 * res * 12));

    for (int j = 0; j <= res; ++j)
        for (int i = 0; i <= res; ++i)
            v.emplace_back(static_cast<float>(i) * step, 0.0f, static_cast<float>(j) * step);

    auto at = [res](int i, int j) { return static_cast<std::uint32_t>(j * (res + 1) + i); };
    for (int j = 0; j < res; ++j)
        for (int i = 0; i < res; ++i) {
            const std::uint32_t a = at(i, j), b = at(i + 1, j), c = at(i, j + 1), d = at(i + 1, j + 1);
            idx.insert(idx.end(), {a, c, b, b, c, d});  // CCW vu de +Y
        }

    // Jupes : quads verticaux le long des 4 bords, émis dans les deux sens (pas de souci de culling).
    auto addSkirt = [&](auto edgeVertex) {
        const auto base = static_cast<std::uint32_t>(v.size());
        for (int k = 0; k <= res; ++k) {
            glm::vec3 p = v[edgeVertex(k)];
            p.y = 1.0f;
            v.push_back(p);
        }
        for (int k = 0; k < res; ++k) {
            const std::uint32_t e0 = edgeVertex(k), e1 = edgeVertex(k + 1);
            const std::uint32_t s0 = base + static_cast<std::uint32_t>(k), s1 = s0 + 1;
            idx.insert(idx.end(), {e0, s0, e1, e1, s0, s1, e0, e1, s0, e1, s1, s0});
        }
    };
    addSkirt([&](int k) { return at(k, 0); });
    addSkirt([&](int k) { return at(k, res); });
    addSkirt([&](int k) { return at(0, k); });
    addSkirt([&](int k) { return at(res, k); });

    grid.vertices.CreateFrom(std::span<const glm::vec3>(v), 0, debugName);
    grid.indices.CreateFrom(std::span<const std::uint32_t>(idx), 0, debugName);
    grid.vao.Create(debugName);
    grid.vao.SetVertexBuffer(0, grid.vertices, 0, sizeof(glm::vec3));
    grid.vao.SetElementBuffer(grid.indices);
    grid.vao.SetAttribute(0, 0, 3, GL_FLOAT, 0);
    grid.indexCount = static_cast<GLsizei>(idx.size());
    grid.resolution = res;
}

/// Plans gauche / droite / bas / haut de viewProj (Gribb-Hartmann). Le plan lointain est infini et
/// le plan proche négligeable pour des chunks de 256 m : 4 plans suffisent.
std::array<glm::vec4, 4> FrustumPlanes(const glm::mat4& m) {
    auto row = [&](int r) { return glm::vec4(m[0][r], m[1][r], m[2][r], m[3][r]); };
    return {row(3) + row(0), row(3) - row(0), row(3) + row(1), row(3) - row(1)};
}

bool AabbInFrustum(const std::array<glm::vec4, 4>& planes, const glm::vec3& mn, const glm::vec3& mx) {
    for (const glm::vec4& p : planes) {
        const glm::vec3 positive(p.x >= 0.0f ? mx.x : mn.x, p.y >= 0.0f ? mx.y : mn.y, p.z >= 0.0f ? mx.z : mn.z);
        if (glm::dot(glm::vec3(p), positive) + p.w < 0.0f) return false;
    }
    return true;
}

float DistanceToAabb(const glm::vec3& p, const glm::vec3& mn, const glm::vec3& mx) {
    return glm::length(glm::max(glm::max(mn - p, p - mx), glm::vec3(0.0f)));
}

class DeferredGPassSystem final : public ISystemModule {
public:
    void Init(EngineContext& ctx) override {
        terrainProg_ = &ctx.shaders.LoadGraphics("render.gbuffer_terrain", "render/gbuffer_terrain.vert",
                                                 "render/gbuffer_terrain.frag");
        foliageProg_ = &ctx.shaders.LoadGraphics("render.gbuffer_foliage", "render/gbuffer_foliage.vert",
                                                 "render/gbuffer_foliage.frag");

        // Végétation : pas d'attribut, element buffer identité -> gl_VertexID = index.
        std::vector<std::uint32_t> identity(kFoliageIndexCapacity);
        for (std::uint32_t i = 0; i < kFoliageIndexCapacity; ++i) identity[i] = i;
        foliageIndices_.CreateFrom(std::span<const std::uint32_t>(identity), 0, "gpass.foliageIndices");
        foliageVao_.Create("gpass.foliageVao");
        foliageVao_.SetElementBuffer(foliageIndices_);
        materialUbo_.Create(sizeof(MaterialParamsUbo), GL_DYNAMIC_STORAGE_BIT, nullptr, "gpass.materialParams");

        gpu::Framebuffer::Desc desc;
        desc.width  = ScaledSize(ctx.viewportSize.x);
        desc.height = ScaledSize(ctx.viewportSize.y);
        desc.colors = {
            {GL_SRGB8_ALPHA8, {}, "gbuffer.albedo"},
            {GL_RGBA16F, {}, "gbuffer.normal"},
            {GL_RGBA8, {}, "gbuffer.material"},
            {GL_RG16F, {}, "gbuffer.velocity"},
        };
        desc.depthFormat = GL_DEPTH_COMPONENT32F;
        desc.debugName   = "gbuffer";
        gbuffer_.Create(desc);
        Publish(ctx);
    }

    void Update(EngineContext& ctx) override {
        // --- Taille du G-Buffer : les ids de texture changent à chaque réallocation -> republier ---
        if (gbuffer_.Resize(ScaledSize(ctx.viewportSize.x), ScaledSize(ctx.viewportSize.y))) Publish(ctx);

        // --- Chunks visibles + niveau de détail (CPU uniquement) ---
        drawList_.clear();
        chunksCulled_ = 0;
        const auto      planes = FrustumPlanes(ctx.environment.viewProj);
        const glm::vec3 cam    = glm::vec3(ctx.environment.cameraPosition);

        auto view = ctx.registry.view<const ChunkCoordComponent, const ChunkStateComponent, const ChunkGpuResources>();
        for (const entt::entity e : view) {
            const auto& [coord, state, res] = view.get(e);
            if (state.state == ChunkState::Evicting) continue;
            if ((state.readyMask & ChunkReady::HeightNormal) == 0 || res.heightNormalTex == 0) continue;

            const glm::vec3 offset = ChunkOriginRelative(coord.coord, ctx.renderOriginChunk);
            // heightRange (0, 0) = inconnu : plage large par défaut.
            const bool      known  = res.heightRange.x != 0.0f || res.heightRange.y != 0.0f;
            const float     hMin   = known ? res.heightRange.x : -500.0f;
            const float     hMax   = known ? res.heightRange.y : 3000.0f;
            const glm::vec3 mn     = offset + glm::vec3(0.0f, hMin - skirtDepth_, 0.0f);
            const glm::vec3 mx     = offset + glm::vec3(kChunkSizeMeters, hMax + kFoliageHeightMargin, kChunkSizeMeters);
            if (!AabbInFrustum(planes, mn, mx)) {
                ++chunksCulled_;
                continue;
            }

            const float dist = DistanceToAabb(cam, mn, mx);
            int lod = 0;
            while (lod < kLodCount - 1 && dist > lodDistances_[lod]) ++lod;

            DrawItem item;
            item.entity  = e;
            item.offset  = offset;
            item.lod     = lod;
            item.foliage = (state.readyMask & ChunkReady::Foliage) != 0 && res.foliageInstanceSSBO != 0 &&
                           res.foliageIndirectCmdBuffer != 0 && res.foliageCommandCount > 0 &&
                           dist <= foliageDrawDistance_;
            drawList_.push_back(item);
        }
    }

    void Render(EngineContext& ctx) override {
        if (builtGridResolution_ != terrainGridResolution_) RebuildTerrainGrids();

        // Effacement (toujours : le G-Buffer est publié même si un shader est en erreur).
        gbuffer_.Bind();
        for (GLuint i = 0; i < 4; ++i) gbuffer_.ClearColor(i, 0.0f, 0.0f, 0.0f, 0.0f);
        gbuffer_.ClearDepth(0.0f);  // Reverse-Z : 0 = infini

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_GREATER);
        glDepthMask(GL_TRUE);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        // L'albédo est SRGB8_ALPHA8 : les shaders écrivent du linéaire, le GPU encode en sRGB.
        glEnable(GL_FRAMEBUFFER_SRGB);
        if (wireframe_) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        const GLuint neutralBiome = ctx.blackboard.FallbackTexture(bb::Fallback::NeutralBiomeHydro).id;
        chunksDrawn_ = 0;

        // Paramètres de matériaux, aussi lus par la voxélisation VXGI (stage LightingWater, plus loin dans
        // la frame) : le binding indexé persiste.
        MaterialParamsUbo material;
        material.snowLine = glm::vec4(snowLineFirst_, snowLineFull_, snowTransition_, 0.0f);
        materialUbo_.UploadObject(material);
        materialUbo_.BindBase(GL_UNIFORM_BUFFER, env::bind::ubo::MaterialParams);

        // --- Passe terrain ---
        if (drawTerrain_ && terrainProg_->IsValid()) {
            terrainProg_->Use();
            terrainProg_->Set("uSkirtDepth", skirtDepth_);
            for (const DrawItem& item : drawList_) {
                const ChunkGpuResources* res = Resources(ctx, item);
                if (!res) continue;
                const auto& state = ctx.registry.get<ChunkStateComponent>(item.entity);
                const bool hasBiome = (state.readyMask & ChunkReady::BiomeHydro) != 0 && res->biomeHydroTex != 0;
                glBindTextureUnit(env::bind::tex::HeightNormal, res->heightNormalTex);
                glBindTextureUnit(env::bind::tex::BiomeHydro, hasBiome ? res->biomeHydroTex : neutralBiome);
                terrainProg_->Set("uChunkOffset", item.offset);

                const TerrainGrid& grid = terrainGrids_[static_cast<std::size_t>(item.lod)];
                grid.vao.Bind();
                glDrawElements(GL_TRIANGLES, grid.indexCount, GL_UNSIGNED_INT, nullptr);
                ++chunksDrawn_;
            }
        }

        // --- Passe végétation ---
        if (drawFoliage_ && foliageProg_->IsValid()) {
            foliageProg_->Use();
            foliageProg_->Set("uWindStrength", windStrength_);
            foliageVao_.Bind();
            for (const DrawItem& item : drawList_) {
                if (!item.foliage) continue;
                const ChunkGpuResources* res = Resources(ctx, item);
                if (!res) continue;
                foliageProg_->Set("uChunkOffset", item.offset);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, env::bind::ssbo::FoliageInstances, res->foliageInstanceSSBO);
                glBindBuffer(GL_DRAW_INDIRECT_BUFFER, res->foliageIndirectCmdBuffer);
                glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, nullptr,
                                            static_cast<GLsizei>(res->foliageCommandCount), 0);
            }
        }

        // Restauration (le core remet aussi l'état à zéro, mais pas GL_FRAMEBUFFER_SRGB).
        glDisable(GL_FRAMEBUFFER_SRGB);
        glDisable(GL_CULL_FACE);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
        glBindVertexArray(0);
    }

    void Shutdown(EngineContext& ctx) override {
        for (std::string_view key : {bb::GBufferAlbedo, bb::GBufferNormal, bb::GBufferMaterial, bb::GBufferVelocity,
                                     bb::GBufferDepth})
            ctx.blackboard.UnpublishIfProducer(key, kProducer);
        gbuffer_.Destroy();
        for (TerrainGrid& g : terrainGrids_) g = TerrainGrid{};
        foliageVao_.Destroy();
        foliageIndices_.Destroy();
        materialUbo_.Destroy();
        terrainProg_ = foliageProg_ = nullptr;
    }

    void OnGui(EngineContext&) override {
        ImGui::Text("Chunks dessines : %d  (hors champ : %d)", chunksDrawn_, chunksCulled_);
        ImGui::Text("G-Buffer : %d x %d", gbuffer_.Width(), gbuffer_.Height());
        ImGui::SliderFloat("Echelle de resolution", &resolutionScale_, 0.25f, 1.0f, "%.2f");
        ImGui::Checkbox("Terrain", &drawTerrain_);
        ImGui::SameLine();
        ImGui::Checkbox("Vegetation", &drawFoliage_);
        ImGui::SameLine();
        ImGui::Checkbox("Wireframe", &wireframe_);
        ImGui::SliderInt("Resolution grille terrain", &terrainGridResolution_, 16, 256);
        ImGui::SetItemTooltip("Quads par cote au LOD 0 (256 = 1 quad par texel). LOD n = resolution / 2^n.");
        ImGui::DragFloat3("Distances LOD terrain", lodDistances_, 4.0f, 16.0f, 8192.0f, "%.0f m");
        ImGui::SliderFloat("Profondeur des jupes", &skirtDepth_, 0.0f, 50.0f, "%.1f m");
        ImGui::SliderFloat("Distance vegetation", &foliageDrawDistance_, 32.0f, 4096.0f, "%.0f m");
        ImGui::SliderFloat("Force du vent (vegetation)", &windStrength_, 0.0f, 4.0f);

        ImGui::SeparatorText("Neige (quantite : panneau Celestial)");
        ImGui::DragFloat("Premieres neiges", &snowLineFirst_, 1.0f, -500.0f, 9000.0f, "%.0f m");
        ImGui::SetItemTooltip("Altitude de la limite des neiges quand il commence a neiger.");
        ImGui::DragFloat("Neige maximale", &snowLineFull_, 1.0f, -500.0f, 9000.0f, "%.0f m");
        ImGui::SetItemTooltip("Altitude atteinte par la limite des neiges quand la neige meteo vaut 1.");
        ImGui::SliderFloat("Transition", &snowTransition_, 0.5f, 100.0f, "%.1f m", ImGuiSliderFlags_Logarithmic);
    }

private:
    struct DrawItem {
        entt::entity entity = entt::null;
        glm::vec3    offset{0.0f};
        int          lod     = 0;
        bool         foliage = false;
    };

    GLsizei ScaledSize(int viewportPixels) const {
        return std::max(1, static_cast<GLsizei>(static_cast<float>(viewportPixels) * resolutionScale_ + 0.5f));
    }

    void Publish(EngineContext& ctx) {
        ctx.blackboard.PublishTexture(bb::GBufferAlbedo,   gbuffer_.Color(0), kProducer);
        ctx.blackboard.PublishTexture(bb::GBufferNormal,   gbuffer_.Color(1), kProducer);
        ctx.blackboard.PublishTexture(bb::GBufferMaterial, gbuffer_.Color(2), kProducer);
        ctx.blackboard.PublishTexture(bb::GBufferVelocity, gbuffer_.Color(3), kProducer);
        ctx.blackboard.PublishTexture(bb::GBufferDepth,    gbuffer_.Depth(),  kProducer);
    }

    void RebuildTerrainGrids() {
        static constexpr const char* kNames[kLodCount] = {"gpass.terrainLod0", "gpass.terrainLod1",
                                                          "gpass.terrainLod2", "gpass.terrainLod3"};
        for (int lod = 0; lod < kLodCount; ++lod)
            BuildTerrainGrid(terrainGrids_[static_cast<std::size_t>(lod)], std::max(8, terrainGridResolution_ >> lod),
                             kNames[lod]);
        builtGridResolution_ = terrainGridResolution_;
    }

    /// Ressources du chunk, ou nullptr si l'entité a disparu entre Update et Render.
    static const ChunkGpuResources* Resources(EngineContext& ctx, const DrawItem& item) {
        if (!ctx.registry.valid(item.entity)) return nullptr;
        return ctx.registry.try_get<ChunkGpuResources>(item.entity);
    }

    gpu::ShaderProgram*                    terrainProg_ = nullptr;
    gpu::ShaderProgram*                    foliageProg_ = nullptr;
    gpu::Framebuffer                       gbuffer_;
    std::array<TerrainGrid, kLodCount>     terrainGrids_;
    gpu::GpuBuffer                         foliageIndices_;
    gpu::VertexArray                       foliageVao_;
    gpu::GpuBuffer                         materialUbo_;
    std::vector<DrawItem>                  drawList_;
    int                                    builtGridResolution_ = 0;
    int                                    chunksDrawn_         = 0;
    int                                    chunksCulled_        = 0;

    // Paramètres (OnGui)
    float resolutionScale_       = 1.0f;
    bool  drawTerrain_           = true;
    bool  drawFoliage_           = true;
    bool  wireframe_             = false;
    int   terrainGridResolution_ = 128;
    float lodDistances_[3]       = {256.0f, 768.0f, 2048.0f};
    float skirtDepth_            = 8.0f;
    float foliageDrawDistance_   = 600.0f;
    float windStrength_          = 1.0f;
    // Neige : valeurs adaptées au mock (sommet de la colline ≈ 72 m) ; à régler pour le terrain de Dev 1.
    float snowLineFirst_         = 60.0f;
    float snowLineFull_          = -10.0f;
    float snowTransition_        = 4.0f;
};

}  // namespace
REGISTER_SYSTEM(DeferredGPassSystem, PassStage::GBufferPass);
}  // namespace env::render
