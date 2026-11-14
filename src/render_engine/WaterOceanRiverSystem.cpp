// =============================================================================
//  src/render_engine/WaterOceanRiverSystem.cpp                      [DEV 2]
//  Stage : LightingWater (ordre 10, après DeferredLightingSystem)
//
//  Rôle : rendu de l'eau en forward par-dessus scene.color (HDR) : océan FFT au niveau
//  waterTime.x, rivières et marais décrits par les SDF des chunks (ADR-005). Les niveaux
//  d'eau sont écrits par BiomeHydrologySystem (Dev 1) et ne sont que lus ici.
//
//  * Océan (Tessendorf 2001) : spectre JONSWAP + Donelan-Banner piloté par le vent de la météo
//    (ocean_spectrum.comp, recalculé quand le vent change), FFT inverse en mémoire partagée,
//    3 cascades dans UNE texture 2D array (ocean_fft.comp, 2 dispatchs par frame), écume par le
//    jacobien, mips par compute (ocean_mip.comp). Surface : grille projetée (ocean_surface.vert).
//  * Rivières / marais : la hauteur de l'eau d'une rivière n'est pas dans le contrat ; elle est
//    reconstruite une fois par chunk à partir des berges (river_surface.comp, minimum doux des deux
//    rives), avec la direction d'écoulement. Marais au niveau waterTime.y. Données dans un pool
//    (SSBO_Dev2Scratch, lié par passe) de kRiverSlots emplacements, attribués aux chunks proches.
//  * Shading partagé (water_surface.frag) : même BRDF, ombres, ambiant et brouillard que le sol
//    (surface_lighting.glsl, réglages de LightingShared.hpp), réfraction d'une copie de scene.color,
//    Beer-Lambert, diffusion, SSR + cône VXGI + ciel.
//
//  Cartes de l'océan NON publiées sous water.oceanDisplacement / water.oceanNormal : le contrat les
//  déclare Texture2D alors que les cascades forment une 2D array. Les publier demanderait un ADR
//  (signalé) ; aucun consommateur n'existe aujourd'hui. Les slots TEX_OceanDisplacement et IMG_WaterOut
//  du contrat sont utilisés.
//
//  Ne dessine rien si scene.color n'est pas HDR cette frame (tonemap inactif : l'éclairage a déjà encodé
//  l'image).
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
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "contracts/ISystemModule.hpp"
#include "contracts/WorldContracts.hpp"
#include "contracts/GpuBindings.hpp"
#include "core/SystemRegistry.hpp"
#include "core/RenderBlackboard.hpp"
#include "core/gpu/ShaderManager.hpp"
#include "core/gpu/GpuBuffer.hpp"
#include "core/gpu/Texture2D.hpp"
#include "render_engine/LightingShared.hpp"

namespace env::render {
namespace {

constexpr int           kCascades     = 3;
constexpr int           kRiverVerts   = 129;  ///< sommets par côté de la grille d'un chunk (2 m)
constexpr int           kRiverSlots   = 81;   ///< chunks porteurs d'eau simultanés (9 × 9)
constexpr GLsizeiptr    kRiverSlotBytes = static_cast<GLsizeiptr>(kRiverVerts) * kRiverVerts * 16;
constexpr int           kRiverJobsPerFrame = 2;
constexpr float         kPi = 3.14159265358979f;

/// Miroir std140 de WaterParamsBlock (shaders/render/ocean_common.glsl), UBO_WaterParams.
struct WaterParamsUbo {
    glm::ivec4 oceanInfo{0};
    glm::vec4  cascadeSize{0.0f};
    glm::vec4  cutLow{0.0f};
    glm::vec4  cutHigh{0.0f};
    glm::vec4  oceanMisc{0.0f};
    glm::vec4  absorption{0.0f};
    glm::vec4  scattering{0.0f};
    glm::vec4  riverParams{0.0f};
    glm::vec4  swampColor{0.0f};
    glm::vec4  originModX{0.0f};
    glm::vec4  originModZ{0.0f};
    glm::vec4  slopeVariance{0.0f};
};
static_assert(sizeof(WaterParamsUbo) == 12 * 16);

/// Variance des pentes ∫ k² S(k) dk de la bande [kLo, kHi) du spectre JONSWAP (omnidirectionnel, même
/// modèle que ocean_spectrum.comp, eau profonde), × 2 (h(k,t) combine h0(k) et h0(-k)).
float SlopeVariance(float windSpeed, float fetch, float amplitude, float kLo, float kHi) {
    constexpr float g = 9.81f;
    const float U = std::max(windSpeed, 0.5f), F = std::max(fetch, 100.0f);
    const float alpha = 0.076f * std::pow(U * U / (F * g), 0.22f);
    const float wp    = 22.0f * std::pow(g * g / (U * F), 1.0f / 3.0f);
    const float lo = std::max(kLo, 1e-3f), hi = std::min(kHi, 2000.0f);
    if (hi <= lo) return 0.0f;
    constexpr int n = 256;
    const float   r = std::log(hi / lo) / n;
    double        sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const float k  = lo * std::exp((static_cast<float>(i) + 0.5f) * r);
        const float dk = k * r;
        const float w  = std::sqrt(g * k);
        const float sigma = w <= wp ? 0.07f : 0.09f;
        const float rr = std::exp(-(w - wp) * (w - wp) / (2.0f * sigma * sigma * wp * wp));
        const float S  = alpha * g * g / std::pow(w, 5.0f) * std::exp(-1.25f * std::pow(wp / w, 4.0f)) * std::pow(3.3f, rr);
        const float Sk = S * g / (2.0f * w) * std::exp(-k * k * 1e-4f);
        sum += static_cast<double>(k * k * Sk * dk);
    }
    return static_cast<float>(2.0 * sum) * amplitude * amplitude;
}

/// Texture 2D array possédée (aucun wrapper du core ne gère les arrays ; le core est gelé).
class TextureArray2D {
public:
    TextureArray2D() = default;
    ~TextureArray2D() { Destroy(); }
    TextureArray2D(const TextureArray2D&)            = delete;
    TextureArray2D& operator=(const TextureArray2D&) = delete;

    void Create(GLsizei size, GLsizei layers, GLenum format, const char* name) {
        Destroy();
        levels_ = gpu::MipLevelCount(size, size);
        glCreateTextures(GL_TEXTURE_2D_ARRAY, 1, &id_);
        glTextureStorage3D(id_, levels_, format, size, size, layers);
        glTextureParameteri(id_, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTextureParameteri(id_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(id_, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTextureParameteri(id_, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTextureParameterf(id_, GL_TEXTURE_MAX_ANISOTROPY, 8.0f);
        glObjectLabel(GL_TEXTURE, id_, -1, name);
        const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        glClearTexImage(id_, 0, GL_RGBA, GL_FLOAT, zero);
        format_ = format;
    }
    void Destroy() noexcept {
        if (id_ != 0) glDeleteTextures(1, &id_);
        id_ = 0;
    }
    GLuint Id() const noexcept { return id_; }
    GLenum Format() const noexcept { return format_; }

private:
    GLuint  id_     = 0;
    GLsizei levels_ = 1;
    GLenum  format_ = GL_RGBA16F;
};

struct Grid {
    gpu::GpuBuffer   vertices;
    gpu::GpuBuffer   indices;
    gpu::VertexArray vao;
    GLsizei          indexCount = 0;
};

/// Grille cx × cy cellules ; sommet = (i, j) * scale (vec2).
void BuildGrid(Grid& grid, int cx, int cy, glm::vec2 scale, const char* name) {
    std::vector<glm::vec2>     v;
    std::vector<std::uint32_t> idx;
    v.reserve(static_cast<std::size_t>((cx + 1) * (cy + 1)));
    for (int j = 0; j <= cy; ++j)
        for (int i = 0; i <= cx; ++i) v.emplace_back(static_cast<float>(i) * scale.x, static_cast<float>(j) * scale.y);
    auto at = [cx](int i, int j) { return static_cast<std::uint32_t>(j * (cx + 1) + i); };
    for (int j = 0; j < cy; ++j)
        for (int i = 0; i < cx; ++i) {
            const std::uint32_t a = at(i, j), b = at(i + 1, j), c = at(i, j + 1), d = at(i + 1, j + 1);
            idx.insert(idx.end(), {a, c, b, b, c, d});
        }
    grid.vertices.CreateFrom(std::span<const glm::vec2>(v), 0, name);
    grid.indices.CreateFrom(std::span<const std::uint32_t>(idx), 0, name);
    grid.vao.Create(name);
    grid.vao.SetVertexBuffer(0, grid.vertices, 0, sizeof(glm::vec2));
    grid.vao.SetElementBuffer(grid.indices);
    grid.vao.SetAttribute(0, 0, 2, GL_FLOAT, 0);
    grid.indexCount = static_cast<GLsizei>(idx.size());
}

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

class WaterOceanRiverSystem final : public ISystemModule {
public:
    void Init(EngineContext& ctx) override {
        spectrumProg_   = &ctx.shaders.LoadCompute("render.ocean_spectrum", "render/ocean_spectrum.comp");
        fftProg_        = &ctx.shaders.LoadCompute("render.ocean_fft", "render/ocean_fft.comp");
        mipProg_        = &ctx.shaders.LoadCompute("render.ocean_mip", "render/ocean_mip.comp");
        riverBakeProg_  = &ctx.shaders.LoadCompute("render.river_surface_bake", "render/river_surface.comp");
        depthProg_      = &ctx.shaders.LoadGraphics("render.water_depth", "render/deferred_lighting.vert",
                                                    "render/water_depth.frag");
        oceanProg_      = &ctx.shaders.LoadGraphics("render.ocean_surface", "render/ocean_surface.vert",
                                                    "render/water_surface.frag");
        riverProg_      = &ctx.shaders.LoadGraphics("render.river_surface", "render/river_surface.vert",
                                                    "render/water_surface.frag");

        ubo_.Create(sizeof(WaterParamsUbo), GL_DYNAMIC_STORAGE_BIT, nullptr, "water.params");
        emptyVao_.Create("water.emptyVao");
        BuildGrid(oceanGrid_, kOceanGridX, kOceanGridY,
                  glm::vec2(1.0f / kOceanGridX, 1.0f / kOceanGridY), "water.oceanGrid");
        BuildGrid(riverGrid_, kRiverVerts - 1, kRiverVerts - 1, glm::vec2(1.0f), "water.riverGrid");
        riverPool_.Create(kRiverSlotBytes * (kRiverSlots + 1), GL_DYNAMIC_STORAGE_BIT, nullptr, "water.riverPool");
        freeSlots_.clear();
        for (int i = kRiverSlots - 1; i >= 0; --i) freeSlots_.push_back(i);
    }

    void Update(EngineContext& ctx) override {
        // --- Océan : (ré)allocation selon la résolution FFT, spectre à refaire si le vent change ---
        const int n = kFftSizes[std::clamp(fftSizeIndex_, 0, 2)];
        if (n != allocatedN_) {
            allocatedN_ = n;
            needsAlloc_ = true;
            spectrumDirty_ = true;
        }
        const glm::vec2 wind  = glm::vec2(ctx.environment.windDirSpeed);
        const float     speed = ctx.environment.windDirSpeed.z;
        if (std::abs(speed - spectrumWindSpeed_) > 0.05f * std::max(1.0f, spectrumWindSpeed_) ||
            glm::dot(wind, spectrumWindDir_) < 0.995f)
            spectrumDirty_ = true;

        // --- Copie de scene.color pour la réfraction (taille du viewport) ---
        const glm::ivec2 vp = glm::max(ctx.viewportSize, glm::ivec2(1));
        if (vp != sceneCopySize_) {
            gpu::Texture2D::Desc d;
            d.width          = vp.x;
            d.height         = vp.y;
            d.internalFormat = GL_RGBA16F;
            d.debugName      = "water.sceneCopy";
            sceneCopy_.Create(d);
            sceneCopySize_ = vp;
        }

        // --- Rivières / marais : attribution des emplacements du pool aux chunks proches ---
        UpdateRiverPool(ctx);

        // Coordonnées des cascades stables au recalage de l'origine (double précision).
        const double ox = static_cast<double>(ctx.renderOriginChunk.x) * kChunkSizeMeters;
        const double oz = static_cast<double>(ctx.renderOriginChunk.y) * kChunkSizeMeters;
        for (int c = 0; c < kCascades; ++c) {
            const double L = cascadeSizes_[c];
            originMod_[0][c] = static_cast<float>(ox - std::floor(ox / L) * L);
            originMod_[1][c] = static_cast<float>(oz - std::floor(oz / L) * L);
        }
    }

    void Render(EngineContext& ctx) override {
        const std::uint64_t frame = ctx.time.frameIndex;
        if (!SceneColorIsHdr(frame)) return;  // repli LDR : pas d'eau sur une image déjà exposée
        const bool wantOcean  = oceanEnabled_;
        const bool wantRivers = (riversEnabled_ || swampsEnabled_) && !riverDraws_.empty();
        if (!wantOcean && !wantRivers && riverJobs_.empty()) return;

        const core::RenderBlackboard& bb = ctx.blackboard;
        UploadParams();
        ubo_.BindBase(GL_UNIFORM_BUFFER, env::bind::ubo::WaterParams);

        // --- 1. FFT de l'océan (aussi source des vaguelettes des rivières) ---
        if (needsAlloc_) AllocateOcean();
        bool mapsReady = RunOceanFft(ctx);

        // --- 2. Surfaces de rivière des chunks nouveaux / régénérés ---
        BakeRiverChunks(ctx);
        if (!mapsReady) return;

        // --- 3. Copie de scene.color (réfraction, SSR) et profondeur opaque dans scene.depth ---
        const core::TextureRef color = bb.GetTexture(env::bb::SceneColor);
        if (color.size.x != sceneCopySize_.x || color.size.y != sceneCopySize_.y) return;  // redimensionnement en cours
        glCopyImageSubData(color.id, GL_TEXTURE_2D, 0, 0, 0, 0, sceneCopy_.Id(), GL_TEXTURE_2D, 0, 0, 0, 0,
                           sceneCopySize_.x, sceneCopySize_.y, 1);
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

        BindSurfaceLightingResources(bb);
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_ALWAYS);
        glDepthMask(GL_TRUE);
        if (depthProg_->IsValid() && SharedLighting().gbufferAvailable) {
            depthProg_->Use();
            emptyVao_.Bind();
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDepthFunc(GL_GREATER);  // Reverse-Z

        // --- 4. Surfaces : ressources communes ---
        glBindTextureUnit(env::bind::tex::OceanDisplacement, oceanMaps_.Id());
        sceneCopy_.BindUnit(env::bind::tex::SceneColor);
        glDisable(GL_CULL_FACE);  // vagues vues de dos aux angles rasants

        if (wantOcean && oceanProg_->IsValid()) {
            const gpu::ShaderProgram& p = *oceanProg_;
            SetSurfaceLightingUniforms(p);
            p.Set("uKind", 0);
            p.Set("uGridMargin", 0.15f);
            p.Set("uGridCells", glm::vec2(kOceanGridX, kOceanGridY));
            p.Set("uMaxDistance", oceanMaxDistance_);
            p.Use();
            oceanGrid_.vao.Bind();
            glDrawElements(GL_TRIANGLES, oceanGrid_.indexCount, GL_UNSIGNED_INT, nullptr);
        }

        if (wantRivers && riverProg_->IsValid()) {
            const gpu::ShaderProgram& p = *riverProg_;
            SetSurfaceLightingUniforms(p);
            p.Set("uKind", 1);
            p.Set("uRivers", riversEnabled_);
            p.Set("uSwamps", swampsEnabled_);
            riverPool_.BindBase(GL_SHADER_STORAGE_BUFFER, env::bind::ssbo::Dev2Scratch);
            // Pas d'écriture de profondeur : early_fragment_tests l'écrirait aussi pour les fragments rejetés
            // (hors de l'eau). Rivières et marais sont peu profonds : scene.depth garde leur fond.
            glDepthMask(GL_FALSE);
            p.Use();
            riverGrid_.vao.Bind();
            for (const RiverDraw& d : riverDraws_) {
                const auto it = riverChunks_.find(d.entity);
                if (it == riverChunks_.end() || !it->second.baked) continue;
                p.Set("uSlotBase", static_cast<std::uint32_t>(it->second.slot * kRiverVerts * kRiverVerts));
                p.Set("uChunkOffset", d.offset);
                glDrawElements(GL_TRIANGLES, riverGrid_.indexCount, GL_UNSIGNED_INT, nullptr);
            }
            glDepthMask(GL_TRUE);
        }

        glBindVertexArray(0);
    }

    void Shutdown(EngineContext&) override {
        oceanMaps_.Destroy();
        spectrum_.Destroy();
        riverPool_.Destroy();
        ubo_.Destroy();
        sceneCopy_.Destroy();
        emptyVao_.Destroy();
        oceanGrid_ = Grid{};
        riverGrid_ = Grid{};
        riverChunks_.clear();
        spectrumProg_ = fftProg_ = mipProg_ = riverBakeProg_ = depthProg_ = oceanProg_ = riverProg_ = nullptr;
    }

    void OnGui(EngineContext&) override {
        const bool hdr = SharedLighting().hdrFrame != ~std::uint64_t{0};
        if (!hdr) ImGui::TextColored(ImVec4(0.95f, 0.7f, 0.3f, 1.0f), "[--] tonemap inactif : eau non dessinee");
        ImGui::Text("Chunks d'eau : %d dessines, %d en memoire / %d", static_cast<int>(riverDraws_.size()),
                    static_cast<int>(riverChunks_.size()), kRiverSlots);

        ImGui::SeparatorText("Ocean FFT");
        ImGui::Checkbox("Ocean", &oceanEnabled_);
        ImGui::TextDisabled("Niveau : waterTime.x (BiomeHydrology) | vent : panneau Celestial");
        static const char* kSizes[] = {"128", "256", "512"};
        ImGui::Combo("Resolution FFT", &fftSizeIndex_, kSizes, IM_ARRAYSIZE(kSizes));
        if (ImGui::DragFloat3("Cascades (m)", cascadeSizes_, 1.0f, 1.0f, 4096.0f, "%.0f")) spectrumDirty_ = true;
        if (ImGui::SliderFloat("Amplitude", &amplitude_, 0.0f, 10.0f)) spectrumDirty_ = true;
        if (ImGui::SliderFloat("Fetch", &fetchKm_, 1.0f, 1000.0f, "%.0f km", ImGuiSliderFlags_Logarithmic))
            spectrumDirty_ = true;
        ImGui::SliderFloat("Choppiness", &choppiness_, 0.0f, 3.0f);
        ImGui::SliderFloat("Seuil d'ecume", &foamThreshold_, 0.0f, 1.0f);
        ImGui::SetItemTooltip("Jacobien sous lequel la surface se replie et produit de l'ecume.");
        ImGui::SliderFloat("Decroissance d'ecume", &foamDecay_, 0.0f, 0.99f);
        ImGui::SetItemTooltip("Part de l'ecume qui subsiste apres 1 s.");
        ImGui::SliderFloat("Distance max", &oceanMaxDistance_, 1000.0f, 100000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);

        ImGui::SeparatorText("Rivieres / marais");
        ImGui::Checkbox("Rivieres", &riversEnabled_);
        ImGui::SameLine();
        ImGui::Checkbox("Marais", &swampsEnabled_);
        ImGui::TextDisabled("Niveau du marais : waterTime.y (BiomeHydrology)");
        ImGui::SliderFloat("Vitesse d'ecoulement", &flowSpeed_, 0.0f, 5.0f, "%.2f m/s");
        ImGui::SliderFloat("Force des normales", &normalStrength_, 0.0f, 2.0f);
        ImGui::SliderFloat("Douceur des berges", &bankSoftness_, 0.1f, 10.0f, "%.1f m");
        ImGui::SliderFloat("Turbidite du marais", &murkiness_, 0.0f, 1.0f);
        ImGui::SliderFloat("Distance des rivieres", &riverDistance_, 256.0f, 2048.0f, "%.0f m");
        if (ImGui::SmallButton("Recalculer les surfaces")) {
            for (auto& [e, rc] : riverChunks_) rc.baked = false;
        }

        ImGui::SeparatorText("Shading");
        ImGui::ColorEdit3("Absorption (1/m)", absorption_, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
        ImGui::ColorEdit3("Diffusion (1/m)", scattering_, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
        ImGui::SliderFloat("Refraction", &refractionStrength_, 0.0f, 1.0f);
        ImGui::Checkbox("Reflexions ecran (SSR)", &ssr_);
    }

private:
    struct RiverChunk {
        int           slot       = -1;
        bool          baked      = false;
        std::uint32_t generation = 0;
        GLuint        heightTex  = 0;
        GLuint        biomeTex   = 0;
        glm::ivec2    coord{0};
        float         heightMin  = 0.0f, heightMax = 0.0f;
    };
    struct RiverDraw {
        entt::entity entity = entt::null;
        glm::vec3    offset{0.0f};
    };

    static constexpr int kFftSizes[3] = {128, 256, 512};
    static constexpr int kOceanGridX  = 240;
    static constexpr int kOceanGridY  = 160;

    void UploadParams() {
        WaterParamsUbo u;
        const int n = allocatedN_;
        u.oceanInfo   = glm::ivec4(n, static_cast<int>(std::lround(std::log2(n))), kCascades, 0);
        u.cascadeSize = glm::vec4(cascadeSizes_[0], cascadeSizes_[1], cascadeSizes_[2], 0.0f);
        // Bandes de nombres d'onde : chaque cascade s'arrête où la suivante (plus fine) prend le relais,
        // à 6 fois son pas fréquentiel (assez d'échantillons dans la bande).
        for (int c = 0; c < kCascades; ++c) {
            const float lo = c == 0 ? 0.0f : 2.0f * kPi / cascadeSizes_[c] * 6.0f;
            const float hi = c == kCascades - 1 ? 1.0e9f : 2.0f * kPi / cascadeSizes_[c + 1] * 6.0f;
            u.cutLow[c]  = lo;
            u.cutHigh[c] = hi;
        }
        // Décroissance (part restante après 1 s) -> demi-vie.
        const float keep = std::clamp(foamDecay_, 0.01f, 0.99f);
        u.oceanMisc   = glm::vec4(choppiness_, foamThreshold_, std::log(0.5f) / std::log(keep), 0.0f);
        u.absorption  = glm::vec4(absorption_[0], absorption_[1], absorption_[2], refractionStrength_);
        u.scattering  = glm::vec4(scattering_[0], scattering_[1], scattering_[2], murkiness_);
        u.riverParams = glm::vec4(flowSpeed_, normalStrength_, bankSoftness_, ssr_ ? 1.0f : 0.0f);
        u.swampColor  = glm::vec4(0.9f, 1.1f, 1.8f, 0.6f);  // tanins / algues : le bleu est absorbé, l'eau brunit
        u.originModX  = glm::vec4(originMod_[0][0], originMod_[0][1], originMod_[0][2], 0.0f);
        u.originModZ  = glm::vec4(originMod_[1][0], originMod_[1][1], originMod_[1][2], 0.0f);
        for (int c = 0; c < kCascades; ++c)
            u.slopeVariance[c] = SlopeVariance(spectrumWindSpeed_, fetchKm_ * 1000.0f, amplitude_, u.cutLow[c], u.cutHigh[c]);
        ubo_.UploadObject(u);
    }

    void AllocateOcean() {
        const int n = allocatedN_;
        oceanMaps_.Create(n, 2 * kCascades, GL_RGBA16F, "water.oceanMaps");
        // h0 (3 N²) + résultat des lignes (2 × 3 N²), vec4.
        spectrum_.Create(static_cast<GLsizeiptr>(n) * n * kCascades * 3 * 16, GL_DYNAMIC_STORAGE_BIT, nullptr,
                         "water.oceanSpectrum");
        needsAlloc_ = false;
    }

    bool RunOceanFft(EngineContext& ctx) {
        if (!spectrumProg_->IsValid() || !fftProg_->IsValid() || oceanMaps_.Id() == 0) return false;
        const int n = allocatedN_;
        spectrum_.BindBase(GL_SHADER_STORAGE_BUFFER, env::bind::ssbo::OceanSpectrum);

        if (spectrumDirty_) {
            const glm::vec2 wind = glm::vec2(ctx.environment.windDirSpeed);
            spectrumWindDir_     = glm::length(wind) > 1e-4f ? glm::normalize(wind) : glm::vec2(1.0f, 0.0f);
            spectrumWindSpeed_   = ctx.environment.windDirSpeed.z;
            const gpu::ShaderProgram& p = *spectrumProg_;
            p.Set("uWindDir", spectrumWindDir_);
            p.Set("uWindSpeed", spectrumWindSpeed_);
            p.Set("uFetch", fetchKm_ * 1000.0f);
            p.Set("uAmplitude", amplitude_);
            p.Use();
            glDispatchCompute(static_cast<GLuint>(n / 16), static_cast<GLuint>(n / 16), kCascades);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            spectrumDirty_ = false;
        }

        const gpu::ShaderProgram& f = *fftProg_;
        f.Set("uTime", static_cast<float>(ctx.time.time));
        f.Set("uDt", ctx.time.dt);
        glBindImageTexture(env::bind::image::WaterOut, oceanMaps_.Id(), 0, GL_TRUE, 0, GL_READ_WRITE, GL_RGBA16F);
        f.Use();
        f.Set("uPass", 0);
        glDispatchCompute(static_cast<GLuint>(n), kCascades, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        f.Set("uPass", 1);
        glDispatchCompute(static_cast<GLuint>(n), kCascades, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
        GenerateOceanMips();
        glBindImageTexture(env::bind::image::WaterOut, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
        return true;
    }

    /// Mips par compute (ocean_mip.comp) : glGenerateTextureMipmap coûte ~0,19 ms sur cette 2D array.
    void GenerateOceanMips() {
        if (!mipProg_->IsValid()) return;
        mipProg_->Use();
        glBindTextureUnit(env::bind::tex::OceanDisplacement, oceanMaps_.Id());
        const int levels = gpu::MipLevelCount(allocatedN_, allocatedN_);
        for (int level = 1; level < levels; ++level) {
            const int size = std::max(allocatedN_ >> level, 1);
            glBindImageTexture(env::bind::image::WaterOut, oceanMaps_.Id(), level, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);
            mipProg_->Set("uSrcLevel", level - 1);
            glDispatchCompute(static_cast<GLuint>((size + 7) / 8), static_cast<GLuint>((size + 7) / 8), 2 * kCascades);
            glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Rivières : pool d'emplacements (CPU, Update) et précalcul (GPU, Render)
    // ---------------------------------------------------------------------------------------------
    void UpdateRiverPool(EngineContext& ctx) {
        riverDraws_.clear();
        riverJobs_.clear();
        const glm::vec3 cam    = glm::vec3(ctx.environment.cameraPosition);
        const auto      planes = FrustumPlanes(ctx.environment.viewProj);

        struct Candidate {
            entt::entity e;
            float        dist;
            glm::vec3    offset;
        };
        std::vector<Candidate> wanted;
        auto view = ctx.registry.view<const ChunkCoordComponent, const ChunkStateComponent, const ChunkGpuResources>();
        for (const entt::entity e : view) {
            const auto& [coord, state, res] = view.get(e);
            const bool ready = state.state != ChunkState::Evicting && (state.readyMask & ChunkReady::HeightNormal) &&
                               (state.readyMask & ChunkReady::BiomeHydro) && res.heightNormalTex != 0 &&
                               res.biomeHydroTex != 0;
            if (!ready) continue;
            const glm::vec3 offset = ChunkOriginRelative(coord.coord, ctx.renderOriginChunk);
            const glm::vec2 center = glm::vec2(offset.x, offset.z) + 0.5f * kChunkSizeMeters;
            const float     dist   = std::max(0.0f, glm::length(center - glm::vec2(cam.x, cam.z)) - 0.7071f * kChunkSizeMeters);
            if (dist > riverDistance_) continue;
            wanted.push_back({e, dist, offset});
        }
        std::sort(wanted.begin(), wanted.end(), [](const Candidate& a, const Candidate& b) { return a.dist < b.dist; });
        if (wanted.size() > static_cast<std::size_t>(kRiverSlots)) wanted.resize(kRiverSlots);

        // Libère les emplacements des chunks disparus ou trop loin.
        for (auto it = riverChunks_.begin(); it != riverChunks_.end();) {
            const bool keep = std::any_of(wanted.begin(), wanted.end(), [&](const Candidate& c) { return c.e == it->first; });
            if (!keep) {
                freeSlots_.push_back(it->second.slot);
                it = riverChunks_.erase(it);
            } else {
                ++it;
            }
        }

        for (const Candidate& c : wanted) {
            const auto& state = ctx.registry.get<ChunkStateComponent>(c.e);
            const auto& res   = ctx.registry.get<ChunkGpuResources>(c.e);
            auto it = riverChunks_.find(c.e);
            if (it == riverChunks_.end()) {
                if (freeSlots_.empty()) continue;
                RiverChunk rc;
                rc.slot = freeSlots_.back();
                freeSlots_.pop_back();
                it = riverChunks_.emplace(c.e, rc).first;
            }
            RiverChunk& rc = it->second;
            if (rc.generation != state.generation || rc.heightTex != res.heightNormalTex || rc.biomeTex != res.biomeHydroTex)
                rc.baked = false;
            rc.generation = state.generation;
            rc.heightTex  = res.heightNormalTex;
            rc.biomeTex   = res.biomeHydroTex;
            rc.heightMin  = res.heightRange.x;
            rc.heightMax  = res.heightRange.y;
            if (!rc.baked) {
                if (riverJobs_.size() < static_cast<std::size_t>(kRiverJobsPerFrame)) riverJobs_.push_back(c.e);
                continue;
            }
            const bool  known = rc.heightMin != 0.0f || rc.heightMax != 0.0f;
            const float hMin  = known ? rc.heightMin - 12.0f : -500.0f;
            const float hMax  = known ? rc.heightMax + 2.0f : 3000.0f;
            const glm::vec3 mn = c.offset + glm::vec3(0.0f, std::min(hMin, ctx.environment.waterTime.y), 0.0f);
            const glm::vec3 mx = c.offset + glm::vec3(kChunkSizeMeters, std::max(hMax, ctx.environment.waterTime.y + 1.0f),
                                                      kChunkSizeMeters);
            if (AabbInFrustum(planes, mn, mx)) riverDraws_.push_back({c.e, c.offset});
        }
    }

    void BakeRiverChunks(EngineContext& ctx) {
        if (riverJobs_.empty() || !riverBakeProg_->IsValid()) return;
        const gpu::ShaderProgram& p = *riverBakeProg_;
        riverPool_.BindBase(GL_SHADER_STORAGE_BUFFER, env::bind::ssbo::Dev2Scratch);
        p.Use();
        const GLuint groups  = static_cast<GLuint>((kRiverVerts + 7) / 8);
        const auto   tmpBase = static_cast<std::uint32_t>(kRiverSlots * kRiverVerts * kRiverVerts);
        for (const entt::entity e : riverJobs_) {
            auto it = riverChunks_.find(e);
            if (it == riverChunks_.end() || !ctx.registry.valid(e)) continue;
            RiverChunk& rc = it->second;
            // Lecture seule des textures du chunk (slots Dev 1 existants).
            glBindTextureUnit(env::bind::tex::HeightNormal, rc.heightTex);
            glBindTextureUnit(env::bind::tex::BiomeHydro, rc.biomeTex);
            p.Set("uPass", 0);
            p.Set("uDstBase", tmpBase);
            glDispatchCompute(groups, groups, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            p.Set("uPass", 1);
            p.Set("uSrcBase", tmpBase);
            p.Set("uDstBase", static_cast<std::uint32_t>(rc.slot * kRiverVerts * kRiverVerts));
            glDispatchCompute(groups, groups, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            rc.baked = true;
        }
        riverJobs_.clear();
    }

    // Programmes
    gpu::ShaderProgram* spectrumProg_  = nullptr;
    gpu::ShaderProgram* fftProg_       = nullptr;
    gpu::ShaderProgram* mipProg_       = nullptr;
    gpu::ShaderProgram* riverBakeProg_ = nullptr;
    gpu::ShaderProgram* depthProg_     = nullptr;
    gpu::ShaderProgram* oceanProg_     = nullptr;
    gpu::ShaderProgram* riverProg_     = nullptr;

    // Ressources
    gpu::GpuBuffer   ubo_;
    gpu::GpuBuffer   spectrum_;
    gpu::GpuBuffer   riverPool_;
    TextureArray2D   oceanMaps_;
    gpu::Texture2D   sceneCopy_;
    glm::ivec2       sceneCopySize_{0};
    gpu::VertexArray emptyVao_;
    Grid             oceanGrid_;
    Grid             riverGrid_;

    // État
    int       allocatedN_        = 0;
    bool      needsAlloc_        = true;
    bool      spectrumDirty_     = true;
    glm::vec2 spectrumWindDir_{1.0f, 0.0f};
    float     spectrumWindSpeed_ = -1.0f;
    float     originMod_[2][kCascades] = {};
    std::unordered_map<entt::entity, RiverChunk> riverChunks_;
    std::vector<int>          freeSlots_;
    std::vector<entt::entity> riverJobs_;
    std::vector<RiverDraw>    riverDraws_;

    // Paramètres (OnGui)
    bool  oceanEnabled_       = true;
    int   fftSizeIndex_       = 1;  // 256
    float cascadeSizes_[3]    = {250.0f, 37.0f, 7.0f};
    float amplitude_          = 1.0f;
    float fetchKm_            = 100.0f;
    float choppiness_         = 1.2f;
    float foamThreshold_      = 0.35f;
    float foamDecay_          = 0.6f;
    float oceanMaxDistance_   = 40000.0f;
    bool  riversEnabled_      = true;
    bool  swampsEnabled_      = true;
    float flowSpeed_          = 1.2f;
    float normalStrength_     = 0.8f;
    float bankSoftness_       = 1.5f;
    float murkiness_          = 0.7f;
    float riverDistance_      = 1024.0f;
    // Eau claire (ordre de grandeur de l'eau de mer du large) : le rouge est absorbé en quelques mètres,
    // la diffusion favorise le bleu -> bleu profond au large, transparence turquoise sur les hauts-fonds.
    float absorption_[3]      = {0.45f, 0.06f, 0.02f};
    float scattering_[3]      = {0.002f, 0.008f, 0.02f};
    float refractionStrength_ = 0.3f;
    bool  ssr_                = true;
};

}  // namespace
REGISTER_SYSTEM_ORDERED(WaterOceanRiverSystem, PassStage::LightingWater, 10);
}  // namespace env::render
