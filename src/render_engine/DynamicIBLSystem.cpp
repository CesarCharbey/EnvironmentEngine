// =============================================================================
//  src/render_engine/DynamicIBLSystem.cpp                           [DEV 2]
//  Stage : AtmosphereIBL (ordre 10, après CelestialAtmosphereSystem)
//
//  Rôle : éclairage par image dynamique (split-sum, Karis 2013) capturé depuis le ciel de
//  CelestialAtmosphereSystem. Produit ibl.environment, ibl.irradiance, ibl.prefiltered et ibl.brdfLUT,
//  lus par DeferredLightingSystem (reflets du ciel, ambiant diffuse hors de la grille VXGI) et, plus
//  tard, par l'eau et les particules.
//
//  * Time-slicing (ADR-004) : un cycle = capture (6 faces en un dispatch) -> mips de l'environnement
//    -> irradiance -> préfiltrage GGX, un mip par étape ; au plus `stepsPerFrame_` étapes par frame.
//  * Trois jeux de cubemaps : deux jeux de CALCUL en double tampon (le cycle écrit dans le jeu de
//    travail ; le dernier cycle terminé devient la cible) et le jeu PUBLIÉ, qui converge chaque frame
//    vers la cible par lissage exponentiel (ibl_blend.comp, τ = `smoothingTime_`). Remplacer l'IBL d'un
//    coup en fin de cycle faisait sauter les reflets de 10 à 40 % au crépuscule (luminance du ciel
//    exponentielle) : mesuré sur le lac, maintenant continu. Le jeu publié est mis à jour EN ENTIER
//    dans ce stage, avant l'éclairage : jamais de cubemap à moitié mise à jour à l'écran.
//  * Update (CPU) : décide de démarrer un cycle (intervalle écoulé, soleil ou lune déplacés au-delà du
//    seuil, demande manuelle) ; jamais pendant un cycle, jamais quand l'IBL est gelée.
//  * Changement de résolution : seul le jeu de travail est réalloué au début d'un cycle ; le jeu publié
//    suit à la fin du cycle (copie directe, aucun retour au repli gris).
//  * LUT BRDF 512² RG16F calculée une seule fois, à la première frame.
//
//  RÈGLE : ce fichier n'inclut jamais src/world_engine/ (vérifié au build).
// =============================================================================
#include <imgui.h>
#include <glad/glad.h>
#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string_view>

#include "contracts/ISystemModule.hpp"
#include "contracts/WorldContracts.hpp"
#include "contracts/GpuBindings.hpp"
#include "core/SystemRegistry.hpp"
#include "core/RenderBlackboard.hpp"
#include "core/gpu/ShaderManager.hpp"
#include "core/gpu/Cubemap.hpp"
#include "core/gpu/Texture2D.hpp"

namespace env::render {
namespace {

constexpr std::string_view kProducer = "DynamicIBLSystem";

constexpr GLsizei kBrdfLutSize          = 512;
constexpr GLsizei kIrradianceSourceSize = 16;  // face du mip d'environnement convolué pour l'irradiance
constexpr GLuint  kLocalSize            = 8;   // local_size_x/y des shaders ibl_*.comp

constexpr std::array<int, 4> kEnvironmentSizes = {64, 128, 256, 512};
constexpr std::array<int, 4> kIrradianceSizes  = {8, 16, 32, 64};

GLuint Groups(GLsizei size) { return gpu::ShaderProgram::GroupCount(static_cast<GLuint>(size), kLocalSize); }

float AngleDeg(const glm::vec3& a, const glm::vec3& b) {
    const float la = glm::length(a), lb = glm::length(b);
    if (la <= 0.0f || lb <= 0.0f) return 0.0f;
    return glm::degrees(std::acos(std::clamp(glm::dot(a, b) / (la * lb), -1.0f, 1.0f)));
}

/// Barrière entre deux étapes : écritures image visibles pour les lectures image et par sampler.
void StepBarrier() { glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT); }

/// Un jeu complet de cubemaps (calcul ou publié).
struct IblSet {
    gpu::Cubemap environment;  // ciel capturé (+ chaîne de mips pour les jeux de calcul : source du préfiltrage)
    gpu::Cubemap irradiance;   // radiance diffuse E / π
    gpu::Cubemap prefiltered;  // mip = rugosité perceptuelle × (mips - 1)
    GLsizei      envSize   = 0;
    GLsizei      irrSize   = 0;
    GLsizei      prefMips  = 0;
};

enum class Step { Capture, EnvironmentMips, Irradiance, Prefilter };

class DynamicIBLSystem final : public ISystemModule {
public:
    void Init(EngineContext& ctx) override {
        captureProgram_    = &ctx.shaders.LoadCompute("ibl.capture",    "render/ibl_capture.comp");
        downsampleProgram_ = &ctx.shaders.LoadCompute("ibl.downsample", "render/ibl_downsample.comp");
        irradianceProgram_ = &ctx.shaders.LoadCompute("ibl.irradiance", "render/ibl_irradiance.comp");
        prefilterProgram_  = &ctx.shaders.LoadCompute("ibl.prefilter",  "render/ibl_prefilter.comp");
        brdfProgram_       = &ctx.shaders.LoadCompute("ibl.brdf",       "render/ibl_brdf.comp");
        blendProgram_      = &ctx.shaders.LoadCompute("ibl.blend",      "render/ibl_blend.comp");

        brdfLut_.Create({.width = kBrdfLutSize, .height = kBrdfLutSize, .internalFormat = GL_RG16F, .levels = 1,
                         .sampler = {}, .debugName = "ibl.brdfLUT"});
    }

    void Update(EngineContext& ctx) override {
        const core::RenderBlackboard& bb = ctx.blackboard;
        atmoAvailable_ = bb.IsPublished(env::bb::AtmoSkyViewLUT) && bb.IsPublished(env::bb::AtmoTransmittanceLUT);

        const glm::vec3 sunDir(ctx.environment.sunDirIlluminance);
        const glm::vec3 moonDir(ctx.environment.moonDirPhase);
        lastSunDelta_  = AngleDeg(sunDir, cycleSunDir_);
        lastMoonDelta_ = AngleDeg(moonDir, cycleMoonDir_);

        if (cycleActive_) return;
        timeSinceCycle_ += ctx.time.dt;
        if (!atmoAvailable_) return;  // ciel pas encore calculé : rien à capturer

        const bool manual = requestUpdate_;
        bool start = manual;
        if (!freeze_) {
            start = start || target_ < 0 || timeSinceCycle_ >= updateInterval_ ||
                    std::max(lastSunDelta_, lastMoonDelta_) > sunAngleThresholdDeg_;
        }
        if (!start) return;

        cycleActive_    = true;
        step_           = Step::Capture;
        prefilterMip_   = 0;
        cycleFrames_    = 0;
        requestUpdate_  = false;
        timeSinceCycle_ = 0.0f;
        cycleSunDir_    = sunDir;
        cycleMoonDir_   = moonDir;
    }

    void Render(EngineContext& ctx) override {
        const gpu::ShaderProgram* programs[] = {captureProgram_,   downsampleProgram_, irradianceProgram_,
                                                prefilterProgram_, brdfProgram_,      blendProgram_};
        for (const gpu::ShaderProgram* p : programs)
            if (!p || !p->IsValid()) return;  // le jeu publié (s'il existe) reste en place

        if (!brdfReady_) {
            ComputeBrdfLut();
            ctx.blackboard.PublishTexture(env::bb::IblBrdfLUT, brdfLut_, kProducer);
            brdfReady_ = true;
        }
        if (cycleActive_) {
            ++cycleFrames_;
            for (int i = 0; i < stepsPerFrame_ && cycleActive_; ++i) RunStep(ctx);
        }
        if (target_ >= 0) UpdateDisplay(ctx);  // après un éventuel fin de cycle de cette frame
    }

    void Shutdown(EngineContext& ctx) override {
        for (std::string_view key : {env::bb::IblEnvironment, env::bb::IblIrradiance, env::bb::IblPrefiltered,
                                     env::bb::IblBrdfLUT})
            ctx.blackboard.UnpublishIfProducer(key, kProducer);
        for (IblSet* set : {&sets_[0], &sets_[1], &display_}) {
            set->environment.Destroy();
            set->irradiance.Destroy();
            set->prefiltered.Destroy();
        }
        brdfLut_.Destroy();
        target_        = -1;
        displayValid_  = false;
    }

    void OnGui(EngineContext& ctx) override {
        if (cycleActive_) {
            ImGui::Text("Cycle en cours : etape %d / %d", StepIndex() + 1, TotalSteps(sets_[WorkIndex()]));
        } else if (target_ >= 0) {
            ImGui::Text("Au repos : dernier cycle en %d frame(s), il y a %.1f s", lastCycleFrames_, timeSinceCycle_);
        } else {
            ImGui::TextDisabled(atmoAvailable_ ? "En attente du premier cycle" : "En attente des LUTs d'atmosphere");
        }
        ImGui::Text("Cycles termines : %u   ecart soleil %.2f deg, lune %.2f deg", cyclesDone_, lastSunDelta_,
                    lastMoonDelta_);
        if (displayValid_) {
            ImGui::Text("Publie : environnement %d^2, irradiance %d^2, %d mips prefiltres", display_.envSize,
                        display_.irrSize, display_.prefMips);
            ImGui::Text("Lissage : %.1f %% de l'ancien ciel restant", 100.0f * blendResidual_);
        }

        ImGui::SeparatorText("Qualite");
        ComboPow2("Resolution cubemap", kEnvironmentSizes, environmentResolution_);
        ComboPow2("Resolution irradiance", kIrradianceSizes, irradianceResolution_);
        ImGui::SliderInt("Mips prefiltres", &prefilterMips_, 2, 9);
        ImGui::SetItemTooltip("Niveaux de rugosite (mip = rugosite x (mips - 1)), limites par la resolution.");
        ImGui::SliderInt("Echantillons GGX (mip 1)", &prefilterSamples_, 16, 256);
        ImGui::SetItemTooltip("Doubles a chaque mip suivant, plafond 1024.");
        ImGui::Checkbox("Sol sous l'horizon", &includeGround_);
        ImGui::SetItemTooltip("Hemisphere bas = sol (albedo de l'atmosphere eclaire) au lieu du seul air.");

        ImGui::SeparatorText("Mise a jour (time-slicing)");
        ImGui::SliderInt("Etapes / frame", &stepsPerFrame_, 1, 16);
        ImGui::SliderFloat("Intervalle de mise a jour", &updateInterval_, 0.1f, 30.0f, "%.1f s");
        ImGui::SliderFloat("Seuil angulaire soleil", &sunAngleThresholdDeg_, 0.05f, 5.0f, "%.2f deg");
        ImGui::SliderFloat("Lissage temporel", &smoothingTime_, 0.0f, 2.0f, "%.2f s");
        ImGui::SetItemTooltip("Constante de temps du fondu vers le dernier cycle (0 = remplacement direct).");
        ImGui::Checkbox("Geler", &freeze_);
        ImGui::SetItemTooltip("Plus de mise a jour automatique ; le bouton ci-dessous reste actif.");
        if (ImGui::Button("Mettre a jour maintenant")) requestUpdate_ = true;

        ImGui::SeparatorText("Inspection");
        if (ImGui::Button("Environnement")) ctx.blackboard.Inspect(env::bb::IblEnvironment);
        ImGui::SameLine();
        if (ImGui::Button("Irradiance")) ctx.blackboard.Inspect(env::bb::IblIrradiance);
        ImGui::SameLine();
        if (ImGui::Button("Prefiltree")) ctx.blackboard.Inspect(env::bb::IblPrefiltered);
        ImGui::SameLine();
        if (ImGui::Button("LUT BRDF")) ctx.blackboard.Inspect(env::bb::IblBrdfLUT);
    }

private:
    /// Jeu de calcul dans lequel écrit le cycle : l'autre que la cible (jamais lue par le jeu publié).
    int WorkIndex() const { return target_ < 0 ? 0 : target_ ^ 1; }

    /// Échantillons GGX du mip : la base au mip 1, doublée à chaque mip (lobes plus larges, beaucoup moins
    /// de texels), plafonnée à 1024.
    std::uint32_t PrefilterSampleCount(GLint mip) const {
        const int shift = std::clamp(static_cast<int>(mip) - 1, 0, 10);
        return static_cast<std::uint32_t>(std::min(prefilterSamples_ << shift, 1024));
    }

    static int TotalSteps(const IblSet& set) { return 3 + std::max<int>(set.prefMips, 1); }
    int StepIndex() const {
        switch (step_) {
            case Step::Capture:         return 0;
            case Step::EnvironmentMips: return 1;
            case Step::Irradiance:      return 2;
            case Step::Prefilter:       return 3 + prefilterMip_;
        }
        return 0;
    }

    template <std::size_t N>
    static void ComboPow2(const char* label, const std::array<int, N>& values, int& current) {
        char preview[16];
        std::snprintf(preview, sizeof(preview), "%d", current);
        if (!ImGui::BeginCombo(label, preview)) return;
        for (int v : values) {
            char item[16];
            std::snprintf(item, sizeof(item), "%d", v);
            if (ImGui::Selectable(item, v == current)) current = v;
        }
        ImGui::EndCombo();
    }

    /// (Ré)alloue le jeu si les réglages ont changé. Appelé uniquement sur le jeu de TRAVAIL, jamais publié.
    void EnsureSet(IblSet& set) {
        const GLsizei envSize  = environmentResolution_;
        const GLsizei irrSize  = irradianceResolution_;
        const GLsizei prefMips = std::clamp<GLsizei>(prefilterMips_, 1, gpu::MipLevelCount(envSize, envSize));
        if (set.environment && set.envSize == envSize && set.irrSize == irrSize && set.prefMips == prefMips) return;

        const bool       isA = &set == &sets_[0];
        gpu::SamplerDesc mipmapped{};
        mipmapped.minFilter = GL_LINEAR_MIPMAP_LINEAR;

        set.environment.Create({.size = envSize, .internalFormat = GL_RGBA16F, .levels = 0, .sampler = mipmapped,
                                .debugName = isA ? "ibl.environmentA" : "ibl.environmentB"});
        set.irradiance.Create({.size = irrSize, .internalFormat = GL_RGBA16F, .levels = 1, .sampler = {},
                               .debugName = isA ? "ibl.irradianceA" : "ibl.irradianceB"});
        set.prefiltered.Create({.size = envSize, .internalFormat = GL_RGBA16F, .levels = prefMips,
                                .sampler = mipmapped, .debugName = isA ? "ibl.prefilteredA" : "ibl.prefilteredB"});
        set.envSize  = envSize;
        set.irrSize  = irrSize;
        set.prefMips = prefMips;
    }

    void RunStep(EngineContext& ctx) {
        IblSet& set = sets_[WorkIndex()];
        switch (step_) {
            case Step::Capture: {
                EnsureSet(set);
                ctx.blackboard.BindTexture(env::bb::AtmoTransmittanceLUT, env::bind::tex::AtmoTransmittance);
                ctx.blackboard.BindTexture(env::bb::AtmoSkyViewLUT, env::bind::tex::AtmoSkyView);
                captureProgram_->Set("uAtmoAvailable", atmoAvailable_);
                captureProgram_->Set("uIncludeGround", includeGround_);
                set.environment.BindImage(env::bind::image::IblOut, GL_WRITE_ONLY, 0, GL_TRUE);
                captureProgram_->Dispatch(Groups(set.envSize), Groups(set.envSize), gpu::Cubemap::kFaceCount);
                StepBarrier();
                step_ = Step::EnvironmentMips;
                break;
            }
            case Step::EnvironmentMips: {
                set.environment.BindUnit(env::bind::tex::IblEnvironment);  // lecture du niveau N-1
                for (GLint level = 1; level < set.environment.Levels(); ++level) {
                    set.environment.BindImage(env::bind::image::IblOut, GL_WRITE_ONLY, level, GL_TRUE);
                    downsampleProgram_->Set("uSrcLevel", static_cast<int>(level - 1));
                    const GLsizei s = set.environment.FaceSize(level);
                    downsampleProgram_->Dispatch(Groups(s), Groups(s), gpu::Cubemap::kFaceCount);
                    StepBarrier();
                }
                step_ = Step::Irradiance;
                break;
            }
            case Step::Irradiance: {
                // Mip de 16² par face (ou le niveau 0 si la cubemap est plus petite).
                GLint srcLevel = 0;
                while (set.environment.FaceSize(srcLevel) > kIrradianceSourceSize &&
                       srcLevel + 1 < set.environment.Levels())
                    ++srcLevel;
                set.environment.BindUnit(env::bind::tex::IblEnvironment);
                set.irradiance.BindImage(env::bind::image::IblOut, GL_WRITE_ONLY, 0, GL_TRUE);
                irradianceProgram_->Set("uSrcLevel", static_cast<int>(srcLevel));
                irradianceProgram_->Set("uSrcSize", static_cast<int>(set.environment.FaceSize(srcLevel)));
                irradianceProgram_->Dispatch(Groups(set.irrSize), Groups(set.irrSize), gpu::Cubemap::kFaceCount);
                StepBarrier();
                step_         = Step::Prefilter;
                prefilterMip_ = 0;
                break;
            }
            case Step::Prefilter: {
                set.environment.BindUnit(env::bind::tex::IblEnvironment);
                set.prefiltered.BindImage(env::bind::image::IblOut, GL_WRITE_ONLY, prefilterMip_, GL_TRUE);
                const GLsizei s = set.prefiltered.FaceSize(prefilterMip_);
                if (prefilterMip_ == 0) {
                    // Rugosité 0 = miroir : copie exacte du niveau 0 (un échantillon au centre de chaque texel).
                    downsampleProgram_->Set("uSrcLevel", 0);
                    downsampleProgram_->Dispatch(Groups(s), Groups(s), gpu::Cubemap::kFaceCount);
                } else {
                    const float roughness = static_cast<float>(prefilterMip_) / static_cast<float>(set.prefMips - 1);
                    prefilterProgram_->Set("uRoughness", roughness);
                    prefilterProgram_->Set("uSampleCount", PrefilterSampleCount(prefilterMip_));
                    prefilterProgram_->Set("uEnvSize", static_cast<int>(set.envSize));
                    prefilterProgram_->Set("uEnvMaxLod", static_cast<float>(set.environment.Levels() - 1));
                    // K threads par texel (~32 échantillons chacun, optimum mesuré), 64 / K texels par groupe.
                    const std::uint32_t samples = PrefilterSampleCount(prefilterMip_);
                    const std::uint32_t k       = std::clamp<std::uint32_t>(std::bit_floor(samples / 32u), 1u, 64u);
                    const auto          texels  = static_cast<GLuint>(s) * static_cast<GLuint>(s) * gpu::Cubemap::kFaceCount;
                    prefilterProgram_->Set("uThreadsPerTexel", k);
                    prefilterProgram_->Dispatch(gpu::ShaderProgram::GroupCount(texels, 64u / k));
                }
                StepBarrier();
                if (++prefilterMip_ >= set.prefMips) FinishCycle(ctx);
                break;
            }
        }
    }

    /// Le jeu de travail devient la cible ; le jeu publié commence à converger vers lui (UpdateDisplay).
    void FinishCycle(EngineContext&) {
        target_        = WorkIndex();
        blendResidual_ = 1.0f;
        cycleActive_     = false;
        lastCycleFrames_ = cycleFrames_;
        ++cyclesDone_;
    }

    /// (Ré)alloue le jeu publié aux tailles de la cible. Retourne true s'il vient d'être (ré)alloué.
    bool EnsureDisplay(const IblSet& t) {
        if (display_.environment && display_.envSize == t.envSize && display_.irrSize == t.irrSize &&
            display_.prefMips == t.prefMips)
            return false;
        gpu::SamplerDesc mipmapped{};
        mipmapped.minFilter = GL_LINEAR_MIPMAP_LINEAR;
        // Environnement publié sans mips : le flou par rugosité est dans ibl.prefiltered.
        display_.environment.Create({.size = t.envSize, .internalFormat = GL_RGBA16F, .levels = 1, .sampler = {},
                                     .debugName = "ibl.environment"});
        display_.irradiance.Create({.size = t.irrSize, .internalFormat = GL_RGBA16F, .levels = 1, .sampler = {},
                                    .debugName = "ibl.irradiance"});
        display_.prefiltered.Create({.size = t.envSize, .internalFormat = GL_RGBA16F, .levels = t.prefMips,
                                     .sampler = mipmapped, .debugName = "ibl.prefiltered"});
        display_.envSize  = t.envSize;
        display_.irrSize  = t.irrSize;
        display_.prefMips = t.prefMips;
        return true;
    }

    /// Lissage exponentiel du jeu publié vers la cible, indépendant du framerate. S'arrête une fois
    /// convergé (aucun coût au repos) ; copie directe au premier cycle ou après une réallocation.
    void UpdateDisplay(EngineContext& ctx) {
        const IblSet& t     = sets_[target_];
        const bool    fresh = EnsureDisplay(t) || !displayValid_;
        if (!fresh && blendResidual_ <= 0.0f) return;

        float weight = 1.0f;
        if (!fresh && smoothingTime_ > 0.0f) {
            weight = 1.0f - std::exp(-std::max(ctx.time.dt, 0.0f) / smoothingTime_);
            if (blendResidual_ * (1.0f - weight) < 0.002f) weight = 1.0f;  // dernier pas : atterrit exactement
        }

        blendProgram_->Set("uWeight", weight);
        auto blend = [&](const gpu::Cubemap& src, const gpu::Cubemap& dst, GLint level) {
            src.BindUnit(env::bind::tex::IblEnvironment);
            dst.BindImage(env::bind::image::IblOut, GL_READ_WRITE, level, GL_TRUE);
            blendProgram_->Set("uLevel", static_cast<int>(level));
            const GLsizei s = dst.FaceSize(level);
            blendProgram_->Dispatch(Groups(s), Groups(s), gpu::Cubemap::kFaceCount);
        };
        blend(t.environment, display_.environment, 0);
        blend(t.irradiance, display_.irradiance, 0);
        for (GLint level = 0; level < display_.prefMips; ++level) blend(t.prefiltered, display_.prefiltered, level);
        StepBarrier();  // niveaux et cubemaps indépendants : une seule barrière avant les consommateurs

        blendResidual_ = weight >= 1.0f ? 0.0f : blendResidual_ * (1.0f - weight);
        if (fresh) {
            ctx.blackboard.PublishTexture(env::bb::IblEnvironment, display_.environment, kProducer);
            ctx.blackboard.PublishTexture(env::bb::IblIrradiance, display_.irradiance, kProducer);
            ctx.blackboard.PublishTexture(env::bb::IblPrefiltered, display_.prefiltered, kProducer);
            displayValid_ = true;
        }
    }

    void ComputeBrdfLut() {
        brdfLut_.BindImage(env::bind::image::IblOut, GL_WRITE_ONLY, 0);
        brdfProgram_->Dispatch(Groups(kBrdfLutSize), Groups(kBrdfLutSize));
        StepBarrier();
    }

    gpu::ShaderProgram* captureProgram_    = nullptr;
    gpu::ShaderProgram* downsampleProgram_ = nullptr;
    gpu::ShaderProgram* irradianceProgram_ = nullptr;
    gpu::ShaderProgram* prefilterProgram_  = nullptr;
    gpu::ShaderProgram* brdfProgram_       = nullptr;
    gpu::ShaderProgram* blendProgram_      = nullptr;

    std::array<IblSet, 2> sets_;                 // jeux de calcul (double tampon)
    int                   target_ = -1;          // dernier cycle terminé (-1 : aucun)
    IblSet                display_;              // jeu publié, lissé vers la cible
    bool                  displayValid_  = false;
    float                 blendResidual_ = 0.0f;  // part de l'ancien ciel restant dans le jeu publié
    gpu::Texture2D        brdfLut_;
    bool                  brdfReady_ = false;

    // Machine à états
    bool          cycleActive_     = false;
    Step          step_            = Step::Capture;
    GLint         prefilterMip_    = 0;
    int           cycleFrames_     = 0;
    int           lastCycleFrames_ = 0;
    std::uint32_t cyclesDone_      = 0;
    float         timeSinceCycle_  = 0.0f;
    glm::vec3     cycleSunDir_{0.0f, 1.0f, 0.0f};
    glm::vec3     cycleMoonDir_{0.0f, 1.0f, 0.0f};
    float         lastSunDelta_  = 0.0f;
    float         lastMoonDelta_ = 0.0f;
    bool          atmoAvailable_ = false;
    bool          requestUpdate_ = false;

    // Paramètres (OnGui)
    int   environmentResolution_ = 256;
    int   irradianceResolution_  = 32;
    int   prefilterMips_         = 6;
    int   prefilterSamples_      = 64;
    int   stepsPerFrame_         = 2;
    float updateInterval_        = 2.0f;
    float sunAngleThresholdDeg_  = 0.25f;
    float smoothingTime_         = 0.3f;
    bool  includeGround_         = true;
    bool  freeze_                = false;
};

}  // namespace
REGISTER_SYSTEM_ORDERED(DynamicIBLSystem, PassStage::AtmosphereIBL, 10);
}  // namespace env::render
