#pragma once
// =============================================================================
//  include/contracts/ISystemModule.hpp
//  CONTRAT FIGÉ — Interface de tous les systèmes (Dev 1 et Dev 2).
//
//  Un système = une classe dérivant de ISystemModule dans UN .cpp de son dossier
//  métier, enregistrée par REGISTER_SYSTEM(ClassName, PassStage::X) (core/SystemRegistry.hpp).
//
//  Cycle de vie (piloté par core/Application.cpp) :
//    Init(ctx)      une fois, contexte GL et services prêts
//    Update(ctx)    chaque frame, CPU / ECS uniquement           (ordre : stage, order, nom)
//    Render(ctx)    chaque frame, TOUTES les commandes GPU       (même ordre)
//    OnGui(ctx)     chaque frame, widgets ImGui (le core ouvre la fenêtre dockée)
//    Shutdown(ctx)  une fois, contexte GL encore valide
//  Update/Render ne sont appelés que si `enabled` ; OnGui l'est toujours.
// =============================================================================
#include <cstdint>
#include <string_view>

#include <entt/entity/fwd.hpp>
#include <glm/vec2.hpp>

namespace env {

struct GlobalEnvironmentUBO;

namespace core {
class RenderBlackboard;
class PhysicsCore;
class Camera;
class SystemRegistry;
}  // namespace core

namespace gpu {
class ShaderManager;
}  // namespace gpu

/// Étapes de la frame, exécutées dans cet ordre (Update puis Render).
enum class PassStage : std::uint8_t {
    WorldStreaming = 0,  ///< Dev 1 : ensemble de chunks désiré, création / éviction d'entités
    WorldCompute,        ///< Dev 1 : génération GPU (hauteur, biomes, hydrologie, scatter)
    PhysicsSync,         ///< Dev 1 : promotion / démotion Jolt de proximité
    GBufferPass,         ///< Dev 2 : remplissage du G-Buffer
    AtmosphereIBL,       ///< Dev 2 : soleil / lune / météo, LUTs d'atmosphère, IBL time-sliced
    LightingWater,       ///< Dev 2 : résolution PBR différée, eau
    VolumetricsPost,     ///< Dev 2 : nuages, god rays, exposition / tonemap
    DebugOverlay,        ///< tous : overlays de debug
    Count
};

constexpr std::string_view PassStageName(PassStage stage) noexcept {
    switch (stage) {
        case PassStage::WorldStreaming:  return "WorldStreaming";
        case PassStage::WorldCompute:    return "WorldCompute";
        case PassStage::PhysicsSync:     return "PhysicsSync";
        case PassStage::GBufferPass:     return "GBufferPass";
        case PassStage::AtmosphereIBL:   return "AtmosphereIBL";
        case PassStage::LightingWater:   return "LightingWater";
        case PassStage::VolumetricsPost: return "VolumetricsPost";
        case PassStage::DebugOverlay:    return "DebugOverlay";
        case PassStage::Count:           break;
    }
    return "Invalid";
}

/// Stages appartenant au World Engine (Dev 1).
constexpr bool IsWorldStage(PassStage stage) noexcept {
    return stage == PassStage::WorldStreaming || stage == PassStage::WorldCompute ||
           stage == PassStage::PhysicsSync;
}

struct FrameTime {
    double        time       = 0.0;   ///< secondes depuis le lancement
    float         dt         = 0.0f;  ///< secondes depuis la frame précédente (bornée à 0.25 s)
    std::uint64_t frameIndex = 0;
};

/// Tout ce qu'un système a le droit de toucher. Construit une fois par le core.
struct EngineContext {
    entt::registry&          registry;     ///< ECS : chunks, entités physiques (WorldContracts.hpp section 5)
    core::RenderBlackboard&  blackboard;   ///< bus de textures / buffers globaux avec fallbacks
    gpu::ShaderManager&      shaders;      ///< chargement + hot-reload (F5) des programmes GLSL
    core::PhysicsCore&       physics;      ///< Jolt (ADR-006)
    const core::Camera&      camera;       ///< caméra freefly (position monde double)
    GlobalEnvironmentUBO&    environment;  ///< miroir CPU de l'UBO, uploadé après tous les Update()

    FrameTime  time{};
    glm::ivec2 viewportSize{1, 1};         ///< taille de scene.color en pixels
    glm::ivec2 renderOriginChunk{0, 0};    ///< origine de rendu flottante (ADR-007)
    bool       renderOriginShifted = false;  ///< true pendant la frame où l'origine a changé
};

class ISystemModule {
public:
    virtual ~ISystemModule() = default;

    virtual void Init(EngineContext& ctx)   = 0;
    virtual void Update(EngineContext& ctx) = 0;
    virtual void Render(EngineContext& ctx) = 0;
    virtual void OnGui(EngineContext& ctx)  = 0;
    virtual void Shutdown(EngineContext& ctx) { (void)ctx; }

    /// Activé / désactivé depuis l'UI ; désactivé automatiquement si le système lève une exception.
    bool enabled = true;

    /// Nom de la classe et stage, renseignés par SystemRegistry à l'instanciation.
    std::string_view Name() const noexcept { return name_; }
    PassStage        Stage() const noexcept { return stage_; }

private:
    friend class core::SystemRegistry;
    std::string_view name_  = "Unregistered";
    PassStage        stage_ = PassStage::DebugOverlay;
};

}  // namespace env
