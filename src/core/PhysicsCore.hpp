#pragma once
// =============================================================================
//  src/core/PhysicsCore.hpp — Initialisation et pas de simulation Jolt (ADR-006)
//
//  * RegisterDefaultAllocator, Factory, RegisterTypes, TempAllocatorImpl,
//    JobSystemThreadPool, PhysicsSystem + couches broad-phase / objets.
//  * Pas fixe (1/60 s) avec accumulateur et nombre de sous-pas borné.
//  * Positions Jolt RELATIVES à l'origine de rendu flottante (ADR-007) :
//    le core appelle TranslateAllBodies() quand l'origine change.
//  * Les en-têtes Jolt ne sont pas inclus ici (compilation rapide) : un système
//    qui crée des bodies inclut <Jolt/Jolt.h> puis les en-têtes dont il a besoin.
// =============================================================================
#include <cstdint>
#include <memory>

#include <glm/glm.hpp>

namespace JPH {
class PhysicsSystem;
class BodyInterface;
class TempAllocator;
class JobSystem;
}  // namespace JPH

namespace env::core {

/// Couches d'objets Jolt (JPH::ObjectLayer). Collisions : voir PhysicsCore.cpp.
namespace PhysicsLayers {
inline constexpr std::uint16_t Static    = 0;  ///< terrain (heightfield), troncs, rochers promus
inline constexpr std::uint16_t Dynamic   = 1;  ///< objets simulés
inline constexpr std::uint16_t Character = 2;  ///< contrôleurs de personnage
inline constexpr std::uint16_t Sensor    = 3;  ///< déclencheurs
inline constexpr std::uint16_t Count     = 4;
}  // namespace PhysicsLayers

/// Couches broad-phase (JPH::BroadPhaseLayer).
namespace PhysicsBroadPhase {
inline constexpr std::uint8_t NonMoving = 0;
inline constexpr std::uint8_t Moving    = 1;
inline constexpr std::uint8_t Count     = 2;
}  // namespace PhysicsBroadPhase

class PhysicsCore {
public:
    struct Settings {
        std::uint32_t maxBodies             = 16384;
        std::uint32_t maxBodyPairs          = 16384;
        std::uint32_t maxContactConstraints = 8192;
        std::uint32_t tempAllocatorBytes    = 16u * 1024u * 1024u;
        float         fixedTimeStep         = 1.0f / 60.0f;
        int           maxSubSteps           = 4;
        int           collisionSteps        = 1;
        int           workerThreads         = -1;  ///< -1 = cœurs matériels - 1
    };

    PhysicsCore();
    ~PhysicsCore();
    PhysicsCore(const PhysicsCore&)            = delete;
    PhysicsCore& operator=(const PhysicsCore&) = delete;

    bool Init();  ///< réglages par défaut
    bool Init(const Settings& settings);
    void Shutdown();

    /// Avance la simulation de dt secondes par pas fixes (accumulateur).
    void Step(float dt);

    /// Origine flottante : translate tous les bodies de `delta` (mètres).
    void TranslateAllBodies(const glm::vec3& delta);

    bool IsInitialized() const noexcept { return impl_ != nullptr; }

    JPH::PhysicsSystem& System();
    JPH::BodyInterface& Bodies();  ///< interface avec verrouillage (sûre hors callbacks Jolt)
    JPH::TempAllocator& TempAllocator();
    JPH::JobSystem&     Jobs();

    std::uint32_t BodyCount() const;
    std::uint32_t ActiveBodyCount() const;
    /// Fraction du pas fixe restant dans l'accumulateur (interpolation de rendu).
    float InterpolationAlpha() const noexcept;

    /// Widgets du panneau "Physics (Jolt)" (le core ouvre la fenêtre).
    void OnGui();

    bool paused = false;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Settings              settings_{};
    double                accumulator_   = 0.0;
    std::uint64_t         stepCount_     = 0;
    int                   stepsLastFrame_ = 0;
    float                 lastStepMs_    = 0.0f;
    int                   threadCount_   = 0;
    bool                  ownsFactory_   = false;
};

}  // namespace env::core
