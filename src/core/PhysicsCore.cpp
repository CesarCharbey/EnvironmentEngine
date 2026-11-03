// =============================================================================
//  src/core/PhysicsCore.cpp — Intégration Jolt Physics (ADR-006)
// =============================================================================
#include "core/PhysicsCore.hpp"

// Jolt.h doit être inclus en premier (configuration, macros JPH_*).
#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <thread>

#include <imgui.h>

namespace env::core {
namespace {

void JoltTrace(const char* format, ...) {
    va_list args;
    va_start(args, format);
    char buffer[1024];
    std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    std::fprintf(stderr, "[Jolt] %s\n", buffer);
}

#ifdef JPH_ENABLE_ASSERTS
bool JoltAssertFailed(const char* expression, const char* message, const char* file, JPH::uint line) {
    std::fprintf(stderr, "[Jolt] ASSERT %s:%u : (%s) %s\n", file, line, expression, message ? message : "");
    return true;  // true = point d'arrêt
}
#endif

/// Object layer -> broad-phase layer.
class BroadPhaseLayerMapping final : public JPH::BroadPhaseLayerInterface {
public:
    BroadPhaseLayerMapping() {
        mapping_[PhysicsLayers::Static]    = JPH::BroadPhaseLayer(PhysicsBroadPhase::NonMoving);
        mapping_[PhysicsLayers::Dynamic]   = JPH::BroadPhaseLayer(PhysicsBroadPhase::Moving);
        mapping_[PhysicsLayers::Character] = JPH::BroadPhaseLayer(PhysicsBroadPhase::Moving);
        mapping_[PhysicsLayers::Sensor]    = JPH::BroadPhaseLayer(PhysicsBroadPhase::Moving);
    }

    JPH::uint GetNumBroadPhaseLayers() const override { return PhysicsBroadPhase::Count; }

    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer layer) const override {
        JPH_ASSERT(layer < PhysicsLayers::Count);
        return mapping_[layer];
    }

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer layer) const override {
        switch (static_cast<JPH::BroadPhaseLayer::Type>(layer)) {
            case PhysicsBroadPhase::NonMoving: return "NonMoving";
            case PhysicsBroadPhase::Moving:    return "Moving";
            default:                           return "Invalid";
        }
    }
#endif

private:
    JPH::BroadPhaseLayer mapping_[PhysicsLayers::Count];
};

/// Object layer vs broad-phase layer : les objets statiques ne testent que le broad-phase mobile.
class ObjectVsBroadPhaseFilter final : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer layer, JPH::BroadPhaseLayer broadPhase) const override {
        if (layer == PhysicsLayers::Static) return broadPhase == JPH::BroadPhaseLayer(PhysicsBroadPhase::Moving);
        return true;
    }
};

/// Matrice de collision entre object layers.
class ObjectPairFilter final : public JPH::ObjectLayerPairFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer a, JPH::ObjectLayer b) const override {
        auto pair = [&](JPH::ObjectLayer x, JPH::ObjectLayer y) {
            return (a == x && b == y) || (a == y && b == x);
        };
        if (pair(PhysicsLayers::Static, PhysicsLayers::Static)) return false;
        if (pair(PhysicsLayers::Static, PhysicsLayers::Sensor)) return false;
        if (pair(PhysicsLayers::Sensor, PhysicsLayers::Sensor)) return false;
        return true;
    }
};

}  // namespace

struct PhysicsCore::Impl {
    BroadPhaseLayerMapping                     broadPhaseLayers;
    ObjectVsBroadPhaseFilter                   objectVsBroadPhase;
    ObjectPairFilter                           objectPairs;
    std::unique_ptr<JPH::TempAllocatorImpl>    tempAllocator;
    std::unique_ptr<JPH::JobSystemThreadPool>  jobSystem;
    std::unique_ptr<JPH::PhysicsSystem>        system;
};

PhysicsCore::PhysicsCore()  = default;
PhysicsCore::~PhysicsCore() { Shutdown(); }

bool PhysicsCore::Init() { return Init(Settings{}); }

bool PhysicsCore::Init(const Settings& settings) {
    if (impl_) return true;
    settings_ = settings;

    JPH::RegisterDefaultAllocator();
    JPH::Trace = JoltTrace;
    JPH_IF_ENABLE_ASSERTS(JPH::AssertFailed = JoltAssertFailed;)
    if (JPH::Factory::sInstance == nullptr) {
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();
        ownsFactory_ = true;
    }

    impl_                = std::make_unique<Impl>();
    impl_->tempAllocator = std::make_unique<JPH::TempAllocatorImpl>(settings_.tempAllocatorBytes);
    const int hardware   = static_cast<int>(std::thread::hardware_concurrency());
    threadCount_         = settings_.workerThreads >= 0 ? settings_.workerThreads : std::max(1, hardware - 1);
    impl_->jobSystem = std::make_unique<JPH::JobSystemThreadPool>(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, threadCount_);
    impl_->system    = std::make_unique<JPH::PhysicsSystem>();
    impl_->system->Init(settings_.maxBodies, 0, settings_.maxBodyPairs, settings_.maxContactConstraints,
                        impl_->broadPhaseLayers, impl_->objectVsBroadPhase, impl_->objectPairs);
    impl_->system->SetGravity(JPH::Vec3(0.0f, -9.81f, 0.0f));

    std::printf("[PhysicsCore] Jolt initialise : %d thread(s), %u bodies max, pas fixe %.4f s\n", threadCount_,
                settings_.maxBodies, static_cast<double>(settings_.fixedTimeStep));
    return true;
}

void PhysicsCore::Shutdown() {
    if (impl_) {
        impl_->system.reset();
        impl_->jobSystem.reset();
        impl_->tempAllocator.reset();
        impl_.reset();
    }
    if (ownsFactory_) {
        JPH::UnregisterTypes();
        delete JPH::Factory::sInstance;
        JPH::Factory::sInstance = nullptr;
        ownsFactory_            = false;
    }
}

void PhysicsCore::Step(float dt) {
    stepsLastFrame_ = 0;
    if (!impl_ || paused) return;
    const auto start = std::chrono::steady_clock::now();
    accumulator_ += static_cast<double>(dt);
    const double fixed = static_cast<double>(settings_.fixedTimeStep);
    while (accumulator_ >= fixed && stepsLastFrame_ < settings_.maxSubSteps) {
        impl_->system->Update(settings_.fixedTimeStep, settings_.collisionSteps, impl_->tempAllocator.get(),
                              impl_->jobSystem.get());
        accumulator_ -= fixed;
        ++stepsLastFrame_;
        ++stepCount_;
    }
    // Spirale de la mort : si on a atteint le maximum de sous-pas, on abandonne le retard.
    if (stepsLastFrame_ == settings_.maxSubSteps && accumulator_ >= fixed) accumulator_ = 0.0;
    lastStepMs_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void PhysicsCore::TranslateAllBodies(const glm::vec3& delta) {
    if (!impl_) return;
    JPH::BodyIDVector ids;
    impl_->system->GetBodies(ids);
    JPH::BodyInterface& bodies = impl_->system->GetBodyInterface();
    const JPH::RVec3    offset(delta.x, delta.y, delta.z);
    for (const JPH::BodyID& id : ids)
        bodies.SetPosition(id, bodies.GetPosition(id) + offset, JPH::EActivation::DontActivate);
}

JPH::PhysicsSystem& PhysicsCore::System() {
    if (!impl_) throw std::runtime_error("PhysicsCore non initialise");
    return *impl_->system;
}

JPH::BodyInterface& PhysicsCore::Bodies() { return System().GetBodyInterface(); }

JPH::TempAllocator& PhysicsCore::TempAllocator() {
    if (!impl_) throw std::runtime_error("PhysicsCore non initialise");
    return *impl_->tempAllocator;
}

JPH::JobSystem& PhysicsCore::Jobs() {
    if (!impl_) throw std::runtime_error("PhysicsCore non initialise");
    return *impl_->jobSystem;
}

std::uint32_t PhysicsCore::BodyCount() const { return impl_ ? impl_->system->GetNumBodies() : 0u; }

std::uint32_t PhysicsCore::ActiveBodyCount() const {
    return impl_ ? impl_->system->GetNumActiveBodies(JPH::EBodyType::RigidBody) : 0u;
}

float PhysicsCore::InterpolationAlpha() const noexcept {
    return static_cast<float>(accumulator_ / static_cast<double>(settings_.fixedTimeStep));
}

void PhysicsCore::OnGui() {
    if (!impl_) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.3f, 1.0f), "Jolt non initialise");
        return;
    }
    ImGui::Text("Threads : %d   Pas fixe : %.2f ms   Sous-pas max : %d", threadCount_,
                static_cast<double>(settings_.fixedTimeStep * 1000.0f), settings_.maxSubSteps);
    ImGui::Text("Bodies : %u (actifs %u) / %u max", BodyCount(), ActiveBodyCount(), settings_.maxBodies);
    ImGui::Text("Sous-pas cette frame : %d   Cout : %.3f ms   Total : %llu pas", stepsLastFrame_,
                static_cast<double>(lastStepMs_), static_cast<unsigned long long>(stepCount_));
    ImGui::Checkbox("Pause", &paused);
    const JPH::Vec3 g       = impl_->system->GetGravity();
    float           gravity = g.GetY();
    if (ImGui::SliderFloat("Gravite Y", &gravity, -30.0f, 0.0f, "%.2f m/s2"))
        impl_->system->SetGravity(JPH::Vec3(g.GetX(), gravity, g.GetZ()));
    if (ImGui::Button("Optimiser le broad-phase")) impl_->system->OptimizeBroadPhase();
    ImGui::TextDisabled("Promotion de proximite : PhysicsProximitySystem (Dev 1, rayon 20 m)");
}

}  // namespace env::core
