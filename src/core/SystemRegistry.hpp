#pragma once
// =============================================================================
//  src/core/SystemRegistry.hpp — Auto-enregistrement statique des systèmes (ADR-002)
//
//  Usage, dans le .cpp d'un système (à portée de namespace, après la classe) :
//
//      REGISTER_SYSTEM(MySystem, PassStage::WorldCompute);
//      REGISTER_SYSTEM_ORDERED(MyOtherSystem, PassStage::LightingWater, 10);
//
//  * Pas de header, pas de liste centrale : le .cpp est collecté par GLOB_RECURSE
//    et s'enregistre pendant l'initialisation statique.
//  * Registre = singleton de Meyers : aucun problème d'ordre d'initialisation statique.
//  * Ordre d'exécution déterministe : (stage, order, nom), indépendant de l'éditeur de liens.
// =============================================================================
#include <algorithm>
#include <cstdio>
#include <memory>
#include <string_view>
#include <type_traits>
#include <vector>

#include "contracts/ISystemModule.hpp"

namespace env::core {

using SystemFactory = std::unique_ptr<ISystemModule> (*)();

struct SystemDescriptor {
    std::string_view name;
    PassStage        stage = PassStage::DebugOverlay;
    int              order = 0;
    SystemFactory    factory = nullptr;
    std::string_view sourceFile;
};

struct SystemInstance {
    SystemDescriptor               descriptor;
    std::unique_ptr<ISystemModule> module;
};

class SystemRegistry {
public:
    static SystemRegistry& Instance() {
        static SystemRegistry registry;
        return registry;
    }

    /// Appelé par REGISTER_SYSTEM pendant l'initialisation statique. Refuse les doublons de nom.
    bool Register(const SystemDescriptor& descriptor) {
        for (const SystemDescriptor& existing : descriptors_) {
            if (existing.name == descriptor.name) {
                std::fprintf(stderr,
                             "[SystemRegistry] Doublon refuse : '%.*s' (%.*s) deja enregistre par %.*s\n",
                             static_cast<int>(descriptor.name.size()), descriptor.name.data(),
                             static_cast<int>(descriptor.sourceFile.size()), descriptor.sourceFile.data(),
                             static_cast<int>(existing.sourceFile.size()), existing.sourceFile.data());
                return false;
            }
        }
        descriptors_.push_back(descriptor);
        return true;
    }

    /// Descripteurs triés par (stage, order, nom).
    std::vector<SystemDescriptor> SortedDescriptors() const {
        std::vector<SystemDescriptor> sorted = descriptors_;
        std::stable_sort(sorted.begin(), sorted.end(), [](const SystemDescriptor& a, const SystemDescriptor& b) {
            if (a.stage != b.stage) return static_cast<int>(a.stage) < static_cast<int>(b.stage);
            if (a.order != b.order) return a.order < b.order;
            return a.name < b.name;
        });
        return sorted;
    }

    /// Instancie tous les systèmes enregistrés, dans l'ordre d'exécution.
    std::vector<SystemInstance> InstantiateAll() const {
        std::vector<SystemInstance> instances;
        for (const SystemDescriptor& d : SortedDescriptors()) {
            std::unique_ptr<ISystemModule> module = d.factory ? d.factory() : nullptr;
            if (!module) continue;
            module->name_  = d.name;
            module->stage_ = d.stage;
            instances.push_back(SystemInstance{d, std::move(module)});
        }
        return instances;
    }

    const std::vector<SystemDescriptor>& Descriptors() const noexcept { return descriptors_; }

private:
    SystemRegistry() = default;
    std::vector<SystemDescriptor> descriptors_;
};

}  // namespace env::core

#define ENV_DETAIL_CONCAT_IMPL(a, b) a##b
#define ENV_DETAIL_CONCAT(a, b) ENV_DETAIL_CONCAT_IMPL(a, b)

/// Enregistre ClassName (dérivée de env::ISystemModule, constructible par défaut)
/// dans le stage Stage avec un ordre explicite (croissant) à l'intérieur du stage.
#define REGISTER_SYSTEM_ORDERED(ClassName, Stage, Order)                                                   \
    namespace {                                                                                            \
    [[maybe_unused]] const bool ENV_DETAIL_CONCAT(kEnvSystemRegistered_, ClassName) =                      \
        ::env::core::SystemRegistry::Instance().Register(::env::core::SystemDescriptor{                    \
            #ClassName, (Stage), (Order),                                                                  \
            []() -> ::std::unique_ptr<::env::ISystemModule> {                                              \
                static_assert(::std::is_base_of_v<::env::ISystemModule, ClassName>,                        \
                              #ClassName " doit deriver de env::ISystemModule");                           \
                return ::std::make_unique<ClassName>();                                                    \
            },                                                                                             \
            __FILE__});                                                                                    \
    }                                                                                                      \
    static_assert(true, "")

/// Enregistre ClassName dans le stage Stage (ordre 0).
#define REGISTER_SYSTEM(ClassName, Stage) REGISTER_SYSTEM_ORDERED(ClassName, Stage, 0)
