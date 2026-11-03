// =============================================================================
//  src/world_engine/PhysicsProximitySystem.cpp                      [DEV 1]
//  Stage : PhysicsSync
//
//  Rôle : promotion dynamique de proximité vers Jolt Physics (ADR-006). Seules
//  les entités promouvables proches de la caméra (ou d'une entité active)
//  reçoivent un body ; démotion au-delà du rayon avec hystérésis ; colliders de
//  terrain uniquement pour les chunks à portée.
//  Le core avance la simulation après tous les Update et translate lui-même
//  les bodies lors d'un recalage de l'origine de rendu (ADR-007).
//
//  Coquille vide : toute la logique est à écrire par Dev 1.
//  RÈGLE : ce fichier n'inclut jamais src/render_engine/ (vérifié au build).
// =============================================================================
#include <imgui.h>

#include "contracts/ISystemModule.hpp"
#include "contracts/WorldContracts.hpp"
#include "core/SystemRegistry.hpp"

namespace env::world {
namespace {

void TodoBanner(const char* scope) {
    ImGui::TextColored(ImVec4(1.0f, 0.62f, 0.15f, 1.0f), "TODO Dev 1");
    ImGui::SameLine();
    ImGui::TextDisabled("%s", scope);
}

class PhysicsProximitySystem final : public ISystemModule {
public:
    void Init(EngineContext&) override {
        // [DEV 1 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : le PhysicsCore fourni par EngineContext et ses couches d'objets (Static, Dynamic,
        //            Character, Sensor).
        // Sorties  : formes de collision partagées par archétype (tronc, rocher, ...).
        // Objectif : créer une seule fois les formes réutilisées par tous les bodies promus.
    }

    void Update(EngineContext&) override {
        // [DEV 1 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : position monde double précision de la caméra (et des entités actives) ; entités
        //            portant TransformComponent et PhysicsPromotableTag sans PhysicsBodyComponent ; entités
        //            déjà promues ; chunks à portée et leur hauteur ; origine de rendu courante.
        // Sorties  : bodies Jolt créés pour les entités entrées dans le rayon (position relative à
        //            l'origine de rendu, couche d'objets adaptée) et PhysicsBodyComponent ajouté ; bodies
        //            retirés puis détruits et composant supprimé au-delà du rayon augmenté de l'hystérésis ;
        //            colliders de terrain (champ de hauteur) pour les chunks à portée et bit
        //            PhysicsCollider ajouté à readyMask ; TransformComponent mis à jour depuis les bodies
        //            dynamiques (position monde double reconstruite à partir de l'origine de rendu).
        // Objectif : garder un nombre de bodies borné et indépendant de la taille du monde, avec un budget
        //            de promotions par frame et une relecture asynchrone, sans blocage, des hauteurs.
    }

    void Render(EngineContext&) override {
        // [DEV 1 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : aucune.
        // Sorties  : aucune.
        // Objectif : aucune commande GPU attendue ; un affichage de debug des colliders relèverait d'un
        //            système du stage DebugOverlay.
    }

    void Shutdown(EngineContext&) override {
        // [DEV 1 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : bodies et formes créés par ce système.
        // Sorties  : tous les bodies retirés et détruits, PhysicsBodyComponent supprimé, formes libérées.
        // Objectif : rendre Jolt vide avant l'arrêt du PhysicsCore.
    }

    void OnGui(EngineContext&) override {
        TodoBanner("promotion / demotion Jolt dans un rayon de 20 m, colliders de terrain (ADR-006)");
        ImGui::SliderFloat("Rayon de promotion", &promotionRadius_, 5.0f, 100.0f, "%.1f m");
        ImGui::SliderFloat("Hysteresis de demotion", &demotionMargin_, 0.0f, 20.0f, "%.1f m");
        ImGui::SliderInt("Promotions max / frame", &maxPromotionsPerFrame_, 1, 256);
        ImGui::Checkbox("Colliders de terrain", &terrainColliders_);
        ImGui::SliderInt("Resolution heightfield", &terrainColliderResolution_, 8, 128);
    }

private:
    // Variables liées aux sliders d'OnGui (aucune logique branchée).
    float promotionRadius_           = kPhysicsPromotionRadius;
    float demotionMargin_            = 5.0f;
    int   maxPromotionsPerFrame_     = 32;
    bool  terrainColliders_          = true;
    int   terrainColliderResolution_ = 32;
};

}  // namespace
REGISTER_SYSTEM(PhysicsProximitySystem, PassStage::PhysicsSync);
}  // namespace env::world
