// =============================================================================
//  src/world_engine/ChunkStreamingSystem.cpp                        [DEV 1]
//  Stage : WorldStreaming
//
//  Rôle : maintenir l'ensemble des chunks résidents autour de la caméra dans un
//  monde infini (création / éviction des entités chunk), posséder le réservoir
//  de ressources GPU des chunks et publier le chunk situé sous la caméra.
//
//  Coquille vide : toute la logique est à écrire par Dev 1.
//  RÈGLE : ce fichier n'inclut jamais src/render_engine/ (vérifié au build).
// =============================================================================
#include <imgui.h>

#include "contracts/ISystemModule.hpp"
#include "core/SystemRegistry.hpp"

namespace env::world {
namespace {

void TodoBanner(const char* scope) {
    ImGui::TextColored(ImVec4(1.0f, 0.62f, 0.15f, 1.0f), "TODO Dev 1");
    ImGui::SameLine();
    ImGui::TextDisabled("%s", scope);
}

class ChunkStreamingSystem final : public ISystemModule {
public:
    void Init(EngineContext&) override {
        // [DEV 1 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : constantes du contrat (taille d'un chunk, résolution des textures de chunk, capacité
        //            maximale d'instances par chunk, nombre d'archétypes de végétation) et rayons du panneau.
        // Sorties  : un réservoir de jeux de ressources GPU par chunk, alloués une seule fois : texture
        //            hauteur + normale RGBA32F, texture biome + hydrologie RGBA16F, buffer d'instances de
        //            végétation, buffer de commandes de tirage indirect (une par archétype).
        // Objectif : ne jamais allouer de mémoire GPU pendant le streaming. Le World Engine reste
        //            propriétaire de ces ressources ; le composant ChunkGpuResources n'en est qu'une vue.
    }

    void Update(EngineContext&) override {
        // [DEV 1 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : position monde double précision de la caméra, entités chunk existantes
        //            (ChunkCoordComponent, ChunkStateComponent) hors entités portant MockChunkTag,
        //            temps de la frame, paramètres du panneau.
        // Sorties  : nouvelles entités chunk à l'état Requested (coordonnées entières + LOD) ;
        //            chunks sortis de la zone passés à l'état Evicting, puis détruits à la frame suivante
        //            (ChunkGpuResources retiré AVANT de rendre le jeu de ressources au réservoir) ;
        //            lastTouchedTime mis à jour ; clés world.focus.heightNormal et world.focus.biomeHydro
        //            publiées pour le chunk sous la caméra.
        // Objectif : calculer l'ensemble désiré (anneaux ou spirale autour du chunk de la caméra, LOD par
        //            anneau, priorité par distance et cône de vision), plafonner créations et évictions par
        //            frame, appliquer une hystérésis entre rayon de vue et rayon de déchargement, politique
        //            LRU. Un recalage de l'origine de rendu ne demande rien ici (coordonnées entières).
    }

    void Render(EngineContext&) override {
        // [DEV 1 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : aucune.
        // Sorties  : aucune.
        // Objectif : aucune commande GPU attendue ; la génération des chunks a lieu dans les systèmes
        //            du stage WorldCompute.
    }

    void Shutdown(EngineContext&) override {
        // [DEV 1 WORKSPACE] — Spécification fonctionnelle
        // Entrées  : entités chunk créées par ce système, réservoir de ressources.
        // Sorties  : entités détruites, clés world.focus.* retirées si ce système en est le producteur,
        //            ressources GPU du réservoir libérées.
        // Objectif : laisser l'ECS et le blackboard propres avant la destruction du contexte OpenGL.
    }

    void OnGui(EngineContext&) override {
        TodoBanner("ensemble desire, creation / eviction d'entites chunk, reservoir GPU, chunk focus");
        ImGui::SliderInt("Rayon de vue (chunks)", &viewRadiusChunks_, 1, 32);
        ImGui::SliderInt("Rayon de dechargement (chunks)", &unloadRadiusChunks_, 1, 40);
        ImGui::SliderInt("Largeur d'anneau LOD (chunks)", &lodRingWidth_, 1, 16);
        ImGui::SliderInt("Creations max / frame", &maxRequestsPerFrame_, 1, 32);
        ImGui::SliderInt("Evictions max / frame", &maxEvictionsPerFrame_, 1, 64);
        ImGui::Checkbox("Geler le streaming", &freezeStreaming_);
    }

private:
    // Variables liées aux sliders d'OnGui (aucune logique branchée).
    int  viewRadiusChunks_     = 8;
    int  unloadRadiusChunks_   = 10;
    int  lodRingWidth_         = 4;
    int  maxRequestsPerFrame_  = 4;
    int  maxEvictionsPerFrame_ = 8;
    bool freezeStreaming_      = false;
};

}  // namespace
REGISTER_SYSTEM(ChunkStreamingSystem, PassStage::WorldStreaming);
}  // namespace env::world
