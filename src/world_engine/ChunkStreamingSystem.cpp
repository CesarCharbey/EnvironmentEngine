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
#include <vector>
#include <unordered_map>
#include <algorithm>

#include "contracts/ISystemModule.hpp"
#include "contracts/WorldContracts.hpp"
#include "core/SystemRegistry.hpp"
#include "core/RenderBlackboard.hpp"
#include "core/gpu/Texture2D.hpp"
#include <entt/entt.hpp>
#include <glm/gtx/norm.hpp>

namespace env::world {
namespace {

// Permet d'utiliser ivec2 comme clé de notre std::unordered_map
struct IVec2Hash {
    std::size_t operator()(const glm::ivec2& v) const noexcept {
        return std::hash<int>()(v.x) ^ (std::hash<int>()(v.y) << 1);
    }
};

// Structure privée à Dev 1 : contient les VRAIS objets OpenGL (RAII).
// Quand cette structure est détruite, glDeleteTextures est appelé automatiquement.
struct PooledChunk {
    gpu::Texture2D heightNormalTex;
    gpu::Texture2D biomeHydroTex;
    // (Jalon 3) On ajoutera ici le GpuBuffer pour la végétation
};

class ChunkStreamingSystem final : public ISystemModule {
public:
    void Init(EngineContext& ctx) override {
        int maxPoolSize = (unloadRadiusChunks_ * 2 + 1) * (unloadRadiusChunks_ * 2 + 1) + 16;
        
        for (int i = 0; i < maxPoolSize; ++i) {
            gpu::Texture2D::Desc hDesc;
            hDesc.width = kChunkTexResolution;
            hDesc.height = kChunkTexResolution;
            hDesc.internalFormat = GL_RGBA32F;
            hDesc.levels = 1;
            hDesc.sampler = gpu::SamplerDesc::LinearMipmapped(1.0f);
            hDesc.sampler.minFilter = GL_LINEAR;
            // ---- AJOUTEZ CES DEUX LIGNES ----
            hDesc.sampler.wrapS = GL_CLAMP_TO_EDGE;
            hDesc.sampler.wrapT = GL_CLAMP_TO_EDGE;
            // ---------------------------------
            hDesc.debugName = "chunk.heightNormal";
            
            gpu::Texture2D::Desc bDesc;
            bDesc.width = kChunkTexResolution;
            bDesc.height = kChunkTexResolution;
            bDesc.internalFormat = GL_RGBA16F;
            bDesc.levels = 1;
            bDesc.sampler = gpu::SamplerDesc::LinearMipmapped(1.0f);
            bDesc.sampler.minFilter = GL_NEAREST;
            // ---- AJOUTEZ CES DEUX LIGNES ----
            bDesc.sampler.wrapS = GL_CLAMP_TO_EDGE;
            bDesc.sampler.wrapT = GL_CLAMP_TO_EDGE;
            // ---------------------------------
            bDesc.debugName = "chunk.biomeHydro";

            PooledChunk pc;
            pc.heightNormalTex.Create(hDesc);
            pc.biomeHydroTex.Create(bDesc);
            pool_.push_back(std::move(pc));
        }
    }

    void Update(EngineContext& ctx) override {
        if (freezeStreaming_) {
            UpdateFocus(ctx);
            return;
        }

        // 1. Calcul du chunk sous la caméra (Règle d'Origine Flottante)
        glm::dvec3 absoluteCamPos = RenderOriginWorld(ctx.renderOriginChunk) + glm::dvec3(ctx.environment.cameraPosition);
        glm::ivec2 centerChunk = WorldToChunk(absoluteCamPos);

        // 2. Éviction (Destruction des chunks trop éloignés)
        int evictions = 0;
        auto view = ctx.registry.view<ChunkCoordComponent, ChunkStateComponent>();
        float unloadSq = static_cast<float>(unloadRadiusChunks_ * unloadRadiusChunks_);

        for (auto [entity, coord, state] : view.each()) {
            if (ctx.registry.any_of<MockChunkTag>(entity)) continue;

            float distSq = glm::distance2(glm::vec2(centerChunk), glm::vec2(coord.coord));

            if (distSq > unloadSq) {
                if (state.state != ChunkState::Evicting && evictions < maxEvictionsPerFrame_) {
                    state.state = ChunkState::Evicting;
                    evictions++;

                    // Récupération des objets RAII pour les remettre dans le pool
                    auto it = activeResources_.find(entity);
                    if (it != activeResources_.end()) {
                        pool_.push_back(std::move(it->second));
                        activeResources_.erase(it);
                    }
                    
                    // On retire le composant avant destruction pour que Dev 2 ne lise pas une mémoire réattribuée
                    ctx.registry.remove<ChunkGpuResources>(entity); 
                    chunkMap_.erase(coord.coord);
                    ctx.registry.destroy(entity);
                }
            } else {
                state.lastTouchedTime = ctx.environment.waterTime.z; // Horodatage LRU
            }
        }

        // 3. Création (Demande de nouveaux chunks en spirale/grille)
        int requests = 0;
        int viewSq = viewRadiusChunks_ * viewRadiusChunks_;

        for (int x = -viewRadiusChunks_; x <= viewRadiusChunks_ && requests < maxRequestsPerFrame_; ++x) {
            for (int y = -viewRadiusChunks_; y <= viewRadiusChunks_ && requests < maxRequestsPerFrame_; ++y) {
                if (x * x + y * y <= viewSq) {
                    glm::ivec2 target(centerChunk.x + x, centerChunk.y + y);
                    
                    // Si le chunk n'existe pas et qu'il reste de la place dans le pool
                    if (chunkMap_.find(target) == chunkMap_.end() && !pool_.empty()) {
                        entt::entity entity = ctx.registry.create();
                        
                        ChunkCoordComponent c{};
                        c.coord = target;
                        ctx.registry.emplace<ChunkCoordComponent>(entity, c);
                        
                        ChunkStateComponent s{};
                        s.state = ChunkState::Requested;
                        s.lastTouchedTime = ctx.environment.waterTime.z;
                        ctx.registry.emplace<ChunkStateComponent>(entity, s);
                        
                        // On prend les ressources RAII du pool
                        PooledChunk pc = std::move(pool_.back());
                        pool_.pop_back();

                        // On expose la VUE (les IDs OpenGL bruts) pour le moteur de rendu
                        ChunkGpuResources resView{};
                        resView.heightNormalTex = pc.heightNormalTex.Id();
                        resView.biomeHydroTex = pc.biomeHydroTex.Id();
                        resView.foliageInstanceCapacity = kFoliageMaxInstancesPerChunk;
                        resView.foliageCommandCount = kFoliageArchetypeCount;
                        ctx.registry.emplace<ChunkGpuResources>(entity, resView);

                        // On garde la PROPRIÉTÉ des objets côté World Engine
                        activeResources_[entity] = std::move(pc);
                        chunkMap_[target] = entity;
                        
                        requests++;
                    }
                }
            }
        }

        UpdateFocus(ctx);
    }

    void Render(EngineContext&) override {}

    void Shutdown(EngineContext& ctx) override {
        // On retire nos textures du blackboard
        ctx.blackboard.UnpublishIfProducer(bb::WorldFocusHeightNormal, "Dev 1");
        ctx.blackboard.UnpublishIfProducer(bb::WorldFocusBiomeHydro, "Dev 1");
        
        // Nettoyage de l'ECS
        auto view = ctx.registry.view<ChunkCoordComponent>();
        ctx.registry.destroy(view.begin(), view.end());
        
        // Les destructeurs de gpu::Texture2D (RAII) se chargeront des glDeleteTextures
        activeResources_.clear();
        pool_.clear();
        chunkMap_.clear();
    }

    void OnGui(EngineContext&) override {
        ImGui::SeparatorText("Streaming");
        ImGui::SliderInt("Rayon de vue (chunks)", &viewRadiusChunks_, 1, 32);
        ImGui::SliderInt("Rayon de dechargement (chunks)", &unloadRadiusChunks_, 1, 40);
        ImGui::SliderInt("Creations max / frame", &maxRequestsPerFrame_, 1, 32);
        ImGui::SliderInt("Evictions max / frame", &maxEvictionsPerFrame_, 1, 64);
        ImGui::Checkbox("Geler le streaming", &freezeStreaming_);
        
        ImGui::Text("Chunks en memoire GPU : %zu (Actifs) / %zu (Pool)", activeResources_.size(), pool_.size());
    }

private:
    void UpdateFocus(EngineContext& ctx) {
        glm::dvec3 absoluteCamPos = RenderOriginWorld(ctx.renderOriginChunk) + glm::dvec3(ctx.environment.cameraPosition);
        glm::ivec2 centerChunk = WorldToChunk(absoluteCamPos);
        
        auto it = chunkMap_.find(centerChunk);
        if (it != chunkMap_.end()) {
            auto resIt = activeResources_.find(it->second);
            if (resIt != activeResources_.end()) {
                // On met à disposition la texture du chunk central pour le G-Buffer et le Debug
                ctx.blackboard.PublishTexture(bb::WorldFocusHeightNormal, resIt->second.heightNormalTex, "Dev 1");
                ctx.blackboard.PublishTexture(bb::WorldFocusBiomeHydro, resIt->second.biomeHydroTex, "Dev 1");
            }
        }
    }

    int  viewRadiusChunks_     = 8;
    int  unloadRadiusChunks_   = 10;
    int  maxRequestsPerFrame_  = 4;
    int  maxEvictionsPerFrame_ = 8;
    bool freezeStreaming_      = false;

    // Gestion de la mémoire
    std::vector<PooledChunk> pool_;
    std::unordered_map<entt::entity, PooledChunk> activeResources_;
    std::unordered_map<glm::ivec2, entt::entity, IVec2Hash> chunkMap_;
};

}  // namespace
REGISTER_SYSTEM(ChunkStreamingSystem, PassStage::WorldStreaming);
}  // namespace env::world