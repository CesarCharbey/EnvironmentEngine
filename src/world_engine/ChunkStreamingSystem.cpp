// src/world_engine/ChunkStreamingSystem.cpp
#include <imgui.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>

#include "contracts/ISystemModule.hpp"
#include "contracts/WorldContracts.hpp"
#include "core/SystemRegistry.hpp"
#include "core/RenderBlackboard.hpp"
#include "core/gpu/Texture2D.hpp"
#include <entt/entt.hpp>
#include <glm/gtx/norm.hpp>

namespace env::world {
namespace {

struct IVec2Hash {
    std::size_t operator()(const glm::ivec2& v) const noexcept {
        return std::hash<int>()(v.x) ^ (std::hash<int>()(v.y) << 1);
    }
};

struct PooledChunk {
    gpu::Texture2D heightNormalTex;
    gpu::Texture2D biomeHydroTex;
};

class ChunkStreamingSystem final : public ISystemModule {
public:
    void Init(EngineContext&) override {
        // Pré-allocation ajustée au disque réel (pi * r^2 + marge) au lieu du carré unloadRadius
        const int initialPoolSize = static_cast<int>(3.15f * viewRadiusChunks_ * viewRadiusChunks_) + 32;
        pool_.reserve(initialPoolSize);
        for (int i = 0; i < initialPoolSize; ++i) {
            pool_.push_back(AllocateSlot());
        }
    }

    void Update(EngineContext& ctx) override {
        if (freezeStreaming_) {
            UpdateFocus(ctx);
            return;
        }

        const glm::dvec3 absoluteCamPos = RenderOriginWorld(ctx.renderOriginChunk) + glm::dvec3(ctx.environment.cameraPosition);
        const glm::ivec2 centerChunk = WorldToChunk(absoluteCamPos);

        // 1. Finalisation des évictions marquées à la frame N-1 (destruction sécurisée à N+1)
        std::vector<entt::entity> toDestroy;
        auto view = ctx.registry.view<ChunkCoordComponent, ChunkStateComponent>(entt::exclude<MockChunkTag>);
        for (auto [entity, coord, state] : view.each()) {
            if (state.state == ChunkState::Evicting) {
                toDestroy.push_back(entity);
            }
        }
        for (entt::entity entity : toDestroy) {
            const auto& coord = ctx.registry.get<ChunkCoordComponent>(entity);
            auto it = activeResources_.find(entity);
            if (it != activeResources_.end()) {
                pool_.push_back(std::move(it->second));
                activeResources_.erase(it);
            }
            ctx.registry.remove<ChunkGpuResources>(entity);
            chunkMap_.erase(coord.coord);
            ctx.registry.destroy(entity);
        }

        // 2. Marquage Evicting pour les chunks hors de unloadRadiusChunks_ (frame N)
        int evictions = 0;
        const float unloadSq = static_cast<float>(unloadRadiusChunks_ * unloadRadiusChunks_);
        for (auto [entity, coord, state] : view.each()) {
            const float distSq = glm::distance2(glm::vec2(centerChunk), glm::vec2(coord.coord));
            if (distSq > unloadSq) {
                if (state.state != ChunkState::Evicting && evictions < maxEvictionsPerFrame_) {
                    state.state = ChunkState::Evicting;
                    evictions++;
                }
            } else {
                state.lastTouchedTime = ctx.environment.waterTime.z;
            }
        }

        // 3. Collecte et tri des chunks manquants par distance croissante à la caméra
        struct Candidate { glm::ivec2 coord; int distSq; };
        std::vector<Candidate> candidates;
        const int viewSq = viewRadiusChunks_ * viewRadiusChunks_;

        for (int x = -viewRadiusChunks_; x <= viewRadiusChunks_; ++x) {
            for (int y = -viewRadiusChunks_; y <= viewRadiusChunks_; ++y) {
                const int dSq = x * x + y * y;
                if (dSq <= viewSq) {
                    glm::ivec2 target(centerChunk.x + x, centerChunk.y + y);
                    if (chunkMap_.find(target) == chunkMap_.end()) {
                        candidates.push_back({target, dSq});
                    }
                }
            }
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) { return a.distSq < b.distSq; });

        // 4. Création des chunks les plus proches en priorité
        int requests = 0;
        for (const auto& cand : candidates) {
            if (requests >= maxRequestsPerFrame_) break;

            if (pool_.empty()) {
                pendingSlotsNeeded_ = std::max(pendingSlotsNeeded_, maxRequestsPerFrame_ - requests);
                break;
            }

            entt::entity entity = ctx.registry.create();
            ctx.registry.emplace<ChunkCoordComponent>(entity, ChunkCoordComponent{cand.coord, 0});

            ChunkStateComponent s{};
            s.state = ChunkState::Requested;
            s.lastTouchedTime = ctx.environment.waterTime.z;
            ctx.registry.emplace<ChunkStateComponent>(entity, s);

            PooledChunk pc = std::move(pool_.back());
            pool_.pop_back();

            ChunkGpuResources resView{};
            resView.heightNormalTex = pc.heightNormalTex.Id();
            resView.biomeHydroTex   = pc.biomeHydroTex.Id();
            resView.foliageInstanceCapacity = kFoliageMaxInstancesPerChunk;
            resView.foliageCommandCount     = kFoliageArchetypeCount;
            ctx.registry.emplace<ChunkGpuResources>(entity, resView);

            activeResources_[entity] = std::move(pc);
            chunkMap_[cand.coord] = entity;
            requests++;
        }

        UpdateFocus(ctx);
    }

    void Render(EngineContext&) override {
        while (pendingSlotsNeeded_ > 0) {
            pool_.push_back(AllocateSlot());
            pendingSlotsNeeded_--;
        }
    }

    void Shutdown(EngineContext& ctx) override {
        ctx.blackboard.UnpublishIfProducer(bb::WorldFocusHeightNormal, "Dev 1");
        ctx.blackboard.UnpublishIfProducer(bb::WorldFocusBiomeHydro, "Dev 1");

        auto view = ctx.registry.view<ChunkCoordComponent>(entt::exclude<MockChunkTag>);
        ctx.registry.destroy(view.begin(), view.end());

        activeResources_.clear();
        pool_.clear();
        chunkMap_.clear();
    }

    void OnGui(EngineContext&) override {
        ImGui::SeparatorText("Streaming");
        ImGui::SliderInt("Rayon de vue (chunks)", &viewRadiusChunks_, 1, 32);
        ImGui::SliderInt("Rayon de dechargement (chunks)", &unloadRadiusChunks_, viewRadiusChunks_ + 1, 40);
        ImGui::SliderInt("Creations max / frame", &maxRequestsPerFrame_, 1, 32);
        ImGui::SliderInt("Evictions max / frame", &maxEvictionsPerFrame_, 1, 64);
        ImGui::Checkbox("Geler le streaming", &freezeStreaming_);
        ImGui::Text("Chunks GPU : %zu actifs / %zu pool", activeResources_.size(), pool_.size());
    }

private:
    PooledChunk AllocateSlot() const {
        gpu::Texture2D::Desc hDesc;
        hDesc.width = kChunkTexResolution;
        hDesc.height = kChunkTexResolution;
        hDesc.internalFormat = GL_RGBA32F;
        hDesc.levels = 1;
        hDesc.sampler.minFilter = GL_LINEAR;
        hDesc.sampler.magFilter = GL_LINEAR;
        hDesc.sampler.wrapS = GL_CLAMP_TO_EDGE;
        hDesc.sampler.wrapT = GL_CLAMP_TO_EDGE;
        hDesc.debugName = "chunk.heightNormal";

        gpu::Texture2D::Desc bDesc;
        bDesc.width = kChunkTexResolution;
        bDesc.height = kChunkTexResolution;
        bDesc.internalFormat = GL_RGBA16F;
        bDesc.levels = 1;
        bDesc.sampler.minFilter = GL_LINEAR;
        bDesc.sampler.magFilter = GL_LINEAR;
        bDesc.sampler.wrapS = GL_CLAMP_TO_EDGE;
        bDesc.sampler.wrapT = GL_CLAMP_TO_EDGE;
        bDesc.debugName = "chunk.biomeHydro";

        PooledChunk pc;
        pc.heightNormalTex.Create(hDesc);
        pc.biomeHydroTex.Create(bDesc);
        return pc;
    }

    void UpdateFocus(EngineContext& ctx) {
        const glm::dvec3 absoluteCamPos = RenderOriginWorld(ctx.renderOriginChunk) + glm::dvec3(ctx.environment.cameraPosition);
        const glm::ivec2 centerChunk = WorldToChunk(absoluteCamPos);

        auto it = chunkMap_.find(centerChunk);
        if (it != chunkMap_.end()) {
            const auto& state = ctx.registry.get<ChunkStateComponent>(it->second);
            if ((state.readyMask & ChunkReady::HeightNormal) != 0 && state.state != ChunkState::Evicting) {
                auto resIt = activeResources_.find(it->second);
                if (resIt != activeResources_.end()) {
                    ctx.blackboard.PublishTexture(bb::WorldFocusHeightNormal, resIt->second.heightNormalTex, "Dev 1");
                    ctx.blackboard.PublishTexture(bb::WorldFocusBiomeHydro,   resIt->second.biomeHydroTex,   "Dev 1");
                }
            }
        }
    }

    int  viewRadiusChunks_     = 8;
    int  unloadRadiusChunks_   = 10;
    int  maxRequestsPerFrame_  = 4;
    int  maxEvictionsPerFrame_ = 8;
    bool freezeStreaming_      = false;
    int pendingSlotsNeeded_ = 0;

    std::vector<PooledChunk> pool_;
    std::unordered_map<entt::entity, PooledChunk> activeResources_;
    std::unordered_map<glm::ivec2, entt::entity, IVec2Hash> chunkMap_;
};

}  // namespace
REGISTER_SYSTEM(ChunkStreamingSystem, PassStage::WorldStreaming);
}  // namespace env::world