#pragma once
// =============================================================================
//  src/core/MockDataProvider.hpp — Monde de test "Mock-First" (ADR-003)
//
//  Quand "Use Mock World" est coché (par défaut), crée une VRAIE entité chunk
//  EnTT (coord (0,0), tag MockChunkTag) conforme à WorldContracts.hpp :
//    * heightNormalTex  RGBA32F analytique (colline + vallée + plateau marécageux)
//    * biomeHydroTex    RGBA16F analytique (rivière sinusoïdale SDF, marais, humidité, biomes)
//    * SSBO de FoliageInstance (1000 par défaut, 4 archétypes) + 4 DrawElementsIndirectCommand
//  et publie world.focus.heightNormal / world.focus.biomeHydro dans le blackboard.
//
//  "Draw Debug Preview" dessine en argile / fausses couleurs TOUTES les entités
//  chunk dont readyMask contient ChunkReady::HeightNormal (mock ET vrais chunks
//  de Dev 1), via le même chemin indirect que la G-pass de Dev 2
//  (glMultiDrawElementsIndirect + gl_BaseInstance).
//  Shaders : shaders/core_debug/mock_preview.vert / .frag
// =============================================================================
#include <cstdint>

#include <entt/entity/registry.hpp>
#include <glm/glm.hpp>

#include "contracts/ISystemModule.hpp"
#include "core/gpu/GpuBuffer.hpp"
#include "core/gpu/Texture2D.hpp"

namespace env::gpu {
class ShaderManager;
class ShaderProgram;
}  // namespace env::gpu

namespace env::core {

class RenderBlackboard;

class MockDataProvider {
public:
    enum class ViewMode : int { Clay = 0, Height, Normal, Biome, RiverSDF, MarshSDF, Moisture, ChunkGrid, Count };

    struct TerrainParams {
        float hillHeight      = 58.0f;   ///< m
        float hillRadius      = 62.0f;   ///< m
        float riverAmplitude  = 38.0f;   ///< m (méandre)
        float riverFrequency  = 1.25f;   ///< cycles par chunk
        float riverHalfWidth  = 7.0f;    ///< m
        float valleyDepth     = 12.0f;   ///< m
        float marshRadius     = 40.0f;   ///< m
        float marshLevel      = 9.0f;    ///< m (plateau aplani)
        int   instanceCount   = 1000;
        std::uint32_t seed    = 1337u;
    };

    void Init(gpu::ShaderManager& shaders);
    void Shutdown(entt::registry& registry, RenderBlackboard& blackboard);

    /// Crée / détruit le chunk mock selon useMockWorld. Appelé avant les Update() des systèmes.
    void Update(EngineContext& ctx);
    /// Aperçu argile / fausses couleurs de tous les chunks prêts, dans la cible courante (scene).
    void Render(EngineContext& ctx);
    /// Widgets du panneau "Mock World" (le core ouvre la fenêtre).
    void OnGui(EngineContext& ctx);

    bool IsActive() const noexcept { return generated_; }

    bool          useMockWorld  = true;
    bool          drawPreview   = true;
    bool          drawTerrain   = true;
    bool          drawInstances = true;
    bool          wireframe     = false;
    ViewMode      viewMode      = ViewMode::Clay;
    glm::vec2     heightRange{0.0f, 80.0f};
    float         sdfRange      = 30.0f;
    int           gridResolution = 128;  ///< quads par côté du maillage de prévisualisation
    TerrainParams params{};

private:
    void Generate(EngineContext& ctx);
    void Destroy(entt::registry& registry, RenderBlackboard& blackboard);
    void BuildPreviewMeshes();

    gpu::ShaderProgram* program_ = nullptr;

    // Ressources du chunk mock (possédées ; ChunkGpuResources n'en est qu'une vue).
    gpu::Texture2D heightNormal_;
    gpu::Texture2D biomeHydro_;
    gpu::GpuBuffer instances_;
    gpu::GpuBuffer indirect_;
    entt::entity   entity_ = entt::null;

    // Maillages de prévisualisation (grille de terrain + cube unitaire).
    gpu::GpuBuffer   terrainVertices_, terrainIndices_, cubeVertices_, cubeIndices_;
    gpu::VertexArray terrainVao_, cubeVao_;
    GLsizei          terrainIndexCount_ = 0;
    int              builtGridResolution_ = 0;

    bool          generated_          = false;
    bool          regenerateRequested_ = false;
    float         generationMs_       = 0.0f;
    glm::vec2     generatedHeightRange_{0.0f, 1.0f};
    std::uint32_t instanceCounts_[4]  = {0, 0, 0, 0};
    std::uint32_t chunksDrawn_        = 0;
};

}  // namespace env::core
