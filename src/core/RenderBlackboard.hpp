#pragma once
// =============================================================================
//  src/core/RenderBlackboard.hpp — Bus nommé de textures / buffers GPU (ADR-003)
//
//  * NON-PROPRIÉTAIRE : le producteur garde ses objets RAII et publie leur
//    référence (PublishTexture / PublishBuffer). Republier après réallocation.
//  * Le consommateur reçoit TOUJOURS un objet GL valide : la ressource publiée,
//    ou le fallback déclaré pour la clé dans env::bb::kKeys (WorldContracts.hpp).
//    Une passe peut donc tourner avant que ses dépendances existent.
//  * Inspecteur ImGui intégré : liste des clés (publiée / fallback / demandée),
//    visualisation en fausses couleurs (canal, plage, colormap, isolignes),
//    affichage plein Viewport.
//  * Les fallbacks sont partagés et en LECTURE SEULE.
// =============================================================================
#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <string_view>

#include <glad/glad.h>
#include <glm/glm.hpp>

#include "contracts/WorldContracts.hpp"
#include "core/gpu/Cubemap.hpp"
#include "core/gpu/Framebuffer.hpp"
#include "core/gpu/GpuBuffer.hpp"
#include "core/gpu/Texture2D.hpp"
#include "core/gpu/Texture3D.hpp"

namespace env::gpu {
class ShaderManager;
class ShaderProgram;
}  // namespace env::gpu

namespace env::core {

struct TextureRef {
    GLuint           id             = 0;
    GLenum           target         = GL_TEXTURE_2D;  ///< GL_TEXTURE_2D, GL_TEXTURE_3D ou GL_TEXTURE_CUBE_MAP
    GLenum           internalFormat = 0;
    glm::ivec3       size{0};
    GLint            levels     = 1;
    bool             isFallback = true;
    std::string_view producer;  ///< valide jusqu'à la prochaine publication de la clé
};

struct BufferRef {
    GLuint           id         = 0;
    GLsizeiptr       size       = 0;
    bool             isFallback = true;
    std::string_view producer;
};

class RenderBlackboard {
public:
    /// Crée les ressources de fallback et l'inspecteur. Contexte GL requis.
    void Init(gpu::ShaderManager& shaders);
    void Shutdown();
    /// Appelé par le core en début de frame (remet le ZeroBuffer à zéro).
    void BeginFrame();

    // --- Publication (producteurs) ---------------------------------------------
    void PublishTexture(std::string_view key, const gpu::Texture2D& texture, std::string_view producer);
    void PublishTexture(std::string_view key, const gpu::Texture3D& texture, std::string_view producer);
    void PublishTexture(std::string_view key, const gpu::Cubemap& texture, std::string_view producer);
    void PublishTextureRaw(std::string_view key, GLuint id, GLenum target, GLenum internalFormat, glm::ivec3 size,
                           GLint levels, std::string_view producer);
    void PublishBuffer(std::string_view key, const gpu::GpuBuffer& buffer, std::string_view producer);
    void PublishBufferRaw(std::string_view key, GLuint id, GLsizeiptr size, std::string_view producer);
    void Unpublish(std::string_view key);
    /// Retire la clé seulement si elle est encore publiée par `producer`.
    void UnpublishIfProducer(std::string_view key, std::string_view producer);

    // --- Consommation ------------------------------------------------------------
    /// Ressource publiée, sinon fallback déclaré (Checker pour une clé inconnue). Jamais id = 0.
    TextureRef GetTexture(std::string_view key) const;
    /// Idem, mais une clé inconnue reçoit un fallback du type demandé (2D / 3D / cube).
    TextureRef GetTexture(std::string_view key, GLenum expectedTarget) const;
    BufferRef  GetBuffer(std::string_view key) const;
    bool       IsPublished(std::string_view key) const;

    /// glBindTextureUnit(unit, GetTexture(key).id). Retourne l'id lié.
    GLuint BindTexture(std::string_view key, GLuint unit) const;
    /// glBindBufferBase(target, index, GetBuffer(key).id). Retourne l'id lié.
    GLuint BindBuffer(std::string_view key, GLenum target, GLuint index) const;

    const TextureRef& FallbackTexture(bb::Fallback fallback) const;
    const BufferRef&  FallbackBuffer() const noexcept { return zeroBufferRef_; }

    // --- Outils --------------------------------------------------------------------
    /// Widgets du panneau "Render Blackboard" (le core ouvre la fenêtre).
    void OnGui();
    /// Rend la texture inspectée en fausses couleurs (appelé par le core après les passes).
    void RenderInspector();
    /// Sélectionne une clé dans l'inspecteur (ex. depuis un panneau système).
    void Inspect(std::string_view key);
    bool   InspectorShowsInViewport() const noexcept { return showInViewport_ && !inspectKey_.empty(); }
    GLuint InspectorOutputTexture() const noexcept;
    glm::ivec2 InspectorOutputSize() const noexcept { return {inspectorTarget_.Width(), inspectorTarget_.Height()}; }

private:
    struct Entry {
        std::string   producer;
        bool          isBuffer = false;
        TextureRef    texture{};
        BufferRef     buffer{};
        std::uint64_t publishCount = 0;
    };

    TextureRef ResolveTexture(std::string_view key, GLenum expectedTarget, bool track) const;
    void       TrackRequest(std::string_view key) const;
    void       ApplyPreset(int preset);
    void       AutoRange(const TextureRef& ref);
    void       DrawKeyTable();
    void       DrawInspector();

    std::map<std::string, Entry, std::less<>>         entries_;
    mutable std::set<std::string, std::less<>>        requested_;
    std::array<TextureRef, static_cast<std::size_t>(bb::Fallback::Count)> fallbackRefs_{};
    BufferRef zeroBufferRef_{};

    // Ressources de fallback (possédées).
    gpu::Texture2D checker_, black_, white_, flatNormalTS_, flatNormalWS_, flatHeightNormal_, neutralBiomeHydro_;
    gpu::Texture3D black3D_, white3D_;
    gpu::Cubemap   greyCube_;
    gpu::GpuBuffer zeroBuffer_;

    // Inspecteur.
    gpu::ShaderProgram* inspectorProgram_ = nullptr;
    gpu::Framebuffer    inspectorTarget_;
    gpu::VertexArray    emptyVao_;
    std::string         inspectKey_;
    std::string         filter_;
    std::string         inspectorMessage_;
    int                 channel_        = 0;  ///< 0 RGB, 1 R, 2 G, 3 B, 4 A, 5 GBA
    int                 colormap_       = 0;  ///< 0 gris, 1 turbo, 2 SDF divergente, 3 biomes, 4 normale
    float               rangeMin_       = 0.0f;
    float               rangeMax_       = 1.0f;
    float               slice_          = 0.5f;
    float               mip_            = 0.0f;
    float               isoStep_        = 0.0f;
    bool                srgbEncode_     = false;
    bool                showInViewport_ = false;
    bool                inspectorValid_ = false;
    bool                initialized_    = false;
};

}  // namespace env::core
