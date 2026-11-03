#pragma once
// =============================================================================
//  src/core/gpu/Framebuffer.hpp — Framebuffer OpenGL 4.6 DSA (RAII, move-only)
//
//  Deux modes :
//    * Possédant (Create) : alloue N attachements couleur + profondeur optionnelle
//      (G-Buffer, cibles HDR). Resize() réalloue en gardant la description.
//      ATTENTION : les ids de texture changent -> republier dans le blackboard.
//    * Externe (CreateEmpty + Attach*) : attache des textures possédées ailleurs
//      (faces de cubemap pour l'IBL, couches de texture array, mips).
//
//  Reverse-Z (ADR-007) : effacer la profondeur à 0 (ClearDepth(0.0f)).
// =============================================================================
#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

#include "core/gpu/Texture2D.hpp"

namespace env::gpu {

class Framebuffer {
public:
    struct ColorAttachmentDesc {
        GLenum      internalFormat = GL_RGBA16F;
        SamplerDesc sampler{};
        const char* debugName = nullptr;  ///< lu uniquement pendant Create()
    };

    struct Desc {
        GLsizei                          width  = 1;
        GLsizei                          height = 1;
        std::vector<ColorAttachmentDesc> colors;
        GLenum      depthFormat = GL_DEPTH_COMPONENT32F;  ///< 0 = pas de profondeur
        const char* debugName   = nullptr;                ///< lu uniquement pendant Create()
    };

    Framebuffer() = default;
    explicit Framebuffer(const Desc& desc) { Create(desc); }
    ~Framebuffer() { Destroy(); }

    Framebuffer(const Framebuffer&)            = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;
    Framebuffer(Framebuffer&& other) noexcept { MoveFrom(other); }
    Framebuffer& operator=(Framebuffer&& other) noexcept {
        if (this != &other) {
            Destroy();
            MoveFrom(other);
        }
        return *this;
    }

    /// Mode possédant. Retourne false si le framebuffer est incomplet (message sur stderr).
    bool Create(const Desc& desc) {
        Destroy();
        desc_        = desc;
        desc_.width  = std::max(desc.width, 1);
        desc_.height = std::max(desc.height, 1);
        glCreateFramebuffers(1, &id_);
        if (desc.debugName) glObjectLabel(GL_FRAMEBUFFER, id_, -1, desc.debugName);

        colors_.clear();
        colors_.reserve(desc_.colors.size());
        std::vector<GLenum> drawBuffers;
        for (std::size_t i = 0; i < desc_.colors.size(); ++i) {
            const ColorAttachmentDesc& c = desc_.colors[i];
            Texture2D::Desc td;
            td.width          = desc_.width;
            td.height         = desc_.height;
            td.internalFormat = c.internalFormat;
            td.levels         = 1;
            td.sampler        = c.sampler;
            td.debugName      = c.debugName;
            colors_.emplace_back(td);
            const GLenum attachment = GL_COLOR_ATTACHMENT0 + static_cast<GLenum>(i);
            glNamedFramebufferTexture(id_, attachment, colors_.back().Id(), 0);
            drawBuffers.push_back(attachment);
        }

        if (desc_.depthFormat != 0) {
            Texture2D::Desc dd;
            dd.width          = desc_.width;
            dd.height         = desc_.height;
            dd.internalFormat = desc_.depthFormat;
            dd.levels         = 1;
            dd.sampler        = SamplerDesc::Nearest();
            depth_.Create(dd);
            glNamedFramebufferTexture(id_, IsDepthStencilFormat(desc_.depthFormat) ? GL_DEPTH_STENCIL_ATTACHMENT
                                                                                   : GL_DEPTH_ATTACHMENT,
                                      depth_.Id(), 0);
        }

        if (drawBuffers.empty())
            glNamedFramebufferDrawBuffer(id_, GL_NONE);
        else
            glNamedFramebufferDrawBuffers(id_, static_cast<GLsizei>(drawBuffers.size()), drawBuffers.data());

        // Les noms de debug ne survivent pas à l'appel (pointeurs possiblement temporaires).
        desc_.debugName = nullptr;
        for (ColorAttachmentDesc& c : desc_.colors) c.debugName = nullptr;
        return CheckComplete();
    }

    /// Mode externe : framebuffer vide, attachements fournis par l'appelant.
    void CreateEmpty(const char* debugName = nullptr) {
        Destroy();
        glCreateFramebuffers(1, &id_);
        if (debugName) glObjectLabel(GL_FRAMEBUFFER, id_, -1, debugName);
        external_ = true;
    }

    /// Réalloue les attachements possédés à la nouvelle taille. Retourne true si la taille a changé.
    bool Resize(GLsizei width, GLsizei height) {
        width  = std::max(width, 1);
        height = std::max(height, 1);
        if (id_ == 0 || external_ || (width == desc_.width && height == desc_.height)) return false;
        Desc d   = desc_;
        d.width  = width;
        d.height = height;
        Create(d);
        return true;
    }

    void Destroy() noexcept {
        colors_.clear();
        depth_.Destroy();
        if (id_ != 0) {
            glDeleteFramebuffers(1, &id_);
            id_ = 0;
        }
        external_ = false;
    }

    // --- Mode externe ----------------------------------------------------------
    void AttachExternalColor(GLuint index, GLuint texture, GLint level = 0) {
        glNamedFramebufferTexture(id_, GL_COLOR_ATTACHMENT0 + index, texture, level);
    }
    /// Couche d'une texture array / face de cubemap (layer = face) / tranche 3D.
    void AttachExternalColorLayer(GLuint index, GLuint texture, GLint level, GLint layer) {
        glNamedFramebufferTextureLayer(id_, GL_COLOR_ATTACHMENT0 + index, texture, level, layer);
    }
    void AttachExternalDepth(GLuint texture, GLint level = 0) {
        glNamedFramebufferTexture(id_, GL_DEPTH_ATTACHMENT, texture, level);
    }
    void SetDrawBufferCount(GLsizei count) {
        std::vector<GLenum> buffers;
        for (GLsizei i = 0; i < count; ++i) buffers.push_back(GL_COLOR_ATTACHMENT0 + static_cast<GLenum>(i));
        if (buffers.empty())
            glNamedFramebufferDrawBuffer(id_, GL_NONE);
        else
            glNamedFramebufferDrawBuffers(id_, count, buffers.data());
    }

    // --- Utilisation -------------------------------------------------------------
    /// Lie comme cible de rendu et règle le viewport sur la taille possédée.
    void Bind() const {
        glBindFramebuffer(GL_FRAMEBUFFER, id_);
        glViewport(0, 0, desc_.width, desc_.height);
    }
    /// Lie avec un viewport explicite (mode externe, rendu dans un mip).
    void Bind(GLsizei viewportWidth, GLsizei viewportHeight) const {
        glBindFramebuffer(GL_FRAMEBUFFER, id_);
        glViewport(0, 0, viewportWidth, viewportHeight);
    }
    static void BindDefault(GLsizei width, GLsizei height) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, width, height);
    }

    void ClearColor(GLuint index, float r, float g, float b, float a) const {
        const GLfloat value[4] = {r, g, b, a};
        glClearNamedFramebufferfv(id_, GL_COLOR, static_cast<GLint>(index), value);
    }
    void ClearColorUint(GLuint index, GLuint r, GLuint g = 0, GLuint b = 0, GLuint a = 0) const {
        const GLuint value[4] = {r, g, b, a};
        glClearNamedFramebufferuiv(id_, GL_COLOR, static_cast<GLint>(index), value);
    }
    /// Efface la profondeur (Reverse-Z : 0 = infini). Force temporairement glDepthMask(GL_TRUE).
    void ClearDepth(float depth = 0.0f) const {
        GLboolean previousMask = GL_TRUE;
        glGetBooleanv(GL_DEPTH_WRITEMASK, &previousMask);
        glDepthMask(GL_TRUE);
        glClearNamedFramebufferfv(id_, GL_DEPTH, 0, &depth);
        glDepthMask(previousMask);
    }
    void ClearDepthStencil(float depth = 0.0f, GLint stencil = 0) const {
        glClearNamedFramebufferfi(id_, GL_DEPTH_STENCIL, 0, depth, stencil);
    }

    /// Copie vers un autre framebuffer (0 = défaut).
    void BlitTo(GLuint destinationFbo, GLsizei dstWidth, GLsizei dstHeight, GLbitfield mask = GL_COLOR_BUFFER_BIT,
                GLenum filter = GL_LINEAR) const {
        glBlitNamedFramebuffer(id_, destinationFbo, 0, 0, desc_.width, desc_.height, 0, 0, dstWidth, dstHeight, mask,
                               filter);
    }

    bool CheckComplete() const {
        const GLenum status = glCheckNamedFramebufferStatus(id_, GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            std::fprintf(stderr, "[Framebuffer] Incomplet (status 0x%04X)\n", status);
            return false;
        }
        return true;
    }

    Texture2D&       Color(std::size_t index) { return colors_.at(index); }
    const Texture2D& Color(std::size_t index) const { return colors_.at(index); }
    std::size_t      ColorCount() const noexcept { return colors_.size(); }
    Texture2D&       Depth() { return depth_; }
    const Texture2D& Depth() const { return depth_; }
    bool             HasDepth() const noexcept { return depth_.IsValid(); }

    GLuint      Id() const noexcept { return id_; }
    GLsizei     Width() const noexcept { return desc_.width; }
    GLsizei     Height() const noexcept { return desc_.height; }
    const Desc& GetDesc() const noexcept { return desc_; }
    bool        IsValid() const noexcept { return id_ != 0; }

private:
    static bool IsDepthStencilFormat(GLenum format) noexcept {
        return format == GL_DEPTH24_STENCIL8 || format == GL_DEPTH32F_STENCIL8;
    }

    void MoveFrom(Framebuffer& other) noexcept {
        id_       = std::exchange(other.id_, 0);
        desc_     = std::move(other.desc_);
        colors_   = std::move(other.colors_);
        depth_    = std::move(other.depth_);
        external_ = std::exchange(other.external_, false);
    }

    GLuint                 id_ = 0;
    Desc                   desc_{};
    std::vector<Texture2D> colors_;
    Texture2D              depth_;
    bool                   external_ = false;
};

}  // namespace env::gpu
