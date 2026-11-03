#pragma once
// =============================================================================
//  src/core/gpu/Texture2D.hpp — Texture 2D OpenGL 4.6 DSA (RAII, move-only)
//
//  Allocation immuable (glTextureStorage2D), upload glTextureSubImage2D, mipmaps
//  glGenerateTextureMipmap, liaison glBindTextureUnit / glBindImageTexture.
//  Contient aussi les éléments partagés par Texture3D et Cubemap (SamplerDesc,
//  MipLevelCount, ApplySampler).
// =============================================================================
#include <algorithm>
#include <cstdint>
#include <utility>

#include <glad/glad.h>

namespace env::gpu {

/// Nombre de niveaux d'une chaîne de mips complète.
inline GLsizei MipLevelCount(GLsizei width, GLsizei height = 1, GLsizei depth = 1) noexcept {
    GLsizei size   = std::max({width, height, depth, 1});
    GLsizei levels = 1;
    while (size > 1) {
        size >>= 1;
        ++levels;
    }
    return levels;
}

struct SamplerDesc {
    GLenum minFilter     = GL_LINEAR;
    GLenum magFilter     = GL_LINEAR;
    GLenum wrapS         = GL_CLAMP_TO_EDGE;
    GLenum wrapT         = GL_CLAMP_TO_EDGE;
    GLenum wrapR         = GL_CLAMP_TO_EDGE;
    float  maxAnisotropy = 1.0f;  ///< > 1 active le filtrage anisotrope (GL 4.6 core)

    static SamplerDesc Nearest() noexcept {
        SamplerDesc s;
        s.minFilter = GL_NEAREST;
        s.magFilter = GL_NEAREST;
        return s;
    }
    static SamplerDesc Repeat(GLenum filter = GL_LINEAR) noexcept {
        SamplerDesc s;
        s.minFilter = s.magFilter = filter;
        s.wrapS = s.wrapT = s.wrapR = GL_REPEAT;
        return s;
    }
    static SamplerDesc LinearMipmapped(float aniso = 8.0f) noexcept {
        SamplerDesc s = Repeat(GL_LINEAR);
        s.minFilter     = GL_LINEAR_MIPMAP_LINEAR;
        s.maxAnisotropy = aniso;
        return s;
    }
};

namespace detail {
inline void ApplySampler(GLuint texture, const SamplerDesc& s) noexcept {
    glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(s.minFilter));
    glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(s.magFilter));
    glTextureParameteri(texture, GL_TEXTURE_WRAP_S, static_cast<GLint>(s.wrapS));
    glTextureParameteri(texture, GL_TEXTURE_WRAP_T, static_cast<GLint>(s.wrapT));
    glTextureParameteri(texture, GL_TEXTURE_WRAP_R, static_cast<GLint>(s.wrapR));
    if (s.maxAnisotropy > 1.0f) glTextureParameterf(texture, GL_TEXTURE_MAX_ANISOTROPY, s.maxAnisotropy);
}
}  // namespace detail

class Texture2D {
public:
    struct Desc {
        GLsizei     width          = 1;
        GLsizei     height         = 1;
        GLenum      internalFormat = GL_RGBA8;
        GLsizei     levels         = 1;  ///< 0 = chaîne de mips complète
        SamplerDesc sampler{};
        const char* debugName = nullptr;  ///< lu uniquement pendant Create() (glObjectLabel)
    };

    Texture2D() = default;
    explicit Texture2D(const Desc& desc) { Create(desc); }
    ~Texture2D() { Destroy(); }

    Texture2D(const Texture2D&)            = delete;
    Texture2D& operator=(const Texture2D&) = delete;
    Texture2D(Texture2D&& other) noexcept : id_(std::exchange(other.id_, 0)), desc_(other.desc_) {}
    Texture2D& operator=(Texture2D&& other) noexcept {
        if (this != &other) {
            Destroy();
            id_   = std::exchange(other.id_, 0);
            desc_ = other.desc_;
        }
        return *this;
    }

    void Create(const Desc& desc) {
        Destroy();
        desc_           = desc;
        desc_.width     = std::max(desc.width, 1);
        desc_.height    = std::max(desc.height, 1);
        desc_.levels    = desc.levels <= 0 ? MipLevelCount(desc_.width, desc_.height) : desc.levels;
        desc_.debugName = nullptr;
        glCreateTextures(GL_TEXTURE_2D, 1, &id_);
        glTextureStorage2D(id_, desc_.levels, desc_.internalFormat, desc_.width, desc_.height);
        detail::ApplySampler(id_, desc_.sampler);
        if (desc.debugName) glObjectLabel(GL_TEXTURE, id_, -1, desc.debugName);
    }

    void Destroy() noexcept {
        if (id_ != 0) {
            glDeleteTextures(1, &id_);
            id_ = 0;
        }
    }

    /// Upload d'un niveau complet. format/type décrivent les données CPU (ex. GL_RGBA, GL_FLOAT).
    void Upload(const void* pixels, GLenum format, GLenum type, GLint level = 0) {
        glTextureSubImage2D(id_, level, 0, 0, LevelWidth(level), LevelHeight(level), format, type, pixels);
    }

    void UploadRegion(GLint level, GLint x, GLint y, GLsizei w, GLsizei h, GLenum format, GLenum type,
                      const void* pixels) {
        glTextureSubImage2D(id_, level, x, y, w, h, format, type, pixels);
    }

    void GenerateMipmaps() { glGenerateTextureMipmap(id_); }

    /// Remplit un niveau avec une valeur (nullptr = zéro). Ex. Clear(GL_RGBA, GL_FLOAT, color).
    void Clear(GLenum format, GLenum type, const void* value = nullptr, GLint level = 0) {
        glClearTexImage(id_, level, format, type, value);
    }

    void BindUnit(GLuint unit) const { glBindTextureUnit(unit, id_); }

    /// Liaison image pour compute (imageLoad / imageStore). Les formats sRGB ne sont pas supportés.
    void BindImage(GLuint unit, GLenum access, GLint level = 0, GLenum format = 0) const {
        glBindImageTexture(unit, id_, level, GL_FALSE, 0, access, format != 0 ? format : desc_.internalFormat);
    }

    /// Relecture synchrone (bloquante) : réservée aux outils / debug.
    void Download(GLenum format, GLenum type, GLsizei bufferSize, void* out, GLint level = 0) const {
        glGetTextureImage(id_, level, format, type, bufferSize, out);
    }

    void SetSampler(const SamplerDesc& sampler) {
        desc_.sampler = sampler;
        detail::ApplySampler(id_, sampler);
    }

    /// Active la comparaison de profondeur (sampler2DShadow). GL_NONE pour désactiver.
    void SetCompareMode(GLenum compareFunc) {
        glTextureParameteri(id_, GL_TEXTURE_COMPARE_MODE, compareFunc == GL_NONE ? GL_NONE : GL_COMPARE_REF_TO_TEXTURE);
        if (compareFunc != GL_NONE) glTextureParameteri(id_, GL_TEXTURE_COMPARE_FUNC, static_cast<GLint>(compareFunc));
    }

    GLsizei LevelWidth(GLint level) const noexcept { return std::max(desc_.width >> level, 1); }
    GLsizei LevelHeight(GLint level) const noexcept { return std::max(desc_.height >> level, 1); }

    GLuint      Id() const noexcept { return id_; }
    GLsizei     Width() const noexcept { return desc_.width; }
    GLsizei     Height() const noexcept { return desc_.height; }
    GLsizei     Levels() const noexcept { return desc_.levels; }
    GLenum      InternalFormat() const noexcept { return desc_.internalFormat; }
    const Desc& GetDesc() const noexcept { return desc_; }
    bool        IsValid() const noexcept { return id_ != 0; }
    explicit    operator bool() const noexcept { return id_ != 0; }

    /// Abandonne la propriété (l'appelant devient responsable de glDeleteTextures).
    GLuint Release() noexcept { return std::exchange(id_, 0); }

private:
    GLuint id_ = 0;
    Desc   desc_{};
};

}  // namespace env::gpu
