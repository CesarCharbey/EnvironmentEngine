#pragma once
// =============================================================================
//  src/core/gpu/Cubemap.hpp — Cubemap OpenGL 4.6 DSA (RAII, move-only)
//
//  Faces : 0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z (convention OpenGL).
//  En DSA une cubemap s'adresse comme un tableau de 6 couches : glTextureSubImage3D
//  avec zoffset = face. Pour l'écriture compute de toutes les faces d'un mip :
//  BindImage(unit, access, level, GL_TRUE) -> imageCube / image2DArray côté GLSL.
//  GL_TEXTURE_CUBE_MAP_SEAMLESS est activé globalement par le core.
// =============================================================================
#include <algorithm>
#include <utility>

#include "core/gpu/Texture2D.hpp"

namespace env::gpu {

class Cubemap {
public:
    static constexpr int kFaceCount = 6;

    struct Desc {
        GLsizei     size           = 1;  ///< largeur = hauteur d'une face
        GLenum      internalFormat = GL_RGBA16F;
        GLsizei     levels         = 1;  ///< 0 = chaîne de mips complète (IBL spéculaire préfiltrée)
        SamplerDesc sampler{};
        const char* debugName = nullptr;  ///< lu uniquement pendant Create()
    };

    Cubemap() = default;
    explicit Cubemap(const Desc& desc) { Create(desc); }
    ~Cubemap() { Destroy(); }

    Cubemap(const Cubemap&)            = delete;
    Cubemap& operator=(const Cubemap&) = delete;
    Cubemap(Cubemap&& other) noexcept : id_(std::exchange(other.id_, 0)), desc_(other.desc_) {}
    Cubemap& operator=(Cubemap&& other) noexcept {
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
        desc_.size      = std::max(desc.size, 1);
        desc_.levels    = desc.levels <= 0 ? MipLevelCount(desc_.size, desc_.size) : desc.levels;
        desc_.debugName = nullptr;
        glCreateTextures(GL_TEXTURE_CUBE_MAP, 1, &id_);
        glTextureStorage2D(id_, desc_.levels, desc_.internalFormat, desc_.size, desc_.size);
        detail::ApplySampler(id_, desc_.sampler);
        if (desc.debugName) glObjectLabel(GL_TEXTURE, id_, -1, desc.debugName);
    }

    void Destroy() noexcept {
        if (id_ != 0) {
            glDeleteTextures(1, &id_);
            id_ = 0;
        }
    }

    /// Upload d'une face complète (face dans [0, 5]).
    void UploadFace(int face, const void* pixels, GLenum format, GLenum type, GLint level = 0) {
        const GLsizei s = FaceSize(level);
        glTextureSubImage3D(id_, level, 0, 0, face, s, s, 1, format, type, pixels);
    }

    /// Upload des 6 faces contiguës (face0 | face1 | ... | face5).
    void UploadAllFaces(const void* pixels, GLenum format, GLenum type, GLint level = 0) {
        const GLsizei s = FaceSize(level);
        glTextureSubImage3D(id_, level, 0, 0, 0, s, s, kFaceCount, format, type, pixels);
    }

    void GenerateMipmaps() { glGenerateTextureMipmap(id_); }

    void Clear(GLenum format, GLenum type, const void* value = nullptr, GLint level = 0) {
        glClearTexImage(id_, level, format, type, value);
    }

    void BindUnit(GLuint unit) const { glBindTextureUnit(unit, id_); }

    /// layered = GL_TRUE : les 6 faces du mip (imageCube) ; GL_FALSE : une seule face `face` (image2D).
    void BindImage(GLuint unit, GLenum access, GLint level = 0, GLboolean layered = GL_TRUE, GLint face = 0,
                   GLenum format = 0) const {
        glBindImageTexture(unit, id_, level, layered, face, access, format != 0 ? format : desc_.internalFormat);
    }

    void SetSampler(const SamplerDesc& sampler) {
        desc_.sampler = sampler;
        detail::ApplySampler(id_, sampler);
    }

    GLsizei FaceSize(GLint level) const noexcept { return std::max(desc_.size >> level, 1); }

    GLuint      Id() const noexcept { return id_; }
    GLsizei     Size() const noexcept { return desc_.size; }
    GLsizei     Levels() const noexcept { return desc_.levels; }
    GLenum      InternalFormat() const noexcept { return desc_.internalFormat; }
    const Desc& GetDesc() const noexcept { return desc_; }
    bool        IsValid() const noexcept { return id_ != 0; }
    explicit    operator bool() const noexcept { return id_ != 0; }
    GLuint      Release() noexcept { return std::exchange(id_, 0); }

private:
    GLuint id_ = 0;
    Desc   desc_{};
};

}  // namespace env::gpu
