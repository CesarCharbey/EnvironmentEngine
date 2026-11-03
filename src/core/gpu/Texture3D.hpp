#pragma once
// =============================================================================
//  src/core/gpu/Texture3D.hpp — Texture 3D OpenGL 4.6 DSA (RAII, move-only)
//  Usage typique : bruits volumétriques des nuages, LUTs 3D, froxels.
// =============================================================================
#include <algorithm>
#include <utility>

#include "core/gpu/Texture2D.hpp"

namespace env::gpu {

class Texture3D {
public:
    struct Desc {
        GLsizei     width          = 1;
        GLsizei     height         = 1;
        GLsizei     depth          = 1;
        GLenum      internalFormat = GL_RGBA8;
        GLsizei     levels         = 1;  ///< 0 = chaîne de mips complète
        SamplerDesc sampler{};
        const char* debugName = nullptr;  ///< lu uniquement pendant Create()
    };

    Texture3D() = default;
    explicit Texture3D(const Desc& desc) { Create(desc); }
    ~Texture3D() { Destroy(); }

    Texture3D(const Texture3D&)            = delete;
    Texture3D& operator=(const Texture3D&) = delete;
    Texture3D(Texture3D&& other) noexcept : id_(std::exchange(other.id_, 0)), desc_(other.desc_) {}
    Texture3D& operator=(Texture3D&& other) noexcept {
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
        desc_.depth     = std::max(desc.depth, 1);
        desc_.levels    = desc.levels <= 0 ? MipLevelCount(desc_.width, desc_.height, desc_.depth) : desc.levels;
        desc_.debugName = nullptr;
        glCreateTextures(GL_TEXTURE_3D, 1, &id_);
        glTextureStorage3D(id_, desc_.levels, desc_.internalFormat, desc_.width, desc_.height, desc_.depth);
        detail::ApplySampler(id_, desc_.sampler);
        if (desc.debugName) glObjectLabel(GL_TEXTURE, id_, -1, desc.debugName);
    }

    void Destroy() noexcept {
        if (id_ != 0) {
            glDeleteTextures(1, &id_);
            id_ = 0;
        }
    }

    void Upload(const void* voxels, GLenum format, GLenum type, GLint level = 0) {
        glTextureSubImage3D(id_, level, 0, 0, 0, LevelWidth(level), LevelHeight(level), LevelDepth(level), format,
                            type, voxels);
    }

    void UploadRegion(GLint level, GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d, GLenum format,
                      GLenum type, const void* voxels) {
        glTextureSubImage3D(id_, level, x, y, z, w, h, d, format, type, voxels);
    }

    void GenerateMipmaps() { glGenerateTextureMipmap(id_); }

    void Clear(GLenum format, GLenum type, const void* value = nullptr, GLint level = 0) {
        glClearTexImage(id_, level, format, type, value);
    }

    void BindUnit(GLuint unit) const { glBindTextureUnit(unit, id_); }

    /// layered = GL_TRUE : image3D sur tout le volume ; GL_FALSE : une tranche (image2D) `layer`.
    void BindImage(GLuint unit, GLenum access, GLint level = 0, GLboolean layered = GL_TRUE, GLint layer = 0,
                   GLenum format = 0) const {
        glBindImageTexture(unit, id_, level, layered, layer, access, format != 0 ? format : desc_.internalFormat);
    }

    void Download(GLenum format, GLenum type, GLsizei bufferSize, void* out, GLint level = 0) const {
        glGetTextureImage(id_, level, format, type, bufferSize, out);
    }

    void SetSampler(const SamplerDesc& sampler) {
        desc_.sampler = sampler;
        detail::ApplySampler(id_, sampler);
    }

    GLsizei LevelWidth(GLint level) const noexcept { return std::max(desc_.width >> level, 1); }
    GLsizei LevelHeight(GLint level) const noexcept { return std::max(desc_.height >> level, 1); }
    GLsizei LevelDepth(GLint level) const noexcept { return std::max(desc_.depth >> level, 1); }

    GLuint      Id() const noexcept { return id_; }
    GLsizei     Width() const noexcept { return desc_.width; }
    GLsizei     Height() const noexcept { return desc_.height; }
    GLsizei     Depth() const noexcept { return desc_.depth; }
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
