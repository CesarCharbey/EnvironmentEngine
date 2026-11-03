#pragma once
// =============================================================================
//  src/core/gpu/GpuBuffer.hpp — Buffers OpenGL 4.6 DSA (RAII, move-only)
//
//  GpuBuffer   : stockage immuable glNamedBufferStorage (UBO, SSBO, indirect,
//                vertex, index), écriture glNamedBufferSubData (exige
//                GL_DYNAMIC_STORAGE_BIT), lecture glGetNamedBufferSubData,
//                mapping persistant optionnel (GL_MAP_PERSISTENT_BIT).
//  VertexArray : VAO DSA (glCreateVertexArrays + glVertexArray*).
// =============================================================================
#include <cassert>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <glad/glad.h>

namespace env::gpu {

class GpuBuffer {
public:
    GpuBuffer() = default;
    GpuBuffer(GLsizeiptr size, GLbitfield flags, const void* data = nullptr, const char* debugName = nullptr) {
        Create(size, flags, data, debugName);
    }
    ~GpuBuffer() { Destroy(); }

    GpuBuffer(const GpuBuffer&)            = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
    GpuBuffer(GpuBuffer&& other) noexcept { MoveFrom(other); }
    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) {
            Destroy();
            MoveFrom(other);
        }
        return *this;
    }

    /// Alloue un stockage immuable. flags : GL_DYNAMIC_STORAGE_BIT, GL_MAP_READ_BIT, GL_MAP_WRITE_BIT,
    /// GL_MAP_PERSISTENT_BIT, GL_MAP_COHERENT_BIT, GL_CLIENT_STORAGE_BIT.
    void Create(GLsizeiptr size, GLbitfield flags = GL_DYNAMIC_STORAGE_BIT, const void* data = nullptr,
                const char* debugName = nullptr) {
        Destroy();
        size_  = size > 0 ? size : 4;
        flags_ = flags;
        glCreateBuffers(1, &id_);
        glNamedBufferStorage(id_, size_, data, flags_);
        if (flags_ & GL_MAP_PERSISTENT_BIT) {
            const GLbitfield access = flags_ & (GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
            mapped_ = glMapNamedBufferRange(id_, 0, size_, access);
        }
        if (debugName) glObjectLabel(GL_BUFFER, id_, -1, debugName);
    }

    template <class T>
    void CreateFrom(std::span<const T> data, GLbitfield flags = GL_DYNAMIC_STORAGE_BIT, const char* debugName = nullptr) {
        Create(static_cast<GLsizeiptr>(data.size_bytes()), flags, data.data(), debugName);
    }

    void Destroy() noexcept {
        if (id_ != 0) {
            if (mapped_) glUnmapNamedBuffer(id_);
            glDeleteBuffers(1, &id_);
        }
        id_     = 0;
        size_   = 0;
        flags_  = 0;
        mapped_ = nullptr;
    }

    /// Écriture CPU -> GPU (exige GL_DYNAMIC_STORAGE_BIT).
    void Upload(const void* data, GLsizeiptr size, GLintptr offset = 0) {
        assert((flags_ & GL_DYNAMIC_STORAGE_BIT) && "GpuBuffer::Upload exige GL_DYNAMIC_STORAGE_BIT");
        assert(offset + size <= size_);
        glNamedBufferSubData(id_, offset, size, data);
    }

    template <class T>
    void UploadObject(const T& value, GLintptr offset = 0) {
        Upload(&value, static_cast<GLsizeiptr>(sizeof(T)), offset);
    }

    template <class T>
    void UploadSpan(std::span<const T> values, GLintptr offsetBytes = 0) {
        Upload(values.data(), static_cast<GLsizeiptr>(values.size_bytes()), offsetBytes);
    }

    /// Lecture GPU -> CPU synchrone (bloquante) : outils / debug uniquement.
    void Download(void* out, GLsizeiptr size, GLintptr offset = 0) const {
        glGetNamedBufferSubData(id_, offset, size, out);
    }

    template <class T>
    std::vector<T> DownloadVector(std::size_t count, std::size_t firstElement = 0) const {
        std::vector<T> out(count);
        Download(out.data(), static_cast<GLsizeiptr>(count * sizeof(T)), static_cast<GLintptr>(firstElement * sizeof(T)));
        return out;
    }

    /// Met tout le buffer à zéro côté GPU (taille multiple de 4 octets).
    void ClearZero() {
        const std::uint32_t zero = 0;
        glClearNamedBufferData(id_, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    }

    void ClearRangeZero(GLintptr offset, GLsizeiptr size) {
        const std::uint32_t zero = 0;
        glClearNamedBufferSubData(id_, GL_R32UI, offset, size, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    }

    /// Mapping ponctuel (exige les bits GL_MAP_* correspondants à la création).
    void* Map(GLbitfield access, GLintptr offset = 0, GLsizeiptr length = -1) {
        return glMapNamedBufferRange(id_, offset, length < 0 ? size_ - offset : length, access);
    }
    void Unmap() { glUnmapNamedBuffer(id_); }

    /// Pointeur persistant (non nul si créé avec GL_MAP_PERSISTENT_BIT).
    void* PersistentPointer() const noexcept { return mapped_; }
    void  FlushRange(GLintptr offset, GLsizeiptr length) { glFlushMappedNamedBufferRange(id_, offset, length); }

    /// Binding indexé : GL_UNIFORM_BUFFER, GL_SHADER_STORAGE_BUFFER, GL_ATOMIC_COUNTER_BUFFER.
    void BindBase(GLenum target, GLuint index) const { glBindBufferBase(target, index, id_); }
    void BindRange(GLenum target, GLuint index, GLintptr offset, GLsizeiptr size) const {
        glBindBufferRange(target, index, id_, offset, size);
    }

    /// Binding non indexé, sans équivalent DSA : GL_DRAW_INDIRECT_BUFFER,
    /// GL_DISPATCH_INDIRECT_BUFFER, GL_PARAMETER_BUFFER.
    void Bind(GLenum target) const { glBindBuffer(target, id_); }

    void CopyTo(GpuBuffer& destination, GLsizeiptr size, GLintptr srcOffset = 0, GLintptr dstOffset = 0) const {
        glCopyNamedBufferSubData(id_, destination.id_, srcOffset, dstOffset, size);
    }

    GLuint     Id() const noexcept { return id_; }
    GLsizeiptr Size() const noexcept { return size_; }
    GLbitfield Flags() const noexcept { return flags_; }
    bool       IsValid() const noexcept { return id_ != 0; }
    explicit   operator bool() const noexcept { return id_ != 0; }

private:
    void MoveFrom(GpuBuffer& other) noexcept {
        id_     = std::exchange(other.id_, 0);
        size_   = std::exchange(other.size_, 0);
        flags_  = std::exchange(other.flags_, 0);
        mapped_ = std::exchange(other.mapped_, nullptr);
    }

    GLuint     id_     = 0;
    GLsizeiptr size_   = 0;
    GLbitfield flags_  = 0;
    void*      mapped_ = nullptr;
};

class VertexArray {
public:
    VertexArray() = default;
    ~VertexArray() { Destroy(); }

    VertexArray(const VertexArray&)            = delete;
    VertexArray& operator=(const VertexArray&) = delete;
    VertexArray(VertexArray&& other) noexcept : id_(std::exchange(other.id_, 0)) {}
    VertexArray& operator=(VertexArray&& other) noexcept {
        if (this != &other) {
            Destroy();
            id_ = std::exchange(other.id_, 0);
        }
        return *this;
    }

    void Create(const char* debugName = nullptr) {
        Destroy();
        glCreateVertexArrays(1, &id_);
        if (debugName) glObjectLabel(GL_VERTEX_ARRAY, id_, -1, debugName);
    }

    void Destroy() noexcept {
        if (id_ != 0) {
            glDeleteVertexArrays(1, &id_);
            id_ = 0;
        }
    }

    void SetVertexBuffer(GLuint bindingIndex, const GpuBuffer& buffer, GLintptr offset, GLsizei stride) {
        glVertexArrayVertexBuffer(id_, bindingIndex, buffer.Id(), offset, stride);
    }

    void SetElementBuffer(const GpuBuffer& buffer) { glVertexArrayElementBuffer(id_, buffer.Id()); }

    /// Attribut flottant (type = GL_FLOAT, GL_HALF_FLOAT, GL_UNSIGNED_BYTE + normalized, ...).
    void SetAttribute(GLuint attribIndex, GLuint bindingIndex, GLint components, GLenum type, GLuint relativeOffset,
                      GLboolean normalized = GL_FALSE) {
        glEnableVertexArrayAttrib(id_, attribIndex);
        glVertexArrayAttribFormat(id_, attribIndex, components, type, normalized, relativeOffset);
        glVertexArrayAttribBinding(id_, attribIndex, bindingIndex);
    }

    /// Attribut entier (ivec / uvec côté GLSL).
    void SetAttributeInteger(GLuint attribIndex, GLuint bindingIndex, GLint components, GLenum type,
                             GLuint relativeOffset) {
        glEnableVertexArrayAttrib(id_, attribIndex);
        glVertexArrayAttribIFormat(id_, attribIndex, components, type, relativeOffset);
        glVertexArrayAttribBinding(id_, attribIndex, bindingIndex);
    }

    /// Diviseur d'instanciation d'un binding (0 = par sommet).
    void SetBindingDivisor(GLuint bindingIndex, GLuint divisor) {
        glVertexArrayBindingDivisor(id_, bindingIndex, divisor);
    }

    void Bind() const { glBindVertexArray(id_); }

    GLuint   Id() const noexcept { return id_; }
    bool     IsValid() const noexcept { return id_ != 0; }
    explicit operator bool() const noexcept { return id_ != 0; }

private:
    GLuint id_ = 0;
};

}  // namespace env::gpu
