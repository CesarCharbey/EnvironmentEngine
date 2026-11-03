// =============================================================================
//  src/core/RenderBlackboard.cpp
// =============================================================================
#include "core/RenderBlackboard.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include <imgui.h>

#include "contracts/GpuBindings.hpp"
#include "core/gpu/ShaderManager.hpp"

namespace env::core {
namespace {

constexpr std::string_view kFallbackProducer = "fallback";
constexpr GLsizeiptr       kZeroBufferSize   = 64 * 1024;
constexpr int              kInspectorMaxSize = 512;

// --- Shader de l'inspecteur (outil du core, embarqué : pas de fichier dans shaders/) ---
constexpr const char* kInspectorVertex = R"GLSL(#version 460 core
out vec2 vUV;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

constexpr const char* kInspectorFragment = R"GLSL(#version 460 core
#include "common/global_ubo.glsl"
in vec2 vUV;
layout(location = 0) out vec4 oColor;

layout(binding = TEX_CoreInspect2D)   uniform sampler2D   uTex2D;
layout(binding = TEX_CoreInspect3D)   uniform sampler3D   uTex3D;
layout(binding = TEX_CoreInspectCube) uniform samplerCube uTexCube;

uniform int   uKind;      // 0 = 2D, 1 = 3D (tranche), 2 = cubemap (lat-long)
uniform int   uChannel;   // 0 RGB, 1 R, 2 G, 3 B, 4 A, 5 GBA
uniform int   uColormap;  // 0 gris, 1 turbo, 2 SDF divergente, 3 palette biomes, 4 normale
uniform vec2  uRange;
uniform float uSlice;
uniform float uMip;
uniform float uIsoStep;
uniform int   uSrgb;

void main() {
    vec4 s;
    if (uKind == 0) {
        s = textureLod(uTex2D, vUV, uMip);
    } else if (uKind == 1) {
        s = textureLod(uTex3D, vec3(vUV, uSlice), uMip);
    } else {
        float phi = (vUV.x * 2.0 - 1.0) * 3.14159265;
        float lat = (vUV.y - 0.5) * 3.14159265;
        vec3 dir = vec3(cos(lat) * sin(phi), sin(lat), -cos(lat) * cos(phi));
        s = textureLod(uTexCube, dir, uMip);
    }

    float span   = max(uRange.y - uRange.x, 1e-6);
    float scalar = 0.0;
    vec3  rgb;
    if (uChannel == 0 || uChannel == 5) {
        vec3 v = (uChannel == 0) ? s.rgb : s.gba;
        rgb    = (uColormap == 4) ? v * 0.5 + 0.5 : (v - uRange.x) / span;
        scalar = v.x;
    } else {
        scalar  = s[uChannel - 1];
        float t = (scalar - uRange.x) / span;
        if (uColormap == 0)      rgb = vec3(clamp(t, 0.0, 1.0));
        else if (uColormap == 1) rgb = envTurbo(t);
        else if (uColormap == 2) rgb = envDivergingSdf(scalar, max(max(abs(uRange.x), abs(uRange.y)), 1e-3));
        else if (uColormap == 3) rgb = envBiomeDebugColor(int(scalar + 0.5));
        else                     rgb = vec3(scalar * 0.5 + 0.5);
    }

    if (uIsoStep > 0.0) {
        float f = scalar / uIsoStep;
        float d = abs(fract(f + 0.5) - 0.5) / max(fwidth(f), 1e-5);
        rgb = mix(rgb * 0.55, rgb, smoothstep(0.0, 1.0, d));
    }
    rgb = clamp(rgb, 0.0, 1.0);
    if (uSrgb == 1) rgb = pow(rgb, vec3(1.0 / 2.2));
    oColor = vec4(rgb, 1.0);
}
)GLSL";

const char* KindName(bb::Kind kind) noexcept {
    switch (kind) {
        case bb::Kind::Texture2D: return "Tex2D";
        case bb::Kind::Texture3D: return "Tex3D";
        case bb::Kind::Cubemap:   return "Cube";
        case bb::Kind::Buffer:    return "Buffer";
    }
    return "?";
}

const char* TargetName(GLenum target) noexcept {
    switch (target) {
        case GL_TEXTURE_2D:       return "Tex2D";
        case GL_TEXTURE_3D:       return "Tex3D";
        case GL_TEXTURE_CUBE_MAP: return "Cube";
        default:                  return "Tex?";
    }
}

const char* FallbackName(bb::Fallback f) noexcept {
    switch (f) {
        case bb::Fallback::Checker:           return "Checker";
        case bb::Fallback::Black:             return "Black";
        case bb::Fallback::White:             return "White";
        case bb::Fallback::FlatNormalTS:      return "FlatNormalTS";
        case bb::Fallback::FlatNormalWS:      return "FlatNormalWS";
        case bb::Fallback::FlatHeightNormal:  return "FlatHeightNormal";
        case bb::Fallback::NeutralBiomeHydro: return "NeutralBiomeHydro";
        case bb::Fallback::Black3D:           return "Black3D";
        case bb::Fallback::White3D:           return "White3D";
        case bb::Fallback::GreyCube:          return "GreyCube";
        case bb::Fallback::ZeroBuffer:        return "ZeroBuffer";
        case bb::Fallback::Count:             break;
    }
    return "?";
}

const char* FormatName(GLenum format) noexcept {
    switch (format) {
        case GL_RGBA8:                return "RGBA8";
        case GL_SRGB8_ALPHA8:         return "SRGB8_A8";
        case GL_RGBA16F:              return "RGBA16F";
        case GL_RGBA32F:              return "RGBA32F";
        case GL_RG16F:                return "RG16F";
        case GL_RG32F:                return "RG32F";
        case GL_R16F:                 return "R16F";
        case GL_R32F:                 return "R32F";
        case GL_R8:                   return "R8";
        case GL_RG8:                  return "RG8";
        case GL_R11F_G11F_B10F:       return "R11G11B10F";
        case GL_RGB10_A2:             return "RGB10A2";
        case GL_DEPTH_COMPONENT32F:   return "DEPTH32F";
        case GL_DEPTH_COMPONENT24:    return "DEPTH24";
        case GL_DEPTH24_STENCIL8:     return "D24S8";
        case GL_DEPTH32F_STENCIL8:    return "D32FS8";
        case GL_R32UI:                return "R32UI";
        case GL_RG32UI:               return "RG32UI";
        case GL_RGBA32UI:             return "RGBA32UI";
        case GL_R32I:                 return "R32I";
        case GL_R8UI:                 return "R8UI";
        case GL_R16UI:                return "R16UI";
        default:                      return "?";
    }
}

bool IsIntegerFormat(GLenum format) noexcept {
    switch (format) {
        case GL_R8UI: case GL_R16UI: case GL_R32UI: case GL_RG8UI: case GL_RG16UI: case GL_RG32UI:
        case GL_RGBA8UI: case GL_RGBA16UI: case GL_RGBA32UI: case GL_R8I: case GL_R16I: case GL_R32I:
        case GL_RG8I: case GL_RG16I: case GL_RG32I: case GL_RGBA8I: case GL_RGBA16I: case GL_RGBA32I:
            return true;
        default:
            return false;
    }
}

bool IsDepthFormat(GLenum format) noexcept {
    return format == GL_DEPTH_COMPONENT32F || format == GL_DEPTH_COMPONENT24 || format == GL_DEPTH_COMPONENT16 ||
           format == GL_DEPTH24_STENCIL8 || format == GL_DEPTH32F_STENCIL8;
}

gpu::Texture2D MakeSolid2D(GLenum internalFormat, GLsizei size, const gpu::SamplerDesc& sampler, const char* name) {
    gpu::Texture2D::Desc d;
    d.width          = size;
    d.height         = size;
    d.internalFormat = internalFormat;
    d.levels         = 1;
    d.sampler        = sampler;
    d.debugName      = name;
    return gpu::Texture2D(d);
}

gpu::Texture3D MakeSolid3D(GLsizei size, const char* name) {
    gpu::Texture3D::Desc d;
    d.width          = size;
    d.height         = size;
    d.depth          = size;
    d.internalFormat = GL_RGBA8;
    d.levels         = 1;
    d.sampler        = gpu::SamplerDesc::Repeat(GL_LINEAR);
    d.debugName      = name;
    return gpu::Texture3D(d);
}

TextureRef MakeRef(const gpu::Texture2D& t) {
    TextureRef r;
    r.id             = t.Id();
    r.target         = GL_TEXTURE_2D;
    r.internalFormat = t.InternalFormat();
    r.size           = {t.Width(), t.Height(), 1};
    r.levels         = t.Levels();
    r.isFallback     = true;
    r.producer       = kFallbackProducer;
    return r;
}

TextureRef MakeRef(const gpu::Texture3D& t) {
    TextureRef r;
    r.id             = t.Id();
    r.target         = GL_TEXTURE_3D;
    r.internalFormat = t.InternalFormat();
    r.size           = {t.Width(), t.Height(), t.Depth()};
    r.levels         = t.Levels();
    r.isFallback     = true;
    r.producer       = kFallbackProducer;
    return r;
}

TextureRef MakeRef(const gpu::Cubemap& t) {
    TextureRef r;
    r.id             = t.Id();
    r.target         = GL_TEXTURE_CUBE_MAP;
    r.internalFormat = t.InternalFormat();
    r.size           = {t.Size(), t.Size(), 6};
    r.levels         = t.Levels();
    r.isFallback     = true;
    r.producer       = kFallbackProducer;
    return r;
}

}  // namespace

// =============================================================================
// Cycle de vie
// =============================================================================
void RenderBlackboard::Init(gpu::ShaderManager& shaders) {
    // Damier "texture manquante" 64x64, cases de 8 px.
    {
        std::vector<std::uint8_t> pixels(64 * 64 * 4);
        for (int y = 0; y < 64; ++y)
            for (int x = 0; x < 64; ++x) {
                const bool    odd = (((x / 8) + (y / 8)) & 1) != 0;
                std::uint8_t* p   = &pixels[static_cast<std::size_t>((y * 64 + x) * 4)];
                p[0] = odd ? 255 : 24;
                p[1] = odd ? 0 : 24;
                p[2] = odd ? 255 : 24;
                p[3] = 255;
            }
        checker_ = MakeSolid2D(GL_RGBA8, 64, gpu::SamplerDesc::Repeat(GL_NEAREST), "Fallback.Checker");
        checker_.Upload(pixels.data(), GL_RGBA, GL_UNSIGNED_BYTE);
    }

    auto fill8 = [](gpu::Texture2D& t, std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) {
        const std::uint8_t c[4] = {r, g, b, a};
        t.Clear(GL_RGBA, GL_UNSIGNED_BYTE, c);
    };
    auto fillF = [](auto& t, float r, float g, float b, float a) {
        const float c[4] = {r, g, b, a};
        t.Clear(GL_RGBA, GL_FLOAT, c);
    };

    black_ = MakeSolid2D(GL_RGBA8, 4, gpu::SamplerDesc::Repeat(GL_NEAREST), "Fallback.Black");
    fill8(black_, 0, 0, 0, 0);
    white_ = MakeSolid2D(GL_RGBA8, 4, gpu::SamplerDesc::Repeat(GL_NEAREST), "Fallback.White");
    fill8(white_, 255, 255, 255, 255);
    flatNormalTS_ = MakeSolid2D(GL_RGBA8, 4, gpu::SamplerDesc::Repeat(GL_NEAREST), "Fallback.FlatNormalTS");
    fill8(flatNormalTS_, 128, 128, 255, 255);
    flatNormalWS_ = MakeSolid2D(GL_RGBA16F, 4, gpu::SamplerDesc::Repeat(GL_NEAREST), "Fallback.FlatNormalWS");
    fillF(flatNormalWS_, 0.0f, 1.0f, 0.0f, 0.0f);
    flatHeightNormal_ = MakeSolid2D(GL_RGBA32F, 4, gpu::SamplerDesc::Nearest(), "Fallback.FlatHeightNormal");
    fillF(flatHeightNormal_, 0.0f, 0.0f, 1.0f, 0.0f);
    neutralBiomeHydro_ = MakeSolid2D(GL_RGBA16F, 4, gpu::SamplerDesc::Nearest(), "Fallback.NeutralBiomeHydro");
    fillF(neutralBiomeHydro_, static_cast<float>(BiomeType::Plains), kRiverSdfFar, kRiverSdfFar, 0.5f);

    black3D_ = MakeSolid3D(4, "Fallback.Black3D");
    fillF(black3D_, 0.0f, 0.0f, 0.0f, 0.0f);
    white3D_ = MakeSolid3D(4, "Fallback.White3D");
    fillF(white3D_, 1.0f, 1.0f, 1.0f, 1.0f);

    {
        gpu::Cubemap::Desc d;
        d.size           = 4;
        d.internalFormat = GL_RGBA16F;
        d.levels         = 1;
        d.debugName      = "Fallback.GreyCube";
        greyCube_.Create(d);
        fillF(greyCube_, 0.5f, 0.5f, 0.5f, 1.0f);
    }

    {
        const std::vector<std::uint8_t> zeros(static_cast<std::size_t>(kZeroBufferSize), 0);
        zeroBuffer_.Create(kZeroBufferSize, GL_DYNAMIC_STORAGE_BIT, zeros.data(), "Fallback.ZeroBuffer");
        zeroBufferRef_ = BufferRef{zeroBuffer_.Id(), zeroBuffer_.Size(), true, kFallbackProducer};
    }

    auto set = [this](bb::Fallback f, TextureRef ref) { fallbackRefs_[static_cast<std::size_t>(f)] = ref; };
    set(bb::Fallback::Checker, MakeRef(checker_));
    set(bb::Fallback::Black, MakeRef(black_));
    set(bb::Fallback::White, MakeRef(white_));
    set(bb::Fallback::FlatNormalTS, MakeRef(flatNormalTS_));
    set(bb::Fallback::FlatNormalWS, MakeRef(flatNormalWS_));
    set(bb::Fallback::FlatHeightNormal, MakeRef(flatHeightNormal_));
    set(bb::Fallback::NeutralBiomeHydro, MakeRef(neutralBiomeHydro_));
    set(bb::Fallback::Black3D, MakeRef(black3D_));
    set(bb::Fallback::White3D, MakeRef(white3D_));
    set(bb::Fallback::GreyCube, MakeRef(greyCube_));
    set(bb::Fallback::ZeroBuffer, MakeRef(black_));  // jamais servi comme texture : sécurité

    emptyVao_.Create("Blackboard.InspectorVAO");
    std::vector<gpu::ShaderStageSource> stages(2);
    stages[0].stage        = gpu::ShaderStage::Vertex;
    stages[0].inlineSource = kInspectorVertex;
    stages[1].stage        = gpu::ShaderStage::Fragment;
    stages[1].inlineSource = kInspectorFragment;
    inspectorProgram_      = &shaders.LoadProgram("core.blackboard_inspector", std::move(stages));
    initialized_           = true;
}

void RenderBlackboard::Shutdown() {
    entries_.clear();
    requested_.clear();
    inspectorTarget_.Destroy();
    emptyVao_.Destroy();
    checker_.Destroy();
    black_.Destroy();
    white_.Destroy();
    flatNormalTS_.Destroy();
    flatNormalWS_.Destroy();
    flatHeightNormal_.Destroy();
    neutralBiomeHydro_.Destroy();
    black3D_.Destroy();
    white3D_.Destroy();
    greyCube_.Destroy();
    zeroBuffer_.Destroy();
    fallbackRefs_.fill(TextureRef{});
    zeroBufferRef_    = BufferRef{};
    inspectorProgram_ = nullptr;
    initialized_      = false;
}

void RenderBlackboard::BeginFrame() {
    // Si une passe a écrit par erreur dans le fallback, on le remet à zéro chaque frame.
    if (zeroBuffer_.IsValid()) zeroBuffer_.ClearZero();
}

// =============================================================================
// Publication
// =============================================================================
void RenderBlackboard::PublishTexture(std::string_view key, const gpu::Texture2D& t, std::string_view producer) {
    PublishTextureRaw(key, t.Id(), GL_TEXTURE_2D, t.InternalFormat(), {t.Width(), t.Height(), 1}, t.Levels(), producer);
}

void RenderBlackboard::PublishTexture(std::string_view key, const gpu::Texture3D& t, std::string_view producer) {
    PublishTextureRaw(key, t.Id(), GL_TEXTURE_3D, t.InternalFormat(), {t.Width(), t.Height(), t.Depth()}, t.Levels(),
                      producer);
}

void RenderBlackboard::PublishTexture(std::string_view key, const gpu::Cubemap& t, std::string_view producer) {
    PublishTextureRaw(key, t.Id(), GL_TEXTURE_CUBE_MAP, t.InternalFormat(), {t.Size(), t.Size(), 6}, t.Levels(),
                      producer);
}

void RenderBlackboard::PublishTextureRaw(std::string_view key, GLuint id, GLenum target, GLenum internalFormat,
                                         glm::ivec3 size, GLint levels, std::string_view producer) {
    if (id == 0) {
        Unpublish(key);
        return;
    }
    auto it = entries_.find(key);
    if (it == entries_.end()) it = entries_.emplace(std::string(key), Entry{}).first;
    Entry& e   = it->second;
    e.producer = std::string(producer);
    e.isBuffer = false;
    e.texture  = TextureRef{id, target, internalFormat, size, levels, false, e.producer};
    e.buffer   = BufferRef{};
    ++e.publishCount;
}

void RenderBlackboard::PublishBuffer(std::string_view key, const gpu::GpuBuffer& buffer, std::string_view producer) {
    PublishBufferRaw(key, buffer.Id(), buffer.Size(), producer);
}

void RenderBlackboard::PublishBufferRaw(std::string_view key, GLuint id, GLsizeiptr size, std::string_view producer) {
    if (id == 0) {
        Unpublish(key);
        return;
    }
    auto it = entries_.find(key);
    if (it == entries_.end()) it = entries_.emplace(std::string(key), Entry{}).first;
    Entry& e   = it->second;
    e.producer = std::string(producer);
    e.isBuffer = true;
    e.buffer   = BufferRef{id, size, false, e.producer};
    e.texture  = TextureRef{};
    ++e.publishCount;
}

void RenderBlackboard::Unpublish(std::string_view key) {
    if (auto it = entries_.find(key); it != entries_.end()) entries_.erase(it);
}

void RenderBlackboard::UnpublishIfProducer(std::string_view key, std::string_view producer) {
    if (auto it = entries_.find(key); it != entries_.end() && it->second.producer == producer) entries_.erase(it);
}

// =============================================================================
// Consommation
// =============================================================================
void RenderBlackboard::TrackRequest(std::string_view key) const {
    if (requested_.find(key) == requested_.end()) requested_.emplace(key);
}

TextureRef RenderBlackboard::ResolveTexture(std::string_view key, GLenum expectedTarget, bool track) const {
    if (track) TrackRequest(key);
    if (auto it = entries_.find(key); it != entries_.end() && !it->second.isBuffer) return it->second.texture;

    if (const bb::KeyInfo* info = bb::FindKey(key)) return FallbackTexture(info->fallback);
    switch (expectedTarget) {
        case GL_TEXTURE_3D:       return FallbackTexture(bb::Fallback::Black3D);
        case GL_TEXTURE_CUBE_MAP: return FallbackTexture(bb::Fallback::GreyCube);
        default:                  return FallbackTexture(bb::Fallback::Checker);
    }
}

TextureRef RenderBlackboard::GetTexture(std::string_view key) const { return ResolveTexture(key, GL_TEXTURE_2D, true); }

TextureRef RenderBlackboard::GetTexture(std::string_view key, GLenum expectedTarget) const {
    return ResolveTexture(key, expectedTarget, true);
}

BufferRef RenderBlackboard::GetBuffer(std::string_view key) const {
    TrackRequest(key);
    if (auto it = entries_.find(key); it != entries_.end() && it->second.isBuffer) return it->second.buffer;
    return zeroBufferRef_;
}

bool RenderBlackboard::IsPublished(std::string_view key) const { return entries_.find(key) != entries_.end(); }

GLuint RenderBlackboard::BindTexture(std::string_view key, GLuint unit) const {
    const TextureRef ref = GetTexture(key);
    glBindTextureUnit(unit, ref.id);
    return ref.id;
}

GLuint RenderBlackboard::BindBuffer(std::string_view key, GLenum target, GLuint index) const {
    const BufferRef ref = GetBuffer(key);
    glBindBufferBase(target, index, ref.id);
    return ref.id;
}

const TextureRef& RenderBlackboard::FallbackTexture(bb::Fallback fallback) const {
    const auto index = static_cast<std::size_t>(fallback);
    return fallbackRefs_[index < fallbackRefs_.size() ? index : 0];
}

// =============================================================================
// Inspecteur
// =============================================================================
GLuint RenderBlackboard::InspectorOutputTexture() const noexcept {
    return (inspectorValid_ && inspectorTarget_.ColorCount() > 0) ? inspectorTarget_.Color(0).Id() : 0;
}

void RenderBlackboard::Inspect(std::string_view key) {
    inspectKey_       = std::string(key);
    inspectorMessage_.clear();
    mip_              = 0.0f;
}

void RenderBlackboard::RenderInspector() {
    inspectorValid_ = false;
    if (!initialized_ || inspectKey_.empty() || !inspectorProgram_ || !inspectorProgram_->IsValid()) return;
    if (entries_.count(inspectKey_) != 0 && entries_.find(inspectKey_)->second.isBuffer) return;
    if (const bb::KeyInfo* info = bb::FindKey(inspectKey_); info && info->kind == bb::Kind::Buffer) return;

    const TextureRef ref = ResolveTexture(inspectKey_, GL_TEXTURE_2D, false);
    if (IsIntegerFormat(ref.internalFormat)) return;

    // Taille de sortie : on conserve le rapport d'aspect (2:1 pour le dépliage lat-long d'une cubemap).
    float aspect = ref.target == GL_TEXTURE_CUBE_MAP ? 2.0f
                                                     : static_cast<float>(std::max(ref.size.x, 1)) /
                                                           static_cast<float>(std::max(ref.size.y, 1));
    aspect            = std::clamp(aspect, 0.125f, 8.0f);
    const int outW    = aspect >= 1.0f ? kInspectorMaxSize : std::max(8, static_cast<int>(kInspectorMaxSize * aspect));
    const int outH    = aspect >= 1.0f ? std::max(8, static_cast<int>(kInspectorMaxSize / aspect)) : kInspectorMaxSize;
    if (!inspectorTarget_.IsValid()) {
        gpu::Framebuffer::Desc d;
        d.width  = outW;
        d.height = outH;
        gpu::Framebuffer::ColorAttachmentDesc color;
        color.internalFormat = GL_RGBA8;
        color.debugName      = "Blackboard.InspectorOutput";
        d.colors.push_back(color);
        d.depthFormat = 0;
        d.debugName   = "Blackboard.InspectorFBO";
        inspectorTarget_.Create(d);
    } else {
        inspectorTarget_.Resize(outW, outH);
    }

    int kind = 0;
    glBindTextureUnit(bind::tex::CoreInspect2D, black_.Id());
    glBindTextureUnit(bind::tex::CoreInspect3D, black3D_.Id());
    glBindTextureUnit(bind::tex::CoreInspectCube, greyCube_.Id());
    if (ref.target == GL_TEXTURE_3D) {
        kind = 1;
        glBindTextureUnit(bind::tex::CoreInspect3D, ref.id);
    } else if (ref.target == GL_TEXTURE_CUBE_MAP) {
        kind = 2;
        glBindTextureUnit(bind::tex::CoreInspectCube, ref.id);
    } else {
        glBindTextureUnit(bind::tex::CoreInspect2D, ref.id);
    }

    inspectorTarget_.Bind();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    const gpu::ShaderProgram& p = *inspectorProgram_;
    p.Set("uKind", kind);
    p.Set("uChannel", channel_);
    p.Set("uColormap", colormap_);
    p.Set("uRange", glm::vec2(rangeMin_, rangeMax_));
    p.Set("uSlice", slice_);
    p.Set("uMip", std::clamp(mip_, 0.0f, static_cast<float>(std::max(ref.levels - 1, 0))));
    p.Set("uIsoStep", isoStep_);
    p.Set("uSrgb", srgbEncode_ ? 1 : 0);
    p.Use();
    emptyVao_.Bind();
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glUseProgram(0);
    glEnable(GL_DEPTH_TEST);
    inspectorValid_ = true;
}

void RenderBlackboard::ApplyPreset(int preset) {
    const TextureRef ref = ResolveTexture(inspectKey_, GL_TEXTURE_2D, false);
    isoStep_             = 0.0f;
    switch (preset) {
        case 0:  // Hauteur (R)
            channel_  = 1;
            colormap_ = 1;
            AutoRange(ref);
            isoStep_ = 5.0f;
            break;
        case 1:  // Normale (GBA de heightNormalTex)
            channel_  = 5;
            colormap_ = 4;
            break;
        case 2:  // Biome (R)
            channel_  = 1;
            colormap_ = 3;
            rangeMin_ = 0.0f;
            rangeMax_ = static_cast<float>(kBiomeCount - 1);
            break;
        case 3:  // SDF rivière (G)
            channel_  = 2;
            colormap_ = 2;
            rangeMin_ = -30.0f;
            rangeMax_ = 30.0f;
            isoStep_  = 5.0f;
            break;
        case 4:  // SDF marais (B)
            channel_  = 3;
            colormap_ = 2;
            rangeMin_ = -30.0f;
            rangeMax_ = 30.0f;
            isoStep_  = 5.0f;
            break;
        case 5:  // Humidité (A)
            channel_  = 4;
            colormap_ = 1;
            rangeMin_ = 0.0f;
            rangeMax_ = 1.0f;
            break;
        case 6:  // Profondeur reverse-Z (R)
            channel_  = 1;
            colormap_ = 0;
            AutoRange(ref);
            break;
        default:
            break;
    }
}

void RenderBlackboard::AutoRange(const TextureRef& ref) {
    if (ref.target != GL_TEXTURE_2D || IsIntegerFormat(ref.internalFormat) || ref.id == 0) {
        inspectorMessage_ = "Auto range : textures 2D flottantes uniquement";
        return;
    }
    // Choisit un mip <= 256 px si disponible pour limiter le coût de la relecture.
    GLint level = 0;
    while (level + 1 < ref.levels && std::max(ref.size.x >> level, ref.size.y >> level) > 256) ++level;
    const int  w     = std::max(ref.size.x >> level, 1);
    const int  h     = std::max(ref.size.y >> level, 1);
    const bool depth = IsDepthFormat(ref.internalFormat);
    const int  comps = depth ? 1 : 4;
    std::vector<float> data(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * static_cast<std::size_t>(comps));
    glGetTextureImage(ref.id, level, depth ? GL_DEPTH_COMPONENT : GL_RGBA, GL_FLOAT,
                      static_cast<GLsizei>(data.size() * sizeof(float)), data.data());

    float lo = std::numeric_limits<float>::max();
    float hi = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < data.size(); i += static_cast<std::size_t>(comps)) {
        int first = 0, last = 0;
        if (!depth) {
            if (channel_ == 0) { first = 0; last = 2; }
            else if (channel_ == 5) { first = 1; last = 3; }
            else { first = last = channel_ - 1; }
        }
        for (int c = first; c <= last; ++c) {
            const float v = data[i + static_cast<std::size_t>(c)];
            if (!std::isfinite(v) || std::fabs(v) >= kRiverSdfFar * 0.999f) continue;  // ignore "loin"
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
    }
    if (lo > hi) {
        inspectorMessage_ = "Auto range : aucune valeur exploitable";
        return;
    }
    if (hi - lo < 1e-6f) hi = lo + 1e-3f;
    rangeMin_ = lo;
    rangeMax_ = hi;
    if (colormap_ == 2) {
        const float m = std::max(std::fabs(lo), std::fabs(hi));
        rangeMin_     = -m;
        rangeMax_     = m;
    }
    char buffer[128];
    std::snprintf(buffer, sizeof(buffer), "Auto range (mip %d, %dx%d) : [%.4g, %.4g]", level, w, h, lo, hi);
    inspectorMessage_ = buffer;
}

void RenderBlackboard::OnGui() {
    ImGui::TextDisabled("Bus non-proprietaire : publie = ressource du producteur, sinon fallback du contrat.");
    std::size_t published = 0, fallback = 0;
    for (const bb::KeyInfo& info : bb::kKeys) (IsPublished(info.key) ? published : fallback)++;
    ImGui::Text("%zu cle(s) publiee(s), %zu en fallback, %zu cle(s) hors contrat", published, fallback,
                entries_.size() > published ? entries_.size() - published : std::size_t{0});
    DrawKeyTable();
    if (!inspectKey_.empty()) {
        ImGui::Separator();
        DrawInspector();
    }
}

void RenderBlackboard::DrawKeyTable() {
    char filter[64] = {};
    std::snprintf(filter, sizeof(filter), "%s", filter_.c_str());
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::InputTextWithHint("##bbfilter", "Filtrer les cles (ex. world, gbuffer, ibl)", filter, sizeof(filter)))
        filter_ = filter;

    struct Row {
        std::string      key;
        std::string_view kind;
        std::string      producer;
        int              state;  // 0 publiee, 1 fallback, 2 demandee hors contrat
        std::string      detail;
    };
    std::vector<Row> rows;
    auto matches = [this](std::string_view key) { return filter_.empty() || key.find(filter_) != std::string_view::npos; };

    for (const bb::KeyInfo& info : bb::kKeys) {
        if (!matches(info.key)) continue;
        Row row{std::string(info.key), KindName(info.kind), std::string(bb::ProducerName(info.producer)), 1, ""};
        if (auto it = entries_.find(info.key); it != entries_.end()) {
            row.state    = 0;
            row.producer = it->second.producer;
            const Entry& e = it->second;
            char buffer[96];
            if (e.isBuffer)
                std::snprintf(buffer, sizeof(buffer), "id %u, %lld o", e.buffer.id, static_cast<long long>(e.buffer.size));
            else
                std::snprintf(buffer, sizeof(buffer), "id %u, %s %dx%dx%d", e.texture.id, FormatName(e.texture.internalFormat),
                              e.texture.size.x, e.texture.size.y, e.texture.size.z);
            row.detail = buffer;
        } else {
            row.detail = std::string(info.format) + " -> " + FallbackName(info.fallback);
        }
        rows.push_back(std::move(row));
    }
    for (const auto& [key, e] : entries_) {
        if (bb::FindKey(key) || !matches(key)) continue;
        char buffer[96];
        if (e.isBuffer)
            std::snprintf(buffer, sizeof(buffer), "id %u, %lld o", e.buffer.id, static_cast<long long>(e.buffer.size));
        else
            std::snprintf(buffer, sizeof(buffer), "id %u, %s %dx%dx%d", e.texture.id, FormatName(e.texture.internalFormat),
                          e.texture.size.x, e.texture.size.y, e.texture.size.z);
        rows.push_back(Row{key, e.isBuffer ? "Buffer" : TargetName(e.texture.target), e.producer, 0, buffer});
    }
    for (const std::string& key : requested_) {
        if (bb::FindKey(key) || entries_.count(key) || !matches(key)) continue;
        rows.push_back(Row{key, "?", "?", 2, "demandee mais ni declaree ni publiee -> Checker"});
    }

    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY |
                                  ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_Resizable;
    if (ImGui::BeginTable("##bbkeys", 5, flags, ImVec2(0.0f, ImGui::GetTextLineHeightWithSpacing() * 12.0f))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Cle", ImGuiTableColumnFlags_WidthStretch, 2.2f);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthStretch, 0.6f);
        ImGui::TableSetupColumn("Producteur", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("Etat", ImGuiTableColumnFlags_WidthStretch, 0.8f);
        ImGui::TableSetupColumn("Detail", ImGuiTableColumnFlags_WidthStretch, 2.4f);
        ImGui::TableHeadersRow();
        for (const Row& row : rows) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            const bool selected = row.key == inspectKey_;
            if (ImGui::Selectable(row.key.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns)) Inspect(row.key);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(row.kind.data(), row.kind.data() + row.kind.size());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(row.producer.c_str());
            ImGui::TableNextColumn();
            if (row.state == 0)
                ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "PUBLIEE");
            else if (row.state == 1)
                ImGui::TextColored(ImVec4(0.95f, 0.85f, 0.3f, 1.0f), requested_.count(row.key) ? "FALLBACK*" : "FALLBACK");
            else
                ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.2f, 1.0f), "DEMANDEE");
            if (row.state == 1 && requested_.count(row.key) && ImGui::IsItemHovered())
                ImGui::SetTooltip("Un consommateur lit cette cle : il recoit le fallback");
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(row.detail.c_str());
        }
        ImGui::EndTable();
    }
}

void RenderBlackboard::DrawInspector() {
    const TextureRef      ref  = ResolveTexture(inspectKey_, GL_TEXTURE_2D, false);
    const bb::KeyInfo*    info = bb::FindKey(inspectKey_);
    const bool isBuffer = (info && info->kind == bb::Kind::Buffer) ||
                          (entries_.count(inspectKey_) && entries_.find(inspectKey_)->second.isBuffer);

    ImGui::Text("Inspecteur : %s", inspectKey_.c_str());
    ImGui::SameLine();
    if (ImGui::SmallButton("Fermer")) {
        inspectKey_.clear();
        showInViewport_ = false;
        return;
    }
    if (isBuffer) {
        const BufferRef b = GetBuffer(inspectKey_);
        ImGui::Text("Buffer id %u, %lld octets%s", b.id, static_cast<long long>(b.size), b.isFallback ? " (fallback ZeroBuffer)" : "");
        return;
    }
    ImGui::Text("%s id %u, %s %dx%dx%d, %d mip(s)%s", TargetName(ref.target), ref.id, FormatName(ref.internalFormat),
                ref.size.x, ref.size.y, ref.size.z, ref.levels, ref.isFallback ? "  [FALLBACK]" : "");
    if (IsIntegerFormat(ref.internalFormat)) {
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.2f, 1.0f), "Format entier : non visualisable ici");
        return;
    }

    ImGui::TextDisabled("Presets :");
    static const char* kPresets[] = {"Height", "Normal", "Biome", "River SDF", "Marsh SDF", "Moisture", "Depth"};
    for (int i = 0; i < 7; ++i) {
        ImGui::SameLine();
        if (ImGui::SmallButton(kPresets[i])) ApplyPreset(i);
    }

    static const char* kChannels[]  = {"RGB", "R", "G", "B", "A", "GBA (normale)"};
    static const char* kColormaps[] = {"Grey", "Turbo", "Diverging SDF", "Biome palette", "Normal (x*0.5+0.5)"};
    ImGui::SetNextItemWidth(140.0f);
    ImGui::Combo("Canal", &channel_, kChannels, IM_ARRAYSIZE(kChannels));
    ImGui::SameLine();
    ImGui::SetNextItemWidth(170.0f);
    ImGui::Combo("Colormap", &colormap_, kColormaps, IM_ARRAYSIZE(kColormaps));
    float range[2] = {rangeMin_, rangeMax_};
    ImGui::SetNextItemWidth(220.0f);
    if (ImGui::DragFloat2("Plage", range, 0.05f)) {
        rangeMin_ = range[0];
        rangeMax_ = range[1];
    }
    ImGui::SameLine();
    if (ImGui::Button("Auto range")) AutoRange(ref);
    ImGui::SetNextItemWidth(140.0f);
    ImGui::DragFloat("Isolignes (pas)", &isoStep_, 0.1f, 0.0f, 1000.0f, isoStep_ > 0.0f ? "%.2f" : "off");
    ImGui::SameLine();
    ImGui::Checkbox("sRGB", &srgbEncode_);
    ImGui::SameLine();
    ImGui::Checkbox("Show in Viewport", &showInViewport_);
    if (ref.levels > 1) {
        ImGui::SetNextItemWidth(220.0f);
        ImGui::SliderFloat("Mip", &mip_, 0.0f, static_cast<float>(ref.levels - 1), "%.1f");
    }
    if (ref.target == GL_TEXTURE_3D) {
        ImGui::SetNextItemWidth(220.0f);
        ImGui::SliderFloat("Tranche Z", &slice_, 0.0f, 1.0f);
    }
    if (ref.target == GL_TEXTURE_CUBE_MAP) ImGui::TextDisabled("Cubemap depliee en lat-long (azimut 0 = -Z au centre)");
    if (!inspectorMessage_.empty()) ImGui::TextDisabled("%s", inspectorMessage_.c_str());

    const GLuint output = InspectorOutputTexture();
    if (output != 0) {
        const float  width  = std::max(ImGui::GetContentRegionAvail().x, 64.0f);
        const float  aspect = static_cast<float>(inspectorTarget_.Height()) / static_cast<float>(inspectorTarget_.Width());
        ImGui::Image(static_cast<ImTextureID>(output), ImVec2(width, width * aspect), ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));
    } else if (!inspectorProgram_ || !inspectorProgram_->IsValid()) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.3f, 1.0f), "Shader de l'inspecteur invalide (voir panneau Shaders)");
    }
}

}  // namespace env::core
