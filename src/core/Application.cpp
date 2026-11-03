// =============================================================================
//  src/core/Application.cpp — Boucle de frame, fenêtre, dockspace, caméra
//  FICHIER GELÉ (AGENTS.md) : aucune feature ne s'ajoute ici.
// =============================================================================
#include "core/Application.hpp"

// glad AVANT GLFW (GLFW_INCLUDE_NONE est défini par CMake).
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_internal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "contracts/GpuBindings.hpp"

namespace env::core {
namespace {

constexpr const char* kWinViewport    = "Viewport";
constexpr const char* kWinSystems     = "Systems";
constexpr const char* kWinMock        = "Mock World";
constexpr const char* kWinBlackboard  = "Render Blackboard";
constexpr const char* kWinShaders     = "Shaders";
constexpr const char* kWinCamera      = "Camera";
constexpr const char* kWinPhysics     = "Physics (Jolt)";
constexpr const char* kWinEnvironment = "Environment UBO";
constexpr const char* kWinBindings    = "GPU Bindings";

/// Couleur de fond de scene.color (espace d'affichage), reprise par le brouillard du shader mock.
constexpr float kSceneClear[4] = {0.62f, 0.70f, 0.80f, 1.0f};

constexpr int kOwnerFrame = -1;
constexpr int kOwnerMock  = -2;

const ImVec4 kColorWorld(1.0f, 0.72f, 0.30f, 1.0f);
const ImVec4 kColorRender(0.40f, 0.80f, 1.0f, 1.0f);
const ImVec4 kColorDebug(0.75f, 0.75f, 0.75f, 1.0f);
const ImVec4 kColorError(1.0f, 0.35f, 0.30f, 1.0f);
const ImVec4 kColorOk(0.40f, 0.90f, 0.40f, 1.0f);

using Clock = std::chrono::steady_clock;

float Ema(float previous, float sample, float k = 0.1f) { return previous + (sample - previous) * k; }

float MsSince(Clock::time_point start) {
    return std::chrono::duration<float, std::milli>(Clock::now() - start).count();
}

ImVec4 StageColor(PassStage stage) {
    if (IsWorldStage(stage)) return kColorWorld;
    return stage == PassStage::DebugOverlay ? kColorDebug : kColorRender;
}

ImVec4 OwnerColor(bind::Owner owner) {
    switch (owner) {
        case bind::Owner::Core: return kColorDebug;
        case bind::Owner::Dev1: return kColorWorld;
        case bind::Owner::Dev2: return kColorRender;
    }
    return kColorDebug;
}

glm::vec3 DirectionFromAzimuthElevation(float azimuthDeg, float elevationDeg) {
    const float az = glm::radians(azimuthDeg), el = glm::radians(elevationDeg);
    return glm::normalize(glm::vec3(std::cos(el) * std::sin(az), std::sin(el), -std::cos(el) * std::cos(az)));
}

std::filesystem::path ResolveShaderRoot() {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (const char* fromEnv = std::getenv("ENV_SHADER_DIR")) {
        const fs::path p(fromEnv);
        if (fs::exists(p / "common" / "global_ubo.glsl", ec)) return p;
    }
#ifdef ENV_SHADER_DIR
    {
        const fs::path p(ENV_SHADER_DIR);
        if (fs::exists(p / "common" / "global_ubo.glsl", ec)) return p;
    }
#endif
    return fs::current_path(ec) / "shaders";
}

#ifdef ENV_GL_DEBUG
std::atomic<std::uint32_t> gGlDebugMessages{0};

void APIENTRY GlDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
                              const GLchar* message, const void* userParam) {
    (void)source;
    (void)length;
    (void)userParam;
    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) return;
    // Messages d'information NVIDIA sans intérêt (allocation de buffers, recompilation, ...).
    if (id == 131169 || id == 131185 || id == 131204 || id == 131218 || id == 131154) return;
    const std::uint32_t count = ++gGlDebugMessages;
    if (count > 200) {
        if (count == 201) std::fprintf(stderr, "[GL] Trop de messages : les suivants sont masques\n");
        return;
    }
    const char* sev = severity == GL_DEBUG_SEVERITY_HIGH ? "HIGH" : severity == GL_DEBUG_SEVERITY_MEDIUM ? "MEDIUM" : "LOW";
    const char* kind = type == GL_DEBUG_TYPE_ERROR                 ? "ERROR"
                       : type == GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR ? "DEPRECATED"
                       : type == GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR  ? "UNDEFINED"
                       : type == GL_DEBUG_TYPE_PERFORMANCE         ? "PERFORMANCE"
                       : type == GL_DEBUG_TYPE_PORTABILITY         ? "PORTABILITY"
                                                                   : "OTHER";
    std::fprintf(stderr, "[GL %s %s #%u] %s\n", sev, kind, id, message);
}
#endif

}  // namespace

// =============================================================================
// Chronométrage GPU par système (GL_TIMESTAMP, anneau de 4 frames, sans stall)
// =============================================================================
class Application::GpuTimerRing {
public:
    static constexpr int kFrames    = 4;
    static constexpr int kMaxScopes = 64;

    void Create() {
        for (FrameQueries& f : frames_) glCreateQueries(GL_TIMESTAMP, kMaxScopes * 2, f.queries.data());
        created_ = true;
    }

    void Destroy() {
        if (!created_) return;
        for (FrameQueries& f : frames_) glDeleteQueries(kMaxScopes * 2, f.queries.data());
        created_ = false;
    }

    /// Lit les résultats du slot (écrit il y a kFrames frames) s'ils sont disponibles, sans bloquer.
    template <class Fn>
    void Resolve(int slot, Fn&& onResult) {
        FrameQueries& f = frames_[static_cast<std::size_t>(slot)];
        if (!created_ || f.used == 0 || f.resolved) return;
        GLint available = 0;
        glGetQueryObjectiv(f.queries[static_cast<std::size_t>((f.used - 1) * 2 + 1)], GL_QUERY_RESULT_AVAILABLE, &available);
        if (!available) return;
        for (int s = 0; s < f.used; ++s) {
            GLuint64 t0 = 0, t1 = 0;
            glGetQueryObjectui64v(f.queries[static_cast<std::size_t>(s * 2)], GL_QUERY_RESULT, &t0);
            glGetQueryObjectui64v(f.queries[static_cast<std::size_t>(s * 2 + 1)], GL_QUERY_RESULT, &t1);
            onResult(f.owners[static_cast<std::size_t>(s)], static_cast<float>(static_cast<double>(t1 - t0) * 1e-6));
        }
        f.resolved = true;
    }

    void BeginFrame(int slot) {
        current_                 = slot;
        frames_[static_cast<std::size_t>(slot)].used     = 0;
        frames_[static_cast<std::size_t>(slot)].resolved = false;
    }

    int Begin(int owner) {
        FrameQueries& f = frames_[static_cast<std::size_t>(current_)];
        if (!created_ || f.used >= kMaxScopes) return -1;
        const int scope                          = f.used++;
        f.owners[static_cast<std::size_t>(scope)] = owner;
        glQueryCounter(f.queries[static_cast<std::size_t>(scope * 2)], GL_TIMESTAMP);
        return scope;
    }

    void End(int scope) {
        if (scope < 0) return;
        glQueryCounter(frames_[static_cast<std::size_t>(current_)].queries[static_cast<std::size_t>(scope * 2 + 1)], GL_TIMESTAMP);
    }

private:
    struct FrameQueries {
        std::array<GLuint, kMaxScopes * 2> queries{};
        std::array<int, kMaxScopes>        owners{};
        int                                used     = 0;
        bool                               resolved = false;
    };
    std::array<FrameQueries, kFrames> frames_{};
    int                               current_ = 0;
    bool                              created_ = false;
};

// =============================================================================
// Cycle de vie
// =============================================================================
Application::Application(ApplicationConfig config)
    : config_(std::move(config)), gpuTimers_(std::make_unique<GpuTimerRing>()) {
    vsync_ = config_.vsync;
}

Application::~Application() { Shutdown(); }

int Application::Run() {
    if (!InitWindow() || !InitGraphics()) {
        Shutdown();
        return EXIT_FAILURE;
    }
    InitImGui();
    InitServices();
    InitSystems();

    double       last          = glfwGetTime();
    const double runStart      = last;
    while (!glfwWindowShouldClose(window_) && (config_.maxFrames == 0 || frameIndex_ < config_.maxFrames)) {
        glfwPollEvents();
        int fbWidth = 0, fbHeight = 0;
        glfwGetFramebufferSize(window_, &fbWidth, &fbHeight);
        if (fbWidth == 0 || fbHeight == 0) {  // fenêtre minimisée
            glfwWaitEventsTimeout(0.1);
            last = glfwGetTime();
            continue;
        }
        const double now = glfwGetTime();
        const float  dt  = static_cast<float>(std::clamp(now - last, 0.0, 0.25));
        last             = now;
        Frame(now, dt);
    }

    if (frameIndex_ > 0) {
        const double elapsed = glfwGetTime() - runStart;
        std::printf("[Application] %llu frame(s) en %.2f s : %.1f FPS moyen | GPU %.2f ms | %zu programme(s) GLSL en erreur\n",
                    static_cast<unsigned long long>(frameIndex_), elapsed, static_cast<double>(frameIndex_) / std::max(elapsed, 1e-6),
                    static_cast<double>(gpuFrameMs_), shaders_ ? shaders_->ErrorCount() : std::size_t{0});
    }
    if (!config_.capturePath.empty() && servicesReady_) SaveViewportCapture(config_.capturePath);
    Shutdown();
    return EXIT_SUCCESS;
}

bool Application::InitWindow() {
    glfwSetErrorCallback([](int code, const char* description) {
        std::fprintf(stderr, "[GLFW] Erreur %d : %s\n", code, description);
    });
    if (!glfwInit()) {
        std::fprintf(stderr, "[Application] glfwInit a echoue\n");
        return false;
    }
    glfwReady_ = true;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#ifdef ENV_GL_DEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif

    window_ = glfwCreateWindow(config_.width, config_.height, config_.title.c_str(), nullptr, nullptr);
    if (window_ == nullptr) {
        std::fprintf(stderr, "[Application] Impossible de creer une fenetre OpenGL 4.6 Core (pilote trop ancien ?)\n");
        return false;
    }
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(vsync_ ? 1 : 0);
    glfwSetWindowUserPointer(window_, this);
    // Installé AVANT ImGui : le backend GLFW d'ImGui chaîne les callbacks existants.
    glfwSetScrollCallback(window_, &Application::ScrollCallback);
    return true;
}

bool Application::InitGraphics() {
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        std::fprintf(stderr, "[Application] gladLoadGLLoader a echoue\n");
        return false;
    }
    if (GLVersion.major < 4 || (GLVersion.major == 4 && GLVersion.minor < 6)) {
        std::fprintf(stderr, "[Application] OpenGL 4.6 requis, contexte obtenu : %d.%d\n", GLVersion.major, GLVersion.minor);
        return false;
    }
    std::printf("[Application] OpenGL %s | %s | %s\n", reinterpret_cast<const char*>(glGetString(GL_VERSION)),
                reinterpret_cast<const char*>(glGetString(GL_RENDERER)),
                reinterpret_cast<const char*>(glGetString(GL_VENDOR)));

#ifdef ENV_GL_DEBUG
    GLint contextFlags = 0;
    glGetIntegerv(GL_CONTEXT_FLAGS, &contextFlags);
    if (contextFlags & GL_CONTEXT_FLAG_DEBUG_BIT) {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(GlDebugCallback, nullptr);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);
        std::printf("[Application] Contexte de debug OpenGL actif (glDebugMessageCallback)\n");
    }
#endif

    QueryGpuLimits();

    // Conventions globales (ADR-007) : Reverse-Z infini, clip [0, 1], cubemaps sans couture.
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);
    glClearDepth(0.0);
    return true;
}

void Application::QueryGpuLimits() {
    glGetIntegerv(GL_MAX_UNIFORM_BUFFER_BINDINGS, &gpuLimits_[0]);
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &gpuLimits_[1]);
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &gpuLimits_[2]);
    glGetIntegerv(GL_MAX_IMAGE_UNITS, &gpuLimits_[3]);
    const std::uint32_t needed[4] = {bind::detail::MaxSlot(bind::kUboTable) + 1, bind::detail::MaxSlot(bind::kSsboTable) + 1,
                                     bind::detail::MaxSlot(bind::kTextureTable) + 1, bind::detail::MaxSlot(bind::kImageTable) + 1};
    const char* names[4] = {"UBO", "SSBO", "unites de texture", "unites d'image"};
    for (int i = 0; i < 4; ++i)
        if (static_cast<std::uint32_t>(gpuLimits_[i]) < needed[i])
            std::fprintf(stderr, "[Application] ATTENTION : le GPU expose %d %s, GpuBindings.hpp en utilise %u\n",
                         gpuLimits_[i], names[i], needed[i]);
}

void Application::InitImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.IniFilename = "envengine_layout.ini";

    ImGui::StyleColorsDark();
    ImGuiStyle& style    = ImGui::GetStyle();
    style.WindowRounding = 4.0f;
    style.FrameRounding  = 3.0f;
    style.GrabRounding   = 3.0f;
    style.TabRounding    = 3.0f;

    // HiDPI : si la plateforme met déjà à l'échelle (Wayland, macOS), on ne double pas.
    int   winW = 0, winH = 0, fbW = 0, fbH = 0;
    float scaleX = 1.0f, scaleY = 1.0f;
    glfwGetWindowSize(window_, &winW, &winH);
    glfwGetFramebufferSize(window_, &fbW, &fbH);
    glfwGetWindowContentScale(window_, &scaleX, &scaleY);
    const bool platformScales = winW > 0 && fbW > winW + 1;
    if (!platformScales && scaleX > 1.05f) {
        style.ScaleAllSizes(scaleX);
        io.FontGlobalScale = scaleX;
    }

    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 460");
    imguiReady_ = true;
}

void Application::InitServices() {
    shaders_ = std::make_unique<gpu::ShaderManager>(ResolveShaderRoot());
    blackboard_.Init(*shaders_);
    physics_.Init();

    environmentUbo_.Create(sizeof(GlobalEnvironmentUBO), GL_DYNAMIC_STORAGE_BIT, nullptr, "GlobalEnvironmentUBO");
    blackboard_.PublishBuffer(bb::FrameGlobalUBO, environmentUbo_, "Core");
    SetEnvironmentDefaults();

    camera_.SetWorldPosition(glm::dvec3(128.0, 95.0, 360.0));
    camera_.LookAt(glm::dvec3(128.0, 18.0, 128.0));
    EnsureSceneTarget();

    ctx_.reset(new EngineContext{registry_, blackboard_, *shaders_, physics_, camera_, environment_});
    ctx_->renderOriginChunk = ComputeRenderOriginChunk(camera_.WorldPosition());
    ctx_->viewportSize      = glm::ivec2(sceneTarget_.Width(), sceneTarget_.Height());

    mock_.Init(*shaders_);
    gpuTimers_->Create();
    servicesReady_ = true;
}

void Application::InitSystems() {
    std::vector<SystemInstance> instances = SystemRegistry::Instance().InstantiateAll();
    std::printf("[Application] %zu systeme(s) auto-enregistre(s), ordre d'execution :\n", instances.size());
    for (SystemInstance& instance : instances) {
        SystemSlot slot;
        slot.descriptor         = instance.descriptor;
        slot.module             = std::move(instance.module);
        const PassStage   stage = slot.descriptor.stage;
        const std::string name(slot.descriptor.name);
        const char* tag = IsWorldStage(stage) ? "[World] " : (stage == PassStage::DebugOverlay ? "[Debug] " : "[Render] ");
        slot.windowTitle = tag + name + "###sys_" + name;
        const std::string_view stageName = PassStageName(stage);
        std::printf("  %-16.*s %4d  %s\n", static_cast<int>(stageName.size()), stageName.data(), slot.descriptor.order,
                    name.c_str());
        systems_.push_back(std::move(slot));
    }
    for (SystemSlot& slot : systems_) {
        try {
            slot.module->Init(*ctx_);
        } catch (const std::exception& e) {
            DisableFaultySystem(slot, "Init", e.what());
        } catch (...) {
            DisableFaultySystem(slot, "Init", "exception inconnue");
        }
    }
}

void Application::SetEnvironmentDefaults() {
    environment_ = GlobalEnvironmentUBO{};
    // Valeurs initiales cohérentes avec les sliders par défaut de CelestialAtmosphereSystem.
    environment_.sunDirIlluminance = glm::vec4(DirectionFromAzimuthElevation(52.0f, 47.0f), 100000.0f);
    environment_.sunColorAngular   = glm::vec4(1.0f, 0.96f, 0.90f, glm::radians(0.2665f));
    environment_.moonDirPhase      = glm::vec4(DirectionFromAzimuthElevation(230.0f, 35.0f), 0.5f);
    environment_.moonParams        = glm::vec4(0.25f, glm::radians(0.259f), 0.02f, 0.0f);
    environment_.windDirSpeed      = glm::vec4(glm::normalize(glm::vec2(1.0f, 0.3f)), 4.0f, 0.3f);
    environment_.windGust          = glm::vec4(0.2f, 0.35f, 0.0f, 0.0f);
    environment_.weather           = glm::vec4(0.0f, 0.0f, 0.0f, 0.5f);
    environment_.weatherExtra      = glm::vec4(0.35f, 0.5f, 0.0006f, 0.0f);
    environment_.waterTime         = glm::vec4(0.0f, 8.4f, 0.0f, 0.0f);
}

void Application::Shutdown() {
    if (servicesReady_) {
        for (auto it = systems_.rbegin(); it != systems_.rend(); ++it) {
            try {
                it->module->Shutdown(*ctx_);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[Application] %s::Shutdown : %s\n", it->windowTitle.c_str(), e.what());
            } catch (...) {
            }
        }
        systems_.clear();
        mock_.Shutdown(registry_, blackboard_);
        registry_.clear();
        physics_.Shutdown();
        blackboard_.Shutdown();
        gpuTimers_->Destroy();
        sceneTarget_.Destroy();
        environmentUbo_.Destroy();
        if (shaders_) shaders_->DestroyAll();
        ctx_.reset();
        shaders_.reset();
        servicesReady_ = false;
    }
    if (imguiReady_) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        imguiReady_ = false;
    }
    if (window_ != nullptr) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    if (glfwReady_) {
        glfwTerminate();
        glfwReady_ = false;
    }
}

// =============================================================================
// Frame
// =============================================================================
void Application::Frame(double now, float dt) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    HandleShortcuts();
    shaders_->Update(now);
    UpdateCamera(dt);

    ctx_->time = FrameTime{now, dt, frameIndex_};
    UpdateRenderOrigin();
    EnsureSceneTarget();
    ctx_->viewportSize = glm::ivec2(sceneTarget_.Width(), sceneTarget_.Height());
    WriteCoreEnvironment(now, dt);
    blackboard_.BeginFrame();

    RunUpdates();
    physics_.Step(dt);

    // Upload unique de l'UBO, après tous les Update() (un champ = un écrivain).
    environmentUbo_.UploadObject(environment_);
    environmentUbo_.BindBase(GL_UNIFORM_BUFFER, bind::ubo::GlobalEnvironment);

    RunRenders();

    int fbWidth = 0, fbHeight = 0;
    glfwGetFramebufferSize(window_, &fbWidth, &fbHeight);
    gpu::Framebuffer::BindDefault(fbWidth, fbHeight);
    const GLfloat background[4] = {0.08f, 0.08f, 0.09f, 1.0f};
    glClearNamedFramebufferfv(0, GL_COLOR, 0, background);

    DrawGui();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    const bool lastFrame = config_.maxFrames != 0 && frameIndex_ + 1 == config_.maxFrames;
    if (lastFrame && !config_.windowCapturePath.empty()) SaveWindowCapture(config_.windowCapturePath);
    glfwSwapBuffers(window_);

    const float frameMs                 = dt * 1000.0f;
    frameMsSmoothed_                    = Ema(frameMsSmoothed_, frameMs, 0.05f);
    frameHistory_[frameHistoryHead_]    = frameMs;
    frameHistoryHead_                   = (frameHistoryHead_ + 1) % frameHistory_.size();
    ++frameIndex_;
}

void Application::HandleShortcuts() {
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantTextInput) return;
    if (ImGui::IsKeyPressed(ImGuiKey_F5, false)) shaders_->RequestReloadAll();
    if (ImGui::IsKeyPressed(ImGuiKey_F12, false))
        SaveViewportCapture("envengine_capture_" + std::to_string(frameIndex_) + ".ppm");
}

namespace {
bool WritePpm(const std::string& path, int width, int height, const std::vector<unsigned char>& rgbBottomUp) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::fprintf(stderr, "[Application] Capture impossible : %s\n", path.c_str());
        return false;
    }
    file << "P6\n" << width << " " << height << "\n255\n";
    const std::size_t stride = static_cast<std::size_t>(width) * 3u;
    for (int y = height - 1; y >= 0; --y)  // OpenGL : origine en bas à gauche
        file.write(reinterpret_cast<const char*>(&rgbBottomUp[static_cast<std::size_t>(y) * stride]), static_cast<std::streamsize>(stride));
    std::printf("[Application] Capture : %s (%dx%d)\n", path.c_str(), width, height);
    return true;
}
}  // namespace

TextureRef Application::ViewportSource(const char** label) const {
    TextureRef  ref = blackboard_.GetTexture(bb::SceneColor);
    const char* name = "scene.color";
    if (blackboard_.InspectorShowsInViewport() && blackboard_.InspectorOutputTexture() != 0) {
        ref.id             = blackboard_.InspectorOutputTexture();
        ref.internalFormat = GL_RGBA8;
        ref.size           = glm::ivec3(blackboard_.InspectorOutputSize(), 1);
        name               = "inspecteur du blackboard";
    } else if (blackboard_.IsPublished(bb::SceneDisplay)) {
        ref  = blackboard_.GetTexture(bb::SceneDisplay);
        name = "scene.display";
    }
    if (label != nullptr) *label = name;
    return ref;
}

bool Application::SaveWindowCapture(const std::string& path) {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    if (width <= 0 || height <= 0) return false;
    std::vector<unsigned char> rgb(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glNamedFramebufferReadBuffer(0, GL_BACK);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadnPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, static_cast<GLsizei>(rgb.size()), rgb.data());
    return WritePpm(path, width, height, rgb);
}

bool Application::SaveViewportCapture(const std::string& path) {
    // Exactement ce qu'affiche le Viewport (inspecteur, scene.display ou scene.color).
    const TextureRef source = ViewportSource(nullptr);
    const int width = source.size.x, height = source.size.y;
    if (source.id == 0 || width <= 0 || height <= 0) return false;
    std::vector<float> pixels(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u);
    glGetTextureImage(source.id, 0, GL_RGBA, GL_FLOAT, static_cast<GLsizei>(pixels.size() * sizeof(float)), pixels.data());

    std::vector<unsigned char> rgb(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u);
    for (std::size_t i = 0, n = static_cast<std::size_t>(width) * static_cast<std::size_t>(height); i < n; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            rgb[i * 3u + c] = static_cast<unsigned char>(std::clamp(pixels[i * 4u + c], 0.0f, 1.0f) * 255.0f + 0.5f);
    return WritePpm(path, width, height, rgb);
}

void Application::UpdateCamera(float dt) {
    ImGuiIO&   io  = ImGui::GetIO();
    const bool rmb = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

    if (!capturingMouse_ && rmb && viewportHovered_) {
        capturingMouse_ = true;
        glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        if (glfwRawMouseMotionSupported()) glfwSetInputMode(window_, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
        glfwGetCursorPos(window_, &lastCursorX_, &lastCursorY_);
        io.ConfigFlags |= ImGuiConfigFlags_NoMouse;
    } else if (capturingMouse_ && (!rmb || glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)) {
        capturingMouse_ = false;
        glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        io.ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
    }

    CameraInput input;
    if (capturingMouse_) {
        double x = 0.0, y = 0.0;
        glfwGetCursorPos(window_, &x, &y);
        input.lookDelta = glm::vec2(static_cast<float>(x - lastCursorX_), static_cast<float>(y - lastCursorY_));
        lastCursorX_    = x;
        lastCursorY_    = y;
        // Codes GLFW = position physique des touches : WASD (QWERTY) == ZQSD (AZERTY).
        auto key = [this](int k) { return glfwGetKey(window_, k) == GLFW_PRESS ? 1.0f : 0.0f; };
        input.move.z = key(GLFW_KEY_W) - key(GLFW_KEY_S);
        input.move.x = key(GLFW_KEY_D) - key(GLFW_KEY_A);
        input.move.y = std::max(key(GLFW_KEY_E), key(GLFW_KEY_SPACE)) - std::max(key(GLFW_KEY_Q), key(GLFW_KEY_LEFT_CONTROL));
        input.fast   = key(GLFW_KEY_LEFT_SHIFT) > 0.0f;
        input.slow   = key(GLFW_KEY_LEFT_ALT) > 0.0f;
    }
    if (capturingMouse_ || viewportHovered_) input.speedSteps = pendingScroll_;
    pendingScroll_ = 0.0f;
    camera_.ApplyInput(input, dt);
}

void Application::UpdateRenderOrigin() {
    const glm::ivec2 newOrigin = ComputeRenderOriginChunk(camera_.WorldPosition());
    const glm::ivec2 oldOrigin = ctx_->renderOriginChunk;
    ctx_->renderOriginShifted  = newOrigin != oldOrigin;
    if (!ctx_->renderOriginShifted) return;
    // Les positions Jolt sont relatives à l'origine : on les translate de (ancienne - nouvelle).
    const glm::dvec3 delta = RenderOriginWorld(oldOrigin) - RenderOriginWorld(newOrigin);
    physics_.TranslateAllBodies(glm::vec3(delta));
    ctx_->renderOriginChunk = newOrigin;
    ++originGeneration_;
    std::printf("[Application] Origine de rendu recalee sur le chunk (%d, %d)\n", newOrigin.x, newOrigin.y);
}

void Application::WriteCoreEnvironment(double now, float dt) {
    const glm::dvec3 originWorld = RenderOriginWorld(ctx_->renderOriginChunk);
    const float      width       = static_cast<float>(std::max(sceneTarget_.Width(), 1));
    const float      height      = static_cast<float>(std::max(sceneTarget_.Height(), 1));
    const float      aspect      = width / height;

    const glm::mat4 view     = camera_.ViewMatrix(originWorld);
    const glm::mat4 proj     = camera_.ProjectionMatrix(aspect);
    const glm::mat4 viewProj = proj * view;

    environment_.view         = view;
    environment_.proj         = proj;
    environment_.viewProj     = viewProj;
    environment_.invView      = glm::inverse(view);
    environment_.invProj      = glm::inverse(proj);
    environment_.invViewProj  = glm::inverse(viewProj);
    environment_.prevViewProj = (frameIndex_ == 0 || ctx_->renderOriginShifted) ? viewProj : prevViewProj_;
    prevViewProj_             = viewProj;

    environment_.cameraPosition = glm::vec4(glm::vec3(camera_.WorldPosition() - originWorld), 1.0f);
    environment_.cameraParams   = glm::vec4(camera_.nearPlane, camera_.renderDistance,
                                            std::tan(glm::radians(camera_.fovYDegrees) * 0.5f), aspect);
    environment_.viewportSize   = glm::vec4(width, height, 1.0f / width, 1.0f / height);
    environment_.originChunk    = glm::ivec4(ctx_->renderOriginChunk.x, ctx_->renderOriginChunk.y,
                                             static_cast<int>(originGeneration_), 0);
    environment_.waterTime.z    = static_cast<float>(now);
    environment_.waterTime.w    = dt;
    environment_.frameInfo.x    = static_cast<std::uint32_t>(frameIndex_);
    environment_.frameInfo.z    = (ctx_->renderOriginShifted ? 1u : 0u) | (mock_.IsActive() ? 2u : 0u);
}

void Application::EnsureSceneTarget() {
    const GLsizei width  = std::max(viewportSize_.x, 1);
    const GLsizei height = std::max(viewportSize_.y, 1);
    bool          changed = false;
    if (!sceneTarget_.IsValid()) {
        gpu::Framebuffer::Desc desc;
        desc.width  = width;
        desc.height = height;
        gpu::Framebuffer::ColorAttachmentDesc color;
        color.internalFormat = GL_RGBA16F;
        color.debugName      = "scene.color";
        desc.colors.push_back(color);
        desc.depthFormat = GL_DEPTH_COMPONENT32F;
        desc.debugName   = "SceneTarget";
        sceneTarget_.Create(desc);
        changed = true;
    } else {
        changed = sceneTarget_.Resize(width, height);
    }
    if (changed) {
        blackboard_.PublishTexture(bb::SceneColor, sceneTarget_.Color(0), "Core");
        blackboard_.PublishTexture(bb::SceneDepth, sceneTarget_.Depth(), "Core");
    }
}

void Application::DisableFaultySystem(SystemSlot& slot, const char* phase, const char* what) {
    slot.module->enabled = false;
    slot.lastError       = std::string(phase) + " : " + what;
    std::fprintf(stderr, "[Application] Systeme '%.*s' desactive (%s)\n", static_cast<int>(slot.descriptor.name.size()),
                 slot.descriptor.name.data(), slot.lastError.c_str());
}

void Application::RunUpdates() {
    mock_.Update(*ctx_);
    for (SystemSlot& slot : systems_) {
        if (!slot.module->enabled) continue;
        const auto start = Clock::now();
        try {
            slot.module->Update(*ctx_);
        } catch (const std::exception& e) {
            DisableFaultySystem(slot, "Update", e.what());
        } catch (...) {
            DisableFaultySystem(slot, "Update", "exception inconnue");
        }
        slot.cpuUpdateMs = Ema(slot.cpuUpdateMs, MsSince(start));
    }
}

void Application::ResetRenderState() {
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_STENCIL_TEST);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);  // Reverse-Z
    glDepthMask(GL_TRUE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glBindVertexArray(0);
    glUseProgram(0);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
    environmentUbo_.BindBase(GL_UNIFORM_BUFFER, bind::ubo::GlobalEnvironment);
}

void Application::RunRenders() {
    const int timerSlot = static_cast<int>(frameIndex_ % GpuTimerRing::kFrames);
    gpuTimers_->Resolve(timerSlot, [this](int owner, float ms) {
        if (owner == kOwnerFrame)
            gpuFrameMs_ = Ema(gpuFrameMs_, ms);
        else if (owner == kOwnerMock)
            gpuMockMs_ = Ema(gpuMockMs_, ms);
        else if (owner >= 0 && static_cast<std::size_t>(owner) < systems_.size())
            systems_[static_cast<std::size_t>(owner)].gpuMs = Ema(systems_[static_cast<std::size_t>(owner)].gpuMs, ms);
    });
    gpuTimers_->BeginFrame(timerSlot);
    const int frameScope = gpuTimers_->Begin(kOwnerFrame);

    sceneTarget_.Bind();
    ResetRenderState();
    sceneTarget_.ClearColor(0, kSceneClear[0], kSceneClear[1], kSceneClear[2], kSceneClear[3]);
    sceneTarget_.ClearDepth(0.0f);  // Reverse-Z : 0 = infini

    const int mockScope = gpuTimers_->Begin(kOwnerMock);
    mock_.Render(*ctx_);
    gpuTimers_->End(mockScope);

    for (std::size_t i = 0; i < systems_.size(); ++i) {
        SystemSlot& slot = systems_[i];
        if (!slot.module->enabled) continue;
        sceneTarget_.Bind();
        ResetRenderState();
        const int  scope = gpuTimers_->Begin(static_cast<int>(i));
        const auto start = Clock::now();
        try {
            slot.module->Render(*ctx_);
        } catch (const std::exception& e) {
            DisableFaultySystem(slot, "Render", e.what());
        } catch (...) {
            DisableFaultySystem(slot, "Render", "exception inconnue");
        }
        gpuTimers_->End(scope);
        slot.cpuRenderMs = Ema(slot.cpuRenderMs, MsSince(start));
    }

    sceneTarget_.Bind();
    ResetRenderState();
    blackboard_.RenderInspector();
    gpuTimers_->End(frameScope);
}

// =============================================================================
// Interface
// =============================================================================
void Application::ScrollCallback(GLFWwindow* window, double xOffset, double yOffset) {
    (void)xOffset;
    if (auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window))) app->pendingScroll_ += static_cast<float>(yOffset);
}

void Application::DrawGui() {
    DrawMainMenu();
    DrawDockspace();
    DrawViewport();
    if (showSystems_) DrawSystemsPanel();

    auto panel = [](const char* title, bool& open, auto&& body) {
        if (!open) return;
        if (ImGui::Begin(title, &open)) body();
        ImGui::End();
    };
    panel(kWinMock, showMock_, [this] { mock_.OnGui(*ctx_); });
    panel(kWinBlackboard, showBlackboard_, [this] { blackboard_.OnGui(); });
    panel(kWinShaders, showShaders_, [this] { shaders_->OnGui(); });
    panel(kWinCamera, showCamera_, [this] { camera_.OnGui(); });
    panel(kWinPhysics, showPhysics_, [this] { physics_.OnGui(); });
    if (showEnvironment_) DrawEnvironmentPanel();
    if (showBindings_) DrawBindingsPanel();
    DrawSystemWindows();
    if (showImGuiDemo_) ImGui::ShowDemoWindow(&showImGuiDemo_);
}

void Application::DrawMainMenu() {
    if (!ImGui::BeginMainMenuBar()) return;
    if (ImGui::BeginMenu("Fichier")) {
        if (ImGui::MenuItem("Recharger les shaders", "F5")) shaders_->RequestReloadAll();
        ImGui::Separator();
        if (ImGui::MenuItem("Quitter", "Alt+F4")) glfwSetWindowShouldClose(window_, GLFW_TRUE);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Affichage")) {
        ImGui::MenuItem(kWinSystems, nullptr, &showSystems_);
        ImGui::MenuItem(kWinMock, nullptr, &showMock_);
        ImGui::MenuItem(kWinBlackboard, nullptr, &showBlackboard_);
        ImGui::MenuItem(kWinShaders, nullptr, &showShaders_);
        ImGui::MenuItem(kWinCamera, nullptr, &showCamera_);
        ImGui::MenuItem(kWinPhysics, nullptr, &showPhysics_);
        ImGui::MenuItem(kWinEnvironment, nullptr, &showEnvironment_);
        ImGui::MenuItem(kWinBindings, nullptr, &showBindings_);
        if (ImGui::BeginMenu("Panneaux des systemes")) {
            for (SystemSlot& slot : systems_) ImGui::MenuItem(slot.windowTitle.c_str(), nullptr, &slot.showWindow);
            ImGui::EndMenu();
        }
        ImGui::Separator();
        ImGui::MenuItem("Demo Dear ImGui", nullptr, &showImGuiDemo_);
        if (ImGui::MenuItem("Reinitialiser la disposition")) resetLayout_ = true;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Rendu")) {
        if (ImGui::MenuItem("VSync", nullptr, &vsync_)) glfwSwapInterval(vsync_ ? 1 : 0);
        ImGui::EndMenu();
    }
    char stats[96];
    std::snprintf(stats, sizeof(stats), "%.0f FPS | %.2f ms | GPU %.2f ms", 1000.0 / std::max(frameMsSmoothed_, 0.001f),
                  static_cast<double>(frameMsSmoothed_), static_cast<double>(gpuFrameMs_));
    const float textWidth = ImGui::CalcTextSize(stats).x;
    ImGui::SameLine(std::max(ImGui::GetWindowWidth() - textWidth - 16.0f, 0.0f));
    ImGui::TextDisabled("%s", stats);
    ImGui::EndMainMenuBar();
}

void Application::DrawDockspace() {
    const ImGuiID dockspaceId = ImGui::GetID("EnvEngineDockspace");
    if (resetLayout_ || ImGui::DockBuilderGetNode(dockspaceId) == nullptr) {
        BuildDefaultLayout(dockspaceId);
        resetLayout_ = false;
    }
    ImGui::DockSpaceOverViewport(dockspaceId, ImGui::GetMainViewport(), ImGuiDockNodeFlags_None);
}

void Application::BuildDefaultLayout(unsigned int dockspaceId) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::DockBuilderRemoveNode(dockspaceId);
    ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspaceId, viewport->WorkSize);

    ImGuiID center      = dockspaceId;
    ImGuiID left        = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.22f, nullptr, &center);
    ImGuiID right       = ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.30f, nullptr, &center);
    ImGuiID leftBottom  = ImGui::DockBuilderSplitNode(left, ImGuiDir_Down, 0.50f, nullptr, &left);
    ImGuiID rightBottom = ImGui::DockBuilderSplitNode(right, ImGuiDir_Down, 0.50f, nullptr, &right);
    ImGuiID bottom      = ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, 0.30f, nullptr, &center);

    ImGui::DockBuilderDockWindow(kWinViewport, center);
    ImGui::DockBuilderDockWindow(kWinSystems, left);
    ImGui::DockBuilderDockWindow(kWinCamera, left);
    ImGui::DockBuilderDockWindow(kWinPhysics, left);
    ImGui::DockBuilderDockWindow(kWinMock, leftBottom);
    ImGui::DockBuilderDockWindow(kWinEnvironment, leftBottom);
    ImGui::DockBuilderDockWindow(kWinBlackboard, bottom);
    ImGui::DockBuilderDockWindow(kWinShaders, bottom);
    ImGui::DockBuilderDockWindow(kWinBindings, bottom);
    for (const SystemSlot& slot : systems_)
        ImGui::DockBuilderDockWindow(slot.windowTitle.c_str(), IsWorldStage(slot.descriptor.stage) ? right : rightBottom);
    ImGui::DockBuilderFinish(dockspaceId);
}

void Application::DrawViewport() {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    const bool open = ImGui::Begin(kWinViewport, nullptr,
                                   ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse);
    ImGui::PopStyleVar();
    if (!open) {
        viewportHovered_ = false;
        ImGui::End();
        return;
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    avail.x      = std::max(avail.x, 1.0f);
    avail.y      = std::max(avail.y, 1.0f);
    const ImVec2 scale = ImGui::GetIO().DisplayFramebufferScale;
    viewportSize_      = glm::ivec2(std::max(1, static_cast<int>(avail.x * scale.x)), std::max(1, static_cast<int>(avail.y * scale.y)));

    const char*  source  = nullptr;
    const GLuint texture = ViewportSource(&source).id;

    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::Image(static_cast<ImTextureID>(texture), avail, ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));
    viewportHovered_ = ImGui::IsItemHovered();

    // Overlay d'information (dessiné directement dans la draw list : n'intercepte pas la souris).
    const glm::dvec3 cam   = camera_.WorldPosition();
    const glm::ivec2 chunk = WorldToChunk(cam);
    struct Line {
        ImU32 color;
        char  text[160];
    };
    Line        lines[5];
    int         lineCount = 0;
    const ImU32 white     = IM_COL32(235, 235, 235, 255);
    const ImU32 grey      = IM_COL32(170, 170, 170, 255);
    auto add = [&](ImU32 color, const char* format, auto... args) {
        lines[lineCount].color = color;
        std::snprintf(lines[lineCount].text, sizeof(lines[lineCount].text), format, args...);
        ++lineCount;
    };
    add(ImGui::GetColorU32(kColorOk), "%.0f FPS  %.2f ms  (GPU %.2f ms)", 1000.0 / std::max(frameMsSmoothed_, 0.001f),
        static_cast<double>(frameMsSmoothed_), static_cast<double>(gpuFrameMs_));
    add(white, "Camera (%.1f, %.1f, %.1f) m | chunk (%d, %d) | origine (%d, %d)", cam.x, cam.y, cam.z, chunk.x, chunk.y,
        ctx_->renderOriginChunk.x, ctx_->renderOriginChunk.y);
    add(grey, "Source : %s (%dx%d)", source, static_cast<int>(sceneTarget_.Width()), static_cast<int>(sceneTarget_.Height()));
    if (const std::size_t errors = shaders_->ErrorCount(); errors > 0)
        add(ImGui::GetColorU32(kColorError), "%zu programme(s) GLSL en erreur - voir le panneau Shaders", errors);
    if (!capturingMouse_) add(grey, "%s", "Clic droit : regarder | WASD/ZQSD + E/Q | Shift rapide | F5 : shaders | F12 : capture");

    const float lineHeight = ImGui::GetTextLineHeightWithSpacing();
    float       maxWidth   = 0.0f;
    for (int i = 0; i < lineCount; ++i) maxWidth = std::max(maxWidth, ImGui::CalcTextSize(lines[i].text).x);
    ImDrawList*  drawList = ImGui::GetWindowDrawList();
    const ImVec2 boxMin(origin.x + 6.0f, origin.y + 6.0f);
    const ImVec2 boxMax(boxMin.x + maxWidth + 12.0f, boxMin.y + lineHeight * static_cast<float>(lineCount) + 8.0f);
    drawList->AddRectFilled(boxMin, boxMax, IM_COL32(0, 0, 0, 150), 4.0f);
    for (int i = 0; i < lineCount; ++i)
        drawList->AddText(ImVec2(boxMin.x + 6.0f, boxMin.y + 4.0f + lineHeight * static_cast<float>(i)), lines[i].color,
                          lines[i].text);
    ImGui::End();
}

void Application::DrawSystemsPanel() {
    if (!ImGui::Begin(kWinSystems, &showSystems_)) {
        ImGui::End();
        return;
    }
    ImGui::Text("%.1f FPS | CPU %.2f ms | GPU %.2f ms (mock %.2f ms)", 1000.0 / std::max(frameMsSmoothed_, 0.001f),
                static_cast<double>(frameMsSmoothed_), static_cast<double>(gpuFrameMs_), static_cast<double>(gpuMockMs_));
    ImGui::PlotLines("##frametimes", frameHistory_.data(), static_cast<int>(frameHistory_.size()),
                     static_cast<int>(frameHistoryHead_), "frame (ms)", 0.0f, 40.0f, ImVec2(-1.0f, 48.0f));
    if (ImGui::Checkbox("VSync", &vsync_)) glfwSwapInterval(vsync_ ? 1 : 0);
    ImGui::SameLine();
    ImGui::TextDisabled("%zu systeme(s) | %zu chunk(s) dans l'ECS | %zu programme(s) GLSL", systems_.size(),
                        static_cast<std::size_t>(registry_.view<ChunkCoordComponent>().size()), shaders_->ProgramCount());

    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_Resizable |
                                  ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollX;
    if (ImGui::BeginTable("##systems", 8, flags)) {
        ImGui::TableSetupColumn("On");
        ImGui::TableSetupColumn("Stage");
        ImGui::TableSetupColumn("Ord.");
        ImGui::TableSetupColumn("Systeme");
        ImGui::TableSetupColumn("Update ms");
        ImGui::TableSetupColumn("Render ms");
        ImGui::TableSetupColumn("GPU ms");
        ImGui::TableSetupColumn("Fichier");
        ImGui::TableHeadersRow();
        for (std::size_t i = 0; i < systems_.size(); ++i) {
            SystemSlot&     slot  = systems_[i];
            const PassStage stage = slot.descriptor.stage;
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::PushID(static_cast<int>(i));
            ImGui::Checkbox("##on", &slot.module->enabled);
            ImGui::PopID();
            ImGui::TableNextColumn();
            const std::string_view stageName = PassStageName(stage);
            ImGui::TextColored(StageColor(stage), "%.*s", static_cast<int>(stageName.size()), stageName.data());
            ImGui::TableNextColumn();
            ImGui::Text("%d", slot.descriptor.order);
            ImGui::TableNextColumn();
            ImGui::Text("%.*s", static_cast<int>(slot.descriptor.name.size()), slot.descriptor.name.data());
            if (!slot.lastError.empty()) {
                ImGui::SameLine();
                ImGui::TextColored(kColorError, "(!)");
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", slot.lastError.c_str());
            }
            const bool on = slot.module->enabled;
            ImGui::TableNextColumn();
            on ? ImGui::Text("%.3f", static_cast<double>(slot.cpuUpdateMs)) : ImGui::TextDisabled("-");
            ImGui::TableNextColumn();
            on ? ImGui::Text("%.3f", static_cast<double>(slot.cpuRenderMs)) : ImGui::TextDisabled("-");
            ImGui::TableNextColumn();
            on ? ImGui::Text("%.3f", static_cast<double>(slot.gpuMs)) : ImGui::TextDisabled("-");
            ImGui::TableNextColumn();
            std::string_view file = slot.descriptor.sourceFile;
            if (const std::size_t pos = file.rfind("src/"); pos != std::string_view::npos) file = file.substr(pos + 4);
            ImGui::TextDisabled("%.*s", static_cast<int>(file.size()), file.data());
        }
        ImGui::EndTable();
    }
    ImGui::TextDisabled("Ordre deterministe : (stage, ordre, nom). Ajouter un systeme = un .cpp + REGISTER_SYSTEM.");
    ImGui::End();
}

void Application::DrawEnvironmentPanel() {
    if (!ImGui::Begin(kWinEnvironment, &showEnvironment_)) {
        ImGui::End();
        return;
    }
    const GlobalEnvironmentUBO& e = environment_;
    ImGui::TextDisabled("Miroir CPU du GlobalEnvironmentUBO (std140, %zu octets, binding %u)", sizeof(GlobalEnvironmentUBO),
                        bind::ubo::GlobalEnvironment);
    auto v4 = [](const char* label, const glm::vec4& v) {
        ImGui::Text("%-18s %9.4f %9.4f %9.4f %9.4f", label, static_cast<double>(v.x), static_cast<double>(v.y),
                    static_cast<double>(v.z), static_cast<double>(v.w));
    };
    ImGui::SeparatorText("Camera [Core]");
    v4("cameraPosition", e.cameraPosition);
    v4("cameraParams", e.cameraParams);
    v4("viewportSize", e.viewportSize);
    ImGui::Text("%-18s %9d %9d %9d", "originChunk", e.originChunk.x, e.originChunk.y, e.originChunk.z);
    ImGui::SeparatorText("Soleil / Lune [CelestialAtmosphereSystem - Dev 2]");
    v4("sunDirIlluminance", e.sunDirIlluminance);
    v4("sunColorAngular", e.sunColorAngular);
    v4("moonDirPhase", e.moonDirPhase);
    v4("moonParams", e.moonParams);
    ImGui::SeparatorText("Vent / Meteo [CelestialAtmosphereSystem - Dev 2]");
    v4("windDirSpeed", e.windDirSpeed);
    v4("windGust", e.windGust);
    v4("weather", e.weather);
    v4("weatherExtra", e.weatherExtra);
    ImGui::SeparatorText("Eau [BiomeHydrology - Dev 1] / Temps [Core]");
    v4("waterTime", e.waterTime);
    ImGui::SeparatorText("Frame [Core]");
    ImGui::Text("frame %u | flags 0x%X (bit0 origine recalee, bit1 mock actif)", e.frameInfo.x, e.frameInfo.z);
    int debugView = static_cast<int>(environment_.frameInfo.y);
    ImGui::SetNextItemWidth(120.0f);
    if (ImGui::InputInt("Vue de debug globale (frameInfo.y)", &debugView))
        environment_.frameInfo.y = static_cast<std::uint32_t>(std::max(debugView, 0));
    ImGui::End();
}

void Application::DrawBindingsPanel() {
    if (!ImGui::Begin(kWinBindings, &showBindings_)) {
        ImGui::End();
        return;
    }
    ImGui::Text("Limites du GPU : UBO %d | SSBO %d | unites de texture %d | unites d'image %d", gpuLimits_[0],
                gpuLimits_[1], gpuLimits_[2], gpuLimits_[3]);
    ImGui::TextDisabled("Source unique : include/contracts/GpuBindings.hpp (C++ env::bind::*, GLSL PREFIXE_Nom)");
    auto table = [](const char* label, const auto& rows, std::string_view prefix) {
        if (!ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_DefaultOpen)) return;
        if (!ImGui::BeginTable(label, 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchProp))
            return;
        ImGui::TableSetupColumn("Slot", ImGuiTableColumnFlags_WidthFixed, 40.0f);
        ImGui::TableSetupColumn("GLSL");
        ImGui::TableSetupColumn("Owner", ImGuiTableColumnFlags_WidthFixed, 60.0f);
        ImGui::TableSetupColumn("Role");
        ImGui::TableHeadersRow();
        for (const bind::SlotInfo& s : rows) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%u", s.slot);
            ImGui::TableNextColumn();
            ImGui::Text("%.*s%.*s", static_cast<int>(prefix.size()), prefix.data(), static_cast<int>(s.name.size()), s.name.data());
            ImGui::TableNextColumn();
            const std::string_view owner = bind::OwnerName(s.owner);
            ImGui::TextColored(OwnerColor(s.owner), "%.*s", static_cast<int>(owner.size()), owner.data());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(s.description.data(), s.description.data() + s.description.size());
        }
        ImGui::EndTable();
    };
    table("UBO (Core 0-1 | Dev 1 2-7 | Dev 2 8-15)", bind::kUboTable, bind::kGlslPrefixUbo);
    table("SSBO (Core 0-1 | Dev 1 2-7 | Dev 2 8-15)", bind::kSsboTable, bind::kGlslPrefixSsbo);
    table("Unites de texture (Core 0-5 | Dev 1 6-9 | Dev 2 10-31)", bind::kTextureTable, bind::kGlslPrefixTexture);
    table("Unites d'image (Dev 1 0-2 | Core 3 | Dev 2 4-7)", bind::kImageTable, bind::kGlslPrefixImage);
    ImGui::End();
}

void Application::DrawSystemWindows() {
    for (std::size_t i = 0; i < systems_.size(); ++i) {
        SystemSlot& slot = systems_[i];
        if (!slot.showWindow) continue;
        if (ImGui::Begin(slot.windowTitle.c_str(), &slot.showWindow)) {
            ImGui::Checkbox("Enabled", &slot.module->enabled);
            ImGui::SameLine();
            const std::string_view stageName = PassStageName(slot.descriptor.stage);
            ImGui::TextDisabled("%.*s | ordre %d | CPU %.3f / %.3f ms | GPU %.3f ms", static_cast<int>(stageName.size()),
                                stageName.data(), slot.descriptor.order, static_cast<double>(slot.cpuUpdateMs),
                                static_cast<double>(slot.cpuRenderMs), static_cast<double>(slot.gpuMs));
            if (!slot.lastError.empty()) {
                ImGui::TextColored(kColorError, "Desactive apres exception : %s", slot.lastError.c_str());
                ImGui::SameLine();
                if (ImGui::SmallButton("Reessayer")) {
                    slot.lastError.clear();
                    slot.module->enabled = true;
                }
            }
            ImGui::Separator();
            ImGui::PushID(static_cast<int>(i));
            ImGuiErrorRecoveryState recovery;
            ImGui::ErrorRecoveryStoreState(&recovery);
            try {
                slot.module->OnGui(*ctx_);
            } catch (const std::exception& e) {
                ImGui::ErrorRecoveryTryToRecoverState(&recovery);
                DisableFaultySystem(slot, "OnGui", e.what());
            } catch (...) {
                ImGui::ErrorRecoveryTryToRecoverState(&recovery);
                DisableFaultySystem(slot, "OnGui", "exception inconnue");
            }
            ImGui::PopID();
        }
        ImGui::End();
    }
}

}  // namespace env::core
