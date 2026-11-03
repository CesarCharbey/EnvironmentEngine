#pragma once
// =============================================================================
//  src/core/Application.hpp — Fenêtre GLFW, boucle de frame, dockspace ImGui
//
//  FICHIER GELÉ (AGENTS.md) : n'ajoutez PAS de feature ici. Une feature = un .cpp
//  dans src/world_engine/ ou src/render_engine/ + REGISTER_SYSTEM.
//
//  Ordre de la frame : voir AGENTS.md section 3.
// =============================================================================
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <entt/entity/registry.hpp>
#include <glm/glm.hpp>

#include "contracts/ISystemModule.hpp"
#include "contracts/WorldContracts.hpp"
#include "core/Camera.hpp"
#include "core/MockDataProvider.hpp"
#include "core/PhysicsCore.hpp"
#include "core/RenderBlackboard.hpp"
#include "core/SystemRegistry.hpp"
#include "core/gpu/Framebuffer.hpp"
#include "core/gpu/GpuBuffer.hpp"
#include "core/gpu/ShaderManager.hpp"

struct GLFWwindow;

namespace env::core {

struct ApplicationConfig {
    int         width  = 1600;
    int         height = 900;
    std::string title  = "EnvEngine - Infinite Environment Engine";
    bool        vsync  = true;
    std::uint64_t maxFrames = 0;   ///< > 0 : quitte après N frames (smoke test / CI : --frames N)
    std::string   capturePath;     ///< si non vide : écrit l'image du Viewport (PPM) en quittant (--capture f.ppm)
    std::string   windowCapturePath;  ///< si non vide : écrit la fenêtre complète, UI incluse (--capture-window f.ppm)
};

class Application {
public:
    explicit Application(ApplicationConfig config = ApplicationConfig{});
    ~Application();

    Application(const Application&)            = delete;
    Application& operator=(const Application&) = delete;

    /// Initialise tout, exécute la boucle jusqu'à fermeture, libère tout. Code de sortie du processus.
    int Run();

private:
    struct SystemSlot {
        SystemDescriptor               descriptor;
        std::unique_ptr<ISystemModule> module;
        std::string                    windowTitle;
        std::string                    lastError;
        float                          cpuUpdateMs = 0.0f;
        float                          cpuRenderMs = 0.0f;
        float                          gpuMs       = 0.0f;
        bool                           showWindow  = true;
    };
    class GpuTimerRing;

    // --- Initialisation / arrêt ---------------------------------------------------
    bool InitWindow();
    bool InitGraphics();
    void InitImGui();
    void InitServices();
    void InitSystems();
    void Shutdown();
    void SetEnvironmentDefaults();
    void QueryGpuLimits();

    // --- Frame ----------------------------------------------------------------------
    void Frame(double now, float dt);
    void HandleShortcuts();
    void UpdateCamera(float dt);
    void UpdateRenderOrigin();
    void WriteCoreEnvironment(double now, float dt);
    void EnsureSceneTarget();
    void RunUpdates();
    void RunRenders();
    void ResetRenderState();
    void DisableFaultySystem(SystemSlot& slot, const char* phase, const char* what);
    bool SaveViewportCapture(const std::string& path);
    bool SaveWindowCapture(const std::string& path);
    /// Texture affichée par le Viewport : inspecteur (Show in Viewport), sinon scene.display, sinon scene.color.
    TextureRef ViewportSource(const char** label) const;

    // --- Interface ------------------------------------------------------------------
    void DrawGui();
    void DrawMainMenu();
    void DrawDockspace();
    void BuildDefaultLayout(unsigned int dockspaceId);
    void DrawViewport();
    void DrawSystemsPanel();
    void DrawEnvironmentPanel();
    void DrawBindingsPanel();
    void DrawSystemWindows();

    static void ScrollCallback(GLFWwindow* window, double xOffset, double yOffset);

    ApplicationConfig config_;
    GLFWwindow*       window_      = nullptr;
    bool              glfwReady_   = false;
    bool              imguiReady_  = false;
    bool              servicesReady_ = false;

    // --- Services du core -----------------------------------------------------------
    entt::registry                      registry_;
    std::unique_ptr<gpu::ShaderManager> shaders_;
    RenderBlackboard                    blackboard_;
    PhysicsCore                         physics_;
    Camera                              camera_;
    MockDataProvider                    mock_;
    GlobalEnvironmentUBO                environment_{};
    gpu::GpuBuffer                      environmentUbo_;
    gpu::Framebuffer                    sceneTarget_;
    std::unique_ptr<EngineContext>      ctx_;
    std::vector<SystemSlot>             systems_;
    std::unique_ptr<GpuTimerRing>       gpuTimers_;

    // --- Viewport / entrées -----------------------------------------------------------
    glm::ivec2 viewportSize_{1280, 720};
    bool       viewportHovered_ = false;
    bool       capturingMouse_  = false;
    double     lastCursorX_     = 0.0;
    double     lastCursorY_     = 0.0;
    float      pendingScroll_   = 0.0f;

    // --- État de l'interface ------------------------------------------------------------
    bool showSystems_     = true;
    bool showMock_        = true;
    bool showBlackboard_  = true;
    bool showShaders_     = true;
    bool showCamera_      = true;
    bool showPhysics_     = true;
    bool showEnvironment_ = true;
    bool showBindings_    = true;
    bool showImGuiDemo_   = false;
    bool resetLayout_     = false;
    bool vsync_           = true;

    // --- Statistiques -------------------------------------------------------------------
    std::uint64_t          frameIndex_       = 0;
    float                  frameMsSmoothed_  = 16.6f;
    std::array<float, 240> frameHistory_{};
    std::size_t            frameHistoryHead_ = 0;
    float                  gpuFrameMs_       = 0.0f;
    float                  gpuMockMs_        = 0.0f;
    glm::mat4              prevViewProj_{1.0f};
    std::uint32_t          originGeneration_ = 0;
    int                    gpuLimits_[4]     = {0, 0, 0, 0};  ///< UBO, SSBO, unités de texture, unités d'image
};

}  // namespace env::core
