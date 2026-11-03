// =============================================================================
//  src/main.cpp — Point d'entrée d'EnvEngine
//
//  FICHIER GELÉ (AGENTS.md, ADR-002) : il ne doit JAMAIS être modifié pour
//  ajouter une feature. Les systèmes de src/world_engine/ (Dev 1) et
//  src/render_engine/ (Dev 2) s'enregistrent eux-mêmes via REGISTER_SYSTEM.
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "core/Application.hpp"

// Options (smoke test / CI) :
//   --frames N        quitte après N frames et affiche le FPS moyen
//   --capture f.ppm   écrit l'image du Viewport en quittant
//   --capture-window f.ppm  écrit la fenêtre complète (UI incluse) à la dernière frame de --frames
//   --no-vsync        désactive la synchronisation verticale
int main(int argc, char** argv) {
    try {
        env::core::ApplicationConfig config;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--frames" && i + 1 < argc)
                config.maxFrames = std::strtoull(argv[++i], nullptr, 10);
            else if (arg == "--capture" && i + 1 < argc)
                config.capturePath = argv[++i];
            else if (arg == "--capture-window" && i + 1 < argc)
                config.windowCapturePath = argv[++i];
            else if (arg == "--no-vsync")
                config.vsync = false;
            else
                std::fprintf(stderr, "[EnvEngine] Option ignoree : %s\n", arg.c_str());
        }
        env::core::Application app(config);
        return app.Run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[EnvEngine] Exception fatale : %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "[EnvEngine] Exception fatale inconnue\n");
    }
    return EXIT_FAILURE;
}
