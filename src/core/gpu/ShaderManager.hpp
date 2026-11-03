#pragma once
// =============================================================================
//  src/core/gpu/ShaderManager.hpp — Programmes GLSL avec hot-reload (F5)
//
//  * LoadGraphics / LoadCompute / LoadProgram retournent un ShaderProgram& STABLE :
//    l'adresse ne change jamais, seul l'id GL change après un rechargement réussi.
//  * Préprocesseur : #include "x.glsl" (relatif au fichier, puis à la racine
//    shaders/), inclusion unique, directives #line pour des erreurs localisables.
//  * Préambule injecté après #version : bindings (UBO_/SSBO_/TEX_/IMG_*) et
//    constantes du contrat (CHUNK_SIZE_METERS, BIOME_*, FOLIAGE_*, ...).
//  * Échec de compilation : l'ancien programme reste actif ; l'erreur est
//    affichée dans la console et dans le panneau ImGui "Shaders".
//  * Uniforms via glProgramUniform* (DSA : pas besoin de glUseProgram).
// =============================================================================
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <glad/glad.h>
#include <glm/glm.hpp>

namespace env::gpu {

enum class ShaderStage : std::uint8_t { Vertex, TessControl, TessEvaluation, Geometry, Fragment, Compute };

struct ShaderStageSource {
    ShaderStage stage = ShaderStage::Vertex;
    std::string path;          ///< relatif à la racine shaders/ (ex. "render/terrain.vert")
    std::string inlineSource;  ///< si non vide : source embarquée (outils du core), path ignoré
};

class ShaderProgram {
public:
    GLuint             Id() const noexcept { return id_; }
    bool               IsValid() const noexcept { return id_ != 0; }
    bool               HasError() const noexcept { return !error_.empty(); }
    const std::string& Name() const noexcept { return name_; }
    const std::string& ErrorLog() const noexcept { return error_; }
    std::uint32_t      Generation() const noexcept { return generation_; }  ///< +1 à chaque build réussi

    void Use() const { glUseProgram(id_); }

    /// Location mise en cache (vidé à chaque rechargement). -1 si absent / optimisé.
    GLint UniformLocation(const char* name) const;

    void Set(const char* name, int value) const;
    void Set(const char* name, std::uint32_t value) const;
    void Set(const char* name, float value) const;
    void Set(const char* name, bool value) const;
    void Set(const char* name, const glm::vec2& value) const;
    void Set(const char* name, const glm::vec3& value) const;
    void Set(const char* name, const glm::vec4& value) const;
    void Set(const char* name, const glm::ivec2& value) const;
    void Set(const char* name, const glm::ivec3& value) const;
    void Set(const char* name, const glm::ivec4& value) const;
    void Set(const char* name, const glm::uvec2& value) const;
    void Set(const char* name, const glm::mat3& value) const;
    void Set(const char* name, const glm::mat4& value) const;

    /// glUseProgram + glDispatchCompute.
    void Dispatch(GLuint groupsX, GLuint groupsY = 1, GLuint groupsZ = 1) const;
    /// glUseProgram + glDispatchComputeIndirect (buffer lié sur GL_DISPATCH_INDIRECT_BUFFER).
    void DispatchIndirect(GLuint indirectBuffer, GLintptr offset = 0) const;

    /// Nombre de groupes pour couvrir `size` éléments avec des groupes de `localSize`.
    static constexpr GLuint GroupCount(GLuint size, GLuint localSize) noexcept {
        return (size + localSize - 1) / localSize;
    }

private:
    friend class ShaderManager;

    std::string                                    name_;
    std::vector<ShaderStageSource>                 sources_;
    GLuint                                         id_         = 0;
    std::uint32_t                                  generation_ = 0;
    std::string                                    error_;
    std::vector<std::filesystem::path>             dependencies_;
    std::vector<std::filesystem::file_time_type>   timestamps_;
    mutable std::unordered_map<std::string, GLint> locationCache_;
};

class ShaderManager {
public:
    explicit ShaderManager(std::filesystem::path shaderRoot);
    ~ShaderManager();

    ShaderManager(const ShaderManager&)            = delete;
    ShaderManager& operator=(const ShaderManager&) = delete;

    /// Programme vertex + fragment. Un nom déjà chargé retourne le programme existant.
    ShaderProgram& LoadGraphics(const std::string& name, const std::string& vertexPath, const std::string& fragmentPath);
    /// Programme compute.
    ShaderProgram& LoadCompute(const std::string& name, const std::string& computePath);
    /// Programme arbitraire (géométrie, tessellation, sources embarquées).
    ShaderProgram& LoadProgram(const std::string& name, std::vector<ShaderStageSource> stages);

    ShaderProgram* Find(std::string_view name);

    /// Demande un rechargement de tous les programmes (F5), traité au prochain Update().
    void RequestReloadAll() noexcept { reloadRequested_ = true; }
    /// Recharge immédiatement tous les programmes. Retourne true si tous compilent.
    bool ReloadAll();
    /// À appeler une fois par frame : rechargement demandé + surveillance des fichiers.
    void Update(double timeSeconds);

    /// Widgets du panneau "Shaders" (le core ouvre la fenêtre).
    void OnGui();

    void DestroyAll();

    const std::filesystem::path& Root() const noexcept { return root_; }
    const std::string&           Preamble() const noexcept { return preamble_; }
    std::size_t                  ErrorCount() const noexcept;
    std::size_t                  ProgramCount() const noexcept { return programs_.size(); }

    bool autoReload = false;  ///< surveille la date de modification des fichiers (toutes les 0.5 s)

private:
    bool Build(ShaderProgram& program);
    bool LoadStageSource(const ShaderStageSource& stage, const std::string& programName, std::string& outSource,
                         std::vector<std::string>& fileTable, std::vector<std::filesystem::path>& dependencies,
                         std::string& log);
    bool Preprocess(const std::string& text, const std::filesystem::path& currentFile, int fileIndex,
                    int firstLineNumber, int depth, std::string& out, std::vector<std::string>& fileTable,
                    std::vector<std::filesystem::path>& dependencies, std::vector<std::filesystem::path>& included,
                    std::string& log);
    std::filesystem::path ResolveInclude(const std::string& includePath, const std::filesystem::path& currentFile) const;
    bool                  DependenciesChanged(const ShaderProgram& program) const;
    static std::string    BuildPreamble();

    std::filesystem::path                       root_;
    std::string                                 preamble_;
    std::vector<std::unique_ptr<ShaderProgram>> programs_;
    bool                                        reloadRequested_ = false;
    double                                      lastPollTime_    = 0.0;
    std::uint32_t                               reloadCount_     = 0;
    std::string                                 lastReloadSummary_;
};

}  // namespace env::gpu
