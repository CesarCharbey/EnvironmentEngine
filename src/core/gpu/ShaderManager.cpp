// =============================================================================
//  src/core/gpu/ShaderManager.cpp
// =============================================================================
#include "core/gpu/ShaderManager.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <system_error>

#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>

#include "contracts/GpuBindings.hpp"
#include "contracts/WorldContracts.hpp"

namespace fs = std::filesystem;

namespace env::gpu {
namespace {

GLenum ToGlStage(ShaderStage stage) noexcept {
    switch (stage) {
        case ShaderStage::Vertex:         return GL_VERTEX_SHADER;
        case ShaderStage::TessControl:    return GL_TESS_CONTROL_SHADER;
        case ShaderStage::TessEvaluation: return GL_TESS_EVALUATION_SHADER;
        case ShaderStage::Geometry:       return GL_GEOMETRY_SHADER;
        case ShaderStage::Fragment:       return GL_FRAGMENT_SHADER;
        case ShaderStage::Compute:        return GL_COMPUTE_SHADER;
    }
    return GL_VERTEX_SHADER;
}

const char* StageName(ShaderStage stage) noexcept {
    switch (stage) {
        case ShaderStage::Vertex:         return "vertex";
        case ShaderStage::TessControl:    return "tess-control";
        case ShaderStage::TessEvaluation: return "tess-eval";
        case ShaderStage::Geometry:       return "geometry";
        case ShaderStage::Fragment:       return "fragment";
        case ShaderStage::Compute:        return "compute";
    }
    return "?";
}

bool ReadTextFile(const fs::path& path, std::string& out) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    std::ostringstream ss;
    ss << file.rdbuf();
    out = ss.str();
    return true;
}

std::string CamelToUpperSnake(std::string_view name) {
    std::string out;
    for (std::size_t i = 0; i < name.size(); ++i) {
        const auto c = static_cast<unsigned char>(name[i]);
        if (i > 0 && std::isupper(c) && !std::isupper(static_cast<unsigned char>(name[i - 1]))) out += '_';
        out += static_cast<char>(std::toupper(c));
    }
    return out;
}

/// Reconnaît `#include "chemin"` ou `#include <chemin>`.
bool ParseInclude(std::string_view line, std::string& outPath) {
    std::size_t i = 0;
    auto skipSpaces = [&] {
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
    };
    skipSpaces();
    if (i >= line.size() || line[i] != '#') return false;
    ++i;
    skipSpaces();
    if (line.substr(i, 7) != "include") return false;
    i += 7;
    skipSpaces();
    if (i >= line.size()) return false;
    const char open  = line[i];
    const char close = open == '"' ? '"' : (open == '<' ? '>' : '\0');
    if (close == '\0') return false;
    const std::size_t end = line.find(close, i + 1);
    if (end == std::string_view::npos) return false;
    outPath = std::string(line.substr(i + 1, end - i - 1));
    return true;
}

std::vector<fs::file_time_type> CollectTimestamps(const std::vector<fs::path>& files) {
    std::vector<fs::file_time_type> stamps;
    stamps.reserve(files.size());
    for (const fs::path& f : files) {
        std::error_code ec;
        const fs::file_time_type t = fs::last_write_time(f, ec);
        stamps.push_back(ec ? fs::file_time_type::min() : t);
    }
    return stamps;
}

std::string ShaderInfoLog(GLuint shader) {
    GLint length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
    glGetShaderInfoLog(shader, length, nullptr, log.data());
    while (!log.empty() && (log.back() == '\0' || log.back() == '\n')) log.pop_back();
    return log;
}

std::string ProgramInfoLog(GLuint program) {
    GLint length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
    std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
    glGetProgramInfoLog(program, length, nullptr, log.data());
    while (!log.empty() && (log.back() == '\0' || log.back() == '\n')) log.pop_back();
    return log;
}

}  // namespace

// =============================================================================
// ShaderProgram
// =============================================================================
GLint ShaderProgram::UniformLocation(const char* name) const {
    if (id_ == 0) return -1;
    if (auto it = locationCache_.find(name); it != locationCache_.end()) return it->second;
    const GLint location = glGetUniformLocation(id_, name);
    locationCache_.emplace(name, location);
    return location;
}

#define ENV_UNIFORM_SET(Type, Call)                         \
    void ShaderProgram::Set(const char* name, Type v) const { \
        const GLint loc = UniformLocation(name);            \
        if (loc >= 0) Call;                                 \
    }

ENV_UNIFORM_SET(int, glProgramUniform1i(id_, loc, v))
ENV_UNIFORM_SET(std::uint32_t, glProgramUniform1ui(id_, loc, v))
ENV_UNIFORM_SET(float, glProgramUniform1f(id_, loc, v))
ENV_UNIFORM_SET(bool, glProgramUniform1i(id_, loc, v ? 1 : 0))
ENV_UNIFORM_SET(const glm::vec2&, glProgramUniform2fv(id_, loc, 1, glm::value_ptr(v)))
ENV_UNIFORM_SET(const glm::vec3&, glProgramUniform3fv(id_, loc, 1, glm::value_ptr(v)))
ENV_UNIFORM_SET(const glm::vec4&, glProgramUniform4fv(id_, loc, 1, glm::value_ptr(v)))
ENV_UNIFORM_SET(const glm::ivec2&, glProgramUniform2iv(id_, loc, 1, glm::value_ptr(v)))
ENV_UNIFORM_SET(const glm::ivec3&, glProgramUniform3iv(id_, loc, 1, glm::value_ptr(v)))
ENV_UNIFORM_SET(const glm::ivec4&, glProgramUniform4iv(id_, loc, 1, glm::value_ptr(v)))
ENV_UNIFORM_SET(const glm::uvec2&, glProgramUniform2uiv(id_, loc, 1, glm::value_ptr(v)))
ENV_UNIFORM_SET(const glm::mat3&, glProgramUniformMatrix3fv(id_, loc, 1, GL_FALSE, glm::value_ptr(v)))
ENV_UNIFORM_SET(const glm::mat4&, glProgramUniformMatrix4fv(id_, loc, 1, GL_FALSE, glm::value_ptr(v)))

#undef ENV_UNIFORM_SET

void ShaderProgram::Dispatch(GLuint groupsX, GLuint groupsY, GLuint groupsZ) const {
    if (id_ == 0) return;
    glUseProgram(id_);
    glDispatchCompute(std::max(groupsX, 1u), std::max(groupsY, 1u), std::max(groupsZ, 1u));
}

void ShaderProgram::DispatchIndirect(GLuint indirectBuffer, GLintptr offset) const {
    if (id_ == 0) return;
    glUseProgram(id_);
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, indirectBuffer);
    glDispatchComputeIndirect(offset);
}

// =============================================================================
// ShaderManager
// =============================================================================
ShaderManager::ShaderManager(fs::path shaderRoot) : root_(std::move(shaderRoot)), preamble_(BuildPreamble()) {
    std::printf("[ShaderManager] Racine des shaders : %s\n", root_.string().c_str());
}

ShaderManager::~ShaderManager() { DestroyAll(); }

ShaderProgram& ShaderManager::LoadGraphics(const std::string& name, const std::string& vertexPath,
                                           const std::string& fragmentPath) {
    std::vector<ShaderStageSource> stages(2);
    stages[0].stage = ShaderStage::Vertex;
    stages[0].path  = vertexPath;
    stages[1].stage = ShaderStage::Fragment;
    stages[1].path  = fragmentPath;
    return LoadProgram(name, std::move(stages));
}

ShaderProgram& ShaderManager::LoadCompute(const std::string& name, const std::string& computePath) {
    std::vector<ShaderStageSource> stages(1);
    stages[0].stage = ShaderStage::Compute;
    stages[0].path  = computePath;
    return LoadProgram(name, std::move(stages));
}

ShaderProgram& ShaderManager::LoadProgram(const std::string& name, std::vector<ShaderStageSource> stages) {
    if (ShaderProgram* existing = Find(name)) return *existing;
    auto program      = std::make_unique<ShaderProgram>();
    program->name_    = name;
    program->sources_ = std::move(stages);
    Build(*program);
    programs_.push_back(std::move(program));
    return *programs_.back();
}

ShaderProgram* ShaderManager::Find(std::string_view name) {
    for (auto& p : programs_)
        if (p->name_ == name) return p.get();
    return nullptr;
}

bool ShaderManager::ReloadAll() {
    std::size_t ok = 0, failed = 0;
    for (auto& p : programs_) (Build(*p) ? ok : failed)++;
    ++reloadCount_;
    lastReloadSummary_ = "Rechargement #" + std::to_string(reloadCount_) + " : " + std::to_string(ok) + " OK, " +
                         std::to_string(failed) + " en erreur";
    std::printf("[ShaderManager] %s\n", lastReloadSummary_.c_str());
    return failed == 0;
}

void ShaderManager::Update(double timeSeconds) {
    if (reloadRequested_) {
        reloadRequested_ = false;
        ReloadAll();
    }
    if (autoReload && timeSeconds - lastPollTime_ > 0.5) {
        lastPollTime_ = timeSeconds;
        for (auto& p : programs_) {
            if (!DependenciesChanged(*p)) continue;
            const bool ok = Build(*p);
            std::printf("[ShaderManager] Auto-reload '%s' : %s\n", p->name_.c_str(), ok ? "OK" : "ERREUR");
        }
    }
}

void ShaderManager::DestroyAll() {
    for (auto& p : programs_) {
        if (p->id_ != 0) {
            glDeleteProgram(p->id_);
            p->id_ = 0;
        }
        p->locationCache_.clear();
    }
}

std::size_t ShaderManager::ErrorCount() const noexcept {
    return static_cast<std::size_t>(
        std::count_if(programs_.begin(), programs_.end(), [](const auto& p) { return p->HasError(); }));
}

bool ShaderManager::DependenciesChanged(const ShaderProgram& program) const {
    const std::vector<fs::file_time_type> now = CollectTimestamps(program.dependencies_);
    return now != program.timestamps_;
}

bool ShaderManager::Build(ShaderProgram& program) {
    std::string           log;
    std::vector<fs::path> dependencies;
    std::vector<GLuint>   shaders;
    bool                  ok = true;

    for (const ShaderStageSource& stage : program.sources_) {
        std::string              source;
        std::vector<std::string> fileTable;
        if (!LoadStageSource(stage, program.name_, source, fileTable, dependencies, log)) {
            ok = false;
            continue;
        }
        const GLuint shader = glCreateShader(ToGlStage(stage.stage));
        const char*  text   = source.c_str();
        glShaderSource(shader, 1, &text, nullptr);
        glCompileShader(shader);
        GLint status = GL_FALSE;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
        if (status != GL_TRUE) {
            log += "[" + std::string(StageName(stage.stage)) + "] " +
                   (stage.inlineSource.empty() ? stage.path : std::string("<inline>")) + "\n" + ShaderInfoLog(shader) +
                   "\n  Index des fichiers (#line) :\n";
            for (std::size_t i = 0; i < fileTable.size(); ++i)
                log += "    " + std::to_string(i) + " = " + fileTable[i] + "\n";
            glDeleteShader(shader);
            ok = false;
            continue;
        }
        shaders.push_back(shader);
    }

    GLuint linked = 0;
    if (ok) {
        linked = glCreateProgram();
        for (GLuint s : shaders) glAttachShader(linked, s);
        glLinkProgram(linked);
        GLint status = GL_FALSE;
        glGetProgramiv(linked, GL_LINK_STATUS, &status);
        if (status != GL_TRUE) {
            log += "[link]\n" + ProgramInfoLog(linked) + "\n";
            ok = false;
        }
        for (GLuint s : shaders) glDetachShader(linked, s);
    }
    for (GLuint s : shaders) glDeleteShader(s);

    program.dependencies_ = dependencies;
    program.timestamps_   = CollectTimestamps(dependencies);

    if (!ok) {
        if (linked != 0) glDeleteProgram(linked);
        program.error_ = log.empty() ? std::string("Erreur inconnue") : log;
        std::fprintf(stderr, "[ShaderManager] ERREUR '%s' (%s)\n%s\n", program.name_.c_str(),
                     program.id_ != 0 ? "ancien programme conserve" : "programme indisponible", program.error_.c_str());
        return false;
    }

    if (program.id_ != 0) glDeleteProgram(program.id_);
    program.id_ = linked;
    ++program.generation_;
    program.error_.clear();
    program.locationCache_.clear();
    glObjectLabel(GL_PROGRAM, linked, -1, program.name_.c_str());
    return true;
}

bool ShaderManager::LoadStageSource(const ShaderStageSource& stage, const std::string& programName,
                                    std::string& outSource, std::vector<std::string>& fileTable,
                                    std::vector<fs::path>& dependencies, std::string& log) {
    std::string text;
    fs::path    file;
    if (!stage.inlineSource.empty()) {
        text = stage.inlineSource;
        file = root_ / ("inline_" + programName);
        fileTable.push_back("<inline:" + programName + ">");
    } else {
        file = root_ / stage.path;
        fileTable.push_back(file.generic_string());
        dependencies.push_back(file);
        if (!ReadTextFile(file, text)) {
            log += "Fichier introuvable : " + file.string() + "\n";
            return false;
        }
    }

    // #version doit rester la première directive : le préambule est inséré juste après.
    std::size_t pos = 0, restStart = std::string::npos;
    int         versionLine = 1;
    std::string version;
    while (pos < text.size()) {
        std::size_t end = text.find('\n', pos);
        if (end == std::string::npos) end = text.size();
        std::string line = text.substr(pos, end - pos);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::size_t first = line.find_first_not_of(" \t");
        if (first != std::string::npos && line.compare(first, 8, "#version") == 0) {
            version   = line;
            restStart = std::min(end + 1, text.size());
            break;
        }
        ++versionLine;
        pos = end + 1;
    }
    if (restStart == std::string::npos) {
        log += fileTable.front() + " : directive #version manquante\n";
        return false;
    }

    outSource = version + "\n" + preamble_ + "#line " + std::to_string(versionLine + 1) + " 0\n";
    std::vector<fs::path> included;
    return Preprocess(text.substr(restStart), file, 0, versionLine + 1, 0, outSource, fileTable, dependencies, included,
                      log);
}

bool ShaderManager::Preprocess(const std::string& text, const fs::path& currentFile, int fileIndex, int firstLineNumber,
                               int depth, std::string& out, std::vector<std::string>& fileTable,
                               std::vector<fs::path>& dependencies, std::vector<fs::path>& included, std::string& log) {
    if (depth > 16) {
        log += "Profondeur d'#include > 16 (inclusion circulaire ?)\n";
        return false;
    }
    bool        ok     = true;
    int         lineNo = firstLineNumber;
    std::size_t pos    = 0;
    while (pos < text.size()) {
        std::size_t end = text.find('\n', pos);
        if (end == std::string::npos) end = text.size();
        std::string line = text.substr(pos, end - pos);
        if (!line.empty() && line.back() == '\r') line.pop_back();

        std::string includePath;
        if (ParseInclude(line, includePath)) {
            const fs::path resolved = ResolveInclude(includePath, currentFile);
            std::string    includeText;
            if (resolved.empty() || !ReadTextFile(resolved, includeText)) {
                log += fileTable[static_cast<std::size_t>(fileIndex)] + "(" + std::to_string(lineNo) +
                       ") : #include introuvable \"" + includePath + "\"\n";
                ok = false;
                out += "\n";
            } else if (std::find(included.begin(), included.end(), resolved) != included.end()) {
                out += "// (deja inclus) " + includePath + "\n";
            } else {
                included.push_back(resolved);
                dependencies.push_back(resolved);
                const int index = static_cast<int>(fileTable.size());
                fileTable.push_back(resolved.generic_string());
                out += "#line 1 " + std::to_string(index) + "\n";
                ok &= Preprocess(includeText, resolved, index, 1, depth + 1, out, fileTable, dependencies, included, log);
                out += "#line " + std::to_string(lineNo + 1) + " " + std::to_string(fileIndex) + "\n";
            }
        } else {
            out += line;
            out += '\n';
        }
        ++lineNo;
        pos = end + 1;
    }
    return ok;
}

fs::path ShaderManager::ResolveInclude(const std::string& includePath, const fs::path& currentFile) const {
    std::error_code ec;
    const fs::path  candidates[] = {currentFile.parent_path() / includePath, root_ / includePath};
    for (const fs::path& candidate : candidates) {
        if (fs::exists(candidate, ec)) {
            fs::path canonical = fs::weakly_canonical(candidate, ec);
            return ec ? candidate : canonical;
        }
    }
    return {};
}

std::string ShaderManager::BuildPreamble() {
    std::string p;
    p += "// ---- Preambule genere par ShaderManager depuis include/contracts (ne pas editer) ----\n";
    p += "#define ENV_SHADER_PREAMBLE 1\n";

    auto emitTable = [&p](std::string_view prefix, const auto& table) {
        for (const bind::SlotInfo& slot : table) {
            p += "#define ";
            p += prefix;
            p += slot.name;
            p += ' ';
            p += std::to_string(slot.slot);
            p += '\n';
        }
    };
    emitTable(bind::kGlslPrefixUbo, bind::kUboTable);
    emitTable(bind::kGlslPrefixSsbo, bind::kSsboTable);
    emitTable(bind::kGlslPrefixTexture, bind::kTextureTable);
    emitTable(bind::kGlslPrefixImage, bind::kImageTable);

    char buffer[128];
    std::snprintf(buffer, sizeof(buffer), "#define CHUNK_SIZE_METERS %.6f\n", static_cast<double>(kChunkSizeMeters));
    p += buffer;
    std::snprintf(buffer, sizeof(buffer), "#define RIVER_SDF_FAR %.6f\n", static_cast<double>(kRiverSdfFar));
    p += buffer;
    std::snprintf(buffer, sizeof(buffer), "#define PHYSICS_PROMOTION_RADIUS %.6f\n",
                  static_cast<double>(kPhysicsPromotionRadius));
    p += buffer;
    p += "#define CHUNK_TEX_RESOLUTION " + std::to_string(kChunkTexResolution) + "\n";
    p += "#define ORIGIN_REBASE_CHUNKS " + std::to_string(kOriginRebaseChunks) + "\n";
    p += "#define FOLIAGE_ARCHETYPE_COUNT " + std::to_string(kFoliageArchetypeCount) + "\n";
    p += "#define FOLIAGE_MAX_INSTANCES_PER_CHUNK " + std::to_string(kFoliageMaxInstancesPerChunk) + "u\n";

    for (std::size_t i = 0; i < kBiomeCount; ++i)
        p += "#define BIOME_" + CamelToUpperSnake(kBiomeNames[i]) + " " + std::to_string(i) + "\n";
    p += "#define BIOME_COUNT " + std::to_string(kBiomeCount) + "\n";

    for (std::size_t i = 0; i < kFoliageArchetypeCount; ++i)
        p += "#define FOLIAGE_" + CamelToUpperSnake(kFoliageArchetypeNames[i]) + " " + std::to_string(i) + "\n";

    p += "#define CHUNK_READY_HEIGHT_NORMAL " + std::to_string(ChunkReady::HeightNormal) + "u\n";
    p += "#define CHUNK_READY_BIOME_HYDRO " + std::to_string(ChunkReady::BiomeHydro) + "u\n";
    p += "#define CHUNK_READY_FOLIAGE " + std::to_string(ChunkReady::Foliage) + "u\n";
    p += "#define CHUNK_READY_PHYSICS_COLLIDER " + std::to_string(ChunkReady::PhysicsCollider) + "u\n";
    p += "// ---- fin du preambule ----\n";
    return p;
}

void ShaderManager::OnGui() {
    if (ImGui::Button("Recharger tout (F5)")) RequestReloadAll();
    ImGui::SameLine();
    ImGui::Checkbox("Auto reload", &autoReload);
    ImGui::SameLine();
    ImGui::TextDisabled("(surveille les fichiers)");
    ImGui::TextDisabled("Racine : %s", root_.string().c_str());
    if (!lastReloadSummary_.empty()) ImGui::TextUnformatted(lastReloadSummary_.c_str());

    const std::size_t errors = ErrorCount();
    if (errors > 0)
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.3f, 1.0f), "%zu programme(s) en erreur (ancienne version active)", errors);
    else
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "%zu programme(s), aucune erreur", programs_.size());

    if (ImGui::BeginTable("##shaderTable", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Programme");
        ImGui::TableSetupColumn("Etat");
        ImGui::TableSetupColumn("Gen.");
        ImGui::TableSetupColumn("Fichiers");
        ImGui::TableHeadersRow();
        for (const auto& p : programs_) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(p->name_.c_str());
            ImGui::TableNextColumn();
            if (p->HasError())
                ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.3f, 1.0f), p->IsValid() ? "ERREUR (ancien actif)" : "ERREUR");
            else
                ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "OK (id %u)", p->id_);
            ImGui::TableNextColumn();
            ImGui::Text("%u", p->generation_);
            ImGui::TableNextColumn();
            ImGui::Text("%zu", p->dependencies_.size());
            if (ImGui::IsItemHovered() && !p->dependencies_.empty()) {
                ImGui::BeginTooltip();
                for (const fs::path& d : p->dependencies_) ImGui::TextUnformatted(d.generic_string().c_str());
                ImGui::EndTooltip();
            }
        }
        ImGui::EndTable();
    }

    for (const auto& p : programs_) {
        if (!p->HasError()) continue;
        ImGui::PushID(p.get());
        if (ImGui::CollapsingHeader(("Erreur : " + p->name_).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.55f, 0.5f, 1.0f));
            ImGui::TextUnformatted(p->error_.c_str());
            ImGui::PopStyleColor();
        }
        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Preambule injecte apres #version")) ImGui::TextUnformatted(preamble_.c_str());
}

}  // namespace env::gpu
