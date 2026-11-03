# =============================================================================
#  EnvEngine - Moteur environnemental temps reel a monde infini
#  C++20 / OpenGL 4.6 Core (DSA exclusif)
#  Dependances : GLFW, glad, glm, Dear ImGui (docking), EnTT, JoltPhysics
#
#  Toutes les dependances sont telechargees par FetchContent (URL + SHA256 :
#  builds reproductibles, aucune dependance systeme a installer a la main).
#  Prerequis : CMake >= 3.25, compilateur C++20, Python 3 (generateur glad,
#  bibliotheque standard uniquement), GPU OpenGL 4.6.
#
#  ADR-002 "Zero-Conflict" : il n'y a AUCUNE liste de sources dans ce fichier.
#  Tout .cpp sous src/ est compile automatiquement (GLOB_RECURSE CONFIGURE_DEPENDS)
#  et s'auto-enregistre via REGISTER_SYSTEM. Ce fichier ne doit pas changer quand
#  un developpeur ajoute un systeme.
# =============================================================================
cmake_minimum_required(VERSION 3.25)

project(EnvEngine VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Type de build (Debug, Release, RelWithDebInfo)" FORCE)
endif()

option(ENV_ENFORCE_BOUNDARIES "Echec du build si world_engine <-> render_engine s'incluent ou si l'API GL bind-to-edit est utilisee" ON)
option(ENV_WARNINGS_AS_ERRORS "Traiter les warnings du moteur comme des erreurs" OFF)
option(ENV_GL_DEBUG "Contexte OpenGL de debug + glDebugMessageCallback meme hors Debug" OFF)

# glad 0.1.x declare un cmake_minimum_required tres ancien que CMake >= 4.0
# refuse sans cette variable (sans effet sur nos propres cibles).
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# -----------------------------------------------------------------------------
# GLFW 3.4 - fenetrage / contexte OpenGL / entrees
# -----------------------------------------------------------------------------
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)

if(UNIX AND NOT APPLE)
    # Le backend Wayland de GLFW exige wayland-scanner + en-tetes de dev.
    # S'ils sont absents, seul le backend X11 est construit (fonctionne sous XWayland).
    find_program(WAYLAND_SCANNER_EXECUTABLE wayland-scanner)
    find_package(PkgConfig QUIET)
    set(_env_has_wayland OFF)
    if(WAYLAND_SCANNER_EXECUTABLE AND PkgConfig_FOUND)
        pkg_check_modules(_ENV_WL QUIET wayland-client wayland-cursor wayland-egl xkbcommon)
        if(_ENV_WL_FOUND)
            set(_env_has_wayland ON)
        endif()
    endif()
    set(GLFW_BUILD_WAYLAND ${_env_has_wayland} CACHE BOOL "" FORCE)
    set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)
    message(STATUS "[EnvEngine] GLFW backend Wayland : ${_env_has_wayland}")
endif()

FetchContent_Declare(glfw
    URL      https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip
    URL_HASH SHA256=b5ec004b2712fd08e8861dc271428f048775200a2df719ccf575143ba749a3e9
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM)

# -----------------------------------------------------------------------------
# glad 0.1.36 - loader OpenGL 4.6 Core (genere a la configuration, spec embarquee)
# -----------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(GLAD_PROFILE      "core"   CACHE STRING "" FORCE)
set(GLAD_API          "gl=4.6" CACHE STRING "" FORCE)
set(GLAD_GENERATOR    "c"      CACHE STRING "" FORCE)
set(GLAD_EXTENSIONS   ""       CACHE STRING "" FORCE)
set(GLAD_REPRODUCIBLE ON       CACHE BOOL   "" FORCE)  # n'utilise que le gl.xml embarque (pas de reseau)
set(GLAD_INSTALL      OFF      CACHE BOOL   "" FORCE)
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

FetchContent_Declare(glad
    URL      https://github.com/Dav1dde/glad/archive/refs/tags/v0.1.36.tar.gz
    URL_HASH SHA256=8470ed1b0e9fbe88e10c34770505c8a1dc8ccb78cadcf673331aaf5224f963d2
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM)

# -----------------------------------------------------------------------------
# glm 1.0.1 - mathematiques (header-only)
# -----------------------------------------------------------------------------
set(GLM_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_INSTALL OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_LIBRARY OFF CACHE BOOL "" FORCE)

FetchContent_Declare(glm
    URL      https://github.com/g-truc/glm/archive/refs/tags/1.0.1.tar.gz
    URL_HASH SHA256=9f3174561fd26904b23f0db5e560971cbf9b3cbda0b280f04d5c379d03bf234c
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM)

# -----------------------------------------------------------------------------
# EnTT 3.15.0 - ECS (header-only)
# -----------------------------------------------------------------------------
FetchContent_Declare(EnTT
    URL      https://github.com/skypjack/entt/archive/refs/tags/v3.15.0.tar.gz
    URL_HASH SHA256=01466fcbf77618a79b62891510c0bbf25ac2804af5751c84982b413852234d66
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM)

# -----------------------------------------------------------------------------
# Dear ImGui 1.91.9b (branche docking) - pas de CMakeLists amont : cible locale
# -----------------------------------------------------------------------------
FetchContent_Declare(imgui
    URL      https://github.com/ocornut/imgui/archive/refs/tags/v1.91.9b-docking.tar.gz
    URL_HASH SHA256=466fdef9b18de15f0bb6e288e3d00ffa3d82200ec458ce5e4f724a161d9528a5
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

# -----------------------------------------------------------------------------
# JoltPhysics 5.6.0 - physique (ADR-006)
# -----------------------------------------------------------------------------
set(TARGET_UNIT_TESTS       OFF CACHE BOOL "" FORCE)
set(TARGET_HELLO_WORLD      OFF CACHE BOOL "" FORCE)
set(TARGET_PERFORMANCE_TEST OFF CACHE BOOL "" FORCE)
set(TARGET_SAMPLES          OFF CACHE BOOL "" FORCE)
set(TARGET_VIEWER           OFF CACHE BOOL "" FORCE)
set(ENABLE_ALL_WARNINGS     OFF CACHE BOOL "" FORCE)  # sinon -Werror sur les compilateurs recents
set(ENABLE_INSTALL          OFF CACHE BOOL "" FORCE)
set(ENABLE_OBJECT_STREAM    OFF CACHE BOOL "" FORCE)
set(INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)
set(GENERATE_DEBUG_SYMBOLS  OFF CACHE BOOL "" FORCE)
set(CPP_RTTI_ENABLED        ON  CACHE BOOL "" FORCE)  # nos classes derivent d'interfaces Jolt
set(FLOATING_POINT_EXCEPTIONS_ENABLED OFF CACHE BOOL "" FORCE)
set(USE_STATIC_MSVC_RUNTIME_LIBRARY OFF CACHE BOOL "" FORCE)
# Backends de calcul GPU de Jolt : inutiles ici (et exigeraient le SDK Vulkan/DX12/Metal).
set(JPH_USE_DX12            OFF CACHE BOOL "" FORCE)
set(JPH_USE_VK              OFF CACHE BOOL "" FORCE)
set(JPH_USE_MTL             OFF CACHE BOOL "" FORCE)
set(JPH_USE_CPU_COMPUTE     OFF CACHE BOOL "" FORCE)

FetchContent_Declare(JoltPhysics
    URL      https://github.com/jrouwe/JoltPhysics/archive/refs/tags/v5.6.0.tar.gz
    URL_HASH SHA256=6e069ee0172478cc78182047aac87e5310ba14a67a53348ae14cc37801fd3f8e
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR Build
    SYSTEM)

# Les CMakeLists amont (glad, glm) declarent des versions minimales anciennes :
# on masque leurs avertissements de deprecation, sans toucher a nos propres cibles.
set(_env_warn_deprecated ON)
if(DEFINED CMAKE_WARN_DEPRECATED AND NOT "${CMAKE_WARN_DEPRECATED}" STREQUAL "")
    set(_env_warn_deprecated ${CMAKE_WARN_DEPRECATED})
endif()
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw glad glm EnTT imgui JoltPhysics)
set(CMAKE_WARN_DEPRECATED ${_env_warn_deprecated} CACHE BOOL "" FORCE)

find_package(Threads REQUIRED)

# Dear ImGui : bibliotheque statique locale (coeur + backends GLFW / OpenGL3).
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui SYSTEM PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw ${CMAKE_DL_LIBS})

# -----------------------------------------------------------------------------
# Executable du moteur (ADR-002 : collecte automatique, aucune liste a editer)
# -----------------------------------------------------------------------------
# IMPORTANT : les sources sont compilees DIRECTEMENT dans l'executable. Une
# bibliotheque statique intermediaire ferait eliminer par l'editeur de liens les
# .o jamais references... et donc les enregistrements REGISTER_SYSTEM.
file(GLOB_RECURSE ENV_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE ENV_HEADERS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)
file(GLOB_RECURSE ENV_SHADERS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.geom
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.tesc
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.tese
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp)

add_executable(EnvEngine ${ENV_SOURCES} ${ENV_HEADERS} ${ENV_SHADERS})
set_source_files_properties(${ENV_SHADERS} PROPERTIES HEADER_FILE_ONLY TRUE)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${ENV_SOURCES} ${ENV_HEADERS} ${ENV_SHADERS})

target_include_directories(EnvEngine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include   # "contracts/..."
    ${CMAKE_CURRENT_SOURCE_DIR}/src)      # "core/..."

target_link_libraries(EnvEngine PRIVATE glfw glad glm::glm EnTT::EnTT imgui Jolt Threads::Threads)

target_compile_definitions(EnvEngine PRIVATE
    GLFW_INCLUDE_NONE
    GLM_ENABLE_EXPERIMENTAL
    # Chemin absolu des shaders sources : le hot-reload (F5) recharge directement
    # les fichiers que vous editez. Surcharge possible : variable d'env ENV_SHADER_DIR.
    ENV_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    $<$<OR:$<CONFIG:Debug>,$<BOOL:${ENV_GL_DEBUG}>>:ENV_GL_DEBUG=1>)

if(MSVC)
    target_compile_options(EnvEngine PRIVATE /W4 /permissive- /utf-8 /Zc:__cplusplus)
    if(ENV_WARNINGS_AS_ERRORS)
        target_compile_options(EnvEngine PRIVATE /WX)
    endif()
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT EnvEngine)
else()
    target_compile_options(EnvEngine PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
    if(ENV_WARNINGS_AS_ERRORS)
        target_compile_options(EnvEngine PRIVATE -Werror)
    endif()
endif()

# -----------------------------------------------------------------------------
# Garde-fou d'architecture (ADR-002) execute a CHAQUE build :
#   1. src/world_engine  n'inclut jamais src/render_engine (et reciproquement)
#   2. src/core et include/ n'incluent jamais un dossier metier
#   3. aucune fonction OpenGL "bind-to-edit" dans src/ (DSA exclusif, AGENTS.md 5)
# Le script est genere dans le dossier de build : l'arborescence source reste intacte.
# -----------------------------------------------------------------------------
if(ENV_ENFORCE_BOUNDARIES)
    set(_env_boundary_script ${CMAKE_CURRENT_BINARY_DIR}/env_check_boundaries.cmake)
    file(WRITE ${_env_boundary_script} [=[
# Genere par CMakeLists.txt (ADR-002) - ne pas editer.
# cmake -DENV_SRC_DIR=<src> -DENV_INCLUDE_DIR=<include> -P env_check_boundaries.cmake
set(_violations "")

# function (et non macro) : les arguments ne sont evalues qu'une fois, les
# antislashs des regex restent intacts.
function(_env_scan _files_var _regex _message)
    foreach(_f IN LISTS ${_files_var})
        file(STRINGS "${_f}" _hits REGEX "${_regex}")
        if(_hits)
            # On rapporte le motif fautif lui-meme (les ';' du C++ sont des separateurs de liste CMake).
            list(JOIN _hits " " _joined)
            string(REGEX MATCH "${_regex}" _first "${_joined}")
            string(STRIP "${_first}" _first)
            file(RELATIVE_PATH _rel "${ENV_SRC_DIR}/.." "${_f}")
            list(APPEND _violations "  ${_rel} : ${_message}\n      > ${_first}")
        endif()
    endforeach()
    set(_violations "${_violations}" PARENT_SCOPE)
endfunction()

file(GLOB_RECURSE _world  "${ENV_SRC_DIR}/world_engine/*.cpp"  "${ENV_SRC_DIR}/world_engine/*.hpp"  "${ENV_SRC_DIR}/world_engine/*.h")
file(GLOB_RECURSE _render "${ENV_SRC_DIR}/render_engine/*.cpp" "${ENV_SRC_DIR}/render_engine/*.hpp" "${ENV_SRC_DIR}/render_engine/*.h")
file(GLOB_RECURSE _core   "${ENV_SRC_DIR}/core/*.cpp" "${ENV_SRC_DIR}/core/*.hpp" "${ENV_SRC_DIR}/main.cpp" "${ENV_INCLUDE_DIR}/*.hpp")
file(GLOB_RECURSE _all    "${ENV_SRC_DIR}/*.cpp" "${ENV_SRC_DIR}/*.hpp" "${ENV_SRC_DIR}/*.h")

_env_scan(_world  "^[ \t]*#[ \t]*include[ \t]*[<\"][^>\"]*render_engine/"
          "world_engine (Dev 1) ne doit jamais inclure render_engine (Dev 2)")
_env_scan(_render "^[ \t]*#[ \t]*include[ \t]*[<\"][^>\"]*world_engine/"
          "render_engine (Dev 2) ne doit jamais inclure world_engine (Dev 1)")
_env_scan(_core   "^[ \t]*#[ \t]*include[ \t]*[<\"][^>\"]*(world_engine|render_engine)/"
          "core / contracts ne doivent jamais inclure un dossier metier")
_env_scan(_all    "(^|[^A-Za-z0-9_])(glGenTextures|glBindTexture|glActiveTexture|glGenerateMipmap|glGenBuffers|glBufferData|glBufferSubData|glGenFramebuffers|glDrawBuffers|glBlitFramebuffer|glGenVertexArrays|glVertexAttribPointer|glVertexAttribIPointer|glEnableVertexAttribArray|glUniformBlockBinding)[ \t]*\\("
          "API OpenGL bind-to-edit interdite : utiliser l'equivalent DSA (AGENTS.md section 5)")
_env_scan(_all    "(^|[^A-Za-z0-9_])(glTexImage[123]D|glTexStorage[123]D|glTexSubImage[123]D|glTexParameter[A-Za-z]*|glMapBuffer[A-Za-z]*|glFramebufferTexture[A-Za-z0-9]*|glUniform[1-4][a-z]+|glUniformMatrix[0-9x]+fv)[ \t]*\\("
          "API OpenGL bind-to-edit interdite : utiliser l'equivalent DSA (AGENTS.md section 5)")

if(_violations)
    list(JOIN _violations "\n" _report)
    message(FATAL_ERROR "\n[EnvEngine] Violation(s) des regles d'architecture (AGENTS.md / ADR-002) :\n${_report}\n")
endif()
]=])

    add_custom_target(env_boundary_check
        COMMAND ${CMAKE_COMMAND}
                -DENV_SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}/src
                -DENV_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/include
                -P ${_env_boundary_script}
        COMMENT "EnvEngine : verification des frontieres Dev 1 / Dev 2 et de l'usage DSA"
        VERBATIM)
    add_dependencies(EnvEngine env_boundary_check)
endif()

message(STATUS "[EnvEngine] Shaders : ${CMAKE_CURRENT_SOURCE_DIR}/shaders (hot-reload F5)")
