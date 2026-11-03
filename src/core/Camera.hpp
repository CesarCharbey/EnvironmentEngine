#pragma once
// =============================================================================
//  src/core/Camera.hpp — Caméra freefly (position monde double, ADR-007)
//
//  * Position stockée en double (monde infini). Les matrices sont construites
//    relativement à l'origine de rendu flottante : ViewMatrix(renderOriginWorld).
//  * Projection Reverse-Z à plan lointain infini (clip [0, 1], depth clear = 0,
//    glDepthFunc(GL_GREATER)).
//  * Convention : Y vers le haut, yaw 0 = regarde vers -Z, yaw 90 = vers +X.
//  * Aucune dépendance à GLFW : Application traduit les entrées en CameraInput.
// =============================================================================
#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

namespace env::core {

struct CameraInput {
    glm::vec3 move{0.0f};       ///< x = droite, y = haut (monde), z = avant ; composantes dans [-1, 1]
    glm::vec2 lookDelta{0.0f};  ///< déplacement souris en pixels
    float     speedSteps = 0.0f;  ///< crans de molette (vitesse multipliée par 1.2 par cran)
    bool      fast = false;
    bool      slow = false;
};

class Camera {
public:
    float fovYDegrees     = 60.0f;
    float nearPlane       = 0.1f;
    float renderDistance  = 6000.0f;  ///< distance de culling / brouillard (le plan lointain est infini)
    float moveSpeed       = 30.0f;    ///< m/s
    float fastMultiplier  = 8.0f;
    float slowMultiplier  = 0.15f;
    float lookSensitivity = 0.12f;    ///< degrés par pixel

    void SetWorldPosition(const glm::dvec3& position) noexcept { position_ = position; }
    const glm::dvec3& WorldPosition() const noexcept { return position_; }

    void SetYawPitch(float yawDegrees, float pitchDegrees) noexcept {
        yaw_   = yawDegrees;
        pitch_ = std::clamp(pitchDegrees, -89.0f, 89.0f);
    }
    float YawDegrees() const noexcept { return yaw_; }
    float PitchDegrees() const noexcept { return pitch_; }

    void LookAt(const glm::dvec3& target) noexcept {
        glm::dvec3 d = target - position_;
        const double len = glm::length(d);
        if (len < 1e-9) return;
        d /= len;
        SetYawPitch(static_cast<float>(glm::degrees(std::atan2(d.x, -d.z))),
                    static_cast<float>(glm::degrees(std::asin(std::clamp(d.y, -1.0, 1.0)))));
    }

    glm::vec3 Forward() const noexcept {
        const float yaw = glm::radians(yaw_), pitch = glm::radians(pitch_);
        return {std::sin(yaw) * std::cos(pitch), std::sin(pitch), -std::cos(yaw) * std::cos(pitch)};
    }
    glm::vec3 Right() const noexcept { return glm::normalize(glm::cross(Forward(), glm::vec3(0.0f, 1.0f, 0.0f))); }
    glm::vec3 Up() const noexcept { return glm::cross(Right(), Forward()); }

    void ApplyInput(const CameraInput& input, float dt) noexcept {
        SetYawPitch(yaw_ + input.lookDelta.x * lookSensitivity, pitch_ - input.lookDelta.y * lookSensitivity);
        if (input.speedSteps != 0.0f)
            moveSpeed = std::clamp(moveSpeed * std::pow(1.2f, input.speedSteps), 0.5f, 5000.0f);

        float speed = moveSpeed;
        if (input.fast) speed *= fastMultiplier;
        if (input.slow) speed *= slowMultiplier;

        const glm::vec3 velocity = Right() * input.move.x + glm::vec3(0.0f, 1.0f, 0.0f) * input.move.y +
                                   Forward() * input.move.z;
        if (glm::dot(velocity, velocity) > 0.0f)
            position_ += glm::dvec3(glm::normalize(velocity) * speed * dt);
    }

    /// Matrice de vue relative à l'origine de rendu (positions GPU en float précis).
    glm::mat4 ViewMatrix(const glm::dvec3& renderOriginWorld) const noexcept {
        const glm::vec3 eye = glm::vec3(position_ - renderOriginWorld);
        return glm::lookAt(eye, eye + Forward(), glm::vec3(0.0f, 1.0f, 0.0f));
    }

    glm::mat4 ProjectionMatrix(float aspect) const noexcept {
        return InfiniteReverseZ(glm::radians(fovYDegrees), aspect, nearPlane);
    }

    /// Projection perspective Reverse-Z, plan lointain infini, profondeur clip dans [0, 1]
    /// (nécessite glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE)). Profondeur 1 = near, 0 = infini.
    static glm::mat4 InfiniteReverseZ(float fovYRadians, float aspect, float zNear) noexcept {
        const float f = 1.0f / std::tan(fovYRadians * 0.5f);
        glm::mat4 p(0.0f);
        p[0][0] = f / std::max(aspect, 1e-6f);
        p[1][1] = f;
        p[2][3] = -1.0f;  // w_clip = -z_view
        p[3][2] = zNear;  // z_clip = near  =>  z_ndc = near / -z_view
        return p;
    }

    void OnGui() {
        double pos[3] = {position_.x, position_.y, position_.z};
        if (ImGui::DragScalarN("Position (m)", ImGuiDataType_Double, pos, 3, 1.0f)) position_ = {pos[0], pos[1], pos[2]};
        float yawPitch[2] = {yaw_, pitch_};
        if (ImGui::DragFloat2("Yaw / Pitch", yawPitch, 0.25f)) SetYawPitch(yawPitch[0], yawPitch[1]);
        ImGui::SliderFloat("Vitesse (m/s)", &moveSpeed, 0.5f, 5000.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Multiplicateur rapide", &fastMultiplier, 1.0f, 50.0f);
        ImGui::SliderFloat("FOV vertical", &fovYDegrees, 20.0f, 110.0f, "%.1f deg");
        ImGui::SliderFloat("Near", &nearPlane, 0.01f, 10.0f, "%.3f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Distance de rendu", &renderDistance, 100.0f, 50000.0f, "%.0f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Sensibilite", &lookSensitivity, 0.01f, 1.0f);
        ImGui::Separator();
        ImGui::TextDisabled("Clic droit maintenu dans le Viewport : regarder");
        ImGui::TextDisabled("WASD / ZQSD : deplacement, E/Espace : monter, Q/A/Ctrl : descendre");
        ImGui::TextDisabled("Shift : rapide, Alt : lent, molette : vitesse");
    }

private:
    glm::dvec3 position_{128.0, 95.0, 360.0};
    float      yaw_   = 0.0f;
    float      pitch_ = -15.0f;
};

}  // namespace env::core
