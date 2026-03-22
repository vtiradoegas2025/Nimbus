/**
 * @file camera_controller.hpp
 * @brief Camera rig math and motion integration for Vulkan volume rendering.
 *
 * Centralizes camera-mode normalization, per-frame timestep handling,
 * constrained free-fly updates, and view-basis construction for orbit,
 * fixed, and interactive free-fly rigs.
 */

#pragma once

#include "rendering/render_backend.hpp"

#include <array>
#include <chrono>
#include <string>

namespace vkcpp::camera 
{

/** @brief Lightweight 3D vector used for camera basis outputs. */
struct Vec3 
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

/** @brief Derived camera pose consumed by push-constant packing. */
struct Pose 
{
    Vec3 position{};
    Vec3 forward{};
    Vec3 right{};
    Vec3 up{};
};

/** @brief Normalize and validate camera mode aliases. */
bool normalize_mode(const std::string& input, std::string& normalized_mode);

/** @brief Compute stable per-frame camera timestep with a defensive clamp. */
float compute_delta_seconds(bool& clock_initialized,
                            std::chrono::steady_clock::time_point& last_update_time,
                            std::chrono::steady_clock::time_point now,
                            float input_delta_seconds);

/** @brief Advance orbit azimuth based on configured revolutions-per-second. */
void advance_orbit(double& orbit_angle_rad, float orbit_fps, float delta_seconds);

/** @brief Update constrained free-fly pose/orientation from latest input state. */
void update_freefly_pose(bool& pose_initialized,
                         std::array<float, 3>& camera_position,
                         float& camera_yaw_rad,
                         float& camera_pitch_rad,
                         double orbit_angle_rad,
                         float camera_distance,
                         float camera_height,
                         const CameraInputState& input,
                         float delta_seconds);

/** @brief Build camera pose for current mode from runtime camera state. */
Pose build_pose(const std::string& mode,
                double orbit_angle_rad,
                float camera_distance,
                float camera_height,
                const std::array<float, 3>& freefly_position,
                float freefly_yaw_rad,
                float freefly_pitch_rad);

}  // namespace vkcpp::camera
