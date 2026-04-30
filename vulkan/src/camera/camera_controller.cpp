/**
 * @file camera_controller.cpp
 * @brief Camera rig implementation for Vulkan volume rendering.
 *
 * Implements orbit/fixed/free-fly camera integration, including bounded
 * free-fly movement, yaw/pitch constraints, and robust basis construction
 * used to keep storm views stable and physically coherent during motion.
 */

#include "camera/camera_controller.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace vkcpp::camera 
{
namespace 
{

constexpr float kTwoPi = 6.28318530718f;

Vec3 operator+(const Vec3& a, const Vec3& b) {return {a.x + b.x, a.y + b.y, a.z + b.z};}

Vec3 operator-(const Vec3& a, const Vec3& b) {return {a.x - b.x, a.y - b.y, a.z - b.z};}

Vec3 operator*(const Vec3& v, float scalar) {return {v.x * scalar, v.y * scalar, v.z * scalar};}

float length(const Vec3& v) {return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);}

Vec3 normalize(const Vec3& v) 
{
    const float len = length(v);
    if (len < 1e-6f) 
    {
        return {0.0f, 0.0f, 0.0f};
    }
    return {v.x / len, v.y / len, v.z / len};
}

Vec3 cross(const Vec3& a, const Vec3& b) 
{
    return 
    {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

Vec3 forward_from_angles(float yaw_rad, float pitch_rad) 
{
    const float cos_pitch = std::cos(pitch_rad);
    return normalize({
        cos_pitch * std::cos(yaw_rad),
        std::sin(pitch_rad),
        cos_pitch * std::sin(yaw_rad),
    });
}

void build_basis(const Vec3& position, const Vec3& forward, Pose& out_pose) 
{
    const Vec3 world_up{0.0f, 1.0f, 0.0f};
    Vec3 right = normalize(cross(forward, world_up));

    if (length(right) < 1e-6f) 
    {
        right = {1.0f, 0.0f, 0.0f};
    }
    const Vec3 up = normalize(cross(right, forward));

    out_pose.position = position;
    out_pose.forward = forward;
    out_pose.right = right;
    out_pose.up = up;
}

void reset_freefly_pose_impl(bool& pose_initialized,
                             std::array<float, 3>& camera_position,
                             float& camera_yaw_rad,
                             float& camera_pitch_rad,
                             double orbit_angle_rad,
                             float camera_distance,
                             float camera_height) {
    const float start_radius = std::max(camera_distance, 0.95f);
    camera_position = 
    {
        static_cast<float>(std::cos(orbit_angle_rad) * static_cast<double>(start_radius)),
        camera_height,
        static_cast<float>(std::sin(orbit_angle_rad) * static_cast<double>(start_radius)),
    };

    const Vec3 target{0.0f, 0.05f, 0.0f};
    const Vec3 from{camera_position[0], camera_position[1], camera_position[2]};
    const Vec3 forward = normalize(target - from);
    camera_yaw_rad = std::atan2(forward.z, forward.x);
    camera_pitch_rad = std::asin(std::clamp(forward.y, -1.0f, 1.0f));
    pose_initialized = true;
}

}  // namespace

bool normalize_mode(const std::string& input, std::string& normalized_mode) 
{
    std::string mode = input;
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) 
    {
        return static_cast<char>(std::tolower(c));
    });

    if (mode.empty() || mode == "orbit" || mode == "turntable" || mode == "spin") 
    {
        normalized_mode = "orbit";
        return true;
    }
    if (mode == "fixed" || mode == "static") 
    {
        normalized_mode = "fixed";
        return true;
    }
    if (mode == "freefly" || mode == "free" || mode == "fly" || mode == "firstperson" || mode == "first-person") 
    {
        normalized_mode = "freefly";
        return true;
    }

    return false;
}

float compute_delta_seconds(bool& clock_initialized,
                            std::chrono::steady_clock::time_point& last_update_time,
                            const std::chrono::steady_clock::time_point now,
                            const float input_delta_seconds) {
    if (!clock_initialized) 
    {
        clock_initialized = true;
        last_update_time = now;
    }

    const std::chrono::duration<double> elapsed = now - last_update_time;
    last_update_time = now;
    const float fallback_dt = static_cast<float>(std::max(0.0, elapsed.count()));
    const float dt = input_delta_seconds > 0.0f ? input_delta_seconds : fallback_dt;
    return std::clamp(dt, 0.0f, 0.10f);
}

void advance_orbit(double& orbit_angle_rad, const float orbit_fps, const float delta_seconds) 
{
    if (orbit_fps <= 0.0f || delta_seconds <= 0.0f) {return;}

    orbit_angle_rad += static_cast<double>(delta_seconds) * static_cast<double>(kTwoPi) *
                       static_cast<double>(orbit_fps);
    orbit_angle_rad = std::fmod(orbit_angle_rad, static_cast<double>(kTwoPi));

    if (orbit_angle_rad < 0.0) { orbit_angle_rad += static_cast<double>(kTwoPi);}
}

void update_freefly_pose(bool& pose_initialized,
                         std::array<float, 3>& camera_position,
                         float& camera_yaw_rad,
                         float& camera_pitch_rad,
                         const double orbit_angle_rad,
                         const float camera_distance,
                         const float camera_height,
                         const CameraInputState& input,
                         const float delta_seconds) 
{
    if (!pose_initialized || input.reset_pose) 
    {
        reset_freefly_pose_impl(
            pose_initialized,
            camera_position,
            camera_yaw_rad,
            camera_pitch_rad,
            orbit_angle_rad,
            camera_distance,
            camera_height);
    }

    if (delta_seconds <= 0.0f) {return;}

    const float keyboard_yaw = (input.yaw_right ? 1.0f : 0.0f) - (input.yaw_left ? 1.0f : 0.0f);
    const float keyboard_pitch = (input.pitch_up ? 1.0f : 0.0f) - (input.pitch_down ? 1.0f : 0.0f);

    camera_yaw_rad += keyboard_yaw * 1.55f * delta_seconds;
    camera_pitch_rad += keyboard_pitch * 1.25f * delta_seconds;

    if (input.look_active) 
    {
        // Mouse deltas are pixel-space; keep sensitivity independent of frame delta.
        camera_yaw_rad += input.look_delta_x * 0.0030f;
        camera_pitch_rad -= input.look_delta_y * 0.0020f;
    }

    camera_pitch_rad = std::clamp(camera_pitch_rad, -1.15f, 1.15f);
    camera_yaw_rad = std::fmod(camera_yaw_rad, kTwoPi);

    const Vec3 forward = forward_from_angles(camera_yaw_rad, camera_pitch_rad);
    Vec3 right = normalize(cross(forward, {0.0f, 1.0f, 0.0f}));
    if (length(right) < 1e-6f) {
        right = {1.0f, 0.0f, 0.0f};
    }

    Vec3 move{};
    if (input.forward) {move = move + forward;}
    if (input.backward) {move = move - forward;}
    if (input.strafe_right) {move = move + right;}
    if (input.strafe_left) {move = move - right;}
    if (input.move_up) {move = move + Vec3{0.0f, 1.0f, 0.0f};}
    if (input.move_down) {move = move - Vec3{0.0f, 1.0f, 0.0f};}

    const float move_len = length(move);
    if (move_len > 1e-6f) 
    {
        const float speed_scale = input.speed_boost ? 2.2f : 1.0f;
        const float move_speed = 1.60f * speed_scale;
        const Vec3 step = normalize(move) * (move_speed * delta_seconds);
        const Vec3 current{camera_position[0], camera_position[1], camera_position[2]};
        const Vec3 next = current + step;
        camera_position = {next.x, next.y, next.z};
    }

    camera_position[1] = std::clamp(camera_position[1], -0.25f, 2.40f);
    const float radius = std::sqrt(camera_position[0] * camera_position[0] + camera_position[2] * camera_position[2]);

    if (radius < 0.85f || radius > 6.50f) 
    {
        const float clamped_radius = std::clamp(radius, 0.85f, 6.50f);
        const float scale = (radius > 1e-5f) ? (clamped_radius / radius) : 1.0f;
        camera_position[0] *= scale;
        camera_position[2] *= scale;
    }
}

Pose build_pose(const std::string& mode,
                const double orbit_angle_rad,
                const float camera_distance,
                const float camera_height,
                const std::array<float, 3>& freefly_position,
                const float freefly_yaw_rad,
                const float freefly_pitch_rad) 
{
    Pose pose{};

    if (mode == "freefly") 
    {
        const Vec3 position{freefly_position[0], freefly_position[1], freefly_position[2]};
        const Vec3 forward = forward_from_angles(freefly_yaw_rad, freefly_pitch_rad);
        build_basis(position, forward, pose);
        return pose;
    }

    const float distance = std::max(camera_distance, 0.80f);
    const double azimuth_rad = (mode == "orbit" || mode == "fixed") ? orbit_angle_rad : 0.78;
    const Vec3 position{
        static_cast<float>(std::cos(azimuth_rad) * static_cast<double>(distance)),
        camera_height,
        static_cast<float>(std::sin(azimuth_rad) * static_cast<double>(distance)),
    };

    // Slight target lift keeps inflow near ground and the updraft core visible together.
    const Vec3 target{0.0f, 0.05f, 0.0f};
    const Vec3 forward = normalize(target - position);
    build_basis(position, forward, pose);
    return pose;
}

}  // namespace vkcpp::camera
