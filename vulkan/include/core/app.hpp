/**
 * @file app.hpp
 * @brief Top-level Vulkan application options and bootstrap entrypoints.
 *
 * Defines CLI-derived runtime settings and the coordinator that initializes
 * Vulkan context, prints runtime diagnostics, and optionally starts the
 * windowed swapchain test loop with the selected render backend.
 */

#pragma once

#include "core/context.hpp"

#include <array>
#include <string>

namespace vkcpp 
{

/**
 * @brief Parsed runtime configuration for the Vulkan viewer executable.
 */
struct Options 
{
    bool dry_run = false; 
    bool validate_input = false;
    bool enable_validation = false;
    bool list_devices = false;
    int preferred_device_index = -1;
    bool window_test = false;
    uint32_t window_width = 1280;
    uint32_t window_height = 720;
    // 0 means unlimited (run until the user closes the window).
    int window_frames = 0;
    // 0 means unlimited. A positive value exits cleanly after wall-clock time.
    float window_seconds = 0.0f;
    // Non-negative schedules one deterministic resize before that frame.
    int window_resize_frame = -1;
    uint32_t window_resize_width = 960;
    uint32_t window_resize_height = 540;
    float viewer_fps = 30.0f;
    std::string present_mode = "fifo";
    std::string render_backend = "clear"; 
    std::string input_dir = "data/exports";
    std::string field = "theta";
    std::string fields_csv;
    std::string volume_mode = "supercell";
    std::string isolate_field;
    float component_cycle_fps = 0.50f;
    std::string style = "default";
    // Scientific default: render only simulated structure. "natural" remains
    // an explicit aesthetic presentation mode.
    std::string texture_mode = "smooth";
    std::string camera_mode = "freefly";
    float camera_orbit_fps = 0.02f;
    float camera_distance = 2.25f;
    float camera_height = 0.85f;
    float camera_fov_deg = 55.0f;
    float playback_fps = 1.0f;
    int ray_steps = 192;
    float ray_threshold = 0.30f;
    float ray_opacity = 1.15f;
    float ray_brightness = 1.10f;
    float ray_ambient = 0.90f;
    float ray_anisotropy = 0.58f;
    float ray_max_distance = 5.0f;
    std::array<float, 3> voxel_spacing_m = {0.0f, 0.0f, 0.0f};
    float vertical_exaggeration = 1.0f;
    float sun_dir_x = 0.66f;
    float sun_dir_y = 0.34f;
    float sun_dir_z = 0.67f;
    float capture_interval = 0.0f;
    // 0 means unlimited interval captures; positive values cap the session.
    int capture_limit = 0;
    std::string capture_dir = "captures";
    // Diverging-colormap poles for signed (bipolar) fields (w, vorticity, ...):
    // negative pole = downdraft (blue), positive pole = updraft (red). r,g,b in [0,1].
    std::array<float, 3> diverge_neg_color = {0.16f, 0.40f, 0.92f};
    std::array<float, 3> diverge_pos_color = {0.92f, 0.24f, 0.16f};
};

/**
 * @brief High-level orchestrator for Vulkan startup and runtime execution.
 */
class VulkanBootstrap 
{
public:
    /**
     * @brief Execute initialization and optional rendering based on `options`.
     * @return Process-style status code (`0` success, non-zero failure).
     */
    int run(const Options& options);

private:
    /**
     * @brief Print resolved runtime configuration and selected GPU summary.
     */
    void print_runtime_summary(const Options& options) const;

    VulkanContext context_;
};

}  // namespace vkcpp
