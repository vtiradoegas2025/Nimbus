/**
 * @file camera_input_tracker.hpp
 * @brief Window-input to camera-control state translation helpers.
 *
 * Encapsulates key/mouse state sampling for GLFW/SFML windows and maintains
 * short-lived input latches (look deltas, reset edge detection) so the
 * renderer receives consistent per-frame camera control snapshots.
 */

#pragma once

#include "rendering/render_backend.hpp"

#if VKCPP_USE_GLFW
#include <GLFW/glfw3.h>
#else
#include <SFML/Window.hpp>
#endif

namespace vkcpp::camera 
{

/** @brief Tracks and updates per-frame camera input state from window APIs. */
class CameraInputTracker 
{
public:
    /** @brief Reset all latched state and input snapshot values. */
    void reset();

    /** @brief Start a new frame and clear one-frame deltas/edges. */
    void begin_frame();

    /** @brief Update elapsed frame time used by camera integrators. */
    void set_delta_seconds(float delta_seconds);

    /** @brief Return immutable snapshot consumed by render backends. */
    [[nodiscard]] const CameraInputState& snapshot() const;

#if VKCPP_USE_GLFW
    /** @brief Sample camera controls from GLFW key/mouse state. */
    void update_from_glfw(GLFWwindow* window);
#else
    /** @brief Sample camera controls from SFML key/mouse state. */
    void update_from_sfml(sf::Window& window);
#endif

    /** @brief Accumulate scroll input from GLFW callback (called from static trampoline). */
    void accumulate_scroll(float y_offset);

private:
    CameraInputState input_{};
    bool reset_pose_latched_ = false;
    bool capture_latched_ = false;
    bool cursor_position_initialized_ = false;
    bool look_active_last_frame_ = false;
    bool orbit_drag_last_frame_ = false;
    float last_cursor_x_ = 0.0f;
    float last_cursor_y_ = 0.0f;
    float accumulated_scroll_ = 0.0f;
};

}  // namespace vkcpp::camera
