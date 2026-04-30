/**
 * @file camera_input_tracker.cpp
 * @brief Camera input-tracking implementation for Vulkan window backends.
 *
 * Converts backend-specific keyboard/mouse queries into a shared camera input
 * snapshot, including right-mouse look deltas and reset edge detection, so
 * free-fly and orbit camera rigs behave consistently across platforms.
 */

#include "camera/camera_input_tracker.hpp"

#include <algorithm>

#if VKCPP_USE_GLFW
#include <GLFW/glfw3.h>
#else
#include <SFML/Window.hpp>
#endif

namespace vkcpp::camera 
{

void CameraInputTracker::reset()
{
    input_ = {};
    reset_pose_latched_ = false;
    capture_latched_ = false;
    cursor_position_initialized_ = false;
    look_active_last_frame_ = false;
    last_cursor_x_ = 0.0f;
    last_cursor_y_ = 0.0f;
}

void CameraInputTracker::begin_frame()
{
    input_.look_delta_x = 0.0f;
    input_.look_delta_y = 0.0f;
    input_.orbit_drag_delta_x = 0.0f;
    input_.orbit_drag_delta_y = 0.0f;
    input_.scroll_delta = accumulated_scroll_;
    accumulated_scroll_ = 0.0f;
    input_.reset_pose = false;
    input_.capture_requested = false;
}

void CameraInputTracker::accumulate_scroll(float y_offset)
{
    accumulated_scroll_ += y_offset;
}

void CameraInputTracker::set_delta_seconds(const float delta_seconds) 
{
    input_.delta_seconds = std::clamp(delta_seconds, 0.0f, 0.12f);
}

const CameraInputState& CameraInputTracker::snapshot() const 
{
    return input_;
}

#if VKCPP_USE_GLFW
void CameraInputTracker::update_from_glfw(GLFWwindow* window) 
{
    if (window == nullptr) {return;}

    const auto key_pressed = [&](int key) 
    {
        return glfwGetKey(window, key) == GLFW_PRESS;
    };

    input_.forward = key_pressed(GLFW_KEY_W);
    input_.backward = key_pressed(GLFW_KEY_S);
    input_.strafe_left = key_pressed(GLFW_KEY_A);
    input_.strafe_right = key_pressed(GLFW_KEY_D);
    input_.move_up = key_pressed(GLFW_KEY_E);
    input_.move_down = key_pressed(GLFW_KEY_Q);
    input_.yaw_left = key_pressed(GLFW_KEY_LEFT);
    input_.yaw_right = key_pressed(GLFW_KEY_RIGHT);
    input_.pitch_up = key_pressed(GLFW_KEY_UP);
    input_.pitch_down = key_pressed(GLFW_KEY_DOWN);
    input_.speed_boost = key_pressed(GLFW_KEY_LEFT_SHIFT) || key_pressed(GLFW_KEY_RIGHT_SHIFT);

    const bool reset_down = key_pressed(GLFW_KEY_R);
    input_.reset_pose = reset_down && !reset_pose_latched_;
    reset_pose_latched_ = reset_down;

    const bool capture_down = key_pressed(GLFW_KEY_F12);
    input_.capture_requested = capture_down && !capture_latched_;
    capture_latched_ = capture_down;

    const bool look_active = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    input_.look_active = look_active;

    const bool orbit_drag = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    input_.orbit_drag_active = orbit_drag;

    double cursor_x = 0.0;
    double cursor_y = 0.0;
    glfwGetCursorPos(window, &cursor_x, &cursor_y);

    if (cursor_position_initialized_)
    {
        const float dx = static_cast<float>(cursor_x - static_cast<double>(last_cursor_x_));
        const float dy = static_cast<float>(cursor_y - static_cast<double>(last_cursor_y_));

        if (look_active && look_active_last_frame_)
        {
            input_.look_delta_x = dx;
            input_.look_delta_y = dy;
        }
        if (orbit_drag && orbit_drag_last_frame_)
        {
            input_.orbit_drag_delta_x = dx;
            input_.orbit_drag_delta_y = dy;
        }
    }

    last_cursor_x_ = static_cast<float>(cursor_x);
    last_cursor_y_ = static_cast<float>(cursor_y);
    cursor_position_initialized_ = true;
    look_active_last_frame_ = look_active;
    orbit_drag_last_frame_ = orbit_drag;
}
#else
void CameraInputTracker::update_from_sfml(sf::Window& window) 
{
    input_.forward = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W);
    input_.backward = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S);
    input_.strafe_left = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A);
    input_.strafe_right = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D);
    input_.move_up = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::E);
    input_.move_down = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Q);
    input_.yaw_left = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Left);
    input_.yaw_right = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Right);
    input_.pitch_up = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Up);
    input_.pitch_down = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Down);
    input_.speed_boost = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LShift) || sf::Keyboard::isKeyPressed(sf::Keyboard::Key::RShift);

    const bool reset_down = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::R);
    input_.reset_pose = reset_down && !reset_pose_latched_;
    reset_pose_latched_ = reset_down;

    const bool capture_down = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::F12);
    input_.capture_requested = capture_down && !capture_latched_;
    capture_latched_ = capture_down;

    const bool look_active = sf::Mouse::isButtonPressed(sf::Mouse::Button::Right);
    input_.look_active = look_active;

    const sf::Vector2i cursor = sf::Mouse::getPosition(window);
    if (cursor_position_initialized_ && look_active && look_active_last_frame_) 
    {
        input_.look_delta_x = static_cast<float>(cursor.x) - last_cursor_x_;
        input_.look_delta_y = static_cast<float>(cursor.y) - last_cursor_y_;
    }

    last_cursor_x_ = static_cast<float>(cursor.x);
    last_cursor_y_ = static_cast<float>(cursor.y);
    cursor_position_initialized_ = true;
    look_active_last_frame_ = look_active;
}
#endif

}  // namespace vkcpp::camera
