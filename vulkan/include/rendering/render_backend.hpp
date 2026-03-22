/**
 * @file render_backend.hpp
 * @brief Abstract rendering backend contract for swapchain command recording.
 *
 * Backends implement initialization, swapchain-resize handling, per-frame update,
 * and command-buffer recording while `VulkanSwapchainRunner` manages window and
 * synchronization orchestration. This keeps rendering techniques pluggable.
 */

#pragma once

#include <cstdint>
#include <string>

#include <vulkan/vulkan.h>

namespace vkcpp 
{

class VulkanContext;

/**
 * @brief Per-frame camera/navigation input snapshot from the window layer.
 */
struct CameraInputState {
    float delta_seconds = 0.0f;
    bool forward = false;
    bool backward = false;
    bool strafe_left = false;
    bool strafe_right = false;
    bool move_up = false;
    bool move_down = false;
    bool yaw_left = false;
    bool yaw_right = false;
    bool pitch_up = false;
    bool pitch_down = false;
    bool speed_boost = false;
    bool reset_pose = false;
    bool look_active = false;
    float look_delta_x = 0.0f;
    float look_delta_y = 0.0f;
};

/**
 * @brief Interface for render backends used by the swapchain runner.
 */
class RenderBackend 
{
public:
    virtual ~RenderBackend() = default;

    /**
     * @brief Initialize backend resources for current render pass and extent.
     * @return `true` on success.
     */
    virtual bool initialize(VulkanContext& context,
                            VkRenderPass render_pass,
                            VkExtent2D extent,
                            std::string& error) = 0;

    /**
     * @brief Rebuild swapchain-dependent resources after resize/recreation.
     * @return `true` on success.
     */
    virtual bool on_swapchain_recreated(VkRenderPass render_pass,
                                        VkExtent2D extent,
                                        std::string& error) = 0;

    /**
     * @brief Record draw commands into the provided command buffer.
     * @return `true` on success.
     */
    virtual bool record(VkCommandBuffer command_buffer,
                        VkFramebuffer framebuffer,
                        VkExtent2D extent,
                        std::string& error) = 0;

    /**
     * @brief Optional per-frame update hook before command submission.
     * @return `true` on success.
     */
    virtual bool update_for_frame(VulkanContext&, std::uint64_t, std::string&) {return true;}

    /** @brief Optional camera/navigation input hook for interactive backends. */
    virtual void set_camera_input(const CameraInputState&) {}

    /** @brief Destroy backend-owned Vulkan resources. */
    virtual void shutdown(VkDevice device) = 0;
};

}  // namespace vkcpp
