/**
 * @file clear_backend.hpp
 * @brief Minimal render backend that clears the swapchain target each frame.
 *
 * Provides a lightweight smoke-test path for swapchain health without shader
 * complexity. Useful for validating surface/presentation synchronization before
 * enabling the heavier volume-rendering backend.
 */

#pragma once

#include "rendering/render_backend.hpp"

namespace vkcpp 
{

/**
 * @brief Simple backend that records a color-clear render pass.
 */
class ClearBackend : public RenderBackend 
{
public:
    /**
     * @brief Construct with configurable clear color.
     * @param clear_color RGBA clear value.
     */
    explicit ClearBackend(VkClearColorValue clear_color = {{0.05f, 0.08f, 0.12f, 1.0f}}): clear_color_(clear_color) {}

    /** @copydoc RenderBackend::initialize */
    bool initialize(VulkanContext& context, VkRenderPass render_pass, VkExtent2D extent, std::string& error) override;

    /** @copydoc RenderBackend::on_swapchain_recreated */
    bool on_swapchain_recreated(VkRenderPass render_pass, VkExtent2D extent, std::string& error) override;

    /** @copydoc RenderBackend::record */
    bool record(VkCommandBuffer command_buffer, VkFramebuffer framebuffer, VkExtent2D extent, std::string& error) override;

    /** @copydoc RenderBackend::shutdown */
    void shutdown(VkDevice device) override;

private:
    VkRenderPass render_pass_ = VK_NULL_HANDLE;
    VkExtent2D extent_{};
    VkClearColorValue clear_color_{};
};

}  // namespace vkcpp
