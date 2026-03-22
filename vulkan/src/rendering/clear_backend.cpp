/**
 * @file clear_backend.cpp
 * @brief Clear-pass render backend used for swapchain smoke testing.
 *
 * Records a minimal render pass that only clears the target framebuffer.
 * This backend helps isolate platform/swapchain issues from shader or data
 * complexity when validating Vulkan context and presentation plumbing.
 */

#include "rendering/clear_backend.hpp"

#include "core/context.hpp"

namespace vkcpp 
{

/** @brief Cache render-pass and extent state for the clear backend. */
bool ClearBackend::initialize(VulkanContext&, VkRenderPass render_pass, VkExtent2D extent, std::string&) 
{
    render_pass_ = render_pass;
    extent_ = extent;
    return true;
}

/** @brief Refresh render-pass and extent after swapchain recreation. */
bool ClearBackend::on_swapchain_recreated(VkRenderPass render_pass,
                                          VkExtent2D extent,
                                          std::string&) 
{
    render_pass_ = render_pass;
    extent_ = extent;
    return true;
}

/** @brief Record one clear-only render pass command buffer. */
bool ClearBackend::record(VkCommandBuffer command_buffer,
                          VkFramebuffer framebuffer,
                          VkExtent2D extent,
                          std::string& error) 
{
    if (render_pass_ == VK_NULL_HANDLE) 
    {
        error = "clear backend called without a valid render pass";
        return false;
    }

    if (extent.width == 0 || extent.height == 0) {extent = extent_;}

    if (framebuffer == VK_NULL_HANDLE) 
    {
        error = "clear backend received null framebuffer";
        return false;
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    const VkResult begin_result = vkBeginCommandBuffer(command_buffer, &begin_info);
    if (begin_result != VK_SUCCESS) 
    {
        error = "vkBeginCommandBuffer failed with VkResult=" + std::to_string(static_cast<int>(begin_result));
        return false;
    }

    VkClearValue clear_color{};
    clear_color.color = clear_color_;

    VkRenderPassBeginInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = render_pass_;
    render_pass_info.framebuffer = framebuffer;
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = extent;
    render_pass_info.clearValueCount = 1;
    render_pass_info.pClearValues = &clear_color;

    vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdEndRenderPass(command_buffer);

    const VkResult end_result = vkEndCommandBuffer(command_buffer);
    if (end_result != VK_SUCCESS) 
    {
        error = "vkEndCommandBuffer failed with VkResult=" + std::to_string(static_cast<int>(end_result));
        return false;
    }

    return true;
}

/** @brief Reset cached render state. */
void ClearBackend::shutdown(VkDevice) 
{
    render_pass_ = VK_NULL_HANDLE;
    extent_ = {};
}

}  // namespace vkcpp
