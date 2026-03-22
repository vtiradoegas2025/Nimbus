/**
 * @file window_renderer.cpp
 * @brief Windowed swapchain lifecycle and render-loop execution.
 *
 * Bridges platform windowing with Vulkan presentation primitives, including
 * swapchain creation, command recording, synchronization, and resize recovery.
 * Delegates actual draw commands to the selected render backend implementation.
 */

#include "rendering/window_renderer.hpp"

#include "rendering/clear_backend.hpp"
#include "camera/camera_input_tracker.hpp"
#include "core/context.hpp"
#include "rendering/volume_backend.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <optional>
#include <string>

namespace vkcpp 
{

#if VKCPP_USE_GLFW
/** @brief GLFW callback that marks swapchain resources dirty after resize. */
void VulkanSwapchainRunner::framebuffer_resize_callback(GLFWwindow* window, int, int) 
{
    if (window == nullptr) {return;}

    auto* self = static_cast<VulkanSwapchainRunner*>(glfwGetWindowUserPointer(window));

    if (self != nullptr) {self->framebuffer_resized_ = true;}
}
#endif

/** @brief RAII cleanup entrypoint for swapchain runner resources. */
VulkanSwapchainRunner::~VulkanSwapchainRunner() {shutdown();}

/** @brief Build complete swapchain rendering state from an initialized context. */
bool VulkanSwapchainRunner::initialize(VulkanContext& context, const WindowRunConfig& config, std::string& error) 
{
    shutdown();

    context_ = &context;
    config_ = config;
    delete camera_input_tracker_;
    camera_input_tracker_ = new camera::CameraInputTracker();

    if (!create_window(error)) {shutdown(); return false;}

    if (!create_surface(error)) {shutdown(); return false;}

    if (!ensure_present_support(error)) {shutdown(); return false;}

    if (!create_swapchain(error)) {shutdown(); return false;}

    if (!create_image_views(error)) {shutdown(); return false;}

    if (!create_render_pass(error)) {shutdown(); return false;}

    if (!create_render_backend(error)) {shutdown(); return false;}

    if (!create_command_pool(error)) {shutdown(); return false;}

    if (!create_framebuffers(error)) {shutdown(); return false;}

    if (!create_command_buffers(error)) {shutdown(); return false;}

    if (!record_command_buffers(error)) {shutdown(); return false;}

    if (!create_sync_objects(error)) {shutdown(); return false;}

    return true;
}

/** @brief Run the frame loop and submit frames until exit conditions are reached. */
bool VulkanSwapchainRunner::run(std::string& error) 
{
    if (context_ == nullptr) 
    {
        error = "swapchain runner not initialized";
        return false;
    }

    int rendered_frames = 0;
    auto last_input_time = std::chrono::steady_clock::now();
    while (is_window_open()) 
    {
        process_events();

        if (camera_input_tracker_ == nullptr) 
        {
            camera_input_tracker_ = new camera::CameraInputTracker();
        }

        const auto now = std::chrono::steady_clock::now();
        const std::chrono::duration<float> elapsed = now - last_input_time;
        last_input_time = now;
        camera_input_tracker_->set_delta_seconds(elapsed.count());

        if (render_backend_ != nullptr) 
        {
            render_backend_->set_camera_input(camera_input_tracker_->snapshot());
        }

        if (!is_window_open()) {break;}

        if (config_.max_frames > 0 && rendered_frames >= config_.max_frames) {break;}

        if (render_backend_ != nullptr && !render_backend_->update_for_frame(*context_, static_cast<std::uint64_t>(rendered_frames), error)) 
        {
            return false;
        }

        if (!draw_frame(error)) {return false; }

        ++rendered_frames;
    }

    vkDeviceWaitIdle(context_->device());
    return true;
}

/** @brief Destroy sync objects, backend state, swapchain objects, and window resources. */
void VulkanSwapchainRunner::shutdown() 
{
    if (context_ != nullptr && context_->device() != VK_NULL_HANDLE) 
    {
        vkDeviceWaitIdle(context_->device());
    }

    if (context_ != nullptr && context_->device() != VK_NULL_HANDLE) 
    {
        for (std::size_t i = 0; i < kMaxFramesInFlight; ++i) 
        {
            if (render_finished_semaphores_[i] != VK_NULL_HANDLE) 
            {
                vkDestroySemaphore(context_->device(), render_finished_semaphores_[i], nullptr);
                render_finished_semaphores_[i] = VK_NULL_HANDLE;
            }

            if (image_available_semaphores_[i] != VK_NULL_HANDLE) 
            {
                vkDestroySemaphore(context_->device(), image_available_semaphores_[i], nullptr);
                image_available_semaphores_[i] = VK_NULL_HANDLE;
            }

            if (in_flight_fences_[i] != VK_NULL_HANDLE) 
            {
                vkDestroyFence(context_->device(), in_flight_fences_[i], nullptr);
                in_flight_fences_[i] = VK_NULL_HANDLE;
            }
        }
    }

    if (render_backend_ != nullptr) 
    {
        if (context_ != nullptr && context_->device() != VK_NULL_HANDLE) 
        {
            render_backend_->shutdown(context_->device());
        }
        render_backend_.reset();
    }

    images_in_flight_.clear();
    cleanup_swapchain();

    if (context_ != nullptr && context_->device() != VK_NULL_HANDLE && command_pool_ != VK_NULL_HANDLE) 
    {
        vkDestroyCommandPool(context_->device(), command_pool_, nullptr);
    }
    command_pool_ = VK_NULL_HANDLE;

    if (context_ != nullptr && context_->instance() != VK_NULL_HANDLE && surface_ != VK_NULL_HANDLE) 
    {
        vkDestroySurfaceKHR(context_->instance(), surface_, nullptr);
    }
    surface_ = VK_NULL_HANDLE;

    close_window();

    context_ = nullptr;
    current_frame_ = 0;
    framebuffer_resized_ = false;
    delete camera_input_tracker_;
    camera_input_tracker_ = nullptr;
}

/** @brief Create the platform window with Vulkan-compatible settings. */
bool VulkanSwapchainRunner::create_window(std::string& error) 
{
#if VKCPP_USE_GLFW
    if (glfwInit() != GLFW_TRUE) 
    {
        error = "glfwInit failed";
        return false;
    }
    glfw_initialized_ = true;

    if (glfwVulkanSupported() != GLFW_TRUE) 
    {
        error = "GLFW reports Vulkan support is unavailable";
        return false;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_TRUE);
#endif

    window_ = glfwCreateWindow(
        static_cast<int>(config_.width),
        static_cast<int>(config_.height),
        "TornadoModel Vulkan Window Test",
        nullptr,
        nullptr);

    if (window_ == nullptr) 
    {
        error = "glfwCreateWindow failed";
        return false;
    }

    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, framebuffer_resize_callback);
    return true;
#else
    window_.create(
        sf::VideoMode({config_.width, config_.height}),
        "TornadoModel Vulkan Window Test",
        sf::Style::Default,
        sf::State::Windowed);

    if (!window_.isOpen()) 
    {
        error = "failed to create SFML window";
        return false;
    }

    window_created_ = true;
    return true;
#endif
}

/** @brief Create a Vulkan presentation surface bound to the active window. */
bool VulkanSwapchainRunner::create_surface(std::string& error) 
{
    if (context_ == nullptr || context_->instance() == VK_NULL_HANDLE) 
    {
        error = "Vulkan instance is not initialized";
        return false;
    }

#if VKCPP_USE_GLFW
    const VkResult result = glfwCreateWindowSurface(context_->instance(), window_, nullptr, &surface_);
    if (result != VK_SUCCESS) 
    {
        error = "glfwCreateWindowSurface failed with VkResult=" + std::to_string(static_cast<int>(result));
        return false;
    }
#else
    if (!window_.createVulkanSurface(context_->instance(), surface_, nullptr)) 
    {
        error = "sf::Window::createVulkanSurface failed";
        return false;
    }
#endif

    return true;
}

/** @brief Ensure selected queue family can present images to the created surface. */
bool VulkanSwapchainRunner::ensure_present_support(std::string& error)
{
    VkBool32 present_supported = VK_FALSE;
    const VkResult result = vkGetPhysicalDeviceSurfaceSupportKHR(
        context_->physical_device(),
        context_->graphics_queue_family_index(),
        surface_,
        &present_supported);

    if (result != VK_SUCCESS) 
    {
        error = "vkGetPhysicalDeviceSurfaceSupportKHR failed with VkResult=" +
                std::to_string(static_cast<int>(result));
        return false;
    }

    if (present_supported == VK_FALSE) 
    {
        error = "selected graphics queue family does not support presenting to this surface";
        return false;
    }

    return true;
}

/** @brief Create swapchain images and cache format/extent metadata. */
bool VulkanSwapchainRunner::create_swapchain(std::string& error) 
{
    SwapchainSupportDetails support;
    if (!query_swapchain_support(context_->physical_device(), support, error)) 
    {
        return false;
    }

    if (support.formats.empty()) 
    {
        error = "no supported surface formats reported for swapchain";
        return false;
    }

    if (support.present_modes.empty()) 
    {
        error = "no supported present modes reported for swapchain";
        return false;
    }

    const VkSurfaceFormatKHR surface_format = choose_surface_format(support.formats);
    const VkPresentModeKHR present_mode = choose_present_mode(support.present_modes);
    const VkExtent2D extent = choose_extent(support.capabilities);

    uint32_t image_count = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 && image_count > support.capabilities.maxImageCount) 
    {
        image_count = support.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = surface_;
    create_info.minImageCount = image_count;
    create_info.imageFormat = surface_format.format;
    create_info.imageColorSpace = surface_format.colorSpace;
    create_info.imageExtent = extent;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.preTransform = support.capabilities.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = present_mode;
    create_info.clipped = VK_TRUE;
    create_info.oldSwapchain = VK_NULL_HANDLE;

    VkResult result = vkCreateSwapchainKHR(context_->device(), &create_info, nullptr, &swapchain_);
    if (result != VK_SUCCESS) 
    {
        error = "vkCreateSwapchainKHR failed with VkResult=" + std::to_string(static_cast<int>(result));
        return false;
    }

    uint32_t actual_image_count = 0;
    result = vkGetSwapchainImagesKHR(context_->device(), swapchain_, &actual_image_count, nullptr);
    if (result != VK_SUCCESS) 
    {
        error = "vkGetSwapchainImagesKHR(count) failed with VkResult=" + std::to_string(static_cast<int>(result));
        return false;
    }
    if (actual_image_count == 0) 
    {
        error = "swapchain returned zero images";
        return false;
    }

    swapchain_images_.resize(actual_image_count);
    result = vkGetSwapchainImagesKHR(context_->device(), swapchain_, &actual_image_count, swapchain_images_.data());
    if (result != VK_SUCCESS) 
    {
        error = "vkGetSwapchainImagesKHR(list) failed with VkResult=" + std::to_string(static_cast<int>(result));
        swapchain_images_.clear();
        return false;
    }
    swapchain_images_.resize(actual_image_count);

    swapchain_image_format_ = surface_format.format;
    swapchain_extent_ = extent;

    return true;
}

/** @brief Create per-image 2D views for swapchain images. */
bool VulkanSwapchainRunner::create_image_views(std::string& error) 
{
    swapchain_image_views_.resize(swapchain_images_.size(), VK_NULL_HANDLE);

    for (std::size_t i = 0; i < swapchain_images_.size(); ++i) 
    {
        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = swapchain_images_[i];
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = swapchain_image_format_;
        view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;

        const VkResult result =
            vkCreateImageView(context_->device(), &view_info, nullptr, &swapchain_image_views_[i]);
        if (result != VK_SUCCESS) 
        {
            error = "vkCreateImageView failed with VkResult=" + std::to_string(static_cast<int>(result));
            return false;
        }
    }

    return true;
}

/** @brief Create a simple color-only render pass for backend rendering. */
bool VulkanSwapchainRunner::create_render_pass(std::string& error) 
{
    VkAttachmentDescription color_attachment{};
    color_attachment.format = swapchain_image_format_;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_attachment_ref{};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = 1;
    render_pass_info.pAttachments = &color_attachment;
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = 1;
    render_pass_info.pDependencies = &dependency;

    const VkResult result = vkCreateRenderPass(context_->device(), &render_pass_info, nullptr, &render_pass_);
    if (result != VK_SUCCESS) 
    {
        error = "vkCreateRenderPass failed with VkResult=" + std::to_string(static_cast<int>(result));
        return false;
    }

    return true;
}

/** @brief Build framebuffers by pairing swapchain image views with render pass. */
bool VulkanSwapchainRunner::create_framebuffers(std::string& error) 
{
    swapchain_framebuffers_.resize(swapchain_image_views_.size(), VK_NULL_HANDLE);

    for (std::size_t i = 0; i < swapchain_image_views_.size(); ++i) 
    {
        VkImageView attachments[] = {swapchain_image_views_[i]};

        VkFramebufferCreateInfo framebuffer_info{};
        framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebuffer_info.renderPass = render_pass_;
        framebuffer_info.attachmentCount = 1;
        framebuffer_info.pAttachments = attachments;
        framebuffer_info.width = swapchain_extent_.width;
        framebuffer_info.height = swapchain_extent_.height;
        framebuffer_info.layers = 1;

        const VkResult result =
            vkCreateFramebuffer(context_->device(), &framebuffer_info, nullptr, &swapchain_framebuffers_[i]);

        if (result != VK_SUCCESS) {
            error = "vkCreateFramebuffer failed with VkResult=" + std::to_string(static_cast<int>(result));
            return false;
        }
    }

    return true;
}

/** @brief Create command pool for primary graphics command buffers. */
bool VulkanSwapchainRunner::create_command_pool(std::string& error) 
{
    if (command_pool_ != VK_NULL_HANDLE) { return true;}

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = context_->graphics_queue_family_index();

    const VkResult result = vkCreateCommandPool(context_->device(), &pool_info, nullptr, &command_pool_);
    if (result != VK_SUCCESS) 
    {
        error = "vkCreateCommandPool failed with VkResult=" + std::to_string(static_cast<int>(result));
        return false;
    }

    return true;
}

/** @brief Allocate one primary command buffer per swapchain framebuffer. */
bool VulkanSwapchainRunner::create_command_buffers(std::string& error) 
{
    if (!command_buffers_.empty())
    {
        vkFreeCommandBuffers(
            context_->device(),
            command_pool_,
            static_cast<uint32_t>(command_buffers_.size()),
            command_buffers_.data());
        command_buffers_.clear();
    }

    command_buffers_.resize(swapchain_framebuffers_.size(), VK_NULL_HANDLE);
    if (command_buffers_.empty()) 
    {
        error = "cannot allocate command buffers for an empty swapchain";
        return false;
    }

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = static_cast<uint32_t>(command_buffers_.size());

    const VkResult result = vkAllocateCommandBuffers(context_->device(), &alloc_info, command_buffers_.data());
    if (result != VK_SUCCESS) 
    {
        error = "vkAllocateCommandBuffers failed with VkResult=" + std::to_string(static_cast<int>(result));
        return false;
    }

    return true;
}

/** @brief Record backend draw commands into every swapchain command buffer. */
bool VulkanSwapchainRunner::record_command_buffers(std::string& error) 
{
    if (render_backend_ == nullptr) 
    {
        error = "render backend not initialized";
        return false;
    }

    for (std::size_t i = 0; i < command_buffers_.size(); ++i) 
    {
        if (!render_backend_->record(
                command_buffers_[i], swapchain_framebuffers_[i], swapchain_extent_, error)) 
                {
            return false;
        }
    }

    return true;
}

/** @brief Instantiate the configured render backend and initialize it. */
bool VulkanSwapchainRunner::create_render_backend(std::string& error) 
{
    if (config_.backend_name == "clear") 
    {
        render_backend_ = std::make_unique<ClearBackend>();
    } else if (config_.backend_name == "volume") 
    {
        VolumeBackend::Config volume_config;
        volume_config.input_dir = config_.input_dir;
        volume_config.field = config_.field;
        volume_config.fields_csv = config_.fields_csv;
        volume_config.volume_mode = config_.volume_mode;
        volume_config.isolate_field = config_.isolate_field;
        volume_config.component_cycle_fps = config_.component_cycle_fps;
        volume_config.cinematic_bw = (config_.style == "cinematic-bw");
        volume_config.texture_mode = config_.texture_mode;
        volume_config.camera_mode = config_.camera_mode;
        volume_config.camera_orbit_fps = config_.camera_orbit_fps;
        volume_config.camera_distance = config_.camera_distance;
        volume_config.camera_height = config_.camera_height;
        volume_config.camera_fov_deg = config_.camera_fov_deg;
        volume_config.playback_fps = config_.playback_fps;
        volume_config.ray_steps = config_.ray_steps;
        volume_config.ray_threshold = config_.ray_threshold;
        volume_config.ray_opacity = config_.ray_opacity;
        volume_config.ray_brightness = config_.ray_brightness;
        volume_config.ray_ambient = config_.ray_ambient;
        volume_config.ray_anisotropy = config_.ray_anisotropy;
        volume_config.ray_max_distance = config_.ray_max_distance;
        volume_config.sun_dir = {config_.sun_dir_x, config_.sun_dir_y, config_.sun_dir_z};
        render_backend_ = std::make_unique<VolumeBackend>(volume_config);
    } else 
    {
        error = "unknown render backend: " + config_.backend_name;
        return false;
    }

    if (!render_backend_->initialize(*context_, render_pass_, swapchain_extent_, error)) 
    {
        render_backend_.reset();
        return false;
    }

    return true;
}

/** @brief Create per-frame semaphores/fences and image ownership tracking. */
bool VulkanSwapchainRunner::create_sync_objects(std::string& error) 
{
    if (swapchain_images_.empty())
    {
        error = "cannot create synchronization objects without swapchain images";
        return false;
    }

    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (std::size_t i = 0; i < kMaxFramesInFlight; ++i) 
    {
        if (vkCreateSemaphore(context_->device(), &semaphore_info, nullptr, &image_available_semaphores_[i]) !=
                VK_SUCCESS ||
            vkCreateSemaphore(context_->device(), &semaphore_info, nullptr, &render_finished_semaphores_[i]) !=
                VK_SUCCESS ||
            vkCreateFence(context_->device(), &fence_info, nullptr, &in_flight_fences_[i]) != VK_SUCCESS) {
            error = "failed to create synchronization objects";
            return false;
        }
    }

    images_in_flight_.assign(swapchain_images_.size(), VK_NULL_HANDLE);
    return true;
}

/** @brief Recreate swapchain-dependent resources after resize/out-of-date signals. */
bool VulkanSwapchainRunner::recreate_swapchain(std::string& error) 
{
    const auto [width, height] = framebuffer_size();
    if (width == 0 || height == 0) {return true;}

    vkDeviceWaitIdle(context_->device());

    cleanup_swapchain();

    if (!create_swapchain(error) || !create_image_views(error) || !create_render_pass(error) ||
        (render_backend_ != nullptr && !render_backend_->on_swapchain_recreated(render_pass_, swapchain_extent_, error)) ||
        !create_framebuffers(error) || !create_command_buffers(error) || !record_command_buffers(error)) 
    {
        return false;
    }

    images_in_flight_.assign(swapchain_images_.size(), VK_NULL_HANDLE);
    return true;
}

/** @brief Submit one command buffer and present the rendered image. */
bool VulkanSwapchainRunner::draw_frame(std::string& error) 
{
    VkResult fence_result = vkWaitForFences(context_->device(), 1, &in_flight_fences_[current_frame_], VK_TRUE, UINT64_MAX);
    if (fence_result != VK_SUCCESS) 
    {
        error = "vkWaitForFences(frame) failed with VkResult=" + std::to_string(static_cast<int>(fence_result));
        return false;
    }

    uint32_t image_index = 0;
    VkResult result = vkAcquireNextImageKHR(
        context_->device(),
        swapchain_,
        UINT64_MAX,
        image_available_semaphores_[current_frame_],
        VK_NULL_HANDLE,
        &image_index);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) 
    {
        return recreate_swapchain(error);
    }

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) 
    {
        error = "vkAcquireNextImageKHR failed with VkResult=" + std::to_string(static_cast<int>(result));
        return false;
    }

    if (image_index >= command_buffers_.size() || image_index >= images_in_flight_.size()) 
    {
        error = "swapchain image index out of range for command/fence tracking";
        return false;
    }

    if (images_in_flight_[image_index] != VK_NULL_HANDLE) 
    {
        fence_result = vkWaitForFences(context_->device(), 1, &images_in_flight_[image_index], VK_TRUE, UINT64_MAX);

        if (fence_result != VK_SUCCESS) 
        {
            error = "vkWaitForFences(image) failed with VkResult=" + std::to_string(static_cast<int>(fence_result));
            return false;
        }
    }

    images_in_flight_[image_index] = in_flight_fences_[current_frame_];

    if (render_backend_ == nullptr)
    {
        error = "render backend not initialized";
        return false;
    }

    // Re-record each frame so push-constant state (camera orbit, isolate/cycle selection) stays live.
    if (!render_backend_->record(command_buffers_[image_index], swapchain_framebuffers_[image_index],
            swapchain_extent_, error)) 
    {
        return false;
    }

    VkSemaphore wait_semaphores[] = {image_available_semaphores_[current_frame_]};
    VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSemaphore signal_semaphores[] = {render_finished_semaphores_[current_frame_]};

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = wait_semaphores;
    submit_info.pWaitDstStageMask = wait_stages;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffers_[image_index];
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = signal_semaphores;

    result = vkResetFences(context_->device(), 1, &in_flight_fences_[current_frame_]);
    if (result != VK_SUCCESS) 
    {
        error = "vkResetFences failed with VkResult=" + std::to_string(static_cast<int>(result));
        return false;
    }

    result = vkQueueSubmit(context_->graphics_queue(), 1, &submit_info, in_flight_fences_[current_frame_]);
    if (result != VK_SUCCESS) 
    {
        error = "vkQueueSubmit failed with VkResult=" + std::to_string(static_cast<int>(result));
        return false;
    }

    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = signal_semaphores;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swapchain_;
    present_info.pImageIndices = &image_index;

    result = vkQueuePresentKHR(context_->graphics_queue(), &present_info);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebuffer_resized_) 
    {
        framebuffer_resized_ = false;
        if (!recreate_swapchain(error)) {return false;}
    } 
    else if (result != VK_SUCCESS) 
    {
        error = "vkQueuePresentKHR failed with VkResult=" + std::to_string(static_cast<int>(result));
        return false;
    }

    current_frame_ = (current_frame_ + 1) % kMaxFramesInFlight;
    return true;
}

/** @brief Pump OS events and track resize/close transitions. */
void VulkanSwapchainRunner::process_events() 
{
    if (camera_input_tracker_ == nullptr) {return;}
    camera_input_tracker_->begin_frame();

#if VKCPP_USE_GLFW
    glfwPollEvents();

    if (window_ == nullptr) {return;}
    camera_input_tracker_->update_from_glfw(window_);
#else
    while (const std::optional<sf::Event> event = window_.pollEvent()) 
    {
        if (event->is<sf::Event::Closed>()) 
        {
            window_.close();
        }

        if (event->is<sf::Event::Resized>()) 
        {
            framebuffer_resized_ = true;
        }
    }
    camera_input_tracker_->update_from_sfml(window_);
#endif
}

/** @brief Return `true` while the user has not closed the window. */
bool VulkanSwapchainRunner::is_window_open() const 
{
#if VKCPP_USE_GLFW
    return window_ != nullptr && glfwWindowShouldClose(window_) == GLFW_FALSE;
#else
    return window_.isOpen();
#endif
}

/** @brief Destroy native window resources and windowing runtime state. */
void VulkanSwapchainRunner::close_window() 
{
#if VKCPP_USE_GLFW
    if (window_ != nullptr) 
    {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }

    if (glfw_initialized_) 
    {
        glfwTerminate();
        glfw_initialized_ = false;
    }
#else
    if (window_created_) 
    {
        window_.close();
        window_created_ = false;
    }
#endif
}

/** @brief Return framebuffer pixel dimensions for swapchain sizing. */
std::pair<uint32_t, uint32_t> VulkanSwapchainRunner::framebuffer_size() const 
{
#if VKCPP_USE_GLFW
    if (window_ == nullptr) { return {0, 0};}

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);

    if (width < 0) { width = 0;}
    if (height < 0) { height = 0;}

    return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
#else
    const sf::Vector2u size = window_.getSize();
    return {size.x, size.y};
#endif
}

/** @brief Query all swapchain support details required for creation decisions. */
bool VulkanSwapchainRunner::query_swapchain_support(VkPhysicalDevice device, SwapchainSupportDetails& out_details, std::string& error) const 
{
    out_details = SwapchainSupportDetails{};

    VkResult result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_, &out_details.capabilities);
    if (result != VK_SUCCESS) 
    {
        error = "vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed with VkResult=" +
                std::to_string(static_cast<int>(result));
        return false;
    }

    uint32_t format_count = 0;
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &format_count, nullptr);
    if (result != VK_SUCCESS) 
    {
        error = "vkGetPhysicalDeviceSurfaceFormatsKHR(count) failed with VkResult=" +
                std::to_string(static_cast<int>(result));
        return false;
    }
    if (format_count > 0) 
    {
        out_details.formats.resize(format_count);
        result = vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &format_count, out_details.formats.data());
        if (result != VK_SUCCESS) 
        {
            error = "vkGetPhysicalDeviceSurfaceFormatsKHR(list) failed with VkResult=" +
                    std::to_string(static_cast<int>(result));
            return false;
        }
        out_details.formats.resize(format_count);
    }

    uint32_t present_mode_count = 0;
    result = vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &present_mode_count, nullptr);
    if (result != VK_SUCCESS) 
    {
        error = "vkGetPhysicalDeviceSurfacePresentModesKHR(count) failed with VkResult=" +
                std::to_string(static_cast<int>(result));
        return false;
    }
    if (present_mode_count > 0) 
    {
        out_details.present_modes.resize(present_mode_count);
        result = vkGetPhysicalDeviceSurfacePresentModesKHR(
            device,
            surface_,
            &present_mode_count,
            out_details.present_modes.data());
        if (result != VK_SUCCESS) 
        {
            error = "vkGetPhysicalDeviceSurfacePresentModesKHR(list) failed with VkResult=" +
                    std::to_string(static_cast<int>(result));
            return false;
        }
        out_details.present_modes.resize(present_mode_count);
    }

    return true;
}

/** @brief Select preferred SRGB surface format when available. */
VkSurfaceFormatKHR VulkanSwapchainRunner::choose_surface_format(
    const std::vector<VkSurfaceFormatKHR>& formats) const 
    {
    for (const auto& format : formats) 
    {
        if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) 
        {
            return format;
        }
    }

    return formats.front();
}

/** @brief Select mailbox present mode when available, fallback FIFO otherwise. */
VkPresentModeKHR VulkanSwapchainRunner::choose_present_mode(const std::vector<VkPresentModeKHR>& modes) const 
{
    for (const auto mode : modes) 
    {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {return mode;}
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

/** @brief Resolve swapchain image extent using surface limits and framebuffer size. */
VkExtent2D VulkanSwapchainRunner::choose_extent(const VkSurfaceCapabilitiesKHR& capabilities) const 
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) 
    {
        return capabilities.currentExtent;
    }

    const auto [width, height] = framebuffer_size();

    VkExtent2D extent{};
    extent.width = std::clamp(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    extent.height = std::clamp(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    return extent;
}

/** @brief Destroy command buffers and all swapchain-dependent Vulkan resources. */
void VulkanSwapchainRunner::cleanup_swapchain() 
{
    if (context_ != nullptr && context_->device() != VK_NULL_HANDLE && command_pool_ != VK_NULL_HANDLE && !command_buffers_.empty()) 
    {
        vkFreeCommandBuffers(
            context_->device(),
            command_pool_,
            static_cast<uint32_t>(command_buffers_.size()),
            command_buffers_.data());
    }
    command_buffers_.clear();

    if (context_ != nullptr && context_->device() != VK_NULL_HANDLE) 
    {
        for (VkFramebuffer framebuffer : swapchain_framebuffers_) 
        {
            if (framebuffer != VK_NULL_HANDLE) 
            {
                vkDestroyFramebuffer(context_->device(), framebuffer, nullptr);
            }
        }
    }
    swapchain_framebuffers_.clear();

    if (context_ != nullptr && context_->device() != VK_NULL_HANDLE && render_pass_ != VK_NULL_HANDLE) 
    {
        vkDestroyRenderPass(context_->device(), render_pass_, nullptr);
    }
    render_pass_ = VK_NULL_HANDLE;

    if (context_ != nullptr && context_->device() != VK_NULL_HANDLE) 
    {
        for (VkImageView view : swapchain_image_views_) 
        {
            if (view != VK_NULL_HANDLE) 
            {
                vkDestroyImageView(context_->device(), view, nullptr);
            }
        }
    }
    swapchain_image_views_.clear();

    swapchain_images_.clear();

    if (context_ != nullptr && context_->device() != VK_NULL_HANDLE && swapchain_ != VK_NULL_HANDLE) 
    {
        vkDestroySwapchainKHR(context_->device(), swapchain_, nullptr);
    }
    swapchain_ = VK_NULL_HANDLE;
}

}  // namespace vkcpp
