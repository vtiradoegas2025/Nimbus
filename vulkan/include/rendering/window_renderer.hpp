/**
 * @file window_renderer.hpp
 * @brief Swapchain runner for interactive Vulkan window rendering.
 *
 * Owns surface/swapchain lifecycle, per-frame synchronization, and backend
 * dispatch for either clear-pass or volume rendering. Encapsulates resize
 * handling so higher-level orchestration stays focused on runtime control.
 */

#pragma once

#include "rendering/render_backend.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <vulkan/vulkan.h>

#ifndef VKCPP_USE_GLFW
#define VKCPP_USE_GLFW 0
#endif

#if VKCPP_USE_GLFW
#include <GLFW/glfw3.h>
#else
#include <SFML/Window.hpp>
#endif

namespace vkcpp 
{

class VulkanContext;
namespace camera {
class CameraInputTracker;
}

/**
 * @brief Runtime configuration for the windowed swapchain loop.
 */
struct WindowRunConfig 
{
    uint32_t width = 1280;
    uint32_t height = 720;
    // 0 means unlimited (run until the user closes the window).
    int max_frames = 0;
    std::string backend_name = "clear";
    std::string input_dir = "data/exports";
    std::string field = "theta";
    std::string fields_csv;
    std::string volume_mode = "supercell";
    std::string isolate_field;
    float component_cycle_fps = 0.50f;
    std::string style = "default";
    std::string texture_mode = "natural";
    std::string camera_mode = "orbit";
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
    float sun_dir_x = 0.66f;
    float sun_dir_y = 0.34f;
    float sun_dir_z = 0.67f;
    float capture_interval = 0.0f;
    std::string capture_dir = "captures";
};

/**
 * @brief Manages window events, swapchain state, and frame submission.
 */
class VulkanSwapchainRunner 
{
public:
    VulkanSwapchainRunner() = default;
    /** @brief Destroy all owned Vulkan/window resources. */
    ~VulkanSwapchainRunner();

    VulkanSwapchainRunner(const VulkanSwapchainRunner&) = delete;
    VulkanSwapchainRunner& operator=(const VulkanSwapchainRunner&) = delete;

    /**
     * @brief Initialize window, surface, swapchain, and selected render backend.
     * @param context Initialized Vulkan context.
     * @param config Window and backend configuration.
     * @param error Output message on failure.
     * @return `true` on success.
     */
    bool initialize(VulkanContext& context, const WindowRunConfig& config, std::string& error);

    /**
     * @brief Execute the render loop until frame limit or window close.
     * @param error Output message on failure.
     * @return `true` on success.
     */
    bool run(std::string& error);

    /** @brief Free all resources created by `initialize()`. */
    void shutdown();

private:
    struct SwapchainSupportDetails 
    {
        VkSurfaceCapabilitiesKHR capabilities{};
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> present_modes;
    };

    /** @brief Create the native OS window. */
    bool create_window(std::string& error);

    /** @brief Create a Vulkan surface bound to the active window. */
    bool create_surface(std::string& error);

    /** @brief Verify selected queue family supports presentation to the surface. */
    bool ensure_present_support(std::string& error);

    /** @brief Create swapchain images for the current surface capabilities. */
    bool create_swapchain(std::string& error);

    /** @brief Create per-image views for swapchain color attachments. */
    bool create_image_views(std::string& error);

    /** @brief Create render pass used by active backend. */
    bool create_render_pass(std::string& error);

    /** @brief Create framebuffers that bind swapchain views to render pass. */
    bool create_framebuffers(std::string& error);

    /** @brief Create command pool for graphics queue submissions. */
    bool create_command_pool(std::string& error);

    /** @brief Allocate command buffers matching swapchain image count. */
    bool create_command_buffers(std::string& error);

    /** @brief Record draw commands for each framebuffer. */
    bool record_command_buffers(std::string& error);

    /** @brief Instantiate and initialize requested render backend. */
    bool create_render_backend(std::string& error);

    /** @brief Create semaphores and fences used by frame pacing. */
    bool create_sync_objects(std::string& error);

    /** @brief Rebuild swapchain-dependent resources after resize/out-of-date events. */
    bool recreate_swapchain(std::string& error);

    /** @brief Submit one frame and present it to the surface. */
    bool draw_frame(std::string& error);

    /** @brief Pump pending platform events and refresh camera input snapshot. */
    void process_events();

    /** @brief Check whether the window remains open. */
    [[nodiscard]] bool is_window_open() const;

    /** @brief Destroy the window and backend windowing runtime state. */
    void close_window();
    
    /** @brief Query current framebuffer dimensions in pixels. */
    [[nodiscard]] std::pair<uint32_t, uint32_t> framebuffer_size() const;

#if VKCPP_USE_GLFW
    static void framebuffer_resize_callback(GLFWwindow* window, int width, int height);
    static void scroll_callback(GLFWwindow* window, double x_offset, double y_offset);
#endif

    /**
     * @brief Query surface capabilities, formats, and present modes.
     * @param device Selected physical device.
     * @param out_details Populated swapchain support details.
     * @param error Output message on failure.
     * @return `true` on success.
     */
    bool query_swapchain_support(VkPhysicalDevice device,
                                 SwapchainSupportDetails& out_details,
                                 std::string& error) const;
    /** @brief Choose preferred swapchain surface format from supported list. */
    VkSurfaceFormatKHR choose_surface_format(const std::vector<VkSurfaceFormatKHR>& formats) const;
    /** @brief Choose preferred presentation mode from supported list. */
    VkPresentModeKHR choose_present_mode(const std::vector<VkPresentModeKHR>& modes) const;
    /** @brief Choose render extent for current window dimensions and surface limits. */
    VkExtent2D choose_extent(const VkSurfaceCapabilitiesKHR& capabilities) const;

    /** @brief Destroy resources bound to the current swapchain. */
    void cleanup_swapchain();

    /** @brief Update window title with frame rate statistics. */
    void update_title_fps(float delta_seconds);

    /** @brief Print periodic performance summary to stderr. */
    void log_perf_summary();

    /** @brief Capture the last-rendered swapchain image to a PPM file on disk. */
    bool capture_frame(uint32_t image_index, std::string& error);

    /** @brief Create or recreate the staging buffer used for screenshot readback. */
    bool ensure_capture_staging(std::string& error);

    /** @brief Release staging buffer resources used for capture. */
    void destroy_capture_resources();

    VulkanContext* context_ = nullptr;
    WindowRunConfig config_{};

#if VKCPP_USE_GLFW
    GLFWwindow* window_ = nullptr;
    bool glfw_initialized_ = false;
#else
    sf::Window window_;
    bool window_created_ = false;
#endif
    bool framebuffer_resized_ = false;
    camera::CameraInputTracker* camera_input_tracker_ = nullptr;

    VkSurfaceKHR surface_ = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat swapchain_image_format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchain_extent_{};
    std::vector<VkImage> swapchain_images_;
    std::vector<VkImageView> swapchain_image_views_;
    std::vector<VkFramebuffer> swapchain_framebuffers_;

    VkRenderPass render_pass_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> command_buffers_;
    std::unique_ptr<RenderBackend> render_backend_;

    static constexpr std::size_t kMaxFramesInFlight = 2;
    std::array<VkSemaphore, kMaxFramesInFlight> image_available_semaphores_{};
    std::array<VkSemaphore, kMaxFramesInFlight> render_finished_semaphores_{};
    std::array<VkFence, kMaxFramesInFlight> in_flight_fences_{};
    std::vector<VkFence> images_in_flight_;
    std::size_t current_frame_ = 0;

    // Performance timing.
    std::chrono::steady_clock::time_point perf_epoch_{};
    std::chrono::steady_clock::time_point last_title_update_{};
    std::chrono::steady_clock::time_point last_perf_log_{};
    int title_frame_count_ = 0;
    float title_frame_time_sum_ = 0.0f;
    int log_frame_count_ = 0;
    float log_frame_time_sum_ = 0.0f;
    float log_frame_time_min_ = 0.0f;
    float log_frame_time_max_ = 0.0f;

    // Screen capture.
    bool capture_pending_ = false;
    uint32_t capture_image_index_ = 0;
    double capture_accumulator_ = 0.0;
    int capture_count_ = 0;
    VkBuffer capture_staging_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory capture_staging_memory_ = VK_NULL_HANDLE;
    VkDeviceSize capture_staging_size_ = 0;
    uint32_t capture_staging_width_ = 0;
    uint32_t capture_staging_height_ = 0;
};

}  // namespace vkcpp
