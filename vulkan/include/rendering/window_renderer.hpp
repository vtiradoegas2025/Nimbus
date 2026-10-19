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
#include <chrono>
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
    // 0 means unlimited. A positive value is a graceful wall-clock deadline.
    float max_seconds = 0.0f;
    // Non-negative schedules one deterministic qualification resize.
    int resize_frame = -1;
    uint32_t resize_width = 960;
    uint32_t resize_height = 540;
    float viewer_fps = 30.0f;
    std::string present_mode = "fifo";
    std::string backend_name = "clear";
    std::string input_dir = "data/exports";
    std::string field = "theta";
    std::string fields_csv;
    std::string volume_mode = "supercell";
    std::string isolate_field;
    float component_cycle_fps = 0.50f;
    std::string style = "default";
    std::string texture_mode = "smooth";
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
    std::array<float, 3> voxel_spacing_m = {0.0f, 0.0f, 0.0f};
    float vertical_exaggeration = 1.0f;
    float sun_dir_x = 0.66f;
    float sun_dir_y = 0.34f;
    float sun_dir_z = 0.67f;
    float capture_interval = 0.0f;
    // 0 means unlimited interval captures; positive values cap the session.
    int capture_limit = 0;
    std::string capture_dir = "captures";
    // Diverging-colormap poles for signed (bipolar) fields (w, vorticity, ...).
    std::array<float, 3> diverge_neg_color = {0.16f, 0.40f, 0.92f};
    std::array<float, 3> diverge_pos_color = {0.92f, 0.24f, 0.16f};
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

    /** @brief Create the native window before Vulkan instance/device selection. */
    bool prepare_window(const WindowRunConfig& config, std::string& error);

    /** @brief Return the platform's required Vulkan instance extensions. */
    bool required_instance_extensions(std::vector<const char*>& extensions, std::string& error) const;

    /** @brief Bind an initialized instance and create this window's surface. */
    bool create_surface(VulkanContext& context, std::string& error);

    /** @brief Build swapchain/rendering state after surface-aware device selection. */
    bool initialize_rendering(VulkanContext& context, std::string& error);

    [[nodiscard]] VkSurfaceKHR surface() const { return surface_; }

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
    bool create_bound_surface(std::string& error);

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

    /** @brief Create one presentation-completion semaphore per swapchain image. */
    bool create_render_finished_semaphores(std::string& error);

    /** @brief Destroy swapchain-image-owned presentation semaphores. */
    void destroy_render_finished_semaphores();

    /** @brief Rebuild swapchain-dependent resources after resize/out-of-date events. */
    bool recreate_swapchain(std::string& error);

    /** @brief Submit one frame and present it to the surface. */
    bool draw_frame(std::string& error);

    /** @brief Wait/pump platform events and refresh camera input snapshot. */
    void process_events(double wait_timeout_seconds);

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

    /** @brief Collect the completed GPU timestamp pair for one frame slot. */
    void collect_render_timestamp(std::size_t frame_slot);

    /** @brief Record readback before presentation into the current capture slot. */
    bool record_capture_copy(std::size_t slot_index, uint32_t image_index, std::string& error);

    /** @brief Create or recreate one capture-ring staging buffer. */
    bool ensure_capture_staging(std::size_t slot_index, std::string& error);

    /** @brief Save a completed capture-ring slot without touching its swapchain image. */
    bool write_capture(std::size_t slot_index, std::string& error);

    /** @brief Save every completed capture after the device has become idle. */
    void flush_pending_captures();

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
    // Presentation completion is owned by the acquired swapchain image, not
    // the CPU frame slot: reacquiring that image proves its prior wait is done.
    std::vector<VkSemaphore> render_finished_semaphores_;
    std::array<VkFence, kMaxFramesInFlight> in_flight_fences_{};
    std::vector<VkFence> images_in_flight_;
    std::size_t current_frame_ = 0;
    std::uint64_t swapchain_recreation_count_ = 0;

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
    std::array<VkQueryPool, kMaxFramesInFlight> render_timestamp_query_pools_{};
    std::array<bool, kMaxFramesInFlight> render_timestamp_pending_{};
    bool render_timestamps_enabled_ = false;
    uint32_t render_timestamp_valid_bits_ = 0;
    double render_timestamp_period_ns_ = 0.0;
    double log_gpu_time_ms_ = 0.0;
    std::uint64_t log_gpu_sample_count_ = 0;
    double session_gpu_time_ms_ = 0.0;
    std::uint64_t session_gpu_sample_count_ = 0;

    struct CaptureSlot
    {
        VkBuffer staging_buffer = VK_NULL_HANDLE;
        VkDeviceMemory staging_memory = VK_NULL_HANDLE;
        VkDeviceSize staging_size = 0;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        uint32_t width = 0;
        uint32_t height = 0;
        VkFormat format = VK_FORMAT_UNDEFINED;
        int capture_number = 0;
        std::chrono::system_clock::time_point wall_time{};
        bool pending_write = false;
    };

    // Screen capture. Slots share the normal per-frame fences, so readback from
    // frame N is encoded only when that frame slot is naturally reusable.
    bool capture_pending_ = false;
    bool capture_supported_ = false;
    double capture_accumulator_ = 0.0;
    int capture_count_ = 0;
    std::array<CaptureSlot, kMaxFramesInFlight> capture_slots_{};
};

}  // namespace vkcpp
