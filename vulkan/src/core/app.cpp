/**
 * @file app.cpp
 * @brief High-level Vulkan runtime orchestration for viewer execution.
 *
 * Initializes the shared Vulkan context, prints device/runtime diagnostics,
 * and dispatches either dry-run flow or windowed rendering with the selected
 * backend. Centralizes startup/teardown policy for command-line entrypoints.
 */

#include "core/app.hpp"

#include "rendering/window_renderer.hpp"

#include <iostream>
#include <sstream>

namespace vkcpp 
{
namespace 
{

/** @brief Format Vulkan API version components into human-readable text. */
std::string vk_version_string(uint32_t version) 
{
    std::ostringstream oss;
    oss << VK_API_VERSION_MAJOR(version) << "."
        << VK_API_VERSION_MINOR(version) << "."
        << VK_API_VERSION_PATCH(version);
    return oss.str();
}

/** @brief Convert physical-device type enum to a compact label. */
const char* device_type_string(VkPhysicalDeviceType type) 
{
    switch (type) 
    {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            return "integrated";
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            return "discrete";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            return "virtual";
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            return "cpu";
        default:
            return "other";
    }
}

}  // namespace

/** @brief Print current runtime configuration and detected GPU summary. */
void VulkanBootstrap::print_runtime_summary(const Options& options) const 
{
#if VKCPP_USE_GLFW
    std::cout << "Window backend: GLFW\n";
#else
    std::cout << "Window backend: SFML (fallback)\n";
#endif
    std::cout << "Vulkan loader API: " << vk_version_string(context_.loader_api_version()) << "\n";
    std::cout << "Validation: " << (context_.validation_enabled() ? "enabled" : "disabled") << "\n";
    std::cout << "Portability enumeration: "
              << (context_.portability_instance_enabled() ? "enabled" : "disabled") << "\n";
    std::cout << "Render backend: " << options.render_backend << "\n";


    if (options.render_backend == "volume") 
    {
        std::cout << "Volume input dir: " << options.input_dir << "\n";
        if (!options.fields_csv.empty()) 
        {
            std::cout << "Volume fields: " << options.fields_csv << "\n";
        } 
        else 
        {
            std::cout << "Volume field: " << options.field << "\n";
        }
        std::cout << "Volume mode: " << options.volume_mode << "\n";
        if (!options.isolate_field.empty()) 
        {
            std::cout << "Isolate field: " << options.isolate_field << "\n";
        }


        std::cout << "Component cycle FPS: " << options.component_cycle_fps << "\n";
        std::cout << "Volume style: " << options.style << "\n";
        std::cout << "Texture mode: " << options.texture_mode << "\n";
        std::cout << "Camera mode: " << options.camera_mode << "\n";
        std::cout << "Camera orbit FPS: " << options.camera_orbit_fps << "\n";
        std::cout << "Camera distance: " << options.camera_distance << "\n";
        std::cout << "Camera height: " << options.camera_height << "\n";
        std::cout << "Camera FOV (deg): " << options.camera_fov_deg << "\n";
        std::cout << "Playback FPS: " << options.playback_fps << "\n";
        std::cout << "Ray steps: " << options.ray_steps << "\n";
        std::cout << "Ray threshold: " << options.ray_threshold << "\n";
        std::cout << "Ray opacity: " << options.ray_opacity << "\n";
        std::cout << "Ray brightness: " << options.ray_brightness << "\n";
        std::cout << "Ray ambient: " << options.ray_ambient << "\n";
        std::cout << "Ray anisotropy: " << options.ray_anisotropy << "\n";
        std::cout << "Ray max distance: " << options.ray_max_distance << "\n";
        std::cout << "Sun direction: (" << options.sun_dir_x << ", " << options.sun_dir_y << ", " << options.sun_dir_z
                  << ")\n";
    }

    if (options.capture_interval > 0.0f)
    {
        std::cout << "Capture interval: " << options.capture_interval << "s -> " << options.capture_dir << "/\n";
    }

    const auto& devices = context_.devices();
    std::cout << "Detected physical devices: " << devices.size() << "\n";

    for (std::size_t i = 0; i < devices.size(); ++i) 
    {
        const auto& device = devices[i];
        const bool selected = i == context_.selected_device_index();

        std::cout << "  [" << i << "] " << device.properties.deviceName
                  << " | type=" << device_type_string(device.properties.deviceType)
                  << " | api=" << vk_version_string(device.properties.apiVersion)
                  << " | score=" << device.score
                  << " | graphics=" << (device.queues.has_graphics() ? "yes" : "no");

        if (selected) {std::cout << " | selected";}

        std::cout << "\n";
    }

    if (context_.physical_device() != VK_NULL_HANDLE) 
    {
        const auto& selected = devices[context_.selected_device_index()];
        std::cout << "Selected device: [" << context_.selected_device_index() << "] "
                  << selected.properties.deviceName
                  << " (graphics queue family=" << context_.graphics_queue_family_index() << ")\n";
    }

    if (options.list_devices) {std::cout << "Device listing complete.\n";}
}

/** @brief Execute context initialization and optional swapchain test loop. */
int VulkanBootstrap::run(const Options& options) 
{
    std::string error;
    if (!context_.initialize(options, error)) 
    {
        std::cerr << "[vulkan] Initialization failed: " << error << "\n";
        return 1;
    }

    print_runtime_summary(options);

    if (options.window_test) 
    {
        WindowRunConfig config;
        config.width = options.window_width;
        config.height = options.window_height;
        config.max_frames = options.window_frames;
        config.backend_name = options.render_backend;
        config.input_dir = options.input_dir;
        config.field = options.field;
        config.fields_csv = options.fields_csv;
        config.volume_mode = options.volume_mode;
        config.isolate_field = options.isolate_field;
        config.component_cycle_fps = options.component_cycle_fps;
        config.style = options.style;
        config.texture_mode = options.texture_mode;
        config.camera_mode = options.camera_mode;
        config.camera_orbit_fps = options.camera_orbit_fps;
        config.camera_distance = options.camera_distance;
        config.camera_height = options.camera_height;
        config.camera_fov_deg = options.camera_fov_deg;
        config.playback_fps = options.playback_fps;
        config.ray_steps = options.ray_steps;
        config.ray_threshold = options.ray_threshold;
        config.ray_opacity = options.ray_opacity;
        config.ray_brightness = options.ray_brightness;
        config.ray_ambient = options.ray_ambient;
        config.ray_anisotropy = options.ray_anisotropy;
        config.ray_max_distance = options.ray_max_distance;
        config.sun_dir_x = options.sun_dir_x;
        config.sun_dir_y = options.sun_dir_y;
        config.sun_dir_z = options.sun_dir_z;
        config.capture_interval = options.capture_interval;
        config.capture_dir = options.capture_dir;

        VulkanSwapchainRunner runner;
        if (!runner.initialize(context_, config, error)) 
        {
            std::cerr << "[vulkan] Window test setup failed: " << error << "\n";
            return 1;
        }

        if (config.max_frames > 0) 
        {
            std::cout << "Running window/swapchain test for " << config.max_frames << " frames...\n";
        } 
        else 
        {
            std::cout << "Running window/swapchain test (unlimited frames, close window to exit)...\n";
        }
        if (!runner.run(error)) 
        {
            std::cerr << "[vulkan] Window test failed: " << error << "\n";
            return 1;
        }
        std::cout << "Window/swapchain test complete.\n";
        return 0;
    }

    if (options.dry_run) 
    {
        std::cout << "Dry-run complete.\n";
    } 
    else 
    {
        std::cout << "Context ready. Next step: graphics pipeline creation.\n";
    }

    return 0;
}

}  // namespace vkcpp
