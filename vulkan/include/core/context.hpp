/**
 * @file context.hpp
 * @brief Vulkan instance/device bootstrap and capability discovery interfaces.
 *
 * Encapsulates loader negotiation, instance creation, physical-device probing,
 * queue-family selection, and logical-device setup. Provides readonly accessors
 * so render systems can consume selected handles without owning lifecycle logic.
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace vkcpp 
{

struct Options;

/** @brief Thread-safe fail-closed tally for validation callback severities. */
class ValidationMessageTally
{
public:
    void record(VkDebugUtilsMessageSeverityFlagBitsEXT severity)
    {
        if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0)
        {
            error_count_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void reset() { error_count_.store(0, std::memory_order_relaxed); }

    [[nodiscard]] std::uint64_t error_count() const
    {
        return error_count_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<std::uint64_t> error_count_{0};
};

/**
 * @brief Tracks queue-family indices required by this application.
 */
struct QueueFamilySelection 
{
    uint32_t graphics_family = std::numeric_limits<uint32_t>::max();
    uint32_t present_family = std::numeric_limits<uint32_t>::max();

    [[nodiscard]] bool has_graphics() const 
    {
        return graphics_family != std::numeric_limits<uint32_t>::max();
    }

    [[nodiscard]] bool has_present() const
    {
        return present_family != std::numeric_limits<uint32_t>::max();
    }

    [[nodiscard]] bool complete(bool require_present) const
    {
        return has_graphics() && (!require_present || has_present());
    }
};

/** @brief Pure queue policy: prefer one family supporting both roles, else keep separate roles. */
[[nodiscard]] inline QueueFamilySelection choose_queue_families(
    const std::vector<VkQueueFamilyProperties>& families,
    const std::vector<VkBool32>& present_support)
{
    QueueFamilySelection selection;
    for (std::size_t i = 0; i < families.size(); ++i)
    {
        const bool graphics = (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        const bool present = i < present_support.size() && present_support[i] == VK_TRUE;
        if (graphics && !selection.has_graphics())
        {
            selection.graphics_family = static_cast<uint32_t>(i);
        }
        if (present && !selection.has_present())
        {
            selection.present_family = static_cast<uint32_t>(i);
        }
        if (graphics && present)
        {
            selection.graphics_family = static_cast<uint32_t>(i);
            selection.present_family = static_cast<uint32_t>(i);
            break;
        }
    }
    return selection;
}

/**
 * @brief Snapshot of physical-device properties and suitability scoring.
 */
struct PhysicalDeviceInfo 
{
    VkPhysicalDevice handle = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceFeatures features{};
    QueueFamilySelection queues{};
    std::vector<VkExtensionProperties> extensions;
    int score = -1;
    bool suitable = false;
};

/**
 * @brief Owns Vulkan instance/device lifecycle for the viewer runtime.
 */
class VulkanContext 
{
public:
    VulkanContext() = default;

    /** @brief Ensure all Vulkan handles are destroyed before object teardown. */
    ~VulkanContext();
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    /**
     * @brief Build Vulkan instance, select a physical device, and create logical device.
     * @param options Runtime flags controlling validation and selection.
     * @param error Output message on failure.
     * @return `true` on success.
     */
    bool initialize(const Options& options, std::string& error);

    /** @brief Create only the instance, using extensions supplied by the window system. */
    bool initialize_instance(const Options& options,
                             const std::vector<const char*>& required_instance_extensions,
                             std::string& error);

    /** @brief Select a surface-capable device and create its logical queues. */
    bool initialize_device(const Options& options, VkSurfaceKHR surface, std::string& error);

    /** @brief Destroy all owned Vulkan resources and reset cached state. */
    void shutdown();


    /*==============================
    Public members for the VulkanContext class
    ==============================*/
    [[nodiscard]] uint32_t loader_api_version() const { return loader_api_version_; }
    [[nodiscard]] bool validation_enabled() const { return validation_enabled_; }
    [[nodiscard]] std::uint64_t validation_error_count() const
    {
        return validation_messages_.error_count();
    }
    [[nodiscard]] bool portability_instance_enabled() const { return portability_instance_enabled_; }

    [[nodiscard]] VkInstance instance() const { return instance_; }
    [[nodiscard]] VkPhysicalDevice physical_device() const { return physical_device_; }
    [[nodiscard]] VkDevice device() const { return device_; }
    [[nodiscard]] VkQueue graphics_queue() const { return graphics_queue_; }
    [[nodiscard]] VkQueue present_queue() const { return present_queue_; }
    [[nodiscard]] uint32_t graphics_queue_family_index() const { return graphics_queue_family_index_; }
    [[nodiscard]] uint32_t present_queue_family_index() const { return present_queue_family_index_; }
    [[nodiscard]] std::size_t selected_device_index() const { return selected_device_index_; }
    [[nodiscard]] const std::vector<PhysicalDeviceInfo>& devices() const { return devices_; }

private:
    /** @brief Create a Vulkan instance with required layers/extensions. */
    bool create_instance(bool enable_validation,
                         const std::vector<const char*>& required_instance_extensions,
                         std::string& error);

    /** @brief Register validation debug messenger when enabled. */
    bool setup_debug_messenger(std::string& error);

    /** @brief Enumerate physical devices and cache capabilities. */
    bool discover_physical_devices(VkSurfaceKHR surface, std::string& error);

    /** @brief Select a suitable physical device based on user choice/score. */
    bool select_physical_device(int preferred_device_index, std::string& error);

    /** @brief Create logical device and graphics queue from selected GPU. */
    bool create_logical_device(std::string& error);

    /** @brief Locate queue-family indices relevant to this renderer. */
    static QueueFamilySelection find_queue_families(VkPhysicalDevice device, VkSurfaceKHR surface);

    /** @brief Check whether validation layer is advertised by loader. */
    static bool has_validation_layer();

    /** @brief Check whether a given Vulkan instance extension is advertised. */
    static bool has_instance_extension(const char* extension_name);

    /** @brief Check whether a given Vulkan device extension is advertised. */
    static bool has_device_extension(VkPhysicalDevice device, const char* extension_name);

    /** @brief Query highest Vulkan API version exposed by current loader. */
    static uint32_t query_loader_api_version();


    /*==============================
    Private members for the VulkanContext class
    ==============================*/
    uint32_t loader_api_version_ = VK_API_VERSION_1_0;
    bool validation_enabled_ = false;
    bool debug_utils_enabled_ = false;
    bool portability_instance_enabled_ = false;
    bool require_swapchain_extension_ = false;
    ValidationMessageTally validation_messages_{};

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;

    std::vector<PhysicalDeviceInfo> devices_;
    std::size_t selected_device_index_ = 0;

    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    VkQueue present_queue_ = VK_NULL_HANDLE;
    uint32_t graphics_queue_family_index_ = std::numeric_limits<uint32_t>::max();
    uint32_t present_queue_family_index_ = std::numeric_limits<uint32_t>::max();
};

}  // namespace vkcpp
