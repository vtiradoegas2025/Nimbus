/**
 * @file context.cpp
 * @brief Vulkan bootstrap implementation for instance and device setup.
 *
 * Implements loader-version probing, layer/extension negotiation, physical-device
 * discovery, suitability scoring, and logical-device creation. Keeps initialization
 * failures explicit so callers can report deterministic diagnostics to users.
 */

#include "core/context.hpp"

#include "core/app.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace vkcpp 
{
namespace 
{

/*==============================
Constants for the VulkanContext class
==============================*/
constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

#ifndef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
#define VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME "VK_KHR_portability_enumeration"
#endif

#ifndef VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
#define VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR 0x00000001
#endif

#ifndef VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
#define VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME "VK_KHR_get_physical_device_properties2"
#endif

#ifndef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
#define VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME "VK_KHR_portability_subset"
#endif

#ifndef VK_KHR_SURFACE_EXTENSION_NAME
#define VK_KHR_SURFACE_EXTENSION_NAME "VK_KHR_surface"
#endif

#ifndef VK_EXT_METAL_SURFACE_EXTENSION_NAME
#define VK_EXT_METAL_SURFACE_EXTENSION_NAME "VK_EXT_metal_surface"
#endif


VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* callback_data,void*) 
{
    const char* level = "INFO";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) 
    {
        level = "ERROR";
    } 
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) 
    {
        level = "WARN";
    }

    if (callback_data != nullptr && callback_data->pMessage != nullptr) 
    {
        std::cerr << "[vulkan][validation][" << level << "] " << callback_data->pMessage << "\n";
    }

    return VK_FALSE;
}

/** @brief Build debug messenger configuration used during validation runs. */
VkDebugUtilsMessengerCreateInfoEXT build_debug_create_info() 
{
    VkDebugUtilsMessengerCreateInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = debug_callback;
    return info;
}

/** @brief Assign a coarse preference score by physical-device type. */
int device_type_score(VkPhysicalDeviceType type) 
{
    switch (type) 
    {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            return 1000;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            return 750;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            return 500;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            return 250;
        default:
            return 100;
    }
}

/** @brief Join extension names into a compact diagnostics string. */
std::string join_extension_names(const std::vector<const char*>& names)
{
    std::ostringstream oss;
    for (std::size_t i = 0; i < names.size(); ++i)
    {
        if (i > 0)
        {
            oss << ", ";
        }
        oss << names[i];
    }
    return oss.str();
}

/** @brief Extra diagnostics guidance for common instance-creation failures. */
std::string instance_failure_hint(VkResult result)
{
#if defined(__APPLE__)
    if (result == VK_ERROR_INCOMPATIBLE_DRIVER)
    {
        return " (hint: ensure Homebrew packages `vulkan-loader` and `molten-vk` are installed; "
               "if needed set VK_ICD_FILENAMES to /opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json)";
    }
#endif
    (void)result;
    return "";
}

}  // namespace

/** @brief Ensure Vulkan state is torn down when context leaves scope. */
VulkanContext::~VulkanContext() 
{
    shutdown();
}

/** @brief Initialize Vulkan instance/device resources for runtime usage. */
bool VulkanContext::initialize(const Options& options, std::string& error) 
{
    shutdown();

    loader_api_version_ = query_loader_api_version();
    validation_enabled_ = options.enable_validation;
    require_swapchain_extension_ = options.window_test;

    if (!create_instance(options.enable_validation, error)) {shutdown(); return false;}

    if (!setup_debug_messenger(error)) {shutdown(); return false;}

    if (!discover_physical_devices(error)) {shutdown(); return false;}

    if (!select_physical_device(options.preferred_device_index, error)) {shutdown(); return false;}

    if (!create_logical_device(error)) {shutdown(); return false;}

    return true;
}

/** @brief Destroy Vulkan handles and clear cached discovery metadata. */
void VulkanContext::shutdown() 
{
    if (device_ != VK_NULL_HANDLE) 
    {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    if (debug_messenger_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) 
    {
        auto* destroy_debug_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy_debug_messenger != nullptr) 
        {
            destroy_debug_messenger(instance_, debug_messenger_, nullptr);
        }
        debug_messenger_ = VK_NULL_HANDLE;
    }

    if (instance_ != VK_NULL_HANDLE) 
    {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }

    devices_.clear();
    selected_device_index_ = 0;
    physical_device_ = VK_NULL_HANDLE;
    graphics_queue_ = VK_NULL_HANDLE;
    graphics_queue_family_index_ = std::numeric_limits<uint32_t>::max();
    portability_instance_enabled_ = false;
    debug_utils_enabled_ = false;
    require_swapchain_extension_ = false;
}

/** @brief Create a Vulkan instance with selected validation and platform extensions. */
bool VulkanContext::create_instance(bool enable_validation, std::string& error) {
    debug_utils_enabled_ = false;
    portability_instance_enabled_ = false;

    std::vector<const char*> enabled_layers;
    std::vector<const char*> enabled_extensions;
    VkInstanceCreateFlags create_flags = 0;

    auto append_extension = [&enabled_extensions](const char* extension_name) 
    {
        const bool already_enabled =
            std::any_of(enabled_extensions.begin(), enabled_extensions.end(), [extension_name](const char* existing) 
            {
                return std::strcmp(existing, extension_name) == 0;
            });
        if (!already_enabled) 
        {
            enabled_extensions.push_back(extension_name);
        }
    };

    if (enable_validation) 
    {
        if (!has_validation_layer()) 
        {
            error = "validation requested but VK_LAYER_KHRONOS_validation is not available "
                    "(install vulkan-validationlayers)";
            return false;
        }
        enabled_layers.push_back(kValidationLayer);

        if (has_instance_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) 
        {
            append_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            debug_utils_enabled_ = true;
        }
    }

    if (require_swapchain_extension_) 
    {
        std::vector<const char*> required_extensions;
        required_extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#if defined(__APPLE__)
        required_extensions.push_back(VK_EXT_METAL_SURFACE_EXTENSION_NAME);
#endif

        for (const char* extension : required_extensions) 
        {
            if (!has_instance_extension(extension)) 
            {
                error = "required instance extension is unavailable: " + std::string(extension);
                return false;
            }
            append_extension(extension);
        }
    }

    const bool has_portability_enumeration =
        has_instance_extension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    const bool has_properties2 =
        has_instance_extension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    if (has_portability_enumeration) 
    {
        append_extension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        create_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        portability_instance_enabled_ = true;

        if (has_properties2) 
        {
            append_extension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        }
    }

    std::cout << "[vulkan] instance support: portability_enumeration="
              << (has_portability_enumeration ? "yes" : "no")
              << ", properties2=" << (has_properties2 ? "yes" : "no")
              << ", validation_layer="
              << (has_validation_layer() ? "yes" : "no") << "\n";

    const uint32_t preferred_api_version = std::min(loader_api_version_, VK_API_VERSION_1_1);

    VkDebugUtilsMessengerCreateInfoEXT debug_info = build_debug_create_info();

    auto try_create_instance = [&](uint32_t api_version,
                                   const std::vector<const char*>& extensions,
                                   VkInstanceCreateFlags flags,
                                   VkResult& out_result,
                                   VkInstance& out_instance)
    {
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "TornadoModel Vulkan Scaffold";
        app_info.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
        app_info.pEngineName = "TornadoModel";
        app_info.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
        app_info.apiVersion = api_version;

        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;
        create_info.flags = flags;
        create_info.enabledLayerCount = static_cast<uint32_t>(enabled_layers.size());
        create_info.ppEnabledLayerNames = enabled_layers.empty() ? nullptr : enabled_layers.data();
        create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        create_info.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();

        if (debug_utils_enabled_) {create_info.pNext = &debug_info;}

        out_result = vkCreateInstance(&create_info, nullptr, &out_instance);
        return out_result == VK_SUCCESS;
    };

    VkResult result = VK_ERROR_INITIALIZATION_FAILED;
    VkInstance created_instance = VK_NULL_HANDLE;
    std::vector<const char*> active_extensions = enabled_extensions;
    VkInstanceCreateFlags active_flags = create_flags;
    uint32_t active_api_version = preferred_api_version;

    if (try_create_instance(active_api_version, active_extensions, active_flags, result, created_instance))
    {
        instance_ = created_instance;
        return true;
    }

    if (enable_validation && result == VK_ERROR_LAYER_NOT_PRESENT) 
    {
        std::cerr << "[vulkan] Validation layer failed to load at runtime; retrying without validation.\n";
        validation_enabled_ = false;
        return create_instance(false, error);
    }

    if (result == VK_ERROR_INCOMPATIBLE_DRIVER && active_api_version != VK_API_VERSION_1_0)
    {
        std::cerr << "[vulkan] vkCreateInstance reported VK_ERROR_INCOMPATIBLE_DRIVER with API "
                  << VK_API_VERSION_MAJOR(active_api_version) << "."
                  << VK_API_VERSION_MINOR(active_api_version) << "."
                  << VK_API_VERSION_PATCH(active_api_version)
                  << "; retrying with Vulkan 1.0\n";
        active_api_version = VK_API_VERSION_1_0;
        if (try_create_instance(active_api_version, active_extensions, active_flags, result, created_instance))
        {
            instance_ = created_instance;
            return true;
        }
    }

    if (has_portability_enumeration)
    {
        std::cerr << "[vulkan] vkCreateInstance failed with portability enumeration enabled; retrying without it\n";
        active_extensions.erase(
            std::remove_if(
                active_extensions.begin(),
                active_extensions.end(),
                [](const char* extension_name)
                {
                    return std::strcmp(extension_name, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0 ||
                        std::strcmp(extension_name, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) == 0;
                }),
            active_extensions.end());
        active_flags &= ~VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        portability_instance_enabled_ = false;
        if (try_create_instance(active_api_version, active_extensions, active_flags, result, created_instance))
        {
            instance_ = created_instance;
            return true;
        }
    }

    error = "vkCreateInstance failed with VkResult=" + std::to_string(static_cast<int>(result)) +
            " (requested_api=" + std::to_string(VK_API_VERSION_MAJOR(preferred_api_version)) + "." +
            std::to_string(VK_API_VERSION_MINOR(preferred_api_version)) + "." +
            std::to_string(VK_API_VERSION_PATCH(preferred_api_version)) +
            ", active_api=" + std::to_string(VK_API_VERSION_MAJOR(active_api_version)) + "." +
            std::to_string(VK_API_VERSION_MINOR(active_api_version)) + "." +
            std::to_string(VK_API_VERSION_PATCH(active_api_version)) +
            ", extensions=[" + join_extension_names(active_extensions) + "])" +
            instance_failure_hint(result);
    return false;
}

/** @brief Create validation debug messenger if validation is active. */
bool VulkanContext::setup_debug_messenger(std::string& error) 
{
    if (!validation_enabled_ || !debug_utils_enabled_ || instance_ == VK_NULL_HANDLE) 
    {
        return true;
    }

    auto* create_debug_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
    if (create_debug_messenger == nullptr) 
    {
        error = "vkCreateDebugUtilsMessengerEXT function not available";
        return false;
    }

    const VkDebugUtilsMessengerCreateInfoEXT info = build_debug_create_info();
    const VkResult result = create_debug_messenger(instance_, &info, nullptr, &debug_messenger_);
    if (result != VK_SUCCESS) 
    {
        error = "vkCreateDebugUtilsMessengerEXT failed with VkResult=" +
                std::to_string(static_cast<int>(result));
        return false;
    }

    return true;
}

/** @brief Enumerate physical devices and cache properties/features/extensions. */
bool VulkanContext::discover_physical_devices(std::string& error) 
{
    uint32_t count = 0;
    VkResult result = vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (result != VK_SUCCESS) 
    {
        error = "vkEnumeratePhysicalDevices(count) failed with VkResult=" +
                std::to_string(static_cast<int>(result));
        return false;
    }

    if (count == 0) {error = "no Vulkan physical devices found"; return false;}

    std::vector<VkPhysicalDevice> handles(count, VK_NULL_HANDLE);
    result = vkEnumeratePhysicalDevices(instance_, &count, handles.data());
    if (result != VK_SUCCESS) 
    {
        error = "vkEnumeratePhysicalDevices(list) failed with VkResult=" +
                std::to_string(static_cast<int>(result));
        return false;
    }

    handles.resize(count);
    devices_.clear();
    devices_.reserve(handles.size());

    for (VkPhysicalDevice device : handles) 
    {
        PhysicalDeviceInfo info{};
        info.handle = device;
        vkGetPhysicalDeviceProperties(device, &info.properties);
        vkGetPhysicalDeviceFeatures(device, &info.features);
        info.queues = find_queue_families(device);

        uint32_t extension_count = 0;
        VkResult extension_result = vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);
        if (extension_result == VK_SUCCESS) 
        {
            info.extensions.resize(extension_count);
            if (extension_count > 0) 
            {
                extension_result = vkEnumerateDeviceExtensionProperties(
                    device,
                    nullptr,
                    &extension_count,
                    info.extensions.data());
                if (extension_result == VK_SUCCESS) 
                {
                    info.extensions.resize(extension_count);
                } 
                else 
                {
                    info.extensions.clear();
                    std::cerr << "[vulkan] warning: vkEnumerateDeviceExtensionProperties(list) failed with VkResult="
                              << static_cast<int>(extension_result) << "\n";
                }
            }
        }

        if (!info.queues.has_graphics()) 
        {
            info.score = -1;
            info.suitable = false;
        } 
        else 
        {
            info.score = device_type_score(info.properties.deviceType);
            info.score += static_cast<int>(info.properties.limits.maxImageDimension3D / 1024);
            info.suitable = true;
        }

        devices_.push_back(std::move(info));
    }

    return true;
}

/** @brief Choose either the requested GPU or best-scored suitable device. */
bool VulkanContext::select_physical_device(int preferred_device_index, std::string& error) 
{
    if (devices_.empty()) 
    {
        error = "physical device list is empty";
        return false;
    }

    if (preferred_device_index >= 0) 
    {
        const std::size_t index = static_cast<std::size_t>(preferred_device_index);
        if (index >= devices_.size()) 
        {
            error = "requested device index is out of range";
            return false;
        }
        if (!devices_[index].suitable) 
        {
            error = "requested device does not expose a graphics queue";
            return false;
        }
        selected_device_index_ = index;
    } 
    else 
    {
        int best_score = -1;
        std::size_t best_index = 0;
        bool found = false;

        for (std::size_t i = 0; i < devices_.size(); ++i) 
        {
            if (!devices_[i].suitable) {continue;}

            if (!found || devices_[i].score > best_score) 
            {
                best_score = devices_[i].score;
                best_index = i;
                found = true;
            }
        }

        if (!found) 
        {
            error = "no suitable Vulkan device (graphics queue required)";
            return false;
        }

        selected_device_index_ = best_index;
    }

    physical_device_ = devices_[selected_device_index_].handle;
    graphics_queue_family_index_ = devices_[selected_device_index_].queues.graphics_family;
    return true;
}

/** @brief Create logical device and obtain the primary graphics queue handle. */
bool VulkanContext::create_logical_device(std::string& error) 
{
    if (physical_device_ == VK_NULL_HANDLE || graphics_queue_family_index_ == std::numeric_limits<uint32_t>::max()) 
    {
        error = "physical device or graphics queue family is not selected";
        return false;
    }

    float queue_priority = 1.0f;

    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = graphics_queue_family_index_;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    std::vector<const char*> device_extensions;
    if (require_swapchain_extension_) 
    {
        if (!has_device_extension(physical_device_, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) 
        {
            error = "VK_KHR_swapchain is required for window mode but is not supported by selected device";
            return false;
        }
        device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    if (has_device_extension(physical_device_, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) 
    {
        device_extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    }

    VkPhysicalDeviceFeatures requested_features{};

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    device_info.ppEnabledExtensionNames =
        device_extensions.empty() ? nullptr : device_extensions.data();
    device_info.pEnabledFeatures = &requested_features;

    if (validation_enabled_) 
    {
        device_info.enabledLayerCount = 1;
        device_info.ppEnabledLayerNames = &kValidationLayer;
    }

    const VkResult result = vkCreateDevice(physical_device_, &device_info, nullptr, &device_);
    if (result != VK_SUCCESS) 
    {
        error = "vkCreateDevice failed with VkResult=" + std::to_string(static_cast<int>(result));
        return false;
    }

    vkGetDeviceQueue(device_, graphics_queue_family_index_, 0, &graphics_queue_);
    return true;
}

/** @brief Find first queue family that supports graphics commands. */
QueueFamilySelection VulkanContext::find_queue_families(VkPhysicalDevice device) 
{
    QueueFamilySelection selection;

    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, nullptr);
    if (family_count == 0) 
    {
        return selection;
    }

    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, families.data());

    for (uint32_t i = 0; i < family_count; ++i) 
    {
        if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) 
        {
            selection.graphics_family = i;
            break;
        }
    }

    return selection;
}

/** @brief Check if the standard Khronos validation layer is available. */
bool VulkanContext::has_validation_layer() 
{
    uint32_t count = 0;
    if (vkEnumerateInstanceLayerProperties(&count, nullptr) != VK_SUCCESS) 
    {
        return false;
    }

    std::vector<VkLayerProperties> layers(count);
    if (count > 0 && vkEnumerateInstanceLayerProperties(&count, layers.data()) != VK_SUCCESS) 
    {
        return false;
    }

    return std::any_of(layers.begin(), layers.end(), [](const VkLayerProperties& layer)
    {
        return std::strcmp(layer.layerName, kValidationLayer) == 0;
    });
}

/** @brief Check if a specific instance extension is exposed by the loader. */
bool VulkanContext::has_instance_extension(const char* extension_name) 
{
    uint32_t count = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) != VK_SUCCESS) 
    {
        return false;
    }

    std::vector<VkExtensionProperties> extensions(count);
    if (count > 0 && vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data()) != VK_SUCCESS) 
    {
        return false;
    }

    return std::any_of(extensions.begin(), extensions.end(), [extension_name](const VkExtensionProperties& ext) 
    {
        return std::strcmp(ext.extensionName, extension_name) == 0;
    });
}

/** @brief Check if a specific extension is supported by a physical device. */
bool VulkanContext::has_device_extension(VkPhysicalDevice device, const char* extension_name) 
{
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr) != VK_SUCCESS) 
    {
        return false;
    }

    std::vector<VkExtensionProperties> extensions(count);
    if (count > 0 && vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data()) != VK_SUCCESS) 
    {
        return false;
    }

    return std::any_of(extensions.begin(), extensions.end(), [extension_name](const VkExtensionProperties& ext) 
    {
        return std::strcmp(ext.extensionName, extension_name) == 0;
    });
}

/** @brief Query highest Vulkan API version reported by the loader. */
uint32_t VulkanContext::query_loader_api_version() 
{
    uint32_t version = VK_API_VERSION_1_0;

    auto* enumerate_instance_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
        vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));

    if (enumerate_instance_version != nullptr) 
    {
        enumerate_instance_version(&version);
    }

    return version;
}

}  // namespace vkcpp
