#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

#ifdef VK_NV_cuda_kernel_launch
#include "compute/cuda_retained_flux_ptx.hpp"

namespace tmv_vulkan
{
class CudaFluxKernel
{
public:
    bool inspect(VkPhysicalDevice physical, PFN_vkGetPhysicalDeviceFeatures2 get_features,
                 PFN_vkGetPhysicalDeviceProperties2 get_properties, std::string& error)
    {
        address_features_ = {};
        address_features_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
        cuda_features_ = {};
        cuda_features_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV;
        address_features_.pNext = &cuda_features_;
        VkPhysicalDeviceFeatures2 features{};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features.pNext = &address_features_;
        get_features(physical, &features);
        VkPhysicalDeviceCudaKernelLaunchPropertiesNV capability{};
        capability.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_PROPERTIES_NV;
        VkPhysicalDeviceProperties2 properties{};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties.pNext = &capability;
        get_properties(physical, &properties);
        if (!address_features_.bufferDeviceAddress || !cuda_features_.cudaKernelLaunchFeatures ||
            capability.computeCapabilityMajor * 10 + capability.computeCapabilityMinor < 75)
        {
            error = "CUDA assist requires Vulkan CUDA launch, buffer device addresses, and compute capability 7.5 or newer";
            return false;
        }
        address_features_.bufferDeviceAddressCaptureReplay = VK_FALSE;
        address_features_.bufferDeviceAddressMultiDevice = VK_FALSE;
        return true;
    }

    const void* feature_chain(const void* next)
    {
        address_features_.pNext = &cuda_features_;
        cuda_features_.pNext = const_cast<void*>(next);
        return &address_features_;
    }

    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr loader, std::string& error)
    {
        device_ = device;
        create_module_ = reinterpret_cast<PFN_vkCreateCudaModuleNV>(loader(device, "vkCreateCudaModuleNV"));
        create_function_ = reinterpret_cast<PFN_vkCreateCudaFunctionNV>(loader(device, "vkCreateCudaFunctionNV"));
        destroy_module_ = reinterpret_cast<PFN_vkDestroyCudaModuleNV>(loader(device, "vkDestroyCudaModuleNV"));
        destroy_function_ = reinterpret_cast<PFN_vkDestroyCudaFunctionNV>(loader(device, "vkDestroyCudaFunctionNV"));
        launch_ = reinterpret_cast<PFN_vkCmdCudaLaunchKernelNV>(loader(device, "vkCmdCudaLaunchKernelNV"));
        buffer_address_ = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(loader(device, "vkGetBufferDeviceAddressKHR"));
        if (!buffer_address_)
            buffer_address_ = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(loader(device, "vkGetBufferDeviceAddress"));
        if (!create_module_ || !create_function_ || !destroy_module_ || !destroy_function_ || !launch_ || !buffer_address_)
        {
            error = "CUDA assist entry points are unavailable";
            return false;
        }
        VkCudaModuleCreateInfoNV module_info{};
        module_info.sType = VK_STRUCTURE_TYPE_CUDA_MODULE_CREATE_INFO_NV;
        module_info.dataSize = sizeof(cuda_flux_ptx);
        module_info.pData = cuda_flux_ptx;
        VkResult result = create_module_(device_, &module_info, nullptr, &module_);
        if (result != VK_SUCCESS)
        {
            error = "CUDA module creation failed: " + std::to_string(result);
            return false;
        }
        VkCudaFunctionCreateInfoNV function_info{};
        function_info.sType = VK_STRUCTURE_TYPE_CUDA_FUNCTION_CREATE_INFO_NV;
        function_info.module = module_;
        function_info.pName = "nimbus_retained_flux";
        result = create_function_(device_, &function_info, nullptr, &function_);
        if (result != VK_SUCCESS)
        {
            error = "CUDA function creation failed: " + std::to_string(result);
            return false;
        }
        return true;
    }

    bool ready() const noexcept { return function_ != VK_NULL_HANDLE; }

    VkDeviceAddress address(VkBuffer buffer) const
    {
        VkBufferDeviceAddressInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        info.buffer = buffer;
        return buffer_address_(device_, &info);
    }

    void record(VkCommandBuffer command, const std::array<VkDeviceAddress, 7>& addresses,
                int nx, int ny, int nz, float frame_u, float frame_v, double weight,
                const VkPhysicalDeviceLimits& limits) const
    {
        const void* parameters[] = {&addresses[0], &addresses[1], &addresses[2], &addresses[3],
                                    &addresses[4], &addresses[5], &addresses[6],
                                    &nx, &ny, &nz, &frame_u, &frame_v, &weight};
        const auto groups = static_cast<std::uint32_t>((std::uint64_t(nx) * ny * nz + 63) / 64);
        VkCudaLaunchInfoNV launch{};
        launch.sType = VK_STRUCTURE_TYPE_CUDA_LAUNCH_INFO_NV;
        launch.function = function_;
        launch.gridDimX = std::min(groups, limits.maxComputeWorkGroupCount[0]);
        launch.gridDimY = (groups + launch.gridDimX - 1) / launch.gridDimX;
        launch.gridDimZ = 1;
        launch.blockDimX = 64;
        launch.blockDimY = launch.blockDimZ = 1;
        launch.paramCount = std::size(parameters);
        launch.pParams = parameters;
        launch_(command, &launch);
    }

    void reset() noexcept
    {
        if (function_ && destroy_function_) destroy_function_(device_, function_, nullptr);
        if (module_ && destroy_module_) destroy_module_(device_, module_, nullptr);
        function_ = VK_NULL_HANDLE;
        module_ = VK_NULL_HANDLE;
        device_ = VK_NULL_HANDLE;
    }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkCudaModuleNV module_ = VK_NULL_HANDLE;
    VkCudaFunctionNV function_ = VK_NULL_HANDLE;
    VkPhysicalDeviceBufferDeviceAddressFeatures address_features_{};
    VkPhysicalDeviceCudaKernelLaunchFeaturesNV cuda_features_{};
    PFN_vkCreateCudaModuleNV create_module_ = nullptr;
    PFN_vkCreateCudaFunctionNV create_function_ = nullptr;
    PFN_vkDestroyCudaModuleNV destroy_module_ = nullptr;
    PFN_vkDestroyCudaFunctionNV destroy_function_ = nullptr;
    PFN_vkCmdCudaLaunchKernelNV launch_ = nullptr;
    PFN_vkGetBufferDeviceAddress buffer_address_ = nullptr;
};
}
#endif
