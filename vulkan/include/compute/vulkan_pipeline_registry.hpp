#pragma once

/**
 * @file vulkan_pipeline_registry.hpp
 * @brief Vulkan pipeline/cache ownership indexed by backend-neutral KernelId.
 */

#include "compute/kernel_manifest.hpp"
#include "compute/vulkan_descriptor_cache.hpp"
#include "compute/vulkan_submission_lifecycle.hpp"

#if defined(__has_include)
#if __has_include(<vulkan/vulkan.h>)
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#define NIMBUS_VULKAN_PIPELINE_REGISTRY_AVAILABLE 1
#endif
#endif

#if defined(NIMBUS_VULKAN_PIPELINE_REGISTRY_AVAILABLE)

#include <array>
#include <string>

namespace tmv_vulkan
{

struct ComputePipelineState
{
    // Set zero serves public single-field operations. Additional immutable
    // sets support one-recording batches: diffusion uses five and homogeneous
    // Recycling may bind up to 63 independently resident fields. The
    // scientific owner-root count is smaller, but active schemes can add
    // private moments, residuals, and persistent stochastic caches.
    static constexpr std::size_t kMaximumDescriptorSets = 64;

    std::string debug_name;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kMaximumDescriptorSets> descriptor_sets{};
    std::uint32_t descriptor_set_count = 0;
    std::uint32_t binding_count = 0;
    std::uint32_t push_constant_size = 0;
    const SubmissionLifecycle* lifecycle = nullptr;
    SubmissionLifecycle::Generation owner_generation = 0;
    tmv::vulkan_compute::DescriptorBindingCache descriptor_cache;
    std::array<tmv::vulkan_compute::DescriptorBindingCache,
               kMaximumDescriptorSets> descriptor_set_caches{};

    bool is_ready() const noexcept
    {
        return lifecycle != nullptr && lifecycle->pipeline_is_ready(
            owner_generation, pipeline != VK_NULL_HANDLE);
    }
};

class PipelineRegistry
{
public:
    ComputePipelineState& get(nimbus::compute::KernelId id) noexcept
    {
        return states_[static_cast<std::size_t>(id)];
    }

    const ComputePipelineState& get(nimbus::compute::KernelId id) const noexcept
    {
        return states_[static_cast<std::size_t>(id)];
    }

    auto begin() noexcept { return states_.begin(); }
    auto end() noexcept { return states_.end(); }
    auto begin() const noexcept { return states_.begin(); }
    auto end() const noexcept { return states_.end(); }

    void abandon() noexcept
    {
        for (auto& state : states_) state = {};
    }

private:
    std::array<ComputePipelineState, nimbus::compute::kernel_count()> states_{};
};

} // namespace tmv_vulkan

#endif
