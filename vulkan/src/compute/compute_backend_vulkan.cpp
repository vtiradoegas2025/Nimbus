/**
 * @file compute_backend_vulkan.cpp
 * @brief Vulkan-specific runtime compute backend implementation.
 *
 * This file intentionally contains only Vulkan backend resource management and
 * capability checks. Runtime backend selection/fallback orchestration remains
 * in src/core/compute_backend.cpp for engine-level ownership.
 * I HIGHTLY ADVISE NOT EDITING THIS FILE IF YOU'RE NOT A VULKAN EXPERT.
 * If any edits need to be made, it will be very obvious with my comments 
   pointing out where to make the changes and how to make them.
 */

#include "compute/compute_backend.hpp"
#include "compute/compute_backend_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "core/simulation.hpp"
#include "compute/gpu_buffer_pool.hpp"

#if defined(__has_include)
#if __has_include(<vulkan/vulkan.h>)
#define TMV_HAS_VULKAN_COMPUTE_HEADERS 1
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#endif
#endif

#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS)
#if defined(__APPLE__) || defined(__linux__)
#include <dlfcn.h>
#define TMV_HAS_VULKAN_COMPUTE_DLOPEN 1
#endif

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
#endif

namespace
{

class VulkanComputeBackend final : public ComputeBackend
{
public:
    std::string name() const override { return "vulkan"; }

    bool initialize(std::string& error) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        shutdown();

        if (!open_loader(error))
        {
            shutdown();
            return false;
        }
        if (!load_global_functions(error))
        {
            shutdown();
            return false;
        }
        if (!create_instance(error))
        {
            shutdown();
            return false;
        }
        if (!load_instance_functions(error))
        {
            shutdown();
            return false;
        }
        if (!select_physical_device(error))
        {
            shutdown();
            return false;
        }
        if (!create_logical_device(error))
        {
            shutdown();
            return false;
        }
        if (!load_device_functions(error))
        {
            shutdown();
            return false;
        }

        if (log_normal_enabled())
        {
            std::cout << "[COMPUTE][VULKAN] loader_api="
                      << version_to_string(loader_api_version_)
                      << ", device_api="
                      << version_to_string(selected_device_properties_.apiVersion)
                      << ", device_index=" << selected_device_index_
                      << ", device=\"" << selected_device_properties_.deviceName << "\""
                      << ", type=" << device_type_name(selected_device_properties_.deviceType)
                      << ", compute_queue_family=" << compute_queue_family_index_
                      << ", queue_count=" << selected_compute_queue_family_.queueCount
                      << ", max_workgroup_invocations="
                      << selected_device_properties_.limits.maxComputeWorkGroupInvocations
                      << ", shared_mem_bytes="
                      << selected_device_properties_.limits.maxComputeSharedMemorySize
                      << std::endl;
        }

        // Set up compute pipelines (shaders, descriptors, command infrastructure)
        // Non-fatal: if shaders are missing we still initialize but without dispatch capability
        {
            std::string pipeline_error;
            if (!setup_compute_pipelines(pipeline_error))
            {
                log_vulkan_warning("compute pipeline setup failed: " + pipeline_error +
                                  " (GPU dispatch will be unavailable, CPU fallback will be used)");
            }
        }

        return true;
#elif defined(TMV_HAS_VULKAN_COMPUTE_HEADERS)
        error = "vulkan compute backend is unavailable on this platform "
                "(dynamic Vulkan loader path is unsupported)";
        return false;
#else
        error = "vulkan compute backend requested, but Vulkan headers were not found at build time";
        return false;
#endif
    }

    bool supports_vertical_flux_dispatch() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return tvd_pipeline_.is_ready();
#else
        return false;
#endif
    }

    bool supports_radial_advection_dispatch() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return radial_pipeline_.is_ready();
#else
        return false;
#endif
    }

    bool supports_azimuthal_advection_dispatch() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return azimuthal_pipeline_.is_ready();
#else
        return false;
#endif
    }

    bool supports_diffusion_dispatch() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return diffusion_pipeline_.is_ready();
#else
        return false;
#endif
    }

    bool supports_supercell_tendencies_dispatch() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return supercell_pipeline_.is_ready();
#else
        return false;
#endif
    }

    bool supports_cartesian_tendencies_dispatch() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return cartesian_pipeline_.is_ready();
#else
        return false;
#endif
    }

    bool supports_advection_x_dispatch() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return advection_x_pipeline_.is_ready();
#else
        return false;
#endif
    }

    bool supports_advection_y_dispatch() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return advection_y_pipeline_.is_ready();
#else
        return false;
#endif
    }

    bool supports_tornado_tendencies_dispatch() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return tornado_pipeline_.is_ready();
#else
        return false;
#endif
    }

    bool supports_kessler_pointwise_dispatch() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return kessler_pointwise_pipeline_.is_ready();
#else
        return false;
#endif
    }

    bool supports_kessler_sedimentation_dispatch() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return kessler_sedimentation_pipeline_.is_ready();
#else
        return false;
#endif
    }

    bool supports_acoustic_pressure_dispatch() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return acoustic_pressure_pipeline_.is_ready();
#else
        return false;
#endif
    }

    bool supports_acoustic_momentum_dispatch() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return acoustic_momentum_pipeline_.is_ready();
#else
        return false;
#endif
    }

    bool dispatch_acoustic_pressure(
        const float* u_data, const float* v_data, const float* w_data,
        const float* rho_in, const float* p_in,
        float* rho_out, float* p_out,
        int nr, int nth, int nz,
        float dr_val, float dtheta_val, float dz_val,
        float gamma_val, float dt_small,
        float rho_floor, float p_floor) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!acoustic_pressure_pipeline_.is_ready()) return false;

        AcousticPressurePushConstants pc{};
        pc.nr = nr; pc.nth = nth; pc.nz = nz;
        pc.dr = dr_val; pc.dtheta = dtheta_val; pc.dz_val = dz_val;
        pc.gamma_val = gamma_val; pc.dt_small = dt_small;
        pc.rho_floor = rho_floor; pc.p_floor = p_floor;

        const uint32_t total_points = static_cast<uint32_t>(nr) * uint32_t(nth) * uint32_t(nz);
        const float* inputs[5] = { u_data, v_data, w_data, rho_in, p_in };
        float* outputs[2] = { rho_out, p_out };

        return dispatch_multi_field_kernel(
            acoustic_pressure_pipeline_, &pc,
            inputs, 5, outputs, 2,
            nr, nth, nz, total_points);
#else
        (void)u_data; (void)v_data; (void)w_data;
        (void)rho_in; (void)p_in; (void)rho_out; (void)p_out;
        (void)nr; (void)nth; (void)nz;
        (void)dr_val; (void)dtheta_val; (void)dz_val;
        (void)gamma_val; (void)dt_small;
        (void)rho_floor; (void)p_floor;
        return false;
#endif
    }

    bool dispatch_acoustic_momentum(
        const float* rho_data, const float* p_data,
        const float* u_in, const float* v_in, const float* w_in,
        float* u_out, float* v_out, float* w_out,
        int nr, int nth, int nz,
        float dr_val, float dtheta_val, float dz_val,
        float dt_small,
        float wind_clamp_h, float wind_clamp_v) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!acoustic_momentum_pipeline_.is_ready()) return false;

        AcousticMomentumPushConstants pc{};
        pc.nr = nr; pc.nth = nth; pc.nz = nz;
        pc.dr = dr_val; pc.dtheta = dtheta_val; pc.dz_val = dz_val;
        pc.dt_small = dt_small;
        pc.wind_clamp_h = wind_clamp_h; pc.wind_clamp_v = wind_clamp_v;
        pc.padding = 0.0f;

        const uint32_t total_points = static_cast<uint32_t>(nr) * uint32_t(nth) * uint32_t(nz);
        const float* inputs[5] = { rho_data, p_data, u_in, v_in, w_in };
        float* outputs[3] = { u_out, v_out, w_out };

        return dispatch_multi_field_kernel(
            acoustic_momentum_pipeline_, &pc,
            inputs, 5, outputs, 3,
            nr, nth, nz, total_points);
#else
        (void)rho_data; (void)p_data;
        (void)u_in; (void)v_in; (void)w_in;
        (void)u_out; (void)v_out; (void)w_out;
        (void)nr; (void)nth; (void)nz;
        (void)dr_val; (void)dtheta_val; (void)dz_val;
        (void)dt_small;
        (void)wind_clamp_h; (void)wind_clamp_v;
        return false;
#endif
    }

    bool supports_batched_advection_dispatch() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Requires radial, azimuthal, and diffusion pipelines
        return radial_pipeline_.is_ready() &&
               azimuthal_pipeline_.is_ready() &&
               diffusion_pipeline_.is_ready();
#else
        return false;
#endif
    }

    bool dispatch_advection_batch_pre_vertical(
        const float* scalar_in, float* result_out,
        const float* u_data, const float* v_data,
        int nr, int nth, int nz,
        float dr_val, float dtheta_val, float dt_half) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!supports_batched_advection_dispatch())
        {
            return false;
        }

        const int total_cells = nr * nth * nz;
        const VkDeviceSize field_bytes =
            static_cast<VkDeviceSize>(total_cells) * sizeof(float);
        const uint32_t interior_points =
            static_cast<uint32_t>(nr - 2) * static_cast<uint32_t>(nth) *
            static_cast<uint32_t>(nz - 2);

        // We need 4 buffers: scalar_A (ping), scalar_B (pong), u, v
        // Radial: A→B reads (A, u) writes B
        // Azimuthal: B→A reads (B, v) writes A
        // But we output to result_out, so the final result is in A (the pong
        // buffer for azimuthal). We download A at the end.
        int slots[4];
        if (!acquire_pool_slots(field_bytes, 4, slots))
        {
            log_vulkan_warning("failed to acquire pool slots for batched pre-vertical");
            return false;
        }
        const int slot_A = slots[0];  // scalar ping
        const int slot_B = slots[1];  // scalar pong
        const int slot_u = slots[2];  // radial velocity
        const int slot_v = slots[3];  // azimuthal velocity

        // On unified memory, device buffers are host-visible — read/write directly.
        // On discrete memory, use separate staging buffers for H2D/D2H transfers.
        const bool unified = has_unified_memory_;
        auto host_ptr = [&](int slot) -> void*
        {
            return unified ? buffer_pool_.device(slot).mapped
                           : buffer_pool_.staging(slot).mapped;
        };

        // Upload data (to device buffer on unified, staging buffer on discrete)
        std::memcpy(host_ptr(slot_A), scalar_in, field_bytes);
        std::memcpy(host_ptr(slot_B), scalar_in, field_bytes); // boundaries
        std::memcpy(host_ptr(slot_u), u_data, field_bytes);
        std::memcpy(host_ptr(slot_v), v_data, field_bytes);

        // Record command buffer
        vkResetCommandBuffer_(cmd_buf_, 0);
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer_(cmd_buf_, &begin_info) != VK_SUCCESS)
        {
            release_pool_slots(slots, 4);
            return false;
        }

        VkBufferCopy copy_region{};
        copy_region.size = field_bytes;

        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;

        if (unified)
        {
            // Unified path: host writes are coherent; barrier ensures GPU visibility
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
        else
        {
            // Discrete path: copy staging → device, then transfer barrier
            for (int i = 0; i < 4; ++i)
            {
                vkCmdCopyBuffer_(cmd_buf_,
                                 buffer_pool_.staging(slots[i]).buffer,
                                 buffer_pool_.device(slots[i]).buffer,
                                 1, &copy_region);
            }

            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0, 1, &barrier, 0, nullptr, 0, nullptr);
        }

        // ─── Step 1: Radial advection (A → B) ───
        {
            // Update radial descriptor set: binding 0=A(src), 1=u, 2=B(dst)
            int radial_bindings[3] = {slot_A, slot_u, slot_B};
            update_pooled_descriptor_set(radial_pipeline_, radial_bindings,
                                         field_bytes, 3);

            RadialAdvectionPushConstants pc{};
            pc.nr = nr; pc.nth = nth; pc.nz = nz;
            pc.dr = dr_val; pc.dt = dt_half;
            pc.padding[0] = pc.padding[1] = pc.padding[2] = 0.0f;

            vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE,
                               radial_pipeline_.pipeline);
            vkCmdBindDescriptorSets_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     radial_pipeline_.pipeline_layout, 0, 1,
                                     &radial_pipeline_.descriptor_set, 0, nullptr);
            vkCmdPushConstants_(cmd_buf_, radial_pipeline_.pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT,
                                0, sizeof(pc), &pc);
            vkCmdDispatch_(cmd_buf_, (interior_points + 63u) / 64u, 1, 1);
        }

        // Barrier: compute → compute (radial output → azimuthal input)
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier_(cmd_buf_,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0, 1, &barrier, 0, nullptr, 0, nullptr);

        // Copy B → A boundaries before azimuthal step (ensures boundary data)
        vkCmdCopyBuffer_(cmd_buf_,
                         buffer_pool_.device(slot_B).buffer,
                         buffer_pool_.device(slot_A).buffer,
                         1, &copy_region);

        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier_(cmd_buf_,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0, 1, &barrier, 0, nullptr, 0, nullptr);

        // ─── Step 2: Azimuthal advection (B → A) ───
        {
            int azimuthal_bindings[3] = {slot_B, slot_v, slot_A};
            update_pooled_descriptor_set(azimuthal_pipeline_, azimuthal_bindings,
                                         field_bytes, 3);

            AzimuthalAdvectionPushConstants pc{};
            pc.nr = nr; pc.nth = nth; pc.nz = nz;
            pc.dr = dr_val; pc.dtheta = dtheta_val; pc.dt = dt_half;
            pc.padding[0] = pc.padding[1] = 0.0f;

            vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE,
                               azimuthal_pipeline_.pipeline);
            vkCmdBindDescriptorSets_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     azimuthal_pipeline_.pipeline_layout, 0, 1,
                                     &azimuthal_pipeline_.descriptor_set, 0, nullptr);
            vkCmdPushConstants_(cmd_buf_, azimuthal_pipeline_.pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT,
                                0, sizeof(pc), &pc);
            vkCmdDispatch_(cmd_buf_, (interior_points + 63u) / 64u, 1, 1);
        }

        if (unified)
        {
            // Unified path: barrier so CPU can read GPU writes after fence
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
        else
        {
            // Discrete path: barrier compute → transfer, then copy device → staging
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0, 1, &barrier, 0, nullptr, 0, nullptr);

            vkCmdCopyBuffer_(cmd_buf_,
                             buffer_pool_.device(slot_A).buffer,
                             buffer_pool_.staging(slot_A).buffer,
                             1, &copy_region);
        }

        if (vkEndCommandBuffer_(cmd_buf_) != VK_SUCCESS)
        {
            release_pool_slots(slots, 4);
            return false;
        }

        // Submit and wait
        vkResetFences_(device_, 1, &fence_);
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buf_;
        if (vkQueueSubmit_(compute_queue_, 1, &submit_info, fence_) != VK_SUCCESS)
        {
            release_pool_slots(slots, 4);
            return false;
        }

        constexpr uint64_t kTimeoutNs = 10ULL * 1000000000ULL;
        if (vkWaitForFences_(device_, 1, &fence_, VK_TRUE, kTimeoutNs) != VK_SUCCESS)
        {
            release_pool_slots(slots, 4);
            return false;
        }

        // Download result (from device buffer on unified, staging on discrete)
        std::memcpy(result_out, host_ptr(slot_A), field_bytes);
        release_pool_slots(slots, 4);
        return true;
#else
        (void)scalar_in; (void)result_out;
        (void)u_data; (void)v_data;
        (void)nr; (void)nth; (void)nz;
        (void)dr_val; (void)dtheta_val; (void)dt_half;
        return false;
#endif
    }

    bool dispatch_advection_batch_post_vertical(
        const float* scalar_in, float* result_out,
        const float* u_data, const float* v_data,
        int nr, int nth, int nz,
        float dr_val, float dtheta_val, float dz_val,
        float dt_half, float dt_full, float kappa_val) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!supports_batched_advection_dispatch())
        {
            return false;
        }

        const int total_cells = nr * nth * nz;
        const VkDeviceSize field_bytes =
            static_cast<VkDeviceSize>(total_cells) * sizeof(float);
        const uint32_t interior_points =
            static_cast<uint32_t>(nr - 2) * static_cast<uint32_t>(nth) *
            static_cast<uint32_t>(nz - 2);

        // We need 4 buffers: scalar_A (ping), scalar_B (pong), u, v
        // Azimuthal: A→B
        // Radial: B→A
        // Diffusion: A→B (reuse B as output)
        int slots[4];
        if (!acquire_pool_slots(field_bytes, 4, slots))
        {
            log_vulkan_warning("failed to acquire pool slots for batched post-vertical");
            return false;
        }
        const int slot_A = slots[0];
        const int slot_B = slots[1];
        const int slot_u = slots[2];
        const int slot_v = slots[3];

        // On unified memory, device buffers are host-visible — read/write directly.
        // On discrete memory, use separate staging buffers for H2D/D2H transfers.
        const bool unified = has_unified_memory_;
        auto host_ptr = [&](int slot) -> void*
        {
            return unified ? buffer_pool_.device(slot).mapped
                           : buffer_pool_.staging(slot).mapped;
        };

        // Upload data (to device buffer on unified, staging buffer on discrete)
        std::memcpy(host_ptr(slot_A), scalar_in, field_bytes);
        std::memcpy(host_ptr(slot_B), scalar_in, field_bytes); // boundaries
        std::memcpy(host_ptr(slot_u), u_data, field_bytes);
        std::memcpy(host_ptr(slot_v), v_data, field_bytes);

        // Record command buffer
        vkResetCommandBuffer_(cmd_buf_, 0);
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer_(cmd_buf_, &begin_info) != VK_SUCCESS)
        {
            release_pool_slots(slots, 4);
            return false;
        }

        VkBufferCopy copy_region{};
        copy_region.size = field_bytes;

        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;

        if (unified)
        {
            // Unified path: host writes are coherent; barrier ensures GPU visibility
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
        else
        {
            // Discrete path: copy staging → device, then transfer barrier
            for (int i = 0; i < 4; ++i)
            {
                vkCmdCopyBuffer_(cmd_buf_,
                                 buffer_pool_.staging(slots[i]).buffer,
                                 buffer_pool_.device(slots[i]).buffer,
                                 1, &copy_region);
            }

            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0, 1, &barrier, 0, nullptr, 0, nullptr);
        }

        // ─── Step 1: Azimuthal advection (A → B) ───
        {
            int azimuthal_bindings[3] = {slot_A, slot_v, slot_B};
            update_pooled_descriptor_set(azimuthal_pipeline_, azimuthal_bindings,
                                         field_bytes, 3);

            AzimuthalAdvectionPushConstants pc{};
            pc.nr = nr; pc.nth = nth; pc.nz = nz;
            pc.dr = dr_val; pc.dtheta = dtheta_val; pc.dt = dt_half;
            pc.padding[0] = pc.padding[1] = 0.0f;

            vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE,
                               azimuthal_pipeline_.pipeline);
            vkCmdBindDescriptorSets_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     azimuthal_pipeline_.pipeline_layout, 0, 1,
                                     &azimuthal_pipeline_.descriptor_set, 0, nullptr);
            vkCmdPushConstants_(cmd_buf_, azimuthal_pipeline_.pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT,
                                0, sizeof(pc), &pc);
            vkCmdDispatch_(cmd_buf_, (interior_points + 63u) / 64u, 1, 1);
        }

        // Barrier + boundary copy for radial step
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier_(cmd_buf_,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 1, &barrier, 0, nullptr, 0, nullptr);

        // Copy B → A (boundaries for radial input)
        vkCmdCopyBuffer_(cmd_buf_,
                         buffer_pool_.device(slot_B).buffer,
                         buffer_pool_.device(slot_A).buffer,
                         1, &copy_region);

        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier_(cmd_buf_,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0, 1, &barrier, 0, nullptr, 0, nullptr);

        // ─── Step 2: Radial advection (B → A) ───
        {
            int radial_bindings[3] = {slot_B, slot_u, slot_A};
            update_pooled_descriptor_set(radial_pipeline_, radial_bindings,
                                         field_bytes, 3);

            RadialAdvectionPushConstants pc{};
            pc.nr = nr; pc.nth = nth; pc.nz = nz;
            pc.dr = dr_val; pc.dt = dt_half;
            pc.padding[0] = pc.padding[1] = pc.padding[2] = 0.0f;

            vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE,
                               radial_pipeline_.pipeline);
            vkCmdBindDescriptorSets_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     radial_pipeline_.pipeline_layout, 0, 1,
                                     &radial_pipeline_.descriptor_set, 0, nullptr);
            vkCmdPushConstants_(cmd_buf_, radial_pipeline_.pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT,
                                0, sizeof(pc), &pc);
            vkCmdDispatch_(cmd_buf_, (interior_points + 63u) / 64u, 1, 1);
        }

        // Barrier + boundary copy for diffusion step
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier_(cmd_buf_,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 1, &barrier, 0, nullptr, 0, nullptr);

        // Copy A → B (boundaries for diffusion)
        vkCmdCopyBuffer_(cmd_buf_,
                         buffer_pool_.device(slot_A).buffer,
                         buffer_pool_.device(slot_B).buffer,
                         1, &copy_region);

        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier_(cmd_buf_,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0, 1, &barrier, 0, nullptr, 0, nullptr);

        // ─── Step 3: Diffusion (A → B) ───
        if (kappa_val > 0.0f)
        {
            int diffusion_bindings[2] = {slot_A, slot_B};
            update_pooled_descriptor_set(diffusion_pipeline_, diffusion_bindings,
                                         field_bytes, 2);

            DiffusionPushConstants pc{};
            pc.nr = nr; pc.nth = nth; pc.nz = nz;
            pc.dr = dr_val; pc.dtheta = dtheta_val; pc.dz = dz_val;
            pc.dt = dt_full; pc.kappa = kappa_val;

            vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE,
                               diffusion_pipeline_.pipeline);
            vkCmdBindDescriptorSets_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     diffusion_pipeline_.pipeline_layout, 0, 1,
                                     &diffusion_pipeline_.descriptor_set, 0, nullptr);
            vkCmdPushConstants_(cmd_buf_, diffusion_pipeline_.pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT,
                                0, sizeof(pc), &pc);
            vkCmdDispatch_(cmd_buf_, (interior_points + 63u) / 64u, 1, 1);
        }
        else
        {
            // No diffusion — result stays in A. Copy A → B so download path
            // is consistent (always downloads from B when kappa>0, A otherwise).
            // Actually, just swap which slot we download from below.
        }

        // The final result is in B (if diffusion ran) or A (if kappa<=0)
        const int result_slot = (kappa_val > 0.0f) ? slot_B : slot_A;

        if (unified)
        {
            // Unified path: barrier so CPU can read GPU writes after fence
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
        else
        {
            // Discrete path: barrier compute → transfer, then copy device → staging
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0, 1, &barrier, 0, nullptr, 0, nullptr);

            vkCmdCopyBuffer_(cmd_buf_,
                             buffer_pool_.device(result_slot).buffer,
                             buffer_pool_.staging(result_slot).buffer,
                             1, &copy_region);
        }

        if (vkEndCommandBuffer_(cmd_buf_) != VK_SUCCESS)
        {
            release_pool_slots(slots, 4);
            return false;
        }

        // Submit and wait
        vkResetFences_(device_, 1, &fence_);
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buf_;
        if (vkQueueSubmit_(compute_queue_, 1, &submit_info, fence_) != VK_SUCCESS)
        {
            release_pool_slots(slots, 4);
            return false;
        }

        constexpr uint64_t kTimeoutNs = 10ULL * 1000000000ULL;
        if (vkWaitForFences_(device_, 1, &fence_, VK_TRUE, kTimeoutNs) != VK_SUCCESS)
        {
            release_pool_slots(slots, 4);
            return false;
        }

        // Download result (from device buffer on unified, staging on discrete)
        std::memcpy(result_out, host_ptr(result_slot), field_bytes);
        release_pool_slots(slots, 4);
        return true;
#else
        (void)scalar_in; (void)result_out;
        (void)u_data; (void)v_data;
        (void)nr; (void)nth; (void)nz;
        (void)dr_val; (void)dtheta_val; (void)dz_val;
        (void)dt_half; (void)dt_full; (void)kappa_val;
        return false;
#endif
    }

    bool dispatch_vertical_flux(
        const float* q_data, const float* w_data, float* dqdt_data,
        int nr, int nth, int nz,
        const double* dz_data, int dz_count,
        int limiter_id, bool positivity, double positivity_dt,
        double cfl_target,
        VerticalFluxDispatchResult& out_result) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!tvd_pipeline_.is_ready())
        {
            return false;
        }

        const int total_cells = nr * nth * nz;
        const int total_columns = nr * nth;
        const VkDeviceSize field_bytes = static_cast<VkDeviceSize>(total_cells) * sizeof(float);
        const VkDeviceSize dz_bytes = static_cast<VkDeviceSize>(nz) * sizeof(float);

        // Ensure buffers are large enough — reallocate if dimensions changed
        if (!ensure_buffers(field_bytes, dz_bytes))
        {
            log_vulkan_warning("failed to allocate GPU buffers for dispatch");
            return false;
        }

        // Upload q to device
        std::memcpy(staging_q_.mapped, q_data, field_bytes);
        // Upload w to device
        std::memcpy(staging_w_.mapped, w_data, field_bytes);
        // Upload dz (convert from double to float)
        {
            auto* dz_staging = static_cast<float*>(staging_dz_.mapped);
            const int count = std::min(dz_count, nz);
            for (int k = 0; k < count; ++k)
            {
                dz_staging[k] = static_cast<float>(dz_data[k]);
            }
        }
        // Zero the output staging buffer
        std::memset(staging_dqdt_.mapped, 0, field_bytes);

        // Record and submit command buffer
        vkResetCommandBuffer_(cmd_buf_, 0);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer_(cmd_buf_, &begin_info) != VK_SUCCESS)
        {
            log_vulkan_warning("vkBeginCommandBuffer failed");
            return false;
        }

        // Copy staging → device buffers
        VkBufferCopy copy_region{};
        copy_region.size = field_bytes;
        vkCmdCopyBuffer_(cmd_buf_, staging_q_.buffer, device_q_.buffer, 1, &copy_region);
        vkCmdCopyBuffer_(cmd_buf_, staging_w_.buffer, device_w_.buffer, 1, &copy_region);
        copy_region.size = dz_bytes;
        vkCmdCopyBuffer_(cmd_buf_, staging_dz_.buffer, device_dz_.buffer, 1, &copy_region);
        copy_region.size = field_bytes;
        vkCmdCopyBuffer_(cmd_buf_, staging_dqdt_.buffer, device_dqdt_.buffer, 1, &copy_region);

        // Barrier: transfer → compute
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier_(cmd_buf_,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0, 1, &barrier, 0, nullptr, 0, nullptr);

        // Bind pipeline and descriptor set
        vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, tvd_pipeline_.pipeline);
        vkCmdBindDescriptorSets_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 tvd_pipeline_.pipeline_layout, 0, 1,
                                 &tvd_pipeline_.descriptor_set, 0, nullptr);

        // Push constants
        TvdPushConstants pc{};
        pc.nr = nr;
        pc.nth = nth;
        pc.nz = nz;
        pc.limiter_id = limiter_id;
        pc.positivity = positivity ? 1 : 0;
        pc.positivity_dt = static_cast<float>(positivity_dt);
        pc.cfl_target = static_cast<float>(cfl_target);
        pc.padding = 0.0f;
        vkCmdPushConstants_(cmd_buf_, tvd_pipeline_.pipeline_layout,
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(TvdPushConstants), &pc);

        // Dispatch — one invocation per column, 64 per workgroup
        const uint32_t workgroup_count = (static_cast<uint32_t>(total_columns) + 63u) / 64u;
        vkCmdDispatch_(cmd_buf_, workgroup_count, 1, 1);

        // Barrier: compute → transfer
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier_(cmd_buf_,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 1, &barrier, 0, nullptr, 0, nullptr);

        // Copy device dqdt → staging
        copy_region.size = field_bytes;
        vkCmdCopyBuffer_(cmd_buf_, device_dqdt_.buffer, staging_dqdt_.buffer, 1, &copy_region);

        if (vkEndCommandBuffer_(cmd_buf_) != VK_SUCCESS)
        {
            log_vulkan_warning("vkEndCommandBuffer failed");
            return false;
        }

        // Submit and wait
        vkResetFences_(device_, 1, &fence_);

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buf_;
        if (vkQueueSubmit_(compute_queue_, 1, &submit_info, fence_) != VK_SUCCESS)
        {
            log_vulkan_warning("vkQueueSubmit failed");
            return false;
        }

        constexpr uint64_t kTimeoutNs = 10ULL * 1000000000ULL; // 10 seconds
        VkResult wait_result = vkWaitForFences_(device_, 1, &fence_, VK_TRUE, kTimeoutNs);
        if (wait_result != VK_SUCCESS)
        {
            log_vulkan_warning("vkWaitForFences failed or timed out (result=" +
                              std::to_string(static_cast<int>(wait_result)) + ")");
            return false;
        }

        // Download results
        std::memcpy(dqdt_data, staging_dqdt_.mapped, field_bytes);

        // CFL reduction on CPU (avoids a second shader pass)
        double max_cfl = 0.0;
        for (int idx = 0; idx < total_cells; ++idx)
        {
            const int k = idx % nz;
            const double w_val = static_cast<double>(w_data[idx]);
            const double dz_val = (k < dz_count) ? dz_data[k] : 1.0;
            if (dz_val > 0.0)
            {
                const double local_cfl = std::abs(w_val) / dz_val;
                if (local_cfl > max_cfl)
                {
                    max_cfl = local_cfl;
                }
            }
        }
        out_result.max_cfl_z = max_cfl;
        out_result.suggested_dt = (max_cfl > 1.0e-30) ? (cfl_target / max_cfl) : 1.0e30;

        return true;
#else
        (void)q_data; (void)w_data; (void)dqdt_data;
        (void)nr; (void)nth; (void)nz;
        (void)dz_data; (void)dz_count;
        (void)limiter_id; (void)positivity; (void)positivity_dt;
        (void)cfl_target; (void)out_result;
        return false;
#endif
    }

    bool dispatch_radial_advection(
        const float* src, const float* u_data, float* dst,
        int nr, int nth, int nz,
        float dr_val, float dt_val) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!radial_pipeline_.is_ready())
        {
            return false;
        }

        RadialAdvectionPushConstants pc{};
        pc.nr = nr;
        pc.nth = nth;
        pc.nz = nz;
        pc.dr = dr_val;
        pc.dt = dt_val;
        pc.padding[0] = pc.padding[1] = pc.padding[2] = 0.0f;

        const uint32_t interior_points =
            static_cast<uint32_t>(nr - 2) * static_cast<uint32_t>(nth) *
            static_cast<uint32_t>(nz - 2);

        return dispatch_field3_kernel(
            radial_pipeline_, &pc,
            src, u_data, dst,
            nr, nth, nz,
            2,  // 2 inputs (src, u)
            interior_points);
#else
        (void)src; (void)u_data; (void)dst;
        (void)nr; (void)nth; (void)nz;
        (void)dr_val; (void)dt_val;
        return false;
#endif
    }

    bool dispatch_azimuthal_advection(
        const float* src, const float* v_data, float* dst,
        int nr, int nth, int nz,
        float dr_val, float dtheta_val, float dt_val) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!azimuthal_pipeline_.is_ready())
        {
            return false;
        }

        AzimuthalAdvectionPushConstants pc{};
        pc.nr = nr;
        pc.nth = nth;
        pc.nz = nz;
        pc.dr = dr_val;
        pc.dtheta = dtheta_val;
        pc.dt = dt_val;
        pc.padding[0] = pc.padding[1] = 0.0f;

        const uint32_t interior_points =
            static_cast<uint32_t>(nr - 2) * static_cast<uint32_t>(nth) *
            static_cast<uint32_t>(nz - 2);

        return dispatch_field3_kernel(
            azimuthal_pipeline_, &pc,
            src, v_data, dst,
            nr, nth, nz,
            2,
            interior_points);
#else
        (void)src; (void)v_data; (void)dst;
        (void)nr; (void)nth; (void)nz;
        (void)dr_val; (void)dtheta_val; (void)dt_val;
        return false;
#endif
    }

    bool dispatch_diffusion(
        const float* src, float* dst,
        int nr, int nth, int nz,
        float dr_val, float dtheta_val, float dz_val,
        float dt_val, float kappa_val) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!diffusion_pipeline_.is_ready())
        {
            return false;
        }

        DiffusionPushConstants pc{};
        pc.nr = nr;
        pc.nth = nth;
        pc.nz = nz;
        pc.dr = dr_val;
        pc.dtheta = dtheta_val;
        pc.dz = dz_val;
        pc.dt = dt_val;
        pc.kappa = kappa_val;

        const uint32_t interior_points =
            static_cast<uint32_t>(nr - 2) * static_cast<uint32_t>(nth) *
            static_cast<uint32_t>(nz - 2);

        // Diffusion has only 2 bindings (src, dst) — pass nullptr for input_b
        return dispatch_field3_kernel(
            diffusion_pipeline_, &pc,
            src, nullptr, dst,
            nr, nth, nz,
            1,  // 1 input (src only)
            interior_points);
#else
        (void)src; (void)dst;
        (void)nr; (void)nth; (void)nz;
        (void)dr_val; (void)dtheta_val; (void)dz_val;
        (void)dt_val; (void)kappa_val;
        return false;
#endif
    }

    bool dispatch_supercell_tendencies(
        const float* u_r_data, const float* u_theta_data, const float* u_z_data,
        const float* rho_data, const float* p_data, const float* theta_data,
        const float* loading_data,
        float* du_r_dt_data, float* du_theta_dt_data, float* du_z_dt_data,
        float* drho_dt_data, float* dp_dt_data,
        int nr, int nth, int nz,
        float dr_val, float dtheta_val, float dz_val,
        float g_val, float gamma_val, float theta0_val) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!supercell_pipeline_.is_ready())
        {
            return false;
        }

        SupercellTendenciesPushConstants pc{};
        pc.nr = nr;
        pc.nth = nth;
        pc.nz = nz;
        pc.dr = dr_val;
        pc.dtheta = dtheta_val;
        pc.dz = dz_val;
        pc.g = g_val;
        pc.gamma_val = gamma_val;
        pc.theta0 = theta0_val;
        pc.padding = 0.0f;

        const uint32_t interior_points =
            static_cast<uint32_t>(nr - 2) * static_cast<uint32_t>(nth) *
            static_cast<uint32_t>(nz - 2);

        const float* inputs[7] = {
            u_r_data, u_theta_data, u_z_data,
            rho_data, p_data, theta_data,
            loading_data
        };
        float* outputs[5] = {
            du_r_dt_data, du_theta_dt_data, du_z_dt_data,
            drho_dt_data, dp_dt_data
        };

        return dispatch_multi_field_kernel(
            supercell_pipeline_, &pc,
            inputs, 7,
            outputs, 5,
            nr, nth, nz,
            interior_points);
#else
        (void)u_r_data; (void)u_theta_data; (void)u_z_data;
        (void)rho_data; (void)p_data; (void)theta_data;
        (void)loading_data;
        (void)du_r_dt_data; (void)du_theta_dt_data; (void)du_z_dt_data;
        (void)drho_dt_data; (void)dp_dt_data;
        (void)nr; (void)nth; (void)nz;
        (void)dr_val; (void)dtheta_val; (void)dz_val;
        (void)g_val; (void)gamma_val; (void)theta0_val;
        return false;
#endif
    }

    // ── Cartesian tendencies (8 inputs including 1D profiles, 5 outputs) ──

    bool dispatch_cartesian_tendencies(
        const float* u_x_data, const float* u_y_data, const float* w_data,
        const float* rho_data, const float* p_data, const float* theta_data,
        const float* p0_base_data, const float* rho0_base_data,
        const float* loading_data,
        const float* u0_base_data, const float* v0_base_data,
        float* du_x_dt_data, float* du_y_dt_data, float* dw_dt_data,
        float* drho_dt_data, float* dp_dt_data,
        int nr, int nth, int nz,
        float dx_val, float dy_val, float dz_val,
        float g_val, float gamma_val, float coriolis_f_val) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!cartesian_pipeline_.is_ready())
        {
            return false;
        }

        CartesianTendenciesPushConstants pc{};
        pc.nr = nr;
        pc.nth = nth;
        pc.nz = nz;
        pc.dx = dx_val;
        pc.dy = dy_val;
        pc.dz = dz_val;
        pc.g = g_val;
        pc.gamma_val = gamma_val;
        pc.coriolis_f = coriolis_f_val;
        pc.padding = 0.0f;

        // Cartesian interior: skip all 6 boundary faces
        const uint32_t interior_points =
            static_cast<uint32_t>(nr - 2) * static_cast<uint32_t>(nth - 2) *
            static_cast<uint32_t>(nz - 2);

        // The 1D profiles (p0_base, rho0_base, u0_base, v0_base) are passed
        // as full-size buffers to fit dispatch_multi_field_kernel's uniform-
        // size requirement. The shader only reads indices [0..nz-1].
        const size_t total_cells = static_cast<size_t>(nr) * nth * nz;
        std::vector<float> p0_padded(total_cells, 0.0f);
        std::vector<float> rho0_padded(total_cells, 0.0f);
        std::vector<float> u0_padded(total_cells, 0.0f);
        std::vector<float> v0_padded(total_cells, 0.0f);
        for (int k = 0; k < nz; ++k)
        {
            p0_padded[k] = p0_base_data[k];
            rho0_padded[k] = rho0_base_data[k];
            u0_padded[k] = u0_base_data[k];
            v0_padded[k] = v0_base_data[k];
        }

        const float* inputs[11] = {
            u_x_data, u_y_data, w_data,
            rho_data, p_data, theta_data,
            p0_padded.data(), rho0_padded.data(),
            loading_data,
            u0_padded.data(), v0_padded.data()
        };
        float* outputs[5] = {
            du_x_dt_data, du_y_dt_data, dw_dt_data,
            drho_dt_data, dp_dt_data
        };

        return dispatch_multi_field_kernel(
            cartesian_pipeline_, &pc,
            inputs, 11,
            outputs, 5,
            nr, nth, nz,
            interior_points);
#else
        (void)u_x_data; (void)u_y_data; (void)w_data;
        (void)rho_data; (void)p_data; (void)theta_data;
        (void)p0_base_data; (void)rho0_base_data;
        (void)loading_data;
        (void)u0_base_data; (void)v0_base_data;
        (void)du_x_dt_data; (void)du_y_dt_data; (void)dw_dt_data;
        (void)drho_dt_data; (void)dp_dt_data;
        (void)nr; (void)nth; (void)nz;
        (void)dx_val; (void)dy_val; (void)dz_val;
        (void)g_val; (void)gamma_val; (void)coriolis_f_val;
        return false;
#endif
    }

    // ── Cartesian x-advection ──

    bool dispatch_advection_x(
        const float* src, const float* u_data, float* dst,
        int nr, int nth, int nz,
        float dx_val, float dt_val) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!advection_x_pipeline_.is_ready())
        {
            return false;
        }

        AdvectionXPushConstants pc{};
        pc.nr = nr;
        pc.nth = nth;
        pc.nz = nz;
        pc.dx = dx_val;
        pc.dt = dt_val;
        pc.padding[0] = pc.padding[1] = pc.padding[2] = 0.0f;

        // x-advection interior: i=1..nr-2, all j, k=1..nz-2
        const uint32_t interior_points =
            static_cast<uint32_t>(nr - 2) * static_cast<uint32_t>(nth) *
            static_cast<uint32_t>(nz - 2);

        return dispatch_field3_kernel(
            advection_x_pipeline_, &pc,
            src, u_data, dst,
            nr, nth, nz,
            2,
            interior_points);
#else
        (void)src; (void)u_data; (void)dst;
        (void)nr; (void)nth; (void)nz;
        (void)dx_val; (void)dt_val;
        return false;
#endif
    }

    // ── Cartesian y-advection ──

    bool dispatch_advection_y(
        const float* src, const float* v_data, float* dst,
        int nr, int nth, int nz,
        float dy_val, float dt_val) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!advection_y_pipeline_.is_ready())
        {
            return false;
        }

        AdvectionYPushConstants pc{};
        pc.nr = nr;
        pc.nth = nth;
        pc.nz = nz;
        pc.dy = dy_val;
        pc.dt = dt_val;
        pc.padding[0] = pc.padding[1] = pc.padding[2] = 0.0f;

        // y-advection interior: all i, j=1..nth-2, k=1..nz-2
        const uint32_t interior_points =
            static_cast<uint32_t>(nr) * static_cast<uint32_t>(nth - 2) *
            static_cast<uint32_t>(nz - 2);

        return dispatch_field3_kernel(
            advection_y_pipeline_, &pc,
            src, v_data, dst,
            nr, nth, nz,
            2,
            interior_points);
#else
        (void)src; (void)v_data; (void)dst;
        (void)nr; (void)nth; (void)nz;
        (void)dy_val; (void)dt_val;
        return false;
#endif
    }

    bool dispatch_tornado_tendencies(
        const float* u_r_data, const float* u_theta_data, const float* u_z_data,
        const float* rho_data, const float* p_data, const float* theta_data,
        const float* loading_data,
        float* du_r_dt_data, float* du_theta_dt_data, float* du_z_dt_data,
        float* drho_dt_data, float* dp_dt_data,
        int nr, int nth, int nz,
        float dr_val, float dz_val,
        float g_val, float theta0_val, float eps_val, float friction_val) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!tornado_pipeline_.is_ready())
        {
            return false;
        }

        TornadoTendenciesPushConstants pc{};
        pc.nr = nr;
        pc.nth = nth;
        pc.nz = nz;
        pc.dr = dr_val;
        pc.dz = dz_val;
        pc.g = g_val;
        pc.theta0 = theta0_val;
        pc.eps = eps_val;
        pc.friction_coeff = friction_val;
        pc.padding = 0.0f;

        const uint32_t interior_points =
            static_cast<uint32_t>(nr - 2) * static_cast<uint32_t>(nth) *
            static_cast<uint32_t>(nz - 2);

        const float* inputs[7] = {
            u_r_data, u_theta_data, u_z_data,
            rho_data, p_data, theta_data,
            loading_data
        };
        float* outputs[5] = {
            du_r_dt_data, du_theta_dt_data, du_z_dt_data,
            drho_dt_data, dp_dt_data
        };

        return dispatch_multi_field_kernel(
            tornado_pipeline_, &pc,
            inputs, 7,
            outputs, 5,
            nr, nth, nz,
            interior_points);
#else
        (void)u_r_data; (void)u_theta_data; (void)u_z_data;
        (void)rho_data; (void)p_data; (void)theta_data;
        (void)loading_data;
        (void)du_r_dt_data; (void)du_theta_dt_data; (void)du_z_dt_data;
        (void)drho_dt_data; (void)dp_dt_data;
        (void)nr; (void)nth; (void)nz;
        (void)dr_val; (void)dz_val;
        (void)g_val; (void)theta0_val; (void)eps_val; (void)friction_val;
        return false;
#endif
    }

    bool dispatch_kessler_pointwise(
        const float* temperature_data, const float* p_data,
        const float* qv_data,
        const float* qc_data, const float* qr_data,
        const float* qg_data, const float* qh_data,
        float* dtheta_dt_data, float* dqv_dt_data,
        float* dqc_dt_data, float* dqr_dt_data,
        float* dqg_dt_data, float* dqh_dt_data,
        int nr, int nth, int nz,
        float qc0, float c_auto_val, float c_accr_val, float c_evap_val,
        float c_freeze_val, float c_rime_val, float c_melt_val, float c_subl_val,
        float Lv_cp, float Lf_cp, float Ls_cp, float T0_val) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!kessler_pointwise_pipeline_.is_ready())
        {
            return false;
        }

        KesslerPointwisePushConstants pc{};
        pc.nr = nr;
        pc.nth = nth;
        pc.nz = nz;
        pc.qc0 = qc0;
        pc.c_auto = c_auto_val;
        pc.c_accr = c_accr_val;
        pc.c_evap = c_evap_val;
        pc.c_freeze = c_freeze_val;
        pc.c_rime = c_rime_val;
        pc.c_melt = c_melt_val;
        pc.c_subl = c_subl_val;
        pc.Lv_cp = Lv_cp;
        pc.Lf_cp = Lf_cp;
        pc.Ls_cp = Ls_cp;
        pc.T0 = T0_val;
        pc.padding = 0.0f;

        const uint32_t total_cells =
            static_cast<uint32_t>(nr) * static_cast<uint32_t>(nth) *
            static_cast<uint32_t>(nz);

        const float* inputs[7] = {
            temperature_data, p_data,
            qv_data, qc_data, qr_data, qg_data, qh_data
        };
        float* outputs[6] = {
            dtheta_dt_data, dqv_dt_data, dqc_dt_data,
            dqr_dt_data, dqg_dt_data, dqh_dt_data
        };

        return dispatch_multi_field_kernel(
            kessler_pointwise_pipeline_, &pc,
            inputs, 7,
            outputs, 6,
            nr, nth, nz,
            total_cells);
#else
        (void)temperature_data; (void)p_data; (void)qv_data;
        (void)qc_data; (void)qr_data;
        (void)qg_data; (void)qh_data;
        (void)dtheta_dt_data; (void)dqv_dt_data;
        (void)dqc_dt_data; (void)dqr_dt_data;
        (void)dqg_dt_data; (void)dqh_dt_data;
        (void)nr; (void)nth; (void)nz;
        (void)qc0; (void)c_auto_val; (void)c_accr_val; (void)c_evap_val;
        (void)c_freeze_val; (void)c_rime_val; (void)c_melt_val; (void)c_subl_val;
        (void)Lv_cp; (void)Lf_cp; (void)Ls_cp; (void)T0_val;
        return false;
#endif
    }

    bool dispatch_kessler_sedimentation(
        const float* qr_data, const float* qg_data, const float* qh_data,
        float* dqr_dt_data, float* dqg_dt_data, float* dqh_dt_data,
        int nr, int nth, int nz,
        float dz_val,
        float a_rain, float b_rain, float Vt_max_rain,
        float a_grau, float b_grau, float Vt_max_grau,
        float a_hail, float b_hail, float Vt_max_hail) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!kessler_sedimentation_pipeline_.is_ready())
        {
            return false;
        }

        KesslerSedimentationPushConstants pc{};
        pc.nr = nr;
        pc.nth = nth;
        pc.nz = nz;
        pc.dz_val = dz_val;
        pc.a_rain = a_rain;
        pc.b_rain = b_rain;
        pc.Vt_max_rain = Vt_max_rain;
        pc.a_grau = a_grau;
        pc.b_grau = b_grau;
        pc.Vt_max_grau = Vt_max_grau;
        pc.a_hail = a_hail;
        pc.b_hail = b_hail;
        pc.Vt_max_hail = Vt_max_hail;
        pc.padding[0] = pc.padding[1] = pc.padding[2] = 0.0f;

        const uint32_t total_columns =
            static_cast<uint32_t>(nr) * static_cast<uint32_t>(nth);

        const float* inputs[3] = { qr_data, qg_data, qh_data };
        float* outputs[3] = { dqr_dt_data, dqg_dt_data, dqh_dt_data };

        return dispatch_multi_field_kernel(
            kessler_sedimentation_pipeline_, &pc,
            inputs, 3,
            outputs, 3,
            nr, nth, nz,
            total_columns);
#else
        (void)qr_data; (void)qg_data; (void)qh_data;
        (void)dqr_dt_data; (void)dqg_dt_data; (void)dqh_dt_data;
        (void)nr; (void)nth; (void)nz;
        (void)dz_val;
        (void)a_rain; (void)b_rain; (void)Vt_max_rain;
        (void)a_grau; (void)b_grau; (void)Vt_max_grau;
        (void)a_hail; (void)b_hail; (void)Vt_max_hail;
        return false;
#endif
    }

    void shutdown() override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (device_ != VK_NULL_HANDLE && vkDeviceWaitIdle_ != nullptr)
        {
            vkDeviceWaitIdle_(device_);
        }

        // Destroy compute pipeline resources
        destroy_compute_resources();

        if (device_ != VK_NULL_HANDLE && vkDestroyDevice_ != nullptr)
        {
            vkDestroyDevice_(device_, nullptr);
        }

        if (instance_ != VK_NULL_HANDLE && vkDestroyInstance_ != nullptr)
        {
            vkDestroyInstance_(instance_, nullptr);
        }

        if (loader_handle_ != nullptr)
        {
            dlclose(loader_handle_);
        }

        reset_state();
#endif
    }

private:
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)

    // ── Push constants layouts (must match respective shaders) ─────────
    struct TvdPushConstants
    {
        int32_t nr;
        int32_t nth;
        int32_t nz;
        int32_t limiter_id;
        int32_t positivity;
        float   positivity_dt;
        float   cfl_target;
        float   padding;
    };
    static_assert(sizeof(TvdPushConstants) == 32, "TVD push constants must be 32 bytes");

    struct RadialAdvectionPushConstants
    {
        int32_t nr;
        int32_t nth;
        int32_t nz;
        float   dr;
        float   dt;
        float   padding[3];
    };
    static_assert(sizeof(RadialAdvectionPushConstants) == 32, "radial push constants must be 32 bytes");

    struct AzimuthalAdvectionPushConstants
    {
        int32_t nr;
        int32_t nth;
        int32_t nz;
        float   dr;
        float   dtheta;
        float   dt;
        float   padding[2];
    };
    static_assert(sizeof(AzimuthalAdvectionPushConstants) == 32, "azimuthal push constants must be 32 bytes");

    struct DiffusionPushConstants
    {
        int32_t nr;
        int32_t nth;
        int32_t nz;
        float   dr;
        float   dtheta;
        float   dz;
        float   dt;
        float   kappa;
    };

    struct SupercellTendenciesPushConstants
    {
        int32_t nr;
        int32_t nth;
        int32_t nz;
        float   dr;
        float   dtheta;
        float   dz;
        float   g;
        float   gamma_val;
        float   theta0;
        float   padding;
    };
    static_assert(sizeof(SupercellTendenciesPushConstants) == 40,
                  "supercell push constants must be 40 bytes");

    struct CartesianTendenciesPushConstants
    {
        int32_t nr;
        int32_t nth;
        int32_t nz;
        float   dx;
        float   dy;
        float   dz;
        float   g;
        float   gamma_val;
        float   coriolis_f;
        float   padding;
    };
    static_assert(sizeof(CartesianTendenciesPushConstants) == 40,
                  "cartesian push constants must be 40 bytes");

    struct AdvectionXPushConstants
    {
        int32_t nr;
        int32_t nth;
        int32_t nz;
        float   dx;
        float   dt;
        float   padding[3];
    };
    static_assert(sizeof(AdvectionXPushConstants) == 32, "advection-x push constants must be 32 bytes");

    struct AdvectionYPushConstants
    {
        int32_t nr;
        int32_t nth;
        int32_t nz;
        float   dy;
        float   dt;
        float   padding[3];
    };
    static_assert(sizeof(AdvectionYPushConstants) == 32, "advection-y push constants must be 32 bytes");

    struct TornadoTendenciesPushConstants
    {
        int32_t nr;
        int32_t nth;
        int32_t nz;
        float   dr;
        float   dz;
        float   g;
        float   theta0;
        float   eps;
        float   friction_coeff;
        float   padding;
    };
    static_assert(sizeof(TornadoTendenciesPushConstants) == 40,
                  "tornado push constants must be 40 bytes");

    static_assert(sizeof(DiffusionPushConstants) == 32, "diffusion push constants must be 32 bytes");

    struct KesslerPointwisePushConstants
    {
        int32_t nr;
        int32_t nth;
        int32_t nz;
        float   qc0;
        float   c_auto;
        float   c_accr;
        float   c_evap;
        float   c_freeze;
        float   c_rime;
        float   c_melt;
        float   c_subl;
        float   Lv_cp;
        float   Lf_cp;
        float   Ls_cp;
        float   T0;
        float   padding;
    };
    static_assert(sizeof(KesslerPointwisePushConstants) == 64,
                  "kessler pointwise push constants must be 64 bytes");

    struct KesslerSedimentationPushConstants
    {
        int32_t nr;
        int32_t nth;
        int32_t nz;
        float   dz_val;
        float   a_rain;
        float   b_rain;
        float   Vt_max_rain;
        float   a_grau;
        float   b_grau;
        float   Vt_max_grau;
        float   a_hail;
        float   b_hail;
        float   Vt_max_hail;
        float   padding[3];
    };
    static_assert(sizeof(KesslerSedimentationPushConstants) == 64,
                  "kessler sedimentation push constants must be 64 bytes");

    struct AcousticPressurePushConstants
    {
        int32_t nr;
        int32_t nth;
        int32_t nz;
        float   dr;
        float   dtheta;
        float   dz_val;
        float   gamma_val;
        float   dt_small;
        float   rho_floor;
        float   p_floor;
    };
    static_assert(sizeof(AcousticPressurePushConstants) == 40,
                  "acoustic pressure push constants must be 40 bytes");

    struct AcousticMomentumPushConstants
    {
        int32_t nr;
        int32_t nth;
        int32_t nz;
        float   dr;
        float   dtheta;
        float   dz_val;
        float   dt_small;
        float   wind_clamp_h;
        float   wind_clamp_v;
        float   padding;
    };
    static_assert(sizeof(AcousticMomentumPushConstants) == 40,
                  "acoustic momentum push constants must be 40 bytes");

    // ── GPU buffer wrapper (aliased from pool header) ──────────────────
    using GpuBuffer = tmv_vulkan::GpuBuffer;

    // ── Reusable compute pipeline state ────────────────────────────────
    struct ComputePipelineState
    {
        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
        VkPipelineLayout      pipeline_layout = VK_NULL_HANDLE;
        VkPipeline            pipeline = VK_NULL_HANDLE;
        VkDescriptorPool      descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet       descriptor_set = VK_NULL_HANDLE;
        uint32_t              binding_count = 0;
        uint32_t              push_constant_size = 0;

        bool is_ready() const { return pipeline != VK_NULL_HANDLE; }
    };

    struct PhysicalDeviceCandidate
    {
        std::size_t enumeration_index = 0;
        VkPhysicalDevice handle = VK_NULL_HANDLE;
        VkPhysicalDeviceProperties properties{};
        VkPhysicalDeviceFeatures features{};
        VkQueueFamilyProperties compute_queue_family{};
        uint32_t compute_queue_family_index = std::numeric_limits<uint32_t>::max();
        int score = -1;
        bool suitable = false;
        bool has_compute_queue = false;
        bool limits_usable = false;
        bool supports_portability_subset = false;
        std::string unsuitable_reason;
    };

    template <typename ProcType>
    static bool load_global_proc(PFN_vkGetInstanceProcAddr get_instance_proc_addr,
                                 const char* name,
                                 ProcType& out_proc)
    {
        if (get_instance_proc_addr == nullptr)
        {
            out_proc = nullptr;
            return false;
        }
        out_proc = reinterpret_cast<ProcType>(get_instance_proc_addr(VK_NULL_HANDLE, name));
        return out_proc != nullptr;
    }

    template <typename ProcType>
    static bool load_instance_proc(PFN_vkGetInstanceProcAddr get_instance_proc_addr,
                                   VkInstance instance,
                                   const char* name,
                                   ProcType& out_proc)
    {
        if (get_instance_proc_addr == nullptr || instance == VK_NULL_HANDLE)
        {
            out_proc = nullptr;
            return false;
        }
        out_proc = reinterpret_cast<ProcType>(get_instance_proc_addr(instance, name));
        return out_proc != nullptr;
    }

    template <typename ProcType>
    static bool load_device_proc(PFN_vkGetDeviceProcAddr get_device_proc_addr,
                                 VkDevice device,
                                 const char* name,
                                 ProcType& out_proc)
    {
        if (get_device_proc_addr == nullptr || device == VK_NULL_HANDLE)
        {
            out_proc = nullptr;
            return false;
        }
        out_proc = reinterpret_cast<ProcType>(get_device_proc_addr(device, name));
        return out_proc != nullptr;
    }

    static std::string version_to_string(uint32_t version)
    {
        return std::to_string(VK_VERSION_MAJOR(version)) + "." +
            std::to_string(VK_VERSION_MINOR(version)) + "." +
            std::to_string(VK_VERSION_PATCH(version));
    }

    static const char* device_type_name(VkPhysicalDeviceType type)
    {
        switch (type)
        {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                return "discrete";
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                return "integrated";
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                return "virtual";
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                return "cpu";
            default:
                return "other";
        }
    }

    static int device_type_score(VkPhysicalDeviceType type)
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

    static std::string join_extension_names(const std::vector<const char*>& names)
    {
        std::ostringstream stream;
        for (std::size_t i = 0; i < names.size(); ++i)
        {
            if (i > 0)
            {
                stream << ", ";
            }
            stream << names[i];
        }
        return stream.str();
    }

    static std::string vk_result_string(VkResult result)
    {
        return std::to_string(static_cast<int>(result));
    }

    static std::string instance_failure_hint(VkResult result)
    {
#if defined(__APPLE__)
        if (result == VK_ERROR_INCOMPATIBLE_DRIVER)
        {
            return
#if defined(__APPLE__)
                " (hint: ensure Homebrew packages `vulkan-loader` and `molten-vk` are installed; "
#elif defined(__linux__)
                " (hint: install libvulkan-dev and vulkan-tools, e.g. `sudo apt install libvulkan-dev vulkan-tools`; "
#else
                " (hint: ensure Vulkan SDK is installed; "
#endif
                "if needed set VK_ICD_FILENAMES to /opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json)";
        }
#endif
        return "";
    }

    static void log_vulkan_info(const std::string& message)
    {
        if (log_normal_enabled())
        {
            std::cout << "[COMPUTE][VULKAN] " << message << std::endl;
        }
    }

    static void log_vulkan_warning(const std::string& message)
    {
        if (log_normal_enabled())
        {
            std::cerr << "[COMPUTE][VULKAN] " << message << std::endl;
        }
    }

    void reset_state()
    {
        loader_handle_ = nullptr;
        loader_api_version_ = VK_API_VERSION_1_0;

        instance_ = VK_NULL_HANDLE;
        physical_device_ = VK_NULL_HANDLE;
        device_ = VK_NULL_HANDLE;
        compute_queue_ = VK_NULL_HANDLE;

        compute_queue_family_index_ = std::numeric_limits<uint32_t>::max();
        selected_device_index_ = -1;
        selected_device_properties_ = {};
        selected_device_features_ = {};
        selected_compute_queue_family_ = {};
        selected_supports_portability_subset_ = false;

        vkGetInstanceProcAddr_ = nullptr;
        vkCreateInstance_ = nullptr;
        vkEnumerateInstanceVersion_ = nullptr;
        vkEnumerateInstanceExtensionProperties_ = nullptr;
        vkEnumerateInstanceLayerProperties_ = nullptr;
        vkDestroyInstance_ = nullptr;
        vkEnumeratePhysicalDevices_ = nullptr;
        vkGetPhysicalDeviceProperties_ = nullptr;
        vkGetPhysicalDeviceFeatures_ = nullptr;
        vkGetPhysicalDeviceQueueFamilyProperties_ = nullptr;
        vkEnumerateDeviceExtensionProperties_ = nullptr;
        vkCreateDevice_ = nullptr;
        vkGetDeviceProcAddr_ = nullptr;
        vkDestroyDevice_ = nullptr;
        vkGetDeviceQueue_ = nullptr;
        vkDeviceWaitIdle_ = nullptr;
        vkGetPhysicalDeviceMemoryProperties_ = nullptr;

        vkCreateBuffer_ = nullptr;
        vkDestroyBuffer_ = nullptr;
        vkAllocateMemory_ = nullptr;
        vkFreeMemory_ = nullptr;
        vkBindBufferMemory_ = nullptr;
        vkMapMemory_ = nullptr;
        vkUnmapMemory_ = nullptr;
        vkGetBufferMemoryRequirements_ = nullptr;
        vkCreateShaderModule_ = nullptr;
        vkDestroyShaderModule_ = nullptr;
        vkCreateDescriptorSetLayout_ = nullptr;
        vkDestroyDescriptorSetLayout_ = nullptr;
        vkCreateDescriptorPool_ = nullptr;
        vkDestroyDescriptorPool_ = nullptr;
        vkAllocateDescriptorSets_ = nullptr;
        vkUpdateDescriptorSets_ = nullptr;
        vkCreatePipelineLayout_ = nullptr;
        vkDestroyPipelineLayout_ = nullptr;
        vkCreateComputePipelines_ = nullptr;
        vkDestroyPipeline_ = nullptr;
        vkCreateCommandPool_ = nullptr;
        vkDestroyCommandPool_ = nullptr;
        vkAllocateCommandBuffers_ = nullptr;
        vkBeginCommandBuffer_ = nullptr;
        vkEndCommandBuffer_ = nullptr;
        vkResetCommandBuffer_ = nullptr;
        vkCmdBindPipeline_ = nullptr;
        vkCmdBindDescriptorSets_ = nullptr;
        vkCmdPushConstants_ = nullptr;
        vkCmdDispatch_ = nullptr;
        vkCmdPipelineBarrier_ = nullptr;
        vkCmdCopyBuffer_ = nullptr;
        vkQueueSubmit_ = nullptr;
        vkCreateFence_ = nullptr;
        vkDestroyFence_ = nullptr;
        vkWaitForFences_ = nullptr;
        vkResetFences_ = nullptr;

        tvd_pipeline_ = {};
        radial_pipeline_ = {};
        azimuthal_pipeline_ = {};
        diffusion_pipeline_ = {};
        supercell_pipeline_ = {};
        tornado_pipeline_ = {};
        kessler_pointwise_pipeline_ = {};
        kessler_sedimentation_pipeline_ = {};
        cmd_pool_ = VK_NULL_HANDLE;
        cmd_buf_ = VK_NULL_HANDLE;
        fence_ = VK_NULL_HANDLE;

        staging_q_ = {};
        staging_w_ = {};
        staging_dz_ = {};
        staging_dqdt_ = {};
        device_q_ = {};
        device_w_ = {};
        device_dz_ = {};
        device_dqdt_ = {};

        buffer_pool_.destroy_all();
    }

    bool open_loader(std::string& error)
    {
#if defined(__APPLE__)
        const std::vector<const char*> loader_candidates = {
            "libvulkan.1.dylib",
            "libvulkan.dylib",
            "/opt/homebrew/lib/libvulkan.1.dylib",
            "/opt/homebrew/lib/libvulkan.dylib",
            "/opt/homebrew/opt/vulkan-loader/lib/libvulkan.1.dylib",
            "/opt/homebrew/opt/vulkan-loader/lib/libvulkan.dylib",
            "/usr/local/lib/libvulkan.1.dylib",
            "/usr/local/lib/libvulkan.dylib"
        };
#elif defined(__linux__)
        const std::vector<const char*> loader_candidates = {
            "libvulkan.so.1",
            "libvulkan.so",
            "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",   // Debian/Ubuntu
            "/usr/lib64/libvulkan.so.1",                   // Fedora/RHEL
            "/usr/lib/libvulkan.so.1",                     // Arch
        };
#else
        const std::vector<const char*> loader_candidates;
#endif

        std::vector<std::string> tried_candidates;

        const char* loader_override = std::getenv("TMV_VULKAN_LOADER_PATH");
        if (loader_override != nullptr && loader_override[0] != '\0')
        {
            loader_handle_ = dlopen(loader_override, RTLD_NOW | RTLD_LOCAL);
            tried_candidates.push_back(loader_override);
        }

        for (const char* candidate : loader_candidates)
        {
            if (loader_handle_ != nullptr)
            {
                break;
            }
            loader_handle_ = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
            tried_candidates.emplace_back(candidate);
        }

        if (loader_handle_ == nullptr)
        {
            std::ostringstream stream;
            stream << "failed to load Vulkan loader shared library (libvulkan). Tried: ";
            for (std::size_t i = 0; i < tried_candidates.size(); ++i)
            {
                if (i > 0)
                {
                    stream << ", ";
                }
                stream << tried_candidates[i];
            }
            stream << ". You can override the loader path via TMV_VULKAN_LOADER_PATH.";
            error = stream.str();
            return false;
        }

        vkGetInstanceProcAddr_ = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
            dlsym(loader_handle_, "vkGetInstanceProcAddr"));
        if (vkGetInstanceProcAddr_ == nullptr)
        {
            error = "Vulkan loader is missing vkGetInstanceProcAddr";
            return false;
        }

        log_vulkan_info("loaded Vulkan loader successfully");
        return true;
    }

    bool load_global_functions(std::string& error)
    {
        if (!load_global_proc(vkGetInstanceProcAddr_, "vkCreateInstance", vkCreateInstance_))
        {
            error = "failed to load vkCreateInstance from Vulkan loader";
            return false;
        }
        if (!load_global_proc(vkGetInstanceProcAddr_,
                              "vkEnumerateInstanceExtensionProperties",
                              vkEnumerateInstanceExtensionProperties_))
        {
            error = "failed to load vkEnumerateInstanceExtensionProperties from Vulkan loader";
            return false;
        }
        if (!load_global_proc(vkGetInstanceProcAddr_,
                              "vkEnumerateInstanceLayerProperties",
                              vkEnumerateInstanceLayerProperties_))
        {
            error = "failed to load vkEnumerateInstanceLayerProperties from Vulkan loader";
            return false;
        }

        load_global_proc(vkGetInstanceProcAddr_,
                         "vkEnumerateInstanceVersion",
                         vkEnumerateInstanceVersion_);

        loader_api_version_ = VK_API_VERSION_1_0;
        if (vkEnumerateInstanceVersion_ != nullptr)
        {
            uint32_t version = VK_API_VERSION_1_0;
            const VkResult result = vkEnumerateInstanceVersion_(&version);
            if (result == VK_SUCCESS)
            {
                loader_api_version_ = version;
            }
        }

        log_vulkan_info("loader API version=" + version_to_string(loader_api_version_));
        return true;
    }

    bool has_instance_extension(const char* extension_name) const
    {
        if (vkEnumerateInstanceExtensionProperties_ == nullptr || extension_name == nullptr)
        {
            return false;
        }

        uint32_t count = 0;
        if (vkEnumerateInstanceExtensionProperties_(nullptr, &count, nullptr) != VK_SUCCESS)
        {
            return false;
        }

        std::vector<VkExtensionProperties> extensions(count);
        if (count > 0 &&
            vkEnumerateInstanceExtensionProperties_(
                nullptr,
                &count,
                extensions.data()) != VK_SUCCESS)
        {
            return false;
        }

        extensions.resize(count);
        return std::any_of(
            extensions.begin(),
            extensions.end(),
            [extension_name](const VkExtensionProperties& extension)
            {
                return std::strcmp(extension.extensionName, extension_name) == 0;
            });
    }

    bool has_instance_layer(const char* layer_name) const
    {
        if (vkEnumerateInstanceLayerProperties_ == nullptr || layer_name == nullptr)
        {
            return false;
        }

        uint32_t count = 0;
        if (vkEnumerateInstanceLayerProperties_(&count, nullptr) != VK_SUCCESS)
        {
            return false;
        }

        std::vector<VkLayerProperties> layers(count);
        if (count > 0 &&
            vkEnumerateInstanceLayerProperties_(&count, layers.data()) != VK_SUCCESS)
        {
            return false;
        }

        layers.resize(count);
        return std::any_of(
            layers.begin(),
            layers.end(),
            [layer_name](const VkLayerProperties& layer)
            {
                return std::strcmp(layer.layerName, layer_name) == 0;
            });
    }

    bool create_instance(std::string& error)
    {
        constexpr const char* kValidationLayerName = "VK_LAYER_KHRONOS_validation";
        std::vector<const char*> extensions;
        VkInstanceCreateFlags create_flags = 0;
        bool portability_enabled = false;
        bool portability_props2_enabled = false;

        auto append_extension = [&extensions](const char* extension_name)
        {
            const bool already_enabled =
                std::any_of(extensions.begin(),
                            extensions.end(),
                            [extension_name](const char* existing)
                            {
                                return std::strcmp(existing, extension_name) == 0;
                            });
            if (!already_enabled)
            {
                extensions.push_back(extension_name);
            }
        };

        if (has_instance_extension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME))
        {
            append_extension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            create_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
            portability_enabled = true;

            if (has_instance_extension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
            {
                append_extension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
                portability_props2_enabled = true;
            }
        }

        const bool validation_layer_available = has_instance_layer(kValidationLayerName);
        log_vulkan_info(
            "instance support: portability_enumeration=" + std::string(portability_enabled ? "yes" : "no") +
            ", properties2=" + std::string(portability_props2_enabled ? "yes" : "no") +
            ", validation_layer=" + std::string(validation_layer_available ? "yes" : "no"));

        const uint32_t preferred_api_version = std::min(loader_api_version_, VK_API_VERSION_1_1);

        auto try_create_instance = [&](uint32_t api_version,
                                       const std::vector<const char*>& requested_extensions,
                                       VkInstanceCreateFlags flags,
                                       VkResult& out_result,
                                       VkInstance& out_instance)
        {
            VkApplicationInfo app_info{};
            app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            app_info.pApplicationName = "TornadoModel Compute Backend";
            app_info.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
            app_info.pEngineName = "TornadoModel";
            app_info.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
            app_info.apiVersion = api_version;

            VkInstanceCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            create_info.pApplicationInfo = &app_info;
            create_info.flags = flags;
            create_info.enabledExtensionCount = static_cast<uint32_t>(requested_extensions.size());
            create_info.ppEnabledExtensionNames =
                requested_extensions.empty() ? nullptr : requested_extensions.data();

            out_result = vkCreateInstance_(&create_info, nullptr, &out_instance);
            return out_result == VK_SUCCESS;
        };

        VkResult result = VK_ERROR_INITIALIZATION_FAILED;
        VkInstance created_instance = VK_NULL_HANDLE;
        std::vector<const char*> active_extensions = extensions;
        VkInstanceCreateFlags active_flags = create_flags;
        uint32_t active_api_version = preferred_api_version;

        if (try_create_instance(active_api_version, active_extensions, active_flags, result, created_instance))
        {
            instance_ = created_instance;
            return true;
        }

        if (result == VK_ERROR_INCOMPATIBLE_DRIVER && active_api_version != VK_API_VERSION_1_0)
        {
            log_vulkan_warning(
                "vkCreateInstance reported VK_ERROR_INCOMPATIBLE_DRIVER with API " +
                version_to_string(active_api_version) + "; retrying with Vulkan 1.0");
            active_api_version = VK_API_VERSION_1_0;
            if (try_create_instance(active_api_version, active_extensions, active_flags, result, created_instance))
            {
                instance_ = created_instance;
                return true;
            }
        }

        if (portability_enabled)
        {
            log_vulkan_warning(
                "vkCreateInstance failed with portability enumeration enabled; retrying without portability extension");
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

            if (try_create_instance(active_api_version, active_extensions, active_flags, result, created_instance))
            {
                instance_ = created_instance;
                return true;
            }
        }

        error = "vkCreateInstance failed with VkResult=" + vk_result_string(result) +
            " (requested_api=" + version_to_string(preferred_api_version) +
            ", active_api=" + version_to_string(active_api_version) +
            ", extensions=[" + join_extension_names(active_extensions) + "])" +
            instance_failure_hint(result);
        return false;
    }

    bool load_instance_functions(std::string& error)
    {
        if (!load_instance_proc(vkGetInstanceProcAddr_, instance_, "vkDestroyInstance", vkDestroyInstance_))
        {
            error = "failed to load vkDestroyInstance";
            return false;
        }
        if (!load_instance_proc(vkGetInstanceProcAddr_,
                                instance_,
                                "vkEnumeratePhysicalDevices",
                                vkEnumeratePhysicalDevices_))
        {
            error = "failed to load vkEnumeratePhysicalDevices";
            return false;
        }
        if (!load_instance_proc(vkGetInstanceProcAddr_,
                                instance_,
                                "vkGetPhysicalDeviceProperties",
                                vkGetPhysicalDeviceProperties_))
        {
            error = "failed to load vkGetPhysicalDeviceProperties";
            return false;
        }
        if (!load_instance_proc(vkGetInstanceProcAddr_,
                                instance_,
                                "vkGetPhysicalDeviceFeatures",
                                vkGetPhysicalDeviceFeatures_))
        {
            error = "failed to load vkGetPhysicalDeviceFeatures";
            return false;
        }
        if (!load_instance_proc(vkGetInstanceProcAddr_,
                                instance_,
                                "vkGetPhysicalDeviceQueueFamilyProperties",
                                vkGetPhysicalDeviceQueueFamilyProperties_))
        {
            error = "failed to load vkGetPhysicalDeviceQueueFamilyProperties";
            return false;
        }
        if (!load_instance_proc(vkGetInstanceProcAddr_,
                                instance_,
                                "vkEnumerateDeviceExtensionProperties",
                                vkEnumerateDeviceExtensionProperties_))
        {
            error = "failed to load vkEnumerateDeviceExtensionProperties";
            return false;
        }
        if (!load_instance_proc(vkGetInstanceProcAddr_, instance_, "vkCreateDevice", vkCreateDevice_))
        {
            error = "failed to load vkCreateDevice";
            return false;
        }
        if (!load_instance_proc(vkGetInstanceProcAddr_, instance_, "vkGetDeviceProcAddr", vkGetDeviceProcAddr_))
        {
            error = "failed to load vkGetDeviceProcAddr";
            return false;
        }
        if (!load_instance_proc(vkGetInstanceProcAddr_, instance_,
                                "vkGetPhysicalDeviceMemoryProperties",
                                vkGetPhysicalDeviceMemoryProperties_))
        {
            error = "failed to load vkGetPhysicalDeviceMemoryProperties";
            return false;
        }
        return true;
    }

    bool has_device_extension(VkPhysicalDevice device, const char* extension_name) const
    {
        if (vkEnumerateDeviceExtensionProperties_ == nullptr ||
            device == VK_NULL_HANDLE ||
            extension_name == nullptr)
        {
            return false;
        }

        uint32_t count = 0;
        if (vkEnumerateDeviceExtensionProperties_(device, nullptr, &count, nullptr) != VK_SUCCESS)
        {
            return false;
        }

        std::vector<VkExtensionProperties> extensions(count);
        if (count > 0 &&
            vkEnumerateDeviceExtensionProperties_(device, nullptr, &count, extensions.data()) != VK_SUCCESS)
        {
            return false;
        }

        extensions.resize(count);
        return std::any_of(
            extensions.begin(),
            extensions.end(),
            [extension_name](const VkExtensionProperties& extension)
            {
                return std::strcmp(extension.extensionName, extension_name) == 0;
            });
    }

    bool select_physical_device(std::string& error)
    {
        uint32_t physical_device_count = 0;
        VkResult result =
            vkEnumeratePhysicalDevices_(instance_, &physical_device_count, nullptr);
        if (result != VK_SUCCESS)
        {
            error = "vkEnumeratePhysicalDevices(count) failed with VkResult=" +
                std::to_string(static_cast<int>(result));
            return false;
        }
        if (physical_device_count == 0)
        {
            error = "no Vulkan physical devices found";
            return false;
        }

        std::vector<VkPhysicalDevice> physical_devices(physical_device_count, VK_NULL_HANDLE);
        result = vkEnumeratePhysicalDevices_(
            instance_,
            &physical_device_count,
            physical_devices.data());
        if (result != VK_SUCCESS)
        {
            error = "vkEnumeratePhysicalDevices(list) failed with VkResult=" +
                std::to_string(static_cast<int>(result));
            return false;
        }
        physical_devices.resize(physical_device_count);

        std::vector<PhysicalDeviceCandidate> candidates;
        candidates.reserve(physical_devices.size());

        for (std::size_t i = 0; i < physical_devices.size(); ++i)
        {
            PhysicalDeviceCandidate candidate{};
            candidate.enumeration_index = i;
            candidate.handle = physical_devices[i];
            vkGetPhysicalDeviceProperties_(candidate.handle, &candidate.properties);
            vkGetPhysicalDeviceFeatures_(candidate.handle, &candidate.features);

            uint32_t queue_family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties_(
                candidate.handle,
                &queue_family_count,
                nullptr);

            if (queue_family_count > 0)
            {
                std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
                vkGetPhysicalDeviceQueueFamilyProperties_(
                    candidate.handle,
                    &queue_family_count,
                    queue_families.data());

                queue_families.resize(queue_family_count);
                for (uint32_t queue_index = 0; queue_index < queue_family_count; ++queue_index)
                {
                    const VkQueueFamilyProperties& queue_family = queue_families[queue_index];
                    if (queue_family.queueCount == 0)
                    {
                        continue;
                    }
                    if ((queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT) == 0)
                    {
                        continue;
                    }

                    candidate.compute_queue_family_index = queue_index;
                    candidate.compute_queue_family = queue_family;
                    break;
                }
            }

            const bool has_compute_queue =
                candidate.compute_queue_family_index != std::numeric_limits<uint32_t>::max();
            candidate.has_compute_queue = has_compute_queue;

            const bool limits_usable =
                candidate.properties.limits.maxComputeWorkGroupInvocations > 0 &&
                candidate.properties.limits.maxComputeWorkGroupCount[0] > 0 &&
                candidate.properties.limits.maxComputeWorkGroupCount[1] > 0 &&
                candidate.properties.limits.maxComputeWorkGroupCount[2] > 0 &&
                candidate.properties.limits.maxComputeSharedMemorySize > 0;
            candidate.limits_usable = limits_usable;

            candidate.suitable = has_compute_queue && limits_usable;
            if (candidate.suitable)
            {
                candidate.score = device_type_score(candidate.properties.deviceType);
                candidate.score +=
                    static_cast<int>(candidate.properties.limits.maxComputeSharedMemorySize / 1024);
                candidate.score +=
                    static_cast<int>(candidate.properties.limits.maxComputeWorkGroupInvocations / 32);
            }

            candidate.supports_portability_subset =
                has_device_extension(candidate.handle, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);

            if (!candidate.has_compute_queue)
            {
                candidate.unsuitable_reason = "no compute queue family";
            }
            else if (!candidate.limits_usable)
            {
                candidate.unsuitable_reason = "insufficient compute workgroup/shared-memory limits";
            }
            else
            {
                candidate.unsuitable_reason = "none";
            }

            candidates.push_back(candidate);
        }

        for (const PhysicalDeviceCandidate& candidate : candidates)
        {
            std::ostringstream stream;
            stream << "device[" << candidate.enumeration_index << "] "
                   << "name=\"" << candidate.properties.deviceName << "\""
                   << ", type=" << device_type_name(candidate.properties.deviceType)
                   << ", api=" << version_to_string(candidate.properties.apiVersion)
                   << ", compute_queue=" << (candidate.has_compute_queue ? "yes" : "no");
            if (candidate.has_compute_queue)
            {
                stream << "(family=" << candidate.compute_queue_family_index
                       << ", queues=" << candidate.compute_queue_family.queueCount << ")";
            }
            stream << ", max_invocations="
                   << candidate.properties.limits.maxComputeWorkGroupInvocations
                   << ", shared_mem_bytes="
                   << candidate.properties.limits.maxComputeSharedMemorySize
                   << ", portability_subset="
                   << (candidate.supports_portability_subset ? "yes" : "no")
                   << ", suitable=" << (candidate.suitable ? "yes" : "no");
            if (!candidate.suitable)
            {
                stream << ", reason=" << candidate.unsuitable_reason;
            }
            if (candidate.suitable)
            {
                stream << ", score=" << candidate.score;
            }
            log_vulkan_info(stream.str());
        }

        const int preferred_device_index = global_compute_backend_config.device_index;
        PhysicalDeviceCandidate selected_candidate{};
        bool selected = false;

        if (preferred_device_index >= 0)
        {
            const std::size_t index = static_cast<std::size_t>(preferred_device_index);
            if (index >= candidates.size())
            {
                error = "requested numerics.compute.device_index " +
                    std::to_string(preferred_device_index) + " is out of range (detected devices: " +
                    std::to_string(candidates.size()) + ")";
                return false;
            }

            selected_candidate = candidates[index];
            if (!selected_candidate.suitable)
            {
                error = "requested Vulkan device index " +
                    std::to_string(preferred_device_index) +
                    " does not expose a usable compute queue/capabilities";
                return false;
            }
            selected = true;
        }
        else
        {
            int best_score = std::numeric_limits<int>::min();
            for (const PhysicalDeviceCandidate& candidate : candidates)
            {
                if (!candidate.suitable)
                {
                    continue;
                }

                if (!selected || candidate.score > best_score)
                {
                    selected_candidate = candidate;
                    best_score = candidate.score;
                    selected = true;
                }
            }

            if (!selected)
            {
                error = "no suitable Vulkan physical device found with compute queue support";
                return false;
            }
        }

        selected_device_index_ = static_cast<int>(selected_candidate.enumeration_index);
        physical_device_ = selected_candidate.handle;
        selected_device_properties_ = selected_candidate.properties;
        selected_device_features_ = selected_candidate.features;
        compute_queue_family_index_ = selected_candidate.compute_queue_family_index;
        selected_compute_queue_family_ = selected_candidate.compute_queue_family;
        selected_supports_portability_subset_ = selected_candidate.supports_portability_subset;
        log_vulkan_info(
            "selected device[" + std::to_string(selected_device_index_) + "] \"" +
            std::string(selected_device_properties_.deviceName) + "\"");
        return true;
    }

    bool create_logical_device(std::string& error)
    {
        if (physical_device_ == VK_NULL_HANDLE)
        {
            error = "Vulkan physical device is not selected";
            return false;
        }
        if (compute_queue_family_index_ == std::numeric_limits<uint32_t>::max())
        {
            error = "no compute queue family selected";
            return false;
        }

        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = compute_queue_family_index_;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &queue_priority;

        std::vector<const char*> device_extensions;
        if (selected_supports_portability_subset_)
        {
            device_extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
        }

        VkPhysicalDeviceFeatures requested_features{};

        VkDeviceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.queueCreateInfoCount = 1;
        create_info.pQueueCreateInfos = &queue_info;
        create_info.enabledExtensionCount =
            static_cast<uint32_t>(device_extensions.size());
        create_info.ppEnabledExtensionNames =
            device_extensions.empty() ? nullptr : device_extensions.data();
        create_info.pEnabledFeatures = &requested_features;

        const VkResult result = vkCreateDevice_(physical_device_, &create_info, nullptr, &device_);
        if (result != VK_SUCCESS)
        {
            error = "vkCreateDevice failed with VkResult=" +
                std::to_string(static_cast<int>(result));
            return false;
        }
        log_vulkan_info("created logical device successfully");
        return true;
    }

    bool load_device_functions(std::string& error)
    {
        if (!load_device_proc(vkGetDeviceProcAddr_, device_, "vkDestroyDevice", vkDestroyDevice_))
        {
            error = "failed to load vkDestroyDevice";
            return false;
        }
        if (!load_device_proc(vkGetDeviceProcAddr_, device_, "vkGetDeviceQueue", vkGetDeviceQueue_))
        {
            error = "failed to load vkGetDeviceQueue";
            return false;
        }

        load_device_proc(vkGetDeviceProcAddr_, device_, "vkDeviceWaitIdle", vkDeviceWaitIdle_);

        vkGetDeviceQueue_(device_, compute_queue_family_index_, 0, &compute_queue_);
        if (compute_queue_ == VK_NULL_HANDLE)
        {
            error = "selected Vulkan compute queue handle is null";
            return false;
        }
        log_vulkan_info("resolved compute queue handle");

        // Load compute resource function pointers
        bool ok = true;
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkCreateBuffer", vkCreateBuffer_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkDestroyBuffer", vkDestroyBuffer_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkAllocateMemory", vkAllocateMemory_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkFreeMemory", vkFreeMemory_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkBindBufferMemory", vkBindBufferMemory_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkMapMemory", vkMapMemory_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkUnmapMemory", vkUnmapMemory_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkGetBufferMemoryRequirements", vkGetBufferMemoryRequirements_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkCreateShaderModule", vkCreateShaderModule_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkDestroyShaderModule", vkDestroyShaderModule_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkCreateDescriptorSetLayout", vkCreateDescriptorSetLayout_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkDestroyDescriptorSetLayout", vkDestroyDescriptorSetLayout_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkCreateDescriptorPool", vkCreateDescriptorPool_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkDestroyDescriptorPool", vkDestroyDescriptorPool_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkAllocateDescriptorSets", vkAllocateDescriptorSets_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkUpdateDescriptorSets", vkUpdateDescriptorSets_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkCreatePipelineLayout", vkCreatePipelineLayout_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkDestroyPipelineLayout", vkDestroyPipelineLayout_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkCreateComputePipelines", vkCreateComputePipelines_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkDestroyPipeline", vkDestroyPipeline_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkCreateCommandPool", vkCreateCommandPool_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkDestroyCommandPool", vkDestroyCommandPool_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkAllocateCommandBuffers", vkAllocateCommandBuffers_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkBeginCommandBuffer", vkBeginCommandBuffer_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkEndCommandBuffer", vkEndCommandBuffer_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkResetCommandBuffer", vkResetCommandBuffer_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkCmdBindPipeline", vkCmdBindPipeline_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkCmdBindDescriptorSets", vkCmdBindDescriptorSets_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkCmdPushConstants", vkCmdPushConstants_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkCmdDispatch", vkCmdDispatch_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkCmdPipelineBarrier", vkCmdPipelineBarrier_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkCmdCopyBuffer", vkCmdCopyBuffer_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkQueueSubmit", vkQueueSubmit_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkCreateFence", vkCreateFence_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkDestroyFence", vkDestroyFence_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkWaitForFences", vkWaitForFences_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_, "vkResetFences", vkResetFences_);

        if (!ok)
        {
            error = "failed to load one or more Vulkan compute device functions";
            return false;
        }
        log_vulkan_info("loaded all compute device functions");
        return true;
    }

    // ── Buffer management helpers ────────────────────────────────────

    uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags props) const
    {
        VkPhysicalDeviceMemoryProperties mem_props{};
        vkGetPhysicalDeviceMemoryProperties_(physical_device_, &mem_props);
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i)
        {
            if ((type_bits & (1u << i)) &&
                (mem_props.memoryTypes[i].propertyFlags & props) == props)
            {
                return i;
            }
        }
        return std::numeric_limits<uint32_t>::max();
    }

    bool create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags mem_props, GpuBuffer& out)
    {
        VkBufferCreateInfo buf_info{};
        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_info.size = size;
        buf_info.usage = usage;
        buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer_(device_, &buf_info, nullptr, &out.buffer) != VK_SUCCESS)
        {
            return false;
        }

        VkMemoryRequirements reqs{};
        vkGetBufferMemoryRequirements_(device_, out.buffer, &reqs);

        const uint32_t mem_type_idx = find_memory_type(reqs.memoryTypeBits, mem_props);
        if (mem_type_idx == std::numeric_limits<uint32_t>::max())
        {
            vkDestroyBuffer_(device_, out.buffer, nullptr);
            out.buffer = VK_NULL_HANDLE;
            return false;
        }

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = reqs.size;
        alloc_info.memoryTypeIndex = mem_type_idx;

        if (vkAllocateMemory_(device_, &alloc_info, nullptr, &out.memory) != VK_SUCCESS)
        {
            vkDestroyBuffer_(device_, out.buffer, nullptr);
            out.buffer = VK_NULL_HANDLE;
            return false;
        }

        if (vkBindBufferMemory_(device_, out.buffer, out.memory, 0) != VK_SUCCESS)
        {
            vkFreeMemory_(device_, out.memory, nullptr);
            vkDestroyBuffer_(device_, out.buffer, nullptr);
            out = {};
            return false;
        }

        out.size = size;

        // Map host-visible buffers persistently
        if ((mem_props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0)
        {
            if (vkMapMemory_(device_, out.memory, 0, size, 0, &out.mapped) != VK_SUCCESS)
            {
                out.mapped = nullptr;
            }
        }

        return true;
    }

    void destroy_buffer(GpuBuffer& buf)
    {
        if (buf.mapped != nullptr && vkUnmapMemory_ != nullptr)
        {
            vkUnmapMemory_(device_, buf.memory);
        }
        if (buf.memory != VK_NULL_HANDLE && vkFreeMemory_ != nullptr)
        {
            vkFreeMemory_(device_, buf.memory, nullptr);
        }
        if (buf.buffer != VK_NULL_HANDLE && vkDestroyBuffer_ != nullptr)
        {
            vkDestroyBuffer_(device_, buf.buffer, nullptr);
        }
        buf = {};
    }

    // ── Buffer pool callbacks ────────────────────────────────────────────

    static bool pool_create_buffer(void* user_data, VkDeviceSize size,
                                   bool is_staging, GpuBuffer& out_buf)
    {
        auto* self = static_cast<VulkanComputeBackend*>(user_data);

        // Unified memory path: both staging and device use a single buffer
        // with HOST_VISIBLE + DEVICE_LOCAL. The staging slot gets a no-op
        // placeholder (device buffer is used directly for both CPU and GPU).
        if (self->has_unified_memory_)
        {
            if (is_staging)
            {
                // Staging is unused on unified memory — create a placeholder.
                // The dispatch path will read/write via the device buffer's mapping.
                out_buf = {};
                return true;
            }
            // Device buffer: unified memory (accessible from both CPU and GPU)
            constexpr VkBufferUsageFlags kUnifiedUsage =
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            constexpr VkMemoryPropertyFlags kUnifiedProps =
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            return self->create_buffer(size, kUnifiedUsage, kUnifiedProps, out_buf);
        }

        // Discrete memory path: separate staging (host) and device (GPU) buffers
        constexpr VkBufferUsageFlags kStagingUsage =
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        constexpr VkBufferUsageFlags kDeviceUsage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        constexpr VkMemoryPropertyFlags kHostProps =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        constexpr VkMemoryPropertyFlags kDeviceProps =
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        if (is_staging)
        {
            return self->create_buffer(size, kStagingUsage, kHostProps, out_buf);
        }
        return self->create_buffer(size, kDeviceUsage, kDeviceProps, out_buf);
    }

    static void pool_destroy_buffer(void* user_data, GpuBuffer& buf)
    {
        auto* self = static_cast<VulkanComputeBackend*>(user_data);
        self->destroy_buffer(buf);
    }

    /**
     * @brief Detects whether the device supports unified memory (HOST_VISIBLE + DEVICE_LOCAL).
     *
     * On Apple Silicon (via MoltenVK) and integrated GPUs, a single memory type
     * can be both host-visible and device-local, eliminating the need for separate
     * staging buffers and explicit H2D/D2H transfers.
     */
    void detect_unified_memory()
    {
        has_unified_memory_ = false;
        if (vkGetPhysicalDeviceMemoryProperties_ == nullptr ||
            physical_device_ == VK_NULL_HANDLE)
        {
            return;
        }

        VkPhysicalDeviceMemoryProperties mem_props{};
        vkGetPhysicalDeviceMemoryProperties_(physical_device_, &mem_props);

        constexpr VkMemoryPropertyFlags kUnifiedFlags =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i)
        {
            if ((mem_props.memoryTypes[i].propertyFlags & kUnifiedFlags) == kUnifiedFlags)
            {
                has_unified_memory_ = true;
                log_vulkan_info("unified memory detected (HOST_VISIBLE + DEVICE_LOCAL) — "
                                "zero-copy buffer path enabled");
                return;
            }
        }
        log_vulkan_info("discrete memory layout — using staging buffer transfers");
    }

    void init_buffer_pool()
    {
        buffer_pool_.set_allocator(&pool_create_buffer, &pool_destroy_buffer, this);
    }

    /**
     * @brief Acquires N pool slots and populates slot_indices.
     * @param field_bytes  Minimum byte size per buffer.
     * @param count        Number of slots needed.
     * @param slot_indices Output array of slot indices (must have room for count).
     * @return True if all slots acquired successfully.
     */
    bool acquire_pool_slots(VkDeviceSize field_bytes, int count, int* slot_indices)
    {
        for (int i = 0; i < count; ++i)
        {
            slot_indices[i] = buffer_pool_.acquire(field_bytes);
            if (slot_indices[i] < 0)
            {
                // Roll back acquired slots
                for (int j = 0; j < i; ++j)
                {
                    buffer_pool_.release(slot_indices[j]);
                }
                return false;
            }
        }
        return true;
    }

    void release_pool_slots(const int* slot_indices, int count)
    {
        for (int i = 0; i < count; ++i)
        {
            buffer_pool_.release(slot_indices[i]);
        }
    }

    bool ensure_buffers(VkDeviceSize field_bytes, VkDeviceSize dz_bytes)
    {
        if (staging_q_.size >= field_bytes && staging_dz_.size >= dz_bytes)
        {
            return true; // Already large enough
        }

        // Free existing buffers
        destroy_buffer(staging_q_);
        destroy_buffer(staging_w_);
        destroy_buffer(staging_dz_);
        destroy_buffer(staging_dqdt_);
        destroy_buffer(device_q_);
        destroy_buffer(device_w_);
        destroy_buffer(device_dz_);
        destroy_buffer(device_dqdt_);

        constexpr VkMemoryPropertyFlags kHostProps =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        constexpr VkMemoryPropertyFlags kDeviceProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        constexpr VkBufferUsageFlags kStagingSrc =
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        constexpr VkBufferUsageFlags kDeviceSsbo =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        bool ok = true;
        ok = ok && create_buffer(field_bytes, kStagingSrc, kHostProps, staging_q_);
        ok = ok && create_buffer(field_bytes, kStagingSrc, kHostProps, staging_w_);
        ok = ok && create_buffer(dz_bytes,    kStagingSrc, kHostProps, staging_dz_);
        ok = ok && create_buffer(field_bytes, kStagingSrc, kHostProps, staging_dqdt_);
        ok = ok && create_buffer(field_bytes, kDeviceSsbo, kDeviceProps, device_q_);
        ok = ok && create_buffer(field_bytes, kDeviceSsbo, kDeviceProps, device_w_);
        ok = ok && create_buffer(dz_bytes,    kDeviceSsbo, kDeviceProps, device_dz_);
        ok = ok && create_buffer(field_bytes, kDeviceSsbo, kDeviceProps, device_dqdt_);

        if (!ok)
        {
            log_vulkan_warning("buffer allocation failed — freeing partial allocations");
            destroy_buffer(staging_q_); destroy_buffer(staging_w_);
            destroy_buffer(staging_dz_); destroy_buffer(staging_dqdt_);
            destroy_buffer(device_q_); destroy_buffer(device_w_);
            destroy_buffer(device_dz_); destroy_buffer(device_dqdt_);
            return false;
        }

        // Update descriptor set with new buffer handles
        update_tvd_descriptor_set(field_bytes, dz_bytes);
        return true;
    }

    void update_tvd_descriptor_set(VkDeviceSize field_bytes, VkDeviceSize dz_bytes)
    {
        VkDescriptorBufferInfo q_info{device_q_.buffer, 0, field_bytes};
        VkDescriptorBufferInfo w_info{device_w_.buffer, 0, field_bytes};
        VkDescriptorBufferInfo dz_info{device_dz_.buffer, 0, dz_bytes};
        VkDescriptorBufferInfo dqdt_info{device_dqdt_.buffer, 0, field_bytes};

        VkWriteDescriptorSet writes[4]{};
        for (int i = 0; i < 4; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = tvd_pipeline_.descriptor_set;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        writes[0].pBufferInfo = &q_info;
        writes[1].pBufferInfo = &w_info;
        writes[2].pBufferInfo = &dz_info;
        writes[3].pBufferInfo = &dqdt_info;

        vkUpdateDescriptorSets_(device_, 4, writes, 0, nullptr);
    }

    // ── Descriptor set update for pooled buffers ──────────────────────

    void update_pooled_descriptor_set(ComputePipelineState& pipeline,
                                      const int* slot_indices,
                                      VkDeviceSize field_bytes, int count)
    {
        std::vector<VkDescriptorBufferInfo> infos(static_cast<std::size_t>(count));
        std::vector<VkWriteDescriptorSet> writes(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i)
        {
            const auto idx = static_cast<std::size_t>(i);
            infos[idx] = {buffer_pool_.device(slot_indices[i]).buffer, 0, field_bytes};
            writes[idx] = {};
            writes[idx].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[idx].dstSet = pipeline.descriptor_set;
            writes[idx].dstBinding = static_cast<uint32_t>(i);
            writes[idx].descriptorCount = 1;
            writes[idx].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[idx].pBufferInfo = &infos[idx];
        }
        vkUpdateDescriptorSets_(device_, static_cast<uint32_t>(count),
                                writes.data(), 0, nullptr);
    }

    // ── Generic 3-field dispatch helper (uses buffer pool) ──────────

    bool dispatch_field3_kernel(
        ComputePipelineState& pipeline,
        const void* push_constants,
        const float* input_a, const float* input_b, float* output,
        int nr, int nth, int nz,
        int input_count,
        uint32_t workgroup_divisor)
    {
        const int total_cells = nr * nth * nz;
        const VkDeviceSize field_bytes = static_cast<VkDeviceSize>(total_cells) * sizeof(float);
        const int buf_count = input_count + 1; // inputs + 1 output

        int slots[16];
        if (!acquire_pool_slots(field_bytes, buf_count, slots))
        {
            log_vulkan_warning("failed to acquire pool slots for field3 dispatch");
            return false;
        }
        update_pooled_descriptor_set(pipeline, slots, field_bytes, buf_count);

        // On unified memory, device buffers are host-visible — read/write directly.
        // On discrete memory, use separate staging buffers for H2D/D2H transfers.
        const bool unified = has_unified_memory_;
        auto upload_ptr = [&](int slot) -> void*
        {
            return unified ? buffer_pool_.device(slot).mapped
                           : buffer_pool_.staging(slot).mapped;
        };

        // Upload inputs
        std::memcpy(upload_ptr(slots[0]), input_a, field_bytes);
        if (input_count >= 2 && input_b != nullptr)
        {
            std::memcpy(upload_ptr(slots[1]), input_b, field_bytes);
        }
        std::memset(upload_ptr(slots[buf_count - 1]), 0, field_bytes);

        // Record command buffer
        vkResetCommandBuffer_(cmd_buf_, 0);
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer_(cmd_buf_, &begin_info) != VK_SUCCESS)
        {
            release_pool_slots(slots, buf_count);
            return false;
        }

        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;

        if (unified)
        {
            // Unified path: host writes are coherent; barrier ensures GPU visibility
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
        else
        {
            // Discrete path: copy staging → device
            VkBufferCopy copy_region{};
            copy_region.size = field_bytes;
            for (int i = 0; i < buf_count; ++i)
            {
                vkCmdCopyBuffer_(cmd_buf_,
                                 buffer_pool_.staging(slots[i]).buffer,
                                 buffer_pool_.device(slots[i]).buffer,
                                 1, &copy_region);
            }

            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0, 1, &barrier, 0, nullptr, 0, nullptr);
        }

        // Bind pipeline and descriptors
        vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
        vkCmdBindDescriptorSets_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipeline.pipeline_layout, 0, 1,
                                 &pipeline.descriptor_set, 0, nullptr);

        // Push constants
        vkCmdPushConstants_(cmd_buf_, pipeline.pipeline_layout,
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0, pipeline.push_constant_size, push_constants);

        // Dispatch
        const uint32_t total_work = static_cast<uint32_t>(workgroup_divisor);
        const uint32_t workgroup_count = (total_work + 63u) / 64u;
        vkCmdDispatch_(cmd_buf_, workgroup_count, 1, 1);

        if (unified)
        {
            // Unified path: barrier so CPU can read GPU writes after fence
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
        else
        {
            // Discrete path: copy device output → staging
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0, 1, &barrier, 0, nullptr, 0, nullptr);

            VkBufferCopy copy_region{};
            copy_region.size = field_bytes;
            vkCmdCopyBuffer_(cmd_buf_,
                             buffer_pool_.device(slots[buf_count - 1]).buffer,
                             buffer_pool_.staging(slots[buf_count - 1]).buffer,
                             1, &copy_region);
        }

        if (vkEndCommandBuffer_(cmd_buf_) != VK_SUCCESS)
        {
            release_pool_slots(slots, buf_count);
            return false;
        }

        // Submit and wait
        vkResetFences_(device_, 1, &fence_);
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buf_;
        if (vkQueueSubmit_(compute_queue_, 1, &submit_info, fence_) != VK_SUCCESS)
        {
            release_pool_slots(slots, buf_count);
            return false;
        }

        constexpr uint64_t kTimeoutNs = 10ULL * 1000000000ULL;
        if (vkWaitForFences_(device_, 1, &fence_, VK_TRUE, kTimeoutNs) != VK_SUCCESS)
        {
            release_pool_slots(slots, buf_count);
            return false;
        }

        // Download output: device buffer on unified, staging on discrete
        const void* output_src = unified
            ? buffer_pool_.device(slots[buf_count - 1]).mapped
            : buffer_pool_.staging(slots[buf_count - 1]).mapped;
        std::memcpy(output, output_src, field_bytes);
        release_pool_slots(slots, buf_count);
        return true;
    }

    // ── Multi-field dispatch helper (uses buffer pool) ───────────────

    bool dispatch_multi_field_kernel(
        ComputePipelineState& pipeline,
        const void* push_constants,
        const float* const* inputs, int input_count,
        float* const* outputs, int output_count,
        int nr, int nth, int nz,
        uint32_t total_invocations)
    {
        const int total_cells = nr * nth * nz;
        const VkDeviceSize field_bytes = static_cast<VkDeviceSize>(total_cells) * sizeof(float);
        const int buf_count = input_count + output_count;

        int slots[16];
        if (!acquire_pool_slots(field_bytes, buf_count, slots))
        {
            log_vulkan_warning("failed to acquire pool slots for multi-field dispatch");
            return false;
        }
        update_pooled_descriptor_set(pipeline, slots, field_bytes, buf_count);

        const bool unified = has_unified_memory_;
        auto upload_ptr = [&](int slot) -> void*
        {
            return unified ? buffer_pool_.device(slot).mapped
                           : buffer_pool_.staging(slot).mapped;
        };

        // Upload input fields
        for (int i = 0; i < input_count; ++i)
        {
            std::memcpy(upload_ptr(slots[i]), inputs[i], field_bytes);
        }
        // Upload output data (preserves existing values for read-write buffers
        // like sedimentation tendencies; callers that want zero-initialized
        // outputs pass pre-zeroed arrays)
        for (int i = 0; i < output_count; ++i)
        {
            std::memcpy(upload_ptr(slots[input_count + i]), outputs[i], field_bytes);
        }

        // Record command buffer
        vkResetCommandBuffer_(cmd_buf_, 0);
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer_(cmd_buf_, &begin_info) != VK_SUCCESS)
        {
            release_pool_slots(slots, buf_count);
            return false;
        }

        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;

        if (unified)
        {
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
        else
        {
            // Copy all staging → device
            VkBufferCopy copy_region{};
            copy_region.size = field_bytes;
            for (int i = 0; i < buf_count; ++i)
            {
                vkCmdCopyBuffer_(cmd_buf_,
                                 buffer_pool_.staging(slots[i]).buffer,
                                 buffer_pool_.device(slots[i]).buffer,
                                 1, &copy_region);
            }

            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0, 1, &barrier, 0, nullptr, 0, nullptr);
        }

        // Bind pipeline and descriptors
        vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
        vkCmdBindDescriptorSets_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipeline.pipeline_layout, 0, 1,
                                 &pipeline.descriptor_set, 0, nullptr);

        // Push constants
        vkCmdPushConstants_(cmd_buf_, pipeline.pipeline_layout,
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0, pipeline.push_constant_size, push_constants);

        // Dispatch
        const uint32_t workgroup_count = (total_invocations + 63u) / 64u;
        vkCmdDispatch_(cmd_buf_, workgroup_count, 1, 1);

        if (unified)
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
        else
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0, 1, &barrier, 0, nullptr, 0, nullptr);

            VkBufferCopy copy_region{};
            copy_region.size = field_bytes;
            for (int i = 0; i < output_count; ++i)
            {
                vkCmdCopyBuffer_(cmd_buf_,
                                 buffer_pool_.device(slots[input_count + i]).buffer,
                                 buffer_pool_.staging(slots[input_count + i]).buffer,
                                 1, &copy_region);
            }
        }

        if (vkEndCommandBuffer_(cmd_buf_) != VK_SUCCESS)
        {
            release_pool_slots(slots, buf_count);
            return false;
        }

        // Submit and wait
        vkResetFences_(device_, 1, &fence_);
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buf_;
        if (vkQueueSubmit_(compute_queue_, 1, &submit_info, fence_) != VK_SUCCESS)
        {
            release_pool_slots(slots, buf_count);
            return false;
        }

        constexpr uint64_t kTimeoutNs = 10ULL * 1000000000ULL;
        if (vkWaitForFences_(device_, 1, &fence_, VK_TRUE, kTimeoutNs) != VK_SUCCESS)
        {
            release_pool_slots(slots, buf_count);
            return false;
        }

        // Download output fields
        for (int i = 0; i < output_count; ++i)
        {
            const void* src = unified
                ? buffer_pool_.device(slots[input_count + i]).mapped
                : buffer_pool_.staging(slots[input_count + i]).mapped;
            std::memcpy(outputs[i], src, field_bytes);
        }
        release_pool_slots(slots, buf_count);
        return true;
    }

    // ── SPIR-V loading ─────────────────────────────────────────────────

    static std::vector<uint32_t> load_spirv(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open())
        {
            return {};
        }
        const auto size = file.tellg();
        if (size <= 0 || size % 4 != 0)
        {
            return {};
        }
        std::vector<uint32_t> code(static_cast<std::size_t>(size) / 4);
        file.seekg(0);
        file.read(reinterpret_cast<char*>(code.data()), size);
        return code;
    }

    static std::string resolve_shader_path(const std::string& shader_filename)
    {
        const char* override_dir = std::getenv("TMV_SHADER_DIR");
        if (override_dir != nullptr && override_dir[0] != '\0')
        {
            return std::string(override_dir) + "/" + shader_filename;
        }
        const std::vector<std::string> candidates = {
            "vulkan/shaders/compute/" + shader_filename,
            "../vulkan/shaders/compute/" + shader_filename,
            "shaders/compute/" + shader_filename,
        };
        for (const auto& candidate : candidates)
        {
            std::ifstream test(candidate);
            if (test.good())
            {
                return candidate;
            }
        }
        return candidates[0];
    }

    // ── Pipeline factory ────────────────────────────────────────────────

    bool create_pipeline_state(const std::string& shader_spv_filename,
                               uint32_t binding_count,
                               uint32_t push_constant_size,
                               ComputePipelineState& out,
                               std::string& error)
    {
        const std::string shader_path = resolve_shader_path(shader_spv_filename);
        const std::vector<uint32_t> spirv = load_spirv(shader_path);
        if (spirv.empty())
        {
            error = "failed to load SPIR-V from '" + shader_path + "'";
            return false;
        }
        log_vulkan_info("loaded shader: " + shader_path +
                       " (" + std::to_string(spirv.size() * 4) + " bytes)");

        VkShaderModuleCreateInfo shader_info{};
        shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shader_info.codeSize = spirv.size() * sizeof(uint32_t);
        shader_info.pCode = spirv.data();

        VkShaderModule shader_module = VK_NULL_HANDLE;
        if (vkCreateShaderModule_(device_, &shader_info, nullptr, &shader_module) != VK_SUCCESS)
        {
            error = "vkCreateShaderModule failed for " + shader_spv_filename;
            return false;
        }

        // Descriptor set layout: N SSBO bindings
        std::vector<VkDescriptorSetLayoutBinding> bindings(binding_count);
        for (uint32_t i = 0; i < binding_count; ++i)
        {
            bindings[i] = {};
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = binding_count;
        layout_info.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout_(device_, &layout_info, nullptr,
                                         &out.descriptor_set_layout) != VK_SUCCESS)
        {
            vkDestroyShaderModule_(device_, shader_module, nullptr);
            error = "vkCreateDescriptorSetLayout failed for " + shader_spv_filename;
            return false;
        }

        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = push_constant_size;

        VkPipelineLayoutCreateInfo pl_info{};
        pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_info.setLayoutCount = 1;
        pl_info.pSetLayouts = &out.descriptor_set_layout;
        pl_info.pushConstantRangeCount = 1;
        pl_info.pPushConstantRanges = &push_range;

        if (vkCreatePipelineLayout_(device_, &pl_info, nullptr,
                                    &out.pipeline_layout) != VK_SUCCESS)
        {
            vkDestroyShaderModule_(device_, shader_module, nullptr);
            error = "vkCreatePipelineLayout failed for " + shader_spv_filename;
            return false;
        }

        VkComputePipelineCreateInfo pipe_info{};
        pipe_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipe_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipe_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipe_info.stage.module = shader_module;
        pipe_info.stage.pName = "main";
        pipe_info.layout = out.pipeline_layout;

        VkResult result = vkCreateComputePipelines_(
            device_, VK_NULL_HANDLE, 1, &pipe_info, nullptr, &out.pipeline);
        vkDestroyShaderModule_(device_, shader_module, nullptr);
        if (result != VK_SUCCESS)
        {
            error = "vkCreateComputePipelines failed for " + shader_spv_filename;
            return false;
        }

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = binding_count;

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;

        if (vkCreateDescriptorPool_(device_, &pool_info, nullptr,
                                    &out.descriptor_pool) != VK_SUCCESS)
        {
            error = "vkCreateDescriptorPool failed for " + shader_spv_filename;
            return false;
        }

        VkDescriptorSetAllocateInfo ds_alloc{};
        ds_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ds_alloc.descriptorPool = out.descriptor_pool;
        ds_alloc.descriptorSetCount = 1;
        ds_alloc.pSetLayouts = &out.descriptor_set_layout;

        if (vkAllocateDescriptorSets_(device_, &ds_alloc, &out.descriptor_set) != VK_SUCCESS)
        {
            error = "vkAllocateDescriptorSets failed for " + shader_spv_filename;
            return false;
        }

        out.binding_count = binding_count;
        out.push_constant_size = push_constant_size;
        return true;
    }

    void destroy_pipeline_state(ComputePipelineState& state)
    {
        if (state.pipeline != VK_NULL_HANDLE && vkDestroyPipeline_ != nullptr)
        {
            vkDestroyPipeline_(device_, state.pipeline, nullptr);
        }
        if (state.descriptor_pool != VK_NULL_HANDLE && vkDestroyDescriptorPool_ != nullptr)
        {
            vkDestroyDescriptorPool_(device_, state.descriptor_pool, nullptr);
        }
        if (state.pipeline_layout != VK_NULL_HANDLE && vkDestroyPipelineLayout_ != nullptr)
        {
            vkDestroyPipelineLayout_(device_, state.pipeline_layout, nullptr);
        }
        if (state.descriptor_set_layout != VK_NULL_HANDLE && vkDestroyDescriptorSetLayout_ != nullptr)
        {
            vkDestroyDescriptorSetLayout_(device_, state.descriptor_set_layout, nullptr);
        }
        state = {};
    }

    // ── Shared command infrastructure setup ───────────────────────────

    bool setup_command_infrastructure(std::string& error)
    {
        VkCommandPoolCreateInfo cp_info{};
        cp_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cp_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cp_info.queueFamilyIndex = compute_queue_family_index_;

        if (vkCreateCommandPool_(device_, &cp_info, nullptr, &cmd_pool_) != VK_SUCCESS)
        {
            error = "vkCreateCommandPool failed";
            return false;
        }

        VkCommandBufferAllocateInfo cb_alloc{};
        cb_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cb_alloc.commandPool = cmd_pool_;
        cb_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cb_alloc.commandBufferCount = 1;

        if (vkAllocateCommandBuffers_(device_, &cb_alloc, &cmd_buf_) != VK_SUCCESS)
        {
            error = "vkAllocateCommandBuffers failed";
            return false;
        }

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        if (vkCreateFence_(device_, &fence_info, nullptr, &fence_) != VK_SUCCESS)
        {
            error = "vkCreateFence failed";
            return false;
        }

        return true;
    }

    // ── Pipeline setup (all compute pipelines) ───────────────────────

    bool setup_compute_pipelines(std::string& error)
    {
        if (!setup_command_infrastructure(error))
        {
            return false;
        }

        // Detect unified memory before initializing the buffer pool
        detect_unified_memory();

        // Initialize the buffer pool with our create/destroy callbacks
        init_buffer_pool();

        // TVD vertical flux (required — if this fails, no GPU dispatch at all)
        if (!create_pipeline_state("tvd_vertical_flux.comp.spv", 4,
                                   sizeof(TvdPushConstants), tvd_pipeline_, error))
        {
            return false;
        }
        log_vulkan_info("pipeline ready: TVD vertical flux");

        // Radial advection (optional — missing shader is non-fatal)
        {
            std::string radial_error;
            if (create_pipeline_state("advect_radial.comp.spv", 3,
                                      sizeof(RadialAdvectionPushConstants),
                                      radial_pipeline_, radial_error))
            {
                log_vulkan_info("pipeline ready: radial advection");
            }
            else
            {
                log_vulkan_warning("radial advection pipeline unavailable: " + radial_error);
            }
        }

        // Azimuthal advection (optional)
        {
            std::string azimuthal_error;
            if (create_pipeline_state("advect_azimuthal.comp.spv", 3,
                                      sizeof(AzimuthalAdvectionPushConstants),
                                      azimuthal_pipeline_, azimuthal_error))
            {
                log_vulkan_info("pipeline ready: azimuthal advection");
            }
            else
            {
                log_vulkan_warning("azimuthal advection pipeline unavailable: " + azimuthal_error);
            }
        }

        // Diffusion (optional)
        {
            std::string diffusion_error;
            if (create_pipeline_state("diffusion.comp.spv", 2,
                                      sizeof(DiffusionPushConstants),
                                      diffusion_pipeline_, diffusion_error))
            {
                log_vulkan_info("pipeline ready: diffusion");
            }
            else
            {
                log_vulkan_warning("diffusion pipeline unavailable: " + diffusion_error);
            }
        }

        // Supercell tendencies (optional — 12 SSBOs: 7 input + 5 output)
        {
            std::string supercell_error;
            if (create_pipeline_state("supercell_tendencies.comp.spv", 12,
                                      sizeof(SupercellTendenciesPushConstants),
                                      supercell_pipeline_, supercell_error))
            {
                log_vulkan_info("pipeline ready: supercell tendencies");
            }
            else
            {
                log_vulkan_warning("supercell tendencies pipeline unavailable: " + supercell_error);
            }
        }

        // Cartesian tendencies (optional — 16 SSBOs: 11 input + 5 output)
        {
            std::string cartesian_error;
            if (create_pipeline_state("cartesian_tendencies.comp.spv", 16,
                                      sizeof(CartesianTendenciesPushConstants),
                                      cartesian_pipeline_, cartesian_error))
            {
                log_vulkan_info("pipeline ready: cartesian tendencies");
            }
            else
            {
                log_vulkan_warning("cartesian tendencies pipeline unavailable: " + cartesian_error);
            }
        }

        // Cartesian x-advection (optional — 3 SSBOs: 2 input + 1 output)
        {
            std::string advx_error;
            if (create_pipeline_state("advect_x.comp.spv", 3,
                                      sizeof(AdvectionXPushConstants),
                                      advection_x_pipeline_, advx_error))
            {
                log_vulkan_info("pipeline ready: cartesian x-advection");
            }
            else
            {
                log_vulkan_warning("cartesian x-advection pipeline unavailable: " + advx_error);
            }
        }

        // Cartesian y-advection (optional — 3 SSBOs: 2 input + 1 output)
        {
            std::string advy_error;
            if (create_pipeline_state("advect_y.comp.spv", 3,
                                      sizeof(AdvectionYPushConstants),
                                      advection_y_pipeline_, advy_error))
            {
                log_vulkan_info("pipeline ready: cartesian y-advection");
            }
            else
            {
                log_vulkan_warning("cartesian y-advection pipeline unavailable: " + advy_error);
            }
        }

        // Tornado tendencies (optional — 12 SSBOs: 7 input + 5 output)
        {
            std::string tornado_error;
            if (create_pipeline_state("tornado_tendencies.comp.spv", 12,
                                      sizeof(TornadoTendenciesPushConstants),
                                      tornado_pipeline_, tornado_error))
            {
                log_vulkan_info("pipeline ready: tornado tendencies");
            }
            else
            {
                log_vulkan_warning("tornado tendencies pipeline unavailable: " + tornado_error);
            }
        }

        /// Kessler point-wise microphysics (optional — 13 SSBOs: 7 input + 6 output)
        {
            std::string kessler_pw_error;
            if (create_pipeline_state("kessler_pointwise.comp.spv", 13,
                                      sizeof(KesslerPointwisePushConstants),
                                      kessler_pointwise_pipeline_, kessler_pw_error))
            {
                log_vulkan_info("pipeline ready: Kessler point-wise microphysics");
            }
            else
            {
                log_vulkan_warning("Kessler point-wise pipeline unavailable: " + kessler_pw_error);
            }
        }

        // Kessler sedimentation (optional — 6 SSBOs: 3 input + 3 read-write)
        {
            std::string kessler_sed_error;
            if (create_pipeline_state("kessler_sedimentation.comp.spv", 6,
                                      sizeof(KesslerSedimentationPushConstants),
                                      kessler_sedimentation_pipeline_, kessler_sed_error))
            {
                log_vulkan_info("pipeline ready: Kessler sedimentation");
            }
            else
            {
                log_vulkan_warning("Kessler sedimentation pipeline unavailable: " + kessler_sed_error);
            }
        }

        // Acoustic pressure substep (optional — 7 SSBOs: 5 input + 2 output)
        {
            std::string acoustic_p_error;
            if (create_pipeline_state("acoustic_pressure.comp.spv", 7,
                                      sizeof(AcousticPressurePushConstants),
                                      acoustic_pressure_pipeline_, acoustic_p_error))
            {
                log_vulkan_info("pipeline ready: acoustic pressure substep");
            }
            else
            {
                log_vulkan_warning("acoustic pressure pipeline unavailable: " + acoustic_p_error);
            }
        }

        // Acoustic momentum substep (optional — 8 SSBOs: 5 input + 3 output)
        {
            std::string acoustic_m_error;
            if (create_pipeline_state("acoustic_momentum.comp.spv", 8,
                                      sizeof(AcousticMomentumPushConstants),
                                      acoustic_momentum_pipeline_, acoustic_m_error))
            {
                log_vulkan_info("pipeline ready: acoustic momentum substep");
            }
            else
            {
                log_vulkan_warning("acoustic momentum pipeline unavailable: " + acoustic_m_error);
            }
        }

        return true;
    }

    void destroy_compute_resources()
    {
        destroy_buffer(staging_q_);  destroy_buffer(staging_w_);
        destroy_buffer(staging_dz_); destroy_buffer(staging_dqdt_);
        destroy_buffer(device_q_);   destroy_buffer(device_w_);
        destroy_buffer(device_dz_);  destroy_buffer(device_dqdt_);

        // Destroy pooled buffers
        buffer_pool_.destroy_all();

        if (fence_ != VK_NULL_HANDLE && vkDestroyFence_ != nullptr)
        {
            vkDestroyFence_(device_, fence_, nullptr);
            fence_ = VK_NULL_HANDLE;
        }
        if (cmd_pool_ != VK_NULL_HANDLE && vkDestroyCommandPool_ != nullptr)
        {
            vkDestroyCommandPool_(device_, cmd_pool_, nullptr);
            cmd_pool_ = VK_NULL_HANDLE;
            cmd_buf_ = VK_NULL_HANDLE;
        }

        destroy_pipeline_state(tvd_pipeline_);
        destroy_pipeline_state(radial_pipeline_);
        destroy_pipeline_state(azimuthal_pipeline_);
        destroy_pipeline_state(diffusion_pipeline_);
        destroy_pipeline_state(supercell_pipeline_);
        destroy_pipeline_state(tornado_pipeline_);
        destroy_pipeline_state(kessler_pointwise_pipeline_);
        destroy_pipeline_state(kessler_sedimentation_pipeline_);
    }

    // ── Member variables ───────────────────────────────────────────────

    void* loader_handle_ = nullptr;
    uint32_t loader_api_version_ = VK_API_VERSION_1_0;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    uint32_t compute_queue_family_index_ = std::numeric_limits<uint32_t>::max();

    int selected_device_index_ = -1;
    VkPhysicalDeviceProperties selected_device_properties_{};
    VkPhysicalDeviceFeatures selected_device_features_{};
    VkQueueFamilyProperties selected_compute_queue_family_{};
    bool selected_supports_portability_subset_ = false;

    /// True when the device has a memory type with both HOST_VISIBLE and DEVICE_LOCAL.
    /// On Apple Silicon and integrated GPUs, this enables zero-copy buffer access
    /// (no staging buffers, no H2D/D2H transfers).
    bool has_unified_memory_ = false;

    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_ = nullptr;
    PFN_vkCreateInstance vkCreateInstance_ = nullptr;
    PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion_ = nullptr;
    PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties_ = nullptr;
    PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerProperties_ = nullptr;

    PFN_vkDestroyInstance vkDestroyInstance_ = nullptr;
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices_ = nullptr;
    PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties_ = nullptr;
    PFN_vkGetPhysicalDeviceFeatures vkGetPhysicalDeviceFeatures_ = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties_ = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties_ = nullptr;
    PFN_vkCreateDevice vkCreateDevice_ = nullptr;
    PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr_ = nullptr;

    PFN_vkDestroyDevice vkDestroyDevice_ = nullptr;
    PFN_vkGetDeviceQueue vkGetDeviceQueue_ = nullptr;
    PFN_vkDeviceWaitIdle vkDeviceWaitIdle_ = nullptr;

    // Instance-level (for memory properties)
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties_ = nullptr;

    // Buffer and memory management
    PFN_vkCreateBuffer vkCreateBuffer_ = nullptr;
    PFN_vkDestroyBuffer vkDestroyBuffer_ = nullptr;
    PFN_vkAllocateMemory vkAllocateMemory_ = nullptr;
    PFN_vkFreeMemory vkFreeMemory_ = nullptr;
    PFN_vkBindBufferMemory vkBindBufferMemory_ = nullptr;
    PFN_vkMapMemory vkMapMemory_ = nullptr;
    PFN_vkUnmapMemory vkUnmapMemory_ = nullptr;
    PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements_ = nullptr;

    // Shader and pipeline
    PFN_vkCreateShaderModule vkCreateShaderModule_ = nullptr;
    PFN_vkDestroyShaderModule vkDestroyShaderModule_ = nullptr;
    PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout_ = nullptr;
    PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout_ = nullptr;
    PFN_vkCreateDescriptorPool vkCreateDescriptorPool_ = nullptr;
    PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool_ = nullptr;
    PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets_ = nullptr;
    PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets_ = nullptr;
    PFN_vkCreatePipelineLayout vkCreatePipelineLayout_ = nullptr;
    PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout_ = nullptr;
    PFN_vkCreateComputePipelines vkCreateComputePipelines_ = nullptr;
    PFN_vkDestroyPipeline vkDestroyPipeline_ = nullptr;

    // Command buffers
    PFN_vkCreateCommandPool vkCreateCommandPool_ = nullptr;
    PFN_vkDestroyCommandPool vkDestroyCommandPool_ = nullptr;
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers_ = nullptr;
    PFN_vkBeginCommandBuffer vkBeginCommandBuffer_ = nullptr;
    PFN_vkEndCommandBuffer vkEndCommandBuffer_ = nullptr;
    PFN_vkResetCommandBuffer vkResetCommandBuffer_ = nullptr;
    PFN_vkCmdBindPipeline vkCmdBindPipeline_ = nullptr;
    PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets_ = nullptr;
    PFN_vkCmdPushConstants vkCmdPushConstants_ = nullptr;
    PFN_vkCmdDispatch vkCmdDispatch_ = nullptr;
    PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier_ = nullptr;
    PFN_vkCmdCopyBuffer vkCmdCopyBuffer_ = nullptr;

    // Synchronization
    PFN_vkQueueSubmit vkQueueSubmit_ = nullptr;
    PFN_vkCreateFence vkCreateFence_ = nullptr;
    PFN_vkDestroyFence vkDestroyFence_ = nullptr;
    PFN_vkWaitForFences vkWaitForFences_ = nullptr;
    PFN_vkResetFences vkResetFences_ = nullptr;

    // Compute pipelines (one per kernel type)
    ComputePipelineState tvd_pipeline_{};
    ComputePipelineState radial_pipeline_{};
    ComputePipelineState azimuthal_pipeline_{};
    ComputePipelineState diffusion_pipeline_{};
    ComputePipelineState supercell_pipeline_{};
    ComputePipelineState cartesian_pipeline_{};
    ComputePipelineState advection_x_pipeline_{};
    ComputePipelineState advection_y_pipeline_{};
    ComputePipelineState tornado_pipeline_{};
    ComputePipelineState kessler_pointwise_pipeline_{};
    ComputePipelineState kessler_sedimentation_pipeline_{};
    ComputePipelineState acoustic_pressure_pipeline_{};
    ComputePipelineState acoustic_momentum_pipeline_{};

    // Shared command infrastructure
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_buf_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;

    // GPU buffers for TVD vertical flux (4 bindings: q, w, dz, dqdt)
    GpuBuffer staging_q_{};
    GpuBuffer staging_w_{};
    GpuBuffer staging_dz_{};
    GpuBuffer staging_dqdt_{};
    GpuBuffer device_q_{};
    GpuBuffer device_w_{};
    GpuBuffer device_dz_{};
    GpuBuffer device_dqdt_{};

    // Pooled GPU buffer pairs for multi-field dispatch (radial, azimuthal,
    // diffusion, supercell, kessler, batched advection).
    tmv_vulkan::GpuBufferPool buffer_pool_;
#endif
};

} // namespace

std::unique_ptr<ComputeBackend> create_vulkan_compute_backend()
{
    return std::make_unique<VulkanComputeBackend>();
}
