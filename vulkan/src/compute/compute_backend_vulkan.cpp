#ifndef VK_ENABLE_BETA_EXTENSIONS
#define VK_ENABLE_BETA_EXTENSIONS
#endif
#include "compute/acoustic_cgrid_dispatch_contract.hpp"
#include "util/packaged_runtime.hpp"
#include "compute/domain_recycling_contract.hpp"
#include "compute/cartesian_scalar_capture.hpp"
#include "compute/lin_contract.hpp"
#include "compute/kessler_sedimentation_contract.hpp"
// Vulkan resource ownership, pipeline dispatch, residency, and telemetry.
// Backend selection and fallback remain in compute_backend.cpp.

#include "compute/compute_backend.hpp"
#include "core/runtime/simulation_state.hpp"
#include "compute/compute_backend_factory.hpp"
#include "compute/dispatch_graph.hpp"
#include "compute/kernel_abi.hpp"
#include "compute/kernel_manifest.hpp"
#include "compute/kernel_startup_plan.hpp"
#include "compute/thompson_sedimentation_contract.hpp"
#include "compute/thompson_cloud_contract.hpp"
#include "compute/thompson_warm_contract.hpp"
#include "compute/milbrandt_warm_contract.hpp"
#include "compute/milbrandt_cloud_contract.hpp"
#include "compute/rk_combination_contract.hpp"
#include "compute/rk_tendency_contract.hpp"
#include "compute/collocated_sgs_contract.hpp"
#include "compute/conservative_remap_contract.hpp"
#include "compute/earth_geometry_contract.hpp"
#include "compute/conservative_reduction_contract.hpp"
#include "compute/conservative_momentum_contract.hpp"
#include "compute/moving_momentum_contract.hpp"
#include "compute/moving_energy_contract.hpp"
#include "compute/reference_atmosphere_contract.hpp"
#include "compute/column_remap_contract.hpp"
#include "compute/held_sources_contract.hpp"
#include "compute/cgrid_boundary_contract.hpp"
#include "compute/simple_grey_contract.hpp"
#include "compute/slab_column_contract.hpp"
#include "compute/mapped_boundary_contract.hpp"
#include "compute/milbrandt_sedimentation_contract.hpp"
#include "compute/milbrandt_cold_contract.hpp"
#include "compute/milbrandt_melting_contract.hpp"
#include "compute/thompson_phase_contract.hpp"
#include "compute/thompson_freezing_contract.hpp"
#include "compute/thompson_deposition_contract.hpp"
#include "compute/thompson_completion_contract.hpp"
#include "compute/tke_dispatch_contract.hpp"
#include "compute/implicit_diffusion_contract.hpp"
#include "compute/resident_field_store.hpp"
#include "compute/vulkan_submission_lifecycle.hpp"
#include "core/infra/hardware_info.hpp"
#include "numerics/advection/advection_cartesian.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/runtime/process_state.hpp"
#include "compute/gpu_buffer_pool.hpp"
#include "util/parallel_copy.hpp"
#include "compute/vulkan_compute_telemetry.hpp"
#include "compute/vulkan_descriptor_cache.hpp"
#include "compute/vulkan_device_context.hpp"
#include "compute/vulkan_dispatch_policy.hpp"
#include "compute/vulkan_pipeline_cache_file.hpp"
#include "compute/vulkan_pipeline_cache_signature.hpp"
#include "compute/generated_division_variants.hpp"
#include "compute/vulkan_pipeline_registry.hpp"
#include "compute/vulkan_cuda_kernel.hpp"
#include "compute/vulkan_residency_policy.hpp"
#include "compute/vulkan_loader_policy.hpp"
#include "util/log_profile.hpp"

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
#include <unistd.h> // readlink, ssize_t (executable-relative shader resolution)
#define TMV_HAS_VULKAN_COMPUTE_DLOPEN 1
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define TMV_HAS_VULKAN_COMPUTE_DLOPEN 1
#endif
#if defined(__APPLE__)
#include <mach-o/dyld.h> // _NSGetExecutablePath
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

using namespace nimbus::compute::abi;

bool distinct_field_destinations(std::initializer_list<float*> destinations,
                                 std::size_t bytes) noexcept
{
    // Input/output aliasing is safe because every stage finishes into private
    // candidates before publication. Two output ranges cannot share storage:
    // sequential publication would otherwise make field ownership ambiguous.
    if (bytes == 0)
        return false;
    for (auto first = destinations.begin(); first != destinations.end(); ++first)
    {
        if (*first == nullptr)
            return false;
        const auto first_address = reinterpret_cast<std::uintptr_t>(*first);
        for (auto second = std::next(first); second != destinations.end(); ++second)
        {
            if (*second == nullptr)
                return false;
            const auto second_address = reinterpret_cast<std::uintptr_t>(*second);
            const auto distance = first_address < second_address
                ? second_address - first_address : first_address - second_address;
            if (distance < bytes)
                return false;
        }
    }
    return true;
}

#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
bool open_dynamic_library(const char* path, void*& library, std::string& error)
{
    return tmv_vulkan::loader_policy::open_with_lifetime_policy(
        [&]
        {
#if defined(_WIN32)
            library = reinterpret_cast<void*>(LoadLibraryA(path));
#else
            library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
        },
        error);
}

void* load_dynamic_symbol(void* library, const char* name)
{
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(library), name));
#else
    return dlsym(library, name);
#endif
}

void close_dynamic_library(void* library)
{
#if defined(_WIN32)
    FreeLibrary(reinterpret_cast<HMODULE>(library));
#else
    dlclose(library);
#endif
}
#endif

class VulkanComputeBackend final : public ComputeBackend,
                                   public AdvectionComputeKernels,
                                   public DynamicsComputeKernels,
                                   public MicrophysicsComputeKernels,
                                   public TurbulenceComputeKernels,
                                   public DiffusionComputeKernels,
                                   public BoundaryLayerComputeKernels,
                                   public RadiationComputeKernels,
                                   public TerrainComputeKernels
{
public:
    using KernelId = nimbus::compute::KernelId;

    std::string name() const override
    {
        return "vulkan";
    }

    std::uint64_t successful_dispatch_count() const override
    {
        return successful_dispatch_count_;
    }

    std::uint64_t resident_upload_count() const override
    {
        return compute_telemetry_.resident_misses;
    }

    std::uint64_t resident_reuse_count() const override
    {
        return compute_telemetry_.resident_hits;
    }

    std::uint64_t successful_kernel_dispatch_count(KernelId kernel) const override
    {
        const std::size_t index = static_cast<std::size_t>(kernel);
        return index < successful_kernel_dispatch_counts_.size() ? successful_kernel_dispatch_counts_[index] : 0;
    }

    void configure(const ComputeBackendConfig& config) override
    {
        requested_config_ = config;
    }

    void set_coordinate_system(CoordinateSystem cs) override
    {
        coord_system_ = cs;
    }

    void populate_hardware_info(HardwareInfo& info) const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // GPU fields keep their CPU-only defaults until a logical device
        // actually exists (initialize() succeeded), so the hardware banner
        // never claims a GPU the run cannot use.
        if (device_context_.device == VK_NULL_HANDLE)
            return;
        info.gpu_available   = true;
        info.gpu_device_name = device_context_.selected_device_properties.deviceName;
#else
        (void)info;
#endif
    }

    ComputeMemoryProperties memory_properties() const override
    {
        ComputeMemoryProperties result;
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!device_context_.available() || vkGetPhysicalDeviceMemoryProperties_ == nullptr)
        {
            return result;
        }
        result.accelerator                = true;
        result.unified_host_device_memory = device_context_.has_unified_memory;
        result.max_storage_buffer_bytes =
            static_cast<std::uint64_t>(device_context_.selected_device_properties.limits.maxStorageBufferRange);

        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties_(device_context_.physical_device, &memory);
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
        budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        if (memory_budget_enabled_ && vkGetPhysicalDeviceMemoryProperties2_ != nullptr)
        {
            VkPhysicalDeviceMemoryProperties2 properties{};
            properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
            properties.pNext = &budget;
            vkGetPhysicalDeviceMemoryProperties2_(device_context_.physical_device, &properties);
            memory = properties.memoryProperties;
            result.device_budget_is_live = true;
        }
        const auto working_type = device_context_.memory_plan.working_memory_type;
        const auto working_heap = working_type < memory.memoryTypeCount
            ? memory.memoryTypes[working_type].heapIndex : memory.memoryHeapCount;
        for (std::uint32_t heap = 0; heap < memory.memoryHeapCount; ++heap)
        {
            if ((memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0 ||
                (working_heap < memory.memoryHeapCount && heap != working_heap))
            {
                continue;
            }
            const std::uint64_t bytes = static_cast<std::uint64_t>(memory.memoryHeaps[heap].size);
            if (result.device_local_heap_bytes > std::numeric_limits<std::uint64_t>::max() - bytes)
            {
                result.device_local_heap_bytes = std::numeric_limits<std::uint64_t>::max();
                break;
            }
            result.device_local_heap_bytes += bytes;
            if (result.device_budget_is_live)
            {
                const auto limit = std::min(bytes, static_cast<std::uint64_t>(budget.heapBudget[heap]));
                const auto used = static_cast<std::uint64_t>(budget.heapUsage[heap]);
                const auto available = limit > used ? limit - used : 0;
                result.device_available_budget_bytes = available >
                    std::numeric_limits<std::uint64_t>::max() - result.device_available_budget_bytes
                    ? std::numeric_limits<std::uint64_t>::max()
                    : result.device_available_budget_bytes + available;
            }
        }
#endif
        return result;
    }

    AdvectionComputeKernels* advection_kernels() override
    {
        return this;
    }
    const AdvectionComputeKernels* advection_kernels() const override
    {
        return this;
    }

    DynamicsComputeKernels* dynamics_kernels() override
    {
        return this;
    }
    const DynamicsComputeKernels* dynamics_kernels() const override
    {
        return this;
    }

    MicrophysicsComputeKernels* microphysics_kernels() override
    {
        return this;
    }
    const MicrophysicsComputeKernels* microphysics_kernels() const override
    {
        return this;
    }

    TurbulenceComputeKernels* turbulence_kernels() override
    {
        return this;
    }
    const TurbulenceComputeKernels* turbulence_kernels() const override
    {
        return this;
    }

    DiffusionComputeKernels* diffusion_kernels() override
    {
        return this;
    }
    const DiffusionComputeKernels* diffusion_kernels() const override
    {
        return this;
    }

    BoundaryLayerComputeKernels* boundary_layer_kernels() override
    {
        return this;
    }
    const BoundaryLayerComputeKernels* boundary_layer_kernels() const override
    {
        return this;
    }

    RadiationComputeKernels* radiation_kernels() override
    {
        return this;
    }
    const RadiationComputeKernels* radiation_kernels() const override
    {
        return this;
    }

    TerrainComputeKernels* terrain_kernels() override
    {
        return this;
    }
    const TerrainComputeKernels* terrain_kernels() const override
    {
        return this;
    }

    bool supports_mapped_mass_continuity() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return pipeline(KernelId::MappedTerrainMassContinuity).is_ready();
#else
        return false;
#endif
    }

    bool dispatch_mapped_mass_continuity(const MappedTerrainMassContinuityDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        const bool pointers = d.rho && d.u && d.v && d.w && d.p && d.jacobian &&
            d.jacobian_x_positive_face && d.jacobian_y_positive_face &&
            d.metric_x_positive_zeta_face && d.metric_y_positive_zeta_face &&
            d.metric_x_ground_face && d.metric_y_ground_face && d.ground_physical_w && d.drho_dt && d.dp_dt &&
            d.mass_flux_x_positive && d.mass_flux_y_positive && d.mass_flux_zeta_positive && d.report;
        if (!supports_mapped_mass_continuity() || !pointers || d.nx < 3 || d.ny < 3 || d.nz < 1 ||
            !std::isfinite(d.dx) || d.dx <= 0.0f || !std::isfinite(d.dy) || d.dy <= 0.0f ||
            !std::isfinite(d.dzeta) || d.dzeta <= 0.0f || !std::isfinite(d.gamma) || d.gamma <= 1.0f ||
            !std::isfinite(d.frame_u_ms) || !std::isfinite(d.frame_v_ms))
        {
            return false;
        }
        std::uint32_t total_points = 0;
        VkDeviceSize field_bytes = 0;
        if (!checked_field_dispatch(d.nx, d.ny, d.nz, total_points, field_bytes,
                                    "mapped terrain mass continuity"))
            return false;
        const std::size_t ncell = total_points;
        const std::size_t bytes = static_cast<std::size_t>(field_bytes);
        if (!distinct_field_destinations(
                {d.ground_physical_w, d.drho_dt, d.dp_dt, d.mass_flux_x_positive,
                 d.mass_flux_y_positive, d.mass_flux_zeta_positive}, bytes))
            return false;
        try
        {
            for (std::size_t index = 0; index < ncell; ++index)
                if (!std::isfinite(d.rho[index]) || d.rho[index] <= 0.0f || !std::isfinite(d.u[index]) ||
                    !std::isfinite(d.v[index]) || !std::isfinite(d.w[index]) || !std::isfinite(d.p[index]))
                    return false;
            for (std::size_t index = 0; index < ncell; ++index)
                if (!std::isfinite(d.jacobian[index]) || d.jacobian[index] <= 0.0f)
                    return false;

            MappedTerrainMassContinuityPushConstants pc{};
            pc.nx = d.nx;
            pc.ny = d.ny;
            pc.nz = d.nz;
            pc.dx = d.dx;
            pc.dy = d.dy;
            pc.dzeta = d.dzeta;
            pc.gamma = d.gamma;
            pc.frame_u_ms = d.frame_u_ms;
            pc.frame_v_ms = d.frame_v_ms;

            const auto run = [&](int stage, std::initializer_list<const float*> inputs,
                                 std::initializer_list<float*> outputs)
            {
                const int input_count = static_cast<int>(inputs.size());
                const int output_count = static_cast<int>(outputs.size());
                std::array<const float*, 18> input_fields{};
                std::array<float*, 18> output_fields{};
                std::copy(inputs.begin(), inputs.end(), input_fields.begin());
                std::copy(outputs.begin(), outputs.end(), output_fields.begin());
                std::array<std::uint8_t, 18> descriptor_order{};
                const int logical_count = input_count + output_count;
                for (int binding = 0; binding < 18; ++binding)
                    descriptor_order[static_cast<std::size_t>(binding)] =
                        static_cast<std::uint8_t>(binding < logical_count ? binding : 0);
                pc.stage = stage;
                return dispatch_multi_field_kernel(pipeline(KernelId::MappedTerrainMassContinuity),
                                                   KernelId::MappedTerrainMassContinuity,
                                                   &pc,
                                                   input_fields.data(),
                                                   input_count,
                                                   output_fields.data(),
                                                   output_count,
                                                   d.nx,
                                                   d.ny,
                                                   d.nz,
                                                   static_cast<std::uint32_t>(ncell),
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   true,
                                                   descriptor_order.data(),
                                                   nullptr,
                                                   nullptr,
                                                   18);
            };
            std::vector<float> ground_physical_w(ncell), drho(ncell), dp(ncell),
                flux_x(ncell), flux_y(ncell), flux_zeta(ncell);
            if (!run(0, {d.u, d.v, d.w, d.metric_x_ground_face, d.metric_y_ground_face},
                     {ground_physical_w.data()}))
                return false;
            if (!run(1, {d.rho, d.u, d.v, d.w, d.p, d.jacobian,
                         d.jacobian_x_positive_face, d.jacobian_y_positive_face,
                         d.metric_x_positive_zeta_face, d.metric_y_positive_zeta_face},
                     {drho.data(), dp.data(), flux_x.data(), flux_y.data(), flux_zeta.data()}))
                return false;
            for (const auto* field : {&ground_physical_w, &drho, &dp, &flux_x, &flux_y, &flux_zeta})
                if (!std::all_of(field->begin(), field->end(), [](float value) { return std::isfinite(value); }))
                    return false;
            if (d.completed_observer != nullptr)
                d.completed_observer(d.completed_observer_context,
                                     ground_physical_w.data(),
                                     drho.data(),
                                     dp.data(),
                                     flux_x.data(),
                                     flux_y.data(),
                                     flux_zeta.data());
            for (const auto& copy : {std::pair<float*, const std::vector<float>*>(d.ground_physical_w, &ground_physical_w),
                     {d.drho_dt, &drho}, {d.dp_dt, &dp}, {d.mass_flux_x_positive, &flux_x},
                     {d.mass_flux_y_positive, &flux_y}, {d.mass_flux_zeta_positive, &flux_zeta}})
                std::memcpy(copy.first, copy.second->data(), bytes);
            *d.report = {1u, 1u, true};
            return true;
        }
        catch (const std::bad_alloc&)
        {
            return false;
        }
        catch (const std::length_error&)
        {
            return false;
        }
#else
        (void)d;
        return false;
#endif
    }

    bool supports_mapped_fast_momentum() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return pipeline(KernelId::MappedTerrainFastMomentum).is_ready();
#else
        return false;
#endif
    }

    bool dispatch_mapped_fast_momentum(const MappedTerrainFastMomentumDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!supports_mapped_fast_momentum() || !d.rho || !d.p || !d.physical_height ||
            !d.reference_pressure || !d.du_dt || !d.dv_dt || !d.dw_dt || !d.report ||
            d.nx < 3 || d.ny < 3 || d.nz < 2 || !std::isfinite(d.dx) || d.dx <= 0.0f ||
            !std::isfinite(d.dy) || d.dy <= 0.0f || !std::isfinite(d.nominal_dz) ||
            d.nominal_dz <= 0.0f)
        {
            return false;
        }
        std::uint32_t total_points = 0;
        VkDeviceSize field_bytes = 0;
        if (!checked_field_dispatch(d.nx, d.ny, d.nz, total_points, field_bytes,
                                    "mapped terrain fast momentum"))
            return false;
        const std::size_t ncell = total_points;
        const std::size_t bytes = static_cast<std::size_t>(field_bytes);
        if (!distinct_field_destinations({d.du_dt, d.dv_dt, d.dw_dt}, bytes))
            return false;
        for (std::size_t index = 0; index < ncell; ++index)
            if (!std::isfinite(d.rho[index]) || d.rho[index] <= 1.0e-6f ||
                !std::isfinite(d.p[index]) || !std::isfinite(d.physical_height[index]))
                return false;
        for (int k = 0; k < d.nz; ++k)
            if (!std::isfinite(d.reference_pressure[k]) || d.reference_pressure[k] <= 0.0f)
                return false;

        if (d.sampled_reference_state)
        {
            if (ncell > (std::numeric_limits<std::uint32_t>::max() - std::size_t(d.nz)) / 3)
                return false;
            for (std::size_t n = 0; n < ncell; ++n)
                for (std::size_t field = 0; field < 3; ++field)
                {
                    const float value = d.reference_pressure[d.nz + 3 * n + field];
                    if (!std::isfinite(value) || (field < 2 ? value <= 0 : value < 0))
                        return false;
                }
        }
        try
        {
            MappedTerrainFastMomentumPushConstants pc{};
            pc.sampled_reference = d.sampled_reference_state ? 1 : 0;
            pc.nx = d.nx;
            pc.ny = d.ny;
            pc.nz = d.nz;
            pc.dx = d.dx;
            pc.dy = d.dy;
            pc.nominal_dz = d.nominal_dz;
            std::vector<float> du(ncell), dv(ncell), dw(ncell);
            const float* inputs[] = {d.rho, d.p, d.physical_height, d.reference_pressure};
            float* outputs[] = {du.data(), dv.data(), dw.data()};
            const VkDeviceSize input_bytes[] = {
                field_bytes, field_bytes, field_bytes,
                (static_cast<VkDeviceSize>(d.nz) + (d.sampled_reference_state ? 3 * ncell : 0)) * sizeof(float),
            };
            const bool write_only[] = {true, true, true};
            if (!dispatch_multi_field_kernel(pipeline(KernelId::MappedTerrainFastMomentum),
                                             KernelId::MappedTerrainFastMomentum,
                                             &pc,
                                             inputs,
                                             4,
                                             outputs,
                                             3,
                                             d.nx,
                                             d.ny,
                                             d.nz,
                                             total_points,
                                             input_bytes,
                                             nullptr,
                                             write_only))
                return false;
            for (const auto* field : {&du, &dv, &dw})
                if (!std::all_of(field->begin(), field->end(),
                                 [](float value) { return std::isfinite(value); }))
                    return false;
            if (d.completed_observer)
                d.completed_observer(d.completed_observer_context,
                                     du.data(), dv.data(), dw.data());
            std::memcpy(d.du_dt, du.data(), bytes);
            std::memcpy(d.dv_dt, dv.data(), bytes);
            std::memcpy(d.dw_dt, dw.data(), bytes);
            *d.report = {1u, true};
            return true;
        }
        catch (const std::bad_alloc&)
        {
            return false;
        }
        catch (const std::length_error&)
        {
            return false;
        }
#else
        (void)d;
        return false;
#endif
    }

    bool supports_mapped_slow_tendencies() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return pipeline(KernelId::MappedTerrainSlowTendencies).is_ready();
#else
        return false;
#endif
    }

    bool dispatch_mapped_slow_tendencies(const MappedTerrainSlowTendenciesDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        const bool pointers = d.u && d.v && d.w && d.rho && d.p && d.physical_height &&
            d.jacobian_x_positive_face && d.jacobian_y_positive_face &&
            d.metric_x_positive_zeta_face && d.metric_y_positive_zeta_face &&
            d.ground_physical_w && d.reference_pressure && d.reference_theta &&
            (!d.has_moisture || (d.qv && d.reference_qv)) &&
            (!d.has_loading || d.hydrometeor_loading) && d.du_dt && d.dv_dt && d.dw_dt &&
            d.drho_dt && d.dp_dt && d.du_advection_dt && d.dv_advection_dt &&
            d.dw_advection_dt && d.report;
        if (!supports_mapped_slow_tendencies() || !pointers || d.nx < 3 || d.ny < 3 || d.nz < 2 ||
            !std::isfinite(d.dx) || d.dx <= 0.0f || !std::isfinite(d.dy) || d.dy <= 0.0f ||
            !std::isfinite(d.dzeta) || d.dzeta <= 0.0f || !std::isfinite(d.nominal_dz) ||
            d.nominal_dz <= 0.0f || !std::isfinite(d.gravity) || d.gravity <= 0.0f)
            return false;
        std::uint32_t total_points = 0;
        VkDeviceSize field_bytes = 0;
        if (!checked_field_dispatch(d.nx, d.ny, d.nz, total_points, field_bytes,
                                    "mapped terrain slow tendencies"))
            return false;
        const std::size_t ncell = total_points;
        const std::size_t bytes = static_cast<std::size_t>(field_bytes);
        if (!distinct_field_destinations(
                {d.du_dt, d.dv_dt, d.dw_dt, d.drho_dt, d.dp_dt,
                 d.du_advection_dt, d.dv_advection_dt, d.dw_advection_dt}, bytes))
            return false;
        for (std::size_t index = 0; index < ncell; ++index)
        {
            const float values[] = {d.u[index], d.v[index], d.w[index], d.rho[index], d.p[index],
                d.physical_height[index], d.jacobian_x_positive_face[index],
                d.jacobian_y_positive_face[index], d.metric_x_positive_zeta_face[index],
                d.metric_y_positive_zeta_face[index], d.ground_physical_w[index]};
            const int j = static_cast<int>((index / static_cast<std::size_t>(d.nz)) %
                                           static_cast<std::size_t>(d.ny));
            const int i = static_cast<int>(index /
                (static_cast<std::size_t>(d.ny) * static_cast<std::size_t>(d.nz)));
            const bool active_column = i > 0 && i < d.nx - 1 && j > 0 && j < d.ny - 1;
            if (!std::all_of(std::begin(values), std::end(values),
                             [](float value) { return std::isfinite(value); }) ||
                d.rho[index] <= 1.0e-6f ||
                (active_column && (d.jacobian_x_positive_face[index] <= 0.0f ||
                                   d.jacobian_y_positive_face[index] <= 0.0f)) ||
                (d.has_moisture && (!std::isfinite(d.qv[index]) || d.qv[index] < 0.0f)) ||
                (d.has_loading && (!std::isfinite(d.hydrometeor_loading[index]) ||
                                   d.hydrometeor_loading[index] < 0.0f)))
                return false;
        }
        for (int k = 0; k < d.nz; ++k)
            if (!std::isfinite(d.reference_pressure[k]) || d.reference_pressure[k] <= 0.0f ||
                !std::isfinite(d.reference_theta[k]) || d.reference_theta[k] <= 0.0f ||
                (d.has_moisture && (!std::isfinite(d.reference_qv[k]) || d.reference_qv[k] < 0.0f)))
                return false;

        if (d.sampled_reference_state)
        {
            if (ncell > (std::numeric_limits<std::uint32_t>::max() - std::size_t(d.nz)) / 3)
                return false;
            for (std::size_t n = 0; n < ncell; ++n)
                for (std::size_t field = 0; field < 3; ++field)
                {
                    const float value = d.reference_pressure[d.nz + 3 * n + field];
                    if (!std::isfinite(value) || (field < 2 ? value <= 0 : value < 0))
                        return false;
                }
        }
        try
        {
            MappedTerrainSlowTendenciesPushConstants pc{};
            pc.sampled_reference = d.sampled_reference_state ? 1 : 0;
            pc.nx = d.nx; pc.ny = d.ny; pc.nz = d.nz;
            pc.has_moisture = d.has_moisture ? 1 : 0;
            pc.has_loading = d.has_loading ? 1 : 0;
            pc.dx = d.dx; pc.dy = d.dy; pc.dzeta = d.dzeta;
            pc.nominal_dz = d.nominal_dz; pc.gravity = d.gravity;
            std::array<std::vector<float>, 8> candidate;
            for (auto& field : candidate) field.resize(ncell);
            std::vector<float> disabled_field;
            std::vector<float> disabled_profile;
            if (!d.has_moisture || !d.has_loading)
                disabled_field.assign(ncell, 0.0f);
            if (!d.has_moisture)
                disabled_profile.assign(static_cast<std::size_t>(d.nz), 0.0f);
            const float* qv = d.has_moisture ? d.qv : disabled_field.data();
            const float* qv0 = d.has_moisture ? d.reference_qv : disabled_profile.data();
            const float* loading = d.has_loading ? d.hydrometeor_loading : disabled_field.data();
            const float* inputs[] = {d.u, d.v, d.w, d.rho, d.p, d.physical_height,
                d.jacobian_x_positive_face, d.jacobian_y_positive_face,
                d.metric_x_positive_zeta_face, d.metric_y_positive_zeta_face,
                d.ground_physical_w, d.reference_pressure, d.reference_theta,
                qv, qv0, loading};
            float* outputs[] = {candidate[0].data(), candidate[1].data(), candidate[2].data(),
                candidate[3].data(), candidate[4].data(), candidate[5].data(),
                candidate[6].data(), candidate[7].data()};
            VkDeviceSize input_bytes[16];
            std::fill(std::begin(input_bytes), std::end(input_bytes), field_bytes);
            const VkDeviceSize profile_bytes = static_cast<VkDeviceSize>(d.nz) * sizeof(float);
            input_bytes[11] = profile_bytes + (d.sampled_reference_state ? 3 * field_bytes : 0);
            input_bytes[12] = profile_bytes;
            input_bytes[14] = profile_bytes;
            const bool write_only[8] = {true, true, true, true, true, true, true, true};
            if (!dispatch_multi_field_kernel(pipeline(KernelId::MappedTerrainSlowTendencies),
                                             KernelId::MappedTerrainSlowTendencies, &pc,
                                             inputs, 16, outputs, 8, d.nx, d.ny, d.nz,
                                             total_points, input_bytes, nullptr, write_only))
                return false;
            for (const auto& field : candidate)
                if (!std::all_of(field.begin(), field.end(),
                                 [](float value) { return std::isfinite(value); }))
                    return false;
            if (d.completed_observer)
                d.completed_observer(d.completed_observer_context,
                    candidate[0].data(), candidate[1].data(), candidate[2].data(),
                    candidate[3].data(), candidate[4].data(), candidate[5].data(),
                    candidate[6].data(), candidate[7].data());
            float* destinations[] = {d.du_dt, d.dv_dt, d.dw_dt, d.drho_dt, d.dp_dt,
                d.du_advection_dt, d.dv_advection_dt, d.dw_advection_dt};
            for (std::size_t index = 0; index < candidate.size(); ++index)
                std::memcpy(destinations[index], candidate[index].data(), bytes);
            *d.report = {1u, true, true};
            return true;
        }
        catch (const std::bad_alloc&)
        {
            return false;
        }
        catch (const std::length_error&)
        {
            return false;
        }
#else
        (void)d;
        return false;
#endif
    }

    bool supports_mapped_scalar_transport() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return pipeline(KernelId::AdvectConservativeScalarFluxCartesianCgrid).is_ready() &&
               pipeline(KernelId::MappedTerrainScalarLimiter).is_ready() &&
               pipeline(KernelId::MappedTerrainScalarUpdate).is_ready();
#else
        return false;
#endif
    }

    bool dispatch_mapped_scalar_transport(const MappedTerrainScalarTransportDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        const bool pointers = d.scalar_begin && d.density_begin && d.density_end && d.mass_flux_x && d.mass_flux_y &&
                              d.mass_flux_zeta && d.jacobian && d.scalar_end && d.report;
        if ((d.preserve_positive_support && !d.positive_definite) || !supports_mapped_scalar_transport() || !pointers ||
            d.nx < 3 || d.ny < 3 || d.nz < 2 || !std::isfinite(d.dx) || d.dx <= 0.0f || !std::isfinite(d.dy) ||
            d.dy <= 0.0f || !std::isfinite(d.dzeta) || d.dzeta <= 0.0f || !std::isfinite(d.dt) || d.dt <= 0.0f ||
            !std::isfinite(d.density_floor) || d.density_floor <= 0.0f)
            return false;
        std::uint32_t total_points = 0;
        VkDeviceSize field_bytes = 0;
        if (!checked_field_dispatch(d.nx, d.ny, d.nz, total_points, field_bytes,
                                    "mapped terrain scalar transport"))
            return false;
        const std::size_t ncell = total_points;
        for (std::size_t index = 0; index < ncell; ++index)
        {
            const float values[] = {d.scalar_begin[index], d.density_begin[index], d.density_end[index],
                d.mass_flux_x[index], d.mass_flux_y[index], d.mass_flux_zeta[index], d.jacobian[index]};
            if (!std::all_of(std::begin(values), std::end(values),
                             [](float value) { return std::isfinite(value); }) ||
                d.density_begin[index] <= 0.0f || d.density_end[index] <= 0.0f ||
                d.jacobian[index] <= 0.0f)
                return false;
        }

        try
        {
            std::array<std::vector<float>, 3> scalar_flux;
            for (auto& field : scalar_flux)
                field.resize(ncell);
            AdvectFluxDimsCartesianCgridPushConstants flux_pc{};
            flux_pc.nx = d.nx;
            flux_pc.ny = d.ny;
            flux_pc.nz                  = d.nz;
            flux_pc.pad0                = (d.weno ? 0 : 1) | (d.preserve_positive_support ? 2 : 0);
            const float* flux_inputs[]  = {d.scalar_begin, d.mass_flux_x, d.mass_flux_y, d.mass_flux_zeta};
            float* flux_outputs[] = {
                scalar_flux[0].data(), scalar_flux[1].data(), scalar_flux[2].data()};
            const bool flux_write_only[] = {true, true, true};
            if (!dispatch_multi_field_kernel(
                    pipeline(KernelId::AdvectConservativeScalarFluxCartesianCgrid),
                    KernelId::AdvectConservativeScalarFluxCartesianCgrid,
                    &flux_pc,
                    flux_inputs,
                    4,
                    flux_outputs,
                    3,
                    d.nx,
                    d.ny,
                    d.nz,
                    total_points,
                    nullptr,
                    nullptr,
                    flux_write_only))
                return false;

            std::vector<float> limiter_factors(ncell, 1.0f);
            if (d.positive_definite)
            {
                MappedTerrainScalarLimiterPushConstants limiter_pc{};
                limiter_pc.nx = d.nx;
                limiter_pc.ny = d.ny;
                limiter_pc.nz                     = d.nz;
                limiter_pc.padding0               = d.preserve_positive_support ? 1 : 0;
                limiter_pc.inv_dx                 = 1.0f / d.dx;
                limiter_pc.inv_dy = 1.0f / d.dy;
                limiter_pc.inv_dzeta = 1.0f / d.dzeta;
                limiter_pc.dt = d.dt;
                const float* limiter_inputs[] = {d.scalar_begin, d.density_begin,
                    scalar_flux[0].data(), scalar_flux[1].data(), scalar_flux[2].data(), d.jacobian};
                float* limiter_outputs[] = {limiter_factors.data()};
                const bool limiter_write_only[] = {true};
                if (!dispatch_multi_field_kernel(pipeline(KernelId::MappedTerrainScalarLimiter),
                                                 KernelId::MappedTerrainScalarLimiter,
                                                 &limiter_pc,
                                                 limiter_inputs,
                                                 6,
                                                 limiter_outputs,
                                                 1,
                                                 d.nx,
                                                 d.ny,
                                                 d.nz,
                                                 total_points,
                                                 nullptr,
                                                 nullptr,
                                                 limiter_write_only))
                    return false;
                if (!std::all_of(limiter_factors.begin(), limiter_factors.end(),
                                 [](float value) { return std::isfinite(value) && value >= 0.0f && value <= 1.0f; }))
                    return false;
            }

            MappedTerrainScalarUpdatePushConstants update_pc{};
            update_pc.nx = d.nx;
            update_pc.ny = d.ny;
            update_pc.nz = d.nz;
            update_pc.positive_definite = d.positive_definite ? 1 : 0;
            update_pc.inv_dx = 1.0f / d.dx;
            update_pc.inv_dy = 1.0f / d.dy;
            update_pc.inv_dzeta = 1.0f / d.dzeta;
            update_pc.dt = d.dt;
            update_pc.density_floor = d.density_floor;
            std::vector<float> candidate(ncell);
            std::array<std::vector<float>, 5> applied;
            for (auto& field : applied)
                field.resize(ncell);
            const float* update_inputs[] = {d.scalar_begin, d.density_begin, d.density_end,
                scalar_flux[0].data(), scalar_flux[1].data(), scalar_flux[2].data(), d.jacobian,
                limiter_factors.data()};
            float* update_outputs[] = {candidate.data(), applied[0].data(), applied[1].data(),
                applied[2].data(), applied[3].data(), applied[4].data()};
            const bool update_write_only[] = {true, true, true, true, true, true};
            if (!dispatch_multi_field_kernel(pipeline(KernelId::MappedTerrainScalarUpdate),
                                             KernelId::MappedTerrainScalarUpdate,
                                             &update_pc,
                                             update_inputs,
                                             8,
                                             update_outputs,
                                             6,
                                             d.nx,
                                             d.ny,
                                             d.nz,
                                             total_points,
                                             nullptr,
                                             nullptr,
                                             update_write_only))
                return false;
            if (!std::all_of(candidate.begin(), candidate.end(),
                             [](float value) { return std::isfinite(value); }))
                return false;
            for (const auto& field : applied)
                if (!std::all_of(field.begin(), field.end(), [](float value) { return std::isfinite(value); }))
                    return false;
            if (d.applied_observer)
                d.applied_observer(d.applied_observer_context, candidate.data(), applied[0].data(),
                    applied[1].data(), applied[2].data(), applied[3].data(), applied[4].data());
            if (d.completed_observer)
                d.completed_observer(d.completed_observer_context,
                                     candidate.data(),
                                     scalar_flux[0].data(),
                                     scalar_flux[1].data(),
                                     scalar_flux[2].data(),
                                     limiter_factors.data());
            std::memcpy(d.scalar_end, candidate.data(), static_cast<std::size_t>(field_bytes));
            *d.report = {1u, d.positive_definite ? 1u : 0u, 1u, true};
            return true;
        }
        catch (const std::bad_alloc&)
        {
            return false;
        }
        catch (const std::length_error&)
        {
            return false;
        }
#else
        (void)d;
        return false;
#endif
    }

    bool supports_mapped_final_state() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return pipeline(KernelId::MappedTerrainEos).is_ready() &&
               pipeline(KernelId::MappedTerrainBoundary).is_ready();
#else
        return false;
#endif
    }

    bool supports_mapped_state_update() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return pipeline(KernelId::MappedTerrainStateUpdate).is_ready();
#else
        return false;
#endif
    }

    bool dispatch_mapped_state_update(const MappedTerrainStateUpdateDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        const bool pointers = d.u && d.v && d.w && d.rho && d.p && d.theta &&
            d.du_dt && d.dv_dt && d.dw_dt && d.drho_dt && d.dp_dt &&
            d.u_candidate && d.v_candidate && d.w_candidate && d.rho_candidate &&
            d.p_candidate && d.report;
        const auto mode = static_cast<std::uint32_t>(d.mode);
        if (!supports_mapped_state_update() || !pointers || d.nx < 3 || d.ny < 3 || d.nz < 2 ||
            mode > static_cast<std::uint32_t>(MappedTerrainStateUpdateMode::FastMomentum) ||
            !std::isfinite(d.dt) || d.dt <= 0.0f)
            return false;
        std::uint32_t total_points = 0;
        VkDeviceSize field_bytes = 0;
        if (!checked_field_dispatch(d.nx, d.ny, d.nz, total_points, field_bytes,
                                    "mapped terrain state update"))
            return false;
        const std::size_t ncell = total_points;
        const std::size_t bytes = static_cast<std::size_t>(field_bytes);
        if (!distinct_field_destinations(
                {d.u_candidate, d.v_candidate, d.w_candidate, d.rho_candidate,
                 d.p_candidate}, bytes))
            return false;
        for (std::size_t index = 0; index < ncell; ++index)
        {
            const float values[] = {d.u[index], d.v[index], d.w[index], d.rho[index], d.p[index],
                d.theta[index], d.du_dt[index], d.dv_dt[index], d.dw_dt[index],
                d.drho_dt[index], d.dp_dt[index]};
            if (!std::all_of(std::begin(values), std::end(values),
                             [](float value) { return std::isfinite(value); }) ||
                d.rho[index] <= 0.0f || d.p[index] <= 0.0f || d.theta[index] <= 0.0f)
                return false;
        }
        try
        {
            MappedTerrainStateUpdatePushConstants pc{};
            pc.nx = d.nx;
            pc.ny = d.ny;
            pc.nz = d.nz;
            pc.mode = mode;
            pc.dt = d.dt;
            std::array<std::vector<float>, 5> candidate;
            for (auto& field : candidate)
                field.resize(ncell);
            const float* inputs[] = {d.u, d.v, d.w, d.rho, d.p, d.theta,
                d.du_dt, d.dv_dt, d.dw_dt, d.drho_dt, d.dp_dt};
            float* outputs[] = {candidate[0].data(), candidate[1].data(), candidate[2].data(),
                candidate[3].data(), candidate[4].data()};
            const bool write_only[] = {true, true, true, true, true};
            if (!dispatch_multi_field_kernel(pipeline(KernelId::MappedTerrainStateUpdate),
                                             KernelId::MappedTerrainStateUpdate, &pc,
                                             inputs, 11, outputs, 5, d.nx, d.ny, d.nz,
                                             total_points, nullptr, nullptr, write_only))
                return false;
            for (std::size_t index = 0; index < ncell; ++index)
            {
                if (!std::isfinite(candidate[0][index]) || !std::isfinite(candidate[1][index]) ||
                    !std::isfinite(candidate[2][index]) || !std::isfinite(candidate[3][index]) ||
                    !std::isfinite(candidate[4][index]) || candidate[3][index] <= 0.0f ||
                    candidate[4][index] <= 0.0f)
                    return false;
            }
            if (d.completed_observer)
                d.completed_observer(d.completed_observer_context,
                    candidate[0].data(), candidate[1].data(), candidate[2].data(),
                    candidate[3].data(), candidate[4].data());
            float* destinations[] = {d.u_candidate, d.v_candidate, d.w_candidate,
                d.rho_candidate, d.p_candidate};
            for (std::size_t index = 0; index < candidate.size(); ++index)
                std::memcpy(destinations[index], candidate[index].data(), bytes);
            *d.report = {1u, d.mode, true};
            return true;
        }
        catch (const std::bad_alloc&)
        {
            return false;
        }
        catch (const std::length_error&)
        {
            return false;
        }
#else
        (void)d;
        return false;
#endif
    }

    bool supports_mapped_flux_accumulation() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return pipeline(KernelId::MappedTerrainFluxAccumulation).is_ready();
#else
        return false;
#endif
    }

    bool dispatch_mapped_flux_accumulation(
        const MappedTerrainFluxAccumulationDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        const bool pointers = d.accumulated_x && d.accumulated_y && d.accumulated_zeta &&
            d.step_x && d.step_y && d.step_zeta && d.accumulated_x_candidate &&
            d.accumulated_y_candidate && d.accumulated_zeta_candidate && d.report;
        if (!supports_mapped_flux_accumulation() || !pointers || d.nx < 3 || d.ny < 3 || d.nz < 2 ||
            !std::isfinite(d.time_weight) || d.time_weight <= 0.0f || d.time_weight > 1.0f)
            return false;
        std::uint32_t total_points = 0;
        VkDeviceSize field_bytes = 0;
        if (!checked_field_dispatch(d.nx, d.ny, d.nz, total_points, field_bytes,
                                    "mapped terrain flux accumulation"))
            return false;
        const std::size_t ncell = total_points;
        const std::size_t bytes = static_cast<std::size_t>(field_bytes);
        if (!distinct_field_destinations(
                {d.accumulated_x_candidate, d.accumulated_y_candidate,
                 d.accumulated_zeta_candidate}, bytes))
            return false;
        for (std::size_t index = 0; index < ncell; ++index)
        {
            const float values[] = {d.accumulated_x[index], d.accumulated_y[index],
                d.accumulated_zeta[index], d.step_x[index], d.step_y[index], d.step_zeta[index]};
            if (!std::all_of(std::begin(values), std::end(values),
                             [](float value) { return std::isfinite(value); }))
                return false;
        }
        try
        {
            MappedTerrainFluxAccumulationPushConstants pc{};
            pc.nx = d.nx;
            pc.ny = d.ny;
            pc.nz = d.nz;
            pc.time_weight = d.time_weight;
            std::array<std::vector<float>, 3> candidate;
            for (auto& field : candidate)
                field.resize(ncell);
            const float* inputs[] = {d.accumulated_x, d.accumulated_y, d.accumulated_zeta,
                d.step_x, d.step_y, d.step_zeta};
            float* outputs[] = {candidate[0].data(), candidate[1].data(), candidate[2].data()};
            const bool write_only[] = {true, true, true};
            if (!dispatch_multi_field_kernel(pipeline(KernelId::MappedTerrainFluxAccumulation),
                                             KernelId::MappedTerrainFluxAccumulation, &pc,
                                             inputs, 6, outputs, 3, d.nx, d.ny, d.nz,
                                             total_points, nullptr, nullptr, write_only))
                return false;
            for (const auto& field : candidate)
                if (!std::all_of(field.begin(), field.end(),
                                 [](float value) { return std::isfinite(value); }))
                    return false;
            if (d.completed_observer)
                d.completed_observer(d.completed_observer_context, candidate[0].data(),
                                     candidate[1].data(), candidate[2].data());
            float* destinations[] = {d.accumulated_x_candidate, d.accumulated_y_candidate,
                d.accumulated_zeta_candidate};
            for (std::size_t index = 0; index < candidate.size(); ++index)
                std::memcpy(destinations[index], candidate[index].data(), bytes);
            *d.report = {1u, true};
            return true;
        }
        catch (const std::bad_alloc&)
        {
            return false;
        }
        catch (const std::length_error&)
        {
            return false;
        }
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_mapped_final_state(const MappedTerrainFinalStateDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        const bool pointers = d.u && d.v && d.w && d.rho && d.p && d.theta &&
            d.u_final && d.v_final && d.w_final && d.rho_final && d.p_final && d.report;
        if (!supports_mapped_final_state() || !nimbus::compute::mapped_boundary::valid_options(d) || !pointers || d.nx < 3 || d.ny < 3 || d.nz < 2)
            return false;
        std::uint32_t total_points = 0;
        VkDeviceSize field_bytes = 0;
        if (!checked_field_dispatch(d.nx, d.ny, d.nz, total_points, field_bytes,
                                    "mapped terrain final state"))
            return false;
        const std::size_t ncell = total_points;
        const std::size_t bytes = static_cast<std::size_t>(field_bytes);
        if (!distinct_field_destinations(
                {d.u_final, d.v_final, d.w_final, d.rho_final, d.p_final}, bytes))
            return false;
        for (std::size_t index = 0; index < ncell; ++index)
        {
            const float values[] = {
                d.u[index], d.v[index], d.w[index], d.rho[index], d.p[index], d.theta[index]};
            if (!std::all_of(std::begin(values), std::end(values),
                             [](float value) { return std::isfinite(value); }) ||
                d.rho[index] <= 0.0f || d.p[index] <= 0.0f || d.theta[index] <= 0.0f)
                return false;
        }

        try
        {
            std::vector<float> eos_pressure(d.p, d.p + ncell);
            if (d.apply_eos)
            {
                MappedTerrainEosPushConstants eos_pc{};
                eos_pc.nx = d.nx;
                eos_pc.ny = d.ny;
                eos_pc.nz = d.nz;
                const float* eos_inputs[] = {d.rho, d.theta};
                float* eos_outputs[] = {eos_pressure.data()};
                const bool eos_write_only[] = {true};
                if (!dispatch_multi_field_kernel(pipeline(KernelId::MappedTerrainEos),
                                                 KernelId::MappedTerrainEos,
                                                 &eos_pc,
                                                 eos_inputs,
                                                 2,
                                                 eos_outputs,
                                                 1,
                                                 d.nx,
                                                 d.ny,
                                                 d.nz,
                                                 total_points,
                                                 nullptr,
                                                 nullptr,
                                                 eos_write_only))
                    return false;
            }
            if (!std::all_of(eos_pressure.begin(), eos_pressure.end(),
                             [](float value) { return std::isfinite(value) && value > 0.0f; }))
                return false;

            const auto boundary_pc = nimbus::compute::mapped_boundary::constants(d);
            std::array<std::vector<float>, 5> candidate;
            for (auto& field : candidate)
                field.resize(ncell);
            const float* boundary_inputs[] = {d.u, d.v, d.w, d.rho, eos_pressure.data()};
            float* boundary_outputs[] = {candidate[0].data(), candidate[1].data(), candidate[2].data(),
                                         candidate[3].data(), candidate[4].data()};
            const bool boundary_write_only[] = {true, true, true, true, true};
            if (!dispatch_multi_field_kernel(pipeline(KernelId::MappedTerrainBoundary),
                                             KernelId::MappedTerrainBoundary,
                                             &boundary_pc,
                                             boundary_inputs,
                                             5,
                                             boundary_outputs,
                                             5,
                                             d.nx,
                                             d.ny,
                                             d.nz,
                                             total_points,
                                             nullptr,
                                             nullptr,
                                             boundary_write_only))
                return false;
            for (const auto& field : candidate)
                if (!std::all_of(field.begin(), field.end(),
                                 [](float value) { return std::isfinite(value); }))
                    return false;
            if (d.completed_observer)
                d.completed_observer(d.completed_observer_context,
                                     candidate[0].data(), candidate[1].data(), candidate[2].data(),
                                     candidate[3].data(), candidate[4].data());
            float* destinations[] = {d.u_final, d.v_final, d.w_final, d.rho_final, d.p_final};
            for (std::size_t index = 0; index < candidate.size(); ++index)
                std::memcpy(destinations[index], candidate[index].data(), bytes);
            *d.report = {d.apply_eos ? 1u : 0u, 1u, true};
            return true;
        }
        catch (const std::bad_alloc&)
        {
            return false;
        }
        catch (const std::length_error&)
        {
            return false;
        }
#else
        (void)d;
        return false;
#endif
    }

    bool initialize(std::string& error) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        shutdown();
        if (requested_config_.startup_plan != nullptr &&
            !requested_config_.startup_plan->validate(nimbus::compute::kernel_manifest(), error))
        {
            return false;
        }
        startup_memory_enabled_ = process_memory_trace_enabled(std::getenv("NIMBUS_VULKAN_STARTUP_MEMORY"));
        configure_memory_window();
        log_startup_memory("init_begin");

        if (!open_loader(error))
        {
            shutdown();
            return false;
        }
        log_startup_memory("loader_ready");
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
        log_startup_memory("instance_ready");
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
        log_startup_memory("device_selected");
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
        log_startup_memory("device_ready");

        if (log_normal_enabled())
        {
            std::cout << "[COMPUTE][VULKAN] loader_api=" << version_to_string(device_context_.loader_api_version)
                      << ", device_api=" << version_to_string(device_context_.selected_device_properties.apiVersion)
                      << ", device_index=" << device_context_.selected_device_index << ", device=\""
                      << device_context_.selected_device_properties.deviceName << "\""
                      << ", type=" << device_type_name(device_context_.selected_device_properties.deviceType)
                      << ", compute_queue_family=" << device_context_.compute_queue_family_index
                      << ", queue_count=" << device_context_.selected_compute_queue_family.queueCount
                      << ", max_workgroup_invocations="
                      << device_context_.selected_device_properties.limits.maxComputeWorkGroupInvocations
                      << ", shared_mem_bytes="
                      << device_context_.selected_device_properties.limits.maxComputeSharedMemorySize << std::endl;
        }

        // A selected owner plan requires every pipeline. Unspecified callers
        // retain the manifest's required/optional compatibility policy.
        std::string pipeline_error;
        if (!setup_compute_pipelines(pipeline_error))
        {
            error = "Vulkan compute setup failed: " + pipeline_error;
            log_vulkan_warning(error);
            shutdown();
            return false;
        }
#ifdef VK_NV_cuda_kernel_launch
        if (cuda_assist_enabled_ && !cuda_flux_.initialize(device_context_.device, vkGetDeviceProcAddr_, pipeline_error))
        {
            error = "CUDA assist setup failed: " + pipeline_error;
            shutdown();
            return false;
        }
        if (cuda_assist_enabled_)
            log_vulkan_info("CUDA assist active: retained-flux PTX sha256=" + std::string(tmv_vulkan::cuda_flux_sha256));
#endif

        log_startup_memory("init_end");
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

    bool supports(AdvectionKernelOp op) const override
    {
        switch (op)
        {
        case AdvectionKernelOp::VerticalFlux:
            return supports_vertical_flux_dispatch();
        case AdvectionKernelOp::ConservativeCartesianCgridTransport:
            return supports_advect_scalars_conservative_cartesian_cgrid_dispatch();
        case AdvectionKernelOp::AdvectionX:
            return supports_advection_x_dispatch();
        case AdvectionKernelOp::AdvectionY:
            return supports_advection_y_dispatch();
        case AdvectionKernelOp::BatchedAdvection:
            return false;
        }
        return false;
    }

    bool supports(DynamicsKernelOp op) const override
    {
        switch (op)
        {
        case DynamicsKernelOp::AcousticContinuityMassFluxCartesianAgrid:
            return supports_acoustic_continuity_mass_flux_cartesian_agrid_dispatch();
        case DynamicsKernelOp::AcousticPressureEosCartesianAgrid:
            return supports_acoustic_pressure_eos_cartesian_agrid_dispatch();
        case DynamicsKernelOp::AcousticMomentumCartesianAgrid:
            return supports_acoustic_momentum_cartesian_agrid_dispatch();
        case DynamicsKernelOp::AcousticNullModeConstraintCartesianAgrid:
            return supports_acoustic_null_mode_constraint_cartesian_agrid_dispatch();
        case DynamicsKernelOp::AcousticBoundaryCartesianCgrid:
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
            return pipeline(KernelId::AcousticBcFaceCartesianCgrid).is_ready();
#else
            return false;
#endif
        case DynamicsKernelOp::AcousticBoundaryCartesianAgrid:
            return supports_acoustic_boundary_cartesian_agrid_dispatch();
        case DynamicsKernelOp::AcousticPressureCartesianCgrid:
            return supports_acoustic_pressure_cartesian_cgrid_dispatch();
        case DynamicsKernelOp::AcousticMomentumCartesianCgrid:
            return supports_acoustic_momentum_cartesian_cgrid_dispatch();
        case DynamicsKernelOp::AcousticSubstepFusedCartesianCgrid:
            return supports_acoustic_substep_fused_cartesian_cgrid_dispatch();
        case DynamicsKernelOp::AcousticSubstepsBatchedCartesianCgrid:
            return supports_acoustic_substeps_batched_cartesian_cgrid_dispatch();
        case DynamicsKernelOp::StateRelaxationCartesian:
            return supports_state_relaxation_cartesian_dispatch();
        case DynamicsKernelOp::SlowApply:
            return supports_slow_apply_dispatch();
        case DynamicsKernelOp::ConservativeReduction:
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
            return pipeline(KernelId::ConservativeReduction).is_ready();
#else
            return false;
#endif
        case DynamicsKernelOp::ConservativeMomentum:
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
            return pipeline(KernelId::ConservativeMomentum).is_ready();
#else
            return false;
#endif
        case DynamicsKernelOp::CartesianCgridSlowTendencies:
            return supports_cartesian_cgrid_slow_tendencies_dispatch();
        case DynamicsKernelOp::CartesianTendencies:
            return supports_cartesian_tendencies_dispatch();
        case DynamicsKernelOp::AcousticPressure:
            return supports_acoustic_pressure_dispatch();
        case DynamicsKernelOp::AcousticMomentum:
            return supports_acoustic_momentum_dispatch();
        case DynamicsKernelOp::AcousticSubstepFused:
            return supports_acoustic_substep_fused_dispatch();
        case DynamicsKernelOp::AcousticSubstepsBatched:
            return supports_acoustic_substeps_batched_dispatch();
        }
        return false;
    }

    bool supports(MicrophysicsKernelOp op) const override
    {
        switch (op)
        {
        case MicrophysicsKernelOp::ThompsonCompletion:
            return pipeline(KernelId::ThompsonCompletion).is_ready();
        case MicrophysicsKernelOp::ThompsonDeposition:
            return pipeline(KernelId::ThompsonDeposition).is_ready();
        case MicrophysicsKernelOp::ThompsonFreezing:
            return pipeline(KernelId::ThompsonFreezing).is_ready();
        case MicrophysicsKernelOp::ThompsonCloudPhase:
            return pipeline(KernelId::ThompsonCloudPhase).is_ready();
        case MicrophysicsKernelOp::MilbrandtMelting:
            return pipeline(KernelId::MilbrandtMelting).is_ready();
        case MicrophysicsKernelOp::MilbrandtSedimentation:
            return pipeline(KernelId::MilbrandtSedimentation).is_ready();
        case MicrophysicsKernelOp::MilbrandtCold:
            return pipeline(KernelId::MilbrandtCold).is_ready();
        case MicrophysicsKernelOp::MilbrandtCloudAdjustment:
            return pipeline(KernelId::MilbrandtCloudAdjustment).is_ready();
        case MicrophysicsKernelOp::MilbrandtWarmRain:
            return pipeline(KernelId::MilbrandtWarmRain).is_ready();
        case MicrophysicsKernelOp::ThompsonWarmRain:
            return pipeline(KernelId::ThompsonWarmRain).is_ready();
        case MicrophysicsKernelOp::ThompsonCloudAdjustment:
            return pipeline(KernelId::ThompsonCloudAdjustment).is_ready();
        case MicrophysicsKernelOp::ThompsonMomentSedimentation:
            return pipeline(KernelId::ThompsonMomentSedimentation).is_ready();
        case MicrophysicsKernelOp::LinFused:
            return dependency_batching_enabled() && pipeline(KernelId::LinPointwise).is_ready() &&
                   pipeline(KernelId::LinSedimentation).is_ready();
        case MicrophysicsKernelOp::LinPointwise:
            return supports_lin_pointwise_dispatch();
        case MicrophysicsKernelOp::LinSedimentation:
            return supports_lin_sedimentation_dispatch();
        case MicrophysicsKernelOp::MicrophysicsApply:
            return supports_microphysics_apply_dispatch();
        case MicrophysicsKernelOp::KesslerPointwise:
            return supports_kessler_pointwise_dispatch();
        case MicrophysicsKernelOp::KesslerSedimentation:
            return supports_kessler_sedimentation_dispatch();
        case MicrophysicsKernelOp::ArchivedLegacyThompsonLinPointwise:
        case MicrophysicsKernelOp::ArchivedLegacyThompsonLinSedimentation:
            return false;
        }
        return false;
    }

    bool supports(TurbulenceKernelOp op) const override
    {
        switch (op)
        {
        case TurbulenceKernelOp::Smagorinsky:
            return supports_smagorinsky_dispatch();
        case TurbulenceKernelOp::CollocatedSgs:
            return pipeline(KernelId::CollocatedSgs).is_ready();
        case TurbulenceKernelOp::Tke:
            return supports_tke_dispatch();
        case TurbulenceKernelOp::SgsApply:
            return supports_sgs_apply_dispatch();
        }
        return false;
    }

    bool supports(DiffusionKernelOp op) const override
    {
        switch (op)
        {
        case DiffusionKernelOp::Diffusion:
            return supports_diffusion_dispatch();
        case DiffusionKernelOp::Implicit:
            return supports_implicit_diffusion_dispatch();
        }
        return false;
    }

    bool supports(BoundaryLayerKernelOp op) const override
    {
        switch (op)
        {
        case BoundaryLayerKernelOp::SlabColumn:
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
            return pipeline(KernelId::SlabColumn).is_ready();
#else
            return false;
#endif
        case BoundaryLayerKernelOp::MynnLevel25Stability:
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
            return pipeline(KernelId::MynnLevel25Stability).is_ready();
#else
            return false;
#endif
        }
        return false;
    }

    bool dispatch_mynn_level25_stability(const MynnLevel25StabilityDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!supports(BoundaryLayerKernelOp::MynnLevel25Stability) || dispatch.count <= 0 ||
            dispatch.gm_s2 == nullptr || dispatch.gh_s2 == nullptr || dispatch.mixing_length_m == nullptr ||
            dispatch.q_squared_m2s2 == nullptr || dispatch.sm == nullptr || dispatch.sh == nullptr ||
            dispatch.gradient_richardson == nullptr || dispatch.flux_richardson == nullptr ||
            dispatch.equilibrium_limiter == nullptr)
        {
            return false;
        }
        const void* inputs_untyped[4] = {
            dispatch.gm_s2, dispatch.gh_s2, dispatch.mixing_length_m, dispatch.q_squared_m2s2};
        void* outputs_untyped[5] = {dispatch.sm,
                                    dispatch.sh,
                                    dispatch.gradient_richardson,
                                    dispatch.flux_richardson,
                                    dispatch.equilibrium_limiter};
        for (int output = 0; output < 5; ++output)
        {
            for (int prior = 0; prior < output; ++prior)
                if (outputs_untyped[output] == outputs_untyped[prior])
                    return false;
            for (const void* input : inputs_untyped)
                if (outputs_untyped[output] == input)
                    return false;
        }
        for (int index = 0; index < dispatch.count; ++index)
        {
            if (!std::isfinite(dispatch.gm_s2[index]) || dispatch.gm_s2[index] < 0.0f ||
                !std::isfinite(dispatch.gh_s2[index]) || !std::isfinite(dispatch.mixing_length_m[index]) ||
                dispatch.mixing_length_m[index] <= 0.0f || !std::isfinite(dispatch.q_squared_m2s2[index]) ||
                dispatch.q_squared_m2s2[index] <= 0.0f)
            {
                return false;
            }
        }
        try
        {
            std::array<std::vector<float>, 5> candidates;
            for (auto& candidate : candidates)
                candidate.resize(static_cast<std::size_t>(dispatch.count));
            MynnLevel25StabilityPushConstants constants{};
            constants.count        = dispatch.count;
            const float* inputs[4] = {
                dispatch.gm_s2, dispatch.gh_s2, dispatch.mixing_length_m, dispatch.q_squared_m2s2};
            float*     outputs[5]    = {candidates[0].data(),
                                        candidates[1].data(),
                                        candidates[2].data(),
                                        candidates[3].data(),
                                        candidates[4].data()};
            const bool write_only[5] = {true, true, true, true, true};
            if (!dispatch_multi_field_kernel(pipeline(KernelId::MynnLevel25Stability),
                                             KernelId::MynnLevel25Stability,
                                             &constants,
                                             inputs,
                                             4,
                                             outputs,
                                             5,
                                             dispatch.count,
                                             1,
                                             1,
                                             static_cast<std::uint32_t>(dispatch.count),
                                             nullptr,
                                             nullptr,
                                             write_only))
            {
                return false;
            }
            for (int index = 0; index < dispatch.count; ++index)
            {
                if (!std::isfinite(candidates[0][index]) || candidates[0][index] < 0.0f ||
                    !std::isfinite(candidates[1][index]) || candidates[1][index] < 0.0f ||
                    !std::isfinite(candidates[2][index]) || !std::isfinite(candidates[3][index]) ||
                    !std::isfinite(candidates[4][index]) || candidates[4][index] <= 0.0f || candidates[4][index] > 1.0f)
                {
                    return false;
                }
            }
            float*            destinations[5] = {dispatch.sm,
                                                 dispatch.sh,
                                                 dispatch.gradient_richardson,
                                                 dispatch.flux_richardson,
                                                 dispatch.equilibrium_limiter};
            const std::size_t bytes           = static_cast<std::size_t>(dispatch.count) * sizeof(float);
            for (int output = 0; output < 5; ++output)
                std::memcpy(destinations[output], candidates[output].data(), bytes);
            return true;
        }
        catch (const std::bad_alloc&)
        {
            return false;
        }
        catch (const std::length_error&)
        {
            return false;
        }
#else
        (void)dispatch;
        return false;
#endif
    }

    bool supports(RadiationKernelOp op) const override
    {
        switch (op)
        {
        case RadiationKernelOp::SimpleGrey:
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
            return pipeline(KernelId::SimpleGrey).is_ready();
#else
            return false;
#endif
        case RadiationKernelOp::RrtmgLongwaveTransfer:
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
            return pipeline(KernelId::RrtmgLongwaveTransfer).is_ready();
#else
            return false;
#endif
        case RadiationKernelOp::RrtmgShortwaveTransfer:
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
            return pipeline(KernelId::RrtmgShortwaveTransfer).is_ready();
#else
            return false;
#endif
        case RadiationKernelOp::RrtmgLongwaveTransferBatch:
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
            return pipeline(KernelId::RrtmgLongwaveTransferBatch).is_ready();
#else
            return false;
#endif
        case RadiationKernelOp::RrtmgShortwaveTransferBatch:
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
            return pipeline(KernelId::RrtmgShortwaveTransferBatch).is_ready();
#else
            return false;
#endif
        }
        return false;
    }

    bool dispatch_rrtmg_longwave_transfer(const RrtmgLongwaveTransferDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        constexpr std::size_t g_points = 140;
        constexpr std::size_t bands    = 16;
        if (!supports(RadiationKernelOp::RrtmgLongwaveTransfer) || dispatch.layers <= 0 ||
            dispatch.absorption_optical_depth == nullptr || dispatch.planck_fraction == nullptr ||
            dispatch.layer_planck == nullptr || dispatch.interface_planck == nullptr ||
            dispatch.surface_planck == nullptr || dispatch.band_width_cm == nullptr || dispatch.upward_wm2 == nullptr ||
            dispatch.downward_wm2 == nullptr || dispatch.upward_wm2 == dispatch.downward_wm2 ||
            !std::isfinite(dispatch.surface_emissivity) || dispatch.surface_emissivity < 0.0f ||
            dispatch.surface_emissivity > 1.0f || !std::isfinite(dispatch.precipitable_water_cm) ||
            dispatch.precipitable_water_cm < 0.0f)
        {
            return false;
        }
        const std::size_t layers = static_cast<std::size_t>(dispatch.layers);
        if (layers > (std::numeric_limits<std::size_t>::max() / g_points) - 1 ||
            layers > (std::numeric_limits<std::size_t>::max() / bands) - 1)
            return false;
        const std::size_t spectral_count       = layers * g_points;
        const std::size_t layer_band_count     = layers * bands;
        const std::size_t interface_band_count = (layers + 1) * bands;
        const std::size_t partial_count        = (layers + 1) * g_points;
        if (partial_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
            interface_band_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
            dispatch.layers == std::numeric_limits<int>::max())
            return false;
        const void* inputs_untyped[7] = {dispatch.absorption_optical_depth,
                                         dispatch.planck_fraction,
                                         dispatch.layer_planck,
                                         dispatch.interface_planck,
                                         dispatch.surface_planck,
                                         dispatch.band_width_cm,
                                         dispatch.cloud_absorption_optical_depth};
        for (const void* input : inputs_untyped)
        {
            if (input != nullptr && (input == dispatch.upward_wm2 || input == dispatch.downward_wm2))
                return false;
        }
        const auto valid_nonnegative = [](const float* values, std::size_t count)
        {
            for (std::size_t index = 0; index < count; ++index)
                if (!std::isfinite(values[index]) || values[index] < 0.0f)
                    return false;
            return true;
        };
        if (!valid_nonnegative(dispatch.absorption_optical_depth, spectral_count) ||
            !valid_nonnegative(dispatch.planck_fraction, spectral_count) ||
            !valid_nonnegative(dispatch.layer_planck, layer_band_count) ||
            !valid_nonnegative(dispatch.interface_planck, interface_band_count) ||
            !valid_nonnegative(dispatch.surface_planck, bands) || !valid_nonnegative(dispatch.band_width_cm, bands) ||
            (dispatch.cloud_absorption_optical_depth != nullptr &&
             !valid_nonnegative(dispatch.cloud_absorption_optical_depth, layer_band_count)))
        {
            return false;
        }
        for (std::size_t index = 0; index < bands; ++index)
            if (dispatch.band_width_cm[index] <= 0.0f)
                return false;
        try
        {
            std::vector<float> zero_clouds;
            const float*       cloud = dispatch.cloud_absorption_optical_depth;
            if (cloud == nullptr)
            {
                zero_clouds.assign(layer_band_count, 0.0f);
                cloud = zero_clouds.data();
            }
            std::vector<float>                 upward_partial(partial_count, 0.0f);
            std::vector<float>                 downward_partial(partial_count, 0.0f);
            RrtmgLongwaveTransferPushConstants constants{};
            constants.layers                  = dispatch.layers;
            constants.has_clouds              = dispatch.cloud_absorption_optical_depth != nullptr ? 1 : 0;
            constants.surface_emissivity      = dispatch.surface_emissivity;
            constants.precipitable_water_cm   = dispatch.precipitable_water_cm;
            const float*       inputs[7]      = {dispatch.absorption_optical_depth,
                                                 dispatch.planck_fraction,
                                                 dispatch.layer_planck,
                                                 dispatch.interface_planck,
                                                 dispatch.surface_planck,
                                                 dispatch.band_width_cm,
                                                 cloud};
            const VkDeviceSize input_bytes[7] = {static_cast<VkDeviceSize>(spectral_count * sizeof(float)),
                                                 static_cast<VkDeviceSize>(spectral_count * sizeof(float)),
                                                 static_cast<VkDeviceSize>(layer_band_count * sizeof(float)),
                                                 static_cast<VkDeviceSize>(interface_band_count * sizeof(float)),
                                                 static_cast<VkDeviceSize>(bands * sizeof(float)),
                                                 static_cast<VkDeviceSize>(bands * sizeof(float)),
                                                 static_cast<VkDeviceSize>(layer_band_count * sizeof(float))};
            float*             outputs[2]     = {upward_partial.data(), downward_partial.data()};
            if (!dispatch_multi_field_kernel(pipeline(KernelId::RrtmgLongwaveTransfer),
                                             KernelId::RrtmgLongwaveTransfer,
                                             &constants,
                                             inputs,
                                             7,
                                             outputs,
                                             2,
                                             dispatch.layers + 1,
                                             static_cast<int>(g_points),
                                             1,
                                             static_cast<std::uint32_t>(g_points),
                                             input_bytes))
            {
                return false;
            }
            std::vector<float> upward(layers + 1, 0.0f);
            std::vector<float> downward(layers + 1, 0.0f);
            for (std::size_t g = 0; g < g_points; ++g)
                for (std::size_t interface_index = 0; interface_index <= layers; ++interface_index)
                {
                    const std::size_t index = g * (layers + 1) + interface_index;
                    upward[interface_index] += upward_partial[index];
                    downward[interface_index] += downward_partial[index];
                }
            for (std::size_t interface_index = 0; interface_index <= layers; ++interface_index)
                if (!std::isfinite(upward[interface_index]) || upward[interface_index] < 0.0f ||
                    !std::isfinite(downward[interface_index]) || downward[interface_index] < 0.0f)
                    return false;
            const std::size_t bytes = (layers + 1) * sizeof(float);
            std::memcpy(dispatch.upward_wm2, upward.data(), bytes);
            std::memcpy(dispatch.downward_wm2, downward.data(), bytes);
            return true;
        }
        catch (const std::bad_alloc&)
        {
            return false;
        }
        catch (const std::length_error&)
        {
            return false;
        }
#else
        (void)dispatch;
        return false;
#endif
    }

    bool dispatch_rrtmg_shortwave_transfer(const RrtmgShortwaveTransferDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        constexpr std::size_t g_points  = 112;
        constexpr std::size_t bands     = 14;
        const bool            any_cloud = dispatch.cloud_optical_depth != nullptr ||
                                          dispatch.cloud_single_scattering_albedo != nullptr ||
                                          dispatch.cloud_asymmetry != nullptr;
        const bool            all_cloud = dispatch.cloud_optical_depth != nullptr &&
                                          dispatch.cloud_single_scattering_albedo != nullptr &&
                                          dispatch.cloud_asymmetry != nullptr;
        if (!supports(RadiationKernelOp::RrtmgShortwaveTransfer) || dispatch.layers <= 0 ||
            dispatch.layers == std::numeric_limits<int>::max() || dispatch.absorption_optical_depth == nullptr ||
            dispatch.rayleigh_optical_depth == nullptr || dispatch.solar_source_fraction == nullptr ||
            dispatch.upward_wm2 == nullptr || dispatch.downward_wm2 == nullptr ||
            dispatch.upward_wm2 == dispatch.downward_wm2 || (any_cloud && !all_cloud) ||
            !std::isfinite(dispatch.cosine_zenith) || dispatch.cosine_zenith < 0.0f || dispatch.cosine_zenith > 1.0f ||
            !std::isfinite(dispatch.solar_irradiance_wm2) || dispatch.solar_irradiance_wm2 < 0.0f ||
            !std::isfinite(dispatch.surface_albedo) || dispatch.surface_albedo < 0.0f || dispatch.surface_albedo > 1.0f)
            return false;
        const std::size_t layers = static_cast<std::size_t>(dispatch.layers);
        if (layers > (std::numeric_limits<std::size_t>::max() / g_points) - 1 ||
            layers > (std::numeric_limits<std::size_t>::max() / bands) - 1)
            return false;
        const std::size_t spectral_count = layers * g_points;
        const std::size_t cloud_count    = layers * bands;
        const std::size_t partial_count  = (layers + 1) * g_points;
        if (partial_count > std::numeric_limits<std::size_t>::max() / sizeof(float))
            return false;
        const auto valid_range = [](const float* values, std::size_t count, float maximum)
        {
            for (std::size_t index = 0; index < count; ++index)
                if (!std::isfinite(values[index]) || values[index] < 0.0f || values[index] > maximum)
                    return false;
            return true;
        };
        if (!valid_range(dispatch.absorption_optical_depth, spectral_count, std::numeric_limits<float>::max()) ||
            !valid_range(dispatch.rayleigh_optical_depth, spectral_count, std::numeric_limits<float>::max()) ||
            !valid_range(dispatch.solar_source_fraction, g_points, std::numeric_limits<float>::max()) ||
            (all_cloud && (!valid_range(dispatch.cloud_optical_depth, cloud_count, std::numeric_limits<float>::max()) ||
                           !valid_range(dispatch.cloud_single_scattering_albedo, cloud_count, 1.0f) ||
                           !valid_range(dispatch.cloud_asymmetry, cloud_count, 1.0f))))
            return false;
        float source_sum = 0.0f;
        for (std::size_t g = 0; g < g_points; ++g)
            source_sum += dispatch.solar_source_fraction[g];
        if (!std::isfinite(source_sum) || source_sum <= 0.0f)
            return false;
        if (dispatch.cosine_zenith == 0.0f || dispatch.solar_irradiance_wm2 == 0.0f)
        {
            std::fill_n(dispatch.upward_wm2, layers + 1, 0.0f);
            std::fill_n(dispatch.downward_wm2, layers + 1, 0.0f);
            return true;
        }
        try
        {
            std::array<std::vector<float>, 3> zero_clouds;
            const float*                      cloud_inputs[3] = {
                dispatch.cloud_optical_depth, dispatch.cloud_single_scattering_albedo, dispatch.cloud_asymmetry};
            if (!all_cloud)
                for (int field = 0; field < 3; ++field)
                {
                    zero_clouds[field].assign(cloud_count, 0.0f);
                    cloud_inputs[field] = zero_clouds[field].data();
                }
            std::array<std::vector<float>, 12> candidates;
            for (auto& candidate : candidates)
                candidate.assign(partial_count, 0.0f);
            RrtmgShortwaveTransferPushConstants constants{};
            constants.layers                  = dispatch.layers;
            constants.has_clouds              = all_cloud ? 1 : 0;
            constants.cosine_zenith           = dispatch.cosine_zenith;
            constants.solar_irradiance_wm2    = dispatch.solar_irradiance_wm2;
            constants.surface_albedo          = dispatch.surface_albedo;
            constants.source_sum              = source_sum;
            const float*       inputs[6]      = {dispatch.absorption_optical_depth,
                                                 dispatch.rayleigh_optical_depth,
                                                 dispatch.solar_source_fraction,
                                                 cloud_inputs[0],
                                                 cloud_inputs[1],
                                                 cloud_inputs[2]};
            const VkDeviceSize input_bytes[6] = {static_cast<VkDeviceSize>(spectral_count * sizeof(float)),
                                                 static_cast<VkDeviceSize>(spectral_count * sizeof(float)),
                                                 static_cast<VkDeviceSize>(g_points * sizeof(float)),
                                                 static_cast<VkDeviceSize>(cloud_count * sizeof(float)),
                                                 static_cast<VkDeviceSize>(cloud_count * sizeof(float)),
                                                 static_cast<VkDeviceSize>(cloud_count * sizeof(float))};
            float*             outputs[12];
            for (int field = 0; field < 12; ++field)
                outputs[field] = candidates[field].data();
            if (!dispatch_multi_field_kernel(pipeline(KernelId::RrtmgShortwaveTransfer),
                                             KernelId::RrtmgShortwaveTransfer,
                                             &constants,
                                             inputs,
                                             6,
                                             outputs,
                                             12,
                                             dispatch.layers + 1,
                                             static_cast<int>(g_points),
                                             1,
                                             static_cast<std::uint32_t>(g_points),
                                             input_bytes))
                return false;
            std::vector<float> upward(layers + 1, 0.0f), downward(layers + 1, 0.0f);
            for (std::size_t g = 0; g < g_points; ++g)
                for (std::size_t interface_index = 0; interface_index <= layers; ++interface_index)
                {
                    const std::size_t index = g * (layers + 1) + interface_index;
                    upward[interface_index] += candidates[0][index];
                    downward[interface_index] += candidates[1][index];
                }
            for (std::size_t interface_index = 0; interface_index <= layers; ++interface_index)
                if (!std::isfinite(upward[interface_index]) || upward[interface_index] < 0.0f ||
                    !std::isfinite(downward[interface_index]) || downward[interface_index] < 0.0f)
                    return false;
            const std::size_t bytes = (layers + 1) * sizeof(float);
            std::memcpy(dispatch.upward_wm2, upward.data(), bytes);
            std::memcpy(dispatch.downward_wm2, downward.data(), bytes);
            return true;
        }
        catch (const std::bad_alloc&)
        {
            return false;
        }
        catch (const std::length_error&)
        {
            return false;
        }
#else
        (void)dispatch;
        return false;
#endif
    }

    bool dispatch_rrtmg_longwave_transfer_batch(const RrtmgLongwaveTransferBatchDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        constexpr std::size_t g_points = 140, bands = 16;
        if (!supports(RadiationKernelOp::RrtmgLongwaveTransferBatch) || dispatch.columns <= 0 || dispatch.layers <= 0 ||
            dispatch.layers == std::numeric_limits<int>::max() || dispatch.absorption_optical_depth == nullptr ||
            dispatch.planck_fraction == nullptr || dispatch.layer_planck == nullptr ||
            dispatch.interface_planck == nullptr || dispatch.surface_planck == nullptr ||
            dispatch.band_width_cm == nullptr || dispatch.surface_emissivity == nullptr ||
            dispatch.precipitable_water_cm == nullptr || dispatch.upward_wm2 == nullptr ||
            dispatch.downward_wm2 == nullptr || dispatch.upward_wm2 == dispatch.downward_wm2)
            return false;
        const std::size_t columns         = static_cast<std::size_t>(dispatch.columns);
        const std::size_t layers          = static_cast<std::size_t>(dispatch.layers);
        const auto        checked_product = [](std::size_t a, std::size_t b, std::size_t c, std::size_t& result)
        {
            if (a == 0 || b == 0 || c == 0 || a > std::numeric_limits<std::size_t>::max() / b)
                return false;
            result = a * b;
            if (result > std::numeric_limits<std::size_t>::max() / c)
                return false;
            result *= c;
            return result <= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) &&
                   result <= std::numeric_limits<std::size_t>::max() / sizeof(float);
        };
        std::size_t spectral_count = 0, layer_band_count = 0, interface_band_count = 0, partial_count = 0;
        if (!checked_product(columns, layers, g_points, spectral_count) ||
            !checked_product(columns, layers, bands, layer_band_count) ||
            !checked_product(columns, layers + 1, bands, interface_band_count) ||
            !checked_product(columns, layers + 1, g_points, partial_count) ||
            partial_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            return false;
        const auto valid_nonnegative = [](const float* values, std::size_t count)
        {
            for (std::size_t i = 0; i < count; ++i)
                if (!std::isfinite(values[i]) || values[i] < 0.0f)
                    return false;
            return true;
        };
        if (!valid_nonnegative(dispatch.absorption_optical_depth, spectral_count) ||
            !valid_nonnegative(dispatch.planck_fraction, spectral_count) ||
            !valid_nonnegative(dispatch.layer_planck, layer_band_count) ||
            !valid_nonnegative(dispatch.interface_planck, interface_band_count) ||
            !valid_nonnegative(dispatch.surface_planck, columns * bands) ||
            !valid_nonnegative(dispatch.band_width_cm, bands) ||
            (dispatch.cloud_absorption_optical_depth &&
             !valid_nonnegative(dispatch.cloud_absorption_optical_depth, layer_band_count)))
            return false;
        for (std::size_t band = 0; band < bands; ++band)
            if (dispatch.band_width_cm[band] <= 0.0f)
                return false;
        try
        {
            std::vector<RrtmgLongwaveColumnParameters> parameters(columns);
            for (std::size_t column = 0; column < columns; ++column)
            {
                float e = dispatch.surface_emissivity[column], w = dispatch.precipitable_water_cm[column];
                if (!std::isfinite(e) || e < 0.0f || e > 1.0f || !std::isfinite(w) || w < 0.0f)
                    return false;
                parameters[column].surface_emissivity    = e;
                parameters[column].precipitable_water_cm = w;
            }
            std::vector<float> zero_clouds;
            const float*       cloud = dispatch.cloud_absorption_optical_depth;
            if (!cloud)
            {
                zero_clouds.assign(layer_band_count, 0.0f);
                cloud = zero_clouds.data();
            }
            std::vector<float>                      partial_up(partial_count, 0.0f), partial_down(partial_count, 0.0f);
            RrtmgLongwaveTransferBatchPushConstants constants{};
            constants.layers                   = dispatch.layers;
            constants.columns                  = dispatch.columns;
            constants.has_clouds               = dispatch.cloud_absorption_optical_depth ? 1 : 0;
            const float*        inputs[8]      = {dispatch.absorption_optical_depth,
                                                  dispatch.planck_fraction,
                                                  dispatch.layer_planck,
                                                  dispatch.interface_planck,
                                                  dispatch.surface_planck,
                                                  dispatch.band_width_cm,
                                                  cloud,
                                                  reinterpret_cast<const float*>(parameters.data())};
            const VkDeviceSize  input_bytes[8] = {spectral_count * sizeof(float),
                                                  spectral_count * sizeof(float),
                                                  layer_band_count * sizeof(float),
                                                  interface_band_count * sizeof(float),
                                                  columns * bands * sizeof(float),
                                                  bands * sizeof(float),
                                                  layer_band_count * sizeof(float),
                                                  columns * sizeof(RrtmgLongwaveColumnParameters)};
            float*              outputs[2]     = {partial_up.data(), partial_down.data()};
            const std::uint32_t invocations    = static_cast<std::uint32_t>(columns * g_points);
            if (!dispatch_multi_field_kernel(pipeline(KernelId::RrtmgLongwaveTransferBatch),
                                             KernelId::RrtmgLongwaveTransferBatch,
                                             &constants,
                                             inputs,
                                             8,
                                             outputs,
                                             2,
                                             static_cast<int>(partial_count),
                                             1,
                                             1,
                                             invocations,
                                             input_bytes))
                return false;
            const std::size_t  result_count = columns * (layers + 1);
            std::vector<float> upward(result_count, 0.0f), downward(result_count, 0.0f);
            for (std::size_t column = 0; column < columns; ++column)
                for (std::size_t g = 0; g < g_points; ++g)
                    for (std::size_t interface_index = 0; interface_index <= layers; ++interface_index)
                    {
                        std::size_t p = (column * g_points + g) * (layers + 1) + interface_index,
                                    r = column * (layers + 1) + interface_index;
                        upward[r] += partial_up[p];
                        downward[r] += partial_down[p];
                    }
            for (std::size_t i = 0; i < result_count; ++i)
                if (!std::isfinite(upward[i]) || upward[i] < 0.0f || !std::isfinite(downward[i]) || downward[i] < 0.0f)
                    return false;
            std::memcpy(dispatch.upward_wm2, upward.data(), result_count * sizeof(float));
            std::memcpy(dispatch.downward_wm2, downward.data(), result_count * sizeof(float));
            return true;
        }
        catch (const std::bad_alloc&)
        {
            return false;
        }
        catch (const std::length_error&)
        {
            return false;
        }
#else
        (void)dispatch;
        return false;
#endif
    }

    bool dispatch_rrtmg_shortwave_transfer_batch(const RrtmgShortwaveTransferBatchDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        constexpr std::size_t g_points = 112, bands = 14;
        const auto            reject = [](const char* reason)
        {
            if (log_debug_enabled())
                std::cerr << "[COMPUTE][VULKAN] rejected RRTMG shortwave batch: " << reason << '\n';
            return false;
        };
        const bool any_cloud =
            dispatch.cloud_optical_depth || dispatch.cloud_single_scattering_albedo || dispatch.cloud_asymmetry;
        const bool all_cloud =
            dispatch.cloud_optical_depth && dispatch.cloud_single_scattering_albedo && dispatch.cloud_asymmetry;
        if (!supports(RadiationKernelOp::RrtmgShortwaveTransferBatch) || dispatch.columns <= 0 ||
            dispatch.layers <= 0 || dispatch.layers == std::numeric_limits<int>::max() || any_cloud != all_cloud ||
            !dispatch.absorption_optical_depth || !dispatch.rayleigh_optical_depth || !dispatch.solar_source_fraction ||
            !dispatch.cosine_zenith || !dispatch.solar_irradiance_wm2 || !dispatch.surface_albedo ||
            !dispatch.upward_wm2 || !dispatch.downward_wm2 || dispatch.upward_wm2 == dispatch.downward_wm2)
            return reject("invalid dispatch contract");
        const std::size_t columns         = static_cast<std::size_t>(dispatch.columns),
                          layers          = static_cast<std::size_t>(dispatch.layers);
        const auto        checked_product = [](std::size_t a, std::size_t b, std::size_t c, std::size_t& r)
        {
            if (!a || !b || !c || a > std::numeric_limits<std::size_t>::max() / b)
                return false;
            r = a * b;
            if (r > std::numeric_limits<std::size_t>::max() / c)
                return false;
            r *= c;
            return r <= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) &&
                   r <= std::numeric_limits<std::size_t>::max() / sizeof(float);
        };
        std::size_t spectral_count = 0, cloud_count = 0, partial_count = 0;
        if (!checked_product(columns, layers, g_points, spectral_count) ||
            !checked_product(columns, layers, bands, cloud_count) ||
            !checked_product(columns, layers + 1, g_points, partial_count) ||
            partial_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            return reject("invalid or overflowing dimensions");
        const auto valid_range = [](const float* v, std::size_t n, float maximum)
        {
            for (std::size_t i = 0; i < n; ++i)
                if (!std::isfinite(v[i]) || v[i] < 0.0f || v[i] > maximum)
                    return false;
            return true;
        };
        if (!valid_range(dispatch.absorption_optical_depth, spectral_count, std::numeric_limits<float>::max()) ||
            !valid_range(dispatch.rayleigh_optical_depth, spectral_count, std::numeric_limits<float>::max()) ||
            !valid_range(dispatch.solar_source_fraction, columns * g_points, std::numeric_limits<float>::max()) ||
            (all_cloud && (!valid_range(dispatch.cloud_optical_depth, cloud_count, std::numeric_limits<float>::max()) ||
                           !valid_range(dispatch.cloud_single_scattering_albedo, cloud_count, 1.0f) ||
                           !valid_range(dispatch.cloud_asymmetry, cloud_count, 1.0f))))
            return reject("non-finite or out-of-range optical input");
        try
        {
            std::vector<RrtmgShortwaveColumnParameters> parameters(columns);
            for (std::size_t column = 0; column < columns; ++column)
            {
                float mu = dispatch.cosine_zenith[column], solar = dispatch.solar_irradiance_wm2[column],
                      albedo = dispatch.surface_albedo[column];
                if (!std::isfinite(mu) || mu < 0.0f || mu > 1.0f || !std::isfinite(solar) || solar < 0.0f ||
                    !std::isfinite(albedo) || albedo < 0.0f || albedo > 1.0f)
                    return reject("invalid column parameters");
                float sum = 0.0f;
                for (std::size_t g = 0; g < g_points; ++g)
                    sum += dispatch.solar_source_fraction[column * g_points + g];
                if ((!std::isfinite(sum) || sum <= 0.0f) && mu != 0.0f && solar != 0.0f)
                    return reject("invalid daytime solar-source normalization");
                parameters[column] = {mu, solar, albedo, std::isfinite(sum) && sum > 0.0f ? sum : 1.0f};
            }
            std::vector<float> zeros;
            const float*       clouds[3] = {
                dispatch.cloud_optical_depth, dispatch.cloud_single_scattering_albedo, dispatch.cloud_asymmetry};
            if (!all_cloud)
            {
                zeros.assign(cloud_count, 0.0f);
                for (const float*& cloud : clouds)
                    cloud = zeros.data();
            }
            std::array<std::vector<float>, 12> candidates;
            for (auto& candidate : candidates)
                candidate.assign(partial_count, 0.0f);
            RrtmgShortwaveTransferBatchPushConstants constants{};
            constants.layers                  = dispatch.layers;
            constants.columns                 = dispatch.columns;
            constants.has_clouds              = all_cloud ? 1 : 0;
            const float*       inputs[7]      = {dispatch.absorption_optical_depth,
                                                 dispatch.rayleigh_optical_depth,
                                                 dispatch.solar_source_fraction,
                                                 clouds[0],
                                                 clouds[1],
                                                 clouds[2],
                                                 reinterpret_cast<const float*>(parameters.data())};
            const VkDeviceSize input_bytes[7] = {spectral_count * sizeof(float),
                                                 spectral_count * sizeof(float),
                                                 columns * g_points * sizeof(float),
                                                 cloud_count * sizeof(float),
                                                 cloud_count * sizeof(float),
                                                 cloud_count * sizeof(float),
                                                 columns * sizeof(RrtmgShortwaveColumnParameters)};
            float*             outputs[12];
            for (int i = 0; i < 12; ++i)
                outputs[i] = candidates[static_cast<std::size_t>(i)].data();
            const std::uint32_t invocations = static_cast<std::uint32_t>(columns * g_points);
            if (!dispatch_multi_field_kernel(pipeline(KernelId::RrtmgShortwaveTransferBatch),
                                             KernelId::RrtmgShortwaveTransferBatch,
                                             &constants,
                                             inputs,
                                             7,
                                             outputs,
                                             12,
                                             static_cast<int>(partial_count),
                                             1,
                                             1,
                                             invocations,
                                             input_bytes))
                return reject("device submission failed");
            const std::size_t  result_count = columns * (layers + 1);
            std::vector<float> upward(result_count, 0.0f), downward(result_count, 0.0f);
            for (std::size_t column = 0; column < columns; ++column)
                for (std::size_t g = 0; g < g_points; ++g)
                    for (std::size_t interface_index = 0; interface_index <= layers; ++interface_index)
                    {
                        std::size_t p = (column * g_points + g) * (layers + 1) + interface_index,
                                    r = column * (layers + 1) + interface_index;
                        upward[r] += candidates[0][p];
                        downward[r] += candidates[1][p];
                    }
            for (std::size_t i = 0; i < result_count; ++i)
                if (!std::isfinite(upward[i]) || upward[i] < 0.0f || !std::isfinite(downward[i]) || downward[i] < 0.0f)
                {
                    if (log_debug_enabled())
                        std::cerr << "[COMPUTE][VULKAN] invalid RRTMG shortwave batch output at index " << i
                                  << ": upward=" << upward[i] << ", downward=" << downward[i] << '\n';
                    return reject("device output was non-finite or negative");
                }
            std::memcpy(dispatch.upward_wm2, upward.data(), result_count * sizeof(float));
            std::memcpy(dispatch.downward_wm2, downward.data(), result_count * sizeof(float));
            return true;
        }
        catch (const std::bad_alloc&)
        {
            return false;
        }
        catch (const std::length_error&)
        {
            return false;
        }
#else
        (void)dispatch;
        return false;
#endif
    }

    bool supports_vertical_flux_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return pipeline(KernelId::TvdVerticalFlux).is_ready();
#else
        return false;
#endif
    }

    bool supports_implicit_diffusion_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return pipeline(KernelId::ImplicitDiffusionStencil).is_ready() &&
               pipeline(KernelId::ImplicitDiffusionVector).is_ready();
#else
        return false;
#endif
    }

#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
    template <class Constants, std::size_t Inputs>
    bool dispatch_implicit_fields(KernelId                                kernel,
                                  const Constants&                        constants,
                                  const std::array<const float*, Inputs>& inputs,
                                  const std::array<std::size_t, Inputs>&  input_sizes,
                                  const std::array<float*, 5>&            outputs,
                                  const std::array<std::size_t, 5>&       output_sizes,
                                  int                                     nx,
                                  int                                     ny,
                                  int                                     nz,
                                  std::size_t                             columns)
    {
        std::array<VkDeviceSize, Inputs> input_bytes;
        for (std::size_t i = 0; i < Inputs; ++i)
            input_bytes[i] = input_sizes[i] * sizeof(float);
        std::array<std::vector<float>, 5> candidates;
        std::array<float*, 5>             destinations;
        for (std::size_t i = 0; i < 5; ++i)
        {
            // The common dispatcher stages full volumes; only the declared
            // column prefix of a reduction buffer is published below.
            candidates[i].resize(static_cast<std::size_t>(nx) * ny * nz, 0.0f);
            destinations[i] = candidates[i].data();
        }
        if (!dispatch_multi_field_kernel(pipeline(kernel),
                                         kernel,
                                         &constants,
                                         inputs.data(),
                                         Inputs,
                                         destinations.data(),
                                         5,
                                         nx,
                                         ny,
                                         nz,
                                         static_cast<uint32_t>(columns),
                                         input_bytes.data()))
            return false;
        for (std::size_t f = 0; f < 5; ++f)
            for (float value : candidates[f])
                if (!std::isfinite(value) || (f >= 3 && value < 0) ||
                    (kernel == KernelId::ImplicitDiffusionStencil && f == 1 && value < 0))
                    return false;
        for (std::size_t f = 0; f < 5; ++f)
            std::copy_n(candidates[f].begin(), output_sizes[f], outputs[f]);
        return true;
    }
#endif

    bool dispatch_implicit_stencil(const ImplicitDiffusionStencilDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        std::size_t n = 0, c = 0;
        if (!supports_implicit_diffusion_dispatch() || !nimbus::compute::implicit_contract::validate(d, n, c))
            return false;
        const auto& a = d.diffusion;
        return dispatch_implicit_fields(
            KernelId::ImplicitDiffusionStencil,
            nimbus::compute::implicit_contract::constants(d),
            std::array<const float*, 4>{a.field, a.rho, a.dz, a.vertical_K ? a.vertical_K : a.rho},
            std::array<std::size_t, 4>{n, n, static_cast<std::size_t>(a.nz), n},
            std::array<float*, 5>{a.tendency, d.diagonal, d.dot, d.preconditioned_norm, d.norm},
            std::array<std::size_t, 5>{n, n, c, c, c},
            a.nx,
            a.ny,
            a.nz,
            c);
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_implicit_vector(const ImplicitDiffusionVectorDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        std::size_t n = 0, c = 0;
        if (!supports_implicit_diffusion_dispatch() || !nimbus::compute::implicit_contract::validate(d, n, c))
            return false;
        return dispatch_implicit_fields(
            KernelId::ImplicitDiffusionVector,
            nimbus::compute::implicit_contract::constants(d),
            std::array<const float*, 7>{d.original, d.rhs, d.diagonal, d.solution, d.residual, d.direction, d.image},
            std::array<std::size_t, 7>{n, n, n, n, n, n, n},
            std::array<float*, 5>{d.next_solution, d.next_residual, d.next_direction, d.preconditioned_norm, d.norm},
            std::array<std::size_t, 5>{n, n, n, c, c},
            d.nx,
            d.ny,
            d.nz,
            c);
#else
        (void)d;
        return false;
#endif
    }

    bool supports_diffusion_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Native live-density Cartesian finite-volume tendency ABI.
        return pipeline(KernelId::Diffusion).is_ready();
#else
        return false;
#endif
    }

    bool supports_cartesian_cgrid_slow_tendencies_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Cartesian-only shader (GPU.md section 2.0 routing rule).
        if (coord_system_ != CoordinateSystem::Cartesian)
            return false;
        return pipeline(KernelId::CartesianCgridSlowTendencies).is_ready();
#else
        return false;
#endif
    }

    bool supports_cartesian_tendencies_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Cartesian-only shader (GPU.md section 2.0 routing rule).
        if (coord_system_ != CoordinateSystem::Cartesian)
            return false;
        return pipeline(KernelId::CartesianTendencies).is_ready();
#else
        return false;
#endif
    }

    bool supports_advection_x_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Cartesian-only shader (GPU.md section 2.0 routing rule).
        if (coord_system_ != CoordinateSystem::Cartesian)
            return false;
        return pipeline(KernelId::AdvectX).is_ready();
#else
        return false;
#endif
    }

    bool supports_advection_y_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Cartesian-only shader (GPU.md section 2.0 routing rule).
        if (coord_system_ != CoordinateSystem::Cartesian)
            return false;
        return pipeline(KernelId::AdvectY).is_ready();
#else
        return false;
#endif
    }

    bool supports_kessler_pointwise_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return pipeline(KernelId::KesslerPointwise).is_ready();
#else
        return false;
#endif
    }

    bool supports_kessler_sedimentation_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return pipeline(KernelId::KesslerSedimentation).is_ready();
#else
        return false;
#endif
    }

    bool supports_acoustic_pressure_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return pipeline(KernelId::AcousticPressureCartesian).is_ready();
#else
        return false;
#endif
    }

    bool supports_acoustic_momentum_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return pipeline(KernelId::AcousticMomentumCartesian).is_ready();
#else
        return false;
#endif
    }

    bool dispatch_acoustic_pressure(const AcousticPressureDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        const float* u_data     = dispatch.u_data;
        const float* v_data     = dispatch.v_data;
        const float* w_data     = dispatch.w_data;
        const float* rho_in     = dispatch.rho_in;
        const float* p_in       = dispatch.p_in;
        float*       rho_out    = dispatch.rho_out;
        float*       p_out      = dispatch.p_out;
        const int    nr         = dispatch.nr;
        const int    nth        = dispatch.nth;
        const int    nz         = dispatch.nz;
        const float  dr_val     = dispatch.dr;
        const float  dtheta_val = dispatch.dtheta;
        const float  dz_val     = dispatch.dz;
        const float  gamma_val  = dispatch.gamma_val;
        const float  dt_small   = dispatch.dt_small;
        const float  rho_floor  = dispatch.rho_floor;
        const float  p_floor    = dispatch.p_floor;

        auto& pipeline = active_acoustic_pressure_pipeline();
        if (!pipeline.is_ready())
            return false;

        AcousticPressurePushConstants pc{};
        pc.nr        = nr;
        pc.nth       = nth;
        pc.nz        = nz;
        pc.dr        = dr_val;
        pc.dtheta    = dtheta_val;
        pc.dz_val    = dz_val;
        pc.gamma_val = gamma_val;
        pc.dt_small  = dt_small;
        pc.rho_floor = rho_floor;
        pc.p_floor   = p_floor;

        uint32_t     total_points  = 0;
        VkDeviceSize checked_bytes = 0;
        if (!checked_field_dispatch(nr, nth, nz, total_points, checked_bytes, "acoustic pressure"))
            return false;
        const float* inputs[5]            = {u_data, v_data, w_data, rho_in, p_in};
        float*       outputs[2]           = {rho_out, p_out};
        const bool   output_write_only[2] = {true, true};

        return dispatch_multi_field_kernel(pipeline,
                                           KernelId::AcousticPressureCartesian,
                                           &pc,
                                           inputs,
                                           5,
                                           outputs,
                                           2,
                                           nr,
                                           nth,
                                           nz,
                                           total_points,
                                           nullptr,
                                           nullptr,
                                           output_write_only);
#else
        (void)dispatch;
        return false;
#endif
    }

    bool dispatch_acoustic_momentum(const AcousticMomentumDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        const float* rho_data     = dispatch.rho_data;
        const float* p_data       = dispatch.p_data;
        const float* u_in         = dispatch.u_in;
        const float* v_in         = dispatch.v_in;
        const float* w_in         = dispatch.w_in;
        float*       u_out        = dispatch.u_out;
        float*       v_out        = dispatch.v_out;
        float*       w_out        = dispatch.w_out;
        const int    nr           = dispatch.nr;
        const int    nth          = dispatch.nth;
        const int    nz           = dispatch.nz;
        const float  dr_val       = dispatch.dr;
        const float  dtheta_val   = dispatch.dtheta;
        const float  dz_val       = dispatch.dz;
        const float  dt_small     = dispatch.dt_small;
        const float  wind_clamp_h = dispatch.wind_clamp_h;
        const float  wind_clamp_v = dispatch.wind_clamp_v;

        auto& pipeline = active_acoustic_momentum_pipeline();
        if (!pipeline.is_ready())
            return false;

        AcousticMomentumPushConstants pc{};
        pc.nr           = nr;
        pc.nth          = nth;
        pc.nz           = nz;
        pc.dr           = dr_val;
        pc.dtheta       = dtheta_val;
        pc.dz_val       = dz_val;
        pc.dt_small     = dt_small;
        pc.wind_clamp_h = wind_clamp_h;
        pc.wind_clamp_v = wind_clamp_v;
        pc.padding      = 0.0f;

        uint32_t     total_points  = 0;
        VkDeviceSize checked_bytes = 0;
        if (!checked_field_dispatch(nr, nth, nz, total_points, checked_bytes, "acoustic momentum"))
            return false;
        const float* inputs[5]            = {rho_data, p_data, u_in, v_in, w_in};
        float*       outputs[3]           = {u_out, v_out, w_out};
        const bool   output_write_only[3] = {true, true, true};

        return dispatch_multi_field_kernel(pipeline,
                                           KernelId::AcousticMomentumCartesian,
                                           &pc,
                                           inputs,
                                           5,
                                           outputs,
                                           3,
                                           nr,
                                           nth,
                                           nz,
                                           total_points,
                                           nullptr,
                                           nullptr,
                                           output_write_only);
#else
        (void)dispatch;
        return false;
#endif
    }

    // ── Fused acoustic substep (pressure + momentum in one submission) ──

    bool supports_acoustic_substep_fused_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Readiness of the Cartesian pipelines, mirroring
        // supports_acoustic_pressure_dispatch / _momentum_dispatch.
        return pipeline(KernelId::AcousticPressureCartesian).is_ready() &&
               pipeline(KernelId::AcousticMomentumCartesian).is_ready();
#else
        return false;
#endif
    }

    bool dispatch_acoustic_substep_fused(const AcousticSubstepFusedDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        float*      u            = dispatch.u;
        float*      v            = dispatch.v;
        float*      w            = dispatch.w;
        float*      rho          = dispatch.rho;
        float*      p            = dispatch.p;
        const int   nr           = dispatch.nr;
        const int   nth          = dispatch.nth;
        const int   nz           = dispatch.nz;
        const float dr_val       = dispatch.dr;
        const float dtheta_val   = dispatch.dtheta;
        const float dz_val       = dispatch.dz;
        const float gamma_val    = dispatch.gamma_val;
        const float dt_small     = dispatch.dt_small;
        const float rho_floor    = dispatch.rho_floor;
        const float p_floor      = dispatch.p_floor;
        const float wind_clamp_h = dispatch.wind_clamp_h;
        const float wind_clamp_v = dispatch.wind_clamp_v;

        if (!supports_acoustic_substep_fused_dispatch())
            return false;

        uint32_t     total_points = 0;
        VkDeviceSize field_bytes  = 0;
        if (!checked_field_dispatch(nr, nth, nz, total_points, field_bytes, "fused acoustic substep"))
            return false;

        // 5 pool slots: u(0), v(1), w(2), rho(3), p(4)
        constexpr int kSlotCount = 5;
        int           slots[kSlotCount];
        if (!acquire_pool_slots(field_bytes, kSlotCount, slots))
        {
            log_vulkan_warning("failed to acquire pool slots for fused acoustic substep");
            return false;
        }
        const int slot_u   = slots[0];
        const int slot_v   = slots[1];
        const int slot_w   = slots[2];
        const int slot_rho = slots[3];
        const int slot_p   = slots[4];

        const bool unified  = device_context_.has_unified_memory;
        auto       host_ptr = [&](int slot) -> void*
        { return unified ? buffer_pool_.device(slot).mapped : buffer_pool_.staging(slot).mapped; };

        // Upload all 5 fields
        tracked_memcpy(host_ptr(slot_u), u, field_bytes);
        tracked_memcpy(host_ptr(slot_v), v, field_bytes);
        tracked_memcpy(host_ptr(slot_w), w, field_bytes);
        tracked_memcpy(host_ptr(slot_rho), rho, field_bytes);
        tracked_memcpy(host_ptr(slot_p), p, field_bytes);

        // Record command buffer
        vkResetCommandBuffer_(cmd_buf_, 0);
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (begin_profiled_command_buffer(begin_info) != VK_SUCCESS)
        {
            release_pool_slots(slots, kSlotCount);
            return false;
        }

        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;

        // ── H2D transfer ──
        if (unified)
        {
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }
        else
        {
            VkBufferCopy copy_region{};
            copy_region.size = field_bytes;
            for (int i = 0; i < kSlotCount; ++i)
            {
                tracked_cmd_copy_buffer(
                    buffer_pool_.staging(slots[i]).buffer, buffer_pool_.device(slots[i]).buffer, 1, &copy_region);
            }

            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }

        const uint32_t workgroup_count = 1u + (total_points - 1u) / 64u;

        // Cartesian collocated acoustic pipelines.
        auto& pressure_pipeline = active_acoustic_pressure_pipeline();
        auto& momentum_pipeline = active_acoustic_momentum_pipeline();

        // ── Dispatch 1: Acoustic pressure (reads u,v,w,rho,p; writes rho,p) ──
        {
            // Pressure shader bindings: 0=u, 1=v, 2=w, 3=rho_in, 4=p_in, 5=rho_out, 6=p_out
            // For in-place: rho_in == rho_out (slot_rho), p_in == p_out (slot_p)
            int pressure_bindings[7] = {slot_u, slot_v, slot_w, slot_rho, slot_p, slot_rho, slot_p};
            update_pooled_descriptor_set(pressure_pipeline, pressure_bindings, field_bytes, 7);

            AcousticPressurePushConstants pc{};
            pc.nr        = nr;
            pc.nth       = nth;
            pc.nz        = nz;
            pc.dr        = dr_val;
            pc.dtheta    = dtheta_val;
            pc.dz_val    = dz_val;
            pc.gamma_val = gamma_val;
            pc.dt_small  = dt_small;
            pc.rho_floor = rho_floor;
            pc.p_floor   = p_floor;

            vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pressure_pipeline.pipeline);
            vkCmdBindDescriptorSets_(cmd_buf_,
                                     VK_PIPELINE_BIND_POINT_COMPUTE,
                                     pressure_pipeline.pipeline_layout,
                                     0,
                                     1,
                                     &pressure_pipeline.descriptor_set,
                                     0,
                                     nullptr);
            vkCmdPushConstants_(
                cmd_buf_, pressure_pipeline.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            record_linear_group_dispatch(workgroup_count, KernelId::AcousticPressureCartesian);
        }

        // ── Barrier: pressure writes complete before momentum reads ──
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier_(cmd_buf_,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0,
                              1,
                              &barrier,
                              0,
                              nullptr,
                              0,
                              nullptr);

        // ── Dispatch 2: Acoustic momentum (reads rho,p [updated],u,v,w; writes u,v,w) ──
        {
            // Momentum shader bindings: 0=rho, 1=p, 2=u_in, 3=v_in, 4=w_in, 5=u_out, 6=v_out, 7=w_out
            // For in-place: u_in == u_out (slot_u), etc.
            int momentum_bindings[8] = {slot_rho, slot_p, slot_u, slot_v, slot_w, slot_u, slot_v, slot_w};
            update_pooled_descriptor_set(momentum_pipeline, momentum_bindings, field_bytes, 8);

            AcousticMomentumPushConstants pc{};
            pc.nr           = nr;
            pc.nth          = nth;
            pc.nz           = nz;
            pc.dr           = dr_val;
            pc.dtheta       = dtheta_val;
            pc.dz_val       = dz_val;
            pc.dt_small     = dt_small;
            pc.wind_clamp_h = wind_clamp_h;
            pc.wind_clamp_v = wind_clamp_v;
            pc.padding      = 0.0f;

            vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, momentum_pipeline.pipeline);
            vkCmdBindDescriptorSets_(cmd_buf_,
                                     VK_PIPELINE_BIND_POINT_COMPUTE,
                                     momentum_pipeline.pipeline_layout,
                                     0,
                                     1,
                                     &momentum_pipeline.descriptor_set,
                                     0,
                                     nullptr);
            vkCmdPushConstants_(
                cmd_buf_, momentum_pipeline.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            record_linear_group_dispatch(workgroup_count, KernelId::AcousticMomentumCartesian);
        }

        // ── D2H transfer (all 5 fields modified) ──
        if (unified)
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }
        else
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);

            VkBufferCopy copy_region{};
            copy_region.size = field_bytes;
            for (int i = 0; i < kSlotCount; ++i)
            {
                tracked_cmd_copy_buffer(
                    buffer_pool_.device(slots[i]).buffer, buffer_pool_.staging(slots[i]).buffer, 1, &copy_region);
            }
        }

        if (end_profiled_command_buffer() != VK_SUCCESS)
        {
            release_pool_slots(slots, kSlotCount);
            return false;
        }

        // Submit and wait
        VkSubmitInfo submit_info{};
        submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers    = &cmd_buf_;
        if (reset_fence_and_submit("dispatch_acoustic_substep_fused", submit_info) != VK_SUCCESS)
        {
            release_pool_slots(slots, kSlotCount);
            return false;
        }

        if (timed_wait_fence("dispatch_acoustic_substep_fused") != VK_SUCCESS)
        {
            release_pool_slots(slots, kSlotCount);
            return false;
        }

        // Download all 5 fields
        tracked_memcpy(u, host_ptr(slot_u), field_bytes);
        tracked_memcpy(v, host_ptr(slot_v), field_bytes);
        tracked_memcpy(w, host_ptr(slot_w), field_bytes);
        tracked_memcpy(rho, host_ptr(slot_rho), field_bytes);
        tracked_memcpy(p, host_ptr(slot_p), field_bytes);

        release_pool_slots(slots, kSlotCount);
        return true;
#else
        (void)dispatch;
        return false;
#endif
    }

    // ── Cartesian C-grid acoustic substeps (GPU audit Phase 3.1) ──

    bool supports_acoustic_continuity_mass_flux_cartesian_agrid_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return coord_system_ == CoordinateSystem::Cartesian &&
            pipeline(KernelId::AcousticContinuityMassFluxCartesianAgrid).is_ready();
#else
        return false;
#endif
    }

    bool dispatch_acoustic_continuity_mass_flux_cartesian_agrid(
        const AcousticContinuityMassFluxCartesianAgridDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!supports_acoustic_continuity_mass_flux_cartesian_agrid_dispatch() ||
            d.nx < 4 || d.ny < 4 || d.nz < 4 || d.base_len < d.nz ||
            d.u == nullptr || d.v == nullptr || d.w == nullptr ||
            d.rho == nullptr || d.p == nullptr || d.p0_base_data == nullptr ||
            d.drho_dt == nullptr ||
            d.mass_flux_x == nullptr || d.mass_flux_y == nullptr ||
            d.mass_flux_z == nullptr ||
            !std::isfinite(d.dx) || d.dx <= 0.0f ||
            !std::isfinite(d.dy) || d.dy <= 0.0f ||
            !std::isfinite(d.dz) || d.dz <= 0.0f ||
            !std::isfinite(d.dt_small) || d.dt_small <= 0.0f ||
            !std::isfinite(d.mass_flux_weight) ||
            (d.accumulate_mass_flux && d.mass_flux_weight <= 0.0f) ||
            !std::isfinite(d.frame_velocity_u_ms) ||
            !std::isfinite(d.frame_velocity_v_ms) ||
            d.lateral_bc_west < 0 || d.lateral_bc_west > 5 ||
            d.lateral_bc_east < 0 || d.lateral_bc_east > 5 ||
            d.lateral_bc_south < 0 || d.lateral_bc_south > 5 ||
            d.lateral_bc_north < 0 || d.lateral_bc_north > 5 ||
            d.top_bc < 0 || d.top_bc > 2)
        {
            return false;
        }
        const float* inputs[6] = {d.u, d.v, d.w, d.rho, d.p, d.p0_base_data};
        float* outputs[4] = {
            d.drho_dt, d.mass_flux_x, d.mass_flux_y, d.mass_flux_z};
        for (int out = 0; out < 4; ++out)
        {
            for (const float* input : inputs)
                if (outputs[out] == input) return false;
            for (int prior = 0; prior < out; ++prior)
                if (outputs[out] == outputs[prior]) return false;
        }

        AcousticContinuityMassFluxCartesianAgridPushConstants pc{};
        pc.nx = d.nx; pc.ny = d.ny; pc.nz = d.nz;
        pc.dx = d.dx; pc.dy = d.dy; pc.dz = d.dz;
        pc.dt_small = d.dt_small;
        pc.lateral_bc_west = d.lateral_bc_west;
        pc.lateral_bc_east = d.lateral_bc_east;
        pc.lateral_bc_south = d.lateral_bc_south;
        pc.lateral_bc_north = d.lateral_bc_north;
        pc.top_bc = d.top_bc;
        pc.accumulate_mass_flux = d.accumulate_mass_flux ? 1 : 0;
        pc.mass_flux_weight = d.mass_flux_weight;
        pc.frame_velocity_u_ms = d.frame_velocity_u_ms;
        pc.frame_velocity_v_ms = d.frame_velocity_v_ms;

        uint32_t total_points = 0;
        VkDeviceSize field_bytes = 0;
        if (!checked_field_dispatch(d.nx, d.ny, d.nz, total_points,
                                    field_bytes, "A-grid continuity/mass flux"))
            return false;
        const VkDeviceSize profile_bytes =
            static_cast<VkDeviceSize>(d.nz) * sizeof(float);
        const VkDeviceSize input_bytes[6] = {
            0, 0, 0, 0, 0, profile_bytes};
        const bool output_write_only[4] = {true, false, false, false};
        return dispatch_multi_field_kernel(
            pipeline(KernelId::AcousticContinuityMassFluxCartesianAgrid),
            KernelId::AcousticContinuityMassFluxCartesianAgrid,
            &pc, inputs, 6, outputs, 4,
            d.nx, d.ny, d.nz, total_points,
            input_bytes, nullptr, output_write_only);
#else
        (void)d;
        return false;
#endif
    }

    bool supports_acoustic_pressure_eos_cartesian_agrid_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return coord_system_ == CoordinateSystem::Cartesian &&
            pipeline(KernelId::AcousticPressureEosCartesianAgrid).is_ready();
#else
        return false;
#endif
    }

    bool supports_acoustic_momentum_cartesian_agrid_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return coord_system_ == CoordinateSystem::Cartesian &&
            pipeline(KernelId::AcousticMomentumCartesianAgrid).is_ready();
#else
        return false;
#endif
    }

    bool supports_acoustic_null_mode_constraint_cartesian_agrid_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return coord_system_ == CoordinateSystem::Cartesian &&
            pipeline(KernelId::AcousticNullModeConstraintCartesianAgrid).is_ready();
#else
        return false;
#endif
    }

    bool supports_acoustic_boundary_cartesian_agrid_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return coord_system_ == CoordinateSystem::Cartesian &&
            pipeline(KernelId::AcousticBoundaryCartesianAgrid).is_ready();
#else
        return false;
#endif
    }

    bool dispatch_acoustic_pressure_eos_cartesian_agrid(
        const AcousticPressureEosCartesianAgridDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        uint32_t total_points = 0;
        VkDeviceSize field_bytes = 0;
        if (!supports_acoustic_pressure_eos_cartesian_agrid_dispatch() ||
            d.nx < 4 || d.ny < 4 || d.nz < 4 ||
            !checked_field_dispatch(d.nx, d.ny, d.nz, total_points, field_bytes,
                                    "A-grid pressure/EOS") ||
            d.rho_in == nullptr || d.theta == nullptr || d.drho_dt == nullptr ||
            d.rho_out == nullptr || d.p_out == nullptr || d.rho_out == d.p_out ||
            !std::isfinite(d.dt_small) || d.dt_small <= 0.0f ||
            !std::isfinite(d.reference_pressure) || d.reference_pressure <= 0.0f ||
            !std::isfinite(d.gas_constant_dry_air) || d.gas_constant_dry_air <= 0.0f ||
            !std::isfinite(d.gamma) || d.gamma <= 0.0f ||
            !(d.rho_min > 0.0f && d.rho_max > d.rho_min) ||
            !(d.p_min > 0.0f && d.p_max > d.p_min))
        {
            return false;
        }
        for (uint32_t n = 0; n < total_points; ++n)
        {
            if (!std::isfinite(d.rho_in[n]) || d.rho_in[n] <= 0.0f ||
                !std::isfinite(d.theta[n]) || d.theta[n] <= 0.0f ||
                !std::isfinite(d.drho_dt[n]))
            {
                return false;
            }
        }

        AcousticPressureEosCartesianAgridPushConstants pc{};
        pc.nx = d.nx;
        pc.ny = d.ny;
        pc.nz = d.nz;
        pc.dt_small = d.dt_small;
        pc.reference_pressure = d.reference_pressure;
        pc.gas_constant_dry_air = d.gas_constant_dry_air;
        pc.gamma = d.gamma;
        pc.rho_min = d.rho_min;
        pc.rho_max = d.rho_max;
        pc.p_min = d.p_min;
        pc.p_max = d.p_max;

        const float* inputs[3] = {d.rho_in, d.theta, d.drho_dt};
        std::vector<float> rho_candidate(total_points);
        std::vector<float> p_candidate(total_points);
        float* outputs[2] = {rho_candidate.data(), p_candidate.data()};
        const bool write_only[2] = {true, true};
        if (!dispatch_multi_field_kernel(
                pipeline(KernelId::AcousticPressureEosCartesianAgrid),
                KernelId::AcousticPressureEosCartesianAgrid, &pc,
                inputs, 3, outputs, 2, d.nx, d.ny, d.nz, total_points,
                nullptr, nullptr, write_only))
        {
            return false;
        }
        for (uint32_t n = 0; n < total_points; ++n)
        {
            if (!std::isfinite(rho_candidate[n]) || rho_candidate[n] <= 0.0f ||
                !std::isfinite(p_candidate[n]) || p_candidate[n] <= 0.0f)
            {
                return false;
            }
        }
        if (d.completed_observer != nullptr)
        {
            d.completed_observer(d.completed_observer_context,
                                 rho_candidate.data(), p_candidate.data());
        }
        std::memcpy(d.rho_out, rho_candidate.data(), static_cast<std::size_t>(field_bytes));
        std::memcpy(d.p_out, p_candidate.data(), static_cast<std::size_t>(field_bytes));
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_acoustic_momentum_cartesian_agrid(
        const AcousticMomentumCartesianAgridDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        uint32_t total_points = 0;
        VkDeviceSize field_bytes = 0;
        if (!supports_acoustic_momentum_cartesian_agrid_dispatch() ||
            d.nx < 4 || d.ny < 4 || d.nz < 4 || d.base_len < d.nz ||
            !checked_field_dispatch(d.nx, d.ny, d.nz, total_points, field_bytes,
                                    "A-grid momentum") ||
            d.rho == nullptr || d.p == nullptr || d.p0_base_data == nullptr ||
            d.u_in == nullptr || d.v_in == nullptr || d.w_in == nullptr ||
            d.u_out == nullptr || d.v_out == nullptr || d.w_out == nullptr ||
            d.u_out == d.v_out || d.u_out == d.w_out || d.v_out == d.w_out ||
            !std::isfinite(d.dx) || d.dx <= 0.0f ||
            !std::isfinite(d.dy) || d.dy <= 0.0f ||
            !std::isfinite(d.dz) || d.dz <= 0.0f ||
            !std::isfinite(d.dt_small) || d.dt_small <= 0.0f ||
            !std::isfinite(d.wind_clamp_h) || d.wind_clamp_h <= 0.0f ||
            !std::isfinite(d.wind_clamp_v) || d.wind_clamp_v <= 0.0f)
        {
            return false;
        }
        for (uint32_t n = 0; n < total_points; ++n)
        {
            if (!std::isfinite(d.rho[n]) || d.rho[n] <= 0.0f ||
                !std::isfinite(d.p[n]) || !std::isfinite(d.u_in[n]) ||
                !std::isfinite(d.v_in[n]) || !std::isfinite(d.w_in[n]))
            {
                return false;
            }
        }
        for (int k = 0; k < d.nz; ++k)
        {
            if (!std::isfinite(d.p0_base_data[k])) return false;
        }

        AcousticMomentumCartesianAgridPushConstants pc{};
        pc.nx = d.nx;
        pc.ny = d.ny;
        pc.nz = d.nz;
        pc.dx = d.dx;
        pc.dy = d.dy;
        pc.dz = d.dz;
        pc.dt_small = d.dt_small;
        pc.wind_clamp_h = d.wind_clamp_h;
        pc.wind_clamp_v = d.wind_clamp_v;
        const float* inputs[6] = {d.rho, d.p, d.p0_base_data, d.u_in, d.v_in, d.w_in};
        const VkDeviceSize input_bytes[6] = {
            0, 0, static_cast<VkDeviceSize>(d.nz) * sizeof(float), 0, 0, 0};
        std::vector<float> u_candidate(total_points);
        std::vector<float> v_candidate(total_points);
        std::vector<float> w_candidate(total_points);
        float* outputs[3] = {u_candidate.data(), v_candidate.data(), w_candidate.data()};
        const bool write_only[3] = {true, true, true};
        if (!dispatch_multi_field_kernel(
                pipeline(KernelId::AcousticMomentumCartesianAgrid),
                KernelId::AcousticMomentumCartesianAgrid, &pc,
                inputs, 6, outputs, 3, d.nx, d.ny, d.nz, total_points,
                input_bytes, nullptr, write_only))
        {
            return false;
        }
        for (int field = 0; field < 3; ++field)
        {
            for (uint32_t n = 0; n < total_points; ++n)
            {
                if (!std::isfinite(outputs[field][n])) return false;
            }
        }
        if (d.completed_observer != nullptr)
        {
            d.completed_observer(d.completed_observer_context,
                                 u_candidate.data(), v_candidate.data(), w_candidate.data());
        }
        std::memcpy(d.u_out, u_candidate.data(), static_cast<std::size_t>(field_bytes));
        std::memcpy(d.v_out, v_candidate.data(), static_cast<std::size_t>(field_bytes));
        std::memcpy(d.w_out, w_candidate.data(), static_cast<std::size_t>(field_bytes));
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_acoustic_null_mode_constraint_cartesian_agrid(
        const AcousticNullModeConstraintCartesianAgridDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        uint32_t total_points = 0;
        VkDeviceSize field_bytes = 0;
        const float* scalars[8] = {d.rho, d.qv, d.qc, d.qr, d.qi, d.qs, d.qg, d.qh};
        const float* velocities[3] = {d.u_in, d.v_in, d.w_in};
        float* destinations[3] = {d.u_out, d.v_out, d.w_out};
        if (!supports_acoustic_null_mode_constraint_cartesian_agrid_dispatch() ||
            d.nx < 4 || d.ny < 4 || d.nz < 4 ||
            !checked_field_dispatch(d.nx, d.ny, d.nz, total_points, field_bytes,
                                    "A-grid null-mode constraint"))
        {
            return false;
        }
        for (const float* field : scalars) if (field == nullptr) return false;
        for (const float* field : velocities) if (field == nullptr) return false;
        for (float* field : destinations) if (field == nullptr) return false;
        if (destinations[0] == destinations[1] || destinations[0] == destinations[2] ||
            destinations[1] == destinations[2]) return false;
        for (uint32_t n = 0; n < total_points; ++n)
        {
            if (!std::isfinite(d.rho[n]) || d.rho[n] <= 0.0f) return false;
            for (int q = 1; q < 8; ++q)
            {
                if (!std::isfinite(scalars[q][n]) || scalars[q][n] < 0.0f) return false;
            }
            for (const float* field : velocities)
            {
                if (!std::isfinite(field[n])) return false;
            }
        }

        std::vector<float> candidates[3] = {
            std::vector<float>(d.u_in, d.u_in + total_points),
            std::vector<float>(d.v_in, d.v_in + total_points),
            std::vector<float>(d.w_in, d.w_in + total_points)};
        float* outputs[3] = {
            candidates[0].data(), candidates[1].data(), candidates[2].data()};
        for (int pass = 0; pass < 3; ++pass)
        {
            AcousticLinePassCartesianAgridPushConstants pc{d.nx, d.ny, d.nz, pass};
            const uint32_t line_count = pass == 0
                ? static_cast<uint32_t>((d.ny - 2) * (d.nz - 2))
                : pass == 1
                    ? static_cast<uint32_t>((d.nx - 2) * (d.nz - 2))
                    : static_cast<uint32_t>((d.nx - 2) * (d.ny - 2));
            if (!dispatch_multi_field_kernel(
                    pipeline(KernelId::AcousticNullModeConstraintCartesianAgrid),
                    KernelId::AcousticNullModeConstraintCartesianAgrid, &pc,
                    scalars, 8, outputs, 3, d.nx, d.ny, d.nz, line_count))
            {
                return false;
            }
        }
        for (int field = 0; field < 3; ++field)
        {
            for (uint32_t n = 0; n < total_points; ++n)
            {
                if (!std::isfinite(outputs[field][n])) return false;
            }
        }
        if (d.completed_observer != nullptr)
        {
            d.completed_observer(d.completed_observer_context,
                                 outputs[0], outputs[1], outputs[2]);
        }
        for (int field = 0; field < 3; ++field)
        {
            std::memcpy(destinations[field], outputs[field], static_cast<std::size_t>(field_bytes));
        }
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_acoustic_boundary_cartesian_agrid(
        const AcousticBoundaryCartesianAgridDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        uint32_t total_points = 0;
        VkDeviceSize field_bytes = 0;
        const float* profiles[4] = {
            d.p0_base_data, d.rho0_base_data, d.u0_base_data, d.v0_base_data};
        float* state[5] = {d.u, d.v, d.w, d.rho, d.p};
        if (!supports_acoustic_boundary_cartesian_agrid_dispatch() ||
            d.nx < 4 || d.ny < 4 || d.nz < 4 || d.base_len < d.nz ||
            !checked_field_dispatch(d.nx, d.ny, d.nz, total_points, field_bytes,
                                    "A-grid rigid boundary"))
        {
            return false;
        }
        for (const float* field : profiles) if (field == nullptr) return false;
        for (float* field : state) if (field == nullptr) return false;
        for (int a = 0; a < 5; ++a)
        {
            for (int b = 0; b < a; ++b) if (state[a] == state[b]) return false;
        }
        for (int k = 0; k < d.nz; ++k)
        {
            for (const float* profile : profiles)
            {
                if (!std::isfinite(profile[k])) return false;
            }
        }
        for (uint32_t n = 0; n < total_points; ++n)
        {
            for (float* field : state) if (!std::isfinite(field[n])) return false;
            if (d.rho[n] <= 0.0f || d.p[n] <= 0.0f) return false;
        }

        std::vector<float> candidates[5];
        float* outputs[5]{};
        for (int field = 0; field < 5; ++field)
        {
            candidates[field].assign(state[field], state[field] + total_points);
            outputs[field] = candidates[field].data();
        }
        const VkDeviceSize profile_bytes = static_cast<VkDeviceSize>(d.nz) * sizeof(float);
        const VkDeviceSize input_bytes[4] = {
            profile_bytes, profile_bytes, profile_bytes, profile_bytes};
        for (int pass = 0; pass < 3; ++pass)
        {
            AcousticLinePassCartesianAgridPushConstants pc{d.nx, d.ny, d.nz, pass};
            const uint32_t face_points = pass == 0
                ? static_cast<uint32_t>(d.ny * d.nz)
                : pass == 1
                    ? static_cast<uint32_t>(d.nx * d.nz)
                    : static_cast<uint32_t>(d.nx * d.ny);
            if (!dispatch_multi_field_kernel(
                    pipeline(KernelId::AcousticBoundaryCartesianAgrid),
                    KernelId::AcousticBoundaryCartesianAgrid, &pc,
                    profiles, 4, outputs, 5, d.nx, d.ny, d.nz, face_points,
                    input_bytes))
            {
                return false;
            }
        }
        for (int field = 0; field < 5; ++field)
        {
            for (uint32_t n = 0; n < total_points; ++n)
            {
                if (!std::isfinite(outputs[field][n])) return false;
            }
        }
        for (uint32_t n = 0; n < total_points; ++n)
        {
            if (outputs[3][n] <= 0.0f || outputs[4][n] <= 0.0f) return false;
        }
        if (d.completed_observer != nullptr)
        {
            d.completed_observer(d.completed_observer_context,
                                 outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]);
        }
        for (int field = 0; field < 5; ++field)
        {
            std::memcpy(state[field], outputs[field], static_cast<std::size_t>(field_bytes));
        }
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool supports_acoustic_pressure_cartesian_cgrid_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Cartesian-only shader (GPU.md section 2.0 routing rule).
        if (coord_system_ != CoordinateSystem::Cartesian)
            return false;
        return pipeline(KernelId::AcousticPressureCartesianCgrid).is_ready();
#else
        return false;
#endif
    }

    bool supports_acoustic_momentum_cartesian_cgrid_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Cartesian-only shader (GPU.md section 2.0 routing rule).
        if (coord_system_ != CoordinateSystem::Cartesian)
            return false;
        return pipeline(KernelId::AcousticMomentumCartesianCgrid).is_ready();
#else
        return false;
#endif
    }

    bool supports_acoustic_substep_fused_cartesian_cgrid_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Cartesian-only shaders (GPU.md section 2.0 routing rule).
        if (coord_system_ != CoordinateSystem::Cartesian)
            return false;
        return pipeline(KernelId::AcousticPressureCartesianCgrid).is_ready() &&
               pipeline(KernelId::AcousticMomentumCartesianCgrid).is_ready();
#else
        return false;
#endif
    }

    bool dispatch_acoustic_pressure_cartesian_cgrid(const AcousticPressureCartesianCgridDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (acoustic_velocity_reuse_[residency_scope_index(active_residency_scope_)])
            for (const char* name : {"acoustic_u", "acoustic_v", "acoustic_w"})
                active_resident_fields().release(name);
        if (acoustic_pressure_output_reuse_[residency_scope_index(active_residency_scope_)])
            for (const char* name : {"acoustic_rho_pressure", "acoustic_p_pressure"})
                active_resident_fields().release(name);
        if (!valid_acoustic_cgrid_dispatch(dispatch))
            return false;
        const auto* u_data              = dispatch.u;
        const auto* v_data              = dispatch.v;
        const auto* w_data              = dispatch.w;
        const auto* rho_in              = dispatch.rho_in;
        const auto* p_in                = dispatch.p_in;
        const auto* rho0_base_data      = dispatch.rho0_base_data;
        const auto* p0_base_data        = dispatch.p0_base_data;
        const int   base_len            = dispatch.base_len;
        const auto* theta_data          = dispatch.theta_data;
        auto*       rho_out             = dispatch.rho_out;
        auto*       p_out               = dispatch.p_out;
        const int   nx                  = dispatch.nx;
        const int   ny                  = dispatch.ny;
        const int   nz                  = dispatch.nz;
        const float dx_val              = dispatch.dx;
        const float dy_val              = dispatch.dy;
        const float dz_val              = dispatch.dz;
        const float gamma_val           = dispatch.gamma_val;
        const float dt_small            = dispatch.dt_small;
        const float rho_min             = dispatch.rho_min;
        const float rho_max             = dispatch.rho_max;
        const float p_min               = dispatch.p_min;
        const float p_max               = dispatch.p_max;
        const float p_ref_fallback      = dispatch.p_ref_fallback;
        const float frame_velocity_u_ms = dispatch.frame_velocity_u_ms;
        const float frame_velocity_v_ms = dispatch.frame_velocity_v_ms;

        // Cartesian-only shader.
        if (coord_system_ != CoordinateSystem::Cartesian)
        {
            log_vulkan_warning("acoustic pressure (cartesian c-grid) dispatch "
                               "refused: shader is cartesian-only and the "
                               "active coordinate system is not cartesian");
            return false;
        }
        if (!pipeline(KernelId::AcousticPressureCartesianCgrid).is_ready())
        {
            return false;
        }
        if (rho0_base_data == nullptr || p0_base_data == nullptr || base_len < nz)
        {
            return false;
        }

        AcousticPressureCartesianCgridPushConstants pc{};
        pc.nx                  = nx;
        pc.ny                  = ny;
        pc.nz                  = nz;
        pc.dx                  = dx_val;
        pc.dy                  = dy_val;
        pc.dz_val              = dz_val;
        pc.gamma_val           = gamma_val;
        pc.gamma_low           = static_cast<float>(dispatch.gamma_val - static_cast<double>(gamma_val));
        pc.dt_small            = dt_small;
        pc.rho_min             = rho_min;
        pc.rho_max             = rho_max;
        pc.p_min               = p_min;
        pc.p_max               = p_max;
        pc.p_ref_fallback      = p_ref_fallback;
        pc.preserve_unrepaired = dispatch.repair_primary_state ? 0 : 1;
        pc.frame_velocity_u_ms = frame_velocity_u_ms;
        pc.frame_velocity_v_ms = frame_velocity_v_ms;

        // The two base profiles upload compact (nz floats; the shader
        // indexes them by k) - no per-(i, j) broadcast (Phase 4.3).
        const VkDeviceSize prof_bytes          = static_cast<VkDeviceSize>(nz) * sizeof(float);
        uint32_t           total_points        = 0;
        VkDeviceSize       field_bytes_checked = 0;
        if (!checked_field_dispatch(nx, ny, nz, total_points, field_bytes_checked, "acoustic pressure c-grid"))
            return false;
        const float*       inputs[8] = {u_data, v_data, w_data, rho_in, p_in, rho0_base_data, p0_base_data, theta_data};
        const VkDeviceSize input_bytes[8]       = {0, 0, 0, 0, 0, prof_bytes, prof_bytes, 0};
        float*             outputs[2]           = {rho_out, p_out};
        const bool         output_write_only[2] = {true, true};

        const bool loop_active = acoustic_loop_active_[residency_scope_index(active_residency_scope_)];
        const bool retain_wind = acoustic_velocity_reuse_[residency_scope_index(active_residency_scope_)];
        const char* input_names[8] = {
            retain_wind ? "acoustic_u" : nullptr,
            retain_wind ? "acoustic_v" : nullptr,
            retain_wind ? "acoustic_w" : nullptr,
            nullptr, nullptr, "acoustic_rho0", "acoustic_p0", "acoustic_theta"};
        const bool retain_pressure = acoustic_pressure_output_reuse_[residency_scope_index(active_residency_scope_)];
        const char* output_names[2] = {"acoustic_rho_pressure", "acoustic_p_pressure"};
        return dispatch_multi_field_kernel(pipeline(KernelId::AcousticPressureCartesianCgrid),
                                           KernelId::AcousticPressureCartesianCgrid,
                                           &pc,
                                           inputs,
                                           8,
                                           outputs,
                                           2,
                                           nx,
                                           ny,
                                           nz,
                                           total_points,
                                           input_bytes,
                                           loop_active ? input_names : nullptr,
                                           output_write_only,
                                           true, nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr,
                                           retain_pressure ? output_names : nullptr);
#else
        (void)dispatch;
        return false;
#endif
    }

    bool dispatch_acoustic_momentum_cartesian_cgrid(const AcousticMomentumCartesianCgridDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        struct ReleaseSubstepInputs
        {
            nimbus::compute::ResidentFieldStore* fields;
            ~ReleaseSubstepInputs()
            {
                if (fields)
                    for (const char* name : {"acoustic_u", "acoustic_v", "acoustic_w",
                                              "acoustic_rho_pressure", "acoustic_p_pressure"})
                        fields->release(name);
            }
        } release_substep{(acoustic_velocity_reuse_[residency_scope_index(active_residency_scope_)] ||
                            acoustic_pressure_output_reuse_[residency_scope_index(active_residency_scope_)])
                               ? &active_resident_fields() : nullptr};
        if (!valid_acoustic_cgrid_dispatch(dispatch))
            return false;
        const auto* rho_data         = dispatch.rho_data;
        const auto* p_data           = dispatch.p_data;
        const auto* dp0_dz_data      = dispatch.dp0_dz_data;
        const int   base_len         = dispatch.base_len;
        const auto* u_in             = dispatch.u_in;
        const auto* v_in             = dispatch.v_in;
        const auto* w_in             = dispatch.w_in;
        auto*       u_out            = dispatch.u_out;
        auto*       v_out            = dispatch.v_out;
        auto*       w_out            = dispatch.w_out;
        const int   nx               = dispatch.nx;
        const int   ny               = dispatch.ny;
        const int   nz               = dispatch.nz;
        const float dx_val           = dispatch.dx;
        const float dy_val           = dispatch.dy;
        const float dz_val           = dispatch.dz;
        const float dt_small         = dispatch.dt_small;
        const float wind_clamp_h_val = dispatch.wind_clamp_h;
        const float wind_clamp_v_val = dispatch.wind_clamp_v;

        // Cartesian-only shader; see dispatch_acoustic_pressure_cartesian_cgrid.
        if (coord_system_ != CoordinateSystem::Cartesian)
        {
            log_vulkan_warning("acoustic momentum (cartesian c-grid) dispatch "
                               "refused: shader is cartesian-only and the "
                               "active coordinate system is not cartesian");
            return false;
        }
        if (!pipeline(KernelId::AcousticMomentumCartesianCgrid).is_ready())
        {
            return false;
        }
        if (dp0_dz_data == nullptr || base_len < nz)
        {
            return false;
        }

        AcousticMomentumCartesianCgridPushConstants pc{};
        pc.nx           = nx;
        pc.ny           = ny;
        pc.nz           = nz;
        pc.dx           = dx_val;
        pc.dy           = dy_val;
        pc.dz_val       = dz_val;
        pc.dt_small     = dt_small;
        pc.wind_clamp_h = wind_clamp_h_val;
        pc.wind_clamp_v = wind_clamp_v_val;
        pc.preserve_unrepaired = dispatch.repair_primary_state ? 0 : 1;
        pc.paired_base_gradient = dispatch.dp0_dz_low_data ? 1 : 0;

        // dp0_dz uploads compact (nz floats; shader indexes by k).
        std::vector<float> paired_gradient;
        const float* gradient = pack_acoustic_base_gradient(dispatch, paired_gradient);
        const VkDeviceSize prof_bytes = static_cast<VkDeviceSize>(nz) * sizeof(float) * (dispatch.dp0_dz_low_data ? 2 : 1);
        uint32_t           total_points        = 0;
        VkDeviceSize       field_bytes_checked = 0;
        if (!checked_field_dispatch(nx, ny, nz, total_points, field_bytes_checked, "acoustic momentum c-grid"))
            return false;
        const float*       inputs[6]            = {rho_data, p_data, gradient, u_in, v_in, w_in};
        const VkDeviceSize input_bytes[6]       = {0, 0, prof_bytes, 0, 0, 0};
        float*             outputs[3]           = {u_out, v_out, w_out};
        const bool         output_write_only[3] = {true, true, true};

        const bool loop_active = acoustic_loop_active_[residency_scope_index(active_residency_scope_)];
        const bool retain_wind = acoustic_velocity_reuse_[residency_scope_index(active_residency_scope_)];
        const char* input_names[6] = {
            acoustic_pressure_output_reuse_[residency_scope_index(active_residency_scope_)] ? "acoustic_rho_pressure" : nullptr,
            acoustic_pressure_output_reuse_[residency_scope_index(active_residency_scope_)] ? "acoustic_p_pressure" : nullptr,
            "acoustic_dp0_dz",
            retain_wind ? "acoustic_u" : nullptr,
            retain_wind ? "acoustic_v" : nullptr,
            retain_wind ? "acoustic_w" : nullptr};
        const bool completed = dispatch_multi_field_kernel(pipeline(KernelId::AcousticMomentumCartesianCgrid),
                                           KernelId::AcousticMomentumCartesianCgrid,
                                           &pc,
                                           inputs,
                                           6,
                                           outputs,
                                           3,
                                           nx,
                                           ny,
                                           nz,
                                           total_points,
                                           input_bytes,
                                           loop_active ? input_names : nullptr,
                                           output_write_only);
        return completed;
#else
        (void)dispatch;
        return false;
#endif
    }

    bool supports_unrepaired_acoustic_fused() const override
    {
        return supports_acoustic_substep_fused_cartesian_cgrid_dispatch();
    }

    bool
    dispatch_acoustic_substep_fused_cartesian_cgrid(const AcousticSubstepFusedCartesianCgridDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if ((!dispatch.repair_primary_state || dispatch.dp0_dz_low_data != nullptr) &&
            !valid_acoustic_cgrid_dispatch(dispatch))
            return false;
        // Orchestration owns native energy attribution and its rollback using
        // exact pre-commit fields and deterministic FP64 reductions.
        if (dispatch.energy_report_host != nullptr)
        {
            log_vulkan_warning("fused acoustic C-grid dispatch requires orchestration-owned energy attribution");
            return false;
        }
        auto*       u                   = dispatch.u;
        auto*       v                   = dispatch.v;
        auto*       w                   = dispatch.w;
        auto*       rho                 = dispatch.rho;
        auto*       p                   = dispatch.p;
        const auto* theta_data          = dispatch.theta_data;
        const auto* rho0_base_data      = dispatch.rho0_base_data;
        const auto* p0_base_data        = dispatch.p0_base_data;
        AcousticMomentumCartesianCgridDispatch gradient_dispatch;
        gradient_dispatch.dp0_dz_data = dispatch.dp0_dz_data;
        gradient_dispatch.dp0_dz_low_data = dispatch.dp0_dz_low_data;
        gradient_dispatch.nz = dispatch.nz;
        std::vector<float> paired_gradient;
        const auto* dp0_dz_data = dispatch.dp0_dz_low_data != nullptr
            ? pack_acoustic_base_gradient(gradient_dispatch, paired_gradient) : dispatch.dp0_dz_data;
        const int   base_len            = dispatch.base_len;
        const int   nx                  = dispatch.nx;
        const int   ny                  = dispatch.ny;
        const int   nz                  = dispatch.nz;
        const float dx_val              = dispatch.dx;
        const float dy_val              = dispatch.dy;
        const float dz_val              = dispatch.dz;
        const float gamma_val           = dispatch.gamma_val;
        const float dt_small            = dispatch.dt_small;
        const float rho_min             = dispatch.rho_min;
        const float rho_max             = dispatch.rho_max;
        const float p_min               = dispatch.p_min;
        const float p_max               = dispatch.p_max;
        const float p_ref_fallback      = dispatch.p_ref_fallback;
        const float wind_clamp_h_val    = dispatch.wind_clamp_h;
        const float wind_clamp_v_val    = dispatch.wind_clamp_v;
        const float frame_velocity_u_ms = dispatch.frame_velocity_u_ms;
        const float frame_velocity_v_ms = dispatch.frame_velocity_v_ms;

        // Cartesian-only shaders; see dispatch_acoustic_pressure_cartesian_cgrid.
        if (coord_system_ != CoordinateSystem::Cartesian)
        {
            log_vulkan_warning("fused acoustic substep (cartesian c-grid) "
                               "dispatch refused: shaders are cartesian-only "
                               "and the active coordinate system is not "
                               "cartesian");
            return false;
        }
        if (!supports_acoustic_substep_fused_cartesian_cgrid_dispatch())
            return false;
        if (rho0_base_data == nullptr || p0_base_data == nullptr || dp0_dz_data == nullptr || base_len < nz)
        {
            return false;
        }

        uint32_t     total_points = 0;
        VkDeviceSize field_bytes  = 0;
        if (!checked_field_dispatch(nx, ny, nz, total_points, field_bytes, "fused acoustic c-grid substep"))
            return false;
        const uint32_t workgroup_count = 1u + (total_points - 1u) / 64u;

        // 11 slots: u(0), v(1), w(2), rho(3), p(4), rho0(5), p0(6),
        // dp0_dz(7), rho_new(8), p_new(9), theta(10). The pressure pass writes its
        // outputs to the SEPARATE slots 8/9 because every thread reads
        // neighbor rho values (an in-place rho write would race across
        // threads); the momentum pass then reads rho/p from 8/9 and updates
        // u/v/w in place (safe: it reads velocities only at the thread's
        // own index). Theta and the three base-profile rows are immutable
        // throughout one acoustic loop, so retain them in named slots across
        // attributed per-substep submissions. A residency generation begins
        // at every large model step, which forces a fresh first upload and
        // prevents pointer-stable profile scratch from becoming stale.
        constexpr int kSlotCount = 11;
        int           slots[kSlotCount];
        for (int& slot : slots)
            slot = -1;
        const VkDeviceSize       prof_bytes         = static_cast<VkDeviceSize>(nz) * sizeof(float);
        static const char* const kInvariantNames[4] = {
            "acoustic_rho0", "acoustic_p0", "acoustic_dp0_dz", "acoustic_theta"};
        const VkDeviceSize gradient_bytes = prof_bytes * (dispatch.dp0_dz_low_data ? 2 : 1);
        const VkDeviceSize invariant_sizes[4] = {prof_bytes, prof_bytes, gradient_bytes, field_bytes};
        int                invariant_slots[4];
        if (!named_slots_acquire_many(kInvariantNames, invariant_sizes, 4, invariant_slots))
        {
            log_vulkan_warning("failed to pin invariant slots for fused acoustic substep "
                               "(cartesian c-grid)");
            return false;
        }
        slots[5]  = invariant_slots[0];
        slots[6]  = invariant_slots[1];
        slots[7]  = invariant_slots[2];
        slots[10] = invariant_slots[3];

        constexpr int kTransientCount                    = 7;
        const int     transient_logical[kTransientCount] = {0, 1, 2, 3, 4, 8, 9};
        VkDeviceSize  transient_sizes[kTransientCount];
        for (VkDeviceSize& size : transient_sizes)
            size = field_bytes;
        int transient_slots[kTransientCount];
        if (!buffer_pool_.acquire_many(transient_sizes, kTransientCount, transient_slots))
        {
            log_vulkan_warning("failed to acquire transient slots for fused acoustic "
                               "substep (cartesian c-grid)");
            return false;
        }
        for (int s = 0; s < kTransientCount; ++s)
            slots[transient_logical[s]] = transient_slots[s];
        const auto release_transient_slots = [&]() { release_pool_slots(transient_slots, kTransientCount); };

        const int slot_theta   = slots[10];
        const int slot_u       = slots[0];
        const int slot_v       = slots[1];
        const int slot_w       = slots[2];
        const int slot_rho     = slots[3];
        const int slot_p       = slots[4];
        const int slot_rho0    = slots[5];
        const int slot_p0      = slots[6];
        const int slot_dp0     = slots[7];
        const int slot_rho_new = slots[8];
        const int slot_p_new   = slots[9];

        const bool unified  = device_context_.has_unified_memory;
        auto       host_ptr = [&](int slot) -> void*
        { return unified ? buffer_pool_.device(slot).mapped : buffer_pool_.staging(slot).mapped; };

        const float* invariant_inputs[4]   = {rho0_base_data, p0_base_data, dp0_dz_data, theta_data};
        const bool   invariant_reusable[4] = {
            named_slot_reusable(kInvariantNames[0], invariant_inputs[0], prof_bytes),
            named_slot_reusable(kInvariantNames[1], invariant_inputs[1], prof_bytes),
            named_slot_reusable(kInvariantNames[2], invariant_inputs[2], gradient_bytes),
            named_slot_reusable(kInvariantNames[3], invariant_inputs[3], field_bytes),
        };
        for (int s = 0; s < 4; ++s)
        {
            if (!invariant_reusable[s])
            {
                named_slot_commit(kInvariantNames[s], invariant_inputs[s], invariant_sizes[s]);
                named_slot_invalidate(kInvariantNames[s]);
            }
        }

        // Upload the five mutable state inputs every observable substep.
        // Upload invariant theta/profile inputs only on the first substep of
        // the model-step residency generation. Slots 8/9 are pure outputs.
        tracked_memcpy(host_ptr(slot_u), u, field_bytes);
        tracked_memcpy(host_ptr(slot_v), v, field_bytes);
        tracked_memcpy(host_ptr(slot_w), w, field_bytes);
        tracked_memcpy(host_ptr(slot_rho), rho, field_bytes);
        tracked_memcpy(host_ptr(slot_p), p, field_bytes);
        for (int s = 0; s < 4; ++s)
        {
            if (!invariant_reusable[s])
            {
                tracked_memcpy(host_ptr(invariant_slots[s]), invariant_inputs[s], invariant_sizes[s]);
            }
        }

        // Descriptor sets for both passes (updated before submission).
        int pressure_bindings[10] = {
            slot_u, slot_v, slot_w, slot_rho, slot_p, slot_rho0, slot_p0, slot_theta, slot_rho_new, slot_p_new};
        const VkDeviceSize pressure_ranges[10] = {field_bytes,
                                                  field_bytes,
                                                  field_bytes,
                                                  field_bytes,
                                                  field_bytes,
                                                  prof_bytes,
                                                  prof_bytes,
                                                  field_bytes,
                                                  field_bytes,
                                                  field_bytes};
        update_pooled_descriptor_set(
            pipeline(KernelId::AcousticPressureCartesianCgrid), pressure_bindings, field_bytes, 10, pressure_ranges);

        int momentum_bindings[9] = {slot_rho_new, slot_p_new, slot_dp0, slot_u, slot_v, slot_w, slot_u, slot_v, slot_w};
        const VkDeviceSize momentum_ranges[9] = {field_bytes,
                                                 field_bytes,
                                                 gradient_bytes,
                                                 field_bytes,
                                                 field_bytes,
                                                 field_bytes,
                                                 field_bytes,
                                                 field_bytes,
                                                 field_bytes};
        update_pooled_descriptor_set(
            pipeline(KernelId::AcousticMomentumCartesianCgrid), momentum_bindings, field_bytes, 9, momentum_ranges);

        vkResetCommandBuffer_(cmd_buf_, 0);
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (begin_profiled_command_buffer(begin_info) != VK_SUCCESS)
        {
            release_transient_slots();
            return false;
        }

        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;

        // ── H2D transfer ──
        if (unified)
        {
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }
        else
        {
            const int mutable_upload_slots[5] = {slot_u, slot_v, slot_w, slot_rho, slot_p};
            for (int s = 0; s < 5; ++s)
            {
                VkBufferCopy copy_region{};
                copy_region.size = field_bytes;
                tracked_cmd_copy_buffer(buffer_pool_.staging(mutable_upload_slots[s]).buffer,
                                        buffer_pool_.device(mutable_upload_slots[s]).buffer,
                                        1,
                                        &copy_region);
            }
            for (int s = 0; s < 4; ++s)
            {
                if (invariant_reusable[s])
                    continue;
                VkBufferCopy copy_region{};
                copy_region.size = invariant_sizes[s];
                tracked_cmd_copy_buffer(buffer_pool_.staging(invariant_slots[s]).buffer,
                                        buffer_pool_.device(invariant_slots[s]).buffer,
                                        1,
                                        &copy_region);
            }
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }

        // ── Dispatch 1: pressure (reads u,v,w,rho,p,rho0,p0; writes rho_new,p_new) ──
        {
            AcousticPressureCartesianCgridPushConstants pc{};
            pc.nx                  = nx;
            pc.ny                  = ny;
            pc.nz                  = nz;
            pc.dx                  = dx_val;
            pc.dy                  = dy_val;
            pc.dz_val              = dz_val;
            pc.gamma_val           = gamma_val;
            pc.dt_small            = dt_small;
            pc.rho_min             = rho_min;
            pc.rho_max             = rho_max;
            pc.p_min               = p_min;
            pc.p_max               = p_max;
            pc.p_ref_fallback      = p_ref_fallback;
            pc.gamma_low           = dispatch.repair_primary_state ? 0.0f : dispatch.gamma_low;
            pc.preserve_unrepaired = dispatch.repair_primary_state ? 0 : 1;
            pc.frame_velocity_u_ms = frame_velocity_u_ms;
            pc.frame_velocity_v_ms = frame_velocity_v_ms;

            vkCmdBindPipeline_(
                cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline(KernelId::AcousticPressureCartesianCgrid).pipeline);
            vkCmdBindDescriptorSets_(cmd_buf_,
                                     VK_PIPELINE_BIND_POINT_COMPUTE,
                                     pipeline(KernelId::AcousticPressureCartesianCgrid).pipeline_layout,
                                     0,
                                     1,
                                     &pipeline(KernelId::AcousticPressureCartesianCgrid).descriptor_set,
                                     0,
                                     nullptr);
            vkCmdPushConstants_(cmd_buf_,
                                pipeline(KernelId::AcousticPressureCartesianCgrid).pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT,
                                0,
                                sizeof(pc),
                                &pc);
            record_linear_group_dispatch(workgroup_count, KernelId::AcousticPressureCartesianCgrid);
        }

        // ── Barrier: pressure writes complete before momentum reads ──
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier_(cmd_buf_,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0,
                              1,
                              &barrier,
                              0,
                              nullptr,
                              0,
                              nullptr);

        // ── Dispatch 2: momentum (reads rho_new,p_new,p0,u,v,w; writes u,v,w) ──
        {
            AcousticMomentumCartesianCgridPushConstants pc{};
            pc.nx           = nx;
            pc.ny           = ny;
            pc.nz           = nz;
            pc.dx           = dx_val;
            pc.dy           = dy_val;
            pc.dz_val       = dz_val;
            pc.dt_small     = dt_small;
            pc.wind_clamp_h = wind_clamp_h_val;
            pc.wind_clamp_v = wind_clamp_v_val;
            pc.preserve_unrepaired = dispatch.repair_primary_state ? 0 : 1;
            pc.paired_base_gradient = dispatch.dp0_dz_low_data ? 1 : 0;

            vkCmdBindPipeline_(
                cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline(KernelId::AcousticMomentumCartesianCgrid).pipeline);
            vkCmdBindDescriptorSets_(cmd_buf_,
                                     VK_PIPELINE_BIND_POINT_COMPUTE,
                                     pipeline(KernelId::AcousticMomentumCartesianCgrid).pipeline_layout,
                                     0,
                                     1,
                                     &pipeline(KernelId::AcousticMomentumCartesianCgrid).descriptor_set,
                                     0,
                                     nullptr);
            vkCmdPushConstants_(cmd_buf_,
                                pipeline(KernelId::AcousticMomentumCartesianCgrid).pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT,
                                0,
                                sizeof(pc),
                                &pc);
            record_linear_group_dispatch(workgroup_count, KernelId::AcousticMomentumCartesianCgrid);
        }

        // ── D2H transfer (u, v, w in place; rho/p from the output slots) ──
        if (unified)
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }
        else
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);

            VkBufferCopy copy_region{};
            copy_region.size             = field_bytes;
            const int writeback_slots[5] = {slot_u, slot_v, slot_w, slot_rho_new, slot_p_new};
            for (int s = 0; s < 5; ++s)
            {
                tracked_cmd_copy_buffer(buffer_pool_.device(writeback_slots[s]).buffer,
                                        buffer_pool_.staging(writeback_slots[s]).buffer,
                                        1,
                                        &copy_region);
            }
        }

        if (end_profiled_command_buffer() != VK_SUCCESS)
        {
            release_transient_slots();
            return false;
        }

        VkSubmitInfo submit_info{};
        submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers    = &cmd_buf_;
        if (reset_fence_and_submit("acoustic_fused_cartesian_cgrid", submit_info) != VK_SUCCESS)
        {
            release_transient_slots();
            return false;
        }

        if (timed_wait_fence("acoustic_fused_cartesian_cgrid") != VK_SUCCESS)
        {
            release_transient_slots();
            return false;
        }

        // Copy back the 5 modified fields (rho/p from the output slots).
        if (!dispatch.repair_primary_state && !valid_unrepaired_acoustic_state({
                static_cast<const float*>(host_ptr(slot_u)), static_cast<const float*>(host_ptr(slot_v)),
                static_cast<const float*>(host_ptr(slot_w)), static_cast<const float*>(host_ptr(slot_rho_new)),
                static_cast<const float*>(host_ptr(slot_p_new))}, total_points))
        {
            release_transient_slots();
            return false;
        }
        tracked_memcpy(u, host_ptr(slot_u), field_bytes);
        tracked_memcpy(v, host_ptr(slot_v), field_bytes);
        tracked_memcpy(w, host_ptr(slot_w), field_bytes);
        tracked_memcpy(rho, host_ptr(slot_rho_new), field_bytes);
        tracked_memcpy(p, host_ptr(slot_p_new), field_bytes);

        for (int s = 0; s < 4; ++s)
        {
            named_slot_commit(kInvariantNames[s], invariant_inputs[s], invariant_sizes[s]);
        }
        release_transient_slots();
        return true;
#else
        (void)dispatch;
        return false;
#endif
    }

    void begin_acoustic_loop_residency(bool retain_velocity_inputs = false,
                                       bool retain_pressure_outputs = false) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        release_acoustic_loop_residency();
        const auto scope = residency_scope_index(active_residency_scope_);
        acoustic_loop_active_[scope] = true;
        acoustic_velocity_reuse_[scope] = retain_velocity_inputs;
        acoustic_pressure_output_reuse_[scope] = retain_pressure_outputs;
#else
        (void)retain_velocity_inputs;
        (void)retain_pressure_outputs;
#endif
    }

    void release_acoustic_loop_residency() override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        static const char* const kInvariantNames[9] = {
            "acoustic_rho0", "acoustic_p0", "acoustic_dp0_dz", "acoustic_theta",
            "acoustic_u", "acoustic_v", "acoustic_w", "acoustic_rho_pressure", "acoustic_p_pressure"};
        for (const char* name : kInvariantNames)
            active_resident_fields().release(name);
        const auto scope = residency_scope_index(active_residency_scope_);
        acoustic_loop_active_[scope] = false;
        acoustic_velocity_reuse_[scope] = false;
        acoustic_pressure_output_reuse_[scope] = false;
#endif
    }

    // ── Conservative scalar-transport batch (GPU audit Phase 3.2) ──

    bool supports_advect_scalars_conservative_cartesian_cgrid_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Cartesian-only shaders (GPU.md section 2.0 routing rule).
        if (coord_system_ != CoordinateSystem::Cartesian)
            return false;
        return pipeline(KernelId::AdvectConservativeMassFluxCartesianCgrid).is_ready() &&
               pipeline(KernelId::AdvectConservativeScalarFluxCartesianCgrid).is_ready() &&
               pipeline(KernelId::AdvectPdLimiterCartesianCgrid).is_ready() &&
               pipeline(KernelId::AdvectConservativeUpdateCartesianCgrid).is_ready();
#else
        return false;
#endif
    }

    bool supports_supplied_cartesian_cgrid_mass_flux() const override
    {
        return supports_advect_scalars_conservative_cartesian_cgrid_dispatch();
    }

    bool dispatch_advect_scalars_conservative_cartesian_cgrid(
        const ConservativeCartesianCgridTransportDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        const float*                                    u_data                 = dispatch.u;
        const float*                                    v_data                 = dispatch.v;
        const float*                                    w_data                 = dispatch.w;
        const float*                                    rho_data               = dispatch.rho_data;
        const float* const*                             q_in                   = dispatch.q_in;
        float* const*                                   q_out                  = dispatch.q_out;
        const float*                                    dt_kappas              = dispatch.dt_kappas;
        const int*                                      positive_definite      = dispatch.positive_definite;
        const int                                       n_scalars              = dispatch.n_scalars;
        const int                                       nx                     = dispatch.nx;
        const int                                       ny                     = dispatch.ny;
        const int                                       nz                     = dispatch.nz;
        const float                                     inv_dx                 = dispatch.inv_dx;
        const float                                     inv_dy                 = dispatch.inv_dy;
        const float                                     inv_dz                 = dispatch.inv_dz;
        const float                                     dt_step                = dispatch.dt_step;
        const float                                     rho_floor              = dispatch.rho_floor;
        const float*                                    supplied_rho_begin     = dispatch.density_begin;
        const float*                                    supplied_mfx           = dispatch.mass_flux_x;
        const float*                                    supplied_mfy           = dispatch.mass_flux_y;
        const float*                                    supplied_mfz           = dispatch.mass_flux_z;
        const bool                                      use_supplied_mass_flux = dispatch.use_supplied_mass_flux;
        const float                                     frame_velocity_u_ms    = dispatch.frame_velocity_u_ms;
        const float                                     frame_velocity_v_ms    = dispatch.frame_velocity_v_ms;
        tmv::ScalarBoundaryTransportReport* const*      boundary_reports       = dispatch.host_boundary_reports;
        const tmv::ScalarMoistEnergyKind*               energy_kinds           = dispatch.host_energy_kinds;
        tmv::ScalarMoistEnergyTransportReportV1* const* energy_reports         = dispatch.host_energy_reports;

        // Cartesian-only shaders; see dispatch_acoustic_pressure_cartesian_cgrid.
        if (coord_system_ != CoordinateSystem::Cartesian)
        {
            log_vulkan_warning("conservative scalar transport (cartesian c-grid) "
                               "dispatch refused: shaders are cartesian-only and "
                               "the active coordinate system is not cartesian");
            return false;
        }
        if (!supports_advect_scalars_conservative_cartesian_cgrid_dispatch())
        {
            return false;
        }
        constexpr int kMaxBatchScalars = 16;
        if (q_in == nullptr || q_out == nullptr || dt_kappas == nullptr || n_scalars < 1 ||
            n_scalars > kMaxBatchScalars ||
            (use_supplied_mass_flux && (supplied_rho_begin == nullptr || supplied_mfx == nullptr ||
                                        supplied_mfy == nullptr || supplied_mfz == nullptr)))
        {
            return false;
        }
        for (int s = 0; s < n_scalars; ++s)
        {
            if (dispatch.preserve_positive_support && (!positive_definite || positive_definite[s] == 0))
                return false;
            if (q_in[s] == nullptr || q_out[s] == nullptr)
            {
                return false;
            }
            if (!std::isfinite(dt_kappas[s]) || dt_kappas[s] != 0.0f)
            {
                log_vulkan_warning("conservative scalar transport rejected nonzero chained "
                                   "point diffusion; use the live-density diffusion scheme");
                return false;
            }
            tmv::ScalarMoistEnergyTransportReportV1* energy_report =
                energy_reports != nullptr ? energy_reports[s] : nullptr;
            const tmv::ScalarMoistEnergyKind energy_kind =
                energy_kinds != nullptr ? energy_kinds[s] : tmv::ScalarMoistEnergyKind::None;
            if ((energy_report == nullptr) != (energy_kind == tmv::ScalarMoistEnergyKind::None))
            {
                log_vulkan_warning("conservative scalar transport rejected a partial "
                                   "moist-energy report descriptor");
                return false;
            }
            const bool report_requested =
                (boundary_reports != nullptr && boundary_reports[s] != nullptr) || energy_report != nullptr;
            if (report_requested && !use_supplied_mass_flux)
            {
                log_vulkan_warning("conservative scalar transport reports require the "
                                   "retained split-continuity mass-flux context");
                return false;
            }
        }

        uint32_t     total_points = 0;
        VkDeviceSize field_bytes  = 0;
        if (!checked_field_dispatch(nx, ny, nz, total_points, field_bytes, "conservative scalar transport"))
            return false;
        const uint32_t workgroup_count = 1u + (total_points - 1u) / 64u;
        nimbus::compute::CartesianScalarCapture capture;
        if (!capture.prepare(dispatch) ||
            capture.bytes() > device_context_.selected_device_properties.limits.maxStorageBufferRange)
            return false;


        // Slot layout: u(0) v(1) w(2) rho(3) | mf x/y/z (4-6) | sf x/y/z
        // (7-9) | q_work(10) q_out_work(11) r_lim(12) |
        // beginning density (13) | one q_io slot per scalar (14..13+n).
        // Each pipeline's single descriptor set is bound ONCE
        // against fixed slots; scalars cycle through the fixed q_work /
        // q_out_work slots via device-to-device copies, which is what lets
        // the whole batch (mass flux once + per-scalar flux, update and
        // update passes) record into one submission with one fence.
        constexpr int kFixedSlots = 14;
        const int     slot_count  = kFixedSlots + n_scalars;
        int           slots[kFixedSlots + kMaxBatchScalars];
        // u/v/w/rho live in pinned named slots (GPU audit Phase
        // 4.3d): this dispatch is the step's store site for the winds and
        // density, and the fused Smagorinsky chain later in the step
        // reuses u/v/w without re-uploading (nothing on the host writes
        // them in between - microphysics and the bounds sweeps touch
        // theta/q/rho/p only; the SGS integrate runs after Smagorinsky).
        static const char* const kStateNames[4] = {"u", "v", "w", "rho"};
        const VkDeviceSize       state_sizes[4] = {field_bytes, field_bytes, field_bytes, field_bytes};
        if (!named_slots_acquire_many(kStateNames, state_sizes, 4, slots))
        {
            log_vulkan_warning("failed to pin named slots for conservative scalar transport (cartesian c-grid)");
            return false;
        }
        bool mass_flux_slots_named = false;
        if (use_supplied_mass_flux)
        {
            static const char* const kMassFluxNames[3] = {
                "cgrid_mass_flux_x", "cgrid_mass_flux_y", "cgrid_mass_flux_z"};
            const VkDeviceSize mass_flux_sizes[3] = {field_bytes, field_bytes, field_bytes};
            if (!named_slots_acquire_many(kMassFluxNames, mass_flux_sizes, 3, slots + 4))
            {
                log_vulkan_warning("failed to pin supplied C-grid mass-flux slots");
                return false;
            }
            mass_flux_slots_named = true;
        }

        const int transient_start = mass_flux_slots_named ? 7 : 4;
        const int transient_count = slot_count - transient_start;
        if (!acquire_pool_slots(field_bytes, transient_count, slots + transient_start))
        {
            log_vulkan_warning("failed to acquire pool slots for conservative scalar transport (cartesian c-grid)");
            return false;
        }
        int interface_slot = -1;
        if (!acquire_pool_slots(capture.bytes(), 1, &interface_slot))
        {
            release_pool_slots(slots + transient_start, transient_count);
            return false;
        }
        std::array<int, kMaxBatchScalars + 1> requested_slots{};
        const int requested_count = capture.report_count() ? capture.report_count() + 1 : 0;
        if (requested_count && !acquire_pool_slots(field_bytes, requested_count, requested_slots.data()))
        {
            release_pool_slots(&interface_slot, 1);
            release_pool_slots(slots + transient_start, transient_count);
            return false;
        }
        const auto release_transient_slots = [&]()
        {
            if (requested_count)
                release_pool_slots(requested_slots.data(), requested_count);
            release_pool_slots(&interface_slot, 1);
            release_pool_slots(slots + transient_start, transient_count);
        };
        const int  slot_u                  = slots[0];
        const int  slot_v                  = slots[1];
        const int  slot_w                  = slots[2];
        const int  slot_rho                = slots[3];
        const int  slot_mfx                = slots[4];
        const int  slot_mfy                = slots[5];
        const int  slot_mfz                = slots[6];
        const int  slot_sfx                = slots[7];
        const int  slot_sfy                = slots[8];
        const int  slot_sfz                = slots[9];
        const int  slot_qw                 = slots[10];
        const int  slot_qout               = slots[11];
        const int  slot_rlim               = slots[12];
        const int  slot_rho_begin          = slots[13];

        const bool unified  = device_context_.has_unified_memory;
        auto       host_ptr = [&](int slot) -> void*
        { return unified ? buffer_pool_.device(slot).mapped : buffer_pool_.staging(slot).mapped; };

        // Reuse audited read-only state produced earlier in this model step.
        // Each lookup is independent so telemetry reports the exact field hit
        // rate and a partial hit still avoids its own host/device upload.
        const bool shared_state_reusable[4] = {
            named_slot_reusable("u", u_data, field_bytes),
            named_slot_reusable("v", v_data, field_bytes),
            named_slot_reusable("w", w_data, field_bytes),
            named_slot_reusable("rho", rho_data, field_bytes),
        };
        const float*             shared_state[4]      = {u_data, v_data, w_data, rho_data};
        static const char* const kSharedStateNames[4] = {"u", "v", "w", "rho"};
        for (int s = 0; s < 4; ++s)
        {
            if (!shared_state_reusable[s])
            {
                named_slot_invalidate(kSharedStateNames[s]);
                tracked_memcpy(host_ptr(slots[s]), shared_state[s], field_bytes);
            }
        }
        tracked_memcpy(host_ptr(slot_rho_begin), use_supplied_mass_flux ? supplied_rho_begin : rho_data, field_bytes);
        const bool supplied_flux_reusable = use_supplied_mass_flux &&
                                            named_slot_reusable("cgrid_mass_flux_x", supplied_mfx, field_bytes) &&
                                            named_slot_reusable("cgrid_mass_flux_y", supplied_mfy, field_bytes) &&
                                            named_slot_reusable("cgrid_mass_flux_z", supplied_mfz, field_bytes);
        if (use_supplied_mass_flux && !supplied_flux_reusable)
        {
            tracked_memcpy(host_ptr(slot_mfx), supplied_mfx, field_bytes);
            tracked_memcpy(host_ptr(slot_mfy), supplied_mfy, field_bytes);
            tracked_memcpy(host_ptr(slot_mfz), supplied_mfz, field_bytes);
        }
        for (int s = 0; s < n_scalars; ++s)
        {
            tracked_memcpy(host_ptr(slots[kFixedSlots + s]), q_in[s], field_bytes);
        }

        // Fixed descriptor sets, one per pipeline, bound once.
        int mass_bindings[7] = {slot_u, slot_v, slot_w, slot_rho, slot_mfx, slot_mfy, slot_mfz};
        update_pooled_descriptor_set(
            pipeline(KernelId::AdvectConservativeMassFluxCartesianCgrid), mass_bindings, field_bytes, 7);
        int flux_bindings[7] = {slot_qw, slot_mfx, slot_mfy, slot_mfz, slot_sfx, slot_sfy, slot_sfz};
        update_pooled_descriptor_set(
            pipeline(KernelId::AdvectConservativeScalarFluxCartesianCgrid), flux_bindings, field_bytes, 7);
        int update_bindings[13] = {slot_qw,
                                   slot_rho,
                                   slot_mfx,
                                   slot_mfy,
                                   slot_mfz,
                                   slot_sfx,
                                   slot_sfy,
                                   slot_sfz,
                                   slot_qout,
                                   slot_rlim,
                                   slot_rho_begin,
                                   interface_slot,
                                   requested_count ? requested_slots[0] : interface_slot};
        std::array<VkDeviceSize, 13> update_ranges;
        update_ranges.fill(field_bytes);
        update_ranges[11] = capture.bytes();
        update_ranges[12] = requested_count ? field_bytes : 4;
        update_pooled_descriptor_set(
            pipeline(KernelId::AdvectConservativeUpdateCartesianCgrid), update_bindings, field_bytes, 13,
            update_ranges.data());
        int pd_bindings[10] = {
            slot_qw, slot_rho, slot_sfx, slot_sfy, slot_sfz, slot_rlim, slot_mfx, slot_mfy, slot_mfz, slot_rho_begin};
        update_pooled_descriptor_set(pipeline(KernelId::AdvectPdLimiterCartesianCgrid), pd_bindings, field_bytes, 10);
        vkResetCommandBuffer_(cmd_buf_, 0);
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (begin_profiled_command_buffer(begin_info) != VK_SUCCESS)
        {
            release_transient_slots();
            return false;
        }

        // Conservative full barrier between every stage: the recording
        // interleaves compute dispatches with device-to-device transfer
        // copies, so cover both stages and both access directions.
        VkMemoryBarrier full_barrier{};
        full_barrier.sType                     = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        full_barrier.srcAccessMask             = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        full_barrier.dstAccessMask             = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                                 VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        const VkPipelineStageFlags both_stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        auto                       stage_barrier = [&]()
        { vkCmdPipelineBarrier_(cmd_buf_, both_stages, both_stages, 0, 1, &full_barrier, 0, nullptr, 0, nullptr); };

        // ── H2D transfer ──
        if (unified)
        {
            VkMemoryBarrier host_barrier{};
            host_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            host_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            host_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(
                cmd_buf_, VK_PIPELINE_STAGE_HOST_BIT, both_stages, 0, 1, &host_barrier, 0, nullptr, 0, nullptr);
        }
        else
        {
            VkBufferCopy copy_region{};
            copy_region.size = field_bytes;
            for (int s = 0; s < 4; ++s)
            {
                if (shared_state_reusable[s])
                {
                    continue;
                }
                tracked_cmd_copy_buffer(
                    buffer_pool_.staging(slots[s]).buffer, buffer_pool_.device(slots[s]).buffer, 1, &copy_region);
            }
            tracked_cmd_copy_buffer(buffer_pool_.staging(slot_rho_begin).buffer,
                                    buffer_pool_.device(slot_rho_begin).buffer,
                                    1,
                                    &copy_region);
            if (use_supplied_mass_flux && !supplied_flux_reusable)
            {
                for (int s = 4; s <= 6; ++s)
                {
                    tracked_cmd_copy_buffer(
                        buffer_pool_.staging(slots[s]).buffer, buffer_pool_.device(slots[s]).buffer, 1, &copy_region);
                }
            }
            for (int s = 0; s < n_scalars; ++s)
            {
                tracked_cmd_copy_buffer(buffer_pool_.staging(slots[kFixedSlots + s]).buffer,
                                        buffer_pool_.device(slots[kFixedSlots + s]).buffer,
                                        1,
                                        &copy_region);
            }
            stage_barrier();
        }

        // Standalone dispatches retain the historical final-state mass-flux
        // pass. Coupled production dispatches consume the acoustic loop's
        // supplied time-averaged fields and must not recompute them.
        if (!use_supplied_mass_flux)
        {
            AdvectMassFluxCartesianCgridPushConstants pc{};
            pc.nx                  = nx;
            pc.ny                  = ny;
            pc.nz                  = nz;
            pc.pad0                = 0;
            pc.frame_velocity_u_ms = frame_velocity_u_ms;
            pc.frame_velocity_v_ms = frame_velocity_v_ms;
            vkCmdBindPipeline_(cmd_buf_,
                               VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline(KernelId::AdvectConservativeMassFluxCartesianCgrid).pipeline);
            vkCmdBindDescriptorSets_(cmd_buf_,
                                     VK_PIPELINE_BIND_POINT_COMPUTE,
                                     pipeline(KernelId::AdvectConservativeMassFluxCartesianCgrid).pipeline_layout,
                                     0,
                                     1,
                                     &pipeline(KernelId::AdvectConservativeMassFluxCartesianCgrid).descriptor_set,
                                     0,
                                     nullptr);
            vkCmdPushConstants_(cmd_buf_,
                                pipeline(KernelId::AdvectConservativeMassFluxCartesianCgrid).pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT,
                                0,
                                sizeof(pc),
                                &pc);
            record_linear_group_dispatch(workgroup_count, KernelId::AdvectConservativeMassFluxCartesianCgrid);
        }
        stage_barrier();

        // ── Per-scalar: copy in, flux, update, copy out ──
        VkBufferCopy field_copy{};
        field_copy.size = field_bytes;
        for (int s = 0; s < n_scalars; ++s)
        {
            const int slot_qio = slots[kFixedSlots + s];

            tracked_cmd_copy_buffer(
                buffer_pool_.device(slot_qio).buffer, buffer_pool_.device(slot_qw).buffer, 1, &field_copy);
            stage_barrier();

            {
                AdvectFluxDimsCartesianCgridPushConstants pc{};
                pc.nx   = nx;
                pc.ny   = ny;
                pc.nz   = nz;
                pc.pad0 = (dispatch.weno ? 0 : 1) | (dispatch.preserve_positive_support ? 2 : 0) |
                          (positive_definite && positive_definite[s] != 0 ? 4 : 0);
                vkCmdBindPipeline_(cmd_buf_,
                                   VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline(KernelId::AdvectConservativeScalarFluxCartesianCgrid).pipeline);
                vkCmdBindDescriptorSets_(cmd_buf_,
                                         VK_PIPELINE_BIND_POINT_COMPUTE,
                                         pipeline(KernelId::AdvectConservativeScalarFluxCartesianCgrid).pipeline_layout,
                                         0,
                                         1,
                                         &pipeline(KernelId::AdvectConservativeScalarFluxCartesianCgrid).descriptor_set,
                                         0,
                                         nullptr);
                vkCmdPushConstants_(cmd_buf_,
                                    pipeline(KernelId::AdvectConservativeScalarFluxCartesianCgrid).pipeline_layout,
                                    VK_SHADER_STAGE_COMPUTE_BIT,
                                    0,
                                    sizeof(pc),
                                    &pc);
                record_linear_group_dispatch(workgroup_count, KernelId::AdvectConservativeScalarFluxCartesianCgrid);
            }
            stage_barrier();

            // Positive-definite renormalization pass (SCIENCE.md
            // M5): only for PD scalars (moisture, tracer) - theta
            // skips both the dispatch and the donor scaling.
            const bool pd_scalar = positive_definite != nullptr && positive_definite[s] != 0 &&
                                   pipeline(KernelId::AdvectPdLimiterCartesianCgrid).is_ready();
            if (pd_scalar)
            {
                AdvectPdLimiterPushConstants pc{};
                pc.nx               = nx;
                pc.ny               = ny;
                pc.nz               = nz;
                pc.use_coupled_mass = (use_supplied_mass_flux ? 1 : 0) | (dispatch.preserve_positive_support ? 2 : 0);
                pc.inv_dx           = inv_dx;
                pc.inv_dy           = inv_dy;
                pc.inv_dz           = inv_dz;
                pc.dt_step          = dt_step;
                vkCmdBindPipeline_(cmd_buf_,
                                   VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline(KernelId::AdvectPdLimiterCartesianCgrid).pipeline);
                vkCmdBindDescriptorSets_(cmd_buf_,
                                         VK_PIPELINE_BIND_POINT_COMPUTE,
                                         pipeline(KernelId::AdvectPdLimiterCartesianCgrid).pipeline_layout,
                                         0,
                                         1,
                                         &pipeline(KernelId::AdvectPdLimiterCartesianCgrid).descriptor_set,
                                         0,
                                         nullptr);
                vkCmdPushConstants_(cmd_buf_,
                                    pipeline(KernelId::AdvectPdLimiterCartesianCgrid).pipeline_layout,
                                    VK_SHADER_STAGE_COMPUTE_BIT,
                                    0,
                                    sizeof(pc),
                                    &pc);
                record_linear_group_dispatch(workgroup_count, KernelId::AdvectPdLimiterCartesianCgrid);
                stage_barrier();
            }

            {
                AdvectUpdateCartesianCgridPushConstants pc{};
                pc.nx               = nx;
                pc.ny               = ny;
                pc.nz               = nz;
                pc.use_pd           = pd_scalar ? 1 : 0;
                pc.inv_dx           = inv_dx;
                pc.inv_dy           = inv_dy;
                pc.inv_dz           = inv_dz;
                pc.dt_step          = dt_step;
                pc.rho_floor        = rho_floor;
                pc.use_coupled_mass = use_supplied_mass_flux ? 1 : 0;
                capture.configure(s, pc);
                pc.weno = (dispatch.weno ? 1 : 0) | (dispatch.preserve_positive_support ? 2 : 0);
                vkCmdBindPipeline_(cmd_buf_,
                                   VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline(KernelId::AdvectConservativeUpdateCartesianCgrid).pipeline);
                vkCmdBindDescriptorSets_(cmd_buf_,
                                         VK_PIPELINE_BIND_POINT_COMPUTE,
                                         pipeline(KernelId::AdvectConservativeUpdateCartesianCgrid).pipeline_layout,
                                         0,
                                         1,
                                         &pipeline(KernelId::AdvectConservativeUpdateCartesianCgrid).descriptor_set,
                                         0,
                                         nullptr);
                vkCmdPushConstants_(cmd_buf_,
                                    pipeline(KernelId::AdvectConservativeUpdateCartesianCgrid).pipeline_layout,
                                    VK_SHADER_STAGE_COMPUTE_BIT,
                                    0,
                                    sizeof(pc),
                                    &pc);
                record_linear_group_dispatch(workgroup_count, KernelId::AdvectConservativeUpdateCartesianCgrid);
            }
            stage_barrier();

            // The update pass's q_out_work is always the final value. Scalar
            // diffusion is a separate live-density physics/numerics stage.
            tracked_cmd_copy_buffer(
                buffer_pool_.device(slot_qout).buffer, buffer_pool_.device(slot_qio).buffer, 1, &field_copy);
            const int report_index = capture.report_index(s);
            if (report_index >= 0)
                tracked_cmd_copy_buffer(buffer_pool_.device(requested_slots[0]).buffer,
                    buffer_pool_.device(requested_slots[1 + report_index]).buffer, 1, &field_copy);
            stage_barrier();
        }

        // ── D2H transfer (the q_io slots hold the results) ──
        if (unified)
        {
            VkMemoryBarrier out_barrier{};
            out_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            out_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            out_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  0,
                                  1,
                                  &out_barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }
        else
        {
            VkBufferCopy copy_region{};
            copy_region.size = field_bytes;
            for (int s = 0; s < n_scalars; ++s)
            {
                tracked_cmd_copy_buffer(buffer_pool_.device(slots[kFixedSlots + s]).buffer,
                                        buffer_pool_.staging(slots[kFixedSlots + s]).buffer,
                                        1,
                                        &copy_region);
            }
        }

        if (!unified)
        {
            VkBufferCopy requested_copy{};
            requested_copy.size = field_bytes;
            for (int index = 1; index < requested_count; ++index)
                tracked_cmd_copy_buffer(buffer_pool_.device(requested_slots[index]).buffer,
                    buffer_pool_.staging(requested_slots[index]).buffer, 1, &requested_copy);
        }

        if (!unified && capture.enabled())
        {
            VkBufferCopy interface_copy{};
            interface_copy.size = capture.bytes();
            tracked_cmd_copy_buffer(buffer_pool_.device(interface_slot).buffer,
                                    buffer_pool_.staging(interface_slot).buffer, 1, &interface_copy);
        }

        if (end_profiled_command_buffer() != VK_SUCCESS)
        {
            release_transient_slots();
            return false;
        }

        VkSubmitInfo submit_info{};
        submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers    = &cmd_buf_;
        if (reset_fence_and_submit("transport_batch_cartesian_cgrid", submit_info) != VK_SUCCESS)
        {
            release_transient_slots();
            return false;
        }

        if (timed_wait_fence("transport_batch_cartesian_cgrid") != VK_SUCCESS)
        {
            release_transient_slots();
            return false;
        }

        try
        {
            nimbus::compute::publish_cartesian_scalar_batch(
                dispatch, capture, static_cast<const float*>(host_ptr(interface_slot)), total_points,
                [&](int scalar)
                {
                    return static_cast<const float*>(host_ptr(slots[kFixedSlots + scalar]));
                },
                scalar_attribution_workspace_,
                [&](int scalar) -> const float*
                {
                    const int index = capture.report_index(scalar);
                    return index >= 0 ? static_cast<const float*>(host_ptr(requested_slots[1 + index])) : nullptr;
                },
                [&](void* destination, const void* source, std::size_t bytes)
                {
                    tracked_memcpy(destination, source, bytes);
                });
        }
        catch (...)
        {
            release_transient_slots();
            throw;
        }

        // Store site: the named u/v/w/rho slots now hold this step's
        // uploaded content (read-only in every pass above).
        named_slot_commit("u", u_data, field_bytes);
        named_slot_commit("v", v_data, field_bytes);
        named_slot_commit("w", w_data, field_bytes);
        named_slot_commit("rho", rho_data, field_bytes);
        if (use_supplied_mass_flux)
        {
            named_slot_commit("cgrid_mass_flux_x", supplied_mfx, field_bytes);
            named_slot_commit("cgrid_mass_flux_y", supplied_mfy, field_bytes);
            named_slot_commit("cgrid_mass_flux_z", supplied_mfz, field_bytes);
        }

        release_transient_slots();
        return true;
#else
        (void)dispatch;
        return false;
#endif
    }

    // ── Lin microphysics (GPU audit Phase 3.3) ─────────────────

    bool supports_lin_pointwise_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Pointwise physics: coordinate-agnostic (no spatial stencil).
        return pipeline(KernelId::LinPointwise).is_ready();
#else
        return false;
#endif
    }

    bool supports_lin_sedimentation_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Vertical-column physics: coordinate-agnostic.
        return pipeline(KernelId::LinSedimentation).is_ready();
#else
        return false;
#endif
    }

    bool supports_atomic_lin_fused() const override
    {
        return supports(MicrophysicsKernelOp::LinFused);
    }

    bool dispatch_lin_fused(const LinFusedDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        const LinPointwiseDispatch&     point = dispatch.pointwise;
        const LinSedimentationDispatch& sed   = dispatch.sedimentation;
        std::size_t                     cells = 0, bytes = 0;
        if (!nimbus::compute::lin::valid_lin_fused(dispatch, cells, bytes) ||
            !pipeline(KernelId::LinPointwise).is_ready() || !pipeline(KernelId::LinSedimentation).is_ready() ||
            !dependency_batching_enabled() || !submission_lifecycle_.can_dispatch())
            return false;

        uint32_t     total_points = 0;
        VkDeviceSize field_bytes  = 0;
        if (!checked_field_dispatch(point.nx, point.ny, point.nz, total_points, field_bytes, "fused Lin pointwise"))
        {
            return false;
        }
        uint32_t     total_columns        = 0;
        VkDeviceSize ignored_column_bytes = 0;
        if (!checked_field_dispatch(
                point.nx, point.ny, 1, total_columns, ignored_column_bytes, "fused Lin sedimentation"))
        {
            return false;
        }

        // The backend-neutral scheduler owns the dependency decision. Vulkan
        // below only materializes the API-specific memory barrier it requests.
        nimbus::compute::DispatchGraph graph;
        const auto                     point_node = graph.add(KernelId::LinPointwise,
                                                              {{"dqr_dt", nimbus::compute::ResourceAccess::Write},
                                                               {"dqs_dt", nimbus::compute::ResourceAccess::Write},
                                                               {"dqg_dt", nimbus::compute::ResourceAccess::Write},
                                                               { "dqh_dt",
                                                                 nimbus::compute::ResourceAccess::Write }});
        const auto sedimentation_node             = graph.add(KernelId::LinSedimentation,
                                                              {{"dqr_dt", nimbus::compute::ResourceAccess::ReadWrite},
                                                               {"dqs_dt", nimbus::compute::ResourceAccess::ReadWrite},
                                                               {"dqg_dt", nimbus::compute::ResourceAccess::ReadWrite},
                                                               { "dqh_dt",
                                                                 nimbus::compute::ResourceAccess::ReadWrite }});

        const auto point_pc = nimbus::compute::lin::lin_pointwise_constants(point);
        const auto sed_pc   = nimbus::compute::lin::lin_sedimentation_constants(sed);

        // One stable allocation graph for both pipelines:
        // State 0..8 (p at 1), rho_d 9, final tendencies 10..17, layer thickness 18.
        constexpr int kSlotCount = 19;
        int           slots[kSlotCount];
        for (int& slot : slots)
            slot = -1;
        bool         named[kSlotCount]{};
        const char*  state_names[9]  = {"theta", nullptr, "qv", "qc", "qr", "qi", "qs", "qg", "qh"};
        const float* state_inputs[9] = {
            point.theta_data, point.p_data, point.qv, point.qc, point.qr, point.qi, point.qs, point.qg, point.qh};
        bool          state_reuse[9]{};
        constexpr int kNamedStateCount = 8;
        const char*   named_state_names[kNamedStateCount];
        VkDeviceSize  named_state_sizes[kNamedStateCount];
        int           named_state_logical[kNamedStateCount];
        int           named_state_slots[kNamedStateCount];
        int           named_state_count = 0;
        for (int i = 0; i < 9; ++i)
        {
            if (state_names[i] != nullptr)
            {
                named[i]                               = true;
                named_state_names[named_state_count]   = state_names[i];
                named_state_sizes[named_state_count]   = field_bytes;
                named_state_logical[named_state_count] = i;
                ++named_state_count;
            }
        }
        bool acquire_failed =
            !named_slots_acquire_many(named_state_names, named_state_sizes, named_state_count, named_state_slots);
        if (!acquire_failed)
        {
            for (int n = 0; n < named_state_count; ++n)
            {
                const int logical    = named_state_logical[n];
                slots[logical]       = named_state_slots[n];
                state_reuse[logical] = named_slot_reusable(named_state_names[n], state_inputs[logical], field_bytes);
                if (!state_reuse[logical])
                {
                    named_slot_commit(named_state_names[n], state_inputs[logical], field_bytes);
                    named_slot_invalidate(named_state_names[n]);
                }
            }
        }
        const int kTransientCount       = sed.layer_thickness_m ? 11 : 10;
        const int transient_logical[11] = {1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
        int       transient_slots[11];
        if (!acquire_failed)
        {
            acquire_failed = !acquire_pool_slots(field_bytes, kTransientCount, transient_slots);
            if (!acquire_failed)
                for (int n = 0; n < kTransientCount; ++n)
                    slots[transient_logical[n]] = transient_slots[n];
        }
        const auto release_transients = [&]()
        {
            if (!submission_lifecycle_.may_release_transients())
                return;
            for (int i = 0; i < kSlotCount; ++i)
            {
                if (!named[i] && slots[i] >= 0)
                    buffer_pool_.release(slots[i]);
            }
        };
        if (acquire_failed)
        {
            release_transients();
            log_vulkan_warning("failed to acquire slots for fused Lin chain");
            return false;
        }

        int point_bindings[17] = {slots[0],
                                  slots[1],
                                  slots[2],
                                  slots[3],
                                  slots[4],
                                  slots[5],
                                  slots[6],
                                  slots[7],
                                  slots[8],
                                  slots[10],
                                  slots[11],
                                  slots[12],
                                  slots[13],
                                  slots[14],
                                  slots[15],
                                  slots[16],
                                  slots[17]};
        int sed_bindings[10]   = {slots[4],
                                  slots[6],
                                  slots[7],
                                  slots[8],
                                  slots[9],
                                  sed.layer_thickness_m ? slots[18] : slots[9],
                                  slots[13],
                                  slots[15],
                                  slots[16],
                                  slots[17]};
        update_pooled_descriptor_set(pipeline(KernelId::LinPointwise), point_bindings, field_bytes, 17);
        update_pooled_descriptor_set(pipeline(KernelId::LinSedimentation), sed_bindings, field_bytes, 10);

        const bool unified  = device_context_.has_unified_memory;
        auto       host_ptr = [&](int slot) -> void*
        { return unified ? buffer_pool_.device(slot).mapped : buffer_pool_.staging(slot).mapped; };
        for (int i = 0; i < 9; ++i)
        {
            if (!state_reuse[i])
            {
                tracked_memcpy(host_ptr(slots[i]), state_inputs[i], field_bytes);
            }
        }
        tracked_memcpy(host_ptr(slots[9]), sed.rho_d, field_bytes);
        if (sed.layer_thickness_m)
            tracked_memcpy(host_ptr(slots[18]), sed.layer_thickness_m, field_bytes);
        compute_telemetry_.note_writeonly_output_upload_avoided(8u * static_cast<std::uint64_t>(field_bytes));

        vkResetCommandBuffer_(cmd_buf_, 0);
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (begin_profiled_command_buffer(begin_info) != VK_SUCCESS)
        {
            release_transients();
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
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }
        else
        {
            for (int i = 0; i < 9; ++i)
            {
                if (state_reuse[i])
                    continue;
                VkBufferCopy region{};
                region.size = field_bytes;
                tracked_cmd_copy_buffer(
                    buffer_pool_.staging(slots[i]).buffer, buffer_pool_.device(slots[i]).buffer, 1, &region);
            }
            VkBufferCopy rho_region{};
            rho_region.size = field_bytes;
            tracked_cmd_copy_buffer(
                buffer_pool_.staging(slots[9]).buffer, buffer_pool_.device(slots[9]).buffer, 1, &rho_region);
            if (sed.layer_thickness_m)
                tracked_cmd_copy_buffer(
                    buffer_pool_.staging(slots[18]).buffer, buffer_pool_.device(slots[18]).buffer, 1, &rho_region);
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }

        vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline(KernelId::LinPointwise).pipeline);
        vkCmdBindDescriptorSets_(cmd_buf_,
                                 VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipeline(KernelId::LinPointwise).pipeline_layout,
                                 0,
                                 1,
                                 &pipeline(KernelId::LinPointwise).descriptor_set,
                                 0,
                                 nullptr);
        vkCmdPushConstants_(cmd_buf_,
                            pipeline(KernelId::LinPointwise).pipeline_layout,
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0,
                            sizeof(point_pc),
                            &point_pc);
        record_linear_group_dispatch(1u + (total_points - 1u) / 64u, KernelId::LinPointwise);

        if (graph.requires_barrier(point_node, sedimentation_node))
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }

        vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline(KernelId::LinSedimentation).pipeline);
        vkCmdBindDescriptorSets_(cmd_buf_,
                                 VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipeline(KernelId::LinSedimentation).pipeline_layout,
                                 0,
                                 1,
                                 &pipeline(KernelId::LinSedimentation).descriptor_set,
                                 0,
                                 nullptr);
        vkCmdPushConstants_(cmd_buf_,
                            pipeline(KernelId::LinSedimentation).pipeline_layout,
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0,
                            sizeof(sed_pc),
                            &sed_pc);
        record_linear_group_dispatch(1u + (total_columns - 1u) / 64u, KernelId::LinSedimentation);

        if (unified)
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }
        else
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
            for (int i = 10; i < 18; ++i)
            {
                VkBufferCopy region{};
                region.size = field_bytes;
                tracked_cmd_copy_buffer(
                    buffer_pool_.device(slots[i]).buffer, buffer_pool_.staging(slots[i]).buffer, 1, &region);
            }
        }

        if (end_profiled_command_buffer() != VK_SUCCESS)
        {
            release_transients();
            return false;
        }
        VkSubmitInfo submit_info{};
        submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers    = &cmd_buf_;
        if (reset_fence_and_submit("lin_pointwise_sedimentation_batch", submit_info) != VK_SUCCESS ||
            timed_wait_fence("lin_pointwise_sedimentation_batch") != VK_SUCCESS)
        {
            release_transients();
            return false;
        }

        std::array<const float*, 8> result{};
        for (std::size_t i = 0; i < result.size(); ++i)
            result[i] = static_cast<const float*>(host_ptr(slots[10 + i]));
        if (!nimbus::compute::lin::finite_result(point, result))
        {
            release_transients();
            return false;
        }
        float* tendency_outputs[8] = {point.dtheta_dt,
                                      point.dqv_dt,
                                      point.dqc_dt,
                                      point.dqr_dt,
                                      point.dqi_dt,
                                      point.dqs_dt,
                                      point.dqg_dt,
                                      point.dqh_dt};
        for (int i = 0; i < 8; ++i)
        {
            tracked_memcpy(tendency_outputs[i], host_ptr(slots[10 + i]), field_bytes);
        }
        for (int i = 0; i < 9; ++i)
        {
            if (state_names[i] != nullptr)
            {
                named_slot_commit(state_names[i], state_inputs[i], field_bytes);
            }
        }
        release_transients();
        return true;
#else
        (void)dispatch;
        return false;
#endif
    }

    bool dispatch_lin_pointwise(const LinPointwiseDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        std::size_t cells = 0, bytes = 0;
        if (!pipeline(KernelId::LinPointwise).is_ready() || !nimbus::compute::lin::valid_lin_pointwise(d, cells, bytes))
            return false;
        std::uint32_t work          = 0;
        VkDeviceSize  checked_bytes = 0;
        if (!checked_field_dispatch(d.nx, d.ny, d.nz, work, checked_bytes, "Lin pointwise"))
            return false;
        const auto   pc        = nimbus::compute::lin::lin_pointwise_constants(d);
        const float* inputs[9] = {d.theta_data, d.p_data, d.qv, d.qc, d.qr, d.qi, d.qs, d.qg, d.qh};
        float* destinations[8] = {d.dtheta_dt, d.dqv_dt, d.dqc_dt, d.dqr_dt, d.dqi_dt, d.dqs_dt, d.dqg_dt, d.dqh_dt};
        std::array<std::vector<float>, 8> candidate;
        float*                            outputs[8] = {};
        std::array<const float*, 8>       result{};
        for (std::size_t f = 0; f < candidate.size(); ++f)
        {
            candidate[f].resize(cells, 0);
            outputs[f] = candidate[f].data();
            result[f]  = outputs[f];
        }
        const char* names[9]      = {"theta", nullptr, "qv", "qc", "qr", "qi", "qs", "qg", "qh"};
        const bool  write_only[8] = {true, true, true, true, true, true, true, true};
        if (!dispatch_multi_field_kernel(pipeline(KernelId::LinPointwise),
                                         KernelId::LinPointwise,
                                         &pc,
                                         inputs,
                                         9,
                                         outputs,
                                         8,
                                         d.nx,
                                         d.ny,
                                         d.nz,
                                         work,
                                         nullptr,
                                         names,
                                         write_only) ||
            !nimbus::compute::lin::finite_result(d, result))
            return false;
        for (std::size_t f = 0; f < candidate.size(); ++f)
            std::copy_n(result[f], cells, destinations[f]);
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_thompson_completion(const ThompsonCompletionDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::thompson_completion;
        if (!supports(MicrophysicsKernelOp::ThompsonCompletion) || !contract::validate(d))
            return false;
        constexpr const char* table_name = "thompson_completion_coefficients";
        if (thompson_completion_tables_ != d.tables)
        {
            named_slot_invalidate(table_name);
            thompson_completion_tables_ = d.tables;
        }
        const auto         pc        = contract::constants(d);
        const float*       inputs[2] = {d.input, d.tables->values().data()};
        const VkDeviceSize sizes[2]  = {d.cells * nimbus::compute::thompson_completion_input_stride * sizeof(float),
                                        d.tables->values().size() * sizeof(float)};
        std::vector<float> candidate(d.cells * 18, 0);
        float*             outputs[1]   = {candidate.data()};
        const char*        names[2]     = {nullptr, table_name};
        const bool         immutable[2] = {false, true};
        if (!dispatch_multi_field_kernel(pipeline(KernelId::ThompsonCompletion),
                                         KernelId::ThompsonCompletion,
                                         &pc,
                                         inputs,
                                         2,
                                         outputs,
                                         1,
                                         static_cast<int>(d.cells * 18),
                                         1,
                                         1,
                                         static_cast<uint32_t>(d.cells),
                                         sizes,
                                         names,
                                         nullptr,
                                         true,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         0,
                                         nullptr,
                                         nullptr,
                                         immutable) ||
            !contract::valid_candidate(d, candidate.data()))
            return false;
        std::copy(candidate.begin(), candidate.end(), d.output);
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_thompson_deposition(const ThompsonDepositionDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::thompson_deposition;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::ThompsonDeposition) || !contract::validate(d, cells))
        {
            return false;
        }
        const auto                   constants = contract::constants(d);
        const auto                   inputs    = contract::inputs(d);
        std::array<VkDeviceSize, 13> sizes;
        sizes.fill(cells * sizeof(float));
        sizes[12] = d.tables->values().size() * sizeof(float);
        std::array<std::vector<float>, 13> candidates;
        std::array<float*, 13>             outputs;
        std::array<const float*, 13>       views;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            candidates[f].resize(cells, 0.0f);
            views[f] = outputs[f] = candidates[f].data();
        }
        if (!dispatch_multi_field_kernel(pipeline(KernelId::ThompsonDeposition),
                                         KernelId::ThompsonDeposition,
                                         &constants,
                                         inputs.data(),
                                         inputs.size(),
                                         outputs.data(),
                                         outputs.size(),
                                         d.nx,
                                         d.ny,
                                         d.nz,
                                         static_cast<uint32_t>(cells),
                                         sizes.data()) ||
            !contract::valid_candidate(d, views, cells))
        {
            return false;
        }
        const auto destinations = contract::outputs(d);
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(outputs[f], cells, destinations[f]);
        }
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_thompson_freezing(const ThompsonFreezingDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::thompson_freezing;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::ThompsonFreezing) || !contract::validate(d, cells))
        {
            return false;
        }
        const auto                   constants = contract::constants(d);
        const auto                   inputs    = contract::inputs(d);
        std::array<VkDeviceSize, 16> sizes;
        sizes.fill(cells * sizeof(float));
        sizes[15] = d.tables->values().size() * sizeof(float);
        std::array<std::vector<float>, 13> candidates;
        std::array<float*, 13>             outputs;
        std::array<const float*, 13>       views;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            candidates[f].resize(cells, 0.0f);
            views[f] = outputs[f] = candidates[f].data();
        }
        if (!dispatch_multi_field_kernel(pipeline(KernelId::ThompsonFreezing),
                                         KernelId::ThompsonFreezing,
                                         &constants,
                                         inputs.data(),
                                         inputs.size(),
                                         outputs.data(),
                                         outputs.size(),
                                         d.nx,
                                         d.ny,
                                         d.nz,
                                         static_cast<uint32_t>(cells),
                                         sizes.data()) ||
            !contract::valid_candidate(d, views, cells))
        {
            return false;
        }
        const auto destinations = contract::outputs(d);
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(outputs[f], cells, destinations[f]);
        }
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_thompson_cloud_phase(const ThompsonCloudPhaseDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::thompson_phase;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::ThompsonCloudPhase) || !contract::validate(d, cells))
        {
            return false;
        }
        const auto                   constants = contract::constants(d);
        const auto                   inputs    = contract::inputs(d);
        std::array<VkDeviceSize, 12> sizes;
        sizes.fill(cells * sizeof(float));
        std::array<std::vector<float>, 6> candidates;
        std::array<float*, 6>             outputs;
        std::array<const float*, 6>       views;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            candidates[f].resize(cells, 0.0f);
            views[f] = outputs[f] = candidates[f].data();
        }
        if (!dispatch_multi_field_kernel(pipeline(KernelId::ThompsonCloudPhase),
                                         KernelId::ThompsonCloudPhase,
                                         &constants,
                                         inputs.data(),
                                         inputs.size(),
                                         outputs.data(),
                                         outputs.size(),
                                         d.nx,
                                         d.ny,
                                         d.nz,
                                         static_cast<uint32_t>(cells),
                                         sizes.data()) ||
            !contract::valid_candidate(d, views, cells))
        {
            return false;
        }
        const auto destinations = contract::outputs(d);
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(outputs[f], cells, destinations[f]);
        }
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool supports_rk_tendency() const override
    {
        return pipeline(KernelId::RkTendency).is_ready();
    }
    bool supports_held_sources() const override
    {
        return pipeline(KernelId::HeldSources).is_ready();
    }
    bool supports_column_remap() const override
    {
        return pipeline(KernelId::ColumnRemap).is_ready();
    }
    bool dispatch_cgrid_boundary(const CgridBoundaryDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract=nimbus::compute::cgrid_boundary;
        if(!contract::validate(d) || !pipeline(KernelId::AcousticBcFaceCartesianCgrid).is_ready())return false;
        const auto count=contract::cells(d),wave_count=contract::waves(d),extent=std::max(count,wave_count);
        std::array<std::vector<float>,6> candidate;
        for(std::size_t f=0;f<5;++f){candidate[f].resize(extent);std::copy_n(d.fields[f],count,candidate[f].data());}
        candidate[5].resize(extent);std::copy_n(d.wave,wave_count,candidate[5].data());
        std::array<float*,6> pointers;std::array<const float*,6> observed;
        for(std::size_t f=0;f<6;++f){pointers[f]=candidate[f].data();observed[f]=pointers[f];}
        std::array<nimbus::compute::abi::AcousticBcFacePushConstants, 5> constants;
        for(int face=0;face<5;++face)
        {
            constants[face] = d.constants;
            constants[face].face = face;
        }
        if (!dispatch_multi_field_kernel(
                pipeline(KernelId::AcousticBcFaceCartesianCgrid), KernelId::AcousticBcFaceCartesianCgrid,
                constants.data(), nullptr, 0, pointers.data(), 6, int(extent), 1, 1, uint32_t(count),
                nullptr, nullptr, nullptr, true, nullptr, nullptr, nullptr, 0,
                nullptr, nullptr, nullptr, nullptr, constants.size()))
            return false;
        if(!contract::finite_candidate(observed,d))return false;
        for(std::size_t f=0;f<5;++f)std::copy_n(candidate[f].data(),count,d.output[f]);
        std::copy_n(candidate[5].data(),wave_count,d.wave_output);
        return true;
#else
        (void)d;return false;
#endif
    }

    bool dispatch_slab_column(const SlabColumnDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::slab_column;
        if (!contract::validate(d) || !supports(BoundaryLayerKernelOp::SlabColumn))
            return false;
        const auto         cells       = std::size_t(d.constants.columns);
        const auto         count       = contract::output_count(d);
        const VkDeviceSize input_bytes = contract::input_count(d) * sizeof(float);
        std::vector<float> candidate(count);
        float*             output = candidate.data();
        if (!dispatch_multi_field_kernel(pipeline(KernelId::SlabColumn),
                                         KernelId::SlabColumn,
                                         &d.constants,
                                         &d.input,
                                         1,
                                         &output,
                                         1,
                                         int(count),
                                         1,
                                         1,
                                         uint32_t(cells),
                                         &input_bytes))
            return false;
        if (!contract::finite_candidate(candidate.data(), d))
            return false;
        std::copy(candidate.begin(), candidate.end(), d.output);
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_simple_grey(const SimpleGreyDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::simple_grey;
        if (!contract::validate(d) || !supports(RadiationKernelOp::SimpleGrey))
            return false;
        const auto         cells       = std::size_t(d.constants.columns);
        const auto         count       = contract::output_count(d);
        const VkDeviceSize input_bytes = contract::input_count(d) * sizeof(float);
        std::vector<float> candidate(count);
        float*             output = candidate.data();
        if (!dispatch_multi_field_kernel(pipeline(KernelId::SimpleGrey),
                                         KernelId::SimpleGrey,
                                         &d.constants,
                                         &d.input,
                                         1,
                                         &output,
                                         1,
                                         int(count),
                                         1,
                                         1,
                                         uint32_t(cells),
                                         &input_bytes))
            return false;
        if (!contract::finite_candidate(candidate.data(), d))
            return false;
        std::copy(candidate.begin(), candidate.end(), d.output);
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_held_sources(const HeldSourcesDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::held_sources;
        if (!contract::validate(d) || !supports_held_sources())
            return false;
        const auto         cells       = contract::cells(d);
        const auto         count       = cells * HeldSourcesDispatch::output_stride;
        const VkDeviceSize input_bytes = cells * HeldSourcesDispatch::input_stride * sizeof(float);
        std::vector<float> candidate(count);
        float*             output = candidate.data();
        if (!dispatch_multi_field_kernel(pipeline(KernelId::HeldSources),
                                         KernelId::HeldSources,
                                         &d.constants,
                                         &d.input,
                                         1,
                                         &output,
                                         1,
                                         int(count),
                                         1,
                                         1,
                                         uint32_t(cells),
                                         &input_bytes))
            return false;
        if (!contract::finite_candidate(candidate.data(), d))
            return false;
        std::copy(candidate.begin(), candidate.end(), d.output);
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_column_remap(const ColumnRemapDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::column_remap;
        if (!contract::validate(d) || !supports_column_remap())
            return false;
        const auto         cells       = contract::cells(d);
        const auto         count       = cells * ColumnRemapDispatch::output_stride;
        const VkDeviceSize input_bytes = cells * ColumnRemapDispatch::input_stride * sizeof(float);
        std::vector<float> candidate(count);
        float*             output = candidate.data();
        if (!dispatch_multi_field_kernel(pipeline(KernelId::ColumnRemap),
                                         KernelId::ColumnRemap,
                                         &d.constants,
                                         &d.input,
                                         1,
                                         &output,
                                         1,
                                         int(count),
                                         1,
                                         1,
                                         uint32_t(cells),
                                         &input_bytes))
            return false;
        if (!contract::finite_candidate(candidate.data(), d))
            return false;
        std::copy(candidate.begin(), candidate.end(), d.output);
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool supports_reference_atmosphere() const override
    {
        return pipeline(KernelId::ReferenceAtmosphere).is_ready();
    }
    bool dispatch_reference_atmosphere(const ReferenceAtmosphereDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::reference_atmosphere;
        if (!contract::validate(d) || !supports_reference_atmosphere())
            return false;
        const auto         cells       = std::size_t(d.constants.count);
        const auto         count       = cells * ReferenceAtmosphereDispatch::output_stride;
        const VkDeviceSize input_bytes = contract::input_count(d) * sizeof(float);
        std::vector<float> candidate(count);
        float*             output = candidate.data();
        if (!dispatch_multi_field_kernel(pipeline(KernelId::ReferenceAtmosphere),
                                         KernelId::ReferenceAtmosphere,
                                         &d.constants,
                                         &d.input,
                                         1,
                                         &output,
                                         1,
                                         int(count),
                                         1,
                                         1,
                                         uint32_t(cells),
                                         &input_bytes))
            return false;
        if (!contract::finite_candidate(candidate.data(), d))
            return false;
        std::copy(candidate.begin(), candidate.end(), d.output);
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool supports_moving_energy() const override
    {
        return pipeline(KernelId::MovingEnergy).is_ready();
    }
    bool dispatch_moving_energy(const MovingEnergyDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::moving_energy;
        if (!contract::validate(d) || !supports_moving_energy())
            return false;
        const auto         cells       = contract::cells(d);
        const auto         count       = cells * MovingEnergyDispatch::output_stride;
        const VkDeviceSize input_bytes = cells * MovingEnergyDispatch::input_stride * sizeof(float);
        std::vector<float> candidate(count);
        float*             output = candidate.data();
        if (!dispatch_multi_field_kernel(pipeline(KernelId::MovingEnergy),
                                         KernelId::MovingEnergy,
                                         &d.constants,
                                         &d.input,
                                         1,
                                         &output,
                                         1,
                                         int(count),
                                         1,
                                         1,
                                         uint32_t(cells),
                                         &input_bytes))
            return false;
        if (!contract::finite_candidate(candidate.data(), d))
            return false;
        std::copy(candidate.begin(), candidate.end(), d.output);
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool supports_moving_momentum() const override
    {
        return pipeline(KernelId::MovingMomentum).is_ready();
    }
    bool dispatch_conservative_reduction(const ConservativeReductionDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::conservative_reduction;
        if (!contract::validate(d) || !pipeline(KernelId::ConservativeReduction).is_ready())
            return false;
        const auto         cells       = std::size_t(d.constants.rows);
        const auto         count       = contract::output_values(d);
        const VkDeviceSize input_bytes = contract::input_values(d) * sizeof(float);
        std::vector<float> candidate(count);
        float*             output = candidate.data();
        if (!dispatch_multi_field_kernel(pipeline(KernelId::ConservativeReduction),
                                         KernelId::ConservativeReduction,
                                         &d.constants,
                                         &d.input,
                                         1,
                                         &output,
                                         1,
                                         int(count),
                                         1,
                                         1,
                                         uint32_t(cells),
                                         &input_bytes))
            return false;
        if (!contract::finite_candidate(candidate.data(), d))
            return false;
        std::copy(candidate.begin(), candidate.end(), d.output);
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_conservative_momentum(const ConservativeMomentumDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::conservative_momentum;
        if (!contract::validate(d) || !pipeline(KernelId::ConservativeMomentum).is_ready())
            return false;
        const auto         cells       = contract::cells(d);
        const auto         count       = contract::output_values(d);
        const VkDeviceSize input_bytes = cells * ConservativeMomentumDispatch::input_stride * sizeof(float);
        std::vector<float> candidate(count);
        float*             output = candidate.data();
        if (!dispatch_multi_field_kernel(pipeline(KernelId::ConservativeMomentum),
                                         KernelId::ConservativeMomentum,
                                         &d.constants,
                                         &d.input,
                                         1,
                                         &output,
                                         1,
                                         int(count),
                                         1,
                                         1,
                                         uint32_t(cells),
                                         &input_bytes))
            return false;
        if (!contract::finite_candidate(candidate.data(), d))
            return false;
        std::copy(candidate.begin(), candidate.end(), d.output);
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_moving_momentum(const MovingMomentumDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::moving_momentum;
        if (!contract::validate(d) || !supports_moving_momentum())
            return false;
        const auto         cells       = contract::cells(d);
        const auto         count       = cells * MovingMomentumDispatch::output_stride;
        const VkDeviceSize input_bytes = cells * MovingMomentumDispatch::input_stride * sizeof(float);
        std::vector<float> candidate(count);
        float*             output = candidate.data();
        if (!dispatch_multi_field_kernel(pipeline(KernelId::MovingMomentum),
                                         KernelId::MovingMomentum,
                                         &d.constants,
                                         &d.input,
                                         1,
                                         &output,
                                         1,
                                         int(count),
                                         1,
                                         1,
                                         uint32_t(cells),
                                         &input_bytes))
            return false;
        if (!contract::finite_candidate(candidate.data(), d))
            return false;
        std::copy(candidate.begin(), candidate.end(), d.output);
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool supports_earth_geometry() const override
    {
        return pipeline(KernelId::EarthGeometry).is_ready();
    }
    bool dispatch_earth_geometry(const EarthGeometryDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::earth_geometry;
        if (!contract::validate(d))
            return false;
        if (!supports_earth_geometry())
            return false;
        const auto                         constants = contract::constants(d);
        const auto                         inputs    = d.input;
        std::array<std::vector<float>, 20> candidates;
        std::array<float*, 20>             outputs;
        std::array<const float*, 20>       views;
        for (int f = 0; f < 20; ++f)
        {
            candidates[f].resize(contract::cells(d));
            views[f] = outputs[f] = candidates[f].data();
        }
        if (!dispatch_multi_field_kernel(pipeline(KernelId::EarthGeometry),
                                         KernelId::EarthGeometry,
                                         &constants,
                                         inputs.data(),
                                         inputs.size(),
                                         outputs.data(),
                                         outputs.size(),
                                         int(contract::cells(d)),
                                         1,
                                         1,
                                         uint32_t(contract::cells(d))))
            return false;
        if (!contract::finite_candidate(views, d))
            return false;
        for (int f = 0; f < 20; ++f)
            std::copy_n(views[f], contract::cells(d), d.output[f]);
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool supports_conservative_remap() const override
    {
        return pipeline(KernelId::ConservativeRemap).is_ready();
    }
    bool dispatch_conservative_remap(const ConservativeRemapDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::conservative_remap;
        if (!contract::validate(d))
            return false;
        if (!supports_conservative_remap())
            return false;
        const auto                         constants = contract::constants(d);
        const auto                         inputs    = d.input;
        std::array<std::vector<float>, 10> candidates;
        std::array<float*, 10>             outputs;
        std::array<const float*, 10>       views;
        for (int f = 0; f < 10; ++f)
        {
            candidates[f].resize(contract::cells(d));
            views[f] = outputs[f] = candidates[f].data();
        }
        if (!dispatch_multi_field_kernel(pipeline(KernelId::ConservativeRemap),
                                         KernelId::ConservativeRemap,
                                         &constants,
                                         inputs.data(),
                                         inputs.size(),
                                         outputs.data(),
                                         outputs.size(),
                                         int(contract::cells(d)),
                                         1,
                                         1,
                                         uint32_t(contract::cells(d))))
            return false;
        if (!contract::finite_candidate(views, d))
            return false;
        for (int f = 0; f < 10; ++f)
            std::copy_n(views[f], contract::cells(d), d.output[f]);
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_collocated_sgs(const CollocatedSgsDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::collocated_sgs;
        if (!contract::validate(d))
            return false;
        if (!supports(TurbulenceKernelOp::CollocatedSgs))
            return false;
        const auto                        constants = contract::constants(d);
        const auto                        inputs    = d.input;
        std::array<std::vector<float>, 8> candidates;
        std::array<float*, 8>             outputs;
        std::array<const float*, 8>       views;
        for (int f = 0; f < 8; ++f)
        {
            candidates[f].resize(contract::cells(d));
            views[f] = outputs[f] = candidates[f].data();
        }
        if (!dispatch_multi_field_kernel(pipeline(KernelId::CollocatedSgs),
                                         KernelId::CollocatedSgs,
                                         &constants,
                                         inputs.data(),
                                         inputs.size(),
                                         outputs.data(),
                                         outputs.size(),
                                         int(contract::cells(d)),
                                         1,
                                         1,
                                         uint32_t(contract::cells(d))))
            return false;
        if (!contract::finite_candidate(views, contract::cells(d)))
            return false;
        for (int f = 0; f < 8; ++f)
            std::copy_n(views[f], contract::cells(d), d.output[f]);
        return true;
#else
        (void)d;
        return false;
#endif
    }

    RkCombinationStatus dispatch_rk_tendency(const RkTendencyDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::rk_tendency;
        if (!contract::validate(d))
            return RkCombinationStatus::InvalidInput;
        if (!supports_rk_tendency())
            return RkCombinationStatus::Unavailable;
        const auto                        constants = contract::constants(d);
        const auto                        inputs    = d.input;
        std::array<std::vector<float>, 5> candidates;
        std::array<float*, 5>             outputs;
        std::array<const float*, 5>       views;
        for (int f = 0; f < 5; ++f)
        {
            candidates[f].resize(contract::cells(d));
            views[f] = outputs[f] = candidates[f].data();
        }
        if (!dispatch_multi_field_kernel(pipeline(KernelId::RkTendency),
                                         KernelId::RkTendency,
                                         &constants,
                                         inputs.data(),
                                         inputs.size(),
                                         outputs.data(),
                                         outputs.size(),
                                         int(contract::cells(d)),
                                         1,
                                         1,
                                         uint32_t(contract::cells(d))))
            return RkCombinationStatus::Unavailable;
        if (!contract::finite_candidate(views, contract::cells(d)))
            return RkCombinationStatus::NonFinite;
        for (int f = 0; f < 5; ++f)
            std::copy_n(views[f], contract::cells(d), d.output[f]);
        return RkCombinationStatus::Completed;
#else
        (void)d;
        return RkCombinationStatus::Unavailable;
#endif
    }

    bool supports_rk_combination() const override
    {
        return pipeline(KernelId::RkCombination).is_ready();
    }
    RkCombinationStatus dispatch_rk_combination(const RkCombinationDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::rk_combination;
        if (!contract::validate(d))
            return RkCombinationStatus::InvalidInput;
        if (!supports_rk_combination())
            return RkCombinationStatus::Unavailable;
        const auto                        constants = contract::constants(d);
        const auto                        inputs    = contract::inputs(d);
        std::array<std::vector<float>, 4> candidates;
        std::array<float*, 4>             outputs;
        std::array<const float*, 4>       views;
        for (int f = 0; f < 4; ++f)
        {
            candidates[f].resize(d.cells);
            views[f] = outputs[f] = candidates[f].data();
        }
        if (!dispatch_multi_field_kernel(pipeline(KernelId::RkCombination),
                                         KernelId::RkCombination,
                                         &constants,
                                         inputs.data(),
                                         inputs.size(),
                                         outputs.data(),
                                         outputs.size(),
                                         int(d.cells),
                                         1,
                                         1,
                                         uint32_t(d.cells)))
            return RkCombinationStatus::Unavailable;
        if (!contract::finite_candidate(views, d.cells))
            return RkCombinationStatus::NonFinite;
        for (int f = 0; f < 4; ++f)
            std::copy_n(views[f], d.cells, d.output[f]);
        return RkCombinationStatus::Completed;
#else
        (void)d;
        return RkCombinationStatus::Unavailable;
#endif
    }

    bool dispatch_milbrandt_sedimentation(const MilbrandtSedimentationDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::milbrandt_sedimentation;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::MilbrandtSedimentation) || !contract::validate(d, cells))
        {
            return false;
        }
        const auto                  constants = contract::constants(d);
        const auto                  table     = contract::coefficients(d);
        std::vector<float>          scratch(12 * cells);
        const auto                  inputs = contract::inputs(d, scratch.data(), table.data());
        std::array<VkDeviceSize, 9> sizes;
        sizes.fill(cells * sizeof(float));
        sizes[7]     = scratch.size() * sizeof(float);
        sizes.back() = sizeof(table);
        std::array<std::vector<float>, 3> candidates;
        std::array<float*, 3>             outputs;
        std::array<const float*, 3>       views;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            candidates[f].resize(cells, 0.0f);
            views[f] = outputs[f] = candidates[f].data();
        }
        if (!dispatch_multi_field_kernel(pipeline(KernelId::MilbrandtSedimentation),
                                         KernelId::MilbrandtSedimentation,
                                         &constants,
                                         inputs.data(),
                                         inputs.size(),
                                         outputs.data(),
                                         outputs.size(),
                                         d.nx,
                                         d.ny,
                                         d.nz,
                                         static_cast<uint32_t>(d.nx) * d.ny,
                                         sizes.data()) ||
            !contract::valid_candidate(d, views, cells))
        {
            return false;
        }
        const auto destinations = d.output;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(outputs[f], cells, destinations[f]);
        }
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_milbrandt_melting(const MilbrandtMeltingDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::milbrandt_melting;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::MilbrandtMelting) || !contract::validate(d, cells))
        {
            return false;
        }
        const auto                   constants = contract::constants(d);
        const auto                   table     = contract::coefficients(d);
        const auto                   inputs    = contract::inputs(d, table.data());
        std::array<VkDeviceSize, 13> sizes;
        sizes.fill(cells * sizeof(float));
        sizes.back() = sizeof(table);
        std::array<std::vector<float>, 11> candidates;
        std::array<float*, 11>             outputs;
        std::array<const float*, 11>       views;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            candidates[f].resize(cells, 0.0f);
            views[f] = outputs[f] = candidates[f].data();
        }
        if (!dispatch_multi_field_kernel(pipeline(KernelId::MilbrandtMelting),
                                         KernelId::MilbrandtMelting,
                                         &constants,
                                         inputs.data(),
                                         inputs.size(),
                                         outputs.data(),
                                         outputs.size(),
                                         d.nx,
                                         d.ny,
                                         d.nz,
                                         static_cast<uint32_t>(cells),
                                         sizes.data()) ||
            !contract::valid_candidate(d, views, cells))
        {
            return false;
        }
        const auto destinations = d.output;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(outputs[f], cells, destinations[f]);
        }
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_milbrandt_cold(const MilbrandtColdDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::milbrandt_cold;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::MilbrandtCold) || !contract::validate(d, cells))
        {
            return false;
        }
        const auto                        constants = contract::constants(d);
        const auto                        packed    = contract::pack(d, cells);
        const std::array<const float*, 2> inputs{packed[0].data(), packed[1].data()};
        std::array<VkDeviceSize, 2>       sizes;
        sizes[0] = packed[0].size() * sizeof(float);
        sizes[1] = packed[1].size() * sizeof(float);
        std::array<std::vector<float>, 14> candidates;
        std::array<float*, 14>             outputs;
        std::array<const float*, 14>       views;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            candidates[f].resize(cells, 0.0f);
            views[f] = outputs[f] = candidates[f].data();
        }
        if (!dispatch_multi_field_kernel(pipeline(KernelId::MilbrandtCold),
                                         KernelId::MilbrandtCold,
                                         &constants,
                                         inputs.data(),
                                         inputs.size(),
                                         outputs.data(),
                                         outputs.size(),
                                         d.nx,
                                         d.ny,
                                         d.nz,
                                         static_cast<uint32_t>(cells),
                                         sizes.data()) ||
            !contract::valid_candidate(d, views, cells))
        {
            return false;
        }
        const auto destinations = d.output;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(outputs[f], cells, destinations[f]);
        }
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_milbrandt_cloud_adjustment(const MilbrandtCloudAdjustmentDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::milbrandt_cloud;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::MilbrandtCloudAdjustment) || !contract::validate(d, cells))
        {
            return false;
        }
        const auto                  constants = contract::constants(d);
        std::vector<float>          velocity;
        const auto                  inputs = contract::inputs(d, cells, velocity);
        std::array<VkDeviceSize, 7> sizes;
        sizes.fill(cells * sizeof(float));
        sizes[6] = contract::velocity_value_count(d, cells) * sizeof(float);
        std::array<std::vector<float>, 7> candidates;
        std::array<float*, 7>             outputs;
        std::array<const float*, 7>       views;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            candidates[f].resize(cells, 0.0f);
            views[f] = outputs[f] = candidates[f].data();
        }
        if (!dispatch_multi_field_kernel(pipeline(KernelId::MilbrandtCloudAdjustment),
                                         KernelId::MilbrandtCloudAdjustment,
                                         &constants,
                                         inputs.data(),
                                         inputs.size(),
                                         outputs.data(),
                                         outputs.size(),
                                         d.nx,
                                         d.ny,
                                         d.nz,
                                         static_cast<uint32_t>(cells),
                                         sizes.data()) ||
            !contract::valid_candidate(d, views, cells))
        {
            return false;
        }
        const auto destinations = d.output;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(outputs[f], cells, destinations[f]);
        }
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_milbrandt_warm_rain(const MilbrandtWarmRainDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::milbrandt_warm;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::MilbrandtWarmRain) || !contract::validate(d, cells))
        {
            return false;
        }
        const auto                   constants    = contract::constants(d);
        const auto                   coefficients = contract::coefficients(d);
        const auto                   inputs       = contract::inputs(d, coefficients.data());
        std::array<VkDeviceSize, 15> sizes;
        sizes.fill(cells * sizeof(float));
        sizes[14] = coefficients.size() * sizeof(float);
        std::array<std::vector<float>, 6> candidates;
        std::array<float*, 6>             outputs;
        std::array<const float*, 6>       views;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            candidates[f].resize(cells, 0.0f);
            views[f] = outputs[f] = candidates[f].data();
        }
        if (!dispatch_multi_field_kernel(pipeline(KernelId::MilbrandtWarmRain),
                                         KernelId::MilbrandtWarmRain,
                                         &constants,
                                         inputs.data(),
                                         inputs.size(),
                                         outputs.data(),
                                         outputs.size(),
                                         d.nx,
                                         d.ny,
                                         d.nz,
                                         static_cast<uint32_t>(cells),
                                         sizes.data()) ||
            !contract::valid_candidate(d, views, cells))
        {
            return false;
        }
        const auto destinations = d.output;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(outputs[f], cells, destinations[f]);
        }
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_thompson_warm_rain(const ThompsonWarmRainDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::thompson_warm;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::ThompsonWarmRain) || !contract::validate(d, cells))
        {
            return false;
        }
        const auto                   constants = contract::constants(d);
        const auto                   inputs    = contract::inputs(d);
        std::array<VkDeviceSize, 14> sizes;
        sizes.fill(cells * sizeof(float));
        sizes[13] = d.coefficients->size() * sizeof(float);
        std::array<std::vector<float>, 10> candidates;
        std::array<float*, 10>             outputs;
        std::array<const float*, 10>       views;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            candidates[f].resize(cells, 0.0f);
            views[f] = outputs[f] = candidates[f].data();
        }
        if (!dispatch_multi_field_kernel(pipeline(KernelId::ThompsonWarmRain),
                                         KernelId::ThompsonWarmRain,
                                         &constants,
                                         inputs.data(),
                                         inputs.size(),
                                         outputs.data(),
                                         outputs.size(),
                                         d.nx,
                                         d.ny,
                                         d.nz,
                                         static_cast<uint32_t>(cells),
                                         sizes.data()) ||
            !contract::valid_candidate(d, views, cells))
        {
            return false;
        }
        const auto destinations = contract::outputs(d);
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(outputs[f], cells, destinations[f]);
        }
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_thompson_cloud_adjustment(const ThompsonCloudAdjustmentDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::thompson_cloud;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::ThompsonCloudAdjustment) || !contract::validate(d, cells))
        {
            return false;
        }
        const auto                   constants = contract::constants(d);
        const auto                   inputs    = contract::inputs(d);
        std::array<VkDeviceSize, 11> sizes;
        sizes.fill(cells * sizeof(float));
        sizes[8] = d.tables->activation().size() * sizeof(float);
        sizes[9] = d.tables->evaporation().size() * sizeof(float);
        std::array<std::vector<float>, 7> candidates;
        std::array<float*, 7>             outputs;
        std::array<const float*, 7>       views;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            candidates[f].resize(cells, 0.0f);
            views[f] = outputs[f] = candidates[f].data();
        }
        if (!dispatch_multi_field_kernel(pipeline(KernelId::ThompsonCloudAdjustment),
                                         KernelId::ThompsonCloudAdjustment,
                                         &constants,
                                         inputs.data(),
                                         inputs.size(),
                                         outputs.data(),
                                         outputs.size(),
                                         d.nx,
                                         d.ny,
                                         d.nz,
                                         static_cast<uint32_t>(cells),
                                         sizes.data()) ||
            !contract::valid_candidate(views, cells))
        {
            return false;
        }
        const auto destinations = contract::outputs(d);
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(outputs[f], cells, destinations[f]);
        }
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_thompson_moment_sedimentation(const ThompsonMomentSedimentationDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        namespace contract = nimbus::compute::thompson_sedimentation;
        std::size_t cells = 0, columns = 0;
        if (!supports(MicrophysicsKernelOp::ThompsonMomentSedimentation) || !contract::validate(d, cells, columns))
        {
            return false;
        }
        const auto                        constants = contract::constants(d);
        const auto                        inputs    = contract::inputs(d);
        std::array<std::vector<float>, 7> candidates;
        std::array<float*, 7>             outputs;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            candidates[f].resize(cells, 0.0f);
            outputs[f] = candidates[f].data();
        }
        if (!dispatch_multi_field_kernel(pipeline(KernelId::ThompsonMomentSedimentation),
                                         KernelId::ThompsonMomentSedimentation,
                                         &constants,
                                         inputs.data(),
                                         inputs.size(),
                                         outputs.data(),
                                         outputs.size(),
                                         d.nx,
                                         d.ny,
                                         d.nz,
                                         static_cast<uint32_t>(columns)) ||
            !contract::valid_candidate(outputs[0], outputs[1], outputs[2], cells, columns))
        {
            return false;
        }
        std::copy_n(outputs[0], cells, d.mass_tendency);
        std::copy_n(outputs[1], cells, d.number_tendency);
        std::copy_n(outputs[2], columns, d.surface_mass);
        return true;
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_lin_sedimentation(const LinSedimentationDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        std::size_t cells=0,bytes=0;
        if(!pipeline(KernelId::LinSedimentation).is_ready() || !nimbus::compute::lin::valid_lin_sedimentation(d,cells,bytes))return false;
        std::uint32_t work=0;VkDeviceSize checked_bytes=0;
        if(!checked_field_dispatch(d.nx,d.ny,1,work,checked_bytes,"Lin sedimentation"))return false;
        const auto pc=nimbus::compute::lin::lin_sedimentation_constants(d);
        const float* inputs[6]={d.qr,d.qs,d.qg,d.qh,d.rho_d,d.layer_thickness_m?d.layer_thickness_m:d.rho_d};
        float* destinations[4]={d.dqr_dt,d.dqs_dt,d.dqg_dt,d.dqh_dt};
        std::array<std::vector<float>,4> candidate;
        float* outputs[4]={};
        std::array<const float*,4> result{};
        for(std::size_t f=0;f<candidate.size();++f)
        {
            candidate[f].assign(destinations[f],destinations[f]+cells);
            outputs[f]=candidate[f].data();result[f]=outputs[f];
        }
        const char* names[6]={"qr","qs","qg","qh",nullptr,nullptr};

        if(!dispatch_multi_field_kernel(pipeline(KernelId::LinSedimentation),KernelId::LinSedimentation,&pc,inputs,6,outputs,4,d.nx,d.ny,d.nz,work,nullptr,names) ||
            !nimbus::compute::lin::finite_result(d,result))return false;
        for(std::size_t f=0;f<candidate.size();++f)std::copy_n(result[f],cells,destinations[f]);
        return true;
#else
        (void)d;
        return false;
#endif
    }

    // ── Smagorinsky turbulence (GPU audit Phase 3.4) ────────────

    bool supports_smagorinsky_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Pointwise strain + Laplacian physics on uniform spacings:
        // coordinate-agnostic (the dispatch site guards flat terrain and
        // the stability-correction mode).
        return pipeline(KernelId::SmagorinskyCoefficients).is_ready() &&
               pipeline(KernelId::SmagorinskyTendencies).is_ready();
#else
        return false;
#endif
    }

    bool supports_tke_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return pipeline(KernelId::TkeClosure).is_ready() && pipeline(KernelId::TkeTendency).is_ready() &&
               supports_smagorinsky_dispatch() && supports_sgs_apply_dispatch();
#else
        return false;
#endif
    }

#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
    template <class Constants, std::size_t N>
    bool dispatch_tke_fields(KernelId                           id,
                             const Constants&                   pc,
                             const std::array<const float*, 7>& inputs,
                             const std::array<float*, N>&       outputs,
                             int                                nx,
                             int                                ny,
                             int                                nz,
                             std::size_t                        cells)
    {
        uint32_t     points = 0;
        VkDeviceSize bytes  = 0;
        if (!checked_field_dispatch(nx, ny, nz, points, bytes, "TKE"))
            return false;
        std::array<std::vector<float>, N> candidates;
        std::array<float*, N>             destinations;
        for (std::size_t f = 0; f < N; ++f)
        {
            candidates[f].resize(cells);
            destinations[f] = candidates[f].data();
        }
        VkDeviceSize        sizes[7] = {0, 0, 0, 0, 0, 0, static_cast<VkDeviceSize>(nz) * sizeof(float)};
        std::array<bool, N> write_only;
        write_only.fill(true);
        if (!dispatch_multi_field_kernel(pipeline(id),
                                         id,
                                         &pc,
                                         inputs.data(),
                                         7,
                                         destinations.data(),
                                         N,
                                         nx,
                                         ny,
                                         nz,
                                         points,
                                         sizes,
                                         nullptr,
                                         write_only.data()))
            return false;
        for (std::size_t f = 0; f < N; ++f)
            for (float value : candidates[f])
                if (!std::isfinite(value) || (id == KernelId::TkeClosure && f != 5 && value < 0))
                    return false;
        for (std::size_t f = 0; f < N; ++f)
            std::copy(candidates[f].begin(), candidates[f].end(), outputs[f]);
        return true;
    }
#endif

    bool dispatch_tke_closure(const TkeClosureDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        std::size_t cells = 0;
        if (!supports_tke_dispatch() || !nimbus::compute::tke_contract::validate(d, cells))
            return false;
        return dispatch_tke_fields(
            KernelId::TkeClosure,
            nimbus::compute::tke_contract::constants(d),
            std::array<const float*, 7>{d.u, d.v, d.w, d.theta, d.qv, d.tke, d.dz},
            std::array<float*, 8>{
                d.nu, d.K_theta, d.K_q, d.K_tke, d.shear, d.buoyancy, d.dissipation, d.effective_prandtl},
            d.nx,
            d.ny,
            d.nz,
            cells);
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_tke_tendency(const TkeTendencyDispatch& d) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        std::size_t cells = 0;
        if (!supports_tke_dispatch() || !nimbus::compute::tke_contract::validate(d, cells))
            return false;
        return dispatch_tke_fields(
            KernelId::TkeTendency,
            nimbus::compute::tke_contract::constants(d),
            std::array<const float*, 7>{d.rho, d.tke, d.K_tke, d.shear, d.buoyancy, d.dissipation, d.dz},
            std::array<float*, 1>{d.tendency},
            d.nx,
            d.ny,
            d.nz,
            cells);
#else
        (void)d;
        return false;
#endif
    }

    bool dispatch_smagorinsky_coefficients(const SmagorinskyCoefficientsDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        const auto* u_data         = dispatch.u;
        const auto* v_data         = dispatch.v;
        const auto* w_data         = dispatch.w;
        const auto* dz_prof        = dispatch.dz_prof;
        const auto* dzc_prof       = dispatch.dzc_prof;
        const int   prof_len       = dispatch.prof_len;
        auto*       nu_t           = dispatch.nu_t;
        auto*       K_theta        = dispatch.K_theta;
        auto*       K_q            = dispatch.K_q;
        const int   nx             = dispatch.nx;
        const int   ny             = dispatch.ny;
        const int   nz             = dispatch.nz;
        const int   filter_mode    = dispatch.filter_mode;
        const bool  use_horizontal = dispatch.use_horizontal;
        const bool  use_vertical   = dispatch.use_vertical;
        const bool  periodic_x     = dispatch.periodic_x;
        const bool  periodic_y     = dispatch.periodic_y;
        const float dx_val         = dispatch.dx;
        const float dy_val         = dispatch.dy;
        const float Cs             = dispatch.Cs;
        const float Pr_t           = dispatch.Pr_t;
        const float Sc_t           = dispatch.Sc_t;
        const float nu_t_max       = dispatch.nu_t_max;
        const float K_max          = dispatch.K_max;

        if (!pipeline(KernelId::SmagorinskyCoefficients).is_ready())
        {
            return false;
        }
        if (dz_prof == nullptr || dzc_prof == nullptr || prof_len < nz)
        {
            return false;
        }
        SmagorinskyCoefficientsPushConstants pc{};
        pc.nx             = nx;
        pc.ny             = ny;
        pc.nz             = nz;
        pc.filter_mode    = filter_mode;
        pc.use_horizontal = use_horizontal ? 1 : 0;
        pc.use_vertical   = use_vertical ? 1 : 0;
        pc.dx             = dx_val;
        pc.dy             = dy_val;
        pc.Cs             = Cs;
        pc.Pr_t           = Pr_t;
        pc.Sc_t           = Sc_t;
        pc.nu_t_max       = nu_t_max;
        pc.K_max          = K_max;
        pc.periodic_y     = periodic_y ? 1 : 0;
        pc.periodic_x     = periodic_x ? 1 : 0;

        // dz/dzc upload compact (nz floats; shader indexes by k) -
        // Phase 4.3.
        const VkDeviceSize prof_bytes          = static_cast<VkDeviceSize>(nz) * sizeof(float);
        uint32_t           total_points        = 0;
        VkDeviceSize       field_bytes_checked = 0;
        if (!checked_field_dispatch(nx, ny, nz, total_points, field_bytes_checked, "Smagorinsky coefficients"))
            return false;
        const float*       inputs[5]            = {u_data, v_data, w_data, dz_prof, dzc_prof};
        const VkDeviceSize input_bytes[5]       = {0, 0, 0, prof_bytes, prof_bytes};
        float*             outputs[3]           = {nu_t, K_theta, K_q};
        const bool         output_write_only[3] = {true, true, true};

        return dispatch_multi_field_kernel(pipeline(KernelId::SmagorinskyCoefficients),
                                           KernelId::SmagorinskyCoefficients,
                                           &pc,
                                           inputs,
                                           5,
                                           outputs,
                                           3,
                                           nx,
                                           ny,
                                           nz,
                                           total_points,
                                           input_bytes,
                                           nullptr,
                                           output_write_only);
#else
        (void)dispatch;
        return false;
#endif
    }

    bool dispatch_smagorinsky_tendencies(const SmagorinskyTendenciesDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        const auto* u_data         = dispatch.u;
        const auto* v_data         = dispatch.v;
        const auto* w_data         = dispatch.w;
        const auto* theta_data     = dispatch.theta_data;
        const auto* qv_data        = dispatch.qv_data;
        const auto* nu_t           = dispatch.nu_t;
        const auto* K_theta        = dispatch.K_theta;
        const auto* K_q            = dispatch.K_q;
        const auto* dz_prof        = dispatch.dz_prof;
        const auto* rho            = dispatch.rho;
        const int   prof_len       = dispatch.prof_len;
        auto*       dudt_sgs       = dispatch.dudt_sgs;
        auto*       dvdt_sgs       = dispatch.dvdt_sgs;
        auto*       dwdt_sgs       = dispatch.dwdt_sgs;
        auto*       dthetadt_sgs   = dispatch.dthetadt_sgs;
        auto*       dqvdt_sgs      = dispatch.dqvdt_sgs;
        const int   nx             = dispatch.nx;
        const int   ny             = dispatch.ny;
        const int   nz             = dispatch.nz;
        const bool  use_horizontal = dispatch.use_horizontal;
        const bool  use_vertical   = dispatch.use_vertical;
        const bool  has_qv         = dispatch.has_qv;
        const bool  periodic_x     = dispatch.periodic_x;
        const bool  periodic_y     = dispatch.periodic_y;
        const float dx_val         = dispatch.dx;
        const float dy_val         = dispatch.dy;

        if (!pipeline(KernelId::SmagorinskyTendencies).is_ready())
        {
            return false;
        }
        if (dz_prof == nullptr || rho == nullptr || prof_len < nz)
        {
            return false;
        }

        SmagorinskyTendenciesPushConstants pc{};
        pc.nx             = nx;
        pc.ny             = ny;
        pc.nz             = nz;
        pc.use_horizontal = use_horizontal ? 1 : 0;
        pc.use_vertical   = use_vertical ? 1 : 0;
        pc.has_qv         = has_qv ? 1 : 0;
        pc.periodic_x     = periodic_x ? 1 : 0;
        pc.periodic_y     = periodic_y ? 1 : 0;
        pc.dx             = dx_val;
        pc.dy             = dy_val;

        pc.closed_lateral = turbulence_closed_lateral_mask(dispatch.open_lateral);
        // dz is a compact vertical profile; rho is the full live dry-density
        // field. A base-state or unity fallback would conserve the wrong
        // prognostic momentum and scalar-mass currency.
        const VkDeviceSize prof_bytes          = static_cast<VkDeviceSize>(nz) * sizeof(float);
        uint32_t           total_points        = 0;
        VkDeviceSize       field_bytes_checked = 0;
        if (!checked_field_dispatch(nx, ny, nz, total_points, field_bytes_checked, "Smagorinsky tendencies"))
            return false;
        const float*       inputs[10] = {u_data, v_data, w_data, theta_data, qv_data, nu_t, K_theta, K_q, dz_prof, rho};
        const VkDeviceSize input_bytes[10]      = {0, 0, 0, 0, 0, 0, 0, 0, prof_bytes, 0};
        float*             outputs[5]           = {dudt_sgs, dvdt_sgs, dwdt_sgs, dthetadt_sgs, dqvdt_sgs};
        const bool         output_write_only[5] = {true, true, true, true, true};

        return dispatch_multi_field_kernel(pipeline(KernelId::SmagorinskyTendencies),
                                           KernelId::SmagorinskyTendencies,
                                           &pc,
                                           inputs,
                                           10,
                                           outputs,
                                           5,
                                           nx,
                                           ny,
                                           nz,
                                           total_points,
                                           input_bytes,
                                           nullptr,
                                           output_write_only);
#else
        (void)dispatch;
        return false;
#endif
    }

    // ── Fused Smagorinsky chain (GPU audit Phase 4.3c) ──────────
    //
    // Coefficients + tendencies in ONE submission with one fence; the
    // eddy coefficient fields live on device between the two passes
    // (uploaded never, downloaded once for diagnostics and the parity
    // replay), and u/v/w upload once instead of twice.

    bool dispatch_smagorinsky_fused(const SmagorinskyFusedDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        const auto* u_data                = dispatch.u_data;
        const auto* v_data                = dispatch.v_data;
        const auto* w_data                = dispatch.w_data;
        const auto* theta_data            = dispatch.theta_data;
        const auto* qv_data               = dispatch.qv_data;
        const auto* dz_prof               = dispatch.dz_prof;
        const auto* dzc_prof              = dispatch.dzc_prof;
        const auto* rho                   = dispatch.rho;
        const int   prof_len              = dispatch.prof_len;
        auto*       nu_t                  = dispatch.nu_t;
        auto*       K_theta               = dispatch.K_theta;
        auto*       K_q                   = dispatch.K_q;
        auto*       dudt_sgs              = dispatch.dudt_sgs;
        auto*       dvdt_sgs              = dispatch.dvdt_sgs;
        auto*       dwdt_sgs              = dispatch.dwdt_sgs;
        auto*       dthetadt_sgs          = dispatch.dthetadt_sgs;
        auto*       dqvdt_sgs             = dispatch.dqvdt_sgs;
        const int   nx                    = dispatch.nx;
        const int   ny                    = dispatch.ny;
        const int   nz                    = dispatch.nz;
        const int   filter_mode           = dispatch.filter_mode;
        const bool  use_horizontal        = dispatch.use_horizontal;
        const bool  use_vertical          = dispatch.use_vertical;
        const bool  has_qv                = dispatch.has_qv;
        const bool  periodic_x            = dispatch.periodic_x;
        const bool  periodic_y            = dispatch.periodic_y;
        const float dx_val                = dispatch.dx_val;
        const float dy_val                = dispatch.dy_val;
        const float Cs                    = dispatch.Cs;
        const float Pr_t                  = dispatch.Pr_t;
        const float Sc_t                  = dispatch.Sc_t;
        const float nu_t_max              = dispatch.nu_t_max;
        const float K_max                 = dispatch.K_max;
        const bool  download_coefficients = dispatch.download_coefficients;

        if (!pipeline(KernelId::SmagorinskyCoefficients).is_ready() ||
            !pipeline(KernelId::SmagorinskyTendencies).is_ready())
        {
            return false;
        }
        if (dz_prof == nullptr || dzc_prof == nullptr || rho == nullptr || prof_len < nz)
        {
            return false;
        }
        if (download_coefficients && (nu_t == nullptr || K_theta == nullptr || K_q == nullptr))
        {
            return false;
        }

        uint32_t     total_cells = 0;
        VkDeviceSize field_bytes = 0;
        if (!checked_field_dispatch(nx, ny, nz, total_cells, field_bytes, "Smagorinsky"))
            return false;
        const VkDeviceSize prof_bytes      = static_cast<VkDeviceSize>(nz) * sizeof(float);
        const uint32_t     workgroup_count = 1u + (total_cells - 1u) / 64u;

        // Slots: u(0) v(1) w(2) dz(3) dzc(4) nu_t(5) K_theta(6) K_q(7)
        // theta(8) qv(9) tendencies(10-14), live rho(15). u/v/w and the five
        // tendency outputs bind pinned named
        // slots (GPU audit Phase 4.3d): when the thermodynamics
        // transport batch stored them earlier this step they reuse
        // WITHOUT re-upload - the step graph performs no host write to
        // the winds between that batch and this dispatch (microphysics
        // and the bounds sweeps touch theta/q/rho/p only; the SGS
        // integrate runs after this). The parity replay cross-checks the
        // assertion at sample cadence. When no store happened this step,
        // this dispatch uploads into the named slots and becomes the
        // store.
        const bool reuse_uvw[3] = {
            named_slot_reusable("u", u_data, field_bytes),
            named_slot_reusable("v", v_data, field_bytes),
            named_slot_reusable("w", w_data, field_bytes),
        };

        constexpr int kSlotCount = 16;
        int           slots[kSlotCount];
        for (int& slot : slots)
        {
            slot = -1;
        }
        static const char* const kSgsTendencyNames[5] = {
            "sgs_dudt", "sgs_dvdt", "sgs_dwdt", "sgs_dthetadt", "sgs_dqvdt"};
        float* const             tendency_outputs[5]  = {dudt_sgs, dvdt_sgs, dwdt_sgs, dthetadt_sgs, dqvdt_sgs};
        static const char* const kSmagorinskyNames[8] = {
            "u", "v", "w", "sgs_dudt", "sgs_dvdt", "sgs_dwdt", "sgs_dthetadt", "sgs_dqvdt"};
        VkDeviceSize smagorinsky_sizes[8];
        for (VkDeviceSize& size : smagorinsky_sizes)
            size = field_bytes;
        int smagorinsky_named_slots[8];
        if (!named_slots_acquire_many(kSmagorinskyNames, smagorinsky_sizes, 8, smagorinsky_named_slots))
        {
            log_vulkan_warning("failed to pin named slots for fused smagorinsky");
            return false;
        }
        slots[0] = smagorinsky_named_slots[0];
        slots[1] = smagorinsky_named_slots[1];
        slots[2] = smagorinsky_named_slots[2];
        for (int s = 0; s < 5; ++s)
            slots[10 + s] = smagorinsky_named_slots[3 + s];
        // Acquire every transient in one exact-size transaction. dz/dzc are
        // compact vertical profiles; the remaining fields span the volume.
        constexpr int      kTransientCount                    = 8;
        const int          transient_logical[kTransientCount] = {3, 4, 5, 6, 7, 8, 9, 15};
        const VkDeviceSize transient_sizes[kTransientCount]   = {
            prof_bytes, prof_bytes, field_bytes, field_bytes, field_bytes, field_bytes, field_bytes, field_bytes};
        int transient_slots[kTransientCount];
        if (!buffer_pool_.acquire_many(transient_sizes, kTransientCount, transient_slots))
        {
            log_vulkan_warning("failed to acquire transient slots for fused smagorinsky");
            return false;
        }
        for (int s = 0; s < kTransientCount; ++s)
            slots[transient_logical[s]] = transient_slots[s];
        const auto release_transient_slots     = [&]() { release_pool_slots(transient_slots, kTransientCount); };
        static const char* const kWindNames[3] = {"u", "v", "w"};
        for (int s = 0; s < 3; ++s)
        {
            if (!reuse_uvw[s])
            {
                named_slot_invalidate(kWindNames[s]);
            }
        }
        for (const char* name : kSgsTendencyNames)
        {
            // Full-field outputs supersede any prior contents. Keep them
            // invalid until the producer fence and host publication finish.
            named_slot_invalidate(name);
        }

        const bool unified  = device_context_.has_unified_memory;
        auto       host_ptr = [&](int slot) -> void*
        { return unified ? buffer_pool_.device(slot).mapped : buffer_pool_.staging(slot).mapped; };

        const float* wind_inputs[3] = {u_data, v_data, w_data};
        for (int s = 0; s < 3; ++s)
        {
            if (!reuse_uvw[s])
            {
                tracked_memcpy(host_ptr(slots[s]), wind_inputs[s], field_bytes);
            }
        }
        tracked_memcpy(host_ptr(slots[3]), dz_prof, prof_bytes);
        tracked_memcpy(host_ptr(slots[4]), dzc_prof, prof_bytes);
        tracked_memcpy(host_ptr(slots[8]), theta_data, field_bytes);
        tracked_memcpy(host_ptr(slots[9]), qv_data, field_bytes);
        tracked_memcpy(host_ptr(slots[15]), rho, field_bytes);

        int coeff_bindings[8] = {slots[0], slots[1], slots[2], slots[3], slots[4], slots[5], slots[6], slots[7]};
        update_pooled_descriptor_set(pipeline(KernelId::SmagorinskyCoefficients), coeff_bindings, field_bytes, 8);
        int tend_bindings[15] = {slots[0],
                                 slots[1],
                                 slots[2],
                                 slots[8],
                                 slots[9],
                                 slots[5],
                                 slots[6],
                                 slots[7],
                                 slots[3],
                                 slots[15],
                                 slots[10],
                                 slots[11],
                                 slots[12],
                                 slots[13],
                                 slots[14]};
        update_pooled_descriptor_set(pipeline(KernelId::SmagorinskyTendencies), tend_bindings, field_bytes, 15);

        vkResetCommandBuffer_(cmd_buf_, 0);
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (begin_profiled_command_buffer(begin_info) != VK_SUCCESS)
        {
            release_transient_slots();
            return false;
        }

        const VkPipelineStageFlags both_stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkMemoryBarrier            barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;

        // ── H2D ──
        if (unified)
        {
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(
                cmd_buf_, VK_PIPELINE_STAGE_HOST_BIT, both_stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
        else
        {
            const int upload_slots[8] = {
                slots[0], slots[1], slots[2], slots[3], slots[4], slots[8], slots[9], slots[15]};
            for (int s = 0; s < 8; ++s)
            {
                if (s < 3 && reuse_uvw[s])
                {
                    continue; // named u/v/w already device-resident
                }
                VkBufferCopy copy_region{};
                copy_region.size = (s == 3 || s == 4) ? prof_bytes : field_bytes;
                tracked_cmd_copy_buffer(buffer_pool_.staging(upload_slots[s]).buffer,
                                        buffer_pool_.device(upload_slots[s]).buffer,
                                        1,
                                        &copy_region);
            }
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_, both_stages, both_stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        }

        // ── Pass 1: coefficients (reads u,v,w,dz,dzc; writes nu_t,K_*) ──
        SmagorinskyCoefficientsPushConstants coeff_pc{};
        coeff_pc.nx             = nx;
        coeff_pc.ny             = ny;
        coeff_pc.nz             = nz;
        coeff_pc.filter_mode    = filter_mode;
        coeff_pc.use_horizontal = use_horizontal ? 1 : 0;
        coeff_pc.use_vertical   = use_vertical ? 1 : 0;
        coeff_pc.dx             = dx_val;
        coeff_pc.dy             = dy_val;
        coeff_pc.Cs             = Cs;
        coeff_pc.Pr_t           = Pr_t;
        coeff_pc.Sc_t           = Sc_t;
        coeff_pc.nu_t_max       = nu_t_max;
        coeff_pc.K_max          = K_max;
        coeff_pc.periodic_y     = periodic_y ? 1 : 0;
        coeff_pc.periodic_x     = periodic_x ? 1 : 0;

        vkCmdBindPipeline_(
            cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline(KernelId::SmagorinskyCoefficients).pipeline);
        vkCmdBindDescriptorSets_(cmd_buf_,
                                 VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipeline(KernelId::SmagorinskyCoefficients).pipeline_layout,
                                 0,
                                 1,
                                 &pipeline(KernelId::SmagorinskyCoefficients).descriptor_set,
                                 0,
                                 nullptr);
        vkCmdPushConstants_(cmd_buf_,
                            pipeline(KernelId::SmagorinskyCoefficients).pipeline_layout,
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0,
                            sizeof(coeff_pc),
                            &coeff_pc);
        record_linear_group_dispatch(workgroup_count, KernelId::SmagorinskyCoefficients);

        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier_(cmd_buf_, both_stages, both_stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // ── Pass 2: tendencies (reads u,v,w,theta,qv,nu_t,K_*,dz) ──
        SmagorinskyTendenciesPushConstants tend_pc{};
        tend_pc.nx             = nx;
        tend_pc.ny             = ny;
        tend_pc.nz             = nz;
        tend_pc.use_horizontal = use_horizontal ? 1 : 0;
        tend_pc.use_vertical   = use_vertical ? 1 : 0;
        tend_pc.has_qv         = has_qv ? 1 : 0;
        tend_pc.periodic_x     = periodic_x ? 1 : 0;
        tend_pc.periodic_y     = periodic_y ? 1 : 0;
        tend_pc.dx             = dx_val;
        tend_pc.dy             = dy_val;
        tend_pc.closed_lateral = turbulence_closed_lateral_mask(dispatch.open_lateral);

        vkCmdBindPipeline_(
            cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline(KernelId::SmagorinskyTendencies).pipeline);
        vkCmdBindDescriptorSets_(cmd_buf_,
                                 VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipeline(KernelId::SmagorinskyTendencies).pipeline_layout,
                                 0,
                                 1,
                                 &pipeline(KernelId::SmagorinskyTendencies).descriptor_set,
                                 0,
                                 nullptr);
        vkCmdPushConstants_(cmd_buf_,
                            pipeline(KernelId::SmagorinskyTendencies).pipeline_layout,
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0,
                            sizeof(tend_pc),
                            &tend_pc);
        record_linear_group_dispatch(workgroup_count, KernelId::SmagorinskyTendencies);

        // ── D2H: tendencies plus optional diagnostic/parity coefficients ──
        if (unified)
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier_(
                cmd_buf_, both_stages, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
        else
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_, both_stages, both_stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
            VkBufferCopy copy_region{};
            copy_region.size = field_bytes;
            if (download_coefficients)
            {
                for (int s = 5; s <= 7; ++s)
                {
                    tracked_cmd_copy_buffer(
                        buffer_pool_.device(slots[s]).buffer, buffer_pool_.staging(slots[s]).buffer, 1, &copy_region);
                }
            }
            for (int s = 10; s <= 14; ++s)
            {
                tracked_cmd_copy_buffer(
                    buffer_pool_.device(slots[s]).buffer, buffer_pool_.staging(slots[s]).buffer, 1, &copy_region);
            }
        }

        if (end_profiled_command_buffer() != VK_SUCCESS)
        {
            release_transient_slots();
            return false;
        }

        VkSubmitInfo submit_info{};
        submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers    = &cmd_buf_;
        if (reset_fence_and_submit("smagorinsky_fused", submit_info) != VK_SUCCESS)
        {
            release_transient_slots();
            return false;
        }

        if (timed_wait_fence("smagorinsky_fused") != VK_SUCCESS)
        {
            release_transient_slots();
            return false;
        }

        for (int s = 0; s < 3; ++s)
        {
            if (!reuse_uvw[s])
            {
                // This dispatch was the step's store site for this wind
                // (read-only in both passes above).
                named_slot_commit(kWindNames[s], wind_inputs[s], field_bytes);
            }
        }
        if (download_coefficients)
        {
            tracked_memcpy(nu_t, host_ptr(slots[5]), field_bytes);
            tracked_memcpy(K_theta, host_ptr(slots[6]), field_bytes);
            tracked_memcpy(K_q, host_ptr(slots[7]), field_bytes);
        }
        tracked_memcpy(dudt_sgs, host_ptr(slots[10]), field_bytes);
        tracked_memcpy(dvdt_sgs, host_ptr(slots[11]), field_bytes);
        tracked_memcpy(dwdt_sgs, host_ptr(slots[12]), field_bytes);
        tracked_memcpy(dthetadt_sgs, host_ptr(slots[13]), field_bytes);
        tracked_memcpy(dqvdt_sgs, host_ptr(slots[14]), field_bytes);
        for (int s = 0; s < 5; ++s)
        {
            named_slot_commit(kSgsTendencyNames[s], tendency_outputs[s], field_bytes);
        }

        release_transient_slots();
        return true;
#else
        (void)dispatch;
        return false;
#endif
    }

    // ── Fused state relaxation (GPU audit Phase 3.5) ────────────

    bool supports_state_relaxation_cartesian_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Cartesian-only (the lateral sponge's nearest-face distance form;
        // GPU.md section 2.0 routing rule).
        if (coord_system_ != CoordinateSystem::Cartesian)
            return false;
        return pipeline(KernelId::StateRelaxationCartesian).is_ready();
#else
        return false;
#endif
    }

    bool dispatch_state_relaxation_cartesian(const StateRelaxationDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        auto*       u             = dispatch.u;
        auto*       v             = dispatch.v;
        auto*       w             = dispatch.w;
        auto*       theta_data    = dispatch.theta_data;
        const auto* profiles      = dispatch.profiles;
        const int   profiles_len  = dispatch.profiles_len;
        const int   nx            = dispatch.nx;
        const int   ny            = dispatch.ny;
        const int   nz            = dispatch.nz;
        const int   lateral_width = dispatch.lateral_width;
        const bool  do_upper      = dispatch.do_upper;
        const bool  do_w_only     = dispatch.do_w_only;
        const bool  do_lateral    = dispatch.do_lateral;
        const float dt_step       = dispatch.dt_step;

        // Cartesian-only; see supports_state_relaxation_cartesian_dispatch.
        if (coord_system_ != CoordinateSystem::Cartesian)
        {
            log_vulkan_warning("state relaxation dispatch refused: shader is "
                               "cartesian-only and the active coordinate "
                               "system is not cartesian");
            return false;
        }
        if (!pipeline(KernelId::StateRelaxationCartesian).is_ready())
        {
            return false;
        }
        if (!std::isfinite(dt_step) || dt_step <= 0.0f || lateral_width < 0 || lateral_width > nz)
        {
            return false;
        }
        if (profiles == nullptr || profiles_len < 7 * nz)
        {
            return false;
        }

        StateRelaxationPushConstants pc{};
        pc.nx            = nx;
        pc.ny            = ny;
        pc.nz            = nz;
        pc.lateral_width = lateral_width;
        pc.reserved_zero = 0;
        pc.do_upper      = do_upper ? 1 : 0;
        pc.do_w_only     = do_w_only ? 1 : 0;
        pc.do_lateral    = do_lateral ? 1 : 0;
        pc.dt_step       = dt_step;
        pc.reserved1     = 0.0f;
        pc.reserved2     = 0.0f;
        pc.reserved3     = 0.0f;

        // The packed 7-row profile block uploads compact (the shader
        // already indexes prof[row*nz + k]) - Phase 4.3. The slot itself
        // stays field-sized; only the upload shrinks.
        const VkDeviceSize prof_bytes          = static_cast<VkDeviceSize>(7 * nz) * sizeof(float);
        uint32_t           total_points        = 0;
        VkDeviceSize       field_bytes_checked = 0;
        if (!checked_field_dispatch(nx, ny, nz, total_points, field_bytes_checked, "state relaxation"))
            return false;
        const float*       inputs[5]            = {u, v, w, theta_data, profiles};
        const VkDeviceSize input_bytes[5]       = {0, 0, 0, 0, prof_bytes};
        float*             outputs[4]           = {u, v, w, theta_data};
        const bool         output_write_only[4] = {true, true, true, true};

        return dispatch_multi_field_kernel(pipeline(KernelId::StateRelaxationCartesian),
                                           KernelId::StateRelaxationCartesian,
                                           &pc,
                                           inputs,
                                           5,
                                           outputs,
                                           4,
                                           nx,
                                           ny,
                                           nz,
                                           total_points,
                                           input_bytes,
                                           nullptr,
                                           output_write_only,
                                           true,
                                           nullptr,
                                           dispatch.completed_observer,
                                           dispatch.completed_observer_context);
#else
        (void)dispatch;
        return false;
#endif
    }

    // ── Named-slot residency step boundary (GPU audit 4.3d) ─────

    void begin_resident_step() override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        compute_telemetry_.memory_window.boundary(
            compute_telemetry_.model_steps + 1, compute_telemetry_.memory_counters(),
            compute_telemetry_.memory_accounting_valid, std::cout, memory_window_process_id());
        compute_telemetry_.note_model_step();
        active_resident_fields().begin_step();
#endif
    }

    void select_residency_scope(ComputeResidencyScope scope) noexcept override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        active_residency_scope_ = scope;
#else
        (void)scope;
#endif
    }

    void invalidate_resident_field(const char* field_id) noexcept override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (field_id != nullptr)
            active_resident_fields().note_host_write(field_id);
#else
        (void)field_id;
#endif
    }

    void invalidate_all_resident_fields() noexcept override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        invalidate_residency_scope(active_residency_scope_);
#endif
    }

    void invalidate_residency_scope(ComputeResidencyScope scope) noexcept override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        const std::size_t invalidated = resident_field_scopes_[residency_scope_index(scope)].invalidate_all();
        compute_telemetry_.note_resident_full_invalidation(static_cast<std::uint64_t>(invalidated));
#else
        (void)scope;
#endif
    }

    bool supports_homogeneous_domain_field_shift() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return coord_system_ == CoordinateSystem::Cartesian &&
               pipeline(KernelId::DomainRecyclingHomogeneousFieldShift).is_ready();
#else
        return false;
#endif
    }

    bool supports_earth_control_evolution() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return pipeline(KernelId::EarthControlEvolution).is_ready();
#else
        return false;
#endif
    }

    bool dispatch_earth_control_evolution(
        const EarthControlEvolutionDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!supports_earth_control_evolution() || dispatch.previous == nullptr ||
            dispatch.destination == nullptr || dispatch.control_i_count <= 0 ||
            dispatch.control_j_count <= 0 || dispatch.nz <= 0 ||
            !std::isfinite(dispatch.rho) || dispatch.rho < 0.0f || dispatch.rho > 1.0f ||
            static_cast<std::uint32_t>(dispatch.mode) >
                static_cast<std::uint32_t>(EarthControlEvolutionMode::Reconstruct))
            return false;

        std::size_t cells = 0;
        VkDeviceSize bytes = 0;
        std::string extent_error;
        if (!tmv::vulkan_dispatch::checked_extent(
                dispatch.control_i_count, dispatch.control_j_count,
                dispatch.nz, sizeof(float), cells, bytes, extent_error) ||
            cells > std::numeric_limits<std::uint32_t>::max())
            return false;

        EarthControlEvolutionPushConstants constants{};
        constants.seed = dispatch.seed;
        std::memcpy(&constants.member_bits, &dispatch.member_id,
                    sizeof(constants.member_bits));
        constants.initial_process_key = dispatch.initial_process_key;
        constants.innovation_process_key = dispatch.innovation_process_key;
        constants.innovation_epoch = dispatch.innovation_epoch;
        constants.control_i_min = dispatch.control_i_min;
        constants.control_j_min = dispatch.control_j_min;
        constants.control_i_count = dispatch.control_i_count;
        constants.control_j_count = dispatch.control_j_count;
        constants.nz = dispatch.nz;
        constants.rho = static_cast<float>(dispatch.rho);
        constants.rho_low = static_cast<float>(dispatch.rho - constants.rho);
        constants.mode = static_cast<std::uint32_t>(dispatch.mode);

        const float* inputs[1] = {dispatch.previous};
        float* outputs[1] = {dispatch.destination};
        const bool output_write_only[1] = {true};
        return dispatch_multi_field_kernel(
            pipeline(KernelId::EarthControlEvolution),
            KernelId::EarthControlEvolution, &constants, inputs, 1, outputs, 1,
            dispatch.control_i_count, dispatch.control_j_count, dispatch.nz,
            static_cast<std::uint32_t>(cells), nullptr, nullptr,
            output_write_only);
#else
        (void)dispatch;
        return false;
#endif
    }

    bool dispatch_homogeneous_domain_field_shift(const HomogeneousDomainFieldShiftDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!supports_homogeneous_domain_field_shift() || !nimbus::compute::domain_shift::valid_region(dispatch))
        {
            log_vulkan_warning("homogeneous domain-field shift refused: invalid canonical dispatch");
            return false;
        }

        std::size_t  total_cells = 0;
        VkDeviceSize field_bytes = 0;
        std::string  extent_error;
        if (!tmv::vulkan_dispatch::checked_extent(
                dispatch.nr, dispatch.nth, dispatch.nz, sizeof(float), total_cells, field_bytes, extent_error) ||
            total_cells > std::numeric_limits<std::uint32_t>::max())
        {
            log_vulkan_warning("homogeneous domain-field shift refused: " + extent_error);
            return false;
        }

        HomogeneousDomainFieldShiftPushConstants pc{};
        pc.nr                     = dispatch.nr;
        pc.nth                    = dispatch.nth;
        pc.nz                     = dispatch.nz;
        pc.shift_i_cells          = dispatch.shift_i_cells;
        pc.shift_j_cells          = dispatch.shift_j_cells;
        pc.destination_i_begin    = dispatch.destination_i_begin;
        pc.destination_i_end      = dispatch.destination_i_end;
        pc.destination_j_begin    = dispatch.destination_j_begin;
        pc.destination_j_end      = dispatch.destination_j_end;
        pc.leading_profile_offset = 0;
        pc.leading_is_field = dispatch.leading_field != nullptr;

        const float*       inputs[2]            = {dispatch.source, nimbus::compute::domain_shift::leading_values(dispatch)};
        const VkDeviceSize input_bytes[2]       = {field_bytes, static_cast<VkDeviceSize>(nimbus::compute::domain_shift::leading_count(dispatch, total_cells)) * sizeof(float)};
        float*             outputs[1]           = {dispatch.destination};
        const bool         output_write_only[1] = {true};
        return dispatch_multi_field_kernel(pipeline(KernelId::DomainRecyclingHomogeneousFieldShift),
                                           KernelId::DomainRecyclingHomogeneousFieldShift,
                                           &pc,
                                           inputs,
                                           2,
                                           outputs,
                                           1,
                                           dispatch.nr,
                                           dispatch.nth,
                                           dispatch.nz,
                                           static_cast<std::uint32_t>(total_cells),
                                           input_bytes,
                                           nullptr,
                                           output_write_only);
#else
        (void)dispatch;
        return false;
#endif
    }

    bool dispatch_homogeneous_domain_field_shift_batch(const HomogeneousDomainFieldShiftBatchDispatch& batch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        constexpr int         kMaximumFields = 63;
        ComputePipelineState& shift_pipeline = pipeline(KernelId::DomainRecyclingHomogeneousFieldShift);
        if (!supports_homogeneous_domain_field_shift() || batch.fields == nullptr || batch.resident_ids == nullptr ||
            batch.count < 1 || batch.count > kMaximumFields ||
            shift_pipeline.descriptor_set_count < static_cast<std::uint32_t>(batch.count + 1))
        {
            return false;
        }

        const auto valid_field = [](const HomogeneousDomainFieldShiftDispatch& d)
        {
            return nimbus::compute::domain_shift::valid_region(d);
        };

        const auto& first = batch.fields[0];
        if (!valid_field(first))
            return false;
        for (int field = 0; field < batch.count; ++field)
        {
            const auto& d  = batch.fields[field];
            const char* id = batch.resident_ids[field];
            if (!valid_field(d) || id == nullptr || id[0] == '\0' || d.nr != first.nr || d.nth != first.nth ||
                d.nz != first.nz || d.shift_i_cells != first.shift_i_cells || d.shift_j_cells != first.shift_j_cells)
            {
                return false;
            }
            for (int prior = 0; prior < field; ++prior)
            {
                const auto& p = batch.fields[prior];
                if (std::strcmp(id, batch.resident_ids[prior]) == 0 || d.source == p.source ||
                    d.destination == p.destination || d.source == p.destination || d.destination == p.source)
                {
                    log_vulkan_warning("homogeneous D5 batch refused: owners must be distinct");
                    return false;
                }
            }
        }

        std::size_t  total_cells = 0;
        VkDeviceSize field_bytes = 0;
        std::string  extent_error;
        if (!tmv::vulkan_dispatch::checked_extent(
                first.nr, first.nth, first.nz, sizeof(float), total_cells, field_bytes, extent_error) ||
            total_cells > std::numeric_limits<std::uint32_t>::max() ||
            field_bytes > device_context_.selected_device_properties.limits.maxStorageBufferRange)
        {
            log_vulkan_warning("homogeneous D5 batch refused: " + extent_error);
            return false;
        }
        std::array<std::int32_t, kMaximumFields> profile_offsets{};
        std::uint64_t profile_values = 0;
        for (int field = 0; field < batch.count; ++field)
        {
            profile_offsets[field] = static_cast<std::int32_t>(profile_values);
            profile_values += nimbus::compute::domain_shift::leading_count(batch.fields[field], total_cells);
            if (profile_values > std::uint64_t(std::numeric_limits<std::int32_t>::max()))
                return false;
        }
        if (profile_values == 0 || profile_values > static_cast<std::uint64_t>(std::numeric_limits<int32_t>::max()) ||
            profile_values > std::numeric_limits<VkDeviceSize>::max() / sizeof(float))
        {
            return false;
        }
        const VkDeviceSize profile_bytes = static_cast<VkDeviceSize>(profile_values * sizeof(float));
        if (profile_bytes > device_context_.selected_device_properties.limits.maxStorageBufferRange)
        {
            return false;
        }

        std::array<VkDeviceSize, kMaximumFields> named_sizes{};
        std::array<int, kMaximumFields>          named_slots{};
        for (int field = 0; field < batch.count; ++field)
            named_sizes[static_cast<std::size_t>(field)] = field_bytes;
        if (!named_slots_acquire_many(batch.resident_ids, named_sizes.data(), batch.count, named_slots.data()))
        {
            return false;
        }

        const VkDeviceSize transient_sizes[2] = {profile_bytes, field_bytes};
        int                transient_slots[2] = {-1, -1};
        if (!buffer_pool_.acquire_many(transient_sizes, 2, transient_slots))
        {
            return false;
        }
        const int  profile_slot       = transient_slots[0];
        const int  scratch_slot       = transient_slots[1];
        const auto release_transients = [&]()
        {
            if (submission_lifecycle_.may_release_transients())
                release_pool_slots(transient_slots, 2);
        };

        const bool unified    = device_context_.has_unified_memory;
        const auto upload_ptr = [&](int slot) -> void*
        { return unified ? buffer_pool_.device(slot).mapped : buffer_pool_.staging(slot).mapped; };
        auto* packed_profiles = static_cast<float*>(upload_ptr(profile_slot));
        for (int field = 0; field < batch.count; ++field)
        {
            tracked_memcpy(packed_profiles + profile_offsets[field],
                           nimbus::compute::domain_shift::leading_values(batch.fields[field]),
                           nimbus::compute::domain_shift::leading_count(batch.fields[field], total_cells) * sizeof(float));
        }

        std::array<bool, kMaximumFields> reusable{};
        for (int field = 0; field < batch.count; ++field)
        {
            reusable[static_cast<std::size_t>(field)] =
                named_slot_reusable(batch.resident_ids[field], batch.fields[field].source, field_bytes);
            if (!reusable[static_cast<std::size_t>(field)])
            {
                named_slot_invalidate(batch.resident_ids[field]);
                tracked_memcpy(
                    upload_ptr(named_slots[static_cast<std::size_t>(field)]), batch.fields[field].source, field_bytes);
            }
        }
        compute_telemetry_.note_writeonly_output_upload_avoided(static_cast<std::uint64_t>(field_bytes) * batch.count);

        // Each immutable descriptor set binds one resident source, the shared
        // packed profile buffer, and the one high-water scratch destination.
        for (int field = 0; field < batch.count; ++field)
        {
            const int          slots[3]  = {named_slots[static_cast<std::size_t>(field)], profile_slot, scratch_slot};
            const VkDeviceSize ranges[3] = {field_bytes, profile_bytes, field_bytes};
            std::uintptr_t     handles[3]{};
            std::size_t        cached_ranges[3]{};
            for (int binding = 0; binding < 3; ++binding)
            {
                const VkBuffer buffer = buffer_pool_.device(slots[binding]).buffer;
                static_assert(sizeof(buffer) <= sizeof(std::uintptr_t));
                std::memcpy(&handles[binding], &buffer, sizeof(buffer));
                cached_ranges[binding] = static_cast<std::size_t>(ranges[binding]);
            }
            auto&      cache  = shift_pipeline.descriptor_set_caches[static_cast<std::size_t>(field + 1)];
            const bool cached = cache.matches_or_update(handles, cached_ranges, 3);
            compute_telemetry_.note_descriptor_lookup(cached);
            if (cached)
                continue;

            VkDescriptorBufferInfo infos[3]{};
            VkWriteDescriptorSet   writes[3]{};
            for (int binding = 0; binding < 3; ++binding)
            {
                infos[binding]                  = {buffer_pool_.device(slots[binding]).buffer, 0, ranges[binding]};
                writes[binding].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[binding].dstSet          = shift_pipeline.descriptor_sets[static_cast<std::size_t>(field + 1)];
                writes[binding].dstBinding      = static_cast<std::uint32_t>(binding);
                writes[binding].descriptorCount = 1;
                writes[binding].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[binding].pBufferInfo     = &infos[binding];
            }
            vkUpdateDescriptorSets_(device_context_.device, 3, writes, 0, nullptr);
        }

        vkResetCommandBuffer_(cmd_buf_, 0);
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (begin_profiled_command_buffer(begin_info) != VK_SUCCESS)
        {
            release_transients();
            return false;
        }

        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        if (unified)
        {
            barrier.srcAccessMask =
                VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }
        else
        {
            VkBufferCopy profile_copy{};
            profile_copy.size = profile_bytes;
            tracked_cmd_copy_buffer(
                buffer_pool_.staging(profile_slot).buffer, buffer_pool_.device(profile_slot).buffer, 1, &profile_copy);
            for (int field = 0; field < batch.count; ++field)
            {
                if (reusable[static_cast<std::size_t>(field)])
                    continue;
                VkBufferCopy field_copy{};
                field_copy.size = field_bytes;
                const int slot  = named_slots[static_cast<std::size_t>(field)];
                tracked_cmd_copy_buffer(
                    buffer_pool_.staging(slot).buffer, buffer_pool_.device(slot).buffer, 1, &field_copy);
            }
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }

        vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, shift_pipeline.pipeline);
        const std::uint32_t workgroups = 1u + (static_cast<std::uint32_t>(total_cells) - 1u) / 64u;
        for (int field = 0; field < batch.count; ++field)
        {
            const VkDescriptorSet descriptor_set = shift_pipeline.descriptor_sets[static_cast<std::size_t>(field + 1)];
            vkCmdBindDescriptorSets_(cmd_buf_,
                                     VK_PIPELINE_BIND_POINT_COMPUTE,
                                     shift_pipeline.pipeline_layout,
                                     0,
                                     1,
                                     &descriptor_set,
                                     0,
                                     nullptr);

            const auto&                              d = batch.fields[field];
            HomogeneousDomainFieldShiftPushConstants pc{};
            pc.nr                     = d.nr;
            pc.nth                    = d.nth;
            pc.nz                     = d.nz;
            pc.shift_i_cells          = d.shift_i_cells;
            pc.shift_j_cells          = d.shift_j_cells;
            pc.destination_i_begin    = d.destination_i_begin;
            pc.destination_i_end      = d.destination_i_end;
            pc.destination_j_begin    = d.destination_j_begin;
            pc.destination_j_end      = d.destination_j_end;
            pc.leading_profile_offset = profile_offsets[field];
            pc.leading_is_field = d.leading_field != nullptr;
            vkCmdPushConstants_(cmd_buf_,
                                shift_pipeline.pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT,
                                0,
                                shift_pipeline.push_constant_size,
                                &pc);
            record_linear_group_dispatch(workgroups, KernelId::DomainRecyclingHomogeneousFieldShift);

            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
            VkBufferCopy copy{};
            copy.size = field_bytes;
            tracked_cmd_copy_buffer(buffer_pool_.device(scratch_slot).buffer,
                                    buffer_pool_.device(named_slots[static_cast<std::size_t>(field)]).buffer,
                                    1,
                                    &copy);
            if (field + 1 < batch.count)
            {
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier_(cmd_buf_,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                      0,
                                      1,
                                      &barrier,
                                      0,
                                      nullptr,
                                      0,
                                      nullptr);
            }
        }

        if (unified)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }
        else
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
            for (int field = 0; field < batch.count; ++field)
            {
                VkBufferCopy copy{};
                copy.size      = field_bytes;
                const int slot = named_slots[static_cast<std::size_t>(field)];
                tracked_cmd_copy_buffer(buffer_pool_.device(slot).buffer, buffer_pool_.staging(slot).buffer, 1, &copy);
            }
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }

        if (end_profiled_command_buffer() != VK_SUCCESS)
        {
            release_transients();
            return false;
        }
        VkSubmitInfo submit_info{};
        submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers    = &cmd_buf_;
        constexpr const char* kLabel   = "domain_recycling.homogeneous_field_shift_batch";
        if (reset_fence_and_submit(kLabel, submit_info) != VK_SUCCESS || timed_wait_fence(kLabel) != VK_SUCCESS)
        {
            release_transients();
            return false;
        }

        const nimbus::compute::FieldExtent extent{total_cells, sizeof(float)};
        for (int field = 0; field < batch.count; ++field)
        {
            const int   slot   = named_slots[static_cast<std::size_t>(field)];
            const void* source = unified ? buffer_pool_.device(slot).mapped : buffer_pool_.staging(slot).mapped;
            active_resident_fields().note_device_write(
                batch.resident_ids[field], batch.fields[field].destination, extent);
            tracked_memcpy(batch.fields[field].destination, source, field_bytes);
            active_resident_fields().note_host_synced(batch.resident_ids[field]);
        }
        release_transients();
        return true;
#else
        (void)batch;
        return false;
#endif
    }

    // ── Microphysics tendency application (GPU audit Phase 4.2) ──

    bool supports_microphysics_apply_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Pointwise and coordinate-agnostic (no metric terms, no stencil),
        // so no coordinate guard - the same kernel serves every grid.
        return pipeline(KernelId::MicrophysicsApply).is_ready();
#else
        return false;
#endif
    }

    bool dispatch_microphysics_apply(const MicrophysicsApplyDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        auto*       theta_data     = dispatch.theta_data;
        auto*       qv_data        = dispatch.qv_data;
        auto*       qc_data        = dispatch.qc_data;
        auto*       qr_data        = dispatch.qr_data;
        auto*       qi_data        = dispatch.qi_data;
        auto*       qs_data        = dispatch.qs_data;
        auto*       qg_data        = dispatch.qg_data;
        auto*       qh_data        = dispatch.qh_data;
        const auto* dtheta         = dispatch.dtheta;
        const auto* dqv            = dispatch.dqv;
        const auto* dqc            = dispatch.dqc;
        const auto* dqr            = dispatch.dqr;
        const auto* dqi            = dispatch.dqi;
        const auto* dqs            = dispatch.dqs;
        const auto* dqg            = dispatch.dqg;
        const auto* dqh            = dispatch.dqh;
        const auto* dtheta_rad     = dispatch.dtheta_rad;
        const auto* dtheta_pbl     = dispatch.dtheta_pbl;
        const auto* dqv_pbl        = dispatch.dqv_pbl;
        const int   nx             = dispatch.nx;
        const int   ny             = dispatch.ny;
        const int   nz             = dispatch.nz;
        const int   i_begin        = dispatch.i_begin;
        const int   i_end          = dispatch.i_end;
        const int   j_begin        = dispatch.j_begin;
        const int   j_end          = dispatch.j_end;
        const float dt_step        = dispatch.dt_step;
        const float theta_fallback = dispatch.theta_fallback;
        const float theta_min      = dispatch.theta_min;
        const float theta_max      = dispatch.theta_max;
        const float max_theta_step = dispatch.max_theta_step;
        const float qv_max_val     = dispatch.qv_max_val;
        const float hydro_max_val  = dispatch.hydro_max_val;
        auto*       stats_out      = dispatch.stats_out;
        const int   stats_len      = dispatch.stats_len;

        if (!pipeline(KernelId::MicrophysicsApply).is_ready())
        {
            return false;
        }
        if (i_begin < 0 || i_begin >= i_end || i_end > nx || j_begin < 0 || j_begin >= j_end || j_end > ny)
        {
            return false;
        }
        // Stats layout: 10 scalars + per-k + per-i clamp histograms.
        const int stats_needed = 10 + nz + nx;
        if (stats_out == nullptr || stats_len < stats_needed)
        {
            return false;
        }

        uint32_t     total_cells = 0;
        VkDeviceSize field_bytes = 0;
        if (!checked_field_dispatch(nx, ny, nz, total_cells, field_bytes, "microphysics apply"))
            return false;
        if (static_cast<VkDeviceSize>(stats_needed) * sizeof(uint32_t) > field_bytes)
        {
            return false;
        }
        const uint32_t total_points    = total_cells;
        const uint32_t workgroup_count = 1u + (total_points - 1u) / 64u;

        // Slots: 8 in-place state fields, 8 tendencies, 3 auxiliary
        // tendencies, 1 stats block (zero-filled; the shader accumulates
        // guard counters into it with atomics).
        // The 8 in-place state fields bind pinned named slots
        // (GPU audit Phase 4.3d): pointwise and sedimentation dispatches
        // publish the state they only read, while the intervening driver work
        // touches tendencies rather than state.  Reuse is deliberately
        // field-by-field: Kessler publishes only its three authoritative
        // precipitation fields, whereas Lin can publish all eight. One miss
        // must not force seven otherwise-valid fields back through host upload.
        // The parity replay cross-checks the assertion at sample cadence.
        // Either way this dispatch re-commits after its D2H (device and
        // host agree again).
        static const char* const kStateNames[8] = {"theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh"};
        float* const state_ptrs[8]  = {theta_data, qv_data, qc_data, qr_data, qi_data, qs_data, qg_data, qh_data};
        bool         reuse_state[8] = {};
        for (int i = 0; i < 8; ++i)
        {
            reuse_state[i] = named_slot_reusable(kStateNames[i], state_ptrs[i], field_bytes);
        }

        constexpr int kSlotCount = 20;
        int           slots[kSlotCount];
        VkDeviceSize  state_sizes[8];
        for (VkDeviceSize& size : state_sizes)
            size = field_bytes;
        if (!named_slots_acquire_many(kStateNames, state_sizes, 8, slots))
        {
            log_vulkan_warning("failed to pin named slots for microphysics apply");
            return false;
        }
        if (!acquire_pool_slots(field_bytes, kSlotCount - 8, slots + 8))
        {
            log_vulkan_warning("failed to acquire pool slots for microphysics apply");
            return false;
        }
        for (int i = 0; i < 8; ++i)
        {
            if (!reuse_state[i])
            {
                // The mapped slot is about to be overwritten from the host;
                // do not leave an older generation valid if recording fails.
                named_slot_invalidate(kStateNames[i]);
            }
        }
        const int slot_stats = slots[19];

        const bool unified  = device_context_.has_unified_memory;
        auto       host_ptr = [&](int slot) -> void*
        { return unified ? buffer_pool_.device(slot).mapped : buffer_pool_.staging(slot).mapped; };

        const float* uploads[19] = {theta_data,
                                    qv_data,
                                    qc_data,
                                    qr_data,
                                    qi_data,
                                    qs_data,
                                    qg_data,
                                    qh_data,
                                    dtheta,
                                    dqv,
                                    dqc,
                                    dqr,
                                    dqi,
                                    dqs,
                                    dqg,
                                    dqh,
                                    dtheta_rad,
                                    dtheta_pbl,
                                    dqv_pbl};
        for (int s = 0; s < 19; ++s)
        {
            if (s < 8 && reuse_state[s])
            {
                continue; // named state already device-resident
            }
            tracked_memcpy(host_ptr(slots[s]), uploads[s], field_bytes);
        }
        tracked_memset(host_ptr(slot_stats), 0, static_cast<std::size_t>(stats_needed) * sizeof(uint32_t));

        update_pooled_descriptor_set(pipeline(KernelId::MicrophysicsApply), slots, field_bytes, kSlotCount);

        vkResetCommandBuffer_(cmd_buf_, 0);
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (begin_profiled_command_buffer(begin_info) != VK_SUCCESS)
        {
            release_pool_slots(slots + 8, kSlotCount - 8);
            return false;
        }

        const VkPipelineStageFlags both_stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;

        // ── H2D transfer ──
        if (unified)
        {
            VkMemoryBarrier host_barrier{};
            host_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            host_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            host_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(
                cmd_buf_, VK_PIPELINE_STAGE_HOST_BIT, both_stages, 0, 1, &host_barrier, 0, nullptr, 0, nullptr);
        }
        else
        {
            VkBufferCopy copy_region{};
            copy_region.size = field_bytes;
            for (int s = 0; s < kSlotCount; ++s)
            {
                if (s < 8 && reuse_state[s])
                {
                    continue; // named state already device-resident
                }
                tracked_cmd_copy_buffer(
                    buffer_pool_.staging(slots[s]).buffer, buffer_pool_.device(slots[s]).buffer, 1, &copy_region);
            }
            VkMemoryBarrier full_barrier{};
            full_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            full_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            full_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_, both_stages, both_stages, 0, 1, &full_barrier, 0, nullptr, 0, nullptr);
        }

        MicrophysicsApplyPushConstants pc{};
        pc.nx             = nx;
        pc.ny             = ny;
        pc.nz             = nz;
        pc.pad0           = 0;
        pc.i_begin        = i_begin;
        pc.i_end          = i_end;
        pc.j_begin        = j_begin;
        pc.j_end          = j_end;
        pc.dt_step        = dt_step;
        pc.theta_fallback = theta_fallback;
        pc.theta_min      = theta_min;
        pc.theta_max      = theta_max;
        pc.max_theta_step = max_theta_step;
        pc.qv_max         = qv_max_val;
        pc.hydro_max      = hydro_max_val;
        pc.pad1           = 0.0f;

        vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline(KernelId::MicrophysicsApply).pipeline);
        vkCmdBindDescriptorSets_(cmd_buf_,
                                 VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipeline(KernelId::MicrophysicsApply).pipeline_layout,
                                 0,
                                 1,
                                 &pipeline(KernelId::MicrophysicsApply).descriptor_set,
                                 0,
                                 nullptr);
        vkCmdPushConstants_(cmd_buf_,
                            pipeline(KernelId::MicrophysicsApply).pipeline_layout,
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0,
                            sizeof(pc),
                            &pc);
        record_linear_group_dispatch(workgroup_count, KernelId::MicrophysicsApply);

        // ── D2H transfer (8 state fields + stats) ──
        if (unified)
        {
            VkMemoryBarrier out_barrier{};
            out_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            out_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            out_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier_(
                cmd_buf_, both_stages, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &out_barrier, 0, nullptr, 0, nullptr);
        }
        else
        {
            VkMemoryBarrier shader_to_transfer{};
            shader_to_transfer.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            shader_to_transfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            shader_to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(
                cmd_buf_, both_stages, both_stages, 0, 1, &shader_to_transfer, 0, nullptr, 0, nullptr);
            VkBufferCopy copy_region{};
            copy_region.size = field_bytes;
            for (int s = 0; s < 8; ++s)
            {
                tracked_cmd_copy_buffer(
                    buffer_pool_.device(slots[s]).buffer, buffer_pool_.staging(slots[s]).buffer, 1, &copy_region);
            }
            tracked_cmd_copy_buffer(
                buffer_pool_.device(slot_stats).buffer, buffer_pool_.staging(slot_stats).buffer, 1, &copy_region);
        }

        if (end_profiled_command_buffer() != VK_SUCCESS)
        {
            release_pool_slots(slots + 8, kSlotCount - 8);
            return false;
        }

        VkSubmitInfo submit_info{};
        submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers    = &cmd_buf_;
        if (reset_fence_and_submit("microphysics_apply", submit_info) != VK_SUCCESS)
        {
            release_pool_slots(slots + 8, kSlotCount - 8);
            return false;
        }

        if (timed_wait_fence("microphysics_apply") != VK_SUCCESS)
        {
            // Execution state unknown after a timeout and this dispatch
            // writes the named state slots in place - their content can
            // no longer be trusted.
            for (int i = 0; i < 8; ++i)
            {
                named_slot_invalidate(kStateNames[i]);
            }
            release_pool_slots(slots + 8, kSlotCount - 8);
            return false;
        }

        float* writebacks[8] = {theta_data, qv_data, qc_data, qr_data, qi_data, qs_data, qg_data, qh_data};
        for (int s = 0; s < 8; ++s)
        {
            tracked_memcpy(writebacks[s], host_ptr(slots[s]), field_bytes);
        }
        tracked_memcpy(stats_out, host_ptr(slot_stats), static_cast<std::size_t>(stats_needed) * sizeof(uint32_t));

        // Device and host state agree again after the writebacks above;
        // keep the named entries valid for the rest of the step.
        for (int i = 0; i < 8; ++i)
        {
            named_slot_commit(kStateNames[i], state_ptrs[i], field_bytes);
        }

        release_pool_slots(slots + 8, kSlotCount - 8);
        return true;
#else
        (void)dispatch;
        return false;
#endif
    }

    // ── SGS tendency application (GPU audit Phase 4.2) ──

    bool supports_sgs_apply_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Pointwise and coordinate-agnostic.
        return pipeline(KernelId::SgsApply).is_ready();
#else
        return false;
#endif
    }

    bool dispatch_sgs_apply(const SgsApplyDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (dispatch.collocated && (dispatch.nx < 3 || dispatch.ny < 3 || dispatch.nz < 3))
            return false;
        auto*       u              = dispatch.u;
        auto*       v              = dispatch.v;
        auto*       w              = dispatch.w;
        auto*       theta_data     = dispatch.theta_data;
        auto*       qv_data        = dispatch.qv_data;
        auto*       tke_data       = dispatch.tke_data;
        const auto* dudt           = dispatch.dudt;
        const auto* dvdt           = dispatch.dvdt;
        const auto* dwdt           = dispatch.dwdt;
        const auto* dthetadt       = dispatch.dthetadt;
        const auto* dqvdt          = dispatch.dqvdt;
        const auto* dtkedt         = dispatch.dtkedt;
        const auto* dtkedt_pbl     = dispatch.dtkedt_pbl;
        const int   nx             = dispatch.nx;
        const int   ny             = dispatch.ny;
        const int   nz             = dispatch.nz;
        const bool  has_qv         = dispatch.has_qv;
        const bool  has_tke        = dispatch.has_tke;
        const float dt_step        = dispatch.dt_step;
        const float theta_fallback = dispatch.theta_fallback;
        const float theta_min      = dispatch.theta_min;
        const float theta_max      = dispatch.theta_max;
        const float qv_max_val     = dispatch.qv_max_val;
        const float wind_h_max     = dispatch.wind_h_max;
        const float wind_v_max     = dispatch.wind_v_max;

        if (!pipeline(KernelId::SgsApply).is_ready())
        {
            return false;
        }

        SgsApplyPushConstants pc{};
        pc.nx             = nx;
        pc.ny             = ny;
        pc.nz             = nz;
        pc.has_qv         = has_qv ? 1 : 0;
        pc.has_tke        = has_tke ? 1 : 0;
        pc.dt_step        = dt_step;
        pc.theta_fallback = theta_fallback;
        pc.theta_min      = theta_min;
        pc.theta_max      = theta_max;
        pc.qv_max         = qv_max_val;
        pc.wind_h_max     = wind_h_max;
        pc.wind_v_max     = wind_v_max;
        pc.sanitize       = dispatch.sanitize ? 1 : 0;
        pc.collocated     = dispatch.collocated ? 1 : 0;

        uint32_t     total_points        = 0;
        VkDeviceSize field_bytes_checked = 0;
        if (!checked_field_dispatch(nx, ny, nz, total_points, field_bytes_checked, "SGS apply"))
            return false;
        const float* inputs[13] = {
            u, v, w, theta_data, qv_data, tke_data, dudt, dvdt, dwdt, dthetadt, dqvdt, dtkedt, dtkedt_pbl};
        float*      outputs[6]           = {u, v, w, theta_data, qv_data, tke_data};
        const char* input_names[13]      = {"u",
                                            "v",
                                            "w",
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            dispatch.resident_tendencies_unchanged ? "sgs_dudt" : nullptr,
                                            dispatch.resident_tendencies_unchanged ? "sgs_dvdt" : nullptr,
                                            dispatch.resident_tendencies_unchanged ? "sgs_dwdt" : nullptr,
                                            dispatch.resident_tendencies_unchanged ? "sgs_dthetadt" : nullptr,
                                            dispatch.resident_tendencies_unchanged ? "sgs_dqvdt" : nullptr,
                                            nullptr,
                                            nullptr};
        const bool  output_write_only[6] = {true, true, true, true, true, true};

        const bool dispatched = dispatch_multi_field_kernel(pipeline(KernelId::SgsApply),
                                                            KernelId::SgsApply,
                                                            &pc,
                                                            inputs,
                                                            13,
                                                            outputs,
                                                            6,
                                                            nx,
                                                            ny,
                                                            nz,
                                                            total_points,
                                                            nullptr,
                                                            input_names,
                                                            output_write_only,
                                                            false,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            0,
                                                            dispatch.completed_observer,
                                                            dispatch.completed_observer_context);
        if (dispatched)
        {
            // The shader wrote newer winds into separate output slots; the
            // named input slots still contain the pre-apply values.
            named_slot_invalidate("u");
            named_slot_invalidate("v");
            named_slot_invalidate("w");
        }
        return dispatched;
#else
        (void)dispatch;
        return false;
#endif
    }

    // ── Slow-tendency application (GPU audit Phase 4.2) ──

    bool supports_slow_apply_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Pointwise and coordinate-agnostic.
        return pipeline(KernelId::SlowApply).is_ready();
#else
        return false;
#endif
    }

    bool dispatch_slow_apply(const SlowApplyDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!pipeline(KernelId::SlowApply).is_ready())
        {
            return false;
        }

        SlowApplyPushConstants pc{};
        pc.nx       = dispatch.nx;
        pc.ny       = dispatch.ny;
        pc.nz       = dispatch.nz;
        pc.sanitize = dispatch.sanitize ? 1 : 0;
        pc.dt_large = dispatch.dt_large;
        pc.pad1 = pc.pad2 = pc.pad3 = 0.0f;

        uint32_t     total_points        = 0;
        VkDeviceSize field_bytes_checked = 0;
        if (!checked_field_dispatch(
                dispatch.nx, dispatch.ny, dispatch.nz, total_points, field_bytes_checked, "slow apply"))
            return false;
        const float* inputs[10]           = {dispatch.u,
                                             dispatch.v,
                                             dispatch.w,
                                             dispatch.p_data,
                                             dispatch.du_dt,
                                             dispatch.dv_dt,
                                             dispatch.dw_dt,
                                             dispatch.dp_dt,
                                             dispatch.du_dt_pbl,
                                             dispatch.dv_dt_pbl};
        float*       outputs[4]           = {dispatch.u, dispatch.v, dispatch.w, dispatch.p_data};
        const bool   output_write_only[4] = {true, true, true, true};

        return dispatch_multi_field_kernel(pipeline(KernelId::SlowApply),
                                           KernelId::SlowApply,
                                           &pc,
                                           inputs,
                                           10,
                                           outputs,
                                           4,
                                           dispatch.nx,
                                           dispatch.ny,
                                           dispatch.nz,
                                           total_points,
                                           nullptr,
                                           nullptr,
                                           output_write_only,
                                           true,
                                           nullptr,
                                           dispatch.completed_observer,
                                           dispatch.completed_observer_context);
#else
        (void)dispatch;
        return false;
#endif
    }

    // ── Batched Cartesian C-grid acoustic loop (GPU audit 4.1) ──

    bool supports_acoustic_substeps_batched_cartesian_cgrid_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Cartesian-only shaders (GPU.md section 2.0 routing rule).
        if (coord_system_ != CoordinateSystem::Cartesian)
            return false;
        return pipeline(KernelId::AcousticPressureCartesianCgrid).is_ready() &&
               pipeline(KernelId::AcousticMomentumCartesianCgrid).is_ready() &&
               pipeline(KernelId::AcousticBcFaceCartesianCgrid).is_ready();
#else
        return false;
#endif
    }

    bool supports_batched_cartesian_cgrid_mass_flux_export() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return coord_system_ == CoordinateSystem::Cartesian &&
               pipeline(KernelId::AcousticPressureMassFluxCartesianCgrid).is_ready() &&
               pipeline(KernelId::AcousticMomentumCartesianCgrid).is_ready() &&
               pipeline(KernelId::AcousticBcFaceCartesianCgrid).is_ready();
#else
        return false;
#endif
    }

    bool supports_unrepaired_acoustic_batched() const override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        return supports_acoustic_substeps_batched_cartesian_cgrid_dispatch() &&
            (cuda_assist_enabled_ || pipeline(KernelId::AcousticRetainedFluxFp64).is_ready());
#else
        return false;
#endif
    }

    bool dispatch_acoustic_substeps_batched_cartesian_cgrid(
        const AcousticSubstepsBatchedCartesianCgridDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        const bool strict = !dispatch.repair_primary_state;
        if (dispatch.nx < 4 || dispatch.ny < 4 || dispatch.nz < 2 ||
            (strict && (!supports_unrepaired_acoustic_batched() || !valid_strict_acoustic_batch(dispatch))) ||
            (!strict && (dispatch.dp0_dz_low_data || dispatch.wave_history || dispatch.wave_history_output ||
                         dispatch.mass_flux_weight != 0.0 || dispatch.gamma_low != 0.0f)))
            return false;
        std::vector<float> gradient_storage;
        AcousticMomentumCartesianCgridDispatch gradient;
        gradient.dp0_dz_data = dispatch.dp0_dz_data;
        gradient.dp0_dz_low_data = dispatch.dp0_dz_low_data;
        gradient.nz = dispatch.nz;
        auto*       u                    = dispatch.u;
        auto*       v                    = dispatch.v;
        auto*       w                    = dispatch.w;
        auto*       rho                  = dispatch.rho;
        auto*       p                    = dispatch.p;
        const auto* theta_data           = dispatch.theta_data;
        const auto* rho0_base_data       = dispatch.rho0_base_data;
        const auto* p0_base_data         = dispatch.p0_base_data;
        const auto* dp0_dz_data          = pack_acoustic_base_gradient(gradient, gradient_storage);
        const int   base_len             = dispatch.base_len;
        const int   nx                   = dispatch.nx;
        const int   ny                   = dispatch.ny;
        const int   nz                   = dispatch.nz;
        const float dx_val               = dispatch.dx;
        const float dy_val               = dispatch.dy;
        const float dz_val               = dispatch.dz;
        const float gamma_val            = dispatch.gamma_val;
        const float dt_small             = dispatch.dt_small;
        const int   n_substeps           = dispatch.n_substeps;
        const float rho_min              = dispatch.rho_min;
        const float rho_max              = dispatch.rho_max;
        const float p_min                = dispatch.p_min;
        const float p_max                = dispatch.p_max;
        const float p_ref_fallback       = dispatch.p_ref_fallback;
        const float wind_clamp_h_val     = dispatch.wind_clamp_h;
        const float wind_clamp_v_val     = dispatch.wind_clamp_v;
        const float frame_velocity_u_ms  = dispatch.frame_velocity_u_ms;
        const float frame_velocity_v_ms  = dispatch.frame_velocity_v_ms;
        const int   bc_west              = dispatch.bc_west;
        const int   bc_east              = dispatch.bc_east;
        const int   bc_south             = dispatch.bc_south;
        const int   bc_north             = dispatch.bc_north;
        const int   bc_bottom            = dispatch.bc_bottom;
        const int   bc_top               = dispatch.bc_top;
        float*      mass_flux_x          = dispatch.mass_flux_x;
        float*      mass_flux_y          = dispatch.mass_flux_y;
        float*      mass_flux_z          = dispatch.mass_flux_z;
        const bool  accumulate_mass_flux = dispatch.accumulate_mass_flux;

        if (coord_system_ != CoordinateSystem::Cartesian)
        {
            log_vulkan_warning("batched acoustic (cartesian c-grid) dispatch "
                               "refused: shaders are cartesian-only and the "
                               "active coordinate system is not cartesian");
            return false;
        }
        if (!supports_acoustic_substeps_batched_cartesian_cgrid_dispatch())
        {
            return false;
        }
        if (rho0_base_data == nullptr || p0_base_data == nullptr || dp0_dz_data == nullptr || base_len < nz ||
            n_substeps < 1 ||
            (accumulate_mass_flux && (mass_flux_x == nullptr || mass_flux_y == nullptr || mass_flux_z == nullptr)))
        {
            return false;
        }

        uint32_t     total_points = 0;
        VkDeviceSize field_bytes  = 0;
        if (!checked_field_dispatch(nx, ny, nz, total_points, field_bytes, "batched acoustic c-grid"))
            return false;
        const uint32_t workgroup_count = 1u + (total_points - 1u) / 64u;
        const VkDeviceSize prof_bytes = static_cast<VkDeviceSize>(nz) * sizeof(float);
        const VkDeviceSize gradient_bytes = prof_bytes * (dispatch.dp0_dz_low_data ? 2 : 1);
        const std::size_t wave_count = 2 * (std::size_t(nx) + ny) * nz;
        const VkDeviceSize wave_bytes = wave_count * sizeof(float);

        if (strict)
        {
            int valid = 1;
            const float* inputs[] = {u, v, w, rho, p, theta_data};
            #pragma omp parallel reduction(&:valid) if(total_points >= 65536)
            {
                std::size_t worker = 0, workers = 1;
#ifdef _OPENMP
                worker = static_cast<std::size_t>(omp_get_thread_num());
                workers = static_cast<std::size_t>(omp_get_num_threads());
#endif
                const auto begin = std::size_t(total_points) * worker / workers;
                const auto end = std::size_t(total_points) * (worker + 1) / workers;
                for (int field = 0; field < 6; ++field)
                {
                    const auto* values = inputs[field];
                    int field_valid = 1;
                    #pragma omp simd reduction(&:field_valid)
                    for (std::size_t index = begin; index < end; ++index)
                        field_valid &= std::isfinite(values[index]) && (field < 3 || values[index] > 0.0f);
                    valid &= field_valid;
                }
            }
            for (std::size_t n = 0; n < wave_count; ++n)
                valid = valid && std::isfinite(dispatch.wave_history[n]);
            for (int k = 0; k < nz; ++k)
                valid = valid && std::isfinite(rho0_base_data[k]) && rho0_base_data[k] > 0 &&
                    std::isfinite(p0_base_data[k]) && p0_base_data[k] > 0 && std::isfinite(dispatch.dp0_dz_data[k]) &&
                    (!dispatch.dp0_dz_low_data || std::isfinite(dispatch.dp0_dz_low_data[k]));
            if (!valid) return false;
        }

        // Same 10-slot layout as the fused substep, but the per-substep
        // recording copies rho_new/p_new back into the canonical rho/p
        // slots so every substep (and every BC dispatch) reads slots 3/4 -
        // the device-device copy is what lets the single descriptor set per
        // pipeline serve all N substeps.
        constexpr int            kSlotCount       = 12;
        constexpr int            kNamedStateSlots = 3;
        int                      slots[kSlotCount];
        static const char* const kWindNames[kNamedStateSlots] = {"u", "v", "w"};
        const VkDeviceSize       wind_sizes[kNamedStateSlots] = {field_bytes, field_bytes, field_bytes};
        const VkDeviceSize transient_sizes[9] = {
            field_bytes, field_bytes, prof_bytes, prof_bytes, gradient_bytes,
            field_bytes, field_bytes, field_bytes, wave_bytes};
        if (!named_slots_acquire_many(kWindNames, wind_sizes, kNamedStateSlots, slots) ||
            !buffer_pool_.acquire_many(transient_sizes, 9, slots + kNamedStateSlots))
        {
            log_vulkan_warning("failed to acquire pool slots for batched acoustic (cartesian c-grid)");
            return false;
        }
        named_slot_invalidate("u");
        named_slot_invalidate("v");
        named_slot_invalidate("w");
        const auto release_transient_slots = [&]()
        { release_pool_slots(slots + kNamedStateSlots, kSlotCount - kNamedStateSlots); };
        const int slot_theta = slots[10];
        const int slot_rad     = slots[11];
        const int slot_u       = slots[0];
        const int slot_v       = slots[1];
        const int slot_w       = slots[2];
        const int slot_rho     = slots[3];
        const int slot_p       = slots[4];
        const int slot_rho0    = slots[5];
        const int slot_p0      = slots[6];
        const int slot_dp0     = slots[7];
        const int slot_rho_new = slots[8];
        const int slot_p_new   = slots[9];

        // The exported acoustic mass flux lives in named slots so the scalar
        // batch can bind the exact producer buffers without a second upload.
        // They are still copied to the caller after the fence, preserving a
        // complete CPU fallback/parity representation.
        int slot_mfx = -1;
        int slot_mfy = -1;
        int slot_mfz = -1;
        if (accumulate_mass_flux)
        {
            static const char* const kMassFluxNames[3] = {
                "cgrid_mass_flux_x", "cgrid_mass_flux_y", "cgrid_mass_flux_z"};
            const VkDeviceSize mass_flux_sizes[3] = {field_bytes, field_bytes, field_bytes};
            int                mass_flux_slots[3];
            if (!named_slots_acquire_many(kMassFluxNames, mass_flux_sizes, 3, mass_flux_slots))
            {
                release_transient_slots();
                log_vulkan_warning("failed to pin batched acoustic mass-flux export slots");
                return false;
            }
            slot_mfx = mass_flux_slots[0];
            slot_mfy = mass_flux_slots[1];
            slot_mfz = mass_flux_slots[2];
            named_slot_invalidate("cgrid_mass_flux_x");
            named_slot_invalidate("cgrid_mass_flux_y");
            named_slot_invalidate("cgrid_mass_flux_z");
        }

        const bool unified  = device_context_.has_unified_memory;
        auto       host_ptr = [&](int slot) -> void*
        { return unified ? buffer_pool_.device(slot).mapped : buffer_pool_.staging(slot).mapped; };

        // The three base profiles upload compact (nz floats; the shaders
        // index them by k) - no per-(i, j) broadcast (Phase 4.3).
        tracked_memcpy(host_ptr(slot_u), u, field_bytes);
        tracked_memcpy(host_ptr(slot_v), v, field_bytes);
        tracked_memcpy(host_ptr(slot_w), w, field_bytes);
        tracked_memcpy(host_ptr(slot_rho), rho, field_bytes);
        tracked_memcpy(host_ptr(slot_p), p, field_bytes);
        tracked_memcpy(host_ptr(slot_rho0), rho0_base_data, prof_bytes);
        tracked_memcpy(host_ptr(slot_p0), p0_base_data, prof_bytes);
        tracked_memcpy(host_ptr(slot_dp0), dp0_dz_data, gradient_bytes);
        tracked_memcpy(host_ptr(slot_theta), theta_data, field_bytes);
        if (strict) tracked_memcpy(host_ptr(slot_rad), dispatch.wave_history, wave_bytes);
        else tracked_memset(host_ptr(slot_rad), 0, wave_bytes);
        for (int index = 0; index < 3; ++index)
        {
            const float* wind[] = {u, v, w};
            named_slot_commit(kWindNames[index], wind[index], field_bytes);
            named_slot_invalidate(kWindNames[index]);
        }
        if (accumulate_mass_flux)
        {
            const char* names[] = {"cgrid_mass_flux_x", "cgrid_mass_flux_y", "cgrid_mass_flux_z"};
            const float* flux[] = {mass_flux_x, mass_flux_y, mass_flux_z};
            for (int index = 0; index < 3; ++index)
            {
                named_slot_commit(names[index], flux[index], field_bytes);
                named_slot_invalidate(names[index]);
            }
        }

        if (accumulate_mass_flux)
        {
            // One batched dispatch owns the whole large-step average.
            tracked_memset(host_ptr(slot_mfx), 0, field_bytes);
            tracked_memset(host_ptr(slot_mfy), 0, field_bytes);
            tracked_memset(host_ptr(slot_mfz), 0, field_bytes);
        }

        const VkDeviceSize pressure_ranges[13] = {
            field_bytes, field_bytes, field_bytes, field_bytes, field_bytes, prof_bytes, prof_bytes,
            field_bytes, field_bytes, field_bytes, field_bytes, field_bytes, field_bytes};
        if (accumulate_mass_flux && !strict)
        {
            int pressure_bindings[13] = {slot_u,
                                         slot_v,
                                         slot_w,
                                         slot_rho,
                                         slot_p,
                                         slot_rho0,
                                         slot_p0,
                                         slot_theta,
                                         slot_rho_new,
                                         slot_p_new,
                                         slot_mfx,
                                         slot_mfy,
                                         slot_mfz};
            update_pooled_descriptor_set(
                pipeline(KernelId::AcousticPressureMassFluxCartesianCgrid), pressure_bindings, field_bytes, 13, pressure_ranges);
        }
        else
        {
            int pressure_bindings[10] = {
                slot_u, slot_v, slot_w, slot_rho, slot_p, slot_rho0, slot_p0, slot_theta, slot_rho_new, slot_p_new};
            update_pooled_descriptor_set(
                pipeline(KernelId::AcousticPressureCartesianCgrid), pressure_bindings, field_bytes, 10, pressure_ranges);
        }
        int momentum_bindings[9] = {slot_rho, slot_p, slot_dp0, slot_u, slot_v, slot_w, slot_u, slot_v, slot_w};
        const VkDeviceSize momentum_ranges[9] = {
            field_bytes, field_bytes, gradient_bytes, field_bytes, field_bytes,
            field_bytes, field_bytes, field_bytes, field_bytes};
        update_pooled_descriptor_set(
            pipeline(KernelId::AcousticMomentumCartesianCgrid), momentum_bindings, field_bytes, 9, momentum_ranges);
        int bc_bindings[6] = {slot_u, slot_v, slot_w, slot_rho, slot_p, slot_rad};
        const VkDeviceSize bc_ranges[6] = {field_bytes, field_bytes, field_bytes, field_bytes, field_bytes, wave_bytes};
        update_pooled_descriptor_set(pipeline(KernelId::AcousticBcFaceCartesianCgrid), bc_bindings, field_bytes, 6, bc_ranges);
        if (strict && !cuda_assist_enabled_)
        {
            const int flux_bindings[7] = {slot_rho, slot_u, slot_v, slot_w, slot_mfx, slot_mfy, slot_mfz};
            update_pooled_descriptor_set(pipeline(KernelId::AcousticRetainedFluxFp64), flux_bindings, field_bytes, 7);
        }

#ifdef VK_NV_cuda_kernel_launch
        std::array<VkDeviceAddress, 7> cuda_addresses{};
        if (strict && cuda_assist_enabled_)
        {
            const int slots[] = {slot_rho, slot_u, slot_v, slot_w, slot_mfx, slot_mfy, slot_mfz};
            for (std::size_t field = 0; field < cuda_addresses.size(); ++field)
            {
                cuda_addresses[field] = cuda_flux_.address(buffer_pool_.device(slots[field]).buffer);
                if (!cuda_addresses[field])
                {
                    release_transient_slots();
                    return false;
                }
            }
        }
#endif
        vkResetCommandBuffer_(cmd_buf_, 0);
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (begin_profiled_command_buffer(begin_info) != VK_SUCCESS)
        {
            release_transient_slots();
            return false;
        }

        VkMemoryBarrier full_barrier{};
        full_barrier.sType                     = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        full_barrier.srcAccessMask             = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        full_barrier.dstAccessMask             = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                                 VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        const VkPipelineStageFlags both_stages = cuda_assist_enabled_ ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT :
            (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
        if (cuda_assist_enabled_)
        {
            full_barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            full_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        }
        auto                       stage_barrier = [&]()
        { vkCmdPipelineBarrier_(cmd_buf_, both_stages, both_stages, 0, 1, &full_barrier, 0, nullptr, 0, nullptr); };

        // ── H2D transfer ──
        if (unified)
        {
            VkMemoryBarrier host_barrier{};
            host_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            host_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            host_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(
                cmd_buf_, VK_PIPELINE_STAGE_HOST_BIT, both_stages, 0, 1, &host_barrier, 0, nullptr, 0, nullptr);
        }
        else
        {
            for (int s = 0; s < kSlotCount; ++s)
            {
                if (s == 8 || s == 9)
                {
                    continue; // rho_new/p_new are pure outputs
                }
                VkBufferCopy copy_region{};
                copy_region.size = s == 7 ? gradient_bytes : s == 11 ? wave_bytes : (s == 5 || s == 6) ? prof_bytes : field_bytes;
                tracked_cmd_copy_buffer(
                    buffer_pool_.staging(slots[s]).buffer, buffer_pool_.device(slots[s]).buffer, 1, &copy_region);
            }
            if (accumulate_mass_flux)
            {
                VkBufferCopy copy_region{};
                copy_region.size        = field_bytes;
                const int mass_slots[3] = {slot_mfx, slot_mfy, slot_mfz};
                for (int s = 0; s < 3; ++s)
                {
                    tracked_cmd_copy_buffer(buffer_pool_.staging(mass_slots[s]).buffer,
                                            buffer_pool_.device(mass_slots[s]).buffer,
                                            1,
                                            &copy_region);
                }
            }
            stage_barrier();
        }

        AcousticPressureCartesianCgridPushConstants pressure_pc{};
        pressure_pc.nx                  = nx;
        pressure_pc.ny                  = ny;
        pressure_pc.nz                  = nz;
        pressure_pc.dx                  = dx_val;
        pressure_pc.dy                  = dy_val;
        pressure_pc.dz_val              = dz_val;
        pressure_pc.gamma_val           = gamma_val;
        pressure_pc.dt_small            = dt_small;
        pressure_pc.rho_min             = rho_min;
        pressure_pc.rho_max             = rho_max;
        pressure_pc.p_min               = p_min;
        pressure_pc.p_max               = p_max;
        pressure_pc.p_ref_fallback      = p_ref_fallback;
        pressure_pc.pad0                = accumulate_mass_flux ? 1.0f / static_cast<float>(n_substeps) : 0.0f;
        pressure_pc.frame_velocity_u_ms = frame_velocity_u_ms;
        pressure_pc.frame_velocity_v_ms = frame_velocity_v_ms;
        pressure_pc.preserve_unrepaired = strict ? 1 : 0;
        pressure_pc.gamma_low = dispatch.gamma_low;

        ComputePipelineState& pressure_pipeline = accumulate_mass_flux && !strict
                                                      ? pipeline(KernelId::AcousticPressureMassFluxCartesianCgrid)
                                                      : pipeline(KernelId::AcousticPressureCartesianCgrid);

        AcousticMomentumCartesianCgridPushConstants momentum_pc{};
        momentum_pc.nx           = nx;
        momentum_pc.ny           = ny;
        momentum_pc.nz           = nz;
        momentum_pc.dx           = dx_val;
        momentum_pc.dy           = dy_val;
        momentum_pc.dz_val       = dz_val;
        momentum_pc.dt_small     = dt_small;
        momentum_pc.wind_clamp_h = wind_clamp_h_val;
        momentum_pc.wind_clamp_v = wind_clamp_v_val;
        momentum_pc.preserve_unrepaired = strict ? 1 : 0;
        momentum_pc.paired_base_gradient = dispatch.dp0_dz_low_data ? 1 : 0;
        nimbus::compute::abi::AcousticRetainedFluxFp64PushConstants flux_pc{};
        flux_pc.nx = nx; flux_pc.ny = ny; flux_pc.nz = nz;
        flux_pc.frame_u_ms = frame_velocity_u_ms; flux_pc.frame_v_ms = frame_velocity_v_ms;
        std::uint64_t weight_bits = 0;
        std::memcpy(&weight_bits, &dispatch.mass_flux_weight, sizeof(weight_bits));
        flux_pc.weight_low = static_cast<std::uint32_t>(weight_bits);
        flux_pc.weight_high = static_cast<std::uint32_t>(weight_bits >> 32);

        VkBufferCopy field_copy{};
        field_copy.size = field_bytes;

        const auto record_bc_face = [&](int face_code)
        {
            AcousticBcFacePushConstants bc_pc{};
            bc_pc.nx                  = nx;
            bc_pc.ny                  = ny;
            bc_pc.nz                  = nz;
            bc_pc.face                = face_code;
            bc_pc.bc_west             = bc_west;
            bc_pc.bc_east             = bc_east;
            bc_pc.bc_south            = bc_south;
            bc_pc.bc_north            = bc_north;
            bc_pc.bc_bottom           = bc_bottom;
            bc_pc.bc_top              = bc_top;
            bc_pc.frame_velocity_u_ms = frame_velocity_u_ms;
            bc_pc.frame_velocity_v_ms = frame_velocity_v_ms;
            bc_pc.dt_small            = dt_small;
            bc_pc.inv_dx              = (dx_val > 0.0f) ? 1.0f / dx_val : 0.0f;
            bc_pc.inv_dy              = (dy_val > 0.0f) ? 1.0f / dy_val : 0.0f;
            bc_pc.pad2                = 0.0f;

            vkCmdBindPipeline_(
                cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline(KernelId::AcousticBcFaceCartesianCgrid).pipeline);
            vkCmdBindDescriptorSets_(cmd_buf_,
                                     VK_PIPELINE_BIND_POINT_COMPUTE,
                                     pipeline(KernelId::AcousticBcFaceCartesianCgrid).pipeline_layout,
                                     0,
                                     1,
                                     &pipeline(KernelId::AcousticBcFaceCartesianCgrid).descriptor_set,
                                     0,
                                     nullptr);
            vkCmdPushConstants_(cmd_buf_,
                                pipeline(KernelId::AcousticBcFaceCartesianCgrid).pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT,
                                0,
                                sizeof(bc_pc),
                                &bc_pc);
            record_linear_group_dispatch(workgroup_count, KernelId::AcousticBcFaceCartesianCgrid);
            stage_barrier();
        };

        // Seed the KW78 radiation state from the uploaded (pre-step)
        // faces - the same values the CPU stash re-seeds from at the
        // step boundary.
        if (!strict) record_bc_face(5);

        for (int n = 0; n < n_substeps; ++n)
        {
            if (strict)
            {
#ifdef VK_NV_cuda_kernel_launch
                if (cuda_assist_enabled_)
                {
                    cuda_flux_.record(cmd_buf_, cuda_addresses, nx, ny, nz, frame_velocity_u_ms,
                        frame_velocity_v_ms, dispatch.mass_flux_weight, device_context_.selected_device_properties.limits);
                    ++pending_dispatch_count_;
                    ++pending_kernel_dispatch_counts_[static_cast<std::size_t>(KernelId::AcousticRetainedFluxFp64)];
                }
                else
#endif
                {
                const auto& flux_pipeline = pipeline(KernelId::AcousticRetainedFluxFp64);
                vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, flux_pipeline.pipeline);
                vkCmdBindDescriptorSets_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, flux_pipeline.pipeline_layout,
                    0, 1, &flux_pipeline.descriptor_set, 0, nullptr);
                vkCmdPushConstants_(cmd_buf_, flux_pipeline.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                    0, sizeof(flux_pc), &flux_pc);
                record_linear_group_dispatch(workgroup_count, KernelId::AcousticRetainedFluxFp64);
                }
                stage_barrier();
            }
            // Pressure (reads u,v,w,rho,p; writes rho_new,p_new).
            vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pressure_pipeline.pipeline);
            vkCmdBindDescriptorSets_(cmd_buf_,
                                     VK_PIPELINE_BIND_POINT_COMPUTE,
                                     pressure_pipeline.pipeline_layout,
                                     0,
                                     1,
                                     &pressure_pipeline.descriptor_set,
                                     0,
                                     nullptr);
            vkCmdPushConstants_(cmd_buf_,
                                pressure_pipeline.pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT,
                                0,
                                sizeof(pressure_pc),
                                &pressure_pc);
            record_linear_group_dispatch(workgroup_count, accumulate_mass_flux && !strict
                ? KernelId::AcousticPressureMassFluxCartesianCgrid : KernelId::AcousticPressureCartesianCgrid);
            stage_barrier();

            // Canonicalize: rho_new/p_new -> rho/p slots.
            tracked_cmd_copy_buffer(
                buffer_pool_.device(slot_rho_new).buffer, buffer_pool_.device(slot_rho).buffer, 1, &field_copy);
            tracked_cmd_copy_buffer(
                buffer_pool_.device(slot_p_new).buffer, buffer_pool_.device(slot_p).buffer, 1, &field_copy);
            stage_barrier();

            // Momentum (reads rho,p,dp0,u,v,w; writes u,v,w in place).
            vkCmdBindPipeline_(
                cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline(KernelId::AcousticMomentumCartesianCgrid).pipeline);
            vkCmdBindDescriptorSets_(cmd_buf_,
                                     VK_PIPELINE_BIND_POINT_COMPUTE,
                                     pipeline(KernelId::AcousticMomentumCartesianCgrid).pipeline_layout,
                                     0,
                                     1,
                                     &pipeline(KernelId::AcousticMomentumCartesianCgrid).descriptor_set,
                                     0,
                                     nullptr);
            vkCmdPushConstants_(cmd_buf_,
                                pipeline(KernelId::AcousticMomentumCartesianCgrid).pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT,
                                0,
                                sizeof(momentum_pc),
                                &momentum_pc);
            record_linear_group_dispatch(workgroup_count, KernelId::AcousticMomentumCartesianCgrid);
            stage_barrier();

            // Acoustic BCs: five face dispatches in the CPU's order (corner
            // ghosts read values earlier faces wrote).
            for (int face = 0; face < 5; ++face)
            {
                record_bc_face(face);
            }
        }

        // ── D2H transfer (u, v, w, rho, p) ──
        if (unified)
        {
            VkMemoryBarrier out_barrier{};
            out_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            out_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            out_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier_(
                cmd_buf_, both_stages, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &out_barrier, 0, nullptr, 0, nullptr);
        }
        else
        {
            VkBufferCopy copy_region{};
            copy_region.size             = field_bytes;
            const int writeback_slots[5] = {slot_u, slot_v, slot_w, slot_rho, slot_p};
            for (int s = 0; s < 5; ++s)
            {
                tracked_cmd_copy_buffer(buffer_pool_.device(writeback_slots[s]).buffer,
                                        buffer_pool_.staging(writeback_slots[s]).buffer,
                                        1,
                                        &copy_region);
            }
            if (accumulate_mass_flux)
            {
                const int mass_slots[3] = {slot_mfx, slot_mfy, slot_mfz};
                for (int s = 0; s < 3; ++s)
                {
                    tracked_cmd_copy_buffer(buffer_pool_.device(mass_slots[s]).buffer,
                                            buffer_pool_.staging(mass_slots[s]).buffer,
                                            1,
                                            &copy_region);
                }
            }
        }

        if (strict && !unified)
        {
            VkBufferCopy copy_region{};
            copy_region.size = wave_bytes;
            tracked_cmd_copy_buffer(buffer_pool_.device(slot_rad).buffer, buffer_pool_.staging(slot_rad).buffer,
                1, &copy_region);
        }

        if (end_profiled_command_buffer() != VK_SUCCESS)
        {
            release_transient_slots();
            return false;
        }

        VkSubmitInfo submit_info{};
        submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers    = &cmd_buf_;
        if (reset_fence_and_submit("acoustic_batched_cartesian_cgrid", submit_info) != VK_SUCCESS)
        {
            release_transient_slots();
            return false;
        }

        if (timed_wait_fence("acoustic_batched_cartesian_cgrid") != VK_SUCCESS)
        {
            release_transient_slots();
            return false;
        }

        if (strict)
        {
            const int output_slots[] = {slot_u, slot_v, slot_w, slot_rho, slot_p, slot_mfx, slot_mfy, slot_mfz, slot_rad};
            int valid = 1;
            #pragma omp parallel reduction(&:valid) if(total_points >= 65536)
            {
                std::size_t worker = 0, workers = 1;
#ifdef _OPENMP
                worker = static_cast<std::size_t>(omp_get_thread_num());
                workers = static_cast<std::size_t>(omp_get_num_threads());
#endif
                for (int field = 0; field < 9; ++field)
                {
                    const auto count = field == 8 ? wave_count : static_cast<std::size_t>(total_points);
                    const auto* values = static_cast<const float*>(host_ptr(output_slots[field]));
                    const auto begin = count * worker / workers;
                    const auto end = count * (worker + 1) / workers;
                    int field_valid = 1;
                    #pragma omp simd reduction(&:field_valid)
                    for (std::size_t index = begin; index < end; ++index)
                        field_valid &= std::isfinite(values[index]) && ((field != 3 && field != 4) || values[index] > 0.0f);
                    valid &= field_valid;
                }
            }
            if (!valid)
            {
                release_transient_slots();
                return false;
            }
        }
        tracked_memcpy(u, host_ptr(slot_u), field_bytes);
        tracked_memcpy(v, host_ptr(slot_v), field_bytes);
        tracked_memcpy(w, host_ptr(slot_w), field_bytes);
        tracked_memcpy(rho, host_ptr(slot_rho), field_bytes);
        tracked_memcpy(p, host_ptr(slot_p), field_bytes);
        named_slot_commit("u", u, field_bytes);
        named_slot_commit("v", v, field_bytes);
        named_slot_commit("w", w, field_bytes);

        if (accumulate_mass_flux)
        {
            tracked_memcpy(mass_flux_x, host_ptr(slot_mfx), field_bytes);
            tracked_memcpy(mass_flux_y, host_ptr(slot_mfy), field_bytes);
            tracked_memcpy(mass_flux_z, host_ptr(slot_mfz), field_bytes);
            named_slot_commit("cgrid_mass_flux_x", mass_flux_x, field_bytes);
            named_slot_commit("cgrid_mass_flux_y", mass_flux_y, field_bytes);
            named_slot_commit("cgrid_mass_flux_z", mass_flux_z, field_bytes);
        }

        if (strict) tracked_memcpy(dispatch.wave_history_output, host_ptr(slot_rad), wave_bytes);
        release_transient_slots();
        return true;
#else
        (void)dispatch;
        return false;
#endif
    }

    // ── Batched acoustic substeps (all N in one command buffer) ──────

    bool supports_acoustic_substeps_batched_dispatch() const
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Readiness of the Cartesian pipelines, mirroring
        // supports_acoustic_substep_fused_dispatch.
        return pipeline(KernelId::AcousticPressureCartesian).is_ready() &&
               pipeline(KernelId::AcousticMomentumCartesian).is_ready();
#else
        return false;
#endif
    }

    bool dispatch_acoustic_substeps_batched(const AcousticSubstepsBatchedDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        float*      u            = dispatch.u;
        float*      v            = dispatch.v;
        float*      w            = dispatch.w;
        float*      rho          = dispatch.rho;
        float*      p            = dispatch.p;
        const int   nr           = dispatch.nr;
        const int   nth          = dispatch.nth;
        const int   nz           = dispatch.nz;
        const float dr_val       = dispatch.dr;
        const float dtheta_val   = dispatch.dtheta;
        const float dz_val       = dispatch.dz;
        const float gamma_val    = dispatch.gamma_val;
        const float dt_small     = dispatch.dt_small;
        const int   n_substeps   = dispatch.n_substeps;
        const float rho_floor    = dispatch.rho_floor;
        const float p_floor      = dispatch.p_floor;
        const float wind_clamp_h = dispatch.wind_clamp_h;
        const float wind_clamp_v = dispatch.wind_clamp_v;

        if (!supports_acoustic_substeps_batched_dispatch())
            return false;
        if (n_substeps <= 0)
            return false;

        uint32_t     total_points = 0;
        VkDeviceSize field_bytes  = 0;
        if (!checked_field_dispatch(nr, nth, nz, total_points, field_bytes, "batched acoustic"))
            return false;
        const uint32_t workgroup_count = 1u + (total_points - 1u) / 64u;

        // 5 pool slots: u(0), v(1), w(2), rho(3), p(4)
        constexpr int kSlotCount = 5;
        int           slots[kSlotCount];
        if (!acquire_pool_slots(field_bytes, kSlotCount, slots))
        {
            log_vulkan_warning("failed to acquire pool slots for batched acoustic substeps");
            return false;
        }
        const int slot_u   = slots[0];
        const int slot_v   = slots[1];
        const int slot_w   = slots[2];
        const int slot_rho = slots[3];
        const int slot_p   = slots[4];

        const bool unified  = device_context_.has_unified_memory;
        auto       host_ptr = [&](int slot) -> void*
        { return unified ? buffer_pool_.device(slot).mapped : buffer_pool_.staging(slot).mapped; };

        // Upload all 5 fields once
        tracked_memcpy(host_ptr(slot_u), u, field_bytes);
        tracked_memcpy(host_ptr(slot_v), v, field_bytes);
        tracked_memcpy(host_ptr(slot_w), w, field_bytes);
        tracked_memcpy(host_ptr(slot_rho), rho, field_bytes);
        tracked_memcpy(host_ptr(slot_p), p, field_bytes);

        // Prepare push constants (same for every substep)
        AcousticPressurePushConstants pressure_pc{};
        pressure_pc.nr        = nr;
        pressure_pc.nth       = nth;
        pressure_pc.nz        = nz;
        pressure_pc.dr        = dr_val;
        pressure_pc.dtheta    = dtheta_val;
        pressure_pc.dz_val    = dz_val;
        pressure_pc.gamma_val = gamma_val;
        pressure_pc.dt_small  = dt_small;
        pressure_pc.rho_floor = rho_floor;
        pressure_pc.p_floor   = p_floor;

        AcousticMomentumPushConstants momentum_pc{};
        momentum_pc.nr           = nr;
        momentum_pc.nth          = nth;
        momentum_pc.nz           = nz;
        momentum_pc.dr           = dr_val;
        momentum_pc.dtheta       = dtheta_val;
        momentum_pc.dz_val       = dz_val;
        momentum_pc.dt_small     = dt_small;
        momentum_pc.wind_clamp_h = wind_clamp_h;
        momentum_pc.wind_clamp_v = wind_clamp_v;
        momentum_pc.padding      = 0.0f;

        // Cartesian collocated acoustic pipelines.
        auto& pressure_pipeline = active_acoustic_pressure_pipeline();
        auto& momentum_pipeline = active_acoustic_momentum_pipeline();

        // Set up descriptor sets once (shared slots for in-place operation)
        int pressure_bindings[7] = {slot_u, slot_v, slot_w, slot_rho, slot_p, slot_rho, slot_p};
        update_pooled_descriptor_set(pressure_pipeline, pressure_bindings, field_bytes, 7);

        int momentum_bindings[8] = {slot_rho, slot_p, slot_u, slot_v, slot_w, slot_u, slot_v, slot_w};
        update_pooled_descriptor_set(momentum_pipeline, momentum_bindings, field_bytes, 8);

        // Record command buffer with all N substeps
        vkResetCommandBuffer_(cmd_buf_, 0);
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (begin_profiled_command_buffer(begin_info) != VK_SUCCESS)
        {
            release_pool_slots(slots, kSlotCount);
            return false;
        }

        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;

        // ── H2D transfer ──
        if (unified)
        {
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }
        else
        {
            VkBufferCopy copy_region{};
            copy_region.size = field_bytes;
            for (int i = 0; i < kSlotCount; ++i)
            {
                tracked_cmd_copy_buffer(
                    buffer_pool_.staging(slots[i]).buffer, buffer_pool_.device(slots[i]).buffer, 1, &copy_region);
            }
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }

        // ── Record N acoustic substeps ──
        for (int n = 0; n < n_substeps; ++n)
        {
            // Pressure dispatch (reads u,v,w,rho,p; writes rho,p)
            vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pressure_pipeline.pipeline);
            vkCmdBindDescriptorSets_(cmd_buf_,
                                     VK_PIPELINE_BIND_POINT_COMPUTE,
                                     pressure_pipeline.pipeline_layout,
                                     0,
                                     1,
                                     &pressure_pipeline.descriptor_set,
                                     0,
                                     nullptr);
            vkCmdPushConstants_(cmd_buf_,
                                pressure_pipeline.pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT,
                                0,
                                sizeof(pressure_pc),
                                &pressure_pc);
            record_linear_group_dispatch(workgroup_count, KernelId::AcousticPressureCartesian);

            // Barrier: pressure writes -> momentum reads
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);

            // Momentum dispatch (reads rho,p [updated],u,v,w; writes u,v,w)
            vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, momentum_pipeline.pipeline);
            vkCmdBindDescriptorSets_(cmd_buf_,
                                     VK_PIPELINE_BIND_POINT_COMPUTE,
                                     momentum_pipeline.pipeline_layout,
                                     0,
                                     1,
                                     &momentum_pipeline.descriptor_set,
                                     0,
                                     nullptr);
            vkCmdPushConstants_(cmd_buf_,
                                momentum_pipeline.pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT,
                                0,
                                sizeof(momentum_pc),
                                &momentum_pc);
            record_linear_group_dispatch(workgroup_count, KernelId::AcousticMomentumCartesian);

            // Barrier between substeps (momentum writes -> next pressure reads)
            if (n < n_substeps - 1)
            {
                vkCmdPipelineBarrier_(cmd_buf_,
                                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                      0,
                                      1,
                                      &barrier,
                                      0,
                                      nullptr,
                                      0,
                                      nullptr);
            }
        }

        // ── D2H transfer (all 5 fields modified) ──
        if (unified)
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }
        else
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);

            VkBufferCopy copy_region{};
            copy_region.size = field_bytes;
            for (int i = 0; i < kSlotCount; ++i)
            {
                tracked_cmd_copy_buffer(
                    buffer_pool_.device(slots[i]).buffer, buffer_pool_.staging(slots[i]).buffer, 1, &copy_region);
            }
        }

        if (end_profiled_command_buffer() != VK_SUCCESS)
        {
            release_pool_slots(slots, kSlotCount);
            return false;
        }

        // Submit and wait (single submission for all N substeps)
        VkSubmitInfo submit_info{};
        submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers    = &cmd_buf_;
        if (reset_fence_and_submit("dispatch_acoustic_substeps_batched", submit_info) != VK_SUCCESS)
        {
            release_pool_slots(slots, kSlotCount);
            return false;
        }

        if (timed_wait_fence("dispatch_acoustic_substeps_batched") != VK_SUCCESS)
        {
            release_pool_slots(slots, kSlotCount);
            return false;
        }

        // Download all 5 fields once
        tracked_memcpy(u, host_ptr(slot_u), field_bytes);
        tracked_memcpy(v, host_ptr(slot_v), field_bytes);
        tracked_memcpy(w, host_ptr(slot_w), field_bytes);
        tracked_memcpy(rho, host_ptr(slot_rho), field_bytes);
        tracked_memcpy(p, host_ptr(slot_p), field_bytes);

        release_pool_slots(slots, kSlotCount);
        return true;
#else
        (void)dispatch;
        return false;
#endif
    }

    bool dispatch_vertical_flux(const VerticalFluxDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!pipeline(KernelId::TvdVerticalFlux).is_ready())
        {
            return false;
        }

        uint32_t     total_cells = 0;
        VkDeviceSize field_bytes = 0;
        if (!checked_field_dispatch(dispatch.nr, dispatch.nth, dispatch.nz, total_cells, field_bytes, "vertical flux"))
            return false;
        uint32_t     total_columns         = 0;
        VkDeviceSize columns_bytes_checked = 0;
        if (!checked_field_dispatch(
                dispatch.nr, dispatch.nth, 1, total_columns, columns_bytes_checked, "vertical flux columns"))
            return false;
        const VkDeviceSize dz_bytes = static_cast<VkDeviceSize>(dispatch.nz) * sizeof(float);

        // Ensure buffers are large enough — reallocate if dimensions changed
        if (!ensure_buffers(field_bytes, dz_bytes))
        {
            log_vulkan_warning("failed to allocate GPU buffers for dispatch");
            return false;
        }

        // Upload q to device
        tracked_memcpy(staging_q_.mapped, dispatch.q_data, field_bytes);
        // Upload w to device
        tracked_memcpy(staging_w_.mapped, dispatch.w_data, field_bytes);
        // Upload dz (convert from double to float)
        {
            auto*     dz_staging = static_cast<float*>(staging_dz_.mapped);
            const int count      = std::min(dispatch.dz_count, dispatch.nz);
            for (int k = 0; k < count; ++k)
            {
                dz_staging[k] = static_cast<float>(dispatch.dz_data[k]);
            }
        }
        // Zero the output staging buffer
        tracked_memset(staging_dqdt_.mapped, 0, field_bytes);

        // Record and submit command buffer
        vkResetCommandBuffer_(cmd_buf_, 0);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (begin_profiled_command_buffer(begin_info) != VK_SUCCESS)
        {
            log_vulkan_warning("vkBeginCommandBuffer failed");
            return false;
        }

        // Copy staging → device buffers
        VkBufferCopy copy_region{};
        copy_region.size = field_bytes;
        tracked_cmd_copy_buffer(staging_q_.buffer, device_q_.buffer, 1, &copy_region);
        tracked_cmd_copy_buffer(staging_w_.buffer, device_w_.buffer, 1, &copy_region);
        copy_region.size = dz_bytes;
        tracked_cmd_copy_buffer(staging_dz_.buffer, device_dz_.buffer, 1, &copy_region);
        copy_region.size = field_bytes;
        tracked_cmd_copy_buffer(staging_dqdt_.buffer, device_dqdt_.buffer, 1, &copy_region);

        // Barrier: transfer → compute
        VkMemoryBarrier barrier{};
        barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier_(cmd_buf_,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0,
                              1,
                              &barrier,
                              0,
                              nullptr,
                              0,
                              nullptr);

        // Bind pipeline and descriptor set
        vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline(KernelId::TvdVerticalFlux).pipeline);
        vkCmdBindDescriptorSets_(cmd_buf_,
                                 VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipeline(KernelId::TvdVerticalFlux).pipeline_layout,
                                 0,
                                 1,
                                 &pipeline(KernelId::TvdVerticalFlux).descriptor_set,
                                 0,
                                 nullptr);

        // Push constants
        TvdPushConstants pc{};
        pc.nr            = dispatch.nr;
        pc.nth           = dispatch.nth;
        pc.nz            = dispatch.nz;
        pc.limiter_id    = dispatch.limiter_id;
        pc.positivity    = dispatch.positivity ? 1 : 0;
        pc.positivity_dt = static_cast<float>(dispatch.positivity_dt);
        pc.cfl_target    = static_cast<float>(dispatch.cfl_target);
        pc.padding       = 0.0f;
        vkCmdPushConstants_(cmd_buf_,
                            pipeline(KernelId::TvdVerticalFlux).pipeline_layout,
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0,
                            sizeof(TvdPushConstants),
                            &pc);

        // Dispatch — one invocation per column, 64 per workgroup
        const uint32_t workgroup_count = 1u + (total_columns - 1u) / 64u;
        record_linear_group_dispatch(workgroup_count, KernelId::TvdVerticalFlux);

        // Barrier: compute → transfer
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier_(cmd_buf_,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0,
                              1,
                              &barrier,
                              0,
                              nullptr,
                              0,
                              nullptr);

        // Copy device dqdt → staging
        copy_region.size = field_bytes;
        tracked_cmd_copy_buffer(device_dqdt_.buffer, staging_dqdt_.buffer, 1, &copy_region);

        if (end_profiled_command_buffer() != VK_SUCCESS)
        {
            log_vulkan_warning("vkEndCommandBuffer failed");
            return false;
        }

        // Submit and wait
        VkSubmitInfo submit_info{};
        submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers    = &cmd_buf_;
        if (reset_fence_and_submit("vertical_flux", submit_info) != VK_SUCCESS)
        {
            log_vulkan_warning("vkQueueSubmit failed");
            return false;
        }

        VkResult wait_result = timed_wait_fence("vertical_flux");
        if (wait_result != VK_SUCCESS)
        {
            log_vulkan_warning(
                "vkWaitForFences failed or timed out (result=" + std::to_string(static_cast<int>(wait_result)) + ")");
            return false;
        }

        // Download results
        tracked_memcpy(dispatch.dqdt_data, staging_dqdt_.mapped, field_bytes);

        // CFL reduction on CPU (avoids a second shader pass)
        double max_cfl = 0.0;
        for (std::uint32_t idx = 0; idx < total_cells; ++idx)
        {
            const int    k      = static_cast<int>(idx % static_cast<std::uint32_t>(dispatch.nz));
            const double w_val  = static_cast<double>(dispatch.w_data[idx]);
            const double dz_val = (k < dispatch.dz_count) ? dispatch.dz_data[k] : 1.0;
            if (dz_val > 0.0)
            {
                const double local_cfl = std::abs(w_val) / dz_val;
                if (local_cfl > max_cfl)
                {
                    max_cfl = local_cfl;
                }
            }
        }
        if (dispatch.result != nullptr)
        {
            dispatch.result->max_cfl_z    = max_cfl;
            dispatch.result->suggested_dt = (max_cfl > 1.0e-30) ? (dispatch.cfl_target / max_cfl) : 1.0e30;
        }

        return true;
#else
        (void)dispatch;
        return false;
#endif
    }

    // Cartesian C-grid slow tendencies. 19-binding layout with a
    // push-constant struct whose dx->dr / dy->dtheta slots carry the grid
    // spacing and padding0 carries coriolis_f. Cartesian upwind advection,
    // perturbation-wind Coriolis, no 1/r, no axis.
    bool dispatch_cartesian_cgrid_slow_tendencies(const CartesianCgridSlowTendenciesDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        const float* u_data           = dispatch.u_data;
        const float* v_data           = dispatch.v_data;
        const float* w_data           = dispatch.w_data;
        const float* rho_data         = dispatch.rho_data;
        const float* p_data           = dispatch.p_data;
        const float* rho0_base_data   = dispatch.rho0_base_data;
        const int    rho0_base_len    = dispatch.rho0_base_len;
        const float* qv0_base_data    = dispatch.qv0_base_data;
        const int    qv0_base_len     = dispatch.qv0_base_len;
        const float* u0_base_data     = dispatch.u0_base_data;
        const float* v0_base_data     = dispatch.v0_base_data;
        const int    wind0_base_len   = dispatch.wind0_base_len;
        const float* qv_data          = dispatch.qv_data;
        const float* loading_data     = dispatch.loading_data;
        float*       du_dt_data       = dispatch.du_dt_data;
        float*       dv_dt_data       = dispatch.dv_dt_data;
        float*       dw_dt_data       = dispatch.dw_dt_data;
        float*       drho_dt_data     = dispatch.drho_dt_data;
        float*       dp_dt_data       = dispatch.dp_dt_data;
        const int    nr               = dispatch.nr;
        const int    nth              = dispatch.nth;
        const int    nz               = dispatch.nz;
        const float  dx_val           = dispatch.dx;
        const float  dy_val           = dispatch.dy;
        const float  dz_val           = dispatch.dz;
        const float  g_val            = dispatch.g;
        const float  coriolis_f_val   = dispatch.coriolis_f_val;
        const float  coriolis_beta    = dispatch.coriolis_beta;
        const bool   has_moisture     = dispatch.has_moisture;
        const bool   has_loading      = dispatch.has_loading;
        const bool   export_advection = dispatch.du_advection_dt_data != nullptr ||
                                        dispatch.dv_advection_dt_data != nullptr ||
                                        dispatch.dw_advection_dt_data != nullptr;

        if (!pipeline(KernelId::CartesianCgridSlowTendencies).is_ready())
        {
            return false;
        }
        if (!std::isfinite(coriolis_f_val) || !std::isfinite(coriolis_beta))
            return false;
        if (rho0_base_data == nullptr || rho0_base_len < nz)
        {
            return false;
        }

        if ((has_moisture && qv_data == nullptr) || (has_loading && loading_data == nullptr))
        {
            return false;
        }
        if (export_advection && (dispatch.du_advection_dt_data == nullptr || dispatch.dv_advection_dt_data == nullptr ||
                                 dispatch.dw_advection_dt_data == nullptr))
        {
            return false;
        }

        // Base-state profiles are already compact. Retained scratch is used
        // only when a caller supplies a short/absent optional profile.
        const float  zero_scalar = 0.0f;
        const float* qv0_upload  = &zero_scalar;
        if (has_moisture && qv0_base_data != nullptr && qv0_base_len >= nz)
        {
            qv0_upload = qv0_base_data;
        }
        else if (has_moisture)
        {
            qv0_upload_scratch_.assign(static_cast<std::size_t>(nz), 0.0f);
            const int qv0_n = qv0_base_data != nullptr ? std::max(0, std::min(qv0_base_len, nz)) : 0;
            if (qv0_n > 0)
            {
                std::copy_n(qv0_base_data, static_cast<std::size_t>(qv0_n), qv0_upload_scratch_.data());
            }
            qv0_upload = qv0_upload_scratch_.data();
        }

        // Environmental wind profiles for the perturbation-wind Coriolis
        // (SCIENCE.md M1); complete profiles upload directly.
        const float* u0_upload = u0_base_data;
        const float* v0_upload = v0_base_data;
        if (u0_base_data == nullptr || v0_base_data == nullptr || wind0_base_len < nz)
        {
            u0_upload_scratch_.assign(static_cast<std::size_t>(nz), 0.0f);
            v0_upload_scratch_.assign(static_cast<std::size_t>(nz), 0.0f);
            if (u0_base_data != nullptr && v0_base_data != nullptr)
            {
                const int wind0_n = std::max(0, std::min(wind0_base_len, nz));
                if (wind0_n > 0)
                {
                    std::copy_n(u0_base_data, static_cast<std::size_t>(wind0_n), u0_upload_scratch_.data());
                    std::copy_n(v0_base_data, static_cast<std::size_t>(wind0_n), v0_upload_scratch_.data());
                }
            }
            u0_upload = u0_upload_scratch_.data();
            v0_upload = v0_upload_scratch_.data();
        }

        SupercellCgridSlowTendenciesPushConstants pc{};
        pc.nr                  = nr;
        pc.nth                 = nth;
        pc.nz                  = nz;
        pc.dr                  = dx_val;
        pc.dtheta              = dy_val;
        pc.dz_val              = dz_val;
        pc.g                   = g_val;
        pc.has_moisture        = has_moisture ? 1 : 0;
        pc.has_loading         = has_loading ? 1 : 0;
        pc.padding0            = coriolis_f_val; // coriolis_f in the cartesian shader
        pc.padding1            = export_advection ? 1.0f : 0.0f;
        pc.coriolis_beta       = coriolis_beta;
        pc.frame_velocity_u_ms = dispatch.frame_velocity_u_ms;
        pc.frame_velocity_v_ms = dispatch.frame_velocity_v_ms;

        const VkDeviceSize prof_bytes          = static_cast<VkDeviceSize>(nz) * sizeof(float);
        uint32_t           total_points        = 0;
        VkDeviceSize       field_bytes_checked = 0;
        if (!checked_field_dispatch(nr, nth, nz, total_points, field_bytes_checked, "c-grid slow tendencies"))
            return false;
        const float*       inputs[11]      = {u_data,
                                              v_data,
                                              w_data,
                                              rho_data,
                                              p_data,
                                              rho0_base_data,
                                              qv0_upload,
                                              u0_upload,
                                              v0_upload,
                                              has_moisture ? qv_data : &zero_scalar,
                                              has_loading ? loading_data : &zero_scalar};
        const VkDeviceSize input_bytes[11] = {0,
                                              0,
                                              0,
                                              0,
                                              0,
                                              prof_bytes,
                                              has_moisture ? prof_bytes : sizeof(float),
                                              prof_bytes,
                                              prof_bytes,
                                              has_moisture ? 0 : sizeof(float),
                                              has_loading ? 0 : sizeof(float)};
        float*             outputs[8]      = {du_dt_data,
                                              dv_dt_data,
                                              dw_dt_data,
                                              drho_dt_data,
                                              dp_dt_data,
                                              dispatch.du_advection_dt_data,
                                              dispatch.dv_advection_dt_data,
                                              dispatch.dw_advection_dt_data};
        // Every invocation initializes every tendency before applying its
        // component-specific active-domain equation. Boundary and inactive
        // storage are therefore exact zero and no prior host output value is
        // observable by the shader.
        static constexpr bool kOutputsWriteOnly[8] = {true, true, true, true, true, true, true, true};
        // Only the orchestration-owned Heun pair may publish/reuse these
        // bindings. Its predictor changes u/v/w/p, but not rho or qv. Loading
        // and compact profiles remain transient: retaining them would trade a
        // full persistent allocation (or four pool slots) for a narrow upload
        // saving and would worsen the engine's memory Pareto point.
        const char* input_names[11] = {nullptr,
                                       nullptr,
                                       nullptr,
                                       dispatch.slow_rk2_invariant_input_residency ? "rho" : nullptr,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       dispatch.slow_rk2_invariant_input_residency && has_moisture ? "qv" : nullptr,
                                       nullptr};

        // In ordinary runs bindings 16..18 are never written. Alias them to
        // the total-tendency output slots so the expanded shader ABI costs no
        // extra allocations, uploads, or downloads. Report-bearing runs own
        // three additional write-only outputs.
        static constexpr std::uint8_t kDescriptorOrderNoAttribution[19] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 11, 12, 13,
        };
        static constexpr std::uint8_t kDescriptorOrderWithAttribution[19] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        };
        return dispatch_multi_field_kernel(pipeline(KernelId::CartesianCgridSlowTendencies),
                                           KernelId::CartesianCgridSlowTendencies,
                                           &pc,
                                           inputs,
                                           11,
                                           outputs,
                                           export_advection ? 8 : 5,
                                           nr,
                                           nth,
                                           nz,
                                           total_points,
                                           input_bytes,
                                           input_names,
                                           kOutputsWriteOnly,
                                           true,
                                           export_advection ? kDescriptorOrderWithAttribution
                                                            : kDescriptorOrderNoAttribution,
                                           nullptr,
                                           nullptr,
                                           19);
#else
        (void)dispatch;
        return false;
#endif
    }

    bool dispatch_diffusion(const DiffusionDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!pipeline(KernelId::Diffusion).is_ready())
        {
            return false;
        }

        if (dispatch.field == nullptr || dispatch.rho == nullptr || dispatch.dz == nullptr ||
            dispatch.tendency == nullptr || dispatch.nx < 3 || dispatch.ny < 3 || dispatch.nz < 2 ||
            !std::isfinite(dispatch.dx) || dispatch.dx <= 0.0f || !std::isfinite(dispatch.dy) || dispatch.dy <= 0.0f ||
            !std::isfinite(dispatch.K_h) || dispatch.K_h < 0.0f || !std::isfinite(dispatch.K_v) || dispatch.K_v < 0.0f)
        {
            return false;
        }
        const int location = static_cast<int>(dispatch.location);
        if (location < static_cast<int>(DiffusionFieldLocation::Scalar) ||
            location > static_cast<int>(DiffusionFieldLocation::WFace))
        {
            return false;
        }

        DiffusionPushConstants pc{};
        pc.nx         = dispatch.nx;
        pc.ny         = dispatch.ny;
        pc.nz         = dispatch.nz;
        pc.location   = location;
        pc.c_grid     = dispatch.c_grid ? 1 : 0;
        pc.periodic_x = dispatch.periodic_x ? 1 : 0;
        pc.periodic_y = dispatch.periodic_y ? 1 : 0;
        pc.padding_i  = dispatch.vertical_K == nullptr ? 0 : 1;
        pc.inv_dx2    = 1.0f / (dispatch.dx * dispatch.dx);
        pc.inv_dy2    = 1.0f / (dispatch.dy * dispatch.dy);
        pc.K_h        = dispatch.K_h;
        pc.K_v        = dispatch.K_v;

        uint32_t     total_points        = 0;
        VkDeviceSize field_bytes_checked = 0;
        if (!checked_field_dispatch(dispatch.nx,
                                    dispatch.ny,
                                    dispatch.nz,
                                    total_points,
                                    field_bytes_checked,
                                    "diffusion finite-volume field"))
            return false;
        const float*       inputs[4]            = {dispatch.field, dispatch.rho, dispatch.dz,
                                                   dispatch.vertical_K == nullptr
                                                       ? dispatch.rho : dispatch.vertical_K};
        const VkDeviceSize input_bytes[4]       = {0, 0, static_cast<VkDeviceSize>(dispatch.nz) * sizeof(float), 0};
        float*             outputs[1]           = {dispatch.tendency};
        const bool         output_write_only[1] = {true};
        static constexpr std::uint8_t descriptor_order[5] = {0, 1, 2, 4, 3};
        return dispatch_multi_field_kernel(pipeline(KernelId::Diffusion),
                                           KernelId::Diffusion,
                                           &pc,
                                           inputs,
                                           4,
                                           outputs,
                                           1,
                                           dispatch.nx,
                                           dispatch.ny,
                                           dispatch.nz,
                                           total_points,
                                           input_bytes,
                                           nullptr,
                                           output_write_only,
                                           true,
                                           descriptor_order);
#else
        (void)dispatch;
        return false;
#endif
    }

    bool dispatch_diffusion_batch(const DiffusionBatchDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        // Diffusion intentionally retains its qualified five-target batch;
        // the larger descriptor array belongs to the 32-owner recycle transaction.
        constexpr int         kMaximumTargets    = 5;
        ComputePipelineState& diffusion_pipeline = pipeline(KernelId::Diffusion);
        if (!diffusion_pipeline.is_ready() || dispatch.fields == nullptr || dispatch.tendencies == nullptr ||
            dispatch.locations == nullptr || dispatch.rho == nullptr || dispatch.dz == nullptr || dispatch.count < 1 ||
            dispatch.count > kMaximumTargets ||
            diffusion_pipeline.descriptor_set_count < static_cast<uint32_t>(dispatch.count + 1) || dispatch.nx < 3 ||
            dispatch.ny < 3 || dispatch.nz < 2 || !std::isfinite(dispatch.dx) || dispatch.dx <= 0.0f ||
            !std::isfinite(dispatch.dy) || dispatch.dy <= 0.0f || !std::isfinite(dispatch.K_h) || dispatch.K_h < 0.0f ||
            !std::isfinite(dispatch.K_v) || dispatch.K_v < 0.0f)
        {
            return false;
        }
        for (int target = 0; target < dispatch.count; ++target)
        {
            const int location = static_cast<int>(dispatch.locations[target]);
            if (dispatch.fields[target] == nullptr || dispatch.tendencies[target] == nullptr ||
                location < static_cast<int>(DiffusionFieldLocation::Scalar) ||
                location > static_cast<int>(DiffusionFieldLocation::WFace))
            {
                return false;
            }
        }

        uint32_t     total_points = 0;
        VkDeviceSize field_bytes  = 0;
        if (!checked_field_dispatch(
                dispatch.nx, dispatch.ny, dispatch.nz, total_points, field_bytes, "diffusion finite-volume batch"))
        {
            return false;
        }
        const std::size_t cell_count = static_cast<std::size_t>(total_points);
        for (int k = 0; k < dispatch.nz; ++k)
            if (!std::isfinite(dispatch.dz[k]) || dispatch.dz[k] <= 0.0f)
                return false;
        for (std::size_t index = 0; index < cell_count; ++index)
        {
            if (!std::isfinite(dispatch.rho[index]) || dispatch.rho[index] <= 0.0f)
                return false;
            for (int target = 0; target < dispatch.count; ++target)
            {
                if (!std::isfinite(dispatch.fields[target][index]))
                    return false;
                if (dispatch.vertical_K != nullptr &&
                    (dispatch.vertical_K[target] == nullptr ||
                     !std::isfinite(dispatch.vertical_K[target][index]) ||
                     dispatch.vertical_K[target][index] < 0.0f))
                    return false;
            }
        }
        const VkDeviceSize dz_bytes         = static_cast<VkDeviceSize>(dispatch.nz) * sizeof(float);
        constexpr int      kMaximumSlots    = 3 * kMaximumTargets + 2;
        const int          rho_slot_logical = dispatch.count;
        const int          dz_slot_logical  = dispatch.count + 1;
        const int          k_start           = dispatch.count + 2;
        const int          output_start      = 2 * dispatch.count + 2;
        const int          slot_count        = 3 * dispatch.count + 2;
        VkDeviceSize       slot_sizes[kMaximumSlots];
        for (int target = 0; target < dispatch.count; ++target)
        {
            slot_sizes[target]                = field_bytes;
            slot_sizes[k_start + target]      = field_bytes;
            slot_sizes[output_start + target] = field_bytes;
        }
        slot_sizes[rho_slot_logical] = field_bytes;
        slot_sizes[dz_slot_logical]  = dz_bytes;
        int slots[kMaximumSlots];
        if (!buffer_pool_.acquire_many(slot_sizes, static_cast<std::size_t>(slot_count), slots))
        {
            log_vulkan_warning("failed to acquire pooled diffusion batch buffers");
            return false;
        }
        const auto release_slots = [&]() { release_pool_slots(slots, slot_count); };

        const bool unified  = device_context_.has_unified_memory;
        const auto host_ptr = [&](int logical) -> void*
        { return unified ? buffer_pool_.device(slots[logical]).mapped : buffer_pool_.staging(slots[logical]).mapped; };
        for (int target = 0; target < dispatch.count; ++target)
        {
            tracked_memcpy(host_ptr(target), dispatch.fields[target], field_bytes);
            compute_telemetry_.note_writeonly_output_upload_avoided(static_cast<std::uint64_t>(field_bytes));
        }
        tracked_memcpy(host_ptr(rho_slot_logical), dispatch.rho, field_bytes);
        tracked_memcpy(host_ptr(dz_slot_logical), dispatch.dz, dz_bytes);
        for (int target = 0; target < dispatch.count; ++target)
            tracked_memcpy(host_ptr(k_start + target),
                           dispatch.vertical_K == nullptr
                               ? dispatch.rho : dispatch.vertical_K[target],
                           field_bytes);

        // Each descriptor set binds one qualified single-field diffusion
        // operation. Sets are immutable for the duration of the command
        // buffer, allowing independent targets to share one submission.
        for (int target = 0; target < dispatch.count; ++target)
        {
            const int logical[5] = {target, rho_slot_logical, dz_slot_logical,
                                    output_start + target, k_start + target};
            const VkDeviceSize ranges[5] = {field_bytes, field_bytes, dz_bytes,
                                            field_bytes, field_bytes};
            std::uintptr_t handles[5]{};
            std::size_t cached_ranges[5]{};
            for (int binding = 0; binding < 5; ++binding)
            {
                const VkBuffer buffer = buffer_pool_.device(slots[logical[binding]]).buffer;
                static_assert(sizeof(buffer) <= sizeof(std::uintptr_t), "VkBuffer cache key must fit uintptr_t");
                std::memcpy(&handles[binding], &buffer, sizeof(buffer));
                cached_ranges[binding] = static_cast<std::size_t>(ranges[binding]);
            }
            const bool descriptor_cached =
                diffusion_pipeline.descriptor_set_caches[static_cast<std::size_t>(target + 1)].matches_or_update(
                    handles, cached_ranges, 5);
            compute_telemetry_.note_descriptor_lookup(descriptor_cached);
            if (descriptor_cached)
                continue;

            VkDescriptorBufferInfo infos[5]{};
            VkWriteDescriptorSet   writes[5]{};
            for (int binding = 0; binding < 5; ++binding)
            {
                infos[binding]             = {buffer_pool_.device(slots[logical[binding]]).buffer, 0, ranges[binding]};
                writes[binding].sType      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[binding].dstSet     = diffusion_pipeline.descriptor_sets[static_cast<std::size_t>(target + 1)];
                writes[binding].dstBinding = static_cast<uint32_t>(binding);
                writes[binding].descriptorCount = 1;
                writes[binding].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[binding].pBufferInfo     = &infos[binding];
            }
            vkUpdateDescriptorSets_(device_context_.device, 5, writes, 0, nullptr);
        }

        vkResetCommandBuffer_(cmd_buf_, 0);
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (begin_profiled_command_buffer(begin_info) != VK_SUCCESS)
        {
            release_slots();
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
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }
        else
        {
            for (int logical = 0; logical < output_start; ++logical)
            {
                VkBufferCopy copy{};
                copy.size = slot_sizes[logical];
                tracked_cmd_copy_buffer(
                    buffer_pool_.staging(slots[logical]).buffer, buffer_pool_.device(slots[logical]).buffer, 1, &copy);
            }
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }

        vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, diffusion_pipeline.pipeline);
        const uint32_t workgroup_count = 1u + (total_points - 1u) / 64u;
        for (int target = 0; target < dispatch.count; ++target)
        {
            const VkDescriptorSet descriptor_set =
                diffusion_pipeline.descriptor_sets[static_cast<std::size_t>(target + 1)];
            vkCmdBindDescriptorSets_(cmd_buf_,
                                     VK_PIPELINE_BIND_POINT_COMPUTE,
                                     diffusion_pipeline.pipeline_layout,
                                     0,
                                     1,
                                     &descriptor_set,
                                     0,
                                     nullptr);

            DiffusionPushConstants pc{};
            pc.nx         = dispatch.nx;
            pc.ny         = dispatch.ny;
            pc.nz         = dispatch.nz;
            pc.location   = static_cast<int>(dispatch.locations[target]);
            pc.c_grid     = dispatch.c_grid ? 1 : 0;
            pc.periodic_x = dispatch.periodic_x ? 1 : 0;
            pc.periodic_y = dispatch.periodic_y ? 1 : 0;
            pc.padding_i  = dispatch.vertical_K == nullptr ? 0 : 1;
            pc.inv_dx2    = 1.0f / (dispatch.dx * dispatch.dx);
            pc.inv_dy2    = 1.0f / (dispatch.dy * dispatch.dy);
            pc.K_h        = dispatch.K_h;
            pc.K_v        = dispatch.K_v;
            vkCmdPushConstants_(cmd_buf_,
                                diffusion_pipeline.pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT,
                                0,
                                diffusion_pipeline.push_constant_size,
                                &pc);
            record_linear_group_dispatch(workgroup_count, KernelId::Diffusion);
        }

        if (unified)
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }
        else
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
            for (int target = 0; target < dispatch.count; ++target)
            {
                VkBufferCopy copy{};
                copy.size         = field_bytes;
                const int logical = output_start + target;
                tracked_cmd_copy_buffer(
                    buffer_pool_.device(slots[logical]).buffer, buffer_pool_.staging(slots[logical]).buffer, 1, &copy);
            }
        }

        if (end_profiled_command_buffer() != VK_SUCCESS)
        {
            release_slots();
            return false;
        }
        VkSubmitInfo submit{};
        submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers    = &cmd_buf_;
        if (reset_fence_and_submit("diffusion_batch", submit) != VK_SUCCESS ||
            timed_wait_fence("diffusion_batch") != VK_SUCCESS)
        {
            release_slots();
            return false;
        }

        for (int target = 0; target < dispatch.count; ++target)
        {
            const int   logical = output_start + target;
            const void* source =
                unified ? buffer_pool_.device(slots[logical]).mapped : buffer_pool_.staging(slots[logical]).mapped;
            tracked_memcpy(dispatch.tendencies[target], source, field_bytes);
        }
        release_slots();
        return true;
#else
        (void)dispatch;
        return false;
#endif
    }

    // ── Cartesian tendencies (13 state/profile inputs, 5 outputs) ──

    bool dispatch_cartesian_tendencies(const CartesianTendenciesDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        const float* u_x_data       = dispatch.u_x_data;
        const float* u_y_data       = dispatch.u_y_data;
        const float* w_data         = dispatch.w_data;
        const float* rho_data       = dispatch.rho_data;
        const float* p_data         = dispatch.p_data;
        const float* theta_data     = dispatch.theta_data;
        const float* p0_base_data   = dispatch.p0_base_data;
        const float* rho0_base_data = dispatch.rho0_base_data;
        const float* loading_data   = dispatch.loading_data;
        const float* u0_base_data   = dispatch.u0_base_data;
        const float* v0_base_data   = dispatch.v0_base_data;
        const float* qv_data        = dispatch.qv_data;
        const float* qv0_base_data  = dispatch.qv0_base_data;
        float*       du_x_dt_data   = dispatch.du_x_dt_data;
        float*       du_y_dt_data   = dispatch.du_y_dt_data;
        float*       dw_dt_data     = dispatch.dw_dt_data;
        float*       drho_dt_data   = dispatch.drho_dt_data;
        float*       dp_dt_data     = dispatch.dp_dt_data;
        const int    nr             = dispatch.nr;
        const int    nth            = dispatch.nth;
        const int    nz             = dispatch.nz;
        const float  dx_val         = dispatch.dx;
        const float  dy_val         = dispatch.dy;
        const float  dz_val         = dispatch.dz;
        const float  g_val          = dispatch.g;
        const float  gamma_val      = dispatch.gamma_val;
        const float  coriolis_f_val = dispatch.coriolis_f_val;
        const float  coriolis_beta  = dispatch.coriolis_beta;
        const bool   has_moisture   = dispatch.has_moisture;

        if (!pipeline(KernelId::CartesianTendencies).is_ready())
        {
            return false;
        }
        if (u_x_data == nullptr || u_y_data == nullptr || w_data == nullptr || rho_data == nullptr ||
            p_data == nullptr || theta_data == nullptr || p0_base_data == nullptr || rho0_base_data == nullptr ||
            loading_data == nullptr || u0_base_data == nullptr || v0_base_data == nullptr || du_x_dt_data == nullptr ||
            du_y_dt_data == nullptr || dw_dt_data == nullptr || drho_dt_data == nullptr || dp_dt_data == nullptr)
        {
            return false;
        }
        if (has_moisture && (qv_data == nullptr || qv0_base_data == nullptr))
        {
            return false;
        }
        const auto valid_lateral_bc = [](int value) { return value >= 0 && value <= 5; };
        if (nr < 3 || nth < 3 || nz < 3 || !valid_lateral_bc(dispatch.lateral_bc_west) ||
            !valid_lateral_bc(dispatch.lateral_bc_east) || !valid_lateral_bc(dispatch.lateral_bc_south) ||
            !valid_lateral_bc(dispatch.lateral_bc_north) || dispatch.top_bc < 0 || dispatch.top_bc > 2)
        {
            return false;
        }
        if (!std::isfinite(coriolis_f_val) || !std::isfinite(coriolis_beta))
            return false;

        CartesianTendenciesPushConstants pc{};
        pc.nr               = nr;
        pc.nth              = nth;
        pc.nz               = nz;
        pc.dx               = dx_val;
        pc.dy               = dy_val;
        pc.dz               = dz_val;
        pc.g                = g_val;
        pc.gamma_val        = gamma_val;
        pc.coriolis_f       = coriolis_f_val;
        pc.coriolis_beta    = coriolis_beta;
        pc.lateral_bc_west  = dispatch.lateral_bc_west;
        pc.lateral_bc_east  = dispatch.lateral_bc_east;
        pc.lateral_bc_south = dispatch.lateral_bc_south;
        pc.lateral_bc_north = dispatch.lateral_bc_north;
        pc.top_bc           = dispatch.top_bc;
        pc.has_moisture     = (has_moisture ? 1 : 0) |
                              (dispatch.split_slow_only ? 2 : 0);

        // The shader indexes compact base-state profiles by k. Upload exactly
        // NZ values into field-capacity slots; never materialize padded 3-D
        // host copies.
        uint32_t     total_cells_u32     = 0;
        VkDeviceSize field_bytes_checked = 0;
        if (!checked_field_dispatch(nr, nth, nz, total_cells_u32, field_bytes_checked, "Cartesian tendencies"))
            return false;
        const VkDeviceSize profile_bytes = static_cast<VkDeviceSize>(nz) * sizeof(float);
        for (uint32_t n = 0; n < total_cells_u32; ++n)
        {
            if (!std::isfinite(u_x_data[n]) || !std::isfinite(u_y_data[n]) ||
                !std::isfinite(w_data[n]) || !std::isfinite(rho_data[n]) ||
                rho_data[n] <= 0.0f || !std::isfinite(p_data[n]) ||
                !std::isfinite(theta_data[n]) || !std::isfinite(loading_data[n]) ||
                loading_data[n] < 0.0f ||
                (has_moisture && (!std::isfinite(qv_data[n]) || qv_data[n] < 0.0f)))
            {
                return false;
            }
        }
        for (int k = 0; k < nz; ++k)
        {
            if (!std::isfinite(p0_base_data[k]) || !std::isfinite(rho0_base_data[k]) ||
                rho0_base_data[k] <= 0.0f || !std::isfinite(u0_base_data[k]) ||
                !std::isfinite(v0_base_data[k]) ||
                (has_moisture && (!std::isfinite(qv0_base_data[k]) || qv0_base_data[k] < 0.0f)))
            {
                return false;
            }
        }

        const float           zero_scalar          = 0.0f;
        const float*          inputs[13]           = {u_x_data,
                                                      u_y_data,
                                                      w_data,
                                                      rho_data,
                                                      p_data,
                                                      theta_data,
                                                      p0_base_data,
                                                      rho0_base_data,
                                                      loading_data,
                                                      u0_base_data,
                                                      v0_base_data,
                                                      has_moisture ? qv_data : &zero_scalar,
                                                      has_moisture ? qv0_base_data : &zero_scalar};
        const VkDeviceSize    input_bytes[13]      = {0,
                                                      0,
                                                      0,
                                                      0,
                                                      0,
                                                      0,
                                                      profile_bytes,
                                                      profile_bytes,
                                                      0,
                                                      profile_bytes,
                                                      profile_bytes,
                                                      has_moisture ? 0 : sizeof(float),
                                                      has_moisture ? profile_bytes : sizeof(float)};
        std::array<std::vector<float>, 5> candidates;
        for (auto& field : candidates) field.resize(total_cells_u32);
        float* outputs[5] = {candidates[0].data(), candidates[1].data(),
                             candidates[2].data(), candidates[3].data(),
                             candidates[4].data()};
        static constexpr bool kOutputsWriteOnly[5] = {true, true, true, true, true};
        // Shader ABI bindings are:
        //   state/profile inputs 0..7, outputs 8..12,
        //   loading/u0/v0/qv/qv0 inputs 13..17.
        // Logical dispatch storage remains inputs 0..12 then outputs 13..17.
        static constexpr std::uint8_t kDescriptorLogicalOrder[18] = {
            0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 17, 8, 9, 10, 11, 12};

        if (!dispatch_multi_field_kernel(pipeline(KernelId::CartesianTendencies),
                                         KernelId::CartesianTendencies,
                                         &pc,
                                         inputs,
                                         13,
                                         outputs,
                                         5,
                                         nr,
                                         nth,
                                         nz,
                                         total_cells_u32,
                                         input_bytes,
                                         nullptr,
                                         kOutputsWriteOnly,
                                         true,
                                         kDescriptorLogicalOrder))
        {
            return false;
        }
        for (const auto& field : candidates)
            for (float value : field)
                if (!std::isfinite(value)) return false;
        float* destinations[5] = {
            du_x_dt_data, du_y_dt_data, dw_dt_data, drho_dt_data, dp_dt_data};
        for (int field = 0; field < 5; ++field)
        {
            std::memcpy(destinations[field], candidates[field].data(),
                        static_cast<std::size_t>(field_bytes_checked));
        }
        return true;
#else
        (void)dispatch;
        return false;
#endif
    }

    // ── Cartesian x-advection ──

    bool dispatch_advection_x(const ScalarAdvectionDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!pipeline(KernelId::AdvectX).is_ready())
        {
            return false;
        }

        AdvectionXPushConstants pc{};
        pc.nr         = dispatch.nr;
        pc.nth        = dispatch.nth;
        pc.nz         = dispatch.nz;
        pc.dx         = dispatch.primary_spacing;
        pc.dt         = dispatch.dt;
        pc.padding[0] = pc.padding[1] = pc.padding[2] = 0.0f;

        // x-advection interior: i=1..nr-2, all j, k=1..nz-2
        uint32_t     interior_points        = 0;
        VkDeviceSize interior_bytes_checked = 0;
        if (!checked_field_dispatch(dispatch.nr - 2,
                                    dispatch.nth,
                                    dispatch.nz - 2,
                                    interior_points,
                                    interior_bytes_checked,
                                    "advection-x interior"))
            return false;

        return dispatch_field3_kernel(pipeline(KernelId::AdvectX),
                                      KernelId::AdvectX,
                                      &pc,
                                      dispatch.src,
                                      dispatch.velocity,
                                      dispatch.dst,
                                      dispatch.nr,
                                      dispatch.nth,
                                      dispatch.nz,
                                      2,
                                      interior_points);
#else
        (void)dispatch;
        return false;
#endif
    }

    // ── Cartesian y-advection ──

    bool dispatch_advection_y(const ScalarAdvectionDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        if (!pipeline(KernelId::AdvectY).is_ready())
        {
            return false;
        }

        AdvectionYPushConstants pc{};
        pc.nr         = dispatch.nr;
        pc.nth        = dispatch.nth;
        pc.nz         = dispatch.nz;
        pc.dy         = dispatch.primary_spacing;
        pc.dt         = dispatch.dt;
        pc.padding[0] = pc.padding[1] = pc.padding[2] = 0.0f;

        // y-advection interior: all i, j=1..nth-2, k=1..nz-2
        uint32_t     interior_points        = 0;
        VkDeviceSize interior_bytes_checked = 0;
        if (!checked_field_dispatch(dispatch.nr,
                                    dispatch.nth - 2,
                                    dispatch.nz - 2,
                                    interior_points,
                                    interior_bytes_checked,
                                    "advection-y interior"))
            return false;

        return dispatch_field3_kernel(pipeline(KernelId::AdvectY),
                                      KernelId::AdvectY,
                                      &pc,
                                      dispatch.src,
                                      dispatch.velocity,
                                      dispatch.dst,
                                      dispatch.nr,
                                      dispatch.nth,
                                      dispatch.nz,
                                      2,
                                      interior_points);
#else
        (void)dispatch;
        return false;
#endif
    }

    bool dispatch_kessler_pointwise(const KesslerPointwiseDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        const auto* temperature_data       = dispatch.temperature_data;
        const auto* p_data                 = dispatch.p_data;
        const auto* qv_data                = dispatch.qv_data;
        const auto* qc_data                = dispatch.qc_data;
        const auto* qr_data                = dispatch.qr_data;
        const auto* rho_d_data             = dispatch.rho_d_data;
        const auto* qh_data                = dispatch.qh_data;
        auto*       dtheta_dt_data         = dispatch.dtheta_dt_data;
        auto*       dqv_dt_data            = dispatch.dqv_dt_data;
        auto*       dqc_dt_data            = dispatch.dqc_dt_data;
        auto*       dqr_dt_data            = dispatch.dqr_dt_data;
        auto*       dqg_dt_data            = dispatch.dqg_dt_data;
        auto*       dqh_dt_data            = dispatch.dqh_dt_data;
        const int   nr                     = dispatch.nr;
        const int   nth                    = dispatch.nth;
        const int   nz                     = dispatch.nz;
        const int   i_begin                = dispatch.i_begin;
        const int   i_end                  = dispatch.i_end;
        const int   j_begin                = dispatch.j_begin;
        const int   j_end                  = dispatch.j_end;
        const int   k_begin                = dispatch.k_begin;
        const int   k_end                  = dispatch.k_end;
        const float qc0                    = dispatch.qc0;
        const float c_auto_val             = dispatch.c_auto;
        const float c_accr_val             = dispatch.c_accr;
        const float evaporation_efficiency = dispatch.evaporation_efficiency;
        const float Lv_cp                  = dispatch.Lv_cp;
        const float T0_val                 = dispatch.T0;
        const float dt_step                = dispatch.dt_step;

        if (!pipeline(KernelId::KesslerPointwise).is_ready())
        {
            return false;
        }

        KesslerPointwisePushConstants pc{};
        pc.nr                     = nr;
        pc.nth                    = nth;
        pc.nz                     = nz;
        pc.i_begin                = i_begin;
        pc.i_end                  = i_end;
        pc.j_begin                = j_begin;
        pc.j_end                  = j_end;
        pc.k_begin                = k_begin;
        pc.k_end                  = k_end;
        pc.qc0                    = qc0;
        pc.c_auto                 = c_auto_val;
        pc.c_accr                 = c_accr_val;
        pc.evaporation_efficiency = evaporation_efficiency;
        pc.Lv_cp                  = Lv_cp;
        pc.T0                     = T0_val;
        pc.dt_step                = dt_step;

        if (i_begin < 0 || i_begin >= i_end || i_end > nr || j_begin < 0 || j_begin >= j_end || j_end > nth ||
            k_begin < 0 || k_begin >= k_end || k_end > nz)
        {
            return false;
        }

        uint32_t     total_cells         = 0;
        VkDeviceSize field_bytes_checked = 0;
        if (!checked_field_dispatch(nr, nth, nz, total_cells, field_bytes_checked, "Kessler pointwise"))
            return false;

        const float* inputs[7]  = {temperature_data, p_data, qv_data, qc_data, qr_data, rho_d_data, qh_data};
        float*       outputs[6] = {dtheta_dt_data, dqv_dt_data, dqc_dt_data, dqr_dt_data, dqg_dt_data, dqh_dt_data};
        const bool   output_write_only[6] = {true, true, true, true, true, true};
        // Kessler reads but does not alter these precipitating fields. Live
        // dry density is authoritative prognostic state and may reuse the
        // current resident rho allocation across pointwise/sedimentation.
        // qv/qc deliberately remain unnamed: the scheme feeds saturation-
        // adjusted working copies here, not the authoritative prognostic
        // arrays consumed by the final tendency apply.  The exact pointer
        // ownership check would reject them there, but publishing temporary
        // state under a prognostic field name is misleading and retains
        // useless slots.
        const char* input_names[7] = {nullptr, nullptr, nullptr, nullptr, "qr", "rho", nullptr};

        return dispatch_multi_field_kernel(pipeline(KernelId::KesslerPointwise),
                                           KernelId::KesslerPointwise,
                                           &pc,
                                           inputs,
                                           7,
                                           outputs,
                                           6,
                                           nr,
                                           nth,
                                           nz,
                                           total_cells,
                                           nullptr,
                                           input_names,
                                           output_write_only);
#else
        (void)dispatch;
        return false;
#endif
    }

    bool dispatch_kessler_sedimentation(const KesslerSedimentationDispatch& dispatch) override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        const auto* qr_data     = dispatch.qr_data;
        const auto* qg_data     = dispatch.qg_data;
        const auto* qh_data     = dispatch.qh_data;
        const auto* rho_d_data  = dispatch.rho_d_data;
        auto*       dqr_dt_data = dispatch.dqr_dt_data;
        auto*       dqg_dt_data = dispatch.dqg_dt_data;
        auto*       dqh_dt_data = dispatch.dqh_dt_data;
        const int   nr          = dispatch.nr;
        const int   nth         = dispatch.nth;
        const int   nz          = dispatch.nz;
        const int   i_begin     = dispatch.i_begin;
        const int   i_end       = dispatch.i_end;
        const int   j_begin     = dispatch.j_begin;
        const int   j_end       = dispatch.j_end;
        const int   k_begin     = dispatch.k_begin;
        const int   k_end       = dispatch.k_end;
        const float dz_val      = dispatch.dz_val;
        const float Vt_max_rain = dispatch.Vt_max_rain;
        const float vt_cap      = dispatch.vt_cap;
        const float dt_step     = dispatch.dt_step;

        if (!pipeline(KernelId::KesslerSedimentation).is_ready())
        {
            return false;
        }

        if (!nimbus::compute::valid_kessler_sedimentation(dispatch))
        {
            return false;
        }

        KesslerSedimentationPushConstants pc{};
        pc.nr                   = nr;
        pc.nth                  = nth;
        pc.nz                   = nz;
        pc.i_begin              = i_begin;
        pc.i_end                = i_end;
        pc.j_begin              = j_begin;
        pc.j_end                = j_end;
        pc.k_begin              = k_begin;
        pc.k_end                = k_end;
        pc.dz_val               = dz_val;
        pc.Vt_max_rain          = Vt_max_rain;
        pc.vt_cap               = vt_cap;
        pc.have_rho_d           = 1;
        pc.dt_step              = dt_step;
        pc.have_layer_thickness = dispatch.layer_thickness_m != nullptr;
        pc.pad2                 = 0.0f;

        if (i_begin < 0 || i_begin >= i_end || i_end > nr || j_begin < 0 || j_begin >= j_end || j_end > nth ||
            k_begin < 0 || k_begin >= k_end || k_end > nz)
        {
            return false;
        }
        if (!(dt_step > 0.0f) || !std::isfinite(dt_step))
        {
            return false;
        }

        uint32_t     total_columns         = 0;
        VkDeviceSize columns_bytes_checked = 0;
        if (!checked_field_dispatch(nr, nth, 1, total_columns, columns_bytes_checked, "Kessler sedimentation columns"))
            return false;

        const float*  inputs[5]   = {qr_data,
                                     qg_data,
                                     qh_data,
                                     rho_d_data,
                                     dispatch.layer_thickness_m ? dispatch.layer_thickness_m : rho_d_data};
        std::uint32_t cells       = 0;
        VkDeviceSize  field_bytes = 0;
        if (!checked_field_dispatch(nr, nth, nz, cells, field_bytes, "Kessler sedimentation state"))
            return false;
        std::array<std::vector<float>, 3> candidate{std::vector<float>(dqr_dt_data, dqr_dt_data + cells),
                                                    std::vector<float>(dqg_dt_data, dqg_dt_data + cells),
                                                    std::vector<float>(dqh_dt_data, dqh_dt_data + cells)};
        float*                            outputs[3] = {candidate[0].data(), candidate[1].data(), candidate[2].data()};
        const char*                       input_names[5] = {"qr", "qg", "qh", "rho", nullptr};

        if (!dispatch_multi_field_kernel(pipeline(KernelId::KesslerSedimentation),
                                         KernelId::KesslerSedimentation,
                                         &pc,
                                         inputs,
                                         5,
                                         outputs,
                                         3,
                                         nr,
                                         nth,
                                         nz,
                                         total_columns,
                                         nullptr,
                                         input_names))
            return false;
        if (!nimbus::compute::finite_kessler_sedimentation(dispatch, {outputs[0], outputs[1], outputs[2]}))
            return false;
        std::copy_n(outputs[0], cells, dqr_dt_data);
        std::copy_n(outputs[1], cells, dqg_dt_data);
        std::copy_n(outputs[2], cells, dqh_dt_data);
        return true;
#else
        (void)dispatch;
        return false;
#endif
    }

    void shutdown() override
    {
#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)
        compute_telemetry_.memory_window.abort(
            compute_telemetry_.memory_counters(), compute_telemetry_.memory_accounting_valid,
            std::cout, memory_window_process_id());
        print_gpu_submission_stats();
        bool device_quiesced = false;
        if (device_context_.device != VK_NULL_HANDLE && vkDeviceWaitIdle_ != nullptr)
        {
            const VkResult idle_result = vkDeviceWaitIdle_(device_context_.device);
            if (idle_result == VK_SUCCESS)
            {
                submission_lifecycle_.note_device_quiesced();
                device_quiesced = true;
            }
            else
            {
                // Do not individually destroy child handles after a failed
                // device wait. Device destruction is the terminal ownership
                // boundary and host wrappers are only abandoned below.
                submission_lifecycle_.poison_after_wait();
                log_vulkan_warning("vkDeviceWaitIdle failed during shutdown (result=" + vk_result_string(idle_result) +
                                   ")");
            }
        }
        else if (device_context_.device != VK_NULL_HANDLE)
        {
            submission_lifecycle_.poison_after_wait();
            log_vulkan_warning("vkDeviceWaitIdle unavailable during shutdown; terminal device "
                               "destruction will replace child-handle teardown");
        }

        const bool abandon_children = !(device_quiesced && submission_lifecycle_.may_destroy_child_resources());
        if (!abandon_children)
        {
            destroy_compute_resources();
        }

        if (device_context_.device != VK_NULL_HANDLE && vkDestroyDevice_ != nullptr)
        {
            vkDestroyDevice_(device_context_.device, nullptr);
        }
        if (abandon_children)
        {
            // Device ownership ended without any child destroy callbacks.
            // Forget their now-invalid values before reset/reinitialization.
            abandon_compute_resources();
        }

        if (device_context_.instance != VK_NULL_HANDLE && vkDestroyInstance_ != nullptr)
        {
            vkDestroyInstance_(device_context_.instance, nullptr);
        }

        if (device_context_.loader_handle != nullptr)
        {
            close_dynamic_library(device_context_.loader_handle);
        }

        reset_state();
#endif
    }

private:
    // Coordinate system selection is plain configuration state (not a Vulkan
    // type), so it lives outside the TMV_HAS_VULKAN_COMPUTE_HEADERS guard:
    // set_coordinate_system() is compiled unconditionally and must always have
    // a backing member, including on hosts built without Vulkan headers.
    CoordinateSystem                                           coord_system_              = CoordinateSystem::Cartesian;
    std::uint64_t                                              successful_dispatch_count_ = 0;
    std::array<std::uint64_t, nimbus::compute::kernel_count()> successful_kernel_dispatch_counts_{};
    ComputeBackendConfig                                       requested_config_{};

#if defined(TMV_HAS_VULKAN_COMPUTE_HEADERS) && defined(TMV_HAS_VULKAN_COMPUTE_DLOPEN)

    // ── GPU buffer wrapper (aliased from pool header) ──────────────────
    using GpuBuffer = tmv_vulkan::GpuBuffer;

    using ComputePipelineState = tmv_vulkan::ComputePipelineState;

    ComputePipelineState& pipeline(KernelId id)
    {
        return pipeline_registry_.get(id);
    }

    const ComputePipelineState& pipeline(KernelId id) const
    {
        return pipeline_registry_.get(id);
    }

    bool properties2_available_  = false;
    bool memory_budget_enabled_ = false;
    bool float_controls_enabled_ = false;
    bool float64_controls_enabled_ = false;
    bool strict_fma_enabled_ = false;
    bool strict_fma64_enabled_ = false;
    bool cuda_assist_enabled_ = false;
#ifdef VK_NV_cuda_kernel_launch
    tmv_vulkan::CudaFluxKernel cuda_flux_;
#endif
    std::array<bool, 2> acoustic_loop_active_{};
    std::array<bool, 2> acoustic_velocity_reuse_{};
    std::array<bool, 2> acoustic_pressure_output_reuse_{};

    struct PhysicalDeviceCandidate
    {
        std::size_t                enumeration_index = 0;
        VkPhysicalDevice           handle            = VK_NULL_HANDLE;
        VkPhysicalDeviceProperties properties{};
        VkPhysicalDeviceFeatures   features{};
        VkQueueFamilyProperties    compute_queue_family{};
        uint32_t                   compute_queue_family_index  = std::numeric_limits<uint32_t>::max();
        int                        score                       = -1;
        bool                       suitable                    = false;
        bool                       has_compute_queue           = false;
        bool                       limits_usable               = false;
        bool                       supports_portability_subset = false;
        std::string                unsuitable_reason;
    };

    template <typename ProcType>
    static bool load_global_proc(PFN_vkGetInstanceProcAddr get_instance_proc_addr, const char* name, ProcType& out_proc)
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
                                   VkInstance                instance,
                                   const char*               name,
                                   ProcType&                 out_proc)
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
                                 VkDevice                device,
                                 const char*             name,
                                 ProcType&               out_proc)
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
        return std::to_string(VK_VERSION_MAJOR(version)) + "." + std::to_string(VK_VERSION_MINOR(version)) + "." +
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
            return " (hint: ensure Homebrew packages `vulkan-loader` and `molten-vk` are installed; "
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

    void log_startup_memory(const char* phase, const char* operation = nullptr,
                            std::size_t ordinal = 0, std::uint64_t payload_bytes = 0) const noexcept
    {
        if (!startup_memory_enabled_)
        {
            return;
        }
        write_process_memory_trace(std::cout, true, "VULKAN STARTUP MEMORY",
                                   phase, operation, ordinal, payload_bytes);
    }

    // ── Pipeline routing for the collocated acoustic dispatches ─────
    //
    // The collocated acoustic substep runs on the Cartesian pipelines; the
    // caller checks readiness and falls back to the CPU when they are not
    // ready.

    ComputePipelineState& active_acoustic_pressure_pipeline()
    {
        return pipeline(KernelId::AcousticPressureCartesian);
    }

    ComputePipelineState& active_acoustic_momentum_pipeline()
    {
        return pipeline(KernelId::AcousticMomentumCartesian);
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
        device_context_.loader_handle      = nullptr;
        device_context_.loader_api_version = VK_API_VERSION_1_0;

        device_context_.instance        = VK_NULL_HANDLE;
        device_context_.physical_device = VK_NULL_HANDLE;
        device_context_.device          = VK_NULL_HANDLE;
        device_context_.compute_queue   = VK_NULL_HANDLE;

        device_context_.compute_queue_family_index           = std::numeric_limits<uint32_t>::max();
        device_context_.selected_device_index                = -1;
        device_context_.selected_device_properties           = {};
        device_context_.selected_device_features             = {};
        device_context_.selected_compute_queue_family        = {};
        device_context_.selected_supports_portability_subset = false;
        properties2_available_                               = false;
        memory_budget_enabled_                               = false;
        float_controls_enabled_                              = false;
        float64_controls_enabled_                            = false;
        strict_fma_enabled_                                  = false;
        strict_fma64_enabled_                                = false;
        cuda_assist_enabled_                                 = false;

        vkGetInstanceProcAddr_                    = nullptr;
        vkCreateInstance_                         = nullptr;
        vkEnumerateInstanceVersion_               = nullptr;
        vkEnumerateInstanceExtensionProperties_   = nullptr;
        vkEnumerateInstanceLayerProperties_       = nullptr;
        vkDestroyInstance_                        = nullptr;
        vkEnumeratePhysicalDevices_               = nullptr;
        vkGetPhysicalDeviceProperties_            = nullptr;
        vkGetPhysicalDeviceFeatures_              = nullptr;
        vkGetPhysicalDeviceQueueFamilyProperties_ = nullptr;
        vkEnumerateDeviceExtensionProperties_     = nullptr;
        vkCreateDevice_                           = nullptr;
        vkGetDeviceProcAddr_                      = nullptr;
        vkDestroyDevice_                          = nullptr;
        vkGetDeviceQueue_                         = nullptr;
        vkDeviceWaitIdle_                         = nullptr;
        vkGetPhysicalDeviceMemoryProperties_      = nullptr;
        vkGetPhysicalDeviceMemoryProperties2_     = nullptr;

        vkCreateBuffer_                 = nullptr;
        vkDestroyBuffer_                = nullptr;
        vkAllocateMemory_               = nullptr;
        vkFreeMemory_                   = nullptr;
        vkBindBufferMemory_             = nullptr;
        vkMapMemory_                    = nullptr;
        vkUnmapMemory_                  = nullptr;
        vkFlushMappedMemoryRanges_      = nullptr;
        vkInvalidateMappedMemoryRanges_ = nullptr;
        vkGetBufferMemoryRequirements_  = nullptr;
        vkCreateShaderModule_           = nullptr;
        vkDestroyShaderModule_          = nullptr;
        vkCreateDescriptorSetLayout_    = nullptr;
        vkDestroyDescriptorSetLayout_   = nullptr;
        vkCreateDescriptorPool_         = nullptr;
        vkDestroyDescriptorPool_        = nullptr;
        vkAllocateDescriptorSets_       = nullptr;
        vkUpdateDescriptorSets_         = nullptr;
        vkCreatePipelineLayout_         = nullptr;
        vkDestroyPipelineLayout_        = nullptr;
        vkCreateComputePipelines_       = nullptr;
        vkDestroyPipeline_              = nullptr;
        vkCreatePipelineCache_          = nullptr;
        vkDestroyPipelineCache_         = nullptr;
        vkGetPipelineCacheData_         = nullptr;
        vkCreateCommandPool_            = nullptr;
        vkDestroyCommandPool_           = nullptr;
        vkAllocateCommandBuffers_       = nullptr;
        vkBeginCommandBuffer_           = nullptr;
        vkEndCommandBuffer_             = nullptr;
        vkResetCommandBuffer_           = nullptr;
        vkCmdBindPipeline_              = nullptr;
        vkCmdBindDescriptorSets_        = nullptr;
        vkCmdPushConstants_             = nullptr;
        vkCmdDispatch_                  = nullptr;
        vkCmdPipelineBarrier_           = nullptr;
        vkCmdCopyBuffer_                = nullptr;
        vkQueueSubmit_                  = nullptr;
        vkCreateFence_                  = nullptr;
        vkDestroyFence_                 = nullptr;
        vkWaitForFences_                = nullptr;
        vkResetFences_                  = nullptr;
        vkCreateQueryPool_              = nullptr;
        vkDestroyQueryPool_             = nullptr;
        vkCmdResetQueryPool_            = nullptr;
        vkCmdWriteTimestamp_            = nullptr;
        vkGetQueryPoolResults_          = nullptr;

        for (ComputePipelineState& pipeline_state : pipeline_registry_)
        {
            pipeline_state = {};
        }
        cmd_pool_                = VK_NULL_HANDLE;
        cmd_buf_                 = VK_NULL_HANDLE;
        fence_                   = VK_NULL_HANDLE;
        timestamp_query_pool_    = VK_NULL_HANDLE;
        gpu_timestamp_available_ = false;
        pending_dispatch_count_  = 0;
        pipeline_cache_          = VK_NULL_HANDLE;
        pipeline_cache_path_.clear();
        pipeline_cache_identity_ = {};
        pipeline_cache_loaded_   = false;

        staging_q_          = {};
        staging_w_          = {};
        staging_dz_         = {};
        staging_dqdt_       = {};
        device_q_           = {};
        device_w_           = {};
        device_dz_          = {};
        device_dqdt_        = {};
        qv0_upload_scratch_ = {};
        u0_upload_scratch_  = {};
        v0_upload_scratch_  = {};
        pooled_memory_allocations_.clear();

        device_context_.memory_plan        = {};
        device_context_.has_unified_memory = false;
        clear_resident_field_scopes();
        active_residency_scope_ = ComputeResidencyScope::Root;
        gpu_submission_stats_.clear();
        completed_readback_stat_ = nullptr;
        compute_telemetry_           = {};
        attributed_compute_activity_ = {};
        successful_dispatch_count_   = 0;
        successful_kernel_dispatch_counts_.fill(0);
        pending_kernel_dispatch_counts_.fill(0);
        submitted_kernel_dispatch_counts_.fill(0);
        submission_lifecycle_.reset_for_new_device();
    }

    bool open_loader(std::string& error)
    {
#if defined(__APPLE__)
        const auto bundled = tmv::packaged_runtime::library("libMoltenVK.dylib");
        std::vector<const char*> loader_candidates = {"libvulkan.1.dylib",
                                                            "libvulkan.dylib",
                                                            "/opt/homebrew/lib/libvulkan.1.dylib",
                                                            "/opt/homebrew/lib/libvulkan.dylib",
                                                            "/opt/homebrew/opt/vulkan-loader/lib/libvulkan.1.dylib",
                                                            "/opt/homebrew/opt/vulkan-loader/lib/libvulkan.dylib",
                                                            "/usr/local/lib/libvulkan.1.dylib",
                                                            "/usr/local/lib/libvulkan.dylib"};
        if (!bundled.empty()) loader_candidates = {bundled.c_str()};
#elif defined(__linux__)
        const std::vector<const char*> loader_candidates = {
            "libvulkan.so.1",
            "libvulkan.so",
            "/usr/lib/x86_64-linux-gnu/libvulkan.so.1", // Debian/Ubuntu
            "/usr/lib64/libvulkan.so.1",                // Fedora/RHEL
            "/usr/lib/libvulkan.so.1",                  // Arch
        };
#elif defined(_WIN32)
        const std::vector<const char*> loader_candidates = {
            "vulkan-1.dll",
        };
#else
        const std::vector<const char*> loader_candidates;
#endif

        std::vector<std::string> tried_candidates;

        const char* loader_override = std::getenv("TMV_VULKAN_LOADER_PATH");
        if (loader_override != nullptr && loader_override[0] != '\0')
        {
            if (!open_dynamic_library(loader_override, device_context_.loader_handle, error))
            {
                return false;
            }
            tried_candidates.push_back(loader_override);
        }

        for (const char* candidate : loader_candidates)
        {
            if (device_context_.loader_handle != nullptr)
            {
                break;
            }
            if (!open_dynamic_library(candidate, device_context_.loader_handle, error))
            {
                return false;
            }
            tried_candidates.emplace_back(candidate);
        }

        if (device_context_.loader_handle == nullptr)
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
            load_dynamic_symbol(device_context_.loader_handle, "vkGetInstanceProcAddr"));
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
        if (!load_global_proc(
                vkGetInstanceProcAddr_, "vkEnumerateInstanceLayerProperties", vkEnumerateInstanceLayerProperties_))
        {
            error = "failed to load vkEnumerateInstanceLayerProperties from Vulkan loader";
            return false;
        }

        load_global_proc(vkGetInstanceProcAddr_, "vkEnumerateInstanceVersion", vkEnumerateInstanceVersion_);

        device_context_.loader_api_version = VK_API_VERSION_1_0;
        if (vkEnumerateInstanceVersion_ != nullptr)
        {
            uint32_t       version = VK_API_VERSION_1_0;
            const VkResult result  = vkEnumerateInstanceVersion_(&version);
            if (result == VK_SUCCESS)
            {
                device_context_.loader_api_version = version;
            }
        }

        log_vulkan_info("loader API version=" + version_to_string(device_context_.loader_api_version));
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
        if (count > 0 && vkEnumerateInstanceExtensionProperties_(nullptr, &count, extensions.data()) != VK_SUCCESS)
        {
            return false;
        }

        extensions.resize(count);
        return std::any_of(extensions.begin(),
                           extensions.end(),
                           [extension_name](const VkExtensionProperties& extension)
                           { return std::strcmp(extension.extensionName, extension_name) == 0; });
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
        if (count > 0 && vkEnumerateInstanceLayerProperties_(&count, layers.data()) != VK_SUCCESS)
        {
            return false;
        }

        layers.resize(count);
        return std::any_of(layers.begin(),
                           layers.end(),
                           [layer_name](const VkLayerProperties& layer)
                           { return std::strcmp(layer.layerName, layer_name) == 0; });
    }

    bool create_instance(std::string& error)
    {
        constexpr const char*    kValidationLayerName = "VK_LAYER_KHRONOS_validation";
        std::vector<const char*> extensions;
        VkInstanceCreateFlags    create_flags               = 0;
        bool                     portability_enabled        = false;
        bool                     portability_props2_enabled = false;

        auto append_extension = [&extensions](const char* extension_name)
        {
            const bool already_enabled = std::any_of(extensions.begin(),
                                                     extensions.end(),
                                                     [extension_name](const char* existing)
                                                     { return std::strcmp(existing, extension_name) == 0; });
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
        log_vulkan_info("instance support: portability_enumeration=" + std::string(portability_enabled ? "yes" : "no") +
                        ", properties2=" + std::string(portability_props2_enabled ? "yes" : "no") +
                        ", validation_layer=" + std::string(validation_layer_available ? "yes" : "no"));

        const uint32_t preferred_api_version = std::min(device_context_.loader_api_version, VK_API_VERSION_1_1);

        auto try_create_instance = [&](uint32_t                        api_version,
                                       const std::vector<const char*>& requested_extensions,
                                       VkInstanceCreateFlags           flags,
                                       VkResult&                       out_result,
                                       VkInstance&                     out_instance)
        {
            VkApplicationInfo app_info{};
            app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            app_info.pApplicationName   = "TornadoModel Compute Backend";
            app_info.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
            app_info.pEngineName        = "TornadoModel";
            app_info.engineVersion      = VK_MAKE_API_VERSION(0, 0, 1, 0);
            app_info.apiVersion         = api_version;

            VkInstanceCreateInfo create_info{};
            create_info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            create_info.pApplicationInfo        = &app_info;
            create_info.flags                   = flags;
            create_info.enabledExtensionCount   = static_cast<uint32_t>(requested_extensions.size());
            create_info.ppEnabledExtensionNames = requested_extensions.empty() ? nullptr : requested_extensions.data();

            out_result = vkCreateInstance_(&create_info, nullptr, &out_instance);
            if (out_result == VK_SUCCESS)
            {
                properties2_available_ =
                    api_version >= VK_API_VERSION_1_1 ||
                    std::any_of(
                        requested_extensions.begin(),
                        requested_extensions.end(),
                        [](const char* name)
                        { return std::strcmp(name, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) == 0; });
            }
            return out_result == VK_SUCCESS;
        };

        VkResult                 result             = VK_ERROR_INITIALIZATION_FAILED;
        VkInstance               created_instance   = VK_NULL_HANDLE;
        std::vector<const char*> active_extensions  = extensions;
        VkInstanceCreateFlags    active_flags       = create_flags;
        uint32_t                 active_api_version = preferred_api_version;

        if (try_create_instance(active_api_version, active_extensions, active_flags, result, created_instance))
        {
            device_context_.instance = created_instance;
            return true;
        }

        if (result == VK_ERROR_INCOMPATIBLE_DRIVER && active_api_version != VK_API_VERSION_1_0)
        {
            log_vulkan_warning("vkCreateInstance reported VK_ERROR_INCOMPATIBLE_DRIVER with API " +
                               version_to_string(active_api_version) + "; retrying with Vulkan 1.0");
            active_api_version = VK_API_VERSION_1_0;
            if (try_create_instance(active_api_version, active_extensions, active_flags, result, created_instance))
            {
                device_context_.instance = created_instance;
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
                device_context_.instance = created_instance;
                return true;
            }
        }

        error = "vkCreateInstance failed with VkResult=" + vk_result_string(result) +
                " (requested_api=" + version_to_string(preferred_api_version) +
                ", active_api=" + version_to_string(active_api_version) + ", extensions=[" +
                join_extension_names(active_extensions) + "])" + instance_failure_hint(result);
        return false;
    }

    bool load_instance_functions(std::string& error)
    {
        if (!load_instance_proc(
                vkGetInstanceProcAddr_, device_context_.instance, "vkDestroyInstance", vkDestroyInstance_))
        {
            error = "failed to load vkDestroyInstance";
            return false;
        }
        if (!load_instance_proc(vkGetInstanceProcAddr_,
                                device_context_.instance,
                                "vkEnumeratePhysicalDevices",
                                vkEnumeratePhysicalDevices_))
        {
            error = "failed to load vkEnumeratePhysicalDevices";
            return false;
        }
        if (!load_instance_proc(vkGetInstanceProcAddr_,
                                device_context_.instance,
                                "vkGetPhysicalDeviceProperties",
                                vkGetPhysicalDeviceProperties_))
        {
            error = "failed to load vkGetPhysicalDeviceProperties";
            return false;
        }
        if (!load_instance_proc(vkGetInstanceProcAddr_,
                                device_context_.instance,
                                "vkGetPhysicalDeviceFeatures",
                                vkGetPhysicalDeviceFeatures_))
        {
            error = "failed to load vkGetPhysicalDeviceFeatures";
            return false;
        }
        if (!load_instance_proc(vkGetInstanceProcAddr_,
                                device_context_.instance,
                                "vkGetPhysicalDeviceQueueFamilyProperties",
                                vkGetPhysicalDeviceQueueFamilyProperties_))
        {
            error = "failed to load vkGetPhysicalDeviceQueueFamilyProperties";
            return false;
        }
        if (!load_instance_proc(vkGetInstanceProcAddr_,
                                device_context_.instance,
                                "vkEnumerateDeviceExtensionProperties",
                                vkEnumerateDeviceExtensionProperties_))
        {
            error = "failed to load vkEnumerateDeviceExtensionProperties";
            return false;
        }
        if (!load_instance_proc(vkGetInstanceProcAddr_, device_context_.instance, "vkCreateDevice", vkCreateDevice_))
        {
            error = "failed to load vkCreateDevice";
            return false;
        }
        if (!load_instance_proc(
                vkGetInstanceProcAddr_, device_context_.instance, "vkGetDeviceProcAddr", vkGetDeviceProcAddr_))
        {
            error = "failed to load vkGetDeviceProcAddr";
            return false;
        }
        if (!load_instance_proc(vkGetInstanceProcAddr_,
                                device_context_.instance,
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
        if (vkEnumerateDeviceExtensionProperties_ == nullptr || device == VK_NULL_HANDLE || extension_name == nullptr)
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
        return std::any_of(extensions.begin(),
                           extensions.end(),
                           [extension_name](const VkExtensionProperties& extension)
                           { return std::strcmp(extension.extensionName, extension_name) == 0; });
    }

    bool select_physical_device(std::string& error)
    {
        uint32_t physical_device_count = 0;
        VkResult result = vkEnumeratePhysicalDevices_(device_context_.instance, &physical_device_count, nullptr);
        if (result != VK_SUCCESS)
        {
            error =
                "vkEnumeratePhysicalDevices(count) failed with VkResult=" + std::to_string(static_cast<int>(result));
            return false;
        }
        if (physical_device_count == 0)
        {
            error = "no Vulkan physical devices found";
            return false;
        }

        std::vector<VkPhysicalDevice> physical_devices(physical_device_count, VK_NULL_HANDLE);
        result = vkEnumeratePhysicalDevices_(device_context_.instance, &physical_device_count, physical_devices.data());
        if (result != VK_SUCCESS)
        {
            error = "vkEnumeratePhysicalDevices(list) failed with VkResult=" + std::to_string(static_cast<int>(result));
            return false;
        }
        physical_devices.resize(physical_device_count);

        std::vector<PhysicalDeviceCandidate> candidates;
        candidates.reserve(physical_devices.size());

        for (std::size_t i = 0; i < physical_devices.size(); ++i)
        {
            PhysicalDeviceCandidate candidate{};
            candidate.enumeration_index = i;
            candidate.handle            = physical_devices[i];
            vkGetPhysicalDeviceProperties_(candidate.handle, &candidate.properties);
            vkGetPhysicalDeviceFeatures_(candidate.handle, &candidate.features);

            uint32_t queue_family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties_(candidate.handle, &queue_family_count, nullptr);

            if (queue_family_count > 0)
            {
                std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
                vkGetPhysicalDeviceQueueFamilyProperties_(candidate.handle, &queue_family_count, queue_families.data());

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
                    candidate.compute_queue_family       = queue_family;
                    break;
                }
            }

            const bool has_compute_queue = candidate.compute_queue_family_index != std::numeric_limits<uint32_t>::max();
            candidate.has_compute_queue  = has_compute_queue;

            const bool limits_usable =
                candidate.properties.limits.maxComputeWorkGroupInvocations >= 64 &&
                candidate.properties.limits.maxComputeWorkGroupSize[0] >= 64 &&
                candidate.properties.limits.maxComputeWorkGroupCount[0] > 0 &&
                candidate.properties.limits.maxComputeWorkGroupCount[1] > 0 &&
                candidate.properties.limits.maxComputeWorkGroupCount[2] > 0 &&
                static_cast<std::uint64_t>(candidate.properties.limits.maxComputeWorkGroupCount[0]) *
                        candidate.properties.limits.maxComputeWorkGroupCount[1] >=
                    ((static_cast<std::uint64_t>(std::numeric_limits<uint32_t>::max()) + 63u) / 64u) &&
                candidate.properties.limits.maxComputeSharedMemorySize > 0;
            candidate.limits_usable = limits_usable;

            candidate.suitable = has_compute_queue && limits_usable;
            if (candidate.suitable)
            {
                candidate.score = device_type_score(candidate.properties.deviceType);
                candidate.score += static_cast<int>(candidate.properties.limits.maxComputeSharedMemorySize / 1024);
                candidate.score += static_cast<int>(candidate.properties.limits.maxComputeWorkGroupInvocations / 32);
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
            stream << ", max_invocations=" << candidate.properties.limits.maxComputeWorkGroupInvocations
                   << ", shared_mem_bytes=" << candidate.properties.limits.maxComputeSharedMemorySize
                   << ", portability_subset=" << (candidate.supports_portability_subset ? "yes" : "no")
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

        const int               preferred_device_index = requested_config_.device_index;
        PhysicalDeviceCandidate selected_candidate{};

        if (preferred_device_index >= 0)
        {
            const std::size_t index = static_cast<std::size_t>(preferred_device_index);
            if (index >= candidates.size())
            {
                error = "requested numerics.compute.device_index " + std::to_string(preferred_device_index) +
                        " is out of range (detected devices: " + std::to_string(candidates.size()) + ")";
                return false;
            }

            selected_candidate = candidates[index];
            if (!selected_candidate.suitable)
            {
                error = "requested Vulkan device index " + std::to_string(preferred_device_index) +
                        " does not expose a usable compute queue/capabilities";
                return false;
            }
        }
        else
        {
            bool selected   = false;
            int  best_score = std::numeric_limits<int>::min();
            for (const PhysicalDeviceCandidate& candidate : candidates)
            {
                if (!candidate.suitable)
                {
                    continue;
                }

                if (!selected || candidate.score > best_score)
                {
                    selected_candidate = candidate;
                    best_score         = candidate.score;
                    selected           = true;
                }
            }

            if (!selected)
            {
                error = "no suitable Vulkan physical device found with compute queue support";
                return false;
            }
        }

        device_context_.selected_device_index                = static_cast<int>(selected_candidate.enumeration_index);
        device_context_.physical_device                      = selected_candidate.handle;
        device_context_.selected_device_properties           = selected_candidate.properties;
        device_context_.selected_device_features             = selected_candidate.features;
        device_context_.compute_queue_family_index           = selected_candidate.compute_queue_family_index;
        device_context_.selected_compute_queue_family        = selected_candidate.compute_queue_family;
        device_context_.selected_supports_portability_subset = selected_candidate.supports_portability_subset;
        log_vulkan_info("selected device[" + std::to_string(device_context_.selected_device_index) + "] \"" +
                        std::string(device_context_.selected_device_properties.deviceName) + "\"");
        return true;
    }

    bool create_logical_device(std::string& error)
    {
        if (device_context_.physical_device == VK_NULL_HANDLE)
        {
            error = "Vulkan physical device is not selected";
            return false;
        }
        if (device_context_.compute_queue_family_index == std::numeric_limits<uint32_t>::max())
        {
            error = "no compute queue family selected";
            return false;
        }

        float                   queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = device_context_.compute_queue_family_index;
        queue_info.queueCount       = 1;
        queue_info.pQueuePriorities = &queue_priority;

        std::vector<const char*> device_extensions;
        if (device_context_.selected_supports_portability_subset)
        {
            device_extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
        }

        memory_budget_enabled_ = false;
        if (properties2_available_ &&
            has_device_extension(device_context_.physical_device, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME))
        {
            vkGetPhysicalDeviceMemoryProperties2_ = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
                vkGetInstanceProcAddr_(device_context_.instance, "vkGetPhysicalDeviceMemoryProperties2"));
            if (!vkGetPhysicalDeviceMemoryProperties2_)
                vkGetPhysicalDeviceMemoryProperties2_ = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
                    vkGetInstanceProcAddr_(device_context_.instance, "vkGetPhysicalDeviceMemoryProperties2KHR"));
            if (vkGetPhysicalDeviceMemoryProperties2_)
            {
                device_extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
                memory_budget_enabled_ = true;
            }
        }

        float_controls_enabled_ = false;
        if (properties2_available_ &&
            has_device_extension(device_context_.physical_device, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME))
        {
            auto get_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
                vkGetInstanceProcAddr_(device_context_.instance, "vkGetPhysicalDeviceProperties2"));
            if (!get_properties)
            {
                get_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
                    vkGetInstanceProcAddr_(device_context_.instance, "vkGetPhysicalDeviceProperties2KHR"));
            }
            if (get_properties)
            {
                VkPhysicalDeviceFloatControlsProperties controls{};
                controls.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
                VkPhysicalDeviceProperties2 properties{};
                properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                properties.pNext = &controls;
                get_properties(device_context_.physical_device, &properties);
                float_controls_enabled_ =
                    controls.shaderSignedZeroInfNanPreserveFloat32 && controls.shaderRoundingModeRTEFloat32;
                float64_controls_enabled_ = float_controls_enabled_ &&
                    controls.shaderSignedZeroInfNanPreserveFloat64 && controls.shaderRoundingModeRTEFloat64;
                if (float_controls_enabled_)
                {
                    device_extensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
                }
            }
        }

        VkPhysicalDeviceFeatures requested_features{};
        requested_features.shaderInt64 =
            device_context_.selected_device_features.shaderInt64;
        requested_features.shaderFloat64 = device_context_.selected_device_features.shaderFloat64 && float64_controls_enabled_;

        bool strict_fma_available = false;
#ifdef VK_KHR_shader_fma
        VkPhysicalDeviceShaderFmaFeaturesKHR requested_fma{};
        requested_fma.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR;
        if (properties2_available_ &&
            has_device_extension(device_context_.physical_device, VK_KHR_SHADER_FMA_EXTENSION_NAME))
        {
            auto get_features = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
                vkGetInstanceProcAddr_(device_context_.instance, "vkGetPhysicalDeviceFeatures2"));
            if (!get_features)
                get_features = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
                    vkGetInstanceProcAddr_(device_context_.instance, "vkGetPhysicalDeviceFeatures2KHR"));
            if (get_features)
            {
                VkPhysicalDeviceFeatures2 available{};
                available.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
                available.pNext = &requested_fma;
                get_features(device_context_.physical_device, &available);
                requested_fma.shaderFmaFloat16 = VK_FALSE;
                requested_fma.shaderFmaFloat64 = requested_fma.shaderFmaFloat64 && requested_features.shaderFloat64;
                strict_fma64_enabled_ = requested_fma.shaderFmaFloat64;
                strict_fma_available = requested_fma.shaderFmaFloat32;
                if (strict_fma_available || strict_fma64_enabled_)
                    device_extensions.push_back(VK_KHR_SHADER_FMA_EXTENSION_NAME);
            }
        }
#endif
        strict_fma_enabled_ = strict_fma_available;

        if (requested_config_.cuda_assist)
        {
#ifdef VK_NV_cuda_kernel_launch
            if (device_context_.selected_device_properties.vendorID != 0x10de || !properties2_available_ ||
                !has_device_extension(device_context_.physical_device, VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME) ||
                !has_device_extension(device_context_.physical_device, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME))
            {
                error = "CUDA assist requires a compatible NVIDIA Vulkan device and driver";
                return false;
            }
            auto features = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
                vkGetInstanceProcAddr_(device_context_.instance, "vkGetPhysicalDeviceFeatures2"));
            if (!features) features = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
                vkGetInstanceProcAddr_(device_context_.instance, "vkGetPhysicalDeviceFeatures2KHR"));
            auto properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
                vkGetInstanceProcAddr_(device_context_.instance, "vkGetPhysicalDeviceProperties2"));
            if (!properties) properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
                vkGetInstanceProcAddr_(device_context_.instance, "vkGetPhysicalDeviceProperties2KHR"));
            if (!features || !properties || !cuda_flux_.inspect(device_context_.physical_device, features, properties, error))
            {
                if (error.empty()) error = "CUDA assist feature queries are unavailable";
                return false;
            }
            device_extensions.push_back(VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME);
            device_extensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
            cuda_assist_enabled_ = true;
#else
            error = "CUDA assist requires Vulkan headers with VK_NV_cuda_kernel_launch";
            return false;
#endif
        }

        VkDeviceCreateInfo create_info{};
        create_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.queueCreateInfoCount    = 1;
        create_info.pQueueCreateInfos       = &queue_info;
        create_info.enabledExtensionCount   = static_cast<uint32_t>(device_extensions.size());
        create_info.ppEnabledExtensionNames = device_extensions.empty() ? nullptr : device_extensions.data();
        create_info.pEnabledFeatures        = &requested_features;
#ifdef VK_KHR_shader_fma
        if (strict_fma_available || strict_fma64_enabled_)
            create_info.pNext = &requested_fma;
#endif
#ifdef VK_NV_cuda_kernel_launch
        if (cuda_assist_enabled_) create_info.pNext = cuda_flux_.feature_chain(create_info.pNext);
#endif

        const VkResult result =
            vkCreateDevice_(device_context_.physical_device, &create_info, nullptr, &device_context_.device);
        if (result != VK_SUCCESS)
        {
            error = "vkCreateDevice failed with VkResult=" + std::to_string(static_cast<int>(result));
            return false;
        }
        log_vulkan_info("created logical device successfully");
        return true;
    }

    bool load_device_functions(std::string& error)
    {
        if (!load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkDestroyDevice", vkDestroyDevice_))
        {
            error = "failed to load vkDestroyDevice";
            return false;
        }
        if (!load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkGetDeviceQueue", vkGetDeviceQueue_))
        {
            error = "failed to load vkGetDeviceQueue";
            return false;
        }

        load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkDeviceWaitIdle", vkDeviceWaitIdle_);

        vkGetDeviceQueue_(
            device_context_.device, device_context_.compute_queue_family_index, 0, &device_context_.compute_queue);
        if (device_context_.compute_queue == VK_NULL_HANDLE)
        {
            error = "selected Vulkan compute queue handle is null";
            return false;
        }
        log_vulkan_info("resolved compute queue handle");

        // Load compute resource function pointers
        bool ok = true;
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkCreateBuffer", vkCreateBuffer_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkDestroyBuffer", vkDestroyBuffer_);
        ok =
            ok && load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkAllocateMemory", vkAllocateMemory_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkFreeMemory", vkFreeMemory_);
        ok = ok &&
             load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkBindBufferMemory", vkBindBufferMemory_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkMapMemory", vkMapMemory_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkUnmapMemory", vkUnmapMemory_);
        ok = ok &&
             load_device_proc(
                 vkGetDeviceProcAddr_, device_context_.device, "vkFlushMappedMemoryRanges", vkFlushMappedMemoryRanges_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_,
                                    device_context_.device,
                                    "vkInvalidateMappedMemoryRanges",
                                    vkInvalidateMappedMemoryRanges_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_,
                                    device_context_.device,
                                    "vkGetBufferMemoryRequirements",
                                    vkGetBufferMemoryRequirements_);
        ok = ok && load_device_proc(
                       vkGetDeviceProcAddr_, device_context_.device, "vkCreateShaderModule", vkCreateShaderModule_);
        ok = ok && load_device_proc(
                       vkGetDeviceProcAddr_, device_context_.device, "vkDestroyShaderModule", vkDestroyShaderModule_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_,
                                    device_context_.device,
                                    "vkCreateDescriptorSetLayout",
                                    vkCreateDescriptorSetLayout_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_,
                                    device_context_.device,
                                    "vkDestroyDescriptorSetLayout",
                                    vkDestroyDescriptorSetLayout_);
        ok = ok && load_device_proc(
                       vkGetDeviceProcAddr_, device_context_.device, "vkCreateDescriptorPool", vkCreateDescriptorPool_);
        ok = ok &&
             load_device_proc(
                 vkGetDeviceProcAddr_, device_context_.device, "vkDestroyDescriptorPool", vkDestroyDescriptorPool_);
        ok = ok &&
             load_device_proc(
                 vkGetDeviceProcAddr_, device_context_.device, "vkAllocateDescriptorSets", vkAllocateDescriptorSets_);
        ok = ok && load_device_proc(
                       vkGetDeviceProcAddr_, device_context_.device, "vkUpdateDescriptorSets", vkUpdateDescriptorSets_);
        ok = ok && load_device_proc(
                       vkGetDeviceProcAddr_, device_context_.device, "vkCreatePipelineLayout", vkCreatePipelineLayout_);
        ok = ok &&
             load_device_proc(
                 vkGetDeviceProcAddr_, device_context_.device, "vkDestroyPipelineLayout", vkDestroyPipelineLayout_);
        ok = ok &&
             load_device_proc(
                 vkGetDeviceProcAddr_, device_context_.device, "vkCreateComputePipelines", vkCreateComputePipelines_);
        ok = ok &&
             load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkDestroyPipeline", vkDestroyPipeline_);
        ok = ok && load_device_proc(
                       vkGetDeviceProcAddr_, device_context_.device, "vkCreateCommandPool", vkCreateCommandPool_);
        ok = ok && load_device_proc(
                       vkGetDeviceProcAddr_, device_context_.device, "vkDestroyCommandPool", vkDestroyCommandPool_);
        ok = ok &&
             load_device_proc(
                 vkGetDeviceProcAddr_, device_context_.device, "vkAllocateCommandBuffers", vkAllocateCommandBuffers_);
        ok = ok && load_device_proc(
                       vkGetDeviceProcAddr_, device_context_.device, "vkBeginCommandBuffer", vkBeginCommandBuffer_);
        ok = ok &&
             load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkEndCommandBuffer", vkEndCommandBuffer_);
        ok = ok && load_device_proc(
                       vkGetDeviceProcAddr_, device_context_.device, "vkResetCommandBuffer", vkResetCommandBuffer_);
        ok = ok &&
             load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkCmdBindPipeline", vkCmdBindPipeline_);
        ok = ok &&
             load_device_proc(
                 vkGetDeviceProcAddr_, device_context_.device, "vkCmdBindDescriptorSets", vkCmdBindDescriptorSets_);
        ok = ok &&
             load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkCmdPushConstants", vkCmdPushConstants_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkCmdDispatch", vkCmdDispatch_);
        ok = ok && load_device_proc(
                       vkGetDeviceProcAddr_, device_context_.device, "vkCmdPipelineBarrier", vkCmdPipelineBarrier_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkCmdCopyBuffer", vkCmdCopyBuffer_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkQueueSubmit", vkQueueSubmit_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkCreateFence", vkCreateFence_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkDestroyFence", vkDestroyFence_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkWaitForFences", vkWaitForFences_);
        ok = ok && load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkResetFences", vkResetFences_);

        // Pipeline caching is an optional startup optimization. Missing cache
        // entry points must not disable an otherwise scientifically valid GPU.
        bool cache_functions_available = true;
        cache_functions_available &= load_device_proc(
            vkGetDeviceProcAddr_, device_context_.device, "vkCreatePipelineCache", vkCreatePipelineCache_);
        cache_functions_available &= load_device_proc(
            vkGetDeviceProcAddr_, device_context_.device, "vkDestroyPipelineCache", vkDestroyPipelineCache_);
        cache_functions_available &= load_device_proc(
            vkGetDeviceProcAddr_, device_context_.device, "vkGetPipelineCacheData", vkGetPipelineCacheData_);
        if (!cache_functions_available)
        {
            vkCreatePipelineCache_  = nullptr;
            vkDestroyPipelineCache_ = nullptr;
            vkGetPipelineCacheData_ = nullptr;
            log_vulkan_warning("pipeline-cache entry points unavailable; compiling pipelines "
                               "without a persistent cache");
        }

        // Profiling is optional and must never make the compute backend unavailable.
        load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkCreateQueryPool", vkCreateQueryPool_);
        load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkDestroyQueryPool", vkDestroyQueryPool_);
        load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkCmdResetQueryPool", vkCmdResetQueryPool_);
        load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkCmdWriteTimestamp", vkCmdWriteTimestamp_);
        load_device_proc(vkGetDeviceProcAddr_, device_context_.device, "vkGetQueryPoolResults", vkGetQueryPoolResults_);

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
        vkGetPhysicalDeviceMemoryProperties_(device_context_.physical_device, &mem_props);
        std::uint32_t                   preferred    = tmv_vulkan::kInvalidVulkanMemoryIndex;
        constexpr VkMemoryPropertyFlags kHostFlags   = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        const bool                      needs_host   = tmv_vulkan::vulkan_memory_flags_contain(props, kHostFlags);
        const bool                      needs_device = (props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        if (needs_host && needs_device)
        {
            preferred = device_context_.memory_plan.direct_memory_type;
        }
        else if (needs_device)
        {
            preferred = device_context_.memory_plan.working_memory_type;
        }
        else if (needs_host)
        {
            preferred = device_context_.memory_plan.staging_memory_type;
        }
        return tmv_vulkan::select_vulkan_memory_type(mem_props, type_bits, props, preferred);
    }

    struct SharedBufferAllocation
    {
        VkDeviceMemory        memory            = VK_NULL_HANDLE;
        VkDeviceSize          size              = 0;
        void*                 mapped_base       = nullptr;
        std::size_t           live_buffers      = 0;
        VkMemoryPropertyFlags memory_properties = 0;
        bool                  device_allocation = false;
    };

    static bool align_buffer_offset(VkDeviceSize value, VkDeviceSize alignment, VkDeviceSize& aligned)
    {
        if (alignment == 0)
        {
            aligned = value;
            return true;
        }
        const VkDeviceSize remainder = value % alignment;
        const VkDeviceSize padding   = remainder == 0 ? 0 : alignment - remainder;
        if (value > std::numeric_limits<VkDeviceSize>::max() - padding)
        {
            return false;
        }
        aligned = value + padding;
        return true;
    }

    /**
     * Create independent VkBuffer handles backed by one exact-sized,
     * alignment-correct VkDeviceMemory allocation.
     */
    bool create_buffer_batch(const VkDeviceSize*   sizes,
                             VkBufferUsageFlags    usage,
                             VkMemoryPropertyFlags mem_props,
                             std::size_t           count,
                             GpuBuffer*            out)
    {
        if (sizes == nullptr || count == 0 || out == nullptr)
        {
            return false;
        }

        std::vector<VkBuffer>             buffers(count, VK_NULL_HANDLE);
        std::vector<VkMemoryRequirements> requirements(count);
        std::vector<VkDeviceSize>         offsets(count, 0);
        uint32_t                          compatible_memory_types = std::numeric_limits<uint32_t>::max();
        VkDeviceSize                      total_size              = 0;

        VkBufferCreateInfo buffer_info{};
        buffer_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.usage       = usage;
#ifdef VK_NV_cuda_kernel_launch
        if (cuda_assist_enabled_) buffer_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
#endif
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        const auto destroy_created_buffers = [&]()
        {
            if (vkDestroyBuffer_ == nullptr)
                return;
            for (VkBuffer buffer : buffers)
            {
                if (buffer != VK_NULL_HANDLE)
                {
                    vkDestroyBuffer_(device_context_.device, buffer, nullptr);
                }
            }
        };

        for (std::size_t i = 0; i < count; ++i)
        {
            if (sizes[i] == 0)
            {
                destroy_created_buffers();
                return false;
            }
            buffer_info.size = sizes[i];
            if (vkCreateBuffer_(device_context_.device, &buffer_info, nullptr, &buffers[i]) != VK_SUCCESS)
            {
                destroy_created_buffers();
                return false;
            }
            vkGetBufferMemoryRequirements_(device_context_.device, buffers[i], &requirements[i]);
            compatible_memory_types &= requirements[i].memoryTypeBits;
            if (!align_buffer_offset(total_size, requirements[i].alignment, offsets[i]) ||
                requirements[i].size > std::numeric_limits<VkDeviceSize>::max() - offsets[i])
            {
                destroy_created_buffers();
                return false;
            }
            total_size = offsets[i] + requirements[i].size;
        }

        const uint32_t memory_type = find_memory_type(compatible_memory_types, mem_props);
        if (memory_type == std::numeric_limits<uint32_t>::max())
        {
            destroy_created_buffers();
            return false;
        }

        VkMemoryAllocateInfo allocation_info{};
        allocation_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
#ifdef VK_NV_cuda_kernel_launch
        VkMemoryAllocateFlagsInfo address_flags{};
        address_flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        address_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        if (cuda_assist_enabled_) allocation_info.pNext = &address_flags;
#endif
        allocation_info.allocationSize  = total_size;
        allocation_info.memoryTypeIndex = memory_type;
        VkDeviceMemory memory           = VK_NULL_HANDLE;
        if (vkAllocateMemory_(device_context_.device, &allocation_info, nullptr, &memory) != VK_SUCCESS)
        {
            destroy_created_buffers();
            return false;
        }

        for (std::size_t i = 0; i < count; ++i)
        {
            if (vkBindBufferMemory_(device_context_.device, buffers[i], memory, offsets[i]) != VK_SUCCESS)
            {
                destroy_created_buffers();
                vkFreeMemory_(device_context_.device, memory, nullptr);
                return false;
            }
        }

        void* mapped_base = nullptr;
        if ((mem_props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 &&
            vkMapMemory_(device_context_.device, memory, 0, total_size, 0, &mapped_base) != VK_SUCCESS)
        {
            destroy_created_buffers();
            vkFreeMemory_(device_context_.device, memory, nullptr);
            return false;
        }

        auto allocation          = std::make_unique<SharedBufferAllocation>();
        allocation->memory       = memory;
        allocation->size         = total_size;
        allocation->mapped_base  = mapped_base;
        allocation->live_buffers = count;
        VkPhysicalDeviceMemoryProperties physical_memory{};
        vkGetPhysicalDeviceMemoryProperties_(device_context_.physical_device, &physical_memory);
        allocation->memory_properties  = physical_memory.memoryTypes[memory_type].propertyFlags;
        allocation->device_allocation  = (usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0;
        SharedBufferAllocation* shared = allocation.get();
        pooled_memory_allocations_.push_back(std::move(allocation));

        for (std::size_t i = 0; i < count; ++i)
        {
            out[i].buffer            = buffers[i];
            out[i].memory            = memory;
            out[i].size              = sizes[i];
            out[i].allocation_size   = requirements[i].size;
            out[i].memory_offset     = offsets[i];
            out[i].mapped            = mapped_base == nullptr
                                           ? nullptr
                                           : static_cast<void*>(static_cast<unsigned char*>(mapped_base) + offsets[i]);
            out[i].shared_allocation = shared;
            out[i].memory_properties = shared->memory_properties;
            out[i].device_allocation = shared->device_allocation;
            compute_telemetry_.note_allocation(static_cast<std::uint64_t>(requirements[i].size),
                                               shared->device_allocation);
        }
        compute_telemetry_.note_memory_allocation(static_cast<std::uint64_t>(total_size), shared->device_allocation);
        return true;
    }

    bool create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_props, GpuBuffer& out)
    {
        VkBufferCreateInfo buf_info{};
        buf_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_info.size        = size;
        buf_info.usage       = usage;
#ifdef VK_NV_cuda_kernel_launch
        if (cuda_assist_enabled_) buf_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
#endif
        buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer_(device_context_.device, &buf_info, nullptr, &out.buffer) != VK_SUCCESS)
        {
            return false;
        }

        VkMemoryRequirements reqs{};
        vkGetBufferMemoryRequirements_(device_context_.device, out.buffer, &reqs);

        const uint32_t mem_type_idx = find_memory_type(reqs.memoryTypeBits, mem_props);
        if (mem_type_idx == std::numeric_limits<uint32_t>::max())
        {
            vkDestroyBuffer_(device_context_.device, out.buffer, nullptr);
            out.buffer = VK_NULL_HANDLE;
            return false;
        }

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
#ifdef VK_NV_cuda_kernel_launch
        VkMemoryAllocateFlagsInfo address_flags{};
        address_flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        address_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        if (cuda_assist_enabled_) alloc_info.pNext = &address_flags;
#endif
        alloc_info.allocationSize  = reqs.size;
        alloc_info.memoryTypeIndex = mem_type_idx;

        if (vkAllocateMemory_(device_context_.device, &alloc_info, nullptr, &out.memory) != VK_SUCCESS)
        {
            vkDestroyBuffer_(device_context_.device, out.buffer, nullptr);
            out.buffer = VK_NULL_HANDLE;
            return false;
        }

        if (vkBindBufferMemory_(device_context_.device, out.buffer, out.memory, 0) != VK_SUCCESS)
        {
            vkDestroyBuffer_(device_context_.device, out.buffer, nullptr);
            vkFreeMemory_(device_context_.device, out.memory, nullptr);
            out = {};
            return false;
        }

        out.size            = size;
        out.allocation_size = reqs.size;
        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties_(device_context_.physical_device, &memory);
        out.memory_properties = memory.memoryTypes[mem_type_idx].propertyFlags;
        out.device_allocation = (usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0;

        // Map host-visible buffers persistently
        if ((mem_props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0)
        {
            if (vkMapMemory_(device_context_.device, out.memory, 0, out.allocation_size, 0, &out.mapped) != VK_SUCCESS)
            {
                // A host-visible buffer without a valid mapping is unusable by
                // every upload/download path.  Fail allocation transactionally
                // instead of deferring a null memcpy to dispatch time.
                vkDestroyBuffer_(device_context_.device, out.buffer, nullptr);
                vkFreeMemory_(device_context_.device, out.memory, nullptr);
                out = {};
                return false;
            }
        }

        compute_telemetry_.note_allocation(static_cast<std::uint64_t>(out.allocation_size), out.device_allocation);
        compute_telemetry_.note_memory_allocation(static_cast<std::uint64_t>(out.allocation_size),
                                                  out.device_allocation);

        return true;
    }

    void destroy_buffer(GpuBuffer& buf)
    {
        if (buf.memory != VK_NULL_HANDLE && buf.allocation_size > 0)
        {
            compute_telemetry_.note_release(static_cast<std::uint64_t>(buf.allocation_size), buf.device_allocation);
        }
        if (buf.buffer != VK_NULL_HANDLE && vkDestroyBuffer_ != nullptr)
        {
            vkDestroyBuffer_(device_context_.device, buf.buffer, nullptr);
        }

        if (buf.shared_allocation != nullptr)
        {
            auto* shared = static_cast<SharedBufferAllocation*>(buf.shared_allocation);
            if (shared->live_buffers > 0)
                --shared->live_buffers;
            if (shared->live_buffers == 0 && shared->memory != VK_NULL_HANDLE)
            {
                if (shared->mapped_base != nullptr && vkUnmapMemory_ != nullptr)
                {
                    vkUnmapMemory_(device_context_.device, shared->memory);
                }
                if (vkFreeMemory_ != nullptr)
                {
                    vkFreeMemory_(device_context_.device, shared->memory, nullptr);
                }
                compute_telemetry_.note_memory_release(static_cast<std::uint64_t>(shared->size),
                                                       shared->device_allocation);
                shared->memory      = VK_NULL_HANDLE;
                shared->mapped_base = nullptr;
            }
            buf = {};
            return;
        }

        if (buf.mapped != nullptr && vkUnmapMemory_ != nullptr)
        {
            vkUnmapMemory_(device_context_.device, buf.memory);
        }
        if (buf.memory != VK_NULL_HANDLE && vkFreeMemory_ != nullptr)
        {
            vkFreeMemory_(device_context_.device, buf.memory, nullptr);
            compute_telemetry_.note_memory_release(static_cast<std::uint64_t>(buf.allocation_size),
                                                   buf.device_allocation);
        }
        buf = {};
    }

    // ── Buffer pool callbacks ────────────────────────────────────────────

    static bool pool_create_buffer(void* user_data, VkDeviceSize size, bool is_staging, GpuBuffer& out_buf)
    {
        auto* self = static_cast<VulkanComputeBackend*>(user_data);

        // Unified memory path: both staging and device use a single buffer
        // with HOST_VISIBLE + DEVICE_LOCAL. The staging slot gets a no-op
        // placeholder (device buffer is used directly for both CPU and GPU).
        if (self->device_context_.has_unified_memory)
        {
            if (is_staging)
            {
                // Staging is unused on unified memory — create a placeholder.
                // The dispatch path will read/write via the device buffer's mapping.
                out_buf = {};
                return true;
            }
            // Device buffer: unified memory (accessible from both CPU and GPU)
            constexpr VkBufferUsageFlags    kUnifiedUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            constexpr VkMemoryPropertyFlags kUnifiedProps =
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            return self->create_buffer(size, kUnifiedUsage, kUnifiedProps, out_buf);
        }

        // Discrete memory path: separate staging (host) and device (GPU) buffers
        constexpr VkBufferUsageFlags kStagingUsage =
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        constexpr VkBufferUsageFlags kDeviceUsage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        constexpr VkMemoryPropertyFlags kHostProps   = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        constexpr VkMemoryPropertyFlags kDeviceProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        if (is_staging)
        {
            return self->create_buffer(size, kStagingUsage, kHostProps, out_buf);
        }
        return self->create_buffer(size, kDeviceUsage, kDeviceProps, out_buf);
    }

    static bool pool_create_buffer_batch(void*               user_data,
                                         const VkDeviceSize* sizes,
                                         std::size_t         count,
                                         GpuBuffer*          staging_buffers,
                                         GpuBuffer*          device_buffers)
    {
        auto* self = static_cast<VulkanComputeBackend*>(user_data);
        if (self->device_context_.has_unified_memory)
        {
            constexpr VkBufferUsageFlags    kUnifiedUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            constexpr VkMemoryPropertyFlags kUnifiedProps =
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            return self->create_buffer_batch(sizes, kUnifiedUsage, kUnifiedProps, count, device_buffers);
        }

        constexpr VkBufferUsageFlags kStagingUsage =
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        constexpr VkBufferUsageFlags kDeviceUsage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        constexpr VkMemoryPropertyFlags kHostProps   = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        constexpr VkMemoryPropertyFlags kDeviceProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        if (!self->create_buffer_batch(sizes, kStagingUsage, kHostProps, count, staging_buffers))
        {
            return false;
        }
        if (!self->create_buffer_batch(sizes, kDeviceUsage, kDeviceProps, count, device_buffers))
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                self->destroy_buffer(staging_buffers[i]);
            }
            return false;
        }
        return true;
    }

    static void pool_destroy_buffer(void* user_data, GpuBuffer& buf)
    {
        auto* self = static_cast<VulkanComputeBackend*>(user_data);
        self->destroy_buffer(buf);
    }

    /** @brief Probe and classify the actual storage-buffer memory topology. */
    bool detect_unified_memory(std::string& error)
    {
        device_context_.memory_plan        = {};
        device_context_.has_unified_memory = false;
        if (device_context_.device == VK_NULL_HANDLE || vkCreateBuffer_ == nullptr ||
            vkGetBufferMemoryRequirements_ == nullptr || vkDestroyBuffer_ == nullptr ||
            vkGetPhysicalDeviceMemoryProperties_ == nullptr)
        {
            error = "Vulkan memory-plan probe functions are unavailable";
            return false;
        }

        VkBufferCreateInfo probe_info{};
        probe_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        probe_info.size  = sizeof(float);
        probe_info.usage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        probe_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer probe         = VK_NULL_HANDLE;
        if (vkCreateBuffer_(device_context_.device, &probe_info, nullptr, &probe) != VK_SUCCESS)
        {
            log_vulkan_info("unified-memory probe buffer creation failed — using staging transfers");
            error = "representative Vulkan storage-buffer probe creation failed";
            return false;
        }
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements_(device_context_.device, probe, &requirements);
        vkDestroyBuffer_(device_context_.device, probe, nullptr);

        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties_(device_context_.physical_device, &memory);
        device_context_.memory_plan = tmv_vulkan::classify_vulkan_memory_plan(
            memory, requirements.memoryTypeBits, device_context_.selected_device_properties.deviceType);

        tmv_vulkan::VulkanMemoryPolicy policy           = tmv_vulkan::VulkanMemoryPolicy::Auto;
        const char*                    policy_env       = std::getenv("NIMBUS_VULKAN_MEMORY_POLICY");
        const std::string              requested_policy = policy_env != nullptr ? policy_env : "auto";
        if (!tmv_vulkan::parse_vulkan_memory_policy(requested_policy, policy))
        {
            error = "invalid NIMBUS_VULKAN_MEMORY_POLICY='" + requested_policy + "' (expected auto, direct, or staged)";
            return false;
        }
        if (!tmv_vulkan::apply_vulkan_memory_policy(device_context_.memory_plan, policy))
        {
            error = "Vulkan memory policy '" + requested_policy + "' is incompatible with detected topology '" +
                    tmv_vulkan::vulkan_memory_topology_name(device_context_.memory_plan.topology) + "'";
            return false;
        }
        device_context_.has_unified_memory = device_context_.memory_plan.uses_direct_shared_memory();

        const auto topology = device_context_.memory_plan.topology;
        if (policy == tmv_vulkan::VulkanMemoryPolicy::Staged)
        {
            log_vulkan_info("memory plan=staged_forced (physical_topology=" +
                            std::string(tmv_vulkan::vulkan_memory_topology_name(topology)) + ")");
            return true;
        }
        if (topology == tmv_vulkan::VulkanMemoryTopology::DirectShared)
        {
            log_vulkan_info("memory plan=direct_shared (compatible host-visible primary "
                            "device heap, policy=" +
                            requested_policy + ")");
            return true;
        }
        if (topology == tmv_vulkan::VulkanMemoryTopology::HostVisibleDeviceLocalAperture)
        {
            log_vulkan_info("memory plan=host_visible_device_local_aperture; retaining "
                            "staged transfers through the primary device heap");
            return true;
        }
        if (topology == tmv_vulkan::VulkanMemoryTopology::StagedDiscrete)
        {
            log_vulkan_info("memory plan=staged_discrete");
            return true;
        }
        log_vulkan_info("no coherent Vulkan storage/staging memory plan is available");
        error = "no compatible coherent Vulkan storage/staging memory plan is "
                "available";
        return false;
    }

    void init_buffer_pool()
    {
        buffer_pool_.set_allocator(&pool_create_buffer, &pool_destroy_buffer, this);
        buffer_pool_.set_batch_allocator(&pool_create_buffer_batch);
        for (auto& store : resident_field_scopes_)
        {
            store.set_allocator(&resident_slot_acquire, &resident_slot_release, this);
            store.set_batch_allocator(&resident_slots_acquire_many);
        }
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
        if (!submission_lifecycle_.can_dispatch())
        {
            return false;
        }
        return count > 0 && buffer_pool_.acquire_many(field_bytes, static_cast<std::size_t>(count), slot_indices);
    }

    // ── GPU submission timing (GPU audit Phase 4.5) ─────────────
    //
    // Env-gated (NIMBUS_GPU_KERNEL_TIMING=1): accumulates the fence-wait
    // wall time per submission label. The fence wait approximates the
    // GPU execution time of the submitted command buffer (plus driver
    // sync overhead, which is also signal). Printed at shutdown, this
    // ranks the submissions so kernel-tuning effort (tiling, workgroup
    // shape) goes where the time is.

    struct GpuSubmissionStat
    {
        double                                fence_ms = 0.0;
        double                                gpu_ms   = 0.0;
        std::uint64_t                         gpu_timestamp_samples = 0;
        tmv::vulkan_compute::ActivityCounters activity;
    };

    static bool gpu_timing_enabled()
    {
        static const bool enabled = []()
        {
            const char* env = std::getenv("NIMBUS_GPU_KERNEL_TIMING");
            return env != nullptr && env[0] == '1';
        }();
        return enabled;
    }

    static bool mapped_range_contains(const GpuBuffer& buffer, const void* pointer, std::size_t bytes)
    {
        if (buffer.mapped == nullptr || pointer == nullptr)
        {
            return false;
        }
        const auto base    = reinterpret_cast<std::uintptr_t>(buffer.mapped);
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        return address >= base && address - base <= buffer.size &&
               bytes <= static_cast<std::size_t>(buffer.size - (address - base));
    }

    bool is_mapped_gpu_range(const void* pointer, std::size_t bytes) const
    {
        const GpuBuffer* fixed[] = {
            &staging_q_, &staging_w_, &staging_dz_, &staging_dqdt_, &device_q_, &device_w_, &device_dz_, &device_dqdt_};
        for (const GpuBuffer* buffer : fixed)
        {
            if (mapped_range_contains(*buffer, pointer, bytes))
            {
                return true;
            }
        }
        for (std::size_t i = 0; i < buffer_pool_.slot_count(); ++i)
        {
            if (mapped_range_contains(buffer_pool_.staging(static_cast<int>(i)), pointer, bytes) ||
                mapped_range_contains(buffer_pool_.device(static_cast<int>(i)), pointer, bytes))
            {
                return true;
            }
        }
        return false;
    }

    VkResult synchronize_noncoherent_mapped_memory(bool invalidate)
    {
        std::vector<VkMappedMemoryRange> ranges;
        std::vector<VkDeviceMemory>      seen;
        const VkDeviceSize atom_size = device_context_.selected_device_properties.limits.nonCoherentAtomSize;
        const auto         add       = [&](const GpuBuffer& buffer)
        {
            if (buffer.memory == VK_NULL_HANDLE || buffer.mapped == nullptr ||
                (buffer.memory_properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0 ||
                std::find(seen.begin(), seen.end(), buffer.memory) != seen.end())
            {
                return;
            }
            VkDeviceSize allocation_size = buffer.allocation_size;
            if (buffer.shared_allocation != nullptr)
            {
                allocation_size = static_cast<const SharedBufferAllocation*>(buffer.shared_allocation)->size;
            }
            tmv_vulkan::VulkanNonCoherentRange aligned;
            if (!tmv_vulkan::align_vulkan_noncoherent_range(0, allocation_size, allocation_size, atom_size, aligned))
            {
                return;
            }
            VkMappedMemoryRange range{};
            range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            range.memory = buffer.memory;
            range.offset = aligned.offset;
            range.size   = aligned.size;
            ranges.push_back(range);
            seen.push_back(buffer.memory);
        };

        const GpuBuffer* fixed[] = {
            &staging_q_, &staging_w_, &staging_dz_, &staging_dqdt_, &device_q_, &device_w_, &device_dz_, &device_dqdt_};
        for (const GpuBuffer* buffer : fixed)
            add(*buffer);
        for (std::size_t i = 0; i < buffer_pool_.slot_count(); ++i)
        {
            add(buffer_pool_.staging(static_cast<int>(i)));
            add(buffer_pool_.device(static_cast<int>(i)));
        }
        if (ranges.empty())
            return VK_SUCCESS;
        return invalidate ? vkInvalidateMappedMemoryRanges_(
                                device_context_.device, static_cast<std::uint32_t>(ranges.size()), ranges.data())
                          : vkFlushMappedMemoryRanges_(
                                device_context_.device, static_cast<std::uint32_t>(ranges.size()), ranges.data());
    }

    void tracked_memcpy(void* destination, const void* source, std::size_t bytes)
    {
        if (gpu_timing_enabled())
        {
            if (is_mapped_gpu_range(destination, bytes))
            {
                compute_telemetry_.note_host_upload(bytes);
            }
            else if (is_mapped_gpu_range(source, bytes))
            {
                compute_telemetry_.note_host_download(bytes);
                // Readback follows the completed fence; the next submission's
                // preparation delta must not inherit this operation's copies.
                auto& producer = completed_readback_stat_ != nullptr ? *completed_readback_stat_ :
                    gpu_submission_stats_["host_readback_without_submission"];
                tmv::vulkan_compute::attribute_host_readback(
                    producer.activity, attributed_compute_activity_, bytes);
            }
        }
        tmv::copy_bytes(destination, source, bytes);
    }

    void tracked_memset(void* destination, int value, std::size_t bytes)
    {
        if (gpu_timing_enabled() && is_mapped_gpu_range(destination, bytes))
        {
            compute_telemetry_.note_host_upload(bytes);
        }
        std::memset(destination, value, bytes);
    }

    enum class BufferRole
    {
        Unknown,
        Staging,
        Device
    };

    BufferRole buffer_role(VkBuffer handle) const
    {
        const GpuBuffer* staging[] = {&staging_q_, &staging_w_, &staging_dz_, &staging_dqdt_};
        const GpuBuffer* device[]  = {&device_q_, &device_w_, &device_dz_, &device_dqdt_};
        for (const GpuBuffer* buffer : staging)
        {
            if (handle != VK_NULL_HANDLE && buffer->buffer == handle)
                return BufferRole::Staging;
        }
        for (const GpuBuffer* buffer : device)
        {
            if (handle != VK_NULL_HANDLE && buffer->buffer == handle)
                return BufferRole::Device;
        }
        for (std::size_t i = 0; i < buffer_pool_.slot_count(); ++i)
        {
            if (buffer_pool_.staging(static_cast<int>(i)).buffer == handle)
                return BufferRole::Staging;
            if (buffer_pool_.device(static_cast<int>(i)).buffer == handle)
                return BufferRole::Device;
        }
        return BufferRole::Unknown;
    }

    void
    tracked_cmd_copy_buffer(VkBuffer source, VkBuffer destination, uint32_t region_count, const VkBufferCopy* regions)
    {
        if (gpu_timing_enabled())
        {
            std::uint64_t bytes = 0;
            for (uint32_t i = 0; i < region_count; ++i)
            {
                bytes += static_cast<std::uint64_t>(regions[i].size);
            }
            const BufferRole source_role      = buffer_role(source);
            const BufferRole destination_role = buffer_role(destination);
            if (source_role == BufferRole::Staging && destination_role == BufferRole::Device)
            {
                compute_telemetry_.note_device_upload(bytes);
            }
            else if (source_role == BufferRole::Device && destination_role == BufferRole::Staging)
            {
                compute_telemetry_.note_device_download(bytes);
            }
        }
        vkCmdCopyBuffer_(cmd_buf_, source, destination, region_count, regions);
    }

    VkResult begin_profiled_command_buffer(const VkCommandBufferBeginInfo& begin_info)
    {
        completed_readback_stat_ = nullptr;
        pending_dispatch_count_ = 0;
        pending_kernel_dispatch_counts_.fill(0);
        const VkResult result = vkBeginCommandBuffer_(cmd_buf_, &begin_info);
        if (result == VK_SUCCESS && gpu_timestamp_available_)
        {
            vkCmdResetQueryPool_(cmd_buf_, timestamp_query_pool_, 0, 2);
            vkCmdWriteTimestamp_(cmd_buf_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamp_query_pool_, 0);
        }
        return result;
    }

    VkResult end_profiled_command_buffer()
    {
        if (gpu_timestamp_available_)
        {
            vkCmdWriteTimestamp_(cmd_buf_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamp_query_pool_, 1);
        }
        return vkEndCommandBuffer_(cmd_buf_);
    }

    VkResult reset_fence_and_submit(const char* label, const VkSubmitInfo& submit_info)
    {
        if (!submission_lifecycle_.can_dispatch())
        {
            return VK_ERROR_DEVICE_LOST;
        }

        const VkResult flush_result = synchronize_noncoherent_mapped_memory(false);
        if (flush_result != VK_SUCCESS)
        {
            submitted_kernel_dispatch_counts_.fill(0);
            submission_lifecycle_.poison_before_submit();
            log_vulkan_warning(std::string(label) + ": non-coherent mapped-memory flush failed (result=" +
                               vk_result_string(flush_result) + "); Vulkan backend disabled before submission");
            return flush_result;
        }

        const VkResult reset_result = vkResetFences_(device_context_.device, 1, &fence_);
        if (reset_result != VK_SUCCESS)
        {
            submitted_kernel_dispatch_counts_.fill(0);
            submission_lifecycle_.poison_before_submit();
            log_vulkan_warning(std::string(label) + ": vkResetFences failed (result=" + vk_result_string(reset_result) +
                               "); Vulkan backend disabled; CPU fallback remains available");
            return reset_result;
        }

        const VkResult submit_result = vkQueueSubmit_(device_context_.compute_queue, 1, &submit_info, fence_);
        if (submit_result == VK_SUCCESS)
        {
            submitted_kernel_dispatch_counts_ = pending_kernel_dispatch_counts_;
            submission_lifecycle_.note_submit_succeeded();
            compute_telemetry_.note_submission(pending_dispatch_count_);
            if (gpu_timing_enabled())
            {
                const tmv::vulkan_compute::ActivityCounters current = compute_telemetry_.activity();
                const tmv::vulkan_compute::ActivityCounters delta =
                    tmv::vulkan_compute::activity_delta(current, attributed_compute_activity_);
                GpuSubmissionStat& stat = gpu_submission_stats_[label];
                stat.activity.submitted_command_buffers += delta.submitted_command_buffers;
                stat.activity.submitted_dispatches += delta.submitted_dispatches;
                stat.activity.host_upload_bytes += delta.host_upload_bytes;
                stat.activity.host_download_bytes += delta.host_download_bytes;
                stat.activity.device_upload_bytes += delta.device_upload_bytes;
                stat.activity.device_download_bytes += delta.device_download_bytes;
                stat.activity.resident_hits += delta.resident_hits;
                stat.activity.resident_misses += delta.resident_misses;
                stat.activity.resident_upload_bytes_saved += delta.resident_upload_bytes_saved;
                stat.activity.writeonly_output_upload_bytes_saved += delta.writeonly_output_upload_bytes_saved;
                stat.activity.descriptor_cache_hits += delta.descriptor_cache_hits;
                stat.activity.descriptor_cache_misses += delta.descriptor_cache_misses;
                attributed_compute_activity_ = current;
            }
        }
        else
        {
            submitted_kernel_dispatch_counts_.fill(0);
            // Treat the submission outcome as uncertain.  Pool slots and the
            // shared command buffer remain pinned until shutdown; subsequent
            // capability checks return false so callers can use CPU fallback
            // against their separate host arrays.
            submission_lifecycle_.poison_after_submit_attempt();
            log_vulkan_warning(std::string(label) +
                               ": vkQueueSubmit failed (result=" + vk_result_string(submit_result) +
                               "); Vulkan backend poisoned and resources retained until shutdown");
            // Do not charge work performed while constructing this failed
            // submission to a later operation if a test double permits the
            // poisoned backend to continue.
            if (gpu_timing_enabled())
            {
                attributed_compute_activity_ = compute_telemetry_.activity();
            }
        }
        pending_dispatch_count_ = 0;
        pending_kernel_dispatch_counts_.fill(0);
        return submit_result;
    }

    VkResult timed_wait_fence(const char* label)
    {
        constexpr uint64_t kTimeoutNs     = 10ULL * 1000000000ULL;
        const bool         timing_enabled = gpu_timing_enabled();
        const auto         t0             = std::chrono::steady_clock::now();
        compute_telemetry_.note_synchronous_wait();
        const VkResult result = vkWaitForFences_(device_context_.device, 1, &fence_, VK_TRUE, kTimeoutNs);
        if (timing_enabled)
        {
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            GpuSubmissionStat& stat = gpu_submission_stats_[label];
            stat.fence_ms += ms;
        }

        if (result == VK_SUCCESS)
        {
            submission_lifecycle_.note_wait_succeeded();
            const VkResult invalidate_result = synchronize_noncoherent_mapped_memory(true);
            if (invalidate_result != VK_SUCCESS)
            {
                submitted_kernel_dispatch_counts_.fill(0);
                submission_lifecycle_.poison_before_submit();
                log_vulkan_warning(std::string(label) + ": non-coherent mapped-memory invalidate failed (result=" +
                                   vk_result_string(invalidate_result) +
                                   "); Vulkan backend disabled after completed submission");
                return invalidate_result;
            }
            ++successful_dispatch_count_;
            for (std::size_t index = 0; index < successful_kernel_dispatch_counts_.size(); ++index)
            {
                successful_kernel_dispatch_counts_[index] += submitted_kernel_dispatch_counts_[index];
            }
            submitted_kernel_dispatch_counts_.fill(0);
            if (timing_enabled && gpu_timestamp_available_)
            {
                std::uint64_t  timestamps[2]{};
                const VkResult query_result = vkGetQueryPoolResults_(device_context_.device,
                                                                     timestamp_query_pool_,
                                                                     0,
                                                                     2,
                                                                     sizeof(timestamps),
                                                                     timestamps,
                                                                     sizeof(std::uint64_t),
                                                                     VK_QUERY_RESULT_64_BIT);
                if (query_result == VK_SUCCESS)
                {
                    ++gpu_submission_stats_[label].gpu_timestamp_samples;
                    const uint32_t      valid_bits = device_context_.selected_compute_queue_family.timestampValidBits;
                    const std::uint64_t delta = valid_bits >= 64
                                                    ? timestamps[1] - timestamps[0]
                                                    : (timestamps[1] - timestamps[0]) & ((1ULL << valid_bits) - 1ULL);
                    gpu_submission_stats_[label].gpu_ms +=
                        static_cast<double>(delta) *
                        static_cast<double>(device_context_.selected_device_properties.limits.timestampPeriod) / 1.0e6;
                }
            }
            if (timing_enabled) completed_readback_stat_ = &gpu_submission_stats_[label];
        }
        else
        {
            submitted_kernel_dispatch_counts_.fill(0);
            submission_lifecycle_.poison_after_wait();
            log_vulkan_warning(std::string(label) +
                               ": fence wait failed or timed out (result=" + vk_result_string(result) +
                               "); Vulkan backend poisoned and in-flight resources retained until shutdown");
        }
        return result;
    }

    static std::uint64_t memory_window_process_id() noexcept
    {
#if defined(_WIN32)
        return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
        return static_cast<std::uint64_t>(getpid());
#endif
    }

    void configure_memory_window()
    {
        const char* token = std::getenv("NIMBUS_C6_MEMORY_RUN_TOKEN");
        const char* first = std::getenv("NIMBUS_C6_MEMORY_FIRST_BOUNDARY");
        const char* end = std::getenv("NIMBUS_C6_MEMORY_END_BOUNDARY_EXCLUSIVE");
        if (token == nullptr && first == nullptr && end == nullptr)
        {
            return;
        }
        auto parse = [](const char* text, std::uint64_t& value)
        {
            if (text == nullptr || *text == '\0')
            {
                return false;
            }
            const char* last = text + std::strlen(text);
            const auto result = std::from_chars(text, last, value);
            return result.ec == std::errc{} && result.ptr == last;
        };
        std::uint64_t first_boundary = 0;
        std::uint64_t end_boundary_exclusive = 0;
        if (!parse(first, first_boundary) || !parse(end, end_boundary_exclusive) ||
            !compute_telemetry_.memory_window.configure(token, first_boundary, end_boundary_exclusive))
        {
            std::cout << "[C6 MEMORY WINDOW REJECTED] invalid window request\n";
        }
    }

    void print_gpu_submission_stats()
    {
        if (!gpu_timing_enabled())
        {
            return;
        }
        std::vector<std::pair<std::string, GpuSubmissionStat>> rows(gpu_submission_stats_.begin(),
                                                                    gpu_submission_stats_.end());
        std::sort(rows.begin(),
                  rows.end(),
                  [](const auto& a, const auto& b) { return a.second.fence_ms > b.second.fence_ms; });
        std::cout
            << "[GPU SUBMIT TIMING] label gpu_total_ms fence_total_ms submissions gpu_ms_per_submit fence_ms_per_submit"
            << std::endl;
        for (const auto& [label, stat] : rows)
        {
            const std::uint64_t submissions = stat.activity.submitted_command_buffers;
            std::cout << "[GPU SUBMIT TIMING] " << label << " " << stat.gpu_ms << " " << stat.fence_ms << " "
                      << submissions << " " << (submissions > 0 ? stat.gpu_ms / static_cast<double>(submissions) : 0.0)
                      << " " << (submissions > 0 ? stat.fence_ms / static_cast<double>(submissions) : 0.0) << std::endl;
            std::cout << "[GPU STAGE TELEMETRY] " << label
                      << " command_buffers=" << stat.activity.submitted_command_buffers
                      << " timestamped_command_buffers=" << stat.gpu_timestamp_samples
                      << " dispatches=" << stat.activity.submitted_dispatches
                      << " host_upload_bytes=" << stat.activity.host_upload_bytes
                      << " host_download_bytes=" << stat.activity.host_download_bytes
                      << " device_upload_bytes=" << stat.activity.device_upload_bytes
                      << " device_download_bytes=" << stat.activity.device_download_bytes
                      << " resident_hits=" << stat.activity.resident_hits
                      << " resident_misses=" << stat.activity.resident_misses
                      << " resident_upload_bytes_saved=" << stat.activity.resident_upload_bytes_saved
                      << " writeonly_output_upload_bytes_saved=" << stat.activity.writeonly_output_upload_bytes_saved
                      << " descriptor_cache_hits=" << stat.activity.descriptor_cache_hits
                      << " descriptor_cache_misses=" << stat.activity.descriptor_cache_misses << std::endl;
        }
        std::cout << "[GPU TELEMETRY] model_steps=" << compute_telemetry_.model_steps
                  << " command_buffers=" << compute_telemetry_.submitted_command_buffers
                  << " dispatches=" << compute_telemetry_.submitted_dispatches
                  << " host_upload_bytes=" << compute_telemetry_.host_upload_bytes
                  << " host_download_bytes=" << compute_telemetry_.host_download_bytes
                  << " device_upload_bytes=" << compute_telemetry_.device_upload_bytes
                  << " device_download_bytes=" << compute_telemetry_.device_download_bytes
                  << " resident_hits=" << compute_telemetry_.resident_hits
                  << " resident_misses=" << compute_telemetry_.resident_misses
                  << " resident_upload_bytes_saved=" << compute_telemetry_.resident_upload_bytes_saved
                  << " writeonly_output_upload_bytes_saved=" << compute_telemetry_.writeonly_output_upload_bytes_saved
                  << " resident_full_invalidations=" << compute_telemetry_.resident_full_invalidations
                  << " resident_entries_invalidated=" << compute_telemetry_.resident_entries_invalidated
                  << " batched_command_buffers=" << compute_telemetry_.batched_command_buffers
                  << " synchronous_waits=" << compute_telemetry_.synchronous_waits
                  << " descriptor_cache_hits=" << compute_telemetry_.descriptor_cache_hits
                  << " descriptor_cache_misses=" << compute_telemetry_.descriptor_cache_misses
                  << " staging_current_bytes=" << compute_telemetry_.staging_allocated_bytes
                  << " staging_high_water_bytes=" << compute_telemetry_.staging_high_water_bytes
                  << " device_current_bytes=" << compute_telemetry_.device_allocated_bytes
                  << " device_high_water_bytes=" << compute_telemetry_.device_high_water_bytes
                  << " staging_memory_allocations=" << compute_telemetry_.staging_memory_allocations
                  << " staging_memory_allocations_high_water="
                  << compute_telemetry_.staging_memory_allocations_high_water
                  << " device_memory_allocations=" << compute_telemetry_.device_memory_allocations
                  << " device_memory_allocations_high_water=" << compute_telemetry_.device_memory_allocations_high_water
                  << " staging_memory_reserved_bytes=" << compute_telemetry_.staging_memory_reserved_bytes
                  << " staging_memory_reserved_high_water_bytes="
                  << compute_telemetry_.staging_memory_reserved_high_water_bytes
                  << " device_memory_reserved_bytes=" << compute_telemetry_.device_memory_reserved_bytes
                  << " device_memory_reserved_high_water_bytes="
                  << compute_telemetry_.device_memory_reserved_high_water_bytes << std::endl;
    }

    void release_pool_slots(const int* slot_indices, int count)
    {
        if (!submission_lifecycle_.may_release_transients())
        {
            return;
        }
        for (int i = 0; i < count; ++i)
        {
            buffer_pool_.release(slot_indices[i]);
        }
    }

    // ── Backend-neutral resident field store adapters ────────────────

    static nimbus::compute::ResidentFieldStore::Allocation resident_slot_acquire(void* user_data, std::size_t bytes)
    {
        auto* self = static_cast<VulkanComputeBackend*>(user_data);
        return static_cast<nimbus::compute::ResidentFieldStore::Allocation>(
            self->buffer_pool_.acquire(static_cast<VkDeviceSize>(bytes)));
    }

    static bool resident_slots_acquire_many(void*                                            user_data,
                                            const std::size_t*                               bytes,
                                            std::size_t                                      count,
                                            nimbus::compute::ResidentFieldStore::Allocation* allocations)
    {
        constexpr std::size_t kMaximumNamedBatch = 64;
        if (bytes == nullptr || allocations == nullptr || count == 0 || count > kMaximumNamedBatch)
            return false;
        auto*                                        self = static_cast<VulkanComputeBackend*>(user_data);
        std::array<VkDeviceSize, kMaximumNamedBatch> sizes{};
        std::array<int, kMaximumNamedBatch>          slots{};
        for (std::size_t i = 0; i < count; ++i)
        {
            sizes[i]       = static_cast<VkDeviceSize>(bytes[i]);
            slots[i]       = -1;
            allocations[i] = nimbus::compute::ResidentFieldStore::invalid_allocation;
        }
        if (!self->buffer_pool_.acquire_many(sizes.data(), count, slots.data()))
        {
            return false;
        }
        for (std::size_t i = 0; i < count; ++i)
        {
            allocations[i] = static_cast<nimbus::compute::ResidentFieldStore::Allocation>(slots[i]);
        }
        return true;
    }

    static void resident_slot_release(void* user_data, nimbus::compute::ResidentFieldStore::Allocation allocation)
    {
        auto* self = static_cast<VulkanComputeBackend*>(user_data);
        self->buffer_pool_.release(static_cast<int>(allocation));
    }

    static constexpr std::size_t residency_scope_index(ComputeResidencyScope scope) noexcept
    {
        return scope == ComputeResidencyScope::Child ? 1u : 0u;
    }

    nimbus::compute::ResidentFieldStore& active_resident_fields() noexcept
    {
        return resident_field_scopes_[residency_scope_index(active_residency_scope_)];
    }

    void clear_resident_field_scopes()
    {
        for (auto& store : resident_field_scopes_)
            store.clear();
        acoustic_loop_active_.fill(false);
        acoustic_velocity_reuse_.fill(false);
        acoustic_pressure_output_reuse_.fill(false);
    }

    bool named_slots_acquire_many(const char* const* names, const VkDeviceSize* bytes, int count, int* slots)
    {
        constexpr int kMaximumNamedBatch = 64;
        if (!submission_lifecycle_.can_dispatch() || names == nullptr || bytes == nullptr || slots == nullptr ||
            count < 1 || count > kMaximumNamedBatch)
        {
            return false;
        }
        std::array<std::size_t, kMaximumNamedBatch>                                     host_sizes{};
        std::array<nimbus::compute::ResidentFieldStore::Allocation, kMaximumNamedBatch> allocations{};
        for (int i = 0; i < count; ++i)
            host_sizes[static_cast<std::size_t>(i)] = static_cast<std::size_t>(bytes[i]);
        if (!active_resident_fields().acquire_many(
                names, host_sizes.data(), static_cast<std::size_t>(count), allocations.data()))
        {
            return false;
        }
        for (int i = 0; i < count; ++i)
        {
            if (allocations[static_cast<std::size_t>(i)] < 0 ||
                allocations[static_cast<std::size_t>(i)] > std::numeric_limits<int>::max())
            {
                return false;
            }
            slots[i] = static_cast<int>(allocations[static_cast<std::size_t>(i)]);
        }
        return true;
    }

    bool named_slot_reusable(const char* name, const float* host_ptr, VkDeviceSize bytes)
    {
        const nimbus::compute::FieldExtent extent{static_cast<std::size_t>(bytes) / sizeof(float), sizeof(float)};
        const bool hit = active_resident_fields().reusable(name, host_ptr, extent, resident_reuse_enabled());
        compute_telemetry_.note_resident_lookup(hit, hit ? static_cast<std::uint64_t>(bytes) : 0);
        return hit;
    }

    void named_slot_commit(const char* name, const float* host_ptr, VkDeviceSize bytes)
    {
        active_resident_fields().commit_host_mirror(
            name, host_ptr, {static_cast<std::size_t>(bytes) / sizeof(float), sizeof(float)});
    }

    void named_slot_invalidate(const char* name)
    {
        active_resident_fields().invalidate(name);
    }

    static bool resident_reuse_enabled()
    {
        static const bool enabled = []()
        {
            const char* env = std::getenv("NIMBUS_GPU_RESIDENT");
            return !(env != nullptr && env[0] == '0');
        }();
        return enabled;
    }

    static bool dependency_batching_enabled()
    {
        static const bool enabled = []()
        {
            const char* env = std::getenv("NIMBUS_GPU_LIN_BATCH");
            return !(env != nullptr && env[0] == '0');
        }();
        return enabled;
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

        constexpr VkMemoryPropertyFlags kHostProps   = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        constexpr VkMemoryPropertyFlags kDeviceProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        constexpr VkBufferUsageFlags kStagingSrc = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        constexpr VkBufferUsageFlags kDeviceSsbo =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        bool ok = true;
        ok      = ok && create_buffer(field_bytes, kStagingSrc, kHostProps, staging_q_);
        ok      = ok && create_buffer(field_bytes, kStagingSrc, kHostProps, staging_w_);
        ok      = ok && create_buffer(dz_bytes, kStagingSrc, kHostProps, staging_dz_);
        ok      = ok && create_buffer(field_bytes, kStagingSrc, kHostProps, staging_dqdt_);
        ok      = ok && create_buffer(field_bytes, kDeviceSsbo, kDeviceProps, device_q_);
        ok      = ok && create_buffer(field_bytes, kDeviceSsbo, kDeviceProps, device_w_);
        ok      = ok && create_buffer(dz_bytes, kDeviceSsbo, kDeviceProps, device_dz_);
        ok      = ok && create_buffer(field_bytes, kDeviceSsbo, kDeviceProps, device_dqdt_);

        if (!ok)
        {
            log_vulkan_warning("buffer allocation failed — freeing partial allocations");
            destroy_buffer(staging_q_);
            destroy_buffer(staging_w_);
            destroy_buffer(staging_dz_);
            destroy_buffer(staging_dqdt_);
            destroy_buffer(device_q_);
            destroy_buffer(device_w_);
            destroy_buffer(device_dz_);
            destroy_buffer(device_dqdt_);
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
            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = pipeline(KernelId::TvdVerticalFlux).descriptor_set;
            writes[i].dstBinding      = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        writes[0].pBufferInfo = &q_info;
        writes[1].pBufferInfo = &w_info;
        writes[2].pBufferInfo = &dz_info;
        writes[3].pBufferInfo = &dqdt_info;

        vkUpdateDescriptorSets_(device_context_.device, 4, writes, 0, nullptr);
    }

    // ── Descriptor set update for pooled buffers ──────────────────────

    std::shared_ptr<const nimbus::compute::ThompsonCompletionTables> thompson_completion_tables_;
    static constexpr int kMaxPooledBindings = 32;

    void update_pooled_descriptor_set(ComputePipelineState& pipeline,
                                      const int*            slot_indices,
                                      VkDeviceSize          field_bytes,
                                      int                   count,
                                      const VkDeviceSize*   buffer_ranges = nullptr)
    {
        std::array<std::uintptr_t, kMaxPooledBindings> handles{};
        std::array<std::size_t, kMaxPooledBindings>    ranges{};
        for (int i = 0; i < count; ++i)
        {
            const VkBuffer buffer = buffer_pool_.device(slot_indices[i]).buffer;
            static_assert(sizeof(buffer) <= sizeof(std::uintptr_t), "VkBuffer cache key must fit uintptr_t");
            std::memcpy(&handles[static_cast<std::size_t>(i)], &buffer, sizeof(buffer));
            ranges[static_cast<std::size_t>(i)] =
                static_cast<std::size_t>(buffer_ranges != nullptr ? buffer_ranges[i] : field_bytes);
        }
        const bool cached =
            pipeline.descriptor_cache.matches_or_update(handles.data(), ranges.data(), static_cast<std::size_t>(count));
        compute_telemetry_.note_descriptor_lookup(cached);
        if (cached)
            return;

        std::array<VkDescriptorBufferInfo, kMaxPooledBindings> infos{};
        std::array<VkWriteDescriptorSet, kMaxPooledBindings>   writes{};
        for (int i = 0; i < count; ++i)
        {
            const auto idx              = static_cast<std::size_t>(i);
            infos[idx]                  = {buffer_pool_.device(slot_indices[i]).buffer,
                                           0,
                                           buffer_ranges != nullptr ? buffer_ranges[i] : field_bytes};
            writes[idx]                 = {};
            writes[idx].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[idx].dstSet          = pipeline.descriptor_set;
            writes[idx].dstBinding      = static_cast<uint32_t>(i);
            writes[idx].descriptorCount = 1;
            writes[idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[idx].pBufferInfo     = &infos[idx];
        }
        vkUpdateDescriptorSets_(device_context_.device, static_cast<uint32_t>(count), writes.data(), 0, nullptr);
    }

    /** @brief Map a linear workgroup count across X/Y within selected-device limits. */
    void record_linear_group_dispatch(uint32_t group_count, KernelId kernel)
    {
        const auto&    limits   = device_context_.selected_device_properties.limits;
        const uint32_t groups_x = std::min(group_count, limits.maxComputeWorkGroupCount[0]);
        const uint32_t groups_y =
            static_cast<uint32_t>((static_cast<std::uint64_t>(group_count) + groups_x - 1u) / groups_x);
        vkCmdDispatch_(cmd_buf_, groups_x, groups_y, 1);
        ++pending_dispatch_count_;
        ++pending_kernel_dispatch_counts_[static_cast<std::size_t>(kernel)];
    }

    bool checked_field_dispatch(
        int x, int y, int z, uint32_t& total_points, VkDeviceSize& field_bytes, const char* label) const
    {
        std::size_t elements = 0;
        std::string error;
        if (!tmv::vulkan_dispatch::checked_extent(x, y, z, sizeof(float), elements, field_bytes, error) ||
            elements > std::numeric_limits<uint32_t>::max() ||
            field_bytes > device_context_.selected_device_properties.limits.maxStorageBufferRange)
        {
            if (error.empty())
            {
                error = elements > std::numeric_limits<uint32_t>::max()
                            ? "linear invocation count exceeds uint32 shader indexing"
                            : "field exceeds maxStorageBufferRange";
            }
            log_vulkan_warning(std::string(label) + " refused: " + error);
            return false;
        }
        total_points = static_cast<uint32_t>(elements);
        return true;
    }

    // ── Generic 3-field dispatch helper (uses buffer pool) ──────────

    bool dispatch_field3_kernel(ComputePipelineState& pipeline,
                                KernelId              kernel,
                                const void*           push_constants,
                                const float*          input_a,
                                const float*          input_b,
                                float*                output,
                                int                   nr,
                                int                   nth,
                                int                   nz,
                                int                   input_count,
                                uint32_t              workgroup_divisor)
    {
        if (!submission_lifecycle_.can_dispatch())
        {
            return false;
        }
        std::size_t  total_cells = 0;
        VkDeviceSize field_bytes = 0;
        std::string  extent_error;
        if (!tmv::vulkan_dispatch::checked_extent(nr, nth, nz, sizeof(float), total_cells, field_bytes, extent_error) ||
            total_cells > std::numeric_limits<uint32_t>::max() || workgroup_divisor == 0 ||
            workgroup_divisor > total_cells ||
            field_bytes > device_context_.selected_device_properties.limits.maxStorageBufferRange)
        {
            log_vulkan_warning("field3 dispatch refused: " +
                               (extent_error.empty()
                                    ? std::string("invalid invocation domain or field exceeds device limits")
                                    : extent_error));
            return false;
        }
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
        const bool unified    = device_context_.has_unified_memory;
        auto       upload_ptr = [&](int slot) -> void*
        { return unified ? buffer_pool_.device(slot).mapped : buffer_pool_.staging(slot).mapped; };

        // Upload inputs
        tracked_memcpy(upload_ptr(slots[0]), input_a, field_bytes);
        if (input_count >= 2 && input_b != nullptr)
        {
            tracked_memcpy(upload_ptr(slots[1]), input_b, field_bytes);
        }
        tracked_memset(upload_ptr(slots[buf_count - 1]), 0, field_bytes);

        // Record command buffer
        vkResetCommandBuffer_(cmd_buf_, 0);
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (begin_profiled_command_buffer(begin_info) != VK_SUCCESS)
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
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }
        else
        {
            // Discrete path: copy staging → device
            VkBufferCopy copy_region{};
            copy_region.size = field_bytes;
            for (int i = 0; i < buf_count; ++i)
            {
                tracked_cmd_copy_buffer(
                    buffer_pool_.staging(slots[i]).buffer, buffer_pool_.device(slots[i]).buffer, 1, &copy_region);
            }

            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }

        // Bind pipeline and descriptors
        vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
        vkCmdBindDescriptorSets_(cmd_buf_,
                                 VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipeline.pipeline_layout,
                                 0,
                                 1,
                                 &pipeline.descriptor_set,
                                 0,
                                 nullptr);

        // Push constants
        vkCmdPushConstants_(cmd_buf_,
                            pipeline.pipeline_layout,
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0,
                            pipeline.push_constant_size,
                            push_constants);

        // Dispatch
        const uint32_t total_work      = static_cast<uint32_t>(workgroup_divisor);
        const uint32_t workgroup_count = 1u + (total_work - 1u) / 64u;
        record_linear_group_dispatch(workgroup_count, kernel);

        if (unified)
        {
            // Unified path: barrier so CPU can read GPU writes after fence
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }
        else
        {
            // Discrete path: copy device output → staging
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);

            VkBufferCopy copy_region{};
            copy_region.size = field_bytes;
            tracked_cmd_copy_buffer(buffer_pool_.device(slots[buf_count - 1]).buffer,
                                    buffer_pool_.staging(slots[buf_count - 1]).buffer,
                                    1,
                                    &copy_region);
        }

        if (end_profiled_command_buffer() != VK_SUCCESS)
        {
            release_pool_slots(slots, buf_count);
            return false;
        }

        // Submit and wait
        VkSubmitInfo submit_info{};
        submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers    = &cmd_buf_;
        if (reset_fence_and_submit("dispatch_field3_kernel", submit_info) != VK_SUCCESS)
        {
            release_pool_slots(slots, buf_count);
            return false;
        }

        if (timed_wait_fence("dispatch_field3_kernel") != VK_SUCCESS)
        {
            release_pool_slots(slots, buf_count);
            return false;
        }

        // Download output: device buffer on unified, staging on discrete
        const void* output_src = unified ? buffer_pool_.device(slots[buf_count - 1]).mapped
                                         : buffer_pool_.staging(slots[buf_count - 1]).mapped;
        tracked_memcpy(output, output_src, field_bytes);
        release_pool_slots(slots, buf_count);
        return true;
    }

    // ── Multi-field dispatch helper (uses buffer pool) ───────────────

    // `input_bytes` (optional, GPU audit Phase 4.3): per-input upload
    // byte counts for compact buffers - 1D profiles upload nz floats into
    // a field-sized slot instead of being broadcast across (i, j) on the
    // host. nullptr means every input is a full field. Outputs are always
    // full fields. `input_names` enables exact-generation resident reuse and
    // publishes a fresh binding after success. `output_write_only` skips
    // initialization uploads only for outputs the shader overwrites fully.
    // A transform whose outputs supersede named inputs can disable publishing
    // those now-stale input bindings and invalidate them in its caller.
    // `descriptor_logical_order` maps descriptor binding -> logical
    // input/output slot for kernels whose ABI interleaves outputs and inputs.
    bool dispatch_multi_field_kernel(ComputePipelineState&                pipeline,
                                     KernelId                             kernel,
                                     const void*                          push_constants,
                                     const float* const*                  inputs,
                                     int                                  input_count,
                                     float* const*                        outputs,
                                     int                                  output_count,
                                     int                                  nr,
                                     int                                  nth,
                                     int                                  nz,
                                     uint32_t                             total_invocations,
                                     const VkDeviceSize*                  input_bytes                        = nullptr,
                                     const char* const*                   input_names                        = nullptr,
                                     const bool*                          output_write_only                  = nullptr,
                                     bool                                 publish_named_inputs_after_success = true,
                                     const std::uint8_t*                  descriptor_logical_order           = nullptr,
                                     SlowApplyDispatch::CompletedObserver completed_observer                 = nullptr,
                                     void*                                completed_observer_context         = nullptr,
                                     int                                  descriptor_binding_count           = 0,
                                     SgsApplyDispatch::CompletedObserver  six_field_completed_observer       = nullptr,
                                     void*       six_field_completed_observer_context                        = nullptr,
                                     const bool* immutable_inputs                                            = nullptr,
                                     const char* const* output_names                                         = nullptr,
                                     std::size_t push_constant_count                                         = 1)
    {
        if (!submission_lifecycle_.can_dispatch() || push_constant_count == 0 || push_constant_count > 64)
        {
            return false;
        }
        std::size_t  total_cells = 0;
        VkDeviceSize field_bytes = 0;
        std::string  extent_error;
        if (!tmv::vulkan_dispatch::checked_extent(nr, nth, nz, sizeof(float), total_cells, field_bytes, extent_error) ||
            total_cells > std::numeric_limits<uint32_t>::max() || total_invocations == 0 ||
            total_invocations > total_cells ||
            field_bytes > device_context_.selected_device_properties.limits.maxStorageBufferRange)
        {
            log_vulkan_warning("multi-field dispatch refused: " +
                               (extent_error.empty()
                                    ? std::string("invalid invocation domain or field exceeds device limits")
                                    : extent_error));
            return false;
        }
        bool has_write_only_output = false;
        if (output_write_only != nullptr)
        {
            for (int i = 0; i < output_count; ++i)
            {
                has_write_only_output = has_write_only_output || output_write_only[i];
            }
        }
        if (has_write_only_output &&
            !tmv::vulkan_residency::covers_full_field(total_invocations, static_cast<std::uint64_t>(total_cells)))
        {
            log_vulkan_warning("multi-field dispatch refused: write-only output optimization "
                               "requires a full-field invocation domain");
            return false;
        }
        const int buf_count        = input_count + output_count;
        const int descriptor_count = descriptor_binding_count > 0 ? descriptor_binding_count : buf_count;

        constexpr int kMaxPooledBuffers = kMaxPooledBindings;
        if (buf_count < 1 || buf_count > kMaxPooledBuffers)
        {
            log_vulkan_warning("multi-field dispatch refused: buffer count " + std::to_string(buf_count) +
                               " outside pool-slot capacity 1.." + std::to_string(kMaxPooledBuffers));
            return false;
        }
        if (descriptor_count < buf_count || descriptor_count > kMaxPooledBuffers ||
            (descriptor_count != buf_count && descriptor_logical_order == nullptr))
        {
            log_vulkan_warning("multi-field dispatch refused: invalid descriptor span");
            return false;
        }

        VkDeviceSize logical_buffer_sizes[kMaxPooledBuffers];
        for (int i = 0; i < input_count; ++i)
        {
            logical_buffer_sizes[i] = input_bytes != nullptr && input_bytes[i] > 0 ? input_bytes[i] : field_bytes;
            if (logical_buffer_sizes[i] > device_context_.selected_device_properties.limits.maxStorageBufferRange ||
                logical_buffer_sizes[i] % sizeof(float) != 0)
            {
                log_vulkan_warning("multi-field dispatch refused: invalid input buffer size");
                return false;
            }
        }
        for (int i = 0; i < output_count; ++i)
        {
            logical_buffer_sizes[input_count + i] = field_bytes;
        }

        int slots[kMaxPooledBuffers];
        for (int i = 0; i < kMaxPooledBuffers; ++i)
        {
            slots[i] = -1;
        }
        bool         slot_named[kMaxPooledBuffers]     = {};
        bool         input_reusable[kMaxPooledBuffers] = {};
        bool         acquire_failed                    = false;
        const char*  named_names[kMaxPooledBuffers];
        VkDeviceSize named_sizes[kMaxPooledBuffers];
        int          named_logical_indices[kMaxPooledBuffers];
        int          named_slots[kMaxPooledBuffers];
        int          named_count = 0;
        int          transient_logical_indices[kMaxPooledBuffers];
        int          transient_count = 0;
        for (int i = 0; i < buf_count; ++i)
        {
            const char* name = i < input_count
                ? (input_names != nullptr ? input_names[i] : nullptr)
                : (output_names != nullptr ? output_names[i - input_count] : nullptr);
            if (name != nullptr)
            {
                slot_named[i]                      = true;
                named_names[named_count]           = name;
                named_sizes[named_count]           = logical_buffer_sizes[i];
                named_logical_indices[named_count] = i;
                ++named_count;
            }
            else
            {
                transient_logical_indices[transient_count++] = i;
            }
        }
        if (named_count > 0)
        {
            acquire_failed = !named_slots_acquire_many(named_names, named_sizes, named_count, named_slots);
            if (!acquire_failed)
            {
                for (int n = 0; n < named_count; ++n)
                {
                    const int logical = named_logical_indices[n];
                    slots[logical]    = named_slots[n];
                    if (logical >= input_count)
                    {
                        // Allocate version metadata before dispatch so final publication cannot allocate.
                        named_slot_commit(named_names[n], outputs[logical - input_count], logical_buffer_sizes[logical]);
                        named_slot_invalidate(named_names[n]);
                        continue;
                    }
                    input_reusable[logical] =
                        immutable_inputs != nullptr && immutable_inputs[logical]
                            ? active_resident_fields().reusable(
                                  named_names[n],
                                  inputs[logical],
                                  {static_cast<std::size_t>(logical_buffer_sizes[logical]) / sizeof(float),
                                   sizeof(float)},
                                  true)
                            : named_slot_reusable(named_names[n], inputs[logical], logical_buffer_sizes[logical]);
                    if (!input_reusable[logical])
                    {
                        named_slot_commit(named_names[n], inputs[logical], logical_buffer_sizes[logical]);
                        named_slot_invalidate(named_names[n]);
                    }
                }
            }
        }
        if (!acquire_failed && transient_count > 0)
        {
            int          transient_slots[kMaxPooledBuffers];
            VkDeviceSize transient_sizes[kMaxPooledBuffers];
            for (int i = 0; i < transient_count; ++i)
            {
                transient_sizes[i] = logical_buffer_sizes[transient_logical_indices[i]];
            }
            acquire_failed =
                !buffer_pool_.acquire_many(transient_sizes, static_cast<std::size_t>(transient_count), transient_slots);
            if (!acquire_failed)
            {
                for (int i = 0; i < transient_count; ++i)
                {
                    slots[transient_logical_indices[i]] = transient_slots[i];
                }
            }
        }
        const auto release_transients = [&]()
        {
            if (!submission_lifecycle_.may_release_transients())
            {
                return;
            }
            for (int i = 0; i < buf_count; ++i)
            {
                if (!slot_named[i] && slots[i] >= 0)
                {
                    buffer_pool_.release(slots[i]);
                }
            }
        };
        if (acquire_failed)
        {
            release_transients();
            log_vulkan_warning("failed to acquire pool slots for multi-field dispatch");
            return false;
        }
        std::array<int, kMaxPooledBuffers>          descriptor_slots{};
        std::array<VkDeviceSize, kMaxPooledBuffers> descriptor_ranges{};
        for (int binding = 0; binding < descriptor_count; ++binding)
        {
            const int logical =
                descriptor_logical_order != nullptr ? static_cast<int>(descriptor_logical_order[binding]) : binding;
            if (logical < 0 || logical >= buf_count)
            {
                release_transients();
                log_vulkan_warning("multi-field dispatch refused: descriptor binding map "
                                   "references an invalid logical slot");
                return false;
            }
            descriptor_slots[static_cast<std::size_t>(binding)]  = slots[logical];
            descriptor_ranges[static_cast<std::size_t>(binding)] = logical_buffer_sizes[logical];
        }
        update_pooled_descriptor_set(
            pipeline, descriptor_slots.data(), field_bytes, descriptor_count, descriptor_ranges.data());

        const bool unified    = device_context_.has_unified_memory;
        // The flat slow shader writes positive zero to every density-tendency
        // cell. Keep its descriptor allocation, but materialize that constant
        // only at the original publication position after a successful fence.
        const int zero_output = !unified && kernel == KernelId::CartesianCgridSlowTendencies &&
                                        completed_observer == nullptr && six_field_completed_observer == nullptr
                                    ? 3 : -1;
        auto       upload_ptr = [&](int slot) -> void*
        { return unified ? buffer_pool_.device(slot).mapped : buffer_pool_.staging(slot).mapped; };

        // Upload input fields (compact inputs copy only their own bytes)
        bool has_reused_device_input = false;
        for (int i = 0; i < input_count; ++i)
        {
            if (input_reusable[i])
            {
                has_reused_device_input = true;
                continue;
            }
            tracked_memcpy(upload_ptr(slots[i]), inputs[i], logical_buffer_sizes[i]);
        }
        // Upload read-write output data. A caller may mark a fully overwritten
        // output write-only and avoid initializing its allocation entirely.
        for (int i = 0; i < output_count; ++i)
        {
            if (output_write_only != nullptr && output_write_only[i])
            {
                compute_telemetry_.note_writeonly_output_upload_avoided(static_cast<std::uint64_t>(field_bytes));
                continue;
            }
            tracked_memcpy(upload_ptr(slots[input_count + i]), outputs[i], field_bytes);
        }

        // Record command buffer
        vkResetCommandBuffer_(cmd_buf_, 0);
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (begin_profiled_command_buffer(begin_info) != VK_SUCCESS)
        {
            release_transients();
            return false;
        }

        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;

        if (unified)
        {
            // Some named inputs can come directly from a shader in an
            // earlier submission while other inputs were just written by
            // the host. A fence wait completes that submission but does not
            // replace the Vulkan memory dependency that makes its storage
            // writes visible to this shader read.
            barrier.srcAccessMask =
                VK_ACCESS_HOST_WRITE_BIT | (has_reused_device_input ? VK_ACCESS_SHADER_WRITE_BIT : 0);
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_HOST_BIT |
                                      (has_reused_device_input ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : 0),
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }
        else
        {
            // Copy all staging → device (compact inputs copy their own bytes)
            for (int i = 0; i < buf_count; ++i)
            {
                if (i < input_count && input_reusable[i])
                {
                    continue;
                }
                if (i >= input_count && output_write_only != nullptr && output_write_only[i - input_count])
                {
                    continue;
                }
                VkBufferCopy copy_region{};
                copy_region.size = logical_buffer_sizes[i];
                tracked_cmd_copy_buffer(
                    buffer_pool_.staging(slots[i]).buffer, buffer_pool_.device(slots[i]).buffer, 1, &copy_region);
            }

            barrier.srcAccessMask =
                VK_ACCESS_TRANSFER_WRITE_BIT | (has_reused_device_input ? VK_ACCESS_SHADER_WRITE_BIT : 0);
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT |
                                      (has_reused_device_input ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : 0),
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }

        // Bind pipeline and descriptors
        vkCmdBindPipeline_(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
        vkCmdBindDescriptorSets_(cmd_buf_,
                                 VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipeline.pipeline_layout,
                                 0,
                                 1,
                                 &pipeline.descriptor_set,
                                 0,
                                 nullptr);

        const uint32_t workgroup_count = 1u + (total_invocations - 1u) / 64u;
        for (std::size_t stage = 0; stage < push_constant_count; ++stage)
        {
            vkCmdPushConstants_(cmd_buf_, pipeline.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                0, pipeline.push_constant_size,
                                static_cast<const unsigned char*>(push_constants) + stage * pipeline.push_constant_size);
            record_linear_group_dispatch(workgroup_count, kernel);
            if (stage + 1 < push_constant_count)
            {
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier_(cmd_buf_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
                                      0, nullptr, 0, nullptr);
            }
        }

        if (unified)
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_HOST_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);
        }
        else
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(cmd_buf_,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0,
                                  1,
                                  &barrier,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr);

            VkBufferCopy copy_region{};
            copy_region.size = field_bytes;
            for (int i = 0; i < output_count; ++i)
            {
                if (i == zero_output)
                    continue;
                tracked_cmd_copy_buffer(buffer_pool_.device(slots[input_count + i]).buffer,
                                        buffer_pool_.staging(slots[input_count + i]).buffer,
                                        1,
                                        &copy_region);
            }
        }

        if (end_profiled_command_buffer() != VK_SUCCESS)
        {
            release_transients();
            return false;
        }

        // Submit and wait
        VkSubmitInfo submit_info{};
        submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers    = &cmd_buf_;
        if (reset_fence_and_submit(pipeline.debug_name.c_str(), submit_info) != VK_SUCCESS)
        {
            release_transients();
            return false;
        }

        if (timed_wait_fence(pipeline.debug_name.c_str()) != VK_SUCCESS)
        {
            release_transients();
            return false;
        }

        // A report-bearing slow commit needs the actual device result and the
        // still-untouched old host fields at the same instant. Observe the
        // mapped output buffers after the fence and before copy-back; this
        // adds neither snapshots nor another submission/fence.
        if (completed_observer != nullptr || six_field_completed_observer != nullptr)
        {
            const auto output_src = [&](int i) -> const float*
            {
                return static_cast<const float*>(unified ? buffer_pool_.device(slots[input_count + i]).mapped
                                                         : buffer_pool_.staging(slots[input_count + i]).mapped);
            };
            try
            {
                if (completed_observer != nullptr && output_count != 4)
                {
                    throw std::logic_error("slow completed observer requires four output fields");
                }
                if (six_field_completed_observer != nullptr && output_count != 6)
                {
                    throw std::logic_error("SGS completed observer requires six output fields");
                }
                if (completed_observer != nullptr)
                {
                    completed_observer(
                        completed_observer_context, output_src(0), output_src(1), output_src(2), output_src(3));
                }
                if (six_field_completed_observer != nullptr)
                {
                    six_field_completed_observer(six_field_completed_observer_context,
                                                 output_src(0),
                                                 output_src(1),
                                                 output_src(2),
                                                 output_src(3),
                                                 output_src(4),
                                                 output_src(5));
                }
            }
            catch (...)
            {
                release_transients();
                throw;
            }
        }

        // Download output fields
        for (int i = 0; i < output_count; ++i)
        {
            if (i == zero_output)
            {
                std::memset(outputs[i], 0, field_bytes);
                continue;
            }
            const void* src = unified ? buffer_pool_.device(slots[input_count + i]).mapped
                                      : buffer_pool_.staging(slots[input_count + i]).mapped;
            tracked_memcpy(outputs[i], src, field_bytes);
        }
        if (input_names != nullptr && publish_named_inputs_after_success)
        {
            for (int i = 0; i < input_count; ++i)
            {
                if (input_names[i] != nullptr)
                {
                    named_slot_commit(input_names[i], inputs[i], logical_buffer_sizes[i]);
                }
            }
        }
        if (output_names != nullptr)
            for (int i = 0; i < output_count; ++i)
                if (output_names[i] != nullptr)
                {
                    active_resident_fields().note_device_write(output_names[i], outputs[i], {total_cells, sizeof(float)});
                    active_resident_fields().note_host_synced(output_names[i]);
                }
        release_transients();
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

    static std::string spirv_fnv1a64(const std::vector<uint32_t>& words)
    {
        std::uint64_t     value = 14695981039346656037ULL;
        const auto*       bytes = reinterpret_cast<const std::uint8_t*>(words.data());
        const std::size_t size  = words.size() * sizeof(uint32_t);
        for (std::size_t i = 0; i < size; ++i)
        {
            value ^= bytes[i];
            value *= 1099511628211ULL;
        }
        std::ostringstream stream;
        stream << std::hex << std::setfill('0') << std::setw(16) << value;
        return stream.str();
    }

    tmv_vulkan::pipeline_cache_file::Identity pipeline_cache_identity() const
    {
        tmv_vulkan::pipeline_cache_file::Identity identity;
        const VkPhysicalDeviceProperties&         properties = device_context_.selected_device_properties;
        identity.vendor_id                                   = properties.vendorID;
        identity.device_id                                   = properties.deviceID;
        identity.driver_version                              = properties.driverVersion;
        static_assert(VK_UUID_SIZE == 16, "Vulkan cache UUID size changed");
        std::copy_n(properties.pipelineCacheUUID, identity.uuid.size(), identity.uuid.begin());
        identity.manifest_signature = tmv_vulkan::pipeline_cache_manifest_signature(
            nimbus::compute::kernel_manifest(), requested_config_.startup_plan.get());
        if (properties.vendorID == 0x10de)
        {
            identity.manifest_signature += "-" + std::string(tmv_vulkan::division_variant_signature);
        }
#ifdef VK_NV_cuda_kernel_launch
        if (cuda_assist_enabled_) identity.manifest_signature += "-cuda-" + std::string(tmv_vulkan::cuda_flux_sha256);
#endif
        return identity;
    }

    static bool persistent_pipeline_cache_enabled()
    {
        const char* value = std::getenv("NIMBUS_VULKAN_PIPELINE_CACHE");
        if (value == nullptr)
            return true;
        const std::string setting(value);
        return setting != "0" && setting != "false" && setting != "FALSE" && setting != "off" && setting != "OFF";
    }

    static std::filesystem::path pipeline_cache_directory()
    {
        if (const char* override_dir = std::getenv("NIMBUS_VULKAN_PIPELINE_CACHE_DIR"))
        {
            if (override_dir[0] != '\0')
            {
                return std::filesystem::path(override_dir);
            }
        }

#if defined(_WIN32)
        if (const char* local_app_data = std::getenv("LOCALAPPDATA"))
        {
            return std::filesystem::path(local_app_data) / "Nimbus" / "cache" / "vulkan";
        }
#elif defined(__APPLE__)
        if (const char* home = std::getenv("HOME"))
        {
            return std::filesystem::path(home) / "Library" / "Caches" / "Nimbus" / "vulkan";
        }
#elif defined(__linux__)
        if (const char* xdg_cache = std::getenv("XDG_CACHE_HOME"))
        {
            if (xdg_cache[0] != '\0')
            {
                return std::filesystem::path(xdg_cache) / "nimbus" / "vulkan";
            }
        }
        if (const char* home = std::getenv("HOME"))
        {
            return std::filesystem::path(home) / ".cache" / "nimbus" / "vulkan";
        }
#endif
        return {};
    }

    static std::vector<std::uint8_t> read_pipeline_cache_file(const std::filesystem::path& path)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open())
            return {};

        const std::streamoff end = file.tellg();
        if (end < static_cast<std::streamoff>(tmv_vulkan::pipeline_cache_file::kHeaderBytes) ||
            end > static_cast<std::streamoff>(tmv_vulkan::pipeline_cache_file::kMaximumBytes))
        {
            return {};
        }

        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!file || static_cast<std::size_t>(file.gcount()) != bytes.size())
        {
            return {};
        }
        return bytes;
    }

    void initialize_pipeline_cache()
    {
        pipeline_cache_identity_ = pipeline_cache_identity();
        pipeline_cache_path_.clear();
        pipeline_cache_loaded_ = false;

        if (vkCreatePipelineCache_ == nullptr)
        {
            return;
        }

        std::vector<std::uint8_t> initial_data;
        if (persistent_pipeline_cache_enabled())
        {
            const std::filesystem::path directory = pipeline_cache_directory();
            if (!directory.empty())
            {
                pipeline_cache_path_ = directory / tmv_vulkan::pipeline_cache_file::filename(pipeline_cache_identity_);
                std::error_code exists_error;
                const bool      cache_file_exists = std::filesystem::exists(pipeline_cache_path_, exists_error);
                initial_data                      = read_pipeline_cache_file(pipeline_cache_path_);
                log_startup_memory("cache_input_loaded", nullptr, 0, initial_data.size());
                if (!initial_data.empty() &&
                    !tmv_vulkan::pipeline_cache_file::valid_vulkan_header(initial_data, pipeline_cache_identity_))
                {
                    log_vulkan_warning("ignored incompatible Vulkan pipeline cache '" + pipeline_cache_path_.string() +
                                       "'");
                    initial_data.clear();
                }
                else if (cache_file_exists && initial_data.empty())
                {
                    log_vulkan_warning("ignored empty, unreadable, or oversized Vulkan "
                                       "pipeline cache '" +
                                       pipeline_cache_path_.string() + "'");
                }
            }
        }
        else
        {
            log_vulkan_info("persistent pipeline cache disabled by "
                            "NIMBUS_VULKAN_PIPELINE_CACHE");
        }

        log_startup_memory("cache_input_ready", nullptr, 0, initial_data.size());
        VkPipelineCacheCreateInfo create_info{};
        create_info.sType           = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        create_info.initialDataSize = initial_data.size();
        create_info.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();

        VkResult result = vkCreatePipelineCache_(device_context_.device, &create_info, nullptr, &pipeline_cache_);
        if (result != VK_SUCCESS && !initial_data.empty())
        {
            // A driver can reject an otherwise well-keyed opaque payload. Drop
            // it and compile normally rather than disabling GPU dispatch.
            log_vulkan_warning("driver rejected persisted Vulkan pipeline cache (result=" + vk_result_string(result) +
                               "); retrying empty");
            create_info.initialDataSize = 0;
            create_info.pInitialData    = nullptr;
            result = vkCreatePipelineCache_(device_context_.device, &create_info, nullptr, &pipeline_cache_);
        }
        log_startup_memory(result == VK_SUCCESS ? "cache_created" : "cache_create_failed",
                           nullptr, 0, initial_data.size());
        if (result != VK_SUCCESS)
        {
            pipeline_cache_ = VK_NULL_HANDLE;
            log_vulkan_warning("vkCreatePipelineCache failed (result=" + vk_result_string(result) +
                               "); compiling pipelines without a cache");
            return;
        }

        pipeline_cache_loaded_ = !initial_data.empty();
        log_vulkan_info(pipeline_cache_loaded_ ? "loaded compatible Vulkan pipeline cache"
                                               : "created empty Vulkan pipeline cache");
    }

    void persist_pipeline_cache()
    {
        if (pipeline_cache_ == VK_NULL_HANDLE || vkGetPipelineCacheData_ == nullptr || pipeline_cache_path_.empty())
        {
            return;
        }

        std::size_t byte_count = 0;
        VkResult    result     = vkGetPipelineCacheData_(device_context_.device, pipeline_cache_, &byte_count, nullptr);
        if (result != VK_SUCCESS || byte_count < tmv_vulkan::pipeline_cache_file::kHeaderBytes ||
            byte_count > tmv_vulkan::pipeline_cache_file::kMaximumBytes)
        {
            log_vulkan_warning("did not persist Vulkan pipeline cache: invalid size query");
            return;
        }

        std::vector<std::uint8_t> bytes(byte_count);
        result = vkGetPipelineCacheData_(device_context_.device, pipeline_cache_, &byte_count, bytes.data());
        if (result != VK_SUCCESS || byte_count < tmv_vulkan::pipeline_cache_file::kHeaderBytes ||
            byte_count > bytes.size())
        {
            log_vulkan_warning("did not persist Vulkan pipeline cache: data query failed "
                               "(result=" +
                               vk_result_string(result) + ")");
            return;
        }
        bytes.resize(byte_count);
        log_startup_memory("cache_payload_materialized", nullptr, 0, bytes.size());
        if (!tmv_vulkan::pipeline_cache_file::valid_vulkan_header(bytes, pipeline_cache_identity_))
        {
            log_vulkan_warning("did not persist Vulkan pipeline cache: driver returned an "
                               "incompatible header");
            return;
        }

        std::error_code directory_error;
        std::filesystem::create_directories(pipeline_cache_path_.parent_path(), directory_error);
        if (directory_error)
        {
            log_vulkan_warning("could not create Vulkan pipeline-cache directory: " + directory_error.message());
            return;
        }

        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
#if defined(_WIN32)
        const unsigned long process_id = GetCurrentProcessId();
#else
        const long process_id = static_cast<long>(::getpid());
#endif
        std::filesystem::path temporary = pipeline_cache_path_;
        temporary += ".tmp-" + std::to_string(process_id) + "-" + std::to_string(nonce);

        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        file.flush();
        const bool write_ok = file.good();
        file.close();
        if (!write_ok)
        {
            std::remove(temporary.string().c_str());
            log_vulkan_warning("could not write temporary Vulkan pipeline cache");
            return;
        }

#if defined(_WIN32)
        const bool replaced = MoveFileExA(temporary.string().c_str(),
                                          pipeline_cache_path_.string().c_str(),
                                          MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
        const bool replaced = std::rename(temporary.string().c_str(), pipeline_cache_path_.string().c_str()) == 0;
#endif
        if (!replaced)
        {
            std::remove(temporary.string().c_str());
            log_vulkan_warning("could not atomically publish Vulkan pipeline cache");
            return;
        }

        log_vulkan_info("persisted Vulkan pipeline cache (" + std::to_string(bytes.size()) + " bytes)");
    }

    // Directory containing the running executable, canonicalized (symlinks and
    // ./.. resolved). Empty if it cannot be determined (unknown platform or a
    // failed syscall) -- callers fall back to cwd-relative lookup. Computed once
    // and cached; the C++11 magic-static init is thread-safe.
    static const std::string& executable_directory()
    {
        static const std::string dir = []() -> std::string
        {
            std::string raw;
#if defined(__APPLE__)
            uint32_t size = 0;
            _NSGetExecutablePath(nullptr, &size); // probe for the length
            std::vector<char> buf(size + 1, '\0');
            if (_NSGetExecutablePath(buf.data(), &size) != 0)
            {
                return {};
            }
            raw.assign(buf.data());
#elif defined(__linux__)
            char          link_buf[4096];
            const ssize_t len = ::readlink("/proc/self/exe", link_buf, sizeof(link_buf) - 1);
            if (len <= 0)
            {
                return {};
            }
            link_buf[len] = '\0';
            raw.assign(link_buf);
#elif defined(_WIN32)
            std::vector<char> buffer(32768, '\0');
            const DWORD       length = GetModuleFileNameA(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
            if (length == 0 || length >= buffer.size())
            {
                return {};
            }
            raw.assign(buffer.data(), length);
#else
            return {};
#endif
#if defined(__APPLE__) || defined(__linux__)
            // Canonicalize so the "../" candidate below is stable regardless of
            // how the binary was invoked (relative path, symlink, PATH lookup).
            if (char* canon = ::realpath(raw.c_str(), nullptr))
            {
                raw.assign(canon);
                std::free(canon);
            }
            const std::size_t slash = raw.find_last_of('/');
            if (slash == std::string::npos)
            {
                return {};
            }
            return raw.substr(0, slash);
#elif defined(_WIN32)
            const std::size_t slash = raw.find_last_of("/\\");
            if (slash == std::string::npos)
            {
                return {};
            }
            return raw.substr(0, slash);
#endif
        }();
        return dir;
    }

    static std::string resolve_shader_path(const std::string& shader_filename)
    {
        const char* override_dir = std::getenv("TMV_SHADER_DIR");
        if (override_dir != nullptr && override_dir[0] != '\0')
        {
            return std::string(override_dir) + "/" + shader_filename;
        }

        std::vector<std::string> candidates;
        // Prefer paths relative to the executable so the binary resolves its
        // shaders from any working directory. cwd-relative lookup alone made a
        // run from any other directory silently degrade to CPU-only.
        const std::string& exe_dir = executable_directory();
        if (!exe_dir.empty())
        {
            candidates.push_back(exe_dir + "/vulkan/shaders/compute/" + shader_filename);
            candidates.push_back(exe_dir + "/../vulkan/shaders/compute/" + shader_filename);
            candidates.push_back(exe_dir + "/shaders/compute/" + shader_filename);
        }
        // cwd-relative fallbacks (repo-root invocation; backward compatible).
        candidates.push_back("vulkan/shaders/compute/" + shader_filename);
        candidates.push_back("../vulkan/shaders/compute/" + shader_filename);
        candidates.push_back("shaders/compute/" + shader_filename);

        for (const auto& candidate : candidates)
        {
            std::ifstream test(candidate);
            if (test.good())
            {
                return candidate;
            }
        }
        return candidates.front();
    }

    // ── Pipeline factory ────────────────────────────────────────────────

    bool create_pipeline_state(const nimbus::compute::KernelManifestEntry& kernel,
                               ComputePipelineState&                       out,
                               std::string&                                error)
    {
        if (kernel.precision == nimbus::compute::KernelPrecision::Float32StorageFloat64Arithmetic &&
            (!float64_controls_enabled_ || !strict_fma64_enabled_ || !device_context_.selected_device_features.shaderInt64))
        {
            error = "kernel requires native FP64 round-to-nearest, shaderFmaFloat64, and integer storage conversion";
            return false;
        }
        if ((kernel.id == KernelId::ConservativeReduction || kernel.id == KernelId::ConservativeMomentum || kernel.id == KernelId::HeldSources || kernel.id == KernelId::SlabColumn || kernel.id == KernelId::SimpleGrey || kernel.id == KernelId::ColumnRemap || kernel.id == KernelId::ReferenceAtmosphere || kernel.id == KernelId::MovingEnergy || kernel.id == KernelId::MovingMomentum || kernel.id == KernelId::EarthGeometry || kernel.id == KernelId::ConservativeRemap || kernel.id == KernelId::CollocatedSgs || kernel.id == KernelId::RkTendency ||
             kernel.id == KernelId::RkCombination || kernel.id == KernelId::MilbrandtSedimentation ||
             kernel.id == KernelId::MilbrandtCold || kernel.id == KernelId::MilbrandtMelting ||
             kernel.id == KernelId::MilbrandtCloudAdjustment || kernel.id == KernelId::MilbrandtWarmRain ||
             kernel.id == KernelId::ThompsonCloudAdjustment || kernel.id == KernelId::ThompsonWarmRain ||
             kernel.id == KernelId::ThompsonCloudPhase || kernel.id == KernelId::ThompsonFreezing ||
             kernel.id == KernelId::ThompsonDeposition || kernel.id == KernelId::ThompsonCompletion) &&
            !float_controls_enabled_)
        {
            error = "compensated microphysics requires FP32 signed-zero/NaN preservation and round-to-nearest";
            return false;
        }
        const std::string             shader_spv_filename = kernel.artifact_name;
        const uint32_t                binding_count       = kernel.resources.descriptor_span;
        const uint32_t                push_constant_size  = kernel.constants.bytes;
        const VkPhysicalDeviceLimits& limits              = device_context_.selected_device_properties.limits;
        if (binding_count > limits.maxPerStageDescriptorStorageBuffers ||
            binding_count > limits.maxDescriptorSetStorageBuffers)
        {
            error = shader_spv_filename + " requires " + std::to_string(binding_count) +
                    " storage buffers but device limits are per-stage=" +
                    std::to_string(limits.maxPerStageDescriptorStorageBuffers) +
                    ", per-set=" + std::to_string(limits.maxDescriptorSetStorageBuffers);
            return false;
        }
        if (push_constant_size > limits.maxPushConstantsSize)
        {
            error = shader_spv_filename + " requires " + std::to_string(push_constant_size) +
                    " push-constant bytes but device limit is " + std::to_string(limits.maxPushConstantsSize);
            return false;
        }

        std::string selected_artifact = shader_spv_filename;
        std::string expected_hash = kernel.artifact_hash;
        if (device_context_.selected_device_properties.vendorID == 0x10de)
        {
            for (const auto& variant : tmv_vulkan::division_variants)
            {
                if (variant.source_artifact != shader_spv_filename)
                    continue;
                if (variant.source_hash != expected_hash)
                {
                    error = "native division variant does not match the kernel manifest: " + shader_spv_filename;
                    return false;
                }
                if (!device_context_.selected_device_features.shaderInt64)
                {
                    error = "NVIDIA round-to-nearest division requires native shaderInt64 support";
                    return false;
                }
                if (variant.requires_fma && !strict_fma_enabled_)
                {
                    error = "NVIDIA compensated arithmetic requires a simulator built with VK_KHR_shader_fma support and a driver exposing shaderFmaFloat32";
                    return false;
                }
                selected_artifact = variant.artifact;
                expected_hash = variant.artifact_hash;
                log_vulkan_info("native strict FP32 arithmetic: " + std::string(kernel.operation_id));
                break;
            }
        }
        const std::string           shader_path = resolve_shader_path(selected_artifact);
        const std::vector<uint32_t> spirv       = load_spirv(shader_path);
        if (spirv.empty())
        {
            error = "failed to load SPIR-V from '" + shader_path + "'";
            return false;
        }
        const std::string loaded_hash = spirv_fnv1a64(spirv);
        if (loaded_hash != expected_hash)
        {
            error =
                selected_artifact + " hash mismatch: manifest=" + expected_hash + ", loaded=" + loaded_hash;
            return false;
        }
        log_vulkan_info("loaded shader: " + shader_path + " (" + std::to_string(spirv.size() * 4) +
                        " bytes, fnv1a64=" + loaded_hash + ")");

        // Build into a local owner.  `out` remains either its previous valid
        // state or empty until every dependent Vulkan object exists.
        ComputePipelineState candidate{};
        candidate.debug_name = shader_spv_filename;

        VkShaderModuleCreateInfo shader_info{};
        shader_info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shader_info.codeSize = spirv.size() * sizeof(uint32_t);
        shader_info.pCode    = spirv.data();

        VkShaderModule shader_module = VK_NULL_HANDLE;
        if (vkCreateShaderModule_(device_context_.device, &shader_info, nullptr, &shader_module) != VK_SUCCESS)
        {
            error = "vkCreateShaderModule failed for " + shader_spv_filename;
            return false;
        }

        // Descriptor set layout: N SSBO bindings
        std::vector<VkDescriptorSetLayoutBinding> bindings(binding_count);
        for (uint32_t i = 0; i < binding_count; ++i)
        {
            bindings[i]                 = {};
            bindings[i].binding         = i;
            bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = binding_count;
        layout_info.pBindings    = bindings.data();

        if (vkCreateDescriptorSetLayout_(
                device_context_.device, &layout_info, nullptr, &candidate.descriptor_set_layout) != VK_SUCCESS)
        {
            vkDestroyShaderModule_(device_context_.device, shader_module, nullptr);
            error = "vkCreateDescriptorSetLayout failed for " + shader_spv_filename;
            return false;
        }

        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset     = 0;
        push_range.size       = push_constant_size;

        VkPipelineLayoutCreateInfo pl_info{};
        pl_info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_info.setLayoutCount         = 1;
        pl_info.pSetLayouts            = &candidate.descriptor_set_layout;
        pl_info.pushConstantRangeCount = 1;
        pl_info.pPushConstantRanges    = &push_range;

        if (vkCreatePipelineLayout_(device_context_.device, &pl_info, nullptr, &candidate.pipeline_layout) !=
            VK_SUCCESS)
        {
            vkDestroyShaderModule_(device_context_.device, shader_module, nullptr);
            destroy_pipeline_state(candidate);
            error = "vkCreatePipelineLayout failed for " + shader_spv_filename;
            return false;
        }

        VkComputePipelineCreateInfo pipe_info{};
        pipe_info.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipe_info.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipe_info.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        pipe_info.stage.module = shader_module;
        pipe_info.stage.pName  = "main";
        pipe_info.layout       = candidate.pipeline_layout;

        VkResult result = vkCreateComputePipelines_(
            device_context_.device, pipeline_cache_, 1, &pipe_info, nullptr, &candidate.pipeline);
        vkDestroyShaderModule_(device_context_.device, shader_module, nullptr);
        if (result != VK_SUCCESS)
        {
            destroy_pipeline_state(candidate);
            error = "vkCreateComputePipelines failed for " + shader_spv_filename;
            return false;
        }

        const uint32_t descriptor_set_count = kernel.id == KernelId::Diffusion ? 6u
                                              : kernel.id == KernelId::DomainRecyclingHomogeneousFieldShift
                                                  ? static_cast<uint32_t>(ComputePipelineState::kMaximumDescriptorSets)
                                                  : 1u;

        VkDescriptorPoolSize pool_size{};
        pool_size.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = binding_count * descriptor_set_count;

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets       = descriptor_set_count;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes    = &pool_size;

        if (vkCreateDescriptorPool_(device_context_.device, &pool_info, nullptr, &candidate.descriptor_pool) !=
            VK_SUCCESS)
        {
            destroy_pipeline_state(candidate);
            error = "vkCreateDescriptorPool failed for " + shader_spv_filename;
            return false;
        }

        VkDescriptorSetAllocateInfo ds_alloc{};
        ds_alloc.sType          = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ds_alloc.descriptorPool = candidate.descriptor_pool;
        std::array<VkDescriptorSetLayout, ComputePipelineState::kMaximumDescriptorSets> set_layouts{};
        set_layouts.fill(candidate.descriptor_set_layout);
        ds_alloc.descriptorSetCount = descriptor_set_count;
        ds_alloc.pSetLayouts        = set_layouts.data();

        if (vkAllocateDescriptorSets_(device_context_.device, &ds_alloc, candidate.descriptor_sets.data()) !=
            VK_SUCCESS)
        {
            destroy_pipeline_state(candidate);
            error = "vkAllocateDescriptorSets failed for " + shader_spv_filename;
            return false;
        }
        candidate.descriptor_set       = candidate.descriptor_sets[0];
        candidate.descriptor_set_count = descriptor_set_count;

        candidate.binding_count      = binding_count;
        candidate.push_constant_size = push_constant_size;
        candidate.lifecycle          = &submission_lifecycle_;
        candidate.owner_generation   = submission_lifecycle_.generation();

        destroy_pipeline_state(out);
        out = std::move(candidate);
        return true;
    }

    void destroy_pipeline_state(ComputePipelineState& state)
    {
        if (state.pipeline != VK_NULL_HANDLE && vkDestroyPipeline_ != nullptr)
        {
            vkDestroyPipeline_(device_context_.device, state.pipeline, nullptr);
        }
        if (state.descriptor_pool != VK_NULL_HANDLE && vkDestroyDescriptorPool_ != nullptr)
        {
            vkDestroyDescriptorPool_(device_context_.device, state.descriptor_pool, nullptr);
        }
        if (state.pipeline_layout != VK_NULL_HANDLE && vkDestroyPipelineLayout_ != nullptr)
        {
            vkDestroyPipelineLayout_(device_context_.device, state.pipeline_layout, nullptr);
        }
        if (state.descriptor_set_layout != VK_NULL_HANDLE && vkDestroyDescriptorSetLayout_ != nullptr)
        {
            vkDestroyDescriptorSetLayout_(device_context_.device, state.descriptor_set_layout, nullptr);
        }
        state = {};
    }

    // ── Shared command infrastructure setup ───────────────────────────

    bool setup_command_infrastructure(std::string& error)
    {
        VkCommandPoolCreateInfo cp_info{};
        cp_info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cp_info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cp_info.queueFamilyIndex = device_context_.compute_queue_family_index;

        if (vkCreateCommandPool_(device_context_.device, &cp_info, nullptr, &cmd_pool_) != VK_SUCCESS)
        {
            error = "vkCreateCommandPool failed";
            return false;
        }

        VkCommandBufferAllocateInfo cb_alloc{};
        cb_alloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cb_alloc.commandPool        = cmd_pool_;
        cb_alloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cb_alloc.commandBufferCount = 1;

        if (vkAllocateCommandBuffers_(device_context_.device, &cb_alloc, &cmd_buf_) != VK_SUCCESS)
        {
            error = "vkAllocateCommandBuffers failed";
            return false;
        }

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        if (vkCreateFence_(device_context_.device, &fence_info, nullptr, &fence_) != VK_SUCCESS)
        {
            error = "vkCreateFence failed";
            return false;
        }

        const bool timestamp_functions_available = vkCreateQueryPool_ != nullptr && vkDestroyQueryPool_ != nullptr &&
                                                   vkCmdResetQueryPool_ != nullptr && vkCmdWriteTimestamp_ != nullptr &&
                                                   vkGetQueryPoolResults_ != nullptr;
        if (gpu_timing_enabled() && timestamp_functions_available &&
            device_context_.selected_compute_queue_family.timestampValidBits > 0)
        {
            VkQueryPoolCreateInfo query_info{};
            query_info.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            query_info.queryType  = VK_QUERY_TYPE_TIMESTAMP;
            query_info.queryCount = 2;
            if (vkCreateQueryPool_(device_context_.device, &query_info, nullptr, &timestamp_query_pool_) == VK_SUCCESS)
            {
                gpu_timestamp_available_ = true;
                log_vulkan_info("device timestamp telemetry enabled");
            }
            else
            {
                log_vulkan_warning("device timestamp query pool unavailable; retaining fence telemetry");
            }
        }
        else if (gpu_timing_enabled())
        {
            log_vulkan_warning("compute queue has no usable timestamp queries; retaining fence telemetry");
        }

        return true;
    }

    // ── Pipeline setup (all compute pipelines) ───────────────────────

    bool setup_compute_pipelines(std::string& error)
    {
        if (!setup_command_infrastructure(error))
            return false;

        if (!detect_unified_memory(error))
            return false;
        init_buffer_pool();
        log_startup_memory("infrastructure_ready");

        const char* manifest_error = nullptr;
        if (!nimbus::compute::validate_kernel_manifest(nimbus::compute::kernel_manifest(), &manifest_error))
        {
            error = std::string("invalid compute kernel manifest: ") +
                    (manifest_error != nullptr ? manifest_error : "unknown error");
            return false;
        }

        log_startup_memory("cache_initialize_begin");
        initialize_pipeline_cache();
        log_startup_memory("cache_initialize_end");

        if (!nimbus::compute::prepare_startup_pipelines(
                requested_config_.startup_plan.get(), nimbus::compute::kernel_manifest(),
                [&](const nimbus::compute::KernelManifestEntry& kernel, std::string& pipeline_error)
                {
                    return create_pipeline_state(kernel, pipeline(kernel.id), pipeline_error);
                },
                [&](const nimbus::compute::KernelManifestEntry& kernel)
                {
                    destroy_pipeline_state(pipeline(kernel.id));
                },
                [&](const nimbus::compute::KernelManifestEntry& kernel, bool success,
                    std::size_t pipeline_ordinal, const std::string& pipeline_error)
                {
                    if (success)
                    {
                        log_startup_memory("pipeline_ready", kernel.operation_id, pipeline_ordinal);
                        log_vulkan_info(std::string("pipeline ready: ") + kernel.operation_id);
                    }
                    else
                    {
                        log_startup_memory("pipeline_unavailable", kernel.operation_id, pipeline_ordinal);
                        if (requested_config_.startup_plan == nullptr && !kernel.startup_required)
                        {
                            log_vulkan_warning(std::string("pipeline unavailable: ") + kernel.operation_id + ": " + pipeline_error);
                        }
                    }
                }, error))
        {
            return false;
        }
        log_startup_memory("cache_persist_begin");
        persist_pipeline_cache();
        log_startup_memory("cache_persist_end");
        return true;
    }

    void destroy_compute_resources()
    {
#ifdef VK_NV_cuda_kernel_launch
        cuda_flux_.reset();
#endif
        destroy_buffer(staging_q_);
        destroy_buffer(staging_w_);
        destroy_buffer(staging_dz_);
        destroy_buffer(staging_dqdt_);
        destroy_buffer(device_q_);
        destroy_buffer(device_w_);
        destroy_buffer(device_dz_);
        destroy_buffer(device_dqdt_);

        // Named entries are non-owning pool indices.  Invalidate them before
        // their backing slots disappear so repeat initialization cannot index
        // a new pool through an old name.
        clear_resident_field_scopes();
        active_residency_scope_ = ComputeResidencyScope::Root;
        buffer_pool_.destroy_all();
        pooled_memory_allocations_.clear();

        if (timestamp_query_pool_ != VK_NULL_HANDLE && vkDestroyQueryPool_ != nullptr)
        {
            vkDestroyQueryPool_(device_context_.device, timestamp_query_pool_, nullptr);
            timestamp_query_pool_    = VK_NULL_HANDLE;
            gpu_timestamp_available_ = false;
        }
        if (fence_ != VK_NULL_HANDLE && vkDestroyFence_ != nullptr)
        {
            vkDestroyFence_(device_context_.device, fence_, nullptr);
            fence_ = VK_NULL_HANDLE;
        }
        if (cmd_pool_ != VK_NULL_HANDLE && vkDestroyCommandPool_ != nullptr)
        {
            vkDestroyCommandPool_(device_context_.device, cmd_pool_, nullptr);
            cmd_pool_ = VK_NULL_HANDLE;
            cmd_buf_  = VK_NULL_HANDLE;
        }

        for (ComputePipelineState& pipeline_state : pipeline_registry_)
        {
            destroy_pipeline_state(pipeline_state);
        }
        if (pipeline_cache_ != VK_NULL_HANDLE && vkDestroyPipelineCache_ != nullptr)
        {
            vkDestroyPipelineCache_(device_context_.device, pipeline_cache_, nullptr);
            pipeline_cache_ = VK_NULL_HANDLE;
        }
    }

    void abandon_compute_resources()
    {
        staging_q_          = {};
        staging_w_          = {};
        staging_dz_         = {};
        staging_dqdt_       = {};
        device_q_           = {};
        device_w_           = {};
        device_dz_          = {};
        device_dqdt_        = {};
        qv0_upload_scratch_ = {};
        u0_upload_scratch_  = {};
        v0_upload_scratch_  = {};

        clear_resident_field_scopes();
        active_residency_scope_ = ComputeResidencyScope::Root;
        buffer_pool_.abandon_all();
        pooled_memory_allocations_.clear();

        fence_                   = VK_NULL_HANDLE;
        cmd_pool_                = VK_NULL_HANDLE;
        cmd_buf_                 = VK_NULL_HANDLE;
        timestamp_query_pool_    = VK_NULL_HANDLE;
        gpu_timestamp_available_ = false;
        pending_dispatch_count_  = 0;
        pending_kernel_dispatch_counts_.fill(0);
        submitted_kernel_dispatch_counts_.fill(0);
        pipeline_cache_ = VK_NULL_HANDLE;
        pipeline_cache_path_.clear();
        pipeline_cache_identity_ = {};
        pipeline_cache_loaded_   = false;
        for (ComputePipelineState& pipeline_state : pipeline_registry_)
        {
            pipeline_state = {};
        }
    }

    // ── Member variables ───────────────────────────────────────────────

    tmv_vulkan::DeviceContext device_context_;

    PFN_vkGetInstanceProcAddr                  vkGetInstanceProcAddr_                  = nullptr;
    PFN_vkCreateInstance                       vkCreateInstance_                       = nullptr;
    PFN_vkEnumerateInstanceVersion             vkEnumerateInstanceVersion_             = nullptr;
    PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties_ = nullptr;
    PFN_vkEnumerateInstanceLayerProperties     vkEnumerateInstanceLayerProperties_     = nullptr;

    PFN_vkDestroyInstance                        vkDestroyInstance_                        = nullptr;
    PFN_vkEnumeratePhysicalDevices               vkEnumeratePhysicalDevices_               = nullptr;
    PFN_vkGetPhysicalDeviceProperties            vkGetPhysicalDeviceProperties_            = nullptr;
    PFN_vkGetPhysicalDeviceFeatures              vkGetPhysicalDeviceFeatures_              = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties_ = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties     vkEnumerateDeviceExtensionProperties_     = nullptr;
    PFN_vkCreateDevice                           vkCreateDevice_                           = nullptr;
    PFN_vkGetDeviceProcAddr                      vkGetDeviceProcAddr_                      = nullptr;

    PFN_vkDestroyDevice  vkDestroyDevice_  = nullptr;
    PFN_vkGetDeviceQueue vkGetDeviceQueue_ = nullptr;
    PFN_vkDeviceWaitIdle vkDeviceWaitIdle_ = nullptr;

    // Instance-level (for memory properties)
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties_ = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties2 vkGetPhysicalDeviceMemoryProperties2_ = nullptr;

    // Buffer and memory management
    PFN_vkCreateBuffer                 vkCreateBuffer_                 = nullptr;
    PFN_vkDestroyBuffer                vkDestroyBuffer_                = nullptr;
    PFN_vkAllocateMemory               vkAllocateMemory_               = nullptr;
    PFN_vkFreeMemory                   vkFreeMemory_                   = nullptr;
    PFN_vkBindBufferMemory             vkBindBufferMemory_             = nullptr;
    PFN_vkMapMemory                    vkMapMemory_                    = nullptr;
    PFN_vkUnmapMemory                  vkUnmapMemory_                  = nullptr;
    PFN_vkFlushMappedMemoryRanges      vkFlushMappedMemoryRanges_      = nullptr;
    PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRanges_ = nullptr;
    PFN_vkGetBufferMemoryRequirements  vkGetBufferMemoryRequirements_  = nullptr;

    // Shader and pipeline
    PFN_vkCreateShaderModule         vkCreateShaderModule_         = nullptr;
    PFN_vkDestroyShaderModule        vkDestroyShaderModule_        = nullptr;
    PFN_vkCreateDescriptorSetLayout  vkCreateDescriptorSetLayout_  = nullptr;
    PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout_ = nullptr;
    PFN_vkCreateDescriptorPool       vkCreateDescriptorPool_       = nullptr;
    PFN_vkDestroyDescriptorPool      vkDestroyDescriptorPool_      = nullptr;
    PFN_vkAllocateDescriptorSets     vkAllocateDescriptorSets_     = nullptr;
    PFN_vkUpdateDescriptorSets       vkUpdateDescriptorSets_       = nullptr;
    PFN_vkCreatePipelineLayout       vkCreatePipelineLayout_       = nullptr;
    PFN_vkDestroyPipelineLayout      vkDestroyPipelineLayout_      = nullptr;
    PFN_vkCreateComputePipelines     vkCreateComputePipelines_     = nullptr;
    PFN_vkDestroyPipeline            vkDestroyPipeline_            = nullptr;
    PFN_vkCreatePipelineCache        vkCreatePipelineCache_        = nullptr;
    PFN_vkDestroyPipelineCache       vkDestroyPipelineCache_       = nullptr;
    PFN_vkGetPipelineCacheData       vkGetPipelineCacheData_       = nullptr;

    // Command buffers
    PFN_vkCreateCommandPool      vkCreateCommandPool_      = nullptr;
    PFN_vkDestroyCommandPool     vkDestroyCommandPool_     = nullptr;
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers_ = nullptr;
    PFN_vkBeginCommandBuffer     vkBeginCommandBuffer_     = nullptr;
    PFN_vkEndCommandBuffer       vkEndCommandBuffer_       = nullptr;
    PFN_vkResetCommandBuffer     vkResetCommandBuffer_     = nullptr;
    PFN_vkCmdBindPipeline        vkCmdBindPipeline_        = nullptr;
    PFN_vkCmdBindDescriptorSets  vkCmdBindDescriptorSets_  = nullptr;
    PFN_vkCmdPushConstants       vkCmdPushConstants_       = nullptr;
    PFN_vkCmdDispatch            vkCmdDispatch_            = nullptr;
    PFN_vkCmdPipelineBarrier     vkCmdPipelineBarrier_     = nullptr;
    PFN_vkCmdCopyBuffer          vkCmdCopyBuffer_          = nullptr;

    // Synchronization
    PFN_vkQueueSubmit         vkQueueSubmit_         = nullptr;
    PFN_vkCreateFence         vkCreateFence_         = nullptr;
    PFN_vkDestroyFence        vkDestroyFence_        = nullptr;
    PFN_vkWaitForFences       vkWaitForFences_       = nullptr;
    PFN_vkResetFences         vkResetFences_         = nullptr;
    PFN_vkCreateQueryPool     vkCreateQueryPool_     = nullptr;
    PFN_vkDestroyQueryPool    vkDestroyQueryPool_    = nullptr;
    PFN_vkCmdResetQueryPool   vkCmdResetQueryPool_   = nullptr;
    PFN_vkCmdWriteTimestamp   vkCmdWriteTimestamp_   = nullptr;
    PFN_vkGetQueryPoolResults vkGetQueryPoolResults_ = nullptr;

    // Backend-specific pipeline cache indexed by the shared operation ID.
    // Orphan/deferred operations retain an empty slot and cannot accidentally
    // advertise readiness merely because an artifact exists on disk.
    tmv_vulkan::PipelineRegistry pipeline_registry_;

    // Shared command infrastructure
    VkCommandPool                                              cmd_pool_                = VK_NULL_HANDLE;
    VkCommandBuffer                                            cmd_buf_                 = VK_NULL_HANDLE;
    VkFence                                                    fence_                   = VK_NULL_HANDLE;
    VkQueryPool                                                timestamp_query_pool_    = VK_NULL_HANDLE;
    bool                                                       gpu_timestamp_available_ = false;
    std::uint64_t                                              pending_dispatch_count_  = 0;
    std::array<std::uint64_t, nimbus::compute::kernel_count()> pending_kernel_dispatch_counts_{};
    std::array<std::uint64_t, nimbus::compute::kernel_count()> submitted_kernel_dispatch_counts_{};
    VkPipelineCache                                            pipeline_cache_ = VK_NULL_HANDLE;
    std::filesystem::path                                      pipeline_cache_path_;
    tmv_vulkan::pipeline_cache_file::Identity                  pipeline_cache_identity_{};
    bool                                                       pipeline_cache_loaded_ = false;
    bool                                                       startup_memory_enabled_ = false;

    // GPU buffers for TVD vertical flux (4 bindings: q, w, dz, dqdt)
    GpuBuffer                                      staging_q_{};
    GpuBuffer                                      staging_w_{};
    GpuBuffer                                      staging_dz_{};
    GpuBuffer                                      staging_dqdt_{};
    GpuBuffer                                      device_q_{};
    GpuBuffer                                      device_w_{};
    GpuBuffer                                      device_dz_{};
    GpuBuffer                                      device_dqdt_{};
    std::vector<float>                             qv0_upload_scratch_;
    std::vector<float>                             u0_upload_scratch_;
    std::vector<float>                             v0_upload_scratch_;
    tmv::FlatCGridDeviceScalarAttributionWorkspace scalar_attribution_workspace_;

    // Pooled GPU buffer pairs for multi-field dispatch (radial, azimuthal,
    // diffusion, supercell, kessler, batched advection).
    tmv_vulkan::GpuBufferPool                            buffer_pool_;
    std::vector<std::unique_ptr<SharedBufferAllocation>> pooled_memory_allocations_;
    std::array<nimbus::compute::ResidentFieldStore, 2>   resident_field_scopes_{};
    ComputeResidencyScope                                active_residency_scope_ = ComputeResidencyScope::Root;
    std::map<std::string, GpuSubmissionStat>             gpu_submission_stats_;
    tmv::vulkan_compute::Telemetry                       compute_telemetry_;
    tmv::vulkan_compute::ActivityCounters                attributed_compute_activity_;
    GpuSubmissionStat*                                 completed_readback_stat_ = nullptr;
    tmv_vulkan::SubmissionLifecycle                      submission_lifecycle_;
#endif
};

} // namespace

std::unique_ptr<ComputeBackend> create_vulkan_compute_backend()
{
    return std::make_unique<VulkanComputeBackend>();
}
