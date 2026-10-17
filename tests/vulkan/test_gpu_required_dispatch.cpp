/**
 * @file test_gpu_required_dispatch.cpp
 * @brief Fail-closed Vulkan qualification smoke for required GPU CI lanes.
 *
 * Unlike the broad parity binaries, this test never treats device or pipeline
 * absence as an optional skip. A green result proves that the requested
 * backend stayed Vulkan, exposed a physical/software device, executed an odd
 * non-cubic compute dispatch, copied changed output back, and incremented the
 * completed-device-submission counter.
 */

#include "catch2/catch.hpp"

#include "compute/compute_backend.hpp"
#include "core/infra/hardware_info.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace
{

struct BackendShutdown
{
    ~BackendShutdown() { shutdown_compute_backend_runtime(); }
};

std::size_t index_of(int i, int j, int k, int ny, int nz)
{
    return (static_cast<std::size_t>(i) * static_cast<std::size_t>(ny)
            + static_cast<std::size_t>(j)) * static_cast<std::size_t>(nz)
         + static_cast<std::size_t>(k);
}

} // namespace

TEST_CASE("required Vulkan lane completes a counted device dispatch",
          "[vulkan][gpu_required]")
{
    BackendShutdown cleanup;
    global_compute_backend_config.backend = "vulkan";
    global_compute_backend_config.device_index = -1;
    global_compute_backend_config.allow_fallback = false;
    global_compute_backend_config.validate_parity = false;

    std::string error;
    const bool initialized = initialize_compute_backend_runtime(error);
    INFO("Vulkan initialization error: " << error);
    REQUIRE(initialized);
    REQUIRE(active_compute_backend_kind() == ComputeBackendKind::Vulkan);
    REQUIRE_FALSE(compute_backend_fallback_active());

    auto* backend = mutable_compute_backend();
    REQUIRE(backend != nullptr);
    REQUIRE(backend->name() == "vulkan");

    const ComputeMemoryProperties memory = backend->memory_properties();
    REQUIRE(memory.accelerator);
    if (const char* policy = std::getenv("NIMBUS_VULKAN_MEMORY_POLICY"))
    {
        if (std::string(policy) == "staged")
        {
            REQUIRE_FALSE(memory.unified_host_device_memory);
        }
        else if (std::string(policy) == "direct")
        {
            REQUIRE(memory.unified_host_device_memory);
        }
    }

    HardwareInfo hardware{};
    backend->populate_hardware_info(hardware);
    INFO("Vulkan device: " << hardware.gpu_device_name);
    REQUIRE(hardware.gpu_available);
    REQUIRE_FALSE(hardware.gpu_device_name.empty());

    auto* kernels = backend->diffusion_kernels();
    REQUIRE(kernels != nullptr);
    REQUIRE(kernels->supports(DiffusionKernelOp::Diffusion));

    constexpr int nx = 17;
    constexpr int ny = 13;
    constexpr int nz = 9;
    const std::size_t cells = static_cast<std::size_t>(nx) * ny * nz;
    std::vector<float> source(cells, 0.0f);
    std::vector<float> density(cells, 1.0f);
    std::vector<float> dz_profile(static_cast<std::size_t>(nz), 0.75f);
    std::vector<float> result(cells, -1234.0f);
    const int cx = nx / 2;
    const int cy = ny / 2;
    const int cz = nz / 2;
    source[index_of(cx, cy, cz, ny, nz)] = 100.0f;

    const std::uint64_t before = backend->successful_dispatch_count();
    const auto diffusion_kernel =
        nimbus::compute::KernelId::Diffusion;
    const auto unrelated_kernel =
        nimbus::compute::KernelId::KesslerPointwise;
    const std::uint64_t before_kernel =
        backend->successful_kernel_dispatch_count(diffusion_kernel);
    const std::uint64_t before_unrelated =
        backend->successful_kernel_dispatch_count(unrelated_kernel);
    const DiffusionDispatch dispatch{
        source.data(), density.data(), dz_profile.data(), result.data(),
        nx, ny, nz, 1.0f, 1.5f, 0.2f, 0.2f,
        DiffusionFieldLocation::Scalar, true, false, false,
    };
    REQUIRE(kernels->dispatch_diffusion(dispatch));
    const std::uint64_t after = backend->successful_dispatch_count();
    INFO("completed Vulkan dispatches: " << after);
    REQUIRE(after == before + 1);
    REQUIRE(backend->successful_kernel_dispatch_count(diffusion_kernel) ==
            before_kernel + 1);
    REQUIRE(backend->successful_kernel_dispatch_count(unrelated_kernel) ==
            before_unrelated);

    DiffusionDispatch rejected = dispatch;
    rejected.nx = 0;
    REQUIRE_FALSE(kernels->dispatch_diffusion(rejected));
    REQUIRE(backend->successful_dispatch_count() == after);
    REQUIRE(backend->successful_kernel_dispatch_count(diffusion_kernel) ==
            before_kernel + 1);
    REQUIRE(backend->successful_kernel_dispatch_count(
                nimbus::compute::KernelId::Count) == 0);

    const float center = result[index_of(cx, cy, cz, ny, nz)];
    REQUIRE(std::isfinite(center));
    REQUIRE(center < 0.0f);
    REQUIRE(result[index_of(cx + 1, cy, cz, ny, nz)] > 0.0f);
    REQUIRE(result[index_of(cx, cy, cz + 1, ny, nz)] > 0.0f);
}
