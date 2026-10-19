/**
 * @file test_compute_backend.cpp
 * @brief Vulkan compute implementation, policy, and fallback tests.
 */
#include "catch2/catch.hpp"
#include "compute/compute_backend.hpp"
#include "compute/compute_backend_factory.hpp"
#include "compute/kernel_startup_plan.hpp"
#include "compute/gpu_buffer_pool.hpp"
#include "compute/vulkan_descriptor_cache.hpp"
#include "compute/vulkan_pipeline_cache_file.hpp"
#include "compute/vulkan_submission_lifecycle.hpp"
#include "compute/vulkan_dispatch_policy.hpp"
#include "compute/vulkan_compute_telemetry.hpp"
#include "compute/vulkan_memory_plan.hpp"
#include "compute/vulkan_residency_policy.hpp"

#include <array>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

namespace
{

class ScopedEnvironmentValue
{
public:
    ScopedEnvironmentValue(const char* name, const char* value)
        : name_(name)
    {
        if (const char* previous = std::getenv(name))
        {
            had_previous_ = true;
            previous_ = previous;
        }
        setenv(name_.c_str(), value, 1);
    }

    ~ScopedEnvironmentValue()
    {
        if (had_previous_)
        {
            setenv(name_.c_str(), previous_.c_str(), 1);
        }
        else
        {
            unsetenv(name_.c_str());
        }
    }

private:
    std::string name_;
    std::string previous_;
    bool had_previous_ = false;
};

#if defined(TMV_GPU_BUFFER_POOL_HAS_VULKAN)
struct FakePoolAllocator
{
    int creates = 0;
    int destroys = 0;
    int batch_calls = 0;
    std::size_t batch_items = 0;
};

bool fake_pool_create(void* user_data, VkDeviceSize size,
                      bool, tmv_vulkan::GpuBuffer& out)
{
    auto& stats = *static_cast<FakePoolAllocator*>(user_data);
    ++stats.creates;
    out.size = size;
    return true;
}

void fake_pool_destroy(void* user_data, tmv_vulkan::GpuBuffer& out)
{
    auto& stats = *static_cast<FakePoolAllocator*>(user_data);
    ++stats.destroys;
    out = {};
}

bool fake_pool_create_batch(
    void* user_data, const VkDeviceSize* sizes, std::size_t count,
    tmv_vulkan::GpuBuffer* staging, tmv_vulkan::GpuBuffer* device)
{
    auto& stats = *static_cast<FakePoolAllocator*>(user_data);
    ++stats.batch_calls;
    stats.batch_items += count;
    for (std::size_t i = 0; i < count; ++i)
    {
        staging[i].size = sizes[i];
        device[i].size = sizes[i];
    }
    return true;
}
#endif

} // namespace

TEST_CASE("Vulkan pipeline cache identity keys device driver UUID and manifest",
          "[vulkan][backend][pipeline-cache]")
{
    using tmv_vulkan::pipeline_cache_file::Identity;
    Identity identity;
    identity.vendor_id = 0x106b;
    identity.device_id = 0x0001;
    identity.driver_version = 0x00020003;
    for (std::size_t i = 0; i < identity.uuid.size(); ++i)
    {
        identity.uuid[i] = static_cast<std::uint8_t>(i);
    }
    identity.manifest_signature = "0123456789abcdef";

    const std::string baseline =
        tmv_vulkan::pipeline_cache_file::filename(identity);
    REQUIRE(baseline ==
            "compute-v1-0000106b-00000001-00020003-"
            "000102030405060708090a0b0c0d0e0f-0123456789abcdef.bin");

    Identity changed_driver = identity;
    ++changed_driver.driver_version;
    REQUIRE(tmv_vulkan::pipeline_cache_file::filename(changed_driver) !=
            baseline);

    Identity changed_manifest = identity;
    changed_manifest.manifest_signature = "fedcba9876543210";
    REQUIRE(tmv_vulkan::pipeline_cache_file::filename(changed_manifest) !=
            baseline);
}

TEST_CASE("Vulkan pipeline cache accepts only a bounded matching standard header",
          "[vulkan][backend][pipeline-cache]")
{
    using namespace tmv_vulkan::pipeline_cache_file;
    Identity identity;
    identity.vendor_id = 0x106b;
    identity.device_id = 0x1234;
    for (std::size_t i = 0; i < identity.uuid.size(); ++i)
    {
        identity.uuid[i] = static_cast<std::uint8_t>(0xa0 + i);
    }

    std::vector<std::uint8_t> bytes(64, 0);
    const auto write_u32 = [&bytes](std::size_t offset, std::uint32_t value) {
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    };
    write_u32(0, static_cast<std::uint32_t>(kHeaderBytes));
    write_u32(4, kHeaderVersionOne);
    write_u32(8, identity.vendor_id);
    write_u32(12, identity.device_id);
    std::memcpy(bytes.data() + 16, identity.uuid.data(), identity.uuid.size());
    REQUIRE(valid_vulkan_header(bytes, identity));

    write_u32(8, identity.vendor_id + 1);
    REQUIRE_FALSE(valid_vulkan_header(bytes, identity));
    write_u32(8, identity.vendor_id);

    bytes[16] ^= 0x1;
    REQUIRE_FALSE(valid_vulkan_header(bytes, identity));
    bytes[16] ^= 0x1;

    write_u32(4, kHeaderVersionOne + 1);
    REQUIRE_FALSE(valid_vulkan_header(bytes, identity));
    write_u32(4, kHeaderVersionOne);

    write_u32(0, static_cast<std::uint32_t>(bytes.size() + 1));
    REQUIRE_FALSE(valid_vulkan_header(bytes, identity));
    bytes.resize(kHeaderBytes - 1);
    REQUIRE_FALSE(valid_vulkan_header(bytes, identity));
}

TEST_CASE("Vulkan pipeline cache supports cold warm and corrupt-cache startup",
          "[vulkan][backend][pipeline-cache]")
{
    const auto nonce =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path cache_directory =
        std::filesystem::temp_directory_path() /
        ("nimbus-vulkan-pipeline-cache-" + std::to_string(nonce));
    const std::string cache_directory_text = cache_directory.string();
    ScopedEnvironmentValue cache_dir(
        "NIMBUS_VULKAN_PIPELINE_CACHE_DIR", cache_directory_text.c_str());
    ScopedEnvironmentValue cache_enabled(
        "NIMBUS_VULKAN_PIPELINE_CACHE", "1");

    const auto initialize_usable_backend = [](
            std::unique_ptr<ComputeBackend>& backend) {
        using namespace nimbus::compute;
        backend = create_vulkan_compute_backend();
        std::string error;
        ComputeBackendConfig config;
        config.backend = "vulkan";
        config.allow_fallback = false;
        // Keep cold-cache coverage bounded while exercising a compensated shader variant.
        config.startup_plan = KernelStartupPlan::make(
            "pipeline-cache-contract", {KernelStartupOwner::CartesianExplicitDiffusion},
            {KernelId::Diffusion, KernelId::RkCombination}, kernel_manifest(), error);
        REQUIRE(config.startup_plan);
        backend->configure(config);
        if (!backend->initialize(error))
        {
            return false;
        }
        return backend->supports_rk_combination();
    };

    std::unique_ptr<ComputeBackend> cold;
    if (!initialize_usable_backend(cold))
    {
        if (cold) cold->shutdown();
        SUCCEED("No usable Vulkan device/pipeline on this test host");
        return;
    }
    cold->shutdown();

    std::vector<std::filesystem::path> cache_files;
    for (const auto& entry :
         std::filesystem::directory_iterator(cache_directory))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".bin")
        {
            cache_files.push_back(entry.path());
        }
    }
    REQUIRE(cache_files.size() == 1);
    REQUIRE(std::filesystem::file_size(cache_files.front()) >=
            tmv_vulkan::pipeline_cache_file::kHeaderBytes);

    std::unique_ptr<ComputeBackend> warm;
    REQUIRE(initialize_usable_backend(warm));
    warm->shutdown();

    // A truncated cache must be ignored, rebuilt, and atomically replaced. It
    // must never make the valid GPU science path unavailable.
    {
        std::ofstream corrupt(cache_files.front(),
                              std::ios::binary | std::ios::trunc);
        const std::array<std::uint8_t, 8> invalid{{0, 1, 2, 3, 4, 5, 6, 7}};
        corrupt.write(reinterpret_cast<const char*>(invalid.data()),
                      static_cast<std::streamsize>(invalid.size()));
        REQUIRE(corrupt.good());
    }
    REQUIRE(std::filesystem::file_size(cache_files.front()) == 8);

    std::unique_ptr<ComputeBackend> repaired;
    REQUIRE(initialize_usable_backend(repaired));
    repaired->shutdown();
    REQUIRE(std::filesystem::file_size(cache_files.front()) >=
            tmv_vulkan::pipeline_cache_file::kHeaderBytes);
}

TEST_CASE("Stable descriptor bindings skip redundant updates",
          "[vulkan][backend][descriptor_cache]")
{
    tmv::vulkan_compute::DescriptorBindingCache cache;
    const std::uintptr_t buffers_a[3] = {11, 22, 33};
    const std::size_t ranges_a[3] = {1024, 1024, 1024};

    REQUIRE_FALSE(cache.matches_or_update(buffers_a, ranges_a, 3));
    REQUIRE(cache.matches_or_update(buffers_a, ranges_a, 3));

    const std::uintptr_t buffers_b[3] = {11, 22, 44};
    REQUIRE_FALSE(cache.matches_or_update(buffers_b, ranges_a, 3));
    REQUIRE(cache.matches_or_update(buffers_b, ranges_a, 3));

    const std::size_t ranges_b[3] = {1024, 512, 1024};
    REQUIRE_FALSE(cache.matches_or_update(buffers_b, ranges_b, 3));
    cache.clear();
    REQUIRE_FALSE(cache.matches_or_update(buffers_b, ranges_b, 3));
}

TEST_CASE("Compute memory admission follows the initialized fallback backend",
          "[vulkan][backend][runtime][memory-preflight]")
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend = "vulkan";
    runtime.config.device_index = std::numeric_limits<int>::max();
    runtime.config.allow_fallback = true;

    std::string error;
    REQUIRE(initialize_compute_backend_runtime(
        runtime, CoordinateSystem::Cartesian, error));
    REQUIRE(compute_backend_fallback_active(runtime));
    REQUIRE(active_compute_backend_kind(runtime) == ComputeBackendKind::Cpu);
    const ComputeBackend* backend = active_compute_backend(runtime);
    REQUIRE(backend != nullptr);
    const ComputeMemoryProperties memory = backend->memory_properties();
    REQUIRE_FALSE(memory.accelerator);
    REQUIRE(memory.device_local_heap_bytes == 0);
    shutdown_compute_backend_runtime(runtime);
}

TEST_CASE("Vulkan submission timeout pins resources and poisons dispatch",
          "[vulkan][backend][lifecycle]")
{
    tmv_vulkan::SubmissionLifecycle lifecycle;
    const auto generation = lifecycle.generation();

    REQUIRE(lifecycle.pipeline_is_ready(generation, true));
    lifecycle.note_submit_succeeded();
    REQUIRE_FALSE(lifecycle.can_dispatch());
    REQUIRE_FALSE(lifecycle.may_release_transients());

    lifecycle.poison_after_wait();
    REQUIRE(lifecycle.poisoned());
    REQUIRE(lifecycle.submission_may_be_in_flight());
    REQUIRE_FALSE(lifecycle.pipeline_is_ready(generation, true));
    REQUIRE_FALSE(lifecycle.may_release_transients());
    REQUIRE_FALSE(lifecycle.may_destroy_child_resources());

    lifecycle.note_device_quiesced();
    REQUIRE(lifecycle.may_destroy_child_resources());
    REQUIRE_FALSE(lifecycle.can_dispatch());
}

TEST_CASE("Vulkan pre-submit failure permits release but disables the backend",
          "[vulkan][backend][lifecycle]")
{
    tmv_vulkan::SubmissionLifecycle lifecycle;
    const auto generation = lifecycle.generation();

    lifecycle.poison_before_submit();
    REQUIRE(lifecycle.poisoned());
    REQUIRE_FALSE(lifecycle.submission_may_be_in_flight());
    REQUIRE(lifecycle.may_release_transients());
    REQUIRE_FALSE(lifecycle.pipeline_is_ready(generation, true));
}

#if defined(TMV_GPU_BUFFER_POOL_HAS_VULKAN)
TEST_CASE("Vulkan pool abandon forgets device-owned children without callbacks",
          "[vulkan][backend][lifecycle]")
{
    FakePoolAllocator stats;
    tmv_vulkan::GpuBufferPool pool;
    pool.set_allocator(&fake_pool_create, &fake_pool_destroy, &stats);

    REQUIRE(pool.acquire(1024) == 0);
    REQUIRE(stats.creates == 2);
    REQUIRE(pool.slot_count() == 1);

    pool.abandon_all();
    REQUIRE(pool.slot_count() == 0);
    REQUIRE(stats.destroys == 0);
    REQUIRE(pool.acquire(1024) == -1);
}

TEST_CASE("Vulkan pool batches only storage that is not already reusable",
          "[vulkan][backend][allocation]")
{
    FakePoolAllocator stats;
    tmv_vulkan::GpuBufferPool pool;
    pool.set_allocator(&fake_pool_create, &fake_pool_destroy, &stats);
    pool.set_batch_allocator(&fake_pool_create_batch);

    int first[6]{};
    REQUIRE(pool.acquire_many(256, 6, first));
    REQUIRE(stats.batch_calls == 1);
    REQUIRE(stats.batch_items == 6);
    REQUIRE(pool.active_count() == 6);

    for (int slot : first) pool.release(slot);

    int reused[4]{};
    REQUIRE(pool.acquire_many(128, 4, reused));
    REQUIRE(stats.batch_calls == 1);
    REQUIRE(stats.batch_items == 6);

    // Four sufficient slots stay live. The two remaining free slots are
    // replaced together, so a larger request still issues one exact batch.
    int larger[2]{};
    REQUIRE(pool.acquire_many(1024, 2, larger));
    REQUIRE(stats.batch_calls == 2);
    REQUIRE(stats.batch_items == 8);
    REQUIRE(stats.destroys == 4);
    REQUIRE(pool.active_count() == 6);

    pool.destroy_all();
    REQUIRE(stats.destroys == 16);
}

TEST_CASE("Vulkan pool preserves mixed-size request order and best-fit reuse",
          "[vulkan][backend][allocation]")
{
    FakePoolAllocator stats;
    tmv_vulkan::GpuBufferPool pool;
    pool.set_allocator(&fake_pool_create, &fake_pool_destroy, &stats);
    pool.set_batch_allocator(&fake_pool_create_batch);

    const VkDeviceSize initial_sizes[3] = {64, 1024, 128};
    int initial[3]{};
    REQUIRE(pool.acquire_many(initial_sizes, 3, initial));
    REQUIRE(pool.device(initial[0]).size == 64);
    REQUIRE(pool.device(initial[1]).size == 1024);
    REQUIRE(pool.device(initial[2]).size == 128);
    for (int slot : initial) pool.release(slot);

    const VkDeviceSize reuse_sizes[2] = {1000, 100};
    int reused[2]{};
    REQUIRE(pool.acquire_many(reuse_sizes, 2, reused));
    REQUIRE(reused[0] == initial[1]);
    REQUIRE(reused[1] == initial[2]);
    REQUIRE(stats.batch_calls == 1);
    REQUIRE(stats.batch_items == 3);
    pool.destroy_all();
    REQUIRE(stats.destroys == 6);
}
#endif

TEST_CASE("Vulkan uncertain submit failure pins resources and disables reuse",
          "[vulkan][backend][lifecycle]")
{
    tmv_vulkan::SubmissionLifecycle lifecycle;
    const auto generation = lifecycle.generation();

    lifecycle.poison_after_submit_attempt();
    REQUIRE(lifecycle.poisoned());
    REQUIRE(lifecycle.submission_may_be_in_flight());
    REQUIRE_FALSE(lifecycle.may_release_transients());
    REQUIRE_FALSE(lifecycle.pipeline_is_ready(generation, true));
}

TEST_CASE("Vulkan failed reinitialization cannot expose stale pipeline readiness",
          "[vulkan][backend][lifecycle]")
{
    tmv_vulkan::SubmissionLifecycle lifecycle;
    const auto first_device_generation = lifecycle.generation();
    REQUIRE(lifecycle.pipeline_is_ready(first_device_generation, true));

    // initialize() starts by shutting down the previous device.  Even if the
    // replacement then fails before constructing a pipeline, the old token is
    // from a different generation and cannot advertise readiness.
    lifecycle.reset_for_new_device();
    REQUIRE_FALSE(lifecycle.pipeline_is_ready(first_device_generation, true));
    REQUIRE_FALSE(lifecycle.pipeline_is_ready(lifecycle.generation(), false));

    const auto failed_device_generation = lifecycle.generation();
    lifecycle.reset_for_new_device();
    REQUIRE_FALSE(lifecycle.pipeline_is_ready(failed_device_generation, true));
}

TEST_CASE("Vulkan backend repeat initialization clears every stale capability",
          "[vulkan][backend][lifecycle]")
{
    auto backend = create_vulkan_compute_backend();
    REQUIRE(backend != nullptr);
    auto* advection = backend->advection_kernels();
    auto* dynamics = backend->dynamics_kernels();
    auto* microphysics = backend->microphysics_kernels();
    auto* turbulence = backend->turbulence_kernels();
    auto* diffusion = backend->diffusion_kernels();
    REQUIRE(advection != nullptr);
    REQUIRE(dynamics != nullptr);
    REQUIRE(microphysics != nullptr);
    REQUIRE(turbulence != nullptr);
    REQUIRE(diffusion != nullptr);

    std::string first_error;
    backend->initialize(first_error);

    // Force pipeline construction to fail on the replacement device.  This
    // remains useful on a host without a Vulkan device: both failed attempts
    // must leave every capability false.  On a Vulkan host it additionally
    // catches handles retained from the successful first generation.
    {
        ScopedEnvironmentValue shader_dir(
            "TMV_SHADER_DIR", "/nimbus-test/nonexistent-vulkan-shaders");
        std::string second_error;
        backend->initialize(second_error);
    }

    for (const auto op : {
             AdvectionKernelOp::VerticalFlux,
             AdvectionKernelOp::ConservativeCartesianCgridTransport,
             AdvectionKernelOp::AdvectionX,
             AdvectionKernelOp::AdvectionY})
    {
        REQUIRE_FALSE(advection->supports(op));
    }
    for (const auto op : {
             DynamicsKernelOp::AcousticPressureCartesianCgrid,
             DynamicsKernelOp::AcousticMomentumCartesianCgrid,
             DynamicsKernelOp::AcousticSubstepFusedCartesianCgrid,
             DynamicsKernelOp::AcousticSubstepsBatchedCartesianCgrid,
             DynamicsKernelOp::StateRelaxationCartesian,
             DynamicsKernelOp::SlowApply,
             DynamicsKernelOp::CartesianCgridSlowTendencies,
             DynamicsKernelOp::CartesianTendencies,
             DynamicsKernelOp::AcousticPressure,
             DynamicsKernelOp::AcousticMomentum,
             DynamicsKernelOp::AcousticSubstepFused,
             DynamicsKernelOp::AcousticSubstepsBatched})
    {
        REQUIRE_FALSE(dynamics->supports(op));
    }
    for (const auto op : {
             MicrophysicsKernelOp::LinFused,
             MicrophysicsKernelOp::LinPointwise,
             MicrophysicsKernelOp::LinSedimentation,
             MicrophysicsKernelOp::MicrophysicsApply,
             MicrophysicsKernelOp::KesslerPointwise,
             MicrophysicsKernelOp::KesslerSedimentation})
    {
        REQUIRE_FALSE(microphysics->supports(op));
    }
    for (const auto op : {
             TurbulenceKernelOp::Smagorinsky,
             TurbulenceKernelOp::SgsApply})
    {
        REQUIRE_FALSE(turbulence->supports(op));
    }
    REQUIRE_FALSE(diffusion->supports(DiffusionKernelOp::Diffusion));
    REQUIRE_FALSE(backend->supports_homogeneous_domain_field_shift());

    backend->shutdown();
    REQUIRE_FALSE(advection->supports(AdvectionKernelOp::VerticalFlux));
    REQUIRE_FALSE(backend->supports_homogeneous_domain_field_shift());
    REQUIRE_NOTHROW(backend->shutdown());
    REQUIRE_FALSE(advection->supports(AdvectionKernelOp::VerticalFlux));
    REQUIRE_FALSE(backend->supports_homogeneous_domain_field_shift());
}

TEST_CASE("Vulkan explicit startup scope rolls back required pipelines and recovers",
          "[vulkan][backend][startup-plan][device]")
{
    using namespace nimbus::compute;
    const std::vector<KernelStartupOwner> owners{
        KernelStartupOwner::FlatCartesianCgridDynamics,
        KernelStartupOwner::CartesianScalarTransport,
        KernelStartupOwner::CartesianExplicitDiffusion};
    std::vector<KernelId> kernels;
    for (const auto owner : owners)
    {
        const auto dependencies = startup_owner_kernels(owner);
        kernels.insert(kernels.end(), dependencies.begin(), dependencies.end());
    }
    std::string error;
    ComputeBackendConfig configuration;
    configuration.backend = "vulkan";
    configuration.allow_fallback = false;
    configuration.startup_plan = KernelStartupPlan::make(
        "flat-dry-cgrid18-v1", owners, kernels, kernel_manifest(), error);
    REQUIRE(configuration.startup_plan);
    auto backend = create_vulkan_compute_backend();
    REQUIRE(backend);
    backend->configure(configuration);
    ScopedEnvironmentValue persistence("NIMBUS_VULKAN_PIPELINE_CACHE", "0");
    if (!backend->initialize(error))
    {
        WARN("No usable Vulkan device for explicit startup lifecycle: " << error);
        return;
    }
    auto* advection = backend->advection_kernels();
    auto* dynamics = backend->dynamics_kernels();
    auto* microphysics = backend->microphysics_kernels();
    auto* turbulence = backend->turbulence_kernels();
    auto* diffusion = backend->diffusion_kernels();
    REQUIRE(advection);
    REQUIRE(dynamics);
    REQUIRE(microphysics);
    REQUIRE(turbulence);
    REQUIRE(diffusion);
    const auto check_capabilities = [&](bool ready)
    {
        for (int value = static_cast<int>(AdvectionKernelOp::VerticalFlux);
             value <= static_cast<int>(AdvectionKernelOp::BatchedAdvection); ++value)
        {
            INFO("advection operation " << value);
            const auto operation = static_cast<AdvectionKernelOp>(value);
            REQUIRE(advection->supports(operation) ==
                    (ready && operation != AdvectionKernelOp::BatchedAdvection));
        }
        for (int value = static_cast<int>(DynamicsKernelOp::AcousticContinuityMassFluxCartesianAgrid);
             value <= static_cast<int>(DynamicsKernelOp::ConservativeReduction); ++value)
        {
            const auto operation = static_cast<DynamicsKernelOp>(value);
            const bool selected = (operation >= DynamicsKernelOp::AcousticPressureCartesianCgrid &&
                                   operation <= DynamicsKernelOp::CartesianCgridSlowTendencies) ||
                                  operation == DynamicsKernelOp::AcousticBoundaryCartesianCgrid;
            REQUIRE(dynamics->supports(operation) == (ready && selected));
        }
        for (int value = static_cast<int>(MicrophysicsKernelOp::LinFused);
             value <= static_cast<int>(MicrophysicsKernelOp::MilbrandtSedimentation); ++value)
        {
            REQUIRE_FALSE(microphysics->supports(static_cast<MicrophysicsKernelOp>(value)));
        }
        for (int value = static_cast<int>(TurbulenceKernelOp::Smagorinsky);
             value <= static_cast<int>(TurbulenceKernelOp::CollocatedSgs); ++value)
        {
            REQUIRE_FALSE(turbulence->supports(static_cast<TurbulenceKernelOp>(value)));
        }
        REQUIRE(diffusion->supports(DiffusionKernelOp::Diffusion) == ready);
        REQUIRE_FALSE(backend->supports_homogeneous_domain_field_shift());
    };
    check_capabilities(true);
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto shader_root = std::filesystem::temp_directory_path() /
        ("nimbus-vulkan-explicit-startup-" + std::to_string(nonce));
    REQUIRE_FALSE(std::filesystem::exists(shader_root));
    REQUIRE(std::filesystem::create_directory(shader_root));
    for (const auto id : kernels)
    {
        if (id == KernelId::Diffusion)
        {
            continue;
        }
        const auto& entry = kernel_manifest_entry(id);
        const auto target = shader_root / entry.artifact_name;
        std::filesystem::create_directories(target.parent_path());
        REQUIRE(std::filesystem::copy_file(
            std::filesystem::path("vulkan/shaders/compute") / entry.artifact_name, target));
    }
    const std::string shader_root_text = shader_root.string();
    {
        ScopedEnvironmentValue selected_shader_root("TMV_SHADER_DIR", shader_root_text.c_str());
        REQUIRE_FALSE(backend->initialize(error));
        REQUIRE(error.find("diffusion") != std::string::npos);
        REQUIRE(error.find("failed to load SPIR-V") != std::string::npos);
    }
    check_capabilities(false);
    REQUIRE_NOTHROW(backend->shutdown());
    REQUIRE_NOTHROW(backend->shutdown());
    check_capabilities(false);
    configuration.device_index = std::numeric_limits<int>::max();
    backend->configure(configuration);
    REQUIRE_FALSE(backend->initialize(error));
    check_capabilities(false);
    configuration.device_index = -1;
    backend->configure(configuration);
    REQUIRE(backend->initialize(error));
    check_capabilities(true);
    backend->shutdown();
    check_capabilities(false);
}

TEST_CASE("Vulkan dispatch sizing rejects invalid extents and overflow",
          "[vulkan][backend][dispatch]")
{
    std::size_t elements = 0;
    VkDeviceSize bytes = 0;
    std::string error;
    REQUIRE(tmv::vulkan_dispatch::checked_extent(256, 256, 80, sizeof(float),
                                                  elements, bytes, error));
    REQUIRE(elements == 5242880);
    REQUIRE(bytes == 20971520);

    error.clear();
    REQUIRE_FALSE(tmv::vulkan_dispatch::checked_extent(0, 10, 10, sizeof(float),
                                                        elements, bytes, error));
    REQUIRE_FALSE(error.empty());
    error.clear();
    REQUIRE_FALSE(tmv::vulkan_dispatch::checked_extent(
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(),
        sizeof(float), elements, bytes, error));
    REQUIRE_FALSE(error.empty());
}

TEST_CASE("Vulkan linear planner maps oversized X dispatches into Y",
          "[vulkan][backend][dispatch]")
{
    VkPhysicalDeviceLimits limits{};
    limits.maxComputeWorkGroupInvocations = 256;
    limits.maxComputeWorkGroupSize[0] = 256;
    limits.maxComputeWorkGroupCount[0] = 65535;
    limits.maxComputeWorkGroupCount[1] = 65535;
    tmv::vulkan_dispatch::LinearPlan plan{};
    std::string error;

    REQUIRE(tmv::vulkan_dispatch::plan_linear(5242880, 64, limits, plan, error));
    REQUIRE(plan.groups_x == 65535);
    REQUIRE(plan.groups_y == 2);
    REQUIRE(plan.groups_z == 1);

    limits.maxComputeWorkGroupSize[0] = 32;
    error.clear();
    REQUIRE_FALSE(tmv::vulkan_dispatch::plan_linear(1024, 64, limits, plan, error));
    REQUIRE_FALSE(error.empty());

    limits.maxComputeWorkGroupSize[0] = 256;
    limits.maxComputeWorkGroupCount[0] = 1;
    limits.maxComputeWorkGroupCount[1] = 1;
    error.clear();
    REQUIRE_FALSE(tmv::vulkan_dispatch::plan_linear(129, 64, limits, plan, error));
    REQUIRE_FALSE(error.empty());

    limits.maxComputeWorkGroupCount[0] = std::numeric_limits<uint32_t>::max();
    limits.maxComputeWorkGroupCount[1] = std::numeric_limits<uint32_t>::max();
    error.clear();
    REQUIRE(tmv::vulkan_dispatch::plan_linear(
        std::numeric_limits<std::uint64_t>::max(), 64, limits, plan, error));
    REQUIRE(plan.groups_x == std::numeric_limits<uint32_t>::max());
    REQUIRE(plan.groups_y == 67108865u);
}

TEST_CASE("Vulkan compute telemetry tracks submissions, traffic, and allocation high-water",
          "[vulkan][backend][telemetry]")
{
    tmv::vulkan_compute::Telemetry telemetry;
    telemetry.note_submission(3);
    telemetry.note_submission(2);
    telemetry.note_host_upload(100);
    telemetry.note_host_download(40);
    telemetry.note_device_upload(80);
    telemetry.note_device_download(20);
    telemetry.note_resident_lookup(true, 512);
    telemetry.note_resident_lookup(false);
    telemetry.note_resident_lookup(false);
    telemetry.note_writeonly_output_upload_avoided(768);
    telemetry.note_resident_full_invalidation(3);
    telemetry.note_resident_full_invalidation(0);

    telemetry.note_allocation(1024, false);
    telemetry.note_allocation(2048, true);
    telemetry.note_allocation(4096, true);
    telemetry.note_memory_allocation(1024, false);
    telemetry.note_memory_allocation(6144, true);
    telemetry.note_release(2048, true);
    telemetry.note_allocation(1024, true);

    REQUIRE(telemetry.submitted_command_buffers == 2);
    REQUIRE(telemetry.submitted_dispatches == 5);
    REQUIRE(telemetry.host_upload_bytes == 100);
    REQUIRE(telemetry.host_download_bytes == 40);
    REQUIRE(telemetry.device_upload_bytes == 80);
    REQUIRE(telemetry.device_download_bytes == 20);
    REQUIRE(telemetry.resident_hits == 1);
    REQUIRE(telemetry.resident_misses == 2);
    REQUIRE(telemetry.resident_upload_bytes_saved == 512);
    REQUIRE(telemetry.writeonly_output_upload_bytes_saved == 768);
    REQUIRE(telemetry.resident_full_invalidations == 2);
    REQUIRE(telemetry.resident_entries_invalidated == 3);
    REQUIRE(telemetry.staging_allocated_bytes == 1024);
    REQUIRE(telemetry.staging_high_water_bytes == 1024);
    REQUIRE(telemetry.device_allocated_bytes == 5120);
    REQUIRE(telemetry.device_high_water_bytes == 6144);
    REQUIRE(telemetry.staging_memory_allocations == 1);
    REQUIRE(telemetry.device_memory_allocations == 1);
    REQUIRE(telemetry.device_memory_reserved_bytes == 6144);
    REQUIRE(telemetry.device_memory_reserved_high_water_bytes == 6144);
    telemetry.note_memory_release(6144, true);
    REQUIRE(telemetry.device_memory_allocations == 0);
    REQUIRE(telemetry.device_memory_reserved_bytes == 0);
}

TEST_CASE("Vulkan activity snapshots attribute exact work between submissions",
          "[vulkan][backend][telemetry]")
{
    tmv::vulkan_compute::Telemetry telemetry;
    telemetry.note_model_step();
    telemetry.note_submission(3);
    telemetry.note_host_upload(100);
    telemetry.note_descriptor_lookup(true);
    const auto before = telemetry.activity();

    telemetry.note_model_step();
    telemetry.note_submission(4);
    telemetry.note_host_upload(17);
    telemetry.note_host_download(19);
    telemetry.note_device_upload(23);
    telemetry.note_device_download(29);
    telemetry.note_resident_lookup(true, 31);
    telemetry.note_resident_lookup(false);
    telemetry.note_writeonly_output_upload_avoided(37);
    telemetry.note_descriptor_lookup(true);
    telemetry.note_descriptor_lookup(false);

    const auto delta = tmv::vulkan_compute::activity_delta(
        telemetry.activity(), before);
    REQUIRE(delta.model_steps == 1);
    REQUIRE(delta.submitted_command_buffers == 1);
    REQUIRE(delta.submitted_dispatches == 4);
    REQUIRE(delta.host_upload_bytes == 17);
    REQUIRE(delta.host_download_bytes == 19);
    REQUIRE(delta.device_upload_bytes == 23);
    REQUIRE(delta.device_download_bytes == 29);
    REQUIRE(delta.resident_hits == 1);
    REQUIRE(delta.resident_misses == 1);
    REQUIRE(delta.resident_upload_bytes_saved == 31);
    REQUIRE(delta.writeonly_output_upload_bytes_saved == 37);
    REQUIRE(delta.descriptor_cache_hits == 1);
    REQUIRE(delta.descriptor_cache_misses == 1);
}

TEST_CASE("Vulkan named residency requires exact current-generation ownership",
          "[vulkan][backend][residency]")
{
    float host_a[8]{};
    float host_b[8]{};
    const tmv::vulkan_residency::ResidentBinding committed{
        host_a, sizeof(host_a), 7, true};

    REQUIRE(tmv::vulkan_residency::reusable(
        committed, 7, host_a, sizeof(host_a), true));
    REQUIRE_FALSE(tmv::vulkan_residency::reusable(
        committed, 0, host_a, sizeof(host_a), true));
    REQUIRE_FALSE(tmv::vulkan_residency::reusable(
        committed, 6, host_a, sizeof(host_a), true));
    REQUIRE_FALSE(tmv::vulkan_residency::reusable(
        committed, 7, host_b, sizeof(host_b), true));
    REQUIRE_FALSE(tmv::vulkan_residency::reusable(
        committed, 7, host_a, sizeof(host_a) - sizeof(float), true));
    REQUIRE_FALSE(tmv::vulkan_residency::reusable(
        committed, 7, host_a, sizeof(host_a), false));

    auto invalid = committed;
    invalid.device_valid = false;
    REQUIRE_FALSE(tmv::vulkan_residency::reusable(
        invalid, 7, host_a, sizeof(host_a), true));
}

TEST_CASE("Vulkan write-only outputs require a full-field dispatch",
          "[vulkan][backend][residency]")
{
    REQUIRE(tmv::vulkan_residency::covers_full_field(64, 64));
    REQUIRE_FALSE(tmv::vulkan_residency::covers_full_field(63, 64));
    REQUIRE_FALSE(tmv::vulkan_residency::covers_full_field(65, 64));
    REQUIRE_FALSE(tmv::vulkan_residency::covers_full_field(0, 0));
}
