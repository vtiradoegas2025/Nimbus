/**
 * @file test_backend_contracts.cpp
 * @brief Backend-neutral kernel metadata, dispatch, and capability contracts.
 */

#include "catch2/catch.hpp"
#include "compute/compute_backend.hpp"
#include "compute/dispatch_graph.hpp"
#include "compute/kernel_manifest.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <string>

namespace
{

class DummyComputeBackend final : public ComputeBackend
{
public:
    std::string name() const override { return "dummy"; }
    bool initialize(std::string&) override { return true; }
    void shutdown() override {}
};

class AdvectionOnlyComputeBackend final :
    public ComputeBackend,
    public AdvectionComputeKernels
{
public:
    std::string name() const override { return "advection-only"; }
    bool initialize(std::string&) override { return true; }
    void shutdown() override {}

    AdvectionComputeKernels* advection_kernels() override { return this; }
    const AdvectionComputeKernels* advection_kernels() const override
    {
        return this;
    }

    bool supports(AdvectionKernelOp op) const override
    {
        return op == AdvectionKernelOp::VerticalFlux;
    }
};

class StagedDiffusionKernels final : public DiffusionComputeKernels
{
public:
    int fail_on_call = 0;
    int calls = 0;

    bool dispatch_diffusion(const DiffusionDispatch& dispatch) override
    {
        ++calls;
        const std::size_t cells = static_cast<std::size_t>(dispatch.nx) *
            static_cast<std::size_t>(dispatch.ny) *
            static_cast<std::size_t>(dispatch.nz);
        std::fill_n(dispatch.tendency, cells,
                    20.0f + static_cast<float>(calls));
        return calls != fail_on_call;
    }
};

} // namespace

TEST_CASE("Kernel manifest is complete and owns parity and ABI metadata",
          "[compute][backend][manifest]")
{
    const char* reason = nullptr;
    const auto& manifest = nimbus::compute::kernel_manifest();
    REQUIRE(nimbus::compute::validate_kernel_manifest(manifest, &reason));
    REQUIRE(reason == nullptr);
    REQUIRE(manifest.size() == nimbus::compute::kernel_count());

    const auto& lin = nimbus::compute::kernel_manifest_entry(
        nimbus::compute::KernelId::LinPointwise);
    REQUIRE(std::string(lin.operation_id) == "microphysics.lin_pointwise");
    REQUIRE(lin.constants.bytes == 112);
    REQUIRE(lin.resources.descriptor_span == 17);
    REQUIRE(lin.resources.access(0) == nimbus::compute::ResourceAccess::Read);
    REQUIRE(lin.resources.access(9) == nimbus::compute::ResourceAccess::Write);
    REQUIRE(lin.parity_tier == nimbus::parity::Tier::Microphysics);
    REQUIRE(lin.cpu_reference != nullptr);

    const auto* by_name = nimbus::compute::find_kernel(
        "microphysics/lin_pointwise.comp");
    REQUIRE(by_name != nullptr);
    REQUIRE(by_name->id == nimbus::compute::KernelId::LinPointwise);
}

TEST_CASE("Kernel manifest rejects identity enum hash and ABI drift",
          "[compute][backend][manifest]")
{
    using namespace nimbus::compute;

    const auto require_invalid = [](const KernelManifest& candidate)
    {
        const char* reason = nullptr;
        REQUIRE_FALSE(validate_kernel_manifest(candidate, &reason));
        REQUIRE(reason != nullptr);
    };

    SECTION("every constants byte count is tied to its host ABI")
    {
        for (std::size_t i = 0; i < kernel_count(); ++i)
        {
            INFO("kernel ordinal=" << i);
            KernelManifest candidate = kernel_manifest();
            ++candidate[i].constants.bytes;
            require_invalid(candidate);
        }
    }

    SECTION("alignment and version are canonical")
    {
        KernelManifest candidate = kernel_manifest();
        candidate[0].constants.alignment = 8;
        require_invalid(candidate);

        candidate = kernel_manifest();
        candidate[0].constants.version = 2;
        require_invalid(candidate);
    }

    SECTION("IDs and identities are unique")
    {
        KernelManifest candidate = kernel_manifest();
        candidate[1].id = candidate[0].id;
        require_invalid(candidate);

        candidate = kernel_manifest();
        candidate[1].operation_id = candidate[0].operation_id;
        require_invalid(candidate);

        candidate = kernel_manifest();
        candidate[1].kernel_name = candidate[0].kernel_name;
        candidate[1].artifact_name = candidate[0].artifact_name;
        require_invalid(candidate);
    }

    SECTION("artifact identity and hash use canonical spelling")
    {
        KernelManifest candidate = kernel_manifest();
        candidate[0].artifact_name = "acoustic/not-the-source.comp.spv";
        require_invalid(candidate);

        candidate = kernel_manifest();
        candidate[0].artifact_hash = "69917B3448F4D0B5";
        require_invalid(candidate);

        candidate = kernel_manifest();
        candidate[0].artifact_hash = "69917b3448f4d0bz";
        require_invalid(candidate);

        candidate = kernel_manifest();
        candidate[0].artifact_hash = "69917b3448f4d0b";
        require_invalid(candidate);
    }

    SECTION("enum values stay inside their declared domains")
    {
        KernelManifest candidate = kernel_manifest();
        candidate[0].shape = static_cast<DispatchShape>(255);
        require_invalid(candidate);

        candidate = kernel_manifest();
        candidate[0].parity_tier = static_cast<nimbus::parity::Tier>(255);
        require_invalid(candidate);

        candidate = kernel_manifest();
        candidate[0].availability =
            static_cast<KernelAvailability>(255);
        require_invalid(candidate);
    }
}

TEST_CASE("Default diffusion batch compatibility path is failure atomic",
          "[compute][backend][diffusion]")
{
    constexpr int nx = 2;
    constexpr int ny = 3;
    constexpr int nz = 4;
    constexpr std::size_t cells = nx * ny * nz;
    constexpr int targets = 3;

    std::array<std::array<float, cells>, targets> fields{};
    std::array<std::array<float, cells>, targets> destinations{};
    for (int target = 0; target < targets; ++target)
    {
        std::fill(fields[target].begin(), fields[target].end(),
                  1.0f + static_cast<float>(target));
        std::fill(destinations[target].begin(), destinations[target].end(),
                  -10.0f - static_cast<float>(target));
    }
    const auto original = destinations;
    std::array<float, cells> rho{};
    std::array<float, nz> dz{};
    rho.fill(1.0f);
    dz.fill(1.0f);

    const float* field_pointers[targets] = {
        fields[0].data(), fields[1].data(), fields[2].data()};
    float* destination_pointers[targets] = {
        destinations[0].data(), destinations[1].data(),
        destinations[2].data()};
    const DiffusionFieldLocation locations[targets] = {
        DiffusionFieldLocation::Scalar, DiffusionFieldLocation::UFace,
        DiffusionFieldLocation::WFace};
    DiffusionBatchDispatch dispatch{
        field_pointers, destination_pointers, locations, targets,
        rho.data(), dz.data(), nx, ny, nz, 1.0f, 1.0f, 2.0f, 3.0f,
        true, false, false};

    StagedDiffusionKernels kernels;
    DiffusionBatchDispatch missing_shared_input = dispatch;
    missing_shared_input.rho = nullptr;
    REQUIRE_FALSE(kernels.dispatch_diffusion_batch(missing_shared_input));
    missing_shared_input = dispatch;
    missing_shared_input.dz = nullptr;
    REQUIRE_FALSE(kernels.dispatch_diffusion_batch(missing_shared_input));
    REQUIRE(kernels.calls == 0);
    REQUIRE(std::memcmp(destinations.data(), original.data(),
                        sizeof(destinations)) == 0);

    kernels.fail_on_call = targets;
    REQUIRE_FALSE(kernels.dispatch_diffusion_batch(dispatch));
    REQUIRE(kernels.calls == targets);
    REQUIRE(std::memcmp(destinations.data(), original.data(),
                        sizeof(destinations)) == 0);

    kernels.calls = 0;
    kernels.fail_on_call = 0;
    REQUIRE(kernels.dispatch_diffusion_batch(dispatch));
    REQUIRE(kernels.calls == targets);
    for (int target = 0; target < targets; ++target)
    {
        for (float value : destinations[target])
            REQUIRE(value == 21.0f + static_cast<float>(target));
    }

    kernels.calls = 0;
    dispatch.nx = std::numeric_limits<int>::max();
    dispatch.ny = std::numeric_limits<int>::max();
    dispatch.nz = std::numeric_limits<int>::max();
    REQUIRE_FALSE(kernels.dispatch_diffusion_batch(dispatch));
    REQUIRE(kernels.calls == 0);
}

TEST_CASE("Dispatch graph materializes only data hazards",
          "[compute][backend][dispatch_graph]")
{
    using nimbus::compute::DispatchGraph;
    using nimbus::compute::KernelId;
    using nimbus::compute::ResourceAccess;

    DispatchGraph graph;
    const auto producer = graph.add(KernelId::LinPointwise, {
        {"dqr_dt", ResourceAccess::Write},
        {"dqr_dt", ResourceAccess::Read},
        {"unrelated", ResourceAccess::Read}});
    const auto independent = graph.add(KernelId::DiffuseCartesian, {
        {"theta", ResourceAccess::Read}});
    const auto consumer = graph.add(KernelId::LinSedimentation, {
        {"dqr_dt", ResourceAccess::ReadWrite}});

    REQUIRE(graph.valid());
    REQUIRE(graph.nodes()[producer].fields.size() == 2);
    REQUIRE(graph.nodes()[producer].fields[0].access ==
            ResourceAccess::ReadWrite);
    REQUIRE(graph.nodes()[producer].dependencies.empty());
    REQUIRE_FALSE(graph.depends_on(independent, producer));
    REQUIRE(graph.depends_on(consumer, producer));
    REQUIRE_FALSE(graph.requires_barrier(producer, independent));
    REQUIRE(graph.requires_barrier(producer, consumer));
    REQUIRE(graph.nodes()[consumer].dependencies.size() == 1);
    REQUIRE(graph.nodes()[consumer].dependencies[0] == producer);
}

TEST_CASE("ComputeBackend exposes narrow kernel capability views",
          "[compute][backend][capabilities]")
{
    DummyComputeBackend backend;
    const DummyComputeBackend& const_backend = backend;

    REQUIRE(backend.advection_kernels() == nullptr);
    REQUIRE(backend.dynamics_kernels() == nullptr);
    REQUIRE(backend.microphysics_kernels() == nullptr);
    REQUIRE(backend.turbulence_kernels() == nullptr);
    REQUIRE(backend.diffusion_kernels() == nullptr);

    REQUIRE(const_backend.advection_kernels() == nullptr);
    REQUIRE(const_backend.dynamics_kernels() == nullptr);
    REQUIRE(const_backend.microphysics_kernels() == nullptr);
    REQUIRE(const_backend.turbulence_kernels() == nullptr);
    REQUIRE(const_backend.diffusion_kernels() == nullptr);
    REQUIRE_FALSE(const_backend.supports_advection(
        AdvectionKernelOp::VerticalFlux));
    REQUIRE_FALSE(const_backend.supports_dynamics(
        DynamicsKernelOp::SlowApply));
    REQUIRE_FALSE(const_backend.supports_microphysics(
        MicrophysicsKernelOp::MicrophysicsApply));
    REQUIRE_FALSE(const_backend.supports_turbulence(
        TurbulenceKernelOp::Smagorinsky));
    REQUIRE_FALSE(const_backend.supports_diffusion(
        DiffusionKernelOp::Diffusion));
    REQUIRE_FALSE(
        const_backend.supports_supplied_cartesian_cgrid_mass_flux());
    REQUIRE_FALSE(
        const_backend.supports_batched_cartesian_cgrid_mass_flux_export());
}

TEST_CASE("ComputeBackend can opt into a single kernel family",
          "[compute][backend][capabilities]")
{
    AdvectionOnlyComputeBackend backend;
    const AdvectionOnlyComputeBackend& const_backend = backend;

    REQUIRE(backend.advection_kernels() ==
            static_cast<AdvectionComputeKernels*>(&backend));
    REQUIRE(backend.dynamics_kernels() == nullptr);
    REQUIRE(backend.microphysics_kernels() == nullptr);
    REQUIRE(backend.turbulence_kernels() == nullptr);
    REQUIRE(backend.diffusion_kernels() == nullptr);

    REQUIRE(const_backend.advection_kernels() ==
            static_cast<const AdvectionComputeKernels*>(&const_backend));
    REQUIRE(const_backend.advection_kernels()->supports(
        AdvectionKernelOp::VerticalFlux));
    REQUIRE_FALSE(const_backend.advection_kernels()->supports(
        AdvectionKernelOp::AdvectionX));
    REQUIRE(const_backend.supports_advection(
        AdvectionKernelOp::VerticalFlux));
    REQUIRE_FALSE(const_backend.supports_advection(
        AdvectionKernelOp::AdvectionX));
    REQUIRE(const_backend.dynamics_kernels() == nullptr);
}
