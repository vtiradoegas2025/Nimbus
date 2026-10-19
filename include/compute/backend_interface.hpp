#pragma once

#include "compute/backend_types.hpp"
#include "compute/dispatch/rk_combination.hpp"
#include "compute/dispatch/rk_tendency.hpp"
#include "compute/compute_kernel_interfaces.hpp"
#include "compute/kernel_manifest.hpp"
#include "core/infra/coordinate_system.hpp"

#include <cstdint>
#include <string>

struct HardwareInfo;

// Backend implementation contract. Runtime ownership and selection policy live
// in backend_runtime.hpp so backend implementations do not inherit lifecycle
// orchestration declarations.

class ComputeBackend
{
public:
    virtual ~ComputeBackend() = default;

    virtual std::string name() const = 0;

    // Supplies the owning runtime's selection policy before initialize().
    // Backends must not recover per-runtime configuration through globals:
    // independent simulation/session runtimes may request different devices.
    virtual void configure(const ComputeBackendConfig& /*config*/) {}

    virtual bool initialize(std::string& error) = 0;

    virtual void shutdown() = 0;

    // Called once after initialization. The model is currently Cartesian-only.
    virtual void set_coordinate_system(CoordinateSystem /*cs*/) {}

    // CPU leaves hardware defaults; accelerators add selected-device details.
    virtual void populate_hardware_info(HardwareInfo& /*info*/) const {}

    // Available after initialize(). Runtime admission uses the active backend,
    // so a declared fallback is planned as CPU rather than fictitious GPU
    // residency.
    virtual ComputeMemoryProperties memory_properties() const { return {}; }

    // Completed, fence-confirmed submissions in this backend generation.
    virtual std::uint64_t successful_dispatch_count() const { return 0; }

    // Backend-neutral residency telemetry used by production diagnostics.
    // Uploads are first materializations of a resident field generation;
    // reuses avoid that upload. CPU and backends without resident storage
    // retain the zero defaults.
    virtual std::uint64_t resident_upload_count() const { return 0; }
    virtual std::uint64_t resident_reuse_count() const { return 0; }

    // Successful scientific operations by stable kernel identity in this
    // backend generation. Accelerators count completed, fence-confirmed
    // shader dispatches; CPU may count an equivalent fused operation. The
    // aggregate counter above retains its completed-submission meaning.
    virtual std::uint64_t successful_kernel_dispatch_count(
        nimbus::compute::KernelId /*kernel*/) const
    {
        return 0;
    }

    // Null families are unavailable. The caller's runtime policy decides
    // whether that is a fail-closed error or permits a local CPU path.
    virtual AdvectionComputeKernels* advection_kernels() { return nullptr; }
    virtual const AdvectionComputeKernels* advection_kernels() const { return nullptr; }

    virtual DynamicsComputeKernels* dynamics_kernels() { return nullptr; }
    virtual const DynamicsComputeKernels* dynamics_kernels() const { return nullptr; }

    virtual MicrophysicsComputeKernels* microphysics_kernels() { return nullptr; }
    virtual const MicrophysicsComputeKernels* microphysics_kernels() const { return nullptr; }

    virtual TurbulenceComputeKernels* turbulence_kernels() { return nullptr; }
    virtual const TurbulenceComputeKernels* turbulence_kernels() const { return nullptr; }

    virtual DiffusionComputeKernels* diffusion_kernels() { return nullptr; }
    virtual const DiffusionComputeKernels* diffusion_kernels() const { return nullptr; }

    virtual BoundaryLayerComputeKernels* boundary_layer_kernels() { return nullptr; }
    virtual const BoundaryLayerComputeKernels* boundary_layer_kernels() const { return nullptr; }

    virtual RadiationComputeKernels* radiation_kernels() { return nullptr; }
    virtual const RadiationComputeKernels* radiation_kernels() const { return nullptr; }

    virtual TerrainComputeKernels* terrain_kernels() { return nullptr; }
    virtual const TerrainComputeKernels* terrain_kernels() const { return nullptr; }

    virtual bool supports_rk_tendency() const
    {
        return false;
    }
    virtual RkCombinationStatus dispatch_rk_tendency(const RkTendencyDispatch&)
    {
        return RkCombinationStatus::Unavailable;
    }
    virtual bool supports_rk_combination() const
    {
        return false;
    }
    virtual RkCombinationStatus dispatch_rk_combination(const RkCombinationDispatch&)
    {
        return RkCombinationStatus::Unavailable;
    }

    // Exact recycling field primitive. Live execution requires a later atomic
    // multi-owner transaction; this operation cannot by itself authorize a
    // model-state commit.
    virtual bool supports_homogeneous_domain_field_shift() const
    {
        return false;
    }
    virtual bool dispatch_homogeneous_domain_field_shift(
        const HomogeneousDomainFieldShiftDispatch& /*dispatch*/)
    {
        return false;
    }
    virtual bool dispatch_homogeneous_domain_field_shift_batch(
        const HomogeneousDomainFieldShiftBatchDispatch& /*dispatch*/)
    {
        return false;
    }

    virtual bool supports_earth_control_evolution() const { return false; }
    virtual bool dispatch_earth_control_evolution(
        const EarthControlEvolutionDispatch& /*dispatch*/)
    {
        return false;
    }

    // Cheap gates used before packing profiles or accelerator arguments.
    bool supports_advection(AdvectionKernelOp op) const
    {
        const auto* kernels = advection_kernels();
        return kernels != nullptr && kernels->supports(op);
    }

    bool supports_dynamics(DynamicsKernelOp op) const
    {
        const auto* kernels = dynamics_kernels();
        return kernels != nullptr && kernels->supports(op);
    }

    bool supports_microphysics(MicrophysicsKernelOp op) const
    {
        const auto* kernels = microphysics_kernels();
        return kernels != nullptr && kernels->supports(op);
    }

    bool supports_turbulence(TurbulenceKernelOp op) const
    {
        const auto* kernels = turbulence_kernels();
        return kernels != nullptr && kernels->supports(op);
    }

    bool supports_diffusion(DiffusionKernelOp op) const
    {
        const auto* kernels = diffusion_kernels();
        return kernels != nullptr && kernels->supports(op);
    }

    bool supports_boundary_layer(BoundaryLayerKernelOp op) const
    {
        const auto* kernels = boundary_layer_kernels();
        return kernels != nullptr && kernels->supports(op);
    }

    bool supports_radiation(RadiationKernelOp op) const
    {
        const auto* kernels = radiation_kernels();
        return kernels != nullptr && kernels->supports(op);
    }

    bool supports_mapped_terrain_mass_continuity() const
    {
        const auto* kernels = terrain_kernels();
        return kernels != nullptr && kernels->supports_mapped_mass_continuity();
    }

    bool supports_mapped_terrain_fast_momentum() const
    {
        const auto* kernels = terrain_kernels();
        return kernels != nullptr && kernels->supports_mapped_fast_momentum();
    }

    bool supports_mapped_terrain_slow_tendencies() const
    {
        const auto* kernels = terrain_kernels();
        return kernels != nullptr && kernels->supports_mapped_slow_tendencies();
    }

    bool supports_mapped_terrain_scalar_transport() const
    {
        const auto* kernels = terrain_kernels();
        return kernels != nullptr && kernels->supports_mapped_scalar_transport();
    }

    bool supports_mapped_terrain_final_state() const
    {
        const auto* kernels = terrain_kernels();
        return kernels != nullptr && kernels->supports_mapped_final_state();
    }

    bool supports_mapped_terrain_state_update() const
    {
        const auto* kernels = terrain_kernels();
        return kernels != nullptr && kernels->supports_mapped_state_update();
    }

    bool supports_mapped_terrain_flux_accumulation() const
    {
        const auto* kernels = terrain_kernels();
        return kernels != nullptr && kernels->supports_mapped_flux_accumulation();
    }

    bool supports_supplied_cartesian_cgrid_mass_flux() const
    {
        const auto* kernels = advection_kernels();
        return kernels != nullptr
            && kernels->supports_supplied_cartesian_cgrid_mass_flux();
    }

    bool supports_batched_cartesian_cgrid_mass_flux_export() const
    {
        const auto* kernels = dynamics_kernels();
        return kernels != nullptr
            && kernels->supports_batched_cartesian_cgrid_mass_flux_export();
    }

    // Starts a residency generation. Without step boundaries, dispatches
    // upload fresh data so direct kernel tests cannot reuse stale fields.
    virtual void begin_resident_step() {}

    // Selects the fixed level-owned residency namespace used by subsequent
    // operations. Switching changes no allocation or scientific state.
    virtual void select_residency_scope(
        ComputeResidencyScope /*scope*/) noexcept {}

    // Invalidates one fixed level-owned namespace without changing the active
    // namespace. Nested host mutations use this operation so child authority
    // cannot survive a binding edge while unrelated root lineage remains
    // eligible. This is an ownership-safety contract, not a residency reuse
    // or performance guarantee.
    virtual void invalidate_residency_scope(
        ComputeResidencyScope /*scope*/) noexcept {}

    // Marks a stable field ID as host-modified.
    virtual void invalidate_resident_field(const char* /*field_id*/) noexcept {}

    // Invalidates all device copies after an unclassified host mutation.
    virtual void invalidate_all_resident_fields() noexcept {}
};
