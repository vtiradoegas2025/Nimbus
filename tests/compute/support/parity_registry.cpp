#include "microphysics/schemes/milbrandt/sedimentation.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/melting.hpp"
#include "microphysics/schemes/milbrandt/activation.hpp"
#include "microphysics/schemes/milbrandt/warm_rain.hpp"
/**
 * @file parity_registry.cpp
 * @brief Test-only shader-to-CPU-reference registry and its anchors.
 *
 * Every CPU reference named in the registry table below is pinned by a
 * NIMBUS_PARITY_ANCHOR: an unevaluated decltype of the symbol's address.
 * Renaming or removing the CPU entry point breaks this translation unit at
 * compile time, which is the whole point - the pairing can no longer rot the
 * way the shader header comments did (GPU.md section 2.4).
 *
 * CPU kernels that are file-static (the legacy 1D split kernels in
 * advection.cpp) cannot be anchored across translation units; for those the
 * closest public entry point is anchored and the table's cpu_reference string
 * names the static kernel it routes to.
 */
#include "compute/support/parity_registry.hpp"
#include "compute/kernel_manifest.hpp"

#include "core/infra/domain_recycling.hpp"
#include "core/infra/nested_scalar_feedback.hpp"
#include "core/infra/rayleigh_damping.hpp"
#include "dynamics/schemes/klemp_wilhelmson_split/cartesian_agrid.hpp"
#include "dynamics/schemes/klemp_wilhelmson_split/cartesian_cgrid_3d.hpp"
#include "numerics/advection/advection.hpp"
#include "numerics/advection/advection_cartesian.hpp"
#include "turbulence/schemes/smagorinsky/smagorinsky.hpp"
#include "turbulence/schemes/tke/tke.hpp"
#include "numerics/diffusion/schemes/implicit/implicit.hpp"
#include "turbulence/base/eddy_viscosity.hpp"

#include <cstring>
#include <type_traits>

// The microphysics anchors live in parity_registry_microphysics_anchors.cpp:
// the dynamics scheme headers pull in simulation.hpp, whose legacy
// GLOBAL-SCOPE physics constants (T0, L_v, L_f, ...) collide with the
// function-local `using namespace microphysics_constants` directives in
// thermodynamics.hpp when both header families share one translation unit.

namespace nimbus::parity {

namespace {

using K = nimbus::compute::KernelId;
constexpr const char* metal_fixture =
    "tests/metal/test_metal_compute_backend.cpp";

constexpr ParityCoverage runtime_coverage(
    const char* outputs, const char* sample,
    HorizonRule horizon = HorizonRule::SingleDispatch)
{
    return {outputs, sample, nullptr, metal_fixture, horizon,
            ReplayOwner::RuntimeAndHardware,
            RequiredHardware::VulkanAndMetal, nullptr};
}

constexpr ParityCoverage hardware_coverage(
    const char* outputs, const char* test,
    HorizonRule horizon = HorizonRule::SingleDispatch)
{
    return {outputs, nullptr, test, metal_fixture, horizon,
            ReplayOwner::HardwareFixture,
            RequiredHardware::VulkanAndMetal, nullptr};
}

constexpr ParityCoverage combined_coverage(
    const char* outputs, const char* sample, const char* test,
    HorizonRule horizon = HorizonRule::SingleDispatch)
{
    return {outputs, sample, test, metal_fixture, horizon,
            ReplayOwner::RuntimeAndHardware,
            RequiredHardware::VulkanAndMetal, nullptr};
}

constexpr ParityCoverage waived_coverage(const char* reason)
{
    return {nullptr, nullptr, nullptr, nullptr,
            HorizonRule::SingleDispatch, ReplayOwner::Waived,
            RequiredHardware::None, reason};
}

ParityCoverage coverage_for(K id)
{
    constexpr const char* gpu = "tests/vulkan/test_gpu_parity.cpp";
    constexpr const char* acoustic =
        "tests/dynamics/test_acoustic_cartesian_cgrid_gpu_parity.cpp";
    constexpr const char* transport =
        "tests/numerics/test_advect_conservative_cartesian_cgrid_gpu_parity.cpp";
    constexpr const char* slow =
        "tests/dynamics/test_cartesian_cgrid_slow_tendencies_gpu_parity.cpp";
    constexpr auto accumulated = HorizonRule::AccumulatedSubsteps;

    switch (id)
    {
    case K::ConservativeMomentum:
        return {"du_dt,dv_dt,dw_dt,interface_momentum_flux,source_impulse",
                nullptr,
                "tests/numerics/test_native_conservative_momentum.cpp",
                "tests/numerics/test_native_conservative_momentum.cpp",
                HorizonRule::SingleDispatch,
                ReplayOwner::HardwareFixture,
                RequiredHardware::VulkanAndMetal,
                nullptr};
    case K::ConservativeReduction:
        return {"inventory,stored,storage_rounding",
                nullptr,
                "tests/numerics/test_native_conservative_momentum.cpp",
                "tests/numerics/test_native_conservative_momentum.cpp",
                HorizonRule::SingleDispatch,
                ReplayOwner::HardwareFixture,
                RequiredHardware::VulkanAndMetal,
                nullptr};
    case K::AcousticMomentumCartesian:
        return hardware_coverage("u,v,w", gpu);
    case K::AcousticMomentumCartesianCgrid:
        return combined_coverage("u,v,w", "acoustic_batched_cartesian_cgrid",
                                 acoustic, accumulated);
    case K::AcousticPressureCartesian:
        return hardware_coverage("rho,p", gpu);
    case K::AcousticPressureCartesianCgrid:
        return combined_coverage("rho,p", "acoustic_batched_cartesian_cgrid",
                                 acoustic, accumulated);
    case K::AcousticPressureMassFluxCartesianCgrid:
        return hardware_coverage("rho,p,mass_flux_x,mass_flux_y,mass_flux_z",
                                 acoustic, accumulated);
    case K::AcousticRetainedFluxFp64:
        return {"mass_flux_x,mass_flux_y,mass_flux_z",
                nullptr,
                "tests/vulkan/test_acoustic_input_reuse.cpp",
                nullptr,
                accumulated,
                ReplayOwner::HardwareFixture,
                RequiredHardware::Vulkan,
                nullptr};
    case K::AcousticContinuityMassFluxCartesianAgrid:
        return hardware_coverage(
            "drho_dt,mass_flux_x,mass_flux_y,mass_flux_z", gpu);
    case K::EarthControlEvolution:
        return hardware_coverage("earth_control_state", gpu);
    case K::AcousticPressureEosCartesianAgrid:
        return hardware_coverage("rho,p", gpu, accumulated);
    case K::AcousticMomentumCartesianAgrid:
        return hardware_coverage("u,v,w", gpu, accumulated);
    case K::AcousticNullModeConstraintCartesianAgrid:
        return hardware_coverage("u,v,w", gpu, accumulated);
    case K::AcousticBoundaryCartesianAgrid:
        return hardware_coverage("u,v,w,rho,p", gpu, accumulated);
    case K::AdvectConservativeMassFluxCartesianCgrid:
    case K::AdvectConservativeScalarFluxCartesianCgrid:
    case K::AdvectConservativeUpdateCartesianCgrid:
    case K::AdvectPdLimiterCartesianCgrid:
        return combined_coverage(
            "scalar", "advect_scalars_conservative_cartesian_cgrid", transport);
    case K::AdvectX:
        return combined_coverage("scalar", "advect_x", gpu);
    case K::AdvectY:
        return runtime_coverage("scalar", "advect_y");
    case K::TvdVerticalFlux:
        return runtime_coverage("dqdt_adv", "tvd_vertical_flux");
    case K::DiffuseCartesian:
        return waived_coverage(
            "unweighted transport-chained point diffusion is intentionally unreachable");
    case K::Diffusion:
        return hardware_coverage("scalar", gpu);
    case K::DomainRecyclingHomogeneousFieldShift:
        return hardware_coverage("field", gpu);
    case K::AcousticBcFaceCartesianCgrid:
        return combined_coverage("u,v,w,rho,p", "acoustic_batched_cartesian_cgrid",
                                 acoustic, accumulated);
    case K::CartesianCgridSlowTendencies:
        return combined_coverage(
            "du_dt,dv_dt,dw_dt,drho_dt,dp_dt",
            "cartesian_cgrid_slow_tendencies", slow);
    case K::CartesianTendencies:
        return hardware_coverage("du_dt,dv_dt,dw_dt,drho_dt,dp_dt", gpu);
    case K::SlowApply:
        return runtime_coverage("u,v,w,p", "slow_apply");
    case K::StateRelaxationCartesian:
        return runtime_coverage("u,v,w,theta,rho,p",
                                "state_relaxation_cartesian");
    case K::KesslerPointwise:
    case K::KesslerSedimentation:
        return hardware_coverage(
            "dtheta_dt,dqv_dt,dqc_dt,dqr_dt,dqg_dt,dqh_dt", gpu);
    case K::LinPointwise:
    case K::LinSedimentation:
        return combined_coverage(
            "dtheta_dt,dqv_dt,dqc_dt,dqr_dt,dqi_dt,dqs_dt,dqg_dt,dqh_dt",
            "lin_microphysics", gpu);
    case K::MicrophysicsApply:
        return combined_coverage("theta,qv,qc,qr,qi,qs,qg,qh",
                                 "microphysics_apply", gpu);
    case K::ArchivedLegacyThompsonLinPointwise:
    case K::ArchivedLegacyThompsonLinSedimentation:
        return waived_coverage("compiled shader is orphaned and not dispatchable");
    case K::SgsApply:
        return combined_coverage("u,v,w,theta,qv,tke", "sgs_apply", gpu);
    case K::SmagorinskyCoefficients:
        return combined_coverage("nu_t,K_theta,K_q", "smagorinsky_sgs", gpu);
    case K::TkeClosure:
    case K::TkeTendency:
        return {"nu_t,K_theta,K_q,K_tke,shear,buoyancy,dissipation,Pr_t_eff,dtke_dt",
                nullptr,
                "tests/physics/test_native_tke.cpp",
                "tests/physics/test_native_tke.cpp",
                HorizonRule::SingleDispatch,
                ReplayOwner::HardwareFixture,
                RequiredHardware::VulkanAndMetal,
                nullptr};
    case K::ThompsonCloudAdjustment:
        return {"vapor_change,temperature_change,cloud_number,aerosol,lookup_index",
                nullptr,
                "tests/physics/test_native_thompson_cloud.cpp",
                "tests/physics/test_native_thompson_cloud.cpp",
                HorizonRule::SingleDispatch,
                ReplayOwner::HardwareFixture,
                RequiredHardware::VulkanAndMetal,
                nullptr};
    case K::ThompsonCompletion:
        return {"coupled_water_heat_number_tendencies,snow_multiplier,lookup_indices,status",
                nullptr,
                "tests/physics/test_native_thompson_completion.cpp",
                "tests/physics/test_native_thompson_completion.cpp",
                HorizonRule::SingleDispatch,
                ReplayOwner::HardwareFixture,
                RequiredHardware::VulkanAndMetal,
                nullptr};
    case K::ThompsonDeposition:
        return {"ice_snow_graupel_exchange,latent_heat,ice_number,snow_vapor_rate,lookup_index,status",
                nullptr,
                "tests/physics/test_native_thompson_deposition.cpp",
                "tests/physics/test_native_thompson_deposition.cpp",
                HorizonRule::SingleDispatch,
                ReplayOwner::HardwareFixture,
                RequiredHardware::VulkanAndMetal,
                nullptr};
    case K::ThompsonFreezing:
        return {"frozen_cloud,rain_to_ice,rain_to_graupel,number_sources,lookup_indices,status",
                nullptr,
                "tests/physics/test_native_thompson_freezing.cpp",
                "tests/physics/test_native_thompson_freezing.cpp",
                HorizonRule::SingleDispatch,
                ReplayOwner::HardwareFixture,
                RequiredHardware::VulkanAndMetal,
                nullptr};
    case K::ThompsonCloudPhase:
        return {"cloud_water,cloud_ice,cloud_number,ice_number,latent_heat,status",
                nullptr,
                "tests/physics/test_native_thompson_phase.cpp",
                "tests/physics/test_native_thompson_phase.cpp",
                HorizonRule::SingleDispatch,
                ReplayOwner::HardwareFixture,
                RequiredHardware::VulkanAndMetal,
                nullptr};
    case K::MilbrandtMelting:
        return {"ice,snow,graupel,hail,rain,heat,number_tendencies",
                nullptr,
                "tests/physics/test_native_milbrandt_melting.cpp",
                "tests/physics/test_native_milbrandt_melting.cpp",
                HorizonRule::SingleDispatch,
                ReplayOwner::HardwareFixture,
                RequiredHardware::VulkanAndMetal,
                nullptr};
    case K::MilbrandtSedimentation:
        return {"sedimented_mass_number_surface_loss",
                nullptr,
                "tests/physics/test_native_milbrandt_sedimentation.cpp",
                "tests/physics/test_native_milbrandt_sedimentation.cpp",
                HorizonRule::SingleDispatch,
                ReplayOwner::HardwareFixture,
                RequiredHardware::VulkanAndMetal,
                nullptr};
    case K::SlabColumn:
        return {"slab_height_surface_fluxes_and_conservative_mixing", nullptr,
                "tests/physics/test_native_slab.cpp", "tests/physics/test_native_slab.cpp",
                HorizonRule::SingleDispatch, ReplayOwner::HardwareFixture, RequiredHardware::VulkanAndMetal, nullptr};
    case K::SimpleGrey:
        return {"grey_optics_fluxes_heating_and_diagnostics", nullptr,
                "tests/physics/test_native_simple_grey.cpp", "tests/physics/test_native_simple_grey.cpp",
                HorizonRule::SingleDispatch, ReplayOwner::HardwareFixture, RequiredHardware::VulkanAndMetal, nullptr};
    case K::HeldSources:
        return {"live_mass_held_source_reconstruction", nullptr,
                "tests/core/test_native_held_sources.cpp", "tests/core/test_native_held_sources.cpp",
                HorizonRule::SingleDispatch, ReplayOwner::HardwareFixture, RequiredHardware::VulkanAndMetal, nullptr};
    case K::ColumnRemap:
        return {"held_columns_slab_and_rainfall", nullptr,
                "tests/core/test_held_source_remap.cpp", "tests/core/test_held_source_remap.cpp",
                HorizonRule::SingleDispatch, ReplayOwner::HardwareFixture, RequiredHardware::VulkanAndMetal, nullptr};
    case K::ReferenceAtmosphere:
        return {"terrain_height_environmental_ghosts", nullptr,
                "tests/dynamics/test_moving_environment.cpp", "tests/dynamics/test_moving_environment.cpp",
                HorizonRule::SingleDispatch, ReplayOwner::HardwareFixture, RequiredHardware::VulkanAndMetal, nullptr};
    case K::MovingEnergy:
        return {"moist_energy_and_thermodynamic_storage", nullptr,
                "tests/dynamics/test_native_moving_energy.cpp", "tests/dynamics/test_native_moving_energy.cpp",
                HorizonRule::SingleDispatch, ReplayOwner::HardwareFixture, RequiredHardware::VulkanAndMetal, nullptr};
    case K::MovingMomentum:
        return {"moving_dual_momentum_and_work_ledgers", nullptr,
                "tests/dynamics/test_native_moving_momentum.cpp", "tests/dynamics/test_native_moving_momentum.cpp",
                HorizonRule::SingleDispatch, ReplayOwner::HardwareFixture, RequiredHardware::VulkanAndMetal, nullptr};
    case K::EarthGeometry:
        return {"earth_surface_means_metrics_and_swept_volumes", nullptr,
                "tests/core/test_native_earth_geometry.cpp", "tests/core/test_native_earth_geometry.cpp",
                HorizonRule::SingleDispatch, ReplayOwner::HardwareFixture, RequiredHardware::VulkanAndMetal, nullptr};
    case K::ConservativeRemap:
        return {"remap_value_and_inventory_ledger",nullptr,
                "tests/core/test_native_conservative_remap.cpp","tests/core/test_native_conservative_remap.cpp",
                HorizonRule::SingleDispatch,ReplayOwner::HardwareFixture,RequiredHardware::VulkanAndMetal,nullptr};
    case K::CollocatedSgs:
        return {"collocated_stress_and_scalar_tendencies", nullptr,
                "tests/physics/test_native_collocated_sgs.cpp", "tests/physics/test_native_collocated_sgs.cpp",
                HorizonRule::SingleDispatch, ReplayOwner::HardwareFixture, RequiredHardware::VulkanAndMetal, nullptr};
    case K::RkTendency:
        return {"acoustic_and_transport_tendencies", nullptr,
                "tests/numerics/test_native_rk_tendency.cpp",
                "tests/numerics/test_native_rk_tendency.cpp",
                HorizonRule::SingleDispatch, ReplayOwner::HardwareFixture,
                RequiredHardware::VulkanAndMetal, nullptr};
    case K::RkCombination:
        return {"stage_value_and_roundoff_scale",
                nullptr,
                "tests/numerics/test_native_rk_combination.cpp",
                "tests/numerics/test_native_rk_combination.cpp",
                HorizonRule::SingleDispatch,
                ReplayOwner::HardwareFixture,
                RequiredHardware::VulkanAndMetal,
                nullptr};
    case K::MilbrandtCold:
        return {"cold_mass_heat_number_tendencies",
                nullptr,
                "tests/physics/test_native_milbrandt_cold.cpp",
                "tests/physics/test_native_milbrandt_cold.cpp",
                HorizonRule::SingleDispatch,
                ReplayOwner::HardwareFixture,
                RequiredHardware::VulkanAndMetal,
                nullptr};
    case K::MilbrandtCloudAdjustment:
        return {"adjusted_temperature,vapor,cloud,heat_water_number_tendencies",
                nullptr,
                "tests/physics/test_native_milbrandt_cloud.cpp",
                "tests/physics/test_native_milbrandt_cloud.cpp",
                HorizonRule::SingleDispatch,
                ReplayOwner::HardwareFixture,
                RequiredHardware::VulkanAndMetal,
                nullptr};
    case K::MilbrandtWarmRain:
        return {"cloud,rain,vapor,temperature,cloud_number,rain_number_tendencies",
                nullptr,
                "tests/physics/test_native_milbrandt_warm.cpp",
                "tests/physics/test_native_milbrandt_warm.cpp",
                HorizonRule::SingleDispatch,
                ReplayOwner::HardwareFixture,
                RequiredHardware::VulkanAndMetal,
                nullptr};
    case K::ThompsonWarmRain:
        return {"cloud_to_rain,rain_to_vapor,temperature_change,cloud_number,rain_number,aerosol,status",
                nullptr,
                "tests/physics/test_native_thompson_warm.cpp",
                "tests/physics/test_native_thompson_warm.cpp",
                HorizonRule::SingleDispatch,
                ReplayOwner::HardwareFixture,
                RequiredHardware::VulkanAndMetal,
                nullptr};
    case K::ThompsonMomentSedimentation:
        return {"mass_tendency,number_tendency,surface_precipitation",
                nullptr,
                "tests/physics/test_native_thompson_sedimentation.cpp",
                "tests/physics/test_native_thompson_sedimentation.cpp",
                HorizonRule::SingleDispatch,
                ReplayOwner::HardwareFixture,
                RequiredHardware::VulkanAndMetal,
                nullptr};
    case K::ImplicitDiffusionStencil:
    case K::ImplicitDiffusionVector:
        return {"backward_euler_tendencies,diagonal,physical_mask,residual,conservation", nullptr,
                "tests/numerics/test_native_implicit.cpp", "tests/numerics/test_native_implicit.cpp",
                HorizonRule::SingleDispatch, ReplayOwner::HardwareFixture,
                RequiredHardware::VulkanAndMetal, nullptr};
    case K::SmagorinskyTendencies:
        return combined_coverage(
            "dudt_sgs,dvdt_sgs,dwdt_sgs,dthetadt_sgs,dqvdt_sgs",
            "smagorinsky_sgs", gpu);
    case K::MynnLevel25Stability:
        return hardware_coverage(
            "sm,sh,gradient_richardson,flux_richardson,equilibrium_limiter",
            gpu);
    case K::RrtmgLongwaveTransfer:
    case K::RrtmgLongwaveTransferBatch:
        return hardware_coverage("upward_wm2,downward_wm2", gpu);
    case K::RrtmgShortwaveTransfer:
    case K::RrtmgShortwaveTransferBatch:
        return hardware_coverage("upward_wm2,downward_wm2", gpu);
    case K::MappedTerrainMassContinuity:
        return hardware_coverage(
            "ground_w,drho_dt,dp_dt,mass_flux_x,mass_flux_y,mass_flux_zeta",
            "tests/vulkan/test_mapped_mass_continuity.cpp");
    case K::MappedTerrainFastMomentum:
        return hardware_coverage(
            "du_dt,dv_dt,dw_dt", "tests/vulkan/test_mapped_mass_continuity.cpp");
    case K::MappedTerrainSlowTendencies:
        return hardware_coverage(
            "du_dt,dv_dt,dw_dt,drho_dt,dp_dt,advection_attribution",
            "tests/vulkan/test_mapped_mass_continuity.cpp");
    case K::MappedTerrainScalarUpdate:
    case K::MappedTerrainScalarLimiter:
        return hardware_coverage(
            "scalar,scalar_flux,limiter_factors", "tests/vulkan/test_mapped_mass_continuity.cpp");
    case K::MappedTerrainStateUpdate:
        return hardware_coverage(
            "u,v,w,rho,p", "tests/vulkan/test_mapped_mass_continuity.cpp");
    case K::MappedTerrainFluxAccumulation:
        return hardware_coverage(
            "mass_flux_x,mass_flux_y,mass_flux_zeta",
            "tests/vulkan/test_mapped_mass_continuity.cpp");
    case K::MappedTerrainEos:
        return hardware_coverage(
            "u,v,w,rho,p,eos,boundary_attribution", "tests/vulkan/test_mapped_mass_continuity.cpp");
    case K::MappedTerrainBoundary:
        return {"full_step_boundary_fields_and_wave_history", nullptr,
                "tests/dynamics/test_native_cgrid_boundary.cpp", "tests/dynamics/test_native_cgrid_boundary.cpp",
                HorizonRule::SingleDispatch, ReplayOwner::HardwareFixture, RequiredHardware::VulkanAndMetal, nullptr};
    case K::Count:
        break;
    }
    return waived_coverage("invalid kernel identifier");
}

/// Pins a CPU entry point at compile time without ODR-using its definition
/// (no link dependency): a rename breaks the build here, next to the table.
#define NIMBUS_PARITY_ANCHOR(symbol) \
    static_assert(std::is_member_function_pointer_v<decltype(&symbol)> || \
                      std::is_pointer_v<decltype(&symbol)>, \
                  "parity registry anchor: " #symbol)

// Acoustic substeps (Cartesian collocated A-grid).
NIMBUS_PARITY_ANCHOR(CartesianAGridScheme::compute_fast_pressure_tendencies);
NIMBUS_PARITY_ANCHOR(CartesianAGridScheme::compute_fast_momentum_tendencies);

// Cartesian C-grid acoustic substeps (GPU audit Phase 3.1).
NIMBUS_PARITY_ANCHOR(CartesianCGrid3DScheme::compute_fast_pressure_tendencies);
NIMBUS_PARITY_ANCHOR(CartesianCGrid3DScheme::compute_fast_momentum_tendencies);

// Slow / momentum tendencies.
NIMBUS_PARITY_ANCHOR(CartesianCGrid3DScheme::compute_slow_tendencies);
NIMBUS_PARITY_ANCHOR(prepare_nested_scalar_feedback_flat);
NIMBUS_PARITY_ANCHOR(prepare_nested_scalar_feedback_mapped);
NIMBUS_PARITY_ANCHOR(CartesianAGridScheme::compute_momentum_tendencies);

// Advection (public entry points; the legacy 1D split kernels are
// file-static inside advection.cpp and are routed via advect_scalar_3d).
static_assert(std::is_same_v<void, decltype(advect_scalar_3d(
    std::declval<const SimulationStateView&>(), std::declval<Field3D&>(), 0.0))>);
static_assert(std::is_same_v<void, decltype(advect_scalar_conservative_3d_cartesian(
    std::declval<const SimulationStateView&>(), std::declval<const Field3D&>(),
    std::declval<Field3D&>(), 0.0))>);
static_assert(std::is_same_v<void, decltype(apply_diffusion_kernel_cartesian(
    std::declval<const SimulationStateView&>(), std::declval<const Field3D&>(),
    std::declval<Field3D&>(), 0.0, 0.0))>);
NIMBUS_PARITY_ANCHOR(tmv::domain::execute_homogeneous_field_shift);

// Microphysics anchors: see parity_registry_microphysics_anchors.cpp.

// State relaxation (GPU audit Phase 3.5). The production sponge entry
// points now return exact operator-time energy attribution.
using SpongeEntryPoint = tmv::SpongeEnergyReportV1 (*)(double);
static_assert(std::is_pointer_v<decltype(static_cast<SpongeEntryPoint>(
                  &apply_rayleigh_damping))>,
              "parity registry anchor: apply_rayleigh_damping");
static_assert(std::is_pointer_v<decltype(static_cast<SpongeEntryPoint>(
                  &apply_lateral_damping))>,
              "parity registry anchor: apply_lateral_damping");

// Turbulence (GPU audit Phase 3.4).
NIMBUS_PARITY_ANCHOR(SmagorinskyScheme::compute);
NIMBUS_PARITY_ANCHOR(TKEScheme::compute);
NIMBUS_PARITY_ANCHOR(ImplicitDiffusionScheme::compute_diffusion_tendencies);
NIMBUS_PARITY_ANCHOR(eddy_viscosity::compute_strain_rate_3d);
using CgridMomentumDiffusionEntryPoint = void (*)(
    const TurbulenceConfig&, const TurbulenceStateView&, const GridMetrics&,
    const Field3D&, Field3D&, Field3D&, Field3D&, bool, bool, double*);
static_assert(std::is_pointer_v<decltype(static_cast<CgridMomentumDiffusionEntryPoint>(
                  &eddy_viscosity::compute_momentum_diffusion_tendencies))>,
              "parity registry anchor: C-grid momentum diffusion");
NIMBUS_PARITY_ANCHOR(eddy_viscosity::compute_scalar_diffusion_tendencies);

NIMBUS_PARITY_ANCHOR(tmv::milbrandt_detail::warm_rain_cell);
NIMBUS_PARITY_ANCHOR(tmv::milbrandt_detail::cold_cell);
NIMBUS_PARITY_ANCHOR(tmv::milbrandt_detail::sediment_column);
NIMBUS_PARITY_ANCHOR(tmv::milbrandt_detail::melting_cell);
NIMBUS_PARITY_ANCHOR(tmv::milbrandt_detail::continental_activated_ccn);
NIMBUS_PARITY_ANCHOR(thermodynamics::saturation_adjustment);

#undef NIMBUS_PARITY_ANCHOR

}  // namespace

const std::vector<ParityEntry>& registry()
{
    static const std::vector<ParityEntry> entries = [] {
        std::vector<ParityEntry> result;
        result.reserve(nimbus::compute::kernel_count());
        for (const auto& kernel : nimbus::compute::kernel_manifest())
        {
            const auto coordinate =
                kernel.coordinate == nimbus::compute::KernelCoordinate::Cartesian
                    ? ParityEntry::Coordinate::Cartesian
                    : ParityEntry::Coordinate::Any;
            ShaderStatus status = ShaderStatus::Active;
            if (kernel.availability == nimbus::compute::KernelAvailability::Deferred)
                status = ShaderStatus::Deferred;
            else if (kernel.availability == nimbus::compute::KernelAvailability::Orphan)
                status = ShaderStatus::Orphan;
            result.push_back({
                kernel.kernel_name,
                kernel.cpu_reference,
                kernel.parity_tier,
                coordinate,
                kernel.comparison_domain,
                status,
                coverage_for(kernel.id)});
        }
        return result;
    }();
    return entries;
}

bool validate_registry(const std::vector<ParityEntry>& entries,
                       const char** reason) noexcept
{
    const auto fail = [reason](const char* message) {
        if (reason) *reason = message;
        return false;
    };
    const auto present = [](const char* value) {
        return value != nullptr && value[0] != '\0';
    };

    if (entries.size() != nimbus::compute::kernel_count())
        return fail("parity registry must cover every kernel manifest entry");

    for (std::size_t i = 0; i < entries.size(); ++i)
    {
        const auto& entry = entries[i];
        if (!present(entry.shader) || !present(entry.cpu_reference))
            return fail("parity identity and CPU reference are required");
        for (std::size_t j = 0; j < i; ++j)
        {
            if (std::strcmp(entries[j].shader, entry.shader) == 0)
                return fail("duplicate parity shader identity");
        }

        const bool runtime = present(entry.coverage.runtime_sample);
        const bool vulkan_hardware = present(entry.coverage.hardware_test);
        const bool metal_hardware =
            present(entry.coverage.metal_hardware_test);
        const bool hardware = vulkan_hardware || metal_hardware;
        if (entry.status == ShaderStatus::Active)
        {
            if (!present(entry.coverage.outputs))
                return fail("active parity entry must declare checked outputs");
            if (!runtime && !hardware)
                return fail("active parity entry needs runtime or hardware coverage");
            const bool vulkan_only = nimbus::compute::kernel_manifest()[i].id == K::AcousticRetainedFluxFp64;
            const auto required    = vulkan_only ? RequiredHardware::Vulkan : RequiredHardware::VulkanAndMetal;
            if (entry.coverage.required_hardware != required)
                return fail("active GPU kernel must require its implemented physical backends");
            if (vulkan_only && (!vulkan_hardware || metal_hardware))
                return fail("FP64 retained flux requires a Vulkan fixture and no Metal fixture");
            if (!vulkan_only && !metal_hardware)
                return fail("active GPU kernel needs a Metal hardware fixture");
            if (present(entry.coverage.waiver) ||
                entry.coverage.replay_owner == ReplayOwner::Waived)
                return fail("active parity entry cannot use a coverage waiver");
            if ((entry.coverage.replay_owner == ReplayOwner::RuntimeCallSite &&
                 (!runtime || hardware)) ||
                (entry.coverage.replay_owner == ReplayOwner::HardwareFixture &&
                 (runtime || !hardware)) ||
                (entry.coverage.replay_owner == ReplayOwner::RuntimeAndHardware &&
                 (!runtime || !hardware)))
                return fail("parity replay owner disagrees with coverage paths");
        }
        else
        {
            if (entry.coverage.replay_owner != ReplayOwner::Waived ||
                !present(entry.coverage.waiver))
                return fail("inactive parity entry requires an explicit waiver");
        }
    }

    if (reason) *reason = nullptr;
    return true;
}

}  // namespace nimbus::parity
