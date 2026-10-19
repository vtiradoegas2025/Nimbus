#include "core/runtime/readiness.hpp"

namespace tmv::runtime
{

const char* readiness_state_name(ReadinessState state) noexcept
{
    switch (state)
    {
    case ReadinessState::EngineeringIncomplete:
        return "engineering_incomplete";
    case ReadinessState::Ready:
        return "ready";
    case ReadinessState::CampaignValidated:
        return "campaign_validated";
    case ReadinessState::ExternallyReviewed:
        return "externally_reviewed";
    case ReadinessState::FullyQualified:
        return "fully_qualified";
    }
    return "engineering_incomplete";
}

const std::vector<ReadinessItem>& tracked_readiness_items()
{
    using State = ReadinessState;
    static const std::vector<ReadinessItem> items = {
        {"terrain.earth_fixed_native",
         "terrain",
         "Earth-fixed Bell C-grid conservative motion on Vulkan and Metal with the maintained "
         "Kessler or canonical Lin with slab/simple-grey/full-stochastic coupling and constant frame components within "
         "+/-180 m/s",
         State::Ready,
         "tests/fixtures/configs/qualification/earth_fixed_terrain_motion_metal_ready.yaml; tests/fixtures/configs/qualification/earth_fixed_terrain_motion_vulkan_ready.yaml; tests/fixtures/configs/qualification/earth_fixed_terrain_motion_lin_metal_ready.yaml; tests/fixtures/configs/qualification/earth_fixed_terrain_motion_lin_vulkan_ready.yaml; scripts/tests/science/test_native_earth_fixed_ready.py; scripts/tests/science/test_earth_fixed_terrain_motion_ready.py; tests/core/test_native_earth_fixed_step.cpp; tests/dynamics/test_native_cgrid_boundary.cpp; tests/dynamics/test_native_moist_mapped_acoustic.cpp; tests/core/test_environment_column_refresh.cpp; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"advection.weno5_native_strict",
         "advection",
         "Exact flat C-grid WENO5/RRTMG reference with strict native scalar transport on Vulkan and Metal",
         State::Ready,
         "tests/fixtures/configs/qualification/cgrid_weno5_rrtmg_metal_ready.yaml; tests/fixtures/configs/qualification/cgrid_weno5_rrtmg_vulkan_ready.yaml; tests/numerics/test_advection_cartesian.cpp; scripts/tests/science/test_native_weno_ready.py; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"physics.native_remaining",
         "compute",
         "Broader native Thompson, Milbrandt, RK3/RK4, TKE and implicit-diffusion combinations",
         State::EngineeringIncomplete,
         "",
         "",
         "",
         "Exact native Thompson, Milbrandt, TKE/implicit and stationary periodic RK3/RK4 references are Ready; other "
         "retained operators and "
         "broader "
         "couplings still require native execution, public admission and bounded parity"},
        {"nesting.moist_two_way",
         "nesting",
         "Two-way nested feedback with active moist physics and moving domains",
         State::EngineeringIncomplete,
         "",
         "",
         "",
         "Flat stationary dry/Kessler/Lin, moving-child Kessler, tracked-window Kessler and stationary mapped "
         "Bell/Schar "
         "2:1/3:1 dry/Kessler/Lin reference families are Ready on CPU, Metal and Vulkan with TVD/MC or WENO. "
         "Integer moving-child mapped Kessler/Lin Bell/Schar 2:1/3:1 references and the exact seeded stationary "
         "Milbrandt and Thompson private-moment references are also Ready. Their maintained integer moving-child "
         "families and flat private parent-recycling references are Ready in their declared scopes. "
         "The unseeded Thompson Schar 2:1 references are Ready with source-consistent particle feedback. "
         "Thompson WENO parent recycling is Ready with precise native transport arithmetic. "
         "The maintained unseeded Thompson/Milbrandt stationary families are Ready across the same geometries. "
         "Mapped parent recycling, continuously moving parent/child terrain and additional private-state/physics "
         "combinations still require engineering, "
         "public admission and bounded validation"},
        {"turbulence.richardson_public",
         "turbulence",
         "Exact flat periodic C-grid CPU Lilly-equilibrium Smagorinsky/RRTMG profile with versioned Richardson "
         "selection",
         State::Ready,
         "tests/physics/test_richardson_smagorinsky.cpp; tests/core/test_simulation_checkpoint.cpp; "
         "scripts/tests/science/test_richardson_smagorinsky_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_richardson_smagorinsky_ready.yaml",
         "",
         "",
         ""},
        {"acceptance.complete_combination_matrix",
         "admission",
         "Generated retained physics/backend/output/initialization, boundary, compute, staggering, qualified-fixture, "
         "and public-preset admission matrices",
         State::Ready,
         "tests/core/test_scheme_catalog.cpp; scripts/tests/science/test_ready_admission_matrix.py; "
         "scripts/tests/config/test_public_preset_admission.py; tests/fixtures/configs/qualification; configs",
         "",
         "",
         ""},
        {"output.zfp_bounded_history",
         "output",
         "All public non-radar presets on flat/Bell/Schär plus exact flat-CPU selected radar product subsets and "
         "both named radar presets in qualified ZFP history, with unit-aware field budgets, bounded ownership, "
         "immutable terrain geometry, indexed random access, and Studio decode",
         State::EngineeringIncomplete,
         "tests/fixtures/configs/qualification/zfp_core_history.yaml; tests/core/test_zfp_roundtrip.cpp; scripts/qualification/output/qualify_zfp_core_history.py; ui/src-tauri/src/zfp3d.rs; docs/MATHEMATICS.pdf",
         "",
         "",
         "Exact flat-CPU reversible-v4 radar ZFP public/native/Studio bounded gates and the maintained ZFP "
         "qualifier passed on the prior exact d1a source/binary. Retained terrain/output combinations and whole-history "
         "readiness reconciliation remain open; long campaigns and independent scientific review are separate."},
        {"output.bounded_io_pipeline",
         "output",
         "Bounded flat/Bell/Schär NPY, every non-radar netCDF preset, non-radar qualified ZFP presets and planar "
         "fields, plus checkpoint, trajectory, geometry, SHM, native-window, cache, and "
         "incremental-analysis pipeline",
         State::Ready,
         "tests/core/test_output_writer.cpp; tests/core/test_simulation_checkpoint.cpp; tests/diagnostics/test_trajectories.cpp; scripts/qualification/output/qualify_live_shm.py; ui/src-tauri/src/commands/outputs.rs; tests/ui/unit/run/test_field_read_cancellation.ts; tests/ui/unit/comparison/test_comparison_series_cache.ts; tests/ui/unit/plot/test_plot_fidelity.ts; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"agrid.flat_cpu_core",
         "dynamics",
         "Flat Cartesian A-grid CPU family with split-explicit/SSPRK3/RK4 stationary integration plus a "
         "supported-boundary constant-velocity CPU family with f-plane/beta-plane Coriolis, every retained "
         "microphysics/SGS/PBL/chaos owner, and bounded SSPRK3/RK4 compositions, each with complete commit-level "
         "energy ownership",
         State::Ready,
         "tests/dynamics/test_cartesian_dynamics.cpp; tests/physics/test_turbulence_sgs.cpp; tests/physics/test_tke_scheme.cpp; tests/physics/test_boundary_layer.cpp; tests/physics/test_radiation.cpp; tests/physics/test_microphysics.cpp; tests/physics/test_chaos_runtime_state.cpp; tests/numerics/test_diffusion.cpp; tests/numerics/test_advection_cartesian.cpp; tests/numerics/test_time_stepping.cpp; scripts/tests/science/test_agrid_core_ready.py; scripts/tests/science/test_agrid_turbulence_family_ready.py; scripts/tests/science/test_agrid_column_family_ready.py; scripts/tests/science/test_agrid_milbrandt_ready.py; scripts/tests/science/test_agrid_lin_ready.py; scripts/tests/science/test_agrid_thompson_ready.py; scripts/tests/science/test_agrid_implicit_ready.py; scripts/tests/science/test_agrid_weno_ready.py; scripts/tests/science/test_agrid_rk_ready.py; scripts/tests/science/test_agrid_chaos_ready.py; scripts/qualification/domain_motion/qualify_constant_velocity_domain.py; tests/science/agrid/boundaries/qualify_agrid_boundary_matrix.py; tests/science/agrid/physics/qualify_agrid_kessler_coupling.py; tests/fixtures/configs/qualification/agrid_flat_cpu_core_ready.yaml; tests/fixtures/configs/qualification/agrid_kessler_radiation_coupled.yaml; tests/fixtures/configs/qualification/agrid_milbrandt_rrtmg_ready.yaml; tests/fixtures/configs/qualification/agrid_lin_rrtmg_ready.yaml; tests/fixtures/configs/qualification/agrid_thompson_rrtmg_ready.yaml; tests/fixtures/configs/qualification/agrid_thompson_rrtmg_implicit_ready.yaml; tests/fixtures/configs/qualification/agrid_thompson_rrtmg_implicit_weno_ready.yaml; tests/fixtures/configs/qualification/agrid_thompson_rrtmg_implicit_weno_ssprk3_ready.yaml; tests/fixtures/configs/qualification/agrid_thompson_rrtmg_implicit_weno_rk4_ready.yaml; tests/fixtures/configs/qualification/agrid_full_stochastic_ready.yaml; tests/fixtures/configs/qualification/constant_velocity_agrid_ready.yaml; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"agrid.complete_vulkan",
         "dynamics",
         "Exact flat dry rigid-boundary A-grid timestep on fallback-off Vulkan",
         State::Ready,
         "tests/vulkan/test_gpu_parity.cpp; scripts/tests/science/test_agrid_native_complete_ready.py; "
         "tests/fixtures/configs/qualification/agrid_flat_vulkan_complete_ready.yaml",
         "",
         "",
         ""},
        {"agrid.complete_metal",
         "dynamics",
         "Exact flat dry rigid-boundary A-grid timestep on fallback-off Metal",
         State::Ready,
         "tests/metal/test_metal_compute_backend.cpp; scripts/tests/science/test_agrid_native_complete_ready.py; "
         "tests/fixtures/configs/qualification/agrid_flat_metal_complete_ready.yaml",
         "",
         "",
         ""},
        {"terrain.stationary_cgrid_cpu",
         "terrain",
         "Structural stationary Bell mapped Cartesian C-grid CPU family representative with slab PBL and full implicit "
         "scalar/momentum diffusion",
         State::Ready,
         "tests/physics/test_terrain.cpp; tests/numerics/test_diffusion.cpp; "
         "scripts/tests/science/test_mapped_terrain_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_stationary_bell_pbl_implicit_ready.yaml",
         "",
         "",
         ""},
        {"terrain.coupled_cgrid_cpu",
         "terrain",
         "Structural stationary/constant-velocity Bell/Schär mapped C-grid CPU family with all retained microphysics "
         "and chaos schemes, 3-D Smagorinsky/TKE without PBL or slab/YSU/MYNN PBL plus horizontal Smagorinsky/TKE, "
         "auto/TVD/WENO transport, clear or water/cloud-absorbing simple-grey longwave/shortwave or 112-point RRTMG "
         "longwave/shortwave, and constant explicit or backward-Euler implicit scalar/momentum diffusion",
         State::Ready,
         "tests/physics/test_terrain.cpp; tests/physics/test_turbulence_sgs.cpp; tests/physics/test_tke_scheme.cpp; "
         "tests/physics/test_boundary_layer.cpp; tests/physics/test_boundary_layer_runtime_state.cpp; "
         "tests/physics/test_radiation.cpp; tests/physics/test_radiation_runtime_state.cpp; "
         "tests/numerics/test_diffusion.cpp; scripts/tests/science/test_mapped_rrtmg_shortwave_ready.py; "
         "tests/science/cgrid/terrain/qualify_cgrid_terrain_coupling.py; "
         "tests/fixtures/configs/qualification/cgrid_terrain_smagorinsky_coupled.yaml",
         "",
         "",
         ""},
        {"terrain.mapped_pbl_motion",
         "terrain",
         "Structural constant-velocity Schar mapped Cartesian C-grid CPU family representative with slab PBL and full "
         "implicit scalar/momentum diffusion",
         State::Ready,
         "tests/physics/test_terrain.cpp; tests/numerics/test_diffusion.cpp; "
         "scripts/tests/science/test_mapped_terrain_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_translating_schar_pbl_implicit_ready.yaml",
         "",
         "",
         ""},
        {"terrain.complete_vulkan",
         "terrain",
         "Exact dry stationary Bell and Schar mapped Cartesian C-grid Heun timesteps on fallback-off Vulkan",
         State::Ready,
         "tests/vulkan/test_mapped_mass_continuity.cpp; scripts/tests/science/test_native_mapped_terrain_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_stationary_bell_vulkan_complete_ready.yaml; "
         "tests/fixtures/configs/qualification/cgrid_stationary_schar_vulkan_complete_ready.yaml",
         "",
         "",
         ""},
        {"terrain.complete_metal",
         "terrain",
         "Exact dry stationary Bell and Schar mapped Cartesian C-grid Heun timesteps on fallback-off Metal",
         State::Ready,
         "tests/metal/test_metal_compute_backend.cpp; scripts/tests/science/test_native_mapped_terrain_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_stationary_bell_metal_complete_ready.yaml; "
         "tests/fixtures/configs/qualification/cgrid_stationary_schar_metal_complete_ready.yaml",
         "",
         "",
         ""},
        {"terrain.collocated",
         "terrain",
         "Mapped Bell/Schär Cartesian A-grid CPU family with dry Bell motion and stationary all-microphysics/all-chaos "
         "active ownership across none/TVD/WENO5 advection",
         State::Ready,
         "tests/fixtures/configs/qualification/agrid_stationary_bell_cpu_ready.yaml; tests/fixtures/configs/qualification/agrid_mapped_full_stochastic_none_ready.yaml; tests/fixtures/configs/qualification/agrid_mapped_full_stochastic_kessler_ready.yaml; tests/fixtures/configs/qualification/agrid_mapped_full_stochastic_lin_ready.yaml; tests/fixtures/configs/qualification/agrid_mapped_full_stochastic_ready.yaml; tests/fixtures/configs/qualification/agrid_mapped_full_stochastic_milbrandt_ready.yaml; tests/fixtures/configs/qualification/agrid_mapped_full_stochastic_schar_none_ready.yaml; tests/fixtures/configs/qualification/agrid_mapped_full_stochastic_schar_kessler_ready.yaml; tests/fixtures/configs/qualification/agrid_mapped_full_stochastic_schar_lin_ready.yaml; tests/fixtures/configs/qualification/agrid_mapped_full_stochastic_schar_thompson_ready.yaml; tests/fixtures/configs/qualification/agrid_mapped_full_stochastic_schar_milbrandt_ready.yaml; scripts/tests/science/test_collocated_mapped_terrain_ready.py; tests/dynamics/test_collocated_mapped_timestep.cpp; tests/physics/test_microphysics.cpp; tests/physics/test_chaos_runtime_state.cpp; tests/physics/test_terrain.cpp; scripts/tests/science/test_thompson_stochastic_ready.py; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"terrain.earth_fixed_remap",
         "terrain",
         "Exact tracked-window Earth-fixed Bell geometry/remap transaction on CPU",
         State::Ready,
         "tests/fixtures/configs/qualification/earth_fixed_terrain_recycling_ready.yaml; "
         "scripts/tests/science/test_earth_fixed_terrain_recycling_ready.py; tests/physics/test_terrain.cpp; "
         "tests/core/test_simulation_checkpoint.cpp; tests/core/test_domain_recycling.cpp",
         "",
         "",
         ""},
        {"terrain.earth_fixed_continuous_cpu",
         "terrain",
         "Earth-fixed Bell C-grid CPU conservative coordinate remapping with the maintained "
         "Kessler or canonical Lin with slab/simple-grey/full-stochastic coupling and constant frame components within "
         "+/-180 m/s",
         State::Ready,
         "tests/fixtures/configs/qualification/earth_fixed_terrain_motion_ready.yaml; "
         "tests/fixtures/configs/qualification/earth_fixed_terrain_motion_lin_cpu_ready.yaml; "
         "tests/fixtures/configs/qualification/earth_fixed_terrain_motion_diagnostics_ready.yaml; "
         "scripts/tests/science/test_earth_fixed_terrain_motion_ready.py; "
         "tests/core/test_earth_fixed_remap_reference.cpp; tests/core/test_earth_fixed_step_transaction.cpp; "
         "tests/core/test_simulation_checkpoint.cpp",
         "",
         "",
         ""},
        {"recycling.flat_cpu",
         "domain_motion",
         "Flat homogeneous whole-cell D5 recycling on CPU",
         State::Ready,
         "tests/fixtures/configs/qualification/domain_recycling_kessler_ready.yaml; "
         "scripts/qualification/domain_motion/qualify_domain_recycling.py; tests/core/test_domain_recycling.cpp",
         "",
         "",
         ""},
        {"recycling.flat_vulkan",
         "domain_motion",
         "Flat homogeneous whole-cell D5 recycling on Vulkan",
         State::Ready,
         "tests/fixtures/configs/qualification/domain_recycling_kessler_ready.yaml; "
         "scripts/qualification/domain_motion/qualify_domain_recycling.py; tests/vulkan/test_gpu_parity.cpp",
         "",
         "",
         ""},
        {"recycling.flat_metal",
         "domain_motion",
         "Flat homogeneous whole-cell D5 recycling on Metal",
         State::Ready,
         "tests/fixtures/configs/qualification/domain_recycling_kessler_ready.yaml; "
         "scripts/qualification/domain_motion/qualify_domain_recycling.py; tests/metal/test_metal_compute_backend.cpp",
         "",
         "",
         ""},
        {"recycling.nested_grid",
         "domain_motion",
         "Exact flat one-way and moist two-way nested-grid D5 recycling profiles on CPU",
         State::Ready,
         "tests/fixtures/configs/qualification/nested_grid_recycling_ready.yaml; tests/fixtures/configs/qualification/nested_grid_recycling_two_way_cpu_ready.yaml; scripts/tests/science/test_nested_grid_recycling_ready.py; tests/core/test_domain_recycling.cpp; tests/core/test_simulation_checkpoint.cpp; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"recycling.terrain",
         "domain_motion",
         "Exact tracked-window D5 recycling with Earth-fixed Bell terrain on CPU",
         State::Ready,
         "tests/fixtures/configs/qualification/earth_fixed_terrain_recycling_ready.yaml; "
         "scripts/tests/science/test_earth_fixed_terrain_recycling_ready.py; tests/core/test_domain_recycling.cpp; "
         "tests/core/test_simulation_checkpoint.cpp",
         "",
         "",
         ""},
        {"recycling.mapped_two_way",
         "domain_motion",
         "Exact Earth-fixed Bell/Schar 2:1/3:1 fixed-footprint Kessler/slab/simple-grey/full-stochastic two-way parent recycling on CPU/Metal/Vulkan",
         State::Ready,
         "tests/fixtures/configs/qualification/mapped_nested_recycling_kessler_bell2_tvd_cpu_ready.yaml; tests/fixtures/configs/qualification/mapped_nested_recycling_kessler_bell2_tvd_metal_ready.yaml; tests/fixtures/configs/qualification/mapped_nested_recycling_kessler_bell2_tvd_vulkan_ready.yaml; tests/fixtures/configs/qualification/mapped_nested_recycling_kessler_bell2_tvd_native_slow_metal_ready.yaml; tests/fixtures/configs/qualification/mapped_nested_recycling_kessler_bell2_tvd_native_slow_vulkan_ready.yaml; tests/fixtures/configs/qualification/mapped_nested_recycling_kessler_bell3_tvd_cpu_ready.yaml; tests/fixtures/configs/qualification/mapped_nested_recycling_kessler_schar2_tvd_cpu_ready.yaml; tests/fixtures/configs/qualification/mapped_nested_recycling_kessler_schar3_tvd_cpu_ready.yaml; scripts/tests/science/test_mapped_nested_recycling_ready.py; tests/compute/test_native_domain_recycling.cpp; tests/core/test_nested_terrain_geometry.cpp; tests/core/test_domain_recycling.cpp; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"domain_motion.tracked_window",
         "domain_motion",
         "Exact flat CPU tracked-window recycling profile",
         State::Ready,
         "tests/fixtures/configs/qualification/tracked_window_kessler_ready.yaml; "
         "scripts/tests/science/test_tracked_window_ready.py; "
         "scripts/qualification/domain_motion/qualify_domain_recycling.py; tests/core/test_domain_recycling.cpp",
         "",
         "",
         ""},
        {"nesting.stationary_cpu",
         "nesting",
         "Stationary one-way parent and child execution on CPU",
         State::Ready,
         "tests/core/test_nested_production_step.cpp",
         "",
         "",
         ""},
        {"nesting.stationary_vulkan",
         "nesting",
         "Stationary one-way parent and child execution on Vulkan",
         State::Ready,
         "tests/core/test_nested_production_step.cpp; scripts/tests/gpu/test_stationary_nested_vulkan_admission.py; "
         "scripts/qualification/nested_domains/qualify_stationary_nested_vulkan_runtime.py; "
         "tests/fixtures/configs/qualification/stationary_nested_vulkan_dry.yaml",
         "",
         "",
         ""},
        {"diagnostics.resolved_cartesian",
         "diagnostics",
         "Shared physical-grid pressure potentials, resolved angular momentum and vorticity for maintained CPU "
         "A/C-grid and native C-grid output profiles",
         State::Ready,
         "tests/diagnostics/test_pressure_projection.cpp; tests/diagnostics/test_resolved_diagnostics.cpp; scripts/qualification/output/qualify_live_shm.py; tests/fixtures/configs/qualification/agrid_mapped_none_bell_lin_ready.yaml; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"nesting.two_way_moving_mapped_native",
         "nesting",
         "Integer moving-child mapped Bell/Schar 2:1/3:1 Kessler/Lin TVD/MC and WENO two-way "
         "CPU/Metal/Vulkan reference families",
         State::Ready,
         "tests/core/test_nested_production_step.cpp; tests/core/test_nested_terrain_geometry.cpp; tests/core/test_simulation_checkpoint.cpp; scripts/tests/science/test_native_nested_motion_ready.py; scripts/tests/science/test_nested_terrain_checkpoint_compatibility.py; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"nesting.thompson_two_way_moving",
         "nesting",
         "Cloud-seeded Thompson integer moving-child C-grid references: flat 2:1 and Bell/Schar 2:1/3:1, TVD/MC or WENO, CPU/Metal/Vulkan",
         State::Ready,
         "scripts/tests/science/test_private_nested_motion_matrix.py; tests/core/test_nested_production_step.cpp; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"nesting.milbrandt_two_way_moving",
         "nesting",
         "Cloud-seeded Milbrandt integer moving-child references with either hail setting: flat 2:1 and Bell/Schar 2:1/3:1, TVD/MC or WENO, CPU/Metal/Vulkan",
         State::Ready,
         "scripts/tests/science/test_private_nested_motion_matrix.py; tests/core/test_nested_production_step.cpp; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"nesting.private_parent_recycling",
         "nesting",
         "Flat tracked-parent private two-way references: Thompson TVD/WENO and Milbrandt TVD/WENO with either hail setting on CPU/Metal/Vulkan",
         State::Ready,
         "scripts/tests/science/test_private_nested_motion_matrix.py; tests/core/test_domain_recycling.cpp; tests/core/test_simulation_checkpoint.cpp; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"nesting.thompson_two_way_stationary",
         "nesting",
         "Cloud-seeded and unseeded stationary Thompson, flat 2:1 and Bell/Schar 2:1/3:1, "
         "TVD/MC or WENO on CPU/Metal/Vulkan",
         State::Ready,
         "scripts/tests/science/test_thompson_two_way_ready.py; tests/core/test_nested_production_step.cpp; tests/core/test_nested_scalar_feedback.cpp; tests/core/test_simulation_checkpoint.cpp; tests/physics/test_microphysics.cpp; scripts/tests/science/test_unseeded_private_two_way_ready.py; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"nesting.milbrandt_two_way_stationary",
         "nesting",
         "Cloud-seeded and unseeded stationary Milbrandt with either hail setting, flat 2:1 and Bell/Schar 2:1/3:1, "
         "TVD/MC or WENO on CPU/Metal/Vulkan",
         State::Ready,
         "scripts/tests/science/test_native_two_way_nested_ready.py; tests/core/test_nested_production_step.cpp; tests/core/test_nested_scalar_feedback.cpp; scripts/tests/science/test_unseeded_private_two_way_ready.py; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"nesting.two_way_mapped_native",
         "nesting",
         "Stationary mapped Bell/Schar 2:1/3:1 dry/Kessler/Lin TVD/MC and WENO two-way CPU/Metal/Vulkan reference "
         "families",
         State::Ready,
         "scripts/tests/science/test_native_two_way_nested_ready.py; tests/numerics/test_native_conservative_momentum.cpp; tests/dynamics/test_mapped_reconstruction.cpp; tests/dynamics/test_native_moist_mapped_acoustic.cpp; tests/fixtures/configs/qualification/stationary_mapped_nested_kessler_bell_tvd_metal_ready.yaml; tests/fixtures/configs/qualification/stationary_mapped_nested_lin_schar3_weno_vulkan_ready.yaml; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"nesting.two_way_stationary_native",
         "nesting",
         "Stationary flat dry/Kessler/Lin TVD/MC and WENO two-way CPU/Metal/Vulkan reference families",
         State::Ready,
         "scripts/tests/science/test_native_two_way_nested_ready.py; tests/numerics/test_native_conservative_momentum.cpp; tests/numerics/test_native_phase_feedback.cpp; tests/numerics/test_native_scalar_capture.cpp; tests/core/test_simulation_checkpoint.cpp; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"nesting.two_way_moving_native",
         "nesting",
         "Flat moving-child Kessler/Smagorinsky/slab/full-stochastic TVD/MC and WENO two-way CPU/Metal/Vulkan "
         "references",
         State::Ready,
         "scripts/tests/science/test_native_nested_motion_ready.py; tests/numerics/test_native_conservative_momentum.cpp; tests/numerics/test_native_phase_feedback.cpp; tests/numerics/test_native_scalar_capture.cpp; tests/core/test_simulation_checkpoint.cpp; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"nesting.two_way_recycling_native",
         "nesting",
         "Flat tracked-window Kessler/Smagorinsky/slab/full-stochastic TVD/MC and WENO two-way CPU/Metal/Vulkan "
         "references",
         State::Ready,
         "scripts/tests/science/test_native_nested_motion_ready.py; tests/numerics/test_native_conservative_momentum.cpp; tests/numerics/test_native_phase_feedback.cpp; tests/numerics/test_native_scalar_capture.cpp; tests/core/test_simulation_checkpoint.cpp; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"nesting.two_way_moving_cpu",
         "nesting",
         "Exact flat moving-child Kessler/Smagorinsky/slab two-way CPU reference",
         State::Ready,
         "tests/fixtures/configs/qualification/moving_nested_two_way_cpu_ready.yaml; scripts/tests/science/test_moving_two_way_nested_ready.py; tests/core/test_nested_production_step.cpp; tests/diagnostics/test_energy_budget.cpp; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"nesting.moving_cpu_public",
         "nesting",
         "Ordinary public moving-nest admission on CPU",
         State::Ready,
         "scripts/tests/gpu/test_moving_nested_vulkan_admission.py; tests/fixtures/configs/qualification/moving_nested_supercell_cpu_v1.yaml; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"nesting.moving_vulkan",
         "nesting",
         "Exact public moving-supercell profile on Vulkan",
         State::Ready,
         "tests/core/test_nested_production_step.cpp; scripts/tests/gpu/test_moving_nested_vulkan_admission.py; "
         "tests/fixtures/configs/qualification/moving_nested_supercell_vulkan_v1.yaml",
         "",
         "",
         ""},
        {"nesting.moving_metal",
         "nesting",
         "Exact public moving-supercell profile on Metal",
         State::Ready,
         "tests/fixtures/configs/qualification/moving_nested_supercell_metal_v1.yaml; "
         "scripts/tests/gpu/test_moving_nested_vulkan_admission.py; tests/metal/test_metal_runtime.py",
         "",
         "",
         ""},
        {"nesting.two_way_feedback",
         "nesting",
         "Stationary dry/Kessler/Lin Cartesian C-grid two-way feedback on CPU for flat 2:1 and mapped Bell/Schär "
         "2:1/3:1",
         State::Ready,
         "scripts/tests/science/test_moist_two_way_nested_ready.py; tests/core/test_nested_scalar_feedback.cpp; scripts/tests/science/test_two_way_nested_ready.py; scripts/tests/science/test_active_terrain_nested_ready.py; tests/fixtures/configs/qualification/stationary_nested_cpu_two_way_ready.yaml; tests/fixtures/configs/qualification/stationary_nested_active_terrain_cpu_two_way_ready.yaml; tests/fixtures/configs/qualification/stationary_nested_active_schar_cpu_two_way_ready.yaml; tests/fixtures/configs/qualification/stationary_nested_active_bell_cpu_two_way_3to1_ready.yaml; tests/fixtures/configs/qualification/stationary_nested_active_schar_cpu_two_way_3to1_ready.yaml; tests/core/test_nested_grid.cpp; tests/core/test_nested_boundary_forcing.cpp; tests/core/test_nested_terrain_geometry.cpp; tests/core/test_simulation_checkpoint.cpp; tests/numerics/test_advection_cartesian.cpp; tests/dynamics/test_cartesian_cgrid_slow_tendencies_gpu_parity.cpp; tests/core/test_nested_production_step.cpp; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"nesting.active_terrain",
         "nesting",
         "Stationary isotropic 2:1/3:1 Bell/Schär mapped one-/two-way nesting on CPU",
         State::Ready,
         "tests/fixtures/configs/qualification/stationary_nested_active_bell_cpu_one_way_ready.yaml; "
         "tests/fixtures/configs/qualification/stationary_nested_active_terrain_cpu_two_way_ready.yaml; "
         "tests/fixtures/configs/qualification/stationary_nested_active_schar_cpu_one_way_ready.yaml; "
         "tests/fixtures/configs/qualification/stationary_nested_active_schar_cpu_two_way_ready.yaml; "
         "tests/fixtures/configs/qualification/stationary_nested_active_bell_cpu_one_way_3to1_ready.yaml; "
         "tests/fixtures/configs/qualification/stationary_nested_active_bell_cpu_two_way_3to1_ready.yaml; "
         "tests/fixtures/configs/qualification/stationary_nested_active_schar_cpu_one_way_3to1_ready.yaml; "
         "tests/fixtures/configs/qualification/stationary_nested_active_schar_cpu_two_way_3to1_ready.yaml; "
         "scripts/tests/science/test_active_terrain_nested_ready.py; tests/core/test_nested_terrain_geometry.cpp; "
         "tests/core/test_nested_grid.cpp; tests/core/test_nested_level_runtime.cpp; "
         "tests/core/test_simulation_checkpoint.cpp",
         "",
         "",
         ""},
        {"diffusion.variable_k_vulkan",
         "diffusion",
         "Exact flat C-grid explicit linear-height vertical diffusion on Vulkan",
         State::Ready,
         "tests/numerics/test_diffusion.cpp; tests/vulkan/test_gpu_parity.cpp; "
         "scripts/tests/science/test_variable_diffusion_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_variable_diffusion_vulkan_ready.yaml; "
         "vulkan/shaders/compute/diffusion/diffusion.comp",
         "",
         "",
         ""},
        {"diffusion.variable_k_metal",
         "diffusion",
         "Exact flat C-grid explicit linear-height vertical diffusion on Metal",
         State::Ready,
         "tests/numerics/test_diffusion.cpp; tests/metal/test_metal_compute_backend.cpp; "
         "scripts/tests/science/test_variable_diffusion_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_variable_diffusion_metal_ready.yaml; "
         "metal/include/compute/metal/shaders/diffusion/diffusion_msl.hpp",
         "",
         "",
         ""},
        {"time_stepping.rk3_coupled",
         "time_stepping",
         "Flat CPU SSPRK3 families and stationary periodic native C-grid/collocated integration with staged "
         "public/private fields, "
         "constant explicit/implicit diffusion and admitted physics compositions",
         State::Ready,
         "tests/numerics/test_time_stepping.cpp; tests/numerics/test_native_rk_combination.cpp; "
         "tests/numerics/test_native_rk_tendency.cpp; scripts/tests/science/test_coupled_rk_accelerator_ready.py; "
         "scripts/tests/science/test_coupled_rk_ready.py; "
         "scripts/tests/science/test_agrid_rk_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_ssprk3_coupled_ready.yaml; "
         "tests/fixtures/configs/qualification/agrid_thompson_rrtmg_implicit_weno_ssprk3_ready.yaml",
         "",
         "",
         ""},
        {"time_stepping.rk4_coupled",
         "time_stepping",
         "Flat CPU RK4 families and stationary periodic native C-grid/collocated integration with staged "
         "public/private fields, "
         "constant explicit/implicit diffusion and admitted physics compositions",
         State::Ready,
         "tests/numerics/test_time_stepping.cpp; tests/numerics/test_native_rk_combination.cpp; tests/numerics/test_native_rk_tendency.cpp; scripts/tests/science/test_coupled_rk_accelerator_ready.py; scripts/tests/science/test_coupled_rk_ready.py; scripts/tests/science/test_agrid_rk_ready.py; tests/fixtures/configs/qualification/cgrid_rk4_coupled_ready.yaml; tests/fixtures/configs/qualification/agrid_thompson_rrtmg_implicit_weno_rk4_ready.yaml; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"radiation.rrtmg",
         "radiation",
         "RRTMG coupled public profiles",
         State::Ready,
         "tests/physics/test_radiation.cpp; tests/physics/test_radiation_runtime_state.cpp; "
         "tests/core/test_simulation_checkpoint.cpp",
         "",
         "",
         ""},
        {"radiation.simple_grey_general",
         "radiation",
         "Exact CPU simple-grey pure-absorption water/cloud-path profile",
         State::Ready,
         "tests/physics/test_radiation.cpp; tests/physics/test_radiation_runtime_state.cpp; "
         "tests/core/test_simulation_checkpoint.cpp; "
         "scripts/tests/science/test_simple_grey_absorbing_water_cloud_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_simple_grey_absorbing_water_cloud_cpu_ready.yaml; "
         "tests/fixtures/configs/qualification/soundings/simple_grey_condensing_profile.txt",
         "",
         "",
         ""},
        {"boundary_layer.mynn",
         "boundary_layer",
         "MYNN coupled public profiles",
         State::Ready,
         "tests/physics/test_boundary_layer.cpp; tests/physics/test_boundary_layer_runtime_state.cpp; "
         "tests/core/test_simulation_checkpoint.cpp",
         "",
         "",
         ""},
        {"boundary_layer.slab",
         "boundary_layer",
         "Slab PBL public profiles",
         State::Ready,
         "tests/physics/test_boundary_layer.cpp; tests/physics/test_boundary_layer_runtime_state.cpp; "
         "tests/core/test_simulation_checkpoint.cpp",
         "",
         "",
         ""},
        {"boundary_layer.ysu",
         "boundary_layer",
         "YSU public profiles",
         State::Ready,
         "tests/physics/test_boundary_layer.cpp; tests/physics/test_boundary_layer_runtime_state.cpp; "
         "tests/fixtures/configs/qualification/cgrid_ysu_rrtmg_ready.yaml",
         "",
         "",
         ""},
        {"boundary_layer.pbl_top_theta",
         "boundary_layer",
         "Exact YSU potential-temperature-increase PBL-top diagnosis on CPU",
         State::Ready,
         "include/boundary_layer/base/pbl_height.hpp; tests/physics/test_boundary_layer.cpp; "
         "scripts/tests/science/test_pbl_top_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_ysu_theta_top_ready.yaml",
         "",
         "",
         ""},
        {"boundary_layer.pbl_top_tke",
         "boundary_layer",
         "Exact MYNN prognostic-TKE PBL-top diagnosis on CPU",
         State::Ready,
         "include/boundary_layer/base/pbl_height.hpp; tests/physics/test_boundary_layer.cpp; "
         "scripts/tests/science/test_pbl_top_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_mynn_tke_top_ready.yaml",
         "",
         "",
         ""},
        {"boundary_layer.cloud_top_radiative_ysu",
         "boundary_layer",
         "Exact CPU surface-coupled Lock-2000 radiative YSU with linearized theta_l and qv/qc/qi transport",
         State::Ready,
         "include/boundary_layer/base/cloud_top_radiative_mixing.hpp; tests/physics/test_boundary_layer.cpp; "
         "tests/physics/test_boundary_layer_runtime_state.cpp; tests/physics/test_microphysics_runtime_state.cpp; "
         "tests/core/test_simulation_checkpoint.cpp; scripts/tests/science/test_cloud_top_radiative_ysu_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_lin_rrtmg_cloud_top_ysu_ready.yaml",
         "",
         "",
         ""},
        {"turbulence.tke",
         "turbulence",
         "Prognostic 1.5-order TKE closure",
         State::Ready,
         "tests/physics/test_tke_scheme.cpp; scripts/tests/science/test_tke_rrtmg_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_tke_rrtmg_ready.yaml",
         "",
         "",
         ""},
        {"physics.tke_implicit_native",
         "compute",
         "Exact flat periodic C-grid TKE/implicit scalar-and-momentum diffusion/RRTMG reference on CPU, Vulkan and "
         "Metal",
         State::Ready,
         "tests/fixtures/configs/qualification/cgrid_tke_implicit_rrtmg_cpu_ready.yaml; tests/fixtures/configs/qualification/cgrid_tke_implicit_rrtmg_metal_ready.yaml; tests/fixtures/configs/qualification/cgrid_tke_implicit_rrtmg_vulkan_ready.yaml; tests/numerics/test_diffusion.cpp; scripts/tests/science/test_native_tke_ready.py; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"turbulence.tke_native",
         "turbulence",
         "Exact flat periodic C-grid TKE/RRTMG profile with native closure, energy transport, SGS commit and strict "
         "slow commit on Vulkan and Metal",
         State::Ready,
         "tests/physics/test_native_tke.cpp; scripts/tests/science/test_native_tke_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_tke_rrtmg_metal_ready.yaml; "
         "tests/fixtures/configs/qualification/cgrid_tke_rrtmg_vulkan_ready.yaml",
         "",
         "",
         ""},
        {"turbulence.user_filter_width",
         "turbulence",
         "Exact 500 m user-prescribed Smagorinsky filter width on CPU",
         State::Ready,
         "tests/physics/test_turbulence_sgs.cpp; tests/physics/test_turbulence_runtime_state.cpp; "
         "tests/core/test_simulation_checkpoint.cpp; scripts/tests/science/test_user_filter_width_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_user_filter_width_ready.yaml",
         "",
         "",
         ""},
        {"turbulence.dynamic_smagorinsky",
         "turbulence",
         "Exact flat periodic CPU Favre Germano-Lilly dynamic Smagorinsky closure",
         State::Ready,
         "src/turbulence/schemes/smagorinsky/dynamic_smagorinsky.cpp; tests/physics/test_turbulence_sgs.cpp; "
         "tests/physics/test_turbulence_runtime_state.cpp; tests/core/test_simulation_checkpoint.cpp; "
         "scripts/tests/science/test_dynamic_smagorinsky_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_dynamic_smagorinsky_ready.yaml",
         "",
         "",
         ""},
        {"turbulence.vertical_only_sgs",
         "turbulence",
         "Static vertical-only Smagorinsky on the exact flat periodic C-grid CPU reference, with no active PBL overlap",
         State::EngineeringIncomplete,
         "tests/physics/test_turbulence_sgs.cpp; tests/physics/test_sgs_apply_runtime_state.cpp; docs/MATHEMATICS.pdf",
         "",
         "",
         "The exact flat-CPU static vertical-only candidate has ordinary qualified admission and a whole-step "
         "state/six-ledger transaction; final same-binary public failure, numeric/restart and Studio gates remain "
         "under review. Broader terrain/accelerator modes and external scientific review remain open; dynamic "
         "Smagorinsky requires 3-D mode."},
        {"turbulence.tke_stability_correction",
         "turbulence",
         "Exact flat periodic CPU Deardorff TKE stability correction",
         State::Ready,
         "tests/physics/test_tke_scheme.cpp; tests/physics/test_turbulence_runtime_state.cpp; "
         "tests/core/test_simulation_checkpoint.cpp; scripts/tests/science/test_tke_rrtmg_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_tke_rrtmg_ready.yaml",
         "",
         "",
         ""},
        {"advection.weno5",
         "advection",
         "WENO5 transported-scalar profiles on flat C-grid, mapped C-grid, flat collocated CPU storage, and mapped "
         "Bell/Schär collocated CPU storage",
         State::Ready,
         "tests/numerics/test_weno5_reconstruction.cpp; tests/numerics/test_advection_cartesian.cpp; "
         "scripts/tests/science/test_cgrid_numerics_ready.py; scripts/tests/science/test_agrid_weno_ready.py; "
         "scripts/tests/science/test_collocated_mapped_terrain_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_weno5_rrtmg_ready.yaml; "
         "tests/fixtures/configs/qualification/agrid_thompson_rrtmg_implicit_weno_ready.yaml; "
         "tests/fixtures/configs/qualification/agrid_mapped_full_stochastic_schar_milbrandt_ready.yaml",
         "",
         "",
         ""},
        {"diffusion.implicit",
         "diffusion",
         "Constant-coefficient CPU backward-Euler scalar/momentum diffusion on flat C-grid, mapped C-grid, and flat "
         "collocated storage",
         State::Ready,
         "tests/numerics/test_diffusion.cpp; scripts/tests/science/test_cgrid_numerics_ready.py; "
         "scripts/tests/science/test_agrid_implicit_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_implicit_rrtmg_ready.yaml; "
         "tests/fixtures/configs/qualification/agrid_thompson_rrtmg_implicit_ready.yaml",
         "",
         "",
         ""},
        {"diffusion.implicit_native",
         "diffusion",
         "Exact flat C-grid backward-Euler scalar and momentum diffusion with RRTMG on Vulkan and Metal",
         State::Ready,
         "tests/numerics/test_native_implicit.cpp; scripts/tests/science/test_native_implicit_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_implicit_rrtmg_metal_ready.yaml; "
         "tests/fixtures/configs/qualification/cgrid_implicit_rrtmg_vulkan_ready.yaml",
         "",
         "",
         ""},
        {"microphysics.lin",
         "microphysics",
         "Exact fallback-off CPU Lin/YSU/cloudy-RRTMG profile",
         State::Ready,
         "tests/qualification/test_lin_reference_processes.cpp; tests/physics/test_microphysics.cpp; "
         "scripts/tests/science/test_lin_rrtmg_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_lin_rrtmg_ready.yaml",
         "",
         "",
         ""},
        {"microphysics.lin_vulkan",
         "microphysics",
         "Exact fallback-off Vulkan Lin/YSU/cloudy-RRTMG profile",
         State::Ready,
         "tests/vulkan/test_gpu_parity.cpp; tests/physics/test_microphysics.cpp; "
         "scripts/tests/science/test_lin_rrtmg_accelerator_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_lin_rrtmg_vulkan_ready.yaml",
         "",
         "",
         ""},
        {"microphysics.lin_metal",
         "microphysics",
         "Exact fallback-off Metal Lin/YSU/cloudy-RRTMG profile",
         State::Ready,
         "tests/metal/test_metal_compute_backend.cpp; tests/physics/test_microphysics.cpp; "
         "scripts/tests/science/test_lin_rrtmg_accelerator_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_lin_rrtmg_metal_ready.yaml",
         "",
         "",
         ""},
        {"physics.flat_lin_storm",
         "physics",
         "Flat stationary radiative C-grid file-sounding Lin/TVD/Smagorinsky family on CPU, Metal and Vulkan",
         State::Ready,
         "scripts/qualification/accelerators/qualify_flat_lin_storm.py; tests/init/config/test_v1_loader.cpp; tests/qualification/test_lin_reference_processes.cpp; docs/MATHEMATICS.pdf",
         "", "", ""},
        {"compute.strict_acoustic_residency",
         "compute",
         "Native acoustic-loop retention on FP64-capable Vulkan devices, with exact flux/history publication and complete host report ownership",
         State::Ready,
         "tests/vulkan/test_acoustic_input_reuse.cpp; tests/dynamics/test_native_acoustic_boundary.cpp; scripts/qualification/accelerators/qualify_flat_lin_storm.py; docs/MATHEMATICS.pdf",
         "", "", ""},
        {"compute.nvidia_cuda_assist",
         "compute",
         "Optional retained-flux CUDA execution in Vulkan command buffers on compatible NVIDIA drivers with buffer device addresses",
         State::Ready,
         "tests/vulkan/test_acoustic_input_reuse.cpp; tests/init/config/test_configuration_compiler.cpp; scripts/qualification/accelerators/qualify_flat_lin_storm.py; docs/MATHEMATICS.pdf",
         "", "", ""},
        {"physics.thompson_ysu_weno_native",
         "compute",
         "Exact C-grid WENO5/Thompson/YSU/clear-sky-RRTMG references with and without implicit diffusion on CPU, "
         "Vulkan and Metal",
         State::Ready,
         "scripts/tests/science/test_thompson_weno_ready.py; scripts/tests/science/test_thompson_accelerator_ready.py; tests/numerics/test_advection_cartesian.cpp; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"physics.tke_weno_native",
         "compute",
         "Exact seeded C-grid WENO5/TKE/RRTMG references without microphysics, with and without implicit diffusion, on "
         "CPU, Vulkan and Metal",
         State::Ready,
         "scripts/tests/science/test_tke_weno_ready.py; scripts/tests/science/test_native_tke_ready.py; tests/numerics/test_advection_cartesian.cpp; tests/physics/test_native_tke.cpp; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"physics.thompson_tke_weno_native",
         "compute",
         "Exact seeded C-grid WENO5/Thompson/TKE/clear-sky-RRTMG references with and without implicit diffusion on "
         "CPU, Vulkan and Metal",
         State::Ready,
         "scripts/tests/science/test_thompson_tke_weno_ready.py; scripts/tests/science/test_native_tke_ready.py; tests/numerics/test_advection_cartesian.cpp; tests/metal/test_metal_compute_backend.cpp; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"physics.thompson_tke_native",
         "compute",
         "Exact seeded C-grid Thompson/TKE/clear-sky-RRTMG references, with and without implicit diffusion, on CPU, "
         "Vulkan and Metal",
         State::Ready,
         "scripts/tests/science/test_thompson_tke_ready.py; scripts/tests/science/test_native_tke_ready.py; tests/fixtures/configs/qualification/cgrid_thompson_tke_cpu_ready.yaml; tests/fixtures/configs/qualification/cgrid_thompson_tke_metal_ready.yaml; tests/fixtures/configs/qualification/cgrid_thompson_tke_vulkan_ready.yaml; tests/fixtures/configs/qualification/cgrid_thompson_tke_implicit_cpu_ready.yaml; tests/fixtures/configs/qualification/cgrid_thompson_tke_implicit_metal_ready.yaml; tests/fixtures/configs/qualification/cgrid_thompson_tke_implicit_vulkan_ready.yaml; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"physics.thompson_implicit_native",
         "compute",
         "Exact flat periodic C-grid Thompson/implicit theta-vapor-momentum diffusion/YSU/clear-sky-RRTMG reference on "
         "CPU, Vulkan and Metal",
         State::Ready,
         "tests/fixtures/configs/qualification/cgrid_thompson_implicit_cpu_ready.yaml; tests/fixtures/configs/qualification/cgrid_thompson_implicit_metal_ready.yaml; tests/fixtures/configs/qualification/cgrid_thompson_implicit_vulkan_ready.yaml; tests/physics/test_native_thompson_implicit.cpp; scripts/tests/science/test_thompson_implicit_ready.py; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"microphysics.milbrandt_native",
         "microphysics",
         "Default Milbrandt-Yau mixed-phase C-grid/YSU/clear-sky-RRTMG reference on CPU, Vulkan and Metal; both "
         "hail-production settings",
         State::Ready,
         "tests/fixtures/configs/qualification/cgrid_milbrandt_cpu_ready.yaml; tests/fixtures/configs/qualification/cgrid_milbrandt_vulkan_ready.yaml; tests/fixtures/configs/qualification/cgrid_milbrandt_metal_ready.yaml; scripts/tests/science/test_milbrandt_accelerator_ready.py; tests/physics/test_native_milbrandt_scheme.cpp; tests/physics/test_native_milbrandt_cold.cpp; tests/physics/test_native_milbrandt_sedimentation.cpp; tests/physics/test_native_milbrandt_cloud.cpp; tests/physics/test_native_milbrandt_warm.cpp; tests/physics/test_native_milbrandt_melting.cpp; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"microphysics.milbrandt_triple_moment",
         "microphysics",
         "Retained Milbrandt-Yau triple-moment selection with five independent sedimenting-species reflectivity moments",
         State::EngineeringIncomplete,
         "src/microphysics/schemes/milbrandt/milbrandt.cpp; "
         "src/microphysics/schemes/milbrandt/native.cpp; "
         "include/microphysics/milbrandt_triple_scope.hpp",
         "",
         "",
         "Ordinary strict-v1 selection remains fail-closed. Five reflectivity process and sedimentation tendencies, "
         "transport, checkpoint, resource, CPU and native backend integration and bounded scientific gates remain."},
        {"microphysics.thompson_native",
         "microphysics",
         "Exact flat periodic C-grid Thompson-Eidhammer/YSU/clear-sky-RRTMG profile on Vulkan and Metal",
         State::Ready,
         "tests/fixtures/configs/qualification/cgrid_thompson_vulkan_ready.yaml; tests/fixtures/configs/qualification/cgrid_thompson_metal_ready.yaml; scripts/tests/science/test_thompson_accelerator_ready.py; tests/physics/test_native_thompson_completion.cpp; tests/physics/test_native_thompson_sedimentation.cpp; docs/MATHEMATICS.pdf",
         "",
         "",
         ""},
        {"microphysics.thompson",
         "microphysics",
         "Exact flat periodic CPU Thompson-Eidhammer/YSU/clear-sky-RRTMG profile",
         State::Ready,
         "tests/fixtures/configs/qualification/cgrid_thompson_cpu_ready.yaml; "
         "scripts/tests/science/test_thompson_cpu_ready.py; tests/physics/test_microphysics.cpp; "
         "tests/physics/test_thompson_cold_processes.cpp; tests/physics/test_thompson_required_packaged_data.cpp",
         "",
         "",
         ""},
        {"radar.reflectivity_velocity",
         "radar",
         "Exact CPU fast-DA reflectivity and point radial-velocity profile",
         State::Ready,
         "tests/physics/test_reflectivity.cpp; tests/physics/test_radar_velocity.cpp; "
         "tests/physics/test_radar_runtime_state.cpp; tests/core/test_simulation_checkpoint.cpp; "
         "scripts/tests/science/test_radar_ready.py; tests/fixtures/configs/qualification/cgrid_radar_ready.yaml",
         "",
         "",
         ""},
        {"radar.zdr",
         "radar",
         "Exact CPU Zhang-2021 fixed-N0 rain-only profile",
         State::Ready,
         "tests/physics/test_radar_zdr.cpp; tests/physics/test_radar_runtime_state.cpp; "
         "tests/core/test_simulation_checkpoint.cpp; scripts/tests/science/test_radar_polarimetric_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_radar_polarimetric_ready.yaml",
         "",
         "",
         ""},
        {"radar.kdp",
         "radar",
         "Exact CPU Zhang-2021 fixed-N0 rain-only profile",
         State::Ready,
         "tests/physics/test_radar_zdr.cpp; tests/physics/test_radar_runtime_state.cpp; "
         "tests/core/test_simulation_checkpoint.cpp; scripts/tests/science/test_radar_polarimetric_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_radar_polarimetric_ready.yaml",
         "",
         "",
         ""},
        {"radar.rhohv",
         "radar",
         "Exact CPU Zhang-2021 fixed-N0 rain-only profile",
         State::Ready,
         "tests/physics/test_radar_zdr.cpp; tests/physics/test_radar_runtime_state.cpp; "
         "tests/core/test_simulation_checkpoint.cpp; scripts/tests/science/test_radar_polarimetric_ready.py; "
         "tests/fixtures/configs/qualification/cgrid_radar_polarimetric_ready.yaml",
         "",
         "",
         ""},
        {"radar.product_set_composition",
         "radar",
         "Every non-empty reflectivity/velocity/ZDR/KDP/rhoHV product subset on the canonical flat CPU Kessler/RRTMG "
         "radar environment, with selection-authored output",
         State::Ready,
         "tests/physics/test_reflectivity.cpp; tests/physics/test_radar_velocity.cpp; "
         "tests/physics/test_radar_zdr.cpp; tests/physics/test_radar_runtime_state.cpp; "
         "tests/core/test_output_config.cpp; scripts/qualification/output/qualify_radar_product_sets.py",
         "",
         "",
         ""},
        {"chaos.cpu_profiles",
         "chaos",
         "Qualified Earth-addressed CPU stochastic profiles",
         State::Ready,
         "tests/physics/test_chaos_runtime_state.cpp",
         "",
         "",
         ""},
        {"chaos.native_vulkan",
         "chaos",
         "Exact fallback-off Vulkan Earth-control evolution in host-composed chaos",
         State::Ready,
         "tests/physics/test_chaos_runtime_state.cpp; tests/vulkan/test_gpu_parity.cpp; "
         "scripts/tests/science/test_native_chaos_accelerator_ready.py; "
         "tests/fixtures/configs/qualification/domain_recycling_kessler_native_chaos_vulkan_ready.yaml",
         "",
         "",
         ""},
        {"chaos.native_metal",
         "chaos",
         "Exact fallback-off Metal Earth-control evolution in host-composed chaos",
         State::Ready,
         "tests/physics/test_chaos_runtime_state.cpp; tests/metal/test_metal_compute_backend.cpp; "
         "scripts/tests/science/test_native_chaos_accelerator_ready.py; "
         "tests/fixtures/configs/qualification/domain_recycling_kessler_native_chaos_metal_ready.yaml",
         "",
         "",
         ""},
        {"ledgers.nonlinear_boundaries",
         "conservation",
         "Exact qualified CPU, Vulkan, and Metal radiative-boundary energy ownership",
         State::Ready,
         "tests/dynamics/test_cartesian_lateral_bc_yaml.cpp; tests/numerics/test_advection_cartesian.cpp; "
         "tests/core/test_simulation_checkpoint.cpp; scripts/qualification/checkpointing/qualify_vulkan_restart.py; "
         "scripts/tests/science/test_mapped_terrain_ready.py; "
         "scripts/tests/science/test_native_chaos_accelerator_ready.py",
         "",
         "",
         ""},
        {"catalog.runtime_cli_studio",
         "capability_truth",
         "Exact-binary runtime, CLI, and Studio capability agreement",
         State::EngineeringIncomplete,
         "tests/core/test_scheme_catalog.cpp; tests/ui/unit/config/test_scheme_catalog.ts; "
         "ui/src-tauri/src/commands/config_preflight.rs",
         "",
         "",
         "Browser snapshot and maintained-config Studio/native parity pass, but exact generated Metal "
         "scope preflight and installed-package consumer agreement remain unproven"},
        {"release.provenance",
         "release",
         "Exact release artifact, binary, configuration, toolchain, hardware, and evidence binding",
         State::Ready,
         "scripts/qualification/provenance/release_manifest.py; tests/qualification/test_release_manifest.py",
         "",
         "",
         ""},
    };
    return items;
}

} // namespace tmv::runtime
