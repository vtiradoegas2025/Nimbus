# Explicit source ownership manifest. Keep groups module-local and reviewable;
# source discovery belongs in the verification gate, not the build graph.

VALIDATION_SRCS := src/diagnostics/contracts/field_contract.cpp \
	src/diagnostics/contracts/field_validation.cpp

CORE_INFRA_SRCS := src/core/infra/coordinate_system.cpp \
	src/core/infra/domain_motion.cpp \
	src/core/infra/domain_recycling.cpp \
	src/core/infra/domain_recycling_provenance.cpp \
	src/core/infra/domain_recycling_state.cpp \
	src/core/infra/field_sanitization.cpp \
	src/core/infra/hardware_info.cpp \
	src/core/infra/nested_execution.cpp \
	src/core/infra/nested_level_runtime.cpp \
	src/core/infra/nested_grid.cpp \
	src/core/infra/nested_boundary_forcing.cpp \
	src/core/infra/nested_feedback.cpp \
	src/core/infra/nested_scalar_feedback.cpp \
	src/core/infra/nested_terrain_geometry.cpp \
	src/core/infra/rayleigh_damping.cpp \
	src/core/infra/sha256.cpp \
	src/core/infra/simd_utils.cpp

CORE_OUTPUT_SRCS := src/core/output/npy_writer.cpp \
	src/core/output/domain_recycling_swath.cpp \
	src/core/output/history_index.cpp \
	src/core/output/netcdf_history_writer.cpp \
		src/core/output/output_config.cpp \
		src/core/output/output_writer.cpp \
		src/core/output/retention_policy.cpp \
		src/core/output/shm_writer.cpp \
	src/core/output/stagger_interpolation.cpp

CORE_RUNTIME_SRCS := src/core/runtime/boundary_layer_checkpoint.cpp \
	src/core/runtime/earth_fixed_step_transaction.cpp \
	src/core/runtime/earth_fixed_remap.cpp \
	src/core/runtime/held_source_remap.cpp \
	src/core/runtime/native_held_sources.cpp \
	src/core/runtime/environment_column_refresh.cpp \
	src/core/runtime/native_column_remap.cpp \
	src/core/runtime/native_earth_fixed_transport.cpp \
	src/core/runtime/global_runtime_bindings.cpp \
	src/core/runtime/headless_runtime.cpp \
	src/core/runtime/nimbus_sim.cpp \
	src/core/runtime/readiness.cpp \
	src/core/runtime/runtime_config.cpp \
	src/core/runtime/scheme_catalog.cpp \
	src/core/runtime/simulation_checkpoint.cpp \
	src/core/runtime/simulation_state.cpp \
	src/core/runtime/termination_signal.cpp \
	src/core/runtime/yaml_helpers.cpp

CORE_ORCHESTRATION_SRCS := src/core/orchestration/dynamics/dynamics.cpp \
	src/core/orchestration/dynamics/eos_apply.cpp \
	src/core/orchestration/dynamics/equations.cpp \
	src/core/orchestration/dynamics/initial_conditions_cartesian.cpp \
	src/core/orchestration/dynamics/nested_child_step.cpp \
	src/core/orchestration/dynamics/sgs_apply.cpp \
	src/core/orchestration/dynamics/thermodynamic_closure.cpp \
	src/core/orchestration/physics/diffusion_step.cpp \
	src/core/orchestration/physics/microphysics_step.cpp

COMPUTE_RUNTIME_SRC := src/compute/backend_runtime.cpp
COMPUTE_RESIDENCY_SRC := src/compute/resident_field_store.cpp

COMPUTE_CORE_SRCS := src/compute/backend_types.cpp \
	$(COMPUTE_RUNTIME_SRC) \
	$(COMPUTE_RESIDENCY_SRC) \
	src/compute/compute_backend.cpp \
	src/compute/cpu/cpu_backend.cpp \
	src/compute/cpu/conservative_transport.cpp \
	src/compute/cpu/acoustic_cgrid.cpp \
	src/compute/cpu/acoustic_cgrid_fast.cpp \
	src/compute/cpu/acoustic_cgrid_energy.cpp \
	src/compute/dispatch/vertical_flux.cpp \
	src/compute/dispatch/internal/vertical_flux_common.cpp \
	src/compute/dispatch/internal/vertical_flux_v1.cpp \
	src/compute/dispatch/internal/vertical_flux_v2.cpp \
	src/compute/dispatch/advection.cpp \
	src/compute/dispatch/dynamics.cpp \
	src/compute/dispatch/microphysics.cpp \
	src/compute/dispatch/turbulence.cpp \
	src/compute/dispatch/diffusion.cpp \
	src/compute/dispatch/residency.cpp \
	src/compute/kernel_manifest.cpp \
	src/compute/parity_comparison.cpp \
	src/compute/parity_check.cpp

COMPUTE_SRCS := $(COMPUTE_CORE_SRCS)

BOUNDARY_CONDITION_SCHEME_SRCS := src/boundary_conditions/boundary_conditions_cartesian.cpp \
	src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp \
	src/boundary_conditions/factory.cpp
BOUNDARY_CONDITION_SRCS := src/boundary_conditions/runtime.cpp \
	$(BOUNDARY_CONDITION_SCHEME_SRCS)

DYNAMICS_KERNEL_SRCS := src/dynamics/factory.cpp \
	src/dynamics/cartesian_cgrid_momentum_flux.cpp \
	src/dynamics/moving_cgrid_transport.cpp \
	src/dynamics/moving_boundary.cpp \
	src/dynamics/native_cgrid_boundary.cpp \
	src/dynamics/moving_environment.cpp \
	src/dynamics/native_moving_environment.cpp \
	src/dynamics/moving_energy.cpp \
	src/dynamics/native_moving_cgrid_transport.cpp \
	src/dynamics/native_moving_boundary.cpp \
	src/dynamics/native_moving_energy.cpp \
	src/dynamics/schemes/klemp_wilhelmson_split/cartesian_agrid.cpp \
	src/dynamics/schemes/klemp_wilhelmson_split/cartesian_cgrid_3d.cpp
DYNAMICS_SCHEME_SRCS := src/dynamics/runtime.cpp \
	src/dynamics/device_acoustic_energy_attribution.cpp \
	src/dynamics/device_slow_energy_attribution.cpp \
	$(DYNAMICS_KERNEL_SRCS)

ADVECTION_SCHEME_SRCS := src/numerics/advection/factory.cpp \
	src/numerics/advection/schemes/none/none.cpp \
	src/numerics/advection/schemes/tvd/tvd.cpp \
	src/numerics/advection/schemes/weno5/weno5.cpp
TERRAIN_TRANSPORT_SRC := src/terrain/base/conservative_transport.cpp

ADVECTION_SRCS := src/numerics/advection/advection.cpp \
	src/numerics/advection/advection_batch.cpp \
	src/numerics/advection/advection_cartesian.cpp \
	src/numerics/advection/cartesian_interface_flux_register.cpp \
	src/numerics/advection/advection_perf.cpp \
	$(TERRAIN_TRANSPORT_SRC) \
	$(ADVECTION_SCHEME_SRCS)

DIFFUSION_SCHEME_SRCS := src/numerics/diffusion/factory.cpp \
	src/numerics/diffusion/linear_height_profile.cpp \
	src/numerics/diffusion/schemes/explicit/explicit.cpp \
	src/numerics/diffusion/schemes/implicit/implicit.cpp

TIME_STEPPING_SCHEME_SRCS := src/numerics/time_stepping/factory.cpp \
	src/numerics/time_stepping/native_rk.cpp \
	src/numerics/time_stepping/schemes/rk3/rk3.cpp \
	src/numerics/time_stepping/schemes/rk4/rk4.cpp \
	src/numerics/time_stepping/schemes/split_explicit/split_explicit.cpp

NUMERICS_SRCS := src/numerics/runtime.cpp $(ADVECTION_SRCS) \
	$(DIFFUSION_SCHEME_SRCS) $(TIME_STEPPING_SCHEME_SRCS)

MICROPHYSICS_SCHEME_SRCS := src/microphysics/base/thermodynamics.cpp \
	src/microphysics/factory.cpp \
	src/microphysics/schemes/kessler/kessler.cpp \
	src/microphysics/schemes/lin/lin.cpp \
	src/microphysics/schemes/thompson/activation_table.cpp \
	src/microphysics/schemes/thompson/aerosol_scavenging.cpp \
	src/microphysics/schemes/thompson/cloud_evaporation.cpp \
	src/microphysics/schemes/thompson/cold_process_tables.cpp \
	src/microphysics/schemes/thompson/cold_processes.cpp \
	src/microphysics/schemes/thompson/collection_processes.cpp \
	src/microphysics/schemes/thompson/collection_tables.cpp \
	src/microphysics/schemes/thompson/frozen_melting.cpp \
	src/microphysics/schemes/thompson/moment_initialization.cpp \
	src/microphysics/schemes/thompson/nucleation.cpp \
	src/microphysics/schemes/thompson/particle_distributions.cpp \
	src/microphysics/schemes/thompson/prognostic_state.cpp \
	src/microphysics/schemes/thompson/required_packaged_data.cpp \
	src/microphysics/schemes/thompson/sedimentation.cpp \
	src/microphysics/schemes/thompson/sedimentation_native.cpp \
	src/microphysics/schemes/thompson/native_cloud_tables.cpp \
	src/microphysics/schemes/thompson/native_cloud_adjustment.cpp \
	src/microphysics/schemes/thompson/native_warm_rain.cpp \
	src/microphysics/schemes/thompson/native_freezing_tables.cpp \
	src/microphysics/schemes/thompson/native_freezing.cpp \
	src/microphysics/schemes/thompson/native_deposition_tables.cpp \
	src/microphysics/schemes/thompson/native_deposition.cpp \
	src/microphysics/schemes/thompson/completion_tables.cpp \
	src/microphysics/schemes/thompson/native_completion.cpp \
	src/microphysics/schemes/thompson/cloud_phase.cpp \
	src/microphysics/schemes/thompson/cloud_phase_runtime.cpp \
	src/microphysics/schemes/thompson/state.cpp \
	src/microphysics/schemes/thompson/thermodynamic_helpers.cpp \
	src/microphysics/schemes/thompson/warm_rain_processes.cpp \
	src/microphysics/schemes/thompson/thompson.cpp \
	src/microphysics/schemes/milbrandt/native.cpp \
	src/microphysics/schemes/milbrandt/triple_moment.cpp \
	src/microphysics/schemes/milbrandt/triple_sedimentation.cpp \
	src/microphysics/schemes/milbrandt/triple_internal.cpp \
	src/microphysics/schemes/milbrandt/triple_warm_events.cpp \
	src/microphysics/schemes/milbrandt/triple_warm_phase.cpp \
	src/microphysics/schemes/milbrandt/triple_warm_adapter.cpp \
	src/microphysics/schemes/milbrandt/triple_warm_primary.cpp \
	src/microphysics/schemes/milbrandt/triple_warm_primary_internal.cpp \
	src/microphysics/schemes/milbrandt/triple_cloud_adjustment.cpp \
	src/microphysics/schemes/milbrandt/triple_cloud_adjustment_internal.cpp \
	src/microphysics/schemes/milbrandt/triple_warm_cloud.cpp \
	src/microphysics/schemes/milbrandt/triple_warm_cloud_internal.cpp \
	src/microphysics/schemes/milbrandt/triple_warm_spectrum.cpp \
	src/microphysics/schemes/milbrandt/triple_warm_dynamic.cpp \
	src/microphysics/schemes/milbrandt/triple_warm_dynamic_adapter.cpp \
	src/microphysics/schemes/milbrandt/triple_warm_subcycle.cpp \
	src/microphysics/schemes/milbrandt/triple_frozen_collection.cpp \
	src/microphysics/schemes/milbrandt/triple_frozen_collection_internal.cpp \
	src/microphysics/schemes/milbrandt/triple_cold_cloud_collection.cpp \
	src/microphysics/schemes/milbrandt/triple_cold_cloud_collection_internal.cpp \
	src/microphysics/schemes/milbrandt/triple_cold_cloud_diffusion.cpp \
	src/microphysics/schemes/milbrandt/triple_cold_cloud_diffusion_internal.cpp \
	src/microphysics/schemes/milbrandt/triple_ice_cloud_collection.cpp \
	src/microphysics/schemes/milbrandt/triple_rime_splintering.cpp \
	src/microphysics/schemes/milbrandt/triple_cold_riming.cpp \
	src/microphysics/schemes/milbrandt/triple_ice_graupel_conversion.cpp \
	src/microphysics/schemes/milbrandt/triple_snow_graupel_conversion.cpp \
	src/microphysics/schemes/milbrandt/triple_homogeneous_cloud_freezing.cpp \
	src/microphysics/schemes/milbrandt/triple_graupel_hail_conversion.cpp \
	src/microphysics/schemes/milbrandt/triple_cold_riming_internal.cpp \
	src/microphysics/schemes/milbrandt/triple_warm_internal.cpp \
	src/microphysics/schemes/milbrandt/triple_cold_freezing.cpp \
	src/microphysics/schemes/milbrandt/triple_cold_internal.cpp \
	src/microphysics/schemes/milbrandt/triple_cold_diffusion.cpp \
	src/microphysics/schemes/milbrandt/triple_melting.cpp \
	src/microphysics/schemes/milbrandt/triple_ice_melting.cpp \
	src/microphysics/schemes/milbrandt/triple_snow_melting.cpp \
	src/microphysics/schemes/milbrandt/triple_cold_diffusion_internal.cpp \
	src/microphysics/schemes/milbrandt/triple_cold_joint_internal.cpp \
	src/microphysics/schemes/milbrandt/triple_melting_internal.cpp \
	src/microphysics/schemes/milbrandt/triple_ice_melting_internal.cpp \
	src/microphysics/schemes/milbrandt/triple_snow_melting_internal.cpp \
	src/microphysics/schemes/milbrandt/milbrandt.cpp \
	src/microphysics/schemes/none/none.cpp
MICROPHYSICS_SRCS := src/microphysics/runtime.cpp src/microphysics/grid_geometry.cpp src/microphysics/grid_velocity.cpp \
	src/microphysics/domain_recycling.cpp $(MICROPHYSICS_SCHEME_SRCS)

RADIATION_SCHEME_SRCS := src/radiation/base/radiative_transfer.cpp \
	src/radiation/factory.cpp \
	src/radiation/schemes/rrtmg/rrtmg.cpp \
	src/radiation/schemes/rrtmg/rrtmg_batch.cpp \
	src/radiation/schemes/rrtmg/rrtmg_atmosphere.cpp \
	src/radiation/schemes/rrtmg/rrtmg_coefficients.cpp \
	src/radiation/schemes/rrtmg/rrtmg_cloud_optics.cpp \
	src/radiation/schemes/rrtmg/rrtmg_preprocessing.cpp \
	src/radiation/schemes/rrtmg/rrtmg_optics_internal.cpp \
	src/radiation/schemes/rrtmg/rrtmg_longwave_minor_optics.cpp \
	src/radiation/schemes/rrtmg/rrtmg_longwave_optics.cpp \
	src/radiation/schemes/rrtmg/rrtmg_shortwave_optics.cpp \
	src/radiation/schemes/rrtmg/rrtmg_longwave_transfer.cpp \
	src/radiation/schemes/rrtmg/rrtmg_shortwave_transfer.cpp \
	src/radiation/schemes/simple_grey/simple_grey.cpp \
	src/radiation/schemes/simple_grey/simple_grey_native.cpp \
	src/radiation/schemes/none/none.cpp
RADIATION_SRCS := src/radiation/runtime.cpp \
	src/radiation/native_batch_runtime.cpp src/radiation/energy_ledger.cpp \
	src/radiation/column_evaluation.cpp \
	$(RADIATION_SCHEME_SRCS)

BOUNDARY_LAYER_SCHEME_SRCS := src/boundary_layer/base/surface_fluxes.cpp \
	src/boundary_layer/config_validation.cpp \
	src/boundary_layer/factory.cpp \
	src/boundary_layer/schemes/slab/slab.cpp \
	src/boundary_layer/schemes/slab/slab_native.cpp \
	src/boundary_layer/schemes/ysu/ysu.cpp \
	src/boundary_layer/schemes/mynn/mynn_stability.cpp \
	src/boundary_layer/schemes/mynn/mynn.cpp \
	src/boundary_layer/schemes/mynn/mynn_batch.cpp \
	src/boundary_layer/schemes/none/none.cpp
BOUNDARY_LAYER_SRCS := src/boundary_layer/runtime.cpp \
	src/boundary_layer/native_batch_runtime.cpp \
	src/boundary_layer/column_evaluation.cpp $(BOUNDARY_LAYER_SCHEME_SRCS)

TURBULENCE_SCHEME_SRCS := src/turbulence/base/eddy_viscosity.cpp \
	src/turbulence/base/collocated_eddy_viscosity.cpp \
	src/turbulence/base/native_collocated_eddy_viscosity.cpp \
	src/turbulence/factory.cpp \
	src/turbulence/schemes/smagorinsky/dynamic_smagorinsky.cpp \
	src/turbulence/schemes/smagorinsky/smagorinsky.cpp \
	src/turbulence/schemes/tke/tke.cpp \
	src/turbulence/schemes/none/none.cpp
TURBULENCE_SRCS := src/turbulence/runtime.cpp $(TURBULENCE_SCHEME_SRCS)

CHAOS_SCHEME_SRCS := src/chaos/base/random_generator.cpp \
	src/chaos/base/perturbation_field.cpp \
	src/chaos/base/earth_correlated_noise.cpp \
	src/chaos/base/correlation_filter.cpp \
	src/chaos/factory.cpp \
	src/chaos/schemes/none/none.cpp \
	src/chaos/schemes/initial_conditions/initial_conditions.cpp \
	src/chaos/schemes/boundary_layer/boundary_layer.cpp \
	src/chaos/schemes/full_stochastic/full_stochastic.cpp
CHAOS_SRCS := src/chaos/chaos.cpp src/chaos/init.cpp \
	src/chaos/runtime_state.cpp $(CHAOS_SCHEME_SRCS)

TERRAIN_SCHEME_SRCS := src/terrain/base/topography.cpp \
	src/terrain/base/earth_fixed_geometry.cpp src/terrain/base/earth_fixed_volume_geometry.cpp \
	src/terrain/factory.cpp \
	src/terrain/schemes/bell/bell.cpp \
	src/terrain/schemes/schar/schar.cpp \
	src/terrain/schemes/none/none.cpp
TERRAIN_SRCS := src/terrain/runtime.cpp $(TERRAIN_SCHEME_SRCS)
TERRAIN_NATIVE_SRCS := src/terrain/base/native_mapped_transaction.cpp src/terrain/base/native_conservative_remap.cpp src/terrain/base/native_earth_geometry.cpp
TERRAIN_COLLOCATED_SRCS := src/terrain/base/collocated_mapped_timestep.cpp

RADAR_SCHEME_SRCS := src/radar/base/radar_base.cpp \
	src/radar/factory.cpp \
	src/radar/schemes/reflectivity/reflectivity.cpp \
	src/radar/schemes/velocity/velocity.cpp \
	src/radar/schemes/zdr/zdr.cpp
RADAR_SYSTEM_SRCS := src/radar/system.cpp $(RADAR_SCHEME_SRCS)
RADAR_SRCS := src/radar/runtime.cpp $(RADAR_SYSTEM_SRCS)

SOUNDING_ALGORITHM_SRCS := src/soundings/base/sounding_algorithms.cpp
SHARPY_SOUNDING_SRCS := src/soundings/schemes/sharpy/sharpy_sounding.cpp \
	src/soundings/schemes/sharpy/sharpy_native_reader.cpp \
	src/soundings/schemes/sharpy/sharpy_netcdf_dynamic_reader.cpp \
	src/soundings/schemes/sharpy/sharpy_hdf5_reader.cpp \
	src/soundings/schemes/sharpy/sharpy_interpolation.cpp \
	src/soundings/schemes/sharpy/sharpy_profile_parse.cpp
CM1_SOUNDING_SRCS := src/soundings/schemes/cm1/cm1_sounding.cpp
AUTO_SOUNDING_SRCS := src/soundings/schemes/auto/auto_sounding.cpp
SOUNDING_BACKEND_SRCS := src/soundings/factory.cpp \
	$(SOUNDING_ALGORITHM_SRCS) $(SHARPY_SOUNDING_SRCS) \
	$(CM1_SOUNDING_SRCS) $(AUTO_SOUNDING_SRCS)
SOUNDING_READER_SRCS := src/soundings/soundings.cpp $(SOUNDING_BACKEND_SRCS)

INIT_SOUNDING_SRCS := src/init/sounding/sounding_source.cpp \
	src/init/sounding/factory.cpp \
	src/init/sounding/diagnostics.cpp \
	src/init/sounding/sources/parametric_cape.cpp \
	src/init/sounding/sources/file_sounding.cpp \
	src/init/sounding/sources/parametric_targets.cpp \
	src/init/sounding/sources/fallback_source.cpp
INIT_HODOGRAPH_SRCS := src/init/hodograph/factory.cpp \
	src/init/hodograph/sources/wk_param.cpp \
	src/init/hodograph/sources/zero.cpp
INIT_TRIGGER_SRCS := src/init/trigger/factory.cpp \
	src/init/trigger/sources/warm_bubble.cpp \
	src/init/trigger/sources/none.cpp \
	src/init/trigger/sources/vortex_seed.cpp
CONFIG_CODEC_SRCS := src/init/config/config_v1.cpp \
	src/init/config/config_v1_physics.cpp \
	src/init/config/config_v1_initialization.cpp \
	src/init/config/config_v1_output.cpp
CONFIG_MIGRATION_SRCS := src/init/config/v0_to_v1.cpp \
	src/init/config/v0_to_v1_initialization.cpp \
	src/init/config/v0_to_v1_physics.cpp \
	src/init/config/v0_to_v1_runtime.cpp \
	src/init/config/config_v1_serialization_common.cpp \
	src/init/config/qualified_pbl_top_profile.cpp \
	src/init/config/qualified_terrain_profile.cpp \
	src/init/config/config_v1_serialize.cpp \
	src/init/config/config_v1_projection.cpp
CONFIG_SRCS := $(CONFIG_CODEC_SRCS) src/init/config/v1_loader.cpp \
	$(CONFIG_MIGRATION_SRCS)
INIT_SRCS := src/init/scheme_profile.cpp $(INIT_SOUNDING_SRCS) \
	$(INIT_HODOGRAPH_SRCS) $(INIT_TRIGGER_SRCS) $(CONFIG_SRCS)

DIAGNOSTICS_SRCS := src/diagnostics/budgets/conservation_budget.cpp \
	src/diagnostics/budgets/momentum_budget.cpp \
	src/diagnostics/budgets/energy_budget.cpp \
	src/diagnostics/budgets/velocity_energy_pair.cpp \
	src/diagnostics/budgets/moisture_budget.cpp \
	src/diagnostics/storm_tracker.cpp \
	src/diagnostics/storm_rotation.cpp \
	src/diagnostics/derived/column_water_paths.cpp \
	src/diagnostics/derived/pressure_projection.cpp \
	src/diagnostics/derived/resolved_diagnostics.cpp \
	src/diagnostics/derived/derived_slice_compute.cpp \
	src/diagnostics/derived/tranche_product_registry.cpp \
	src/diagnostics/trajectories/trajectories.cpp \
	$(VALIDATION_SRCS)

VULKAN_BACKEND_SRCS := vulkan/src/compute/compute_backend_vulkan.cpp
ifeq ($(CPU_ONLY),1)
  VULKAN_BACKEND_SRCS :=
endif

SRCS := $(CORE_INFRA_SRCS) $(CORE_OUTPUT_SRCS) $(CORE_RUNTIME_SRCS) \
	$(CORE_ORCHESTRATION_SRCS) $(COMPUTE_SRCS) $(BOUNDARY_CONDITION_SRCS) \
	$(DYNAMICS_SCHEME_SRCS) $(NUMERICS_SRCS) $(MICROPHYSICS_SRCS) \
	$(RADIATION_SRCS) $(BOUNDARY_LAYER_SRCS) $(TURBULENCE_SRCS) \
	$(CHAOS_SRCS) $(TERRAIN_SRCS) $(TERRAIN_NATIVE_SRCS) $(TERRAIN_COLLOCATED_SRCS) $(RADAR_SRCS) $(SOUNDING_READER_SRCS) \
	$(INIT_SRCS) $(DIAGNOSTICS_SRCS) $(VULKAN_BACKEND_SRCS)

FIELD_VALIDATOR_SRCS := src/tools/field_validator.cpp $(VALIDATION_SRCS) \
	src/core/io/npy_reader.cpp
OPT_IN_SRCS := src/core/output/zfp_reader.cpp \
	src/core/infra/allocation_telemetry.cpp
STANDALONE_TOOL_SRCS := src/tools/thompson_activation_table_tool.cpp \
	src/tools/thompson_collection_table_tool.cpp \
	src/tools/thompson_cold_table_tool.cpp
OUTPUT_WRITER_SRCS := src/core/output/output_writer.cpp \
		src/core/output/output_config.cpp src/core/output/npy_writer.cpp \
		src/core/output/history_index.cpp src/core/infra/sha256.cpp \
		src/core/output/retention_policy.cpp \
		src/core/output/netcdf_history_writer.cpp \
	src/diagnostics/contracts/field_contract.cpp

BACKEND_COMMON_SRCS := $(COMPUTE_CORE_SRCS) \
	src/core/infra/hardware_info.cpp \
	src/core/infra/simd_utils.cpp \
	src/numerics/advection/advection_cartesian.cpp \
	src/numerics/advection/advection_perf.cpp \
	$(TERRAIN_TRANSPORT_SRC) \
	$(VULKAN_BACKEND_SRCS)

CHECKPOINT_SCHEME_SRCS := src/core/runtime/simulation_checkpoint.cpp \
	src/core/runtime/boundary_layer_checkpoint.cpp \
	src/diagnostics/storm_tracker.cpp \
	src/core/infra/domain_motion.cpp \
	src/core/runtime/simulation_state.cpp \
	src/core/infra/coordinate_system.cpp \
	$(RADIATION_SCHEME_SRCS) $(TURBULENCE_SCHEME_SRCS) \
	$(BOUNDARY_LAYER_SCHEME_SRCS) $(CHAOS_SCHEME_SRCS) \
	$(MICROPHYSICS_SCHEME_SRCS) $(TERRAIN_SCHEME_SRCS) \
	$(RADAR_SCHEME_SRCS) $(ADVECTION_SCHEME_SRCS) \
	$(DIFFUSION_SCHEME_SRCS) $(TIME_STEPPING_SCHEME_SRCS)
