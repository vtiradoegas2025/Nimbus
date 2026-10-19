#pragma once

#include "compute/metal/metal_backend_evidence.hpp"

#include <cstdint>

namespace nimbus::metal::detail
{

// Mutable execution measurements are kept together so reset and evidence
// projection cannot drift away from the dispatch paths that update them.
struct MetalTelemetry
{
    bool          timestamp_support                                   = false;
    bool          last_completed                                      = false;
    std::uint64_t completed_command_buffers                           = 0;
    std::uint64_t completed_shift_dispatches                          = 0;
    std::uint64_t completed_shift_batches                             = 0;
    std::uint64_t resident_uploads                                    = 0;
    std::uint64_t resident_reuses                                     = 0;
    std::uint64_t completed_smagorinsky_coefficients_dispatches       = 0;
    std::uint64_t completed_smagorinsky_tendencies_dispatches         = 0;
    std::uint64_t completed_smagorinsky_fused_dispatches              = 0;
    std::uint64_t completed_sgs_apply_dispatches                      = 0;
    std::uint64_t sgs_resident_tendency_reuses                        = 0;
    std::uint64_t completed_diffusion_dispatches                      = 0;
    std::uint64_t completed_diffusion_batches                         = 0;
    std::uint64_t completed_kessler_pointwise_dispatches              = 0;
    std::uint64_t completed_kessler_sedimentation_dispatches          = 0;
    std::uint64_t completed_kessler_fused_dispatches                  = 0;
    std::uint64_t kessler_resident_tendency_writes                    = 0;
    std::uint64_t completed_lin_pointwise_dispatches                  = 0;
    std::uint64_t completed_lin_sedimentation_dispatches              = 0;
    std::uint64_t completed_lin_fused_dispatches                      = 0;
    std::uint64_t lin_resident_tendency_writes                        = 0;
    std::uint64_t completed_microphysics_apply_dispatches             = 0;
    std::uint64_t microphysics_apply_resident_tendency_reuses         = 0;
    std::uint64_t completed_advection_x_dispatches                    = 0;
    std::uint64_t completed_advection_y_dispatches                    = 0;
    std::uint64_t completed_advection_pre_vertical_batches            = 0;
    std::uint64_t completed_advection_post_vertical_batches           = 0;
    std::uint64_t completed_vertical_flux_dispatches                  = 0;
    std::uint64_t completed_conservative_cgrid_mass_flux_dispatches   = 0;
    std::uint64_t completed_conservative_cgrid_batches                = 0;
    std::uint64_t completed_conservative_cgrid_scalars                = 0;
    std::uint64_t completed_conservative_cgrid_scalar_flux_dispatches = 0;
    std::uint64_t completed_conservative_cgrid_update_dispatches      = 0;
    std::uint64_t completed_conservative_cgrid_pd_scalars             = 0;
    std::uint64_t conservative_cgrid_resident_reuses                  = 0;
    std::uint64_t completed_state_relaxation_dispatches               = 0;
    std::uint64_t completed_slow_apply_dispatches                     = 0;
    std::uint64_t dynamics_state_resident_reuses                      = 0;
    std::uint64_t completed_cartesian_cgrid_slow_tendency_dispatches  = 0;
    std::uint64_t slow_tendency_invariant_input_uploads               = 0;
    std::uint64_t slow_tendency_invariant_input_reuses                = 0;
    std::uint64_t slow_apply_resident_tendency_reuses                 = 0;
    std::uint64_t completed_acoustic_pressure_dispatches              = 0;
    std::uint64_t completed_acoustic_momentum_dispatches              = 0;
    std::uint64_t completed_acoustic_fused_dispatches                 = 0;
    std::uint64_t acoustic_invariant_uploads                          = 0;
    std::uint64_t acoustic_invariant_reuses                           = 0;
    std::uint64_t completed_acoustic_batched_dispatches               = 0;
    std::uint64_t completed_acoustic_boundary_face_dispatches         = 0;
    std::uint64_t completed_acoustic_mass_flux_exports                = 0;
    std::uint64_t completed_acoustic_mass_flux_pressure_dispatches    = 0;
    std::uint64_t completed_cartesian_agrid_tendency_dispatches       = 0;
    std::uint64_t completed_acoustic_agrid_pressure_dispatches        = 0;
    std::uint64_t completed_acoustic_agrid_momentum_dispatches        = 0;
    std::uint64_t completed_acoustic_agrid_fused_dispatches           = 0;
    std::uint64_t completed_acoustic_agrid_batched_dispatches         = 0;
    std::uint64_t completed_acoustic_agrid_continuity_dispatches      = 0;
    std::uint64_t completed_acoustic_agrid_pressure_eos_dispatches    = 0;
    std::uint64_t completed_acoustic_agrid_native_momentum_dispatches = 0;
    std::uint64_t completed_acoustic_agrid_constraint_dispatches      = 0;
    std::uint64_t completed_acoustic_agrid_boundary_dispatches        = 0;
    std::uint64_t completed_mynn_level25_stability_dispatches         = 0;
    std::uint64_t completed_rrtmg_longwave_transfer_dispatches        = 0;
    std::uint64_t completed_rrtmg_shortwave_transfer_dispatches       = 0;
    std::uint64_t completed_rrtmg_longwave_transfer_batch_dispatches  = 0;
    std::uint64_t completed_rrtmg_shortwave_transfer_batch_dispatches = 0;
    std::uint64_t completed_earth_control_evolution_dispatches        = 0;
    std::uint64_t completed_mapped_state_update_dispatches            = 0;
    std::uint64_t completed_mapped_flux_accumulation_dispatches       = 0;
    std::uint64_t completed_mapped_mass_continuity_dispatches         = 0;
    std::uint64_t completed_mapped_fast_momentum_dispatches           = 0;
    std::uint64_t completed_mapped_slow_tendencies_dispatches         = 0;
    std::uint64_t completed_mapped_scalar_limiter_dispatches          = 0;
    std::uint64_t completed_mapped_scalar_update_dispatches           = 0;
    std::uint64_t completed_mapped_eos_dispatches                     = 0;
    std::uint64_t completed_mapped_boundary_dispatches                = 0;
    std::uint64_t completed_milbrandt_melting_dispatches              = 0;
    std::uint64_t completed_milbrandt_sedimentation_dispatches        = 0;
    std::uint64_t completed_slab_column_dispatches = 0;
    std::uint64_t completed_simple_grey_dispatches = 0;
    std::uint64_t completed_held_sources_dispatches = 0;
    std::uint64_t completed_column_remap_dispatches = 0;
    std::uint64_t completed_reference_atmosphere_dispatches = 0;
    std::uint64_t completed_moving_energy_dispatches = 0;
    std::uint64_t completed_conservative_reduction_dispatches = 0;
    std::uint64_t completed_conservative_momentum_dispatches = 0;
    std::uint64_t completed_moving_momentum_dispatches = 0;
    std::uint64_t completed_earth_geometry_dispatches = 0;
    std::uint64_t completed_conservative_remap_dispatches = 0;
    std::uint64_t completed_collocated_sgs_dispatches = 0;
    std::uint64_t completed_rk_tendency_dispatches = 0;
    std::uint64_t completed_rk_combination_dispatches                 = 0;
    std::uint64_t completed_milbrandt_cold_dispatches                 = 0;
    std::uint64_t completed_milbrandt_cloud_adjustment_dispatches     = 0;
    std::uint64_t completed_milbrandt_warm_rain_dispatches            = 0;
    std::uint64_t completed_thompson_warm_rain_dispatches             = 0;
    std::uint64_t completed_thompson_cloud_phase_dispatches           = 0;
    std::uint64_t completed_thompson_freezing_dispatches              = 0;
    std::uint64_t completed_thompson_deposition_dispatches            = 0;
    std::uint64_t completed_thompson_completion_dispatches            = 0;
    std::uint64_t completed_thompson_cloud_adjustment_dispatches      = 0;
    std::uint64_t completed_thompson_moment_sedimentation_dispatches  = 0;
    std::uint64_t completed_tke_closure_dispatches = 0;
    std::uint64_t completed_tke_tendency_dispatches = 0;
    std::uint64_t completed_implicit_stencil_dispatches               = 0;
    std::uint64_t completed_implicit_vector_dispatches                = 0;
    double        last_device_ms                                      = 0.0;
    double        total_device_ms                                     = 0.0;
    double        total_submit_wait_ms                                = 0.0;
    std::uint64_t timestamped_command_buffers                          = 0;

    void reset() noexcept
    {
        *this = {};
    }

    std::uint64_t kernel_dispatches(nimbus::compute::KernelId kernel) const noexcept
    {
        using nimbus::compute::KernelId;
        switch (kernel)
        {
        case KernelId::DomainRecyclingHomogeneousFieldShift:
            return completed_shift_dispatches;
        case KernelId::SmagorinskyCoefficients:
            return completed_smagorinsky_coefficients_dispatches;
        case KernelId::SmagorinskyTendencies:
            return completed_smagorinsky_tendencies_dispatches;
        case KernelId::TkeClosure:
            return completed_tke_closure_dispatches;
        case KernelId::TkeTendency:
            return completed_tke_tendency_dispatches;
        case KernelId::ThompsonCompletion:
            return completed_thompson_completion_dispatches;
        case KernelId::ThompsonDeposition:
            return completed_thompson_deposition_dispatches;
        case KernelId::ThompsonFreezing:
            return completed_thompson_freezing_dispatches;
        case KernelId::ThompsonCloudPhase:
            return completed_thompson_cloud_phase_dispatches;
        case KernelId::MilbrandtMelting:
            return completed_milbrandt_melting_dispatches;
        case KernelId::MilbrandtSedimentation:
            return completed_milbrandt_sedimentation_dispatches;
        case KernelId::SlabColumn:
            return completed_slab_column_dispatches;
        case KernelId::SimpleGrey:
            return completed_simple_grey_dispatches;
        case KernelId::HeldSources:
            return completed_held_sources_dispatches;
        case KernelId::ColumnRemap:
            return completed_column_remap_dispatches;
        case KernelId::ReferenceAtmosphere:
            return completed_reference_atmosphere_dispatches;
        case KernelId::MovingEnergy:
            return completed_moving_energy_dispatches;
        case KernelId::ConservativeReduction:
            return completed_conservative_reduction_dispatches;
        case KernelId::ConservativeMomentum:
            return completed_conservative_momentum_dispatches;
        case KernelId::MovingMomentum:
            return completed_moving_momentum_dispatches;
        case KernelId::EarthGeometry:
            return completed_earth_geometry_dispatches;
        case KernelId::ConservativeRemap:
            return completed_conservative_remap_dispatches;
        case KernelId::CollocatedSgs:
            return completed_collocated_sgs_dispatches;
        case KernelId::RkTendency:
            return completed_rk_tendency_dispatches;
        case KernelId::RkCombination:
            return completed_rk_combination_dispatches;
        case KernelId::MilbrandtCold:
            return completed_milbrandt_cold_dispatches;
        case KernelId::MilbrandtCloudAdjustment:
            return completed_milbrandt_cloud_adjustment_dispatches;
        case KernelId::MilbrandtWarmRain:
            return completed_milbrandt_warm_rain_dispatches;
        case KernelId::ThompsonWarmRain:
            return completed_thompson_warm_rain_dispatches;
        case KernelId::ThompsonCloudAdjustment:
            return completed_thompson_cloud_adjustment_dispatches;
        case KernelId::ThompsonMomentSedimentation:
            return completed_thompson_moment_sedimentation_dispatches;
        case KernelId::ImplicitDiffusionStencil:
            return completed_implicit_stencil_dispatches;
        case KernelId::ImplicitDiffusionVector:
            return completed_implicit_vector_dispatches;
        case KernelId::SgsApply:
            return completed_sgs_apply_dispatches;
        case KernelId::Diffusion:
            return completed_diffusion_dispatches;
        case KernelId::KesslerPointwise:
            return completed_kessler_pointwise_dispatches;
        case KernelId::KesslerSedimentation:
            return completed_kessler_sedimentation_dispatches;
        case KernelId::LinPointwise:
            return completed_lin_pointwise_dispatches;
        case KernelId::LinSedimentation:
            return completed_lin_sedimentation_dispatches;
        case KernelId::MicrophysicsApply:
            return completed_microphysics_apply_dispatches;
        case KernelId::AdvectX:
            return completed_advection_x_dispatches;
        case KernelId::AdvectY:
            return completed_advection_y_dispatches;
        case KernelId::TvdVerticalFlux:
            return completed_vertical_flux_dispatches;
        case KernelId::AdvectConservativeMassFluxCartesianCgrid:
            return completed_conservative_cgrid_mass_flux_dispatches;
        case KernelId::AdvectConservativeScalarFluxCartesianCgrid:
            return completed_conservative_cgrid_scalar_flux_dispatches;
        case KernelId::AdvectConservativeUpdateCartesianCgrid:
            return completed_conservative_cgrid_update_dispatches;
        case KernelId::AdvectPdLimiterCartesianCgrid:
            return 0;
        case KernelId::StateRelaxationCartesian:
            return completed_state_relaxation_dispatches;
        case KernelId::SlowApply:
            return completed_slow_apply_dispatches;
        case KernelId::CartesianCgridSlowTendencies:
            return completed_cartesian_cgrid_slow_tendency_dispatches;
        case KernelId::CartesianTendencies:
            return completed_cartesian_agrid_tendency_dispatches;
        case KernelId::AcousticPressureCartesian:
            return completed_acoustic_agrid_pressure_dispatches;
        case KernelId::AcousticMomentumCartesian:
            return completed_acoustic_agrid_momentum_dispatches;
        case KernelId::AcousticContinuityMassFluxCartesianAgrid:
            return completed_acoustic_agrid_continuity_dispatches;
        case KernelId::AcousticPressureEosCartesianAgrid:
            return completed_acoustic_agrid_pressure_eos_dispatches;
        case KernelId::AcousticMomentumCartesianAgrid:
            return completed_acoustic_agrid_native_momentum_dispatches;
        case KernelId::AcousticNullModeConstraintCartesianAgrid:
            return completed_acoustic_agrid_constraint_dispatches;
        case KernelId::AcousticBoundaryCartesianAgrid:
            return completed_acoustic_agrid_boundary_dispatches;
        case KernelId::AcousticPressureCartesianCgrid:
            return completed_acoustic_pressure_dispatches;
        case KernelId::AcousticMomentumCartesianCgrid:
            return completed_acoustic_momentum_dispatches;
        case KernelId::AcousticPressureMassFluxCartesianCgrid:
            return completed_acoustic_mass_flux_pressure_dispatches;
        case KernelId::AcousticBcFaceCartesianCgrid:
            return completed_acoustic_boundary_face_dispatches;
        case KernelId::MynnLevel25Stability:
            return completed_mynn_level25_stability_dispatches;
        case KernelId::RrtmgLongwaveTransfer:
            return completed_rrtmg_longwave_transfer_dispatches;
        case KernelId::RrtmgShortwaveTransfer:
            return completed_rrtmg_shortwave_transfer_dispatches;
        case KernelId::RrtmgLongwaveTransferBatch:
            return completed_rrtmg_longwave_transfer_batch_dispatches;
        case KernelId::RrtmgShortwaveTransferBatch:
            return completed_rrtmg_shortwave_transfer_batch_dispatches;
        case KernelId::EarthControlEvolution:
            return completed_earth_control_evolution_dispatches;
        case KernelId::DiffuseCartesian:
        case KernelId::AcousticRetainedFluxFp64:
        case KernelId::ArchivedLegacyThompsonLinPointwise:
        case KernelId::ArchivedLegacyThompsonLinSedimentation:
            return 0;
        case KernelId::MappedTerrainMassContinuity:
            return completed_mapped_mass_continuity_dispatches;
        case KernelId::MappedTerrainFastMomentum:
            return completed_mapped_fast_momentum_dispatches;
        case KernelId::MappedTerrainSlowTendencies:
            return completed_mapped_slow_tendencies_dispatches;
        case KernelId::MappedTerrainScalarUpdate:
            return completed_mapped_scalar_update_dispatches;
        case KernelId::MappedTerrainScalarLimiter:
            return completed_mapped_scalar_limiter_dispatches;
        case KernelId::MappedTerrainEos:
            return completed_mapped_eos_dispatches;
        case KernelId::MappedTerrainBoundary:
            return completed_mapped_boundary_dispatches;
        case KernelId::MappedTerrainStateUpdate:
            return completed_mapped_state_update_dispatches;
        case KernelId::MappedTerrainFluxAccumulation:
            return completed_mapped_flux_accumulation_dispatches;
        case KernelId::Count:
            return 0;
        }
        return 0;
    }

    void write_to(MetalExecutionRecord& record) const noexcept
    {
        record.timestamps_supported                               = timestamp_support;
        record.completed_command_buffers                          = completed_command_buffers;
        record.completed_shift_dispatches                         = completed_shift_dispatches;
        record.completed_shift_batches                            = completed_shift_batches;
        record.resident_uploads                                   = resident_uploads;
        record.resident_reuses                                    = resident_reuses;
        record.completed_smagorinsky_coefficients_dispatches      = completed_smagorinsky_coefficients_dispatches;
        record.completed_smagorinsky_tendencies_dispatches        = completed_smagorinsky_tendencies_dispatches;
        record.completed_smagorinsky_fused_dispatches             = completed_smagorinsky_fused_dispatches;
        record.completed_sgs_apply_dispatches                     = completed_sgs_apply_dispatches;
        record.sgs_resident_tendency_reuses                       = sgs_resident_tendency_reuses;
        record.completed_diffusion_dispatches                     = completed_diffusion_dispatches;
        record.completed_diffusion_batches                        = completed_diffusion_batches;
        record.completed_kessler_pointwise_dispatches             = completed_kessler_pointwise_dispatches;
        record.completed_kessler_sedimentation_dispatches         = completed_kessler_sedimentation_dispatches;
        record.completed_kessler_fused_dispatches                 = completed_kessler_fused_dispatches;
        record.kessler_resident_tendency_writes                   = kessler_resident_tendency_writes;
        record.completed_lin_pointwise_dispatches                 = completed_lin_pointwise_dispatches;
        record.completed_lin_sedimentation_dispatches             = completed_lin_sedimentation_dispatches;
        record.completed_lin_fused_dispatches                     = completed_lin_fused_dispatches;
        record.lin_resident_tendency_writes                       = lin_resident_tendency_writes;
        record.completed_microphysics_apply_dispatches            = completed_microphysics_apply_dispatches;
        record.microphysics_apply_resident_tendency_reuses        = microphysics_apply_resident_tendency_reuses;
        record.completed_advection_x_dispatches                   = completed_advection_x_dispatches;
        record.completed_advection_y_dispatches                   = completed_advection_y_dispatches;
        record.completed_advection_pre_vertical_batches           = completed_advection_pre_vertical_batches;
        record.completed_advection_post_vertical_batches          = completed_advection_post_vertical_batches;
        record.completed_vertical_flux_dispatches                 = completed_vertical_flux_dispatches;
        record.completed_conservative_cgrid_batches               = completed_conservative_cgrid_batches;
        record.completed_conservative_cgrid_scalars               = completed_conservative_cgrid_scalars;
        record.conservative_cgrid_resident_reuses                 = conservative_cgrid_resident_reuses;
        record.completed_state_relaxation_dispatches              = completed_state_relaxation_dispatches;
        record.completed_slow_apply_dispatches                    = completed_slow_apply_dispatches;
        record.dynamics_state_resident_reuses                     = dynamics_state_resident_reuses;
        record.completed_cartesian_cgrid_slow_tendency_dispatches = completed_cartesian_cgrid_slow_tendency_dispatches;
        record.slow_tendency_invariant_input_uploads              = slow_tendency_invariant_input_uploads;
        record.slow_tendency_invariant_input_reuses               = slow_tendency_invariant_input_reuses;
        record.slow_apply_resident_tendency_reuses                = slow_apply_resident_tendency_reuses;
        record.completed_acoustic_pressure_dispatches             = completed_acoustic_pressure_dispatches;
        record.completed_acoustic_momentum_dispatches             = completed_acoustic_momentum_dispatches;
        record.completed_acoustic_fused_dispatches                = completed_acoustic_fused_dispatches;
        record.acoustic_invariant_uploads                         = acoustic_invariant_uploads;
        record.acoustic_invariant_reuses                          = acoustic_invariant_reuses;
        record.completed_acoustic_batched_dispatches              = completed_acoustic_batched_dispatches;
        record.completed_acoustic_boundary_face_dispatches        = completed_acoustic_boundary_face_dispatches;
        record.completed_acoustic_mass_flux_exports               = completed_acoustic_mass_flux_exports;
        record.completed_acoustic_mass_flux_pressure_dispatches   = completed_acoustic_mass_flux_pressure_dispatches;
        record.completed_cartesian_agrid_tendency_dispatches      = completed_cartesian_agrid_tendency_dispatches;
        record.completed_acoustic_agrid_pressure_dispatches       = completed_acoustic_agrid_pressure_dispatches;
        record.completed_acoustic_agrid_momentum_dispatches       = completed_acoustic_agrid_momentum_dispatches;
        record.completed_acoustic_agrid_fused_dispatches          = completed_acoustic_agrid_fused_dispatches;
        record.completed_acoustic_agrid_batched_dispatches        = completed_acoustic_agrid_batched_dispatches;
        record.completed_acoustic_agrid_continuity_dispatches     = completed_acoustic_agrid_continuity_dispatches;
        record.completed_acoustic_agrid_pressure_eos_dispatches   = completed_acoustic_agrid_pressure_eos_dispatches;
        record.completed_acoustic_agrid_native_momentum_dispatches =
            completed_acoustic_agrid_native_momentum_dispatches;
        record.completed_acoustic_agrid_constraint_dispatches = completed_acoustic_agrid_constraint_dispatches;
        record.completed_acoustic_agrid_boundary_dispatches   = completed_acoustic_agrid_boundary_dispatches;
        record.last_device_milliseconds                       = last_device_ms;
        record.total_device_milliseconds                      = total_device_ms;
        record.total_submit_wait_milliseconds                 = total_submit_wait_ms;
        record.timestamped_command_buffers                    = timestamped_command_buffers;
    }
};

} // namespace nimbus::metal::detail
