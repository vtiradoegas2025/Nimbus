// Stable public accessors backed by the single private source catalog.
const char* MetalComputeBackend::domain_shift_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::DomainRecyclingHomogeneousFieldShift);
}
const char* MetalComputeBackend::domain_shift_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::DomainRecyclingHomogeneousFieldShift);
}
const char* MetalComputeBackend::earth_control_evolution_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::EarthControlEvolution);
}
const char* MetalComputeBackend::earth_control_evolution_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::EarthControlEvolution);
}
const char* MetalComputeBackend::smagorinsky_coefficients_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::SmagorinskyCoefficients);
}
const char* MetalComputeBackend::smagorinsky_coefficients_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::SmagorinskyCoefficients);
}
const char* MetalComputeBackend::smagorinsky_tendencies_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::SmagorinskyTendencies);
}
const char* MetalComputeBackend::smagorinsky_tendencies_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::SmagorinskyTendencies);
}
const char* MetalComputeBackend::sgs_apply_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::SgsApply);
}
const char* MetalComputeBackend::sgs_apply_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::SgsApply);
}
const char* MetalComputeBackend::diffusion_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::Diffusion);
}
const char* MetalComputeBackend::diffusion_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::Diffusion);
}
const char* MetalComputeBackend::kessler_pointwise_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::KesslerPointwise);
}
const char* MetalComputeBackend::kessler_pointwise_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::KesslerPointwise);
}
const char* MetalComputeBackend::kessler_sedimentation_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::KesslerSedimentation);
}
const char* MetalComputeBackend::kessler_sedimentation_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::KesslerSedimentation);
}
const char* MetalComputeBackend::lin_pointwise_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::LinPointwise);
}
const char* MetalComputeBackend::lin_pointwise_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::LinPointwise);
}
const char* MetalComputeBackend::lin_sedimentation_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::LinSedimentation);
}
const char* MetalComputeBackend::lin_sedimentation_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::LinSedimentation);
}
const char* MetalComputeBackend::microphysics_apply_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::MicrophysicsApply);
}
const char* MetalComputeBackend::microphysics_apply_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::MicrophysicsApply);
}
const char* MetalComputeBackend::advection_x_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::AdvectX);
}
const char* MetalComputeBackend::advection_x_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::AdvectX);
}
const char* MetalComputeBackend::advection_y_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::AdvectY);
}
const char* MetalComputeBackend::advection_y_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::AdvectY);
}
const char* MetalComputeBackend::tvd_vertical_flux_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::TvdVerticalFlux);
}
const char* MetalComputeBackend::tvd_vertical_flux_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::TvdVerticalFlux);
}
const char* MetalComputeBackend::conservative_cgrid_transport_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::AdvectConservativeMassFluxCartesianCgrid);
}
const char* MetalComputeBackend::conservative_cgrid_transport_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::AdvectConservativeMassFluxCartesianCgrid);
}
const char* MetalComputeBackend::dynamics_apply_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::StateRelaxationCartesian);
}
const char* MetalComputeBackend::dynamics_apply_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::StateRelaxationCartesian);
}
const char* MetalComputeBackend::cartesian_cgrid_slow_tendencies_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::CartesianCgridSlowTendencies);
}
const char* MetalComputeBackend::cartesian_cgrid_slow_tendencies_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::CartesianCgridSlowTendencies);
}
const char* MetalComputeBackend::acoustic_cartesian_cgrid_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::AcousticPressureCartesianCgrid);
}
const char* MetalComputeBackend::acoustic_cartesian_cgrid_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::AcousticPressureCartesianCgrid);
}
const char* MetalComputeBackend::acoustic_batched_cartesian_cgrid_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::AcousticPressureMassFluxCartesianCgrid);
}
const char* MetalComputeBackend::acoustic_batched_cartesian_cgrid_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::AcousticPressureMassFluxCartesianCgrid);
}
const char* MetalComputeBackend::cartesian_agrid_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::CartesianTendencies);
}
const char* MetalComputeBackend::cartesian_agrid_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::CartesianTendencies);
}
const char* MetalComputeBackend::acoustic_agrid_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::AcousticPressureCartesian);
}
const char* MetalComputeBackend::acoustic_agrid_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::AcousticPressureCartesian);
}
const char* MetalComputeBackend::acoustic_agrid_continuity_msl_source() noexcept
{
    return detail::metal_source_for(
        KernelId::AcousticContinuityMassFluxCartesianAgrid);
}
const char* MetalComputeBackend::acoustic_agrid_continuity_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(
        KernelId::AcousticContinuityMassFluxCartesianAgrid);
}
const char* MetalComputeBackend::mapped_state_flux_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::MappedTerrainStateUpdate);
}
const char* MetalComputeBackend::mapped_state_flux_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::MappedTerrainStateUpdate);
}
const char* MetalComputeBackend::mapped_acoustic_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::MappedTerrainMassContinuity);
}
const char* MetalComputeBackend::mapped_acoustic_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::MappedTerrainMassContinuity);
}
const char* MetalComputeBackend::mapped_slow_tendencies_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::MappedTerrainSlowTendencies);
}
const char* MetalComputeBackend::mapped_slow_tendencies_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::MappedTerrainSlowTendencies);
}
const char* MetalComputeBackend::mapped_scalar_transport_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::MappedTerrainScalarUpdate);
}
const char* MetalComputeBackend::mapped_scalar_transport_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::MappedTerrainScalarUpdate);
}
const char* MetalComputeBackend::mapped_final_state_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::MappedTerrainEos);
}
const char* MetalComputeBackend::mapped_final_state_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::MappedTerrainEos);
}
const char* MetalComputeBackend::mynn_level25_stability_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::MynnLevel25Stability);
}
const char* MetalComputeBackend::mynn_level25_stability_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::MynnLevel25Stability);
}
const char* MetalComputeBackend::rrtmg_longwave_transfer_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::RrtmgLongwaveTransfer);
}
const char* MetalComputeBackend::rrtmg_longwave_transfer_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::RrtmgLongwaveTransfer);
}
const char* MetalComputeBackend::rrtmg_shortwave_transfer_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::RrtmgShortwaveTransfer);
}
const char* MetalComputeBackend::rrtmg_shortwave_transfer_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::RrtmgShortwaveTransfer);
}
const char* MetalComputeBackend::rrtmg_longwave_transfer_batch_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::RrtmgLongwaveTransferBatch);
}
const char* MetalComputeBackend::rrtmg_longwave_transfer_batch_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::RrtmgLongwaveTransferBatch);
}
const char* MetalComputeBackend::rrtmg_shortwave_transfer_batch_msl_source() noexcept
{
    return detail::metal_source_for(KernelId::RrtmgShortwaveTransferBatch);
}
const char* MetalComputeBackend::rrtmg_shortwave_transfer_batch_msl_source_hash() noexcept
{
    return detail::metal_source_hash_for(KernelId::RrtmgShortwaveTransferBatch);
}
