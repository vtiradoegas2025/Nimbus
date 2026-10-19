// Snapshot of device identity, timings, dispatches, and source hashes.
MetalExecutionRecord MetalComputeBackend::execution_record() const
{
    MetalExecutionRecord        record;
    const MetalManifestCoverage coverage = manifest_coverage();
    record.api_name                      = api_name();
    record.device_name                   = impl_->hardware.name;
    record.operating_system              = impl_->hardware.operating_system;
    record.host_toolchain                = __clang_version__;
    const auto source_hash               = [&](nimbus::compute::KernelId kernel, const char* fallback)
    {
        const std::string& registered = impl_->pipelines.source_hash(kernel);
        return registered.empty() ? std::string(fallback) : registered;
    };
    using nimbus::compute::KernelId;
    record.msl_source_hash =
        source_hash(KernelId::DomainRecyclingHomogeneousFieldShift, domain_shift_msl_source_hash());
    record.smagorinsky_coefficients_msl_source_hash =
        source_hash(KernelId::SmagorinskyCoefficients, smagorinsky_coefficients_msl_source_hash());
    record.smagorinsky_tendencies_msl_source_hash =
        source_hash(KernelId::SmagorinskyTendencies, smagorinsky_tendencies_msl_source_hash());
    record.sgs_apply_msl_source_hash = source_hash(KernelId::SgsApply, sgs_apply_msl_source_hash());
    record.diffusion_msl_source_hash = source_hash(KernelId::Diffusion, diffusion_msl_source_hash());
    record.kessler_pointwise_msl_source_hash =
        source_hash(KernelId::KesslerPointwise, kessler_pointwise_msl_source_hash());
    record.kessler_sedimentation_msl_source_hash =
        source_hash(KernelId::KesslerSedimentation, kessler_sedimentation_msl_source_hash());
    record.lin_pointwise_msl_source_hash = source_hash(KernelId::LinPointwise, lin_pointwise_msl_source_hash());
    record.lin_sedimentation_msl_source_hash =
        source_hash(KernelId::LinSedimentation, lin_sedimentation_msl_source_hash());
    record.microphysics_apply_msl_source_hash =
        source_hash(KernelId::MicrophysicsApply, microphysics_apply_msl_source_hash());
    record.advection_x_msl_source_hash = source_hash(KernelId::AdvectX, advection_x_msl_source_hash());
    record.advection_y_msl_source_hash = source_hash(KernelId::AdvectY, advection_y_msl_source_hash());
    record.tvd_vertical_flux_msl_source_hash =
        source_hash(KernelId::TvdVerticalFlux, tvd_vertical_flux_msl_source_hash());
    record.conservative_cgrid_transport_msl_source_hash =
        source_hash(KernelId::AdvectConservativeMassFluxCartesianCgrid, conservative_cgrid_transport_msl_source_hash());
    record.dynamics_apply_msl_source_hash =
        source_hash(KernelId::StateRelaxationCartesian, dynamics_apply_msl_source_hash());
    record.cartesian_cgrid_slow_tendencies_msl_source_hash =
        source_hash(KernelId::CartesianCgridSlowTendencies, cartesian_cgrid_slow_tendencies_msl_source_hash());
    record.acoustic_cartesian_cgrid_msl_source_hash =
        source_hash(KernelId::AcousticPressureCartesianCgrid, acoustic_cartesian_cgrid_msl_source_hash());
    record.acoustic_batched_cartesian_cgrid_msl_source_hash = source_hash(
        KernelId::AcousticPressureMassFluxCartesianCgrid, acoustic_batched_cartesian_cgrid_msl_source_hash());
    record.cartesian_agrid_msl_source_hash =
        source_hash(KernelId::CartesianTendencies, cartesian_agrid_msl_source_hash());
    record.acoustic_agrid_msl_source_hash =
        source_hash(KernelId::AcousticPressureCartesian, acoustic_agrid_msl_source_hash());
    record.acoustic_agrid_continuity_msl_source_hash = source_hash(
        KernelId::AcousticContinuityMassFluxCartesianAgrid,
        acoustic_agrid_continuity_msl_source_hash());
    record.acoustic_agrid_pressure_eos_msl_source_hash = source_hash(
        KernelId::AcousticPressureEosCartesianAgrid,
        metal_artifact_source_hash(KernelId::AcousticPressureEosCartesianAgrid));
    record.acoustic_agrid_native_momentum_msl_source_hash = source_hash(
        KernelId::AcousticMomentumCartesianAgrid,
        metal_artifact_source_hash(KernelId::AcousticMomentumCartesianAgrid));
    record.acoustic_agrid_constraint_msl_source_hash = source_hash(
        KernelId::AcousticNullModeConstraintCartesianAgrid,
        metal_artifact_source_hash(KernelId::AcousticNullModeConstraintCartesianAgrid));
    record.acoustic_agrid_boundary_msl_source_hash = source_hash(
        KernelId::AcousticBoundaryCartesianAgrid,
        metal_artifact_source_hash(KernelId::AcousticBoundaryCartesianAgrid));
    record.canonical_kernel_manifest_hash        = coverage.canonical_manifest_hash;
    record.manifest_kernels                      = coverage.manifest_kernels;
    record.active_manifest_kernels               = coverage.active_kernels;
    record.ready_active_manifest_kernels         = coverage.ready_active_kernels;
    record.hardware_limited_manifest_kernels     = coverage.hardware_limited_kernels;
    record.deferred_manifest_kernels             = coverage.deferred_kernels;
    record.orphan_manifest_kernels               = coverage.orphan_kernels;
    record.unexpected_available_manifest_kernels = coverage.unexpected_available_kernels;
    record.missing_active_artifact_hashes        = coverage.missing_artifact_hashes;
    record.manifest_complete                     = coverage.complete;
    record.unified_memory                        = impl_->hardware.unified_memory;
    impl_->telemetry.write_to(record);
    return record;
}
