#pragma once

/**
 * @file metal_compute_backend.hpp
 * @brief Native Metal compute backend for Nimbus.
 *
 * The implementation remains independently qualifiable from metal/ while the
 * shared runtime factory exposes it only in native macOS production builds.
 */

#include "compute/backend_services.hpp"
#include "compute/compute_backend.hpp"
#include "compute/metal/metal_backend_evidence.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace nimbus::metal
{

class MetalComputeBackend final : public ComputeBackend,
                                  public nimbus::compute::DeviceContext,
                                  public nimbus::compute::BufferAllocator,
                                  public nimbus::compute::PipelineCache,
                                  public nimbus::compute::CommandSubmission,
                                  public nimbus::compute::TimestampProvider,
                                  public AdvectionComputeKernels,
                                  public DynamicsComputeKernels,
                                  public MicrophysicsComputeKernels,
                                  public TurbulenceComputeKernels,
                                  public DiffusionComputeKernels,
                                  public BoundaryLayerComputeKernels,
                                  public RadiationComputeKernels,
                                  public TerrainComputeKernels
{
public:
    MetalComputeBackend();
    ~MetalComputeBackend() override;

    MetalComputeBackend(const MetalComputeBackend&)            = delete;
    MetalComputeBackend& operator=(const MetalComputeBackend&) = delete;

    // ComputeBackend base contract plus directly qualified capability families.
    std::string             name() const override;
    void                    configure(const ComputeBackendConfig& config) override;
    bool                    initialize(std::string& error) override;
    void                    shutdown() override;
    void                    populate_hardware_info(HardwareInfo& info) const override;
    ComputeMemoryProperties memory_properties() const override;
    std::uint64_t           successful_dispatch_count() const override;
    std::uint64_t           successful_kernel_dispatch_count(nimbus::compute::KernelId kernel) const override;
    bool                    supports_homogeneous_domain_field_shift() const override;
    bool dispatch_homogeneous_domain_field_shift(const HomogeneousDomainFieldShiftDispatch& dispatch) override;
    bool dispatch_homogeneous_domain_field_shift_batch(const HomogeneousDomainFieldShiftBatchDispatch& batch) override;
    bool supports_earth_control_evolution() const override;
    bool dispatch_earth_control_evolution(const EarthControlEvolutionDispatch& dispatch) override;

    AdvectionComputeKernels*       advection_kernels() override;
    const AdvectionComputeKernels* advection_kernels() const override;
    bool                           supports(AdvectionKernelOp op) const override;
    bool                           supports_advection_x() const noexcept;
    bool                           supports_advection_y() const noexcept;
    bool                           supports_advection_pre_vertical_batch() const noexcept;
    bool                           supports_vertical_flux_dispatch() const noexcept;
    bool                           supports_conservative_cgrid_transport() const noexcept;
    bool                           supports_supplied_cartesian_cgrid_mass_flux() const override;
    bool                           dispatch_advection_x(const ScalarAdvectionDispatch& dispatch) override;
    bool                           dispatch_advection_y(const ScalarAdvectionDispatch& dispatch) override;
    bool                           dispatch_vertical_flux(const VerticalFluxDispatch& dispatch) override;
    bool                           dispatch_advect_scalars_conservative_cartesian_cgrid(
        const ConservativeCartesianCgridTransportDispatch& dispatch) override;
    bool dispatch_advection_batch_pre_vertical(const AdvectionBatchPreVerticalDispatch& dispatch) override;
    bool dispatch_advection_batch_post_vertical(const AdvectionBatchPostVerticalDispatch& dispatch) override;

    DynamicsComputeKernels*       dynamics_kernels() override;
    const DynamicsComputeKernels* dynamics_kernels() const override;
    bool                          supports(DynamicsKernelOp op) const override;
    bool                          dispatch_state_relaxation_cartesian(const StateRelaxationDispatch& dispatch) override;
    bool                          dispatch_slow_apply(const SlowApplyDispatch& dispatch) override;
    bool dispatch_cartesian_cgrid_slow_tendencies(const CartesianCgridSlowTendenciesDispatch& dispatch) override;
    bool dispatch_cartesian_tendencies(const CartesianTendenciesDispatch& dispatch) override;
    bool dispatch_acoustic_continuity_mass_flux_cartesian_agrid(
        const AcousticContinuityMassFluxCartesianAgridDispatch& dispatch) override;
    bool
    dispatch_acoustic_pressure_eos_cartesian_agrid(const AcousticPressureEosCartesianAgridDispatch& dispatch) override;
    bool dispatch_acoustic_momentum_cartesian_agrid(const AcousticMomentumCartesianAgridDispatch& dispatch) override;
    bool dispatch_acoustic_null_mode_constraint_cartesian_agrid(
        const AcousticNullModeConstraintCartesianAgridDispatch& dispatch) override;
    bool dispatch_acoustic_boundary_cartesian_agrid(const AcousticBoundaryCartesianAgridDispatch& dispatch) override;
    bool dispatch_acoustic_pressure(const AcousticPressureDispatch& dispatch) override;
    bool dispatch_acoustic_momentum(const AcousticMomentumDispatch& dispatch) override;
    bool dispatch_acoustic_substep_fused(const AcousticSubstepFusedDispatch& dispatch) override;
    bool dispatch_acoustic_substeps_batched(const AcousticSubstepsBatchedDispatch& dispatch) override;
    bool dispatch_acoustic_pressure_cartesian_cgrid(const AcousticPressureCartesianCgridDispatch& dispatch) override;
    bool dispatch_acoustic_momentum_cartesian_cgrid(const AcousticMomentumCartesianCgridDispatch& dispatch) override;
    bool dispatch_acoustic_substep_fused_cartesian_cgrid(
        const AcousticSubstepFusedCartesianCgridDispatch& dispatch) override;
    bool supports_unrepaired_acoustic_fused() const override;
    bool supports_batched_cartesian_cgrid_mass_flux_export() const override;
    bool dispatch_acoustic_substeps_batched_cartesian_cgrid(
        const AcousticSubstepsBatchedCartesianCgridDispatch& dispatch) override;
    void release_acoustic_loop_residency() override;

    TurbulenceComputeKernels*       turbulence_kernels() override;
    const TurbulenceComputeKernels* turbulence_kernels() const override;
    bool                            supports(TurbulenceKernelOp op) const override;
    bool dispatch_tke_closure(const TkeClosureDispatch&) override;
    bool dispatch_tke_tendency(const TkeTendencyDispatch&) override;

    void begin_resident_step() override;
    void invalidate_resident_field(const char* field_id) noexcept override;
    void invalidate_all_resident_fields() noexcept override;

    // The measured Smagorinsky coefficient phase is directly testable, but
    // full TurbulenceKernelOp::Smagorinsky support remains unadvertised until
    // the dependent conservative tendency phase is native too.
    bool supports_smagorinsky_coefficients() const noexcept;
    bool dispatch_smagorinsky_coefficients(const SmagorinskyCoefficientsDispatch& dispatch) override;
    bool supports_smagorinsky_tendencies() const noexcept;
    bool dispatch_smagorinsky_tendencies(const SmagorinskyTendenciesDispatch& dispatch) override;
    bool supports_smagorinsky_fused() const noexcept;
    bool dispatch_smagorinsky_fused(const SmagorinskyFusedDispatch& dispatch) override;
    bool supports_sgs_apply() const noexcept;
    bool dispatch_sgs_apply(const SgsApplyDispatch& dispatch) override;

    DiffusionComputeKernels*       diffusion_kernels() override;
    const DiffusionComputeKernels* diffusion_kernels() const override;
    bool                           supports(DiffusionKernelOp op) const override;
    bool                           dispatch_implicit_stencil(const ImplicitDiffusionStencilDispatch&) override;
    bool                           dispatch_implicit_vector(const ImplicitDiffusionVectorDispatch&) override;
    bool                           supports_diffusion_dispatch() const noexcept;
    bool                           dispatch_diffusion(const DiffusionDispatch& dispatch) override;
    bool                           dispatch_diffusion_batch(const DiffusionBatchDispatch& dispatch) override;

    BoundaryLayerComputeKernels*       boundary_layer_kernels() override;
    const BoundaryLayerComputeKernels* boundary_layer_kernels() const override;
    bool                               supports(BoundaryLayerKernelOp op) const override;
    bool dispatch_mynn_level25_stability(const MynnLevel25StabilityDispatch& dispatch) override;

    RadiationComputeKernels*       radiation_kernels() override;
    const RadiationComputeKernels* radiation_kernels() const override;
    bool                           supports(RadiationKernelOp op) const override;
    bool dispatch_rrtmg_longwave_transfer(const RrtmgLongwaveTransferDispatch& dispatch) override;
    bool dispatch_rrtmg_shortwave_transfer(const RrtmgShortwaveTransferDispatch& dispatch) override;
    bool dispatch_rrtmg_longwave_transfer_batch(const RrtmgLongwaveTransferBatchDispatch& dispatch) override;
    bool dispatch_rrtmg_shortwave_transfer_batch(const RrtmgShortwaveTransferBatchDispatch& dispatch) override;

    TerrainComputeKernels*       terrain_kernels() override;
    const TerrainComputeKernels* terrain_kernels() const override;
    bool supports_mapped_mass_continuity() const override;
    bool dispatch_mapped_mass_continuity(const MappedTerrainMassContinuityDispatch& dispatch) override;
    bool supports_mapped_fast_momentum() const override;
    bool dispatch_mapped_fast_momentum(const MappedTerrainFastMomentumDispatch& dispatch) override;
    bool supports_mapped_slow_tendencies() const override;
    bool dispatch_mapped_slow_tendencies(const MappedTerrainSlowTendenciesDispatch& dispatch) override;
    bool supports_mapped_scalar_transport() const override;
    bool dispatch_mapped_scalar_transport(const MappedTerrainScalarTransportDispatch& dispatch) override;
    bool supports_mapped_final_state() const override;
    bool dispatch_mapped_final_state(const MappedTerrainFinalStateDispatch& dispatch) override;
    bool                         supports_mapped_state_update() const override;
    bool dispatch_mapped_state_update(const MappedTerrainStateUpdateDispatch& dispatch) override;
    bool supports_mapped_flux_accumulation() const override;
    bool dispatch_mapped_flux_accumulation(const MappedTerrainFluxAccumulationDispatch& dispatch) override;

    MicrophysicsComputeKernels*       microphysics_kernels() override;
    const MicrophysicsComputeKernels* microphysics_kernels() const override;
    bool                              supports(MicrophysicsKernelOp op) const override;
    bool                              supports_kessler_pointwise() const noexcept;
    bool                              supports_kessler_sedimentation() const noexcept;
    bool                              dispatch_kessler_pointwise(const KesslerPointwiseDispatch& dispatch) override;
    bool dispatch_kessler_sedimentation(const KesslerSedimentationDispatch& dispatch) override;
    bool supports_kessler_fused() const noexcept;
    bool dispatch_kessler_fused(const MetalKesslerFusedDispatch& dispatch);
    bool supports_lin_pointwise() const noexcept;
    bool supports_lin_sedimentation() const noexcept;
    bool dispatch_thompson_cloud_phase(const ThompsonCloudPhaseDispatch&) override;
    bool dispatch_thompson_freezing(const ThompsonFreezingDispatch&) override;
    bool dispatch_thompson_deposition(const ThompsonDepositionDispatch&) override;
    bool dispatch_thompson_completion(const ThompsonCompletionDispatch&) override;
    bool dispatch_milbrandt_melting(const MilbrandtMeltingDispatch& d) override;
    bool dispatch_milbrandt_cold(const MilbrandtColdDispatch& d) override;
    bool dispatch_milbrandt_sedimentation(const MilbrandtSedimentationDispatch& d) override;
    bool dispatch_cgrid_boundary(const CgridBoundaryDispatch&) override;
    bool dispatch_slab_column(const SlabColumnDispatch&) override;
    bool dispatch_simple_grey(const SimpleGreyDispatch&) override;
    bool supports_held_sources() const override;
    bool dispatch_held_sources(const HeldSourcesDispatch&) override;
    bool supports_column_remap() const override;
    bool dispatch_column_remap(const ColumnRemapDispatch&) override;
    bool supports_reference_atmosphere() const override;
    bool dispatch_reference_atmosphere(const ReferenceAtmosphereDispatch&) override;
    bool supports_moving_energy() const override;
    bool dispatch_moving_energy(const MovingEnergyDispatch&) override;
    bool dispatch_conservative_reduction(const ConservativeReductionDispatch&) override;
    bool dispatch_conservative_momentum(const ConservativeMomentumDispatch&) override;
    bool supports_moving_momentum() const override;
    bool dispatch_moving_momentum(const MovingMomentumDispatch&) override;
    bool supports_earth_geometry() const override;
    bool dispatch_earth_geometry(const EarthGeometryDispatch&) override;
    bool supports_conservative_remap() const override;
    bool dispatch_conservative_remap(const ConservativeRemapDispatch& d) override;
    bool dispatch_collocated_sgs(const CollocatedSgsDispatch& d) override;
    bool supports_rk_tendency() const override;
    RkCombinationStatus dispatch_rk_tendency(const RkTendencyDispatch& d) override;
    bool                supports_rk_combination() const override;
    RkCombinationStatus dispatch_rk_combination(const RkCombinationDispatch& d) override;
    bool dispatch_milbrandt_cloud_adjustment(const MilbrandtCloudAdjustmentDispatch&) override;
    bool dispatch_milbrandt_warm_rain(const MilbrandtWarmRainDispatch&) override;
    bool dispatch_thompson_warm_rain(const ThompsonWarmRainDispatch&) override;
    bool dispatch_thompson_cloud_adjustment(const ThompsonCloudAdjustmentDispatch&) override;
    bool dispatch_thompson_moment_sedimentation(const ThompsonMomentSedimentationDispatch&) override;
    bool dispatch_lin_pointwise(const LinPointwiseDispatch& dispatch) override;
    bool dispatch_lin_sedimentation(const LinSedimentationDispatch& dispatch) override;
    bool dispatch_lin_fused(const LinFusedDispatch& dispatch) override;
    bool supports_atomic_lin_fused() const override;
    bool supports_microphysics_apply() const noexcept;
    bool dispatch_microphysics_apply(const MicrophysicsApplyDispatch& dispatch) override;

    // backend_services.hpp contracts, directly inspectable by Metal tests.
    const char* api_name() const noexcept override;
    const char* device_name() const noexcept override;
    bool        available() const noexcept override;

    nimbus::compute::BackendAllocation allocate(std::size_t bytes) override;
    void                               release(nimbus::compute::BackendAllocation allocation) override;

    nimbus::compute::BackendPipeline pipeline(nimbus::compute::KernelId kernel) const noexcept override;
    bool                             ready(nimbus::compute::KernelId kernel) const noexcept override;

    // Submits and waits for an empty bootstrap command buffer. A bare graph
    // has hazards but no resource/dimension/constants bindings, so non-empty
    // submissions fail closed. Bound fused paths above consume validated graph
    // schedules while retaining their typed dispatch contracts.
    bool submit(const nimbus::compute::DispatchGraph& graph) override;

    bool   supported() const noexcept override;
    double last_device_milliseconds() const noexcept override;

    bool                  last_command_buffer_completed() const noexcept;
    std::uint64_t         completed_command_buffer_count() const noexcept;
    std::size_t           live_shared_allocation_count() const noexcept;
    std::size_t           live_shared_allocation_bytes() const noexcept;
    std::size_t           resident_field_count() const noexcept;
    std::uint64_t         resident_upload_count() const noexcept override;
    std::uint64_t         resident_reuse_count() const noexcept override;
    std::uint64_t         completed_shift_batch_count() const noexcept;
    bool                  run_domain_shift_storage_benchmark(const MetalStorageBenchmarkConfig& config,
                                                             MetalStorageBenchmarkResult&       result);
    MetalManifestCoverage manifest_coverage() const;
    MetalExecutionRecord  execution_record() const;

    static const char* canonical_kernel_manifest_hash() noexcept;
    static const char* metal_artifact_source_hash(nimbus::compute::KernelId kernel) noexcept;

    static const char* domain_shift_msl_source() noexcept;
    static const char* domain_shift_msl_source_hash() noexcept;
    static const char* earth_control_evolution_msl_source() noexcept;
    static const char* earth_control_evolution_msl_source_hash() noexcept;
    static const char* smagorinsky_coefficients_msl_source() noexcept;
    static const char* smagorinsky_coefficients_msl_source_hash() noexcept;
    static const char* smagorinsky_tendencies_msl_source() noexcept;
    static const char* smagorinsky_tendencies_msl_source_hash() noexcept;
    static const char* sgs_apply_msl_source() noexcept;
    static const char* sgs_apply_msl_source_hash() noexcept;
    static const char* diffusion_msl_source() noexcept;
    static const char* diffusion_msl_source_hash() noexcept;
    static const char* kessler_pointwise_msl_source() noexcept;
    static const char* kessler_pointwise_msl_source_hash() noexcept;
    static const char* kessler_sedimentation_msl_source() noexcept;
    static const char* kessler_sedimentation_msl_source_hash() noexcept;
    static const char* lin_pointwise_msl_source() noexcept;
    static const char* lin_pointwise_msl_source_hash() noexcept;
    static const char* lin_sedimentation_msl_source() noexcept;
    static const char* lin_sedimentation_msl_source_hash() noexcept;
    static const char* microphysics_apply_msl_source() noexcept;
    static const char* microphysics_apply_msl_source_hash() noexcept;
    static const char* advection_x_msl_source() noexcept;
    static const char* advection_x_msl_source_hash() noexcept;
    static const char* advection_y_msl_source() noexcept;
    static const char* advection_y_msl_source_hash() noexcept;
    static const char* tvd_vertical_flux_msl_source() noexcept;
    static const char* tvd_vertical_flux_msl_source_hash() noexcept;
    static const char* conservative_cgrid_transport_msl_source() noexcept;
    static const char* conservative_cgrid_transport_msl_source_hash() noexcept;
    static const char* dynamics_apply_msl_source() noexcept;
    static const char* dynamics_apply_msl_source_hash() noexcept;
    static const char* cartesian_cgrid_slow_tendencies_msl_source() noexcept;
    static const char* cartesian_cgrid_slow_tendencies_msl_source_hash() noexcept;
    static const char* acoustic_cartesian_cgrid_msl_source() noexcept;
    static const char* acoustic_cartesian_cgrid_msl_source_hash() noexcept;
    static const char* acoustic_batched_cartesian_cgrid_msl_source() noexcept;
    static const char* acoustic_batched_cartesian_cgrid_msl_source_hash() noexcept;
    static const char* cartesian_agrid_msl_source() noexcept;
    static const char* cartesian_agrid_msl_source_hash() noexcept;
    static const char* acoustic_agrid_msl_source() noexcept;
    static const char* acoustic_agrid_msl_source_hash() noexcept;
    static const char* acoustic_agrid_continuity_msl_source() noexcept;
    static const char* acoustic_agrid_continuity_msl_source_hash() noexcept;
    static const char* mapped_state_flux_msl_source() noexcept;
    static const char* mapped_state_flux_msl_source_hash() noexcept;
    static const char* mapped_acoustic_msl_source() noexcept;
    static const char* mapped_acoustic_msl_source_hash() noexcept;
    static const char* mapped_slow_tendencies_msl_source() noexcept;
    static const char* mapped_slow_tendencies_msl_source_hash() noexcept;
    static const char* mapped_scalar_transport_msl_source() noexcept;
    static const char* mapped_scalar_transport_msl_source_hash() noexcept;
    static const char* mapped_final_state_msl_source() noexcept;
    static const char* mapped_final_state_msl_source_hash() noexcept;
    static const char* mynn_level25_stability_msl_source() noexcept;
    static const char* mynn_level25_stability_msl_source_hash() noexcept;
    static const char* rrtmg_longwave_transfer_msl_source() noexcept;
    static const char* rrtmg_longwave_transfer_msl_source_hash() noexcept;
    static const char* rrtmg_shortwave_transfer_msl_source() noexcept;
    static const char* rrtmg_shortwave_transfer_msl_source_hash() noexcept;
    static const char* rrtmg_longwave_transfer_batch_msl_source() noexcept;
    static const char* rrtmg_longwave_transfer_batch_msl_source_hash() noexcept;
    static const char* rrtmg_shortwave_transfer_batch_msl_source() noexcept;
    static const char* rrtmg_shortwave_transfer_batch_msl_source_hash() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nimbus::metal
