#pragma once
#include "compute/metal/shaders/dynamics/conservative_reduction_msl.hpp"
#include "compute/metal/shaders/dynamics/conservative_momentum_msl.hpp"

#include "compute/kernel_manifest.hpp"

#include "compute/metal/shaders/acoustics/acoustic_batched_cartesian_cgrid_msl.hpp"
#include "compute/metal/shaders/dynamics/acoustic_bc_face_cartesian_cgrid_msl.hpp"
#include "compute/metal/shaders/acoustics/acoustic_cartesian_cgrid_msl.hpp"
#include "compute/metal/shaders/acoustics/acoustic_continuity_massflux_cartesian_agrid_msl.hpp"
#include "compute/metal/shaders/acoustics/acoustic_complete_cartesian_agrid_msl.hpp"
#include "compute/metal/shaders/advection/advection_xy_msl.hpp"
#include "compute/metal/shaders/advection/conservative_cgrid_transport_msl.hpp"
#include "compute/metal/shaders/advection/tvd_vertical_flux_msl.hpp"
#include "compute/metal/shaders/diffusion/diffusion_msl.hpp"
#include "compute/metal/shaders/diffusion/implicit_diffusion_stencil_msl.hpp"
#include "compute/metal/shaders/diffusion/implicit_diffusion_vector_msl.hpp"
#include "compute/metal/shaders/domain/domain_recycling_homogeneous_field_msl.hpp"
#include "compute/metal/shaders/dynamics/cartesian_agrid_msl.hpp"
#include "compute/metal/shaders/dynamics/cartesian_cgrid_slow_tendencies_msl.hpp"
#include "compute/metal/shaders/dynamics/dynamics_apply_msl.hpp"
#include "compute/metal/shaders/microphysics/kessler_msl.hpp"
#include "compute/metal/shaders/microphysics/kessler_sedimentation_msl.hpp"
#include "compute/metal/shaders/microphysics/lin_msl.hpp"
#include "compute/metal/shaders/microphysics/thompson_moment_sedimentation_msl.hpp"
#include "compute/metal/shaders/microphysics/thompson_cloud_adjustment_msl.hpp"
#include "compute/metal/shaders/microphysics/thompson_warm_rain_msl.hpp"
#include "compute/metal/shaders/microphysics/milbrandt_warm_rain_msl.hpp"
#include "compute/metal/shaders/microphysics/milbrandt_cloud_adjustment_msl.hpp"
#include "compute/metal/shaders/microphysics/milbrandt_cold_msl.hpp"
#include "compute/metal/shaders/time_stepping/rk_combination_msl.hpp"
#include "compute/metal/shaders/time_stepping/rk_tendency_msl.hpp"
#include "compute/metal/shaders/turbulence/collocated_sgs_msl.hpp"
#include "compute/metal/shaders/terrain/conservative_remap_msl.hpp"
#include "compute/metal/shaders/terrain/earth_geometry_msl.hpp"
#include "compute/metal/shaders/terrain/moving_momentum_msl.hpp"
#include "compute/metal/shaders/terrain/moving_energy_msl.hpp"
#include "compute/metal/shaders/terrain/reference_atmosphere_msl.hpp"
#include "compute/metal/shaders/terrain/column_remap_msl.hpp"
#include "compute/metal/shaders/terrain/held_sources_msl.hpp"
#include "compute/metal/shaders/radiation/simple_grey_msl.hpp"
#include "compute/metal/shaders/boundary_layer/slab_column_msl.hpp"
#include "compute/metal/shaders/microphysics/milbrandt_sedimentation_msl.hpp"
#include "compute/metal/shaders/microphysics/milbrandt_melting_msl.hpp"
#include "compute/metal/shaders/microphysics/thompson_cloud_phase_msl.hpp"
#include "compute/metal/shaders/microphysics/thompson_freezing_msl.hpp"
#include "compute/metal/shaders/microphysics/thompson_deposition_msl.hpp"
#include "compute/metal/shaders/microphysics/thompson_completion_msl.hpp"
#include "compute/metal/shaders/microphysics/microphysics_apply_msl.hpp"
#include "compute/metal/shaders/turbulence/sgs_apply_msl.hpp"
#include "compute/metal/shaders/turbulence/tke_closure_msl.hpp"
#include "compute/metal/shaders/turbulence/tke_tendency_msl.hpp"
#include "compute/metal/shaders/turbulence/smagorinsky_coefficients_msl.hpp"
#include "compute/metal/shaders/turbulence/smagorinsky_tendencies_msl.hpp"
#include "compute/metal/shaders/terrain/mapped_state_flux_msl.hpp"
#include "compute/metal/shaders/terrain/mapped_acoustic_msl.hpp"
#include "compute/metal/shaders/terrain/mapped_slow_tendencies_msl.hpp"
#include "compute/metal/shaders/terrain/mapped_scalar_transport_msl.hpp"
#include "compute/metal/shaders/advection/advect_conservative_scalar_flux_cartesian_cgrid_msl.hpp"
#include "compute/metal/shaders/terrain/mapped_final_state_msl.hpp"
#include "compute/metal/shaders/terrain/mapped_boundary_msl.hpp"
#include "compute/metal/shaders/boundary_layer/mynn_level25_stability_msl.hpp"
#include "compute/metal/shaders/radiation/rrtmg_longwave_transfer_msl.hpp"
#include "compute/metal/shaders/radiation/rrtmg_shortwave_transfer_msl.hpp"
#include "compute/metal/shaders/radiation/rrtmg_longwave_transfer_batch_msl.hpp"
#include "compute/metal/shaders/radiation/rrtmg_shortwave_transfer_batch_msl.hpp"
#include "compute/metal/shaders/chaos/earth_control_evolution_msl.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace nimbus::metal::detail
{

struct MetalKernelSpec
{
    nimbus::compute::KernelId id         = nimbus::compute::KernelId::Count;
    const char*               entrypoint = nullptr;
};

struct MetalLibrarySpec
{
    const char*                    label  = nullptr;
    const char*                    source = nullptr;
    std::array<MetalKernelSpec, 3> kernels{};
    std::size_t                    kernel_count = 0;
    std::string                    source_hash;
    bool                           precise_math = false;

    MetalLibrarySpec(const char* label_value, const char* source_value, MetalKernelSpec first, bool precise = false)
        : label(label_value), source(source_value), kernels{first, {}, {}}, kernel_count(1),
          source_hash(hash(source_value)), precise_math(precise)
    {
    }

    MetalLibrarySpec(const char* label_value, const char* source_value, MetalKernelSpec first, MetalKernelSpec second, bool precise = false)
        : label(label_value), source(source_value), kernels{first, second, {}}, kernel_count(2),
          source_hash(hash(source_value)), precise_math(precise)
    {
    }

    MetalLibrarySpec(const char* label_value, const char* source_value,
                     MetalKernelSpec first, MetalKernelSpec second,
                     MetalKernelSpec third)
        : label(label_value), source(source_value), kernels{first, second, third},
          kernel_count(3), source_hash(hash(source_value))
    {
    }

    static std::string hash(const char* text)
    {
        std::uint64_t value = UINT64_C(14695981039346656037);
        if (text != nullptr)
        {
            for (const unsigned char* byte = reinterpret_cast<const unsigned char*>(text); *byte != 0; ++byte)
            {
                value ^= static_cast<std::uint64_t>(*byte);
                value *= UINT64_C(1099511628211);
            }
        }
        static constexpr char digits[] = "0123456789abcdef";
        std::string           result(16, '0');
        for (std::size_t index = result.size(); index > 0; --index)
        {
            result[index - 1] = digits[value & UINT64_C(0xf)];
            value >>= 4;
        }
        return result;
    }
};

struct MetalPipelineAlias
{
    nimbus::compute::KernelId alias;
    nimbus::compute::KernelId target;
};

inline const std::array<MetalLibrarySpec, 67>& metal_library_catalog()
{
    using nimbus::compute::KernelId;
    static const std::array<MetalLibrarySpec, 67> catalog = {{
        {"domain recycling",
         shaders::kDomainRecyclingHomogeneousFieldMsl,
         {KernelId::DomainRecyclingHomogeneousFieldShift, "domain_recycling_homogeneous_field_shift"}},
        {"Smagorinsky coefficients",
         shaders::kSmagorinskyCoefficientsMsl,
         {KernelId::SmagorinskyCoefficients, "smagorinsky_coefficients"}},
        {"Smagorinsky tendencies",
         shaders::kSmagorinskyTendenciesMsl,
         {KernelId::SmagorinskyTendencies, "smagorinsky_tendencies"}},
        {"TKE closure", shaders::kTkeClosureMsl, {KernelId::TkeClosure, "tke_closure"}},
        {"TKE tendency", shaders::kTkeTendencyMsl, {KernelId::TkeTendency, "tke_tendency"}},
        {"SGS apply", shaders::kSgsApplyMsl, {KernelId::SgsApply, "sgs_apply"}},
        {"diffusion", shaders::kDiffusionMsl, {KernelId::Diffusion, "diffusion"}},
        {"implicit diffusion stencil",
         shaders::kImplicitDiffusionStencilMsl,
         {KernelId::ImplicitDiffusionStencil, "implicit_diffusion_stencil"}},
        {"implicit diffusion vector",
         shaders::kImplicitDiffusionVectorMsl,
         {KernelId::ImplicitDiffusionVector, "implicit_diffusion_vector"}},
        {"Kessler pointwise", shaders::kKesslerPointwiseMsl, {KernelId::KesslerPointwise, "kessler_pointwise"}},
        {"Kessler sedimentation",
         shaders::kKesslerSedimentationMsl,
         {KernelId::KesslerSedimentation, "kessler_sedimentation"}},
        {"Lin pointwise", shaders::kLinPointwiseMsl, {KernelId::LinPointwise, "lin_pointwise"}},
        {"Thompson completion",
         shaders::kThompsonCompletionMsl,
         {KernelId::ThompsonCompletion, "thompson_completion"},
         true},
        {"Thompson deposition",
         shaders::kThompsonDepositionMsl,
         {KernelId::ThompsonDeposition, "thompson_deposition"},
         true},
        {"Thompson freezing", shaders::kThompsonFreezingMsl, {KernelId::ThompsonFreezing, "thompson_freezing"}, true},
        {"Thompson cloud phase",
         shaders::kThompsonCloudPhaseMsl,
         {KernelId::ThompsonCloudPhase, "thompson_cloud_phase"},
         true},
        {"Milbrandt sedimentation",
         shaders::kMilbrandtSedimentationMsl,
         {KernelId::MilbrandtSedimentation, "milbrandt_sedimentation"},
         true},
        {"Slab column", shaders::kSlabColumnMsl, {KernelId::SlabColumn, "slab_column"}, true},
        {"Simple grey radiation", shaders::kSimpleGreyMsl, {KernelId::SimpleGrey, "simple_grey"}, true},
        {"Held source reconstruction", shaders::kHeldSourcesMsl, {KernelId::HeldSources, "held_sources"}, true},
        {"Held column remap", shaders::kColumnRemapMsl, {KernelId::ColumnRemap, "column_remap"}, true},
        {"Reference atmosphere", shaders::kReferenceAtmosphereMsl, {KernelId::ReferenceAtmosphere, "reference_atmosphere"}, true},
        {"Moving energy", shaders::kMovingEnergyMsl, {KernelId::MovingEnergy, "moving_energy"}, true},
        {"Conservative reduction", shaders::kConservativeReductionMsl, {KernelId::ConservativeReduction, "conservative_reduction"}, true},
        {"Conservative momentum", shaders::kConservativeMomentumMsl, {KernelId::ConservativeMomentum, "conservative_momentum"}, true},
        {"Moving momentum", shaders::kMovingMomentumMsl, {KernelId::MovingMomentum, "moving_momentum"}, true},
        {"Earth geometry", shaders::kEarthGeometryMsl, {KernelId::EarthGeometry, "earth_geometry"}, true},
        {"Conservative remap", shaders::kConservativeRemapMsl, {KernelId::ConservativeRemap, "conservative_remap"}, true},
        {"Collocated SGS", shaders::kCollocatedSgsMsl, {KernelId::CollocatedSgs, "collocated_sgs"}, true},
        {"RK tendency", shaders::kRkTendencyMsl, {KernelId::RkTendency, "rk_tendency"}, true},
        {"RK combination", shaders::kRkCombinationMsl, {KernelId::RkCombination, "rk_combination"}, true},
        {"Milbrandt cold", shaders::kMilbrandtColdMsl, {KernelId::MilbrandtCold, "milbrandt_cold"}, true},
        {"Milbrandt melting", shaders::kMilbrandtMeltingMsl, {KernelId::MilbrandtMelting, "milbrandt_melting"}, true},
        {"Milbrandt cloud adjustment",
         shaders::kMilbrandtCloudAdjustmentMsl,
         {KernelId::MilbrandtCloudAdjustment, "milbrandt_cloud_adjustment"},
         true},
        {"Milbrandt warm rain",
         shaders::kMilbrandtWarmRainMsl,
         {KernelId::MilbrandtWarmRain, "milbrandt_warm_rain"},
         true},
        {"Thompson warm rain", shaders::kThompsonWarmRainMsl, {KernelId::ThompsonWarmRain, "thompson_warm_rain"}, true},
        {"Thompson cloud adjustment",
         shaders::kThompsonCloudAdjustmentMsl,
         {KernelId::ThompsonCloudAdjustment, "thompson_cloud_adjustment"},
         true},
        {"Thompson moment sedimentation",
         shaders::kThompsonMomentSedimentationMsl,
         {KernelId::ThompsonMomentSedimentation, "thompson_moment_sedimentation"}},
        {"Lin sedimentation", shaders::kLinSedimentationMsl, {KernelId::LinSedimentation, "lin_sedimentation"}},
        {"microphysics apply", shaders::kMicrophysicsApplyMsl, {KernelId::MicrophysicsApply, "microphysics_apply"}},
        {"advection X", shaders::kAdvectionXMsl, {KernelId::AdvectX, "advect_x"}},
        {"advection Y", shaders::kAdvectionYMsl, {KernelId::AdvectY, "advect_y"}},
        {"TVD vertical flux", shaders::kTvdVerticalFluxMsl, {KernelId::TvdVerticalFlux, "tvd_vertical_flux"}},
        // Fast transport math can amplify roundoff into different cloud-activation branches.
        {"conservative C-grid transport",
         shaders::kConservativeCgridTransportMsl,
         {KernelId::AdvectConservativeMassFluxCartesianCgrid, "conservative_mass_flux"},
         {KernelId::AdvectConservativeUpdateCartesianCgrid, "conservative_scalar_update"},
         true},
        {"dynamics apply",
         shaders::kDynamicsApplyMsl,
         {KernelId::StateRelaxationCartesian, "state_relaxation_cartesian"},
         {KernelId::SlowApply, "slow_apply"}},
        {"Cartesian C-grid slow tendencies",
         shaders::kCartesianCgridSlowTendenciesMsl,
         {KernelId::CartesianCgridSlowTendencies, "cartesian_cgrid_slow_tendencies"}},
        {"Cartesian C-grid acoustics",
         shaders::kAcousticCartesianCgridMsl,
         {KernelId::AcousticPressureCartesianCgrid, "acoustic_pressure_cartesian_cgrid"},
         {KernelId::AcousticMomentumCartesianCgrid, "acoustic_momentum_cartesian_cgrid"}, true},
        {"batched Cartesian C-grid acoustics",
         shaders::kAcousticBatchedCartesianCgridMsl,
         {KernelId::AcousticPressureMassFluxCartesianCgrid, "acoustic_pressure_massflux_cartesian_cgrid"}},
        {"C-grid acoustic boundary", shaders::kAcousticBcFaceCartesianCgridMsl,
         {KernelId::AcousticBcFaceCartesianCgrid, "acoustic_bc_face_cartesian_cgrid"}, true},
        {"Cartesian A-grid tendencies",
         shaders::kCartesianAgridMsl,
         {KernelId::CartesianTendencies, "cartesian_agrid_tendencies"}},
        {"acoustic A-grid",
         shaders::kAcousticAgridMsl,
         {KernelId::AcousticPressureCartesian, "acoustic_pressure_agrid"},
         {KernelId::AcousticMomentumCartesian, "acoustic_momentum_agrid"}},
        {"production A-grid continuity",
         shaders::kAcousticContinuityMassFluxCartesianAgridMsl,
         {KernelId::AcousticContinuityMassFluxCartesianAgrid, "acoustic_continuity_massflux_cartesian_agrid"}},
        {"complete A-grid acoustic state",
         shaders::kAcousticCompleteAgridMsl,
         {KernelId::AcousticPressureEosCartesianAgrid, "acoustic_pressure_eos_cartesian_agrid"},
         {KernelId::AcousticMomentumCartesianAgrid, "acoustic_momentum_cartesian_agrid"}},
        {"complete A-grid acoustic closure",
         shaders::kAcousticAgridClosureMsl,
         {KernelId::AcousticNullModeConstraintCartesianAgrid, "acoustic_null_mode_constraint_cartesian_agrid"},
         {KernelId::AcousticBoundaryCartesianAgrid, "acoustic_boundary_cartesian_agrid"}},
        {"mapped terrain state and retained flux",
         shaders::kMappedStateFluxMsl,
         {KernelId::MappedTerrainStateUpdate, "mapped_state_update"},
         {KernelId::MappedTerrainFluxAccumulation, "mapped_flux_accumulation"}},
        {"mapped terrain acoustic tendencies",
         shaders::kMappedAcousticMsl,
         {KernelId::MappedTerrainMassContinuity, "mapped_mass_continuity"},
         {KernelId::MappedTerrainFastMomentum, "mapped_fast_momentum"}, true},
        {"mapped terrain slow tendencies",
         shaders::kMappedSlowTendenciesMsl,
         {KernelId::MappedTerrainSlowTendencies, "mapped_slow_tendencies"}, true},
        {"mapped scalar flux", shaders::kAdvectConservativeScalarFluxCartesianCgridMsl,
         {KernelId::AdvectConservativeScalarFluxCartesianCgrid, "advect_conservative_scalar_flux_cartesian_cgrid"}, true},
        // Precise arithmetic keeps cloud-edge transport within cross-backend error bounds.
        {"mapped terrain scalar transport",
         shaders::kMappedScalarTransportMsl,

         {KernelId::MappedTerrainScalarLimiter, "mapped_scalar_limiter"},
         {KernelId::MappedTerrainScalarUpdate, "mapped_scalar_update"}, true},
        {"mapped terrain final state",
         shaders::kMappedFinalStateMsl,
         {KernelId::MappedTerrainEos, "mapped_eos"}},
        {"mapped terrain boundary", shaders::kMappedBoundaryMsl,
         {KernelId::MappedTerrainBoundary, "mapped_boundary"}},
        {"MYNN level-2.5 stability",
         shaders::kMynnLevel25StabilityMsl,
         {KernelId::MynnLevel25Stability, "mynn_level25_stability"}},
        {"RRTMG longwave transfer",
         shaders::kRrtmgLongwaveTransferMsl,
         {KernelId::RrtmgLongwaveTransfer, "rrtmg_longwave_transfer"}},
        {"RRTMG shortwave transfer",
         shaders::kRrtmgShortwaveTransferMsl,
         {KernelId::RrtmgShortwaveTransfer, "rrtmg_shortwave_transfer"}},
        {"RRTMG longwave transfer batch",
         shaders::kRrtmgLongwaveTransferBatchMsl,
         {KernelId::RrtmgLongwaveTransferBatch, "rrtmg_longwave_transfer_batch"}},
        {"RRTMG shortwave transfer batch",
         shaders::kRrtmgShortwaveTransferBatchMsl,
         {KernelId::RrtmgShortwaveTransferBatch, "rrtmg_shortwave_transfer_batch"}},
        {"Earth-control evolution",
         shaders::kEarthControlEvolutionMsl,
         {KernelId::EarthControlEvolution, "earth_control_evolution"}, true},
    }};
    return catalog;
}

inline const std::array<MetalPipelineAlias, 1>& metal_pipeline_aliases()
{
    using nimbus::compute::KernelId;
    static constexpr std::array<MetalPipelineAlias, 1> aliases = {{
        {KernelId::AdvectPdLimiterCartesianCgrid, KernelId::AdvectConservativeUpdateCartesianCgrid},
    }};
    return aliases;
}

inline nimbus::compute::KernelId canonical_metal_kernel(nimbus::compute::KernelId kernel) noexcept
{
    for (const MetalPipelineAlias& alias : metal_pipeline_aliases())
        if (alias.alias == kernel)
            return alias.target;
    return kernel;
}

inline const MetalLibrarySpec* metal_library_for(nimbus::compute::KernelId kernel) noexcept
{
    kernel = canonical_metal_kernel(kernel);
    for (const MetalLibrarySpec& library : metal_library_catalog())
        for (std::size_t index = 0; index < library.kernel_count; ++index)
            if (library.kernels[index].id == kernel)
                return &library;
    return nullptr;
}

inline const char* metal_source_for(nimbus::compute::KernelId kernel) noexcept
{
    const MetalLibrarySpec* library = metal_library_for(kernel);
    return library == nullptr ? nullptr : library->source;
}

inline const char* metal_source_hash_for(nimbus::compute::KernelId kernel) noexcept
{
    const MetalLibrarySpec* library = metal_library_for(kernel);
    return library == nullptr ? nullptr : library->source_hash.c_str();
}

} // namespace nimbus::metal::detail
