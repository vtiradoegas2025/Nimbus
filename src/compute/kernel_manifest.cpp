#include "compute/kernel_manifest.hpp"
#include "compute/thompson_completion.hpp"
#include "compute/kernel_abi.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace nimbus::compute
{
namespace
{

constexpr std::uint32_t bits(unsigned first, unsigned last)
{
    return last < first ? 0u
         : ((last == 31 ? ~0u : ((std::uint32_t{1} << (last + 1)) - 1u))
            & ~((std::uint32_t{1} << first) - 1u));
}

constexpr ResourceContract resources(std::uint8_t span,
                                     std::uint32_t read,
                                     std::uint32_t write)
{
    return {span, read, write};
}

using T = parity::Tier;
using D = parity::ComparisonDomain;
using S = DispatchShape;
using C = KernelCoordinate;
using A = KernelAvailability;

template <typename Constants>
constexpr ConstantsAbi constants_abi() noexcept
{
    static_assert(std::is_standard_layout<Constants>::value,
                  "kernel constants must use a standard-layout ABI");
    static_assert(std::is_trivially_copyable<Constants>::value,
                  "kernel constants must be byte-copyable");
    static_assert(sizeof(Constants) <=
                  std::numeric_limits<std::uint16_t>::max(),
                  "kernel constants size must fit manifest metadata");
    static_assert(alignof(Constants) <=
                  std::numeric_limits<std::uint8_t>::max(),
                  "kernel constants alignment must fit manifest metadata");
    return {static_cast<std::uint16_t>(sizeof(Constants)),
            static_cast<std::uint8_t>(alignof(Constants)), 1};
}

constexpr ConstantsAbi expected_constants_abi(KernelId id) noexcept
{
    using namespace abi;
    switch (id)
    {
    case KernelId::ThompsonCompletion:
        return constants_abi<ThompsonCompletionConstants>();
    case KernelId::AcousticRetainedFluxFp64:
        return constants_abi<AcousticRetainedFluxFp64PushConstants>();
    case KernelId::AcousticMomentumCartesian:
        return constants_abi<AcousticMomentumPushConstants>();
    case KernelId::AcousticMomentumCartesianCgrid:
        return constants_abi<AcousticMomentumCartesianCgridPushConstants>();
    case KernelId::AcousticPressureCartesian:
        return constants_abi<AcousticPressurePushConstants>();
    case KernelId::AcousticPressureCartesianCgrid:
    case KernelId::AcousticPressureMassFluxCartesianCgrid:
        return constants_abi<AcousticPressureCartesianCgridPushConstants>();
    case KernelId::AcousticContinuityMassFluxCartesianAgrid:
        return constants_abi<AcousticContinuityMassFluxCartesianAgridPushConstants>();
    case KernelId::AcousticPressureEosCartesianAgrid:
        return constants_abi<AcousticPressureEosCartesianAgridPushConstants>();
    case KernelId::AcousticMomentumCartesianAgrid:
        return constants_abi<AcousticMomentumCartesianAgridPushConstants>();
    case KernelId::AcousticNullModeConstraintCartesianAgrid:
    case KernelId::AcousticBoundaryCartesianAgrid:
        return constants_abi<AcousticLinePassCartesianAgridPushConstants>();
    case KernelId::EarthControlEvolution:
        return constants_abi<EarthControlEvolutionPushConstants>();
    case KernelId::MappedTerrainMassContinuity:
        return constants_abi<MappedTerrainMassContinuityPushConstants>();
    case KernelId::MappedTerrainFastMomentum:
        return constants_abi<MappedTerrainFastMomentumPushConstants>();
    case KernelId::MappedTerrainSlowTendencies:
        return constants_abi<MappedTerrainSlowTendenciesPushConstants>();
    case KernelId::MappedTerrainScalarUpdate:
        return constants_abi<MappedTerrainScalarUpdatePushConstants>();
    case KernelId::MappedTerrainScalarLimiter:
        return constants_abi<MappedTerrainScalarLimiterPushConstants>();
    case KernelId::MappedTerrainEos:
        return constants_abi<MappedTerrainEosPushConstants>();
    case KernelId::MappedTerrainBoundary:
        return constants_abi<MappedTerrainBoundaryPushConstants>();
    case KernelId::MappedTerrainStateUpdate:
        return constants_abi<MappedTerrainStateUpdatePushConstants>();
    case KernelId::MappedTerrainFluxAccumulation:
        return constants_abi<MappedTerrainFluxAccumulationPushConstants>();
    case KernelId::AdvectConservativeMassFluxCartesianCgrid:
        return constants_abi<AdvectMassFluxCartesianCgridPushConstants>();
    case KernelId::AdvectConservativeScalarFluxCartesianCgrid:
        return constants_abi<AdvectFluxDimsCartesianCgridPushConstants>();
    case KernelId::AdvectConservativeUpdateCartesianCgrid:
        return constants_abi<AdvectUpdateCartesianCgridPushConstants>();
    case KernelId::AdvectPdLimiterCartesianCgrid:
        return constants_abi<AdvectPdLimiterPushConstants>();
    case KernelId::AdvectX:
        return constants_abi<AdvectionXPushConstants>();
    case KernelId::AdvectY:
        return constants_abi<AdvectionYPushConstants>();
    case KernelId::TvdVerticalFlux:
        return constants_abi<TvdPushConstants>();
    case KernelId::DiffuseCartesian:
        return constants_abi<DiffuseCartesianPushConstants>();
    case KernelId::Diffusion:
        return constants_abi<DiffusionPushConstants>();
    case KernelId::DomainRecyclingHomogeneousFieldShift:
        return constants_abi<HomogeneousDomainFieldShiftPushConstants>();
    case KernelId::AcousticBcFaceCartesianCgrid:
        return constants_abi<AcousticBcFacePushConstants>();
    case KernelId::CartesianCgridSlowTendencies:
        return constants_abi<SupercellCgridSlowTendenciesPushConstants>();
    case KernelId::CartesianTendencies:
        return constants_abi<CartesianTendenciesPushConstants>();
    case KernelId::SlowApply:
        return constants_abi<SlowApplyPushConstants>();
    case KernelId::StateRelaxationCartesian:
        return constants_abi<StateRelaxationPushConstants>();
    case KernelId::KesslerPointwise:
        return constants_abi<KesslerPointwisePushConstants>();
    case KernelId::KesslerSedimentation:
        return constants_abi<KesslerSedimentationPushConstants>();
    case KernelId::LinPointwise:
        return constants_abi<LinPointwisePushConstants>();
    case KernelId::LinSedimentation:
        return constants_abi<LinSedimentationPushConstants>();
    case KernelId::MicrophysicsApply:
        return constants_abi<MicrophysicsApplyPushConstants>();
    case KernelId::ArchivedLegacyThompsonLinPointwise:
        return constants_abi<ArchivedLegacyThompsonLinPointwisePushConstants>();
    case KernelId::ArchivedLegacyThompsonLinSedimentation:
        return constants_abi<ArchivedLegacyThompsonLinSedimentationPushConstants>();
    case KernelId::SgsApply:
        return constants_abi<SgsApplyPushConstants>();
    case KernelId::SmagorinskyCoefficients:
        return constants_abi<SmagorinskyCoefficientsPushConstants>();
    case KernelId::TkeClosure:
        return constants_abi<TkeClosurePushConstants>();
    case KernelId::TkeTendency:
        return constants_abi<TkeTendencyPushConstants>();
    case KernelId::HeldSources:
        return constants_abi<HeldSourcesPushConstants>();
    case KernelId::SlabColumn:
        return constants_abi<SlabColumnPushConstants>();
    case KernelId::SimpleGrey:
        return constants_abi<SimpleGreyPushConstants>();
    case KernelId::ColumnRemap:
        return constants_abi<ColumnRemapPushConstants>();
    case KernelId::ReferenceAtmosphere:
        return constants_abi<ReferenceAtmospherePushConstants>();
    case KernelId::MovingEnergy:
        return constants_abi<MovingEnergyPushConstants>();
    case KernelId::ConservativeReduction:
        return constants_abi<ConservativeReductionPushConstants>();
    case KernelId::ConservativeMomentum:
        return constants_abi<ConservativeMomentumPushConstants>();
    case KernelId::MovingMomentum:
        return constants_abi<MovingMomentumPushConstants>();
    case KernelId::EarthGeometry:
        return constants_abi<EarthGeometryPushConstants>();
    case KernelId::ConservativeRemap:
        return constants_abi<ConservativeRemapPushConstants>();
    case KernelId::CollocatedSgs:
        return constants_abi<CollocatedSgsPushConstants>();
    case KernelId::RkTendency:
        return constants_abi<RkTendencyPushConstants>();
    case KernelId::RkCombination:
        return constants_abi<RkCombinationPushConstants>();
    case KernelId::MilbrandtSedimentation:
        return constants_abi<MilbrandtSedimentationPushConstants>();
    case KernelId::MilbrandtCold:
        return constants_abi<MilbrandtColdPushConstants>();
    case KernelId::MilbrandtCloudAdjustment:
        return constants_abi<MilbrandtCloudAdjustmentPushConstants>();
    case KernelId::MilbrandtMelting:
    case KernelId::MilbrandtWarmRain:
    case KernelId::ThompsonWarmRain:
    case KernelId::ThompsonCloudPhase:
    case KernelId::ThompsonFreezing:
    case KernelId::ThompsonDeposition:
        return constants_abi<ThompsonCloudAdjustmentPushConstants>();
    case KernelId::ThompsonCloudAdjustment:
        return constants_abi<ThompsonCloudSourcePushConstants>();
    case KernelId::ThompsonMomentSedimentation:
        return constants_abi<ThompsonMomentSedimentationPushConstants>();
    case KernelId::ImplicitDiffusionStencil:
        return constants_abi<ImplicitDiffusionStencilPushConstants>();
    case KernelId::ImplicitDiffusionVector:
        return constants_abi<ImplicitDiffusionVectorPushConstants>();
    case KernelId::SmagorinskyTendencies:
        return constants_abi<SmagorinskyTendenciesPushConstants>();
    case KernelId::MynnLevel25Stability:
        return constants_abi<MynnLevel25StabilityPushConstants>();
    case KernelId::RrtmgLongwaveTransfer:
        return constants_abi<RrtmgLongwaveTransferPushConstants>();
    case KernelId::RrtmgShortwaveTransfer:
        return constants_abi<RrtmgShortwaveTransferPushConstants>();
    case KernelId::RrtmgLongwaveTransferBatch:
        return constants_abi<RrtmgLongwaveTransferBatchPushConstants>();
    case KernelId::RrtmgShortwaveTransferBatch:
        return constants_abi<RrtmgShortwaveTransferBatchPushConstants>();
    case KernelId::Count:
        return {};
    }
    return {};
}

constexpr bool valid(DispatchShape value) noexcept
{
    return value == DispatchShape::LinearCells ||
        value == DispatchShape::LinearColumns ||
        value == DispatchShape::BoundaryFace;
}

constexpr bool valid(KernelPrecision value) noexcept
{
    return value == KernelPrecision::Float32 || value == KernelPrecision::Float32StorageFloat64Arithmetic;
}

constexpr bool valid(CpuFallback value) noexcept
{
    return value == CpuFallback::Required;
}

constexpr bool valid(KernelCoordinate value) noexcept
{
    return value == KernelCoordinate::Cartesian ||
        value == KernelCoordinate::Any;
}

constexpr bool valid(KernelAvailability value) noexcept
{
    return value == KernelAvailability::Active ||
        value == KernelAvailability::Deferred ||
        value == KernelAvailability::Orphan;
}

constexpr bool valid(parity::Tier value) noexcept
{
    return value == parity::Tier::Pointwise ||
        value == parity::Tier::Stencil ||
        value == parity::Tier::Microphysics ||
        value == parity::Tier::MicrophysicsHeating;
}

constexpr bool valid(parity::ComparisonDomain value) noexcept
{
    return value == parity::ComparisonDomain::StrictInterior ||
        value == parity::ComparisonDomain::FullField ||
        value == parity::ComparisonDomain::BoundaryShell ||
        value == parity::ComparisonDomain::WestFace ||
        value == parity::ComparisonDomain::EastFace ||
        value == parity::ComparisonDomain::SouthFace ||
        value == parity::ComparisonDomain::NorthFace ||
        value == parity::ComparisonDomain::BottomFace ||
        value == parity::ComparisonDomain::TopFace;
}

bool canonical_hash(const char* hash) noexcept
{
    if (hash == nullptr || std::strlen(hash) != 16) return false;
    for (const char* cursor = hash; *cursor != '\0'; ++cursor)
    {
        if (!((*cursor >= '0' && *cursor <= '9') ||
              (*cursor >= 'a' && *cursor <= 'f')))
        {
            return false;
        }
    }
    return true;
}

bool canonical_artifact(const char* kernel_name,
                        const char* artifact_name) noexcept
{
    const std::size_t kernel_length = std::strlen(kernel_name);
    const std::size_t artifact_length = std::strlen(artifact_name);
    return artifact_length == kernel_length + 4 &&
        std::memcmp(kernel_name, artifact_name, kernel_length) == 0 &&
        std::strcmp(artifact_name + kernel_length, ".spv") == 0;
}

const KernelManifest manifest{{
    {KernelId::AcousticMomentumCartesian,
     "dynamics.acoustic_momentum_cartesian",
     "acoustic/acoustic_momentum_cartesian.comp",
     "acoustic/acoustic_momentum_cartesian.comp.spv",
     "69917b3448f4d0b5",
     resources(8, bits(0, 4), bits(5, 7)),
     S::LinearCells,
     {40},
     {},
     {},
     "CartesianAGridScheme::compute_fast_momentum_tendencies",
     T::Stencil,
     D::StrictInterior,
     C::Cartesian},
    {KernelId::AcousticMomentumCartesianCgrid,
     "dynamics.acoustic_momentum_cartesian_cgrid",
     "acoustic/acoustic_momentum_cartesian_cgrid.comp",
     "acoustic/acoustic_momentum_cartesian_cgrid.comp.spv",
     "5aaa9864523d9afe",
     resources(9, bits(0, 5), bits(6, 8)),
     S::LinearCells,
     {44},
     {},
     {},
     "CartesianCGrid3DScheme::compute_fast_momentum_tendencies + dynamics.cpp cpu_fast_momentum",
     T::Stencil,
     D::StrictInterior,
     C::Cartesian},
    {KernelId::AcousticPressureCartesian,
     "dynamics.acoustic_pressure_cartesian",
     "acoustic/acoustic_pressure_cartesian.comp",
     "acoustic/acoustic_pressure_cartesian.comp.spv",
     "1509da583c7ea394",
     resources(7, bits(0, 4), bits(5, 6)),
     S::LinearCells,
     {40},
     {},
     {},
     "CartesianAGridScheme::compute_fast_pressure_tendencies",
     T::Stencil,
     D::StrictInterior,
     C::Cartesian},
    {KernelId::AcousticPressureCartesianCgrid,
     "dynamics.acoustic_pressure_cartesian_cgrid",
     "acoustic/acoustic_pressure_cartesian_cgrid.comp",
     "acoustic/acoustic_pressure_cartesian_cgrid.comp.spv",
     "5af79cbe372681f1",
     resources(10, bits(0, 7), bits(8, 9)),
     S::LinearCells,
     {72},
     {},
     {},
     "CartesianCGrid3DScheme::compute_fast_pressure_tendencies + dynamics.cpp cpu_fast_pressure",
     T::Stencil,
     D::StrictInterior,
     C::Cartesian},
    {KernelId::AcousticPressureMassFluxCartesianCgrid,
     "dynamics.acoustic_pressure_massflux_cartesian_cgrid",
     "acoustic/acoustic_pressure_massflux_cartesian_cgrid.comp",
     "acoustic/acoustic_pressure_massflux_cartesian_cgrid.comp.spv",
     "b4e52fb2d0205218",
     resources(13, bits(0, 7) | bits(10, 12), bits(8, 12)),
     S::LinearCells,
     {72},
     {},
     {},
     "Cartesian C-grid acoustic pressure/continuity + time-averaged mass-flux export",
     T::Stencil,
     D::StrictInterior,
     C::Cartesian},
    {KernelId::AdvectConservativeMassFluxCartesianCgrid,
     "advection.conservative_mass_flux_cartesian_cgrid",
     "advection/advect_conservative_mass_flux_cartesian_cgrid.comp",
     "advection/advect_conservative_mass_flux_cartesian_cgrid.comp.spv",
     "540e782784b5afef",
     resources(7, bits(0, 3), bits(4, 6)),
     S::LinearCells,
     {24},
     {},
     {},
     "advect_scalar_conservative_3d_cartesian (mass-flux phase)",
     T::Stencil,
     D::StrictInterior,
     C::Cartesian},
    {KernelId::AdvectConservativeScalarFluxCartesianCgrid,
     "advection.conservative_scalar_flux_cartesian_cgrid",
     "advection/advect_conservative_scalar_flux_cartesian_cgrid.comp",
     "advection/advect_conservative_scalar_flux_cartesian_cgrid.comp.spv",
     "d06fdf77b3f8dc9d",
     resources(7, bits(0, 3), bits(4, 6)),
     S::LinearCells,
     {16},
     {},
     {},
     "advect_scalar_conservative_3d_cartesian (WENO5/MC face-flux phase)",
     T::Stencil,
     D::StrictInterior,
     C::Cartesian},
    {KernelId::AdvectConservativeUpdateCartesianCgrid,
     "advection.conservative_update_cartesian_cgrid",
     "advection/advect_conservative_update_cartesian_cgrid.comp",
     "advection/advect_conservative_update_cartesian_cgrid.comp.spv",
     "5bb5c8b80e9a9c36",
     resources(13, bits(0, 7) | bits(9, 10), bits(8, 8) | bits(11, 12)),
     S::LinearCells,
     {76},
     {},
     {},
     "advect_scalar_conservative_3d_cartesian (divergence + recovery phase)",
     T::Stencil,
     D::StrictInterior,
     C::Cartesian},
    {KernelId::AdvectPdLimiterCartesianCgrid,
     "advection.pd_limiter_cartesian_cgrid",
     "advection/advect_pd_limiter_cartesian_cgrid.comp",
     "advection/advect_pd_limiter_cartesian_cgrid.comp.spv",
     "cce1050331a8b908",
     resources(10, bits(0, 4) | bits(6, 9), bits(5, 5)),
     S::LinearCells,
     {32},
     {},
     {},
     "advect_scalar_conservative_3d_cartesian (positive-definite flux renormalization)",
     T::Stencil,
     D::StrictInterior,
     C::Cartesian},
    {KernelId::AdvectX,
     "advection.x",
     "advection/advect_x.comp",
     "advection/advect_x.comp.spv",
     "be7ee3339932f72d",
     resources(3, bits(0, 2), bits(2, 2)),
     S::LinearCells,
     {32},
     {},
     {},
     "advect_scalar_1d_x_kernel_cartesian (via advect_scalar_3d)",
     T::Pointwise,
     D::StrictInterior,
     C::Cartesian},
    {KernelId::AdvectY,
     "advection.y",
     "advection/advect_y.comp",
     "advection/advect_y.comp.spv",
     "7d5c8804d1368417",
     resources(3, bits(0, 2), bits(2, 2)),
     S::LinearCells,
     {32},
     {},
     {},
     "advect_scalar_1d_y_kernel_cartesian (via advect_scalar_3d)",
     T::Pointwise,
     D::StrictInterior,
     C::Cartesian},
    {KernelId::TvdVerticalFlux,
     "advection.tvd_vertical_flux",
     "advection/tvd_vertical_flux.comp",
     "advection/tvd_vertical_flux.comp.spv",
     "f7889f53c08a8857",
     resources(4, bits(0, 3), bits(3, 3)),
     S::LinearColumns,
     {32},
     {},
     {},
     "TVD vertical flux kernel (via advect_scalar_3d)",
     T::Pointwise,
     D::StrictInterior,
     C::Any,
     A::Active,
     true},
    {KernelId::DiffuseCartesian,
     "diffusion.cartesian",
     "diffusion/diffuse_cartesian.comp",
     "diffusion/diffuse_cartesian.comp.spv",
     "d75684cdeda6f0dd",
     resources(2, bits(0, 0), bits(1, 1)),
     S::LinearCells,
     {32},
     {},
     {},
     "removed transport-chained point filter",
     T::Pointwise,
     D::StrictInterior,
     C::Cartesian,
     A::Orphan},
    {KernelId::Diffusion,
     "diffusion.cartesian_fv",
     "diffusion/diffusion.comp",
     "diffusion/diffusion.comp.spv",
     "e4725d28ee57dddf",
     resources(5, bits(0, 2) | bits(4, 4), bits(3, 3)),
     S::LinearCells,
     {64},
     {},
     {},
     "ExplicitDiffusionScheme live-density finite-volume tendency",
     T::Stencil,
     D::FullField,
     C::Cartesian},
    {KernelId::DomainRecyclingHomogeneousFieldShift,
     "domain_recycling.homogeneous_field_shift",
     "domain/domain_recycling_homogeneous_field.comp",
     "domain/domain_recycling_homogeneous_field.comp.spv",
     "71cc8ee3b62dc269",
     resources(3, bits(0, 1), bits(2, 2)),
     S::LinearCells,
     {44},
     {},
     {},
     "tmv::domain::execute_homogeneous_field_shift",
     T::Pointwise,
     D::FullField,
     C::Cartesian},
    {KernelId::AcousticBcFaceCartesianCgrid,
     "dynamics.acoustic_bc_face_cartesian_cgrid",
     "dynamics/acoustic_bc_face_cartesian_cgrid.comp",
     "dynamics/acoustic_bc_face_cartesian_cgrid.comp.spv",
     "7e9000e547904439",
     resources(6, bits(0, 5), bits(0, 5)),
     S::BoundaryFace,
     {64},
     {},
     {},
     "CartesianCGridBCScheme::apply_acoustic",
     T::Pointwise,
     D::FullField,
     C::Cartesian},
    {KernelId::CartesianCgridSlowTendencies,
     "dynamics.cartesian_cgrid_slow_tendencies",
     "dynamics/cartesian_cgrid_slow_tendencies.comp",
     "dynamics/cartesian_cgrid_slow_tendencies.comp.spv",
     "c2a0a8f2a3cab774",
     resources(19, bits(0, 18), bits(11, 18)),
     S::LinearCells,
     {56},
     {},
     {},
     "CartesianCGrid3DScheme::compute_slow_tendencies",
     T::Stencil,
     D::StrictInterior,
     C::Cartesian},
    {KernelId::CartesianTendencies,
     "dynamics.cartesian_tendencies",
     "dynamics/cartesian_tendencies.comp",
     "dynamics/cartesian_tendencies.comp.spv",
     "48a704bf983903e4",
     resources(18, bits(0, 17), bits(8, 12)),
     S::LinearCells,
     {64},
     {},
     {},
     "CartesianAGridScheme::compute_momentum_tendencies",
     T::Stencil,
     D::StrictInterior,
     C::Cartesian},
    {KernelId::SlowApply,
     "dynamics.slow_apply",
     "dynamics/slow_apply.comp",
     "dynamics/slow_apply.comp.spv",
     "9bf6902bc54b2bec",
     resources(14, bits(0, 9), bits(10, 13)),
     S::LinearCells,
     {32},
     {},
     {},
     "step_dynamics_split_explicit apply_slow_tendencies",
     T::Pointwise,
     D::StrictInterior,
     C::Any},
    {KernelId::StateRelaxationCartesian,
     "dynamics.state_relaxation_cartesian",
     "dynamics/state_relaxation_cartesian.comp",
     "dynamics/state_relaxation_cartesian.comp.spv",
     "53c5f1aeb63c3c36",
     resources(9, bits(0, 4), bits(5, 8)),
     S::LinearCells,
     {48},
     {},
     {},
     "Rayleigh/lateral sponge damping",
     T::Pointwise,
     D::StrictInterior,
     C::Cartesian},
    {KernelId::KesslerPointwise,
     "microphysics.kessler_pointwise",
     "microphysics/kessler_pointwise.comp",
     "microphysics/kessler_pointwise.comp.spv",
     "bd79d138e715eb5a",
     resources(13, bits(0, 6), bits(7, 12)),
     S::LinearCells,
     {64},
     {},
     {},
     "KesslerScheme::compute_tendencies",
     T::Microphysics,
     D::StrictInterior,
     C::Any},
    {KernelId::KesslerSedimentation,
     "microphysics.kessler_sedimentation",
     "microphysics/kessler_sedimentation.comp",
     "microphysics/kessler_sedimentation.comp.spv",
     "19469f666b3b9510",
     resources(8, bits(0, 7), bits(5, 7)),
     S::LinearColumns,
     {64},
     {},
     {},
     "KesslerScheme::compute_sedimentation",
     T::Microphysics,
     D::StrictInterior,
     C::Any},
    {KernelId::LinPointwise,
     "microphysics.lin_pointwise",
     "microphysics/lin_pointwise.comp",
     "microphysics/lin_pointwise.comp.spv",
     "f6edabe911c28193",
     resources(17, bits(0, 8), bits(9, 16)),
     S::LinearCells,
     {112},
     {},
     {},
     "LinScheme::compute_tendencies",
     T::Microphysics,
     D::StrictInterior,
     C::Any},
    {KernelId::LinSedimentation,
     "microphysics.lin_sedimentation",
     "microphysics/lin_sedimentation.comp",
     "microphysics/lin_sedimentation.comp.spv",
     "2a069f95e8679798",
     resources(10, bits(0, 9), bits(6, 9)),
     S::LinearColumns,
     {64},
     {},
     {},
     "LinScheme::compute_sedimentation",
     T::Microphysics,
     D::StrictInterior,
     C::Any},
    {KernelId::MicrophysicsApply,
     "microphysics.apply",
     "microphysics/microphysics_apply.comp",
     "microphysics/microphysics_apply.comp.spv",
     "9a2c8d06dfad498c",
     resources(20, bits(0, 19), bits(0, 7) | bits(19, 19)),
     S::LinearCells,
     {64},
     {},
     {},
     "step_microphysics tendency apply loop",
     T::Pointwise,
     D::StrictInterior,
     C::Any},
    {KernelId::ArchivedLegacyThompsonLinPointwise,
     "archive.legacy_thompson_lin_pointwise",
     "microphysics/thompson_pointwise.comp",
     "microphysics/thompson_pointwise.comp.spv",
     "9212844269e6742b",
     resources(17, bits(0, 8), bits(9, 16)),
     S::LinearCells,
     {64},
     {},
     {},
     "archived legacy Lin-class shader",
     T::Microphysics,
     D::StrictInterior,
     C::Any,
     A::Orphan},
    {KernelId::ArchivedLegacyThompsonLinSedimentation,
     "archive.legacy_thompson_lin_sedimentation",
     "microphysics/thompson_sedimentation.comp",
     "microphysics/thompson_sedimentation.comp.spv",
     "00f233788e6e86b6",
     resources(8, bits(0, 7), bits(4, 7)),
     S::LinearColumns,
     {64},
     {},
     {},
     "archived legacy Lin-class shader",
     T::Microphysics,
     D::StrictInterior,
     C::Any,
     A::Orphan},
    {KernelId::SgsApply,
     "turbulence.sgs_apply",
     "turbulence/sgs_apply.comp",
     "turbulence/sgs_apply.comp.spv",
     "9b18ad85b1048aef",
     resources(19, bits(0, 12), bits(13, 18)),
     S::LinearCells,
     {56},
     {},
     {},
     "step_dynamics_new SGS integrate loop",
     T::Pointwise,
     D::StrictInterior,
     C::Any},
    {KernelId::SmagorinskyCoefficients,
     "turbulence.smagorinsky_coefficients",
     "turbulence/smagorinsky_coefficients.comp",
     "turbulence/smagorinsky_coefficients.comp.spv",
     "cd1c67a017b16b98",
     resources(8, bits(0, 4), bits(5, 7)),
     S::LinearCells,
     {60},
     {},
     {},
     "SmagorinskyScheme::compute_eddy_coefficients + directional strain rate",
     T::Stencil,
     D::StrictInterior,
     C::Any},
    {KernelId::SmagorinskyTendencies,
     "turbulence.smagorinsky_tendencies",
     "turbulence/smagorinsky_tendencies.comp",
     "turbulence/smagorinsky_tendencies.comp.spv",
     "1b49cb776a39f1df",
     resources(15, bits(0, 9), bits(10, 14)),
     S::LinearCells,
     {44},
     {},
     {},
     "eddy_viscosity momentum/scalar diffusion tendencies",
     T::Stencil,
     D::StrictInterior,
     C::Any},
    {KernelId::MynnLevel25Stability,
     "boundary_layer.mynn_level25_stability",
     "boundary_layer/mynn_level25_stability.comp",
     "boundary_layer/mynn_level25_stability.comp.spv",
     "2a40af78b09df0b4",
     resources(9, bits(0, 3), bits(4, 8)),
     S::LinearCells,
     {16},
     {},
     {},
     "tmv::mynn::level25_stability_functions",
     T::Pointwise,
     D::FullField,
     C::Any},
    {KernelId::RrtmgLongwaveTransfer,
     "radiation.rrtmg_longwave_transfer",
     "radiation/rrtmg_longwave_transfer.comp",
     "radiation/rrtmg_longwave_transfer.comp.spv",
     "8941721960b4f0e1",
     resources(9, bits(0, 6), bits(7, 8)),
     S::LinearColumns,
     {32},
     {},
     {},
     "tmv::rrtmg::solve_longwave",
     T::Pointwise,
     D::FullField,
     C::Any},
    {KernelId::RrtmgShortwaveTransfer,
     "radiation.rrtmg_shortwave_transfer",
     "radiation/rrtmg_shortwave_transfer.comp",
     "radiation/rrtmg_shortwave_transfer.comp.spv",
     "dda7fc32cb8af87e",
     resources(18, bits(0, 5) | bits(8, 17), bits(6, 17)),
     S::LinearColumns,
     {32},
     {},
     {},
     "tmv::rrtmg::solve_shortwave",
     T::Pointwise,
     D::FullField,
     C::Any},
    {KernelId::RrtmgLongwaveTransferBatch,
     "radiation.rrtmg_longwave_transfer_batch",
     "radiation/rrtmg_longwave_transfer_batch.comp",
     "radiation/rrtmg_longwave_transfer_batch.comp.spv",
     "8abba943a395b1c6",
     resources(10, bits(0, 7), bits(8, 9)),
     S::LinearColumns,
     {32},
     {},
     {},
     "tmv::rrtmg::solve_longwave column batch",
     T::Pointwise,
     D::FullField,
     C::Any},
    {KernelId::RrtmgShortwaveTransferBatch,
     "radiation.rrtmg_shortwave_transfer_batch",
     "radiation/rrtmg_shortwave_transfer_batch.comp",
     "radiation/rrtmg_shortwave_transfer_batch.comp.spv",
     "01098c98c15e87fc",
     resources(19, bits(0, 6) | bits(9, 18), bits(7, 18)),
     S::LinearColumns,
     {32},
     {},
     {},
     "tmv::rrtmg::solve_shortwave column batch",
     T::Pointwise,
     D::FullField,
     C::Any},
    {KernelId::AcousticContinuityMassFluxCartesianAgrid,
     "dynamics.acoustic_continuity_massflux_cartesian_agrid",
     "acoustic/acoustic_continuity_massflux_cartesian_agrid.comp",
     "acoustic/acoustic_continuity_massflux_cartesian_agrid.comp.spv",
     "785ce7c646242243",
     resources(10, bits(0, 5) | bits(7, 9), bits(6, 9)),
     S::LinearCells,
     {64},
     {},
     {},
     "CartesianAGridScheme::compute_fast_pressure_tendencies + retained mass flux",
     T::Stencil,
     D::FullField,
     C::Cartesian},
    {KernelId::EarthControlEvolution,
     "chaos.earth_control_evolution",
     "chaos/earth_control_evolution.comp",
     "chaos/earth_control_evolution.comp.spv",
     "ad1ec8614e502ab2",
     resources(2, bits(0, 0), bits(1, 1)),
     S::LinearCells,
     {80, 8},
     {},
     {},
     "EarthCorrelatedNoise Gaussian initialization, AR(1) advance, and reconstruction",
     T::Pointwise,
     D::FullField,
     C::Any},
    {KernelId::AcousticPressureEosCartesianAgrid,
     "dynamics.acoustic_pressure_eos_cartesian_agrid",
     "acoustic/acoustic_pressure_eos_cartesian_agrid.comp",
     "acoustic/acoustic_pressure_eos_cartesian_agrid.comp.spv",
     "f70661b5aa63bfbd",
     resources(5, bits(0, 2), bits(3, 4)),
     S::LinearCells,
     {48},
     {},
     {},
     "A-grid acoustic continuity commit and nonlinear rho-theta EOS",
     T::Pointwise,
     D::FullField,
     C::Cartesian},
    {KernelId::AcousticMomentumCartesianAgrid,
     "dynamics.acoustic_momentum_cartesian_agrid",
     "acoustic/acoustic_momentum_cartesian_agrid.comp",
     "acoustic/acoustic_momentum_cartesian_agrid.comp.spv",
     "646196f9f4003fc7",
     resources(9, bits(0, 5), bits(6, 8)),
     S::LinearCells,
     {40},
     {},
     {},
     "CartesianAGridScheme::compute_fast_momentum_tendencies and commit",
     T::Stencil,
     D::FullField,
     C::Cartesian},
    {KernelId::AcousticNullModeConstraintCartesianAgrid,
     "dynamics.acoustic_null_mode_constraint_cartesian_agrid",
     "acoustic/acoustic_null_mode_constraint_cartesian_agrid.comp",
     "acoustic/acoustic_null_mode_constraint_cartesian_agrid.comp.spv",
     "2fb08245e66313bf",
     resources(11, bits(0, 10), bits(8, 10)),
     S::LinearColumns,
     {16},
     {},
     {},
     "tmv::collocated_velocity::constrain_directional_null_modes",
     T::Stencil,
     D::StrictInterior,
     C::Cartesian},
    {KernelId::AcousticBoundaryCartesianAgrid,
     "dynamics.acoustic_boundary_cartesian_agrid",
     "acoustic/acoustic_boundary_cartesian_agrid.comp",
     "acoustic/acoustic_boundary_cartesian_agrid.comp.spv",
     "99bb7a86c5d9ce3a",
     resources(9, bits(0, 8), bits(4, 8)),
     S::BoundaryFace,
     {16},
     {},
     {},
     "CartesianBCScheme::apply_acoustic exact rigid profile",
     T::Pointwise,
     D::FullField,
     C::Cartesian},
    {KernelId::MappedTerrainMassContinuity,
     "terrain.mapped_mass_continuity",
     "terrain/mapped_mass_continuity.comp",
     "terrain/mapped_mass_continuity.comp.spv",
     "849460320bcdee9a",
     resources(18, bits(0, 17), bits(0, 17)),
     S::LinearCells,
     {64},
     {},
     {},
     "CartesianCGrid3DScheme::compute_fast_pressure_tendencies",
     T::Stencil,
     D::FullField,
     C::Cartesian},
    {KernelId::MappedTerrainFastMomentum,
     "terrain.mapped_fast_momentum",
     "terrain/mapped_fast_momentum.comp",
     "terrain/mapped_fast_momentum.comp.spv",
     "313be64de3aa37cc",
     resources(7, bits(0, 3), bits(4, 6)),
     S::LinearCells,
     {32},
     {},
     {},
     "CartesianCGrid3DScheme::compute_fast_momentum_tendencies",
     T::Stencil,
     D::FullField,
     C::Cartesian},
    {KernelId::MappedTerrainSlowTendencies,
     "terrain.mapped_slow_tendencies",
     "terrain/mapped_slow_tendencies.comp",
     "terrain/mapped_slow_tendencies.comp.spv",
     "6df8c3f191d64a34",
     resources(24, bits(0, 15), bits(16, 23)),
     S::LinearCells,
     {64},
     {},
     {},
     "CartesianCGrid3DScheme::compute_slow_tendencies mapped branch",
     T::Stencil,
     D::FullField,
     C::Cartesian},
    {KernelId::MappedTerrainScalarUpdate,
     "terrain.mapped_scalar_update",
     "terrain/mapped_scalar_update.comp",
     "terrain/mapped_scalar_update.comp.spv",
     "69ee3b1adb8063a6",
     resources(14, bits(0, 7), bits(8, 13)),
     S::LinearCells,
     {48},
     {},
     {},
     "advect_scalar_conservative_3d_cartesian mapped retained update",
     T::Stencil,
     D::FullField,
     C::Cartesian},
    {KernelId::MappedTerrainScalarLimiter,
     "terrain.mapped_scalar_limiter",
     "terrain/mapped_scalar_limiter.comp",
     "terrain/mapped_scalar_limiter.comp.spv",
     "4153765379975d7c",
     resources(7, bits(0, 5), bits(6, 6)),
     S::LinearCells,
     {48},
     {},
     {},
     "advect_scalar_conservative_3d_cartesian mapped positive-definite limiter",
     T::Stencil,
     D::FullField,
     C::Cartesian},
    {KernelId::MappedTerrainEos,
     "terrain.mapped_eos",
     "terrain/mapped_eos.comp",
     "terrain/mapped_eos.comp.spv",
     "7150bd92bf411fab",
     resources(3, bits(0, 1), bits(2, 2)),
     S::LinearCells,
     {16},
     {},
     {},
     "tmv::dynamics_eos::apply_pressure_projection",
     T::Pointwise,
     D::FullField,
     C::Cartesian},
    {KernelId::MappedTerrainBoundary,
     "terrain.mapped_boundary",
     "terrain/mapped_boundary.comp",
     "terrain/mapped_boundary.comp.spv",
     "9a50b2b53f8b66fd",
     resources(10, bits(0, 4), bits(5, 9)),
     S::LinearCells,
     {48},
     {},
     {},
     "CartesianCGridBCScheme::apply_full_attributed full-step lateral/vertical closures",
     T::Pointwise,
     D::FullField,
     C::Cartesian},
    {KernelId::MappedTerrainStateUpdate,
     "terrain.mapped_state_update",
     "terrain/mapped_state_update.comp",
     "terrain/mapped_state_update.comp.spv",
     "5ff13858381aaae6",
     resources(16, bits(0, 10), bits(11, 15)),
     S::LinearCells,
     {32},
     {},
     {},
     "split-explicit mapped C-grid slow/continuity/momentum state commits",
     T::Pointwise,
     D::FullField,
     C::Cartesian},
    {KernelId::MappedTerrainFluxAccumulation,
     "terrain.mapped_flux_accumulation",
     "terrain/mapped_flux_accumulation.comp",
     "terrain/mapped_flux_accumulation.comp.spv",
     "c6de0a6065396063",
     resources(9, bits(0, 5), bits(6, 8)),
     S::LinearCells,
     {32},
     {},
     {},
     "CartesianCGridTransportContext retained operator-time mass-flux accumulation",
     T::Pointwise,
     D::FullField,
     C::Cartesian},
    {KernelId::TkeClosure,
     "turbulence.tke_closure",
     "turbulence/tke_closure.comp",
     "turbulence/tke_closure.comp.spv",
     "9e69b39f4a535491",
     resources(15, bits(0, 6), bits(7, 14)),
     S::LinearCells,
     {64},
     {},
     {},
     "TKEScheme coefficients, directional shear, buoyancy and dissipation",
     T::Stencil,
     D::FullField,
     C::Cartesian},
    {KernelId::TkeTendency,
     "turbulence.tke_tendency",
     "turbulence/tke_tendency.comp",
     "turbulence/tke_tendency.comp.spv",
     "bb9e74065fb488b8",
     resources(8, bits(0, 6), bits(7, 7)),
     S::LinearCells,
     {40},
     {},
     {},
     "TKEScheme conservative energy transport and production balance",
     T::Stencil,
     D::FullField,
     C::Cartesian},
    {KernelId::ImplicitDiffusionStencil,
     "diffusion.implicit_stencil",
     "diffusion/implicit_diffusion_stencil.comp",
     "diffusion/implicit_diffusion_stencil.comp.spv",
     "ad3472218080626f",
     resources(9, bits(0, 3), bits(4, 8)),
     S::LinearColumns,
     {80},
     {},
     {},
     "ImplicitDiffusionScheme backward-Euler live-mass operator",
     T::Stencil,
     D::FullField,
     C::Cartesian},
    {KernelId::ImplicitDiffusionVector,
     "diffusion.implicit_vector",
     "diffusion/implicit_diffusion_vector.comp",
     "diffusion/implicit_diffusion_vector.comp.spv",
     "4c402eb107c49eef",
     resources(12, bits(0, 6), bits(7, 11)),
     S::LinearColumns,
     {32},
     {},
     {},
     "ImplicitDiffusionScheme Krylov corrections and stored tendencies",
     T::Pointwise,
     D::FullField,
     C::Cartesian},
    {KernelId::ThompsonMomentSedimentation,
     "microphysics.thompson_moment_sedimentation",
     "microphysics/thompson_moment_sedimentation.comp",
     "microphysics/thompson_moment_sedimentation.comp.spv",
     "f22921b5a490cbe0",
     resources(14, bits(0, 6) | bits(10, 13), bits(7, 13)),
     S::LinearColumns,
     {48},
     {},
     {},
     "ThompsonScheme physical-column mass and number sedimentation",
     T::Stencil,
     D::FullField,
     C::Any},
    {KernelId::ThompsonCloudAdjustment,
     "microphysics.thompson_cloud_adjustment",
     "microphysics/thompson_cloud_adjustment.comp",
     "microphysics/thompson_cloud_adjustment.comp.spv",
     "1b71246919ab88d0",
     resources(18, bits(0, 10), bits(11, 17)),
     S::LinearCells,
     {48},
     {},
     {},
     "ThompsonScheme cloud saturation and tabulated aerosol exchange",
     T::Pointwise,
     D::FullField,
     C::Any},
    {KernelId::ThompsonWarmRain,
     "microphysics.thompson_warm_rain",
     "microphysics/thompson_warm_rain.comp",
     "microphysics/thompson_warm_rain.comp.spv",
     "43d2e861e86269c9",
     resources(24, bits(0, 13), bits(14, 23)),
     S::LinearCells,
     {44},
     {},
     {},
     "ThompsonScheme warm-rain conversion, number normalization and rain evaporation",
     T::Pointwise,
     D::FullField,
     C::Any},
    {KernelId::ThompsonCloudPhase,
     "microphysics.thompson_cloud_phase",
     "microphysics/thompson_cloud_phase.comp",
     "microphysics/thompson_cloud_phase.comp.spv",
     "fb9940e42734e5ab",
     resources(18, bits(0, 11), bits(12, 17)),
     S::LinearCells,
     {44},
     {},
     {},
     "ThompsonScheme post-sedimentation cloud melting and homogeneous freezing",
     T::Pointwise,
     D::FullField,
     C::Any},
    {KernelId::ThompsonFreezing,
     "microphysics.thompson_freezing",
     "microphysics/thompson_freezing.comp",
     "microphysics/thompson_freezing.comp.spv",
     "a1c0710cde07a404",
     resources(29, bits(0, 15), bits(16, 28)),
     S::LinearCells,
     {44},
     {},
     {},
     "ThompsonScheme factored rain and cloud freezing",
     T::Pointwise,
     D::FullField,
     C::Any},
    {KernelId::ThompsonDeposition,
     "microphysics.thompson_deposition",
     "microphysics/thompson_deposition.comp",
     "microphysics/thompson_deposition.comp.spv",
     "e73ee9547171b5f0",
     resources(26, bits(0, 12), bits(13, 25)),
     S::LinearCells,
     {44},
     {},
     {},
     "ThompsonScheme ice and snow deposition and frozen-particle sublimation",
     T::Pointwise,
     D::FullField,
     C::Any},
    {KernelId::ThompsonCompletion,
     "microphysics.thompson_completion",
     "microphysics/thompson_completion.comp",
     "microphysics/thompson_completion.comp.spv",
     "4413901236eaab8f",
     resources(3, bits(0, 1), bits(2, 2)),
     S::LinearCells,
     {12},
     {},
     {},
     "ThompsonScheme native coupled cold processes and pointwise state assembly",
     T::Pointwise,
     D::FullField,
     C::Any},
    {KernelId::MilbrandtWarmRain,
     "microphysics.milbrandt_warm_rain",
     "microphysics/milbrandt_warm_rain.comp",
     "microphysics/milbrandt_warm_rain.comp.spv",
     "5b12883a9c382a6d",
     resources(21, bits(0, 14), bits(15, 20)),
     S::LinearCells,
     {44},
     {},
     {},
     "tmv::milbrandt_detail::warm_rain_cell (default rain population)",
     T::Pointwise,
     D::FullField,
     C::Any},

    {KernelId::MilbrandtCloudAdjustment,
     "microphysics.milbrandt_cloud_adjustment",
     "microphysics/milbrandt_cloud_adjustment.comp",
     "microphysics/milbrandt_cloud_adjustment.comp.spv",
     "dbc1cfd29eef710f",
     resources(14, bits(0, 6), bits(7, 13)),
     S::LinearCells,
     {48},
     {},
     {},
     "thermodynamics::saturation_adjustment and milbrandt_detail::continental_activated_ccn",
     T::Pointwise,
     D::FullField,
     C::Any},

    {KernelId::MilbrandtMelting,
     "microphysics.milbrandt_melting",
     "microphysics/milbrandt_melting.comp",
     "microphysics/milbrandt_melting.comp.spv",
     "c8597652137fab74",
     resources(24, bits(0, 12), bits(13, 23)),
     S::LinearCells,
     {44},
     {},
     {},
     "tmv::milbrandt_detail::melting_cell",
     T::Pointwise,
     D::FullField,
     C::Any},

    {KernelId::MilbrandtCold,
     "microphysics.milbrandt_cold",
     "microphysics/milbrandt_cold.comp",
     "microphysics/milbrandt_cold.comp.spv",
     "9884ccfca60a2f35",
     resources(16, bits(0, 1), bits(2, 15)),
     S::LinearCells,
     {48},
     {},
     {},
     "tmv::milbrandt_detail::cold_cell",
     T::Pointwise,
     D::FullField,
     C::Any},

    {KernelId::MilbrandtSedimentation,
     "microphysics.milbrandt_sedimentation",
     "microphysics/milbrandt_sedimentation.comp",
     "microphysics/milbrandt_sedimentation.comp.spv",
     "b87921fd33f236b2",
     resources(12, bits(0, 8), bits(7, 7) | bits(9, 11)),
     S::LinearColumns,
     {48},
     {},
     {},
     "tmv::milbrandt_detail::sediment_column",
     T::Pointwise,
     D::FullField,
     C::Any},

    {KernelId::RkCombination,
     "time_stepping.rk_combination",
     "time_stepping/rk_combination.comp",
     "time_stepping/rk_combination.comp.spv",
     "0f3defca792671f1",
     resources(13, bits(0, 8), bits(9, 12)),
     S::LinearCells,
     {64},
     {},
     {},
     "tmv::rk::combine with dry-mass reconstruction",
     T::Pointwise,
     D::FullField,
     C::Any},

    {KernelId::RkTendency,
     "time_stepping.rk_tendency",
     "time_stepping/rk_tendency.comp",
     "time_stepping/rk_tendency.comp.spv",
     "fe9f61763281b3d7",
     resources(12, bits(0, 6), bits(7, 11)),
     S::LinearCells,
     {64},
     {},
     {},
     "Cartesian unsplit acoustic and conservative transport tendencies",
     T::Pointwise,
     D::FullField,
     C::Any},

    {KernelId::CollocatedSgs,
     "turbulence.collocated_sgs",
     "turbulence/collocated_sgs.comp",
     "turbulence/collocated_sgs.comp.spv",
     "e4cdc2eb9bde0730",
     resources(23, bits(0, 14), bits(15, 22)),
     S::LinearCells,
     {64},
     {},
     {},
     "collocated_eddy_viscosity flat stress, scalar diffusion and TKE transfer",
     T::Pointwise,
     D::FullField,
     C::Any},

    {KernelId::ConservativeRemap,
     "terrain.conservative_remap",
     "terrain/conservative_remap.comp",
     "terrain/conservative_remap.comp.spv",
     "a54a3b07668cfa98",
     resources(26, bits(0, 15), bits(16, 25)),
     S::LinearCells,
     {24},
     {},
     {},
     "Earth-fixed conservative cell inventory and shared donor flux remap",
     T::Pointwise,
     D::FullField,
     C::Any},

    {KernelId::EarthGeometry,
     "terrain.earth_geometry",
     "terrain/earth_geometry.comp",
     "terrain/earth_geometry.comp.spv",
     "6fdd830b23565fd3",
     resources(28, bits(0, 7), bits(8, 27)),
     S::LinearCells,
     {112},
     {},
     {},
     "Earth-fixed analytic surface, exact bilinear means, metrics and swept volumes",
     T::Pointwise,
     D::FullField,
     C::Any},

    {KernelId::MovingMomentum,
     "terrain.moving_momentum",
     "terrain/moving_momentum.comp",
     "terrain/moving_momentum.comp.spv",
     "583fdc339bb49da8",
     resources(2, bits(0, 0), bits(1, 1)),
     S::LinearCells,
     {64},
     {},
     {},
     "C-grid moving momentum, boundary corrections and conservative impulse/kinetic-work ledgers",
     T::Pointwise,
     D::FullField,
     C::Any},

    {KernelId::MovingEnergy,
     "terrain.moving_energy",
     "terrain/moving_energy.comp",
     "terrain/moving_energy.comp.spv",
     "8d569d0a392f57aa",
     resources(2, bits(0, 0), bits(1, 1)),
     S::LinearCells,
     {128},
     {},
     {},
     "Moist mapped energy density and conserved-energy thermodynamic reconstruction",
     T::Pointwise,
     D::FullField,
     C::Any},

    {KernelId::ReferenceAtmosphere,
     "terrain.reference_atmosphere",
     "terrain/reference_atmosphere.comp",
     "terrain/reference_atmosphere.comp.spv",
     "794ec6786cf6867b",
     resources(2, bits(0, 0), bits(1, 1)),
     S::LinearCells,
     {64},
     {},
     {},
     "Terrain-height reference atmosphere interpolation and dry-density reconstruction",
     T::Pointwise,
     D::FullField,
     C::Any},

    {KernelId::ColumnRemap,
     "terrain.column_remap",
     "terrain/column_remap.comp",
     "terrain/column_remap.comp.spv",
     "d6f606e47593105f",
     resources(2, bits(0, 0), bits(1, 1)),
     S::LinearCells,
     {32},
     {},
     {},
     "Held column inventories and depth-weighted slab remapping",
     T::Pointwise,
     D::FullField,
     C::Any},
    {KernelId::SimpleGrey,
     "radiation.simple_grey",
     "radiation/simple_grey.comp",
     "radiation/simple_grey.comp.spv",
     "7d1a2f6aaf0e148c",
     resources(2, bits(0, 1), bits(1, 1)),
     S::LinearCells,
     {32},
     {},
     {},
     "SimpleGreyScheme optical depths, fluxes and column heating",
     T::Pointwise,
     D::FullField,
     C::Any},
    {KernelId::SlabColumn,
     "boundary_layer.slab_column",
     "boundary_layer/slab_column.comp",
     "boundary_layer/slab_column.comp.spv",
     "f5b5659db7b11bff",
     resources(2, bits(0, 0), bits(1, 1)),
     S::LinearCells,
     {32},
     {},
     {},
     "SlabScheme depth, surface fluxes and conservative mixing",
     T::Pointwise,
     D::FullField,
     C::Any},
    {KernelId::HeldSources,
     "terrain.held_sources",
     "terrain/held_sources.comp",
     "terrain/held_sources.comp.spv",
     "740eec00158d6c5d",
     resources(2, bits(0, 0), bits(1, 1)),
     S::LinearCells,
     {84},
     {},
     {},
     "Held PBL and radiation sources on current scalar and dual masses",
     T::Pointwise,
     D::FullField,
     C::Cartesian},
    {KernelId::ConservativeMomentum,
     "dynamics.conservative_momentum",
     "dynamics/conservative_momentum.comp",
     "dynamics/conservative_momentum.comp.spv",
     "09fff86ade92406e",
     resources(2, bits(0, 0), bits(1, 1)),
     S::LinearCells,
     {80},
     {},
     {},
     "CartesianCGrid3DScheme conservative momentum and pressure",
     T::Stencil,
     D::StrictInterior,
     C::Cartesian},

    {KernelId::ConservativeReduction,
     "dynamics.conservative_reduction",
     "dynamics/conservative_reduction.comp",
     "dynamics/conservative_reduction.comp.spv",
     "2eda27e1074ccd5e",
     resources(2, bits(0, 0), bits(1, 1)),
     S::LinearCells,
     {8},
     {},
     {},
     "Nested conservative restriction, reflux and storage",
     T::Pointwise,
     D::FullField,
     C::Cartesian},

    {KernelId::AcousticRetainedFluxFp64,
     "dynamics.acoustic_retained_flux_fp64",
     "acoustic/acoustic_retained_flux_fp64.comp",
     "acoustic/acoustic_retained_flux_fp64.comp.spv",
     "2698b06dd388c6be",
     resources(7, bits(0, 6), bits(4, 6)),
     S::LinearCells,
     {32},
     KernelPrecision::Float32StorageFloat64Arithmetic,
     {},
     "CartesianCGridTransportContext::accumulate_impl NativeCompensated",
     T::Pointwise,
     D::FullField,
     C::Cartesian},

}};

} // namespace

const KernelManifest& kernel_manifest() { return manifest; }

const KernelManifestEntry* find_kernel(KernelId id) noexcept
{
    const auto index = static_cast<std::size_t>(id);
    return index < manifest.size() ? &manifest[index] : nullptr;
}

const KernelManifestEntry& kernel_manifest_entry(KernelId id)
{
    const KernelManifestEntry* entry = find_kernel(id);
    if (entry == nullptr)
        throw std::out_of_range("invalid compute kernel identifier");
    return *entry;
}

const KernelManifestEntry* find_kernel(const char* kernel_name) noexcept
{
    if (kernel_name == nullptr) return nullptr;
    for (const auto& entry : manifest)
    {
        if (std::strcmp(entry.kernel_name, kernel_name) == 0) return &entry;
    }
    return nullptr;
}

bool validate_kernel_manifest(const KernelManifest& candidate,
                              const char** reason) noexcept
{
    auto fail = [reason](const char* message) {
        if (reason != nullptr) *reason = message;
        return false;
    };
    std::array<bool, kernel_count()> seen{};
    for (std::size_t i = 0; i < candidate.size(); ++i)
    {
        const auto& entry = candidate[i];
        const auto ordinal = static_cast<std::size_t>(entry.id);
        if (ordinal >= candidate.size() || ordinal != i)
            return fail("kernel IDs must be dense and manifest-ordered");
        if (seen[ordinal]) return fail("duplicate kernel ID");
        seen[ordinal] = true;
        if (!entry.operation_id || !*entry.operation_id || !entry.kernel_name ||
            !*entry.kernel_name || !entry.artifact_name || !*entry.artifact_name ||
            !entry.cpu_reference || !*entry.cpu_reference)
            return fail("kernel identity, artifact, and CPU oracle are required");
        if (!canonical_artifact(entry.kernel_name, entry.artifact_name))
            return fail("kernel artifacts must canonically append .spv to source names");
        if (!canonical_hash(entry.artifact_hash))
            return fail("artifact hashes must be canonical lowercase 16-hex strings");

        for (std::size_t prior = 0; prior < i; ++prior)
        {
            if (std::strcmp(entry.operation_id,
                            candidate[prior].operation_id) == 0)
                return fail("operation IDs must be unique");
            if (std::strcmp(entry.kernel_name,
                            candidate[prior].kernel_name) == 0)
                return fail("kernel source names must be unique");
            if (std::strcmp(entry.artifact_name,
                            candidate[prior].artifact_name) == 0)
                return fail("kernel artifact names must be unique");
        }

        if (!valid(entry.shape) || !valid(entry.precision) ||
            !valid(entry.cpu_fallback) || !valid(entry.parity_tier) ||
            !valid(entry.comparison_domain) || !valid(entry.coordinate) ||
            !valid(entry.availability))
            return fail("kernel metadata contains an invalid enum value");

        const ConstantsAbi expected = expected_constants_abi(entry.id);
        if (entry.constants.bytes != expected.bytes ||
            entry.constants.alignment != expected.alignment ||
            entry.constants.version != expected.version)
            return fail("kernel constants ABI does not match its host layout");
        if (entry.constants.alignment == 0 ||
            (entry.constants.alignment & (entry.constants.alignment - 1)) != 0 ||
            entry.constants.bytes % entry.constants.alignment != 0 ||
            entry.constants.version == 0)
            return fail("kernel constants ABI alignment or version is invalid");

        if (entry.resources.descriptor_span == 0 ||
            entry.resources.descriptor_span > 32)
            return fail("resource span must be within the descriptor mask");
        const std::uint32_t valid = entry.resources.descriptor_span == 32
            ? ~0u : ((std::uint32_t{1} << entry.resources.descriptor_span) - 1u);
        if (((entry.resources.read_mask | entry.resources.write_mask) & ~valid) != 0 ||
            (entry.resources.read_mask | entry.resources.write_mask) == 0)
            return fail("resource access masks must fit the descriptor span");
    }
    if (reason != nullptr) *reason = nullptr;
    return true;
}

} // namespace nimbus::compute
