#pragma once

#include "compute/kernel_abi.hpp"

#include <cstdint>

namespace nimbus::metal
{

using PushConstants                     = nimbus::compute::abi::HomogeneousDomainFieldShiftPushConstants;
using SmagorinskyConstants              = nimbus::compute::abi::SmagorinskyCoefficientsPushConstants;
using SmagorinskyTendenciesConstants    = nimbus::compute::abi::SmagorinskyTendenciesPushConstants;
using SgsApplyConstants                 = nimbus::compute::abi::SgsApplyPushConstants;
using DiffusionConstants                = nimbus::compute::abi::DiffusionPushConstants;
using KesslerPointwiseConstants         = nimbus::compute::abi::KesslerPointwisePushConstants;
using KesslerSedimentationConstants     = nimbus::compute::abi::KesslerSedimentationPushConstants;
using LinPointwiseConstants             = nimbus::compute::abi::LinPointwisePushConstants;
using LinSedimentationConstants         = nimbus::compute::abi::LinSedimentationPushConstants;
using MicrophysicsApplyConstants        = nimbus::compute::abi::MicrophysicsApplyPushConstants;
using AdvectionXConstants               = nimbus::compute::abi::AdvectionXPushConstants;
using AdvectionYConstants               = nimbus::compute::abi::AdvectionYPushConstants;
using TvdVerticalFluxConstants          = nimbus::compute::abi::TvdPushConstants;
using AdvectMassFluxConstants           = nimbus::compute::abi::AdvectMassFluxCartesianCgridPushConstants;
using AdvectUpdateConstants             = nimbus::compute::abi::AdvectUpdateCartesianCgridPushConstants;
using StateRelaxationConstants          = nimbus::compute::abi::StateRelaxationPushConstants;
using SlowApplyConstants                = nimbus::compute::abi::SlowApplyPushConstants;
using CartesianCgridSlowConstants       = nimbus::compute::abi::SupercellCgridSlowTendenciesPushConstants;
using AcousticPressureCgridConstants    = nimbus::compute::abi::AcousticPressureCartesianCgridPushConstants;
using AcousticMomentumCgridConstants    = nimbus::compute::abi::AcousticMomentumCartesianCgridPushConstants;
using AcousticBcFaceConstants           = nimbus::compute::abi::AcousticBcFacePushConstants;
using CartesianAgridConstants           = nimbus::compute::abi::CartesianTendenciesPushConstants;
using AcousticAgridPressureConstants    = nimbus::compute::abi::AcousticPressurePushConstants;
using AcousticAgridMomentumConstants    = nimbus::compute::abi::AcousticMomentumPushConstants;
using AcousticAgridContinuityConstants  = nimbus::compute::abi::AcousticContinuityMassFluxCartesianAgridPushConstants;
using AcousticAgridPressureEosConstants = nimbus::compute::abi::AcousticPressureEosCartesianAgridPushConstants;
using AcousticAgridNativeMomentumConstants   = nimbus::compute::abi::AcousticMomentumCartesianAgridPushConstants;
using AcousticAgridLinePassConstants         = nimbus::compute::abi::AcousticLinePassCartesianAgridPushConstants;
using MappedTerrainStateUpdateConstants      = nimbus::compute::abi::MappedTerrainStateUpdatePushConstants;
using MappedTerrainFluxAccumulationConstants = nimbus::compute::abi::MappedTerrainFluxAccumulationPushConstants;
using MappedTerrainMassContinuityConstants = nimbus::compute::abi::MappedTerrainMassContinuityPushConstants;
using MappedTerrainFastMomentumConstants = nimbus::compute::abi::MappedTerrainFastMomentumPushConstants;
using MappedTerrainSlowTendenciesConstants = nimbus::compute::abi::MappedTerrainSlowTendenciesPushConstants;
using MappedTerrainScalarFluxConstants = nimbus::compute::abi::AdvectFluxDimsCartesianCgridPushConstants;
using MappedTerrainScalarLimiterConstants = nimbus::compute::abi::MappedTerrainScalarLimiterPushConstants;
using MappedTerrainScalarUpdateConstants = nimbus::compute::abi::MappedTerrainScalarUpdatePushConstants;
using MappedTerrainEosConstants = nimbus::compute::abi::MappedTerrainEosPushConstants;
using MappedTerrainBoundaryConstants = nimbus::compute::abi::MappedTerrainBoundaryPushConstants;
using MynnLevel25StabilityConstants          = nimbus::compute::abi::MynnLevel25StabilityPushConstants;
using RrtmgLongwaveTransferConstants         = nimbus::compute::abi::RrtmgLongwaveTransferPushConstants;
using RrtmgShortwaveTransferConstants        = nimbus::compute::abi::RrtmgShortwaveTransferPushConstants;
using RrtmgLongwaveTransferBatchConstants    = nimbus::compute::abi::RrtmgLongwaveTransferBatchPushConstants;
using RrtmgShortwaveTransferBatchConstants   = nimbus::compute::abi::RrtmgShortwaveTransferBatchPushConstants;
using RrtmgLongwaveColumnParameters          = nimbus::compute::abi::RrtmgLongwaveColumnParameters;
using RrtmgShortwaveColumnParameters         = nimbus::compute::abi::RrtmgShortwaveColumnParameters;

static_assert(sizeof(PushConstants) == 44, "Metal domain-shift constants ABI changed");
static_assert(alignof(PushConstants) == alignof(std::int32_t), "Metal domain-shift constants alignment changed");
static_assert(sizeof(SmagorinskyConstants) == 60, "Metal Smagorinsky coefficients ABI changed");
static_assert(sizeof(SmagorinskyTendenciesConstants) == 44, "Metal Smagorinsky tendencies ABI changed");
static_assert(sizeof(SgsApplyConstants) == 56, "Metal SGS Apply ABI changed");
static_assert(sizeof(DiffusionConstants) == 64, "Metal diffusion ABI changed");
static_assert(sizeof(KesslerPointwiseConstants) == 64, "Metal Kessler pointwise ABI changed");
static_assert(sizeof(KesslerSedimentationConstants) == 64, "Metal Kessler sedimentation ABI changed");
static_assert(sizeof(LinPointwiseConstants) == 112, "Metal Lin pointwise ABI changed");
static_assert(sizeof(LinSedimentationConstants) == 64, "Metal Lin sedimentation ABI changed");
static_assert(sizeof(MicrophysicsApplyConstants) == 64, "Metal microphysics apply ABI changed");
static_assert(sizeof(AdvectionXConstants) == 32, "Metal advection-X ABI changed");
static_assert(sizeof(AdvectionYConstants) == 32, "Metal advection-Y ABI changed");
static_assert(sizeof(TvdVerticalFluxConstants) == 32, "Metal TVD vertical-flux ABI changed");
static_assert(sizeof(AdvectMassFluxConstants) == 24, "Metal conservative mass-flux ABI changed");
static_assert(sizeof(AdvectUpdateConstants) == 76, "Metal conservative update ABI changed");
static_assert(sizeof(StateRelaxationConstants) == 48, "Metal state-relaxation ABI changed");
static_assert(sizeof(SlowApplyConstants) == 32, "Metal slow-apply ABI changed");
static_assert(sizeof(CartesianCgridSlowConstants) == 56, "Metal C-grid slow-tendency ABI changed");
static_assert(sizeof(AcousticPressureCgridConstants) == 72, "Metal C-grid acoustic-pressure ABI changed");
static_assert(sizeof(AcousticMomentumCgridConstants) == 44, "Metal C-grid acoustic-momentum ABI changed");
static_assert(sizeof(AcousticBcFaceConstants) == 64, "Metal C-grid acoustic-BC ABI changed");
static_assert(sizeof(CartesianAgridConstants) == 64, "Metal A-grid tendencies ABI changed");
static_assert(sizeof(AcousticAgridPressureConstants) == 40, "Metal A-grid pressure ABI changed");
static_assert(sizeof(AcousticAgridMomentumConstants) == 40, "Metal A-grid momentum ABI changed");
static_assert(sizeof(AcousticAgridContinuityConstants) == 64, "Metal A-grid continuity ABI changed");
static_assert(sizeof(AcousticAgridPressureEosConstants) == 48, "Metal A-grid pressure/EOS ABI changed");
static_assert(sizeof(AcousticAgridNativeMomentumConstants) == 40, "Metal A-grid native momentum ABI changed");
static_assert(sizeof(AcousticAgridLinePassConstants) == 16, "Metal A-grid line-pass ABI changed");
static_assert(sizeof(MappedTerrainStateUpdateConstants) == 32, "Metal mapped state-update ABI changed");
static_assert(sizeof(MappedTerrainFluxAccumulationConstants) == 32, "Metal mapped flux-accumulation ABI changed");
static_assert(sizeof(MappedTerrainMassContinuityConstants) == 64, "Metal mapped mass-continuity ABI changed");
static_assert(sizeof(MappedTerrainFastMomentumConstants) == 32, "Metal mapped fast-momentum ABI changed");
static_assert(sizeof(MappedTerrainSlowTendenciesConstants) == 64, "Metal mapped slow-tendencies ABI changed");
static_assert(sizeof(MappedTerrainScalarFluxConstants) == 16, "Metal mapped scalar-flux ABI changed");
static_assert(sizeof(MappedTerrainScalarLimiterConstants) == 48, "Metal mapped scalar-limiter ABI changed");
static_assert(sizeof(MappedTerrainScalarUpdateConstants) == 48, "Metal mapped scalar-update ABI changed");
static_assert(sizeof(MappedTerrainEosConstants) == 16, "Metal mapped EOS ABI changed");
static_assert(sizeof(MappedTerrainBoundaryConstants) == 48, "Metal mapped boundary ABI changed");
static_assert(sizeof(MynnLevel25StabilityConstants) == 16, "Metal MYNN stability ABI changed");
static_assert(sizeof(RrtmgLongwaveTransferConstants) == 32, "Metal RRTMG longwave ABI changed");
static_assert(sizeof(RrtmgShortwaveTransferConstants) == 32, "Metal RRTMG shortwave ABI changed");
static_assert(sizeof(RrtmgLongwaveTransferBatchConstants) == 32, "Metal RRTMG longwave batch ABI changed");
static_assert(sizeof(RrtmgShortwaveTransferBatchConstants) == 32, "Metal RRTMG shortwave batch ABI changed");
static_assert(sizeof(RrtmgLongwaveColumnParameters) == 16, "Metal RRTMG longwave parameters ABI changed");
static_assert(sizeof(RrtmgShortwaveColumnParameters) == 16, "Metal RRTMG shortwave parameters ABI changed");

} // namespace nimbus::metal
