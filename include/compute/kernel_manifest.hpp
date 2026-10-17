#pragma once

/**
 * @file kernel_manifest.hpp
 * @brief Canonical, backend-neutral scientific compute-kernel contract.
 *
 * Vulkan, Metal, parity qualification, and scheduling consume this same table.
 * Backend artifacts may differ, but operation identity, resources, dispatch
 * shape, constants ABI, precision, CPU oracle, and tolerance may not.
 */

#include "compute/parity_types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace nimbus::compute
{

enum class KernelId : std::uint8_t
{
    AcousticMomentumCartesian                  = 0,
    AcousticMomentumCartesianCgrid             = 1,
    AcousticPressureCartesian                  = 2,
    AcousticPressureCartesianCgrid             = 3,
    AcousticPressureMassFluxCartesianCgrid     = 4,
    AdvectConservativeMassFluxCartesianCgrid   = 5,
    AdvectConservativeScalarFluxCartesianCgrid = 6,
    AdvectConservativeUpdateCartesianCgrid     = 7,
    AdvectPdLimiterCartesianCgrid              = 8,
    AdvectX                                    = 9,
    AdvectY                                    = 10,
    TvdVerticalFlux                            = 11,
    DiffuseCartesian                           = 12,
    Diffusion                                  = 13,
    DomainRecyclingHomogeneousFieldShift       = 14,
    AcousticBcFaceCartesianCgrid               = 15,
    CartesianCgridSlowTendencies               = 16,
    CartesianTendencies                        = 17,
    SlowApply                                  = 18,
    StateRelaxationCartesian                   = 19,
    KesslerPointwise                           = 20,
    KesslerSedimentation                       = 21,
    LinPointwise                               = 22,
    LinSedimentation                           = 23,
    MicrophysicsApply                          = 24,
    ArchivedLegacyThompsonLinPointwise         = 25,
    ArchivedLegacyThompsonLinSedimentation     = 26,
    SgsApply                                   = 27,
    SmagorinskyCoefficients                    = 28,
    SmagorinskyTendencies                      = 29,
    MynnLevel25Stability                       = 30,
    RrtmgLongwaveTransfer                      = 31,
    RrtmgShortwaveTransfer                     = 32,
    RrtmgLongwaveTransferBatch                 = 33,
    RrtmgShortwaveTransferBatch                = 34,
    AcousticContinuityMassFluxCartesianAgrid   = 35,
    EarthControlEvolution                      = 36,
    AcousticPressureEosCartesianAgrid          = 37,
    AcousticMomentumCartesianAgrid             = 38,
    AcousticNullModeConstraintCartesianAgrid   = 39,
    AcousticBoundaryCartesianAgrid             = 40,
    MappedTerrainMassContinuity                = 41,
    MappedTerrainFastMomentum                  = 42,
    MappedTerrainSlowTendencies                = 43,
    MappedTerrainScalarUpdate                  = 44,
    MappedTerrainScalarLimiter                 = 45,
    MappedTerrainEos                           = 46,
    MappedTerrainBoundary                      = 47,
    MappedTerrainStateUpdate                   = 48,
    MappedTerrainFluxAccumulation              = 49,
    TkeClosure                                 = 50,
    TkeTendency                                = 51,
    ImplicitDiffusionStencil                   = 52,
    ImplicitDiffusionVector                    = 53,
    ThompsonMomentSedimentation                = 54,
    ThompsonCloudAdjustment                    = 55,
    ThompsonWarmRain                           = 56,
    ThompsonCloudPhase                         = 57,
    ThompsonFreezing                           = 58,
    ThompsonDeposition                         = 59,
    ThompsonCompletion                         = 60,
    MilbrandtWarmRain                          = 61,
    MilbrandtCloudAdjustment                   = 62,
    MilbrandtMelting                           = 63,
    MilbrandtCold                              = 64,
    MilbrandtSedimentation                     = 65,
    RkCombination                              = 66,
    RkTendency                                 = 67,
    CollocatedSgs                              = 68,
    ConservativeRemap                          = 69,
    EarthGeometry                              = 70,
    MovingMomentum                             = 71,
    MovingEnergy                               = 72,
    ReferenceAtmosphere                        = 73,
    ColumnRemap                                = 74,
    SimpleGrey                                 = 75,
    SlabColumn                                 = 76,
    HeldSources                                = 77,
    ConservativeMomentum                       = 78,
    ConservativeReduction                      = 79,
    AcousticRetainedFluxFp64                   = 80,
    Count                                      = 81
};

constexpr std::size_t kernel_count() noexcept
{
    return static_cast<std::size_t>(KernelId::Count);
}

enum class DispatchShape : std::uint8_t
{
    LinearCells,
    LinearColumns,
    BoundaryFace
};

enum class KernelPrecision : std::uint8_t { Float32, Float32StorageFloat64Arithmetic };
enum class CpuFallback : std::uint8_t { Required };
enum class KernelCoordinate : std::uint8_t { Cartesian, Any };
enum class KernelAvailability : std::uint8_t { Active, Deferred, Orphan };

enum class ResourceAccess : std::uint8_t
{
    None = 0,
    Read = 1,
    Write = 2,
    ReadWrite = 3
};

struct ResourceContract
{
    // Number of descriptor positions, including deliberately sparse ones.
    std::uint8_t descriptor_span = 0;
    std::uint32_t read_mask = 0;
    std::uint32_t write_mask = 0;

    constexpr ResourceAccess access(std::uint8_t binding) const noexcept
    {
        if (binding >= 32) return ResourceAccess::None;
        const std::uint32_t bit = std::uint32_t{1} << binding;
        return static_cast<ResourceAccess>(
            ((read_mask & bit) ? 1u : 0u) | ((write_mask & bit) ? 2u : 0u));
    }
};

struct ConstantsAbi
{
    std::uint16_t bytes = 0;
    std::uint8_t alignment = 4;
    std::uint8_t version = 1;
};

struct KernelManifestEntry
{
    KernelId id{};
    const char* operation_id = nullptr;
    const char* kernel_name = nullptr;
    const char* artifact_name = nullptr;
    // Canonical SPIR-V FNV-1a64. Metal artifacts record their own hash while
    // retaining this operation contract and canonical source identity.
    const char* artifact_hash = nullptr;
    ResourceContract resources{};
    DispatchShape shape = DispatchShape::LinearCells;
    ConstantsAbi constants{};
    KernelPrecision precision = KernelPrecision::Float32;
    CpuFallback cpu_fallback = CpuFallback::Required;
    const char* cpu_reference = nullptr;
    parity::Tier parity_tier = parity::Tier::Pointwise;
    parity::ComparisonDomain comparison_domain =
        parity::ComparisonDomain::StrictInterior;
    KernelCoordinate coordinate = KernelCoordinate::Any;
    KernelAvailability availability = KernelAvailability::Active;
    bool startup_required = false;
};

using KernelManifest = std::array<KernelManifestEntry, kernel_count()>;

const KernelManifest& kernel_manifest();
const KernelManifestEntry* find_kernel(KernelId id) noexcept;
const KernelManifestEntry& kernel_manifest_entry(KernelId id);
const KernelManifestEntry* find_kernel(const char* kernel_name) noexcept;
bool validate_kernel_manifest(const KernelManifest& manifest,
                              const char** reason = nullptr) noexcept;

} // namespace nimbus::compute
