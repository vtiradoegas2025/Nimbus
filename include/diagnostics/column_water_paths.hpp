#pragma once

#include "core/field/field3d.hpp"
#include "core/runtime/simulation_state.hpp"

#include <limits>
#include <memory>
#include <vector>

/**
 * @file column_water_paths.hpp
 * @brief Terrain-aware dry-density-basis atmospheric water-path integrals.
 *
 * Nimbus prognoses dry-air density and dry-air mixing ratios.  Consequently,
 * rho_d * q_x * dz_phys is the exact species mass per unit horizontal area in
 * a scalar control volume.  The physical layer thickness is obtained from the
 * shared control-volume geometry, including terrain-following interfaces.
 */
namespace tmv::diagnostics
{

struct ColumnWaterPathSelection
{
    bool precipitable_water = false;
    bool cloud_water        = false;
    bool liquid_water       = false;
    bool total_condensate   = false;

    [[nodiscard]] bool any() const noexcept
    {
        return precipitable_water || cloud_water || liquid_water || total_condensate;
    }
};

struct ColumnWaterPathFields
{
    const Field3D& dry_air_density;
    const Field3D& water_vapor;
    const Field3D& cloud_water;
    const Field3D& rain_water;
    const Field3D& cloud_ice;
    const Field3D& snow;
    const Field3D& graupel;
    const Field3D& hail;
};

struct ColumnWaterPathResult
{
    double precipitable_water_mm      = std::numeric_limits<double>::quiet_NaN();
    double cloud_water_path_kgm2      = std::numeric_limits<double>::quiet_NaN();
    double liquid_water_path_kgm2     = std::numeric_limits<double>::quiet_NaN();
    double total_condensate_path_kgm2 = std::numeric_limits<double>::quiet_NaN();
};

/**
 * Validated, allocation-free view used for many column evaluations.
 *
 * Construction performs shape validation once.  evaluate() then touches only
 * the species selected by ColumnWaterPathSelection and returns false for any
 * non-finite density/mixing ratio, negative mixing ratio, or malformed layer.
 */
class ColumnWaterPathIntegrator
{
public:
    ColumnWaterPathIntegrator(const SimulationStateView&   state,
                              const ColumnWaterPathFields& fields,
                              ColumnWaterPathSelection     selection);
    ~ColumnWaterPathIntegrator();

    ColumnWaterPathIntegrator(ColumnWaterPathIntegrator&&) noexcept;
    ColumnWaterPathIntegrator& operator=(ColumnWaterPathIntegrator&&) noexcept;
    ColumnWaterPathIntegrator(const ColumnWaterPathIntegrator&)            = delete;
    ColumnWaterPathIntegrator& operator=(const ColumnWaterPathIntegrator&) = delete;

    [[nodiscard]] const ColumnWaterPathSelection& selection() const noexcept;

    [[nodiscard]] bool evaluate(int i, int j, int k_begin, int k_end, ColumnWaterPathResult& result) const noexcept;

private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

enum class ColumnWaterPathKind
{
    PrecipitableWater,
    CloudWater,
    LiquidWater,
    TotalCondensate,
};

/** Lazy request-aware 2-D cache for output paths sharing the same 3-D read. */
class ColumnWaterPathCache
{
public:
    ColumnWaterPathCache(const SimulationStateView&   state,
                         const ColumnWaterPathFields& fields,
                         ColumnWaterPathSelection     selection,
                         int                          k_begin,
                         int                          k_end);
    ~ColumnWaterPathCache();

    ColumnWaterPathCache(ColumnWaterPathCache&&) noexcept;
    ColumnWaterPathCache& operator=(ColumnWaterPathCache&&) noexcept;
    ColumnWaterPathCache(const ColumnWaterPathCache&)            = delete;
    ColumnWaterPathCache& operator=(const ColumnWaterPathCache&) = delete;

    const std::vector<float>& values(ColumnWaterPathKind kind);
    [[nodiscard]] bool        valid() const noexcept;

private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

} // namespace tmv::diagnostics
