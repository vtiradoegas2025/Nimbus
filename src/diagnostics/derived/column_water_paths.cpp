#include "diagnostics/column_water_paths.hpp"

#include "core/infra/physical_constants.hpp"
#include "core/infra/physical_grid_metrics.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace tmv::diagnostics
{
namespace
{

class CompensatedSum
{
public:
    void add(double value) noexcept
    {
        const double corrected = value - correction_;
        const double updated   = sum_ + corrected;
        correction_            = (updated - sum_) - corrected;
        sum_                   = updated;
    }

    [[nodiscard]] double value() const noexcept
    {
        return sum_;
    }

private:
    double sum_        = 0.0;
    double correction_ = 0.0;
};

bool valid_mixing_ratio(double value) noexcept
{
    return std::isfinite(value) && value >= 0.0;
}

} // namespace

struct ColumnWaterPathIntegrator::Implementation
{
    Implementation(const SimulationStateView&   state,
                   const ColumnWaterPathFields& fields_in,
                   ColumnWaterPathSelection     selection_in)
        : volumes(state), fields(fields_in), selection(selection_in), nr(state.grid.nr), nth(state.grid.nth),
          nz(state.grid.nz), inverse_horizontal_area(1.0 / (state.grid.dr * state.grid.dtheta))
    {
        const double area = state.grid.dr * state.grid.dtheta;
        if (nr <= 0 || nth <= 0 || nz <= 0 || !std::isfinite(area) || area <= 0.0)
            throw std::invalid_argument("column water paths require a non-empty grid with positive horizontal area");

        validate_shape(fields.dry_air_density, "rho");
        if (selection.precipitable_water)
            validate_shape(fields.water_vapor, "qv");
        if (selection.cloud_water || selection.liquid_water || selection.total_condensate)
            validate_shape(fields.cloud_water, "qc");
        if (selection.liquid_water || selection.total_condensate)
            validate_shape(fields.rain_water, "qr");
        if (selection.total_condensate)
        {
            validate_shape(fields.cloud_ice, "qi");
            validate_shape(fields.snow, "qs");
            validate_shape(fields.graupel, "qg");
            validate_shape(fields.hail, "qh");
        }
    }

    void validate_shape(const Field3D& field, const char* name) const
    {
        if (field.size_r() != nr || field.size_th() != nth || field.size_z() != nz)
            throw std::invalid_argument("column water path field '" + std::string(name) + "' does not match the grid");
    }

    physical_grid_metrics::ControlVolumeMassView volumes;
    ColumnWaterPathFields                        fields;
    ColumnWaterPathSelection                     selection;
    int                                          nr                      = 0;
    int                                          nth                     = 0;
    int                                          nz                      = 0;
    double                                       inverse_horizontal_area = 0.0;
};

ColumnWaterPathIntegrator::ColumnWaterPathIntegrator(const SimulationStateView&   state,
                                                     const ColumnWaterPathFields& fields,
                                                     ColumnWaterPathSelection     selection)
    : implementation_(std::make_unique<Implementation>(state, fields, selection))
{
}

ColumnWaterPathIntegrator::~ColumnWaterPathIntegrator()                                               = default;
ColumnWaterPathIntegrator::ColumnWaterPathIntegrator(ColumnWaterPathIntegrator&&) noexcept            = default;
ColumnWaterPathIntegrator& ColumnWaterPathIntegrator::operator=(ColumnWaterPathIntegrator&&) noexcept = default;

const ColumnWaterPathSelection& ColumnWaterPathIntegrator::selection() const noexcept
{
    return implementation_->selection;
}

bool ColumnWaterPathIntegrator::evaluate(
    int i, int j, int k_begin, int k_end, ColumnWaterPathResult& result) const noexcept
{
    result           = {};
    const auto& impl = *implementation_;
    if (!impl.selection.any() || i < 0 || i >= impl.nr || j < 0 || j >= impl.nth || k_begin < 0 || k_end > impl.nz ||
        k_begin >= k_end)
        return false;

    CompensatedSum vapor_path;
    CompensatedSum cloud_path;
    CompensatedSum liquid_path;
    CompensatedSum condensate_path;

    for (int k = k_begin; k < k_end; ++k)
    {
        const double density = static_cast<double>(impl.fields.dry_air_density(i, j, k));
        const double volume  = impl.volumes.scalar_volume_m3(i, j, k);
        if (!std::isfinite(density) || density <= 0.0 || !std::isfinite(volume) || volume <= 0.0)
            return false;
        const double dry_air_mass_per_area = density * volume * impl.inverse_horizontal_area;

        double qc_value = 0.0;
        double qr_value = 0.0;
        if (impl.selection.precipitable_water)
        {
            const double qv_value = static_cast<double>(impl.fields.water_vapor(i, j, k));
            if (!valid_mixing_ratio(qv_value))
                return false;
            vapor_path.add(dry_air_mass_per_area * qv_value);
        }
        if (impl.selection.cloud_water || impl.selection.liquid_water || impl.selection.total_condensate)
        {
            qc_value = static_cast<double>(impl.fields.cloud_water(i, j, k));
            if (!valid_mixing_ratio(qc_value))
                return false;
            if (impl.selection.cloud_water)
                cloud_path.add(dry_air_mass_per_area * qc_value);
        }
        if (impl.selection.liquid_water || impl.selection.total_condensate)
        {
            qr_value = static_cast<double>(impl.fields.rain_water(i, j, k));
            if (!valid_mixing_ratio(qr_value))
                return false;
            if (impl.selection.liquid_water)
                liquid_path.add(dry_air_mass_per_area * (qc_value + qr_value));
        }
        if (impl.selection.total_condensate)
        {
            const double qi_value = static_cast<double>(impl.fields.cloud_ice(i, j, k));
            const double qs_value = static_cast<double>(impl.fields.snow(i, j, k));
            const double qg_value = static_cast<double>(impl.fields.graupel(i, j, k));
            const double qh_value = static_cast<double>(impl.fields.hail(i, j, k));
            if (!valid_mixing_ratio(qi_value) || !valid_mixing_ratio(qs_value) || !valid_mixing_ratio(qg_value) ||
                !valid_mixing_ratio(qh_value))
                return false;
            condensate_path.add(dry_air_mass_per_area *
                                (qc_value + qr_value + qi_value + qs_value + qg_value + qh_value));
        }
    }

    constexpr double kgm2_to_liquid_mm = 1000.0 / physical_constants::liquid_water_density_kgm3;
    if (impl.selection.precipitable_water)
        result.precipitable_water_mm = vapor_path.value() * kgm2_to_liquid_mm;
    if (impl.selection.cloud_water)
        result.cloud_water_path_kgm2 = cloud_path.value();
    if (impl.selection.liquid_water)
        result.liquid_water_path_kgm2 = liquid_path.value();
    if (impl.selection.total_condensate)
        result.total_condensate_path_kgm2 = condensate_path.value();
    return true;
}

struct ColumnWaterPathCache::Implementation
{
    Implementation(const SimulationStateView&   state,
                   const ColumnWaterPathFields& fields,
                   ColumnWaterPathSelection     selection_in,
                   int                          k_begin_in,
                   int                          k_end_in)
        : integrator(state, fields, selection_in), selection(selection_in), nr(state.grid.nr), nth(state.grid.nth),
          k_begin(k_begin_in), k_end(k_end_in),
          precipitable_water(selection.precipitable_water ? map_size() : 0, nan()),
          cloud_water(selection.cloud_water ? map_size() : 0, nan()),
          liquid_water(selection.liquid_water ? map_size() : 0, nan()),
          total_condensate(selection.total_condensate ? map_size() : 0, nan())
    {
        if (k_begin < 0 || k_end > state.grid.nz || k_begin >= k_end)
            throw std::invalid_argument("column water-path cache received an invalid vertical range");
    }

    [[nodiscard]] std::size_t map_size() const noexcept
    {
        return static_cast<std::size_t>(nr) * static_cast<std::size_t>(nth);
    }

    [[nodiscard]] static float nan() noexcept
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    void ensure()
    {
        if (ready)
            return;
        int invalid_columns = 0;
#pragma omp parallel for collapse(2)                                                                                   \
    reduction(+ : invalid_columns) if (static_cast<std::size_t>(nr) * static_cast<std::size_t>(nth) >= 4096)
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nth; ++j)
            {
                ColumnWaterPathResult result;
                if (!integrator.evaluate(i, j, k_begin, k_end, result))
                {
                    ++invalid_columns;
                    continue;
                }
                const std::size_t index =
                    static_cast<std::size_t>(i) * static_cast<std::size_t>(nth) + static_cast<std::size_t>(j);
                if (selection.precipitable_water)
                    precipitable_water[index] = static_cast<float>(result.precipitable_water_mm);
                if (selection.cloud_water)
                    cloud_water[index] = static_cast<float>(result.cloud_water_path_kgm2);
                if (selection.liquid_water)
                    liquid_water[index] = static_cast<float>(result.liquid_water_path_kgm2);
                if (selection.total_condensate)
                    total_condensate[index] = static_cast<float>(result.total_condensate_path_kgm2);
            }
        valid = invalid_columns == 0;
        ready = true;
    }

    ColumnWaterPathIntegrator integrator;
    ColumnWaterPathSelection  selection;
    int                       nr      = 0;
    int                       nth     = 0;
    int                       k_begin = 0;
    int                       k_end   = 0;
    bool                      ready   = false;
    bool                      valid   = false;
    std::vector<float>        precipitable_water;
    std::vector<float>        cloud_water;
    std::vector<float>        liquid_water;
    std::vector<float>        total_condensate;
};

ColumnWaterPathCache::ColumnWaterPathCache(const SimulationStateView&   state,
                                           const ColumnWaterPathFields& fields,
                                           ColumnWaterPathSelection     selection,
                                           int                          k_begin,
                                           int                          k_end)
    : implementation_(std::make_unique<Implementation>(state, fields, selection, k_begin, k_end))
{
}

ColumnWaterPathCache::~ColumnWaterPathCache()                                          = default;
ColumnWaterPathCache::ColumnWaterPathCache(ColumnWaterPathCache&&) noexcept            = default;
ColumnWaterPathCache& ColumnWaterPathCache::operator=(ColumnWaterPathCache&&) noexcept = default;

const std::vector<float>& ColumnWaterPathCache::values(ColumnWaterPathKind kind)
{
    implementation_->ensure();
    switch (kind)
    {
    case ColumnWaterPathKind::PrecipitableWater:
        return implementation_->precipitable_water;
    case ColumnWaterPathKind::CloudWater:
        return implementation_->cloud_water;
    case ColumnWaterPathKind::LiquidWater:
        return implementation_->liquid_water;
    case ColumnWaterPathKind::TotalCondensate:
        return implementation_->total_condensate;
    }
    throw std::logic_error("unknown column water-path cache kind");
}

bool ColumnWaterPathCache::valid() const noexcept
{
    return implementation_->ready && implementation_->valid;
}

} // namespace tmv::diagnostics
