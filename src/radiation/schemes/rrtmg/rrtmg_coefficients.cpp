#include "rrtmg_coefficients.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#ifdef HAVE_NETCDF
#include <netcdf.h>
#endif

namespace tmv::rrtmg
{

#ifdef HAVE_NETCDF
namespace
{

[[noreturn]] void throw_netcdf(int status, const std::string& operation, const std::string& path)
{
    throw std::runtime_error("RRTMG coefficient " + operation + " failed for '" + path + "': " + nc_strerror(status));
}

class NetcdfReader
{
public:
    explicit NetcdfReader(std::string path) : path_(std::move(path))
    {
        const int status = nc_open(path_.c_str(), NC_NOWRITE, &file_id_);
        if (status != NC_NOERR)
            throw_netcdf(status, "open", path_);
    }

    ~NetcdfReader()
    {
        if (file_id_ >= 0)
            nc_close(file_id_);
    }

    NetcdfReader(const NetcdfReader&)            = delete;
    NetcdfReader& operator=(const NetcdfReader&) = delete;

    std::size_t dimension(const char* name) const
    {
        int dimension_id = -1;
        int status       = nc_inq_dimid(file_id_, name, &dimension_id);
        if (status != NC_NOERR)
            throw_netcdf(status, std::string("dimension '") + name + "'", path_);
        std::size_t length = 0;
        status             = nc_inq_dimlen(file_id_, dimension_id, &length);
        if (status != NC_NOERR)
            throw_netcdf(status, std::string("dimension '") + name + "'", path_);
        return length;
    }

    void require_dimension(const char* name, std::size_t expected) const
    {
        const std::size_t actual = dimension(name);
        if (actual != expected)
            throw std::runtime_error("RRTMG coefficient dimension '" + std::string(name) + "' is " +
                                     std::to_string(actual) + ", expected " + std::to_string(expected));
    }

    std::string attribute(const char* name) const
    {
        std::size_t length = 0;
        int         status = nc_inq_attlen(file_id_, NC_GLOBAL, name, &length);
        if (status != NC_NOERR)
            throw_netcdf(status, std::string("attribute '") + name + "'", path_);
        std::string value(length, '\0');
        status = nc_get_att_text(file_id_, NC_GLOBAL, name, value.data());
        if (status != NC_NOERR)
            throw_netcdf(status, std::string("attribute '") + name + "'", path_);
        return value;
    }

    CoefficientArray doubles(const char* name, const std::vector<std::size_t>& expected_shape) const
    {
        const int variable_id = variable(name);
        require_shape(variable_id, name, expected_shape);
        CoefficientArray result;
        result.shape = expected_shape;
        result.values.resize(element_count(expected_shape, name));
        const int status = nc_get_var_double(file_id_, variable_id, result.values.data());
        if (status != NC_NOERR)
            throw_netcdf(status, std::string("variable '") + name + "'", path_);
        double     missing_value = 0.0;
        const bool has_missing_value =
            nc_get_att_double(file_id_, variable_id, "missing_value", &missing_value) == NC_NOERR;
        for (double& value : result.values)
        {
            if (value == NC_FILL_DOUBLE || (has_missing_value && value == missing_value))
                value = 0.0;
        }
        if (std::any_of(result.values.begin(), result.values.end(), [](double value) { return !std::isfinite(value); }))
            throw std::runtime_error("RRTMG coefficient variable '" + std::string(name) +
                                     "' contains a non-finite value");
        return result;
    }

    std::vector<double> vector(const char* name, std::size_t expected_size) const
    {
        return doubles(name, {expected_size}).values;
    }

    std::vector<int> integers(const char* name, const std::vector<std::size_t>& expected_shape) const
    {
        const int variable_id = variable(name);
        require_shape(variable_id, name, expected_shape);
        std::vector<int> values(element_count(expected_shape, name));
        const int        status = nc_get_var_int(file_id_, variable_id, values.data());
        if (status != NC_NOERR)
            throw_netcdf(status, std::string("variable '") + name + "'", path_);
        return values;
    }

    std::vector<std::string> strings(const char* name, const std::vector<std::size_t>& expected_shape) const
    {
        if (expected_shape.size() < 2)
            throw std::logic_error("RRTMG string variables require a character width");
        const int variable_id = variable(name);
        require_shape(variable_id, name, expected_shape);
        const std::size_t count = element_count(expected_shape, name);
        std::vector<char> characters(count);
        const int         status = nc_get_var_text(file_id_, variable_id, characters.data());
        if (status != NC_NOERR)
            throw_netcdf(status, std::string("variable '") + name + "'", path_);
        const std::size_t        width = expected_shape.back();
        std::vector<std::string> values(count / width);
        for (std::size_t row = 0; row < values.size(); ++row)
        {
            std::string value(characters.data() + row * width, characters.data() + (row + 1) * width);
            while (!value.empty() && (value.back() == ' ' || value.back() == '\0'))
                value.pop_back();
            values[row] = std::move(value);
        }
        return values;
    }

private:
    int variable(const char* name) const
    {
        int       variable_id = -1;
        const int status      = nc_inq_varid(file_id_, name, &variable_id);
        if (status != NC_NOERR)
            throw_netcdf(status, std::string("variable '") + name + "'", path_);
        return variable_id;
    }

    void require_shape(int variable_id, const char* name, const std::vector<std::size_t>& expected) const
    {
        int dimension_count = 0;
        int status          = nc_inq_varndims(file_id_, variable_id, &dimension_count);
        if (status != NC_NOERR)
            throw_netcdf(status, std::string("shape of '") + name + "'", path_);
        if (dimension_count != static_cast<int>(expected.size()))
            throw std::runtime_error("RRTMG coefficient variable '" + std::string(name) + "' has an unexpected rank");
        std::vector<int> dimension_ids(expected.size());
        status = nc_inq_vardimid(file_id_, variable_id, dimension_ids.data());
        if (status != NC_NOERR)
            throw_netcdf(status, std::string("shape of '") + name + "'", path_);
        for (std::size_t index = 0; index < expected.size(); ++index)
        {
            std::size_t actual = 0;
            status             = nc_inq_dimlen(file_id_, dimension_ids[index], &actual);
            if (status != NC_NOERR)
                throw_netcdf(status, std::string("shape of '") + name + "'", path_);
            if (actual != expected[index])
                throw std::runtime_error("RRTMG coefficient variable '" + std::string(name) +
                                         "' has an unexpected shape");
        }
    }

    static std::size_t element_count(const std::vector<std::size_t>& shape, const char* name)
    {
        std::size_t count = 1;
        for (const std::size_t extent : shape)
        {
            if (extent == 0 || count > std::numeric_limits<std::size_t>::max() / extent)
                throw std::runtime_error("RRTMG coefficient variable '" + std::string(name) + "' has an invalid shape");
            count *= extent;
        }
        return count;
    }

    std::string path_;
    int         file_id_ = -1;
};

CoefficientIdentity
identity(const NetcdfReader& reader, const char* version_attribute, const char* expected_title_fragment)
{
    CoefficientIdentity result;
    result.title             = reader.attribute("Title");
    result.model_version     = reader.attribute(version_attribute);
    result.continuum_version = reader.attribute("Water_vapor_continuum_version");
    result.file_version      = reader.attribute("File_version");
    if (result.title.find(expected_title_fragment) == std::string::npos || result.model_version.empty() ||
        result.continuum_version.empty() || result.file_version.empty())
        throw std::runtime_error("RRTMG coefficient identity is not recognized");
    return result;
}

std::vector<int> reduced_g_points(const NetcdfReader& reader, std::size_t bands, int expected_total)
{
    const std::vector<int> all = reader.integers("NumGPoints", {2, bands});
    std::vector<int>       reduced(bands);
    for (std::size_t band = 0; band < bands; ++band)
    {
        if (all[band] != 16 || all[bands + band] <= 0 || all[bands + band] > 16)
            throw std::runtime_error("RRTMG g-point reduction table is malformed");
        reduced[band] = all[bands + band];
    }
    if (std::accumulate(reduced.begin(), reduced.end(), 0) != expected_total)
        throw std::runtime_error("RRTMG reduced g-point total is incorrect");
    return reduced;
}

std::vector<std::size_t> g_point_offsets(const std::vector<int>& counts)
{
    std::vector<std::size_t> offsets(counts.size() + 1, 0);
    for (std::size_t band = 0; band < counts.size(); ++band)
        offsets[band + 1] = offsets[band] + static_cast<std::size_t>(counts[band]);
    return offsets;
}

enum class ReductionRule
{
    WeightedAverage,
    Sum,
};

constexpr std::array<double, 16> quadrature_weights{{
    0.1527534276,
    0.1491729617,
    0.1420961469,
    0.1316886544,
    0.1181945205,
    0.1019300893,
    0.0832767040,
    0.0626720116,
    0.0424925000,
    0.0046269894,
    0.0038279891,
    0.0030260086,
    0.0022199750,
    0.0014140010,
    0.0005330000,
    0.0000750000,
}};

const std::vector<std::vector<int>>& longwave_groups()
{
    static const std::vector<std::vector<int>> groups{
        {1, 1, 2, 2, 2, 2, 2, 2, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {2, 2, 2, 2, 2, 2, 2, 2},
        {2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2},
        {2, 2, 2, 2, 2, 2, 2, 2},
        {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2},
        {2, 2, 2, 2, 4, 4},
        {1, 1, 2, 2, 2, 2, 3, 3},
        {1, 1, 1, 1, 2, 2, 4, 4},
        {3, 3, 4, 6},
        {8, 8},
        {8, 8},
        {4, 12},
    };
    return groups;
}

const std::vector<std::vector<int>>& shortwave_groups()
{
    static const std::vector<std::vector<int>> groups{
        {2, 2, 2, 2, 4, 4},
        {1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2},
        {1, 1, 1, 1, 2, 2, 4, 4},
        {1, 1, 1, 1, 2, 2, 4, 4},
        {1, 1, 1, 1, 1, 1, 1, 1, 2, 6},
        {1, 1, 1, 1, 1, 1, 1, 1, 2, 6},
        {8, 8},
        {2, 2, 1, 1, 1, 1, 1, 1, 2, 4},
        {2, 2, 2, 2, 2, 2, 2, 2},
        {1, 1, 2, 2, 4, 6},
        {1, 1, 2, 2, 4, 6},
        {1, 1, 1, 1, 1, 1, 4, 6},
        {1, 1, 2, 2, 4, 6},
        {1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1},
    };
    return groups;
}

void reduce_g_points(CoefficientArray&                    array,
                     std::size_t                          band_axis,
                     std::size_t                          g_point_axis,
                     const std::vector<std::vector<int>>& groups,
                     ReductionRule                        rule)
{
    if (array.shape.empty() || array.shape[0] != 2 || band_axis >= array.shape.size() ||
        g_point_axis >= array.shape.size() || array.shape[band_axis] != groups.size() ||
        array.shape[g_point_axis] != 16)
        throw std::runtime_error("RRTMG coefficient reduction shape is invalid");
    std::vector<std::size_t> stride(array.shape.size(), 1);
    for (std::size_t axis = array.shape.size(); axis-- > 1;)
        stride[axis - 1] = stride[axis] * array.shape[axis];
    std::size_t other_count = 1;
    for (std::size_t axis = 1; axis < array.shape.size(); ++axis)
        if (axis != band_axis && axis != g_point_axis)
            other_count *= array.shape[axis];

    for (std::size_t band = 0; band < groups.size(); ++band)
    {
        std::size_t original_begin = 0;
        for (std::size_t reduced = 0; reduced < groups[band].size(); ++reduced)
        {
            const std::size_t group_size = static_cast<std::size_t>(groups[band][reduced]);
            if (group_size == 0 || original_begin + group_size > 16)
                throw std::runtime_error("RRTMG g-point grouping is invalid");
            double weight_sum = 0.0;
            for (std::size_t original = original_begin; original < original_begin + group_size; ++original)
                weight_sum += quadrature_weights[original];
            for (std::size_t other = 0; other < other_count; ++other)
            {
                std::size_t remainder     = other;
                std::size_t common_offset = 0;
                for (std::size_t axis = array.shape.size(); axis-- > 1;)
                {
                    if (axis == band_axis || axis == g_point_axis)
                        continue;
                    const std::size_t coordinate = remainder % array.shape[axis];
                    remainder /= array.shape[axis];
                    common_offset += coordinate * stride[axis];
                }
                double value = 0.0;
                for (std::size_t original = original_begin; original < original_begin + group_size; ++original)
                {
                    const double weight = rule == ReductionRule::Sum ? 1.0 : quadrature_weights[original] / weight_sum;
                    const std::size_t source =
                        band * stride[band_axis] + original * stride[g_point_axis] + common_offset;
                    value += weight * array.values[source];
                }
                const std::size_t destination =
                    stride[0] + band * stride[band_axis] + reduced * stride[g_point_axis] + common_offset;
                array.values[destination] = value;
            }
            original_begin += group_size;
        }
        if (original_begin != 16)
            throw std::runtime_error("RRTMG g-point grouping does not cover 16 points");
    }
}

void reduce_longwave(LongwaveCoefficients& c)
{
    const auto& g = longwave_groups();
    reduce_g_points(c.planck_fraction_lower, 1, 3, g, ReductionRule::Sum);
    reduce_g_points(c.planck_fraction_upper, 1, 3, g, ReductionRule::Sum);
    reduce_g_points(c.key_absorption_lower, 1, 2, g, ReductionRule::WeightedAverage);
    reduce_g_points(c.key_absorption_upper, 1, 2, g, ReductionRule::WeightedAverage);
    reduce_g_points(c.water_self_absorption, 1, 2, g, ReductionRule::WeightedAverage);
    reduce_g_points(c.water_foreign_absorption, 1, 2, g, ReductionRule::WeightedAverage);
    reduce_g_points(c.minor_absorption_lower, 1, 3, g, ReductionRule::WeightedAverage);
    reduce_g_points(c.minor_absorption_upper, 1, 3, g, ReductionRule::WeightedAverage);
}

void reduce_shortwave(ShortwaveCoefficients& c)
{
    const auto& g = shortwave_groups();
    reduce_g_points(c.solar_source_lower, 1, 3, g, ReductionRule::Sum);
    reduce_g_points(c.solar_source_upper, 1, 3, g, ReductionRule::Sum);
    reduce_g_points(c.rayleigh_extinction_lower, 1, 3, g, ReductionRule::WeightedAverage);
    reduce_g_points(c.rayleigh_extinction_upper, 1, 3, g, ReductionRule::WeightedAverage);
    reduce_g_points(c.key_absorption_lower, 1, 2, g, ReductionRule::WeightedAverage);
    reduce_g_points(c.key_absorption_upper, 1, 2, g, ReductionRule::WeightedAverage);
    reduce_g_points(c.water_self_absorption, 1, 2, g, ReductionRule::WeightedAverage);
    reduce_g_points(c.water_foreign_absorption_lower, 1, 2, g, ReductionRule::WeightedAverage);
    reduce_g_points(c.water_foreign_absorption_upper, 1, 2, g, ReductionRule::WeightedAverage);
    reduce_g_points(c.minor_absorption_lower, 1, 3, g, ReductionRule::WeightedAverage);
    reduce_g_points(c.minor_absorption_upper, 1, 3, g, ReductionRule::WeightedAverage);
    reduce_g_points(c.quiet_sun_lower, 1, 3, g, ReductionRule::Sum);
    reduce_g_points(c.quiet_sun_upper, 1, 3, g, ReductionRule::Sum);
    reduce_g_points(c.facular_lower, 1, 3, g, ReductionRule::Sum);
    reduce_g_points(c.facular_upper, 1, 3, g, ReductionRule::Sum);
    reduce_g_points(c.sunspot_lower, 1, 3, g, ReductionRule::Sum);
    reduce_g_points(c.sunspot_upper, 1, 3, g, ReductionRule::Sum);
}

void validate_band_limits(const SpectralGrid& grid)
{
    const std::size_t bands = grid.band_numbers.size();
    if (grid.band_lower_wavenumber_cm.size() != bands || grid.band_upper_wavenumber_cm.size() != bands)
        throw std::runtime_error("RRTMG spectral band table has inconsistent shapes");
    for (std::size_t band = 0; band < bands; ++band)
        if (!std::isfinite(grid.band_lower_wavenumber_cm[band]) ||
            !std::isfinite(grid.band_upper_wavenumber_cm[band]) || grid.band_lower_wavenumber_cm[band] < 0.0 ||
            grid.band_upper_wavenumber_cm[band] <= grid.band_lower_wavenumber_cm[band])
            throw std::runtime_error("RRTMG spectral band limit is invalid");
}

SpectralGrid common_grid(const NetcdfReader& reader, std::size_t bands, int first_band, int expected_g_points)
{
    SpectralGrid grid;
    grid.absorber_names = reader.strings("AbsorberNames", {12, 5});
    const std::vector<std::string> expected_absorbers{
        "N2", "CCL4", "CFC11", "CFC12", "CFC22", "H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2"};
    if (grid.absorber_names != expected_absorbers)
        throw std::runtime_error("RRTMG absorber ordering is malformed");
    grid.band_numbers.resize(bands);
    std::iota(grid.band_numbers.begin(), grid.band_numbers.end(), first_band);
    grid.reduced_g_points_per_band = reduced_g_points(reader, bands, expected_g_points);
    grid.reduced_g_point_offsets   = g_point_offsets(grid.reduced_g_points_per_band);
    grid.band_lower_wavenumber_cm  = reader.vector("BandWavenumberLowerLimit", bands);
    grid.band_upper_wavenumber_cm  = reader.vector("BandWavenumberUpperLimit", bands);
    grid.reference_pressure_hpa    = reader.vector("Pressure", 59);
    grid.reference_temperature_k   = reader.vector("ReferenceTemperature", 59);
    grid.lower_key_species_ratios  = reader.vector("KeySpeciesRatioLowerAtmos", 9);
    grid.upper_key_species_ratios  = reader.vector("KeySpeciesRatioUpperAtmos", 5);
    grid.temperature_offsets_k     = reader.vector("TemperatureDiffFromMLS", 5);
    validate_band_limits(grid);
    return grid;
}

} // namespace
#endif

bool coefficient_loading_available() noexcept
{
#ifdef HAVE_NETCDF
    return true;
#else
    return false;
#endif
}

LongwaveCoefficients load_longwave_coefficients(const std::string& path)
{
#ifdef HAVE_NETCDF
    NetcdfReader reader(path);
    reader.require_dimension("Band", 16);
    reader.require_dimension("GPoint", 16);
    reader.require_dimension("GPointSet", 2);
    reader.require_dimension("Absorber", 12);
    reader.require_dimension("Pressure", 59);
    reader.require_dimension("PressureLowerAtmos", 13);
    reader.require_dimension("PressureUpperAtmos", 47);

    LongwaveCoefficients result;
    result.identity                        = identity(reader, "RRTM_LW_k-distribution_data_source_version", "Longwave");
    result.grid                            = common_grid(reader, 16, 1, 140);
    result.lower_reference_pressure_hpa    = reader.vector("PressureLowerAtmos", 13);
    result.upper_reference_pressure_hpa    = reader.vector("PressureUpperAtmos", 47);
    result.self_continuum_temperature_k    = reader.vector("TemperatureH20Self", 10);
    result.foreign_continuum_temperature_k = reader.vector("TemperatureH20Foreign", 4);
    result.foreign_continuum_pressure_hpa  = reader.vector("PressureH20Foreign", 4);
    result.absorption_temperature_k        = reader.vector("Temperature", 19);
    result.planck_temperature_k            = reader.vector("TemperaturePlanckValues", 181);
    result.planck_fraction_upper           = reader.doubles("PlanckFractionUpperAtmos", {2, 16, 5, 16});
    result.planck_fraction_lower           = reader.doubles("PlanckFractionLowerAtmos", {2, 16, 9, 16});
    result.key_absorption_lower  = reader.doubles("KeySpeciesAbsorptionCoefficientsLowerAtmos", {2, 16, 16, 13, 5, 9});
    result.key_absorption_upper  = reader.doubles("KeySpeciesAbsorptionCoefficientsUpperAtmos", {2, 16, 16, 47, 5, 5});
    result.water_self_absorption = reader.doubles("H20SelfAbsorptionCoefficients", {2, 16, 16, 10});
    result.water_foreign_absorption   = reader.doubles("H20ForeignAbsorptionCoefficients", {2, 16, 16, 4});
    result.minor_absorption_lower     = reader.doubles("AbsorptionCoefficientsLowerAtmos", {2, 16, 12, 16, 19, 9});
    result.minor_absorption_upper     = reader.doubles("AbsorptionCoefficientsUpperAtmos", {2, 16, 12, 16, 19, 5});
    result.absorber_amount_mls        = reader.doubles("AbsorberAmountMLS", {59, 12});
    result.integrated_planck_function = reader.doubles("IntegratedPlanckFunction", {16, 181});
    result.integrated_planck_function_band_16 = reader.doubles("IntegratedPlanckFunctionBand16", {181});
    reduce_longwave(result);
    return result;
#else
    (void)path;
    throw std::runtime_error("native RRTMG coefficient loading requires a Nimbus build with netCDF");
#endif
}

ShortwaveCoefficients load_shortwave_coefficients(const std::string& path)
{
#ifdef HAVE_NETCDF
    NetcdfReader reader(path);
    reader.require_dimension("Band", 14);
    reader.require_dimension("GPoint", 16);
    reader.require_dimension("GPointSet", 2);
    reader.require_dimension("Absorber", 12);
    reader.require_dimension("Pressure", 59);
    reader.require_dimension("PressureLowerAtmos", 13);
    reader.require_dimension("PressureUpperAtmos", 47);

    ShortwaveCoefficients result;
    result.identity                     = identity(reader, "RRTM_SW_k-distribution_data_source_version", "Shortwave");
    result.grid                         = common_grid(reader, 14, 16, 112);
    const std::vector<int> band_numbers = reader.integers("RRTMBandNumber", {14});
    if (band_numbers != result.grid.band_numbers)
        throw std::runtime_error("RRTMG shortwave band numbering is malformed");
    result.lower_reference_pressure_hpa          = reader.vector("PressureLowerAtmos", 13);
    result.upper_reference_pressure_hpa          = reader.vector("PressureUpperAtmos", 47);
    result.self_continuum_temperature_k          = reader.vector("TemperatureH2OSelf", 10);
    result.foreign_continuum_temperature_lower_k = reader.vector("TemperatureH2OForeignLowerAtmos", 3);
    result.foreign_continuum_temperature_upper_k = reader.vector("TemperatureH2OForeignUpperAtmos", 2);
    result.foreign_continuum_pressure_hpa        = reader.vector("PressureH2OForeign", 4);
    result.absorption_temperature_k              = reader.vector("Temperature", 19);
    result.lower_key_species_names               = reader.strings("KeySpeciesNamesLowerAtmos", {2, 14, 3});
    result.upper_key_species_names               = reader.strings("KeySpeciesNamesUpperAtmos", {2, 14, 3});
    result.solar_source_lower                    = reader.doubles("SolarSourceFunctionLowerAtmos", {2, 14, 9, 16});
    result.solar_source_upper                    = reader.doubles("SolarSourceFunctionUpperAtmos", {2, 14, 5, 16});
    result.rayleigh_extinction_lower = reader.doubles("RayleighExtinctionCoefficientsLowerAtmos", {2, 14, 9, 16});
    result.rayleigh_extinction_upper = reader.doubles("RayleighExtinctionCoefficientsUpperAtmos", {2, 14, 5, 16});
    result.key_absorption_lower  = reader.doubles("KeySpeciesAbsorptionCoefficientsLowerAtmos", {2, 14, 16, 13, 5, 9});
    result.key_absorption_upper  = reader.doubles("KeySpeciesAbsorptionCoefficientsUpperAtmos", {2, 14, 16, 47, 5, 5});
    result.water_self_absorption = reader.doubles("H2OSelfAbsorptionCoefficients", {2, 14, 16, 10});
    result.water_foreign_absorption_lower =
        reader.doubles("H2OForeignAbsorptionCoefficientsLowerAtmos", {2, 14, 16, 3});
    result.water_foreign_absorption_upper =
        reader.doubles("H2OForeignAbsorptionCoefficientsUpperAtmos", {2, 14, 16, 2});
    result.minor_absorption_lower = reader.doubles("AbsorptionCoefficientsLowerAtmos", {2, 14, 12, 16, 19, 9});
    result.minor_absorption_upper = reader.doubles("AbsorptionCoefficientsUpperAtmos", {2, 14, 12, 16, 19, 5});
    result.quiet_sun_upper        = reader.doubles("NRLSSI2SSFQuietSunUpperAtmos", {2, 14, 5, 16});
    result.facular_upper          = reader.doubles("NRLSSI2SSFFacularUpperAtmos", {2, 14, 5, 16});
    result.sunspot_upper          = reader.doubles("NRLSSI2SSFSunspotUpperAtmos", {2, 14, 5, 16});
    result.quiet_sun_lower        = reader.doubles("NRLSSI2SSFQuietSunLowerAtmos", {2, 14, 9, 16});
    result.facular_lower          = reader.doubles("NRLSSI2SSFFacularLowerAtmos", {2, 14, 9, 16});
    result.sunspot_lower          = reader.doubles("NRLSSI2SSFSunspotLowerAtmos", {2, 14, 9, 16});
    reduce_shortwave(result);
    return result;
#else
    (void)path;
    throw std::runtime_error("native RRTMG coefficient loading requires a Nimbus build with netCDF");
#endif
}

namespace
{

template <typename Coefficients>
using CoefficientCache = std::unordered_map<std::string, std::weak_ptr<const Coefficients>>;

std::mutex& coefficient_cache_mutex()
{
    static std::mutex mutex;
    return mutex;
}

template <typename Coefficients, typename Loader>
std::shared_ptr<const Coefficients>
shared_coefficients(const std::string& path, CoefficientCache<Coefficients>& cache, std::mutex& mutex, Loader&& loader)
{
    std::lock_guard<std::mutex> lock(mutex);
    auto                        existing = cache.find(path);
    if (existing != cache.end())
    {
        if (auto coefficients = existing->second.lock())
            return coefficients;
    }

    auto coefficients = std::make_shared<const Coefficients>(loader(path));
    cache[path]       = coefficients;
    return coefficients;
}

} // namespace

std::shared_ptr<const LongwaveCoefficients> shared_longwave_coefficients(const std::string& path)
{
    static CoefficientCache<LongwaveCoefficients> cache;
    return shared_coefficients(path, cache, coefficient_cache_mutex(), load_longwave_coefficients);
}

std::shared_ptr<const ShortwaveCoefficients> shared_shortwave_coefficients(const std::string& path)
{
    static CoefficientCache<ShortwaveCoefficients> cache;
    return shared_coefficients(path, cache, coefficient_cache_mutex(), load_shortwave_coefficients);
}

} // namespace tmv::rrtmg
