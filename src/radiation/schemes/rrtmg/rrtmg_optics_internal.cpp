#include "rrtmg_optics_internal.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <stdexcept>

namespace tmv::rrtmg::optics_internal
{
namespace
{

constexpr std::size_t coefficient_g_points = 16;
constexpr std::size_t temperature_count    = 5;
constexpr double      column_scale         = 1.0e-20;
constexpr double      one_minus            = 1.0 - 1.0e-12;

bool has_shape(const CoefficientArray& table, std::initializer_list<std::size_t> expected) noexcept
{
    return table.shape.size() == expected.size() &&
           std::equal(table.shape.begin(), table.shape.end(), expected.begin());
}

std::string normalized_name(std::string name)
{
    name.erase(std::remove_if(name.begin(),
                              name.end(),
                              [](unsigned char character) { return character == ' ' || character == '\0'; }),
               name.end());
    std::transform(name.begin(),
                   name.end(),
                   name.begin(),
                   [](unsigned char character) { return static_cast<char>(std::toupper(character)); });
    return name;
}

double reference_amount(const CoefficientArray& reference_amounts, std::size_t pressure, std::size_t absorber)
{
    return reference_amounts.values[pressure * 12 + absorber];
}

double key_value(const CoefficientArray& table,
                 std::size_t             bands,
                 std::size_t             pressure_count,
                 std::size_t             ratio_count,
                 std::size_t             band,
                 std::size_t             g_point,
                 std::size_t             pressure,
                 std::size_t             temperature,
                 std::size_t             ratio)
{
    const std::size_t index =
        (((((bands + band) * coefficient_g_points + g_point) * pressure_count + pressure) * temperature_count +
          temperature) *
             ratio_count +
         ratio);
    return table.values[index];
}

std::pair<std::size_t, double> ratio_cell(double eta, std::size_t count)
{
    if (count < 2)
        return {0, 0.0};
    const double      coordinate = static_cast<double>(count - 1) * std::clamp(eta, 0.0, one_minus);
    const std::size_t lower      = std::min(static_cast<std::size_t>(coordinate), count - 2);
    return {lower, coordinate - static_cast<double>(lower)};
}

double interpolate_key_cell(const CoefficientArray& table,
                            std::size_t             bands,
                            std::size_t             pressure_count,
                            std::size_t             ratio_count,
                            std::size_t             band,
                            std::size_t             g_point,
                            std::size_t             pressure,
                            std::size_t             temperature,
                            double                  eta,
                            bool                    fourth_order_ratio_edges)
{
    if (fourth_order_ratio_edges && ratio_count == 9 && eta < 0.125)
    {
        const double coordinate = 8.0 * std::clamp(eta, 0.0, one_minus);
        const double p          = coordinate - 1.0;
        const double p4         = p * p * p * p;
        const double weight0    = p4;
        const double weight1    = 1.0 - p - 2.0 * p4;
        const double weight2    = p + p4;
        const double candidate =
            weight0 * key_value(table, bands, pressure_count, ratio_count, band, g_point, pressure, temperature, 0) +
            weight1 * key_value(table, bands, pressure_count, ratio_count, band, g_point, pressure, temperature, 1) +
            weight2 * key_value(table, bands, pressure_count, ratio_count, band, g_point, pressure, temperature, 2);
        if (std::isfinite(candidate) && candidate >= 0.0)
            return candidate;
    }
    if (fourth_order_ratio_edges && ratio_count == 9 && eta > 0.875)
    {
        const double coordinate = 8.0 * std::clamp(eta, 0.0, one_minus);
        const double fraction   = coordinate - std::floor(coordinate);
        const double p          = -fraction;
        const double p4         = p * p * p * p;
        const double weight0    = p + p4;
        const double weight1    = 1.0 - p - 2.0 * p4;
        const double weight2    = p4;
        const double candidate =
            weight0 * key_value(table, bands, pressure_count, ratio_count, band, g_point, pressure, temperature, 6) +
            weight1 * key_value(table, bands, pressure_count, ratio_count, band, g_point, pressure, temperature, 7) +
            weight2 * key_value(table, bands, pressure_count, ratio_count, band, g_point, pressure, temperature, 8);
        if (std::isfinite(candidate) && candidate >= 0.0)
            return candidate;
    }
    const auto   ratio = ratio_cell(eta, ratio_count);
    const double lower =
        key_value(table, bands, pressure_count, ratio_count, band, g_point, pressure, temperature, ratio.first);
    const double upper = key_value(table,
                                   bands,
                                   pressure_count,
                                   ratio_count,
                                   band,
                                   g_point,
                                   pressure,
                                   temperature,
                                   ratio.first + (ratio_count > 1 ? 1 : 0));
    return lower + ratio.second * (upper - lower);
}

} // namespace

const GasProfile* find_gas(const PreparedAtmosphere& atmosphere, const std::string& name) noexcept
{
    const std::string key = normalized_name(name);
    if (key.empty())
        return nullptr;
    if (key == "N2")
        return &atmosphere.gases.nitrogen;
    if (key == "CCL4")
        return &atmosphere.gases.ccl4;
    if (key == "CFC11")
        return &atmosphere.gases.cfc11;
    if (key == "CFC12")
        return &atmosphere.gases.cfc12;
    if (key == "CFC22")
        return &atmosphere.gases.cfc22;
    if (key == "H2O")
        return &atmosphere.gases.water_vapor;
    if (key == "CO2")
        return &atmosphere.gases.carbon_dioxide;
    if (key == "O3")
        return &atmosphere.gases.ozone;
    if (key == "N2O")
        return &atmosphere.gases.nitrous_oxide;
    if (key == "CO")
        return &atmosphere.gases.carbon_monoxide;
    if (key == "CH4")
        return &atmosphere.gases.methane;
    if (key == "O2")
        return &atmosphere.gases.oxygen;
    return nullptr;
}

std::size_t absorber_index(const SpectralGrid& grid, const std::string& name)
{
    const std::string key = normalized_name(name);
    for (std::size_t index = 0; index < grid.absorber_names.size(); ++index)
    {
        if (normalized_name(grid.absorber_names[index]) == key)
            return index;
    }
    throw std::invalid_argument("RRTMG key species is absent from the absorber table");
}

KeySpecies key_species(const SpectralGrid&       grid,
                       const PreparedAtmosphere& atmosphere,
                       const std::string&        first,
                       const std::string&        second)
{
    KeySpecies result;
    result.first  = find_gas(atmosphere, first);
    result.second = find_gas(atmosphere, second);
    if (result.first == nullptr && result.second != nullptr)
        throw std::invalid_argument("RRTMG key-species pair has no first gas");
    if (result.first != nullptr)
        result.first_absorber = absorber_index(grid, first);
    if (result.second != nullptr)
        result.second_absorber = absorber_index(grid, second);
    return result;
}

double scaled_column(const GasProfile& gas, std::size_t layer) noexcept
{
    return column_scale * gas.column_molecules_cm2[layer];
}

double interpolate_key_absorption(const CoefficientArray&   table,
                                  std::size_t               bands,
                                  std::size_t               pressure_count,
                                  std::size_t               ratio_count,
                                  std::size_t               band,
                                  std::size_t               local_g_point,
                                  const LayerInterpolation& interpolation,
                                  const KeySpecies&         species,
                                  const CoefficientArray*   reference_amounts,
                                  std::size_t               layer,
                                  double                    fixed_reference_ratio,
                                  bool                      fourth_order_ratio_edges)
{
    if (species.first == nullptr)
        return 0.0;
    const std::size_t expected_values =
        2 * bands * coefficient_g_points * pressure_count * temperature_count * ratio_count;
    if (!has_shape(table, {2, bands, coefficient_g_points, pressure_count, temperature_count, ratio_count}) ||
        table.values.size() != expected_values)
        throw std::invalid_argument("RRTMG key-absorption table is malformed");
    if (reference_amounts != nullptr &&
        (!has_shape(*reference_amounts, {59, 12}) || reference_amounts->values.size() != 59 * 12))
        throw std::invalid_argument("RRTMG reference atmosphere is malformed");
    const std::size_t global_pressure = interpolation.pressure_index;
    const std::size_t table_pressure  = interpolation.lower_atmosphere ? global_pressure : global_pressure - 12;
    if (table_pressure + 1 >= pressure_count)
        throw std::invalid_argument("RRTMG pressure interpolation is out of range");

    const double first_column = scaled_column(*species.first, layer);
    const auto   endpoint     = [&](std::size_t pressure)
    {
        if (species.second == nullptr)
            return std::pair<double, double>{first_column, 0.0};
        double reference_ratio = fixed_reference_ratio;
        if (reference_amounts != nullptr)
        {
            const double denominator = reference_amount(*reference_amounts, pressure, species.second_absorber);
            if (!(denominator > 0.0))
                throw std::invalid_argument("RRTMG reference key-species ratio is invalid");
            reference_ratio = reference_amount(*reference_amounts, pressure, species.first_absorber) / denominator;
        }
        if (!std::isfinite(reference_ratio) || reference_ratio <= 0.0)
            throw std::invalid_argument("RRTMG fixed key-species ratio is invalid");
        const double combined = first_column + reference_ratio * scaled_column(*species.second, layer);
        const double eta      = combined > 0.0 ? first_column / combined : 0.0;
        return std::pair<double, double>{combined, eta};
    };
    const auto        lower             = endpoint(global_pressure);
    const auto        upper             = endpoint(global_pressure + 1);
    const auto&       weight            = interpolation.pressure_temperature_weights;
    const std::size_t lower_temperature = interpolation.temperature_index_at_lower_pressure;
    const std::size_t upper_temperature = interpolation.temperature_index_at_upper_pressure;

    const double lower_coefficient = weight[0] * interpolate_key_cell(table,
                                                                      bands,
                                                                      pressure_count,
                                                                      ratio_count,
                                                                      band,
                                                                      local_g_point,
                                                                      table_pressure,
                                                                      lower_temperature,
                                                                      lower.second,
                                                                      fourth_order_ratio_edges) +
                                     weight[2] * interpolate_key_cell(table,
                                                                      bands,
                                                                      pressure_count,
                                                                      ratio_count,
                                                                      band,
                                                                      local_g_point,
                                                                      table_pressure,
                                                                      lower_temperature + 1,
                                                                      lower.second,
                                                                      fourth_order_ratio_edges);
    const double upper_coefficient = weight[1] * interpolate_key_cell(table,
                                                                      bands,
                                                                      pressure_count,
                                                                      ratio_count,
                                                                      band,
                                                                      local_g_point,
                                                                      table_pressure + 1,
                                                                      upper_temperature,
                                                                      upper.second,
                                                                      fourth_order_ratio_edges) +
                                     weight[3] * interpolate_key_cell(table,
                                                                      bands,
                                                                      pressure_count,
                                                                      ratio_count,
                                                                      band,
                                                                      local_g_point,
                                                                      table_pressure + 1,
                                                                      upper_temperature + 1,
                                                                      upper.second,
                                                                      fourth_order_ratio_edges);
    return lower.first * lower_coefficient + upper.first * upper_coefficient;
}

double interpolate_ratio_table(const CoefficientArray& table,
                               std::size_t             bands,
                               std::size_t             ratio_count,
                               std::size_t             band,
                               std::size_t             local_g_point,
                               double                  eta)
{
    if (!has_shape(table, {2, bands, ratio_count, coefficient_g_points}) ||
        table.values.size() != 2 * bands * ratio_count * coefficient_g_points)
        throw std::invalid_argument("RRTMG ratio table is malformed");
    const auto cell  = ratio_cell(eta, ratio_count);
    const auto value = [&](std::size_t ratio)
    { return table.values[(((bands + band) * ratio_count + ratio) * coefficient_g_points + local_g_point)]; };
    const double lower = value(cell.first);
    const double upper = value(cell.first + (ratio_count > 1 ? 1 : 0));
    return lower + cell.second * (upper - lower);
}

double interpolate_continuum(const CoefficientArray& table,
                             std::size_t             bands,
                             std::size_t             temperatures,
                             std::size_t             band,
                             std::size_t             local_g_point,
                             std::size_t             temperature_index,
                             double                  temperature_fraction)
{
    if (!has_shape(table, {2, bands, coefficient_g_points, temperatures}) ||
        table.values.size() != 2 * bands * coefficient_g_points * temperatures || temperature_index + 1 >= temperatures)
        throw std::invalid_argument("RRTMG continuum table is malformed");
    const auto value = [&](std::size_t index)
    { return table.values[(((bands + band) * coefficient_g_points + local_g_point) * temperatures + index)]; };
    const double lower = value(temperature_index);
    return lower + temperature_fraction * (value(temperature_index + 1) - lower);
}

double interpolate_minor_absorption(const CoefficientArray& table,
                                    std::size_t             bands,
                                    std::size_t             ratio_count,
                                    std::size_t             band,
                                    std::size_t             absorber,
                                    std::size_t             local_g_point,
                                    std::size_t             temperature_index,
                                    double                  temperature_fraction,
                                    double                  eta)
{
    constexpr std::size_t absorbers    = 12;
    constexpr std::size_t temperatures = 19;
    if (!has_shape(table, {2, bands, absorbers, coefficient_g_points, temperatures, ratio_count}) ||
        table.values.size() != 2 * bands * absorbers * coefficient_g_points * temperatures * ratio_count ||
        absorber >= absorbers || temperature_index + 1 >= temperatures)
        throw std::invalid_argument("RRTMG minor-absorption table is malformed");
    const auto ratio = ratio_cell(eta, ratio_count);
    const auto value = [&](std::size_t temperature, std::size_t ratio_index)
    {
        return table
            .values[(((((bands + band) * absorbers + absorber) * coefficient_g_points + local_g_point) * temperatures +
                      temperature) *
                         ratio_count +
                     ratio_index)];
    };
    const auto at_temperature = [&](std::size_t temperature)
    {
        const double lower = value(temperature, ratio.first);
        const double upper = value(temperature, ratio.first + (ratio_count > 1 ? 1 : 0));
        return lower + ratio.second * (upper - lower);
    };
    const double lower = at_temperature(temperature_index);
    return lower + temperature_fraction * (at_temperature(temperature_index + 1) - lower);
}

double minor_cross_section(const CoefficientArray& table,
                           std::size_t             bands,
                           std::size_t             band,
                           std::size_t             absorber,
                           std::size_t             local_g_point,
                           std::size_t             ratio_count)
{
    constexpr std::size_t absorbers    = 12;
    constexpr std::size_t temperatures = 19;
    if (!has_shape(table, {2, bands, absorbers, coefficient_g_points, temperatures, ratio_count}) ||
        table.values.size() != 2 * bands * absorbers * coefficient_g_points * temperatures * ratio_count ||
        band >= bands || absorber >= absorbers || local_g_point >= coefficient_g_points)
        throw std::invalid_argument("RRTMG minor cross-section table is malformed");
    return table
        .values[((((bands + band) * absorbers + absorber) * coefficient_g_points + local_g_point) * temperatures) *
                ratio_count];
}

void require_finite_nonnegative(double value, const char* quantity)
{
    if (!std::isfinite(value) || value < 0.0)
        throw std::runtime_error(std::string("RRTMG produced invalid ") + quantity);
}

} // namespace tmv::rrtmg::optics_internal
