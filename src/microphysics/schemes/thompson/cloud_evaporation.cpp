#include "microphysics/schemes/thompson/cloud_evaporation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tmv::thompson
{
namespace
{

constexpr double pi = 3.1415926536;
constexpr double rho_water = 1000.0;
constexpr double mass_coefficient = pi * rho_water / 6.0;
constexpr double first_diameter_m = 1.0e-6;
constexpr double first_number_edge_cm3 = 1.0;
constexpr double last_number_edge_cm3 = 3000.0;

constexpr std::array<double, CloudEvaporationMetadata::cloud_contents>
    cloud_content_grid = {
        1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6,
        1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5,
        1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4,
        1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3,
        1e-2};

double cloud_number_grid(std::size_t index)
{
    const double log_span = std::log(last_number_edge_cm3
                                     / first_number_edge_cm3);
    const double lower = std::exp(static_cast<double>(index)
                                  / CloudEvaporationMetadata::cloud_numbers
                                  * log_span);
    const double upper = std::exp(static_cast<double>(index + 1)
                                  / CloudEvaporationMetadata::cloud_numbers
                                  * log_span);
    return std::sqrt(lower * upper) * 1.0e6;
}

int fortran_nint(double value)
{
    return static_cast<int>(std::floor(value + 0.5));
}

void require_finite_positive(double value, const char* name)
{
    if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument(std::string("invalid Thompson ") + name);
}

} // namespace

std::size_t cloud_evaporation_table_size()
{
    return CloudEvaporationMetadata::diameters
        * CloudEvaporationMetadata::cloud_contents
        * CloudEvaporationMetadata::cloud_numbers;
}

std::size_t cloud_evaporation_index(std::size_t diameter,
                                    std::size_t cloud_content,
                                    std::size_t cloud_number)
{
    if (diameter >= CloudEvaporationMetadata::diameters
        || cloud_content >= CloudEvaporationMetadata::cloud_contents
        || cloud_number >= CloudEvaporationMetadata::cloud_numbers)
        throw std::out_of_range("Thompson cloud-evaporation table index");
    return (cloud_number * CloudEvaporationMetadata::cloud_contents
            + cloud_content)
        * CloudEvaporationMetadata::diameters + diameter;
}

CloudEvaporationTable generate_cloud_evaporation_table()
{
    CloudEvaporationTable table;
    table.evaporated_mass_m3.resize(cloud_evaporation_table_size());
    table.evaporated_number_m3.resize(cloud_evaporation_table_size());
    std::array<double, CloudEvaporationMetadata::diameters> diameters{};
    std::array<double, CloudEvaporationMetadata::diameters> bin_widths{};
    std::array<double, CloudEvaporationMetadata::diameters> masses{};
    for (std::size_t i = 0; i < diameters.size(); ++i)
    {
        diameters[i] = static_cast<double>(i + 1) * first_diameter_m;
        bin_widths[i] = first_diameter_m;
        masses[i] = mass_coefficient * std::pow(diameters[i], 3.0);
    }

    for (std::size_t number_index = 0;
         number_index < CloudEvaporationMetadata::cloud_numbers;
         ++number_index)
    {
        const double number = cloud_number_grid(number_index);
        const int shape = std::min(15, fortran_nint(1.0e9 / number) + 2);
        const double gamma_number = std::tgamma(static_cast<double>(shape + 1));
        const double gamma_mass = std::tgamma(static_cast<double>(shape + 4));
        for (std::size_t content_index = 0;
             content_index < CloudEvaporationMetadata::cloud_contents;
             ++content_index)
        {
            const double slope = std::cbrt(
                number * mass_coefficient * gamma_mass
                / gamma_number / cloud_content_grid[content_index]);
            const double intercept = number / gamma_number
                * std::pow(slope, static_cast<double>(shape + 1));
            double accumulated_mass = 0.0;
            double accumulated_number = 0.0;
            for (std::size_t diameter_index = 0;
                 diameter_index < CloudEvaporationMetadata::diameters;
                 ++diameter_index)
            {
                const double density = intercept
                    * std::pow(diameters[diameter_index], shape)
                    * std::exp(-slope * diameters[diameter_index])
                    * bin_widths[diameter_index];
                accumulated_mass += masses[diameter_index] * density;
                accumulated_number += density;
                const auto index = cloud_evaporation_index(
                    diameter_index, content_index, number_index);
                table.evaporated_mass_m3[index] = accumulated_mass;
                table.evaporated_number_m3[index] = accumulated_number;
            }
        }
    }
    return table;
}

bool validate_cloud_evaporation_table(const CloudEvaporationTable& table,
                                      std::string& error)
{
    const auto expected = cloud_evaporation_table_size();
    if (table.evaporated_mass_m3.size() != expected
        || table.evaporated_number_m3.size() != expected)
    {
        error = "Thompson cloud-evaporation table dimension mismatch";
        return false;
    }
    for (std::size_t index = 0; index < expected; ++index)
    {
        if (!std::isfinite(table.evaporated_mass_m3[index])
            || table.evaporated_mass_m3[index] < 0.0
            || !std::isfinite(table.evaporated_number_m3[index])
            || table.evaporated_number_m3[index] < 0.0)
        {
            error = "invalid Thompson cloud-evaporation table value";
            return false;
        }
    }
    error.clear();
    return true;
}

std::size_t cloud_evaporation_diameter_index(double diameter_m)
{
    require_finite_positive(diameter_m, "evaporated cloud diameter");
    const int one_based = std::clamp(
        static_cast<int>(std::min(1.0e6 * diameter_m, static_cast<double>(CloudEvaporationMetadata::diameters))),
        1,
        static_cast<int>(CloudEvaporationMetadata::diameters));
    return static_cast<std::size_t>(one_based - 1);
}

std::size_t cloud_evaporation_content_index(double cloud_water_kg_m3)
{
    require_finite_positive(cloud_water_kg_m3, "cloud water content");
    if (cloud_water_kg_m3 <= cloud_content_grid.front()) return 0;
    const int decade = fortran_nint(std::log10(cloud_water_kg_m3));
    int exponent = decade;
    for (int candidate = decade - 1; candidate <= decade + 1; ++candidate)
    {
        const double normalized = cloud_water_kg_m3 / std::pow(10.0, candidate);
        if (normalized >= 1.0 && normalized < 10.0)
        {
            exponent = candidate;
            break;
        }
    }
    const int one_based = static_cast<int>(
        cloud_water_kg_m3 / std::pow(10.0, exponent))
        + 9 * (exponent + 6);
    return static_cast<std::size_t>(std::clamp(one_based, 1, 37) - 1);
}

std::size_t cloud_evaporation_number_index(double cloud_number_m3)
{
    require_finite_positive(cloud_number_m3, "cloud number");
    const double first = cloud_number_grid(0);
    const double last = cloud_number_grid(
        CloudEvaporationMetadata::cloud_numbers - 1);
    if (cloud_number_m3 <= first)
    {
        return 0;
    }
    if (cloud_number_m3 >= last)
    {
        return CloudEvaporationMetadata::cloud_numbers - 1;
    }
    const double span = std::log(last / first);
    const int one_based = fortran_nint(
        1.0 + CloudEvaporationMetadata::cloud_numbers
            * std::log(cloud_number_m3 / first) / span);
    return static_cast<std::size_t>(std::clamp(one_based, 1, 100) - 1);
}

CloudEvaporationNumberExchange return_evaporated_cloud_number(
    const CloudEvaporationTable& table,
    double maximum_evaporated_diameter_m,
    double cloud_water_kg_m3,
    double cloud_number_kg_dry,
    double dry_air_density_kg_m3)
{
    if (table.evaporated_mass_m3.size() != cloud_evaporation_table_size()
        || table.evaporated_number_m3.size() != cloud_evaporation_table_size())
        throw std::invalid_argument(
            "Thompson cloud-evaporation table dimension mismatch");
    require_finite_positive(cloud_number_kg_dry, "cloud number");
    require_finite_positive(dry_air_density_kg_m3, "dry-air density");
    const double cloud_number_m3 = cloud_number_kg_dry * dry_air_density_kg_m3;
    const auto index = cloud_evaporation_index(
        cloud_evaporation_diameter_index(maximum_evaporated_diameter_m),
        cloud_evaporation_content_index(cloud_water_kg_m3),
        cloud_evaporation_number_index(cloud_number_m3));
    if (!std::isfinite(table.evaporated_number_m3[index])
        || table.evaporated_number_m3[index] < 0.0)
        throw std::invalid_argument(
            "invalid Thompson cloud-evaporation table value");
    const double returned = std::min(
        0.99 * cloud_number_kg_dry,
        table.evaporated_number_m3[index] / dry_air_density_kg_m3);
    return {-returned, returned};
}

double maximum_fully_evaporated_cloud_diameter(
    double temperature_k,
    double pressure_pa,
    double dry_air_density_kg_m3,
    double vapor_kg_dry,
    double saturation_vapor_kg_dry,
    double dt_s)
{
    require_finite_positive(temperature_k, "temperature");
    require_finite_positive(pressure_pa, "pressure");
    require_finite_positive(dry_air_density_kg_m3, "dry-air density");
    require_finite_positive(saturation_vapor_kg_dry, "water saturation");
    require_finite_positive(dt_s, "microphysics timestep");
    if (!std::isfinite(vapor_kg_dry) || vapor_kg_dry < 0.0)
        throw std::invalid_argument("invalid Thompson vapor mixing ratio");
    const double supersaturation = vapor_kg_dry / saturation_vapor_kg_dry - 1.0;
    if (supersaturation >= -1.0e-6) return 0.0;
    const double temperature_c = temperature_k - 273.15;
    const double inverse_temperature = 1.0 / temperature_k;
    const double latent_heat = 2.5e6 + (2106.0 - 4218.0) * temperature_c;
    const double diffusivity = 2.11e-5 * std::pow(temperature_k / 273.15, 1.94)
        * (101325.0 / pressure_pa);
    const double conductivity = (5.69 + 0.0168 * temperature_c)
        * 1.0e-5 * 418.936;
    const double saturated_density = dry_air_density_kg_m3
        * saturation_vapor_kg_dry;
    const double derivative = saturated_density * inverse_temperature
        * (latent_heat * inverse_temperature / 461.5 - 1.0);
    const double second_derivative = saturated_density
        * (std::pow(inverse_temperature
                    * (latent_heat * inverse_temperature / 461.5 - 1.0), 2.0)
           - 2.0 * latent_heat * std::pow(inverse_temperature, 3.0) / 461.5
           + inverse_temperature * inverse_temperature);
    const double gamma = latent_heat * diffusivity / conductivity * derivative;
    double alpha = 0.5 * std::pow(gamma / (1.0 + gamma), 2.0)
        * second_derivative / derivative * saturated_density / derivative;
    alpha = std::max(1.0e-9, alpha);
    const double evaporation = 2.0 * pi
        * (1.0 - alpha * supersaturation
           + 2.0 * alpha * alpha * supersaturation * supersaturation
           - 5.0 * alpha * alpha * alpha
               * supersaturation * supersaturation * supersaturation)
        / (1.0 + gamma);
    const double squared = -2.0 * dt_s * evaporation / (2.0 * pi)
        * 4.0 * diffusivity * supersaturation * saturated_density / rho_water;
    if (!std::isfinite(squared) || squared <= 0.0)
        throw std::runtime_error("invalid Thompson cloud evaporation diameter");
    return std::sqrt(squared);
}

} // namespace tmv::thompson
