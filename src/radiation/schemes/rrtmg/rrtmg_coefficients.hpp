#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace tmv::rrtmg
{

struct CoefficientArray
{
    std::vector<std::size_t> shape;
    std::vector<double>      values;
};

struct CoefficientIdentity
{
    std::string title;
    std::string model_version;
    std::string continuum_version;
    std::string file_version;
};

struct SpectralGrid
{
    std::vector<std::string> absorber_names;
    std::vector<int>         band_numbers;
    std::vector<int>         reduced_g_points_per_band;
    std::vector<std::size_t> reduced_g_point_offsets;
    std::vector<double>      band_lower_wavenumber_cm;
    std::vector<double>      band_upper_wavenumber_cm;
    std::vector<double>      reference_pressure_hpa;
    std::vector<double>      reference_temperature_k;
    std::vector<double>      lower_key_species_ratios;
    std::vector<double>      upper_key_species_ratios;
    std::vector<double>      temperature_offsets_k;
};

struct LongwaveCoefficients
{
    CoefficientIdentity identity;
    SpectralGrid        grid;
    std::vector<double> lower_reference_pressure_hpa;
    std::vector<double> upper_reference_pressure_hpa;
    std::vector<double> self_continuum_temperature_k;
    std::vector<double> foreign_continuum_temperature_k;
    std::vector<double> foreign_continuum_pressure_hpa;
    std::vector<double> absorption_temperature_k;
    std::vector<double> planck_temperature_k;

    CoefficientArray planck_fraction_lower;
    CoefficientArray planck_fraction_upper;
    CoefficientArray key_absorption_lower;
    CoefficientArray key_absorption_upper;
    CoefficientArray water_self_absorption;
    CoefficientArray water_foreign_absorption;
    CoefficientArray minor_absorption_lower;
    CoefficientArray minor_absorption_upper;
    CoefficientArray absorber_amount_mls;
    CoefficientArray integrated_planck_function;
    CoefficientArray integrated_planck_function_band_16;
};

struct ShortwaveCoefficients
{
    CoefficientIdentity      identity;
    SpectralGrid             grid;
    std::vector<double>      lower_reference_pressure_hpa;
    std::vector<double>      upper_reference_pressure_hpa;
    std::vector<double>      self_continuum_temperature_k;
    std::vector<double>      foreign_continuum_temperature_lower_k;
    std::vector<double>      foreign_continuum_temperature_upper_k;
    std::vector<double>      foreign_continuum_pressure_hpa;
    std::vector<double>      absorption_temperature_k;
    std::vector<std::string> lower_key_species_names;
    std::vector<std::string> upper_key_species_names;

    CoefficientArray solar_source_lower;
    CoefficientArray solar_source_upper;
    CoefficientArray rayleigh_extinction_lower;
    CoefficientArray rayleigh_extinction_upper;
    CoefficientArray key_absorption_lower;
    CoefficientArray key_absorption_upper;
    CoefficientArray water_self_absorption;
    CoefficientArray water_foreign_absorption_lower;
    CoefficientArray water_foreign_absorption_upper;
    CoefficientArray minor_absorption_lower;
    CoefficientArray minor_absorption_upper;
    CoefficientArray quiet_sun_lower;
    CoefficientArray quiet_sun_upper;
    CoefficientArray facular_lower;
    CoefficientArray facular_upper;
    CoefficientArray sunspot_lower;
    CoefficientArray sunspot_upper;
};

bool coefficient_loading_available() noexcept;

LongwaveCoefficients  load_longwave_coefficients(const std::string& path);
ShortwaveCoefficients load_shortwave_coefficients(const std::string& path);

// Runtime instances share immutable coefficient tables by coefficient path. The
// value-returning loaders above remain available for schema validation and
// focused coefficient tests.
std::shared_ptr<const LongwaveCoefficients>  shared_longwave_coefficients(const std::string& path);
std::shared_ptr<const ShortwaveCoefficients> shared_shortwave_coefficients(const std::string& path);

} // namespace tmv::rrtmg
