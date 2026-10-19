#include "microphysics/schemes/thompson/thermodynamic_helpers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <limits>
#include "microphysics/base/thermodynamics.hpp"

namespace tmv::thompson
{
namespace
{
float storage_upper(float value)
{
    return std::nextafter(value, std::numeric_limits<float>::max());
}
float storage_lower(float value)
{
    return std::nextafter(value, 0.0f);
}
void require_resolution_inputs(float thermal, float pressure, float vapor, float evaporated_cloud)
{
    if (!std::isfinite(thermal) || thermal <= 0 || !std::isfinite(pressure) || pressure <= 0 || !std::isfinite(vapor) ||
        vapor < 0 || !std::isfinite(evaporated_cloud) || evaporated_cloud < 0)
        throw std::invalid_argument("invalid Thompson saturation-resolution state");
}
LiquidSaturationResolution saturation_interval(
    float temperature_lower, float temperature_upper, float pressure, float vapor, float evaporated_cloud)
{
    constexpr double latent      = microphysics_constants::L_v / microphysics_constants::cp;
    double           vapor_lower = storage_lower(vapor), vapor_upper = storage_upper(vapor);
    if (evaporated_cloud > 0)
    {
        temperature_lower = float(double(temperature_lower) - latent * storage_upper(evaporated_cloud));
        temperature_upper = float(double(temperature_upper) - latent * storage_lower(evaporated_cloud));
        vapor_lower += storage_lower(evaporated_cloud);
        vapor_upper += storage_upper(evaporated_cloud);
    }
    const double upper = thermodynamics::saturation_mixing_ratio_water(temperature_upper, storage_lower(pressure));
    const double lower = thermodynamics::saturation_mixing_ratio_water(temperature_lower, storage_upper(pressure));
    return {vapor_lower > upper, vapor_upper < lower};
}
} // namespace

LiquidSaturationResolution
liquid_saturation_resolution(float temperature, float pressure, float vapor, float evaporated_cloud)
{
    require_resolution_inputs(temperature, pressure, vapor, evaporated_cloud);
    return saturation_interval(
        storage_lower(temperature), storage_upper(temperature), pressure, vapor, evaporated_cloud);
}

LiquidSaturationResolution
liquid_saturation_resolution_from_theta(float theta, float pressure, float vapor, float evaporated_cloud)
{
    require_resolution_inputs(theta, pressure, vapor, evaporated_cloud);
    const float lower = float(thermodynamics::theta_to_temperature(storage_lower(theta), storage_lower(pressure)));
    const float upper = float(thermodynamics::theta_to_temperature(storage_upper(theta), storage_upper(pressure)));
    return saturation_interval(lower, upper, pressure, vapor, evaporated_cloud);
}

namespace
{

void require_positive(double value, const char* message)
{
    if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument(message);
}

double wet_bulb_potential_temperature(double theta_e_k)
{
    constexpr std::array<double, 7> low = {
        -1.00922292e-10, -1.47945344e-8, -1.7303757e-6, -0.00012709, 1.15849867e-6, -3.518296861e-9, 3.5741522e-12};
    constexpr std::array<double, 7> high = {
        0.0, -3.5223513e-10, -5.7250807e-8, -5.83975422e-6, 4.72445163e-8, -1.13402845e-10, 8.729580402e-14};
    const double x = std::min(475.0, theta_e_k);
    const auto&  c = x <= 335.5 ? low : high;
    return c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * (c[5] + x * c[6]))))) + 273.15;
}

double temperature_from_equivalent_potential(double theta_e_k, double pressure_pa)
{
    double guess =
        (theta_e_k - 0.5 * std::pow(std::max(theta_e_k - 270.0, 0.0), 1.05)) * std::pow(pressure_pa / 100000.0, 0.2);
    for (int iteration = 0; iteration < 100; ++iteration)
    {
        const double w1          = liquid_saturation_mixing_ratio(pressure_pa, guess);
        const double w2          = liquid_saturation_mixing_ratio(pressure_pa, guess + 1.0);
        const double theta1      = equivalent_potential_temperature(pressure_pa, guess, w1, guess);
        const double theta2      = equivalent_potential_temperature(pressure_pa, guess + 1.0, w2, guess + 1.0);
        const double denominator = theta2 - theta1;
        if (!std::isfinite(denominator) || denominator == 0.0)
            break;
        const double correction = (theta_e_k - theta1) / denominator;
        if (!std::isfinite(correction))
            break;
        guess += correction;
        if (std::abs(correction) < 0.01)
            return guess;
    }
    return wet_bulb_potential_temperature(theta_e_k) * std::pow(pressure_pa / 100000.0, 0.286);
}

} // namespace

double liquid_saturation_mixing_ratio(double pressure_pa, double temperature_k)
{
    require_positive(pressure_pa, "invalid Thompson saturation pressure");
    require_positive(temperature_k, "invalid Thompson saturation temperature");
    constexpr std::array<double, 9> c              = {0.611583699e3,
                                                      0.444606896e2,
                                                      0.143177157e1,
                                                      0.264224321e-1,
                                                      0.299291081e-3,
                                                      0.203154182e-5,
                                                      0.702620698e-8,
                                                      0.379534310e-11,
                                                      -0.321582393e-13};
    const double                    x              = std::max(-80.0, temperature_k - 273.16);
    double                          vapor_pressure = c[8];
    for (int index = 7; index >= 0; --index)
        vapor_pressure = c[static_cast<std::size_t>(index)] + x * vapor_pressure;
    vapor_pressure      = std::min(vapor_pressure, pressure_pa * 0.15);
    const double result = 0.622 * vapor_pressure / (pressure_pa - vapor_pressure);
    if (!std::isfinite(result) || result < 0.0)
        throw std::overflow_error("invalid Thompson saturation result");
    return result;
}

double dewpoint_temperature(double pressure_pa, double mixing_ratio)
{
    require_positive(pressure_pa, "invalid Thompson dewpoint pressure");
    if (!std::isfinite(mixing_ratio) || mixing_ratio < 0.0)
        throw std::invalid_argument("invalid Thompson dewpoint mixing ratio");
    const double adjusted       = mixing_ratio + 1.0e-8;
    const double vapor_pressure = pressure_pa * adjusted / (0.622 + adjusted);
    const double logarithm      = std::log(vapor_pressure);
    const double result         = (35.86 * logarithm - 4947.2325) / (logarithm - 23.6837);
    require_positive(result, "invalid Thompson dewpoint result");
    return result;
}

double lifting_condensation_temperature(double temperature_k, double dewpoint_k)
{
    require_positive(temperature_k, "invalid Thompson LCL temperature");
    require_positive(dewpoint_k, "invalid Thompson LCL dewpoint");
    const double denominator = 1.0 / (dewpoint_k - 56.0) + std::log(temperature_k / dewpoint_k) / 800.0;
    const double result      = 1.0 / denominator + 56.0;
    require_positive(result, "invalid Thompson LCL result");
    return result;
}

double equivalent_potential_temperature(double pressure_pa,
                                        double temperature_k,
                                        double mixing_ratio,
                                        double lcl_temperature_k)
{
    require_positive(pressure_pa, "invalid Thompson theta-e pressure");
    require_positive(temperature_k, "invalid Thompson theta-e temperature");
    require_positive(lcl_temperature_k, "invalid Thompson theta-e LCL");
    if (!std::isfinite(mixing_ratio) || mixing_ratio < 0.0)
        throw std::invalid_argument("invalid Thompson theta-e mixing ratio");
    const double adjusted  = mixing_ratio + 1.0e-8;
    const double dry_theta = temperature_k * std::pow(100000.0 / pressure_pa, 0.2854 * (1.0 - 0.28 * adjusted));
    const double exponent  = (3.376 / lcl_temperature_k - 0.00254) * (adjusted * 1000.0) * (1.0 + 0.81 * adjusted);
    const double result    = dry_theta * std::exp(exponent);
    require_positive(result, "invalid Thompson theta-e result");
    return result;
}

double wet_bulb_temperature(double pressure_pa, double temperature_k, double mixing_ratio)
{
    require_positive(temperature_k, "invalid Thompson wet-bulb temperature");
    const double saturation = liquid_saturation_mixing_ratio(pressure_pa, temperature_k);
    if (mixing_ratio >= 0.999 * saturation)
        return temperature_k;
    const double dewpoint = std::min(temperature_k - 0.001, dewpoint_temperature(pressure_pa, mixing_ratio));
    const double lcl      = lifting_condensation_temperature(temperature_k, dewpoint);
    const double theta_e  = equivalent_potential_temperature(pressure_pa, temperature_k, mixing_ratio, lcl);
    const double result   = std::min(temperature_k, temperature_from_equivalent_potential(theta_e, pressure_pa));
    require_positive(result, "invalid Thompson wet-bulb result");
    return result;
}

} // namespace tmv::thompson
