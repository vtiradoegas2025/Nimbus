#include "triple_snow_melting.hpp"
#include "triple_sedimentation.hpp"
#include "distribution.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tmv::milbrandt_detail
{
namespace
{

bool valid_state(const MeltingTripleMomentState& state, TripleMomentSpecies species,
                 double& alpha, double& lambda, std::string& error)
{
    for (double value : {state.mass_per_kg, state.number_per_kg, state.reflectivity_per_kg})
    {
        if (!std::isfinite(value) || value < 0.0 || value > std::numeric_limits<float>::max() ||
            (value != 0.0 && value < std::numeric_limits<float>::denorm_min()))
        {
            error = "retained triple snow melting q/N/Z is not FP32 representable";
            return false;
        }
    }
    if (state.mass_per_kg == 0.0 && state.number_per_kg == 0.0 && state.reflectivity_per_kg == 0.0)
    {
        alpha = 0.0;
        lambda = 0.0;
        return true;
    }
    return diagnose_triple_gamma_shape(state.mass_per_kg, state.number_per_kg,
        state.reflectivity_per_kg, current_nimbus_triple_sedimentation_parameters(species).mass_law,
        alpha, lambda, error);
}

bool round_endpoint(const MeltingTripleMomentState& state, TripleMomentSpecies species,
                    MeltingTripleMomentEndpoint& result, std::string& error)
{
    double alpha = 0.0;
    double lambda = 0.0;
    if (!valid_state(state, species, alpha, lambda, error))
    {
        return false;
    }
    const MeltingTripleMomentEndpoint candidate{
        static_cast<float>(state.mass_per_kg), static_cast<float>(state.number_per_kg),
        static_cast<float>(state.reflectivity_per_kg)};
    if (!valid_state({candidate.mass_per_kg, candidate.number_per_kg, candidate.reflectivity_per_kg},
                     species, alpha, lambda, error))
    {
        return false;
    }
    result = candidate;
    return true;
}

bool source(const std::array<double, 16>& cell, double duration,
            const MeltingNoCollectionProfile& profile, const RetainedSnowMeltingTripleState& initial,
            double& alpha, double& lambda, double& mass_squared, std::string& error)
{
    if (!std::isfinite(duration) || duration <= 0.0 || !std::isfinite(cell[0]) || cell[0] < 150.0 ||
        !std::isfinite(cell[1]) || cell[1] < 100.0 || cell[1] > 200000.0 ||
        !std::isfinite(cell[2]) || cell[2] <= 0.0)
    {
        error = "retained triple snow melting context is invalid";
        return false;
    }
    for (std::size_t index = 3; index < cell.size(); ++index)
    {
        if (!std::isfinite(cell[index]) || cell[index] < 0.0)
        {
            error = "retained triple snow melting water or number input is invalid";
            return false;
        }
    }
    if (!validate_melting_no_collection_profile(profile, error) ||
        !validate_retained_snow_melting_unowned_channels(
            {cell[4], cell[6], cell[8], cell[9], cell[10], cell[12], cell[14], cell[15]}, error))
    {
        return false;
    }
    if (cell[5] != initial.rain.mass_per_kg || cell[11] != initial.rain.number_per_kg ||
        cell[7] != initial.snow.mass_per_kg || cell[13] != initial.snow.number_per_kg)
    {
        error = "retained triple snow melting cell differs from exact source q/N";
        return false;
    }
    double rain_alpha = 0.0;
    double rain_lambda = 0.0;
    if (!valid_state(initial.rain, TripleMomentSpecies::Rain, rain_alpha, rain_lambda, error) ||
        !valid_state(initial.snow, TripleMomentSpecies::Snow, alpha, lambda, error))
    {
        return false;
    }
    mass_squared = 0.0;
    if (initial.snow.mass_per_kg > 0.0)
    {
        const double a = alpha + 1.0;
        const double d = current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Snow).mass_law.exponent;
        const double ratio = std::exp(std::lgamma(a + 2.0 * d) + std::lgamma(a) - 2.0 * std::lgamma(a + d));
        mass_squared = initial.snow.mass_per_kg * initial.snow.mass_per_kg
            / initial.snow.number_per_kg * ratio;
        if (!std::isfinite(mass_squared) || mass_squared <= 0.0)
        {
            error = "retained triple snow melting mass-squared moment is invalid";
            return false;
        }
    }
    return true;
}

bool make_event(const std::array<double, 16>& cell, double duration,
                const RetainedSnowMeltingTripleState& initial, double alpha, double lambda,
                double mass_squared, double mass, RetainedSnowMeltingAcceptedEvent& event,
                std::string& error)
{
    if (!std::isfinite(mass) || mass < 0.0 || mass > initial.snow.mass_per_kg ||
        (cell[0] <= microphysics_constants::T0 && mass != 0.0))
    {
        error = "retained triple snow melting admitted mass is invalid";
        return false;
    }
    const double fraction = initial.snow.mass_per_kg > 0.0 ? mass / initial.snow.mass_per_kg : 0.0;
    const double number = initial.snow.number_per_kg * fraction;
    const double c_r = current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Rain).mass_law.coefficient;
    const double rain_z = fraction * mass_squared / (c_r * c_r);
    if (!std::isfinite(number) || !std::isfinite(rain_z) ||
        (mass > 0.0 && (!(fraction > 0.0) || !(number > 0.0) || !(rain_z > 0.0))))
    {
        error = "retained triple snow melting nonzero transfer is unrepresentable";
        return false;
    }
    event = {initial.snow.mass_per_kg, initial.snow.number_per_kg, initial.snow.reflectivity_per_kg,
        alpha, lambda, mass_squared, mass, number, fraction, rain_z,
        cell[0], cell[1], cell[2], cell[3], duration};
    return true;
}

bool same_event(const RetainedSnowMeltingAcceptedEvent& a, const RetainedSnowMeltingAcceptedEvent& b) noexcept
{
    return a.entry_mass_per_kg == b.entry_mass_per_kg && a.entry_number_per_kg == b.entry_number_per_kg &&
        a.entry_reflectivity_per_kg == b.entry_reflectivity_per_kg && a.source_alpha == b.source_alpha &&
        a.source_lambda_per_m == b.source_lambda_per_m && a.source_mass_squared_per_kg == b.source_mass_squared_per_kg &&
        a.admitted_mass_transfer_per_kg == b.admitted_mass_transfer_per_kg &&
        a.admitted_number_transfer_per_kg == b.admitted_number_transfer_per_kg &&
        a.admitted_fraction == b.admitted_fraction && a.admitted_rain_reflectivity_per_kg == b.admitted_rain_reflectivity_per_kg &&
        a.temperature_k == b.temperature_k && a.pressure_pa == b.pressure_pa &&
        a.dry_density_kg_m3 == b.dry_density_kg_m3 && a.vapor_per_kg == b.vapor_per_kg && a.duration_s == b.duration_s;
}

} // namespace

bool validate_retained_snow_melting_unowned_channels(const std::array<double, 8>& channels, std::string& error)
{
    for (std::size_t index = 0; index < channels.size(); ++index)
    {
        if (channels[index] != 0.0)
        {
            error = std::string("retained triple snow melting observed unowned channel: ")
                + retained_snow_melting_unowned_channel_names[index];
            return false;
        }
    }
    error.clear();
    return true;
}

bool apply_retained_triple_snow_melting(const std::array<double, 16>& cell, double duration_s,
    const MeltingNoCollectionProfile& profile, const RetainedSnowMeltingTripleState& initial,
    const RetainedSnowMeltingAcceptedEvent& event, RetainedSnowMeltingTripleResult& result, std::string& error)
{
    double alpha = 0.0;
    double lambda = 0.0;
    double mass_squared = 0.0;
    if (!source(cell, duration_s, profile, initial, alpha, lambda, mass_squared, error))
    {
        return false;
    }
    RetainedSnowMeltingAcceptedEvent expected;
    if (!make_event(cell, duration_s, initial, alpha, lambda, mass_squared,
                    event.admitted_mass_transfer_per_kg, expected, error))
    {
        return false;
    }
    if (!same_event(event, expected))
    {
        error = "retained triple snow melting event differs from source, moment mapping or context";
        return false;
    }
    RetainedSnowMeltingTripleResult candidate;
    candidate.event = event;
    candidate.unrounded = {
        {initial.rain.mass_per_kg + event.admitted_mass_transfer_per_kg,
         initial.rain.number_per_kg + event.admitted_number_transfer_per_kg,
         initial.rain.reflectivity_per_kg + event.admitted_rain_reflectivity_per_kg},
        {initial.snow.mass_per_kg - event.admitted_mass_transfer_per_kg,
         initial.snow.number_per_kg - event.admitted_number_transfer_per_kg,
         initial.snow.reflectivity_per_kg - event.admitted_fraction * initial.snow.reflectivity_per_kg}};
    if (!round_endpoint(candidate.unrounded.rain, TripleMomentSpecies::Rain, candidate.rain, error) ||
        !round_endpoint(candidate.unrounded.snow, TripleMomentSpecies::Snow, candidate.snow, error))
    {
        return false;
    }
    result = candidate;
    error.clear();
    return true;
}

bool evaluate_retained_triple_snow_melting_cell(const std::array<double, 16>& cell, double duration_s,
    const MeltingNoCollectionProfile& profile, const RetainedSnowMeltingTripleState& initial,
    std::array<float, 11>& rates, RetainedSnowMeltingTripleResult& result, std::string& error)
{
    double alpha = 0.0;
    double lambda = 0.0;
    double mass_squared = 0.0;
    if (!source(cell, duration_s, profile, initial, alpha, lambda, mass_squared, error))
    {
        return false;
    }
    double mass = 0.0;
    if (cell[0] > microphysics_constants::T0 && initial.snow.mass_per_kg > 0.0)
    {
        const auto air = diagnose_air_transport(cell[0], cell[1], cell[2]);
        for (double value : {air.vapor_diffusivity, air.thermal_conductivity, air.kinematic_viscosity,
                             air.schmidt_one_third, air.density_factor})
        {
            if (!std::isfinite(value) || value <= 0.0)
            {
                error = "retained triple snow melting air transport is invalid";
                return false;
            }
        }
        constexpr double pi = 3.14159265358979323846;
        const double a = alpha + 1.0;
        const double r_eq = std::cbrt(initial.snow.mass_per_kg /
            ((pi / 6.0 * 100.0) * initial.snow.number_per_kg * a * (a + 1.0) * (a + 2.0)));
        constexpr double s = (3.0 + 0.41) / 2.0;
        const double sphere_term = 0.78 * a * r_eq + 0.30 * air.schmidt_one_third
            * std::sqrt(air.density_factor * 11.72 / air.kinematic_viscosity)
            * std::exp(std::lgamma(a + s) - std::lgamma(a)) * std::pow(r_eq, s);
        const double heat_supply = air.thermal_conductivity * (cell[0] - microphysics_constants::T0)
            - cell[2] * microphysics_constants::L_v * air.vapor_diffusivity *
                (thermodynamics::saturation_mixing_ratio_water(microphysics_constants::T0, cell[1]) - cell[3]);
        const double thermal_rate = 2.0 * pi / microphysics_constants::L_f
            * initial.snow.number_per_kg * sphere_term * heat_supply;
        if (!std::isfinite(r_eq) || r_eq <= 0.0 || !std::isfinite(sphere_term) || sphere_term <= 0.0 ||
            !std::isfinite(heat_supply) || !std::isfinite(thermal_rate))
        {
            error = "retained triple snow melting spherical heat closure is unrepresentable";
            return false;
        }
        const double uncapped = duration_s * thermal_rate;
        if (std::isnan(uncapped))
        {
            error = "retained triple snow melting uncapped increment is invalid";
            return false;
        }
        mass = std::clamp(uncapped, 0.0, initial.snow.mass_per_kg);
    }
    RetainedSnowMeltingAcceptedEvent event;
    if (!make_event(cell, duration_s, initial, alpha, lambda, mass_squared, mass, event, error))
    {
        return false;
    }
    const double mass_rate = event.admitted_mass_transfer_per_kg / duration_s;
    const double number_rate = event.admitted_number_transfer_per_kg / duration_s;
    const double heat_rate = -microphysics_constants::L_f / microphysics_constants::cp * mass / duration_s;
    if ((mass != 0.0 && (mass_rate == 0.0 || heat_rate == 0.0)) ||
        (event.admitted_number_transfer_per_kg != 0.0 && number_rate == 0.0))
    {
        error = "retained triple snow melting nonzero rate underflowed";
        return false;
    }
    const std::array<double, 11> raw_rates{0.0, -mass_rate, 0.0, 0.0, mass_rate, heat_rate,
                                        number_rate, 0.0, -number_rate, 0.0, 0.0};
    for (double rate : raw_rates)
    {
        if (!melting_rate_representable(rate))
        {
            error = "retained triple snow melting rate is not FP32 representable";
            return false;
        }
    }
    RetainedSnowMeltingTripleResult candidate;
    if (!apply_retained_triple_snow_melting(cell, duration_s, profile, initial, event, candidate, error))
    {
        return false;
    }
    std::array<float, 11> candidate_rates{};
    for (std::size_t index = 0; index < candidate_rates.size(); ++index)
    {
        candidate_rates[index] = static_cast<float>(raw_rates[index]);
    }
    rates = candidate_rates;
    result = candidate;
    error.clear();
    return true;
}

} // namespace tmv::milbrandt_detail
