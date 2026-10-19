#pragma once

#include "activation.hpp"
#include "triple_cold_cloud_diffusion.hpp"
#include "triple_source_eos_energy.hpp"
#include "microphysics/base/thermodynamics.hpp"

namespace tmv::milbrandt_detail
{

struct HailSharedVaporAdmission
{
    double surface_proposal = 0.0;
    double snow_proposal = 0.0;
    double surface_transfer = 0.0;
    double snow_transfer = 0.0;
    double original_vapor_remaining = 0.0;
    double common_factor = 1.0;
    unsigned downward_rounding_steps = 0;
};

inline bool admit_hail_shared_vapor(double original_vapor, double surface_proposal, double snow_proposal,
                                    HailSharedVaporAdmission& result, std::string& error)
{
    if (!std::isfinite(original_vapor) || original_vapor < 0.0 || !std::isfinite(surface_proposal) ||
        !std::isfinite(snow_proposal) || snow_proposal < 0.0)
    { error = "wet shared vapor has an invalid source or unowned snow sublimation"; return false; }
    HailSharedVaporAdmission candidate;
    candidate.surface_proposal = surface_proposal;
    candidate.snow_proposal = snow_proposal;
    const double demand = std::max(0.0, surface_proposal) + snow_proposal;
    if (!std::isfinite(demand)) { error = "wet shared vapor demand is unrepresentable"; return false; }
    if (demand > original_vapor) candidate.common_factor = original_vapor / demand;
    for (;;)
    {
        candidate.surface_transfer = surface_proposal > 0.0 ? candidate.common_factor * surface_proposal : surface_proposal;
        candidate.snow_transfer = candidate.common_factor * snow_proposal;
        const double remaining = original_vapor - std::max(0.0, candidate.surface_transfer);
        if (candidate.snow_transfer <= remaining)
        {
            candidate.original_vapor_remaining = remaining - candidate.snow_transfer;
            break;
        }
        if (candidate.downward_rounding_steps++ == 4)
        { error = "wet shared vapor cap cannot be represented without donor overdraw"; return false; }
        candidate.common_factor = std::nextafter(candidate.common_factor, 0.0);
    }
    if ((surface_proposal > 0.0 && candidate.surface_transfer == 0.0) ||
        (snow_proposal > 0.0 && candidate.snow_transfer == 0.0))
    { error = "wet shared vapor positive transfer underflows its admission"; return false; }
    result = candidate;
    error.clear();
    return true;
}

struct ColdCloudEosEquilibrium
{
    double cloud_transfer = 0.0;
    double cloud_number = 0.0;
    double activated_number_per_m3 = 0.0;
    double temperature = 0.0;
    double pressure = 0.0;
    double saturation_residual = 0.0;
    bool cloud_exhausted = false;
};

inline bool solve_cold_cloud_eos_equilibrium(const SourceEosEnergyState& before,
    const std::array<double, 7>& base_water, double reference_temperature,
    double original_cloud_remaining, double cloud_number_remaining, double original_vapor_remaining,
    double vertical_velocity, ColdCloudEosEquilibrium& result, std::string& error)
{
    using namespace physical_constants;
    const auto reject = [&](const char* message) { error = message; return false; };
    SourceEosEnergyInput identity;
    identity.before = before;
    std::copy(before.water.begin(), before.water.end(), identity.accepted_water.begin());
    identity.generation = 1;
    identity.duration_s = 1.0;
    SourceEosEnergyTransaction context;
    if (!context.prepare(identity, error)) return false;
    if (!std::isfinite(reference_temperature) || reference_temperature != context.receipt().temperature_eos_reference ||
        !(context.receipt().temperature_before > freezing_temperature_k - 40.0 &&
          context.receipt().temperature_before < freezing_temperature_k))
        return reject("cold cloud/EOS equilibrium has a foreign reference or source temperature");
    long double water_change = 0.0L, water_scale = 0.0L;
    for (std::size_t i = 0; i < base_water.size(); ++i)
    {
        if (!std::isfinite(base_water[i]) || base_water[i] < 0.0 || base_water[i] > std::numeric_limits<float>::max())
            return reject("cold cloud/EOS base water is invalid");
        water_change += static_cast<long double>(base_water[i]) - before.water[i];
        water_scale += before.water[i];
    }
    const double base_capacity = source_eos_energy_detail::capacity(base_water);
    if (!std::isfinite(base_capacity) || base_capacity <= 0.0 ||
        std::abs(water_change) > 64.0L * std::numeric_limits<double>::epsilon() * water_scale)
        return reject("cold cloud/EOS base does not own a closed material inventory");
    if (!std::isfinite(original_cloud_remaining) || original_cloud_remaining < 0.0 ||
        original_cloud_remaining != base_water[1] || !std::isfinite(cloud_number_remaining) || cloud_number_remaining < 0.0 ||
        (original_cloud_remaining > 0.0 && cloud_number_remaining <= 0.0) ||
        !std::isfinite(original_vapor_remaining) || original_vapor_remaining < 0.0 ||
        original_vapor_remaining > base_water[0] || !std::isfinite(vertical_velocity))
        return reject("cold cloud/EOS equilibrium has invalid original donor bounds");
    const auto water = [&](double transfer)
    {
        auto q = base_water;
        q[0] -= transfer;
        q[1] += transfer;
        return q;
    };
    const auto temperature = [&](double transfer)
    {
        const auto q = water(transfer);
        return reference_temperature - source_eos_energy_detail::composition_work(before.water, q, reference_temperature) /
            source_eos_energy_detail::capacity(q);
    };
    const auto transfer_at_temperature = [&](double t)
    {
        const double work = source_eos_energy_detail::composition_work(before.water, base_water, reference_temperature) +
            source_eos_energy_detail::capacity(base_water) * (t - reference_temperature);
        const double coefficient = (specific_heat_liquid_water_jkgk - specific_heat_water_vapor_cp_jkgk +
            gas_constant_water_vapor_jkgk) * (t - freezing_temperature_k) -
            latent_heat_vaporization_jkg + gas_constant_water_vapor_jkgk * freezing_temperature_k;
        return -work / coefficient;
    };
    const double lower_mass = -original_cloud_remaining;
    double lower = std::max(lower_mass, transfer_at_temperature(freezing_temperature_k - 40.0));
    double upper = std::min(original_vapor_remaining, transfer_at_temperature(freezing_temperature_k));
    if (!std::isfinite(lower) || !std::isfinite(upper) || lower > upper)
        return reject("cold cloud/EOS equilibrium has no owned cold interval");
    const auto residual = [&](double transfer)
    {
        const double t = temperature(transfer);
        const double p = before.density * gas_constant_dry_air_jkgk * t;
        return base_water[0] - transfer - thermodynamics::saturation_mixing_ratio_water(t, p);
    };
    double lower_residual = residual(lower), upper_residual = residual(upper);
    if (!std::isfinite(lower_residual) || !std::isfinite(upper_residual))
        return reject("cold cloud/EOS saturation bracket is unrepresentable");
    ColdCloudEosEquilibrium candidate;
    if (lower_residual < 0.0)
    {
        if (lower != lower_mass)
            return reject("cold cloud/EOS evaporation crosses its cold interval");
        candidate.cloud_transfer = lower;
        candidate.cloud_exhausted = true;
    }
    else
    {
        if (upper_residual > 0.0)
            return reject("cold cloud/EOS condensation requires unowned vapor or a warm transition");
        for (int iteration = 0; iteration < 96 && std::nextafter(lower, upper) < upper; ++iteration)
        {
            const double middle = lower + .5 * (upper - lower);
            if (middle == lower || middle == upper) break;
            const double value = residual(middle);
            if (!std::isfinite(value)) return reject("cold cloud/EOS root is unrepresentable");
            if (value > 0.0) { lower = middle; lower_residual = value; }
            else { upper = middle; upper_residual = value; }
        }
        candidate.cloud_transfer = std::abs(lower_residual) < std::abs(upper_residual) ? lower : upper;
        candidate.cloud_exhausted = candidate.cloud_transfer == lower_mass && candidate.cloud_transfer < 0.0;
    }
    candidate.temperature = temperature(candidate.cloud_transfer);
    candidate.pressure = before.density * gas_constant_dry_air_jkgk * candidate.temperature;
    candidate.saturation_residual = residual(candidate.cloud_transfer);
    if (!(candidate.temperature > freezing_temperature_k - 40.0 && candidate.temperature < freezing_temperature_k))
        return reject("cold cloud/EOS root lies on an unowned thermal boundary");
    const double cloud_after = original_cloud_remaining + candidate.cloud_transfer;
    candidate.cloud_number = cloud_number_remaining;
    if (candidate.cloud_transfer > 0.0)
    {
        candidate.activated_number_per_m3 = continental_activated_ccn(vertical_velocity, candidate.temperature, candidate.pressure);
        candidate.cloud_number = std::max(cloud_number_remaining, candidate.activated_number_per_m3 / before.density);
    }
    else if (candidate.cloud_transfer < 0.0)
        candidate.cloud_number = cloud_number_remaining * (cloud_after / original_cloud_remaining);
    if (!std::isfinite(candidate.cloud_number) || candidate.cloud_number < 0.0 ||
        (cloud_after > 0.0 && candidate.cloud_number <= 0.0))
        return reject("cold cloud/EOS equilibrium number endpoint is invalid");
    if (!candidate.cloud_exhausted && std::abs(candidate.saturation_residual) >
        16.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(base_water[0])))
        return reject("cold cloud/EOS equilibrium cannot close its saturation residual");
    const auto endpoint_water = water(candidate.cloud_transfer);
    for (std::size_t i = 0; i < endpoint_water.size(); ++i)
        if (!source_eos_energy_detail::bounded(endpoint_water[i], 0.0, i == 0 ? qv_max_kgkg : hydrometeor_max_kgkg) ||
            (endpoint_water[i] != 0.0 && endpoint_water[i] < std::numeric_limits<float>::denorm_min()))
            return reject("cold cloud/EOS water endpoint is outside its storage domain");
    if (!source_eos_energy_detail::bounded(candidate.pressure, pressure_min_pa, pressure_max_pa))
        return reject("cold cloud/EOS pressure endpoint requires repair");
    result = candidate;
    error.clear();
    return true;
}

} // namespace tmv::milbrandt_detail
