#pragma once

#include "distribution.hpp"
#include "melting_accepted_events.hpp"
#include "microphysics/milbrandt_cold_parameters.hpp"
#include <array>

namespace tmv::milbrandt_detail
{
template <bool PublishedTriple>
inline void melting_cell_impl(const std::array<double, 13>& parameters,
                              const std::array<double, 12>& cell,
                              double dt,
                              std::array<float, 11>& rates,
                              const MeltingNoCollectionProfile* profile = nullptr,
                              MeltingAcceptedEventBuffer* observer = nullptr)
{
    constexpr double pi                  = 3.14159265358979323846;
    constexpr double snow_spherical_mass = pi / 6.0 * 100.0;
    const double     T                   = cell[0];
    const double     temperature_c       = T - microphysics_constants::T0;
    if constexpr (PublishedTriple)
    {
        std::string error;
        if (!profile || !observer || observer->capacity > observer->events.size() ||
            observer->size > observer->capacity ||
            !validate_triple_melting_context(cell, dt, *profile, error))
        {
            throw std::runtime_error(error.empty()
                ? "published triple melting event buffer or profile is invalid" : error);
        }
        if (observer->capacity - observer->size < 2)
        {
            throw std::runtime_error("published triple melting event capacity is exhausted");
        }
        for (float rate : rates)
        {
            if (rate != 0.0f)
            {
                throw std::runtime_error("published triple melting has an incoming unowned tendency");
            }
        }
        const auto& p = default_cold_parameters;
        const std::array<double, 13> expected{
            p[9], p[4], p[14], parameters[3], p[15], p[20], p[0], p[10],
            parameters[8], p[16], p[21], p[1], p[11]};
        if (parameters != expected || !std::isfinite(parameters[3]) || parameters[3] <= -1.0 ||
            !std::isfinite(parameters[8]) || parameters[8] <= -1.0)
        {
            throw std::runtime_error("published triple melting physical parameters differ");
        }
        for (const bool graupel : {true, false})
        {
            const double q = cell[graupel ? 6 : 7];
            const double n = cell[graupel ? 10 : 11];
            const auto distribution = diagnose_distribution(
                cell[2], q, n, parameters[graupel ? 3 : 8], parameters[graupel ? 4 : 9], 3.0);
            if ((q == 0.0) != (n == 0.0) ||
                (q > 0.0 && (!(distribution.intercept > 0.0) ||
                             !(distribution.inverse_lambda > 0.0))))
            {
                throw std::runtime_error("published triple melting distribution is unrepresentable");
            }
        }
    }
    if (!(temperature_c > 0.0))
    {
        if constexpr (PublishedTriple)
        {
            const std::array<MeltingAcceptedEvent, 2> events{
                make_melting_accepted_event(MeltingFrozenSpecies::Graupel,
                    cell[6], cell[10], parameters[3], 0.0, 0.0,
                    cell[0], cell[1], cell[2], cell[3], dt),
                make_melting_accepted_event(MeltingFrozenSpecies::Hail,
                    cell[7], cell[11], parameters[8], 0.0, 0.0,
                    cell[0], cell[1], cell[2], cell[3], dt)};
            for (const auto& event : events)
            {
                observer->events[observer->size++] = event;
            }
        }
        return;
    }
    const double       pressure = cell[1];
    const double       density  = cell[2];
    const AirTransport air      = diagnose_air_transport(T, pressure, density);
    if constexpr (PublishedTriple)
    {
        if (cell[6] > 0.0 || (cell[7] > 0.0 && temperature_c > 5.0))
        {
            for (double value : {air.vapor_diffusivity, air.thermal_conductivity,
                                 air.kinematic_viscosity, air.schmidt_one_third, air.density_factor})
            {
                if (!std::isfinite(value) || value <= 0.0)
                {
                    throw std::runtime_error("published triple melting air transport is invalid");
                }
            }
        }
    }
    const double       saturation_at_freezing =
        thermodynamics::saturation_mixing_ratio_water(microphysics_constants::T0, pressure);
    const double vapor_deficit_at_freezing = saturation_at_freezing - static_cast<double>(cell[3]);

    // Pristine ice is assumed to melt completely within the
    // microphysics step, as in the operational double-moment
    // closure.
    const double melted_ice        = std::max(0.0, static_cast<double>(cell[4]));
    const double melted_ice_number = melted_ice > 0.0 ? std::max(0.0, static_cast<double>(cell[8])) : 0.0;

    const auto melted_increment = [&](double category_q,
                                      double category_n,
                                      double alpha,
                                      double c_mass,
                                      double d_mass,
                                      double fall_a,
                                      double fall_b,
                                      bool   thermally_active)
    {
        if (!thermally_active)
            return 0.0;
        const GammaDistribution distribution =
            diagnose_distribution(density, category_q, category_n, alpha, c_mass, d_mass);
        if (!(distribution.intercept > 0.0))
            return 0.0;
        const double ventilation = ventilation_integral(distribution, alpha, fall_a, fall_b, air);
        const double heat_supply = [&]
        {
            if constexpr (PublishedTriple)
            {
                return air.thermal_conductivity * temperature_c -
                    density * microphysics_constants::L_v * air.vapor_diffusivity * vapor_deficit_at_freezing;
            }
            else
            {
                return air.thermal_conductivity * temperature_c -
                    microphysics_constants::L_v * air.vapor_diffusivity * vapor_deficit_at_freezing;
            }
        }();
        if constexpr (PublishedTriple)
        {
            if (!std::isfinite(ventilation) || ventilation <= 0.0 || !std::isfinite(heat_supply))
            {
                throw std::runtime_error("published triple melting ventilation or heat supply is invalid");
            }
        }
        const double increment   = dt * 2.0 * pi / (density * microphysics_constants::L_f) * distribution.intercept *
                                   ventilation * heat_supply;
        if constexpr (PublishedTriple)
        {
            if (std::isnan(increment))
            {
                throw std::runtime_error("published triple melting uncapped increment is invalid");
            }
        }
        return std::clamp(increment, 0.0, category_q);
    };

    // Snow ventilation in the published melt equation uses its
    // spherical 100 kg m-3 equivalent distribution.
    const double melted_snow =
        melted_increment(cell[5], cell[9], parameters[0], snow_spherical_mass, 3.0, parameters[1], parameters[2], true);
    const double melted_graupel = melted_increment(
        cell[6], cell[10], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], true);
    const double melted_hail = melted_increment(cell[7],
                                                cell[11],
                                                parameters[8],
                                                parameters[9],
                                                parameters[10],
                                                parameters[11],
                                                parameters[12],
                                                temperature_c > 5.0);

    const double melted_snow_number    = cell[5] > 0.0 ? cell[9] * melted_snow / cell[5] : 0.0;
    const double melted_graupel_number = [&]
    {
        if constexpr (PublishedTriple)
        {
            return cell[6] > 0.0 ? cell[10] * (melted_graupel / cell[6]) : 0.0;
        }
        else
        {
            return cell[6] > 0.0 ? cell[10] * melted_graupel / cell[6] : 0.0;
        }
    }();
    const double melted_hail_number = [&]
    {
        if constexpr (PublishedTriple)
        {
            return cell[7] > 0.0 ? cell[11] * (melted_hail / cell[7]) : 0.0;
        }
        else
        {
            return cell[7] > 0.0 ? cell[11] * melted_hail / cell[7] : 0.0;
        }
    }();
    std::array<MeltingAcceptedEvent, PublishedTriple ? 2 : 0> events{};
    if constexpr (PublishedTriple)
    {
        std::string error;
        if (!validate_melting_unowned_material_channels(
                {melted_ice, melted_snow, melted_ice_number, melted_snow_number}, error))
        {
            throw std::runtime_error(error);
        }
        events[0] = make_melting_accepted_event(MeltingFrozenSpecies::Graupel,
            cell[6], cell[10], parameters[3], melted_graupel, melted_graupel_number,
            cell[0], cell[1], cell[2], cell[3], dt);
        events[1] = make_melting_accepted_event(MeltingFrozenSpecies::Hail,
            cell[7], cell[11], parameters[8], melted_hail, melted_hail_number,
            cell[0], cell[1], cell[2], cell[3], dt);
    }
    const double total_melt            = melted_ice + melted_snow + melted_graupel + melted_hail;
    const double total_melt_number =
        melted_ice_number + melted_snow_number + melted_graupel_number + melted_hail_number;
    if constexpr (PublishedTriple)
    {
        const std::array<double, 11> raw_rates{
            -melted_ice / dt, -melted_snow / dt, -melted_graupel / dt, -melted_hail / dt,
            total_melt / dt, -microphysics_constants::L_f / microphysics_constants::cp * total_melt / dt,
            total_melt_number / dt, -melted_ice_number / dt, -melted_snow_number / dt,
            -melted_graupel_number / dt, -melted_hail_number / dt};
        for (double rate : raw_rates)
        {
            if (!melting_rate_representable(rate))
            {
                throw std::runtime_error("published triple melting rate is not FP32 representable");
            }
        }
    }

    rates[0] -= static_cast<float>(melted_ice / dt);
    rates[1] -= static_cast<float>(melted_snow / dt);
    rates[2] -= static_cast<float>(melted_graupel / dt);
    rates[3] -= static_cast<float>(melted_hail / dt);
    rates[4] += static_cast<float>(total_melt / dt);
    rates[5] -= static_cast<float>(microphysics_constants::L_f / microphysics_constants::cp * total_melt / dt);
    rates[6] += static_cast<float>(total_melt_number / dt);
    rates[7] -= static_cast<float>(melted_ice_number / dt);
    rates[8] -= static_cast<float>(melted_snow_number / dt);
    rates[9] -= static_cast<float>(melted_graupel_number / dt);
    rates[10] -= static_cast<float>(melted_hail_number / dt);
    if constexpr (PublishedTriple)
    {
        for (const auto& event : events)
        {
            observer->events[observer->size++] = event;
        }
    }
}

inline void melting_cell(const std::array<double, 13>& parameters,
                         const std::array<double, 12>& cell,
                         double dt,
                         std::array<float, 11>& rates)
{
    melting_cell_impl<false>(parameters, cell, dt, rates);
}

inline void published_triple_melting_cell(
    const std::array<double, 13>& parameters,
    const std::array<double, 12>& cell,
    double dt,
    const MeltingNoCollectionProfile& profile,
    std::array<float, 11>& rates,
    MeltingAcceptedEventBuffer& observer)
{
    melting_cell_impl<true>(parameters, cell, dt, rates, &profile, &observer);
}
} // namespace tmv::milbrandt_detail
