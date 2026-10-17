#pragma once

#include "collision.hpp"
#include "cold_rain_freezing_events.hpp"
#include "cold_frozen_diffusion_events.hpp"
#include "cold_joint_frozen_diffusion_events.hpp"
#include "cold_snow_riming_admission.hpp"
#include "microphysics/milbrandt_cold_parameters.hpp"
#include <array>
#include <utility>

namespace tmv::milbrandt_detail
{
struct ColdParameters
{
    double a_g_;
    double a_h_;
    double a_i_;
    double a_r_;
    double a_s_;
    double alpha_g_;
    double alpha_h_;
    double alpha_i_;
    double alpha_r_;
    double alpha_s_;
    double b_g_;
    double b_h_;
    double b_i_;
    double b_r_;
    double b_s_;
    double c_g_;
    double c_h_;
    double c_i_;
    double c_r_;
    double c_s_;
    double d_g_;
    double d_h_;
    double d_i_;
    double d_r_;
    double d_s_;
    bool   hail_processes;
};

template <bool PublishedFrozenDiffusion, bool JointFrozenDiffusion = false>
inline void cold_cell_impl(
    const ColdParameters& parameters, const std::array<double, 16>& cell, double dt,
    std::array<float, 14>& rates, ColdRainFreezingAcceptedEventBuffer* observer,
    ColdFrozenDiffusionAcceptedEventBuffer* diffusion_observer,
    ColdJointFrozenDiffusionAcceptedEventBuffer* joint_diffusion_observer = nullptr,
    ColdSnowRimingAdmissionObservation* snow_riming_observer = nullptr,
    unsigned snow_riming_max_passes = cold_snow_riming_admission_max_passes)
{
    static_assert(!JointFrozenDiffusion || PublishedFrozenDiffusion);
    if (snow_riming_max_passes == 0 || snow_riming_max_passes > cold_snow_riming_admission_max_passes)
    {
        throw ColdSnowRimingAdmissionError(ColdSnowRimingAdmissionFailure::InvalidResource,
            "cold snow riming admission resource cap is invalid");
    }
    ColdSnowRimingAdmissionObservation snow_riming;
    constexpr double pi                    = 3.14159265358979323846;
    constexpr double embryo_ice_mass       = 1.0e-12;
    constexpr double snow_embryo_diameter  = 125.0e-6;
    constexpr double maximum_ice_number_m3 = 1.0e7;

    if constexpr (PublishedFrozenDiffusion && !JointFrozenDiffusion)
    {
        if (!diffusion_observer ||
            (diffusion_observer->species != ColdFrozenDiffusionSpecies::Graupel &&
             diffusion_observer->species != ColdFrozenDiffusionSpecies::Hail) ||
            diffusion_observer->capacity > diffusion_observer->events.size() ||
            diffusion_observer->size > diffusion_observer->capacity ||
            !std::isfinite(dt) || dt <= 0.0 ||
            !std::isfinite(cell[0]) || cell[0] < 150.0 ||
            cell[0] >= microphysics_constants::T0 ||
            !std::isfinite(cell[1]) || cell[1] < 100.0 || cell[1] > 200000.0 ||
            !std::isfinite(cell[2]) || cell[2] <= 0.0)
        {
            throw std::runtime_error("published frozen diffusion context or event buffer is invalid");
        }
        for (std::size_t index = 3; index < cell.size(); ++index)
        {
            if (!std::isfinite(cell[index]) || cell[index] < 0.0)
            {
                throw std::runtime_error("published frozen diffusion water or number input is invalid");
            }
        }
        for (float rate : rates)
        {
            if (rate != 0.0f)
            {
                throw std::runtime_error("published frozen diffusion has an incoming unowned tendency");
            }
        }
        std::array<double, 25> expected = default_cold_parameters;
        const bool graupel = diffusion_observer->species == ColdFrozenDiffusionSpecies::Graupel;
        const double alpha = graupel ? parameters.alpha_g_ : parameters.alpha_h_;
        expected[graupel ? 5 : 6] = alpha;
        const std::array<double, 25> supplied{
            parameters.a_g_, parameters.a_h_, parameters.a_i_, parameters.a_r_, parameters.a_s_,
            parameters.alpha_g_, parameters.alpha_h_, parameters.alpha_i_, parameters.alpha_r_, parameters.alpha_s_,
            parameters.b_g_, parameters.b_h_, parameters.b_i_, parameters.b_r_, parameters.b_s_,
            parameters.c_g_, parameters.c_h_, parameters.c_i_, parameters.c_r_, parameters.c_s_,
            parameters.d_g_, parameters.d_h_, parameters.d_i_, parameters.d_r_, parameters.d_s_};
        if (supplied != expected || !std::isfinite(alpha) || alpha <= -1.0)
        {
            throw std::runtime_error("published frozen diffusion physical parameters differ");
        }
        const double q = cell[graupel ? 8 : 9];
        const double n = cell[graupel ? 14 : 15];
        const auto distribution = diagnose_distribution(
            cell[2], q, n, alpha,
            graupel ? parameters.c_g_ : parameters.c_h_, 3.0);
        if ((q == 0.0) != (n == 0.0) ||
            (q > 0.0 && (!(distribution.intercept > 0.0) ||
                         !(distribution.inverse_lambda > 0.0))))
        {
            throw std::runtime_error("published frozen diffusion distribution is unrepresentable");
        }
    }

    if constexpr (JointFrozenDiffusion)
    {
        if (!joint_diffusion_observer || observer || diffusion_observer ||
            joint_diffusion_observer->capacity > joint_diffusion_observer->events.size() ||
            joint_diffusion_observer->size > joint_diffusion_observer->capacity ||
            !std::isfinite(dt) || dt <= 0.0 ||
            !std::isfinite(cell[0]) || cell[0] < 150.0 ||
            cell[0] >= microphysics_constants::T0 ||
            !std::isfinite(cell[1]) || cell[1] < 100.0 || cell[1] > 200000.0 ||
            !std::isfinite(cell[2]) || cell[2] <= 0.0)
        {
            throw std::runtime_error("published joint frozen diffusion context or event buffer is invalid");
        }
        for (std::size_t index = 3; index < cell.size(); ++index)
        {
            if (!std::isfinite(cell[index]) || cell[index] < 0.0)
            {
                throw std::runtime_error("published joint frozen diffusion water or number input is invalid");
            }
        }
        for (float rate : rates)
        {
            if (rate != 0.0f)
            {
                throw std::runtime_error("published joint frozen diffusion has an incoming unowned tendency");
            }
        }
        std::array<double, 25> expected = default_cold_parameters;
        expected[5] = parameters.alpha_g_;
        expected[6] = parameters.alpha_h_;
        const std::array<double, 25> supplied{
            parameters.a_g_, parameters.a_h_, parameters.a_i_, parameters.a_r_, parameters.a_s_,
            parameters.alpha_g_, parameters.alpha_h_, parameters.alpha_i_, parameters.alpha_r_, parameters.alpha_s_,
            parameters.b_g_, parameters.b_h_, parameters.b_i_, parameters.b_r_, parameters.b_s_,
            parameters.c_g_, parameters.c_h_, parameters.c_i_, parameters.c_r_, parameters.c_s_,
            parameters.d_g_, parameters.d_h_, parameters.d_i_, parameters.d_r_, parameters.d_s_};
        if (supplied != expected || !parameters.hail_processes ||
            !std::isfinite(parameters.alpha_g_) || parameters.alpha_g_ <= -1.0 ||
            !std::isfinite(parameters.alpha_h_) || parameters.alpha_h_ <= -1.0)
        {
            throw std::runtime_error("published joint frozen diffusion physical parameters differ");
        }
        for (const bool graupel : {true, false})
        {
            const double q = cell[graupel ? 8 : 9];
            const double n = cell[graupel ? 14 : 15];
            const auto distribution = diagnose_distribution(
                cell[2], q, n, graupel ? parameters.alpha_g_ : parameters.alpha_h_,
                graupel ? parameters.c_g_ : parameters.c_h_, 3.0);
            if ((q == 0.0) != (n == 0.0) ||
                (q > 0.0 && (!(distribution.intercept > 0.0) ||
                             !(distribution.inverse_lambda > 0.0))))
            {
                throw std::runtime_error("published joint frozen diffusion distribution is unrepresentable");
            }
        }
    }

    if (observer)
    {
        if (observer->capacity > observer->events.size() || observer->size > observer->capacity)
        {
            throw std::runtime_error("Milbrandt rain-freezing observer buffer is malformed");
        }
        if (!std::isfinite(dt) || !(dt > 0.0) ||
            !std::isfinite(cell[2]) || !(cell[2] > 0.0))
        {
            throw std::runtime_error("Milbrandt rain-freezing observer received invalid duration or density");
        }
    }

    const double T = cell[0];
    if (!(T < microphysics_constants::T0))
        return;
    const double pressure = cell[1];
    const double density  = cell[2];
    // Number moments entering the cold process family include
    // saturation activation and the already-admitted warm-rain
    // update. Diagnosing cold rates from the stale beginning-of-
    // step moments would make process order depend on storage.
    const double cloud_number_state = std::max(0.0, static_cast<double>(cell[10]) + dt * static_cast<double>(rates[8]));
    const double rain_number_state  = std::max(0.0, static_cast<double>(cell[11]) + dt * static_cast<double>(rates[9]));
    const double ice_number_state  = std::max(0.0, static_cast<double>(cell[12]) + dt * static_cast<double>(rates[10]));
    const double snow_number_state = std::max(0.0, static_cast<double>(cell[13]) + dt * static_cast<double>(rates[11]));
    const double graupel_number_state =
        std::max(0.0, static_cast<double>(cell[14]) + dt * static_cast<double>(rates[12]));
    const double hail_number_state = std::max(0.0, static_cast<double>(cell[15]) + dt * static_cast<double>(rates[13]));
    const double temperature_c     = T - microphysics_constants::T0;
    const double vapor             = cell[3];
    const double qsi               = thermodynamics::saturation_mixing_ratio_ice(T, pressure);
    const double saturation_ice    = vapor / std::max(qsi, 1.0e-30);
    const AirTransport air         = diagnose_air_transport(T, pressure, density);
    const double       inverse_diffusion_resistance =
        1.0 / (microphysics_constants::L_s * microphysics_constants::L_s /
                   (air.thermal_conductivity * microphysics_constants::R_v * T * T) +
               1.0 / (density * std::max(qsi, 1.0e-30) * air.vapor_diffusivity));

    const auto admission_ratio = [&](double source, double sink)
    {
        if (!(sink > 0.0))
            return 1.0;
        double factor = std::clamp(std::max(0.0, source) / sink, 0.0, 1.0);
        // An actual overdraw may round its quotient to one; back off paired products instead of the budget guard.
        if (snow_riming.active && sink > source && factor > 0.0)
            factor *= cold_snow_riming_overdraw_factor;
        return factor;
    };

    double vapor_to_ice                     = 0.0;
    double cloud_to_ice                     = 0.0;
    double rain_to_frozen                   = 0.0;
    double ice_to_snow                      = 0.0;
    double nc_to_ice                        = 0.0;
    double nr_to_frozen                     = 0.0;
    double new_ice_number                   = 0.0;
    double ice_number_sublimated            = 0.0;
    double snow_number_sublimated           = 0.0;
    double graupel_number_sublimated        = 0.0;
    double hail_number_sublimated           = 0.0;
    double ice_number_to_snow               = 0.0;
    double cloud_to_snow                    = 0.0;
    double cloud_to_graupel                 = 0.0;
    double cloud_to_hail                    = 0.0;
    double cloud_number_to_snow             = 0.0;
    double cloud_number_to_graupel          = 0.0;
    double cloud_number_to_hail             = 0.0;
    double ice_collected_by_snow            = 0.0;
    double ice_collected_by_graupel         = 0.0;
    double ice_collected_by_hail            = 0.0;
    double ice_number_collected_by_snow     = 0.0;
    double ice_number_collected_by_graupel  = 0.0;
    double ice_number_collected_by_hail     = 0.0;
    double snow_self_collection             = 0.0;
    double splinter_mass_from_snow          = 0.0;
    double splinter_mass_from_graupel       = 0.0;
    double splinter_number_from_snow        = 0.0;
    double splinter_number_from_graupel     = 0.0;
    double rain_collected_by_ice            = 0.0;
    double ice_collected_by_rain            = 0.0;
    double rain_number_collected_by_ice     = 0.0;
    double ice_number_collected_by_rain     = 0.0;
    double rain_collected_by_snow           = 0.0;
    double snow_collected_by_rain           = 0.0;
    double rain_number_collected_by_snow    = 0.0;
    double snow_number_collected_by_rain    = 0.0;
    double rain_collected_by_graupel        = 0.0;
    double graupel_collected_by_rain        = 0.0;
    double rain_number_collected_by_graupel = 0.0;
    double graupel_number_collected_by_rain = 0.0;
    double collision_mass_to_snow           = 0.0;
    double collision_mass_to_graupel        = 0.0;
    double collision_mass_to_hail           = 0.0;
    double collision_number_to_snow         = 0.0;
    double collision_number_to_graupel      = 0.0;
    double collision_number_to_hail         = 0.0;
    double snow_to_graupel                  = 0.0;
    double snow_number_to_graupel           = 0.0;
    double graupel_to_hail                  = 0.0;
    double graupel_number_to_hail           = 0.0;
    double hail_number_from_graupel         = 0.0;
    double hail_to_graupel                  = 0.0;
    double hail_number_to_graupel           = 0.0;
    double rain_collected_by_hail           = 0.0;
    double rain_number_collected_by_hail    = 0.0;
    double snow_collected_by_hail           = 0.0;
    double snow_number_collected_by_hail    = 0.0;
    double shed_rain_from_hail              = 0.0;
    double shed_rain_number                 = 0.0;

    // Homogeneous cloud freezing follows the Bigg-volume
    // probability used by Milbrandt-Yau. It transfers the same
    // fraction of cloud mass and droplet number.
    const double cloud        = cell[4];
    const double cloud_number = cloud_number_state;
    if (cloud > 1.0e-14 && cloud_number > 0.0)
    {
        const double tc2                       = temperature_c * temperature_c;
        const double log10_j                   = std::max(-20.0,
                                                          -606.3952 - 52.6611 * temperature_c - 1.7439 * tc2 -
                                                              0.0265 * tc2 * temperature_c - 1.536e-4 * tc2 * tc2);
        const double droplet_diameter_cubed_cm = 1.0e6 * cloud / (cloud_number * parameters.c_r_);
        double       fraction = 1.0 - std::exp(-std::pow(10.0, log10_j) * (pi / 6.0) * droplet_diameter_cubed_cm * dt);
        if (temperature_c > -30.0)
            fraction = 0.0;
        if (temperature_c < -50.0)
            fraction = 1.0;
        fraction     = std::clamp(fraction, 0.0, 1.0);
        cloud_to_ice = fraction * cloud;
        nc_to_ice    = fraction * cloud_number;
    }

    // Cooper (1986) primary ice nucleation, the operational
    // Milbrandt-Yau option-2 closure.
    const double ice_number_volume = density * ice_number_state;
    if (temperature_c < -5.0 && saturation_ice > 1.08)
    {
        const double target_number =
            std::min(maximum_ice_number_m3, 5.0 * std::exp(0.304 * (microphysics_constants::T0 - std::max(233.0, T))));
        new_ice_number                         = std::max(0.0, target_number - ice_number_volume);
        const double maximum_number_from_vapor = density * std::max(0.0, vapor - qsi) / embryo_ice_mass;
        new_ice_number                         = std::min(new_ice_number, maximum_number_from_vapor);
        vapor_to_ice += embryo_ice_mass * new_ice_number / density;
    }

    const auto deposit_on_category = [&](double  category_q,
                                         double  category_n,
                                         double  alpha,
                                         double  c_mass,
                                         double  d_mass,
                                         double  fall_a,
                                         double  fall_b,
                                         double  capacitance_factor,
                                         bool    remove_number_during_sublimation,
                                         double& number_sublimated,
                                         double* raw_increment = nullptr)
    {
        const GammaDistribution distribution =
            diagnose_distribution(density, category_q, category_n, alpha, c_mass, d_mass);
        if (!(distribution.intercept > 0.0))
            return 0.0;
        const double ventilation = ventilation_integral(distribution, alpha, fall_a, fall_b, air);
        double       increment   = dt / density * capacitance_factor * inverse_diffusion_resistance * 2.0 * pi *
                                   (saturation_ice - 1.0) * distribution.intercept * ventilation;
        if (raw_increment)
        {
            *raw_increment = increment;
        }
        const double denominator = T - 7.66;
        const double saturation_response =
            1.0 + microphysics_constants::L_s * microphysics_constants::L_s * qsi /
                      (microphysics_constants::cp * microphysics_constants::R_v * denominator * denominator);
        const double vapor_limit = (vapor - qsi) / saturation_response;
        if (saturation_ice >= 1.0)
            increment = std::clamp(increment, 0.0, std::max(0.0, vapor_limit));
        else
            increment = std::clamp(increment, -category_q, 0.0);
        if (increment < 0.0 && remove_number_during_sublimation)
        {
            if constexpr (PublishedFrozenDiffusion)
            {
                number_sublimated = category_n * (increment / category_q);
            }
            else
            {
                number_sublimated = std::min(0.0, category_n / category_q * increment);
            }
        }
        return increment;
    };

    double ice_deposition     = deposit_on_category(cell[6],
                                                    ice_number_state,
                                                    parameters.alpha_i_,
                                                    parameters.c_i_,
                                                    parameters.d_i_,
                                                    parameters.a_i_,
                                                    parameters.b_i_,
                                                    0.5,
                                                    true,
                                                    ice_number_sublimated);
    double snow_deposition    = deposit_on_category(cell[7],
                                                    snow_number_state,
                                                    parameters.alpha_s_,
                                                    parameters.c_s_,
                                                    parameters.d_s_,
                                                    parameters.a_s_,
                                                    parameters.b_s_,
                                                    0.5,
                                                    true,
                                                    snow_number_sublimated,
                                                    &snow_riming.raw_ambient_snow_mass_increment_per_kg);
    double graupel_deposition = deposit_on_category(cell[8],
                                                    graupel_number_state,
                                                    parameters.alpha_g_,
                                                    parameters.c_g_,
                                                    parameters.d_g_,
                                                    parameters.a_g_,
                                                    parameters.b_g_,
                                                    1.0,
                                                    PublishedFrozenDiffusion,
                                                    graupel_number_sublimated);
    double hail_deposition    = deposit_on_category(cell[9],
                                                    hail_number_state,
                                                    parameters.alpha_h_,
                                                    parameters.c_h_,
                                                    parameters.d_h_,
                                                    parameters.a_h_,
                                                    parameters.b_h_,
                                                    1.0,
                                                    PublishedFrozenDiffusion,
                                                    hail_number_sublimated);

    // Published stochastic-collection equations for cloud and
    // pristine ice intercepted by snow, graupel, and hail.
    // All N values below are volumetric, matching the derivation.
    const double     nc_volume          = density * cloud_number;
    const double     ni_mixing          = std::max(0.0, ice_number_state);
    const double     ni_volume          = density * ni_mixing;
    constexpr double alpha_cloud        = 1.0;
    constexpr double mu_cloud           = 3.0;
    const double     gamma_cloud_number = std::tgamma(1.0 + alpha_cloud);
    const double     cloud_inverse_lambda =
        cloud > 1.0e-14 && nc_volume > 0.0
            ? std::cbrt(density * cloud / nc_volume /
                        (parameters.c_r_ * std::tgamma(1.0 + alpha_cloud + 3.0 / mu_cloud) / gamma_cloud_number))
            : 0.0;
    const GammaDistribution ice_psd = diagnose_distribution(
        density, cell[6], ice_number_state, parameters.alpha_i_, parameters.c_i_, parameters.d_i_);
    const double ice_mass_speed = ice_psd.inverse_lambda > 0.0
                                      ? air.density_factor * parameters.a_i_ *
                                            std::pow(ice_psd.inverse_lambda, parameters.b_i_) *
                                            std::tgamma(1.0 + parameters.alpha_i_ + parameters.d_i_ + parameters.b_i_) /
                                            std::tgamma(1.0 + parameters.alpha_i_ + parameters.d_i_)
                                      : 0.0;
    const double dynamic_viscosity = air.kinematic_viscosity * density;

    const auto collect_cloud = [&](double  collector_q,
                                   double  collector_n,
                                   double  alpha,
                                   double  c_mass,
                                   double  d_mass,
                                   double  fall_a,
                                   double  fall_b,
                                   int     category,
                                   double& mass_increment,
                                   double& number_increment)
    {
        if (!(cloud_inverse_lambda > 0.0) || !(collector_q > 1.0e-14) || !(collector_n > 0.0))
            return;
        const GammaDistribution collector =
            diagnose_distribution(density, collector_q, collector_n, alpha, c_mass, d_mass);
        if (!(collector.inverse_lambda > 0.0))
            return;
        const double collector_speed = air.density_factor * fall_a * std::pow(collector.inverse_lambda, fall_b) *
                                       std::tgamma(1.0 + alpha + d_mass + fall_b) / std::tgamma(1.0 + alpha + d_mass);
        double       efficiency      = 0.0;
        if (category == 0)
        {
            efficiency = std::min(collector.mean_mass_diameter, 1.0e-3) * 1.0e3;
            efficiency = std::min(std::cbrt(density * cloud / (parameters.c_r_ * nc_volume)), 30.0e-6) * 3.333e4 *
                         std::sqrt(efficiency);
        }
        else if (category == 1)
        {
            if (!(collector.mean_mass_diameter > 0.0))
                return;
            const double cloud_diameter = std::cbrt(density * cloud / (parameters.c_r_ * nc_volume));
            double       stokes = microphysics_constants::rho_w * collector_speed * cloud_diameter * cloud_diameter /
                                  (9.0 * dynamic_viscosity * collector.mean_mass_diameter);
            stokes              = std::clamp(stokes, 1.5, 10.0);
            efficiency          = 0.55 * std::log10(2.51 * stokes);
        }
        else
        {
            const double cloud_diameter = std::cbrt(density * cloud / (parameters.c_r_ * nc_volume));
            efficiency = std::exp(-8.68e-7 * std::pow(cloud_diameter, -1.6) * collector.mean_mass_diameter);
        }
        efficiency = std::clamp(efficiency, 0.0, 1.0);

        const double ilc    = cloud_inverse_lambda;
        const double ilx    = collector.inverse_lambda;
        const double gc11   = std::tgamma(1.0 / mu_cloud + 1.0 + alpha_cloud);
        const double gc12   = std::tgamma(2.0 / mu_cloud + 1.0 + alpha_cloud);
        const double gc13   = std::tgamma(3.0 / mu_cloud + 1.0 + alpha_cloud);
        const double gc14   = std::tgamma(4.0 / mu_cloud + 1.0 + alpha_cloud);
        const double gc15   = std::tgamma(5.0 / mu_cloud + 1.0 + alpha_cloud);
        const double gx1    = std::tgamma(1.0 + alpha);
        const double gx_b1  = std::tgamma(1.0 + alpha + fall_b);
        const double gx_b2  = std::tgamma(2.0 + alpha + fall_b);
        const double gx_b3  = std::tgamma(3.0 + alpha + fall_b);
        const double common = dt * air.density_factor * fall_a * (pi / 4.0) * efficiency * nc_volume *
                              (density * collector_n) / (gamma_cloud_number * gx1);
        mass_increment   = common * parameters.c_r_ / density *
                           (gc13 * gx_b3 * std::pow(ilc, 3.0) * std::pow(ilx, fall_b + 2.0) +
                            2.0 * gc14 * gx_b2 * std::pow(ilc, 4.0) * std::pow(ilx, fall_b + 1.0) +
                            gc15 * gx_b1 * std::pow(ilc, 5.0) * std::pow(ilx, fall_b));
        number_increment = common * (gamma_cloud_number * gx_b3 * std::pow(ilx, fall_b + 2.0) +
                                     2.0 * gc11 * gx_b2 * ilc * std::pow(ilx, fall_b + 1.0) +
                                     gc12 * gx_b1 * ilc * ilc * std::pow(ilx, fall_b));
        if (category == 0)
        {
            // Brandes non-spherical snow projected-area
            // correction used by the operational configuration.
            mass_increment *= 0.6366;
            number_increment *= 0.6366;
        }
        snow_riming.raw_cloud_mass_per_kg[category] = mass_increment;
        snow_riming.raw_cloud_number_per_kg[category] = number_increment / density;
        mass_increment   = std::clamp(mass_increment, 0.0, cloud);
        number_increment = std::clamp(number_increment / density, 0.0, cloud_number);
        snow_riming.pre_admission_cloud_mass_per_kg[category] = mass_increment;
        snow_riming.pre_admission_cloud_number_per_kg[category] = number_increment;
    };

    collect_cloud(cell[7],
                  snow_number_state,
                  parameters.alpha_s_,
                  parameters.c_s_,
                  parameters.d_s_,
                  parameters.a_s_,
                  parameters.b_s_,
                  0,
                  cloud_to_snow,
                  cloud_number_to_snow);
    collect_cloud(cell[8],
                  graupel_number_state,
                  parameters.alpha_g_,
                  parameters.c_g_,
                  parameters.d_g_,
                  parameters.a_g_,
                  parameters.b_g_,
                  1,
                  cloud_to_graupel,
                  cloud_number_to_graupel);
    collect_cloud(cell[9],
                  hail_number_state,
                  parameters.alpha_h_,
                  parameters.c_h_,
                  parameters.d_h_,
                  parameters.a_h_,
                  parameters.b_h_,
                  2,
                  cloud_to_hail,
                  cloud_number_to_hail);

    // Admission fixes the liquid journal before deriving its dependent heating and conversions.
    if (cloud_to_snow > 0.0)
    {
        const double cloud_sink = cloud_to_ice + cloud_to_snow + cloud_to_graupel + cloud_to_hail;
        const double cloud_number_sink = nc_to_ice + cloud_number_to_snow +
            cloud_number_to_graupel + cloud_number_to_hail;
        double cloud_scale = std::min(admission_ratio(cloud, cloud_sink),
            admission_ratio(cloud_number, cloud_number_sink));
        const std::array<double, 4> original_mass{cloud_to_ice, cloud_to_snow, cloud_to_graupel, cloud_to_hail};
        const std::array<double, 4> original_number{nc_to_ice, cloud_number_to_snow,
            cloud_number_to_graupel, cloud_number_to_hail};
        if (cloud_scale < 1.0)
        {
            cloud_scale = std::floor(cloud_scale * cold_snow_riming_cloud_factor_scale) *
                cold_snow_riming_cloud_factor_quantum;
            snow_riming.cloud_factor_quantized = true;
        }
        const auto allocate_cloud = [&]
        {
            cloud_to_ice = original_mass[0] * cloud_scale;
            nc_to_ice = original_number[0] * cloud_scale;
            cloud_to_snow = original_mass[1] * cloud_scale;
            cloud_number_to_snow = original_number[1] * cloud_scale;
            cloud_to_graupel = original_mass[2] * cloud_scale;
            cloud_number_to_graupel = original_number[2] * cloud_scale;
            cloud_to_hail = original_mass[3] * cloud_scale;
            cloud_number_to_hail = original_number[3] * cloud_scale;
        };
        const auto cloud_closed = [&]
        {
            const std::array<double, 4> mass{cloud_to_ice, cloud_to_snow, cloud_to_graupel, cloud_to_hail};
            const std::array<double, 4> number{nc_to_ice, cloud_number_to_snow,
                cloud_number_to_graupel, cloud_number_to_hail};
            for (std::size_t channel = 0; channel < mass.size(); ++channel)
            {
                if (!std::isfinite(mass[channel]) || mass[channel] < 0.0 ||
                    !std::isfinite(number[channel]) || number[channel] < 0.0)
                {
                    throw ColdSnowRimingAdmissionError(ColdSnowRimingAdmissionFailure::NonfiniteBudget,
                        "cold snow riming cloud pre-admission has a nonfinite or negative transfer");
                }
            }
            return cloud_to_ice + cloud_to_snow + cloud_to_graupel + cloud_to_hail <= cloud &&
                nc_to_ice + cloud_number_to_snow + cloud_number_to_graupel + cloud_number_to_hail <= cloud_number;
        };
        allocate_cloud();
        if (!cloud_closed() && snow_riming.cloud_factor_quantized)
        {
            cloud_scale -= cold_snow_riming_cloud_factor_quantum;
            snow_riming.cloud_factor_backoffs = 1;
            allocate_cloud();
        }
        if (!cloud_closed())
        {
            throw ColdSnowRimingAdmissionError(ColdSnowRimingAdmissionFailure::BudgetCap,
                "cold snow riming cloud pre-admission cannot close its exact paired donor budgets");
        }
        snow_riming.cloud_admission_factor = cloud_scale;
        snow_riming.active = cloud_to_snow > 0.0;
    }
    if (snow_riming.active)
    {
        snow_riming.riming_snow_mass_increment_per_kg = -inverse_diffusion_resistance *
            microphysics_constants::L_s * microphysics_constants::L_f /
            (air.thermal_conductivity * microphysics_constants::R_v * T * T) * cloud_to_snow;
        snow_riming.raw_heated_snow_mass_increment_per_kg =
            snow_riming.raw_ambient_snow_mass_increment_per_kg +
            snow_riming.riming_snow_mass_increment_per_kg;
        const double saturation_response =
            1.0 + microphysics_constants::L_s * microphysics_constants::L_s * qsi /
                      (microphysics_constants::cp * microphysics_constants::R_v * (T - 7.66) * (T - 7.66));
        const double vapor_limit = (vapor - qsi) / saturation_response;
        snow_deposition = snow_riming.raw_heated_snow_mass_increment_per_kg < 0.0
            ? std::max(snow_riming.raw_heated_snow_mass_increment_per_kg, -static_cast<double>(cell[7]))
            : std::min(snow_riming.raw_heated_snow_mass_increment_per_kg, std::max(0.0, vapor_limit));
        snow_number_sublimated =
            snow_deposition < 0.0 && cell[7] > 0.0 ? snow_number_state / cell[7] * snow_deposition : 0.0;
    }

    const auto collect_pristine_ice = [&](double  collector_q,
                                          double  collector_n,
                                          double  alpha,
                                          double  c_mass,
                                          double  d_mass,
                                          double  fall_a,
                                          double  fall_b,
                                          double  efficiency,
                                          double& mass_increment,
                                          double& number_increment)
    {
        if (!(cell[6] > 1.0e-14) || !(ni_volume > 0.0) || !(collector_q > 1.0e-14) || !(collector_n > 0.0) ||
            !(ice_psd.inverse_lambda > 0.0))
            return;
        const GammaDistribution collector =
            diagnose_distribution(density, collector_q, collector_n, alpha, c_mass, d_mass);
        if (!(collector.inverse_lambda > 0.0))
            return;
        const double collector_speed = air.density_factor * fall_a * std::pow(collector.inverse_lambda, fall_b) *
                                       std::tgamma(1.0 + alpha + d_mass + fall_b) / std::tgamma(1.0 + alpha + d_mass);
        const double relative_speed =
            std::sqrt(std::pow(collector_speed - ice_mass_speed, 2.0) + 0.04 * collector_speed * ice_mass_speed);
        const double ili = ice_psd.inverse_lambda;
        const double ilx = collector.inverse_lambda;
        const double common_number =
            dt * (pi / 4.0) * efficiency * ni_volume * (density * collector_n) * relative_speed;
        mass_increment =
            dt * parameters.c_i_ / density * pi * 6.0 * efficiency * ni_volume * (density * collector_n) *
            relative_speed *
            (0.5 * ilx * ilx * std::pow(ili, 3.0) + 2.0 * ilx * std::pow(ili, 4.0) + 5.0 * std::pow(ili, 5.0));
        const double number_collision = common_number * (std::tgamma(3.0 + parameters.alpha_i_) * ilx * ilx +
                                                         2.0 * std::tgamma(2.0 + parameters.alpha_i_) * ili * ilx +
                                                         std::tgamma(1.0 + parameters.alpha_i_) * ili * ili);
        mass_increment                = std::clamp(mass_increment, 0.0, static_cast<double>(cell[6]));
        number_increment =
            std::clamp(std::min(mass_increment * ni_mixing / cell[6], number_collision / density), 0.0, ni_mixing);
    };

    const double ice_snow_efficiency    = std::min(1.0, 0.05 * std::exp(0.1 * temperature_c));
    const double ice_graupel_efficiency = std::min(1.0, 0.01 * std::exp(0.1 * temperature_c));
    collect_pristine_ice(cell[7],
                         snow_number_state,
                         parameters.alpha_s_,
                         parameters.c_s_,
                         parameters.d_s_,
                         parameters.a_s_,
                         parameters.b_s_,
                         ice_snow_efficiency,
                         ice_collected_by_snow,
                         ice_number_collected_by_snow);
    collect_pristine_ice(cell[8],
                         graupel_number_state,
                         parameters.alpha_g_,
                         parameters.c_g_,
                         parameters.d_g_,
                         parameters.a_g_,
                         parameters.b_g_,
                         ice_graupel_efficiency,
                         ice_collected_by_graupel,
                         ice_number_collected_by_graupel);
    collect_pristine_ice(cell[9],
                         hail_number_state,
                         parameters.alpha_h_,
                         parameters.c_h_,
                         parameters.d_h_,
                         parameters.a_h_,
                         parameters.b_h_,
                         ice_graupel_efficiency,
                         ice_collected_by_hail,
                         ice_number_collected_by_hail);

    const double ns_volume = density * snow_number_state;
    if (cell[7] > 1.0e-14 && ns_volume > 0.0)
    {
        snow_self_collection =
            std::min(0.5 * ns_volume,
                     dt * 0.93952 * ice_snow_efficiency * std::pow(density * cell[7], (2.0 + parameters.b_s_) / 3.0) *
                         std::pow(ns_volume, (4.0 - parameters.b_s_) / 3.0)) /
            density;
    }

    // Hallett-Mossop rime splintering uses the cloud-riming
    // masses just diagnosed and conserves the tiny embryo mass.
    double splinters_per_kg = 0.0;
    if (temperature_c >= -8.0 && temperature_c <= -5.0)
        splinters_per_kg = 3.5e8 * (temperature_c + 8.0) / 3.0;
    else if (temperature_c > -5.0 && temperature_c < -3.0)
        splinters_per_kg = 3.5e8 * (-3.0 - temperature_c) * 0.5;
    splinter_number_from_snow    = splinters_per_kg * cloud_to_snow;
    splinter_number_from_graupel = splinters_per_kg * cloud_to_graupel;
    splinter_mass_from_snow      = std::min(cloud_to_snow, embryo_ice_mass * splinter_number_from_snow);
    splinter_mass_from_graupel   = std::min(cloud_to_graupel, embryo_ice_mass * splinter_number_from_graupel);

    // Bigg probabilistic rain freezing. Existing hail is always
    // physically treated; disabling hail production routes newly
    // frozen rain to graupel.
    const double rain               = cell[5];
    const double rain_number        = rain_number_state;
    const double rain_number_volume = density * rain_number;
    if (temperature_c < -10.0 && rain > 1.0e-14 && rain_number_volume > 0.0)
    {
        const double frozen_number_volume = std::min(rain_number_volume,
                                                     -dt * 100.0 * (std::exp(0.66 * temperature_c) - 1.0) * density *
                                                         rain / microphysics_constants::rho_w);
        nr_to_frozen                      = frozen_number_volume / density;
        rain_to_frozen                    = std::min(rain, nr_to_frozen * rain / rain_number);
    }

    // Three-component freezing: rain colliding with pristine
    // ice, snow, or graupel. The two collected masses and two
    // collided number populations are integrated independently.
    // The combined particle is routed by its bulk density; a
    // sufficiently large dense rain component is required before
    // newly produced hail is admitted.
    const GammaDistribution rain_psd =
        diagnose_distribution(density, rain, rain_number, parameters.alpha_r_, parameters.c_r_, parameters.d_r_);
    const GammaDistribution snow_psd = diagnose_distribution(
        density, cell[7], snow_number_state, parameters.alpha_s_, parameters.c_s_, parameters.d_s_);
    const GammaDistribution graupel_psd = diagnose_distribution(
        density, cell[8], graupel_number_state, parameters.alpha_g_, parameters.c_g_, parameters.d_g_);
    const GammaDistribution hail_psd = diagnose_distribution(
        density, cell[9], hail_number_state, parameters.alpha_h_, parameters.c_h_, parameters.d_h_);
    const double rain_speed =
        mass_weighted_fall_speed(rain_psd, parameters.alpha_r_, parameters.d_r_, parameters.a_r_, parameters.b_r_, air);
    const double snow_speed =
        mass_weighted_fall_speed(snow_psd, parameters.alpha_s_, parameters.d_s_, parameters.a_s_, parameters.b_s_, air);
    const double graupel_speed = mass_weighted_fall_speed(
        graupel_psd, parameters.alpha_g_, parameters.d_g_, parameters.a_g_, parameters.b_g_, air);
    const double hail_speed =
        mass_weighted_fall_speed(hail_psd, parameters.alpha_h_, parameters.d_h_, parameters.a_h_, parameters.b_h_, air);
    const auto relative_speed = [](double first, double second)
    { return std::sqrt((first - second) * (first - second) + 0.04 * first * second); };
    enum class FrozenDestination
    {
        Snow,
        Graupel,
        Hail
    };
    const auto route_collision = [&](double                   first_mass,
                                     double                   second_mass,
                                     double                   first_number,
                                     double                   second_number,
                                     const GammaDistribution& frozen_psd,
                                     double                   frozen_bulk_density,
                                     FrozenDestination        destination)
    {
        const double total_mass = first_mass + second_mass;
        if (!(total_mass > 0.0))
            return;
        const double maximum_diameter = std::max(frozen_psd.mean_mass_diameter, rain_psd.mean_mass_diameter);
        if (!(maximum_diameter > 0.0))
            return;
        double destination_density = 400.0;
        if (destination == FrozenDestination::Snow)
            destination_density = frozen_bulk_density;
        else if (destination == FrozenDestination::Hail)
            destination_density = 900.0;
        const double combined_number =
            total_mass / (destination_density * (pi / 6.0) * maximum_diameter * maximum_diameter * maximum_diameter);
        const double admitted_number = std::min(combined_number, std::min(first_number, second_number));
        if (destination == FrozenDestination::Snow)
        {
            collision_mass_to_snow += total_mass;
            collision_number_to_snow += admitted_number;
        }
        else if (destination == FrozenDestination::Graupel)
        {
            collision_mass_to_graupel += total_mass;
            collision_number_to_graupel += admitted_number;
        }
        else
        {
            collision_mass_to_hail += total_mass;
            collision_number_to_hail += admitted_number;
        }
    };
    const auto density_destination = [&](const GammaDistribution& frozen_psd, double frozen_density, bool allow_snow)
    {
        const double maximum_diameter = std::max(frozen_psd.mean_mass_diameter, rain_psd.mean_mass_diameter);
        if (!(maximum_diameter > 0.0))
            return FrozenDestination::Graupel;
        const double combined_density = (frozen_density * std::pow(frozen_psd.mean_mass_diameter, 3.0) +
                                         microphysics_constants::rho_w * std::pow(rain_psd.mean_mass_diameter, 3.0)) /
                                        std::pow(maximum_diameter, 3.0);
        if (allow_snow && combined_density <= 0.5 * (frozen_density + 400.0))
        {
            return FrozenDestination::Snow;
        }
        if (parameters.hail_processes && rain_psd.mean_mass_diameter > 2.5e-3 && combined_density >= 650.0)
        {
            return FrozenDestination::Hail;
        }
        return FrozenDestination::Graupel;
    };
    bool              ice_rain_collision          = false;
    bool              snow_rain_collision         = false;
    bool              graupel_rain_collision      = false;
    FrozenDestination ice_rain_destination        = FrozenDestination::Graupel;
    FrozenDestination snow_rain_destination       = FrozenDestination::Graupel;
    FrozenDestination graupel_rain_destination    = FrozenDestination::Graupel;
    double            snow_collision_bulk_density = 100.0;
    bool              rain_graupel_stays_graupel  = false;

    if (rain_psd.intercept > 0.0 && ice_psd.intercept > 0.0)
    {
        const PairCollision collision = collide_gamma_populations(density,
                                                                  dt,
                                                                  1.0,
                                                                  relative_speed(ice_mass_speed, rain_speed),
                                                                  cell[6],
                                                                  ice_number_state,
                                                                  ice_psd,
                                                                  parameters.alpha_i_,
                                                                  parameters.c_i_,
                                                                  parameters.d_i_,
                                                                  rain,
                                                                  rain_number,
                                                                  rain_psd,
                                                                  parameters.alpha_r_,
                                                                  parameters.c_r_,
                                                                  parameters.d_r_);
        ice_collected_by_rain         = collision.first_mass;
        rain_collected_by_ice         = collision.second_mass;
        ice_number_collected_by_rain  = collision.first_number;
        rain_number_collected_by_ice  = collision.second_number;
        ice_rain_collision            = true;
        ice_rain_destination          = density_destination(ice_psd, 500.0, false);
    }

    if (temperature_c < -5.0 && rain_psd.intercept > 0.0 && snow_psd.intercept > 0.0)
    {
        const PairCollision collision = collide_gamma_populations(density,
                                                                  dt,
                                                                  1.0,
                                                                  relative_speed(snow_speed, rain_speed),
                                                                  cell[7],
                                                                  snow_number_state,
                                                                  snow_psd,
                                                                  parameters.alpha_s_,
                                                                  parameters.c_s_,
                                                                  parameters.d_s_,
                                                                  rain,
                                                                  rain_number,
                                                                  rain_psd,
                                                                  parameters.alpha_r_,
                                                                  parameters.c_r_,
                                                                  parameters.d_r_);
        snow_collected_by_rain        = collision.first_mass;
        rain_collected_by_snow        = collision.second_mass;
        snow_number_collected_by_rain = collision.first_number;
        rain_number_collected_by_snow = collision.second_number;
        snow_collision_bulk_density   = std::min(
            500.0, parameters.c_s_ / (pi / 6.0) * std::pow(snow_psd.mean_mass_diameter, parameters.d_s_ - 3.0));
        snow_rain_collision   = true;
        snow_rain_destination = density_destination(snow_psd, snow_collision_bulk_density, true);
    }

    if (temperature_c < -5.0 && rain_psd.intercept > 0.0 && graupel_psd.intercept > 0.0)
    {
        double stokes                    = microphysics_constants::rho_w * std::abs(graupel_speed - rain_speed) *
                                           rain_psd.mean_mass_diameter * rain_psd.mean_mass_diameter /
                                           (9.0 * dynamic_viscosity * graupel_psd.mean_mass_diameter);
        stokes                           = std::clamp(stokes, 1.5, 10.0);
        const double        efficiency   = std::clamp(0.55 * std::log10(2.51 * stokes), 0.0, 1.0);
        const PairCollision collision    = collide_gamma_populations(density,
                                                                     dt,
                                                                     efficiency,
                                                                     relative_speed(graupel_speed, rain_speed),
                                                                     cell[8],
                                                                     graupel_number_state,
                                                                     graupel_psd,
                                                                     parameters.alpha_g_,
                                                                     parameters.c_g_,
                                                                     parameters.d_g_,
                                                                     rain,
                                                                     rain_number,
                                                                     rain_psd,
                                                                     parameters.alpha_r_,
                                                                     parameters.c_r_,
                                                                     parameters.d_r_);
        graupel_collected_by_rain        = collision.first_mass;
        rain_collected_by_graupel        = collision.second_mass;
        graupel_number_collected_by_rain = collision.first_number;
        rain_number_collected_by_graupel = collision.second_number;
        graupel_rain_collision           = true;
        graupel_rain_destination         = density_destination(graupel_psd, 400.0, false);
        rain_graupel_stays_graupel       = graupel_rain_destination == FrozenDestination::Graupel;
    }

    // Hail collects rain and snow in addition to the cloud and
    // pristine-ice terms above. When the thermodynamic wet-growth
    // capacity is exceeded, excess collected rain is shed as a
    // one-millimetre drop population instead of disappearing.
    if (hail_psd.intercept > 0.0)
    {
        PairCollision rain_hail;
        if (rain_psd.intercept > 0.0)
        {
            rain_hail                     = collide_gamma_populations(density,
                                                                      dt,
                                                                      1.0,
                                                                      relative_speed(rain_speed, hail_speed),
                                                                      rain,
                                                                      rain_number,
                                                                      rain_psd,
                                                                      parameters.alpha_r_,
                                                                      parameters.c_r_,
                                                                      parameters.d_r_,
                                                                      cell[9],
                                                                      hail_number_state,
                                                                      hail_psd,
                                                                      parameters.alpha_h_,
                                                                      parameters.c_h_,
                                                                      parameters.d_h_);
            rain_collected_by_hail        = rain_hail.first_mass;
            rain_number_collected_by_hail = rain_hail.first_number;
        }

        PairCollision snow_hail;
        if (snow_psd.intercept > 0.0)
        {
            snow_hail                     = collide_gamma_populations(density,
                                                                      dt,
                                                                      ice_graupel_efficiency,
                                                                      relative_speed(snow_speed, hail_speed),
                                                                      cell[7],
                                                                      snow_number_state,
                                                                      snow_psd,
                                                                      parameters.alpha_s_,
                                                                      parameters.c_s_,
                                                                      parameters.d_s_,
                                                                      cell[9],
                                                                      hail_number_state,
                                                                      hail_psd,
                                                                      parameters.alpha_h_,
                                                                      parameters.c_h_,
                                                                      parameters.d_h_);
            snow_collected_by_hail        = snow_hail.first_mass;
            snow_number_collected_by_hail = snow_hail.first_number;
        }

        const double hail_ventilation =
            ventilation_integral(hail_psd, parameters.alpha_h_, parameters.a_h_, parameters.b_h_, air);
        const double water_saturation_at_freezing =
            thermodynamics::saturation_mixing_ratio_water(microphysics_constants::T0, pressure);
        constexpr double liquid_heat_capacity   = 4218.0;
        constexpr double ice_heat_capacity      = 2115.3;
        const double     wet_growth_denominator = microphysics_constants::L_f + liquid_heat_capacity * temperature_c;
        double           wet_growth_capacity    = 0.0;
        if (wet_growth_denominator > 0.0)
        {
            wet_growth_capacity =
                std::max(0.0,
                         dt * 2.0 * pi *
                                 (density * microphysics_constants::L_v * air.vapor_diffusivity *
                                      (water_saturation_at_freezing - vapor) -
                                  air.thermal_conductivity * temperature_c) *
                                 hail_psd.intercept / density / wet_growth_denominator * hail_ventilation +
                             (ice_collected_by_hail / std::max(ice_graupel_efficiency, 1.0e-12) +
                              snow_collected_by_hail / std::max(ice_graupel_efficiency, 1.0e-12)) *
                                 (1.0 - ice_heat_capacity * temperature_c / wet_growth_denominator));
        }

        const double dry_collection =
            cloud_to_hail + rain_collected_by_hail + ice_collected_by_hail + snow_collected_by_hail;
        if (temperature_c > -40.0 && wet_growth_capacity < dry_collection)
        {
            if (ice_psd.intercept > 0.0)
            {
                const PairCollision wet_ice_hail = collide_gamma_populations(density,
                                                                             dt,
                                                                             1.0,
                                                                             relative_speed(ice_mass_speed, hail_speed),
                                                                             cell[6],
                                                                             ice_number_state,
                                                                             ice_psd,
                                                                             parameters.alpha_i_,
                                                                             parameters.c_i_,
                                                                             parameters.d_i_,
                                                                             cell[9],
                                                                             hail_number_state,
                                                                             hail_psd,
                                                                             parameters.alpha_h_,
                                                                             parameters.c_h_,
                                                                             parameters.d_h_);
                ice_collected_by_hail            = wet_ice_hail.first_mass;
                ice_number_collected_by_hail     = wet_ice_hail.first_number;
            }
            if (snow_psd.intercept > 0.0)
            {
                const PairCollision wet_snow_hail = collide_gamma_populations(density,
                                                                              dt,
                                                                              1.0,
                                                                              relative_speed(snow_speed, hail_speed),
                                                                              cell[7],
                                                                              snow_number_state,
                                                                              snow_psd,
                                                                              parameters.alpha_s_,
                                                                              parameters.c_s_,
                                                                              parameters.d_s_,
                                                                              cell[9],
                                                                              hail_number_state,
                                                                              hail_psd,
                                                                              parameters.alpha_h_,
                                                                              parameters.c_h_,
                                                                              parameters.d_h_);
                snow_collected_by_hail            = wet_snow_hail.first_mass;
                snow_number_collected_by_hail     = wet_snow_hail.first_number;
            }
            const double retained_rain =
                std::clamp(wet_growth_capacity - cloud_to_hail - ice_collected_by_hail - snow_collected_by_hail,
                           0.0,
                           rain_collected_by_hail);
            shed_rain_from_hail                 = rain_collected_by_hail - retained_rain;
            rain_collected_by_hail              = retained_rain;
            constexpr double shed_drop_diameter = 1.0e-3;
            shed_rain_number =
                shed_rain_from_hail / (parameters.c_r_ * shed_drop_diameter * shed_drop_diameter * shed_drop_diameter);
        }
    }

    const auto admit_vapor = [&]
    {
        const double vapor_source = vapor + std::max(-ice_deposition, 0.0) + std::max(-snow_deposition, 0.0) +
            std::max(-graupel_deposition, 0.0) + std::max(-hail_deposition, 0.0);
        const double vapor_sink = vapor_to_ice + std::max(ice_deposition, 0.0) + std::max(snow_deposition, 0.0) +
            std::max(graupel_deposition, 0.0) + std::max(hail_deposition, 0.0);
        const double scale = admission_ratio(vapor_source, vapor_sink);
        if (scale < 1.0)
        {
            vapor_to_ice *= scale;
            new_ice_number *= scale;
            if (ice_deposition > 0.0)
                ice_deposition *= scale;
            if (snow_deposition > 0.0)
                snow_deposition *= scale;
            if (graupel_deposition > 0.0)
                graupel_deposition *= scale;
            if (hail_deposition > 0.0)
                hail_deposition *= scale;
        }
    };
    if (snow_riming.active)
    {
        admit_vapor();
    }

    // Riming-driven snow conversion. Collected cloud mass coats
    // existing snow particles; it does not create a second
    // particle population, so transferred number follows the
    // fraction of the post-riming snow mass that changes category.
    if (cell[7] > 1.0e-14 && cloud_to_snow > 0.0 && cloud_to_snow > snow_deposition)
    {
        snow_riming.snow_conversion_predicate = true;
        snow_to_graupel        = std::min(static_cast<double>(cell[7]) + cloud_to_snow,
                                          4.0 * cloud_to_snow / 3.0);
        snow_number_to_graupel = std::min(
            snow_number_state, snow_number_state * snow_to_graupel / (static_cast<double>(cell[7]) + cloud_to_snow));
    }

    // Schumann-Ludlam wet-growth size limit. Diagnose the
    // graupel tail after liquid riming and transfer only the
    // original graupel reservoir represented by that tail.
    if (parameters.hail_processes && cell[8] > 1.0e-14 && graupel_number_state > 0.0 && cloud_to_graupel > 0.0)
    {
        const double liquid_loading = std::max(1.0, 1.1e4 * density * (static_cast<double>(cell[4]) + rain) + 1.0);
        const double size_limit =
            std::clamp(0.02 * (std::exp(std::min(20.0, -temperature_c / liquid_loading)) - 1.0), 1.0e-4, 1.0);
        const double            post_riming_graupel = static_cast<double>(cell[8]) + cloud_to_graupel +
                                                      (rain_graupel_stays_graupel ? rain_collected_by_graupel : 0.0);
        const GammaDistribution post_riming_psd     = diagnose_distribution(
            density, post_riming_graupel, graupel_number_state, parameters.alpha_g_, parameters.c_g_, parameters.d_g_);
        if (post_riming_psd.intercept > 0.0 && std::abs(parameters.alpha_g_) < 1.0e-12 &&
            std::abs(parameters.d_g_ - 3.0) < 1.0e-12)
        {
            const double inverse_lambda     = post_riming_psd.inverse_lambda;
            const double exponential_tail   = std::exp(-size_limit / inverse_lambda);
            const double tail_number_volume = post_riming_psd.intercept * inverse_lambda * exponential_tail;
            if (tail_number_volume > 1.0)
            {
                const double d = size_limit;
                const double tail_mass =
                    parameters.c_g_ * post_riming_psd.intercept * exponential_tail *
                    (d * d * d * inverse_lambda + 3.0 * d * d * inverse_lambda * inverse_lambda +
                     6.0 * d * std::pow(inverse_lambda, 3.0) + 6.0 * std::pow(inverse_lambda, 4.0)) /
                    density;
                graupel_to_hail          = std::min(static_cast<double>(cell[8]), tail_mass);
                graupel_number_to_hail   = std::min(graupel_number_state, tail_number_volume / density);
                hail_number_from_graupel = graupel_number_to_hail;
            }
        }
    }

    // Hail smaller than the declared one-millimetre category
    // boundary is graupel, independent of whether new hail
    // production is enabled.
    // The joint adapter classifies the admitted post-diffusion population.
    if constexpr (!JointFrozenDiffusion)
    {
        if (hail_psd.mean_mass_diameter > 0.0 && hail_psd.mean_mass_diameter < 1.0e-3)
        {
            hail_to_graupel        = cell[9];
            hail_number_to_graupel = hail_number_state;
        }
    }

    // Convert the large-size tail of the pristine-ice
    // distribution to snow. For alpha_i=0 and d_i=3 these are
    // the analytical upper-tail integrals used by the scheme.
    const double provisional_ice =
        std::max(0.0,
                 static_cast<double>(cell[6]) + cloud_to_ice + vapor_to_ice + ice_deposition - ice_collected_by_snow -
                     ice_collected_by_graupel - ice_collected_by_hail - ice_collected_by_rain +
                     splinter_mass_from_snow + splinter_mass_from_graupel);
    const double provisional_ice_number =
        std::max(0.0,
                 ice_number_state + nc_to_ice + new_ice_number / density + ice_number_sublimated -
                     ice_number_collected_by_snow - ice_number_collected_by_graupel - ice_number_collected_by_hail -
                     ice_number_collected_by_rain + splinter_number_from_snow + splinter_number_from_graupel);
    const GammaDistribution ice_distribution = diagnose_distribution(
        density, provisional_ice, provisional_ice_number, parameters.alpha_i_, parameters.c_i_, parameters.d_i_);
    if (ice_distribution.intercept > 0.0)
    {
        const double inverse_lambda     = ice_distribution.inverse_lambda;
        const double exponential_tail   = std::exp(-snow_embryo_diameter / inverse_lambda);
        const double number_tail_volume = ice_distribution.intercept * inverse_lambda * exponential_tail;
        const double d                  = snow_embryo_diameter;
        const double mass_tail_volume = parameters.c_i_ * ice_distribution.intercept * exponential_tail *
                                        (d * d * d * inverse_lambda + 3.0 * d * d * inverse_lambda * inverse_lambda +
                                         6.0 * d * std::pow(inverse_lambda, 3.0) + 6.0 * std::pow(inverse_lambda, 4.0));
        ice_number_to_snow            = std::min(provisional_ice_number, number_tail_volume / density);
        ice_to_snow                   = std::min(provisional_ice, mass_tail_volume / density);
    }

    // Non-riming overdepletion retains its two-pass donor-ordered adjustment.
    // The heated-snow branch closes the fixed-rate admission graph instead. A common factor is not
    // equivalent: exhausting cloud water must not suppress an
    // independent vapor-deposition or hail-conversion pathway.
    // Each donor pass scales only its competing transfers and the
    // number changes carried by those transfers. Repeating the
    // vapor/cloud/rain/ice/snow/graupel/hail sequence propagates
    // any scaled source into downstream donor availability.
    const auto routed_collision_mass = [&](FrozenDestination destination)
    {
        double result = 0.0;
        if (ice_rain_collision && ice_rain_destination == destination)
            result += ice_collected_by_rain + rain_collected_by_ice;
        if (snow_rain_collision && snow_rain_destination == destination)
            result += snow_collected_by_rain + rain_collected_by_snow;
        if (graupel_rain_collision && graupel_rain_destination == destination)
            result += graupel_collected_by_rain + rain_collected_by_graupel;
        return result;
    };
    const auto collision_number = [&](double first_mass, double second_mass,
                                      double first_number, double second_number,
                                      const GammaDistribution& distribution,
                                      double bulk_density, FrozenDestination destination)
    {
        const double total = first_mass + second_mass;
        const double diameter = std::max(distribution.mean_mass_diameter, rain_psd.mean_mass_diameter);
        if (!(total > 0.0) || !(diameter > 0.0))
            return 0.0;
        const double destination_density = destination == FrozenDestination::Snow ? bulk_density
            : destination == FrozenDestination::Hail ? 900.0 : 400.0;
        const double combined = total /
            (destination_density * (pi / 6.0) * diameter * diameter * diameter);
        return std::min(combined, std::min(first_number, second_number));
    };
    const auto routed_snow_number = [&]
    {
        double result = 0.0;
        if (ice_rain_collision && ice_rain_destination == FrozenDestination::Snow)
            result += collision_number(ice_collected_by_rain, rain_collected_by_ice,
                ice_number_collected_by_rain, rain_number_collected_by_ice,
                ice_psd, 500.0, ice_rain_destination);
        if (snow_rain_collision && snow_rain_destination == FrozenDestination::Snow)
            result += collision_number(snow_collected_by_rain, rain_collected_by_snow,
                snow_number_collected_by_rain, rain_number_collected_by_snow,
                snow_psd, snow_collision_bulk_density, snow_rain_destination);
        if (graupel_rain_collision && graupel_rain_destination == FrozenDestination::Snow)
            result += collision_number(graupel_collected_by_rain, rain_collected_by_graupel,
                graupel_number_collected_by_rain, rain_number_collected_by_graupel,
                graupel_psd, 400.0, graupel_rain_destination);
        return result;
    };
    const auto snapshot_budgets = [&]
    {
        snow_riming.donor_mass_source_per_kg = {
            vapor + std::max(-ice_deposition, 0.0) + std::max(-snow_deposition, 0.0) +
                std::max(-graupel_deposition, 0.0) + std::max(-hail_deposition, 0.0),
            cloud,
            rain,
            static_cast<double>(cell[6]) + cloud_to_ice + vapor_to_ice + std::max(ice_deposition, 0.0) +
                splinter_mass_from_snow + splinter_mass_from_graupel,
            static_cast<double>(cell[7]) + std::max(snow_deposition, 0.0) + cloud_to_snow +
                ice_to_snow + ice_collected_by_snow + routed_collision_mass(FrozenDestination::Snow),
            static_cast<double>(cell[8]) + std::max(graupel_deposition, 0.0) + cloud_to_graupel +
                ice_collected_by_graupel + routed_collision_mass(FrozenDestination::Graupel) +
                snow_to_graupel + hail_to_graupel + (parameters.hail_processes ? 0.0 : rain_to_frozen),
            static_cast<double>(cell[9]) + std::max(hail_deposition, 0.0) + cloud_to_hail + ice_collected_by_hail +
                snow_collected_by_hail + rain_collected_by_hail + routed_collision_mass(FrozenDestination::Hail) +
                graupel_to_hail + (parameters.hail_processes ? rain_to_frozen : 0.0)};
        snow_riming.donor_mass_sink_per_kg = {
            vapor_to_ice + std::max(ice_deposition, 0.0) + std::max(snow_deposition, 0.0) +
                std::max(graupel_deposition, 0.0) + std::max(hail_deposition, 0.0),
            cloud_to_ice + cloud_to_snow + cloud_to_graupel + cloud_to_hail,
            rain_to_frozen + rain_collected_by_ice + rain_collected_by_snow + rain_collected_by_graupel + rain_collected_by_hail,
            std::max(-ice_deposition, 0.0) + ice_to_snow + ice_collected_by_snow +
                ice_collected_by_graupel + ice_collected_by_hail + ice_collected_by_rain,
            std::max(-snow_deposition, 0.0) + snow_collected_by_rain + snow_collected_by_hail +
                splinter_mass_from_snow + snow_to_graupel,
            std::max(-graupel_deposition, 0.0) + graupel_collected_by_rain + splinter_mass_from_graupel + graupel_to_hail,
            std::max(-hail_deposition, 0.0) + hail_to_graupel};
        snow_riming.snow_number_source_per_kg = snow_number_state + ice_number_to_snow + routed_snow_number();
        snow_riming.snow_number_sink_per_kg = std::max(-snow_number_sublimated, 0.0) +
            snow_number_collected_by_rain + snow_number_collected_by_hail + snow_self_collection + snow_number_to_graupel;
    };
    const auto budgets_closed = [&]
    {
        snapshot_budgets();
        bool closed = true;
        for (std::size_t donor = 0; donor < snow_riming.donor_mass_source_per_kg.size(); ++donor)
        {
            const double source = snow_riming.donor_mass_source_per_kg[donor];
            const double sink = snow_riming.donor_mass_sink_per_kg[donor];
            if (!std::isfinite(source) || source < 0.0 || !std::isfinite(sink) || sink < 0.0)
            {
                throw ColdSnowRimingAdmissionError(ColdSnowRimingAdmissionFailure::NonfiniteBudget,
                    "cold snow riming admission has a nonfinite or negative donor budget");
            }
            closed = closed && source >= sink;
        }
        if (!std::isfinite(snow_riming.snow_number_source_per_kg) || snow_riming.snow_number_source_per_kg < 0.0 ||
            !std::isfinite(snow_riming.snow_number_sink_per_kg) || snow_riming.snow_number_sink_per_kg < 0.0)
        {
            throw ColdSnowRimingAdmissionError(ColdSnowRimingAdmissionFailure::NonfiniteBudget,
                "cold snow riming admission has a nonfinite or negative number budget");
        }
        return closed && snow_riming.snow_number_source_per_kg >= snow_riming.snow_number_sink_per_kg;
    };
    const unsigned admission_limit = snow_riming.active ? snow_riming_max_passes : 2;
    bool admission_closed = false;
    for (unsigned admission_pass = 0; admission_pass < admission_limit; ++admission_pass)
    {
        const double vapor_source = vapor + std::max(-ice_deposition, 0.0) + std::max(-snow_deposition, 0.0) +
                                    std::max(-graupel_deposition, 0.0) + std::max(-hail_deposition, 0.0);
        const double vapor_sink   = vapor_to_ice + std::max(ice_deposition, 0.0) + std::max(snow_deposition, 0.0) +
                                    std::max(graupel_deposition, 0.0) + std::max(hail_deposition, 0.0);
        const double vapor_scale  = admission_ratio(vapor_source, vapor_sink);
        if (vapor_scale < 1.0)
        {
            vapor_to_ice *= vapor_scale;
            new_ice_number *= vapor_scale;
            if (ice_deposition > 0.0)
                ice_deposition *= vapor_scale;
            if (snow_deposition > 0.0)
                snow_deposition *= vapor_scale;
            if (graupel_deposition > 0.0)
                graupel_deposition *= vapor_scale;
            if (hail_deposition > 0.0)
                hail_deposition *= vapor_scale;
        }

        const double cloud_sink  = cloud_to_ice + cloud_to_snow + cloud_to_graupel + cloud_to_hail;
        const double cloud_scale = admission_ratio(cloud, cloud_sink);
        if (cloud_scale < 1.0 && !snow_riming.active)
        {
            snow_riming.cloud_admission_factor *= cloud_scale;
            cloud_to_ice *= cloud_scale;
            nc_to_ice *= cloud_scale;
            cloud_to_snow *= cloud_scale;
            cloud_number_to_snow *= cloud_scale;
            cloud_to_graupel *= cloud_scale;
            cloud_number_to_graupel *= cloud_scale;
            cloud_to_hail *= cloud_scale;
            cloud_number_to_hail *= cloud_scale;
            splinter_mass_from_snow *= cloud_scale;
            splinter_mass_from_graupel *= cloud_scale;
            splinter_number_from_snow *= cloud_scale;
            splinter_number_from_graupel *= cloud_scale;
        }

        const double rain_sink  = rain_to_frozen + rain_collected_by_ice + rain_collected_by_snow +
                                  rain_collected_by_graupel + rain_collected_by_hail;
        const double rain_scale = admission_ratio(rain, rain_sink);
        if (rain_scale < 1.0)
        {
            rain_to_frozen *= rain_scale;
            nr_to_frozen *= rain_scale;
            rain_collected_by_ice *= rain_scale;
            rain_number_collected_by_ice *= rain_scale;
            rain_collected_by_snow *= rain_scale;
            rain_number_collected_by_snow *= rain_scale;
            rain_collected_by_graupel *= rain_scale;
            rain_number_collected_by_graupel *= rain_scale;
            rain_collected_by_hail *= rain_scale;
            rain_number_collected_by_hail *= rain_scale;
            shed_rain_number *= rain_scale;
        }

        const double ice_source = static_cast<double>(cell[6]) + cloud_to_ice + vapor_to_ice +
                                  std::max(ice_deposition, 0.0) + splinter_mass_from_snow + splinter_mass_from_graupel;
        const double ice_sink   = std::max(-ice_deposition, 0.0) + ice_to_snow + ice_collected_by_snow +
                                  ice_collected_by_graupel + ice_collected_by_hail + ice_collected_by_rain;
        const double ice_scale  = admission_ratio(ice_source, ice_sink);
        if (ice_scale < 1.0)
        {
            if (ice_deposition < 0.0)
            {
                ice_deposition *= ice_scale;
                ice_number_sublimated *= ice_scale;
            }
            ice_to_snow *= ice_scale;
            ice_number_to_snow *= ice_scale;
            ice_collected_by_snow *= ice_scale;
            ice_number_collected_by_snow *= ice_scale;
            ice_collected_by_graupel *= ice_scale;
            ice_number_collected_by_graupel *= ice_scale;
            ice_collected_by_hail *= ice_scale;
            ice_number_collected_by_hail *= ice_scale;
            ice_collected_by_rain *= ice_scale;
            ice_number_collected_by_rain *= ice_scale;
        }

        const double snow_source = static_cast<double>(cell[7]) + std::max(snow_deposition, 0.0) + cloud_to_snow +
                                   ice_to_snow + ice_collected_by_snow + routed_collision_mass(FrozenDestination::Snow);
        const double snow_sink   = std::max(-snow_deposition, 0.0) + snow_collected_by_rain + snow_collected_by_hail +
                                   splinter_mass_from_snow + snow_to_graupel;
        double snow_scale = admission_ratio(snow_source, snow_sink);
        if (snow_riming.active)
        {
            const double number_source = snow_number_state + ice_number_to_snow + routed_snow_number();
            const double number_sink = std::max(-snow_number_sublimated, 0.0) +
                snow_number_collected_by_rain + snow_number_collected_by_hail + snow_self_collection + snow_number_to_graupel;
            snow_scale = std::min(snow_scale, admission_ratio(number_source, number_sink));
        }
        if (snow_scale < 1.0)
        {
            if (snow_deposition < 0.0)
            {
                snow_deposition *= snow_scale;
                snow_number_sublimated *= snow_scale;
            }
            snow_collected_by_rain *= snow_scale;
            snow_number_collected_by_rain *= snow_scale;
            snow_collected_by_hail *= snow_scale;
            snow_number_collected_by_hail *= snow_scale;
            splinter_mass_from_snow *= snow_scale;
            splinter_number_from_snow *= snow_scale;
            snow_to_graupel *= snow_scale;
            snow_number_to_graupel *= snow_scale;
            if (snow_riming.active)
                snow_self_collection *= snow_scale;
        }

        const double graupel_source = static_cast<double>(cell[8]) + std::max(graupel_deposition, 0.0) +
                                      cloud_to_graupel + ice_collected_by_graupel +
                                      routed_collision_mass(FrozenDestination::Graupel) + snow_to_graupel +
                                      hail_to_graupel + (parameters.hail_processes ? 0.0 : rain_to_frozen);
        const double graupel_sink   = std::max(-graupel_deposition, 0.0) + graupel_collected_by_rain +
                                      splinter_mass_from_graupel + graupel_to_hail;
        const double graupel_scale  = admission_ratio(graupel_source, graupel_sink);
        if (graupel_scale < 1.0)
        {
            if (graupel_deposition < 0.0)
            {
                graupel_deposition *= graupel_scale;
                graupel_number_sublimated *= graupel_scale;
            }
            graupel_collected_by_rain *= graupel_scale;
            graupel_number_collected_by_rain *= graupel_scale;
            splinter_mass_from_graupel *= graupel_scale;
            splinter_number_from_graupel *= graupel_scale;
            graupel_to_hail *= graupel_scale;
            graupel_number_to_hail *= graupel_scale;
            hail_number_from_graupel *= graupel_scale;
        }

        const double hail_source = static_cast<double>(cell[9]) + std::max(hail_deposition, 0.0) + cloud_to_hail +
                                   ice_collected_by_hail + snow_collected_by_hail + rain_collected_by_hail +
                                   routed_collision_mass(FrozenDestination::Hail) + graupel_to_hail +
                                   (parameters.hail_processes ? rain_to_frozen : 0.0);
        const double hail_sink   = std::max(-hail_deposition, 0.0) + hail_to_graupel;
        const double hail_scale  = admission_ratio(hail_source, hail_sink);
        if (hail_scale < 1.0)
        {
            if (hail_deposition < 0.0)
            {
                hail_deposition *= hail_scale;
                hail_number_sublimated *= hail_scale;
            }
            hail_to_graupel *= hail_scale;
            hail_number_to_graupel *= hail_scale;
        }
        if (snow_riming.active)
        {
            snow_riming.admission_passes = admission_pass + 1;
            if (budgets_closed())
            {
                admission_closed = true;
                break;
            }
        }
    }

    if (snow_riming.active && !admission_closed)
    {
        throw ColdSnowRimingAdmissionError(ColdSnowRimingAdmissionFailure::BudgetCap,
            "cold snow riming admission exceeded its bounded donor-closure pass cap");
    }

    collision_mass_to_snow      = 0.0;
    collision_mass_to_graupel   = 0.0;
    collision_mass_to_hail      = 0.0;
    collision_number_to_snow    = 0.0;
    collision_number_to_graupel = 0.0;
    collision_number_to_hail    = 0.0;
    if (ice_rain_collision)
        route_collision(ice_collected_by_rain,
                        rain_collected_by_ice,
                        ice_number_collected_by_rain,
                        rain_number_collected_by_ice,
                        ice_psd,
                        500.0,
                        ice_rain_destination);
    if (snow_rain_collision)
        route_collision(snow_collected_by_rain,
                        rain_collected_by_snow,
                        snow_number_collected_by_rain,
                        rain_number_collected_by_snow,
                        snow_psd,
                        snow_collision_bulk_density,
                        snow_rain_destination);
    if (graupel_rain_collision)
        route_collision(graupel_collected_by_rain,
                        rain_collected_by_graupel,
                        graupel_number_collected_by_rain,
                        rain_number_collected_by_graupel,
                        graupel_psd,
                        400.0,
                        graupel_rain_destination);

    if (snow_riming.active)
    {
        snapshot_budgets();
        snow_riming.final_snow_mass_per_kg =
            snow_riming.donor_mass_source_per_kg[4] - snow_riming.donor_mass_sink_per_kg[4];
        snow_riming.final_snow_number_per_kg =
            snow_riming.snow_number_source_per_kg - snow_riming.snow_number_sink_per_kg;
        if (!std::isfinite(snow_riming.final_snow_mass_per_kg) || snow_riming.final_snow_mass_per_kg < 0.0 ||
            !std::isfinite(snow_riming.final_snow_number_per_kg) || snow_riming.final_snow_number_per_kg < 0.0 ||
            ((snow_riming.final_snow_mass_per_kg == 0.0) != (snow_riming.final_snow_number_per_kg == 0.0)))
        {
            throw ColdSnowRimingAdmissionError(ColdSnowRimingAdmissionFailure::InvalidEndpoint,
                "cold snow riming admission produced an incoherent final snow mass/number endpoint");
        }
    }

    ColdFrozenDiffusionAcceptedEvent staged_diffusion_event{};
    if constexpr (PublishedFrozenDiffusion && !JointFrozenDiffusion)
    {
        const bool graupel = diffusion_observer->species == ColdFrozenDiffusionSpecies::Graupel;
        const std::pair<const char*, double> unowned[]{
            {"vapor_to_ice", vapor_to_ice}, {"cloud_to_ice", cloud_to_ice},
            {"rain_to_frozen", rain_to_frozen}, {"ice_to_snow", ice_to_snow},
            {"nc_to_ice", nc_to_ice}, {"nr_to_frozen", nr_to_frozen},
            {"new_ice_number", new_ice_number},
            {"ice_number_sublimated", ice_number_sublimated},
            {"snow_number_sublimated", snow_number_sublimated},
            {"graupel_number_sublimated", graupel ? 0.0 : graupel_number_sublimated},
            {"hail_number_sublimated", graupel ? hail_number_sublimated : 0.0},
            {"ice_number_to_snow", ice_number_to_snow},
            {"cloud_to_snow", cloud_to_snow}, {"cloud_to_graupel", cloud_to_graupel},
            {"cloud_to_hail", cloud_to_hail},
            {"cloud_number_to_snow", cloud_number_to_snow},
            {"cloud_number_to_graupel", cloud_number_to_graupel},
            {"cloud_number_to_hail", cloud_number_to_hail},
            {"ice_collected_by_snow", ice_collected_by_snow},
            {"ice_collected_by_graupel", ice_collected_by_graupel},
            {"ice_collected_by_hail", ice_collected_by_hail},
            {"ice_number_collected_by_snow", ice_number_collected_by_snow},
            {"ice_number_collected_by_graupel", ice_number_collected_by_graupel},
            {"ice_number_collected_by_hail", ice_number_collected_by_hail},
            {"snow_self_collection", snow_self_collection},
            {"splinter_mass_from_snow", splinter_mass_from_snow},
            {"splinter_mass_from_graupel", splinter_mass_from_graupel},
            {"splinter_number_from_snow", splinter_number_from_snow},
            {"splinter_number_from_graupel", splinter_number_from_graupel},
            {"rain_collected_by_ice", rain_collected_by_ice},
            {"ice_collected_by_rain", ice_collected_by_rain},
            {"rain_number_collected_by_ice", rain_number_collected_by_ice},
            {"ice_number_collected_by_rain", ice_number_collected_by_rain},
            {"rain_collected_by_snow", rain_collected_by_snow},
            {"snow_collected_by_rain", snow_collected_by_rain},
            {"rain_number_collected_by_snow", rain_number_collected_by_snow},
            {"snow_number_collected_by_rain", snow_number_collected_by_rain},
            {"rain_collected_by_graupel", rain_collected_by_graupel},
            {"graupel_collected_by_rain", graupel_collected_by_rain},
            {"rain_number_collected_by_graupel", rain_number_collected_by_graupel},
            {"graupel_number_collected_by_rain", graupel_number_collected_by_rain},
            {"collision_mass_to_snow", collision_mass_to_snow},
            {"collision_mass_to_graupel", collision_mass_to_graupel},
            {"collision_mass_to_hail", collision_mass_to_hail},
            {"collision_number_to_snow", collision_number_to_snow},
            {"collision_number_to_graupel", collision_number_to_graupel},
            {"collision_number_to_hail", collision_number_to_hail},
            {"snow_to_graupel", snow_to_graupel},
            {"snow_number_to_graupel", snow_number_to_graupel},
            {"graupel_to_hail", graupel_to_hail},
            {"graupel_number_to_hail", graupel_number_to_hail},
            {"hail_number_from_graupel", hail_number_from_graupel},
            {"hail_to_graupel", hail_to_graupel},
            {"hail_number_to_graupel", hail_number_to_graupel},
            {"rain_collected_by_hail", rain_collected_by_hail},
            {"rain_number_collected_by_hail", rain_number_collected_by_hail},
            {"snow_collected_by_hail", snow_collected_by_hail},
            {"snow_number_collected_by_hail", snow_number_collected_by_hail},
            {"shed_rain_from_hail", shed_rain_from_hail},
            {"shed_rain_number", shed_rain_number},
            {"ice_deposition", ice_deposition}, {"snow_deposition", snow_deposition},
            {"graupel_deposition", graupel ? 0.0 : graupel_deposition},
            {"hail_deposition", graupel ? hail_deposition : 0.0}};
        for (const auto& channel : unowned)
        {
            if (channel.second != 0.0)
            {
                throw std::runtime_error(
                    std::string("published frozen diffusion observed unowned cold channel: ")
                    + channel.first);
            }
        }
        staged_diffusion_event = make_published_frozen_diffusion_event(
            diffusion_observer->species, cell[graupel ? 8 : 9],
            graupel ? graupel_number_state : hail_number_state,
            graupel ? parameters.alpha_g_ : parameters.alpha_h_,
            graupel ? graupel_deposition : hail_deposition,
            graupel ? graupel_number_sublimated : hail_number_sublimated,
            density, dt);
        if (diffusion_observer->size == diffusion_observer->capacity)
        {
            throw std::runtime_error("published frozen diffusion event capacity is exhausted");
        }
    }

    std::array<ColdFrozenDiffusionAcceptedEvent, JointFrozenDiffusion ? 2 : 0>
        staged_joint_diffusion_events{};
    if constexpr (JointFrozenDiffusion)
    {
        const std::array<double, 64> channels{
            vapor_to_ice, cloud_to_ice, rain_to_frozen, ice_to_snow,
            nc_to_ice, nr_to_frozen, new_ice_number,
            ice_number_sublimated, snow_number_sublimated,
            graupel_number_sublimated, hail_number_sublimated, ice_number_to_snow,
            cloud_to_snow, cloud_to_graupel, cloud_to_hail,
            cloud_number_to_snow, cloud_number_to_graupel, cloud_number_to_hail,
            ice_collected_by_snow, ice_collected_by_graupel, ice_collected_by_hail,
            ice_number_collected_by_snow, ice_number_collected_by_graupel,
            ice_number_collected_by_hail, snow_self_collection,
            splinter_mass_from_snow, splinter_mass_from_graupel,
            splinter_number_from_snow, splinter_number_from_graupel,
            rain_collected_by_ice, ice_collected_by_rain,
            rain_number_collected_by_ice, ice_number_collected_by_rain,
            rain_collected_by_snow, snow_collected_by_rain,
            rain_number_collected_by_snow, snow_number_collected_by_rain,
            rain_collected_by_graupel, graupel_collected_by_rain,
            rain_number_collected_by_graupel, graupel_number_collected_by_rain,
            collision_mass_to_snow, collision_mass_to_graupel, collision_mass_to_hail,
            collision_number_to_snow, collision_number_to_graupel, collision_number_to_hail,
            snow_to_graupel, snow_number_to_graupel,
            graupel_to_hail, graupel_number_to_hail, hail_number_from_graupel,
            hail_to_graupel, hail_number_to_graupel,
            rain_collected_by_hail, rain_number_collected_by_hail,
            snow_collected_by_hail, snow_number_collected_by_hail,
            shed_rain_from_hail, shed_rain_number,
            ice_deposition, snow_deposition, graupel_deposition, hail_deposition};
        std::string error;
        if (!validate_published_joint_frozen_diffusion_channels(channels, error))
        {
            throw std::runtime_error(error);
        }
        staged_joint_diffusion_events[0] = make_published_frozen_diffusion_event(
            ColdFrozenDiffusionSpecies::Graupel, cell[8], graupel_number_state,
            parameters.alpha_g_, graupel_deposition, graupel_number_sublimated,
            density, dt);
        staged_joint_diffusion_events[1] = make_published_frozen_diffusion_event(
            ColdFrozenDiffusionSpecies::Hail, cell[9], hail_number_state,
            parameters.alpha_h_, hail_deposition, hail_number_sublimated,
            density, dt);
        if (joint_diffusion_observer->capacity - joint_diffusion_observer->size < 2)
        {
            throw std::runtime_error("published joint frozen diffusion event capacity is exhausted");
        }
        const double deposition = graupel_deposition + hail_deposition;
        const std::array<double, 6> raw_rates{
            graupel_deposition / dt, hail_deposition / dt, -deposition / dt,
            microphysics_constants::L_s * deposition / (microphysics_constants::cp * dt),
            graupel_number_sublimated / dt, hail_number_sublimated / dt};
        for (double rate : raw_rates)
        {
            if (!published_joint_frozen_diffusion_rate_representable(rate))
            {
                throw std::runtime_error("published joint frozen diffusion rate is not FP32 representable");
            }
        }
    }

    ColdRainFreezingAcceptedEvent staged_freezing_event{};
    const bool publish_freezing_event = observer && (rain_to_frozen > 0.0 || nr_to_frozen > 0.0);
    if (publish_freezing_event)
    {
        staged_freezing_event = make_cold_rain_freezing_accepted_event(
            parameters.hail_processes ? ColdRainFreezingDestination::Hail :
                                        ColdRainFreezingDestination::Graupel,
            rain, rain_number, rain_to_frozen, nr_to_frozen, density, dt);
        if (observer->size == observer->capacity)
        {
            throw std::runtime_error("Milbrandt rain-freezing observer capacity is exhausted");
        }
    }

    const double deposition_total =
        vapor_to_ice + ice_deposition + snow_deposition + graupel_deposition + hail_deposition;
    auto output_rates = rates;
    output_rates[6] -= static_cast<float>(deposition_total / dt);
    output_rates[0] -= static_cast<float>((cloud_to_ice + cloud_to_snow + cloud_to_graupel + cloud_to_hail) / dt);
    output_rates[1] -= static_cast<float>((rain_to_frozen + rain_collected_by_ice + rain_collected_by_snow +
                                    rain_collected_by_graupel + rain_collected_by_hail) /
                                   dt);
    output_rates[2] += static_cast<float>((cloud_to_ice + vapor_to_ice + ice_deposition - ice_to_snow - ice_collected_by_snow -
                                    ice_collected_by_graupel - ice_collected_by_hail - ice_collected_by_rain +
                                    splinter_mass_from_snow + splinter_mass_from_graupel) /
                                   dt);
    output_rates[3] += static_cast<float>((snow_deposition + ice_to_snow + cloud_to_snow + ice_collected_by_snow -
                                    snow_collected_by_rain - snow_collected_by_hail - splinter_mass_from_snow +
                                    collision_mass_to_snow - snow_to_graupel) /
                                   dt);
    if (parameters.hail_processes)
    {
        output_rates[5] += static_cast<float>((hail_deposition + rain_to_frozen + cloud_to_hail + ice_collected_by_hail +
                                        snow_collected_by_hail + rain_collected_by_hail + collision_mass_to_hail +
                                        graupel_to_hail - hail_to_graupel) /
                                       dt);
        output_rates[4] +=
            static_cast<float>((graupel_deposition + cloud_to_graupel + ice_collected_by_graupel -
                                graupel_collected_by_rain - splinter_mass_from_graupel + collision_mass_to_graupel +
                                snow_to_graupel - graupel_to_hail + hail_to_graupel) /
                               dt);
    }
    else
    {
        output_rates[5] += static_cast<float>((hail_deposition + cloud_to_hail + ice_collected_by_hail +
                                        snow_collected_by_hail + rain_collected_by_hail - hail_to_graupel) /
                                       dt);
        output_rates[4] +=
            static_cast<float>((graupel_deposition + rain_to_frozen + cloud_to_graupel + ice_collected_by_graupel -
                                graupel_collected_by_rain - splinter_mass_from_graupel + collision_mass_to_graupel +
                                snow_to_graupel + hail_to_graupel) /
                               dt);
    }
    output_rates[7] += static_cast<float>(
        (microphysics_constants::L_s * deposition_total +
         microphysics_constants::L_f *
             (cloud_to_ice + rain_to_frozen + rain_collected_by_ice + rain_collected_by_snow +
              rain_collected_by_graupel + rain_collected_by_hail + cloud_to_snow + cloud_to_graupel + cloud_to_hail)) /
        (microphysics_constants::cp * dt));

    output_rates[8] -=
        static_cast<float>((nc_to_ice + cloud_number_to_snow + cloud_number_to_graupel + cloud_number_to_hail) / dt);
    output_rates[9] -=
        static_cast<float>((nr_to_frozen + rain_number_collected_by_ice + rain_number_collected_by_snow +
                            rain_number_collected_by_graupel + rain_number_collected_by_hail - shed_rain_number) /
                           dt);
    output_rates[10] += static_cast<float>((nc_to_ice + new_ice_number / density + ice_number_sublimated - ice_number_to_snow -
                                     ice_number_collected_by_snow - ice_number_collected_by_graupel -
                                     ice_number_collected_by_hail - ice_number_collected_by_rain +
                                     splinter_number_from_snow + splinter_number_from_graupel) /
                                    dt);
    output_rates[11] += static_cast<float>((snow_number_sublimated + ice_number_to_snow - snow_number_collected_by_rain -
                                     snow_number_collected_by_hail - snow_self_collection + collision_number_to_snow -
                                     snow_number_to_graupel) /
                                    dt);
    output_rates[12] += static_cast<float>((graupel_number_sublimated + (parameters.hail_processes ? 0.0 : nr_to_frozen) -
                                     graupel_number_collected_by_rain + collision_number_to_graupel +
                                     snow_number_to_graupel - graupel_number_to_hail + hail_number_to_graupel) /
                                    dt);
    output_rates[13] += static_cast<float>((hail_number_sublimated + (parameters.hail_processes ? nr_to_frozen : 0.0) +
                                     collision_number_to_hail + hail_number_from_graupel - hail_number_to_graupel) /
                                    dt);
    if (snow_riming.active)
    {
        const double mass = cell[7] + dt *
            (static_cast<double>(output_rates[3]) - static_cast<double>(rates[3]));
        const double number = cell[13] + dt * static_cast<double>(output_rates[11]);
        const float stored_mass = static_cast<float>(mass);
        const float stored_number = static_cast<float>(number);
        if (!std::isfinite(mass) || mass < 0.0 || !std::isfinite(number) || number < 0.0 ||
            ((mass == 0.0) != (number == 0.0)) ||
            !std::isfinite(stored_mass) || stored_mass < 0.0f ||
            !std::isfinite(stored_number) || stored_number < 0.0f ||
            ((stored_mass == 0.0f) != (stored_number == 0.0f)))
        {
            throw ColdSnowRimingAdmissionError(ColdSnowRimingAdmissionFailure::InvalidEndpoint,
                "cold snow riming admission produced an incoherent published snow mass/number endpoint");
        }
    }
    rates = output_rates;
    if (publish_freezing_event)
    {
        observer->events[observer->size] = staged_freezing_event;
        ++observer->size;
    }
    if constexpr (PublishedFrozenDiffusion && !JointFrozenDiffusion)
    {
        diffusion_observer->events[diffusion_observer->size] = staged_diffusion_event;
        ++diffusion_observer->size;
    }
    if constexpr (JointFrozenDiffusion)
    {
        for (const auto& event : staged_joint_diffusion_events)
        {
            joint_diffusion_observer->events[joint_diffusion_observer->size] = event;
            ++joint_diffusion_observer->size;
        }
    }
    if (snow_riming_observer)
    {
        snow_riming.temperature_k = T;
        snow_riming.pressure_pa = pressure;
        snow_riming.dry_density_kg_m3 = density;
        snow_riming.duration_s = dt;
        snow_riming.admitted_cloud_mass_per_kg = {cloud_to_snow, cloud_to_graupel, cloud_to_hail};
        snow_riming.admitted_cloud_number_per_kg = {cloud_number_to_snow, cloud_number_to_graupel, cloud_number_to_hail};
        snow_riming.admitted_snow_mass_increment_per_kg = snow_deposition;
        snow_riming.admitted_snow_number_increment_per_kg = snow_number_sublimated;
        *snow_riming_observer = snow_riming;
    }
}

inline void cold_cell(
    const ColdParameters& parameters, const std::array<double, 16>& cell, double dt,
    std::array<float, 14>& rates, ColdRainFreezingAcceptedEventBuffer* observer = nullptr,
    ColdSnowRimingAdmissionObservation* snow_riming_observer = nullptr,
    unsigned snow_riming_max_passes = cold_snow_riming_admission_max_passes)
{
    cold_cell_impl<false>(parameters, cell, dt, rates, observer, nullptr, nullptr,
        snow_riming_observer, snow_riming_max_passes);
}

inline void published_frozen_diffusion_cold_cell(
    const ColdParameters& parameters, const std::array<double, 16>& cell, double dt,
    std::array<float, 14>& rates, ColdFrozenDiffusionAcceptedEventBuffer& observer)
{
    cold_cell_impl<true>(parameters, cell, dt, rates, nullptr, &observer);
}

inline void published_joint_frozen_diffusion_cold_cell(
    const ColdParameters& parameters, const std::array<double, 16>& cell, double dt,
    std::array<float, 14>& rates, ColdJointFrozenDiffusionAcceptedEventBuffer& observer)
{
    cold_cell_impl<true, true>(parameters, cell, dt, rates, nullptr, nullptr, &observer);
}
} // namespace tmv::milbrandt_detail
