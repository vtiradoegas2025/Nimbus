#pragma once

#include "microphysics/base/thermodynamics.hpp"
#include "microphysics/milbrandt_warm_parameters.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace tmv::milbrandt_detail
{
struct WarmRainCell
{
    float temperature, pressure, density, vapor, cloud, rain, cloud_number, rain_number;
};

struct DoubleWarmRainCell
{
    double temperature, pressure, density, vapor, cloud, rain, cloud_number, rain_number;
};

struct WarmRainTendencies
{
    float cloud, rain, vapor, temperature, cloud_number, rain_number;
};

enum class WarmRainAcceptedEventKind
{
    InitialCloudNumberClamp,
    InitialRainNumberClamp,
    PreRainNumberRegularization,
    Autoconversion,
    Accretion,
    CloudSelfCollection,
    RainSelfCollection,
    CloudMassClip,
    RainMassClip,
    CloudNumberClip,
    RainNumberClip,
    PostRainNumberRegularization,
    RainToCloud,
    RainReset,
    Evaporation,
    RainBreakup
};

struct WarmRainAcceptedEvent
{
    WarmRainAcceptedEventKind kind = WarmRainAcceptedEventKind::Autoconversion;
    double cloud_mass_rate_per_kg_s = 0.0;
    double rain_mass_rate_per_kg_s = 0.0;
    double vapor_mass_rate_per_kg_s = 0.0;
    double cloud_number_rate_per_m3_s = 0.0;
    double rain_number_rate_per_m3_s = 0.0;
    double temperature_rate_k_s = 0.0;
    double dry_density_kg_m3 = 0.0;
    double cloud_donor_scale = 1.0;
    double seed_diameter_m = 0.0;
    bool seed_mass_number_coherent = true;
};

struct WarmRainAcceptedEventBuffer
{
    std::array<WarmRainAcceptedEvent, 16> events{};
    std::size_t size = 0;
    std::size_t capacity = 16;
};

static_assert(std::is_trivially_copyable_v<WarmRainAcceptedEvent>);

struct WarmRainEvaporationBudget
{
    bool present = false;
    double mass_before_per_kg = 0.0;
    double number_before_per_m3 = 0.0;
    double admitted_mass_per_kg = 0.0;
    double mass_after_per_kg = 0.0;
    double number_after_per_m3 = 0.0;
    bool donor_exhausted = false;
};

struct WarmRainReclassificationBudget
{
    bool present = false;
    double rain_mass_before_per_kg = 0.0;
    double rain_number_before_per_m3 = 0.0;
    double cloud_mass_before_per_kg = 0.0;
    double cloud_number_before_per_m3 = 0.0;
};

struct WarmRainDoubleEndpoints
{
    double vapor_per_kg = 0.0;
    double cloud_per_kg = 0.0;
    double rain_per_kg = 0.0;
    double cloud_number_per_kg = 0.0;
    double rain_number_per_kg = 0.0;
    double temperature_increment_k = 0.0;
    double cloud_donor_budget_per_kg = 0.0;
    bool cloud_donor_exhausted = false;
    WarmRainEvaporationBudget evaporation{};
    WarmRainReclassificationBudget reclassification{};
};

class TripleWarmRainExactResult
{
public:
    const WarmRainCell& source_cell() const noexcept
    {
        return source_;
    }
    const WarmRainParameters& parameters() const noexcept
    {
        return parameters_;
    }
    const WarmRainTendencies& incoming_rates() const noexcept
    {
        return incoming_;
    }
    const WarmRainTendencies& rates() const noexcept
    {
        return rates_;
    }
    const WarmRainAcceptedEventBuffer& events() const noexcept
    {
        return events_;
    }
    const WarmRainDoubleEndpoints& endpoints() const noexcept
    {
        return endpoints_;
    }
    double duration_s() const noexcept
    {
        return duration_s_;
    }
    double dry_density_kg_m3() const noexcept
    {
        return source_.density;
    }

private:
    friend void triple_warm_rain_cell_exact(
        const WarmRainParameters&, const WarmRainCell&, double,
        const WarmRainTendencies&, TripleWarmRainExactResult&, std::size_t);
    WarmRainCell source_{};
    WarmRainParameters parameters_{};
    WarmRainTendencies incoming_{};
    WarmRainTendencies rates_{};
    WarmRainAcceptedEventBuffer events_{};
    WarmRainDoubleEndpoints endpoints_{};
    double duration_s_ = 0.0;
};

class TripleWarmRainResult
{
public:
    const WarmRainTendencies& rates() const noexcept { return rates_; }
    const WarmRainAcceptedEventBuffer& events() const noexcept
    {
        return events_;
    }
    double duration_s() const noexcept { return duration_s_; }
    double dry_density_kg_m3() const noexcept { return dry_density_kg_m3_; }

private:
    friend void triple_warm_rain_cell(
        const WarmRainParameters&, const WarmRainCell&, double,
        const WarmRainTendencies&, TripleWarmRainResult&, std::size_t);
    WarmRainTendencies rates_{};
    WarmRainAcceptedEventBuffer events_{};
    double duration_s_ = 0.0;
    double dry_density_kg_m3_ = 0.0;
};

struct RainTripleMomentState;
struct ExactRainTripleMomentState;
class TripleWarmRainDynamicResult;
class TripleWarmRainDynamicFP64Result;
struct DynamicWarmFP64Policies;

class WarmRainObserverCapacityError : public std::runtime_error
{
public:
    WarmRainObserverCapacityError()
        : std::runtime_error("Milbrandt warm-event observer capacity is exhausted")
    {
    }
};

class TripleWarmRainDoubleSourceResult
{
public:
    const DoubleWarmRainCell& source_cell() const noexcept
    {
        return source_;
    }
    const WarmRainParameters& parameters() const noexcept
    {
        return parameters_;
    }
    const WarmRainTendencies& incoming_rates() const noexcept
    {
        return incoming_;
    }
    const WarmRainTendencies& rates() const noexcept
    {
        return rates_;
    }
    const WarmRainAcceptedEventBuffer& events() const noexcept
    {
        return events_;
    }
    const WarmRainDoubleEndpoints& endpoints() const noexcept
    {
        return endpoints_;
    }
    double duration_s() const noexcept
    {
        return duration_s_;
    }
    double dry_density_kg_m3() const noexcept
    {
        return source_.density;
    }
    bool uses_dynamic_spectrum() const noexcept
    {
        return uses_dynamic_spectrum_;
    }

private:
    friend void triple_warm_rain_cell_exact_double_source(
        const WarmRainParameters&, const DoubleWarmRainCell&, double,
        const WarmRainTendencies&, TripleWarmRainDoubleSourceResult&, std::size_t);
    friend void triple_warm_rain_cell_dynamic(
        const WarmRainParameters&, const DoubleWarmRainCell&, double,
        const WarmRainTendencies&, const RainTripleMomentState&,
        TripleWarmRainDynamicResult&, std::size_t);
    friend void triple_warm_rain_cell_dynamic_fp64(
        const WarmRainParameters&, const DoubleWarmRainCell&, double,
        const WarmRainTendencies&, const ExactRainTripleMomentState&,
        TripleWarmRainDynamicFP64Result&, std::size_t, const DynamicWarmFP64Policies&);
    DoubleWarmRainCell source_{};
    WarmRainParameters parameters_{};
    WarmRainTendencies incoming_{};
    WarmRainTendencies rates_{};
    WarmRainAcceptedEventBuffer events_{};
    WarmRainDoubleEndpoints endpoints_{};
    double duration_s_ = 0.0;
    bool uses_dynamic_spectrum_ = false;
};

struct FixedWarmRainSpectrumPolicy
{
    static constexpr bool dynamic = false;
    static constexpr bool typed_resource = false;
};

template <bool CoherentTripleAutoNumber, typename Cell,
          typename SpectrumPolicy = FixedWarmRainSpectrumPolicy>
inline void
warm_rain_cell_impl(const WarmRainParameters& parameters,
                    const Cell& cell,
                    double dt,
                    WarmRainTendencies& rates,
                    WarmRainAcceptedEventBuffer* observer,
                    WarmRainDoubleEndpoints* exact_endpoints = nullptr,
                    SpectrumPolicy* spectrum_policy = nullptr)
{
    constexpr double eps_q                = 1.0e-14;
    constexpr double eps_n                = 1.0e-3; // volumetric, m-3
    constexpr double sigma_threshold_m    = 15.0e-6;
    constexpr double kk1                  = 3.03e3;
    constexpr double kk2                  = 2.59e15;
    constexpr double rain_hump_diameter_m = 82.0e-6;
    constexpr double rain_max_diameter_m  = 5.0e-3;
    if constexpr (SpectrumPolicy::dynamic)
    {
        static_assert(CoherentTripleAutoNumber &&
                      std::is_same_v<Cell, DoubleWarmRainCell>);
        if (!spectrum_policy || !observer || !exact_endpoints)
        {
            throw std::invalid_argument("dynamic triple warm capture requires its phase owners");
        }
    }
    if constexpr (CoherentTripleAutoNumber)
    {
        if (exact_endpoints && !observer)
        {
            throw std::invalid_argument("exact triple warm capture requires its event observer");
        }
    }
    bool apply_pre_number_regularization = true;
    bool apply_post_number_regularization = true;
    bool apply_reclassification = true;
    if constexpr (SpectrumPolicy::typed_resource)
    {
        apply_pre_number_regularization = spectrum_policy->pre_number_regularization();
        apply_post_number_regularization = spectrum_policy->post_number_regularization();
        apply_reclassification = spectrum_policy->reclassification();
    }

    std::array<WarmRainAcceptedEvent, 16> staged_events{};
    std::size_t staged_count = 0;
    const auto stage_event = [&](WarmRainAcceptedEvent event)
    {
        if (!observer)
        {
            return;
        }
        if (staged_count >= staged_events.size())
        {
            if constexpr (SpectrumPolicy::typed_resource)
            {
                throw WarmRainObserverCapacityError();
            }
            else
            {
                throw std::runtime_error("Milbrandt warm-event observer capacity is exhausted");
            }
        }
        staged_events[staged_count++] = event;
    };

    const double density       = cell.density;
    const double pressure      = cell.pressure;
    double       vapor         = cell.vapor;
    double       cloud         = cell.cloud;
    double       rain          = cell.rain;
    const double temperature_k = cell.temperature;
    const double nc_old_mix    = cell.cloud_number;
    const double nr_old_mix    = cell.rain_number;
    double       nc            = density * (nc_old_mix + dt * static_cast<double>(rates.cloud_number));
    double       nr            = density * nr_old_mix;
    const double raw_nc        = nc;
    const double raw_nr        = nr;
    nc                         = std::max(0.0, nc);
    nr                         = std::max(0.0, nr);
    if (observer && nc != raw_nc)
    {
        stage_event({WarmRainAcceptedEventKind::InitialCloudNumberClamp,
                     0.0, 0.0, 0.0, (nc - raw_nc) / dt, 0.0,
                     0.0, density});
    }
    if (observer && nr != raw_nr)
    {
        stage_event({WarmRainAcceptedEventKind::InitialRainNumberClamp,
                     0.0, 0.0, 0.0, 0.0, (nr - raw_nr) / dt,
                     0.0, density});
    }

    double i_lambda_c          = 0.0;
    double i_lambda_c3         = 0.0;
    double i_lambda_c6         = 0.0;
    double cloud_mean_diameter = 0.0;
    double cloud_sigma         = 0.0;
    double liquid_water_scale  = 0.0;
    double autoconversion_tau  = 0.0;
    if (cloud > eps_q && nc > eps_n)
    {
        i_lambda_c                   = std::cbrt(density * cloud / nc * parameters.inv_cloud_mass_factor);
        i_lambda_c3                  = i_lambda_c * i_lambda_c * i_lambda_c;
        i_lambda_c6                  = i_lambda_c3 * i_lambda_c3;
        cloud_mean_diameter          = i_lambda_c * std::cbrt(parameters.gc2 / parameters.gc1);
        const double variance_factor = std::max(
            0.0,
            parameters.gc3 / parameters.gc1 - (parameters.gc2 / parameters.gc1) * (parameters.gc2 / parameters.gc1));
        cloud_sigma = i_lambda_c * std::pow(variance_factor, 1.0 / 6.0);
        liquid_water_scale =
            0.027 * density * cloud * (6.25e18 * cloud_sigma * cloud_sigma * cloud_sigma * cloud_mean_diameter - 0.4);
        if (cloud_sigma > sigma_threshold_m)
        {
            autoconversion_tau = 3.7 / (density * cloud * (0.5e6 * cloud_sigma - 7.5));
        }
    }

    double rain_mean_diameter = 0.0;
    double i_lambda_r         = 0.0;
    double i_lambda_r3        = 0.0;
    double i_lambda_r6        = 0.0;
    WarmRainReclassificationBudget reclassification_budget;
    if (rain > eps_q && nr > eps_n)
    {
        rain_mean_diameter = std::cbrt(density * rain / (nr * parameters.c_r));
        if (rain_mean_diameter > 3.0e-3 && apply_pre_number_regularization)
        {
            const double number_before_regularization = nr;
            const double excess = rain_mean_diameter - 3.0e-3;
            nr *= std::max(1.0 + 2.0e4 * excess * excess, std::pow(rain_mean_diameter / rain_max_diameter_m, 3.0));
            if (observer)
            {
                stage_event({WarmRainAcceptedEventKind::PreRainNumberRegularization,
                             0.0, 0.0, 0.0, 0.0,
                             (nr - number_before_regularization) / dt,
                             0.0, density});
            }
            rain_mean_diameter = std::cbrt(density * rain / (nr * parameters.c_r));
        }
        i_lambda_r  = std::cbrt(density * rain / nr * parameters.inv_rain_mass_factor);
        i_lambda_r3 = i_lambda_r * i_lambda_r * i_lambda_r;
        i_lambda_r6 = i_lambda_r3 * i_lambda_r3;
    }
    if constexpr (SpectrumPolicy::dynamic)
    {
        spectrum_policy->before_physical(staged_events, staged_count,
                                         rain, nr / density);
    }

    double auto_mass_rate        = 0.0;
    double auto_number_rate      = 0.0;
    double cloud_self_collection = 0.0;
    double initial_rain_diameter = 0.0;
    double observed_seed_diameter = 0.0;
    if (cloud > eps_q && cloud_sigma > sigma_threshold_m && autoconversion_tau > 0.0)
    {
        auto_mass_rate                   = std::min(std::max(liquid_water_scale / autoconversion_tau, 0.0), cloud / dt);
        initial_rain_diameter            = std::max(83.0e-6, 12.6e-4 / (0.5e6 * cloud_sigma - 3.5));
        const double conversion_diameter = std::max(initial_rain_diameter, rain_mean_diameter);
        auto_number_rate = auto_mass_rate * density / (parameters.c_r * std::pow(conversion_diameter, parameters.d_r));
        if (observer)
        {
            observed_seed_diameter = conversion_diameter;
        }
        cloud_self_collection = std::min(kk2 * nc * nc * parameters.gc3 / parameters.gc1 * i_lambda_c6, nc / dt);
    }

    double     accretion_mass_rate   = 0.0;
    double     accretion_number_rate = 0.0;
    double     rain_self_collection  = 0.0;
    const bool rain_present          = rain > eps_q && nr > eps_n;
    if (rain_present && (rain > 1.2 * std::max(liquid_water_scale, 0.0) / density ||
                         rain_mean_diameter > std::max(5.0e-6, initial_rain_diameter)))
    {
        if (cloud > eps_q && liquid_water_scale > 0.0)
        {
            if (rain_mean_diameter >= 100.0e-6)
            {
                if constexpr (SpectrumPolicy::dynamic)
                {
                    accretion_number_rate =
                        kk1 * nc * nr *
                        (parameters.gc2 / parameters.gc1 * i_lambda_c3 +
                         spectrum_policy->diameter_moment_3_m3());
                    accretion_mass_rate =
                        parameters.c_r / density * kk1 * nc * nr * i_lambda_c3 *
                        (parameters.gc3 / parameters.gc1 * i_lambda_c3 +
                         parameters.gc2 / parameters.gc1 *
                             spectrum_policy->diameter_moment_3_m3());
                }
                else
                {
                    accretion_number_rate =
                        kk1 * nc * nr *
                        (parameters.gc2 / parameters.gc1 * i_lambda_c3 + parameters.gr34 / parameters.gr31 * i_lambda_r3);
                    accretion_mass_rate =
                        parameters.c_r / density * kk1 * nc * nr * i_lambda_c3 *
                        (parameters.gc3 / parameters.gc1 * i_lambda_c3 +
                         parameters.gc2 / parameters.gc1 * parameters.gr34 / parameters.gr31 * i_lambda_r3);
                }
            }
            else
            {
                if constexpr (SpectrumPolicy::dynamic)
                {
                    accretion_number_rate =
                        kk2 * nc * nr *
                        (parameters.gc3 / parameters.gc1 * i_lambda_c6 +
                         spectrum_policy->diameter_moment_6_m6());
                    accretion_mass_rate =
                        parameters.c_r / density * kk2 * nc * nr * i_lambda_c3 *
                        (parameters.gc4 / parameters.gc1 * i_lambda_c6 +
                         parameters.gc2 / parameters.gc1 *
                             spectrum_policy->diameter_moment_6_m6());
                }
                else
                {
                    accretion_number_rate =
                        kk2 * nc * nr *
                        (parameters.gc3 / parameters.gc1 * i_lambda_c6 + parameters.gr37 / parameters.gr31 * i_lambda_r6);
                    accretion_mass_rate =
                        parameters.c_r / density * kk2 * nc * nr * i_lambda_c3 *
                        (parameters.gc4 / parameters.gr31 * i_lambda_c6 +
                         parameters.gc2 / parameters.gc1 * parameters.gr37 / parameters.gr31 * i_lambda_r6);
                }
            }
            accretion_number_rate = std::min(accretion_number_rate, nc / dt);
            accretion_mass_rate   = std::min(accretion_mass_rate, cloud / dt);
        }

        const double nr_squared      = nr * nr;
        if constexpr (SpectrumPolicy::dynamic)
        {
            rain_self_collection = rain_mean_diameter >= 100.0e-6
                ? kk1 * nr_squared * spectrum_policy->diameter_moment_3_m3()
                : kk2 * nr_squared * spectrum_policy->diameter_moment_6_m6();
        }
        else
        {
            rain_self_collection         = rain_mean_diameter >= 100.0e-6
                                               ? kk1 * nr_squared * parameters.gr34 / parameters.gr31 * i_lambda_r3
                                               : kk2 * nr_squared * parameters.gr37 / parameters.gr31 * i_lambda_r6;
        }
        double collection_efficiency = 1.0;
        double breakup_inverse_slope = i_lambda_r;
        if constexpr (SpectrumPolicy::dynamic)
        {
            breakup_inverse_slope =
                spectrum_policy->alpha_zero_equivalent_inverse_slope_m();
        }
        if (breakup_inverse_slope > 300.0e-6)
        {
            collection_efficiency = 2.0 - std::exp(2300.0 * (breakup_inverse_slope - 300.0e-6));
        }
        rain_self_collection = std::min(collection_efficiency * rain_self_collection, 0.5 * nr / dt);
    }

    double observed_donor_scale = 1.0;
    double exact_cloud_budget = 0.0;
    bool exact_cloud_exhausted = false;
    // A capped rate can exhaust the donor even when its rounded dt product does not exceed it.
    if constexpr (CoherentTripleAutoNumber && std::is_same_v<Cell, DoubleWarmRainCell>)
    {
        if (exact_endpoints && cloud > 0.0 &&
            auto_mass_rate + accretion_mass_rate >= cloud / dt)
        {
            exact_cloud_budget = cloud;
            exact_cloud_exhausted = true;
        }
    }
    const double cloud_sink = (auto_mass_rate + accretion_mass_rate) * dt;
    if (cloud_sink > cloud && cloud_sink > 0.0)
    {
        const double scale = cloud / cloud_sink;
        if constexpr (CoherentTripleAutoNumber)
        {
            if (exact_endpoints)
            {
                exact_cloud_budget = cloud;
                exact_cloud_exhausted = true;
            }
        }
        if (observer)
        {
            observed_donor_scale = scale;
        }
        auto_mass_rate *= scale;
        accretion_mass_rate *= scale;
        accretion_number_rate *= scale;
        if constexpr (CoherentTripleAutoNumber)
        {
            auto_number_rate *= scale;
        }
    }

    if (observer)
    {
        if (auto_mass_rate != 0.0 || auto_number_rate != 0.0)
        {
            stage_event({WarmRainAcceptedEventKind::Autoconversion,
                         -auto_mass_rate, auto_mass_rate, 0.0,
                         0.0, auto_number_rate, 0.0, density,
                         observed_donor_scale, observed_seed_diameter,
                         CoherentTripleAutoNumber ||
                             observed_donor_scale == 1.0});
        }
        if (accretion_mass_rate != 0.0 || accretion_number_rate != 0.0)
        {
            stage_event({WarmRainAcceptedEventKind::Accretion,
                         -accretion_mass_rate, accretion_mass_rate,
                         0.0, -accretion_number_rate, 0.0,
                         0.0, density, observed_donor_scale});
        }
        if (cloud_self_collection != 0.0)
        {
            stage_event({WarmRainAcceptedEventKind::CloudSelfCollection,
                         0.0, 0.0, 0.0, -cloud_self_collection,
                         0.0, 0.0, density});
        }
        if (rain_self_collection != 0.0)
        {
            auto kind = WarmRainAcceptedEventKind::RainSelfCollection;
            if constexpr (SpectrumPolicy::dynamic)
            {
                if (rain_self_collection < 0.0)
                {
                    kind = WarmRainAcceptedEventKind::RainBreakup;
                }
            }
            stage_event({kind,
                         0.0, 0.0, 0.0, 0.0,
                         -rain_self_collection, 0.0, density});
        }
    }

    const double observed_cloud_before = observer ? cloud : 0.0;
    const double observed_rain_before = observer ? rain : 0.0;
    const double observed_nc_before = observer ? nc : 0.0;
    const double observed_nr_before = observer ? nr : 0.0;
    cloud = std::max(0.0, cloud - dt * (auto_mass_rate + accretion_mass_rate));
    rain  = std::max(0.0, rain + dt * (auto_mass_rate + accretion_mass_rate));
    nc    = std::max(0.0, nc - dt * (accretion_number_rate + cloud_self_collection));
    nr    = std::max(0.0, nr + dt * (auto_number_rate - rain_self_collection));
    if constexpr (CoherentTripleAutoNumber)
    {
        if (exact_cloud_exhausted)
        {
            cloud = observed_cloud_before - exact_cloud_budget;
            rain = observed_rain_before + exact_cloud_budget;
        }
    }
    if (observer)
    {
        double unclipped_cloud =
            observed_cloud_before - dt * (auto_mass_rate + accretion_mass_rate);
        double unclipped_rain =
            observed_rain_before + dt * (auto_mass_rate + accretion_mass_rate);
        if constexpr (CoherentTripleAutoNumber)
        {
            if (exact_cloud_exhausted)
            {
                unclipped_cloud = observed_cloud_before - exact_cloud_budget;
                unclipped_rain = observed_rain_before + exact_cloud_budget;
            }
        }
        const double unclipped_nc =
            observed_nc_before - dt * (accretion_number_rate + cloud_self_collection);
        const double unclipped_nr =
            observed_nr_before + dt * (auto_number_rate - rain_self_collection);
        if (cloud != unclipped_cloud)
        {
            stage_event({WarmRainAcceptedEventKind::CloudMassClip,
                         (cloud - unclipped_cloud) / dt, 0.0, 0.0,
                         0.0, 0.0, 0.0, density});
        }
        if (rain != unclipped_rain)
        {
            stage_event({WarmRainAcceptedEventKind::RainMassClip,
                         0.0, (rain - unclipped_rain) / dt, 0.0,
                         0.0, 0.0, 0.0, density});
        }
        if (nc != unclipped_nc)
        {
            stage_event({WarmRainAcceptedEventKind::CloudNumberClip,
                         0.0, 0.0, 0.0,
                         (nc - unclipped_nc) / dt, 0.0,
                         0.0, density});
        }
        if (nr != unclipped_nr)
        {
            stage_event({WarmRainAcceptedEventKind::RainNumberClip,
                         0.0, 0.0, 0.0, 0.0,
                         (nr - unclipped_nr) / dt,
                         0.0, density});
        }
    }

    bool post_rain_population = rain > eps_q && nr > eps_n;
    if constexpr (CoherentTripleAutoNumber && std::is_same_v<Cell, DoubleWarmRainCell>)
    {
        // A coherent seed remains represented even below the legacy process-rate cutoffs.
        post_rain_population = rain > 0.0 && nr > 0.0;
    }
    if (post_rain_population)
    {
        rain_mean_diameter = std::cbrt(density * rain / (nr * parameters.c_r));
        if (rain_mean_diameter > 3.0e-3 && apply_post_number_regularization)
        {
            const double number_before_regularization = nr;
            const double excess = rain_mean_diameter - 3.0e-3;
            nr *= std::max(1.0 + 2.0e4 * excess * excess, std::pow(rain_mean_diameter / rain_max_diameter_m, 3.0));
            if (observer)
            {
                stage_event({WarmRainAcceptedEventKind::PostRainNumberRegularization,
                             0.0, 0.0, 0.0, 0.0,
                             (nr - number_before_regularization) / dt,
                             0.0, density});
            }
        }
        else if (rain_mean_diameter < rain_hump_diameter_m && apply_reclassification)
        {
            if constexpr (CoherentTripleAutoNumber && std::is_same_v<Cell, DoubleWarmRainCell>)
            {
                if (exact_endpoints)
                {
                    reclassification_budget = {true, rain, nr, cloud, nc};
                }
            }
            if (observer)
            {
                stage_event({WarmRainAcceptedEventKind::RainToCloud,
                             rain / dt, -rain / dt, 0.0,
                             nr / dt, -nr / dt, 0.0, density});
            }
            cloud += rain;
            nc += nr;
            rain = 0.0;
            nr   = 0.0;
        }
    }
    else
    {
        if (observer && (rain != 0.0 || nr != 0.0))
        {
            stage_event({WarmRainAcceptedEventKind::RainReset,
                         0.0, -rain / dt, 0.0, 0.0,
                         -nr / dt, 0.0, density});
        }
        rain = 0.0;
        nr   = 0.0;
    }
    if constexpr (SpectrumPolicy::dynamic)
    {
        spectrum_policy->before_evaporation(staged_events, staged_count,
                                            rain, nr / density);
    }

    // Diffusion/ventilation-limited rain evaporation. This is the
    // MY operational equation, including latent-cooling feedback.
    const double qsat               = thermodynamics::saturation_mixing_ratio_water(temperature_k, pressure);
    double       temperature_change = 0.0;
    WarmRainEvaporationBudget evaporation_budget;
    if (vapor < qsat && rain > eps_q && nr > eps_n)
    {
        const double saturation_ratio    = vapor / qsat - 1.0;
        const double tc                  = temperature_k - microphysics_constants::T0;
        const double vapor_diffusivity   = std::max(1.62e-5, 2.2157e-5 + 0.0155e-5 * tc) * 1.0e5 / pressure;
        const double dynamic_viscosity   = std::max(1.51e-5, 1.7153e-5 + 0.0050e-5 * tc);
        const double conductivity        = std::max(2.07e-2, 2.3971e-2 + 0.0078e-2 * tc);
        const double kinematic_viscosity = dynamic_viscosity / density;
        const double schmidt_third       = std::cbrt(kinematic_viscosity / vapor_diffusivity);
        double evaporated = 0.0;
        if constexpr (SpectrumPolicy::dynamic)
        {
            const double density_factor = std::sqrt(1.225 / density);
            const double ventilation = nr *
                (0.78 * spectrum_policy->diameter_moment_1_m() +
                 0.30 * schmidt_third *
                     std::sqrt(density_factor * parameters.a_r / kinematic_viscosity) *
                     spectrum_policy->ventilation_diameter_moment());
            const double resistance = microphysics_constants::L_v * microphysics_constants::L_v /
                (conductivity * microphysics_constants::R_v * temperature_k * temperature_k) +
                1.0 / (density * qsat * vapor_diffusivity);
            evaporated = std::min(rain,
                std::max(0.0, -dt / density * 2.0 * 3.14159265358979323846 *
                    saturation_ratio * ventilation / resistance));
        }
        else
        {
            i_lambda_r                       = std::cbrt(density * rain / nr * parameters.inv_rain_mass_factor);
            const double lambda_r            = 1.0 / i_lambda_r;
            const double intercept           = nr * std::pow(lambda_r, 1.0 + parameters.alpha_r) / parameters.gr31;
            const double density_factor      = std::sqrt(1.225 / density);
            const double ventilation =
                0.78 * parameters.gamma_ventilation * std::pow(i_lambda_r, 2.0 + parameters.alpha_r) +
                0.30 * schmidt_third * std::sqrt(density_factor * parameters.a_r / kinematic_viscosity) *
                    parameters.gamma_fall_ventilation *
                    std::pow(i_lambda_r, 2.5 + parameters.alpha_r + 0.5 * parameters.b_r);
            const double resistance = microphysics_constants::L_v * microphysics_constants::L_v /
                                          (conductivity * microphysics_constants::R_v * temperature_k * temperature_k) +
                                      1.0 / (density * qsat * vapor_diffusivity);
            evaporated = std::min(rain,
                                               std::max(0.0,
                                                        -dt / density * 2.0 * 3.14159265358979323846 * saturation_ratio *
                                                            intercept * ventilation / resistance));
        }
        if (evaporated > 0.0)
        {
            const double rain_before = rain;
            const double number_before_evaporation = nr;
            vapor += evaporated;
            rain -= evaporated;
            nr *= rain / rain_before;
            temperature_change -= microphysics_constants::L_v / microphysics_constants::cp * evaporated;
            if constexpr (CoherentTripleAutoNumber)
            {
                if (exact_endpoints)
                {
                    evaporation_budget = {true, rain_before, number_before_evaporation, evaporated,
                                          rain, nr, evaporated == rain_before};
                }
            }
            if (observer)
            {
                stage_event({WarmRainAcceptedEventKind::Evaporation,
                             0.0, -evaporated / dt, evaporated / dt,
                             0.0, (nr - number_before_evaporation) / dt,
                             -microphysics_constants::L_v /
                                 microphysics_constants::cp * evaporated / dt,
                             density});
            }
        }
    }

    if (observer)
    {
        if (observer->capacity > observer->events.size() ||
            observer->size > observer->capacity ||
            staged_count > observer->capacity - observer->size)
        {
            if constexpr (SpectrumPolicy::typed_resource)
            {
                throw WarmRainObserverCapacityError();
            }
            else
            {
                throw std::runtime_error("Milbrandt warm-event observer capacity is exhausted");
            }
        }
        const auto close = [dt](double initial, double rate_sum, double final_value)
        {
            const double reconstructed = initial + dt * rate_sum;
            const double scale = std::max({std::abs(initial),
                                           std::abs(reconstructed),
                                           std::abs(final_value), 1.0e-30});
            return std::isfinite(reconstructed) &&
                std::isfinite(final_value) &&
                std::abs(reconstructed - final_value) <=
                    64.0 * std::numeric_limits<double>::epsilon() * scale;
        };
        double cloud_mass_sum = 0.0;
        double rain_mass_sum = 0.0;
        double vapor_mass_sum = 0.0;
        double cloud_number_sum = 0.0;
        double rain_number_sum = 0.0;
        double temperature_sum = 0.0;
        for (std::size_t index = 0; index < staged_count; ++index)
        {
            const auto& event = staged_events[index];
            cloud_mass_sum += event.cloud_mass_rate_per_kg_s;
            rain_mass_sum += event.rain_mass_rate_per_kg_s;
            vapor_mass_sum += event.vapor_mass_rate_per_kg_s;
            cloud_number_sum += event.cloud_number_rate_per_m3_s;
            rain_number_sum += event.rain_number_rate_per_m3_s;
            temperature_sum += event.temperature_rate_k_s;
        }
        if (!close(cell.cloud, cloud_mass_sum, cloud))
        {
            throw std::runtime_error("Milbrandt warm-event cloud mass sum differs from admitted output");
        }
        if (!close(cell.rain, rain_mass_sum, rain))
        {
            throw std::runtime_error("Milbrandt warm-event rain mass sum differs from admitted output");
        }
        if (!close(cell.vapor, vapor_mass_sum, vapor))
        {
            throw std::runtime_error("Milbrandt warm-event vapor mass sum differs from admitted output");
        }
        if (!close(raw_nc, cloud_number_sum, nc))
        {
            throw std::runtime_error("Milbrandt warm-event cloud number sum differs from admitted output");
        }
        if (!close(raw_nr, rain_number_sum, nr))
        {
            throw std::runtime_error("Milbrandt warm-event rain number sum differs from admitted output");
        }
        if (!close(0.0, temperature_sum, temperature_change))
        {
            throw std::runtime_error("Milbrandt warm-event temperature sum differs from admitted output");
        }
    }

    if constexpr (CoherentTripleAutoNumber)
    {
        if (exact_endpoints)
        {
            *exact_endpoints = {vapor, cloud, rain, nc / density, nr / density, temperature_change,
                               exact_cloud_budget, exact_cloud_exhausted, evaporation_budget,
                               reclassification_budget};
        }
    }
    rates.cloud += static_cast<float>((cloud - static_cast<double>(cell.cloud)) / dt);
    rates.rain += static_cast<float>((rain - static_cast<double>(cell.rain)) / dt);
    rates.vapor += static_cast<float>((vapor - static_cast<double>(cell.vapor)) / dt);
    rates.temperature += static_cast<float>(temperature_change / dt);
    rates.cloud_number +=
        static_cast<float>((nc / density - (nc_old_mix + dt * static_cast<double>(rates.cloud_number))) / dt);
    rates.rain_number += static_cast<float>((nr / density - nr_old_mix) / dt);
    if (observer)
    {
        for (std::size_t index = 0; index < staged_count; ++index)
        {
            observer->events[observer->size + index] = staged_events[index];
        }
        observer->size += staged_count;
    }
}

inline void warm_rain_cell(
    const WarmRainParameters& parameters,
    const WarmRainCell& cell,
    double dt,
    WarmRainTendencies& rates,
    WarmRainAcceptedEventBuffer* observer = nullptr)
{
    warm_rain_cell_impl<false>(parameters, cell, dt, rates, observer);
}

template <typename Cell>
inline void validate_triple_warm_rain_source_impl(
    const WarmRainParameters& parameters,
    const Cell& cell,
    double dt,
    const WarmRainTendencies& incoming_rates)
{
    const auto supported = warm_rain_parameters(
        0.0, 523.598775598, 3.0, 149.1, 0.5);
    const std::array<double, 17> actual_parameters{
        parameters.alpha_r, parameters.c_r, parameters.d_r,
        parameters.a_r, parameters.b_r, parameters.inv_mu_c,
        parameters.gc1, parameters.gc2, parameters.gc3,
        parameters.gc4, parameters.gr31, parameters.gr34,
        parameters.gr37, parameters.inv_cloud_mass_factor,
        parameters.inv_rain_mass_factor,
        parameters.gamma_ventilation,
        parameters.gamma_fall_ventilation};
    const std::array<double, 17> supported_parameters{
        supported.alpha_r, supported.c_r, supported.d_r,
        supported.a_r, supported.b_r, supported.inv_mu_c,
        supported.gc1, supported.gc2, supported.gc3,
        supported.gc4, supported.gr31, supported.gr34,
        supported.gr37, supported.inv_cloud_mass_factor,
        supported.inv_rain_mass_factor,
        supported.gamma_ventilation,
        supported.gamma_fall_ventilation};
    for (std::size_t index = 0; index < actual_parameters.size(); ++index)
    {
        if (!std::isfinite(actual_parameters[index]) ||
            actual_parameters[index] != supported_parameters[index])
        {
            throw std::invalid_argument(
                "triple warm physical parameter law is invalid");
        }
    }
    if (!std::isfinite(dt) || dt <= 0.0 ||
        !std::isfinite(cell.temperature) || cell.temperature <= 0.0f ||
        !std::isfinite(cell.pressure) || cell.pressure <= 0.0f ||
        !std::isfinite(cell.density) || cell.density <= 0.0f)
    {
        throw std::invalid_argument(
            "triple warm cell timestep, temperature, pressure or density is invalid");
    }
    for (decltype(cell.vapor) value : {cell.vapor, cell.cloud, cell.rain,
                                      cell.cloud_number, cell.rain_number})
    {
        if (!std::isfinite(value) || value < 0.0f)
        {
            throw std::invalid_argument(
                "triple warm cell water or number input is invalid");
        }
    }
    for (float value : {incoming_rates.cloud, incoming_rates.rain,
                        incoming_rates.vapor, incoming_rates.temperature,
                        incoming_rates.cloud_number,
                        incoming_rates.rain_number})
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument(
                "triple warm cell incoming rate is nonfinite");
        }
    }
}

inline void validate_triple_warm_rain_source(
    const WarmRainParameters& parameters,
    const WarmRainCell& cell,
    double dt,
    const WarmRainTendencies& incoming_rates)
{
    validate_triple_warm_rain_source_impl(parameters, cell, dt, incoming_rates);
}

inline void validate_triple_warm_rain_source(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double dt,
    const WarmRainTendencies& incoming_rates)
{
    validate_triple_warm_rain_source_impl(parameters, cell, dt, incoming_rates);
    for (double value : {cell.temperature, cell.pressure, cell.density,
                         cell.vapor, cell.cloud, cell.rain,
                         cell.cloud_number, cell.rain_number})
    {
        if (value > std::numeric_limits<float>::max() ||
            (value > 0.0 && value < std::numeric_limits<float>::denorm_min()))
        {
            throw std::invalid_argument(
                "triple warm double source input is not FP32 representable");
        }
    }
    if (cell.temperature <= microphysics_constants::T0 || cell.temperature >= 400.0 ||
        cell.pressure < 100.0 || cell.pressure > 2.0e5)
    {
        throw std::invalid_argument(
            "triple warm double source is outside its warm thermodynamic bounds");
    }
}

inline void triple_warm_rain_cell(
    const WarmRainParameters& parameters,
    const WarmRainCell& cell,
    double dt,
    const WarmRainTendencies& incoming_rates,
    TripleWarmRainResult& result,
    std::size_t observer_capacity = 16)
{
    validate_triple_warm_rain_source(parameters, cell, dt, incoming_rates);
    TripleWarmRainResult candidate;
    candidate.rates_ = incoming_rates;
    candidate.events_.capacity = observer_capacity;
    warm_rain_cell_impl<true>(
        parameters, cell, dt, candidate.rates_, &candidate.events_);
    candidate.duration_s_ = dt;
    candidate.dry_density_kg_m3_ = cell.density;
    result = candidate;
}

inline void triple_warm_rain_cell_exact(
    const WarmRainParameters& parameters,
    const WarmRainCell& cell,
    double dt,
    const WarmRainTendencies& incoming_rates,
    TripleWarmRainExactResult& result,
    std::size_t observer_capacity = 16)
{
    validate_triple_warm_rain_source(parameters, cell, dt, incoming_rates);
    TripleWarmRainExactResult candidate;
    candidate.source_ = cell;
    candidate.parameters_ = parameters;
    candidate.incoming_ = incoming_rates;
    candidate.rates_ = incoming_rates;
    candidate.events_.capacity = observer_capacity;
    warm_rain_cell_impl<true>(parameters, cell, dt, candidate.rates_,
                             &candidate.events_, &candidate.endpoints_);
    candidate.duration_s_ = dt;
    result = candidate;
}

inline void triple_warm_rain_cell_exact_double_source(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double dt,
    const WarmRainTendencies& incoming_rates,
    TripleWarmRainDoubleSourceResult& result,
    std::size_t observer_capacity = 16)
{
    validate_triple_warm_rain_source(parameters, cell, dt, incoming_rates);
    TripleWarmRainDoubleSourceResult candidate;
    candidate.source_ = cell;
    candidate.parameters_ = parameters;
    candidate.incoming_ = incoming_rates;
    candidate.rates_ = incoming_rates;
    candidate.events_.capacity = observer_capacity;
    warm_rain_cell_impl<true>(parameters, cell, dt, candidate.rates_,
                             &candidate.events_, &candidate.endpoints_);
    candidate.duration_s_ = dt;
    result = candidate;
}
} // namespace tmv::milbrandt_detail
