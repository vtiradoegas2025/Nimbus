#include "catch2/catch.hpp"

#include "microphysics/schemes/lin/lin.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

constexpr int extent = 3;
constexpr int center = 1;
constexpr double pressure_pa = 100000.0;
constexpr double density_kgm3 = 1.2;
constexpr double step_s = 1.0;
constexpr double layer_m = 500.0;
constexpr double freezing_k = 273.15;
constexpr double latent_vaporization_jkg = 2.5e6;
constexpr double latent_fusion_jkg = 3.34e5;
constexpr double heat_capacity_jkgk = 1004.0;

struct State
{
    Field3D p{extent, extent, extent, static_cast<float>(pressure_pa)};
    Field3D theta{extent, extent, extent, 300.0f};
    Field3D rho{extent, extent, extent, static_cast<float>(density_kgm3)};
    Field3D qv{extent, extent, extent, 0.0f};
    Field3D qc{extent, extent, extent, 0.0f};
    Field3D qr{extent, extent, extent, 0.0f};
    Field3D qi{extent, extent, extent, 0.0f};
    Field3D qs{extent, extent, extent, 0.0f};
    Field3D qg{extent, extent, extent, 0.0f};
    Field3D qh{extent, extent, extent, 0.0f};
};

struct Tendencies
{
    Field3D theta;
    Field3D qv;
    Field3D qc;
    Field3D qr;
    Field3D qi;
    Field3D qs;
    Field3D qg;
    Field3D qh;
};

double saturation_mixing_ratio(double temperature_k, bool ice)
{
    const double celsius = temperature_k - freezing_k;
    const double vapor_pressure = ice
        ? 611.15 * std::exp(
            (23.036 - celsius / 333.7) * celsius / (279.82 + celsius))
        : 611.21 * std::exp(
            (18.678 - celsius / 234.5) * celsius / (257.14 + celsius));
    return 0.62198 * vapor_pressure / (pressure_pa - vapor_pressure);
}

std::array<double, 2> literal_liquid_adjustment(
    double vapor, double cloud, double temperature_k)
{
    const double qvs = saturation_mixing_ratio(temperature_k, false);
    const double gamma = latent_vaporization_jkg * latent_vaporization_jkg
        * qvs / (heat_capacity_jkgk * 461.5
                 * temperature_k * temperature_k);
    if (vapor > qvs)
    {
        const double amount = std::min(vapor, (vapor - qvs) / (1.0 + gamma));
        return {vapor - amount, cloud + amount};
    }
    const double amount = std::min(cloud, (qvs - vapor) / (1.0 + gamma));
    return {vapor + amount, cloud - amount};
}

double marshall_palmer_lambda(double mixing_ratio, double particle_density,
                              double intercept,
                              double air_density = density_kgm3)
{
    constexpr double pi = 3.14159265358979323846;
    return std::pow(
        pi * particle_density * intercept
            / (air_density * mixing_ratio),
        0.25);
}

double literal_bigg_rate(double rain, double supercooling_k,
                         double air_density)
{
    constexpr double pi = 3.14159265358979323846;
    constexpr double water_density = 1000.0;
    constexpr double rain_intercept = 8.0e6;
    const double lambda = marshall_palmer_lambda(
        rain, water_density, rain_intercept, air_density);
    return 20.0 * pi * pi * 100.0 * rain_intercept
        * (water_density / air_density)
        * (std::exp(0.66 * supercooling_k) - 1.0)
        / std::pow(lambda, 7.0);
}

double literal_snow_cloud_rate(double snow, double air_density)
{
    constexpr double pi = 3.14159265358979323846;
    constexpr double snow_density = 100.0;
    constexpr double snow_intercept = 3.0e6;
    constexpr double snow_speed_coefficient = 4.83607122;
    constexpr double gamma_3p25 = 2.5492514;
    const double lambda = marshall_palmer_lambda(
        snow, snow_density, snow_intercept, air_density);
    return (pi / 4.0) * snow_intercept * snow_speed_coefficient
        * gamma_3p25 / std::pow(lambda, 3.25);
}

double literal_fall_speed(double mixing_ratio, int species)
{
    if (species == 0)
    {
        const double lambda = marshall_palmer_lambda(
            mixing_ratio, 1000.0, 8.0e6);
        return 841.99667 * 17.837862
            / (6.0 * std::pow(lambda, 0.8));
    }
    if (species == 1)
    {
        const double lambda = marshall_palmer_lambda(
            mixing_ratio, 100.0, 3.0e6);
        return 4.83607122 * 8.2850847
            / (6.0 * std::pow(lambda, 0.25));
    }
    const double particle_density = species == 2 ? 400.0 : 913.0;
    const double intercept = species == 2 ? 4.0e5 : 4.0e4;
    const double lambda = marshall_palmer_lambda(
        mixing_ratio, particle_density, intercept);
    const double drag = std::sqrt(
        4.0 * 9.80665 * particle_density
            / (3.0 * 0.6 * density_kgm3));
    return (11.6317284 / 6.0) * drag / std::sqrt(lambda);
}

Tendencies run(LinScheme& scheme, const State& state)
{
    Tendencies result;
    scheme.set_vertical_spacing_m(layer_m);
    scheme.compute_tendencies(
        state.p, state.theta, state.rho,
        state.qv, state.qc, state.qr, state.qi,
        state.qs, state.qg, state.qh, step_s,
        result.theta, result.qv, result.qc, result.qr,
        result.qi, result.qs, result.qg, result.qh);
    return result;
}

double value(const Field3D& field)
{
    return field[center][center][center];
}

double total_water_tendency(const Tendencies& tendencies)
{
    return value(tendencies.qv) + value(tendencies.qc)
        + value(tendencies.qr) + value(tendencies.qi)
        + value(tendencies.qs) + value(tendencies.qg)
        + value(tendencies.qh);
}

LinScheme isolated(double cloud_threshold = 1.0e-3,
                   double autoconversion = 0.0,
                   double accretion = 0.0,
                   double rain_evaporation = 0.0,
                   double cloud_freezing = 0.0,
                   double ice_riming = 0.0,
                   double aggregation = 0.0,
                   double snow_melting = 0.0,
                   double graupel_melting = 0.0,
                   double hail_melting = 0.0,
                   double ice_vapor_exchange = 0.0)
{
    return LinScheme(
        cloud_threshold, autoconversion, accretion, rain_evaporation,
        cloud_freezing, ice_riming, aggregation,
        snow_melting, graupel_melting, hail_melting,
        ice_vapor_exchange);
}

void require_conservative(const Tendencies& tendencies)
{
    REQUIRE(total_water_tendency(tendencies)
            == Approx(0.0).margin(2.0e-10));
}

} // namespace

TEST_CASE("Lin warm processes match independent literal equations",
          "[qualification][microphysics][lin][reference][warm]")
{
    SECTION("saturation adjustment")
    {
        State state;
        const double qvs = saturation_mixing_ratio(300.0, false);
        state.qv.fill(static_cast<float>(qvs + 1.0e-3));
        auto scheme = isolated();
        const Tendencies tendencies = run(scheme, state);
        const double initial_vapor = state.qv[0][0][0];
        const double gamma = latent_vaporization_jkg
            * latent_vaporization_jkg * qvs
            / (heat_capacity_jkgk * 461.5 * 300.0 * 300.0);
        const double condensed = (initial_vapor - qvs) / (1.0 + gamma);
        REQUIRE(value(tendencies.qv) == Approx(-condensed).epsilon(2.0e-5));
        REQUIRE(value(tendencies.qc) == Approx(condensed).epsilon(2.0e-5));
        REQUIRE(value(tendencies.theta)
                == Approx(condensed * latent_vaporization_jkg
                          / heat_capacity_jkgk)
                    .epsilon(2.0e-5));
        require_conservative(tendencies);
    }

    SECTION("cloud evaporation")
    {
        State state;
        const double qvs = saturation_mixing_ratio(300.0, false);
        state.qv.fill(static_cast<float>(qvs - 8.0e-4));
        state.qc.fill(2.0e-3f);
        auto scheme = isolated();
        const Tendencies tendencies = run(scheme, state);
        const double initial_vapor = state.qv[0][0][0];
        const double gamma = latent_vaporization_jkg
            * latent_vaporization_jkg * qvs
            / (heat_capacity_jkgk * 461.5 * 300.0 * 300.0);
        const double evaporated = (qvs - initial_vapor) / (1.0 + gamma);
        REQUIRE(value(tendencies.qv) == Approx(evaporated).epsilon(2.0e-5));
        REQUIRE(value(tendencies.qc) == Approx(-evaporated).epsilon(2.0e-5));
        require_conservative(tendencies);
    }

    SECTION("autoconversion")
    {
        State state;
        state.qv.fill(static_cast<float>(saturation_mixing_ratio(300.0, false)));
        state.qc.fill(3.0e-3f);
        auto scheme = isolated(1.0e-3, 2.0e-3);
        const Tendencies tendencies = run(scheme, state);
        const double converted = 2.0e-3 * (state.qc[0][0][0] - 1.0e-3);
        REQUIRE(value(tendencies.qc) == Approx(-converted).epsilon(2.0e-5));
        REQUIRE(value(tendencies.qr) == Approx(converted).epsilon(2.0e-5));
        REQUIRE(value(tendencies.theta) == Approx(0.0).margin(3.0e-7));
        require_conservative(tendencies);
    }

    SECTION("accretion")
    {
        State state;
        state.qv.fill(static_cast<float>(saturation_mixing_ratio(300.0, false)));
        state.qc.fill(5.0e-4f);
        state.qr.fill(1.0e-3f);
        auto scheme = isolated(1.0e-3, 0.0, 2.2);
        const Tendencies tendencies = run(scheme, state);
        const double collected = 2.2 * state.qc[0][0][0] * state.qr[0][0][0];
        REQUIRE(value(tendencies.qc)
                == Approx(-collected).margin(2.0e-10).epsilon(2.0e-5));
        REQUIRE(value(tendencies.qr)
                == Approx(collected).margin(2.0e-10).epsilon(2.0e-5));
        require_conservative(tendencies);
    }

    SECTION("rain evaporation")
    {
        State state;
        const double qvs = saturation_mixing_ratio(300.0, false);
        state.qv.fill(static_cast<float>(0.7 * qvs));
        state.qr.fill(1.0e-3f);
        auto scheme = isolated(1.0e-3, 0.0, 0.0, 2.0e-3);
        const Tendencies tendencies = run(scheme, state);
        const double evaporated = 2.0e-3
            * (1.0 - state.qv[0][0][0] / qvs) * state.qr[0][0][0];
        REQUIRE(value(tendencies.qv) == Approx(evaporated).epsilon(2.0e-5));
        REQUIRE(value(tendencies.qr) == Approx(-evaporated).epsilon(2.0e-5));
        REQUIRE(value(tendencies.theta) < 0.0);
        require_conservative(tendencies);
    }
}

TEST_CASE("Lin cold processes match independent literal equations",
          "[qualification][microphysics][lin][reference][cold]")
{
    SECTION("homogeneous cloud freezing")
    {
        State state;
        constexpr double supercooling = 19.0;
        state.theta.fill(static_cast<float>(freezing_k - supercooling));
        state.qv.fill(static_cast<float>(
            saturation_mixing_ratio(freezing_k - supercooling, false)));
        state.qc.fill(8.0e-4f);
        auto scheme = isolated(1.0e-3, 0.0, 0.0, 0.0, 2.0e-3);
        const Tendencies tendencies = run(scheme, state);
        const double actual_temperature = state.theta[0][0][0];
        const auto adjusted = literal_liquid_adjustment(
            state.qv[0][0][0], state.qc[0][0][0], actual_temperature);
        const double liquid_adjustment = adjusted[1] - state.qc[0][0][0];
        const double frozen = 2.0e-3
            * ((freezing_k - actual_temperature) / 38.0) * adjusted[1];
        REQUIRE(value(tendencies.qc)
                == Approx(liquid_adjustment - frozen)
                    .margin(2.0e-10).epsilon(3.0e-5));
        REQUIRE(value(tendencies.qi)
                == Approx(frozen).margin(2.0e-10).epsilon(3.0e-5));
        const double condensed = state.qv[0][0][0] - adjusted[0];
        REQUIRE(value(tendencies.theta)
                == Approx((condensed * latent_vaporization_jkg
                           + frozen * latent_fusion_jkg)
                          / heat_capacity_jkgk)
                    .epsilon(3.0e-5));
        require_conservative(tendencies);
    }

    SECTION("Bigg rain freezing")
    {
        State state;
        constexpr double supercooling = 15.0;
        state.theta.fill(static_cast<float>(freezing_k - supercooling));
        state.qv.fill(static_cast<float>(
            saturation_mixing_ratio(freezing_k - supercooling, false)));
        state.qr.fill(1.0e-3f);
        auto scheme = isolated();
        const Tendencies tendencies = run(scheme, state);
        const double actual_temperature = state.theta[0][0][0];
        const double rho_air = pressure_pa / (287.0 * actual_temperature);
        const double frozen = literal_bigg_rate(
            state.qr[0][0][0], freezing_k - actual_temperature, rho_air);
        REQUIRE(value(tendencies.qr) == Approx(-frozen).epsilon(3.0e-5));
        REQUIRE(value(tendencies.qh) == Approx(frozen).epsilon(3.0e-5));
        require_conservative(tendencies);
    }

    SECTION("vapor deposition")
    {
        State state;
        constexpr double temperature = freezing_k - 10.0;
        const double qvs_water = saturation_mixing_ratio(temperature, false);
        const double qvs_ice = saturation_mixing_ratio(temperature, true);
        state.theta.fill(static_cast<float>(temperature));
        state.qv.fill(static_cast<float>(0.5 * (qvs_water + qvs_ice)));
        state.qi.fill(1.0e-4f);
        auto scheme = isolated(
            1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 2.0e-3);
        const Tendencies tendencies = run(scheme, state);
        const double deposited = 2.0e-3 * (state.qv[0][0][0] - qvs_ice);
        REQUIRE(value(tendencies.qv) == Approx(-deposited).epsilon(3.0e-5));
        REQUIRE(value(tendencies.qi) == Approx(deposited).epsilon(3.0e-5));
        REQUIRE(value(tendencies.theta) > 0.0);
        require_conservative(tendencies);
    }

    SECTION("ice riming")
    {
        State state;
        constexpr double temperature = freezing_k - 10.0;
        state.theta.fill(static_cast<float>(temperature));
        state.qv.fill(static_cast<float>(
            saturation_mixing_ratio(temperature, false)));
        state.qc.fill(5.0e-4f);
        state.qi.fill(4.0e-4f);
        auto scheme = isolated(
            1.0e-3, 0.0, 0.0, 0.0, 0.0, 1.5);
        const Tendencies tendencies = run(scheme, state);
        const double actual_temperature = state.theta[0][0][0];
        const auto adjusted = literal_liquid_adjustment(
            state.qv[0][0][0], state.qc[0][0][0], actual_temperature);
        const double liquid_adjustment = adjusted[1] - state.qc[0][0][0];
        const double rimed = 1.5 * adjusted[1] * state.qi[0][0][0];
        REQUIRE(value(tendencies.qc)
                == Approx(liquid_adjustment - rimed)
                    .margin(2.0e-10).epsilon(3.0e-5));
        REQUIRE(value(tendencies.qg) == Approx(rimed).epsilon(3.0e-5));
        require_conservative(tendencies);
    }

    SECTION("snow collection of cloud")
    {
        State state;
        constexpr double temperature = freezing_k - 10.0;
        state.theta.fill(static_cast<float>(temperature));
        state.qv.fill(static_cast<float>(
            saturation_mixing_ratio(temperature, false)));
        state.qc.fill(5.0e-4f);
        state.qs.fill(1.0e-3f);
        auto scheme = isolated();
        const Tendencies tendencies = run(scheme, state);
        const double actual_temperature = state.theta[0][0][0];
        const auto adjusted = literal_liquid_adjustment(
            state.qv[0][0][0], state.qc[0][0][0], actual_temperature);
        const double liquid_adjustment = adjusted[1] - state.qc[0][0][0];
        const double rho_air = pressure_pa / (287.0 * actual_temperature);
        const double collection_rate = literal_snow_cloud_rate(
            state.qs[0][0][0], rho_air);
        const double rimed = collection_rate * adjusted[1];
        REQUIRE(value(tendencies.qc)
                == Approx(liquid_adjustment - rimed)
                    .margin(2.0e-10).epsilon(3.0e-5));
        REQUIRE(value(tendencies.qs) == Approx(rimed).epsilon(3.0e-5));
        require_conservative(tendencies);
    }

    SECTION("ice aggregation")
    {
        State state;
        constexpr double temperature = freezing_k - 20.0;
        state.theta.fill(static_cast<float>(temperature));
        state.qv.fill(static_cast<float>(
            saturation_mixing_ratio(temperature, true)));
        state.qi.fill(2.0e-3f);
        auto scheme = isolated(
            1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0e-3);
        const Tendencies tendencies = run(scheme, state);
        const double aggregated = 2.0e-3
            * (state.qi[0][0][0] - 1.0e-3)
            * std::exp(0.025 * (temperature - freezing_k));
        REQUIRE(value(tendencies.qi) == Approx(-aggregated).epsilon(3.0e-5));
        REQUIRE(value(tendencies.qs) == Approx(aggregated).epsilon(3.0e-5));
        REQUIRE(value(tendencies.theta) == 0.0);
        require_conservative(tendencies);
    }

    SECTION("ice and snow sublimation")
    {
        State state;
        constexpr double temperature = freezing_k - 15.0;
        const double qvs_ice = saturation_mixing_ratio(temperature, true);
        state.theta.fill(static_cast<float>(temperature));
        state.qv.fill(static_cast<float>(0.6 * qvs_ice));
        state.qi.fill(5.0e-4f);
        state.qs.fill(7.0e-4f);
        auto scheme = isolated(
            1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 2.0e-3);
        const Tendencies tendencies = run(scheme, state);
        const double rh = state.qv[0][0][0] / qvs_ice;
        const double ice_loss = 2.0e-3 * (1.0 - rh) * state.qi[0][0][0];
        const double snow_loss = 2.0e-3 * (1.0 - rh) * state.qs[0][0][0];
        REQUIRE(value(tendencies.qi) == Approx(-ice_loss).epsilon(3.0e-5));
        REQUIRE(value(tendencies.qs) == Approx(-snow_loss).epsilon(3.0e-5));
        REQUIRE(value(tendencies.qv)
                == Approx(ice_loss + snow_loss).epsilon(3.0e-5));
        REQUIRE(value(tendencies.theta) < 0.0);
        require_conservative(tendencies);
    }
}

TEST_CASE("Lin melting and sedimentation match independent literal equations",
          "[qualification][microphysics][lin][reference][transport]")
{
    SECTION("species-specific melting")
    {
        State state;
        constexpr double warm_excess = 2.0;
        state.theta.fill(static_cast<float>(freezing_k + warm_excess));
        state.qv.fill(static_cast<float>(
            saturation_mixing_ratio(freezing_k + warm_excess, false)));
        state.qs.fill(1.0e-3f);
        state.qg.fill(8.0e-4f);
        state.qh.fill(6.0e-4f);
        auto scheme = isolated(
            1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            4.3e-3, 4.3e-4, 7.7e-5);
        const Tendencies tendencies = run(scheme, state);
        const double snow = 4.3e-3 * warm_excess * state.qs[0][0][0];
        const double graupel = 4.3e-4 * warm_excess * state.qg[0][0][0];
        const double hail = 7.7e-5 * warm_excess * state.qh[0][0][0];
        REQUIRE(value(tendencies.qs) == Approx(-snow).epsilon(3.0e-5));
        REQUIRE(value(tendencies.qg) == Approx(-graupel).epsilon(3.0e-5));
        REQUIRE(value(tendencies.qh) == Approx(-hail).epsilon(3.0e-5));
        REQUIRE(value(tendencies.qr)
                == Approx(snow + graupel + hail).epsilon(3.0e-5));
        require_conservative(tendencies);
    }

    SECTION("surface fallout")
    {
        State state;
        state.qv.fill(static_cast<float>(saturation_mixing_ratio(300.0, false)));
        const std::array<double, 4> amounts{{1.0e-3, 8.0e-4, 6.0e-4, 4.0e-4}};
        state.qr[center][center][0] = static_cast<float>(amounts[0]);
        state.qs[center][center][0] = static_cast<float>(amounts[1]);
        state.qg[center][center][0] = static_cast<float>(amounts[2]);
        state.qh[center][center][0] = static_cast<float>(amounts[3]);
        auto scheme = isolated();
        const Tendencies tendencies = run(scheme, state);
        const std::array<const Field3D*, 4> fields{{
            &tendencies.qr, &tendencies.qs,
            &tendencies.qg, &tendencies.qh}};
        double expected_sink = 0.0;
        for (int species = 0; species < 4; ++species)
        {
            const double speed = literal_fall_speed(amounts[species], species);
            const double expected = -speed * amounts[species] / layer_m;
            REQUIRE((*fields[species])[center][center][0]
                    == Approx(expected).margin(2.0e-9).epsilon(4.0e-5));
            expected_sink -= expected;
        }
        REQUIRE(scheme.bottom_sedimentation_sink_rate(
                    state.rho, state.qr, state.qs,
                    state.qg, state.qh, center, center, step_s)
                == Approx(expected_sink).margin(3.0e-9).epsilon(4.0e-5));
    }
}

TEST_CASE("Lin process chain remains positive conservative deterministic and restart portable",
          "[qualification][microphysics][lin][contract]")
{
    State state;
    for (int i = 0; i < extent; ++i)
        for (int j = 0; j < extent; ++j)
            for (int k = 0; k < extent; ++k)
            {
                const float pattern = static_cast<float>(1 + i + 3 * j + 5 * k);
                state.p[i][j][k] = 78000.0f + 80.0f * pattern;
                state.theta[i][j][k] = 250.0f + 4.0f * k + 0.1f * pattern;
                state.rho[i][j][k] = 0.75f + 0.01f * pattern;
                state.qv[i][j][k] = 8.0e-4f + 2.0e-4f * pattern;
                state.qc[i][j][k] = 8.0e-5f * static_cast<float>((i + j + k) % 4);
                state.qr[i][j][k] = 6.0e-5f * static_cast<float>((2 * i + j + k) % 4);
                state.qi[i][j][k] = 5.0e-5f * static_cast<float>((i + 2 * j + k) % 4);
                state.qs[i][j][k] = 4.0e-5f * static_cast<float>((i + j + 2 * k) % 4);
                state.qg[i][j][k] = 3.0e-5f * static_cast<float>((i + 3 * j + k) % 4);
                state.qh[i][j][k] = 2.0e-5f * static_cast<float>((3 * i + j + k) % 4);
            }

    LinScheme original;
    REQUIRE(original.requires_runtime_column_donor_admission());
    const Tendencies first = run(original, state);
    const std::array<const Field3D*, 7> initial{{
        &state.qv, &state.qc, &state.qr, &state.qi,
        &state.qs, &state.qg, &state.qh}};
    const std::array<const Field3D*, 7> rates{{
        &first.qv, &first.qc, &first.qr, &first.qi,
        &first.qs, &first.qg, &first.qh}};
    double admitted_factor = 1.0;
    for (std::size_t species = 0; species < initial.size(); ++species)
        for (std::size_t n = 0; n < initial[species]->size(); ++n)
        {
            REQUIRE(std::isfinite(rates[species]->data()[n]));
            const double rate = rates[species]->data()[n];
            if (rate < 0.0)
                admitted_factor = std::min(
                    admitted_factor,
                    static_cast<double>(initial[species]->data()[n])
                        / (-step_s * rate));
        }
    REQUIRE(admitted_factor > 0.0);
    REQUIRE(admitted_factor <= 1.0);
    for (std::size_t species = 0; species < initial.size(); ++species)
        for (std::size_t n = 0; n < initial[species]->size(); ++n)
            REQUIRE(static_cast<double>(initial[species]->data()[n])
                        + step_s * admitted_factor
                            * rates[species]->data()[n]
                    >= -2.0e-10);

    const PortableSchemeState saved = original.portable_state();
    LinScheme restored;
    std::string error;
    REQUIRE(restored.restore_portable_state(saved, error));
    REQUIRE(error.empty());
    const Tendencies replay = run(restored, state);
    const std::array<const Field3D*, 8> first_fields{{
        &first.theta, &first.qv, &first.qc, &first.qr,
        &first.qi, &first.qs, &first.qg, &first.qh}};
    const std::array<const Field3D*, 8> replay_fields{{
        &replay.theta, &replay.qv, &replay.qc, &replay.qr,
        &replay.qi, &replay.qs, &replay.qg, &replay.qh}};
    for (std::size_t field = 0; field < first_fields.size(); ++field)
        for (std::size_t n = 0; n < first_fields[field]->size(); ++n)
            REQUIRE(first_fields[field]->data()[n]
                    == replay_fields[field]->data()[n]);

#ifdef _OPENMP
    const int previous_dynamic = omp_get_dynamic();
    const int previous_threads = omp_get_max_threads();
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    LinScheme single_thread_scheme;
    const Tendencies single_thread = run(single_thread_scheme, state);
    omp_set_num_threads(4);
    LinScheme four_thread_scheme;
    const Tendencies four_thread = run(four_thread_scheme, state);
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
    const std::array<const Field3D*, 8> single_thread_fields{{
        &single_thread.theta, &single_thread.qv, &single_thread.qc,
        &single_thread.qr, &single_thread.qi, &single_thread.qs,
        &single_thread.qg, &single_thread.qh}};
    const std::array<const Field3D*, 8> four_thread_fields{{
        &four_thread.theta, &four_thread.qv, &four_thread.qc,
        &four_thread.qr, &four_thread.qi, &four_thread.qs,
        &four_thread.qg, &four_thread.qh}};
    for (std::size_t field = 0; field < single_thread_fields.size(); ++field)
        for (std::size_t n = 0; n < single_thread_fields[field]->size(); ++n)
            REQUIRE(single_thread_fields[field]->data()[n]
                    == four_thread_fields[field]->data()[n]);
#endif

    PortableSchemeState malformed = saved;
    malformed.reals[3] = std::numeric_limits<double>::quiet_NaN();
    const PortableSchemeState before = restored.portable_state();
    REQUIRE_FALSE(restored.restore_portable_state(malformed, error));
    REQUIRE(restored.portable_state().reals == before.reals);
    REQUIRE(restored.portable_state().integers == before.integers);
    REQUIRE(restored.portable_state().fields.size() == before.fields.size());
}
