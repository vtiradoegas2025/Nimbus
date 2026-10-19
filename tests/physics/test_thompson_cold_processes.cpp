#include "catch2/catch.hpp"
#include "microphysics/base/thermodynamics.hpp"
#include "microphysics/schemes/thompson/cold_processes.hpp"
#include "microphysics/schemes/thompson/freezing_factors.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

void require_close(double actual, double expected)
{
    REQUIRE(actual == Approx(expected).epsilon(4.0e-13));
}

tmv::thompson::RainFreezingState reference_state()
{
    constexpr double pi = 3.1415926536;
    constexpr double water_mass_coefficient = pi * 1000.0 / 6.0;
    constexpr double rain_content = 1.0e-4;
    constexpr double exponential_intercept = 1.0e8;
    const double lambda = std::pow(
        6.0 * water_mass_coefficient * exponential_intercept
            / rain_content,
        0.25);
    const double rain_number = rain_content * lambda * lambda * lambda
        / (6.0 * water_mass_coefficient);
    return {258.15, rain_content, rain_number, 1.0e3};
}

tmv::thompson::CloudFreezingState cloud_reference_state()
{
    const double cloud_number = std::exp(
        49.5 / 100.0 * std::log(3000.0)) * 1.0e6;
    return {258.15, 1.0e-4, cloud_number, 1.0e3};
}

tmv::thompson::ColdProcessTables ice_reference_tables()
{
    tmv::thompson::ColdProcessTables tables;
    tables.ice_snow.resize(tmv::thompson::checked_ice_snow_size());
    tables.ice_snow[tmv::thompson::flatten(
        tmv::thompson::IceSnowIndex{36, 27})] =
        tmv::thompson::generate_ice_snow_cell({36, 27});
    return tables;
}

tmv::thompson::IceDepositionState ice_reference_state(double saturation)
{
    const double qvsi = tmv::thermodynamics::saturation_mixing_ratio_ice(
        258.15f, 80000.0f);
    return {80000.0, 258.15, 1.0, saturation * qvsi, 1.0e-6, 1.0e3};
}

} // namespace

TEST_CASE("Thompson rain-freezing lookup reproduces the WRF mp=28 coordinates",
          "[microphysics][thompson][cold][freezing][reference]")
{
    const auto index = tmv::thompson::rain_freezing_lookup_index(
        reference_state());
    REQUIRE(index.rain_content == 18);
    REQUIRE(index.rain_intercept == 18);
    REQUIRE(index.temperature == 14);
    REQUIRE(index.ice_nuclei == 27);
}

TEST_CASE("Thompson cloud-freezing lookup reproduces the WRF mp=28 coordinates",
          "[microphysics][thompson][cold][freezing][reference]")
{
    const auto index = tmv::thompson::cloud_freezing_lookup_index(
        cloud_reference_state());
    REQUIRE(index.cloud_content == 18);
    REQUIRE(index.cloud_number == 49);
    REQUIRE(index.temperature == 14);
    REQUIRE(index.ice_nuclei == 27);
}

TEST_CASE("Thompson Bigg cloud freezing matches the pinned WRF table exchange",
          "[microphysics][thompson][cold][freezing][reference]")
{
    using namespace tmv::thompson;
    const auto state = cloud_reference_state();
    const auto cell = generate_cloud_freezing_cell({18, 49, 14, 27});
    const auto exchange = cloud_freezing_exchange(cell, state);

    require_close(exchange.cloud_kg_m3, -6.3181239101899094e-14);
    require_close(exchange.ice_kg_m3, 6.3181239101899094e-14);
    require_close(exchange.cloud_number_m3, -2.0398661219860986e-2);
    require_close(exchange.ice_number_m3, 2.0398661219860986e-2);
    REQUIRE(exchange.cloud_kg_m3 + exchange.ice_kg_m3
            == Approx(0.0).margin(1.0e-28));
    REQUIRE(exchange.cloud_number_m3 + exchange.ice_number_m3
            == Approx(0.0).margin(1.0e-18));
}

TEST_CASE("Thompson cloud freezing is donor bounded and failure atomic",
          "[microphysics][thompson][cold][freezing][atomic]")
{
    using namespace tmv::thompson;
    const CloudFreezingState limited{250.0, 0.25, 4.0, 1000.0};
    const CloudFreezingCell excessive{2.0, 30.0};
    CloudFreezingExchange destination = cloud_freezing_exchange(
        excessive, limited);
    REQUIRE(destination.cloud_kg_m3 == -0.25);
    REQUIRE(destination.ice_kg_m3 == 0.25);
    REQUIRE(-destination.cloud_number_m3 <= limited.cloud_number_m3);
    REQUIRE(destination.cloud_number_m3 + destination.ice_number_m3
            == Approx(0.0).margin(1.0e-15));
    const CloudFreezingExchange before = destination;

    CloudFreezingCell corrupt = excessive;
    corrupt.ice_number = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS_AS(
        destination = cloud_freezing_exchange(corrupt, limited),
        std::invalid_argument);
    REQUIRE(std::memcmp(&destination, &before, sizeof(destination)) == 0);
}

TEST_CASE("Thompson Bigg rain freezing matches the pinned WRF table exchange",
          "[microphysics][thompson][cold][freezing][reference]")
{
    using namespace tmv::thompson;
    const auto state = reference_state();
    const auto cell = generate_rain_freezing_cell({18, 18, 14, 27});
    const auto exchange = rain_freezing_exchange(cell, state);

    require_close(exchange.rain_kg_m3, -8.4849088670790473e-11);
    require_close(exchange.ice_kg_m3, 8.6387778901638637e-12);
    require_close(exchange.graupel_kg_m3, 7.6210310780626602e-11);
    require_close(exchange.rain_number_m3, -3.3823735203584960e-3);
    require_close(exchange.ice_number_m3, 1.8509222757646616e-3);
    require_close(exchange.graupel_number_m3, 1.5314512445938346e-3);
    REQUIRE(exchange.rain_kg_m3 + exchange.ice_kg_m3
                + exchange.graupel_kg_m3
            == Approx(0.0).margin(2.0e-26));
    REQUIRE(exchange.rain_number_m3 + exchange.ice_number_m3
                + exchange.graupel_number_m3
            == Approx(0.0).margin(2.0e-18));
}

TEST_CASE("Thompson ice deposition and sublimation match pinned WRF rates",
          "[microphysics][thompson][cold][deposition][reference]")
{
    using namespace tmv::thompson;
    const auto tables = ice_reference_tables();
    const auto deposition = ice_deposition_rates(
        tables, ice_reference_state(1.2), 2.0);
    require_close(deposition.vapor_kg_m3_s, -2.0437152320611431e-9);
    require_close(deposition.ice_kg_m3_s, 1.8877132511879696e-9);
    require_close(deposition.snow_kg_m3_s, 1.5600198087317360e-10);
    REQUIRE(deposition.ice_number_m3_s == 0.0);
    require_close(deposition.temperature_k_s, 5.7688137128100391e-6);
    REQUIRE(deposition.vapor_kg_m3_s + deposition.ice_kg_m3_s
                + deposition.snow_kg_m3_s
            == Approx(0.0).margin(2.0e-24));
    REQUIRE(deposition.temperature_k_s
            == Approx(-2.834e6 / 1004.0 * deposition.vapor_kg_m3_s)
                .epsilon(3.0e-13));

    const auto sublimation = ice_deposition_rates(
        tables, ice_reference_state(0.8), 2.0);
    require_close(sublimation.vapor_kg_m3_s, 2.0676840358737479e-9);
    require_close(sublimation.ice_kg_m3_s, -2.0676840358737479e-9);
    REQUIRE(sublimation.snow_kg_m3_s == 0.0);
    require_close(sublimation.ice_number_m3_s, -1.9384537836316382e-1);
    require_close(sublimation.temperature_k_s, -5.8364706749663359e-6);
}

TEST_CASE("Thompson snow and diagnostic graupel sublimation match pinned WRF rates",
          "[microphysics][thompson][cold][deposition][reference]")
{
    auto tables = ice_reference_tables();
    const double qvsi = tmv::thermodynamics::saturation_mixing_ratio_ice(
        258.15f, 80000.0f);
    const auto exchange = tmv::thompson::ice_deposition_rates(
        tables,
        {80000.0, 258.15, 1.0, 0.8 * qvsi,
         0.0, 0.0, 5.0e-4, 5.0e-4},
        2.0);
    require_close(exchange.vapor_kg_m3_s, 4.5941664812664459e-7);
    REQUIRE(exchange.ice_kg_m3_s == 0.0);
    require_close(exchange.snow_kg_m3_s, -4.3753259601558543e-7);
    require_close(exchange.graupel_kg_m3_s, -2.1884052111059148e-8);
    REQUIRE(exchange.ice_number_m3_s == 0.0);
    require_close(exchange.temperature_k_s, -1.2967995824610665e-3);
    REQUIRE(exchange.vapor_kg_m3_s + exchange.snow_kg_m3_s
                + exchange.graupel_kg_m3_s
            == Approx(0.0).margin(2.0e-22));

    const auto bounded = tmv::thompson::ice_deposition_rates(
        tables,
        {80000.0, 258.15, 1.0, 0.0,
         0.0, 0.0, 5.0e-4, 5.0e-4},
        1.0e9);
    REQUIRE(bounded.snow_kg_m3_s >= -5.0e-13);
    REQUIRE(bounded.graupel_kg_m3_s >= -5.0e-13);
}

TEST_CASE("Thompson ice deposition is donor bounded and failure atomic",
          "[microphysics][thompson][cold][deposition][atomic]")
{
    using namespace tmv::thompson;
    auto tables = ice_reference_tables();
    const auto sublimation = ice_deposition_rates(
        tables, ice_reference_state(0.0), 1.0e6);
    REQUIRE(sublimation.ice_kg_m3_s >= -1.0e-12);
    REQUIRE(sublimation.ice_number_m3_s >= -1.0e-3);
    IceDepositionExchange destination = ice_deposition_rates(
        tables, ice_reference_state(1.2), 2.0);
    const IceDepositionExchange before = destination;
    tables.ice_snow[flatten(IceSnowIndex{36, 27})]
        .ice_deposition_fraction =
        std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(
        destination = ice_deposition_rates(
            tables, ice_reference_state(1.2), 2.0),
        std::invalid_argument);
    REQUIRE(std::memcmp(&destination, &before, sizeof(destination)) == 0);
}

TEST_CASE("Thompson snow riming matches pinned WRF collection rates",
          "[microphysics][thompson][cold][riming][reference]")
{
    const auto exchange = tmv::thompson::snow_riming_rates(
        {258.15, 1.0, 2.0e-4, 2.0e8, 5.0e-4}, 2.0);
    require_close(exchange.cloud_kg_m3_s, -5.6376290105608364e-7);
    require_close(exchange.snow_kg_m3_s, 5.6376290105608364e-7);
    require_close(exchange.cloud_number_m3_s, -5.6376290105608362e5);
    require_close(exchange.temperature_k_s, 1.8754662246288042e-4);
    REQUIRE(exchange.cloud_kg_m3_s + exchange.snow_kg_m3_s
            == Approx(0.0).margin(1.0e-21));
    REQUIRE(exchange.temperature_k_s
            == Approx(3.34e5 / 1004.0 * exchange.snow_kg_m3_s)
                .epsilon(3.0e-13));
}

TEST_CASE("Thompson diagnostic mp=28 graupel riming matches pinned WRF rates",
          "[microphysics][thompson][cold][riming][reference]")
{
    const auto exchange = tmv::thompson::snow_riming_rates(
        {258.15, 1.0, 2.0e-4, 2.0e8, 0.0, 5.0e-4}, 2.0);
    require_close(exchange.cloud_kg_m3_s, -8.1233025080078053e-8);
    REQUIRE(exchange.snow_kg_m3_s == 0.0);
    require_close(exchange.graupel_kg_m3_s, 8.1233025080078053e-8);
    require_close(exchange.cloud_number_m3_s, -8.1233025080078049e4);
    require_close(exchange.temperature_k_s, 2.7023735435006044e-5);
    REQUIRE(exchange.cloud_kg_m3_s + exchange.graupel_kg_m3_s
            == Approx(0.0).margin(1.0e-22));
}

TEST_CASE("Thompson rimed-snow conversion and Hallett-Mossop match WRF rates",
          "[microphysics][thompson][cold][riming][secondary][reference]")
{
    const auto exchange = tmv::thompson::snow_riming_rates(
        {268.15, 1.0, 2.0e-4, 2.0e8,
         5.0e-4, 5.0e-4, 1.0e-8},
        2.0);
    require_close(exchange.cloud_kg_m3_s, -8.6523969271119415e-7);
    require_close(exchange.snow_kg_m3_s, 5.1870345506264507e-8);
    require_close(exchange.graupel_kg_m3_s, 8.1334171648392160e-7);
    require_close(exchange.ice_kg_m3_s, 2.7630721008040504e-11);
    require_close(exchange.ice_number_m3_s, 2.7630721008040506e1);
    require_close(exchange.cloud_number_m3_s, -8.6523969271119416e5);
    require_close(exchange.temperature_k_s, 2.8783870255531758e-4);
    REQUIRE(exchange.cloud_kg_m3_s + exchange.snow_kg_m3_s
                + exchange.graupel_kg_m3_s + exchange.ice_kg_m3_s
            == Approx(0.0).margin(2.0e-22));
}

TEST_CASE("Thompson ice-to-snow conversion matches pinned WRF table rates",
          "[microphysics][thompson][cold][ice-snow][reference]")
{
    const auto tables = ice_reference_tables();
    const auto exchange = tmv::thompson::ice_to_snow_rates(
        tables, {1.0e-6, 1.0e3}, 2.0);
    require_close(exchange.ice_kg_m3_s, -1.7897691377250469e-7);
    require_close(exchange.snow_kg_m3_s, 1.7897691377250469e-7);
    require_close(exchange.ice_number_m3_s, -7.1298454535289899);
    REQUIRE(exchange.ice_kg_m3_s + exchange.snow_kg_m3_s
            == Approx(0.0).margin(1.0e-22));
}

TEST_CASE("Thompson ice-to-snow conversion is donor bounded and atomic",
          "[microphysics][thompson][cold][ice-snow][atomic]")
{
    using namespace tmv::thompson;
    auto tables = ice_reference_tables();
    const IceSnowConversionState state{1.0e-6, 1.0e3};
    const auto bounded = ice_to_snow_rates(tables, state, 1.0e-9);
    REQUIRE(-bounded.ice_kg_m3_s <= 0.99 * state.ice_kg_m3 / 1.0e-9);
    REQUIRE(-bounded.ice_number_m3_s
            <= 0.95 * state.ice_number_m3 / 1.0e-9);
    IceSnowConversionExchange destination = ice_to_snow_rates(
        tables, state, 2.0);
    const IceSnowConversionExchange before = destination;
    tables.ice_snow[flatten(IceSnowIndex{36, 27})].snow_mass =
        std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(
        destination = ice_to_snow_rates(tables, state, 2.0),
        std::invalid_argument);
    REQUIRE(std::memcmp(&destination, &before, sizeof(destination)) == 0);
}

TEST_CASE("Thompson snow riming is donor bounded and failure atomic",
          "[microphysics][thompson][cold][riming][atomic]")
{
    using namespace tmv::thompson;
    const SnowRimingState limited{258.15, 1.0, 2.0e-4, 2.0e8, 5.0e-4};
    const auto bounded = snow_riming_rates(limited, 1.0e9);
    REQUIRE(-bounded.cloud_kg_m3_s <= limited.cloud_kg_m3 / 1.0e9);
    REQUIRE(-bounded.cloud_number_m3_s
            <= limited.cloud_number_m3 / 1.0e9);
    SnowRimingExchange destination = snow_riming_rates(limited, 2.0);
    const SnowRimingExchange before = destination;
    SnowRimingState invalid = limited;
    invalid.snow_kg_m3 = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(
        destination = snow_riming_rates(invalid, 2.0),
        std::invalid_argument);
    REQUIRE(std::memcmp(&destination, &before, sizeof(destination)) == 0);
}

TEST_CASE("Thompson rain freezing bounds donors and preserves destination on failure",
          "[microphysics][thompson][cold][freezing][atomic]")
{
    using namespace tmv::thompson;
    const RainFreezingCell excessive{2.0, 3.0, 20.0, 30.0};
    const RainFreezingState limited{250.0, 0.5, 4.0, 1000.0};
    RainFreezingExchange destination = rain_freezing_exchange(
        excessive, limited);
    REQUIRE(-destination.rain_kg_m3 <= limited.rain_kg_m3);
    REQUIRE(-destination.rain_number_m3 <= limited.rain_number_m3);
    REQUIRE(destination.rain_kg_m3 + destination.ice_kg_m3
                + destination.graupel_kg_m3
            == Approx(0.0).margin(1.0e-15));
    const RainFreezingExchange before = destination;

    RainFreezingCell corrupt = excessive;
    corrupt.ice_mass = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(
        destination = rain_freezing_exchange(corrupt, limited),
        std::invalid_argument);
    REQUIRE(std::memcmp(&destination, &before, sizeof(destination)) == 0);

    const auto homogeneous = rain_freezing_exchange(
        RainFreezingCell{}, {230.0, 5.0e-7, 40.0, 1000.0});
    REQUIRE(homogeneous.rain_kg_m3 == -5.0e-7);
    REQUIRE(homogeneous.ice_kg_m3 == 5.0e-7);
    REQUIRE(homogeneous.rain_number_m3 == -40.0);
    REQUIRE(homogeneous.ice_number_m3 == 40.0);
}

TEST_CASE("Thompson rain freezing is bitwise identical at one and four threads",
          "[microphysics][thompson][cold][freezing][threads]")
{
    using namespace tmv::thompson;
    constexpr std::size_t count = 128;
    const auto cell = generate_rain_freezing_cell({18, 18, 14, 27});
    std::array<RainFreezingState, count> states{};
    for (std::size_t index = 0; index < count; ++index)
    {
        states[index] = reference_state();
        states[index].rain_kg_m3 *= 1.0 + 1.0e-3 * index;
        states[index].rain_number_m3 *= 1.0 + 2.0e-3 * index;
    }
    const auto evaluate = [&](int threads) {
        std::array<RainFreezingExchange, count> exchanges{};
#ifdef _OPENMP
        omp_set_num_threads(threads);
#else
        static_cast<void>(threads);
#endif
#pragma omp parallel for schedule(static)
        for (std::size_t index = 0; index < count; ++index)
            exchanges[index] = rain_freezing_exchange(cell, states[index]);
        return exchanges;
    };
    const auto one = evaluate(1);
    const auto four = evaluate(4);
    REQUIRE(std::memcmp(one.data(), four.data(), sizeof(one)) == 0);
}

TEST_CASE("Thompson cloud freezing is bitwise identical at one and four threads",
          "[microphysics][thompson][cold][freezing][threads]")
{
    using namespace tmv::thompson;
    constexpr std::size_t count = 128;
    const auto cell = generate_cloud_freezing_cell({18, 49, 14, 27});
    std::array<CloudFreezingState, count> states{};
    for (std::size_t index = 0; index < count; ++index)
    {
        states[index] = cloud_reference_state();
        states[index].cloud_kg_m3 *= 1.0 + 1.0e-3 * index;
        states[index].cloud_number_m3 *= 1.0 + 2.0e-3 * index;
    }
    const auto evaluate = [&](int threads) {
        std::array<CloudFreezingExchange, count> exchanges{};
#ifdef _OPENMP
        omp_set_num_threads(threads);
#else
        static_cast<void>(threads);
#endif
#pragma omp parallel for schedule(static)
        for (std::size_t index = 0; index < count; ++index)
            exchanges[index] = cloud_freezing_exchange(cell, states[index]);
        return exchanges;
    };
    const auto one = evaluate(1);
    const auto four = evaluate(4);
    REQUIRE(std::memcmp(one.data(), four.data(), sizeof(one)) == 0);
}

TEST_CASE("Thompson ice deposition is bitwise identical at one and four threads",
          "[microphysics][thompson][cold][deposition][threads]")
{
    using namespace tmv::thompson;
    constexpr std::size_t count = 128;
    const auto tables = ice_reference_tables();
    std::array<IceDepositionState, count> states{};
    for (std::size_t index = 0; index < count; ++index)
    {
        states[index] = ice_reference_state(
            index % 2 == 0 ? 1.2 : 0.8);
        states[index].ice_kg_m3 *= 1.0 + 1.0e-4 * index;
        states[index].ice_number_m3 *= 1.0 + 1.0e-4 * index;
        if (index % 2 != 0)
        {
            states[index].snow_kg_m3 = 5.0e-4;
            states[index].graupel_kg_m3 = 5.0e-4;
        }
    }
    const auto evaluate = [&](int threads) {
        std::array<IceDepositionExchange, count> exchanges{};
#ifdef _OPENMP
        omp_set_num_threads(threads);
#else
        static_cast<void>(threads);
#endif
#pragma omp parallel for schedule(static)
        for (std::size_t index = 0; index < count; ++index)
            exchanges[index] = ice_deposition_rates(
                tables, states[index], 2.0);
        return exchanges;
    };
    const auto one = evaluate(1);
    const auto four = evaluate(4);
    REQUIRE(std::memcmp(one.data(), four.data(), sizeof(one)) == 0);
}

TEST_CASE("Thompson snow riming is bitwise identical at one and four threads",
          "[microphysics][thompson][cold][riming][threads]")
{
    using namespace tmv::thompson;
    constexpr std::size_t count = 128;
    std::array<SnowRimingState, count> states{};
    for (std::size_t index = 0; index < count; ++index)
    {
        states[index] = {258.15, 1.0, 2.0e-4, 2.0e8, 5.0e-4};
        states[index].cloud_kg_m3 *= 1.0 + 1.0e-4 * index;
        states[index].cloud_number_m3 *= 1.0 + 1.0e-4 * index;
        states[index].snow_kg_m3 *= 1.0 + 1.0e-4 * index;
        if (index % 2 != 0)
        {
            states[index].temperature_k = 268.15;
            states[index].graupel_kg_m3 = 5.0e-4
                * (1.0 + 1.0e-4 * index);
            states[index].snow_deposition_kg_m3_s = 1.0e-8;
        }
    }
    const auto evaluate = [&](int threads) {
        std::array<SnowRimingExchange, count> exchanges{};
#ifdef _OPENMP
        omp_set_num_threads(threads);
#else
        static_cast<void>(threads);
#endif
#pragma omp parallel for schedule(static)
        for (std::size_t index = 0; index < count; ++index)
            exchanges[index] = snow_riming_rates(states[index], 2.0);
        return exchanges;
    };
    const auto one = evaluate(1);
    const auto four = evaluate(4);
    REQUIRE(std::memcmp(one.data(), four.data(), sizeof(one)) == 0);
}

TEST_CASE("Thompson ice-to-snow conversion is identical at one and four threads",
          "[microphysics][thompson][cold][ice-snow][threads]")
{
    using namespace tmv::thompson;
    constexpr std::size_t count = 128;
    const auto tables = ice_reference_tables();
    std::array<IceSnowConversionState, count> states{};
    for (std::size_t index = 0; index < count; ++index)
    {
        const double factor = 1.0 + 1.0e-4 * index;
        states[index] = {1.0e-6 * factor, 1.0e3 * factor};
    }
    const auto evaluate = [&](int threads) {
        std::array<IceSnowConversionExchange, count> exchanges{};
#ifdef _OPENMP
        omp_set_num_threads(threads);
#else
        static_cast<void>(threads);
#endif
#pragma omp parallel for schedule(static)
        for (std::size_t index = 0; index < count; ++index)
            exchanges[index] = ice_to_snow_rates(
                tables, states[index], 2.0);
        return exchanges;
    };
    const auto one = evaluate(1);
    const auto four = evaluate(4);
    REQUIRE(std::memcmp(one.data(), four.data(), sizeof(one)) == 0);
}

TEST_CASE("Thompson sublimation cannot consume diagnostic particles absent from the state",
          "[microphysics][thompson][cold][deposition][donor]")
{
    using namespace tmv::thompson;
    const auto tables = ice_reference_tables();
    for (double number : {0.0, 1.0})
    {
        CAPTURE(number);
        const double dt       = 1e6;
        const auto   exchange = ice_deposition_rates(tables, {80000.0, 258.15, 1.0, 0.0, 1e-4, number, 0.0, 0.0}, dt);
        CHECK(exchange.ice_kg_m3_s < 0);
        CHECK(number + dt * exchange.ice_number_m3_s >= 0);
        CHECK(exchange.ice_number_m3_s <= 0);
    }
}

TEST_CASE("Thompson ice-to-snow number loss respects the stored population",
          "[microphysics][thompson][cold][ice-snow][donor]")
{
    using namespace tmv::thompson;
    const auto tables = ice_reference_tables();
    for (double number : {1.0, 10.0})
    {
        CAPTURE(number);
        const double dt       = 2.0;
        const auto   exchange = ice_to_snow_rates(tables, {1e-3, number}, dt);
        CHECK(exchange.ice_kg_m3_s < 0);
        CHECK(number + dt * exchange.ice_number_m3_s >= 0);
        CHECK(exchange.ice_number_m3_s >= -.95 * number / dt);
    }
}

TEST_CASE("Thompson sparse cloud freezing continues the canonical particle quadrature", "[thompson][freezing][sparse]")
{
    using namespace tmv::thompson;
    const auto               factors = generate_freezing_factors();
    const CloudFreezingIndex index{0, 10, 36, 10};
    const double             population  = factors.cloud_number[index.cloud_number];
    const int                shape       = std::min(15, int(std::floor(1e9 / population + .5)) + 2);
    const double             coefficient = 3.1415926536 * 1000 / 6;
    for (double fraction : {.01, .1, .5, .999999, 1.0})
    {
        const double mass = factors.water_content[0] * fraction;
        const double slope =
            std::cbrt(population * coefficient * std::tgamma(shape + 4.0) / std::tgamma(shape + 1.0) / mass);
        double expected_mass = 0, expected_number = 0;
        for (int bin = 0; bin < 100; ++bin)
        {
            const double diameter = (bin + 1) * 1e-6;
            const double count    = population * slope / std::tgamma(shape + 1.0) * std::pow(slope * diameter, shape) *
                                    std::exp(-slope * diameter) * 1e-6;
            const double probability =
                factors.cloud_probability[(index.ice_nuclei * 45 + index.temperature) * 100 + bin];
            expected_number = std::min(population, expected_number + probability * count);
            expected_mass += probability * count * coefficient * diameter * diameter * diameter;
            if (expected_mass >= mass)
                break;
        }
        const auto actual = generate_cloud_freezing_cell(index, mass);
        CAPTURE(fraction);
        CHECK(actual.ice_mass == Approx(expected_mass).epsilon(1e-10).margin(1e-28));
        CHECK(actual.ice_number == Approx(expected_number).epsilon(1e-10).margin(1e-12));
    }
    const auto boundary = generate_cloud_freezing_cell(index);
    const auto below    = generate_cloud_freezing_cell(index, std::nextafter(factors.water_content[0], 0.0));
    CHECK(below.ice_mass == Approx(boundary.ice_mass).epsilon(1e-12));
    CHECK(below.ice_number == Approx(boundary.ice_number).epsilon(1e-12));
    CHECK_THROWS(generate_cloud_freezing_cell(index, -1));
    CHECK_THROWS(generate_cloud_freezing_cell(index, std::numeric_limits<double>::quiet_NaN()));
}

TEST_CASE("Thompson cloud freezing interpolates raw table moments continuously", "[thompson][freezing][interpolation]")
{
    using namespace tmv::thompson;
    ColdProcessTables tables;
    tables.cloud_freezing.resize(checked_cloud_freezing_size());
    const CloudFreezingState state{258.15, 6.5e-6, 1e8, 1000};
    auto                     index = cloud_freezing_lookup_index(state);
    REQUIRE(freezing_water_content(index.cloud_content) == Approx(6e-6));
    tables.cloud_freezing[flatten(index)] = {2e-6, 2e5};
    auto upper                            = index;
    ++upper.cloud_content;
    tables.cloud_freezing[flatten(upper)] = {8e-6, 8e5};
    auto upper_number                     = index;
    ++upper_number.cloud_number;
    tables.cloud_freezing[flatten(upper_number)] = {2e-6, 2e5};
    ++upper_number.cloud_content;
    tables.cloud_freezing[flatten(upper_number)] = {8e-6, 8e5};
    const auto middle                            = cloud_freezing_exchange(tables, state);
    CHECK(middle.ice_kg_m3 == Approx(5e-6));
    CHECK(middle.ice_number_m3 == Approx(5e5));
    for (double mass : {6e-6, 7e-6})
    {
        auto left_state = state, right_state = state;
        left_state.cloud_kg_m3  = std::nextafter(mass, 0.0);
        right_state.cloud_kg_m3 = std::nextafter(mass, 1.0);
        const auto left         = cloud_freezing_exchange(tables, left_state);
        const auto right        = cloud_freezing_exchange(tables, right_state);
        CHECK(left.ice_kg_m3 == Approx(right.ice_kg_m3).epsilon(1e-12));
        CHECK(left.ice_number_m3 == Approx(right.ice_number_m3).epsilon(1e-12));
        CHECK(left.cloud_kg_m3 == -left.ice_kg_m3);
        CHECK(left.cloud_number_m3 == -left.ice_number_m3);
    }
    auto limited_state            = state;
    limited_state.cloud_number_m3 = 1e4;
    const auto limited_index      = cloud_freezing_lookup_index(limited_state);
    auto       limited_upper      = limited_index;
    ++limited_upper.cloud_content;
    tables.cloud_freezing[flatten(limited_index)] = {2e-6, 2e5};
    tables.cloud_freezing[flatten(limited_upper)] = {8e-6, 8e5};
    CHECK(cloud_freezing_exchange(tables, limited_state).ice_number_m3 == 1e4);
    tables.cloud_freezing[flatten(upper)].ice_mass = -1;
    CHECK_THROWS(cloud_freezing_exchange(tables, state));
    CHECK_THROWS(freezing_water_content(ColdProcessTableMetadata::water_contents));
}

TEST_CASE("Thompson cloud freezing interpolates logarithmic population centers", "[thompson][freezing][interpolation]")
{
    using namespace tmv::thompson;
    ColdProcessTables tables;
    tables.cloud_freezing.resize(checked_cloud_freezing_size());
    const double       lower = freezing_cloud_number(33), upper = freezing_cloud_number(34);
    CloudFreezingState state{258.15, freezing_water_content(18), std::sqrt(lower * upper), 1000};
    const auto         index = cloud_freezing_lookup_index(state);
    REQUIRE(index.cloud_number == 33);
    auto next = index;
    ++next.cloud_number;
    tables.cloud_freezing[flatten(index)] = {2e-6, 2e5};
    tables.cloud_freezing[flatten(next)]  = {8e-6, 8e5};
    const auto middle                     = cloud_freezing_exchange(tables, state);
    CHECK(middle.ice_kg_m3 == Approx(5e-6));
    CHECK(middle.ice_number_m3 == Approx(5e5));
    state.cloud_number_m3 = lower;
    CHECK(cloud_freezing_exchange(tables, state).ice_kg_m3 == Approx(2e-6));
    state.cloud_number_m3 = std::nextafter(upper, 0.0);
    const auto left       = cloud_freezing_exchange(tables, state);
    state.cloud_number_m3 = std::nextafter(upper, std::numeric_limits<double>::infinity());
    const auto right      = cloud_freezing_exchange(tables, state);
    CHECK(left.ice_kg_m3 == Approx(right.ice_kg_m3).epsilon(1e-12));
    CHECK(left.ice_number_m3 == Approx(right.ice_number_m3).epsilon(1e-12));
    CHECK_THROWS(freezing_cloud_number(ColdProcessTableMetadata::cloud_numbers));
}

TEST_CASE("Thompson ice conversion interpolates independent tail-distribution moments",
          "[thompson][ice-snow][interpolation][reference]")
{
    using namespace tmv::thompson;
    const auto quadrature = [](double mass, double number)
    {
        const double     coefficient = 3.1415926536 * 890 / 6;
        const double     slope       = std::cbrt(6 * coefficient * number / mass);
        constexpr double minimum = 1.2898438095973233e-5, maximum = 600e-6;
        IceSnowCell      result;
        for (int bin = 0; bin < 100; ++bin)
        {
            const double left     = minimum * std::pow(maximum / minimum, bin / 100.0);
            const double right    = minimum * std::pow(maximum / minimum, (bin + 1) / 100.0);
            const double diameter = std::sqrt(left * right);
            if (diameter < 300e-6)
                continue;
            const double count = number * slope * std::exp(-slope * diameter) * (right - left);
            result.snow_mass += count * coefficient * diameter * diameter * diameter;
            result.ice_number += count;
        }
        return result;
    };
    ColdProcessTables tables;
    tables.ice_snow.resize(checked_ice_snow_size());
    for (std::size_t n : {53, 54})
        for (std::size_t m : {55, 56})
            tables.ice_snow[flatten(IceSnowIndex{m, n})] = quadrature(ice_table_content(m), ice_table_number(n));
    constexpr double mass = .00021191393505309448, number = 999990.98395577073, dt = 1.0 / 6.0;
    const auto       direct = quadrature(mass, number);
    CHECK(direct.snow_mass == Approx(1.61231235847e-5).epsilon(1e-9));
    CHECK(direct.ice_number == Approx(819.82976284).epsilon(1e-9));
    const auto exchange = ice_to_snow_rates(tables, {mass, number}, dt);
    CHECK(dt * exchange.snow_kg_m3_s == Approx(1.66620485464e-5).epsilon(1e-9));
    CHECK(-dt * exchange.ice_number_m3_s == Approx(841.23527281).epsilon(1e-9));
    const auto left  = ice_to_snow_rates(tables, {mass, std::nextafter(1e6, 0.0)}, dt);
    const auto right = ice_to_snow_rates(tables, {mass, std::nextafter(1e6, 2e6)}, dt);
    CHECK(left.snow_kg_m3_s == Approx(right.snow_kg_m3_s).epsilon(1e-12));
    CHECK(left.ice_number_m3_s == Approx(right.ice_number_m3_s).epsilon(1e-12));
    CHECK(exchange.ice_kg_m3_s == -exchange.snow_kg_m3_s);
    CHECK_THROWS(ice_table_content(ColdProcessTableMetadata::ice_contents));
    CHECK_THROWS(ice_table_number(ColdProcessTableMetadata::ice_numbers));
}
