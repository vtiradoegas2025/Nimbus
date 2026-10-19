#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_frozen_collection.hpp"
#include "../support/milbrandt_frozen_collection_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_frozen_collection_reference;
using Population = FrozenCollectionPopulation;
using State = FrozenCollectionState;
using Cell = FrozenCollectionCell;
using Result = FrozenCollectionResult;
using Producer = FrozenCollectionProducer;
using Failure = FrozenCollectionFailure;

ColdParameters parameters()
{
    return {19.3, 206.89, 71.34, 149.1, 11.72, 0, 0, 0, 0, 0,
            .37, .6384, .6635, .5, .41, 209.439510239, 471.238898038,
            440, 523.598775598, .1597, 3, 3, 3, 3, 2.078, true};
}

Cell physical_cell(long double alpha = 2, double density = .65,
                   double temperature = 253.15)
{
    Cell result{temperature, 90000, density, {}};
    constexpr std::array<double, 4> mass{2e-4, 5e-4, 3e-4, 2e-4};
    constexpr std::array<double, 4> number{2e4, 40, 15, 2};
    for (std::size_t species = 0; species < 4; ++species)
    {
        result.initial.frozen[species] = reference::population(
            species, mass[species], number[species], alpha);
    }
    return result;
}

void same_population(const Population& actual, const Population& expected)
{
    CHECK(actual.mass_per_kg == expected.mass_per_kg);
    CHECK(actual.number_per_kg == expected.number_per_kg);
    CHECK(actual.reflectivity_per_kg == expected.reflectivity_per_kg);
}

void same_producer(const Producer& actual, const Producer& expected)
{
    CHECK(actual.source().temperature_k == expected.source().temperature_k);
    CHECK(actual.source().pressure_pa == expected.source().pressure_pa);
    CHECK(actual.source().dry_density_kg_m3 == expected.source().dry_density_kg_m3);
    CHECK(actual.parameters() == expected.parameters());
    CHECK(actual.hail_processes() == expected.hail_processes());
    CHECK(actual.duration_s() == expected.duration_s());
    CHECK(actual.event_count() == expected.event_count());
    CHECK(actual.source_alpha() == expected.source_alpha());
    CHECK(actual.source_lambda_per_m() == expected.source_lambda_per_m());
    CHECK(actual.mass_weighted_speed_m_s() == expected.mass_weighted_speed_m_s());
    for (std::size_t species = 0; species < 4; ++species)
    {
        same_population(actual.source().initial.frozen[species],
                        expected.source().initial.frozen[species]);
    }
    for (std::size_t channel = 0; channel < 5; ++channel)
    {
        const auto& a = actual.channels()[channel];
        const auto& e = expected.channels()[channel];
        CHECK(a.channel == e.channel);
        CHECK(a.raw_mass_increment_per_kg == e.raw_mass_increment_per_kg);
        CHECK(a.raw_number_increment_per_kg == e.raw_number_increment_per_kg);
        CHECK(a.admitted_mass_increment_per_kg == e.admitted_mass_increment_per_kg);
        CHECK(a.admitted_number_increment_per_kg == e.admitted_number_increment_per_kg);
        CHECK(a.donor_admission_factor == e.donor_admission_factor);
    }
}

void same_result(const Result& actual, const Result& expected)
{
    CHECK(actual.mass_increment_per_kg == expected.mass_increment_per_kg);
    CHECK(actual.number_increment_per_kg == expected.number_increment_per_kg);
    CHECK(actual.reflectivity_increment_per_kg == expected.reflectivity_increment_per_kg);
    CHECK(actual.temperature_increment_k == expected.temperature_increment_k);
    for (std::size_t species = 0; species < 4; ++species)
    {
        same_population(actual.unrounded.frozen[species], expected.unrounded.frozen[species]);
        CHECK(actual.rounded[species].mass_per_kg == expected.rounded[species].mass_per_kg);
        CHECK(actual.rounded[species].number_per_kg == expected.rounded[species].number_per_kg);
        CHECK(actual.rounded[species].reflectivity_per_kg == expected.rounded[species].reflectivity_per_kg);
    }
    same_producer(actual.producer, expected.producer);
}

void check_capture(const Cell& cell, double duration, const Producer& actual,
                   const reference::Reference& expected)
{
    constexpr std::array<FrozenCollectionChannel, 5> channel_names{
        FrozenCollectionChannel::IceToSnow, FrozenCollectionChannel::IceToGraupel,
        FrozenCollectionChannel::IceToHail, FrozenCollectionChannel::SnowToHail,
        FrozenCollectionChannel::SnowSelfCollection};
    CHECK(actual.event_count() == 5);
    CHECK(actual.parameters() == default_cold_parameters);
    CHECK(actual.hail_processes());
    CHECK(actual.duration_s() == duration);
    CHECK(actual.source().temperature_k == cell.temperature_k);
    CHECK(actual.source().pressure_pa == cell.pressure_pa);
    CHECK(actual.source().dry_density_kg_m3 == cell.dry_density_kg_m3);
    for (std::size_t species = 0; species < 4; ++species)
    {
        same_population(actual.source().initial.frozen[species], cell.initial.frozen[species]);
        if (cell.initial.frozen[species].mass_per_kg > 0)
        {
            CHECK(actual.source_alpha()[species] == Approx(
                static_cast<double>(expected.spectra[species].alpha)).epsilon(2e-9).margin(2e-10));
            CHECK(actual.source_lambda_per_m()[species] == Approx(
                static_cast<double>(expected.spectra[species].lambda)).epsilon(2e-10));
            CHECK(actual.mass_weighted_speed_m_s()[species] == Approx(
                static_cast<double>(expected.spectra[species].velocity)).epsilon(2e-10));
        }
    }
    for (std::size_t channel = 0; channel < 5; ++channel)
    {
        const auto& a = actual.channels()[channel];
        const auto& e = expected.channels[channel];
        CHECK(a.channel == channel_names[channel]);
        CHECK(a.raw_mass_increment_per_kg == Approx(static_cast<double>(e.raw_mass)).epsilon(3e-10).margin(1e-28));
        CHECK(a.raw_number_increment_per_kg == Approx(static_cast<double>(e.raw_number)).epsilon(3e-10).margin(1e-22));
        CHECK(a.admitted_mass_increment_per_kg == Approx(static_cast<double>(e.mass)).epsilon(3e-10).margin(1e-28));
        CHECK(a.admitted_number_increment_per_kg == Approx(static_cast<double>(e.number)).epsilon(3e-10).margin(1e-22));
        CHECK(a.donor_admission_factor == Approx(static_cast<double>(e.factor)).epsilon(3e-10));
    }
}

void check_endpoint(const Cell& cell, const Result& actual,
                    const reference::Reference& expected)
{
    CHECK(actual.temperature_increment_k == 0);
    long double total_mass = 0;
    long double transferred_mass = 0;
    long double number_loss = 0;
    for (std::size_t species = 0; species < 4; ++species)
    {
        const auto& endpoint = actual.unrounded.frozen[species];
        CHECK(actual.mass_increment_per_kg[species] == Approx(
            static_cast<double>(expected.mass_increment[species])).epsilon(3e-10).margin(1e-28));
        CHECK(actual.number_increment_per_kg[species] == Approx(
            static_cast<double>(expected.number_increment[species])).epsilon(3e-10).margin(1e-22));
        CHECK(actual.reflectivity_increment_per_kg[species] == Approx(
            static_cast<double>(expected.reflectivity_increment[species])).epsilon(3e-10).margin(1e-38));
        CHECK(endpoint.mass_per_kg == Approx(static_cast<double>(expected.endpoint[species][0])).epsilon(3e-12));
        CHECK(endpoint.number_per_kg == Approx(static_cast<double>(expected.endpoint[species][1])).epsilon(3e-12));
        CHECK(endpoint.reflectivity_per_kg == Approx(static_cast<double>(expected.endpoint[species][2])).epsilon(3e-12));
        CHECK(actual.rounded[species].mass_per_kg == static_cast<float>(endpoint.mass_per_kg));
        CHECK(actual.rounded[species].number_per_kg == static_cast<float>(endpoint.number_per_kg));
        CHECK(actual.rounded[species].reflectivity_per_kg == static_cast<float>(endpoint.reflectivity_per_kg));
        total_mass += actual.mass_increment_per_kg[species];
        transferred_mass += std::abs(actual.mass_increment_per_kg[species]);
        number_loss += actual.number_increment_per_kg[species];
        if (endpoint.mass_per_kg > 0)
        {
            const auto represented_alpha = reference::represented_alpha(endpoint, species);
            CHECK(represented_alpha > -1);
            CHECK(represented_alpha <= 200 + 1e-8L);
            const Population rounded{actual.rounded[species].mass_per_kg,
                actual.rounded[species].number_per_kg, actual.rounded[species].reflectivity_per_kg};
            CHECK(reference::represented_alpha(rounded, species) > -1);
            CHECK(reference::represented_alpha(rounded, species) <= 200 + 1e-8L);
        }
    }
    CHECK(std::abs(total_mass) <= 8 * std::numeric_limits<double>::epsilon()
        * std::max(transferred_mass, 1e-30L));
    CHECK(number_loss <= 0);
    CHECK(actual.number_increment_per_kg[2] == 0);
    CHECK(actual.number_increment_per_kg[3] == 0);
    if (cell.initial.frozen[1].mass_per_kg > 0)
    {
        const auto& snow = cell.initial.frozen[1];
        const long double spherical = snow.reflectivity_per_kg
            * (2 * expected.mass_increment[1] / snow.mass_per_kg
               - expected.number_increment[1] / snow.number_per_kg);
        if (expected.reflectivity_increment[1] != 0)
        {
            CHECK(std::abs(spherical - expected.reflectivity_increment[1])
                  > .1L * std::abs(expected.reflectivity_increment[1]));
        }
    }
}

Result successful_sentinel()
{
    Result result;
    std::string error;
    REQUIRE(evaluate_exact_triple_frozen_collection_cell(
        parameters(), physical_cell(), 1e-3, result, error));
    return result;
}

std::array<double ColdParameters::*, 25> parameter_members()
{
    return {&ColdParameters::a_g_, &ColdParameters::a_h_, &ColdParameters::a_i_,
        &ColdParameters::a_r_, &ColdParameters::a_s_, &ColdParameters::alpha_g_,
        &ColdParameters::alpha_h_, &ColdParameters::alpha_i_, &ColdParameters::alpha_r_,
        &ColdParameters::alpha_s_, &ColdParameters::b_g_, &ColdParameters::b_h_,
        &ColdParameters::b_i_, &ColdParameters::b_r_, &ColdParameters::b_s_,
        &ColdParameters::c_g_, &ColdParameters::c_h_, &ColdParameters::c_i_,
        &ColdParameters::c_r_, &ColdParameters::c_s_, &ColdParameters::d_g_,
        &ColdParameters::d_h_, &ColdParameters::d_i_, &ColdParameters::d_r_,
        &ColdParameters::d_s_};
}

} // namespace

TEST_CASE("triple frozen collection matches independent dimensional kernels and noncubic moments",
          "[physics][milbrandt][triple_frozen_collection]")
{
    for (long double alpha : {-.5L, 0.0L, 2.0L, 20.0L, 199.9L})
    {
        for (double density : {.35, .65, 1.2})
        {
            for (double temperature : {233.15, 253.15, 272.15})
            {
                const auto cell = physical_cell(alpha, density, temperature);
                INFO("shape=" << static_cast<double>(alpha) << " rho=" << density
                     << " T=" << temperature);
                constexpr double duration = 1e-4;
                const auto expected = reference::evaluate(cell, duration);
                Result result;
                std::string error = "sentinel";
                Failure failure = Failure::InvalidProducer;
                REQUIRE(evaluate_exact_triple_frozen_collection_cell(
                    parameters(), cell, duration, result, error, 5, &failure));
                CHECK(error.empty());
                CHECK(failure == Failure::None);
                check_capture(cell, duration, result.producer, expected);
                check_endpoint(cell, result, expected);
                Result repeated;
                REQUIRE(apply_exact_triple_frozen_collection_cell(
                    parameters(), cell, duration, result.producer, repeated, error));
                same_result(repeated, result);
            }
        }
        const auto cell = physical_cell(alpha);
        const auto analytic = reference::evaluate(cell, 1e-4);
        const auto quadrature = reference::evaluate(cell, 1e-4, true);
        for (std::size_t channel = 0; channel < 4; ++channel)
        {
            CHECK(static_cast<double>(analytic.channels[channel].raw_mass) == Approx(
                static_cast<double>(quadrature.channels[channel].raw_mass)).epsilon(4e-11));
            CHECK(static_cast<double>(analytic.channels[channel].raw_number) == Approx(
                static_cast<double>(quadrature.channels[channel].raw_number)).epsilon(4e-11));
        }
    }
}

TEST_CASE("triple frozen collection admission jointly limits donors without publishing depletion",
          "[physics][milbrandt][triple_frozen_collection]")
{
    const auto p = parameters();
    std::string error;
    const auto sentinel = successful_sentinel();
    auto ice_cell = physical_cell();
    const auto unit = reference::evaluate(ice_cell, 1);
    const long double ice_demand = unit.channels[0].raw_mass
        + unit.channels[1].raw_mass + unit.channels[2].raw_mass;
    const double duration = static_cast<double>(2 * ice_cell.initial.frozen[0].mass_per_kg / ice_demand);
    const auto expected = reference::evaluate(ice_cell, duration);
    REQUIRE(expected.channels[0].factor < 1);
    Producer producer;
    Failure failure = Failure::InvalidProducer;
    REQUIRE(produce_exact_frozen_collection_admission(
        p, ice_cell, duration, producer, error, 5, &failure));
    CHECK(failure == Failure::None);
    check_capture(ice_cell, duration, producer, expected);
    for (std::size_t channel = 1; channel < 3; ++channel)
    {
        CHECK(producer.channels()[channel].donor_admission_factor
              == producer.channels()[0].donor_admission_factor);
    }
    CHECK(producer.channels()[0].admitted_mass_increment_per_kg
          + producer.channels()[1].admitted_mass_increment_per_kg
          + producer.channels()[2].admitted_mass_increment_per_kg
          == Approx(ice_cell.initial.frozen[0].mass_per_kg).epsilon(2e-14));
    Result result = sentinel;
    CHECK_FALSE(apply_exact_triple_frozen_collection_cell(
        p, ice_cell, duration, producer, result, error, 5, &failure));
    CHECK(failure == Failure::IntervalRejected);
    CHECK_FALSE(error.empty());
    same_result(result, sentinel);
    result = sentinel;
    CHECK_FALSE(evaluate_exact_triple_frozen_collection_cell(
        p, ice_cell, duration, result, error, 5, &failure));
    CHECK(failure == Failure::IntervalRejected);
    same_result(result, sentinel);

    auto snow_cell = physical_cell();
    snow_cell.initial.frozen[0] = reference::population(0, 1e-5, 1e4, 2);
    snow_cell.initial.frozen[3] = reference::population(3, .01, 50, 2);
    double snow_duration = 1;
    auto snow_expected = reference::evaluate(snow_cell, snow_duration);
    for (int attempt = 0; snow_expected.channels[3].factor == 1 && attempt < 30; ++attempt)
    {
        snow_duration *= 2;
        snow_expected = reference::evaluate(snow_cell, snow_duration);
    }
    REQUIRE(snow_expected.channels[3].factor < 1);
    REQUIRE(snow_expected.channels[0].mass > 0);
    REQUIRE(snow_expected.channels[4].number > 0);
    REQUIRE(produce_exact_frozen_collection_admission(
        p, snow_cell, snow_duration, producer, error, 5, &failure));
    check_capture(snow_cell, snow_duration, producer, snow_expected);
    CHECK(producer.channels()[3].donor_admission_factor
          == producer.channels()[4].donor_admission_factor);
    const long double factor_without_addback = std::min(1.0L,
        std::min(static_cast<long double>(snow_cell.initial.frozen[1].mass_per_kg)
                 / snow_expected.channels[3].raw_mass,
                 static_cast<long double>(snow_cell.initial.frozen[1].number_per_kg)
                 / (snow_expected.channels[3].raw_number + snow_expected.channels[4].raw_number)));
    CHECK(snow_expected.channels[3].factor >= factor_without_addback);
    result = sentinel;
    CHECK_FALSE(apply_exact_triple_frozen_collection_cell(
        p, snow_cell, snow_duration, producer, result, error, 5, &failure));
    CHECK(failure == Failure::IntervalRejected);
    same_result(result, sentinel);
}

TEST_CASE("triple frozen snow self collection keeps its operational cap and general mass law",
          "[physics][milbrandt][triple_frozen_collection]")
{
    Cell cell{253.15, 90000, .65, {}};
    cell.initial.frozen[1] = reference::population(1, .005, 100, 0);
    const auto unit = reference::evaluate(cell, 1);
    REQUIRE(unit.channels[4].raw_number > 0);
    const double duration = static_cast<double>(cell.initial.frozen[1].number_per_kg
        / unit.channels[4].raw_number);
    const auto expected = reference::evaluate(cell, duration);
    REQUIRE(expected.channels[4].raw_number == .5L * cell.initial.frozen[1].number_per_kg);
    Result result;
    std::string error;
    REQUIRE(evaluate_exact_triple_frozen_collection_cell(parameters(), cell, duration, result, error));
    check_capture(cell, duration, result.producer, expected);
    check_endpoint(cell, result, expected);
    CHECK(result.unrounded.frozen[1].mass_per_kg == cell.initial.frozen[1].mass_per_kg);
    CHECK(result.unrounded.frozen[1].number_per_kg == .5 * cell.initial.frozen[1].number_per_kg);
    CHECK(result.unrounded.frozen[1].reflectivity_per_kg > cell.initial.frozen[1].reflectivity_per_kg);
    for (std::size_t channel = 0; channel < 4; ++channel)
    {
        CHECK(result.producer.channels()[channel].raw_mass_increment_per_kg == 0);
        CHECK(result.producer.channels()[channel].raw_number_increment_per_kg == 0);
    }
}

TEST_CASE("triple frozen collection rejects invalid sources contexts and capacity atomically",
          "[physics][milbrandt][triple_frozen_collection]")
{
    const auto original = physical_cell();
    const auto p = parameters();
    const auto sentinel = successful_sentinel();
    const auto reject = [&](const ColdParameters& supplied, const Cell& cell,
                            double duration, std::size_t capacity, Failure expected)
    {
        Result result = sentinel;
        Producer producer = sentinel.producer;
        std::string error;
        Failure failure = Failure::None;
        CHECK_FALSE(evaluate_exact_triple_frozen_collection_cell(
            supplied, cell, duration, result, error, capacity, &failure));
        CHECK(failure == expected);
        CHECK_FALSE(error.empty());
        same_result(result, sentinel);
        CHECK_FALSE(produce_exact_frozen_collection_admission(
            supplied, cell, duration, producer, error, capacity, &failure));
        CHECK(failure == expected);
        same_producer(producer, sentinel.producer);
    };
    for (std::size_t capacity : {0u, 1u, 2u, 3u, 4u, 6u,
                                 std::numeric_limits<unsigned>::max()})
    {
        reject(p, original, 1e-3, capacity, Failure::Capacity);
    }
    for (double duration : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()})
    {
        reject(p, original, duration, 5, Failure::InvalidContext);
    }
    for (double temperature : {149.999, 273.15, 400.0,
                                std::numeric_limits<double>::quiet_NaN()})
    {
        auto cell = original;
        cell.temperature_k = temperature;
        reject(p, cell, 1e-3, 5, Failure::InvalidContext);
    }
    for (double pressure : {99.999, 200001.0, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()})
    {
        auto cell = original;
        cell.pressure_pa = pressure;
        reject(p, cell, 1e-3, 5, Failure::InvalidContext);
    }
    for (double density : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()})
    {
        auto cell = original;
        cell.dry_density_kg_m3 = density;
        reject(p, cell, 1e-3, 5, Failure::InvalidContext);
    }
    for (auto member : parameter_members())
    {
        auto supplied = p;
        supplied.*member = std::nextafter(supplied.*member, std::numeric_limits<double>::infinity());
        reject(supplied, original, 1e-3, 5, Failure::InvalidContext);
    }
    constexpr std::array<double Population::*, 3> members{
        &Population::mass_per_kg, &Population::number_per_kg, &Population::reflectivity_per_kg};
    for (std::size_t species = 0; species < 4; ++species)
    {
        for (auto member : members)
        {
            for (double value : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::quiet_NaN()})
            {
                auto cell = original;
                cell.initial.frozen[species].*member = value;
                reject(p, cell, 1e-3, 5, Failure::InvalidSource);
            }
        }
        auto above_shape = original;
        above_shape.initial.frozen[species] = reference::population(species,
            original.initial.frozen[species].mass_per_kg,
            original.initial.frozen[species].number_per_kg, 201);
        reject(p, above_shape, 1e-3, 5, Failure::InvalidSource);
        auto cauchy = original;
        const auto& population = cauchy.initial.frozen[species];
        cauchy.initial.frozen[species].reflectivity_per_kg = static_cast<double>(
            population.number_per_kg * std::pow(population.mass_per_kg
            / (reference::mass_coefficient[species] * population.number_per_kg),
            6 / reference::mass_exponent[species]) * .99L);
        reject(p, cauchy, 1e-3, 5, Failure::InvalidSource);
    }
}

TEST_CASE("triple frozen immutable capture rejects transplantation and repairs detached endpoints",
          "[physics][milbrandt][triple_frozen_collection]")
{
    const auto p = parameters();
    const auto cell = physical_cell();
    constexpr double duration = 1e-3;
    const auto sentinel = successful_sentinel();
    Result admitted;
    std::string error;
    REQUIRE(evaluate_exact_triple_frozen_collection_cell(p, cell, duration, admitted, error));
    const auto reject = [&](const ColdParameters& supplied, const Cell& transplanted,
                            double dt, const Producer& producer)
    {
        Result result = sentinel;
        Failure failure = Failure::None;
        CHECK_FALSE(apply_exact_triple_frozen_collection_cell(
            supplied, transplanted, dt, producer, result, error, 5, &failure));
        CHECK(failure != Failure::None);
        CHECK_FALSE(error.empty());
        same_result(result, sentinel);
    };
    reject(p, cell, duration, Producer{});
    reject(p, cell, std::nextafter(duration, 1.0), admitted.producer);
    for (double Cell::*member : {&Cell::temperature_k, &Cell::pressure_pa, &Cell::dry_density_kg_m3})
    {
        auto transplanted = cell;
        transplanted.*member = std::nextafter(transplanted.*member, std::numeric_limits<double>::infinity());
        reject(p, transplanted, duration, admitted.producer);
    }
    for (std::size_t species = 0; species < 4; ++species)
    {
        for (double Population::*member : {&Population::mass_per_kg,
                &Population::number_per_kg, &Population::reflectivity_per_kg})
        {
            auto transplanted = cell;
            transplanted.initial.frozen[species].*member = std::nextafter(
                transplanted.initial.frozen[species].*member, std::numeric_limits<double>::infinity());
            reject(p, transplanted, duration, admitted.producer);
        }
    }
    for (auto member : parameter_members())
    {
        auto supplied = p;
        supplied.*member = std::nextafter(supplied.*member, std::numeric_limits<double>::infinity());
        reject(supplied, cell, duration, admitted.producer);
    }
    auto disabled_hail = p;
    disabled_hail.hail_processes = false;
    reject(disabled_hail, cell, duration, admitted.producer);
    Result existing_hail;
    REQUIRE(evaluate_exact_triple_frozen_collection_cell(
        disabled_hail, cell, duration, existing_hail, error));
    CHECK_FALSE(existing_hail.producer.hail_processes());
    CHECK(existing_hail.mass_increment_per_kg == admitted.mass_increment_per_kg);
    CHECK(existing_hail.number_increment_per_kg == admitted.number_increment_per_kg);
    CHECK(existing_hail.reflectivity_increment_per_kg == admitted.reflectivity_increment_per_kg);
    reject(p, cell, duration, existing_hail.producer);
    Result detached = admitted;
    detached.unrounded.frozen[0].mass_per_kg = -1;
    detached.rounded[1].reflectivity_per_kg = -2;
    detached.mass_increment_per_kg[2] = -3;
    detached.number_increment_per_kg[3] = -4;
    detached.reflectivity_increment_per_kg[0] = -5;
    detached.temperature_increment_k = 6;
    REQUIRE(apply_exact_triple_frozen_collection_cell(
        p, cell, duration, detached.producer, detached, error));
    same_result(detached, admitted);
}

TEST_CASE("triple frozen empty boundaries and FP32 publication keep atomic realizability",
          "[physics][milbrandt][triple_frozen_collection]")
{
    const auto p = parameters();
    std::string error;
    for (double temperature : {150.0, 253.15, std::nextafter(273.15, 0.0)})
    {
        for (double pressure : {100.0, 90000.0, 200000.0})
        {
            Cell cell{temperature, pressure, .65, {}};
            Result result;
            REQUIRE(evaluate_exact_triple_frozen_collection_cell(p, cell, 1, result, error));
            check_capture(cell, 1, result.producer, reference::evaluate(cell, 1));
            for (std::size_t species = 0; species < 4; ++species)
            {
                same_population(result.unrounded.frozen[species], {});
                CHECK(result.rounded[species].mass_per_kg == 0);
                CHECK(result.rounded[species].number_per_kg == 0);
                CHECK(result.rounded[species].reflectivity_per_kg == 0);
            }
        }
    }
    const auto sentinel = successful_sentinel();
    Cell tiny{253.15, 90000, .65, {}};
    tiny.initial.frozen[0] = reference::population(0, 1e-40, 1e-3, 2);
    REQUIRE(tiny.initial.frozen[0].reflectivity_per_kg > 0);
    REQUIRE(static_cast<float>(tiny.initial.frozen[0].reflectivity_per_kg) == 0);
    Producer producer;
    Failure failure = Failure::None;
    REQUIRE(produce_exact_frozen_collection_admission(p, tiny, 1e-3, producer, error, 5, &failure));
    Result result = sentinel;
    CHECK_FALSE(apply_exact_triple_frozen_collection_cell(p, tiny, 1e-3, producer, result, error, 5, &failure));
    CHECK(failure == Failure::IntervalRejected);
    same_result(result, sentinel);

    const auto original = physical_cell(20);
    Cell restored = original;
    for (auto& population : restored.initial.frozen)
    {
        for (double Population::*member : {&Population::mass_per_kg,
                &Population::number_per_kg, &Population::reflectivity_per_kg})
        {
            std::ostringstream encoded;
            encoded.imbue(std::locale::classic());
            encoded << std::setprecision(std::numeric_limits<double>::max_digits10) << population.*member;
            std::istringstream decoded(encoded.str());
            decoded.imbue(std::locale::classic());
            decoded >> population.*member;
            REQUIRE_FALSE(decoded.fail());
        }
    }
    Result original_result;
    Result restored_result;
    REQUIRE(evaluate_exact_triple_frozen_collection_cell(p, original, 1e-3, original_result, error));
    REQUIRE(evaluate_exact_triple_frozen_collection_cell(p, restored, 1e-3, restored_result, error));
    same_result(restored_result, original_result);

    Cell threshold{253.15, 90000, .65, {}};
    for (std::size_t species = 0; species < 4; ++species)
    {
        threshold.initial.frozen[species] = reference::population(species, 1e-14, 1e-4, 2);
    }
    REQUIRE(evaluate_exact_triple_frozen_collection_cell(p, threshold, 1, result, error));
    for (const auto& channel : result.producer.channels())
    {
        CHECK(channel.raw_mass_increment_per_kg == 0);
        CHECK(channel.raw_number_increment_per_kg == 0);
    }
    for (std::size_t species = 0; species < 4; ++species)
    {
        same_population(result.unrounded.frozen[species], threshold.initial.frozen[species]);
        threshold.initial.frozen[species] = reference::population(
            species, std::nextafter(1e-14, 1.0), 1e-4, 2);
    }
    REQUIRE(evaluate_exact_triple_frozen_collection_cell(p, threshold, 1, result, error));
    for (std::size_t channel = 0; channel < 4; ++channel)
    {
        CHECK(result.producer.channels()[channel].raw_mass_increment_per_kg > 0);
        CHECK(result.producer.channels()[channel].raw_number_increment_per_kg > 0);
    }
    CHECK(result.producer.channels()[4].raw_number_increment_per_kg > 0);

    Cell empty{253.15, 90000, .65, {}};
    REQUIRE(produce_exact_frozen_collection_admission(p, empty, 1, producer, error));
    empty.initial.frozen[0].mass_per_kg = -0.0;
    result = sentinel;
    CHECK_FALSE(apply_exact_triple_frozen_collection_cell(p, empty, 1, producer, result, error, 5, &failure));
    CHECK(failure == Failure::InvalidProducer);
    same_result(result, sentinel);
}

TEST_CASE("triple frozen collection pure one and four worker continuations agree exactly",
          "[physics][milbrandt][triple_frozen_collection]")
{
    constexpr int columns = 32;
    const auto p = parameters();
    std::array<Cell, columns> sources{};
    std::array<Result, columns> serial{};
    std::array<Result, columns> parallel{};
    std::array<bool, columns> passed{};
    std::array<std::string, columns> errors{};
    int observed_workers = 1;
    for (int column = 0; column < columns; ++column)
    {
        sources[column] = physical_cell(column % 4 * 3, .4 + .025 * column,
                                          233.15 + column);
        REQUIRE(evaluate_exact_triple_frozen_collection_cell(
            p, sources[column], .01, serial[column], errors[column]));
    }
#ifdef _OPENMP
#pragma omp parallel num_threads(4)
    {
#pragma omp single
        observed_workers = omp_get_num_threads();
#pragma omp for
#endif
    for (int column = 0; column < columns; ++column)
    {
        passed[column] = evaluate_exact_triple_frozen_collection_cell(
            p, sources[column], .01, parallel[column], errors[column]);
    }
#ifdef _OPENMP
    }
    REQUIRE(observed_workers == 4);
#else
    CHECK(observed_workers == 1);
#endif
    for (int column = 0; column < columns; ++column)
    {
        REQUIRE(passed[column]);
        same_result(parallel[column], serial[column]);
        sources[column].initial = serial[column].unrounded;
        REQUIRE(evaluate_exact_triple_frozen_collection_cell(
            p, sources[column], .01, serial[column], errors[column]));
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(4)
#endif
    for (int column = 0; column < columns; ++column)
    {
        passed[column] = evaluate_exact_triple_frozen_collection_cell(
            p, sources[column], .01, parallel[column], errors[column]);
    }
    for (int column = 0; column < columns; ++column)
    {
        REQUIRE(passed[column]);
        same_result(parallel[column], serial[column]);
        check_endpoint(sources[column], parallel[column], reference::evaluate(sources[column], .01));
    }
}

TEST_CASE("triple frozen collection timestep refinement converges to independently integrated moments",
          "[physics][milbrandt][triple_frozen_collection]")
{
    using Vector = std::array<std::array<long double, 3>, 4>;
    const auto p = parameters();
    const auto initial = physical_cell();
    constexpr long double duration = 10;
    Vector source{};
    for (std::size_t species = 0; species < 4; ++species)
    {
        const auto& population = initial.initial.frozen[species];
        source[species] = {population.mass_per_kg, population.number_per_kg,
                           population.reflectivity_per_kg};
    }
    const auto slope = [&](const Vector& state)
    {
        Cell cell = initial;
        for (std::size_t species = 0; species < 4; ++species)
        {
            cell.initial.frozen[species] = {static_cast<double>(state[species][0]),
                static_cast<double>(state[species][1]), static_cast<double>(state[species][2])};
        }
        const auto flow = reference::evaluate(cell, 1);
        Vector result{};
        for (const auto& channel : flow.channels)
        {
            REQUIRE(channel.factor == 1);
        }
        REQUIRE(flow.channels[4].raw_number < .5L * cell.initial.frozen[1].number_per_kg);
        for (std::size_t species = 0; species < 4; ++species)
        {
            result[species] = {flow.mass_increment[species], flow.number_increment[species],
                               flow.reflectivity_increment[species]};
        }
        return result;
    };
    const auto stage = [](const Vector& state, const Vector& slope, long double interval)
    {
        Vector result = state;
        for (std::size_t species = 0; species < 4; ++species)
        {
            for (std::size_t moment = 0; moment < 3; ++moment)
            {
                result[species][moment] += interval * slope[species][moment];
            }
        }
        return result;
    };
    const auto integrated_reference = [&](int steps)
    {
        Vector state = source;
        const long double h = duration / steps;
        for (int step = 0; step < steps; ++step)
        {
            const auto k1 = slope(state);
            const auto k2 = slope(stage(state, k1, h / 2));
            const auto k3 = slope(stage(state, k2, h / 2));
            const auto k4 = slope(stage(state, k3, h));
            for (std::size_t species = 0; species < 4; ++species)
            {
                for (std::size_t moment = 0; moment < 3; ++moment)
                {
                    state[species][moment] += h / 6 * (k1[species][moment]
                        + 2 * k2[species][moment] + 2 * k3[species][moment]
                        + k4[species][moment]);
                }
            }
        }
        return state;
    };
    const auto expected = integrated_reference(64);
    const auto refined = integrated_reference(128);
    for (std::size_t species = 0; species < 4; ++species)
    {
        for (std::size_t moment = 0; moment < 3; ++moment)
        {
            CHECK(std::abs(expected[species][moment] - refined[species][moment])
                  / source[species][moment] < 2e-12L);
        }
    }
    std::array<long double, 3> error{};
    for (std::size_t refinement = 0; refinement < error.size(); ++refinement)
    {
        const int steps = 4 << refinement;
        const double h = static_cast<double>(duration / steps);
        Cell cell = initial;
        Result result;
        std::string diagnostic;
        for (int step = 0; step < steps; ++step)
        {
            REQUIRE(evaluate_exact_triple_frozen_collection_cell(p, cell, h, result, diagnostic));
            cell.initial = result.unrounded;
        }
        for (std::size_t species = 0; species < 4; ++species)
        {
            const std::array<long double, 3> actual{
                cell.initial.frozen[species].mass_per_kg,
                cell.initial.frozen[species].number_per_kg,
                cell.initial.frozen[species].reflectivity_per_kg};
            for (std::size_t moment = 0; moment < 3; ++moment)
            {
                error[refinement] = std::max(error[refinement],
                    std::abs(actual[moment] - refined[species][moment]) / source[species][moment]);
            }
        }
    }
    CHECK(error[0] > 1e-11L);
    CHECK(error[1] < .55L * error[0]);
    CHECK(error[2] < .55L * error[1]);
    CHECK(error[2] < 1e-5L);
}
