#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_rime_splintering.hpp"
#include "support/milbrandt_frozen_collection_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace
{

using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_frozen_collection_reference;

ColdParameters parameters(bool hail = true)
{
    const auto& p = default_cold_parameters;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], hail};
}

RimeSplinteringCell source(bool ice_present = true, long double shape = 2)
{
    RimeSplinteringCell cell;
    cell.temperature_k = 268.15;
    cell.pressure_pa = 80000;
    cell.dry_density_kg_m3 = .65;
    cell.cloud_mass_per_kg = 2e-4;
    const std::array<double, 4> mass{2e-4, 5e-4, 3e-4, 1e-4};
    const std::array<double, 4> number{500, 400, 100, 10};
    for (std::size_t index = 0; index < mass.size(); ++index)
        cell.initial[index] = reference::population(index, mass[index], number[index], shape);
    cell.admitted_cloud_mass_per_kg = {1e-6, 2e-6, 3e-6, 4e-6};
    if (!ice_present)
    {
        cell.initial[0] = {};
        cell.admitted_cloud_mass_per_kg[0] = 0;
    }
    return cell;
}

long double yield(double temperature)
{
    const long double tc = static_cast<long double>(temperature) - static_cast<long double>(273.15);
    if (tc < -8 || tc >= -3)
        return 0;
    return tc <= -5 ? 3.5e8L * (tc + 8) / 3 : 3.5e8L * (-3 - tc) / 2;
}

void near(double actual, long double expected)
{
    INFO("actual " << actual << ", expected " << static_cast<double>(expected));
    CHECK(std::isfinite(actual));
    CHECK(std::abs(static_cast<long double>(actual) - expected)
        <= 2e-12L * std::max(std::abs(expected), 1e-290L));
}

void same_population(const FrozenCollectionPopulation& actual,
                     const FrozenCollectionPopulation& expected)
{
    CHECK(actual.mass_per_kg == expected.mass_per_kg);
    CHECK(actual.number_per_kg == expected.number_per_kg);
    CHECK(actual.reflectivity_per_kg == expected.reflectivity_per_kg);
}

void same_producer(const RimeSplinteringProducer& actual, const RimeSplinteringProducer& expected)
{
    CHECK(actual.event_count() == expected.event_count());
    CHECK(actual.parameters() == expected.parameters());
    CHECK(actual.hail_processes() == expected.hail_processes());
    CHECK(actual.duration_s() == expected.duration_s());
    CHECK(actual.splinters_per_kg_collected() == expected.splinters_per_kg_collected());
    CHECK(actual.mass_increment_per_kg() == expected.mass_increment_per_kg());
    CHECK(actual.number_increment_per_kg() == expected.number_increment_per_kg());
    CHECK(actual.reflectivity_increment_per_kg() == expected.reflectivity_increment_per_kg());
    CHECK(actual.source().temperature_k == expected.source().temperature_k);
    CHECK(actual.source().pressure_pa == expected.source().pressure_pa);
    CHECK(actual.source().dry_density_kg_m3 == expected.source().dry_density_kg_m3);
    CHECK(actual.source().cloud_mass_per_kg == expected.source().cloud_mass_per_kg);
    CHECK(actual.source().admitted_cloud_mass_per_kg == expected.source().admitted_cloud_mass_per_kg);
    for (std::size_t index = 0; index < 4; ++index)
        same_population(actual.source().initial[index], expected.source().initial[index]);
    for (std::size_t index = 0; index < 3; ++index)
    {
        CHECK(actual.channels()[index].species == expected.channels()[index].species);
        CHECK(actual.channels()[index].embryo_mass_per_kg == expected.channels()[index].embryo_mass_per_kg);
        CHECK(actual.channels()[index].number_increment_per_kg == expected.channels()[index].number_increment_per_kg);
        CHECK(actual.channels()[index].reflectivity_increment_per_kg == expected.channels()[index].reflectivity_increment_per_kg);
    }
}

void same_result(const RimeSplinteringResult& actual, const RimeSplinteringResult& expected)
{
    CHECK(actual.mass_increment_per_kg == expected.mass_increment_per_kg);
    CHECK(actual.number_increment_per_kg == expected.number_increment_per_kg);
    CHECK(actual.reflectivity_increment_per_kg == expected.reflectivity_increment_per_kg);
    CHECK(actual.temperature_increment_k == expected.temperature_increment_k);
    CHECK(actual.vapor_mass_increment_per_kg == expected.vapor_mass_increment_per_kg);
    CHECK(actual.cloud_mass_increment_per_kg == expected.cloud_mass_increment_per_kg);
    for (std::size_t index = 0; index < 4; ++index)
    {
        same_population(actual.unrounded[index], expected.unrounded[index]);
        CHECK(actual.rounded[index].mass_per_kg == expected.rounded[index].mass_per_kg);
        CHECK(actual.rounded[index].number_per_kg == expected.rounded[index].number_per_kg);
        CHECK(actual.rounded[index].reflectivity_per_kg == expected.rounded[index].reflectivity_per_kg);
    }
    same_producer(actual.producer, expected.producer);
}

} // namespace

TEST_CASE("Published HM journals use the triangular yield and literal B4 embryo transfers",
          "[physics][milbrandt][triple][rime_splintering]")
{
    constexpr std::array<double, 12> temperatures{
        264.15, 265.15, 265.1501, 266.15, 267.15, 268.15,
        268.1501, 269.15, 270.1499, 270.15, 271.15, 272.15};
    const long double g0 = std::tgamma(7.0L) * std::tgamma(1.0L) /
        (std::tgamma(4.0L) * std::tgamma(4.0L));
    for (bool hail : {false, true})
    {
        for (double density : {.3, .65, 1.2})
        {
            for (double temperature : temperatures)
            {
                auto cell = source();
                cell.temperature_k = temperature;
                cell.dry_density_kg_m3 = density;
                RimeSplinteringProducer produced;
                std::string error;
                REQUIRE(produce_exact_rime_splintering_admission(parameters(hail), cell, .2, produced, error));
                REQUIRE(produced.event_count() == 3);
                const long double f = yield(temperature);
                near(produced.splinters_per_kg_collected(), f);
                long double number = 0, mass = 0, reflectivity = 0;
                for (std::size_t channel = 0; channel < 3; ++channel)
                {
                    const long double n = f * cell.admitted_cloud_mass_per_kg[channel];
                    const long double q = 1e-12L * n;
                    const long double z = n == 0 ? 0 : g0 * q * q / (440.0L * 440.0L * n);
                    near(produced.channels()[channel].number_increment_per_kg, n);
                    near(produced.channels()[channel].embryo_mass_per_kg, q);
                    near(produced.channels()[channel].reflectivity_increment_per_kg, z);
                    number += n;
                    reflectivity += z;
                    if (channel > 0)
                    {
                        mass += q;
                        near(produced.mass_increment_per_kg()[channel], -q);
                        near(produced.reflectivity_increment_per_kg()[channel], -z);
                        CHECK(produced.number_increment_per_kg()[channel] == 0);
                    }
                }
                near(produced.mass_increment_per_kg()[0], mass);
                near(produced.number_increment_per_kg()[0], number);
                near(produced.reflectivity_increment_per_kg()[0], reflectivity);
                CHECK(produced.mass_increment_per_kg()[3] == 0);
                CHECK(produced.number_increment_per_kg()[3] == 0);
                CHECK(produced.reflectivity_increment_per_kg()[3] == 0);
            }
        }
    }
}

TEST_CASE("HM self-ice uses virtual mass and adds no duplicate freezing heat",
          "[physics][milbrandt][triple][rime_splintering]")
{
    auto cell = source();
    cell.admitted_cloud_mass_per_kg = {2e-6, 0, 0, 0};
    RimeSplinteringResult result;
    std::string error;
    REQUIRE(evaluate_exact_triple_rime_splintering_cell(parameters(), cell, .05, result, error));
    REQUIRE(result.producer.channels()[0].embryo_mass_per_kg > 0);
    CHECK(result.mass_increment_per_kg[0] == 0);
    CHECK(result.unrounded[0].mass_per_kg == cell.initial[0].mass_per_kg);
    CHECK(result.number_increment_per_kg[0] > 0);
    CHECK(result.reflectivity_increment_per_kg[0] > 0);
    CHECK(result.temperature_increment_k == 0);
    CHECK(result.vapor_mass_increment_per_kg == 0);
    CHECK(result.cloud_mass_increment_per_kg == 0);
    for (std::size_t index = 1; index < 4; ++index)
        same_population(result.unrounded[index], cell.initial[index]);
    RimeSplinteringResult replayed;
    REQUIRE(apply_exact_triple_rime_splintering_cell(parameters(), cell, .05, result.producer, replayed, error));
    same_result(replayed, result);
}

TEST_CASE("HM initializes empty ice from admitted snow and graupel and subtracts the same seed Z",
          "[physics][milbrandt][triple][rime_splintering]")
{
    const auto cell = source(false);
    RimeSplinteringResult result;
    std::string error;
    REQUIRE(evaluate_exact_triple_rime_splintering_cell(parameters(false), cell, .1, result, error));
    near(result.unrounded[0].mass_per_kg, 1e-12L * 3.5e8L *
        (cell.admitted_cloud_mass_per_kg[1] + cell.admitted_cloud_mass_per_kg[2]));
    const long double q = result.unrounded[0].mass_per_kg;
    const long double n = result.unrounded[0].number_per_kg;
    near(result.unrounded[0].reflectivity_per_kg, 20 * q * q / (440.0L * 440.0L * n));
    long double water = 0;
    for (std::size_t index = 0; index < 4; ++index)
    {
        water += static_cast<long double>(result.unrounded[index].mass_per_kg) - cell.initial[index].mass_per_kg;
        CHECK(result.rounded[index].mass_per_kg == static_cast<float>(result.unrounded[index].mass_per_kg));
        CHECK(result.rounded[index].number_per_kg == static_cast<float>(result.unrounded[index].number_per_kg));
        CHECK(result.rounded[index].reflectivity_per_kg == static_cast<float>(result.unrounded[index].reflectivity_per_kg));
    }
    CHECK(std::abs(water) < 1e-18L);
    for (std::size_t donor : {1u, 2u})
    {
        CHECK(result.reflectivity_increment_per_kg[donor] == -result.producer.channels()[donor].reflectivity_increment_per_kg);
        const long double b3 = -(6 / reference::mass_exponent[donor]) * cell.initial[donor].reflectivity_per_kg *
            result.producer.channels()[donor].embryo_mass_per_kg / cell.initial[donor].mass_per_kg;
        CHECK(std::abs(result.reflectivity_increment_per_kg[donor] - b3) > .9L * std::abs(b3));
    }
    CHECK(result.temperature_increment_k == 0);
    same_population(result.unrounded[3], cell.initial[3]);
}

TEST_CASE("HM captures freshly funded donor transfers before isolated endpoint admission",
          "[physics][milbrandt][triple][rime_splintering]")
{
    auto cell = source(false);
    cell.initial[1] = reference::population(1, 1e-10, 100, 2);
    cell.cloud_mass_per_kg = .002;
    cell.admitted_cloud_mass_per_kg = {0, .002, 0, 0};
    RimeSplinteringProducer producer;
    RimeSplinteringFailure failure;
    std::string error;
    REQUIRE(produce_exact_rime_splintering_admission(parameters(), cell, .2, producer, error, 3, &failure));
    REQUIRE(producer.channels()[1].embryo_mass_per_kg > cell.initial[1].mass_per_kg);
    RimeSplinteringResult sentinel;
    sentinel.temperature_increment_k = -123;
    const auto before = sentinel;
    CHECK_FALSE(apply_exact_triple_rime_splintering_cell(parameters(), cell, .2, producer, sentinel, error, 3, &failure));
    CHECK(failure == RimeSplinteringFailure::IntervalRejected);
    same_result(sentinel, before);
}

TEST_CASE("HM rejects invalid context cloud journals and source moments atomically even with zero yield",
          "[physics][milbrandt][triple][rime_splintering]")
{
    const auto valid = source();
    RimeSplinteringProducer sentinel;
    std::string error;
    REQUIRE(produce_exact_rime_splintering_admission(parameters(), valid, .1, sentinel, error));
    const auto before = sentinel;
    for (int corruption = 0; corruption < 18; ++corruption)
    {
        auto cell = valid;
        cell.temperature_k = 264.15;
        auto p = parameters();
        double dt = .1;
        std::size_t capacity = 3;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        switch (corruption)
        {
        case 0: cell.temperature_k = nan; break;
        case 1: cell.temperature_k = 149; break;
        case 2: cell.temperature_k = 273.15; break;
        case 3: cell.pressure_pa = 99; break;
        case 4: cell.pressure_pa = 200001; break;
        case 5: cell.dry_density_kg_m3 = 0; break;
        case 6: cell.cloud_mass_per_kg = -1; break;
        case 7: cell.admitted_cloud_mass_per_kg[1] = -1e-6; break;
        case 8: cell.admitted_cloud_mass_per_kg[1] = nan; break;
        case 9: cell.admitted_cloud_mass_per_kg[1] = cell.cloud_mass_per_kg; break;
        case 10: cell.initial[0] = {}; break;
        case 11: cell.initial[1].number_per_kg = 0; break;
        case 12: cell.initial[2].reflectivity_per_kg = 0; break;
        case 13: cell.initial[3].mass_per_kg = -1; break;
        case 14: p.c_i_ = 441; break;
        case 15: dt = 0; break;
        case 16: capacity = 2; break;
        case 17: capacity = 4; break;
        }
        RimeSplinteringFailure failure;
        CHECK_FALSE(produce_exact_rime_splintering_admission(p, cell, dt, sentinel, error, capacity, &failure));
        CHECK(failure != RimeSplinteringFailure::None);
        CHECK(failure != RimeSplinteringFailure::IntervalRejected);
        same_producer(sentinel, before);
    }
}

TEST_CASE("HM immutable replay binds every original population accepted journal and interval",
          "[physics][milbrandt][triple][rime_splintering]")
{
    const auto valid = source();
    RimeSplinteringProducer producer;
    std::string error;
    REQUIRE(produce_exact_rime_splintering_admission(parameters(), valid, .1, producer, error));
    RimeSplinteringResult sentinel;
    REQUIRE(evaluate_exact_triple_rime_splintering_cell(parameters(), valid, .1, sentinel, error));
    const auto before = sentinel;
    for (int change = 0; change < 24; ++change)
    {
        auto cell = valid;
        auto p = parameters();
        double dt = .1;
        RimeSplinteringProducer capture = producer;
        if (change < 12)
        {
            auto& population = cell.initial[change / 3];
            double* field = change % 3 == 0 ? &population.mass_per_kg
                : change % 3 == 1 ? &population.number_per_kg : &population.reflectivity_per_kg;
            *field = std::nextafter(*field, std::numeric_limits<double>::infinity());
        }
        else if (change < 16)
            cell.admitted_cloud_mass_per_kg[change - 12] *= 1.001;
        else if (change == 16)
            cell.temperature_k += .001;
        else if (change == 17)
            cell.pressure_pa += 1;
        else if (change == 18)
            cell.dry_density_kg_m3 += .01;
        else if (change == 19)
            cell.cloud_mass_per_kg *= 1.001;
        else if (change == 20)
            dt *= 2;
        else if (change == 21)
            p.hail_processes = false;
        else if (change == 22)
            p.alpha_s_ = -0.0;
        else
            capture = {};
        RimeSplinteringFailure failure;
        CHECK_FALSE(apply_exact_triple_rime_splintering_cell(p, cell, dt, capture, sentinel, error, 3, &failure));
        CHECK(failure == RimeSplinteringFailure::InvalidProducer);
        same_result(sentinel, before);
    }
    for (double dt : {.001, 1.0, 10.0})
    {
        RimeSplinteringProducer other;
        REQUIRE(produce_exact_rime_splintering_admission(parameters(), valid, dt, other, error));
        CHECK(other.mass_increment_per_kg() == producer.mass_increment_per_kg());
        CHECK(other.number_increment_per_kg() == producer.number_increment_per_kg());
        CHECK(other.reflectivity_increment_per_kg() == producer.reflectivity_increment_per_kg());
    }
}
