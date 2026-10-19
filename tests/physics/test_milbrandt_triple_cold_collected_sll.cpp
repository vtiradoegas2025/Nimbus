#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_collected_sll.hpp"
#include "support/milbrandt_cold_collected_sll_reference.hpp"

#include <cmath>
#include <limits>

namespace
{

using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_cold_collected_sll_reference;
constexpr auto domain = ColdSpectrumDomain::GraupelAndHail512;

ColdParameters parameters(bool hail = true)
{
    const auto& p = default_cold_parameters;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], hail};
}

ColdRimingCell source(double temperature = 272.15, long double high_shape = 2)
{
    ColdRimingCell cell;
    cell.temperature_k = temperature;
    cell.pressure_pa = 85000;
    cell.dry_density_kg_m3 = .65;
    cell.initial.vapor_mass_per_kg = static_cast<double>(
        reference::riming::thermal::saturation(temperature, cell.pressure_pa).mass);
    cell.initial.cloud_mass_per_kg = .005;
    cell.initial.cloud_number_per_kg = 2e8;
    constexpr std::array<double, 4> masses{2e-5, 1e-4, 2e-4, 2e-4};
    constexpr std::array<double, 4> numbers{5000, 5000, 600, 20};
    for (std::size_t species = 0; species < 4; ++species)
    {
        cell.initial.frozen[species] = reference::frozen::population(species,
            masses[species], numbers[species], species >= 2 ? high_shape : 2);
    }
    return cell;
}

void near(double actual, long double expected)
{
    INFO("actual=" << actual << ", independent=" << static_cast<double>(expected));
    REQUIRE(std::isfinite(actual));
    CHECK(std::abs(static_cast<long double>(actual) - expected)
        <= 6e-8L * std::max(std::abs(expected), 1e-35L));
}

ColdCollectedSllResult accepted(const ColdRimingCell& cell, double duration = 1e-4, bool hail = true)
{
    ColdCollectedSllResult result;
    std::string error;
    const bool success = evaluate_exact_triple_cold_collected_sll_cell(parameters(hail), cell,
        duration, result, error);
    INFO(error);
    REQUIRE(success);
    return result;
}

void same_result(const ColdCollectedSllResult& actual, const ColdCollectedSllResult& expected)
{
    CHECK(reference::riming::state({0, 85000, .65, actual.combined.unrounded})
        == reference::riming::state({0, 85000, .65, expected.combined.unrounded}));
    CHECK(actual.combined.mass_increment_per_kg == expected.combined.mass_increment_per_kg);
    CHECK(actual.combined.number_increment_per_kg == expected.combined.number_increment_per_kg);
    CHECK(actual.combined.reflectivity_increment_per_kg == expected.combined.reflectivity_increment_per_kg);
    CHECK(actual.combined.cloud_mass_increment_per_kg == expected.combined.cloud_mass_increment_per_kg);
    CHECK(actual.combined.cloud_number_increment_per_kg == expected.combined.cloud_number_increment_per_kg);
    CHECK(actual.combined.vapor_mass_increment_per_kg == expected.combined.vapor_mass_increment_per_kg);
    CHECK(actual.combined.temperature_increment_k == expected.combined.temperature_increment_k);
    CHECK(actual.collection.event_count() == expected.collection.event_count());
    CHECK(actual.hail_conversion.mass_increment_per_kg() == expected.hail_conversion.mass_increment_per_kg());
}

void compare(const ColdRimingCell& cell, double duration, bool hail,
             const ColdCollectedSllResult& actual)
{
    const auto expected = reference::evaluate(cell, duration, hail);
    REQUIRE(reference::original_donors_admissible(cell, expected));
    REQUIRE(actual.collection.event_count() == 5);
    CHECK(actual.collection.spectrum_domain() == domain);
    CHECK(actual.combined.producer.spectrum_domain() == domain);
    REQUIRE(actual.combined.producer.homogeneous_freezing_enabled());
    CHECK(actual.combined.producer.event_count() == 5);
    near(actual.combined.cloud_mass_increment_per_kg, expected.cloud_increment);
    near(actual.combined.cloud_number_increment_per_kg, expected.cloud_number_increment);
    near(actual.combined.vapor_mass_increment_per_kg, expected.vapor_increment);
    near(actual.combined.temperature_increment_k, expected.temperature_increment);
    near(actual.hail_conversion.raw_mass_transfer_per_kg(), expected.sll.raw);
    near(actual.hail_conversion.admitted_mass_transfer_per_kg(), expected.sll.admitted);
    CHECK(actual.hail_conversion.source().rain_mass_per_kg == 0);
    CHECK(actual.hail_conversion.source().admitted_accretion_mass_per_kg == std::array<double, 3>{
        actual.combined.producer.channels()[2].collection_mass_per_kg, 0,
        actual.collection.channels()[1].admitted_mass_increment_per_kg});
    long double water = actual.combined.cloud_mass_increment_per_kg
        + static_cast<long double>(actual.combined.vapor_mass_increment_per_kg);
    for (std::size_t species = 0; species < 4; ++species)
    {
        near(actual.combined.mass_increment_per_kg[species], expected.mass_increment[species]);
        near(actual.combined.number_increment_per_kg[species], expected.number_increment[species]);
        near(actual.combined.reflectivity_increment_per_kg[species], expected.z_increment[species]);
        water += actual.combined.mass_increment_per_kg[species];
        CHECK(actual.combined.rounded[species].mass_per_kg
            == static_cast<float>(actual.combined.unrounded.frozen[species].mass_per_kg));
        CHECK(actual.combined.rounded[species].number_per_kg
            == static_cast<float>(actual.combined.unrounded.frozen[species].number_per_kg));
        CHECK(actual.combined.rounded[species].reflectivity_per_kg
            == static_cast<float>(actual.combined.unrounded.frozen[species].reflectivity_per_kg));
    }
    CHECK(std::abs(water) <= 1e-15L * (cell.initial.cloud_mass_per_kg + cell.initial.vapor_mass_per_kg));
    for (std::size_t channel = 0; channel < 5; ++channel)
    {
        const auto& journal = actual.collection.channels()[channel];
        const auto& oracle = expected.collection.channels[channel];
        near(journal.raw_mass_increment_per_kg, oracle.raw_mass);
        near(journal.raw_number_increment_per_kg, oracle.raw_number);
        near(journal.admitted_mass_increment_per_kg, oracle.mass);
        near(journal.admitted_number_increment_per_kg, oracle.number);
    }
}

reference::State integrate(const ColdRimingCell& source_cell, double duration, int steps)
{
    auto cell = source_cell;
    for (int step = 0; step < steps; ++step)
    {
        const auto result = accepted(cell, duration / steps);
        REQUIRE(result.combined.producer.cloud_admission_factor() == 1);
        cell.temperature_k += result.combined.temperature_increment_k;
        cell.initial = result.combined.unrounded;
    }
    return reference::riming::state(cell);
}

long double difference(const reference::State& a, const reference::State& b,
                       const reference::State& initial, const reference::State& target)
{
    long double value = 0;
    for (std::size_t index = 0; index < a.size(); ++index)
    {
        const long double floor = index == 0 ? 1e-8L : index == 3 || (index >= 4 && index % 3 == 2)
            ? 1e-8L : index >= 4 && index % 3 == 0 ? 1e-38L : 1e-22L;
        value = std::max(value, std::abs(a[index] - b[index])
            / std::max(std::abs(target[index] - initial[index]), floor));
    }
    return value;
}

} // namespace

TEST_CASE("collected SLL composes independently integrated original frozen spectra and five way cloud moments once",
          "[physics][milbrandt][triple_cold_collected_sll]")
{
    for (double temperature : {272.15, 268.15, 238.15, 222.15})
    {
        for (bool hail : {false, true})
        {
            const auto cell = source(temperature);
            const auto result = accepted(cell, 1e-4, hail);
            compare(cell, 1e-4, hail, result);
            ColdCollectedSllResult replay;
            std::string error;
            REQUIRE(apply_exact_triple_cold_collected_sll_cell(parameters(hail), cell, 1e-4,
                result.collection, result.ice_conversion, result.snow_conversion, result.hail_conversion,
                replay, error));
            same_result(replay, result);
        }
    }
    const auto cell = source();
    const auto analytic = reference::evaluate(cell, 1e-4);
    const auto integral = reference::evaluate(cell, 1e-4, true, false, true);
    for (std::size_t channel = 0; channel < 5; ++channel)
    {
        near(static_cast<double>(analytic.collection.channels[channel].raw_mass), integral.collection.channels[channel].raw_mass);
        near(static_cast<double>(analytic.collection.channels[channel].raw_number), integral.collection.channels[channel].raw_number);
    }
}

TEST_CASE("collected SLL feeds admitted original ice accretion with zero cloud accretion and no collection heat",
          "[physics][milbrandt][triple_cold_collected_sll]")
{
    auto cell = source(microphysics_constants::T0 - 1e-5);
    cell.initial.cloud_mass_per_kg = 0;
    cell.initial.frozen[0] = reference::frozen::population(0, 4e-8, 100, 2);
    cell.initial.frozen[1] = {};
    cell.initial.frozen[3] = {};
    const auto result = accepted(cell);
    REQUIRE(result.combined.producer.channels()[2].collection_mass_per_kg == 0);
    REQUIRE(result.collection.channels()[1].admitted_mass_increment_per_kg > 0);
    REQUIRE(result.hail_conversion.admitted_mass_transfer_per_kg() > 0);
    compare(cell, 1e-4, true, result);
    const auto prefix = reference::homogeneous::evaluate(cell, 1e-4);
    near(result.combined.temperature_increment_k, prefix.temperature_increment);
    CHECK(result.ice_conversion.admitted_mass_transfer_per_kg() == 0);
    CHECK(result.snow_conversion.admitted_mass_transfer_per_kg() == 0);
}

TEST_CASE("collected SLL rejects joint original donor overspend rather than spending fresh frozen cloud mass",
          "[physics][milbrandt][triple_cold_collected_sll]")
{
    auto cell = source(237.15);
    cell.initial.frozen[0] = reference::frozen::population(0, 1e-6, 1000, 2);
    cell.initial.frozen[1] = reference::frozen::population(1, .04, .01, 2);
    cell.initial.frozen[2] = {};
    cell.initial.frozen[3] = {};
    bool demonstrated = false;
    for (double duration : {10.0, 100.0, 1000.0})
    {
        const auto expected = reference::evaluate(cell, duration);
        if (expected.relaxation <= 1 && !reference::original_donors_admissible(cell, expected))
        {
            demonstrated = true;
            REQUIRE(expected.admitted_freezing[0] > 0);
            ColdCollectedSllResult result;
            result.combined.cloud_mass_increment_per_kg = 42;
            std::string error;
            ColdRimingFailure failure = ColdRimingFailure::None;
            REQUIRE_FALSE(evaluate_exact_triple_cold_collected_sll_cell(parameters(), cell, duration,
                result, error, 8, &failure));
            CHECK(failure == ColdRimingFailure::IntervalRejected);
            CHECK(error.find("original") != std::string::npos);
            CHECK(result.combined.cloud_mass_increment_per_kg == 42);
            break;
        }
    }
    REQUIRE(demonstrated);
}

TEST_CASE("collected SLL replay binds all source parameters feature domain and exact eight journal capacity atomically",
          "[physics][milbrandt][triple_cold_collected_sll]")
{
    const auto cell = source();
    const auto original = accepted(cell);
    auto output = original;
    std::string error;
    ColdRimingFailure failure = ColdRimingFailure::None;
    for (std::size_t capacity : {0u, 3u, 5u, 7u, 9u})
    {
        CHECK_FALSE(apply_exact_triple_cold_collected_sll_cell(parameters(), cell, 1e-4,
            original.collection, original.ice_conversion, original.snow_conversion, original.hail_conversion,
            output, error, capacity, &failure));
        CHECK(failure == ColdRimingFailure::Capacity);
        same_result(output, original);
    }
    auto changed = cell;
    changed.initial.frozen[0].number_per_kg = std::nextafter(changed.initial.frozen[0].number_per_kg,
        std::numeric_limits<double>::infinity());
    CHECK_FALSE(apply_exact_triple_cold_collected_sll_cell(parameters(), changed, 1e-4,
        original.collection, original.ice_conversion, original.snow_conversion, original.hail_conversion,
        output, error, 8, &failure));
    CHECK(failure == ColdRimingFailure::InvalidProducer);
    same_result(output, original);
    CHECK_FALSE(apply_exact_triple_cold_collected_sll_cell(parameters(false), cell, 1e-4,
        original.collection, original.ice_conversion, original.snow_conversion, original.hail_conversion,
        output, error, 8, &failure));
    CHECK(failure == ColdRimingFailure::InvalidProducer);
    same_result(output, original);
    FrozenCollectionProducer foreign;
    const FrozenCollectionCell frozen_cell{cell.temperature_k, cell.pressure_pa, cell.dry_density_kg_m3,
        {cell.initial.frozen}};
    REQUIRE(produce_exact_frozen_collection_admission(parameters(), frozen_cell, 1e-4, foreign, error));
    CHECK_FALSE(apply_exact_triple_cold_collected_sll_cell(parameters(), cell, 1e-4,
        foreign, original.ice_conversion, original.snow_conversion, original.hail_conversion,
        output, error, 8, &failure));
    CHECK(failure == ColdRimingFailure::InvalidProducer);
    same_result(output, original);
    for (std::size_t index = 0; index < default_cold_parameters.size(); ++index)
    {
        auto p = parameters();
        std::array<double*, 25> values{&p.a_g_, &p.a_h_, &p.a_i_, &p.a_r_, &p.a_s_, &p.alpha_g_, &p.alpha_h_,
            &p.alpha_i_, &p.alpha_r_, &p.alpha_s_, &p.b_g_, &p.b_h_, &p.b_i_, &p.b_r_, &p.b_s_,
            &p.c_g_, &p.c_h_, &p.c_i_, &p.c_r_, &p.c_s_, &p.d_g_, &p.d_h_, &p.d_i_, &p.d_r_, &p.d_s_};
        *values[index] = std::nextafter(*values[index], std::numeric_limits<double>::infinity());
        CHECK_FALSE(evaluate_exact_triple_cold_collected_sll_cell(p, cell, 1e-4, output, error, 8, &failure));
        same_result(output, original);
    }
}

TEST_CASE("collected SLL retains higher G H sources and source only collection increments without changing old leaf defaults",
          "[physics][milbrandt][triple_cold_collected_sll]")
{
    auto cell = source(272.15, 300);
    cell.initial.frozen[0] = {};
    cell.initial.frozen[1] = {};
    for (double duration : {1e-4, 2e-4})
    {
        const auto result = accepted(cell, duration);
        compare(cell, duration, true, result);
        const FrozenCollectionCell frozen_cell{cell.temperature_k, cell.pressure_pa, cell.dry_density_kg_m3,
            {cell.initial.frozen}};
        FrozenCollectionProducer producer;
        std::string error;
        FrozenCollectionFailure failure = FrozenCollectionFailure::None;
        CHECK_FALSE(produce_exact_frozen_collection_admission(parameters(), frozen_cell, duration, producer, error));
        REQUIRE(produce_exact_frozen_collection_admission(parameters(), frozen_cell, duration, producer, error, 5, &failure, domain));
        FrozenCollectionIncrements increments;
        REQUIRE(evaluate_exact_frozen_collection_increments(parameters(), frozen_cell, duration, producer,
            increments, error, 5, &failure, domain));
        auto changed = frozen_cell;
        changed.temperature_k = microphysics_constants::T0;
        increments.mass_increment_per_kg[0] = 42;
        CHECK_FALSE(evaluate_exact_frozen_collection_increments(parameters(), changed, duration, producer,
            increments, error, 5, &failure, domain));
        CHECK(increments.mass_increment_per_kg[0] == 42);
        CHECK_FALSE(validate_exact_frozen_collection_admission(parameters(), frozen_cell, duration, producer,
            error, 5, &failure, ColdSpectrumDomain::Base200));
        cell.temperature_k += result.combined.temperature_increment_k;
        cell.initial = result.combined.unrounded;
        for (std::size_t species : {2u, 3u})
        {
            const long double alpha = reference::frozen::represented_alpha(cell.initial.frozen[species], species);
            CHECK(alpha > 200);
            CHECK(alpha <= 512);
        }
    }
}

TEST_CASE("collected SLL full state source rate integration refines against independent compensated RK4",
          "[physics][milbrandt][triple_cold_collected_sll]")
{
    const auto cell = source();
    constexpr double duration = .01;
    const auto initial = reference::riming::state(cell);
    const auto target = reference::rk4(cell, duration, 2048);
    const auto target_half = reference::rk4(cell, duration, 1024);
    const auto coarse = integrate(cell, duration, 16);
    const auto medium = integrate(cell, duration, 32);
    const auto fine = integrate(cell, duration, 64);
    const long double e0 = difference(coarse, target, initial, target);
    const long double e1 = difference(medium, target, initial, target);
    const long double e2 = difference(fine, target, initial, target);
    CAPTURE(e0, e1, e2);
    REQUIRE(e0 > 1e-9L);
    CHECK(e1 < .75L * e0);
    CHECK(e2 < .75L * e1);
    CHECK(difference(target_half, target, initial, target) < 1e-8L);
}
