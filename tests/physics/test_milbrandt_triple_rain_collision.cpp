#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_collected_sll.hpp"
#include "support/milbrandt_rain_collision_reference.hpp"

#include <limits>
#include <thread>

namespace
{
using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_rain_collision_reference;
constexpr auto domain = ColdSpectrumDomain::GraupelAndHail512;

ColdParameters parameters(bool hail = true)
{
    const auto& p = default_cold_parameters;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], hail};
}

RainCollectionCell source(long double shape = 2, double density = 1)
{
    RainCollectionCell cell;
    cell.frozen = {268.15, 85000, density, {}};
    const std::array<double, 4> mass{2e-5, 1e-4, 2e-4, 2e-4}, number{5000, 5000, 600, 20};
    for (std::size_t index = 0; index < 4; ++index)
        cell.frozen.initial.frozen[index] = reference::frozen::population(index, mass[index], number[index], index >= 2 ? shape : 2);
    cell.rain = reference::rain::rain_population(.002, 4000, std::min(shape, 190.L));
    return cell;
}

void near(double value, long double expected, long double tolerance = 3e-9L)
{
    CAPTURE(value, expected);
    CHECK(std::abs(static_cast<long double>(value) - expected) <= tolerance * std::max(std::abs(expected), 1e-32L));
}

RainCollisionResult accepted(const RainCollectionCell& cell, double dt = 1e-4, bool hail = true)
{
    RainCollisionResult result;
    std::string error;
    const bool ok = evaluate_exact_triple_rain_collision_cell(parameters(hail), cell, dt, result, error, 7, nullptr, domain);
    INFO(error);
    REQUIRE(ok);
    return result;
}

void compare(const RainCollectionCell& cell, double dt, bool hail, const RainCollisionResult& actual)
{
    const auto oracle = reference::evaluate(cell, dt, hail);
    CHECK(actual.producer.event_count() == 7);
    near(actual.increments.rain.mass_per_kg, oracle.mass[0]);
    near(actual.increments.rain.number_per_kg, oracle.number[0]);
    near(actual.increments.rain.reflectivity_per_kg, oracle.z[0]);
    near(actual.increments.temperature_k, oracle.temperature);
    long double water = actual.increments.rain.mass_per_kg;
    long double number = actual.increments.rain.number_per_kg;
    for (std::size_t index = 0; index < 4; ++index)
    {
        near(actual.increments.frozen.mass_increment_per_kg[index], oracle.mass[index + 1]);
        near(actual.increments.frozen.number_increment_per_kg[index], oracle.number[index + 1]);
        near(actual.increments.frozen.reflectivity_increment_per_kg[index], oracle.z[index + 1]);
        water += actual.increments.frozen.mass_increment_per_kg[index];
        number += actual.increments.frozen.number_increment_per_kg[index];
        const auto& endpoint = actual.unrounded.frozen.initial.frozen[index];
        CHECK(actual.rounded_frozen[index].mass_per_kg == static_cast<float>(endpoint.mass_per_kg));
        CHECK(actual.rounded_frozen[index].number_per_kg == static_cast<float>(endpoint.number_per_kg));
        CHECK(actual.rounded_frozen[index].reflectivity_per_kg == static_cast<float>(endpoint.reflectivity_per_kg));
        if (index == 3) continue;
        const auto& pair = actual.producer.channels()[index];
        const auto& expected = oracle.pairs[index];
        near(pair.frozen.raw_mass_per_kg, expected.raw_frozen_mass);
        near(pair.frozen.raw_number_per_kg, expected.raw_number);
        near(pair.admission_factor, expected.factor);
        near(pair.rain.mass_per_kg, expected.rain_mass);
        near(pair.frozen.mass_per_kg, expected.frozen_mass);
        near(pair.product_mass_per_kg, expected.product_mass);
        near(pair.product_number_per_kg, expected.product_number);
        near(pair.product_reflectivity_per_kg, expected.product_z);
        CHECK(static_cast<std::size_t>(pair.destination) == expected.destination + 1);
        CHECK(pair.rain.number_per_kg == pair.frozen.number_per_kg);
        CHECK(pair.product_number_per_kg <= pair.frozen.number_per_kg);
    }
    CHECK(std::abs(water) < 1e-18L);
    CHECK(number <= 0);
    near(actual.increments.temperature_k * 1004, -334000.L * actual.increments.rain.mass_per_kg);
}

ColdRimingCell coupled_source(const RainCollectionCell& leaf)
{
    ColdRimingCell cell{leaf.frozen.temperature_k, leaf.frozen.pressure_pa, leaf.frozen.dry_density_kg_m3, {}};
    cell.initial.frozen = leaf.frozen.initial.frozen;
    cell.initial.rain = leaf.rain;
    cell.initial.cloud_mass_per_kg = .001;
    cell.initial.cloud_number_per_kg = 2e8;
    cell.initial.vapor_mass_per_kg = static_cast<double>(reference::rain::riming::thermal::saturation(cell.temperature_k, cell.pressure_pa).mass);
    return cell;
}
}

TEST_CASE("rain collision pairs integrate both original spectra and route conservative products", "[physics][milbrandt][triple_rain_collision]")
{
    for (long double shape : {2.L, 20.L, 350.L})
    for (double density : {.4, 1., 1.6})
    for (bool hail : {false, true})
    {
        const auto cell = source(shape, density);
        const auto result = accepted(cell, 1e-4, hail);
        compare(cell, 1e-4, hail, result);
        RainCollisionResult replay;
        std::string error;
        REQUIRE(apply_exact_triple_rain_collision_cell(parameters(hail), cell, 1e-4, result.producer,
            replay, error, 7, nullptr, domain));
        CHECK(reference::state(replay.unrounded) == reference::state(result.unrounded));
    }
    const auto cell = source(20);
    const auto integral = reference::evaluate(cell, 1e-4, true, true);
    const auto result = accepted(cell);
    for (std::size_t index = 0; index < 3; ++index)
    {
        near(result.producer.channels()[index].frozen.raw_mass_per_kg, integral.pairs[index].raw_frozen_mass);
        near(result.producer.channels()[index].rain.raw_mass_per_kg, integral.pairs[index].raw_rain_mass);
        near(result.producer.channels()[index].frozen.raw_number_per_kg, integral.pairs[index].raw_number);
    }
}

TEST_CASE("rain collision preserves retained hail size gate number caps and noncubic snow products", "[physics][milbrandt][triple_rain_collision]")
{
    for (double diameter : {.00249999, .0025, .00250001, .004})
    for (bool hail : {false, true})
    {
        auto cell = source();
        cell.frozen.initial.frozen[1] = cell.frozen.initial.frozen[2] = cell.frozen.initial.frozen[3] = {};
        const auto p = parameters(hail);
        cell.rain = reference::rain::rain_population(p.c_r_ * 10 * diameter * diameter * diameter, 10, 2);
        const auto result = accepted(cell, 1e-4, hail);
        const auto& pair = result.producer.channels()[0];
        const auto retained = diagnose_distribution(cell.frozen.dry_density_kg_m3, cell.rain.mass_per_kg,
            cell.rain.number_per_kg, 2, p.c_r_, 3);
        CHECK(pair.rain_mean_mass_diameter_m == retained.mean_mass_diameter);
        REQUIRE(pair.mixed_density_kg_m3 >= 650);
        CHECK(pair.destination == (hail && retained.mean_mass_diameter > .0025 ? TripleMomentSpecies::Hail : TripleMomentSpecies::Graupel));
        REQUIRE(pair.product_number_per_kg <= pair.frozen.number_per_kg);
        const auto destination = static_cast<std::size_t>(pair.destination) - 1;
        REQUIRE(result.unrounded.frozen.initial.frozen[destination].number_per_kg > 0);
        double alpha = 1, lambda = 0;
        std::string error;
        REQUIRE(diagnose_triple_gamma_shape(pair.product_mass_per_kg, pair.product_number_per_kg,
            pair.product_reflectivity_per_kg, current_nimbus_triple_sedimentation_parameters(pair.destination).mass_law,
            alpha, lambda, error));
        CHECK(std::abs(alpha) < 1e-9);
    }
    auto snow = source();
    snow.frozen.initial.frozen = {};
    snow.frozen.initial.frozen[1] = reference::frozen::population(1, .001, 100, 2);
    snow.rain = reference::rain::rain_population(1e-5, 10000, 2);
    const auto result = accepted(snow);
    const auto& pair = result.producer.channels()[1];
    REQUIRE(pair.destination == TripleMomentSpecies::Snow);
    const auto& donor = snow.frozen.initial.frozen[1];
    const auto distribution = diagnose_distribution(snow.frozen.dry_density_kg_m3, donor.mass_per_kg, donor.number_per_kg,
        2, parameters().c_s_, parameters().d_s_);
    const double density = std::min(500.0, parameters().c_s_ / (physical_constants::pi / 6.0) *
        std::pow(distribution.mean_mass_diameter, parameters().d_s_ - 3.0));
    CHECK(pair.destination_density_kg_m3 == density);
    compare(snow, 1e-4, true, result);
    const double cubic_shortcut = 20 * pair.product_mass_per_kg * pair.product_mass_per_kg /
        (parameters().c_s_ * parameters().c_s_ * pair.product_number_per_kg);
    CHECK(std::abs(cubic_shortcut / pair.product_reflectivity_per_kg - 1) > .1);
}

TEST_CASE("rain collision budgets bind both donors before endpoint admission", "[physics][milbrandt][triple_rain_collision]")
{
    auto cell = source();
    for (auto& frozen : cell.frozen.initial.frozen)
    {
        frozen.mass_per_kg *= 1e-6;
        frozen.number_per_kg *= 1e-6;
        frozen.reflectivity_per_kg *= 1e-6;
    }
    const auto retained = accepted(cell);
    RainCollisionProducer limited;
    std::string error;
    REQUIRE(produce_exact_rain_collision_admission(parameters(), cell, 1e5, limited, error, 7, nullptr, domain));
    bool frozen_limited = false;
    long double mass = limited.hail_collection().mass_per_kg, number = limited.hail_collection().number_per_kg;
    for (std::size_t index = 0; index < 3; ++index)
    {
        const auto& pair = limited.channels()[index];
        mass += pair.rain.mass_per_kg;
        number += pair.rain.number_per_kg;
        CHECK(pair.frozen.mass_per_kg <= cell.frozen.initial.frozen[index].mass_per_kg);
        CHECK(pair.frozen.number_per_kg <= cell.frozen.initial.frozen[index].number_per_kg);
        CHECK(pair.rain.number_per_kg == pair.frozen.number_per_kg);
        CHECK(pair.admission_factor <= limited.rain_source().admission_factor());
        frozen_limited |= pair.admission_factor < limited.rain_source().admission_factor();
    }
    CHECK(mass <= cell.rain.mass_per_kg);
    CHECK(number <= cell.rain.number_per_kg);
    CHECK(frozen_limited);
    auto unpublished = retained;
    CHECK_FALSE(apply_exact_triple_rain_collision_cell(parameters(), cell, 1e5, limited, unpublished, error, 7, nullptr, domain));
    CHECK(reference::state(unpublished.unrounded) == reference::state(retained.unrounded));
    for (int mutation = 0; mutation < 14; ++mutation)
    {
        auto changed = cell;
        auto p = parameters();
        double dt = 1e-4;
        std::size_t capacity = 7;
        switch (mutation)
        {
        case 0: changed.rain.mass_per_kg *= 1.01; break;
        case 1: changed.rain.number_per_kg *= 1.01; break;
        case 2: changed.rain.reflectivity_per_kg *= 1.01; break;
        case 3: changed.frozen.initial.frozen[0].number_per_kg *= 1.01; break;
        case 4: changed.frozen.pressure_pa += 1; break;
        case 5: changed.frozen.temperature_k += .1; break;
        case 6: changed.frozen.dry_density_kg_m3 *= 1.01; break;
        case 7: p.hail_processes = false; break;
        case 8: p.c_s_ += .001; break;
        case 9: dt *= 2; break;
        case 10: capacity = 6; break;
        case 11: changed.rain.reflectivity_per_kg = std::numeric_limits<double>::quiet_NaN(); break;
        case 12: changed.frozen.initial.frozen[0].mass_per_kg = -1; break;
        case 13: changed.rain.number_per_kg = 0; break;
        }
        CHECK_FALSE(apply_exact_triple_rain_collision_cell(p, changed, dt, retained.producer, unpublished, error, capacity, nullptr, domain));
        CHECK(reference::state(unpublished.unrounded) == reference::state(retained.unrounded));
        if (mutation == 8 || mutation >= 10)
        {
            CHECK_FALSE(evaluate_exact_triple_rain_collision_cell(p, changed, dt, unpublished, error, capacity, nullptr, domain));
            CHECK(reference::state(unpublished.unrounded) == reference::state(retained.unrounded));
        }
    }
    auto forged = retained.producer;
    auto& pair = const_cast<RainCollisionChannel&>(forged.channels()[0]);
    pair.product_number_per_kg *= 1.01;
    CHECK_FALSE(apply_exact_triple_rain_collision_cell(parameters(), cell, 1e-4, forged, unpublished, error, 7, nullptr, domain));
    CHECK(reference::state(unpublished.unrounded) == reference::state(retained.unrounded));
}

TEST_CASE("rain collision endpoints continue deterministically and refine toward independent evolution", "[physics][milbrandt][triple_rain_collision]")
{
    const auto cell = source();
    const auto advance = [&](RainCollectionCell current, int steps, double duration)
    {
        for (int step = 0; step < steps; ++step) current = accepted(current, duration / steps).unrounded;
        return current;
    };
    const auto initial = reference::state(cell);
    const auto target = reference::rk4(cell, .002, 64);
    const auto norm = [&](const reference::State& state)
    {
        long double error = 0;
        for (std::size_t index = 0; index < state.size(); ++index)
            if (std::abs(target[index] - initial[index]) > 1e-14L * std::abs(initial[index]))
                error = std::max(error, std::abs(state[index] - target[index]) / std::abs(target[index] - initial[index]));
        return error;
    };
    CHECK(norm(reference::rk4(cell, .002, 128)) < 1e-6L);
    const auto coarse = norm(reference::state(advance(cell, 2, .002)));
    const auto medium = norm(reference::state(advance(cell, 4, .002)));
    CHECK(medium < .55L * coarse);
    CHECK(norm(reference::state(advance(cell, 8, .002))) < .55L * medium);
    const auto mid = advance(cell, 4, .0004);
    CHECK(reference::state(advance(mid, 4, .0004)) == reference::state(advance(cell, 8, .0008)));
    std::array<reference::State, 16> serial{}, parallel{};
    std::array<bool, 16> ok{};
    const auto evaluate = [&](std::size_t index)
    {
        RainCollisionResult result;
        std::string error;
        const auto source_cell = source(2 + index);
        ok[index] = evaluate_exact_triple_rain_collision_cell(parameters(), source_cell, 1e-4, result, error, 7, nullptr, domain);
        parallel[index] = reference::state(result.unrounded);
    };
    for (std::size_t index = 0; index < 16; ++index) { evaluate(index); REQUIRE(ok[index]); serial[index] = parallel[index]; }
    std::array<std::thread, 4> workers;
    for (std::size_t worker = 0; worker < 4; ++worker)
        workers[worker] = std::thread([&, worker] { for (std::size_t index = worker; index < 16; index += 4) evaluate(index); });
    for (auto& worker : workers) worker.join();
    CHECK(serial == parallel);
    for (bool result : ok) CHECK(result);
}

TEST_CASE("cold collision SLL joins pair products once and limits all original frozen donors", "[physics][milbrandt][triple_rain_collision]")
{
    for (bool hail : {false, true})
    for (double temperature : {272.15, 268.15, 238.15})
    {
        auto cell = coupled_source(source());
        cell.temperature_k = temperature;
        cell.initial.vapor_mass_per_kg = static_cast<double>(reference::rain::riming::thermal::saturation(temperature, cell.pressure_pa).mass);
        ColdCollectedSllResult result;
        std::string error;
        const bool ok = evaluate_exact_triple_cold_collision_sll_cell(parameters(hail), cell, 1e-4, result, error);
        INFO(error);
        REQUIRE(ok);
        REQUIRE(result.combined.producer.rain_collision_enabled());
        CHECK(result.combined.producer.event_count() == 12);
        const auto expected = reference::evaluate_coupled(cell, 1e-4, hail);
        REQUIRE(reference::rain::prefix::original_donors_admissible(cell, expected));
        near(result.combined.temperature_increment_k, expected.temperature_increment);
        near(result.combined.vapor_mass_increment_per_kg, expected.vapor_increment);
        near(result.combined.cloud_mass_increment_per_kg, expected.cloud_increment);
        near(result.combined.rain_mass_increment_per_kg, expected.rain_mass);
        near(result.combined.rain_number_increment_per_kg, expected.rain_number);
        near(result.combined.rain_reflectivity_increment_per_kg, expected.rain_z);
        for (std::size_t index = 0; index < 4; ++index)
        {
            near(result.combined.mass_increment_per_kg[index], expected.mass_increment[index]);
            near(result.combined.number_increment_per_kg[index], expected.number_increment[index]);
            near(result.combined.reflectivity_increment_per_kg[index], expected.z_increment[index]);
        }
        ColdCollectedSllResult replay;
        REQUIRE(apply_exact_triple_cold_collision_sll_cell(parameters(hail), cell, 1e-4, result.collection,
            result.ice_conversion, result.snow_conversion, result.hail_conversion, replay, error));
        CHECK(result.combined.mass_increment_per_kg == replay.combined.mass_increment_per_kg);
        CHECK(result.combined.reflectivity_increment_per_kg == replay.combined.reflectivity_increment_per_kg);
        auto changed = cell;
        changed.initial.rain.number_per_kg *= 1.01;
        CHECK_FALSE(apply_exact_triple_cold_collision_sll_cell(parameters(hail), changed, 1e-4, result.collection,
            result.ice_conversion, result.snow_conversion, result.hail_conversion, replay, error));
        CHECK(result.combined.mass_increment_per_kg == replay.combined.mass_increment_per_kg);
    }
}

TEST_CASE("collision SLL excludes rain routed directly to hail and rejects joint original graupel overspend", "[physics][milbrandt][triple_rain_collision]")
{
    auto leaf = source();
    leaf.rain = reference::rain::rain_population(.002, .002 / (parameters().c_r_ * .003 * .003 * .003), 2);
    auto cell = coupled_source(leaf);
    cell.temperature_k = 272.15;
    cell.initial.cloud_mass_per_kg = .005;
    cell.initial.vapor_mass_per_kg = static_cast<double>(reference::rain::riming::thermal::saturation(cell.temperature_k, cell.pressure_pa).mass);
    cell.initial.frozen[0] = cell.initial.frozen[1] = cell.initial.frozen[3] = {};
    ColdCollectedSllResult result;
    std::string error;
    REQUIRE(evaluate_exact_triple_cold_collision_sll_cell(parameters(), cell, 1e-4, result, error));
    const auto& pair = result.combined.producer.rain_collision().channels()[2];
    REQUIRE(pair.destination == TripleMomentSpecies::Hail);
    REQUIRE(pair.rain.mass_per_kg > 0);
    REQUIRE(result.hail_conversion.source().admitted_accretion_mass_per_kg[1] == 0);
    REQUIRE(result.hail_conversion.admitted_mass_transfer_per_kg() > 0);
    const auto expected = reference::evaluate_coupled(cell, 1e-4);
    for (std::size_t index = 0; index < 4; ++index)
    {
        near(result.combined.mass_increment_per_kg[index], expected.mass_increment[index]);
        near(result.combined.number_increment_per_kg[index], expected.number_increment[index]);
        near(result.combined.reflectivity_increment_per_kg[index], expected.z_increment[index]);
    }
    double duration = static_cast<double>(1.001L * cell.initial.frozen[2].mass_per_kg * 1e-4L / expected.original_mass_sink[2]);
    auto limit = reference::evaluate_coupled(cell, duration);
    for (int attempt = 0; attempt < 16 && limit.original_mass_sink[2] <= cell.initial.frozen[2].mass_per_kg; ++attempt)
    {
        duration *= 1.1;
        limit = reference::evaluate_coupled(cell, duration);
    }
    ColdRimingProducer riming;
    const bool produced = produce_exact_cold_riming_admission(parameters(), cell, duration, riming, error, 12, nullptr, domain, true, true, true);
    INFO(error);
    REQUIRE(produced);
    REQUIRE(limit.original_mass_sink[2] > cell.initial.frozen[2].mass_per_kg);
    REQUIRE(riming.rain_collision().channels()[2].frozen.mass_per_kg < cell.initial.frozen[2].mass_per_kg);
    REQUIRE(limit.sll.admitted + std::max(-limit.diffusion[2], 0.L) < cell.initial.frozen[2].mass_per_kg);
    auto untouched = result;
    ColdRimingFailure failure;
    CHECK_FALSE(evaluate_exact_triple_cold_collision_sll_cell(parameters(), cell, duration, untouched, error, 15, &failure));
    CHECK(failure == ColdRimingFailure::IntervalRejected);
    CHECK(error.find("original frozen") != std::string::npos);
    CHECK(untouched.combined.mass_increment_per_kg == result.combined.mass_increment_per_kg);
}

TEST_CASE("collision SLL preserves the zero rain composition and rejects absent nested captures atomically", "[physics][milbrandt][triple_rain_collision]")
{
    auto cell = coupled_source(source());
    cell.initial.rain = {};
    ColdCollectedSllResult result, previous;
    std::string error;
    REQUIRE(evaluate_exact_triple_cold_collision_sll_cell(parameters(), cell, 1e-4, result, error));
    REQUIRE(evaluate_exact_triple_cold_collected_sll_cell(parameters(), cell, 1e-4, previous, error));
    CHECK(reference::rain::state({cell.temperature_k + result.combined.temperature_increment_k,
        cell.pressure_pa, cell.dry_density_kg_m3, result.combined.unrounded}) ==
        reference::rain::state({cell.temperature_k + previous.combined.temperature_increment_k,
            cell.pressure_pa, cell.dry_density_kg_m3, previous.combined.unrounded}));
    CHECK(result.combined.mass_increment_per_kg == previous.combined.mass_increment_per_kg);
    CHECK(result.combined.number_increment_per_kg == previous.combined.number_increment_per_kg);
    CHECK(result.combined.reflectivity_increment_per_kg == previous.combined.reflectivity_increment_per_kg);
    auto unpublished = result;
    CHECK_FALSE(apply_exact_triple_cold_collision_sll_cell(parameters(), cell, 1e-4, {}, {}, {}, {}, unpublished, error));
    CHECK(unpublished.combined.mass_increment_per_kg == result.combined.mass_increment_per_kg);
    cell.initial.rain = source().rain;
    REQUIRE(evaluate_exact_triple_cold_collision_sll_cell(parameters(), cell, 1e-4, result, error));
    unpublished = result;
    auto detached = result.ice_conversion;
    auto& capture = const_cast<RainCollisionChannel&>(detached.riming_source().rain_collision().channels()[0]);
    capture.frozen.mass_per_kg = std::nextafter(capture.frozen.mass_per_kg, 1.0);
    CHECK_FALSE(apply_exact_triple_cold_collision_sll_cell(parameters(), cell, 1e-4, result.collection,
        detached, result.snow_conversion, result.hail_conversion, unpublished, error));
    CHECK(unpublished.combined.mass_increment_per_kg == result.combined.mass_increment_per_kg);
    CHECK_FALSE(evaluate_exact_triple_cold_collision_sll_cell(parameters(), cell, 1e-4, unpublished, error, 14));
    CHECK(unpublished.combined.mass_increment_per_kg == result.combined.mass_increment_per_kg);
}
