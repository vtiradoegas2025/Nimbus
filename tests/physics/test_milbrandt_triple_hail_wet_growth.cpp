#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_collected_sll.hpp"
#include "support/milbrandt_hail_wet_growth_reference.hpp"

#include <limits>
#include <thread>

namespace
{
using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_hail_wet_growth_reference;
constexpr auto domain = ColdSpectrumDomain::GraupelAndHail512;

ColdParameters parameters(bool hail = true)
{
    const auto& p = default_cold_parameters;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], hail};
}

ColdRimingCell source(bool rain = true, bool mixed = true, long double shape = 2, double temperature = 272.15)
{
    ColdRimingCell cell{temperature, 85000, .65, {}};
    cell.initial.cloud_mass_per_kg = .005;
    cell.initial.cloud_number_per_kg = 1e8;
    cell.initial.vapor_mass_per_kg = static_cast<double>(reference::cold::riming::thermal::saturation(temperature, cell.pressure_pa).mass);
    if (rain) cell.initial.rain = reference::cold::rain_population(.001, 4000, 2);
    if (mixed)
    {
        cell.initial.frozen[0] = reference::frozen::population(0, 2e-6, 50, 2);
        cell.initial.frozen[1] = reference::frozen::population(1, 2e-5, 50, 2);
        cell.initial.frozen[2] = reference::frozen::population(2, 2e-4, 600, shape);
    }
    cell.initial.frozen[3] = reference::frozen::population(3, .001, 20, shape);
    return cell;
}

void near(double actual, long double expected, long double floor = 1e-30L)
{
    CAPTURE(actual, expected);
    CHECK(std::abs(static_cast<long double>(actual) - expected) <= 3e-8L * std::max(std::abs(expected), floor));
}

ColdCollectedSllResult accepted(const ColdRimingCell& cell, double dt = 1e-4, bool hail = true)
{
    ColdCollectedSllResult result;
    std::string error;
    const bool ok = evaluate_exact_triple_cold_wet_growth_sll_cell(parameters(hail), cell, dt, result, error);
    INFO(error);
    REQUIRE(ok);
    return result;
}

reference::cold::State state(const ColdRimingCell& cell, const ColdCollectedSllResult& result)
{
    return reference::cold::state({cell.temperature_k + result.combined.temperature_increment_k,
        cell.pressure_pa, cell.dry_density_kg_m3, result.combined.unrounded});
}

void compare(const ColdRimingCell& cell, double dt, bool hail, const ColdCollectedSllResult& actual)
{
    const auto expected = reference::evaluate(cell, dt, hail);
    const auto& result = actual.combined;
    const auto& wet = result.producer.wet_growth();
    REQUIRE(result.producer.wet_growth_enabled());
    CHECK(result.producer.event_count() == 13);
    CHECK(wet.event_count() == 1);
    CHECK((wet.regime() == HailWetGrowthRegime::Wet) == expected.thermal.wet);
    CHECK_FALSE(expected.thermal.unsupported);
    near(wet.normalized_ventilation_m(), expected.thermal.ventilation);
    near(wet.atmospheric_capacity_per_kg(), expected.thermal.atmosphere);
    near(wet.frozen_capacity_per_kg(), expected.thermal.frozen);
    near(wet.growth_capacity_per_kg(), expected.thermal.capacity);
    near(wet.retained_liquid_per_kg(), expected.thermal.retained);
    near(wet.shed_mass_per_kg(), expected.thermal.shed);
    near(wet.shed_number_per_kg(), expected.thermal.number);
    near(wet.shed_reflectivity_per_kg(), expected.thermal.z);
    near(result.cloud_mass_increment_per_kg, expected.cloud_increment);
    near(result.cloud_number_increment_per_kg, expected.cloud_number_increment);
    near(result.vapor_mass_increment_per_kg, expected.vapor_increment);
    near(result.rain_mass_increment_per_kg, expected.rain_mass);
    near(result.rain_number_increment_per_kg, expected.rain_number);
    near(result.rain_reflectivity_increment_per_kg, expected.rain_z);
    near(result.temperature_increment_k, expected.temperature_increment);
    long double water = result.cloud_mass_increment_per_kg + static_cast<long double>(result.vapor_mass_increment_per_kg)
        + result.rain_mass_increment_per_kg;
    for (std::size_t index = 0; index < 4; ++index)
    {
        near(result.mass_increment_per_kg[index], expected.mass_increment[index]);
        near(result.number_increment_per_kg[index], expected.number_increment[index]);
        near(result.reflectivity_increment_per_kg[index], expected.z_increment[index]);
        water += result.mass_increment_per_kg[index];
    }
    CHECK(std::abs(water) < 1e-18L);
    near(result.temperature_increment_k * 1004,
        -334000.L * (static_cast<long double>(result.cloud_mass_increment_per_kg) + result.rain_mass_increment_per_kg)
            - 2834000.L * result.vapor_mass_increment_per_kg);
}
}

TEST_CASE("wet hail capacity and cold composition agree with independent spectral and thermal references", "[physics][milbrandt][triple_hail_wet_growth]")
{
    int wet = 0, dry = 0;
    for (double temperature : {272.15, 268.15, 250.15})
    for (long double shape : {2.L, 20.L, 350.L})
    for (bool hail : {false, true})
    {
        const auto cell = source(true, true, shape, temperature);
        const auto result = accepted(cell, 1e-4, hail);
        compare(cell, 1e-4, hail, result);
        if (result.combined.producer.wet_growth().regime() == HailWetGrowthRegime::Wet) ++wet;
        else ++dry;
        ColdCollectedSllResult replay;
        std::string error;
        REQUIRE(apply_exact_triple_cold_wet_growth_sll_cell(parameters(hail), cell, 1e-4, result.collection,
            result.ice_conversion, result.snow_conversion, result.hail_conversion, replay, error));
        CHECK(state(cell, result) == state(cell, replay));
    }
    CHECK(wet > 0);
    CHECK(dry > 0);
    const auto cell = source(true, true, 350);
    const auto expected = reference::evaluate(cell, 1e-4);
    const auto quadrature = reference::thermal(cell, 1e-4, expected.dry_collected, expected.wet_collected, true);
    const auto result = accepted(cell);
    near(result.combined.producer.wet_growth().normalized_ventilation_m(), quadrature.ventilation);
    near(result.combined.producer.wet_growth().shed_reflectivity_per_kg(), quadrature.z);
}

TEST_CASE("wet hail sheds collected cloud into a fresh one millimetre alpha two rain population", "[physics][milbrandt][triple_hail_wet_growth]")
{
    for (bool hail : {false, true})
    {
        const auto cell = source(false, false);
        const auto result = accepted(cell, 1e-4, hail);
        compare(cell, 1e-4, hail, result);
        const auto& wet = result.combined.producer.wet_growth();
        REQUIRE(wet.regime() == HailWetGrowthRegime::Wet);
        REQUIRE(wet.shed_mass_per_kg() > 0);
        CHECK(wet.source().wet_collection_per_kg[1] == 0);
        CHECK(wet.source().wet_collection_per_kg[0] > wet.source().dry_collection_per_kg[0]);
        CHECK(result.combined.unrounded.rain.mass_per_kg == wet.shed_mass_per_kg());
        CHECK(result.combined.unrounded.rain.number_per_kg == wet.shed_number_per_kg());
        CHECK(result.combined.unrounded.rain.reflectivity_per_kg == wet.shed_reflectivity_per_kg());
        near(wet.shed_mass_per_kg() / wet.shed_number_per_kg(), 523.598775598L * 1e-9L);
        double alpha = 0, lambda = 0;
        std::string error;
        REQUIRE(diagnose_triple_gamma_shape(wet.shed_mass_per_kg(), wet.shed_number_per_kg(), wet.shed_reflectivity_per_kg(),
            current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Rain).mass_law, alpha, lambda, error));
        CHECK(alpha == Approx(2).epsilon(1e-10));
        near(wet.retained_liquid_per_kg(), wet.growth_capacity_per_kg());
        CHECK(wet.retained_liquid_per_kg() < wet.source().dry_collection_per_kg[0]);
    }
    const auto cell = source(true, false);
    const auto result = accepted(cell);
    const auto& rain = cell.initial.rain;
    const double wrong_net_closure = rain.reflectivity_per_kg *
        (2 * result.combined.rain_mass_increment_per_kg / rain.mass_per_kg - result.combined.rain_number_increment_per_kg / rain.number_per_kg);
    CHECK(std::abs(wrong_net_closure - result.combined.rain_reflectivity_increment_per_kg) > .01 * std::abs(result.combined.rain_reflectivity_increment_per_kg));
}

TEST_CASE("wet hail preserves dry cold and absent collector limits without spending new hail", "[physics][milbrandt][triple_hail_wet_growth]")
{
    for (int scenario = 0; scenario < 3; ++scenario)
    {
        auto cell = source(true, true, 2, scenario == 1 ? 222.15 : 250.15);
        if (scenario == 0) cell.initial.cloud_mass_per_kg = 1e-7;
        if (scenario == 2) cell.initial.frozen[3] = {};
        const auto result = accepted(cell);
        CHECK(result.combined.producer.wet_growth().regime() != HailWetGrowthRegime::Wet);
        ColdCollectedSllResult previous;
        std::string error;
        REQUIRE(evaluate_exact_triple_cold_collision_sll_cell(parameters(), cell, 1e-4, previous, error));
        CHECK(state(cell, result) == state(cell, previous));
        CHECK(result.combined.mass_increment_per_kg == previous.combined.mass_increment_per_kg);
        CHECK(result.combined.reflectivity_increment_per_kg == previous.combined.reflectivity_increment_per_kg);
    }
}

TEST_CASE("wet hail invalid source policy and thermal journals fail atomically", "[physics][milbrandt][triple_hail_wet_growth]")
{
    const auto cell = source();
    const auto result = accepted(cell);
    auto untouched = result;
    std::string error;
    for (int mutation = 0; mutation < 14; ++mutation)
    {
        auto changed = cell;
        auto p = parameters();
        double dt = 1e-4;
        std::size_t capacity = 16;
        switch (mutation)
        {
        case 0: changed.initial.cloud_mass_per_kg *= 1.01; break;
        case 1: changed.initial.cloud_number_per_kg *= 1.01; break;
        case 2: changed.initial.rain.number_per_kg *= 1.01; break;
        case 3: changed.initial.vapor_mass_per_kg *= 1.01; break;
        case 4: changed.initial.frozen[3].reflectivity_per_kg *= 1.01; break;
        case 5: changed.temperature_k += .1; break;
        case 6: changed.pressure_pa += 1; break;
        case 7: changed.dry_density_kg_m3 *= 1.01; break;
        case 8: p.hail_processes = false; break;
        case 9: dt = 0; break;
        case 10: capacity = 15; break;
        case 11: changed.initial.cloud_mass_per_kg = -1; break;
        case 12: changed.initial.frozen[3].number_per_kg = 0; break;
        case 13: changed.initial.rain.reflectivity_per_kg = std::numeric_limits<double>::quiet_NaN(); break;
        }
        CHECK_FALSE(apply_exact_triple_cold_wet_growth_sll_cell(p, changed, dt, result.collection,
            result.ice_conversion, result.snow_conversion, result.hail_conversion, untouched, error, capacity));
        CHECK(state(cell, untouched) == state(cell, result));
        if (mutation >= 9)
        {
            CHECK_FALSE(evaluate_exact_triple_cold_wet_growth_sll_cell(p, changed, dt, untouched, error, capacity));
            CHECK(state(cell, untouched) == state(cell, result));
        }
    }
    auto detached = result.ice_conversion;
    auto& budget = const_cast<HailWetGrowthCell&>(detached.riming_source().wet_growth().source()).wet_collection_per_kg;
    budget[0] = std::nextafter(budget[0], 1.0);
    CHECK_FALSE(apply_exact_triple_cold_wet_growth_sll_cell(parameters(), cell, 1e-4, result.collection,
        detached, result.snow_conversion, result.hail_conversion, untouched, error));
    CHECK(state(cell, untouched) == state(cell, result));
    auto unsupported = result.combined.producer.wet_growth().source();
    unsupported.source.initial.vapor_mass_per_kg = .2;
    HailWetGrowthProducer journal = result.combined.producer.wet_growth();
    FrozenCollectionFailure failure;
    CHECK_FALSE(produce_exact_hail_wet_growth_admission(parameters(), unsupported, 1e-4, journal, error, 1, &failure, domain));
    CHECK(failure == FrozenCollectionFailure::IntervalRejected);
    CHECK(journal.shed_mass_per_kg() == result.combined.producer.wet_growth().shed_mass_per_kg());
}

TEST_CASE("wet hail continuation is deterministic and converges to an independent coupled integration", "[physics][milbrandt][triple_hail_wet_growth]")
{
    const auto cell = source(true, false);
    const auto advance = [&](ColdRimingCell current, double duration, int steps)
    {
        for (int step = 0; step < steps; ++step)
        {
            const auto result = accepted(current, duration / steps);
            CHECK(result.combined.producer.cloud_admission_factor() == 1);
            current.temperature_k += result.combined.temperature_increment_k;
            current.initial = result.combined.unrounded;
        }
        return current;
    };
    const auto initial = reference::cold::state(cell);
    const auto oracle = reference::rk4(cell, .02, 64);
    const auto norm = [&](const reference::cold::State& values)
    {
        long double error = 0;
        for (std::size_t index = 0; index < values.size(); ++index)
            if (std::abs(oracle[index] - initial[index]) > std::max(1e-32L, 1e-13L * std::abs(initial[index])))
                error = std::max(error, std::abs(values[index] - oracle[index]) / std::abs(oracle[index] - initial[index]));
        return error;
    };
    CHECK(norm(reference::rk4(cell, .02, 128)) < 1e-6L);
    const auto coarse = norm(reference::cold::state(advance(cell, .02, 2)));
    const auto medium = norm(reference::cold::state(advance(cell, .02, 4)));
    CHECK(medium < .55L * coarse);
    CHECK(norm(reference::cold::state(advance(cell, .02, 8))) < .55L * medium);
    CHECK(reference::cold::state(advance(advance(cell, .0004, 4), .0004, 4)) == reference::cold::state(advance(cell, .0008, 8)));
    std::array<reference::cold::State, 16> serial{}, parallel{};
    std::array<bool, 16> ok{};
    const auto run = [&](std::size_t index)
    {
        const auto current = source(index % 2, false, 2 + index);
        ColdCollectedSllResult result;
        std::string error;
        ok[index] = evaluate_exact_triple_cold_wet_growth_sll_cell(parameters(), current, 1e-4, result, error);
        parallel[index] = state(current, result);
    };
    for (std::size_t index = 0; index < 16; ++index) { run(index); REQUIRE(ok[index]); serial[index] = parallel[index]; }
    std::array<std::thread, 4> workers;
    for (std::size_t worker = 0; worker < 4; ++worker)
        workers[worker] = std::thread([&, worker] { for (std::size_t index = worker; index < 16; index += 4) run(index); });
    for (auto& worker : workers) worker.join();
    for (bool value : ok) CHECK(value);
    CHECK(serial == parallel);
}
