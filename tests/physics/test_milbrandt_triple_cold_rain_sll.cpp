#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_collected_sll.hpp"
#include "microphysics/milbrandt_triple_scope.hpp"
#include "support/milbrandt_cold_rain_sll_reference.hpp"

#include <limits>
#include <thread>

namespace
{
using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_cold_rain_sll_reference;
constexpr auto domain = ColdSpectrumDomain::GraupelAndHail512;

ColdParameters parameters(bool hail = true)
{
    const auto& p = default_cold_parameters;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], hail};
}

ColdRimingCell source(double temperature = 272.15, long double shape = 2)
{
    ColdRimingCell cell;
    cell.temperature_k = temperature;
    cell.pressure_pa = 85000;
    cell.dry_density_kg_m3 = .65;
    cell.initial.vapor_mass_per_kg = static_cast<double>(reference::riming::thermal::saturation(temperature, 85000).mass);
    cell.initial.cloud_mass_per_kg = .005;
    cell.initial.cloud_number_per_kg = 2e8;
    constexpr std::array<double, 4> masses{2e-5, 1e-4, 2e-4, 2e-4};
    constexpr std::array<double, 4> numbers{5000, 5000, 600, 20};
    for (std::size_t index = 0; index < 4; ++index)
        cell.initial.frozen[index] = reference::frozen::population(index, masses[index], numbers[index],
            index >= 2 ? shape : 2);
    cell.initial.rain = reference::rain_population(.002, 4000, std::min(shape, 190.0L));
    return cell;
}

void near(double actual, long double expected, long double relative = 3e-9L)
{
    CAPTURE(actual, expected);
    CHECK(std::abs(static_cast<long double>(actual) - expected) <= relative * std::abs(expected) + 1e-35L);
}

ColdCollectedSllResult accepted(const ColdRimingCell& cell, double dt, bool hail = true)
{
    ColdCollectedSllResult result;
    std::string error;
    const bool ok = evaluate_exact_triple_cold_rain_sll_cell(parameters(hail), cell, dt, result, error);
    INFO(error);
    REQUIRE(ok);
    return result;
}

reference::State endpoint(const ColdRimingCell& source, const ColdCollectedSllResult& result)
{
    auto cell = source;
    cell.temperature_k += result.combined.temperature_increment_k;
    cell.initial = result.combined.unrounded;
    return reference::state(cell);
}

void compare(const ColdRimingCell& cell, double dt, bool hail, const ColdCollectedSllResult& actual)
{
    const auto expected = reference::evaluate(cell, dt, hail);
    const auto& combined = actual.combined;
    REQUIRE(reference::prefix::original_donors_admissible(cell, expected));
    CHECK(combined.producer.event_count() == 9);
    CHECK(combined.producer.rain_collection().event_count() == 4);
    CHECK(combined.producer.spectrum_domain() == domain);
    near(combined.rain_mass_increment_per_kg, expected.rain_mass);
    near(combined.rain_number_increment_per_kg, expected.rain_number);
    near(combined.rain_reflectivity_increment_per_kg, expected.rain_z);
    near(combined.cloud_mass_increment_per_kg, expected.cloud_increment);
    near(combined.cloud_number_increment_per_kg, expected.cloud_number_increment);
    near(combined.vapor_mass_increment_per_kg, expected.vapor_increment);
    near(combined.temperature_increment_k, expected.temperature_increment);
    near(actual.hail_conversion.admitted_mass_transfer_per_kg(), expected.sll.admitted);
    const auto& rain = combined.producer.rain_collection();
    near(rain.admission_factor(), expected.rain_factor);
    long double closure = combined.cloud_mass_increment_per_kg + static_cast<long double>(combined.vapor_mass_increment_per_kg)
        + combined.rain_mass_increment_per_kg;
    for (std::size_t species = 0; species < 4; ++species)
    {
        near(rain.channels()[species].raw_mass_per_kg, expected.rain[species].raw_mass);
        near(rain.channels()[species].raw_number_per_kg, expected.rain[species].raw_number);
        near(rain.channels()[species].mass_per_kg, expected.rain[species].mass);
        near(rain.channels()[species].number_per_kg, expected.rain[species].number);
        near(combined.producer.channels()[species].diffusion_mass_per_kg, expected.diffusion[species]);
        near(combined.mass_increment_per_kg[species], expected.mass_increment[species]);
        near(combined.number_increment_per_kg[species], expected.number_increment[species]);
        near(combined.reflectivity_increment_per_kg[species], expected.z_increment[species]);
        closure += combined.mass_increment_per_kg[species];
    }
    CHECK(std::abs(closure) < 1e-18L);
    const auto heat = (-334000.L * (combined.cloud_mass_increment_per_kg + combined.rain_mass_increment_per_kg)
        - 2834000.L * combined.vapor_mass_increment_per_kg) / 1004.L;
    near(combined.temperature_increment_k, heat);
    CHECK(actual.hail_conversion.source().rain_mass_per_kg == cell.initial.rain.mass_per_kg);
    CHECK(actual.hail_conversion.source().admitted_accretion_mass_per_kg[1] == rain.channels()[2].mass_per_kg);
    CHECK(combined.rounded_rain.mass_per_kg == static_cast<float>(combined.unrounded.rain.mass_per_kg));
    CHECK(combined.rounded_rain.number_per_kg == static_cast<float>(combined.unrounded.rain.number_per_kg));
    CHECK(combined.rounded_rain.reflectivity_per_kg == static_cast<float>(combined.unrounded.rain.reflectivity_per_kg));
}

ColdRimingCell advance(ColdRimingCell cell, double duration, int steps)
{
    for (int step = 0; step < steps; ++step)
    {
        const auto result = accepted(cell, duration / steps);
        cell.temperature_k += result.combined.temperature_increment_k;
        cell.initial = result.combined.unrounded;
    }
    return cell;
}
}

TEST_CASE("rain SLL shares original rain and heat across four collection channels", "[physics][milbrandt][triple_cold_rain_sll]")
{
    for (double temperature : {272.15, 268.15, 250.15, 238.15, 222.15})
    for (long double shape : {2.L, 20.L, 350.L})
    for (bool hail : {false, true})
    {
        const auto cell = source(temperature, shape);
        const auto result = accepted(cell, 1e-5, hail);
        compare(cell, 1e-5, hail, result);
        ColdCollectedSllResult replay;
        std::string error;
        REQUIRE(apply_exact_triple_cold_rain_sll_cell(parameters(hail), cell, 1e-5, result.collection,
            result.ice_conversion, result.snow_conversion, result.hail_conversion, replay, error));
        CHECK(endpoint(cell, result) == endpoint(cell, replay));
    }
}

TEST_CASE("rain collection normalized integrals and paired donor admission have independent references", "[physics][milbrandt][triple_cold_rain_sll]")
{
    for (long double shape : {2.L, 20.L, 190.L})
    {
        auto cell = source(268.15, shape);
        const auto integral = reference::evaluate(cell, 1e-4, true, false, true);
        const auto result = accepted(cell, 1e-4);
        for (std::size_t species = 0; species < 4; ++species)
        {
            near(result.combined.producer.rain_collection().channels()[species].raw_mass_per_kg, integral.rain[species].raw_mass);
            near(result.combined.producer.rain_collection().channels()[species].raw_number_per_kg, integral.rain[species].raw_number);
        }
        RainCollectionProducer capped;
        std::string error;
        const RainCollectionCell leaf{{cell.temperature_k, cell.pressure_pa, cell.dry_density_kg_m3, {cell.initial.frozen}}, cell.initial.rain};
        REQUIRE(produce_exact_rain_collection_admission(parameters(), leaf, 1e6, capped, error, 4, nullptr, domain));
        REQUIRE(capped.admission_factor() < 1);
        long double mass = 0, number = 0;
        for (const auto& channel : capped.channels()) { mass += channel.mass_per_kg; number += channel.number_per_kg; }
        CHECK(mass <= cell.initial.rain.mass_per_kg);
        CHECK(number <= cell.initial.rain.number_per_kg);
        const auto cap_reference = reference::evaluate(cell, 1e6);
        near(capped.admission_factor(), cap_reference.rain_factor);
        auto sentinel = result;
        CHECK_FALSE(evaluate_exact_triple_cold_rain_sll_cell(parameters(), cell, 1e6, sentinel, error));
        CHECK(endpoint(cell, sentinel) == endpoint(cell, result));
    }
}

TEST_CASE("rain alone activates SLL and existing hail collects when hail creation is disabled", "[physics][milbrandt][triple_cold_rain_sll]")
{
    auto cell = source();
    cell.initial.cloud_mass_per_kg = cell.initial.cloud_number_per_kg = 0;
    cell.initial.frozen[0] = cell.initial.frozen[1] = {};
    const auto result = accepted(cell, 1e-4);
    compare(cell, 1e-4, true, result);
    CHECK(result.hail_conversion.regime() == GraupelHailSllRegime::Active);
    CHECK(result.hail_conversion.admitted_mass_transfer_per_kg() > 0);
    CHECK(result.combined.cloud_mass_increment_per_kg == 0);
    CHECK(result.combined.cloud_number_increment_per_kg == 0);
    CHECK(result.combined.producer.channels()[2].splinter_number_per_kg == 0);
    const auto disabled = accepted(cell, 1e-4, false);
    CHECK(disabled.hail_conversion.regime() == GraupelHailSllRegime::HailDisabled);
    CHECK(disabled.combined.producer.rain_collection().channels()[3].mass_per_kg > 0);
    cell.initial.frozen[3] = {};
    const auto seeded = accepted(cell, 1e-4);
    CHECK(seeded.combined.unrounded.frozen[3].number_per_kg > 0);
    CHECK(seeded.combined.producer.rain_collection().channels()[3].mass_per_kg == 0);
}

TEST_CASE("rain SLL rejects detached journals invalid sources and exhausted intervals atomically", "[physics][milbrandt][triple_cold_rain_sll]")
{
    const auto cell = source();
    const auto result = accepted(cell, 1e-4);
    for (int mutation = 0; mutation < 21; ++mutation)
    {
        auto changed = cell;
        auto p = parameters();
        double dt = 1e-4;
        std::size_t capacity = 12;
        switch (mutation)
        {
        case 0: changed.initial.rain.mass_per_kg *= 1.01; break;
        case 1: changed.initial.rain.number_per_kg *= 1.01; break;
        case 2: changed.initial.rain.reflectivity_per_kg *= 1.01; break;
        case 3: changed.initial.vapor_mass_per_kg *= 1.01; break;
        case 4: changed.initial.cloud_mass_per_kg *= 1.01; break;
        case 5: changed.temperature_k += .1; break;
        case 6: changed.pressure_pa += 1; break;
        case 7: changed.dry_density_kg_m3 *= 1.01; break;
        case 8: changed.initial.frozen[2].number_per_kg *= 1.01; break;
        case 9: dt *= 2; break;
        case 10: capacity = 11; break;
        case 11: p.hail_processes = false; break;
        case 12: p.a_r_ += 1; break;
        case 13: changed.initial.rain.mass_per_kg = -1; break;
        case 14: changed.initial.rain.number_per_kg = 0; break;
        case 15: changed.initial.rain.reflectivity_per_kg = std::numeric_limits<double>::quiet_NaN(); break;
        case 16: changed.initial.rain.reflectivity_per_kg = 0; break;
        case 17: dt = 0; break;
        case 18: changed.initial.rain = reference::rain_population(.002, 4000, 210); break;
        case 19: changed.temperature_k = 273.15; break;
        case 20: changed.initial.frozen[3] = reference::frozen::population(3, .0002, 20, 520); break;
        }
        auto unpublished = result;
        std::string error;
        CHECK_FALSE(apply_exact_triple_cold_rain_sll_cell(p, changed, dt, result.collection, result.ice_conversion,
            result.snow_conversion, result.hail_conversion, unpublished, error, capacity));
        CHECK_FALSE(error.empty());
        CHECK(endpoint(cell, unpublished) == endpoint(cell, result));
        if (mutation >= 12)
        {
            CHECK_FALSE(evaluate_exact_triple_cold_rain_sll_cell(p, changed, dt, unpublished, error, capacity));
            CHECK(endpoint(cell, unpublished) == endpoint(cell, result));
        }
    }
    std::string error;
    auto unpublished = result;
    CHECK_FALSE(apply_exact_triple_cold_rain_sll_cell(parameters(), cell, 1e-4, {}, {}, {}, {}, unpublished, error));
    CHECK(endpoint(cell, unpublished) == endpoint(cell, result));
    CHECK_FALSE(evaluate_exact_triple_cold_collected_sll_cell(parameters(), cell, 1e-4, unpublished, error));
    CHECK_FALSE(evaluate_exact_triple_cold_rain_sll_cell(parameters(), cell, std::numeric_limits<double>::max(), unpublished, error));
    CHECK(endpoint(cell, unpublished) == endpoint(cell, result));
}

TEST_CASE("rain SLL continuation is deterministic across independent workers", "[physics][milbrandt][triple_cold_rain_sll]")
{
    const auto cell = source();
    constexpr std::size_t count = 16;
    std::array<reference::State, count> serial{}, parallel{};
    std::array<bool, count> ok{};
    const auto integrate = [&](std::size_t index)
    {
        auto current = source(272.15 - .01 * index);
        for (int step = 0; step < 8; ++step)
        {
            ColdCollectedSllResult result;
            std::string error;
            if (!evaluate_exact_triple_cold_rain_sll_cell(parameters(), current, 1e-4, result, error)) return false;
            current.temperature_k += result.combined.temperature_increment_k;
            current.initial = result.combined.unrounded;
        }
        parallel[index] = reference::state(current);
        return true;
    };
    for (std::size_t i = 0; i < count; ++i) { REQUIRE(integrate(i)); serial[i] = parallel[i]; }
    std::array<std::thread, 4> workers;
    for (std::size_t worker = 0; worker < workers.size(); ++worker)
        workers[worker] = std::thread([&, worker] { for (std::size_t i = worker; i < count; i += 4) ok[i] = integrate(i); });
    for (auto& worker : workers) worker.join();
    for (std::size_t i = 0; i < count; ++i) { CHECK(ok[i]); CHECK(serial[i] == parallel[i]); }
    const auto midpoint = advance(cell, .0004, 4);
    CHECK(reference::state(advance(midpoint, .0004, 4)) == reference::state(advance(cell, .0008, 8)));
}

TEST_CASE("rain SLL cannot use incoming rain to fund original graupel loss or replay a detached rain source", "[physics][milbrandt][triple_cold_rain_sll]")
{
    auto cell = source();
    cell.initial.cloud_mass_per_kg = cell.initial.cloud_number_per_kg = 0;
    cell.initial.frozen[0] = cell.initial.frozen[1] = {};
    const auto reference = reference::evaluate(cell, 1e-4);
    const double duration = static_cast<double>(1.01L * cell.initial.frozen[2].mass_per_kg * 1e-4L /
        (reference.sll.admitted + std::max(-reference.diffusion[2], 0.L)));
    ColdRimingProducer riming;
    std::string error;
    const bool produced = produce_exact_cold_riming_admission(parameters(), cell, duration, riming, error, 9, nullptr, domain, true, true);
    INFO(error);
    REQUIRE(produced);
    REQUIRE(riming.rain_collection().admission_factor() == 1);
    const auto result = accepted(cell, 1e-4);
    auto unpublished = result;
    ColdRimingFailure failure;
    CHECK_FALSE(evaluate_exact_triple_cold_rain_sll_cell(parameters(), cell, duration, unpublished, error, 12, &failure));
    CHECK(failure == ColdRimingFailure::IntervalRejected);
    CHECK(error.find("original graupel") != std::string::npos);
    CHECK(endpoint(cell, unpublished) == endpoint(cell, result));
    auto detached = result.ice_conversion;
    auto& rain = const_cast<RainCollectionCell&>(detached.riming_source().rain_collection().source()).rain;
    rain.number_per_kg = std::nextafter(rain.number_per_kg, std::numeric_limits<double>::infinity());
    CHECK_FALSE(apply_exact_triple_cold_rain_sll_cell(parameters(), cell, 1e-4, result.collection,
        detached, result.snow_conversion, result.hail_conversion, unpublished, error, 12, &failure));
    CHECK(failure == ColdRimingFailure::InvalidProducer);
    CHECK(endpoint(cell, unpublished) == endpoint(cell, result));
}

TEST_CASE("rain SLL refines toward an independent coupled integration and preserves zero rain", "[physics][milbrandt][triple_cold_rain_sll]")
{
    const auto cell = source();
    const auto initial = reference::state(cell);
    const auto oracle = reference::rk4(cell, .002, 64);
    const auto refined_oracle = reference::rk4(cell, .002, 128);
    const auto norm = [&](const reference::State& values)
    {
        long double error = 0;
        for (std::size_t i = 0; i < values.size(); ++i)
            if (std::abs(oracle[i] - initial[i]) > 1e-14L * std::abs(initial[i]))
                error = std::max(error, std::abs(values[i] - oracle[i]) / std::abs(oracle[i] - initial[i]));
        return error;
    };
    CHECK(norm(refined_oracle) < 1e-6L);
    const auto coarse = norm(reference::state(advance(cell, .002, 2)));
    const auto medium = norm(reference::state(advance(cell, .002, 4)));
    const auto fine = norm(reference::state(advance(cell, .002, 8)));
    CHECK(medium < .55L * coarse);
    CHECK(fine < .55L * medium);
    for (double temperature : {272.15, 268.15, 238.15})
    {
        auto no_rain = source(temperature);
        no_rain.initial.rain = {};
        const auto extended = accepted(no_rain, 1e-4);
        ColdCollectedSllResult old;
        std::string error;
        REQUIRE(evaluate_exact_triple_cold_collected_sll_cell(parameters(), no_rain, 1e-4, old, error));
        CHECK(endpoint(no_rain, extended) == endpoint(no_rain, old));
        CHECK(extended.combined.mass_increment_per_kg == old.combined.mass_increment_per_kg);
        CHECK(extended.combined.number_increment_per_kg == old.combined.number_increment_per_kg);
        CHECK(extended.combined.reflectivity_increment_per_kg == old.combined.reflectivity_increment_per_kg);
        CHECK(extended.combined.temperature_increment_k == old.combined.temperature_increment_k);
    }
    CHECK_FALSE(tmv::microphysics::kMilbrandtTripleMomentPublicAdmissionOpen);
}
