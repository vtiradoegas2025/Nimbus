#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_graupel_conversions.hpp"
#include "support/milbrandt_cold_homogeneous_reference.hpp"

#include <cmath>
#include <limits>

namespace
{

using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_cold_homogeneous_reference;

ColdParameters parameters(bool hail = true)
{
    const auto& p = default_cold_parameters;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], hail};
}

ColdRimingCell source(double temperature = 233.15, int selection = 0, double saturation = 1)
{
    ColdRimingCell cell;
    cell.temperature_k = temperature;
    cell.pressure_pa = 85000;
    cell.dry_density_kg_m3 = .65;
    cell.initial.vapor_mass_per_kg = static_cast<double>(reference::riming::thermal::saturation(temperature, 85000).mass) * saturation;
    cell.initial.cloud_mass_per_kg = 5e-4;
    cell.initial.cloud_number_per_kg = 2e8;
    for (std::size_t species = 0; species < 4; ++species)
    {
        if (selection == 2 || (selection == 1 && species == 0))
        {
            continue;
        }
        cell.initial.frozen[species] = reference::riming::collection::frozen::population(species,
            std::array<double, 4>{2e-5, 1e-4, 2e-4, 3e-4}[species],
            std::array<double, 4>{5000, 5000, 600, 80}[species], 2);
    }
    return cell;
}

void near(double actual, long double expected)
{
    INFO("actual=" << actual << ", reference=" << static_cast<double>(expected));
    REQUIRE(std::isfinite(actual));
    CHECK(std::abs(actual - expected) <= 6e-8L * std::max(std::abs(expected), 1e-35L));
}

void check_reference(const ColdRimingCell& cell, double duration, const ColdGraupelConversionsResult& actual)
{
    const auto expected = reference::evaluate(cell, duration);
    const auto& result = actual.combined;
    const auto& prefix = result.producer;
    REQUIRE(prefix.event_count() == 5);
    REQUIRE(prefix.homogeneous_freezing_enabled());
    REQUIRE(prefix.homogeneous_freezing().event_count() == 1);
    near(prefix.cloud_admission_factor(), expected.factor);
    near(prefix.homogeneous_freezing().channel().fraction, expected.raw_fraction);
    for (std::size_t component = 0; component < 3; ++component)
    {
        near(prefix.homogeneous_increments_per_kg()[component], expected.admitted_freezing[component]);
    }
    near(result.temperature_increment_k, expected.temperature_increment);
    near(result.cloud_mass_increment_per_kg, expected.cloud_increment);
    near(result.cloud_number_increment_per_kg, expected.cloud_number_increment);
    near(result.vapor_mass_increment_per_kg, expected.vapor_increment);
    long double water = result.vapor_mass_increment_per_kg + static_cast<long double>(result.cloud_mass_increment_per_kg);
    for (std::size_t species = 0; species < 4; ++species)
    {
        near(result.mass_increment_per_kg[species], expected.mass_increment[species]);
        near(result.number_increment_per_kg[species], expected.number_increment[species]);
        near(result.reflectivity_increment_per_kg[species], expected.z_increment[species]);
        near(prefix.channels()[species].collection_mass_per_kg, expected.collected[species]);
        near(prefix.channels()[species].diffusion_mass_per_kg, expected.diffusion[species]);
        water += result.mass_increment_per_kg[species];
        CHECK(result.rounded[species].mass_per_kg == static_cast<float>(result.unrounded.frozen[species].mass_per_kg));
        CHECK(result.rounded[species].number_per_kg == static_cast<float>(result.unrounded.frozen[species].number_per_kg));
        CHECK(result.rounded[species].reflectivity_per_kg == static_cast<float>(result.unrounded.frozen[species].reflectivity_per_kg));
    }
    CHECK(std::abs(water) <= 1e-15L * (cell.initial.vapor_mass_per_kg + cell.initial.cloud_mass_per_kg));
}

void same_result(const ColdGraupelConversionsResult& a, const ColdGraupelConversionsResult& b)
{
    CHECK(reference::riming::state({0, 85000, .65, a.combined.unrounded}) ==
          reference::riming::state({0, 85000, .65, b.combined.unrounded}));
    CHECK(a.combined.mass_increment_per_kg == b.combined.mass_increment_per_kg);
    CHECK(a.combined.number_increment_per_kg == b.combined.number_increment_per_kg);
    CHECK(a.combined.reflectivity_increment_per_kg == b.combined.reflectivity_increment_per_kg);
    CHECK(a.combined.temperature_increment_k == b.combined.temperature_increment_k);
    CHECK(a.combined.producer.event_count() == b.combined.producer.event_count());
}

} // namespace

TEST_CASE("homogeneous freezing composes with four collectors diffusion and conversions from one source",
          "[physics][milbrandt][triple_cold_homogeneous]")
{
    for (double temperature : {244.15, 243.15, 238.15, 233.15, 228.15, 222.15})
    {
        for (int selection : {0, 1, 2})
        {
            for (double saturation : {.98, 1.0, 1.02})
            {
                INFO("T=" << temperature << ", selection=" << selection << ",RH=" << saturation);
                const auto cell = source(temperature, selection, saturation);
                ColdGraupelConversionsResult result;
                std::string error;
                const bool accepted = evaluate_exact_triple_cold_graupel_conversions_cell(
                    parameters(), cell, 1e-4, result, error, 2, nullptr, true);
                INFO(error);
                REQUIRE(accepted);
                check_reference(cell, 1e-4, result);
                ColdGraupelConversionsResult replay;
                REQUIRE(apply_exact_triple_cold_graupel_conversions_cell(parameters(), cell, 1e-4,
                    result.ice_conversion, result.snow_conversion, replay, error, 2, nullptr, true));
                same_result(result, replay);
                if (selection == 2 && temperature < 243.15)
                {
                    CHECK(result.combined.unrounded.frozen[0].mass_per_kg > 0);
                    CHECK(result.combined.number_increment_per_kg[0] == result.combined.producer.homogeneous_increments_per_kg()[1]);
                    CHECK(result.combined.reflectivity_increment_per_kg[0] == result.combined.producer.homogeneous_increments_per_kg()[2]);
                }
                if (selection == 2 && temperature < 223.15)
                {
                    CHECK(result.combined.unrounded.cloud_mass_per_kg == 0);
                    CHECK(result.combined.unrounded.cloud_number_per_kg == 0);
                }
            }
        }
    }
}

TEST_CASE("five competing cloud consumers share mass number and frozen-droplet spectrum admission",
          "[physics][milbrandt][triple_cold_homogeneous]")
{
    const auto cell = source(222.15);
    ColdGraupelConversionsResult result;
    std::string error;
    REQUIRE(evaluate_exact_triple_cold_graupel_conversions_cell(parameters(), cell, 1e-4, result, error, 2, nullptr, true));
    const auto& producer = result.combined.producer;
    REQUIRE(producer.cloud_admission_factor() < 1);
    REQUIRE(producer.homogeneous_freezing().channel().fraction == 1);
    double mass = producer.homogeneous_increments_per_kg()[0];
    double number = producer.homogeneous_increments_per_kg()[1];
    for (const auto& channel : producer.channels())
    {
        CHECK(channel.collection_mass_per_kg == producer.cloud_admission_factor() * channel.raw_collection_mass_per_kg);
        CHECK(channel.collection_number_per_kg == producer.cloud_admission_factor() * channel.raw_collection_number_per_kg);
        mass += channel.collection_mass_per_kg;
        number += channel.collection_number_per_kg;
    }
    CHECK(mass <= cell.initial.cloud_mass_per_kg);
    CHECK(number <= cell.initial.cloud_number_per_kg);
    CHECK(producer.homogeneous_increments_per_kg()[2] == producer.cloud_admission_factor()
        * producer.homogeneous_freezing().channel().raw_reflectivity_increment_per_kg);
    check_reference(cell, 1e-4, result);
}

TEST_CASE("homogeneous process selection and every source value bind replay before numerical evaluation",
          "[physics][milbrandt][triple_cold_homogeneous]")
{
    const auto cell = source();
    ColdGraupelConversionsResult accepted;
    std::string error;
    REQUIRE(evaluate_exact_triple_cold_graupel_conversions_cell(parameters(), cell, 1e-4, accepted, error, 2, nullptr, true));
    auto result = accepted;
    ColdRimingFailure failure;
    CHECK_FALSE(apply_exact_triple_cold_graupel_conversions_cell(parameters(), cell, 1e-4,
        accepted.ice_conversion, accepted.snow_conversion, result, error, 2, &failure));
    CHECK(failure == ColdRimingFailure::InvalidProducer);
    same_result(result, accepted);
    for (int mutation = 0; mutation < 20; ++mutation)
    {
        auto changed = cell;
        double duration = 1e-4;
        bool hail = true;
        std::array<double*, 18> fields{&changed.temperature_k, &changed.pressure_pa, &changed.dry_density_kg_m3,
            &changed.initial.vapor_mass_per_kg, &changed.initial.cloud_mass_per_kg, &changed.initial.cloud_number_per_kg};
        for (std::size_t species = 0; species < 4; ++species)
        {
            fields[6 + species * 3] = &changed.initial.frozen[species].mass_per_kg;
            fields[7 + species * 3] = &changed.initial.frozen[species].number_per_kg;
            fields[8 + species * 3] = &changed.initial.frozen[species].reflectivity_per_kg;
        }
        if (mutation < 18)
        {
            *fields[mutation] = std::nextafter(*fields[mutation], std::numeric_limits<double>::infinity());
        }
        else if (mutation == 18)
        {
            duration = 1e6;
        }
        else
        {
            hail = false;
        }
        CHECK_FALSE(apply_exact_triple_cold_graupel_conversions_cell(parameters(hail), changed, duration,
            accepted.ice_conversion, accepted.snow_conversion, result, error, 2, &failure, true));
        CHECK(failure == ColdRimingFailure::InvalidProducer);
        same_result(result, accepted);
    }
    ColdRimingProducer preserved = accepted.combined.producer;
    CHECK_FALSE(produce_exact_cold_riming_admission(parameters(), cell, 1e-4, preserved, error,
        4, &failure, ColdSpectrumDomain::SnowConvertedGraupel512, true));
    CHECK(failure == ColdRimingFailure::Capacity);
    CHECK(preserved.event_count() == 5);
    REQUIRE(evaluate_exact_triple_cold_graupel_conversions_cell(parameters(), cell, 1e-4, result, error, 2, nullptr, true));
    same_result(result, accepted);
}

TEST_CASE("finite-hazard homogeneous coupling refines toward independent full-state RK4",
          "[physics][milbrandt][triple_cold_homogeneous]")
{
    constexpr double duration = .005;
    const auto initial = source(236.15);
    const auto expected256 = reference::rk4(initial, duration, 256);
    const auto expected512 = reference::rk4(initial, duration, 512);
    const auto before = reference::riming::state(initial);
    const auto distance = [&](const reference::State& values)
    {
        long double largest = 0;
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            const long double scale = std::max(std::abs(expected512[index] - before[index]), std::abs(before[index]) * 1e-8L);
            largest = std::max(largest, std::abs(values[index] - expected512[index]) / std::max(scale, 1e-35L));
        }
        return largest;
    };
    CHECK(distance(expected256) < 1e-6L);
    std::array<long double, 3> errors{};
    for (int refinement = 0; refinement < 3; ++refinement)
    {
        auto cell = initial;
        const int steps = 4 << refinement;
        for (int step = 0; step < steps; ++step)
        {
            ColdGraupelConversionsResult result;
            std::string error;
            REQUIRE(evaluate_exact_triple_cold_graupel_conversions_cell(parameters(), cell, duration / steps,
                result, error, 2, nullptr, true));
            cell.initial = result.combined.unrounded;
            cell.temperature_k += result.combined.temperature_increment_k;
        }
        errors[refinement] = distance(reference::riming::state(cell));
    }
    INFO("errors=" << static_cast<double>(errors[0]) << "," << static_cast<double>(errors[1]) << "," << static_cast<double>(errors[2]));
    CHECK(errors[1] < .6L * errors[0]);
    CHECK(errors[2] < .6L * errors[1]);
    CHECK(errors[2] < .02L);
}
