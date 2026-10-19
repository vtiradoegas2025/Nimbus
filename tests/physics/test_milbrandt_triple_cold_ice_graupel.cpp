#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_ice_graupel.hpp"
#include "support/milbrandt_cold_ice_graupel_reference.hpp"

#include <cmath>
#include <limits>

namespace
{

using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_cold_ice_graupel_reference;

ColdParameters parameters(bool hail = true)
{
    const auto& p = default_cold_parameters;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], hail};
}

ColdRimingCell source(long double alpha = 2, double saturation = 1, bool graupel = true)
{
    ColdRimingCell cell;
    cell.temperature_k = 268.15;
    cell.pressure_pa = 85000;
    cell.dry_density_kg_m3 = .65;
    cell.initial.vapor_mass_per_kg = static_cast<double>(reference::riming::thermal::saturation(268.15, 85000).mass) * saturation;
    cell.initial.cloud_mass_per_kg = 5e-4;
    cell.initial.cloud_number_per_kg = 2e8;
    for (std::size_t species = 0; species < 4; ++species)
    {
        if (species == 2 && !graupel)
        {
            continue;
        }
        cell.initial.frozen[species] = reference::riming::collection::frozen::population(species,
            std::array<double, 4>{2e-5, 1e-4, 2e-4, 3e-4}[species],
            std::array<double, 4>{5000, 5000, 600, 80}[species], alpha);
    }
    return cell;
}

void near(double actual, long double expected)
{
    INFO("actual=" << actual << ", independent=" << static_cast<double>(expected));
    REQUIRE(std::isfinite(actual));
    CHECK(std::abs(actual - expected) <= 4e-8L * std::max(std::abs(expected), 1e-30L));
}

void check_reference(const ColdRimingCell& cell, double duration, const ColdIceGraupelResult& actual)
{
    const auto expected = reference::evaluate(cell, duration);
    const auto& result = actual.combined;
    near(actual.conversion.raw_mass_transfer_per_kg(), expected.conversion_mass);
    CHECK(actual.conversion.raw_mass_transfer_per_kg() == actual.conversion.admitted_mass_transfer_per_kg());
    near(actual.conversion.donor_number_loss_per_kg(), expected.ice_number_loss);
    near(actual.conversion.recipient_number_gain_per_kg(), expected.graupel_number_gain);
    near(result.temperature_increment_k, expected.temperature_increment);
    near(result.vapor_mass_increment_per_kg, expected.vapor_increment);
    near(result.cloud_mass_increment_per_kg, expected.cloud_increment);
    near(result.cloud_number_increment_per_kg, expected.cloud_number_increment);
    long double water = result.vapor_mass_increment_per_kg + static_cast<long double>(result.cloud_mass_increment_per_kg);
    for (std::size_t index = 0; index < 4; ++index)
    {
        near(result.mass_increment_per_kg[index], expected.mass_increment[index]);
        near(result.number_increment_per_kg[index], expected.number_increment[index]);
        near(result.reflectivity_increment_per_kg[index], expected.z_increment[index]);
        water += result.mass_increment_per_kg[index];
        CHECK(result.rounded[index].mass_per_kg == static_cast<float>(result.unrounded.frozen[index].mass_per_kg));
        CHECK(result.rounded[index].number_per_kg == static_cast<float>(result.unrounded.frozen[index].number_per_kg));
        CHECK(result.rounded[index].reflectivity_per_kg == static_cast<float>(result.unrounded.frozen[index].reflectivity_per_kg));
    }
    CHECK(std::abs(water) <= 1e-16L * (cell.initial.vapor_mass_per_kg + cell.initial.cloud_mass_per_kg));
}

void same_result(const ColdIceGraupelResult& a, const ColdIceGraupelResult& b)
{
    CHECK(reference::riming::state({0, 85000, .65, a.combined.unrounded}) ==
          reference::riming::state({0, 85000, .65, b.combined.unrounded}));
    CHECK(a.combined.mass_increment_per_kg == b.combined.mass_increment_per_kg);
    CHECK(a.combined.number_increment_per_kg == b.combined.number_increment_per_kg);
    CHECK(a.combined.reflectivity_increment_per_kg == b.combined.reflectivity_increment_per_kg);
    CHECK(a.combined.temperature_increment_k == b.combined.temperature_increment_k);
    CHECK(a.conversion.event_count() == b.conversion.event_count());
}

} // namespace

TEST_CASE("coupled ice to graupel uses accepted source rates and conserves cold water without extra heat",
          "[physics][milbrandt][triple_cold_ice_graupel]")
{
    bool active = false;
    bool inactive = false;
    for (long double alpha : {0.0L, 2.0L, 20.0L})
    {
        for (double saturation : {.95, 1.0, 1.1})
        {
            for (bool graupel : {false, true})
            {
                for (bool hail : {false, true})
                {
                    const auto cell = source(alpha, saturation, graupel);
                    ColdIceGraupelResult result;
                    std::string error;
                    const bool accepted = evaluate_exact_triple_cold_ice_graupel_cell(parameters(hail), cell, 1e-4, result, error);
                    INFO(error);
                    REQUIRE(accepted);
                    check_reference(cell, 1e-4, result);
                    active |= result.conversion.raw_mass_transfer_per_kg() > 0;
                    inactive |= result.conversion.raw_mass_transfer_per_kg() == 0;
                    if (!graupel && result.conversion.raw_mass_transfer_per_kg() > 0)
                    {
                        REQUIRE(result.combined.unrounded.frozen[2].mass_per_kg > 0);
                        CHECK(result.combined.number_increment_per_kg[2] == result.conversion.recipient_number_gain_per_kg());
                    }
                    ColdRimingResult prefix;
                    REQUIRE(evaluate_exact_triple_cold_riming_cell(parameters(hail), cell, 1e-4, prefix, error));
                    CHECK(result.combined.temperature_increment_k == prefix.temperature_increment_k);
                    CHECK(result.combined.vapor_mass_increment_per_kg == prefix.vapor_mass_increment_per_kg);
                    CHECK(result.combined.cloud_mass_increment_per_kg == prefix.cloud_mass_increment_per_kg);
                    ColdIceGraupelResult replay;
                    REQUIRE(apply_exact_triple_cold_ice_graupel_cell(parameters(hail), cell, 1e-4, result.conversion, replay, error));
                    same_result(result, replay);
                }
            }
        }
    }
    CHECK(active);
    auto deposition_dominated = source(2, 1.1);
    deposition_dominated.initial.cloud_mass_per_kg = 1e-8;
    ColdIceGraupelResult result;
    std::string error;
    REQUIRE(evaluate_exact_triple_cold_ice_graupel_cell(parameters(), deposition_dominated, 1e-4, result, error));
    const auto& channel = result.combined.producer.channels()[0];
    REQUIRE(channel.collection_mass_per_kg > 0);
    REQUIRE(channel.diffusion_mass_per_kg >= channel.collection_mass_per_kg);
    check_reference(deposition_dominated, 1e-4, result);
    inactive |= result.conversion.raw_mass_transfer_per_kg() == 0;
    CHECK(inactive);
}

TEST_CASE("coupled conversion enforces capture provenance before retryable interval admission",
          "[physics][milbrandt][triple_cold_ice_graupel]")
{
    const auto cell = source();
    ColdIceGraupelResult accepted;
    std::string error;
    REQUIRE(evaluate_exact_triple_cold_ice_graupel_cell(parameters(), cell, 1e-4, accepted, error));
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
        auto result = accepted;
        ColdRimingFailure failure;
        CHECK_FALSE(apply_exact_triple_cold_ice_graupel_cell(parameters(hail), changed, duration,
            accepted.conversion, result, error, 1, &failure));
        CHECK(failure == ColdRimingFailure::InvalidProducer);
        same_result(result, accepted);
    }
    for (std::size_t capacity : {0u, 2u})
    {
        auto result = accepted;
        ColdRimingFailure failure;
        CHECK_FALSE(evaluate_exact_triple_cold_ice_graupel_cell(parameters(), cell, 1e-4, result, error, capacity, &failure));
        CHECK(failure == ColdRimingFailure::Capacity);
        same_result(result, accepted);
    }
    auto result = accepted;
    CHECK_FALSE(apply_exact_triple_cold_ice_graupel_cell(parameters(), cell, 1e-4,
        IceGraupelConversionProducer{}, result, error));
    same_result(result, accepted);
    REQUIRE(evaluate_exact_triple_cold_ice_graupel_cell(parameters(), cell, 1e-4, result, error));
    same_result(result, accepted);
}

TEST_CASE("coupled ice conversion converges to independent full state RK4 without a timestep squared rate",
          "[physics][milbrandt][triple_cold_ice_graupel]")
{
    constexpr double duration = .01;
    for (double saturation : {.95, 1.0, 1.1})
    {
        const auto initial = source(2, saturation);
        const auto expected32 = reference::rk4(initial, duration, 32);
        const auto expected64 = reference::rk4(initial, duration, 64);
        const auto before = reference::riming::state(initial);
        const auto distance = [&](const reference::State& values)
        {
            long double largest = 0;
            for (std::size_t index = 0; index < values.size(); ++index)
            {
                const long double scale = std::max(std::abs(expected64[index] - before[index]), std::abs(before[index]) * 1e-8L);
                largest = std::max(largest, std::abs(values[index] - expected64[index]) / std::max(scale, 1e-30L));
            }
            return largest;
        };
        CHECK(distance(expected32) < 1e-6L);
        std::array<long double, 3> errors{};
        for (int refinement = 0; refinement < 3; ++refinement)
        {
            auto cell = initial;
            const int steps = 4 << refinement;
            for (int step = 0; step < steps; ++step)
            {
                ColdIceGraupelResult result;
                std::string error;
                REQUIRE(evaluate_exact_triple_cold_ice_graupel_cell(parameters(), cell, duration / steps, result, error));
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
}

TEST_CASE("ice conversion rejects a joint original donor overdraw before endpoint repair or secondary production",
          "[physics][milbrandt][triple_cold_ice_graupel]")
{
    auto cell = source();
    cell.initial.cloud_mass_per_kg = .02;
    cell.initial.frozen[1] = {};
    cell.initial.frozen[2] = {};
    cell.initial.frozen[3] = {};
    const auto unit = reference::riming::evaluate(cell, 1);
    const double duration = static_cast<double>(.75L * cell.initial.frozen[0].mass_per_kg / unit.collected[0]);
    ColdRimingProducer prefix;
    std::string error;
    const bool captured = produce_exact_cold_riming_admission(parameters(), cell, duration, prefix, error);
    INFO(error);
    REQUIRE(captured);
    REQUIRE(2 * prefix.channels()[0].collection_mass_per_kg > cell.initial.frozen[0].mass_per_kg);
    IceGraupelConversionProducer conversion;
    IceGraupelConversionFailure failure;
    CHECK_FALSE(produce_exact_ice_graupel_conversion_admission(prefix, conversion, error, 1, &failure));
    CHECK(failure == IceGraupelConversionFailure::IntervalRejected);
    CHECK(conversion.event_count() == 0);
    ColdIceGraupelResult result;
    result.combined.temperature_increment_k = 123;
    CHECK_FALSE(evaluate_exact_triple_cold_ice_graupel_cell(parameters(), cell, duration, result, error));
    CHECK(result.combined.temperature_increment_k == 123);
    REQUIRE(evaluate_exact_triple_cold_ice_graupel_cell(parameters(), source(), 1e-4, result, error));
    const auto accepted = result.conversion;
    conversion = accepted;
    CHECK_FALSE(produce_exact_ice_graupel_conversion_admission(ColdRimingProducer{}, conversion, error, 1, &failure));
    CHECK(failure == IceGraupelConversionFailure::InvalidProducer);
    CHECK(conversion.mass_increment_per_kg() == accepted.mass_increment_per_kg());
    CHECK(conversion.number_increment_per_kg() == accepted.number_increment_per_kg());
    CHECK(conversion.reflectivity_increment_per_kg() == accepted.reflectivity_increment_per_kg());
}
