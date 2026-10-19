#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_graupel_conversions.hpp"
#include "support/milbrandt_cold_graupel_conversions_reference.hpp"

#include <cmath>
#include <limits>

namespace
{

using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_cold_graupel_conversions_reference;

ColdParameters parameters(bool hail = true)
{
    const auto& p = default_cold_parameters;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], hail};
}

ColdRimingCell source(long double snow_alpha = 2, double saturation = 1, bool snow_only = false)
{
    ColdRimingCell cell;
    cell.temperature_k = 271.15;
    cell.pressure_pa = 85000;
    cell.dry_density_kg_m3 = .65;
    cell.initial.vapor_mass_per_kg = static_cast<double>(reference::riming::thermal::saturation(271.15, 85000).mass) * saturation;
    cell.initial.cloud_mass_per_kg = 5e-4;
    cell.initial.cloud_number_per_kg = 2e8;
    for (std::size_t species = 0; species < 4; ++species)
    {
        if (snow_only && species != 1)
        {
            continue;
        }
        cell.initial.frozen[species] = reference::riming::collection::frozen::population(species,
            std::array<double, 4>{2e-5, 1e-4, 2e-4, 3e-4}[species],
            std::array<double, 4>{5000, 5000, 600, 80}[species], species == 1 ? snow_alpha : 2);
    }
    return cell;
}

void near(double actual, long double expected)
{
    INFO("actual=" << actual << ", reference=" << static_cast<double>(expected));
    REQUIRE(std::isfinite(actual));
    CHECK(std::abs(actual - expected) <= 5e-8L * std::max(std::abs(expected), 1e-30L));
}

void check_reference(const ColdRimingCell& cell, double duration, const ColdGraupelConversionsResult& actual)
{
    const auto expected = reference::evaluate(cell, duration);
    const auto& result = actual.combined;
    near(actual.ice_conversion.raw_mass_transfer_per_kg(), expected.conversion_mass);
    near(actual.snow_conversion.raw_mass_transfer_per_kg(), expected.snow_conversion_mass);
    near(actual.snow_conversion.donor_number_loss_per_kg(), expected.snow_conversion_number);
    CHECK(actual.snow_conversion.donor_number_loss_per_kg() == actual.snow_conversion.recipient_number_gain_per_kg());
    CHECK(actual.snow_conversion.admitted_mass_transfer_per_kg() == actual.snow_conversion.raw_mass_transfer_per_kg());
    if (expected.snow_conversion_mass > 0)
    {
        near(actual.snow_conversion.source_mass_squared_per_kg(), expected.snow_mass_squared);
    }
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
    CHECK(a.ice_conversion.event_count() == b.ice_conversion.event_count());
    CHECK(a.snow_conversion.event_count() == b.snow_conversion.event_count());
}

} // namespace

TEST_CASE("joint graupel conversions preserve independent moments mass and once-only latent heat",
          "[physics][milbrandt][triple_cold_graupel_conversions]")
{
    for (long double alpha : {0.0L, 2.0L, 20.0L, 199.9L})
    {
        for (double saturation : {.95, 1.0, 1.1})
        {
            for (bool snow_only : {false, true})
            {
                for (bool hail : {false, true})
                {
                    const auto cell = source(alpha, saturation, snow_only);
                    ColdGraupelConversionsResult result;
                    std::string error;
                    const bool accepted = evaluate_exact_triple_cold_graupel_conversions_cell(parameters(hail), cell, 1e-4, result, error);
                    INFO(error);
                    REQUIRE(accepted);
                    check_reference(cell, 1e-4, result);
                    CHECK(result.combined.producer.spectrum_domain() == ColdSpectrumDomain::SnowConvertedGraupel512);
                    ColdGraupelConversionsResult replay;
                    REQUIRE(apply_exact_triple_cold_graupel_conversions_cell(parameters(hail), cell, 1e-4,
                        result.ice_conversion, result.snow_conversion, replay, error));
                    same_result(result, replay);
                }
            }
        }
    }
}

TEST_CASE("noncubic snow conversion creates narrow graupel and continues without clipping shape or moments",
          "[physics][milbrandt][triple_cold_graupel_conversions]")
{
    auto cell = source(199.9L, 1, true);
    std::string error;
    for (int step = 0; step < 3; ++step)
    {
        ColdGraupelConversionsResult result;
        REQUIRE(evaluate_exact_triple_cold_graupel_conversions_cell(parameters(), cell, 1e-4, result, error));
        check_reference(cell, 1e-4, result);
        const auto& graupel = result.combined.rounded[2];
        double alpha = 0;
        double lambda = 0;
        REQUIRE(diagnose_triple_gamma_shape(graupel.mass_per_kg, graupel.number_per_kg, graupel.reflectivity_per_kg,
            current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Graupel).mass_law, alpha, lambda, error));
        CHECK(alpha > 400);
        CHECK(alpha < 512);
        cell.initial = result.combined.unrounded;
        for (std::size_t species = 0; species < 4; ++species)
        {
            cell.initial.frozen[species] = {result.combined.rounded[species].mass_per_kg,
                result.combined.rounded[species].number_per_kg, result.combined.rounded[species].reflectivity_per_kg};
        }
        cell.temperature_k += result.combined.temperature_increment_k;
    }
    ColdRimingProducer preserved;
    CHECK_FALSE(produce_exact_cold_riming_admission(parameters(), cell, 1e-4, preserved, error));
    REQUIRE(produce_exact_cold_riming_admission(parameters(), cell, 1e-4, preserved, error, 4, nullptr,
        ColdSpectrumDomain::SnowConvertedGraupel512));
}

TEST_CASE("joint conversion rejects mixed prefixes resources and foreign domains before publication",
          "[physics][milbrandt][triple_cold_graupel_conversions]")
{
    const auto cell = source();
    ColdGraupelConversionsResult accepted;
    std::string error;
    REQUIRE(evaluate_exact_triple_cold_graupel_conversions_cell(parameters(), cell, 1e-4, accepted, error));
    ColdIceGraupelResult base;
    REQUIRE(evaluate_exact_triple_cold_ice_graupel_cell(parameters(), cell, 1e-4, base, error));
    auto result = accepted;
    ColdRimingFailure failure;
    CHECK_FALSE(apply_exact_triple_cold_graupel_conversions_cell(parameters(), cell, 1e-4,
        base.conversion, accepted.snow_conversion, result, error, 2, &failure));
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
            accepted.ice_conversion, accepted.snow_conversion, result, error, 2, &failure));
        CHECK(failure == ColdRimingFailure::InvalidProducer);
        same_result(result, accepted);
    }
    for (std::size_t capacity : {0u, 1u, 3u})
    {
        CHECK_FALSE(evaluate_exact_triple_cold_graupel_conversions_cell(parameters(), cell, 1e-4, result, error, capacity, &failure));
        CHECK(failure == ColdRimingFailure::Capacity);
        same_result(result, accepted);
    }
    REQUIRE(evaluate_exact_triple_cold_graupel_conversions_cell(parameters(), cell, 1e-4, result, error));
    same_result(result, accepted);
}

TEST_CASE("joint conversion refinement follows independent full-state RK4 and a linear conversion rate",
          "[physics][milbrandt][triple_cold_graupel_conversions]")
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
                ColdGraupelConversionsResult result;
                std::string error;
                REQUIRE(evaluate_exact_triple_cold_graupel_conversions_cell(parameters(), cell, duration / steps, result, error));
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
