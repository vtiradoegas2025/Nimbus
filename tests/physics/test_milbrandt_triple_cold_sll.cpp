#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_sll.hpp"
#include "support/milbrandt_cold_sll_reference.hpp"

#include <cmath>
#include <limits>

namespace
{

using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_cold_sll_reference;

ColdParameters parameters(bool hail = true)
{
    const auto& p = default_cold_parameters;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], hail};
}

ColdRimingCell source(double temperature = 272.15, bool graupel = true)
{
    ColdRimingCell cell;
    cell.temperature_k = temperature;
    cell.pressure_pa = 85000;
    cell.dry_density_kg_m3 = .65;
    cell.initial.vapor_mass_per_kg = static_cast<double>(
        reference::riming::thermal::saturation(temperature, 85000).mass);
    cell.initial.cloud_mass_per_kg = .005;
    cell.initial.cloud_number_per_kg = 2e8;
    for (std::size_t species = 0; species < 3; ++species)
    {
        if (species == 2 && !graupel)
        {
            continue;
        }
        cell.initial.frozen[species] = reference::riming::collection::frozen::population(species,
            std::array<double, 3>{2e-5, 1e-4, 2e-4}[species],
            std::array<double, 3>{5000, 5000, 600}[species], 2);
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

void same_result(const ColdSllResult& actual, const ColdSllResult& expected)
{
    CHECK(reference::riming::state({0, 85000, .65, actual.combined.unrounded}) ==
        reference::riming::state({0, 85000, .65, expected.combined.unrounded}));
    CHECK(actual.combined.mass_increment_per_kg == expected.combined.mass_increment_per_kg);
    CHECK(actual.combined.number_increment_per_kg == expected.combined.number_increment_per_kg);
    CHECK(actual.combined.reflectivity_increment_per_kg == expected.combined.reflectivity_increment_per_kg);
    CHECK(actual.combined.cloud_mass_increment_per_kg == expected.combined.cloud_mass_increment_per_kg);
    CHECK(actual.combined.cloud_number_increment_per_kg == expected.combined.cloud_number_increment_per_kg);
    CHECK(actual.combined.vapor_mass_increment_per_kg == expected.combined.vapor_mass_increment_per_kg);
    CHECK(actual.combined.temperature_increment_k == expected.combined.temperature_increment_k);
    CHECK(actual.combined.producer.event_count() == expected.combined.producer.event_count());
    CHECK(actual.ice_conversion.mass_increment_per_kg() == expected.ice_conversion.mass_increment_per_kg());
    CHECK(actual.snow_conversion.mass_increment_per_kg() == expected.snow_conversion.mass_increment_per_kg());
    CHECK(actual.hail_conversion.mass_increment_per_kg() == expected.hail_conversion.mass_increment_per_kg());
}

ColdSllResult accepted(const ColdRimingCell& cell, bool hail = true)
{
    ColdSllResult result;
    std::string error;
    const bool ok = evaluate_exact_triple_cold_sll_cell(parameters(hail), cell, 1e-4, result, error);
    INFO(error);
    REQUIRE(ok);
    return result;
}

} // namespace

TEST_CASE("cloud SLL shares one independently referenced five-way source and adds no second heat or cloud debit",
          "[physics][milbrandt][triple_cold_sll]")
{
    bool active = false, inactive = false, freezing = false;
    for (double temperature : {272.15, 268.15, 238.15, 222.15})
    {
        for (bool hail : {false, true})
        {
            const auto cell = source(temperature);
            const auto result = accepted(cell, hail);
            const auto expected = reference::evaluate(cell, 1e-4, hail);
            const auto& prefix = result.combined.producer;
            REQUIRE(prefix.event_count() == 5);
            REQUIRE(prefix.homogeneous_freezing_enabled());
            CHECK(prefix.spectrum_domain() == ColdSpectrumDomain::GraupelAndHail512);
            CHECK(result.hail_conversion.source().rain_mass_per_kg == 0);
            CHECK(result.hail_conversion.source().admitted_accretion_mass_per_kg ==
                std::array<double, 3>{prefix.channels()[2].collection_mass_per_kg, 0, 0});
            CHECK(result.hail_conversion.source().graupel_diffusion_mass_per_kg == prefix.channels()[2].diffusion_mass_per_kg);
            CHECK(result.hail_conversion.source().graupel_diffusion_number_per_kg == prefix.channels()[2].diffusion_number_per_kg);
            near(result.hail_conversion.raw_mass_transfer_per_kg(), expected.sll.raw);
            near(result.hail_conversion.admitted_mass_transfer_per_kg(), expected.sll.admitted);
            near(result.combined.cloud_mass_increment_per_kg, expected.cloud_increment);
            near(result.combined.cloud_number_increment_per_kg, expected.cloud_number_increment);
            near(result.combined.vapor_mass_increment_per_kg, expected.vapor_increment);
            near(result.combined.temperature_increment_k, expected.temperature_increment);
            long double water = result.combined.cloud_mass_increment_per_kg
                + static_cast<long double>(result.combined.vapor_mass_increment_per_kg);
            for (std::size_t species = 0; species < 4; ++species)
            {
                near(result.combined.mass_increment_per_kg[species], expected.mass_increment[species]);
                near(result.combined.number_increment_per_kg[species], expected.number_increment[species]);
                near(result.combined.reflectivity_increment_per_kg[species], expected.z_increment[species]);
                water += result.combined.mass_increment_per_kg[species];
                CHECK(result.combined.rounded[species].mass_per_kg ==
                    static_cast<float>(result.combined.unrounded.frozen[species].mass_per_kg));
            }
            CHECK(std::abs(water) <= 1e-15L * (cell.initial.cloud_mass_per_kg + cell.initial.vapor_mass_per_kg));
            active |= result.hail_conversion.admitted_mass_transfer_per_kg() > 0;
            inactive |= result.hail_conversion.admitted_mass_transfer_per_kg() == 0;
            freezing |= prefix.homogeneous_increments_per_kg()[0] > 0;
            ColdSllResult replay;
            std::string error;
            REQUIRE(apply_exact_triple_cold_sll_cell(parameters(hail), cell, 1e-4,
                result.ice_conversion, result.snow_conversion, result.hail_conversion, replay, error));
            same_result(replay, result);
        }
    }
    CHECK(active);
    CHECK(inactive);
    CHECK(freezing);
}

TEST_CASE("cloud SLL does not borrow fresh graupel particles from same-interval ice or snow conversion",
          "[physics][milbrandt][triple_cold_sll]")
{
    const auto cell = source(272.15, false);
    const auto result = accepted(cell);
    REQUIRE(result.snow_conversion.admitted_mass_transfer_per_kg() > 0);
    REQUIRE(result.combined.unrounded.frozen[2].mass_per_kg > 0);
    CHECK(result.hail_conversion.source().source.initial.frozen[2].mass_per_kg == 0);
    CHECK(result.hail_conversion.source().admitted_accretion_mass_per_kg[0] == 0);
    CHECK(result.hail_conversion.regime() == GraupelHailSllRegime::NoGraupel);
    CHECK(result.hail_conversion.admitted_mass_transfer_per_kg() == 0);
    CHECK(result.combined.unrounded.frozen[3].mass_per_kg == 0);
}

TEST_CASE("cloud SLL replay rejects changed context foreign accretion and disabled homogeneous prefixes before regeneration",
          "[physics][milbrandt][triple_cold_sll]")
{
    const auto cell = source();
    const auto result = accepted(cell);
    std::string error;
    ColdRimingFailure failure;
    auto output = result;
    auto changed = cell;
    changed.initial.cloud_mass_per_kg = std::nextafter(changed.initial.cloud_mass_per_kg,
        std::numeric_limits<double>::infinity());
    CHECK_FALSE(apply_exact_triple_cold_sll_cell(parameters(), changed, 1e6,
        result.ice_conversion, result.snow_conversion, result.hail_conversion, output, error, 3, &failure));
    CHECK(failure == ColdRimingFailure::InvalidProducer);
    same_result(output, result);
    auto foreign_cell = result.hail_conversion.source();
    foreign_cell.rain_mass_per_kg = 1e-4;
    foreign_cell.admitted_accretion_mass_per_kg[1] = 1e-8;
    GraupelHailConversionProducer foreign;
    REQUIRE(produce_exact_graupel_hail_conversion_admission(parameters(), foreign_cell, 1e-4,
        foreign, error, 1, nullptr, ColdSpectrumDomain::GraupelAndHail512));
    CHECK_FALSE(apply_exact_triple_cold_sll_cell(parameters(), cell, 1e-4,
        result.ice_conversion, result.snow_conversion, foreign, output, error, 3, &failure));
    CHECK(failure == ColdRimingFailure::InvalidProducer);
    same_result(output, result);
    GraupelHailConversionFailure leaf_failure;
    CHECK_FALSE(validate_exact_graupel_hail_conversion_admission(parameters(), result.hail_conversion.source(),
        1e-4, foreign, error, 1, &leaf_failure, ColdSpectrumDomain::GraupelAndHail512));
    CHECK(leaf_failure == GraupelHailConversionFailure::InvalidProducer);
    ColdRimingProducer four_way;
    REQUIRE(produce_exact_cold_riming_admission(parameters(), cell, 1e-4, four_way,
        error, 4, nullptr, ColdSpectrumDomain::GraupelAndHail512));
    IceGraupelConversionProducer ice;
    SnowGraupelConversionProducer snow;
    REQUIRE(produce_exact_ice_graupel_conversion_admission(four_way, ice, error));
    REQUIRE(produce_exact_snow_graupel_conversion_admission(four_way, snow, error));
    CHECK_FALSE(apply_exact_triple_cold_sll_cell(parameters(), cell, 1e-4,
        ice, snow, result.hail_conversion, output, error, 3, &failure));
    CHECK(failure == ColdRimingFailure::InvalidProducer);
    same_result(output, result);
}

TEST_CASE("cloud SLL capacity invalid captures and stiff intervals preserve the complete candidate and recover",
          "[physics][milbrandt][triple_cold_sll]")
{
    const auto cell = source();
    const auto result = accepted(cell);
    std::string error;
    ColdRimingFailure failure;
    for (std::size_t capacity : {0u, 2u, 4u})
    {
        auto output = result;
        CHECK_FALSE(evaluate_exact_triple_cold_sll_cell(parameters(), cell, 1e-4,
            output, error, capacity, &failure));
        CHECK(failure == ColdRimingFailure::Capacity);
        same_result(output, result);
    }
    auto output = result;
    CHECK_FALSE(apply_exact_triple_cold_sll_cell(parameters(), cell, 1e-4,
        result.ice_conversion, result.snow_conversion, GraupelHailConversionProducer{}, output, error, 3, &failure));
    CHECK(failure == ColdRimingFailure::InvalidProducer);
    same_result(output, result);
    CHECK_FALSE(evaluate_exact_triple_cold_sll_cell(parameters(), cell, 1e6,
        output, error, 3, &failure));
    CHECK(failure == ColdRimingFailure::IntervalRejected);
    same_result(output, result);
    REQUIRE(evaluate_exact_triple_cold_sll_cell(parameters(), cell, 1e-4,
        output, error, 3, &failure));
    CHECK(failure == ColdRimingFailure::None);
    same_result(output, result);
}
