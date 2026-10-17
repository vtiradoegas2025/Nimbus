#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_snow_graupel_conversion.hpp"
#include "support/milbrandt_cold_riming_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace
{

using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_cold_riming_reference;
namespace spectrum = milbrandt_frozen_collection_reference;

ColdParameters parameters(bool hail = true)
{
    const auto& p = default_cold_parameters;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], hail};
}

ColdRimingCell source(long double alpha = 2, double saturation = 1)
{
    ColdRimingCell cell;
    cell.temperature_k = 268.15;
    cell.pressure_pa = 85000;
    cell.dry_density_kg_m3 = .65;
    cell.initial.vapor_mass_per_kg = static_cast<double>(
        reference::thermal::saturation(cell.temperature_k, cell.pressure_pa).mass) * saturation;
    cell.initial.cloud_mass_per_kg = 5e-4;
    cell.initial.cloud_number_per_kg = 2e8;
    cell.initial.frozen[1] = spectrum::population(1, 1e-4, 5000, alpha);
    return cell;
}

void near(double actual, long double expected, long double relative = 4e-8L)
{
    INFO("actual=" << actual << ", independent=" << static_cast<double>(expected));
    REQUIRE(std::isfinite(actual));
    CHECK(std::abs(static_cast<long double>(actual) - expected)
        <= relative * std::max(std::abs(expected), 1e-290L));
}

ColdRimingProducer capture(const ColdRimingCell& cell, double duration,
                          ColdSpectrumDomain domain = ColdSpectrumDomain::Base200,
                          bool hail = true)
{
    ColdRimingProducer result;
    std::string error;
    const bool accepted = produce_exact_cold_riming_admission(
        parameters(hail), cell, duration, result, error, 4, nullptr, domain);
    INFO(error);
    REQUIRE(accepted);
    return result;
}

SnowGraupelConversionProducer convert(const ColdRimingProducer& prefix)
{
    SnowGraupelConversionProducer result;
    std::string error;
    const bool accepted = produce_exact_snow_graupel_conversion_admission(prefix, result, error);
    INFO(error);
    REQUIRE(accepted);
    return result;
}

void same(const SnowGraupelConversionProducer& actual, const SnowGraupelConversionProducer& expected)
{
    CHECK(actual.event_count() == expected.event_count());
    CHECK(actual.raw_mass_transfer_per_kg() == expected.raw_mass_transfer_per_kg());
    CHECK(actual.admitted_mass_transfer_per_kg() == expected.admitted_mass_transfer_per_kg());
    CHECK(actual.donor_number_loss_per_kg() == expected.donor_number_loss_per_kg());
    CHECK(actual.recipient_number_gain_per_kg() == expected.recipient_number_gain_per_kg());
    CHECK(actual.source_mass_squared_per_kg() == expected.source_mass_squared_per_kg());
    CHECK(actual.source_alpha() == expected.source_alpha());
    CHECK(actual.source_lambda_per_m() == expected.source_lambda_per_m());
    CHECK(actual.admitted_fraction() == expected.admitted_fraction());
    CHECK(actual.mass_increment_per_kg() == expected.mass_increment_per_kg());
    CHECK(actual.number_increment_per_kg() == expected.number_increment_per_kg());
    CHECK(actual.reflectivity_increment_per_kg() == expected.reflectivity_increment_per_kg());
    CHECK(actual.riming_source().parameters() == expected.riming_source().parameters());
    CHECK(actual.riming_source().hail_processes() == expected.riming_source().hail_processes());
    CHECK(actual.riming_source().spectrum_domain() == expected.riming_source().spectrum_domain());
    CHECK(actual.riming_source().duration_s() == expected.riming_source().duration_s());
    CHECK(actual.riming_source().cloud_admission_factor() == expected.riming_source().cloud_admission_factor());
    CHECK(actual.riming_source().relaxation_number() == expected.riming_source().relaxation_number());
    CHECK(actual.riming_source().source().pressure_pa == expected.riming_source().source().pressure_pa);
    CHECK(actual.riming_source().source().dry_density_kg_m3 == expected.riming_source().source().dry_density_kg_m3);
    CHECK(reference::state(actual.riming_source().source()) == reference::state(expected.riming_source().source()));
    for (std::size_t index = 0; index < 4; ++index)
    {
        const auto& a = actual.riming_source().channels()[index];
        const auto& b = expected.riming_source().channels()[index];
        CHECK(a.raw_collection_mass_per_kg == b.raw_collection_mass_per_kg);
        CHECK(a.raw_collection_number_per_kg == b.raw_collection_number_per_kg);
        CHECK(a.collection_mass_per_kg == b.collection_mass_per_kg);
        CHECK(a.collection_number_per_kg == b.collection_number_per_kg);
        CHECK(a.environmental_diffusion_mass_per_kg == b.environmental_diffusion_mass_per_kg);
        CHECK(a.riming_diffusion_mass_per_kg == b.riming_diffusion_mass_per_kg);
        CHECK(a.diffusion_mass_per_kg == b.diffusion_mass_per_kg);
        CHECK(a.diffusion_number_per_kg == b.diffusion_number_per_kg);
        CHECK(a.splinter_embryo_mass_per_kg == b.splinter_embryo_mass_per_kg);
        CHECK(a.splinter_number_per_kg == b.splinter_number_per_kg);
        CHECK(a.splinter_reflectivity_per_kg == b.splinter_reflectivity_per_kg);
    }
}

long double mass_squared(const FrozenCollectionPopulation& snow)
{
    const long double alpha = spectrum::represented_alpha(snow, 1);
    const long double d = spectrum::mass_exponent[1];
    const long double ratio = spectrum::gamma_ratio(alpha, 2 * d)
        / std::pow(spectrum::gamma_ratio(alpha, d), 2);
    return static_cast<long double>(snow.mass_per_kg) * snow.mass_per_kg
        / snow.number_per_kg * ratio;
}

} // namespace

TEST_CASE("snow conversion journals use the published mass rate and conservative noncubic moments",
          "[physics][milbrandt][triple_snow_graupel_conversion]")
{
    bool active = false;
    for (long double alpha : {0.0L, 2.0L, 20.0L, 100.0L})
    {
        for (double saturation : {.95, 1.0, 1.1})
        {
            for (bool hail : {false, true})
            {
                const auto cell = source(alpha, saturation);
                const auto prefix = capture(cell, 1e-4, ColdSpectrumDomain::Base200, hail);
                const auto result = convert(prefix);
                const auto expected = reference::evaluate(cell, 1e-4);
                const long double mass = expected.collected[1] > expected.diffusion[1]
                    ? (4.0L / 3) * expected.collected[1] : 0;
                const auto& snow = cell.initial.frozen[1];
                const long double fraction = mass / snow.mass_per_kg;
                const long double m2 = mass_squared(snow);
                near(result.raw_mass_transfer_per_kg(), mass);
                CHECK(result.raw_mass_transfer_per_kg() == result.admitted_mass_transfer_per_kg());
                near(result.source_mass_squared_per_kg(), m2);
                const long double represented_alpha = spectrum::represented_alpha(snow, 1);
                CHECK(std::abs(result.source_alpha() - represented_alpha)
                    <= 4e-8L * std::max(std::abs(represented_alpha), 1.0L));
                near(result.source_lambda_per_m(), spectrum::spectrum(snow, 1, cell.dry_density_kg_m3).lambda);
                near(result.admitted_fraction(), fraction);
                near(result.donor_number_loss_per_kg(), snow.number_per_kg * fraction);
                CHECK(result.donor_number_loss_per_kg() == result.recipient_number_gain_per_kg());
                near(result.mass_increment_per_kg()[1], -mass);
                near(result.mass_increment_per_kg()[2], mass);
                near(result.number_increment_per_kg()[1], -snow.number_per_kg * fraction);
                near(result.number_increment_per_kg()[2], snow.number_per_kg * fraction);
                near(result.reflectivity_increment_per_kg()[1], -snow.reflectivity_per_kg * fraction);
                near(result.reflectivity_increment_per_kg()[2], fraction * m2
                    / (spectrum::mass_coefficient[2] * spectrum::mass_coefficient[2]));
                CHECK(result.mass_increment_per_kg()[1] + result.mass_increment_per_kg()[2] == 0);
                CHECK(result.number_increment_per_kg()[1] + result.number_increment_per_kg()[2] == 0);
                for (std::size_t index : {0u, 3u})
                {
                    CHECK(result.mass_increment_per_kg()[index] == 0);
                    CHECK(result.number_increment_per_kg()[index] == 0);
                    CHECK(result.reflectivity_increment_per_kg()[index] == 0);
                }
                if (mass > 0)
                {
                    active = true;
                    const long double incorrectly_cubic = -result.reflectivity_increment_per_kg()[1]
                        * std::pow(spectrum::mass_coefficient[1] / spectrum::mass_coefficient[2], 2);
                    CHECK(std::abs(result.reflectivity_increment_per_kg()[2] - incorrectly_cubic)
                        > .1L * std::abs(result.reflectivity_increment_per_kg()[2]));
                }
                same(convert(prefix), result);
            }
        }
    }
    CHECK(active);
    auto inactive = source(2, 1.1);
    inactive.initial.cloud_mass_per_kg = 1e-8;
    const auto prefix = capture(inactive, 1e-4);
    REQUIRE(prefix.channels()[1].collection_mass_per_kg > 0);
    REQUIRE(prefix.channels()[1].diffusion_mass_per_kg >= prefix.channels()[1].collection_mass_per_kg);
    CHECK(convert(prefix).raw_mass_transfer_per_kg() == 0);
}

TEST_CASE("snow conversion is linear in a frozen-source interval unlike the retained double quadratic law",
          "[physics][milbrandt][triple_snow_graupel_conversion]")
{
    const auto cell = source();
    const auto short_prefix = capture(cell, 1e-4);
    const auto long_prefix = capture(cell, 2e-4);
    REQUIRE(short_prefix.cloud_admission_factor() == 1);
    REQUIRE(long_prefix.cloud_admission_factor() == 1);
    const auto short_result = convert(short_prefix);
    const auto long_result = convert(long_prefix);
    REQUIRE(short_result.raw_mass_transfer_per_kg() > 0);
    near(long_result.raw_mass_transfer_per_kg(), 2 * short_result.raw_mass_transfer_per_kg(), 2e-13L);
    const long double q = cell.initial.frozen[1].mass_per_kg;
    const long double c1 = short_prefix.channels()[1].collection_mass_per_kg;
    const long double c2 = long_prefix.channels()[1].collection_mass_per_kg;
    const long double legacy1 = 100 * c1 * c1 / q;
    const long double legacy2 = 100 * c2 * c2 / q;
    REQUIRE(legacy1 < q + c1);
    REQUIRE(legacy2 < q + c2);
    CHECK(std::abs(legacy2 / legacy1 - 4) < 1e-12L);
    CHECK(std::abs((long_result.raw_mass_transfer_per_kg() / 2e-4)
        / (short_result.raw_mass_transfer_per_kg() / 1e-4) - 1) < 1e-12);
    CHECK(std::abs((legacy2 / 2e-4) / (legacy1 / 1e-4) - 2) < 1e-12L);
}

TEST_CASE("snow alpha near 200 maps independent mass-squared into graupel alpha near 418 point 72",
          "[physics][milbrandt][triple_snow_graupel_conversion]")
{
    auto cell = source(200);
    for (int step = 0; step < 4096; ++step)
    {
        cell.initial.frozen[1].reflectivity_per_kg = std::nextafter(
            cell.initial.frozen[1].reflectivity_per_kg, std::numeric_limits<double>::infinity());
    }
    const auto prefix = capture(cell, 1e-4, ColdSpectrumDomain::SnowConvertedGraupel512);
    const auto result = convert(prefix);
    REQUIRE(result.raw_mass_transfer_per_kg() > 0);
    const auto& snow = cell.initial.frozen[1];
    const long double alpha = spectrum::represented_alpha(snow, 1);
    CHECK(std::abs(alpha - 200) < 1e-6L);
    const long double f = result.raw_mass_transfer_per_kg() / snow.mass_per_kg;
    const long double m2 = mass_squared(snow);
    near(result.source_mass_squared_per_kg(), m2);
    const FrozenCollectionPopulation receiver{result.mass_increment_per_kg()[2],
        result.number_increment_per_kg()[2], result.reflectivity_increment_per_kg()[2]};
    const long double receiver_alpha = spectrum::represented_alpha(receiver, 2);
    INFO("receiver alpha=" << static_cast<double>(receiver_alpha));
    CHECK(receiver_alpha > 200);
    CHECK(receiver_alpha < 512);
    CHECK(std::abs(receiver_alpha - 418.7215725131708L) < 1e-4L);
    near(receiver.reflectivity_per_kg, f * m2 / std::pow(spectrum::mass_coefficient[2], 2));
    const long double transferred_m2 = spectrum::mass_coefficient[2]
        * spectrum::mass_coefficient[2] * receiver.reflectivity_per_kg;
    CHECK(std::abs(transferred_m2 - f * m2) < 4e-8L * f * m2);
}

TEST_CASE("snow conversion journals retain distinct base and extended spectrum provenance",
          "[physics][milbrandt][triple_snow_graupel_conversion]")
{
    const auto cell = source();
    const auto base = convert(capture(cell, 1e-4));
    const auto extended = convert(capture(cell, 1e-4, ColdSpectrumDomain::SnowConvertedGraupel512));
    CHECK(base.riming_source().spectrum_domain() == ColdSpectrumDomain::Base200);
    CHECK(extended.riming_source().spectrum_domain() == ColdSpectrumDomain::SnowConvertedGraupel512);
    CHECK(base.mass_increment_per_kg() == extended.mass_increment_per_kg());
    CHECK(base.number_increment_per_kg() == extended.number_increment_per_kg());
    CHECK(base.reflectivity_increment_per_kg() == extended.reflectivity_increment_per_kg());
    auto changed = cell;
    changed.initial.frozen[2] = spectrum::population(2, 2e-4, 600, 450);
    ColdRimingProducer rejected = base.riming_source();
    const auto before = rejected;
    std::string error;
    ColdRimingFailure failure;
    CHECK_FALSE(produce_exact_cold_riming_admission(parameters(), changed, 1e-4,
        rejected, error, 4, &failure, ColdSpectrumDomain::Base200));
    CHECK(failure == ColdRimingFailure::InvalidSource);
    CHECK(reference::state(rejected.source()) == reference::state(before.source()));
    CHECK(rejected.spectrum_domain() == before.spectrum_domain());
    CHECK(rejected.duration_s() == before.duration_s());
    const auto admitted = convert(capture(changed, 1e-4, ColdSpectrumDomain::SnowConvertedGraupel512));
    CHECK(admitted.riming_source().spectrum_domain() == ColdSpectrumDomain::SnowConvertedGraupel512);
    CHECK(spectrum::represented_alpha(admitted.riming_source().source().initial.frozen[2], 2) > 400);
}

TEST_CASE("snow conversion rejects missing captures and invalid capacity atomically then recovers",
          "[physics][milbrandt][triple_snow_graupel_conversion]")
{
    const auto prefix = capture(source(), 1e-4);
    const auto accepted = convert(prefix);
    std::string error;
    SnowGraupelConversionFailure failure;
    for (std::size_t capacity : {0u, 2u, 8u})
    {
        auto output = accepted;
        CHECK_FALSE(produce_exact_snow_graupel_conversion_admission(prefix, output, error, capacity, &failure));
        CHECK(failure == SnowGraupelConversionFailure::Capacity);
        same(output, accepted);
    }
    auto output = accepted;
    CHECK_FALSE(produce_exact_snow_graupel_conversion_admission(
        ColdRimingProducer{}, output, error, 1, &failure));
    CHECK(failure == SnowGraupelConversionFailure::InvalidProducer);
    same(output, accepted);
    REQUIRE(produce_exact_snow_graupel_conversion_admission(prefix, output, error, 1, &failure));
    CHECK(failure == SnowGraupelConversionFailure::None);
    CHECK(error.empty());
    same(output, accepted);
}

TEST_CASE("snow conversion jointly debits original sublimating snow without charging fresh-cloud HM twice",
          "[physics][milbrandt][triple_snow_graupel_conversion]")
{
    auto cell = source();
    cell.initial.cloud_mass_per_kg = .02;
    const auto unit = reference::evaluate(cell, 1, true);
    const long double loss_rate = (4.0L / 3) * unit.collected[1]
        + std::max(-unit.diffusion[1], 0.0L);
    REQUIRE(unit.collected[1] > 0);
    REQUIRE(unit.diffusion[1] < 0);
    REQUIRE(loss_rate > 0);
    const double near_duration = static_cast<double>(.99995L * cell.initial.frozen[1].mass_per_kg / loss_rate);
    const auto funded_prefix = capture(cell, near_duration);
    const auto funded = convert(funded_prefix);
    const auto& accepted_snow = funded_prefix.channels()[1];
    REQUIRE(accepted_snow.diffusion_mass_per_kg < 0);
    REQUIRE(accepted_snow.diffusion_number_per_kg < 0);
    REQUIRE(accepted_snow.splinter_embryo_mass_per_kg > 0);
    CHECK(accepted_snow.splinter_embryo_mass_per_kg < accepted_snow.collection_mass_per_kg);
    const double original_loss = funded.raw_mass_transfer_per_kg() - accepted_snow.diffusion_mass_per_kg;
    CHECK(original_loss < cell.initial.frozen[1].mass_per_kg);
    CHECK(original_loss + accepted_snow.splinter_embryo_mass_per_kg > cell.initial.frozen[1].mass_per_kg);
    CHECK(funded.donor_number_loss_per_kg() - accepted_snow.diffusion_number_per_kg
        < cell.initial.frozen[1].number_per_kg);
    CHECK(cell.initial.frozen[1].mass_per_kg + accepted_snow.collection_mass_per_kg
        + accepted_snow.diffusion_mass_per_kg - funded.raw_mass_transfer_per_kg()
        - accepted_snow.splinter_embryo_mass_per_kg > 0);

    const double excessive_duration = static_cast<double>(1.01L * cell.initial.frozen[1].mass_per_kg / loss_rate);
    const auto excessive = capture(cell, excessive_duration);
    const auto& snow = excessive.channels()[1];
    const double conversion_mass = (4.0 / 3) * snow.collection_mass_per_kg;
    REQUIRE(snow.diffusion_mass_per_kg < 0);
    REQUIRE(conversion_mass < cell.initial.frozen[1].mass_per_kg);
    REQUIRE(-snow.diffusion_mass_per_kg < cell.initial.frozen[1].mass_per_kg);
    REQUIRE(conversion_mass - snow.diffusion_mass_per_kg > cell.initial.frozen[1].mass_per_kg);
    REQUIRE(cell.initial.frozen[1].number_per_kg * conversion_mass / cell.initial.frozen[1].mass_per_kg
        - snow.diffusion_number_per_kg > cell.initial.frozen[1].number_per_kg);
    auto output = funded;
    std::string error;
    SnowGraupelConversionFailure failure;
    CHECK_FALSE(produce_exact_snow_graupel_conversion_admission(excessive, output, error, 1, &failure));
    CHECK(failure == SnowGraupelConversionFailure::IntervalRejected);
    same(output, funded);
    REQUIRE(produce_exact_snow_graupel_conversion_admission(funded_prefix, output, error, 1, &failure));
    CHECK(failure == SnowGraupelConversionFailure::None);
    same(output, funded);
}
