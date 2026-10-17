#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_snow_graupel_conversion.hpp"
#include "support/milbrandt_cold_riming_reference.hpp"
#include "support/milbrandt_graupel_hail_conversion_reference.hpp"

#include <cstring>
#include <limits>

namespace
{

using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_graupel_hail_conversion_reference;
namespace spectrum = milbrandt_frozen_collection_reference;

ColdParameters parameters(bool hail = true)
{
    const auto& p = default_cold_parameters;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], hail};
}

GraupelHailConversionCell source(long double alpha = 2)
{
    GraupelHailConversionCell cell;
    cell.source.temperature_k = 272.15;
    cell.source.pressure_pa = 85000;
    cell.source.dry_density_kg_m3 = .65;
    cell.source.initial.vapor_mass_per_kg = .002;
    cell.source.initial.cloud_mass_per_kg = .005;
    cell.source.initial.cloud_number_per_kg = 2e8;
    cell.source.initial.frozen[0] = spectrum::population(0, 3e-5, 3000, 2);
    cell.source.initial.frozen[1] = spectrum::population(1, 3e-4, 1000, 2);
    cell.source.initial.frozen[2] = spectrum::population(2, 2e-4, 1000, alpha);
    cell.rain_mass_per_kg = 1e-4;
    cell.admitted_accretion_mass_per_kg = {1e-8, 2e-8, 3e-8};
    return cell;
}

std::array<double*, 24> fields(GraupelHailConversionCell& cell)
{
    auto& s = cell.source;
    std::array<double*, 24> result{&s.temperature_k, &s.pressure_pa, &s.dry_density_kg_m3,
        &s.initial.vapor_mass_per_kg, &s.initial.cloud_mass_per_kg, &s.initial.cloud_number_per_kg};
    for (std::size_t index = 0; index < 4; ++index)
    {
        result[6 + 3 * index] = &s.initial.frozen[index].mass_per_kg;
        result[7 + 3 * index] = &s.initial.frozen[index].number_per_kg;
        result[8 + 3 * index] = &s.initial.frozen[index].reflectivity_per_kg;
    }
    result[18] = &cell.rain_mass_per_kg;
    for (std::size_t index = 0; index < 3; ++index)
    {
        result[19 + index] = &cell.admitted_accretion_mass_per_kg[index];
    }
    result[22] = &cell.graupel_diffusion_mass_per_kg;
    result[23] = &cell.graupel_diffusion_number_per_kg;
    return result;
}

void bitwise(double actual, double expected)
{
    CHECK(std::memcmp(&actual, &expected, sizeof(double)) == 0);
}

void near(double actual, long double expected, long double relative = 4e-11L)
{
    INFO("actual=" << actual << ", independent=" << static_cast<double>(expected));
    REQUIRE(std::isfinite(actual));
    CHECK(std::abs(static_cast<long double>(actual) - expected)
        <= relative * std::max(std::abs(expected), 1e-290L));
}

void same(const GraupelHailConversionProducer& actual, const GraupelHailConversionProducer& expected)
{
    CHECK(actual.event_count() == expected.event_count());
    CHECK(actual.parameters() == expected.parameters());
    CHECK(actual.hail_processes() == expected.hail_processes());
    CHECK(actual.spectrum_domain() == expected.spectrum_domain());
    CHECK(actual.regime() == expected.regime());
    const std::array<double, 11> a{actual.duration_s(), actual.loading_denominator(), actual.source_alpha(),
        actual.source_lambda_per_m(), actual.mean_mass_diameter_m(), actual.hail_embryo_diameter_m(),
        actual.size_ratio(), actual.raw_mass_transfer_per_kg(), actual.admitted_mass_transfer_per_kg(),
        actual.admitted_fraction(), actual.donor_number_loss_per_kg()};
    const std::array<double, 11> b{expected.duration_s(), expected.loading_denominator(), expected.source_alpha(),
        expected.source_lambda_per_m(), expected.mean_mass_diameter_m(), expected.hail_embryo_diameter_m(),
        expected.size_ratio(), expected.raw_mass_transfer_per_kg(), expected.admitted_mass_transfer_per_kg(),
        expected.admitted_fraction(), expected.donor_number_loss_per_kg()};
    for (std::size_t index = 0; index < a.size(); ++index)
    {
        bitwise(a[index], b[index]);
    }
    CHECK(actual.mass_increment_per_kg() == expected.mass_increment_per_kg());
    CHECK(actual.number_increment_per_kg() == expected.number_increment_per_kg());
    CHECK(actual.reflectivity_increment_per_kg() == expected.reflectivity_increment_per_kg());
    auto ac = actual.source();
    auto bc = expected.source();
    const auto af = fields(ac);
    const auto bf = fields(bc);
    for (std::size_t index = 0; index < af.size(); ++index)
    {
        bitwise(*af[index], *bf[index]);
    }
}

void same(const GraupelHailConversionResult& actual, const GraupelHailConversionResult& expected)
{
    for (std::size_t index = 0; index < 4; ++index)
    {
        bitwise(actual.unrounded[index].mass_per_kg, expected.unrounded[index].mass_per_kg);
        bitwise(actual.unrounded[index].number_per_kg, expected.unrounded[index].number_per_kg);
        bitwise(actual.unrounded[index].reflectivity_per_kg, expected.unrounded[index].reflectivity_per_kg);
        CHECK(actual.rounded[index].mass_per_kg == expected.rounded[index].mass_per_kg);
        CHECK(actual.rounded[index].number_per_kg == expected.rounded[index].number_per_kg);
        CHECK(actual.rounded[index].reflectivity_per_kg == expected.rounded[index].reflectivity_per_kg);
    }
    CHECK(actual.mass_increment_per_kg == expected.mass_increment_per_kg);
    CHECK(actual.number_increment_per_kg == expected.number_increment_per_kg);
    CHECK(actual.reflectivity_increment_per_kg == expected.reflectivity_increment_per_kg);
    bitwise(actual.cloud_mass_increment_per_kg, expected.cloud_mass_increment_per_kg);
    bitwise(actual.rain_mass_increment_per_kg, expected.rain_mass_increment_per_kg);
    bitwise(actual.vapor_mass_increment_per_kg, expected.vapor_mass_increment_per_kg);
    bitwise(actual.temperature_increment_k, expected.temperature_increment_k);
    same(actual.producer, expected.producer);
}

GraupelHailConversionResult evaluate(const GraupelHailConversionCell& cell,
                                   ColdSpectrumDomain domain = ColdSpectrumDomain::Base200,
                                   bool hail = true, double duration = 1e-4)
{
    GraupelHailConversionResult result;
    std::string error;
    const bool accepted = evaluate_exact_triple_graupel_hail_conversion_cell(
        parameters(hail), cell, duration, result, error, 1, nullptr, domain);
    INFO(error);
    REQUIRE(accepted);
    return result;
}

} // namespace

TEST_CASE("SLL conversion uses published loading and all three accretion amounts with conservative cubic transfers",
          "[physics][milbrandt][triple_graupel_hail_conversion]")
{
    bool active = false, inactive = false;
    for (long double alpha : {0.0L, 2.0L, 20.0L})
    {
        for (double temperature : {263.15, 268.15, 272.15})
        {
            for (double cloud : {1e-4, .005, .02})
            {
                for (bool hail : {false, true})
                {
                    auto cell = source(alpha);
                    cell.source.temperature_k = temperature;
                    cell.source.initial.cloud_mass_per_kg = cloud;
                    const auto expected = reference::evaluate(cell, hail);
                    const auto result = evaluate(cell, ColdSpectrumDomain::Base200, hail);
                    const auto& producer = result.producer;
                    near(producer.loading_denominator(), expected.loading);
                    near(producer.mean_mass_diameter_m(), expected.diameter);
                    near(producer.raw_mass_transfer_per_kg(), expected.raw);
                    near(producer.admitted_mass_transfer_per_kg(), expected.admitted);
                    near(producer.admitted_fraction(), expected.fraction);
                    CHECK(producer.donor_number_loss_per_kg() == producer.recipient_number_gain_per_kg());
                    for (std::size_t index = 0; index < 4; ++index)
                    {
                        near(result.mass_increment_per_kg[index], expected.mass[index]);
                        near(result.number_increment_per_kg[index], expected.number[index]);
                        near(result.reflectivity_increment_per_kg[index], expected.reflectivity[index]);
                        CHECK(result.rounded[index].mass_per_kg == static_cast<float>(result.unrounded[index].mass_per_kg));
                        CHECK(result.rounded[index].number_per_kg == static_cast<float>(result.unrounded[index].number_per_kg));
                        CHECK(result.rounded[index].reflectivity_per_kg == static_cast<float>(result.unrounded[index].reflectivity_per_kg));
                    }
                    CHECK(result.mass_increment_per_kg[2] + result.mass_increment_per_kg[3] == 0);
                    CHECK(result.number_increment_per_kg[2] + result.number_increment_per_kg[3] == 0);
                    const long double m2_loss = -result.reflectivity_increment_per_kg[2]
                        * spectrum::mass_coefficient[2] * spectrum::mass_coefficient[2];
                    const long double m2_gain = result.reflectivity_increment_per_kg[3]
                        * spectrum::mass_coefficient[3] * spectrum::mass_coefficient[3];
                    CHECK(std::abs(m2_loss - m2_gain) <= 4e-15L * std::max(m2_gain, 1e-290L));
                    CHECK(result.cloud_mass_increment_per_kg == 0);
                    CHECK(result.rain_mass_increment_per_kg == 0);
                    CHECK(result.vapor_mass_increment_per_kg == 0);
                    CHECK(result.temperature_increment_k == 0);
                    if (expected.raw > 0)
                    {
                        active = true;
                        near(producer.hail_embryo_diameter_m(), expected.embryo_diameter);
                        near(producer.size_ratio(), expected.ratio);
                        REQUIRE(producer.regime() == GraupelHailSllRegime::Active);
                    }
                    else
                    {
                        inactive = true;
                    }
                    GraupelHailConversionResult replay;
                    std::string error;
                    REQUIRE(apply_exact_triple_graupel_hail_conversion_cell(parameters(hail), cell,
                        1e-4, producer, replay, error));
                    same(replay, result);
                }
            }
        }
    }
    CHECK(active);
    CHECK(inactive);
    for (std::size_t channel = 0; channel < 3; ++channel)
    {
        auto cell = source();
        cell.admitted_accretion_mass_per_kg = {};
        cell.admitted_accretion_mass_per_kg[channel] = 1e-8;
        const auto result = evaluate(cell);
        REQUIRE(result.producer.raw_mass_transfer_per_kg() > 0);
        near(result.producer.raw_mass_transfer_per_kg(), reference::evaluate(cell).raw);
    }
}

TEST_CASE("SLL transfer is linear in accepted source-frozen interval accretion",
          "[physics][milbrandt][triple_graupel_hail_conversion]")
{
    const auto cell = source();
    auto longer = cell;
    for (double& amount : longer.admitted_accretion_mass_per_kg)
    {
        amount *= 2;
    }
    const auto first = evaluate(cell, ColdSpectrumDomain::Base200, true, 1e-4);
    const auto second = evaluate(longer, ColdSpectrumDomain::Base200, true, 2e-4);
    REQUIRE(first.producer.raw_mass_transfer_per_kg() > 0);
    CHECK(first.producer.raw_mass_transfer_per_kg() == first.producer.admitted_mass_transfer_per_kg());
    CHECK(second.producer.raw_mass_transfer_per_kg() == second.producer.admitted_mass_transfer_per_kg());
    near(second.producer.raw_mass_transfer_per_kg(), 2 * first.producer.raw_mass_transfer_per_kg());
    near(second.producer.raw_mass_transfer_per_kg() / second.producer.duration_s(),
        first.producer.raw_mass_transfer_per_kg() / first.producer.duration_s());
}

TEST_CASE("SLL inactive regimes use the exact logarithmic ratio gate without diameter clamps",
          "[physics][milbrandt][triple_graupel_hail_conversion]")
{
    auto cell = source();
    cell.source.initial.frozen[0] = spectrum::population(0, .05, 50000, 2);
    const auto blocked = evaluate(cell);
    CHECK(blocked.producer.loading_denominator() < 0);
    CHECK(blocked.producer.regime() == GraupelHailSllRegime::NonpositiveLoading);
    CHECK(blocked.producer.admitted_mass_transfer_per_kg() == 0);
    auto no_accretion = source();
    no_accretion.admitted_accretion_mass_per_kg = {};
    CHECK(evaluate(no_accretion).producer.regime() == GraupelHailSllRegime::NoAccretion);
    no_accretion.source.initial.frozen[2] = {};
    CHECK(evaluate(no_accretion).producer.regime() == GraupelHailSllRegime::NoGraupel);
    CHECK(evaluate(source(), ColdSpectrumDomain::Base200, false).producer.regime() == GraupelHailSllRegime::HailDisabled);
    auto cold = source();
    cold.source.temperature_k = 150;
    cold.source.initial.cloud_mass_per_kg = 0;
    cold.source.initial.frozen[0] = {};
    cold.rain_mass_per_kg = 1e-10;
    cold.admitted_accretion_mass_per_kg = {0, 1e-12, 0};
    const auto safely_inactive = evaluate(cold);
    CHECK(safely_inactive.producer.regime() == GraupelHailSllRegime::BelowSizeRatio);
    CHECK(safely_inactive.producer.raw_mass_transfer_per_kg() == 0);
    CHECK(std::isfinite(safely_inactive.producer.hail_embryo_diameter_m()));
    auto boundary = source();
    const auto independent = reference::evaluate(boundary);
    const double critical = static_cast<double>(static_cast<long double>(273.15)
        - independent.loading * std::log1p(1000 * independent.diameter));
    boundary.source.temperature_k = critical - 1e-7;
    CHECK(evaluate(boundary).producer.regime() == GraupelHailSllRegime::BelowSizeRatio);
    boundary.source.temperature_k = critical + 1e-7;
    const auto admitted = evaluate(boundary);
    CHECK(admitted.producer.regime() == GraupelHailSllRegime::Active);
    CHECK(admitted.producer.size_ratio() >= .1);
}

TEST_CASE("approved higher hail domain admits actual snow-converted graupel shape without a reset",
          "[physics][milbrandt][triple_graupel_hail_conversion]")
{
    CHECK(cold_spectrum_shape_limit(ColdSpectrumDomain::Base200, TripleMomentSpecies::Hail) == 200);
    CHECK(cold_spectrum_shape_limit(ColdSpectrumDomain::SnowConvertedGraupel512, TripleMomentSpecies::Hail) == 200);
    CHECK(cold_spectrum_shape_limit(ColdSpectrumDomain::GraupelAndHail512, TripleMomentSpecies::Hail) == 512);
    CHECK(cold_spectrum_shape_limit(ColdSpectrumDomain::GraupelAndHail512, TripleMomentSpecies::Snow) == 200);
    ColdRimingCell snow_source;
    snow_source.temperature_k = 268.15;
    snow_source.pressure_pa = 85000;
    snow_source.dry_density_kg_m3 = .65;
    snow_source.initial.vapor_mass_per_kg = static_cast<double>(milbrandt_cold_riming_reference::thermal::saturation(268.15, 85000).mass);
    snow_source.initial.cloud_mass_per_kg = 5e-4;
    snow_source.initial.cloud_number_per_kg = 2e8;
    snow_source.initial.frozen[1] = spectrum::population(1, 1e-4, 5000, 200);
    for (int step = 0; step < 4096; ++step)
    {
        snow_source.initial.frozen[1].reflectivity_per_kg = std::nextafter(
            snow_source.initial.frozen[1].reflectivity_per_kg, std::numeric_limits<double>::infinity());
    }
    ColdRimingProducer riming;
    SnowGraupelConversionProducer converted;
    std::string error;
    REQUIRE(produce_exact_cold_riming_admission(parameters(), snow_source, 1e-4, riming,
        error, 4, nullptr, ColdSpectrumDomain::SnowConvertedGraupel512));
    REQUIRE(produce_exact_snow_graupel_conversion_admission(riming, converted, error));
    GraupelHailConversionCell cell = source();
    cell.source.initial.frozen[2] = {converted.mass_increment_per_kg()[2],
        converted.number_increment_per_kg()[2], converted.reflectivity_increment_per_kg()[2]};
    cell.admitted_accretion_mass_per_kg = {.01 * cell.source.initial.frozen[2].mass_per_kg, 0, 0};
    const long double original_alpha = spectrum::represented_alpha(cell.source.initial.frozen[2], 2);
    REQUIRE(original_alpha > 400);
    CHECK(std::abs(original_alpha - 418.7215725131708L) < 1e-4L);
    GraupelHailConversionProducer journal;
    REQUIRE(produce_exact_graupel_hail_conversion_admission(parameters(), cell, 1e-4, journal,
        error, 1, nullptr, ColdSpectrumDomain::SnowConvertedGraupel512));
    REQUIRE(journal.admitted_mass_transfer_per_kg() > 0);
    const auto admitted = evaluate(cell, ColdSpectrumDomain::GraupelAndHail512);
    const long double hail_alpha = spectrum::represented_alpha(admitted.unrounded[3], 3);
    CHECK(hail_alpha > 400);
    CHECK(std::abs(hail_alpha - original_alpha) < 1e-6L);
    near(admitted.mass_increment_per_kg[3], reference::evaluate(cell).admitted);
    auto rejected = admitted;
    GraupelHailConversionFailure failure;
    CHECK_FALSE(evaluate_exact_triple_graupel_hail_conversion_cell(parameters(), cell, 1e-4,
        rejected, error, 1, &failure, ColdSpectrumDomain::SnowConvertedGraupel512));
    CHECK(failure == GraupelHailConversionFailure::IntervalRejected);
    same(rejected, admitted);
    CHECK_FALSE(evaluate_exact_triple_graupel_hail_conversion_cell(parameters(), cell, 1e-4,
        rejected, error, 1, &failure, ColdSpectrumDomain::Base200));
    CHECK(failure == GraupelHailConversionFailure::InvalidSource);
    same(rejected, admitted);
}

TEST_CASE("SLL journal replay binds every source amount context parameter duration and policy before regeneration",
          "[physics][milbrandt][triple_graupel_hail_conversion]")
{
    const auto cell = source();
    const auto accepted = evaluate(cell, ColdSpectrumDomain::GraupelAndHail512);
    for (int mutation = 0; mutation < 28; ++mutation)
    {
        auto changed = cell;
        auto p = parameters();
        double duration = 1e-4;
        auto domain = ColdSpectrumDomain::GraupelAndHail512;
        if (mutation < 24)
        {
            double& value = *fields(changed)[mutation];
            value = value == 0 ? -0.0 : std::nextafter(value, std::numeric_limits<double>::infinity());
        }
        else if (mutation == 24)
        {
            duration = 1e6;
        }
        else if (mutation == 25)
        {
            p.a_g_ = std::nextafter(p.a_g_, std::numeric_limits<double>::infinity());
        }
        else if (mutation == 26)
        {
            p.hail_processes = false;
        }
        else
        {
            domain = ColdSpectrumDomain::SnowConvertedGraupel512;
        }
        auto output = accepted;
        GraupelHailConversionFailure failure;
        std::string error;
        CHECK_FALSE(apply_exact_triple_graupel_hail_conversion_cell(p, changed, duration,
            accepted.producer, output, error, 1, &failure, domain));
        CHECK(failure == GraupelHailConversionFailure::InvalidProducer);
        same(output, accepted);
    }
    auto output = accepted;
    std::string error;
    GraupelHailConversionFailure failure;
    CHECK_FALSE(apply_exact_triple_graupel_hail_conversion_cell(parameters(), cell, 1e-4,
        GraupelHailConversionProducer{}, output, error, 1, &failure, ColdSpectrumDomain::GraupelAndHail512));
    CHECK(failure == GraupelHailConversionFailure::InvalidProducer);
    same(output, accepted);
}

TEST_CASE("SLL validates explicit donors and rejects original graupel borrowing atomically with recovery",
          "[physics][milbrandt][triple_graupel_hail_conversion]")
{
    const auto cell = source();
    const auto accepted = evaluate(cell);
    std::string error;
    GraupelHailConversionFailure failure;
    for (std::size_t capacity : {0u, 2u})
    {
        auto journal = accepted.producer;
        CHECK_FALSE(produce_exact_graupel_hail_conversion_admission(parameters(), cell, 1e-4,
            journal, error, capacity, &failure));
        CHECK(failure == GraupelHailConversionFailure::Capacity);
        same(journal, accepted.producer);
    }
    for (int invalid = 0; invalid < 9; ++invalid)
    {
        auto changed = cell;
        switch (invalid)
        {
            case 0: changed.admitted_accretion_mass_per_kg[0] = .006; break;
            case 1: changed.admitted_accretion_mass_per_kg[1] = .001; break;
            case 2: changed.admitted_accretion_mass_per_kg[2] = .001; break;
            case 3: changed.admitted_accretion_mass_per_kg[0] = -1; break;
            case 4: changed.admitted_accretion_mass_per_kg[1] = std::numeric_limits<double>::quiet_NaN(); break;
            case 5: changed.source.initial.frozen[2].number_per_kg = 0; break;
            case 6: changed.source.initial.frozen[3].reflectivity_per_kg = -1; break;
            case 7: changed.source.initial.cloud_number_per_kg = 0; break;
            case 8: changed.graupel_diffusion_number_per_kg = -1; break;
        }
        auto output = accepted;
        CHECK_FALSE(evaluate_exact_triple_graupel_hail_conversion_cell(parameters(), changed,
            1e-4, output, error, 1, &failure));
        CHECK(failure == GraupelHailConversionFailure::InvalidSource);
        same(output, accepted);
    }
    auto original_overdraw = cell;
    const double multiplier = static_cast<double>(reference::evaluate(cell).ratio / 2);
    original_overdraw.admitted_accretion_mass_per_kg = {
        .75 * cell.source.initial.frozen[2].mass_per_kg / multiplier, 0, 0};
    original_overdraw.graupel_diffusion_mass_per_kg = -.3 * cell.source.initial.frozen[2].mass_per_kg;
    original_overdraw.graupel_diffusion_number_per_kg = cell.source.initial.frozen[2].number_per_kg
        * (original_overdraw.graupel_diffusion_mass_per_kg / cell.source.initial.frozen[2].mass_per_kg);
    REQUIRE(reference::evaluate(original_overdraw).admitted < cell.source.initial.frozen[2].mass_per_kg);
    auto output = accepted;
    CHECK_FALSE(evaluate_exact_triple_graupel_hail_conversion_cell(parameters(), original_overdraw,
        1e-4, output, error, 1, &failure));
    CHECK(failure == GraupelHailConversionFailure::IntervalRejected);
    same(output, accepted);
    auto capped = cell;
    capped.admitted_accretion_mass_per_kg = {cell.source.initial.cloud_mass_per_kg, 0, 0};
    const auto capped_reference = reference::evaluate(capped);
    REQUIRE(capped_reference.raw > capped_reference.admitted);
    REQUIRE(capped_reference.admitted > cell.source.initial.frozen[2].mass_per_kg);
    CHECK_FALSE(evaluate_exact_triple_graupel_hail_conversion_cell(parameters(), capped,
        1e-4, output, error, 1, &failure));
    CHECK(failure == GraupelHailConversionFailure::IntervalRejected);
    same(output, accepted);
    REQUIRE(evaluate_exact_triple_graupel_hail_conversion_cell(parameters(), cell, 1e-4,
        output, error, 1, &failure));
    CHECK(failure == GraupelHailConversionFailure::None);
    same(output, accepted);
}

TEST_CASE("SLL consumes the authoritative five-way diffusion number without changing its arithmetic or accepting a neighboring value",
          "[physics][milbrandt][triple_graupel_hail_conversion]")
{
    auto cell = source();
    cell.rain_mass_per_kg = 0;
    bool distinct_order = false;
    bool active = false;
    for (double duration : {1e-6, 1e-5, 1e-4, 2e-4, 3e-4, 1e-3, 2e-3, 5e-3})
    {
        ColdRimingProducer prefix;
        std::string error;
        REQUIRE(produce_exact_cold_riming_admission(parameters(), cell.source, duration,
            prefix, error, 5, nullptr, ColdSpectrumDomain::GraupelAndHail512, true));
        const auto& channel = prefix.channels()[2];
        REQUIRE(channel.diffusion_mass_per_kg < 0);
        REQUIRE(channel.diffusion_number_per_kg < 0);
        cell.admitted_accretion_mass_per_kg = {channel.collection_mass_per_kg, 0, 0};
        cell.graupel_diffusion_mass_per_kg = channel.diffusion_mass_per_kg;
        cell.graupel_diffusion_number_per_kg = channel.diffusion_number_per_kg;
        const auto& g = cell.source.initial.frozen[2];
        bitwise(cell.graupel_diffusion_number_per_kg,
            g.number_per_kg * (cell.graupel_diffusion_mass_per_kg / g.mass_per_kg));
        const double other_order = g.number_per_kg * cell.graupel_diffusion_mass_per_kg / g.mass_per_kg;
        distinct_order |= other_order != cell.graupel_diffusion_number_per_kg;
        GraupelHailConversionProducer accepted;
        GraupelHailConversionFailure failure;
        const bool captured = produce_exact_graupel_hail_conversion_admission(parameters(), cell,
            duration, accepted, error, 1, &failure, ColdSpectrumDomain::GraupelAndHail512);
        INFO(error);
        REQUIRE(captured);
        CHECK(failure == GraupelHailConversionFailure::None);
        bitwise(accepted.source().graupel_diffusion_number_per_kg, channel.diffusion_number_per_kg);
        active |= accepted.admitted_mass_transfer_per_kg() > 0;
        auto forged = cell;
        forged.graupel_diffusion_number_per_kg = std::nextafter(
            forged.graupel_diffusion_number_per_kg, -std::numeric_limits<double>::infinity());
        auto output = accepted;
        CHECK_FALSE(produce_exact_graupel_hail_conversion_admission(parameters(), forged,
            duration, output, error, 1, &failure, ColdSpectrumDomain::GraupelAndHail512));
        CHECK(failure == GraupelHailConversionFailure::InvalidSource);
        same(output, accepted);
    }
    CHECK(distinct_order);
    CHECK(active);
}
