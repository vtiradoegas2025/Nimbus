#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_cloud_collection.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_cloud_diffusion.hpp"
#include "microphysics/schemes/milbrandt/triple_rime_splintering.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace
{

using namespace tmv::milbrandt_detail;
constexpr ColdSpectrumDomain expanded = ColdSpectrumDomain::SnowConvertedGraupel512;

ColdParameters parameters()
{
    const auto& p = default_cold_parameters;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], true};
}

long double ratio(long double alpha)
{
    return (alpha + 4) * (alpha + 5) * (alpha + 6)
        / ((alpha + 1) * (alpha + 2) * (alpha + 3));
}

FrozenCollectionPopulation population(long double alpha, bool stored)
{
    const double q = static_cast<float>(.001);
    const double n = 1000;
    constexpr double coefficient = 209.439510239;
    const double z = static_cast<double>(ratio(alpha) * static_cast<long double>(q) * q
        / (static_cast<long double>(coefficient) * coefficient * n));
    return {q, n, stored ? static_cast<double>(static_cast<float>(z)) : z};
}

long double represented_alpha(const FrozenCollectionPopulation& value)
{
    constexpr long double coefficient = static_cast<long double>(209.439510239);
    const long double measured = coefficient * coefficient * value.number_per_kg * value.reflectivity_per_kg
        / (static_cast<long double>(value.mass_per_kg) * value.mass_per_kg);
    long double lower = -1 + 1e-9L;
    long double upper = 1024;
    for (int iteration = 0; iteration < 120; ++iteration)
    {
        const long double middle = (lower + upper) / 2;
        if (ratio(middle) > measured)
        {
            lower = middle;
        }
        else
        {
            upper = middle;
        }
    }
    return (lower + upper) / 2;
}

ColdCloudCollectionCell collection(long double alpha, bool stored)
{
    ColdCloudCollectionCell source{253.15, 90000, .65, {}};
    source.initial.frozen[1] = population(alpha, stored);
    return source;
}

ColdCloudDiffusionCell diffusion(const ColdCloudCollectionCell& source)
{
    return {source.temperature_k, source.pressure_pa, source.dry_density_kg_m3,
        tmv::thermodynamics::saturation_mixing_ratio_ice(source.temperature_k, source.pressure_pa), source.initial};
}

RimeSplinteringCell splintering(const ColdCloudCollectionCell& source)
{
    RimeSplinteringCell value;
    value.temperature_k = source.temperature_k;
    value.pressure_pa = source.pressure_pa;
    value.dry_density_kg_m3 = source.dry_density_kg_m3;
    value.initial[2] = source.initial.frozen[1];
    return value;
}

bool same(double a, double b)
{
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

void same_population(const FrozenCollectionPopulation& a, const FrozenCollectionPopulation& b)
{
    CHECK(same(a.mass_per_kg, b.mass_per_kg));
    CHECK(same(a.number_per_kg, b.number_per_kg));
    CHECK(same(a.reflectivity_per_kg, b.reflectivity_per_kg));
}

}

TEST_CASE("explicit converted graupel domain admits represented higher spectra across cold operators",
          "[physics][milbrandt][triple_cold_spectrum_domain]")
{
    for (const long double shape : {201.0L, 418.72L, 500.0L})
    {
        for (const bool stored : {false, true})
        {
            const auto source = collection(shape, stored);
            const auto measured = represented_alpha(source.initial.frozen[1]);
            REQUIRE(measured > 200);
            REQUIRE(measured < 512);
            std::string error;
            ColdCloudCollectionResult collected;
            ColdCloudCollectionFailure cf;
            REQUIRE(evaluate_exact_triple_cold_cloud_collection_cell(parameters(), source, 1e-4,
                collected, error, 3, &cf, expanded));
            REQUIRE(collected.producer.spectrum_domain() == expanded);
            CHECK(collected.producer.source_alpha()[1] == Approx(static_cast<double>(measured)).epsilon(2e-10));
            ColdCloudCollectionResult replay;
            REQUIRE(apply_exact_triple_cold_cloud_collection_cell(parameters(), source, 1e-4,
                collected.producer, replay, error, 3, &cf, expanded));
            same_population(collected.unrounded.frozen[1], replay.unrounded.frozen[1]);
            CHECK_FALSE(apply_exact_triple_cold_cloud_collection_cell(parameters(), source, 1e-4,
                collected.producer, replay, error));
            CHECK_FALSE(evaluate_exact_triple_cold_cloud_collection_cell(parameters(), source, 1e-4,
                replay, error));

            const auto ds = diffusion(source);
            ColdCloudDiffusionResult diffused;
            ColdCloudDiffusionFailure df;
            REQUIRE(evaluate_exact_triple_cold_cloud_diffusion_cell(parameters(), ds, 1e-4,
                diffused, error, 3, &df, expanded));
            REQUIRE(diffused.producer.spectrum_domain() == expanded);
            REQUIRE(diffused.producer.collection_producer().spectrum_domain() == expanded);
            ColdCloudDiffusionResult dr;
            REQUIRE(apply_exact_triple_cold_cloud_diffusion_cell(parameters(), ds, 1e-4,
                diffused.producer, dr, error, 3, &df, expanded));
            same_population(diffused.unrounded.frozen[1], dr.unrounded.frozen[1]);
            CHECK_FALSE(apply_exact_triple_cold_cloud_diffusion_cell(parameters(), ds, 1e-4,
                diffused.producer, dr, error));
            CHECK_FALSE(evaluate_exact_triple_cold_cloud_diffusion_cell(parameters(), ds, 1e-4, dr, error));

            const auto hs = splintering(source);
            RimeSplinteringResult hm;
            RimeSplinteringFailure hf;
            REQUIRE(evaluate_exact_triple_rime_splintering_cell(parameters(), hs, 1e-4,
                hm, error, 3, &hf, expanded));
            REQUIRE(hm.producer.spectrum_domain() == expanded);
            RimeSplinteringResult hr;
            REQUIRE(apply_exact_triple_rime_splintering_cell(parameters(), hs, 1e-4,
                hm.producer, hr, error, 3, &hf, expanded));
            same_population(hm.unrounded[2], hr.unrounded[2]);
            CHECK_FALSE(apply_exact_triple_rime_splintering_cell(parameters(), hs, 1e-4, hm.producer, hr, error));
            CHECK_FALSE(evaluate_exact_triple_rime_splintering_cell(parameters(), hs, 1e-4, hr, error));
        }
    }
}

TEST_CASE("converted graupel domain is bounded and cannot widen other species or invalid policies",
          "[physics][milbrandt][triple_cold_spectrum_domain]")
{
    for (const auto policy : {ColdSpectrumDomain::Base200, expanded})
    {
        CHECK(cold_spectrum_shape_limit(policy, TripleMomentSpecies::Snow) == 200);
        CHECK(cold_spectrum_shape_limit(policy, TripleMomentSpecies::Hail) == 200);
        CHECK(cold_spectrum_shape_limit(policy, TripleMomentSpecies::Ice) == 200);
        CHECK(cold_spectrum_shape_limit(policy, TripleMomentSpecies::Rain) == 200);
    }
    for (const bool stored : {false, true})
    {
        auto source = collection(513, stored);
        REQUIRE(represented_alpha(source.initial.frozen[1]) > 512);
        std::string error;
        ColdCloudCollectionResult c;
        CHECK_FALSE(evaluate_exact_triple_cold_cloud_collection_cell(parameters(), source, 1e-4, c,
            error, 3, nullptr, expanded));
        ColdCloudDiffusionResult d;
        CHECK_FALSE(evaluate_exact_triple_cold_cloud_diffusion_cell(parameters(), diffusion(source), 1e-4, d,
            error, 3, nullptr, expanded));
        RimeSplinteringResult h;
        CHECK_FALSE(evaluate_exact_triple_rime_splintering_cell(parameters(), splintering(source), 1e-4, h,
            error, 3, nullptr, expanded));
    }
    const auto invalid = static_cast<ColdSpectrumDomain>(-1);
    CHECK_FALSE(valid_cold_spectrum_domain(invalid));
    auto source = collection(2, true);
    std::string error;
    ColdCloudCollectionFailure cf;
    ColdCloudCollectionProducer c;
    CHECK_FALSE(produce_exact_cold_cloud_collection_admission(parameters(), source, 1e-4, c, error, 3, &cf, invalid));
    CHECK(cf == ColdCloudCollectionFailure::InvalidContext);
    ColdCloudDiffusionFailure df;
    ColdCloudDiffusionProducer d;
    CHECK_FALSE(produce_exact_cold_cloud_diffusion_admission(parameters(), diffusion(source), 1e-4, d, error, 3, &df, invalid));
    CHECK(df == ColdCloudDiffusionFailure::InvalidContext);
    RimeSplinteringFailure hf;
    RimeSplinteringProducer h;
    CHECK_FALSE(produce_exact_rime_splintering_admission(parameters(), splintering(source), 1e-4, h, error, 3, &hf, invalid));
    CHECK(hf == RimeSplinteringFailure::InvalidContext);
}

TEST_CASE("default cold spectrum domain preserves base numerical endpoints and binds replay policy",
          "[physics][milbrandt][triple_cold_spectrum_domain]")
{
    auto source = collection(2, true);
    source.initial.cloud_mass_per_kg = 1e-4;
    source.initial.cloud_number_per_kg = 1e8;
    std::string error;
    ColdCloudCollectionResult old, base, extended;
    REQUIRE(evaluate_exact_triple_cold_cloud_collection_cell(parameters(), source, 1e-4, old, error));
    REQUIRE(evaluate_exact_triple_cold_cloud_collection_cell(parameters(), source, 1e-4, base, error,
        3, nullptr, ColdSpectrumDomain::Base200));
    REQUIRE(evaluate_exact_triple_cold_cloud_collection_cell(parameters(), source, 1e-4, extended, error,
        3, nullptr, expanded));
    for (std::size_t index = 0; index < 3; ++index)
    {
        same_population(old.unrounded.frozen[index], base.unrounded.frozen[index]);
        same_population(old.unrounded.frozen[index], extended.unrounded.frozen[index]);
    }
    CHECK(same(old.temperature_increment_k, base.temperature_increment_k));
    CHECK(same(old.temperature_increment_k, extended.temperature_increment_k));
    ColdCloudCollectionResult sentinel;
    sentinel.temperature_increment_k = 42;
    CHECK_FALSE(apply_exact_triple_cold_cloud_collection_cell(parameters(), source, 1e-4, old.producer,
        sentinel, error, 3, nullptr, expanded));
    CHECK(sentinel.temperature_increment_k == 42);

    const auto ds = diffusion(source);
    ColdCloudDiffusionResult d0, d1, d2;
    REQUIRE(evaluate_exact_triple_cold_cloud_diffusion_cell(parameters(), ds, 1e-4, d0, error));
    REQUIRE(evaluate_exact_triple_cold_cloud_diffusion_cell(parameters(), ds, 1e-4, d1, error,
        3, nullptr, ColdSpectrumDomain::Base200));
    REQUIRE(evaluate_exact_triple_cold_cloud_diffusion_cell(parameters(), ds, 1e-4, d2, error,
        3, nullptr, expanded));
    for (std::size_t index = 0; index < 3; ++index)
    {
        same_population(d0.unrounded.frozen[index], d1.unrounded.frozen[index]);
        same_population(d0.unrounded.frozen[index], d2.unrounded.frozen[index]);
    }
    CHECK(same(d0.temperature_increment_k, d1.temperature_increment_k));
    CHECK(same(d0.temperature_increment_k, d2.temperature_increment_k));
    d1.temperature_increment_k = 42;
    CHECK_FALSE(apply_exact_triple_cold_cloud_diffusion_cell(parameters(), ds, 1e-4, d0.producer,
        d1, error, 3, nullptr, expanded));
    CHECK(d1.temperature_increment_k == 42);
    auto hs = splintering(source);
    hs.temperature_k = 268.15;
    hs.cloud_mass_per_kg = 1e-4;
    hs.admitted_cloud_mass_per_kg[2] = 1e-10;
    const double ice_mass = static_cast<float>(1e-5);
    hs.initial[0] = {ice_mass, 1000, static_cast<double>(ratio(2) * ice_mass * ice_mass / (440.0L * 440.0L * 1000))};
    RimeSplinteringResult h0, h1, h2;
    REQUIRE(evaluate_exact_triple_rime_splintering_cell(parameters(), hs, 1e-4, h0, error));
    REQUIRE(evaluate_exact_triple_rime_splintering_cell(parameters(), hs, 1e-4, h1, error,
        3, nullptr, ColdSpectrumDomain::Base200));
    REQUIRE(evaluate_exact_triple_rime_splintering_cell(parameters(), hs, 1e-4, h2, error,
        3, nullptr, expanded));
    for (std::size_t index = 0; index < 4; ++index)
    {
        same_population(h0.unrounded[index], h1.unrounded[index]);
        same_population(h0.unrounded[index], h2.unrounded[index]);
    }
    h1.temperature_increment_k = 42;
    CHECK_FALSE(apply_exact_triple_rime_splintering_cell(parameters(), hs, 1e-4, h0.producer,
        h1, error, 3, nullptr, expanded));
    CHECK(h1.temperature_increment_k == 42);
}

TEST_CASE("converted spectrum ceiling tests represented boundary without admitting other high shape species",
          "[physics][milbrandt][triple_cold_spectrum_domain]")
{
    auto source = collection(512, true);
    source.initial.frozen[1].reflectivity_per_kg = std::nextafter(
        static_cast<float>(source.initial.frozen[1].reflectivity_per_kg), std::numeric_limits<float>::infinity());
    const auto measured = represented_alpha(source.initial.frozen[1]);
    REQUIRE(measured > 511.9);
    REQUIRE(measured <= 512);
    std::string error;
    ColdCloudCollectionResult c;
    REQUIRE(evaluate_exact_triple_cold_cloud_collection_cell(parameters(), source, 1e-4,
        c, error, 3, nullptr, expanded));
    ColdCloudDiffusionResult d;
    REQUIRE(evaluate_exact_triple_cold_cloud_diffusion_cell(parameters(), diffusion(source), 1e-4,
        d, error, 3, nullptr, expanded));
    RimeSplinteringResult h;
    REQUIRE(evaluate_exact_triple_rime_splintering_cell(parameters(), splintering(source), 1e-4,
        h, error, 3, nullptr, expanded));
    for (const TripleMomentSpecies category : {TripleMomentSpecies::Snow, TripleMomentSpecies::Hail, TripleMomentSpecies::Ice})
    {
        const auto law = current_nimbus_triple_sedimentation_parameters(category).mass_law;
        constexpr long double shape = 201;
        const double q = static_cast<float>(.001);
        const double n = 1000;
        const long double log_lambda = (std::log(static_cast<long double>(law.coefficient) * n / q)
            + std::lgamma(shape + law.exponent + 1) - std::lgamma(shape + 1)) / law.exponent;
        const double z = static_cast<double>(std::exp(std::log(static_cast<long double>(n))
            + std::lgamma(shape + 7) - std::lgamma(shape + 1) - 6 * log_lambda));
        ColdCloudCollectionCell non_g{253.15, 90000, .65, {}};
        if (category != TripleMomentSpecies::Ice)
        {
            const std::size_t index = category == TripleMomentSpecies::Snow ? 0 : 2;
            non_g.initial.frozen[index] = {q, n, z};
            CHECK_FALSE(evaluate_exact_triple_cold_cloud_collection_cell(parameters(), non_g, 1e-4,
                c, error, 3, nullptr, expanded));
            CHECK_FALSE(evaluate_exact_triple_cold_cloud_diffusion_cell(parameters(), diffusion(non_g), 1e-4,
                d, error, 3, nullptr, expanded));
        }
        RimeSplinteringCell hm{253.15, 90000, .65, 0, {}, {}};
        const std::size_t index = category == TripleMomentSpecies::Ice ? 0 : category == TripleMomentSpecies::Snow ? 1 : 3;
        hm.initial[index] = {q, n, z};
        CHECK_FALSE(evaluate_exact_triple_rime_splintering_cell(parameters(), hm, 1e-4,
            h, error, 3, nullptr, expanded));
    }
}

TEST_CASE("higher converted graupel spectra advance nonzero collection diffusion and HM transfers",
          "[physics][milbrandt][triple_cold_spectrum_domain]")
{
    for (const long double shape : {201.0L, 418.72L, 500.0L})
    {
        auto source = collection(shape, true);
        source.temperature_k = 268.15;
        source.initial.cloud_mass_per_kg = 1e-4;
        source.initial.cloud_number_per_kg = 1e8;
        std::string error;
        ColdCloudCollectionResult c;
        const bool collected = evaluate_exact_triple_cold_cloud_collection_cell(parameters(), source, 1e-7,
            c, error, 3, nullptr, expanded);
        INFO(error);
        REQUIRE(collected);
        CHECK(c.mass_increment_per_kg[1] > 0);
        CHECK(c.reflectivity_increment_per_kg[1] > 0);
        auto ds = diffusion(source);
        ds.vapor_mass_per_kg *= .9;
        ColdCloudDiffusionResult d;
        const bool diffused = evaluate_exact_triple_cold_cloud_diffusion_cell(parameters(), ds, 1e-7,
            d, error, 3, nullptr, expanded);
        INFO(error);
        REQUIRE(diffused);
        CHECK(d.producer.channels()[1].admitted_mass_increment_per_kg < 0);
        CHECK(d.producer.channels()[1].admitted_number_increment_per_kg < 0);
        auto hs = splintering(source);
        hs.cloud_mass_per_kg = 1e-4;
        hs.admitted_cloud_mass_per_kg[2] = 1e-10;
        const double ice_mass = static_cast<float>(1e-5);
        hs.initial[0] = {ice_mass, 1000, static_cast<double>(ratio(2) * ice_mass * ice_mass / (440.0L * 440.0L * 1000))};
        RimeSplinteringResult h;
        const bool splintered = evaluate_exact_triple_rime_splintering_cell(parameters(), hs, 1e-7,
            h, error, 3, nullptr, expanded);
        INFO(error);
        REQUIRE(splintered);
        CHECK(h.number_increment_per_kg[0] > 0);
        CHECK(h.mass_increment_per_kg[2] < 0);
    }
}
