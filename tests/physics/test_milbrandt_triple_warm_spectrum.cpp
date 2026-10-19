#include "catch2/catch.hpp"
#include "microphysics/milbrandt_warm_parameters.hpp"
#include "microphysics/schemes/milbrandt/triple_warm_spectrum.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace
{

using tmv::milbrandt_detail::ExactRainTripleMomentState;
using tmv::milbrandt_detail::TripleWarmRainSpectrum;

constexpr long double rain_mass_coefficient = 523.598775598L;

long double rising_product(long double alpha, int order)
{
    long double product = 1.0L;
    for (int factor = 1; factor <= order; ++factor)
    {
        product *= alpha + factor;
    }
    return product;
}

long double scaled_gamma_integral(long double alpha, long double order)
{
    constexpr int intervals = 12000;
    const long double upper_x = std::max(
        120.0L, alpha + 1.0L + 30.0L * std::sqrt(alpha + 1.0L) + 80.0L);
    const long double upper_u = std::pow(upper_x, 0.25L);
    const long double step = upper_u / intervals;
    const long double shift = alpha > 0.0L
        ? alpha * std::log(alpha) - alpha : 0.0L;
    long double sum = 0.0L;
    for (int index = 1; index <= intervals; ++index)
    {
        const long double u = index * step;
        const long double x = u * u * u * u;
        const long double integrand = 4.0L * std::exp(
            (4.0L * (alpha + order) + 3.0L) * std::log(u) - x - shift);
        const int weight = index == intervals ? 1 : index % 2 == 0 ? 2 : 4;
        sum += weight * integrand;
    }
    return sum * step / 3.0L;
}

ExactRainTripleMomentState reference_source(double mass, double number,
                                           long double alpha)
{
    const long double shape = rising_product(alpha, 6) /
        std::pow(rising_product(alpha, 3), 2.0L);
    const long double log_z = std::log(shape) +
        2.0L * std::log(static_cast<long double>(mass)) -
        2.0L * std::log(rain_mass_coefficient) -
        std::log(static_cast<long double>(number));
    return {mass, number, static_cast<double>(std::exp(log_z))};
}

long double reference_lambda(const ExactRainTripleMomentState& source,
                             long double alpha)
{
    return std::cbrt(rain_mass_coefficient * source.number_per_kg *
                     rising_product(alpha, 3) / source.mass_per_kg);
}

long double represented_shape_ratio(const ExactRainTripleMomentState& source)
{
    return static_cast<long double>(source.reflectivity_per_kg) *
        source.number_per_kg * rain_mass_coefficient * rain_mass_coefficient /
        (static_cast<long double>(source.mass_per_kg) * source.mass_per_kg);
}

long double represented_alpha_reference(const ExactRainTripleMomentState& source)
{
    const long double ratio = represented_shape_ratio(source);
    long double lower = -1.0L + 1.0e-9L;
    long double upper = 1.0e6L;
    for (int iteration = 0; iteration < 80; ++iteration)
    {
        const long double middle = 0.5L * (lower + upper);
        const long double closure = (middle + 4.0L) * (middle + 5.0L) * (middle + 6.0L) /
            ((middle + 1.0L) * (middle + 2.0L) * (middle + 3.0L));
        if (closure > ratio)
        {
            lower = middle;
        }
        else
        {
            upper = middle;
        }
    }
    return 0.5L * (lower + upper);
}

void check_same_spectrum(const TripleWarmRainSpectrum& actual,
                         const TripleWarmRainSpectrum& expected)
{
    CHECK(actual.source().mass_per_kg == expected.source().mass_per_kg);
    CHECK(actual.source().number_per_kg == expected.source().number_per_kg);
    CHECK(actual.source().reflectivity_per_kg == expected.source().reflectivity_per_kg);
    CHECK(actual.alpha() == expected.alpha());
    CHECK(actual.lambda_per_m() == expected.lambda_per_m());
    CHECK(actual.mean_mass_diameter_m() == expected.mean_mass_diameter_m());
    CHECK(actual.alpha_zero_equivalent_inverse_slope_m() ==
          expected.alpha_zero_equivalent_inverse_slope_m());
    CHECK(actual.diameter_moment_1_m() == expected.diameter_moment_1_m());
    CHECK(actual.diameter_moment_3_m3() == expected.diameter_moment_3_m3());
    CHECK(actual.diameter_moment_6_m6() == expected.diameter_moment_6_m6());
    CHECK(actual.ventilation_diameter_moment() == expected.ventilation_diameter_moment());
    CHECK(actual.ventilation_diameter_order() == expected.ventilation_diameter_order());
}

} // namespace

TEST_CASE("triple warm normalized rain spectra match independent gamma quadrature through alpha 200",
          "[physics][milbrandt][triple_warm_spectrum]")
{
    const std::array<double, 8> orders{0.0, 0.5, 1.0, 1.75, 2.5, 3.0, 4.25, 6.0};
    const std::array<std::array<double, 2>, 3> populations{{
        {{1.0e-6, 1.0e3}}, {{2.0e-4, 1.0e5}}, {{5.0e-3, 2.0e6}}
    }};
    for (double alpha : {-0.5, 0.0, 2.0, 20.0, 200.0})
    {
        std::array<long double, 8> reference_ratios{};
        const long double denominator = scaled_gamma_integral(alpha, 0.0L);
        for (std::size_t index = 0; index < orders.size(); ++index)
        {
            reference_ratios[index] = scaled_gamma_integral(alpha, orders[index]) / denominator;
        }
        for (const auto& population : populations)
        {
            INFO("alpha=" << alpha << " q=" << population[0] << " n=" << population[1]);
            const auto source = reference_source(population[0], population[1], alpha);
            TripleWarmRainSpectrum spectrum;
            std::string error = "sentinel";
            REQUIRE(tmv::milbrandt_detail::diagnose_triple_warm_rain_spectrum(
                source, spectrum, error));
            CHECK(error.empty());
            CHECK(spectrum.source().mass_per_kg == source.mass_per_kg);
            CHECK(spectrum.source().number_per_kg == source.number_per_kg);
            CHECK(spectrum.source().reflectivity_per_kg == source.reflectivity_per_kg);
            CHECK(spectrum.alpha() == Approx(alpha).epsilon(2.0e-10).margin(2.0e-10));
            const long double lambda = reference_lambda(source, alpha);
            CHECK(spectrum.lambda_per_m() == Approx(static_cast<double>(lambda)).epsilon(2.0e-12));
            const long double represented_alpha = represented_alpha_reference(source);
            CHECK(spectrum.alpha() ==
                  Approx(static_cast<double>(represented_alpha)).epsilon(2.0e-12).margin(2.0e-12));
            CHECK(spectrum.lambda_per_m() ==
                  Approx(static_cast<double>(reference_lambda(source, represented_alpha))).epsilon(2.0e-12));
            const long double diagnosed_alpha = spectrum.alpha();
            const long double diagnosed_shape =
                (diagnosed_alpha + 4.0L) * (diagnosed_alpha + 5.0L) * (diagnosed_alpha + 6.0L) /
                ((diagnosed_alpha + 1.0L) * (diagnosed_alpha + 2.0L) * (diagnosed_alpha + 3.0L));
            CHECK(static_cast<double>(diagnosed_shape) ==
                  Approx(static_cast<double>(represented_shape_ratio(source))).epsilon(4.0e-14));
            TripleWarmRainSpectrum repeated;
            REQUIRE(tmv::milbrandt_detail::diagnose_triple_warm_rain_spectrum(source, repeated, error));
            check_same_spectrum(repeated, spectrum);
            for (std::size_t index = 0; index < orders.size(); ++index)
            {
                double moment = -7.0;
                REQUIRE(spectrum.normalized_diameter_moment(orders[index], moment, error));
                CHECK(error.empty());
                const long double quadrature = reference_ratios[index] /
                    std::pow(lambda, static_cast<long double>(orders[index]));
                CHECK(moment == Approx(static_cast<double>(quadrature)).epsilon(8.0e-12));
                if (orders[index] == std::floor(orders[index]))
                {
                    const long double polynomial = rising_product(alpha, static_cast<int>(orders[index])) /
                        std::pow(lambda, static_cast<long double>(orders[index]));
                    CHECK(moment == Approx(static_cast<double>(polynomial)).epsilon(2.0e-12));
                }
            }
            CHECK(spectrum.diameter_moment_1_m() ==
                  Approx(static_cast<double>((alpha + 1.0L) / lambda)).epsilon(2.0e-12));
            CHECK(spectrum.diameter_moment_3_m3() ==
                  Approx(source.mass_per_kg / (static_cast<double>(rain_mass_coefficient) *
                                               source.number_per_kg)).epsilon(2.0e-12));
            CHECK(spectrum.diameter_moment_6_m6() ==
                  Approx(source.reflectivity_per_kg / source.number_per_kg).epsilon(2.0e-12));
            double ventilation = -7.0;
            REQUIRE(spectrum.normalized_diameter_moment(
                spectrum.ventilation_diameter_order(), ventilation, error));
            CHECK(spectrum.ventilation_diameter_moment() == ventilation);
            const double mean_diameter = std::cbrt(source.mass_per_kg /
                (static_cast<double>(rain_mass_coefficient) * source.number_per_kg));
            CHECK(spectrum.mean_mass_diameter_m() == Approx(mean_diameter).epsilon(3.0e-14));
            CHECK(spectrum.alpha_zero_equivalent_inverse_slope_m() ==
                  Approx(mean_diameter / std::cbrt(6.0)).epsilon(3.0e-14));
        }
    }
}

TEST_CASE("triple warm default moments preserve the fixed alpha zero law without raw gamma overflow",
          "[physics][milbrandt][triple_warm_spectrum]")
{
    std::string error;
    const auto defaults = tmv::milbrandt_detail::warm_rain_parameters(
        0.0, static_cast<double>(rain_mass_coefficient), 3.0, 149.1, 0.5);
    const auto source = reference_source(2.0e-4, 1.0e5, 0.0L);
    TripleWarmRainSpectrum spectrum;
    REQUIRE(tmv::milbrandt_detail::diagnose_triple_warm_rain_spectrum(source, spectrum, error));
    const double inverse_lambda = std::cbrt(source.mass_per_kg / source.number_per_kg *
                                           defaults.inv_rain_mass_factor);
    CHECK(spectrum.diameter_moment_1_m() == Approx(inverse_lambda).epsilon(2.0e-13));
    CHECK(spectrum.diameter_moment_3_m3() ==
          Approx(defaults.gr34 / defaults.gr31 * std::pow(inverse_lambda, 3)).epsilon(2.0e-13));
    CHECK(spectrum.diameter_moment_6_m6() ==
          Approx(defaults.gr37 / defaults.gr31 * std::pow(inverse_lambda, 6)).epsilon(2.0e-13));
    CHECK(spectrum.ventilation_diameter_moment() ==
          Approx(defaults.gamma_fall_ventilation / defaults.gr31 *
                 std::pow(inverse_lambda, 1.75)).epsilon(2.0e-13));
    CHECK(spectrum.alpha_zero_equivalent_inverse_slope_m() ==
          Approx(inverse_lambda).epsilon(3.0e-14));

    const auto raw_high_shape = tmv::milbrandt_detail::warm_rain_parameters(
        200.0, static_cast<double>(rain_mass_coefficient), 3.0, 149.1, 0.5);
    CHECK_FALSE(std::isfinite(raw_high_shape.gr31));
    const auto high_shape_source = reference_source(2.0e-4, 1.0e5, 200.0L);
    TripleWarmRainSpectrum high_shape;
    REQUIRE(tmv::milbrandt_detail::diagnose_triple_warm_rain_spectrum(
        high_shape_source, high_shape, error));
    CHECK(std::isfinite(high_shape.ventilation_diameter_moment()));
    CHECK(high_shape.ventilation_diameter_moment() > 0.0);
    CHECK(high_shape.mean_mass_diameter_m() == spectrum.mean_mass_diameter_m());
    CHECK(high_shape.alpha_zero_equivalent_inverse_slope_m() ==
          spectrum.alpha_zero_equivalent_inverse_slope_m());
    CHECK(high_shape.diameter_moment_6_m6() < spectrum.diameter_moment_6_m6());

    const auto corrected_source = ExactRainTripleMomentState{
        high_shape_source.mass_per_kg, 4.0 * high_shape_source.number_per_kg,
        high_shape_source.reflectivity_per_kg / 4.0};
    TripleWarmRainSpectrum corrected;
    REQUIRE(tmv::milbrandt_detail::diagnose_triple_warm_rain_spectrum(
        corrected_source, corrected, error));
    CHECK(corrected.alpha() == Approx(high_shape.alpha()).epsilon(2.0e-10).margin(2.0e-10));
    CHECK(corrected.mean_mass_diameter_m() ==
          Approx(high_shape.mean_mass_diameter_m() / std::cbrt(4.0)).epsilon(3.0e-14));
    CHECK(corrected.diameter_moment_3_m3() ==
          Approx(high_shape.diameter_moment_3_m3() / 4.0).epsilon(2.0e-12));
    CHECK(corrected.diameter_moment_6_m6() ==
          Approx(high_shape.diameter_moment_6_m6() / 16.0).epsilon(2.0e-12));
}

TEST_CASE("triple warm spectrum rejects invalid and nonrepresentable inputs atomically",
          "[physics][milbrandt][triple_warm_spectrum]")
{
    std::string error;
    const auto source = reference_source(2.0e-4, 1.0e5, 20.0L);
    TripleWarmRainSpectrum spectrum;
    REQUIRE(tmv::milbrandt_detail::diagnose_triple_warm_rain_spectrum(source, spectrum, error));
    const auto before = spectrum;
    for (std::size_t member = 0; member < 3; ++member)
    {
        for (double invalid : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity()})
        {
            auto bad = source;
            std::array<double*, 3> values{
                &bad.mass_per_kg, &bad.number_per_kg, &bad.reflectivity_per_kg};
            *values[member] = invalid;
            CHECK_FALSE(tmv::milbrandt_detail::diagnose_triple_warm_rain_spectrum(
                bad, spectrum, error));
            CHECK_FALSE(error.empty());
            check_same_spectrum(spectrum, before);
        }
    }
    const double lower_bound = source.mass_per_kg * source.mass_per_kg /
        (static_cast<double>(rain_mass_coefficient * rain_mass_coefficient) * source.number_per_kg);
    const std::array<ExactRainTripleMomentState, 7> invalid_populations{{
        {},
        {source.mass_per_kg, source.number_per_kg, 0.5 * lower_bound},
        {source.mass_per_kg, source.number_per_kg, lower_bound},
        reference_source(1.0e-70, 1.0e150, 20.0L),
        reference_source(1.0e20, 1.0e-150, 20.0L),
        reference_source(2.0e-4, 1.0e5, -1.0L + 5.0e-10L),
        reference_source(2.0e-4, 1.0e5, 600000.0L)
    }};
    for (const auto& bad : invalid_populations)
    {
        CHECK_FALSE(tmv::milbrandt_detail::diagnose_triple_warm_rain_spectrum(
            bad, spectrum, error));
        CHECK_FALSE(error.empty());
        check_same_spectrum(spectrum, before);
    }
    for (double order : {-1.0, 6.01, std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::infinity()})
    {
        double output = -7.0;
        CHECK_FALSE(spectrum.normalized_diameter_moment(order, output, error));
        CHECK(output == -7.0);
        CHECK_FALSE(error.empty());
        check_same_spectrum(spectrum, before);
    }
    const TripleWarmRainSpectrum unavailable;
    double output = -7.0;
    CHECK_FALSE(unavailable.normalized_diameter_moment(1.0, output, error));
    CHECK(output == -7.0);
    CHECK_FALSE(error.empty());
    REQUIRE(spectrum.normalized_diameter_moment(1.0, output, error));
    CHECK(error.empty());
    CHECK(output == spectrum.diameter_moment_1_m());
    check_same_spectrum(spectrum, before);
}
