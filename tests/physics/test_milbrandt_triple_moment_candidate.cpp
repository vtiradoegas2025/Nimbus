#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/triple_moment.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace
{

using tmv::milbrandt_detail::TripleMomentCandidateState;
using tmv::milbrandt_detail::TripleMomentFieldRefs;
using tmv::milbrandt_detail::TripleMomentMassLaws;

constexpr TripleMomentMassLaws kLaws{{
    {523.598775598, 3.0},
    {440.0, 3.0},
    {0.1597, 2.078},
    {209.439510239, 3.0},
    {471.238898038, 3.0}
}};

long double gamma_integral(long double order)
{
    constexpr int intervals = 8000;
    constexpr long double upper = 80.0L;
    const long double step = upper / intervals;
    long double sum = 0.0L;
    for (int index = 0; index <= intervals; ++index)
    {
        const long double position = step * index;
        const long double value = std::pow(position, order) * std::exp(-position);
        const int weight = index == 0 || index == intervals ? 1 : index % 2 == 0 ? 2 : 4;
        sum += weight * value;
    }
    return sum * step / 3.0L;
}

long double integrated_moment(long double density,
                              long double mass_per_kg,
                              long double number_per_kg,
                              long double alpha,
                              tmv::milbrandt_detail::TripleMomentMassLaw law,
                              long double order)
{
    const long double number_volume = density * number_per_kg;
    const long double mass_volume = density * mass_per_kg;
    const long double slope = std::pow(
        static_cast<long double>(law.coefficient) * number_volume *
            std::tgamma(alpha + static_cast<long double>(law.exponent) + 1.0L) /
            (mass_volume * std::tgamma(alpha + 1.0L)),
        1.0L / static_cast<long double>(law.exponent));
    return number_volume * gamma_integral(alpha + order) /
        (std::tgamma(alpha + 1.0L) * std::pow(slope, order));
}

TripleMomentFieldRefs refs(const std::array<Field3D, 5>& fields)
{
    return {&fields[0], &fields[1], &fields[2], &fields[3], &fields[4]};
}

} // namespace

TEST_CASE("five isolated Milbrandt sixth moments follow independent volumetric gamma integrals",
          "[physics][milbrandt][triple_candidate]")
{
    constexpr std::array<double, 5> mass{{1.0e-4, 8.0e-5, 5.0e-5, 7.0e-5, 9.0e-5}};
    constexpr std::array<double, 5> number{{1.0e6, 2.0e5, 8.0e4, 3.0e4, 1.0e4}};
    for (double alpha : {0.0, 2.0, 5.0})
    {
        for (double density : {0.65, 1.2, 1.8})
        {
            for (std::size_t species = 0; species < kLaws.size(); ++species)
            {
                float represented = 0.0f;
                REQUIRE(tmv::milbrandt_detail::gamma_sixth_moment_per_kg(
                    density, mass[species], number[species], alpha,
                    kLaws[species], represented));
                const long double m0 = integrated_moment(
                    density, mass[species], number[species], alpha,
                    kLaws[species], 0.0L);
                const long double md = integrated_moment(
                    density, mass[species], number[species], alpha,
                    kLaws[species], kLaws[species].exponent);
                const long double m6 = integrated_moment(
                    density, mass[species], number[species], alpha,
                    kLaws[species], 6.0L);
                CHECK(std::abs(static_cast<double>(m0 / density) - number[species]) /
                    number[species] < 5.0e-5);
                CHECK(std::abs(static_cast<double>(kLaws[species].coefficient * md / density) -
                               mass[species]) / mass[species] < 5.0e-5);
                CHECK(std::abs(static_cast<double>(m6 / density) - represented) /
                    represented < 5.0e-5);
            }
        }
    }

    float represented = 1.0f;
    CHECK_FALSE(tmv::milbrandt_detail::gamma_sixth_moment_per_kg(
        1.0, 1.0e-4, 1.0e5, -1.0, kLaws[0], represented));
    CHECK_FALSE(tmv::milbrandt_detail::gamma_sixth_moment_per_kg(
        1.0, 1.0e-4, 0.0, 0.0, kLaws[0], represented));
    CHECK_FALSE(tmv::milbrandt_detail::gamma_sixth_moment_per_kg(
        1.0, 1.0e30, 1.0, 0.0, kLaws[0], represented));
    CHECK_FALSE(tmv::milbrandt_detail::gamma_sixth_moment_per_kg(
        std::numeric_limits<double>::quiet_NaN(), 1.0e-4, 1.0e5,
        0.0, kLaws[0], represented));
    CHECK_FALSE(tmv::milbrandt_detail::gamma_sixth_moment_per_kg(
        1.0, 1.0e-4, 1.0e5, 0.0,
        {kLaws[0].coefficient, 6.0}, represented));
    CHECK(represented == 1.0f);
}

TEST_CASE("isolated five-Z state rejects unrealizable and late invalid candidates atomically",
          "[physics][milbrandt][triple_candidate]")
{
    Field3D density(2, 2, 2, 1.2f);
    density(0, 0, 0) = 0.65f;
    density(1, 1, 1) = 1.8f;
    std::array<Field3D, 5> mass;
    std::array<Field3D, 5> number;
    for (std::size_t species = 0; species < mass.size(); ++species)
    {
        mass[species].resize(2, 2, 2, 1.0e-4f);
        number[species].resize(2, 2, 2, 1.0e5f);
    }
    TripleMomentCandidateState state;
    std::string error;
    REQUIRE(state.prepare(density, refs(mass), refs(number), kLaws,
                          {0.0, 2.0, 5.0, 0.0, 2.0}, error));
    for (std::size_t species = 0; species < mass.size(); ++species)
    {
        CHECK(state.moment(species).size() == density.size());
        CHECK(state.tendency(species)(0, 0, 0) == 0.0f);
    }
    CHECK(state.moment(0)(0, 0, 0) == state.moment(0)(1, 1, 1));

    const float original_first = state.moment(0)(0, 0, 0);
    const float original_last = state.moment(4)(0, 0, 0);
    state.tendency(0)(0, 0, 0) = original_first * 0.1f;
    state.tendency(4)(0, 0, 0) = -2.0f * original_last;
    CHECK_FALSE(state.commit(1.0, density, refs(mass), refs(number), kLaws, error));
    CHECK(state.moment(0)(0, 0, 0) == original_first);
    CHECK(state.moment(4)(0, 0, 0) == original_last);
    state.tendency(4)(0, 0, 0) = 0.0f;

    const double lower = 1.0e5 * std::pow(
        1.0e-4 / (kLaws[0].coefficient * 1.0e5), 6.0 / kLaws[0].exponent);
    state.tendency(0)(0, 0, 0) = static_cast<float>(0.5 * lower - original_first);
    CHECK_FALSE(state.commit(1.0, density, refs(mass), refs(number), kLaws, error));
    CHECK(state.moment(0)(0, 0, 0) == original_first);
    state.tendency(0)(0, 0, 0) = original_first * 0.1f;

    std::array<Field3D, 5> wrong_shape = mass;
    wrong_shape[4].resize(1, 1, 1, 1.0e-4f);
    CHECK_FALSE(state.commit(1.0, density, refs(wrong_shape), refs(number), kLaws, error));
    CHECK(state.moment(0)(0, 0, 0) == original_first);
    CHECK_FALSE(state.commit(0.0, density, refs(mass), refs(number), kLaws, error));
    CHECK(state.moment(0)(0, 0, 0) == original_first);

    state.tendency(4)(0, 0, 0) = std::numeric_limits<float>::max();
    CHECK_FALSE(state.commit(2.0, density, refs(mass), refs(number), kLaws, error));
    CHECK(state.moment(0)(0, 0, 0) == original_first);
    state.tendency(4)(0, 0, 0) = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(state.commit(1.0, density, refs(mass), refs(number), kLaws, error));
    CHECK(state.moment(0)(0, 0, 0) == original_first);
    state.tendency(4)(0, 0, 0) = 0.0f;
    REQUIRE(state.commit(1.0, density, refs(mass), refs(number), kLaws, error));
    CHECK(state.moment(0)(0, 0, 0) > original_first);

    mass[4](0, 0, 0) = -1.0f;
    const float committed_first = state.moment(0)(0, 0, 0);
    CHECK_FALSE(state.prepare(density, refs(mass), refs(number), kLaws,
                              {0.0, 2.0, 5.0, 0.0, 2.0}, error));
    CHECK(state.moment(0)(0, 0, 0) == committed_first);
}
