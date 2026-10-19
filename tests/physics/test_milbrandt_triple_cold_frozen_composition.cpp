#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_diffusion.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

using namespace tmv::milbrandt_detail;
using State = ColdJointFrozenDiffusionTripleState;
using Result = ColdJointFrozenDiffusionTripleResult;
using Population = std::array<long double, 3>;
constexpr std::array<long double, 2> kMass{209.439510239L, 471.238898038L};
constexpr std::array<long double, 2> kFall{19.3L, 206.89L};
constexpr std::array<long double, 2> kFallExponent{.37L, .6384L};

long double shape_ratio(long double alpha)
{
    return (alpha + 4) * (alpha + 5) * (alpha + 6)
        / ((alpha + 1) * (alpha + 2) * (alpha + 3));
}

ColdFrozenDiffusionTripleState population(std::size_t species, long double alpha,
                                         double q = 5e-4f, double n = 10.0f)
{
    if (q == 0 && n == 0)
    {
        return {};
    }
    const float z = static_cast<float>(shape_ratio(alpha) * q * q
        / (kMass[species] * kMass[species] * n));
    return {q, n, z};
}

long double reference_shape(const ColdFrozenDiffusionTripleState& source,
                            std::size_t species)
{
    if (source.mass_per_kg == 0)
    {
        return 0;
    }
    const long double q = source.mass_per_kg;
    const long double ratio = kMass[species] * kMass[species]
        * source.number_per_kg * source.reflectivity_per_kg / (q * q);
    REQUIRE(ratio > 1);
    long double lower = -1.0L + 1e-9L;
    long double upper = 1e6L;
    for (int iteration = 0; iteration < 120; ++iteration)
    {
        const long double middle = (lower + upper) / 2;
        if (shape_ratio(middle) > ratio)
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

std::array<double, 16> cell(const State& initial, double density = .65,
                           double pressure = 90000, double saturation = .75,
                           double temperature = 253.15)
{
    std::array<double, 16> result{};
    result[0] = temperature;
    result[1] = pressure;
    result[2] = density;
    result[3] = saturation * thermodynamics::saturation_mixing_ratio_ice(temperature, pressure);
    result[8] = initial.graupel.mass_per_kg;
    result[9] = initial.hail.mass_per_kg;
    result[14] = initial.graupel.number_per_kg;
    result[15] = initial.hail.number_per_kg;
    return result;
}

long double numerical_gamma(long double order)
{
    constexpr int intervals = 32768;
    constexpr long double end = 160;
    constexpr long double width = end / intervals;
    long double sum = 0;
    for (int index = 1; index < intervals; ++index)
    {
        const long double x = index * width;
        sum += (index % 2 ? 4.0L : 2.0L) * std::pow(x, order) * std::exp(-x);
    }
    return (sum + std::pow(end, order) * std::exp(-end)) * width / 3;
}

long double reference_increment(const std::array<double, 16>& input,
                                const ColdFrozenDiffusionTripleState& source,
                                std::size_t species, long double duration)
{
    if (source.mass_per_kg == 0)
    {
        return 0;
    }
    const long double alpha = reference_shape(source, species);
    const long double q = source.mass_per_kg;
    const long double n = source.number_per_kg;
    const long double t = input[0];
    const long double tc = t - microphysics_constants::T0;
    const long double vapor_pressure = 611.15L
        * std::exp((23.036L - tc / 333.7L) * tc / (279.82L + tc));
    const long double qsi = .62198L * vapor_pressure / (input[1] - vapor_pressure);
    const long double density = input[2];
    const long double vapor = input[3];
    const long double diffusivity = (2.2157e-5L + .0155e-5L * tc) * 1e5L / input[1];
    const long double conductivity = 2.3971e-2L + .0078e-2L * tc;
    const long double viscosity = 1.72e-5L * 393.0L / (t + 120)
        * std::pow(t / microphysics_constants::T0, 1.5L) / density;
    const long double lambda = std::cbrt(kMass[species] * n
        * (alpha + 1) * (alpha + 2) * (alpha + 3) / q);
    const long double intercept = density * n * std::pow(lambda, alpha + 1)
        / std::tgamma(alpha + 1);
    const long double ventilation = .78L * numerical_gamma(alpha + 1)
        / std::pow(lambda, alpha + 2)
        + .30L * std::cbrt(viscosity / diffusivity)
        * std::sqrt(std::sqrt(1.225L / density) * kFall[species] / viscosity)
        * numerical_gamma(alpha + 1.5L + .5L * kFallExponent[species])
        / std::pow(lambda, alpha + 2.5L + .5L * kFallExponent[species]);
    const long double resistance = microphysics_constants::L_s * microphysics_constants::L_s
        / (conductivity * microphysics_constants::R_v * t * t)
        + 1 / (density * qsi * diffusivity);
    const long double raw = duration / density * 2 * std::acos(-1.0L)
        * (vapor / qsi - 1) * intercept * ventilation / resistance;
    if (vapor >= qsi)
    {
        const long double response = 1 + microphysics_constants::L_s * microphysics_constants::L_s
            * qsi / (microphysics_constants::cp * microphysics_constants::R_v
                     * (t - 7.66L) * (t - 7.66L));
        return std::clamp(raw, 0.0L, std::max((vapor - qsi) / response, 0.0L));
    }
    return std::clamp(raw, -q, 0.0L);
}

struct Reference
{
    std::array<long double, 2> dq;
    std::array<long double, 2> dn;
    std::array<Population, 2> diffusion;
    std::array<Population, 2> final;
    bool converted = false;
};

Reference reference(const std::array<double, 16>& input, const State& initial,
                     double duration)
{
    Reference result;
    const std::array<ColdFrozenDiffusionTripleState, 2> source{initial.graupel, initial.hail};
    for (std::size_t species = 0; species < 2; ++species)
    {
        result.dq[species] = reference_increment(input, source[species], species, duration);
    }
    for (int pass = 0; pass < 2; ++pass)
    {
        const long double available = input[3] + std::max(-result.dq[0], 0.0L)
            + std::max(-result.dq[1], 0.0L);
        const long double demand = std::max(result.dq[0], 0.0L)
            + std::max(result.dq[1], 0.0L);
        if (demand > available)
        {
            for (auto& increment : result.dq)
            {
                if (increment > 0)
                {
                    increment *= available / demand;
                }
            }
        }
    }
    for (std::size_t species = 0; species < 2; ++species)
    {
        const auto& s = source[species];
        const long double q = s.mass_per_kg;
        const long double n = s.number_per_kg;
        const long double z = s.reflectivity_per_kg;
        result.dn[species] = result.dq[species] < 0 ? n * result.dq[species] / q : 0;
        const long double dz = q == 0 ? 0
            : z * (2 * result.dq[species] / q - result.dn[species] / n);
        result.diffusion[species] = {q + result.dq[species], n + result.dn[species], z + dz};
    }
    result.final = result.diffusion;
    const auto& h = result.diffusion[1];
    if (h[0] > 0 && std::cbrt(h[0] / (kMass[1] * h[1])) < 1e-3L)
    {
        result.converted = true;
        result.final[0][0] += h[0];
        result.final[0][1] += h[1];
        result.final[0][2] += std::pow(kMass[1] / kMass[0], 2) * h[2];
        result.final[1] = {};
    }
    return result;
}

void check_reference(const std::array<double, 16>& input, const State& initial,
                     double duration, const std::array<float, 14>& rates,
                     const Result& result)
{
    const auto expected = reference(input, initial, duration);
    const std::array<ColdFrozenDiffusionTripleState, 2> source{initial.graupel, initial.hail};
    const std::array<ColdFrozenDiffusionTripleState, 2> diffusion{
        result.diffusion_endpoint.graupel, result.diffusion_endpoint.hail};
    const std::array<ColdFrozenDiffusionTripleEndpoint, 2> endpoint{result.graupel, result.hail};
    CHECK(result.terminal_conversion.transferred == expected.converted);
    for (std::size_t species = 0; species < 2; ++species)
    {
        const auto& event = result.diffusion_events[species];
        CHECK(event.species == (species == 0 ? ColdFrozenDiffusionSpecies::Graupel : ColdFrozenDiffusionSpecies::Hail));
        CHECK(event.entry_mass_per_kg == source[species].mass_per_kg);
        CHECK(event.entry_number_per_kg == source[species].number_per_kg);
        CHECK(event.dry_density_kg_m3 == input[2]);
        CHECK(event.duration_s == duration);
        CHECK(event.source_alpha == Approx(static_cast<double>(reference_shape(source[species], species))).epsilon(3e-6).margin(1e-10));
        CHECK(event.admitted_mass_increment_per_kg == Approx(static_cast<double>(expected.dq[species])).epsilon(3e-6).margin(1e-22));
        CHECK(event.admitted_number_increment_per_kg == Approx(static_cast<double>(expected.dn[species])).epsilon(3e-6).margin(1e-22));
        CHECK(diffusion[species].mass_per_kg == Approx(static_cast<double>(expected.diffusion[species][0])).epsilon(4e-7).margin(1e-22));
        CHECK(diffusion[species].number_per_kg == Approx(static_cast<double>(expected.diffusion[species][1])).epsilon(4e-7).margin(1e-22));
        CHECK(diffusion[species].reflectivity_per_kg == Approx(static_cast<double>(expected.diffusion[species][2])).epsilon(4e-7).margin(1e-30));
        CHECK(endpoint[species].mass_per_kg == Approx(static_cast<double>(expected.final[species][0])).epsilon(4e-7).margin(1e-22));
        CHECK(endpoint[species].number_per_kg == Approx(static_cast<double>(expected.final[species][1])).epsilon(4e-7).margin(1e-22));
        CHECK(endpoint[species].reflectivity_per_kg == Approx(static_cast<double>(expected.final[species][2])).epsilon(4e-7).margin(1e-30));
        const long double dq = expected.final[species][0] - source[species].mass_per_kg;
        const long double dn = expected.final[species][1] - source[species].number_per_kg;
        CHECK(rates[4 + species] == Approx(static_cast<double>(dq / duration)).epsilon(3e-6).margin(1e-18));
        CHECK(rates[12 + species] == Approx(static_cast<double>(dn / duration)).epsilon(3e-6).margin(1e-18));
    }
    const long double deposition = expected.dq[0] + expected.dq[1];
    CHECK(rates[6] == Approx(static_cast<double>(-deposition / duration)).epsilon(3e-6).margin(1e-18));
    CHECK(rates[7] == Approx(static_cast<double>(microphysics_constants::L_s
        / microphysics_constants::cp * deposition / duration)).epsilon(3e-6).margin(1e-15));
    for (std::size_t index = 0; index < rates.size(); ++index)
    {
        if (index != 4 && index != 5 && index != 6 && index != 7 && index != 12 && index != 13)
        {
            CHECK(rates[index] == 0);
        }
    }
    for (const long double scale : {1.0L, static_cast<long double>(input[2])})
    {
        const long double after_water = result.graupel.mass_per_kg
            + static_cast<long double>(result.hail.mass_per_kg)
            + input[3] + duration * static_cast<long double>(rates[6]);
        const long double before_water = initial.graupel.mass_per_kg
            + static_cast<long double>(initial.hail.mass_per_kg) + input[3];
        CHECK(static_cast<double>(scale * after_water) == Approx(static_cast<double>(scale * before_water)).epsilon(4e-7));
        const long double final_number = result.graupel.number_per_kg
            + static_cast<long double>(result.hail.number_per_kg);
        CHECK(static_cast<double>(scale * final_number) == Approx(static_cast<double>(scale
            * (expected.diffusion[0][1] + expected.diffusion[1][1]))).epsilon(4e-7).margin(1e-22));
        const long double scaled_z = kMass[0] * kMass[0] * result.graupel.reflectivity_per_kg
            + kMass[1] * kMass[1] * result.hail.reflectivity_per_kg;
        CHECK(static_cast<double>(scale * scaled_z) == Approx(static_cast<double>(scale
            * (kMass[0] * kMass[0] * expected.diffusion[0][2]
                + kMass[1] * kMass[1] * expected.diffusion[1][2]))).epsilon(4e-7).margin(1e-25));
    }
}

Result evaluate(const std::array<double, 16>& input, const State& initial,
                double duration, std::array<float, 14>& rates)
{
    Result result;
    std::string error;
    const bool accepted = evaluate_published_joint_triple_frozen_diffusion_cell(
        input, duration, initial, rates, result, error);
    INFO(error);
    REQUIRE(accepted);
    return result;
}

template <typename T>
std::array<unsigned char, sizeof(T)> object_bytes(const T& value)
{
    std::array<unsigned char, sizeof(T)> result;
    std::memcpy(result.data(), &value, sizeof(value));
    return result;
}

ColdParameters parameters()
{
    const auto& p = default_cold_parameters;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], true};
}

std::array<double*, 25> parameter_fields(ColdParameters& p)
{
    return {&p.a_g_, &p.a_h_, &p.a_i_, &p.a_r_, &p.a_s_,
        &p.alpha_g_, &p.alpha_h_, &p.alpha_i_, &p.alpha_r_, &p.alpha_s_,
        &p.b_g_, &p.b_h_, &p.b_i_, &p.b_r_, &p.b_s_,
        &p.c_g_, &p.c_h_, &p.c_i_, &p.c_r_, &p.c_s_,
        &p.d_g_, &p.d_h_, &p.d_i_, &p.d_r_, &p.d_s_};
}

std::array<double, 7> event_values(const ColdFrozenDiffusionAcceptedEvent& e)
{
    return {e.entry_mass_per_kg, e.entry_number_per_kg, e.source_alpha,
        e.admitted_mass_increment_per_kg, e.admitted_number_increment_per_kg,
        e.dry_density_kg_m3, e.duration_s};
}

std::array<double, 8> terminal_values(const ColdTerminalHailToGraupelAcceptedEvent& e)
{
    return {e.entry_mass_per_kg, e.entry_number_per_kg, e.entry_reflectivity_per_kg,
        e.source_alpha, e.source_mean_mass_diameter_m,
        e.graupel_reflectivity_increment_per_kg, e.dry_density_kg_m3, e.duration_s};
}

void check_identical(const Result& a, const Result& b)
{
    static_assert(sizeof(ColdFrozenDiffusionTripleState) == 3 * sizeof(double));
    static_assert(sizeof(ColdFrozenDiffusionTripleEndpoint) == 3 * sizeof(float));
    CHECK(object_bytes(a.diffusion_endpoint.graupel) == object_bytes(b.diffusion_endpoint.graupel));
    CHECK(object_bytes(a.diffusion_endpoint.hail) == object_bytes(b.diffusion_endpoint.hail));
    CHECK(object_bytes(a.unrounded.graupel) == object_bytes(b.unrounded.graupel));
    CHECK(object_bytes(a.unrounded.hail) == object_bytes(b.unrounded.hail));
    CHECK(object_bytes(a.graupel) == object_bytes(b.graupel));
    CHECK(object_bytes(a.hail) == object_bytes(b.hail));
    for (std::size_t species = 0; species < 2; ++species)
    {
        CHECK(a.diffusion_events[species].species == b.diffusion_events[species].species);
        CHECK(object_bytes(event_values(a.diffusion_events[species]))
            == object_bytes(event_values(b.diffusion_events[species])));
    }
    CHECK(a.terminal_conversion.transferred == b.terminal_conversion.transferred);
    CHECK(object_bytes(terminal_values(a.terminal_conversion))
        == object_bytes(terminal_values(b.terminal_conversion)));
}

} // namespace

TEST_CASE("joint frozen diffusion matches independent live shapes ventilation B1 B3 and shared water heat",
          "[physics][milbrandt][triple_cold_frozen_composition]")
{
    for (const auto shapes : {std::array<long double, 2>{0, 2}, {2, 20}, {20, 0}})
    {
        const State initial{population(0, shapes[0]), population(1, shapes[1])};
        for (const double density : {.65, 1.3})
        {
            for (const double pressure : {70000.0, 90000.0})
            {
                for (const double saturation : {.75, 1.0, 1.03})
                {
                    for (const double duration : {.05, 2.0})
                    {
                        const auto input = cell(initial, density, pressure, saturation);
                        std::array<float, 14> rates{};
                        const auto result = evaluate(input, initial, duration, rates);
                        check_reference(input, initial, duration, rates, result);
                    }
                }
            }
        }
    }
}

TEST_CASE("joint frozen donor admission shares vapor before terminal classification",
          "[physics][milbrandt][triple_cold_frozen_composition]")
{
    constexpr double duration = 1e7;
    const State initial{population(0, 2, .02f, 1e5f), population(1, 2, .02f, 1e5f)};
    const auto input = cell(initial, .65, 200000, 4, 270.15);
    const long double independent_demand = reference_increment(input, initial.graupel, 0, duration)
        + reference_increment(input, initial.hail, 1, duration);
    REQUIRE(independent_demand > input[3]);
    std::array<float, 14> rates{};
    const auto result = evaluate(input, initial, duration, rates);
    check_reference(input, initial, duration, rates, result);
    const long double admitted = result.diffusion_events[0].admitted_mass_increment_per_kg
        + static_cast<long double>(result.diffusion_events[1].admitted_mass_increment_per_kg);
    CHECK(static_cast<double>(admitted) == Approx(input[3]).epsilon(3e-6));
    CHECK(input[0] + duration * rates[7] > microphysics_constants::T0);
}

TEST_CASE("terminal small hail conversion preserves q N and scaled Z without added latent heat",
          "[physics][milbrandt][triple_cold_frozen_composition]")
{
    for (const bool existing_graupel : {false, true})
    {
        const double qh = static_cast<float>(kMass[1] * std::pow(.7e-3L, 3));
        const State initial{existing_graupel ? population(0, 20) : population(0, 0, 0, 0),
                            population(1, 2, qh, 1)};
        const auto input = cell(initial, .65, 90000, 1);
        std::array<float, 14> rates{};
        const auto result = evaluate(input, initial, .0625, rates);
        REQUIRE(result.terminal_conversion.transferred);
        check_reference(input, initial, .0625, rates, result);
        CHECK(result.terminal_conversion.entry_mass_per_kg == initial.hail.mass_per_kg);
        CHECK(result.terminal_conversion.entry_number_per_kg == initial.hail.number_per_kg);
        CHECK(result.terminal_conversion.entry_reflectivity_per_kg == initial.hail.reflectivity_per_kg);
        CHECK(result.terminal_conversion.graupel_reflectivity_increment_per_kg
            == Approx(static_cast<double>(std::pow(kMass[1] / kMass[0], 2)
                * initial.hail.reflectivity_per_kg)).epsilon(4e-7));
        CHECK(result.hail.mass_per_kg == 0);
        CHECK(result.hail.number_per_kg == 0);
        CHECK(result.hail.reflectivity_per_kg == 0);
        CHECK(rates[6] == 0);
        CHECK(rates[7] == 0);
        CHECK(rates[4] == -rates[5]);
        CHECK(rates[12] == -rates[13]);
    }
}

TEST_CASE("terminal hail classification retains exact one millimetre and separates its neighbors",
          "[physics][milbrandt][triple_cold_frozen_composition]")
{
    const double coefficient = 471.238898038;
    const std::array<double, 3> diameters{
        std::nextafter(1e-3, 0.0), 1e-3,
        std::nextafter(1e-3, std::numeric_limits<double>::infinity())};
    for (std::size_t neighbor = 0; neighbor < diameters.size(); ++neighbor)
    {
        const double q = coefficient * diameters[neighbor] * diameters[neighbor] * diameters[neighbor];
        const State initial{population(0, 0, 0, 0), population(1, 2, q, 1)};
        const auto input = cell(initial, .65, 90000, 1);
        std::array<float, 14> rates{};
        const auto result = evaluate(input, initial, .0625, rates);
        CHECK(result.terminal_conversion.source_mean_mass_diameter_m == diameters[neighbor]);
        CHECK(result.terminal_conversion.transferred == (neighbor == 0));
        CHECK(rates[6] == 0);
        CHECK(rates[7] == 0);
    }
}

TEST_CASE("hail below the entry boundary grows above it before terminal classification",
          "[physics][milbrandt][triple_cold_frozen_composition]")
{
    constexpr double duration = 8192;
    const double qh = static_cast<float>(kMass[1] * std::pow(.97e-3L, 3));
    const State initial{population(0, 2), population(1, 2, qh, 1)};
    const auto input = cell(initial, .65, 90000, 1.07);
    REQUIRE(std::cbrt(initial.hail.mass_per_kg / (471.238898038 * initial.hail.number_per_kg)) < 1e-3);
    std::array<float, 14> rates{};
    const auto result = evaluate(input, initial, duration, rates);
    check_reference(input, initial, duration, rates, result);
    CHECK(result.terminal_conversion.source_mean_mass_diameter_m > 1e-3);
    CHECK_FALSE(result.terminal_conversion.transferred);
    CHECK(result.hail.mass_per_kg > initial.hail.mass_per_kg);
    CHECK(result.hail.number_per_kg == initial.hail.number_per_kg);
    CHECK(result.terminal_conversion.entry_mass_per_kg == result.diffusion_endpoint.hail.mass_per_kg);
    CHECK(result.terminal_conversion.entry_number_per_kg == result.diffusion_endpoint.hail.number_per_kg);
    CHECK(result.terminal_conversion.entry_reflectivity_per_kg == result.diffusion_endpoint.hail.reflectivity_per_kg);
}

TEST_CASE("joint frozen shapes continue through cold steps and extinguish without a moment reset",
          "[physics][milbrandt][triple_cold_frozen_composition]")
{
    State live{population(0, 2, 1e-4f), population(1, 20, 1e-4f)};
    auto input = cell(live, .65, 90000, 1.07);
    const long double old_alpha = reference_shape(live.graupel, 0);
    for (int step = 0; step < 3; ++step)
    {
        std::array<float, 14> rates{};
        const auto result = evaluate(input, live, 2048, rates);
        check_reference(input, live, 2048, rates, result);
        live = {{result.graupel.mass_per_kg, result.graupel.number_per_kg, result.graupel.reflectivity_per_kg},
                {result.hail.mass_per_kg, result.hail.number_per_kg, result.hail.reflectivity_per_kg}};
        input[0] += 2048 * static_cast<double>(rates[7]);
        input[3] += 2048 * static_cast<double>(rates[6]);
        REQUIRE(input[0] < microphysics_constants::T0);
        CHECK(reference_shape(live.graupel, 0) > old_alpha + 1e-6L);
        input[8] = live.graupel.mass_per_kg;
        input[9] = live.hail.mass_per_kg;
        input[14] = live.graupel.number_per_kg;
        input[15] = live.hail.number_per_kg;
    }
    const State initial{population(0, 2, 1e-7f, .01f), population(1, 20, 1e-7f, .01f)};
    const auto dry_input = cell(initial, .65, 90000, 0);
    std::array<float, 14> rates{};
    const auto drained = evaluate(dry_input, initial, 1048576, rates);
    check_reference(dry_input, initial, 1048576, rates, drained);
    for (const auto& endpoint : {drained.graupel, drained.hail})
    {
        CHECK(endpoint.mass_per_kg == 0);
        CHECK(endpoint.number_per_kg == 0);
        CHECK(endpoint.reflectivity_per_kg == 0);
    }
    const State empty{};
    auto continuation = cell(empty);
    continuation[0] = dry_input[0] + 1048576 * static_cast<double>(rates[7]);
    continuation[3] = 1048576 * static_cast<double>(rates[6]);
    rates = {};
    const auto after = evaluate(continuation, empty, .0625, rates);
    check_reference(continuation, empty, .0625, rates, after);
    for (const float rate : rates)
    {
        CHECK(rate == 0);
    }
}

TEST_CASE("joint frozen pure evaluation preserves exact science bytes on one and four workers",
          "[physics][milbrandt][triple_cold_frozen_composition]")
{
    struct Outcome
    {
        std::array<float, 14> rates{};
        Result result;
        std::string error;
        bool accepted = false;
    };
    std::array<std::array<Outcome, 32>, 2> outcomes;
    for (int run = 0; run < 2; ++run)
    {
        const int workers = run == 0 ? 1 : 4;
#ifdef _OPENMP
        #pragma omp parallel for num_threads(workers)
#endif
        for (int index = 0; index < 32; ++index)
        {
            const State initial{population(0, index % 3 == 0 ? 0 : 2),
                                population(1, index % 3 == 0 ? 20 : 2)};
            const auto input = cell(initial, index % 2 ? .65 : 1.3,
                index % 4 < 2 ? 90000 : 70000, index % 3 == 0 ? 1.03 : .75);
            auto& out = outcomes[run][index];
            out.accepted = evaluate_published_joint_triple_frozen_diffusion_cell(
                input, 2, initial, out.rates, out.result, out.error);
        }
        (void)workers;
    }
    for (std::size_t index = 0; index < outcomes[0].size(); ++index)
    {
        INFO(outcomes[0][index].error << "; " << outcomes[1][index].error);
        REQUIRE(outcomes[0][index].accepted);
        REQUIRE(outcomes[1][index].accepted);
        CHECK(object_bytes(outcomes[0][index].rates) == object_bytes(outcomes[1][index].rates));
        check_identical(outcomes[0][index].result, outcomes[1][index].result);
    }
}

TEST_CASE("joint frozen producer inventories and individually guards all sixty four material channels",
          "[physics][milbrandt][triple_cold_frozen_composition]")
{
    const std::array<const char*, 64> expected{
        "vapor_to_ice", "cloud_to_ice", "rain_to_frozen", "ice_to_snow", "nc_to_ice", "nr_to_frozen",
        "new_ice_number", "ice_number_sublimated", "snow_number_sublimated", "graupel_number_sublimated",
        "hail_number_sublimated", "ice_number_to_snow", "cloud_to_snow", "cloud_to_graupel", "cloud_to_hail",
        "cloud_number_to_snow", "cloud_number_to_graupel", "cloud_number_to_hail", "ice_collected_by_snow",
        "ice_collected_by_graupel", "ice_collected_by_hail", "ice_number_collected_by_snow",
        "ice_number_collected_by_graupel", "ice_number_collected_by_hail", "snow_self_collection",
        "splinter_mass_from_snow", "splinter_mass_from_graupel", "splinter_number_from_snow",
        "splinter_number_from_graupel", "rain_collected_by_ice", "ice_collected_by_rain",
        "rain_number_collected_by_ice", "ice_number_collected_by_rain", "rain_collected_by_snow",
        "snow_collected_by_rain", "rain_number_collected_by_snow", "snow_number_collected_by_rain",
        "rain_collected_by_graupel", "graupel_collected_by_rain", "rain_number_collected_by_graupel",
        "graupel_number_collected_by_rain", "collision_mass_to_snow", "collision_mass_to_graupel",
        "collision_mass_to_hail", "collision_number_to_snow", "collision_number_to_graupel",
        "collision_number_to_hail", "snow_to_graupel", "snow_number_to_graupel", "graupel_to_hail",
        "graupel_number_to_hail", "hail_number_from_graupel", "hail_to_graupel", "hail_number_to_graupel",
        "rain_collected_by_hail", "rain_number_collected_by_hail", "snow_collected_by_hail",
        "snow_number_collected_by_hail", "shed_rain_from_hail", "shed_rain_number", "ice_deposition",
        "snow_deposition", "graupel_deposition", "hail_deposition"};
    for (std::size_t channel = 0; channel < expected.size(); ++channel)
    {
        const bool owned = channel == 9 || channel == 10 || channel == 62 || channel == 63;
        CHECK(std::string(published_joint_frozen_diffusion_material_channels[channel].name) == expected[channel]);
        CHECK(published_joint_frozen_diffusion_material_channels[channel].owned == owned);
        for (const double value : {-1e-7, 1e-7, std::numeric_limits<double>::quiet_NaN()})
        {
            std::array<double, 64> values{};
            values[channel] = value;
            const auto before = object_bytes(values);
            std::string error;
            CHECK(validate_published_joint_frozen_diffusion_channels(values, error) == (owned && std::isfinite(value)));
            CHECK(object_bytes(values) == before);
            if (!owned || !std::isfinite(value))
            {
                CHECK(error.find(expected[channel]) != std::string::npos);
            }
        }
    }
    std::array<double, 64> canceled{};
    canceled[0] = 1e-7;
    canceled[1] = -1e-7;
    std::string error;
    CHECK_FALSE(validate_published_joint_frozen_diffusion_channels(canceled, error));
    CHECK(error.find("vapor_to_ice") != std::string::npos);
}

TEST_CASE("joint frozen producer rejects invalid zero events buffers and late channels before publication",
          "[physics][milbrandt][triple_cold_frozen_composition]")
{
    const State empty{};
    for (int mutation = 0; mutation < 57; ++mutation)
    {
        auto input = cell(empty, .65, 90000, 1);
        auto p = parameters();
        double duration = .0625;
        std::array<float, 14> rates{};
        ColdJointFrozenDiffusionAcceptedEventBuffer buffer;
        if (mutation < 25)
        {
            if (mutation == 5 || mutation == 6)
            {
                *parameter_fields(p)[mutation] = -1;
            }
            else
            {
                auto* value = parameter_fields(p)[mutation];
                *value = std::nextafter(*value, std::numeric_limits<double>::infinity());
            }
        }
        else if (mutation >= 38 && mutation < 51)
        {
            input[mutation - 35] = std::numeric_limits<double>::quiet_NaN();
        }
        else
        {
            switch (mutation)
            {
                case 25: p.hail_processes = false; break;
                case 26: input[0] = 149; break;
                case 27: input[0] = microphysics_constants::T0; break;
                case 28: input[1] = 0; break;
                case 29: input[1] = 200001; break;
                case 30: input[2] = 0; break;
                case 31: input[3] = -1; break;
                case 32: duration = 0; break;
                case 33: duration = std::numeric_limits<double>::infinity(); break;
                case 34: buffer.capacity = 1; break;
                case 35: buffer.capacity = 3; break;
                case 36: buffer.size = 1; break;
                case 37: rates[12] = 1; break;
                case 51: input[3] *= 2; break;
                case 52: input[4] = 1e-4; input[10] = 1e7; input[0] = 233.15; break;
                case 53: input[5] = 1e-4; input[11] = 1e3; break;
                case 54: input[6] = 1e-4; input[12] = 1e3; break;
                case 55: input[7] = 1e-4; input[13] = 1e3; break;
                case 56: input[8] = 1e-4; input[14] = 0; break;
            }
        }
        const auto input_before = object_bytes(input);
        const auto rates_before = object_bytes(rates);
        const auto buffer_before = object_bytes(buffer);
        INFO("producer mutation " << mutation);
        CHECK_THROWS(published_joint_frozen_diffusion_cold_cell(p, input, duration, rates, buffer));
        CHECK(object_bytes(input) == input_before);
        CHECK(object_bytes(rates) == rates_before);
        CHECK(object_bytes(buffer) == buffer_before);
    }
}

TEST_CASE("joint frozen provenance and mixed gamma or FP32 failures leave all endpoints unpublished",
          "[physics][milbrandt][triple_cold_frozen_composition]")
{
    const State initial{population(0, 2), population(1, 20)};
    const auto input = cell(initial);
    std::array<float, 14> rates{};
    const auto accepted = evaluate(input, initial, .0625, rates);
    for (int mutation = 0; mutation < 15; ++mutation)
    {
        auto events = accepted.diffusion_events;
        auto source = initial;
        switch (mutation)
        {
            case 0: events[0].entry_mass_per_kg = std::nextafter(events[0].entry_mass_per_kg, 1.0); break;
            case 1: events[1].entry_number_per_kg = std::nextafter(events[1].entry_number_per_kg, 100.0); break;
            case 2: events[1].source_alpha = std::nextafter(events[1].source_alpha, 100.0); break;
            case 3: events[0].dry_density_kg_m3 = 1.3; break;
            case 4: events[1].duration_s = .125; break;
            case 5: events[1].admitted_number_increment_per_kg = 0; break;
            case 6: events[0].admitted_mass_increment_per_kg = -2 * initial.graupel.mass_per_kg; break;
            case 7: events[1].admitted_number_increment_per_kg = 1; break;
            case 8: std::swap(events[0], events[1]); break;
            case 9: events[0].species = static_cast<ColdFrozenDiffusionSpecies>(42); break;
            case 10: source.graupel.reflectivity_per_kg = 0; break;
            case 11: source.hail.number_per_kg = std::numeric_limits<double>::quiet_NaN(); break;
            case 12: source.hail.reflectivity_per_kg = std::numeric_limits<float>::denorm_min(); break;
            case 13: events[1].admitted_mass_increment_per_kg = std::numeric_limits<double>::infinity(); break;
            case 14: events[0].admitted_mass_increment_per_kg = 1e9; events[0].admitted_number_increment_per_kg = 0; break;
        }
        Result result;
        result.graupel = {.125f, 23, .001f};
        const auto before = object_bytes(result);
        const auto source_before = object_bytes(source);
        const auto events_before = object_bytes(events);
        std::string error;
        CHECK_FALSE(apply_published_joint_frozen_diffusion(source, .65, .0625, events, result, error));
        CHECK_FALSE(error.empty());
        CHECK(object_bytes(result) == before);
        CHECK(object_bytes(source) == source_before);
        CHECK(object_bytes(events) == events_before);
    }
    const double maximum = std::numeric_limits<float>::max();
    const std::array<State, 4> rejected{
        State{population(0, 2, 1e-3f, 1e-8f), population(1, 2, 1e-7f, 1e7f)},
        State{population(0, 2, maximum, maximum), population(1, 2, 1e32f, maximum)},
        initial,
        State{population(0, 0, 0, 0), population(1, 2, static_cast<float>(kMass[1] * std::pow(.7e-3L, 3)), 1)}};
    for (std::size_t failure = 0; failure < rejected.size(); ++failure)
    {
        const double duration = failure == 2 ? 1e300 : failure == 3 ? 1e-300 : .0625;
        auto source_input = cell(rejected[failure], .65, 90000, failure == 2 ? .75 : 1);
        std::array<float, 14> unpublished;
        unpublished.fill(42);
        Result result;
        result.hail = {.125f, 23, .001f};
        const auto rates_before = object_bytes(unpublished);
        const auto result_before = object_bytes(result);
        const auto source_before = object_bytes(rejected[failure]);
        std::string error;
        INFO("mixed or rate failure " << failure);
        CHECK_FALSE(evaluate_published_joint_triple_frozen_diffusion_cell(
            source_input, duration, rejected[failure], unpublished, result, error));
        CHECK_FALSE(error.empty());
        CHECK(object_bytes(unpublished) == rates_before);
        CHECK(object_bytes(result) == result_before);
        CHECK(object_bytes(rejected[failure]) == source_before);
    }
}
