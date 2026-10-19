#include "catch2/catch.hpp"
#include "microphysics/milbrandt_melting_parameters.hpp"
#include "microphysics/schemes/milbrandt/melting.hpp"
#include "microphysics/schemes/milbrandt/triple_melting.hpp"

#include <algorithm>
#include <array>
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
using State = MeltingTripleState;
using Result = MeltingTripleResult;
using Population = std::array<long double, 3>;
constexpr std::array<long double, 3> kMass{523.598775598L, 209.439510239L, 471.238898038L};
constexpr std::array<long double, 2> kFall{19.3L, 206.89L};
constexpr std::array<long double, 2> kFallExponent{.37L, .6384L};

long double shape_ratio(long double alpha)
{
    return (alpha + 4) * (alpha + 5) * (alpha + 6)
        / ((alpha + 1) * (alpha + 2) * (alpha + 3));
}

MeltingTripleMomentState population(std::size_t species, long double alpha,
                                    double q = 5e-4f, double n = 10.0f)
{
    if (q == 0 && n == 0)
    {
        return {};
    }
    return {q, n, static_cast<float>(shape_ratio(alpha) * q * q
        / (kMass[species] * kMass[species] * n))};
}

long double reference_shape(const MeltingTripleMomentState& source, std::size_t species)
{
    if (source.mass_per_kg == 0)
    {
        return 0;
    }
    const long double q = source.mass_per_kg;
    const long double ratio = kMass[species] * kMass[species] * source.number_per_kg
        * source.reflectivity_per_kg / (q * q);
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

std::array<double, 12> cell(const State& initial, double temperature = 283.15,
                           double density = .65, double pressure = 90000,
                           double freezing_saturation = 1)
{
    std::array<double, 12> result{};
    result[0] = temperature;
    result[1] = pressure;
    result[2] = density;
    result[3] = freezing_saturation * thermodynamics::saturation_mixing_ratio_water(
        microphysics_constants::T0, pressure);
    result[6] = initial.graupel.mass_per_kg;
    result[7] = initial.hail.mass_per_kg;
    result[10] = initial.graupel.number_per_kg;
    result[11] = initial.hail.number_per_kg;
    return result;
}

long double reference_melt(const std::array<double, 12>& input,
                           const MeltingTripleMomentState& source,
                           std::size_t donor, long double duration)
{
    const long double t = input[0];
    const long double tc = t - microphysics_constants::T0;
    if (source.mass_per_kg == 0 || tc <= 0 || (donor == 1 && tc <= 5))
    {
        return 0;
    }
    const long double q = source.mass_per_kg;
    const long double n = source.number_per_kg;
    const long double density = input[2];
    const long double alpha = reference_shape(source, donor + 1);
    const long double diffusivity = (2.2157e-5L + .0155e-5L * tc) * 1e5L / input[1];
    const long double conductivity = 2.3971e-2L + .0078e-2L * tc;
    const long double viscosity = 1.72e-5L * 393.0L / (t + 120)
        * std::pow(t / microphysics_constants::T0, 1.5L) / density;
    const long double lambda = std::cbrt(kMass[donor + 1] * n
        * (alpha + 1) * (alpha + 2) * (alpha + 3) / q);
    const long double intercept = density * n * std::pow(lambda, alpha + 1)
        / std::tgamma(alpha + 1);
    const long double ventilation = .78L * numerical_gamma(alpha + 1)
        / std::pow(lambda, alpha + 2)
        + .30L * std::cbrt(viscosity / diffusivity)
        * std::sqrt(std::sqrt(1.225L / density) * kFall[donor] / viscosity)
        * numerical_gamma(alpha + 1.5L + .5L * kFallExponent[donor])
        / std::pow(lambda, alpha + 2.5L + .5L * kFallExponent[donor]);
    const long double q0 = .62198L * 611.21L / (input[1] - 611.21L);
    const long double heat = conductivity * tc - density * microphysics_constants::L_v
        * diffusivity * (q0 - input[3]);
    const long double raw = duration * 2 * std::acos(-1.0L)
        / (density * microphysics_constants::L_f) * intercept * ventilation * heat;
    return std::clamp(raw, 0.0L, q);
}

struct Reference
{
    std::array<long double, 2> melt;
    std::array<long double, 2> number;
    std::array<Population, 3> final;
};

Reference reference(const std::array<double, 12>& input, const State& initial, double duration)
{
    Reference result;
    const std::array<MeltingTripleMomentState, 3> source{initial.rain, initial.graupel, initial.hail};
    for (std::size_t species = 0; species < source.size(); ++species)
    {
        result.final[species] = {source[species].mass_per_kg, source[species].number_per_kg,
                                source[species].reflectivity_per_kg};
    }
    for (std::size_t donor = 0; donor < 2; ++donor)
    {
        const auto& s = source[1 + donor];
        result.melt[donor] = reference_melt(input, s, donor, duration);
        const long double fraction = s.mass_per_kg > 0 ? result.melt[donor] / s.mass_per_kg : 0;
        result.number[donor] = s.number_per_kg * fraction;
        const long double reflectivity = s.reflectivity_per_kg * fraction;
        result.final[1 + donor][0] -= result.melt[donor];
        result.final[1 + donor][1] -= result.number[donor];
        result.final[1 + donor][2] -= reflectivity;
        result.final[0][0] += result.melt[donor];
        result.final[0][1] += result.number[donor];
        result.final[0][2] += std::pow(kMass[1 + donor] / kMass[0], 2) * reflectivity;
    }
    return result;
}

Result evaluate(const std::array<double, 12>& input, const State& initial,
                double duration, std::array<float, 11>& rates)
{
    Result result;
    std::string error;
    const bool accepted = evaluate_published_triple_melting_cell(
        input, duration, {}, initial, rates, result, error);
    INFO(error);
    REQUIRE(accepted);
    return result;
}

std::array<double, 42> science_members(const Result& result)
{
    std::array<double, 42> values{};
    std::size_t index = 0;
    for (const auto& state : {result.unrounded.rain, result.unrounded.graupel, result.unrounded.hail})
    {
        for (double value : {state.mass_per_kg, state.number_per_kg, state.reflectivity_per_kg})
        {
            values[index++] = value;
        }
    }
    for (const auto& state : {result.rain, result.graupel, result.hail})
    {
        for (float value : {state.mass_per_kg, state.number_per_kg, state.reflectivity_per_kg})
        {
            values[index++] = value;
        }
    }
    for (const auto& event : result.events)
    {
        for (double value : {static_cast<double>(event.species), event.entry_mass_per_kg,
             event.entry_number_per_kg, event.source_alpha, event.admitted_mass_transfer_per_kg,
             event.admitted_number_transfer_per_kg, event.admitted_fraction, event.temperature_k,
             event.pressure_pa, event.dry_density_kg_m3, event.vapor_per_kg, event.duration_s})
        {
            values[index++] = value;
        }
    }
    return values;
}

State stored_state(const Result& result)
{
    return {{result.rain.mass_per_kg, result.rain.number_per_kg, result.rain.reflectivity_per_kg},
            {result.graupel.mass_per_kg, result.graupel.number_per_kg, result.graupel.reflectivity_per_kg},
            {result.hail.mass_per_kg, result.hail.number_per_kg, result.hail.reflectivity_per_kg}};
}

void check_reference(const std::array<double, 12>& input, const State& initial,
                     double duration, const std::array<float, 11>& rates, const Result& result)
{
    const auto expected = reference(input, initial, duration);
    const std::array<MeltingTripleMomentState, 3> source{initial.rain, initial.graupel, initial.hail};
    const std::array<MeltingTripleMomentState, 3> exact{
        result.unrounded.rain, result.unrounded.graupel, result.unrounded.hail};
    const std::array<MeltingTripleMomentEndpoint, 3> rounded{result.rain, result.graupel, result.hail};
    for (std::size_t species = 0; species < 3; ++species)
    {
        CHECK(exact[species].mass_per_kg == Approx(static_cast<double>(expected.final[species][0])).epsilon(4e-7).margin(1e-22));
        CHECK(exact[species].number_per_kg == Approx(static_cast<double>(expected.final[species][1])).epsilon(4e-7).margin(1e-22));
        CHECK(exact[species].reflectivity_per_kg == Approx(static_cast<double>(expected.final[species][2])).epsilon(4e-7).margin(1e-30));
        CHECK(rounded[species].mass_per_kg == Approx(static_cast<double>(expected.final[species][0])).epsilon(4e-7).margin(1e-22));
        CHECK(rounded[species].number_per_kg == Approx(static_cast<double>(expected.final[species][1])).epsilon(4e-7).margin(1e-22));
        CHECK(rounded[species].reflectivity_per_kg == Approx(static_cast<double>(expected.final[species][2])).epsilon(4e-7).margin(1e-30));
    }
    for (std::size_t donor = 0; donor < 2; ++donor)
    {
        const auto& event = result.events[donor];
        CHECK(event.species == (donor == 0 ? MeltingFrozenSpecies::Graupel : MeltingFrozenSpecies::Hail));
        CHECK(event.entry_mass_per_kg == source[donor + 1].mass_per_kg);
        CHECK(event.entry_number_per_kg == source[donor + 1].number_per_kg);
        CHECK(event.source_alpha == Approx(static_cast<double>(reference_shape(source[donor + 1], donor + 1))).epsilon(3e-6).margin(1e-10));
        CHECK(event.admitted_mass_transfer_per_kg == Approx(static_cast<double>(expected.melt[donor])).epsilon(3e-6).margin(1e-22));
        CHECK(event.admitted_number_transfer_per_kg == Approx(static_cast<double>(expected.number[donor])).epsilon(3e-6).margin(1e-22));
        CHECK(event.admitted_fraction == Approx(static_cast<double>(source[donor + 1].mass_per_kg > 0
            ? expected.melt[donor] / source[donor + 1].mass_per_kg : 0)).epsilon(3e-6).margin(1e-18));
        CHECK(event.temperature_k == input[0]);
        CHECK(event.pressure_pa == input[1]);
        CHECK(event.dry_density_kg_m3 == input[2]);
        CHECK(event.vapor_per_kg == input[3]);
        CHECK(event.duration_s == duration);
        CHECK(rates[2 + donor] == Approx(static_cast<double>(-expected.melt[donor] / duration)).epsilon(3e-6).margin(1e-18));
        CHECK(rates[9 + donor] == Approx(static_cast<double>(-expected.number[donor] / duration)).epsilon(3e-6).margin(1e-18));
    }
    const long double melted = expected.melt[0] + expected.melt[1];
    CHECK(rates[4] == Approx(static_cast<double>(melted / duration)).epsilon(3e-6).margin(1e-18));
    CHECK(rates[5] == Approx(static_cast<double>(-microphysics_constants::L_f
        / microphysics_constants::cp * melted / duration)).epsilon(3e-6).margin(1e-15));
    CHECK(rates[6] == Approx(static_cast<double>((expected.number[0] + expected.number[1]) / duration)).epsilon(3e-6).margin(1e-18));
    for (std::size_t index : {0, 1, 7, 8})
    {
        CHECK(rates[index] == 0);
    }
    for (const long double scale : {1.0L, static_cast<long double>(input[2])})
    {
        for (std::size_t moment = 0; moment < 3; ++moment)
        {
            long double before = 0;
            long double after = 0;
            for (std::size_t species = 0; species < 3; ++species)
            {
                const long double factor = moment == 2 ? kMass[species] * kMass[species] : 1;
                const Population old{source[species].mass_per_kg, source[species].number_per_kg,
                                     source[species].reflectivity_per_kg};
                const Population stored{rounded[species].mass_per_kg, rounded[species].number_per_kg,
                                        rounded[species].reflectivity_per_kg};
                before += factor * old[moment];
                after += factor * stored[moment];
            }
            CHECK(static_cast<double>(scale * after) == Approx(static_cast<double>(scale * before)).epsilon(4e-7).margin(1e-25));
        }
    }
}

} // namespace

TEST_CASE("zero collection triple melting matches independent density weighted ventilation B1 B3 B5 and heat",
          "[physics][milbrandt][triple_melting]")
{
    for (const auto shapes : {std::array<long double, 2>{0, 2}, {2, 20}, {20, 0}})
    {
        for (const bool receiver : {false, true})
        {
            const State initial{receiver ? population(0, 2) : population(0, 0, 0, 0),
                                population(1, shapes[0]), population(2, shapes[1])};
            for (const double density : {.65, 1.3})
            {
                for (const double pressure : {70000.0, 90000.0})
                {
                    for (const double humidity : {0.0, 1.0, 1.1})
                    {
                        for (const double duration : {.05, 2.0})
                        {
                            const auto input = cell(initial, 283.15, density, pressure, humidity);
                            const double vapor_before = input[3];
                            std::array<float, 11> rates{};
                            const auto result = evaluate(input, initial, duration, rates);
                            check_reference(input, initial, duration, rates, result);
                            CHECK(input[3] == vapor_before);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("triple melting retains strict zero and five Celsius activation boundaries",
          "[physics][milbrandt][triple_melting]")
{
    const State initial{population(0, 0, 0, 0), population(1, 2), population(2, 20)};
    for (const double boundary : {microphysics_constants::T0, microphysics_constants::T0 + 5})
    {
        for (const double temperature : {std::nextafter(boundary, 0.0), boundary,
                                        std::nextafter(boundary, std::numeric_limits<double>::infinity())})
        {
            const auto input = cell(initial, temperature, .65, 90000, 1.1);
            std::array<float, 11> rates{};
            const auto result = evaluate(input, initial, .0625, rates);
            check_reference(input, initial, .0625, rates, result);
            CHECK((result.events[0].admitted_mass_transfer_per_kg > 0) == (temperature > microphysics_constants::T0));
            CHECK((result.events[1].admitted_mass_transfer_per_kg > 0) == (temperature - microphysics_constants::T0 > 5));
        }
    }
}

TEST_CASE("published melting density correction preserves legacy mass heat at unit density and reverses humidity contrast",
          "[physics][milbrandt][triple_melting]")
{
    const State initial{population(0, 0, 0, 0), population(1, 2), population(2, 20)};
    for (const double humidity : {.5, 1.0, 1.1})
    {
        for (const double density : {.65, 1.0, 1.3})
        {
            const auto input = cell(initial, 283.15, density, 90000, humidity);
            std::array<float, 11> rates{};
            const auto result = evaluate(input, initial, .05, rates);
            check_reference(input, initial, .05, rates, result);
            auto p = default_melting_parameters;
            p[3] = result.events[0].source_alpha;
            p[8] = result.events[1].source_alpha;
            std::array<float, 11> legacy{};
            melting_cell(p, input, .05, legacy);
            if (density == 1 || humidity == 1)
            {
                for (std::size_t rate : {2, 3, 4, 5})
                {
                    CHECK(rates[rate] == legacy[rate]);
                }
            }
            else
            {
                const bool new_greater = humidity < 1 ? density < 1 : density > 1;
                CHECK((rates[4] > legacy[4]) == new_greater);
                CHECK(rates[4] != legacy[4]);
            }
        }
    }
}

TEST_CASE("whole dual donor melting extinguishes q N Z algebraically into empty or existing rain",
          "[physics][milbrandt][triple_melting]")
{
    for (const bool receiver : {false, true})
    {
        const State initial{receiver ? population(0, 2) : population(0, 0, 0, 0),
                            population(1, 2, 1e-7f, .01f), population(2, 20, 1e-7f, .01f)};
        const auto input = cell(initial);
        std::array<float, 11> rates{};
        const auto result = evaluate(input, initial, 1048576, rates);
        check_reference(input, initial, 1048576, rates, result);
        for (std::size_t donor = 0; donor < 2; ++donor)
        {
            CHECK(result.events[donor].admitted_fraction == 1);
        }
        for (const auto& endpoint : {result.graupel, result.hail})
        {
            CHECK(endpoint.mass_per_kg == 0);
            CHECK(endpoint.number_per_kg == 0);
            CHECK(endpoint.reflectivity_per_kg == 0);
        }
    }
}

TEST_CASE("triple melting rejects unowned channels and producer mutations before publishing rates or events",
          "[physics][milbrandt][triple_melting]")
{
    const State initial{population(0, 2), population(1, 2), population(2, 20)};
    std::array<float, 11> accepted_rates{};
    const Result accepted = evaluate(cell(initial), initial, .1, accepted_rates);
    auto parameters = default_melting_parameters;
    parameters[3] = accepted.events[0].source_alpha;
    parameters[8] = accepted.events[1].source_alpha;
    const auto reject = [&](const auto& input, double duration, const auto& p,
                            const MeltingNoCollectionProfile& profile,
                            std::array<float, 11> rates, MeltingAcceptedEventBuffer buffer)
    {
        const auto rates_before = rates;
        const auto buffer_before = buffer;
        bool rejected = false;
        try
        {
            published_triple_melting_cell(p, input, duration, profile, rates, buffer);
        }
        catch (const std::runtime_error&)
        {
            rejected = true;
        }
        REQUIRE(rejected);
        CHECK(std::memcmp(rates.data(), rates_before.data(), sizeof(rates)) == 0);
        CHECK(buffer.size == buffer_before.size);
        CHECK(buffer.capacity == buffer_before.capacity);
        Result a, b;
        a.events = buffer.events;
        b.events = buffer_before.events;
        CHECK(science_members(a) == science_members(b));
    };
    MeltingAcceptedEventBuffer buffer;
    buffer.events = accepted.events;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (double temperature : {250.0, 283.15})
    {
        const auto input = cell(initial, temperature);
        for (std::size_t index = 0; index < 8; ++index)
        {
            for (double value : {-1.0, 1.0, nan, std::numeric_limits<double>::infinity()})
            {
                MeltingNoCollectionProfile profile;
                profile.unowned_collection_increments[index] = value;
                reject(input, .1, parameters, profile, {}, buffer);
            }
        }
        MeltingNoCollectionProfile cancellation;
        cancellation.unowned_collection_increments = {1, -1, 1, -1, 1, -1, 1, -1};
        reject(input, .1, parameters, cancellation, {}, buffer);
        for (std::size_t index : {4, 5, 8, 9})
        {
            auto unowned = input;
            unowned[index] = 1;
            reject(unowned, .1, parameters, {}, {}, buffer);
        }
        for (std::size_t index = 0; index < 11; ++index)
        {
            for (float value : {-1.0f, 1.0f, std::numeric_limits<float>::quiet_NaN()})
            {
                std::array<float, 11> incoming{};
                incoming[index] = value;
                reject(input, .1, parameters, {}, incoming, buffer);
            }
        }
        for (const auto extent : {std::array<std::size_t, 2>{0, 0}, {0, 1}, {0, 3}, {1, 2}, {3, 2}})
        {
            auto invalid = buffer;
            invalid.size = extent[0];
            invalid.capacity = extent[1];
            reject(input, .1, parameters, {}, {}, invalid);
        }
        for (std::size_t index = 0; index < parameters.size(); ++index)
        {
            auto invalid = parameters;
            invalid[index] = (index == 3 || index == 8) ? -1.0
                : std::nextafter(invalid[index], std::numeric_limits<double>::infinity());
            reject(input, .1, invalid, {}, {}, buffer);
        }
    }
    std::string error;
    for (std::size_t index = 0; index < 4; ++index)
    {
        for (double value : {-1.0, 1.0, nan, std::numeric_limits<double>::infinity()})
        {
            std::array<double, 4> channels{};
            channels[index] = value;
            CHECK_FALSE(validate_melting_unowned_material_channels(channels, error));
            CHECK(error.find(melting_unowned_material_channel_names[index]) != std::string::npos);
        }
    }
    CHECK_FALSE(validate_melting_unowned_material_channels({1, -1, 1, -1}, error));
    reject(cell(initial), std::numeric_limits<double>::max(), parameters, {}, {}, buffer);
}

TEST_CASE("triple melting rejects invalid zero event context and source populations atomically",
          "[physics][milbrandt][triple_melting]")
{
    const State initial{population(0, 2), population(1, 2), population(2, 20)};
    std::array<float, 11> rates{};
    const Result accepted = evaluate(cell(initial), initial, .1, rates);
    const auto reject = [&](const auto& input, const State& source, double duration,
                            const MeltingNoCollectionProfile& profile = MeltingNoCollectionProfile{})
    {
        Result out = accepted;
        std::array<float, 11> output_rates;
        output_rates.fill(7.0f);
        const auto before = output_rates;
        std::string error;
        CHECK_FALSE(evaluate_published_triple_melting_cell(
            input, duration, profile, source, output_rates, out, error));
        CHECK_FALSE(error.empty());
        CHECK(output_rates == before);
        CHECK(science_members(out) == science_members(accepted));
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    for (const State& source : {initial, State{}})
    {
        for (double temperature : {250.0, 283.15})
        {
            const auto input = cell(source, temperature);
            for (double duration : {0.0, -1.0, nan, infinity})
            {
                reject(input, source, duration);
            }
            for (std::size_t index = 0; index < input.size(); ++index)
            {
                for (double value : {-1.0, nan, infinity})
                {
                    auto invalid = input;
                    invalid[index] = value;
                    reject(invalid, source, .1);
                }
            }
            for (const auto boundary : {std::array<double, 2>{0, 149}, {1, 99}, {1, 200001}, {2, 0}})
            {
                auto invalid = input;
                invalid[static_cast<std::size_t>(boundary[0])] = boundary[1];
                reject(invalid, source, .1);
            }
            for (std::size_t channel = 0; channel < 8; ++channel)
            {
                MeltingNoCollectionProfile profile;
                profile.unowned_collection_increments[channel] = 1;
                reject(input, source, .1, profile);
            }
        }
    }
    for (std::size_t species = 0; species < 3; ++species)
    {
        for (std::size_t moment = 0; moment < 3; ++moment)
        {
            for (double value : {-1.0, 0.0, nan, infinity,
                 static_cast<double>(std::numeric_limits<float>::denorm_min()) / 2,
                 static_cast<double>(std::numeric_limits<float>::max()) * 2})
            {
                auto source = initial;
                auto& population = species == 0 ? source.rain : species == 1 ? source.graupel : source.hail;
                double* target = moment == 0 ? &population.mass_per_kg
                    : moment == 1 ? &population.number_per_kg : &population.reflectivity_per_kg;
                *target = value;
                reject(cell(source), source, .1);
            }
        }
    }
    auto invalid = initial;
    invalid.rain.reflectivity_per_kg = 1e-30;
    reject(cell(invalid), invalid, .1);
    auto mismatched = cell(initial);
    mismatched[6] = std::nextafter(mismatched[6], infinity);
    reject(mismatched, initial, .1);
}

TEST_CASE("triple melting binds paired event provenance and rejects mixed FP32 overflow atomically",
          "[physics][milbrandt][triple_melting]")
{
    const State initial{population(0, 2), population(1, 2), population(2, 20)};
    const auto input = cell(initial);
    std::array<float, 11> rates{};
    const Result accepted = evaluate(input, initial, .1, rates);
    for (std::size_t donor = 0; donor < 2; ++donor)
    {
        for (std::size_t field = 0; field < 12; ++field)
        {
            auto events = accepted.events;
            auto& e = events[donor];
            switch (field)
            {
                case 0: e.species = static_cast<MeltingFrozenSpecies>(7); break;
                case 1: e.entry_mass_per_kg *= 2; break;
                case 2: e.entry_number_per_kg *= 2; break;
                case 3: e.source_alpha += 1; break;
                case 4: e.admitted_mass_transfer_per_kg = e.entry_mass_per_kg * 2; break;
                case 5: e.admitted_number_transfer_per_kg *= 2; break;
                case 6: e.admitted_fraction += .1; break;
                case 7: e.temperature_k += 1; break;
                case 8: e.pressure_pa += 1; break;
                case 9: e.dry_density_kg_m3 += 1; break;
                case 10: e.vapor_per_kg += 1; break;
                case 11: e.duration_s += 1; break;
            }
            Result out = accepted;
            std::string error;
            CHECK_FALSE(apply_published_triple_melting(input, .1, {}, initial, events, out, error));
            CHECK_FALSE(error.empty());
            CHECK(science_members(out) == science_members(accepted));
        }
    }
    const State large{population(0, 2, 1e22, 3e38), population(1, 2, 1e22, 3e38),
                      population(2, 20, 1e22, 3e38)};
    const auto large_cold = cell(large, 250);
    const auto zero = evaluate(large_cold, large, .1, rates);
    auto full_events = zero.events;
    const auto large_warm = cell(large);
    for (auto& e : full_events)
    {
        e = make_melting_accepted_event(e.species, e.entry_mass_per_kg, e.entry_number_per_kg,
            e.source_alpha, e.entry_mass_per_kg, e.entry_number_per_kg, large_warm[0],
            large_warm[1], large_warm[2], large_warm[3], .1);
    }
    Result out = accepted;
    std::string error;
    CHECK_FALSE(apply_published_triple_melting(large_warm, .1, {}, large, full_events, out, error));
    CHECK(error.find("FP32") != std::string::npos);
    CHECK(science_members(out) == science_members(accepted));
}

TEST_CASE("triple melting short cell continuation and actual workers preserve exact science members",
          "[physics][milbrandt][triple_melting]")
{
    const State initial{population(0, 2), population(1, 2), population(2, 20)};
    State continued = initial;
    double temperature = 283.15;
    for (double duration : {.05, .1, .3, .0625})
    {
        const auto input = cell(continued, temperature);
        std::array<float, 11> rates{};
        const auto result = evaluate(input, continued, duration, rates);
        check_reference(input, continued, duration, rates, result);
        temperature += duration * rates[5];
        CHECK(temperature > microphysics_constants::T0 + 5);
        continued = stored_state(result);
    }
    std::array<float, 11> rates{};
    const auto extinct = evaluate(cell(initial), initial, 1048576, rates);
    continued = stored_state(extinct);
    CHECK(continued.graupel.mass_per_kg == 0);
    CHECK(continued.hail.mass_per_kg == 0);
    const auto empty = evaluate(cell(continued, temperature), continued, .05, rates);
    CHECK(empty.events[0].admitted_fraction == 0);
    CHECK(empty.events[1].admitted_fraction == 0);
    CHECK(empty.rain.mass_per_kg == extinct.rain.mass_per_kg);
    CHECK(empty.rain.number_per_kg == extinct.rain.number_per_kg);
    CHECK(empty.rain.reflectivity_per_kg == extinct.rain.reflectivity_per_kg);
    CHECK((rates == std::array<float, 11>{}));
#ifdef _OPENMP
    struct Column
    {
        bool accepted = true;
        int team = 0;
        Result result;
        std::array<float, 11> rates{};
        double temperature = 283.15;
    };
    const int dynamic_before = omp_get_dynamic();
    omp_set_dynamic(0);
    const auto run = [&](int workers)
    {
        std::array<Column, 64> columns{};
#pragma omp parallel for num_threads(workers)
        for (int index = 0; index < 64; ++index)
        {
            auto& c = columns[static_cast<std::size_t>(index)];
            c.team = omp_get_num_threads();
            State state = initial;
            for (double duration : {.05, .1, .3, .0625})
            {
                const auto input = cell(state, c.temperature, index % 2 ? .65 : 1.3);
                std::string error;
                if (!evaluate_published_triple_melting_cell(input, duration, {}, state,
                        c.rates, c.result, error))
                {
                    c.accepted = false;
                    break;
                }
                c.temperature += duration * c.rates[5];
                state = stored_state(c.result);
            }
        }
        return columns;
    };
    const auto one = run(1);
    const auto four = run(4);
    omp_set_dynamic(dynamic_before);
    for (std::size_t index = 0; index < one.size(); ++index)
    {
        REQUIRE(one[index].accepted);
        REQUIRE(four[index].accepted);
        CHECK(one[index].team == 1);
        CHECK(four[index].team == 4);
        CHECK(science_members(one[index].result) == science_members(four[index].result));
        CHECK(one[index].rates == four[index].rates);
        CHECK(one[index].temperature == four[index].temperature);
    }
#else
    FAIL("triple melting worker evidence requires OpenMP");
#endif
}
