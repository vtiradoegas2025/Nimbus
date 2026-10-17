#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/triple_snow_melting.hpp"
#include "support/milbrandt_snow_melting_reference.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace
{

using namespace tmv::milbrandt_detail;
using State = RetainedSnowMeltingTripleState;
using Result = RetainedSnowMeltingTripleResult;

MeltingTripleMomentState rain(bool receiver = true)
{
    constexpr long double c = 523.598775598L;
    return receiver ? MeltingTripleMomentState{1e-4f, 10.0f,
        static_cast<float>(5.6L * static_cast<long double>(1e-4f) * 1e-4f / (c * c * 10))}
        : MeltingTripleMomentState{};
}

std::array<double, 16> cell(const State& state, double temperature = 283.15,
                           double density = .65, double pressure = 90000, double humidity = 1)
{
    return {temperature, pressure, density, humidity * thermodynamics::saturation_mixing_ratio_water(
        microphysics_constants::T0, pressure), 0, state.rain.mass_per_kg, 0, state.snow.mass_per_kg,
        0, 0, 0, state.rain.number_per_kg, 0, state.snow.number_per_kg, 0, 0};
}

Result evaluate(const State& state, const std::array<double, 16>& input,
                double duration, std::array<float, 11>& rates)
{
    Result result;
    std::string error;
    const bool accepted = evaluate_retained_triple_snow_melting_cell(
        input, duration, {}, state, rates, result, error);
    INFO(error);
    REQUIRE(accepted);
    return result;
}

std::array<double, 12> endpoints(const Result& r)
{
    return {r.unrounded.rain.mass_per_kg, r.unrounded.rain.number_per_kg, r.unrounded.rain.reflectivity_per_kg,
        r.unrounded.snow.mass_per_kg, r.unrounded.snow.number_per_kg, r.unrounded.snow.reflectivity_per_kg,
        r.rain.mass_per_kg, r.rain.number_per_kg, r.rain.reflectivity_per_kg,
        r.snow.mass_per_kg, r.snow.number_per_kg, r.snow.reflectivity_per_kg};
}

} // namespace

TEST_CASE("retained noncubic snow melting agrees with independent thermal and mass squared quadrature",
          "[physics][milbrandt][triple_snow_melting]")
{
    for (long double alpha : {2.0L, 20.0L, 200.0L})
    {
        for (bool receiver : {false, true})
        {
            const State initial{rain(receiver), snow_melting_reference::population(alpha)};
            for (double density : {.65, 1.3})
            {
                for (double pressure : {70000.0, 90000.0})
                {
                    for (double humidity : {.5, 1.1})
                    {
                        const auto input = cell(initial, 283.15, density, pressure, humidity);
                        const auto base = snow_melting_reference::evaluate(initial.snow,
                            input[0], input[1], input[2], input[3], .1);
                        for (double duration : {.05, .3, 1e6})
                        {
                            const long double melted = std::min(static_cast<long double>(initial.snow.mass_per_kg),
                                base.melted_mass * (duration / .1));
                            const long double fraction = melted / initial.snow.mass_per_kg;
                            const long double transferred_number = initial.snow.number_per_kg * fraction;
                            const long double transferred_z = fraction * base.mass_squared
                                / (snow_melting_reference::rain_coefficient * snow_melting_reference::rain_coefficient);
                            std::array<float, 11> rates{};
                            const auto result = evaluate(initial, input, duration, rates);
                            CHECK(result.unrounded.rain.mass_per_kg == Approx(static_cast<double>(initial.rain.mass_per_kg + melted)).epsilon(4e-7).margin(1e-22));
                            CHECK(result.unrounded.rain.number_per_kg == Approx(static_cast<double>(initial.rain.number_per_kg + transferred_number)).epsilon(4e-7));
                            CHECK(result.unrounded.rain.reflectivity_per_kg == Approx(static_cast<double>(initial.rain.reflectivity_per_kg + transferred_z)).epsilon(4e-7).margin(1e-30));
                            CHECK(result.snow.mass_per_kg == Approx(static_cast<double>(initial.snow.mass_per_kg - melted)).epsilon(4e-7).margin(1e-22));
                            CHECK(result.snow.number_per_kg == Approx(static_cast<double>(initial.snow.number_per_kg - transferred_number)).epsilon(4e-7).margin(1e-22));
                            CHECK(result.snow.reflectivity_per_kg == Approx(static_cast<double>((1 - fraction) * initial.snow.reflectivity_per_kg)).epsilon(4e-7).margin(1e-30));
                            CHECK(result.event.admitted_mass_transfer_per_kg == Approx(static_cast<double>(melted)).epsilon(4e-7).margin(1e-22));
                            CHECK(result.event.admitted_number_transfer_per_kg == Approx(static_cast<double>(transferred_number)).epsilon(4e-7).margin(1e-22));
                            CHECK(result.event.source_mass_squared_per_kg == Approx(static_cast<double>(base.mass_squared)).epsilon(4e-7));
                            CHECK(result.event.source_lambda_per_m == Approx(static_cast<double>(base.lambda)).epsilon(4e-7));
                            CHECK(result.event.admitted_rain_reflectivity_per_kg == Approx(static_cast<double>(transferred_z)).epsilon(4e-7).margin(1e-30));
                            CHECK(rates[1] == Approx(static_cast<double>(-melted / duration)).epsilon(4e-7).margin(1e-22));
                            CHECK(rates[4] == -rates[1]);
                            CHECK(rates[6] == Approx(static_cast<double>(transferred_number / duration)).epsilon(4e-7).margin(1e-22));
                            CHECK(rates[8] == -rates[6]);
                            CHECK(rates[5] == Approx(static_cast<double>(-microphysics_constants::L_f
                                / microphysics_constants::cp * melted / duration)).epsilon(4e-7).margin(1e-18));
                            for (long double scale : {1.0L, static_cast<long double>(density) * 30})
                            {
                                CHECK(static_cast<double>(scale * (result.rain.mass_per_kg + static_cast<long double>(result.snow.mass_per_kg)))
                                    == Approx(static_cast<double>(scale * (initial.rain.mass_per_kg + static_cast<long double>(initial.snow.mass_per_kg)))).epsilon(4e-7));
                                CHECK(static_cast<double>(scale * (result.rain.number_per_kg + static_cast<long double>(result.snow.number_per_kg)))
                                    == Approx(static_cast<double>(scale * (initial.rain.number_per_kg + static_cast<long double>(initial.snow.number_per_kg)))).epsilon(4e-7));
                                const long double before = std::pow(snow_melting_reference::rain_coefficient, 2)
                                    * initial.rain.reflectivity_per_kg + base.mass_squared;
                                const long double after = std::pow(snow_melting_reference::rain_coefficient, 2)
                                    * result.rain.reflectivity_per_kg + (1 - fraction) * base.mass_squared;
                                CHECK(static_cast<double>(scale * after) == Approx(static_cast<double>(scale * before)).epsilon(4e-7));
                            }
                            if (fraction == 1)
                            {
                                CHECK(result.snow.mass_per_kg == 0);
                                CHECK(result.snow.number_per_kg == 0);
                                CHECK(result.snow.reflectivity_per_kg == 0);
                                const long double naive = std::pow(snow_melting_reference::coefficient
                                    / snow_melting_reference::rain_coefficient, 2) * initial.snow.reflectivity_per_kg;
                                CHECK(std::abs(transferred_z - naive) > .01L * transferred_z);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("noncubic snow zero boundaries continuation event corruption and guards are atomic",
          "[physics][milbrandt][triple_snow_melting]")
{
    const State initial{rain(), snow_melting_reference::population(2)};
    std::array<float, 11> rates{};
    const auto warm = cell(initial);
    const auto accepted = evaluate(initial, warm, .1, rates);
    for (double temperature : {250.0, microphysics_constants::T0})
    {
        const auto zero = evaluate(initial, cell(initial, temperature), .1, rates);
        CHECK(zero.event.admitted_fraction == 0);
        CHECK(zero.snow.mass_per_kg == initial.snow.mass_per_kg);
        CHECK((rates == std::array<float, 11>{}));
    }
    const auto full = evaluate(initial, warm, 1e6, rates);
    const State empty{{full.rain.mass_per_kg, full.rain.number_per_kg, full.rain.reflectivity_per_kg}, {}};
    const auto continued = evaluate(empty, cell(empty), .1, rates);
    CHECK(continued.rain.reflectivity_per_kg == full.rain.reflectivity_per_kg);
    CHECK((rates == std::array<float, 11>{}));
    for (int mutation = 0; mutation < 9; ++mutation)
    {
        auto event = accepted.event;
        switch (mutation)
        {
            case 0: event.entry_mass_per_kg *= 2; break;
            case 1: event.entry_number_per_kg *= 2; break;
            case 2: event.entry_reflectivity_per_kg *= 2; break;
            case 3: event.source_alpha += 1; break;
            case 4: event.source_lambda_per_m *= 2; break;
            case 5: event.source_mass_squared_per_kg *= 2; break;
            case 6: event.admitted_rain_reflectivity_per_kg *= 2; break;
            case 7: event.temperature_k += 1; break;
            case 8: event.duration_s += 1; break;
        }
        Result out = accepted;
        std::string error;
        CHECK_FALSE(apply_retained_triple_snow_melting(warm, .1, {}, initial, event, out, error));
        CHECK_FALSE(error.empty());
        CHECK(endpoints(out) == endpoints(accepted));
    }
    const auto reject = [&](const State& source, const auto& input, double duration,
                            const MeltingNoCollectionProfile& profile = MeltingNoCollectionProfile{})
    {
        auto out = accepted;
        std::array<float, 11> output;
        output.fill(42);
        const auto before = output;
        std::string error;
        CHECK_FALSE(evaluate_retained_triple_snow_melting_cell(input, duration, profile, source, output, out, error));
        CHECK_FALSE(error.empty());
        CHECK(output == before);
        CHECK(endpoints(out) == endpoints(accepted));
    };
    for (double temperature : {250.0, 283.15})
    {
        for (std::size_t index : {4, 6, 8, 9, 10, 12, 14, 15})
        {
            for (double value : {-1.0, 1.0, std::numeric_limits<double>::quiet_NaN()})
            {
                auto input = cell(initial, temperature);
                input[index] = value;
                reject(initial, input, .1);
            }
        }
        for (std::size_t index = 0; index < 8; ++index)
        {
            MeltingNoCollectionProfile profile;
            profile.unowned_collection_increments[index] = 1;
            reject(initial, cell(initial, temperature), .1, profile);
        }
        MeltingNoCollectionProfile profile;
        profile.unowned_collection_increments = {1, -1, 1, -1, 1, -1, 1, -1};
        reject(initial, cell(initial, temperature), .1, profile);
    }
    for (double duration : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::max()})
    {
        reject(initial, warm, duration);
    }
    const auto tiny = evaluate(initial, warm, std::numeric_limits<double>::min(), rates);
    CHECK(tiny.event.admitted_mass_transfer_per_kg > 0);
    CHECK(rates[4] > 0);
    CHECK(rates[1] == -rates[4]);
    CHECK(rates[8] == -rates[6]);
    CHECK(rates[6] == Approx(initial.snow.number_per_kg / initial.snow.mass_per_kg * rates[4]).epsilon(4e-7));
    CHECK(rates[5] == Approx(-microphysics_constants::L_f / microphysics_constants::cp * rates[4]).epsilon(4e-7));
    for (std::size_t index : {0, 1, 2, 3})
    {
        auto input = warm;
        input[index] = -1;
        reject(initial, input, .1);
    }
    auto source = initial;
    source.snow.reflectivity_per_kg = 0;
    reject(source, cell(source), .1);
    source = initial;
    source.rain.number_per_kg = 0;
    reject(source, cell(source), .1);
}
