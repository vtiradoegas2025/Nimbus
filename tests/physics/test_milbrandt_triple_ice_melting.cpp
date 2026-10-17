#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/triple_ice_melting.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace
{

using namespace tmv::milbrandt_detail;
using State = RetainedIceMeltingTripleState;
using Result = RetainedIceMeltingTripleResult;
constexpr std::array<long double, 2> coefficients{523.598775598L, 440.0L};

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
        / (coefficients[species] * coefficients[species] * n))};
}

std::array<double, 16> cell(const State& state, double temperature = 283.15,
                           double density = .65, double pressure = 90000)
{
    return {temperature, pressure, density, .004, 0, state.rain.mass_per_kg,
        state.ice.mass_per_kg, 0, 0, 0, 0, state.rain.number_per_kg,
        state.ice.number_per_kg, 0, 0, 0};
}

std::array<double, 24> members(const Result& r)
{
    return {r.unrounded.rain.mass_per_kg, r.unrounded.rain.number_per_kg,
        r.unrounded.rain.reflectivity_per_kg, r.unrounded.ice.mass_per_kg,
        r.unrounded.ice.number_per_kg, r.unrounded.ice.reflectivity_per_kg,
        r.rain.mass_per_kg, r.rain.number_per_kg, r.rain.reflectivity_per_kg,
        r.ice.mass_per_kg, r.ice.number_per_kg, r.ice.reflectivity_per_kg,
        r.event.entry_mass_per_kg, r.event.entry_number_per_kg,
        r.event.entry_reflectivity_per_kg, r.event.source_alpha,
        r.event.admitted_mass_transfer_per_kg, r.event.admitted_number_transfer_per_kg,
        r.event.admitted_fraction, r.event.temperature_k, r.event.pressure_pa,
        r.event.dry_density_kg_m3, r.event.vapor_per_kg, r.event.duration_s};
}

Result evaluate(const State& initial, const std::array<double, 16>& input,
                double duration, std::array<float, 11>& rates)
{
    Result result;
    std::string error;
    const bool accepted = evaluate_retained_triple_ice_melting_cell(
        input, duration, {}, initial, rates, result, error);
    INFO(error);
    REQUIRE(accepted);
    return result;
}

} // namespace

TEST_CASE("retained instantaneous ice melting conserves independent mass number scaled Z and heat",
          "[physics][milbrandt][triple_ice_melting]")
{
    for (long double alpha : {0.0L, 2.0L, 20.0L})
    {
        for (bool receiver : {false, true})
        {
            const State initial{receiver ? population(0, 2) : population(0, 0, 0, 0), population(1, alpha)};
            for (double density : {.65, 1.3})
            {
                for (double pressure : {70000.0, 90000.0})
                {
                    for (double duration : {.05, .1, .3, 2.0})
                    {
                        std::array<float, 11> rates{};
                        const auto input = cell(initial, 283.15, density, pressure);
                        const auto result = evaluate(initial, input, duration, rates);
                        const long double rain_q = static_cast<long double>(initial.rain.mass_per_kg) + initial.ice.mass_per_kg;
                        const long double rain_n = static_cast<long double>(initial.rain.number_per_kg) + initial.ice.number_per_kg;
                        const long double rain_z = initial.rain.reflectivity_per_kg
                            + std::pow(coefficients[1] / coefficients[0], 2) * initial.ice.reflectivity_per_kg;
                        CHECK(result.unrounded.rain.mass_per_kg == static_cast<double>(rain_q));
                        CHECK(result.unrounded.rain.number_per_kg == static_cast<double>(rain_n));
                        CHECK(result.unrounded.rain.reflectivity_per_kg == Approx(static_cast<double>(rain_z)).epsilon(1e-12));
                        CHECK(result.rain.mass_per_kg == static_cast<float>(rain_q));
                        CHECK(result.rain.number_per_kg == static_cast<float>(rain_n));
                        CHECK(result.rain.reflectivity_per_kg == Approx(static_cast<double>(rain_z)).epsilon(4e-7));
                        CHECK(result.unrounded.ice.mass_per_kg == 0);
                        CHECK(result.unrounded.ice.number_per_kg == 0);
                        CHECK(result.unrounded.ice.reflectivity_per_kg == 0);
                        CHECK(result.ice.mass_per_kg == 0);
                        CHECK(result.ice.number_per_kg == 0);
                        CHECK(result.ice.reflectivity_per_kg == 0);
                        CHECK(result.event.admitted_mass_transfer_per_kg == initial.ice.mass_per_kg);
                        CHECK(result.event.admitted_number_transfer_per_kg == initial.ice.number_per_kg);
                        CHECK(result.event.admitted_fraction == 1);
                        CHECK(result.event.entry_reflectivity_per_kg == initial.ice.reflectivity_per_kg);
                        CHECK(result.event.temperature_k == input[0]);
                        CHECK(result.event.pressure_pa == input[1]);
                        CHECK(result.event.dry_density_kg_m3 == input[2]);
                        CHECK(result.event.vapor_per_kg == input[3]);
                        CHECK(result.event.duration_s == duration);
                        CHECK(rates[0] == static_cast<float>(-initial.ice.mass_per_kg / duration));
                        CHECK(rates[4] == -rates[0]);
                        CHECK(rates[7] == static_cast<float>(-initial.ice.number_per_kg / duration));
                        CHECK(rates[6] == -rates[7]);
                        CHECK(rates[5] == Approx(static_cast<double>(-microphysics_constants::L_f
                            / microphysics_constants::cp * static_cast<long double>(initial.ice.mass_per_kg) / duration)).epsilon(4e-7));
                        for (std::size_t index : {1, 2, 3, 8, 9, 10})
                        {
                            CHECK(rates[index] == 0);
                        }
                        for (long double dry_mass : {1.0L, static_cast<long double>(density) * 30})
                        {
                            CHECK(static_cast<double>(dry_mass * result.rain.mass_per_kg)
                                == Approx(static_cast<double>(dry_mass * rain_q)).epsilon(4e-7));
                            CHECK(static_cast<double>(dry_mass * result.rain.number_per_kg)
                                == Approx(static_cast<double>(dry_mass * rain_n)).epsilon(4e-7));
                            CHECK(static_cast<double>(dry_mass * coefficients[0] * coefficients[0] * result.rain.reflectivity_per_kg)
                                == Approx(static_cast<double>(dry_mass * (coefficients[0] * coefficients[0]
                                    * initial.rain.reflectivity_per_kg + coefficients[1] * coefficients[1]
                                    * initial.ice.reflectivity_per_kg))).epsilon(4e-7));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("retained ice melting validates cold boundary empty continuation and exact event provenance",
          "[physics][milbrandt][triple_ice_melting]")
{
    const State initial{population(0, 2), population(1, 20)};
    for (double temperature : {250.0, std::nextafter(microphysics_constants::T0, 0.0),
         microphysics_constants::T0, std::nextafter(microphysics_constants::T0, 1000.0), 283.15})
    {
        std::array<float, 11> rates{};
        const auto input = cell(initial, temperature);
        const auto result = evaluate(initial, input, .1, rates);
        const bool active = temperature > microphysics_constants::T0;
        CHECK(result.event.admitted_fraction == (active ? 1 : 0));
        CHECK(result.ice.mass_per_kg == (active ? 0 : initial.ice.mass_per_kg));
        CHECK(result.rain.mass_per_kg == static_cast<float>(initial.rain.mass_per_kg + (active ? initial.ice.mass_per_kg : 0)));
    }
    std::array<float, 11> rates{};
    const auto warm = cell(initial);
    const auto full = evaluate(initial, warm, .1, rates);
    const State empty{{full.rain.mass_per_kg, full.rain.number_per_kg, full.rain.reflectivity_per_kg}, {}};
    const auto continued = evaluate(empty, cell(empty), .3, rates);
    CHECK(continued.rain.mass_per_kg == full.rain.mass_per_kg);
    CHECK(continued.rain.number_per_kg == full.rain.number_per_kg);
    CHECK(continued.rain.reflectivity_per_kg == full.rain.reflectivity_per_kg);
    CHECK(continued.event.admitted_fraction == 0);
    CHECK((rates == std::array<float, 11>{}));
    for (int mutation = 0; mutation < 13; ++mutation)
    {
        auto event = full.event;
        switch (mutation)
        {
            case 0: event.entry_mass_per_kg *= 2; break;
            case 1: event.entry_number_per_kg *= 2; break;
            case 2: event.entry_reflectivity_per_kg *= 2; break;
            case 3: event.source_alpha += 1; break;
            case 4: event.admitted_mass_transfer_per_kg *= .5; break;
            case 5: event.admitted_number_transfer_per_kg *= .5; break;
            case 6: event.admitted_fraction = .5; break;
            case 7: event.temperature_k += 1; break;
            case 8: event.pressure_pa += 1; break;
            case 9: event.dry_density_kg_m3 += 1; break;
            case 10: event.vapor_per_kg += 1; break;
            case 11: event.duration_s += 1; break;
            case 12:
                event.admitted_mass_transfer_per_kg *= .5;
                event.admitted_number_transfer_per_kg *= .5;
                event.admitted_fraction = .5;
                break;
        }
        Result out = full;
        std::string error;
        CHECK_FALSE(apply_retained_triple_ice_melting(warm, .1, {}, initial, event, out, error));
        CHECK_FALSE(error.empty());
        CHECK(members(out) == members(full));
    }
}

TEST_CASE("retained ice melting rejects every unowned channel invalid source and rate atomically",
          "[physics][milbrandt][triple_ice_melting]")
{
    const State initial{population(0, 2), population(1, 20)};
    std::array<float, 11> rates{};
    const auto accepted = evaluate(initial, cell(initial), .1, rates);
    const auto reject = [&](const State& source, const std::array<double, 16>& input,
                            double duration, const MeltingNoCollectionProfile& profile = MeltingNoCollectionProfile{})
    {
        Result out = accepted;
        std::array<float, 11> output;
        output.fill(42.0f);
        const auto before = output;
        std::string error;
        CHECK_FALSE(evaluate_retained_triple_ice_melting_cell(input, duration, profile,
            source, output, out, error));
        CHECK_FALSE(error.empty());
        CHECK(output == before);
        CHECK(members(out) == members(accepted));
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    for (double temperature : {250.0, 283.15})
    {
        const auto input = cell(initial, temperature);
        for (std::size_t index : {4, 7, 8, 9, 10, 13, 14, 15})
        {
            for (double value : {-1.0, 1.0, nan, infinity})
            {
                auto invalid = input;
                invalid[index] = value;
                reject(initial, invalid, .1);
            }
        }
        for (std::size_t index = 0; index < 8; ++index)
        {
            for (double value : {-1.0, 1.0, nan, infinity})
            {
                MeltingNoCollectionProfile profile;
                profile.unowned_collection_increments[index] = value;
                reject(initial, input, .1, profile);
            }
        }
        for (double duration : {0.0, -1.0, nan, infinity})
        {
            reject(initial, input, duration);
        }
        for (std::size_t index : {0, 1, 2, 3})
        {
            for (double value : {-1.0, nan, infinity})
            {
                auto invalid = input;
                invalid[index] = value;
                reject(initial, invalid, .1);
            }
        }
        MeltingNoCollectionProfile cancellation;
        cancellation.unowned_collection_increments = {1, -1, 1, -1, 1, -1, 1, -1};
        reject(initial, input, .1, cancellation);
    }
    std::string error;
    for (std::size_t index = 0; index < 8; ++index)
    {
        for (double value : {-1.0, 1.0, nan, infinity})
        {
            std::array<double, 8> channels{};
            channels[index] = value;
            CHECK_FALSE(validate_retained_ice_melting_unowned_channels(channels, error));
            CHECK(error.find(retained_ice_melting_unowned_channel_names[index]) != std::string::npos);
        }
    }
    CHECK_FALSE(validate_retained_ice_melting_unowned_channels({1, -1, 1, -1, 1, -1, 1, -1}, error));
    for (bool rain : {false, true})
    {
        for (std::size_t moment = 0; moment < 3; ++moment)
        {
            for (double value : {-1.0, 0.0, nan, infinity,
                 static_cast<double>(std::numeric_limits<float>::denorm_min()) / 2,
                 static_cast<double>(std::numeric_limits<float>::max()) * 2})
            {
                auto source = initial;
                auto& p = rain ? source.rain : source.ice;
                double* target = moment == 0 ? &p.mass_per_kg
                    : moment == 1 ? &p.number_per_kg : &p.reflectivity_per_kg;
                *target = value;
                reject(source, cell(source), .1);
            }
        }
    }
    auto mismatched = cell(initial);
    mismatched[12] = std::nextafter(mismatched[12], infinity);
    reject(initial, mismatched, .1);
    reject(initial, cell(initial), std::numeric_limits<double>::max());
    reject(initial, cell(initial), std::numeric_limits<double>::min());
    const State large{population(0, 2, 1e22, 3e38), population(1, 20, 1e22, 3e38)};
    reject(large, cell(large), 1);
    const State empty{};
    auto invalid_empty = cell(empty, 250);
    invalid_empty[3] = -1;
    reject(empty, invalid_empty, .1);
}
