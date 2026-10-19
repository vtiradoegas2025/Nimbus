#include "catch2/catch.hpp"
#include "microphysics/milbrandt_cold_parameters.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_freezing.hpp"
#include "microphysics/schemes/milbrandt/triple_sedimentation.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>

namespace
{

using namespace tmv::milbrandt_detail;

struct ParticleBin
{
    long double diameter_m;
    long double number_per_kg;
};

constexpr std::array<ParticleBin, 3> rain_bins{
    ParticleBin{.3e-3L, 200.0L}, {.8e-3L, 40.0L}, {1.4e-3L, 3.0L}};
constexpr std::array<ParticleBin, 3> frozen_bins{
    ParticleBin{.4e-3L, 80.0L}, {.9e-3L, 7.0L}, {1.7e-3L, 2.0L}};

ColdFreezingTripleMomentState spectrum_state(
    const std::array<ParticleBin, 3>& bins, long double coefficient)
{
    std::array<long double, 3> sums{};
    for (const auto& bin : bins)
    {
        sums[0] += bin.number_per_kg * coefficient * std::pow(bin.diameter_m, 3.0L);
        sums[1] += bin.number_per_kg;
        sums[2] += bin.number_per_kg * std::pow(bin.diameter_m, 6.0L);
    }
    return {static_cast<double>(sums[0]), static_cast<double>(sums[1]),
            static_cast<double>(sums[2])};
}

TripleMomentMassLaw rain_law()
{
    return current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Rain).mass_law;
}

TripleMomentMassLaw frozen_law(ColdRainFreezingDestination destination)
{
    return current_nimbus_triple_sedimentation_parameters(
        destination == ColdRainFreezingDestination::Graupel ?
            TripleMomentSpecies::Graupel : TripleMomentSpecies::Hail).mass_law;
}

ColdRainFreezingAcceptedEvent event_for(
    const ColdFreezingTripleMomentState& source,
    ColdRainFreezingDestination destination,
    double fraction, double density = .65, double dt = .2)
{
    if (fraction == 0.0)
    {
        return {destination, source.mass_per_kg, source.number_per_kg,
                0.0, 0.0, 0.0, density, dt};
    }
    return make_cold_rain_freezing_accepted_event(
        destination, source.mass_per_kg, source.number_per_kg,
        fraction * source.mass_per_kg, fraction * source.number_per_kg, density, dt);
}

double relative_error(double actual, double expected)
{
    return std::abs(actual - expected) / std::max(std::abs(expected), 1e-300);
}

ColdFreezingTripleMomentResult sentinel_result()
{
    return {{.0125f, 23.0f, .001f}, {.025f, 47.0f, .002f}};
}

double cubic_gamma_z(double q, double number, double coefficient, double alpha = 0.0)
{
    const double ratio = (alpha + 4.0) * (alpha + 5.0) * (alpha + 6.0) /
        ((alpha + 1.0) * (alpha + 2.0) * (alpha + 3.0));
    return ratio * q * q / (coefficient * coefficient * number);
}

} // namespace

TEST_CASE("Equal-fraction freezing matches independent particle mass and sixth moments",
          "[physics][milbrandt][triple_cold_freezing]")
{
    const auto source_law = rain_law();
    const auto source = spectrum_state(rain_bins, source_law.coefficient);
    for (auto destination : {ColdRainFreezingDestination::Graupel, ColdRainFreezingDestination::Hail})
    {
        const auto destination_law = frozen_law(destination);
        const long double diameter_scale = std::pow(
            static_cast<long double>(source_law.coefficient) / destination_law.coefficient, 1.0L / 3.0L);
        for (const auto& bin : rain_bins)
        {
            const long double source_particle_mass =
                source_law.coefficient * std::pow(bin.diameter_m, 3.0L);
            const long double destination_particle_mass =
                destination_law.coefficient * std::pow(diameter_scale * bin.diameter_m, 3.0L);
            CHECK(std::abs(destination_particle_mass / source_particle_mass - 1.0L) < 2e-15L);
        }
        for (bool existing_destination : {false, true})
        {
            const auto initial_destination = existing_destination ?
                spectrum_state(frozen_bins, destination_law.coefficient) : ColdFreezingTripleMomentState{};
            for (double fraction : {0.0, .125, .5, 1.0})
            {
                for (double density : {.4, .65, 1.6})
                {
                    const auto event = event_for(source, destination, fraction, density);
                    auto result = sentinel_result();
                    std::string error;
                    INFO(error);
                    REQUIRE(apply_equal_fraction_rain_freezing(
                        source, initial_destination, density, .2, event,
                        source_law, destination_law, result, error));
                    CHECK(error.empty());
                    std::array<long double, 3> source_expected{};
                    std::array<long double, 3> destination_expected{
                        initial_destination.mass_per_kg, initial_destination.number_per_kg,
                        initial_destination.reflectivity_per_kg};
                    for (const auto& bin : rain_bins)
                    {
                        const long double surviving_number = (1.0L - fraction) * bin.number_per_kg;
                        const long double frozen_number = fraction * bin.number_per_kg;
                        const long double destination_diameter = bin.diameter_m * diameter_scale;
                        source_expected[0] += surviving_number * source_law.coefficient *
                            std::pow(bin.diameter_m, 3.0L);
                        source_expected[1] += surviving_number;
                        source_expected[2] += surviving_number * std::pow(bin.diameter_m, 6.0L);
                        destination_expected[0] += frozen_number * destination_law.coefficient *
                            std::pow(destination_diameter, 3.0L);
                        destination_expected[1] += frozen_number;
                        destination_expected[2] += frozen_number * std::pow(destination_diameter, 6.0L);
                    }
                    const std::array<double, 3> actual_source{
                        result.rain.mass_per_kg, result.rain.number_per_kg, result.rain.reflectivity_per_kg};
                    const std::array<double, 3> actual_destination{
                        result.frozen.mass_per_kg, result.frozen.number_per_kg, result.frozen.reflectivity_per_kg};
                    for (std::size_t field = 0; field < 3; ++field)
                    {
                        CHECK(relative_error(actual_source[field], static_cast<double>(source_expected[field])) < 4e-7);
                        CHECK(relative_error(actual_destination[field], static_cast<double>(destination_expected[field])) < 4e-7);
                    }
                    CHECK(relative_error(density * (result.rain.mass_per_kg + double(result.frozen.mass_per_kg)),
                        density * (source.mass_per_kg + initial_destination.mass_per_kg)) < 4e-7);
                    CHECK(relative_error(density * (result.rain.number_per_kg + double(result.frozen.number_per_kg)),
                        density * (source.number_per_kg + initial_destination.number_per_kg)) < 4e-7);
                    const double scaled_z_before = source_law.coefficient * source_law.coefficient *
                        source.reflectivity_per_kg + destination_law.coefficient * destination_law.coefficient *
                        initial_destination.reflectivity_per_kg;
                    const double scaled_z_after = source_law.coefficient * source_law.coefficient *
                        result.rain.reflectivity_per_kg + destination_law.coefficient * destination_law.coefficient *
                        result.frozen.reflectivity_per_kg;
                    CHECK(relative_error(density * scaled_z_after, density * scaled_z_before) < 4e-7);
                    if (fraction == 1.0)
                    {
                        CHECK(result.rain.mass_per_kg == 0.0f);
                        CHECK(result.rain.number_per_kg == 0.0f);
                        CHECK(result.rain.reflectivity_per_kg == 0.0f);
                    }
                }
            }
        }
    }
}

TEST_CASE("Equal-fraction freezing preserves zero-transfer state including empty rain",
          "[physics][milbrandt][triple_cold_freezing]")
{
    for (auto destination : {ColdRainFreezingDestination::Graupel, ColdRainFreezingDestination::Hail})
    {
        const auto source_law = rain_law();
        const auto destination_law = frozen_law(destination);
        const auto destination_input = spectrum_state(frozen_bins, destination_law.coefficient);
        for (bool empty_source : {false, true})
        {
            const auto source_input = empty_source ? ColdFreezingTripleMomentState{} :
                spectrum_state(rain_bins, source_law.coefficient);
            const auto event = event_for(source_input, destination, 0.0);
            ColdFreezingTripleMomentResult result;
            std::string error;
            REQUIRE(apply_equal_fraction_rain_freezing(
                source_input, destination_input, .65, .2, event,
                source_law, destination_law, result, error));
            const ColdFreezingTripleMomentResult expected{
                {static_cast<float>(source_input.mass_per_kg),
                 static_cast<float>(source_input.number_per_kg),
                 static_cast<float>(source_input.reflectivity_per_kg)},
                {static_cast<float>(destination_input.mass_per_kg),
                 static_cast<float>(destination_input.number_per_kg),
                 static_cast<float>(destination_input.reflectivity_per_kg)}};
            CHECK(std::memcmp(&result, &expected, sizeof(result)) == 0);
        }
    }
}

TEST_CASE("Equal-fraction freezing requires exact event entry and rejects invalid inputs atomically",
          "[physics][milbrandt][triple_cold_freezing]")
{
    const auto source_law = rain_law();
    const auto destination_law = frozen_law(ColdRainFreezingDestination::Graupel);
    const auto initial_source = spectrum_state(rain_bins, source_law.coefficient);
    const auto initial_destination = spectrum_state(frozen_bins, destination_law.coefficient);
    for (int invalid = 0; invalid < 24; ++invalid)
    {
        auto source = initial_source;
        auto destination = initial_destination;
        auto source_mass_law = source_law;
        auto destination_mass_law = destination_law;
        auto event = event_for(source, ColdRainFreezingDestination::Graupel, .25);
        double density = .65;
        double dt = .2;
        switch (invalid)
        {
        case 0: event.entry_rain_mass_per_kg = std::nextafter(event.entry_rain_mass_per_kg, 1.0); break;
        case 1: event.entry_rain_number_per_kg = std::nextafter(event.entry_rain_number_per_kg, 1e9); break;
        case 2: event.dry_density_kg_m3 = std::nextafter(event.dry_density_kg_m3, 1.0); break;
        case 3: event.duration_s = std::nextafter(event.duration_s, 1.0); break;
        case 4: event.admitted_fraction = std::nextafter(event.admitted_fraction, 1.0); break;
        case 5: event.admitted_mass_increment_per_kg = -1.0; break;
        case 6: event.admitted_mass_increment_per_kg = 1.01 * source.mass_per_kg; break;
        case 7: event.admitted_number_increment_per_kg = 1.01 * source.number_per_kg; break;
        case 8: event.admitted_mass_increment_per_kg *= 2.0; break;
        case 9: event.admitted_fraction = std::numeric_limits<double>::quiet_NaN(); break;
        case 10: source_mass_law.coefficient = std::nextafter(source_mass_law.coefficient, 1e9); break;
        case 11: source_mass_law.exponent = 2.078; break;
        case 12: destination_mass_law.coefficient = frozen_law(ColdRainFreezingDestination::Hail).coefficient; break;
        case 13: destination_mass_law.exponent = 2.078; break;
        case 14: event.destination = static_cast<ColdRainFreezingDestination>(9); break;
        case 15: source.reflectivity_per_kg = 0.0; break;
        case 16: source.reflectivity_per_kg = -1.0; break;
        case 17: destination.reflectivity_per_kg = std::numeric_limits<double>::quiet_NaN(); break;
        case 18: destination.number_per_kg = 0.0; break;
        case 19: density = 0.0; break;
        case 20: dt = 0.0; break;
        case 21:
            event.admitted_mass_increment_per_kg = 0.0;
            event.admitted_number_increment_per_kg = 0.0;
            break;
        case 22: event.admitted_mass_increment_per_kg = std::numeric_limits<double>::infinity(); break;
        case 23: source_mass_law.coefficient = std::numeric_limits<double>::quiet_NaN(); break;
        }
        const auto before_source = source;
        const auto before_destination = destination;
        auto result = sentinel_result();
        const auto before_result = result;
        std::string error;
        INFO(invalid);
        CHECK_FALSE(apply_equal_fraction_rain_freezing(
            source, destination, density, dt, event, source_mass_law,
            destination_mass_law, result, error));
        CHECK_FALSE(error.empty());
        CHECK(std::memcmp(&result, &before_result, sizeof(result)) == 0);
        CHECK(std::memcmp(&source, &before_source, sizeof(source)) == 0);
        CHECK(std::memcmp(&destination, &before_destination, sizeof(destination)) == 0);
    }
}

TEST_CASE("Equal-fraction freezing keeps the shared gamma mixture bracket closed",
          "[physics][milbrandt][triple_cold_freezing]")
{
    const auto source_law = rain_law();
    const auto destination_law = frozen_law(ColdRainFreezingDestination::Hail);
    const ColdFreezingTripleMomentState source{
        1e-3, 1e-8, cubic_gamma_z(1e-3, 1e-8, source_law.coefficient)};
    const ColdFreezingTripleMomentState destination{
        1e-12, 1e12, cubic_gamma_z(1e-12, 1e12, destination_law.coefficient)};
    const auto event = event_for(source, ColdRainFreezingDestination::Hail, .5);
    double alpha = 0.0;
    double lambda = 0.0;
    std::string error;
    REQUIRE(diagnose_triple_gamma_shape(source.mass_per_kg, source.number_per_kg,
        source.reflectivity_per_kg, source_law, alpha, lambda, error));
    REQUIRE(diagnose_triple_gamma_shape(destination.mass_per_kg, destination.number_per_kg,
        destination.reflectivity_per_kg, destination_law, alpha, lambda, error));
    const double ratio = source_law.coefficient / destination_law.coefficient;
    CHECK_FALSE(diagnose_triple_gamma_shape(
        destination.mass_per_kg + .5 * source.mass_per_kg,
        destination.number_per_kg + .5 * source.number_per_kg,
        destination.reflectivity_per_kg + .5 * source.reflectivity_per_kg * ratio * ratio,
        destination_law, alpha, lambda, error));
    CHECK(error.find("bounded inversion bracket") != std::string::npos);
    auto result = sentinel_result();
    const auto before = result;
    CHECK_FALSE(apply_equal_fraction_rain_freezing(
        source, destination, .65, .2, event, source_law, destination_law, result, error));
    CHECK(error.find("destination endpoint") != std::string::npos);
    CHECK(error.find("bounded inversion bracket") != std::string::npos);
    CHECK(std::memcmp(&result, &before, sizeof(result)) == 0);
}

TEST_CASE("Equal-fraction freezing rejects late destination FP32 overflow and underflow atomically",
          "[physics][milbrandt][triple_cold_freezing]")
{
    const auto source_law = rain_law();
    const auto destination_law = frozen_law(ColdRainFreezingDestination::Hail);
    for (bool underflow : {false, true})
    {
        const double huge_number = .75 * std::numeric_limits<float>::max();
        const double tiny_z = 2.0 * std::numeric_limits<float>::denorm_min();
        const double tiny_q = source_law.coefficient * std::sqrt(tiny_z / 20.0);
        const ColdFreezingTripleMomentState source = underflow ?
            ColdFreezingTripleMomentState{tiny_q, 1.0, tiny_z} :
            ColdFreezingTripleMomentState{1.0, huge_number,
                cubic_gamma_z(1.0, huge_number, source_law.coefficient)};
        const ColdFreezingTripleMomentState destination = underflow ?
            ColdFreezingTripleMomentState{} : ColdFreezingTripleMomentState{
                1.0, huge_number, cubic_gamma_z(1.0, huge_number, destination_law.coefficient)};
        const double fraction = underflow ? .25 : .5;
        const auto event = event_for(source, ColdRainFreezingDestination::Hail, fraction);
        auto result = sentinel_result();
        const auto before = result;
        std::string error;
        CHECK_FALSE(apply_equal_fraction_rain_freezing(
            source, destination, .65, .2, event, source_law, destination_law, result, error));
        CHECK(error == "rain-freezing rounded destination endpoint is not FP32 representable");
        CHECK(std::memcmp(&result, &before, sizeof(result)) == 0);
    }
}

TEST_CASE("A real limited cold observer event reaches the isolated equal-fraction transfer",
          "[physics][milbrandt][triple_cold_freezing]")
{
    const auto& p = default_cold_parameters;
    const ColdParameters parameters{
        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], false};
    std::array<double, 16> cell{
        233, 60000, double(.6f), 0, double(1e-5f), double(.0005f),
        double(.003f), 0, double(1e-10f), double(1e-5f),
        2e8, 1, 100, 1e4, 1e6, 0};
    std::array<float, 14> rates{};
    rates[9] = 1.0f;
    const auto incoming = rates;
    auto unobserved = rates;
    cold_cell(parameters, cell, .2, unobserved);
    ColdRainFreezingAcceptedEventBuffer buffer;
    cold_cell(parameters, cell, .2, rates, &buffer);
    REQUIRE(buffer.size == 1);
    CHECK(std::memcmp(rates.data(), unobserved.data(), sizeof(rates)) == 0);
    const auto& event = buffer.events[0];
    const double entry_number = cell[11] + .2 * incoming[9];
    CHECK(event.entry_rain_number_per_kg == entry_number);
    const auto source_law = rain_law();
    const auto destination_law = frozen_law(event.destination);
    const ColdFreezingTripleMomentState source{
        cell[5], entry_number, cubic_gamma_z(cell[5], entry_number, source_law.coefficient)};
    const ColdFreezingTripleMomentState destination{
        .0003, 2.0, cubic_gamma_z(.0003, 2.0, destination_law.coefficient)};
    ColdFreezingTripleMomentResult result;
    std::string error;
    INFO(error);
    REQUIRE(apply_equal_fraction_rain_freezing(
        source, destination, cell[2], .2, event, source_law, destination_law, result, error));
    const double coefficient_ratio = source_law.coefficient / destination_law.coefficient;
    CHECK(result.rain.mass_per_kg == static_cast<float>(source.mass_per_kg - event.admitted_mass_increment_per_kg));
    CHECK(result.rain.number_per_kg == static_cast<float>(source.number_per_kg - event.admitted_number_increment_per_kg));
    CHECK(result.rain.reflectivity_per_kg == static_cast<float>(
        source.reflectivity_per_kg - event.admitted_fraction * source.reflectivity_per_kg));
    CHECK(result.frozen.reflectivity_per_kg == static_cast<float>(
        destination.reflectivity_per_kg + event.admitted_fraction * source.reflectivity_per_kg *
            coefficient_ratio * coefficient_ratio));
}
