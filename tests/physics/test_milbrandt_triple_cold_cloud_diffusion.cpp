#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_cloud_diffusion.hpp"
#include "support/milbrandt_cold_cloud_diffusion_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace
{

using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_cold_cloud_diffusion_reference;

ColdParameters parameters()
{
    const auto& p = default_cold_parameters;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], true};
}

ColdCloudDiffusionCell source(double alpha = 2, double density = 1,
                              double temperature = 263, double humidity = 1)
{
    ColdCloudDiffusionCell cell;
    cell.temperature_k = temperature;
    cell.pressure_pa = 80000;
    cell.dry_density_kg_m3 = density;
    cell.vapor_mass_per_kg = static_cast<double>(reference::saturation(temperature, 80000).mass) * humidity;
    cell.initial.cloud_mass_per_kg = 2e-4;
    cell.initial.cloud_number_per_kg = 8e7;
    const std::array<double, 3> mass{1e-4, 2e-4, 3e-4};
    const std::array<double, 3> number{5000, 600, 80};
    for (std::size_t index = 0; index < 3; ++index)
    {
        cell.initial.frozen[index] = reference::collection::population(index, mass[index], number[index], alpha);
    }
    return cell;
}

void near(double actual, long double expected, long double relative = 2e-8L)
{
    INFO("actual " << actual << ", expected " << static_cast<double>(expected));
    REQUIRE(std::isfinite(actual));
    CHECK(std::abs(static_cast<long double>(actual) - expected)
        <= relative * std::max(std::abs(expected), 1e-280L));
}

void check_reference(const ColdCloudDiffusionCell& cell, double duration,
                     const ColdCloudDiffusionResult& result, bool quadrature = false)
{
    const auto expected = reference::evaluate(cell, duration, quadrature);
    const auto& thermal = result.producer.thermodynamics();
    near(thermal.saturation_ice_per_kg, expected.saturation.mass, 2e-12L);
    near(thermal.saturation_temperature_derivative_per_kg_k, expected.saturation.derivative, 2e-12L);
    near(thermal.diffusion_resistance_m_s_per_kg, expected.resistance, 2e-12L);
    near(thermal.riming_feedback, expected.feedback, 2e-12L);
    near(thermal.relaxation_number, expected.relaxation);
    near(result.vapor_mass_per_kg, expected.vapor_after);
    near(result.cloud_mass_increment_per_kg, expected.collection.cloud_mass_increment);
    near(result.cloud_number_increment_per_kg, expected.collection.cloud_number_increment);
    near(result.temperature_increment_k, expected.temperature_increment);
    long double water = result.vapor_mass_per_kg - static_cast<long double>(cell.vapor_mass_per_kg)
        + result.unrounded.cloud_mass_per_kg - static_cast<long double>(cell.initial.cloud_mass_per_kg);
    for (std::size_t index = 0; index < 3; ++index)
    {
        near(thermal.thermal_lambda_per_m[index], expected.thermal_lambda[index]);
        near(thermal.ventilation_m[index], expected.ventilation[index]);
        near(result.producer.channels()[index].riming_mass_increment_per_kg, expected.riming[index]);
        near(result.producer.channels()[index].admitted_mass_increment_per_kg, expected.diffusion[index]);
        near(result.number_increment_per_kg[index], expected.number_increment[index]);
        near(result.reflectivity_increment_per_kg[index], expected.reflectivity_increment[index]);
        near(result.unrounded.frozen[index].mass_per_kg, expected.endpoint[index][0]);
        near(result.unrounded.frozen[index].number_per_kg, expected.endpoint[index][1]);
        near(result.unrounded.frozen[index].reflectivity_per_kg, expected.endpoint[index][2]);
        CHECK(result.rounded[index].mass_per_kg == static_cast<float>(result.unrounded.frozen[index].mass_per_kg));
        CHECK(result.rounded[index].number_per_kg == static_cast<float>(result.unrounded.frozen[index].number_per_kg));
        CHECK(result.rounded[index].reflectivity_per_kg == static_cast<float>(result.unrounded.frozen[index].reflectivity_per_kg));
        water += result.unrounded.frozen[index].mass_per_kg - static_cast<long double>(cell.initial.frozen[index].mass_per_kg);
    }
    CHECK(std::abs(water) < 2e-18L);
    const long double heat = reference::collection::heat_capacity * result.temperature_increment_k;
    const long double expected_heat = -reference::collection::latent_heat_fusion * result.cloud_mass_increment_per_kg
        - reference::latent_heat_sublimation * result.vapor_mass_increment_per_kg;
    CHECK(std::abs(heat - expected_heat) <= 2e-14L * std::max(std::abs(expected_heat), 1e-30L));
}

ColdCloudDiffusionResult evaluate(const ColdCloudDiffusionCell& cell, double duration)
{
    ColdCloudDiffusionResult result;
    std::string error;
    const bool accepted = evaluate_exact_triple_cold_cloud_diffusion_cell(parameters(), cell, duration, result, error);
    INFO(error);
    REQUIRE(accepted);
    return result;
}

ColdCloudDiffusionCell advance(const ColdCloudDiffusionCell& cell, const ColdCloudDiffusionResult& result)
{
    auto next = cell;
    next.temperature_k += result.temperature_increment_k;
    next.vapor_mass_per_kg = result.vapor_mass_per_kg;
    next.initial = result.unrounded;
    return next;
}

} // namespace

TEST_CASE("coupled cold diffusion matches independent spectra heat and water references",
          "[physics][milbrandt][triple_cold_cloud_diffusion]")
{
    for (double alpha : {-0.5, 0.0, 2.0, 20.0, 199.9})
    {
        for (double density : {.6, 1.0, 1.4})
        {
            for (double humidity : {.75, 1.0, 1.05})
            {
                const auto cell = source(alpha, density, 263, humidity);
                INFO("alpha " << alpha << ", density " << density << ", humidity " << humidity);
                check_reference(cell, 1e-3, evaluate(cell, 1e-3));
            }
        }
    }
}

TEST_CASE("coupled diffusion preserves surface heating sign and spherical snow ventilation",
          "[physics][milbrandt][triple_cold_cloud_diffusion]")
{
    for (double density : {.6, 1.4})
    {
        auto cell = source(2, density);
        const auto result = evaluate(cell, 1e-3);
        check_reference(cell, 1e-3, result, true);
        CHECK(result.temperature_increment_k > 0);
        const auto& producer = result.producer;
        CHECK(std::abs(producer.thermodynamics().thermal_lambda_per_m[0]
            / producer.collection_producer().source_lambda_per_m()[0] - 1) > .01);
        long double relative_excess = 1;
        const auto independent = reference::evaluate(cell, 1e-3);
        for (std::size_t index = 0; index < 3; ++index)
        {
            CHECK(producer.channels()[index].admitted_mass_increment_per_kg < 0);
            const auto collected = producer.collection_producer().channels()[index].admitted_mass_increment_per_kg;
            REQUIRE(collected > 0);
            near(-producer.channels()[index].riming_mass_increment_per_kg / collected, independent.feedback, 2e-12L);
            relative_excess = std::min(relative_excess, -independent.riming[index]
                / (1e-3L * independent.coefficients[index] * independent.saturation.mass));
        }
        cell.vapor_mass_per_kg = static_cast<double>(independent.saturation.mass * (1 + relative_excess / 4));
        const auto supersaturated = evaluate(cell, 1e-3);
        REQUIRE(supersaturated.producer.thermodynamics().saturation_ratio_ice > 1);
        for (const auto& channel : supersaturated.producer.channels())
        {
            CHECK(channel.admitted_mass_increment_per_kg < 0);
        }
    }
}

TEST_CASE("coupled diffusion Buck derivative follows retained unclipped and plateau branches",
          "[physics][milbrandt][triple_cold_cloud_diffusion]")
{
    for (const auto context : {std::array<double, 2>{230, 80000}, {263, 80000}, {270, 100}})
    {
        auto cell = source();
        cell.temperature_k = context[0];
        cell.pressure_pa = context[1];
        cell.initial = {};
        cell.vapor_mass_per_kg = .001;
        const auto result = evaluate(cell, .01);
        const auto expected = reference::saturation(context[0], context[1]);
        near(result.producer.thermodynamics().saturation_ice_per_kg, expected.mass, 2e-12L);
        near(result.producer.thermodynamics().saturation_temperature_derivative_per_kg_k, expected.derivative, 2e-12L);
        const long double numerical = (reference::saturation(context[0] + 1e-3L, context[1]).mass
            - reference::saturation(context[0] - 1e-3L, context[1]).mass) / 2e-3L;
        near(result.producer.thermodynamics().saturation_temperature_derivative_per_kg_k, numerical, 2e-8L);
    }
}

TEST_CASE("coupled diffusion source relaxation and invalid state reject without output mutation",
          "[physics][milbrandt][triple_cold_cloud_diffusion]")
{
    const auto original = source();
    const auto good = evaluate(original, 1e-3);
    for (int mutation = 0; mutation < 12; ++mutation)
    {
        auto cell = original;
        auto p = parameters();
        double duration = 1e-3;
        std::size_t capacity = 3;
        switch (mutation)
        {
            case 0: cell.vapor_mass_per_kg = -1; break;
            case 1: cell.vapor_mass_per_kg = std::numeric_limits<double>::quiet_NaN(); break;
            case 2: cell.temperature_k = 273.15; break;
            case 3: cell.pressure_pa = 99; break;
            case 4: cell.dry_density_kg_m3 = 0; break;
            case 5: cell.initial.cloud_number_per_kg = 0; break;
            case 6: cell.initial.frozen[0].number_per_kg = 0; break;
            case 7: cell.initial.frozen[1].reflectivity_per_kg = -1; break;
            case 8: duration = 0; break;
            case 9: capacity = 2; break;
            case 10: p.b_s_ = std::nextafter(p.b_s_, 1.0); break;
            case 11: duration = 1.01e-3 / good.producer.thermodynamics().relaxation_number; break;
        }
        auto result = good;
        ColdCloudDiffusionFailure failure = ColdCloudDiffusionFailure::None;
        std::string error;
        REQUIRE_FALSE(evaluate_exact_triple_cold_cloud_diffusion_cell(p, cell, duration, result, error, capacity, &failure));
        CHECK(result.vapor_mass_per_kg == good.vapor_mass_per_kg);
        CHECK(result.temperature_increment_k == good.temperature_increment_k);
        CHECK(result.producer.source().vapor_mass_per_kg == original.vapor_mass_per_kg);
        CHECK_FALSE(error.empty());
        if (mutation == 11)
        {
            CHECK(failure == ColdCloudDiffusionFailure::IntervalRejected);
            CHECK(error.find("relaxation") != std::string::npos);
        }
    }
}

TEST_CASE("coupled diffusion captures bind source bits interval and collection policy",
          "[physics][milbrandt][triple_cold_cloud_diffusion]")
{
    const auto cell = source();
    const auto good = evaluate(cell, 1e-3);
    for (int mutation = 0; mutation < 7; ++mutation)
    {
        auto changed = cell;
        auto p = parameters();
        double duration = 1e-3;
        auto capture = good.producer;
        switch (mutation)
        {
            case 0: changed.vapor_mass_per_kg = std::nextafter(changed.vapor_mass_per_kg, 1.0); break;
            case 1: changed.initial.cloud_number_per_kg = std::nextafter(changed.initial.cloud_number_per_kg, 0.0); break;
            case 2: changed.initial.frozen[2].reflectivity_per_kg *= 1.0001; break;
            case 3: changed.temperature_k = std::nextafter(changed.temperature_k, 300.0); break;
            case 4: p.hail_processes = false; break;
            case 5: duration = std::nextafter(duration, 1.0); break;
            case 6: capture = {}; break;
        }
        auto output = good;
        std::string error;
        ColdCloudDiffusionFailure failure{};
        REQUIRE_FALSE(apply_exact_triple_cold_cloud_diffusion_cell(p, changed, duration, capture, output, error, 3, &failure));
        CHECK(failure == ColdCloudDiffusionFailure::InvalidProducer);
        CHECK(output.vapor_mass_per_kg == good.vapor_mass_per_kg);
    }
    auto alias = good;
    std::string error;
    REQUIRE(apply_exact_triple_cold_cloud_diffusion_cell(parameters(), cell, 1e-3, alias.producer, alias, error));
    CHECK(alias.vapor_mass_per_kg == good.vapor_mass_per_kg);
}

TEST_CASE("coupled diffusion Euler refinement converges to independent full state RK4",
          "[physics][milbrandt][triple_cold_cloud_diffusion]")
{
    for (double humidity : {.75, 1.0, 1.05})
    {
        const auto cell = source(2, 1, 263, humidity);
        constexpr double duration = .4;
        const auto coarse_reference = reference::rk4(cell, duration, 32);
        const auto fine_reference = reference::rk4(cell, duration, 64);
        const auto initial = reference::state_vector(cell);
        const auto norm = [&](const reference::StateVector& state)
        {
            long double maximum = 0;
            for (std::size_t index = 0; index < state.size(); ++index)
            {
                maximum = std::max(maximum, std::abs(state[index] - fine_reference[index])
                    / std::max(std::abs(initial[index]), 1e-30L));
            }
            return maximum;
        };
        REQUIRE(norm(coarse_reference) < 1e-9L);
        long double previous = 1;
        for (int steps : {4, 8, 16})
        {
            auto current = cell;
            for (int step = 0; step < steps; ++step)
            {
                current = advance(current, evaluate(current, duration / steps));
            }
            const auto error = norm(reference::state_vector(current));
            INFO("humidity " << humidity << ", steps " << steps << ", error " << static_cast<double>(error));
            REQUIRE(error > 1e-14L);
            CHECK(error < previous * .7L);
            previous = error;
        }
    }
}

TEST_CASE("coupled diffusion surface heating uses the admitted cloud budget",
          "[physics][milbrandt][triple_cold_cloud_diffusion]")
{
    auto cell = source();
    cell.initial.frozen = {};
    cell.initial.frozen[2] = reference::collection::population(2, .01, 100, 2);
    const auto initial_reference = reference::evaluate(cell, 1);
    const double duration = static_cast<double>(2 * cell.initial.cloud_mass_per_kg
        / initial_reference.collection.channels[2].raw_mass);
    ColdCloudDiffusionProducer producer;
    std::string error;
    const bool accepted = produce_exact_cold_cloud_diffusion_admission(parameters(), cell, duration, producer, error);
    INFO(error);
    REQUIRE(accepted);
    REQUIRE(producer.thermodynamics().relaxation_number < 1);
    const auto& collection = producer.collection_producer().channels()[2];
    REQUIRE(collection.donor_admission_factor > 0);
    REQUIRE(collection.donor_admission_factor < 1);
    REQUIRE(collection.raw_mass_increment_per_kg > collection.admitted_mass_increment_per_kg);
    const auto expected = reference::evaluate(cell, duration);
    near(collection.admitted_mass_increment_per_kg, expected.collection.channels[2].mass);
    near(producer.channels()[2].riming_mass_increment_per_kg,
        -expected.feedback * expected.collection.channels[2].mass);
    CHECK(std::abs(producer.channels()[2].riming_mass_increment_per_kg)
        < .75 * expected.feedback * collection.raw_mass_increment_per_kg);
}

TEST_CASE("coupled diffusion empty populations and unrepresentable vapor retain atomic storage",
          "[physics][milbrandt][triple_cold_cloud_diffusion]")
{
    auto cell = source();
    cell.initial = {};
    cell.initial.cloud_number_per_kg = 7;
    cell.vapor_mass_per_kg = 0;
    const auto empty = evaluate(cell, 100);
    CHECK(empty.temperature_increment_k == 0);
    CHECK(empty.vapor_mass_per_kg == 0);
    CHECK(empty.unrounded.cloud_number_per_kg == 7);
    CHECK(empty.producer.thermodynamics().relaxation_number == 0);
    for (const auto& channel : empty.producer.channels())
    {
        CHECK(channel.admitted_mass_increment_per_kg == 0);
        CHECK(channel.admitted_number_increment_per_kg == 0);
    }
    cell.vapor_mass_per_kg = std::numeric_limits<double>::denorm_min();
    auto unchanged = empty;
    std::string error;
    ColdCloudDiffusionFailure failure{};
    REQUIRE_FALSE(evaluate_exact_triple_cold_cloud_diffusion_cell(
        parameters(), cell, 1e-3, unchanged, error, 3, &failure));
    CHECK(failure == ColdCloudDiffusionFailure::IntervalRejected);
    CHECK(unchanged.vapor_mass_per_kg == empty.vapor_mass_per_kg);
    CHECK(unchanged.unrounded.cloud_number_per_kg == 7);
}
