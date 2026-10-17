#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/triple_warm_subcycle.hpp"
#include "microphysics/schemes/milbrandt/triple_warm_cloud.hpp"
#include "support/milbrandt_cloud_adjustment_reference.hpp"
#include "support/milbrandt_dynamic_warm_reference.hpp"
#include "support/milbrandt_subcycle_reference.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace
{

using namespace tmv;
using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_subcycle_reference;
namespace dynamic_reference = milbrandt_dynamic_warm_reference;

const WarmRainParameters parameters = warm_rain_parameters(0, 523.598775598, 3, 149.1, .5);

DoubleWarmRainCell source_cell(const RainTripleMomentState& rain, double density,
                               double cloud = 0, double cloud_number = 0,
                               double humidity = 1.1)
{
    const double saturation = thermodynamics::saturation_mixing_ratio_water(290.125, 90000);
    return {290.125, 90000, density, humidity * saturation, cloud, rain.mass_per_kg,
            cloud_number, rain.number_per_kg};
}

std::array<double, 8> source_values(const DoubleWarmRainCell& source)
{
    return {source.temperature, source.pressure, source.density, source.vapor,
            source.cloud, source.rain, source.cloud_number, source.rain_number};
}

std::array<double, 16> endpoint_values(const WarmRainDoubleEndpoints& endpoint)
{
    return {endpoint.vapor_per_kg, endpoint.cloud_per_kg, endpoint.rain_per_kg,
            endpoint.cloud_number_per_kg, endpoint.rain_number_per_kg,
            endpoint.temperature_increment_k, endpoint.cloud_donor_budget_per_kg,
            endpoint.evaporation.mass_before_per_kg, endpoint.evaporation.number_before_per_m3,
            endpoint.evaporation.admitted_mass_per_kg, endpoint.evaporation.mass_after_per_kg,
            endpoint.evaporation.number_after_per_m3, endpoint.reclassification.rain_mass_before_per_kg,
            endpoint.reclassification.rain_number_before_per_m3,
            endpoint.reclassification.cloud_mass_before_per_kg,
            endpoint.reclassification.cloud_number_before_per_m3};
}

std::array<double, 3> rain_values(const ExactRainTripleMomentState& rain)
{
    return {rain.mass_per_kg, rain.number_per_kg, rain.reflectivity_per_kg};
}

void compare_unchanged(const WarmSubcycleCellResult& actual,
                       const WarmSubcycleCellResult& expected)
{
    CHECK(endpoint_values(actual.endpoints) == endpoint_values(expected.endpoints));
    CHECK(rain_values(actual.rain) == rain_values(expected.rain));
    CHECK(actual.rounded_rain.mass_per_kg == expected.rounded_rain.mass_per_kg);
    CHECK(actual.rounded_rain.number_per_kg == expected.rounded_rain.number_per_kg);
    CHECK(actual.rounded_rain.reflectivity_per_kg == expected.rounded_rain.reflectivity_per_kg);
    CHECK(std::memcmp(&actual.rates, &expected.rates, sizeof(actual.rates)) == 0);
    CHECK(actual.producer.accepted_steps() == expected.producer.accepted_steps());
    CHECK(actual.producer.rejected_trials() == expected.producer.rejected_trials());
    CHECK(actual.producer.completed_duration_s() == expected.producer.completed_duration_s());
    CHECK(actual.producer.trials() == expected.producer.trials());
    CHECK(actual.producer.method_version() == expected.producer.method_version());
    CHECK(rain_values(actual.producer.pre_regularized_rain()) == rain_values(expected.producer.pre_regularized_rain()));
    CHECK(rain_values(actual.producer.post_regularization_before()) == rain_values(expected.producer.post_regularization_before()));
    CHECK(rain_values(actual.producer.final_rain()) == rain_values(expected.producer.final_rain()));
    CHECK(source_values(actual.producer.source_cell()) == source_values(expected.producer.source_cell()));
    CHECK(actual.producer.original_rain().mass_per_kg == expected.producer.original_rain().mass_per_kg);
    CHECK(actual.producer.original_rain().number_per_kg == expected.producer.original_rain().number_per_kg);
    CHECK(actual.producer.original_rain().reflectivity_per_kg == expected.producer.original_rain().reflectivity_per_kg);
    REQUIRE(actual.producer.steps().size() == expected.producer.steps().size());
    for (std::size_t index = 0; index < actual.producer.steps().size(); ++index)
    {
        const auto& a = actual.producer.steps()[index];
        const auto& e = expected.producer.steps()[index];
        CHECK(a.duration_s == e.duration_s);
        CHECK(a.error_norm == e.error_norm);
        CHECK(source_values(a.source_cell) == source_values(e.source_cell));
        CHECK(rain_values(a.initial_rain) == rain_values(e.initial_rain));
        CHECK(rain_values(a.final_rain) == rain_values(e.final_rain));
        CHECK(endpoint_values(a.endpoints) == endpoint_values(e.endpoints));
        CHECK(a.endpoints.cloud_donor_exhausted == e.endpoints.cloud_donor_exhausted);
        CHECK(a.endpoints.evaporation.present == e.endpoints.evaporation.present);
        CHECK(a.endpoints.evaporation.donor_exhausted == e.endpoints.evaporation.donor_exhausted);
        CHECK(a.endpoints.reclassification.present == e.endpoints.reclassification.present);
        for (std::size_t probe = 0; probe < a.euler_probes.size(); ++probe)
        {
            const auto& actual_probe = a.euler_probes[probe];
            const auto& expected_probe = e.euler_probes[probe];
            CHECK(actual_probe.duration_s == expected_probe.duration_s);
            CHECK(source_values(actual_probe.source_cell) == source_values(expected_probe.source_cell));
            CHECK(rain_values(actual_probe.initial_rain) == rain_values(expected_probe.initial_rain));
            CHECK(rain_values(actual_probe.final_rain) == rain_values(expected_probe.final_rain));
            CHECK(endpoint_values(actual_probe.endpoints) == endpoint_values(expected_probe.endpoints));
            CHECK(actual_probe.endpoints.evaporation.present == expected_probe.endpoints.evaporation.present);
            CHECK(actual_probe.endpoints.evaporation.donor_exhausted == expected_probe.endpoints.evaporation.donor_exhausted);
            CHECK(actual_probe.endpoints.reclassification.present == expected_probe.endpoints.reclassification.present);
            CHECK(actual_probe.endpoints.cloud_donor_exhausted == expected_probe.endpoints.cloud_donor_exhausted);
        }
    }
}

void check_trace(const DoubleWarmRainCell& original, double duration,
                 const WarmSubcycleCellResult& result)
{
    const auto& producer = result.producer;
    REQUIRE_FALSE(producer.steps().empty());
    CHECK(producer.accepted_steps() == producer.steps().size());
    CHECK(producer.completed_duration_s() == duration);
    CHECK(producer.method_version() == 2);
    CHECK(source_values(producer.source_cell()) == source_values(original));
    CHECK(result.rounded_rain.mass_per_kg == static_cast<float>(result.rain.mass_per_kg));
    CHECK(result.rounded_rain.number_per_kg == static_cast<float>(result.rain.number_per_kg));
    CHECK(result.rounded_rain.reflectivity_per_kg == static_cast<float>(result.rain.reflectivity_per_kg));
    CHECK(result.endpoints.rain_per_kg == result.rain.mass_per_kg);
    CHECK(result.endpoints.rain_number_per_kg == result.rain.number_per_kg);
    CHECK_FALSE(result.endpoints.evaporation.present);
    CHECK_FALSE(result.endpoints.reclassification.present);
    CHECK_FALSE(result.endpoints.cloud_donor_exhausted);
    CHECK(rain_values(producer.pre_regularized_rain()) == rain_values(producer.steps().front().initial_rain));
    CHECK(rain_values(producer.post_regularization_before()) == rain_values(producer.steps().back().final_rain));
    CHECK(rain_values(producer.final_rain()) == rain_values(result.rain));
    long double elapsed = 0;
    long double heat = 0;
    const double initial_water = original.vapor + original.cloud + original.rain;
    for (std::size_t index = 0; index < producer.steps().size(); ++index)
    {
        const auto& step = producer.steps()[index];
        INFO("accepted FP64 step " << index << " of " << producer.steps().size());
        REQUIRE(step.duration_s > 0);
        CHECK(std::isfinite(step.error_norm));
        CHECK(step.error_norm >= 0);
        CHECK(step.error_norm <= 1);
        CHECK(step.source_cell.pressure == original.pressure);
        CHECK(step.source_cell.density == original.density);
        CHECK(step.source_cell.rain == step.initial_rain.mass_per_kg);
        CHECK(step.source_cell.rain_number == step.initial_rain.number_per_kg);
        CHECK(step.endpoints.rain_per_kg == step.final_rain.mass_per_kg);
        CHECK(step.endpoints.rain_number_per_kg == step.final_rain.number_per_kg);
        CHECK_FALSE(step.endpoints.evaporation.present);
        CHECK_FALSE(step.endpoints.reclassification.present);
        CHECK_FALSE(step.endpoints.cloud_donor_exhausted);
        CHECK(step.euler_probes[0].duration_s == step.duration_s);
        CHECK(step.euler_probes[1].duration_s == step.duration_s / 2);
        CHECK(step.euler_probes[2].duration_s == step.duration_s / 2);
        CHECK(source_values(step.euler_probes[0].source_cell) == source_values(step.source_cell));
        CHECK(source_values(step.euler_probes[1].source_cell) == source_values(step.source_cell));
        CHECK(rain_values(step.euler_probes[0].initial_rain) == rain_values(step.initial_rain));
        CHECK(rain_values(step.euler_probes[1].initial_rain) == rain_values(step.initial_rain));
        CHECK(rain_values(step.euler_probes[2].initial_rain) == rain_values(step.euler_probes[1].final_rain));
        CHECK(step.euler_probes[2].source_cell.vapor == step.euler_probes[1].endpoints.vapor_per_kg);
        CHECK(step.euler_probes[2].source_cell.cloud == step.euler_probes[1].endpoints.cloud_per_kg);
        CHECK(step.euler_probes[2].source_cell.cloud_number == step.euler_probes[1].endpoints.cloud_number_per_kg);
        CHECK(step.euler_probes[2].source_cell.temperature == Approx(step.source_cell.temperature +
            step.euler_probes[1].endpoints.temperature_increment_k).epsilon(0)
                .margin(2 * std::numeric_limits<double>::epsilon() * original.temperature));
        CHECK(step.endpoints.temperature_increment_k ==
            (step.euler_probes[1].endpoints.temperature_increment_k +
             step.euler_probes[2].endpoints.temperature_increment_k) * 2 -
              step.euler_probes[0].endpoints.temperature_increment_k);
        const auto corrected = [](double initial, double coarse, double fine)
        {
            return initial + (fine - initial) + (fine - coarse);
        };
        const auto& coarse = step.euler_probes[0].endpoints;
        const auto& fine = step.euler_probes[2].endpoints;
        CHECK(step.endpoints.vapor_per_kg == corrected(step.source_cell.vapor,
            coarse.vapor_per_kg, fine.vapor_per_kg));
        CHECK(step.endpoints.cloud_per_kg == corrected(step.source_cell.cloud,
            coarse.cloud_per_kg, fine.cloud_per_kg));
        CHECK(step.endpoints.rain_per_kg == corrected(step.source_cell.rain,
            coarse.rain_per_kg, fine.rain_per_kg));
        CHECK(step.endpoints.cloud_number_per_kg == corrected(step.source_cell.cloud_number,
            coarse.cloud_number_per_kg, fine.cloud_number_per_kg));
        CHECK(step.endpoints.rain_number_per_kg == corrected(step.source_cell.rain_number,
            coarse.rain_number_per_kg, fine.rain_number_per_kg));
        CHECK(step.final_rain.reflectivity_per_kg == corrected(step.initial_rain.reflectivity_per_kg,
            step.euler_probes[0].final_rain.reflectivity_per_kg,
            step.euler_probes[2].final_rain.reflectivity_per_kg));
        for (const auto& probe : step.euler_probes)
        {
            CHECK(probe.source_cell.rain == probe.initial_rain.mass_per_kg);
            CHECK(probe.source_cell.rain_number == probe.initial_rain.number_per_kg);
            CHECK(probe.endpoints.rain_per_kg == probe.final_rain.mass_per_kg);
            CHECK(probe.endpoints.rain_number_per_kg == probe.final_rain.number_per_kg);
            CHECK(probe.endpoints.vapor_per_kg + probe.endpoints.cloud_per_kg + probe.endpoints.rain_per_kg
                == Approx(probe.source_cell.vapor + probe.source_cell.cloud + probe.source_cell.rain)
                    .epsilon(3e-14).margin(1e-30));
            CHECK(probe.endpoints.temperature_increment_k == Approx(-2.5e6 / 1004 *
                (probe.endpoints.vapor_per_kg - probe.source_cell.vapor)).epsilon(1e-9).margin(2e-12));
        }
        CHECK(step.endpoints.vapor_per_kg + step.endpoints.cloud_per_kg + step.endpoints.rain_per_kg
            == Approx(step.source_cell.vapor + step.source_cell.cloud + step.source_cell.rain)
                .epsilon(3e-14).margin(1e-30));
        CHECK(step.endpoints.temperature_increment_k == Approx(-2.5e6 / 1004 *
            (step.endpoints.vapor_per_kg - step.source_cell.vapor)).epsilon(1e-9).margin(2e-12));
        if (index)
        {
            const auto& previous = producer.steps()[index - 1];
            CHECK(step.source_cell.temperature == Approx(static_cast<double>(
                static_cast<long double>(original.temperature) + heat)).epsilon(0)
                    .margin(2 * std::numeric_limits<double>::epsilon() * original.temperature));
            CHECK(step.source_cell.vapor == previous.endpoints.vapor_per_kg);
            CHECK(step.source_cell.cloud == previous.endpoints.cloud_per_kg);
            CHECK(step.source_cell.cloud_number == previous.endpoints.cloud_number_per_kg);
            CHECK(rain_values(step.initial_rain) == rain_values(previous.final_rain));
        }
        elapsed += step.duration_s;
        heat += step.endpoints.temperature_increment_k;
    }
    CHECK(static_cast<double>(elapsed) == Approx(duration).epsilon(3e-14));
    CHECK(result.endpoints.temperature_increment_k == Approx(static_cast<double>(heat))
        .epsilon(3e-14).margin(2e-12));
    CHECK(result.endpoints.vapor_per_kg + result.endpoints.cloud_per_kg + result.endpoints.rain_per_kg
        == Approx(initial_water).epsilon(3e-13).margin(1e-30));
    CHECK(result.rates.rain == static_cast<float>((result.endpoints.rain_per_kg - original.rain) / duration));
    CHECK(result.rates.vapor == static_cast<float>((result.endpoints.vapor_per_kg - original.vapor) / duration));
    CHECK(result.rates.cloud == static_cast<float>((result.endpoints.cloud_per_kg - original.cloud) / duration));
    CHECK(result.rates.temperature == static_cast<float>(result.endpoints.temperature_increment_k / duration));
}

long double relative_error(double actual, long double expected)
{
    return std::abs(static_cast<long double>(actual) / expected - 1);
}

std::array<double, 7> cloud_source_values(const TripleCloudAdjustmentCell& source)
{
    return {source.temperature_k, source.pressure_pa, source.dry_density_kg_m3,
            source.vapor_per_kg, source.cloud_per_kg, source.cloud_number_per_kg,
            source.vertical_velocity_ms};
}

std::array<double, 7> cloud_endpoint_values(const TripleCloudAdjustmentEndpoints& endpoint)
{
    return {endpoint.vapor_per_kg, endpoint.cloud_per_kg, endpoint.cloud_number_per_kg,
            endpoint.temperature_increment_k, endpoint.temperature_after_k,
            endpoint.admitted_cloud_increment_per_kg, endpoint.activated_number_per_m3};
}

void compare_composed_unchanged(const ExactTripleSubcycledWarmCloudCellResult& actual,
                               const ExactTripleSubcycledWarmCloudCellResult& expected)
{
    compare_unchanged(actual.warm, expected.warm);
    CHECK(cloud_source_values(actual.cloud.source_cell()) == cloud_source_values(expected.cloud.source_cell()));
    CHECK(cloud_endpoint_values(actual.cloud.endpoints()) == cloud_endpoint_values(expected.cloud.endpoints()));
    CHECK(actual.cloud.endpoints().cloud_donor_exhausted == expected.cloud.endpoints().cloud_donor_exhausted);
    CHECK(actual.cloud.duration_s() == expected.cloud.duration_s());
    CHECK(endpoint_values(actual.endpoints) == endpoint_values(expected.endpoints));
    CHECK(rain_values(actual.rain) == rain_values(expected.rain));
    CHECK(actual.rounded_rain.mass_per_kg == expected.rounded_rain.mass_per_kg);
    CHECK(actual.rounded_rain.number_per_kg == expected.rounded_rain.number_per_kg);
    CHECK(actual.rounded_rain.reflectivity_per_kg == expected.rounded_rain.reflectivity_per_kg);
    CHECK(actual.temperature_increment_k == expected.temperature_increment_k);
}

void check_breakup_error_norm(const WarmSubcycleCellResult& result, double density)
{
    const auto& steps = result.producer.steps();
    const auto& options = result.producer.options();
    REQUIRE(steps.size() % 2 == 0);
    for (std::size_t index = 0; index < steps.size(); index += 2)
    {
        const auto& first = steps[index];
        const auto& second = steps[index + 1];
        INFO("independent dimensioned step-doubling pair " << index / 2);
        CHECK(first.duration_s == second.duration_s);
        CHECK(first.error_norm == second.error_norm);
        const reference::BreakupPopulation initial{first.initial_rain.mass_per_kg,
            first.initial_rain.number_per_kg, first.initial_rain.reflectivity_per_kg};
        const auto coarse = reference::second_order_breakup(initial, density, 2 * first.duration_s);
        const auto half = reference::second_order_breakup(initial, density, first.duration_s);
        const auto fine = reference::second_order_breakup(half, density, first.duration_s);
        CHECK(first.final_rain.number_per_kg == Approx(static_cast<double>(half.n)).epsilon(3e-12));
        CHECK(first.final_rain.reflectivity_per_kg == Approx(static_cast<double>(half.z)).epsilon(3e-12));
        CHECK(second.final_rain.number_per_kg == Approx(static_cast<double>(fine.n)).epsilon(3e-12));
        CHECK(second.final_rain.reflectivity_per_kg == Approx(static_cast<double>(fine.z)).epsilon(3e-12));
        const long double number_scale = options.absolute_number_tolerance +
            options.relative_tolerance * std::max({std::abs(initial.n), std::abs(coarse.n), std::abs(fine.n)});
        const long double reflectivity_scale = options.absolute_reflectivity_tolerance +
            options.relative_tolerance * std::max({std::abs(initial.z), std::abs(coarse.z), std::abs(fine.z)});
        const long double expected = std::max(std::abs(fine.n - coarse.n) / (3 * number_scale),
                                              std::abs(fine.z - coarse.z) / (3 * reflectivity_scale));
        CHECK(first.error_norm == Approx(static_cast<double>(expected)).epsilon(3e-6).margin(3e-8));
        for (const auto* accepted : {&first, &second})
        {
            const reference::BreakupPopulation accepted_source{accepted->initial_rain.mass_per_kg,
                accepted->initial_rain.number_per_kg, accepted->initial_rain.reflectivity_per_kg};
            const auto full_probe = reference::euler_breakup(accepted_source, density, accepted->duration_s);
            const auto first_probe = reference::euler_breakup(accepted_source, density, accepted->duration_s / 2);
            const auto second_probe = reference::euler_breakup(first_probe, density, accepted->duration_s / 2);
            const std::array<reference::BreakupPopulation, 3> expected_probes{full_probe, first_probe, second_probe};
            for (std::size_t probe_index = 0; probe_index < expected_probes.size(); ++probe_index)
            {
                const auto& actual = accepted->euler_probes[probe_index].final_rain;
                const auto& independent = expected_probes[probe_index];
                CHECK(actual.mass_per_kg == Approx(static_cast<double>(independent.q)).epsilon(3e-12));
                CHECK(actual.number_per_kg == Approx(static_cast<double>(independent.n)).epsilon(3e-12));
                CHECK(actual.reflectivity_per_kg == Approx(static_cast<double>(independent.z)).epsilon(3e-12));
            }
            const auto corrected = reference::second_order_breakup(accepted_source, density, accepted->duration_s);
            CHECK(accepted->final_rain.number_per_kg == Approx(static_cast<double>(corrected.n)).epsilon(3e-12));
            CHECK(accepted->final_rain.reflectivity_per_kg == Approx(static_cast<double>(corrected.z)).epsilon(3e-12));
        }
    }
}

}

TEST_CASE("subcycled warm breakup resolves both retained heavy intervals against independent RK4",
          "[physics][milbrandt][triple_warm_subcycle]")
{
    bool saw_unrounded_number = false;
    for (double density : {.65, 1.3})
    {
        for (long double shape : {0.0L, 20.0L})
        {
            const auto initial = reference::rain(.003f, 10.0f, shape);
            const auto source = source_cell(initial, density);
            const auto expected = reference::integrate_breakup(initial, density, .3L);
            const auto refined = reference::integrate_breakup(initial, density, .3L, 65536);
            CHECK(expected.n == Approx(static_cast<double>(refined.n)).epsilon(3e-13));
            CHECK(expected.z == Approx(static_cast<double>(refined.z)).epsilon(3e-13));
            WarmSubcycleCellResult result;
            std::string error;
            if (density == 1.3 || shape == 20)
            {
                ExactTripleDynamicWarmCellResult unsupported;
                REQUIRE_FALSE(evaluate_exact_triple_dynamic_warm_cell(parameters,
                    source, .3, initial, unsupported, error));
                CHECK_FALSE(error.empty());
            }
            const bool accepted = evaluate_exact_triple_warm_subcycled_cell(
                parameters, source, .3, initial, result, error);
            INFO("rho " << density << ", represented shape " << shape << ": " << error);
            REQUIRE(accepted);
            CHECK(result.producer.rejected_trials() > 0);
            CHECK(result.producer.accepted_steps() > 2);
            check_trace(source, .3, result);
            check_breakup_error_norm(result, density);
            CHECK(result.rain.mass_per_kg == source.rain);
            CHECK(relative_error(result.rain.number_per_kg, expected.n) < 3e-3L);
            CHECK(relative_error(result.rain.reflectivity_per_kg, expected.z) < 3e-3L);
            CHECK(relative_error(result.rain.number_per_kg * result.rain.reflectivity_per_kg,
                static_cast<long double>(initial.number_per_kg) * initial.reflectivity_per_kg) < 3e-3L);
            const auto regularized = reference::regularize(reference::population(initial));
            const auto& first = result.producer.steps().front();
            CHECK(first.initial_rain.number_per_kg == Approx(static_cast<double>(regularized.n)).epsilon(3e-14));
            CHECK(first.initial_rain.reflectivity_per_kg == Approx(static_cast<double>(regularized.z)).epsilon(3e-14));
            const auto& last = result.producer.steps().back();
            const auto post = reference::regularize({last.final_rain.mass_per_kg,
                last.final_rain.number_per_kg, last.final_rain.reflectivity_per_kg});
            CHECK(result.rain.number_per_kg == Approx(static_cast<double>(post.n)).epsilon(3e-14));
            CHECK(result.rain.reflectivity_per_kg == Approx(static_cast<double>(post.z)).epsilon(3e-14));
            for (const auto& step : result.producer.steps())
            {
                saw_unrounded_number |= step.final_rain.number_per_kg !=
                    static_cast<double>(static_cast<float>(step.final_rain.number_per_kg));
            }
            WarmSubcycleCellResult replay;
            REQUIRE(apply_exact_triple_warm_subcycled_cell(parameters, source, .3,
                initial, result.producer, replay, error));
            compare_unchanged(replay, result);
        }
    }
    CHECK(saw_unrounded_number);
}

TEST_CASE("subcycled warm tolerance refinement converges without repeating numerical number policy",
          "[physics][milbrandt][triple_warm_subcycle]")
{
    const auto initial = reference::rain(.003f, 10.0f, 20);
    const auto source = source_cell(initial, 1.3);
    const auto expected = reference::integrate_breakup(initial, source.density, .3L, 65536);
    WarmSubcycleOptions loose;
    loose.relative_tolerance = 1e-5;
    WarmSubcycleOptions fine;
    fine.relative_tolerance = 1e-7;
    WarmSubcycleCellResult coarse;
    WarmSubcycleCellResult refined;
    std::string error;
    REQUIRE(evaluate_exact_triple_warm_subcycled_cell(parameters, source, .3,
        initial, coarse, error, loose));
    REQUIRE(evaluate_exact_triple_warm_subcycled_cell(parameters, source, .3,
        initial, refined, error, fine));
    CHECK(refined.producer.accepted_steps() > coarse.producer.accepted_steps());
    CHECK(relative_error(refined.rain.number_per_kg, expected.n) <
        relative_error(coarse.rain.number_per_kg, expected.n) * .5L);
    CHECK(relative_error(refined.rain.reflectivity_per_kg, expected.z) <
        relative_error(coarse.rain.reflectivity_per_kg, expected.z) * .5L);
    CHECK(relative_error(refined.rain.number_per_kg, expected.n) < 1e-3L);
    CHECK(relative_error(refined.rain.reflectivity_per_kg, expected.z) < 1e-3L);
    CHECK(coarse.producer.options().relative_tolerance == loose.relative_tolerance);
    CHECK(refined.producer.options().relative_tolerance == fine.relative_tolerance);
    check_trace(source, .3, refined);
    WarmSubcycleCellResult replay;
    REQUIRE(apply_exact_triple_warm_subcycled_cell(parameters, source, .3,
        initial, refined.producer, replay, error, fine));
    compare_unchanged(replay, refined);
    auto rejected = refined;
    REQUIRE_FALSE(apply_exact_triple_warm_subcycled_cell(parameters, source, .3,
        initial, refined.producer, rejected, error));
    compare_unchanged(rejected, refined);
}

TEST_CASE("subcycled warm sources preserve zero tiny seeded reclassified and evaporating owners",
          "[physics][milbrandt][triple_warm_subcycle]")
{
    struct Case
    {
        RainTripleMomentState rain;
        double cloud;
        double cloud_number;
        double humidity;
        double duration;
    };
    const std::array<Case, 5> cases{{
        {{}, 0, 0, 1.1, .3},
        {dynamic_reference::source_rain(4e-13f, 90e-6, 20), 0, 0, .5, .3},
        {{}, .01, 1e5, .8, .05},
        {dynamic_reference::source_rain(1e-5f, 40e-6, 20), 1e-6, 1e8, 1.1, .1},
        {dynamic_reference::source_rain(2e-4f, 120e-6, 20), 0, 0, .2, .1}
    }};
    for (std::size_t index = 0; index < cases.size(); ++index)
    {
        const auto& fixture = cases[index];
        const auto source = source_cell(fixture.rain, .65, fixture.cloud,
            fixture.cloud_number, fixture.humidity);
        WarmSubcycleCellResult result;
        std::string error;
        const bool accepted = evaluate_exact_triple_warm_subcycled_cell(parameters,
            source, fixture.duration, fixture.rain, result, error);
        INFO("owner fixture " << index << ": " << error);
        REQUIRE(accepted);
        check_trace(source, fixture.duration, result);
        if (index == 0 || index == 1)
        {
            CHECK(result.rain.mass_per_kg == fixture.rain.mass_per_kg);
            CHECK(result.rain.number_per_kg == fixture.rain.number_per_kg);
            CHECK(result.rain.reflectivity_per_kg == fixture.rain.reflectivity_per_kg);
            CHECK(result.endpoints.temperature_increment_k == 0);
        }
        if (index == 2)
        {
            CHECK(result.rain.mass_per_kg > 0);
            CHECK(result.rain.number_per_kg > 0);
            CHECK(result.rain.reflectivity_per_kg > 0);
            CHECK(result.endpoints.cloud_per_kg < source.cloud);
        }
        if (index == 3)
        {
            CHECK(result.rain.mass_per_kg == 0);
            CHECK(result.rain.number_per_kg == 0);
            CHECK(result.rain.reflectivity_per_kg == 0);
            CHECK(result.endpoints.cloud_per_kg == source.cloud + source.rain);
            CHECK(result.endpoints.temperature_increment_k == 0);
            bool saw_transfer = false;
            for (const auto& step : result.producer.steps())
            {
                for (const auto& probe : step.euler_probes)
                {
                    saw_transfer |= probe.endpoints.reclassification.present;
                }
            }
            CHECK(saw_transfer);
        }
        if (index == 4)
        {
            CHECK(result.rain.mass_per_kg < source.rain);
            CHECK(result.endpoints.vapor_per_kg > source.vapor);
            CHECK(result.endpoints.temperature_increment_k < 0);
            bool saw_evaporation = false;
            bool saw_unrounded_temperature = false;
            for (const auto& step : result.producer.steps())
            {
                for (const auto& probe : step.euler_probes)
                {
                    saw_evaporation |= probe.endpoints.evaporation.present;
                }
                saw_unrounded_temperature |= step.source_cell.temperature !=
                    static_cast<double>(static_cast<float>(step.source_cell.temperature));
            }
            CHECK(saw_evaporation);
            CHECK(saw_unrounded_temperature);
        }
    }
}

TEST_CASE("subcycled warm invalid source and immutable trace rejection are failure atomic",
          "[physics][milbrandt][triple_warm_subcycle]")
{
    const auto initial = reference::rain(.003f, 10.0f, 20);
    const auto source = source_cell(initial, .65);
    WarmSubcycleCellResult good;
    std::string error;
    REQUIRE(evaluate_exact_triple_warm_subcycled_cell(parameters, source, .3, initial, good, error));
    for (int mutation = 0; mutation < 14; ++mutation)
    {
        auto changed_source = source;
        auto changed_rain = initial;
        auto changed_parameters = parameters;
        double duration = .3;
        switch (mutation)
        {
            case 0: changed_source.temperature = std::nextafter(source.temperature, 400.0); break;
            case 1: changed_source.pressure = std::nextafter(source.pressure, 2e5); break;
            case 2: changed_source.density = std::nextafter(source.density, 2.0); break;
            case 3: changed_source.vapor = std::nextafter(source.vapor, 1.0); break;
            case 4: changed_source.cloud = std::nextafter(source.cloud, 1.0); break;
            case 5: changed_source.cloud_number = std::nextafter(source.cloud_number, 1e9); break;
            case 6: changed_source.rain = std::nextafter(source.rain, 1.0); break;
            case 7: changed_source.rain_number = std::nextafter(source.rain_number, 1e9); break;
            case 8: changed_rain.reflectivity_per_kg = std::nextafter(initial.reflectivity_per_kg, 1.0f); break;
            case 9: duration = std::nextafter(duration, 1.0); break;
            case 10: changed_parameters.gamma_ventilation = std::nextafter(parameters.gamma_ventilation, 3.0); break;
            case 11: changed_source.density = 0; break;
            case 12: changed_source.vapor = std::numeric_limits<double>::quiet_NaN(); break;
            case 13: changed_rain.reflectivity_per_kg = -1; break;
        }
        auto output = good;
        INFO("bound context mutation " << mutation);
        REQUIRE_FALSE(apply_exact_triple_warm_subcycled_cell(changed_parameters, changed_source,
            duration, changed_rain, good.producer, output, error));
        CHECK_FALSE(error.empty());
        compare_unchanged(output, good);
    }
    for (int mutation = 0; mutation < 17; ++mutation)
    {
        auto producer = good.producer;
        auto& first = const_cast<WarmSubcycleAcceptedStep&>(producer.steps().front());
        switch (mutation)
        {
            case 0: first.duration_s = std::nextafter(first.duration_s, 1.0); break;
            case 1: first.error_norm = std::nextafter(first.error_norm, 2.0); break;
            case 2: first.source_cell.temperature = std::nextafter(first.source_cell.temperature, 400.0); break;
            case 3: first.initial_rain.reflectivity_per_kg *= 2; break;
            case 4: first.final_rain.reflectivity_per_kg *= 2; break;
            case 5: first.endpoints.rain_number_per_kg *= 2; break;
            case 6: first.endpoints.vapor_per_kg = std::nextafter(first.endpoints.vapor_per_kg, 1.0); break;
            case 7: first.error_norm = std::numeric_limits<double>::quiet_NaN(); break;
            case 8: const_cast<WarmSubcycleOptions&>(producer.options()).relative_tolerance *= 2; break;
            case 9: const_cast<DoubleWarmRainCell&>(producer.source_cell()).pressure *= 2; break;
            case 10: const_cast<std::vector<WarmSubcycleAcceptedStep>&>(producer.steps()).pop_back(); break;
            case 11: const_cast<ExactRainTripleMomentState&>(producer.pre_regularized_rain()).reflectivity_per_kg *= 2; break;
            case 12: const_cast<ExactRainTripleMomentState&>(producer.post_regularization_before()).reflectivity_per_kg *= 2; break;
            case 13: const_cast<ExactRainTripleMomentState&>(producer.final_rain()).reflectivity_per_kg *= 2; break;
            case 14: first.euler_probes[0].duration_s = std::nextafter(first.euler_probes[0].duration_s, 1.0); break;
            case 15: first.euler_probes[1].final_rain.reflectivity_per_kg *= 2; break;
            case 16: first.euler_probes[2].source_cell.vapor = std::nextafter(first.euler_probes[2].source_cell.vapor, 1.0); break;
        }
        auto output = good;
        INFO("immutable trace mutation " << mutation);
        REQUIRE_FALSE(apply_exact_triple_warm_subcycled_cell(parameters, source,
            .3, initial, producer, output, error));
        compare_unchanged(output, good);
    }
    for (double duration : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()})
    {
        auto output = good;
        REQUIRE_FALSE(evaluate_exact_triple_warm_subcycled_cell(parameters, source,
            duration, initial, output, error));
        compare_unchanged(output, good);
    }
}

TEST_CASE("subcycled warm controller resources and invalid tolerances publish no partial trace",
          "[physics][milbrandt][triple_warm_subcycle]")
{
    const auto initial = reference::rain(.003f, 10.0f, 20);
    const auto source = source_cell(initial, 1.3);
    WarmSubcycleCellResult good;
    std::string error;
    REQUIRE(evaluate_exact_triple_warm_subcycled_cell(parameters, source, .3, initial, good, error));
    for (int resource = 0; resource < 4; ++resource)
    {
        WarmSubcycleOptions options;
        switch (resource)
        {
            case 0: options.max_accepted_steps = 1; break;
            case 1: options.max_trials = 1; break;
            case 2: options.max_halvings = 0; break;
            case 3: options.observer_capacity = 0; break;
        }
        auto output = good;
        INFO("bounded resource " << resource);
        REQUIRE_FALSE(evaluate_exact_triple_warm_subcycled_cell(parameters, source,
            .3, initial, output, error, options));
        CHECK_FALSE(error.empty());
        compare_unchanged(output, good);
    }
    for (int invalid = 0; invalid < 12; ++invalid)
    {
        WarmSubcycleOptions options;
        switch (invalid)
        {
            case 0: options.relative_tolerance = 0; break;
            case 1: options.relative_tolerance = -1; break;
            case 2: options.relative_tolerance = std::numeric_limits<double>::quiet_NaN(); break;
            case 3: options.relative_tolerance = std::numeric_limits<double>::infinity(); break;
            case 4: options.absolute_mass_tolerance = -1; break;
            case 5: options.absolute_number_tolerance = -1; break;
            case 6: options.absolute_reflectivity_tolerance = -1; break;
            case 7: options.absolute_temperature_tolerance = -1; break;
            case 8: options.absolute_mass_tolerance = std::numeric_limits<double>::quiet_NaN(); break;
            case 9: options.absolute_number_tolerance = std::numeric_limits<double>::infinity(); break;
            case 10: options.max_accepted_steps = 0; break;
            case 11: options.max_trials = 0; break;
        }
        auto output = good;
        INFO("invalid controller option " << invalid);
        REQUIRE_FALSE(evaluate_exact_triple_warm_subcycled_cell(parameters, source,
            .3, initial, output, error, options));
        CHECK_FALSE(error.empty());
        compare_unchanged(output, good);
    }
    for (int invalid = 0; invalid < 10; ++invalid)
    {
        auto altered = source;
        auto rain = initial;
        switch (invalid)
        {
            case 0: altered.temperature = 0; break;
            case 1: altered.pressure = 0; break;
            case 2: altered.density = -1; break;
            case 3: altered.cloud = -1; break;
            case 4: altered.cloud_number = -1; break;
            case 5: altered.rain = -1; break;
            case 6: altered.rain_number = -1; break;
            case 7: altered.temperature = std::numeric_limits<double>::infinity(); break;
            case 8: rain.reflectivity_per_kg = std::numeric_limits<float>::quiet_NaN(); break;
            case 9: rain.number_per_kg = std::nextafter(rain.number_per_kg, 100.0f); break;
        }
        auto output = good;
        INFO("invalid source " << invalid);
        REQUIRE_FALSE(evaluate_exact_triple_warm_subcycled_cell(parameters, altered,
            .3, rain, output, error));
        CHECK_FALSE(error.empty());
        compare_unchanged(output, good);
    }
    auto replay = good;
    REQUIRE(apply_exact_triple_warm_subcycled_cell(parameters, source, .3,
        initial, replay.producer, replay, error));
    compare_unchanged(replay, good);
}

TEST_CASE("subcycled cloud composition preserves independent activation and the exact FP64 handoff",
          "[physics][milbrandt][triple_warm_subcycle]")
{
    bool saw_condensation = false;
    bool saw_evaporation = false;
    bool saw_unrounded_handoff = false;
    bool saw_macro_pre_number = false;
    for (int fixture = 0; fixture < 3; ++fixture)
    {
        const auto initial = fixture == 2 ? reference::rain(.003f, 10.0f, 20)
            : dynamic_reference::source_rain(2e-4f, 120e-6, 20);
        const auto source = source_cell(initial, .65, 1e-4, 2e8,
            fixture == 1 ? .95 : 1.05);
        const double duration = .05;
        const double velocity = .5;
        const TripleCloudAdjustmentCell original{source.temperature, source.pressure,
            source.density, source.vapor, source.cloud, source.cloud_number, velocity};
        const auto independent = cloud_adjustment_reference::evaluate(original);
        ExactTripleSubcycledWarmCloudCellResult result;
        std::string error;
        const bool accepted = produce_exact_triple_subcycled_warm_cloud_cell(parameters,
            source, velocity, duration, initial, result, error);
        INFO("composed source " << fixture << ": " << error);
        REQUIRE(accepted);
        const auto& cloud = result.cloud.endpoints();
        CHECK(cloud_source_values(result.cloud.source_cell()) == cloud_source_values(original));
        CHECK(cloud.vapor_per_kg == Approx(static_cast<double>(independent.vapor)).epsilon(3e-12).margin(3e-15));
        CHECK(cloud.cloud_per_kg == Approx(static_cast<double>(independent.cloud)).epsilon(3e-12).margin(3e-15));
        CHECK(cloud.cloud_number_per_kg == Approx(static_cast<double>(independent.number)).epsilon(3e-12).margin(1e-12));
        CHECK(cloud.temperature_increment_k == Approx(static_cast<double>(independent.heat)).epsilon(3e-12).margin(1e-11));
        CHECK(cloud.temperature_after_k == Approx(static_cast<double>(independent.temperature)).epsilon(3e-14));
        CHECK(cloud.activated_number_per_m3 == Approx(static_cast<double>(independent.target)).epsilon(3e-12));
        const auto& warm_input = result.warm.producer.source_cell();
        CHECK(warm_input.temperature == cloud.temperature_after_k);
        CHECK(warm_input.vapor == cloud.vapor_per_kg);
        CHECK(warm_input.cloud == cloud.cloud_per_kg);
        CHECK(warm_input.cloud_number == cloud.cloud_number_per_kg);
        CHECK(warm_input.pressure == source.pressure);
        CHECK(warm_input.density == source.density);
        CHECK(warm_input.rain == source.rain);
        CHECK(warm_input.rain_number == source.rain_number);
        CHECK(result.warm.producer.original_rain().reflectivity_per_kg == initial.reflectivity_per_kg);
        check_trace(warm_input, duration, result.warm);
        CHECK(result.temperature_increment_k == cloud.temperature_increment_k + result.warm.endpoints.temperature_increment_k);
        CHECK(endpoint_values(result.endpoints) == endpoint_values(result.warm.endpoints));
        CHECK(rain_values(result.rain) == rain_values(result.warm.rain));
        CHECK(result.rounded_rain.mass_per_kg == static_cast<float>(result.rain.mass_per_kg));
        CHECK(result.rounded_rain.number_per_kg == static_cast<float>(result.rain.number_per_kg));
        CHECK(result.rounded_rain.reflectivity_per_kg == static_cast<float>(result.rain.reflectivity_per_kg));
        CHECK(result.endpoints.vapor_per_kg + result.endpoints.cloud_per_kg + result.endpoints.rain_per_kg
            == Approx(source.vapor + source.cloud + source.rain).epsilon(3e-13));
        CHECK(result.temperature_increment_k == Approx(microphysics_constants::L_v / microphysics_constants::cp *
            (source.vapor - result.endpoints.vapor_per_kg)).epsilon(3e-12).margin(1e-11));
        saw_condensation |= independent.transfer > 0;
        saw_evaporation |= independent.transfer < 0;
        saw_unrounded_handoff |= warm_input.temperature != static_cast<double>(static_cast<float>(warm_input.temperature))
            || warm_input.vapor != static_cast<double>(static_cast<float>(warm_input.vapor))
            || warm_input.cloud != static_cast<double>(static_cast<float>(warm_input.cloud))
            || warm_input.cloud_number != static_cast<double>(static_cast<float>(warm_input.cloud_number));
        saw_macro_pre_number |= result.warm.producer.pre_regularized_rain().number_per_kg > source.rain_number;
        ExactTripleSubcycledWarmCloudCellResult replay;
        REQUIRE(apply_exact_triple_subcycled_warm_cloud_cell(parameters, source, velocity,
            duration, initial, result, replay, error));
        compare_composed_unchanged(replay, result);
    }
    CHECK(saw_condensation);
    CHECK(saw_evaporation);
    CHECK(saw_unrounded_handoff);
    CHECK(saw_macro_pre_number);
}

TEST_CASE("subcycled cloud immutable phase bindings and resource failures are atomic",
          "[physics][milbrandt][triple_warm_subcycle]")
{
    const auto initial = dynamic_reference::source_rain(2e-4f, 120e-6, 20);
    const auto source = source_cell(initial, .65, 1e-4, 2e8, 1.05);
    ExactTripleSubcycledWarmCloudCellResult good;
    std::string error;
    REQUIRE(produce_exact_triple_subcycled_warm_cloud_cell(parameters, source,
        .5, .05, initial, good, error));
    for (int mutation = 0; mutation < 12; ++mutation)
    {
        auto changed = source;
        auto rain = initial;
        double velocity = .5;
        double duration = .05;
        switch (mutation)
        {
            case 0: changed.temperature = std::nextafter(changed.temperature, 400.0); break;
            case 1: changed.pressure = std::nextafter(changed.pressure, 2e5); break;
            case 2: changed.density = std::nextafter(changed.density, 2.0); break;
            case 3: changed.vapor = std::nextafter(changed.vapor, 1.0); break;
            case 4: changed.cloud = std::nextafter(changed.cloud, 1.0); break;
            case 5: changed.cloud_number = std::nextafter(changed.cloud_number, 1e9); break;
            case 6: changed.rain = std::nextafter(changed.rain, 1.0); break;
            case 7: changed.rain_number = std::nextafter(changed.rain_number, 1e9); break;
            case 8: velocity = std::nextafter(velocity, 1.0); break;
            case 9: duration = std::nextafter(duration, 1.0); break;
            case 10: rain.reflectivity_per_kg = std::nextafter(rain.reflectivity_per_kg, 1.0f); break;
            case 11: velocity = std::numeric_limits<double>::quiet_NaN(); break;
        }
        auto output = good;
        INFO("composed context mutation " << mutation);
        REQUIRE_FALSE(apply_exact_triple_subcycled_warm_cloud_cell(parameters, changed,
            velocity, duration, rain, good, output, error));
        CHECK_FALSE(error.empty());
        compare_composed_unchanged(output, good);
    }
    for (int mutation = 0; mutation < 13; ++mutation)
    {
        auto producer = good;
        auto& cloud = const_cast<TripleCloudAdjustmentEndpoints&>(producer.cloud.endpoints());
        switch (mutation)
        {
            case 0: cloud.temperature_after_k = std::nextafter(cloud.temperature_after_k, 400.0); break;
            case 1: cloud.cloud_per_kg = std::nextafter(cloud.cloud_per_kg, 1.0); break;
            case 2: cloud.cloud_number_per_kg = std::nextafter(cloud.cloud_number_per_kg, 1e9); break;
            case 3: const_cast<TripleCloudAdjustmentCell&>(producer.cloud.source_cell()).vertical_velocity_ms *= 2; break;
            case 4: producer.endpoints.vapor_per_kg = std::nextafter(producer.endpoints.vapor_per_kg, 1.0); break;
            case 5: producer.endpoints.cloud_per_kg = std::nextafter(producer.endpoints.cloud_per_kg, 1.0); break;
            case 6: producer.rain.reflectivity_per_kg = std::nextafter(producer.rain.reflectivity_per_kg, 1.0); break;
            case 7: producer.rounded_rain.reflectivity_per_kg = std::nextafter(producer.rounded_rain.reflectivity_per_kg, 1.0f); break;
            case 8: producer.temperature_increment_k = std::nextafter(producer.temperature_increment_k, 400.0); break;
            case 9: producer.warm.rates.rain = std::nextafter(producer.warm.rates.rain, 1.0f); break;
            case 10: producer.warm.endpoints.vapor_per_kg = std::nextafter(producer.warm.endpoints.vapor_per_kg, 1.0); break;
            case 11: const_cast<DoubleWarmRainCell&>(producer.warm.producer.source_cell()).vapor *= 2; break;
            case 12: const_cast<WarmSubcycleAcceptedStep&>(producer.warm.producer.steps().front()).endpoints.cloud_per_kg *= 2; break;
        }
        auto output = good;
        INFO("composed phase/capture mutation " << mutation);
        REQUIRE_FALSE(apply_exact_triple_subcycled_warm_cloud_cell(parameters, source,
            .5, .05, initial, producer, output, error));
        compare_composed_unchanged(output, good);
    }
    for (int rejected = 0; rejected < 4; ++rejected)
    {
        WarmSubcycleOptions options;
        double velocity = .5;
        switch (rejected)
        {
            case 0: options.max_accepted_steps = 1; break;
            case 1: options.observer_capacity = 0; break;
            case 2: options.relative_tolerance = 0; break;
            case 3: velocity = std::numeric_limits<double>::infinity(); break;
        }
        auto output = good;
        REQUIRE_FALSE(produce_exact_triple_subcycled_warm_cloud_cell(parameters,
            source, velocity, .05, initial, output, error, options));
        compare_composed_unchanged(output, good);
    }
}

TEST_CASE("FP64 warm steps bind exact sources policies and typed failure classes without Float admission",
          "[physics][milbrandt][triple_warm_subcycle]")
{
    const DynamicWarmFP64Policies policies{false, false, true};
    const double mass = std::nextafter(static_cast<double>(.002f), 1.0);
    const double number = std::nextafter(1000.0, 2000.0);
    const ExactRainTripleMomentState initial{mass, number, static_cast<double>(
        reference::moment_ratio(20) * mass * mass /
        (reference::mass_coefficient * reference::mass_coefficient * number))};
    const DoubleWarmRainCell source{290.125, 90000, .65,
        1.1 * thermodynamics::saturation_mixing_ratio_water(290.125, 90000),
        0, mass, 0, number};
    CHECK(initial.mass_per_kg != static_cast<double>(static_cast<float>(initial.mass_per_kg)));
    CHECK(initial.number_per_kg != static_cast<double>(static_cast<float>(initial.number_per_kg)));
    CHECK(initial.reflectivity_per_kg != static_cast<double>(static_cast<float>(initial.reflectivity_per_kg)));
    const auto compare = [](const ExactTripleDynamicWarmFP64CellResult& actual,
                            const ExactTripleDynamicWarmFP64CellResult& expected)
    {
        CHECK(endpoint_values(actual.endpoints) == endpoint_values(expected.endpoints));
        CHECK(rain_values(actual.rain) == rain_values(expected.rain));
        CHECK(std::memcmp(&actual.rates, &expected.rates, sizeof(actual.rates)) == 0);
        const auto& a = actual.producer;
        const auto& e = expected.producer;
        CHECK(rain_values(a.original_rain()) == rain_values(e.original_rain()));
        CHECK(rain_values(a.pre_physical_rain()) == rain_values(e.pre_physical_rain()));
        CHECK(rain_values(a.pre_evaporation_rain()) == rain_values(e.pre_evaporation_rain()));
        CHECK(a.pre_physical_event_count() == e.pre_physical_event_count());
        CHECK(a.pre_evaporation_event_count() == e.pre_evaporation_event_count());
        CHECK(a.policies().pre_number_regularization == e.policies().pre_number_regularization);
        CHECK(a.policies().post_number_regularization == e.policies().post_number_regularization);
        CHECK(a.policies().reclassification == e.policies().reclassification);
        CHECK(source_values(a.producer().source_cell()) == source_values(e.producer().source_cell()));
        CHECK(a.producer().duration_s() == e.producer().duration_s());
        CHECK(endpoint_values(a.producer().endpoints()) == endpoint_values(e.producer().endpoints()));
        CHECK(a.producer().events().size == e.producer().events().size);
        CHECK(a.producer().events().capacity == e.producer().events().capacity);
    };
    ExactTripleDynamicWarmFP64CellResult good;
    DynamicWarmStepFailure failure = DynamicWarmStepFailure::Resource;
    std::string error;
    REQUIRE(evaluate_exact_triple_dynamic_warm_fp64_cell(parameters, source, .01,
        initial, good, error, &failure, 16, policies));
    CHECK(failure == DynamicWarmStepFailure::None);
    CHECK(error.empty());
    CHECK(std::memcmp(&source, &good.producer.producer().source_cell(), sizeof(source)) == 0);
    CHECK(rain_values(good.producer.original_rain()) == rain_values(initial));
    CHECK(endpoint_values(good.endpoints) == endpoint_values(good.producer.producer().endpoints()));
    CHECK(good.rain.mass_per_kg == good.endpoints.rain_per_kg);
    CHECK(good.rain.number_per_kg == good.endpoints.rain_number_per_kg);
    const auto independent = reference::euler_breakup({initial.mass_per_kg,
        initial.number_per_kg, initial.reflectivity_per_kg}, source.density, .01);
    CHECK(good.rain.number_per_kg == Approx(static_cast<double>(independent.n)).epsilon(3e-12));
    CHECK(good.rain.reflectivity_per_kg == Approx(static_cast<double>(independent.z)).epsilon(3e-12));
    CHECK(good.rain.number_per_kg != static_cast<double>(static_cast<float>(good.rain.number_per_kg)));
    ExactTripleDynamicWarmFP64CellResult replay;
    REQUIRE(apply_exact_triple_dynamic_warm_fp64_cell(parameters, source, .01,
        initial, good.producer, replay, error, &failure, policies));
    CHECK(failure == DynamicWarmStepFailure::None);
    compare(replay, good);
    for (int invalid = 0; invalid < 10; ++invalid)
    {
        auto altered = source;
        auto rain = initial;
        double duration = .01;
        switch (invalid)
        {
            case 0: altered.temperature = std::numeric_limits<double>::quiet_NaN(); break;
            case 1: altered.rain_number = std::nextafter(altered.rain_number, 2000.0); break;
            case 2: rain.reflectivity_per_kg = static_cast<double>(reference::moment_ratio(220) * mass * mass /
                (reference::mass_coefficient * reference::mass_coefficient * number)); break;
            case 3: altered.cloud = 1e-4; break;
            case 4: rain.reflectivity_per_kg = -1; break;
            case 5: altered.density = 0; break;
            case 6: altered.temperature = 401; break;
            case 7: duration = 0; break;
            case 8: altered.vapor = std::numeric_limits<double>::quiet_NaN(); break;
            case 9: rain.mass_per_kg = std::nextafter(rain.mass_per_kg, 1.0); break;
        }
        auto output = good;
        INFO("FP64 invalid starting source " << invalid);
        REQUIRE_FALSE(evaluate_exact_triple_dynamic_warm_fp64_cell(parameters,
            altered, duration, rain, output, error, &failure, 16, policies));
        CHECK(failure == DynamicWarmStepFailure::InvalidSource);
        CHECK_FALSE(error.empty());
        compare(output, good);
    }
    for (int mutation = 0; mutation < 10; ++mutation)
    {
        auto altered = source;
        auto rain = initial;
        auto producer = good.producer;
        auto requested = policies;
        double duration = .01;
        switch (mutation)
        {
            case 0: rain.reflectivity_per_kg = std::nextafter(rain.reflectivity_per_kg, 1.0); break;
            case 1: requested.pre_number_regularization = true; break;
            case 2: requested.post_number_regularization = true; break;
            case 3: requested.reclassification = false; break;
            case 4: altered.cloud = -0.0; break;
            case 5: const_cast<ExactRainTripleMomentState&>(producer.pre_evaporation_rain()).reflectivity_per_kg *= 2; break;
            case 6: const_cast<WarmRainDoubleEndpoints&>(producer.producer().endpoints()).rain_per_kg =
                std::nextafter(producer.producer().endpoints().rain_per_kg, 1.0); break;
            case 7: const_cast<DynamicWarmFP64Policies&>(producer.policies()).pre_number_regularization = true; break;
            case 8: const_cast<DoubleWarmRainCell&>(producer.producer().source_cell()).density =
                std::nextafter(source.density, 2.0); break;
            case 9: duration = std::nextafter(duration, 1.0); break;
        }
        auto output = good;
        INFO("FP64 byte/policy/capture mutation " << mutation);
        REQUIRE_FALSE(apply_exact_triple_dynamic_warm_fp64_cell(parameters, altered,
            duration, rain, producer, output, error, &failure, requested));
        CHECK(failure == DynamicWarmStepFailure::InvalidSource);
        compare(output, good);
    }
    const auto dense_float = reference::rain(.003f, 10.0f, 20);
    const ExactRainTripleMomentState dense{dense_float.mass_per_kg,
        dense_float.number_per_kg, dense_float.reflectivity_per_kg};
    const auto dense_source = source_cell(dense_float, 1.3);
    auto rejected = good;
    REQUIRE_FALSE(evaluate_exact_triple_dynamic_warm_fp64_cell(parameters, dense_source,
        .3, dense, rejected, error, &failure, 16, policies));
    CHECK(failure == DynamicWarmStepFailure::IntervalRejected);
    compare(rejected, good);
    for (std::size_t capacity : {std::size_t(0), std::size_t(17)})
    {
        auto output = good;
        REQUIRE_FALSE(evaluate_exact_triple_dynamic_warm_fp64_cell(parameters, source,
            .01, initial, output, error, &failure, capacity, policies));
        CHECK(failure == DynamicWarmStepFailure::Resource);
        compare(output, good);
    }
    const ExactRainTripleMomentState tiny{1e-50, 1, static_cast<double>(reference::moment_ratio(20) *
        1e-100L / (reference::mass_coefficient * reference::mass_coefficient))};
    auto tiny_source = source;
    tiny_source.rain = tiny.mass_per_kg;
    tiny_source.rain_number = tiny.number_per_kg;
    ExactTripleDynamicWarmFP64CellResult tiny_result;
    REQUIRE(evaluate_exact_triple_dynamic_warm_fp64_cell(parameters, tiny_source,
        .01, tiny, tiny_result, error, &failure, 16, policies));
    CHECK(failure == DynamicWarmStepFailure::None);
    CHECK(tiny_result.endpoints.reclassification.present);
    CHECK(tiny_result.endpoints.cloud_per_kg == tiny.mass_per_kg);
    CHECK(tiny_result.endpoints.cloud_per_kg > 0);
    CHECK(static_cast<float>(tiny_result.endpoints.cloud_per_kg) == 0);
    CHECK(tiny_result.rain.mass_per_kg == 0);
    CHECK(tiny_result.rain.number_per_kg == 0);
    CHECK(tiny_result.rain.reflectivity_per_kg == 0);
    REQUIRE(apply_exact_triple_dynamic_warm_fp64_cell(parameters, tiny_source,
        .01, tiny, tiny_result.producer, replay, error, &failure, policies));
    compare(replay, tiny_result);
}
