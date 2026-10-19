#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/triple_warm_primary.hpp"
#include "microphysics/schemes/milbrandt/triple_warm_dynamic.hpp"
#include "microphysics/schemes/milbrandt/triple_warm_cloud.hpp"
#include "support/milbrandt_cloud_adjustment_reference.hpp"
#include "support/milbrandt_dynamic_warm_reference.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

using namespace tmv;
using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_dynamic_warm_reference;

const WarmRainParameters parameters = warm_rain_parameters(0, 523.598775598, 3, 149.1, .5);

DoubleWarmRainCell source_cell(const RainTripleMomentState& rain, double density,
                               double cloud = .001, double cloud_number = 1e8,
                               double humidity = .7)
{
    const double qsat = thermodynamics::saturation_mixing_ratio_water(290.125, 90000);
    return {290.125, 90000, density, humidity * qsat,
            cloud, rain.mass_per_kg, cloud_number, rain.number_per_kg};
}

const WarmRainAcceptedEvent* find_event(const WarmRainAcceptedEventBuffer& events,
                                      WarmRainAcceptedEventKind kind)
{
    for (std::size_t index = 0; index < events.size; ++index)
    {
        if (events.events[index].kind == kind)
        {
            return &events.events[index];
        }
    }
    return nullptr;
}

void compare_population(const ExactRainTripleMomentState& actual,
                        const reference::Population& expected)
{
    CHECK(actual.mass_per_kg == Approx(static_cast<double>(expected.q)).epsilon(3e-12).margin(1e-30));
    CHECK(actual.number_per_kg == Approx(static_cast<double>(expected.n)).epsilon(3e-12).margin(1e-30));
    CHECK(actual.reflectivity_per_kg == Approx(static_cast<double>(expected.z)).epsilon(3e-12).margin(1e-30));
}

void compare_spectrum(const TripleWarmRainSpectrum* spectrum,
                      const reference::Population& expected)
{
    if (expected.q == 0 && expected.n == 0 && expected.z == 0)
    {
        CHECK(spectrum == nullptr);
        return;
    }
    REQUIRE(spectrum != nullptr);
    compare_population(spectrum->source(), expected);
    const long double shape = reference::alpha(expected);
    CHECK(spectrum->alpha() == Approx(static_cast<double>(shape)).epsilon(2e-12).margin(2e-12));
    CHECK(spectrum->lambda_per_m() == Approx(static_cast<double>(reference::lambda(expected, shape))).epsilon(2e-12));
    CHECK(spectrum->diameter_moment_1_m() == Approx(static_cast<double>(reference::integer_moment(expected, 1))).epsilon(2e-12));
    CHECK(spectrum->diameter_moment_3_m3() == Approx(static_cast<double>(reference::integer_moment(expected, 3))).epsilon(2e-12));
    CHECK(spectrum->diameter_moment_6_m6() == Approx(static_cast<double>(reference::integer_moment(expected, 6))).epsilon(2e-12));
    CHECK(spectrum->ventilation_diameter_moment() == Approx(static_cast<double>(reference::fractional_moment(expected, 1.75L))).epsilon(8e-12));
    CHECK(spectrum->alpha_zero_equivalent_inverse_slope_m() == Approx(static_cast<double>(
        reference::mean_diameter(expected) / std::cbrt(6.0L))).epsilon(3e-14));
}

void compare_event(const WarmRainAcceptedEventBuffer& events,
                   WarmRainAcceptedEventKind kind,
                   long double mass_rate, long double rain_number_rate,
                   long double cloud_number_rate)
{
    const auto* event = find_event(events, kind);
    if (mass_rate == 0 && rain_number_rate == 0 && cloud_number_rate == 0)
    {
        CHECK(event == nullptr);
        return;
    }
    REQUIRE(event != nullptr);
    CHECK(event->rain_mass_rate_per_kg_s == Approx(static_cast<double>(mass_rate)).epsilon(3e-12));
    CHECK(event->rain_number_rate_per_m3_s == Approx(static_cast<double>(rain_number_rate)).epsilon(3e-12));
    CHECK(event->cloud_number_rate_per_m3_s == Approx(static_cast<double>(cloud_number_rate)).epsilon(3e-12));
}

void check_reference(const DoubleWarmRainCell& source, double dt,
                     const RainTripleMomentState& initial,
                     const ExactTripleDynamicWarmCellResult& result)
{
    const auto& dynamic = result.producer;
    const auto& captured = dynamic.producer();
    const auto& events = captured.events();
    const auto original = reference::population(initial);
    const auto physical = reference::replay_prefix(initial, events,
        dynamic.pre_physical_event_count(), dt, source.density);
    const auto evaporation = reference::replay_prefix(initial, events,
        dynamic.pre_evaporation_event_count(), dt, source.density);
    compare_spectrum(dynamic.original_spectrum(), original);
    compare_population(dynamic.pre_physical_rain(), physical);
    compare_spectrum(dynamic.pre_physical_spectrum(), physical);
    compare_population(dynamic.pre_evaporation_rain(), evaporation);
    compare_spectrum(dynamic.pre_evaporation_spectrum(), evaporation);
    CHECK(dynamic.original_rain().mass_per_kg == initial.mass_per_kg);
    CHECK(dynamic.original_rain().number_per_kg == initial.number_per_kg);
    CHECK(dynamic.original_rain().reflectivity_per_kg == initial.reflectivity_per_kg);
    REQUIRE(dynamic.pre_physical_event_count() <= dynamic.pre_evaporation_event_count());
    REQUIRE(dynamic.pre_evaporation_event_count() <= events.size);

    const auto independent = reference::physical_rates(source, physical, dt);
    compare_event(events, WarmRainAcceptedEventKind::Autoconversion,
        independent.autoconversion_mass, independent.autoconversion_number, 0);
    compare_event(events, WarmRainAcceptedEventKind::Accretion,
        independent.accretion_mass, 0, -independent.accretion_number);
    compare_event(events, WarmRainAcceptedEventKind::CloudSelfCollection,
        0, 0, -independent.cloud_self);
    compare_event(events, independent.rain_self < 0
        ? WarmRainAcceptedEventKind::RainBreakup : WarmRainAcceptedEventKind::RainSelfCollection,
        0, -independent.rain_self, 0);
    const auto* automatic = find_event(events, WarmRainAcceptedEventKind::Autoconversion);
    if (automatic)
    {
        CHECK(automatic->seed_diameter_m == Approx(static_cast<double>(independent.seed_diameter)).epsilon(3e-12));
        CHECK(automatic->cloud_donor_scale == Approx(static_cast<double>(independent.donor_scale)).epsilon(3e-12));
        CHECK(automatic->rain_number_rate_per_m3_s == Approx(automatic->rain_mass_rate_per_kg_s *
            source.density / (523.598775598 * std::pow(automatic->seed_diameter_m, 3))).epsilon(2e-14));
    }
    const long double admitted = reference::evaporated(source, evaporation, dt);
    const auto& budget = result.endpoints.evaporation;
    CHECK(budget.present == (admitted > 0));
    CHECK(budget.admitted_mass_per_kg == Approx(static_cast<double>(admitted)).epsilon(8e-12).margin(1e-30));
    reference::Population endpoint = evaporation;
    if (admitted > 0)
    {
        const long double remaining = (evaporation.q - admitted) / evaporation.q;
        endpoint.q -= admitted;
        endpoint.n *= remaining;
        endpoint.z *= remaining;
        CHECK(budget.mass_before_per_kg == Approx(static_cast<double>(evaporation.q)).epsilon(3e-12));
        CHECK(budget.number_before_per_m3 == Approx(static_cast<double>(source.density * evaporation.n)).epsilon(3e-12));
        CHECK(budget.donor_exhausted == (admitted == evaporation.q));
    }
    compare_population(result.rain, endpoint);
    CHECK(result.endpoints.rain_per_kg == result.rain.mass_per_kg);
    CHECK(result.endpoints.rain_number_per_kg == result.rain.number_per_kg);
    CHECK(result.rounded_rain.mass_per_kg == static_cast<float>(result.rain.mass_per_kg));
    CHECK(result.rounded_rain.number_per_kg == static_cast<float>(result.rain.number_per_kg));
    CHECK(result.rounded_rain.reflectivity_per_kg == static_cast<float>(result.rain.reflectivity_per_kg));
    CHECK(result.endpoints.vapor_per_kg == Approx(source.vapor + static_cast<double>(admitted)).epsilon(3e-12));
    CHECK(result.endpoints.temperature_increment_k == Approx(static_cast<double>(-2.5e6L / 1004 * admitted)).epsilon(8e-12).margin(1e-20));
    CHECK(result.endpoints.vapor_per_kg + result.endpoints.cloud_per_kg + result.endpoints.rain_per_kg
        == Approx(source.vapor + source.cloud + source.rain).epsilon(2e-14));
    if (result.endpoints.cloud_donor_exhausted && !result.endpoints.reclassification.present)
    {
        CHECK(result.endpoints.cloud_donor_budget_per_kg == source.cloud);
        CHECK(result.endpoints.cloud_per_kg == 0);
    }
    for (std::size_t index = 0; index < events.size; ++index)
    {
        CHECK(events.events[index].dry_density_kg_m3 == source.density);
        CHECK(events.events[index].kind != WarmRainAcceptedEventKind::RainNumberClip);
        CHECK(events.events[index].kind != WarmRainAcceptedEventKind::RainMassClip);
        CHECK(events.events[index].kind != WarmRainAcceptedEventKind::RainReset);
    }
}

void compare_unchanged(const ExactTripleDynamicWarmCellResult& actual,
                       const ExactTripleDynamicWarmCellResult& expected)
{
    const auto endpoint_values = [](const WarmRainDoubleEndpoints& endpoint)
    {
        return std::array<double, 16>{endpoint.vapor_per_kg, endpoint.cloud_per_kg,
            endpoint.rain_per_kg, endpoint.cloud_number_per_kg, endpoint.rain_number_per_kg,
            endpoint.temperature_increment_k, endpoint.cloud_donor_budget_per_kg,
            endpoint.evaporation.mass_before_per_kg, endpoint.evaporation.number_before_per_m3,
            endpoint.evaporation.admitted_mass_per_kg, endpoint.evaporation.mass_after_per_kg,
            endpoint.evaporation.number_after_per_m3, endpoint.reclassification.rain_mass_before_per_kg,
            endpoint.reclassification.rain_number_before_per_m3,
            endpoint.reclassification.cloud_mass_before_per_kg,
            endpoint.reclassification.cloud_number_before_per_m3};
    };
    CHECK(endpoint_values(actual.endpoints) == endpoint_values(expected.endpoints));
    CHECK(actual.endpoints.cloud_donor_exhausted == expected.endpoints.cloud_donor_exhausted);
    CHECK(actual.endpoints.evaporation.present == expected.endpoints.evaporation.present);
    CHECK(actual.endpoints.evaporation.donor_exhausted == expected.endpoints.evaporation.donor_exhausted);
    CHECK(actual.endpoints.reclassification.present == expected.endpoints.reclassification.present);
    CHECK(std::memcmp(&actual.rain, &expected.rain, sizeof(actual.rain)) == 0);
    CHECK(std::memcmp(&actual.rounded_rain, &expected.rounded_rain, sizeof(actual.rounded_rain)) == 0);
    CHECK(std::memcmp(&actual.rates, &expected.rates, sizeof(actual.rates)) == 0);
    CHECK(actual.producer.original_rain().mass_per_kg == expected.producer.original_rain().mass_per_kg);
    CHECK(actual.producer.original_rain().number_per_kg == expected.producer.original_rain().number_per_kg);
    CHECK(actual.producer.original_rain().reflectivity_per_kg == expected.producer.original_rain().reflectivity_per_kg);
    CHECK(std::memcmp(&actual.producer.pre_physical_rain(), &expected.producer.pre_physical_rain(),
        sizeof(ExactRainTripleMomentState)) == 0);
    CHECK(std::memcmp(&actual.producer.pre_evaporation_rain(), &expected.producer.pre_evaporation_rain(),
        sizeof(ExactRainTripleMomentState)) == 0);
    CHECK(actual.producer.pre_physical_event_count() == expected.producer.pre_physical_event_count());
    CHECK(actual.producer.pre_evaporation_event_count() == expected.producer.pre_evaporation_event_count());
    const auto& a = actual.producer.producer();
    const auto& e = expected.producer.producer();
    CHECK(std::memcmp(&a.source_cell(), &e.source_cell(), sizeof(DoubleWarmRainCell)) == 0);
    CHECK(a.duration_s() == e.duration_s());
    CHECK(a.events().size == e.events().size);
    CHECK(a.events().capacity == e.events().capacity);
    REQUIRE(a.events().size <= a.events().events.size());
    for (std::size_t index = 0; index < a.events().size; ++index)
    {
        const auto& actual_event = a.events().events[index];
        const auto& expected_event = e.events().events[index];
        CHECK(actual_event.kind == expected_event.kind);
        CHECK(actual_event.cloud_mass_rate_per_kg_s == expected_event.cloud_mass_rate_per_kg_s);
        CHECK(actual_event.rain_mass_rate_per_kg_s == expected_event.rain_mass_rate_per_kg_s);
        CHECK(actual_event.vapor_mass_rate_per_kg_s == expected_event.vapor_mass_rate_per_kg_s);
        CHECK(actual_event.cloud_number_rate_per_m3_s == expected_event.cloud_number_rate_per_m3_s);
        CHECK(actual_event.rain_number_rate_per_m3_s == expected_event.rain_number_rate_per_m3_s);
        CHECK(actual_event.temperature_rate_k_s == expected_event.temperature_rate_k_s);
        CHECK(actual_event.dry_density_kg_m3 == expected_event.dry_density_kg_m3);
        CHECK(actual_event.cloud_donor_scale == expected_event.cloud_donor_scale);
        CHECK(actual_event.seed_diameter_m == expected_event.seed_diameter_m);
        CHECK(actual_event.seed_mass_number_coherent == expected_event.seed_mass_number_coherent);
    }
}

void compare_composed_unchanged(const ExactTripleDynamicWarmCloudCellResult& actual,
                                const ExactTripleDynamicWarmCloudCellResult& expected)
{
    compare_unchanged(actual.warm, expected.warm);
    auto actual_endpoint = actual.warm;
    auto expected_endpoint = expected.warm;
    actual_endpoint.endpoints = actual.endpoints;
    actual_endpoint.rain = actual.rain;
    actual_endpoint.rounded_rain = actual.rounded_rain;
    expected_endpoint.endpoints = expected.endpoints;
    expected_endpoint.rain = expected.rain;
    expected_endpoint.rounded_rain = expected.rounded_rain;
    compare_unchanged(actual_endpoint, expected_endpoint);
    CHECK(actual.temperature_increment_k == expected.temperature_increment_k);
    CHECK(std::memcmp(&actual.cloud.source_cell(), &expected.cloud.source_cell(),
        sizeof(TripleCloudAdjustmentCell)) == 0);
    CHECK(actual.cloud.duration_s() == expected.cloud.duration_s());
    const auto cloud_values = [](const TripleCloudAdjustmentEndpoints& endpoint)
    {
        return std::array<double, 7>{endpoint.vapor_per_kg, endpoint.cloud_per_kg,
            endpoint.cloud_number_per_kg, endpoint.temperature_increment_k,
            endpoint.temperature_after_k, endpoint.admitted_cloud_increment_per_kg,
            endpoint.activated_number_per_m3};
    };
    CHECK(cloud_values(actual.cloud.endpoints()) == cloud_values(expected.cloud.endpoints()));
    CHECK(actual.cloud.endpoints().cloud_donor_exhausted == expected.cloud.endpoints().cloud_donor_exhausted);
}

void check_composed_reference(const DoubleWarmRainCell& source, double velocity,
                              double dt, const RainTripleMomentState& initial,
                              const ExactTripleDynamicWarmCloudCellResult& result)
{
    const TripleCloudAdjustmentCell original{source.temperature, source.pressure,
        source.density, source.vapor, source.cloud, source.cloud_number, velocity};
    const auto independent = cloud_adjustment_reference::evaluate(original);
    const auto& cloud = result.cloud.endpoints();
    CHECK(cloud.vapor_per_kg == Approx(static_cast<double>(independent.vapor)).epsilon(3e-12).margin(3e-15));
    CHECK(cloud.cloud_per_kg == Approx(static_cast<double>(independent.cloud)).epsilon(3e-12).margin(3e-15));
    CHECK(cloud.cloud_number_per_kg == Approx(static_cast<double>(independent.number)).epsilon(3e-12).margin(1e-12));
    CHECK(cloud.temperature_increment_k == Approx(static_cast<double>(independent.heat)).epsilon(3e-12).margin(1e-11));
    CHECK(cloud.temperature_after_k == Approx(static_cast<double>(independent.temperature)).epsilon(3e-12).margin(1e-11));
    CHECK(cloud.activated_number_per_m3 == Approx(static_cast<double>(independent.target)).epsilon(3e-12).margin(1e-12));
    const auto& dynamic = result.warm.producer;
    const auto& capture = dynamic.producer();
    const auto& warm = capture.source_cell();
    CHECK(warm.temperature == cloud.temperature_after_k);
    CHECK(warm.vapor == cloud.vapor_per_kg);
    CHECK(warm.cloud == cloud.cloud_per_kg);
    CHECK(warm.cloud_number == cloud.cloud_number_per_kg);
    CHECK(warm.pressure == source.pressure);
    CHECK(warm.density == source.density);
    CHECK(warm.rain == source.rain);
    CHECK(warm.rain_number == source.rain_number);
    const auto physical = reference::replay_prefix(initial, capture.events(),
        dynamic.pre_physical_event_count(), dt, source.density);
    const auto evaporation = reference::replay_prefix(initial, capture.events(),
        dynamic.pre_evaporation_event_count(), dt, source.density);
    compare_spectrum(dynamic.original_spectrum(), reference::population(initial));
    compare_population(dynamic.pre_physical_rain(), physical);
    compare_spectrum(dynamic.pre_physical_spectrum(), physical);
    compare_population(dynamic.pre_evaporation_rain(), evaporation);
    compare_spectrum(dynamic.pre_evaporation_spectrum(), evaporation);
    const auto physical_rates = reference::physical_rates(warm, physical, dt);
    compare_event(capture.events(), WarmRainAcceptedEventKind::Autoconversion,
        physical_rates.autoconversion_mass, physical_rates.autoconversion_number, 0);
    compare_event(capture.events(), WarmRainAcceptedEventKind::Accretion,
        physical_rates.accretion_mass, 0, -physical_rates.accretion_number);
    compare_event(capture.events(), physical_rates.rain_self < 0
        ? WarmRainAcceptedEventKind::RainBreakup : WarmRainAcceptedEventKind::RainSelfCollection,
        0, -physical_rates.rain_self, 0);
    auto endpoint = evaporation;
    const auto& budget = result.warm.endpoints.evaporation;
    if (budget.present)
    {
        const long double remaining = static_cast<long double>(budget.mass_after_per_kg) /
            budget.mass_before_per_kg;
        endpoint.q = budget.mass_after_per_kg;
        endpoint.n *= remaining;
        endpoint.z *= remaining;
    }
    compare_population(result.rain, endpoint);
    CHECK(result.rain.mass_per_kg == result.warm.rain.mass_per_kg);
    CHECK(result.rain.number_per_kg == result.warm.rain.number_per_kg);
    CHECK(result.rain.reflectivity_per_kg == result.warm.rain.reflectivity_per_kg);
    CHECK(result.rounded_rain.mass_per_kg == static_cast<float>(result.rain.mass_per_kg));
    CHECK(result.rounded_rain.number_per_kg == static_cast<float>(result.rain.number_per_kg));
    CHECK(result.rounded_rain.reflectivity_per_kg == static_cast<float>(result.rain.reflectivity_per_kg));
    CHECK(result.endpoints.vapor_per_kg == result.warm.endpoints.vapor_per_kg);
    CHECK(result.endpoints.cloud_per_kg == result.warm.endpoints.cloud_per_kg);
    CHECK(result.endpoints.rain_per_kg == result.warm.endpoints.rain_per_kg);
    CHECK(result.endpoints.cloud_number_per_kg == result.warm.endpoints.cloud_number_per_kg);
    CHECK(result.endpoints.rain_number_per_kg == result.warm.endpoints.rain_number_per_kg);
    CHECK(result.temperature_increment_k == cloud.temperature_increment_k + result.warm.endpoints.temperature_increment_k);
    CHECK(result.temperature_increment_k == Approx(static_cast<double>(independent.heat -
        2.5e6L / 1004 * budget.admitted_mass_per_kg)).epsilon(3e-12).margin(1e-11));
    CHECK(result.temperature_increment_k == Approx(2.5e6 / 1004 *
        (source.vapor - result.endpoints.vapor_per_kg)).epsilon(3e-12).margin(1e-11));
    CHECK(result.endpoints.vapor_per_kg + result.endpoints.cloud_per_kg + result.endpoints.rain_per_kg
        == Approx(source.vapor + source.cloud + source.rain).epsilon(2e-14));
}

} // namespace

TEST_CASE("dynamic warm rates use represented shape and independent polynomial and fractional spectra",
          "[physics][milbrandt][triple_dynamic_warm]")
{
    bool saw_accretion = false;
    bool saw_self_collection = false;
    bool saw_shape_change = false;
    bool saw_quantized_shape = false;
    bool saw_ventilation_change = false;
    for (long double nominal : {0.0L, 2.0L, 20.0L, 200.0L})
    {
        for (double diameter : {90e-6, 120e-6, 1e-3})
        {
            const auto initial = reference::source_rain(2e-4f, diameter, nominal);
            const long double represented = reference::alpha(reference::population(initial));
            saw_quantized_shape |= static_cast<double>(represented) != static_cast<double>(nominal);
            for (double density : {.65, 1.43})
            {
                for (double dt : {.01, .1})
                {
                    const auto source = source_cell(initial, density);
                    ExactTripleDynamicWarmCellResult result;
                    std::string error;
                    const bool accepted = evaluate_exact_triple_dynamic_warm_cell(
                        parameters, source, dt, initial, result, error);
                    INFO("nominal alpha " << nominal << ", represented alpha " << represented
                        << ", D " << diameter << ", rho " << density << ", dt " << dt << ": " << error);
                    REQUIRE(accepted);
                    check_reference(source, dt, initial, result);
                    const auto& events = result.producer.producer().events();
                    saw_accretion |= find_event(events, WarmRainAcceptedEventKind::Accretion) != nullptr;
                    saw_self_collection |= find_event(events, WarmRainAcceptedEventKind::RainSelfCollection) != nullptr;
                    const auto* before = result.producer.pre_physical_spectrum();
                    const auto* after = result.producer.pre_evaporation_spectrum();
                    REQUIRE(before != nullptr);
                    REQUIRE(after != nullptr);
                    saw_shape_change |= std::abs(before->alpha() - after->alpha()) > 1e-5;
                    const auto& a = after->source();
                    if (std::abs(before->alpha() - after->alpha()) > 1e-5)
                    {
                        const long double old_lambda = reference::lambda({a.mass_per_kg,
                            a.number_per_kg, a.reflectivity_per_kg}, before->alpha());
                        const long double frozen_moment = reference::scaled_integral(before->alpha(), 1.75L) /
                            reference::scaled_integral(before->alpha(), 0) / std::pow(old_lambda, 1.75L);
                        saw_ventilation_change |= std::abs(after->ventilation_diameter_moment() /
                            static_cast<double>(frozen_moment) - 1) > 1e-6;
                    }
                    ExactTripleDynamicWarmCellResult replay;
                    REQUIRE(apply_exact_triple_dynamic_warm_cell(parameters, source, dt,
                        initial, result.producer, replay, error));
                    compare_unchanged(replay, result);
                }
            }
        }
    }
    CHECK(saw_accretion);
    CHECK(saw_self_collection);
    CHECK(saw_shape_change);
    CHECK(saw_quantized_shape);
    CHECK(saw_ventilation_change);
}

TEST_CASE("dynamic rain breakup retains alpha zero efficiency at equivalent mean size through shape 200",
          "[physics][milbrandt][triple_dynamic_warm]")
{
    for (double diameter : {.5e-3, 1.6e-3})
    {
        const long double inverse_equivalent = diameter / std::cbrt(6.0L);
        const bool breakup = inverse_equivalent > 300e-6L &&
            2 - std::exp(2300 * (inverse_equivalent - 300e-6L)) < 0;
        for (long double nominal : {0.0L, 2.0L, 20.0L, 200.0L})
        {
            const auto initial = reference::source_rain(.002f, diameter, nominal);
            const auto source = source_cell(initial, .65, 0, 0, 1.1);
            ExactTripleDynamicWarmCellResult result;
            std::string error;
            const bool accepted = evaluate_exact_triple_dynamic_warm_cell(
                parameters, source, .1, initial, result, error);
            INFO("shape " << nominal << ", D " << diameter << ": " << error);
            REQUIRE(accepted);
            check_reference(source, .1, initial, result);
            const auto& events = result.producer.producer().events();
            const auto* event = find_event(events, breakup
                ? WarmRainAcceptedEventKind::RainBreakup : WarmRainAcceptedEventKind::RainSelfCollection);
            REQUIRE(event != nullptr);
            CHECK((event->rain_number_rate_per_m3_s > 0) == breakup);
            CHECK(find_event(events, breakup ? WarmRainAcceptedEventKind::RainSelfCollection
                : WarmRainAcceptedEventKind::RainBreakup) == nullptr);
            CHECK(result.rain.mass_per_kg == source.rain);
            CHECK(result.endpoints.temperature_increment_k == 0);
            CHECK((result.rain.number_per_kg > source.rain_number) == breakup);
            CHECK((result.rain.reflectivity_per_kg < initial.reflectivity_per_kg) == breakup);
            if (nominal == 200 && breakup)
            {
                REQUIRE(result.producer.original_spectrum() != nullptr);
                CHECK(1 / result.producer.original_spectrum()->lambda_per_m() < 300e-6);
                CHECK(result.producer.original_spectrum()->alpha_zero_equivalent_inverse_slope_m() > 300e-6);
            }
        }
    }
}

TEST_CASE("dynamic warm phase boundaries rediagnose pre and post regularization and newly seeded rain",
          "[physics][milbrandt][triple_dynamic_warm]")
{
    for (long double shape : {0.0L, 20.0L, 200.0L})
    {
        const auto initial = reference::source_rain(.002f, .004, shape);
        const auto source = source_cell(initial, 1.43, 0, 0, .5);
        ExactTripleDynamicWarmCellResult result;
        std::string error;
        const bool accepted = evaluate_exact_triple_dynamic_warm_cell(
            parameters, source, .01, initial, result, error);
        INFO("regularization shape " << shape << ": " << error);
        REQUIRE(accepted);
        check_reference(source, .01, initial, result);
        const auto& events = result.producer.producer().events();
        REQUIRE(find_event(events, WarmRainAcceptedEventKind::PreRainNumberRegularization) != nullptr);
        REQUIRE(find_event(events, WarmRainAcceptedEventKind::PostRainNumberRegularization) != nullptr);
        CHECK(result.producer.pre_physical_event_count() == 1);
        CHECK(result.producer.pre_evaporation_event_count() >= 3);
        CHECK(result.producer.pre_physical_rain().number_per_kg > initial.number_per_kg);
        CHECK(result.producer.pre_physical_rain().reflectivity_per_kg < initial.reflectivity_per_kg);
        REQUIRE(result.producer.original_spectrum() != nullptr);
        REQUIRE(result.producer.pre_physical_spectrum() != nullptr);
        REQUIRE(result.producer.pre_evaporation_spectrum() != nullptr);
        CHECK(result.producer.original_spectrum()->alpha() == Approx(result.producer.pre_physical_spectrum()->alpha()).epsilon(3e-12).margin(3e-12));
        CHECK(result.producer.pre_physical_spectrum()->lambda_per_m() != result.producer.original_spectrum()->lambda_per_m());
        CHECK(result.producer.pre_evaporation_spectrum()->lambda_per_m() != result.producer.pre_physical_spectrum()->lambda_per_m());
    }
    for (double dt : {.05, .3})
    {
        const RainTripleMomentState initial{};
        const auto source = source_cell(initial, .65, .01, 1e5, .8);
        ExactTripleDynamicWarmCellResult result;
        std::string error;
        const bool accepted = evaluate_exact_triple_dynamic_warm_cell(
            parameters, source, dt, initial, result, error);
        INFO("new rain dt " << dt << ": " << error);
        REQUIRE(accepted);
        check_reference(source, dt, initial, result);
        CHECK(result.producer.original_spectrum() == nullptr);
        CHECK(result.producer.pre_physical_spectrum() == nullptr);
        REQUIRE(result.producer.pre_evaporation_spectrum() != nullptr);
        REQUIRE(find_event(result.producer.producer().events(), WarmRainAcceptedEventKind::Autoconversion) != nullptr);
        CHECK(result.producer.pre_evaporation_spectrum()->alpha() == Approx(2.0).epsilon(3e-12));
        CHECK(result.rain.mass_per_kg > 0);
        CHECK(result.endpoints.cloud_donor_exhausted);
        CHECK(result.endpoints.cloud_per_kg == 0);
    }
}

TEST_CASE("dynamic warm budgets admit complete evaporation small-drop transfer and tiny coherent rain",
          "[physics][milbrandt][triple_dynamic_warm]")
{
    for (long double shape : {0.0L, 2.0L, 20.0L, 200.0L})
    {
        const auto initial = reference::source_rain(1e-9f, 120e-6, shape);
        const auto source = source_cell(initial, .65, 0, 0, 0);
        ExactTripleDynamicWarmCellResult result;
        std::string error;
        const bool accepted = evaluate_exact_triple_dynamic_warm_cell(
            parameters, source, 1e5, initial, result, error);
        INFO("complete evaporation shape " << shape << ": " << error);
        REQUIRE(accepted);
        check_reference(source, 1e5, initial, result);
        CHECK(result.endpoints.evaporation.present);
        CHECK(result.endpoints.evaporation.donor_exhausted);
        CHECK(result.rain.mass_per_kg == 0);
        CHECK(result.rain.number_per_kg == 0);
        CHECK(result.rain.reflectivity_per_kg == 0);
    }
    for (long double shape : {0.0L, 20.0L, 200.0L})
    {
        const auto initial = reference::source_rain(1e-5f, 40e-6, shape);
        const auto source = source_cell(initial, .65, 1e-6, 1e8, 1.1);
        ExactTripleDynamicWarmCellResult result;
        std::string error;
        const bool accepted = evaluate_exact_triple_dynamic_warm_cell(
            parameters, source, .1, initial, result, error);
        INFO("small-drop transfer shape " << shape << ": " << error);
        REQUIRE(accepted);
        check_reference(source, .1, initial, result);
        CHECK(result.endpoints.reclassification.present);
        CHECK(result.producer.pre_evaporation_spectrum() == nullptr);
        CHECK(result.endpoints.cloud_per_kg == source.cloud + source.rain);
        CHECK(result.rain.reflectivity_per_kg == 0);
        CHECK(result.endpoints.temperature_increment_k == 0);
    }
    const auto tiny = reference::source_rain(4e-13f, 90e-6, 20);
    const auto source = source_cell(tiny, .65, 0, 0, .5);
    ExactTripleDynamicWarmCellResult result;
    std::string error;
    REQUIRE(evaluate_exact_triple_dynamic_warm_cell(parameters, source, .1, tiny, result, error));
    check_reference(source, .1, tiny, result);
    CHECK(result.producer.producer().events().size == 0);
    CHECK(result.rain.mass_per_kg == source.rain);
    CHECK(result.rain.number_per_kg == source.rain_number);
    CHECK(result.rain.reflectivity_per_kg == tiny.reflectivity_per_kg);
}

TEST_CASE("dynamic warm original source phase contexts and late failures publish no partial result",
          "[physics][milbrandt][triple_dynamic_warm]")
{
    const auto initial = reference::source_rain(2e-4f, 120e-6, 20);
    const auto source = source_cell(initial, .65);
    ExactTripleDynamicWarmCellResult good;
    std::string error;
    REQUIRE(evaluate_exact_triple_dynamic_warm_cell(parameters, source, .1, initial, good, error));
    for (int mutation = 0; mutation < 24; ++mutation)
    {
        auto producer = good.producer;
        auto altered_source = source;
        auto altered_rain = initial;
        auto altered_parameters = parameters;
        double dt = .1;
        auto& physical = const_cast<ExactRainTripleMomentState&>(producer.pre_physical_rain());
        auto& evaporation = const_cast<ExactRainTripleMomentState&>(producer.pre_evaporation_rain());
        auto& events = const_cast<WarmRainAcceptedEventBuffer&>(producer.producer().events());
        auto& endpoints = const_cast<WarmRainDoubleEndpoints&>(producer.producer().endpoints());
        switch (mutation)
        {
            case 0: altered_source.temperature = std::nextafter(altered_source.temperature, 400.0); break;
            case 1: altered_source.pressure = std::nextafter(altered_source.pressure, 2e5); break;
            case 2: altered_source.density = std::nextafter(altered_source.density, 2.0); break;
            case 3: altered_source.vapor = std::nextafter(altered_source.vapor, 1.0); break;
            case 4: altered_source.cloud = std::nextafter(altered_source.cloud, 1.0); break;
            case 5: altered_source.cloud_number = std::nextafter(altered_source.cloud_number, 1e9); break;
            case 6: altered_source.rain = std::nextafter(altered_source.rain, 1.0); break;
            case 7: altered_source.rain_number = std::nextafter(altered_source.rain_number, 1e9); break;
            case 8: altered_rain.reflectivity_per_kg = std::nextafter(altered_rain.reflectivity_per_kg, 1.0f); break;
            case 9: dt = std::nextafter(dt, 1.0); break;
            case 10: altered_parameters.gamma_ventilation = std::nextafter(altered_parameters.gamma_ventilation, 3.0); break;
            case 11: physical.mass_per_kg *= 2; break;
            case 12: physical.number_per_kg *= 2; break;
            case 13: physical.reflectivity_per_kg *= 2; break;
            case 14: evaporation.mass_per_kg *= 2; break;
            case 15: evaporation.number_per_kg *= 2; break;
            case 16: evaporation.reflectivity_per_kg *= 2; break;
            case 17: events.capacity = 0; break;
            case 18: events.events[events.size - 1].dry_density_kg_m3 *= 2; break;
            case 19: endpoints.evaporation.admitted_mass_per_kg *= 2; break;
            case 20: endpoints.temperature_increment_k *= 2; break;
            case 21: const_cast<RainTripleMomentState&>(producer.original_rain()).reflectivity_per_kg *= 2; break;
            case 22: const_cast<ExactRainTripleMomentState&>(producer.pre_physical_spectrum()->source()).reflectivity_per_kg *= 2; break;
            case 23: const_cast<ExactRainTripleMomentState&>(producer.pre_evaporation_spectrum()->source()).reflectivity_per_kg *= 2; break;
        }
        auto output = good;
        INFO("capture mutation " << mutation);
        CHECK_FALSE(apply_exact_triple_dynamic_warm_cell(altered_parameters,
            altered_source, dt, altered_rain, producer, output, error));
        CHECK_FALSE(error.empty());
        compare_unchanged(output, good);
        REQUIRE(apply_exact_triple_dynamic_warm_cell(parameters, source, .1,
            initial, good.producer, output, error));
        compare_unchanged(output, good);
    }
    for (std::size_t capacity : {0u, 1u, 17u})
    {
        auto output = good;
        CHECK_FALSE(evaluate_exact_triple_dynamic_warm_cell(parameters, source, .1,
            initial, output, error, capacity));
        compare_unchanged(output, good);
    }
    for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()})
    {
        auto output = good;
        CHECK_FALSE(evaluate_exact_triple_dynamic_warm_cell(parameters, source, invalid,
            initial, output, error));
        compare_unchanged(output, good);
    }
    for (int field = 0; field < 8; ++field)
    {
        auto bad = source;
        const std::array<double*, 8> values{&bad.temperature, &bad.pressure, &bad.density,
            &bad.vapor, &bad.cloud, &bad.rain, &bad.cloud_number, &bad.rain_number};
        *values[field] = std::numeric_limits<double>::quiet_NaN();
        auto output = good;
        CHECK_FALSE(evaluate_exact_triple_dynamic_warm_cell(parameters, bad, .1,
            initial, output, error));
        compare_unchanged(output, good);
    }
    const RainTripleMomentState empty{};
    const auto dry = source_cell(empty, .65, 0, 0, 0);
    ExactTripleDynamicWarmCellResult dry_result;
    REQUIRE(evaluate_exact_triple_dynamic_warm_cell(parameters, dry, .1, empty, dry_result, error));
    CHECK(dry_result.producer.producer().events().size == 0);
    CHECK(dry_result.producer.original_spectrum() == nullptr);
    CHECK(dry_result.producer.pre_physical_spectrum() == nullptr);
    CHECK(dry_result.producer.pre_evaporation_spectrum() == nullptr);
    for (int field = 0; field < 8; ++field)
    {
        auto bad = dry;
        const std::array<double*, 8> values{&bad.temperature, &bad.pressure, &bad.density,
            &bad.vapor, &bad.cloud, &bad.rain, &bad.cloud_number, &bad.rain_number};
        *values[field] = std::numeric_limits<double>::quiet_NaN();
        auto output = good;
        CHECK_FALSE(evaluate_exact_triple_dynamic_warm_cell(parameters, bad, .1, empty, output, error));
        compare_unchanged(output, good);
    }
    auto altered_empty = empty;
    altered_empty.reflectivity_per_kg = std::numeric_limits<float>::denorm_min();
    auto output = good;
    CHECK_FALSE(apply_exact_triple_dynamic_warm_cell(parameters, dry, .1,
        altered_empty, dry_result.producer, output, error));
    compare_unchanged(output, good);
}

TEST_CASE("dynamic warm admitted continuations are identical with one and four actual workers",
          "[physics][milbrandt][triple_dynamic_warm]")
{
    constexpr int cells = 24;
    constexpr int steps = 3;
    std::array<std::array<std::array<ExactTripleDynamicWarmCellResult, cells>, steps>, 2> outputs{};
    std::array<std::array<bool, cells>, 2> passed{};
    std::array<std::array<std::string, cells>, 2> errors{};
    std::array<int, 2> observed{1, 1};
    for (int run = 0; run < 2; ++run)
    {
        const int workers = run == 0 ? 1 : 4;
        #pragma omp parallel num_threads(workers)
        {
            #ifdef _OPENMP
            #pragma omp single
            observed[run] = omp_get_num_threads();
            #endif
            #pragma omp for
            for (int index = 0; index < cells; ++index)
            {
                const std::array<long double, 4> shapes{0, 2, 20, 200};
                const std::array<double, 3> diameters{90e-6, 120e-6, 1e-3};
                auto rain = reference::source_rain(2e-4f, diameters[(index / 4) % 3], shapes[index % 4]);
                auto source = source_cell(rain, index < 12 ? .65 : 1.43);
                passed[run][index] = true;
                for (int step = 0; step < steps; ++step)
                {
                    auto& result = outputs[run][step][index];
                    if (!evaluate_exact_triple_dynamic_warm_cell(parameters, source, .01,
                        rain, result, errors[run][index]))
                    {
                        passed[run][index] = false;
                        break;
                    }
                    source.temperature += result.endpoints.temperature_increment_k;
                    source.vapor = result.endpoints.vapor_per_kg;
                    source.cloud = result.endpoints.cloud_per_kg;
                    source.cloud_number = result.endpoints.cloud_number_per_kg;
                    rain = result.rounded_rain;
                    source.rain = rain.mass_per_kg;
                    source.rain_number = rain.number_per_kg;
                }
            }
        }
        CHECK(observed[run] == workers);
    }
    for (int index = 0; index < cells; ++index)
    {
        INFO("continuation cell " << index << ", one-worker " << errors[0][index]
            << ", four-worker " << errors[1][index]);
        REQUIRE(passed[0][index]);
        REQUIRE(passed[1][index]);
        for (int step = 0; step < steps; ++step)
        {
            compare_unchanged(outputs[0][step][index], outputs[1][step][index]);
        }
    }
}

TEST_CASE("composed dynamic cloud and rain bind unrounded admission and independent thermal moment closure",
          "[physics][milbrandt][triple_dynamic_warm][triple_dynamic_warm_cloud]")
{
    bool saw_unrounded = false;
    bool saw_activation = false;
    bool saw_new_rain = false;
    bool saw_existing_rain = false;
    bool saw_cloud_evaporation = false;
    for (double temperature : {280.125, 290.125})
    {
        for (double density : {.65, 1.43})
        {
            for (long double nominal : {0.0L, 20.0L, 200.0L})
            {
                const auto existing = reference::source_rain(2e-4f, 120e-6, nominal);
                for (bool new_rain : {false, true})
                {
                    const auto initial = new_rain ? RainTripleMomentState{} : existing;
                    for (double humidity : {.8, 1.2})
                    {
                        auto source = source_cell(initial, density,
                            new_rain ? .01 : .001, new_rain ? 1e5 : 1e8, humidity);
                        source.temperature = temperature;
                        source.vapor = humidity * thermodynamics::saturation_mixing_ratio_water(temperature, source.pressure);
                        for (double dt : {.01, .1})
                        {
                            ExactTripleDynamicWarmCloudCellResult result;
                            std::string error;
                            const bool accepted = produce_exact_triple_dynamic_warm_cloud_cell(
                                parameters, source, .5, dt, initial, result, error);
                            INFO("composed T " << temperature << ", rho " << density << ", shape " << nominal
                                << ", new rain " << new_rain << ", RH " << humidity << ", dt " << dt << ": " << error);
                            REQUIRE(accepted);
                            check_composed_reference(source, .5, dt, initial, result);
                            const auto& input = result.warm.producer.producer().source_cell();
                            saw_unrounded |= input.temperature != static_cast<double>(static_cast<float>(input.temperature)) ||
                                input.vapor != static_cast<double>(static_cast<float>(input.vapor)) ||
                                input.cloud != static_cast<double>(static_cast<float>(input.cloud)) ||
                                input.cloud_number != static_cast<double>(static_cast<float>(input.cloud_number));
                            saw_activation |= result.cloud.endpoints().activated_number_per_m3 > 0;
                            saw_new_rain |= new_rain && result.rain.mass_per_kg > 0;
                            saw_existing_rain |= !new_rain && result.warm.producer.original_spectrum() != nullptr;
                            saw_cloud_evaporation |= result.cloud.endpoints().admitted_cloud_increment_per_kg < 0;
                            ExactTripleDynamicWarmCloudCellResult replay;
                            REQUIRE(apply_exact_triple_dynamic_warm_cloud_cell(parameters, source, .5, dt,
                                initial, result, replay, error));
                            compare_composed_unchanged(replay, result);
                        }
                    }
                }
            }
        }
    }
    CHECK(saw_unrounded);
    CHECK(saw_activation);
    CHECK(saw_new_rain);
    CHECK(saw_existing_rain);
    CHECK(saw_cloud_evaporation);
}

TEST_CASE("composed dynamic phase transplants altered budgets and source wind reject without publication",
          "[physics][milbrandt][triple_dynamic_warm][triple_dynamic_warm_cloud]")
{
    const auto initial = reference::source_rain(2e-4f, 120e-6, 20);
    const auto source = source_cell(initial, .65, .001, 1e8, 1.2);
    ExactTripleDynamicWarmCloudCellResult good;
    ExactTripleDynamicWarmCloudCellResult different_wind;
    ExactTripleDynamicWarmCloudCellResult different_shape;
    std::string error;
    REQUIRE(produce_exact_triple_dynamic_warm_cloud_cell(parameters, source, .5, .1, initial, good, error));
    REQUIRE(produce_exact_triple_dynamic_warm_cloud_cell(parameters, source, 1.5, .1, initial, different_wind, error));
    const auto other_rain = reference::source_rain(initial.mass_per_kg, 120e-6, 200);
    REQUIRE(other_rain.number_per_kg == initial.number_per_kg);
    REQUIRE(produce_exact_triple_dynamic_warm_cloud_cell(parameters, source, .5, .1,
        other_rain, different_shape, error));
    for (int mutation = 0; mutation < 24; ++mutation)
    {
        auto capture = good;
        auto original = source;
        auto original_rain = initial;
        double wind = .5;
        double dt = .1;
        auto& cloud = const_cast<TripleCloudAdjustmentEndpoints&>(capture.cloud.endpoints());
        auto& warm_endpoints = const_cast<WarmRainDoubleEndpoints&>(capture.warm.producer.producer().endpoints());
        switch (mutation)
        {
            case 0: capture.cloud = different_wind.cloud; break;
            case 1: capture.warm = different_wind.warm; break;
            case 2: capture.warm.producer = different_shape.warm.producer; break;
            case 3: capture.warm = different_shape.warm; capture.endpoints = different_shape.endpoints;
                    capture.rain = different_shape.rain; capture.rounded_rain = different_shape.rounded_rain;
                    capture.temperature_increment_k = different_shape.temperature_increment_k; break;
            case 4: cloud.cloud_per_kg = std::nextafter(cloud.cloud_per_kg, 1.0); break;
            case 5: cloud.cloud_number_per_kg *= 2; break;
            case 6: cloud.temperature_increment_k *= 2; break;
            case 7: cloud.admitted_cloud_increment_per_kg *= 2; break;
            case 8: warm_endpoints.cloud_donor_budget_per_kg += 1e-4; break;
            case 9: warm_endpoints.evaporation.present = !warm_endpoints.evaporation.present; break;
            case 10: capture.warm.endpoints.vapor_per_kg = std::nextafter(capture.warm.endpoints.vapor_per_kg, 1.0); break;
            case 11: capture.warm.rain.reflectivity_per_kg *= 2; break;
            case 12: capture.warm.rounded_rain.reflectivity_per_kg *= 2; break;
            case 13: capture.warm.rates.temperature = std::nextafter(capture.warm.rates.temperature, 1.0f); break;
            case 14: capture.endpoints.cloud_per_kg = std::nextafter(capture.endpoints.cloud_per_kg, 1.0); break;
            case 15: capture.rain.reflectivity_per_kg *= 2; break;
            case 16: capture.rounded_rain.reflectivity_per_kg *= 2; break;
            case 17: capture.temperature_increment_k = std::nextafter(capture.temperature_increment_k, 1.0); break;
            case 18: wind = std::nextafter(wind, 1.0); break;
            case 19: dt = std::nextafter(dt, 1.0); break;
            case 20: original_rain.reflectivity_per_kg = std::nextafter(original_rain.reflectivity_per_kg, 1.0f); break;
            case 21: original.temperature = std::nextafter(original.temperature, 400.0); break;
            case 22: const_cast<ExactRainTripleMomentState&>(capture.warm.producer.pre_evaporation_rain()).reflectivity_per_kg *= 2; break;
            case 23: const_cast<WarmRainAcceptedEventBuffer&>(capture.warm.producer.producer().events()).capacity = 0; break;
        }
        auto output = good;
        INFO("composed capture mutation " << mutation);
        CHECK_FALSE(apply_exact_triple_dynamic_warm_cloud_cell(parameters, original, wind,
            dt, original_rain, capture, output, error));
        CHECK_FALSE(error.empty());
        compare_composed_unchanged(output, good);
        REQUIRE(apply_exact_triple_dynamic_warm_cloud_cell(parameters, source, .5, .1,
            initial, good, output, error));
        compare_composed_unchanged(output, good);
    }
    for (std::size_t capacity : {0u, 1u, 17u})
    {
        auto output = good;
        CHECK_FALSE(produce_exact_triple_dynamic_warm_cloud_cell(parameters, source, .5,
            .1, initial, output, error, capacity));
        compare_composed_unchanged(output, good);
    }
    for (double wind : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        auto output = good;
        CHECK_FALSE(produce_exact_triple_dynamic_warm_cloud_cell(parameters, source, wind,
            .1, initial, output, error));
        compare_composed_unchanged(output, good);
    }
}
