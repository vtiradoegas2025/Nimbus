#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/triple_warm_adapter.hpp"
#include "microphysics/schemes/milbrandt/triple_warm_primary.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{

using namespace tmv::milbrandt_detail;

constexpr double kRainMassCoefficient = 523.598775598;

float initial_z(double mass, double number, double gamma_factor)
{
    return static_cast<float>(
        gamma_factor / (kRainMassCoefficient * kRainMassCoefficient) *
        mass * mass / number);
}

bool close(double actual, double expected, double relative = 2.0e-6)
{
    return std::abs(actual - expected) <=
        relative * std::max(std::abs(expected), 1.0e-30);
}

AcceptedRainEvent make_auto(double mass_rate, double number_rate_per_kg,
                            double density, double dt)
{
    AcceptedRainEvent event;
    std::string error;
    REQUIRE(make_admitted_autoconversion_event(
        mass_rate, number_rate_per_kg * density, density,
        dt, event, error));
    return event;
}

AcceptedRainEvent make_same(AcceptedRainEventKind kind,
                            double mass_rate, double number_rate,
                            double dt)
{
    AcceptedRainEvent event;
    std::string error;
    REQUIRE(make_admitted_same_category_event(
        kind, mass_rate, number_rate, dt, event, error));
    return event;
}

double independent_triple_warm_z(
    const RainTripleMomentState& initial,
    const WarmRainAcceptedEventBuffer& observed,
    double dt,
    double density,
    bool& saw_accretion,
    bool& saw_self_collection,
    double& autoconversion_b4)
{
    double q = initial.mass_per_kg;
    double n = initial.number_per_kg;
    double z = initial.reflectivity_per_kg;
    for (std::size_t index = 0; index < observed.size; ++index)
    {
        const auto& event = observed.events[index];
        if (event.kind ==
            WarmRainAcceptedEventKind::PreRainNumberRegularization)
        {
            const double next_n = n +
                dt * event.rain_number_rate_per_m3_s / density;
            z *= n / next_n;
            n = next_n;
        }
    }
    const double pre_q = q;
    const double pre_n = n;
    const double pre_z = z;
    double q_rate = 0.0;
    double n_rate = 0.0;
    double z_rate = 0.0;
    const double gamma_factor =
        std::tgamma(9.0) * std::tgamma(3.0)
        / (std::tgamma(6.0) * std::tgamma(6.0));
    for (std::size_t index = 0; index < observed.size; ++index)
    {
        const auto& event = observed.events[index];
        if (event.kind == WarmRainAcceptedEventKind::Autoconversion)
        {
            const double rate_n =
                event.rain_number_rate_per_m3_s / density;
            autoconversion_b4 = gamma_factor
                / (kRainMassCoefficient * kRainMassCoefficient)
                * event.rain_mass_rate_per_kg_s
                * event.rain_mass_rate_per_kg_s / rate_n;
            z_rate += autoconversion_b4;
            q_rate += event.rain_mass_rate_per_kg_s;
            n_rate += rate_n;
        }
        else if (event.kind == WarmRainAcceptedEventKind::Accretion ||
                 event.kind == WarmRainAcceptedEventKind::RainSelfCollection)
        {
            saw_accretion |= event.kind ==
                WarmRainAcceptedEventKind::Accretion;
            saw_self_collection |= event.kind ==
                WarmRainAcceptedEventKind::RainSelfCollection;
            const double rate_n =
                event.rain_number_rate_per_m3_s / density;
            z_rate += pre_z *
                (2.0 * event.rain_mass_rate_per_kg_s / pre_q
                 - rate_n / pre_n);
            q_rate += event.rain_mass_rate_per_kg_s;
            n_rate += rate_n;
        }
    }
    q += dt * q_rate;
    n += dt * n_rate;
    z += dt * z_rate;
    for (std::size_t index = 0; index < observed.size; ++index)
    {
        const auto& event = observed.events[index];
        if (event.kind ==
            WarmRainAcceptedEventKind::PostRainNumberRegularization)
        {
            const double next_n = n +
                dt * event.rain_number_rate_per_m3_s / density;
            z *= n / next_n;
            n = next_n;
        }
        else if (event.kind == WarmRainAcceptedEventKind::Evaporation)
        {
            const double rate_n =
                event.rain_number_rate_per_m3_s / density;
            const double b3 = z *
                (2.0 * event.rain_mass_rate_per_kg_s / q
                 - rate_n / n);
            q += dt * event.rain_mass_rate_per_kg_s;
            n += dt * rate_n;
            z += dt * b3;
        }
    }
    return z;
}

} // namespace

TEST_CASE("simultaneous rain B3/B4 channels use one pre-physical q/N/Z state",
          "[physics][milbrandt][triple_warm_phase]")
{
    constexpr double dt = 0.1;
    const RainTripleMomentState initial{
        1.0e-4f, 1.0e4f, initial_z(1.0e-4, 1.0e4, 20.0)};
    const AcceptedRainEvent autoconversion = make_auto(
        2.0e-6, 100.0, 1.2, dt);
    const AcceptedRainEvent accretion = make_same(
        AcceptedRainEventKind::Accretion, 2.0e-6, 0.0, dt);
    const AcceptedRainEvent self_collection = make_same(
        AcceptedRainEventKind::SelfCollection, 0.0, -300.0, dt);
    AcceptedRainPhaseBatch batch;
    batch.duration_s = dt;
    std::string error;
    for (const auto& event : {autoconversion, accretion, self_collection})
    {
        REQUIRE(append_admitted_rain_phase_event(
            batch, RainEventPhase::SimultaneousPhysical,
            event, error));
    }
    RainTripleMomentState phase{-7.0f, -7.0f, -7.0f};
    REQUIRE(apply_admitted_rain_phases(initial, batch, phase, error));
    const double b4 = 5.6 /
        (kRainMassCoefficient * kRainMassCoefficient) *
        autoconversion.admitted_mass_rate_per_kg_s *
        autoconversion.admitted_mass_rate_per_kg_s /
        autoconversion.admitted_number_rate_per_kg_s;
    const double b3_acc = initial.reflectivity_per_kg *
        2.0 * accretion.admitted_mass_rate_per_kg_s /
        initial.mass_per_kg;
    const double b3_self = -initial.reflectivity_per_kg *
        self_collection.admitted_number_rate_per_kg_s /
        initial.number_per_kg;
    const double expected_z = initial.reflectivity_per_kg +
        dt * (b4 + b3_acc + b3_self);
    CHECK(close(phase.mass_per_kg, initial.mass_per_kg +
        dt * (autoconversion.admitted_mass_rate_per_kg_s +
              accretion.admitted_mass_rate_per_kg_s)));
    CHECK(close(phase.number_per_kg, initial.number_per_kg +
        dt * (autoconversion.admitted_number_rate_per_kg_s +
              self_collection.admitted_number_rate_per_kg_s)));
    CHECK(close(phase.reflectivity_per_kg, expected_z));

    RainTripleMomentState sequential{};
    REQUIRE(apply_admitted_rain_events(
        initial, {autoconversion, accretion, self_collection},
        sequential, error));
    CHECK(std::abs(phase.reflectivity_per_kg -
                   sequential.reflectivity_per_kg) >
          1.0e-6 * phase.reflectivity_per_kg);
}

TEST_CASE("phase order, duplicate channel and late failure leave output unpublished",
          "[physics][milbrandt][triple_warm_phase]")
{
    constexpr double dt = 0.1;
    const RainTripleMomentState initial{
        1.0e-4f, 1.0e4f, initial_z(1.0e-4, 1.0e4, 20.0)};
    const auto auto_event = make_auto(2.0e-6, 100.0, 1.2, dt);
    const auto accretion = make_same(
        AcceptedRainEventKind::Accretion, 1.0e-6, 0.0, dt);
    AcceptedRainEvent pre;
    std::string error;
    REQUIRE(make_admitted_rain_number_correction_event(
        AcceptedRainEventKind::PreRainNumberRegularization,
        100.0, 1.2, dt, pre, error));
    AcceptedRainPhaseBatch batch;
    batch.duration_s = dt;
    REQUIRE(append_admitted_rain_phase_event(
        batch, RainEventPhase::SimultaneousPhysical,
        auto_event, error));
    const std::size_t before_count = batch.size;
    CHECK_FALSE(append_admitted_rain_phase_event(
        batch, RainEventPhase::PreRegularization,
        pre, error));
    CHECK(batch.size == before_count);

    REQUIRE(append_admitted_rain_phase_event(
        batch, RainEventPhase::SimultaneousPhysical,
        accretion, error));
    RainTripleMomentState output{-7.0f, -7.0f, -7.0f};
    auto duplicate = batch;
    REQUIRE(append_admitted_rain_phase_event(
        duplicate, RainEventPhase::SimultaneousPhysical,
        accretion, error));
    CHECK_FALSE(apply_admitted_rain_phases(
        initial, duplicate, output, error));
    CHECK(error.find("duplicate") != std::string::npos);
    CHECK(output.mass_per_kg == -7.0f);
    CHECK(output.number_per_kg == -7.0f);
    CHECK(output.reflectivity_per_kg == -7.0f);

    auto tampered_order = batch;
    tampered_order.events[1].phase = RainEventPhase::PreRegularization;
    CHECK_FALSE(apply_admitted_rain_phases(
        initial, tampered_order, output, error));
    CHECK(output.mass_per_kg == -7.0f);
    auto tampered_kind = batch;
    tampered_kind.events[1].event.kind =
        AcceptedRainEventKind::RainNumberClip;
    CHECK_FALSE(apply_admitted_rain_phases(
        initial, tampered_kind, output, error));
    CHECK(output.reflectivity_per_kg == -7.0f);

    auto overflow = batch;
    overflow.events[1].event.admitted_mass_rate_per_kg_s =
        std::numeric_limits<double>::max();
    CHECK_FALSE(apply_admitted_rain_phases(
        initial, overflow, output, error));
    CHECK(output.mass_per_kg == -7.0f);
}

TEST_CASE("rain numerical correction, removal and rounded Jensen guards stay atomic",
          "[physics][milbrandt][triple_warm_phase]")
{
    constexpr double dt = 0.1;
    const RainTripleMomentState initial{
        1.0e-4f, 1.0e4f, initial_z(1.0e-4, 1.0e4, 20.0)};
    AcceptedRainEvent pre;
    AcceptedRainEvent post;
    AcceptedRainEvent evaporation;
    std::string error;
    REQUIRE(make_admitted_rain_number_correction_event(
        AcceptedRainEventKind::PreRainNumberRegularization,
        120.0, 1.2, dt, pre, error));
    REQUIRE(make_admitted_rain_number_correction_event(
        AcceptedRainEventKind::PostRainNumberRegularization,
        60.0, 1.2, dt, post, error));
    REQUIRE(make_admitted_same_category_event(
        AcceptedRainEventKind::Evaporation,
        -1.0e-6, -10.0, dt, evaporation, error));
    AcceptedRainPhaseBatch batch;
    batch.duration_s = dt;
    REQUIRE(append_admitted_rain_phase_event(
        batch, RainEventPhase::PreRegularization, pre, error));
    REQUIRE(append_admitted_rain_phase_event(
        batch, RainEventPhase::NumericalCorrection, post, error));
    REQUIRE(append_admitted_rain_phase_event(
        batch, RainEventPhase::Evaporation, evaporation, error));
    RainTripleMomentState result{-7.0f, -7.0f, -7.0f};
    REQUIRE(apply_admitted_rain_phases(initial, batch, result, error));
    const double n_pre = initial.number_per_kg +
        dt * pre.admitted_number_rate_per_kg_s;
    const double z_pre = initial.reflectivity_per_kg *
        initial.number_per_kg / n_pre;
    const double n_post = n_pre +
        dt * post.admitted_number_rate_per_kg_s;
    const double z_post = z_pre * n_pre / n_post;
    const double z_evap = z_post + dt * z_post *
        (2.0 * evaporation.admitted_mass_rate_per_kg_s /
             initial.mass_per_kg -
         evaporation.admitted_number_rate_per_kg_s / n_post);
    CHECK(close(result.mass_per_kg,
                initial.mass_per_kg + dt *
                    evaporation.admitted_mass_rate_per_kg_s));
    CHECK(close(result.number_per_kg,
                n_post + dt *
                    evaporation.admitted_number_rate_per_kg_s));
    CHECK(close(result.reflectivity_per_kg, z_evap));

    for (const auto kind : {AcceptedRainEventKind::RainToCloud,
                            AcceptedRainEventKind::RainReset})
    {
        AcceptedRainEvent removal;
        const double cloud_mass_rate = kind ==
            AcceptedRainEventKind::RainToCloud
            ? initial.mass_per_kg / dt : 0.0;
        const double cloud_number_rate = kind ==
            AcceptedRainEventKind::RainToCloud
            ? initial.number_per_kg * 1.2 / dt : 0.0;
        REQUIRE(make_admitted_rain_removal_event(
            kind, cloud_mass_rate,
            -initial.mass_per_kg / dt, cloud_number_rate,
            -initial.number_per_kg * 1.2 / dt,
            1.2, dt, removal, error));
        AcceptedRainPhaseBatch removal_batch;
        removal_batch.duration_s = dt;
        REQUIRE(append_admitted_rain_phase_event(
            removal_batch, RainEventPhase::NumericalCorrection,
            removal, error));
        RainTripleMomentState empty{-7.0f, -7.0f, -7.0f};
        REQUIRE(apply_admitted_rain_phases(
            initial, removal_batch, empty, error));
        CHECK(empty.mass_per_kg == 0.0f);
        CHECK(empty.number_per_kg == 0.0f);
        CHECK(empty.reflectivity_per_kg == 0.0f);
    }

    AcceptedRainEvent exhausting_clip;
    REQUIRE(make_admitted_rain_number_correction_event(
        AcceptedRainEventKind::RainNumberClip,
        -initial.number_per_kg * 1.2 / dt,
        1.2, dt, exhausting_clip, error));
    AcceptedRainPhaseBatch invalid;
    invalid.duration_s = dt;
    REQUIRE(append_admitted_rain_phase_event(
        invalid, RainEventPhase::NumericalCorrection,
        exhausting_clip, error));
    CHECK_FALSE(apply_admitted_rain_phases(
        initial, invalid, result, error));
    CHECK(error.find("clip/reset grouping") != std::string::npos);
    CHECK(result.mass_per_kg > 0.0f);
    AcceptedRainEvent reset_after_clip;
    REQUIRE(make_admitted_rain_removal_event(
        AcceptedRainEventKind::RainReset, 0.0,
        -initial.mass_per_kg / dt, 0.0,
        -initial.number_per_kg * 1.2 / dt,
        1.2, dt, reset_after_clip, error));
    REQUIRE(append_admitted_rain_phase_event(
        invalid, RainEventPhase::NumericalCorrection,
        reset_after_clip, error));
    RainTripleMomentState grouped_output{-7.0f, -7.0f, -7.0f};
    CHECK_FALSE(apply_admitted_rain_phases(
        initial, invalid, grouped_output, error));
    CHECK(error.find("clip/reset grouping") != std::string::npos);
    CHECK(grouped_output.mass_per_kg == -7.0f);
    CHECK(grouped_output.number_per_kg == -7.0f);
    CHECK(grouped_output.reflectivity_per_kg == -7.0f);

    bool found_rounding_neighbor = false;
    RainTripleMomentState unpublished{-7.0f, -7.0f, -7.0f};
    for (float mass : {1.0e-4f, 1.0e-5f, 2.0e-4f, 7.0e-6f})
    {
        for (float number : {1.0e3f, 1.0e4f, 1.0e5f})
        {
            const double lower = static_cast<double>(number) *
                std::pow(static_cast<double>(mass) /
                    (kRainMassCoefficient * number), 2.0);
            const double reflectivity = std::nextafter(
                lower, std::numeric_limits<double>::infinity());
            unpublished = {-7.0f, -7.0f, -7.0f};
            if (!finalize_rain_phase_endpoint(
                    mass, number, reflectivity,
                    unpublished, error) &&
                error.find("rounded rain phase endpoint") !=
                    std::string::npos)
            {
                found_rounding_neighbor = true;
                break;
            }
        }
        if (found_rounding_neighbor)
        {
            break;
        }
    }
    REQUIRE(found_rounding_neighbor);
    CHECK(unpublished.mass_per_kg == -7.0f);
    CHECK(unpublished.number_per_kg == -7.0f);
    CHECK(unpublished.reflectivity_per_kg == -7.0f);
}

TEST_CASE("phase-aware simultaneous rain batches are byte-identical at one and four workers",
          "[physics][milbrandt][triple_warm_phase]")
{
    constexpr int cells = 16;
    constexpr double dt = 0.1;
    AcceptedRainPhaseBatch batch;
    batch.duration_s = dt;
    std::string error;
    const auto autoconversion = make_auto(1.0e-6, 100.0, 1.2, dt);
    const auto accretion = make_same(
        AcceptedRainEventKind::Accretion, 1.0e-6, 0.0, dt);
    const auto self = make_same(
        AcceptedRainEventKind::SelfCollection, 0.0, -100.0, dt);
    for (const auto& event : {autoconversion, accretion, self})
    {
        REQUIRE(append_admitted_rain_phase_event(
            batch, RainEventPhase::SimultaneousPhysical,
            event, error));
    }
    std::array<RainTripleMomentState, cells> initial{};
    for (int cell = 0; cell < cells; ++cell)
    {
        const double mass = 1.0e-4 + cell * 1.0e-7;
        const double number = 1.0e4 + cell * 10.0;
        initial[cell] = {static_cast<float>(mass),
                         static_cast<float>(number),
                         initial_z(mass, number, 20.0)};
    }
    std::array<std::array<RainTripleMomentState, cells>, 2> outputs{};
    std::array<std::array<bool, cells>, 2> passed{};
    for (int run = 0; run < 2; ++run)
    {
        const int workers = run == 0 ? 1 : 4;
        #pragma omp parallel for num_threads(workers)
        for (int cell = 0; cell < cells; ++cell)
        {
            std::string local_error;
            passed[run][cell] = apply_admitted_rain_phases(
                initial[cell], batch, outputs[run][cell], local_error);
        }
    }
    for (int cell = 0; cell < cells; ++cell)
    {
        REQUIRE(passed[0][cell]);
        REQUIRE(passed[1][cell]);
        CHECK(std::memcmp(&outputs[0][cell], &outputs[1][cell],
                          sizeof(RainTripleMomentState)) == 0);
    }
}

TEST_CASE("warm adapter accepts coherent rain cell and rejects limiter mismatch atomically",
          "[physics][milbrandt][triple_warm_adapter]")
{
    const auto parameters = warm_rain_parameters(
        0.0, 523.598775598, 3.0, 149.1, 0.5);
    const float saturation = static_cast<float>(
        thermodynamics::saturation_mixing_ratio_water(
            280.0f, 90000.0f));
    bool found = false;
    bool found_combined = false;
    std::string last_combined_error;
    for (double dt : {0.05, 0.2, 1.0})
    {
        for (float cloud : {0.001f, 0.003f, 0.01f})
        {
            for (float rain : {0.0f, 0.0001f, 0.0005f, 0.002f})
            {
                for (float cloud_number : {1.0e5f, 1.0e7f, 1.0e9f})
                {
                    for (float rain_number : {0.0f, 1.0e3f, 1.0e5f})
                    {
                        if ((rain == 0.0f) != (rain_number == 0.0f))
                        {
                            continue;
                        }
                        const WarmRainCell cell{
                            280.0f, 90000.0f, 1.0f,
                            0.5f * saturation, cloud, rain,
                            cloud_number, rain_number};
                        const RainTripleMomentState initial{
                            rain, rain_number,
                            rain == 0.0f ? 0.0f :
                                initial_z(rain, rain_number, 20.0)};
                        const WarmRainTendencies incoming{};
                        WarmRainTendencies rates{};
                        RainTripleMomentState result{};
                        std::string error;
                        WarmRainTendencies source_rates{};
                        WarmRainAcceptedEventBuffer observed;
                        warm_rain_cell(parameters, cell, dt,
                                       source_rates, &observed);
                        bool has_auto = false;
                        bool has_accretion = false;
                        bool has_self = false;
                        bool coherent = true;
                        for (std::size_t event = 0; event < observed.size;
                             ++event)
                        {
                            const auto& value = observed.events[event];
                            has_auto |= value.kind ==
                                WarmRainAcceptedEventKind::Autoconversion;
                            has_accretion |= value.kind ==
                                WarmRainAcceptedEventKind::Accretion;
                            has_self |= value.kind ==
                                WarmRainAcceptedEventKind::RainSelfCollection;
                            if (value.kind ==
                                WarmRainAcceptedEventKind::Autoconversion)
                            {
                                coherent &= value.seed_mass_number_coherent;
                            }
                        }
                        const bool accepted =
                            evaluate_observed_triple_warm_cell(
                                parameters, cell, dt, incoming, initial,
                                rates, result, error);
                        if (accepted &&
                            (has_auto || has_accretion || has_self))
                        {
                            found = true;
                            CHECK(result.mass_per_kg >= 0.0f);
                            CHECK(result.number_per_kg >= 0.0f);
                            CHECK(result.reflectivity_per_kg >= 0.0f);
                        }
                        if (has_auto && has_accretion && has_self &&
                            coherent)
                        {
                            if (accepted)
                            {
                                found_combined = true;
                                std::cout << std::setprecision(17)
                                          << "[triple-warm-combined] dt=" << dt
                                          << " cloud=" << cloud
                                          << " rain=" << rain
                                          << " cloud_number=" << cloud_number
                                          << " rain_number=" << rain_number
                                          << " initial_z="
                                          << initial.reflectivity_per_kg
                                          << " final_q=" << result.mass_per_kg
                                          << " final_n=" << result.number_per_kg
                                          << " final_z="
                                          << result.reflectivity_per_kg
                                          << '\n';
                            }
                            else
                            {
                                last_combined_error = error;
                            }
                        }
                    }
                    if (found_combined)
                    {
                        break;
                    }
                }
                if (found_combined)
                {
                    break;
                }
            }
            if (found_combined)
            {
                break;
            }
        }
        if (found_combined)
        {
            break;
        }
    }
    REQUIRE(found);
    INFO(last_combined_error);
    REQUIRE(found_combined);

    const WarmRainCell incoherent{
        280.0f, 90000.0f, 1.0f,
        0.5f * saturation, 0.01f, 0.002f,
        1.0e5f, 1.0e5f};
    const RainTripleMomentState initial{
        incoherent.rain, incoherent.rain_number,
        initial_z(incoherent.rain, incoherent.rain_number, 20.0)};
    WarmRainTendencies rates{-7.0f, -7.0f, -7.0f,
                             -7.0f, -7.0f, -7.0f};
    RainTripleMomentState result{-7.0f, -7.0f, -7.0f};
    std::string error;
    CHECK_FALSE(evaluate_observed_triple_warm_cell(
        parameters, incoherent, 10.0, {}, initial,
        rates, result, error));
    CHECK(error.find("incoherent") != std::string::npos);
    CHECK(rates.rain == -7.0f);
    CHECK(result.mass_per_kg == -7.0f);
    CHECK(result.number_per_kg == -7.0f);
    CHECK(result.reflectivity_per_kg == -7.0f);
}

TEST_CASE("warm adapter rejects unsupported observer channels without phase publication",
          "[physics][milbrandt][triple_warm_adapter]")
{
    AcceptedRainPhaseBatch batch;
    batch.duration_s = 0.2;
    WarmRainAcceptedEvent observed;
    observed.dry_density_kg_m3 = 1.0;
    std::string error;
    for (const auto kind : {
             WarmRainAcceptedEventKind::InitialRainNumberClamp,
             WarmRainAcceptedEventKind::InitialCloudNumberClamp,
             WarmRainAcceptedEventKind::CloudMassClip,
             WarmRainAcceptedEventKind::RainMassClip})
    {
        observed.kind = kind;
        CHECK_FALSE(map_observed_rain_event(
            observed, 0.2, 1.0, batch, error));
        CHECK(error.find("unsupported") != std::string::npos);
        CHECK(batch.size == 0);
    }
    observed.kind = WarmRainAcceptedEventKind::CloudSelfCollection;
    observed.rain_mass_rate_per_kg_s = 1.0e-6;
    CHECK_FALSE(map_observed_rain_event(
        observed, 0.2, 1.0, batch, error));
    CHECK(batch.size == 0);
    observed.rain_mass_rate_per_kg_s = 0.0;
    observed.cloud_number_rate_per_m3_s = -100.0;
    observed.seed_diameter_m = 1.0e-4;
    CHECK_FALSE(map_observed_rain_event(
        observed, 0.2, 1.0, batch, error));
    CHECK(error.find("seed or donor metadata") != std::string::npos);
    CHECK(batch.size == 0);
    observed.seed_diameter_m = 0.0;
    observed.cloud_donor_scale = 0.5;
    CHECK_FALSE(map_observed_rain_event(
        observed, 0.2, 1.0, batch, error));
    CHECK(batch.size == 0);
    observed.cloud_donor_scale = 1.0;
    observed.cloud_number_rate_per_m3_s = 100.0;
    CHECK_FALSE(map_observed_rain_event(
        observed, 0.2, 1.0, batch, error));
    CHECK(error.find("cloud-only") != std::string::npos);
    CHECK(batch.size == 0);

    observed = {};
    observed.kind = WarmRainAcceptedEventKind::Evaporation;
    observed.dry_density_kg_m3 = 1.0;
    observed.rain_mass_rate_per_kg_s = -1.0e-6;
    observed.vapor_mass_rate_per_kg_s = 1.0e-6;
    observed.rain_number_rate_per_m3_s = -100.0;
    observed.temperature_rate_k_s =
        microphysics_constants::L_v /
        microphysics_constants::cp *
        observed.rain_mass_rate_per_kg_s;
    auto mutated_cooling = observed;
    mutated_cooling.temperature_rate_k_s *= 1.001;
    CHECK_FALSE(map_observed_rain_event(
        mutated_cooling, 0.2, 1.0, batch, error));
    CHECK(error.find("latent-cooling provenance") != std::string::npos);
    CHECK(batch.size == 0);
    REQUIRE(map_observed_rain_event(
        observed, 0.2, 1.0, batch, error));
    CHECK(batch.size == 1);

    batch = {};
    batch.duration_s = 0.2;
    observed = {};
    observed.kind = WarmRainAcceptedEventKind::Autoconversion;
    observed.dry_density_kg_m3 = 1.0;
    observed.cloud_mass_rate_per_kg_s = -1.0e-6;
    observed.rain_mass_rate_per_kg_s = 1.0e-6;
    observed.rain_number_rate_per_m3_s = 100.0;
    observed.seed_diameter_m = 1.0e-4;
    observed.cloud_donor_scale = 0.5;
    observed.seed_mass_number_coherent = false;
    CHECK_FALSE(map_observed_rain_event(
        observed, 0.2, 1.0, batch, error));
    CHECK(error.find("incoherent") != std::string::npos);
    CHECK(batch.size == 0);
}

TEST_CASE("coherent combined warm adapter is byte-identical at one and four workers",
          "[physics][milbrandt][triple_warm_adapter]")
{
    constexpr int cells = 16;
    const auto parameters = warm_rain_parameters(
        0.0, 523.598775598, 3.0, 149.1, 0.5);
    const float saturation = static_cast<float>(
        thermodynamics::saturation_mixing_ratio_water(
            280.0f, 90000.0f));
    const WarmRainCell cell{
        280.0f, 90000.0f, 1.0f, 0.5f * saturation,
        0.001f, 1.0e-4f, 1.0e5f, 1.0e3f};
    const RainTripleMomentState initial{
        cell.rain, cell.rain_number,
        initial_z(cell.rain, cell.rain_number, 20.0)};
    std::array<std::array<WarmRainTendencies, cells>, 2> rates{};
    std::array<std::array<RainTripleMomentState, cells>, 2> states{};
    std::array<std::array<bool, cells>, 2> passed{};
    for (int run = 0; run < 2; ++run)
    {
        const int workers = run == 0 ? 1 : 4;
        #pragma omp parallel for num_threads(workers)
        for (int index = 0; index < cells; ++index)
        {
            std::string error;
            passed[run][index] = evaluate_observed_triple_warm_cell(
                parameters, cell, 0.05, {}, initial,
                rates[run][index], states[run][index], error);
        }
    }
    for (int index = 0; index < cells; ++index)
    {
        REQUIRE(passed[0][index]);
        REQUIRE(passed[1][index]);
        CHECK(std::memcmp(&rates[0][index], &rates[1][index],
                          sizeof(WarmRainTendencies)) == 0);
        CHECK(std::memcmp(&states[0][index], &states[1][index],
                          sizeof(RainTripleMomentState)) == 0);
    }
}

TEST_CASE("triple-only warm donor policy admits a coherent B2 autoconversion pair",
          "[physics][milbrandt][triple_warm_policy]")
{
    const auto parameters = warm_rain_parameters(
        0.0, 523.598775598, 3.0, 149.1, 0.5);
    const float saturation = static_cast<float>(
        thermodynamics::saturation_mixing_ratio_water(
            280.0f, 90000.0f));
    const WarmRainCell cell{
        280.0f, 90000.0f, 1.0f, 0.5f * saturation,
        0.01f, 0.002f, 1.0e5f, 1.0e5f};
    constexpr double dt = 10.0;
    WarmRainTendencies double_rates{};
    WarmRainAcceptedEventBuffer double_events;
    warm_rain_cell(parameters, cell, dt, double_rates, &double_events);
    TripleWarmRainResult triple;
    REQUIRE_NOTHROW(triple_warm_rain_cell(
        parameters, cell, dt, {}, triple));
    const WarmRainAcceptedEvent* double_auto = nullptr;
    const WarmRainAcceptedEvent* triple_auto = nullptr;
    for (std::size_t index = 0; index < double_events.size; ++index)
    {
        if (double_events.events[index].kind ==
            WarmRainAcceptedEventKind::Autoconversion)
        {
            double_auto = &double_events.events[index];
        }
    }
    for (std::size_t index = 0; index < triple.events().size; ++index)
    {
        if (triple.events().events[index].kind ==
            WarmRainAcceptedEventKind::Autoconversion)
        {
            triple_auto = &triple.events().events[index];
        }
    }
    REQUIRE(double_auto != nullptr);
    REQUIRE(triple_auto != nullptr);
    REQUIRE(triple_auto->cloud_donor_scale < 1.0);
    CHECK_FALSE(double_auto->seed_mass_number_coherent);
    CHECK(triple_auto->seed_mass_number_coherent);
    CHECK(triple_auto->cloud_donor_scale == double_auto->cloud_donor_scale);
    CHECK(triple_auto->rain_mass_rate_per_kg_s ==
          double_auto->rain_mass_rate_per_kg_s);
    CHECK(close(triple_auto->rain_number_rate_per_m3_s,
                double_auto->rain_number_rate_per_m3_s
                    * triple_auto->cloud_donor_scale, 1.0e-12));
    const double b2_number =
        triple_auto->rain_mass_rate_per_kg_s * cell.density /
        (parameters.c_r * std::pow(triple_auto->seed_diameter_m, 3.0));
    CHECK(close(triple_auto->rain_number_rate_per_m3_s,
                b2_number, 1.0e-12));
    CHECK(triple.rates().rain_number != double_rates.rain_number);

    const RainTripleMomentState initial{
        cell.rain, cell.rain_number,
        initial_z(cell.rain, cell.rain_number, 20.0)};
    WarmRainTendencies accepted_rates{-7.0f, -7.0f, -7.0f,
                                      -7.0f, -7.0f, -7.0f};
    RainTripleMomentState accepted_rain{-7.0f, -7.0f, -7.0f};
    std::string error;
    INFO(error);
    REQUIRE(evaluate_triple_only_warm_cell(
        parameters, cell, dt, {}, initial,
        accepted_rates, accepted_rain, error));
    CHECK(accepted_rates.rain_number == triple.rates().rain_number);
    CHECK(accepted_rain.mass_per_kg > 0.0f);
    CHECK(accepted_rain.number_per_kg > 0.0f);
    CHECK(accepted_rain.reflectivity_per_kg > 0.0f);
    bool saw_accretion = false;
    bool saw_self_collection = false;
    double autoconversion_b4 = 0.0;
    const double expected_z = independent_triple_warm_z(
        initial, triple.events(), dt, cell.density,
        saw_accretion, saw_self_collection, autoconversion_b4);
    CHECK(saw_accretion);
    CHECK(saw_self_collection);
    CHECK(close(std::tgamma(9.0) * std::tgamma(3.0)
                / (std::tgamma(6.0) * std::tgamma(6.0)),
                5.6, 1.0e-14));
    CHECK(autoconversion_b4 > 0.0);
    CHECK(close(accepted_rain.reflectivity_per_kg,
                expected_z, 2.0e-6));

    const auto rates_before = triple.rates();
    const auto events_before = triple.events();
    REQUIRE_THROWS_WITH(triple_warm_rain_cell(
        parameters, cell, dt, {}, triple, 0),
        "Milbrandt warm-event observer capacity is exhausted");
    CHECK(std::memcmp(&triple.rates(), &rates_before,
                      sizeof(WarmRainTendencies)) == 0);
    CHECK(triple.events().size == events_before.size);
}

TEST_CASE("triple-only warm path rejects rain-to-cloud and reset without cloud Z",
          "[physics][milbrandt][triple_warm_policy]")
{
    const auto parameters = warm_rain_parameters(
        0.0, 523.598775598, 3.0, 149.1, 0.5);
    const float saturation = static_cast<float>(
        thermodynamics::saturation_mixing_ratio_water(
            280.0f, 90000.0f));
    for (const WarmRainCell cell : {
             WarmRainCell{280.0f, 90000.0f, 1.0f, saturation,
                          0.0f, 1.0e-4f, 0.0f, 1.0e7f},
             WarmRainCell{280.0f, 90000.0f, 1.0f, saturation,
                          0.0f, 1.0e-4f, 0.0f, 0.0f}})
    {
        const RainTripleMomentState initial{
            cell.rain, cell.rain_number,
            cell.rain_number > 0.0f
                ? initial_z(cell.rain, cell.rain_number, 20.0)
                : 0.0f};
        WarmRainTendencies rates{-7.0f, -7.0f, -7.0f,
                                 -7.0f, -7.0f, -7.0f};
        RainTripleMomentState rain{-7.0f, -7.0f, -7.0f};
        std::string error;
        CHECK_FALSE(evaluate_triple_only_warm_cell(
            parameters, cell, 0.05, {}, initial,
            rates, rain, error));
        CHECK(error.find("cloud-Z owner") != std::string::npos);
        CHECK(rates.rain == -7.0f);
        CHECK(rain.reflectivity_per_kg == -7.0f);
    }
}

TEST_CASE("triple-only warm scale-one alias and donor-limited worker identity",
          "[physics][milbrandt][triple_warm_policy]")
{
    const auto parameters = warm_rain_parameters(
        0.0, 523.598775598, 3.0, 149.1, 0.5);
    const float saturation = static_cast<float>(
        thermodynamics::saturation_mixing_ratio_water(
            280.0f, 90000.0f));
    const WarmRainCell scale_one{
        280.0f, 90000.0f, 1.0f, 0.5f * saturation,
        0.001f, 1.0e-4f, 1.0e5f, 1.0e3f};
    const RainTripleMomentState initial_one{
        scale_one.rain, scale_one.rain_number,
        initial_z(scale_one.rain, scale_one.rain_number, 20.0)};
    WarmRainTendencies old_rates{};
    WarmRainTendencies triple_rates{};
    RainTripleMomentState old_rain{};
    RainTripleMomentState triple_rain{};
    std::string error;
    REQUIRE(evaluate_observed_triple_warm_cell(
        parameters, scale_one, 0.05, {}, initial_one,
        old_rates, old_rain, error));
    REQUIRE(evaluate_triple_only_warm_cell(
        parameters, scale_one, 0.05, {}, initial_one,
        triple_rates, triple_rain, error));
    CHECK(std::memcmp(&old_rates, &triple_rates,
                      sizeof(WarmRainTendencies)) == 0);
    CHECK(std::memcmp(&old_rain, &triple_rain,
                      sizeof(RainTripleMomentState)) == 0);

    const WarmRainCell limited{
        280.0f, 90000.0f, 1.0f, 0.5f * saturation,
        0.01f, 0.002f, 1.0e5f, 1.0e5f};
    const RainTripleMomentState initial_limited{
        limited.rain, limited.rain_number,
        initial_z(limited.rain, limited.rain_number, 20.0)};
    constexpr int cells = 16;
    std::array<std::array<WarmRainTendencies, cells>, 2> rates{};
    std::array<std::array<RainTripleMomentState, cells>, 2> rain{};
    std::array<std::array<bool, cells>, 2> passed{};
    for (int run = 0; run < 2; ++run)
    {
        #pragma omp parallel for num_threads(run == 0 ? 1 : 4)
        for (int index = 0; index < cells; ++index)
        {
            std::string local_error;
            passed[run][index] = evaluate_triple_only_warm_cell(
                parameters, limited, 10.0, {}, initial_limited,
                rates[run][index], rain[run][index], local_error);
        }
    }
    for (int index = 0; index < cells; ++index)
    {
        REQUIRE(passed[0][index]);
        REQUIRE(passed[1][index]);
        CHECK(std::memcmp(&rates[0][index], &rates[1][index],
                          sizeof(WarmRainTendencies)) == 0);
        CHECK(std::memcmp(&rain[0][index], &rain[1][index],
                          sizeof(RainTripleMomentState)) == 0);
    }
}

TEST_CASE("triple-only warm invalid and unrealizable inputs leave outputs unpublished",
          "[physics][milbrandt][triple_warm_policy]")
{
    const auto parameters = warm_rain_parameters(
        0.0, 523.598775598, 3.0, 149.1, 0.5);
    const float saturation = static_cast<float>(
        thermodynamics::saturation_mixing_ratio_water(
            280.0f, 90000.0f));
    const WarmRainCell cell{
        280.0f, 90000.0f, 1.0f, 0.5f * saturation,
        0.01f, 0.002f, 1.0e5f, 1.0e5f};
    const RainTripleMomentState valid{
        cell.rain, cell.rain_number,
        initial_z(cell.rain, cell.rain_number, 20.0)};
    WarmRainTendencies rates{-7.0f, -7.0f, -7.0f,
                             -7.0f, -7.0f, -7.0f};
    RainTripleMomentState rain{-7.0f, -7.0f, -7.0f};
    std::string error;
    CHECK_FALSE(evaluate_triple_only_warm_cell(
        parameters, cell, -1.0, {}, valid, rates, rain, error));
    CHECK(error.find("timestep") != std::string::npos);
    CHECK(rates.rain == -7.0f);
    CHECK(rain.reflectivity_per_kg == -7.0f);
    const RainTripleMomentState impossible{
        cell.rain, cell.rain_number, 0.0f};
    CHECK_FALSE(evaluate_triple_only_warm_cell(
        parameters, cell, 10.0, {}, impossible,
        rates, rain, error));
    CHECK(error.find("unrealizable") != std::string::npos);
    CHECK(rates.rain == -7.0f);
    CHECK(rain.reflectivity_per_kg == -7.0f);
}

TEST_CASE("triple-only dry zero-event source rejects invalid physical inputs atomically",
          "[physics][milbrandt][triple_warm_policy]")
{
    const auto parameters = warm_rain_parameters(
        0.0, 523.598775598, 3.0, 149.1, 0.5);
    const WarmRainCell dry{
        280.0f, 90000.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f};
    TripleWarmRainResult produced;
    REQUIRE_NOTHROW(triple_warm_rain_cell(
        parameters, dry, 0.1, {}, produced));
    REQUIRE(produced.events().size == 0);
    const auto rates_before = produced.rates();
    const auto duration_before = produced.duration_s();
    const auto density_before = produced.dry_density_kg_m3();
    const std::array<WarmRainCell, 8> invalid{
        WarmRainCell{dry.temperature, dry.pressure, 0.0f,
                     dry.vapor, dry.cloud, dry.rain,
                     dry.cloud_number, dry.rain_number},
        WarmRainCell{dry.temperature, dry.pressure,
                     std::numeric_limits<float>::quiet_NaN(),
                     dry.vapor, dry.cloud, dry.rain,
                     dry.cloud_number, dry.rain_number},
        WarmRainCell{dry.temperature, dry.pressure, -1.0f,
                     dry.vapor, dry.cloud, dry.rain,
                     dry.cloud_number, dry.rain_number},
        WarmRainCell{dry.temperature, 0.0f, dry.density,
                     dry.vapor, dry.cloud, dry.rain,
                     dry.cloud_number, dry.rain_number},
        WarmRainCell{0.0f, dry.pressure, dry.density,
                     dry.vapor, dry.cloud, dry.rain,
                     dry.cloud_number, dry.rain_number},
        WarmRainCell{dry.temperature, dry.pressure, dry.density,
                     -1.0f, dry.cloud, dry.rain,
                     dry.cloud_number, dry.rain_number},
        WarmRainCell{dry.temperature, dry.pressure, dry.density,
                     dry.vapor, dry.cloud, dry.rain,
                     -1.0f, dry.rain_number},
        WarmRainCell{dry.temperature, dry.pressure, dry.density,
                     dry.vapor,
                     std::numeric_limits<float>::quiet_NaN(),
                     dry.rain, dry.cloud_number, dry.rain_number}};
    for (const auto& cell : invalid)
    {
        REQUIRE_THROWS_AS(triple_warm_rain_cell(
            parameters, cell, 0.1, {}, produced), std::invalid_argument);
        CHECK(produced.events().size == 0);
        CHECK(produced.duration_s() == duration_before);
        CHECK(produced.dry_density_kg_m3() == density_before);
        CHECK(std::memcmp(&produced.rates(), &rates_before,
                          sizeof(WarmRainTendencies)) == 0);
    }
    WarmRainTendencies invalid_rates{};
    invalid_rates.rain_number =
        std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(triple_warm_rain_cell(
        parameters, dry, 0.1, invalid_rates, produced),
        std::invalid_argument);
    CHECK(produced.events().size == 0);
    std::array<WarmRainParameters, 4> invalid_parameters;
    invalid_parameters.fill(parameters);
    invalid_parameters[0].a_r =
        std::numeric_limits<double>::quiet_NaN();
    invalid_parameters[1].b_r = -1.0;
    invalid_parameters[2].gr34 =
        std::numeric_limits<double>::quiet_NaN();
    invalid_parameters[3].inv_rain_mass_factor = -1.0;
    for (const auto& law : invalid_parameters)
    {
        REQUIRE_THROWS_AS(triple_warm_rain_cell(
            law, dry, 0.1, {}, produced), std::invalid_argument);
        CHECK(produced.events().size == 0);
        CHECK(produced.duration_s() == duration_before);
        CHECK(produced.dry_density_kg_m3() == density_before);
        CHECK(std::memcmp(&produced.rates(), &rates_before,
                          sizeof(WarmRainTendencies)) == 0);
    }
    WarmRainTendencies published{-7.0f, -7.0f, -7.0f,
                                 -7.0f, -7.0f, -7.0f};
    RainTripleMomentState rain{-7.0f, -7.0f, -7.0f};
    std::string error;
    CHECK_FALSE(evaluate_triple_only_warm_cell(
        parameters, invalid[0], 0.1, {}, {0.0f, 0.0f, 0.0f},
        published, rain, error));
    CHECK(error.find("density") != std::string::npos);
    CHECK(published.rain == -7.0f);
    CHECK(rain.reflectivity_per_kg == -7.0f);
    CHECK_FALSE(evaluate_triple_only_warm_cell(
        invalid_parameters[0], dry, 0.1, {},
        {0.0f, 0.0f, 0.0f}, published, rain, error));
    CHECK(error.find("physical parameter law") != std::string::npos);
    CHECK(published.rain == -7.0f);
    CHECK(rain.reflectivity_per_kg == -7.0f);
}

TEST_CASE("exact warm stage captures authoritative doubles instead of Float rate replay",
          "[physics][milbrandt][triple_warm_primary]")
{
    const auto parameters = warm_rain_parameters(0, 523.598775598, 3, 149.1, .5);
    const float saturation = static_cast<float>(thermodynamics::saturation_mixing_ratio_water(280, 90000));
    bool saw_replay_difference = false;
    bool saw_limited = false;
    for (double duration : {.05, .1, .3, 10.0})
    {
        for (const auto cell : {WarmRainCell{280, 90000, 1, .5f * saturation, .001f, 1e-4f, 1e5f, 1e3f},
                               WarmRainCell{280, 90000, 1, .5f * saturation, .01f, .002f, 1e5f, 1e5f}})
        {
            const RainTripleMomentState initial{cell.rain, cell.rain_number,
                initial_z(cell.rain, cell.rain_number, 20)};
            ExactTripleWarmCellResult exact;
            std::string error;
            INFO("duration " << duration << ", cloud " << cell.cloud << ", rain " << cell.rain);
            TripleWarmRainExactResult diagnostic_producer;
            triple_warm_rain_cell_exact(parameters, cell, duration, {}, diagnostic_producer);
            INFO("producer cloud " << diagnostic_producer.endpoints().cloud_per_kg
                << ", Nc " << diagnostic_producer.endpoints().cloud_number_per_kg);
            const bool accepted = evaluate_exact_triple_warm_cell(parameters, cell, duration, initial, exact, error);
            INFO(error);
            REQUIRE(accepted);
            TripleWarmRainResult old_producer;
            triple_warm_rain_cell(parameters, cell, duration, {}, old_producer);
            CHECK(std::memcmp(&exact.rates, &old_producer.rates(), sizeof(WarmRainTendencies)) == 0);
            const auto& e = exact.endpoints;
            const auto& produced = exact.producer.endpoints();
            CHECK(e.vapor_per_kg == produced.vapor_per_kg);
            CHECK(e.cloud_per_kg == produced.cloud_per_kg);
            CHECK(e.rain_per_kg == produced.rain_per_kg);
            CHECK(e.cloud_number_per_kg == produced.cloud_number_per_kg);
            CHECK(e.rain_number_per_kg == produced.rain_number_per_kg);
            CHECK(e.temperature_increment_k == produced.temperature_increment_k);
            CHECK(exact.rain.mass_per_kg == e.rain_per_kg);
            CHECK(exact.rain.number_per_kg == e.rain_number_per_kg);
            CHECK(exact.rounded_rain.mass_per_kg == static_cast<float>(e.rain_per_kg));
            CHECK(exact.rounded_rain.number_per_kg == static_cast<float>(e.rain_number_per_kg));
            CHECK(e.vapor_per_kg + e.cloud_per_kg + e.rain_per_kg
                == Approx(static_cast<double>(cell.vapor) + cell.cloud + cell.rain).epsilon(1e-12));
            saw_replay_difference |= e.rain_per_kg != static_cast<double>(cell.rain) + duration * exact.rates.rain;
            saw_replay_difference |= e.vapor_per_kg != static_cast<double>(cell.vapor) + duration * exact.rates.vapor;
            double evaporation_mass = 0;
            for (std::size_t index = 0; index < exact.producer.events().size; ++index)
            {
                const auto& event = exact.producer.events().events[index];
                if (event.kind == WarmRainAcceptedEventKind::Evaporation)
                {
                    evaporation_mass -= duration * event.rain_mass_rate_per_kg_s;
                }
                if (event.kind == WarmRainAcceptedEventKind::Autoconversion)
                {
                    saw_limited |= event.cloud_donor_scale < 1;
                }
            }
            CHECK(e.temperature_increment_k == Approx(-microphysics_constants::L_v
                / microphysics_constants::cp * evaporation_mass).epsilon(1e-12).margin(1e-20));
            bool accretion = false;
            bool self = false;
            double b4 = 0;
            const double expected_z = independent_triple_warm_z(initial, exact.producer.events(),
                duration, cell.density, accretion, self, b4);
            CHECK(exact.rain.reflectivity_per_kg == Approx(expected_z).epsilon(1e-12).margin(1e-30));
        }
    }
    CHECK(saw_replay_difference);
    CHECK(saw_limited);
}

TEST_CASE("exact warm whole rain evaporation consumes its admitted population without residual moments",
          "[physics][milbrandt][triple_warm_evaporation_endpoint]")
{
    const auto parameters = warm_rain_parameters(0, 523.598775598, 3, 149.1, .5);
    for (float density : {.65f, 1.0f, 1.43f})
    {
        for (double duration : {1e3, 1e5, 1e6})
        {
            const WarmRainCell cell{288, 90000, density, 0, 0, 1e-4f, 0, 1e3f};
            const RainTripleMomentState initial{cell.rain, cell.rain_number,
                initial_z(cell.rain, cell.rain_number, 20)};
            TripleWarmRainExactResult producer;
            REQUIRE_NOTHROW(triple_warm_rain_cell_exact(parameters, cell, duration, {}, producer));
            REQUIRE(producer.endpoints().rain_per_kg == 0.0);
            REQUIRE(producer.endpoints().rain_number_per_kg == 0.0);
            const auto& budget = producer.endpoints().evaporation;
            REQUIRE(budget.present);
            CHECK(budget.donor_exhausted);
            CHECK(budget.admitted_mass_per_kg == budget.mass_before_per_kg);
            CHECK(budget.mass_after_per_kg == 0);
            CHECK(budget.number_after_per_m3 == 0);
            ExactTripleWarmCellResult result;
            std::string error;
            const bool accepted = apply_exact_triple_warm_cell(parameters, cell, duration,
                initial, producer, result, error);
            INFO("density " << density << ", duration " << duration << ": " << error);
            REQUIRE(accepted);
            CHECK(result.rain.mass_per_kg == 0.0);
            CHECK(result.rain.number_per_kg == 0.0);
            CHECK(result.rain.reflectivity_per_kg == 0.0);
            CHECK(result.rounded_rain.reflectivity_per_kg == 0.0f);
            CHECK(result.endpoints.vapor_per_kg == static_cast<double>(cell.rain));
            CHECK(result.endpoints.temperature_increment_k ==
                -microphysics_constants::L_v / microphysics_constants::cp * cell.rain);
        }
    }
}

TEST_CASE("exact warm capsule provenance capacity and invalid zero state leave output unchanged",
          "[physics][milbrandt][triple_warm_primary]")
{
    const auto parameters = warm_rain_parameters(0, 523.598775598, 3, 149.1, .5);
    const float saturation = static_cast<float>(thermodynamics::saturation_mixing_ratio_water(280, 90000));
    const WarmRainCell cell{280, 90000, 1, .5f * saturation, .001f, 1e-4f, 1e5f, 1e3f};
    const RainTripleMomentState initial{cell.rain, cell.rain_number,
        initial_z(cell.rain, cell.rain_number, 20)};
    ExactTripleWarmCellResult baseline;
    std::string error;
    REQUIRE(evaluate_exact_triple_warm_cell(parameters, cell, .05, initial, baseline, error));
    const auto check_unchanged = [&](const ExactTripleWarmCellResult& result)
    {
        CHECK(std::memcmp(&result.endpoints, &baseline.endpoints, sizeof(WarmRainDoubleEndpoints)) == 0);
        CHECK(std::memcmp(&result.rain, &baseline.rain, sizeof(ExactRainTripleMomentState)) == 0);
        CHECK(std::memcmp(&result.rates, &baseline.rates, sizeof(WarmRainTendencies)) == 0);
    };
    for (std::size_t index = 0; index < 8; ++index)
    {
        auto source = cell;
        const std::array<float*, 8> values{&source.temperature, &source.pressure, &source.density,
            &source.vapor, &source.cloud, &source.rain, &source.cloud_number, &source.rain_number};
        *values[index] = std::nextafter(*values[index], std::numeric_limits<float>::infinity());
        auto result = baseline;
        CHECK_FALSE(apply_exact_triple_warm_cell(parameters, source, .05, initial, baseline.producer, result, error));
        check_unchanged(result);
    }
    for (double duration : {.1, 0.0, -1.0, std::numeric_limits<double>::quiet_NaN()})
    {
        auto result = baseline;
        CHECK_FALSE(apply_exact_triple_warm_cell(parameters, cell, duration, initial, baseline.producer, result, error));
        check_unchanged(result);
    }
    auto result = baseline;
    CHECK_FALSE(evaluate_exact_triple_warm_cell(parameters, cell, .05, initial, result, error, 0));
    check_unchanged(result);
    TripleWarmRainExactResult unpublished;
    CHECK_FALSE(apply_exact_triple_warm_cell(parameters, cell, .05, initial, unpublished, result, error));
    check_unchanged(result);
    for (std::size_t index = 0; index < 8; ++index)
    {
        WarmRainCell empty{280, 90000, 1, 0, 0, 0, 0, 0};
        const std::array<float*, 8> values{&empty.temperature, &empty.pressure, &empty.density,
            &empty.vapor, &empty.cloud, &empty.rain, &empty.cloud_number, &empty.rain_number};
        *values[index] = std::numeric_limits<float>::quiet_NaN();
        result = baseline;
        CHECK_FALSE(evaluate_exact_triple_warm_cell(parameters, empty, .05, {}, result, error));
        check_unchanged(result);
    }
}

TEST_CASE("owned rain evaporation endpoint preserves B1 B3 spectral moments through extinction",
          "[physics][milbrandt][triple_warm_evaporation_endpoint]")
{
    const auto gamma_integral = [](int order)
    {
        constexpr int intervals = 8192;
        constexpr long double end = 80;
        constexpr long double width = end / intervals;
        long double sum = (order == 0 ? 1 : 0) + std::pow(end, order) * std::exp(-end);
        for (int index = 1; index < intervals; ++index)
        {
            const long double x = index * width;
            sum += (index % 2 ? 4 : 2) * std::pow(x, order) * std::exp(-x);
        }
        return sum * width / 3;
    };
    const long double g0 = gamma_integral(0);
    const long double g3 = gamma_integral(3);
    const long double g6 = gamma_integral(6);
    CHECK(static_cast<double>(g6 * g0 / (g3 * g3)) == Approx(20).epsilon(2e-9));
    const std::array<std::array<double, 2>, 4> profiles{{{0, .05}, {2, .1}, {20, .3}, {200, 1000}}};
    for (const auto& profile : profiles)
    {
        const double alpha = profile[0];
        const double duration = profile[1];
        const double gamma_factor = (alpha + 4) * (alpha + 5) * (alpha + 6)
            / ((alpha + 1) * (alpha + 2) * (alpha + 3));
        const RainTripleMomentState initial{1e-4f, 1e3f, initial_z(1e-4f, 1e3f, gamma_factor)};
        for (double fraction : {0.0, .25, .9, std::nextafter(1.0, 0.0), 1.0})
        {
            AdmittedRainEvaporationEndpoint endpoint;
            AcceptedRainPhaseBatch batch;
            batch.duration_s = duration;
            if (fraction > 0)
            {
                endpoint.present = true;
                endpoint.mass_before_per_kg = initial.mass_per_kg;
                endpoint.number_before_per_kg = initial.number_per_kg;
                endpoint.admitted_mass_per_kg = fraction * endpoint.mass_before_per_kg;
                endpoint.mass_after_per_kg = endpoint.mass_before_per_kg - endpoint.admitted_mass_per_kg;
                endpoint.number_after_per_kg = endpoint.number_before_per_kg
                    * (endpoint.mass_after_per_kg / endpoint.mass_before_per_kg);
                endpoint.donor_exhausted = endpoint.admitted_mass_per_kg == endpoint.mass_before_per_kg;
                const auto event = make_same(AcceptedRainEventKind::Evaporation,
                    -endpoint.admitted_mass_per_kg / duration,
                    (endpoint.number_after_per_kg - endpoint.number_before_per_kg) / duration, duration);
                std::string error;
                REQUIRE(append_admitted_rain_phase_event(batch, RainEventPhase::Evaporation, event, error));
            }
            RainTripleMomentState rounded{-7, -7, -7};
            ExactRainTripleMomentState exact{-7, -7, -7};
            std::string error;
            INFO("alpha " << alpha << ", fraction " << fraction << ", duration " << duration);
            const bool accepted = apply_admitted_rain_phases_with_evaporation_endpoint(
                initial, batch, endpoint, rounded, exact, error);
            INFO(error);
            REQUIRE(accepted);
            const long double remaining = fraction == 0 ? 1 :
                static_cast<long double>(endpoint.mass_after_per_kg) / endpoint.mass_before_per_kg;
            CHECK(exact.mass_per_kg == (fraction == 0 ? initial.mass_per_kg : endpoint.mass_after_per_kg));
            CHECK(exact.number_per_kg == (fraction == 0 ? initial.number_per_kg : endpoint.number_after_per_kg));
            CHECK(exact.reflectivity_per_kg == Approx(static_cast<double>(
                static_cast<long double>(initial.reflectivity_per_kg) * remaining)).epsilon(2e-14).margin(1e-45));
            CHECK(rounded.mass_per_kg == static_cast<float>(exact.mass_per_kg));
            CHECK(rounded.number_per_kg == static_cast<float>(exact.number_per_kg));
            CHECK(rounded.reflectivity_per_kg == static_cast<float>(exact.reflectivity_per_kg));
            if (fraction == 1)
            {
                CHECK(exact.mass_per_kg == 0);
                CHECK(exact.number_per_kg == 0);
                CHECK(exact.reflectivity_per_kg == 0);
            }
            else
            {
                const long double ratio_before = static_cast<long double>(initial.reflectivity_per_kg)
                    * initial.number_per_kg / (static_cast<long double>(initial.mass_per_kg) * initial.mass_per_kg);
                const long double ratio_after = static_cast<long double>(exact.reflectivity_per_kg)
                    * exact.number_per_kg / (static_cast<long double>(exact.mass_per_kg) * exact.mass_per_kg);
                CHECK(static_cast<double>(ratio_after) == Approx(static_cast<double>(ratio_before)).epsilon(2e-14));
            }
        }
    }
}

TEST_CASE("owned evaporation budget presence provenance and late errors leave moments unpublished",
          "[physics][milbrandt][triple_warm_evaporation_endpoint]")
{
    const RainTripleMomentState initial{1e-4f, 1e3f, initial_z(1e-4f, 1e3f, 20)};
    constexpr double duration = .1;
    AdmittedRainEvaporationEndpoint endpoint;
    endpoint.present = true;
    endpoint.mass_before_per_kg = initial.mass_per_kg;
    endpoint.number_before_per_kg = initial.number_per_kg;
    endpoint.admitted_mass_per_kg = .25 * endpoint.mass_before_per_kg;
    endpoint.mass_after_per_kg = endpoint.mass_before_per_kg - endpoint.admitted_mass_per_kg;
    endpoint.number_after_per_kg = .75 * endpoint.number_before_per_kg;
    AcceptedRainPhaseBatch batch;
    batch.duration_s = duration;
    std::string error;
    const auto event = make_same(AcceptedRainEventKind::Evaporation,
        -endpoint.admitted_mass_per_kg / duration,
        (endpoint.number_after_per_kg - endpoint.number_before_per_kg) / duration, duration);
    REQUIRE(append_admitted_rain_phase_event(batch, RainEventPhase::Evaporation, event, error));
    for (int mutation = 0; mutation < 15; ++mutation)
    {
        auto altered = endpoint;
        auto events = batch;
        switch (mutation)
        {
            case 0: altered.present = false; break;
            case 1: altered.mass_before_per_kg = 0; break;
            case 2: altered.mass_before_per_kg *= 2; break;
            case 3: altered.number_before_per_kg = 0; break;
            case 4: altered.number_before_per_kg *= 2; break;
            case 5: altered.admitted_mass_per_kg = 0; break;
            case 6: altered.admitted_mass_per_kg = 2 * altered.mass_before_per_kg; break;
            case 7: altered.mass_after_per_kg *= .5; break;
            case 8: altered.number_after_per_kg *= .5; break;
            case 9: altered.donor_exhausted = true; break;
            case 10: altered.mass_after_per_kg = std::numeric_limits<double>::quiet_NaN(); break;
            case 11: events.events[0].event.admitted_mass_rate_per_kg_s *= .5; break;
            case 12: events.events[0].event.admitted_number_rate_per_kg_s *= .5; break;
            case 13: events.size = 0; break;
            case 14: events.duration_s *= 2; break;
        }
        RainTripleMomentState rounded{-7, -7, -7};
        ExactRainTripleMomentState exact{-7, -7, -7};
        INFO("evaporation budget mutation " << mutation);
        CHECK_FALSE(apply_admitted_rain_phases_with_evaporation_endpoint(
            initial, events, altered, rounded, exact, error));
        CHECK(rounded.mass_per_kg == -7);
        CHECK(rounded.number_per_kg == -7);
        CHECK(rounded.reflectivity_per_kg == -7);
        CHECK(exact.mass_per_kg == -7);
        CHECK(exact.number_per_kg == -7);
        CHECK(exact.reflectivity_per_kg == -7);
    }
    AcceptedRainPhaseBatch empty;
    empty.duration_s = duration;
    AdmittedRainEvaporationEndpoint absent;
    absent.mass_before_per_kg = 1e-4;
    RainTripleMomentState rounded{-7, -7, -7};
    ExactRainTripleMomentState exact{-7, -7, -7};
    CHECK_FALSE(apply_admitted_rain_phases_with_evaporation_endpoint(initial, empty, absent, rounded, exact, error));
    CHECK(exact.reflectivity_per_kg == -7);
}

TEST_CASE("immutable exact producer evaporation provenance cannot be forged into publication",
          "[physics][milbrandt][triple_warm_evaporation_endpoint]")
{
    const auto parameters = warm_rain_parameters(0, 523.598775598, 3, 149.1, .5);
    const WarmRainCell cell{288, 90000, .65f, 0, 0, 1e-4f, 0, 1e3f};
    const RainTripleMomentState initial{cell.rain, cell.rain_number,
        initial_z(cell.rain, cell.rain_number, 20)};
    constexpr double duration = 1000;
    ExactTripleWarmCellResult baseline;
    std::string error;
    REQUIRE(evaluate_exact_triple_warm_cell(parameters, cell, duration, initial, baseline, error));
    REQUIRE(baseline.producer.endpoints().evaporation.donor_exhausted);
    for (int mutation = 0; mutation < 11; ++mutation)
    {
        auto producer = baseline.producer;
        auto& endpoints = const_cast<WarmRainDoubleEndpoints&>(producer.endpoints());
        auto& budget = endpoints.evaporation;
        switch (mutation)
        {
            case 0: budget.present = false; break;
            case 1: budget.mass_before_per_kg *= 2; break;
            case 2: budget.number_before_per_m3 *= 2; break;
            case 3: budget.admitted_mass_per_kg *= .5; break;
            case 4: budget.mass_after_per_kg = 1e-5; break;
            case 5: budget.number_after_per_m3 = 1; break;
            case 6: budget.donor_exhausted = false; break;
            case 7: budget.admitted_mass_per_kg = std::numeric_limits<double>::infinity(); break;
            case 8: endpoints.vapor_per_kg *= .5; break;
            case 9: endpoints.temperature_increment_k *= .5; break;
            case 10: budget.number_before_per_m3 = std::numeric_limits<double>::quiet_NaN(); break;
        }
        auto result = baseline;
        INFO("producer evaporation mutation " << mutation);
        CHECK_FALSE(apply_exact_triple_warm_cell(parameters, cell, duration, initial, producer, result, error));
        CHECK(std::memcmp(&result.endpoints, &baseline.endpoints, sizeof(WarmRainDoubleEndpoints)) == 0);
        CHECK(std::memcmp(&result.rain, &baseline.rain, sizeof(ExactRainTripleMomentState)) == 0);
        CHECK(std::memcmp(&result.rates, &baseline.rates, sizeof(WarmRainTendencies)) == 0);
    }
}
