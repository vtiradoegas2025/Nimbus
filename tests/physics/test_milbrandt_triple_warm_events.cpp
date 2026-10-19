#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/triple_warm_events.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace
{

using tmv::milbrandt_detail::AcceptedRainEvent;
using tmv::milbrandt_detail::AcceptedRainEventKind;
using tmv::milbrandt_detail::RainTripleMomentState;

constexpr double kRainMassCoefficient = 523.598775598;

long double gamma_integral(long double order)
{
    constexpr int intervals = 4000;
    const long double upper = std::pow(80.0L, 0.25L);
    const long double step = upper / intervals;
    long double sum = 0.0L;
    for (int index = 0; index <= intervals; ++index)
    {
        const long double u = step * index;
        const long double x = u * u * u * u;
        const int weight = index == 0 || index == intervals ? 1 : index % 2 == 0 ? 2 : 4;
        sum += weight * 4.0L *
            std::pow(u, 4.0L * (order + 1.0L) - 1.0L) *
            std::exp(-x);
    }
    return sum * step / 3.0L;
}

double gamma_z_per_kg(double mass, double number, double alpha)
{
    const long double lambda = std::pow(
        kRainMassCoefficient * number *
            gamma_integral(alpha + 3.0L) /
            (mass * gamma_integral(alpha)),
        1.0L / 3.0L);
    return number * static_cast<double>(
        gamma_integral(alpha + 6.0L) /
        (gamma_integral(alpha) * std::pow(lambda, 6.0L)));
}

double relative_error(double actual, double expected)
{
    return std::abs(actual - expected) / std::max(std::abs(expected), 1.0e-30);
}

} // namespace

TEST_CASE("admitted autoconversion rain event follows Part II B4 seed and dry-density units",
          "[physics][milbrandt][triple_warm]")
{
    constexpr double available_cloud = 1.0e-4;
    constexpr double requested_auto_mass_rate = 8.0e-5;
    constexpr double requested_accretion_mass_rate = 5.0e-5;
    constexpr double dt = 1.0;
    const double limiter = available_cloud /
        (dt * (requested_auto_mass_rate + requested_accretion_mass_rate));
    REQUIRE(limiter < 1.0);
    const double admitted_mass_rate = requested_auto_mass_rate * limiter;
    constexpr double admitted_drop_diameter = 100.0e-6;
    for (double density : {0.65, 1.2, 1.8})
    {
        const double raw_number_rate_volume =
            requested_auto_mass_rate * density /
            (kRainMassCoefficient * std::pow(admitted_drop_diameter, 3.0));
        const double admitted_number_rate_volume =
            raw_number_rate_volume * limiter;
        AcceptedRainEvent event;
        std::string error;
        REQUIRE(tmv::milbrandt_detail::make_admitted_autoconversion_event(
            admitted_mass_rate, admitted_number_rate_volume, density,
            dt, event, error));
        CHECK(event.kind == AcceptedRainEventKind::Autoconversion);
        CHECK(relative_error(event.admitted_number_rate_per_kg_s,
                             admitted_number_rate_volume / density) < 1.0e-15);
        double z_rate = -1.0;
        REQUIRE(tmv::milbrandt_detail::admitted_rain_reflectivity_rate(
            {}, event, z_rate, error));
        const double reference_z_rate = gamma_z_per_kg(
            admitted_mass_rate, event.admitted_number_rate_per_kg_s, 2.0);
        CHECK(relative_error(z_rate, reference_z_rate) < 1.0e-6);
        CHECK(relative_error(z_rate,
            5.6 / (kRainMassCoefficient * kRainMassCoefficient) *
            admitted_mass_rate * admitted_mass_rate /
            event.admitted_number_rate_per_kg_s) < 1.0e-12);
        RainTripleMomentState output{-7.0f, -7.0f, -7.0f};
        REQUIRE(tmv::milbrandt_detail::apply_admitted_rain_events(
            {}, {event}, output, error));
        CHECK(relative_error(output.mass_per_kg, admitted_mass_rate * dt) < 1.0e-6);
        CHECK(relative_error(output.number_per_kg * density,
                             admitted_number_rate_volume * dt) < 1.0e-6);
        CHECK(relative_error(output.reflectivity_per_kg, reference_z_rate * dt) < 1.0e-6);
        const double remaining_cloud = available_cloud -
            dt * (admitted_mass_rate + requested_accretion_mass_rate * limiter);
        CHECK(std::abs(remaining_cloud) < 1.0e-18);
    }
}

TEST_CASE("rain fixed-shape B3 event rates match independent finite-difference closure",
          "[physics][milbrandt][triple_warm]")
{
    constexpr double mass = 1.0e-4;
    constexpr double number = 1.0e4;
    const RainTripleMomentState initial{
        static_cast<float>(mass), static_cast<float>(number),
        static_cast<float>(gamma_z_per_kg(mass, number, 5.0))};
    constexpr double dt = 0.01;
    const std::array<std::array<double, 3>, 3> inputs{{
        {{-1.0e-6, -100.0, 0.0}},
        {{0.0, -100.0, 1.0}},
        {{1.0e-6, 0.0, 2.0}}
    }};
    for (const auto& input : inputs)
    {
        const auto kind = input[2] == 0.0 ? AcceptedRainEventKind::Evaporation
            : input[2] == 1.0 ? AcceptedRainEventKind::SelfCollection
                              : AcceptedRainEventKind::Accretion;
        AcceptedRainEvent event;
        std::string error;
        REQUIRE(tmv::milbrandt_detail::make_admitted_same_category_event(
            kind, input[0], input[1], dt, event, error));
        double rate = 0.0;
        REQUIRE(tmv::milbrandt_detail::admitted_rain_reflectivity_rate(
            initial, event, rate, error));
        constexpr double increment = 1.0e-3;
        const double coefficient = static_cast<double>(initial.reflectivity_per_kg) *
            initial.number_per_kg /
            (static_cast<double>(initial.mass_per_kg) * initial.mass_per_kg);
        const auto fixed_shape_z = [&](double q, double n)
        {
            return coefficient * q * q / n;
        };
        const double finite_difference =
            (fixed_shape_z(initial.mass_per_kg + increment * input[0],
                           initial.number_per_kg + increment * input[1]) -
             fixed_shape_z(initial.mass_per_kg - increment * input[0],
                           initial.number_per_kg - increment * input[1])) /
            (2.0 * increment);
        CHECK(relative_error(rate, finite_difference) < 1.0e-6);
        RainTripleMomentState output{-7.0f, -7.0f, -7.0f};
        REQUIRE(tmv::milbrandt_detail::apply_admitted_rain_events(
            initial, {event}, output, error));
        CHECK(relative_error(output.mass_per_kg,
                             initial.mass_per_kg + dt * input[0]) < 1.0e-6);
        CHECK(relative_error(output.number_per_kg,
                             initial.number_per_kg + dt * input[1]) < 1.0e-6);
        CHECK(relative_error(output.reflectivity_per_kg,
                             initial.reflectivity_per_kg + dt * rate) < 1.0e-6);
    }
}

TEST_CASE("late invalid admitted rain event cannot publish partial q/N/Z",
          "[physics][milbrandt][triple_warm]")
{
    std::string error;
    AcceptedRainEvent auto_event;
    REQUIRE(tmv::milbrandt_detail::make_admitted_autoconversion_event(
        1.0e-5, 1.0e6, 1.2, 1.0, auto_event, error));
    AcceptedRainEvent overdraw;
    REQUIRE(tmv::milbrandt_detail::make_admitted_same_category_event(
        AcceptedRainEventKind::SelfCollection, 0.0, -1.0e12, 1.0,
        overdraw, error));
    RainTripleMomentState output{-7.0f, -7.0f, -7.0f};
    CHECK_FALSE(tmv::milbrandt_detail::apply_admitted_rain_events(
        {}, {auto_event, overdraw}, output, error));
    CHECK(output.mass_per_kg == -7.0f);
    CHECK(output.number_per_kg == -7.0f);
    CHECK(output.reflectivity_per_kg == -7.0f);

    auto tampered = auto_event;
    tampered.source_dry_density_kg_m3 = 0.0;
    CHECK_FALSE(tmv::milbrandt_detail::apply_admitted_rain_events(
        {}, {tampered}, output, error));
    CHECK(error.find("provenance") != std::string::npos);
    CHECK(output.mass_per_kg == -7.0f);
    CHECK_FALSE(tmv::milbrandt_detail::make_admitted_autoconversion_event(
        -1.0e-5, 1.0e6, 1.2, 1.0, tampered, error));
    CHECK_FALSE(tmv::milbrandt_detail::make_admitted_autoconversion_event(
        1.0e-5, 1.0e6, std::numeric_limits<double>::quiet_NaN(),
        1.0, tampered, error));
    CHECK_FALSE(tmv::milbrandt_detail::make_admitted_same_category_event(
        AcceptedRainEventKind::Autoconversion, 1.0e-5, 1.0e4,
        1.0, tampered, error));
    CHECK_FALSE(tmv::milbrandt_detail::make_admitted_same_category_event(
        AcceptedRainEventKind::Evaporation, 1.0e-5, -1.0e4,
        1.0, tampered, error));

    const RainTripleMomentState unrealizable{
        1.0e-4f, 1.0e4f,
        static_cast<float>(0.5 * 1.0e4 *
            std::pow(1.0e-4 / (kRainMassCoefficient * 1.0e4), 2.0))};
    CHECK_FALSE(tmv::milbrandt_detail::apply_admitted_rain_events(
        unrealizable, {}, output, error));
    CHECK(output.mass_per_kg == -7.0f);

    AcceptedRainEvent overflow;
    REQUIRE(tmv::milbrandt_detail::make_admitted_autoconversion_event(
        std::numeric_limits<double>::max(), 1.0e6, 1.2, 1.0,
        overflow, error));
    CHECK_FALSE(tmv::milbrandt_detail::apply_admitted_rain_events(
        {}, {overflow}, output, error));
    CHECK(output.mass_per_kg == -7.0f);
    CHECK(output.number_per_kg == -7.0f);
    CHECK(output.reflectivity_per_kg == -7.0f);

    std::vector<AcceptedRainEvent> too_many(65, auto_event);
    CHECK_FALSE(tmv::milbrandt_detail::apply_admitted_rain_events(
        {}, too_many, output, error));
    CHECK(error.find("64-event resource budget") != std::string::npos);
    CHECK(output.mass_per_kg == -7.0f);
}

TEST_CASE("independent rain event sequences are byte-identical at one and four workers",
          "[physics][milbrandt][triple_warm]")
{
    constexpr int cells = 16;
    std::array<RainTripleMomentState, cells> initial{};
    for (int cell = 0; cell < cells; ++cell)
    {
        initial[cell] = {static_cast<float>(1.0e-4 + cell * 1.0e-7),
                         static_cast<float>(1.0e4 + cell * 10.0),
                         static_cast<float>(gamma_z_per_kg(
                             1.0e-4 + cell * 1.0e-7,
                             1.0e4 + cell * 10.0, 2.0))};
    }
    AcceptedRainEvent self_collection;
    AcceptedRainEvent accretion;
    std::string error;
    REQUIRE(tmv::milbrandt_detail::make_admitted_same_category_event(
        AcceptedRainEventKind::SelfCollection, 0.0, -10.0, 0.1,
        self_collection, error));
    REQUIRE(tmv::milbrandt_detail::make_admitted_same_category_event(
        AcceptedRainEventKind::Accretion, 1.0e-7, 0.0, 0.1,
        accretion, error));
    const std::vector<AcceptedRainEvent> events{self_collection, accretion};
    std::array<std::array<RainTripleMomentState, cells>, 2> outputs{};
    std::array<std::array<bool, cells>, 2> passed{};
    for (int run = 0; run < 2; ++run)
    {
        const int workers = run == 0 ? 1 : 4;
        #pragma omp parallel for num_threads(workers)
        for (int cell = 0; cell < cells; ++cell)
        {
            std::string local_error;
            passed[run][cell] = tmv::milbrandt_detail::apply_admitted_rain_events(
                initial[cell], events, outputs[run][cell], local_error);
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

TEST_CASE("named rain number corrections use exact fixed-shape endpoints",
          "[physics][milbrandt][triple_warm]")
{
    constexpr double mass = 1.0e-4;
    constexpr double number = 1.0e4;
    constexpr double density = 1.2;
    constexpr double duration = 0.5;
    const RainTripleMomentState initial{
        static_cast<float>(mass), static_cast<float>(number),
        static_cast<float>(gamma_z_per_kg(mass, number, 3.0))};
    for (const auto kind : {
             AcceptedRainEventKind::PreRainNumberRegularization,
             AcceptedRainEventKind::PostRainNumberRegularization,
             AcceptedRainEventKind::RainNumberClip})
    {
        AcceptedRainEvent event;
        std::string error;
        const double number_increment = kind == AcceptedRainEventKind::RainNumberClip
            ? -0.25 * number : 1.0e4 * number;
        REQUIRE(tmv::milbrandt_detail::make_admitted_rain_number_correction_event(
            kind, number_increment * density / duration, density,
            duration, event, error));
        CHECK(event.kind == kind);
        CHECK(event.admitted_mass_rate_per_kg_s == 0.0);
        CHECK(relative_error(event.admitted_number_rate_per_kg_s,
                             number_increment / duration) < 1.0e-15);
        RainTripleMomentState output{-7.0f, -7.0f, -7.0f};
        REQUIRE(tmv::milbrandt_detail::apply_admitted_rain_events(
            initial, {event}, output, error));
        const double expected_number = initial.number_per_kg +
            duration * event.admitted_number_rate_per_kg_s;
        const double expected_z = static_cast<double>(initial.reflectivity_per_kg) *
            initial.number_per_kg / expected_number;
        CHECK(output.mass_per_kg == initial.mass_per_kg);
        CHECK(relative_error(output.number_per_kg, expected_number) < 1.0e-6);
        CHECK(relative_error(output.reflectivity_per_kg, expected_z) < 1.0e-6);
        CHECK(relative_error(output.reflectivity_per_kg,
                             gamma_z_per_kg(output.mass_per_kg,
                                            output.number_per_kg, 3.0)) < 1.0e-6);
        const double lower = output.mass_per_kg * output.mass_per_kg /
            (kRainMassCoefficient * kRainMassCoefficient *
             output.number_per_kg);
        CHECK(output.reflectivity_per_kg >= lower);
        double effective_rate = 0.0;
        REQUIRE(tmv::milbrandt_detail::admitted_rain_reflectivity_rate(
            initial, event, effective_rate, error));
        CHECK(relative_error(effective_rate,
            (expected_z - initial.reflectivity_per_kg) / duration) < 1.0e-12);
        if (kind != AcceptedRainEventKind::RainNumberClip)
        {
            const double euler_b3 = initial.reflectivity_per_kg *
                (1.0 - duration * event.admitted_number_rate_per_kg_s /
                     initial.number_per_kg);
            CHECK(euler_b3 < 0.0);
            CHECK(output.reflectivity_per_kg > 0.0f);
        }
    }
    AcceptedRainEvent positive_clip;
    std::string error;
    REQUIRE(tmv::milbrandt_detail::make_admitted_rain_number_correction_event(
        AcceptedRainEventKind::RainNumberClip, 1200.0,
        density, duration, positive_clip, error));
    RainTripleMomentState clipped{-7.0f, -7.0f, -7.0f};
    REQUIRE(tmv::milbrandt_detail::apply_admitted_rain_events(
        initial, {positive_clip}, clipped, error));
    CHECK(clipped.number_per_kg > initial.number_per_kg);
    CHECK(clipped.reflectivity_per_kg < initial.reflectivity_per_kg);
}

TEST_CASE("typed rain-to-cloud and reset require complete q/N removal and zero Z",
          "[physics][milbrandt][triple_warm]")
{
    constexpr double density = 1.2;
    constexpr double duration = 0.2;
    const RainTripleMomentState initial{
        1.0e-4f, 1.0e4f,
        static_cast<float>(gamma_z_per_kg(1.0e-4, 1.0e4, 2.0))};
    for (const auto kind : {AcceptedRainEventKind::RainToCloud,
                            AcceptedRainEventKind::RainReset})
    {
        const double rain_mass_rate = -static_cast<double>(initial.mass_per_kg) /
            duration;
        const double rain_number_rate =
            -static_cast<double>(initial.number_per_kg) * density / duration;
        const double cloud_mass_rate = kind == AcceptedRainEventKind::RainToCloud
            ? -rain_mass_rate : 0.0;
        const double cloud_number_rate = kind == AcceptedRainEventKind::RainToCloud
            ? -rain_number_rate : 0.0;
        AcceptedRainEvent event;
        std::string error;
        REQUIRE(tmv::milbrandt_detail::make_admitted_rain_removal_event(
            kind, cloud_mass_rate, rain_mass_rate,
            cloud_number_rate, rain_number_rate,
            density, duration, event, error));
        CHECK(event.kind == kind);
        CHECK(relative_error(event.admitted_number_rate_per_kg_s,
                             rain_number_rate / density) < 1.0e-15);
        double z_rate = 0.0;
        REQUIRE(tmv::milbrandt_detail::admitted_rain_reflectivity_rate(
            initial, event, z_rate, error));
        CHECK(relative_error(z_rate,
                             -initial.reflectivity_per_kg / duration) < 1.0e-15);
        RainTripleMomentState output{-7.0f, -7.0f, -7.0f};
        REQUIRE(tmv::milbrandt_detail::apply_admitted_rain_events(
            initial, {event}, output, error));
        CHECK(output.mass_per_kg == 0.0f);
        CHECK(output.number_per_kg == 0.0f);
        CHECK(output.reflectivity_per_kg == 0.0f);
    }
    const RainTripleMomentState tiny{
        1.0e-15f, 1.0e-5f,
        static_cast<float>(gamma_z_per_kg(1.0e-15, 1.0e-5, 2.0))};
    AcceptedRainEvent reset;
    std::string error;
    REQUIRE(tmv::milbrandt_detail::make_admitted_rain_removal_event(
        AcceptedRainEventKind::RainReset, 0.0,
        -static_cast<double>(tiny.mass_per_kg) / duration,
        0.0, -static_cast<double>(tiny.number_per_kg) * density / duration,
        density, duration, reset, error));
    RainTripleMomentState cleared{-7.0f, -7.0f, -7.0f};
    REQUIRE(tmv::milbrandt_detail::apply_admitted_rain_events(
        tiny, {reset}, cleared, error));
    CHECK(cleared.mass_per_kg == 0.0f);
    CHECK(cleared.number_per_kg == 0.0f);
    CHECK(cleared.reflectivity_per_kg == 0.0f);
}

TEST_CASE("rain correction and removal reject bad provenance without partial publication",
          "[physics][milbrandt][triple_warm]")
{
    constexpr double density = 1.2;
    const RainTripleMomentState initial{
        1.0e-4f, 1.0e4f,
        static_cast<float>(gamma_z_per_kg(1.0e-4, 1.0e4, 2.0))};
    RainTripleMomentState output{-7.0f, -7.0f, -7.0f};
    AcceptedRainEvent event;
    event.kind = AcceptedRainEventKind::Accretion;
    event.admitted_mass_rate_per_kg_s = 42.0;
    std::string error;
    CHECK_FALSE(tmv::milbrandt_detail::make_admitted_rain_number_correction_event(
        AcceptedRainEventKind::PreRainNumberRegularization,
        -100.0, density, 1.0, event, error));
    CHECK(event.kind == AcceptedRainEventKind::Accretion);
    CHECK(event.admitted_mass_rate_per_kg_s == 42.0);
    CHECK_FALSE(tmv::milbrandt_detail::make_admitted_rain_number_correction_event(
        AcceptedRainEventKind::RainNumberClip,
        0.0, density, 1.0, event, error));
    CHECK_FALSE(tmv::milbrandt_detail::make_admitted_rain_number_correction_event(
        AcceptedRainEventKind::RainNumberClip,
        100.0, 0.0, 1.0, event, error));
    CHECK_FALSE(tmv::milbrandt_detail::make_admitted_rain_number_correction_event(
        AcceptedRainEventKind::RainNumberClip,
        100.0, density, 0.0, event, error));
    REQUIRE(tmv::milbrandt_detail::make_admitted_rain_number_correction_event(
        AcceptedRainEventKind::RainNumberClip,
        -2.0 * initial.number_per_kg * density, density, 1.0,
        event, error));
    CHECK_FALSE(tmv::milbrandt_detail::apply_admitted_rain_events(
        initial, {event}, output, error));
    CHECK(error.find("exhausts rain number") != std::string::npos);
    CHECK(output.mass_per_kg == -7.0f);
    CHECK(output.number_per_kg == -7.0f);
    CHECK(output.reflectivity_per_kg == -7.0f);

    REQUIRE(tmv::milbrandt_detail::make_admitted_rain_number_correction_event(
        AcceptedRainEventKind::PreRainNumberRegularization,
        100.0, density, 1.0, event, error));
    const AcceptedRainEvent valid_correction = event;
    event.source_dry_density_kg_m3 = 1.0;
    CHECK_FALSE(tmv::milbrandt_detail::apply_admitted_rain_events(
        initial, {event}, output, error));
    CHECK(error.find("provenance") != std::string::npos);
    CHECK(output.mass_per_kg == -7.0f);

    const double mass_rate = -static_cast<double>(initial.mass_per_kg);
    const double number_rate = -static_cast<double>(initial.number_per_kg) * density;
    CHECK_FALSE(tmv::milbrandt_detail::make_admitted_rain_removal_event(
        AcceptedRainEventKind::RainToCloud, -mass_rate, mass_rate,
        0.0, number_rate, density, 1.0, event, error));
    CHECK(event.kind == AcceptedRainEventKind::PreRainNumberRegularization);
    CHECK_FALSE(tmv::milbrandt_detail::make_admitted_rain_removal_event(
        AcceptedRainEventKind::RainReset, -mass_rate, mass_rate,
        -number_rate, number_rate, density, 1.0, event, error));
    CHECK_FALSE(tmv::milbrandt_detail::make_admitted_rain_removal_event(
        AcceptedRainEventKind::RainReset, 0.0,
        -std::numeric_limits<double>::infinity(),
        0.0, number_rate, density, 1.0, event, error));
    REQUIRE(tmv::milbrandt_detail::make_admitted_rain_removal_event(
        AcceptedRainEventKind::RainToCloud, -mass_rate, mass_rate,
        -number_rate, number_rate, density, 1.0, event, error));
    auto partial = event;
    partial.admitted_mass_rate_per_kg_s *= 0.5;
    CHECK_FALSE(tmv::milbrandt_detail::apply_admitted_rain_events(
        initial, {partial}, output, error));
    CHECK(output.mass_per_kg == -7.0f);
    partial = event;
    partial.source_cloud_number_rate_per_m3_s = 0.0;
    CHECK_FALSE(tmv::milbrandt_detail::apply_admitted_rain_events(
        initial, {partial}, output, error));
    CHECK(output.mass_per_kg == -7.0f);
    CHECK_FALSE(tmv::milbrandt_detail::apply_admitted_rain_events(
        initial, {valid_correction, event}, output, error));
    CHECK(output.mass_per_kg == -7.0f);
    CHECK(output.number_per_kg == -7.0f);
    CHECK(output.reflectivity_per_kg == -7.0f);
}

TEST_CASE("independent named rain corrections are byte-identical at one and four workers",
          "[physics][milbrandt][triple_warm]")
{
    constexpr int cells = 16;
    std::array<RainTripleMomentState, cells> initial{};
    for (int cell = 0; cell < cells; ++cell)
    {
        const double mass = 1.0e-4 + cell * 1.0e-7;
        const double number = 1.0e4 + cell * 10.0;
        initial[cell] = {static_cast<float>(mass),
                         static_cast<float>(number),
                         static_cast<float>(gamma_z_per_kg(mass, number, 2.0))};
    }
    std::string error;
    AcceptedRainEvent regularization;
    AcceptedRainEvent clip;
    REQUIRE(tmv::milbrandt_detail::make_admitted_rain_number_correction_event(
        AcceptedRainEventKind::PostRainNumberRegularization,
        1.2e5, 1.2, 0.1, regularization, error));
    REQUIRE(tmv::milbrandt_detail::make_admitted_rain_number_correction_event(
        AcceptedRainEventKind::RainNumberClip,
        -1.2e4, 1.2, 0.1, clip, error));
    const std::vector<AcceptedRainEvent> events{regularization, clip};
    std::array<std::array<RainTripleMomentState, cells>, 2> outputs{};
    std::array<std::array<bool, cells>, 2> passed{};
    for (int run = 0; run < 2; ++run)
    {
        const int workers = run == 0 ? 1 : 4;
        #pragma omp parallel for num_threads(workers)
        for (int cell = 0; cell < cells; ++cell)
        {
            std::string local_error;
            passed[run][cell] = tmv::milbrandt_detail::apply_admitted_rain_events(
                initial[cell], events, outputs[run][cell], local_error);
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
