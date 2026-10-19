#include "catch2/catch.hpp"
#include "core/infra/allocation_telemetry.hpp"
#include "core/infra/sha256.hpp"
#include "microphysics/schemes/milbrandt/warm_rain.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>

namespace
{

bool same_event_fields(const tmv::milbrandt_detail::WarmRainAcceptedEvent& left,
                       const tmv::milbrandt_detail::WarmRainAcceptedEvent& right)
{
    const std::array<double, 9> left_values{
        left.cloud_mass_rate_per_kg_s, left.rain_mass_rate_per_kg_s,
        left.vapor_mass_rate_per_kg_s, left.cloud_number_rate_per_m3_s,
        left.rain_number_rate_per_m3_s, left.temperature_rate_k_s,
        left.dry_density_kg_m3, left.cloud_donor_scale,
        left.seed_diameter_m};
    const std::array<double, 9> right_values{
        right.cloud_mass_rate_per_kg_s, right.rain_mass_rate_per_kg_s,
        right.vapor_mass_rate_per_kg_s, right.cloud_number_rate_per_m3_s,
        right.rain_number_rate_per_m3_s, right.temperature_rate_k_s,
        right.dry_density_kg_m3, right.cloud_donor_scale,
        right.seed_diameter_m};
    return left.kind == right.kind &&
        left.seed_mass_number_coherent == right.seed_mass_number_coherent &&
        std::memcmp(left_values.data(), right_values.data(),
                    left_values.size() * sizeof(double)) == 0;
}

std::string warm_rate_digest(
    bool observed = false,
    std::array<std::size_t, 15>* event_counts = nullptr)
{
    using namespace tmv::milbrandt_detail;
    const auto parameters = warm_rain_parameters(
        0.0, 523.598775598, 3.0, 149.1, 0.5);
    nimbus::crypto::Sha256 hash;
    for (double dt : {0.2, 1.0, 10.0})
    {
        for (int cell = 0; cell < 240; ++cell)
        {
            const float temperature = 250.0f + 10.0f * (cell % 7);
            const float pressure = 60000.0f + 5000.0f * (cell % 8);
            const float cloud = cell % 3 == 0 ? 0.0f
                : cell % 3 == 1 ? 0.001f : 0.004f;
            const float rain = cell % 5 == 0 ? 0.0f : 0.001f;
            const double diameter =
                std::array<double, 6>{60e-6, 90e-6, 100e-6,
                                      500e-6, 3.5e-3, 6e-3}[cell % 6];
            const double saturation =
                thermodynamics::saturation_mixing_ratio_water(
                    temperature, pressure);
            const float vapor = static_cast<float>(saturation *
                (cell % 4 == 0 ? 0.0 : cell % 4 == 1 ? 0.5
                 : cell % 4 == 2 ? 1.0 : 1.05));
            const float density = 0.3f + 0.15f * (cell % 7);
            const float cloud_number = cell % 3 == 0 ? 0.0f
                : cell % 3 == 1 ? 2e8f : 1e5f;
            const float rain_number = static_cast<float>(
                rain / (parameters.c_r * std::pow(diameter, 3.0)));
            WarmRainCell state{temperature, pressure, density, vapor,
                               cloud, rain, cloud_number, rain_number};
            WarmRainTendencies rates{1e-6f, 0.0f, -1e-6f, 0.01f,
                                     0.125f, 0.0f};
            if (cell % 3 == 2 && cell % 2 == 0)
            {
                state.cloud_number = 0.0f;
                rates.cloud_number = static_cast<float>(1e5 / dt);
            }
            WarmRainAcceptedEventBuffer buffer;
            warm_rain_cell(parameters, state, dt, rates,
                           observed ? &buffer : nullptr);
            if (event_counts)
            {
                for (std::size_t index = 0; index < buffer.size; ++index)
                {
                    ++(*event_counts)[static_cast<std::size_t>(buffer.events[index].kind)];
                }
            }
            const std::array<float, 6> packed{
                rates.cloud, rates.rain, rates.vapor,
                rates.temperature, rates.cloud_number,
                rates.rain_number};
            hash.update(packed.data(), packed.size() * sizeof(float));
        }
    }
    return hash.hex_digest();
}

} // namespace

TEST_CASE("Milbrandt no-observer warm-stage rate bytes retain baseline",
          "[physics][milbrandt][warm_observer]")
{
    const std::string digest = warm_rate_digest();
    std::cout << "[warm-observer-baseline] sha256=" << digest << '\n';
    REQUIRE(digest == "b54d9d68d2bb4ccd5642208482cd23e05b1842bc3c8936a0b1210d06c8898a6d");
}

TEST_CASE("Milbrandt observed warm-stage rate bytes match the no-observer baseline",
          "[physics][milbrandt][warm_observer]")
{
    std::array<std::size_t, 15> counts{};
    const std::string digest = warm_rate_digest(true, &counts);
    std::cout << "[warm-observer-observed] sha256=" << digest << '\n';
    std::cout << "[warm-observer-events]";
    for (std::size_t count : counts)
    {
        std::cout << ' ' << count;
    }
    std::cout << '\n';
    REQUIRE(digest == "b54d9d68d2bb4ccd5642208482cd23e05b1842bc3c8936a0b1210d06c8898a6d");
    CHECK(counts[static_cast<std::size_t>(tmv::milbrandt_detail::WarmRainAcceptedEventKind::Autoconversion)] > 0);
    CHECK(counts[static_cast<std::size_t>(tmv::milbrandt_detail::WarmRainAcceptedEventKind::Evaporation)] > 0);
    CHECK(counts[static_cast<std::size_t>(tmv::milbrandt_detail::WarmRainAcceptedEventKind::PreRainNumberRegularization)] > 0);
    CHECK(counts[static_cast<std::size_t>(tmv::milbrandt_detail::WarmRainAcceptedEventKind::PostRainNumberRegularization)] > 0);
    CHECK(counts[static_cast<std::size_t>(tmv::milbrandt_detail::WarmRainAcceptedEventKind::CloudNumberClip)] > 0);
    CHECK(counts[static_cast<std::size_t>(tmv::milbrandt_detail::WarmRainAcceptedEventKind::RainToCloud)] > 0);
}

TEST_CASE("warm observer records actual limited autoconversion pair and fails capacity atomically",
          "[physics][milbrandt][warm_observer]")
{
    using namespace tmv::milbrandt_detail;
    const auto parameters = warm_rain_parameters(
        0.0, 523.598775598, 3.0, 149.1, 0.5);
    bool found = false;
    WarmRainCell selected{};
    WarmRainTendencies selected_rates{};
    WarmRainTendencies selected_result{};
    double selected_dt = 0.0;
    WarmRainAcceptedEvent selected_auto{};
    WarmRainAcceptedEventBuffer selected_buffer;
    for (double dt : {0.2, 1.0, 10.0})
    {
        for (float cloud : {0.001f, 0.003f, 0.01f})
        {
            for (float rain : {0.0005f, 0.002f})
            {
                for (float cloud_number : {1.0e5f, 1.0e7f, 1.0e9f})
                {
                    for (float rain_number : {1.0e3f, 1.0e5f})
                    {
                        const float temperature = 280.0f;
                        const float pressure = 90000.0f;
                        const float density = 1.0f;
                        const float saturation = static_cast<float>(
                            thermodynamics::saturation_mixing_ratio_water(
                                temperature, pressure));
                        WarmRainCell cell{temperature, pressure, density,
                                          0.5f * saturation, cloud, rain,
                                          cloud_number, rain_number};
                        WarmRainTendencies rates{};
                        WarmRainAcceptedEventBuffer buffer;
                        warm_rain_cell(parameters, cell, dt, rates, &buffer);
                        for (std::size_t index = 0; index < buffer.size; ++index)
                        {
                            const auto& event = buffer.events[index];
                            if (event.kind == WarmRainAcceptedEventKind::Autoconversion &&
                                event.cloud_donor_scale < 1.0 &&
                                !event.seed_mass_number_coherent &&
                                (!found || event.cloud_donor_scale <
                                    selected_auto.cloud_donor_scale))
                            {
                                found = true;
                                selected = cell;
                                selected_rates = {};
                                selected_result = rates;
                                selected_dt = dt;
                                selected_auto = event;
                                selected_buffer = buffer;
                            }
                        }
                    }
                }
            }
        }
    }
    REQUIRE(found);
    REQUIRE(selected_buffer.size > 1);
    CHECK(selected_auto.rain_mass_rate_per_kg_s > 0.0);
    CHECK(selected_auto.rain_number_rate_per_m3_s > 0.0);
    const double coherent_number =
        selected_auto.rain_mass_rate_per_kg_s *
        selected_auto.dry_density_kg_m3 /
        (parameters.c_r * std::pow(selected_auto.seed_diameter_m, parameters.d_r));
    CHECK(selected_auto.rain_number_rate_per_m3_s > coherent_number);
    std::cout << std::setprecision(17)
              << "[warm-observer-limited-autoconversion] dt=" << selected_dt
              << " cloud=" << selected.cloud
              << " rain=" << selected.rain
              << " cloud_number=" << selected.cloud_number
              << " rain_number=" << selected.rain_number
              << " donor_scale=" << selected_auto.cloud_donor_scale
              << " admitted_q_rate=" << selected_auto.rain_mass_rate_per_kg_s
              << " actual_number_volume_rate="
              << selected_auto.rain_number_rate_per_m3_s
              << " seed_diameter_m=" << selected_auto.seed_diameter_m
              << '\n';
    CHECK(std::abs(selected_auto.rain_number_rate_per_m3_s /
                   coherent_number - 1.0 / selected_auto.cloud_donor_scale) < 1.0e-10);
    std::array<double, 6> sums{};
    for (std::size_t index = 0; index < selected_buffer.size; ++index)
    {
        const auto& event = selected_buffer.events[index];
        sums[0] += event.cloud_mass_rate_per_kg_s;
        sums[1] += event.rain_mass_rate_per_kg_s;
        sums[2] += event.vapor_mass_rate_per_kg_s;
        sums[3] += event.temperature_rate_k_s;
        sums[4] += event.cloud_number_rate_per_m3_s /
            selected.density;
        sums[5] += event.rain_number_rate_per_m3_s /
            selected.density;
    }
    const std::array<float, 6> published{
        selected_result.cloud, selected_result.rain,
        selected_result.vapor, selected_result.temperature,
        selected_result.cloud_number, selected_result.rain_number};
    for (std::size_t field = 0; field < sums.size(); ++field)
    {
        const double scale = std::max(1.0e-10, std::abs(sums[field]));
        CHECK(std::abs(static_cast<double>(published[field]) - sums[field]) <=
              4.0 * std::numeric_limits<float>::epsilon() * scale);
    }

    WarmRainAcceptedEventBuffer limited;
    limited.capacity = 1;
    const WarmRainAcceptedEventBuffer before_buffer = limited;
    const WarmRainTendencies before_rates = selected_rates;
    REQUIRE_THROWS_WITH(warm_rain_cell(
        parameters, selected, selected_dt, selected_rates, &limited),
        Catch::Matchers::Contains("observer capacity is exhausted"));
    CHECK(std::memcmp(&selected_rates, &before_rates,
                      sizeof(WarmRainTendencies)) == 0);
    CHECK(limited.size == before_buffer.size);
    CHECK(limited.capacity == before_buffer.capacity);
    CHECK(std::memcmp(limited.events.data(), before_buffer.events.data(),
                      sizeof(limited.events)) == 0);
}

TEST_CASE("warm observer emits reset and rejects malformed buffer without rate publication",
          "[physics][milbrandt][warm_observer]")
{
    using namespace tmv::milbrandt_detail;
    const auto parameters = warm_rain_parameters(
        0.0, 523.598775598, 3.0, 149.1, 0.5);
    const WarmRainCell cell{280.0f, 90000.0f, 1.0f,
                            0.01f, 0.0f, 1.0e-15f,
                            0.0f, 100.0f};
    WarmRainTendencies rates{};
    WarmRainAcceptedEventBuffer buffer;
    warm_rain_cell(parameters, cell, 1.0, rates, &buffer);
    bool reset_found = false;
    for (std::size_t index = 0; index < buffer.size; ++index)
    {
        if (buffer.events[index].kind == WarmRainAcceptedEventKind::RainReset)
        {
            reset_found = true;
            CHECK(buffer.events[index].rain_mass_rate_per_kg_s < 0.0);
            CHECK(buffer.events[index].rain_number_rate_per_m3_s < 0.0);
        }
    }
    CHECK(reset_found);
    WarmRainAcceptedEventBuffer malformed;
    malformed.size = malformed.capacity + 1;
    WarmRainTendencies unchanged{};
    const WarmRainTendencies before = unchanged;
    REQUIRE_THROWS_WITH(warm_rain_cell(
        parameters, cell, 1.0, unchanged, &malformed),
        Catch::Matchers::Contains("observer capacity is exhausted"));
    CHECK(std::memcmp(&unchanged, &before,
                      sizeof(WarmRainTendencies)) == 0);
}

TEST_CASE("observed warm cells and fixed event buffers are byte-identical at one and four workers",
          "[physics][milbrandt][warm_observer]")
{
    using namespace tmv::milbrandt_detail;
    constexpr int cells = 16;
    const auto parameters = warm_rain_parameters(
        0.0, 523.598775598, 3.0, 149.1, 0.5);
    std::array<WarmRainCell, cells> inputs{};
    for (int cell = 0; cell < cells; ++cell)
    {
        const float temperature = 255.0f + 5.0f * (cell % 6);
        const float pressure = 60000.0f + 4000.0f * (cell % 5);
        const float saturation = static_cast<float>(
            thermodynamics::saturation_mixing_ratio_water(
                temperature, pressure));
        inputs[cell] = {temperature, pressure,
                        0.5f + 0.05f * (cell % 5),
                        saturation * (cell % 2 == 0 ? 0.5f : 1.05f),
                        cell % 3 == 0 ? 0.004f : 0.001f,
                        cell % 4 == 0 ? 0.0f : 0.001f,
                        cell % 3 == 0 ? 1e5f : 2e8f,
                        cell % 4 == 0 ? 0.0f : 1e5f};
    }
    std::array<std::array<WarmRainTendencies, cells>, 2> rates{};
    std::array<std::array<WarmRainAcceptedEventBuffer, cells>, 2> buffers{};
    std::array<std::array<bool, cells>, 2> passed{};
    for (int run = 0; run < 2; ++run)
    {
        const int workers = run == 0 ? 1 : 4;
        #pragma omp parallel for num_threads(workers)
        for (int cell = 0; cell < cells; ++cell)
        {
            try
            {
                warm_rain_cell(parameters, inputs[cell], 0.2,
                               rates[run][cell], &buffers[run][cell]);
                passed[run][cell] = true;
            }
            catch (...)
            {
                passed[run][cell] = false;
            }
        }
    }
    for (int cell = 0; cell < cells; ++cell)
    {
        REQUIRE(passed[0][cell]);
        REQUIRE(passed[1][cell]);
        CHECK(std::memcmp(&rates[0][cell], &rates[1][cell],
                          sizeof(WarmRainTendencies)) == 0);
        CHECK(buffers[0][cell].size == buffers[1][cell].size);
        for (std::size_t event = 0; event < buffers[0][cell].size; ++event)
        {
            CHECK(same_event_fields(buffers[0][cell].events[event],
                                    buffers[1][cell].events[event]));
        }
    }
}

TEST_CASE("Milbrandt null observer performs no hot-path heap allocation",
          "[physics][milbrandt][warm_observer]")
{
    using namespace tmv::milbrandt_detail;
    const auto parameters = warm_rain_parameters(
        0.0, 523.598775598, 3.0, 149.1, 0.5);
    const float saturation = static_cast<float>(
        thermodynamics::saturation_mixing_ratio_water(280.0f, 90000.0f));
    const WarmRainCell cell{280.0f, 90000.0f, 1.0f,
                            0.5f * saturation, 0.003f, 0.001f,
                            1.0e7f, 1.0e5f};
    WarmRainTendencies warmup{};
    warm_rain_cell(parameters, cell, 0.2, warmup);
    volatile double checksum = 0.0;
    tmv::allocation_telemetry::begin_measurement();
    for (int repetition = 0; repetition < 1000; ++repetition)
    {
        WarmRainTendencies rates{};
        warm_rain_cell(parameters, cell, 0.2, rates);
        checksum += rates.rain;
    }
    const auto allocation = tmv::allocation_telemetry::end_measurement();
    CHECK(allocation.heap_allocations == 0);
    CHECK(allocation.heap_requested_bytes == 0);
    CHECK(std::isfinite(checksum));
}
