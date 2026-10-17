#include "catch2/catch.hpp"
#include "core/infra/allocation_telemetry.hpp"
#include "core/infra/sha256.hpp"
#include "microphysics/milbrandt_cold_parameters.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>

namespace
{

tmv::milbrandt_detail::ColdParameters cold_parameters(bool hail)
{
    const auto& p = tmv::milbrandt_detail::default_cold_parameters;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
            p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16],
            p[17], p[18], p[19], p[20], p[21], p[22], p[23], p[24], hail};
}

std::array<double, 16> cold_input(int index)
{
    std::array<double, 16> state{};
    state[0] = std::array<float, 12>{
        210, 223, 233, 238, 243, 253, 263, 268, 270, 273.15f,
        std::nextafter(273.15f, 300.0f), 280}[index % 12];
    state[1] = 50000 + 5000 * (index % 10);
    state[2] = .3f + .15f * (index % 7);
    const double saturation = thermodynamics::saturation_mixing_ratio_ice(
        state[0], state[1]);
    state[3] = float(saturation *
        std::array<double, 5>{0, .5, 1, 1.1, 1.5}[(index / 12) % 5]);
    for (int field = 0; field < 6; ++field)
    {
        state[4 + field] = std::array<float, 5>{
            0, 1e-10f, 1e-5f, .0005f, .003f}[(index + field) % 5];
        state[10 + field] = field == 0 ? 2e8f :
            std::array<float, 5>{0, 1, 100, 1e4f, 1e6f}[(index / 5 + field) % 5];
    }
    return state;
}

std::array<float, 14> cold_incoming_rates(
    int index, double dt, const std::array<double, 16>& state)
{
    std::array<float, 14> rates{};
    rates[0] = 1e-7f;
    rates[6] = -1e-7f;
    rates[7] = .001f;
    rates[8] = index % 2 == 0 ? float(1e6 / dt) : 0;
    rates[9] = float(state[11] / dt *
        std::array<double, 3>{-.5, 0, .2}[index % 3]);
    return rates;
}

struct ObservationCounts
{
    std::size_t accepted = 0;
    std::size_t rejected = 0;
    std::size_t events = 0;
    std::size_t hail = 0;
    std::size_t graupel = 0;
    std::size_t limited = 0;
    std::size_t incoming_number = 0;
};

std::string cold_rate_digest(bool observed = false, bool explicit_null = false,
                            ObservationCounts* counts = nullptr)
{
    nimbus::crypto::Sha256 hash;
    for (bool hail : {false, true})
    {
        const auto parameters = cold_parameters(hail);
        for (double dt : {.2, 1.0, 10.0})
        {
            for (int index = 0; index < 480; ++index)
            {
                const auto state = cold_input(index);
                auto rates = cold_incoming_rates(index, dt, state);
                const auto incoming = rates;
                tmv::milbrandt_detail::ColdRainFreezingAcceptedEventBuffer buffer;
                std::uint8_t outcome = 0;
                try
                {
                    if (observed || explicit_null)
                    {
                        tmv::milbrandt_detail::cold_cell(
                            parameters, state, dt, rates, observed ? &buffer : nullptr);
                    }
                    else
                    {
                        tmv::milbrandt_detail::cold_cell(parameters, state, dt, rates);
                    }
                    if (counts)
                    {
                        ++counts->accepted;
                    }
                }
                catch (const tmv::milbrandt_detail::ColdSnowRimingAdmissionError& error)
                {
                    using Failure = tmv::milbrandt_detail::ColdSnowRimingAdmissionFailure;
                    CAPTURE(hail, dt, index, error.what());
                    REQUIRE((error.kind() == Failure::BudgetCap || error.kind() == Failure::InvalidEndpoint));
                    CHECK(std::memcmp(rates.data(), incoming.data(), sizeof(rates)) == 0);
                    CHECK(buffer.size == 0);
                    outcome = static_cast<std::uint8_t>(error.kind()) + 1;
                    if (counts)
                    {
                        ++counts->rejected;
                    }
                }
                if (counts && buffer.size)
                {
                    const auto& event = buffer.events[0];
                    ++counts->events;
                    ++(hail ? counts->hail : counts->graupel);
                    const double raw_number = std::min(
                        event.entry_rain_number_per_kg,
                        -dt * 100.0 *
                            (std::exp(.66 * (state[0] - microphysics_constants::T0)) - 1.0) *
                            state[5] / microphysics_constants::rho_w);
                    if (event.admitted_number_increment_per_kg < raw_number * (1.0 - 1e-12))
                    {
                        ++counts->limited;
                    }
                    if (cold_incoming_rates(index, dt, state)[9] != 0.0f)
                    {
                        ++counts->incoming_number;
                    }
                }
                hash.update(&outcome, sizeof(outcome));
                hash.update(rates.data(), rates.size() * sizeof(float));
            }
        }
    }
    return hash.hex_digest();
}

std::array<double, 16> rain_only_input(double density)
{
    std::array<double, 16> state{};
    state[0] = 250.0;
    state[1] = 90000.0;
    state[2] = density;
    state[3] = thermodynamics::saturation_mixing_ratio_ice(state[0], state[1]);
    state[5] = .002;
    state[11] = .1;
    return state;
}

bool same_event(const tmv::milbrandt_detail::ColdRainFreezingAcceptedEvent& left,
                const tmv::milbrandt_detail::ColdRainFreezingAcceptedEvent& right)
{
    const std::array<double, 7> a{
        left.entry_rain_mass_per_kg, left.entry_rain_number_per_kg,
        left.admitted_mass_increment_per_kg, left.admitted_number_increment_per_kg,
        left.admitted_fraction, left.dry_density_kg_m3, left.duration_s};
    const std::array<double, 7> b{
        right.entry_rain_mass_per_kg, right.entry_rain_number_per_kg,
        right.admitted_mass_increment_per_kg, right.admitted_number_increment_per_kg,
        right.admitted_fraction, right.dry_density_kg_m3, right.duration_s};
    return left.destination == right.destination &&
        std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

} // namespace

TEST_CASE("Milbrandt corrected cold-stage outcomes are deterministic across the retained stress corpus",
          "[physics][milbrandt][cold_observer][baseline]")
{
    ObservationCounts counts;
    const auto digest = cold_rate_digest(false, false, &counts);
    std::cout << "[cold-observer-continuous-snow-rate-v4] sha256=" << digest
              << " accepted=" << counts.accepted << " rejected=" << counts.rejected << '\n';
    REQUIRE(counts.accepted + counts.rejected == 2880);
    CHECK(counts.accepted == 2756);
    CHECK(counts.rejected == 124);
    REQUIRE(digest == "aa6a0d7ce7f1abaccc956ab562b177c6b67c2f468a476e1aa2ec536b89502af5");
    REQUIRE(cold_rate_digest() == digest);
}

TEST_CASE("Milbrandt observed and explicit-null cold rates retain corrected outcome bytes",
          "[physics][milbrandt][cold_observer]")
{
    ObservationCounts counts;
    const auto observed = cold_rate_digest(true, false, &counts);
    const auto explicit_null = cold_rate_digest(false, true);
    std::cout << "[cold-observer-observed] sha256=" << observed
              << " events=" << counts.events << " hail=" << counts.hail
              << " graupel=" << counts.graupel << " limited=" << counts.limited
              << " incoming_number=" << counts.incoming_number << '\n';
    REQUIRE(observed == cold_rate_digest());
    REQUIRE(explicit_null == observed);
    CHECK(counts.hail > 0);
    CHECK(counts.graupel > 0);
    CHECK(counts.limited > 0);
    CHECK(counts.incoming_number > 0);
}

TEST_CASE("Rain-freezing observer carries accepted entry q N and physical units",
          "[physics][milbrandt][cold_observer]")
{
    using namespace tmv::milbrandt_detail;
    constexpr double dt = 2.0;
    for (bool hail : {false, true})
    {
        for (double density : {.4, 1.6})
        {
            const auto state = rain_only_input(density);
            std::array<float, 14> incoming{};
            incoming[9] = .025f;
            auto expected = incoming;
            auto observed = incoming;
            cold_cell(cold_parameters(hail), state, dt, expected);
            ColdRainFreezingAcceptedEventBuffer buffer;
            cold_cell(cold_parameters(hail), state, dt, observed, &buffer);
            REQUIRE(buffer.size == 1);
            CHECK(std::memcmp(observed.data(), expected.data(), sizeof(observed)) == 0);
            const auto& event = buffer.events[0];
            const double entry_number = state[11] + dt * incoming[9];
            const double raw_volume_increment = std::min(
                density * entry_number,
                -dt * 100.0 *
                    (std::exp(.66 * (state[0] - microphysics_constants::T0)) - 1.0) *
                    density * state[5] / microphysics_constants::rho_w);
            const double expected_number_increment = raw_volume_increment / density;
            const double expected_mass_increment =
                expected_number_increment * state[5] / entry_number;
            CHECK(event.destination == (hail ? ColdRainFreezingDestination::Hail :
                                               ColdRainFreezingDestination::Graupel));
            CHECK(event.entry_rain_mass_per_kg == state[5]);
            CHECK(event.entry_rain_number_per_kg == entry_number);
            CHECK(event.dry_density_kg_m3 == density);
            CHECK(event.duration_s == dt);
            CHECK(event.admitted_number_increment_per_kg == Approx(expected_number_increment));
            CHECK(event.admitted_mass_increment_per_kg == Approx(expected_mass_increment));
            CHECK(event.admitted_fraction == Approx(expected_number_increment / entry_number));
            CHECK(event.admitted_mass_increment_per_kg / state[5] == Approx(event.admitted_fraction));
            CHECK(density * event.admitted_number_increment_per_kg == Approx(raw_volume_increment));
            CHECK(observed[1] == static_cast<float>(-expected_mass_increment / dt));
            CHECK(observed[hail ? 5 : 4] == static_cast<float>(expected_mass_increment / dt));
            CHECK(observed[9] == incoming[9] - static_cast<float>(expected_number_increment / dt));
            CHECK(observed[hail ? 13 : 12] == static_cast<float>(expected_number_increment / dt));
            CHECK(observed[7] == Approx(microphysics_constants::L_f /
                microphysics_constants::cp * expected_mass_increment / dt));
        }
    }
}

TEST_CASE("Rain-freezing observer uses post-limiter q N without changing cold rates",
          "[physics][milbrandt][cold_observer]")
{
    using namespace tmv::milbrandt_detail;
    bool found = false;
    for (bool hail : {false, true})
    {
        bool route_found = false;
        for (double dt : {.2, 1.0, 10.0})
        {
            for (int index = 0; index < 480 && !route_found; ++index)
            {
                const auto state = cold_input(index);
                auto rates = cold_incoming_rates(index, dt, state);
                auto expected = rates;
                cold_cell(cold_parameters(hail), state, dt, expected);
                ColdRainFreezingAcceptedEventBuffer buffer;
                cold_cell(cold_parameters(hail), state, dt, rates, &buffer);
                if (!buffer.size)
                {
                    continue;
                }
                const auto& event = buffer.events[0];
                const double raw_volume_increment = std::min(
                    state[2] * event.entry_rain_number_per_kg,
                    -dt * 100.0 *
                        (std::exp(.66 * (state[0] - microphysics_constants::T0)) - 1.0) *
                        state[2] * state[5] / microphysics_constants::rho_w);
                const double raw_number = raw_volume_increment / state[2];
                const double limiter = event.admitted_number_increment_per_kg / raw_number;
                if (!(limiter < .9))
                {
                    continue;
                }
                route_found = true;
                found = true;
                CHECK(std::memcmp(rates.data(), expected.data(), sizeof(rates)) == 0);
                CHECK(event.entry_rain_number_per_kg == std::max(
                    0.0, state[11] + dt * cold_incoming_rates(index, dt, state)[9]));
                CHECK(event.admitted_fraction > 0.0);
                CHECK(event.admitted_fraction < 1.0);
                CHECK(event.admitted_mass_increment_per_kg / state[5] == Approx(event.admitted_fraction));
                CHECK(event.admitted_fraction == Approx(limiter * raw_number /
                    event.entry_rain_number_per_kg));
                std::cout << std::setprecision(17)
                          << "[cold-observer-limited] hail=" << hail << " cell=" << index
                          << " dt=" << dt << " rain=" << state[5]
                          << " entry_number=" << event.entry_rain_number_per_kg
                          << " raw_number=" << raw_number << " admitted_number="
                          << event.admitted_number_increment_per_kg
                          << " admitted_mass=" << event.admitted_mass_increment_per_kg
                          << " donor_scale=" << limiter << '\n';
            }
        }
        CHECK(route_found);
    }
    CHECK(found);
}

TEST_CASE("Rain-freezing observer rejects publication failures atomically",
          "[physics][milbrandt][cold_observer]")
{
    using namespace tmv::milbrandt_detail;
    const auto state = rain_only_input(.4);
    std::array<float, 14> incoming{};
    incoming[9] = .025f;
    for (int invalid = 0; invalid < 6; ++invalid)
    {
        auto rates = incoming;
        auto input = state;
        double dt = 2.0;
        ColdRainFreezingAcceptedEventBuffer buffer;
        buffer.events[0].duration_s = 123.0;
        if (invalid == 0)
        {
            buffer.capacity = 0;
        }
        else if (invalid == 1)
        {
            buffer.size = 1;
        }
        else if (invalid == 2)
        {
            buffer.capacity = 2;
        }
        else if (invalid == 3)
        {
            buffer.size = 2;
        }
        else if (invalid == 4)
        {
            dt = 0.0;
        }
        else
        {
            input[2] = std::numeric_limits<double>::quiet_NaN();
        }
        const auto before_buffer = buffer;
        REQUIRE_THROWS(cold_cell(cold_parameters(true), input, dt, rates, &buffer));
        CHECK(std::memcmp(rates.data(), incoming.data(), sizeof(rates)) == 0);
        CHECK(std::memcmp(&buffer, &before_buffer, sizeof(buffer)) == 0);
    }
    ColdRainFreezingAcceptedEventBuffer no_event;
    no_event.capacity = 0;
    auto warm_input = state;
    warm_input[0] = 280.0;
    auto warm_rates = incoming;
    const auto before_no_event = no_event;
    REQUIRE_NOTHROW(cold_cell(cold_parameters(true), warm_input, 2.0,
                             warm_rates, &no_event));
    CHECK(std::memcmp(warm_rates.data(), incoming.data(), sizeof(warm_rates)) == 0);
    CHECK(std::memcmp(&no_event, &before_no_event, sizeof(no_event)) == 0);
}

TEST_CASE("Rain-freezing event helper rejects incoherent fractions and units",
          "[physics][milbrandt][cold_observer]")
{
    using namespace tmv::milbrandt_detail;
    ColdRainFreezingAcceptedEvent result;
    result.duration_s = 123.0;
    const auto before = result;
    REQUIRE_THROWS_WITH(result = make_cold_rain_freezing_accepted_event(
        ColdRainFreezingDestination::Hail, .002, 100.0, .001, 25.0, .4, 2.0),
        Catch::Matchers::Contains("q/N fractions differ"));
    CHECK(same_event(result, before));
    for (int field = 0; field < 6; ++field)
    {
        std::array<double, 6> values{.002, 100.0, .001, 50.0, .4, 2.0};
        values[field] = std::numeric_limits<double>::quiet_NaN();
        REQUIRE_THROWS(result = make_cold_rain_freezing_accepted_event(
            ColdRainFreezingDestination::Hail, values[0], values[1], values[2],
            values[3], values[4], values[5]));
        CHECK(same_event(result, before));
    }
    REQUIRE_THROWS(make_cold_rain_freezing_accepted_event(
        static_cast<ColdRainFreezingDestination>(9), .002, 100.0, .001, 50.0, .4, 2.0));
    REQUIRE_THROWS(make_cold_rain_freezing_accepted_event(
        ColdRainFreezingDestination::Hail, .002, 100.0, .003, 150.0, .4, 2.0));
    const auto complete = make_cold_rain_freezing_accepted_event(
        ColdRainFreezingDestination::Graupel, .002, 100.0, .002, 100.0, .4, 2.0);
    CHECK(complete.admitted_fraction == 1.0);
}

TEST_CASE("Rain-freezing observed rates and events are one-four-worker identical",
          "[physics][milbrandt][cold_observer]")
{
    using namespace tmv::milbrandt_detail;
    constexpr int count = 64;
    std::array<std::array<std::array<float, 14>, count>, 2> rates{};
    std::array<std::array<ColdRainFreezingAcceptedEventBuffer, count>, 2> buffers{};
    std::array<std::array<bool, count>, 2> succeeded{};
    std::array<std::array<int, count>, 2> rejected{};
    for (int run = 0; run < 2; ++run)
    {
        const int workers = run ? 4 : 1;
        #pragma omp parallel for num_threads(workers)
        for (int index = 0; index < count; ++index)
        {
            try
            {
                const auto state = cold_input(120 + index);
                rates[run][index] = cold_incoming_rates(index, .2, state);
                cold_cell(cold_parameters(index % 2), state, .2,
                          rates[run][index], &buffers[run][index]);
                succeeded[run][index] = true;
            }
            catch (const ColdSnowRimingAdmissionError& error)
            {
                rejected[run][index] = static_cast<int>(error.kind()) + 1;
            }
            catch (...)
            {
                succeeded[run][index] = false;
            }
        }
    }
    for (int index = 0; index < count; ++index)
    {
        REQUIRE(succeeded[0][index] == succeeded[1][index]);
        REQUIRE(rejected[0][index] == rejected[1][index]);
        if (!succeeded[0][index])
        {
            REQUIRE((rejected[0][index] == static_cast<int>(ColdSnowRimingAdmissionFailure::BudgetCap) + 1
                || rejected[0][index] == static_cast<int>(ColdSnowRimingAdmissionFailure::InvalidEndpoint) + 1));
            const auto state = cold_input(120 + index);
            const auto incoming = cold_incoming_rates(index, .2, state);
            CHECK(std::memcmp(rates[0][index].data(), incoming.data(), sizeof(incoming)) == 0);
            CHECK(buffers[0][index].size == 0);
        }
        CHECK(std::memcmp(rates[0][index].data(), rates[1][index].data(),
                          sizeof(rates[0][index])) == 0);
        CHECK(buffers[0][index].size == buffers[1][index].size);
        if (buffers[0][index].size)
        {
            CHECK(same_event(buffers[0][index].events[0], buffers[1][index].events[0]));
        }
    }
}

TEST_CASE("Milbrandt null cold observer performs no hot-path heap allocation",
          "[physics][milbrandt][cold_observer][baseline]")
{
    const auto parameters = cold_parameters(true);
    const auto state = cold_input(123);
    auto warmup = cold_incoming_rates(123, .2, state);
    tmv::milbrandt_detail::cold_cell(parameters, state, .2, warmup);
    volatile double checksum = 0.0;
    tmv::allocation_telemetry::begin_measurement();
    for (int repetition = 0; repetition < 1000; ++repetition)
    {
        auto rates = cold_incoming_rates(123, .2, state);
        tmv::milbrandt_detail::cold_cell(parameters, state, .2, rates);
        checksum += rates[1];
    }
    const auto allocation = tmv::allocation_telemetry::end_measurement();
    CHECK(allocation.heap_allocations == 0);
    CHECK(allocation.heap_requested_bytes == 0);
    CHECK(std::isfinite(checksum));
}

TEST_CASE("Milbrandt accepted rain-freezing observer performs no hot-path heap allocation",
          "[physics][milbrandt][cold_observer]")
{
    using namespace tmv::milbrandt_detail;
    const auto parameters = cold_parameters(true);
    const auto state = rain_only_input(.4);
    volatile double checksum = 0.0;
    tmv::allocation_telemetry::begin_measurement();
    for (int repetition = 0; repetition < 1000; ++repetition)
    {
        std::array<float, 14> rates{};
        ColdRainFreezingAcceptedEventBuffer buffer;
        cold_cell(parameters, state, 2.0, rates, &buffer);
        checksum += buffer.events[0].admitted_mass_increment_per_kg;
    }
    const auto allocation = tmv::allocation_telemetry::end_measurement();
    CHECK(allocation.heap_allocations == 0);
    CHECK(allocation.heap_requested_bytes == 0);
    CHECK(std::isfinite(checksum));
    CHECK(checksum > 0.0);
}
