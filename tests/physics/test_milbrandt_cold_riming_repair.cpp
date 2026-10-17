#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "support/milbrandt_cold_cloud_collection_reference.hpp"
#include "support/milbrandt_double_riming_reference.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <iostream>
#include <string>

namespace
{

namespace reference = milbrandt_cold_cloud_collection_reference;
using namespace tmv::milbrandt_detail;

ColdParameters retained_parameters(bool hail = true)
{
    const auto& p = default_cold_parameters;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], hail};
}

std::array<double, 16> input(double density, int selector = 3)
{
    std::array<double, 16> cell{};
    cell[0] = 253;
    cell[1] = 80000;
    cell[2] = density;
    cell[3] = thermodynamics::saturation_mixing_ratio_ice(cell[0], cell[1]);
    cell[4] = 2e-4;
    cell[10] = 8e7;
    for (int species = 0; species < 3; ++species)
    {
        if (selector == 3 || selector == species)
        {
            cell[7 + species] = std::array<double, 3>{1e-4, 2e-4, 3e-4}[species];
            cell[13 + species] = std::array<double, 3>{5000, 600, 80}[species];
        }
    }
    return cell;
}

ColdCloudCollectionCell reference_cell(const std::array<double, 16>& cell)
{
    ColdCloudCollectionCell source;
    source.temperature_k = cell[0];
    source.pressure_pa = cell[1];
    source.dry_density_kg_m3 = cell[2];
    source.initial.cloud_mass_per_kg = cell[4];
    source.initial.cloud_number_per_kg = cell[10];
    for (std::size_t species = 0; species < 3; ++species)
    {
        source.initial.frozen[species] = reference::population(species, cell[7 + species], cell[13 + species], 0);
    }
    return source;
}

std::array<double, 16> stress_input(int index)
{
    std::array<double, 16> cell{};
    cell[0] = std::array<float, 12>{210, 223, 233, 238, 243, 253, 263, 268, 270,
        273.15f, std::nextafter(273.15f, 300.0f), 280}[index % 12];
    cell[1] = 50000 + 5000 * (index % 10);
    cell[2] = .3f + .15f * (index % 7);
    cell[3] = static_cast<float>(thermodynamics::saturation_mixing_ratio_ice(cell[0], cell[1])
        * std::array<double, 5>{0, .5, 1, 1.1, 1.5}[(index / 12) % 5]);
    for (int field = 0; field < 6; ++field)
    {
        cell[4 + field] = std::array<float, 5>{0, 1e-10f, 1e-5f, .0005f, .003f}[(index + field) % 5];
        cell[10 + field] = field == 0 ? 2e8f
            : std::array<float, 5>{0, 1, 100, 1e4f, 1e6f}[(index / 5 + field) % 5];
    }
    return cell;
}

void near(float actual, long double expected)
{
    INFO("actual " << actual << ", independent " << static_cast<double>(expected));
    REQUIRE(std::isfinite(actual));
    CHECK(std::abs(static_cast<long double>(actual) - expected)
        <= 3e-7L * std::max(std::abs(expected), 1e-30L));
}

} // namespace

TEST_CASE("retained double cloud collection includes the frozen fall exponent in all gamma terms",
          "[physics][milbrandt][cold_riming_repair]")
{
    constexpr double duration = 1e-3;
    for (bool hail : {false, true})
    {
        for (double density : {.4, .6, 1.0, 1.6})
        {
            for (int selector = 0; selector < 4; ++selector)
            {
                const auto cell = input(density, selector);
                const auto expected = reference::evaluate(reference_cell(cell), duration);
                std::array<float, 14> rates{};
                REQUIRE_NOTHROW(cold_cell(retained_parameters(hail), cell, duration, rates));
                near(rates[0], expected.cloud_mass_increment / duration);
                near(rates[8], expected.cloud_number_increment / duration);
                long double mass = 0;
                for (std::size_t index = 0; index < 7; ++index)
                {
                    mass += rates[index];
                }
                long double scale = 0;
                for (std::size_t index = 0; index < 7; ++index)
                {
                    scale += std::abs(rates[index]);
                }
                CHECK(std::abs(mass) <= 2e-7L * scale);
            }
        }
    }
}

TEST_CASE("retained double cloud rate agrees with independent velocity weighted quadrature",
          "[physics][milbrandt][cold_riming_repair]")
{
    constexpr double duration = 1e-3;
    for (int selector = 0; selector < 3; ++selector)
    {
        const auto cell = input(.6, selector);
        const auto source = reference_cell(cell);
        const auto expected = reference::evaluate(source, duration, true);
        const auto old = reference::missing_fall_exponent_kernel(source, expected, selector, duration);
        REQUIRE(std::abs(old[0] / expected.channels[selector].raw_mass - 1) > .1);
        REQUIRE(std::abs(old[1] / expected.channels[selector].raw_number - 1) > .1);
        std::array<float, 14> rates{};
        REQUIRE_NOTHROW(cold_cell(retained_parameters(), cell, duration, rates));
        near(rates[0], expected.cloud_mass_increment / duration);
        near(rates[8], expected.cloud_number_increment / duration);
    }
}

TEST_CASE("double snow conversion follows the published continuous mass rate under interval refinement",
          "[physics][milbrandt][double_snow_conversion_rate]")
{
    for (bool hail : {false, true})
    {
        for (double density : {.4, .6, 1.0, 1.6})
        {
            auto cell = input(density, 0);
            cell[3] *= .99;
            double prior_rate = 0.0;
            for (double duration : {.004, .001, .00025})
            {
                const auto collection = reference::evaluate(reference_cell(cell), duration);
                const long double collected = collection.channels[0].mass;
                const long double converted = 400.0L / (400.0L - 100.0L) * collected;
                REQUIRE(converted < cell[7]);
                std::array<float, 14> rates{};
                ColdSnowRimingAdmissionObservation observation;
                REQUIRE_NOTHROW(cold_cell(retained_parameters(hail), cell, duration, rates, nullptr, &observation));
                REQUIRE(observation.snow_conversion_predicate);
                near(rates[4], converted / duration);
                near(rates[12], cell[13] * converted / (cell[7] + collected) / duration);
                if (prior_rate > 0.0)
                {
                    near(rates[4], prior_rate);
                }
                prior_rate = rates[4];
                const auto thermal = milbrandt_double_riming_reference::snow_diffusion(cell, duration, collected);
                near(rates[3], (collected + thermal.environmental + thermal.heating - converted) / duration);
                for (std::size_t donor = 0; donor < 7; ++donor)
                {
                    CHECK(observation.donor_mass_sink_per_kg[donor] <= observation.donor_mass_source_per_kg[donor]);
                }
            }
        }
    }
}

TEST_CASE("snow riming heat is bound to accepted cloud mass with physical units and sign",
          "[physics][milbrandt][cold_riming_repair]")
{
    namespace thermal = milbrandt_double_riming_reference;
    constexpr double duration = 1e-3;
    for (double density : {.4, .6, 1.0, 1.6})
    {
        for (double saturation : {.99, 1.0, 1.00001})
        {
            auto cell = input(density, 0);
            cell[3] *= saturation;
            const auto collection = reference::evaluate(reference_cell(cell), duration);
            const auto expected = thermal::snow_diffusion(cell, duration, collection.channels[0].mass);
            std::array<float, 14> rates{};
            ColdSnowRimingAdmissionObservation observation;
            REQUIRE_NOTHROW(cold_cell(retained_parameters(), cell, duration, rates, nullptr, &observation));
            REQUIRE(observation.active);
            REQUIRE(observation.admission_passes > 0);
            CHECK(observation.admission_passes <= cold_snow_riming_admission_max_passes);
            const long double heating_error = std::abs(observation.riming_snow_mass_increment_per_kg - expected.heating);
            CHECK(heating_error <= 2e-8L * std::abs(expected.heating));
            CHECK(std::abs(observation.raw_ambient_snow_mass_increment_per_kg - expected.environmental)
                <= 2e-12L * cell[7]);
            CHECK(std::abs(observation.raw_heated_snow_mass_increment_per_kg
                - observation.raw_ambient_snow_mass_increment_per_kg
                - observation.riming_snow_mass_increment_per_kg) <= 2e-20L);
            if (saturation == 1)
            {
                REQUIRE(observation.admitted_snow_mass_increment_per_kg < 0);
                CHECK(observation.snow_conversion_predicate);
            }
            CHECK(observation.final_snow_mass_per_kg > 0);
            CHECK(observation.final_snow_number_per_kg > 0);
            for (std::size_t donor = 0; donor < 7; ++donor)
            {
                CHECK(observation.donor_mass_sink_per_kg[donor] <= observation.donor_mass_source_per_kg[donor]);
            }
            CHECK(observation.snow_number_sink_per_kg <= observation.snow_number_source_per_kg);
            if (saturation <= 1)
            {
                near(rates[6], -(expected.environmental + expected.heating) / duration);
                near(rates[7], (thermal::coupled::latent_heat_sublimation * (expected.environmental + expected.heating)
                    + reference::latent_heat_fusion * collection.channels[0].mass) / (reference::heat_capacity * duration));
            }
            const long double thermal_power = reference::heat_capacity * rates[7];
            const long double vapor_power = thermal::coupled::latent_heat_sublimation * rates[6];
            const long double cloud_power = reference::latent_heat_fusion * rates[0];
            CHECK(std::abs(thermal_power + vapor_power + cloud_power)
                <= 2 * std::numeric_limits<float>::epsilon()
                    * (std::abs(thermal_power) + std::abs(vapor_power) + std::abs(cloud_power)));
        }
    }
}

TEST_CASE("snow riming observer is passive and invalid resource rejects before rate publication",
          "[physics][milbrandt][cold_riming_repair]")
{
    const auto cell = input(.6);
    std::array<float, 14> plain{}, observed{};
    ColdSnowRimingAdmissionObservation observation;
    cold_cell(retained_parameters(), cell, 1e-3, plain);
    cold_cell(retained_parameters(), cell, 1e-3, observed, nullptr, &observation);
    CHECK(std::memcmp(plain.data(), observed.data(), sizeof(plain)) == 0);
    REQUIRE(observation.active);
    for (unsigned cap : {0u, cold_snow_riming_admission_max_passes + 1})
    {
        observed.fill(42);
        const auto before = observed;
        auto preserved = observation;
        try
        {
            cold_cell(retained_parameters(), cell, 1e-3, observed, nullptr, &preserved, cap);
            FAIL("invalid admission resource was accepted");
        }
        catch (const ColdSnowRimingAdmissionError& error)
        {
            CHECK(error.kind() == ColdSnowRimingAdmissionFailure::InvalidResource);
        }
        CHECK(std::memcmp(before.data(), observed.data(), sizeof(before)) == 0);
        CHECK(preserved.admission_passes == observation.admission_passes);
        CHECK(preserved.admitted_snow_mass_increment_per_kg == observation.admitted_snow_mass_increment_per_kg);
    }
}

TEST_CASE("snow riming bounded admission settles or rejects the retained donor stress corpus atomically",
          "[physics][milbrandt][cold_riming_repair]")
{
    std::size_t active = 0;
    std::size_t rejected = 0;
    std::size_t competed = 0;
    unsigned maximum_passes = 0;
    bool tested_cap = false;
    for (bool hail : {false, true})
    {
        for (double duration : {.2, 1.0, 10.0})
        {
            for (int index = 0; index < 480; ++index)
            {
                const auto cell = stress_input(index);
                std::array<float, 14> rates{};
                ColdSnowRimingAdmissionObservation observation;
                observation.admission_passes = 123;
                try
                {
                    cold_cell(retained_parameters(hail), cell, duration, rates, nullptr, &observation);
                }
                catch (const ColdSnowRimingAdmissionError& error)
                {
                    REQUIRE((error.kind() == ColdSnowRimingAdmissionFailure::BudgetCap
                        || error.kind() == ColdSnowRimingAdmissionFailure::InvalidEndpoint));
                    CHECK(observation.admission_passes == 123);
                    for (const auto value : rates)
                    {
                        CHECK(value == 0);
                    }
                    ++rejected;
                    continue;
                }
                if (!observation.active)
                {
                    continue;
                }
                ++active;
                maximum_passes = std::max(maximum_passes, observation.admission_passes);
                for (std::size_t donor = 0; donor < 7; ++donor)
                {
                    CHECK(observation.donor_mass_sink_per_kg[donor] <= observation.donor_mass_source_per_kg[donor]);
                }
                CHECK(observation.snow_number_sink_per_kg <= observation.snow_number_source_per_kg);
                if (observation.admitted_cloud_mass_per_kg[0] < observation.pre_admission_cloud_mass_per_kg[0])
                {
                    ++competed;
                    const auto expected = milbrandt_double_riming_reference::snow_diffusion(
                        cell, duration, observation.admitted_cloud_mass_per_kg[0]);
                    CHECK(std::abs(observation.riming_snow_mass_increment_per_kg - expected.heating)
                        <= 2e-8L * std::abs(expected.heating));
                }
                if (!tested_cap && observation.admission_passes > 1)
                {
                    std::array<float, 14> preserved{};
                    auto snapshot = observation;
                    try
                    {
                        cold_cell(retained_parameters(hail), cell, duration, preserved, nullptr, &snapshot,
                            observation.admission_passes - 1);
                        FAIL("insufficient admission budget succeeded");
                    }
                    catch (const ColdSnowRimingAdmissionError& error)
                    {
                        REQUIRE(error.kind() == ColdSnowRimingAdmissionFailure::BudgetCap);
                    }
                    for (const auto value : preserved)
                    {
                        CHECK(value == 0);
                    }
                    CHECK(snapshot.admission_passes == observation.admission_passes);
                    cold_cell(retained_parameters(hail), cell, duration, preserved, nullptr, &snapshot);
                    CHECK(std::memcmp(preserved.data(), rates.data(), sizeof(rates)) == 0);
                    tested_cap = true;
                }
            }
        }
    }
    std::cout << "[snow-riming-stress] active=" << active << " rejected=" << rejected
              << " competed=" << competed << " max_passes=" << maximum_passes << '\n';
    REQUIRE(active > 0);
    REQUIRE(rejected > 0);
    REQUIRE(competed > 0);
    CHECK(maximum_passes >= 1);
}

TEST_CASE("conservative snow admission closes a recorded roundoff-limited source in one pass",
          "[physics][milbrandt][cold_riming_repair]")
{
    const std::array<double, 16> cell{210.296142578125, 77948.65625, 0.6426196694374084,
        1.3215391845733393e-05, 0.0007361726020462811, 5.194074148562322e-08,
        0.00511712534353137, 0.006878420244902372, 0.002300739288330078, 0.005490576848387718,
        32138376, 0.0042386227287352085, 2234.460205078125, 1051.1832275390625,
        86.68053436279297, 67.22148132324219};
    const std::array<float, 14> incoming{1.0000000116860974e-07f, 0, 0, 0, 0, 0,
        -1.0000000116860974e-07f, 0.0010000000474974513f, 652028.75f, 0, 0, 0, 0, 0};
    constexpr double duration = 1.5336747745287318;
    auto rates = incoming;
    ColdSnowRimingAdmissionObservation observation;
    REQUIRE_NOTHROW(cold_cell(retained_parameters(), cell, duration, rates, nullptr, &observation));
    REQUIRE(observation.active);
    REQUIRE(observation.admission_passes == 1);
    for (std::size_t donor = 0; donor < 7; ++donor)
    {
        CHECK(observation.donor_mass_sink_per_kg[donor] <= observation.donor_mass_source_per_kg[donor]);
    }
    CHECK(observation.snow_number_sink_per_kg <= observation.snow_number_source_per_kg);
    const auto expected = rates;
    rates = incoming;
    auto preserved = observation;
    REQUIRE_NOTHROW(cold_cell(retained_parameters(), cell, duration, rates, nullptr, &preserved, 1));
    CHECK(std::memcmp(rates.data(), expected.data(), sizeof(rates)) == 0);
    CHECK(preserved.admission_passes == observation.admission_passes);
}

TEST_CASE("snow cloud admission rounds paired transfers conservatively before freezing its budget",
          "[physics][milbrandt][cold_riming_repair]")
{
    const auto cell = stress_input(96);
    std::array<float, 14> rates{};
    rates[0] = 1e-7f;
    rates[6] = -1e-7f;
    rates[7] = .001f;
    rates[8] = 5e6f;
    ColdSnowRimingAdmissionObservation observation;
    REQUIRE_NOTHROW(cold_cell(retained_parameters(false), cell, .2, rates, nullptr, &observation));
    REQUIRE(observation.active);
    REQUIRE(observation.cloud_factor_quantized);
    CHECK(observation.cloud_factor_backoffs <= 1);
    CHECK(observation.cloud_admission_factor > 0);
    CHECK(observation.cloud_admission_factor < 1);
    const double scaled = std::ldexp(observation.cloud_admission_factor, 40);
    CHECK(scaled == std::floor(scaled));
    CHECK(observation.donor_mass_sink_per_kg[1] <= cell[4]);
    for (std::size_t species = 0; species < 3; ++species)
    {
        CHECK(observation.admitted_cloud_mass_per_kg[species]
            == observation.cloud_admission_factor * observation.pre_admission_cloud_mass_per_kg[species]);
        CHECK(observation.admitted_cloud_number_per_kg[species]
            == observation.cloud_admission_factor * observation.pre_admission_cloud_number_per_kg[species]);
    }
}

TEST_CASE("snow riming closes a near-exhausted rain donor with a viable snow population",
          "[physics][milbrandt][cold_riming_repair]")
{
    const auto cell = stress_input(133);
    std::array<float, 14> rates{};
    rates[0] = 1e-7f;
    rates[6] = -1e-7f;
    rates[7] = .001f;
    ColdSnowRimingAdmissionObservation observation;
    REQUIRE_NOTHROW(cold_cell(retained_parameters(false), cell, .2, rates, nullptr, &observation));
    REQUIRE(observation.active);
    for (std::size_t donor = 0; donor < 7; ++donor)
    {
        CHECK(observation.donor_mass_sink_per_kg[donor] <= observation.donor_mass_source_per_kg[donor]);
    }
    CHECK(observation.donor_mass_sink_per_kg[2] > .99 * cell[5]);
    CHECK(observation.final_snow_mass_per_kg > 0);
    CHECK(observation.final_snow_number_per_kg > .9 * cell[13]);
    CHECK(cell[13] + .2 * rates[11] > .9 * cell[13]);
}

TEST_CASE("snow donor feedback needs bounded repeated admission and rejects an insufficient work cap atomically",
          "[physics][milbrandt][cold_riming_repair]")
{
    const std::array<double, 16> previous_cell{248.3369140625, 97820.109375, 0.5358629822731018,
        8.888878801371902e-5, 0.044002190232276917, 3.182168711646227e-6,
        1.1137425204310603e-8, 0.00019661484111566097, 3.160278652103443e-7,
        1.9988622312894222e-8, 98910494720, 68.88011169433594,
        1.3401981050265022e-5, 17.1782169342041, 32.73395919799805, 16.07222557067871};
    const std::array<float, 14> previous_incoming{1e-7f, 0, 0, 0, 0, 0, -1e-7f, .001f, 127588.5703125f, 0, 0, 0, 0, 0};
    auto previous_rates = previous_incoming;
    ColdSnowRimingAdmissionObservation previous;
    REQUIRE_NOTHROW(cold_cell(retained_parameters(false), previous_cell, 7.837692569478082,
        previous_rates, nullptr, &previous));
    REQUIRE(previous.active);
    REQUIRE(previous.admission_passes == 1);
    for (std::size_t donor = 0; donor < 7; ++donor)
    {
        CHECK(previous.donor_mass_sink_per_kg[donor] <= previous.donor_mass_source_per_kg[donor]);
    }
    CHECK(previous.snow_number_sink_per_kg <= previous.snow_number_source_per_kg);

    const std::array<double, 16> cell{256.3509521484375, 92997.296875, 1.0825551748275757,
        0.00032474694307893515, 6.7725557073572418e-7, 1.9364311043545968e-8,
        3.595914677134715e-8, 0.023380322381854057, 0.00021004240261390805,
        0.0018277277704328299, 50090.95703125, 5.1465916633605957,
        4.5703465730184689e-5, 65093.734375, 378.01608276367188, 2746596864};
    constexpr double duration = 0.040744432778729599;
    const std::array<float, 14> incoming{1e-7f, 0, 0, 0, 0, 0, -1e-7f, .001f, 24543230.0f, 0, 0, 0, 0, 0};
    auto rates = incoming;
    ColdSnowRimingAdmissionObservation observation;
    REQUIRE_NOTHROW(cold_cell(retained_parameters(true), cell, duration, rates, nullptr, &observation));
    REQUIRE(observation.active);
    REQUIRE(observation.admission_passes > 1);
    REQUIRE(observation.admission_passes <= cold_snow_riming_admission_max_passes);
    for (std::size_t donor = 0; donor < 7; ++donor)
    {
        CHECK(observation.donor_mass_sink_per_kg[donor] <= observation.donor_mass_source_per_kg[donor]);
    }
    CHECK(observation.snow_number_sink_per_kg <= observation.snow_number_source_per_kg);
    const auto accepted = rates;
    rates = incoming;
    auto preserved = observation;
    try
    {
        cold_cell(retained_parameters(true), cell, duration, rates, nullptr, &preserved,
            observation.admission_passes - 1);
        FAIL("insufficient donor admission work cap succeeded");
    }
    catch (const ColdSnowRimingAdmissionError& error)
    {
        CHECK(error.kind() == ColdSnowRimingAdmissionFailure::BudgetCap);
    }
    CHECK(std::memcmp(rates.data(), incoming.data(), sizeof(rates)) == 0);
    CHECK(preserved.admission_passes == observation.admission_passes);
    CHECK(preserved.final_snow_number_per_kg == observation.final_snow_number_per_kg);
    REQUIRE_NOTHROW(cold_cell(retained_parameters(true), cell, duration, rates, nullptr, &preserved));
    CHECK(std::memcmp(rates.data(), accepted.data(), sizeof(rates)) == 0);
}

TEST_CASE("snow riming rejects a recorded unresolved donor cycle at the full work cap",
          "[physics][milbrandt][cold_riming_repair]")
{
    const std::array<double, 16> previous_cell{261.26336669921875, 99628.359375, 0.9517037868499756,
        0.013939335942268372, 0.021254945546388626, 0.0019817641004920006,
        1.499657820147604e-8, 2.292152203153819e-6, 4.6748696149734315e-6,
        8.482126577291638e-5, 42946629632, 99081808, 0.001704998197965324,
        15.420398712158203, 2021.1268310546875, 0.11611058562994003};
    const std::array<float, 14> previous_incoming{1e-7f, 0, 0, 0, 0, 0, -1e-7f, .001f, 268196.6875f, 0, 0, 0, 0, 0};
    auto previous_rates = previous_incoming;
    ColdSnowRimingAdmissionObservation previous;
    REQUIRE_NOTHROW(cold_cell(retained_parameters(false), previous_cell, 3.728606661905042,
        previous_rates, nullptr, &previous));
    REQUIRE(previous.active);
    REQUIRE(previous.admission_passes <= cold_snow_riming_admission_max_passes);
    for (std::size_t donor = 0; donor < 7; ++donor)
    {
        CHECK(previous.donor_mass_sink_per_kg[donor] <= previous.donor_mass_source_per_kg[donor]);
    }
    CHECK(previous.snow_number_sink_per_kg <= previous.snow_number_source_per_kg);

    const std::array<double, 16> cell{265.80990600585938, 88899.6875, 0.94878780841827393,
        0.022537695243954659, 3.4392439118846596e-8, 1.3518631931219716e-5,
        0.014146114699542522, 0.015282873064279556, 2.7798416340374388e-5,
        0.010872337035834789, 10540.935546875, 24819536, 4956.21484375,
        676.070068359375, 264328.34375, 52801356};
    constexpr double duration = 5.6542151130055904;
    const std::array<float, 14> incoming{1e-7f, 0, 0, 0, 0, 0, -1e-7f, .001f, 176859.203125f, 0, 0, 0, 0, 0};
    auto rates = incoming;
    ColdSnowRimingAdmissionObservation observation;
    observation.admission_passes = 123;
    std::array<unsigned char, sizeof(observation)> observation_before;
    std::memcpy(observation_before.data(), &observation, sizeof(observation));
    try
    {
        cold_cell(retained_parameters(true), cell, duration, rates, nullptr, &observation);
        FAIL("unresolved donor cycle was accepted");
    }
    catch (const ColdSnowRimingAdmissionError& error)
    {
        CHECK(error.kind() == ColdSnowRimingAdmissionFailure::BudgetCap);
    }
    CHECK(std::memcmp(rates.data(), incoming.data(), sizeof(rates)) == 0);
    CHECK(observation.admission_passes == 123);
    CHECK(std::memcmp(&observation, observation_before.data(), sizeof(observation)) == 0);
    rates.fill(0);
    REQUIRE_NOTHROW(cold_cell(retained_parameters(false), input(.6), 1e-3, rates, nullptr, &observation));
    REQUIRE(observation.active);
}
