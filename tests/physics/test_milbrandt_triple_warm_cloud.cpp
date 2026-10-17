#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/triple_warm_cloud.hpp"
#include "support/milbrandt_cloud_adjustment_reference.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace
{

using namespace tmv;
using namespace tmv::milbrandt_detail;
constexpr long double rain_mass_coefficient = 523.598775598L;

RainTripleMomentState rain_state(const DoubleWarmRainCell& source)
{
    const float q = static_cast<float>(source.rain);
    const float n = static_cast<float>(source.rain_number);
    return {q, n, q == 0 ? 0.0f : static_cast<float>(20.0L * q * q / (rain_mass_coefficient * rain_mass_coefficient * n))};
}

long double reference_z(const RainTripleMomentState& initial,
                        const WarmRainAcceptedEventBuffer& events,
                        const WarmRainEvaporationBudget& evaporation,
                        double duration, double density)
{
    long double q = initial.mass_per_kg;
    long double n = initial.number_per_kg;
    long double z = initial.reflectivity_per_kg;
    for (std::size_t index = 0; index < events.size; ++index)
    {
        const auto& event = events.events[index];
        if (event.kind == WarmRainAcceptedEventKind::PreRainNumberRegularization)
        {
            const long double next_n = n + static_cast<long double>(duration) * event.rain_number_rate_per_m3_s / density;
            z *= n / next_n;
            n = next_n;
        }
    }
    const long double pre_q = q;
    const long double pre_n = n;
    const long double pre_z = z;
    long double q_rate = 0;
    long double n_rate = 0;
    long double z_rate = 0;
    const long double seed_factor = std::tgamma(9.0L) * std::tgamma(3.0L) / std::pow(std::tgamma(6.0L), 2);
    for (std::size_t index = 0; index < events.size; ++index)
    {
        const auto& event = events.events[index];
        const long double rate_n = static_cast<long double>(event.rain_number_rate_per_m3_s) / density;
        if (event.kind == WarmRainAcceptedEventKind::Autoconversion)
        {
            z_rate += seed_factor / (rain_mass_coefficient * rain_mass_coefficient)
                * event.rain_mass_rate_per_kg_s * event.rain_mass_rate_per_kg_s / rate_n;
        }
        else if (event.kind == WarmRainAcceptedEventKind::Accretion || event.kind == WarmRainAcceptedEventKind::RainSelfCollection)
        {
            z_rate += pre_z * (2 * event.rain_mass_rate_per_kg_s / pre_q - rate_n / pre_n);
        }
        else
        {
            continue;
        }
        q_rate += event.rain_mass_rate_per_kg_s;
        n_rate += rate_n;
    }
    q += duration * q_rate;
    n += duration * n_rate;
    z += duration * z_rate;
    for (std::size_t index = 0; index < events.size; ++index)
    {
        const auto& event = events.events[index];
        if (event.kind == WarmRainAcceptedEventKind::PostRainNumberRegularization)
        {
            const long double next_n = n + static_cast<long double>(duration) * event.rain_number_rate_per_m3_s / density;
            z *= n / next_n;
            n = next_n;
        }
    }
    if (evaporation.present)
    {
        z *= static_cast<long double>(evaporation.mass_after_per_kg) / evaporation.mass_before_per_kg;
    }
    return z;
}

void compare_original(const DoubleWarmRainCell& source, double velocity, double duration,
                      const RainTripleMomentState& initial, const ExactTripleWarmCloudCellResult& result)
{
    const TripleCloudAdjustmentCell original{source.temperature, source.pressure, source.density,
        source.vapor, source.cloud, source.cloud_number, velocity};
    const auto independent = cloud_adjustment_reference::evaluate(original);
    const auto& cloud = result.cloud.endpoints();
    CHECK(cloud.vapor_per_kg == Approx(static_cast<double>(independent.vapor)).epsilon(3e-12).margin(3e-15));
    CHECK(cloud.cloud_per_kg == Approx(static_cast<double>(independent.cloud)).epsilon(3e-12).margin(3e-15));
    CHECK(cloud.cloud_number_per_kg == Approx(static_cast<double>(independent.number)).epsilon(3e-12).margin(1e-12));
    CHECK(cloud.temperature_increment_k == Approx(static_cast<double>(independent.heat)).epsilon(3e-12).margin(1e-11));
    const auto& input = result.warm.producer.source_cell();
    CHECK(input.temperature == cloud.temperature_after_k);
    CHECK(input.pressure == source.pressure);
    CHECK(input.density == source.density);
    CHECK(input.vapor == cloud.vapor_per_kg);
    CHECK(input.cloud == cloud.cloud_per_kg);
    CHECK(input.cloud_number == cloud.cloud_number_per_kg);
    CHECK(input.rain == source.rain);
    CHECK(input.rain_number == source.rain_number);
    CHECK(result.temperature_increment_k == cloud.temperature_increment_k + result.warm.endpoints.temperature_increment_k);
    CHECK(result.endpoints.vapor_per_kg == result.warm.endpoints.vapor_per_kg);
    CHECK(result.endpoints.cloud_per_kg == result.warm.endpoints.cloud_per_kg);
    CHECK(result.endpoints.rain_per_kg == result.warm.endpoints.rain_per_kg);
    CHECK(result.rain.reflectivity_per_kg == Approx(static_cast<double>(reference_z(initial,
        result.warm.producer.events(), result.warm.endpoints.evaporation, duration, source.density))).epsilon(3e-12).margin(1e-30));
    CHECK(result.endpoints.vapor_per_kg + result.endpoints.cloud_per_kg + result.endpoints.rain_per_kg
        == Approx(source.vapor + source.cloud + source.rain).epsilon(2e-14));
    const double evaporated_rain = result.warm.endpoints.evaporation.admitted_mass_per_kg;
    CHECK(result.temperature_increment_k == Approx(microphysics_constants::L_v / microphysics_constants::cp
        * (cloud.admitted_cloud_increment_per_kg - evaporated_rain)).epsilon(3e-13).margin(1e-12));
    CHECK(result.rounded_rain.mass_per_kg == static_cast<float>(result.rain.mass_per_kg));
    CHECK(result.rounded_rain.number_per_kg == static_cast<float>(result.rain.number_per_kg));
    CHECK(result.rounded_rain.reflectivity_per_kg == static_cast<float>(result.rain.reflectivity_per_kg));
}

} // namespace

TEST_CASE("composed cloud and warm rain bind the unrounded admitted state and independent moments",
          "[physics][milbrandt][triple_warm_cloud]")
{
    const auto parameters = warm_rain_parameters(0, 523.598775598, 3, 149.1, .5);
    bool saw_unrounded = false;
    bool saw_new_rain = false;
    bool saw_collection = false;
    for (double temperature : {280.125, 290.125})
    {
        for (double density : {static_cast<double>(.65f), static_cast<double>(1.43f)})
        {
            const double qs = thermodynamics::saturation_mixing_ratio_water(temperature, 90000);
            const std::array<DoubleWarmRainCell, 4> sources{{
                {temperature, 90000, density, 1.2 * qs, .03f, 0, 1e5f, 0},
                {temperature, 90000, density, .8 * qs, .01f, 0, 1e5f, 0},
                {temperature, 90000, density, 1.05 * qs, .001f, 1e-4f, 1e5f, 1e3f},
                {temperature, 90000, density, .8 * qs, .01f, .002f, 1e5f, 1e5f}}};
            for (const auto& source : sources)
            {
                for (double duration : {.05, .1, .3})
                {
                    const auto initial = rain_state(source);
                    ExactTripleWarmCloudCellResult result;
                    std::string error;
                    const bool accepted = produce_exact_triple_warm_cloud_cell(parameters, source, .5, duration, initial, result, error);
                    INFO("T " << temperature << ", rho " << density << ", duration " << duration
                        << ", cloud " << source.cloud << ", rain " << source.rain << ": " << error);
                    if (!accepted)
                    {
                        TripleCloudAdjustmentResult cloud;
                        std::string diagnostic;
                        REQUIRE(produce_exact_triple_cloud_adjustment(
                            {source.temperature, source.pressure, source.density, source.vapor, source.cloud,
                             source.cloud_number, .5}, duration, cloud, diagnostic));
                        const auto& e = cloud.endpoints();
                        TripleWarmRainDoubleSourceResult warm;
                        triple_warm_rain_cell_exact_double_source(parameters,
                            {e.temperature_after_k, source.pressure, source.density, e.vapor_per_kg, e.cloud_per_kg,
                             source.rain, e.cloud_number_per_kg, source.rain_number}, duration, {}, warm);
                        std::string kinds;
                        for (std::size_t index = 0; index < warm.events().size; ++index)
                        {
                            const auto& event = warm.events().events[index];
                            kinds += std::to_string(static_cast<int>(event.kind)) + ":"
                                + std::to_string(event.cloud_number_rate_per_m3_s) + " ";
                        }
                        INFO("observed event kind/Nc rate: " << kinds);
                        INFO("warm endpoint qc " << warm.endpoints().cloud_per_kg << ", Nc " << warm.endpoints().cloud_number_per_kg);
                        REQUIRE(accepted);
                    }
                    REQUIRE(accepted);
                    compare_original(source, .5, duration, initial, result);
                    if (result.warm.endpoints.cloud_donor_exhausted)
                    {
                        CHECK(result.warm.endpoints.cloud_donor_budget_per_kg == result.cloud.endpoints().cloud_per_kg);
                        CHECK(result.endpoints.cloud_per_kg == 0);
                        for (std::size_t index = 0; index < result.warm.producer.events().size; ++index)
                        {
                            CHECK(result.warm.producer.events().events[index].kind != WarmRainAcceptedEventKind::CloudMassClip);
                        }
                    }
                    const auto& input = result.warm.producer.source_cell();
                    saw_unrounded |= input.cloud != static_cast<double>(static_cast<float>(input.cloud))
                        || input.vapor != static_cast<double>(static_cast<float>(input.vapor))
                        || input.cloud_number != static_cast<double>(static_cast<float>(input.cloud_number))
                        || input.temperature != static_cast<double>(static_cast<float>(input.temperature));
                    saw_new_rain |= source.rain == 0 && result.rain.mass_per_kg > 0;
                    for (std::size_t index = 0; index < result.warm.producer.events().size; ++index)
                    {
                        const auto kind = result.warm.producer.events().events[index].kind;
                        saw_collection |= kind == WarmRainAcceptedEventKind::Accretion || kind == WarmRainAcceptedEventKind::RainSelfCollection;
                    }
                    ExactTripleWarmCloudCellResult replay;
                    REQUIRE(apply_exact_triple_warm_cloud_cell(parameters, source, .5, duration, initial, result, replay, error));
                    CHECK(replay.rain.reflectivity_per_kg == result.rain.reflectivity_per_kg);
                }
            }
        }
    }
    CHECK(saw_unrounded);
    CHECK(saw_new_rain);
    CHECK(saw_collection);
}

TEST_CASE("composed warm source failure and every altered phase leave the output unpublished",
          "[physics][milbrandt][triple_warm_cloud]")
{
    const auto parameters = warm_rain_parameters(0, 523.598775598, 3, 149.1, .5);
    const DoubleWarmRainCell source{290.125, 90000, .65f,
        1.05 * thermodynamics::saturation_mixing_ratio_water(290.125, 90000), .001f, 1e-4f, 1e5f, 1e3f};
    const auto initial = rain_state(source);
    ExactTripleWarmCloudCellResult baseline;
    std::string error;
    REQUIRE(produce_exact_triple_warm_cloud_cell(parameters, source, .5, .1, initial, baseline, error));
    for (int mutation = 0; mutation < 13; ++mutation)
    {
        auto altered = baseline;
        switch (mutation)
        {
            case 0: altered.endpoints.vapor_per_kg = std::nextafter(altered.endpoints.vapor_per_kg, 1.0); break;
            case 1: altered.endpoints.cloud_per_kg = std::nextafter(altered.endpoints.cloud_per_kg, 1.0); break;
            case 2: altered.endpoints.rain_per_kg = std::nextafter(altered.endpoints.rain_per_kg, 1.0); break;
            case 3: altered.endpoints.cloud_number_per_kg = std::nextafter(altered.endpoints.cloud_number_per_kg, 1e20); break;
            case 4: altered.rain.reflectivity_per_kg = std::nextafter(altered.rain.reflectivity_per_kg, 1.0); break;
            case 5: altered.temperature_increment_k = std::nextafter(altered.temperature_increment_k, 1.0); break;
            case 6: altered.rounded_rain.number_per_kg = std::nextafter(altered.rounded_rain.number_per_kg, 1e20f); break;
            case 7: const_cast<TripleCloudAdjustmentEndpoints&>(altered.cloud.endpoints()).temperature_after_k += .01; break;
            case 8: const_cast<DoubleWarmRainCell&>(altered.warm.producer.source_cell()).cloud += 1e-8; break;
            case 9: const_cast<WarmRainDoubleEndpoints&>(altered.warm.producer.endpoints()).cloud_per_kg += 1e-8; break;
            case 10: altered.warm.rates.vapor = std::nextafter(altered.warm.rates.vapor, 1.0f); break;
            case 11: altered.warm.rain.number_per_kg = std::nextafter(altered.warm.rain.number_per_kg, 1e20); break;
            case 12: const_cast<WarmRainDoubleEndpoints&>(altered.warm.producer.endpoints()).cloud_donor_exhausted = true; break;
        }
        auto result = baseline;
        INFO("composed result mutation " << mutation);
        CHECK_FALSE(apply_exact_triple_warm_cloud_cell(parameters, source, .5, .1, initial, altered, result, error));
        CHECK(result.rain.reflectivity_per_kg == baseline.rain.reflectivity_per_kg);
        CHECK(result.temperature_increment_k == baseline.temperature_increment_k);
        CHECK(result.endpoints.cloud_number_per_kg == baseline.endpoints.cloud_number_per_kg);
    }
    for (int mutation = 0; mutation < 8; ++mutation)
    {
        auto cell = source;
        const std::array<double*, 8> fields{&cell.temperature,&cell.pressure,&cell.density,&cell.vapor,
            &cell.cloud,&cell.rain,&cell.cloud_number,&cell.rain_number};
        *fields[mutation] = std::nextafter(*fields[mutation], 1e20);
        auto result = baseline;
        CHECK_FALSE(apply_exact_triple_warm_cloud_cell(parameters, cell, .5, .1, initial, baseline, result, error));
        CHECK(result.rain.reflectivity_per_kg == baseline.rain.reflectivity_per_kg);
    }
    auto result = baseline;
    auto z_changed = initial;
    z_changed.reflectivity_per_kg = std::nextafter(z_changed.reflectivity_per_kg, 1.0f);
    CHECK_FALSE(apply_exact_triple_warm_cloud_cell(parameters, source, .5, .1, z_changed, baseline, result, error));
    CHECK_FALSE(apply_exact_triple_warm_cloud_cell(parameters, source, 1.0, .1, initial, baseline, result, error));
    CHECK_FALSE(apply_exact_triple_warm_cloud_cell(parameters, source, .5, .3, initial, baseline, result, error));
    CHECK_FALSE(produce_exact_triple_warm_cloud_cell(parameters, source, .5, .1, initial, result, error, 0));
    CHECK(result.endpoints.cloud_per_kg == baseline.endpoints.cloud_per_kg);
}

TEST_CASE("new Double warm source retains the old Float path exactly when supplied the same values",
          "[physics][milbrandt][triple_warm_cloud]")
{
    const auto parameters = warm_rain_parameters(0, 523.598775598, 3, 149.1, .5);
    const WarmRainCell old{280,90000,.65f,.003f,.001f,1e-4f,1e5f,1e3f};
    const DoubleWarmRainCell source{old.temperature,old.pressure,old.density,old.vapor,old.cloud,old.rain,old.cloud_number,old.rain_number};
    const auto initial = rain_state(source);
    for (double duration : {.05,.1,.3})
    {
        ExactTripleWarmCellResult previous;
        ExactTripleWarmDoubleSourceCellResult current;
        std::string error;
        REQUIRE(evaluate_exact_triple_warm_cell(parameters, old, duration, initial, previous, error));
        REQUIRE(evaluate_exact_triple_warm_double_source_cell(parameters, source, duration, initial, current, error));
        CHECK(std::memcmp(&previous.rates, &current.rates, sizeof(WarmRainTendencies)) == 0);
        CHECK(previous.endpoints.vapor_per_kg == current.endpoints.vapor_per_kg);
        CHECK(previous.endpoints.cloud_per_kg == current.endpoints.cloud_per_kg);
        CHECK(previous.endpoints.cloud_number_per_kg == current.endpoints.cloud_number_per_kg);
        CHECK(previous.rain.mass_per_kg == current.rain.mass_per_kg);
        CHECK(previous.rain.number_per_kg == current.rain.number_per_kg);
        CHECK(previous.rain.reflectivity_per_kg == current.rain.reflectivity_per_kg);
    }
}

TEST_CASE("Double warm small-drop reclassification retains water and cloud number",
          "[physics][milbrandt][triple_warm_cloud][rain_reclassification]")
{
    const auto parameters = warm_rain_parameters(0, 523.598775598, 3, 149.1, .5);
    const float q = 1e-5f;
    const float n = static_cast<float>(q / (523.598775598 * std::pow(40e-6, 3)));
    const double saturation = thermodynamics::saturation_mixing_ratio_water(290.125, 90000);
    const DoubleWarmRainCell source{290.125, 90000, .65f, saturation, 0, q, 0, n};
    const auto initial = rain_state(source);
    ExactTripleWarmDoubleSourceCellResult result;
    std::string error;
    const bool accepted = evaluate_exact_triple_warm_double_source_cell(
        parameters, source, .1, initial, result, error);
    INFO(error);
    REQUIRE(accepted);
    CHECK(result.endpoints.cloud_per_kg == source.rain);
    const long double nr = static_cast<long double>(source.density) * source.rain_number;
    const long double inverse_lambda = std::cbrt(source.rain / source.rain_number /
        (rain_mass_coefficient * 6.0L));
    const long double self_collection = std::min(2.59e15L * nr * nr * 720.0L *
        std::pow(inverse_lambda, 6), .5L * nr / .1L);
    CHECK(result.endpoints.cloud_number_per_kg == Approx(static_cast<double>(
        source.rain_number - .1L * self_collection / source.density)).epsilon(2e-14));
    CHECK(result.rain.mass_per_kg == 0);
    CHECK(result.rain.number_per_kg == 0);
    CHECK(result.rain.reflectivity_per_kg == 0);
    CHECK(result.endpoints.vapor_per_kg == source.vapor);
    CHECK(result.endpoints.temperature_increment_k == 0);
}

TEST_CASE("owned small-drop transfer binds preclassification budgets and rejects altered captures",
          "[physics][milbrandt][triple_warm_cloud][rain_reclassification]")
{
    const auto parameters = warm_rain_parameters(0, 523.598775598, 3, 149.1, .5);
    for (long double alpha : {0.0L, 2.0L, 20.0L})
    {
        for (double diameter : {40e-6, 80e-6})
        {
            for (double density : {static_cast<double>(.65f), static_cast<double>(1.43f)})
            {
                for (double duration : {.05, .1, .3})
                {
                    for (bool cloud : {false, true})
                    {
                        const float q = 1e-5f;
                        const float n = static_cast<float>(q / (523.598775598 * std::pow(diameter, 3)));
                        const long double g = (alpha + 4) * (alpha + 5) * (alpha + 6) /
                            ((alpha + 1) * (alpha + 2) * (alpha + 3));
                        const RainTripleMomentState initial{q, n, static_cast<float>(
                            g * q * q / (rain_mass_coefficient * rain_mass_coefficient * n))};
                        const DoubleWarmRainCell source{290.125, 90000, density,
                            thermodynamics::saturation_mixing_ratio_water(290.125, 90000),
                            cloud ? static_cast<double>(1e-6f) : 0.0, q, cloud ? 1e8 : 0.0, n};
                        ExactTripleWarmDoubleSourceCellResult result;
                        std::string error;
                        const bool accepted = evaluate_exact_triple_warm_double_source_cell(
                            parameters, source, duration, initial, result, error);
                        INFO("alpha " << alpha << ", D " << diameter << ", dt " << duration << ": " << error);
                        REQUIRE(accepted);
                        const auto& budget = result.endpoints.reclassification;
                        CHECK(budget.present);
                        CHECK(budget.rain_mass_before_per_kg == source.rain);
                        CHECK(budget.cloud_mass_before_per_kg == source.cloud);
                        CHECK(budget.cloud_number_before_per_m3 == source.density * source.cloud_number);
                        CHECK(result.endpoints.cloud_per_kg == source.cloud + source.rain);
                        CHECK(result.endpoints.cloud_number_per_kg ==
                            (budget.cloud_number_before_per_m3 + budget.rain_number_before_per_m3) / source.density);
                        CHECK(result.rain.mass_per_kg == 0);
                        CHECK(result.rain.number_per_kg == 0);
                        CHECK(result.rain.reflectivity_per_kg == 0);
                        CHECK(result.endpoints.vapor_per_kg == source.vapor);
                        CHECK(result.endpoints.temperature_increment_k == 0);
                        long double number_loss = 0;
                        int transfer_count = 0;
                        for (std::size_t index = 0; index < result.producer.events().size; ++index)
                        {
                            const auto& event = result.producer.events().events[index];
                            if (event.kind == WarmRainAcceptedEventKind::RainSelfCollection)
                            {
                                number_loss += static_cast<long double>(duration) * event.rain_number_rate_per_m3_s;
                            }
                            if (event.kind == WarmRainAcceptedEventKind::RainToCloud)
                            {
                                ++transfer_count;
                                CHECK(event.cloud_mass_rate_per_kg_s == -event.rain_mass_rate_per_kg_s);
                                CHECK(event.cloud_number_rate_per_m3_s == -event.rain_number_rate_per_m3_s);
                                CHECK(event.temperature_rate_k_s == 0);
                            }
                            CHECK(event.kind != WarmRainAcceptedEventKind::RainReset);
                            CHECK(event.kind != WarmRainAcceptedEventKind::Evaporation);
                        }
                        CHECK(transfer_count == 1);
                        CHECK(budget.rain_number_before_per_m3 == Approx(static_cast<double>(
                            static_cast<long double>(source.density) * source.rain_number + number_loss)).epsilon(2e-14));
                        ExactTripleWarmDoubleSourceCellResult replay;
                        REQUIRE(apply_exact_triple_warm_double_source_cell(
                            parameters, source, duration, initial, result.producer, replay, error));
                        CHECK(replay.endpoints.cloud_per_kg == result.endpoints.cloud_per_kg);
                        for (int mutation = 0; mutation < 12; ++mutation)
                        {
                            auto producer = result.producer;
                            auto& e = const_cast<WarmRainDoubleEndpoints&>(producer.endpoints());
                            auto& b = e.reclassification;
                            switch (mutation)
                            {
                                case 0: b.present = false; break;
                                case 1: b.rain_mass_before_per_kg *= 2; break;
                                case 2: b.rain_number_before_per_m3 *= 2; break;
                                case 3: b.cloud_mass_before_per_kg += 1e-5; break;
                                case 4: b.cloud_number_before_per_m3 += 1e4; break;
                                case 5: b.rain_mass_before_per_kg = std::numeric_limits<double>::quiet_NaN(); break;
                                case 6: b.rain_number_before_per_m3 = std::numeric_limits<double>::infinity(); break;
                                case 7: e.rain_per_kg = source.rain; break;
                                case 8: e.rain_number_per_kg = source.rain_number; break;
                                case 9: e.cloud_per_kg = source.cloud; break;
                                case 10: e.cloud_number_per_kg = source.cloud_number; break;
                                case 11: e.evaporation.present = true; break;
                            }
                            auto output = result;
                            CHECK_FALSE(apply_exact_triple_warm_double_source_cell(
                                parameters, source, duration, initial, producer, output, error));
                            CHECK(output.endpoints.cloud_per_kg == result.endpoints.cloud_per_kg);
                            CHECK(output.endpoints.cloud_number_per_kg == result.endpoints.cloud_number_per_kg);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("composed small-drop capture binds diagnosed cloud transfer and preserves reset guards",
          "[physics][milbrandt][triple_warm_cloud][rain_reclassification]")
{
    const auto parameters = warm_rain_parameters(0, 523.598775598, 3, 149.1, .5);
    const float q = 1e-5f;
    const float n = static_cast<float>(q / (523.598775598 * std::pow(40e-6, 3)));
    const DoubleWarmRainCell source{290.125, 90000, .65f,
        thermodynamics::saturation_mixing_ratio_water(290.125, 90000), 0, q, 0, n};
    const auto initial = rain_state(source);
    ExactTripleWarmCloudCellResult result;
    std::string error;
    REQUIRE(produce_exact_triple_warm_cloud_cell(parameters, source, 0, .1, initial, result, error));
    CHECK(result.endpoints.reclassification.present);
    CHECK(result.endpoints.cloud_per_kg == source.rain);
    CHECK(result.temperature_increment_k == 0);
    for (int mutation = 0; mutation < 5; ++mutation)
    {
        auto altered = result;
        auto& b = altered.endpoints.reclassification;
        switch (mutation)
        {
            case 0: b.present = false; break;
            case 1: b.rain_mass_before_per_kg = std::nextafter(b.rain_mass_before_per_kg, 1.0); break;
            case 2: b.rain_number_before_per_m3 = std::nextafter(b.rain_number_before_per_m3, 1e20); break;
            case 3: b.cloud_mass_before_per_kg = 1e-9; break;
            case 4: b.cloud_number_before_per_m3 = 1; break;
        }
        auto output = result;
        CHECK_FALSE(apply_exact_triple_warm_cloud_cell(parameters, source, 0, .1, initial, altered, output, error));
        CHECK(output.endpoints.cloud_per_kg == result.endpoints.cloud_per_kg);
    }
    const WarmRainCell previous{static_cast<float>(source.temperature), static_cast<float>(source.pressure),
        static_cast<float>(source.density), static_cast<float>(source.vapor), 0, q, 0, n};
    ExactTripleWarmCellResult old_result;
    CHECK_FALSE(evaluate_exact_triple_warm_cell(parameters, previous, .1, initial, old_result, error));
    CHECK(error.find("cloud-Z") != std::string::npos);
    auto tiny = source;
    tiny.rain = 1e-15f;
    tiny.rain_number = 1.0f;
    const auto tiny_initial = rain_state(tiny);
    ExactTripleWarmDoubleSourceCellResult output;
    REQUIRE(evaluate_exact_triple_warm_double_source_cell(parameters, tiny, .1, tiny_initial, output, error));
    CHECK(output.endpoints.cloud_per_kg == tiny.rain);
    CHECK(output.endpoints.cloud_number_per_kg == tiny.rain_number);
    CHECK(output.rain.reflectivity_per_kg == 0);
    auto forged = result.warm.producer;
    auto& events = const_cast<WarmRainAcceptedEventBuffer&>(forged.events());
    for (std::size_t index = 0; index < events.size; ++index)
    {
        if (events.events[index].kind == WarmRainAcceptedEventKind::RainToCloud)
        {
            events.events[index].kind = WarmRainAcceptedEventKind::RainReset;
        }
    }
    CHECK_FALSE(apply_exact_triple_warm_double_source_cell(
        parameters, source, .1, initial, forged, output, error));
    CHECK(error.find("reclassification or reset") != std::string::npos);
}

TEST_CASE("Double warm coherent populations below process cutoffs retain their admitted category",
          "[physics][milbrandt][triple_warm_cloud][rain_reclassification]")
{
    const auto parameters = warm_rain_parameters(0, 523.598775598, 3, 149.1, .5);
    for (const auto population : std::array<std::array<float, 2>, 3>{{
        {{4e-13f, .0011f}}, {{1e-15f, 1e-7f}}, {{1e-16f, 1e-6f}}}})
    {
        const DoubleWarmRainCell source{290.125, 90000, .65f,
            thermodynamics::saturation_mixing_ratio_water(290.125, 90000),
            0, population[0], 0, population[1]};
        const auto initial = rain_state(source);
        ExactTripleWarmDoubleSourceCellResult result;
        std::string error;
        const double diameter = std::cbrt(source.rain / (parameters.c_r * source.rain_number));
        const bool transferred = diameter < 82e-6;
        INFO("q " << source.rain << ", N " << source.rain_number << ": " << error);
        REQUIRE(evaluate_exact_triple_warm_double_source_cell(parameters, source, .1, initial, result, error));
        CHECK(result.endpoints.reclassification.present == transferred);
        CHECK(result.rain.mass_per_kg == (transferred ? 0.0 : source.rain));
        CHECK(result.rain.number_per_kg == (transferred ? 0.0 : source.rain_number));
        CHECK(result.rain.reflectivity_per_kg == (transferred ? 0.0 : initial.reflectivity_per_kg));
        CHECK(result.endpoints.cloud_per_kg == (transferred ? source.rain : 0.0));
        CHECK(result.endpoints.vapor_per_kg == source.vapor);
        CHECK(result.endpoints.temperature_increment_k == 0);
        for (std::size_t index = 0; index < result.producer.events().size; ++index)
        {
            CHECK(result.producer.events().events[index].kind != WarmRainAcceptedEventKind::RainReset);
        }
    }
}

TEST_CASE("complete cloud admission and subsequent small-rain transfer keep distinct donor budgets",
          "[physics][milbrandt][triple_warm_cloud][rain_reclassification]")
{
    const auto parameters = warm_rain_parameters(0, 523.598775598, 3, 149.1, .5);
    const float q = .002f;
    const float n = static_cast<float>(q / (523.598775598 * std::pow(40e-6, 3)));
    const DoubleWarmRainCell source{285, 90000, .65f,
        thermodynamics::saturation_mixing_ratio_water(285, 90000), 1e-4f, q, 1e3f, n};
    const auto initial = rain_state(source);
    ExactTripleWarmDoubleSourceCellResult result;
    std::string error;
    const bool accepted = evaluate_exact_triple_warm_double_source_cell(
        parameters, source, 1e6, initial, result, error);
    INFO(error);
    REQUIRE(accepted);
    CHECK(result.endpoints.cloud_donor_exhausted);
    CHECK(result.endpoints.cloud_donor_budget_per_kg == source.cloud);
    CHECK(result.endpoints.reclassification.present);
    CHECK(result.endpoints.reclassification.cloud_mass_before_per_kg == 0);
    CHECK(result.endpoints.cloud_per_kg == source.cloud + source.rain);
    CHECK(result.endpoints.reclassification.rain_mass_before_per_kg == source.cloud + source.rain);
    CHECK(result.rain.mass_per_kg == 0);
    CHECK(result.rain.number_per_kg == 0);
    CHECK(result.rain.reflectivity_per_kg == 0);
    CHECK(result.endpoints.temperature_increment_k == 0);
    ExactTripleWarmDoubleSourceCellResult replay;
    REQUIRE(apply_exact_triple_warm_double_source_cell(
        parameters, source, 1e6, initial, result.producer, replay, error));
    CHECK(replay.endpoints.cloud_per_kg == result.endpoints.cloud_per_kg);
}
