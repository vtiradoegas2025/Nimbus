#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_riming.hpp"
#include "support/milbrandt_cold_riming_reference.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace
{

using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_cold_riming_reference;

ColdParameters parameters(bool hail = true)
{
    const auto& p = default_cold_parameters;
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], hail};
}

ColdRimingCell source(long double alpha = 2, double saturation = 1, double temperature = 268.15,
                     double density = .65, int selected = -1)
{
    ColdRimingCell cell;
    cell.temperature_k = temperature;
    cell.pressure_pa = 85000;
    cell.dry_density_kg_m3 = density;
    cell.initial.vapor_mass_per_kg = static_cast<double>(reference::thermal::saturation(temperature, 85000).mass) * saturation;
    cell.initial.cloud_mass_per_kg = 5e-4;
    cell.initial.cloud_number_per_kg = 2e8;
    for (std::size_t index = 0; index < 4; ++index)
    {
        if (selected >= 0 && selected != static_cast<int>(index))
        {
            continue;
        }
        cell.initial.frozen[index] = reference::collection::frozen::population(index,
            std::array<double, 4>{2e-5, 1e-4, 2e-4, 3e-4}[index],
            std::array<double, 4>{5000, 5000, 600, 80}[index], alpha);
    }
    return cell;
}

void near(double actual, long double expected, long double relative = 3e-8L)
{
    INFO("actual=" << actual << ", independent=" << static_cast<double>(expected));
    REQUIRE(std::isfinite(actual));
    CHECK(std::abs(actual - expected) <= relative * std::max(std::abs(expected), 1e-30L));
}

void check_reference(const ColdRimingCell& cell, double duration, const ColdRimingResult& result)
{
    const auto expected = reference::evaluate(cell, duration);
    REQUIRE(result.producer.event_count() == 4);
    near(result.producer.cloud_admission_factor(), expected.factor);
    near(result.producer.relaxation_number(), expected.relaxation);
    near(result.cloud_mass_increment_per_kg, expected.cloud_increment);
    near(result.cloud_number_increment_per_kg, expected.cloud_number_increment);
    near(result.vapor_mass_increment_per_kg, expected.vapor_increment);
    near(result.temperature_increment_k, expected.temperature_increment);
    long double water = result.vapor_mass_increment_per_kg + static_cast<long double>(result.cloud_mass_increment_per_kg);
    for (std::size_t index = 0; index < 4; ++index)
    {
        const auto& channel = result.producer.channels()[index];
        near(channel.raw_collection_mass_per_kg, expected.raw_mass[index]);
        near(channel.raw_collection_number_per_kg, expected.raw_number[index]);
        near(channel.collection_mass_per_kg, expected.collected[index]);
        near(channel.collection_number_per_kg, expected.cloud_number_loss[index]);
        near(channel.diffusion_mass_per_kg, expected.diffusion[index]);
        near(channel.diffusion_number_per_kg, expected.diffusion_number[index]);
        near(channel.splinter_embryo_mass_per_kg, expected.splinter_mass[index]);
        near(channel.splinter_number_per_kg, expected.splinter_number[index]);
        near(channel.splinter_reflectivity_per_kg, expected.splinter_z[index]);
        near(result.mass_increment_per_kg[index], expected.mass_increment[index]);
        near(result.number_increment_per_kg[index], expected.number_increment[index]);
        near(result.reflectivity_increment_per_kg[index], expected.z_increment[index]);
        CHECK(result.rounded[index].mass_per_kg == static_cast<float>(result.unrounded.frozen[index].mass_per_kg));
        CHECK(result.rounded[index].number_per_kg == static_cast<float>(result.unrounded.frozen[index].number_per_kg));
        CHECK(result.rounded[index].reflectivity_per_kg == static_cast<float>(result.unrounded.frozen[index].reflectivity_per_kg));
        water += result.mass_increment_per_kg[index];
    }
    CHECK(std::abs(water) < 1e-15L * (cell.initial.cloud_mass_per_kg + cell.initial.vapor_mass_per_kg));
    CHECK(result.rounded_vapor_mass_per_kg == static_cast<float>(result.unrounded.vapor_mass_per_kg));
    CHECK(result.rounded_cloud_mass_per_kg == static_cast<float>(result.unrounded.cloud_mass_per_kg));
    CHECK(result.rounded_cloud_number_per_kg == static_cast<float>(result.unrounded.cloud_number_per_kg));
}

void same_result(const ColdRimingResult& a, const ColdRimingResult& b)
{
    CHECK(reference::state({0, 85000, .65, a.unrounded}) == reference::state({0, 85000, .65, b.unrounded}));
    CHECK(a.mass_increment_per_kg == b.mass_increment_per_kg);
    CHECK(a.number_increment_per_kg == b.number_increment_per_kg);
    CHECK(a.reflectivity_increment_per_kg == b.reflectivity_increment_per_kg);
    CHECK(a.temperature_increment_k == b.temperature_increment_k);
    CHECK(a.producer.event_count() == b.producer.event_count());
}

} // namespace

TEST_CASE("coupled four frozen collectors diffusion and secondary ice match independent budgets",
          "[physics][milbrandt][triple_cold_riming]")
{
    for (long double alpha : {0.0L, 2.0L, 20.0L})
    {
        for (double density : {.4, .65, 1.6})
        {
            for (double saturation : {.98, 1.0, 1.02})
            {
                for (double temperature : {264.15, 266.15, 268.15, 269.15, 271.15})
                {
                    INFO("shape=" << static_cast<double>(alpha) << ",rho=" << density
                         << ",saturation=" << saturation << ",T=" << temperature);
                    const auto cell = source(alpha, saturation, temperature, density);
                    ColdRimingResult result;
                    std::string error;
                    const bool ok = evaluate_exact_triple_cold_riming_cell(parameters(), cell, 1e-4, result, error);
                    INFO(error);
                    REQUIRE(ok);
                    check_reference(cell, 1e-4, result);
                    ColdRimingResult replay;
                    REQUIRE(apply_exact_triple_cold_riming_cell(parameters(), cell, 1e-4, result.producer, replay, error));
                    same_result(result, replay);
                }
            }
        }
    }
}

TEST_CASE("coupled secondary ice initializes ice and retains published self and donor reflectivity fluxes",
          "[physics][milbrandt][triple_cold_riming]")
{
    for (bool hail : {false, true})
    {
        for (int selected = 0; selected < 4; ++selected)
        {
            auto cell = source(2, 1, 268.15, .65, selected);
            ColdRimingResult result;
            std::string error;
            REQUIRE(evaluate_exact_triple_cold_riming_cell(parameters(hail), cell, 1e-4, result, error));
            check_reference(cell, 1e-4, result);
            const auto& channel = result.producer.channels()[selected];
            if (selected < 3)
            {
                REQUIRE(channel.splinter_number_per_kg > 0);
                near(channel.splinter_number_per_kg, 3.5e8L * channel.collection_mass_per_kg);
                near(channel.splinter_reflectivity_per_kg, 20e-24L / (440 * 440) * channel.splinter_number_per_kg);
                if (selected > 0)
                {
                    REQUIRE(result.unrounded.frozen[0].mass_per_kg > 0);
                    CHECK(result.number_increment_per_kg[selected] == channel.diffusion_number_per_kg);
                    CHECK(result.mass_increment_per_kg[0] == channel.splinter_embryo_mass_per_kg);
                    CHECK(result.reflectivity_increment_per_kg[0] == channel.splinter_reflectivity_per_kg);
                }
                else
                {
                    CHECK(result.mass_increment_per_kg[0] == channel.collection_mass_per_kg + channel.diffusion_mass_per_kg);
                }
            }
            else
            {
                CHECK(result.unrounded.frozen[0].mass_per_kg == 0);
                CHECK(channel.splinter_number_per_kg == 0);
                CHECK(channel.splinter_reflectivity_per_kg == 0);
            }
        }
    }
}

TEST_CASE("coupled riming rejects borrowed captures and invalid contexts without publication",
          "[physics][milbrandt][triple_cold_riming]")
{
    const auto cell = source();
    ColdRimingResult accepted;
    std::string error;
    REQUIRE(evaluate_exact_triple_cold_riming_cell(parameters(), cell, 1e-4, accepted, error));
    for (int mutation = 0; mutation < 19; ++mutation)
    {
        auto changed = cell;
        double duration = 1e-4;
        bool hail = true;
        double* field = nullptr;
        if (mutation < 3)
        {
            field = mutation == 0 ? &changed.temperature_k
                : mutation == 1 ? &changed.pressure_pa : &changed.dry_density_kg_m3;
        }
        else if (mutation < 6)
        {
            field = mutation == 3 ? &changed.initial.vapor_mass_per_kg
                : mutation == 4 ? &changed.initial.cloud_mass_per_kg : &changed.initial.cloud_number_per_kg;
        }
        else if (mutation < 18)
        {
            auto& population = changed.initial.frozen[(mutation - 6) / 3];
            field = (mutation - 6) % 3 == 0 ? &population.mass_per_kg
                : (mutation - 6) % 3 == 1 ? &population.number_per_kg : &population.reflectivity_per_kg;
        }
        else
        {
            hail = false;
        }
        if (field)
        {
            *field = std::nextafter(*field, std::numeric_limits<double>::infinity());
        }
        auto result = accepted;
        ColdRimingFailure failure;
        CHECK_FALSE(apply_exact_triple_cold_riming_cell(parameters(hail), changed, duration,
            accepted.producer, result, error, 4, &failure));
        CHECK(failure == ColdRimingFailure::InvalidProducer);
        same_result(result, accepted);
    }
    for (std::size_t capacity : {0u, 3u, 5u})
    {
        auto producer = accepted.producer;
        ColdRimingFailure failure;
        CHECK_FALSE(produce_exact_cold_riming_admission(parameters(), cell, 1e-4, producer, error, capacity, &failure));
        CHECK(failure == ColdRimingFailure::Capacity);
        CHECK(producer.duration_s() == accepted.producer.duration_s());
        CHECK(producer.event_count() == 4);
    }
    for (double duration : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN(), 1e6})
    {
        auto result = accepted;
        CHECK_FALSE(evaluate_exact_triple_cold_riming_cell(parameters(), cell, duration, result, error));
        same_result(result, accepted);
    }
    auto result = accepted;
    CHECK_FALSE(apply_exact_triple_cold_riming_cell(parameters(), cell, 1e-4, ColdRimingProducer{}, result, error));
    same_result(result, accepted);
    CHECK_FALSE(apply_exact_triple_cold_riming_cell(parameters(), cell, std::nextafter(1e-4, 1.0), accepted.producer, result, error));
    same_result(result, accepted);
}

TEST_CASE("coupled riming refines toward an independent full state Runge Kutta trajectory",
          "[physics][milbrandt][triple_cold_riming]")
{
    constexpr double duration = .02;
    for (double saturation : {.98, 1.0, 1.02})
    {
        const auto initial = source(2, saturation, 267.15);
        const auto reference32 = reference::rk4(initial, duration, 32);
        const auto reference64 = reference::rk4(initial, duration, 64);
        const auto before = reference::state(initial);
        const auto distance = [&](const reference::State& a, const reference::State& b)
        {
            long double largest = 0;
            for (std::size_t index = 0; index < a.size(); ++index)
            {
                const long double scale = std::max(std::abs(reference64[index] - before[index]),
                    std::abs(before[index]) * 1e-8L);
                largest = std::max(largest, std::abs(a[index] - b[index]) / std::max(scale, 1e-30L));
            }
            return largest;
        };
        CHECK(distance(reference32, reference64) < 1e-6L);
        std::array<long double, 3> errors{};
        for (int run = 0; run < 3; ++run)
        {
            auto cell = initial;
            const int steps = 4 << run;
            const double step = duration / steps;
            for (int index = 0; index < steps; ++index)
            {
                ColdRimingResult result;
                std::string error;
                const bool ok = evaluate_exact_triple_cold_riming_cell(parameters(), cell, step, result, error);
                INFO(error);
                REQUIRE(ok);
                cell.initial = result.unrounded;
                cell.temperature_k += result.temperature_increment_k;
            }
            errors[run] = distance(reference::state(cell), reference64);
        }
        INFO("errors " << static_cast<double>(errors[0]) << ", " << static_cast<double>(errors[1])
             << ", " << static_cast<double>(errors[2]));
        CHECK(errors[1] < .6L * errors[0]);
        CHECK(errors[2] < .6L * errors[1]);
        CHECK(errors[2] < .03L);
    }
}

TEST_CASE("four channel cloud admission binds every paired transfer before heat and splintering",
          "[physics][milbrandt][triple_cold_riming]")
{
    auto cell = source();
    for (std::size_t index = 0; index < 4; ++index)
    {
        cell.initial.frozen[index] = reference::collection::frozen::population(index, .02, 10, 2);
    }
    const auto unit = reference::evaluate(cell, 1);
    long double mass = 0;
    long double number = 0;
    for (std::size_t index = 0; index < 4; ++index)
    {
        mass += unit.raw_mass[index];
        number += unit.raw_number[index];
    }
    const double duration = static_cast<double>(2 * std::min(cell.initial.cloud_mass_per_kg / mass,
        cell.initial.cloud_number_per_kg / number));
    ColdRimingProducer producer;
    std::string error;
    const bool ok = produce_exact_cold_riming_admission(parameters(), cell, duration, producer, error);
    INFO(error);
    REQUIRE(ok);
    REQUIRE(producer.cloud_admission_factor() < 1);
    const auto expected = reference::evaluate(cell, duration);
    near(producer.cloud_admission_factor(), expected.factor);
    long double mass_loss = 0;
    long double number_loss = 0;
    for (std::size_t index = 0; index < 4; ++index)
    {
        const auto& channel = producer.channels()[index];
        CHECK(channel.collection_mass_per_kg == producer.cloud_admission_factor() * channel.raw_collection_mass_per_kg);
        CHECK(channel.collection_number_per_kg == producer.cloud_admission_factor() * channel.raw_collection_number_per_kg);
        near(channel.riming_diffusion_mass_per_kg, -expected.feedback * expected.collected[index]);
        near(channel.splinter_number_per_kg, expected.splinter_number[index]);
        mass_loss += channel.collection_mass_per_kg;
        number_loss += channel.collection_number_per_kg;
    }
    CHECK(mass_loss <= cell.initial.cloud_mass_per_kg);
    CHECK(number_loss <= cell.initial.cloud_number_per_kg);
    ColdRimingResult result;
    REQUIRE(apply_exact_triple_cold_riming_cell(parameters(), cell, duration, producer, result, error));
    check_reference(cell, duration, result);
}

TEST_CASE("coupled riming source guards cover every reservoir and physical parameter",
          "[physics][milbrandt][triple_cold_riming]")
{
    const auto initial = source();
    ColdRimingResult accepted;
    std::string error;
    REQUIRE(evaluate_exact_triple_cold_riming_cell(parameters(), initial, 1e-4, accepted, error));
    for (std::size_t index = 0; index < 18; ++index)
    {
        auto cell = initial;
        std::array<double*, 18> fields{&cell.temperature_k, &cell.pressure_pa, &cell.dry_density_kg_m3,
            &cell.initial.vapor_mass_per_kg, &cell.initial.cloud_mass_per_kg, &cell.initial.cloud_number_per_kg};
        for (std::size_t species = 0; species < 4; ++species)
        {
            fields[6 + species * 3] = &cell.initial.frozen[species].mass_per_kg;
            fields[7 + species * 3] = &cell.initial.frozen[species].number_per_kg;
            fields[8 + species * 3] = &cell.initial.frozen[species].reflectivity_per_kg;
        }
        for (double invalid : {-1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
        {
            *fields[index] = invalid;
            auto result = accepted;
            CHECK_FALSE(evaluate_exact_triple_cold_riming_cell(parameters(), cell, 1e-4, result, error));
            same_result(result, accepted);
        }
    }
    for (std::size_t index = 0; index < 25; ++index)
    {
        auto physical = parameters();
        std::array<double*, 25> fields{&physical.a_g_, &physical.a_h_, &physical.a_i_, &physical.a_r_, &physical.a_s_,
            &physical.alpha_g_, &physical.alpha_h_, &physical.alpha_i_, &physical.alpha_r_, &physical.alpha_s_,
            &physical.b_g_, &physical.b_h_, &physical.b_i_, &physical.b_r_, &physical.b_s_,
            &physical.c_g_, &physical.c_h_, &physical.c_i_, &physical.c_r_, &physical.c_s_,
            &physical.d_g_, &physical.d_h_, &physical.d_i_, &physical.d_r_, &physical.d_s_};
        *fields[index] = std::nextafter(*fields[index], std::numeric_limits<double>::infinity());
        auto result = accepted;
        CHECK_FALSE(evaluate_exact_triple_cold_riming_cell(physical, initial, 1e-4, result, error));
        same_result(result, accepted);
    }
    auto result = accepted;
    ColdRimingFailure failure;
    CHECK_FALSE(apply_exact_triple_cold_riming_cell(parameters(), initial, 1e6,
        accepted.producer, result, error, 4, &failure));
    CHECK(failure == ColdRimingFailure::InvalidProducer);
    same_result(result, accepted);
    REQUIRE(evaluate_exact_triple_cold_riming_cell(parameters(), initial, 1e-4, result, error));
    same_result(result, accepted);
}

TEST_CASE("secondary ice coupled branch boundaries follow the represented temperature thresholds",
          "[physics][milbrandt][triple_cold_riming]")
{
    for (double boundary : {273.15 - 8.0, 273.15 - 5.0, 273.15 - 3.0})
    {
        for (double temperature : {std::nextafter(boundary, 0.0), boundary,
                                   std::nextafter(boundary, 300.0)})
        {
            const auto cell = source(2, 1, temperature);
            ColdRimingResult result;
            std::string error;
            REQUIRE(evaluate_exact_triple_cold_riming_cell(parameters(), cell, 1e-4, result, error));
            check_reference(cell, 1e-4, result);
            if (temperature <= 273.15 - 8.0 || temperature >= 273.15 - 3.0)
            {
                for (const auto& channel : result.producer.channels())
                {
                    CHECK(channel.splinter_number_per_kg == 0);
                }
            }
        }
    }
}
