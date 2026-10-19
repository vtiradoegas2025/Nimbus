#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_ice_cloud_collection.hpp"
#include "support/milbrandt_ice_cloud_collection_reference.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_ice_cloud_collection_reference;
using Cell = IceCloudCollectionCell;
using Result = IceCloudCollectionResult;
using Producer = IceCloudCollectionProducer;
using Failure = IceCloudCollectionFailure;

ColdParameters parameters(bool hail = true)
{
    return {19.3, 206.89, 71.34, 149.1, 11.72, 0, 0, 0, 0, 0,
        .37, .6384, .6635, .5, .41, 209.439510239, 471.238898038,
        440, 523.598775598, .1597, 3, 3, 3, 3, 2.078, hail};
}

Cell physical_cell(long double alpha = 2, double density = .65,
                   double temperature = 253.15)
{
    return {temperature, 90000, density, {.002, 1e8,
        reference::cloud::frozen::population(0, 2e-4, 2e4, alpha)}};
}

void same_producer(const Producer& actual, const Producer& expected)
{
    CHECK(actual.parameters() == expected.parameters());
    CHECK(actual.hail_processes() == expected.hail_processes());
    CHECK(actual.duration_s() == expected.duration_s());
    CHECK(actual.event_count() == expected.event_count());
    CHECK(actual.cloud_inverse_lambda_m() == expected.cloud_inverse_lambda_m());
    CHECK(actual.source_alpha() == expected.source_alpha());
    CHECK(actual.source_lambda_per_m() == expected.source_lambda_per_m());
    const auto source = [](const Cell& cell)
    {
        return std::array<double, 8>{cell.temperature_k, cell.pressure_pa, cell.dry_density_kg_m3,
            cell.initial.cloud_mass_per_kg, cell.initial.cloud_number_per_kg,
            cell.initial.ice.mass_per_kg, cell.initial.ice.number_per_kg, cell.initial.ice.reflectivity_per_kg};
    };
    CHECK(source(actual.source()) == source(expected.source()));
    const auto channel = [](const Producer& producer)
    {
        const auto& c = producer.channel();
        return std::array<double, 5>{c.raw_mass_increment_per_kg, c.raw_number_increment_per_kg,
            c.admitted_mass_increment_per_kg, c.admitted_number_increment_per_kg, c.donor_admission_factor};
    };
    CHECK(channel(actual) == channel(expected));
}

void same_result(const Result& actual, const Result& expected)
{
    CHECK(reference::vector({0, 90000, .65, actual.unrounded})
          == reference::vector({0, 90000, .65, expected.unrounded}));
    CHECK(actual.rounded_cloud_mass_per_kg == expected.rounded_cloud_mass_per_kg);
    CHECK(actual.rounded_cloud_number_per_kg == expected.rounded_cloud_number_per_kg);
    CHECK(actual.rounded_ice.mass_per_kg == expected.rounded_ice.mass_per_kg);
    CHECK(actual.rounded_ice.number_per_kg == expected.rounded_ice.number_per_kg);
    CHECK(actual.rounded_ice.reflectivity_per_kg == expected.rounded_ice.reflectivity_per_kg);
    CHECK(actual.cloud_mass_increment_per_kg == expected.cloud_mass_increment_per_kg);
    CHECK(actual.cloud_number_increment_per_kg == expected.cloud_number_increment_per_kg);
    CHECK(actual.ice_mass_increment_per_kg == expected.ice_mass_increment_per_kg);
    CHECK(actual.ice_reflectivity_increment_per_kg == expected.ice_reflectivity_increment_per_kg);
    CHECK(actual.temperature_increment_k == expected.temperature_increment_k);
    same_producer(actual.producer, expected.producer);
}

void check_reference(const Cell& cell, double duration, const Result& result,
                     const reference::Reference& expected)
{
    const auto& event = result.producer.channel();
    CHECK(result.producer.event_count() == 1);
    CHECK(result.producer.parameters() == default_cold_parameters);
    CHECK(result.producer.duration_s() == duration);
    CHECK(result.producer.source_alpha() == Approx(static_cast<double>(expected.spectrum.alpha)).epsilon(2e-10).margin(2e-10));
    CHECK(result.producer.source_lambda_per_m() == Approx(static_cast<double>(expected.spectrum.lambda)).epsilon(3e-12));
    CHECK(event.raw_mass_increment_per_kg == Approx(static_cast<double>(expected.raw_mass)).epsilon(3e-10));
    CHECK(event.raw_number_increment_per_kg == Approx(static_cast<double>(expected.raw_number)).epsilon(3e-10));
    CHECK(event.donor_admission_factor == Approx(static_cast<double>(expected.factor)).epsilon(3e-12));
    CHECK(result.ice_mass_increment_per_kg == Approx(static_cast<double>(expected.mass)).epsilon(3e-10));
    CHECK(result.cloud_number_increment_per_kg == Approx(static_cast<double>(-expected.number)).epsilon(3e-10));
    CHECK(result.ice_reflectivity_increment_per_kg == Approx(static_cast<double>(expected.reflectivity_increment)).epsilon(3e-10));
    CHECK(result.temperature_increment_k == Approx(static_cast<double>(expected.temperature_increment)).epsilon(3e-10));
    CHECK(result.ice_mass_increment_per_kg == -result.cloud_mass_increment_per_kg);
    CHECK(result.unrounded.ice.number_per_kg == cell.initial.ice.number_per_kg);
    CHECK(result.unrounded.ice.mass_per_kg == Approx(cell.initial.ice.mass_per_kg + static_cast<double>(expected.mass)).epsilon(3e-12));
    CHECK(result.unrounded.ice.reflectivity_per_kg == Approx(cell.initial.ice.reflectivity_per_kg
        + static_cast<double>(expected.reflectivity_increment)).epsilon(3e-12));
    CHECK(result.rounded_cloud_mass_per_kg == static_cast<float>(result.unrounded.cloud_mass_per_kg));
    CHECK(result.rounded_cloud_number_per_kg == static_cast<float>(result.unrounded.cloud_number_per_kg));
    CHECK(result.rounded_ice.mass_per_kg == static_cast<float>(result.unrounded.ice.mass_per_kg));
    CHECK(result.rounded_ice.number_per_kg == static_cast<float>(result.unrounded.ice.number_per_kg));
    CHECK(result.rounded_ice.reflectivity_per_kg == static_cast<float>(result.unrounded.ice.reflectivity_per_kg));
    const long double heat_error = reference::cloud::heat_capacity * result.temperature_increment_k
        - reference::cloud::latent_heat_fusion * result.ice_mass_increment_per_kg;
    CHECK(std::abs(heat_error) <= 16 * std::numeric_limits<double>::epsilon()
        * reference::cloud::latent_heat_fusion * result.ice_mass_increment_per_kg);
}

std::array<double ColdParameters::*, 25> parameter_members()
{
    return {&ColdParameters::a_g_, &ColdParameters::a_h_, &ColdParameters::a_i_,
        &ColdParameters::a_r_, &ColdParameters::a_s_, &ColdParameters::alpha_g_,
        &ColdParameters::alpha_h_, &ColdParameters::alpha_i_, &ColdParameters::alpha_r_,
        &ColdParameters::alpha_s_, &ColdParameters::b_g_, &ColdParameters::b_h_,
        &ColdParameters::b_i_, &ColdParameters::b_r_, &ColdParameters::b_s_,
        &ColdParameters::c_g_, &ColdParameters::c_h_, &ColdParameters::c_i_,
        &ColdParameters::c_r_, &ColdParameters::c_s_, &ColdParameters::d_g_,
        &ColdParameters::d_h_, &ColdParameters::d_i_, &ColdParameters::d_r_, &ColdParameters::d_s_};
}

Result sentinel()
{
    Result result;
    std::string error;
    REQUIRE(evaluate_exact_triple_ice_cloud_collection_cell(parameters(), physical_cell(), 1e-4, result, error));
    return result;
}

} // namespace

TEST_CASE("triple ice cloud collection matches independent velocity weighted integrals without snow factors",
          "[physics][milbrandt][triple_ice_cloud_collection]")
{
    for (long double alpha : {-.5L, 0.0L, 2.0L, 20.0L, 199.9L})
    {
        for (double density : {.35, .65, 1.2})
        {
            for (double temperature : {233.15, 253.15, 272.15})
            {
                const auto cell = physical_cell(alpha, density, temperature);
                constexpr double duration = 1e-4;
                Result result;
                std::string error;
                REQUIRE(evaluate_exact_triple_ice_cloud_collection_cell(parameters(), cell, duration, result, error));
                CHECK(error.empty());
                check_reference(cell, duration, result, reference::evaluate(cell, duration));
                Result replayed;
                REQUIRE(apply_exact_triple_ice_cloud_collection_cell(
                    parameters(), cell, duration, result.producer, replayed, error));
                same_result(replayed, result);
            }
        }
        const auto cell = physical_cell(alpha);
        const auto analytic = reference::evaluate(cell, 1e-4);
        const auto quadrature = reference::evaluate(cell, 1e-4, true);
        CHECK(static_cast<double>(analytic.raw_mass) == Approx(static_cast<double>(quadrature.raw_mass)).epsilon(4e-11));
        CHECK(static_cast<double>(analytic.raw_number) == Approx(static_cast<double>(quadrature.raw_number)).epsilon(4e-11));
    }
}

TEST_CASE("triple ice cloud collection refinement converges to independent full state RK4",
          "[physics][milbrandt][triple_ice_cloud_collection]")
{
    const auto initial = physical_cell();
    const auto source = reference::vector(initial);
    constexpr double duration = 10;
    const auto expected = reference::rk4(initial, duration, 64);
    const auto refined = reference::rk4(initial, duration, 128);
    for (std::size_t index = 0; index < source.size(); ++index)
    {
        CHECK(std::abs(expected[index] - refined[index]) / source[index] < 2e-11L);
    }
    std::array<long double, 3> errors{};
    for (std::size_t refinement = 0; refinement < errors.size(); ++refinement)
    {
        const int steps = 4 << refinement;
        const double h = duration / steps;
        auto cell = initial;
        Result result;
        std::string error;
        for (int step = 0; step < steps; ++step)
        {
            REQUIRE(evaluate_exact_triple_ice_cloud_collection_cell(parameters(), cell, h, result, error));
            cell.initial = result.unrounded;
            cell.temperature_k += result.temperature_increment_k;
        }
        const auto actual = reference::vector(cell);
        for (std::size_t index = 0; index < actual.size(); ++index)
        {
            errors[refinement] = std::max(errors[refinement], std::abs(actual[index] - refined[index]) / source[index]);
        }
    }
    CHECK(errors[0] > 1e-10L);
    CHECK(errors[1] < .6L * errors[0]);
    CHECK(errors[2] < .6L * errors[1]);
    CHECK(errors[2] < 1e-3L);
}

TEST_CASE("triple ice cloud admission pairs donor mass and number without asserting endpoint realizability",
          "[physics][milbrandt][triple_ice_cloud_collection]")
{
    const auto cell = physical_cell();
    const auto unit = reference::evaluate(cell, 1);
    const double duration = static_cast<double>(2 * std::min(
        static_cast<long double>(cell.initial.cloud_mass_per_kg) / unit.raw_mass,
        static_cast<long double>(cell.initial.cloud_number_per_kg) / unit.raw_number));
    const auto expected = reference::evaluate(cell, duration);
    REQUIRE(expected.factor < 1);
    Producer producer;
    std::string error;
    Failure failure = Failure::InvalidProducer;
    REQUIRE(produce_exact_ice_cloud_collection_admission(parameters(), cell, duration, producer, error, 1, &failure));
    CHECK(failure == Failure::None);
    CHECK(producer.channel().raw_mass_increment_per_kg == Approx(static_cast<double>(expected.raw_mass)).epsilon(3e-10));
    CHECK(producer.channel().raw_number_increment_per_kg == Approx(static_cast<double>(expected.raw_number)).epsilon(3e-10));
    CHECK(producer.channel().donor_admission_factor == Approx(static_cast<double>(expected.factor)).epsilon(3e-10));
    CHECK(producer.channel().admitted_mass_increment_per_kg == Approx(static_cast<double>(expected.mass)).epsilon(3e-10));
    CHECK(producer.channel().admitted_number_increment_per_kg == Approx(static_cast<double>(expected.number)).epsilon(3e-10));
    CHECK(producer.channel().admitted_mass_increment_per_kg <= cell.initial.cloud_mass_per_kg * (1 + 4e-15));
    CHECK(producer.channel().admitted_number_increment_per_kg <= cell.initial.cloud_number_per_kg * (1 + 4e-15));
    const auto before = sentinel();
    auto result = before;
    CHECK_FALSE(apply_exact_triple_ice_cloud_collection_cell(
        parameters(), cell, duration, producer, result, error, 1, &failure));
    CHECK(failure == Failure::IntervalRejected);
    same_result(result, before);
    auto warm = physical_cell(2, .65, 273.1499);
    const auto warm_unit = reference::evaluate(warm, 1);
    const double crossing = static_cast<double>(2 * (273.15L - warm.temperature_k) / warm_unit.temperature_increment);
    REQUIRE(reference::evaluate(warm, crossing).factor == 1);
    result = before;
    CHECK_FALSE(evaluate_exact_triple_ice_cloud_collection_cell(
        parameters(), warm, crossing, result, error, 1, &failure));
    CHECK(failure == Failure::IntervalRejected);
    CHECK(error.find("heating") != std::string::npos);
    same_result(result, before);
}

TEST_CASE("triple ice cloud invalid input and resource rejection preserves captured state",
          "[physics][milbrandt][triple_ice_cloud_collection]")
{
    const auto original = physical_cell();
    const auto before = sentinel();
    const auto reject = [&](const ColdParameters& p, const Cell& cell, double duration,
                            std::size_t capacity, Failure expected)
    {
        auto result = before;
        auto producer = before.producer;
        std::string error;
        Failure failure = Failure::None;
        CHECK_FALSE(evaluate_exact_triple_ice_cloud_collection_cell(p, cell, duration, result, error, capacity, &failure));
        CHECK(failure == expected);
        CHECK_FALSE(error.empty());
        same_result(result, before);
        CHECK_FALSE(produce_exact_ice_cloud_collection_admission(p, cell, duration, producer, error, capacity, &failure));
        CHECK(failure == expected);
        same_producer(producer, before.producer);
    };
    for (std::size_t capacity : {0u, 2u, 16u, std::numeric_limits<unsigned>::max()})
    {
        reject(parameters(), original, 1e-4, capacity, Failure::Capacity);
    }
    for (double duration : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        reject(parameters(), original, duration, 1, Failure::InvalidContext);
    }
    for (double temperature : {149.999, 273.15, 400.0, std::numeric_limits<double>::quiet_NaN()})
    {
        auto cell = original;
        cell.temperature_k = temperature;
        reject(parameters(), cell, 1e-4, 1, Failure::InvalidContext);
    }
    for (double pressure : {99.999, 200001.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        auto cell = original;
        cell.pressure_pa = pressure;
        reject(parameters(), cell, 1e-4, 1, Failure::InvalidContext);
    }
    for (double density : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        auto cell = original;
        cell.dry_density_kg_m3 = density;
        reject(parameters(), cell, 1e-4, 1, Failure::InvalidContext);
    }
    for (auto member : parameter_members())
    {
        auto p = parameters();
        p.*member = std::nextafter(p.*member, std::numeric_limits<double>::infinity());
        reject(p, original, 1e-4, 1, Failure::InvalidContext);
    }
    for (double IceCloudCollectionState::*member : {
            &IceCloudCollectionState::cloud_mass_per_kg, &IceCloudCollectionState::cloud_number_per_kg})
    {
        for (double value : {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
        {
            auto cell = original;
            cell.initial.*member = value;
            reject(parameters(), cell, 1e-4, 1, Failure::InvalidSource);
        }
    }
    auto number_exhausted = original;
    number_exhausted.initial.cloud_number_per_kg = 0;
    reject(parameters(), number_exhausted, 1e-4, 1, Failure::InvalidSource);
    for (double FrozenCollectionPopulation::*member : {&FrozenCollectionPopulation::mass_per_kg,
            &FrozenCollectionPopulation::number_per_kg, &FrozenCollectionPopulation::reflectivity_per_kg})
    {
        for (double value : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
        {
            auto cell = original;
            cell.initial.ice.*member = value;
            reject(parameters(), cell, 1e-4, 1, Failure::InvalidSource);
        }
    }
    auto too_narrow = original;
    too_narrow.initial.ice = reference::cloud::frozen::population(0, 2e-4, 2e4, 201);
    reject(parameters(), too_narrow, 1e-4, 1, Failure::InvalidSource);
    auto cauchy = original;
    cauchy.initial.ice.reflectivity_per_kg = .99 * cauchy.initial.ice.mass_per_kg
        * cauchy.initial.ice.mass_per_kg / (440 * 440 * cauchy.initial.ice.number_per_kg);
    reject(parameters(), cauchy, 1e-4, 1, Failure::InvalidSource);
}

TEST_CASE("triple ice cloud immutable capture binds zero source bits parameters and policy",
          "[physics][milbrandt][triple_ice_cloud_collection]")
{
    const auto cell = physical_cell();
    const auto before = sentinel();
    const auto reject = [&](const ColdParameters& p, const Cell& changed, double duration, const Producer& producer)
    {
        auto result = before;
        std::string error;
        Failure failure = Failure::None;
        CHECK_FALSE(apply_exact_triple_ice_cloud_collection_cell(p, changed, duration, producer, result, error, 1, &failure));
        CHECK(failure == Failure::InvalidProducer);
        same_result(result, before);
    };
    reject(parameters(), cell, 1e-4, Producer{});
    reject(parameters(), cell, std::nextafter(1e-4, 1.0), before.producer);
    for (double Cell::*member : {&Cell::temperature_k, &Cell::pressure_pa, &Cell::dry_density_kg_m3})
    {
        auto changed = cell;
        changed.*member = std::nextafter(changed.*member, std::numeric_limits<double>::infinity());
        reject(parameters(), changed, 1e-4, before.producer);
    }
    for (double IceCloudCollectionState::*member : {
            &IceCloudCollectionState::cloud_mass_per_kg, &IceCloudCollectionState::cloud_number_per_kg})
    {
        auto changed = cell;
        changed.initial.*member = std::nextafter(changed.initial.*member, std::numeric_limits<double>::infinity());
        reject(parameters(), changed, 1e-4, before.producer);
    }
    for (double FrozenCollectionPopulation::*member : {&FrozenCollectionPopulation::mass_per_kg,
            &FrozenCollectionPopulation::number_per_kg, &FrozenCollectionPopulation::reflectivity_per_kg})
    {
        auto changed = cell;
        changed.initial.ice.*member = std::nextafter(changed.initial.ice.*member, std::numeric_limits<double>::infinity());
        reject(parameters(), changed, 1e-4, before.producer);
    }
    for (auto member : parameter_members())
    {
        auto p = parameters();
        p.*member = std::nextafter(p.*member, std::numeric_limits<double>::infinity());
        reject(p, cell, 1e-4, before.producer);
    }
    reject(parameters(false), cell, 1e-4, before.producer);
    Result disabled;
    std::string error;
    REQUIRE(evaluate_exact_triple_ice_cloud_collection_cell(parameters(false), cell, 1e-4, disabled, error));
    CHECK(disabled.ice_mass_increment_per_kg == before.ice_mass_increment_per_kg);
    CHECK(disabled.ice_reflectivity_increment_per_kg == before.ice_reflectivity_increment_per_kg);
    CHECK_FALSE(disabled.producer.hail_processes());
    reject(parameters(), cell, 1e-4, disabled.producer);
    auto detached = before;
    detached.unrounded.ice.mass_per_kg = -1;
    detached.ice_reflectivity_increment_per_kg = -2;
    detached.rounded_cloud_number_per_kg = -3;
    detached.temperature_increment_k = 4;
    REQUIRE(apply_exact_triple_ice_cloud_collection_cell(parameters(), cell, 1e-4,
        detached.producer, detached, error));
    same_result(detached, before);
    Cell empty{253.15, 90000, .65, {}};
    Producer producer;
    REQUIRE(produce_exact_ice_cloud_collection_admission(parameters(), empty, 1e-4, producer, error));
    empty.initial.ice.mass_per_kg = -0.0;
    reject(parameters(), empty, 1e-4, producer);
}

TEST_CASE("triple ice cloud zero cutoffs and FP32 endpoints preserve ownership without initiating ice",
          "[physics][milbrandt][triple_ice_cloud_collection]")
{
    std::string error;
    for (double temperature : {150.0, 233.15, std::nextafter(273.15, 0.0)})
    {
        for (double cloud_number : {0.0, 1e8})
        {
            Cell empty{temperature, 100, .65, {0, cloud_number, {}}};
            Result result;
            REQUIRE(evaluate_exact_triple_ice_cloud_collection_cell(parameters(), empty, 1, result, error));
            CHECK(result.unrounded.cloud_number_per_kg == cloud_number);
            CHECK(result.unrounded.ice.mass_per_kg == 0);
            CHECK(result.unrounded.ice.number_per_kg == 0);
            CHECK(result.unrounded.ice.reflectivity_per_kg == 0);
            CHECK(result.temperature_increment_k == 0);
        }
    }
    auto unseeded = physical_cell(2, .65, 233.15);
    unseeded.initial.ice = {};
    Result result;
    REQUIRE(evaluate_exact_triple_ice_cloud_collection_cell(parameters(), unseeded, 1, result, error));
    CHECK(result.unrounded.cloud_mass_per_kg == unseeded.initial.cloud_mass_per_kg);
    CHECK(result.unrounded.cloud_number_per_kg == unseeded.initial.cloud_number_per_kg);
    CHECK(result.unrounded.ice.mass_per_kg == 0);
    CHECK(result.ice_mass_increment_per_kg == 0);
    auto cutoff = physical_cell();
    cutoff.initial.cloud_mass_per_kg = 1e-14;
    REQUIRE(evaluate_exact_triple_ice_cloud_collection_cell(parameters(), cutoff, 1e-4, result, error));
    CHECK(result.producer.cloud_inverse_lambda_m() > 0);
    CHECK(result.producer.channel().raw_mass_increment_per_kg == 0);
    cutoff.initial.cloud_mass_per_kg = std::nextafter(1e-14, 1.0);
    Producer producer;
    REQUIRE(produce_exact_ice_cloud_collection_admission(parameters(), cutoff, 1e-4, producer, error));
    CHECK(producer.channel().raw_mass_increment_per_kg > 0);
    CHECK(producer.channel().raw_number_increment_per_kg > 0);
    Cell tiny{253.15, 90000, .65, {}};
    tiny.initial.ice = reference::cloud::frozen::population(0, 1e-40, 1e-3, 2);
    REQUIRE(tiny.initial.ice.reflectivity_per_kg > 0);
    REQUIRE(static_cast<float>(tiny.initial.ice.reflectivity_per_kg) == 0);
    REQUIRE(produce_exact_ice_cloud_collection_admission(parameters(), tiny, 1e-4, producer, error));
    const auto before = sentinel();
    result = before;
    Failure failure = Failure::None;
    CHECK_FALSE(apply_exact_triple_ice_cloud_collection_cell(parameters(), tiny, 1e-4, producer, result, error, 1, &failure));
    CHECK(failure == Failure::IntervalRejected);
    same_result(result, before);
}

TEST_CASE("triple ice cloud pure one and four worker continuation is exact",
          "[physics][milbrandt][triple_ice_cloud_collection]")
{
    constexpr int columns = 16;
    std::array<Cell, columns> sources{};
    std::array<Result, columns> serial{};
    std::array<Result, columns> parallel{};
    std::array<bool, columns> passed{};
    std::array<std::string, columns> errors{};
    int observed_workers = 1;
    for (int column = 0; column < columns; ++column)
    {
        sources[column] = physical_cell(column % 4 * 3, .4 + .05 * column, 243.15 + column);
        REQUIRE(evaluate_exact_triple_ice_cloud_collection_cell(parameters(), sources[column], .01, serial[column], errors[column]));
    }
#ifdef _OPENMP
#pragma omp parallel num_threads(4)
    {
#pragma omp single
        observed_workers = omp_get_num_threads();
#pragma omp for
#endif
    for (int column = 0; column < columns; ++column)
    {
        passed[column] = evaluate_exact_triple_ice_cloud_collection_cell(parameters(), sources[column], .01, parallel[column], errors[column]);
    }
#ifdef _OPENMP
    }
    REQUIRE(observed_workers == 4);
#else
    CHECK(observed_workers == 1);
#endif
    for (int column = 0; column < columns; ++column)
    {
        REQUIRE(passed[column]);
        same_result(serial[column], parallel[column]);
        sources[column].initial = serial[column].unrounded;
        sources[column].temperature_k += serial[column].temperature_increment_k;
        REQUIRE(evaluate_exact_triple_ice_cloud_collection_cell(parameters(), sources[column], .01, serial[column], errors[column]));
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(4)
#endif
    for (int column = 0; column < columns; ++column)
    {
        passed[column] = evaluate_exact_triple_ice_cloud_collection_cell(parameters(), sources[column], .01, parallel[column], errors[column]);
    }
    for (int column = 0; column < columns; ++column)
    {
        REQUIRE(passed[column]);
        same_result(serial[column], parallel[column]);
        check_reference(sources[column], .01, parallel[column], reference::evaluate(sources[column], .01));
    }
}
