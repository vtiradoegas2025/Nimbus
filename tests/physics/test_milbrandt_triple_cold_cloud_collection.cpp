#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "microphysics/schemes/milbrandt/triple_cold_cloud_collection.hpp"
#include "../support/milbrandt_cold_cloud_collection_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_cold_cloud_collection_reference;
using Population = FrozenCollectionPopulation;
using Cell = ColdCloudCollectionCell;
using Result = ColdCloudCollectionResult;
using Producer = ColdCloudCollectionProducer;
using Failure = ColdCloudCollectionFailure;

ColdParameters parameters(bool hail = true)
{
    return {19.3, 206.89, 71.34, 149.1, 11.72, 0, 0, 0, 0, 0,
            .37, .6384, .6635, .5, .41, 209.439510239, 471.238898038,
            440, 523.598775598, .1597, 3, 3, 3, 3, 2.078, hail};
}

Cell physical_cell(long double alpha = 2, double density = .65,
                   double temperature = 253.15)
{
    Cell result{temperature, 90000, density, {.002, 1e8, {}}};
    constexpr std::array<double, 3> mass{5e-4, 3e-4, 2e-4};
    constexpr std::array<double, 3> number{40, 15, 2};
    for (std::size_t species = 0; species < 3; ++species)
    {
        result.initial.frozen[species] = reference::population(species, mass[species], number[species], alpha);
    }
    return result;
}

void same_population(const Population& actual, const Population& expected)
{
    CHECK(actual.mass_per_kg == expected.mass_per_kg);
    CHECK(actual.number_per_kg == expected.number_per_kg);
    CHECK(actual.reflectivity_per_kg == expected.reflectivity_per_kg);
}

void same_producer(const Producer& actual, const Producer& expected)
{
    CHECK(actual.source().temperature_k == expected.source().temperature_k);
    CHECK(actual.source().pressure_pa == expected.source().pressure_pa);
    CHECK(actual.source().dry_density_kg_m3 == expected.source().dry_density_kg_m3);
    CHECK(actual.source().initial.cloud_mass_per_kg == expected.source().initial.cloud_mass_per_kg);
    CHECK(actual.source().initial.cloud_number_per_kg == expected.source().initial.cloud_number_per_kg);
    CHECK(actual.parameters() == expected.parameters());
    CHECK(actual.hail_processes() == expected.hail_processes());
    CHECK(actual.duration_s() == expected.duration_s());
    CHECK(actual.event_count() == expected.event_count());
    CHECK(actual.cloud_inverse_lambda_m() == expected.cloud_inverse_lambda_m());
    CHECK(actual.source_alpha() == expected.source_alpha());
    CHECK(actual.source_lambda_per_m() == expected.source_lambda_per_m());
    CHECK(actual.mass_weighted_speed_m_s() == expected.mass_weighted_speed_m_s());
    CHECK(actual.efficiencies() == expected.efficiencies());
    for (std::size_t species = 0; species < 3; ++species)
    {
        same_population(actual.source().initial.frozen[species], expected.source().initial.frozen[species]);
        const auto& a = actual.channels()[species];
        const auto& e = expected.channels()[species];
        CHECK(a.channel == e.channel);
        CHECK(a.raw_mass_increment_per_kg == e.raw_mass_increment_per_kg);
        CHECK(a.raw_number_increment_per_kg == e.raw_number_increment_per_kg);
        CHECK(a.admitted_mass_increment_per_kg == e.admitted_mass_increment_per_kg);
        CHECK(a.admitted_number_increment_per_kg == e.admitted_number_increment_per_kg);
        CHECK(a.donor_admission_factor == e.donor_admission_factor);
    }
}

void same_result(const Result& actual, const Result& expected)
{
    CHECK(actual.unrounded.cloud_mass_per_kg == expected.unrounded.cloud_mass_per_kg);
    CHECK(actual.unrounded.cloud_number_per_kg == expected.unrounded.cloud_number_per_kg);
    CHECK(actual.rounded_cloud_mass_per_kg == expected.rounded_cloud_mass_per_kg);
    CHECK(actual.rounded_cloud_number_per_kg == expected.rounded_cloud_number_per_kg);
    CHECK(actual.mass_increment_per_kg == expected.mass_increment_per_kg);
    CHECK(actual.reflectivity_increment_per_kg == expected.reflectivity_increment_per_kg);
    CHECK(actual.cloud_mass_increment_per_kg == expected.cloud_mass_increment_per_kg);
    CHECK(actual.cloud_number_increment_per_kg == expected.cloud_number_increment_per_kg);
    CHECK(actual.temperature_increment_k == expected.temperature_increment_k);
    for (std::size_t species = 0; species < 3; ++species)
    {
        same_population(actual.unrounded.frozen[species], expected.unrounded.frozen[species]);
        CHECK(actual.rounded[species].mass_per_kg == expected.rounded[species].mass_per_kg);
        CHECK(actual.rounded[species].number_per_kg == expected.rounded[species].number_per_kg);
        CHECK(actual.rounded[species].reflectivity_per_kg == expected.rounded[species].reflectivity_per_kg);
    }
    same_producer(actual.producer, expected.producer);
}

void check_capture(const Cell& cell, double duration, const Producer& actual,
                   const reference::Reference& expected)
{
    constexpr std::array<ColdCloudCollectionChannel, 3> channel_names{
        ColdCloudCollectionChannel::CloudToSnow, ColdCloudCollectionChannel::CloudToGraupel,
        ColdCloudCollectionChannel::CloudToHail};
    CHECK(actual.event_count() == 3);
    CHECK(actual.parameters() == default_cold_parameters);
    CHECK(actual.duration_s() == duration);
    CHECK(actual.source().temperature_k == cell.temperature_k);
    CHECK(actual.source().pressure_pa == cell.pressure_pa);
    CHECK(actual.source().dry_density_kg_m3 == cell.dry_density_kg_m3);
    CHECK(actual.source().initial.cloud_mass_per_kg == cell.initial.cloud_mass_per_kg);
    CHECK(actual.source().initial.cloud_number_per_kg == cell.initial.cloud_number_per_kg);
    CHECK(actual.cloud_inverse_lambda_m() == Approx(
        static_cast<double>(expected.cloud_inverse_lambda)).epsilon(3e-14).margin(1e-100));
    for (std::size_t species = 0; species < 3; ++species)
    {
        same_population(actual.source().initial.frozen[species], cell.initial.frozen[species]);
        if (cell.initial.frozen[species].mass_per_kg > 0)
        {
            CHECK(actual.source_alpha()[species] == Approx(
                static_cast<double>(expected.spectra[species].alpha)).epsilon(2e-9).margin(2e-10));
            CHECK(actual.source_lambda_per_m()[species] == Approx(
                static_cast<double>(expected.spectra[species].lambda)).epsilon(2e-10));
            CHECK(actual.mass_weighted_speed_m_s()[species] == Approx(
                static_cast<double>(expected.spectra[species].velocity)).epsilon(2e-10));
        }
        CHECK(actual.efficiencies()[species] == Approx(
            static_cast<double>(expected.efficiencies[species])).epsilon(3e-10).margin(1e-100));
        const auto& a = actual.channels()[species];
        const auto& e = expected.channels[species];
        CHECK(a.channel == channel_names[species]);
        CHECK(a.raw_mass_increment_per_kg == Approx(static_cast<double>(e.raw_mass)).epsilon(3e-10).margin(1e-30));
        CHECK(a.raw_number_increment_per_kg == Approx(static_cast<double>(e.raw_number)).epsilon(3e-10).margin(1e-20));
        CHECK(a.admitted_mass_increment_per_kg == Approx(static_cast<double>(e.mass)).epsilon(3e-10).margin(1e-30));
        CHECK(a.admitted_number_increment_per_kg == Approx(static_cast<double>(e.number)).epsilon(3e-10).margin(1e-20));
        CHECK(a.donor_admission_factor == Approx(static_cast<double>(e.factor)).epsilon(3e-10));
    }
}

void check_endpoint(const Cell& cell, const Result& actual,
                    const reference::Reference& expected)
{
    CHECK(actual.cloud_mass_increment_per_kg == Approx(
        static_cast<double>(expected.cloud_mass_increment)).epsilon(3e-10).margin(1e-30));
    CHECK(actual.cloud_number_increment_per_kg == Approx(
        static_cast<double>(expected.cloud_number_increment)).epsilon(3e-10).margin(1e-20));
    CHECK(actual.unrounded.cloud_mass_per_kg == Approx(
        static_cast<double>(expected.cloud_mass_after)).epsilon(3e-12).margin(1e-30));
    CHECK(actual.unrounded.cloud_number_per_kg == Approx(
        static_cast<double>(expected.cloud_number_after)).epsilon(3e-12).margin(1e-20));
    CHECK(actual.temperature_increment_k == Approx(
        static_cast<double>(expected.temperature_increment)).epsilon(3e-10).margin(1e-30));
    CHECK(actual.rounded_cloud_mass_per_kg == static_cast<float>(actual.unrounded.cloud_mass_per_kg));
    CHECK(actual.rounded_cloud_number_per_kg == static_cast<float>(actual.unrounded.cloud_number_per_kg));
    CHECK(cell.temperature_k + actual.temperature_increment_k < 273.15);
    long double water_change = actual.cloud_mass_increment_per_kg;
    long double transfer_scale = std::abs(actual.cloud_mass_increment_per_kg);
    double cloud_number_loss = 0;
    for (std::size_t species = 0; species < 3; ++species)
    {
        const auto& endpoint = actual.unrounded.frozen[species];
        CHECK(actual.mass_increment_per_kg[species] == Approx(
            static_cast<double>(expected.mass_increment[species])).epsilon(3e-10).margin(1e-30));
        CHECK(actual.reflectivity_increment_per_kg[species] == Approx(
            static_cast<double>(expected.reflectivity_increment[species])).epsilon(3e-10).margin(1e-38));
        CHECK(endpoint.mass_per_kg == Approx(static_cast<double>(expected.endpoint[species][0])).epsilon(3e-12));
        CHECK(endpoint.number_per_kg == cell.initial.frozen[species].number_per_kg);
        CHECK(endpoint.reflectivity_per_kg == Approx(static_cast<double>(expected.endpoint[species][2])).epsilon(3e-12));
        CHECK(actual.rounded[species].mass_per_kg == static_cast<float>(endpoint.mass_per_kg));
        CHECK(actual.rounded[species].number_per_kg == static_cast<float>(endpoint.number_per_kg));
        CHECK(actual.rounded[species].reflectivity_per_kg == static_cast<float>(endpoint.reflectivity_per_kg));
        water_change += actual.mass_increment_per_kg[species];
        transfer_scale += std::abs(actual.mass_increment_per_kg[species]);
        cloud_number_loss += actual.producer.channels()[species].admitted_number_increment_per_kg;
        if (endpoint.mass_per_kg > 0)
        {
            const long double alpha = reference::frozen::represented_alpha(endpoint, species + 1);
            CHECK(alpha > -1);
            CHECK(alpha <= 200 + 1e-8L);
        }
    }
    CHECK(std::abs(water_change) <= 8 * std::numeric_limits<double>::epsilon()
        * std::max(transfer_scale, 1e-30L));
    CHECK(cloud_number_loss == -actual.cloud_number_increment_per_kg);
    CHECK(reference::heat_capacity * actual.temperature_increment_k
          + reference::latent_heat_fusion * actual.cloud_mass_increment_per_kg
          == Approx(0).margin(static_cast<double>(64 * std::numeric_limits<double>::epsilon()
              * reference::latent_heat_fusion * std::max(transfer_scale, 1e-30L))));
    if (expected.mass_increment[0] > 0)
    {
        const long double spherical = 2 * cell.initial.frozen[0].reflectivity_per_kg
            * expected.mass_increment[0] / cell.initial.frozen[0].mass_per_kg;
        CHECK(std::abs(spherical - expected.reflectivity_increment[0])
              > .25L * expected.reflectivity_increment[0]);
    }
}

Result successful_sentinel()
{
    Result result;
    std::string error;
    REQUIRE(evaluate_exact_triple_cold_cloud_collection_cell(
        parameters(), physical_cell(), 1e-4, result, error));
    return result;
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
        &ColdParameters::d_h_, &ColdParameters::d_i_, &ColdParameters::d_r_,
        &ColdParameters::d_s_};
}

} // namespace

TEST_CASE("triple cold cloud collection matches independent generalized cloud and frozen spectra",
          "[physics][milbrandt][triple_cold_cloud_collection]")
{
    for (long double alpha : {-.5L, 0.0L, 2.0L, 20.0L, 199.9L})
    {
        for (double density : {.35, .65, 1.2})
        {
            for (double temperature : {233.15, 253.15, 272.15})
            {
                const auto cell = physical_cell(alpha, density, temperature);
                constexpr double duration = 1e-4;
                INFO("alpha=" << static_cast<double>(alpha) << " rho=" << density
                     << " T=" << temperature);
                const auto expected = reference::evaluate(cell, duration);
                Result result;
                std::string error = "sentinel";
                Failure failure = Failure::InvalidProducer;
                REQUIRE(evaluate_exact_triple_cold_cloud_collection_cell(
                    parameters(), cell, duration, result, error, 3, &failure));
                CHECK(error.empty());
                CHECK(failure == Failure::None);
                check_capture(cell, duration, result.producer, expected);
                check_endpoint(cell, result, expected);
                Result replayed;
                REQUIRE(apply_exact_triple_cold_cloud_collection_cell(
                    parameters(), cell, duration, result.producer, replayed, error));
                same_result(replayed, result);
            }
        }
    }
}

TEST_CASE("triple cold cloud collection velocity weighted quadrature detects old gamma shift defect",
          "[physics][milbrandt][triple_cold_cloud_collection]")
{
    for (long double alpha : {-.5L, 0.0L, 2.0L, 20.0L, 199.9L})
    {
        const auto cell = physical_cell(alpha);
        constexpr double duration = 1e-4;
        const auto analytic = reference::evaluate(cell, duration);
        const auto quadrature = reference::evaluate(cell, duration, true);
        Result result;
        std::string error;
        REQUIRE(evaluate_exact_triple_cold_cloud_collection_cell(parameters(), cell, duration, result, error));
        for (std::size_t species = 0; species < 3; ++species)
        {
            const auto old = reference::missing_fall_exponent_kernel(cell, analytic, species, duration);
            const auto& actual = result.producer.channels()[species];
            CHECK(actual.raw_mass_increment_per_kg == Approx(
                static_cast<double>(quadrature.channels[species].raw_mass)).epsilon(3e-10));
            CHECK(actual.raw_number_increment_per_kg == Approx(
                static_cast<double>(quadrature.channels[species].raw_number)).epsilon(3e-10));
            CHECK(static_cast<double>(analytic.channels[species].raw_mass) == Approx(
                static_cast<double>(quadrature.channels[species].raw_mass)).epsilon(4e-11));
            CHECK(static_cast<double>(analytic.channels[species].raw_number) == Approx(
                static_cast<double>(quadrature.channels[species].raw_number)).epsilon(4e-11));
            CHECK(std::abs(old[0] / quadrature.channels[species].raw_mass - 1) > .1L);
            CHECK(std::abs(old[1] / quadrature.channels[species].raw_number - 1) > .1L);
        }
        CHECK(reference::cloud_moment(analytic.cloud_inverse_lambda, 3)
              == Approx(cell.initial.cloud_mass_per_kg
                        / (static_cast<double>(reference::cloud_mass_coefficient)
                           * cell.initial.cloud_number_per_kg)).epsilon(3e-14));
    }
}

TEST_CASE("triple cold cloud collection retains snow area and all operational efficiency branches",
          "[physics][milbrandt][triple_cold_cloud_collection]")
{
    bool below_stokes = false;
    bool interior_stokes = false;
    bool above_stokes = false;
    for (long double cloud_diameter : {2e-6L, 20e-6L, 50e-6L, 300e-6L})
    {
        for (long double snow_diameter : {.1e-3L, 1e-3L, 20e-3L})
        {
            for (long double graupel_diameter : {.5e-3L, 20e-3L})
            {
                auto cell = physical_cell();
                cell.initial.cloud_number_per_kg = static_cast<double>(
                    cell.initial.cloud_mass_per_kg
                    / (reference::cloud_mass_coefficient * std::pow(cloud_diameter, 3)));
                cell.initial.frozen[0] = reference::population(0, 5e-4,
                    static_cast<double>(5e-4L / (reference::frozen::mass_coefficient[1]
                        * std::pow(snow_diameter, reference::frozen::mass_exponent[1]))), 2);
                cell.initial.frozen[1] = reference::population(1, 3e-4,
                    static_cast<double>(3e-4L / (reference::frozen::mass_coefficient[2]
                        * std::pow(graupel_diameter, 3))), 2);
                constexpr double duration = 1e-6;
                const auto expected = reference::evaluate(cell, duration);
                const auto stokes = reference::graupel_stokes_number(cell, expected.spectra[1]);
                below_stokes = below_stokes || stokes < 1.5L;
                interior_stokes = interior_stokes || (stokes > 1.5L && stokes < 10);
                above_stokes = above_stokes || stokes > 10;
                Result result;
                std::string error;
                REQUIRE(evaluate_exact_triple_cold_cloud_collection_cell(
                    parameters(), cell, duration, result, error));
                check_capture(cell, duration, result.producer, expected);
                check_endpoint(cell, result, expected);
                CHECK(result.producer.efficiencies()[0] == Approx(static_cast<double>(
                    std::min(reference::cloud_mean_mass_diameter(cell), 30e-6L) * 3.333e4L
                    * std::sqrt(std::min(reference::collector_mean_mass_diameter(
                        cell.initial.frozen[0], 0), 1e-3L) * 1e3L))).epsilon(3e-12));
                CHECK(result.producer.efficiencies()[2] == Approx(static_cast<double>(std::exp(
                    -8.68e-7L * std::pow(reference::cloud_mean_mass_diameter(cell), -1.6L)
                    * reference::collector_mean_mass_diameter(cell.initial.frozen[2], 2)))).epsilon(3e-12));
                const auto without_area = expected.channels[0].raw_mass / reference::snow_projected_area_factor;
                CHECK(result.producer.channels()[0].raw_mass_increment_per_kg
                      / static_cast<double>(without_area) == Approx(.6366).epsilon(3e-10));
            }
        }
    }
    CHECK(below_stokes);
    CHECK(interior_stokes);
    CHECK(above_stokes);
    auto cell = physical_cell();
    Producer thin;
    Producer dense;
    std::string error;
    cell.dry_density_kg_m3 = .4;
    REQUIRE(produce_exact_cold_cloud_collection_admission(parameters(), cell, 1e-4, thin, error));
    cell.dry_density_kg_m3 = 1.2;
    REQUIRE(produce_exact_cold_cloud_collection_admission(parameters(), cell, 1e-4, dense, error));
    for (std::size_t species : {0u, 2u})
    {
        CHECK(dense.channels()[species].raw_mass_increment_per_kg
              / thin.channels()[species].raw_mass_increment_per_kg == Approx(std::sqrt(3.0)).epsilon(3e-12));
        CHECK(dense.channels()[species].raw_number_increment_per_kg
              / thin.channels()[species].raw_number_increment_per_kg == Approx(std::sqrt(3.0)).epsilon(3e-12));
    }
}

TEST_CASE("triple cold cloud collection jointly admits cloud mass and number before atomic heat rejection",
          "[physics][milbrandt][triple_cold_cloud_collection]")
{
    auto cell = physical_cell(2, .65, 273.149);
    const auto unit = reference::evaluate(cell, 1);
    long double raw_mass = 0;
    long double raw_number = 0;
    for (const auto& channel : unit.channels)
    {
        raw_mass += channel.raw_mass;
        raw_number += channel.raw_number;
    }
    const double duration = static_cast<double>(2 * std::min(
        static_cast<long double>(cell.initial.cloud_mass_per_kg) / raw_mass,
        static_cast<long double>(cell.initial.cloud_number_per_kg) / raw_number));
    const auto expected = reference::evaluate(cell, duration);
    REQUIRE(expected.channels[0].factor < 1);
    REQUIRE(cell.temperature_k + expected.temperature_increment >= 273.15);
    Producer producer;
    std::string error;
    Failure failure = Failure::InvalidProducer;
    REQUIRE(produce_exact_cold_cloud_collection_admission(
        parameters(), cell, duration, producer, error, 3, &failure));
    CHECK(failure == Failure::None);
    check_capture(cell, duration, producer, expected);
    long double admitted_mass = 0;
    long double admitted_number = 0;
    for (std::size_t species = 0; species < 3; ++species)
    {
        const auto& channel = producer.channels()[species];
        CHECK(channel.donor_admission_factor == producer.channels()[0].donor_admission_factor);
        admitted_mass += channel.admitted_mass_increment_per_kg;
        admitted_number += channel.admitted_number_increment_per_kg;
    }
    CHECK(admitted_mass <= cell.initial.cloud_mass_per_kg * (1 + 4e-15L));
    CHECK(admitted_number <= cell.initial.cloud_number_per_kg * (1 + 4e-15L));
    CHECK(std::max(admitted_mass / cell.initial.cloud_mass_per_kg,
                   admitted_number / cell.initial.cloud_number_per_kg) == Approx(1).epsilon(3e-14));
    const auto sentinel = successful_sentinel();
    Result result = sentinel;
    CHECK_FALSE(apply_exact_triple_cold_cloud_collection_cell(
        parameters(), cell, duration, producer, result, error, 3, &failure));
    CHECK(failure == Failure::IntervalRejected);
    CHECK_FALSE(error.empty());
    same_result(result, sentinel);
    result = sentinel;
    CHECK_FALSE(evaluate_exact_triple_cold_cloud_collection_cell(
        parameters(), cell, duration, result, error, 3, &failure));
    CHECK(failure == Failure::IntervalRejected);
    same_result(result, sentinel);
}

TEST_CASE("triple cold cloud collection invalid sources contexts and capacity preserve every output",
          "[physics][milbrandt][triple_cold_cloud_collection]")
{
    const auto original = physical_cell();
    const auto p = parameters();
    const auto sentinel = successful_sentinel();
    const auto reject = [&](const ColdParameters& supplied, const Cell& cell,
                            double duration, std::size_t capacity, Failure expected)
    {
        Result result = sentinel;
        Producer producer = sentinel.producer;
        std::string error;
        Failure failure = Failure::None;
        CHECK_FALSE(evaluate_exact_triple_cold_cloud_collection_cell(
            supplied, cell, duration, result, error, capacity, &failure));
        CHECK(failure == expected);
        CHECK_FALSE(error.empty());
        same_result(result, sentinel);
        CHECK_FALSE(produce_exact_cold_cloud_collection_admission(
            supplied, cell, duration, producer, error, capacity, &failure));
        CHECK(failure == expected);
        same_producer(producer, sentinel.producer);
    };
    for (std::size_t capacity : {0u, 1u, 2u, 4u, std::numeric_limits<unsigned>::max()})
    {
        reject(p, original, 1e-4, capacity, Failure::Capacity);
    }
    for (double duration : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()})
    {
        reject(p, original, duration, 3, Failure::InvalidContext);
    }
    for (double temperature : {149.999, 273.15, 400.0, std::numeric_limits<double>::quiet_NaN()})
    {
        auto cell = original;
        cell.temperature_k = temperature;
        reject(p, cell, 1e-4, 3, Failure::InvalidContext);
    }
    for (double pressure : {99.999, 200001.0, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()})
    {
        auto cell = original;
        cell.pressure_pa = pressure;
        reject(p, cell, 1e-4, 3, Failure::InvalidContext);
    }
    for (double density : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()})
    {
        auto cell = original;
        cell.dry_density_kg_m3 = density;
        reject(p, cell, 1e-4, 3, Failure::InvalidContext);
    }
    for (auto member : parameter_members())
    {
        auto supplied = p;
        supplied.*member = std::nextafter(supplied.*member, std::numeric_limits<double>::infinity());
        reject(supplied, original, 1e-4, 3, Failure::InvalidContext);
    }
    for (double ColdCloudCollectionState::*member : {
            &ColdCloudCollectionState::cloud_mass_per_kg, &ColdCloudCollectionState::cloud_number_per_kg})
    {
        for (double value : {-1.0, std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()})
        {
            auto cell = original;
            cell.initial.*member = value;
            reject(p, cell, 1e-4, 3, Failure::InvalidSource);
        }
    }
    auto absent_cloud_number = original;
    absent_cloud_number.initial.cloud_number_per_kg = 0;
    reject(p, absent_cloud_number, 1e-4, 3, Failure::InvalidSource);
    for (std::size_t species = 0; species < 3; ++species)
    {
        for (double Population::*member : {&Population::mass_per_kg,
                &Population::number_per_kg, &Population::reflectivity_per_kg})
        {
            for (double value : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::quiet_NaN()})
            {
                auto cell = original;
                cell.initial.frozen[species].*member = value;
                reject(p, cell, 1e-4, 3, Failure::InvalidSource);
            }
        }
        auto above_shape = original;
        above_shape.initial.frozen[species] = reference::population(species,
            original.initial.frozen[species].mass_per_kg,
            original.initial.frozen[species].number_per_kg, 201);
        reject(p, above_shape, 1e-4, 3, Failure::InvalidSource);
        auto cauchy = original;
        const auto& population = cauchy.initial.frozen[species];
        cauchy.initial.frozen[species].reflectivity_per_kg = static_cast<double>(
            population.number_per_kg * std::pow(population.mass_per_kg
            / (reference::frozen::mass_coefficient[species + 1] * population.number_per_kg),
            6 / reference::frozen::mass_exponent[species + 1]) * .99L);
        reject(p, cauchy, 1e-4, 3, Failure::InvalidSource);
    }
}

TEST_CASE("triple cold cloud immutable captures bind all source bits and policy independently of endpoints",
          "[physics][milbrandt][triple_cold_cloud_collection]")
{
    const auto p = parameters();
    const auto cell = physical_cell();
    constexpr double duration = 1e-4;
    const auto sentinel = successful_sentinel();
    Result admitted;
    std::string error;
    REQUIRE(evaluate_exact_triple_cold_cloud_collection_cell(p, cell, duration, admitted, error));
    const auto reject = [&](const ColdParameters& supplied, const Cell& source,
                            double dt, const Producer& producer)
    {
        Result result = sentinel;
        Failure failure = Failure::None;
        CHECK_FALSE(apply_exact_triple_cold_cloud_collection_cell(
            supplied, source, dt, producer, result, error, 3, &failure));
        CHECK(failure == Failure::InvalidProducer);
        CHECK_FALSE(error.empty());
        same_result(result, sentinel);
    };
    reject(p, cell, duration, Producer{});
    reject(p, cell, std::nextafter(duration, 1.0), admitted.producer);
    for (double Cell::*member : {&Cell::temperature_k, &Cell::pressure_pa, &Cell::dry_density_kg_m3})
    {
        auto changed = cell;
        changed.*member = std::nextafter(changed.*member, std::numeric_limits<double>::infinity());
        reject(p, changed, duration, admitted.producer);
    }
    for (double ColdCloudCollectionState::*member : {
            &ColdCloudCollectionState::cloud_mass_per_kg, &ColdCloudCollectionState::cloud_number_per_kg})
    {
        auto changed = cell;
        changed.initial.*member = std::nextafter(changed.initial.*member, std::numeric_limits<double>::infinity());
        reject(p, changed, duration, admitted.producer);
    }
    for (std::size_t species = 0; species < 3; ++species)
    {
        for (double Population::*member : {&Population::mass_per_kg,
                &Population::number_per_kg, &Population::reflectivity_per_kg})
        {
            auto changed = cell;
            changed.initial.frozen[species].*member = std::nextafter(
                changed.initial.frozen[species].*member, std::numeric_limits<double>::infinity());
            reject(p, changed, duration, admitted.producer);
        }
    }
    for (auto member : parameter_members())
    {
        auto changed = p;
        changed.*member = std::nextafter(changed.*member, std::numeric_limits<double>::infinity());
        reject(changed, cell, duration, admitted.producer);
    }
    reject(parameters(false), cell, duration, admitted.producer);
    Result existing_hail;
    REQUIRE(evaluate_exact_triple_cold_cloud_collection_cell(parameters(false), cell, duration, existing_hail, error));
    CHECK_FALSE(existing_hail.producer.hail_processes());
    CHECK(existing_hail.mass_increment_per_kg == admitted.mass_increment_per_kg);
    CHECK(existing_hail.reflectivity_increment_per_kg == admitted.reflectivity_increment_per_kg);
    CHECK(existing_hail.cloud_number_increment_per_kg == admitted.cloud_number_increment_per_kg);
    CHECK(existing_hail.temperature_increment_k == admitted.temperature_increment_k);
    reject(p, cell, duration, existing_hail.producer);
    auto detached = admitted;
    detached.unrounded.cloud_mass_per_kg = -1;
    detached.rounded_cloud_number_per_kg = -2;
    detached.unrounded.frozen[0].mass_per_kg = -3;
    detached.rounded[1].reflectivity_per_kg = -4;
    detached.mass_increment_per_kg[2] = -5;
    detached.temperature_increment_k = 6;
    REQUIRE(apply_exact_triple_cold_cloud_collection_cell(
        p, cell, duration, detached.producer, detached, error));
    same_result(detached, admitted);
    Cell empty{253.15, 90000, .65, {}};
    Producer zero;
    REQUIRE(produce_exact_cold_cloud_collection_admission(p, empty, duration, zero, error));
    empty.initial.cloud_mass_per_kg = -0.0;
    reject(p, empty, duration, zero);
}

TEST_CASE("triple cold cloud zero populations cutoffs and FP32 realization preserve source ownership",
          "[physics][milbrandt][triple_cold_cloud_collection]")
{
    const auto p = parameters();
    std::string error;
    for (double temperature : {150.0, 253.15, std::nextafter(273.15, 0.0)})
    {
        for (double pressure : {100.0, 90000.0, 200000.0})
        {
            for (double retained_cloud_number : {0.0, 1e8})
            {
                Cell cell{temperature, pressure, .65, {0, retained_cloud_number, {}}};
                Result result;
                REQUIRE(evaluate_exact_triple_cold_cloud_collection_cell(p, cell, 1, result, error));
                check_capture(cell, 1, result.producer, reference::evaluate(cell, 1));
                CHECK(result.unrounded.cloud_mass_per_kg == 0);
                CHECK(result.unrounded.cloud_number_per_kg == retained_cloud_number);
                CHECK(result.temperature_increment_k == 0);
                CHECK(result.cloud_number_increment_per_kg == 0);
                for (std::size_t species = 0; species < 3; ++species)
                {
                    same_population(result.unrounded.frozen[species], {});
                }
            }
        }
    }
    auto threshold = physical_cell();
    threshold.initial.cloud_mass_per_kg = 1e-14;
    threshold.initial.cloud_number_per_kg = static_cast<double>(1e-14L
        / (reference::cloud_mass_coefficient * std::pow(20e-6L, 3)));
    Result result;
    REQUIRE(evaluate_exact_triple_cold_cloud_collection_cell(p, threshold, 1e-4, result, error));
    CHECK(result.producer.cloud_inverse_lambda_m() > 0);
    for (const auto& channel : result.producer.channels())
    {
        CHECK(channel.raw_mass_increment_per_kg == 0);
        CHECK(channel.raw_number_increment_per_kg == 0);
    }
    threshold.initial.cloud_mass_per_kg = std::nextafter(1e-14, 1.0);
    Producer above;
    REQUIRE(produce_exact_cold_cloud_collection_admission(p, threshold, 1e-4, above, error));
    for (const auto& channel : above.channels())
    {
        CHECK(channel.raw_mass_increment_per_kg > 0);
        CHECK(channel.raw_number_increment_per_kg > 0);
    }
    const auto sentinel = successful_sentinel();
    Cell tiny{253.15, 90000, .65, {}};
    tiny.initial.frozen[1] = reference::population(1, 1e-40, 1e-3, 2);
    REQUIRE(tiny.initial.frozen[1].reflectivity_per_kg > 0);
    REQUIRE(static_cast<float>(tiny.initial.frozen[1].reflectivity_per_kg) == 0);
    Producer producer;
    Failure failure = Failure::None;
    REQUIRE(produce_exact_cold_cloud_collection_admission(p, tiny, 1e-4, producer, error, 3, &failure));
    result = sentinel;
    CHECK_FALSE(apply_exact_triple_cold_cloud_collection_cell(p, tiny, 1e-4, producer, result, error, 3, &failure));
    CHECK(failure == Failure::IntervalRejected);
    same_result(result, sentinel);
    auto original = physical_cell(20);
    auto restored = original;
    const auto roundtrip = [](double& value)
    {
        std::ostringstream encoded;
        encoded.imbue(std::locale::classic());
        encoded << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
        std::istringstream decoded(encoded.str());
        decoded.imbue(std::locale::classic());
        decoded >> value;
        REQUIRE_FALSE(decoded.fail());
    };
    roundtrip(restored.initial.cloud_mass_per_kg);
    roundtrip(restored.initial.cloud_number_per_kg);
    for (auto& population : restored.initial.frozen)
    {
        roundtrip(population.mass_per_kg);
        roundtrip(population.number_per_kg);
        roundtrip(population.reflectivity_per_kg);
    }
    Result original_result;
    Result restored_result;
    REQUIRE(evaluate_exact_triple_cold_cloud_collection_cell(p, original, 1e-4, original_result, error));
    REQUIRE(evaluate_exact_triple_cold_cloud_collection_cell(p, restored, 1e-4, restored_result, error));
    same_result(restored_result, original_result);
}

TEST_CASE("triple cold cloud collection exact one and four worker continuations agree",
          "[physics][milbrandt][triple_cold_cloud_collection]")
{
    constexpr int columns = 32;
    const auto p = parameters();
    std::array<Cell, columns> sources{};
    std::array<Result, columns> serial{};
    std::array<Result, columns> parallel{};
    std::array<bool, columns> passed{};
    std::array<std::string, columns> errors{};
    int observed_workers = 1;
    for (int column = 0; column < columns; ++column)
    {
        sources[column] = physical_cell(column % 4 * 3, .4 + .025 * column, 233.15 + column);
        REQUIRE(evaluate_exact_triple_cold_cloud_collection_cell(
            p, sources[column], .01, serial[column], errors[column]));
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
        passed[column] = evaluate_exact_triple_cold_cloud_collection_cell(
            p, sources[column], .01, parallel[column], errors[column]);
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
        same_result(parallel[column], serial[column]);
        sources[column].initial = serial[column].unrounded;
        sources[column].temperature_k += serial[column].temperature_increment_k;
        REQUIRE(evaluate_exact_triple_cold_cloud_collection_cell(
            p, sources[column], .01, serial[column], errors[column]));
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(4)
#endif
    for (int column = 0; column < columns; ++column)
    {
        passed[column] = evaluate_exact_triple_cold_cloud_collection_cell(
            p, sources[column], .01, parallel[column], errors[column]);
    }
    for (int column = 0; column < columns; ++column)
    {
        REQUIRE(passed[column]);
        same_result(parallel[column], serial[column]);
        check_endpoint(sources[column], parallel[column], reference::evaluate(sources[column], .01));
    }
}
