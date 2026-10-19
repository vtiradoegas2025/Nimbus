#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/triple_sedimentation.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace
{

using tmv::milbrandt_detail::TripleHailSedimentationParameters;
using tmv::milbrandt_detail::TripleHailSurfaceInventory;

long double integral(long double order)
{
    constexpr int intervals = 8000;
    constexpr long double upper = 80.0L;
    const long double step = upper / intervals;
    long double sum = 0.0L;
    for (int index = 0; index <= intervals; ++index)
    {
        const long double x = index * step;
        const int weight = index == 0 || index == intervals ? 1 : index % 2 == 0 ? 2 : 4;
        sum += weight * std::pow(x, order) * std::exp(-x);
    }
    return sum * step / 3.0L;
}

std::array<double, 3> quadrature_speeds(double density,
                                         double alpha,
                                         double lambda,
                                         const TripleHailSedimentationParameters& parameters)
{
    const std::array<double, 3> order{0.0, parameters.mass_law.exponent, 6.0};
    std::array<double, 3> speeds{};
    for (std::size_t moment = 0; moment < speeds.size(); ++moment)
    {
        const long double numerator = integral(alpha + order[moment] + parameters.fall_exponent);
        const long double denominator = integral(alpha + order[moment]);
        speeds[moment] = std::min(parameters.maximum_speed,
            parameters.fall_coefficient * std::sqrt(1.225 / density) *
            std::pow(lambda, -parameters.fall_exponent) *
            static_cast<double>(numerator / denominator));
    }
    return speeds;
}

float initial_reflectivity(double density, double mass, double number, double alpha)
{
    float result = 0.0f;
    REQUIRE(tmv::milbrandt_detail::gamma_sixth_moment_per_kg(
        density, mass, number, alpha, {471.238898038, 3.0}, result));
    return result;
}

double relative_error(double actual, double expected)
{
    return std::abs(actual - expected) / std::max(std::abs(expected), 1.0e-30);
}

} // namespace

TEST_CASE("hail q/N/Z gamma inversion and fall speeds match independent quadrature",
          "[physics][milbrandt][triple_sedimentation]")
{
    const TripleHailSedimentationParameters parameters;
    for (double density : {0.65, 1.2, 1.8})
    {
        for (double seed_alpha : {0.0, 2.0, 5.0})
        {
            constexpr double mass = 9.0e-5;
            constexpr double number = 1.0e4;
            const float reflectivity = initial_reflectivity(density, mass, number, seed_alpha);
            double recovered_alpha = -10.0;
            double recovered_lambda = -1.0;
            std::string error;
            REQUIRE(tmv::milbrandt_detail::diagnose_triple_gamma_shape(
                mass, number, reflectivity, parameters.mass_law,
                recovered_alpha, recovered_lambda, error));
            CHECK(std::abs(recovered_alpha - seed_alpha) < 5.0e-4);
            const double independent_lambda = std::pow(
                parameters.mass_law.coefficient * number *
                    std::tgamma(seed_alpha + parameters.mass_law.exponent + 1.0) /
                    (mass * std::tgamma(seed_alpha + 1.0)),
                1.0 / parameters.mass_law.exponent);
            CHECK(relative_error(recovered_lambda, independent_lambda) < 5.0e-5);
            std::array<double, 3> speeds{};
            REQUIRE(tmv::milbrandt_detail::triple_hail_fall_speeds(
                density, recovered_alpha, recovered_lambda, parameters, speeds));
            const auto expected = quadrature_speeds(
                density, seed_alpha, independent_lambda, parameters);
            for (std::size_t moment = 0; moment < speeds.size(); ++moment)
            {
                CHECK(relative_error(speeds[moment], expected[moment]) < 5.0e-5);
            }
            CHECK(speeds[2] > speeds[1]);
            CHECK(speeds[1] > speeds[0]);
        }
    }
}

TEST_CASE("hail q/N/Z one-step finite-volume flux closes all three inventories",
          "[physics][milbrandt][triple_sedimentation]")
{
    const TripleHailSedimentationParameters parameters;
    constexpr std::array<float, 2> density{1.4f, 0.8f};
    constexpr std::array<double, 2> thickness{20.0, 15.0};
    constexpr std::array<float, 2> mass{9.0e-5f, 7.0e-5f};
    constexpr std::array<float, 2> number{1.0e4f, 8.0e3f};
    const std::array<float, 2> reflectivity{
        initial_reflectivity(density[0], mass[0], number[0], 2.0),
        initial_reflectivity(density[1], mass[1], number[1], 2.0)};
    constexpr double dt = 0.01;
    std::array<float, 2> mass_after{-7.0f, -7.0f};
    std::array<float, 2> number_after{-7.0f, -7.0f};
    std::array<float, 2> reflectivity_after{-7.0f, -7.0f};
    TripleHailSurfaceInventory surface{-7.0, -7.0, -7.0};
    std::string error;
    REQUIRE(tmv::milbrandt_detail::sediment_triple_hail_column(
        2, mass.data(), number.data(), reflectivity.data(),
        density.data(), thickness.data(), dt, parameters,
        mass_after.data(), number_after.data(), reflectivity_after.data(),
        surface, error));

    std::array<std::array<double, 3>, 2> speed{};
    for (std::size_t level = 0; level < 2; ++level)
    {
        const double lambda = std::pow(
            parameters.mass_law.coefficient * number[level] *
                std::tgamma(2.0 + parameters.mass_law.exponent + 1.0) /
                (mass[level] * std::tgamma(3.0)),
            1.0 / parameters.mass_law.exponent);
        speed[level] = quadrature_speeds(density[level], 2.0, lambda, parameters);
    }
    const std::array<const float*, 3> input{
        mass.data(), number.data(), reflectivity.data()};
    const std::array<const float*, 3> output{
        mass_after.data(), number_after.data(), reflectivity_after.data()};
    const std::array<double, 3> loss{
        surface.mass, surface.number, surface.reflectivity};
    for (std::size_t moment = 0; moment < 3; ++moment)
    {
        const double expected_bottom = input[moment][0] *
            (1.0 - speed[0][moment] * dt / thickness[0]) +
            input[moment][1] * speed[1][moment] * dt /
                thickness[0] * density[1] / density[0];
        const double expected_top = input[moment][1] *
            (1.0 - speed[1][moment] * dt / thickness[1]);
        const double expected_surface = input[moment][0] *
            density[0] * speed[0][moment] * dt;
        CHECK(relative_error(output[moment][0], expected_bottom) < 5.0e-5);
        CHECK(relative_error(output[moment][1], expected_top) < 5.0e-5);
        CHECK(relative_error(loss[moment], expected_surface) < 5.0e-5);
        const double before_inventory = input[moment][0] * density[0] * thickness[0] +
            input[moment][1] * density[1] * thickness[1];
        const double after_inventory = output[moment][0] * density[0] * thickness[0] +
            output[moment][1] * density[1] * thickness[1] + loss[moment];
        CHECK(relative_error(after_inventory, before_inventory) < 5.0e-6);
    }
}

TEST_CASE("high-Courant hail fall remains conservative and rejects late invalid state without publication",
          "[physics][milbrandt][triple_sedimentation]")
{
    constexpr int levels = 8;
    std::array<float, levels> mass{};
    std::array<float, levels> number{};
    std::array<float, levels> reflectivity{};
    std::array<float, levels> density{};
    std::array<double, levels> thickness{};
    for (int level = 0; level < levels; ++level)
    {
        mass[level] = 9.0e-5f;
        number[level] = 1.0e4f;
        density[level] = static_cast<float>(0.8 + 0.05 * level);
        thickness[level] = 5.0 + level;
        reflectivity[level] = initial_reflectivity(
            density[level], mass[level], number[level], 2.0);
    }
    const TripleHailSedimentationParameters parameters;
    std::array<float, levels> q_after{};
    std::array<float, levels> n_after{};
    std::array<float, levels> z_after{};
    TripleHailSurfaceInventory surface{};
    std::string error;
    REQUIRE(tmv::milbrandt_detail::sediment_triple_hail_column(
        levels, mass.data(), number.data(), reflectivity.data(),
        density.data(), thickness.data(), 5.0, parameters,
        q_after.data(), n_after.data(), z_after.data(), surface, error));
    CHECK(surface.mass > 0.0);
    CHECK(surface.number > 0.0);
    CHECK(surface.reflectivity > 0.0);
    const std::array<const float*, 3> input{mass.data(), number.data(), reflectivity.data()};
    const std::array<const float*, 3> output{q_after.data(), n_after.data(), z_after.data()};
    const std::array<double, 3> loss{surface.mass, surface.number, surface.reflectivity};
    for (std::size_t moment = 0; moment < 3; ++moment)
    {
        double before = 0.0;
        double after = loss[moment];
        for (int level = 0; level < levels; ++level)
        {
            before += density[level] * thickness[level] * input[moment][level];
            after += density[level] * thickness[level] * output[moment][level];
        }
        CHECK(relative_error(after, before) < 5.0e-6);
    }

    z_after.fill(-7.0f);
    n_after.fill(-7.0f);
    q_after.fill(-7.0f);
    surface = {-7.0, -7.0, -7.0};
    reflectivity[levels - 1] = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(tmv::milbrandt_detail::sediment_triple_hail_column(
        levels, mass.data(), number.data(), reflectivity.data(),
        density.data(), thickness.data(), 5.0, parameters,
        q_after.data(), n_after.data(), z_after.data(), surface, error));
    for (int level = 0; level < levels; ++level)
    {
        CHECK(q_after[level] == -7.0f);
        CHECK(n_after[level] == -7.0f);
        CHECK(z_after[level] == -7.0f);
    }
    CHECK(surface.mass == -7.0);
    CHECK(surface.number == -7.0);
    CHECK(surface.reflectivity == -7.0);

    reflectivity[levels - 1] = initial_reflectivity(
        density[levels - 1], mass[levels - 1], number[levels - 1], 2.0);
    const double lower = number[levels - 1] * std::pow(
        mass[levels - 1] /
            (parameters.mass_law.coefficient * number[levels - 1]),
        6.0 / parameters.mass_law.exponent);
    reflectivity[levels - 1] = static_cast<float>(0.5 * lower);
    CHECK_FALSE(tmv::milbrandt_detail::sediment_triple_hail_column(
        levels, mass.data(), number.data(), reflectivity.data(),
        density.data(), thickness.data(), 5.0, parameters,
        q_after.data(), n_after.data(), z_after.data(), surface, error));
    CHECK(error.find("realizability") != std::string::npos);
    CHECK(q_after[0] == -7.0f);
    CHECK(surface.mass == -7.0);

    reflectivity[levels - 1] = std::numeric_limits<float>::max();
    CHECK_FALSE(tmv::milbrandt_detail::sediment_triple_hail_column(
        levels, mass.data(), number.data(), reflectivity.data(),
        density.data(), thickness.data(), 5.0, parameters,
        q_after.data(), n_after.data(), z_after.data(), surface, error));
    CHECK(error.find("inversion bracket") != std::string::npos);
    CHECK(q_after[0] == -7.0f);
    CHECK(surface.mass == -7.0);

    reflectivity[levels - 1] = initial_reflectivity(
        density[levels - 1], mass[levels - 1], number[levels - 1], 2.0);
    auto capped = parameters;
    capped.maximum_substeps = 1;
    CHECK_FALSE(tmv::milbrandt_detail::sediment_triple_hail_column(
        levels, mass.data(), number.data(), reflectivity.data(),
        density.data(), thickness.data(), 5.0, capped,
        q_after.data(), n_after.data(), z_after.data(), surface, error));
    CHECK(q_after[0] == -7.0f);
    CHECK(surface.mass == -7.0);

    auto excessive_speed = parameters;
    excessive_speed.fall_coefficient = std::numeric_limits<double>::max();
    density[levels - 1] = std::numeric_limits<float>::denorm_min();
    CHECK_FALSE(tmv::milbrandt_detail::sediment_triple_hail_column(
        levels, mass.data(), number.data(), reflectivity.data(),
        density.data(), thickness.data(), 5.0, excessive_speed,
        q_after.data(), n_after.data(), z_after.data(), surface, error));
    CHECK(q_after[0] == -7.0f);
    CHECK(surface.mass == -7.0);

    CHECK_FALSE(tmv::milbrandt_detail::sediment_triple_hail_column(
        levels, mass.data(), number.data(), reflectivity.data(),
        density.data(), thickness.data(), 5.0, parameters,
        mass.data(), n_after.data(), z_after.data(), surface, error));
    CHECK(mass[0] == 9.0e-5f);
    CHECK(surface.mass == -7.0);

    constexpr int over_budget_levels = 17;
    std::array<float, over_budget_levels> over_budget_mass{};
    std::array<float, over_budget_levels> over_budget_number{};
    std::array<float, over_budget_levels> over_budget_reflectivity{};
    std::array<float, over_budget_levels> over_budget_density{};
    std::array<double, over_budget_levels> over_budget_thickness{};
    std::array<float, over_budget_levels> over_budget_mass_after{};
    std::array<float, over_budget_levels> over_budget_number_after{};
    std::array<float, over_budget_levels> over_budget_reflectivity_after{};
    over_budget_mass_after.fill(-7.0f);
    CHECK_FALSE(tmv::milbrandt_detail::sediment_triple_hail_column(
        over_budget_levels, over_budget_mass.data(), over_budget_number.data(),
        over_budget_reflectivity.data(), over_budget_density.data(),
        over_budget_thickness.data(), 5.0, parameters,
        over_budget_mass_after.data(), over_budget_number_after.data(),
        over_budget_reflectivity_after.data(), surface, error));
    CHECK(error.find("65536 cell-substep resource budget") != std::string::npos);
    CHECK(over_budget_mass_after[0] == -7.0f);
    CHECK(surface.mass == -7.0);
}

TEST_CASE("accepted hail work-cap edge remains bounded and conservative",
          "[physics][milbrandt][triple_sedimentation][cap_edge]")
{
    constexpr int levels = 16;
    std::array<float, levels> mass{};
    std::array<float, levels> number{};
    std::array<float, levels> reflectivity{};
    std::array<float, levels> density{};
    std::array<double, levels> thickness{};
    std::array<float, levels> mass_after{};
    std::array<float, levels> number_after{};
    std::array<float, levels> reflectivity_after{};
    for (int level = 0; level < levels; ++level)
    {
        mass[level] = 9.0e-5f;
        number[level] = 1.0e4f;
        density[level] = static_cast<float>(0.8 + 0.02 * level);
        thickness[level] = 5.0 + 0.25 * level;
        reflectivity[level] = initial_reflectivity(
            density[level], mass[level], number[level], 2.0);
    }
    const TripleHailSedimentationParameters parameters;
    REQUIRE(levels * parameters.maximum_substeps == 65536);
    TripleHailSurfaceInventory surface{};
    std::string error;
    REQUIRE(tmv::milbrandt_detail::sediment_triple_hail_column(
        levels, mass.data(), number.data(), reflectivity.data(),
        density.data(), thickness.data(), 50.0, parameters,
        mass_after.data(), number_after.data(), reflectivity_after.data(),
        surface, error));
    const std::array<const float*, 3> input{
        mass.data(), number.data(), reflectivity.data()};
    const std::array<const float*, 3> output{
        mass_after.data(), number_after.data(), reflectivity_after.data()};
    const std::array<double, 3> loss{
        surface.mass, surface.number, surface.reflectivity};
    for (std::size_t moment = 0; moment < 3; ++moment)
    {
        double before = 0.0;
        double after = loss[moment];
        for (int level = 0; level < levels; ++level)
        {
            before += density[level] * thickness[level] * input[moment][level];
            after += density[level] * thickness[level] * output[moment][level];
        }
        CHECK(relative_error(after, before) < 5.0e-6);
    }
}

TEST_CASE("independent hail columns are byte-identical with one and four workers",
          "[physics][milbrandt][triple_sedimentation]")
{
    constexpr int columns = 16;
    constexpr int levels = 4;
    using Column = std::array<float, levels>;
    std::array<Column, columns> mass{};
    std::array<Column, columns> number{};
    std::array<Column, columns> reflectivity{};
    std::array<Column, columns> density{};
    std::array<std::array<double, levels>, columns> thickness{};
    for (int column = 0; column < columns; ++column)
    {
        for (int level = 0; level < levels; ++level)
        {
            density[column][level] = static_cast<float>(0.9 + 0.02 * level);
            thickness[column][level] = 10.0 + level;
            mass[column][level] = static_cast<float>(8.0e-5 + column * 1.0e-7);
            number[column][level] = static_cast<float>(8.0e3 + column * 10.0);
            reflectivity[column][level] = initial_reflectivity(
                density[column][level], mass[column][level],
                number[column][level], 2.0);
        }
    }
    std::array<std::array<Column, columns>, 2> q_output{};
    std::array<std::array<Column, columns>, 2> n_output{};
    std::array<std::array<Column, columns>, 2> z_output{};
    std::array<std::array<TripleHailSurfaceInventory, columns>, 2> surfaces{};
    std::array<std::array<bool, columns>, 2> passed{};
    const TripleHailSedimentationParameters parameters;
    for (int run = 0; run < 2; ++run)
    {
        const int workers = run == 0 ? 1 : 4;
        #pragma omp parallel for num_threads(workers)
        for (int column = 0; column < columns; ++column)
        {
            std::string error;
            passed[run][column] = tmv::milbrandt_detail::sediment_triple_hail_column(
                levels, mass[column].data(), number[column].data(),
                reflectivity[column].data(), density[column].data(),
                thickness[column].data(), 0.2, parameters,
                q_output[run][column].data(), n_output[run][column].data(),
                z_output[run][column].data(), surfaces[run][column], error);
        }
    }
    for (int column = 0; column < columns; ++column)
    {
        REQUIRE(passed[0][column]);
        REQUIRE(passed[1][column]);
        CHECK(std::memcmp(q_output[0][column].data(), q_output[1][column].data(),
                          sizeof(Column)) == 0);
        CHECK(std::memcmp(n_output[0][column].data(), n_output[1][column].data(),
                          sizeof(Column)) == 0);
        CHECK(std::memcmp(z_output[0][column].data(), z_output[1][column].data(),
                          sizeof(Column)) == 0);
        CHECK(std::memcmp(&surfaces[0][column], &surfaces[1][column],
                          sizeof(TripleHailSurfaceInventory)) == 0);
    }
}
