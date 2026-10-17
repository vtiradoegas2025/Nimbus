#include "catch2/catch.hpp"
#include "microphysics/schemes/milbrandt/triple_sedimentation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace
{

using tmv::milbrandt_detail::TripleMomentSedimentationParameters;
using tmv::milbrandt_detail::TripleMomentSpecies;
using tmv::milbrandt_detail::TripleMomentSurfaceInventory;

constexpr std::array<TripleMomentSpecies, 5> kSpecies{
    TripleMomentSpecies::Rain, TripleMomentSpecies::Ice,
    TripleMomentSpecies::Snow, TripleMomentSpecies::Graupel,
    TripleMomentSpecies::Hail};
constexpr std::array<double, 5> kMass{
    1.0e-4, 8.0e-5, 5.0e-5, 7.0e-5, 9.0e-5};
constexpr std::array<double, 5> kNumber{
    1.0e6, 2.0e5, 8.0e4, 3.0e4, 1.0e4};

long double gamma_integral(long double order, int intervals = 8000)
{
    constexpr long double upper = 80.0L;
    const long double transformed_upper = std::pow(upper, 0.25L);
    const long double step = transformed_upper / intervals;
    long double sum = 0.0L;
    for (int index = 0; index <= intervals; ++index)
    {
        const long double u = step * index;
        const long double x = u * u * u * u;
        const int weight = index == 0 || index == intervals ? 1 : index % 2 == 0 ? 2 : 4;
        const long double transformed =
            4.0L * std::pow(u, 4.0L * (order + 1.0L) - 1.0L) *
            std::exp(-x);
        sum += weight * transformed;
    }
    return sum * step / 3.0L;
}

double reference_lambda(double mass, double number, double alpha,
                        const TripleMomentSedimentationParameters& parameters)
{
    const auto law = parameters.mass_law;
    return std::pow(
        law.coefficient * number * std::tgamma(alpha + law.exponent + 1.0) /
            (mass * std::tgamma(alpha + 1.0)),
        1.0 / law.exponent);
}

double reference_moment_per_kg(double number, double alpha,
                                double lambda, double order)
{
    return number * static_cast<double>(gamma_integral(alpha + order) /
        gamma_integral(alpha)) / std::pow(lambda, order);
}

std::array<double, 3> reference_speeds(
    double density, double alpha, double lambda,
    const TripleMomentSedimentationParameters& parameters,
    int intervals = 8000)
{
    const std::array<double, 3> order{0.0, parameters.mass_law.exponent, 6.0};
    std::array<double, 3> result{};
    for (std::size_t moment = 0; moment < result.size(); ++moment)
    {
        result[moment] = std::min(parameters.maximum_speed,
            parameters.fall_coefficient * std::sqrt(1.225 / density) *
            std::pow(lambda, -parameters.fall_exponent) *
            static_cast<double>(gamma_integral(alpha + order[moment] +
                                                parameters.fall_exponent, intervals) /
                                 gamma_integral(alpha + order[moment], intervals)));
    }
    return result;
}

double relative_error(double actual, double expected)
{
    return std::abs(actual - expected) / std::max(std::abs(expected), 1.0e-30);
}

float initial_z(double mass, double number, double alpha,
                const TripleMomentSedimentationParameters& parameters)
{
    return static_cast<float>(reference_moment_per_kg(
        number, alpha, reference_lambda(mass, number, alpha, parameters), 6.0));
}

} // namespace

TEST_CASE("all five Nimbus q/N/Z laws invert alpha and match independent moment quadrature",
          "[physics][milbrandt][triple_five]")
{
    constexpr std::array<std::array<double, 6>, 5> expected{{
        {{523.598775598, 3.0, 149.1, 0.5, 16.0, 5.0e-3}},
        {{440.0, 3.0, 71.34, 0.6635, 2.0, 5.0e-3}},
        {{0.1597, 2.078, 11.72, 0.41, 4.0, 5.0e-3}},
        {{209.439510239, 3.0, 19.3, 0.37, 6.0, 2.0e-3}},
        {{471.238898038, 3.0, 206.89, 0.6384, 25.0, 80.0e-3}}
    }};
    for (std::size_t species = 0; species < kSpecies.size(); ++species)
    {
        const auto parameters =
            tmv::milbrandt_detail::current_nimbus_triple_sedimentation_parameters(kSpecies[species]);
        CHECK(parameters.mass_law.coefficient == expected[species][0]);
        CHECK(parameters.mass_law.exponent == expected[species][1]);
        CHECK(parameters.fall_coefficient == expected[species][2]);
        CHECK(parameters.fall_exponent == expected[species][3]);
        CHECK(parameters.maximum_speed == expected[species][4]);
        CHECK(parameters.maximum_diameter == expected[species][5]);
        for (double alpha : {0.0, 2.0, 5.0})
        {
            for (double density : {0.65, 1.2, 1.8})
            {
                const double lambda = reference_lambda(
                    kMass[species], kNumber[species], alpha, parameters);
                const double m0_volume = density * reference_moment_per_kg(
                    kNumber[species], alpha, lambda, 0.0);
                const double md_volume = density * reference_moment_per_kg(
                    kNumber[species], alpha, lambda,
                    parameters.mass_law.exponent);
                const float z = initial_z(
                    kMass[species], kNumber[species], alpha, parameters);
                CHECK(relative_error(m0_volume / density, kNumber[species]) < 5.0e-5);
                CHECK(relative_error(parameters.mass_law.coefficient * md_volume / density,
                                     kMass[species]) < 5.0e-5);
                double recovered_alpha = -10.0;
                double recovered_lambda = -1.0;
                std::string error;
                REQUIRE(tmv::milbrandt_detail::diagnose_triple_gamma_shape(
                    kMass[species], kNumber[species], z,
                    parameters.mass_law, recovered_alpha, recovered_lambda,
                    error));
                CHECK(std::abs(recovered_alpha - alpha) < 1.0e-3);
                CHECK(relative_error(recovered_lambda, lambda) < 1.0e-4);
                std::array<double, 3> speeds{};
                REQUIRE(tmv::milbrandt_detail::triple_moment_fall_speeds(
                    density, recovered_alpha, recovered_lambda, parameters,
                    speeds));
                const auto reference = reference_speeds(
                    density, alpha, lambda, parameters);
                if (alpha == 0.0)
                {
                    const auto coarse = reference_speeds(
                        density, alpha, lambda, parameters, 2000);
                    const auto middle = reference_speeds(
                        density, alpha, lambda, parameters, 4000);
                    for (std::size_t moment = 0; moment < 3; ++moment)
                    {
                        CHECK(relative_error(middle[moment], coarse[moment]) < 1.0e-6);
                        CHECK(relative_error(reference[moment], middle[moment]) < 1.0e-7);
                    }
                }
                for (std::size_t moment = 0; moment < 3; ++moment)
                {
                    CHECK(relative_error(speeds[moment], reference[moment]) < 1.0e-4);
                }
            }
        }
    }
    REQUIRE_THROWS_AS(
        tmv::milbrandt_detail::current_nimbus_triple_sedimentation_parameters(
            static_cast<TripleMomentSpecies>(5)), std::invalid_argument);
}

TEST_CASE("all five q/N/Z one-step column fluxes close variable-rho/dz surface inventory",
          "[physics][milbrandt][triple_five]")
{
    constexpr std::array<float, 2> density{1.4f, 0.8f};
    constexpr std::array<double, 2> thickness{20.0, 15.0};
    constexpr double dt = 0.001;
    for (std::size_t species = 0; species < kSpecies.size(); ++species)
    {
        const auto parameters =
            tmv::milbrandt_detail::current_nimbus_triple_sedimentation_parameters(kSpecies[species]);
        const std::array<float, 2> mass{
            static_cast<float>(kMass[species]), static_cast<float>(0.8 * kMass[species])};
        const std::array<float, 2> number{
            static_cast<float>(kNumber[species]), static_cast<float>(0.8 * kNumber[species])};
        const std::array<float, 2> z{
            initial_z(mass[0], number[0], 2.0, parameters),
            initial_z(mass[1], number[1], 2.0, parameters)};
        std::array<float, 2> q_after{-7.0f, -7.0f};
        std::array<float, 2> n_after{-7.0f, -7.0f};
        std::array<float, 2> z_after{-7.0f, -7.0f};
        TripleMomentSurfaceInventory surface{-7.0, -7.0, -7.0};
        std::string error;
        REQUIRE(tmv::milbrandt_detail::sediment_triple_moment_column(
            2, mass.data(), number.data(), z.data(), density.data(),
            thickness.data(), dt, parameters, q_after.data(), n_after.data(),
            z_after.data(), surface, error));
        std::array<std::array<double, 3>, 2> speed{};
        for (std::size_t level = 0; level < 2; ++level)
        {
            speed[level] = reference_speeds(
                density[level], 2.0,
                reference_lambda(mass[level], number[level], 2.0, parameters),
                parameters);
        }
        const std::array<const float*, 3> input{mass.data(), number.data(), z.data()};
        const std::array<const float*, 3> output{q_after.data(), n_after.data(), z_after.data()};
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
            const double expected_surface =
                input[moment][0] * density[0] * speed[0][moment] * dt;
            CHECK(relative_error(output[moment][0], expected_bottom) < 1.0e-4);
            CHECK(relative_error(output[moment][1], expected_top) < 1.0e-4);
            CHECK(relative_error(loss[moment], expected_surface) < 1.0e-4);
            const double before = input[moment][0] * density[0] * thickness[0] +
                input[moment][1] * density[1] * thickness[1];
            const double after = output[moment][0] * density[0] * thickness[0] +
                output[moment][1] * density[1] * thickness[1] + loss[moment];
            CHECK(relative_error(after, before) < 5.0e-6);
        }
    }
}

TEST_CASE("all five high-Courant columns conserve q/N/Z and reject late invalid input atomically",
          "[physics][milbrandt][triple_five]")
{
    constexpr int levels = 8;
    for (std::size_t species = 0; species < kSpecies.size(); ++species)
    {
        const auto parameters =
            tmv::milbrandt_detail::current_nimbus_triple_sedimentation_parameters(kSpecies[species]);
        std::array<float, levels> mass{};
        std::array<float, levels> number{};
        std::array<float, levels> z{};
        std::array<float, levels> density{};
        std::array<double, levels> thickness{};
        for (int level = 0; level < levels; ++level)
        {
            mass[level] = static_cast<float>(kMass[species]);
            number[level] = static_cast<float>(kNumber[species]);
            density[level] = static_cast<float>(0.8 + 0.05 * level);
            thickness[level] = 1.0 + 0.1 * level;
            z[level] = initial_z(mass[level], number[level], 2.0, parameters);
        }
        const auto first_speed = reference_speeds(
            density[0], 2.0,
            reference_lambda(mass[0], number[0], 2.0, parameters),
            parameters);
        const double dt = 8.0 * thickness[0] /
            *std::max_element(first_speed.begin(), first_speed.end());
        std::array<float, levels> q_after{};
        std::array<float, levels> n_after{};
        std::array<float, levels> z_after{};
        TripleMomentSurfaceInventory surface{};
        std::string error;
        REQUIRE(tmv::milbrandt_detail::sediment_triple_moment_column(
            levels, mass.data(), number.data(), z.data(), density.data(),
            thickness.data(), dt, parameters, q_after.data(), n_after.data(),
            z_after.data(), surface, error));
        const std::array<const float*, 3> input{mass.data(), number.data(), z.data()};
        const std::array<const float*, 3> output{q_after.data(), n_after.data(), z_after.data()};
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

        q_after.fill(-7.0f);
        n_after.fill(-7.0f);
        z_after.fill(-7.0f);
        surface = {-7.0, -7.0, -7.0};
        const auto check_unpublished = [&]()
        {
            for (int level = 0; level < levels; ++level)
            {
                CHECK(q_after[level] == -7.0f);
                CHECK(n_after[level] == -7.0f);
                CHECK(z_after[level] == -7.0f);
            }
            CHECK(surface.mass == -7.0);
            CHECK(surface.number == -7.0);
            CHECK(surface.reflectivity == -7.0);
        };
        z[levels - 1] = std::numeric_limits<float>::quiet_NaN();
        CHECK_FALSE(tmv::milbrandt_detail::sediment_triple_moment_column(
            levels, mass.data(), number.data(), z.data(), density.data(),
            thickness.data(), dt, parameters, q_after.data(), n_after.data(),
            z_after.data(), surface, error));
        CHECK(q_after[0] == -7.0f);
        CHECK(n_after[0] == -7.0f);
        CHECK(z_after[0] == -7.0f);
        CHECK(surface.mass == -7.0);
        CHECK(surface.number == -7.0);
        CHECK(surface.reflectivity == -7.0);
        check_unpublished();

        z[levels - 1] = initial_z(
            mass[levels - 1], number[levels - 1], 2.0, parameters);
        number[levels - 1] = 0.0f;
        CHECK_FALSE(tmv::milbrandt_detail::sediment_triple_moment_column(
            levels, mass.data(), number.data(), z.data(), density.data(),
            thickness.data(), dt, parameters, q_after.data(), n_after.data(),
            z_after.data(), surface, error));
        CHECK(q_after[0] == -7.0f);
        CHECK(surface.mass == -7.0);
        check_unpublished();
        number[levels - 1] = static_cast<float>(kNumber[species]);

        z[levels - 1] = initial_z(
            mass[levels - 1], number[levels - 1], 2.0, parameters);
        const double lower = number[levels - 1] * std::pow(
            mass[levels - 1] /
                (parameters.mass_law.coefficient * number[levels - 1]),
            6.0 / parameters.mass_law.exponent);
        z[levels - 1] = static_cast<float>(0.5 * lower);
        CHECK_FALSE(tmv::milbrandt_detail::sediment_triple_moment_column(
            levels, mass.data(), number.data(), z.data(), density.data(),
            thickness.data(), dt, parameters, q_after.data(), n_after.data(),
            z_after.data(), surface, error));
        CHECK(error.find("realizability") != std::string::npos);
        CHECK(q_after[0] == -7.0f);
        CHECK(surface.mass == -7.0);
        check_unpublished();

        z[levels - 1] = std::numeric_limits<float>::max();
        CHECK_FALSE(tmv::milbrandt_detail::sediment_triple_moment_column(
            levels, mass.data(), number.data(), z.data(), density.data(),
            thickness.data(), dt, parameters, q_after.data(), n_after.data(),
            z_after.data(), surface, error));
        CHECK(error.find("inversion bracket") != std::string::npos);
        check_unpublished();

        z[levels - 1] = initial_z(
            mass[levels - 1], number[levels - 1], 2.0, parameters);
        auto capped = parameters;
        capped.maximum_substeps = 1;
        CHECK_FALSE(tmv::milbrandt_detail::sediment_triple_moment_column(
            levels, mass.data(), number.data(), z.data(), density.data(),
            thickness.data(), dt, capped, q_after.data(), n_after.data(),
            z_after.data(), surface, error));
        CHECK(q_after[0] == -7.0f);
        CHECK(surface.mass == -7.0);
        check_unpublished();

        auto excessive_speed = parameters;
        excessive_speed.fall_coefficient = std::numeric_limits<double>::max();
        density[levels - 1] = std::numeric_limits<float>::denorm_min();
        CHECK_FALSE(tmv::milbrandt_detail::sediment_triple_moment_column(
            levels, mass.data(), number.data(), z.data(), density.data(),
            thickness.data(), dt, excessive_speed,
            q_after.data(), n_after.data(), z_after.data(), surface, error));
        CHECK(q_after[0] == -7.0f);
        CHECK(surface.mass == -7.0);
        check_unpublished();

        constexpr int over_budget_levels = 17;
        std::array<float, over_budget_levels> over_budget_mass{};
        std::array<float, over_budget_levels> over_budget_number{};
        std::array<float, over_budget_levels> over_budget_z{};
        std::array<float, over_budget_levels> over_budget_density{};
        std::array<double, over_budget_levels> over_budget_thickness{};
        std::array<float, over_budget_levels> over_budget_q_after{};
        std::array<float, over_budget_levels> over_budget_n_after{};
        std::array<float, over_budget_levels> over_budget_z_after{};
        over_budget_q_after.fill(-7.0f);
        over_budget_n_after.fill(-7.0f);
        over_budget_z_after.fill(-7.0f);
        CHECK_FALSE(tmv::milbrandt_detail::sediment_triple_moment_column(
            over_budget_levels, over_budget_mass.data(),
            over_budget_number.data(), over_budget_z.data(),
            over_budget_density.data(), over_budget_thickness.data(), dt,
            parameters, over_budget_q_after.data(),
            over_budget_n_after.data(), over_budget_z_after.data(),
            surface, error));
        CHECK(error.find("65536 cell-substep resource budget") != std::string::npos);
        for (int level = 0; level < over_budget_levels; ++level)
        {
            CHECK(over_budget_q_after[level] == -7.0f);
            CHECK(over_budget_n_after[level] == -7.0f);
            CHECK(over_budget_z_after[level] == -7.0f);
        }
        CHECK(surface.mass == -7.0);
        CHECK(surface.number == -7.0);
        CHECK(surface.reflectivity == -7.0);
    }
}

TEST_CASE("five species independent columns are byte-identical at one and four workers",
          "[physics][milbrandt][triple_five]")
{
    constexpr int columns = 8;
    constexpr int levels = 4;
    using Column = std::array<float, levels>;
    for (std::size_t species = 0; species < kSpecies.size(); ++species)
    {
        const auto parameters =
            tmv::milbrandt_detail::current_nimbus_triple_sedimentation_parameters(kSpecies[species]);
        std::array<Column, columns> mass{};
        std::array<Column, columns> number{};
        std::array<Column, columns> z{};
        std::array<Column, columns> density{};
        std::array<std::array<double, levels>, columns> thickness{};
        for (int column = 0; column < columns; ++column)
        {
            for (int level = 0; level < levels; ++level)
            {
                mass[column][level] = static_cast<float>(
                    kMass[species] * (1.0 + column * 0.001));
                number[column][level] = static_cast<float>(
                    kNumber[species] * (1.0 + column * 0.001));
                density[column][level] = static_cast<float>(0.9 + 0.02 * level);
                thickness[column][level] = 10.0 + level;
                z[column][level] = initial_z(
                    mass[column][level], number[column][level], 2.0, parameters);
            }
        }
        std::array<std::array<Column, columns>, 2> q_output{};
        std::array<std::array<Column, columns>, 2> n_output{};
        std::array<std::array<Column, columns>, 2> z_output{};
        std::array<std::array<TripleMomentSurfaceInventory, columns>, 2> surfaces{};
        std::array<std::array<bool, columns>, 2> passed{};
        for (int run = 0; run < 2; ++run)
        {
            const int workers = run == 0 ? 1 : 4;
            #pragma omp parallel for num_threads(workers)
            for (int column = 0; column < columns; ++column)
            {
                std::string error;
                passed[run][column] = tmv::milbrandt_detail::sediment_triple_moment_column(
                    levels, mass[column].data(), number[column].data(),
                    z[column].data(), density[column].data(),
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
                              sizeof(TripleMomentSurfaceInventory)) == 0);
        }
    }
}
