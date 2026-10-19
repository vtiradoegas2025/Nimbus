#include <catch2/catch.hpp>
#include "diagnostics/pressure_projection.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
using tmv::diagnostics::PressureProjection;
using tmv::diagnostics::PressureProjectionGrid;
using tmv::diagnostics::PressureProjectionReport;

PressureProjectionGrid grid(bool mapped = false, double scale = 1.0)
{
    PressureProjectionGrid result;
    result.nx = 8;
    result.ny = 7;
    result.nz = 6;
    result.dx = 100.0 * scale;
    result.dy = 120.0 * scale;
    for (int i = 0; i < result.nx; ++i)
        for (int j = 0; j < result.ny; ++j)
            for (int k = 0; k < result.nz; ++k)
            {
                const double h = mapped ? 40.0 * std::exp(-.1 * ((i - 3.0) * (i - 3.0) + (j - 3.0) * (j - 3.0))) : 0.0;
                const double fraction = (k + .5) / result.nz;
                result.height.push_back(scale * (600.0 * fraction + h * (1.0 - fraction)));
                result.volume.push_back(result.dx * result.dy * scale * (600.0 - h) / result.nz);
                result.density.push_back(1.25);
                result.reference_density.push_back(1.0 + .01 * k);
            }
    return result;
}

void remove_volume_mean(std::vector<double>& values, const PressureProjectionGrid& geometry)
{
    double weighted = 0.0;
    double volume   = 0.0;
    for (std::size_t n = 0; n < values.size(); ++n)
    {
        weighted += geometry.volume[n] * values[n];
        volume += geometry.volume[n];
    }
    for (double& value : values)
        value -= weighted / volume;
}
} // namespace

TEST_CASE("pressure projection reconstructs physical hydrostatic pressure on flat and mapped grids",
          "[diagnostics][pressure][projection][reference]")
{
    for (bool mapped : {false, true})
        for (bool periodic : {false, true})
        {
            auto geometry                        = grid(mapped);
            geometry.periodic_x                  = periodic;
            geometry.periodic_y                  = periodic;
            const std::size_t                  n = geometry.height.size();
            std::array<std::vector<double>, 3> acceleration;
            for (auto& field : acceleration)
                field.assign(n, 0.0);
            acceleration[2].assign(n, .25);
            std::vector<double> expected(n);
            for (std::size_t q = 0; q < n; ++q)
                expected[q] = geometry.density[q] * .25 * geometry.height[q];
            remove_volume_mean(expected, geometry);
            PressureProjection       projection(geometry);
            PressureProjectionReport report;
            const auto               actual = projection.solve(acceleration, report);
            REQUIRE(report.iterations > 0);
            CHECK(report.relative_residual <= 1e-10);
            CHECK(std::abs(report.volume_mean_pressure) < 1e-11);
            for (std::size_t q = 0; q < n; ++q)
                CHECK(actual[q] == Approx(expected[q]).margin(2e-7));
        }
}

TEST_CASE("pressure projection has the analytic rotation and strain pressure signs and dimensions",
          "[diagnostics][pressure][projection][reference]")
{
    for (double sign : {-1.0, 1.0})
    {
        std::vector<double> previous;
        for (double scale : {1.0, 2.0})
        {
            auto geometry = grid(false, scale);
            geometry.reference_density.assign(geometry.height.size(), 1.0);
            std::array<std::vector<double>, 3> acceleration;
            std::vector<double>                expected;
            constexpr double                   rate = .01;
            for (int i = 0; i < geometry.nx; ++i)
                for (int j = 0; j < geometry.ny; ++j)
                    for (int k = 0; k < geometry.nz; ++k)
                    {
                        const double x = (i - .5 * (geometry.nx - 1)) * geometry.dx;
                        const double y = (j - .5 * (geometry.ny - 1)) * geometry.dy;
                        acceleration[0].push_back(sign * rate * rate * x);
                        acceleration[1].push_back(sign * rate * rate * y);
                        acceleration[2].push_back(0.0);
                        expected.push_back(sign * .5 * geometry.density[0] * rate * rate * (x * x + y * y));
                    }
            remove_volume_mean(expected, geometry);
            PressureProjection       projection(geometry);
            PressureProjectionReport report;
            const auto               actual = projection.solve(acceleration, report);
            CHECK(report.relative_residual <= 1e-10);
            for (std::size_t q = 0; q < expected.size(); ++q)
            {
                CHECK(actual[q] == Approx(expected[q]).margin(2e-8));
                if (!previous.empty())
                    CHECK(actual[q] == Approx(4.0 * previous[q]).margin(2e-8));
            }
            previous = actual;
        }
    }
}

TEST_CASE("pressure projection retains its null mode and rejects failed solves atomically",
          "[diagnostics][pressure][projection][atomic]")
{
    auto                               geometry = grid(true);
    PressureProjection                 projection(geometry);
    std::array<std::vector<double>, 3> acceleration;
    for (auto& field : acceleration)
        field.assign(geometry.height.size(), 0.0);
    PressureProjectionReport report;
    const auto               rest = projection.solve(acceleration, report);
    CHECK(report.iterations == 0);
    CHECK(std::all_of(rest.begin(), rest.end(), [](double value) { return value == 0.0; }));
    for (std::size_t n = 0; n < acceleration[2].size(); ++n)
        acceleration[2][n] = std::sin(.37 * n);
    report.iterations = -10;
    REQUIRE_THROWS(projection.solve(acceleration, report, 1e-10, 1));
    CHECK(report.iterations == -10);
    acceleration[0][2] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(projection.solve(acceleration, report));
    CHECK(report.iterations == -10);
    acceleration[0][2] = 0.0;
    const auto actual  = projection.solve(acceleration, report);
    CHECK(report.relative_residual <= 1e-10);
    CHECK(std::all_of(actual.begin(), actual.end(), [](double value) { return std::isfinite(value); }));
    auto invalid      = geometry;
    invalid.height[1] = invalid.height[0];
    REQUIRE_THROWS(PressureProjection(invalid));
    invalid            = geometry;
    invalid.density[0] = 0.0;
    REQUIRE_THROWS(PressureProjection(invalid));
}
