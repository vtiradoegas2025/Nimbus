#include <catch2/catch.hpp>

#include <cmath>

#include "diagnostics/physical_derivative_kernel.hpp"
#include "diagnostics/vorticity_budget_kernel.hpp"

namespace
{

TerrainMetrics3D make_sloping_geometry(int nr, int nth, int nz,
                                       double dx, double dy, double dz)
{
    TerrainMetrics3D terrain;
    terrain.z.resize(nr, nth, nz, 0.0);
    terrain.z_face.resize(nr, nth, nz + 1, 0.0);
    terrain.J.resize(nr, nth, nz, dz);
    terrain.mx.resize(nr, nth, nz, 0.0);
    terrain.my.resize(nr, nth, nz, 0.0);
    terrain.zeta.resize(nr, nth, nz, 0.0);
    terrain.dx = dx;
    terrain.dy = dy;
    terrain.dzeta = dz;
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            const double ground = 0.18 * static_cast<double>(i) * dx
                                - 0.11 * static_cast<double>(j) * dy;
            for (int k = 0; k <= nz; ++k)
                terrain.z_face(i, j, k) = ground + static_cast<double>(k) * dz;
            for (int k = 0; k < nz; ++k)
                terrain.z(i, j, k) = ground + (static_cast<double>(k) + 0.5) * dz;
        }
    return terrain;
}

} // namespace

TEST_CASE("physical gradient removes a terrain-coordinate slope exactly",
          "[diagnostics][potential-vorticity][terrain]")
{
    constexpr int nr = 5;
    constexpr int nth = 4;
    constexpr int nz = 5;
    constexpr double dx = 800.0;
    constexpr double dy = 650.0;
    constexpr double dz = 300.0;
    constexpr double ax = 0.003;
    constexpr double ay = -0.002;
    constexpr double az = 0.007;
    const TerrainMetrics3D terrain =
        make_sloping_geometry(nr, nth, nz, dx, dy, dz);
    Field3D theta(nr, nth, nz, 0.0f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                theta(i, j, k) = static_cast<float>(
                    290.0 + ax * static_cast<double>(i) * dx
                    + ay * static_cast<double>(j) * dy
                    + az * terrain.z(i, j, k));

    REQUIRE_NOTHROW(tmv::diagnostics::validate_physical_derivative_geometry(
        nr, nth, nz, dx, dy, dz, &terrain));
    const auto value_at = [&](int i, int j, int k)
    {
        return static_cast<double>(theta(i, j, k));
    };
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const auto gradient = tmv::diagnostics::physical_gradient_at(
                    value_at, i, j, k, nr, nth, nz,
                    dx, dy, dz, &terrain);
                CHECK(gradient.x == Approx(ax).margin(2.0e-7));
                CHECK(gradient.y == Approx(ay).margin(2.0e-7));
                CHECK(gradient.z == Approx(az).margin(2.0e-7));
            }
}

TEST_CASE("Ertel PV uses all absolute-vorticity components and dry density",
          "[diagnostics][potential-vorticity][science]")
{
    const tmv::diagnostics::PhysicalGradientPoint gradient{0.004, -0.002, 0.008};
    constexpr double omega_x = 0.012;
    constexpr double omega_y = -0.006;
    constexpr double omega_z = 0.021;
    constexpr double f = 8.0e-5;
    constexpr double rho_d = 0.8;
    const double expected =
        (omega_x * gradient.x + omega_y * gradient.y
         + (omega_z + f) * gradient.z) / rho_d;
    CHECK(tmv::diagnostics::ertel_potential_vorticity_si(
              gradient, omega_x, omega_y, omega_z, f, rho_d)
          == Approx(expected).epsilon(1.0e-14));
    CHECK(std::isnan(tmv::diagnostics::ertel_potential_vorticity_si(
        gradient, omega_x, omega_y, omega_z, f, 0.0)));
}

TEST_CASE("mapped C-grid curl does not acquire terrain-slope vorticity",
          "[diagnostics][potential-vorticity][terrain][c-grid]")
{
    constexpr int nr = 5;
    constexpr int nth = 5;
    constexpr int nz = 5;
    constexpr double dx = 1000.0;
    constexpr double dy = 800.0;
    constexpr double dz = 400.0;
    constexpr double rotation = 0.01;
    const TerrainMetrics3D terrain =
        make_sloping_geometry(nr, nth, nz, dx, dy, dz);

    Field3D u(nr, nth, nz, 0.0f);
    Field3D v(nr, nth, nz, 0.0f);
    Field3D w(nr, nth, nz, 0.0f);
    Field3D rho(nr, nth, nz, 1.0f);
    Field3D pressure(nr, nth, nz, 90000.0f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                // u/v are sampled on their positive x/y faces. This linear
                // solid-body rotation remains exact after face-to-centre
                // interpolation at the tested interior point.
                u(i, j, k) = static_cast<float>(
                    -rotation * static_cast<double>(j) * dy);
                v(i, j, k) = static_cast<float>(
                    rotation * static_cast<double>(i) * dx);
                const double scalar = 90000.0
                    + 0.02 * static_cast<double>(i) * dx
                    - 0.01 * static_cast<double>(j) * dy
                    + 0.05 * terrain.z(i, j, k);
                pressure(i, j, k) = static_cast<float>(scalar);
                rho(i, j, k) = static_cast<float>(1.0 + 1.0e-6 * scalar);
            }

    const tmv::VorticityBudgetPoint budget =
        tmv::cartesian_cgrid_mapped_vorticity_budget_at(
            u, v, w, rho, pressure, terrain, 2, 2, 2, dx, dy, dz);
    CHECK(budget.omega_x == Approx(0.0).margin(2.0e-10));
    CHECK(budget.omega_y == Approx(0.0).margin(2.0e-10));
    CHECK(budget.omega_z == Approx(2.0 * rotation).margin(2.0e-8));
    CHECK(budget.baroclinic == Approx(0.0).margin(2.0e-9));
}
