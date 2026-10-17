#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <array>
#include <limits>
#include <stdexcept>
#include <iomanip>
namespace
{
struct Fluxes
{
    std::vector<float> x;
    std::size_t        count;
};
void capture(void* context, const float*, const float* x, const float*, const float*, const float*)
{
    auto& f = *static_cast<Fluxes*>(context);
    f.x.assign(x, x + f.count);
}
struct Applied
{
    std::array<std::vector<float>, 6> fields;
    std::size_t                       count;
};
void capture_applied(void*        context,
                     const float* scalar,
                     const float* x,
                     const float* y,
                     const float* z,
                     const float* requested,
                     const float* loss)
{
    auto&                             value = *static_cast<Applied*>(context);
    const std::array<const float*, 6> inputs{scalar, x, y, z, requested, loss};
    for (std::size_t field = 0; field < inputs.size(); ++field)
        value.fields[field].assign(inputs[field], inputs[field] + value.count);
}
void exercise(const char* backend)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    INFO(error);
    auto* kernels = runtime.backend->terrain_kernels();
    REQUIRE(kernels);
    constexpr int         nx = 8, ny = 7, nz = 6;
    constexpr std::size_t n = nx * ny * nz;
    std::vector<float>    q(n), rho(n, 1), jacobian(n, 1), x(n, .1f), zero(n, 0), output(n);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
                q[(i * ny + j) * nz + k] = 3 + float(i * i * i) + .1f * j + .01f * k;
    MappedTerrainScalarTransportDispatch d;
    MappedTerrainScalarTransportReport   report;
    Fluxes                               observed{{}, n};
    d.scalar_begin   = q.data();
    d.density_begin  = rho.data();
    d.density_end    = rho.data();
    d.mass_flux_x    = x.data();
    d.mass_flux_y    = zero.data();
    d.mass_flux_zeta = zero.data();
    d.jacobian       = jacobian.data();
    d.scalar_end     = output.data();
    d.nx             = nx;
    d.ny             = ny;
    d.nz             = nz;
    d.dx = d.dy = d.dzeta        = 1;
    d.dt                         = .01f;
    d.density_floor              = 1e-6f;
    d.report                     = &report;
    d.completed_observer         = capture;
    d.completed_observer_context = &observed;
    d.weno                       = false;
    REQUIRE(kernels->dispatch_mapped_scalar_transport(d));
    REQUIRE(report.complete);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const auto c     = (i * ny + j) * nz + k;
                float      slope = 0;
                if (i >= 1 && i < nx - 1)
                {
                    const float left = q[c] - q[c - ny * nz], right = q[c + ny * nz] - q[c];
                    slope = std::min({.5f * (left + right), 2 * left, 2 * right});
                }
                const float expected = .1f * (q[c] + .5f * slope);
                CHECK(observed.x[c] == Approx(expected).margin(3e-6));
            }
    const auto mc = observed.x;
    d.weno        = true;
    REQUIRE(kernels->dispatch_mapped_scalar_transport(d));
    bool different = false;
    for (std::size_t c = 0; c < n; ++c)
        different = different || std::abs(mc[c] - observed.x[c]) > 1e-5;
    CHECK(different);
    Applied applied{{}, n};
    d.applied_observer         = capture_applied;
    d.applied_observer_context = &applied;
    for (bool weno : {false, true})
        for (bool positive : {false, true})
            for (float scale : {1.0f, 1e-36f})
            {
                CAPTURE(weno, positive, scale);
                d.weno              = weno;
                d.positive_definite = positive;
                d.dt                = .125f;
                for (std::size_t c = 0; c < n; ++c)
                {
                    q[c]        = scale * (1 + float(c % 9));
                    jacobian[c] = .75f + .25f * float(c % 5);
                    x[c]        = c % 2 ? -16.0f : 16.0f;
                }
                REQUIRE(kernels->dispatch_mapped_scalar_transport(d));
                REQUIRE(applied.fields[0] == output);
                double requested_inventory = 0, outward_inventory = 0, scale_inventory = 0;
                for (int i = 1; i < nx - 1; ++i)
                    for (int j = 1; j < ny - 1; ++j)
                        for (int k = 0; k < nz; ++k)
                        {
                            const auto   c         = (i * ny + j) * nz + k;
                            const double requested = double(applied.fields[4][c]) - applied.fields[5][c];
                            const double face_rate = double(applied.fields[1][c - ny * nz]) - applied.fields[1][c];
                            const double rate      = requested * jacobian[c];
                            CHECK(std::isfinite(output[c]));
                            if (positive)
                                CHECK(output[c] >= 0);
                            CHECK(applied.fields[2][c] == 0);
                            CHECK(applied.fields[3][c] == 0);
                            INFO("rate=" << std::scientific << rate << " face_rate=" << face_rate << " west="
                                         << applied.fields[1][c - ny * nz] << " east=" << applied.fields[1][c]);
                            CHECK(std::abs(rate - face_rate) <=
                                  16 * std::numeric_limits<float>::epsilon() *
                                          (std::abs(double(applied.fields[1][c - ny * nz])) +
                                           std::abs(double(applied.fields[1][c]))) +
                                      2 * std::numeric_limits<float>::denorm_min());
                            requested_inventory += rate;
                            scale_inventory += std::abs(face_rate);
                            if (i == 1)
                                outward_inventory -= applied.fields[1][c - ny * nz];
                            if (i == nx - 2)
                                outward_inventory += applied.fields[1][c];
                        }
                CHECK(std::abs(requested_inventory + outward_inventory) <=
                      16 * std::numeric_limits<float>::epsilon() * scale_inventory +
                          2 * n * std::numeric_limits<float>::denorm_min());
            }
    const auto saved_output = output;
    const auto saved_report = report;
    d.applied_observer = +[](void*, const float*, const float*, const float*, const float*, const float*, const float*)
    { throw std::runtime_error("rejected interface capture"); };
    REQUIRE_THROWS(kernels->dispatch_mapped_scalar_transport(d));
    CHECK(output == saved_output);
    CHECK(report.complete == saved_report.complete);
    CHECK(report.scalar_update_dispatch_count == saved_report.scalar_update_dispatch_count);
}
} // namespace
TEST_CASE("Mapped scalar reconstruction honors MC and WENO selection on Metal", "[metal][mapped_reconstruction]")
{
    exercise("metal");
}
TEST_CASE("Mapped scalar reconstruction honors MC and WENO selection on Vulkan", "[vulkan][mapped_reconstruction]")
{
    exercise("vulkan");
}
