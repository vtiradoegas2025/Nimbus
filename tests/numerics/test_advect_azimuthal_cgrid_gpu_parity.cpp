/**
 * @file test_advect_azimuthal_cgrid_gpu_parity.cpp
 * @brief CPU/GPU parity for cylindrical Arakawa C-grid azimuthal advection
 *        (Phase C.9.2 of docs/CoordinateBackend_Plan.md).
 *
 * The C.7 CPU kernel advect_scalar_1d_theta_kernel_cylindrical_cgrid
 * uses double-precision internal arithmetic. The C.9 GPU shader
 * advect_azimuthal_cgrid.comp uses float-precision throughout. The
 * parity gate (1e-3 absolute, per the plan doc) accommodates the
 * float-vs-double drift while still catching structural bugs.
 *
 * Theta is fully periodic so every j is an interior point. The 5-point
 * j-stencil (j-2, j-1, j, j+1, j+2) wraps via modular arithmetic. The
 * verification gates exercise the wrap directly (a bump that straddles
 * j=0).
 *
 * Verification setups:
 *
 *   1. Smooth Gaussian-in-theta + uniform azimuthal flow.  The bump is
 *      placed at theta = pi (NTH/2) and the flow rotates eastward, so
 *      the wrap at j=0 is not yet relevant -- this is the basic
 *      structural check.
 *
 *   2. Wrap-straddling bump + uniform azimuthal flow.  The bump is
 *      placed at theta = 0 (peak at j=0, with non-trivial values at
 *      j=NTH-1, j=1, etc.). Any miscoded periodic-modulo arithmetic
 *      in the shader will diverge from the CPU reference here.
 *
 *   3. Pure radial flow leaves azimuthal step a structural no-op.
 *      With v = 0, div_F_theta = 0 and dst must equal src bit-exactly
 *      in interior cells. Catches a class of bugs where the shader
 *      writes nonzero values into untouched cells.
 */

#include "catch2/catch.hpp"

#include "compute/compute_backend.hpp"
#include "core/coordinate_system.hpp"
#include "core/field3d.hpp"
#include "core/grid_geometry.hpp"
#include "core/runtime_config.hpp"
#include "core/simulation.hpp"
#include "numerics/advection/advection_cylindrical_cgrid.hpp"

#include <cmath>
#include <cstddef>
#include <string>

namespace
{

constexpr double kPi = 3.14159265358979323846;

void setup_cgrid_grid(int nr, int nth, int nz, double dr_m, double dz_m, double dt_s)
{
    NR     = nr;
    NTH    = nth;
    NZ     = nz;
    dr     = dr_m;
    dz     = dz_m;
    dt     = dt_s;
    dtheta = 2.0 * kPi / static_cast<double>(nth);

    global_coordinate_system = CoordinateSystem::Cylindrical;
    global_stagger_type      = StaggerType::CGrid;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
                                    global_coordinate_system,
                                    global_stagger_type);
}

bool try_init_gpu()
{
    std::string error;
    global_compute_backend_config.backend = "vulkan";
    global_compute_backend_config.allow_fallback = false;
    bool ok = initialize_compute_backend_runtime(error);
    if (!ok)
    {
        global_compute_backend_config.backend = "cpu";
        global_compute_backend_config.allow_fallback = true;
        initialize_compute_backend_runtime(error);
        return false;
    }
    return true;
}

void shutdown_backend()
{
    shutdown_compute_backend_runtime();
}

struct InteriorParity
{
    double max_abs = 0.0;
    int worst_i = -1, worst_j = -1, worst_k = -1;
    float worst_cpu = 0.0f, worst_gpu = 0.0f;
};

InteriorParity compare_interior(const float* cpu, const float* gpu,
                                int nr, int nth, int nz)
{
    InteriorParity err;
    for (int i = 1; i < nr - 1; ++i)
    {
        for (int j = 0; j < nth; ++j)
        {
            for (int k = 1; k < nz - 1; ++k)
            {
                const std::size_t idx = static_cast<std::size_t>(
                    (i * nth + j) * nz + k);
                const double diff = std::fabs(static_cast<double>(cpu[idx]) -
                                              static_cast<double>(gpu[idx]));
                if (diff > err.max_abs)
                {
                    err.max_abs = diff;
                    err.worst_i = i;
                    err.worst_j = j;
                    err.worst_k = k;
                    err.worst_cpu = cpu[idx];
                    err.worst_gpu = gpu[idx];
                }
            }
        }
    }
    return err;
}

}  // namespace

TEST_CASE("C.9.2 GPU C-grid azimuthal advection parity (tol 1e-3)",
          "[vulkan][gpu_parity][cgrid][advection]")
{
    setup_cgrid_grid(/*nr=*/16, /*nth=*/32, /*nz=*/16,
                     /*dr=*/100.0, /*dz=*/250.0, /*dt=*/0.05);

    if (!try_init_gpu())
    {
        WARN("GPU unavailable -- skipping C-grid azimuthal parity");
        return;
    }

    auto* backend = mutable_compute_backend();
    if (backend == nullptr || !backend->supports_azimuthal_advection_cgrid_dispatch())
    {
        WARN("GPU does not support C-grid azimuthal advection -- skipping");
        shutdown_backend();
        return;
    }

    SECTION("Smooth theta-Gaussian + uniform eastward flow")
    {
        const int j_center = NTH / 2;
        const double sigma_th = 4.0;  // in cells (so sigma in radians = 4*dtheta)

        u.resize(NR, NTH, NZ, 0.0f);
        v.resize(NR, NTH, NZ, 6.0f);  // eastward (positive theta-face velocity)
        w.resize(NR, NTH, NZ, 0.0f);

        Field3D src(NR, NTH, NZ);
        for (int i = 0; i < NR; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                const double dj = (static_cast<double>(j) - j_center) / sigma_th;
                const double bump = std::exp(-0.5 * dj * dj);
                for (int k = 0; k < NZ; ++k)
                {
                    src(i, j, k) = static_cast<float>(bump);
                }
            }
        }

        Field3D dst_gpu(NR, NTH, NZ, 0.0f);
        Field3D dst_cpu(NR, NTH, NZ, 0.0f);

        REQUIRE(backend->supports_azimuthal_advection_cgrid_dispatch());
        for (std::size_t idx = 0; idx < src.size(); ++idx)
        {
            dst_gpu.data()[idx] = src.data()[idx];
        }
        const bool dispatched = backend->dispatch_azimuthal_advection_cgrid(
            src.data(), v.data(), dst_gpu.data(),
            NR, NTH, NZ,
            static_cast<float>(dr),
            static_cast<float>(dtheta),
            static_cast<float>(dt));
        REQUIRE(dispatched);

        // CPU reference via the kernel's CPU fallback.
        shutdown_backend();
        global_compute_backend_config.backend = "cpu";
        global_compute_backend_config.allow_fallback = true;
        std::string err;
        REQUIRE(initialize_compute_backend_runtime(err));
        advect_scalar_1d_theta_kernel_cylindrical_cgrid(src, dst_cpu, dt);

        // Sanity: kernel must have moved the field. With v != 0 and a
        // non-flat theta profile, the interior must differ from src.
        double max_kernel_motion = 0.0;
        for (int i = 1; i < NR - 1; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                for (int k = 1; k < NZ - 1; ++k)
                {
                    const std::size_t idx =
                        static_cast<std::size_t>((i * NTH + j) * NZ + k);
                    const double diff = std::fabs(
                        static_cast<double>(dst_gpu.data()[idx]) -
                        static_cast<double>(src.data()[idx]));
                    max_kernel_motion = std::max(max_kernel_motion, diff);
                }
            }
        }
        INFO("kernel made motion: max|dst_gpu - src| = " << max_kernel_motion);
        REQUIRE(max_kernel_motion > 1.0e-6);

        const auto parity = compare_interior(dst_cpu.data(), dst_gpu.data(),
                                             NR, NTH, NZ);
        INFO("Smooth + eastward: max_abs=" << parity.max_abs
             << " at (i=" << parity.worst_i
             << ", j=" << parity.worst_j
             << ", k=" << parity.worst_k
             << ") cpu=" << parity.worst_cpu
             << " gpu=" << parity.worst_gpu);
        REQUIRE(parity.max_abs < 1.0e-3);

        shutdown_backend();
        REQUIRE(try_init_gpu());
    }

    SECTION("Wrap-straddling bump exercises periodic theta arithmetic")
    {
        // Bump centered at j=0, so the 5-point stencil at j=0 reaches
        // j=NTH-2, NTH-1, 0, 1, 2 (wrapping). At j=NTH-1 it reaches
        // NTH-3, NTH-2, NTH-1, 0, 1. These wrap cells must be read
        // correctly via the modulo arithmetic in the shader.
        const double sigma_th = 3.0;

        u.resize(NR, NTH, NZ, 0.0f);
        v.resize(NR, NTH, NZ, -4.0f);  // westward
        w.resize(NR, NTH, NZ, 0.0f);

        Field3D src(NR, NTH, NZ);
        for (int i = 0; i < NR; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                // Wrap-aware distance to j=0: min(j, NTH-j)
                const int dist_cells = std::min(j, NTH - j);
                const double dj = static_cast<double>(dist_cells) / sigma_th;
                const double bump = std::exp(-0.5 * dj * dj);
                for (int k = 0; k < NZ; ++k)
                {
                    src(i, j, k) = static_cast<float>(bump);
                }
            }
        }

        Field3D dst_gpu(NR, NTH, NZ, 0.0f);
        Field3D dst_cpu(NR, NTH, NZ, 0.0f);

        for (std::size_t idx = 0; idx < src.size(); ++idx)
        {
            dst_gpu.data()[idx] = src.data()[idx];
        }
        const bool dispatched = backend->dispatch_azimuthal_advection_cgrid(
            src.data(), v.data(), dst_gpu.data(),
            NR, NTH, NZ,
            static_cast<float>(dr),
            static_cast<float>(dtheta),
            static_cast<float>(dt));
        REQUIRE(dispatched);

        shutdown_backend();
        global_compute_backend_config.backend = "cpu";
        global_compute_backend_config.allow_fallback = true;
        std::string err;
        REQUIRE(initialize_compute_backend_runtime(err));
        advect_scalar_1d_theta_kernel_cylindrical_cgrid(src, dst_cpu, dt);

        const auto parity = compare_interior(dst_cpu.data(), dst_gpu.data(),
                                             NR, NTH, NZ);
        INFO("Wrap-straddling: max_abs=" << parity.max_abs
             << " at (i=" << parity.worst_i
             << ", j=" << parity.worst_j
             << ", k=" << parity.worst_k
             << ") cpu=" << parity.worst_cpu
             << " gpu=" << parity.worst_gpu);
        REQUIRE(parity.max_abs < 1.0e-3);

        shutdown_backend();
        REQUIRE(try_init_gpu());
    }

    SECTION("Pure radial flow leaves azimuthal step a structural no-op")
    {
        u.resize(NR, NTH, NZ, 5.0f);
        v.resize(NR, NTH, NZ, 0.0f);
        w.resize(NR, NTH, NZ, 0.0f);

        Field3D src(NR, NTH, NZ);
        for (int i = 0; i < NR; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                const float profile = std::cos(static_cast<float>(j) * 0.4f);
                for (int k = 0; k < NZ; ++k)
                {
                    src(i, j, k) = profile;
                }
            }
        }

        Field3D dst_gpu(NR, NTH, NZ, 0.0f);
        for (std::size_t idx = 0; idx < src.size(); ++idx)
        {
            dst_gpu.data()[idx] = src.data()[idx];
        }
        const bool dispatched = backend->dispatch_azimuthal_advection_cgrid(
            src.data(), v.data(), dst_gpu.data(),
            NR, NTH, NZ,
            static_cast<float>(dr),
            static_cast<float>(dtheta),
            static_cast<float>(dt));
        REQUIRE(dispatched);

        const auto parity = compare_interior(src.data(), dst_gpu.data(),
                                             NR, NTH, NZ);
        INFO("Radial-only no-op: max_abs=" << parity.max_abs
             << " at (i=" << parity.worst_i
             << ", j=" << parity.worst_j
             << ", k=" << parity.worst_k << ")");
        REQUIRE(parity.max_abs < 1.0e-6);
    }

    shutdown_backend();
}
