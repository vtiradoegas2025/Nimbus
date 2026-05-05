/**
 * @file test_advect_vertical_cgrid_gpu_parity.cpp
 * @brief CPU/GPU parity for cylindrical Arakawa C-grid vertical advection
 *        (Phase C.9.3 of docs/CoordinateBackend_Plan.md).
 *
 * Mirrors the C.9.1 / C.9.2 pattern. The CPU kernel
 * advect_scalar_1d_z_kernel_cylindrical_cgrid uses double-precision
 * internal arithmetic; the C.9 GPU shader advect_vertical_cgrid.comp
 * uses float-precision throughout. The parity gate (1e-3 absolute, per
 * the plan doc) accommodates the float-vs-double drift while still
 * catching structural bugs (wrong stencil, swapped k-index, missing
 * face-velocity read, etc.).
 *
 * Vertical advection has no curvature (no 1/r factor) and is not
 * periodic in k, so the boundary slopes (k=0 and k=NZ-1) are forced
 * to zero by the same fall-back-to-first-order convention used in the
 * radial direction.
 *
 * Verification setups:
 *
 *   1. Smooth vertical Gaussian column + uniform upward flow.
 *      Structural test: the GPU shader must read w from the SAME
 *      face index as the CPU kernel (z_face[k]) and produce the same
 *      flux divergence.
 *
 *   2. Sharp vertical step + downward flow.  The MUSCL limiter
 *      activates non-trivially around the step. A miscoded slope-clamp
 *      at the surface (k=0) or lid (k=NZ-1) shows up here.
 *
 *   3. Pure horizontal flow leaves vertical step a structural no-op.
 *      With w = 0, div_F_z = 0 and dst must equal src bit-exactly in
 *      interior cells.
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

TEST_CASE("C.9.3 GPU C-grid vertical advection parity (tol 1e-3)",
          "[vulkan][gpu_parity][cgrid][advection]")
{
    // dt = 0.5s with w up to ~6 m/s gives CFL ~ 0.012 in z, well below
    // the stability limit, but enough that one Forward-Euler step moves
    // the field by O(1e-3) amplitude. That is what makes the parity gate
    // (1e-3 tolerance) sensitive enough to catch a multiplicative bug in
    // the divergence -- with smaller dt the kernel motion is too small
    // for any plausible bug to exceed 1e-3.
    setup_cgrid_grid(/*nr=*/16, /*nth=*/16, /*nz=*/24,
                     /*dr=*/100.0, /*dz=*/250.0, /*dt=*/0.5);

    if (!try_init_gpu())
    {
        WARN("GPU unavailable -- skipping C-grid vertical parity");
        return;
    }

    auto* backend = mutable_compute_backend();
    if (backend == nullptr || !backend->supports_vertical_advection_cgrid_dispatch())
    {
        WARN("GPU does not support C-grid vertical advection -- skipping");
        shutdown_backend();
        return;
    }

    SECTION("Smooth vertical Gaussian + uniform upward flow")
    {
        const double k_center = 12.0;
        const double sigma_k  = 3.0;

        u.resize(NR, NTH, NZ, 0.0f);
        v.resize(NR, NTH, NZ, 0.0f);
        w.resize(NR, NTH, NZ, 6.0f);  // upward (positive z-face velocity)

        Field3D src(NR, NTH, NZ);
        for (int k = 0; k < NZ; ++k)
        {
            const double dk = (static_cast<double>(k) - k_center) / sigma_k;
            const double bump = std::exp(-0.5 * dk * dk);
            for (int i = 0; i < NR; ++i)
            {
                for (int j = 0; j < NTH; ++j)
                {
                    src(i, j, k) = static_cast<float>(bump);
                }
            }
        }

        Field3D dst_gpu(NR, NTH, NZ, 0.0f);
        Field3D dst_cpu(NR, NTH, NZ, 0.0f);

        REQUIRE(backend->supports_vertical_advection_cgrid_dispatch());
        for (std::size_t idx = 0; idx < src.size(); ++idx)
        {
            dst_gpu.data()[idx] = src.data()[idx];
        }
        const bool dispatched = backend->dispatch_vertical_advection_cgrid(
            src.data(), w.data(), dst_gpu.data(),
            NR, NTH, NZ,
            static_cast<float>(dz), static_cast<float>(dt));
        REQUIRE(dispatched);

        // CPU reference via the kernel's CPU fallback.
        shutdown_backend();
        global_compute_backend_config.backend = "cpu";
        global_compute_backend_config.allow_fallback = true;
        std::string err;
        REQUIRE(initialize_compute_backend_runtime(err));
        advect_scalar_1d_z_kernel_cylindrical_cgrid(src, dst_cpu, dt);

        // Sanity: kernel must have moved the field. With w != 0 and a
        // non-flat z profile, the interior must differ from src.
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
        INFO("Smooth + upward: max_abs=" << parity.max_abs
             << " at (i=" << parity.worst_i
             << ", j=" << parity.worst_j
             << ", k=" << parity.worst_k
             << ") cpu=" << parity.worst_cpu
             << " gpu=" << parity.worst_gpu);
        REQUIRE(parity.max_abs < 1.0e-3);

        shutdown_backend();
        REQUIRE(try_init_gpu());
    }

    SECTION("Sharp vertical step + downward flow")
    {
        // A tanh-smoothed step centered between k=8 and k=15 produces
        // a non-zero gradient over a few cells. The MUSCL limiter
        // activates non-trivially over the transition. With downward
        // flow, the step propagates toward k=0 and the surface
        // boundary slope (slope_z[0] = 0) matters.
        u.resize(NR, NTH, NZ, 0.0f);
        v.resize(NR, NTH, NZ, 0.0f);
        w.resize(NR, NTH, NZ, -4.0f);

        Field3D src(NR, NTH, NZ);
        for (int k = 0; k < NZ; ++k)
        {
            const double k_norm = (static_cast<double>(k) - 11.5) / 1.5;
            const double profile = 0.5 * (1.0 + std::tanh(k_norm));
            for (int i = 0; i < NR; ++i)
            {
                for (int j = 0; j < NTH; ++j)
                {
                    src(i, j, k) = static_cast<float>(profile);
                }
            }
        }

        Field3D dst_gpu(NR, NTH, NZ, 0.0f);
        Field3D dst_cpu(NR, NTH, NZ, 0.0f);

        for (std::size_t idx = 0; idx < src.size(); ++idx)
        {
            dst_gpu.data()[idx] = src.data()[idx];
        }
        const bool dispatched = backend->dispatch_vertical_advection_cgrid(
            src.data(), w.data(), dst_gpu.data(),
            NR, NTH, NZ,
            static_cast<float>(dz), static_cast<float>(dt));
        REQUIRE(dispatched);

        shutdown_backend();
        global_compute_backend_config.backend = "cpu";
        global_compute_backend_config.allow_fallback = true;
        std::string err;
        REQUIRE(initialize_compute_backend_runtime(err));
        advect_scalar_1d_z_kernel_cylindrical_cgrid(src, dst_cpu, dt);

        const auto parity = compare_interior(dst_cpu.data(), dst_gpu.data(),
                                             NR, NTH, NZ);
        INFO("Sharp step + downward: max_abs=" << parity.max_abs
             << " at (i=" << parity.worst_i
             << ", j=" << parity.worst_j
             << ", k=" << parity.worst_k
             << ") cpu=" << parity.worst_cpu
             << " gpu=" << parity.worst_gpu);
        REQUIRE(parity.max_abs < 1.0e-3);

        shutdown_backend();
        REQUIRE(try_init_gpu());
    }

    SECTION("Pure horizontal flow leaves vertical step a structural no-op")
    {
        u.resize(NR, NTH, NZ, 5.0f);
        v.resize(NR, NTH, NZ, 3.0f);
        w.resize(NR, NTH, NZ, 0.0f);

        Field3D src(NR, NTH, NZ);
        for (int k = 0; k < NZ; ++k)
        {
            const float profile = std::cos(static_cast<float>(k) * 0.25f);
            for (int i = 0; i < NR; ++i)
            {
                for (int j = 0; j < NTH; ++j)
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
        const bool dispatched = backend->dispatch_vertical_advection_cgrid(
            src.data(), w.data(), dst_gpu.data(),
            NR, NTH, NZ,
            static_cast<float>(dz), static_cast<float>(dt));
        REQUIRE(dispatched);

        const auto parity = compare_interior(src.data(), dst_gpu.data(),
                                             NR, NTH, NZ);
        INFO("Horizontal-only no-op: max_abs=" << parity.max_abs
             << " at (i=" << parity.worst_i
             << ", j=" << parity.worst_j
             << ", k=" << parity.worst_k << ")");
        REQUIRE(parity.max_abs < 1.0e-6);
    }

    shutdown_backend();
}
