/**
 * @file test_advect_radial_cgrid_gpu_parity.cpp
 * @brief CPU/GPU parity for cylindrical Arakawa C-grid radial advection
 *        (Phase C.9.1 of docs/CoordinateBackend_Plan.md).
 *
 * The C.7 CPU kernel advect_scalar_1d_r_kernel_cylindrical_cgrid uses
 * double-precision internal arithmetic, while the C.9 GPU shader
 * advect_radial_cgrid.comp uses float-precision throughout. The parity
 * gate (1e-3 absolute, per the plan doc) accommodates the float-vs-double
 * drift while still catching structural bugs (wrong stencil, swapped
 * indices, missing geometric factors, miscoded MC limiter, etc).
 *
 * When the Vulkan backend is unavailable on the host running the test
 * (e.g., CI runners without a GPU), the test reports SKIPPED via WARN
 * and returns success -- this matches the convention in
 * tests/vulkan/test_gpu_parity.cpp so the same test binary can run on
 * both GPU and non-GPU machines.
 *
 * Verification setups:
 *
 *   1. Smooth Gaussian-bump scalar with uniform inward radial flow.
 *      Pure structural test: any miscoding of the slope, the MC
 *      limiter, or the cylindrical 1/r factor will diverge from the
 *      double-precision CPU reference well above the 1e-3 threshold.
 *
 *   2. Solid-body azimuthal flow + radially-varying scalar.  The
 *      radial advection step should be (very nearly) a no-op because
 *      u = 0 everywhere; the residual is bounded by the boundary-cell
 *      seeding, not by the kernel arithmetic. This catches a class of
 *      bugs where the GPU shader writes nonzero values into cells
 *      that should be untouched.
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

TEST_CASE("C.9.1 GPU C-grid radial advection parity (tol 1e-3)",
          "[vulkan][gpu_parity][cgrid][advection]")
{
    setup_cgrid_grid(/*nr=*/24, /*nth=*/16, /*nz=*/16,
                     /*dr=*/100.0, /*dz=*/250.0, /*dt=*/0.05);

    if (!try_init_gpu())
    {
        WARN("GPU unavailable -- skipping C-grid radial parity");
        return;
    }

    auto* backend = mutable_compute_backend();
    if (backend == nullptr || !backend->supports_radial_advection_cgrid_dispatch())
    {
        WARN("GPU does not support C-grid radial advection -- skipping");
        shutdown_backend();
        return;
    }

    SECTION("Smooth Gaussian bump + uniform inward radial flow")
    {
        const double r_center = 12.0 * dr;
        const double sigma_r  = 3.0 * dr;
        const float  u_uniform = -8.0f;  // inward (negative r-face velocity)

        Field3D src(NR, NTH, NZ, 0.0f);
        u.resize(NR, NTH, NZ, u_uniform);
        v.resize(NR, NTH, NZ, 0.0f);
        w.resize(NR, NTH, NZ, 0.0f);

        for (int i = 0; i < NR; ++i)
        {
            const double r_i = i * dr;
            const double dr_off = (r_i - r_center) / sigma_r;
            const double bump = std::exp(-0.5 * dr_off * dr_off);
            for (int j = 0; j < NTH; ++j)
            {
                for (int k = 0; k < NZ; ++k)
                {
                    src(i, j, k) = static_cast<float>(bump);
                }
            }
        }

        Field3D dst_gpu(NR, NTH, NZ, 0.0f);
        Field3D dst_cpu(NR, NTH, NZ, 0.0f);

        // GPU path: invoke the shader directly so we can assert the
        // dispatch returned true. The high-level kernel will silently
        // fall back to CPU if the dispatch fails, which would mask the
        // very thing we are trying to test.
        REQUIRE(backend->supports_radial_advection_cgrid_dispatch());
        // Seed boundaries from src first (the production CPU kernel
        // does this via seed_destination_from_source before attempting
        // the dispatch).
        for (std::size_t idx = 0; idx < src.size(); ++idx)
        {
            dst_gpu.data()[idx] = src.data()[idx];
        }
        const bool dispatched = backend->dispatch_radial_advection_cgrid(
            src.data(), u.data(), dst_gpu.data(),
            NR, NTH, NZ,
            static_cast<float>(dr), static_cast<float>(dt));
        REQUIRE(dispatched);

        // CPU reference: shut the GPU off and re-run the same kernel.
        shutdown_backend();
        global_compute_backend_config.backend = "cpu";
        global_compute_backend_config.allow_fallback = true;
        std::string err;
        REQUIRE(initialize_compute_backend_runtime(err));
        advect_scalar_1d_r_kernel_cylindrical_cgrid(src, dst_cpu, dt);

        // Sanity: the kernel must actually have done work. If u != 0
        // and the bump straddles many cells, the interior must differ
        // from src, otherwise the kernel is a no-op.
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
        INFO("Gaussian + inward flow: max_abs=" << parity.max_abs
             << " at (i=" << parity.worst_i
             << ", j=" << parity.worst_j
             << ", k=" << parity.worst_k
             << ") cpu=" << parity.worst_cpu
             << " gpu=" << parity.worst_gpu);
        REQUIRE(parity.max_abs < 1.0e-3);

        // Re-init GPU for the next SECTION.
        shutdown_backend();
        REQUIRE(try_init_gpu());
    }

    SECTION("Asymmetric profile + outward radial flow exercises drift")
    {
        // An exponential decay along r combined with outward flow gives
        // non-zero slopes at every cell and a sustained gradient that
        // exposes float-vs-double accumulation between the two paths.
        u.resize(NR, NTH, NZ, 12.0f);
        v.resize(NR, NTH, NZ, 0.0f);
        w.resize(NR, NTH, NZ, 0.0f);

        Field3D src(NR, NTH, NZ);
        for (int i = 0; i < NR; ++i)
        {
            const double r_i = i * dr;
            const double profile = std::exp(-r_i / (8.0 * dr));
            for (int j = 0; j < NTH; ++j)
            {
                for (int k = 0; k < NZ; ++k)
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
        const bool dispatched = backend->dispatch_radial_advection_cgrid(
            src.data(), u.data(), dst_gpu.data(),
            NR, NTH, NZ,
            static_cast<float>(dr), static_cast<float>(dt));
        REQUIRE(dispatched);

        // CPU reference via the kernel's CPU fallback.
        shutdown_backend();
        global_compute_backend_config.backend = "cpu";
        global_compute_backend_config.allow_fallback = true;
        std::string err;
        REQUIRE(initialize_compute_backend_runtime(err));
        advect_scalar_1d_r_kernel_cylindrical_cgrid(src, dst_cpu, dt);

        const auto parity = compare_interior(dst_cpu.data(), dst_gpu.data(),
                                             NR, NTH, NZ);
        INFO("Asymmetric outward: max_abs=" << parity.max_abs
             << " at (i=" << parity.worst_i
             << ", j=" << parity.worst_j
             << ", k=" << parity.worst_k
             << ") cpu=" << parity.worst_cpu
             << " gpu=" << parity.worst_gpu);
        REQUIRE(parity.max_abs < 1.0e-3);

        shutdown_backend();
        REQUIRE(try_init_gpu());
    }

    SECTION("Pure azimuthal flow leaves radial step a structural no-op")
    {
        // u = 0 everywhere -> the kernel should produce dst == src
        // bit-exactly (no flux through any r-face). A GPU shader that
        // accidentally adds nonzero contributions (e.g., reads stale
        // descriptor data, miscomputes geometric factors) shows up here.
        u.resize(NR, NTH, NZ, 0.0f);
        v.resize(NR, NTH, NZ, 5.0f);
        w.resize(NR, NTH, NZ, 0.0f);

        Field3D src(NR, NTH, NZ);
        for (int i = 0; i < NR; ++i)
        {
            const float profile = std::cos(static_cast<float>(i) * 0.2f);
            for (int j = 0; j < NTH; ++j)
            {
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
        const bool dispatched = backend->dispatch_radial_advection_cgrid(
            src.data(), u.data(), dst_gpu.data(),
            NR, NTH, NZ,
            static_cast<float>(dr), static_cast<float>(dt));
        REQUIRE(dispatched);

        // Compare to src directly: u = 0 means div_F_r = 0 in interior cells.
        const auto parity = compare_interior(src.data(), dst_gpu.data(),
                                             NR, NTH, NZ);
        INFO("Azimuthal-only no-op: max_abs=" << parity.max_abs
             << " at (i=" << parity.worst_i
             << ", j=" << parity.worst_j
             << ", k=" << parity.worst_k << ")");
        REQUIRE(parity.max_abs < 1.0e-6);
    }

    shutdown_backend();
}
