/**
 * @file advection_cartesian.cpp
 * @brief Cartesian-grid scalar advection kernel implementations (Phase A.5).
 *
 * Implements the three helpers declared in
 * `include/numerics/advection_cartesian.hpp`:
 *
 *   - advect_scalar_1d_x_kernel_cartesian()
 *   - advect_scalar_1d_y_kernel_cartesian()
 *   - apply_diffusion_kernel_cartesian()
 *
 * The kernels mirror the cylindrical first-order-upwind + Forward-Euler
 * structure of the cylindrical helpers in `src/advection/advection.cpp`.
 * Two structural differences keep the Cartesian path away from the
 * cylindrical-grid traps:
 *
 *   1. The y kernel does NOT carry the `1/r` factor that the cylindrical
 *      θ kernel applies via `(v / r) * dq/dθ`. On a Cartesian grid, the
 *      arc-length spacing is just `dy`, not `r · dθ`.
 *
 *   2. The y kernel does NOT use the periodic wraparound
 *      `j_prev = (j − 1 + NTH) % NTH`. The j = 0 and j = NTH − 1 faces
 *      are physical (zero-gradient) boundaries, not the same cell on the
 *      other side of the singular axis.
 *
 * The diffusion kernel mirrors the cylindrical 7-point Laplacian without
 * the `1/(r² · dθ²)` factor in the y direction.
 *
 * The vertical TVD scheme in `src/numerics/advection/schemes/tvd/tvd.cpp`
 * is already coordinate-agnostic (it loops i, j, k and only operates on
 * the z column at each (i, j) — no r, θ, or periodic wraparound). The
 * Cartesian dispatcher reuses it verbatim for the z step.
 *
 * Boundary convention: every kernel here begins by copying `src` into
 * `dst` (a single contiguous memcpy), then overwrites the interior
 * (1..NR−2, 1..NTH−2 for the y kernel, 1..NZ−2). The six face planes
 * carry the source values through unchanged. This is the discrete
 * equivalent of `∂q/∂n = 0` on every face — the same zero-gradient
 * Cartesian BC the A.3 boundary helpers install for the dynamics fields.
 *
 * This file is linked by:
 *   - the production runtime via the standard SRCS list, and
 *   - the A.5 unit test via `tests/numerics/test_advection_cartesian.cpp`.
 *
 * It must therefore stay free of `#include`-side dependencies on the rest
 * of the advection module — no compute backend, no perf instrumentation,
 * no scheme factories.
 */

#include "numerics/advection_cartesian.hpp"
#include "core/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>

namespace
{

/**
 * @brief Resizes `field` to (NR, NTH, NZ) if it doesn't already match.
 *
 * Mirrors the inline helper of the same name in `src/advection/advection.cpp`.
 * Duplicated rather than shared so this file's link closure stays small.
 */
inline void ensure_field_shape(Field3D& field)
{
    if (field.size_r() != NR || field.size_th() != NTH || field.size_z() != NZ)
    {
        field.resize(NR, NTH, NZ, 0.0f);
    }
}

/**
 * @brief Computes the row-major flat index for a 3D cell.
 *
 * Matches the layout used by `Field3D::data()` — `(i * NTH + j) * NZ + k`.
 */
inline std::size_t idx3(int i, int j, int k)
{
    return (static_cast<std::size_t>(i) * static_cast<std::size_t>(NTH) +
            static_cast<std::size_t>(j)) *
               static_cast<std::size_t>(NZ) +
           static_cast<std::size_t>(k);
}

/**
 * @brief Seeds `dst` with the entire contents of `src`.
 *
 * The Cartesian advection and diffusion kernels only write the interior
 * (1..NR−2, 1..NTH−2, 1..NZ−2) and rely on the six face planes already
 * holding the source values for the zero-gradient boundary condition.
 * Doing the seeding as a single `memcpy` is cheaper than the cylindrical
 * helper's six per-plane copies and produces the same final state.
 */
inline void seed_destination_from_source(const Field3D& src, Field3D& dst)
{
    ensure_field_shape(dst);
    if (src.size() == 0)
    {
        return;
    }
    std::memcpy(dst.data(), src.data(), src.size() * sizeof(float));
}

}  // namespace

/**
 * @brief Cartesian first-order upwind advection in the x direction.
 */
void advect_scalar_1d_x_kernel_cartesian(const Field3D& src, Field3D& dst, double dt)
{
    seed_destination_from_source(src, dst);

    if (NR < 3 || NTH < 1 || NZ < 3)
    {
        // No interior to update — the seed already gives us the
        // zero-gradient extrapolation on every face. Match the
        // cylindrical kernel's "do nothing" behavior on degenerate grids.
        return;
    }

    const float* src_data = src.data();
    const float* u_data = u.data();
    float* dst_data = dst.data();

    const double dx = std::max(dr, 1.0e-12);
    const double inv_dx = 1.0 / dx;

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR - 1; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 1; k < NZ - 1; ++k)
            {
                const std::size_t c = idx3(i, j, k);
                const std::size_t im = idx3(i - 1, j, k);
                const std::size_t ip = idx3(i + 1, j, k);

                const double u_val = static_cast<double>(u_data[c]);
                double dq_dx = 0.0;
                if (u_val > 0.0)
                {
                    dq_dx = (static_cast<double>(src_data[c]) -
                             static_cast<double>(src_data[im])) *
                            inv_dx;
                }
                else if (u_val < 0.0)
                {
                    dq_dx = (static_cast<double>(src_data[ip]) -
                             static_cast<double>(src_data[c])) *
                            inv_dx;
                }

                const double q = static_cast<double>(src_data[c]);
                dst_data[c] = static_cast<float>(q - dt * u_val * dq_dx);
            }
        }
    }
}

/**
 * @brief Cartesian first-order upwind advection in the y direction.
 */
void advect_scalar_1d_y_kernel_cartesian(const Field3D& src, Field3D& dst, double dt)
{
    seed_destination_from_source(src, dst);

    if (NR < 1 || NTH < 3 || NZ < 3)
    {
        return;
    }

    const float* src_data = src.data();
    const float* v_data = v_theta.data();
    float* dst_data = dst.data();

    const double dy = std::max(dr, 1.0e-12);
    const double inv_dy = 1.0 / dy;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 1; j < NTH - 1; ++j)
        {
            for (int k = 1; k < NZ - 1; ++k)
            {
                const std::size_t c = idx3(i, j, k);
                const std::size_t jm = idx3(i, j - 1, k);
                const std::size_t jp = idx3(i, j + 1, k);

                const double v_val = static_cast<double>(v_data[c]);
                double dq_dy = 0.0;
                if (v_val > 0.0)
                {
                    dq_dy = (static_cast<double>(src_data[c]) -
                             static_cast<double>(src_data[jm])) *
                            inv_dy;
                }
                else if (v_val < 0.0)
                {
                    dq_dy = (static_cast<double>(src_data[jp]) -
                             static_cast<double>(src_data[c])) *
                            inv_dy;
                }

                const double q = static_cast<double>(src_data[c]);
                dst_data[c] = static_cast<float>(q - dt * v_val * dq_dy);
            }
        }
    }
}

/**
 * @brief Cartesian explicit diffusion (7-point Laplacian + Forward Euler).
 */
void apply_diffusion_kernel_cartesian(const Field3D& src, Field3D& dst, double dt, double kappa)
{
    if (kappa <= 0.0)
    {
        // Match the cylindrical helper's short-circuit: with κ = 0 the
        // diffusion is a no-op and the destination is just the source.
        seed_destination_from_source(src, dst);
        return;
    }

    seed_destination_from_source(src, dst);

    if (NR < 3 || NTH < 3 || NZ < 3)
    {
        return;
    }

    const float* src_data = src.data();
    float* dst_data = dst.data();

    const double dx = std::max(dr, 1.0e-12);
    const double dy = std::max(dr, 1.0e-12);
    const double dz_safe = std::max(dz, 1.0e-12);

    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz_safe * dz_safe);

    // The cylindrical diffusion kernel uses cache-blocked j-tiling to
    // keep the stencil neighborhood in L1/L2. The Cartesian path uses
    // the same TILE_J value for parity — at NZ = 128 and TILE_J = 32,
    // the working set is ≈ 3 × 34 × 128 × 4 = 52 KB per tile, which
    // fits comfortably in L2 on every target machine in the project's
    // accessibility band.
    static constexpr int TILE_J = 32;

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR - 1; ++i)
    {
        for (int jt = 1; jt < NTH - 1; jt += TILE_J)
        {
            const int j_end = std::min(jt + TILE_J, NTH - 1);

            for (int j = jt; j < j_end; ++j)
            {
                for (int k = 1; k < NZ - 1; ++k)
                {
                    const std::size_t c = idx3(i, j, k);
                    const std::size_t ip = idx3(i + 1, j, k);
                    const std::size_t im = idx3(i - 1, j, k);
                    const std::size_t kp = idx3(i, j, k + 1);
                    const std::size_t km = idx3(i, j, k - 1);
                    const std::size_t jp = idx3(i, j + 1, k);
                    const std::size_t jm = idx3(i, j - 1, k);

                    const double q = static_cast<double>(src_data[c]);
                    const double lap =
                        (static_cast<double>(src_data[ip]) - 2.0 * q +
                         static_cast<double>(src_data[im])) *
                            inv_dx2 +
                        (static_cast<double>(src_data[jp]) - 2.0 * q +
                         static_cast<double>(src_data[jm])) *
                            inv_dy2 +
                        (static_cast<double>(src_data[kp]) - 2.0 * q +
                         static_cast<double>(src_data[km])) *
                            inv_dz2;

                    dst_data[c] = static_cast<float>(q + dt * kappa * lap);
                }
            }
        }
    }
}
