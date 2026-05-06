/**
 * @file advection_cylindrical_cgrid.cpp
 * @brief Cylindrical C-grid scalar advection kernel implementations
 *        (Phase C.7 of docs/CoordinateBackend_Plan.md).
 *
 * Each kernel implements TVD-MUSCL flux-form scalar advection in one
 * direction with Forward-Euler integration:
 *
 *     q^{n+1}[i][j][k] = q^n[i][j][k] - dt * div_F
 *
 * where div_F is the cylindrical control-volume flux divergence:
 *
 *     div_F_r[i]     = (r_face[i]*F_right - r_face[i-1]*F_left ) / (r[i] * dr)
 *     div_F_theta[j] = (F_north - F_south) / (r[i] * dtheta)
 *     div_F_z[k]     = (F_top - F_bot) / dz
 *
 * Field placement (Arakawa C-grid):
 *
 *     u[i][j][k] at (r_face[i], theta[j],      z[k])      -- right face of cell i
 *     v[i][j][k] at (r[i],      theta_{j+1/2}, z[k])      -- "north" face of cell j
 *     w[i][j][k] at (r[i],      theta[j],      z_face[k]) -- top face of cell k
 *     q[i][j][k] at (r[i],      theta[j],      z[k])      -- cell center
 *
 * MUSCL reconstruction: at every cell-center i, compute a limited slope
 * from neighbors (i-1, i, i+1) using the MC limiter (matching the
 * existing TVDScheme default in src/numerics/advection/schemes/tvd):
 *
 *     phi(r) = max(0, min((1+r)/2, 2, 2r))
 *     slope[i] = phi(r_i) * (q[i+1] - q[i]),   r_i = (q[i] - q[i-1]) / (q[i+1] - q[i])
 *
 * Face values are reconstructed by linear extrapolation:
 *
 *     q_at_face_from_left_cell[i+1/2]  = q[i]   + 0.5 * slope[i]
 *     q_at_face_from_right_cell[i+1/2] = q[i+1] - 0.5 * slope[i+1]
 *
 * The upwind face value is then chosen by the sign of the face velocity:
 *
 *     q_face = u_face >= 0 ? q_at_face_from_left_cell : q_at_face_from_right_cell
 *     flux   = u_face * q_face
 *
 * Mass conservation: the flux at every shared face (r_face[i],
 * theta_{j+1/2}, z_face[k]) is evaluated ONCE and used identically by
 * the two cells on either side. The mass debit at cell i and the mass
 * credit at cell i+1 cancel bit-exactly in floating point, so total
 * interior scalar mass is preserved to floating-point precision when
 * the boundary face fluxes vanish (axis u=0, outer wall u=0,
 * surface w=0, lid w=0).
 *
 * Boundary cells: slopes at i=0 and i=NR-1 (in r) and k=0 and k=NZ-1
 * (in z) are taken as ZERO -- the kernel falls back to first-order
 * upwind at those faces. The interior compute domain
 * (i = 1..NR-2, k = 1..NZ-2) is what the kernel actually updates;
 * cells at i=0 / i=NR-1 / k=0 / k=NZ-1 retain their source values
 * for the BC scheme to overwrite.
 *
 * Periodic theta: slopes are well-defined at every j because all three
 * neighbors are valid via the (j +/- 1) % NTH wrap.
 *
 * No GPU dispatch is attempted from these kernels: the collocated
 * cylindrical advection shaders read u, v, w as if they lived at cell
 * centers, which would silently corrupt staggered velocity reads.
 * Phase C.9 will provide stagger-aware compute shaders.
 *
 * Diffusion: the cylindrical-collocated apply_diffusion_kernel in
 * advection.cpp is reused unchanged because it operates only on
 * cell-center scalars and does not read u, v, or w; its math (centered
 * Laplacian with the 1/(r^2 dtheta^2) factor) is identical for
 * collocated and C-grid scalar storage.
 */

#include "numerics/advection/advection_cylindrical_cgrid.hpp"
#include "compute/compute_kernel_template.hpp"
#include "core/infra/grid_geometry.hpp"
#include "core/runtime/simulation.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

namespace
{

/**
 * @brief Resizes @p field to (NR, NTH, NZ) if it doesn't already match.
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
 * Matches the layout used by Field3D::data() -- (i * NTH + j) * NZ + k.
 */
inline std::size_t idx3(int i, int j, int k)
{
    return (static_cast<std::size_t>(i) * static_cast<std::size_t>(NTH) +
            static_cast<std::size_t>(j)) *
               static_cast<std::size_t>(NZ) +
           static_cast<std::size_t>(k);
}

/**
 * @brief Seeds @p dst with the entire contents of @p src.
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

/**
 * @brief MC (monotonized central) limiter, matching the default in
 *        src/numerics/advection/schemes/tvd/tvd.cpp::mc_limiter.
 *
 *        phi(r) = max(0, min((1 + r) / 2, 2, 2 r))
 *
 * Inputs of r >= 0 produce phi in [0, 2] with phi(1) = 1 (no limiting
 * at smooth extrema). Inputs r < 0 (sign change in the slope ratio
 * indicating a local extremum) produce phi = 0 -- the slope is fully
 * limited to zero, falling back to first-order upwind at extrema.
 */
inline double mc_limiter(double r)
{
    return std::max(0.0,
                    std::min({0.5 * (1.0 + r), 2.0, 2.0 * r}));
}

/**
 * @brief Returns the MC-limited cell-centered slope `phi(r) * (q_plus - q_center)`.
 *
 * @param q_minus   value at the previous cell
 * @param q_center  value at this cell
 * @param q_plus    value at the next cell
 *
 * The slope returned is the discrete centered slope after limiting.
 * Reconstruction uses
 *   q_face_left  = q_center - 0.5 * slope
 *   q_face_right = q_center + 0.5 * slope
 *
 * The numerator (q_plus - q_center) is the FORWARD difference; the
 * denominator (q_plus - q_center) appears in r itself, so we add a
 * sign-preserving epsilon to avoid 0/0 at smooth extrema. When the
 * forward difference is exactly zero (or denormal) the limited slope
 * collapses to zero (no extrapolation), which is exactly what the
 * monotonicity preservation requires.
 */
inline double limited_slope(double q_minus, double q_center, double q_plus)
{
    const double forward  = q_plus - q_center;
    const double backward = q_center - q_minus;

    // Sign-preserving epsilon -- matches the convention in TVDScheme::muscl_reconstruct
    // (uses numerics_constants::epsilon as a one-sided bias). Here we
    // pick the sign so that division by `forward + eps_signed` is
    // never amplified in absolute value.
    constexpr double kTinyEpsilon = 1.0e-30;
    const double denom = (forward >= 0.0) ? (forward + kTinyEpsilon)
                                          : (forward - kTinyEpsilon);

    const double r   = backward / denom;
    const double phi = mc_limiter(r);
    return phi * forward;
}

}  // namespace


// ============================================================================
// Radial direction
// ============================================================================
void advect_scalar_1d_r_kernel_cylindrical_cgrid(const Field3D& src, Field3D& dst, double dt)
{
    seed_destination_from_source(src, dst);

    if (NR < 3 || NTH < 1 || NZ < 3)
    {
        return;
    }

    // Phase C.9 GPU dispatch: the boundary cells are already seeded
    // from src via seed_destination_from_source above, so the device
    // only needs to write the interior cells.
    if (dispatch_radial_advection_cgrid_backend(
            src.data(), u.data(), dst.data(),
            NR, NTH, NZ,
            static_cast<float>(dr), static_cast<float>(dt)))
    {
        return;
    }

    const float* src_data = src.data();
    const float* u_data   = u.data();
    float*       dst_data = dst.data();

    const auto&  geo    = global_grid_geometry;
    const double inv_dr = geo.inv_dr;

    #pragma omp parallel
    {
        // Per-thread slope buffer along the r direction. Slopes at the
        // boundary cells (i = 0 and i = NR - 1) stay at 0.0 -- the
        // boundary faces fall back to first-order upwind.
        std::vector<double> slope_r(static_cast<std::size_t>(NR), 0.0);

        #pragma omp for collapse(2)
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 1; k < NZ - 1; ++k)
            {
                // Pre-compute the MC-limited slope at every interior cell.
                for (int i = 1; i < NR - 1; ++i)
                {
                    const std::size_t cm = idx3(i - 1, j, k);
                    const std::size_t cc = idx3(i,     j, k);
                    const std::size_t cp = idx3(i + 1, j, k);
                    slope_r[static_cast<std::size_t>(i)] =
                        limited_slope(static_cast<double>(src_data[cm]),
                                      static_cast<double>(src_data[cc]),
                                      static_cast<double>(src_data[cp]));
                }
                slope_r[0]        = 0.0;
                slope_r[NR - 1]   = 0.0;

                // Apply tendencies at every interior cell using the
                // pre-computed slopes.
                for (int i = 1; i < NR - 1; ++i)
                {
                    const double r_inv_i      = geo.r_inv[i];
                    const double r_face_outer = geo.r_face[i];
                    const double r_face_inner = geo.r_face[i - 1];

                    const std::size_t cm = idx3(i - 1, j, k);
                    const std::size_t cc = idx3(i,     j, k);
                    const std::size_t cp = idx3(i + 1, j, k);

                    // Right face at r_face[i]: q reconstructed from cell i (left side)
                    // and cell i+1 (right side); pick by sign of u[i][j][k].
                    const double u_right = static_cast<double>(u_data[cc]);
                    const double q_right_from_left =
                        static_cast<double>(src_data[cc]) + 0.5 * slope_r[static_cast<std::size_t>(i)];
                    const double q_right_from_right =
                        static_cast<double>(src_data[cp]) - 0.5 * slope_r[static_cast<std::size_t>(i + 1)];
                    const double q_right_face =
                        (u_right >= 0.0) ? q_right_from_left : q_right_from_right;
                    const double F_right = u_right * q_right_face;

                    // Left face at r_face[i-1]: same construction with
                    // the slope of cells i-1 and i.
                    const double u_left = static_cast<double>(u_data[cm]);
                    const double q_left_from_left =
                        static_cast<double>(src_data[cm]) + 0.5 * slope_r[static_cast<std::size_t>(i - 1)];
                    const double q_left_from_right =
                        static_cast<double>(src_data[cc]) - 0.5 * slope_r[static_cast<std::size_t>(i)];
                    const double q_left_face =
                        (u_left >= 0.0) ? q_left_from_left : q_left_from_right;
                    const double F_left = u_left * q_left_face;

                    const double div_r = (r_face_outer * F_right - r_face_inner * F_left)
                                         * inv_dr * r_inv_i;

                    const double q_old = static_cast<double>(src_data[cc]);
                    dst_data[cc] = static_cast<float>(q_old - dt * div_r);
                }
            }
        }
    }
}


// ============================================================================
// Azimuthal direction (periodic in j)
// ============================================================================
void advect_scalar_1d_theta_kernel_cylindrical_cgrid(const Field3D& src, Field3D& dst, double dt)
{
    seed_destination_from_source(src, dst);

    if (NR < 3 || NTH < 2 || NZ < 3)
    {
        return;
    }

    // Phase C.9 GPU dispatch: boundary cells in r and z are seeded
    // by seed_destination_from_source above; theta is fully periodic
    // and every j is an interior point.
    if (dispatch_azimuthal_advection_cgrid_backend(
            src.data(), v.data(), dst.data(),
            NR, NTH, NZ,
            static_cast<float>(dr), static_cast<float>(dtheta),
            static_cast<float>(dt)))
    {
        return;
    }

    const float* src_data = src.data();
    const float* v_data   = v.data();
    float*       dst_data = dst.data();

    const auto&  geo        = global_grid_geometry;
    const double inv_dtheta = geo.inv_dtheta;

    #pragma omp parallel
    {
        std::vector<double> slope_th(static_cast<std::size_t>(NTH), 0.0);

        #pragma omp for collapse(2)
        for (int i = 1; i < NR - 1; ++i)
        {
            for (int k = 1; k < NZ - 1; ++k)
            {
                // Pre-compute the MC-limited slope at every j (periodic).
                for (int j = 0; j < NTH; ++j)
                {
                    const int j_prev = (j - 1 + NTH) % NTH;
                    const int j_next = (j + 1) % NTH;
                    const std::size_t cm = idx3(i, j_prev, k);
                    const std::size_t cc = idx3(i, j,      k);
                    const std::size_t cp = idx3(i, j_next, k);
                    slope_th[static_cast<std::size_t>(j)] =
                        limited_slope(static_cast<double>(src_data[cm]),
                                      static_cast<double>(src_data[cc]),
                                      static_cast<double>(src_data[cp]));
                }

                const double r_inv_i = geo.r_inv[i];

                for (int j = 0; j < NTH; ++j)
                {
                    const int j_prev = (j - 1 + NTH) % NTH;
                    const int j_next = (j + 1) % NTH;

                    const std::size_t cm = idx3(i, j_prev, k);
                    const std::size_t cc = idx3(i, j,      k);
                    const std::size_t cp = idx3(i, j_next, k);

                    // North face at theta_{j+1/2}: v[i][j] is the face
                    // velocity. Upwind cell is j (south side) or j+1
                    // (north side) depending on sign.
                    const double v_north = static_cast<double>(v_data[cc]);
                    const double q_north_from_below =
                        static_cast<double>(src_data[cc]) + 0.5 * slope_th[static_cast<std::size_t>(j)];
                    const double q_north_from_above =
                        static_cast<double>(src_data[cp]) - 0.5 * slope_th[static_cast<std::size_t>(j_next)];
                    const double q_north_face =
                        (v_north >= 0.0) ? q_north_from_below : q_north_from_above;
                    const double F_north = v_north * q_north_face;

                    // South face at theta_{j-1/2}: v[i][j-1] is the face
                    // velocity.
                    const double v_south = static_cast<double>(v_data[cm]);
                    const double q_south_from_below =
                        static_cast<double>(src_data[cm]) + 0.5 * slope_th[static_cast<std::size_t>(j_prev)];
                    const double q_south_from_above =
                        static_cast<double>(src_data[cc]) - 0.5 * slope_th[static_cast<std::size_t>(j)];
                    const double q_south_face =
                        (v_south >= 0.0) ? q_south_from_below : q_south_from_above;
                    const double F_south = v_south * q_south_face;

                    const double div_th = (F_north - F_south) * inv_dtheta * r_inv_i;

                    const double q_old = static_cast<double>(src_data[cc]);
                    dst_data[cc] = static_cast<float>(q_old - dt * div_th);
                }
            }
        }
    }
}


// ============================================================================
// Vertical direction
// ============================================================================
void advect_scalar_1d_z_kernel_cylindrical_cgrid(const Field3D& src, Field3D& dst, double dt)
{
    seed_destination_from_source(src, dst);

    if (NR < 3 || NTH < 1 || NZ < 3)
    {
        return;
    }

    // Phase C.9 GPU dispatch: boundary cells are seeded above, the
    // device only writes interior k.
    if (dispatch_vertical_advection_cgrid_backend(
            src.data(), w.data(), dst.data(),
            NR, NTH, NZ,
            static_cast<float>(dz), static_cast<float>(dt)))
    {
        return;
    }

    const float* src_data = src.data();
    const float* w_data   = w.data();
    float*       dst_data = dst.data();

    const auto&  geo    = global_grid_geometry;
    const double inv_dz = geo.inv_dz;

    #pragma omp parallel
    {
        std::vector<double> slope_z(static_cast<std::size_t>(NZ), 0.0);

        #pragma omp for collapse(2)
        for (int i = 1; i < NR - 1; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                // Pre-compute the MC-limited slope at every interior k.
                for (int k = 1; k < NZ - 1; ++k)
                {
                    const std::size_t cm = idx3(i, j, k - 1);
                    const std::size_t cc = idx3(i, j, k    );
                    const std::size_t cp = idx3(i, j, k + 1);
                    slope_z[static_cast<std::size_t>(k)] =
                        limited_slope(static_cast<double>(src_data[cm]),
                                      static_cast<double>(src_data[cc]),
                                      static_cast<double>(src_data[cp]));
                }
                slope_z[0]        = 0.0;
                slope_z[NZ - 1]   = 0.0;

                for (int k = 1; k < NZ - 1; ++k)
                {
                    const std::size_t cm = idx3(i, j, k - 1);
                    const std::size_t cc = idx3(i, j, k    );
                    const std::size_t cp = idx3(i, j, k + 1);

                    // Top face at z_face[k]: w[i][j][k].
                    const double w_top = static_cast<double>(w_data[cc]);
                    const double q_top_from_below =
                        static_cast<double>(src_data[cc]) + 0.5 * slope_z[static_cast<std::size_t>(k)];
                    const double q_top_from_above =
                        static_cast<double>(src_data[cp]) - 0.5 * slope_z[static_cast<std::size_t>(k + 1)];
                    const double q_top_face =
                        (w_top >= 0.0) ? q_top_from_below : q_top_from_above;
                    const double F_top = w_top * q_top_face;

                    // Bottom face at z_face[k-1]: w[i][j][k-1].
                    const double w_bot = static_cast<double>(w_data[cm]);
                    const double q_bot_from_below =
                        static_cast<double>(src_data[cm]) + 0.5 * slope_z[static_cast<std::size_t>(k - 1)];
                    const double q_bot_from_above =
                        static_cast<double>(src_data[cc]) - 0.5 * slope_z[static_cast<std::size_t>(k)];
                    const double q_bot_face =
                        (w_bot >= 0.0) ? q_bot_from_below : q_bot_from_above;
                    const double F_bot = w_bot * q_bot_face;

                    const double div_z = (F_top - F_bot) * inv_dz;

                    const double q_old = static_cast<double>(src_data[cc]);
                    dst_data[cc] = static_cast<float>(q_old - dt * div_z);
                }
            }
        }
    }
}
