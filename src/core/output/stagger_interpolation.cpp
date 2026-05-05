/**
 * @file stagger_interpolation.cpp
 * @brief Implementations of face-to-center velocity interpolation
 *        helpers used by the output paths (Phase C.8 of
 *        docs/CoordinateBackend_Plan.md).
 *
 * Each helper produces a cell-center Field3D from a face-staggered
 * Field3D using arithmetic averaging. The conventions match
 * `StaggeredCylindricalDerivatives::interp_from_r_face` /
 * `interp_from_theta_face` / `interp_from_z_face` already used by the
 * dynamics diagnostics in
 * `SupercellCGridScheme::compute_vorticity_diagnostics`, so output
 * cell-center velocities and diagnostic cell-center velocities derived
 * from the same prognostic field agree exactly.
 *
 * Output cost is one Field3D::size() reads + one Field3D::size() writes
 * per call -- O(NR*NTH*NZ). For typical grids this is microseconds and
 * is amortized into the export step rather than the inner simulation
 * loop.
 */

#include "core/output/stagger_interpolation.hpp"
#include "core/simulation.hpp"


namespace
{

/**
 * @brief Resizes @p field to (NR, NTH, NZ) if it doesn't already match.
 *
 * Mirrors the inline helpers used in the advection module so the
 * output path makes no assumption about the destination buffer's
 * prior shape.
 */
inline void ensure_field_shape(Field3D& field)
{
    if (field.size_r() != NR || field.size_th() != NTH || field.size_z() != NZ)
    {
        field.resize(NR, NTH, NZ, 0.0f);
    }
}

}  // namespace


void interpolate_u_face_to_center(const Field3D& u_face, Field3D& u_center)
{
    ensure_field_shape(u_center);

    if (u_face.size() == 0)
    {
        return;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                if (i == 0)
                {
                    // Axis cell: only the right face (r_face[0]) exists.
                    // The implicit left face is the axis r=0 with u=0
                    // (no flow through the singular axis), so the
                    // arithmetic mean is 0.5 * (0 + u[0]) = 0.5 * u[0].
                    u_center[0][j][k] = 0.5f * u_face[0][j][k];
                }
                else
                {
                    u_center[i][j][k] =
                        0.5f * (u_face[i - 1][j][k] + u_face[i][j][k]);
                }
            }
        }
    }
}


void interpolate_v_face_to_center(const Field3D& v_face, Field3D& v_center)
{
    ensure_field_shape(v_center);

    if (v_face.size() == 0)
    {
        return;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            const int j_prev = (j - 1 + NTH) % NTH;
            for (int k = 0; k < NZ; ++k)
            {
                v_center[i][j][k] =
                    0.5f * (v_face[i][j_prev][k] + v_face[i][j][k]);
            }
        }
    }
}


void interpolate_w_face_to_center(const Field3D& w_face, Field3D& w_center)
{
    ensure_field_shape(w_center);

    if (w_face.size() == 0)
    {
        return;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                if (k == 0)
                {
                    // Surface cell: only the top face (z_face[0]) exists.
                    // The implicit bottom face is the rigid surface
                    // (z = 0 below z_face[0]) with w=0, so the
                    // arithmetic mean is 0.5 * (0 + w[0]) = 0.5 * w[0].
                    w_center[i][j][0] = 0.5f * w_face[i][j][0];
                }
                else
                {
                    w_center[i][j][k] =
                        0.5f * (w_face[i][j][k - 1] + w_face[i][j][k]);
                }
            }
        }
    }
}
