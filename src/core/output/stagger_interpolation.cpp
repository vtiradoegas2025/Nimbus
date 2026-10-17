#include "core/output/stagger_interpolation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace
{

inline void ensure_field_shape(Field3D& field, const Field3D& source)
{
    if (field.size_r() != source.size_r() || field.size_th() != source.size_th() || field.size_z() != source.size_z())
    {
        field.resize(source.size_r(), source.size_th(), source.size_z(), 0.0f);
    }
}

void interpolate_w_face_to_center_impl(const Field3D&             w_face,
                                       Field3D&                   w_center,
                                       const std::vector<double>* lower_boundary_w)
{
    const int nr  = w_face.size_r();
    const int nth = w_face.size_th();
    const int nz  = w_face.size_z();
    if (lower_boundary_w != nullptr)
    {
        const std::size_t expected = static_cast<std::size_t>(nr) * static_cast<std::size_t>(nth);
        if (lower_boundary_w->size() != expected)
        {
            throw std::invalid_argument(
                "terrain lower-boundary velocity must have one value per horizontal source column");
        }
        if (!std::all_of(
                lower_boundary_w->begin(), lower_boundary_w->end(), [](double value) { return std::isfinite(value); }))
        {
            throw std::invalid_argument("terrain lower-boundary velocity must contain only finite values");
        }
    }

    ensure_field_shape(w_center, w_face);
    if (w_face.size() == 0)
    {
        return;
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < nr; ++i)
    {
        for (int j = 0; j < nth; ++j)
        {
            const double lower_surface_w =
                lower_boundary_w == nullptr
                    ? 0.0
                    : (*lower_boundary_w)[static_cast<std::size_t>(i) * static_cast<std::size_t>(nth) +
                                          static_cast<std::size_t>(j)];
            for (int k = 0; k < nz; ++k)
            {
                if (k == 0)
                {
                    w_center[i][j][0] =
                        static_cast<float>(0.5 * (lower_surface_w + static_cast<double>(w_face[i][j][0])));
                }
                else
                {
                    w_center[i][j][k] = 0.5f * (w_face[i][j][k - 1] + w_face[i][j][k]);
                }
            }
        }
    }
}

} // namespace

void interpolate_u_face_to_center(const Field3D& u_face, Field3D& u_center)
{
    ensure_field_shape(u_center, u_face);

    if (u_face.size() == 0)
    {
        return;
    }

    const int nr  = u_face.size_r();
    const int nth = u_face.size_th();
    const int nz  = u_face.size_z();
#pragma omp parallel for collapse(2)
    for (int i = 0; i < nr; ++i)
    {
        for (int j = 0; j < nth; ++j)
        {
            for (int k = 0; k < nz; ++k)
            {
                if (i == 0)
                {
                    // u_face[0] is the explicitly maintained west boundary
                    // face. Use it one-sided; do not invent an implicit zero
                    // that would halve open-boundary flow in exported data.
                    u_center[0][j][k] = u_face[0][j][k];
                }
                else
                {
                    u_center[i][j][k] = 0.5f * (u_face[i - 1][j][k] + u_face[i][j][k]);
                }
            }
        }
    }
}

void interpolate_v_face_to_center(const Field3D& v_face, Field3D& v_center)
{
    ensure_field_shape(v_center, v_face);

    if (v_face.size() == 0)
    {
        return;
    }

    const int nr  = v_face.size_r();
    const int nth = v_face.size_th();
    const int nz  = v_face.size_z();
#pragma omp parallel for collapse(2)
    for (int i = 0; i < nr; ++i)
    {
        for (int j = 0; j < nth; ++j)
        {
            for (int k = 0; k < nz; ++k)
            {
                if (j == 0)
                {
                    // v_face[0] is the maintained south boundary face. Never
                    // wrap to the north edge: Cartesian y boundaries can be
                    // open or rigid, and periodic ghost values (when selected)
                    // have already been applied by the BC scheme.
                    v_center[i][0][k] = v_face[i][0][k];
                }
                else
                {
                    v_center[i][j][k] = 0.5f * (v_face[i][j - 1][k] + v_face[i][j][k]);
                }
            }
        }
    }
}

void interpolate_w_face_to_center(const Field3D& w_face, Field3D& w_center)
{
    interpolate_w_face_to_center_impl(w_face, w_center, nullptr);
}

void interpolate_w_face_to_center(const Field3D& w_face, Field3D& w_center, const std::vector<double>& lower_boundary_w)
{
    interpolate_w_face_to_center_impl(w_face, w_center, &lower_boundary_w);
}
