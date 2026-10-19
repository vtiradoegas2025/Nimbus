#pragma once

#include "core/field/field3d.hpp"
#include "diagnostics/deterministic_reduction.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace tmv::collocated_velocity
{

namespace detail
{

template <typename DensityAt, typename VelocityAt, typename StoreVelocity>
inline double remove_alternating_line_null_mode(
    int count,
    DensityAt density_at,
    VelocityAt velocity_at,
    StoreVelocity store_velocity,
    bool attribute_committed_work = false)
{
    // Remove only the alternating residual that is orthogonal, in the
    // rho-weighted kinetic-energy inner product, to constant and linear
    // velocity. The latter two moments are therefore preserved exactly in
    // real arithmetic. This is the minimum-energy correction to the exact
    // face-average null vector; it has no tunable damping coefficient.
    if (count < 3) return 0.0;

    const double midpoint = 0.5 * static_cast<double>(count - 1);
    double s0 = 0.0;
    double s1 = 0.0;
    double s2 = 0.0;
    double t0 = 0.0;
    double t1 = 0.0;
    for (int n = 0; n < count; ++n)
    {
        const double mass = density_at(n);
        const double x = static_cast<double>(n) - midpoint;
        const double parity = (n & 1) == 0 ? 1.0 : -1.0;
        s0 += mass;
        s1 += mass * x;
        s2 += mass * x * x;
        t0 += mass * parity;
        t1 += mass * x * parity;
    }

    const double determinant = s0 * s2 - s1 * s1;
    if (!(determinant > 0.0) || !std::isfinite(determinant)) return 0.0;
    const double intercept = (t0 * s2 - t1 * s1) / determinant;
    const double slope = (t1 * s0 - t0 * s1) / determinant;

    double numerator = 0.0;
    double denominator = 0.0;
    for (int n = 0; n < count; ++n)
    {
        const double mass = density_at(n);
        const double x = static_cast<double>(n) - midpoint;
        const double parity = (n & 1) == 0 ? 1.0 : -1.0;
        const double null_residual = parity - intercept - slope * x;
        numerator += mass * null_residual * velocity_at(n);
        denominator += mass * null_residual * null_residual;
    }

    if (!(denominator > 0.0) || !std::isfinite(denominator)
        || !std::isfinite(numerator))
    {
        return 0.0;
    }
    const double amplitude = numerator / denominator;
    if (amplitude == 0.0) return 0.0;

    double committed_work_density = 0.0;
    for (int n = 0; n < count; ++n)
    {
        const double x = static_cast<double>(n) - midpoint;
        const double parity = (n & 1) == 0 ? 1.0 : -1.0;
        const double null_residual = parity - intercept - slope * x;
        const double old_velocity = velocity_at(n);
        store_velocity(n, old_velocity - amplitude * null_residual);
        if (attribute_committed_work)
        {
            const double stored_velocity = velocity_at(n);
            committed_work_density += 0.5 * density_at(n)
                * (stored_velocity * stored_velocity
                   - old_velocity * old_velocity);
        }
    }
    return committed_work_density;
}

} // namespace detail

/**
 * Remove selected exact directional velocity null spaces of a collocated grid.
 *
 * An alternating cell velocity is annihilated by adjacent-cell face
 * averaging, so continuity cannot observe or restore it. For each physical
 * physical line, this routine removes the rho-weighted alternating residual
 * after its best constant/linear fit. The operation preserves the line's
 * rho-weighted constant and linear moments in real arithmetic and is an
 * orthogonal projection, so it cannot increase discrete kinetic energy before
 * FP32 storage rounding. Ghost cells are neither sampled nor written.
 *
 * The caller must close velocity boundaries after this projection.
 */
inline void constrain_directional_null_modes(
    Field3D& u,
    Field3D& v,
    Field3D& w,
    const Field3D& rho,
    int nr,
    int nth,
    int nz,
    bool constrain_x,
    bool constrain_y,
    bool constrain_z,
    std::array<const Field3D*, 7> water_species = {},
    double cell_volume_m3 = 0.0,
    double* committed_mixture_kinetic_work_j = nullptr)
{
    const auto matches = [=](const Field3D& field)
    {
        return field.size_r() == nr && field.size_th() == nth
            && field.size_z() == nz;
    };
    const bool have_any_water = std::any_of(
        water_species.begin(), water_species.end(),
        [](const Field3D* field) { return field != nullptr; });
    const bool have_all_water = std::all_of(
        water_species.begin(), water_species.end(),
        [&](const Field3D* field)
        {
            return field != nullptr && matches(*field);
        });
    if (nr < 4 || nth < 4 || nz < 4
        || !matches(u) || !matches(v) || !matches(w) || !matches(rho))
    {
        throw std::invalid_argument(
            "collocated velocity constraint requires matching fields on a grid of at least 4x4x4");
    }
    if (have_any_water && !have_all_water)
    {
        throw std::invalid_argument(
            "collocated velocity constraint requires all seven matching water fields or none");
    }
    if (!constrain_x && !constrain_y && !constrain_z)
    {
        if (committed_mixture_kinetic_work_j != nullptr)
            *committed_mixture_kinetic_work_j = 0.0;
        return;
    }
    if (committed_mixture_kinetic_work_j != nullptr
        && (!std::isfinite(cell_volume_m3) || cell_volume_m3 <= 0.0))
    {
        throw std::invalid_argument(
            "attributed collocated velocity constraint requires positive finite cell volume");
    }

    // Validate before the first write so malformed direct calls are
    // transactional. Runtime state guards normally make this pass redundant,
    // but the operator boundary must remain safe independently.
    for (std::size_t n = 0; n < rho.size(); ++n)
    {
        if (!std::isfinite(static_cast<double>(rho.data()[n]))
            || rho.data()[n] <= 0.0f
            || !std::isfinite(static_cast<double>(u.data()[n]))
            || !std::isfinite(static_cast<double>(v.data()[n]))
            || !std::isfinite(static_cast<double>(w.data()[n])))
        {
            throw std::invalid_argument(
                "collocated velocity constraint requires finite velocity and strictly positive finite density");
        }
        if (have_all_water)
        {
            for (const Field3D* field : water_species)
            {
                const double value =
                    static_cast<double>(field->data()[n]);
                if (!std::isfinite(value) || value < 0.0)
                {
                    throw std::invalid_argument(
                        "collocated velocity constraint requires finite nonnegative water");
                }
            }
        }
    }
    const auto mixture_density_at = [&](int i, int j, int k)
    {
        double factor = 1.0;
        if (have_all_water)
        {
            for (const Field3D* field : water_species)
                factor += static_cast<double>((*field)[i][j][k]);
        }
        return static_cast<double>(rho[i][j][k]) * factor;
    };
    const int nx_physical = nr - 2;
    const int ny_physical = nth - 2;
    const int nz_physical = nz - 2;

    if (committed_mixture_kinetic_work_j != nullptr)
    {
        double committed_work_density = 0.0;
        if (constrain_x)
        {
            committed_work_density +=
                tmv::diagnostics::fixed_slab_reduce<1>(
                    1, nth - 1,
                    [&](int j, std::array<double, 1>& partial)
                    {
                        for (int k = 1; k < nz - 1; ++k)
                            partial[0] +=
                                detail::remove_alternating_line_null_mode(
                                    nx_physical,
                                    [&](int n) {
                                        return mixture_density_at(
                                            n + 1, j, k);
                                    },
                                    [&](int n) {
                                        return static_cast<double>(
                                            u[n + 1][j][k]);
                                    },
                                    [&](int n, double value) {
                                        u[n + 1][j][k] =
                                            static_cast<float>(value);
                                    },
                                    true);
                    })[0];
        }

        if (constrain_y)
        {
            committed_work_density +=
                tmv::diagnostics::fixed_slab_reduce<1>(
                    1, nr - 1,
                    [&](int i, std::array<double, 1>& partial)
                    {
                        for (int k = 1; k < nz - 1; ++k)
                            partial[0] +=
                                detail::remove_alternating_line_null_mode(
                                    ny_physical,
                                    [&](int n) {
                                        return mixture_density_at(
                                            i, n + 1, k);
                                    },
                                    [&](int n) {
                                        return static_cast<double>(
                                            v[i][n + 1][k]);
                                    },
                                    [&](int n, double value) {
                                        v[i][n + 1][k] =
                                            static_cast<float>(value);
                                    },
                                    true);
                    })[0];
        }

        if (constrain_z)
        {
            committed_work_density +=
                tmv::diagnostics::fixed_slab_reduce<1>(
                    1, nr - 1,
                    [&](int i, std::array<double, 1>& partial)
                    {
                        for (int j = 1; j < nth - 1; ++j)
                            partial[0] +=
                                detail::remove_alternating_line_null_mode(
                                    nz_physical,
                                    [&](int n) {
                                        return mixture_density_at(
                                            i, j, n + 1);
                                    },
                                    [&](int n) {
                                        return static_cast<double>(
                                            w[i][j][n + 1]);
                                    },
                                    [&](int n, double value) {
                                        w[i][j][n + 1] =
                                            static_cast<float>(value);
                                    },
                                    true);
                    })[0];
        }
        const double committed =
            committed_work_density * cell_volume_m3;
        if (!std::isfinite(committed))
        {
            throw std::overflow_error(
                "collocated velocity constraint energy attribution overflowed");
        }
        *committed_mixture_kinetic_work_j = committed;
        return;
    }

    #pragma omp parallel
    {
        if (constrain_x)
        {
            #pragma omp for collapse(2)
            for (int j = 1; j < nth - 1; ++j)
                for (int k = 1; k < nz - 1; ++k)
                    detail::remove_alternating_line_null_mode(
                        nx_physical,
                        [&](int n) {
                            return mixture_density_at(n + 1, j, k);
                        },
                        [&](int n) { return static_cast<double>(u[n + 1][j][k]); },
                        [&](int n, double value) {
                            u[n + 1][j][k] = static_cast<float>(value);
                        });
        }

        if (constrain_y)
        {
            #pragma omp for collapse(2)
            for (int i = 1; i < nr - 1; ++i)
                for (int k = 1; k < nz - 1; ++k)
                    detail::remove_alternating_line_null_mode(
                        ny_physical,
                        [&](int n) {
                            return mixture_density_at(i, n + 1, k);
                        },
                        [&](int n) { return static_cast<double>(v[i][n + 1][k]); },
                        [&](int n, double value) {
                            v[i][n + 1][k] = static_cast<float>(value);
                        });
        }

        if (constrain_z)
        {
            #pragma omp for collapse(2)
            for (int i = 1; i < nr - 1; ++i)
                for (int j = 1; j < nth - 1; ++j)
                    detail::remove_alternating_line_null_mode(
                        nz_physical,
                        [&](int n) {
                            return mixture_density_at(i, j, n + 1);
                        },
                        [&](int n) { return static_cast<double>(w[i][j][n + 1]); },
                        [&](int n, double value) {
                            w[i][j][n + 1] = static_cast<float>(value);
                        });
        }
    }
}

} // namespace tmv::collocated_velocity
