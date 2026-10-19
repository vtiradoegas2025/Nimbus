/**
 * @file cartesian_horizontal_derivatives.hpp
 * @brief Boundary-clamped first derivatives and flow divergence on a Cartesian
 *        C-grid, for the headless diagnostic export path.
 *
 * The diagnostic loop in headless_runtime uses the retained-alias index slots:
 * the `dr` slot carries `dx` and the `dtheta` slot carries `dy` (both in
 * metres). These helpers compute the plain Cartesian first derivatives and
 * divergence on those slots.
 *
 * Differencing convention:
 *  - central difference in the interior, one-sided at a domain edge (the
 *    neighbour index is clamped, never wrapped);
 *  - denominator is 2*h in the interior and h at either edge, so a one-sided
 *    edge difference stays first-order consistent;
 *  - returns NaN if any required neighbour is non-finite or the spacing is not
 *    positive, so a bad cell propagates to NaN rather than a silent 0.
 */
#ifndef TMV_DIAGNOSTICS_CARTESIAN_HORIZONTAL_DERIVATIVES_HPP
#define TMV_DIAGNOSTICS_CARTESIAN_HORIZONTAL_DERIVATIVES_HPP

#include <cmath>
#include <limits>

#include "core/field/field3d.hpp"

namespace tmv
{

/// df/dx at (i, th, k): central in i, clamped one-sided at i == 0 / i == NR-1.
inline float cartesian_ddx(const Field3D& f, int i, int th, int k, int NR, double dx)
{
    if (!(dx > 0.0))
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const int ip = (i + 1 < NR) ? i + 1 : i;
    const int im = (i > 0) ? i - 1 : i;
    const float fp = f(ip, th, k);
    const float fm = f(im, th, k);
    if (!std::isfinite(fp) || !std::isfinite(fm))
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float h = static_cast<float>(dx);
    const float denom = (i == 0 || i == NR - 1) ? h : (2.0f * h);
    return (fp - fm) / denom;
}

/// df/dy at (i, th, k): central in the th (Cartesian y) index, clamped one-sided
/// at th == 0 / th == NTH-1. `dy` is carried in the dtheta slot.
inline float cartesian_ddy(const Field3D& f, int i, int th, int k, int NTH, double dy)
{
    if (!(dy > 0.0))
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const int yp = (th + 1 < NTH) ? th + 1 : th;
    const int ym = (th > 0) ? th - 1 : th;
    const float fp = f(i, yp, k);
    const float fm = f(i, ym, k);
    if (!std::isfinite(fp) || !std::isfinite(fm))
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float h = static_cast<float>(dy);
    const float denom = (th == 0 || th == NTH - 1) ? h : (2.0f * h);
    return (fp - fm) / denom;
}

/// d2f/dy2 at (i, th, k): centered in the interior and one-sided at the
/// physical y boundaries. The four-point edge stencil is second-order
/// accurate; a three-point forward/backward stencil is used on tiny grids.
inline float cartesian_d2dy2(const Field3D& f, int i, int th, int k,
                             int NTH, double dy)
{
    if (!(dy > 0.0) || NTH < 3 || th < 0 || th >= NTH)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    const float inv_dy2 = 1.0f /
        (static_cast<float>(dy) * static_cast<float>(dy));
    if (th == 0)
    {
        const float f0 = f(i, 0, k);
        const float f1 = f(i, 1, k);
        const float f2 = f(i, 2, k);
        if (!std::isfinite(f0) || !std::isfinite(f1) || !std::isfinite(f2))
            return std::numeric_limits<float>::quiet_NaN();
        if (NTH >= 4)
        {
            const float f3 = f(i, 3, k);
            if (!std::isfinite(f3)) return std::numeric_limits<float>::quiet_NaN();
            return (2.0f * f0 - 5.0f * f1 + 4.0f * f2 - f3) * inv_dy2;
        }
        return (f0 - 2.0f * f1 + f2) * inv_dy2;
    }
    if (th == NTH - 1)
    {
        const float f0 = f(i, NTH - 1, k);
        const float f1 = f(i, NTH - 2, k);
        const float f2 = f(i, NTH - 3, k);
        if (!std::isfinite(f0) || !std::isfinite(f1) || !std::isfinite(f2))
            return std::numeric_limits<float>::quiet_NaN();
        if (NTH >= 4)
        {
            const float f3 = f(i, NTH - 4, k);
            if (!std::isfinite(f3)) return std::numeric_limits<float>::quiet_NaN();
            return (2.0f * f0 - 5.0f * f1 + 4.0f * f2 - f3) * inv_dy2;
        }
        return (f0 - 2.0f * f1 + f2) * inv_dy2;
    }

    const float fp = f(i, th + 1, k);
    const float fc = f(i, th, k);
    const float fm = f(i, th - 1, k);
    if (!std::isfinite(fp) || !std::isfinite(fc) || !std::isfinite(fm))
        return std::numeric_limits<float>::quiet_NaN();
    return (fp - 2.0f * fc + fm) * inv_dy2;
}

/// df/dz at (i, th, k): central in k, clamped one-sided at k == 0 / k == NZ-1.
inline float cartesian_ddz(const Field3D& f, int i, int th, int k, int NZ, double dz)
{
    if (!(dz > 0.0))
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const int kp = (k + 1 < NZ) ? k + 1 : k;
    const int km = (k > 0) ? k - 1 : k;
    const float fp = f(i, th, kp);
    const float fm = f(i, th, km);
    if (!std::isfinite(fp) || !std::isfinite(fm))
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float h = static_cast<float>(dz);
    const float denom = (k == 0 || k == NZ - 1) ? h : (2.0f * h);
    return (fp - fm) / denom;
}

/// Flow divergence du/dx + dv/dy + dw/dz at (i, th, k) on a Cartesian grid.
/// `dx`, `dy`, `dz` are the metre spacings (dy carried in the dtheta slot).
/// Returns NaN if any component derivative is non-finite.
inline float cartesian_flow_divergence(int i, int th, int k,
                                       int NR, int NTH, int NZ,
                                       double dx, double dy, double dz,
                                       const Field3D& u, const Field3D& v, const Field3D& w)
{
    const float du_dx = cartesian_ddx(u, i, th, k, NR, dx);
    const float dv_dy = cartesian_ddy(v, i, th, k, NTH, dy);
    const float dw_dz = cartesian_ddz(w, i, th, k, NZ, dz);
    if (!std::isfinite(du_dx) || !std::isfinite(dv_dy) || !std::isfinite(dw_dz))
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return du_dx + dv_dy + dw_dz;
}

}  // namespace tmv

#endif  // TMV_DIAGNOSTICS_CARTESIAN_HORIZONTAL_DERIVATIVES_HPP
