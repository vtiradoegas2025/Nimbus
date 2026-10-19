/**
 * @file weno5_face.hpp
 * @brief Pure WENO5-JS cell-face reconstruction (Jiang & Shu 1996).
 *
 * Single-source definition of the upwind WENO5 face reconstruction used by
 * the conservative Cartesian C-grid transport
 * (advect_scalar_conservative_3d_cartesian in advection_cartesian.cpp) and
 * mirrored literally - same candidate polynomials, smoothness indicators,
 * ideal weights, and epsilon - in the GPU shader
 * advect_conservative_scalar_flux_cartesian_cgrid.comp.
 *
 * Kept dependency-free (only <cmath>) so including it adds nothing to a
 * translation unit's link closure: advection_cartesian.cpp's file-header
 * contract requires staying clear of the advection factory, and a pure
 * inline header satisfies that while making the reconstruction unit-testable
 * (tests/numerics/test_weno5_reconstruction.cpp).
 */
#pragma once

#include <algorithm>
#include <cmath>

/**
 * @brief Upwind WENO5-JS reconstruction of the cell-face value.
 *
 * Reconstructs the scalar value at the face on the `d` side of cell `c` for
 * flow moving c -> d, from the upwind-biased 5-cell stencil
 * (a, b, c, d, e) = (c-2, c-1, c, c+1, c+2). This is the standard
 * Jiang & Shu (1996) WENO5 with ideal weights (1/10, 6/10, 3/10), the JS
 * smoothness indicators, epsilon = numerics_constants::weno_epsilon (1e-6),
 * and nonlinear-weight exponent p = 2.
 *
 * The candidate polynomials are interpolatory (their coefficients each sum to
 * 1), so a spatially constant field reconstructs to that constant exactly --
 * free-stream preservation is retained bit-for-bit. Unlike a TVD/MC limiter,
 * WENO5 retains a high-order, non-zero reconstruction at smooth extrema, so a
 * resolved thermal is advected without spurious 1st-order numerical diffusion;
 * it is essentially non-oscillatory across a discontinuity (the smoothness
 * indicators starve the substencil that straddles the jump). Falls back to
 * returning the upwind value `c` if the nonlinear weights are not finite
 * (degenerate inputs).
 */
inline double weno5_face_value(double a, double b, double c, double d, double e)
{
    constexpr double eps_rel = 1.0e-6;  // relative coefficient (eps below scales with field^2)
    constexpr double d0  = 1.0 / 10.0;
    constexpr double d1  = 6.0 / 10.0;
    constexpr double d2  = 3.0 / 10.0;

    const double p0 = (1.0 / 3.0) * a - (7.0 / 6.0) * b + (11.0 / 6.0) * c;
    const double p1 = -(1.0 / 6.0) * b + (5.0 / 6.0) * c + (1.0 / 3.0) * d;
    const double p2 = (1.0 / 3.0) * c + (5.0 / 6.0) * d - (1.0 / 6.0) * e;

    const double s0a = a - 2.0 * b + c;
    const double s0b = a - 4.0 * b + 3.0 * c;
    const double s1a = b - 2.0 * c + d;
    const double s1b = b - d;
    const double s2a = c - 2.0 * d + e;
    const double s2b = 3.0 * c - 4.0 * d + e;
    const double b0 = (13.0 / 12.0) * s0a * s0a + 0.25 * s0b * s0b;
    const double b1 = (13.0 / 12.0) * s1a * s1a + 0.25 * s1b * s1b;
    const double b2 = (13.0 / 12.0) * s2a * s2a + 0.25 * s2b * s2b;

    // Field-magnitude-relative epsilon (science audit): beta_k ~ (field)^2, so
    // an absolute eps = 1e-6 swamps beta for moisture-scale fields
    // (q ~ 1e-3..1e-4) and collapses the weights to the linear-upwind ideal
    // weights exactly at cloud edges. Scaling eps by the local stencil
    // magnitude^2 holds it at the same relative position vs beta for any field
    // scale (reproducing ~1e-6 for O(1) fields), so WENO5 stays
    // essentially-non-oscillatory at moisture edges. The tiny floor only guards
    // an all-zero stencil (which still reconstructs to 0). The GPU shader and
    // the unit test mirror this exactly.
    const double q2max = std::max(std::max(std::max(a * a, b * b),
                                           std::max(c * c, d * d)), e * e);
    const double eps = eps_rel * q2max + 1.0e-40;
    const double a0 = d0 / ((eps + b0) * (eps + b0));
    const double a1 = d1 / ((eps + b1) * (eps + b1));
    const double a2 = d2 / ((eps + b2) * (eps + b2));
    const double sum = a0 + a1 + a2;
    if (!(sum > 0.0) || !std::isfinite(sum))
    {
        return c;
    }
    const double face = (a0 * p0 + a1 * p1 + a2 * p2) / sum;
    return std::isfinite(face) ? face : c;
}

inline float weno5_face_offset_fp32(float a, float b, float c, float d, float e)
{
    const float magnitude = std::max({std::abs(a), std::abs(b), std::abs(c), std::abs(d), std::abs(e)});
    const float scale = std::max(magnitude, 1e-20f);
    if (!std::isfinite(scale))
        return 0;
    const auto difference = [=](float value)
    {
        const float delta = value - c;
        return std::isfinite(delta) ? delta / scale : value / scale - c / scale;
    };
    const float ac = difference(a), bc = difference(b), dc = difference(d), ec = difference(e);
    const float p0 = (1.0f / 3.0f) * ac - (7.0f / 6.0f) * bc;
    const float p1 = -(1.0f / 6.0f) * bc + (1.0f / 3.0f) * dc;
    const float p2 = (5.0f / 6.0f) * dc - (1.0f / 6.0f) * ec;
    const float s0a = ac - 2.0f * bc, s0b = ac - 4.0f * bc;
    const float s1a = bc + dc, s1b = bc - dc;
    const float s2a = ec - 2.0f * dc, s2b = ec - 4.0f * dc;
    const float b0 = (13.0f / 12.0f) * s0a * s0a + .25f * s0b * s0b;
    const float b1 = (13.0f / 12.0f) * s1a * s1a + .25f * s1b * s1b;
    const float b2 = (13.0f / 12.0f) * s2a * s2a + .25f * s2b * s2b;
    // A common scaling cancels from the weights. Keep their arithmetic bounded
    // without losing the CPU equation's absolute 1e-40 regularization floor.
    const float relative = magnitude / scale;
    const float floor = 1e-20f / scale;
    const float epsilon = 1e-6f * relative * relative + floor * floor;
    const float minimum = std::min({epsilon + b0, epsilon + b1, epsilon + b2});
    const float r0 = minimum / (epsilon + b0), r1 = minimum / (epsilon + b1), r2 = minimum / (epsilon + b2);
    const float a0 = .1f * r0 * r0, a1 = .6f * r1 * r1, a2 = .3f * r2 * r2;
    const float sum = a0 + a1 + a2;
    const float offset = scale * ((a0 * p0 + a1 * p1 + a2 * p2) / sum);
    return std::isfinite(offset) ? offset : 0;
}
