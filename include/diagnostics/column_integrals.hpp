#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace tmv::diagnostics
{
struct ColumnLayerIntegral
{
    double value = 0.0;
    double covered_depth_m = 0.0;
    bool valid = true;
};

template <class FaceHeight, class CellValue>
ColumnLayerIntegral integrate_cell_layer(int begin, int end, double bottom_m, double top_m,
                                        const FaceHeight& face_height, const CellValue& value_at)
{
    ColumnLayerIntegral result;
    result.valid = begin >= 0 && end > begin && std::isfinite(bottom_m) && std::isfinite(top_m) && top_m > bottom_m;
    if (!result.valid) return result;
    for (int k = begin; k < end; ++k)
    {
        const double lower = face_height(k);
        const double upper = face_height(k + 1);
        result.valid = result.valid && std::isfinite(lower) && std::isfinite(upper) && upper > lower;
        const double overlap = std::max(0.0, std::min(upper, top_m) - std::max(lower, bottom_m));
        if (overlap == 0.0) continue;
        const double value = value_at(k);
        result.valid = result.valid && std::isfinite(value);
        result.value += value * overlap;
        result.covered_depth_m += overlap;
    }
    result.valid = result.valid && std::isfinite(result.value);
    return result;
}

template <class CenterHeight, class WindU, class WindV>
double integrate_storm_relative_helicity(int begin, int end, double bottom_m, double top_m,
                                        double storm_u, double storm_v,
                                        const CenterHeight& height_at, const WindU& u_at, const WindV& v_at)
{
    double result = 0.0;
    if (begin < 0 || end <= begin || !std::isfinite(bottom_m) || !std::isfinite(top_m) || top_m <= bottom_m ||
        !std::isfinite(storm_u) || !std::isfinite(storm_v))
        return std::numeric_limits<double>::quiet_NaN();
    for (int k = begin; k + 1 < end; ++k)
    {
        const double z0 = height_at(k), z1 = height_at(k + 1);
        const double u0 = u_at(k) - storm_u, u1 = u_at(k + 1) - storm_u;
        const double v0 = v_at(k) - storm_v, v1 = v_at(k + 1) - storm_v;
        if (!std::isfinite(z0) || !std::isfinite(z1) || z1 <= z0 ||
            !std::isfinite(u0) || !std::isfinite(u1) || !std::isfinite(v0) || !std::isfinite(v1))
            return std::numeric_limits<double>::quiet_NaN();
        const double lower = std::max(z0, bottom_m), upper = std::min(z1, top_m);
        if (upper <= lower) continue;
        const double lo = (lower - z0) / (z1 - z0), hi = (upper - z0) / (z1 - z0);
        const double ulo = u0 + lo * (u1 - u0), uhi = u0 + hi * (u1 - u0);
        const double vlo = v0 + lo * (v1 - v0), vhi = v0 + hi * (v1 - v0);
        result += uhi * vlo - ulo * vhi;
    }
    return result;
}
}
