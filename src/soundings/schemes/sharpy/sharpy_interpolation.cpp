#include "sharpy_interpolation.hpp"

#include "core/infra/physical_constants.hpp"
#include "soundings/base/sounding_algorithms.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tmv::soundings::sharpy
{
/**
 * @brief Performs linear interpolation between two sample points.
 */
double linear_interpolate_value(double x1, double y1, double x2, double y2, double x_target)
{
    if (x1 == x2)
    {
        return y1;
    }
    return y1 + (y2 - y1) * (x_target - x1) / (x2 - x1);
}

/**
 * @brief Performs log-linear interpolation for positive-valued series.
 */
double log_linear_interpolate_positive(double x1, double y1, double x2, double y2, double x_target)
{
    if (x1 == x2)
    {
        return y1;
    }
    if (!std::isfinite(y1) || !std::isfinite(y2) || y1 <= 0.0 || y2 <= 0.0)
    {
        return linear_interpolate_value(x1, y1, x2, y2, x_target);
    }

    const double ln_y1 = std::log(y1);
    const double ln_y2 = std::log(y2);
    const double ln_y = linear_interpolate_value(x1, ln_y1, x2, ln_y2, x_target);
    return std::exp(ln_y);
}

/**
 * @brief Interpolates pressure at one target height using log-linear profile space.
 */
double interpolate_pressure_log_linear(const SoundingData& source,
                                       double target_height_m,
                                       const SoundingConfig& config)
{
    if (source.height_m.empty() || source.pressure_hpa.empty())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (source.height_m.size() == 1 || source.pressure_hpa.size() == 1)
    {
        return source.pressure_hpa.front();
    }

    auto it = std::lower_bound(source.height_m.begin(), source.height_m.end(), target_height_m);

    if (it == source.height_m.begin())
    {
        if (config.extrapolate_below_ground)
        {
            return log_linear_interpolate_positive(
                source.height_m[0], source.pressure_hpa[0],
                source.height_m[1], source.pressure_hpa[1],
                target_height_m);
        }
        return source.pressure_hpa.front();
    }

    if (it == source.height_m.end())
    {
        if (config.extrapolate_above_top)
        {
            const std::size_t idx2 = source.height_m.size() - 1;
            const std::size_t idx1 = idx2 - 1;
            return log_linear_interpolate_positive(
                source.height_m[idx1], source.pressure_hpa[idx1],
                source.height_m[idx2], source.pressure_hpa[idx2],
                target_height_m);
        }
        return source.pressure_hpa.back();
    }

    const std::size_t idx2 = static_cast<std::size_t>(it - source.height_m.begin());
    const std::size_t idx1 = idx2 - 1;
    return log_linear_interpolate_positive(
        source.height_m[idx1], source.pressure_hpa[idx1],
        source.height_m[idx2], source.pressure_hpa[idx2],
        target_height_m);
}

/**
 * @brief Builds pressure profile on target levels using log-linear interpolation.
 */
void apply_log_linear_pressure_profile(const SoundingData& source,
                                       const std::vector<double>& target_heights_m,
                                       const SoundingConfig& config,
                                       SoundingData& interpolated)
{
    interpolated.pressure_hpa.resize(target_heights_m.size());
    for (std::size_t i = 0; i < target_heights_m.size(); ++i)
    {
        interpolated.pressure_hpa[i] = interpolate_pressure_log_linear(source, target_heights_m[i], config);
    }
}

/**
 * @brief Checks whether a monotone PCHIP spline can be safely applied.
 */
bool can_use_monotone_spline(const std::vector<double>& x, const std::vector<double>& y)
{
    if (x.size() < 2 || y.size() != x.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]))
        {
            return false;
        }
        if (i > 0 && !(x[i] > x[i - 1]))
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief Computes endpoint tangent for monotone PCHIP interpolation.
 */
double pchip_endpoint_slope(double h0, double h1, double delta0, double delta1)
{
    const double denom = h0 + h1;
    if (!std::isfinite(denom) || denom == 0.0)
    {
        return delta0;
    }

    double m = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / denom;
    if (m * delta0 <= 0.0)
    {
        m = 0.0;
    }
    else if (std::abs(m) > 3.0 * std::abs(delta0))
    {
        m = 3.0 * delta0;
    }
    return m;
}

/**
 * @brief Computes monotone PCHIP tangents for a sampled series.
 */
std::vector<double> compute_pchip_tangents(const std::vector<double>& x, const std::vector<double>& y)
{
    const std::size_t n = x.size();
    std::vector<double> tangents(n, 0.0);
    if (n == 0)
    {
        return tangents;
    }
    if (n == 1)
    {
        tangents[0] = 0.0;
        return tangents;
    }

    std::vector<double> h(n - 1, 0.0);
    std::vector<double> delta(n - 1, 0.0);
    for (std::size_t i = 0; i + 1 < n; ++i)
    {
        h[i] = x[i + 1] - x[i];
        delta[i] = (y[i + 1] - y[i]) / h[i];
    }

    if (n == 2)
    {
        tangents[0] = delta[0];
        tangents[1] = delta[0];
        return tangents;
    }

    tangents[0] = pchip_endpoint_slope(h[0], h[1], delta[0], delta[1]);
    tangents[n - 1] = pchip_endpoint_slope(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);

    for (std::size_t i = 1; i + 1 < n; ++i)
    {
        if (delta[i - 1] == 0.0 || delta[i] == 0.0 || delta[i - 1] * delta[i] < 0.0)
        {
            tangents[i] = 0.0;
            continue;
        }
        const double w1 = 2.0 * h[i] + h[i - 1];
        const double w2 = h[i] + 2.0 * h[i - 1];
        tangents[i] = (w1 + w2) / ((w1 / delta[i - 1]) + (w2 / delta[i]));
    }

    return tangents;
}

/**
 * @brief Evaluates one cubic Hermite segment at target abscissa.
 */
double evaluate_cubic_hermite(
    double x0,
    double x1,
    double y0,
    double y1,
    double m0,
    double m1,
    double x_target)
{
    const double h = x1 - x0;
    if (h == 0.0)
    {
        return y0;
    }
    const double t = (x_target - x0) / h;
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double h00 = (2.0 * t3) - (3.0 * t2) + 1.0;
    const double h10 = t3 - (2.0 * t2) + t;
    const double h01 = (-2.0 * t3) + (3.0 * t2);
    const double h11 = t3 - t2;
    return (h00 * y0) + (h10 * h * m0) + (h01 * y1) + (h11 * h * m1);
}

/**
 * @brief Interpolates one sounding series using monotone spline when possible.
 */
std::vector<double> interpolate_series_monotone_spline(
    const std::vector<double>& source_heights_m,
    const std::vector<double>& source_values,
    const std::vector<double>& target_heights_m,
    const SoundingConfig& config)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> result(target_heights_m.size(), nan);
    if (source_values.empty() || source_heights_m.empty() || source_values.size() != source_heights_m.size())
    {
        return result;
    }
    if (source_values.size() == 1)
    {
        std::fill(result.begin(), result.end(), source_values.front());
        return result;
    }

    bool use_spline = can_use_monotone_spline(source_heights_m, source_values);
    std::vector<double> tangents;
    if (use_spline)
    {
        tangents = compute_pchip_tangents(source_heights_m, source_values);
        use_spline = tangents.size() == source_values.size();
    }

    for (std::size_t i = 0; i < target_heights_m.size(); ++i)
    {
        const double target = target_heights_m[i];
        auto it = std::lower_bound(source_heights_m.begin(), source_heights_m.end(), target);

        if (it == source_heights_m.begin())
        {
            result[i] = config.extrapolate_below_ground
                ? linear_interpolate_value(
                    source_heights_m[0], source_values[0],
                    source_heights_m[1], source_values[1], target)
                : source_values.front();
            continue;
        }

        if (it == source_heights_m.end())
        {
            const std::size_t idx2 = source_heights_m.size() - 1;
            const std::size_t idx1 = idx2 - 1;
            result[i] = config.extrapolate_above_top
                ? linear_interpolate_value(
                    source_heights_m[idx1], source_values[idx1],
                    source_heights_m[idx2], source_values[idx2], target)
                : source_values.back();
            continue;
        }

        const std::size_t idx2 = static_cast<std::size_t>(it - source_heights_m.begin());
        const std::size_t idx1 = idx2 - 1;
        if (source_heights_m[idx2] == target)
        {
            result[i] = source_values[idx2];
            continue;
        }

        if (use_spline)
        {
            result[i] = evaluate_cubic_hermite(
                source_heights_m[idx1], source_heights_m[idx2],
                source_values[idx1], source_values[idx2],
                tangents[idx1], tangents[idx2],
                target);
        }
        else
        {
            result[i] = linear_interpolate_value(
                source_heights_m[idx1], source_values[idx1],
                source_heights_m[idx2], source_values[idx2],
                target);
        }
    }

    return result;
}

/**
 * @brief Interpolates an entire sounding profile with monotone spline support.
 */
SoundingData interpolate_sounding_monotone_spline(
    const SoundingData& source,
    const std::vector<double>& target_heights_m,
    const SoundingConfig& config)
{
    SoundingData result;
    result.height_m = target_heights_m;
    result.potential_temperature_is_primary =
        source.potential_temperature_is_primary
        && source.potential_temperature_k.size() == source.height_m.size();
    result.pressure_hpa = interpolate_series_monotone_spline(
        source.height_m, source.pressure_hpa, target_heights_m, config);
    result.temperature_k = interpolate_series_monotone_spline(
        source.height_m, source.temperature_k, target_heights_m, config);
    if (result.potential_temperature_is_primary)
    {
        result.potential_temperature_k = interpolate_series_monotone_spline(
            source.height_m, source.potential_temperature_k,
            target_heights_m, config);
    }

    if (!source.dewpoint_k.empty() && source.dewpoint_k.size() == source.height_m.size())
    {
        result.dewpoint_k = interpolate_series_monotone_spline(
            source.height_m, source.dewpoint_k, target_heights_m, config);
    }
    if (!source.mixing_ratio_kgkg.empty()
        && source.mixing_ratio_kgkg.size() == source.height_m.size())
    {
        result.mixing_ratio_kgkg = interpolate_series_monotone_spline(
            source.height_m, source.mixing_ratio_kgkg,
            target_heights_m, config);
    }

    std::vector<double> source_u;
    std::vector<double> source_v;
    if (source.wind_u_ms.size() == source.height_m.size()
        && source.wind_v_ms.size() == source.height_m.size())
    {
        source_u = source.wind_u_ms;
        source_v = source.wind_v_ms;
    }
    else if (source.wind_speed_ms.size() == source.height_m.size()
             && source.wind_direction_deg.size() == source.height_m.size())
    {
        source_u.resize(source.height_m.size());
        source_v.resize(source.height_m.size());
        for (std::size_t i = 0; i < source.height_m.size(); ++i)
        {
            const double direction_rad =
                source.wind_direction_deg[i]
                * physical_constants::pi / 180.0;
            source_u[i] = -source.wind_speed_ms[i] * std::sin(direction_rad);
            source_v[i] = -source.wind_speed_ms[i] * std::cos(direction_rad);
        }
    }
    if (!source_u.empty())
    {
        result.wind_u_ms = interpolate_series_monotone_spline(
            source.height_m, source_u, target_heights_m, config);
        result.wind_v_ms = interpolate_series_monotone_spline(
            source.height_m, source_v, target_heights_m, config);
        result.wind_speed_ms.resize(target_heights_m.size());
        result.wind_direction_deg.resize(target_heights_m.size());
        for (std::size_t i = 0; i < target_heights_m.size(); ++i)
        {
            result.wind_speed_ms[i] =
                std::hypot(result.wind_u_ms[i], result.wind_v_ms[i]);
            if (result.wind_speed_ms[i] == 0.0)
            {
                result.wind_direction_deg[i] = 0.0;
                continue;
            }
            double direction =
                std::atan2(-result.wind_u_ms[i], -result.wind_v_ms[i])
                * 180.0 / physical_constants::pi;
            result.wind_direction_deg[i] =
                direction < 0.0 ? direction + 360.0 : direction;
        }
    }

    calculate_derived_quantities(result);
    return result;
}


}  // namespace tmv::soundings::sharpy
