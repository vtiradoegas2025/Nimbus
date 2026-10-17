/**
 * @file parametric_cape.cpp
 * @brief Procedural sounding source that hits a target CAPE.
 *
 * Builds a physically-honest idealized column and tunes it (closed-loop) so the
 * realized CAPE matches the requested cape_target:
 *
 *   - Well-mixed boundary layer (0 .. z_ml): constant potential temperature
 *     (dry-adiabatic T), so a surface parcel rises neutrally to its LCL instead
 *     of being trapped by a spurious inversion.
 *   - Free troposphere (z_ml .. z_tr): temperature falls at a lapse rate Gamma
 *     that lies between the moist and dry adiabats -- the conditionally-unstable
 *     layer that sets CAPE. Gamma is the control variable.
 *   - Stratosphere (>= z_tr): stable cap (weak warming).
 *
 * Realized CAPE increases monotonically with Gamma, so we bisect Gamma between
 * the moist- and dry-adiabatic bounds until the parcel-ascent CAPE (computed by
 * the same diagnostic the runtime reports) matches cape_target. This makes
 * cape_jkg an actual target rather than an un-honored label, and replaces the
 * previous open-loop lapse formula, which produced a stable column (CAPE = 0)
 * for any normal target.
 */

#include "init/sounding/parametric_cape.hpp"
#include "init/sounding/diagnostics.hpp"
#include "core/infra/physical_constants.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace tmv::init
{

namespace
{

/**
 * @brief Magnus-formula saturation vapor pressure (Pa) over water/ice.
 * Switches to ice formulation below 273.15 K.
 */
double saturation_vapor_pressure_pa(double T_k)
{
    const double T_c = T_k - physical_constants::freezing_temperature_k;
    if (T_k >= physical_constants::freezing_temperature_k)
    {
        return 611.21 * std::exp((18.678 - T_c / 234.5) * T_c / (257.14 + T_c));
    }
    return 611.15 * std::exp((23.036 - T_c / 333.7) * T_c / (279.82 + T_c));
}

/**
 * @brief Saturation mixing ratio (kg/kg) at the supplied (T, p).
 */
double saturation_mixing_ratio(double T_k, double p_pa)
{
    const double e_sat = saturation_vapor_pressure_pa(T_k);
    return 0.622 * e_sat / std::max(p_pa - e_sat, 1.0);
}

}  // namespace

ParametricCAPESoundingSource::ParametricCAPESoundingSource(ParametricCAPEParams params)
    : params_(params)
{
}

Sounding ParametricCAPESoundingSource::build(const std::vector<double>& z_m, double dz) const
{
    if (z_m.empty())
    {
        throw std::invalid_argument("ParametricCAPESoundingSource::build: z_m is empty");
    }
    if (!(dz > 0.0))
    {
        throw std::invalid_argument("ParametricCAPESoundingSource::build: dz must be positive");
    }
    if (!std::isfinite(z_m.front()) || z_m.front() < 0.0)
    {
        throw std::invalid_argument(
            "ParametricCAPESoundingSource::build: model heights must be non-negative");
    }
    for (std::size_t k = 1; k < z_m.size(); ++k)
    {
        if (!std::isfinite(z_m[k]) || !(z_m[k] > z_m[k - 1]))
        {
            throw std::invalid_argument(
                "ParametricCAPESoundingSource::build: model heights must increase strictly");
        }
    }

    if (!std::isfinite(params_.surface_theta_k) || params_.surface_theta_k < 250.0)
    {
        throw std::invalid_argument(
            "ParametricCAPESoundingSource::build: surface theta must be finite and at least 250 K");
    }
    const double surface_theta = params_.surface_theta_k;
    const double surface_pressure = params_.surface_pressure_pa;
    if (!std::isfinite(surface_pressure)
        || surface_pressure < 1000.0 || surface_pressure > 200000.0)
    {
        throw std::invalid_argument(
            "ParametricCAPESoundingSource::build: surface pressure must be finite and in [1000, 200000] Pa");
    }
    if (!std::isfinite(params_.base_moisture_min)
        || !std::isfinite(params_.base_moisture_max)
        || params_.base_moisture_min < 0.0
        || params_.base_moisture_max < params_.base_moisture_min
        || !std::isfinite(params_.surface_qv_kgkg)
        || params_.surface_qv_kgkg < params_.base_moisture_min
        || params_.surface_qv_kgkg > params_.base_moisture_max)
    {
        throw std::invalid_argument(
            "ParametricCAPESoundingSource::build: surface qv must be finite and within the configured moisture bounds");
    }
    if (!std::isfinite(params_.tropopause_z_m) || params_.tropopause_z_m < 8000.0)
    {
        throw std::invalid_argument(
            "ParametricCAPESoundingSource::build: tropopause height must be finite and at least 8000 m");
    }
    const double tropopause_z = params_.tropopause_z_m;
    if (!std::isfinite(params_.mixed_layer_top_m)
        || params_.mixed_layer_top_m < 0.0
        || params_.mixed_layer_top_m > tropopause_z - 1000.0)
    {
        throw std::invalid_argument(
            "ParametricCAPESoundingSource::build: mixed-layer top must be finite and at least 1000 m below the tropopause");
    }
    if (!std::isfinite(params_.cape_target_jkg) || params_.cape_target_jkg < 0.0
        || !std::isfinite(params_.mixed_layer_dtheta_k) || params_.mixed_layer_dtheta_k < 0.0
        || !std::isfinite(params_.rh_cap) || params_.rh_cap <= 0.0 || params_.rh_cap > 1.0
        || !std::isfinite(params_.moisture_decay_z_m) || params_.moisture_decay_z_m < 0.0
        || !std::isfinite(params_.moisture_scale_height_floor_m)
        || params_.moisture_scale_height_floor_m <= 0.0
        || !std::isfinite(params_.moisture_scale_height_factor)
        || params_.moisture_scale_height_factor <= 0.0
        || !std::isfinite(params_.strat_warming_kpm))
    {
        throw std::invalid_argument(
            "ParametricCAPESoundingSource::build: invalid CAPE, cap, moisture, RH, or stratospheric parameter");
    }
    const double surface_qv = params_.surface_qv_kgkg;
    const double z_ml = params_.mixed_layer_top_m;
    const double kappa         = R_d / cp;
    const double dry_lapse     = g / cp;  // K/m; constant-theta (well-mixed) lapse
    const double surface_T     = surface_theta
        * std::pow(surface_pressure / p0, kappa);
    const double cap_dtheta    = params_.mixed_layer_dtheta_k;

    // Moisture column: surface mixing ratio (clamped), exponential decay above
    // moisture_decay_z_m, RH-capped at every level. Fixed across the CAPE
    // bisection (CAPE is varied through the temperature lapse, not moisture).
    const double base_moisture = surface_qv;
    const double moisture_scale_height = std::max(
        params_.moisture_scale_height_floor_m,
        params_.moisture_scale_height_factor * tropopause_z);

    // Build the full hydrostatic column for a candidate free-tropospheric
    // lapse rate Gamma (K/m).
    auto build_profile = [&](double gamma_free) -> Sounding
    {
        auto T_at = [&](double z) -> double
        {
            if (z < z_ml)
            {
                // A smooth environmental inversion gives the parametric-target
                // branch a real CIN control without changing the specified
                // surface theta. With cap_dtheta=0 this is the original
                // constant-theta mixed layer.
                const double x = (z_ml > 0.0)
                    ? std::clamp(z / z_ml, 0.0, 1.0) : 1.0;
                const double smooth_cap = cap_dtheta * x * x * (3.0 - 2.0 * x);
                return surface_T - dry_lapse * z + smooth_cap;
            }
            const double T_ml = surface_T - dry_lapse * z_ml + cap_dtheta;
            if (z < tropopause_z)
            {
                return T_ml - gamma_free * (z - z_ml);  // conditionally unstable
            }
            const double T_tr = T_ml - gamma_free * (tropopause_z - z_ml);
            return T_tr + params_.strat_warming_kpm * (z - tropopause_z);  // stable cap
        };

        const std::size_t nz = z_m.size();
        Sounding s;
        s.z_m = z_m;
        s.T_k.resize(nz);
        s.theta_k.resize(nz);
        s.qv_kgkg.resize(nz);
        s.p_pa.resize(nz);
        s.rho_kgm3.resize(nz);

        // Hydrostatic integration from the physical surface face (z=0) to the
        // first scalar center, then center-to-center. The trapezoidal layer
        // temperature makes each exponential exact for an isothermal layer and
        // second-order for a smoothly varying profile.
        const double surface_air_T = T_at(0.0);
        s.T_k[0]      = T_at(z_m[0]);
        const double first_layer_T = 0.5 * (surface_air_T + s.T_k[0]);
        s.p_pa[0]     = surface_pressure
            * std::exp(-g * z_m[0] / (R_d * first_layer_T));
        s.rho_kgm3[0] = s.p_pa[0] / (R_d * s.T_k[0]);
        for (std::size_t k = 1; k < nz; ++k)
        {
            s.T_k[k]      = T_at(z_m[k]);
            const double T_avg = 0.5 * (s.T_k[k] + s.T_k[k - 1]);
            const double layer_dz = z_m[k] - z_m[k - 1];
            s.p_pa[k]     = s.p_pa[k - 1] * std::exp(-g * layer_dz / (R_d * T_avg));
            s.rho_kgm3[k] = s.p_pa[k] / (R_d * s.T_k[k]);
        }

        for (std::size_t k = 0; k < nz; ++k)
        {
            if (!std::isfinite(s.T_k[k]) || s.T_k[k] <= 0.0
                || !std::isfinite(s.p_pa[k]) || s.p_pa[k] <= 0.0
                || !std::isfinite(s.rho_kgm3[k]) || s.rho_kgm3[k] <= 0.0)
            {
                throw std::runtime_error(
                    "ParametricCAPESoundingSource::build produced an invalid thermodynamic state");
            }
        }

        for (std::size_t k = 0; k < nz; ++k)
        {
            s.theta_k[k] = s.T_k[k] * std::pow(p0 / s.p_pa[k], kappa);
        }

        for (std::size_t k = 0; k < nz; ++k)
        {
            const double z = z_m[k];
            double qv_val;
            if (z < params_.moisture_decay_z_m)
            {
                qv_val = base_moisture;
            }
            else
            {
                qv_val = base_moisture * std::exp(
                    -(z - params_.moisture_decay_z_m) / moisture_scale_height);
            }
            const double qvsat = saturation_mixing_ratio(s.T_k[k], s.p_pa[k]);
            s.qv_kgkg[k] = std::min(qv_val, qvsat * params_.rh_cap);
        }

        return s;
    };

    // Closed-loop CAPE targeting: bisect the free-tropospheric lapse Gamma
    // (monotone: steeper -> more conditional instability -> more CAPE) until the
    // parcel-ascent CAPE matches the request. Bounds bracket sub-moist-adiabatic
    // (~0 CAPE) to ~dry-adiabatic (maximum realizable CAPE).
    const double target   = params_.cape_target_jkg;
    const double cape_tol = 50.0;     // J/kg
    const int    max_iter = 60;
    double gamma_lo = 0.0055;          // K/m, ~5.5 K/km
    double gamma_hi = 0.0098;          // K/m, ~dry-adiabatic
    double gamma    = gamma_lo;

    if (target > 0.0)
    {
        for (int it = 0; it < max_iter; ++it)
        {
            gamma = 0.5 * (gamma_lo + gamma_hi);
            const double cape =
                compute_sounding_diagnostics(build_profile(gamma)).cape_jkg;
            if (std::abs(cape - target) <= cape_tol)
            {
                break;
            }
            if (cape < target)
            {
                gamma_lo = gamma;  // need a steeper lapse for more CAPE
            }
            else
            {
                gamma_hi = gamma;
            }
        }
    }

    return build_profile(gamma);
}

std::string ParametricCAPESoundingSource::describe() const
{
    return "parametric_cape";
}

}  // namespace tmv::init
