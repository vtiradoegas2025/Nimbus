/**
 * @file parametric_targets.cpp
 * @brief ParametricTargetsSoundingSource implementation. Translates a user's
 *        diagnostic-table targets (CAPE, CIN, LCL, LFC, EL, surface theta)
 *        into ParametricCAPEParams and delegates to the parametric source.
 */

#include "init/sounding/parametric_targets.hpp"

#include "core/physical_constants.hpp"

#include <algorithm>
#include <cmath>

namespace tmv::init
{

namespace
{

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
 * @brief Heuristic mapping from CIN target to mixed-layer dtheta cap.
 *
 * The underlying ParametricCAPE build offsets the mixed-layer T by
 * mixed_layer_dtheta_k above the surface theta to keep the boundary layer
 * warm. A larger dtheta produces a stronger cap (the parcel has to climb
 * through more negative buoyancy), which roughly correlates with higher
 * CIN. The relation is non-linear and depends on layer depth + lapse rate;
 * we use the linear approximation
 *
 *   dtheta_k ~= 1.0 + 0.005 * CIN_jkg
 *
 * which gives 1.0 K (today's default) at CIN = 0 and ~1.5 K at CIN = 100,
 * matching the observed dtheta range across our existing scenarios. Above
 * 300 J/kg the heuristic saturates at 2.5 K because the multi-layer T(z)
 * profile can only support so much cap before the upper unstable layer
 * stops producing CAPE at all.
 */
double dtheta_from_cin(double cin_jkg)
{
    const double linear = 1.0 + 0.005 * std::max(0.0, cin_jkg);
    return std::min(2.5, linear);
}

}  // namespace

double ParametricTargetsSoundingSource::qv_from_lcl(double lcl_m,
                                                    double T_surface_k,
                                                    double p_surface_pa)
{
    constexpr double qv_floor = 1.0e-5;
    if (!std::isfinite(lcl_m) || lcl_m <= 0.0)
    {
        return qv_floor;
    }

    // Lawrence (2005): LCL_m ~= 125 * (T - T_d). Solve for T_d.
    constexpr double lcl_per_dewpoint_depression_m_per_k = 125.0;
    const double dewpoint_depression_k = lcl_m / lcl_per_dewpoint_depression_m_per_k;
    const double T_d = T_surface_k - dewpoint_depression_k;

    if (!std::isfinite(T_d) || T_d <= 0.0 || T_d > T_surface_k)
    {
        return qv_floor;
    }

    const double e_d = saturation_vapor_pressure_pa(T_d);
    const double qv = 0.622 * e_d / std::max(p_surface_pa - e_d, 1.0);
    return std::max(qv, qv_floor);
}

ParametricCAPEParams ParametricTargetsSoundingSource::translate(
    const ParametricTargetsParams& t)
{
    ParametricCAPEParams p;
    p.cape_target_jkg = t.target_cape_jkg;
    p.surface_theta_k = t.surface_theta_k;
    p.tropopause_z_m = t.target_el_m;

    // LFC roughly equals the top of the mixed/cap layer, where the parcel
    // first becomes warmer than environment. The unstable layer in
    // ParametricCAPE starts at mixed_layer_top_m, so LFC ~= mixed_layer_top.
    p.mixed_layer_top_m = std::max(50.0, t.target_lfc_m);

    // Cap strength from CIN heuristic. Stronger cap => more CIN.
    p.mixed_layer_dtheta_k = dtheta_from_cin(t.target_cin_jkg);

    // Surface qv from LCL inversion unless explicitly overridden.
    if (t.moisture_qv_kgkg_override > 0.0)
    {
        p.surface_qv_kgkg = t.moisture_qv_kgkg_override;
    }
    else
    {
        // Surface T = surface_theta + mixed_layer_dtheta (matches the
        // parametric build's level-0 sample). We use surface_theta directly
        // for the LCL inversion: small offsets shift the inferred dewpoint
        // by O(0.5 K), within the approximation's noise floor anyway.
        p.surface_qv_kgkg = qv_from_lcl(t.target_lcl_m,
                                        t.surface_theta_k,
                                        t.surface_pressure_pa);
    }

    return p;
}

ParametricTargetsSoundingSource::ParametricTargetsSoundingSource(ParametricTargetsParams params)
    : params_(params)
{
}

Sounding ParametricTargetsSoundingSource::build(const std::vector<double>& z_m,
                                                double dz) const
{
    const ParametricCAPEParams translated = translate(params_);
    return ParametricCAPESoundingSource(translated).build(z_m, dz);
}

std::string ParametricTargetsSoundingSource::describe() const
{
    return "parametric_targets";
}

}  // namespace tmv::init
