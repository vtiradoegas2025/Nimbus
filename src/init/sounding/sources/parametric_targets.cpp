/**
 * @file parametric_targets.cpp
 * @brief ParametricTargetsSoundingSource implementation. Translates a user's
 *        diagnostic-table targets (CAPE, CIN, LCL, LFC, EL, surface theta)
 *        into ParametricCAPEParams and delegates to the parametric source.
 */

#include "init/sounding/parametric_targets.hpp"
#include "init/sounding/parametric_target_math.hpp"
#include "core/infra/physical_constants.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tmv::init
{

namespace
{
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
 *   dtheta_k ~= (2.5 / 300) * CIN_jkg
 *
 * which gives no artificial cap at CIN = 0 and reaches 2.5 K at CIN = 300.
 * Above 300 J/kg the heuristic saturates at 2.5 K because the multi-layer T(z)
 * profile can only support so much cap before the upper unstable layer
 * stops producing CAPE at all.
 */
double dtheta_from_cin(double cin_jkg)
{
    const double linear = (2.5 / 300.0) * std::max(0.0, cin_jkg);
    return std::min(2.5, linear);
}

}  // namespace

double ParametricTargetsSoundingSource::qv_from_lcl(double lcl_m,
                                                    double T_surface_k,
                                                    double p_surface_pa)
{
    return parametric_target_qv_from_lcl(
        lcl_m, T_surface_k, p_surface_pa);
}

ParametricCAPEParams ParametricTargetsSoundingSource::translate(
    const ParametricTargetsParams& t)
{
    if (!std::isfinite(t.target_cape_jkg) || t.target_cape_jkg < 0.0
        || !std::isfinite(t.target_cin_jkg)
        || t.target_cin_jkg < 0.0 || t.target_cin_jkg > 300.0
        || !std::isfinite(t.target_lfc_m) || t.target_lfc_m < 50.0
        || !std::isfinite(t.target_el_m) || t.target_el_m < 8000.0
        || t.target_lfc_m > t.target_el_m - 1000.0
        || !std::isfinite(t.surface_theta_k) || t.surface_theta_k < 250.0
        || !std::isfinite(t.surface_pressure_pa)
        || t.surface_pressure_pa < 1000.0 || t.surface_pressure_pa > 200000.0)
    {
        throw std::invalid_argument(
            "ParametricTargetsSoundingSource::translate: invalid CAPE/CIN/LFC/EL or surface thermodynamic target");
    }
    if (t.lcl_is_targeted()
        && (!std::isfinite(t.target_lcl_m) || t.target_lcl_m <= 0.0))
    {
        throw std::invalid_argument(
            "ParametricTargetsSoundingSource::translate: targeted LCL must be positive and finite");
    }
    if (!t.lcl_is_targeted()
        && (!std::isfinite(t.moisture_qv_kgkg_override)
            || t.moisture_qv_kgkg_override < 0.004
            || t.moisture_qv_kgkg_override > 0.024))
    {
        throw std::invalid_argument(
            "ParametricTargetsSoundingSource::translate: moisture override must be in [0.004, 0.024] kg/kg");
    }

    ParametricCAPEParams p;
    p.cape_target_jkg = t.target_cape_jkg;
    p.surface_theta_k = t.surface_theta_k;
    p.surface_pressure_pa = t.surface_pressure_pa;
    p.tropopause_z_m = t.target_el_m;

    // LFC roughly equals the top of the mixed/cap layer, where the parcel
    // first becomes warmer than environment. The unstable layer in
    // ParametricCAPE starts at mixed_layer_top_m, so LFC ~= mixed_layer_top.
    p.mixed_layer_top_m = t.target_lfc_m;

    // Cap strength from CIN heuristic. Stronger cap => more CIN.
    p.mixed_layer_dtheta_k = dtheta_from_cin(t.target_cin_jkg);

    // Surface qv from LCL inversion unless explicitly overridden.
    if (!t.lcl_is_targeted())
    {
        p.surface_qv_kgkg = t.moisture_qv_kgkg_override;
    }
    else
    {
        const double surface_temperature_k = t.surface_theta_k * std::pow(
            t.surface_pressure_pa / physical_constants::reference_pressure_pa,
            physical_constants::gas_constant_dry_air_jkgk
                / physical_constants::specific_heat_cp_jkgk);
        p.surface_qv_kgkg = qv_from_lcl(t.target_lcl_m,
                                        surface_temperature_k,
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
