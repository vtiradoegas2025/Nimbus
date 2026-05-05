#pragma once

#include "init/sounding/parametric_cape.hpp"
#include "init/sounding/sounding_source.hpp"

namespace tmv::init
{

/**
 * @brief Diagnostic targets the user can read off a sounding image.
 *
 * The Pivotal Weather / SHARPpy parameter table that comes with a forecast
 * skew-T contains exactly these scalars: CAPE, CIN, LCL, LFC, EL plus a
 * surface theta and either a mixed-layer mixing ratio or LCL. This source
 * accepts that exact set of inputs and approximates a thermodynamic profile
 * that hits them. It is a *bridge* for users who have the diagnostic table
 * but not the underlying numeric sounding columns — not a substitute for
 * a real RAOB/HRRR file.
 *
 * Approximations and limits:
 *   - LCL inversion uses Lawrence (2005): LCL_m = 125 * (T - T_d) at the
 *     surface, so the surface dewpoint is inferred as T_sfc - LCL_m / 125.
 *     Surface qv is then the saturation mixing ratio at (T_d, p_sfc).
 *   - CAPE is enforced through ParametricCAPEParams.cape_target_jkg, which
 *     scales the unstable-layer lapse rate. The mapping is approximate
 *     (the column build uses a simple multi-layer T(z); CAPE is not
 *     computed back as a constraint).
 *   - LFC and EL map respectively to mixed_layer_top_m and tropopause_z_m
 *     in the underlying parametric build. CIN influences the
 *     mixed-layer dtheta cap strength via a heuristic.
 *
 * If the user only has CAPE and surface conditions (no LCL/LFC/EL), the
 * defaults retain the parametric_cape profile.
 */
struct ParametricTargetsParams
{
    // Convective targets (the headline numbers a user reads from a table).
    double target_cape_jkg = 2500.0;
    double target_cin_jkg = 50.0;
    double target_lcl_m = 1100.0;
    double target_lfc_m = 1500.0;
    double target_el_m = 12000.0;

    // Surface boundary conditions. T_sfc is needed to invert LCL -> qv;
    // p_sfc is the hydrostatic boundary used by the inner ParametricCAPE
    // build (which always integrates from p0 = 1e5 Pa today).
    double surface_theta_k = 300.0;
    double surface_pressure_pa = 100000.0;

    // Optional override: when > 0, this overrides the LCL inversion and
    // becomes the surface mixing ratio directly. Useful when the user
    // has a known qv and wants a specific LCL "automatically" (the LCL
    // will then end up wherever Magnus puts it).
    double moisture_qv_kgkg_override = 0.0;
};

/**
 * @brief Sounding source driven by diagnostic-table values.
 *
 * Translates ParametricTargetsParams into the corresponding
 * ParametricCAPEParams, delegates to ParametricCAPESoundingSource for
 * the actual column build, and returns the result. Behavior is identical
 * to constructing ParametricCAPESoundingSource directly with the
 * translated parameters; this class exists so users can specify their
 * inputs in the form they have rather than the form the parametric
 * builder uses internally.
 */
class ParametricTargetsSoundingSource final : public SoundingSource
{
public:
    explicit ParametricTargetsSoundingSource(ParametricTargetsParams params);

    Sounding build(const std::vector<double>& z_m, double dz) const override;
    std::string describe() const override;

    /**
     * @brief Translates user-facing diagnostic targets into the underlying
     *        ParametricCAPEParams. Exposed for unit testing.
     *
     * The translation is approximate; see the comment block on
     * ParametricTargetsParams above for the assumptions.
     */
    static ParametricCAPEParams translate(const ParametricTargetsParams& targets);

    /**
     * @brief Magnus-formula inverted LCL: returns the surface mixing ratio
     *        such that a surface parcel saturates at z = lcl_m, given the
     *        surface temperature and pressure.
     *
     * Lawrence (2005): LCL_m ~= 125 * (T - T_d).
     * Inverted: T_d = T - lcl_m / 125; qv = 0.622 * e_sat(T_d) / (p - e_sat(T_d)).
     *
     * Returns 1e-5 (the parametric_cape qv floor) when lcl_m is not finite
     * or the implied dewpoint exceeds the surface temperature (negative
     * dewpoint depression, meaning the air would already be saturated).
     */
    static double qv_from_lcl(double lcl_m, double T_surface_k, double p_surface_pa);

private:
    ParametricTargetsParams params_;
};

}  // namespace tmv::init
