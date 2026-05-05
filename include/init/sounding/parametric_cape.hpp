#pragma once

#include "init/sounding/sounding_source.hpp"

namespace tmv::init 
{

/**
 * @brief Tunable parameters for ParametricCAPESoundingSource.
 *
 * Defaults match the values previously hardcoded inline in
 * `src/core/orchestration/dynamics/equations.cpp::initialize()`. The
 * extraction preserves behavior bit-for-bit when the source is constructed
 * with these defaults and the same surface theta / qv / CAPE / tropopause
 * as the previous globals.
 *
 * The previous-inline knobs that are now first-class config:
 *   - mixed_layer_top_m           (was 1000.0)
 *   - mixed_layer_dtheta_k        (was 1.0)
 *   - unstable_top_*              (was clamped 0.5 * tropopause to [2500, 7000])
 *   - unstable_lapse_*            (was 0.004 + 0.002 * cape_scaling)
 *   - upper_trop_lapse_kpm        (was 0.005)
 *   - tropopause_depth_m          (was 1000.0)
 *   - strat_warming_kpm           (was 0.002)
 *   - moisture_decay_z_m          (was 2000.0)
 *   - moisture_scale_height_*     (was max(1500, 0.30 * tropopause))
 *   - base_moisture_*             (was clamp(qv * (0.85 + 0.15 * scale), 0.004, 0.024))
 *   - rh_cap                      (was 0.95)
 *
 * Surface theta / qv / tropopause are clamped to physically reasonable
 * minima inside build() to match the previous behavior.
 */
struct ParametricCAPEParams
{
    // Primary inputs (the user-facing knobs in environment.* today)
    double cape_target_jkg = 2500.0;
    double surface_theta_k = 300.0;
    double surface_qv_kgkg = 0.014;
    double tropopause_z_m = 12000.0;

    // Mixed layer
    double mixed_layer_top_m = 1000.0;
    double mixed_layer_dtheta_k = 1.0;

    // Conditionally unstable layer
    double unstable_top_floor_m = 2500.0;
    double unstable_top_ceil_m = 7000.0;
    double unstable_top_factor = 0.5;
    double unstable_lapse_base_kpm = 0.004;
    double unstable_lapse_cape_kpm = 0.002;

    // Upper troposphere + tropopause cap + stratosphere
    double upper_trop_lapse_kpm = 0.005;
    double tropopause_depth_m = 1000.0;
    double strat_warming_kpm = 0.002;

    // Moisture column
    double moisture_decay_z_m = 2000.0;
    double moisture_scale_height_floor_m = 1500.0;
    double moisture_scale_height_factor = 0.30;
    double base_moisture_low_factor = 0.85;
    double base_moisture_cape_factor = 0.15;
    double base_moisture_min = 0.004;
    double base_moisture_max = 0.024;
    double rh_cap = 0.95;

    // Reference CAPE at which the unstable_lapse_cape_kpm scaling factor
    // equals 1.0. The original inline code used 2500.0 J/kg here.
    double cape_reference_jkg = 2500.0;

    // Density floor applied after EOS. The original inline code clamped
    // rho >= 0.1 kg/m^3 to keep the hydrostatic profile away from the
    // singular high-altitude limit; the same floor is preserved here.
    double rho_floor_kgm3 = 0.1;
};

/**
 * @brief Procedural sounding from a CAPE target.
 *
 * Five-layer T(z) profile (Weisman & Klemp 1982 style):
 *   1. Mixed layer       (0 to mixed_layer_top): warm, well-mixed
 *   2. Conditionally unstable (mixed_layer_top to unstable_top): CAPE-driven
 *   3. Upper troposphere (unstable_top to tropopause): standard lapse
 *   4. Tropopause cap    (1 km isothermal lid)
 *   5. Stratosphere      (warming with height)
 *
 * Pressure is integrated upward from p0 using the analytical isothermal-layer
 * solution with T_avg between adjacent levels (second-order accurate for
 * arbitrary T(z)). Moisture decays exponentially above moisture_decay_z_m
 * and is capped at rh_cap of saturation at every level.
 */
class ParametricCAPESoundingSource final : public SoundingSource
{
public:
    explicit ParametricCAPESoundingSource(ParametricCAPEParams params);

    Sounding build(const std::vector<double>& z_m, double dz) const override;
    std::string describe() const override;

private:
    ParametricCAPEParams params_;
};

}  // namespace tmv::init
