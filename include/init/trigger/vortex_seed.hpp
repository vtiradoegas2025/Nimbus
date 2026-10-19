#pragma once

#include "init/trigger/trigger_source.hpp"

struct SimulationStateView;

namespace tmv::init
{

/**
 * @brief Parameters for VortexSeedTrigger.
 *
 * Defaults are sensible for a vortex spin-up: r_max = 200 m, v_max = 25 m/s,
 * z_top = 4 km. Set center_x_m / center_y_m to the desired vortex-axis
 * location; defaults (0, 0) put the vortex at the cell origin.
 */
struct VortexSeedParams
{
    double r_max_m = 200.0;
    double v_max_ms = 25.0;
    double z_top_m = 4000.0;
    double center_x_m = 0.0;
    double center_y_m = 0.0;
    double center_z_m = 0.0;
};

/**
 * @brief Rankine vortex initial seed.
 *
 * Tangential profile:
 *   v_theta(r) = v_max * (r / r_max)         for r <= r_max
 *   v_theta(r) = v_max * (r_max / r)         for r > r_max
 * Active only for z <= z_top_m; zero contribution above. The azimuthal
 * component is added to the existing wind field, so the trigger composes
 * with an environmental hodograph already in place.
 *
 * The vortex axis is at (center_x_m, center_y_m); the tangential v_theta is
 * projected onto Cartesian (u, v) at every grid cell using the local azimuth
 * from the axis.
 */
class VortexSeedTrigger final : public TriggerSource
{
public:
    explicit VortexSeedTrigger(VortexSeedParams params);

    void apply() const override;
    void apply(const SimulationStateView& state) const override;
    std::string describe() const override;

private:
    VortexSeedParams params_;
};

}  // namespace tmv::init
