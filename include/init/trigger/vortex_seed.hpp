#pragma once

#include "init/trigger/trigger_source.hpp"

namespace tmv::init
{

/**
 * @brief Parameters for VortexSeedTrigger.
 *
 * Defaults are sensible for an axisymmetric tornado-scheme spin-up:
 *   r_max = 200 m, v_max = 25 m/s, z_top = 4 km. The vortex axis defaults
 *   to the domain center on cylindrical grids (r = 0). For Cartesian
 *   grids, set center_x_m / center_y_m to the desired axis location;
 *   defaults (0, 0) put the vortex at the cell origin.
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
 * with an environmental hodograph (e.g. parametric WK winds at the
 * surroundings, vortex spin near the axis).
 *
 * Coordinate handling:
 *   - Cylindrical (both collocated and C-grid): the vortex axis is at
 *     r = 0 (the cylindrical pole) and the perturbation is added to
 *     v[i][j][k] (the azimuthal component) at every (j) since the
 *     tornado dynamics is axisymmetric. center_x_m / center_y_m are
 *     ignored on cylindrical grids; the vortex is always at r = 0.
 *   - Cartesian: the vortex axis is at (center_x_m, center_y_m). The
 *     tangential v_theta is projected onto Cartesian (u, v) at every
 *     grid cell using the local azimuth from the axis.
 *
 * Acceptable schemes:
 *   - tornado / tornado_cgrid (the natural fit; recommended)
 *   - cartesian (works as a 3D Rankine vortex in a Cartesian box)
 *   - supercell / supercell_cgrid (allowed but unusual; users typically
 *     use warm_bubble instead)
 */
class VortexSeedTrigger final : public TriggerSource
{
public:
    explicit VortexSeedTrigger(VortexSeedParams params);

    void apply() const override;
    std::string describe() const override;

private:
    VortexSeedParams params_;
};

}  // namespace tmv::init
