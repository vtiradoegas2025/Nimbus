/**
 * @file vortex_seed.cpp
 * @brief VortexSeedTrigger implementation. Adds a Rankine vortex to the
 *        Cartesian wind field for vortex spin-up.
 */

#include "init/trigger/vortex_seed.hpp"

#include "core/infra/physical_cell_domain.hpp"
#include "core/runtime/simulation_state.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tmv::init
{

namespace
{

/// Rankine combined-profile tangential velocity.
double rankine_vtheta(double r_m, double r_max_m, double v_max_ms)
{
    if (r_m <= 0.0)
    {
        return 0.0;
    }
    if (r_m <= r_max_m)
    {
        return v_max_ms * (r_m / r_max_m);
    }
    return v_max_ms * (r_max_m / r_m);
}

}  // namespace

VortexSeedTrigger::VortexSeedTrigger(VortexSeedParams params)
    : params_(params)
{
    if (!std::isfinite(params_.r_max_m) || params_.r_max_m <= 0.0)
    {
        throw std::invalid_argument("Vortex seed r_max_m must be positive and finite");
    }
    if (!std::isfinite(params_.v_max_ms))
    {
        throw std::invalid_argument("Vortex seed v_max_ms must be finite");
    }
    if (!std::isfinite(params_.z_top_m) || params_.z_top_m < 0.0)
    {
        throw std::invalid_argument("Vortex seed z_top_m must be non-negative and finite");
    }
    if (!std::isfinite(params_.center_x_m) || !std::isfinite(params_.center_y_m) ||
        !std::isfinite(params_.center_z_m))
    {
        throw std::invalid_argument("Vortex seed center coordinates must be finite");
    }
}

void VortexSeedTrigger::apply() const
{
    apply(global_simulation_state());
}

void VortexSeedTrigger::apply(const SimulationStateView& state) const
{
    const double r_max = params_.r_max_m;
    const double v_max = params_.v_max_ms;
    const double z_top = params_.z_top_m;
    const auto& grid = state.grid;
    const auto& geo = grid.geometry;
    auto& fields = state.fields;
    const PhysicalCellDomain domain = physical_scalar_domain(grid);
    if (domain.empty())
    {
        return;
    }

    // Vortex axis at (center_x, center_y). Project the tangential v_theta(r)
    // onto (u_x, u_y) at every grid cell using the local azimuth from the
    // axis. Additive so the seed composes with any environmental hodograph
    // already in the field.
    const double cx = params_.center_x_m;
    const double cy = params_.center_y_m;

    #pragma omp parallel for collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            const double dx = geo.r[i] - cx;
            const double dy = geo.theta[j] - cy;
            const double r = std::sqrt(dx * dx + dy * dy);
            if (r <= 0.0)
            {
                continue;
            }
            const double inv_r = 1.0 / r;

            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                if (geo.z[k] > z_top)
                {
                    break;
                }
                const double v_theta = rankine_vtheta(r, r_max, v_max);
                // (u, v) Cartesian projection of an azimuthal vector:
                //   u_x = -v_theta * dy / r
                //   u_y = +v_theta * dx / r
                fields.u[i][j][k] += static_cast<float>(-v_theta * dy * inv_r);
                fields.v[i][j][k] += static_cast<float>( v_theta * dx * inv_r);
            }
        }
    }
}

std::string VortexSeedTrigger::describe() const
{
    return "vortex_seed/rankine";
}

}  // namespace tmv::init
