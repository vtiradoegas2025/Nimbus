#pragma once

#include "init/config/config_v1.hpp"

namespace tmv::runtime
{

// Canonical YAML omits these controls; hash membership cannot validate them.
inline bool matches_generated_metal_unserialized_defaults(const tmv::init::config::ConfigV1& config) noexcept
{
    return config.damping.lateral_sponge_enabled && config.damping.lateral_width == 20 &&
           config.damping.lateral_alpha == 0.05 && config.damping.w_only_top_sponge &&
           config.damping.w_only_depth_cells == 8 && config.damping.w_only_alpha_max == 0.20 &&
           config.extensions.solver.slow_rk2 && config.extensions.solver.lid_bc_gamma == -1.0 &&
           !config.extensions.solver.axis_u_m1_filter;
}

} // namespace tmv::runtime
