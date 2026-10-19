#pragma once

#include <array>

struct ComputeBackendRuntimeState;

namespace tmv
{
struct FlowExecutionInputs
{
    ComputeBackendRuntimeState* compute_runtime = nullptr;
    bool                        periodic_x      = false;
    bool                        periodic_y      = false;
    std::array<bool, 4>         open_lateral{{true, true, true, true}};
    bool                        open_top   = false;
    double                      frame_u_ms = 0.0;
    double                      frame_v_ms = 0.0;
    bool                        earth_fixed_terrain = false;
};
} // namespace tmv
