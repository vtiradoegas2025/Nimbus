#pragma once

struct SimulationStateView;

#include "numerics/numerics_base.hpp"

namespace tmv
{
extern GridMetrics& global_grid_metrics;

void initialize_numerics();
void initialize_numerics(const SimulationStateView& execution_state);
void refresh_grid_metrics_from_terrain();
void refresh_grid_metrics_from_terrain(const SimulationStateView& execution_state);
double choose_runtime_timestep();
double choose_runtime_timestep(const SimulationStateView& execution_state);

// Maximum wind component magnitude from the most recent CFL scan.
double last_max_flow_speed();
}

using tmv::global_grid_metrics;
using tmv::initialize_numerics;
using tmv::refresh_grid_metrics_from_terrain;
using tmv::choose_runtime_timestep;
using tmv::last_max_flow_speed;
