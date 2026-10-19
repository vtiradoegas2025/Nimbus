#pragma once

struct SimulationStateView;

// Returns true exactly when apply_runtime_diffusion will execute an operator
// and may commit prognostic state. Invalid active configurations fail closed
// with the same validation used by the apply path.
bool runtime_diffusion_enabled(const SimulationStateView& state);

// Computes and applies one guarded Forward-Euler diffusion update.
void apply_runtime_diffusion(const SimulationStateView& state,
                             double dt_dynamics);
void apply_runtime_diffusion(double dt_dynamics);
