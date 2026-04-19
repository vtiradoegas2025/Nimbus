/**
 * @file diffusion_step.hpp
 * @brief Runtime diffusion tendency driver.
 *
 * Provides the single entry point for applying diffusion tendencies
 * during the dynamics timestep. Internally manages its own buffers,
 * config checks, and tendency application.
 *
 * Extracted from src/core/dynamics.cpp.
 */

#pragma once

/**
 * @brief Computes and applies diffusion tendencies for one timestep.
 *
 * Checks whether diffusion is runtime-enabled (scheme exists, K_h or K_v > 0,
 * apply_to includes momentum and/or scalars). If so, computes tendencies via
 * the active diffusion scheme and applies them with Forward Euler integration,
 * NaN guards, and physical-bounds clamping.
 *
 * @param dt_dynamics The dynamics timestep in seconds.
 */
void apply_runtime_diffusion(double dt_dynamics);
