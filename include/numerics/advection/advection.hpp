#pragma once

#include <cstdint>

#include "core/field/field3d.hpp"

/**
 * @file advection.hpp
 * @brief Public advection entry points used by the simulation runtime.
 *
 * Declares high-level helpers that advect scalar, tracer, and
 * thermodynamic fields on the model grid.
 * These functions wrap the configured advection implementation.
 */

/**
 * @brief Advects a scalar field over one advection step.
 * @param scalar Scalar field updated in place.
 * @param dt Advection step length in seconds.
 * @param kappa Diffusion-like stabilization coefficient.
 */
void advect_scalar_3d(Field3D& scalar, double dt, double kappa = 0.01);

/**
 * @brief Advects thermodynamic fields over one advection step.
 * @param dt_advect Advection step length in seconds.
 * @param kappa_theta Stabilization coefficient for potential temperature.
 * @param kappa_moisture Stabilization coefficient for moisture scalars.
 */
void advect_thermodynamics_3d(double dt_advect,
                              double kappa_theta = 0.01,
                              double kappa_moisture = 0.01);

/**
 * @brief Advects the passive tracer field over one advection step.
 * @param dt_advect Advection step length in seconds.
 * @param kappa Diffusion-like stabilization coefficient.
 */
void advect_tracer_3d(double dt_advect, double kappa = 0.01);

/**
 * @brief Per-step timing snapshot for the first offloaded advection kernel.
 */
struct AdvectionKernelStepTiming
{
    std::uint64_t calls = 0;
    std::uint64_t cpu_calls = 0;
    std::uint64_t gpu_calls = 0;
    double cpu_kernel_s = 0.0;
    double gpu_dispatch_s = 0.0;
    double sync_copy_s = 0.0;
    double kernel_total_s = 0.0;
    const char* backend = "cpu";
    bool fallback_active = false;
};

/**
 * @brief Clears per-step kernel timing counters.
 */
void reset_advection_kernel_step_timing();

/**
 * @brief Returns the current per-step kernel timing counters.
 */
AdvectionKernelStepTiming current_advection_kernel_step_timing();

/**
 * @brief Resets advection performance counters.
 */
void reset_advection_perf_stats();

/**
 * @brief Logs a summary of advection performance counters.
 */
void log_advection_perf_summary();
