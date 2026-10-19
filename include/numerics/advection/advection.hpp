#pragma once

struct SimulationStateView;

#include <cstdint>

#include "numerics/advection/advection_base.hpp"

// Public scalar-transport entry points and performance telemetry.

namespace tmv
{

class CartesianCGridTransportContext;
class MicrophysicsScheme;

/**
 * @brief Advects a scalar field over one advection step.
 * @param scalar Scalar field updated in place.
 * @param dt Advection step length in seconds.
 * @param kappa Must be zero. Use the diffusion scheme for scalar mixing.
 * @param context Optional completed split-continuity mass-flux history.
 */
/// positive_definite routes moisture/tracer scalars through the
/// Skamarock-Weisman 2006 flux renormalization on the C-grid path.
void advect_scalar_3d(Field3D& scalar, double dt, double kappa = 0.0,
                      bool positive_definite = false,
                      const CartesianCGridTransportContext* context = nullptr,
                      AdvectionRuntimeState* runtime_state = nullptr);
    void advect_scalar_3d(const SimulationStateView& execution_state, Field3D& scalar, double dt, double kappa = 0.0,
                      bool positive_definite = false,
                      const CartesianCGridTransportContext* context = nullptr,
                      AdvectionRuntimeState* runtime_state = nullptr);

/**
 * @brief Advects thermodynamic fields over one advection step.
 * @param dt_advect Advection step length in seconds.
 * @param kappa_theta Must be zero; retained for source compatibility.
 * @param kappa_moisture Must be zero; retained for source compatibility.
 */
void advect_thermodynamics_3d(double dt_advect,
                              double kappa_theta = 0.0,
                              double kappa_moisture = 0.0,
                              const CartesianCGridTransportContext* context = nullptr,
                              AdvectionRuntimeState* runtime_state = nullptr);
    void advect_thermodynamics_3d(const SimulationStateView& execution_state, double dt_advect,
                              double kappa_theta = 0.0,
                              double kappa_moisture = 0.0,
                              const CartesianCGridTransportContext* context = nullptr,
                              AdvectionRuntimeState* runtime_state = nullptr);

/**
 * Advect all scheme-owned prognostic moments with the same retained mass
 * fluxes and positive-definite scalar equation as the hydrometeor masses.
 */
void advect_microphysics_prognostics_3d(
    MicrophysicsScheme& scheme,
    double dt_advect,
    const CartesianCGridTransportContext* context = nullptr,
    AdvectionRuntimeState* runtime_state = nullptr);
    void advect_microphysics_prognostics_3d(const SimulationStateView& execution_state,
    MicrophysicsScheme& scheme,
    double dt_advect,
    const CartesianCGridTransportContext* context = nullptr,
    AdvectionRuntimeState* runtime_state = nullptr);

/// Chained post-advection scalar diffusion coefficient. Default 0: this
/// transport-chained diffusion is physically vestigial (the old 0.01 m^2/s
/// gave a months-long e-folding on the 2dx mode - science audit) and is
/// redundant with the configurable diffusion scheme (the YAML diffusion
/// block). Use that block for explicit scalar diffusion; the CPU path
/// short-circuits and the GPU fused term vanishes when kappa = 0.
inline constexpr double kTracerDefaultKappa = 0.0;

/**
 * @brief Advects the passive tracer field over one advection step.
 *        No-op when the preceding advect_thermodynamics_3d C-grid batch
 *        already carried the tracer.
 * @param dt_advect Advection step length in seconds.
 * @param kappa Must be zero. Use the diffusion scheme for scalar mixing.
 */
void advect_tracer_3d(double dt_advect, double kappa = kTracerDefaultKappa,
                      const CartesianCGridTransportContext* context = nullptr,
                      AdvectionRuntimeState* runtime_state = nullptr);
    void advect_tracer_3d(const SimulationStateView& execution_state, double dt_advect, double kappa = kTracerDefaultKappa,
                      const CartesianCGridTransportContext* context = nullptr,
                      AdvectionRuntimeState* runtime_state = nullptr);

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

} // namespace tmv

using tmv::advect_scalar_3d;
using tmv::advect_thermodynamics_3d;
using tmv::advect_microphysics_prognostics_3d;
using tmv::AdvectionRuntimeState;
using tmv::kTracerDefaultKappa;
using tmv::advect_tracer_3d;
using tmv::AdvectionKernelStepTiming;
using tmv::reset_advection_kernel_step_timing;
using tmv::current_advection_kernel_step_timing;
using tmv::reset_advection_perf_stats;
using tmv::log_advection_perf_summary;
