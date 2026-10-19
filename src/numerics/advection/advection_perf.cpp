#include "numerics/advection/advection_perf.hpp"

#include "compute/compute_backend.hpp"
#include "core/runtime/runtime_config.hpp"
#include "util/log.hpp"

#include <algorithm>

namespace
{

tmv::advection_detail::AdvectionPerfTotals perf_totals;

struct KernelStepAccumulator
{
    std::uint64_t calls = 0;
    std::uint64_t cpu_calls = 0;
    std::uint64_t gpu_calls = 0;
    double cpu_kernel_s = 0.0;
    double gpu_dispatch_s = 0.0;
    double sync_copy_s = 0.0;
    double kernel_total_s = 0.0;
};

KernelStepAccumulator kernel_step;

} // namespace

namespace tmv::advection_detail
{

bool perf_enabled()
{
    return global_perf_timing_enabled;
}

AdvectionPerfTotals& perf_totals()
{
    return ::perf_totals;
}

void record_vertical_flux_kernel_timing(
    const VerticalFluxKernelTiming& timing)
{
    ++kernel_step.calls;
    kernel_step.cpu_kernel_s += timing.cpu_kernel_s;
    kernel_step.gpu_dispatch_s += timing.gpu_dispatch_s;
    kernel_step.sync_copy_s += timing.sync_copy_s;
    kernel_step.kernel_total_s += timing.kernel_total_s;
    if (timing.gpu_dispatch)
    {
        ++kernel_step.gpu_calls;
    }
    else
    {
        ++kernel_step.cpu_calls;
    }
}

} // namespace tmv::advection_detail

namespace tmv
{

void reset_advection_perf_stats()
{
    ::perf_totals = advection_detail::AdvectionPerfTotals{};
}

void reset_advection_kernel_step_timing()
{
    kernel_step = KernelStepAccumulator{};
}

AdvectionKernelStepTiming current_advection_kernel_step_timing()
{
    AdvectionKernelStepTiming snapshot{};
    snapshot.calls = kernel_step.calls;
    snapshot.cpu_calls = kernel_step.cpu_calls;
    snapshot.gpu_calls = kernel_step.gpu_calls;
    snapshot.cpu_kernel_s = kernel_step.cpu_kernel_s;
    snapshot.gpu_dispatch_s = kernel_step.gpu_dispatch_s;
    snapshot.sync_copy_s = kernel_step.sync_copy_s;
    snapshot.kernel_total_s = kernel_step.kernel_total_s;
    snapshot.backend = compute_backend_kind_name(active_compute_backend_kind());
    snapshot.fallback_active = compute_backend_fallback_active();
    return snapshot;
}

void log_advection_perf_summary()
{
    if (!global_perf_timing_enabled)
    {
        return;
    }

    const double scalar_total = std::max(::perf_totals.scalar_total_s, 1e-9);
    tmv::log_info("[ADVECTION PERF] scalar_calls=", ::perf_totals.scalar_calls,
                  ", thermo_calls=", ::perf_totals.thermo_calls,
                  ", tracer_calls=", ::perf_totals.tracer_calls,
                  ", scalar_total_s=", ::perf_totals.scalar_total_s);
    tmv::log_info("  r_s=", ::perf_totals.r_s,
                  " (", (100.0 * ::perf_totals.r_s / scalar_total), "%)");
    tmv::log_info("  theta_s=", ::perf_totals.theta_s,
                  " (", (100.0 * ::perf_totals.theta_s / scalar_total), "%)");
    tmv::log_info("  z_s=", ::perf_totals.z_s,
                  " (", (100.0 * ::perf_totals.z_s / scalar_total), "%)");
    tmv::log_info("  diffusion_s=", ::perf_totals.diffusion_s,
                  " (", (100.0 * ::perf_totals.diffusion_s / scalar_total), "%)");
    tmv::log_info("  thermo_total_s=", ::perf_totals.thermo_total_s);
    tmv::log_info("  tracer_total_s=", ::perf_totals.tracer_total_s);
    tmv::log_info("  host_batch_s=", ::perf_totals.host_batch_s,
                  ", host_coefficient_validation_s=", ::perf_totals.host_coefficient_validation_s,
                  ", host_boundary_inventory_s=", ::perf_totals.host_boundary_inventory_s);
}

} // namespace tmv
