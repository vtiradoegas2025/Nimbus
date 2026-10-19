#pragma once

#include "numerics/advection/advection.hpp"

#include <cstdint>

namespace tmv::advection_detail
{

struct AdvectionPerfTotals
{
    std::uint64_t scalar_calls = 0;
    std::uint64_t thermo_calls = 0;
    std::uint64_t tracer_calls = 0;
    double scalar_total_s = 0.0;
    double thermo_total_s = 0.0;
    double tracer_total_s = 0.0;
    double host_batch_s = 0.0;
    double host_coefficient_validation_s = 0.0;
    double host_boundary_inventory_s = 0.0;
    double r_s = 0.0;
    double theta_s = 0.0;
    double z_s = 0.0;
    double diffusion_s = 0.0;
};

struct VerticalFluxKernelTiming
{
    bool gpu_dispatch = false;
    double cpu_kernel_s = 0.0;
    double gpu_dispatch_s = 0.0;
    double sync_copy_s = 0.0;
    double kernel_total_s = 0.0;
};

bool perf_enabled();
AdvectionPerfTotals& perf_totals();
void record_vertical_flux_kernel_timing(
    const VerticalFluxKernelTiming& timing);

} // namespace tmv::advection_detail
