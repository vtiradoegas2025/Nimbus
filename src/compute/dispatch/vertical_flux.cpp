#include "compute/bridge/vertical_flux.hpp"

#include "compute/compute_backend.hpp"
#include "compute/dispatch/internal/vertical_flux.hpp"
#include "compute/parity_check.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace
{


const AdvectionComputeKernels* active_advection_kernels(ComputeBackendRuntimeState& compute)
{
    const ComputeBackend* backend = active_compute_backend(compute);
    return backend != nullptr ? backend->advection_kernels() : nullptr;
}

AdvectionComputeKernels* mutable_advection_kernels(ComputeBackendRuntimeState& compute)
{
    ComputeBackend* backend = mutable_compute_backend(compute);
    return backend != nullptr ? backend->advection_kernels() : nullptr;
}

} // namespace

void set_active_vertical_flux_template_id(const std::string& template_id)
{
    return set_active_vertical_flux_template_id(global_compute_backend_runtime_state, template_id);
}

void set_active_vertical_flux_template_id(ComputeBackendRuntimeState& compute, const std::string& template_id)
{
    if (!has_vertical_flux_template(template_id))
    {
        return;
    }
    const auto canonical = nimbus::compute::dispatch_internal::canonicalize_id(template_id);
    compute.vertical_flux_template_id = canonical == "tvdverticalfluxv1" || canonical == "tvddefaultv1"
        ? "tvd_vertical_flux_v1" : "tvd_vertical_flux_v2";
}

const std::string& active_vertical_flux_template_id()
{
    return active_vertical_flux_template_id(global_compute_backend_runtime_state);
}

const std::string& active_vertical_flux_template_id(ComputeBackendRuntimeState& compute)
{
    return compute.vertical_flux_template_id;
}

bool has_vertical_flux_template(const std::string& template_id)
{
    const std::string id =
        nimbus::compute::dispatch_internal::canonicalize_id(template_id);
    return id == "tvdverticalfluxv1" || id == "tvddefaultv1" ||
           id == "tvdverticalfluxv2" || id == "tvddefaultv2";
}

std::vector<VerticalFluxTemplateDescriptor> list_vertical_flux_templates()
{
    return list_vertical_flux_templates(global_compute_backend_runtime_state);
}

std::vector<VerticalFluxTemplateDescriptor> list_vertical_flux_templates(ComputeBackendRuntimeState& compute)
{
    const AdvectionComputeKernels* kernels = active_advection_kernels(compute);
    const bool gpu_ready = kernels != nullptr && kernels->supports(AdvectionKernelOp::VerticalFlux);
    return {
        {"tvd_vertical_flux_v2",
         "Float-precision TVD vertical flux (default). Same algorithm as v1 but "
         "operates in float for compiler auto-vectorization across columns.",
         gpu_ready},
        {"tvd_vertical_flux_v1",
         "Double-precision TVD vertical flux (reference). Higher precision, lower throughput.",
         gpu_ready},
    };
}

bool dispatch_vertical_flux_template_backend(
    const AdvectionConfig& cfg,
    const AdvectionStateView& state,
    AdvectionTendencies& tendencies,
    AdvectionDiagnostics* diag_opt)
{
    return dispatch_vertical_flux_template_backend(global_compute_backend_runtime_state, cfg, state, tendencies, diag_opt);
}

bool dispatch_vertical_flux_template_backend(ComputeBackendRuntimeState& compute,
    const AdvectionConfig& cfg,
    const AdvectionStateView& state,
    AdvectionTendencies& tendencies,
    AdvectionDiagnostics* diag_opt)
{
    const std::string id =
        nimbus::compute::dispatch_internal::canonicalize_id(
            compute.vertical_flux_template_id);
    const bool is_v1 = (id == "tvdverticalfluxv1" || id == "tvddefaultv1");
    const bool is_v2 = (id == "tvdverticalfluxv2" || id == "tvddefaultv2");
    if (!is_v1 && !is_v2)
    {
        return false;
    }
    if (state.grid != nullptr && state.grid->terrain_metrics_active)
    {
        // The accelerator kernel has the same Cartesian physical-w contract
        // as the CPU templates. Mapped terrain is handled by retained C-grid
        // contravariant mass fluxes, not by substituting a local dz array.
        return false;
    }

    // Try GPU dispatch if backend supports it (shared by v1 and v2)
    AdvectionComputeKernels* kernels = mutable_advection_kernels(compute);
    const bool native_path_selected =
        !compute_backend_fallback_active(compute)
        && active_compute_backend_kind(compute) != ComputeBackendKind::Cpu
        && kernels != nullptr
        && kernels->supports(AdvectionKernelOp::VerticalFlux);
    if (native_path_selected)
    {
        if (!state.q || !state.w || !state.grid)
        {
            return false;
        }
        const int nr = state.q->size_r();
        const int nth = state.q->size_th();
        const int nz = state.q->size_z();
        if (nr <= 0 || nth <= 0 || nz <= 0)
        {
            return false;
        }

        // Marshal limiter string to integer id
        auto limiter_kind =
            nimbus::compute::dispatch_internal::TVDLimiterKind::mc;
        nimbus::compute::dispatch_internal::parse_tvd_limiter_kind(
            cfg.limiter_id, limiter_kind);
        const int limiter_int = static_cast<int>(limiter_kind);

        Field3D candidate_dqdt(nr, nth, nz, 0.0f);

        const std::vector<double>& dz_levels = state.grid->dz;
        const double dt_safe =
            std::max(std::abs(cfg.positivity_dt), 1.0e-12);

        VerticalFluxDispatchResult result{};
        const VerticalFluxDispatch dispatch{
            state.q->data(), state.w->data(), candidate_dqdt.data(),
            nr, nth, nz,
            dz_levels.data(), static_cast<int>(dz_levels.size()),
            limiter_int, cfg.positivity, dt_safe,
            cfg.cfl_target, &result
        };
        const bool parity_sample =
            nimbus::parity::sample("tvd_vertical_flux");
        AdvectionTendencies reference;
        if (parity_sample)
        {
            if (is_v2)
            {
                nimbus::compute::dispatch_internal::
                    dispatch_tvd_vertical_flux_v2(
                        cfg, state, reference, nullptr);
            }
            else
            {
                nimbus::compute::dispatch_internal::
                    dispatch_tvd_vertical_flux_v1(
                        cfg, state, reference, nullptr);
            }
        }

        const bool ok = kernels->dispatch_vertical_flux(dispatch);

        if (ok)
        {
            if (parity_sample)
            {
                nimbus::parity::check_field(
                    "tvd_vertical_flux", "dqdt_adv",
                    reference.dqdt_adv.data(), candidate_dqdt.data(),
                    nr, nth, nz, nimbus::parity::Tier::Pointwise,
                    nimbus::parity::ComparisonDomain::StrictInterior);
            }
            tendencies.dqdt_adv = std::move(candidate_dqdt);
            if (diag_opt)
            {
                diag_opt->max_cfl_z = result.max_cfl_z;
                diag_opt->suggested_dt = result.suggested_dt;
            }
            return true;
        }
        // A native path was selected, so report its result without silently
        // substituting the CPU implementation. The orchestration layer owns
        // fallback policy and strict-mode rejection.
        return false;
    }

    // No accelerator path was selected. Preserve the direct CPU template
    // behavior used by CPU backends and standalone callers.
    if (is_v2)
    {
        return nimbus::compute::dispatch_internal::dispatch_tvd_vertical_flux_v2(
            cfg, state, tendencies, diag_opt);
    }
    return nimbus::compute::dispatch_internal::dispatch_tvd_vertical_flux_v1(
        cfg, state, tendencies, diag_opt);
}
