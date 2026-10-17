// Advection dispatch, conservative transport, guards, and runtime telemetry.

#include "numerics/advection/advection.hpp"
#include "numerics/advection/advection_base.hpp"
#include "numerics/advection/advection_cartesian.hpp"
#include "numerics/advection/advection_perf.hpp"
#include "compute/compute_backend.hpp"
#include "compute/compute_kernel_template.hpp"
#include "compute/parity_check.hpp"
#include "diagnostics/diagnostic_cell_volume.hpp"
#include "util/log.hpp"
#include "core/field/field_pool.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/process_state.hpp"
#include "core/field/field_bounds.hpp"
#include "core/runtime/simulation_state.hpp"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <exception>
#include <stdexcept>
#include <string>



namespace
{

using tmv::advection_detail::VerticalFluxKernelTiming;
using tmv::advection_detail::perf_enabled;
using tmv::advection_detail::record_vertical_flux_kernel_timing;
auto& g_advection_perf_totals = tmv::advection_detail::perf_totals();


/**
 * @brief Resizes a field to the active domain dimensions when needed.
 */
inline void ensure_field_shape(const SimulationStateView& execution_state, Field3D& field)
{

    if (field.size_r() != execution_state.grid.nr || field.size_th() != execution_state.grid.nth || field.size_z() != execution_state.grid.nz)
    {
        field.resize(execution_state.grid.nr, execution_state.grid.nth, execution_state.grid.nz, 0.0f);
    }
}

/**
 * @brief Copies one 3D field into another with shape validation.
 */
inline void copy_field(const SimulationStateView& execution_state, const Field3D& src, Field3D& dst)
{
    ensure_field_shape(execution_state, dst);

    if (src.size() == 0){return;}

    std::memcpy(dst.data(), src.data(), src.size() * sizeof(float));
}

/**
 * @brief Computes the row-major flat index for a 3D cell.
 */
inline size_t idx3(const SimulationStateView& execution_state, const int i, const int j, const int k)
{
    return (static_cast<size_t>(i) * static_cast<size_t>(execution_state.grid.nth) + static_cast<size_t>(j)) *
           static_cast<size_t>(execution_state.grid.nz) + static_cast<size_t>(k);
}

/**
 * @brief Seeds the x (i = 0, NR-1) and vertical (k = 0, NZ-1) boundary planes.
 *
 * The directional-split vertical kernels only write the interior, relying on
 * these face planes carrying the source values through for the zero-gradient
 * boundary condition.
 */
inline void copy_x_and_z_boundaries(const SimulationStateView& execution_state, const Field3D& src, Field3D& dst)
{
    if (src.size() == 0){return;}

    const int nr = execution_state.grid.nr;
    const int nth = execution_state.grid.nth;
    const int nz = execution_state.grid.nz;

    const float* src_data = src.data();
    float* dst_data = dst.data();

    const size_t plane_size = static_cast<size_t>(nth) * static_cast<size_t>(nz);

    std::memcpy(dst_data, src_data, plane_size * sizeof(float));

    if (nr > 1)
    {
        const size_t last_plane = (static_cast<size_t>(nr) - 1) * plane_size;
        std::memcpy(dst_data + last_plane, src_data + last_plane, plane_size * sizeof(float));
    }

    if (nz > 0 && nr > 2)
    {

        #pragma omp parallel for collapse(2)

        for (int i = 1; i < nr - 1; ++i)
        {
            for (int j = 0; j < nth; ++j)
            {
                const size_t k0 = idx3(execution_state, i, j, 0);
                dst_data[k0] = src_data[k0];

                if (nz > 1)
                {
                    const size_t ktop = idx3(execution_state, i, j, nz - 1);
                    dst_data[ktop] = src_data[ktop];
                }
            }
        }
    }
}

/**
 * @brief Returns a lowercase copy of an input string.
 */
std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),[](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

/**
 * @brief Returns true when the active accelerator can run the first offload.
 */
bool first_kernel_gpu_dispatch_enabled(const SimulationStateView& execution_state)
{
    if (compute_backend_fallback_active(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime")))
    {
        return false;
    }
    if (active_compute_backend_kind(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime")) == ComputeBackendKind::Cpu)
    {
        return false;
    }
    return active_compute_backend_supports(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime"), AdvectionKernelOp::VerticalFlux);
}

/**
 * @brief Logs selected vertical-flux kernel path once.
 */
void log_vertical_flux_kernel_path_once(const SimulationStateView& execution_state)
{
    static bool logged = false;
    if (logged || !log_normal_enabled())
    {
        return;
    }
    logged = true;

    if (first_kernel_gpu_dispatch_enabled(execution_state))
    {
        tmv::log_info("[ADVECTION][KERNEL] first_offload=tvd_vertical_flux_divergence",
                      ", dispatch=gpu",
                      ", template=", active_vertical_flux_template_id(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime")),
                      ", active_backend=", compute_backend_kind_name(active_compute_backend_kind(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime"))),
                      ", requested_backend=", requested_compute_backend_name(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime")),
                      ", fallback=", compute_backend_fallback_active(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime")) ? "active" : "off");
    }
    else
    {
        tmv::log_info("[ADVECTION][KERNEL] first_offload=tvd_vertical_flux_divergence",
                      ", dispatch=cpu_reference",
                      ", template=", active_vertical_flux_template_id(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime")),
                      ", active_backend=", compute_backend_kind_name(active_compute_backend_kind(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime"))),
                      ", requested_backend=", requested_compute_backend_name(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime")),
                      ", fallback=", compute_backend_fallback_active(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime")) ? "active" : "off");
    }
}

/**
 * @brief Checks whether the runtime should use numerics vertical transport.
 */
bool use_numerics_vertical_advection(const tmv::AdvectionRuntimeState& runtime)
{
    if (!runtime.scheme) return false;

    const std::string scheme_id = lower_copy(runtime.scheme->name());
    return scheme_id == "tvd" || scheme_id == "weno5";
}

bool advection_disabled(const tmv::AdvectionRuntimeState& runtime)
{
    return lower_copy(runtime.config.scheme_id) == "none";
}

tmv::CartesianFaceReconstruction conservative_reconstruction(
    const tmv::AdvectionRuntimeState& runtime)
{
    const std::string scheme_id = lower_copy(runtime.config.scheme_id);
    if (scheme_id == "tvd")
        return tmv::CartesianFaceReconstruction::MonotonicCentered;
    if (scheme_id == "weno5")
        return tmv::CartesianFaceReconstruction::WENO5;
    throw std::invalid_argument(
        "Cartesian conservative advection requires scheme tvd or weno5");
}

inline bool active_backend_is_cartesian_cgrid(const SimulationStateView& execution_state)
{
    return execution_state.configuration->coordinate_system == CoordinateSystem::Cartesian
        && execution_state.configuration->stagger_type      == StaggerType::CGrid;
}

void require_advection_field_shape(const SimulationStateView& execution_state, const Field3D& field, const char* label)
{
    if (execution_state.grid.nr <= 0 || execution_state.grid.nth <= 0 || execution_state.grid.nz <= 0 ||
        field.size_r() != execution_state.grid.nr || field.size_th() != execution_state.grid.nth ||
        field.size_z() != execution_state.grid.nz)
    {
        throw std::invalid_argument(
            std::string("advection ") + label + " shape does not match the grid");
    }
}

void require_finite_advection_field(const SimulationStateView& execution_state, const Field3D& field, const char* label,
                                    bool non_negative = false)
{
    require_advection_field_shape(execution_state, field, label);
    for (std::size_t index = 0; index < field.size(); ++index)
    {
        const float value = field.data()[index];
        if (!std::isfinite(static_cast<double>(value)) ||
            (non_negative && value < 0.0f))
        {
            throw std::invalid_argument(
                std::string("advection requires finite") +
                (non_negative ? " non-negative " : " ") + label +
                " values throughout the transport stencil");
        }
    }
}

/**
 * @brief Logs the selected runtime advection path once per execution.
 *
 * The Cartesian C-grid path runs conservative rho-weighted flux-form
 * transport. The collocated Cartesian path runs the (x, y) directional
 * split, optionally delegating vertical transport to the
 * numerics.advection scheme.
 */
void log_runtime_advection_path_once(const SimulationStateView& execution_state, const tmv::AdvectionRuntimeState& runtime)
{
    static bool logged = false;
    if (logged || !log_normal_enabled()){return;}
    logged = true;

    const char* split_axes_label = "(x/y)";

    if (advection_disabled(runtime))
    {
        tmv::log_info(
            "[ADVECTION] Runtime path: transported scalar fields held "
            "(dynamics momentum/continuity/pressure remain active).");
    }
    else if (active_backend_is_cartesian_cgrid(execution_state))
    {
        tmv::log_info("[ADVECTION] Runtime path: src/numerics/advection Cartesian C-grid "
                      "conservative rho-weighted flux-form transport (unsplit 3D, ",
                      runtime.config.scheme_id,
                      " face reconstruction; "
                      "continuity-consistent, free-stream preserving).");
    }
    else if (use_numerics_vertical_advection(runtime))
    {
        tmv::log_info("[ADVECTION] Runtime path: src/advection directional split ",
                      split_axes_label, " + src/numerics/advection/",
                      runtime.config.scheme_id, " (vertical tendency)");
    }
    else
    {
        tmv::log_info("[ADVECTION] Runtime path: src/advection directional split ",
                      split_axes_label,
                      " kernels (numerics.advection schemes are initialized but not selected "
                      "for vertical transport).");
    }
}

}

namespace tmv
{

/**
 * @brief Advects a scalar in the vertical direction with first-order upwinding.
 */
static void advect_scalar_1d_z_kernel(const SimulationStateView& execution_state, const Field3D& src, Field3D& dst, double dt, double)
{
    ensure_field_shape(execution_state, dst);
    copy_x_and_z_boundaries(execution_state, src, dst);

    const int nr = execution_state.grid.nr;
    const int nth = execution_state.grid.nth;
    const int nz = execution_state.grid.nz;

    const float* src_data = src.data();
    const float* w_data = execution_state.fields.w.data();
    float* dst_data = dst.data();

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < nr - 1; ++i)
    {
        for (int j = 0; j < nth; ++j)
        {
            for (int k = 1; k < nz - 1; ++k)
            {
                const size_t c = idx3(execution_state, i, j, k);
                const size_t km = idx3(execution_state, i, j, k - 1);
                const size_t kp = idx3(execution_state, i, j, k + 1);

                const double w_val = static_cast<double>(w_data[c]);
                double dq_dz = 0.0;
                if (w_val > 0.0)
                {
                    dq_dz = (static_cast<double>(src_data[c]) - static_cast<double>(src_data[km])) / execution_state.grid.dz;
                }
                else if (w_val < 0.0)
                {
                    dq_dz = (static_cast<double>(src_data[kp]) - static_cast<double>(src_data[c])) / execution_state.grid.dz;
                }
                const double q = static_cast<double>(src_data[c]);
                dst_data[c] = static_cast<float>(q - dt * w_val * dq_dz);
            }
        }
    }
}

/**
 * @brief Applies numerics-module vertical advection and falls back on failure.
 */
static void advect_scalar_1d_z_numerics_kernel(const SimulationStateView& execution_state,
    const Field3D& src, Field3D& dst, double dt, double,
    AdvectionRuntimeState& runtime)
{
    ensure_field_shape(execution_state, dst);
    const int nr = execution_state.grid.nr;
    const int nth = execution_state.grid.nth;
    const int nz = execution_state.grid.nz;
    copy_field(execution_state, src, dst);
    copy_x_and_z_boundaries(execution_state, src, dst);
    log_vertical_flux_kernel_path_once(execution_state);

    if (!runtime.scheme || !use_numerics_vertical_advection(runtime))
    {
        return;
    }

    AdvectionTendencies& tendencies = runtime.vertical_tendencies;
    AdvectionStateView state{};
    state.u = &execution_state.fields.u;
    state.v = &execution_state.fields.v;
    state.w = &execution_state.fields.w;
    state.q = &src;
    state.rho = &execution_state.fields.rho;
    state.grid = &(*execution_state.process_grid_metrics);
    AdvectionConfig runtime_cfg = runtime.config;
    if (!std::isfinite(dt) || dt <= 0.0)
    {
        throw std::invalid_argument(
            "vertical advection requires a positive finite timestep");
    }
    runtime_cfg.positivity_dt = dt;
    const auto kernel_t0 = std::chrono::steady_clock::now();
    VerticalFluxKernelTiming kernel_timing{};

    bool computed = false;
    const bool prefer_gpu_dispatch =
        first_kernel_gpu_dispatch_enabled(execution_state)
        && lower_copy(runtime.scheme->name()) == "tvd";

    if (prefer_gpu_dispatch)
    {
        const auto gpu_t0 = std::chrono::steady_clock::now();
        computed = dispatch_vertical_flux_template_backend(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime"), runtime_cfg, state, tendencies, nullptr);
        kernel_timing.gpu_dispatch_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - gpu_t0).count();
        kernel_timing.gpu_dispatch = computed;
    }

    if (prefer_gpu_dispatch && !computed)
    {
        if (execution_state.configuration->validation_policy.mode == tmv::GuardMode::Strict)
        {
            throw std::runtime_error(
                "qualified/strict advection rejected GPU-to-CPU backend substitution");
        }
        tmv::log_warn("[ADVECTION][KERNEL] GPU dispatch requested but fell back to CPU reference path");
    }

    if (!computed)
    {
        const auto cpu_t0 = std::chrono::steady_clock::now();
        try
        {
            runtime.scheme->compute_flux_divergence(
                runtime_cfg, state, tendencies, nullptr);
        }
        catch (const std::exception& e)
        {
            tmv::log_error("[ADVECTION] selected vertical advection failed ('",
                           runtime.scheme->name(), "'): ", e.what());
            throw;
        }
        kernel_timing.gpu_dispatch = false;
        kernel_timing.cpu_kernel_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - cpu_t0).count();
    }

    if (tendencies.dqdt_adv.size_r() != nr ||
        tendencies.dqdt_adv.size_th() != nth ||
        tendencies.dqdt_adv.size_z() != nz)
    {
        throw std::runtime_error(
            "selected vertical advection produced an unexpected tendency shape");
    }

    for (int i = 1; i < nr - 1; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                const double q_old = static_cast<double>(src[i][j][k]);
                const double dqdt =
                    static_cast<double>(tendencies.dqdt_adv[i][j][k]);
                const double q_new = q_old + dt * dqdt;
                if (!std::isfinite(dqdt) || !std::isfinite(q_new) ||
                    std::abs(q_new) >
                        static_cast<double>(std::numeric_limits<float>::max()))
                {
                    throw std::runtime_error(
                        "selected vertical advection produced a non-finite update");
                }
            }

    const auto sync_t0 = (kernel_timing.gpu_dispatch)
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < nr - 1; ++i)
    {
        for (int j = 0; j < nth; ++j)
        {
            for (int k = 1; k < nz - 1; ++k)
            {
                const double q_new = static_cast<double>(src[i][j][k]) +
                    dt * static_cast<double>(tendencies.dqdt_adv[i][j][k]);
                dst[i][j][k] = static_cast<float>(q_new);
            }
        }
    }

    if (kernel_timing.gpu_dispatch)
    {
        kernel_timing.sync_copy_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - sync_t0).count();
    }
    kernel_timing.kernel_total_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - kernel_t0).count();
    record_vertical_flux_kernel_timing(kernel_timing);
}

/**
 * @brief Advects a scalar field in 3D using directional splitting.
 */
void advect_scalar_3d(Field3D& scalar, double dt, double kappa,
                      bool positive_definite,
                      const CartesianCGridTransportContext* context,
                      AdvectionRuntimeState* runtime_state)
{
    return advect_scalar_3d(global_simulation_state(), scalar, dt, kappa, positive_definite, context, runtime_state);
}

void advect_scalar_3d(const SimulationStateView& execution_state, Field3D& scalar, double dt, double kappa,
                      bool positive_definite,
                      const CartesianCGridTransportContext* context,
                      AdvectionRuntimeState* runtime_state)
{
    if (!std::isfinite(dt) || dt <= 0.0 ||
        !std::isfinite(kappa) || kappa != 0.0 ||
        !std::isfinite(execution_state.grid.dr) || execution_state.grid.dr <= 0.0 ||
        !std::isfinite(execution_state.grid.dtheta) || execution_state.grid.dtheta <= 0.0 ||
        !std::isfinite(execution_state.grid.dz) || execution_state.grid.dz <= 0.0)
    {
        throw std::invalid_argument(
            "advection requires a positive finite timestep/spacing and "
            "kappa=0; configure the live-density diffusion scheme instead "
            "of transport-chained point diffusion");
    }
    require_finite_advection_field(execution_state, scalar, "scalar", positive_definite);
    require_finite_advection_field(execution_state, execution_state.fields.u, "x velocity");
    require_finite_advection_field(execution_state, execution_state.fields.v, "y velocity");
    require_finite_advection_field(execution_state, execution_state.fields.w, "vertical velocity");
    if (active_backend_is_cartesian_cgrid(execution_state) || context != nullptr)
    {
        require_advection_field_shape(execution_state, execution_state.fields.rho, "density");
        for (std::size_t index = 0; index < execution_state.fields.rho.size(); ++index)
        {
            const float value = execution_state.fields.rho.data()[index];
            if (!std::isfinite(static_cast<double>(value)) ||
                value < density_min_kgm3 || value > density_max_kgm3)
            {
                throw std::invalid_argument(
                    "advection density must be finite and within the declared physical bounds throughout the transport stencil");
            }
        }
    }
    AdvectionRuntimeState& runtime = runtime_state
        ? *runtime_state : (*execution_state.advection_runtime);
    if (advection_disabled(runtime))
        return;
    log_runtime_advection_path_once(execution_state, runtime);

    const bool perf_on = perf_enabled();
    const auto scalar_t0 = perf_on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    Field3D& scratch_a = runtime.scalar_scratch_a;
    Field3D& scratch_b = runtime.scalar_scratch_b;
    ensure_field_shape(execution_state, scratch_a);
    ensure_field_shape(execution_state, scratch_b);

    // Cartesian split dynamics: conservative rho-weighted flux-form transport.
    // Replaces the operator-inconsistent advective-horizontal + flux-vertical
    // split, whose spurious -phi*div(u) source (largest at the cloud-base
    // updraft) drove a 2dx theta dipole into the clamp bounds. The collocated
    // The C-grid may dispatch its native accelerator kernel. A retained
    // collocated context takes the same conservative CPU path and supplies
    // face-reconstructed mass fluxes from A-grid continuity.
    if (active_backend_is_cartesian_cgrid(execution_state))
    {
        const CartesianFaceReconstruction reconstruction =
            conservative_reconstruction(runtime);
        const bool native_weno5 =
            reconstruction == CartesianFaceReconstruction::WENO5;
        if (context != nullptr)
        {
            context->require_valid(execution_state, execution_state.grid.nr, execution_state.grid.nth, execution_state.grid.nz, dt);
        }
        // GPU path (GPU audit Phase 4.3): a one-scalar run of the
        // Phase 3.2 transport batch (WENO5 flux + divergence/recovery in one
        // submission). Serves the tracer,
        // which is not part of the thermodynamics batch.
        // The parity kernel name is DISTINCT from the thermo batch's:
        // sample() keeps a per-name call counter, and a shared name would
        // interleave the two call sites' cadence so one of them never
        // samples (observed: the thermo pairs vanished from the beast
        // report until this was split).
        const bool parity_sample = nimbus::parity::sample(
            "advect_scalar_single_cartesian_cgrid");
        Field3D snapshot;
        if (parity_sample)
        {
            snapshot = scalar;
        }
        const float* q_in = scalar.data();
        float* q_out = scalar.data();
        const float dt_kappa = static_cast<float>(dt * kappa);
        const double dx_d = execution_state.grid.dr;
        const double dy_d = execution_state.grid.dtheta;
        const double dz_d = execution_state.grid.dz;
        const int pd_flag = positive_definite ? 1 : 0;
        if (native_weno5 && primary_state_repairs_enabled &&
            dispatch_advect_scalars_conservative_cartesian_cgrid_backend(
                tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime"),
                execution_state.fields.u.data(),
                execution_state.fields.v.data(),
                execution_state.fields.w.data(),
                execution_state.fields.rho.data(),
                &q_in,
                &q_out,
                &dt_kappa,
                &pd_flag,
                1,
                execution_state.grid.nr,
                execution_state.grid.nth,
                execution_state.grid.nz,
                static_cast<float>(1.0 / dx_d),
                static_cast<float>(1.0 / dy_d),
                static_cast<float>(1.0 / dz_d),
                static_cast<float>(1.0 / (dx_d * dx_d)),
                static_cast<float>(1.0 / (dy_d * dy_d)),
                static_cast<float>(1.0 / (dz_d * dz_d)),
                static_cast<float>(dt),
                density_min_kgm3,
                context ? context->density_begin.data() : nullptr,
                context ? context->mass_flux_x.data() : nullptr,
                context ? context->mass_flux_y.data() : nullptr,
                context ? context->mass_flux_z.data() : nullptr,
                context != nullptr,
                nullptr,
                nullptr,
                nullptr,
                0.0,
                nullptr,
                nullptr,
                nullptr,
                static_cast<float>((*execution_state.domain_frame).velocity_ms.u_ms),
                static_cast<float>((*execution_state.domain_frame).velocity_ms.v_ms),
                &execution_state))
        {
            if (parity_sample)
            {
                const char* field_name =
                    (scalar.data() == execution_state.fields.tracer.data()) ? "tracer" : "scalar";
                advect_scalar_conservative_3d_cartesian(execution_state, snapshot, scratch_a, dt,
                                                        positive_definite, context,
                                                        nullptr,
                                                        ScalarMoistEnergyKind::None,
                                                        nullptr, reconstruction);
                nimbus::parity::check_field(
                    "advect_scalar_single_cartesian_cgrid", field_name,
                    scratch_a.data(), scalar.data(),
                    execution_state.grid.nr, execution_state.grid.nth, execution_state.grid.nz, nimbus::parity::Tier::Stencil);
            }
            if (perf_on)
            {
                g_advection_perf_totals.scalar_total_s +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - scalar_t0).count();
                ++g_advection_perf_totals.scalar_calls;
            }
            return;
        }

        advect_scalar_conservative_3d_cartesian(execution_state, scalar, scratch_a, dt,
                                                positive_definite, context,
                                                nullptr,
                                                ScalarMoistEnergyKind::None,
                                                nullptr, reconstruction);
        copy_field(execution_state, scratch_a, scalar);

        if (perf_on)
        {
            g_advection_perf_totals.scalar_total_s +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - scalar_t0).count();
            ++g_advection_perf_totals.scalar_calls;
        }
        return;
    }

    if (context != nullptr)
    {
        context->require_valid(execution_state, execution_state.grid.nr, execution_state.grid.nth, execution_state.grid.nz, dt);
        advect_scalar_conservative_3d_cartesian(execution_state,
            scalar, scratch_a, dt, positive_definite, context, nullptr,
            ScalarMoistEnergyKind::None, nullptr,
            conservative_reconstruction(runtime));
        copy_field(execution_state, scratch_a, scalar);

        if (perf_on)
        {
            g_advection_perf_totals.scalar_total_s +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - scalar_t0).count();
            ++g_advection_perf_totals.scalar_calls;
        }
        return;
    }

    const double dt_half = dt * 0.5;

    // ── Select directional kernels ──
    //
    // Each lambda wraps GPU-try + CPU-fallback for its direction on the
    // collocated Cartesian grid: x/y kernels, no 1/r factor, no periodic
    // wrap.

    auto step_h1 = [&](const Field3D& src, Field3D& dst, double dt_step) {
        ensure_field_shape(execution_state, dst);
        // Parity harness (GPU.md Phase 1): replay a sampled
        // GPU dispatch through the CPU kernel and compare.
        const bool parity_sample = nimbus::parity::sample("advect_x");
        if (!dispatch_advection_x_backend(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime"),
                src.data(), execution_state.fields.u.data(), dst.data(),
                execution_state.grid.nr, execution_state.grid.nth, execution_state.grid.nz, static_cast<float>(execution_state.grid.dr), static_cast<float>(dt_step)))
        {
            advect_scalar_1d_x_kernel_cartesian(execution_state, src, dst, dt_step);
        }
        else if (parity_sample)
        {
            auto ref = FieldPool::instance().scoped_acquire_copy(dst);
            advect_scalar_1d_x_kernel_cartesian(execution_state, src, ref.field, dt_step);
            nimbus::parity::check_field("advect_x", "scalar",
                                        ref.field.data(), dst.data(),
                                        execution_state.grid.nr, execution_state.grid.nth, execution_state.grid.nz,
                                        nimbus::parity::Tier::Pointwise);
        }
    };

    auto step_h2 = [&](const Field3D& src, Field3D& dst, double dt_step) {
        ensure_field_shape(execution_state, dst);
        const bool parity_sample = nimbus::parity::sample("advect_y");
        if (!dispatch_advection_y_backend(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime"),
                src.data(), execution_state.fields.v.data(), dst.data(),
                execution_state.grid.nr, execution_state.grid.nth, execution_state.grid.nz, static_cast<float>(execution_state.grid.dtheta), static_cast<float>(dt_step)))
        {
            advect_scalar_1d_y_kernel_cartesian(execution_state, src, dst, dt_step);
        }
        else if (parity_sample)
        {
            auto ref = FieldPool::instance().scoped_acquire_copy(dst);
            advect_scalar_1d_y_kernel_cartesian(execution_state, src, ref.field, dt_step);
            nimbus::parity::check_field("advect_y", "scalar",
                                        ref.field.data(), dst.data(),
                                        execution_state.grid.nr, execution_state.grid.nth, execution_state.grid.nz,
                                        nimbus::parity::Tier::Pointwise);
        }
    };

    auto step_z = [&](const Field3D& src, Field3D& dst, double dt_step) {
        if (use_numerics_vertical_advection(runtime))
            advect_scalar_1d_z_numerics_kernel(execution_state,
                src, dst, dt_step, kappa, runtime);
        else
            advect_scalar_1d_z_kernel(execution_state, src, dst, dt_step, kappa);
    };

    auto step_diff = [&](const Field3D& src, Field3D& dst, double dt_step, double k) {
        apply_diffusion_kernel_cartesian(execution_state, src, dst, dt_step, k);
    };

    // ── Strang split: x(dt/2) → y(dt/2) → z(dt) → y(dt/2) → x(dt/2) → diffusion ──
    //
    // Each directional step runs individually on the collocated Cartesian grid.

    // ── Pre-vertical: x(dt/2) → y(dt/2) ──
    const auto h1_t0 = perf_on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    step_h1(scalar, scratch_a, dt_half);
    if (perf_on)
        g_advection_perf_totals.r_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - h1_t0).count();

    const auto h2_t0 = perf_on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    step_h2(scratch_a, scratch_b, dt_half);
    if (perf_on)
        g_advection_perf_totals.theta_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - h2_t0).count();

    // ── Vertical step ──
    const auto z_t0 = perf_on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    step_z(scratch_b, scratch_a, dt);
    if (perf_on)
        g_advection_perf_totals.z_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - z_t0).count();

    // ── Post-vertical: y(dt/2) → x(dt/2) → diffusion ──
    const auto h2_post_t0 = perf_on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    step_h2(scratch_a, scratch_b, dt_half);
    if (perf_on)
        g_advection_perf_totals.theta_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - h2_post_t0).count();

    const auto h1_post_t0 = perf_on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    step_h1(scratch_b, scratch_a, dt_half);
    if (perf_on)
        g_advection_perf_totals.r_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - h1_post_t0).count();

    const auto diff_t0 = perf_on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    step_diff(scratch_a, scalar, dt, kappa);
    if (perf_on)
        g_advection_perf_totals.diffusion_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - diff_t0).count();

    if (perf_on)
    {
        g_advection_perf_totals.scalar_total_s +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - scalar_t0).count();
        ++g_advection_perf_totals.scalar_calls;
    }
}

} // namespace tmv
