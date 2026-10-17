#include "turbulence/base/native_collocated_eddy_viscosity.hpp"
// Smagorinsky closure with CPU/GPU parity replay.

#include "smagorinsky.hpp"
#include "dynamic_smagorinsky.hpp"
#include "compute/compute_backend.hpp"
#include "compute/compute_kernel_template.hpp"
#include "compute/parity_check.hpp"
#include "core/field/field_bounds.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/runtime/runtime_config.hpp"
#include "turbulence/runtime.hpp"
#include "util/grid_metric_utils.hpp"
#include "util/log.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>


namespace tmv
{
namespace
{
using SmagPerfClock = std::chrono::steady_clock;

SmagPerfClock::time_point smag_perf_start()
{
    return global_perf_timing_enabled
        ? SmagPerfClock::now()
        : SmagPerfClock::time_point{};
}

void smag_perf_finish(
    TurbulencePerfStage stage,
    SmagPerfClock::time_point started)
{
    if (global_perf_timing_enabled)
    {
        record_turbulence_perf(
            stage,
            std::chrono::duration<double>(
                SmagPerfClock::now() - started).count());
    }
}

}

/**
 * @brief Initializes the Smagorinsky turbulence scheme.
 */
SmagorinskyScheme::SmagorinskyScheme()
    : Cs_(turbulence_constants::C_s_default),
      Pr_t_(turbulence_constants::Pr_t_default),
      Sc_t_(turbulence_constants::Sc_t_default),
      nu_t_max_(1000.0),
      K_max_(1000.0) {
}

/**
 * @brief Gets the required fields.
 */
int SmagorinskyScheme::required_fields() const
{
    return static_cast<int>(TurbulenceRequirements::BASIC);
}

/**
 * @brief Initializes the Smagorinsky turbulence scheme.
 */
void SmagorinskyScheme::initialize(const TurbulenceConfig& cfg)
{
    Cs_ = cfg.Cs;
    Pr_t_ = cfg.Pr_t;
    Sc_t_ = cfg.Sc_t;
    nu_t_max_ = cfg.nu_t_max;
    K_max_ = cfg.K_max;

    tmv::log_info("Initialized Smagorinsky Turbulence:");
    tmv::log_info("  Cs = ", Cs_);
    if (cfg.dynamic_Cs)
        tmv::log_info(
            "  Dynamic coefficient = Favre Germano-Lilly, Cs upper bound = ",
            Cs_);
    tmv::log_info("  Pr_t = ", Pr_t_, ", Sc_t = ", Sc_t_);
    if (cfg.stability_correction == kLillyRichardsonModel)
        tmv::log_info("  Stability correction = Lilly equilibrium, Pr_t = ", Pr_t_);
    tmv::log_info("  Max nu_t = ", nu_t_max_, " m²/s");
}

/**
 * @brief Computes the Smagorinsky turbulence scheme.
 */
void SmagorinskyScheme::compute(const TurbulenceConfig&    cfg,
                                const GridMetrics&         grid,
                                const TurbulenceStateView& state,
                                TurbulenceTendencies&      tend,
                                TurbulenceDiagnostics*     diag_opt)
{
    eddy_viscosity::validate_scheme_inputs(cfg, grid, state, false);
    if (cfg.dynamic_Cs)
    {
        if (state.require_native_dispatch)
            throw std::runtime_error("dynamic Smagorinsky is CPU-owned and cannot satisfy required native dispatch");
        compute_dynamic(cfg, grid, state, tend, diag_opt);
        return;
    }
    nu_t_host_current_             = false;
    tend.resident_source_unchanged = false;
    tend.mapped_boundary_power_w   = std::numeric_limits<double>::quiet_NaN();
    // The momentum/scalar writers below clear their own boundary-owned cells
    // and overwrite their interiors. Resize without an eager full-field fill.
    tend.dudt_sgs.resize(state.NR, state.NTH, state.NZ);
    tend.dvdt_sgs.resize(state.NR, state.NTH, state.NZ);
    tend.dwdt_sgs.resize(state.NR, state.NTH, state.NZ);
    tend.dthetadt_sgs.resize(state.NR, state.NTH, state.NZ);
    tend.dqvdt_sgs.resize(state.NR, state.NTH, state.NZ);
    // Smagorinsky has no TKE source, so this one output is authoritatively zero.
    tend.dtkedt_sgs.resize(state.NR, state.NTH, state.NZ, 0.0f);
    const auto publish_deferred_tendencies = [&]()
    {
        tend.dudt_sgs.fill(0.0f);
        tend.dvdt_sgs.fill(0.0f);
        tend.dwdt_sgs.fill(0.0f);
        tend.dthetadt_sgs.fill(0.0f);
        tend.dqvdt_sgs.fill(0.0f);
        tend.dtkedt_sgs.fill(0.0f);
        tend.resident_source_unchanged = false;
    };
    const auto publish_diagnostics = [&]()
    {
        if (diag_opt == nullptr)
            return;
        diag_opt->nu_t    = nu_t_;
        diag_opt->K_theta = K_theta_;
        diag_opt->K_q     = K_q_;
        diag_opt->K_tke.resize(state.NR, state.NTH, state.NZ, 0.0f);
        diag_opt->shear_prod.resize(state.NR, state.NTH, state.NZ, 0.0f);
        diag_opt->buoy_prod.resize(state.NR, state.NTH, state.NZ, 0.0f);
        diag_opt->dissipation.resize(state.NR, state.NTH, state.NZ, 0.0f);
        diag_opt->Cs_eff.resize(state.NR, state.NTH, state.NZ, static_cast<float>(Cs_));
        diag_opt->Pr_t_eff.resize(state.NR, state.NTH, state.NZ, static_cast<float>(Pr_t_));
    };

    if (!smag_parity_replay_ && !state.c_grid && state.compute_runtime &&
        state.compute_runtime->active_kind != ComputeBackendKind::Cpu)
    {
        collocated_eddy_viscosity::NativeResult result;
        if (collocated_eddy_viscosity::compute_native(cfg, grid, state, false, result))
        {
            nu_t_              = std::move(result.closure[0]);
            K_theta_           = std::move(result.closure[1]);
            K_q_               = std::move(result.closure[2]);
            nu_t_host_current_ = true;
            collocated_eddy_viscosity::publish_tendencies(result, tend);
            publish_diagnostics();
            return;
        }
        if (state.require_native_dispatch || !state.compute_runtime->config.allow_fallback)
            throw std::runtime_error("fallback-off collocated Smagorinsky requires native flat stress transport");
    }

    const bool periodic_x = eddy_viscosity::periodic_x_topology(state);
    const bool periodic_y = eddy_viscosity::periodic_y_topology(state);

    const bool native_dispatch_eligible =
        state.c_grid && state.compute_runtime != nullptr &&
        active_compute_backend_supports(*state.compute_runtime, TurbulenceKernelOp::Smagorinsky) &&
        !grid_metric::has_terrain_metrics(grid) && cfg.stability_correction == "none" && cfg.filter_width != "user";

    // A qualified fallback-off run must reject an unavailable or ineligible
    // native path before computing any CPU scientific tendency. Parity replay
    // is the intentional CPU oracle inside a successful native dispatch.
    if (state.require_native_dispatch && !smag_parity_replay_ && !native_dispatch_eligible)
    {
        throw std::runtime_error("qualified fallback-off accelerator Smagorinsky requires "
                                 "native flat-grid stability-none capability");
    }

    // GPU path (GPU audit Phase 3.4): eddy coefficients + the
    // variable-nu SGS tendencies as two dispatches. Flat terrain and
    // stability "none" only - the Richardson correction and terrain
    // metrics stay CPU. This is a tendency producer, not a prognostic-state
    // repair, so strict guard mode may use it after the common input validator
    // above. The runtime verifies the published tendency fields before any
    // state commit. Skipped during a parity replay, which must take the CPU
    // path below on the same inputs.
    if (!smag_parity_replay_ && native_dispatch_eligible)
    {
        const bool parity_sample = nimbus::parity::sample("smagorinsky_sgs");
        const bool download_coefficients =
            diag_opt != nullptr || parity_sample || state.require_host_transport_coefficients;
        if (download_coefficients)
        {
            nu_t_.resize(state.NR, state.NTH, state.NZ, 0.0f);
            K_theta_.resize(state.NR, state.NTH, state.NZ, 0.0f);
            K_q_.resize(state.NR, state.NTH, state.NZ, 0.0f);
        }

        // Vertical spacing profiles from the same grid_metric calls the
        // CPU path uses (without terrain they are (i, j)-independent).
        std::vector<float> dz_prof(static_cast<std::size_t>(state.NZ));
        std::vector<float> dzc_prof(static_cast<std::size_t>(state.NZ));
        for (int k = 0; k < state.NZ; ++k)
        {
            dz_prof[static_cast<std::size_t>(k)] =
                static_cast<float>(grid_metric::vertical_cell_thickness(grid, 0, 0, k, state.NZ));
            dzc_prof[static_cast<std::size_t>(k)] =
                static_cast<float>(grid_metric::centered_dz_span(grid, 0, 0, k, state.NZ));
        }

        const int   filter_mode = (cfg.filter_width == "dx") ? 1 : 0;
        const bool  use_h       = cfg.mode != "vertical_only";
        const bool  use_v       = cfg.mode != "horizontal_only";
        const bool  has_qv      = (state.qv != nullptr);
        const float dx_f        = static_cast<float>(grid.dx);
        const float dy_f        = static_cast<float>(grid.dy);

        if (state.defer_transport_tendency_build &&
            dispatch_smagorinsky_coefficients_backend(*state.compute_runtime,
                                                      state.u->data(),
                                                      state.v->data(),
                                                      state.w->data(),
                                                      dz_prof.data(),
                                                      dzc_prof.data(),
                                                      state.NZ,
                                                      nu_t_.data(),
                                                      K_theta_.data(),
                                                      K_q_.data(),
                                                      state.NR,
                                                      state.NTH,
                                                      state.NZ,
                                                      filter_mode,
                                                      use_h,
                                                      use_v,
                                                      periodic_x,
                                                      periodic_y,
                                                      dx_f,
                                                      dy_f,
                                                      static_cast<float>(Cs_),
                                                      static_cast<float>(Pr_t_),
                                                      static_cast<float>(Sc_t_),
                                                      static_cast<float>(nu_t_max_),
                                                      static_cast<float>(K_max_)))
        {
            nu_t_host_current_ = true;
            if (parity_sample)
            {
                const Field3D        gpu_nu  = nu_t_;
                const Field3D        gpu_kth = K_theta_;
                const Field3D        gpu_kq  = K_q_;
                TurbulenceTendencies ref;
                smag_parity_replay_ = true;
                compute(cfg, grid, state, ref, nullptr);
                smag_parity_replay_ = false;
                using nimbus::parity::Tier;
                nimbus::parity::check_field("smagorinsky_sgs",
                                            "nu_t",
                                            nu_t_.data(),
                                            gpu_nu.data(),
                                            state.NR,
                                            state.NTH,
                                            state.NZ,
                                            Tier::Stencil);
                nu_t_    = gpu_nu;
                K_theta_ = gpu_kth;
                K_q_     = gpu_kq;
            }
            publish_deferred_tendencies();
            publish_diagnostics();
            return;
        }

        // Fused chain (GPU audit Phase 4.3c): both passes in ONE
        // submission - u/v/w upload once and the eddy coefficients stay
        // on device between them. Normal production leaves them there;
        // diagnostics and the parity replay request a host copy.
        if (!state.defer_transport_tendency_build &&
            dispatch_smagorinsky_fused_backend(*state.compute_runtime,
                                               state.u->data(),
                                               state.v->data(),
                                               state.w->data(),
                                               state.theta->data(),
                                               has_qv ? state.qv->data() : state.theta->data(),
                                               dz_prof.data(),
                                               dzc_prof.data(),
                                               state.rho->data(),
                                               state.NZ,
                                               download_coefficients ? nu_t_.data() : nullptr,
                                               download_coefficients ? K_theta_.data() : nullptr,
                                               download_coefficients ? K_q_.data() : nullptr,
                                               tend.dudt_sgs.data(),
                                               tend.dvdt_sgs.data(),
                                               tend.dwdt_sgs.data(),
                                               tend.dthetadt_sgs.data(),
                                               tend.dqvdt_sgs.data(),
                                               state.NR,
                                               state.NTH,
                                               state.NZ,
                                               filter_mode,
                                               use_h,
                                               use_v,
                                               has_qv,
                                               periodic_x,
                                               periodic_y,
                                               dx_f,
                                               dy_f,
                                               static_cast<float>(Cs_),
                                               static_cast<float>(Pr_t_),
                                               static_cast<float>(Sc_t_),
                                               static_cast<float>(nu_t_max_),
                                               static_cast<float>(K_max_),
                                               download_coefficients, state.open_lateral))
        {
            nu_t_host_current_ = download_coefficients;
            publish_diagnostics();

            if (parity_sample)
            {
                // Parity replay (GPU audit Phase 1): rerun the CPU
                // path into scratch tendencies; the GPU result (including
                // the coefficient members the replay overwrites) stays
                // production.
                Field3D              gpu_nu  = nu_t_;
                Field3D              gpu_kth = K_theta_;
                Field3D              gpu_kq  = K_q_;
                TurbulenceTendencies ref;
                smag_parity_replay_ = true;
                compute(cfg, grid, state, ref, nullptr);
                smag_parity_replay_ = false;

                using nimbus::parity::Tier;
                const char* kn = "smagorinsky_sgs";
                nimbus::parity::check_field(
                    kn, "nu_t", nu_t_.data(), gpu_nu.data(), state.NR, state.NTH, state.NZ, Tier::Stencil);
                nimbus::parity::check_field(kn,
                                            "dudt_sgs",
                                            ref.dudt_sgs.data(),
                                            tend.dudt_sgs.data(),
                                            state.NR,
                                            state.NTH,
                                            state.NZ,
                                            Tier::Stencil);
                nimbus::parity::check_field(kn,
                                            "dvdt_sgs",
                                            ref.dvdt_sgs.data(),
                                            tend.dvdt_sgs.data(),
                                            state.NR,
                                            state.NTH,
                                            state.NZ,
                                            Tier::Stencil);
                nimbus::parity::check_field(kn,
                                            "dwdt_sgs",
                                            ref.dwdt_sgs.data(),
                                            tend.dwdt_sgs.data(),
                                            state.NR,
                                            state.NTH,
                                            state.NZ,
                                            Tier::Stencil);
                nimbus::parity::check_field(kn,
                                            "dthetadt_sgs",
                                            ref.dthetadt_sgs.data(),
                                            tend.dthetadt_sgs.data(),
                                            state.NR,
                                            state.NTH,
                                            state.NZ,
                                            Tier::Stencil);
                nimbus::parity::check_field(kn,
                                            "dqvdt_sgs",
                                            ref.dqvdt_sgs.data(),
                                            tend.dqvdt_sgs.data(),
                                            state.NR,
                                            state.NTH,
                                            state.NZ,
                                            Tier::Stencil);

                nu_t_    = gpu_nu;
                K_theta_ = gpu_kth;
                K_q_     = gpu_kq;
            }
            // The completed accelerator producer owns exact copies of these five
            // host-published tendency fields until a later CPU writer clears
            // this hint. The consuming backend still validates generation,
            // pointers, and extents independently.
            tend.resident_source_unchanged = true;
            return;
        }
        // Coefficients may have dispatched while tendencies did not. An
        // exploratory run may recompute the complete operator on CPU, but a
        // qualified fallback-off accelerator profile must never change hardware
        // paths silently after admission.
        if (state.require_native_dispatch)
        {
            throw std::runtime_error("qualified fallback-off accelerator Smagorinsky dispatch failed");
        }
    }

    const auto coefficients_t0 = smag_perf_start();
    compute_eddy_coefficients(cfg, grid, state);
    nu_t_host_current_ = true;
    smag_perf_finish(TurbulencePerfStage::SmagorinskyCoefficients, coefficients_t0);

    if (state.defer_transport_tendency_build)
    {
        publish_deferred_tendencies();
        publish_diagnostics();
        return;
    }

    const auto momentum_t0 = smag_perf_start();
    eddy_viscosity::compute_momentum_diffusion_tendencies(
        cfg, state, grid, nu_t_, tend.dudt_sgs, tend.dvdt_sgs, tend.dwdt_sgs, &tend.mapped_boundary_power_w);
    smag_perf_finish(TurbulencePerfStage::SmagorinskyMomentum, momentum_t0);

    const auto theta_t0 = smag_perf_start();
    eddy_viscosity::compute_scalar_diffusion_tendencies(cfg, state, grid, K_theta_, *state.theta, tend.dthetadt_sgs);
    smag_perf_finish(TurbulencePerfStage::SmagorinskyTheta, theta_t0);

    if (state.qv)
    {
        const auto moisture_t0 = smag_perf_start();
        eddy_viscosity::compute_scalar_diffusion_tendencies(cfg, state, grid, K_q_, *state.qv, tend.dqvdt_sgs);
        smag_perf_finish(TurbulencePerfStage::SmagorinskyMoisture, moisture_t0);
    }

    publish_diagnostics();
}

bool SmagorinskyScheme::mutable_realized_transport_fields(TurbulenceTransportFields& fields) noexcept
{
    if (!nu_t_host_current_ || nu_t_.empty() || K_theta_.empty()
        || K_q_.empty())
        return false;
    fields.eddy_viscosity = &nu_t_;
    fields.theta_diffusivity = &K_theta_;
    fields.moisture_diffusivity = &K_q_;
    fields.tke_diffusivity = nullptr;
    return true;
}

bool SmagorinskyScheme::rebuild_from_realized_transport(
    const TurbulenceConfig& cfg,
    const GridMetrics& grid,
    const TurbulenceStateView& state,
    TurbulenceTendencies& tend)
{
    if (!nu_t_host_current_ || nu_t_.size() != K_theta_.size()
        || nu_t_.size() != K_q_.size()
        || nu_t_.size() != static_cast<std::size_t>(state.NR) * state.NTH
            * state.NZ)
        return false;
    for (std::size_t index = 0; index < nu_t_.size(); ++index)
    {
        const double nu = static_cast<double>(nu_t_.data()[index]);
        const double kth = static_cast<double>(K_theta_.data()[index]);
        const double kq = static_cast<double>(K_q_.data()[index]);
        if (!std::isfinite(nu) || nu < 0.0
            || !std::isfinite(kth) || kth < 0.0
            || !std::isfinite(kq) || kq < 0.0)
            throw std::runtime_error(
                "stochastic Smagorinsky transport coefficients must remain "
                "finite and non-negative");
        nu_t_.data()[index] = static_cast<float>(
            std::min(nu, static_cast<double>(cfg.nu_t_max)));
        K_theta_.data()[index] = static_cast<float>(
            std::min(kth, static_cast<double>(cfg.K_max)));
        K_q_.data()[index] = static_cast<float>(
            std::min(kq, static_cast<double>(cfg.K_max)));
    }

    eddy_viscosity::compute_momentum_diffusion_tendencies(
        cfg, state, grid, nu_t_, tend.dudt_sgs, tend.dvdt_sgs,
        tend.dwdt_sgs, &tend.mapped_boundary_power_w);
    eddy_viscosity::compute_scalar_diffusion_tendencies(
        cfg, state, grid, K_theta_, *state.theta, tend.dthetadt_sgs);
    if (state.qv)
    {
        eddy_viscosity::compute_scalar_diffusion_tendencies(
            cfg, state, grid, K_q_, *state.qv, tend.dqvdt_sgs);
    }
    else
    {
        tend.dqvdt_sgs.fill(0.0f);
    }
    tend.dtkedt_sgs.fill(0.0f);
    tend.resident_source_unchanged = false;
    return true;
}

/**
 * @brief Computes the eddy coefficients.
 */
void SmagorinskyScheme::compute_eddy_coefficients(
    const TurbulenceConfig& cfg,
    const GridMetrics& grid,
    const TurbulenceStateView& state
)
{
    const bool terrain = grid_metric::has_terrain_metrics(grid);
    // Terrain stencils own only physical scalar rows. Zero the closure/ghost
    // planes once so no mapped stress can inherit stale coefficients.
    nu_t_.resize(state.NR, state.NTH, state.NZ, 0.0f);
    K_theta_.resize(state.NR, state.NTH, state.NZ, 0.0f);
    K_q_.resize(state.NR, state.NTH, state.NZ, 0.0f);
    const PhysicalCellDomain physical = physical_scalar_domain(
        state.NR, state.NTH, state.NZ,
        state.c_grid ? StaggerType::CGrid : StaggerType::Collocated);
    const int i_begin = terrain || !state.c_grid ? physical.i_begin : 0;
    const int i_end = terrain || !state.c_grid ? physical.i_end : state.NR;
    const int j_begin = terrain || !state.c_grid ? physical.j_begin : 0;
    const int j_end = terrain || !state.c_grid ? physical.j_end : state.NTH;
    const int k_begin = !state.c_grid ? physical.k_begin : 0;
    const int k_end = !state.c_grid ? physical.k_end : state.NZ;
    if (!terrain)
    {
        filter_width_profile_.resize(static_cast<std::size_t>(state.NZ));
        for (int k = 0; k < state.NZ; ++k)
        {
            filter_width_profile_[static_cast<std::size_t>(k)] =
                eddy_viscosity::compute_filter_width(cfg, grid, 0, 0, k);
        }
    }
    const bool use_richardson =
        cfg.stability_correction == "ri";

    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(state.NR) * state.NTH * state.NZ >= 32768)
    for (int i = i_begin; i < i_end; ++i)
    {
        for (int j = j_begin; j < j_end; ++j)
        {
            for (int k = k_begin; k < k_end; ++k)
            {
                // Staggered-consistent |S| at the cell center
                // (SCIENCE.md M14).
                const double strain_sq =
                    eddy_viscosity::compute_directional_strain_sq_centered(
                        state, grid, i, j, k, cfg.mode);
                const double strain_mag = cfg.stability_correction == kLillyRichardsonModel
                    ? eddy_viscosity::lilly_equilibrium_strain_rate(strain_sq,
                        eddy_viscosity::compute_buoyancy_frequency_squared(
                            state, grid, i, j, k, cfg.enable_moist_buoyancy), Pr_t_)
                    : std::sqrt(strain_sq);

                const double Delta =
#ifdef NIMBUS_SMAGORINSKY_LEGACY_FILTER_WIDTH_EXPERIMENT
                    eddy_viscosity::compute_filter_width(
                        cfg, grid, i, j, k);
#else
                    terrain
                    ? eddy_viscosity::compute_filter_width(
                        cfg, grid, i, j, k)
                    : filter_width_profile_[static_cast<std::size_t>(k)];
#endif

                const double stability_factor = use_richardson
                    ? apply_stability_correction(
                        cfg, grid, state, i, j, k)
                    : 1.0;

                double nu_t = eddy_viscosity::compute_smagorinsky_viscosity(
                    Cs_, Delta, strain_mag, stability_factor
                );

                nu_t = std::min(nu_t, nu_t_max_);

                nu_t_[i][j][k] = static_cast<float>(nu_t);

                double K_theta, K_q, K_tke;
                eddy_viscosity::compute_eddy_diffusivities(nu_t, Pr_t_, Sc_t_, K_theta, K_q, K_tke);

                K_theta = std::min(K_theta, K_max_);
                K_q = std::min(K_q, K_max_);

                K_theta_[i][j][k] = static_cast<float>(K_theta);
                K_q_[i][j][k] = static_cast<float>(K_q);
            }
        }
    }
}

void SmagorinskyScheme::compute_dynamic(
    const TurbulenceConfig& cfg,
    const GridMetrics& grid,
    const TurbulenceStateView& state,
    TurbulenceTendencies& tendencies,
    TurbulenceDiagnostics* diagnostics)
{
    Field3D coefficient;
    dynamic_smagorinsky::compute_coefficient(cfg, grid, state, coefficient);

    Field3D viscosity(state.NR, state.NTH, state.NZ, 0.0f);
    Field3D theta_diffusivity(state.NR, state.NTH, state.NZ, 0.0f);
    Field3D moisture_diffusivity(state.NR, state.NTH, state.NZ, 0.0f);
    const PhysicalCellDomain domain = physical_scalar_domain(
        state.NR, state.NTH, state.NZ, StaggerType::CGrid);
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double strain = std::sqrt(
                    eddy_viscosity::compute_directional_strain_sq_centered(
                        state, grid, i, j, k, cfg.mode));
                const double delta = eddy_viscosity::compute_filter_width(
                    cfg, grid, i, j, k);
                const double nu = std::min(
                    eddy_viscosity::compute_smagorinsky_viscosity(
                        coefficient(i, j, k), delta, strain),
                    static_cast<double>(cfg.nu_t_max));
                double k_theta = 0.0;
                double k_q = 0.0;
                double k_tke = 0.0;
                eddy_viscosity::compute_eddy_diffusivities(
                    nu, cfg.Pr_t, cfg.Sc_t, k_theta, k_q, k_tke);
                viscosity(i, j, k) = static_cast<float>(nu);
                theta_diffusivity(i, j, k) = static_cast<float>(
                    std::min(k_theta, static_cast<double>(cfg.K_max)));
                moisture_diffusivity(i, j, k) = static_cast<float>(
                    std::min(k_q, static_cast<double>(cfg.K_max)));
            }

    TurbulenceTendencies candidate;
    candidate.dudt_sgs.resize(state.NR, state.NTH, state.NZ, 0.0f);
    candidate.dvdt_sgs.resize(state.NR, state.NTH, state.NZ, 0.0f);
    candidate.dwdt_sgs.resize(state.NR, state.NTH, state.NZ, 0.0f);
    candidate.dthetadt_sgs.resize(state.NR, state.NTH, state.NZ, 0.0f);
    candidate.dqvdt_sgs.resize(state.NR, state.NTH, state.NZ, 0.0f);
    candidate.dtkedt_sgs.resize(state.NR, state.NTH, state.NZ, 0.0f);
    candidate.mapped_boundary_power_w =
        std::numeric_limits<double>::quiet_NaN();
    candidate.resident_source_unchanged = false;
    if (!state.defer_transport_tendency_build)
    {
        eddy_viscosity::compute_momentum_diffusion_tendencies(
            cfg, state, grid, viscosity, candidate.dudt_sgs,
            candidate.dvdt_sgs, candidate.dwdt_sgs,
            &candidate.mapped_boundary_power_w);
        eddy_viscosity::compute_scalar_diffusion_tendencies(
            cfg, state, grid, theta_diffusivity, *state.theta,
            candidate.dthetadt_sgs);
        if (state.qv != nullptr)
            eddy_viscosity::compute_scalar_diffusion_tendencies(
                cfg, state, grid, moisture_diffusivity, *state.qv,
                candidate.dqvdt_sgs);
    }

    TurbulenceDiagnostics diagnostic_candidate;
    if (diagnostics != nullptr)
    {
        diagnostic_candidate.nu_t = viscosity;
        diagnostic_candidate.K_theta = theta_diffusivity;
        diagnostic_candidate.K_q = moisture_diffusivity;
        diagnostic_candidate.K_tke.resize(
            state.NR, state.NTH, state.NZ, 0.0f);
        diagnostic_candidate.shear_prod.resize(
            state.NR, state.NTH, state.NZ, 0.0f);
        diagnostic_candidate.buoy_prod.resize(
            state.NR, state.NTH, state.NZ, 0.0f);
        diagnostic_candidate.dissipation.resize(
            state.NR, state.NTH, state.NZ, 0.0f);
        diagnostic_candidate.Cs_eff = coefficient;
        diagnostic_candidate.Pr_t_eff.resize(
            state.NR, state.NTH, state.NZ,
            static_cast<float>(cfg.Pr_t));
    }

    nu_t_ = std::move(viscosity);
    K_theta_ = std::move(theta_diffusivity);
    K_q_ = std::move(moisture_diffusivity);
    nu_t_host_current_ = true;
    tendencies = std::move(candidate);
    if (diagnostics != nullptr)
        *diagnostics = std::move(diagnostic_candidate);
}

/**
 * @brief Applies the stability correction.
 */
double SmagorinskyScheme::apply_stability_correction(
    const TurbulenceConfig& cfg,
    const GridMetrics& grid,
    const TurbulenceStateView& state,
    int i, int j, int k
)
{
    if (cfg.stability_correction == "ri")
    {
        return eddy_viscosity::cgrid_stability_factor_ri(
            state, grid, i, j, k, cfg.enable_moist_buoyancy);
    }
    else
    {
        return 1.0;
    }
}

} // namespace tmv
