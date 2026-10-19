#include "turbulence/base/collocated_eddy_viscosity.hpp"
#include "turbulence/base/native_collocated_eddy_viscosity.hpp"
#include "tke.hpp"
#include "compute/backend_runtime.hpp"

#include "core/field/field_pool.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "util/grid_metric_utils.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tmv
{
namespace
{

PhysicalCellDomain work_domain(const TurbulenceStateView& state)
{
    return physical_scalar_domain(
        state.NR, state.NTH, state.NZ,
        state.c_grid ? StaggerType::CGrid : StaggerType::Collocated);
}

void shape_and_zero(Field3D& field, const TurbulenceStateView& state)
{
    field.resize(state.NR, state.NTH, state.NZ, 0.0f);
}

bool has_shape(const Field3D& field, const TurbulenceStateView& state)
{
    return field.size_r() == state.NR
        && field.size_th() == state.NTH
        && field.size_z() == state.NZ;
}

} // namespace

std::unique_ptr<TurbulenceSchemeBase> TKEScheme::clone() const
{
    return std::make_unique<TKEScheme>(*this);
}

int TKEScheme::required_fields() const
{
    return static_cast<int>(TurbulenceRequirements::BASIC)
         | static_cast<int>(TurbulenceRequirements::TKE);
}

void TKEScheme::initialize(const TurbulenceConfig& cfg)
{
    if (!std::isfinite(cfg.Pr_t) || cfg.Pr_t <= 0.0
        || !std::isfinite(cfg.Sc_t) || cfg.Sc_t <= 0.0)
    {
        throw std::invalid_argument(
            "TKE scalar transport ratios must be positive and finite");
    }

    tmv::log_info("Initialized Deardorff-Moeng prognostic TKE closure");
    tmv::log_info(
        "  Stability correction = ", cfg.stability_correction);
}

void TKEScheme::compute(const TurbulenceConfig& cfg,
                        const GridMetrics& grid,
                        const TurbulenceStateView& state,
                        TurbulenceTendencies& tendencies,
                        TurbulenceDiagnostics* diagnostics)
{
    eddy_viscosity::validate_scheme_inputs(cfg, grid, state, true);

    if (compute_native(cfg, grid, state, tendencies, diagnostics)) return;
    if (state.require_native_dispatch || (state.compute_runtime != nullptr &&
        state.compute_runtime->active_kind != ComputeBackendKind::Cpu &&
        !state.compute_runtime->config.allow_fallback))
        throw std::runtime_error("fallback-off TKE requires native flat Cartesian closure and transport dispatches");

    Field3D eddy_viscosity;
    Field3D theta_diffusivity;
    Field3D moisture_diffusivity;
    Field3D tke_diffusivity;
    TurbulenceTendencies candidate;
    TurbulenceDiagnostics diagnostic_candidate;
    Field3D* effective_prandtl = diagnostics != nullptr
        ? &diagnostic_candidate.Pr_t_eff : nullptr;

    build_coefficients(cfg, grid, state, *state.tke,
                       eddy_viscosity, theta_diffusivity,
                       moisture_diffusivity, tke_diffusivity,
                       effective_prandtl);

    shape_and_zero(candidate.dudt_sgs, state);
    shape_and_zero(candidate.dvdt_sgs, state);
    shape_and_zero(candidate.dwdt_sgs, state);
    shape_and_zero(candidate.dthetadt_sgs, state);
    shape_and_zero(candidate.dqvdt_sgs, state);
    shape_and_zero(candidate.dtkedt_sgs, state);
    candidate.mapped_boundary_power_w =
        std::numeric_limits<double>::quiet_NaN();
    candidate.resident_source_unchanged = false;

    if (!state.defer_transport_tendency_build)
    {
        eddy_viscosity::compute_momentum_diffusion_tendencies(
            cfg, state, grid, eddy_viscosity,
            candidate.dudt_sgs, candidate.dvdt_sgs,
            candidate.dwdt_sgs, &candidate.mapped_boundary_power_w);
        eddy_viscosity::compute_scalar_diffusion_tendencies(
            cfg, state, grid, theta_diffusivity, *state.theta,
            candidate.dthetadt_sgs);
        if (state.qv != nullptr)
        {
            eddy_viscosity::compute_scalar_diffusion_tendencies(
                cfg, state, grid, moisture_diffusivity, *state.qv,
                candidate.dqvdt_sgs);
        }
        build_tke_tendency(cfg, grid, state, *state.tke,
                           eddy_viscosity, theta_diffusivity,
                           tke_diffusivity,
                           candidate.dtkedt_sgs);
    }

    if (diagnostics != nullptr)
    {
        diagnostic_candidate.nu_t = eddy_viscosity;
        diagnostic_candidate.K_theta = theta_diffusivity;
        diagnostic_candidate.K_q = moisture_diffusivity;
        diagnostic_candidate.K_tke = tke_diffusivity;
        shape_and_zero(diagnostic_candidate.shear_prod, state);
        shape_and_zero(diagnostic_candidate.buoy_prod, state);
        shape_and_zero(diagnostic_candidate.dissipation, state);
        shape_and_zero(diagnostic_candidate.Cs_eff, state);

        build_production(cfg, state, grid, eddy_viscosity,
                         theta_diffusivity,
                         diagnostic_candidate.shear_prod,
                         diagnostic_candidate.buoy_prod);
        const PhysicalCellDomain domain = work_domain(state);
#pragma omp parallel for collapse(2) if(static_cast<std::size_t>(domain.nx()) * domain.ny() * domain.nz() >= 32768)
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                    diagnostic_candidate.dissipation(i, j, k) =
                        static_cast<float>(dissipation(
                            cfg, grid, state,
                            static_cast<double>((*state.tke)(i, j, k)),
                            i, j, k));
    }

    eddy_viscosity_ = std::move(eddy_viscosity);
    theta_diffusivity_ = std::move(theta_diffusivity);
    moisture_diffusivity_ = std::move(moisture_diffusivity);
    tke_diffusivity_ = std::move(tke_diffusivity);
    tendencies = std::move(candidate);
    if (diagnostics != nullptr)
        *diagnostics = std::move(diagnostic_candidate);
}

bool TKEScheme::compute_native(const TurbulenceConfig&    cfg,
                               const GridMetrics&         grid,
                               const TurbulenceStateView& state,
                               TurbulenceTendencies&      tendencies,
                               TurbulenceDiagnostics*     diagnostics)
{
    if (!state.c_grid)
    {
        collocated_eddy_viscosity::NativeResult result;
        if (!collocated_eddy_viscosity::compute_native(cfg, grid, state, true, result))
            return false;
        TurbulenceDiagnostics next_diagnostics;
        if (diagnostics)
        {
            next_diagnostics.nu_t        = result.closure[0];
            next_diagnostics.K_theta     = result.closure[1];
            next_diagnostics.K_q         = result.closure[2];
            next_diagnostics.K_tke       = result.closure[3];
            next_diagnostics.shear_prod  = std::move(result.tendency[6]);
            next_diagnostics.buoy_prod   = std::move(result.closure[4]);
            next_diagnostics.dissipation = std::move(result.closure[5]);
            next_diagnostics.Pr_t_eff    = std::move(result.closure[6]);
            shape_and_zero(next_diagnostics.Cs_eff, state);
        }
        eddy_viscosity_       = std::move(result.closure[0]);
        theta_diffusivity_    = std::move(result.closure[1]);
        moisture_diffusivity_ = std::move(result.closure[2]);
        tke_diffusivity_      = std::move(result.closure[3]);
        collocated_eddy_viscosity::publish_tendencies(result, tendencies);
        if (diagnostics)
            *diagnostics = std::move(next_diagnostics);
        return true;
    }
    auto* backend = state.compute_runtime != nullptr ? state.compute_runtime->backend.get() : nullptr;
    auto* kernels = backend != nullptr ? backend->turbulence_kernels() : nullptr;
    if (!state.c_grid || grid_metric::has_terrain_metrics(grid) || cfg.filter_width == "user" ||
        state.defer_transport_tendency_build || kernels == nullptr || !kernels->supports(TurbulenceKernelOp::Tke))
        return false;

    std::array<Field3D, 8> closure;
    for (auto& field : closure)
        shape_and_zero(field, state);
    std::vector<float> dz(static_cast<std::size_t>(state.NZ));
    for (int k = 0; k < state.NZ; ++k)
        dz[k] = static_cast<float>(grid_metric::vertical_cell_thickness(grid, 0, 0, k, state.NZ));
    Field3D zero;
    if (state.qv == nullptr)
        shape_and_zero(zero, state);
    const float*       vapor = state.qv != nullptr ? state.qv->data() : zero.data();
    TkeClosureDispatch d;
    d.u                 = state.u->data();
    d.v                 = state.v->data();
    d.w                 = state.w->data();
    d.theta             = state.theta->data();
    d.qv                = vapor;
    d.tke               = state.tke->data();
    d.dz                = dz.data();
    d.nu                = closure[0].data();
    d.K_theta           = closure[1].data();
    d.K_q               = closure[2].data();
    d.K_tke             = closure[3].data();
    d.shear             = closure[4].data();
    d.buoyancy          = closure[5].data();
    d.dissipation       = closure[6].data();
    d.effective_prandtl = closure[7].data();
    d.nx                = state.NR;
    d.ny                = state.NTH;
    d.nz                = state.NZ;
    d.profile_length    = state.NZ;
    d.filter_mode       = cfg.filter_width == "dx" ? 1 : 0;
    d.use_horizontal    = cfg.mode != "vertical_only";
    d.use_vertical      = cfg.mode != "horizontal_only";
    d.periodic_x        = eddy_viscosity::periodic_x_topology(state);
    d.periodic_y        = eddy_viscosity::periodic_y_topology(state);
    d.stability         = cfg.stability_correction == "tke_based";
    d.moist_buoyancy    = cfg.enable_moist_buoyancy;
    d.dx                = static_cast<float>(grid.dx);
    d.dy                = static_cast<float>(grid.dy);
    d.Pr_t              = static_cast<float>(cfg.Pr_t);
    d.Sc_t              = static_cast<float>(cfg.Sc_t);
    d.nu_max            = static_cast<float>(cfg.nu_t_max);
    d.K_max             = static_cast<float>(cfg.K_max);
    if (!kernels->dispatch_tke_closure(d))
        return false;

    TurbulenceTendencies candidate;
    for (Field3D* field : {&candidate.dudt_sgs,
                           &candidate.dvdt_sgs,
                           &candidate.dwdt_sgs,
                           &candidate.dthetadt_sgs,
                           &candidate.dqvdt_sgs,
                           &candidate.dtkedt_sgs})
        shape_and_zero(*field, state);
    SmagorinskyTendenciesDispatch transport;
    transport.u              = d.u;
    transport.v              = d.v;
    transport.w              = d.w;
    transport.theta_data     = d.theta;
    transport.qv_data        = vapor;
    transport.nu_t           = d.nu;
    transport.K_theta        = d.K_theta;
    transport.K_q            = d.K_q;
    transport.dz_prof        = d.dz;
    transport.rho            = state.rho->data();
    transport.prof_len       = d.nz;
    transport.dudt_sgs       = candidate.dudt_sgs.data();
    transport.dvdt_sgs       = candidate.dvdt_sgs.data();
    transport.dwdt_sgs       = candidate.dwdt_sgs.data();
    transport.dthetadt_sgs   = candidate.dthetadt_sgs.data();
    transport.dqvdt_sgs      = candidate.dqvdt_sgs.data();
    transport.nx             = d.nx;
    transport.ny             = d.ny;
    transport.nz             = d.nz;
    transport.use_horizontal = d.use_horizontal;
    transport.use_vertical   = d.use_vertical;
    transport.has_qv         = state.qv != nullptr;
    transport.periodic_x     = d.periodic_x;
    transport.periodic_y     = d.periodic_y;
    transport.dx             = d.dx;
    transport.dy             = d.dy;
    transport.open_lateral   = state.open_lateral;
    if (!kernels->dispatch_smagorinsky_tendencies(transport))
        return false;
    TkeTendencyDispatch energy;
    energy.rho            = state.rho->data();
    energy.tke            = d.tke;
    energy.K_tke          = d.K_tke;
    energy.shear          = d.shear;
    energy.buoyancy       = d.buoyancy;
    energy.dissipation    = d.dissipation;
    energy.dz             = d.dz;
    energy.tendency       = candidate.dtkedt_sgs.data();
    energy.nx             = d.nx;
    energy.ny             = d.ny;
    energy.nz             = d.nz;
    energy.profile_length = d.nz;
    energy.use_horizontal = d.use_horizontal;
    energy.use_vertical   = d.use_vertical;
    energy.periodic_x     = d.periodic_x;
    energy.periodic_y     = d.periodic_y;
    energy.dx             = d.dx;
    energy.dy             = d.dy;
    energy.open_lateral   = state.open_lateral;
    if (!kernels->dispatch_tke_tendency(energy))
        return false;
    for (const Field3D* field : {&candidate.dudt_sgs,
                                 &candidate.dvdt_sgs,
                                 &candidate.dwdt_sgs,
                                 &candidate.dthetadt_sgs,
                                 &candidate.dqvdt_sgs,
                                 &candidate.dtkedt_sgs})
        for (std::size_t n = 0; n < field->size(); ++n)
            if (!std::isfinite(field->data()[n]))
                return false;
    TurbulenceDiagnostics next_diagnostics;
    if (diagnostics != nullptr)
    {
        next_diagnostics.nu_t        = closure[0];
        next_diagnostics.K_theta     = closure[1];
        next_diagnostics.K_q         = closure[2];
        next_diagnostics.K_tke       = closure[3];
        next_diagnostics.shear_prod  = std::move(closure[4]);
        next_diagnostics.buoy_prod   = std::move(closure[5]);
        next_diagnostics.dissipation = std::move(closure[6]);
        next_diagnostics.Pr_t_eff    = std::move(closure[7]);
        shape_and_zero(next_diagnostics.Cs_eff, state);
    }
    eddy_viscosity_       = std::move(closure[0]);
    theta_diffusivity_    = std::move(closure[1]);
    moisture_diffusivity_ = std::move(closure[2]);
    tke_diffusivity_      = std::move(closure[3]);
    tendencies            = std::move(candidate);
    if (diagnostics != nullptr)
        *diagnostics = std::move(next_diagnostics);
    return true;
}

const Field3D* TKEScheme::realized_eddy_viscosity() const noexcept
{
    return eddy_viscosity_.empty() ? nullptr : &eddy_viscosity_;
}

bool TKEScheme::mutable_realized_transport_fields(
    TurbulenceTransportFields& fields) noexcept
{
    if (eddy_viscosity_.empty() || theta_diffusivity_.empty()
        || moisture_diffusivity_.empty() || tke_diffusivity_.empty())
    {
        return false;
    }
    fields.eddy_viscosity = &eddy_viscosity_;
    fields.theta_diffusivity = &theta_diffusivity_;
    fields.moisture_diffusivity = &moisture_diffusivity_;
    fields.tke_diffusivity = &tke_diffusivity_;
    return true;
}

bool TKEScheme::rebuild_from_realized_transport(
    const TurbulenceConfig& cfg,
    const GridMetrics& grid,
    const TurbulenceStateView& state,
    TurbulenceTendencies& tendencies)
{
    eddy_viscosity::validate_scheme_inputs(cfg, grid, state, true);
    if (!has_shape(eddy_viscosity_, state)
        || !has_shape(theta_diffusivity_, state)
        || !has_shape(moisture_diffusivity_, state)
        || !has_shape(tke_diffusivity_, state))
    {
        return false;
    }

    Field3D eddy_viscosity = eddy_viscosity_;
    Field3D theta_diffusivity = theta_diffusivity_;
    Field3D moisture_diffusivity = moisture_diffusivity_;
    Field3D tke_diffusivity = tke_diffusivity_;
    Field3D* fields[] = {
        &eddy_viscosity, &theta_diffusivity,
        &moisture_diffusivity, &tke_diffusivity};
    for (std::size_t field_index = 0; field_index < 4; ++field_index)
    {
        const double limit = field_index == 0
            ? static_cast<double>(cfg.nu_t_max)
            : static_cast<double>(cfg.K_max);
        for (std::size_t index = 0; index < fields[field_index]->size(); ++index)
        {
            const double value = fields[field_index]->data()[index];
            if (!std::isfinite(value) || value < 0.0)
            {
                throw std::runtime_error(
                    "stochastic TKE transport coefficients must remain "
                    "finite and non-negative");
            }
            fields[field_index]->data()[index] =
                static_cast<float>(std::min(value, limit));
        }
    }

    TurbulenceTendencies candidate;
    shape_and_zero(candidate.dudt_sgs, state);
    shape_and_zero(candidate.dvdt_sgs, state);
    shape_and_zero(candidate.dwdt_sgs, state);
    shape_and_zero(candidate.dthetadt_sgs, state);
    shape_and_zero(candidate.dqvdt_sgs, state);
    shape_and_zero(candidate.dtkedt_sgs, state);
    candidate.resident_source_unchanged = false;

    eddy_viscosity::compute_momentum_diffusion_tendencies(
        cfg, state, grid, eddy_viscosity,
        candidate.dudt_sgs, candidate.dvdt_sgs,
        candidate.dwdt_sgs, &candidate.mapped_boundary_power_w);
    eddy_viscosity::compute_scalar_diffusion_tendencies(
        cfg, state, grid, theta_diffusivity, *state.theta,
        candidate.dthetadt_sgs);
    if (state.qv != nullptr)
    {
        eddy_viscosity::compute_scalar_diffusion_tendencies(
            cfg, state, grid, moisture_diffusivity, *state.qv,
            candidate.dqvdt_sgs);
    }
    build_tke_tendency(cfg, grid, state, *state.tke,
                       eddy_viscosity, theta_diffusivity,
                       tke_diffusivity,
                       candidate.dtkedt_sgs);

    eddy_viscosity_ = std::move(eddy_viscosity);
    theta_diffusivity_ = std::move(theta_diffusivity);
    moisture_diffusivity_ = std::move(moisture_diffusivity);
    tke_diffusivity_ = std::move(tke_diffusivity);
    tendencies = std::move(candidate);
    return true;
}

double TKEScheme::mixing_length(const TurbulenceConfig& cfg,
                                const GridMetrics& grid,
                                double tke,
                                double buoyancy_frequency,
                                int i, int j, int k) const
{
    const double grid_length =
        eddy_viscosity::compute_filter_width(cfg, grid, i, j, k);
    if (cfg.stability_correction == "none")
        return grid_length;
    if (tke == 0.0 && buoyancy_frequency > 0.0)
        return 0.0;
    const double stable_length = buoyancy_frequency > 0.0
        ? stable_length_coefficient_ * std::sqrt(tke) / buoyancy_frequency
        : grid_length;
    return std::min(grid_length, stable_length);
}

double TKEScheme::dissipation(const TurbulenceConfig& cfg,
                              const GridMetrics& grid,
                              const TurbulenceStateView& state,
                              double tke,
                              int i, int j, int k) const
{
    if (tke == 0.0)
        return 0.0;
    const double buoyancy_frequency =
        eddy_viscosity::compute_brunt_vaisala_frequency(
            state, grid, i, j, k, cfg.enable_moist_buoyancy);
    const double grid_length =
        eddy_viscosity::compute_filter_width(cfg, grid, i, j, k);
    const double length = mixing_length(
        cfg, grid, tke, buoyancy_frequency, i, j, k);

    // Deardorff (1980), equations 10-11. The 2*K_m TKE transport below
    // follows the same prognostic closure rather than scalar diffusivity.
    const double coefficient = dissipation_intercept_
        + dissipation_slope_ * length / grid_length;
    return coefficient * std::pow(tke, 1.5) / length;
}

void TKEScheme::build_coefficients(
    const TurbulenceConfig& cfg,
    const GridMetrics& grid,
    const TurbulenceStateView& state,
    const Field3D& tke,
    Field3D& eddy_viscosity,
    Field3D& theta_diffusivity,
    Field3D& moisture_diffusivity,
    Field3D& tke_diffusivity,
    Field3D* effective_prandtl) const
{
    shape_and_zero(eddy_viscosity, state);
    shape_and_zero(theta_diffusivity, state);
    shape_and_zero(moisture_diffusivity, state);
    shape_and_zero(tke_diffusivity, state);
    if (effective_prandtl != nullptr)
        shape_and_zero(*effective_prandtl, state);

    const PhysicalCellDomain domain = work_domain(state);
#pragma omp parallel for collapse(3) if(static_cast<std::size_t>(domain.nx()) * domain.ny() * domain.nz() >= 32768)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double energy = static_cast<double>(tke(i, j, k));
                const double buoyancy_frequency =
                    eddy_viscosity::compute_brunt_vaisala_frequency(
                        state, grid, i, j, k, cfg.enable_moist_buoyancy);
                const double grid_length =
                    eddy_viscosity::compute_filter_width(cfg, grid, i, j, k);
                const double length = mixing_length(
                    cfg, grid, energy, buoyancy_frequency, i, j, k);
                const double viscosity = std::min(
                    momentum_coefficient_ * length * std::sqrt(energy),
                    static_cast<double>(cfg.nu_t_max));
                const double length_ratio = length / grid_length;
                const double heat_ratio =
                    1.0 + (1.0 / cfg.Pr_t - 1.0) * length_ratio;
                const double moisture_ratio =
                    1.0 + (1.0 / cfg.Sc_t - 1.0) * length_ratio;

                eddy_viscosity(i, j, k) = static_cast<float>(viscosity);
                theta_diffusivity(i, j, k) = static_cast<float>(std::min(
                    viscosity * heat_ratio, static_cast<double>(cfg.K_max)));
                moisture_diffusivity(i, j, k) = static_cast<float>(std::min(
                    viscosity * moisture_ratio, static_cast<double>(cfg.K_max)));
                tke_diffusivity(i, j, k) = static_cast<float>(std::min(
                    2.0 * viscosity, static_cast<double>(cfg.K_max)));
                if (effective_prandtl != nullptr)
                {
                    (*effective_prandtl)(i, j, k) = static_cast<float>(
                        heat_ratio > 0.0 ? 1.0 / heat_ratio : 0.0);
                }
            }
}

void TKEScheme::build_tke_tendency(
    const TurbulenceConfig& cfg,
    const GridMetrics& grid,
    const TurbulenceStateView& state,
    const Field3D& tke,
    const Field3D& eddy_viscosity,
    const Field3D& theta_diffusivity,
    const Field3D& tke_diffusivity,
    Field3D& tendency) const
{
    auto& pool = FieldPool::instance();
    auto shear_guard = pool.scoped_acquire_for_overwrite(
        state.NR, state.NTH, state.NZ);
    auto buoyancy_guard = pool.scoped_acquire_for_overwrite(
        state.NR, state.NTH, state.NZ);
    Field3D& shear = shear_guard.field;
    Field3D& buoyancy = buoyancy_guard.field;
    build_production(cfg, state, grid, eddy_viscosity,
                     theta_diffusivity, shear, buoyancy);

    eddy_viscosity::compute_scalar_diffusion_tendencies(
        cfg, state, grid, tke_diffusivity, tke, tendency);
    const PhysicalCellDomain domain = work_domain(state);
#pragma omp parallel for collapse(2) if(static_cast<std::size_t>(domain.nx()) * domain.ny() * domain.nz() >= 32768)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double sink = dissipation(
                    cfg, grid, state, static_cast<double>(tke(i, j, k)),
                    i, j, k);
                tendency(i, j, k) = static_cast<float>(
                    static_cast<double>(tendency(i, j, k))
                    + static_cast<double>(shear(i, j, k))
                    + static_cast<double>(buoyancy(i, j, k)) - sink);
            }
}

void TKEScheme::build_production(
    const TurbulenceConfig& cfg,
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    const Field3D& eddy_viscosity,
    const Field3D& theta_diffusivity,
    Field3D& shear,
    Field3D& buoyancy) const
{
    shape_and_zero(shear, state);
    shape_and_zero(buoyancy, state);
    const PhysicalCellDomain domain = work_domain(state);
#pragma omp parallel for collapse(2) if(static_cast<std::size_t>(domain.nx()) * domain.ny() * domain.nz() >= 32768)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double viscosity = eddy_viscosity(i, j, k);
                shear(i, j, k) = static_cast<float>(
                    !state.c_grid && !grid.terrain_metrics_active
                    ? collocated_eddy_viscosity::flat_shear_transfer(
                        state, grid, eddy_viscosity, i, j, k, cfg.mode)
                    : viscosity * eddy_viscosity::compute_directional_strain_sq_centered(
                        state, grid, i, j, k, cfg.mode));
                if (cfg.mode != "horizontal_only")
                    buoyancy(i, j, k) = static_cast<float>(
                        -static_cast<double>(theta_diffusivity(i, j, k))
                        * eddy_viscosity::compute_buoyancy_frequency_squared(
                            state, grid, i, j, k, cfg.enable_moist_buoyancy));
            }
}

} // namespace tmv
