#include "implicit.hpp"
#include "../explicit/explicit.hpp"
#include "compute/compute_backend.hpp"
#include "compute/implicit_diffusion_solve.hpp"
#include "terrain/conservative_transport.hpp"
#include "util/grid_metric_utils.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace
{
inline bool matches_shape(const Field3D* field, int nr, int nth, int nz)
{
    return field != nullptr && field->size_r() == nr &&
           field->size_th() == nth && field->size_z() == nz;
}

inline void ensure_tendency_shape(Field3D& field, int nr, int nth, int nz)
{
    if (field.size_r() != nr || field.size_th() != nth ||
        field.size_z() != nz)
        field.resize(nr, nth, nz, 0.0f);
    else
        field.fill(0.0f);
}

inline void validate_config(const DiffusionConfig& cfg)
{
    if (cfg.operator_order != 2 || cfg.operator_type != "laplacian")
        throw std::invalid_argument(
            "implicit diffusion implements only the second-order laplacian");
    if (cfg.apply_to != "scalars" &&
        cfg.apply_to != "scalars_and_momentum" &&
        cfg.apply_to != "momentum" && cfg.apply_to != "all")
        throw std::invalid_argument(
            "implicit diffusion apply_to has an unknown target");
    if (cfg.implicit_dim != "none" && cfg.implicit_dim != "vertical" &&
        cfg.implicit_dim != "z" && cfg.implicit_dim != "all")
        throw std::invalid_argument(
            "implicit diffusion implicit_dim must be none, vertical, z, or all");
    if (!std::isfinite(cfg.K_h) || cfg.K_h < 0.0)
        throw std::invalid_argument(
            "implicit diffusion requires non-negative finite K_h");
    if (!std::isfinite(cfg.K_v) || cfg.K_v < 0.0 ||
        !std::isfinite(cfg.dt_diffusion) || cfg.dt_diffusion <= 0.0)
        throw std::invalid_argument(
            "implicit diffusion requires non-negative finite K_v and positive finite dt_diffusion");
}

struct VerticalDomain
{
    int i_begin;
    int i_end;
    int j_begin;
    int j_end;
    int k_begin;
    int k_end;
};

inline VerticalDomain scalar_domain(int nr, int nth, int nz, bool c_grid)
{
    return {1, nr - 1, 1, nth - 1,
            c_grid ? 0 : 1, c_grid ? nz : nz - 1};
}

inline VerticalDomain component_domain(
    int nr, int nth, int nz, bool c_grid, bool periodic_x,
    bool periodic_y, tmv::ImplicitVerticalComponent component)
{
    if (!c_grid || component == tmv::ImplicitVerticalComponent::Scalar)
        return scalar_domain(nr, nth, nz, c_grid);
    if (component == tmv::ImplicitVerticalComponent::U)
        return {1, periodic_x ? nr - 1 : nr - 2,
                1, nth - 1, 0, nz - 1};
    if (component == tmv::ImplicitVerticalComponent::V)
        return {1, nr - 1,
                1, periodic_y ? nth - 1 : nth - 2, 0, nz - 1};
    // w[nz-1] is the boundary-owned lid face.
    return {1, nr - 1, 1, nth - 1, 0, nz - 1};
}

inline double point_distance(
    const GridMetrics& grid, int lower_k, bool c_grid_w)
{
    if (lower_k < 0 || lower_k + 1 >= static_cast<int>(grid.dz.size()))
        throw std::out_of_range(
            "implicit vertical diffusion point distance is outside the grid");
    if (c_grid_w)
        return grid.dz[static_cast<std::size_t>(lower_k + 1)];
    return 0.5 * (grid.dz[static_cast<std::size_t>(lower_k)] +
                  grid.dz[static_cast<std::size_t>(lower_k + 1)]);
}

inline double control_thickness(
    const GridMetrics& grid, int k, bool c_grid_w)
{
    if (k < 0 || k >= static_cast<int>(grid.dz.size()))
        throw std::out_of_range(
            "implicit vertical diffusion control volume is outside the grid");
    if (!c_grid_w)
        return grid.dz[static_cast<std::size_t>(k)];
    if (k + 1 >= static_cast<int>(grid.dz.size()))
        throw std::out_of_range(
            "implicit C-grid w control volume reaches beyond the lid");
    return 0.5 * (grid.dz[static_cast<std::size_t>(k)] +
                  grid.dz[static_cast<std::size_t>(k + 1)]);
}

inline const char* component_label(tmv::ImplicitVerticalComponent component)
{
    switch (component)
    {
        case tmv::ImplicitVerticalComponent::Scalar: return "scalar";
        case tmv::ImplicitVerticalComponent::U: return "u-momentum";
        case tmv::ImplicitVerticalComponent::V: return "v-momentum";
        case tmv::ImplicitVerticalComponent::W: return "w-momentum";
    }
    return "unknown";
}
}

namespace tmv
{

void ImplicitDiffusionScheme::initialize()
{
    initialize(DiffusionConfig{});
}

void ImplicitDiffusionScheme::initialize(const DiffusionConfig& cfg)
{
    validate_config(cfg);
    tmv::log_info("Initialized conservative backward-Euler diffusion scheme");
}

void ImplicitDiffusionScheme::compute_diffusion_tendencies(
    const DiffusionConfig& cfg,
    const DiffusionStateView& state,
    DiffusionTendencies& tendencies,
    DiffusionDiagnostics* diag_opt)
{
    validate_config(cfg);
    if (!state.grid)
        throw std::runtime_error("Invalid grid for implicit diffusion");

    const Field3D* ref = state.theta ? state.theta :
        (state.u ? state.u : state.qv);
    if (ref == nullptr || ref->empty())
        throw std::invalid_argument(
            "implicit diffusion requires at least one non-empty requested field");
    const int nr = ref->size_r();
    const int nth = ref->size_th();
    const int nz = ref->size_z();
    grid_metric::validate_vertical_metrics(*state.grid, nr, nth, nz);
    if (nr < 3 || nth < 3)
        throw std::invalid_argument(
            "implicit diffusion requires lateral closure storage");

    const bool active = cfg.K_h > 0.0 || cfg.K_v > 0.0 || cfg.use_variable_K;
    const bool native_requested = active && state.compute_runtime != nullptr &&
                                  active_compute_backend_kind(*state.compute_runtime) != ComputeBackendKind::Cpu;
    if (native_requested && state.grid->terrain_metrics_active)
        throw std::runtime_error("native implicit diffusion requires flat geometry; CPU substitution is disabled");
    const bool terrain_active = active && state.grid->terrain_metrics_active;
    if (terrain_active && state.grid->terrain_metrics == nullptr)
        throw std::invalid_argument(
            "implicit terrain diffusion requires bound conservative geometry");
    if (active && !matches_shape(state.rho, nr, nth, nz))
        throw std::invalid_argument(
            "active implicit diffusion requires a matching live-density field");

    for (const Field3D* field :
         {state.theta, state.qv, state.u, state.v, state.w})
        if (field != nullptr && !matches_shape(field, nr, nth, nz))
            throw std::invalid_argument(
                "implicit diffusion target field shapes disagree");
    const int momentum_fields = static_cast<int>(state.u != nullptr) +
        static_cast<int>(state.v != nullptr) +
        static_cast<int>(state.w != nullptr);
    if (momentum_fields != 0 && momentum_fields != 3)
        throw std::invalid_argument(
            "implicit momentum diffusion requires u, v, and w together");
    if (cfg.use_variable_K && (state.theta || state.qv) &&
        !matches_shape(state.K_scalar, nr, nth, nz))
        throw std::invalid_argument(
            "implicit variable scalar diffusion requires a matching coefficient field");
    if (cfg.use_variable_K && state.u &&
        !matches_shape(state.K_momentum, nr, nth, nz))
        throw std::invalid_argument(
            "implicit variable momentum diffusion requires a matching coefficient field");

    // A solve can reject a later target after an earlier target completed.
    // Keep those partial results private until every target and diagnostic has
    // succeeded so caller-owned tendencies remain unchanged on failure.
    DiffusionTendencies candidate;
    ensure_tendency_shape(candidate.dudt_diff, nr, nth, nz);
    ensure_tendency_shape(candidate.dvdt_diff, nr, nth, nz);
    ensure_tendency_shape(candidate.dwdt_diff, nr, nth, nz);
    ensure_tendency_shape(candidate.dthetadt_diff, nr, nth, nz);
    ensure_tendency_shape(candidate.dqvdt_diff, nr, nth, nz);

    const Field3D* scalar_k = cfg.use_variable_K ? state.K_scalar : nullptr;
    const Field3D* momentum_k = cfg.use_variable_K ? state.K_momentum : nullptr;
    const double inv_dt = 1.0 / cfg.dt_diffusion;
    std::vector<float>       device_dz;
    DiffusionComputeKernels* device_kernels = nullptr;
    if (native_requested)
    {
        check_stability(cfg, state);
        auto* backend  = state.compute_runtime->backend.get();
        device_kernels = backend != nullptr ? backend->diffusion_kernels() : nullptr;
        if (device_kernels == nullptr || !device_kernels->supports(DiffusionKernelOp::Implicit))
            throw std::runtime_error("native implicit diffusion kernels are unavailable; CPU substitution is disabled");
        device_dz.assign(state.grid->dz.begin(), state.grid->dz.end());
    }

    auto solve_target = [&](const Field3D* field, const Field3D* K_field,
                            ImplicitVerticalComponent component,
                            Field3D& tendency)
    {
        if (!active || field == nullptr)
            return;
        if (native_requested)
        {
            DiffusionDispatch dispatch;
            dispatch.field      = field->data();
            dispatch.rho        = state.rho->data();
            dispatch.dz         = device_dz.data();
            dispatch.tendency   = tendency.data();
            dispatch.nx         = nr;
            dispatch.ny         = nth;
            dispatch.nz         = nz;
            dispatch.dx         = static_cast<float>(state.grid->dx);
            dispatch.dy         = static_cast<float>(state.grid->dy);
            dispatch.K_h        = static_cast<float>(cfg.K_h);
            dispatch.K_v        = static_cast<float>(cfg.K_v);
            dispatch.location   = static_cast<DiffusionFieldLocation>(component);
            dispatch.c_grid     = state.c_grid;
            dispatch.periodic_x = state.periodic_x;
            dispatch.periodic_y = state.periodic_y;
            dispatch.vertical_K = K_field != nullptr ? K_field->data() : nullptr;
            nimbus::compute::ImplicitDiffusionSolveReport report;
            if (!nimbus::compute::solve_implicit_diffusion(
                    *device_kernels, dispatch, static_cast<float>(cfg.dt_diffusion), report))
                throw std::runtime_error(
                    "native implicit diffusion solve failed or did not converge; no result was published");
            return;
        }
        Field3D updated = *field;
        if (component == ImplicitVerticalComponent::Scalar && cfg.K_h > 0.0)
            implicit_scalar_diffusion(
                *field, *state.rho, K_field, cfg, state, updated);
        else
            implicit_vertical_diffusion(
                *field, *state.rho, K_field, cfg.K_v, *state.grid,
                cfg.dt_diffusion, state.c_grid, state.periodic_x,
                state.periodic_y, component, updated);
        const VerticalDomain domain = component_domain(
            nr, nth, nz, state.c_grid, state.periodic_x,
            state.periodic_y, component);
        #pragma omp parallel for collapse(2)
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                    tendency(i, j, k) = static_cast<float>(
                        (static_cast<double>(updated(i, j, k)) -
                         static_cast<double>((*field)(i, j, k))) * inv_dt);
    };

    solve_target(state.theta, scalar_k, ImplicitVerticalComponent::Scalar,
                 candidate.dthetadt_diff);
    solve_target(state.qv, scalar_k, ImplicitVerticalComponent::Scalar,
                 candidate.dqvdt_diff);
    if (state.u && state.v && state.w && (cfg.K_h == 0.0 || native_requested))
    {
        solve_target(state.u, momentum_k, ImplicitVerticalComponent::U,
                     candidate.dudt_diff);
        solve_target(state.v, momentum_k, ImplicitVerticalComponent::V,
                     candidate.dvdt_diff);
        solve_target(state.w, momentum_k, ImplicitVerticalComponent::W,
                     candidate.dwdt_diff);
    }
    else if (state.u && state.v && state.w)
    {
        Field3D updated_u;
        Field3D updated_v;
        Field3D updated_w;
        implicit_momentum_diffusion(
            *state.u, *state.v, *state.w, *state.rho, momentum_k,
            cfg, state, updated_u, updated_v, updated_w);
        const auto store_tendency = [&](const Field3D& old_field,
                                        const Field3D& new_field,
                                        ImplicitVerticalComponent component,
                                        Field3D& tendency)
        {
            const VerticalDomain domain = component_domain(
                nr, nth, nz, state.c_grid, state.periodic_x,
                state.periodic_y, component);
            for (int i = domain.i_begin; i < domain.i_end; ++i)
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                        tendency(i, j, k) = static_cast<float>(
                            (static_cast<double>(new_field(i, j, k))
                             - static_cast<double>(old_field(i, j, k)))
                            * inv_dt);
        };
        store_tendency(*state.u, updated_u, ImplicitVerticalComponent::U,
                       candidate.dudt_diff);
        store_tendency(*state.v, updated_v, ImplicitVerticalComponent::V,
                       candidate.dvdt_diff);
        store_tendency(*state.w, updated_w, ImplicitVerticalComponent::W,
                       candidate.dwdt_diff);
    }

    Field3D diagnostic_effective_k;
    if (diag_opt != nullptr)
    {
        diagnostic_effective_k.resize(nr, nth, nz, 0.0f);
        const VerticalDomain domain = scalar_domain(
            nr, nth, nz, state.c_grid);
#pragma omp parallel for collapse(2) schedule(static)
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    double maximum = std::max(cfg.K_h, cfg.use_variable_K ? 0.0 : cfg.K_v);
                    if (scalar_k != nullptr && (state.theta != nullptr || state.qv != nullptr))
                        maximum = std::max(maximum, static_cast<double>((*scalar_k)(i, j, k)));
                    if (momentum_k != nullptr && state.u != nullptr)
                        maximum = std::max(maximum, static_cast<double>((*momentum_k)(i, j, k)));
                    diagnostic_effective_k(i, j, k) = static_cast<float>(maximum);
                }
    }

    tendencies = std::move(candidate);
    if (diag_opt != nullptr)
    {
        diag_opt->K_effective = std::move(diagnostic_effective_k);
        diag_opt->max_diffusion_number = 0.0;
    }
}

double ImplicitDiffusionScheme::check_stability(
    const DiffusionConfig& cfg, const DiffusionStateView& state)
{
    validate_config(cfg);
    if (!state.grid)
        throw std::invalid_argument(
            "implicit diffusion stability check requires grid metrics");
    if (cfg.K_h <= 0.0 && cfg.K_v <= 0.0 && !cfg.use_variable_K)
        return 0.0;
    if (state.grid->terrain_metrics_active && state.compute_runtime != nullptr &&
        active_compute_backend_kind(*state.compute_runtime) != ComputeBackendKind::Cpu)
        throw std::runtime_error("native implicit diffusion requires flat geometry; CPU substitution is disabled");
    const Field3D* ref = state.theta ? state.theta :
        (state.u ? state.u : state.qv);
    if (!ref || !matches_shape(state.rho, ref->size_r(), ref->size_th(),
                               ref->size_z()))
        throw std::invalid_argument(
            "active implicit diffusion stability check requires matching state and live density");
    grid_metric::validate_vertical_metrics(
        *state.grid, ref->size_r(), ref->size_th(), ref->size_z());
    if (state.grid->terrain_metrics_active
        && state.grid->terrain_metrics == nullptr)
        throw std::invalid_argument(
            "implicit terrain stability requires bound conservative geometry");

    // Reuse the production assembly to validate every active density,
    // coefficient, control volume, and pivot before claiming the M-matrix
    // has no finite explicit stability limit.
    auto validate_target = [&](const Field3D* field, const Field3D* K_field,
                               ImplicitVerticalComponent component)
    {
        if (!field) return;
        if (!matches_shape(field, ref->size_r(), ref->size_th(), ref->size_z()))
            throw std::invalid_argument(
                "implicit diffusion stability target shape mismatch");
        Field3D scratch = *field;
        if (component == ImplicitVerticalComponent::Scalar && cfg.K_h > 0.0)
            implicit_scalar_diffusion(
                *field, *state.rho, K_field, cfg, state, scratch);
        else
            implicit_vertical_diffusion(
                *field, *state.rho, K_field, cfg.K_v, *state.grid,
                cfg.dt_diffusion, state.c_grid, state.periodic_x,
                state.periodic_y, component, scratch);
    };
    const Field3D* scalar_k = cfg.use_variable_K ? state.K_scalar : nullptr;
    const Field3D* momentum_k = cfg.use_variable_K ? state.K_momentum : nullptr;
    if (cfg.use_variable_K && (state.theta || state.qv) &&
        !matches_shape(scalar_k, ref->size_r(), ref->size_th(), ref->size_z()))
        throw std::invalid_argument(
            "implicit variable scalar stability check requires its coefficient field");
    if (cfg.use_variable_K && state.u &&
        !matches_shape(momentum_k, ref->size_r(), ref->size_th(), ref->size_z()))
        throw std::invalid_argument(
            "implicit variable momentum stability check requires its coefficient field");
    if (!state.grid->terrain_metrics_active)
    {
        const int momentum_fields = static_cast<int>(state.u != nullptr) + static_cast<int>(state.v != nullptr) +
                                    static_cast<int>(state.w != nullptr);
        if (momentum_fields != 0 && momentum_fields != 3)
            throw std::invalid_argument("implicit momentum stability requires u, v, and w together");
        const std::pair<const Field3D*, ImplicitVerticalComponent> fields[] = {
            {state.theta, ImplicitVerticalComponent::Scalar},
            {state.qv, ImplicitVerticalComponent::Scalar},
            {state.u, ImplicitVerticalComponent::U},
            {state.v, ImplicitVerticalComponent::V},
            {state.w, ImplicitVerticalComponent::W}};
        for (const auto& entry : fields)
        {
            const auto* field = entry.first;
            if (field == nullptr)
                continue;
            if (!matches_shape(field, ref->size_r(), ref->size_th(), ref->size_z()))
                throw std::invalid_argument("implicit stability target shape mismatch");
            const auto domain = component_domain(ref->size_r(),
                                                 ref->size_th(),
                                                 ref->size_z(),
                                                 state.c_grid,
                                                 state.periodic_x,
                                                 state.periodic_y,
                                                 entry.second);
            for (int i = domain.i_begin; i < domain.i_end; ++i)
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                        if (!std::isfinite((*field)(i, j, k)))
                            throw std::invalid_argument("implicit diffusion requires finite physical target fields");
        }
        ExplicitDiffusionScheme validator;
        if (!std::isfinite(validator.check_stability(cfg, state)))
            throw std::invalid_argument("implicit diffusion coefficients exceed finite matrix storage");
        return 0.0;
    }
    validate_target(state.theta, scalar_k, ImplicitVerticalComponent::Scalar);
    validate_target(state.qv, scalar_k, ImplicitVerticalComponent::Scalar);
    if (state.u || state.v || state.w)
    {
        if (!(state.u && state.v && state.w))
            throw std::invalid_argument(
                "implicit momentum stability check requires u, v, and w together");
        if (cfg.K_h > 0.0)
        {
            Field3D scratch_u;
            Field3D scratch_v;
            Field3D scratch_w;
            implicit_momentum_diffusion(
                *state.u, *state.v, *state.w, *state.rho, momentum_k,
                cfg, state, scratch_u, scratch_v, scratch_w);
        }
        else
        {
            validate_target(state.u, momentum_k, ImplicitVerticalComponent::U);
            validate_target(state.v, momentum_k, ImplicitVerticalComponent::V);
            validate_target(state.w, momentum_k, ImplicitVerticalComponent::W);
        }
    }
    return 0.0;
}

void ImplicitDiffusionScheme::solve_tridiagonal(
    const std::vector<double>& a, const std::vector<double>& b,
    const std::vector<double>& c, const std::vector<double>& rhs,
    std::vector<double>& x)
{
    if (a.size() != b.size() || b.size() != c.size() ||
        c.size() != rhs.size() || b.empty())
        throw std::invalid_argument(
            "implicit tridiagonal arrays must be non-empty and equal-sized");
    const int n = static_cast<int>(b.size());
    x.assign(static_cast<std::size_t>(n), 0.0);
    std::vector<double> cp(static_cast<std::size_t>(n), 0.0);
    std::vector<double> dp(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) ||
            !std::isfinite(c[i]) || !std::isfinite(rhs[i]))
            throw std::runtime_error(
                "implicit tridiagonal system contains non-finite coefficients");
    if (b[0] <= 0.0)
        throw std::runtime_error(
            "implicit diffusion tridiagonal pivot is not positive");
    cp[0] = n > 1 ? c[0] / b[0] : 0.0;
    dp[0] = rhs[0] / b[0];
    for (int i = 1; i < n; ++i)
    {
        const double pivot = b[i] - a[i] * cp[i - 1];
        if (!std::isfinite(pivot) || pivot <= 0.0)
            throw std::runtime_error(
                "implicit diffusion tridiagonal pivot is not positive");
        cp[i] = i + 1 < n ? c[i] / pivot : 0.0;
        dp[i] = (rhs[i] - a[i] * dp[i - 1]) / pivot;
    }
    x[static_cast<std::size_t>(n - 1)] = dp[static_cast<std::size_t>(n - 1)];
    for (int i = n - 2; i >= 0; --i)
        x[static_cast<std::size_t>(i)] = dp[static_cast<std::size_t>(i)] -
            cp[static_cast<std::size_t>(i)] *
            x[static_cast<std::size_t>(i + 1)];
}

void ImplicitDiffusionScheme::implicit_scalar_diffusion(
    const Field3D& field, const Field3D& rho, const Field3D* K_field,
    const DiffusionConfig& cfg, const DiffusionStateView& state,
    Field3D& field_new)
{
    const int nr = field.size_r();
    const int nth = field.size_th();
    const int nz = field.size_z();
    const VerticalDomain domain = scalar_domain(nr, nth, nz, state.c_grid);
    if (domain.i_begin >= domain.i_end || domain.j_begin >= domain.j_end ||
        domain.k_begin >= domain.k_end)
        throw std::invalid_argument(
            "implicit scalar diffusion has no prognostic control volumes");

    const TerrainMetrics3D* terrain = state.grid->terrain_metrics_active
        ? state.grid->terrain_metrics : nullptr;
    if (state.grid->terrain_metrics_active && terrain == nullptr)
        throw std::invalid_argument(
            "implicit mapped scalar diffusion is missing conservative geometry");

    const auto mass_at = [&](int i, int j, int k)
    {
        const double density = static_cast<double>(rho(i, j, k));
        const double volume = terrain != nullptr
            ? terrain->J(i, j, k) * terrain->dx * terrain->dy
                * terrain->dzeta
            : state.grid->dx * state.grid->dy
                * state.grid->dz[static_cast<std::size_t>(k)];
        const double mass = density * volume;
        if (!std::isfinite(mass) || mass <= 0.0)
            throw std::invalid_argument(
                "implicit scalar diffusion requires positive finite control-volume mass");
        return mass;
    };

    DiffusionConfig operator_cfg = cfg;
    operator_cfg.scheme_id = "explicit";
    operator_cfg.apply_to = "scalars";
    operator_cfg.implicit_dim = "none";
    ExplicitDiffusionScheme explicit_operator;
    DiffusionTendencies operator_tendencies;
    DiffusionStateView operator_state;
    operator_state.rho = &rho;
    operator_state.K_scalar = K_field;
    operator_state.grid = state.grid;
    operator_state.c_grid = state.c_grid;
    operator_state.periodic_x = state.periodic_x;
    operator_state.periodic_y = state.periodic_y;

    Field3D increment(nr, nth, nz, 0.0f);
    Field3D residual(nr, nth, nz, 0.0f);
    Field3D direction(nr, nth, nz, 0.0f);
    Field3D operator_direction(nr, nth, nz, 0.0f);
    field_new = field;

    const auto apply_system = [&](const Field3D& input, Field3D& output)
    {
        operator_state.theta = &input;
        explicit_operator.compute_diffusion_tendencies(
            operator_cfg, operator_state, operator_tendencies, nullptr);
        output.fill(0.0f);
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const double value = static_cast<double>(input(i, j, k))
                        - cfg.dt_diffusion * static_cast<double>(
                            operator_tendencies.dthetadt_diff(i, j, k));
                    const float stored = static_cast<float>(value);
                    if (!std::isfinite(stored))
                        throw std::runtime_error(
                            "implicit scalar system application exceeded finite storage");
                    output(i, j, k) = stored;
                }
    };
    const auto weighted_dot = [&](const Field3D& left, const Field3D& right)
    {
        double sum = 0.0;
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                    sum += mass_at(i, j, k)
                        * static_cast<double>(left(i, j, k))
                        * static_cast<double>(right(i, j, k));
        if (!std::isfinite(sum))
            throw std::runtime_error("implicit scalar solver reduction became non-finite");
        return sum;
    };

    // Solve for the increment so a large background cannot erase the forcing
    // or inflate the convergence tolerance for a small but representable update.
    operator_state.theta = &field;
    explicit_operator.compute_diffusion_tendencies(operator_cfg, operator_state, operator_tendencies, nullptr);
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                residual(i, j, k) = static_cast<float>(cfg.dt_diffusion *
                                                       static_cast<double>(operator_tendencies.dthetadt_diff(i, j, k)));
                direction(i, j, k) = residual(i, j, k);
            }
    double       residual_norm2 = weighted_dot(residual, residual);
    const double initial_norm2  = residual_norm2;
    const double tolerance2     = 1.0e-12 * initial_norm2;
    const int    unknowns =
        (domain.i_end - domain.i_begin) * (domain.j_end - domain.j_begin) * (domain.k_end - domain.k_begin);
    const int maximum_iterations = std::max(32, std::min(8 * unknowns, 4096));
    int       iterations         = 0;
    while (residual_norm2 > tolerance2 && iterations < maximum_iterations)
    {
        apply_system(direction, operator_direction);
        const double denominator = weighted_dot(direction, operator_direction);
        if (!std::isfinite(denominator) || denominator <= 0.0)
            throw std::runtime_error("implicit scalar diffusion lost positive definiteness");
        const double alpha = residual_norm2 / denominator;
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    increment(i, j, k) = static_cast<float>(static_cast<double>(increment(i, j, k)) +
                                                            alpha * static_cast<double>(direction(i, j, k)));
                    residual(i, j, k)  = static_cast<float>(static_cast<double>(residual(i, j, k)) -
                                                            alpha * static_cast<double>(operator_direction(i, j, k)));
                }
        const double next_norm2 = weighted_dot(residual, residual);
        if (next_norm2 <= tolerance2)
        {
            residual_norm2 = next_norm2;
            ++iterations;
            break;
        }
        const double beta = next_norm2 / residual_norm2;
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                    direction(i, j, k) = static_cast<float>(static_cast<double>(residual(i, j, k)) +
                                                            beta * static_cast<double>(direction(i, j, k)));
        residual_norm2 = next_norm2;
        ++iterations;
    }
    if (residual_norm2 > tolerance2)
        throw std::runtime_error("implicit scalar diffusion failed to converge within its bounded iteration budget");
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const float candidate = static_cast<float>(static_cast<double>(field(i, j, k)) + increment(i, j, k));
                if (!std::isfinite(candidate))
                {
                    throw std::runtime_error("implicit scalar update exceeded finite storage");
                }
                field_new(i, j, k) = candidate;
            }
}

void ImplicitDiffusionScheme::implicit_momentum_diffusion(const Field3D&            u,
                                                          const Field3D&            v,
                                                          const Field3D&            w,
                                                          const Field3D&            rho,
                                                          const Field3D*            K_field,
                                                          const DiffusionConfig&    cfg,
                                                          const DiffusionStateView& state,
                                                          Field3D&                  u_new,
                                                          Field3D&                  v_new,
                                                          Field3D&                  w_new)
{
    const int            nr         = u.size_r();
    const int            nth        = u.size_th();
    const int            nz         = u.size_z();
    const VerticalDomain domains[3] = {
        component_domain(nr, nth, nz, state.c_grid, state.periodic_x, state.periodic_y, ImplicitVerticalComponent::U),
        component_domain(nr, nth, nz, state.c_grid, state.periodic_x, state.periodic_y, ImplicitVerticalComponent::V),
        component_domain(nr, nth, nz, state.c_grid, state.periodic_x, state.periodic_y, ImplicitVerticalComponent::W),
    };
    for (const VerticalDomain& domain : domains)
        if (domain.i_begin >= domain.i_end ||
            domain.j_begin >= domain.j_end ||
            domain.k_begin >= domain.k_end)
            throw std::invalid_argument(
                "implicit momentum diffusion has no prognostic control volumes");

    const TerrainMetrics3D* terrain = state.grid->terrain_metrics_active
        ? state.grid->terrain_metrics : nullptr;
    if (state.grid->terrain_metrics_active && terrain == nullptr)
        throw std::invalid_argument(
            "implicit mapped momentum diffusion is missing conservative geometry");
    const auto scalar_mass = [&](int i, int j, int k)
    {
        const double density = static_cast<double>(rho(i, j, k));
        const double volume = terrain != nullptr
            ? terrain->J(i, j, k) * terrain->dx * terrain->dy
                * terrain->dzeta
            : state.grid->dx * state.grid->dy
                * state.grid->dz[static_cast<std::size_t>(k)];
        const double mass = density * volume;
        if (!std::isfinite(mass) || mass <= 0.0)
            throw std::invalid_argument(
                "implicit momentum diffusion requires positive finite scalar mass");
        return mass;
    };
    const auto component_mass = [&](int component, int i, int j, int k)
    {
        if (!state.c_grid)
            return scalar_mass(i, j, k);
        if (component == 0)
        {
            const int right = state.periodic_x && i == nr - 2 ? 1 : i + 1;
            return 0.5 * (scalar_mass(i, j, k)
                        + scalar_mass(right, j, k));
        }
        if (component == 1)
        {
            const int north = state.periodic_y && j == nth - 2 ? 1 : j + 1;
            return 0.5 * (scalar_mass(i, j, k)
                        + scalar_mass(i, north, k));
        }
        return 0.5 * (scalar_mass(i, j, k)
                    + scalar_mass(i, j, k + 1));
    };

    DiffusionConfig operator_cfg = cfg;
    operator_cfg.scheme_id = "explicit";
    operator_cfg.apply_to = "momentum";
    operator_cfg.implicit_dim = "none";
    ExplicitDiffusionScheme explicit_operator;
    DiffusionTendencies operator_tendencies;
    DiffusionStateView operator_state;
    operator_state.rho = &rho;
    operator_state.K_momentum = K_field;
    operator_state.grid = state.grid;
    operator_state.c_grid = state.c_grid;
    operator_state.periodic_x = state.periodic_x;
    operator_state.periodic_y = state.periodic_y;

    struct Vector3
    {
        Field3D u;
        Field3D v;
        Field3D w;
    };
    const auto make_vector = [&]()
    {
        return Vector3{Field3D(nr, nth, nz, 0.0f),
                       Field3D(nr, nth, nz, 0.0f),
                       Field3D(nr, nth, nz, 0.0f)};
    };
    Vector3 image = make_vector();
    Vector3 residual = make_vector();
    Vector3 direction = make_vector();
    Vector3 operator_direction = make_vector();
    Vector3 shadow_residual = make_vector();
    Vector3 intermediate = make_vector();
    Vector3 operator_intermediate = make_vector();
    u_new                         = Field3D(nr, nth, nz, 0);
    v_new                         = Field3D(nr, nth, nz, 0);
    w_new                         = Field3D(nr, nth, nz, 0);
    const auto finalize           = [&]()
    {
        const Field3D* original[3] = {&u, &v, &w};
        Field3D*       result[3]   = {&u_new, &v_new, &w_new};
        for (int component = 0; component < 3; ++component)
            for (int i = 0; i < nr; ++i)
                for (int j = 0; j < nth; ++j)
                    for (int k = 0; k < nz; ++k)
                    {
                        const auto& domain = domains[component];
                        float       value  = (*original[component])(i, j, k);
                        if (i >= domain.i_begin && i < domain.i_end && j >= domain.j_begin && j < domain.j_end &&
                            k >= domain.k_begin && k < domain.k_end)
                        {
                            value = static_cast<float>(static_cast<double>(value) + (*result[component])(i, j, k));
                            if (!std::isfinite(value))
                                throw std::runtime_error("implicit momentum update exceeded finite storage");
                        }
                        (*result[component])(i, j, k) = value;
                    }
    };

    const auto apply_system = [&](const Field3D& input_u,
                                  const Field3D& input_v,
                                  const Field3D& input_w,
                                  Vector3& output)
    {
        operator_state.u = &input_u;
        operator_state.v = &input_v;
        operator_state.w = &input_w;
        explicit_operator.compute_diffusion_tendencies(
            operator_cfg, operator_state, operator_tendencies, nullptr);
        output.u.fill(0.0f);
        output.v.fill(0.0f);
        output.w.fill(0.0f);
        const Field3D* inputs[3] = {&input_u, &input_v, &input_w};
        const Field3D* tendencies[3] = {
            &operator_tendencies.dudt_diff,
            &operator_tendencies.dvdt_diff,
            &operator_tendencies.dwdt_diff,
        };
        Field3D* outputs[3] = {&output.u, &output.v, &output.w};
        for (int component = 0; component < 3; ++component)
        {
            const VerticalDomain& domain = domains[component];
            for (int i = domain.i_begin; i < domain.i_end; ++i)
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        const double value = static_cast<double>(
                            (*inputs[component])(i, j, k))
                            - cfg.dt_diffusion * static_cast<double>(
                                (*tendencies[component])(i, j, k));
                        const float stored = static_cast<float>(value);
                        if (!std::isfinite(stored))
                            throw std::runtime_error(
                                "implicit momentum system application exceeded finite storage");
                        (*outputs[component])(i, j, k) = stored;
                    }
        }
    };
    const auto weighted_dot = [&](const Vector3& left, const Vector3& right)
    {
        const Field3D* left_fields[3] = {&left.u, &left.v, &left.w};
        const Field3D* right_fields[3] = {&right.u, &right.v, &right.w};
        double sum = 0.0;
        for (int component = 0; component < 3; ++component)
        {
            const VerticalDomain& domain = domains[component];
            for (int i = domain.i_begin; i < domain.i_end; ++i)
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                        sum += component_mass(component, i, j, k)
                            * static_cast<double>(
                                (*left_fields[component])(i, j, k))
                            * static_cast<double>(
                                (*right_fields[component])(i, j, k));
        }
        if (!std::isfinite(sum))
            throw std::runtime_error(
                "implicit momentum solver reduction became non-finite");
        return sum;
    };
    const auto copy_difference = [&](Vector3& target,
                                     const Field3D& left_u,
                                     const Field3D& left_v,
                                     const Field3D& left_w,
                                     const Vector3& right)
    {
        const Field3D* left[3] = {&left_u, &left_v, &left_w};
        const Field3D* rhs[3] = {&right.u, &right.v, &right.w};
        Field3D* dst[3] = {&target.u, &target.v, &target.w};
        for (int component = 0; component < 3; ++component)
        {
            const VerticalDomain& domain = domains[component];
            for (int i = domain.i_begin; i < domain.i_end; ++i)
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                        (*dst[component])(i, j, k) = static_cast<float>(
                            static_cast<double>((*left[component])(i, j, k))
                            - static_cast<double>((*rhs[component])(i, j, k)));
        }
    };

    operator_state.u = &u;
    operator_state.v = &v;
    operator_state.w = &w;
    explicit_operator.compute_diffusion_tendencies(operator_cfg, operator_state, operator_tendencies, nullptr);
    Vector3        rhs        = make_vector();
    Field3D*       forcing[3] = {&rhs.u, &rhs.v, &rhs.w};
    const Field3D* rates[3]   = {
        &operator_tendencies.dudt_diff, &operator_tendencies.dvdt_diff, &operator_tendencies.dwdt_diff};
    for (int component = 0; component < 3; ++component)
    {
        const auto& domain = domains[component];
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                    (*forcing[component])(i, j, k) =
                        static_cast<float>(cfg.dt_diffusion * (*rates[component])(i, j, k));
    }
    residual                     = rhs;
    shadow_residual = residual;
    double       residual_norm2  = weighted_dot(residual, residual);
    const double initial_norm2 = residual_norm2;
    const double tolerance2      = 1.0e-12 * initial_norm2;
    const double explicit_number = explicit_operator.check_stability(
        operator_cfg, operator_state);
    if (explicit_number < 0.8)
    {
        int fixed_point_iterations = 0;
        while (residual_norm2 > tolerance2 &&
               fixed_point_iterations < 256)
        {
            Field3D* solution[3] = {&u_new, &v_new, &w_new};
            const Field3D* correction[3] = {
                &residual.u, &residual.v, &residual.w};
            for (int component = 0; component < 3; ++component)
            {
                const VerticalDomain& domain = domains[component];
                for (int i = domain.i_begin; i < domain.i_end; ++i)
                    for (int j = domain.j_begin; j < domain.j_end; ++j)
                        for (int k = domain.k_begin; k < domain.k_end; ++k)
                            (*solution[component])(i, j, k) =
                                static_cast<float>(
                                    static_cast<double>(
                                        (*solution[component])(i, j, k))
                                    + static_cast<double>(
                                        (*correction[component])(i, j, k)));
            }
            apply_system(u_new, v_new, w_new, image);
            copy_difference(residual, rhs.u, rhs.v, rhs.w, image);
            residual_norm2 = weighted_dot(residual, residual);
            ++fixed_point_iterations;
        }
        if (residual_norm2 > tolerance2)
            throw std::runtime_error(
                "implicit momentum fixed-point solve failed to converge within its bounded iteration budget");
        finalize();
        return;
    }
    int unknowns = 0;
    for (const VerticalDomain& domain : domains)
        unknowns += (domain.i_end - domain.i_begin)
            * (domain.j_end - domain.j_begin)
            * (domain.k_end - domain.k_begin);
    const int maximum_iterations = std::max(
        32, std::min(8 * unknowns, 4096));
    int iterations = 0;
    double previous_rho = 1.0;
    double alpha = 1.0;
    double omega = 1.0;
    int restart_iteration = 0;
    while (residual_norm2 > tolerance2 && iterations < maximum_iterations)
    {
        const double rho_value = weighted_dot(shadow_residual, residual);
        if (!std::isfinite(rho_value) || rho_value == 0.0)
            throw std::runtime_error(
                "implicit momentum BiCGSTAB shadow-residual breakdown");
        const double beta = restart_iteration == 0 ? 0.0
            : (rho_value / previous_rho) * (alpha / omega);
        Field3D* direction_fields[3] = {
            &direction.u, &direction.v, &direction.w};
        const Field3D* residual_fields[3] = {
            &residual.u, &residual.v, &residual.w};
        const Field3D* prior_operator_fields[3] = {
            &operator_direction.u, &operator_direction.v,
            &operator_direction.w};
        for (int component = 0; component < 3; ++component)
        {
            const VerticalDomain& domain = domains[component];
            for (int i = domain.i_begin; i < domain.i_end; ++i)
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                        (*direction_fields[component])(i, j, k) =
                            static_cast<float>(
                                static_cast<double>(
                                    (*residual_fields[component])(i, j, k))
                                + beta * (
                                    static_cast<double>(
                                        (*direction_fields[component])(
                                            i, j, k))
                                    - omega * static_cast<double>(
                                        (*prior_operator_fields[component])(
                                            i, j, k))));
        }
        apply_system(direction.u, direction.v, direction.w,
                     operator_direction);
        const double denominator = weighted_dot(
            shadow_residual, operator_direction);
        if (!std::isfinite(denominator) || denominator == 0.0)
            throw std::runtime_error(
                "implicit momentum BiCGSTAB alpha breakdown");
        alpha = rho_value / denominator;
        Field3D* solution[3] = {&u_new, &v_new, &w_new};
        const Field3D* operator_fields[3] = {
            &operator_direction.u, &operator_direction.v,
            &operator_direction.w};
        Field3D* intermediate_fields[3] = {
            &intermediate.u, &intermediate.v, &intermediate.w};
        for (int component = 0; component < 3; ++component)
        {
            const VerticalDomain& domain = domains[component];
            for (int i = domain.i_begin; i < domain.i_end; ++i)
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        (*intermediate_fields[component])(i, j, k) =
                            static_cast<float>(
                                static_cast<double>(
                                    (*residual_fields[component])(i, j, k))
                                - alpha * static_cast<double>(
                                    (*operator_fields[component])(i, j, k)));
                    }
        }
        const double intermediate_norm2 = weighted_dot(
            intermediate, intermediate);
        if (intermediate_norm2 <= tolerance2)
        {
            for (int component = 0; component < 3; ++component)
            {
                const VerticalDomain& domain = domains[component];
                for (int i = domain.i_begin; i < domain.i_end; ++i)
                    for (int j = domain.j_begin; j < domain.j_end; ++j)
                        for (int k = domain.k_begin; k < domain.k_end; ++k)
                            (*solution[component])(i, j, k) =
                                static_cast<float>(
                                    static_cast<double>(
                                        (*solution[component])(i, j, k))
                                    + alpha * static_cast<double>(
                                        (*direction_fields[component])(
                                            i, j, k)));
            }
            residual = intermediate;
            residual_norm2 = intermediate_norm2;
            ++iterations;
            break;
        }
        apply_system(intermediate.u, intermediate.v, intermediate.w,
                     operator_intermediate);
        const double operator_norm2 = weighted_dot(
            operator_intermediate, operator_intermediate);
        if (!std::isfinite(operator_norm2) || operator_norm2 <= 0.0)
            throw std::runtime_error(
                "implicit momentum BiCGSTAB omega breakdown");
        omega = weighted_dot(operator_intermediate, intermediate)
            / operator_norm2;
        if (!std::isfinite(omega) || omega == 0.0)
            throw std::runtime_error(
                "implicit momentum BiCGSTAB omega became singular");
        Field3D* mutable_residual[3] = {
            &residual.u, &residual.v, &residual.w};
        const Field3D* intermediate_const[3] = {
            &intermediate.u, &intermediate.v, &intermediate.w};
        const Field3D* operator_intermediate_fields[3] = {
            &operator_intermediate.u, &operator_intermediate.v,
            &operator_intermediate.w};
        for (int component = 0; component < 3; ++component)
        {
            const VerticalDomain& domain = domains[component];
            for (int i = domain.i_begin; i < domain.i_end; ++i)
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        (*solution[component])(i, j, k) =
                            static_cast<float>(
                                static_cast<double>(
                                    (*solution[component])(i, j, k))
                                + alpha * static_cast<double>(
                                    (*direction_fields[component])(i, j, k))
                                + omega * static_cast<double>(
                                    (*intermediate_const[component])(
                                        i, j, k)));
                        (*mutable_residual[component])(i, j, k) =
                            static_cast<float>(
                                static_cast<double>(
                                    (*intermediate_const[component])(
                                        i, j, k))
                                - omega * static_cast<double>(
                                    (*operator_intermediate_fields[component])(
                                        i, j, k)));
                    }
        }
        residual_norm2 = weighted_dot(residual, residual);
        previous_rho = rho_value;
        ++iterations;
        ++restart_iteration;
        if (restart_iteration == 32 && residual_norm2 > tolerance2)
        {
            apply_system(u_new, v_new, w_new, image);
            copy_difference(residual, rhs.u, rhs.v, rhs.w, image);
            residual_norm2 = weighted_dot(residual, residual);
            shadow_residual = residual;
            direction.u.fill(0.0f);
            direction.v.fill(0.0f);
            direction.w.fill(0.0f);
            operator_direction.u.fill(0.0f);
            operator_direction.v.fill(0.0f);
            operator_direction.w.fill(0.0f);
            previous_rho = 1.0;
            alpha = 1.0;
            omega = 1.0;
            restart_iteration = 0;
        }
    }
    if (residual_norm2 > tolerance2)
    {
        apply_system(u_new, v_new, w_new, image);
        copy_difference(residual, rhs.u, rhs.v, rhs.w, image);
        residual_norm2 = weighted_dot(residual, residual);
    }
    if (residual_norm2 > tolerance2)
        throw std::runtime_error(
            "implicit momentum diffusion failed to converge within its bounded iteration budget");
    finalize();
}

void ImplicitDiffusionScheme::implicit_vertical_diffusion(
    const Field3D& field, const Field3D& rho, const Field3D* K_field,
    double K_default, const GridMetrics& grid, double dt, bool c_grid,
    bool periodic_x, bool periodic_y, ImplicitVerticalComponent component,
    Field3D& field_new)
{
    const int nr = field.size_r();
    const int nth = field.size_th();
    const int nz = field.size_z();
    if (!std::isfinite(dt) || dt <= 0.0 || !std::isfinite(K_default) ||
        K_default < 0.0 || !matches_shape(&rho, nr, nth, nz))
        throw std::invalid_argument(
            "implicit vertical diffusion requires matching density, finite non-negative K, and positive finite dt");
    if (K_field && !matches_shape(K_field, nr, nth, nz))
        throw std::invalid_argument(
            "implicit vertical diffusivity shape mismatch");
    grid_metric::validate_vertical_metrics(grid, nr, nth, nz);
    field_new = field;

    const VerticalDomain domain = component_domain(
        nr, nth, nz, c_grid, periodic_x, periodic_y, component);
    if (domain.i_begin >= domain.i_end || domain.j_begin >= domain.j_end ||
        domain.k_begin >= domain.k_end)
        throw std::invalid_argument(
            std::string("implicit vertical ") + component_label(component) +
            " diffusion has no prognostic control volumes");
    const bool c_grid_w = c_grid && component == ImplicitVerticalComponent::W;

    const TerrainMetrics3D* terrain = grid.terrain_metrics_active
        ? grid.terrain_metrics : nullptr;
    if (terrain != nullptr)
    {
        terrain_transport::validate_geometry(*terrain);
        if (terrain->J.size_r() != nr || terrain->J.size_th() != nth
            || terrain->J.size_z() != nz)
            throw std::invalid_argument(
                "implicit terrain geometry shape does not match the scalar field");
    }
    else if (grid.terrain_metrics_active)
    {
        throw std::invalid_argument(
            "implicit terrain diffusion is missing conservative geometry");
    }

    auto scalar_density = [&](int i, int j, int k)
    {
        const double value = static_cast<double>(rho(i, j, k));
        if (!std::isfinite(value) || value <= 0.0)
            throw std::invalid_argument(
                "implicit diffusion requires positive finite live scalar density");
        return value;
    };
    auto scalar_transport = [&](int i, int j, int k)
    {
        const double diffusivity = K_field ?
            static_cast<double>((*K_field)(i, j, k)) : K_default;
        if (!std::isfinite(diffusivity) || diffusivity < 0.0)
            throw std::invalid_argument(
                "implicit diffusivity must be finite and non-negative on the physical domain");
        return scalar_density(i, j, k) * diffusivity;
    };
    auto density_at = [&](int i, int j, int k)
    {
        if (!c_grid || component == ImplicitVerticalComponent::Scalar)
            return scalar_density(i, j, k);
        if (component == ImplicitVerticalComponent::U)
        {
            const int right = periodic_x && i == nr - 2 ? 1 : i + 1;
            return 0.5 * (scalar_density(i, j, k) +
                          scalar_density(right, j, k));
        }
        if (component == ImplicitVerticalComponent::V)
        {
            const int north = periodic_y && j == nth - 2 ? 1 : j + 1;
            return 0.5 * (scalar_density(i, j, k) +
                          scalar_density(i, north, k));
        }
        return 0.5 * (scalar_density(i, j, k) +
                      scalar_density(i, j, k + 1));
    };
    auto transport_at = [&](int i, int j, int k)
    {
        if (!c_grid || component == ImplicitVerticalComponent::Scalar)
            return scalar_transport(i, j, k);
        if (component == ImplicitVerticalComponent::U)
        {
            const int right = periodic_x && i == nr - 2 ? 1 : i + 1;
            return 0.5 * (scalar_transport(i, j, k) +
                          scalar_transport(right, j, k));
        }
        if (component == ImplicitVerticalComponent::V)
        {
            const int north = periodic_y && j == nth - 2 ? 1 : j + 1;
            return 0.5 * (scalar_transport(i, j, k) +
                          scalar_transport(i, north, k));
        }
        return 0.5 * (scalar_transport(i, j, k) +
                      scalar_transport(i, j, k + 1));
    };
    auto scalar_thickness = [&](int i, int j, int k)
    {
        return terrain != nullptr
            ? terrain->z_face(i, j, k + 1)
                - terrain->z_face(i, j, k)
            : grid.dz[static_cast<std::size_t>(k)];
    };
    auto point_height = [&](int i, int j, int k)
    {
        if (!c_grid || component == ImplicitVerticalComponent::Scalar)
            return terrain->z(i, j, k);
        if (component == ImplicitVerticalComponent::U)
        {
            const int right = periodic_x && i == nr - 2 ? 1 : i + 1;
            return 0.5 * (terrain->z(i, j, k)
                        + terrain->z(right, j, k));
        }
        if (component == ImplicitVerticalComponent::V)
        {
            const int north = periodic_y && j == nth - 2 ? 1 : j + 1;
            return 0.5 * (terrain->z(i, j, k)
                        + terrain->z(i, north, k));
        }
        return terrain->z_face(i, j, k + 1);
    };
    auto control_mass_per_area = [&](int i, int j, int k)
    {
        if (terrain == nullptr)
        {
            if (!c_grid_w)
                return density_at(i, j, k)
                    * control_thickness(grid, k, false);
            return 0.5 * (
                scalar_density(i, j, k)
                    * grid.dz[static_cast<std::size_t>(k)]
                + scalar_density(i, j, k + 1)
                    * grid.dz[static_cast<std::size_t>(k + 1)]);
        }
        const auto scalar_mass = [&](int ii, int jj, int kk)
        {
            return scalar_density(ii, jj, kk)
                * scalar_thickness(ii, jj, kk);
        };
        if (!c_grid || component == ImplicitVerticalComponent::Scalar)
            return scalar_mass(i, j, k);
        if (component == ImplicitVerticalComponent::U)
        {
            const int right = periodic_x && i == nr - 2 ? 1 : i + 1;
            return 0.5 * (
                scalar_mass(i, j, k) + scalar_mass(right, j, k));
        }
        if (component == ImplicitVerticalComponent::V)
        {
            const int north = periodic_y && j == nth - 2 ? 1 : j + 1;
            return 0.5 * (
                scalar_mass(i, j, k) + scalar_mass(i, north, k));
        }
        return 0.5 * (
            scalar_mass(i, j, k) + scalar_mass(i, j, k + 1));
    };

    const int levels = domain.k_end - domain.k_begin;
    std::vector<double> a(static_cast<std::size_t>(levels), 0.0);
    std::vector<double> b(static_cast<std::size_t>(levels), 1.0);
    std::vector<double> c(static_cast<std::size_t>(levels), 0.0);
    std::vector<double> rhs(static_cast<std::size_t>(levels), 0.0);
    std::vector<double> solution;
    // Keep assembly and solve outside an OpenMP region: invalid live science
    // state must propagate as an exception, not terminate the process at an
    // exception boundary owned by the OpenMP runtime.
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            std::fill(a.begin(), a.end(), 0.0);
            std::fill(b.begin(), b.end(), 1.0);
            std::fill(c.begin(), c.end(), 0.0);
            for (int local = 0; local < levels; ++local)
            {
                const int k = domain.k_begin + local;
                const double value = static_cast<double>(field(i, j, k));
                const double density = density_at(i, j, k);
                const double transport = transport_at(i, j, k);
                const double thickness = scalar_thickness(i, j, k);
                const double control_mass =
                    control_mass_per_area(i, j, k);
                if (!std::isfinite(value) || !std::isfinite(density) ||
                    density <= 0.0 || !std::isfinite(transport) ||
                    transport < 0.0 || !std::isfinite(thickness) ||
                    thickness <= 0.0 || !std::isfinite(control_mass) ||
                    control_mass <= 0.0)
                    throw std::invalid_argument(
                        std::string("implicit vertical ") +
                        component_label(component) +
                        " diffusion received invalid physical state or control volume");
                double lower = 0.0;
                double upper = 0.0;
                if (local > 0)
                {
                    const double face_transport = c_grid_w
                        ? scalar_transport(i, j, k)
                        : 0.5 * (transport_at(i, j, k - 1) + transport);
                    const double distance = terrain != nullptr
                        ? point_height(i, j, k)
                            - point_height(i, j, k - 1)
                        : point_distance(grid, k - 1, c_grid_w);
                    lower = face_transport / (control_mass * distance);
                }
                if (local + 1 < levels)
                {
                    const double face_transport = c_grid_w
                        ? scalar_transport(i, j, k + 1)
                        : 0.5 * (transport + transport_at(i, j, k + 1));
                    const double distance = terrain != nullptr
                        ? point_height(i, j, k + 1)
                            - point_height(i, j, k)
                        : point_distance(grid, k, c_grid_w);
                    upper = face_transport / (control_mass * distance);
                }
                if (!std::isfinite(lower) || lower < 0.0 ||
                    !std::isfinite(upper) || upper < 0.0)
                    throw std::invalid_argument(
                        "implicit diffusion assembled an invalid face coefficient");
                a[static_cast<std::size_t>(local)] = -dt * lower;
                b[static_cast<std::size_t>(local)] =
                    1.0 + dt * (lower + upper);
                c[static_cast<std::size_t>(local)] = -dt * upper;
                rhs[static_cast<std::size_t>(local)] = value;
            }
            solve_tridiagonal(a, b, c, rhs, solution);
            for (int local = 0; local < levels; ++local)
            {
                const float value = static_cast<float>(
                    solution[static_cast<std::size_t>(local)]);
                if (!std::isfinite(value))
                    throw std::runtime_error(
                        "implicit diffusion solution exceeds finite storage");
                field_new(i, j, domain.k_begin + local) = value;
            }
        }
}

} // namespace tmv
