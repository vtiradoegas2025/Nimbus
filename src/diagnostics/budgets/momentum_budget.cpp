#include "diagnostics/level_budget_csv.hpp"
/**
 * @file momentum_budget.cpp
 * @brief Domain-integrated momentum budget diagnostics (see header).
 */

#include "diagnostics/momentum_budget.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/grid_coordinates.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/simulation_state.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <mutex>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

tmv::LevelBudgetCsv g_momentum_csv_files;
std::mutex    g_momentum_csv_mutex;

constexpr const char* k_momentum_csv_header = "sim_time_s,stage,"
                                              "d_momentum_x,d_momentum_y,d_momentum_z,d_angular_momentum_z,"
                                              "rel_momentum_x,rel_momentum_y,rel_momentum_z,rel_angular_momentum_z,"
                                              "warn\n";

bool field_matches_grid(const Field3D& field, const GridStateView& grid)
{
    return field.size_r() == grid.nr && field.size_th() == grid.nth && field.size_z() == grid.nz;
}

bool momentum_fields_ready(const PrognosticFieldsView& fields, const GridStateView& grid)
{
    return field_matches_grid(fields.rho, grid) && field_matches_grid(fields.u, grid) &&
           field_matches_grid(fields.v, grid) && field_matches_grid(fields.w, grid);
}

double u_at_scalar_center(const PrognosticFieldsView& fields, int i, int j, int k, bool staggered)
{
    const double east = static_cast<double>(fields.u[i][j][k]);
    if (!staggered || i == 0)
    {
        return east;
    }
    return 0.5 * (static_cast<double>(fields.u[i - 1][j][k]) + east);
}

double v_at_scalar_center(const PrognosticFieldsView& fields, int i, int j, int k, bool staggered)
{
    const double north = static_cast<double>(fields.v[i][j][k]);
    if (!staggered || j == 0)
    {
        return north;
    }
    return 0.5 * (static_cast<double>(fields.v[i][j - 1][k]) + north);
}

double w_at_scalar_center(const PrognosticFieldsView& fields, int i, int j, int k, bool staggered)
{
    const double top = static_cast<double>(fields.w[i][j][k]);
    if (!staggered)
    {
        return top;
    }
    if (k == 0)
    {
        return 0.5 * top;
    }
    return 0.5 * (static_cast<double>(fields.w[i][j][k - 1]) + top);
}

} // namespace

static MomentumBudget compute_momentum_budget_impl(
    const SimulationStateView& state,
    const HorizontalMomentumBudgetExclusion* exclusion, bool force_deterministic = false)
{
    MomentumBudget budget{};
    const auto&    grid   = state.grid;
    const auto&    fields = state.fields;
    const bool all_required_empty =
        fields.rho.empty() && fields.u.empty()
        && fields.v.empty() && fields.w.empty();
    if (all_required_empty)
    {
        return budget;
    }
    if (!momentum_fields_ready(fields, grid))
    {
        throw std::invalid_argument(
            "momentum budget requires rho/u/v/w to match the grid");
    }
    tmv::physical_grid_metrics::validate(state);

    double momentum_x         = 0.0;
    double momentum_y         = 0.0;
    double momentum_z         = 0.0;
    double angular_momentum_z = 0.0;
    double total_mass         = 0.0;

    const PhysicalCellDomain domain = physical_scalar_domain(grid);
    if (exclusion != nullptr &&
        (exclusion->i_begin < 0 || exclusion->i_end > grid.nr ||
         exclusion->j_begin < 0 || exclusion->j_end > grid.nth ||
         exclusion->i_begin >= exclusion->i_end ||
         exclusion->j_begin >= exclusion->j_end))
        throw std::invalid_argument(
            "momentum budget horizontal exclusion is invalid");
    const auto excluded = [&](int i, int j) noexcept
    {
        return exclusion != nullptr && i >= exclusion->i_begin &&
               i < exclusion->i_end && j >= exclusion->j_begin &&
               j < exclusion->j_end;
    };

    // Linear momentum is the Cartesian (frame-invariant) inventory; the axial
    // angular momentum L_z is taken about the domain-center vertical axis, so
    // each column carries the (x_rel, y_rel) offset from that center. Cells are
    // weighted by the uniform Cartesian cell volume (SCIENCE.md C4).
    const double x_mid = 0.5 * static_cast<double>(domain.i_begin + domain.i_end) * grid.dr;
    const double y_mid = 0.5 * static_cast<double>(domain.j_begin + domain.j_end) * grid.dtheta;

    if (force_deterministic || tmv::diagnostics::deterministic_reductions_enabled())
    {
        const auto sums = tmv::diagnostics::fixed_slab_reduce<6>(
            domain.i_begin,
            domain.i_end,
            [&](int i, std::array<double, 6>& partial)
            {
                const double x_coord     = static_cast<std::size_t>(i) < grid.geometry.r.size()
                                               ? grid.geometry.r[static_cast<std::size_t>(i)]
                                               : tmv::grid_coordinates::scalar_center(i, grid.dr);
                const double x_rel       = x_coord - x_mid;
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                {
                    if (excluded(i, j)) continue;
                    const double y_coord = static_cast<std::size_t>(j) < grid.geometry.theta.size()
                                               ? grid.geometry.theta[static_cast<std::size_t>(j)]
                                               : tmv::grid_coordinates::scalar_center(j, grid.dtheta);
                    const double y_rel   = y_coord - y_mid;
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        const double cell_volume =
                            tmv::physical_grid_metrics::cell_volume(
                                state, i, j, k);
                        const double rho_val = static_cast<double>(fields.rho[i][j][k]);
                        const double u_val   = u_at_scalar_center(fields, i, j, k, grid.geometry.staggered);
                        const double v_val   = v_at_scalar_center(fields, i, j, k, grid.geometry.staggered);
                        const double w_val   = w_at_scalar_center(fields, i, j, k, grid.geometry.staggered);
                        if (!std::isfinite(cell_volume) || cell_volume <= 0.0
                            || !std::isfinite(x_rel) || !std::isfinite(y_rel)
                            || !std::isfinite(rho_val) || rho_val <= 0.0
                            || !std::isfinite(u_val) || !std::isfinite(v_val)
                            || !std::isfinite(w_val))
                        {
                            partial[5] += 1.0;
                            continue;
                        }
                        const double rho_dv = rho_val * cell_volume;
                        partial[4] += rho_dv;
                        partial[0] += rho_dv * u_val;
                        partial[1] += rho_dv * v_val;
                        partial[2] += rho_dv * w_val;
                        partial[3] += rho_dv * (x_rel * v_val - y_rel * u_val);
                    }
                }
            });

        if (sums[5] != 0.0)
        {
            throw std::runtime_error(
                "momentum budget rejected non-finite or nonphysical input cells");
        }
        budget.momentum_x           = sums[0];
        budget.momentum_y           = sums[1];
        budget.momentum_z           = sums[2];
        budget.angular_momentum_z   = sums[3];
        budget.horizontal_magnitude = std::sqrt(sums[0] * sums[0] + sums[1] * sums[1]);
        budget.total_magnitude      = std::sqrt(sums[0] * sums[0] + sums[1] * sums[1] + sums[2] * sums[2]);
        const double r_domain       = static_cast<double>(domain.i_end - domain.i_begin) * grid.dr;
        budget.total_mass           = sums[4];
        budget.reference_linear     = sums[4] * MOMENTUM_REF_VELOCITY_MS;
        budget.reference_angular    = sums[4] * MOMENTUM_REF_VELOCITY_MS * r_domain;
        return budget;
    }

    long long invalid_cells = 0;
#pragma omp parallel for reduction(+ : momentum_x, momentum_y, momentum_z, angular_momentum_z, total_mass, invalid_cells)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        const double x_coord     = static_cast<std::size_t>(i) < grid.geometry.r.size()
                                       ? grid.geometry.r[static_cast<std::size_t>(i)]
                                       : tmv::grid_coordinates::scalar_center(i, grid.dr);
        const double x_rel       = x_coord - x_mid;
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            if (excluded(i, j)) continue;
            const double y_coord = static_cast<std::size_t>(j) < grid.geometry.theta.size()
                                       ? grid.geometry.theta[static_cast<std::size_t>(j)]
                                       : tmv::grid_coordinates::scalar_center(j, grid.dtheta);
            const double y_rel   = y_coord - y_mid;
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double cell_volume =
                    tmv::physical_grid_metrics::cell_volume(state, i, j, k);
                const double rho_val = static_cast<double>(fields.rho[i][j][k]);
                // Scalars live at cell centers. Reconstruct C-grid face
                // velocities to the same location before forming rho*u. The
                // first physical center averages its maintained boundary face
                // with the next face, without allocating scratch fields.
                const double u_val = u_at_scalar_center(fields, i, j, k, grid.geometry.staggered);
                const double v_val = v_at_scalar_center(fields, i, j, k, grid.geometry.staggered);
                const double w_val = w_at_scalar_center(fields, i, j, k, grid.geometry.staggered);
                if (!std::isfinite(cell_volume) || cell_volume <= 0.0
                    || !std::isfinite(x_rel) || !std::isfinite(y_rel)
                    || !std::isfinite(rho_val) || rho_val <= 0.0
                    || !std::isfinite(u_val) || !std::isfinite(v_val)
                    || !std::isfinite(w_val))
                {
                    ++invalid_cells;
                    continue;
                }
                const double rho_dv = rho_val * cell_volume;
                total_mass += rho_dv;

                momentum_x += rho_dv * u_val;
                momentum_y += rho_dv * v_val;
                momentum_z += rho_dv * w_val;
                // L_z about the domain-center vertical axis.
                angular_momentum_z += rho_dv * (x_rel * v_val - y_rel * u_val);
            }
        }
    }

    if (invalid_cells != 0)
    {
        throw std::runtime_error(
            "momentum budget rejected non-finite or nonphysical input cells");
    }
    budget.momentum_x           = momentum_x;
    budget.momentum_y           = momentum_y;
    budget.momentum_z           = momentum_z;
    budget.angular_momentum_z   = angular_momentum_z;
    budget.horizontal_magnitude = std::sqrt(momentum_x * momentum_x + momentum_y * momentum_y);
    budget.total_magnitude = std::sqrt(momentum_x * momentum_x + momentum_y * momentum_y + momentum_z * momentum_z);

    // Characteristic momentum scales: domain mass times a reference velocity
    // (and lever arm for angular). Used to keep the drift well-conditioned when
    // the net momentum is near zero. r_domain is the physical x-extent; the
    // C-grid's two lateral scalar ghost columns are not part of that extent.
    const double r_domain    = static_cast<double>(domain.i_end - domain.i_begin) * grid.dr;
    budget.total_mass        = total_mass;
    budget.reference_linear  = total_mass * MOMENTUM_REF_VELOCITY_MS;
    budget.reference_angular = total_mass * MOMENTUM_REF_VELOCITY_MS * r_domain;
    return budget;
}

MomentumBudget compute_momentum_budget(const SimulationStateView& state)
{
    return compute_momentum_budget_impl(state, nullptr);
}

MomentumBudget compute_physical_momentum_budget_v2(const SimulationStateView& state)
{
    auto budget=compute_momentum_budget_impl(state,nullptr,true);
    const auto* terrain=tmv::physical_grid_metrics::active_terrain_metrics(state);
    if (state.fields.rho.empty() || terrain==nullptr || !state.grid.geometry.staggered)
        return budget;
    const auto& grid=state.grid;
    const auto shape=[&](const tmv::TerrainField3D& field) {
        return field.nr==grid.nr && field.nth==grid.nth && field.nz==grid.nz+1
            && field.values.size()==static_cast<std::size_t>(grid.nr)*grid.nth*(grid.nz+1);
    };
    if (grid.nr<3 || grid.nth<3 || !shape(terrain->Gx_face) || !shape(terrain->Gy_face))
        throw std::invalid_argument("physical C-grid momentum requires complete ground-face geometry");
    const auto surface=tmv::physical_grid_metrics::surface_velocity_ms(state);
    const auto domain=physical_scalar_domain(grid);
    const auto& fields=state.fields;
    const auto sums=tmv::diagnostics::fixed_slab_reduce<2>(domain.i_begin,domain.i_end,
        [&](int i,std::array<double,2>& partial) {
            for (int j=domain.j_begin;j<domain.j_end;++j)
            {
                const double ground=-(0.5*(static_cast<double>(fields.u(i-1,j,0))+fields.u(i,j,0))-surface.u_ms)
                        *terrain->Gx_face(i,j,0)
                    -(0.5*(static_cast<double>(fields.v(i,j-1,0))+fields.v(i,j,0))-surface.v_ms)
                        *terrain->Gy_face(i,j,0);
                const double impulse=0.5*fields.rho(i,j,0)
                    *tmv::physical_grid_metrics::cell_volume(state,i,j,0)*ground;
                if (!std::isfinite(impulse)) partial[1]+=1.0;
                else partial[0]+=impulse;
            }
        });
    budget.momentum_z+=sums[0];
    budget.total_magnitude=std::hypot(budget.momentum_x,budget.momentum_y,budget.momentum_z);
    if (sums[1]!=0.0 || !std::isfinite(budget.total_magnitude))
        throw std::runtime_error("physical momentum budget rejected invalid ground momentum");
    return budget;
}

MomentumBudget compute_momentum_budget_excluding_horizontal_rectangle(
    const SimulationStateView& state,
    const HorizontalMomentumBudgetExclusion& exclusion)
{
    return compute_momentum_budget_impl(state, &exclusion);
}

MomentumBudget compute_momentum_budget()
{
    return compute_momentum_budget(global_simulation_state());
}

MomentumForcing compute_momentum_forcing(const SimulationStateView& state)
{
    MomentumForcing forcing{};
    const auto&     grid   = state.grid;
    const auto&     fields = state.fields;
    const auto&     base   = state.base;
    const bool all_required_empty =
        fields.rho.empty() && fields.u.empty()
        && fields.v.empty() && fields.w.empty();
    if (all_required_empty)
    {
        return forcing;
    }
    if (!momentum_fields_ready(fields, grid))
    {
        throw std::invalid_argument(
            "momentum forcing requires rho/u/v/w to match the grid");
    }
    if (!std::isfinite(state.coriolis_f))
    {
        throw std::invalid_argument(
            "momentum forcing requires a finite Coriolis parameter");
    }
    tmv::physical_grid_metrics::validate(state);

    if (!base.rho0.empty()
        && static_cast<int>(base.rho0.size()) != grid.nz)
    {
        throw std::invalid_argument(
            "momentum forcing base-density profile must match grid.nz");
    }
    const bool                have_base = static_cast<int>(base.rho0.size()) == grid.nz;
    const PhysicalCellDomain domain = physical_scalar_domain(grid);

    double weight_z   = 0.0;
    double buoyancy_z = 0.0;

    if (tmv::diagnostics::deterministic_reductions_enabled())
    {
        const auto sums = tmv::diagnostics::fixed_slab_reduce<3>(
            domain.i_begin,
            domain.i_end,
            [&](int i, std::array<double, 3>& partial)
            {
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                {
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        const double cell_volume =
                            tmv::physical_grid_metrics::cell_volume(
                                state, i, j, k);
                        const double rho_val = static_cast<double>(fields.rho[i][j][k]);
                        const double rho0_k = have_base
                            ? base.rho0[static_cast<std::size_t>(k)] : 0.0;
                        if (!std::isfinite(cell_volume) || cell_volume <= 0.0
                            || !std::isfinite(rho_val) || rho_val <= 0.0
                            || !std::isfinite(rho0_k))
                        {
                            partial[2] += 1.0;
                            continue;
                        }
                        partial[0] -= rho_val * g * cell_volume;
                        partial[1] -= g * (rho_val - rho0_k) * cell_volume;
                    }
                }
            });
        if (sums[2] != 0.0)
        {
            throw std::runtime_error(
                "momentum forcing rejected non-finite or nonphysical input cells");
        }
        const MomentumBudget mom = compute_momentum_budget(state);
        forcing.weight_z         = sums[0];
        forcing.buoyancy_z       = sums[1];
        forcing.coriolis_x       = state.coriolis_f * mom.momentum_y;
        forcing.coriolis_y       = -state.coriolis_f * mom.momentum_x;
        return forcing;
    }

    long long invalid_cells = 0;
#pragma omp parallel for reduction(+ : weight_z, buoyancy_z, invalid_cells)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double cell_volume =
                    tmv::physical_grid_metrics::cell_volume(state, i, j, k);
                const double rho_val = static_cast<double>(fields.rho[i][j][k]);
                const double rho0_k = have_base
                    ? base.rho0[static_cast<std::size_t>(k)] : 0.0;
                if (!std::isfinite(cell_volume) || cell_volume <= 0.0
                    || !std::isfinite(rho_val) || rho_val <= 0.0
                    || !std::isfinite(rho0_k))
                {
                    ++invalid_cells;
                    continue;
                }
                weight_z -= rho_val * g * cell_volume;
                buoyancy_z -= g * (rho_val - rho0_k) * cell_volume;
            }
        }
    }

    // Coriolis force on the horizontal linear momentum is exact from the
    // inventory: a_cor = -f k x u_h  =>  F = (f * p_y, -f * p_x).
    if (invalid_cells != 0)
    {
        throw std::runtime_error(
            "momentum forcing rejected non-finite or nonphysical input cells");
    }
    const MomentumBudget mom = compute_momentum_budget(state);
    forcing.weight_z         = weight_z;
    forcing.buoyancy_z       = buoyancy_z;
    forcing.coriolis_x       = state.coriolis_f * mom.momentum_y;
    forcing.coriolis_y       = -state.coriolis_f * mom.momentum_x;
    return forcing;
}

MomentumForcing compute_momentum_forcing()
{
    return compute_momentum_forcing(global_simulation_state());
}

MomentumDrift evaluate_momentum_drift(const MomentumBudget& before, const MomentumBudget& after)
{
    const double values[] = {
        before.momentum_x, before.momentum_y, before.momentum_z,
        before.angular_momentum_z, before.horizontal_magnitude,
        before.total_magnitude, before.total_mass, before.reference_linear,
        before.reference_angular, after.momentum_x, after.momentum_y,
        after.momentum_z, after.angular_momentum_z,
        after.horizontal_magnitude, after.total_magnitude, after.total_mass,
        after.reference_linear, after.reference_angular,
    };
    if (std::any_of(
            std::begin(values), std::end(values),
            [](double value) { return !std::isfinite(value); }))
    {
        throw std::invalid_argument(
            "momentum budget drift requires finite snapshots");
    }
    MomentumDrift drift;

    drift.d_momentum_x         = after.momentum_x - before.momentum_x;
    drift.d_momentum_y         = after.momentum_y - before.momentum_y;
    drift.d_momentum_z         = after.momentum_z - before.momentum_z;
    drift.d_angular_momentum_z = after.angular_momentum_z - before.angular_momentum_z;

    // Normalize by a characteristic momentum: the larger of the instantaneous
    // inventory and the mass-scaled reference floor (1.0 final guard against an
    // all-zero snapshot). Near-zero momentum on a quiescent domain divides by
    // the reference floor, not by ~0, so the drift stays well-conditioned.
    const double lin_scale = std::max({std::abs(before.momentum_x),
                                       std::abs(before.momentum_y),
                                       std::abs(before.momentum_z),
                                       before.reference_linear,
                                       1.0});
    const double ang_scale = std::max({std::abs(before.angular_momentum_z), before.reference_angular, 1.0});

    drift.rel_momentum_x         = std::abs(drift.d_momentum_x) / lin_scale;
    drift.rel_momentum_y         = std::abs(drift.d_momentum_y) / lin_scale;
    drift.rel_momentum_z         = std::abs(drift.d_momentum_z) / lin_scale;
    drift.rel_angular_momentum_z = std::abs(drift.d_angular_momentum_z) / ang_scale;

    drift.warn = drift.rel_momentum_x > BUDGET_THRESH_MOMENTUM || drift.rel_momentum_y > BUDGET_THRESH_MOMENTUM ||
                 drift.rel_momentum_z > BUDGET_THRESH_MOMENTUM || drift.rel_angular_momentum_z > BUDGET_THRESH_MOMENTUM;
    return drift;
}

void report_momentum_transition(const SimulationStateView& state,
                                const char*                stage,
                                const MomentumBudget&      before,
                                const MomentumBudget&      after,
                                double                     dt_stage)
{
    if (!std::isfinite(dt_stage) || dt_stage <= 0.0)
    {
        return;
    }

    const MomentumDrift drift = evaluate_momentum_drift(before, after);

    {
        std::lock_guard<std::mutex> lock(g_momentum_csv_mutex);
        if (auto* selected = g_momentum_csv_files.stream(state.budget_level, g_momentum_csv_mutex))
        {
            auto& g_momentum_csv = *selected;
            g_momentum_csv << state.time_s << ',' << stage << ',' << drift.d_momentum_x << ',' << drift.d_momentum_y
                           << ',' << drift.d_momentum_z << ',' << drift.d_angular_momentum_z << ','
                           << drift.rel_momentum_x << ',' << drift.rel_momentum_y << ',' << drift.rel_momentum_z << ','
                           << drift.rel_angular_momentum_z << ',' << (drift.warn ? 1 : 0) << '\n';
        }
    }

    if (!drift.warn && !log_debug_enabled())
    {
        return;
    }

    auto emit = [&](auto log_fn)
    {
        log_fn("[MOMENTUM BUDGET",
               drift.warn ? " WARN]" : "]",
               " stage=",
               stage,
               " d_momentum_x=",
               drift.d_momentum_x,
               " d_momentum_y=",
               drift.d_momentum_y,
               " d_momentum_z=",
               drift.d_momentum_z,
               " d_angular_momentum_z=",
               drift.d_angular_momentum_z,
               " rel_momentum_x=",
               drift.rel_momentum_x,
               " rel_momentum_y=",
               drift.rel_momentum_y,
               " rel_momentum_z=",
               drift.rel_momentum_z,
               " rel_angular_momentum_z=",
               drift.rel_angular_momentum_z);
    };

    if (drift.warn)
    {
        emit([](auto&&... args) { tmv::log_warn(args...); });
    }
    else
    {
        emit([](auto&&... args) { tmv::log_debug(args...); });
    }
}

void report_momentum_transition(const char*           stage,
                                const MomentumBudget& before,
                                const MomentumBudget& after,
                                double                dt_stage)
{
    report_momentum_transition(global_simulation_state(), stage, before, after, dt_stage);
}

void open_momentum_csv(const std::string& path)
{
    std::lock_guard<std::mutex> lock(g_momentum_csv_mutex);
    if (!g_momentum_csv_files.open(path, k_momentum_csv_header, 12))
    {
        tmv::log_warn("[MOMENTUM CSV] failed to open '", path, "' for writing; time-series output disabled");
        return;
    }
    tmv::log_info("[MOMENTUM CSV] writing budget time series to ", path);
}

void close_momentum_csv()
{
    std::lock_guard<std::mutex> lock(g_momentum_csv_mutex);
    g_momentum_csv_files.close();
}
