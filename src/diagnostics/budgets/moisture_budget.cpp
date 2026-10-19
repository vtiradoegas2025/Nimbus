#include "diagnostics/level_budget_csv.hpp"
/**
 * @file moisture_budget.cpp
 * @brief Domain-integrated moisture budget diagnostics (see header).
 */

#include "diagnostics/moisture_budget.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "core/infra/physical_cell_domain.hpp"
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

double relative_drift(double before_val, double after_val)
{
    return std::abs(after_val - before_val) / std::max(1.0, std::abs(before_val));
}

tmv::LevelBudgetCsv g_moisture_csv_files;
std::mutex    g_moisture_csv_mutex;

constexpr const char* k_moisture_csv_header = "sim_time_s,stage,"
                                              "d_vapor,d_liquid,d_ice,d_total_water,"
                                              "rel_vapor,rel_liquid,rel_ice,rel_total_water,"
                                              "warn\n";

bool field_matches_grid(const Field3D& field, const GridStateView& grid)
{
    return field.size_r() == grid.nr
        && field.size_th() == grid.nth
        && field.size_z() == grid.nz;
}

} // namespace

MoistureBudget compute_moisture_budget(const SimulationStateView& state)
{
    MoistureBudget budget{};
    const auto&    grid   = state.grid;
    const auto&    fields = state.fields;
    const bool all_required_empty =
        fields.rho.empty() && fields.qv.empty() && fields.qc.empty()
        && fields.qr.empty() && fields.qi.empty() && fields.qs.empty()
        && fields.qg.empty() && fields.qh.empty();
    if (all_required_empty)
    {
        return budget;
    }
    if (!field_matches_grid(fields.rho, grid)
        || !field_matches_grid(fields.qv, grid)
        || !field_matches_grid(fields.qc, grid)
        || !field_matches_grid(fields.qr, grid)
        || !field_matches_grid(fields.qi, grid)
        || !field_matches_grid(fields.qs, grid)
        || !field_matches_grid(fields.qg, grid)
        || !field_matches_grid(fields.qh, grid))
    {
        throw std::invalid_argument(
            "moisture budget requires every input field to match the grid");
    }
    tmv::physical_grid_metrics::validate(state);

    double                    vapor  = 0.0;
    double                    liquid = 0.0;
    double                    ice    = 0.0;
    const PhysicalCellDomain domain = physical_scalar_domain(grid);

    if (tmv::diagnostics::deterministic_reductions_enabled())
    {
        const auto sums = tmv::diagnostics::fixed_slab_reduce<4>(
            domain.i_begin,
            domain.i_end,
            [&](int i, std::array<double, 4>& partial)
            {
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                {
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        const double cell_volume =
                            tmv::physical_grid_metrics::cell_volume(
                                state, i, j, k);
                        const double rho_dv = static_cast<double>(fields.rho[i][j][k]) * cell_volume;
                        const double qv_val = static_cast<double>(fields.qv[i][j][k]);
                        const double qc_val = static_cast<double>(fields.qc[i][j][k]);
                        const double qr_val = static_cast<double>(fields.qr[i][j][k]);
                        const double qi_val = static_cast<double>(fields.qi[i][j][k]);
                        const double qs_val = static_cast<double>(fields.qs[i][j][k]);
                        const double qg_val = static_cast<double>(fields.qg[i][j][k]);
                        const double qh_val = static_cast<double>(fields.qh[i][j][k]);
                        if (!std::isfinite(cell_volume) || cell_volume <= 0.0
                            || !std::isfinite(rho_dv) || rho_dv <= 0.0
                            || !std::isfinite(qv_val)
                            || !std::isfinite(qc_val)
                            || !std::isfinite(qr_val)
                            || !std::isfinite(qi_val)
                            || !std::isfinite(qs_val)
                            || !std::isfinite(qg_val)
                            || !std::isfinite(qh_val))
                        {
                            partial[3] += 1.0;
                            continue;
                        }

                        partial[0] += rho_dv * qv_val;
                        partial[1] += rho_dv * (qc_val + qr_val);
                        partial[2] += rho_dv * (qi_val + qs_val + qg_val + qh_val);
                    }
                }
            });
        if (sums[3] != 0.0)
        {
            throw std::runtime_error(
                "moisture budget rejected non-finite or nonphysical input cells");
        }
        budget.vapor       = sums[0];
        budget.liquid      = sums[1];
        budget.ice         = sums[2];
        budget.total_water = sums[0] + sums[1] + sums[2];
        return budget;
    }

    long long invalid_cells = 0;
#pragma omp parallel for reduction(+ : vapor, liquid, ice, invalid_cells)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double cell_volume =
                    tmv::physical_grid_metrics::cell_volume(state, i, j, k);
                const double rho_dv = static_cast<double>(fields.rho[i][j][k]) * cell_volume;

                const double qv_val = static_cast<double>(fields.qv[i][j][k]);
                const double qc_val = static_cast<double>(fields.qc[i][j][k]);
                const double qr_val = static_cast<double>(fields.qr[i][j][k]);
                const double qi_val = static_cast<double>(fields.qi[i][j][k]);
                const double qs_val = static_cast<double>(fields.qs[i][j][k]);
                const double qg_val = static_cast<double>(fields.qg[i][j][k]);
                const double qh_val = static_cast<double>(fields.qh[i][j][k]);
                if (!std::isfinite(cell_volume) || cell_volume <= 0.0
                    || !std::isfinite(rho_dv) || rho_dv <= 0.0
                    || !std::isfinite(qv_val)
                    || !std::isfinite(qc_val)
                    || !std::isfinite(qr_val)
                    || !std::isfinite(qi_val)
                    || !std::isfinite(qs_val)
                    || !std::isfinite(qg_val)
                    || !std::isfinite(qh_val))
                {
                    ++invalid_cells;
                    continue;
                }

                vapor += rho_dv * qv_val;
                liquid += rho_dv * (qc_val + qr_val);
                ice += rho_dv * (qi_val + qs_val + qg_val + qh_val);
            }
        }
    }

    if (invalid_cells != 0)
    {
        throw std::runtime_error(
            "moisture budget rejected non-finite or nonphysical input cells");
    }
    budget.vapor       = vapor;
    budget.liquid      = liquid;
    budget.ice         = ice;
    budget.total_water = vapor + liquid + ice;
    return budget;
}

MoistureBudget compute_moisture_budget()
{
    return compute_moisture_budget(global_simulation_state());
}

MoistureDrift evaluate_moisture_drift(const MoistureBudget& before, const MoistureBudget& after)
{
    const double values[] = {
        before.vapor, before.liquid, before.ice, before.total_water,
        after.vapor, after.liquid, after.ice, after.total_water,
    };
    if (std::any_of(
            std::begin(values), std::end(values),
            [](double value) { return !std::isfinite(value); }))
    {
        throw std::invalid_argument(
            "moisture budget drift requires finite snapshots");
    }
    MoistureDrift drift;

    drift.d_vapor       = after.vapor - before.vapor;
    drift.d_liquid      = after.liquid - before.liquid;
    drift.d_ice         = after.ice - before.ice;
    drift.d_total_water = after.total_water - before.total_water;

    drift.rel_vapor       = relative_drift(before.vapor, after.vapor);
    drift.rel_liquid      = relative_drift(before.liquid, after.liquid);
    drift.rel_ice         = relative_drift(before.ice, after.ice);
    drift.rel_total_water = relative_drift(before.total_water, after.total_water);

    // Only total water must close; phases exchange mass via microphysics.
    drift.warn = drift.rel_total_water > BUDGET_THRESH_MOISTURE;
    return drift;
}

void report_moisture_transition(const SimulationStateView& state,
                                const char*                stage,
                                const MoistureBudget&      before,
                                const MoistureBudget&      after,
                                double                     dt_stage)
{
    if (!std::isfinite(dt_stage) || dt_stage <= 0.0)
    {
        return;
    }

    const MoistureDrift drift = evaluate_moisture_drift(before, after);

    {
        std::lock_guard<std::mutex> lock(g_moisture_csv_mutex);
        if (auto* selected = g_moisture_csv_files.stream(state.budget_level, g_moisture_csv_mutex))
        {
            auto& g_moisture_csv = *selected;
            g_moisture_csv << state.time_s << ',' << stage << ',' << drift.d_vapor << ',' << drift.d_liquid << ','
                           << drift.d_ice << ',' << drift.d_total_water << ',' << drift.rel_vapor << ','
                           << drift.rel_liquid << ',' << drift.rel_ice << ',' << drift.rel_total_water << ','
                           << (drift.warn ? 1 : 0) << '\n';
        }
    }

    if (!drift.warn && !log_debug_enabled())
    {
        return;
    }

    auto emit = [&](auto log_fn)
    {
        log_fn("[MOISTURE BUDGET",
               drift.warn ? " WARN]" : "]",
               " stage=",
               stage,
               " d_vapor=",
               drift.d_vapor,
               " d_liquid=",
               drift.d_liquid,
               " d_ice=",
               drift.d_ice,
               " d_total_water=",
               drift.d_total_water,
               " rel_total_water=",
               drift.rel_total_water);
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

void report_moisture_transition(const char*           stage,
                                const MoistureBudget& before,
                                const MoistureBudget& after,
                                double                dt_stage)
{
    report_moisture_transition(global_simulation_state(), stage, before, after, dt_stage);
}

void open_moisture_csv(const std::string& path)
{
    std::lock_guard<std::mutex> lock(g_moisture_csv_mutex);
    if (!g_moisture_csv_files.open(path, k_moisture_csv_header, 12))
    {
        tmv::log_warn("[MOISTURE CSV] failed to open '", path, "' for writing; time-series output disabled");
        return;
    }
    tmv::log_info("[MOISTURE CSV] writing budget time series to ", path);
}

void close_moisture_csv()
{
    std::lock_guard<std::mutex> lock(g_moisture_csv_mutex);
    g_moisture_csv_files.close();
}
