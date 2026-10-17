#include "diagnostics/level_budget_csv.hpp"
/**
 * @file conservation_budget.cpp
 * @brief Domain-integrated mass/water/energy budget diagnostics.
 *
 * See include/diagnostics/conservation_budget.hpp for the split between
 * transport-conserved (total_mass, density-weighted theta/water) and derived
 * (dry_mass compatibility alias, water, moist-total, thermal-proxy)
 * inventories. The dry-density/dry-air-mixing-ratio basis is declared in the
 * shared runtime state contract.
 */

#include "diagnostics/conservation_budget.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "diagnostics/moisture_budget.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/simulation_state.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

/**
 * @brief Compute |after - before| / max(1, |before|) without dividing by zero.
 */
double relative_drift(double before_val, double after_val)
{
    return std::abs(after_val - before_val) / std::max(1.0, std::abs(before_val));
}

// CSV time-series output. The stream is held open for the lifetime of
// the simulation so we don't pay the open/close cost on every stage.
// Guarded by a mutex because report_budget_transition can in principle
// be called from any thread (the dynamics orchestrator is single-thread
// today, but the diagnostic API doesn't promise that).
tmv::LevelBudgetCsv g_budget_csv_files;
std::mutex    g_budget_csv_mutex;

constexpr const char* k_budget_csv_header = "sim_time_s,stage,"
                                            "d_total_mass,d_theta_volume,d_q_volume,"
                                            "d_dry_mass,d_total_water,d_thermal_energy,"
                                            "rel_total_mass,rel_theta_volume,rel_q_volume,"
                                            "rel_dry_mass,rel_total_water,rel_thermal_energy,"
                                            "warn_kernel,warn_physical,"
                                            "d_moist_total_mass,rel_moist_total_mass,"
                                            "d_qv_mass,d_qc_mass,d_qr_mass,d_qi_mass,"
                                            "d_qs_mass,d_qg_mass,d_qh_mass\n";

bool field_matches_grid(const Field3D& field, const GridStateView& grid)
{
    return field.size_r() == grid.nr
        && field.size_th() == grid.nth
        && field.size_z() == grid.nz;
}

} // namespace

static ConservationBudget compute_conservation_budget_impl(
    const SimulationStateView& state,
    const HorizontalBudgetExclusion* exclusion,
    MoistureBudget* moisture)
{
    ConservationBudget budget{};
    if (moisture != nullptr)
        *moisture = {};
    const auto&        grid   = state.grid;
    const auto&        fields = state.fields;
    const bool all_required_empty =
        fields.rho.empty() && fields.p.empty() && fields.theta.empty()
        && fields.qv.empty() && fields.qc.empty() && fields.qr.empty()
        && fields.qi.empty() && fields.qs.empty() && fields.qg.empty()
        && fields.qh.empty();
    if (all_required_empty)
    {
        return budget;
    }
    if (!field_matches_grid(fields.rho, grid)
        || !field_matches_grid(fields.p, grid)
        || !field_matches_grid(fields.theta, grid)
        || !field_matches_grid(fields.qv, grid)
        || !field_matches_grid(fields.qc, grid)
        || !field_matches_grid(fields.qr, grid)
        || !field_matches_grid(fields.qi, grid)
        || !field_matches_grid(fields.qs, grid)
        || !field_matches_grid(fields.qg, grid)
        || !field_matches_grid(fields.qh, grid))
    {
        throw std::invalid_argument(
            "conservation budget requires every input field to match the grid");
    }

    tmv::physical_grid_metrics::validate(state);
    constexpr double          kappa  = R_d / cp;
    const PhysicalCellDomain domain = physical_scalar_domain(grid);
    if (exclusion != nullptr &&
        (exclusion->i_begin < 0 || exclusion->i_end > grid.nr ||
         exclusion->j_begin < 0 || exclusion->j_end > grid.nth ||
         exclusion->i_begin >= exclusion->i_end ||
         exclusion->j_begin >= exclusion->j_end))
        throw std::invalid_argument(
            "conservation budget horizontal exclusion is invalid");
    const auto excluded = [&](int i, int j) noexcept
    {
        return exclusion != nullptr && i >= exclusion->i_begin &&
               i < exclusion->i_end && j >= exclusion->j_begin &&
               j < exclusion->j_end;
    };

    // Transport-conserved integrals.
    double total_mass     = 0.0;
    double theta_volume   = 0.0;
    double q_volume_total = 0.0;

    // Derived inventories under the declared rho_d + dry-air-mixing-ratio
    // contract. dry_mass is retained as a compatibility alias for total_mass.
    double total_water    = 0.0;
    double thermal_energy = 0.0;
    double qv_mass = 0.0, qc_mass = 0.0, qr_mass = 0.0, qi_mass = 0.0;
    double qs_mass = 0.0, qg_mass = 0.0, qh_mass = 0.0;
    double moisture_vapor = 0.0;
    double moisture_liquid = 0.0;
    double moisture_ice = 0.0;

    if (tmv::diagnostics::deterministic_reductions_enabled())
    {
        // Standalone conservation retains its original 13-slot reduction.
        // The fused endpoint extends the same slab partial with three
        // moisture slots without changing slots 0..12 or their merge order.
        const auto accumulate_slab = [&](int i, auto& partial)
        {
            constexpr bool include_moisture =
                std::tuple_size<
                    std::decay_t<decltype(partial)>>::value > 13;
            for (int j = domain.j_begin; j < domain.j_end; ++j)
            {
                if (excluded(i, j)) continue;
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const double cell_volume =
                        tmv::physical_grid_metrics::cell_volume(
                            state, i, j, k);
                    const double rho_val   = static_cast<double>(fields.rho[i][j][k]);
                    const double qv_val    = static_cast<double>(fields.qv[i][j][k]);
                    const double qc_val    = static_cast<double>(fields.qc[i][j][k]);
                    const double qr_val    = static_cast<double>(fields.qr[i][j][k]);
                    const double qi_val    = static_cast<double>(fields.qi[i][j][k]);
                    const double qs_val    = static_cast<double>(fields.qs[i][j][k]);
                    const double qg_val    = static_cast<double>(fields.qg[i][j][k]);
                    const double qh_val    = static_cast<double>(fields.qh[i][j][k]);
                    const double water     = qv_val + qc_val + qr_val + qi_val + qs_val + qg_val + qh_val;
                    const double theta_val = static_cast<double>(fields.theta[i][j][k]);
                    const double p_val     = static_cast<double>(fields.p[i][j][k]);
                    const double rho_dv    = rho_val * cell_volume;
                    if (!std::isfinite(cell_volume) || cell_volume <= 0.0
                        || !std::isfinite(rho_val) || rho_val <= 0.0
                        || !std::isfinite(theta_val) || theta_val <= 0.0
                        || !std::isfinite(p_val) || p_val <= 0.0
                        || !std::isfinite(qv_val)
                        || !std::isfinite(qc_val)
                        || !std::isfinite(qr_val)
                        || !std::isfinite(qi_val)
                        || !std::isfinite(qs_val)
                        || !std::isfinite(qg_val)
                        || !std::isfinite(qh_val)
                        || !std::isfinite(water)
                        || !std::isfinite(rho_dv))
                    {
                        partial[12] += 1.0;
                        continue;
                    }

                    partial[0] += rho_dv;
                    partial[1] += rho_dv * theta_val;
                    partial[2] += rho_dv * water;
                    partial[3] += rho_dv * water;
                    partial[5] += rho_dv * qv_val;
                    partial[6] += rho_dv * qc_val;
                    partial[7] += rho_dv * qr_val;
                    partial[8] += rho_dv * qi_val;
                    partial[9] += rho_dv * qs_val;
                    partial[10] += rho_dv * qg_val;
                    partial[11] += rho_dv * qh_val;
                    if constexpr (include_moisture)
                    {
                        partial[13] += rho_dv * qv_val;
                        partial[14] += rho_dv * (qc_val + qr_val);
                        partial[15] += rho_dv
                            * (qi_val + qs_val + qg_val + qh_val);
                    }

                    const double temperature =
                        theta_val * std::pow(p_val / p0, kappa);
                    if (!std::isfinite(temperature) || temperature <= 0.0)
                    {
                        partial[12] += 1.0;
                        continue;
                    }
                    partial[4] += rho_dv * cp * temperature;
                }
            }
        };
        const auto commit_conservation = [&](const auto& sums)
        {
            if (sums[12] != 0.0)
            {
                throw std::runtime_error(
                    "conservation budget rejected non-finite or nonphysical input cells");
            }
            budget.total_mass       = sums[0];
            budget.theta_volume     = sums[1];
            budget.q_volume_total   = sums[2];
            budget.dry_mass         = sums[0];
            budget.total_water      = sums[3];
            for (std::size_t species = 0; species < 7; ++species)
                budget.water_species_mass[species] = sums[5 + species];
            budget.moist_total_mass = sums[0] + sums[3];
            budget.thermal_energy   = sums[4];
        };

        if (moisture == nullptr)
        {
            const auto sums = tmv::diagnostics::fixed_slab_reduce<13>(
                domain.i_begin, domain.i_end, accumulate_slab);
            commit_conservation(sums);
            return budget;
        }

        const auto sums = tmv::diagnostics::fixed_slab_reduce<16>(
            domain.i_begin, domain.i_end, accumulate_slab);
        commit_conservation(sums);
        moisture->vapor = sums[13];
        moisture->liquid = sums[14];
        moisture->ice = sums[15];
        moisture->total_water = sums[13] + sums[14] + sums[15];
        return budget;
    }

    long long invalid_cells = 0;
#pragma omp parallel for reduction(+ : total_mass, theta_volume, q_volume_total, total_water, thermal_energy, qv_mass, qc_mass, qr_mass, qi_mass, qs_mass, qg_mass, qh_mass, moisture_vapor, moisture_liquid, moisture_ice, invalid_cells) \
    collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            if (excluded(i, j)) continue;
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double cell_volume =
                    tmv::physical_grid_metrics::cell_volume(state, i, j, k);
                const double rho_val   = static_cast<double>(fields.rho[i][j][k]);
                const double qv_val    = static_cast<double>(fields.qv[i][j][k]);
                const double qc_val    = static_cast<double>(fields.qc[i][j][k]);
                const double qr_val    = static_cast<double>(fields.qr[i][j][k]);
                const double qi_val    = static_cast<double>(fields.qi[i][j][k]);
                const double qs_val    = static_cast<double>(fields.qs[i][j][k]);
                const double qg_val    = static_cast<double>(fields.qg[i][j][k]);
                const double qh_val    = static_cast<double>(fields.qh[i][j][k]);
                const double water     = qv_val + qc_val + qr_val + qi_val + qs_val + qg_val + qh_val;
                const double theta_val = static_cast<double>(fields.theta[i][j][k]);
                const double p_val     = static_cast<double>(fields.p[i][j][k]);

                // Transport-conserved inventories. The production C-grid
                // scalar kernel advances rho*phi in flux form and recovers
                // phi with the post-continuity density, so theta and water
                // must be density weighted here as well.
                const double rho_dv = rho_val * cell_volume;
                if (!std::isfinite(cell_volume) || cell_volume <= 0.0
                    || !std::isfinite(rho_val) || rho_val <= 0.0
                    || !std::isfinite(theta_val) || theta_val <= 0.0
                    || !std::isfinite(p_val) || p_val <= 0.0
                    || !std::isfinite(qv_val)
                    || !std::isfinite(qc_val)
                    || !std::isfinite(qr_val)
                    || !std::isfinite(qi_val)
                    || !std::isfinite(qs_val)
                    || !std::isfinite(qg_val)
                    || !std::isfinite(qh_val)
                    || !std::isfinite(water)
                    || !std::isfinite(rho_dv))
                {
                    ++invalid_cells;
                    continue;
                }
                total_mass += rho_dv;
                theta_volume += rho_dv * theta_val;
                q_volume_total += rho_dv * water;

                // Water mixing ratios are per unit dry air, so rho_d*q is
                // water mass density. Never apply the mass-fraction identity
                // rho*(1-q): rho is already dry-air density.
                total_water += rho_dv * water;
                qv_mass += rho_dv * qv_val;
                qc_mass += rho_dv * qc_val;
                qr_mass += rho_dv * qr_val;
                qi_mass += rho_dv * qi_val;
                qs_mass += rho_dv * qs_val;
                qg_mass += rho_dv * qg_val;
                qh_mass += rho_dv * qh_val;
                if (moisture != nullptr)
                {
                    moisture_vapor += rho_dv * qv_val;
                    moisture_liquid += rho_dv * (qc_val + qr_val);
                    moisture_ice += rho_dv
                        * (qi_val + qs_val + qg_val + qh_val);
                }

                const double temperature =
                    theta_val * std::pow(p_val / p0, kappa);
                if (!std::isfinite(temperature) || temperature <= 0.0)
                {
                    ++invalid_cells;
                    continue;
                }
                thermal_energy += rho_dv * cp * temperature;
            }
        }
    }

    if (invalid_cells != 0)
    {
        throw std::runtime_error(
            "conservation budget rejected non-finite or nonphysical input cells");
    }
    budget.total_mass       = total_mass;
    budget.theta_volume     = theta_volume;
    budget.q_volume_total   = q_volume_total;
    budget.dry_mass         = total_mass;
    budget.total_water      = total_water;
    budget.water_species_mass = {
        qv_mass, qc_mass, qr_mass, qi_mass, qs_mass, qg_mass, qh_mass};
    budget.moist_total_mass = total_mass + total_water;
    budget.thermal_energy   = thermal_energy;
    if (moisture != nullptr)
    {
        moisture->vapor = moisture_vapor;
        moisture->liquid = moisture_liquid;
        moisture->ice = moisture_ice;
        moisture->total_water =
            moisture_vapor + moisture_liquid + moisture_ice;
    }
    return budget;
}

ConservationBudget compute_conservation_budget(const SimulationStateView& state)
{
    return compute_conservation_budget_impl(state, nullptr, nullptr);
}

void compute_conservation_and_moisture_budgets(
    const SimulationStateView& state,
    ConservationBudget& conservation,
    MoistureBudget& moisture)
{
    MoistureBudget next_moisture;
    ConservationBudget next_conservation =
        compute_conservation_budget_impl(
            state, nullptr, &next_moisture);
    conservation = next_conservation;
    moisture = next_moisture;
}

ConservationBudget compute_conservation_budget_excluding_horizontal_rectangle(
    const SimulationStateView& state,
    const HorizontalBudgetExclusion& exclusion)
{
    return compute_conservation_budget_impl(state, &exclusion, nullptr);
}

ConservationBudget compute_conservation_budget()
{
    return compute_conservation_budget(global_simulation_state());
}

BudgetDrift evaluate_budget_drift(const ConservationBudget& before, const ConservationBudget& after)
{
    const auto require_finite = [](const ConservationBudget& budget)
    {
        const double scalars[] = {
            budget.total_mass, budget.theta_volume, budget.q_volume_total,
            budget.dry_mass, budget.total_water, budget.moist_total_mass,
            budget.thermal_energy,
        };
        if (std::any_of(
                std::begin(scalars), std::end(scalars),
                [](double value) { return !std::isfinite(value); })
            || std::any_of(
                budget.water_species_mass.begin(),
                budget.water_species_mass.end(),
                [](double value) { return !std::isfinite(value); }))
        {
            throw std::invalid_argument(
                "conservation budget drift requires finite snapshots");
        }
    };
    require_finite(before);
    require_finite(after);
    BudgetDrift drift;

    drift.d_total_mass       = after.total_mass - before.total_mass;
    drift.d_theta_volume     = after.theta_volume - before.theta_volume;
    drift.d_q_volume         = after.q_volume_total - before.q_volume_total;
    drift.d_dry_mass         = after.dry_mass - before.dry_mass;
    drift.d_total_water      = after.total_water - before.total_water;
    for (std::size_t species = 0; species < drift.d_water_species.size(); ++species)
        drift.d_water_species[species] =
            after.water_species_mass[species] - before.water_species_mass[species];
    drift.d_moist_total_mass = after.moist_total_mass - before.moist_total_mass;
    drift.d_thermal_energy   = after.thermal_energy - before.thermal_energy;

    drift.rel_total_mass       = relative_drift(before.total_mass, after.total_mass);
    drift.rel_theta_volume     = relative_drift(before.theta_volume, after.theta_volume);
    drift.rel_q_volume         = relative_drift(before.q_volume_total, after.q_volume_total);
    drift.rel_dry_mass         = relative_drift(before.dry_mass, after.dry_mass);
    drift.rel_total_water      = relative_drift(before.total_water, after.total_water);
    drift.rel_moist_total_mass = relative_drift(before.moist_total_mass, after.moist_total_mass);
    drift.rel_thermal_energy   = relative_drift(before.thermal_energy, after.thermal_energy);

    drift.warn_kernel = drift.rel_total_mass > BUDGET_THRESH_KERNEL || drift.rel_theta_volume > BUDGET_THRESH_KERNEL ||
                        drift.rel_q_volume > BUDGET_THRESH_KERNEL;
    drift.warn_physical = drift.rel_dry_mass > BUDGET_THRESH_DRY || drift.rel_total_water > BUDGET_THRESH_WATER ||
                          drift.rel_moist_total_mass > BUDGET_THRESH_WATER ||
                          drift.rel_thermal_energy > BUDGET_THRESH_ENERGY;

    return drift;
}

void report_budget_transition(const SimulationStateView& state,
                              const char*                stage,
                              const ConservationBudget&  before,
                              const ConservationBudget&  after,
                              double                     dt_stage)
{
    if (!std::isfinite(dt_stage) || dt_stage <= 0.0)
    {
        return;
    }

    const BudgetDrift drift = evaluate_budget_drift(before, after);
    const bool        warn  = drift.warn_kernel || drift.warn_physical;

    // Always emit the CSV row when the stream is open, regardless of
    // warn level — plotting tools need every transition for the time
    // series. The warn-vs-debug split below only affects console logs.
    {
        std::lock_guard<std::mutex> lock(g_budget_csv_mutex);
        if (auto* selected = g_budget_csv_files.stream(state.budget_level, g_budget_csv_mutex))
        {
            auto& g_budget_csv = *selected;
            g_budget_csv << state.time_s << ',' << stage << ',' << drift.d_total_mass << ',' << drift.d_theta_volume
                         << ',' << drift.d_q_volume << ',' << drift.d_dry_mass << ',' << drift.d_total_water << ','
                         << drift.d_thermal_energy << ',' << drift.rel_total_mass << ',' << drift.rel_theta_volume
                         << ',' << drift.rel_q_volume << ',' << drift.rel_dry_mass << ',' << drift.rel_total_water
                         << ',' << drift.rel_thermal_energy << ',' << (drift.warn_kernel ? 1 : 0) << ','
                         << (drift.warn_physical ? 1 : 0) << ',' << drift.d_moist_total_mass << ','
                         << drift.rel_moist_total_mass;
            for (double species_change : drift.d_water_species)
                g_budget_csv << ',' << species_change;
            g_budget_csv << '\n';
        }
    }

    if (!warn && !log_debug_enabled())
    {
        return;
    }

    auto emit = [&](auto log_fn)
    {
        log_fn("[PHYSICS BUDGET",
               drift.warn_kernel     ? " WARN KERNEL]"
               : drift.warn_physical ? " WARN]"
                                     : "]",
               " stage=",
               stage,
               " d_total_mass=",
               drift.d_total_mass,
               " d_theta_volume=",
               drift.d_theta_volume,
               " d_q_volume=",
               drift.d_q_volume,
               " d_dry_mass=",
               drift.d_dry_mass,
               " d_total_water=",
               drift.d_total_water,
               " d_thermal_energy=",
               drift.d_thermal_energy,
               " rel_total_mass=",
               drift.rel_total_mass,
               " rel_theta_volume=",
               drift.rel_theta_volume,
               " rel_q_volume=",
               drift.rel_q_volume,
               " rel_dry_mass=",
               drift.rel_dry_mass,
               " rel_total_water=",
               drift.rel_total_water,
               " d_moist_total_mass=",
               drift.d_moist_total_mass,
               " rel_moist_total_mass=",
               drift.rel_moist_total_mass,
               " rel_thermal_energy=",
               drift.rel_thermal_energy);
    };

    if (warn)
    {
        emit([](auto&&... args) { tmv::log_warn(args...); });
    }
    else
    {
        emit([](auto&&... args) { tmv::log_debug(args...); });
    }
}

void report_budget_transition(const char*               stage,
                              const ConservationBudget& before,
                              const ConservationBudget& after,
                              double                    dt_stage)
{
    report_budget_transition(global_simulation_state(), stage, before, after, dt_stage);
}

void open_budget_csv(const std::string& path)
{
    std::lock_guard<std::mutex> lock(g_budget_csv_mutex);
    if (!g_budget_csv_files.open(path, k_budget_csv_header, 17))
    {
        tmv::log_warn("[BUDGET CSV] failed to open '", path, "' for writing; time-series output disabled");
        return;
    }
    tmv::log_info("[BUDGET CSV] writing budget time series to ", path);
}

void close_budget_csv()
{
    std::lock_guard<std::mutex> lock(g_budget_csv_mutex);
    g_budget_csv_files.close();
}

namespace
{
// See the header. Both gates default to true so standalone callers/tests keep
// historical behavior; the runtime independently selects scientific budgets
// and consumer-required derived fields for each model step.
bool g_step_diagnostics_enabled                  = true;
bool g_step_derived_diagnostics_enabled          = true;
bool g_step_radar_diagnostics_enabled            = true;
bool g_step_vorticity_diagnostics_enabled        = true;
bool g_step_angular_momentum_diagnostics_enabled = true;
bool g_step_pressure_diagnostics_enabled         = true;
} // namespace

void set_step_diagnostics_enabled(bool enabled)
{
    g_step_diagnostics_enabled = enabled;
}

bool step_diagnostics_enabled()
{
    return g_step_diagnostics_enabled;
}

void set_step_derived_diagnostics_enabled(bool enabled)
{
    g_step_derived_diagnostics_enabled          = enabled;
    g_step_radar_diagnostics_enabled            = enabled;
    g_step_vorticity_diagnostics_enabled        = enabled;
    g_step_angular_momentum_diagnostics_enabled = enabled;
    g_step_pressure_diagnostics_enabled         = enabled;
}

bool step_derived_diagnostics_enabled()
{
    return g_step_derived_diagnostics_enabled;
}

void set_step_derived_diagnostic_requirements(bool radar, bool vorticity, bool angular_momentum, bool pressure)
{
    g_step_radar_diagnostics_enabled            = radar;
    g_step_vorticity_diagnostics_enabled        = vorticity;
    g_step_angular_momentum_diagnostics_enabled = angular_momentum;
    g_step_pressure_diagnostics_enabled         = pressure;
    g_step_derived_diagnostics_enabled          = radar || vorticity || angular_momentum || pressure;
}

bool step_radar_diagnostics_enabled()
{
    return g_step_radar_diagnostics_enabled;
}

bool step_dynamics_diagnostics_enabled()
{
    return g_step_vorticity_diagnostics_enabled || g_step_angular_momentum_diagnostics_enabled ||
           g_step_pressure_diagnostics_enabled;
}

bool step_vorticity_diagnostics_enabled()
{
    return g_step_vorticity_diagnostics_enabled;
}

bool step_angular_momentum_diagnostics_enabled()
{
    return g_step_angular_momentum_diagnostics_enabled;
}

bool step_pressure_diagnostics_enabled()
{
    return g_step_pressure_diagnostics_enabled;
}
