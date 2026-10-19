#include "diagnostics/level_budget_csv.hpp"
/**
 * @file energy_budget.cpp
 * @brief Domain-integrated energy budget diagnostics (see header).
 */

#include "diagnostics/energy_budget.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/simulation_state.hpp"
#include "terrain/conservative_transport.hpp"
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

tmv::LevelBudgetCsv g_energy_csv_files;
std::mutex    g_energy_csv_mutex;

constexpr const char* k_energy_csv_header = "sim_time_s,stage,"
                                            "d_kinetic,d_internal,d_potential,d_latent,d_total,"
                                            "rel_kinetic,rel_internal,rel_potential,rel_latent,rel_total,"
                                            "warn,moist_basis_version,"
                                            "d_moist_dry_internal_j,"
                                            "d_moist_water_sensible_j,"
                                            "d_moist_phase_reference_j,"
                                            "d_moist_resolved_kinetic_j,"
                                            "d_moist_gravitational_j,"
                                            "d_moist_subgrid_tke_j,"
                                            "d_moist_total_water_mass_kg,"
                                            "d_moist_total_j,"
                                            "moist_v2_basis_version,"
                                            "d_moist_v2_dry_internal_j,"
                                            "d_moist_v2_water_sensible_j,"
                                            "d_moist_v2_phase_reference_j,"
                                            "d_moist_v2_resolved_kinetic_j,"
                                            "d_moist_v2_gravitational_j,"
                                            "d_moist_v2_subgrid_tke_j,"
                                            "d_moist_v2_total_water_mass_kg,"
                                            "d_moist_v2_total_j,"
                                            "moist_v2_before_dry_internal_j,"
                                            "moist_v2_before_water_sensible_j,"
                                            "moist_v2_before_phase_reference_j,"
                                            "moist_v2_before_resolved_kinetic_j,"
                                            "moist_v2_before_gravitational_j,"
                                            "moist_v2_before_subgrid_tke_j,"
                                            "moist_v2_before_total_water_mass_kg,"
                                            "moist_v2_before_total_j,"
                                            "moist_v2_after_dry_internal_j,"
                                            "moist_v2_after_water_sensible_j,"
                                            "moist_v2_after_phase_reference_j,"
                                            "moist_v2_after_resolved_kinetic_j,"
                                            "moist_v2_after_gravitational_j,"
                                            "moist_v2_after_subgrid_tke_j,"
                                            "moist_v2_after_total_water_mass_kg,"
                                            "moist_v2_after_total_j\n";

bool field_matches_grid(const Field3D& field, const GridStateView& grid)
{
    return field.size_r() == grid.nr
        && field.size_th() == grid.nth
        && field.size_z() == grid.nz;
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
        // The rigid ground face w[-1] is implicit and zero on the production
        // C-grid, so the lowest scalar center sees half of the stored top face.
        return 0.5 * top;
    }
    return 0.5 * (static_cast<double>(fields.w[i][j][k - 1]) + top);
}

} // namespace

EnergyBudget compute_energy_budget(const SimulationStateView& state)
{
    EnergyBudget budget{};
    const auto&  grid   = state.grid;
    const auto&  fields = state.fields;
    const bool all_inputs_empty =
        fields.rho.empty() && fields.p.empty() && fields.theta.empty()
        && fields.u.empty() && fields.v.empty() && fields.w.empty()
        && fields.qv.empty();
    if (all_inputs_empty)
    {
        return budget;
    }
    if (!field_matches_grid(fields.rho, grid)
        || !field_matches_grid(fields.p, grid)
        || !field_matches_grid(fields.theta, grid)
        || !field_matches_grid(fields.u, grid)
        || !field_matches_grid(fields.v, grid)
        || !field_matches_grid(fields.w, grid)
        || (!fields.qv.empty() && !field_matches_grid(fields.qv, grid)))
    {
        throw std::invalid_argument(
            "energy budget requires every allocated input field to match the grid");
    }
    const bool have_qv = !fields.qv.empty();
    tmv::physical_grid_metrics::validate(state);

    constexpr double kappa     = R_d / cp;
    constexpr double cv        = cp - R_d; // dry-air specific heat at const volume
    constexpr double L_v_const = physical_constants::latent_heat_vaporization_jkg;

    double                    kinetic   = 0.0;
    double                    internal  = 0.0;
    double                    potential = 0.0;
    double                    latent    = 0.0;
    const PhysicalCellDomain domain = physical_scalar_domain(grid);

    if (tmv::diagnostics::deterministic_reductions_enabled())
    {
        const auto sums = tmv::diagnostics::fixed_slab_reduce<5>(
            domain.i_begin,
            domain.i_end,
            [&](int i, std::array<double, 5>& partial)
            {
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                {
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        const double cell_volume =
                            tmv::physical_grid_metrics::cell_volume(
                                state, i, j, k);
                        const double rho_val = static_cast<double>(fields.rho[i][j][k]);
                        const double u_val   = u_at_scalar_center(fields, i, j, k, grid.geometry.staggered);
                        const double v_val   = v_at_scalar_center(fields, i, j, k, grid.geometry.staggered);
                        const double w_val   = w_at_scalar_center(fields, i, j, k, grid.geometry.staggered);
                        const double theta_val = static_cast<double>(fields.theta[i][j][k]);
                        const double p_val     = static_cast<double>(fields.p[i][j][k]);
                        const double qv_val = have_qv
                            ? static_cast<double>(fields.qv[i][j][k]) : 0.0;
                        const double z_center =
                            tmv::physical_grid_metrics::scalar_height(
                                state, i, j, k);
                        if (!std::isfinite(cell_volume) || cell_volume <= 0.0
                            || !std::isfinite(z_center)
                            || !std::isfinite(rho_val) || rho_val <= 0.0
                            || !std::isfinite(p_val) || p_val <= 0.0
                            || !std::isfinite(theta_val) || theta_val <= 0.0
                            || !std::isfinite(u_val) || !std::isfinite(v_val)
                            || !std::isfinite(w_val)
                            || (have_qv && !std::isfinite(qv_val)))
                        {
                            partial[4] += 1.0;
                            continue;
                        }
                        const double rho_dv = rho_val * cell_volume;
                        partial[0] += 0.5 * rho_dv
                            * (u_val * u_val + v_val * v_val + w_val * w_val);
                        partial[2] += rho_dv * g * z_center;

                        const double temperature =
                            theta_val * std::pow(p_val / p0, kappa);
                        if (!std::isfinite(temperature) || temperature <= 0.0)
                        {
                            partial[4] += 1.0;
                            continue;
                        }
                        partial[1] += rho_dv * cv * temperature;
                        if (have_qv)
                        {
                            partial[3] += rho_dv * L_v_const * qv_val;
                        }
                    }
                }
            });
        if (sums[4] != 0.0)
        {
            throw std::runtime_error(
                "energy budget rejected non-finite or nonphysical input cells");
        }
        budget.kinetic   = sums[0];
        budget.internal  = sums[1];
        budget.potential = sums[2];
        budget.latent    = sums[3];
        budget.total     = sums[0] + sums[1] + sums[2] + sums[3];
        return budget;
    }

    long long invalid_cells = 0;
#pragma omp parallel for reduction(+ : kinetic, internal, potential, latent, invalid_cells)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double cell_volume =
                    tmv::physical_grid_metrics::cell_volume(state, i, j, k);
                const double rho_val = static_cast<double>(fields.rho[i][j][k]);
                const double u_val = u_at_scalar_center(fields, i, j, k, grid.geometry.staggered);
                const double v_val = v_at_scalar_center(fields, i, j, k, grid.geometry.staggered);
                const double w_val = w_at_scalar_center(fields, i, j, k, grid.geometry.staggered);
                const double z_center =
                    tmv::physical_grid_metrics::scalar_height(state, i, j, k);
                const double theta_val = static_cast<double>(fields.theta[i][j][k]);
                const double p_val     = static_cast<double>(fields.p[i][j][k]);
                const double qv_val = have_qv
                    ? static_cast<double>(fields.qv[i][j][k]) : 0.0;
                if (!std::isfinite(cell_volume) || cell_volume <= 0.0
                    || !std::isfinite(z_center)
                    || !std::isfinite(rho_val) || rho_val <= 0.0
                    || !std::isfinite(p_val) || p_val <= 0.0
                    || !std::isfinite(theta_val) || theta_val <= 0.0
                    || !std::isfinite(u_val) || !std::isfinite(v_val)
                    || !std::isfinite(w_val)
                    || (have_qv && !std::isfinite(qv_val)))
                {
                    ++invalid_cells;
                    continue;
                }
                const double rho_dv = rho_val * cell_volume;
                kinetic += 0.5 * rho_dv
                    * (u_val * u_val + v_val * v_val + w_val * w_val);
                potential += rho_dv * g * z_center;

                const double temperature =
                    theta_val * std::pow(p_val / p0, kappa);
                if (!std::isfinite(temperature) || temperature <= 0.0)
                {
                    ++invalid_cells;
                    continue;
                }
                internal += rho_dv * cv * temperature;

                if (have_qv)
                {
                    latent += rho_dv * L_v_const * qv_val;
                }
            }
        }
    }

    if (invalid_cells != 0)
    {
        throw std::runtime_error(
            "energy budget rejected non-finite or nonphysical input cells");
    }
    budget.kinetic   = kinetic;
    budget.internal  = internal;
    budget.potential = potential;
    budget.latent    = latent;
    budget.total     = kinetic + internal + potential + latent;
    return budget;
}

EnergyBudget compute_energy_budget()
{
    return compute_energy_budget(global_simulation_state());
}

static MoistMixtureEnergyBudgetV2 upgrade_moist_v1_to_v2(
    const SimulationStateView& state,
    const MoistMixtureEnergyBudgetV1& v1,
    const HorizontalEnergyBudgetExclusion* exclusion,
    bool force_deterministic);

static EnergyDiagnosticSnapshotV1 compute_energy_diagnostic_snapshot_v1_impl(
    const SimulationStateView& state,
    const HorizontalEnergyBudgetExclusion* exclusion,
    bool force_deterministic = false, bool absolute_reservoirs = false)
{
    const auto& grid = state.grid;
    const auto& fields = state.fields;
    const std::array<const Field3D*, 14> required_fields{{
        &fields.rho, &fields.p, &fields.theta,
        &fields.u, &fields.v, &fields.w,
        &fields.qv, &fields.qc, &fields.qr, &fields.qi,
        &fields.qs, &fields.qg, &fields.qh, &fields.tke,
    }};
    const bool all_inputs_empty = std::all_of(
        required_fields.begin(), required_fields.end(),
        [](const Field3D* field) { return field->empty(); });
    if (all_inputs_empty)
        return {};

    for (const Field3D* field : required_fields)
    {
        if (!field_matches_grid(*field, grid))
        {
            throw std::invalid_argument(
                "moist-mixture energy v1 requires every prognostic "
                "thermodynamic, water, velocity, and TKE field to match the grid");
        }
    }
    tmv::physical_grid_metrics::validate(state);

    constexpr double cv_d =
        physical_constants::specific_heat_cp_jkgk
        - physical_constants::gas_constant_dry_air_jkgk;
    constexpr double cv_v =
        physical_constants::specific_heat_water_vapor_cp_jkgk
        - physical_constants::gas_constant_water_vapor_jkgk;
    constexpr double c_l =
        physical_constants::specific_heat_liquid_water_jkgk;
    constexpr double c_i = physical_constants::specific_heat_ice_jkgk;
    constexpr double t_ref = physical_constants::freezing_temperature_k;
    constexpr double vapor_reference =
        physical_constants::latent_heat_vaporization_jkg
        - physical_constants::gas_constant_water_vapor_jkgk * t_ref;
    constexpr double ice_reference =
        -physical_constants::latent_heat_fusion_jkg;
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk
        / physical_constants::specific_heat_cp_jkgk;

    const PhysicalCellDomain domain = physical_scalar_domain(grid);
    if (exclusion != nullptr &&
        (exclusion->i_begin < 0 || exclusion->i_end > grid.nr ||
         exclusion->j_begin < 0 || exclusion->j_end > grid.nth ||
         exclusion->i_begin >= exclusion->i_end ||
         exclusion->j_begin >= exclusion->j_end))
        throw std::invalid_argument(
            "moist-mixture energy horizontal exclusion is invalid");
    const auto excluded = [&](int i, int j) noexcept
    {
        return exclusion != nullptr && i >= exclusion->i_begin &&
               i < exclusion->i_end && j >= exclusion->j_begin &&
               j < exclusion->j_end;
    };
    const tmv::physical_grid_metrics::ControlVolumeMassView mass_geometry(
        state);

    const auto accumulate_cell =
        [&](int i, int j, int k, std::array<double, 12>& sums)
    {
        const double rho_d = static_cast<double>(fields.rho[i][j][k]);
        const double pressure = static_cast<double>(fields.p[i][j][k]);
        const double theta_k = static_cast<double>(fields.theta[i][j][k]);
        const double qv_k = static_cast<double>(fields.qv[i][j][k]);
        const double qc_k = static_cast<double>(fields.qc[i][j][k]);
        const double qr_k = static_cast<double>(fields.qr[i][j][k]);
        const double qi_k = static_cast<double>(fields.qi[i][j][k]);
        const double qs_k = static_cast<double>(fields.qs[i][j][k]);
        const double qg_k = static_cast<double>(fields.qg[i][j][k]);
        const double qh_k = static_cast<double>(fields.qh[i][j][k]);
        const double tke_k = static_cast<double>(fields.tke[i][j][k]);
        const double volume = mass_geometry.scalar_volume_m3(i, j, k);
        const double z = tmv::physical_grid_metrics::scalar_height(
            state, i, j, k);
        const double temperature = theta_k * std::pow(
            pressure / physical_constants::reference_pressure_pa, kappa);
        const double q_liquid = qc_k + qr_k;
        const double q_ice = qi_k + qs_k + qg_k + qh_k;
        const double q_total = qv_k + q_liquid + q_ice;

        const double values[] = {
            rho_d, pressure, theta_k, qv_k, qc_k, qr_k, qi_k, qs_k,
            qg_k, qh_k, tke_k, volume, z, temperature, q_total,
        };
        if (std::any_of(
                std::begin(values), std::end(values),
                [](double value) { return !std::isfinite(value); })
            || rho_d <= 0.0 || pressure <= 0.0 || theta_k <= 0.0
            || temperature <= 0.0 || volume <= 0.0
            || qv_k < 0.0 || qc_k < 0.0 || qr_k < 0.0
            || qi_k < 0.0 || qs_k < 0.0 || qg_k < 0.0
            || qh_k < 0.0 || tke_k < 0.0)
        {
            sums[11] += 1.0;
            return;
        }

        const double dry_mass = rho_d * volume;
        const double mixture_mass = dry_mass * (1.0 + q_total);
        const double delta_t = temperature - t_ref;
        double kinetic = 0.0;
        double legacy_u = 0.0;
        double legacy_v = 0.0;
        double legacy_w = 0.0;
        if (!grid.geometry.staggered)
        {
            const double u_k = static_cast<double>(fields.u[i][j][k]);
            const double v_k = static_cast<double>(fields.v[i][j][k]);
            const double w_k = static_cast<double>(fields.w[i][j][k]);
            if (!std::isfinite(u_k) || !std::isfinite(v_k)
                || !std::isfinite(w_k))
            {
                sums[11] += 1.0;
                return;
            }
            legacy_u = u_k;
            legacy_v = v_k;
            legacy_w = w_k;
            kinetic = 0.5 * mixture_mass
                * (u_k * u_k + v_k * v_k + w_k * w_k);
        }
        else
        {
            // Each scalar cell contributes half its mixture mass to each
            // adjacent component control volume. Interior face halves combine;
            // physical boundary faces retain the one owned half. The C-grid
            // ground face is implicit and exactly zero.
            const double u_w = static_cast<double>(fields.u[i - 1][j][k]);
            const double u_e = static_cast<double>(fields.u[i][j][k]);
            const double v_s = static_cast<double>(fields.v[i][j - 1][k]);
            const double v_n = static_cast<double>(fields.v[i][j][k]);
            const double w_b = k == domain.k_begin
                ? 0.0 : static_cast<double>(fields.w[i][j][k - 1]);
            const double w_t = static_cast<double>(fields.w[i][j][k]);
            if (!std::isfinite(u_w) || !std::isfinite(u_e)
                || !std::isfinite(v_s) || !std::isfinite(v_n)
                || !std::isfinite(w_b) || !std::isfinite(w_t))
            {
                sums[11] += 1.0;
                return;
            }
            legacy_u = 0.5 * (u_w + u_e);
            legacy_v = 0.5 * (v_s + v_n);
            legacy_w = 0.5 * (w_b + w_t);
            kinetic = 0.25 * mixture_mass
                * (u_w * u_w + u_e * u_e
                   + v_s * v_s + v_n * v_n
                   + w_b * w_b + w_t * w_t);
        }

        const std::array<double, 6> reservoirs{
            dry_mass * cv_d * delta_t,
            dry_mass * delta_t * (qv_k * cv_v + q_liquid * c_l + q_ice * c_i),
            dry_mass * (qv_k * vapor_reference + q_ice * ice_reference),
            kinetic, mixture_mass * physical_constants::gravity_ms2 * z, dry_mass * tke_k};
        for (std::size_t reservoir = 0; reservoir < reservoirs.size(); ++reservoir)
            sums[reservoir] += absolute_reservoirs ? std::abs(reservoirs[reservoir]) : reservoirs[reservoir];
        sums[6] += dry_mass * q_total;
        sums[7] += 0.5 * dry_mass
            * (legacy_u * legacy_u + legacy_v * legacy_v
               + legacy_w * legacy_w);
        sums[8] += dry_mass * cv_d * temperature;
        sums[9] += dry_mass
            * physical_constants::gravity_ms2 * z;
        sums[10] += dry_mass
            * physical_constants::latent_heat_vaporization_jkg * qv_k;
    };

    std::array<double, 12> sums{};
    if (force_deterministic ||
        tmv::diagnostics::deterministic_reductions_enabled())
    {
        sums = tmv::diagnostics::fixed_slab_reduce<12>(
            domain.i_begin, domain.i_end,
            [&](int i, std::array<double, 12>& partial)
            {
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                {
                    if (excluded(i, j)) continue;
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                        accumulate_cell(i, j, k, partial);
                }
            });
    }
    else
    {
        double dry_internal = 0.0;
        double water_sensible = 0.0;
        double phase_reference = 0.0;
        double resolved_kinetic = 0.0;
        double gravitational = 0.0;
        double subgrid_tke = 0.0;
        double total_water_mass = 0.0;
        double legacy_kinetic = 0.0;
        double legacy_internal = 0.0;
        double legacy_potential = 0.0;
        double legacy_latent = 0.0;
        double invalid_cells = 0.0;
        #pragma omp parallel for collapse(2) reduction(+:dry_internal,water_sensible,phase_reference,resolved_kinetic,gravitational,subgrid_tke,total_water_mass,legacy_kinetic,legacy_internal,legacy_potential,legacy_latent,invalid_cells)
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
            {
                if (excluded(i, j)) continue;
                std::array<double, 12> partial{};
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                    accumulate_cell(i, j, k, partial);
                dry_internal += partial[0];
                water_sensible += partial[1];
                phase_reference += partial[2];
                resolved_kinetic += partial[3];
                gravitational += partial[4];
                subgrid_tke += partial[5];
                total_water_mass += partial[6];
                legacy_kinetic += partial[7];
                legacy_internal += partial[8];
                legacy_potential += partial[9];
                legacy_latent += partial[10];
                invalid_cells += partial[11];
            }
        sums = {dry_internal, water_sensible, phase_reference,
                resolved_kinetic, gravitational, subgrid_tke,
                total_water_mass, legacy_kinetic, legacy_internal,
                legacy_potential, legacy_latent, invalid_cells};
    }

    if (sums[11] != 0.0)
    {
        throw std::runtime_error(
            "moist-mixture energy v1 rejected non-finite or nonphysical input cells");
    }
    EnergyDiagnosticSnapshotV1 snapshot;
    MoistMixtureEnergyBudgetV1& budget = snapshot.moist_mixture;
    budget.dry_internal = sums[0];
    budget.water_sensible = sums[1];
    budget.phase_reference = sums[2];
    budget.resolved_kinetic = sums[3];
    budget.gravitational = sums[4];
    budget.subgrid_tke = sums[5];
    budget.total_water_mass = sums[6];
    budget.total = sums[0] + sums[1] + sums[2] + sums[3]
        + sums[4] + sums[5];
    const double output_values[] = {
        budget.dry_internal, budget.water_sensible,
        budget.phase_reference, budget.resolved_kinetic,
        budget.gravitational, budget.subgrid_tke,
        budget.total_water_mass, budget.total,
    };
    if (std::any_of(
            std::begin(output_values), std::end(output_values),
            [](double value) { return !std::isfinite(value); }))
    {
        throw std::overflow_error(
            "moist-mixture energy v1 produced a non-finite inventory");
    }
    snapshot.legacy_proxy.kinetic = sums[7];
    snapshot.legacy_proxy.internal = sums[8];
    snapshot.legacy_proxy.potential = sums[9];
    snapshot.legacy_proxy.latent = sums[10];
    snapshot.legacy_proxy.total =
        sums[7] + sums[8] + sums[9] + sums[10];
    if (!std::isfinite(snapshot.legacy_proxy.total))
    {
        throw std::overflow_error(
            "fused legacy energy proxy produced a non-finite inventory");
    }
    snapshot.moist_mixture_v2 = upgrade_moist_v1_to_v2(
        state, budget, exclusion, force_deterministic);
    return snapshot;
}

EnergyDiagnosticSnapshotV1 compute_energy_diagnostic_snapshot_v1(
    const SimulationStateView& state)
{
    return compute_energy_diagnostic_snapshot_v1_impl(state, nullptr);
}

EnergyDiagnosticSnapshotV1 compute_energy_diagnostic_snapshot_v1()
{
    return compute_energy_diagnostic_snapshot_v1(global_simulation_state());
}

EnergyDiagnosticSnapshotV1 compute_energy_diagnostic_snapshot_v1_deterministic(
    const SimulationStateView& state)
{
    return compute_energy_diagnostic_snapshot_v1_impl(state, nullptr, true);
}

MoistMixtureEnergyBudgetV1 compute_moist_mixture_energy_v1(
    const SimulationStateView& state)
{
    return compute_energy_diagnostic_snapshot_v1(state).moist_mixture;
}

MoistMixtureEnergyBudgetV1 compute_moist_mixture_energy_v1()
{
    return compute_energy_diagnostic_snapshot_v1().moist_mixture;
}

static MoistMixtureEnergyBudgetV2 upgrade_moist_v1_to_v2(
    const SimulationStateView& state,
    const MoistMixtureEnergyBudgetV1& v1,
    const HorizontalEnergyBudgetExclusion* exclusion,
    bool force_deterministic)
{
    MoistMixtureEnergyBudgetV2 v2{
        v1.dry_internal,
        v1.water_sensible,
        v1.phase_reference,
        v1.resolved_kinetic,
        v1.gravitational,
        v1.subgrid_tke,
        v1.total_water_mass,
        v1.total,
    };

    const TerrainMetrics3D* metrics =
        tmv::physical_grid_metrics::active_terrain_metrics(state);
    if (metrics == nullptr || !state.grid.geometry.staggered
        || state.fields.rho.empty())
    {
        return v2;
    }

    const auto& fields = state.fields;
    terrain_transport::validate_cgrid_state(
        *metrics, fields.rho, fields.u, fields.v, fields.w);
    const PhysicalCellDomain domain = physical_scalar_domain(state.grid);
    const tmv::physical_grid_metrics::ControlVolumeMassView mass_geometry(
        state);
    const auto ground_correction_for_i =
        [&](int i, std::array<double, 1>& partial)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            if (exclusion != nullptr && i >= exclusion->i_begin &&
                i < exclusion->i_end && j >= exclusion->j_begin &&
                j < exclusion->j_end)
                continue;
            const int k = domain.k_begin;
            const double q_total =
                static_cast<double>(fields.qv[i][j][k])
                + static_cast<double>(fields.qc[i][j][k])
                + static_cast<double>(fields.qr[i][j][k])
                + static_cast<double>(fields.qi[i][j][k])
                + static_cast<double>(fields.qs[i][j][k])
                + static_cast<double>(fields.qg[i][j][k])
                + static_cast<double>(fields.qh[i][j][k]);
            const double mixture_mass =
                mass_geometry.scalar_mass_kg(fields.rho, i, j, k)
                * (1.0 + q_total);
            const double ground_w =
                terrain_transport::cgrid_ground_vertical_velocity(
                    *metrics, fields.u, fields.v, i, j,
                    tmv::physical_grid_metrics::surface_velocity_ms(state).u_ms,
                    tmv::physical_grid_metrics::surface_velocity_ms(state).v_ms);
            const double correction =
                0.25 * mixture_mass * ground_w * ground_w;
            if (!std::isfinite(mixture_mass) || mixture_mass <= 0.0
                || !std::isfinite(ground_w)
                || !std::isfinite(correction))
            {
                throw std::runtime_error(
                    "moist-mixture energy v2 rejected invalid terrain "
                    "ground-face kinetic state");
            }
            partial[0] += correction;
        }
    };

    double ground_kinetic = 0.0;
    if (force_deterministic ||
        tmv::diagnostics::deterministic_reductions_enabled())
    {
        ground_kinetic = tmv::diagnostics::fixed_slab_reduce<1>(
            domain.i_begin, domain.i_end,
            ground_correction_for_i)[0];
    }
    else
    {
        #pragma omp parallel for reduction(+:ground_kinetic)
        for (int i = domain.i_begin; i < domain.i_end; ++i)
        {
            std::array<double, 1> partial{};
            ground_correction_for_i(i, partial);
            ground_kinetic += partial[0];
        }
    }
    if (!std::isfinite(ground_kinetic) || ground_kinetic < 0.0)
    {
        throw std::runtime_error(
            "moist-mixture energy v2 terrain ground correction overflowed");
    }
    v2.resolved_kinetic += ground_kinetic;
    v2.total += ground_kinetic;
    if (!std::isfinite(v2.resolved_kinetic) || !std::isfinite(v2.total))
    {
        throw std::overflow_error(
            "moist-mixture energy v2 terrain inventory overflowed");
    }
    return v2;
}

MoistMixtureEnergyBudgetV2 compute_moist_mixture_energy_v2(
    const SimulationStateView& state)
{
    return compute_energy_diagnostic_snapshot_v1(state).moist_mixture_v2;
}

MoistMixtureEnergyBudgetV2 compute_moist_mixture_energy_v2_deterministic(
    const SimulationStateView& state)
{
    return compute_energy_diagnostic_snapshot_v1_impl(state, nullptr, true)
        .moist_mixture_v2;
}

MoistMixtureEnergyBudgetV2
compute_moist_mixture_energy_v2_excluding_horizontal_rectangle(
    const SimulationStateView& state,
    const HorizontalEnergyBudgetExclusion& exclusion)
{
    return compute_energy_diagnostic_snapshot_v1_impl(state, &exclusion)
        .moist_mixture_v2;
}

MoistMixtureEnergyBudgetV2 compute_moist_mixture_energy_v2_excluding_horizontal_rectangle_deterministic(
    const SimulationStateView& state, const HorizontalEnergyBudgetExclusion& exclusion)
{
    return compute_energy_diagnostic_snapshot_v1_impl(state, &exclusion, true).moist_mixture_v2;
}

std::array<double, 7> moist_mixture_energy_v2_magnitudes(
    const SimulationStateView& state, const HorizontalEnergyBudgetExclusion* exclusion)
{
    const auto result = compute_energy_diagnostic_snapshot_v1_impl(state, exclusion, true, true).moist_mixture_v2;
    return {result.dry_internal, result.water_sensible, result.phase_reference, result.resolved_kinetic,
            result.gravitational, result.subgrid_tke, result.total};
}

MoistMixtureEnergyBudgetV2 compute_moist_mixture_energy_v2()
{
    return compute_moist_mixture_energy_v2(global_simulation_state());
}

EnergyDrift evaluate_energy_drift(const EnergyBudget& before, const EnergyBudget& after)
{
    const double values[] = {
        before.kinetic, before.internal, before.potential, before.latent,
        before.total, after.kinetic, after.internal, after.potential,
        after.latent, after.total,
    };
    if (std::any_of(
            std::begin(values), std::end(values),
            [](double value) { return !std::isfinite(value); }))
    {
        throw std::invalid_argument(
            "energy budget drift requires finite snapshots");
    }
    EnergyDrift drift;

    drift.d_kinetic   = after.kinetic - before.kinetic;
    drift.d_internal  = after.internal - before.internal;
    drift.d_potential = after.potential - before.potential;
    drift.d_latent    = after.latent - before.latent;
    drift.d_total     = after.total - before.total;

    drift.rel_kinetic   = relative_drift(before.kinetic, after.kinetic);
    drift.rel_internal  = relative_drift(before.internal, after.internal);
    drift.rel_potential = relative_drift(before.potential, after.potential);
    drift.rel_latent    = relative_drift(before.latent, after.latent);
    drift.rel_total     = relative_drift(before.total, after.total);

    // Only the total must close; components exchange energy by design.
    drift.warn = drift.rel_total > BUDGET_THRESH_ENERGY_TOTAL;
    return drift;
}

void report_energy_transition(const SimulationStateView& state,
                              const char*                stage,
                              const EnergyBudget&        before,
                              const EnergyBudget&        after,
                              double                     dt_stage)
{
    if (!std::isfinite(dt_stage) || dt_stage <= 0.0)
    {
        return;
    }

    const EnergyDrift drift = evaluate_energy_drift(before, after);

    {
        std::lock_guard<std::mutex> lock(g_energy_csv_mutex);
        if (auto* selected = g_energy_csv_files.stream(state.budget_level, g_energy_csv_mutex))
        {
            auto& g_energy_csv = *selected;
            g_energy_csv << state.time_s << ',' << stage << ',' << drift.d_kinetic << ',' << drift.d_internal << ','
                         << drift.d_potential << ',' << drift.d_latent << ',' << drift.d_total << ','
                         << drift.rel_kinetic << ',' << drift.rel_internal << ',' << drift.rel_potential << ','
                         << drift.rel_latent << ',' << drift.rel_total << ',' << (drift.warn ? 1 : 0)
                         << ",0,0,0,0,0,0,0,0,0\n";
        }
    }

    if (!drift.warn && !log_debug_enabled())
    {
        return;
    }

    auto emit = [&](auto log_fn)
    {
        log_fn("[ENERGY BUDGET",
               drift.warn ? " WARN]" : "]",
               " stage=",
               stage,
               " d_kinetic=",
               drift.d_kinetic,
               " d_internal=",
               drift.d_internal,
               " d_potential=",
               drift.d_potential,
               " d_latent=",
               drift.d_latent,
               " d_total=",
               drift.d_total,
               " rel_total=",
               drift.rel_total);
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

void report_energy_transition(
    const SimulationStateView& state,
    const char* stage,
    const EnergyDiagnosticSnapshotV1& before,
    const EnergyDiagnosticSnapshotV1& after,
    double dt_stage)
{
    if (!std::isfinite(dt_stage) || dt_stage <= 0.0)
        return;

    const EnergyDrift drift = evaluate_energy_drift(
        before.legacy_proxy, after.legacy_proxy);
    const auto& b = before.moist_mixture;
    const auto& a = after.moist_mixture;
    const double moist_changes[] = {
        a.dry_internal - b.dry_internal,
        a.water_sensible - b.water_sensible,
        a.phase_reference - b.phase_reference,
        a.resolved_kinetic - b.resolved_kinetic,
        a.gravitational - b.gravitational,
        a.subgrid_tke - b.subgrid_tke,
        a.total_water_mass - b.total_water_mass,
        a.total - b.total,
    };
    const auto& b2 = before.moist_mixture_v2;
    const auto& a2 = after.moist_mixture_v2;
    const double moist_v2_changes[] = {
        a2.dry_internal - b2.dry_internal,
        a2.water_sensible - b2.water_sensible,
        a2.phase_reference - b2.phase_reference,
        a2.resolved_kinetic - b2.resolved_kinetic,
        a2.gravitational - b2.gravitational,
        a2.subgrid_tke - b2.subgrid_tke,
        a2.total_water_mass - b2.total_water_mass,
        a2.total - b2.total,
    };
    const double moist_v2_before[] = {
        b2.dry_internal, b2.water_sensible, b2.phase_reference,
        b2.resolved_kinetic, b2.gravitational, b2.subgrid_tke,
        b2.total_water_mass, b2.total,
    };
    const double moist_v2_after[] = {
        a2.dry_internal, a2.water_sensible, a2.phase_reference,
        a2.resolved_kinetic, a2.gravitational, a2.subgrid_tke,
        a2.total_water_mass, a2.total,
    };
    if (std::any_of(
            std::begin(moist_changes), std::end(moist_changes),
            [](double value) { return !std::isfinite(value); })
        || std::any_of(
            std::begin(moist_v2_changes), std::end(moist_v2_changes),
            [](double value) { return !std::isfinite(value); })
        || std::any_of(
            std::begin(moist_v2_before), std::end(moist_v2_before),
            [](double value) { return !std::isfinite(value); })
        || std::any_of(
            std::begin(moist_v2_after), std::end(moist_v2_after),
            [](double value) { return !std::isfinite(value); }))
    {
        throw std::invalid_argument(
            "moist-mixture energy transition requires finite snapshots");
    }

    {
        std::lock_guard<std::mutex> lock(g_energy_csv_mutex);
        if (auto* selected = g_energy_csv_files.stream(state.budget_level, g_energy_csv_mutex))
        {
            auto& g_energy_csv = *selected;
            g_energy_csv
                << state.time_s << ',' << stage << ','
                << drift.d_kinetic << ',' << drift.d_internal << ','
                << drift.d_potential << ',' << drift.d_latent << ','
                << drift.d_total << ',' << drift.rel_kinetic << ','
                << drift.rel_internal << ',' << drift.rel_potential << ','
                << drift.rel_latent << ',' << drift.rel_total << ','
                << (drift.warn ? 1 : 0) << ','
                << MoistMixtureEnergyBudgetV1::basis_version;
            for (double change : moist_changes)
                g_energy_csv << ',' << change;
            g_energy_csv << ','
                << MoistMixtureEnergyBudgetV2::basis_version;
            for (double change : moist_v2_changes)
                g_energy_csv << ',' << change;
            for (double inventory : moist_v2_before)
                g_energy_csv << ',' << inventory;
            for (double inventory : moist_v2_after)
                g_energy_csv << ',' << inventory;
            g_energy_csv << '\n';
        }
    }

    if (!drift.warn && !log_debug_enabled())
        return;
    if (drift.warn)
    {
        tmv::log_warn(
            "[ENERGY BUDGET WARN] stage=", stage,
            " d_total=", drift.d_total,
            " rel_total=", drift.rel_total,
            " moist_v1_d_total=", moist_changes[7],
            " moist_v2_d_total=", moist_v2_changes[7]);
    }
    else
    {
        tmv::log_debug(
            "[ENERGY BUDGET] stage=", stage,
            " d_total=", drift.d_total,
            " rel_total=", drift.rel_total,
            " moist_v1_d_total=", moist_changes[7],
            " moist_v2_d_total=", moist_v2_changes[7]);
    }
}

void report_energy_transition(const char* stage, const EnergyBudget& before, const EnergyBudget& after, double dt_stage)
{
    report_energy_transition(global_simulation_state(), stage, before, after, dt_stage);
}

void open_energy_csv(const std::string& path)
{
    std::lock_guard<std::mutex> lock(g_energy_csv_mutex);
    if (!g_energy_csv_files.open(path, k_energy_csv_header, 17))
    {
        tmv::log_warn("[ENERGY CSV] failed to open '", path, "' for writing; time-series output disabled");
        return;
    }
    tmv::log_info("[ENERGY CSV] writing budget time series to ", path);
}

void close_energy_csv()
{
    std::lock_guard<std::mutex> lock(g_energy_csv_mutex);
    g_energy_csv_files.close();
}
