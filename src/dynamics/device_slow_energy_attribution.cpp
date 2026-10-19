#include "dynamics/device_slow_energy_attribution.hpp"

#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/simulation_state.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "terrain/conservative_transport.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace
{

bool field_matches_domain(const Field3D& field, const GridStateView& grid)
{
    return field.size_r() == grid.nr
        && field.size_th() == grid.nth
        && field.size_z() == grid.nz;
}

float finite_or_zero(float value)
{
    return std::isfinite(value) ? value : 0.0f;
}

double mixture_mass_kg(
    const SimulationStateView& state,
    const tmv::physical_grid_metrics::ControlVolumeMassView& mass_geometry,
    int i,
    int j,
    int k)
{
    const auto& fields = state.fields;
    const double rho_d = static_cast<double>(fields.rho[i][j][k]);
    const double water[] = {
        static_cast<double>(fields.qv[i][j][k]),
        static_cast<double>(fields.qc[i][j][k]),
        static_cast<double>(fields.qr[i][j][k]),
        static_cast<double>(fields.qi[i][j][k]),
        static_cast<double>(fields.qs[i][j][k]),
        static_cast<double>(fields.qg[i][j][k]),
        static_cast<double>(fields.qh[i][j][k]),
    };
    if (!std::isfinite(rho_d) || rho_d <= 0.0)
        return std::numeric_limits<double>::quiet_NaN();
    double q_total = 0.0;
    for (double value : water)
    {
        if (!std::isfinite(value) || value < 0.0)
            return std::numeric_limits<double>::quiet_NaN();
        q_total += value;
    }
    const double mass =
        mass_geometry.scalar_mass_kg(fields.rho, i, j, k)
        * (1.0 + q_total);
    return std::isfinite(mass) && mass > 0.0
        ? mass : std::numeric_limits<double>::quiet_NaN();
}

} // namespace

namespace tmv::device_slow_energy
{

void validate_request(
    const SimulationStateView& state,
    const Field3D& du_dt,
    const Field3D& dv_dt,
    const Field3D& dw_dt,
    const SlowMomentumAdvectionTendencyOutputs* advection,
    float dt_large,
    bool attribute_momentum,
    bool attribute_pbl)
{
    if (!std::isfinite(dt_large) || dt_large <= 0.0f
        || (!attribute_momentum && !attribute_pbl))
    {
        throw std::invalid_argument(
            "device slow attribution requires finite dt > 0 and at least one owner");
    }
    const auto& grid = state.grid;
    const auto& fields = state.fields;
    const auto& tendencies = state.tendencies;
    const Field3D* required[] = {
        &fields.u, &fields.v, &fields.w, &fields.p, &fields.rho,
        &fields.theta, &fields.qv, &fields.qc, &fields.qr,
        &fields.qi, &fields.qs, &fields.qg, &fields.qh,
        &du_dt, &dv_dt, &dw_dt,
        &tendencies.du_dt_pbl, &tendencies.dv_dt_pbl,
    };
    if (std::any_of(
            std::begin(required), std::end(required),
            [&](const Field3D* field)
            {
                return field == nullptr || !field_matches_domain(*field, grid);
            }))
    {
        throw std::invalid_argument(
            "device slow attribution requires grid-shaped state and tendencies");
    }
    if (attribute_momentum
        && (advection == nullptr || !advection->complete()
            || !field_matches_domain(*advection->du_dt, grid)
            || !field_matches_domain(*advection->dv_dt, grid)
            || !field_matches_domain(*advection->dw_dt, grid)))
    {
        throw std::invalid_argument(
            "device slow momentum attribution requires complete grid-shaped "
            "advection tendencies");
    }
    if (!grid.geometry.staggered
        && tmv::physical_grid_metrics::active_terrain_metrics(state) != nullptr)
    {
        throw std::invalid_argument(
            "mapped device slow attribution requires the C-grid layout");
    }
    tmv::physical_grid_metrics::validate(state);
}

AttributionResult attribute_completed_commit(
    const SimulationStateView& state,
    const Field3D& du_dt,
    const Field3D& dv_dt,
    const Field3D& dw_dt,
    const SlowMomentumAdvectionTendencyOutputs* advection,
    float dt_large,
    bool attribute_momentum,
    bool attribute_pbl,
    bool wrap_x,
    bool wrap_y,
    const float* u_committed,
    const float* v_committed,
    const float* w_committed,
    const float* p_committed)
{
    validate_request(
        state, du_dt, dv_dt, dw_dt, advection, dt_large,
        attribute_momentum, attribute_pbl);
    if (u_committed == nullptr || v_committed == nullptr
        || w_committed == nullptr || p_committed == nullptr)
    {
        throw std::invalid_argument(
            "device slow attribution requires complete committed buffers");
    }

    const auto& grid = state.grid;
    const auto& fields = state.fields;
    const auto& tendencies = state.tendencies;
    const PhysicalCellDomain domain = physical_scalar_domain(grid);
    const bool cgrid = grid.geometry.staggered;
    const TerrainMetrics3D* metrics =
        tmv::physical_grid_metrics::active_terrain_metrics(state);
    const double frame_u_ms = tmv::physical_grid_metrics::surface_velocity_ms(state).u_ms;
    const double frame_v_ms = tmv::physical_grid_metrics::surface_velocity_ms(state).v_ms;
    const tmv::physical_grid_metrics::ControlVolumeMassView mass_geometry(state);
    const auto linear_index = [&](int i, int j, int k)
    {
        return (static_cast<std::size_t>(i)
                    * static_cast<std::size_t>(grid.nth)
                + static_cast<std::size_t>(j))
                * static_cast<std::size_t>(grid.nz)
            + static_cast<std::size_t>(k);
    };
    const auto final_value =
        [&](const float* values, int i, int j, int k)
        {
            return static_cast<double>(values[linear_index(i, j, k)]);
        };

    // Fixed x slabs make the report bit-identical across OpenMP thread
    // counts. Slots 0..7 are slow energy, 8..10 are the PBL
    // counterfactual, and 11 counts rejected cells.
    const auto sums = tmv::diagnostics::fixed_slab_reduce<12>(
        domain.i_begin, domain.i_end,
        [&](int i, std::array<double, 12>& partial)
        {
            const auto accumulate_momentum =
                [&](double mass, double old_velocity,
                    double new_velocity, double delta_total,
                    double delta_advection)
            {
                const double delta_other =
                    delta_total - delta_advection;
                const double values[] = {
                    mass, old_velocity, new_velocity,
                    delta_total, delta_advection, delta_other,
                };
                if (mass <= 0.0
                    || std::any_of(
                        std::begin(values), std::end(values),
                        [](double value) { return !std::isfinite(value); }))
                {
                    partial[11] += 1.0;
                    return;
                }
                partial[0] += mass * old_velocity * delta_advection;
                partial[1] += mass * old_velocity * delta_other;
                partial[2] +=
                    0.5 * mass * delta_advection * delta_advection;
                partial[3] += 0.5 * mass * delta_other * delta_other;
                partial[4] += mass * delta_advection * delta_other;
                partial[5] += 0.5 * mass
                    * (new_velocity * new_velocity
                       - old_velocity * old_velocity);
            };

            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const double moist_cell_mass =
                        mixture_mass_kg(state, mass_geometry, i, j, k);
                    if (!std::isfinite(moist_cell_mass))
                    {
                        partial[11] += 1.0;
                        continue;
                    }
                    const float total_u = finite_or_zero(
                        du_dt[i][j][k]
                        + tendencies.du_dt_pbl[i][j][k]);
                    const float total_v = finite_or_zero(
                        dv_dt[i][j][k]
                        + tendencies.dv_dt_pbl[i][j][k]);
                    const float total_w = finite_or_zero(dw_dt[i][j][k]);
                    const double delta_u =
                        static_cast<double>(total_u * dt_large);
                    const double delta_v =
                        static_cast<double>(total_v * dt_large);
                    const double delta_w =
                        static_cast<double>(total_w * dt_large);
                    const double adv_u = attribute_momentum
                        ? static_cast<double>(
                            (*advection->du_dt)[i][j][k] * dt_large)
                        : 0.0;
                    const double adv_v = attribute_momentum
                        ? static_cast<double>(
                            (*advection->dv_dt)[i][j][k] * dt_large)
                        : 0.0;
                    const double adv_w = attribute_momentum
                        ? static_cast<double>(
                            (*advection->dw_dt)[i][j][k] * dt_large)
                        : 0.0;

                    if (attribute_momentum)
                    {
                        if (!cgrid)
                        {
                            accumulate_momentum(
                                moist_cell_mass, fields.u[i][j][k],
                                final_value(u_committed, i, j, k),
                                delta_u, adv_u);
                            accumulate_momentum(
                                moist_cell_mass, fields.v[i][j][k],
                                final_value(v_committed, i, j, k),
                                delta_v, adv_v);
                            accumulate_momentum(
                                moist_cell_mass, fields.w[i][j][k],
                                final_value(w_committed, i, j, k),
                                delta_w, adv_w);
                        }
                        else
                        {
                            double u_mass = 0.5 * moist_cell_mass;
                            if (i + 1 < domain.i_end)
                                u_mass += 0.5 * mixture_mass_kg(
                                    state, mass_geometry, i + 1, j, k);
                            double v_mass = 0.5 * moist_cell_mass;
                            if (j + 1 < domain.j_end)
                                v_mass += 0.5 * mixture_mass_kg(
                                    state, mass_geometry, i, j + 1, k);
                            double w_mass = 0.5 * moist_cell_mass;
                            if (k + 1 < domain.k_end)
                                w_mass += 0.5 * mixture_mass_kg(
                                    state, mass_geometry, i, j, k + 1);
                            accumulate_momentum(
                                u_mass, fields.u[i][j][k],
                                final_value(u_committed, i, j, k),
                                delta_u, adv_u);
                            accumulate_momentum(
                                v_mass, fields.v[i][j][k],
                                final_value(v_committed, i, j, k),
                                delta_v, adv_v);
                            accumulate_momentum(
                                w_mass, fields.w[i][j][k],
                                final_value(w_committed, i, j, k),
                                delta_w, adv_w);

                            if (metrics != nullptr && k == domain.k_begin)
                            {
                                const double slope_x =
                                    -metrics->Gx_face(i, j, 0);
                                const double slope_y =
                                    -metrics->Gy_face(i, j, 0);
                                const double ground_old =
                                    terrain_transport::
                                        cgrid_ground_vertical_velocity(
                                            *metrics, fields.u, fields.v,
                                            i, j,
                                            frame_u_ms, frame_v_ms);
                                const double ground_new =
                                    slope_x * (0.5 * (
                                        final_value(
                                            u_committed, i - 1, j, k)
                                        + final_value(
                                            u_committed, i, j, k))
                                        - frame_u_ms)
                                    + slope_y * (0.5 * (
                                        final_value(
                                            v_committed, i, j - 1, k)
                                        + final_value(
                                            v_committed, i, j, k))
                                        - frame_v_ms);
                                const double ground_delta =
                                    slope_x * 0.5 * (
                                        static_cast<double>(
                                            finite_or_zero(
                                                du_dt[i - 1][j][k]
                                                + tendencies.du_dt_pbl
                                                    [i - 1][j][k])
                                            * dt_large)
                                        + delta_u)
                                    + slope_y * 0.5 * (
                                        static_cast<double>(
                                            finite_or_zero(
                                                dv_dt[i][j - 1][k]
                                                + tendencies.dv_dt_pbl
                                                    [i][j - 1][k])
                                            * dt_large)
                                        + delta_v);
                                const double ground_adv =
                                    slope_x * 0.5 * (
                                        static_cast<double>(
                                            (*advection->du_dt)
                                                [i - 1][j][k]
                                            * dt_large)
                                        + adv_u)
                                    + slope_y * 0.5 * (
                                        static_cast<double>(
                                            (*advection->dv_dt)
                                                [i][j - 1][k]
                                            * dt_large)
                                        + adv_v);
                                accumulate_momentum(
                                    0.5 * moist_cell_mass,
                                    ground_old, ground_new,
                                    ground_delta, ground_adv);
                            }
                        }

                        constexpr double cv_d =
                            physical_constants::specific_heat_cp_jkgk
                            - physical_constants::gas_constant_dry_air_jkgk;
                        constexpr double cv_v =
                            physical_constants::
                                specific_heat_water_vapor_cp_jkgk
                            - physical_constants::
                                gas_constant_water_vapor_jkgk;
                        constexpr double c_l =
                            physical_constants::
                                specific_heat_liquid_water_jkgk;
                        constexpr double c_i =
                            physical_constants::specific_heat_ice_jkgk;
                        constexpr double kappa =
                            physical_constants::gas_constant_dry_air_jkgk
                            / physical_constants::specific_heat_cp_jkgk;
                        const double p_old = fields.p[i][j][k];
                        const double p_new =
                            final_value(p_committed, i, j, k);
                        const double theta_k = fields.theta[i][j][k];
                        const double qv_k = fields.qv[i][j][k];
                        const double q_liquid =
                            static_cast<double>(fields.qc[i][j][k])
                            + fields.qr[i][j][k];
                        const double q_ice =
                            static_cast<double>(fields.qi[i][j][k])
                            + fields.qs[i][j][k]
                            + fields.qg[i][j][k]
                            + fields.qh[i][j][k];
                        const double dry_mass =
                            mass_geometry.scalar_mass_kg(
                                fields.rho, i, j, k);
                        const double old_temperature = theta_k * std::pow(
                            p_old
                                / physical_constants::reference_pressure_pa,
                            kappa);
                        const double new_temperature = theta_k * std::pow(
                            p_new
                                / physical_constants::reference_pressure_pa,
                            kappa);
                        const double delta_temperature =
                            new_temperature - old_temperature;
                        const double water_heat_capacity =
                            qv_k * cv_v + q_liquid * c_l + q_ice * c_i;
                        const double values[] = {
                            p_old, p_new, theta_k, qv_k,
                            q_liquid, q_ice, dry_mass,
                            old_temperature, new_temperature,
                            delta_temperature, water_heat_capacity,
                        };
                        if (p_old <= 0.0 || p_new <= 0.0
                            || theta_k <= 0.0 || qv_k < 0.0
                            || q_liquid < 0.0 || q_ice < 0.0
                            || dry_mass <= 0.0
                            || old_temperature <= 0.0
                            || new_temperature <= 0.0
                            || std::any_of(
                                std::begin(values), std::end(values),
                                [](double value)
                                {
                                    return !std::isfinite(value);
                                }))
                        {
                            partial[11] += 1.0;
                        }
                        else
                        {
                            partial[6] += dry_mass * cv_d
                                * delta_temperature;
                            partial[7] += dry_mass
                                * water_heat_capacity
                                * delta_temperature;
                        }
                    }

                    if (attribute_pbl)
                    {
                        const double scalar_mass =
                            mass_geometry.scalar_mass_kg(
                                fields.rho, i, j, k);
                        const float du_dynamics =
                            finite_or_zero(du_dt[i][j][k]);
                        const float dv_dynamics =
                            finite_or_zero(dv_dt[i][j][k]);
                        const float du_pbl =
                            finite_or_zero(
                                tendencies.du_dt_pbl[i][j][k]);
                        const float dv_pbl =
                            finite_or_zero(
                                tendencies.dv_dt_pbl[i][j][k]);
                        const double delta_u_dynamics =
                            static_cast<double>(du_dynamics * dt_large);
                        const double delta_v_dynamics =
                            static_cast<double>(dv_dynamics * dt_large);
                        const double delta_u_pbl =
                            static_cast<double>(du_pbl * dt_large);
                        const double delta_v_pbl =
                            static_cast<double>(dv_pbl * dt_large);
                        const double u_after_dynamics =
                            static_cast<double>(fields.u[i][j][k])
                            + delta_u_dynamics;
                        const double v_after_dynamics =
                            static_cast<double>(fields.v[i][j][k])
                            + delta_v_dynamics;
                        const float u_dynamics_stored =
                            static_cast<float>(u_after_dynamics);
                        const float v_dynamics_stored =
                            static_cast<float>(v_after_dynamics);
                        const bool owns_u_face =
                            !cgrid || i < domain.i_end - 1 || wrap_x;
                        const bool owns_v_face =
                            !cgrid || j < domain.j_end - 1 || wrap_y;
                        if (owns_u_face)
                        {
                            const int i1 = cgrid
                                ? (i < domain.i_end - 1
                                    ? i + 1 : domain.i_begin)
                                : i;
                            const double u_mass = cgrid
                                ? mass_geometry.u_dual_mass_kg(
                                    fields.rho, i, j, k, i1)
                                : scalar_mass;
                            const double u_new =
                                final_value(u_committed, i, j, k);
                            partial[8] += u_mass
                                * u_after_dynamics * delta_u_pbl;
                            partial[9] += 0.5 * u_mass
                                * delta_u_pbl * delta_u_pbl;
                            partial[10] += 0.5 * u_mass
                                * (u_new * u_new
                                   - static_cast<double>(
                                        u_dynamics_stored)
                                     * u_dynamics_stored);
                        }
                        if (owns_v_face)
                        {
                            const int j1 = cgrid
                                ? (j < domain.j_end - 1
                                    ? j + 1 : domain.j_begin)
                                : j;
                            const double v_mass = cgrid
                                ? mass_geometry.v_dual_mass_kg(
                                    fields.rho, i, j, k, j1)
                                : scalar_mass;
                            const double v_new =
                                final_value(v_committed, i, j, k);
                            partial[8] += v_mass
                                * v_after_dynamics * delta_v_pbl;
                            partial[9] += 0.5 * v_mass
                                * delta_v_pbl * delta_v_pbl;
                            partial[10] += 0.5 * v_mass
                                * (v_new * v_new
                                   - static_cast<double>(
                                        v_dynamics_stored)
                                     * v_dynamics_stored);
                        }
                    }
                }
        });

    if (sums[11] != 0.0
        || std::any_of(
            sums.begin(), sums.begin() + 11,
            [](double value) { return !std::isfinite(value); }))
    {
        throw std::runtime_error(
            "device slow attribution rejected a nonphysical committed state");
    }

    AttributionResult result;
    if (attribute_pbl)
    {
        result.pbl.requested_linear_work_j = sums[8];
        result.pbl.finite_step_correction_j = sums[9];
        result.pbl.committed_work_j = sums[10];
        result.pbl.storage_rounding_j =
            result.pbl.committed_work_j
            - result.pbl.requested_linear_work_j
            - result.pbl.finite_step_correction_j;
    }
    if (attribute_momentum)
    {
        auto& report = result.momentum;
        report.advection_requested_linear_work_j = sums[0];
        report.nonadvection_requested_linear_work_j = sums[1];
        report.advection_finite_step_correction_j = sums[2];
        report.nonadvection_finite_step_correction_j = sums[3];
        report.coupled_finite_step_cross_work_j = sums[4];
        report.committed_kinetic_change_j = sums[5];
        const double requested =
            sums[0] + sums[1] + sums[2] + sums[3] + sums[4];
        report.storage_rounding_j =
            report.committed_kinetic_change_j - requested;
        report.closure_residual_j =
            report.committed_kinetic_change_j
            - (requested + report.storage_rounding_j);
        report.pressure_dry_internal_change_j = sums[6];
        report.pressure_water_sensible_change_j = sums[7];
        report.committed_pressure_energy_change_j = sums[6] + sums[7];
        report.complete = true;
    }
    return result;
}

} // namespace tmv::device_slow_energy
