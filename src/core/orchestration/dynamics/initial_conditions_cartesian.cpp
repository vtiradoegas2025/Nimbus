#include "core/orchestration/dynamics/initial_conditions.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/runtime/simulation_state.hpp"
#include "terrain/conservative_transport.hpp"
#include "terrain/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

// Forward declaration: defined in `src/core/nimbus_sim.cpp`. The function
// returns the (u_x, u_y) Cartesian wind components at altitude `z` from the
// `WindProfile` hodograph. We do not include `nimbus_sim.hpp` (no such
// header exists for this signature) because we want to keep the link
// closure of this file as small as possible.
void compute_wind_profile(const WindProfile& profile, double z, double& u, double& v);

namespace
{

const RuntimeConfigurationState& configuration_for(
    const SimulationStateView& state)
{
    return state.configuration != nullptr
        ? *state.configuration
        : global_simulation_storage.configuration;
}

} // namespace

void apply_cartesian_wind_initialization(const SimulationStateView& state)
{
    const auto& grid = state.grid;
    auto& fields = state.fields;
    const auto& config = configuration_for(state);

    // Precompute the (u_x, u_y) profile by k. This is a tiny vector (NZ
    // entries) but it lets us hoist the wind-profile evaluation out of the
    // (i, j) loop, which dominates work for large grids.
    std::vector<double> u_x_by_k(static_cast<std::size_t>(grid.nz));
    std::vector<double> u_y_by_k(static_cast<std::size_t>(grid.nz));
    const auto& geo = grid.geometry;
    const TerrainMetrics3D* terrain_metrics =
        (state.terrain_runtime != nullptr
         && state.terrain_runtime->config.scheme_id != "none")
        ? &state.terrain_runtime->metrics : nullptr;
    if (terrain_metrics != nullptr)
        terrain_transport::validate_geometry(*terrain_metrics);
    for (int k = 0; k < grid.nz; ++k)
    {
        compute_wind_profile(config.wind_profile, geo.z[k],
                             u_x_by_k[static_cast<std::size_t>(k)],
                             u_y_by_k[static_cast<std::size_t>(k)]);
    }
    if (terrain_metrics != nullptr)
    {
        for (int k = 0; k < grid.nz; ++k)
        {
            if (!std::isfinite(u_x_by_k[static_cast<std::size_t>(k)])
                || !std::isfinite(u_y_by_k[static_cast<std::size_t>(k)]))
                throw std::invalid_argument(
                    "terrain wind initialization requires finite reference wind");
        }
        for (int i = 0; i < grid.nr; ++i)
            for (int j = 0; j < grid.nth; ++j)
                for (int k = 0; k < grid.nz; ++k)
                {
                    const double u_height = (i + 1 < grid.nr)
                        ? 0.5 * (terrain_metrics->z(i, j, k)
                               + terrain_metrics->z(i + 1, j, k))
                        : terrain_metrics->z(i, j, k);
                    const double v_height = (j + 1 < grid.nth)
                        ? 0.5 * (terrain_metrics->z(i, j, k)
                               + terrain_metrics->z(i, j + 1, k))
                        : terrain_metrics->z(i, j, k);
                    (void)terrain_transport::reference_profile_value_fp32(
                        u_x_by_k, u_height, grid.dz);
                    (void)terrain_transport::reference_profile_value_fp32(
                        u_y_by_k, v_height, grid.dz);
                }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < grid.nr; ++i)
    {
        for (int j = 0; j < grid.nth; ++j)
        {
            for (int k = 0; k < grid.nz; ++k)
            {
                const std::size_t kz = static_cast<std::size_t>(k);
                if (terrain_metrics != nullptr)
                {
                    const double u_height = (i + 1 < grid.nr)
                        ? 0.5 * (terrain_metrics->z(i, j, k)
                               + terrain_metrics->z(i + 1, j, k))
                        : terrain_metrics->z(i, j, k);
                    const double v_height = (j + 1 < grid.nth)
                        ? 0.5 * (terrain_metrics->z(i, j, k)
                               + terrain_metrics->z(i, j + 1, k))
                        : terrain_metrics->z(i, j, k);
                    fields.u[i][j][k] = static_cast<float>(
                        terrain_transport::reference_profile_value_fp32(
                            u_x_by_k, u_height, grid.dz));
                    fields.v[i][j][k] = static_cast<float>(
                        terrain_transport::reference_profile_value_fp32(
                            u_y_by_k, v_height, grid.dz));
                }
                else
                {
                    fields.u[i][j][k] = static_cast<float>(u_x_by_k[kz]);
                    fields.v[i][j][k] = static_cast<float>(u_y_by_k[kz]);
                }
                fields.w[i][j][k] = 0.0f;
            }
        }
    }
}

void apply_cartesian_wind_initialization()
{
    apply_cartesian_wind_initialization(global_simulation_state());
}

void apply_cartesian_bubble_initialization(const SimulationStateView& state)
{
    const auto& grid = state.grid;
    auto& fields = state.fields;
    const auto& config = configuration_for(state);

    // Read every per-call parameter into a local before the parallel loop
    // so the OpenMP firstprivate semantics are explicit and we don't reread
    // a global on every iteration. The 100 m radius floor keeps the radius
    // positive, which the rest of the code assumes.
    const double bubble_center_x  = std::max(0.0, config.bubble_center_x_m);
    const double bubble_center_y  = std::max(0.0, config.bubble_center_y_m);
    const double bubble_center_z  = std::max(0.0, config.bubble_center_z_m);
    const double bubble_radius    = std::max(100.0, config.bubble_radius_m);
    const double bubble_dtheta    = config.bubble_dtheta_k;
    const double bubble_cloud_water = config.bubble_cloud_water_kgkg;
    const double bubble_radius_sq = bubble_radius * bubble_radius;
    // The Gaussian is exp(−(dist / (radius / 3))²) which equals
    // exp(−(dist * 3 / radius)²). Caching `3 / radius` avoids two divisions
    // per cell in the inner loop.
    const double inv_third_radius = 3.0 / bubble_radius;
    const auto& geo = grid.geometry;
    const TerrainMetrics3D* terrain_metrics =
        (state.terrain_runtime != nullptr
         && state.terrain_runtime->config.scheme_id != "none")
        ? &state.terrain_runtime->metrics : nullptr;
    if (terrain_metrics != nullptr)
        terrain_transport::validate_geometry(*terrain_metrics);
    const PhysicalCellDomain domain = physical_scalar_domain(grid);
    if (domain.empty())
    {
        return;
    }

    // The warm bubble is a MOIST thermal: when the qv and p base states are
    // present, perturb qv to hold the parcel relative humidity constant as theta
    // warms. A dry warm bubble lowers RH and lifts the LCL above the LFC, so the
    // parcel never reaches free convection (the weak/late-triggering failure);
    // RH preservation makes it initiate. Inline Tetens saturation keeps this file
    // free of the microphysics dependency the header forbids.
    const bool moisten = !fields.qv.empty() && !fields.p.empty()
                       && fields.qv.size() == fields.theta.size()
                       && fields.p.size() == fields.theta.size();
    const bool seed_cloud = bubble_cloud_water > 0.0
        && !fields.qc.empty() && fields.qc.size() == fields.theta.size();
    auto qvsat = [](double T, double P) {
        const double Tc = T - 273.15;
        const double es = 611.2 * std::exp(17.67 * Tc / (Tc + 243.5));
        return 0.622 * es / std::max(P - es, 1.0);
    };

    #pragma omp parallel for collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            const double x_dist = geo.r[i] - bubble_center_x;
            const double y_dist = geo.theta[j] - bubble_center_y;
            const double xy_sq  = x_dist * x_dist + y_dist * y_dist;

            // Cull entire columns whose horizontal distance already exceeds
            // the bubble radius — saves the inner k loop entirely for cells
            // far from the bubble center.
            if (xy_sq > bubble_radius_sq)
            {
                continue;
            }

            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double physical_z = terrain_metrics != nullptr
                    ? terrain_metrics->z(i, j, k) : geo.z[k];
                const double z_dist = physical_z - bubble_center_z;
                const double dist_sq = xy_sq + z_dist * z_dist;
                if (dist_sq > bubble_radius_sq)
                {
                    continue;
                }
                const double dist = std::sqrt(dist_sq);
                const double scaled = dist * inv_third_radius;
                const double bubble_factor = std::exp(-(scaled * scaled));
                const double dtheta = bubble_dtheta * bubble_factor;
                const double theta_old = static_cast<double>(fields.theta[i][j][k]);
                const double theta_new = theta_old + dtheta;
                fields.theta[i][j][k] = static_cast<float>(theta_new);

                if (moisten && dtheta > 0.0)
                {
                    const double P = static_cast<double>(fields.p[i][j][k]);
                    const double exner = std::pow(std::max(P, 1.0) / 100000.0, 0.2857);
                    const double qvs_old = qvsat(theta_old * exner, P);
                    const double qvs_new = qvsat(theta_new * exner, P);
                    if (qvs_old > 1.0e-12)
                    {
                        fields.qv[i][j][k] = static_cast<float>(
                            static_cast<double>(fields.qv[i][j][k]) * (qvs_new / qvs_old));
                    }
                }
                if (seed_cloud)
                {
                    fields.qc[i][j][k] = static_cast<float>(
                        static_cast<double>(fields.qc[i][j][k])
                        + bubble_cloud_water * bubble_factor);
                }
            }
        }
    }
}

void apply_cartesian_bubble_initialization()
{
    apply_cartesian_bubble_initialization(global_simulation_state());
}

void apply_cartesian_terrain_thermodynamic_initialization(
    const SimulationStateView& state,
    const std::vector<double>& reference_pressure,
    const std::vector<double>& reference_theta,
    const std::vector<double>& reference_qv)
{
    const auto& grid = state.grid;
    auto& fields = state.fields;
    if (state.terrain_runtime == nullptr
        || state.terrain_runtime->config.scheme_id == "none")
    {
        throw std::invalid_argument(
            "terrain thermodynamic initialization requires active terrain");
    }
    const TerrainMetrics3D& metrics = state.terrain_runtime->metrics;
    terrain_transport::validate_geometry(metrics);
    const std::size_t levels = static_cast<std::size_t>(grid.nz);
    if (reference_pressure.size() != levels
        || reference_theta.size() != levels
        || reference_qv.size() != levels)
    {
        throw std::invalid_argument(
            "terrain thermodynamic initialization requires full reference profiles");
    }
    for (std::size_t level = 0; level < levels; ++level)
    {
        if (!std::isfinite(reference_pressure[level])
            || reference_pressure[level] <= 0.0
            || !std::isfinite(reference_theta[level])
            || reference_theta[level] <= 0.0
            || !std::isfinite(reference_qv[level])
            || reference_qv[level] < 0.0)
        {
            throw std::invalid_argument(
                "terrain reference thermodynamics must be finite, with positive p/theta and nonnegative qv");
        }
    }
    // Validate every extrapolated/interpolated column value before entering
    // OpenMP; exceptions must never escape a worker thread.
    for (int i = 0; i < grid.nr; ++i)
        for (int j = 0; j < grid.nth; ++j)
            for (int k = 0; k < grid.nz; ++k)
            {
                const double height = metrics.z(i, j, k);
                (void)terrain_transport::reference_density_fp32(
                    reference_pressure, reference_theta, height, grid.dz);
                const double vapor =
                    terrain_transport::reference_profile_value_fp32(
                        reference_qv, height, grid.dz);
                if (vapor < 0.0)
                    throw std::invalid_argument(
                        "terrain reference-vapor interpolation became negative");
            }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < grid.nr; ++i)
    {
        for (int j = 0; j < grid.nth; ++j)
        {
            for (int k = 0; k < grid.nz; ++k)
            {
                const double height = metrics.z(i, j, k);
                fields.p[i][j][k] = static_cast<float>(
                    terrain_transport::reference_profile_value_fp32(
                        reference_pressure, height, grid.dz));
                fields.theta[i][j][k] = static_cast<float>(
                    terrain_transport::reference_profile_value_fp32(
                        reference_theta, height, grid.dz));
                fields.rho[i][j][k] = static_cast<float>(
                    terrain_transport::reference_density_fp32(
                        reference_pressure, reference_theta,
                        height, grid.dz));
                fields.qv[i][j][k] = static_cast<float>(
                    terrain_transport::reference_profile_value_fp32(
                        reference_qv, height, grid.dz));
            }
        }
    }
}
