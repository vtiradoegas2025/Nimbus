#include "weno5.hpp"

#include "numerics/advection/weno5_face.hpp"
#include "util/grid_metric_utils.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

void validate_config(const AdvectionConfig& config)
{
    if (!std::isfinite(config.positivity_dt)
        || config.positivity_dt <= 0.0
        || !std::isfinite(config.cfl_target)
        || config.cfl_target <= 0.0
        || config.weno_order != 5)
    {
        throw std::invalid_argument(
            "WENO5 advection requires order 5 and positive finite positivity_dt/cfl_target");
    }
}

void reject_mapped_terrain(const GridMetrics& grid)
{
    if (grid.terrain_metrics_active)
    {
        throw std::invalid_argument(
            "WENO5 one-dimensional vertical transport cannot use terrain-following metrics with physical w; use the Cartesian C-grid mapped conservative transport path");
    }
}

void require_shape(const Field3D& field,
                   int nr,
                   int nth,
                   int nz,
                   const char* label)
{
    if (field.size_r() != nr || field.size_th() != nth || field.size_z() != nz)
    {
        throw std::invalid_argument(
            std::string("WENO5 ") + label + " shape disagrees with q");
    }
}

void require_finite(const Field3D& field, const char* label)
{
    for (std::size_t index = 0; index < field.size(); ++index)
    {
        if (!std::isfinite(field.data()[index]))
        {
            throw std::invalid_argument(
                std::string("WENO5 received non-finite ") + label);
        }
    }
}

void validate_state(const AdvectionStateView& state, bool require_vertical_velocity)
{
    if (state.q == nullptr || state.grid == nullptr
        || (require_vertical_velocity && state.w == nullptr))
    {
        throw std::invalid_argument("WENO5 advection state is incomplete");
    }

    const int nr = state.q->size_r();
    const int nth = state.q->size_th();
    const int nz = state.q->size_z();
    if (nr <= 0 || nth <= 0 || nz <= 0)
    {
        throw std::invalid_argument("WENO5 q shape is empty");
    }

    reject_mapped_terrain(*state.grid);
    grid_metric::validate_vertical_metrics(*state.grid, nr, nth, nz);
    require_finite(*state.q, "q");

    bool has_velocity = false;
    for (const auto [field, label] : {
             std::pair<const Field3D*, const char*>{state.u, "u"},
             {state.v, "v"},
             {state.w, "w"}})
    {
        if (field == nullptr)
            continue;
        has_velocity = true;
        require_shape(*field, nr, nth, nz, label);
        require_finite(*field, label);
    }
    if (!has_velocity)
    {
        throw std::invalid_argument("WENO5 advection requires a velocity field");
    }
}

std::vector<double> vertical_spacing(
    const GridMetrics& grid, int i, int j, int nz)
{
    std::vector<double> result(static_cast<std::size_t>(nz));
    for (int k = 0; k < nz; ++k)
    {
        result[static_cast<std::size_t>(k)] =
            grid_metric::vertical_cell_thickness(grid, i, j, k, nz);
    }
    return result;
}

} // namespace

namespace tmv
{

void WENO5Scheme::initialize()
{
    initialize(AdvectionConfig{});
}

void WENO5Scheme::initialize(const AdvectionConfig& config)
{
    validate_config(config);
    tmv::log_info("Initialized WENO5 advection scheme");
    if (config.positivity)
        tmv::log_info("  Positivity preservation enabled");
}

void WENO5Scheme::compute_flux_divergence(
    const AdvectionConfig& config,
    const AdvectionStateView& state,
    AdvectionTendencies& tendencies,
    AdvectionDiagnostics* diagnostics)
{
    validate_config(config);
    validate_state(state, true);

    const int nr = state.q->size_r();
    const int nth = state.q->size_th();
    const int nz = state.q->size_z();
    Field3D candidate(nr, nth, nz, 0.0f);
    double max_cfl = 0.0;

#pragma omp parallel for collapse(2) reduction(max:max_cfl) schedule(static)
    for (int i = 0; i < nr; ++i)
    {
        for (int j = 0; j < nth; ++j)
        {
            std::vector<double> q_column(static_cast<std::size_t>(nz));
            std::vector<double> w_column(static_cast<std::size_t>(nz));
            for (int k = 0; k < nz; ++k)
            {
                q_column[static_cast<std::size_t>(k)] = (*state.q)[i][j][k];
                w_column[static_cast<std::size_t>(k)] = (*state.w)[i][j][k];
            }

            const std::vector<double> dz_column =
                vertical_spacing(*state.grid, i, j, nz);
            std::vector<double> tendency_column;
            advect_column(q_column, w_column, dz_column,
                          config.positivity_dt, config.positivity,
                          tendency_column);

            for (int k = 0; k < nz; ++k)
            {
                candidate[i][j][k] = static_cast<float>(
                    tendency_column[static_cast<std::size_t>(k)]);
                max_cfl = std::max(
                    max_cfl,
                    std::abs(w_column[static_cast<std::size_t>(k)])
                        / dz_column[static_cast<std::size_t>(k)]);
            }
        }
    }

    tendencies.dqdt_adv = std::move(candidate);
    if (diagnostics != nullptr)
    {
        diagnostics->max_cfl_z = max_cfl;
        diagnostics->suggested_dt = max_cfl > 1.0e-12
            ? config.cfl_target / max_cfl
            : std::numeric_limits<double>::infinity();
    }
}

double WENO5Scheme::suggest_dt(
    const AdvectionConfig& config,
    const AdvectionStateView& state)
{
    validate_config(config);
    validate_state(state, false);

    const int nr = state.q->size_r();
    const int nth = state.q->size_th();
    const int nz = state.q->size_z();
    double max_rate = 0.0;

#pragma omp parallel for collapse(3) reduction(max:max_rate) schedule(static)
    for (int i = 0; i < nr; ++i)
    {
        for (int j = 0; j < nth; ++j)
        {
            for (int k = 0; k < nz; ++k)
            {
                double rate = 0.0;
                if (state.u != nullptr)
                    rate += std::abs(static_cast<double>((*state.u)[i][j][k]))
                        / grid_metric::local_dx(*state.grid, i, j, k);
                if (state.v != nullptr)
                    rate += std::abs(static_cast<double>((*state.v)[i][j][k]))
                        / grid_metric::local_dy(*state.grid, i, j, k);
                if (state.w != nullptr)
                    rate += std::abs(static_cast<double>((*state.w)[i][j][k]))
                        / grid_metric::vertical_cell_thickness(
                            *state.grid, i, j, k, nz);
                max_rate = std::max(max_rate, rate);
            }
        }
    }

    return max_rate > 1.0e-12 ? config.cfl_target / max_rate : 1.0;
}

double WENO5Scheme::reconstruct_left(const std::vector<double>& q, int i)
{
    if (i < 2 || i + 2 >= static_cast<int>(q.size()))
        return q[static_cast<std::size_t>(i)];
    return weno5_face_value(q[static_cast<std::size_t>(i - 2)],
                            q[static_cast<std::size_t>(i - 1)],
                            q[static_cast<std::size_t>(i)],
                            q[static_cast<std::size_t>(i + 1)],
                            q[static_cast<std::size_t>(i + 2)]);
}

double WENO5Scheme::reconstruct_right(const std::vector<double>& q, int i)
{
    if (i < 1 || i + 3 >= static_cast<int>(q.size()))
        return q[static_cast<std::size_t>(i + 1)];
    return weno5_face_value(q[static_cast<std::size_t>(i + 3)],
                            q[static_cast<std::size_t>(i + 2)],
                            q[static_cast<std::size_t>(i + 1)],
                            q[static_cast<std::size_t>(i)],
                            q[static_cast<std::size_t>(i - 1)]);
}

double WENO5Scheme::rusanov_flux(
    double q_left, double q_right, double velocity)
{
    return 0.5 * velocity * (q_left + q_right)
        - 0.5 * std::abs(velocity) * (q_right - q_left);
}

void WENO5Scheme::advect_column(
    const std::vector<double>& q,
    const std::vector<double>& velocity,
    const std::vector<double>& dz,
    double dt,
    bool positivity,
    std::vector<double>& dqdt)
{
    const int count = static_cast<int>(q.size());
    dqdt.assign(q.size(), 0.0);
    std::vector<double> flux(q.size() + 1, 0.0);

    for (int i = 0; i + 1 < count; ++i)
    {
        flux[static_cast<std::size_t>(i + 1)] = rusanov_flux(
            reconstruct_left(q, i), reconstruct_right(q, i),
            velocity[static_cast<std::size_t>(i)]);
    }
    for (int i = 0; i < count; ++i)
    {
        dqdt[static_cast<std::size_t>(i)] =
            -(flux[static_cast<std::size_t>(i + 1)]
              - flux[static_cast<std::size_t>(i)])
            / dz[static_cast<std::size_t>(i)];
    }
    if (positivity)
        apply_positivity(q, dqdt, dt);
}

void WENO5Scheme::apply_positivity(
    const std::vector<double>& q,
    std::vector<double>& dqdt,
    double dt,
    double minimum)
{
    for (std::size_t index = 0; index < q.size(); ++index)
    {
        if (!std::isfinite(dqdt[index]))
        {
            throw std::runtime_error(
                "WENO5 positivity limiter received a non-finite tendency");
        }
        dqdt[index] = std::max(dqdt[index], (minimum - q[index]) / dt);
    }
}

} // namespace tmv
