#include "core/runtime/process_state.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/infra/physical_constants.hpp"
#include "numerics/runtime.hpp"
#include "numerics/advection/advection_base.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "numerics/diffusion/linear_height_profile.hpp"
#include "numerics/advection/factory.hpp"
#include "numerics/diffusion/factory.hpp"
#include "numerics/time_stepping/factory.hpp"
#include "terrain/conservative_transport.hpp"
#include "terrain/terrain_base.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "turbulence/runtime.hpp"
#include "core/field/field_bounds.hpp"
#include "util/log.hpp"
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

double require_positive(double value, const char* name)
{
    if (!std::isfinite(value) || value <= 0.0)
    {
        throw std::invalid_argument(
            std::string(name) + " must be positive and finite");
    }
    return value;
}

struct FlowCflScan
{
    double max_component_speed = 0.0;
    double max_directional_rate = 0.0;
    double max_acoustic_speed = 0.0;
    double min_acoustic_spacing = 0.0;
};

FlowCflScan compute_flow_cfl_scan(const SimulationStateView& execution_state, double dx, double dy,
                                  const std::vector<double>& dz_levels,
                                  double dz_fallback,
                                  double cartesian_min_spacing)
{
    const int nr = execution_state.grid.nr;
    const int nth = execution_state.grid.nth;
    const int nz = execution_state.grid.nz;
    if (nr <= 0 || nth <= 0 || nz <= 0)
    {
        throw std::invalid_argument(
            "CFL scan requires positive grid dimensions");
    }
    if (execution_state.fields.u.size_r() != nr || execution_state.fields.u.size_th() != nth || execution_state.fields.u.size_z() != nz ||
        execution_state.fields.v.size_r() != nr || execution_state.fields.v.size_th() != nth || execution_state.fields.v.size_z() != nz ||
        execution_state.fields.w.size_r() != nr || execution_state.fields.w.size_th() != nth || execution_state.fields.w.size_z() != nz ||
        execution_state.fields.p.size_r() != nr || execution_state.fields.p.size_th() != nth || execution_state.fields.p.size_z() != nz ||
        execution_state.fields.rho.size_r() != nr || execution_state.fields.rho.size_th() != nth || execution_state.fields.rho.size_z() != nz)
    {
        throw std::invalid_argument(
            "CFL scan state field shape does not match the grid");
    }
    if (dz_levels.size() != static_cast<std::size_t>(nz))
        throw std::invalid_argument(
            "CFL scan vertical metric count does not match grid.nz");

    const double inv_dx = (nr > 1) ? 1.0 / require_positive(dx, "grid dx") : 0.0;
    const double inv_dy = (nth > 1) ? 1.0 / require_positive(dy, "grid dy") : 0.0;
    require_positive(dz_fallback, "grid dz");
    for (double dz_level : dz_levels)
        require_positive(dz_level, "vertical grid metric");

    const TerrainMetrics3D* terrain = nullptr;
    if (tmv::require_level_runtime(execution_state.process_grid_metrics, "grid metrics").terrain_metrics_active)
    {
        terrain = tmv::require_level_runtime(execution_state.process_grid_metrics, "grid metrics").terrain_metrics;
        if (terrain == nullptr)
            throw std::invalid_argument(
                "active terrain CFL requires bound conservative metrics");
        terrain_transport::validate_geometry(*terrain);
        if (terrain->J.size_r() != nr || terrain->J.size_th() != nth ||
            terrain->J.size_z() != nz)
            throw std::invalid_argument(
                "terrain CFL metrics do not match the runtime grid");
        if (nr < 3 || nth < 3)
            throw std::invalid_argument(
                "terrain CFL requires horizontal boundary storage");
    }

    const std::size_t cells = execution_state.fields.u.size();
    bool parallel_validation = false;
#ifdef _OPENMP
    parallel_validation = cells >= 65536 && omp_get_max_threads() > 1 && !omp_in_parallel();
#endif
    if (parallel_validation)
    {
        const float* u_values = execution_state.fields.u.data();
        const float* v_values = execution_state.fields.v.data();
        const float* w_values = execution_state.fields.w.data();
        const float* p_values = execution_state.fields.p.data();
        const float* rho_values = execution_state.fields.rho.data();
        int invalid_state = 0;
        #pragma omp parallel for schedule(static) reduction(|:invalid_state)
        for (std::size_t index = 0; index < cells; ++index)
        {
            const double values[] = {static_cast<double>(u_values[index]),
                static_cast<double>(v_values[index]), static_cast<double>(w_values[index]),
                static_cast<double>(p_values[index]), static_cast<double>(rho_values[index])};
            invalid_state |= !std::isfinite(values[0]) || !std::isfinite(values[1]) ||
                !std::isfinite(values[2]) || !std::isfinite(values[3]) || values[3] <= 0.0 ||
                !std::isfinite(values[4]) || values[4] <= 0.0;
        }
        if (invalid_state)
            throw std::invalid_argument("CFL scan requires finite wind and positive finite pressure/density");
    }
    else
    {
        for (int i = 0; i < nr; ++i)
        {
            for (int j = 0; j < nth; ++j)
            {
                for (int k = 0; k < nz; ++k)
                {
                    const double values[] = {
                        static_cast<double>(execution_state.fields.u[i][j][k]),
                        static_cast<double>(execution_state.fields.v[i][j][k]),
                        static_cast<double>(execution_state.fields.w[i][j][k]),
                        static_cast<double>(execution_state.fields.p[i][j][k]),
                        static_cast<double>(execution_state.fields.rho[i][j][k])};
                    if (!std::isfinite(values[0]) || !std::isfinite(values[1]) ||
                        !std::isfinite(values[2]) || !std::isfinite(values[3]) ||
                        values[3] <= 0.0 || !std::isfinite(values[4]) ||
                        values[4] <= 0.0)
                    {
                        throw std::invalid_argument(
                            "CFL scan requires finite wind and positive finite pressure/density");
                    }
                }
            }
        }
    }

    double max_component = 0.0;
    double max_directional_rate = 0.0;
    double max_acoustic_speed = 0.0;
    double min_acoustic_spacing = terrain == nullptr
        ? require_positive(cartesian_min_spacing, "minimum Cartesian spacing")
        : std::numeric_limits<double>::infinity();

    if (terrain != nullptr)
    {
        const bool collocated = tmv::require_run_configuration(execution_state).stagger_type == StaggerType::Collocated;
        if (collocated)
        {
            #pragma omp parallel for collapse(2) reduction(max:max_component,max_directional_rate,max_acoustic_speed) reduction(min:min_acoustic_spacing)
            for (int i = 1; i < nr - 1; ++i)
                for (int j = 1; j < nth - 1; ++j)
                    for (int k = 1; k < nz - 1; ++k)
                    {
                        const double jacobian = terrain->J(i, j, k);
                        const double u_here = static_cast<double>(execution_state.fields.u[i][j][k]);
                        const double v_here = static_cast<double>(execution_state.fields.v[i][j][k]);
                        const double w_here = static_cast<double>(execution_state.fields.w[i][j][k]);
                        const double normal = w_here
                            + jacobian * terrain->mx(i, j, k) * u_here
                            + jacobian * terrain->my(i, j, k) * v_here;
                        const double rate = std::abs(u_here) / terrain->dx
                            + std::abs(v_here) / terrain->dy
                            + std::abs(normal)
                                / (jacobian * terrain->dzeta);
                        max_directional_rate = std::max(
                            max_directional_rate, rate);
                        max_component = std::max(max_component, std::max({
                            std::abs(u_here), std::abs(v_here),
                            std::abs(w_here)}));
                        const double area_lower = std::sqrt(
                            1.0 + terrain->Gx_face(i, j, k)
                                * terrain->Gx_face(i, j, k)
                            + terrain->Gy_face(i, j, k)
                                * terrain->Gy_face(i, j, k));
                        const double area_upper = std::sqrt(
                            1.0 + terrain->Gx_face(i, j, k + 1)
                                * terrain->Gx_face(i, j, k + 1)
                            + terrain->Gy_face(i, j, k + 1)
                                * terrain->Gy_face(i, j, k + 1));
                        const double inverse_spacing = std::max({
                            1.0 / terrain->dx,
                            1.0 / terrain->dy,
                            std::max(area_lower, area_upper)
                                / (jacobian * terrain->dzeta)});
                        min_acoustic_spacing = std::min(
                            min_acoustic_spacing, 1.0 / inverse_spacing);
                        max_acoustic_speed = std::max(
                            max_acoustic_speed,
                            std::sqrt(physical_constants::dry_air_gamma
                                * static_cast<double>(execution_state.fields.p[i][j][k])
                                / static_cast<double>(execution_state.fields.rho[i][j][k])));
                    }
            return {max_component, max_directional_rate,
                    max_acoustic_speed,
                    require_positive(min_acoustic_spacing,
                                     "terrain acoustic spacing")};
        }
        const auto u_center = [&](int i, int j, int k)
        {
            return collocated
                ? static_cast<double>(execution_state.fields.u[i][j][k])
                : 0.5 * (static_cast<double>(execution_state.fields.u[i - 1][j][k]) +
                         static_cast<double>(execution_state.fields.u[i][j][k]));
        };
        const auto v_center = [&](int i, int j, int k)
        {
            return collocated
                ? static_cast<double>(execution_state.fields.v[i][j][k])
                : 0.5 * (static_cast<double>(execution_state.fields.v[i][j - 1][k]) +
                         static_cast<double>(execution_state.fields.v[i][j][k]));
        };
        const auto zeta_volume_flux = [&](int i, int j, int owner_k)
        {
            const double u_face = owner_k + 1 < nz
                ? 0.5 * (u_center(i, j, owner_k) +
                         u_center(i, j, owner_k + 1))
                : u_center(i, j, owner_k);
            const double v_face = owner_k + 1 < nz
                ? 0.5 * (v_center(i, j, owner_k) +
                         v_center(i, j, owner_k + 1))
                : v_center(i, j, owner_k);
            const int face_k = owner_k + 1;
            const double vertical = collocated && owner_k + 1 < nz
                ? 0.5 * (static_cast<double>(execution_state.fields.w[i][j][owner_k])
                         + static_cast<double>(execution_state.fields.w[i][j][owner_k + 1]))
                : static_cast<double>(execution_state.fields.w[i][j][owner_k]);
            return vertical +
                terrain->Gx_face(i, j, face_k) * u_face +
                terrain->Gy_face(i, j, face_k) * v_face;
        };

        #pragma omp parallel for collapse(2) reduction(max:max_component,max_directional_rate,max_acoustic_speed) reduction(min:min_acoustic_spacing)
        for (int i = 1; i < nr - 1; ++i)
        {
            for (int j = 1; j < nth - 1; ++j)
            {
                for (int k = collocated ? 1 : 0;
                     k < (collocated ? nz - 1 : nz); ++k)
                {
                    const double jacobian = terrain->J(i, j, k);
                    const double x_positive = terrain->Jx_face(i + 1, j, k) *
                        (collocated
                            ? 0.5 * (static_cast<double>(execution_state.fields.u[i][j][k])
                                   + static_cast<double>(execution_state.fields.u[i + 1][j][k]))
                            : static_cast<double>(execution_state.fields.u[i][j][k]));
                    const double x_negative = terrain->Jx_face(i, j, k) *
                        (collocated
                            ? 0.5 * (static_cast<double>(execution_state.fields.u[i - 1][j][k])
                                   + static_cast<double>(execution_state.fields.u[i][j][k]))
                            : static_cast<double>(execution_state.fields.u[i - 1][j][k]));
                    const double y_positive = terrain->Jy_face(i, j + 1, k) *
                        (collocated
                            ? 0.5 * (static_cast<double>(execution_state.fields.v[i][j][k])
                                   + static_cast<double>(execution_state.fields.v[i][j + 1][k]))
                            : static_cast<double>(execution_state.fields.v[i][j][k]));
                    const double y_negative = terrain->Jy_face(i, j, k) *
                        (collocated
                            ? 0.5 * (static_cast<double>(execution_state.fields.v[i][j - 1][k])
                                   + static_cast<double>(execution_state.fields.v[i][j][k]))
                            : static_cast<double>(execution_state.fields.v[i][j - 1][k]));
                    const double z_positive =
                        collocated && k == nz - 2
                            ? 0.0 : zeta_volume_flux(i, j, k);
                    const double z_negative =
                        collocated && k == 1
                            ? 0.0
                            : (k > 0 ? zeta_volume_flux(i, j, k - 1) : 0.0);

                    const double rate_x = std::max(
                        std::abs(x_positive), std::abs(x_negative)) /
                        (jacobian * terrain->dx);
                    const double rate_y = std::max(
                        std::abs(y_positive), std::abs(y_negative)) /
                        (jacobian * terrain->dy);
                    const double rate_z = std::max(
                        std::abs(z_positive), std::abs(z_negative)) /
                        (jacobian * terrain->dzeta);
                    max_directional_rate = std::max(
                        max_directional_rate, rate_x + rate_y + rate_z);
                    max_component = std::max(max_component, std::max({
                        std::abs(static_cast<double>(execution_state.fields.u[i][j][k])),
                        std::abs(static_cast<double>(execution_state.fields.v[i][j][k])),
                        std::abs(static_cast<double>(execution_state.fields.w[i][j][k]))}));

                    const auto zeta_area_factor = [&](int face_k)
                    {
                        return std::sqrt(
                            1.0 +
                            terrain->Gx_face(i, j, face_k) *
                                terrain->Gx_face(i, j, face_k) +
                            terrain->Gy_face(i, j, face_k) *
                                terrain->Gy_face(i, j, face_k));
                    };
                    const double inverse_x = std::max(
                        terrain->Jx_face(i, j, k),
                        terrain->Jx_face(i + 1, j, k)) /
                        (jacobian * terrain->dx);
                    const double inverse_y = std::max(
                        terrain->Jy_face(i, j, k),
                        terrain->Jy_face(i, j + 1, k)) /
                        (jacobian * terrain->dy);
                    const double inverse_z = std::max(
                        zeta_area_factor(k), zeta_area_factor(k + 1)) /
                        (jacobian * terrain->dzeta);
                    const double inverse_spacing = std::max({
                        inverse_x, inverse_y, inverse_z});
                    min_acoustic_spacing = std::min(
                        min_acoustic_spacing, 1.0 / inverse_spacing);

                    const double p_here = static_cast<double>(execution_state.fields.p[i][j][k]);
                    const double rho_here = static_cast<double>(execution_state.fields.rho[i][j][k]);
                    max_acoustic_speed = std::max(
                        max_acoustic_speed,
                        std::sqrt(physical_constants::dry_air_gamma *
                                  p_here / rho_here));
                }
            }
        }
        return {max_component, max_directional_rate, max_acoustic_speed,
                require_positive(min_acoustic_spacing,
                                 "terrain acoustic spacing")};
    }

    #pragma omp parallel for collapse(2) reduction(max:max_component,max_directional_rate,max_acoustic_speed)
    for (int i = 0; i < nr; ++i)
    {
        for (int j = 0; j < nth; ++j)
        {
            for (int k = 0; k < nz; ++k)
            {
                const double u_here = std::abs(static_cast<double>(execution_state.fields.u[i][j][k]));
                const double v_here = std::abs(static_cast<double>(execution_state.fields.v[i][j][k]));
                const double w_here = std::abs(static_cast<double>(execution_state.fields.w[i][j][k]));

                // u/v/w are face values on the production C-grid.  A cell's
                // conservative update sees both of its faces in each active
                // direction, so use the larger adjacent face magnitude.  On
                // the collocated compatibility grid this is a conservative
                // one-cell envelope of the same directional CFL condition.
                const double u_cell = (nr > 1 && i > 0)
                    ? std::max(u_here, std::abs(static_cast<double>(execution_state.fields.u[i - 1][j][k])))
                    : u_here;
                const double v_cell = (nth > 1 && j > 0)
                    ? std::max(v_here, std::abs(static_cast<double>(execution_state.fields.v[i][j - 1][k])))
                    : v_here;
                const double w_cell = (nz > 1 && k > 0)
                    ? std::max(w_here, std::abs(static_cast<double>(execution_state.fields.w[i][j][k - 1])))
                    : w_here;

                double dz_local = dz_fallback;
                if (static_cast<std::size_t>(k) < dz_levels.size())
                {
                    dz_local = dz_levels[static_cast<std::size_t>(k)];
                }
                const double inv_dz = (nz > 1)
                    ? 1.0 / dz_local
                    : 0.0;

                const double directional_rate =
                    u_cell * inv_dx + v_cell * inv_dy + w_cell * inv_dz;
                max_component = std::max(max_component,
                                         std::max({u_here, v_here, w_here}));
                max_directional_rate = std::max(max_directional_rate,
                                                directional_rate);

                // Fold the acoustic-speed scan into the existing CFL pass so
                // split-substep safety does not require another full-domain
                // traversal. The fast equations use c^2 = gamma*p/rho.
                const double p_here = static_cast<double>(execution_state.fields.p[i][j][k]);
                const double rho_here = static_cast<double>(execution_state.fields.rho[i][j][k]);
                max_acoustic_speed = std::max(
                    max_acoustic_speed,
                    std::sqrt(physical_constants::dry_air_gamma
                              * p_here / rho_here));
            }
        }
    }
    return {max_component, max_directional_rate, max_acoustic_speed,
            min_acoustic_spacing};
}

bool terrain_metrics_ready_for_grid(const SimulationStateView& execution_state)
{
    return tmv::require_level_runtime(execution_state.terrain_runtime, "terrain_runtime").metrics.z.size_r() == execution_state.grid.nr &&
           tmv::require_level_runtime(execution_state.terrain_runtime, "terrain_runtime").metrics.z.size_th() == execution_state.grid.nth &&
           tmv::require_level_runtime(execution_state.terrain_runtime, "terrain_runtime").metrics.z.size_z() == execution_state.grid.nz &&
           tmv::require_level_runtime(execution_state.terrain_runtime, "terrain_runtime").metrics.J.size_r() == execution_state.grid.nr &&
           tmv::require_level_runtime(execution_state.terrain_runtime, "terrain_runtime").metrics.J.size_th() == execution_state.grid.nth &&
           tmv::require_level_runtime(execution_state.terrain_runtime, "terrain_runtime").metrics.J.size_z() == execution_state.grid.nz;
}
}

namespace tmv
{

std::unique_ptr<AdvectionSchemeBase>& advection_scheme =
    global_advection_runtime_state.scheme;
AdvectionConfig& global_advection_config =
    global_advection_runtime_state.config;
std::unique_ptr<DiffusionSchemeBase>& diffusion_scheme =
    global_diffusion_runtime_state.scheme;
DiffusionConfig& global_diffusion_config =
    global_diffusion_runtime_state.config;
std::unique_ptr<TimeSteppingSchemeBase>& time_stepping_scheme =
    global_time_stepping_runtime_state.scheme;
TimeSteppingConfig& global_time_stepping_config =
    global_time_stepping_runtime_state.config;

GridMetrics& global_grid_metrics =
    global_simulation_storage.process_grid_metrics;

void initialize_advection(AdvectionRuntimeState& runtime,
                          const std::string& scheme_name,
                          const AdvectionConfig& cfg)
{
    AdvectionConfig selected_config = cfg;
    selected_config.scheme_id = lower_copy(
        scheme_name.empty() ? selected_config.scheme_id : scheme_name);
    if (selected_config.scheme_id.empty())
    {
        selected_config.scheme_id = "tvd";
    }

    std::unique_ptr<AdvectionSchemeBase> selected =
        create_advection_scheme(selected_config.scheme_id);
    selected_config.scheme_id = selected->name();
    selected->initialize(selected_config);

    AdvectionRuntimeState initialized;
    initialized.scheme = std::move(selected);
    initialized.config = std::move(selected_config);
    runtime = std::move(initialized);
}

void initialize_advection(const std::string& scheme_name,
                          const AdvectionConfig& cfg)
{
    return initialize_advection(global_simulation_state(), scheme_name, cfg);
}

void initialize_advection(const SimulationStateView& execution_state, const std::string& scheme_name,
                          const AdvectionConfig& cfg)
{
    initialize_advection(tmv::require_level_runtime(execution_state.advection_runtime, "advection_runtime"), scheme_name, cfg);
}

void initialize_diffusion(DiffusionRuntimeState& runtime,
                          const std::string& scheme_name,
                          const DiffusionConfig& cfg)
{
    DiffusionConfig selected_config = cfg;
    selected_config.scheme_id = lower_copy(
        scheme_name.empty() ? selected_config.scheme_id : scheme_name);
    if (selected_config.scheme_id.empty())
    {
        selected_config.scheme_id = "explicit";
    }

    std::unique_ptr<DiffusionSchemeBase> selected =
        create_diffusion_scheme(selected_config.scheme_id);
    selected_config.scheme_id = selected->name();
    selected->initialize(selected_config);

    DiffusionRuntimeState initialized;
    initialized.scheme = std::move(selected);
    initialized.config = std::move(selected_config);
    runtime = std::move(initialized);
}

void initialize_diffusion(const std::string& scheme_name,
                          const DiffusionConfig& cfg)
{
    return initialize_diffusion(global_simulation_state(), scheme_name, cfg);
}

void initialize_diffusion(const SimulationStateView& execution_state, const std::string& scheme_name,
                          const DiffusionConfig& cfg)
{
    initialize_diffusion(tmv::require_level_runtime(execution_state.diffusion_runtime, "diffusion_runtime"), scheme_name, cfg);
}

void initialize_time_stepping(TimeSteppingRuntimeState& runtime,
                              const std::string& scheme_name,
                              const TimeSteppingConfig& cfg,
                              RHSFunction rhs_func)
{
    TimeSteppingConfig selected_config = cfg;
    selected_config.scheme_id = lower_copy(
        scheme_name.empty() ? selected_config.scheme_id : scheme_name);
    if (selected_config.scheme_id.empty())
    {
        selected_config.scheme_id = "rk3";
    }

    std::unique_ptr<TimeSteppingSchemeBase> selected =
        create_time_stepping_scheme(selected_config.scheme_id);
    selected_config.scheme_id = selected->name();
    validate_runtime_time_stepping_contract(selected_config, *selected);
    selected->initialize(selected_config, std::move(rhs_func));

    TimeSteppingRuntimeState initialized;
    initialized.scheme = std::move(selected);
    initialized.config = std::move(selected_config);
    runtime = std::move(initialized);
}

void initialize_time_stepping(const std::string& scheme_name,
                              const TimeSteppingConfig& cfg,
                              RHSFunction rhs_func)
{
    return initialize_time_stepping(global_simulation_state(), scheme_name, cfg, rhs_func);
}

void initialize_time_stepping(const SimulationStateView& execution_state, const std::string& scheme_name,
                              const TimeSteppingConfig& cfg,
                              RHSFunction rhs_func)
{
    initialize_time_stepping(tmv::require_level_runtime(execution_state.time_stepping_runtime, "time_stepping_runtime"),
                             scheme_name, cfg, std::move(rhs_func));
}

void refresh_grid_metrics_from_terrain()
{
    return refresh_grid_metrics_from_terrain(global_simulation_state());
}

void refresh_grid_metrics_from_terrain(const SimulationStateView& execution_state)
{
    const int nr = execution_state.grid.nr;
    const int nth = execution_state.grid.nth;
    const int nz = execution_state.grid.nz;
    if (nr <= 0 || nth <= 0 || nz <= 0)
        throw std::invalid_argument(
            "grid dimensions must be positive before metric construction");
    const double dx_safe = require_positive(execution_state.grid.dr, "grid dx");
    // Cartesian y-spacing: dtheta is already the physical y-spacing in metres
    // (it is what the grid.dy config key sets), so the representative dy IS
    // dtheta. See representative_dy() in coordinate_system.hpp.
    const double dy_safe = require_positive(
        representative_dy(tmv::require_run_configuration(execution_state).coordinate_system, execution_state.grid.dr, execution_state.grid.dtheta), "grid dy");
    const double dz_safe = require_positive(execution_state.grid.dz, "grid dz");

    // Construct a complete candidate. A malformed active-terrain geometry
    // must not leave the live CPU metric cache partially reset to flat values.
    GridMetrics candidate;
    candidate.dx = dx_safe;
    candidate.dy = dy_safe;
    candidate.staggering = tmv::require_run_configuration(execution_state).stagger_type;
    candidate.dz.assign(nz, dz_safe);
    candidate.z_int.assign(nz + 1, 0.0);

    // For flat terrain (scheme "none"), the metrics arrays are identity
    // (J=1.0, scalar z=(k+1/2)*dz) and don't need to be activated. Keeping them
    // inactive enables GPU compute dispatch for dynamics kernels, which
    // assume uniform grid spacing. Only activate for real terrain schemes
    // (bell, schar, etc.) where metric corrections are needed.
    const bool flat_terrain = (tmv::require_level_runtime(execution_state.terrain_runtime, "terrain_runtime").config.scheme_id == "none");

    if (flat_terrain)
    {
        const auto& geometry_faces =
            execution_state.grid.geometry.z_faces;
        if (geometry_faces.size() == static_cast<std::size_t>(nz + 1))
        {
            candidate.z_int = geometry_faces;
            for (int k = 0; k < nz; ++k)
            {
                const double thickness =
                    geometry_faces[static_cast<std::size_t>(k + 1)] -
                    geometry_faces[static_cast<std::size_t>(k)];
                candidate.dz[static_cast<std::size_t>(k)] =
                    require_positive(
                        thickness, "flat vertical layer thickness");
            }
        }
        else
        {
            for (int k = 1; k <= nz; ++k)
                candidate.z_int[k] = candidate.z_int[k - 1] + dz_safe;
        }
        // Retain identity terrain storage for consumers that explicitly ask
        // for it, but do not activate metric corrections or disable flat GPU
        // kernels.
        candidate.terrain_metrics = &tmv::require_level_runtime(execution_state.terrain_runtime, "terrain_runtime").metrics;
        candidate.terrain_topography = &tmv::require_level_runtime(execution_state.terrain_runtime, "terrain_runtime").topography;
        tmv::require_level_runtime(execution_state.process_grid_metrics, "grid metrics") = std::move(candidate);
        return;
    }
    if (!terrain_metrics_ready_for_grid(execution_state))
    {
        throw std::invalid_argument(
            "active terrain metrics do not match the simulation grid");
    }
    terrain_transport::validate_geometry(tmv::require_level_runtime(execution_state.terrain_runtime, "terrain_runtime").metrics);

    candidate.terrain_metrics_active = true;
    candidate.terrain_metrics = &tmv::require_level_runtime(execution_state.terrain_runtime, "terrain_runtime").metrics;
    candidate.terrain_topography = &tmv::require_level_runtime(execution_state.terrain_runtime, "terrain_runtime").topography;

    for (int i = 0; i < nr; ++i)
    {
        for (int j = 0; j < nth; ++j)
        {
            for (int k = 0; k < nz; ++k)
            {
                const double z_here = tmv::require_level_runtime(execution_state.terrain_runtime, "terrain_runtime").metrics.z(i, j, k);
                if (!std::isfinite(z_here) ||
                    (k > 0 &&
                     z_here <= tmv::require_level_runtime(execution_state.terrain_runtime, "terrain_runtime").metrics.z(i, j, k - 1)))
                {
                    throw std::invalid_argument(
                        "terrain scalar heights must be finite and strictly increasing");
                }
            }
        }
    }

    // One region over independent levels avoids launching an OpenMP team nz
    // times. Each level keeps a fixed i,j reduction order, so the cached
    // minimum thickness is independent of thread count.
    #pragma omp parallel for if(static_cast<std::size_t>(nr) * nth * nz >= 32768)
    for (int k = 0; k < nz; ++k)
    {
        double min_dz = std::numeric_limits<double>::infinity();
        for (int i = 0; i < nr; ++i)
        {
            for (int j = 0; j < nth; ++j)
            {
                const double dz_local =
                    tmv::require_level_runtime(execution_state.terrain_runtime, "terrain_runtime").metrics.z_face(i, j, k + 1) -
                    tmv::require_level_runtime(execution_state.terrain_runtime, "terrain_runtime").metrics.z_face(i, j, k);

                min_dz = std::min(min_dz, dz_local);
            }
        }
        candidate.dz[static_cast<std::size_t>(k)] = min_dz;
    }

    candidate.z_int[0] = 0.0;
    for (int k = 1; k <= nz; ++k)
    {
        const double dz_layer = require_positive(
            candidate.dz[static_cast<std::size_t>(k - 1)],
            "terrain layer thickness");
        candidate.z_int[static_cast<std::size_t>(k)] =
            candidate.z_int[static_cast<std::size_t>(k - 1)] + dz_layer;
    }
    tmv::require_level_runtime(execution_state.process_grid_metrics, "grid metrics") = std::move(candidate);
}

void initialize_numerics()
{
    return initialize_numerics(global_simulation_state());
}

void initialize_numerics(const SimulationStateView& execution_state)
{
    refresh_grid_metrics_from_terrain(execution_state);

    initialize_advection(tmv::require_level_runtime(execution_state.advection_runtime, "advection_runtime"),
                         tmv::require_level_runtime(execution_state.advection_runtime, "advection_runtime").config.scheme_id,
                         tmv::require_level_runtime(execution_state.advection_runtime, "advection_runtime").config);

    initialize_diffusion(tmv::require_level_runtime(execution_state.diffusion_runtime, "diffusion_runtime"),
                         tmv::require_level_runtime(execution_state.diffusion_runtime, "diffusion_runtime").config.scheme_id,
                         tmv::require_level_runtime(execution_state.diffusion_runtime, "diffusion_runtime").config);

    initialize_time_stepping(tmv::require_level_runtime(execution_state.time_stepping_runtime, "time_stepping_runtime"),
                             tmv::require_level_runtime(execution_state.time_stepping_runtime, "time_stepping_runtime").config.scheme_id,
                             tmv::require_level_runtime(execution_state.time_stepping_runtime, "time_stepping_runtime").config,
                             nullptr);

    tmv::log_info("Initialized numerics framework:");
    tmv::log_info("  Advection: ", tmv::require_level_runtime(execution_state.advection_runtime, "advection_runtime").scheme->name());
    tmv::log_info("  Diffusion: ", tmv::require_level_runtime(execution_state.diffusion_runtime, "diffusion_runtime").scheme->name());
    tmv::log_info("  Time stepping: ", tmv::require_level_runtime(execution_state.time_stepping_runtime, "time_stepping_runtime").scheme->name());
}

namespace
{
// The CFL scan provides a free blow-up signal before wind guards hide it.
double g_last_max_flow_speed = 0.0;
}  // namespace

double last_max_flow_speed()
{
    return g_last_max_flow_speed;
}

double choose_runtime_timestep()
{
    return choose_runtime_timestep(global_simulation_state());
}

double choose_runtime_timestep(const SimulationStateView& execution_state)
{
    const double dt_current = require_positive(execution_state.grid.dt, "runtime dt");
    const double dt_min = require_positive(
        tmv::require_level_runtime(execution_state.time_stepping_runtime, "time_stepping_runtime").config.dt_min, "time-stepping dt_min");
    const double dt_max = require_positive(
        tmv::require_level_runtime(execution_state.time_stepping_runtime, "time_stepping_runtime").config.dt_max, "time-stepping dt_max");
    if (dt_min > dt_max)
        throw std::invalid_argument(
            "time-stepping dt_min must not exceed dt_max");

    double dt_limited = dt_current;
    if (dt_limited < dt_min)
    {
        dt_limited = dt_min;
    }
    if (dt_limited > dt_max)
    {
        dt_limited = dt_max;
    }

    const double dx_safe = require_positive(tmv::require_level_runtime(execution_state.process_grid_metrics, "grid metrics").dx, "grid metric dx");
    const double dy_safe = require_positive(tmv::require_level_runtime(execution_state.process_grid_metrics, "grid metrics").dy, "grid metric dy");
    if (tmv::require_level_runtime(execution_state.process_grid_metrics, "grid metrics").dz.size() != static_cast<std::size_t>(execution_state.grid.nz))
        throw std::invalid_argument(
            "vertical grid metric count does not match grid.nz");
    (void)require_positive(execution_state.grid.dz, "grid dz");
    double min_dz_safe = std::numeric_limits<double>::infinity();
    for (double dz_level : tmv::require_level_runtime(execution_state.process_grid_metrics, "grid metrics").dz)
        min_dz_safe = std::min(
            min_dz_safe, require_positive(dz_level, "vertical grid metric"));
    const double min_spacing = std::min({dx_safe, dy_safe, min_dz_safe});

    const FlowCflScan flow_cfl = compute_flow_cfl_scan(execution_state,
        dx_safe, dy_safe, tmv::require_level_runtime(execution_state.process_grid_metrics, "grid metrics").dz, min_dz_safe, min_spacing);
    g_last_max_flow_speed = flow_cfl.max_component_speed;

    double scheme_cfl = 1.0;
    if (tmv::require_level_runtime(execution_state.time_stepping_runtime, "time_stepping_runtime").scheme)
    {
        TimeSteppingState ts_state{};
        ts_state.dt = dt_current;
        ts_state.time = simulation_time;
        const double suggested = tmv::require_level_runtime(execution_state.time_stepping_runtime, "time_stepping_runtime").scheme->suggest_dt(tmv::require_level_runtime(execution_state.time_stepping_runtime, "time_stepping_runtime").config, ts_state);
        if (!std::isfinite(suggested) || suggested <= 0.0)
            throw std::runtime_error(
                "time-stepping scheme returned an invalid CFL multiplier");
        scheme_cfl = suggested;
    }
    const double cfl_safety = require_positive(
        tmv::require_level_runtime(execution_state.time_stepping_runtime, "time_stepping_runtime").config.cfl_safety, "time-stepping cfl_safety");

    double dt_advection_cap = std::numeric_limits<double>::infinity();
    if (flow_cfl.max_directional_rate > 1.0e-12)
    {
        // The Cartesian transport is multidimensional: a cell can export
        // through x, y, and z faces during the same large step.  Enforce the
        // directional Courant sum rather than pairing the largest component
        // with the smallest spacing (which misses diagonal flow and is wrong
        // on anisotropic grids).
        dt_advection_cap = cfl_safety * scheme_cfl /
                           flow_cfl.max_directional_rate;
    }

    double dt_diffusion_cap = std::numeric_limits<double>::infinity();
    if (tmv::require_level_runtime(execution_state.diffusion_runtime, "diffusion_runtime").scheme)
    {
        const std::string diffusion_id = lower_copy(tmv::require_level_runtime(execution_state.diffusion_runtime, "diffusion_runtime").scheme->name());
        if (diffusion_id == "explicit")
        {
            if (!std::isfinite(tmv::require_level_runtime(execution_state.diffusion_runtime, "diffusion_runtime").config.K_h) ||
                tmv::require_level_runtime(execution_state.diffusion_runtime, "diffusion_runtime").config.K_h < 0.0 ||
                !std::isfinite(tmv::require_level_runtime(execution_state.diffusion_runtime, "diffusion_runtime").config.K_v) ||
                tmv::require_level_runtime(execution_state.diffusion_runtime, "diffusion_runtime").config.K_v < 0.0)
                throw std::invalid_argument(
                    "diffusion coefficients must be non-negative and finite");
            const double K_h = tmv::require_level_runtime(execution_state.diffusion_runtime, "diffusion_runtime").config.K_h;
            const double K_v = tmv::require_level_runtime(execution_state.diffusion_runtime, "diffusion_runtime").config.K_v;
            if (K_h > 0.0 || K_v > 0.0 ||
                tmv::require_level_runtime(execution_state.diffusion_runtime, "diffusion_runtime").config.use_variable_K)
            {
                DiffusionConfig stability_cfg = tmv::require_level_runtime(execution_state.diffusion_runtime, "diffusion_runtime").config;
                stability_cfg.dt_diffusion = 1.0;
                DiffusionStateView diffusion_state{};
                diffusion_state.compute_runtime = execution_state.compute_runtime;
                diffusion_state.grid = &tmv::require_level_runtime(execution_state.process_grid_metrics, "grid metrics");
                diffusion_state.rho = &execution_state.fields.rho;
                const Field3D* vertical_k = prescribed_vertical_diffusivity(
                    tmv::require_level_runtime(execution_state.diffusion_runtime, "diffusion_runtime"), execution_state.grid.nr, execution_state.grid.nth,
                    tmv::require_level_runtime(execution_state.process_grid_metrics, "grid metrics"));
                diffusion_state.K_momentum = vertical_k;
                diffusion_state.K_scalar = vertical_k;
                diffusion_state.c_grid =
                    tmv::require_run_configuration(execution_state).stagger_type == StaggerType::CGrid;
                diffusion_state.periodic_x =
                    tmv::require_run_configuration(execution_state).lateral_bc_west ==
                        tmv::init::config::LateralBC::Periodic &&
                    tmv::require_run_configuration(execution_state).lateral_bc_east ==
                        tmv::init::config::LateralBC::Periodic;
                diffusion_state.periodic_y =
                    tmv::require_run_configuration(execution_state).lateral_bc_south ==
                        tmv::init::config::LateralBC::Periodic &&
                    tmv::require_run_configuration(execution_state).lateral_bc_north ==
                        tmv::init::config::LateralBC::Periodic;
                const std::string target = lower_copy(
                    tmv::require_level_runtime(execution_state.diffusion_runtime, "diffusion_runtime").config.apply_to);
                if (target == "scalars" ||
                    target == "scalars_and_momentum" || target == "all")
                {
                    diffusion_state.theta = &execution_state.fields.theta;
                    diffusion_state.qv = &execution_state.fields.qv;
                }
                if (target == "momentum" ||
                    target == "scalars_and_momentum" || target == "all")
                {
                    diffusion_state.u = &execution_state.fields.u;
                    diffusion_state.v = &execution_state.fields.v;
                    diffusion_state.w = &execution_state.fields.w;
                }
                const double unit_step_number =
                    tmv::require_level_runtime(execution_state.diffusion_runtime, "diffusion_runtime").scheme->check_stability(
                        stability_cfg, diffusion_state);
                if (!std::isfinite(unit_step_number) ||
                    unit_step_number <= 0.0)
                    throw std::runtime_error(
                        "active explicit diffusion returned an invalid "
                        "unit-step stability number");
                dt_diffusion_cap = 0.9 / unit_step_number;
            }
        }
    }

    // Acoustic CFL limit.  Split-explicit runs control this with their small
    // substeps; unsplit runs must include it in the large-step stability cap.
    // Use the fastest valid level in the base state rather than assuming the
    // surface is always the limiting sound speed.
    double dt_acoustic_cap = std::numeric_limits<double>::infinity();
    double max_sound_speed = flow_cfl.max_acoustic_speed;
    if (execution_state.base.p0.size() != execution_state.base.rho0.size())
        throw std::invalid_argument(
            "base pressure and density profiles must have the same length");
    const std::size_t base_levels = execution_state.base.p0.size();
    for (std::size_t k = 0; k < base_levels; ++k)
    {
        if (!std::isfinite(execution_state.base.p0[k]) || execution_state.base.p0[k] <= 0.0 ||
            !std::isfinite(execution_state.base.rho0[k]) || execution_state.base.rho0[k] <= 0.0)
            throw std::invalid_argument(
                "base pressure and density must be positive and finite");
        max_sound_speed = std::max(
            max_sound_speed,
            std::sqrt(physical_constants::dry_air_gamma
                      * execution_state.base.p0[k] / execution_state.base.rho0[k]));
    }
    if (max_sound_speed > 0.0)
    {
        dt_acoustic_cap = cfl_safety * flow_cfl.min_acoustic_spacing /
                          max_sound_speed;
    }
    // Internal, state-derived input to SplitExplicitScheme. A configured
    // positive acoustic_substeps value is only a minimum; the scheme promotes
    // it as needed to keep c_max*dt_small/dx_min <= 0.5.
    tmv::require_level_runtime(execution_state.time_stepping_runtime, "time_stepping_runtime").config.max_acoustic_speed_ms = max_sound_speed;
    tmv::require_level_runtime(execution_state.time_stepping_runtime, "time_stepping_runtime").config.min_acoustic_spacing_m =
        flow_cfl.min_acoustic_spacing;

    const double dt_wave_cap = tmv::require_level_runtime(execution_state.time_stepping_runtime, "time_stepping_runtime").config.split_acoustic
        ? std::numeric_limits<double>::infinity()
        : dt_acoustic_cap;
    double dt_sgs_cap = std::numeric_limits<double>::infinity();
    const auto& turbulence_runtime = tmv::require_level_runtime(execution_state.turbulence_runtime, "turbulence_runtime");
    if (turbulence_runtime.scheme &&
        turbulence_runtime.scheme->name() != "none" &&
        std::isfinite(
            turbulence_runtime.last_explicit_stability_limit_s))
    {
        dt_sgs_cap = require_positive(
            turbulence_runtime.last_explicit_stability_limit_s,
            "previous realized SGS stability limit");
    }
    const double dt_physical_cap = std::min({dt_advection_cap,
                                             dt_diffusion_cap,
                                             dt_wave_cap,
                                             dt_sgs_cap});
    if (std::isfinite(dt_physical_cap) && dt_physical_cap > 0.0)
    {
        if (tmv::require_level_runtime(execution_state.time_stepping_runtime, "time_stepping_runtime").config.adaptive_dt)
        {
            if (dt_physical_cap < dt_min)
            {
                dt_limited = dt_physical_cap;
            }
            else
            {
                dt_limited = std::min(dt_max, std::max(dt_min, dt_physical_cap));
            }
        }
        else
        {
            dt_limited = std::min(dt_limited, dt_physical_cap);
            if (dt_limited < dt_min && dt_physical_cap < dt_min)
            {
                dt_limited = dt_physical_cap;
            }
        }
    }

    if (!std::isfinite(dt_limited) || dt_limited <= 0.0)
    {
        throw std::runtime_error(
            "runtime timestep selection produced a non-positive or non-finite value");
    }

    if (std::isfinite(dt_advection_cap) || std::isfinite(dt_diffusion_cap) ||
        std::isfinite(dt_sgs_cap) ||
        (!tmv::require_level_runtime(execution_state.time_stepping_runtime, "time_stepping_runtime").config.split_acoustic &&
         std::isfinite(dt_acoustic_cap)))
    {
        tmv::log_debug("[NUMERICS] timestep caps: current=", dt_current,
                       " limited=", dt_limited,
                       " advection_cap=", dt_advection_cap,
                       " diffusion_cap=", dt_diffusion_cap,
                       " sgs_cap=", dt_sgs_cap,
                       " acoustic_cap=", dt_acoustic_cap);
    }

    return dt_limited;
}

} // namespace tmv
