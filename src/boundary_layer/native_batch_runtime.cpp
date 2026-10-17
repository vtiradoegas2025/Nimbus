#include "boundary_layer/native_batch_runtime.hpp"

#include "boundary_layer/column_evaluation.hpp"
#include "boundary_layer/surface_frame.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/simulation_state.hpp"
#include "terrain/conservative_transport.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <vector>

namespace tmv
{
namespace
{

std::size_t boundary_layer_batch_capacity(std::size_t levels, std::size_t column_count)
{
    constexpr std::size_t budget_bytes     = 64u * 1024u * 1024u;
    const std::size_t     bytes_per_column = std::max<std::size_t>(1, levels * 256u);
    return std::max<std::size_t>(1, std::min({column_count, budget_bytes / bytes_per_column, std::size_t{256}}));
}

} // namespace

bool evaluate_native_boundary_layer_batches(const SimulationStateView&    state,
                                            BoundaryLayerSchemeBase&      scheme,
                                            const BoundaryLayerConfig&    config,
                                            const SurfaceConfig&          surface_config,
                                            BoundaryLayerRefreshScratch&  scratch,
                                            const PhysicalCellDomain&     domain,
                                            bool                          stagger_x,
                                            bool                          stagger_y,
                                            bool                          capture_skin,
                                            double                        elapsed_time_s,
                                            double                        horizontal_cell_area,
                                            BoundaryLayerComputeKernels&  kernels,
                                            BoundaryLayerBatchPublication publication)
{
    const auto&       grid         = state.grid;
    const auto&       fields       = state.fields;
    const TerrainMetrics3D* const terrain_metrics =
        physical_grid_metrics::active_terrain_metrics(state);
    if (terrain_metrics != nullptr)
        terrain_transport::validate_geometry(*terrain_metrics);
    const std::size_t column_count = static_cast<std::size_t>(domain.nx()) * domain.ny();
    const std::size_t capacity     = boundary_layer_batch_capacity(
        static_cast<std::size_t>(domain.nz()), column_count);
    scratch.batch_columns.resize(capacity);
    scratch.batch_views.resize(capacity);
    scratch.batch_tendencies.resize(capacity);
    scratch.batch_diagnostics.resize(capacity);
    std::vector<std::exception_ptr> failures(capacity);

    for (std::size_t begin = 0; begin < column_count; begin += capacity)
    {
        const std::size_t count = std::min(capacity, column_count - begin);
        std::fill(failures.begin(), failures.begin() + count, std::exception_ptr{});
#pragma omp parallel for if (count >= 8) schedule(static)
        for (std::size_t local = 0; local < count; ++local)
        {
            try
            {
                const std::size_t linear = begin + local;
                const int         i = domain.i_begin + static_cast<int>(linear / static_cast<std::size_t>(domain.ny()));
                const int         j = domain.j_begin + static_cast<int>(linear % static_cast<std::size_t>(domain.ny()));
                auto&             context = scratch.batch_columns[local];
                physical_grid_metrics::column_geometry(state,
                                                       i,
                                                       j,
                                                       domain.k_begin,
                                                       domain.k_end,
                                                       context.workspace.z_center,
                                                       context.workspace.z_interface,
                                                       context.workspace.layer_thickness,
                                                       /*height_above_ground=*/true);
                const double z_sfc = context.workspace.z_center.front();
                if (!std::isfinite(z_sfc) || z_sfc <= 0.0 || z_sfc <= std::max(surface_config.z0m, surface_config.z0h))
                {
                    throw std::invalid_argument(
                        "first scalar height above local terrain must exceed both roughness lengths");
                }

                const std::size_t offset =
                    static_cast<std::size_t>(domain.k_begin);
                const float* p_row = fields.p.row_ptr(i, j) + offset;
                const float* theta_row = fields.theta.row_ptr(i, j) + offset;
                const float* qv_row = fields.qv.row_ptr(i, j) + offset;
                const float* u_row = fields.u.row_ptr(i, j) + offset;
                const float* v_row = fields.v.row_ptr(i, j) + offset;
                const float* rho_row = fields.rho.row_ptr(i, j) + offset;
                const float* tke_row = fields.tke.empty()
                    ? nullptr : fields.tke.row_ptr(i, j) + offset;
                const std::size_t levels =
                    static_cast<std::size_t>(domain.nz());

                BoundaryLayerColumnStateView column;
                column.p     = BoundaryLayerProfileView(p_row, levels);
                column.theta = BoundaryLayerProfileView(theta_row, levels);
                column.qv    = BoundaryLayerProfileView(qv_row, levels);
                if (stagger_x)
                {
                    context.workspace.centered_u.resize(levels);
                    const float* west = fields.u.row_ptr(i - 1, j) + offset;
                    for (std::size_t level = 0; level < levels; ++level)
                    {
                        context.workspace.centered_u[level] = 0.5 * (static_cast<double>(west[level]) + u_row[level]);
                    }
                    column.u = &context.workspace.centered_u;
                }
                else
                {
                    column.u = BoundaryLayerProfileView(u_row, levels);
                }

                if (stagger_y)
                {
                    context.workspace.centered_v.resize(levels);
                    const float* south = fields.v.row_ptr(i, j - 1) + offset;
                    for (std::size_t level = 0; level < levels; ++level)
                    {
                        context.workspace.centered_v[level] = 0.5 * (static_cast<double>(south[level]) + v_row[level]);
                    }
                    column.v = &context.workspace.centered_v;
                }
                else
                {
                    column.v = BoundaryLayerProfileView(v_row, levels);
                }

                column.rho       = BoundaryLayerProfileView(rho_row, levels);
                column.z_int     = &context.workspace.z_interface;
                column.tke       = tke_row ? BoundaryLayerProfileView(tke_row, levels) : BoundaryLayerProfileView{};
                column.workspace = &context.workspace;

                const std::size_t surface_index = static_cast<std::size_t>(i) * grid.nth + j;
                if (capture_skin)
                {
                    publication.skin_theta[surface_index] = theta_row[0];
                    publication.skin_qv[surface_index]    = qv_row[0];
                }

                const auto surface_wind =
                    boundary_layer_surface_relative_wind(
                        column.u[0], column.v[0], terrain_metrics != nullptr
                            && state.terrain_runtime->config.horizontal_placement_id != "earth_fixed",
                        state.domain_frame);
                column.u_sfc = surface_wind.first;
                column.v_sfc = surface_wind.second;
                if (surface_config.Tsfc > 0.0)
                {
                    const double inverse_exner = std::pow(
                        physical_constants::reference_pressure_pa /
                            static_cast<double>(p_row[0]),
                        physical_constants::gas_constant_dry_air_jkgk / physical_constants::specific_heat_cp_jkgk);
                    column.theta_sfc = surface_config.Tsfc * inverse_exner;
                }
                else
                {
                    column.theta_sfc = publication.skin_theta[surface_index];
                }
                column.qv_sfc = surface_config.qsfc >= 0.0 ? surface_config.qsfc : publication.skin_qv[surface_index];
                column.z_sfc  = z_sfc;
                if (terrain_metrics != nullptr)
                {
                    column.surface_slope_x =
                        -terrain_metrics->Gx_face(i, j, 0);
                    column.surface_slope_y =
                        -terrain_metrics->Gy_face(i, j, 0);
                }
                column.slab_state     = &publication.slab[surface_index];
                column.elapsed_time_s = elapsed_time_s;

                validate_prepared_boundary_layer_column(column, horizontal_cell_area);
                scratch.batch_views[local] = column;
                scratch.batch_diagnostics[local].reset_retain_capacity(false);
                scratch.batch_tendencies[local].clear_retain_capacity();
            }
            catch (...)
            {
                failures[local] = std::current_exception();
            }
        }

        for (std::size_t local = 0; local < count; ++local)
        {
            if (failures[local])
                std::rethrow_exception(failures[local]);
        }

        // Device queues are intentionally driven by this calling thread. Only
        // host-side column preparation/finalization is parallel.
        if (!scheme.compute_columns_native(config,
                                           surface_config,
                                           scratch.batch_views.data(),
                                           count,
                                           scratch.batch_tendencies.data(),
                                           scratch.batch_diagnostics.data(),
                                           kernels))
        {
            return false;
        }

        std::fill(failures.begin(), failures.begin() + count, std::exception_ptr{});
#pragma omp parallel for if (count >= 8) schedule(static)
        for (std::size_t local = 0; local < count; ++local)
        {
            try
            {
                const std::size_t linear = begin + local;
                const int         i = domain.i_begin + static_cast<int>(linear / static_cast<std::size_t>(domain.ny()));
                const int         j = domain.j_begin + static_cast<int>(linear % static_cast<std::size_t>(domain.ny()));
                auto&             context = scratch.batch_columns[local];
                const auto        stored  = store_prepared_boundary_layer_result(scratch.batch_views[local],
                                                                                 horizontal_cell_area,
                                                                                 scratch.batch_tendencies[local],
                                                                                 scratch.batch_diagnostics[local]);
                const std::size_t surface_index     = static_cast<std::size_t>(i) * grid.nth + j;
                publication.ustar[surface_index]    = stored.ustar;
                publication.tau_u[surface_index]    = stored.tau_u;
                publication.tau_v[surface_index]    = stored.tau_v;
                publication.sensible[surface_index] = stored.sensible_heat_flux;
                publication.latent[surface_index]   = stored.latent_heat_flux;
                publication.moisture[surface_index] = stored.moisture_flux;
                publication.height[surface_index]   = stored.pbl_height;

                for (int k = 0; k < domain.nz(); ++k)
                {
                    const std::size_t level = static_cast<std::size_t>(k);
                    const int model_k = domain.k_begin + k;
                    publication.dtheta[i][j][model_k] =
                        context.workspace.stored_dtheta[level];
                    publication.dqv[i][j][model_k] =
                        context.workspace.stored_dqv[level];
                    publication.du[i][j][model_k] =
                        context.workspace.stored_du[level];
                    publication.dv[i][j][model_k] =
                        context.workspace.stored_dv[level];
                    publication.dtke[i][j][model_k] =
                        context.workspace.stored_dtke[level];
                    publication.u_force[i][j][model_k] =
                        context.workspace.stored_u_force_n[level];
                    publication.v_force[i][j][model_k] =
                        context.workspace.stored_v_force_n[level];
                    publication.theta_rate[i][j][model_k] =
                        context.workspace.stored_theta_inventory_rate_kgk_s[level];
                    publication.vapor_rate[i][j][model_k] =
                        context.workspace.stored_vapor_mass_rate_kg_s[level];
                }
            }
            catch (...)
            {
                failures[local] = std::current_exception();
            }
        }

        for (std::size_t local = 0; local < count; ++local)
        {
            if (failures[local])
                std::rethrow_exception(failures[local]);
        }
    }

    return true;
}

} // namespace tmv
