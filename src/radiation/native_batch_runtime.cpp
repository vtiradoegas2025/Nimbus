#include "radiation/native_batch_runtime.hpp"

#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/simulation_state.hpp"
#include "radiation/schemes/simple_grey/simple_grey.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace tmv
{
namespace
{

std::size_t radiation_batch_capacity(const RadiationConfig& config, std::size_t layers, std::size_t column_count)
{
    // Includes packed inputs plus backend spectral partials. The cap bounds
    // host scratch and individual submissions while leaving room for the
    // coefficient-derived preprocessing objects owned by the scheme.
    std::size_t floats_per_column = 2 * (layers + 1);
    if (config.do_lw)
    {
        floats_per_column += layers * (2 * 140 + 2 * 16) + (layers + 1) * (2 * 140 + 16) + 34;
    }
    if (config.do_sw)
    {
        floats_per_column += layers * (2 * 112 + 3 * 14) + (layers + 1) * (12 * 112) + 115;
    }

    constexpr std::size_t budget_bytes     = 64u * 1024u * 1024u;
    const std::size_t     bytes_per_column = std::max<std::size_t>(sizeof(float), floats_per_column * sizeof(float));
    const std::size_t     budget_columns   = std::max<std::size_t>(1, budget_bytes / bytes_per_column);
    return std::max<std::size_t>(1, std::min({column_count, budget_columns, std::size_t{64}}));
}

} // namespace

bool evaluate_native_radiation_batches(const SimulationStateView&         state,
                                       RadiationRuntimeState&             runtime,
                                       RadiationSchemeBase&               scheme,
                                       const RadiationConfig&             config,
                                       const PhysicalCellDomain&          domain,
                                       RadiationComputeKernels&           kernels,
                                       const RadiationNativeBatchRequest& request,
                                       RadiationBatchPublication          publication)
{
    if (request.physical_nz == 0 || (request.terrain_shortwave && request.terrain_workspace == nullptr))
    {
        throw std::invalid_argument(
            "native radiation batching requires a non-empty column and complete terrain workspace");
    }
    const std::size_t column_count = static_cast<std::size_t>(domain.nx()) * domain.ny();
    const std::size_t capacity     = radiation_batch_capacity(config, request.physical_nz, column_count);
    if (!runtime.batch_workspace)
    {
        runtime.batch_workspace = std::make_shared<RadiationBatchRuntimeWorkspace>();
    }

    RadiationBatchRuntimeWorkspace& batch = *runtime.batch_workspace;
    batch.columns.resize(capacity);
    batch.prepared.resize(capacity);
    batch.states.resize(capacity);
    batch.tendencies.resize(capacity);
    if (publication.fluxes)
        batch.fluxes.resize(capacity);
    batch.failures.resize(capacity);

    const auto&      fields = state.fields;
    const auto&      grid   = state.grid;
    constexpr double p0     = physical_constants::reference_pressure_pa;
    constexpr double kappa  = physical_constants::gas_constant_dry_air_jkgk / physical_constants::specific_heat_cp_jkgk;

    for (std::size_t begin = 0; begin < column_count; begin += capacity)
    {
        const std::size_t count = std::min(capacity, column_count - begin);
        std::fill(batch.failures.begin(), batch.failures.begin() + count, std::exception_ptr{});
#pragma omp parallel for if (count >= 8) schedule(static)
        for (std::size_t local = 0; local < count; ++local)
        {
            try
            {
                const std::size_t linear = begin + local;
                const int         i = domain.i_begin + static_cast<int>(linear / static_cast<std::size_t>(domain.ny()));
                const int         j = domain.j_begin + static_cast<int>(linear % static_cast<std::size_t>(domain.ny()));
                RadiationColumnWorkspace& workspace = batch.columns[local].workspace;
                workspace.p.resize(request.physical_nz);
                workspace.T.resize(request.physical_nz);
                workspace.rho.resize(request.physical_nz);
                if (request.has_vapor)
                    workspace.qv.resize(request.physical_nz);
                if (request.has_cloud_liquid)
                    workspace.qc.resize(request.physical_nz);
                if (request.has_cloud_ice)
                    workspace.qi.resize(request.physical_nz);

                physical_grid_metrics::column_geometry(state,
                                                       i,
                                                       j,
                                                       domain.k_begin,
                                                       domain.k_end,
                                                       workspace.z_center,
                                                       workspace.z_interface,
                                                       workspace.layer_thickness);
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const std::size_t level    = static_cast<std::size_t>(k - domain.k_begin);
                    const double      pressure = fields.p[i][j][k];
                    workspace.p[level]         = pressure;
                    workspace.T[level]         = fields.theta[i][j][k] * std::pow(pressure / p0, kappa);
                    workspace.rho[level]       = fields.rho[i][j][k];
                    if (request.has_vapor)
                        workspace.qv[level] = fields.qv[i][j][k];
                    if (request.has_cloud_liquid)
                        workspace.qc[level] = fields.qc[i][j][k];
                    if (request.has_cloud_ice)
                        workspace.qi[level] = fields.qi[i][j][k];
                }

                RadiationPreparedColumnView& prepared = batch.prepared[local];
                prepared.pressure_pa                  = &workspace.p;
                prepared.temperature_k                = &workspace.T;
                prepared.density_kgm3                 = &workspace.rho;
                prepared.vapor_mixing_ratio           = request.has_vapor ? &workspace.qv : nullptr;
                prepared.cloud_liquid_mixing_ratio    = request.has_cloud_liquid ? &workspace.qc : nullptr;
                prepared.cloud_ice_mixing_ratio       = request.has_cloud_ice ? &workspace.qi : nullptr;
                prepared.z_center_m                   = &workspace.z_center;
                prepared.z_interface_m                = &workspace.z_interface;
                prepared.layer_thickness_m            = &workspace.layer_thickness;
                prepared.cosine_solar_zenith          = request.cosine_solar_zenith;
                prepared.surface_temperature_k        = request.surface_temperature;
                prepared.surface_albedo               = request.surface_albedo;
                prepared.surface_emissivity           = request.surface_emissivity;
                prepared.horizontal_cell_area_m2      = request.horizontal_cell_area;
                batch.states[local]                   = prepare_radiation_column_state(config, prepared, workspace);
            }
            catch (...)
            {
                batch.failures[local] = std::current_exception();
            }
        }

        for (std::size_t local = 0; local < count; ++local)
        {
            if (batch.failures[local])
                std::rethrow_exception(batch.failures[local]);
        }

        // Device queues are intentionally driven by this calling thread. Only
        // host-side column preparation/finalization is parallel.
        if (!scheme.compute_columns_native_diagnosed(config,
                                                     batch.states.data(),
                                                     count,
                                                     batch.tendencies.data(),
                                                     kernels,
                                                     publication.fluxes ? batch.fluxes.data() : nullptr))
        {
            return false;
        }

        std::fill(batch.failures.begin(), batch.failures.begin() + count, std::exception_ptr{});
#pragma omp parallel for if (count >= 8) schedule(static)
        for (std::size_t local = 0; local < count; ++local)
        {
            try
            {
                const std::size_t linear = begin + local;
                const int         i = domain.i_begin + static_cast<int>(linear / static_cast<std::size_t>(domain.ny()));
                const int         j = domain.j_begin + static_cast<int>(linear % static_cast<std::size_t>(domain.ny()));
                RadiationColumnWorkspace& workspace = batch.columns[local].workspace;
                const double              stored_power =
                    finalize_prepared_radiation_column(batch.prepared[local], workspace, batch.tendencies[local]);
                const std::size_t storage_column           = static_cast<std::size_t>(i) * grid.nth + j;
                publication.column_power_w[storage_column] = stored_power;
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const std::size_t level             = static_cast<std::size_t>(k - domain.k_begin);
                    publication.theta_tendency[i][j][k] = workspace.theta_tendency[level];
                    publication.cell_power_w[i][j][k]   = workspace.cell_power_w[level];
                }

                if (publication.fluxes)
                {
                    auto&                                           target = *publication.fluxes;
                    const auto&                                     source = batch.fluxes[local];
                    const std::array<const std::vector<double>*, 4> inputs{
                        &source.Fup_lw, &source.Fdn_lw, &source.Fup_sw, &source.Fdn_sw};
                    const std::array<std::vector<double>*, 4> outputs{&target.upward_longwave_wm2,
                                                                      &target.downward_longwave_wm2,
                                                                      &target.upward_shortwave_wm2,
                                                                      &target.downward_shortwave_wm2};
                    for (std::size_t field = 0; field < inputs.size(); ++field)
                    {
                        if (inputs[field]->size() != request.physical_nz + 1 ||
                            outputs[field]->size() != column_count * (request.physical_nz + 1))
                            throw std::invalid_argument("native radiation diagnostics have the wrong shape");
                        std::copy(inputs[field]->begin(),
                                  inputs[field]->end(),
                                  outputs[field]->begin() + linear * (request.physical_nz + 1));
                    }
                }

                if (request.terrain_shortwave)
                {
                    compute_simple_grey_shortwave_layer_optical_depths(config,
                                                                       workspace.p_interface,
                                                                       workspace.layer_thickness,
                                                                       workspace.rho,
                                                                       request.has_vapor ? &workspace.qv : nullptr,
                                                                       request.has_cloud_liquid ? &workspace.qc
                                                                                                : nullptr,
                                                                       request.has_cloud_ice ? &workspace.qi : nullptr,
                                                                       workspace.tau_profile,
                                                                       workspace.tau_layers);
                    std::copy(workspace.z_interface.begin(),
                              workspace.z_interface.end(),
                              request.terrain_workspace->interface_height_m.begin() +
                                  linear * (request.physical_nz + 1));
                    std::copy(workspace.tau_layers.begin(),
                              workspace.tau_layers.end(),
                              request.terrain_workspace->vertical_optical_depth.begin() + linear * request.physical_nz);
                }
            }
            catch (...)
            {
                batch.failures[local] = std::current_exception();
            }
        }

        for (std::size_t local = 0; local < count; ++local)
        {
            if (batch.failures[local])
                std::rethrow_exception(batch.failures[local]);
        }
    }

    return true;
}

} // namespace tmv
