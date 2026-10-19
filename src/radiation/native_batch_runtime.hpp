#pragma once

#include "radiation/column_evaluation.hpp"
#include "radiation/radiation_base.hpp"
#include "radiation/base/radiative_transfer.hpp"
#include "core/infra/physical_cell_domain.hpp"

#include <cstddef>
#include <exception>
#include <vector>

class RadiationComputeKernels;
struct SimulationStateView;

namespace tmv
{

// Replaceable operational storage. Neither type is part of a simulation
// snapshot or checkpoint; both exist only to amortize cadence-update work.
struct TerrainShortwaveRuntimeWorkspace
{
    std::vector<double>                                 interface_height_m;
    std::vector<double>                                 vertical_optical_depth;
    std::vector<double>                                 slope_x;
    std::vector<double>                                 slope_y;
    std::vector<double>                                 spectral_absorption_optical_depth;
    std::vector<double>                                 spectral_scattering_optical_depth;
    std::vector<double>                                 spectral_scattering_asymmetry;
    std::vector<double>                                 spectral_solar_source_fraction;
    radiative_transfer::MappedTerrainShortwaveWorkspace operator_workspace;
    radiative_transfer::MappedTerrainShortwaveResult    operator_result;
    radiative_transfer::MappedSpectralTerrainWorkspace  spectral_operator_workspace;
    radiative_transfer::MappedSpectralTerrainResult     spectral_operator_result;
};

struct RadiationBatchRuntimeWorkspace
{
    std::vector<RadiationWorkerContext>      columns;
    std::vector<RadiationPreparedColumnView> prepared;
    std::vector<RadiationColumnStateView>    states;
    std::vector<RadiationColumnTendencies>   tendencies;
    std::vector<RadiationColumnFluxes>       fluxes;
    std::vector<std::exception_ptr>          failures;
};

struct RadiationNativeBatchRequest
{
    std::size_t                       physical_nz          = 0;
    double                            cosine_solar_zenith  = 0.0;
    double                            surface_temperature  = 0.0;
    double                            surface_albedo       = 0.0;
    double                            surface_emissivity   = 0.0;
    double                            horizontal_cell_area = 0.0;
    bool                              has_vapor            = false;
    bool                              has_cloud_liquid     = false;
    bool                              has_cloud_ice        = false;
    bool                              terrain_shortwave    = false;
    TerrainShortwaveRuntimeWorkspace* terrain_workspace    = nullptr;
};

struct RadiationBatchPublication
{
    Field3D&                  theta_tendency;
    Field3D&                  cell_power_w;
    std::vector<double>&      column_power_w;
    RadiationFluxDiagnostics* fluxes = nullptr;
};

// Prepare and finalize columns in parallel around one serialized native
// submission per bounded chunk. Results remain caller-owned candidates; this
// routine never publishes committed runtime state.
bool evaluate_native_radiation_batches(const SimulationStateView&         state,
                                       RadiationRuntimeState&             runtime,
                                       RadiationSchemeBase&               scheme,
                                       const RadiationConfig&             config,
                                       const PhysicalCellDomain&          domain,
                                       RadiationComputeKernels&           kernels,
                                       const RadiationNativeBatchRequest& request,
                                       RadiationBatchPublication          publication);

} // namespace tmv
