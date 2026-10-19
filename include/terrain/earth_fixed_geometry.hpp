#pragma once

#include "core/infra/domain_motion.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/field/field3d.hpp"
#include <limits>
#include "terrain/terrain_base.hpp"

class TerrainComputeKernels;

namespace tmv
{

struct EarthFixedTerrainBuildRequest
{
    int                       nr   = 0;
    int                       nth  = 0;
    int                       nz   = 0;
    double                    dx_m = 0.0;
    double                    dy_m = 0.0;
    domain::HorizontalPoint   grid_origin_earth_m;
    domain::HorizontalPoint   feature_center_earth_m;
    EarthFixedTerrainSampling sampling = EarthFixedTerrainSampling::Point;
};

struct TerrainSamplingLattice
{
    domain::HorizontalPoint anchor_earth_m;
    std::int64_t            i_origin = 0;
    std::int64_t            j_origin = 0;
};

struct EarthFixedTerrainGeometry
{
    Topography2D       topography;
    TerrainMetrics3D   metrics;
    TerrainDiagnostics diagnostics;
};

struct PreparedEarthFixedTerrainCommit
{
    TerrainConfig              expected_config;
    EarthFixedTerrainPlacement expected_placement;
    EarthFixedTerrainPlacement next_placement;
    EarthFixedTerrainGeometry  geometry;
    bool                       committed = false;
};

struct TerrainMeshSweptVolumes
{
    TerrainField3D x;
    TerrainField3D y;
    TerrainField3D z;
    double         interval_s = 0.0;
};

// Signed physical volumes swept in the positive coordinate directions.
// Their cell divergence equals V(end)-V(begin), including horizontal motion.
TerrainMeshSweptVolumes build_terrain_mesh_swept_volumes(const TerrainMetrics3D& before,
                                                         const TerrainMetrics3D& after,
                                                         domain::HorizontalPoint displacement_m,
                                                         double                  interval_s);

struct MovingMeshInventoryUpdate
{
    double initial_inventory = 0.0;
    double final_inventory   = 0.0;
    double boundary_outflow  = 0.0;
    double storage_rounding  = 0.0;
};

// Fluxes are already integrated over time and physical face area, in the
// moving frame. Field values are inventory per physical volume, not ratios.
MovingMeshInventoryUpdate update_moving_mesh_inventory(const TerrainMetrics3D&           before,
                                                       const TerrainMetrics3D&           after,
                                                       const TerrainContravariantFluxes& integrated_relative_flux,
                                                       const PhysicalCellDomain&         domain,
                                                       const Field3D&                    initial,
                                                       Field3D&                          result,
                                                       double minimum_value = -std::numeric_limits<double>::infinity());

struct PreparedEarthFixedTerrainMotion
{
    PreparedEarthFixedTerrainCommit terrain;
    TerrainMeshSweptVolumes         swept_volumes;
    EarthFixedTerrainGeometry       stage_geometry;
    double                          begin_time_s = 0.0;
    double                          end_time_s   = 0.0;
};

PreparedEarthFixedTerrainMotion prepare_earth_fixed_terrain_motion(const TerrainRuntimeState&           live,
                                                                   const domain::HorizontalDomainFrame& frame,
                                                                   double                               begin_time_s,
                                                                   double                               end_time_s);

PreparedEarthFixedTerrainMotion prepare_earth_fixed_terrain_motion(const TerrainRuntimeState&           live,
                                                                   const domain::HorizontalDomainFrame& frame,
                                                                   double                               begin_time_s,
                                                                   double                               end_time_s,
                                                                   TerrainComputeKernels&               kernels);

EarthFixedTerrainGeometry build_earth_fixed_volume_geometry(const TerrainConfig&                 config,
                                                            const EarthFixedTerrainBuildRequest& request);

struct EarthFixedVolumeGeometryInterval
{
    EarthFixedTerrainGeometry stage;
    EarthFixedTerrainGeometry end;
    TerrainMeshSweptVolumes   swept_volumes;
};

EarthFixedVolumeGeometryInterval
build_earth_fixed_volume_geometry_interval(const TerrainConfig&                 config,
                                           const EarthFixedTerrainBuildRequest& begin,
                                           domain::HorizontalPoint              end_origin_earth_m,
                                           double                               interval_s,
                                           const TerrainMetrics3D*              expected_begin = nullptr);

EarthFixedTerrainGeometry build_earth_fixed_terrain_geometry(const TerrainConfig&                 config,
                                                             const EarthFixedTerrainBuildRequest& request);

EarthFixedTerrainGeometry build_lattice_anchored_terrain_geometry(const TerrainConfig&                 config,
                                                                  const EarthFixedTerrainBuildRequest& request,
                                                                  const TerrainSamplingLattice&        lattice);

PreparedEarthFixedTerrainCommit prepare_earth_fixed_terrain_commit(const TerrainRuntimeState&           live,
                                                                   const EarthFixedTerrainBuildRequest& request);

PreparedEarthFixedTerrainCommit prepare_earth_fixed_terrain_commit(const TerrainRuntimeState&           live,
                                                                   const EarthFixedTerrainBuildRequest& request,
                                                                   TerrainComputeKernels&               kernels);

bool commit_earth_fixed_terrain(PreparedEarthFixedTerrainCommit& prepared, TerrainRuntimeState& live) noexcept;

bool can_commit_earth_fixed_terrain(const PreparedEarthFixedTerrainCommit& prepared,
                                    const TerrainRuntimeState&             live) noexcept;

} // namespace tmv
