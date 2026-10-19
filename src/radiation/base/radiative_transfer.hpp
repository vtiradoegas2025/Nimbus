/**
 * @file radiative_transfer.hpp
 * @brief Declarations for the radiation module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the radiation runtime and scheme implementations.
 * This file is part of the src/radiation subsystem.
 */

#pragma once
#include <optional>
#include <vector>
#include "radiation/radiation_base.hpp"
#include "radiation/memory_policy.hpp"

namespace tmv
{
namespace radiative_transfer
{

void heating_rate_from_flux_divergence(
    const std::vector<double>& rho,
    const std::vector<double>& dz,
    const std::vector<double>& Fnet,
    std::vector<double>& dTdt
);


/**
 * @brief Computes the planck function.
 */
inline double planck_function(double T) 
{
    using namespace radiation_constants;
    return sigma * pow(T, 4.0) / pi;
}

/**
 * @brief Computes the stefan-boltzmann law.
 */
inline double stefan_boltzmann(double T) 
{
    using namespace radiation_constants;
    return sigma * pow(T, 4.0);
}

/**
 * @brief Computes the grey longwave two stream solution.
 */
void grey_lw_two_stream(
    const std::vector<double>& T,
    const std::vector<double>& tau_layers,
    double T_sfc,
    double emissivity,
    std::vector<double>& Fup,
    std::vector<double>& Fdn
);

/** Exact pure-absorption transmittance of Lambertian upward irradiance.
 *
 * A Lambertian boundary has angle-independent radiance. Integrating its
 * attenuated radiance over the upward hemisphere gives `2 E3(tau)`, not the
 * direct-beam `exp(-tau/mu0)` used for a collimated solar ray.
 */
double lambertian_diffuse_transmittance(double optical_depth);

/** Conservative direct-beam boundary terms for one sloped Lambertian cell. */
struct SlopedLambertianDirectBoundary
{
    double surface_area_per_horizontal_area = 1.0;
    double cosine_incidence = 0.0;
    double incident_irradiance_actual_wm2 = 0.0;
    double incident_power_horizontal_wm2 = 0.0;
    double absorbed_power_horizontal_wm2 = 0.0;
    double reflected_power_horizontal_wm2 = 0.0;
    double reflected_radiance_wm2sr = 0.0;
};

/**
 * Apply exact local slope and Lambertian boundary geometry to an already
 * attenuated direct beam.
 *
 * `direct_normal_irradiance_wm2` is irradiance normal to the ray after its
 * three-dimensional atmospheric traversal. `slope_x/y` are dh/dx and dh/dy.
 * Occlusion is deliberately explicit: a terrain ray solver must provide
 * `visible_to_sun`; this helper never invents a no-shadow assumption.
 */
SlopedLambertianDirectBoundary evaluate_sloped_lambertian_direct_boundary(
    double direct_normal_irradiance_wm2,
    const IdealizedSolarGeometry& sun_direction,
    double slope_x,
    double slope_y,
    double albedo,
    bool visible_to_sun);

enum class TerrainShadowBoundary
{
    Open,
    Periodic,
};

struct TerrainHeightMapView
{
    const double* height_m = nullptr;
    std::size_t nx = 0;
    std::size_t ny = 0;
    double dx_m = 0.0;
    double dy_m = 0.0;
    TerrainShadowBoundary x_boundary = TerrainShadowBoundary::Open;
    TerrainShadowBoundary y_boundary = TerrainShadowBoundary::Open;
};

/**
 * Exact bilinear-height-map line-of-sight oracle for one surface sample.
 *
 * This walks every crossed interpolation patch and maximizes the quadratic
 * terrain-minus-ray clearance inside that patch, so a sub-cell obstruction
 * cannot be missed. It remains the independent correctness reference for the
 * accelerated all-cell traversal, not the large-domain implementation itself.
 */
bool terrain_height_map_visible_to_sun_reference(
    const TerrainHeightMapView& terrain,
    std::size_t i,
    std::size_t j,
    const IdealizedSolarGeometry& sun_direction,
    std::uint64_t* bilinear_patches_tested = nullptr);

struct TerrainShadowMaskWorkspace
{
    std::vector<unsigned char> scratch_visibility;
    std::vector<double> hierarchy_max_height_m;
    std::vector<std::size_t> hierarchy_level_offset;
    std::vector<std::size_t> hierarchy_level_nx;
    std::vector<std::size_t> hierarchy_level_ny;
};

struct TerrainShadowTraversalStats
{
    std::uint64_t height_values_validated = 0;
    std::uint64_t rays_traced = 0;
    std::uint64_t hierarchy_nodes_built = 0;
    std::uint64_t hierarchy_blocks_skipped = 0;
    std::uint64_t bilinear_patches_tested = 0;
};

/**
 * Once-validated, allocation-reusing parallel CPU visibility field.
 *
 * Builds a conservative bilinear-patch maximum hierarchy and uses it only to
 * skip blocks proven clear; unresolved leaves retain the exact reference test.
 */
void terrain_height_map_visibility_field_cpu(
    const TerrainHeightMapView& terrain,
    const IdealizedSolarGeometry& sun_direction,
    TerrainShadowMaskWorkspace& workspace,
    std::vector<unsigned char>& visibility,
    TerrainShadowTraversalStats* stats = nullptr);

/** One ordered atmospheric-cell crossing along a direct solar ray. */
struct DirectBeamPathSegment
{
    std::size_t cell_index = 0;
    double path_length_m = 0.0;
    double vertical_layer_optical_depth = 0.0;
    double physical_layer_thickness_m = 0.0;
};

/** Exact conservative disposition of one ordered path segment. */
struct DirectBeamSegmentAbsorption
{
    std::size_t cell_index = 0;
    double slant_optical_depth = 0.0;
    double incoming_power_w = 0.0;
    double absorbed_power_w = 0.0;
    double transmitted_power_w = 0.0;
};

struct DirectBeamPathResult
{
    double incident_power_w = 0.0;
    double absorbed_power_w = 0.0;
    double transmitted_power_w = 0.0;
    double total_slant_optical_depth = 0.0;
};

/**
 * Conservative pure-absorption reference for an ordered 3-D ray path.
 *
 * `vertical_layer_optical_depth` is the cell's pressure-derived vertical
 * optical depth. The physical ray optical depth is
 * `vertical_tau * path_length / physical_layer_thickness`, which reduces
 * exactly to the existing `tau / mu0` flat-column contract. Results are
 * published only after every segment is validated. The routine reuses the
 * caller's result capacity and performs no retained grid-sized allocation.
 */
DirectBeamPathResult attenuate_direct_beam_path_reference(
    double incident_power_w,
    const std::vector<DirectBeamPathSegment>& ordered_segments,
    std::vector<DirectBeamSegmentAbsorption>& segment_absorption);

struct BilinearMappedInterfacePatch
{
    double z00_m = 0.0;
    double z10_m = 0.0;
    double z01_m = 0.0;
    double z11_m = 0.0;
    double x_lower_m = 0.0;
    double y_lower_m = 0.0;
    double dx_m = 0.0;
    double dy_m = 0.0;
};

enum class MappedInterfaceCrossing
{
    FromAbove,
    FromBelow,
};

/**
 * First true crossing of one bilinear mapped interface by a physical ray.
 *
 * The ray direction must be unit length and the complete interval must remain
 * inside the supplied horizontal patch. A downward layer crossing is
 * `FromAbove`; an upward crossing through the same interface is `FromBelow`.
 * Tangent contact is deliberately not a crossing. Zero is returned when the
 * ray begins on the interface and immediately moves to the requested side.
 */
std::optional<double> first_bilinear_mapped_interface_intersection_reference(
    const BilinearMappedInterfacePatch& patch,
    double start_x_m,
    double start_y_m,
    double start_z_m,
    const IdealizedSolarGeometry& ray_direction,
    double maximum_path_length_m,
    MappedInterfaceCrossing crossing);

struct RectilinearDirectBeamGridView
{
    const double* vertical_layer_optical_depth = nullptr;
    const double* z_interface_m = nullptr;
    std::size_t nx = 0;
    std::size_t ny = 0;
    std::size_t nz = 0;
    double dx_m = 0.0;
    double dy_m = 0.0;
    TerrainShadowBoundary x_boundary = TerrainShadowBoundary::Open;
    TerrainShadowBoundary y_boundary = TerrainShadowBoundary::Open;
};

enum class DirectBeamRayExit
{
    Surface,
    OpenLateralBoundary,
    LevelTop,
};

struct DirectBeamRayResult
{
    DirectBeamPathResult energy;
    DirectBeamRayExit exit = DirectBeamRayExit::Surface;
    std::size_t terminal_i = 0;
    std::size_t terminal_j = 0;
    double terminal_x_m = 0.0;
    double terminal_y_m = 0.0;
    double terminal_z_m = 0.0;
};

/**
 * Exact top-entry long-characteristic reference for a rectilinear atmosphere.
 *
 * One ray bundle enters at the center of a top horizontal cell and travels in
 * the photon direction opposite `sun_direction`. Every x/y/z face crossing is
 * resolved by physical path length; periodic axes wrap and open axes publish a
 * lateral escape. The output path and absorption vectors are transactional.
 * This oracle establishes 3-D cell/energy bookkeeping before the mapped-
 * interface tracer; it is not the eventual all-ray production sweep.
 */
DirectBeamRayResult
trace_rectilinear_direct_beam_from_top_reference(
    const RectilinearDirectBeamGridView& grid,
    std::size_t top_i,
    std::size_t top_j,
    const IdealizedSolarGeometry& sun_direction,
    double incident_power_w,
    std::vector<DirectBeamPathSegment>& ordered_segments,
    std::vector<DirectBeamSegmentAbsorption>& segment_absorption);

struct MappedDirectBeamGridView
{
    const double* vertical_layer_optical_depth = nullptr;
    // Scalar-column interface samples in [i][j][k_face] order. Between scalar
    // centers they define the same continuous bilinear surfaces used by mapped
    // trajectory geometry; open edge half-cells use constant extrapolation.
    const double* z_interface_m = nullptr;
    std::size_t nx = 0;
    std::size_t ny = 0;
    std::size_t nz = 0;
    double dx_m = 0.0;
    double dy_m = 0.0;
    TerrainShadowBoundary x_boundary = TerrainShadowBoundary::Open;
    TerrainShadowBoundary y_boundary = TerrainShadowBoundary::Open;
};

/** Exact top-entry long characteristic through continuous mapped interfaces. */
DirectBeamRayResult trace_mapped_direct_beam_from_top_reference(
    const MappedDirectBeamGridView& grid,
    std::size_t top_i,
    std::size_t top_j,
    const IdealizedSolarGeometry& sun_direction,
    double incident_power_w,
    std::vector<DirectBeamPathSegment>& ordered_segments,
    std::vector<DirectBeamSegmentAbsorption>& segment_absorption);

/** Exact arbitrary-direction characteristic from a terrain-cell center. */
DirectBeamRayResult trace_mapped_radiative_ray_from_surface_reference(
    const MappedDirectBeamGridView& grid,
    std::size_t surface_i,
    std::size_t surface_j,
    const IdealizedSolarGeometry& ray_direction,
    double incident_power_w,
    std::vector<DirectBeamPathSegment>& ordered_segments,
    std::vector<DirectBeamSegmentAbsorption>& segment_absorption);

/** Trace an arbitrary diffuse/scattered bundle from a mapped cell center. */
DirectBeamRayResult trace_mapped_radiative_ray_from_cell_reference(
    const MappedDirectBeamGridView& grid,
    std::size_t cell_i,
    std::size_t cell_j,
    std::size_t cell_k,
    const IdealizedSolarGeometry& ray_direction,
    double incident_power_w,
    std::vector<DirectBeamPathSegment>& ordered_segments,
    std::vector<DirectBeamSegmentAbsorption>& segment_absorption);

struct DirectBeamFieldResult
{
    std::vector<double> cell_absorbed_power_w;
    std::vector<double> surface_incident_power_w;
    double incoming_top_power_w = 0.0;
    double incoming_side_power_w = 0.0;
    double atmospheric_absorbed_power_w = 0.0;
    double surface_incident_total_power_w = 0.0;
    double lateral_escape_power_w = 0.0;
    double boundary_closure_residual_w = 0.0;
    std::uint64_t top_rays = 0;
    std::uint64_t side_rays = 0;
    std::uint64_t surface_rays = 0;
    std::uint64_t escaped_rays = 0;
    std::uint64_t interface_values_validated = 0;
    std::uint64_t optical_values_validated = 0;
};

struct DirectBeamFieldWorkspace
{
    struct RayScratch
    {
        std::vector<DirectBeamPathSegment> segments;
        std::vector<DirectBeamSegmentAbsorption> absorption;
        DirectBeamRayResult result;
        double incident_power_w = 0.0;
        bool top_boundary = false;
    };

    std::vector<double> scratch_cell_absorbed_power_w;
    std::vector<double> scratch_surface_incident_power_w;
    std::vector<DirectBeamPathSegment> ray_segments;
    std::vector<DirectBeamSegmentAbsorption> ray_absorption;
    std::vector<RayScratch> ray_batch;
    std::vector<unsigned char> ray_batch_failures;
};

/**
 * Complete incoming-boundary mapped direct-beam field reference.
 *
 * Launches one conservative bundle at every top-face center and at every
 * sun-facing open lateral-face center. Periodic axes have no lateral boundary.
 * All unabsorbed power is published as terrain incidence or open-side escape.
 * Serial fixed-order accumulation is deterministic and transactional; a later
 * production sweep may accelerate this exact field/ledger contract.
 */
void evaluate_mapped_direct_beam_field_reference(
    const MappedDirectBeamGridView& grid,
    const IdealizedSolarGeometry& sun_direction,
    double beam_normal_irradiance_wm2,
    DirectBeamFieldWorkspace& workspace,
    DirectBeamFieldResult& result);

/** Spatially varying boundary irradiance, one value per horizontal column. */
void evaluate_mapped_direct_beam_field_spatial_reference(
    const MappedDirectBeamGridView& grid,
    const IdealizedSolarGeometry& sun_direction,
    const double* beam_normal_irradiance_wm2_by_column,
    DirectBeamFieldWorkspace& workspace,
    DirectBeamFieldResult& result);

struct SlopedLambertianSurfaceFieldResult
{
    std::vector<double> absorbed_power_w;
    std::vector<double> reflected_power_w;
    std::vector<double> reflected_radiance_wm2sr;
    double incident_total_power_w = 0.0;
    double absorbed_total_power_w = 0.0;
    double reflected_total_power_w = 0.0;
    double closure_residual_w = 0.0;
};

struct SlopedLambertianSurfaceFieldWorkspace
{
    std::vector<double> scratch_absorbed_power_w;
    std::vector<double> scratch_reflected_power_w;
    std::vector<double> scratch_reflected_radiance_wm2sr;
};

/** Convert delivered direct-beam cell power into a sloped Lambertian source. */
void evaluate_sloped_lambertian_surface_field(
    const std::vector<double>& incident_power_w,
    const double* slope_x,
    const double* slope_y,
    std::size_t cell_count,
    double horizontal_cell_area_m2,
    double albedo,
    SlopedLambertianSurfaceFieldWorkspace& workspace,
    SlopedLambertianSurfaceFieldResult& result);

struct LambertianAngularQuadrature
{
    std::size_t cosine_nodes = 4;
    std::size_t azimuth_nodes = 8;
};

struct LambertianReflectionSweepResult
{
    std::vector<double> cell_absorbed_power_w;
    std::vector<double> surface_incident_power_w;
    double emitted_power_w = 0.0;
    double atmospheric_absorbed_power_w = 0.0;
    double surface_incident_total_power_w = 0.0;
    double top_escape_power_w = 0.0;
    double lateral_escape_power_w = 0.0;
    double closure_residual_w = 0.0;
    std::uint64_t rays = 0;
    std::uint64_t surface_rays = 0;
    std::uint64_t top_escape_rays = 0;
    std::uint64_t lateral_escape_rays = 0;
    std::uint64_t interface_values_validated = 0;
    std::uint64_t optical_values_validated = 0;
};

struct LambertianReflectionSweepWorkspace
{
    struct RayScratch
    {
        IdealizedSolarGeometry direction;
        double start_x_m = 0.0;
        double start_y_m = 0.0;
        double start_z_m = 0.0;
        double incident_power_w = 0.0;
        DirectBeamRayResult result;
        std::vector<DirectBeamPathSegment> segments;
        std::vector<DirectBeamSegmentAbsorption> absorption;
    };

    std::vector<double> scratch_cell_absorbed_power_w;
    std::vector<double> scratch_surface_incident_power_w;
    std::vector<double> cosine_nodes;
    std::vector<double> cosine_weights;
    std::vector<DirectBeamPathSegment> ray_segments;
    std::vector<DirectBeamSegmentAbsorption> ray_absorption;
    std::vector<RayScratch> ray_batch;
    std::vector<unsigned char> ray_batch_failures;
};

/** One conservative diffuse-reflection sweep through the mapped atmosphere. */
void evaluate_mapped_lambertian_reflection_sweep_reference(
    const MappedDirectBeamGridView& grid,
    const std::vector<double>& reflected_power_w,
    const double* slope_x,
    const double* slope_y,
    const LambertianAngularQuadrature& quadrature,
    LambertianReflectionSweepWorkspace& workspace,
    LambertianReflectionSweepResult& result);

struct LambertianMultipleReflectionOptions
{
    LambertianAngularQuadrature quadrature{};
    double albedo = 0.0;
    std::size_t maximum_bounces = 32;
    double relative_residual_tolerance = 1.0e-10;
    double absolute_residual_tolerance_w = 1.0e-9;
};

struct LambertianMultipleReflectionResult
{
    std::vector<double> cell_absorbed_power_w;
    std::vector<double> surface_absorbed_power_w;
    double initial_reflected_power_w = 0.0;
    double atmospheric_absorbed_power_w = 0.0;
    double surface_absorbed_power_w_total = 0.0;
    double top_escape_power_w = 0.0;
    double lateral_escape_power_w = 0.0;
    double unresolved_reflected_power_w = 0.0;
    double closure_residual_w = 0.0;
    std::uint64_t rays = 0;
    std::size_t completed_bounces = 0;
    bool converged = false;
    std::uint64_t interface_values_validated = 0;
    std::uint64_t optical_values_validated = 0;
};

struct LambertianMultipleReflectionWorkspace
{
    LambertianReflectionSweepWorkspace sweep_workspace;
    LambertianReflectionSweepResult sweep_result;
    std::vector<double> current_reflected_power_w;
    std::vector<double> next_reflected_power_w;
    std::vector<double> scratch_cell_absorbed_power_w;
    std::vector<double> scratch_surface_absorbed_power_w;
};

/** Iterate terrain re-reflection to an explicit residual-power tolerance. */
void evaluate_mapped_lambertian_multiple_reflection_reference(
    const MappedDirectBeamGridView& grid,
    const std::vector<double>& initial_reflected_power_w,
    const double* slope_x,
    const double* slope_y,
    const LambertianMultipleReflectionOptions& options,
    LambertianMultipleReflectionWorkspace& workspace,
    LambertianMultipleReflectionResult& result);

struct MappedTerrainShortwaveResult
{
    std::vector<double> atmospheric_absorbed_power_w;
    std::vector<double> surface_absorbed_power_w;
    double incoming_power_w = 0.0;
    double atmospheric_absorbed_power_w_total = 0.0;
    double surface_absorbed_power_w_total = 0.0;
    double top_escape_power_w = 0.0;
    double lateral_escape_power_w = 0.0;
    double unresolved_reflected_power_w = 0.0;
    double closure_residual_w = 0.0;
    bool reflection_converged = false;
};

struct MappedTerrainShortwaveWorkspace
{
    DirectBeamFieldWorkspace direct_workspace;
    DirectBeamFieldResult direct_result;
    SlopedLambertianSurfaceFieldWorkspace surface_workspace;
    SlopedLambertianSurfaceFieldResult surface_result;
    LambertianMultipleReflectionWorkspace reflection_workspace;
    LambertianMultipleReflectionResult reflection_result;
    std::vector<double> scratch_atmospheric_absorbed_power_w;
    std::vector<double> scratch_surface_absorbed_power_w;
};

/** Complete conservative terrain direct-plus-reflected shortwave operator. */
void evaluate_mapped_terrain_shortwave_reference(
    const MappedDirectBeamGridView& grid,
    const IdealizedSolarGeometry& sun_direction,
    double beam_normal_irradiance_wm2,
    const double* slope_x,
    const double* slope_y,
    double horizontal_cell_area_m2,
    const LambertianMultipleReflectionOptions& reflection_options,
    MappedTerrainShortwaveWorkspace& workspace,
    MappedTerrainShortwaveResult& result);

struct MappedSpectralShortwaveGridView
{
    const double* absorption_optical_depth = nullptr;
    const double* scattering_optical_depth = nullptr;
    const double* scattering_asymmetry = nullptr;
    const double* solar_source_fraction = nullptr;
    const double* z_interface_m = nullptr;
    std::size_t nx = 0;
    std::size_t ny = 0;
    std::size_t nz = 0;
    std::size_t g_points = 0;
    double dx_m = 0.0;
    double dy_m = 0.0;
    TerrainShadowBoundary x_boundary = TerrainShadowBoundary::Open;
    TerrainShadowBoundary y_boundary = TerrainShadowBoundary::Open;
};

struct MappedSpectralDirectResult
{
    std::vector<double> cell_absorbed_power_w;
    std::vector<double> cell_scattered_power_w;
    std::vector<double> surface_incident_power_w;
    double incoming_power_w = 0.0;
    double atmospheric_absorbed_power_w = 0.0;
    double unresolved_scattered_power_w = 0.0;
    double surface_incident_power_w_total = 0.0;
    double top_escape_power_w = 0.0;
    double lateral_escape_power_w = 0.0;
    double closure_residual_w = 0.0;
};

struct MappedSpectralDirectWorkspace
{
    std::vector<double> extinction_optical_depth;
    std::vector<double> boundary_irradiance_wm2;
    std::vector<double> scratch_cell_absorbed_power_w;
    std::vector<double> scratch_cell_scattered_power_w;
    std::vector<double> scratch_surface_incident_power_w;
    DirectBeamFieldWorkspace direct_workspace;
    DirectBeamFieldResult direct_result;
};

/** Trace every reduced g-point and separate absorption from scattering. */
void evaluate_mapped_spectral_direct_shortwave_reference(
    const MappedSpectralShortwaveGridView& grid,
    const IdealizedSolarGeometry& sun_direction,
    double beam_normal_irradiance_wm2,
    MappedSpectralDirectWorkspace& workspace,
    MappedSpectralDirectResult& result);

struct MappedSpectralTerrainResult
{
    std::vector<double> atmospheric_absorbed_power_w;
    std::vector<double> surface_absorbed_power_w;
    double incoming_power_w = 0.0;
    double atmospheric_absorbed_power_w_total = 0.0;
    double surface_absorbed_power_w_total = 0.0;
    double top_escape_power_w = 0.0;
    double lateral_escape_power_w = 0.0;
    double unresolved_scattered_power_w = 0.0;
    double closure_residual_w = 0.0;
    std::uint64_t rays = 0;
    std::size_t completed_iterations = 0;
    bool converged = false;
};

struct MappedSpectralTerrainWorkspace
{
    std::size_t geometry_cache_budget_bytes = mapped_geometry_cache_max_bytes;
    std::size_t geometry_cache_bytes = 0;
    std::uint64_t geometry_traces = 0;
    std::uint64_t geometry_cache_hits = 0;
    MappedSpectralDirectWorkspace direct_workspace;
    MappedSpectralDirectResult direct_result;
    std::vector<double> extinction_optical_depth;
    std::vector<IdealizedSolarGeometry> directions;
    std::vector<double> direction_weights;
    std::vector<double> phase_probabilities;
    std::vector<double> current_scattering;
    std::vector<double> next_directional_scattering;
    std::vector<double> outgoing_directional_source;
    std::vector<double> local_scattering_return;
    std::vector<double> direct_scattering;
    std::vector<double> current_surface_reflection;
    std::vector<double> next_surface_reflection;
    std::vector<double> scratch_atmospheric_absorbed_power_w;
    std::vector<double> scratch_surface_absorbed_power_w;
    std::vector<DirectBeamPathSegment> ray_segments;
    std::vector<DirectBeamSegmentAbsorption> ray_absorption;
};

/** Complete 112-point mapped discrete-ordinates scattering/terrain owner. */
void evaluate_mapped_spectral_terrain_shortwave_reference(
    const MappedSpectralShortwaveGridView& grid,
    const IdealizedSolarGeometry& sun_direction,
    double beam_normal_irradiance_wm2,
    const double* slope_x,
    const double* slope_y,
    double horizontal_cell_area_m2,
    const LambertianMultipleReflectionOptions& options,
    MappedSpectralTerrainWorkspace& workspace,
    MappedSpectralTerrainResult& result);

/**
 * @brief Computes direct-beam grey shortwave plus Lambertian surface return.
 */
void grey_sw_beer_lambert(
    const std::vector<double>& tau_layers,
    double mu0,
    double S0,
    double albedo_sfc,
    std::vector<double>& Fup,
    std::vector<double>& Fdn
);

/**
 * @brief Computes the optical depth profile.
 */
std::vector<double> compute_optical_depth_profile(
    const std::vector<double>& p_interfaces,
    double tau_ref,
    double n
);
void compute_optical_depth_profile(
    const std::vector<double>& p_interfaces,
    double tau_ref,
    double n,
    std::vector<double>& tau_profile
);

std::vector<double> diagnose_pressure_interfaces(
    const std::vector<double>& p_center,
    const std::vector<double>& z_center,
    const std::vector<double>& z_interfaces
);
void diagnose_pressure_interfaces(
    const std::vector<double>& p_center,
    const std::vector<double>& z_center,
    const std::vector<double>& z_interfaces,
    std::vector<double>& p_interfaces
);

/**
 * @brief Computes the layer optical depths.
 */
std::vector<double> compute_layer_optical_depths(
    const std::vector<double>& tau_profile,
    const std::vector<double>& dz
);
void compute_layer_optical_depths(
    const std::vector<double>& tau_profile,
    const std::vector<double>& dz,
    std::vector<double>& layer_optical_depths
);

/**
 * @brief Checks the energy conservation.
 */
double check_energy_conservation(
    const std::vector<double>& rho,
    const std::vector<double>& dz,
    const std::vector<double>& dTdt,
    const std::vector<double>& Fnet
);

} // namespace radiative_transfer
} // namespace tmv

namespace radiative_transfer = tmv::radiative_transfer;
