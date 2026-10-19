#pragma once

#include "core/field/field3d.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "terrain/terrain_base.hpp"

#include <vector>

namespace tmv::diagnostics
{

/**
 * Read-only geometry and velocity view for the storm updraft-helicity
 * diagnostic. The output is packed as (i-i_begin)*ny+(j-j_begin).
 */
struct StormRotationGridView
{
    const Field3D* u = nullptr;
    const Field3D* v = nullptr;
    const Field3D* w = nullptr;

    PhysicalCellDomain domain{};
    StaggerType        staggering = StaggerType::Collocated;
    double             dx_m       = 0.0;
    double             dy_m       = 0.0;
    double             dz_m       = 0.0;

    // Active terrain geometry, or nullptr for a flat Cartesian grid.
    const TerrainMetrics3D* terrain = nullptr;

    // Optional authoritative flat-grid vertical coordinates. If omitted,
    // uniform dz_m centers/interfaces are used.
    const std::vector<double>* flat_z_center_m    = nullptr;
    const std::vector<double>* flat_z_interface_m = nullptr;

    // Required only for terrain C-grid columns whose k=0 cell contributes.
    // Packed over the complete storage plane as i*storage_ny+j.
    const std::vector<double>* cgrid_ground_w_ms = nullptr;
};

struct StormRotationLayer
{
    double bottom_m_agl = 2000.0;
    double top_m_agl    = 5000.0;
};

const char* storm_updraft_helicity_algorithm_id() noexcept;

/**
 * Compute signed updraft helicity, integral(w*zeta dz), in m^2/s^2.
 *
 * Horizontal velocity is first collocated at scalar centers. Vertical
 * vorticity is evaluated at constant physical height; mapped-grid derivatives
 * therefore include the terrain-coordinate chain-rule correction. Cell
 * contributions use exact overlap with the requested AGL layer and midpoint
 * w*zeta values. Every physical column must cover the complete layer.
 *
 * This routine is allocation-free after the caller sizes `output_m2s2` to
 * domain.nx()*domain.ny(). Invalid fields, geometry, or incomplete layer
 * coverage fail closed with an exception; they are never converted to zero.
 */
void compute_storm_updraft_helicity_columns(
    const StormRotationGridView& grid,
    const StormRotationLayer&    layer,
    std::vector<double>&         output_m2s2);

} // namespace tmv::diagnostics
