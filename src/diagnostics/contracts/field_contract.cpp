/**
 * @file field_contract.cpp
 * @brief Implementation for the validation module.
 *
 * Provides executable logic for the validation runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/validation subsystem.
 */

#include "diagnostics/field_contract.hpp"
#include "util/string_utils.hpp"

#include <algorithm>

namespace tmv
{
namespace
{

bool ascii_case_equal(std::string_view left, std::string_view right)
{
    if (left.size() != right.size())
        return false;
    for (std::size_t index = 0; index < left.size(); ++index)
    {
        const unsigned char lhs       = static_cast<unsigned char>(left[index]);
        const unsigned char rhs       = static_cast<unsigned char>(right[index]);
        const unsigned char lhs_lower = lhs >= 'A' && lhs <= 'Z' ? static_cast<unsigned char>(lhs - 'A' + 'a') : lhs;
        const unsigned char rhs_lower = rhs >= 'A' && rhs <= 'Z' ? static_cast<unsigned char>(rhs - 'A' + 'a') : rhs;
        if (lhs_lower != rhs_lower)
            return false;
    }
    return true;
}

/**
 * @brief Builds an inclusive min/max bounds descriptor.
 */
FieldBounds bounds(double min_value, double max_value)
{
    FieldBounds out;
    out.has_min   = true;
    out.has_max   = true;
    out.min_value = min_value;
    out.max_value = max_value;
    return out;
}

FieldBounds lower_bound(double min_value)
{
    FieldBounds out;
    out.has_min   = true;
    out.min_value = min_value;
    return out;
}

std::vector<FieldContract> build_contracts()
{
    std::vector<FieldContract> contracts = {
        {"u",
         "m/s",
         "Radial wind component",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-180.0, 180.0),
         {},
         {"ur", "u_r"},
         FieldRequirementTier::RequiredNow},
        {"v",
         "m/s",
         "Azimuthal wind component",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-180.0, 180.0),
         {},
         {"v", "uth"},
         FieldRequirementTier::RequiredNow},
        {"w",
         "m/s",
         "Vertical wind component",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-120.0, 120.0),
         {},
         {"wz", "vertical_velocity"},
         FieldRequirementTier::RequiredNow},
        {"rho",
         "kg/m^3",
         "Prognostic dry-air density",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.05, 2.0),
         {},
         {},
         FieldRequirementTier::RequiredNow},
        {"p",
         "Pa",
         "Pressure",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(1000.0, 200000.0),
         {},
         {},
         FieldRequirementTier::RequiredNow},
        {"theta",
         "K",
         "Potential temperature",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(200.0, 700.0),
         {},
         {"potential_temperature"},
         FieldRequirementTier::RequiredNow},
        {"qv",
         "kg/kg",
         "Water vapor dry-air mixing ratio",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 0.05),
         {},
         {},
         FieldRequirementTier::RequiredNow},
        {"qc",
         "kg/kg",
         "Cloud water dry-air mixing ratio",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 0.05),
         {},
         {},
         FieldRequirementTier::RequiredNow},
        {"qr",
         "kg/kg",
         "Rain water dry-air mixing ratio",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 0.05),
         {},
         {},
         FieldRequirementTier::RequiredNow},
        {"qh",
         "kg/kg",
         "Hail dry-air mixing ratio",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 0.05),
         {},
         {},
         FieldRequirementTier::RequiredNow},
        {"qg",
         "kg/kg",
         "Graupel dry-air mixing ratio",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 0.05),
         {},
         {},
         FieldRequirementTier::RequiredNow},
        {"radar",
         "mm^6/m^3",
         "Radar reflectivity (linear units)",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 1.0e12),
         {},
         {"radar_reflectivity", "reflectivity_linear"},
         FieldRequirementTier::RequiredNow},
        {"tracer",
         "1",
         "Passive tracer",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 1.5),
         {},
         {},
         FieldRequirementTier::RequiredNow},

        {"tke",
         "m^2/s^2",
         "Prognostic turbulent kinetic energy per unit mass",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         lower_bound(0.0),
         {},
         {},
         FieldRequirementTier::RequiredNow},

        {"qi",
         "kg/kg",
         "Cloud ice dry-air mixing ratio",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 0.05),
         {},
         {},
         FieldRequirementTier::RequiredNow},
        {"qs",
         "kg/kg",
         "Snow dry-air mixing ratio",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 0.05),
         {},
         {"snow_mixing_ratio"},
         FieldRequirementTier::RequiredNow},
        {"vorticity_r",
         "1/s",
         "Cartesian x-component of vorticity (legacy r field id)",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-10.0, 10.0),
         {},
         {}},
        {"vorticity_theta",
         "1/s",
         "Cartesian y-component of vorticity (legacy theta field id)",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-10.0, 10.0),
         {},
         {}},
        {"vorticity_z",
         "1/s",
         "Vertical vorticity",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-10.0, 10.0),
         {},
         {"zeta", "vertical_vorticity"},
         FieldRequirementTier::RequiredNow},
        {"stretching_term",
         "1/s^2",
         "Vorticity stretching term",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-50.0, 50.0),
         {},
         {},
         FieldRequirementTier::RequiredNow},
        {"tilting_term",
         "1/s^2",
         "Vorticity tilting term",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-50.0, 50.0),
         {},
         {},
         FieldRequirementTier::RequiredNow},
        {"baroclinic_term",
         "1/s^2",
         "Baroclinic vorticity generation",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-50.0, 50.0),
         {},
         {},
         FieldRequirementTier::RequiredNow},
        {"p_prime",
         "Pa",
         "Perturbation pressure",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-100000.0, 100000.0),
         {},
         {"pressure_perturbation"},
         FieldRequirementTier::RequiredNow},
        {"dynamic_pressure",
         "Pa",
         "Anelastic pressure potential induced by resolved velocity advection; level-local volume-mean gauge",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-100000.0, 100000.0),
         {},
         {}},
        {"buoyancy_pressure",
         "Pa",
         "Anelastic pressure potential induced by buoyancy; level-local volume-mean gauge",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-100000.0, 100000.0),
         {},
         {}},
        {"angular_momentum",
         "m^2/s",
         "Specific vertical angular momentum about the physical domain center",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"angular_momentum_tendency",
         "m^2/s^2",
         "Resolved inviscid Eulerian angular-momentum tendency: advection, pressure and perturbation Coriolis",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},

        {"horizontal_vorticity_streamwise",
         "1/s",
         "Horizontal vorticity (streamwise)",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-50.0, 50.0),
         {},
         {}},
        {"horizontal_vorticity_crosswise",
         "1/s",
         "Horizontal vorticity (crosswise)",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-50.0, 50.0),
         {},
         {}},
        {"vorticity_magnitude",
         "1/s",
         "Vorticity magnitude",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 50.0),
         {},
         {}},
        {"divergence",
         "1/s",
         "Flow divergence",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-5.0, 5.0),
         {},
         {}},
        {"potential_vorticity",
         "K m^2/(kg s)",
         "Ertel potential vorticity",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-1.0, 1.0),
         {},
         {}},
        {"pressure_gradient_force_x",
         "m/s^2",
         "Pressure gradient force x-component",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-200.0, 200.0),
         {},
         {}},
        {"pressure_gradient_force_y",
         "m/s^2",
         "Pressure gradient force y-component",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-200.0, 200.0),
         {},
         {}},
        {"pressure_gradient_force_z",
         "m/s^2",
         "Pressure gradient force z-component",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-200.0, 200.0),
         {},
         {}},
        {"buoyancy",
         "m/s^2",
         "Cell-centered density buoyancy with vapor correction and condensate loading",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-10.0, 10.0),
         {},
         {}},
        {"theta_prime",
         "K",
         "Perturbation potential temperature",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-50.0, 50.0),
         {},
         {}},
        {"theta_v",
         "K",
         "Virtual potential temperature",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(180.0, 550.0),
         {},
         {}},
        {"theta_e",
         "K",
         "Equivalent potential temperature",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(180.0, 800.0),
         {},
         {}},
        {"theta_w",
         "K",
         "Wet-bulb potential temperature",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(180.0, 800.0),
         {},
         {}},
        {"temperature",
         "K",
         "Temperature",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(180.0, 360.0),
         {},
         {"t"}},
        {"dewpoint",
         "K",
         "Dewpoint temperature",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(150.0, 330.0),
         {},
         {"td"}},
        {"relative_humidity",
         "%",
         "Relative humidity",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 200.0),
         {},
         {"rh"}},
        {"saturation_mixing_ratio",
         "kg/kg",
         "Saturation mixing ratio",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 0.10),
         {},
         {"qvs", "qsat"}},
        {"total_condensate",
         "kg/kg",
         "Total condensate (qc+qr+qi+qs+qg+qh, suspended and falling)",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 0.10),
         {},
         {"qt"}},
        {"cloud_condensate",
         "kg/kg",
         "Cloud condensate (qc+qi+qs, suspended cloud)",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 0.10),
         {},
         {"qcloud"}},
        {"cloud_liquid_condensate",
         "kg/kg",
         "Cloud liquid condensate (qc)",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 0.10),
         {},
         {"qcloud_liquid"}},
        {"cloud_frozen_condensate",
         "kg/kg",
         "Frozen suspended condensate (qi+qs)",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 0.10),
         {},
         {"qcloud_frozen"}},
        {"reflectivity_dbz",
         "dBZ",
         "Radar reflectivity dBZ",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-30.0, 120.0),
         {},
         {"z"}},
        {"radial_velocity",
         "m/s",
         "Radar line-of-sight scatterer velocity",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-325.0, 325.0),
         {},
         {"vr"}},
        {"zdr",
         "dB",
         "Differential reflectivity",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-5.0, 10.0),
         {},
         {"zdr_db"}},
        {"kdp",
         "deg/km",
         "Specific differential phase",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 4.0),
         {},
         {}},
        {"rhohv",
         "1",
         "Correlation coefficient",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.8, 1.0),
         {},
         {}},
        {"streamlines",
         "vector",
         "Streamline diagnostics",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"trajectory_paths",
         "path",
         "Trajectory path diagnostics",
         FieldDimensionality::Trajectory,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"storm_relative_winds",
         "m/s",
         "Storm-relative horizontal wind speed (magnitude proxy)",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 200.0),
         {},
         {}},
        {"helicity_density",
         "m/s^2",
         "Helicity density",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-200.0, 200.0),
         {},
         {}},
        {"q_vectors",
         "vector",
         "Q-vector diagnostic",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"okubo_weiss",
         "1/s^2",
         "Okubo-Weiss parameter",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         bounds(-500.0, 500.0),
         {},
         {}},
        {"turbulent_diffusion_term",
         "varies",
         "Turbulent diffusion term",
         FieldDimensionality::Volume3D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},

        {"u10",
         "m/s",
         "10 m u wind",
         FieldDimensionality::Surface2D,
         FieldImplementationStatus::ExportedNow,
         bounds(-180.0, 180.0),
         {},
         {}},
        {"v10",
         "m/s",
         "10 m v wind",
         FieldDimensionality::Surface2D,
         FieldImplementationStatus::ExportedNow,
         bounds(-180.0, 180.0),
         {},
         {}},
        {"t2",
         "K",
         "2 m temperature",
         FieldDimensionality::Surface2D,
         FieldImplementationStatus::ExportedNow,
         bounds(180.0, 360.0),
         {},
         {}},
        {"td2",
         "K",
         "2 m dewpoint",
         FieldDimensionality::Surface2D,
         FieldImplementationStatus::ExportedNow,
         bounds(150.0, 330.0),
         {},
         {}},
        {"surface_pressure_perturbation",
         "Pa",
         "Surface pressure perturbation",
         FieldDimensionality::Surface2D,
         FieldImplementationStatus::ExportedNow,
         bounds(-10000.0, 10000.0),
         {},
         {}},
        {"surface_sensible_heat_flux",
         "W/m^2",
         "Surface sensible heat flux",
         FieldDimensionality::Surface2D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"surface_latent_heat_flux",
         "W/m^2",
         "Surface latent heat flux",
         FieldDimensionality::Surface2D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"surface_moisture_flux",
         "kg/m^2/s",
         "Surface moisture flux",
         FieldDimensionality::Surface2D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"skin_temperature",
         "K",
         "Surface skin temperature",
         FieldDimensionality::Surface2D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"cold_pool_boundary",
         "K",
         "Positive first-physical-level density-potential-temperature deficit relative to the level-matched initialized base state (surface_density_potential_temperature_deficit_v1)",
         FieldDimensionality::Surface2D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 100.0),
         {},
         {}},
        {"surface_friction_velocity",
         "m/s",
         "Surface friction velocity",
         FieldDimensionality::Surface2D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 10.0),
         {},
         {"ustar"}},
        {"surface_stress_u",
         "N/m^2",
         "Surface stress u-component",
         FieldDimensionality::Surface2D,
         FieldImplementationStatus::ExportedNow,
         bounds(-100.0, 100.0),
         {},
         {}},
        {"surface_stress_v",
         "N/m^2",
         "Surface stress v-component",
         FieldDimensionality::Surface2D,
         FieldImplementationStatus::ExportedNow,
         bounds(-100.0, 100.0),
         {},
         {}},
        {"pbl_height",
         "m",
         "Planetary boundary-layer height",
         FieldDimensionality::Surface2D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 20000.0),
         {},
         {}},

        {"composite_reflectivity",
         "dBZ",
         "Column max reflectivity",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         bounds(-30.0, 120.0),
         {},
         {}},
        {"column_max_w",
         "m/s",
         "Column max vertical velocity",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         bounds(-120.0, 120.0),
         {},
         {}},
        {"column_max_vorticity",
         "1/s",
         "Column max vorticity",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 50.0),
         {},
         {}},
        {"accumulated_rainfall",
         "mm",
         "Accumulated rainfall",
         FieldDimensionality::Surface2D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 10000.0),
         {},
         {}},
        {"precip_rate",
         "mm/h",
         "Instantaneous precipitation rate",
         FieldDimensionality::Surface2D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 2000.0),
         {},
         {}},
        {"vil",
         "kg/m^2",
         "Vertically integrated liquid",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 200.0),
         {},
         {}},
        {"precipitable_water",
         "mm",
         "Precipitable water depth",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 1000.0),
         {},
         {}},
        {"liquid_water_path",
         "kg/m^2",
         "Column-integrated cloud plus rain liquid water path",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 1000.0),
         {},
         {}},
        {"cloud_water_path",
         "kg/m^2",
         "Column-integrated cloud water path",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 1000.0),
         {},
         {}},
        {"cloud_top_height",
         "m",
         "Cloud top height",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"cloud_base_height",
         "m",
         "Cloud base height",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"lcl",
         "m",
         "Lifted condensation level",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"lfc",
         "m",
         "Level of free convection",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"el",
         "m",
         "Equilibrium level",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},

        {"cape",
         "J/kg",
         "Convective available potential energy",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"cin",
         "J/kg",
         "Convective inhibition",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"lifted_index",
         "K",
         "Lifted index",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"k_index", "K", "K-index", FieldDimensionality::Column2D, FieldImplementationStatus::ExportedNow, {}, {}, {}},
        {"showalter_index",
         "K",
         "Showalter index",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"total_totals",
         "K",
         "Total totals index",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"srh_0_1km",
         "m^2/s^2",
         "Storm-relative helicity over resolved winds in 0-1 km AGL",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         bounds(-5000.0, 5000.0),
         {},
         {}},
        {"srh_0_3km",
         "m^2/s^2",
         "Storm-relative helicity over resolved winds in 0-3 km AGL",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         bounds(-5000.0, 5000.0),
         {},
         {}},
        {"updraft_helicity",
         "m^2/s^2",
         "Updraft helicity over the model intersection with 2-5 km AGL, cell-midpoint integral of w*vorticity_z",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         bounds(-50000.0, 50000.0),
         {},
         {"uh", "uh_2_5km"}},
        {"ehi",
         "1",
         "Energy-helicity index",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"scp",
         "1",
         "Supercell composite parameter",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"stp",
         "1",
         "Significant tornado parameter",
         FieldDimensionality::Column2D,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},

        {"cross_section",
         "varies",
         "Arbitrary cross-section diagnostics",
         FieldDimensionality::CrossSection,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"rhi_slice",
         "varies",
         "RHI-style slice",
         FieldDimensionality::CrossSection,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"hodograph_aligned_cross_section",
         "varies",
         "Along-hodograph cross section",
         FieldDimensionality::CrossSection,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},

        {"ppi_sweep",
         "radar",
         "Simulated PPI sweep",
         FieldDimensionality::RadarSynthetic,
         FieldImplementationStatus::ExportedNow,
         bounds(-30.0, 120.0),
         {},
         {}},
        {"rhi_sweep",
         "radar",
         "Simulated RHI sweep",
         FieldDimensionality::RadarSynthetic,
         FieldImplementationStatus::ExportedNow,
         bounds(-30.0, 120.0),
         {},
         {}},
        {"bwer",
         "1",
         "Bounded weak echo region diagnostic",
         FieldDimensionality::RadarSynthetic,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 1.0),
         {},
         {}},
        {"mesocyclone_diagnostic",
         "1",
         "Mesocyclone detection diagnostic",
         FieldDimensionality::RadarSynthetic,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 1.0),
         {},
         {}},
        {"vrot",
         "m/s",
         "Rotational velocity",
         FieldDimensionality::RadarSynthetic,
         FieldImplementationStatus::ExportedNow,
         bounds(0.0, 200.0),
         {},
         {}},

        {"forward_trajectories",
         "path",
         "Forward trajectories",
         FieldDimensionality::Trajectory,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"backward_trajectories",
         "path",
         "Backward trajectories",
         FieldDimensionality::Trajectory,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"parcel_buoyancy_trajectory",
         "m/s^2",
         "Parcel buoyancy along trajectory",
         FieldDimensionality::Trajectory,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"vorticity_trajectory",
         "1/s",
         "Vorticity along trajectory",
         FieldDimensionality::Trajectory,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
        {"circulation_material_surface",
         "m^2/s",
         "Circulation on material surfaces",
         FieldDimensionality::Trajectory,
         FieldImplementationStatus::ExportedNow,
         {},
         {},
         {}},
    };

    // Publication metadata is applied in one place so the pre-existing bounds
    // and validation table remains readable. Nimbus writes every 3-D history
    // variable on scalar cell centers; C-grid velocities are centered before
    // export (their native model staggering is recorded by the grid contract).
    for (auto& contract : contracts)
    {
        if (contract.id == "lifted_index" || contract.id == "showalter_index"
            || contract.id == "k_index" || contract.id == "total_totals")
        {
            contract.allows_missing_values = true;
            contract.description += "; NaN where required pressure levels are absent";
        }
        if (contract.id == "cloud_top_height" || contract.id == "cloud_base_height"
            || contract.id == "lcl" || contract.id == "lfc" || contract.id == "el")
            contract.description += " above local ground";
        switch (contract.dimensionality)
        {
        case FieldDimensionality::Surface2D:
            contract.grid_location = FieldGridLocation::SurfaceCellCenter;
            break;
        case FieldDimensionality::Column2D:
        case FieldDimensionality::CrossSection:
            contract.grid_location = FieldGridLocation::ColumnCellCenter;
            break;
        case FieldDimensionality::Trajectory:
            contract.grid_location = FieldGridLocation::Trajectory;
            break;
        case FieldDimensionality::RadarSynthetic:
            contract.grid_location = FieldGridLocation::Synthetic;
            break;
        case FieldDimensionality::Volume3D:
            contract.grid_location = FieldGridLocation::CellCenter;
            break;
        }

        if (contract.id == "u")
        {
            contract.description    = "Cartesian east-west wind component";
            contract.cm1_aliases    = {"uinterp"};
            contract.state_semantic = FieldStateSemantic::TotalState;
        }
        else if (contract.id == "v")
        {
            contract.description    = "Cartesian south-north wind component";
            contract.cm1_aliases    = {"vinterp"};
            contract.state_semantic = FieldStateSemantic::TotalState;
        }
        else if (contract.id == "w")
        {
            contract.cm1_aliases    = {"winterp"};
            contract.state_semantic = FieldStateSemantic::TotalState;
        }
        else if (contract.id == "p")
        {
            contract.description    = "Total pressure";
            contract.cm1_aliases    = {"prs"};
            contract.state_semantic = FieldStateSemantic::TotalState;
        }
        else if (contract.id == "theta")
        {
            contract.description    = "Total potential temperature";
            contract.cm1_aliases    = {"th"};
            contract.state_semantic = FieldStateSemantic::TotalState;
        }
        else if (contract.id == "rho")
        {
            contract.cm1_aliases    = {"rho"};
            contract.state_semantic = FieldStateSemantic::TotalState;
        }
        else if (contract.id == "qv")
        {
            contract.cm1_aliases    = {"qv"};
            contract.state_semantic = FieldStateSemantic::TotalState;
        }
        else if (contract.id == "qc" || contract.id == "qr" || contract.id == "qi" || contract.id == "qs" ||
                 contract.id == "qg" || contract.id == "qh")
        {
            // CM1 hydrometeor qname identities depend on the selected
            // microphysics scheme. Preserve total-state semantics without
            // advertising an unqualified history-file alias.
            contract.state_semantic = FieldStateSemantic::TotalState;
        }
        else if (contract.id == "p_prime")
        {
            contract.cm1_aliases    = {"prspert"};
            contract.state_semantic = FieldStateSemantic::Perturbation;
        }
        else if (contract.id == "theta_prime")
        {
            contract.cm1_aliases    = {"thpert"};
            contract.state_semantic = FieldStateSemantic::Perturbation;
        }
        else if (contract.id == "reflectivity_dbz")
        {
            contract.cm1_aliases = {"dbz"};
        }
        else if (contract.id == "vorticity_z")
        {
            contract.cm1_aliases = {"zvort"};
        }
        else if (contract.id == "buoyancy")
        {
            contract.cm1_aliases = {"buoyancy"};
        }
        else if (contract.id == "potential_vorticity")
        {
            contract.cm1_aliases = {"pv"};
        }
        else if (contract.id == "composite_reflectivity")
        {
            contract.cm1_aliases = {"cref"};
        }
        else if (contract.id == "accumulated_rainfall")
        {
            contract.cm1_aliases = {"rain"};
        }
        else if (contract.id == "precip_rate")
        {
            contract.cm1_aliases = {"prate"};
        }
        else if (contract.id == "updraft_helicity")
        {
            contract.cm1_aliases = {"uh"};
        }
        else if (contract.id == "precipitable_water")
        {
            contract.cm1_aliases = {"pwat"};
        }
        else if (contract.id == "liquid_water_path")
        {
            contract.cm1_aliases = {"lwp"};
        }
        else if (contract.id == "cloud_water_path")
        {
            contract.cm1_aliases = {"cwp"};
        }
        else if (contract.id == "cape" || contract.id == "cin" || contract.id == "lcl" || contract.id == "lfc")
        {
            contract.cm1_aliases = {contract.id};
        }
        else if (contract.id.find("tendency") != std::string::npos || contract.id.find("_term") != std::string::npos)
        {
            contract.state_semantic = FieldStateSemantic::Tendency;
        }
    }

    return contracts;
}

const std::vector<FieldContract> kContracts = build_contracts();

} // namespace

/**
 * @brief Returns the canonical field contract inventory.
 */
const std::vector<FieldContract>& cm1_field_contracts()
{
    return kContracts;
}

/**
 * @brief Finds a field contract by canonical id or known alias.
 */
const FieldContract* find_field_contract(std::string_view id_or_alias)
{
    for (const auto& contract : kContracts)
    {
        if (ascii_case_equal(contract.id, id_or_alias))
        {
            return &contract;
        }

        for (const auto& alias : contract.aliases)
        {
            if (ascii_case_equal(alias, id_or_alias))
            {
                return &contract;
            }
        }
    }

    return nullptr;
}

/**
 * @brief Returns all contracts matching the provided implementation status.
 */
std::vector<const FieldContract*> contracts_with_status(FieldImplementationStatus status)
{
    std::vector<const FieldContract*> out;
    out.reserve(kContracts.size());

    for (const auto& contract : kContracts)
    {
        if (contract.status == status)
        {
            out.push_back(&contract);
        }
    }

    return out;
}

/**
 * @brief Returns all contracts matching the provided requirement tier.
 */
std::vector<const FieldContract*> contracts_with_requirement(FieldRequirementTier requirement)
{
    std::vector<const FieldContract*> out;
    out.reserve(kContracts.size());

    for (const auto& contract : kContracts)
    {
        if (contract.requirement == requirement)
        {
            out.push_back(&contract);
        }
    }

    return out;
}

/**
 * @brief Lists required-now fields that are still marked not implemented.
 */
std::vector<std::string> known_not_implemented_required_now_inventory()
{
    std::vector<std::string> out;
    for (const auto& contract : kContracts)
    {
        if (contract.status == FieldImplementationStatus::NotImplemented &&
            contract.requirement == FieldRequirementTier::RequiredNow)
        {
            out.push_back(contract.id);
        }
    }
    return out;
}

/**
 * @brief Lists report-only backlog fields that are still not implemented.
 */
std::vector<std::string> known_not_implemented_backlog_inventory()
{
    std::vector<std::string> out;
    for (const auto& contract : kContracts)
    {
        if (contract.status == FieldImplementationStatus::NotImplemented &&
            contract.requirement == FieldRequirementTier::ReportOnly)
        {
            out.push_back(contract.id);
        }
    }
    return out;
}

/**
 * @brief Converts field dimensionality enum to stable string id.
 */
const char* to_string(FieldDimensionality value)
{
    switch (value)
    {
    case FieldDimensionality::Volume3D:
        return "3d";
    case FieldDimensionality::Surface2D:
        return "surface_2d";
    case FieldDimensionality::Column2D:
        return "column_2d";
    case FieldDimensionality::CrossSection:
        return "cross_section";
    case FieldDimensionality::Trajectory:
        return "trajectory";
    case FieldDimensionality::RadarSynthetic:
        return "radar_synthetic";
    default:
        return "unknown";
    }
}

/**
 * @brief Converts implementation status enum to stable string id.
 */
const char* to_string(FieldImplementationStatus value)
{
    switch (value)
    {
    case FieldImplementationStatus::ExportedNow:
        return "exported_now";
    case FieldImplementationStatus::ComputedNotExported:
        return "computed_not_exported";
    case FieldImplementationStatus::NotImplemented:
        return "not_implemented";
    default:
        return "unknown";
    }
}

/**
 * @brief Converts requirement-tier enum to stable string id.
 */
const char* to_string(FieldRequirementTier value)
{
    switch (value)
    {
    case FieldRequirementTier::RequiredNow:
        return "required_now";
    case FieldRequirementTier::ReportOnly:
        return "report_only";
    default:
        return "unknown";
    }
}

const char* to_string(FieldGridLocation value)
{
    switch (value)
    {
    case FieldGridLocation::CellCenter:
        return "cell_center";
    case FieldGridLocation::SurfaceCellCenter:
        return "surface_cell_center";
    case FieldGridLocation::ColumnCellCenter:
        return "column_cell_center";
    case FieldGridLocation::Trajectory:
        return "trajectory";
    case FieldGridLocation::Synthetic:
        return "synthetic";
    default:
        return "unknown";
    }
}

const char* to_string(FieldStateSemantic value)
{
    switch (value)
    {
    case FieldStateSemantic::TotalState:
        return "total_state";
    case FieldStateSemantic::Perturbation:
        return "perturbation";
    case FieldStateSemantic::Diagnostic:
        return "diagnostic";
    case FieldStateSemantic::Tendency:
        return "tendency";
    default:
        return "unknown";
    }
}

} // namespace tmv
