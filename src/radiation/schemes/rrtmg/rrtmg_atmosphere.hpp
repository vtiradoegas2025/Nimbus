#pragma once

#include "radiation/radiation_base.hpp"

#include <vector>

namespace tmv::rrtmg
{

struct GasProfile
{
    std::vector<double> volume_mixing_ratio;
    std::vector<double> column_molecules_cm2;
};

struct GasComposition
{
    GasProfile nitrogen;
    GasProfile ccl4;
    GasProfile cfc11;
    GasProfile cfc12;
    GasProfile cfc22;
    GasProfile water_vapor;
    GasProfile carbon_dioxide;
    GasProfile ozone;
    GasProfile nitrous_oxide;
    GasProfile carbon_monoxide;
    GasProfile methane;
    GasProfile oxygen;
};

/** Immutable, bottom-to-top atmospheric inputs consumed by RRTMG optics. */
struct PreparedAtmosphere
{
    double              surface_temperature_k = 0.0;
    std::vector<double> layer_pressure_pa;
    std::vector<double> interface_pressure_pa;
    std::vector<double> layer_temperature_k;
    std::vector<double> interface_temperature_k;
    std::vector<double> dry_air_column_molecules_cm2;
    GasComposition      gases;
    std::vector<double> cloud_fraction;
    std::vector<double> liquid_water_path_gm2;
    std::vector<double> ice_water_path_gm2;
    std::vector<double> liquid_effective_radius_um;
    std::vector<double> ice_effective_radius_um;
};

/**
 * Validate and translate a Nimbus radiation column into RRTMG currency.
 *
 * Nimbus qv is a mass mixing ratio relative to dry air. Condensate is a mass
 * mixing ratio carried by the resolved cell. Cloudy execution therefore uses
 * a deterministic 0/1 grid-resolved cloud fraction and physical layer water
 * paths; it does not claim a subgrid McICA closure.
 */
PreparedAtmosphere prepare_atmosphere(const RadiationConfig& config, const RadiationColumnStateView& column);

} // namespace tmv::rrtmg
