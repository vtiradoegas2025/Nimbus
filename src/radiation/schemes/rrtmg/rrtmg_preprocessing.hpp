#pragma once

#include "rrtmg_atmosphere.hpp"
#include "rrtmg_coefficients.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace tmv::rrtmg
{

struct LayerInterpolation
{
    std::size_t           pressure_index                      = 0;
    std::size_t           temperature_index_at_lower_pressure = 0;
    std::size_t           temperature_index_at_upper_pressure = 0;
    std::array<double, 4> pressure_temperature_weights{};
    bool                  lower_atmosphere = true;

    std::size_t water_self_index           = 0;
    std::size_t water_foreign_index        = 0;
    std::size_t minor_temperature_index    = 0;
    double      water_self_fraction        = 0.0;
    double      water_foreign_fraction     = 0.0;
    double      minor_temperature_fraction = 0.0;
    double      water_self_factor          = 0.0;
    double      water_foreign_factor       = 0.0;
    double      minor_scale                = 0.0;
    double      nitrogen_minor_scale       = 0.0;
};

struct LongwavePreparation
{
    std::vector<LayerInterpolation> interpolation;
    std::vector<double>             layer_planck;
    std::vector<double>             interface_planck;
    std::vector<double>             surface_planck;
    double                          precipitable_water_cm = 0.0;
};

LongwavePreparation prepare_longwave(const LongwaveCoefficients& coefficients,
                                     const PreparedAtmosphere&   atmosphere,
                                     double                      surface_emissivity);

} // namespace tmv::rrtmg
