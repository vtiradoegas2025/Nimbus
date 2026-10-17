#pragma once

#include "rrtmg_atmosphere.hpp"
#include "rrtmg_coefficients.hpp"
#include "rrtmg_preprocessing.hpp"

#include <cstddef>
#include <vector>

namespace tmv::rrtmg
{

/** Layer-major reduced-g-point optical properties. */
struct LongwaveOptics
{
    std::size_t         layers   = 0;
    std::size_t         g_points = 0;
    std::vector<double> absorption_optical_depth;
    std::vector<double> planck_fraction;
};

/** Layer-major reduced-g-point solar optical properties. */
struct ShortwaveOptics
{
    std::size_t         layers   = 0;
    std::size_t         g_points = 0;
    std::vector<double> absorption_optical_depth;
    std::vector<double> rayleigh_optical_depth;
    // One normalized incident-solar source value per reduced g-point,
    // evaluated at the official band reference layer.
    std::vector<double> solar_source_fraction;
};

LongwaveOptics compute_longwave_gas_optics(const LongwaveCoefficients& coefficients,
                                           const PreparedAtmosphere&   atmosphere,
                                           const LongwavePreparation&  preparation);

ShortwaveOptics compute_shortwave_gas_optics(const ShortwaveCoefficients&           coefficients,
                                             const PreparedAtmosphere&              atmosphere,
                                             const std::vector<LayerInterpolation>& interpolation);

} // namespace tmv::rrtmg
