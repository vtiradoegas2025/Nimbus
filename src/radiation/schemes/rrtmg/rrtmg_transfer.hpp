#pragma once

#include "rrtmg_atmosphere.hpp"
#include "rrtmg_coefficients.hpp"
#include "rrtmg_cloud_optics.hpp"
#include "rrtmg_optics.hpp"
#include "rrtmg_preprocessing.hpp"

#include <vector>

namespace tmv::rrtmg
{

struct SpectralFluxes
{
    std::vector<double> upward_wm2;
    std::vector<double> downward_wm2;
};

SpectralFluxes solve_longwave(const LongwaveCoefficients& coefficients,
                              const PreparedAtmosphere&   atmosphere,
                              const LongwavePreparation&  preparation,
                              const LongwaveOptics&       optics,
                              double                      surface_emissivity,
                              const LongwaveCloudOptics*  clouds = nullptr);

SpectralFluxes solve_shortwave(const ShortwaveCoefficients& coefficients,
                               const ShortwaveOptics&       optics,
                               double                       cosine_zenith,
                               double                       solar_irradiance_wm2,
                               double                       surface_albedo,
                               const ShortwaveCloudOptics* clouds = nullptr);

} // namespace tmv::rrtmg
