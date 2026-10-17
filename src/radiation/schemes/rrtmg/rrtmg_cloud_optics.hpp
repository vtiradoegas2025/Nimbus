#pragma once

#include "rrtmg_atmosphere.hpp"
#include "rrtmg_coefficients.hpp"

#include <cstddef>
#include <vector>

namespace tmv::rrtmg
{

struct LongwaveCloudOptics
{
    std::size_t         layers = 0;
    std::vector<double> absorption_optical_depth; // [layer][16 bands]
};

struct ShortwaveCloudOptics
{
    std::size_t         layers = 0;
    std::vector<double> original_optical_depth; // [layer][14 bands]
    std::vector<double> optical_depth;          // delta-scaled
    std::vector<double> single_scattering_albedo;
    std::vector<double> asymmetry;
};

LongwaveCloudOptics compute_longwave_cloud_optics(const PreparedAtmosphere& atmosphere);

ShortwaveCloudOptics compute_shortwave_cloud_optics(const ShortwaveCoefficients& coefficients,
                                                     const PreparedAtmosphere&   atmosphere);

} // namespace tmv::rrtmg
