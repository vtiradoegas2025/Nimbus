#pragma once

#include "boundary_layer/boundary_layer_base.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tmv::boundary_layer::cloud_top
{

struct RadiativeMixingProfile
{
    bool active = false;
    double cloud_top_height_m = 0.0;
    double cooling_flux_kms = 0.0;
    double velocity_scale_ms = 0.0;
    std::vector<double> diffusivity_m2s;
};

inline RadiativeMixingProfile diagnose_surface_coupled_profile(
    const BoundaryLayerProfileView& theta,
    const BoundaryLayerProfileView& qc,
    const BoundaryLayerProfileView& qi,
    const BoundaryLayerProfileView& radiative_theta_tendency,
    const BoundaryLayerProfileView& z_interface,
    double diagnosed_pbl_height_m,
    double maximum_diffusivity_m2s)
{
    const std::size_t nz = theta.size();
    if (!qc || !qi || !radiative_theta_tendency
        || qc.size() != nz || qi.size() != nz
        || radiative_theta_tendency.size() != nz
        || z_interface.size() != nz + 1)
    {
        throw std::invalid_argument(
            "cloud-top radiative mixing requires aligned qc, qi, radiation, and geometry profiles");
    }
    if (!std::isfinite(diagnosed_pbl_height_m)
        || diagnosed_pbl_height_m <= 0.0
        || !std::isfinite(maximum_diffusivity_m2s)
        || maximum_diffusivity_m2s <= 0.0)
    {
        throw std::invalid_argument(
            "cloud-top radiative mixing requires finite positive bounds");
    }

    RadiativeMixingProfile result;
    result.diffusivity_m2s.assign(nz + 1, 0.0);
    std::size_t cloud_top_cell = nz;
    for (std::size_t k = 0; k < nz; ++k)
    {
        const double liquid = qc[k];
        const double ice = qi[k];
        const double cooling = radiative_theta_tendency[k];
        if (!std::isfinite(liquid) || liquid < 0.0
            || !std::isfinite(ice) || ice < 0.0
            || !std::isfinite(cooling))
        {
            throw std::invalid_argument(
                "cloud-top radiative mixing received non-finite or negative cloud state");
        }
        const double center = 0.5 * (z_interface[k] + z_interface[k + 1]);
        if (center <= diagnosed_pbl_height_m && liquid + ice > 0.0)
            cloud_top_cell = k;
    }
    if (cloud_top_cell == nz)
        return result;

    const double cloud_top_height = std::min(
        z_interface[cloud_top_cell + 1], diagnosed_pbl_height_m);
    double integrated_radiative_tendency = 0.0;
    for (std::size_t k = 0; k <= cloud_top_cell; ++k)
    {
        const double lower = z_interface[k];
        const double upper = std::min(z_interface[k + 1], cloud_top_height);
        if (upper > lower)
            integrated_radiative_tendency +=
                radiative_theta_tendency[k] * (upper - lower);
    }
    const double cooling_flux = std::max(-integrated_radiative_tendency, 0.0);
    if (cooling_flux == 0.0)
        return result;

    const double theta_reference = theta[0];
    if (!std::isfinite(theta_reference) || theta_reference <= 0.0)
        throw std::invalid_argument(
            "cloud-top radiative mixing requires positive finite potential temperature");

    // Lock et al. (2000), DOI 10.1175/1520-0493(2000)128<3187:ANBLMS>2.0.CO;2:
    // V_rad^3 is the depth-weighted cloud-top kinematic cooling flux.
    const double velocity_scale = std::cbrt(
        boundary_layer_constants::g * cloud_top_height
        * cooling_flux / theta_reference);
    for (std::size_t face = 1; face < nz; ++face)
    {
        const double z = z_interface[face];
        if (z >= cloud_top_height)
            break;
        const double eta = z / cloud_top_height;
        const double diffusivity = 0.85 * boundary_layer_constants::kappa
            * velocity_scale * z * eta * std::sqrt(1.0 - eta);
        result.diffusivity_m2s[face] = std::clamp(
            diffusivity, 0.0, maximum_diffusivity_m2s);
    }
    result.active = true;
    result.cloud_top_height_m = cloud_top_height;
    result.cooling_flux_kms = cooling_flux;
    result.velocity_scale_ms = velocity_scale;
    return result;
}

} // namespace tmv::boundary_layer::cloud_top
