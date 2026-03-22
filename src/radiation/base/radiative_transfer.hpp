/**
 * @file radiative_transfer.hpp
 * @brief Declarations for the radiation module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the radiation runtime and scheme implementations.
 * This file is part of the src/radiation subsystem.
 */

#pragma once
#include <vector>
#include "physics/radiation_base.hpp"

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

/**
 * @brief Computes the grey shortwave beer-lambert solution.
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
    const std::vector<double>& p,
    double tau_ref,
    double n
);

/**
 * @brief Computes the layer optical depths.
 */
std::vector<double> compute_layer_optical_depths(
    const std::vector<double>& tau_profile,
    const std::vector<double>& dz
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

}
