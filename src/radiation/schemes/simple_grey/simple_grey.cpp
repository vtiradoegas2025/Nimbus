/**
 * @file simple_grey.cpp
 * @brief Implementation for the radiation module.
 *
 * Provides executable logic for the radiation runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/radiation subsystem.
 */

#include "simple_grey.hpp"
#include <iostream>
#include <algorithm>


/**
 * @brief Initializes the simple grey radiation scheme.
 */
SimpleGreyScheme::SimpleGreyScheme()
    : tau_lw_ref_(6.0), tau_sw_ref_(0.22), n_lw_(4.0), n_sw_(2.0) 
{}

/**
 * @brief Returns the required fields.
 */
int SimpleGreyScheme::required_fields() const 
{
    return static_cast<int>(RadiationRequirements::TEMPERATURE) |
           static_cast<int>(RadiationRequirements::SOLAR_GEOMETRY);
}

/**
 * @brief Initializes the simple grey radiation scheme.
 */
void SimpleGreyScheme::initialize(const RadiationConfig& cfg)
{
    tau_lw_ref_ = cfg.tau_lw_ref;
    tau_sw_ref_ = cfg.tau_sw_ref;
    n_lw_ = cfg.n_lw;
    n_sw_ = cfg.n_sw;

    std::cout << "Initialized Simple Grey Radiation:" << std::endl;
    std::cout << "  LW optical depth: " << tau_lw_ref_ << " (exponent: " << n_lw_ << ")" << std::endl;
    std::cout << "  SW optical depth: " << tau_sw_ref_ << " (exponent: " << n_sw_ << ")" << std::endl;
}

/**
 * @brief Computes the column of the simple grey radiation scheme.
 */
void SimpleGreyScheme::compute_column(
    const RadiationConfig& cfg,
    const RadiationColumnStateView& col,
    RadiationColumnTendencies& tend,
    RadiationColumnFluxes* fluxes_opt
) 
{
    const size_t nz = col.rho->size();

    tend.dTdt_rad.resize(nz, 0.0);
    tend.dTdt_lw.resize(nz, 0.0);
    tend.dTdt_sw.resize(nz, 0.0);

    if (fluxes_opt) 
    {
        fluxes_opt->Fup_lw.resize(nz + 1, 0.0);
        fluxes_opt->Fdn_lw.resize(nz + 1, 0.0);
        fluxes_opt->Fup_sw.resize(nz + 1, 0.0);
        fluxes_opt->Fdn_sw.resize(nz + 1, 0.0);
        fluxes_opt->Fnet_lw.resize(nz + 1, 0.0);
        fluxes_opt->Fnet_sw.resize(nz + 1, 0.0);
        fluxes_opt->Fnet_total.resize(nz + 1, 0.0);
    }

    if (cfg.do_lw) 
    {
        compute_lw_heating(col, tend.dTdt_lw, fluxes_opt);
    }

    if (cfg.do_sw) 
    {
        compute_sw_heating(col, tend.dTdt_sw, fluxes_opt);
    }

    for (size_t k = 0; k < nz; ++k) 
    {
        tend.dTdt_rad[k] = tend.dTdt_lw[k] + tend.dTdt_sw[k];
    }

    if (fluxes_opt) 
    {
        for (size_t k = 0; k <= nz; ++k) 
        {
            fluxes_opt->Fnet_lw[k] = fluxes_opt->Fup_lw[k] - fluxes_opt->Fdn_lw[k];
            fluxes_opt->Fnet_sw[k] = fluxes_opt->Fup_sw[k] - fluxes_opt->Fdn_sw[k];
            fluxes_opt->Fnet_total[k] = fluxes_opt->Fnet_lw[k] + fluxes_opt->Fnet_sw[k];
        }
    }
}

/**
 * @brief Computes the longwave heating.
 */
void SimpleGreyScheme::compute_lw_heating(
    const RadiationColumnStateView& col,
    std::vector<double>& dTdt_lw,
    RadiationColumnFluxes* fluxes_opt
) 
{
    const size_t nz = col.rho->size();

    /**
 * @brief Computes the optical depth profile.
 */
    auto tau_lw_profile = radiative_transfer::compute_optical_depth_profile(
        *col.p, tau_lw_ref_, n_lw_
    );

    /**
 * @brief Computes the layer optical depths.
 */
    auto tau_layers = radiative_transfer::compute_layer_optical_depths(
        tau_lw_profile, *col.dz
    );

    /**
 * @brief Computes the grey longwave two stream solution.
 */
    std::vector<double> Fup_lw, Fdn_lw;
    radiative_transfer::grey_lw_two_stream(
        *col.T, tau_layers, col.Tsfc, col.emissivity_lw,
        Fup_lw, Fdn_lw
    );

    std::vector<double> Fnet_lw(nz + 1);

    for (size_t k = 0; k <= nz; ++k) 
    {
        Fnet_lw[k] = Fup_lw[k] - Fdn_lw[k];
    }

    
    /**
 * @brief Computes the heating rate from the flux divergence.
 */
    radiative_transfer::heating_rate_from_flux_divergence(
        *col.rho, *col.dz, Fnet_lw, dTdt_lw
    );

    if (fluxes_opt) 
    {
        fluxes_opt->Fup_lw = Fup_lw;
        fluxes_opt->Fdn_lw = Fdn_lw;
    }
}

/**
 * @brief Computes the shortwave heating.
 */
void SimpleGreyScheme::compute_sw_heating(
    const RadiationColumnStateView& col,
    std::vector<double>& dTdt_sw,
    RadiationColumnFluxes* fluxes_opt
) 
{
    const size_t nz = col.rho->size();

    if (col.mu0 <= 0.0) 
    {
        std::fill(dTdt_sw.begin(), dTdt_sw.end(), 0.0);
        return;
    }

    /**
 * @brief Computes the optical depth profile.
 */
    auto tau_sw_profile = radiative_transfer::compute_optical_depth_profile(
        *col.p, tau_sw_ref_, n_sw_
    );

    /**
 * @brief Computes the layer optical depths.
 */
    auto tau_layers = radiative_transfer::compute_layer_optical_depths(
        tau_sw_profile, *col.dz
    );

    /**
 * @brief Computes the grey shortwave beer-lambert solution.
 */
    std::vector<double> Fup_sw, Fdn_sw;
    radiative_transfer::grey_sw_beer_lambert(
        tau_layers, col.mu0, col.S0, col.albedo_sw,
        Fup_sw, Fdn_sw
    );

    /**
 * @brief Computes the net flux.
 */
    std::vector<double> Fnet_sw(nz + 1);

    for (size_t k = 0; k <= nz; ++k) 
    {
        Fnet_sw[k] = Fup_sw[k] - Fdn_sw[k];
    }

    /**
 * @brief Computes the heating rate from the flux divergence.
 */
    radiative_transfer::heating_rate_from_flux_divergence(
        *col.rho, *col.dz, Fnet_sw, dTdt_sw
    );

    if (fluxes_opt)
    {
        fluxes_opt->Fup_sw = Fup_sw;
        fluxes_opt->Fdn_sw = Fdn_sw;
    }
}
