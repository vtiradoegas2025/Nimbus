/**
 * @file simple_grey.hpp
 * @brief Declarations for the radiation module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the radiation runtime and scheme implementations.
 * This file is part of the src/radiation subsystem.
 */

#pragma once
#include "radiation/base/radiative_transfer.hpp"
#include "physics/radiation_base.hpp"

/**
 * @brief Implements the simple grey radiation scheme.
 */ 
class SimpleGreyScheme : public RadiationSchemeBase 
{
private:
    double tau_lw_ref_;
    double tau_sw_ref_;
    double n_lw_;
    double n_sw_;

    std::vector<double> tau_lw_profile_;
    std::vector<double> tau_sw_profile_;

public:
    /**
     * @brief Constructs the simple-grey radiation scheme.
     */
    SimpleGreyScheme();

    std::string name() const override { return "simple_grey"; }
    /**
     * @brief Returns required state-field mask for this scheme.
     */
    int required_fields() const override;

    /**
 * @brief Initializes the simple grey radiation scheme.
 */
    void initialize(const RadiationConfig& cfg) override;

    /**
 * @brief Computes the column of the simple grey radiation scheme.
 */
    void compute_column(
        const RadiationConfig& cfg,
        const RadiationColumnStateView& col,
        RadiationColumnTendencies& tend,
        RadiationColumnFluxes* fluxes_opt = nullptr) override;

private:

    /**
 * @brief Computes the longwave heating.
 */
    void compute_lw_heating(
        const RadiationColumnStateView& col,
        std::vector<double>& dTdt_lw,
        RadiationColumnFluxes* fluxes_opt);

    /**
 * @brief Computes the shortwave heating.
 */
    void compute_sw_heating(
        const RadiationColumnStateView& col,
        std::vector<double>& dTdt_sw,
        RadiationColumnFluxes* fluxes_opt);
};
