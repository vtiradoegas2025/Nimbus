/**
 * @file zdr.hpp
 * @brief Declarations for the radar module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the radar runtime and scheme implementations.
 * This file is part of the src/radar subsystem.
 */

#pragma once

#include "radar/radar_base.hpp"
#include <algorithm>
#include <cmath>

/**
 * @brief Differential reflectivity radar scheme
 *
 * Computes Z_DR = 10*log10(Z_H / Z_V) following AMS standards.
 *
 * Supports both simple parameterized and full polarimetric operators.
 */

class ZDRScheme : public RadarSchemeBase 
{
private:
    int NR_, NTH_, NZ_;

public:
    /**
     * @brief Initializes ZDR operator for active grid dimensions.
     */
    void initialize(const RadarConfig& config, int NR, int NTH, int NZ) override;

    /**
     * @brief Computes differential-reflectivity outputs for current state.
     */
    void compute(const RadarConfig& config, const RadarStateView& state, RadarOut& out) override;

    std::string name() const override { return "zdr"; }

private:
    /**
     * @brief Simple ZDR operator (rain-only, parameterized)
     */
    void compute_simple_zdr(const RadarConfig& config, const RadarStateView& state, RadarOut& out);

    /**
     * @brief Full polarimetric forward operator
     */
    void compute_polarimetric_fo(const RadarConfig& config, const RadarStateView& state, RadarOut& out);

    /**
     * @brief Compute Z_H and Z_V for rain drops (oblate spheroids)
     *
     * @param q_rain Rain mixing ratio
     * @param axis_ratio Minor/major axis ratio (function of drop size)
     * @param Z_H Output horizontal reflectivity
     * @param Z_V Output vertical reflectivity
     */
    void compute_rain_polarization(float q_rain, float axis_ratio, float& Z_H, float& Z_V);

    /**
     * @brief Estimate axis ratio for rain drops (simplified)
     *
     * In reality, this depends on drop size distribution and oscillation.
     * Here we use a simple parameterization.
     */
    float estimate_rain_axis_ratio(float q_rain);

    /**
     * @brief Compute polarimetric reflectivities for rain (oblate spheroids)
     *
     * @param q_rain Rain mixing ratio field
     * @param out Radar output structure to update
     */
    void compute_polarimetric_rain(const Field3D& q_rain, RadarOut& out);

    /**
     * @brief Compute polarimetric reflectivities for ice species (simplified)
     *
     * @param q_ice Ice mixing ratio field
     * @param species Species name ("snow", "graupel", "hail", "ice")
     * @param out Radar output structure to update
     */
    void compute_polarimetric_ice(const Field3D& q_ice,
                                  const std::string& species, RadarOut& out);

    /**
     * @brief Convert dBZ to linear reflectivity
     */
    static float dbz_to_linear(float Z_dbz) 
    {
        if (!std::isfinite(static_cast<double>(Z_dbz)))
        {
            return 0.0f;
        }
        const float clamped_dbz = std::clamp(Z_dbz, -40.0f, 100.0f);
        const float linear = std::pow(10.0f, clamped_dbz / 10.0f);
        return std::isfinite(static_cast<double>(linear)) ? linear : 0.0f;
    }
};
