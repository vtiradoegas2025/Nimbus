/**
 * @file velocity.hpp
 * @brief Declarations for the radar module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the radar runtime and scheme implementations.
 * This file is part of the src/radar subsystem.
 */

#pragma once

#include "diagnostics/radar_base.hpp"

/**
 * @brief Doppler radial velocity radar scheme
 *
 * Computes radial velocity V_r = V · ê_r following AMS standards.
 *
 * Optional: scatterer motion correction using hydrometeor fall speeds.
 */

class VelocityScheme : public RadarSchemeBase 
{
private:
    int NR_, NTH_, NZ_;

    double radar_x_, radar_y_, radar_z_;

public:
    /**
     * @brief Initializes velocity operator for active grid and radar geometry.
     */
    void initialize(const RadarConfig& config, int NR, int NTH, int NZ) override;

    /**
     * @brief Computes radial velocity outputs for current radar state.
     */
    void compute(const RadarConfig& config, const RadarStateView& state, RadarOut& out) override;

    std::string name() const override { return "velocity"; }

private:
    /**
     * @brief Compute radial velocity at each grid point
     */
    void compute_radial_velocity(const RadarConfig& config, const RadarStateView& state, RadarOut& out);

    /**
     * @brief Optional scatterer motion correction
     *
     * V_r_scatterer = V_r_air - V_terminal · ê_r
     */
    void apply_scatterer_correction(const RadarConfig& config, const RadarStateView& state, RadarOut& out);

    /**
     * @brief Estimate terminal fall speed for hydrometeors (bulk approximation)
     *
     * Uses species-dependent bulk relations and clamps to physically plausible ranges.
     */
    float estimate_fall_speed(float q_rain, float q_snow, float q_graupel, float q_hail, float q_ice);
};
