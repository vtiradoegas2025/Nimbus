/**
 * @file reflectivity.hpp
 * @brief Declarations for the radar module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the radar runtime and scheme implementations.
 * This file is part of the src/radar subsystem.
 */

#pragma once

#include "radar/radar_base.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

/**
 * @brief Reflectivity radar scheme
 *
 * Computes equivalent radar reflectivity factor Z_e and Z_dBZ
 * following AMS standards for Rayleigh scattering.
 *
 * Supports both fast DA-style operators and PSD-moment operators.
 */

class ReflectivityScheme : public RadarSchemeBase {
private:
    static constexpr double K_w_squared = 0.93;
    static constexpr double Z_min_dBZ = -30.0;

    /**
     * @brief Hydrometeor PSD/density parameters used by reflectivity operators.
     */
    struct HydrometeorProps {
        double density;
        double alpha;
        double c;
        double d;
    };

    static const HydrometeorProps rain_props;
    static const HydrometeorProps snow_props;
    static const HydrometeorProps graupel_props;
    static const HydrometeorProps hail_props;
    static const HydrometeorProps ice_props;

    int NR_, NTH_, NZ_;

public:
    /**
     * @brief Initializes reflectivity operator for active grid dimensions.
     */
    void initialize(const RadarConfig& config, int NR, int NTH, int NZ) override;

    /**
     * @brief Computes reflectivity outputs for current radar state.
     */
    void compute(const RadarConfig& config, const RadarStateView& state, RadarOut& out) override;

    std::string name() const override { return "reflectivity"; }

private:
    /**
     * @brief Fast DA-style reflectivity operator (single/double moment aware)
     */
    void compute_fast_da(const RadarConfig& config, const RadarStateView& state, RadarOut& out);

    /**
     * @brief PSD-moment reflectivity operator (recommended with moments)
     */
    void compute_psd_moment(const RadarConfig& config, const RadarStateView& state, RadarOut& out);

    /**
     * @brief Compute reflectivity contribution from a single species
     *
     * @param q Mixing ratio field
     * @param Nt Number concentration field (optional, nullptr for single-moment)
     * @param props Hydrometeor properties
     * @param Ze_out Output reflectivity field (linear units)
     */
    void compute_species_reflectivity(
        const Field3D& q,
        const Field3D* Nt,
        const HydrometeorProps& props,
        Field3D& Ze_out);

    /**
     * @brief Compute PSD-moment reflectivity using number concentrations
     *
     * @param q Mixing ratio field
     * @param Nt Number concentration field (required for moment calculations)
     * @param props Hydrometeor properties
     * @param Ze_out Output reflectivity field (linear units)
     * @param species_name Species name for diagnostics
     */
    void compute_moment_reflectivity(
        const Field3D& q,
        const Field3D* Nt,
        const HydrometeorProps& props,
        Field3D& Ze_out,
        const std::string& species_name);

    /**
     * @brief Convert linear reflectivity to dBZ
     */
    static float linear_to_dbz(float Ze_linear) {
        if (!std::isfinite(static_cast<double>(Ze_linear)) || Ze_linear <= 1e-10f) {
            return Z_min_dBZ;
        }
        const float clamped = std::clamp(Ze_linear, 1.0e-10f, 1.0e12f);
        const float dbz = 10.0f * std::log10(clamped);
        return std::isfinite(static_cast<double>(dbz)) ? dbz : Z_min_dBZ;
    }

    /**
     * @brief Convert dBZ to linear reflectivity
     */
    static float dbz_to_linear(float Z_dbz) {
        if (!std::isfinite(static_cast<double>(Z_dbz)))
        {
            return 0.0f;
        }
        const float clamped = std::clamp(Z_dbz, -40.0f, 100.0f);
        const float linear = std::pow(10.0f, clamped / 10.0f);
        return std::isfinite(static_cast<double>(linear)) ? linear : 0.0f;
    }
};
