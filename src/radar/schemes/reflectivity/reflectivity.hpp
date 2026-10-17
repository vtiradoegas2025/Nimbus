#pragma once

#include "radar/radar_base.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace tmv
{

// Equivalent reflectivity under Rayleigh scattering, with fast DA and
// particle-size-distribution moment paths.
class ReflectivityScheme : public RadarSchemeBase {
private:
    static constexpr double K_w_squared = 0.93;
    static constexpr double Z_min_dBZ = -30.0;

    // Density and particle-size-distribution parameters for each species.
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
    RadarOut candidate_output_;

    // Local dry-air density field for the current compute() call (species
    // mass concentration = rho_d * q). Set per call from RadarStateView::rho;
    // validated non-null for every active hydrometeor conversion.
    const Field3D* rho_ = nullptr;

public:
    std::unique_ptr<RadarSchemeBase> clone() const override
    {
        auto copy = std::make_unique<ReflectivityScheme>(*this);
        copy->rho_ = nullptr;
        copy->candidate_output_ = {};
        return copy;
    }

    void initialize(const RadarConfig& config, int NR, int NTH, int NZ) override;

    void compute(const RadarConfig& config, const RadarStateView& state, RadarOut& out) override;

    std::string name() const override { return "reflectivity"; }
    bool supports_domain_recycling_invalidation() const noexcept override
    {
        return true;
    }
    void invalidate_generation_state_after_domain_recycling() noexcept override;

private:
    // Fast data-assimilation operator with single/double-moment support.
    void compute_fast_da(const RadarConfig& config, const RadarStateView& state, RadarOut& out);

    // Moment-aware particle-size-distribution operator.
    void compute_psd_moment(const RadarConfig& config, const RadarStateView& state, RadarOut& out);

    // Accumulates one species into linear equivalent reflectivity.
    void compute_species_reflectivity(
        const Field3D& q,
        const Field3D* Nt,
        const HydrometeorProps& props,
        Field3D& Ze_out);

    // Uses mass and number concentration to evaluate the PSD moment.
    void compute_moment_reflectivity(
        const Field3D& q,
        const Field3D* Nt,
        const HydrometeorProps& props,
        Field3D& Ze_out,
        const std::string& species_name);

    // Guarded linear-reflectivity to dBZ conversion.
    static float linear_to_dbz(float Ze_linear) {
        if (!std::isfinite(static_cast<double>(Ze_linear)) || Ze_linear <= 1e-10f) {
            return Z_min_dBZ;
        }
        const float clamped = std::clamp(Ze_linear, 1.0e-10f, 1.0e12f);
        const float dbz = 10.0f * std::log10(clamped);
        return std::isfinite(static_cast<double>(dbz))
            ? std::clamp(dbz, static_cast<float>(Z_min_dBZ), 120.0f)
            : static_cast<float>(Z_min_dBZ);
    }

    // Guarded dBZ to linear-reflectivity conversion.
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

} // namespace tmv

using ReflectivityScheme = tmv::ReflectivityScheme;
