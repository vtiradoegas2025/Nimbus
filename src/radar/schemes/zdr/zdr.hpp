#pragma once

#include "radar/radar_base.hpp"
#include <algorithm>
#include <cmath>

namespace tmv
{

// Conventional Z_DR from a simplified bulk forward operator. Its shape and
// reflectivity relations are not a DSD-qualified polarimetric retrieval.
class ZDRScheme : public RadarSchemeBase 
{
private:
    int NR_, NTH_, NZ_;
    RadarOut candidate_output_;

public:
    std::unique_ptr<RadarSchemeBase> clone() const override
    {
        auto copy = std::make_unique<ZDRScheme>(*this);
        copy->candidate_output_ = {};
        return copy;
    }

    void initialize(const RadarConfig& config, int NR, int NTH, int NZ) override;

    void compute(const RadarConfig& config, const RadarStateView& state, RadarOut& out) override;

    std::string name() const override { return "zdr"; }
    bool supports_domain_recycling_invalidation() const noexcept override
    {
        return true;
    }
    void invalidate_generation_state_after_domain_recycling() noexcept override;

private:
    // Rain-only parameterized path.
    void compute_simple_zdr(const RadarConfig& config, const RadarStateView& state, RadarOut& out);

    // Bulk polarimetric forward-operator path.
    void compute_polarimetric_fo(const RadarConfig& config, const RadarStateView& state, RadarOut& out);

    void compute_zhang2021_rain_mp(const RadarStateView& state, RadarOut& out);

    // Rain Z_H/Z_V from mass concentration and oblate-drop axis ratio.
    void compute_rain_polarization(float rain_mass_concentration,
                                   float axis_ratio,
                                   float& Z_H,
                                   float& Z_V);

    // Simplified axis ratio; it omits DSD and drop-oscillation dependence.
    float estimate_rain_axis_ratio(float q_rain, float rho_dry);

    // Polarimetric rain contribution using oblate drops.
    void compute_polarimetric_rain(const Field3D& q_rain,
                                   const Field3D* rho_dry,
                                   RadarOut& out);

    // Simplified snow, graupel, hail, or cloud-ice contribution.
    void compute_polarimetric_ice(const Field3D& q_ice,
                                  const Field3D* rho_dry,
                                  const std::string& species,
                                  RadarOut& out);

    // Guarded dBZ to linear-reflectivity conversion.
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

} // namespace tmv

using ZDRScheme = tmv::ZDRScheme;
