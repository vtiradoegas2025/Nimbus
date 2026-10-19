#pragma once

#include "radar/radar_base.hpp"

namespace tmv
{

// Doppler velocity V_r = V dot e_r, optionally corrected for hydrometeor
// terminal fall speed.
class VelocityScheme : public RadarSchemeBase 
{
private:
    int NR_, NTH_, NZ_;

    double radar_x_, radar_y_, radar_z_;
    RadarOut candidate_output_;

public:
    std::unique_ptr<RadarSchemeBase> clone() const override
    {
        auto copy = std::make_unique<VelocityScheme>(*this);
        copy->candidate_output_ = {};
        return copy;
    }

    void initialize(const RadarConfig& config, int NR, int NTH, int NZ) override;

    void compute(const RadarConfig& config, const RadarStateView& state, RadarOut& out) override;

    std::string name() const override { return "velocity"; }
    bool supports_domain_recycling_invalidation() const noexcept override
    {
        return true;
    }
    void invalidate_generation_state_after_domain_recycling() noexcept override;

private:
    void compute_radial_velocity(const RadarConfig& config, const RadarStateView& state, RadarOut& out);

    // V_r,scatterer = V_r,air - V_terminal dot e_r.
    void apply_scatterer_correction(const RadarConfig& config, const RadarStateView& state, RadarOut& out);

    // Species-weighted bulk fall speed, bounded to the operator's valid range.
    float estimate_fall_speed(float q_rain, float q_snow, float q_graupel, float q_hail, float q_ice);
};

} // namespace tmv

using VelocityScheme = tmv::VelocityScheme;
