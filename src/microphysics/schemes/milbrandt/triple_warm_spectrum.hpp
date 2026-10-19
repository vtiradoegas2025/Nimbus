#pragma once

#include "triple_warm_phase.hpp"

#include <array>
#include <string>

namespace tmv::milbrandt_detail
{

class TripleWarmRainSpectrum
{
public:
    const ExactRainTripleMomentState& source() const noexcept
    {
        return source_;
    }

    double alpha() const noexcept
    {
        return alpha_;
    }

    double lambda_per_m() const noexcept
    {
        return lambda_per_m_;
    }

    double mean_mass_diameter_m() const noexcept
    {
        return mean_mass_diameter_m_;
    }

    double alpha_zero_equivalent_inverse_slope_m() const noexcept
    {
        return alpha_zero_equivalent_inverse_slope_m_;
    }

    double diameter_moment_1_m() const noexcept
    {
        return moments_[0];
    }

    double diameter_moment_3_m3() const noexcept
    {
        return moments_[1];
    }

    double diameter_moment_6_m6() const noexcept
    {
        return moments_[2];
    }

    double ventilation_diameter_moment() const noexcept
    {
        return moments_[3];
    }

    double ventilation_diameter_order() const noexcept
    {
        return 1.75;
    }

    bool normalized_diameter_moment(double order, double& result,
                                    std::string& error) const;

private:
    friend bool diagnose_triple_warm_rain_spectrum(
        const ExactRainTripleMomentState&, TripleWarmRainSpectrum&,
        std::string&);

    ExactRainTripleMomentState source_{};
    double alpha_ = 0.0;
    double lambda_per_m_ = 0.0;
    double mean_mass_diameter_m_ = 0.0;
    double alpha_zero_equivalent_inverse_slope_m_ = 0.0;
    std::array<double, 4> moments_{};
    bool valid_ = false;
};

bool diagnose_triple_warm_rain_spectrum(
    const ExactRainTripleMomentState& source,
    TripleWarmRainSpectrum& result,
    std::string& error);

} // namespace tmv::milbrandt_detail
