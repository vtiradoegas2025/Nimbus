#pragma once

#include "distribution.hpp"

namespace tmv::milbrandt_detail
{
struct PairCollision
{
    double first_mass    = 0.0;
    double second_mass   = 0.0;
    double first_number  = 0.0;
    double second_number = 0.0;
};

// Stochastic collection for two generalized-gamma populations under the
// Milbrandt-Yau bulk relative-speed approximation. Mass increments are
// mixing ratios and number increments are dry-air number mixing ratios.
inline PairCollision collide_gamma_populations(double                   density,
                                               double                   dt,
                                               double                   efficiency,
                                               double                   relative_speed,
                                               double                   first_q,
                                               double                   first_n,
                                               const GammaDistribution& first,
                                               double                   first_alpha,
                                               double                   first_mass_coefficient,
                                               double                   first_mass_exponent,
                                               double                   second_q,
                                               double                   second_n,
                                               const GammaDistribution& second,
                                               double                   second_alpha,
                                               double                   second_mass_coefficient,
                                               double                   second_mass_exponent)
{
    PairCollision result;
    if (!(density > 0.0) || !(dt > 0.0) || !(efficiency > 0.0) || !(relative_speed > 0.0) || !(first_q > 0.0) ||
        !(first_n > 0.0) || !(second_q > 0.0) || !(second_n > 0.0) || !(first.intercept > 0.0) ||
        !(second.intercept > 0.0))
    {
        return result;
    }

    constexpr double pi_over_four = 0.785398163397448309616;
    const double     common       = dt * pi_over_four * efficiency * relative_speed / density;
    const auto       first_moment = [&](double exponent) { return distribution_moment(first, first_alpha, exponent); };
    const auto second_moment = [&](double exponent) { return distribution_moment(second, second_alpha, exponent); };

    const double collision_number_volume =
        common * density *
        (first_moment(2.0) * second_moment(0.0) + 2.0 * first_moment(1.0) * second_moment(1.0) +
         first_moment(0.0) * second_moment(2.0));
    result.first_mass  = common * first_mass_coefficient *
                         (first_moment(first_mass_exponent + 2.0) * second_moment(0.0) +
                          2.0 * first_moment(first_mass_exponent + 1.0) * second_moment(1.0) +
                          first_moment(first_mass_exponent) * second_moment(2.0));
    result.second_mass = common * second_mass_coefficient *
                         (second_moment(second_mass_exponent + 2.0) * first_moment(0.0) +
                          2.0 * second_moment(second_mass_exponent + 1.0) * first_moment(1.0) +
                          second_moment(second_mass_exponent) * first_moment(2.0));

    result.first_mass  = std::clamp(result.first_mass, 0.0, first_q);
    result.second_mass = std::clamp(result.second_mass, 0.0, second_q);
    result.first_number =
        std::clamp(std::min(collision_number_volume / density, result.first_mass * first_n / first_q), 0.0, first_n);
    result.second_number = std::clamp(
        std::min(collision_number_volume / density, result.second_mass * second_n / second_q), 0.0, second_n);
    return result;
}

} // namespace tmv::milbrandt_detail
