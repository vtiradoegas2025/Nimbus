#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace tmv::rk
{
template <typename Rhs>
inline double combine(double                       base,
                      double                       current,
                      const Rhs&                   rhs,
                      std::size_t                  stages,
                      const std::array<double, 4>& weights,
                      double                       step,
                      bool                         ssp,
                      double&                      increment_scale)
{
    if (ssp && stages > 1)
    {
        const double fraction  = stages == 2 ? .25 : 2.0 / 3.0;
        const double increment = step * rhs(stages - 1);
        increment_scale        = fraction * (std::abs(current) + std::abs(increment));
        return (1.0 - fraction) * base + fraction * (current + increment);
    }
    double value    = base;
    increment_scale = 0;
    for (std::size_t stage = 0; stage < stages; ++stage)
    {
        const double increment = step * weights[stage] * rhs(stage);
        value += increment;
        increment_scale += std::abs(increment);
    }
    return value;
}
} // namespace tmv::rk
