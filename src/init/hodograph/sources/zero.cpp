/**
 * @file zero.cpp
 * @brief ZeroHodograph implementation.
 */

#include "init/hodograph/zero.hpp"

namespace tmv::init
{

WindColumn ZeroHodograph::build(const std::vector<double>& z_m) const
{
    WindColumn out;
    out.u_ms.assign(z_m.size(), 0.0);
    out.v_ms.assign(z_m.size(), 0.0);
    return out;
}

std::string ZeroHodograph::describe() const
{
    return "zero";
}

}  // namespace tmv::init
