#pragma once

#include "init/hodograph/hodograph_source.hpp"

namespace tmv::init
{

/**
 * @brief Zero hodograph: u = v = 0 at every level.
 *
 * Used for hydrostatic / equilibrium tests where any nonzero ambient wind
 * would seed startup transients. Already a recipe in configs/README.md
 * ("Zero the wind") that today is achieved by setting every WK anchor to
 * zero; selecting this source is the cleaner, explicit equivalent.
 */
class ZeroHodograph final : public HodographSource
{
public:
    WindColumn build(const std::vector<double>& z_m) const override;
    std::string describe() const override;
};

}  // namespace tmv::init
