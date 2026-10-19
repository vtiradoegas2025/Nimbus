#pragma once

#include "boundary_conditions/factory.hpp"
#include "boundary_conditions/runtime.hpp"

// Compatibility umbrella and legacy test helper. New code should include the
// factory or runtime header that owns the contract it uses.
inline void apply_cartesian_boundary_conditions()
{
    static auto scheme = tmv::create_cartesian_bc_scheme();
    scheme->apply_full();
}
