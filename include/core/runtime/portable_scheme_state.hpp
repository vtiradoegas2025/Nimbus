#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/field/field3d.hpp"

// Versioned, pointer-free continuation data owned by one scheme. Field order
// is defined by the concrete scheme's payload version.
struct PortableSchemeState
{
    std::uint32_t version = 0;
    std::vector<Field3D> fields;
    std::vector<std::uint64_t> integers;
    std::vector<double> reals;
};

inline bool empty_portable_scheme_state(const PortableSchemeState& state)
{
    return state.version == 0 && state.fields.empty()
        && state.integers.empty() && state.reals.empty();
}

