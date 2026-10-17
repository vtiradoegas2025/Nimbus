#pragma once

#include "numerics/numerics_base.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace tmv::rk
{
inline void validate_state(const std::vector<NumericalState>& fields, const char* scheme)
{
    for (const auto& field : fields)
        for (std::size_t n = 0; n < field.data.size(); ++n)
            if (!std::isfinite(field.data.data()[n]))
                throw std::runtime_error(std::string(scheme) + " state is non-finite for " + field.name);
}
inline void
validate_rhs(const std::vector<NumericalState>& fields, const std::vector<NumericalTendencies>& rhs, const char* scheme)
{
    if (rhs.size() != fields.size())
        throw std::runtime_error(std::string(scheme) + " RHS count mismatch");
    for (std::size_t f = 0; f < fields.size(); ++f)
    {
        const auto& state = fields[f].data;
        const auto& rate  = rhs[f].tendencies;
        if (state.size_r() != rate.size_r() || state.size_th() != rate.size_th() || state.size_z() != rate.size_z())
            throw std::runtime_error(std::string(scheme) + " RHS shape mismatch for " + fields[f].name);
        for (std::size_t n = 0; n < rate.size(); ++n)
            if (!std::isfinite(rate.data()[n]))
                throw std::runtime_error(std::string(scheme) + " RHS is non-finite for " + fields[f].name);
    }
}
} // namespace tmv::rk
