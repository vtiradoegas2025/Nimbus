#pragma once

// Public advection factory surface.

#include "numerics/advection/scheme_descriptor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tmv
{
class AdvectionSchemeBase;

std::unique_ptr<AdvectionSchemeBase> create_advection_scheme(const std::string& scheme_name);
std::vector<std::string> get_available_advection_schemes();
} // namespace tmv

using tmv::create_advection_scheme;
using tmv::get_available_advection_schemes;
using tmv::get_advection_scheme_descriptors;
