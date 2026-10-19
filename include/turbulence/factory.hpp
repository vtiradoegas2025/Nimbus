#pragma once

// Public turbulence factory surface.

#include "turbulence/scheme_descriptor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tmv
{
class TurbulenceSchemeBase;

std::unique_ptr<TurbulenceSchemeBase>
create_turbulence_scheme(const std::string& scheme_name);
std::vector<std::string> get_available_turbulence_schemes();
} // namespace tmv

using tmv::create_turbulence_scheme;
using tmv::get_available_turbulence_schemes;
using tmv::get_turbulence_scheme_descriptors;
