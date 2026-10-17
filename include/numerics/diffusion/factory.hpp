#pragma once

// Public diffusion factory surface.

#include "numerics/diffusion/scheme_descriptor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tmv
{
class DiffusionSchemeBase;

std::unique_ptr<DiffusionSchemeBase> create_diffusion_scheme(const std::string& scheme_name);
std::vector<std::string> get_available_diffusion_schemes();
} // namespace tmv

using tmv::create_diffusion_scheme;
using tmv::get_available_diffusion_schemes;
using tmv::get_diffusion_scheme_descriptors;
