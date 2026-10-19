#pragma once

// Public factory surface; concrete radar operators remain private.

#include "radar/scheme_descriptor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tmv
{

class RadarSchemeBase;

std::unique_ptr<RadarSchemeBase>
create_radar_scheme(const std::string& scheme_name);

std::vector<std::string> get_available_radar_schemes();
} // namespace tmv

using tmv::create_radar_scheme;
using tmv::get_available_radar_schemes;
using tmv::get_radar_scheme_descriptors;
