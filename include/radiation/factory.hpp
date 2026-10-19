#pragma once

// Public radiation factory and capability-query surface.

#include "radiation/scheme_descriptor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tmv
{
class RadiationSchemeBase;

std::unique_ptr<RadiationSchemeBase>
create_radiation_scheme(const std::string& scheme_name);
std::vector<std::string> get_available_radiation_schemes();
std::vector<std::string> get_known_radiation_schemes();
std::string canonicalize_radiation_scheme_id(const std::string& scheme_name);
bool is_radiation_scheme_implemented(const std::string& scheme_name);
} // namespace tmv

using tmv::canonicalize_radiation_scheme_id;
using tmv::create_radiation_scheme;
using tmv::get_available_radiation_schemes;
using tmv::get_known_radiation_schemes;
using tmv::is_radiation_scheme_implemented;
using tmv::get_radiation_scheme_descriptors;
