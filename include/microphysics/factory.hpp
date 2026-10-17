#pragma once

// Public factory surface; concrete schemes remain private under src/.

#include "microphysics/config.hpp"
#include "microphysics/scheme_descriptor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tmv
{

class MicrophysicsScheme;

std::unique_ptr<MicrophysicsScheme>
create_microphysics_scheme(const std::string& scheme_name);

std::unique_ptr<MicrophysicsScheme>
create_microphysics_scheme(const std::string& scheme_name,
                           const MicrophysicsConfig& config);

std::vector<std::string> get_available_microphysics_schemes();

// Temporary source-compatibility spelling for downstream callers.
[[deprecated("use get_available_microphysics_schemes")]]
inline std::vector<std::string> get_available_schemes()
{
    return get_available_microphysics_schemes();
}

} // namespace tmv

using tmv::create_microphysics_scheme;
using tmv::get_available_microphysics_schemes;
using tmv::get_available_schemes;
using tmv::get_microphysics_scheme_descriptors;
