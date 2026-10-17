#pragma once

// Public time-stepping factory and runtime-contract surface.

#include "numerics/time_stepping/scheme_descriptor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tmv
{
struct TimeSteppingConfig;
class TimeSteppingSchemeBase;

std::unique_ptr<TimeSteppingSchemeBase>
create_time_stepping_scheme(const std::string& scheme_name);
void validate_runtime_time_stepping_contract(
    const TimeSteppingConfig& config,
    const TimeSteppingSchemeBase& scheme);
std::vector<std::string> get_available_time_stepping_schemes();
} // namespace tmv

using tmv::create_time_stepping_scheme;
using tmv::get_available_time_stepping_schemes;
using tmv::get_time_stepping_scheme_descriptors;
using tmv::validate_runtime_time_stepping_contract;
