/**
 * @file factory.hpp
 * @brief Declarations for the numerics module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the numerics runtime and scheme implementations.
 * This file is part of the src/numerics subsystem.
 */

#pragma once
#include <memory>
#include <string>
#include "numerics/time_stepping_base.hpp"

class RK3Scheme;
class RK4Scheme;

/**
 * @brief Creates a time-stepping scheme by configured name.
 */
std::unique_ptr<TimeSteppingSchemeBase> create_time_stepping_scheme(const std::string& scheme_name);

/**
 * @brief Returns names of available time-stepping schemes.
 */
std::vector<std::string> get_available_time_stepping_schemes();
