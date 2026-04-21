/**
 * @file factory.hpp
 * @brief Declarations for the radar module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the radar runtime and scheme implementations.
 * This file is part of the src/radar subsystem.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "radar/radar_base.hpp"

class ReflectivityScheme;
class VelocityScheme;
class ZDRScheme;

/**
 * @brief Creates a radar scheme by configured name.
 */
std::unique_ptr<RadarSchemeBase> create_radar_scheme(const std::string& scheme_name);

/**
 * @brief Returns names of available radar schemes.
 */
std::vector<std::string> get_available_radar_schemes();
