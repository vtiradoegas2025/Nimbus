#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

/**
 * @file log.hpp
 * @brief Lightweight logging wrapper for the simulation runtime.
 *
 * Provides leveled logging (info, warn, error, debug) gated by the global
 * LogProfile. New code should prefer these functions over raw std::cerr/cout.
 *
 * Existing raw cerr/cout calls can be migrated incrementally — this header
 * does not force any changes on existing code.
 *
 * Usage:
 *   tmv::log_info("Simulation initialized");
 *   tmv::log_warn("CFL exceeded at step ", step);
 *   tmv::log_error("Failed to open config: ", path);
 *   tmv::log_debug("dt=", dt, " NR=", NR);
 */

// Import log-level gating functions from simulation.hpp.
// simulation.hpp defines LogProfile, global_log_profile, and the
// log_normal_enabled() / log_debug_enabled() inline functions.
#include "core/simulation.hpp"

namespace tmv
{

namespace detail
{

/**
 * @brief Variadic stream helper — writes all arguments into an ostringstream.
 */
template <typename... Args>
std::string compose(Args&&... args)
{
    std::ostringstream oss;
    (oss << ... << std::forward<Args>(args));
    return oss.str();
}

} // namespace detail

/**
 * @brief Log an informational message (visible at LogProfile::normal and above).
 */
template <typename... Args>
inline void log_info(Args&&... args)
{
    if (log_normal_enabled())
    {
        std::cout << "[INFO]  " << detail::compose(std::forward<Args>(args)...) << "\n";
    }
}

/**
 * @brief Log a warning (visible at LogProfile::normal and above, written to stderr).
 */
template <typename... Args>
inline void log_warn(Args&&... args)
{
    if (log_normal_enabled())
    {
        std::cerr << "[WARN]  " << detail::compose(std::forward<Args>(args)...) << "\n";
    }
}

/**
 * @brief Log an error (always visible, written to stderr).
 */
template <typename... Args>
inline void log_error(Args&&... args)
{
    std::cerr << "[ERROR] " << detail::compose(std::forward<Args>(args)...) << "\n";
}

/**
 * @brief Log a debug message (visible only at LogProfile::debug).
 */
template <typename... Args>
inline void log_debug(Args&&... args)
{
    if (log_debug_enabled())
    {
        std::cout << "[DEBUG] " << detail::compose(std::forward<Args>(args)...) << "\n";
    }
}

} // namespace tmv
