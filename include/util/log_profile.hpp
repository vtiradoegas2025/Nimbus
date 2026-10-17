#pragma once

/**
 * @file log_profile.hpp
 * @brief Narrow logging-level state shared by runtime and lightweight logging.
 *
 * Kept separate from simulation.hpp so util/log.hpp does not pull the entire
 * simulation global-state surface into every translation unit that logs.
 */

enum class LogProfile : int
{
    quiet = 0,
    normal = 1,
    debug = 2
};

extern LogProfile& global_log_profile;

/**
 * @brief Returns whether current logging level includes the target level.
 * @param level Minimum desired logging level.
 * @return True when logging at the requested level is enabled.
 */
inline bool log_at_least(LogProfile level)
{
    return static_cast<int>(global_log_profile) >= static_cast<int>(level);
}

/**
 * @brief Returns whether normal logging output is enabled.
 */
inline bool log_normal_enabled()
{
    return log_at_least(LogProfile::normal);
}

/**
 * @brief Returns whether debug logging output is enabled.
 */
inline bool log_debug_enabled()
{
    return log_at_least(LogProfile::debug);
}
