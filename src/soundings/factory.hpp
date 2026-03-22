/**
 * @file factory.hpp
 * @brief Declarations for the soundings module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the soundings runtime and scheme implementations.
 * This file is part of the src/soundings subsystem.
 */

#pragma once
#include "data/soundings_base.hpp"
#include <memory>
#include <string>

/**
 * @brief Factory function declarations for sounding schemes
 * This header declares factory functions for different sounding implementations
 */

/**
 * @brief Create SHARPY sounding scheme
 * @return Unique pointer to SHARPY sounding scheme
 */
std::unique_ptr<SoundingScheme> create_sharpy_sounding_scheme();

/**
 * @brief Create sounding scheme based on scheme ID
 * @param scheme_id Identifier for the desired scheme ("sharpy", "none", etc.)
 * @return Unique pointer to sounding scheme instance
 * @throws std::runtime_error if scheme_id is not recognized
 */
std::unique_ptr<SoundingScheme> create_sounding_scheme(const std::string& scheme_id);
