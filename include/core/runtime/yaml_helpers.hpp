#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

/**
 * @file yaml_helpers.hpp
 * @brief Self-contained YAML parsing and value-conversion utilities.
 *
 * These helpers were extracted from `runtime_config.cpp` so that the v1
 * config loader (and its unit tests) can depend on a small, well-defined
 * module rather than pulling in the entire simulator runtime.
 *
 * The YAML parser is the hand-rolled flat key-value parser the project has
 * always used: indent-based section tracking, dotted-key output, simple
 * comment stripping. It does not support YAML lists, anchors, or block
 * scalars; configs that need those should preprocess them out (or eventually
 * migrate to a real YAML library).
 *
 * `parse_yaml_with_lines` augments the parser with per-key source line
 * tracking so downstream validators can emit line-numbered errors.
 */

/**
 * @brief Returns a lowercase copy of the input.
 */
std::string to_lower_copy(std::string value);

/**
 * @brief Removes matching single or double quotes wrapping a value.
 */
std::string strip_wrapping_quotes(std::string value);

/**
 * @brief Parses a boolean or throws when the value is not a recognized
 *        true/false spelling.
 */
bool parse_bool_value(const std::string& value);

/**
 * @brief Attempts to parse a recognized true/false spelling.
 */
bool try_parse_bool_value(const std::string& value, bool& out);

/**
 * @brief Parses an integer value.
 */
bool try_parse_int_value(const std::string& value, int& out);

/**
 * @brief Parses a non-negative integer value.
 */
bool try_parse_non_negative_int_value(const std::string& value, int& out);

/**
 * @brief Parses a strictly positive integer value.
 */
bool try_parse_positive_int_value(const std::string& value, int& out);

/**
 * @brief Parses an unsigned 64-bit integer value.
 */
bool try_parse_uint64_value(const std::string& value, std::uint64_t& out);

/**
 * @brief Parses a finite floating-point value.
 */
bool try_parse_double_value(const std::string& value, double& out);

/**
 * @brief Parses a comma-separated or YAML bracketed string list.
 */
std::vector<std::string> parse_string_list(const std::string& value);

/**
 * @brief A YAML field value paired with its 1-based source line number.
 */
struct YamlField
{
    std::string value;
    int line_number;
};

/**
 * @brief Parses a simple key-value YAML file.
 *
 * Logs an error and returns an empty map if the file cannot be opened
 * (preserves the historical behavior of the function this code came from).
 *
 * @param filename Input file path.
 * @return Parsed key-value map.
 */
std::unordered_map<std::string, std::string> parse_yaml_simple(const std::string& filename);

/**
 * @brief Parses a simple key-value YAML file, retaining each value's source line.
 *
 * Returns std::nullopt if the file cannot be opened (lets callers
 * distinguish "empty file" from "missing file" — important for the v1
 * loader which has a specific error message for each).
 *
 * @param filename Input file path.
 * @return Map of dotted key path → {value, source line}, or std::nullopt
 *         on file-open failure.
 */
std::optional<std::unordered_map<std::string, YamlField>>
parse_yaml_with_lines(const std::string& filename);

std::unordered_map<std::string, YamlField>
parse_yaml_text_with_lines(std::string_view text);
