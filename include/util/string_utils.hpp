#pragma once

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <string_view>

/**
 * @file string_utils.hpp
 * @brief Lightweight string helpers shared by runtime parsing utilities.
 *
 * Provides case normalization, boolean parsing, and JSON escaping.
 * Functions are header-inline because they are small and reused in
 * configuration and diagnostics code paths.
 */

namespace tmv
{
namespace strutil
{

/**
 * @brief Returns a lowercase copy of the input string.
 * @param value Source string view.
 * @return Lowercased string.
 */
inline std::string lower_copy(std::string_view value)
{
    std::string out(value);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c)
    {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

/**
 * @brief Returns a copy with leading and trailing ASCII whitespace removed.
 * @param value Source string view.
 * @return Trimmed string.
 */
inline std::string trim_copy(std::string_view value)
{
    const auto* first = value.data();
    const auto* end = first + value.size();
    while (first != end && std::isspace(static_cast<unsigned char>(*first)))
    {
        ++first;
    }
    while (end != first && std::isspace(static_cast<unsigned char>(*(end - 1))))
    {
        --end;
    }
    return std::string(first, end);
}

/**
 * @brief Returns a trimmed, lowercased copy of the input string.
 *
 * Convenience composition of trim_copy() and lower_copy().
 * This is the standard pre-processing step for scheme name matching
 * in all factory functions.
 *
 * @param value Source string view.
 * @return Trimmed and lowercased string.
 */
inline std::string trim_and_lower(std::string_view value)
{
    return lower_copy(trim_copy(value));
}

/**
 * @brief Parses common truthy boolean spellings.
 * @param value Input string view.
 * @return True for 1/true/yes/on (case-insensitive), false otherwise.
 */
inline bool parse_bool(std::string_view value)
{
    const std::string normalized = lower_copy(value);
    return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

/**
 * @brief Escapes control characters for safe JSON string emission.
 * @param value Input string view.
 * @return Escaped JSON-safe string.
 */
inline std::string json_escape(std::string_view value)
{
    std::ostringstream oss;
    for (char c : value)
    {
        switch (c)
        {
            case '\\': oss << "\\\\"; break;
            case '"': oss << "\\\""; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default: oss << c; break;
        }
    }
    return oss.str();
}

} // namespace strutil
} // namespace tmv
