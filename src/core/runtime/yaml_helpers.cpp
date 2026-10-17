#include "core/runtime/yaml_helpers.hpp"

#include "util/log.hpp"
#include "util/string_utils.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

std::string to_lower_copy(std::string value)
{
    return tmv::strutil::lower_copy(value);
}

std::string strip_wrapping_quotes(std::string value)
{
    if (value.size() >= 2)
    {
        const char first = value.front();
        const char last = value.back();
        if ((first == '"' && last == '"') || (first == '\'' && last == '\''))
        {
            return value.substr(1, value.size() - 2);
        }
    }
    return value;
}

bool parse_bool_value(const std::string& value)
{
    bool parsed = false;
    if (!try_parse_bool_value(value, parsed))
    {
        throw std::invalid_argument(
            "cannot parse '" + value + "' as boolean; expected one of: "
            "true, false, yes, no, on, off, 1, 0");
    }
    return parsed;
}

bool try_parse_bool_value(const std::string& value, bool& out)
{
    const std::string normalized =
        to_lower_copy(strip_wrapping_quotes(value));
    if (normalized == "1" || normalized == "true" ||
        normalized == "yes" || normalized == "on")
    {
        out = true;
        return true;
    }
    if (normalized == "0" || normalized == "false" ||
        normalized == "no" || normalized == "off")
    {
        out = false;
        return true;
    }
    return false;
}

bool try_parse_int_value(const std::string& value, int& out)
{
    try
    {
        size_t consumed = 0;
        const long long parsed = std::stoll(value, &consumed);
        if (consumed != value.size() ||
            parsed < static_cast<long long>(std::numeric_limits<int>::min()) ||
            parsed > static_cast<long long>(std::numeric_limits<int>::max()))
        {
            return false;
        }
        out = static_cast<int>(parsed);
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool try_parse_non_negative_int_value(const std::string& value, int& out)
{
    int parsed = 0;
    if (!try_parse_int_value(value, parsed))
    {
        return false;
    }
    if (parsed < 0)
    {
        return false;
    }
    out = parsed;
    return true;
}

bool try_parse_positive_int_value(const std::string& value, int& out)
{
    int parsed = 0;
    if (!try_parse_int_value(value, parsed))
    {
        return false;
    }
    if (parsed <= 0)
    {
        return false;
    }
    out = parsed;
    return true;
}

bool try_parse_uint64_value(const std::string& value, std::uint64_t& out)
{
    try
    {
        size_t consumed = 0;
        const unsigned long long parsed = std::stoull(value, &consumed);
        if (consumed != value.size())
        {
            return false;
        }
        out = static_cast<std::uint64_t>(parsed);
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool try_parse_double_value(const std::string& value, double& out)
{
    try
    {
        size_t consumed = 0;
        const double parsed = std::stod(value, &consumed);
        if (consumed != value.size() || !std::isfinite(parsed))
        {
            return false;
        }
        out = parsed;
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

std::vector<std::string> parse_string_list(const std::string& value)
{
    std::string cleaned = value;
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '['), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), ']'), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '"'), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '\''), cleaned.end());

    std::vector<std::string> out;
    std::stringstream ss(cleaned);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), item.end());
        if (!item.empty())
        {
            out.push_back(item);
        }
    }
    return out;
}

namespace
{
std::unordered_map<std::string, YamlField> parse_yaml_stream_with_lines(std::istream& file)
{
    std::unordered_map<std::string, YamlField> config;
    std::string line;
    std::vector<std::string> section_stack;
    int line_number = 0;

    while (std::getline(file, line))
    {
        ++line_number;

        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos)
        {
            line = line.substr(0, comment_pos);
        }

        size_t indent = 0;
        while (indent < line.size() && line[indent] == ' ') indent++;

        size_t indent_level = indent / 2;

        std::string trimmed_line = line;
        trimmed_line.erase(trimmed_line.begin(), std::find_if(trimmed_line.begin(), trimmed_line.end(), [](unsigned char ch)
        {
            return !std::isspace(ch);
        }));
        trimmed_line.erase(std::find_if(trimmed_line.rbegin(), trimmed_line.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), trimmed_line.end());

        if (trimmed_line.empty()) continue;

        line = trimmed_line;

        if (line.back() == ':')
        {
            std::string section_name = line.substr(0, line.size() - 1);

            while (section_stack.size() > indent_level)
            {
                section_stack.pop_back();
            }

            if (section_stack.size() == indent_level)
            {
                section_stack.push_back(section_name);
            }
            else
            {
                section_stack[indent_level] = section_name;
            }

            continue;
        }

        size_t colon_pos = line.find(':');

        if (colon_pos != std::string::npos)
        {
            while (section_stack.size() > indent_level)
            {
                section_stack.pop_back();
            }

            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);

            key.erase(key.begin(), std::find_if(key.begin(), key.end(), [](unsigned char ch)
            {
                return !std::isspace(ch);
            }));

            key.erase(std::find_if(key.rbegin(), key.rend(), [](unsigned char ch)
            {
                return !std::isspace(ch);
            }).base(), key.end());

            value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch)
            {
                return !std::isspace(ch);
            }));

            value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch)
            {
                return !std::isspace(ch);
            }).base(), value.end());
            value = strip_wrapping_quotes(value);

            std::string full_key;
            for (const auto& section : section_stack)
            {
                if (!full_key.empty()) full_key += ".";
                full_key += section;
            }
            if (!full_key.empty()) full_key += ".";
            full_key += key;
            config[full_key] = YamlField{value, line_number};
        }
    }

    return config;
}
}

std::optional<std::unordered_map<std::string, YamlField>>
parse_yaml_with_lines(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        return std::nullopt;
    }
    return parse_yaml_stream_with_lines(file);
}

std::unordered_map<std::string, YamlField> parse_yaml_text_with_lines(std::string_view text)
{
    std::istringstream file{std::string(text)};
    return parse_yaml_stream_with_lines(file);
}

std::unordered_map<std::string, std::string> parse_yaml_simple(const std::string& filename)
{
    std::unordered_map<std::string, std::string> config;
    auto with_lines = parse_yaml_with_lines(filename);
    if (!with_lines.has_value())
    {
        tmv::log_error("Could not open config file: ", filename);
        return config;
    }
    config.reserve(with_lines->size());
    for (const auto& [key, field] : *with_lines)
    {
        config.emplace(key, field.value);
    }
    return config;
}
