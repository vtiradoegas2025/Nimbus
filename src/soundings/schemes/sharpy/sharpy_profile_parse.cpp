#include "sharpy_profile_parse.hpp"
#include "soundings/sounding_input_limits.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace tmv::soundings::sharpy
{
/**
 * @brief Parses a TMV_SHARPY_PROFILE_V1 text file.
 *
 * The format is what `sharpy_extract.py` produces: a magic header line,
 * tab-separated metadata key/value lines (station_id, timestamp_utc,
 * latitude_deg, longitude_deg, elevation_m, levels) followed by `levels`
 * data rows of (z, p_hpa, T_k, Td_k, wspd_ms, wdir_deg). Comment lines
 * starting with '#' and blank lines are tolerated.
 *
 * Used as the canonical fixture format for end-to-end integration tests.
 */
SoundingData read_sharpy_text_file(const std::string& file_path)
{
    if (!std::filesystem::is_regular_file(file_path))
    {
        throw std::runtime_error("TMV sounding text file is not a regular file: " + file_path);
    }
    if (std::filesystem::file_size(file_path) > kMaxTextSoundingFileBytes)
    {
        throw std::runtime_error("TMV sounding text file exceeds the 8 MiB safety limit");
    }
    std::ifstream in(file_path);
    if (!in.is_open())
    {
        throw std::runtime_error("Failed to open TMV sounding text file: " + file_path);
    }

    auto trim = [](std::string s) {
        const auto first = s.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return std::string{};
        const auto last = s.find_last_not_of(" \t\r\n");
        return s.substr(first, last - first + 1);
    };

    std::size_t consumed_bytes = 0;
    const auto bounded_line = [&in, &consumed_bytes](std::string& line)
    {
        std::array<char, kMaxTextSoundingLineBytes + 2> buffer{};
        const bool extracted = static_cast<bool>(
            in.getline(buffer.data(), static_cast<std::streamsize>(buffer.size())));
        const auto read_bytes = static_cast<std::size_t>(in.gcount());
        if (read_bytes > kMaxTextSoundingFileBytes ||
            consumed_bytes > kMaxTextSoundingFileBytes - read_bytes)
        {
            throw std::runtime_error("TMV sounding text file exceeds the 8 MiB safety limit");
        }
        consumed_bytes += read_bytes;
        if (!extracted)
        {
            if (in.fail() && !in.eof())
            {
                throw std::runtime_error("TMV sounding text exceeds the 4096-byte line safety limit");
            }
            if (in.bad())
            {
                throw std::runtime_error("failed to read TMV sounding text file");
            }
            return false;
        }
        line.assign(buffer.data());
        if (line.size() > kMaxTextSoundingLineBytes)
        {
            throw std::runtime_error("TMV sounding text exceeds the 4096-byte line safety limit");
        }
        return true;
    };

    std::string line;
    if (!bounded_line(line) || trim(line) != "TMV_SHARPY_PROFILE_V1")
    {
        throw std::runtime_error("File is not TMV_SHARPY_PROFILE_V1: " + file_path);
    }

    SoundingData data;
    std::size_t expected_levels = 0;
    bool levels_seen = false;
    std::vector<std::vector<double>> rows;

    auto split_tab = [](const std::string& s) {
        std::vector<std::string> parts;
        std::string token;
        for (char c : s)
        {
            if (c == '\t')
            {
                parts.push_back(token);
                token.clear();
            }
            else
            {
                token.push_back(c);
            }
        }
        parts.push_back(token);
        return parts;
    };

    while (bounded_line(line))
    {
        const std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed.front() == '#')
        {
            continue;
        }
        const auto parts = split_tab(line);
        if (parts.size() < 2)
        {
            throw std::runtime_error(
                "Malformed TMV sounding line: '" + line + "' in " + file_path);
        }
        const std::string& key = parts[0];
        const std::string& value = parts[1];

        if (key == "station_id")        { data.station_id = value; }
        else if (key == "timestamp_utc"){ data.timestamp_utc = value; }
        else if (key == "latitude_deg") { data.latitude_deg = std::stod(value); }
        else if (key == "longitude_deg"){ data.longitude_deg = std::stod(value); }
        else if (key == "elevation_m")  { data.elevation_m = std::stod(value); }
        else if (key == "levels")
        {
            if (levels_seen)
            {
                throw std::runtime_error("TMV sounding text has a duplicate levels header");
            }
            const std::string count_text = trim(value);
            const auto [end, error] = std::from_chars(
                count_text.data(), count_text.data() + count_text.size(), expected_levels);
            if (error != std::errc{} || end != count_text.data() + count_text.size() ||
                expected_levels < 2)
            {
                throw std::runtime_error("TMV sounding text has an invalid levels header");
            }
            if (expected_levels > kMaxTextSoundingLevels)
            {
                throw std::runtime_error("TMV sounding text exceeds the 100,000-level safety limit");
            }
            levels_seen = true;
            rows.reserve(expected_levels);
            // Skip the column-header row "h\tp\tt\ttd\twspd\twdir".
            if (bounded_line(line))
            {
                const std::string header = trim(line);
                const bool is_header = (!header.empty()
                    && (header.find("\th") != std::string::npos
                        || header.front() == 'h' || header.front() == '#'));
                if (!is_header)
                {
                    // Treat the line as a data row instead.
                    const auto vals = split_tab(line);
                    if (vals.size() >= 3)
                    {
                        std::vector<double> row;
                        row.reserve(vals.size());
                        for (const auto& v : vals)
                        {
                            row.push_back(v.empty() ? std::numeric_limits<double>::quiet_NaN()
                                                    : std::stod(v));
                        }
                        if (rows.size() >= expected_levels)
                        {
                            throw std::runtime_error("TMV sounding text exceeds declared levels");
                        }
                        rows.push_back(std::move(row));
                    }
                }
            }
        }
        else
        {
            // Unknown keys before the data table are ignored. Once levels
            // is seen, any further line should be a data row, but we
            // already advanced past the header above; the rest is handled
            // below.
            if (!levels_seen)
            {
                continue;
            }
            std::vector<double> row;
            row.reserve(parts.size());
            for (const auto& v : parts)
            {
                row.push_back(v.empty() ? std::numeric_limits<double>::quiet_NaN()
                                        : std::stod(v));
            }
            if (rows.size() >= expected_levels)
            {
                throw std::runtime_error("TMV sounding text exceeds declared levels");
            }
            rows.push_back(std::move(row));
        }
    }

    if (!levels_seen)
    {
        throw std::runtime_error(
            "TMV sounding text file lacks 'levels' header: " + file_path);
    }
    if (rows.size() != expected_levels)
    {
        throw std::runtime_error(
            "TMV sounding row count " + std::to_string(rows.size())
            + " does not match declared levels " + std::to_string(expected_levels)
            + " in " + file_path);
    }

    SoundingData parsed = parse_sharpy_profile_rows(rows);
    parsed.station_id = data.station_id;
    parsed.timestamp_utc = data.timestamp_utc;
    parsed.latitude_deg = data.latitude_deg;
    parsed.longitude_deg = data.longitude_deg;
    parsed.elevation_m = data.elevation_m;
    return parsed;
}


/**
 * @brief Parses the SHARPY profile data.
 */
SoundingData parse_sharpy_profile_rows(const std::vector<std::vector<double>>& profile_data) 
{
    if (profile_data.size() < 2)
    {
        throw std::runtime_error("SHARPY profile parse requires at least 2 levels");
    }

    SoundingData data;
    data.height_m.reserve(profile_data.size());
    data.pressure_hpa.reserve(profile_data.size());
    data.temperature_k.reserve(profile_data.size());
    std::vector<double> dewpoint_values;
    std::vector<double> wind_speed_values;
    std::vector<double> wind_dir_values;
    dewpoint_values.reserve(profile_data.size());
    wind_speed_values.reserve(profile_data.size());
    wind_dir_values.reserve(profile_data.size());

    bool dewpoint_all_finite = true;
    bool wind_speed_all_finite = true;
    bool wind_dir_all_finite = true;

    for (const auto& row : profile_data)
    {
        if (row.size() < 3)
        {
            throw std::runtime_error("SHARPY profile row must contain at least height/pressure/temperature");
        }
        const double h = row[0];
        const double p = row[1];
        const double t = row[2];
        if (!std::isfinite(h) || !std::isfinite(p) || !std::isfinite(t))
        {
            throw std::runtime_error("SHARPY profile has non-finite required values");
        }

        data.height_m.push_back(h);
        data.pressure_hpa.push_back(p);
        data.temperature_k.push_back(t);

        const double td = (row.size() > 3) ? row[3] : std::numeric_limits<double>::quiet_NaN();
        const double ws = (row.size() > 4) ? row[4] : std::numeric_limits<double>::quiet_NaN();
        const double wd = (row.size() > 5) ? row[5] : std::numeric_limits<double>::quiet_NaN();
        dewpoint_values.push_back(td);
        wind_speed_values.push_back(ws);
        wind_dir_values.push_back(wd);
        if (!std::isfinite(td))
        {
            dewpoint_all_finite = false;
        }
        if (!std::isfinite(ws))
        {
            wind_speed_all_finite = false;
        }
        if (!std::isfinite(wd))
        {
            wind_dir_all_finite = false;
        }
    }

    std::vector<std::size_t> order(data.height_m.size());
    for (std::size_t i = 0; i < order.size(); ++i)
    {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return data.height_m[a] < data.height_m[b];
    });

    auto reorder = [&](std::vector<double>& values)
    {
        std::vector<double> sorted(values.size());
        for (std::size_t i = 0; i < order.size(); ++i)
        {
            sorted[i] = values[order[i]];
        }
        values.swap(sorted);
    };

    reorder(data.height_m);
    reorder(data.pressure_hpa);
    reorder(data.temperature_k);
    reorder(dewpoint_values);
    reorder(wind_speed_values);
    reorder(wind_dir_values);

    if (dewpoint_all_finite)
    {
        data.dewpoint_k = std::move(dewpoint_values);
    }
    if (wind_speed_all_finite && wind_dir_all_finite)
    {
        data.wind_speed_ms = std::move(wind_speed_values);
        data.wind_direction_deg = std::move(wind_dir_values);
    }

    return data;
}

}  // namespace tmv::soundings::sharpy
