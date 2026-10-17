#include "cm1_sounding.hpp"

#include "core/infra/physical_constants.hpp"
#include "soundings/schemes/sharpy/sharpy_interpolation.hpp"
#include "soundings/sounding_input_limits.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr std::uintmax_t kMaxFileBytes = tmv::soundings::kMaxTextSoundingFileBytes;
constexpr std::size_t kMaxLevels = tmv::soundings::kMaxTextSoundingLevels;
constexpr double kVirtualPotentialCoefficient = 0.608;

struct NumericRow
{
    std::size_t line = 0;
    std::vector<double> values;
};

std::string strip_comment(std::string line)
{
    const auto bang = line.find('!');
    const auto hash = line.find('#');
    const auto cut = std::min(
        bang == std::string::npos ? line.size() : bang,
        hash == std::string::npos ? line.size() : hash);
    line.resize(cut);
    return line;
}

double parse_number(std::string token, std::size_t line, std::size_t column)
{
    std::replace(token.begin(), token.end(), 'd', 'e');
    std::replace(token.begin(), token.end(), 'D', 'E');
    std::size_t consumed = 0;
    double value = 0.0;
    try
    {
        value = std::stod(token, &consumed);
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(
            "CM1 input_sounding line " + std::to_string(line) + ", column "
            + std::to_string(column) + " is not numeric: '" + token + "'");
    }
    if (consumed != token.size() || !std::isfinite(value))
    {
        throw std::runtime_error(
            "CM1 input_sounding line " + std::to_string(line) + ", column "
            + std::to_string(column) + " is not a finite number: '" + token + "'");
    }
    return value;
}

NumericRow parse_row(const std::string& content,
                     std::size_t line,
                     std::size_t expected_columns)
{
    std::istringstream stream(content);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token)
    {
        tokens.push_back(token);
    }
    if (tokens.size() != expected_columns)
    {
        throw std::runtime_error(
            "CM1 input_sounding line " + std::to_string(line) + " has "
            + std::to_string(tokens.size()) + " columns; expected "
            + std::to_string(expected_columns));
    }

    NumericRow row;
    row.line = line;
    row.values.reserve(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); ++i)
    {
        row.values.push_back(parse_number(tokens[i], line, i + 1));
    }
    return row;
}

std::vector<std::pair<std::size_t, std::string>> substantive_lines(
    const std::string& file_path)
{
    std::ifstream input(file_path);
    if (!input)
    {
        throw std::runtime_error("cannot open CM1 input_sounding: " + file_path);
    }

    std::vector<std::pair<std::size_t, std::string>> rows;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line))
    {
        ++line_number;
        line = strip_comment(std::move(line));
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
        {
            continue;
        }
        const auto last = line.find_last_not_of(" \t\r\n");
        rows.emplace_back(line_number, line.substr(first, last - first + 1));
        if (rows.size() > kMaxLevels + 1)
        {
            throw std::runtime_error(
                "CM1 input_sounding exceeds the 100,000-level safety limit");
        }
    }
    return rows;
}

void require_range(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error("CM1 input_sounding: " + message);
    }
}

} // namespace

void Cm1SoundingScheme::initialize(const SoundingConfig& config)
{
    config_ = config;
    if (config_.file_path.empty())
    {
        throw std::runtime_error("CM1 sounding reader requires a file path");
    }
    if (!std::filesystem::is_regular_file(config_.file_path))
    {
        throw std::runtime_error(
            "CM1 input_sounding is not a regular file: " + config_.file_path);
    }
    const auto bytes = std::filesystem::file_size(config_.file_path);
    if (bytes > kMaxFileBytes)
    {
        throw std::runtime_error(
            "CM1 input_sounding exceeds the 8 MiB safety limit");
    }
    initialized_ = true;
}

SoundingData Cm1SoundingScheme::load_sounding(const std::string& file_path)
{
    if (!initialized_)
    {
        throw std::runtime_error("CM1 sounding reader is not initialized");
    }

    const auto lines = substantive_lines(file_path);
    require_range(!lines.empty(), "contains no numeric rows");
    require_range(lines.size() >= 6,
                  "requires at least five vertical levels after the surface header");

    const NumericRow header = parse_row(lines.front().second, lines.front().first, 3);
    const double surface_pressure_hpa = header.values[0];
    const double surface_theta_k = header.values[1];
    const double surface_qv_gkg = header.values[2];
    require_range(surface_pressure_hpa >= 100.0 && surface_pressure_hpa <= 1100.0,
                  "surface pressure must be in [100, 1100] hPa");
    require_range(surface_theta_k >= 150.0 && surface_theta_k <= 700.0,
                  "surface potential temperature must be in [150, 700] K");
    require_range(surface_qv_gkg >= 0.0 && surface_qv_gkg <= 100.0,
                  "surface water-vapor mixing ratio must be in [0, 100] g/kg");

    SoundingData data;
    const std::size_t levels = lines.size() - 1;
    data.height_m.reserve(levels);
    data.pressure_hpa.reserve(levels);
    data.temperature_k.reserve(levels);
    data.potential_temperature_k.reserve(levels);
    data.mixing_ratio_kgkg.reserve(levels);
    data.wind_u_ms.reserve(levels);
    data.wind_v_ms.reserve(levels);
    data.station_id = "CM1_INPUT_SOUNDING";
    data.potential_temperature_is_primary = true;

    const double kappa = R_d / cp;
    double exner = std::pow(surface_pressure_hpa * 100.0 / p0, kappa);
    double previous_height_m = 0.0;
    double previous_theta_v =
        surface_theta_k * (1.0 + kVirtualPotentialCoefficient * surface_qv_gkg / 1000.0);

    for (std::size_t i = 1; i < lines.size(); ++i)
    {
        const NumericRow row = parse_row(lines[i].second, lines[i].first, 5);
        const double z_m = row.values[0];
        const double theta_k = row.values[1];
        const double qv_gkg = row.values[2];
        const double u_ms = row.values[3];
        const double v_ms = row.values[4];

        require_range(z_m >= 0.0, "height cannot be negative at line "
                      + std::to_string(row.line));
        require_range(i == 1 ? std::abs(z_m) <= 1.0e-9
                             : z_m > data.height_m.back(),
                      i == 1
                          ? "first profile level must be at z=0 m"
                          : "heights must increase strictly at line "
                                + std::to_string(row.line));
        require_range(theta_k >= 150.0 && theta_k <= 700.0,
                      "potential temperature must be in [150, 700] K at line "
                          + std::to_string(row.line));
        require_range(qv_gkg >= 0.0 && qv_gkg <= 100.0,
                      "water-vapor mixing ratio must be in [0, 100] g/kg at line "
                          + std::to_string(row.line));
        require_range(std::abs(u_ms) <= 300.0 && std::abs(v_ms) <= 300.0,
                      "wind component magnitude exceeds 300 m/s at line "
                          + std::to_string(row.line));

        if (i == 1)
        {
            // The duplicated header values can be printed at lower precision
            // than the z=0 row by otherwise valid CM1 case generators.
            require_range(std::abs(theta_k - surface_theta_k) <= 1.0e-3,
                          "surface header theta does not match the z=0 row");
            require_range(std::abs(qv_gkg - surface_qv_gkg) <= 1.0e-3,
                          "surface header qv does not match the z=0 row");
        }

        const double qv_kgkg = qv_gkg / 1000.0;
        const double theta_v =
            theta_k * (1.0 + kVirtualPotentialCoefficient * qv_kgkg);
        const double dz = z_m - previous_height_m;
        exner -= (g / cp) * dz
               * 0.5 * ((1.0 / previous_theta_v) + (1.0 / theta_v));
        require_range(std::isfinite(exner) && exner > 0.0,
                      "hydrostatic Exner pressure became invalid at line "
                          + std::to_string(row.line));

        const double pressure_pa = p0 * std::pow(exner, 1.0 / kappa);
        const double temperature_k = theta_k * exner;
        require_range(std::isfinite(pressure_pa) && pressure_pa > 0.0
                          && std::isfinite(temperature_k) && temperature_k > 0.0,
                      "hydrostatic thermodynamic reconstruction failed at line "
                          + std::to_string(row.line));

        data.height_m.push_back(z_m);
        data.pressure_hpa.push_back(pressure_pa / 100.0);
        data.temperature_k.push_back(temperature_k);
        data.potential_temperature_k.push_back(theta_k);
        data.mixing_ratio_kgkg.push_back(qv_kgkg);
        data.wind_u_ms.push_back(u_ms);
        data.wind_v_ms.push_back(v_ms);

        previous_height_m = z_m;
        previous_theta_v = theta_v;
    }

    const std::size_t parsed_levels = data.num_levels();
    if (!quality_control_sounding(data, config_))
    {
        throw std::runtime_error("CM1 input_sounding failed quality control");
    }
    if (data.num_levels() != parsed_levels)
    {
        throw std::runtime_error(
            "CM1 input_sounding quality control rejected one or more levels; "
            "refusing a silently shortened profile");
    }
    calculate_derived_quantities(data);
    tmv::log_info("Loaded native CM1 input_sounding with ", data.num_levels(),
                  " levels from ", file_path);
    return data;
}

SoundingData Cm1SoundingScheme::interpolate_to_heights(
    const SoundingData& sounding,
    const std::vector<double>& target_heights_m)
{
    if (!initialized_)
    {
        throw std::runtime_error("CM1 sounding reader is not initialized");
    }

    using namespace tmv::soundings::sharpy;
    switch (static_cast<int>(config_.interpolation_method))
    {
    case 0:
        return interpolate_sounding_linear(sounding, target_heights_m, config_);
    case 1:
        return interpolate_sounding_monotone_spline(
            sounding, target_heights_m, config_);
    case 2:
    {
        auto result = interpolate_sounding_linear(
            sounding, target_heights_m, config_);
        apply_log_linear_pressure_profile(
            sounding, target_heights_m, config_, result);
        calculate_derived_quantities(result);
        return result;
    }
    default:
        throw std::runtime_error("unsupported sounding interpolation method");
    }
}
