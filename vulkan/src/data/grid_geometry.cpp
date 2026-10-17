/**
 * @file grid_geometry.cpp
 * @brief Physical grid-manifest parsing and normalized display geometry.
 */

#include "data/grid_geometry.hpp"

#include "util/string_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <regex>

namespace oglcpp
{
namespace
{

bool json_number(const std::string& json, const std::string& key, double& out)
{
    const std::regex pattern(
        "\\\"" + key +
        "\\\"\\s*:\\s*([-+]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][-+]?[0-9]+)?)");
    std::smatch match;
    if (!std::regex_search(json, match, pattern) || match.size() != 2)
    {
        return false;
    }

    try
    {
        std::size_t consumed = 0;
        out = std::stod(match[1].str(), &consumed);
        return consumed == match[1].str().size() && std::isfinite(out);
    }
    catch (...)
    {
        return false;
    }
}

bool json_string(const std::string& json, const std::string& key, std::string& out)
{
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch match;
    if (!std::regex_search(json, match, pattern) || match.size() != 2)
    {
        return false;
    }
    out = match[1].str();
    return true;
}

bool json_object(const std::string& json, const std::string& key, std::string& out)
{
    const std::regex key_pattern("\\\"" + key + "\\\"\\s*:\\s*\\{");
    std::smatch match;
    if (!std::regex_search(json, match, key_pattern))
    {
        return false;
    }

    const std::size_t begin = static_cast<std::size_t>(match.position()) +
                              static_cast<std::size_t>(match.length()) - 1U;
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t pos = begin; pos < json.size(); ++pos)
    {
        const char c = json[pos];
        if (in_string)
        {
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') in_string = true;
        else if (c == '{') ++depth;
        else if (c == '}' && --depth == 0)
        {
            out = json.substr(begin, pos - begin + 1U);
            return true;
        }
    }
    return false;
}

bool json_int_triplet(const std::string& json, const std::string& key,
                      std::array<int, 3>& out)
{
    const std::regex pattern(
        "\\\"" + key +
        "\\\"\\s*:\\s*\\[\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*\\]");
    std::smatch match;
    if (!std::regex_search(json, match, pattern) || match.size() != 4)
    {
        return false;
    }
    try
    {
        for (std::size_t axis = 0; axis < out.size(); ++axis)
        {
            const unsigned long long value = std::stoull(match[axis + 1].str());
            if (value > static_cast<unsigned long long>(std::numeric_limits<int>::max()))
                return false;
            out[axis] = static_cast<int>(value);
        }
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool manifest_dimension(const std::string& json, const char* key, int& out)
{
    double value = 0.0;
    if (!json_number(json, key, value) || value < 1.0 ||
        value > static_cast<double>(std::numeric_limits<int>::max()) ||
        std::floor(value) != value)
    {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

bool nearly_equal(double lhs, double rhs)
{
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= 1e-9 * scale;
}

} // namespace

bool PhysicalGridSpacing::available() const
{
    return std::isfinite(x_m) && std::isfinite(y_m) && std::isfinite(z_m) &&
           x_m > 0.0 && y_m > 0.0 && z_m > 0.0;
}

bool PhysicalIndexRange::valid_for(int nx_value, int ny_value, int nz_value) const
{
    const std::array<int, 3> dimensions{nx_value, ny_value, nz_value};
    for (std::size_t axis = 0; axis < dimensions.size(); ++axis)
    {
        if (dimensions[axis] <= 0 || lower[axis] < 0 ||
            upper[axis] < lower[axis] || upper[axis] >= dimensions[axis])
            return false;
    }
    return true;
}

GridManifestStatus parse_grid_manifest(
    const std::string& json,
    int expected_nx,
    int expected_ny,
    int expected_nz,
    PhysicalGridSpacing& out,
    std::string& detail,
    PhysicalIndexRange* valid_range)
{
    out = {};
    detail.clear();

    PhysicalIndexRange parsed_range{{0, 0, 0},
                                    {expected_nx - 1, expected_ny - 1, expected_nz - 1}};
    if (valid_range) *valid_range = {};

    double version_number = 0.0;
    const bool has_version = json_number(json, "manifest_version", version_number);
    if (has_version && (std::floor(version_number) != version_number || version_number < 1.0 ||
                        version_number > 5.0))
    {
        detail = "unsupported or invalid manifest_version";
        return GridManifestStatus::invalid;
    }
    const int manifest_version = has_version ? static_cast<int>(version_number) : 0;

    std::string grid_json;
    const bool has_grid_object = json_object(json, "grid", grid_json);
    const std::string& grid = has_grid_object ? grid_json : json;

    std::string vertical_geometry_json;
    if (json_object(grid, "vertical_geometry", vertical_geometry_json))
    {
        std::string geometry_kind;
        if (!json_string(vertical_geometry_json, "kind", geometry_kind))
        {
            detail = "vertical_geometry is missing its kind";
            return GridManifestStatus::invalid;
        }
        if (geometry_kind == "terrain_following")
        {
            detail = "terrain-following volume geometry is not supported by the uniform-grid Vulkan viewer";
            return GridManifestStatus::invalid;
        }
        detail = "unsupported vertical_geometry kind";
        return GridManifestStatus::invalid;
    }

    int nx = 0;
    int ny = 0;
    int nz = 0;
    std::string dimensions_json;
    const bool canonical_dimensions = json_object(grid, "dimensions", dimensions_json);
    const bool dimensions_ok = canonical_dimensions
        ? (manifest_dimension(dimensions_json, "x", nx) &&
           manifest_dimension(dimensions_json, "y", ny) &&
           manifest_dimension(dimensions_json, "z", nz))
        : (manifest_dimension(grid, "nr", nx) &&
           manifest_dimension(grid, "nth", ny) &&
           manifest_dimension(grid, "nz", nz));
    if (!dimensions_ok)
    {
        detail = "manifest does not contain valid nr/nth/nz grid dimensions";
        return GridManifestStatus::unavailable;
    }
    if (nx != expected_nx || ny != expected_ny || nz != expected_nz)
    {
        detail = "manifest grid dimensions do not match the volume payload";
        return GridManifestStatus::invalid;
    }

    std::string coordinate_system;
    const bool has_coordinate_system = json_string(grid, "coordinate_system", coordinate_system);
    coordinate_system = tmv::strutil::trim_and_lower(coordinate_system);
    if (has_coordinate_system && coordinate_system != "cartesian")
    {
        detail = "non-Cartesian geometry requires an explicit viewer spacing override";
        return GridManifestStatus::unavailable;
    }

    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
    std::string spacing_json;
    const bool canonical_spacing = json_object(grid, "spacing_m", spacing_json);
    const bool has_dx = json_number(canonical_spacing ? spacing_json : grid,
                                    canonical_spacing ? "x" : "dr_m", dx);
    const bool has_dy = json_number(canonical_spacing ? spacing_json : grid,
                                    canonical_spacing ? "y" : "dy_m", dy);
    const bool has_dz = json_number(canonical_spacing ? spacing_json : grid,
                                    canonical_spacing ? "z" : "dz_m", dz);

    if ((has_dx && dx < 0.0) || (has_dy && dy < 0.0) || (has_dz && dz < 0.0))
    {
        detail = "manifest grid spacing must not be negative";
        return GridManifestStatus::invalid;
    }

    if (!has_dy || dy == 0.0)
    {
        double legacy_dy = 0.0;
        const bool has_legacy_dy = json_number(grid, "dtheta_rad", legacy_dy);
        if (has_legacy_dy && legacy_dy < 0.0)
        {
            detail = "legacy Cartesian y spacing must not be negative";
            return GridManifestStatus::invalid;
        }
        if (has_legacy_dy)
        {
            dy = legacy_dy;
        }
    }

    out = {dx, dy, dz};
    if (!has_dx || !has_dz || !out.available())
    {
        out = {};
        detail = "manifest does not contain positive Cartesian cell spacing";
        return GridManifestStatus::unavailable;
    }

    std::string range_json;
    const bool has_range = json_object(grid, "valid_index_range", range_json);
    if (has_range)
    {
        if (!json_int_triplet(range_json, "lower", parsed_range.lower) ||
            !json_int_triplet(range_json, "upper", parsed_range.upper) ||
            !parsed_range.valid_for(nx, ny, nz))
        {
            detail = "manifest valid_index_range is invalid for the payload dimensions";
            return GridManifestStatus::invalid;
        }
    }
    else if (manifest_version >= 4)
    {
        detail = "manifest v4 requires valid_index_range";
        return GridManifestStatus::invalid;
    }

    if (manifest_version >= 4)
    {
        std::string staggering;
        std::string halo_json;
        std::array<int, 3> halo_lower{};
        std::array<int, 3> halo_upper{};
        if (!json_string(grid, "native_model_staggering", staggering) ||
            !json_object(grid, "halo_width", halo_json) ||
            !json_int_triplet(halo_json, "lower", halo_lower) ||
            !json_int_triplet(halo_json, "upper", halo_upper) ||
            (staggering != "c_grid" && staggering != "collocated"))
        {
            detail = "manifest v4 requires canonical staggering and halo_width";
            return GridManifestStatus::invalid;
        }
        const std::array<int, 3> dimensions{nx, ny, nz};
        for (std::size_t axis = 0; axis < dimensions.size(); ++axis)
        {
            const int expected_halo = (axis < 2 || staggering == "collocated") &&
                                      dimensions[axis] >= 3 ? 1 : 0;
            if (halo_lower[axis] != expected_halo || halo_upper[axis] != expected_halo ||
                parsed_range.lower[axis] != expected_halo ||
                parsed_range.upper[axis] != dimensions[axis] - expected_halo - 1)
            {
                detail = "manifest v4 ownership range contradicts its staggering or halo_width";
                return GridManifestStatus::invalid;
            }
        }
    }

    if (valid_range) *valid_range = parsed_range;

    return GridManifestStatus::available;
}

bool physical_grid_spacing_equal(
    const PhysicalGridSpacing& lhs,
    const PhysicalGridSpacing& rhs)
{
    return lhs.available() && rhs.available() &&
           nearly_equal(lhs.x_m, rhs.x_m) &&
           nearly_equal(lhs.y_m, rhs.y_m) &&
           nearly_equal(lhs.z_m, rhs.z_m);
}

bool physical_index_range_equal(
    const PhysicalIndexRange& lhs,
    const PhysicalIndexRange& rhs)
{
    return lhs.lower == rhs.lower && lhs.upper == rhs.upper;
}

bool normalized_volume_world_scale(
    int nx,
    int ny,
    int nz,
    const PhysicalGridSpacing& spacing,
    float vertical_exaggeration,
    std::array<float, 3>& out_world_scale,
    std::string& error)
{
    out_world_scale = {};
    error.clear();
    if (nx <= 0 || ny <= 0 || nz <= 0)
    {
        error = "volume dimensions must be positive";
        return false;
    }
    if (!spacing.available())
    {
        error = "physical grid spacing must be finite and positive";
        return false;
    }
    if (!std::isfinite(vertical_exaggeration) || vertical_exaggeration <= 0.0f)
    {
        error = "vertical exaggeration must be finite and positive";
        return false;
    }

    const double extent_x = static_cast<double>(nx) * spacing.x_m;
    const double extent_y = static_cast<double>(ny) * spacing.y_m;
    const double extent_z = static_cast<double>(nz) * spacing.z_m;
    const double horizontal = std::max(extent_x, extent_y);
    if (!std::isfinite(horizontal) || !std::isfinite(extent_z) || horizontal <= 0.0)
    {
        error = "physical volume extents overflowed or are invalid";
        return false;
    }

    const std::array<double, 3> scale = {
        extent_x / horizontal,
        extent_z / horizontal * static_cast<double>(vertical_exaggeration),
        extent_y / horizontal,
    };
    for (std::size_t i = 0; i < scale.size(); ++i)
    {
        if (!std::isfinite(scale[i]) || scale[i] <= 0.0 ||
            scale[i] > static_cast<double>(std::numeric_limits<float>::max()))
        {
            error = "normalized physical volume scale is invalid";
            return false;
        }
        out_world_scale[i] = static_cast<float>(scale[i]);
    }
    return true;
}

} // namespace oglcpp
