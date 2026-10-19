/**
 * @file radiative_transfer.cpp
 * @brief Implementation for the radiation module.
 *
 * Provides executable logic for the radiation runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/radiation subsystem.
 */

#include "radiative_transfer.hpp"
#include "core/infra/physical_constants.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>


namespace tmv::radiative_transfer
{

namespace
{
// E1(x) = integral_x^infinity exp(-t)/t dt. The small-x power series and
// large-x continued fraction are the standard complementary representations;
// both converge to binary64 precision over the non-negative optical-depth
// domain used by the grey scheme.
double exponential_integral_e1(double x)
{
    if (!std::isfinite(x) || x <= 0.0)
        throw std::invalid_argument(
            "exponential_integral_e1 requires positive finite input");
    constexpr double euler_gamma =
        0.577215664901532860606512090082402431;
    constexpr int maximum_iterations = 256;
    constexpr double tolerance =
        8.0 * std::numeric_limits<double>::epsilon();
    if (x <= 1.0)
    {
        double sum = -euler_gamma - std::log(x);
        double factorial_term = 1.0;
        for (int k = 1; k <= maximum_iterations; ++k)
        {
            factorial_term *= -x / static_cast<double>(k);
            const double increment = -factorial_term
                / static_cast<double>(k);
            sum += increment;
            if (std::abs(increment)
                <= tolerance * std::max(1.0, std::abs(sum)))
                return sum;
        }
        throw std::runtime_error(
            "shortwave E1 power series did not converge");
    }

    constexpr double inverse_floor = 1.0e300;
    double b = x + 1.0;
    double c = inverse_floor;
    double d = 1.0 / b;
    double fraction = d;
    for (int k = 1; k <= maximum_iterations; ++k)
    {
        const double index = static_cast<double>(k);
        const double a = -index * index;
        b += 2.0;
        d = a * d + b;
        if (std::abs(d) < 1.0 / inverse_floor)
            d = 1.0 / inverse_floor;
        c = b + a / c;
        if (std::abs(c) < 1.0 / inverse_floor)
            c = 1.0 / inverse_floor;
        d = 1.0 / d;
        const double multiplier = c * d;
        fraction *= multiplier;
        if (std::abs(multiplier - 1.0) <= tolerance)
            return fraction * std::exp(-x);
    }
    throw std::runtime_error(
        "shortwave E1 continued fraction did not converge");
}

// E3(x) = integral_1^infinity exp(-x*t)/t^3 dt.  For x > 1, evaluate E3
// directly rather than obtaining it from the E1 recurrence: the latter
// subtracts nearly equal terms as optical depth grows and needlessly loses
// significant bits.  This continued fraction is the order-three form of the
// same convergent representation used above.
double exponential_integral_e3(double x)
{
    if (!std::isfinite(x) || x < 0.0)
        throw std::invalid_argument(
            "exponential_integral_e3 requires finite non-negative input");
    if (x == 0.0)
        return 0.5;
    if (x <= 1.0)
    {
        const double exp_minus_x = std::exp(-x);
        const double e2 = exp_minus_x - x * exponential_integral_e1(x);
        return 0.5 * (exp_minus_x - x * e2);
    }

    constexpr int maximum_iterations = 256;
    constexpr double tolerance =
        8.0 * std::numeric_limits<double>::epsilon();
    constexpr double inverse_floor = 1.0e300;
    double b = x + 3.0;
    double c = inverse_floor;
    double d = 1.0 / b;
    double fraction = d;
    for (int k = 1; k <= maximum_iterations; ++k)
    {
        const double index = static_cast<double>(k);
        const double a = -index * (index + 2.0);
        b += 2.0;
        d = a * d + b;
        if (std::abs(d) < 1.0 / inverse_floor)
            d = 1.0 / inverse_floor;
        c = b + a / c;
        if (std::abs(c) < 1.0 / inverse_floor)
            c = 1.0 / inverse_floor;
        d = 1.0 / d;
        const double multiplier = c * d;
        fraction *= multiplier;
        if (std::abs(multiplier - 1.0) <= tolerance)
            return fraction * std::exp(-x);
    }
    throw std::runtime_error(
        "shortwave E3 continued fraction did not converge");
}
} // namespace

double lambertian_diffuse_transmittance(double optical_depth)
{
    if (!std::isfinite(optical_depth) || optical_depth < 0.0)
        throw std::invalid_argument(
            "Lambertian transmittance requires finite non-negative optical depth");
    if (optical_depth == 0.0)
        return 1.0;
    // The hemispheric angular integral is exactly 2*E3(tau).  Clamp only
    // binary64 endpoint noise.
    const double transmission =
        2.0 * exponential_integral_e3(optical_depth);
    if (!std::isfinite(transmission)
        || transmission < -32.0 * std::numeric_limits<double>::epsilon()
        || transmission > 1.0
            + 32.0 * std::numeric_limits<double>::epsilon())
    {
        throw std::runtime_error(
            "Lambertian diffuse transmittance left its physical range");
    }
    return std::clamp(transmission, 0.0, 1.0);
}

SlopedLambertianDirectBoundary evaluate_sloped_lambertian_direct_boundary(
    double direct_normal_irradiance_wm2,
    const IdealizedSolarGeometry& sun_direction,
    double slope_x,
    double slope_y,
    double albedo,
    bool visible_to_sun)
{
    if (!std::isfinite(direct_normal_irradiance_wm2)
        || direct_normal_irradiance_wm2 < 0.0
        || !std::isfinite(sun_direction.east)
        || !std::isfinite(sun_direction.north)
        || !std::isfinite(sun_direction.up)
        || !std::isfinite(slope_x) || !std::isfinite(slope_y)
        || !std::isfinite(albedo) || albedo < 0.0 || albedo > 1.0)
    {
        throw std::invalid_argument(
            "sloped shortwave boundary requires finite physical inputs");
    }
    const double sun_norm = std::hypot(
        std::hypot(sun_direction.east, sun_direction.north),
        sun_direction.up);
    if (!std::isfinite(sun_norm)
        || std::abs(sun_norm - 1.0) > 1.0e-12)
    {
        throw std::invalid_argument(
            "sloped shortwave boundary requires a unit solar direction");
    }

    SlopedLambertianDirectBoundary result;
    result.surface_area_per_horizontal_area =
        std::hypot(std::hypot(slope_x, slope_y), 1.0);
    if (!visible_to_sun || sun_direction.up <= 0.0
        || direct_normal_irradiance_wm2 == 0.0)
    {
        return result;
    }

    // Upward unit normal of z=h(x,y): (-hx,-hy,1)/sqrt(1+hx^2+hy^2).
    const double normal_dot_sun = (
        -slope_x * sun_direction.east
        - slope_y * sun_direction.north
        + sun_direction.up)
        / result.surface_area_per_horizontal_area;
    result.cosine_incidence = std::clamp(normal_dot_sun, 0.0, 1.0);
    result.incident_irradiance_actual_wm2 =
        direct_normal_irradiance_wm2 * result.cosine_incidence;
    result.incident_power_horizontal_wm2 =
        result.incident_irradiance_actual_wm2
        * result.surface_area_per_horizontal_area;
    result.reflected_power_horizontal_wm2 =
        albedo * result.incident_power_horizontal_wm2;
    // Subtraction makes the local incident = absorbed + reflected ledger exact.
    result.absorbed_power_horizontal_wm2 =
        result.incident_power_horizontal_wm2
        - result.reflected_power_horizontal_wm2;
    result.reflected_radiance_wm2sr =
        albedo * result.incident_irradiance_actual_wm2
        / radiation_constants::pi;
    return result;
}

namespace
{
std::size_t wrapped_index(long long index, std::size_t count)
{
    const long long signed_count = static_cast<long long>(count);
    long long wrapped = index % signed_count;
    if (wrapped < 0)
        wrapped += signed_count;
    return static_cast<std::size_t>(wrapped);
}

double next_lattice_crossing_time(double coordinate,
                                  double velocity,
                                  double spacing)
{
    if (velocity == 0.0)
        return std::numeric_limits<double>::infinity();
    const double scaled = coordinate / spacing;
    const double nearest = std::round(scaled);
    const double snap_tolerance = 32.0
        * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::abs(scaled));
    long long boundary_index = 0;
    if (std::abs(scaled - nearest) <= snap_tolerance)
    {
        boundary_index = static_cast<long long>(nearest)
            + (velocity > 0.0 ? 1 : -1);
    }
    else if (velocity > 0.0)
    {
        boundary_index = static_cast<long long>(std::floor(scaled)) + 1;
    }
    else
    {
        boundary_index = static_cast<long long>(std::ceil(scaled)) - 1;
    }
    const double crossing =
        (static_cast<double>(boundary_index) * spacing - coordinate)
        / velocity;
    if (!std::isfinite(crossing) || crossing <= 0.0)
        throw std::runtime_error(
            "terrain shadow traversal failed to advance to a grid boundary");
    return crossing;
}

double validate_terrain_shadow_inputs(
    const TerrainHeightMapView& terrain,
    const IdealizedSolarGeometry& sun_direction)
{
    if (terrain.height_m == nullptr || terrain.nx < 2 || terrain.ny < 2
        || terrain.nx > std::numeric_limits<std::size_t>::max() / terrain.ny
        || terrain.nx > static_cast<std::size_t>(
            std::numeric_limits<long long>::max())
        || terrain.ny > static_cast<std::size_t>(
            std::numeric_limits<long long>::max())
        || !std::isfinite(terrain.dx_m) || terrain.dx_m <= 0.0
        || !std::isfinite(terrain.dy_m) || terrain.dy_m <= 0.0
        || !std::isfinite(sun_direction.east)
        || !std::isfinite(sun_direction.north)
        || !std::isfinite(sun_direction.up))
        throw std::invalid_argument(
            "terrain shadow requires a finite non-empty height map");
    const double sun_norm = std::hypot(
        std::hypot(sun_direction.east, sun_direction.north),
        sun_direction.up);
    if (!std::isfinite(sun_norm)
        || std::abs(sun_norm - 1.0) > 1.0e-12)
        throw std::invalid_argument(
            "terrain shadow requires a unit solar direction");
    double maximum_height = -std::numeric_limits<double>::infinity();
    const std::size_t count = terrain.nx * terrain.ny;
    for (std::size_t index = 0; index < count; ++index)
    {
        if (!std::isfinite(terrain.height_m[index]))
            throw std::invalid_argument(
                "terrain shadow requires finite heights");
        maximum_height = std::max(maximum_height, terrain.height_m[index]);
    }
    return maximum_height;
}

std::uint64_t build_terrain_shadow_hierarchy(
    const TerrainHeightMapView& terrain,
    TerrainShadowMaskWorkspace& workspace)
{
    const std::size_t patch_nx =
        terrain.x_boundary == TerrainShadowBoundary::Periodic
            ? terrain.nx : terrain.nx - 1;
    const std::size_t patch_ny =
        terrain.y_boundary == TerrainShadowBoundary::Periodic
            ? terrain.ny : terrain.ny - 1;
    workspace.hierarchy_level_offset.clear();
    workspace.hierarchy_level_nx.clear();
    workspace.hierarchy_level_ny.clear();
    std::size_t level_nx = patch_nx;
    std::size_t level_ny = patch_ny;
    std::size_t total_nodes = 0;
    while (true)
    {
        if (level_nx > std::numeric_limits<std::size_t>::max() / level_ny
            || total_nodes > std::numeric_limits<std::size_t>::max()
                - level_nx * level_ny)
            throw std::overflow_error(
                "terrain shadow hierarchy size overflowed");
        workspace.hierarchy_level_offset.push_back(total_nodes);
        workspace.hierarchy_level_nx.push_back(level_nx);
        workspace.hierarchy_level_ny.push_back(level_ny);
        total_nodes += level_nx * level_ny;
        if (level_nx == 1 && level_ny == 1)
            break;
        level_nx = (level_nx + 1) / 2;
        level_ny = (level_ny + 1) / 2;
    }
    workspace.hierarchy_max_height_m.resize(total_nodes);

    const auto vertex = [&](std::size_t x, std::size_t y)
    {
        if (terrain.x_boundary == TerrainShadowBoundary::Periodic)
            x %= terrain.nx;
        if (terrain.y_boundary == TerrainShadowBoundary::Periodic)
            y %= terrain.ny;
        return terrain.height_m[x * terrain.ny + y];
    };
    const std::size_t leaf_offset = workspace.hierarchy_level_offset[0];
    #pragma omp parallel for collapse(2) \
        if(patch_nx * patch_ny >= 4096) schedule(static)
    for (std::size_t x = 0; x < patch_nx; ++x)
        for (std::size_t y = 0; y < patch_ny; ++y)
            workspace.hierarchy_max_height_m[
                leaf_offset + x * patch_ny + y] = std::max({
                    vertex(x, y), vertex(x + 1, y),
                    vertex(x, y + 1), vertex(x + 1, y + 1)});

    for (std::size_t level = 1;
         level < workspace.hierarchy_level_offset.size(); ++level)
    {
        const std::size_t previous_offset =
            workspace.hierarchy_level_offset[level - 1];
        const std::size_t previous_nx =
            workspace.hierarchy_level_nx[level - 1];
        const std::size_t previous_ny =
            workspace.hierarchy_level_ny[level - 1];
        const std::size_t offset =
            workspace.hierarchy_level_offset[level];
        const std::size_t nx = workspace.hierarchy_level_nx[level];
        const std::size_t ny = workspace.hierarchy_level_ny[level];
        #pragma omp parallel for collapse(2) \
            if(nx * ny >= 4096) schedule(static)
        for (std::size_t x = 0; x < nx; ++x)
            for (std::size_t y = 0; y < ny; ++y)
            {
                double maximum = -std::numeric_limits<double>::infinity();
                for (std::size_t child_x = 2 * x;
                     child_x < std::min(2 * x + 2, previous_nx); ++child_x)
                    for (std::size_t child_y = 2 * y;
                         child_y < std::min(2 * y + 2, previous_ny); ++child_y)
                        maximum = std::max(maximum,
                            workspace.hierarchy_max_height_m[
                                previous_offset
                                + child_x * previous_ny + child_y]);
                workspace.hierarchy_max_height_m[offset + x * ny + y] =
                    maximum;
            }
    }
    return static_cast<std::uint64_t>(total_nodes);
}

long long patch_index_ahead(double coordinate,
                            double velocity,
                            double spacing,
                            std::size_t open_patch_count,
                            TerrainShadowBoundary boundary)
{
    const double scaled = coordinate / spacing;
    const double nearest = std::round(scaled);
    const double snap_tolerance = 32.0
        * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::abs(scaled));
    long long patch = static_cast<long long>(std::floor(scaled));
    if (velocity < 0.0
        && std::abs(scaled - nearest) <= snap_tolerance)
        patch = static_cast<long long>(nearest) - 1;
    if (velocity == 0.0 && boundary == TerrainShadowBoundary::Open)
    {
        patch = std::max<long long>(0, std::min<long long>(
            patch, static_cast<long long>(open_patch_count - 1)));
    }
    return patch;
}

struct PeriodicPatchCoordinate
{
    long long cycle = 0;
    std::size_t local = 0;
};

PeriodicPatchCoordinate resolve_patch_coordinate(long long patch,
                                                 std::size_t patch_count)
{
    const long long count = static_cast<long long>(patch_count);
    long long cycle = patch / count;
    long long local = patch % count;
    if (local < 0)
    {
        local += count;
        --cycle;
    }
    return {cycle, static_cast<std::size_t>(local)};
}

double conservative_hierarchy_skip_interval(
    const TerrainHeightMapView& terrain,
    const TerrainShadowMaskWorkspace& hierarchy,
    const IdealizedSolarGeometry& sun_direction,
    double x,
    double y,
    double ray_height,
    double maximum_interval,
    double clearance_tolerance)
{
    const std::size_t patch_nx =
        terrain.x_boundary == TerrainShadowBoundary::Periodic
            ? terrain.nx : terrain.nx - 1;
    const std::size_t patch_ny =
        terrain.y_boundary == TerrainShadowBoundary::Periodic
            ? terrain.ny : terrain.ny - 1;
    const long long patch_x = patch_index_ahead(
        x, sun_direction.east, terrain.dx_m,
        patch_nx, terrain.x_boundary);
    const long long patch_y = patch_index_ahead(
        y, sun_direction.north, terrain.dy_m,
        patch_ny, terrain.y_boundary);
    if ((terrain.x_boundary == TerrainShadowBoundary::Open
         && (patch_x < 0 || patch_x >= static_cast<long long>(patch_nx)))
        || (terrain.y_boundary == TerrainShadowBoundary::Open
            && (patch_y < 0 || patch_y >= static_cast<long long>(patch_ny))))
        return 0.0;
    const PeriodicPatchCoordinate x_coordinate =
        resolve_patch_coordinate(patch_x, patch_nx);
    const PeriodicPatchCoordinate y_coordinate =
        resolve_patch_coordinate(patch_y, patch_ny);

    for (std::size_t reverse = hierarchy.hierarchy_level_offset.size();
         reverse > 0; --reverse)
    {
        const std::size_t level = reverse - 1;
        const std::size_t block_size = std::size_t{1} << level;
        const std::size_t node_x = x_coordinate.local / block_size;
        const std::size_t node_y = y_coordinate.local / block_size;
        const std::size_t level_ny = hierarchy.hierarchy_level_ny[level];
        const double node_maximum = hierarchy.hierarchy_max_height_m[
            hierarchy.hierarchy_level_offset[level]
            + node_x * level_ny + node_y];
        if (node_maximum > ray_height + clearance_tolerance)
            continue;

        const std::size_t local_x_begin = node_x * block_size;
        const std::size_t local_y_begin = node_y * block_size;
        const std::size_t local_x_end = std::min(
            local_x_begin + block_size, patch_nx);
        const std::size_t local_y_end = std::min(
            local_y_begin + block_size, patch_ny);
        const long long unwrapped_x_begin = x_coordinate.cycle
            * static_cast<long long>(patch_nx)
            + static_cast<long long>(local_x_begin);
        const long long unwrapped_x_end = x_coordinate.cycle
            * static_cast<long long>(patch_nx)
            + static_cast<long long>(local_x_end);
        const long long unwrapped_y_begin = y_coordinate.cycle
            * static_cast<long long>(patch_ny)
            + static_cast<long long>(local_y_begin);
        const long long unwrapped_y_end = y_coordinate.cycle
            * static_cast<long long>(patch_ny)
            + static_cast<long long>(local_y_end);
        const auto exit_time = [](double coordinate, double velocity,
                                  double spacing, long long lower,
                                  long long upper)
        {
            if (velocity == 0.0)
                return std::numeric_limits<double>::infinity();
            const long long boundary = velocity > 0.0 ? upper : lower;
            return (static_cast<double>(boundary) * spacing - coordinate)
                / velocity;
        };
        const double interval = std::min({
            exit_time(x, sun_direction.east, terrain.dx_m,
                      unwrapped_x_begin, unwrapped_x_end),
            exit_time(y, sun_direction.north, terrain.dy_m,
                      unwrapped_y_begin, unwrapped_y_end),
            maximum_interval});
        if (std::isfinite(interval) && interval > 0.0)
            return interval;
    }
    return 0.0;
}

bool trace_validated_terrain_visibility(
    const TerrainHeightMapView& terrain,
    std::size_t i,
    std::size_t j,
    const IdealizedSolarGeometry& sun_direction,
    double maximum_height,
    const TerrainShadowMaskWorkspace* hierarchy,
    std::uint64_t* patch_count,
    std::uint64_t* skipped_block_count)
{
    if (sun_direction.up <= 0.0)
        return false;
    const double start_x = static_cast<double>(i) * terrain.dx_m;
    const double start_y = static_cast<double>(j) * terrain.dy_m;
    const double start_z = terrain.height_m[i * terrain.ny + j];
    if (std::hypot(sun_direction.east, sun_direction.north)
            <= 32.0 * std::numeric_limits<double>::epsilon()
        || maximum_height <= start_z)
        return true;

    const double maximum_time = (maximum_height - start_z)
        / sun_direction.up;
    const double x_open_max = static_cast<double>(terrain.nx - 1)
        * terrain.dx_m;
    const double y_open_max = static_cast<double>(terrain.ny - 1)
        * terrain.dy_m;
    const auto is_outside_open_axis = [](double coordinate,
                                         double maximum,
                                         TerrainShadowBoundary boundary)
    {
        return boundary == TerrainShadowBoundary::Open
            && (coordinate < 0.0 || coordinate > maximum);
    };
    const auto height_at = [&](double x, double y)
    {
        const auto coordinate = [](
            double scaled, std::size_t count,
            TerrainShadowBoundary boundary)
        {
            if (boundary == TerrainShadowBoundary::Open)
            {
                if (scaled <= 0.0)
                    return std::pair<long long, double>{0, 0.0};
                const double upper = static_cast<double>(count - 1);
                if (scaled >= upper)
                    return std::pair<long long, double>{
                        static_cast<long long>(count - 2), 1.0};
            }
            const long long lower =
                static_cast<long long>(std::floor(scaled));
            return std::pair<long long, double>{
                lower, scaled - static_cast<double>(lower)};
        };
        const auto x_coordinate = coordinate(
            x / terrain.dx_m, terrain.nx, terrain.x_boundary);
        const auto y_coordinate = coordinate(
            y / terrain.dy_m, terrain.ny, terrain.y_boundary);
        const auto resolve = [](long long index, std::size_t count,
                                TerrainShadowBoundary boundary)
        {
            if (boundary == TerrainShadowBoundary::Periodic)
                return wrapped_index(index, count);
            if (index < 0 || index >= static_cast<long long>(count))
                throw std::out_of_range(
                    "terrain shadow interpolation left an open height map");
            return static_cast<std::size_t>(index);
        };
        const std::size_t x0 = resolve(
            x_coordinate.first, terrain.nx, terrain.x_boundary);
        const std::size_t x1 = resolve(
            x_coordinate.first + 1, terrain.nx, terrain.x_boundary);
        const std::size_t y0 = resolve(
            y_coordinate.first, terrain.ny, terrain.y_boundary);
        const std::size_t y1 = resolve(
            y_coordinate.first + 1, terrain.ny, terrain.y_boundary);
        const double fx = x_coordinate.second;
        const double fy = y_coordinate.second;
        const double h00 = terrain.height_m[x0 * terrain.ny + y0];
        const double h10 = terrain.height_m[x1 * terrain.ny + y0];
        const double h01 = terrain.height_m[x0 * terrain.ny + y1];
        const double h11 = terrain.height_m[x1 * terrain.ny + y1];
        return (1.0 - fx) * (1.0 - fy) * h00
             + fx * (1.0 - fy) * h10
             + (1.0 - fx) * fy * h01
             + fx * fy * h11;
    };

    double time = 0.0;
    const double clearance_tolerance = 256.0
        * std::numeric_limits<double>::epsilon()
        * std::max({1.0, std::abs(maximum_height), std::abs(start_z)});
    while (time < maximum_time)
    {
        const double x = start_x + time * sun_direction.east;
        const double y = start_y + time * sun_direction.north;
        if (hierarchy != nullptr)
        {
            const double skip_interval = conservative_hierarchy_skip_interval(
                terrain, *hierarchy, sun_direction, x, y,
                start_z + time * sun_direction.up,
                maximum_time - time, clearance_tolerance);
            if (skip_interval > 0.0 && time + skip_interval > time)
            {
                time += skip_interval;
                if (skipped_block_count != nullptr)
                    ++(*skipped_block_count);
                continue;
            }
        }
        const double interval = std::min({
            next_lattice_crossing_time(
                x, sun_direction.east, terrain.dx_m),
            next_lattice_crossing_time(
                y, sun_direction.north, terrain.dy_m),
            maximum_time - time});
        if (!std::isfinite(interval) || interval <= 0.0)
            throw std::runtime_error(
                "terrain shadow traversal produced an invalid ray interval");
        const double end_time = time + interval;
        const double end_x = start_x + end_time * sun_direction.east;
        const double end_y = start_y + end_time * sun_direction.north;
        if (is_outside_open_axis(end_x, x_open_max, terrain.x_boundary)
            || is_outside_open_axis(end_y, y_open_max, terrain.y_boundary))
            return true;
        if (patch_count != nullptr)
            ++(*patch_count);

        const auto clearance = [&](double fraction)
        {
            const double sample_time = time + fraction * interval;
            return height_at(
                start_x + sample_time * sun_direction.east,
                start_y + sample_time * sun_direction.north)
                - (start_z + sample_time * sun_direction.up);
        };
        const double q0 = clearance(0.0);
        const double q_half = clearance(0.5);
        const double q1 = clearance(1.0);
        const double quadratic = 2.0 * (q1 + q0 - 2.0 * q_half);
        const double linear = q1 - q0 - quadratic;
        double maximum_clearance = std::max(q_half, q1);
        if (quadratic < 0.0)
        {
            const double vertex = -linear / (2.0 * quadratic);
            if (vertex > 0.0 && vertex < 1.0)
                maximum_clearance = std::max(
                    maximum_clearance,
                    quadratic * vertex * vertex + linear * vertex + q0);
        }
        const double height_scale = std::max(
            {1.0, std::abs(maximum_height), std::abs(start_z),
             std::abs(start_z + end_time * sun_direction.up)});
        if (maximum_clearance
            > 256.0 * std::numeric_limits<double>::epsilon() * height_scale)
            return false;
        time = end_time;
    }
    return true;
}
} // namespace

bool terrain_height_map_visible_to_sun_reference(
    const TerrainHeightMapView& terrain,
    std::size_t i,
    std::size_t j,
    const IdealizedSolarGeometry& sun_direction,
    std::uint64_t* bilinear_patches_tested)
{
    if (i >= terrain.nx || j >= terrain.ny)
        throw std::invalid_argument(
            "terrain shadow reference source is outside the height map");
    const double maximum_height = validate_terrain_shadow_inputs(
        terrain, sun_direction);
    if (bilinear_patches_tested != nullptr)
        *bilinear_patches_tested = 0;
    return trace_validated_terrain_visibility(
        terrain, i, j, sun_direction, maximum_height,
        nullptr, bilinear_patches_tested, nullptr);
}

void terrain_height_map_visibility_field_cpu(
    const TerrainHeightMapView& terrain,
    const IdealizedSolarGeometry& sun_direction,
    TerrainShadowMaskWorkspace& workspace,
    std::vector<unsigned char>& visibility,
    TerrainShadowTraversalStats* stats)
{
    if (&workspace.scratch_visibility == &visibility)
        throw std::invalid_argument(
            "terrain shadow scratch and published visibility must be distinct");
    const double maximum_height = validate_terrain_shadow_inputs(
        terrain, sun_direction);
    const std::size_t count = terrain.nx * terrain.ny;
    workspace.scratch_visibility.resize(count);
    const double horizontal_speed = std::hypot(
        sun_direction.east, sun_direction.north);
    if (sun_direction.up <= 0.0
        || horizontal_speed
            <= 32.0 * std::numeric_limits<double>::epsilon())
    {
        std::fill(workspace.scratch_visibility.begin(),
                  workspace.scratch_visibility.end(),
                  sun_direction.up > 0.0 ? 1 : 0);
        std::swap(visibility, workspace.scratch_visibility);
        if (stats != nullptr)
        {
            *stats = {};
            stats->height_values_validated = count;
        }
        return;
    }
    const std::uint64_t hierarchy_nodes =
        build_terrain_shadow_hierarchy(terrain, workspace);
    std::fill(workspace.scratch_visibility.begin(),
              workspace.scratch_visibility.end(), 0);
    unsigned long long tested_patches = 0;
    unsigned long long skipped_blocks = 0;
    int failed_rays = 0;

    #pragma omp parallel for if(count >= 256) schedule(static) \
        reduction(+:tested_patches,skipped_blocks,failed_rays)
    for (std::size_t index = 0; index < count; ++index)
    {
        try
        {
            std::uint64_t local_patches = 0;
            std::uint64_t local_skipped_blocks = 0;
            const bool visible = trace_validated_terrain_visibility(
                terrain, index / terrain.ny, index % terrain.ny,
                sun_direction, maximum_height, &workspace,
                &local_patches, &local_skipped_blocks);
            workspace.scratch_visibility[index] = visible ? 1 : 0;
            tested_patches += local_patches;
            skipped_blocks += local_skipped_blocks;
        }
        catch (const std::exception&)
        {
            ++failed_rays;
        }
    }
    if (failed_rays != 0)
        throw std::runtime_error(
            "terrain shadow batch traversal failed before publication");
    std::swap(visibility, workspace.scratch_visibility);
    if (stats != nullptr)
    {
        stats->height_values_validated = count;
        stats->rays_traced = count;
        stats->hierarchy_nodes_built = hierarchy_nodes;
        stats->hierarchy_blocks_skipped = skipped_blocks;
        stats->bilinear_patches_tested = tested_patches;
    }
}

DirectBeamPathResult attenuate_direct_beam_path_reference(
    double incident_power_w,
    const std::vector<DirectBeamPathSegment>& ordered_segments,
    std::vector<DirectBeamSegmentAbsorption>& segment_absorption)
{
    if (!std::isfinite(incident_power_w) || incident_power_w < 0.0)
        throw std::invalid_argument(
            "direct-beam path requires finite non-negative incident power");

    // Complete preflight precedes publication. Use long double for the check
    // so two individually finite geometric values cannot overflow silently in
    // the binary64 path optical depth.
    long double total_slant_optical_depth = 0.0L;
    for (const DirectBeamPathSegment& segment : ordered_segments)
    {
        if (!std::isfinite(segment.path_length_m)
            || segment.path_length_m <= 0.0
            || !std::isfinite(segment.vertical_layer_optical_depth)
            || segment.vertical_layer_optical_depth < 0.0
            || !std::isfinite(segment.physical_layer_thickness_m)
            || segment.physical_layer_thickness_m <= 0.0)
        {
            throw std::invalid_argument(
                "direct-beam path segment has invalid geometry or optical depth");
        }
        const long double slant_optical_depth =
            static_cast<long double>(segment.vertical_layer_optical_depth)
            * static_cast<long double>(segment.path_length_m)
            / static_cast<long double>(segment.physical_layer_thickness_m);
        total_slant_optical_depth += slant_optical_depth;
        if (!std::isfinite(slant_optical_depth)
            || slant_optical_depth
                > static_cast<long double>(
                    std::numeric_limits<double>::max())
            || !std::isfinite(total_slant_optical_depth)
            || total_slant_optical_depth
                > static_cast<long double>(
                    std::numeric_limits<double>::max()))
        {
            throw std::overflow_error(
                "direct-beam path optical depth overflowed binary64");
        }
    }

    segment_absorption.resize(ordered_segments.size());
    double transmitted_power_w = incident_power_w;
    for (std::size_t index = 0; index < ordered_segments.size(); ++index)
    {
        const DirectBeamPathSegment& segment = ordered_segments[index];
        const double slant_optical_depth = static_cast<double>(
            static_cast<long double>(segment.vertical_layer_optical_depth)
            * static_cast<long double>(segment.path_length_m)
            / static_cast<long double>(segment.physical_layer_thickness_m));
        const double incoming_power_w = transmitted_power_w;
        // -expm1(-tau) preserves absorption when tau is optically thin. Form
        // transmission by subtraction so each local power identity is exact.
        const double absorbed_fraction = -std::expm1(-slant_optical_depth);
        const double absorbed_power_w =
            incoming_power_w * absorbed_fraction;
        transmitted_power_w = incoming_power_w - absorbed_power_w;
        segment_absorption[index] = {
            segment.cell_index,
            slant_optical_depth,
            incoming_power_w,
            absorbed_power_w,
            transmitted_power_w};
    }

    DirectBeamPathResult result;
    result.incident_power_w = incident_power_w;
    result.transmitted_power_w = transmitted_power_w;
    // Subtraction makes the path boundary identity exact. Segment-resolved
    // aggregation is independently checked within binary64 summation error.
    result.absorbed_power_w = incident_power_w - transmitted_power_w;
    result.total_slant_optical_depth =
        static_cast<double>(total_slant_optical_depth);
    return result;
}

std::optional<double> first_bilinear_mapped_interface_intersection_reference(
    const BilinearMappedInterfacePatch& patch,
    double start_x_m,
    double start_y_m,
    double start_z_m,
    const IdealizedSolarGeometry& ray_direction,
    double maximum_path_length_m,
    MappedInterfaceCrossing crossing)
{
    if (!std::isfinite(patch.z00_m) || !std::isfinite(patch.z10_m)
        || !std::isfinite(patch.z01_m) || !std::isfinite(patch.z11_m)
        || !std::isfinite(patch.x_lower_m)
        || !std::isfinite(patch.y_lower_m)
        || !std::isfinite(patch.dx_m) || patch.dx_m <= 0.0
        || !std::isfinite(patch.dy_m) || patch.dy_m <= 0.0
        || !std::isfinite(start_x_m) || !std::isfinite(start_y_m)
        || !std::isfinite(start_z_m)
        || !std::isfinite(maximum_path_length_m)
        || maximum_path_length_m < 0.0
        || !std::isfinite(ray_direction.east)
        || !std::isfinite(ray_direction.north)
        || !std::isfinite(ray_direction.up)
        || (crossing != MappedInterfaceCrossing::FromAbove
            && crossing != MappedInterfaceCrossing::FromBelow))
    {
        throw std::invalid_argument(
            "bilinear mapped-interface intersection requires finite geometry");
    }
    const double direction_norm = std::hypot(
        std::hypot(ray_direction.east, ray_direction.north),
        ray_direction.up);
    if (!std::isfinite(direction_norm)
        || std::abs(direction_norm - 1.0) > 1.0e-12)
    {
        throw std::invalid_argument(
            "bilinear mapped-interface intersection requires a unit ray");
    }
    const double x_upper_m = patch.x_lower_m + patch.dx_m;
    const double y_upper_m = patch.y_lower_m + patch.dy_m;
    if (!std::isfinite(x_upper_m) || !std::isfinite(y_upper_m))
        throw std::overflow_error(
            "bilinear mapped-interface patch extent overflowed");
    const double end_x_m = start_x_m
        + ray_direction.east * maximum_path_length_m;
    const double end_y_m = start_y_m
        + ray_direction.north * maximum_path_length_m;
    const double coordinate_scale = std::max({
        1.0, std::abs(patch.x_lower_m), std::abs(x_upper_m),
        std::abs(patch.y_lower_m), std::abs(y_upper_m),
        std::abs(start_x_m), std::abs(start_y_m),
        std::abs(end_x_m), std::abs(end_y_m)});
    const double coordinate_tolerance = 128.0
        * std::numeric_limits<double>::epsilon() * coordinate_scale;
    const auto inside = [coordinate_tolerance](
        double value, double lower, double upper)
    {
        return value >= lower - coordinate_tolerance
            && value <= upper + coordinate_tolerance;
    };
    if (!std::isfinite(end_x_m) || !std::isfinite(end_y_m)
        || !inside(start_x_m, patch.x_lower_m, x_upper_m)
        || !inside(end_x_m, patch.x_lower_m, x_upper_m)
        || !inside(start_y_m, patch.y_lower_m, y_upper_m)
        || !inside(end_y_m, patch.y_lower_m, y_upper_m))
    {
        throw std::invalid_argument(
            "bilinear mapped-interface ray interval leaves its patch");
    }

    const double fx0 = std::clamp(
        (start_x_m - patch.x_lower_m) / patch.dx_m, 0.0, 1.0);
    const double fy0 = std::clamp(
        (start_y_m - patch.y_lower_m) / patch.dy_m, 0.0, 1.0);
    const double dfx = ray_direction.east / patch.dx_m;
    const double dfy = ray_direction.north / patch.dy_m;
    const double hx = patch.z10_m - patch.z00_m;
    const double hy = patch.z01_m - patch.z00_m;
    const double hxy = patch.z11_m - patch.z10_m
        - patch.z01_m + patch.z00_m;
    // q(s) = interface_height(s) - ray_height(s). Above the interface q<0;
    // below it q>0.
    const double quadratic = hxy * dfx * dfy;
    const double linear = hx * dfx + hy * dfy
        + hxy * (fx0 * dfy + fy0 * dfx) - ray_direction.up;
    const double constant = patch.z00_m + hx * fx0 + hy * fy0
        + hxy * fx0 * fy0 - start_z_m;
    const double height_scale = std::max({
        1.0, std::abs(patch.z00_m), std::abs(patch.z10_m),
        std::abs(patch.z01_m), std::abs(patch.z11_m),
        std::abs(start_z_m),
        std::abs(start_z_m
            + ray_direction.up * maximum_path_length_m)});
    const double clearance_tolerance = 256.0
        * std::numeric_limits<double>::epsilon() * height_scale;
    if ((crossing == MappedInterfaceCrossing::FromAbove
            && constant > clearance_tolerance)
        || (crossing == MappedInterfaceCrossing::FromBelow
            && constant < -clearance_tolerance))
    {
        throw std::invalid_argument(
            "bilinear mapped-interface ray starts on the wrong side");
    }
    const double derivative_scale = std::max({
        1.0, std::abs(linear),
        std::abs(2.0 * quadratic * maximum_path_length_m)});
    const double derivative_tolerance = 256.0
        * std::numeric_limits<double>::epsilon() * derivative_scale;
    const auto has_requested_sense = [&](double path_length)
    {
        const double derivative =
            2.0 * quadratic * path_length + linear;
        return crossing == MappedInterfaceCrossing::FromAbove
            ? derivative > derivative_tolerance
            : derivative < -derivative_tolerance;
    };
    if (std::abs(constant) <= clearance_tolerance
        && has_requested_sense(0.0))
        return 0.0;
    if (maximum_path_length_m == 0.0)
        return std::nullopt;

    std::array<double, 2> roots{};
    std::size_t root_count = 0;
    const auto append_root = [&](double root)
    {
        roots[root_count++] = root;
    };
    const double coefficient_scale = std::max({
        1.0, std::abs(quadratic) * maximum_path_length_m
            * maximum_path_length_m,
        std::abs(linear) * maximum_path_length_m,
        std::abs(constant)});
    const double coefficient_tolerance = 256.0
        * std::numeric_limits<double>::epsilon() * coefficient_scale;
    if (std::abs(quadratic) * maximum_path_length_m
            * maximum_path_length_m <= coefficient_tolerance)
    {
        if (std::abs(linear) * maximum_path_length_m
                > coefficient_tolerance)
            append_root(-constant / linear);
    }
    else
    {
        double discriminant = linear * linear
            - 4.0 * quadratic * constant;
        const double discriminant_scale = std::max({
            1.0, std::abs(linear * linear),
            std::abs(4.0 * quadratic * constant)});
        const double discriminant_tolerance = 256.0
            * std::numeric_limits<double>::epsilon() * discriminant_scale;
        if (discriminant >= -discriminant_tolerance)
        {
            discriminant = std::max(0.0, discriminant);
            const double square_root = std::sqrt(discriminant);
            const double q = -0.5
                * (linear + std::copysign(square_root, linear));
            if (q != 0.0)
            {
                append_root(q / quadratic);
                append_root(constant / q);
            }
            else
            {
                append_root(-linear / (2.0 * quadratic));
            }
        }
    }
    std::sort(roots.begin(), roots.begin() + root_count);
    const double path_tolerance = 256.0
        * std::numeric_limits<double>::epsilon()
        * std::max(1.0, maximum_path_length_m);
    for (std::size_t index = 0; index < root_count; ++index)
    {
        const double root = roots[index];
        if (!std::isfinite(root) || root <= path_tolerance
            || root > maximum_path_length_m + path_tolerance
            || !has_requested_sense(root))
            continue;
        return std::clamp(root, 0.0, maximum_path_length_m);
    }
    return std::nullopt;
}

DirectBeamRayResult
trace_rectilinear_direct_beam_from_top_reference(
    const RectilinearDirectBeamGridView& grid,
    std::size_t top_i,
    std::size_t top_j,
    const IdealizedSolarGeometry& sun_direction,
    double incident_power_w,
    std::vector<DirectBeamPathSegment>& ordered_segments,
    std::vector<DirectBeamSegmentAbsorption>& segment_absorption)
{
    if (grid.vertical_layer_optical_depth == nullptr
        || grid.z_interface_m == nullptr
        || grid.nx == 0 || grid.ny == 0 || grid.nz == 0
        || grid.nz == std::numeric_limits<std::size_t>::max()
        || grid.nx > std::numeric_limits<std::size_t>::max() / grid.ny
        || grid.nx * grid.ny
            > std::numeric_limits<std::size_t>::max() / grid.nz
        || grid.nx > static_cast<std::size_t>(
            std::numeric_limits<long long>::max())
        || grid.ny > static_cast<std::size_t>(
            std::numeric_limits<long long>::max())
        || !std::isfinite(grid.dx_m) || grid.dx_m <= 0.0
        || !std::isfinite(grid.dy_m) || grid.dy_m <= 0.0
        || (grid.x_boundary != TerrainShadowBoundary::Open
            && grid.x_boundary != TerrainShadowBoundary::Periodic)
        || (grid.y_boundary != TerrainShadowBoundary::Open
            && grid.y_boundary != TerrainShadowBoundary::Periodic)
        || top_i >= grid.nx || top_j >= grid.ny
        || !std::isfinite(incident_power_w) || incident_power_w < 0.0)
    {
        throw std::invalid_argument(
            "rectilinear direct-beam ray has invalid grid, entry, or power");
    }
    const double sun_norm = std::hypot(
        std::hypot(sun_direction.east, sun_direction.north),
        sun_direction.up);
    if (!std::isfinite(sun_direction.east)
        || !std::isfinite(sun_direction.north)
        || !std::isfinite(sun_direction.up)
        || !std::isfinite(sun_norm)
        || std::abs(sun_norm - 1.0) > 1.0e-12
        || sun_direction.up <= 0.0)
    {
        throw std::invalid_argument(
            "rectilinear direct-beam ray requires a daytime unit direction");
    }
    for (std::size_t k = 0; k <= grid.nz; ++k)
    {
        if (!std::isfinite(grid.z_interface_m[k])
            || (k > 0
                && (grid.z_interface_m[k] <= grid.z_interface_m[k - 1]
                    || static_cast<long double>(grid.z_interface_m[k])
                            - static_cast<long double>(
                                grid.z_interface_m[k - 1])
                        > static_cast<long double>(
                            std::numeric_limits<double>::max()))))
        {
            throw std::invalid_argument(
                "rectilinear direct-beam interfaces must increase with height");
        }
    }
    const std::size_t cell_count = grid.nx * grid.ny * grid.nz;
    for (std::size_t index = 0; index < cell_count; ++index)
    {
        if (!std::isfinite(grid.vertical_layer_optical_depth[index])
            || grid.vertical_layer_optical_depth[index] < 0.0)
        {
            throw std::invalid_argument(
                "rectilinear direct-beam optical depth must be finite and non-negative");
        }
    }

    const double velocity_x = -sun_direction.east;
    const double velocity_y = -sun_direction.north;
    const double velocity_z = -sun_direction.up;
    const long double domain_x_m =
        static_cast<long double>(grid.nx) * grid.dx_m;
    const long double domain_y_m =
        static_cast<long double>(grid.ny) * grid.dy_m;
    const long double full_path_length =
        (static_cast<long double>(grid.z_interface_m[grid.nz])
         - static_cast<long double>(grid.z_interface_m[0]))
        / static_cast<long double>(sun_direction.up);
    const long double crossing_bound =
        std::ceil(std::abs(static_cast<long double>(velocity_x))
                  * full_path_length / grid.dx_m)
        + std::ceil(std::abs(static_cast<long double>(velocity_y))
                    * full_path_length / grid.dy_m)
        + static_cast<long double>(grid.nz) + 8.0L;
    if (!std::isfinite(domain_x_m) || !std::isfinite(domain_y_m)
        || domain_x_m
            > static_cast<long double>(std::numeric_limits<double>::max())
        || domain_y_m
            > static_cast<long double>(std::numeric_limits<double>::max())
        || !std::isfinite(full_path_length) || full_path_length <= 0.0L
        || full_path_length
            > static_cast<long double>(std::numeric_limits<double>::max())
        || !std::isfinite(crossing_bound)
        || crossing_bound
            > static_cast<long double>(
                std::numeric_limits<std::size_t>::max()))
    {
        throw std::overflow_error(
            "rectilinear direct-beam traversal bound overflowed");
    }
    const std::size_t maximum_segments =
        static_cast<std::size_t>(crossing_bound);

    std::vector<DirectBeamPathSegment> candidate_segments;
    candidate_segments.reserve(std::min<std::size_t>(
        maximum_segments, grid.nx + grid.ny + grid.nz + 8));
    long long i = static_cast<long long>(top_i);
    long long j = static_cast<long long>(top_j);
    long long k = static_cast<long long>(grid.nz - 1);
    double x = (static_cast<double>(top_i) + 0.5) * grid.dx_m;
    double y = (static_cast<double>(top_j) + 0.5) * grid.dy_m;
    double z = grid.z_interface_m[grid.nz];
    DirectBeamRayExit exit = DirectBeamRayExit::Surface;

    const auto horizontal_face_distance = [](
        double coordinate, double velocity, double spacing, long long cell)
    {
        if (velocity == 0.0)
            return std::numeric_limits<double>::infinity();
        const double face = static_cast<double>(
            velocity > 0.0 ? cell + 1 : cell) * spacing;
        const double distance = (face - coordinate) / velocity;
        const double scale = std::max(
            {1.0, std::abs(face), std::abs(coordinate)});
        if (!std::isfinite(distance)
            || distance < -64.0 * std::numeric_limits<double>::epsilon()
                * scale / std::abs(velocity))
        {
            throw std::runtime_error(
                "rectilinear direct-beam traversal moved behind a face");
        }
        return std::max(0.0, distance);
    };

    bool finished = false;
    for (std::size_t iteration = 0;
         iteration < maximum_segments && !finished; ++iteration)
    {
        const double distance_x = horizontal_face_distance(
            x, velocity_x, grid.dx_m, i);
        const double distance_y = horizontal_face_distance(
            y, velocity_y, grid.dy_m, j);
        const double lower_z = grid.z_interface_m[static_cast<std::size_t>(k)];
        const double distance_z = (lower_z - z) / velocity_z;
        if (!std::isfinite(distance_z) || distance_z < 0.0)
            throw std::runtime_error(
                "rectilinear direct-beam traversal moved below an interface");
        const double path_length = std::min(
            {distance_x, distance_y, distance_z});
        if (!std::isfinite(path_length) || path_length < 0.0)
            throw std::runtime_error(
                "rectilinear direct-beam traversal produced an invalid segment");
        const auto is_crossing = [path_length](double distance)
        {
            if (!std::isfinite(distance))
                return false;
            const double crossing_scale = std::max(
                {1.0, std::abs(distance), std::abs(path_length)});
            return std::abs(distance - path_length)
                <= 128.0 * std::numeric_limits<double>::epsilon()
                    * crossing_scale;
        };
        const bool crosses_x = is_crossing(distance_x);
        const bool crosses_y = is_crossing(distance_y);
        const bool crosses_z = is_crossing(distance_z);
        if (path_length > 0.0)
        {
            const std::size_t cell_index =
                (static_cast<std::size_t>(i) * grid.ny
                 + static_cast<std::size_t>(j)) * grid.nz
                + static_cast<std::size_t>(k);
            candidate_segments.push_back({
                cell_index,
                path_length,
                grid.vertical_layer_optical_depth[cell_index],
                grid.z_interface_m[static_cast<std::size_t>(k + 1)]
                    - grid.z_interface_m[static_cast<std::size_t>(k)]});
            x += velocity_x * path_length;
            y += velocity_y * path_length;
            z += velocity_z * path_length;
        }
        if (!crosses_x && !crosses_y && !crosses_z)
            throw std::runtime_error(
                "rectilinear direct-beam traversal did not cross a face");

        if (crosses_x)
        {
            i += velocity_x > 0.0 ? 1 : -1;
            if (i < 0 || i >= static_cast<long long>(grid.nx))
            {
                if (grid.x_boundary == TerrainShadowBoundary::Open)
                {
                    exit = DirectBeamRayExit::OpenLateralBoundary;
                    finished = true;
                }
                else
                {
                    i = i < 0 ? static_cast<long long>(grid.nx - 1) : 0;
                    x = i == 0
                        ? 0.0 : static_cast<double>(grid.nx) * grid.dx_m;
                }
            }
            else
            {
                x = velocity_x > 0.0
                    ? static_cast<double>(i) * grid.dx_m
                    : static_cast<double>(i + 1) * grid.dx_m;
            }
        }
        if (crosses_y && !finished)
        {
            j += velocity_y > 0.0 ? 1 : -1;
            if (j < 0 || j >= static_cast<long long>(grid.ny))
            {
                if (grid.y_boundary == TerrainShadowBoundary::Open)
                {
                    exit = DirectBeamRayExit::OpenLateralBoundary;
                    finished = true;
                }
                else
                {
                    j = j < 0 ? static_cast<long long>(grid.ny - 1) : 0;
                    y = j == 0
                        ? 0.0 : static_cast<double>(grid.ny) * grid.dy_m;
                }
            }
            else
            {
                y = velocity_y > 0.0
                    ? static_cast<double>(j) * grid.dy_m
                    : static_cast<double>(j + 1) * grid.dy_m;
            }
        }
        if (crosses_z && !finished)
        {
            --k;
            if (k < 0)
            {
                exit = DirectBeamRayExit::Surface;
                finished = true;
                z = grid.z_interface_m[0];
            }
            else
            {
                z = grid.z_interface_m[static_cast<std::size_t>(k + 1)];
            }
        }
    }
    if (!finished || candidate_segments.size() > maximum_segments)
        throw std::runtime_error(
            "rectilinear direct-beam traversal exceeded its face bound");

    std::vector<DirectBeamSegmentAbsorption> candidate_absorption;
    const DirectBeamPathResult energy =
        attenuate_direct_beam_path_reference(
            incident_power_w, candidate_segments, candidate_absorption);
    DirectBeamRayResult result;
    result.energy = energy;
    result.exit = exit;
    result.terminal_i = static_cast<std::size_t>(std::clamp<long long>(
        i, 0, static_cast<long long>(grid.nx - 1)));
    result.terminal_j = static_cast<std::size_t>(std::clamp<long long>(
        j, 0, static_cast<long long>(grid.ny - 1)));
    result.terminal_x_m = x;
    result.terminal_y_m = y;
    result.terminal_z_m = z;
    ordered_segments.swap(candidate_segments);
    segment_absorption.swap(candidate_absorption);
    return result;
}

namespace
{
struct MappedAxisPatch
{
    std::size_t index0 = 0;
    std::size_t index1 = 0;
    double lower_m = 0.0;
    double upper_m = 0.0;
};

MappedAxisPatch mapped_axis_patch(double coordinate,
                                  double velocity,
                                  double spacing,
                                  std::size_t count,
                                  TerrainShadowBoundary boundary)
{
    const double extent = static_cast<double>(count) * spacing;
    if (boundary == TerrainShadowBoundary::Open)
    {
        if (count == 1)
            return {0, 0, 0.0, extent};
        const double first = 0.5 * spacing;
        const double last = (static_cast<double>(count) - 0.5) * spacing;
        const double tolerance = 64.0
            * std::numeric_limits<double>::epsilon()
            * std::max({1.0, std::abs(coordinate), extent});
        if (coordinate < first - tolerance
            || (std::abs(coordinate - first) <= tolerance && velocity < 0.0))
            return {0, 0, 0.0, first};
        if (coordinate > last + tolerance
            || (std::abs(coordinate - last) <= tolerance && velocity > 0.0))
            return {count - 1, count - 1, last, extent};
    }

    const double scalar_coordinate = coordinate / spacing - 0.5;
    const double nearest = std::round(scalar_coordinate);
    const double tolerance = 64.0
        * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::abs(scalar_coordinate));
    long long lower = static_cast<long long>(std::floor(scalar_coordinate));
    if (velocity < 0.0
        && std::abs(scalar_coordinate - nearest) <= tolerance)
        lower = static_cast<long long>(nearest) - 1;
    const auto resolve = [&](long long index)
    {
        if (boundary == TerrainShadowBoundary::Periodic)
            return wrapped_index(index, count);
        return static_cast<std::size_t>(std::clamp<long long>(
            index, 0, static_cast<long long>(count - 1)));
    };
    return {
        resolve(lower), resolve(lower + 1),
        (static_cast<double>(lower) + 0.5) * spacing,
        (static_cast<double>(lower) + 1.5) * spacing};
}

long long mapped_cell_ahead(double coordinate,
                            double velocity,
                            double spacing)
{
    const double face_coordinate = coordinate / spacing;
    const double nearest = std::round(face_coordinate);
    const double tolerance = 64.0
        * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::abs(face_coordinate));
    long long cell = static_cast<long long>(std::floor(face_coordinate));
    if (velocity < 0.0
        && std::abs(face_coordinate - nearest) <= tolerance)
        cell = static_cast<long long>(nearest) - 1;
    return cell;
}

double distance_to_axis_upper_or_lower(double coordinate,
                                       double velocity,
                                       double lower,
                                       double upper)
{
    if (velocity == 0.0)
        return std::numeric_limits<double>::infinity();
    const double face = velocity > 0.0 ? upper : lower;
    const double distance = (face - coordinate) / velocity;
    const double tolerance = 64.0
        * std::numeric_limits<double>::epsilon()
        * std::max({1.0, std::abs(face), std::abs(coordinate)})
        / std::abs(velocity);
    if (!std::isfinite(distance) || distance < -tolerance)
        throw std::runtime_error(
            "mapped direct-beam traversal moved behind a horizontal boundary");
    return std::max(0.0, distance);
}
} // namespace

namespace
{
struct ValidatedMappedDirectBeamContext
{
    double maximum_top_m = 0.0;
    double minimum_ground_m = 0.0;
    std::size_t maximum_iterations = 0;
};

ValidatedMappedDirectBeamContext validate_mapped_direct_beam_context(
    const MappedDirectBeamGridView& grid,
    const IdealizedSolarGeometry& sun_direction)
{
    if (grid.vertical_layer_optical_depth == nullptr
        || grid.z_interface_m == nullptr
        || grid.nx == 0 || grid.ny == 0 || grid.nz == 0
        || grid.nz == std::numeric_limits<std::size_t>::max()
        || grid.nx > std::numeric_limits<std::size_t>::max() / grid.ny
        || grid.nx * grid.ny
            > std::numeric_limits<std::size_t>::max() / grid.nz
        || grid.nx * grid.ny
            > std::numeric_limits<std::size_t>::max() / (grid.nz + 1)
        || grid.nx > static_cast<std::size_t>(
            std::numeric_limits<long long>::max())
        || grid.ny > static_cast<std::size_t>(
            std::numeric_limits<long long>::max())
        || !std::isfinite(grid.dx_m) || grid.dx_m <= 0.0
        || !std::isfinite(grid.dy_m) || grid.dy_m <= 0.0
        || (grid.x_boundary != TerrainShadowBoundary::Open
            && grid.x_boundary != TerrainShadowBoundary::Periodic)
        || (grid.y_boundary != TerrainShadowBoundary::Open
            && grid.y_boundary != TerrainShadowBoundary::Periodic))
        throw std::invalid_argument(
            "mapped direct-beam context has invalid grid geometry");
    const double sun_norm = std::hypot(
        std::hypot(sun_direction.east, sun_direction.north),
        sun_direction.up);
    if (!std::isfinite(sun_direction.east)
        || !std::isfinite(sun_direction.north)
        || !std::isfinite(sun_direction.up)
        || !std::isfinite(sun_norm)
        || std::abs(sun_norm - 1.0) > 1.0e-12
        || sun_direction.up <= 0.0)
        throw std::invalid_argument(
            "mapped direct-beam context requires a daytime unit direction");
    const long double domain_x_m =
        static_cast<long double>(grid.nx) * grid.dx_m;
    const long double domain_y_m =
        static_cast<long double>(grid.ny) * grid.dy_m;
    if (!std::isfinite(domain_x_m) || !std::isfinite(domain_y_m)
        || domain_x_m
            > static_cast<long double>(std::numeric_limits<double>::max())
        || domain_y_m
            > static_cast<long double>(std::numeric_limits<double>::max()))
        throw std::overflow_error(
            "mapped direct-beam horizontal extent overflowed");

    const std::size_t column_count = grid.nx * grid.ny;
    const std::size_t interface_stride = grid.nz + 1;
    ValidatedMappedDirectBeamContext context;
    context.maximum_top_m = -std::numeric_limits<double>::infinity();
    context.minimum_ground_m = std::numeric_limits<double>::infinity();
    for (std::size_t column = 0; column < column_count; ++column)
    {
        const std::size_t base = column * interface_stride;
        for (std::size_t k = 0; k <= grid.nz; ++k)
        {
            const double height = grid.z_interface_m[base + k];
            if (!std::isfinite(height)
                || (k > 0
                    && (height <= grid.z_interface_m[base + k - 1]
                        || static_cast<long double>(height)
                                - static_cast<long double>(
                                    grid.z_interface_m[base + k - 1])
                            > static_cast<long double>(
                                std::numeric_limits<double>::max()))))
                throw std::invalid_argument(
                    "mapped direct-beam interfaces must increase in every column");
        }
        context.minimum_ground_m = std::min(
            context.minimum_ground_m, grid.z_interface_m[base]);
        const double top = grid.z_interface_m[base + grid.nz];
        context.maximum_top_m = std::max(context.maximum_top_m, top);
    }
    for (std::size_t index = 0;
         index < column_count * grid.nz; ++index)
        if (!std::isfinite(grid.vertical_layer_optical_depth[index])
            || grid.vertical_layer_optical_depth[index] < 0.0)
            throw std::invalid_argument(
                "mapped direct-beam optical depth must be finite and non-negative");

    const IdealizedSolarGeometry photon{
        -sun_direction.east, -sun_direction.north, -sun_direction.up};
    const long double maximum_path =
        (static_cast<long double>(context.maximum_top_m)
         - static_cast<long double>(context.minimum_ground_m))
        / static_cast<long double>(sun_direction.up);
    const long double horizontal_events =
        2.0L * std::ceil(
            std::abs(static_cast<long double>(photon.east))
            * maximum_path / grid.dx_m)
        + 2.0L * std::ceil(
            std::abs(static_cast<long double>(photon.north))
            * maximum_path / grid.dy_m) + 4.0L;
    const long double iteration_bound = (horizontal_events + 1.0L)
        * (2.0L * static_cast<long double>(grid.nz) + 3.0L) + 16.0L;
    if (!std::isfinite(maximum_path) || maximum_path <= 0.0L
        || maximum_path
            > static_cast<long double>(std::numeric_limits<double>::max())
        || !std::isfinite(iteration_bound)
        || iteration_bound
            > static_cast<long double>(
                std::numeric_limits<std::size_t>::max()))
        throw std::overflow_error(
            "mapped direct-beam traversal bound overflowed");
    context.maximum_iterations = static_cast<std::size_t>(iteration_bound);
    return context;
}

std::size_t mapped_radiative_ray_iteration_bound(
    const MappedDirectBeamGridView& grid,
    const ValidatedMappedDirectBeamContext& context,
    double start_x_m,
    double start_y_m,
    double start_z_m,
    const IdealizedSolarGeometry& ray_direction)
{
    long double maximum_path =
        std::numeric_limits<long double>::infinity();
    if (ray_direction.up > 0.0)
        maximum_path =
            (static_cast<long double>(context.maximum_top_m) - start_z_m)
            / ray_direction.up;
    else if (ray_direction.up < 0.0)
        maximum_path =
            (static_cast<long double>(start_z_m) - context.minimum_ground_m)
            / -ray_direction.up;
    else
    {
        const long double x_extent =
            static_cast<long double>(grid.nx) * grid.dx_m;
        const long double y_extent =
            static_cast<long double>(grid.ny) * grid.dy_m;
        if (grid.x_boundary == TerrainShadowBoundary::Open
            && ray_direction.east != 0.0)
            maximum_path = std::min(maximum_path,
                (ray_direction.east > 0.0
                    ? x_extent - start_x_m : start_x_m)
                / std::abs(ray_direction.east));
        if (grid.y_boundary == TerrainShadowBoundary::Open
            && ray_direction.north != 0.0)
            maximum_path = std::min(maximum_path,
                (ray_direction.north > 0.0
                    ? y_extent - start_y_m : start_y_m)
                / std::abs(ray_direction.north));
        if (!std::isfinite(maximum_path))
            throw std::invalid_argument(
                "horizontal mapped ray has no exit in a periodic domain");
    }
    const long double path_tolerance = 512.0L
        * std::numeric_limits<double>::epsilon()
        * std::max({1.0L,
            std::abs(static_cast<long double>(context.maximum_top_m)),
            std::abs(static_cast<long double>(context.minimum_ground_m)),
            std::abs(static_cast<long double>(start_z_m))});
    if (maximum_path < 0.0L && maximum_path >= -path_tolerance)
        maximum_path = 0.0L;
    if (!std::isfinite(maximum_path) || maximum_path < 0.0L
        || maximum_path
            > static_cast<long double>(std::numeric_limits<double>::max()))
        throw std::overflow_error(
            "mapped surface ray path bound overflowed");
    const long double horizontal_events =
        2.0L * std::ceil(
            std::abs(static_cast<long double>(ray_direction.east))
            * maximum_path / grid.dx_m)
        + 2.0L * std::ceil(
            std::abs(static_cast<long double>(ray_direction.north))
            * maximum_path / grid.dy_m) + 4.0L;
    const long double iteration_bound = (horizontal_events + 1.0L)
        * (2.0L * static_cast<long double>(grid.nz) + 3.0L) + 16.0L;
    if (!std::isfinite(iteration_bound)
        || iteration_bound
            > static_cast<long double>(
                std::numeric_limits<std::size_t>::max()))
        throw std::overflow_error(
            "mapped surface ray event bound overflowed");
    return static_cast<std::size_t>(iteration_bound);
}

void gauss_legendre_unit_interval(
    std::size_t count,
    std::vector<double>& nodes,
    std::vector<double>& weights)
{
    if (count == 0 || count > 64)
        throw std::invalid_argument(
            "Lambertian cosine quadrature requires 1..64 nodes");
    nodes.resize(count);
    weights.resize(count);
    const std::size_t half = (count + 1) / 2;
    for (std::size_t root = 0; root < half; ++root)
    {
        double x = std::cos(radiation_constants::pi
            * (static_cast<double>(root) + 0.75)
            / (static_cast<double>(count) + 0.5));
        double derivative = 0.0;
        bool converged = false;
        for (std::size_t iteration = 0; iteration < 64; ++iteration)
        {
            double p_previous = 1.0;
            double p = x;
            for (std::size_t degree = 2; degree <= count; ++degree)
            {
                const double p_next = (
                    (2.0 * static_cast<double>(degree) - 1.0) * x * p
                    - (static_cast<double>(degree) - 1.0) * p_previous)
                    / static_cast<double>(degree);
                p_previous = p;
                p = p_next;
            }
            derivative = static_cast<double>(count)
                * (x * p - p_previous) / (x * x - 1.0);
            const double next = x - p / derivative;
            if (std::abs(next - x)
                <= 8.0 * std::numeric_limits<double>::epsilon()
                    * std::max(1.0, std::abs(next)))
            {
                x = next;
                converged = true;
                break;
            }
            x = next;
        }
        if (!converged || !std::isfinite(x) || !std::isfinite(derivative))
            throw std::runtime_error(
                "Lambertian cosine quadrature did not converge");
        // Re-evaluate the derivative at the committed root.
        double p_previous = 1.0;
        double p = x;
        for (std::size_t degree = 2; degree <= count; ++degree)
        {
            const double p_next = (
                (2.0 * static_cast<double>(degree) - 1.0) * x * p
                - (static_cast<double>(degree) - 1.0) * p_previous)
                / static_cast<double>(degree);
            p_previous = p;
            p = p_next;
        }
        derivative = static_cast<double>(count)
            * (x * p - p_previous) / (x * x - 1.0);
        const double weight = 1.0
            / ((1.0 - x * x) * derivative * derivative);
        const std::size_t lower = root;
        const std::size_t upper = count - 1 - root;
        nodes[lower] = 0.5 * (1.0 - x);
        nodes[upper] = 0.5 * (1.0 + x);
        weights[lower] = weight;
        weights[upper] = weight;
    }
}

DirectBeamRayResult trace_mapped_direct_beam_from_entry_reference_impl(
    const MappedDirectBeamGridView& grid,
    double start_x_m,
    double start_y_m,
    double start_z_m,
    std::size_t initial_k,
    const IdealizedSolarGeometry& photon,
    double incident_power_w,
    std::vector<DirectBeamPathSegment>& ordered_segments,
    std::vector<DirectBeamSegmentAbsorption>& segment_absorption,
    const ValidatedMappedDirectBeamContext& validated_context,
    bool permit_level_top_exit,
    bool reuse_output_storage = false)
{
    if (initial_k >= grid.nz
        || !std::isfinite(start_x_m) || !std::isfinite(start_y_m)
        || !std::isfinite(start_z_m)
        || !std::isfinite(incident_power_w) || incident_power_w < 0.0)
    {
        throw std::invalid_argument(
            "mapped direct-beam ray has invalid grid, entry, or power");
    }
    const long double domain_x_m =
        static_cast<long double>(grid.nx) * grid.dx_m;
    const long double domain_y_m =
        static_cast<long double>(grid.ny) * grid.dy_m;
    const std::size_t interface_stride = grid.nz + 1;
    const double maximum_top = validated_context.maximum_top_m;
    const double minimum_ground = validated_context.minimum_ground_m;
    const std::size_t maximum_iterations =
        validated_context.maximum_iterations;

    const double x_extent_m = static_cast<double>(domain_x_m);
    const double y_extent_m = static_cast<double>(domain_y_m);
    const double entry_tolerance = 128.0
        * std::numeric_limits<double>::epsilon()
        * std::max({1.0, x_extent_m, y_extent_m,
                    std::abs(start_x_m), std::abs(start_y_m)});
    if (start_x_m < -entry_tolerance
        || start_x_m > x_extent_m + entry_tolerance
        || start_y_m < -entry_tolerance
        || start_y_m > y_extent_m + entry_tolerance)
        throw std::invalid_argument(
            "mapped direct-beam entry lies outside the horizontal domain");
    double x = std::clamp(start_x_m, 0.0, x_extent_m);
    double y = std::clamp(start_y_m, 0.0, y_extent_m);
    double z = start_z_m;
    long long k = static_cast<long long>(initial_k);
    std::vector<DirectBeamPathSegment> local_candidate_segments;
    std::vector<DirectBeamPathSegment>& candidate_segments =
        reuse_output_storage ? ordered_segments : local_candidate_segments;
    if (reuse_output_storage)
        candidate_segments.clear();
    DirectBeamRayExit exit = DirectBeamRayExit::Surface;
    std::size_t terminal_i = 0;
    std::size_t terminal_j = 0;
    bool finished = false;

    const auto local_cell = [](long long cell, std::size_t count,
                               TerrainShadowBoundary boundary)
    {
        if (boundary == TerrainShadowBoundary::Periodic)
            return wrapped_index(cell, count);
        if (cell < 0 || cell >= static_cast<long long>(count))
            throw std::out_of_range(
                "mapped direct-beam ray left an open horizontal axis");
        return static_cast<std::size_t>(cell);
    };
    const auto interface_value = [&](std::size_t i, std::size_t j,
                                     std::size_t k_face)
    {
        return grid.z_interface_m[
            (i * grid.ny + j) * interface_stride + k_face];
    };

    for (std::size_t iteration = 0;
         iteration < maximum_iterations && !finished; ++iteration)
    {
        const double vertical_boundary_tolerance = 512.0
            * std::numeric_limits<double>::epsilon()
            * std::max({1.0, std::abs(maximum_top),
                        std::abs(minimum_ground), std::abs(z)});
        if (photon.up > 0.0
            && z >= maximum_top - vertical_boundary_tolerance)
        {
            if (!permit_level_top_exit)
                throw std::runtime_error(
                    "mapped downward direct beam escaped through the top boundary");
            z = maximum_top;
            exit = DirectBeamRayExit::LevelTop;
            finished = true;
            break;
        }
        const long long unwrapped_i = mapped_cell_ahead(
            x, photon.east, grid.dx_m);
        const long long unwrapped_j = mapped_cell_ahead(
            y, photon.north, grid.dy_m);
        if ((grid.x_boundary == TerrainShadowBoundary::Open
                && (unwrapped_i < 0
                    || unwrapped_i >= static_cast<long long>(grid.nx)))
            || (grid.y_boundary == TerrainShadowBoundary::Open
                && (unwrapped_j < 0
                    || unwrapped_j >= static_cast<long long>(grid.ny))))
        {
            exit = DirectBeamRayExit::OpenLateralBoundary;
            finished = true;
            break;
        }
        const std::size_t i = local_cell(
            unwrapped_i, grid.nx, grid.x_boundary);
        const std::size_t j = local_cell(
            unwrapped_j, grid.ny, grid.y_boundary);
        terminal_i = i;
        terminal_j = j;
        const MappedAxisPatch x_patch = mapped_axis_patch(
            x, photon.east, grid.dx_m, grid.nx, grid.x_boundary);
        const MappedAxisPatch y_patch = mapped_axis_patch(
            y, photon.north, grid.dy_m, grid.ny, grid.y_boundary);
        const double x_cell_lower = static_cast<double>(unwrapped_i) * grid.dx_m;
        const double y_cell_lower = static_cast<double>(unwrapped_j) * grid.dy_m;
        double remaining_vertical_limit =
            std::numeric_limits<double>::infinity();
        if (photon.up > 0.0)
            remaining_vertical_limit = (maximum_top - z) / photon.up;
        else if (photon.up < 0.0)
            remaining_vertical_limit = (z - minimum_ground) / -photon.up;
        const double vertical_limit_tolerance = 256.0
            * std::numeric_limits<double>::epsilon()
            * std::max({1.0, std::abs(maximum_top),
                        std::abs(minimum_ground), std::abs(z)});
        if (remaining_vertical_limit < 0.0
            && remaining_vertical_limit >= -vertical_limit_tolerance)
            remaining_vertical_limit = 0.0;
        const double distance_x_cell = distance_to_axis_upper_or_lower(
            x, photon.east, x_cell_lower, x_cell_lower + grid.dx_m);
        const double distance_y_cell = distance_to_axis_upper_or_lower(
            y, photon.north, y_cell_lower, y_cell_lower + grid.dy_m);
        const double distance_x_patch = distance_to_axis_upper_or_lower(
            x, photon.east, x_patch.lower_m, x_patch.upper_m);
        const double distance_y_patch = distance_to_axis_upper_or_lower(
            y, photon.north, y_patch.lower_m, y_patch.upper_m);
        const double horizontal_limit = std::min({
            distance_x_cell, distance_y_cell,
            distance_x_patch, distance_y_patch,
            remaining_vertical_limit});
        if (!std::isfinite(horizontal_limit) || horizontal_limit <= 0.0)
            throw std::runtime_error(
                "mapped direct-beam traversal failed to advance horizontally");

        const auto make_patch = [&](std::size_t k_face)
        {
            return BilinearMappedInterfacePatch{
                interface_value(x_patch.index0, y_patch.index0, k_face),
                interface_value(x_patch.index1, y_patch.index0, k_face),
                interface_value(x_patch.index0, y_patch.index1, k_face),
                interface_value(x_patch.index1, y_patch.index1, k_face),
                x_patch.lower_m, y_patch.lower_m,
                x_patch.upper_m - x_patch.lower_m,
                y_patch.upper_m - y_patch.lower_m};
        };
        const auto lower_crossing =
            first_bilinear_mapped_interface_intersection_reference(
                make_patch(static_cast<std::size_t>(k)),
                x, y, z, photon, horizontal_limit,
                MappedInterfaceCrossing::FromAbove);
        const auto upper_crossing =
            first_bilinear_mapped_interface_intersection_reference(
                make_patch(static_cast<std::size_t>(k + 1)),
                x, y, z, photon, horizontal_limit,
                MappedInterfaceCrossing::FromBelow);
        const double lower_distance = lower_crossing.value_or(
            std::numeric_limits<double>::infinity());
        const double upper_distance = upper_crossing.value_or(
            std::numeric_limits<double>::infinity());
        const double path_length = std::min(
            {horizontal_limit, lower_distance, upper_distance});
        const double path_tolerance = 256.0
            * std::numeric_limits<double>::epsilon()
            * std::max(1.0, horizontal_limit);
        const auto is_path_event = [&](double distance)
        {
            return std::isfinite(distance)
                && std::abs(distance - path_length) <= path_tolerance;
        };
        const bool crosses_lower =
            std::isfinite(lower_distance)
            && std::abs(lower_distance - path_length) <= path_tolerance;
        const bool crosses_upper =
            std::isfinite(upper_distance)
            && std::abs(upper_distance - path_length) <= path_tolerance;
        if (crosses_lower && crosses_upper)
            throw std::runtime_error(
                "mapped direct-beam ray crossed both layer interfaces simultaneously");

        const bool reaches_x_cell = is_path_event(distance_x_cell);
        const bool reaches_y_cell = is_path_event(distance_y_cell);
        const bool reaches_x_patch = is_path_event(distance_x_patch);
        const bool reaches_y_patch = is_path_event(distance_y_patch);
        if (path_length > path_tolerance)
        {
            const std::size_t cell_index =
                (i * grid.ny + j) * grid.nz + static_cast<std::size_t>(k);
            const double thickness = interface_value(i, j,
                static_cast<std::size_t>(k + 1))
                - interface_value(i, j, static_cast<std::size_t>(k));
            if (!candidate_segments.empty()
                && candidate_segments.back().cell_index == cell_index
                && candidate_segments.back().vertical_layer_optical_depth
                    == grid.vertical_layer_optical_depth[cell_index]
                && candidate_segments.back().physical_layer_thickness_m
                    == thickness)
                candidate_segments.back().path_length_m += path_length;
            else
                candidate_segments.push_back({
                    cell_index, path_length,
                    grid.vertical_layer_optical_depth[cell_index], thickness});
            x += photon.east * path_length;
            y += photon.north * path_length;
            z += photon.up * path_length;
            if (!crosses_lower && !crosses_upper)
            {
                if (reaches_x_cell)
                    x = photon.east > 0.0
                        ? x_cell_lower + grid.dx_m : x_cell_lower;
                if (reaches_y_cell)
                    y = photon.north > 0.0
                        ? y_cell_lower + grid.dy_m : y_cell_lower;
                if (reaches_x_patch)
                    x = photon.east > 0.0
                        ? x_patch.upper_m : x_patch.lower_m;
                if (reaches_y_patch)
                    y = photon.north > 0.0
                        ? y_patch.upper_m : y_patch.lower_m;
            }
        }
        else if (!crosses_lower && !crosses_upper)
        {
            if (reaches_x_cell)
                x = photon.east > 0.0
                    ? x_cell_lower + grid.dx_m : x_cell_lower;
            if (reaches_y_cell)
                y = photon.north > 0.0
                    ? y_cell_lower + grid.dy_m : y_cell_lower;
            if (reaches_x_patch)
                x = photon.east > 0.0
                    ? x_patch.upper_m : x_patch.lower_m;
            if (reaches_y_patch)
                y = photon.north > 0.0
                    ? y_patch.upper_m : y_patch.lower_m;
            if (!reaches_x_cell && !reaches_y_cell
                && !reaches_x_patch && !reaches_y_patch)
                throw std::runtime_error(
                    "mapped direct-beam traversal produced a zero-length event");
            continue;
        }

        if (crosses_lower)
        {
            --k;
            if (k < 0)
            {
                exit = DirectBeamRayExit::Surface;
                finished = true;
            }
            continue;
        }
        if (crosses_upper)
        {
            ++k;
            if (k >= static_cast<long long>(grid.nz))
            {
                if (!permit_level_top_exit)
                    throw std::runtime_error(
                        "mapped downward direct beam escaped through the top boundary");
                exit = DirectBeamRayExit::LevelTop;
                finished = true;
            }
            continue;
        }

        const double x_extent = static_cast<double>(grid.nx) * grid.dx_m;
        const double y_extent = static_cast<double>(grid.ny) * grid.dy_m;
        const double boundary_tolerance = 256.0
            * std::numeric_limits<double>::epsilon()
            * std::max({1.0, std::abs(x), std::abs(y), x_extent, y_extent});
        if ((grid.x_boundary == TerrainShadowBoundary::Open
                && ((photon.east < 0.0 && x <= boundary_tolerance)
                    || (photon.east > 0.0
                        && x >= x_extent - boundary_tolerance)))
            || (grid.y_boundary == TerrainShadowBoundary::Open
                && ((photon.north < 0.0 && y <= boundary_tolerance)
                    || (photon.north > 0.0
                        && y >= y_extent - boundary_tolerance))))
        {
            exit = DirectBeamRayExit::OpenLateralBoundary;
            finished = true;
        }
    }
    if (!finished)
        throw std::runtime_error(
            "mapped direct-beam traversal exceeded its conservative event bound");

    std::vector<DirectBeamSegmentAbsorption> local_candidate_absorption;
    std::vector<DirectBeamSegmentAbsorption>& candidate_absorption =
        reuse_output_storage ? segment_absorption
                             : local_candidate_absorption;
    const DirectBeamPathResult energy = attenuate_direct_beam_path_reference(
        incident_power_w, candidate_segments, candidate_absorption);
    const auto wrap_coordinate = [](double coordinate, double extent)
    {
        double wrapped = std::fmod(coordinate, extent);
        if (wrapped < 0.0)
            wrapped += extent;
        return wrapped;
    };
    DirectBeamRayResult result;
    result.energy = energy;
    result.exit = exit;
    result.terminal_i = terminal_i;
    result.terminal_j = terminal_j;
    result.terminal_x_m = grid.x_boundary == TerrainShadowBoundary::Periodic
        ? wrap_coordinate(x, static_cast<double>(grid.nx) * grid.dx_m) : x;
    result.terminal_y_m = grid.y_boundary == TerrainShadowBoundary::Periodic
        ? wrap_coordinate(y, static_cast<double>(grid.ny) * grid.dy_m) : y;
    result.terminal_z_m = z;
    if (!reuse_output_storage)
    {
        ordered_segments.swap(candidate_segments);
        segment_absorption.swap(candidate_absorption);
    }
    return result;
}
} // namespace

DirectBeamRayResult trace_mapped_direct_beam_from_top_reference(
    const MappedDirectBeamGridView& grid,
    std::size_t top_i,
    std::size_t top_j,
    const IdealizedSolarGeometry& sun_direction,
    double incident_power_w,
    std::vector<DirectBeamPathSegment>& ordered_segments,
    std::vector<DirectBeamSegmentAbsorption>& segment_absorption)
{
    if (grid.z_interface_m == nullptr || top_i >= grid.nx || top_j >= grid.ny
        || grid.nz == 0 || grid.nz == std::numeric_limits<std::size_t>::max()
        || grid.ny > std::numeric_limits<std::size_t>::max() / (grid.nz + 1)
        || top_i > std::numeric_limits<std::size_t>::max() / grid.ny)
        throw std::invalid_argument(
            "mapped direct-beam top entry is outside the grid");
    const std::size_t column = top_i * grid.ny + top_j;
    if (column > std::numeric_limits<std::size_t>::max() / (grid.nz + 1))
        throw std::overflow_error(
            "mapped direct-beam top entry index overflowed");
    const double top = grid.z_interface_m[
        column * (grid.nz + 1) + grid.nz];
    const ValidatedMappedDirectBeamContext validated_context =
        validate_mapped_direct_beam_context(grid, sun_direction);
    const IdealizedSolarGeometry photon{
        -sun_direction.east, -sun_direction.north, -sun_direction.up};
    return trace_mapped_direct_beam_from_entry_reference_impl(
        grid,
        (static_cast<double>(top_i) + 0.5) * grid.dx_m,
        (static_cast<double>(top_j) + 0.5) * grid.dy_m,
        top, grid.nz - 1, photon, incident_power_w,
        ordered_segments, segment_absorption,
        validated_context, false, false);
}

DirectBeamRayResult trace_mapped_radiative_ray_from_surface_reference(
    const MappedDirectBeamGridView& grid,
    std::size_t surface_i,
    std::size_t surface_j,
    const IdealizedSolarGeometry& ray_direction,
    double incident_power_w,
    std::vector<DirectBeamPathSegment>& ordered_segments,
    std::vector<DirectBeamSegmentAbsorption>& segment_absorption)
{
    if (grid.z_interface_m == nullptr
        || surface_i >= grid.nx || surface_j >= grid.ny
        || grid.nz == 0 || grid.nz == std::numeric_limits<std::size_t>::max()
        || grid.ny > std::numeric_limits<std::size_t>::max() / (grid.nz + 1)
        || surface_i > std::numeric_limits<std::size_t>::max() / grid.ny
        || !std::isfinite(ray_direction.east)
        || !std::isfinite(ray_direction.north)
        || !std::isfinite(ray_direction.up))
        throw std::invalid_argument(
            "mapped surface ray has invalid source or direction");
    const double direction_norm = std::hypot(
        std::hypot(ray_direction.east, ray_direction.north),
        ray_direction.up);
    if (!std::isfinite(direction_norm)
        || std::abs(direction_norm - 1.0) > 1.0e-12)
        throw std::invalid_argument(
            "mapped surface ray requires a unit direction");

    // The vertical direction is used only to obtain the once-validated mapped
    // grid state. The arbitrary ray receives its own conservative event bound.
    ValidatedMappedDirectBeamContext context =
        validate_mapped_direct_beam_context(grid, {0.0, 0.0, 1.0});
    const std::size_t column = surface_i * grid.ny + surface_j;
    if (column > std::numeric_limits<std::size_t>::max() / (grid.nz + 1))
        throw std::overflow_error(
            "mapped surface ray source index overflowed");
    const double ground = grid.z_interface_m[column * (grid.nz + 1)];
    const double x = (static_cast<double>(surface_i) + 0.5) * grid.dx_m;
    const double y = (static_cast<double>(surface_j) + 0.5) * grid.dy_m;

    context.maximum_iterations = mapped_radiative_ray_iteration_bound(
        grid, context, x, y, ground, ray_direction);

    return trace_mapped_direct_beam_from_entry_reference_impl(
        grid, x, y, ground, 0, ray_direction, incident_power_w,
        ordered_segments, segment_absorption,
        context, true, false);
}

DirectBeamRayResult trace_mapped_radiative_ray_from_cell_reference(
    const MappedDirectBeamGridView& grid,
    std::size_t cell_i,
    std::size_t cell_j,
    std::size_t cell_k,
    const IdealizedSolarGeometry& ray_direction,
    double incident_power_w,
    std::vector<DirectBeamPathSegment>& ordered_segments,
    std::vector<DirectBeamSegmentAbsorption>& segment_absorption)
{
    if (grid.z_interface_m == nullptr || cell_i >= grid.nx ||
        cell_j >= grid.ny || cell_k >= grid.nz ||
        !std::isfinite(ray_direction.east) ||
        !std::isfinite(ray_direction.north) ||
        !std::isfinite(ray_direction.up))
        throw std::invalid_argument(
            "mapped cell ray has invalid source or direction");
    const double norm = std::hypot(
        std::hypot(ray_direction.east, ray_direction.north),
        ray_direction.up);
    if (!std::isfinite(norm) || std::abs(norm - 1.0) > 1.0e-12)
        throw std::invalid_argument(
            "mapped cell ray requires a unit direction");
    ValidatedMappedDirectBeamContext context =
        validate_mapped_direct_beam_context(grid, {0.0, 0.0, 1.0});
    const std::size_t column = cell_i * grid.ny + cell_j;
    const std::size_t base = column * (grid.nz + 1);
    const double lower = grid.z_interface_m[base + cell_k];
    const double upper = grid.z_interface_m[base + cell_k + 1];
    const double x = (static_cast<double>(cell_i) + 0.5) * grid.dx_m;
    const double y = (static_cast<double>(cell_j) + 0.5) * grid.dy_m;
    const double z = lower + 0.5 * (upper - lower);
    if (!std::isfinite(z) || !(upper > lower) || !(z > lower) || !(z < upper))
        throw std::invalid_argument(
            "mapped cell ray source has invalid vertical geometry");
    context.maximum_iterations = mapped_radiative_ray_iteration_bound(
        grid, context, x, y, z, ray_direction);
    return trace_mapped_direct_beam_from_entry_reference_impl(
        grid, x, y, z, cell_k, ray_direction, incident_power_w,
        ordered_segments, segment_absorption, context, true, false);
}

void evaluate_mapped_direct_beam_field_reference_impl(
    const MappedDirectBeamGridView& grid,
    const IdealizedSolarGeometry& sun_direction,
    double beam_normal_irradiance_wm2,
    const double* beam_normal_irradiance_wm2_by_column,
    DirectBeamFieldWorkspace& workspace,
    DirectBeamFieldResult& result)
{
    if ((beam_normal_irradiance_wm2_by_column == nullptr &&
         (!std::isfinite(beam_normal_irradiance_wm2) ||
          beam_normal_irradiance_wm2 < 0.0))
        || grid.nz == std::numeric_limits<std::size_t>::max()
        || grid.nx == 0 || grid.ny == 0 || grid.nz == 0
        || grid.nx > std::numeric_limits<std::size_t>::max() / grid.ny
        || grid.nx * grid.ny
            > std::numeric_limits<std::size_t>::max() / grid.nz)
        throw std::invalid_argument(
            "mapped direct-beam field requires finite forcing and grid shape");
    const double sun_norm = std::hypot(
        std::hypot(sun_direction.east, sun_direction.north),
        sun_direction.up);
    if (!std::isfinite(sun_norm)
        || std::abs(sun_norm - 1.0) > 1.0e-12
        || sun_direction.up <= 0.0)
        throw std::invalid_argument(
            "mapped direct-beam field requires a daytime unit sun direction");
    const ValidatedMappedDirectBeamContext validated_context =
        validate_mapped_direct_beam_context(grid, sun_direction);

    const std::size_t column_count = grid.nx * grid.ny;
    const std::size_t cell_count = column_count * grid.nz;
    if (beam_normal_irradiance_wm2_by_column != nullptr)
        for (std::size_t column = 0; column < column_count; ++column)
            if (!std::isfinite(
                    beam_normal_irradiance_wm2_by_column[column]) ||
                beam_normal_irradiance_wm2_by_column[column] < 0.0)
                throw std::invalid_argument(
                    "mapped direct-beam spatial irradiance is invalid");
    const auto boundary_irradiance = [&](std::size_t i, std::size_t j)
    {
        return beam_normal_irradiance_wm2_by_column == nullptr
            ? beam_normal_irradiance_wm2
            : beam_normal_irradiance_wm2_by_column[i * grid.ny + j];
    };
    workspace.scratch_cell_absorbed_power_w.resize(cell_count);
    workspace.scratch_surface_incident_power_w.resize(column_count);
    std::fill(workspace.scratch_cell_absorbed_power_w.begin(),
              workspace.scratch_cell_absorbed_power_w.end(), 0.0);
    std::fill(workspace.scratch_surface_incident_power_w.begin(),
              workspace.scratch_surface_incident_power_w.end(), 0.0);

    struct CompensatedSum
    {
        long double sum = 0.0L;
        long double correction = 0.0L;
        void add(long double value)
        {
            const long double adjusted = value - correction;
            const long double next = sum + adjusted;
            correction = (next - sum) - adjusted;
            sum = next;
        }
    };
    CompensatedSum incoming_top;
    CompensatedSum incoming_side;
    CompensatedSum atmospheric_absorbed;
    CompensatedSum surface_incident;
    CompensatedSum lateral_escape;
    std::uint64_t top_rays = 0;
    std::uint64_t side_rays = 0;
    std::uint64_t surface_rays = 0;
    std::uint64_t escaped_rays = 0;
    const IdealizedSolarGeometry photon{
        -sun_direction.east, -sun_direction.north, -sun_direction.up};
    const std::size_t interface_stride = grid.nz + 1;
    const auto interface_value = [&](std::size_t i, std::size_t j,
                                     std::size_t k_face)
    {
        return grid.z_interface_m[
            (i * grid.ny + j) * interface_stride + k_face];
    };
    const auto commit_ray = [&](
        const DirectBeamRayResult& ray,
        const std::vector<DirectBeamSegmentAbsorption>& absorption,
        double incident_power,
        bool top_boundary)
    {
        if (top_boundary)
        {
            incoming_top.add(incident_power);
            ++top_rays;
        }
        else
        {
            incoming_side.add(incident_power);
            ++side_rays;
        }
        atmospheric_absorbed.add(ray.energy.absorbed_power_w);
        for (const DirectBeamSegmentAbsorption& segment : absorption)
        {
            double& cell_power = workspace.scratch_cell_absorbed_power_w[
                segment.cell_index];
            cell_power += segment.absorbed_power_w;
            if (!std::isfinite(cell_power))
                throw std::overflow_error(
                    "mapped direct-beam cell absorption overflowed");
        }
        if (ray.exit == DirectBeamRayExit::Surface)
        {
            const std::size_t column = ray.terminal_i * grid.ny
                + ray.terminal_j;
            double& surface_power =
                workspace.scratch_surface_incident_power_w[column];
            surface_power += ray.energy.transmitted_power_w;
            if (!std::isfinite(surface_power))
                throw std::overflow_error(
                    "mapped direct-beam surface incidence overflowed");
            surface_incident.add(ray.energy.transmitted_power_w);
            ++surface_rays;
        }
        else
        {
            lateral_escape.add(ray.energy.transmitted_power_w);
            ++escaped_rays;
        }
    };
    const auto process_ray = [&](double x, double y, double z,
                                 std::size_t k,
                                 double incident_power,
                                 bool top_boundary)
    {
        if (!std::isfinite(incident_power) || incident_power < 0.0)
            throw std::overflow_error(
                "mapped direct-beam boundary bundle power overflowed");
        const DirectBeamRayResult ray =
            trace_mapped_direct_beam_from_entry_reference_impl(
                grid, x, y, z, k, photon, incident_power,
                workspace.ray_segments, workspace.ray_absorption,
                validated_context, false, true);
        commit_ray(
            ray, workspace.ray_absorption,
            incident_power, top_boundary);
    };

    const std::size_t top_ray_count = column_count;
    const std::size_t x_side_ray_count =
        grid.x_boundary == TerrainShadowBoundary::Open
            && photon.east != 0.0 ? grid.ny * grid.nz : 0;
    const std::size_t y_side_ray_count =
        grid.y_boundary == TerrainShadowBoundary::Open
            && photon.north != 0.0 ? grid.nx * grid.nz : 0;
    if (top_ray_count > std::numeric_limits<std::size_t>::max()
            - x_side_ray_count
        || top_ray_count + x_side_ray_count
            > std::numeric_limits<std::size_t>::max() - y_side_ray_count)
        throw std::overflow_error(
            "mapped direct-beam boundary ray count overflowed");
    const std::size_t total_ray_count =
        top_ray_count + x_side_ray_count + y_side_ray_count;
    struct RaySpecification
    {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        std::size_t k = 0;
        double incident_power_w = 0.0;
        bool top_boundary = false;
    };
    const auto ray_specification = [&](std::size_t ray_index)
    {
        RaySpecification specification;
        if (ray_index < top_ray_count)
        {
            const std::size_t i = ray_index / grid.ny;
            const std::size_t j = ray_index % grid.ny;
            specification.x = (static_cast<double>(i) + 0.5) * grid.dx_m;
            specification.y = (static_cast<double>(j) + 0.5) * grid.dy_m;
            specification.z = interface_value(i, j, grid.nz);
            specification.k = grid.nz - 1;
            const long double top_bundle_power =
                static_cast<long double>(boundary_irradiance(i, j))
                * sun_direction.up * grid.dx_m * grid.dy_m;
            if (!std::isfinite(top_bundle_power) || top_bundle_power >
                    static_cast<long double>(
                        std::numeric_limits<double>::max()))
                throw std::overflow_error(
                    "mapped direct-beam top bundle power overflowed");
            specification.incident_power_w = static_cast<double>(
                top_bundle_power);
            specification.top_boundary = true;
            return specification;
        }
        ray_index -= top_ray_count;
        if (ray_index < x_side_ray_count)
        {
            const bool west = photon.east > 0.0;
            const std::size_t i = west ? 0 : grid.nx - 1;
            const std::size_t j = ray_index / grid.nz;
            const std::size_t k = ray_index % grid.nz;
            const double lower = interface_value(i, j, k);
            const double upper = interface_value(i, j, k + 1);
            const long double power =
                static_cast<long double>(boundary_irradiance(i, j))
                * std::abs(photon.east) * grid.dy_m * (upper - lower);
            if (!std::isfinite(power)
                || power > static_cast<long double>(
                    std::numeric_limits<double>::max()))
                throw std::overflow_error(
                    "mapped direct-beam x-side bundle power overflowed");
            specification.x = west ? 0.0
                : static_cast<double>(grid.nx) * grid.dx_m;
            specification.y = (static_cast<double>(j) + 0.5) * grid.dy_m;
            specification.z = 0.5 * (lower + upper);
            specification.k = k;
            specification.incident_power_w = static_cast<double>(power);
            return specification;
        }
        ray_index -= x_side_ray_count;
        const bool south = photon.north > 0.0;
        const std::size_t j = south ? 0 : grid.ny - 1;
        const std::size_t i = ray_index / grid.nz;
        const std::size_t k = ray_index % grid.nz;
        const double lower = interface_value(i, j, k);
        const double upper = interface_value(i, j, k + 1);
        const long double power =
            static_cast<long double>(boundary_irradiance(i, j))
            * std::abs(photon.north) * grid.dx_m * (upper - lower);
        if (!std::isfinite(power)
            || power > static_cast<long double>(
                std::numeric_limits<double>::max()))
            throw std::overflow_error(
                "mapped direct-beam y-side bundle power overflowed");
        specification.x = (static_cast<double>(i) + 0.5) * grid.dx_m;
        specification.y = south ? 0.0
            : static_cast<double>(grid.ny) * grid.dy_m;
        specification.z = 0.5 * (lower + upper);
        specification.k = k;
        specification.incident_power_w = static_cast<double>(power);
        return specification;
    };

    constexpr std::size_t parallel_threshold = 256;
    constexpr std::size_t maximum_batch_width = 64;
    if (total_ray_count < parallel_threshold)
    {
        for (std::size_t ray_index = 0;
             ray_index < total_ray_count; ++ray_index)
        {
            const RaySpecification specification =
                ray_specification(ray_index);
            process_ray(
                specification.x, specification.y, specification.z,
                specification.k, specification.incident_power_w,
                specification.top_boundary);
        }
    }
    else
    {
        const std::size_t batch_width = std::min(
            maximum_batch_width, total_ray_count);
        workspace.ray_batch.resize(batch_width);
        workspace.ray_batch_failures.resize(batch_width);
        bool sweep_failed = false;
        std::exception_ptr sweep_exception;
        // One worker team owns the complete boundary sweep. Every worker
        // encounters the same bounded-batch sequence; tracing is distributed,
        // then one worker commits in exact top/x/y and segment order.
        #pragma omp parallel shared(sweep_failed, sweep_exception)
        {
            for (std::size_t batch_begin = 0;
                 batch_begin < total_ray_count;
                 batch_begin += batch_width)
            {
                const std::size_t batch_count = std::min(
                    batch_width, total_ray_count - batch_begin);
                #pragma omp single
                {
                    if (!sweep_failed)
                        std::fill_n(
                            workspace.ray_batch_failures.begin(), batch_count,
                            static_cast<unsigned char>(0));
                }

                #pragma omp for schedule(static)
                for (std::size_t slot = 0; slot < batch_count; ++slot)
                {
                    if (sweep_failed)
                        continue;
                    try
                    {
                        const RaySpecification specification =
                            ray_specification(batch_begin + slot);
                        auto& scratch = workspace.ray_batch[slot];
                        scratch.incident_power_w =
                            specification.incident_power_w;
                        scratch.top_boundary = specification.top_boundary;
                        scratch.result =
                            trace_mapped_direct_beam_from_entry_reference_impl(
                                grid,
                                specification.x, specification.y,
                                specification.z, specification.k,
                                photon, specification.incident_power_w,
                                scratch.segments, scratch.absorption,
                                validated_context, false, true);
                    }
                    catch (...)
                    {
                        workspace.ray_batch_failures[slot] = 1;
                    }
                }

                #pragma omp single
                {
                    if (!sweep_failed)
                    {
                        try
                        {
                            for (std::size_t slot = 0;
                                 slot < batch_count; ++slot)
                            {
                                if (workspace.ray_batch_failures[slot] != 0)
                                    throw std::runtime_error(
                                        "mapped direct-beam parallel ray batch failed");
                            }
                            // Preserve the original top/x/y source and path
                            // order without atomics or dense worker fields.
                            for (std::size_t slot = 0;
                                 slot < batch_count; ++slot)
                            {
                                const auto& scratch =
                                    workspace.ray_batch[slot];
                                commit_ray(
                                    scratch.result, scratch.absorption,
                                    scratch.incident_power_w,
                                    scratch.top_boundary);
                            }
                        }
                        catch (...)
                        {
                            sweep_failed = true;
                            sweep_exception = std::current_exception();
                        }
                    }
                }
            }
        }
        if (sweep_failed)
        {
            if (sweep_exception)
                std::rethrow_exception(sweep_exception);
            throw std::runtime_error("mapped direct-beam sweep failed");
        }
    }

    CompensatedSum resolved_cell_power;
    for (double value : workspace.scratch_cell_absorbed_power_w)
        resolved_cell_power.add(value);
    const long double cell_scale = std::max(
        1.0L, std::abs(atmospheric_absorbed.sum));
    if (std::abs(resolved_cell_power.sum - atmospheric_absorbed.sum)
        > 2048.0L * std::numeric_limits<double>::epsilon()
            * cell_scale * std::max<std::size_t>(1, cell_count))
        throw std::runtime_error(
            "mapped direct-beam resolved cell power does not close ray absorption");
    const long double incoming = incoming_top.sum + incoming_side.sum;
    const long double outgoing = atmospheric_absorbed.sum
        + surface_incident.sum + lateral_escape.sum;
    const long double residual = incoming - outgoing;
    const long double closure_scale = std::max(
        {1.0L, std::abs(incoming), std::abs(outgoing)});
    const std::uint64_t ray_count = top_rays + side_rays;
    if (std::abs(residual)
        > 2048.0L * std::numeric_limits<double>::epsilon()
            * closure_scale * std::max<std::uint64_t>(1, ray_count))
        throw std::runtime_error(
            "mapped direct-beam boundary power does not close");

    std::swap(result.cell_absorbed_power_w,
              workspace.scratch_cell_absorbed_power_w);
    std::swap(result.surface_incident_power_w,
              workspace.scratch_surface_incident_power_w);
    result.incoming_top_power_w = static_cast<double>(incoming_top.sum);
    result.incoming_side_power_w = static_cast<double>(incoming_side.sum);
    result.atmospheric_absorbed_power_w =
        static_cast<double>(atmospheric_absorbed.sum);
    result.surface_incident_total_power_w =
        static_cast<double>(surface_incident.sum);
    result.lateral_escape_power_w =
        static_cast<double>(lateral_escape.sum);
    result.boundary_closure_residual_w = static_cast<double>(residual);
    result.top_rays = top_rays;
    result.side_rays = side_rays;
    result.surface_rays = surface_rays;
    result.escaped_rays = escaped_rays;
    result.interface_values_validated = static_cast<std::uint64_t>(
        column_count * (grid.nz + 1));
    result.optical_values_validated = static_cast<std::uint64_t>(cell_count);
}

void evaluate_mapped_direct_beam_field_reference(
    const MappedDirectBeamGridView& grid,
    const IdealizedSolarGeometry& sun_direction,
    double beam_normal_irradiance_wm2,
    DirectBeamFieldWorkspace& workspace,
    DirectBeamFieldResult& result)
{
    evaluate_mapped_direct_beam_field_reference_impl(
        grid, sun_direction, beam_normal_irradiance_wm2, nullptr,
        workspace, result);
}

void evaluate_mapped_direct_beam_field_spatial_reference(
    const MappedDirectBeamGridView& grid,
    const IdealizedSolarGeometry& sun_direction,
    const double* beam_normal_irradiance_wm2_by_column,
    DirectBeamFieldWorkspace& workspace,
    DirectBeamFieldResult& result)
{
    if (beam_normal_irradiance_wm2_by_column == nullptr)
        throw std::invalid_argument(
            "mapped direct-beam spatial irradiance is missing");
    evaluate_mapped_direct_beam_field_reference_impl(
        grid, sun_direction, 0.0,
        beam_normal_irradiance_wm2_by_column, workspace, result);
}

void evaluate_sloped_lambertian_surface_field(
    const std::vector<double>& incident_power_w,
    const double* slope_x,
    const double* slope_y,
    std::size_t cell_count,
    double horizontal_cell_area_m2,
    double albedo,
    SlopedLambertianSurfaceFieldWorkspace& workspace,
    SlopedLambertianSurfaceFieldResult& result)
{
    if (incident_power_w.size() != cell_count
        || (cell_count != 0 && (slope_x == nullptr || slope_y == nullptr))
        || !std::isfinite(horizontal_cell_area_m2)
        || horizontal_cell_area_m2 <= 0.0
        || !std::isfinite(albedo) || albedo < 0.0 || albedo > 1.0)
        throw std::invalid_argument(
            "sloped Lambertian surface field has invalid shape or constants");
    for (std::size_t index = 0; index < cell_count; ++index)
        if (!std::isfinite(incident_power_w[index])
            || incident_power_w[index] < 0.0
            || !std::isfinite(slope_x[index])
            || !std::isfinite(slope_y[index]))
            throw std::invalid_argument(
                "sloped Lambertian surface field requires finite physical inputs");

    workspace.scratch_absorbed_power_w.resize(cell_count);
    workspace.scratch_reflected_power_w.resize(cell_count);
    workspace.scratch_reflected_radiance_wm2sr.resize(cell_count);
    long double incident_total = 0.0L;
    long double absorbed_total = 0.0L;
    long double reflected_total = 0.0L;
    for (std::size_t index = 0; index < cell_count; ++index)
    {
        const double incident = incident_power_w[index];
        const double area_ratio = std::hypot(
            std::hypot(slope_x[index], slope_y[index]), 1.0);
        const long double actual_area =
            static_cast<long double>(horizontal_cell_area_m2) * area_ratio;
        if (!std::isfinite(area_ratio) || !std::isfinite(actual_area)
            || actual_area <= 0.0L
            || actual_area > static_cast<long double>(
                std::numeric_limits<double>::max()))
            throw std::overflow_error(
                "sloped Lambertian surface area overflowed");
        const double reflected = albedo * incident;
        const double absorbed = incident - reflected;
        const double radiance = reflected
            / (static_cast<double>(actual_area) * radiation_constants::pi);
        if (!std::isfinite(reflected) || !std::isfinite(absorbed)
            || !std::isfinite(radiance))
            throw std::overflow_error(
                "sloped Lambertian surface field power overflowed");
        workspace.scratch_reflected_power_w[index] = reflected;
        workspace.scratch_absorbed_power_w[index] = absorbed;
        workspace.scratch_reflected_radiance_wm2sr[index] = radiance;
        incident_total += incident;
        absorbed_total += absorbed;
        reflected_total += reflected;
    }
    const long double residual = incident_total
        - absorbed_total - reflected_total;
    const long double scale = std::max(
        {1.0L, std::abs(incident_total),
         std::abs(absorbed_total + reflected_total)});
    if (std::abs(residual)
        > 1024.0L * std::numeric_limits<double>::epsilon()
            * scale * std::max<std::size_t>(1, cell_count))
        throw std::runtime_error(
            "sloped Lambertian surface field power does not close");

    std::swap(result.absorbed_power_w,
              workspace.scratch_absorbed_power_w);
    std::swap(result.reflected_power_w,
              workspace.scratch_reflected_power_w);
    std::swap(result.reflected_radiance_wm2sr,
              workspace.scratch_reflected_radiance_wm2sr);
    result.incident_total_power_w = static_cast<double>(incident_total);
    result.absorbed_total_power_w = static_cast<double>(absorbed_total);
    result.reflected_total_power_w = static_cast<double>(reflected_total);
    result.closure_residual_w = static_cast<double>(residual);
}

void evaluate_mapped_lambertian_reflection_sweep_reference_impl(
    const MappedDirectBeamGridView& grid,
    const std::vector<double>& reflected_power_w,
    const double* slope_x,
    const double* slope_y,
    const LambertianAngularQuadrature& quadrature,
    LambertianReflectionSweepWorkspace& workspace,
    LambertianReflectionSweepResult& result,
    const ValidatedMappedDirectBeamContext* prevalidated_context)
{
    if (grid.nx == 0 || grid.ny == 0 || grid.nz == 0
        || grid.nx > std::numeric_limits<std::size_t>::max() / grid.ny)
        throw std::invalid_argument(
            "Lambertian reflection sweep has invalid grid shape");
    const std::size_t column_count = grid.nx * grid.ny;
    if (column_count > std::numeric_limits<std::size_t>::max() / grid.nz
        || reflected_power_w.size() != column_count
        || slope_x == nullptr || slope_y == nullptr
        || quadrature.cosine_nodes == 0 || quadrature.cosine_nodes > 64
        || quadrature.azimuth_nodes == 0
        || quadrature.azimuth_nodes > 1024
        || quadrature.cosine_nodes
            > std::numeric_limits<std::size_t>::max()
                / quadrature.azimuth_nodes)
        throw std::invalid_argument(
            "Lambertian reflection sweep has invalid fields or quadrature");
    std::size_t active_source_count = 0;
    for (std::size_t column = 0; column < column_count; ++column)
    {
        if (!std::isfinite(reflected_power_w[column])
            || reflected_power_w[column] < 0.0
            || !std::isfinite(slope_x[column])
            || !std::isfinite(slope_y[column])
            || !std::isfinite(std::hypot(
                std::hypot(slope_x[column], slope_y[column]), 1.0)))
            throw std::invalid_argument(
                "Lambertian reflection sweep requires finite physical fields");
        if (reflected_power_w[column] > 0.0)
            ++active_source_count;
    }

    const bool performed_grid_validation = prevalidated_context == nullptr;
    ValidatedMappedDirectBeamContext local_context;
    if (prevalidated_context == nullptr)
    {
        local_context =
            validate_mapped_direct_beam_context(grid, {0.0, 0.0, 1.0});
        prevalidated_context = &local_context;
    }
    gauss_legendre_unit_interval(
        quadrature.cosine_nodes,
        workspace.cosine_nodes,
        workspace.cosine_weights);
    const std::size_t cell_count = column_count * grid.nz;
    workspace.scratch_cell_absorbed_power_w.resize(cell_count);
    workspace.scratch_surface_incident_power_w.resize(column_count);
    std::fill(workspace.scratch_cell_absorbed_power_w.begin(),
              workspace.scratch_cell_absorbed_power_w.end(), 0.0);
    std::fill(workspace.scratch_surface_incident_power_w.begin(),
              workspace.scratch_surface_incident_power_w.end(), 0.0);

    struct CompensatedSum
    {
        long double sum = 0.0L;
        long double correction = 0.0L;
        void add(long double value)
        {
            const long double adjusted = value - correction;
            const long double next = sum + adjusted;
            correction = (next - sum) - adjusted;
            sum = next;
        }
    };
    CompensatedSum emitted_total;
    CompensatedSum atmospheric_absorbed;
    CompensatedSum surface_incident;
    CompensatedSum top_escape;
    CompensatedSum lateral_escape;
    std::uint64_t rays = 0;
    std::uint64_t surface_rays = 0;
    std::uint64_t top_escape_rays = 0;
    std::uint64_t lateral_escape_rays = 0;
    const std::size_t angles_per_cell =
        quadrature.cosine_nodes * quadrature.azimuth_nodes;
    if (column_count
        > std::numeric_limits<std::uint64_t>::max() / angles_per_cell)
        throw std::overflow_error(
            "Lambertian reflection sweep ray count overflowed");
    const std::size_t interface_stride = grid.nz + 1;

    // Validate the only source-entry condition that can fail before entering
    // the persistent OpenMP region. Exceptions must never escape an OpenMP
    // structured block: doing so can terminate the process or strand workers
    // at a barrier.
    for (std::size_t column = 0; column < column_count; ++column)
    {
        if (reflected_power_w[column] == 0.0)
            continue;
        const std::size_t interface_base = column * interface_stride;
        const double ground = grid.z_interface_m[interface_base];
        const double first_interface =
            grid.z_interface_m[interface_base + 1];
        const double start_z = std::nextafter(ground, first_interface);
        if (!std::isfinite(start_z) || start_z <= ground
            || start_z >= first_interface)
            throw std::invalid_argument(
                "Lambertian reflection source has no representable clearance");
    }

    const std::size_t active_ray_count =
        active_source_count * angles_per_cell;
    // Below this measured-work floor, team barriers cost more than the mapped
    // characteristics on personal CPUs. The serial and parallel paths share
    // the exact same ordered kernel; only worker-team creation changes.
    constexpr std::size_t minimum_parallel_ray_count = 65536;
    const bool parallel_angular_sweep =
        active_ray_count >= minimum_parallel_ray_count;
    constexpr std::size_t maximum_batch_width = 256;
    const std::size_t batch_width = std::min(
        maximum_batch_width, active_ray_count);
    workspace.ray_batch.resize(batch_width);
    workspace.ray_batch_failures.resize(batch_width);

    long double emitted_from_source_shared = 0.0L;
    std::size_t source_cursor_shared = 0;
    std::size_t angle_cursor_shared = 0;
    std::size_t batch_count_shared = 0;
    bool batch_complete_shared;
    double source_power_shared = 0.0;
    double source_start_x_shared = 0.0;
    double source_start_y_shared = 0.0;
    double source_start_z_shared = 0.0;
    IdealizedSolarGeometry source_normal_shared;
    IdealizedSolarGeometry source_tangent_x_shared;
    IdealizedSolarGeometry source_tangent_y_shared;
    bool sweep_failed = false;
    int sweep_failure_kind = 0;
    std::exception_ptr sweep_exception;
    // Batches may span surface sources, but generation and commit retain the
    // exact source/angle/segment sequence. This avoids a synchronization cycle
    // for every reflecting cell while preserving bounded memory and bitwise
    // worker-count identity.
    #pragma omp parallel if(parallel_angular_sweep) \
        shared(emitted_from_source_shared, source_cursor_shared, \
               angle_cursor_shared, batch_count_shared, \
               batch_complete_shared, source_power_shared, \
               source_start_x_shared, source_start_y_shared, \
               source_start_z_shared, source_normal_shared, \
               source_tangent_x_shared, source_tangent_y_shared, \
               sweep_failed, sweep_failure_kind, sweep_exception)
    {
        while (true)
        {
            #pragma omp single
            {
                batch_count_shared = 0;
                if (!sweep_failed)
                {
                    try
                    {
                        while (batch_count_shared < batch_width)
                        {
                            while (source_cursor_shared < column_count
                                   && reflected_power_w[
                                       source_cursor_shared] == 0.0)
                                ++source_cursor_shared;
                            if (source_cursor_shared == column_count)
                                break;

                            if (angle_cursor_shared == 0)
                            {
                                const std::size_t column =
                                    source_cursor_shared;
                                const std::size_t i = column / grid.ny;
                                const std::size_t j = column % grid.ny;
                                source_power_shared =
                                    reflected_power_w[column];
                                const double sx = slope_x[column];
                                const double sy = slope_y[column];
                                const double normal_scale =
                                    std::hypot(std::hypot(sx, sy), 1.0);
                                source_normal_shared = {
                                    -sx / normal_scale,
                                    -sy / normal_scale,
                                    1.0 / normal_scale};
                                const double tangent_scale =
                                    std::hypot(1.0, sx);
                                source_tangent_x_shared = {
                                    1.0 / tangent_scale, 0.0,
                                    sx / tangent_scale};
                                source_tangent_y_shared = {
                                    -sx * sy
                                        / (normal_scale * tangent_scale),
                                    tangent_scale / normal_scale,
                                    sy / (normal_scale * tangent_scale)};
                                source_start_x_shared =
                                    (static_cast<double>(i) + 0.5)
                                    * grid.dx_m;
                                source_start_y_shared =
                                    (static_cast<double>(j) + 0.5)
                                    * grid.dy_m;
                                const std::size_t interface_base =
                                    column * interface_stride;
                                source_start_z_shared = std::nextafter(
                                    grid.z_interface_m[interface_base],
                                    grid.z_interface_m[interface_base + 1]);
                                emitted_total.add(source_power_shared);
                                emitted_from_source_shared = 0.0L;
                            }
                            const double source_power = source_power_shared;

                            const std::size_t angle_index =
                                angle_cursor_shared;
                            const std::size_t mu_index = angle_index
                                / quadrature.azimuth_nodes;
                            const std::size_t azimuth_index = angle_index
                                % quadrature.azimuth_nodes;
                            const double mu =
                                workspace.cosine_nodes[mu_index];
                            const double radial = std::sqrt(
                                std::max(0.0, 1.0 - mu * mu));
                            const double azimuth =
                                2.0 * radiation_constants::pi
                                * (static_cast<double>(azimuth_index) + 0.5)
                                / static_cast<double>(
                                    quadrature.azimuth_nodes);
                            auto& scratch = workspace.ray_batch[
                                batch_count_shared];
                            scratch.direction = {
                                mu * source_normal_shared.east + radial * (
                                    std::cos(azimuth)
                                        * source_tangent_x_shared.east
                                    + std::sin(azimuth)
                                        * source_tangent_y_shared.east),
                                mu * source_normal_shared.north + radial * (
                                    std::cos(azimuth)
                                        * source_tangent_x_shared.north
                                    + std::sin(azimuth)
                                        * source_tangent_y_shared.north),
                                mu * source_normal_shared.up + radial * (
                                    std::cos(azimuth)
                                        * source_tangent_x_shared.up
                                    + std::sin(azimuth)
                                        * source_tangent_y_shared.up)};
                            const double direction_norm = std::hypot(
                                std::hypot(scratch.direction.east,
                                           scratch.direction.north),
                                scratch.direction.up);
                            scratch.direction.east /= direction_norm;
                            scratch.direction.north /= direction_norm;
                            scratch.direction.up /= direction_norm;
                            scratch.start_x_m = source_start_x_shared;
                            scratch.start_y_m = source_start_y_shared;
                            scratch.start_z_m = source_start_z_shared;

                            double& bundle_power =
                                scratch.incident_power_w;
                            if (angle_index + 1 == angles_per_cell)
                            {
                                long double remainder =
                                    static_cast<long double>(source_power)
                                    - emitted_from_source_shared;
                                const long double tolerance = 512.0L
                                    * std::numeric_limits<double>::epsilon()
                                    * std::max(1.0L,
                                        static_cast<long double>(
                                            source_power));
                                if (remainder < 0.0L
                                    && remainder >= -tolerance)
                                    remainder = 0.0L;
                                if (!std::isfinite(remainder)
                                    || remainder < 0.0L
                                    || remainder > static_cast<long double>(
                                        std::numeric_limits<double>::max()))
                                    throw std::runtime_error(
                                        "Lambertian angular weights do not close");
                                bundle_power =
                                    static_cast<double>(remainder);
                            }
                            else
                            {
                                const long double fraction = 2.0L
                                    * workspace.cosine_weights[mu_index] * mu
                                    / static_cast<long double>(
                                        quadrature.azimuth_nodes);
                                const long double bundle =
                                    static_cast<long double>(source_power)
                                    * fraction;
                                if (!std::isfinite(bundle) || bundle < 0.0L
                                    || bundle > static_cast<long double>(
                                        std::numeric_limits<double>::max()))
                                    throw std::overflow_error(
                                        "Lambertian bundle power overflowed");
                                bundle_power = static_cast<double>(bundle);
                            }
                            emitted_from_source_shared += bundle_power;
                            ++batch_count_shared;
                            ++angle_cursor_shared;

                            if (angle_cursor_shared == angles_per_cell)
                            {
                                const long double source_residual =
                                    static_cast<long double>(source_power)
                                    - emitted_from_source_shared;
                                if (std::abs(source_residual)
                                    > 512.0L
                                        * std::numeric_limits<double>::epsilon()
                                        * std::max(1.0L,
                                            static_cast<long double>(
                                                source_power)))
                                    throw std::runtime_error(
                                        "Lambertian cell bundle power does not close");
                                angle_cursor_shared = 0;
                                ++source_cursor_shared;
                            }
                        }
                        std::fill_n(
                            workspace.ray_batch_failures.begin(),
                            batch_count_shared,
                            static_cast<unsigned char>(0));
                    }
                    catch (...)
                    {
                        sweep_failed = true;
                        sweep_exception = std::current_exception();
                    }
                }
                batch_complete_shared =
                    sweep_failed || batch_count_shared == 0;
            }

            if (batch_complete_shared)
                break;

            #pragma omp for schedule(static)
            for (std::size_t slot = 0;
                 slot < batch_count_shared; ++slot)
            {
                try
                {
                    auto& scratch = workspace.ray_batch[slot];
                    ValidatedMappedDirectBeamContext ray_context =
                        *prevalidated_context;
                    ray_context.maximum_iterations =
                        mapped_radiative_ray_iteration_bound(
                            grid, ray_context,
                            scratch.start_x_m, scratch.start_y_m,
                            scratch.start_z_m, scratch.direction);
                    scratch.result =
                        trace_mapped_direct_beam_from_entry_reference_impl(
                            grid,
                            scratch.start_x_m, scratch.start_y_m,
                            scratch.start_z_m, 0, scratch.direction,
                            scratch.incident_power_w,
                            scratch.segments, scratch.absorption,
                            ray_context, true, true);
                }
                catch (...)
                {
                    workspace.ray_batch_failures[slot] = 1;
                }
            }

            #pragma omp single
            {
                for (std::size_t slot = 0;
                     slot < batch_count_shared; ++slot)
                {
                    if (workspace.ray_batch_failures[slot] != 0)
                    {
                        sweep_failed = true;
                        sweep_failure_kind = 1;
                        break;
                    }
                }
                // Commit in source, angle, and segment order so worker count
                // cannot change any floating-point result.
                if (!sweep_failed)
                {
                    for (std::size_t slot = 0;
                         slot < batch_count_shared; ++slot)
                    {
                        const auto& scratch = workspace.ray_batch[slot];
                        const DirectBeamRayResult& ray = scratch.result;
                        ++rays;
                        atmospheric_absorbed.add(
                            ray.energy.absorbed_power_w);
                        for (const DirectBeamSegmentAbsorption& segment :
                             scratch.absorption)
                            workspace.scratch_cell_absorbed_power_w[
                                segment.cell_index]
                                += segment.absorbed_power_w;
                        if (ray.exit == DirectBeamRayExit::Surface)
                        {
                            workspace.scratch_surface_incident_power_w[
                                ray.terminal_i * grid.ny + ray.terminal_j]
                                += ray.energy.transmitted_power_w;
                            surface_incident.add(
                                ray.energy.transmitted_power_w);
                            ++surface_rays;
                        }
                        else if (ray.exit == DirectBeamRayExit::LevelTop)
                        {
                            top_escape.add(ray.energy.transmitted_power_w);
                            ++top_escape_rays;
                        }
                        else
                        {
                            lateral_escape.add(
                                ray.energy.transmitted_power_w);
                            ++lateral_escape_rays;
                        }
                    }
                }
            }
        }
    }

    if (sweep_failed)
    {
        if (sweep_exception)
            std::rethrow_exception(sweep_exception);
        if (sweep_failure_kind == 1)
            throw std::runtime_error(
                "Lambertian parallel ray batch failed");
        throw std::runtime_error("Lambertian reflection sweep failed");
    }

    CompensatedSum resolved_cell_absorption;
    for (double value : workspace.scratch_cell_absorbed_power_w)
        resolved_cell_absorption.add(value);
    CompensatedSum resolved_surface_incident;
    for (double value : workspace.scratch_surface_incident_power_w)
        resolved_surface_incident.add(value);
    const long double field_tolerance = 2048.0L
        * std::numeric_limits<double>::epsilon()
        * std::max<std::uint64_t>(1, rays);
    if (std::abs(resolved_cell_absorption.sum - atmospheric_absorbed.sum)
            > field_tolerance * std::max(
                1.0L, std::abs(atmospheric_absorbed.sum))
        || std::abs(resolved_surface_incident.sum - surface_incident.sum)
            > field_tolerance * std::max(
                1.0L, std::abs(surface_incident.sum)))
        throw std::runtime_error(
            "Lambertian reflection resolved fields do not close ray ledgers");
    const long double outgoing = atmospheric_absorbed.sum
        + surface_incident.sum + top_escape.sum + lateral_escape.sum;
    const long double residual = emitted_total.sum - outgoing;
    if (std::abs(residual) > field_tolerance * std::max({
            1.0L, std::abs(emitted_total.sum), std::abs(outgoing)}))
        throw std::runtime_error(
            "Lambertian reflection sweep power does not close");

    std::swap(result.cell_absorbed_power_w,
              workspace.scratch_cell_absorbed_power_w);
    std::swap(result.surface_incident_power_w,
              workspace.scratch_surface_incident_power_w);
    result.emitted_power_w = static_cast<double>(emitted_total.sum);
    result.atmospheric_absorbed_power_w =
        static_cast<double>(atmospheric_absorbed.sum);
    result.surface_incident_total_power_w =
        static_cast<double>(surface_incident.sum);
    result.top_escape_power_w = static_cast<double>(top_escape.sum);
    result.lateral_escape_power_w = static_cast<double>(lateral_escape.sum);
    result.closure_residual_w = static_cast<double>(residual);
    result.rays = rays;
    result.surface_rays = surface_rays;
    result.top_escape_rays = top_escape_rays;
    result.lateral_escape_rays = lateral_escape_rays;
    result.interface_values_validated = performed_grid_validation
        ? static_cast<std::uint64_t>(column_count * (grid.nz + 1)) : 0;
    result.optical_values_validated = performed_grid_validation
        ? static_cast<std::uint64_t>(cell_count) : 0;
}

void evaluate_mapped_lambertian_reflection_sweep_reference(
    const MappedDirectBeamGridView& grid,
    const std::vector<double>& reflected_power_w,
    const double* slope_x,
    const double* slope_y,
    const LambertianAngularQuadrature& quadrature,
    LambertianReflectionSweepWorkspace& workspace,
    LambertianReflectionSweepResult& result)
{
    evaluate_mapped_lambertian_reflection_sweep_reference_impl(
        grid, reflected_power_w, slope_x, slope_y, quadrature,
        workspace, result, nullptr);
}

void evaluate_mapped_lambertian_multiple_reflection_reference(
    const MappedDirectBeamGridView& grid,
    const std::vector<double>& initial_reflected_power_w,
    const double* slope_x,
    const double* slope_y,
    const LambertianMultipleReflectionOptions& options,
    LambertianMultipleReflectionWorkspace& workspace,
    LambertianMultipleReflectionResult& result)
{
    if (grid.nx == 0 || grid.ny == 0 || grid.nz == 0
        || grid.nx > std::numeric_limits<std::size_t>::max() / grid.ny)
        throw std::invalid_argument(
            "multiple Lambertian reflection has invalid grid shape");
    const std::size_t column_count = grid.nx * grid.ny;
    if (column_count > std::numeric_limits<std::size_t>::max() / grid.nz
        || initial_reflected_power_w.size() != column_count
        || slope_x == nullptr || slope_y == nullptr
        || !std::isfinite(options.albedo)
        || options.albedo < 0.0 || options.albedo > 1.0
        || options.maximum_bounces == 0
        || !std::isfinite(options.relative_residual_tolerance)
        || options.relative_residual_tolerance < 0.0
        || !std::isfinite(options.absolute_residual_tolerance_w)
        || options.absolute_residual_tolerance_w < 0.0)
        throw std::invalid_argument(
            "multiple Lambertian reflection has invalid fields or controls");
    long double initial_total = 0.0L;
    for (std::size_t column = 0; column < column_count; ++column)
    {
        if (!std::isfinite(initial_reflected_power_w[column])
            || initial_reflected_power_w[column] < 0.0
            || !std::isfinite(slope_x[column])
            || !std::isfinite(slope_y[column])
            || !std::isfinite(std::hypot(
                std::hypot(slope_x[column], slope_y[column]), 1.0)))
            throw std::invalid_argument(
                "multiple Lambertian reflection requires finite physical fields");
        initial_total += initial_reflected_power_w[column];
    }
    if (!std::isfinite(initial_total)
        || initial_total > static_cast<long double>(
            std::numeric_limits<double>::max()))
        throw std::overflow_error(
            "multiple Lambertian reflection source power overflowed");
    const ValidatedMappedDirectBeamContext validated_context =
        validate_mapped_direct_beam_context(grid, {0.0, 0.0, 1.0});

    const std::size_t cell_count = column_count * grid.nz;
    workspace.current_reflected_power_w = initial_reflected_power_w;
    workspace.next_reflected_power_w.resize(column_count);
    workspace.scratch_cell_absorbed_power_w.resize(cell_count);
    workspace.scratch_surface_absorbed_power_w.resize(column_count);
    std::fill(workspace.scratch_cell_absorbed_power_w.begin(),
              workspace.scratch_cell_absorbed_power_w.end(), 0.0);
    std::fill(workspace.scratch_surface_absorbed_power_w.begin(),
              workspace.scratch_surface_absorbed_power_w.end(), 0.0);

    long double atmospheric_absorbed = 0.0L;
    long double surface_absorbed = 0.0L;
    long double top_escape = 0.0L;
    long double lateral_escape = 0.0L;
    long double unresolved = initial_total;
    std::uint64_t rays = 0;
    std::size_t completed_bounces = 0;
    bool converged = false;
    const long double convergence_threshold = std::max(
        static_cast<long double>(options.absolute_residual_tolerance_w),
        static_cast<long double>(options.relative_residual_tolerance)
            * initial_total);

    for (std::size_t bounce = 0;
         bounce < options.maximum_bounces; ++bounce)
    {
        evaluate_mapped_lambertian_reflection_sweep_reference_impl(
            grid, workspace.current_reflected_power_w,
            slope_x, slope_y, options.quadrature,
            workspace.sweep_workspace, workspace.sweep_result,
            &validated_context);
        if (rays > std::numeric_limits<std::uint64_t>::max()
                - workspace.sweep_result.rays)
            throw std::overflow_error(
                "multiple Lambertian reflection ray count overflowed");
        rays += workspace.sweep_result.rays;
        atmospheric_absorbed +=
            workspace.sweep_result.atmospheric_absorbed_power_w;
        top_escape += workspace.sweep_result.top_escape_power_w;
        lateral_escape += workspace.sweep_result.lateral_escape_power_w;
        for (std::size_t cell = 0; cell < cell_count; ++cell)
        {
            const double updated = workspace.scratch_cell_absorbed_power_w[cell]
                + workspace.sweep_result.cell_absorbed_power_w[cell];
            if (!std::isfinite(updated))
                throw std::overflow_error(
                    "multiple Lambertian atmospheric field overflowed");
            workspace.scratch_cell_absorbed_power_w[cell] = updated;
        }

        long double next_total = 0.0L;
        for (std::size_t column = 0; column < column_count; ++column)
        {
            const double incident =
                workspace.sweep_result.surface_incident_power_w[column];
            const double reflected = options.albedo * incident;
            const double absorbed = incident - reflected;
            const double updated =
                workspace.scratch_surface_absorbed_power_w[column] + absorbed;
            if (!std::isfinite(reflected) || !std::isfinite(absorbed)
                || !std::isfinite(updated))
                throw std::overflow_error(
                    "multiple Lambertian surface field overflowed");
            workspace.next_reflected_power_w[column] = reflected;
            workspace.scratch_surface_absorbed_power_w[column] = updated;
            surface_absorbed += absorbed;
            next_total += reflected;
        }
        if (!std::isfinite(next_total)
            || next_total > static_cast<long double>(
                std::numeric_limits<double>::max()))
            throw std::overflow_error(
                "multiple Lambertian residual power overflowed");
        unresolved = next_total;
        ++completed_bounces;
        if (unresolved <= convergence_threshold)
        {
            converged = true;
            break;
        }
        workspace.current_reflected_power_w.swap(
            workspace.next_reflected_power_w);
    }

    long double resolved_cell_absorbed = 0.0L;
    for (double value : workspace.scratch_cell_absorbed_power_w)
        resolved_cell_absorbed += value;
    long double resolved_surface_absorbed = 0.0L;
    for (double value : workspace.scratch_surface_absorbed_power_w)
        resolved_surface_absorbed += value;
    const long double field_scale = std::max(
        1.0L, std::abs(initial_total));
    const long double field_tolerance = 4096.0L
        * std::numeric_limits<double>::epsilon()
        * field_scale * std::max<std::size_t>(1, completed_bounces);
    if (std::abs(resolved_cell_absorbed - atmospheric_absorbed)
            > field_tolerance
        || std::abs(resolved_surface_absorbed - surface_absorbed)
            > field_tolerance)
        throw std::runtime_error(
            "multiple Lambertian resolved fields do not close totals");
    const long double accounted = atmospheric_absorbed + surface_absorbed
        + top_escape + lateral_escape + unresolved;
    const long double residual = initial_total - accounted;
    if (std::abs(residual) > field_tolerance)
        throw std::runtime_error(
            "multiple Lambertian reflection power does not close");

    std::swap(result.cell_absorbed_power_w,
              workspace.scratch_cell_absorbed_power_w);
    std::swap(result.surface_absorbed_power_w,
              workspace.scratch_surface_absorbed_power_w);
    result.initial_reflected_power_w = static_cast<double>(initial_total);
    result.atmospheric_absorbed_power_w =
        static_cast<double>(atmospheric_absorbed);
    result.surface_absorbed_power_w_total =
        static_cast<double>(surface_absorbed);
    result.top_escape_power_w = static_cast<double>(top_escape);
    result.lateral_escape_power_w = static_cast<double>(lateral_escape);
    result.unresolved_reflected_power_w = static_cast<double>(unresolved);
    result.closure_residual_w = static_cast<double>(residual);
    result.rays = rays;
    result.completed_bounces = completed_bounces;
    result.converged = converged;
    result.interface_values_validated = static_cast<std::uint64_t>(
        column_count * (grid.nz + 1));
    result.optical_values_validated = static_cast<std::uint64_t>(cell_count);
}

void evaluate_mapped_terrain_shortwave_reference(
    const MappedDirectBeamGridView& grid,
    const IdealizedSolarGeometry& sun_direction,
    double beam_normal_irradiance_wm2,
    const double* slope_x,
    const double* slope_y,
    double horizontal_cell_area_m2,
    const LambertianMultipleReflectionOptions& reflection_options,
    MappedTerrainShortwaveWorkspace& workspace,
    MappedTerrainShortwaveResult& result)
{
    if (grid.nx == 0 || grid.ny == 0 || grid.nz == 0
        || grid.nx > std::numeric_limits<std::size_t>::max() / grid.ny)
        throw std::invalid_argument(
            "mapped terrain shortwave has invalid grid shape");
    const std::size_t column_count = grid.nx * grid.ny;
    if (column_count > std::numeric_limits<std::size_t>::max() / grid.nz
        || slope_x == nullptr || slope_y == nullptr
        || !std::isfinite(horizontal_cell_area_m2)
        || horizontal_cell_area_m2 <= 0.0)
        throw std::invalid_argument(
            "mapped terrain shortwave has invalid surface geometry");
    const std::size_t cell_count = column_count * grid.nz;

    evaluate_mapped_direct_beam_field_reference(
        grid, sun_direction, beam_normal_irradiance_wm2,
        workspace.direct_workspace, workspace.direct_result);
    evaluate_sloped_lambertian_surface_field(
        workspace.direct_result.surface_incident_power_w,
        slope_x, slope_y, column_count,
        horizontal_cell_area_m2, reflection_options.albedo,
        workspace.surface_workspace, workspace.surface_result);
    evaluate_mapped_lambertian_multiple_reflection_reference(
        grid, workspace.surface_result.reflected_power_w,
        slope_x, slope_y, reflection_options,
        workspace.reflection_workspace, workspace.reflection_result);

    workspace.scratch_atmospheric_absorbed_power_w.resize(cell_count);
    workspace.scratch_surface_absorbed_power_w.resize(column_count);
    long double atmospheric_total = 0.0L;
    long double surface_total = 0.0L;
    for (std::size_t cell = 0; cell < cell_count; ++cell)
    {
        const double power =
            workspace.direct_result.cell_absorbed_power_w[cell]
            + workspace.reflection_result.cell_absorbed_power_w[cell];
        if (!std::isfinite(power) || power < 0.0)
            throw std::overflow_error(
                "mapped terrain shortwave atmospheric field overflowed");
        workspace.scratch_atmospheric_absorbed_power_w[cell] = power;
        atmospheric_total += power;
    }
    for (std::size_t column = 0; column < column_count; ++column)
    {
        const double power = workspace.surface_result.absorbed_power_w[column]
            + workspace.reflection_result.surface_absorbed_power_w[column];
        if (!std::isfinite(power) || power < 0.0)
            throw std::overflow_error(
                "mapped terrain shortwave surface field overflowed");
        workspace.scratch_surface_absorbed_power_w[column] = power;
        surface_total += power;
    }
    const long double incoming =
        static_cast<long double>(workspace.direct_result.incoming_top_power_w)
        + workspace.direct_result.incoming_side_power_w;
    const long double top_escape =
        workspace.reflection_result.top_escape_power_w;
    const long double lateral_escape =
        static_cast<long double>(workspace.direct_result.lateral_escape_power_w)
        + workspace.reflection_result.lateral_escape_power_w;
    const long double unresolved =
        workspace.reflection_result.unresolved_reflected_power_w;
    const long double accounted = atmospheric_total + surface_total
        + top_escape + lateral_escape + unresolved;
    const long double residual = incoming - accounted;
    const long double tolerance = 8192.0L
        * std::numeric_limits<double>::epsilon()
        * std::max({1.0L, std::abs(incoming), std::abs(accounted)})
        * std::max<std::size_t>(
            1, workspace.reflection_result.completed_bounces);
    if (std::abs(residual) > tolerance)
        throw std::runtime_error(
            "mapped terrain shortwave complete power does not close");

    std::swap(result.atmospheric_absorbed_power_w,
              workspace.scratch_atmospheric_absorbed_power_w);
    std::swap(result.surface_absorbed_power_w,
              workspace.scratch_surface_absorbed_power_w);
    result.incoming_power_w = static_cast<double>(incoming);
    result.atmospheric_absorbed_power_w_total =
        static_cast<double>(atmospheric_total);
    result.surface_absorbed_power_w_total = static_cast<double>(surface_total);
    result.top_escape_power_w = static_cast<double>(top_escape);
    result.lateral_escape_power_w = static_cast<double>(lateral_escape);
    result.unresolved_reflected_power_w = static_cast<double>(unresolved);
    result.closure_residual_w = static_cast<double>(residual);
    result.reflection_converged = workspace.reflection_result.converged;
}

void evaluate_mapped_spectral_direct_shortwave_reference(
    const MappedSpectralShortwaveGridView& grid,
    const IdealizedSolarGeometry& sun_direction,
    double beam_normal_irradiance_wm2,
    MappedSpectralDirectWorkspace& workspace,
    MappedSpectralDirectResult& result)
{
    if (grid.nx == 0 || grid.ny == 0 || grid.nz == 0 ||
        grid.g_points == 0 || grid.g_points > 1024 ||
        grid.nx > std::numeric_limits<std::size_t>::max() / grid.ny ||
        grid.absorption_optical_depth == nullptr ||
        grid.scattering_optical_depth == nullptr ||
        grid.scattering_asymmetry == nullptr ||
        grid.solar_source_fraction == nullptr ||
        grid.z_interface_m == nullptr ||
        !std::isfinite(beam_normal_irradiance_wm2) ||
        beam_normal_irradiance_wm2 < 0.0)
        throw std::invalid_argument(
            "mapped spectral direct shortwave has invalid shape or fields");
    const std::size_t column_count = grid.nx * grid.ny;
    if (column_count > std::numeric_limits<std::size_t>::max() / grid.nz)
        throw std::overflow_error(
            "mapped spectral direct shortwave cell count overflowed");
    const std::size_t cell_count = column_count * grid.nz;
    if (cell_count > std::numeric_limits<std::size_t>::max() / grid.g_points)
        throw std::overflow_error(
            "mapped spectral direct shortwave optical count overflowed");

    workspace.extinction_optical_depth.resize(cell_count);
    workspace.scratch_cell_absorbed_power_w.assign(cell_count, 0.0);
    workspace.scratch_cell_scattered_power_w.assign(cell_count, 0.0);
    workspace.scratch_surface_incident_power_w.assign(column_count, 0.0);
    for (std::size_t column = 0; column < column_count; ++column)
    {
        long double source_sum = 0.0L;
        for (std::size_t g = 0; g < grid.g_points; ++g)
        {
            const double source = grid.solar_source_fraction[
                column * grid.g_points + g];
            if (!std::isfinite(source) || source < 0.0)
                throw std::invalid_argument(
                    "mapped spectral direct shortwave has invalid source weights");
            source_sum += source;
        }
        if (std::abs(source_sum - 1.0L) > 1.0e-12L)
            throw std::invalid_argument(
                "mapped spectral direct shortwave source weights must normalize by column");
    }
    workspace.boundary_irradiance_wm2.resize(column_count);

    long double incoming = 0.0L;
    long double absorbed = 0.0L;
    long double scattered = 0.0L;
    long double surface = 0.0L;
    long double top_escape = 0.0L;
    long double lateral_escape = 0.0L;
    for (std::size_t g = 0; g < grid.g_points; ++g)
    {
        for (std::size_t column = 0; column < column_count; ++column)
            workspace.boundary_irradiance_wm2[column] =
                beam_normal_irradiance_wm2 *
                grid.solar_source_fraction[
                    column * grid.g_points + g];
        for (std::size_t cell = 0; cell < cell_count; ++cell)
        {
            const std::size_t spectral = cell * grid.g_points + g;
            const double absorption = grid.absorption_optical_depth[spectral];
            const double scattering_depth = grid.scattering_optical_depth[spectral];
            const double asymmetry = grid.scattering_asymmetry[spectral];
            if (!std::isfinite(absorption) || absorption < 0.0 ||
                !std::isfinite(scattering_depth) || scattering_depth < 0.0 ||
                !std::isfinite(asymmetry) || asymmetry < 0.0 || asymmetry > 1.0)
                throw std::invalid_argument(
                    "mapped spectral direct shortwave has invalid optical state");
            workspace.extinction_optical_depth[cell] =
                absorption + scattering_depth;
        }
        const MappedDirectBeamGridView direct_grid{
            workspace.extinction_optical_depth.data(), grid.z_interface_m,
            grid.nx, grid.ny, grid.nz, grid.dx_m, grid.dy_m,
            grid.x_boundary, grid.y_boundary};
        evaluate_mapped_direct_beam_field_spatial_reference(
            direct_grid, sun_direction,
            workspace.boundary_irradiance_wm2.data(),
            workspace.direct_workspace, workspace.direct_result);
        const auto& direct = workspace.direct_result;
        incoming += direct.incoming_top_power_w + direct.incoming_side_power_w;
        lateral_escape += direct.lateral_escape_power_w;
        for (std::size_t column = 0; column < column_count; ++column)
        {
            workspace.scratch_surface_incident_power_w[column] +=
                direct.surface_incident_power_w[column];
            surface += direct.surface_incident_power_w[column];
        }
        for (std::size_t cell = 0; cell < cell_count; ++cell)
        {
            const std::size_t spectral = cell * grid.g_points + g;
            const double absorption_depth =
                grid.absorption_optical_depth[spectral];
            const double scattering_depth =
                grid.scattering_optical_depth[spectral];
            const double extinction = absorption_depth + scattering_depth;
            const double removed = direct.cell_absorbed_power_w[cell];
            const double cell_absorbed = extinction > 0.0
                ? removed * absorption_depth / extinction : 0.0;
            const double cell_scattered = removed - cell_absorbed;
            workspace.scratch_cell_absorbed_power_w[cell] += cell_absorbed;
            workspace.scratch_cell_scattered_power_w[cell] += cell_scattered;
            absorbed += cell_absorbed;
            scattered += cell_scattered;
        }
    }
    const long double accounted = absorbed + scattered + surface +
        top_escape + lateral_escape;
    const long double residual = incoming - accounted;
    const long double tolerance = 8192.0L *
        std::numeric_limits<double>::epsilon() *
        std::max({1.0L, std::abs(incoming), std::abs(accounted)}) *
        std::max<std::size_t>(1, grid.g_points);
    if (std::abs(residual) > tolerance)
        throw std::runtime_error(
            "mapped spectral direct shortwave power does not close");

    std::swap(result.cell_absorbed_power_w,
              workspace.scratch_cell_absorbed_power_w);
    std::swap(result.cell_scattered_power_w,
              workspace.scratch_cell_scattered_power_w);
    std::swap(result.surface_incident_power_w,
              workspace.scratch_surface_incident_power_w);
    result.incoming_power_w = static_cast<double>(incoming);
    result.atmospheric_absorbed_power_w = static_cast<double>(absorbed);
    result.unresolved_scattered_power_w = static_cast<double>(scattered);
    result.surface_incident_power_w_total = static_cast<double>(surface);
    result.top_escape_power_w = static_cast<double>(top_escape);
    result.lateral_escape_power_w = static_cast<double>(lateral_escape);
    result.closure_residual_w = static_cast<double>(residual);
}

namespace
{
class SpectralRayGeometryCache
{
    struct Segment { std::size_t cell; double length_m; };
    struct Ray
    {
        std::size_t offset = 0, count = 0, terminal_column = 0;
        double terminal_x_m = 0.0, terminal_y_m = 0.0, terminal_z_m = 0.0;
        DirectBeamRayExit exit = DirectBeamRayExit::Surface;
        bool ready = false;
    };
    std::vector<Ray> rays_;
    std::vector<Segment> segments_;
public:
    SpectralRayGeometryCache(std::size_t rays, std::size_t budget)
    {
        budget = std::min(budget, mapped_geometry_cache_max_bytes);
        rays_.resize(std::min(rays, budget / (4 * sizeof(Ray))));
        if (!rays_.empty())
            segments_.reserve((budget - rays_.capacity() * sizeof(Ray)) / sizeof(Segment));
    }
    std::size_t bytes() const
    { return rays_.capacity() * sizeof(Ray) + segments_.capacity() * sizeof(Segment); }
    bool load(std::size_t key, const MappedDirectBeamGridView& grid, double power,
              std::vector<DirectBeamPathSegment>& path,
              std::vector<DirectBeamSegmentAbsorption>& absorption,
              DirectBeamRayResult& result) const
    {
        if (key >= rays_.size() || !rays_[key].ready) return false;
        const auto& ray = rays_[key];
        path.resize(ray.count);
        for (std::size_t n = 0; n < ray.count; ++n)
        {
            const auto& segment = segments_[ray.offset + n];
            const auto k = segment.cell % grid.nz;
            const auto base = (segment.cell / grid.nz) * (grid.nz + 1) + k;
            path[n] = {segment.cell, segment.length_m,
                grid.vertical_layer_optical_depth[segment.cell],
                grid.z_interface_m[base + 1] - grid.z_interface_m[base]};
        }
        result.energy = attenuate_direct_beam_path_reference(power, path, absorption);
        result.exit = ray.exit;
        result.terminal_i = ray.terminal_column / grid.ny;
        result.terminal_j = ray.terminal_column % grid.ny;
        result.terminal_x_m = ray.terminal_x_m;
        result.terminal_y_m = ray.terminal_y_m;
        result.terminal_z_m = ray.terminal_z_m;
        return true;
    }
    void store(std::size_t key, std::size_t ny, const DirectBeamRayResult& result,
               const std::vector<DirectBeamPathSegment>& path)
    {
        if (key >= rays_.size() || path.size() > segments_.capacity() - segments_.size()) return;
        auto& ray = rays_[key];
        ray.offset = segments_.size();
        ray.count = path.size();
        ray.terminal_column = result.terminal_i * ny + result.terminal_j;
        ray.terminal_x_m = result.terminal_x_m;
        ray.terminal_y_m = result.terminal_y_m;
        ray.terminal_z_m = result.terminal_z_m;
        ray.exit = result.exit;
        for (const auto& segment : path) segments_.push_back({segment.cell_index, segment.path_length_m});
        ray.ready = true;
    }
};
}

void evaluate_mapped_spectral_terrain_shortwave_reference(
    const MappedSpectralShortwaveGridView& grid,
    const IdealizedSolarGeometry& sun_direction,
    double beam_normal_irradiance_wm2,
    const double* slope_x,
    const double* slope_y,
    double horizontal_cell_area_m2,
    const LambertianMultipleReflectionOptions& options,
    MappedSpectralTerrainWorkspace& workspace,
    MappedSpectralTerrainResult& result)
{
    if (grid.nx == 0 || grid.ny == 0 || grid.nz == 0 ||
        grid.g_points == 0 || grid.g_points > 1024 ||
        grid.nx > std::numeric_limits<std::size_t>::max() / grid.ny ||
        grid.absorption_optical_depth == nullptr ||
        grid.scattering_optical_depth == nullptr ||
        grid.scattering_asymmetry == nullptr ||
        grid.solar_source_fraction == nullptr ||
        grid.z_interface_m == nullptr || slope_x == nullptr ||
        slope_y == nullptr || !std::isfinite(horizontal_cell_area_m2) ||
        horizontal_cell_area_m2 <= 0.0 ||
        !std::isfinite(options.albedo) || options.albedo < 0.0 ||
        options.albedo > 1.0 || options.quadrature.cosine_nodes == 0 ||
        options.quadrature.cosine_nodes > 32 ||
        options.quadrature.azimuth_nodes == 0 ||
        options.quadrature.azimuth_nodes > 128 ||
        options.maximum_bounces == 0 || options.maximum_bounces > 256 ||
        !std::isfinite(options.relative_residual_tolerance) ||
        options.relative_residual_tolerance < 0.0 ||
        !std::isfinite(options.absolute_residual_tolerance_w) ||
        options.absolute_residual_tolerance_w < 0.0)
        throw std::invalid_argument(
            "mapped spectral terrain shortwave has invalid fields or policy");
    const double sun_norm = std::hypot(
        std::hypot(sun_direction.east, sun_direction.north),
        sun_direction.up);
    if (!std::isfinite(sun_norm) ||
        std::abs(sun_norm - 1.0) > 1.0e-12 || sun_direction.up <= 0.0)
        throw std::invalid_argument(
            "mapped spectral terrain shortwave requires a daytime unit sun direction");
    const std::size_t column_count = grid.nx * grid.ny;
    if (column_count > std::numeric_limits<std::size_t>::max() / grid.nz)
        throw std::overflow_error(
            "mapped spectral terrain shortwave cell count overflowed");
    const std::size_t cell_count = column_count * grid.nz;
    if (cell_count > std::numeric_limits<std::size_t>::max() / grid.g_points)
        throw std::overflow_error(
            "mapped spectral terrain shortwave optical count overflowed");
    for (std::size_t column = 0; column < column_count; ++column)
    {
        if (!std::isfinite(slope_x[column]) ||
            !std::isfinite(slope_y[column]))
            throw std::invalid_argument(
                "mapped spectral terrain shortwave has invalid slopes");
        long double source_sum = 0.0L;
        for (std::size_t g = 0; g < grid.g_points; ++g)
        {
            const double source = grid.solar_source_fraction[
                column * grid.g_points + g];
            if (!std::isfinite(source) || source < 0.0)
                throw std::invalid_argument(
                    "mapped spectral terrain shortwave has invalid source weights");
            source_sum += source;
        }
        if (std::abs(source_sum - 1.0L) > 1.0e-12L)
            throw std::invalid_argument(
                "mapped spectral terrain source weights must normalize by column");
    }

    std::vector<double> mu_nodes;
    std::vector<double> mu_weights;
    gauss_legendre_unit_interval(
        options.quadrature.cosine_nodes, mu_nodes, mu_weights);
    const std::size_t direction_count =
        2 * options.quadrature.cosine_nodes *
        options.quadrature.azimuth_nodes;
    if (cell_count > std::numeric_limits<std::size_t>::max() - column_count
        || cell_count + column_count > std::numeric_limits<std::size_t>::max() / direction_count)
        throw std::overflow_error("mapped spectral angular source count overflowed");
    SpectralRayGeometryCache geometry_cache((cell_count + column_count) * direction_count,
                                           workspace.geometry_cache_budget_bytes);
    workspace.geometry_cache_bytes = geometry_cache.bytes();
    workspace.geometry_traces = 0;
    workspace.geometry_cache_hits = 0;
    workspace.directions.resize(direction_count);
    workspace.direction_weights.resize(direction_count);
    std::size_t direction_cursor = 0;
    for (int sign : {-1, 1})
        for (std::size_t mu_index = 0; mu_index < mu_nodes.size(); ++mu_index)
            for (std::size_t azimuth_index = 0;
                 azimuth_index < options.quadrature.azimuth_nodes;
                 ++azimuth_index)
            {
                const double mu = static_cast<double>(sign) * mu_nodes[mu_index];
                const double radial = std::sqrt(std::max(0.0, 1.0 - mu * mu));
                const double azimuth = 2.0 * radiation_constants::pi *
                    (static_cast<double>(azimuth_index) + 0.5) /
                    static_cast<double>(options.quadrature.azimuth_nodes);
                workspace.directions[direction_cursor] = {
                    radial * std::cos(azimuth),
                    radial * std::sin(azimuth), mu};
                workspace.direction_weights[direction_cursor] =
                    mu_weights[mu_index] /
                    (2.0 * static_cast<double>(
                        options.quadrature.azimuth_nodes));
                ++direction_cursor;
            }
    workspace.phase_probabilities.resize(direction_count);
    workspace.extinction_optical_depth.resize(cell_count);
    workspace.current_scattering.resize(cell_count);
    workspace.next_directional_scattering.resize(
        cell_count * direction_count);
    workspace.outgoing_directional_source.resize(direction_count);
    workspace.local_scattering_return.resize(cell_count);
    workspace.direct_scattering.resize(cell_count);
    workspace.current_surface_reflection.resize(column_count);
    workspace.next_surface_reflection.resize(column_count);
    workspace.scratch_atmospheric_absorbed_power_w.assign(cell_count, 0.0);
    workspace.scratch_surface_absorbed_power_w.assign(column_count, 0.0);

    long double incoming_total = 0.0L;
    long double atmospheric_absorbed_total = 0.0L;
    long double surface_absorbed_total = 0.0L;
    long double top_escape_total = 0.0L;
    long double lateral_escape_total = 0.0L;
    long double unresolved_total = 0.0L;
    std::uint64_t ray_count = 0;
    std::size_t maximum_iterations = 0;
    bool every_g_converged = true;
    const auto delta_scaled_scattering = [&](std::size_t spectral)
    {
        const double scattering = grid.scattering_optical_depth[spectral];
        const double asymmetry = grid.scattering_asymmetry[spectral];
        return scattering * (1.0 - asymmetry);
    };
    std::vector<double> isotropic_probabilities = workspace.direction_weights;
    long double phase_sum = 0.0L;
    for (double probability : isotropic_probabilities)
        phase_sum += probability;
    if (!std::isfinite(phase_sum) || phase_sum <= 0.0L)
        throw std::runtime_error("mapped spectral scattering phase failed to normalize");
    for (double& probability : isotropic_probabilities)
        probability /= static_cast<double>(phase_sum);

    for (std::size_t g = 0; g < grid.g_points; ++g)
    {
        for (std::size_t cell = 0; cell < cell_count; ++cell)
        {
            const std::size_t spectral = cell * grid.g_points + g;
            const double absorption = grid.absorption_optical_depth[spectral];
            const double scattering = grid.scattering_optical_depth[spectral];
            const double asymmetry = grid.scattering_asymmetry[spectral];
            if (!std::isfinite(absorption) || absorption < 0.0 ||
                !std::isfinite(scattering) || scattering < 0.0 ||
                !std::isfinite(asymmetry) || asymmetry < 0.0 ||
                asymmetry > 1.0)
                throw std::invalid_argument(
                    "mapped spectral terrain shortwave has invalid optical state");
            workspace.extinction_optical_depth[cell] = absorption +
                delta_scaled_scattering(spectral);
        }
        const MappedDirectBeamGridView direct_grid{
            workspace.extinction_optical_depth.data(), grid.z_interface_m,
            grid.nx, grid.ny, grid.nz, grid.dx_m, grid.dy_m,
            grid.x_boundary, grid.y_boundary};
        const ValidatedMappedDirectBeamContext spectral_context =
            validate_mapped_direct_beam_context(
                direct_grid, {0.0, 0.0, 1.0});
        workspace.direct_workspace.boundary_irradiance_wm2.resize(column_count);
        for (std::size_t column = 0; column < column_count; ++column)
            workspace.direct_workspace.boundary_irradiance_wm2[column] =
                beam_normal_irradiance_wm2 *
                grid.solar_source_fraction[column * grid.g_points + g];
        evaluate_mapped_direct_beam_field_spatial_reference(
            direct_grid, sun_direction,
            workspace.direct_workspace.boundary_irradiance_wm2.data(),
            workspace.direct_workspace.direct_workspace,
            workspace.direct_workspace.direct_result);
        const auto& direct = workspace.direct_workspace.direct_result;
        const long double g_incoming =
            direct.incoming_top_power_w + direct.incoming_side_power_w;
        incoming_total += g_incoming;
        lateral_escape_total += direct.lateral_escape_power_w;
        std::fill(workspace.current_scattering.begin(),
                  workspace.current_scattering.end(), 0.0);
        std::fill(workspace.direct_scattering.begin(),
                  workspace.direct_scattering.end(), 0.0);
        std::fill(workspace.current_surface_reflection.begin(),
                  workspace.current_surface_reflection.end(), 0.0);
        for (std::size_t cell = 0; cell < cell_count; ++cell)
        {
            const std::size_t spectral = cell * grid.g_points + g;
            const double absorption_depth =
                grid.absorption_optical_depth[spectral];
            const double scattering_depth =
                delta_scaled_scattering(spectral);
            const double extinction = absorption_depth + scattering_depth;
            const double removed = direct.cell_absorbed_power_w[cell];
            const double absorbed = extinction > 0.0
                ? removed * absorption_depth / extinction : 0.0;
            const double scattered = removed - absorbed;
            workspace.scratch_atmospheric_absorbed_power_w[cell] += absorbed;
            workspace.direct_scattering[cell] = scattered;
            atmospheric_absorbed_total += absorbed;
        }
        for (std::size_t column = 0; column < column_count; ++column)
        {
            const double incident = direct.surface_incident_power_w[column];
            const double absorbed = (1.0 - options.albedo) * incident;
            workspace.scratch_surface_absorbed_power_w[column] += absorbed;
            workspace.current_surface_reflection[column] = incident - absorbed;
            surface_absorbed_total += absorbed;
        }

        const auto trace_ray = [&](bool surface, std::size_t source_index,
                                   std::size_t out_direction, double power)
        {
            const std::size_t key = ((surface ? cell_count : 0) + source_index) * direction_count + out_direction;
            DirectBeamRayResult ray;
            if (geometry_cache.load(key, direct_grid, power, workspace.ray_segments,
                                    workspace.ray_absorption, ray))
            {
                ++workspace.geometry_cache_hits;
                return ray;
            }
            const auto& direction = workspace.directions[out_direction];
            const std::size_t column = surface ? source_index : source_index / grid.nz;
            const std::size_t k = surface ? 0 : source_index % grid.nz;
            const std::size_t i = column / grid.ny, j = column % grid.ny;
            const std::size_t base = column * (grid.nz + 1);
            const double lower = grid.z_interface_m[base + k];
            const double upper = grid.z_interface_m[base + k + 1];
            const double z = surface ? std::nextafter(lower, upper) : lower + 0.5 * (upper - lower);
            const double x = (static_cast<double>(i) + 0.5) * grid.dx_m;
            const double y = (static_cast<double>(j) + 0.5) * grid.dy_m;
            ValidatedMappedDirectBeamContext ray_context = spectral_context;
            ray_context.maximum_iterations = mapped_radiative_ray_iteration_bound(
                direct_grid, ray_context, x, y, z, direction);
            ray = trace_mapped_direct_beam_from_entry_reference_impl(
                direct_grid, x, y, z, k, direction, power,
                workspace.ray_segments, workspace.ray_absorption, ray_context, true, true);
            ++workspace.geometry_traces;
            geometry_cache.store(key, grid.ny, ray, workspace.ray_segments);
            return ray;
        };

        for (std::size_t cell = 0; cell < cell_count; ++cell)
        {
            long double local_return = 0.0L;
            for (std::size_t direction = 0;
                 direction < direction_count; ++direction)
            {
                const double bundle = workspace.direction_weights[direction];
                const DirectBeamRayResult probe =
                    trace_ray(false, cell, direction, bundle);
                (void)probe;
                for (const DirectBeamSegmentAbsorption& segment :
                     workspace.ray_absorption)
                {
                    if (segment.cell_index != cell)
                        continue;
                    const std::size_t spectral =
                        cell * grid.g_points + g;
                    const double absorption_depth =
                        grid.absorption_optical_depth[spectral];
                    const double scattering_depth =
                        delta_scaled_scattering(spectral);
                    const double extinction =
                        absorption_depth + scattering_depth;
                    if (extinction > 0.0)
                        local_return += segment.absorbed_power_w *
                            scattering_depth / extinction;
                }
            }
            if (!std::isfinite(local_return) || local_return < 0.0L ||
                local_return >= 1.0L)
                throw std::runtime_error(
                    "mapped spectral local scattering preconditioner is singular");
            workspace.local_scattering_return[cell] =
                static_cast<double>(local_return);
            workspace.direct_scattering[cell] /=
                1.0 - workspace.local_scattering_return[cell];
        }

        const auto trace_and_commit = [&](
            bool surface_source, std::size_t source_index,
            std::size_t out_direction, double power)
        {
            if (power == 0.0)
                return;
            const DirectBeamRayResult ray = trace_ray(surface_source, source_index, out_direction, power);
            ++ray_count;
            for (const DirectBeamSegmentAbsorption& segment :
                 workspace.ray_absorption)
            {
                const std::size_t spectral =
                    segment.cell_index * grid.g_points + g;
                const double absorption_depth =
                    grid.absorption_optical_depth[spectral];
                const double scattering_depth =
                    delta_scaled_scattering(spectral);
                const double extinction = absorption_depth + scattering_depth;
                const double absorbed = extinction > 0.0
                    ? segment.absorbed_power_w * absorption_depth / extinction
                    : 0.0;
                const double scattered = segment.absorbed_power_w - absorbed;
                workspace.scratch_atmospheric_absorbed_power_w[
                    segment.cell_index] += absorbed;
                if (surface_source || segment.cell_index != source_index)
                    workspace.next_directional_scattering[
                        segment.cell_index * direction_count + out_direction] +=
                        scattered;
                atmospheric_absorbed_total += absorbed;
            }
            if (ray.exit == DirectBeamRayExit::Surface)
            {
                const std::size_t column = ray.terminal_i * grid.ny +
                    ray.terminal_j;
                const double absorbed = (1.0 - options.albedo) *
                    ray.energy.transmitted_power_w;
                workspace.scratch_surface_absorbed_power_w[column] += absorbed;
                workspace.next_surface_reflection[column] +=
                    ray.energy.transmitted_power_w - absorbed;
                surface_absorbed_total += absorbed;
            }
            else if (ray.exit == DirectBeamRayExit::LevelTop)
                top_escape_total += ray.energy.transmitted_power_w;
            else
                lateral_escape_total += ray.energy.transmitted_power_w;
        };

        bool g_converged = false;
        std::size_t completed = 0;
        for (std::size_t iteration = 0;
             iteration < options.maximum_bounces; ++iteration)
        {
            long double source_power = 0.0L;
            for (double value : workspace.direct_scattering)
                source_power += value;
            for (double value : workspace.current_scattering)
                source_power += value;
            for (double value : workspace.current_surface_reflection)
                source_power += value;
            const double tolerance = std::max(
                options.absolute_residual_tolerance_w,
                options.relative_residual_tolerance *
                    std::max(1.0, static_cast<double>(g_incoming)));
            if (source_power <= tolerance)
            {
                g_converged = true;
                break;
            }
            std::fill(workspace.next_directional_scattering.begin(),
                      workspace.next_directional_scattering.end(), 0.0);
            std::fill(workspace.next_surface_reflection.begin(),
                      workspace.next_surface_reflection.end(), 0.0);

            const auto distribute_probabilities = [&](
                double power, double* destination)
            {
                long double emitted = 0.0L;
                std::size_t last = direction_count;
                for (std::size_t direction = 0;
                     direction < direction_count; ++direction)
                    if (workspace.phase_probabilities[direction] > 0.0)
                        last = direction;
                if (last == direction_count)
                    throw std::runtime_error(
                        "mapped spectral source has no outgoing direction");
                for (std::size_t direction = 0;
                     direction < direction_count; ++direction)
                {
                    const double probability =
                        workspace.phase_probabilities[direction];
                    if (probability == 0.0)
                        continue;
                    double bundle = 0.0;
                    if (direction == last)
                        bundle = static_cast<double>(
                            static_cast<long double>(power) - emitted);
                    else
                    {
                        bundle = power * probability;
                        emitted += bundle;
                    }
                    if (!std::isfinite(bundle) || bundle < 0.0)
                        throw std::runtime_error(
                            "mapped spectral angular source failed to close");
                    destination[direction] += bundle;
                    if (!std::isfinite(destination[direction]))
                        throw std::overflow_error(
                            "mapped spectral outgoing source overflowed");
                }
            };

            workspace.phase_probabilities = isotropic_probabilities;
            for (std::size_t cell = 0; cell < cell_count; ++cell)
            {
                // The isotropic phase carries one incoming source per cell.
                const long double power = static_cast<long double>(workspace.direct_scattering[cell])
                    + workspace.current_scattering[cell];
                std::fill(workspace.outgoing_directional_source.begin(),
                          workspace.outgoing_directional_source.end(), 0.0);
                if (power > 0.0L)
                    distribute_probabilities(static_cast<double>(power),
                                             workspace.outgoing_directional_source.data());
                workspace.direct_scattering[cell] = 0.0;
                for (std::size_t direction = 0; direction < direction_count; ++direction)
                {
                    const double bundle = workspace.outgoing_directional_source[direction];
                    if (bundle > 0.0) trace_and_commit(false, cell, direction, bundle);
                }
            }
            for (std::size_t column = 0; column < column_count; ++column)
            {
                const double power = workspace.current_surface_reflection[column];
                if (power <= 0.0)
                    continue;
                const double normal_scale = std::hypot(
                    std::hypot(slope_x[column], slope_y[column]), 1.0);
                const IdealizedSolarGeometry normal{
                    -slope_x[column] / normal_scale,
                    -slope_y[column] / normal_scale,
                    1.0 / normal_scale};
                long double probability_sum = 0.0L;
                for (std::size_t direction = 0;
                     direction < direction_count; ++direction)
                {
                    const auto& outgoing = workspace.directions[direction];
                    const double cosine = std::max(0.0,
                        normal.east * outgoing.east +
                        normal.north * outgoing.north +
                        normal.up * outgoing.up);
                    workspace.phase_probabilities[direction] = cosine *
                        workspace.direction_weights[direction];
                    probability_sum +=
                        workspace.phase_probabilities[direction];
                }
                if (!std::isfinite(probability_sum) || probability_sum <= 0.0L)
                    throw std::runtime_error(
                        "mapped spectral Lambertian source failed to normalize");
                for (double& probability : workspace.phase_probabilities)
                    probability /= static_cast<double>(probability_sum);
                std::fill(workspace.outgoing_directional_source.begin(),
                          workspace.outgoing_directional_source.begin() +
                              direction_count, 0.0);
                distribute_probabilities(
                    power, workspace.outgoing_directional_source.data());
                for (std::size_t direction = 0;
                     direction < direction_count; ++direction)
                {
                    const double bundle =
                        workspace.outgoing_directional_source[direction];
                    if (bundle > 0.0)
                        trace_and_commit(true, column, direction, bundle);
                }
            }
            for (std::size_t cell = 0; cell < cell_count; ++cell)
            {
                long double off_diagonal_source = 0.0L;
                for (std::size_t direction = 0;
                     direction < direction_count; ++direction)
                    off_diagonal_source +=
                        workspace.next_directional_scattering[
                            cell * direction_count + direction];
                workspace.current_scattering[cell] = static_cast<double>(
                    off_diagonal_source / (1.0 - workspace.local_scattering_return[cell]));
            }
            workspace.current_surface_reflection.swap(
                workspace.next_surface_reflection);
            ++completed;
        }
        long double unresolved = 0.0L;
        for (std::size_t cell = 0; cell < cell_count; ++cell)
        {
            unresolved += workspace.direct_scattering[cell] *
                (1.0 - workspace.local_scattering_return[cell]);
            const long double directional = workspace.current_scattering[cell];
            unresolved += directional *
                (1.0 - workspace.local_scattering_return[cell]);
        }
        for (double value : workspace.current_surface_reflection)
            unresolved += value;
        const double final_tolerance = std::max(
            options.absolute_residual_tolerance_w,
            options.relative_residual_tolerance *
                std::max(1.0, static_cast<double>(g_incoming)));
        if (unresolved <= final_tolerance)
            g_converged = true;
        unresolved_total += unresolved;
        maximum_iterations = std::max(maximum_iterations, completed);
        every_g_converged = every_g_converged && g_converged;
    }

    const long double accounted = atmospheric_absorbed_total +
        surface_absorbed_total + top_escape_total + lateral_escape_total +
        unresolved_total;
    const long double residual = incoming_total - accounted;
    const long double closure_tolerance = 16384.0L *
        std::numeric_limits<double>::epsilon() *
        std::max({1.0L, std::abs(incoming_total), std::abs(accounted)}) *
        std::max<std::uint64_t>(1, ray_count);
    if (std::abs(residual) > closure_tolerance)
        throw std::runtime_error(
            "mapped spectral terrain shortwave power does not close");

    std::swap(result.atmospheric_absorbed_power_w,
              workspace.scratch_atmospheric_absorbed_power_w);
    std::swap(result.surface_absorbed_power_w,
              workspace.scratch_surface_absorbed_power_w);
    result.incoming_power_w = static_cast<double>(incoming_total);
    result.atmospheric_absorbed_power_w_total =
        static_cast<double>(atmospheric_absorbed_total);
    result.surface_absorbed_power_w_total =
        static_cast<double>(surface_absorbed_total);
    result.top_escape_power_w = static_cast<double>(top_escape_total);
    result.lateral_escape_power_w = static_cast<double>(lateral_escape_total);
    result.unresolved_scattered_power_w = static_cast<double>(unresolved_total);
    result.closure_residual_w = static_cast<double>(residual);
    result.rays = ray_count;
    result.completed_iterations = maximum_iterations;
    result.converged = every_g_converged;
}

/**
 * @brief Computes the heating rate from the flux divergence.
 */
void heating_rate_from_flux_divergence(
    const std::vector<double>& rho,
    const std::vector<double>& dz,
    const std::vector<double>& Fnet,
    std::vector<double>& dTdt
) 
{
    const size_t nz = rho.size();
    dTdt.resize(nz);
    if (dz.size() != nz || Fnet.size() != nz + 1)
    {
        throw std::invalid_argument(
            "heating_rate_from_flux_divergence: inconsistent column sizes");
    }

    constexpr double cp = physical_constants::specific_heat_cp_jkgk;

    for (size_t k = 0; k < nz; ++k) 
    {
        if (rho[k] <= 0.0 || !std::isfinite(rho[k]) || dz[k] <= 0.0 || !std::isfinite(dz[k]))
        {
            throw std::invalid_argument(
                "heating_rate_from_flux_divergence: density and dz must be positive and finite");
        }
        double dFdz = (Fnet[k+1] - Fnet[k]) / dz[k];
        dTdt[k] = -dFdz / (rho[k] * cp);
        if (!std::isfinite(dTdt[k]))
        {
            throw std::runtime_error(
                "heating_rate_from_flux_divergence produced a non-finite tendency");
        }
    }
}

/**
 * @brief Computes the grey longwave two stream solution.
 */
void grey_lw_two_stream(
    const std::vector<double>& T,
    const std::vector<double>& tau_layers,
    double T_sfc,
    double emissivity,
    std::vector<double>& Fup,
    std::vector<double>& Fdn
) 
{
    const size_t nz = T.size();
    Fup.resize(nz + 1);
    Fdn.resize(nz + 1);
    if (tau_layers.size() != nz)
    {
        throw std::invalid_argument("grey_lw_two_stream: inconsistent column sizes");
    }
    if (!std::isfinite(T_sfc) || T_sfc <= 0.0
        || !std::isfinite(emissivity) || emissivity < 0.0 || emissivity > 1.0)
    {
        throw std::invalid_argument(
            "grey_lw_two_stream: invalid surface temperature or emissivity");
    }
    for (size_t k = 0; k < nz; ++k)
    {
        if (!std::isfinite(T[k]) || T[k] <= 0.0
            || !std::isfinite(tau_layers[k]) || tau_layers[k] < 0.0)
        {
            throw std::invalid_argument(
                "grey_lw_two_stream: temperature and optical depth must be physical");
        }
    }

    Fdn[nz] = 0.0;
    for (int k = static_cast<int>(nz) - 1; k >= 0; --k) 
    {
        double tau_k = tau_layers[k];
        double B_k = planck_function(T[k]);


        double exp_tau = std::exp(-tau_k);
        Fdn[k] = Fdn[k+1] * exp_tau + radiation_constants::pi * B_k * (1.0 - exp_tau);
    }

    // The surface emits epsilon*sigma*T^4 and reflects the fraction
    // (1-epsilon) of the atmospheric downwelling longwave. Fdn[0] must be
    // known before this boundary condition is evaluated.
    Fup[0] = emissivity * stefan_boltzmann(T_sfc) +
             (1.0 - emissivity) * Fdn[0];

    for (size_t k = 0; k < nz; ++k) 
    {
        double tau_k = tau_layers[k];
        double B_k = planck_function(T[k]);

        double exp_tau = std::exp(-tau_k);
        Fup[k+1] = Fup[k] * exp_tau + radiation_constants::pi * B_k * (1.0 - exp_tau);
    }
}

/**
 * @brief Computes the grey shortwave two stream solution.
 */
void grey_sw_beer_lambert(
    const std::vector<double>& tau_layers,
    double mu0,
    double S0,
    double albedo_sfc,
    std::vector<double>& Fup,
    std::vector<double>& Fdn
) 
{
    const size_t nz = tau_layers.size();
    Fup.resize(nz + 1);
    Fdn.resize(nz + 1);
    if (nz == 0)
    {
        Fup[0] = 0.0;
        Fdn[0] = 0.0;
        return;
    }
    if (!std::isfinite(mu0) || mu0 <= 0.0 || mu0 > 1.0
        || !std::isfinite(S0) || S0 < 0.0
        || !std::isfinite(albedo_sfc) || albedo_sfc < 0.0 || albedo_sfc > 1.0)
    {
        throw std::invalid_argument(
            "grey_sw_beer_lambert: invalid solar geometry, irradiance, or albedo");
    }
    for (double tau : tau_layers)
    {
        if (!std::isfinite(tau) || tau < 0.0)
        {
            throw std::invalid_argument(
                "grey_sw_beer_lambert: optical depth must be finite and non-negative");
        }
    }

    Fdn[nz] = mu0 * S0;
    Fup[nz] = 0.0;

    for (int k = static_cast<int>(nz) - 1; k >= 0; --k) 
    {
        const double tau_k = tau_layers[k];
        Fdn[k] = Fdn[k + 1] * exp(-tau_k / mu0);
    }

    Fup[0] = albedo_sfc * Fdn[0];

    // The downward beam is collimated and therefore follows mu0. The reflected
    // boundary is Lambertian: its upward radiance is isotropic, so the flux at
    // each interface is the hemispheric integral 2*E3(cumulative_tau).
    double cumulative_optical_depth = 0.0;
    for (size_t k = 0; k < nz; ++k)
    {
        cumulative_optical_depth += tau_layers[k];
        Fup[k + 1] = Fup[0]
            * lambertian_diffuse_transmittance(
                cumulative_optical_depth);
    }
}

/**
 * @brief Computes the optical depth profile.
 */
std::vector<double> compute_optical_depth_profile(
    const std::vector<double>& p_interfaces,
    double tau_ref,
    double n
) 
{
    std::vector<double> tau_profile;
    compute_optical_depth_profile(p_interfaces, tau_ref, n, tau_profile);
    return tau_profile;
}

void compute_optical_depth_profile(
    const std::vector<double>& p_interfaces,
    double tau_ref,
    double n,
    std::vector<double>& tau_profile)
{
    const size_t interface_count = p_interfaces.size();
    tau_profile.resize(interface_count);
    if (interface_count == 0)
    {
        return;
    }
    if (!std::isfinite(tau_ref) || tau_ref < 0.0 || !std::isfinite(n) || n <= 0.0)
    {
        throw std::invalid_argument(
            "compute_optical_depth_profile: invalid reference depth or exponent");
    }
    for (size_t k = 0; k < interface_count; ++k)
    {
        if (!std::isfinite(p_interfaces[k]) || p_interfaces[k] < 0.0
            || (k == 0 && p_interfaces[k] <= 0.0)
            || (k > 0 && p_interfaces[k] > p_interfaces[k - 1]))
        {
            throw std::invalid_argument(
                "compute_optical_depth_profile: interface pressure must be finite, non-negative, non-increasing, and positive at the surface");
        }
    }

    const double p_ref = 100000.0;
    for (size_t k = 0; k < interface_count; ++k)
    {
        const double pressure_ratio = p_interfaces[k] / p_ref;
        tau_profile[k] = tau_ref * std::pow(pressure_ratio, n);
    }
}

std::vector<double> diagnose_pressure_interfaces(
    const std::vector<double>& p_center,
    const std::vector<double>& z_center,
    const std::vector<double>& z_interfaces)
{
    std::vector<double> result;
    diagnose_pressure_interfaces(
        p_center, z_center, z_interfaces, result);
    return result;
}

void diagnose_pressure_interfaces(
    const std::vector<double>& p_center,
    const std::vector<double>& z_center,
    const std::vector<double>& z_interfaces,
    std::vector<double>& result)
{
    const std::size_t nz = p_center.size();
    if (nz == 0 || z_center.size() != nz || z_interfaces.size() != nz + 1)
    {
        throw std::invalid_argument(
            "diagnose_pressure_interfaces: inconsistent column sizes");
    }
    for (std::size_t k = 0; k < nz; ++k)
    {
        if (!std::isfinite(p_center[k]) || p_center[k] <= 0.0 ||
            !std::isfinite(z_center[k]) ||
            (k > 0 && (p_center[k] > p_center[k - 1] ||
                       z_center[k] <= z_center[k - 1])))
        {
            throw std::invalid_argument(
                "diagnose_pressure_interfaces: center pressure/height must be physical and monotonic");
        }
    }
    for (std::size_t k = 0; k <= nz; ++k)
    {
        if (!std::isfinite(z_interfaces[k]) ||
            (k > 0 && z_interfaces[k] <= z_interfaces[k - 1]))
        {
            throw std::invalid_argument(
                "diagnose_pressure_interfaces: interface heights must be finite and strictly increasing");
        }
    }
    if (!(z_interfaces.front() < z_center.front()) ||
        !(z_interfaces.back() > z_center.back()))
    {
        throw std::invalid_argument(
            "diagnose_pressure_interfaces: interfaces must bracket scalar centers");
    }

    result.resize(nz + 1);
    if (nz == 1)
    {
        throw std::invalid_argument(
            "diagnose_pressure_interfaces: at least two centers are required for bounded log-pressure extrapolation");
    }

    const auto log_pressure_at = [&](double z, std::size_t lower,
                                     std::size_t upper)
    {
        const double fraction =
            (z - z_center[lower]) /
            (z_center[upper] - z_center[lower]);
        return std::exp(
            std::log(p_center[lower]) + fraction *
            (std::log(p_center[upper]) - std::log(p_center[lower])));
    };
    result.front() = log_pressure_at(z_interfaces.front(), 0, 1);
    for (std::size_t k = 1; k < nz; ++k)
    {
        result[k] = log_pressure_at(z_interfaces[k], k - 1, k);
    }
    result.back() = log_pressure_at(z_interfaces.back(), nz - 2, nz - 1);
    for (std::size_t k = 0; k <= nz; ++k)
    {
        if (!std::isfinite(result[k]) || result[k] <= 0.0 ||
            (k > 0 && result[k] > result[k - 1]))
        {
            throw std::runtime_error(
                "diagnose_pressure_interfaces produced a non-physical pressure boundary");
        }
    }
}

/**
 * @brief Computes the layer optical depths.
 */
std::vector<double> compute_layer_optical_depths(
    const std::vector<double>& tau_profile,
    const std::vector<double>& dz
) 
{
    std::vector<double> dtau;
    compute_layer_optical_depths(tau_profile, dz, dtau);
    return dtau;
}

void compute_layer_optical_depths(
    const std::vector<double>& tau_profile,
    const std::vector<double>& dz,
    std::vector<double>& dtau)
{
    const size_t nz = dz.size();
    dtau.resize(nz);
    if (tau_profile.size() != nz + 1)
    {
        throw std::invalid_argument(
            "compute_layer_optical_depths: inconsistent column sizes");
    }

    for (size_t k = 0; k < nz; ++k) 
    {
        if (!std::isfinite(dz[k]) || dz[k] <= 0.0
            || !std::isfinite(tau_profile[k]) || tau_profile[k] < 0.0
            || !std::isfinite(tau_profile[k + 1]) || tau_profile[k + 1] < 0.0
            || tau_profile[k + 1] > tau_profile[k])
        {
            throw std::invalid_argument(
                "compute_layer_optical_depths: dz/depth profile is invalid or non-monotonic");
        }
        dtau[k] = tau_profile[k] - tau_profile[k + 1];
    }
}

/**
 * @brief Checks the energy conservation.
 */
double check_energy_conservation(
    const std::vector<double>& rho,
    const std::vector<double>& dz,
    const std::vector<double>& dTdt,
    const std::vector<double>& Fnet
) 

{
    const size_t nz = rho.size();
    if (dz.size() != nz || dTdt.size() != nz || Fnet.size() != nz + 1)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    constexpr double cp = physical_constants::specific_heat_cp_jkgk;

    double integrated_heating = 0.0;

    for (size_t k = 0; k < nz; ++k) 
    {
        integrated_heating += rho[k] * cp * dTdt[k] * dz[k];
    }

    const double net_flux_diff = Fnet[nz] - Fnet[0];

    // rho*cp*dT/dt = -dFnet/dz, so the column integral must equal
    // -(Fnet_top - Fnet_surface).
    return integrated_heating + net_flux_diff;
}

} // namespace tmv::radiative_transfer
