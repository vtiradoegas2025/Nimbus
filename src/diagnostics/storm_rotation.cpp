#include "diagnostics/column_integrals.hpp"
#include "diagnostics/storm_rotation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tmv::diagnostics
{
namespace
{

bool same_shape(const Field3D& lhs, const Field3D& rhs) noexcept
{
    return lhs.size_r() == rhs.size_r() && lhs.size_th() == rhs.size_th() && lhs.size_z() == rhs.size_z();
}

void validate_view(const StormRotationGridView& grid,
                   const StormRotationLayer&    layer,
                   const std::vector<double>&   output)
{
    if (grid.u == nullptr || grid.v == nullptr || grid.w == nullptr || grid.domain.empty())
    {
        throw std::invalid_argument("storm updraft helicity requires non-empty velocity and physical-domain views");
    }
    if (!same_shape(*grid.u, *grid.v) || !same_shape(*grid.u, *grid.w))
    {
        throw std::invalid_argument("storm updraft-helicity velocity fields must have identical storage shapes");
    }
    const int nr = grid.u->size_r();
    const int ny = grid.u->size_th();
    const int nz = grid.u->size_z();
    if (grid.domain.i_begin < 1 || grid.domain.i_end > nr - 1 || grid.domain.j_begin < 1 ||
        grid.domain.j_end > ny - 1 || grid.domain.k_begin < 0 || grid.domain.k_end > nz ||
        grid.domain.k_begin >= grid.domain.k_end)
    {
        throw std::invalid_argument(
            "storm updraft-helicity physical domain lacks the horizontal derivative closure");
    }
    if (!std::isfinite(grid.dx_m) || grid.dx_m <= 0.0 || !std::isfinite(grid.dy_m) || grid.dy_m <= 0.0 ||
        !std::isfinite(grid.dz_m) || grid.dz_m <= 0.0 || !std::isfinite(layer.bottom_m_agl) ||
        !std::isfinite(layer.top_m_agl) || layer.bottom_m_agl < 0.0 || layer.top_m_agl <= layer.bottom_m_agl)
    {
        throw std::invalid_argument("storm updraft-helicity geometry or AGL layer is invalid");
    }
    const std::size_t column_count =
        static_cast<std::size_t>(grid.domain.nx()) * static_cast<std::size_t>(grid.domain.ny());
    if (output.size() != column_count)
    {
        throw std::invalid_argument("storm updraft-helicity output must be pre-sized to the physical column count");
    }
    if ((grid.flat_z_center_m == nullptr) != (grid.flat_z_interface_m == nullptr))
    {
        throw std::invalid_argument("storm updraft-helicity flat vertical geometry is only partially initialized");
    }
    if (grid.flat_z_center_m != nullptr &&
        (grid.flat_z_center_m->size() != static_cast<std::size_t>(nz) ||
         grid.flat_z_interface_m->size() != static_cast<std::size_t>(nz + 1)))
    {
        throw std::invalid_argument("storm updraft-helicity flat vertical geometry has the wrong length");
    }
    if (grid.terrain != nullptr &&
        (grid.terrain->z.size_r() != nr || grid.terrain->z.size_th() != ny || grid.terrain->z.size_z() != nz ||
         grid.terrain->z_face.size_r() != nr || grid.terrain->z_face.size_th() != ny ||
         grid.terrain->z_face.size_z() != nz + 1))
    {
        throw std::invalid_argument("storm updraft-helicity terrain geometry does not match velocity storage");
    }
    if (grid.staggering == StaggerType::CGrid && grid.terrain != nullptr && grid.domain.k_begin == 0 &&
        (grid.cgrid_ground_w_ms == nullptr ||
         grid.cgrid_ground_w_ms->size() != static_cast<std::size_t>(nr) * static_cast<std::size_t>(ny)))
    {
        throw std::invalid_argument("terrain C-grid updraft helicity requires complete physical ground velocity");
    }
}

double scalar_height(const StormRotationGridView& grid, int i, int j, int k) noexcept
{
    if (grid.terrain != nullptr)
        return grid.terrain->z(i, j, k);
    if (grid.flat_z_center_m != nullptr)
        return (*grid.flat_z_center_m)[static_cast<std::size_t>(k)];
    return (static_cast<double>(k) + 0.5) * grid.dz_m;
}

double interface_height(const StormRotationGridView& grid, int i, int j, int k) noexcept
{
    if (grid.terrain != nullptr)
        return grid.terrain->z_face(i, j, k);
    if (grid.flat_z_interface_m != nullptr)
        return (*grid.flat_z_interface_m)[static_cast<std::size_t>(k)];
    return static_cast<double>(k) * grid.dz_m;
}

double centered_u(const StormRotationGridView& grid, int i, int j, int k) noexcept
{
    if (grid.staggering == StaggerType::CGrid)
        return 0.5 * (static_cast<double>((*grid.u)(i - 1, j, k)) + static_cast<double>((*grid.u)(i, j, k)));
    return static_cast<double>((*grid.u)(i, j, k));
}

double centered_v(const StormRotationGridView& grid, int i, int j, int k) noexcept
{
    if (grid.staggering == StaggerType::CGrid)
        return 0.5 * (static_cast<double>((*grid.v)(i, j - 1, k)) + static_cast<double>((*grid.v)(i, j, k)));
    return static_cast<double>((*grid.v)(i, j, k));
}

double centered_w(const StormRotationGridView& grid, int i, int j, int k) noexcept
{
    if (grid.staggering != StaggerType::CGrid)
        return static_cast<double>((*grid.w)(i, j, k));
    double lower = 0.0;
    if (k > 0)
        lower = static_cast<double>((*grid.w)(i, j, k - 1));
    else if (grid.terrain != nullptr)
        lower = (*grid.cgrid_ground_w_ms)[static_cast<std::size_t>(i) *
                                              static_cast<std::size_t>(grid.w->size_th()) +
                                          static_cast<std::size_t>(j)];
    return 0.5 * (lower + static_cast<double>((*grid.w)(i, j, k)));
}

template <typename ValueAt>
double vertical_gradient(const StormRotationGridView& grid,
                         const ValueAt&                value_at,
                         int                           i,
                         int                           j,
                         int                           k) noexcept
{
    const int lower = k > grid.domain.k_begin ? k - 1 : k;
    const int upper = k + 1 < grid.domain.k_end ? k + 1 : k;
    if (lower == upper)
        return 0.0;
    const double dz = scalar_height(grid, i, j, upper) - scalar_height(grid, i, j, lower);
    if (!std::isfinite(dz) || dz <= 0.0)
        return std::numeric_limits<double>::quiet_NaN();
    return (value_at(i, j, upper) - value_at(i, j, lower)) / dz;
}

double vertical_vorticity(const StormRotationGridView& grid, int i, int j, int k) noexcept
{
    const auto u_at = [&](int ii, int jj, int kk) { return centered_u(grid, ii, jj, kk); };
    const auto v_at = [&](int ii, int jj, int kk) { return centered_v(grid, ii, jj, kk); };

    const double dv_dx_zeta = (v_at(i + 1, j, k) - v_at(i - 1, j, k)) / (2.0 * grid.dx_m);
    const double du_dy_zeta = (u_at(i, j + 1, k) - u_at(i, j - 1, k)) / (2.0 * grid.dy_m);
    if (grid.terrain == nullptr)
        return dv_dx_zeta - du_dy_zeta;

    const double dz_dx_zeta =
        (scalar_height(grid, i + 1, j, k) - scalar_height(grid, i - 1, j, k)) / (2.0 * grid.dx_m);
    const double dz_dy_zeta =
        (scalar_height(grid, i, j + 1, k) - scalar_height(grid, i, j - 1, k)) / (2.0 * grid.dy_m);
    const double dv_dz = vertical_gradient(grid, v_at, i, j, k);
    const double du_dz = vertical_gradient(grid, u_at, i, j, k);
    return (dv_dx_zeta - dz_dx_zeta * dv_dz) - (du_dy_zeta - dz_dy_zeta * du_dz);
}

} // namespace

const char* storm_updraft_helicity_algorithm_id() noexcept
{
    return "physical_height_midpoint_uh_2d_v1";
}

void compute_storm_updraft_helicity_columns(const StormRotationGridView& grid,
                                            const StormRotationLayer&    layer,
                                            std::vector<double>&         output_m2s2)
{
    validate_view(grid, layer, output_m2s2);
    const std::size_t ny = static_cast<std::size_t>(grid.domain.ny());
    const double      requested_depth_m = layer.top_m_agl - layer.bottom_m_agl;

#pragma omp parallel for collapse(2) if (output_m2s2.size() >= 4096)
    for (int i = grid.domain.i_begin; i < grid.domain.i_end; ++i)
        for (int j = grid.domain.j_begin; j < grid.domain.j_end; ++j)
        {
            const std::size_t index = static_cast<std::size_t>(i - grid.domain.i_begin) * ny +
                                      static_cast<std::size_t>(j - grid.domain.j_begin);
            const double ground_m = interface_height(grid, i, j, 0);
            const double layer_bottom_m = ground_m + layer.bottom_m_agl;
            const double layer_top_m    = ground_m + layer.top_m_agl;
            const auto integral = integrate_cell_layer(
                grid.domain.k_begin, grid.domain.k_end, layer_bottom_m, layer_top_m,
                [&](int k) { return interface_height(grid, i, j, k); },
                [&](int k) { return centered_w(grid, i, j, k) * vertical_vorticity(grid, i, j, k); });
            const double coverage_tolerance =
                128.0 * std::numeric_limits<double>::epsilon() *
                std::max({1.0, requested_depth_m, std::abs(layer_top_m)});
            const bool valid = std::isfinite(ground_m) && integral.valid &&
                std::abs(integral.covered_depth_m - requested_depth_m) <= coverage_tolerance;
            output_m2s2[index] = valid ? integral.value : std::numeric_limits<double>::quiet_NaN();
        }

    if (std::any_of(output_m2s2.begin(), output_m2s2.end(), [](double value) { return !std::isfinite(value); }))
    {
        throw std::runtime_error(
            "storm updraft-helicity sample has invalid velocity/geometry or incomplete requested AGL coverage");
    }
}

} // namespace tmv::diagnostics
