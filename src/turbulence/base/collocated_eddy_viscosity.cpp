#include "collocated_eddy_viscosity.hpp"

#include "core/infra/physical_cell_domain.hpp"
#include "terrain/conservative_transport.hpp"
#include "util/grid_metric_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tmv::collocated_eddy_viscosity
{
namespace
{

using terrain_transport::PhysicalVector3;
using terrain_transport::SymmetricStressTensor;

struct Domain
{
    PhysicalCellDomain cells;
    bool periodic_x = false;
    bool periodic_y = false;

    int previous_i(int i) const noexcept
    {
        if (i > cells.i_begin) return i - 1;
        return periodic_x ? cells.i_end - 1 : -1;
    }
    int next_i(int i) const noexcept
    {
        if (i + 1 < cells.i_end) return i + 1;
        return periodic_x ? cells.i_begin : -1;
    }
    int previous_j(int j) const noexcept
    {
        if (j > cells.j_begin) return j - 1;
        return periodic_y ? cells.j_end - 1 : -1;
    }
    int next_j(int j) const noexcept
    {
        if (j + 1 < cells.j_end) return j + 1;
        return periodic_y ? cells.j_begin : -1;
    }
};

bool matches(const Field3D& field, const TurbulenceStateView& state) noexcept
{
    return field.size_r() == state.NR && field.size_th() == state.NTH
        && field.size_z() == state.NZ;
}

Domain validate_contract(const TurbulenceConfig& config,
                         const TurbulenceStateView& state,
                         const GridMetrics& grid,
                         const Field3D& coefficient,
                         bool periodic_x,
                         bool periodic_y)
{
    if (state.c_grid || grid.staggering != StaggerType::Collocated)
        throw std::invalid_argument(
            "collocated eddy viscosity requires Cartesian A-grid storage");
    if (grid.terrain_metrics_active && grid.terrain_metrics == nullptr)
        throw std::invalid_argument(
            "mapped collocated eddy viscosity requires complete terrain metrics");
    if (config.mode != "3d" && config.mode != "horizontal_only"
        && config.mode != "vertical_only")
        throw std::invalid_argument(
            "collocated eddy viscosity received an unsupported direction mode");
    if (state.NR < 5 || state.NTH < 5 || state.NZ < 5)
        throw std::invalid_argument(
            "collocated eddy viscosity requires at least 5x5x5 storage");
    if (state.u == nullptr || state.v == nullptr || state.w == nullptr
        || state.rho == nullptr || !matches(*state.u, state)
        || !matches(*state.v, state) || !matches(*state.w, state)
        || !matches(*state.rho, state) || !matches(coefficient, state))
        throw std::invalid_argument(
            "collocated eddy viscosity requires shape-matched velocity, density, and coefficient fields");

    grid_metric::sanitize_positive(grid.dx);
    grid_metric::sanitize_positive(grid.dy);
    if (grid.terrain_metrics_active)
        terrain_transport::validate_geometry(*grid.terrain_metrics);
    else
        grid_metric::validate_vertical_metrics(
            grid, state.NR, state.NTH, state.NZ);

    Domain domain{
        physical_scalar_domain(
            state.NR, state.NTH, state.NZ, StaggerType::Collocated),
        periodic_x,
        periodic_y,
    };
    if (domain.cells.nx() < 3 || domain.cells.ny() < 3
        || domain.cells.nz() < 3)
        throw std::invalid_argument(
            "collocated eddy viscosity requires three physical cells per direction");

    for (int i = domain.cells.i_begin; i < domain.cells.i_end; ++i)
        for (int j = domain.cells.j_begin; j < domain.cells.j_end; ++j)
            for (int k = domain.cells.k_begin; k < domain.cells.k_end; ++k)
            {
                const double rho = (*state.rho)(i, j, k);
                const double nu = coefficient(i, j, k);
                const double u = (*state.u)(i, j, k);
                const double v = (*state.v)(i, j, k);
                const double w = (*state.w)(i, j, k);
                if (!std::isfinite(rho) || rho <= 0.0
                    || !std::isfinite(nu) || nu < 0.0
                    || !std::isfinite(u) || !std::isfinite(v)
                    || !std::isfinite(w))
                    throw std::invalid_argument(
                        "collocated eddy viscosity requires finite physical state, positive density, and non-negative coefficients");
            }
    return domain;
}

class FlatOperator
{
public:
    FlatOperator(const TurbulenceStateView& state,
                 const GridMetrics& grid,
                 const Field3D& coefficient,
                 Domain domain)
        : state_(state), grid_(grid), coefficient_(coefficient),
          domain_(domain)
    {
    }

    double gradient_x(const Field3D& field, int i, int j, int k) const
    {
        const int previous = domain_.previous_i(i);
        const int next = domain_.next_i(i);
        if (previous >= 0 && next >= 0)
            return (field(next, j, k) - field(previous, j, k))
                / (2.0 * grid_.dx);
        if (next >= 0)
            return (field(next, j, k) - field(i, j, k)) / grid_.dx;
        return (field(i, j, k) - field(previous, j, k)) / grid_.dx;
    }

    double gradient_y(const Field3D& field, int i, int j, int k) const
    {
        const int previous = domain_.previous_j(j);
        const int next = domain_.next_j(j);
        if (previous >= 0 && next >= 0)
            return (field(i, next, k) - field(i, previous, k))
                / (2.0 * grid_.dy);
        if (next >= 0)
            return (field(i, next, k) - field(i, j, k)) / grid_.dy;
        return (field(i, j, k) - field(i, previous, k)) / grid_.dy;
    }

    double gradient_z(const Field3D& field, int i, int j, int k) const
    {
        const auto& cells = domain_.cells;
        if (k == cells.k_begin)
            return (field(i, j, k + 1) - field(i, j, k))
                / center_distance(i, j, k);
        if (k + 1 == cells.k_end)
            return (field(i, j, k) - field(i, j, k - 1))
                / center_distance(i, j, k - 1);
        return (field(i, j, k + 1) - field(i, j, k - 1))
            / (center_distance(i, j, k - 1)
               + center_distance(i, j, k));
    }

    double directional_strain_sq(int i, int j, int k,
                                 const std::string& mode) const
    {
        const Field3D& u = *state_.u;
        const Field3D& v = *state_.v;
        const Field3D& w = *state_.w;
        const double du_dx = gradient_x(u, i, j, k);
        const double du_dy = gradient_y(u, i, j, k);
        const double du_dz = gradient_z(u, i, j, k);
        const double dv_dx = gradient_x(v, i, j, k);
        const double dv_dy = gradient_y(v, i, j, k);
        const double dv_dz = gradient_z(v, i, j, k);
        const double dw_dx = gradient_x(w, i, j, k);
        const double dw_dy = gradient_y(w, i, j, k);
        const double dw_dz = gradient_z(w, i, j, k);
        if (mode == "horizontal_only")
        {
            const double s_xy = 0.5 * (du_dy + dv_dx);
            return 2.0 * (du_dx * du_dx + dv_dy * dv_dy)
                + 4.0 * s_xy * s_xy + dw_dx * dw_dx + dw_dy * dw_dy;
        }
        if (mode == "vertical_only")
            return du_dz * du_dz + dv_dz * dv_dz
                + 2.0 * dw_dz * dw_dz;
        const double s_xy = 0.5 * (du_dy + dv_dx);
        const double s_xz = 0.5 * (du_dz + dw_dx);
        const double s_yz = 0.5 * (dv_dz + dw_dy);
        return 2.0 * (du_dx * du_dx + dv_dy * dv_dy
                      + dw_dz * dw_dz)
            + 4.0 * (s_xy * s_xy + s_xz * s_xz + s_yz * s_yz);
    }

    struct XStress { double x = 0.0, y = 0.0, z = 0.0; };
    struct YStress { double x = 0.0, y = 0.0, z = 0.0; };
    struct ZStress { double x = 0.0, y = 0.0, z = 0.0; };

    XStress x_stress(int left, int j, int k,
                     bool use_horizontal, bool use_vertical) const
    {
        if (!use_horizontal || left < 0) return {};
        const int right = domain_.next_i(left);
        if (right < 0) return {};
        const double mu = face_mu(left, j, k, right, j, k);
        const double du_dx = ((*state_.u)(right, j, k)
                              - (*state_.u)(left, j, k)) / grid_.dx;
        const double dv_dx = ((*state_.v)(right, j, k)
                              - (*state_.v)(left, j, k)) / grid_.dx;
        const double dw_dx = ((*state_.w)(right, j, k)
                              - (*state_.w)(left, j, k)) / grid_.dx;
        const double du_dy = 0.5 * (
            gradient_y(*state_.u, left, j, k)
            + gradient_y(*state_.u, right, j, k));
        const double du_dz = 0.5 * (
            gradient_z(*state_.u, left, j, k)
            + gradient_z(*state_.u, right, j, k));
        return {
            2.0 * mu * du_dx,
            mu * (dv_dx + du_dy),
            mu * (dw_dx + (use_vertical ? du_dz : 0.0)),
        };
    }

    YStress y_stress(int i, int south, int k,
                     bool use_horizontal, bool use_vertical) const
    {
        if (!use_horizontal || south < 0) return {};
        const int north = domain_.next_j(south);
        if (north < 0) return {};
        const double mu = face_mu(i, south, k, i, north, k);
        const double du_dy = ((*state_.u)(i, north, k)
                              - (*state_.u)(i, south, k)) / grid_.dy;
        const double dv_dy = ((*state_.v)(i, north, k)
                              - (*state_.v)(i, south, k)) / grid_.dy;
        const double dw_dy = ((*state_.w)(i, north, k)
                              - (*state_.w)(i, south, k)) / grid_.dy;
        const double dv_dx = 0.5 * (
            gradient_x(*state_.v, i, south, k)
            + gradient_x(*state_.v, i, north, k));
        const double dv_dz = 0.5 * (
            gradient_z(*state_.v, i, south, k)
            + gradient_z(*state_.v, i, north, k));
        return {
            mu * (du_dy + dv_dx),
            2.0 * mu * dv_dy,
            mu * (dw_dy + (use_vertical ? dv_dz : 0.0)),
        };
    }

    ZStress z_stress(int i, int j, int lower,
                     bool use_horizontal, bool use_vertical) const
    {
        if (!use_vertical || lower < domain_.cells.k_begin
            || lower + 1 >= domain_.cells.k_end)
            return {};
        const int upper = lower + 1;
        const double distance = center_distance(i, j, lower);
        const double mu = face_mu(i, j, lower, i, j, upper);
        const double du_dz = ((*state_.u)(i, j, upper)
                              - (*state_.u)(i, j, lower)) / distance;
        const double dv_dz = ((*state_.v)(i, j, upper)
                              - (*state_.v)(i, j, lower)) / distance;
        const double dw_dz = ((*state_.w)(i, j, upper)
                              - (*state_.w)(i, j, lower)) / distance;
        const double dw_dx = 0.5 * (
            gradient_x(*state_.w, i, j, lower)
            + gradient_x(*state_.w, i, j, upper));
        const double dw_dy = 0.5 * (
            gradient_y(*state_.w, i, j, lower)
            + gradient_y(*state_.w, i, j, upper));
        return {
            mu * (du_dz + (use_horizontal ? dw_dx : 0.0)),
            mu * (dv_dz + (use_horizontal ? dw_dy : 0.0)),
            2.0 * mu * dw_dz,
        };
    }

    double scalar_x_flux(const Field3D& scalar, int left, int j, int k) const
    {
        if (left < 0) return 0.0;
        const int right = domain_.next_i(left);
        if (right < 0) return 0.0;
        return face_mu(left, j, k, right, j, k)
            * (scalar(right, j, k) - scalar(left, j, k)) / grid_.dx;
    }
    double scalar_y_flux(const Field3D& scalar, int i, int south, int k) const
    {
        if (south < 0) return 0.0;
        const int north = domain_.next_j(south);
        if (north < 0) return 0.0;
        return face_mu(i, south, k, i, north, k)
            * (scalar(i, north, k) - scalar(i, south, k)) / grid_.dy;
    }
    double scalar_z_flux(const Field3D& scalar, int i, int j, int lower) const
    {
        if (lower < domain_.cells.k_begin
            || lower + 1 >= domain_.cells.k_end)
            return 0.0;
        return face_mu(i, j, lower, i, j, lower + 1)
            * (scalar(i, j, lower + 1) - scalar(i, j, lower))
            / center_distance(i, j, lower);
    }

    double shear_transfer(int i, int j, int k, const std::string& mode) const
    {
        const bool use_h = mode != "vertical_only";
        const bool use_v = mode != "horizontal_only";
        const auto work = [&](const auto& stress, int ia, int ja, int ka,
                              int ib, int jb, int kb)
        {
            return stress.x * ((*state_.u)(ib, jb, kb) - (*state_.u)(ia, ja, ka))
                + stress.y * ((*state_.v)(ib, jb, kb) - (*state_.v)(ia, ja, ka))
                + stress.z * ((*state_.w)(ib, jb, kb) - (*state_.w)(ia, ja, ka));
        };
        const auto x_work = [&](int left)
        {
            const int right = left >= 0 ? domain_.next_i(left) : -1;
            return right >= 0 ? work(x_stress(left, j, k, use_h, use_v),
                left, j, k, right, j, k) : 0.0;
        };
        const auto y_work = [&](int south)
        {
            const int north = south >= 0 ? domain_.next_j(south) : -1;
            return north >= 0 ? work(y_stress(i, south, k, use_h, use_v),
                i, south, k, i, north, k) : 0.0;
        };
        const auto z_work = [&](int lower)
        {
            return lower >= domain_.cells.k_begin && lower + 1 < domain_.cells.k_end
                ? work(z_stress(i, j, lower, use_h, use_v),
                       i, j, lower, i, j, lower + 1) : 0.0;
        };
        // Each shared face contributes half its stress work to either cell.
        // Divide by that cell's mass, including its own vertical thickness.
        return 0.5 / (*state_.rho)(i, j, k) * (
            (x_work(domain_.previous_i(i)) + x_work(i)) / grid_.dx
            + (y_work(domain_.previous_j(j)) + y_work(j)) / grid_.dy
            + (z_work(k - 1) + z_work(k)) / grid_.dz[static_cast<std::size_t>(k)]);
    }

private:
    double center_distance(int i, int j, int lower) const
    {
        return grid_metric::vertical_center_distance(
            grid_, i, j, lower, state_.NZ);
    }
    double face_mu(int i0, int j0, int k0,
                   int i1, int j1, int k1) const
    {
        const double first = (*state_.rho)(i0, j0, k0)
            * coefficient_(i0, j0, k0);
        const double second = (*state_.rho)(i1, j1, k1)
            * coefficient_(i1, j1, k1);
        return 0.5 * (first + second);
    }

    const TurbulenceStateView& state_;
    const GridMetrics& grid_;
    const Field3D& coefficient_;
    Domain domain_;
};

class MappedOperator
{
public:
    MappedOperator(const TurbulenceStateView& state,
                   const GridMetrics& grid,
                   const Field3D& coefficient,
                   Domain domain)
        : state_(state), metrics_(*grid.terrain_metrics),
          coefficient_(coefficient), domain_(domain)
    {
    }

    struct Gradient
    {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    Gradient gradient(const Field3D& field, int i, int j, int k) const
    {
        const double vertical = gradient_z(field, i, j, k);
        return {
            computational_x(field, i, j, k)
                - terrain_slope_x(i, j, k) * vertical,
            computational_y(field, i, j, k)
                - terrain_slope_y(i, j, k) * vertical,
            vertical,
        };
    }

    double directional_strain_sq(int i, int j, int k,
                                 const std::string& mode) const
    {
        const auto stress = unit_strain(i, j, k, mode);
        const double result = stress.xx * stress.xx
            + stress.yy * stress.yy + stress.zz * stress.zz
            + 2.0 * (stress.xy * stress.xy
                   + stress.xz * stress.xz
                   + stress.yz * stress.yz);
        if (!std::isfinite(result) || result < 0.0)
            throw std::runtime_error(
                "mapped collocated strain produced an invalid norm");
        return 0.5 * result;
    }

    SymmetricStressTensor stress(int i, int j, int k,
                                 const std::string& mode) const
    {
        SymmetricStressTensor result = unit_strain(i, j, k, mode);
        const double mu = static_cast<double>((*state_.rho)(i, j, k))
            * static_cast<double>(coefficient_(i, j, k));
        result.xx *= mu;
        result.yy *= mu;
        result.zz *= mu;
        result.xy *= mu;
        result.xz *= mu;
        result.yz *= mu;
        return result;
    }

    PhysicalVector3 momentum_flux_x(int left, int j, int k,
                                    const std::string& mode) const
    {
        if (mode == "vertical_only" || left < 0) return {};
        const int right = domain_.next_i(left);
        if (right < 0) return {};
        const auto face = average(stress(left, j, k, mode),
                                  stress(right, j, k, mode));
        const double jacobian = right == domain_.cells.i_begin
            ? 0.5 * (metrics_.J(left, j, k) + metrics_.J(right, j, k))
            : metrics_.Jx_face(left + 1, j, k);
        return terrain_transport::contravariant_stress_flux_x(
            jacobian, face);
    }

    PhysicalVector3 momentum_flux_y(int i, int south, int k,
                                    const std::string& mode) const
    {
        if (mode == "vertical_only" || south < 0) return {};
        const int north = domain_.next_j(south);
        if (north < 0) return {};
        const auto face = average(stress(i, south, k, mode),
                                  stress(i, north, k, mode));
        const double jacobian = north == domain_.cells.j_begin
            ? 0.5 * (metrics_.J(i, south, k) + metrics_.J(i, north, k))
            : metrics_.Jy_face(i, south + 1, k);
        return terrain_transport::contravariant_stress_flux_y(
            jacobian, face);
    }

    PhysicalVector3 momentum_flux_z(int i, int j, int lower,
                                    const std::string& mode) const
    {
        if (lower < domain_.cells.k_begin
            || lower + 1 >= domain_.cells.k_end)
            return {};
        const auto face = average(stress(i, j, lower, mode),
                                  stress(i, j, lower + 1, mode));
        const int k_face = lower + 1;
        return terrain_transport::contravariant_stress_flux_zeta(
            metrics_.Gx_face(i, j, k_face),
            metrics_.Gy_face(i, j, k_face), face);
    }

    double scalar_flux_x(const Field3D& scalar, int left, int j, int k,
                         bool use_horizontal, bool use_vertical) const
    {
        if (!use_horizontal || left < 0) return 0.0;
        const int right = domain_.next_i(left);
        if (right < 0) return 0.0;
        const Gradient first = selected_gradient(
            scalar, left, j, k, use_horizontal, use_vertical);
        const Gradient second = selected_gradient(
            scalar, right, j, k, use_horizontal, use_vertical);
        const double transport = face_transport(
            left, j, k, right, j, k);
        const double jacobian = right == domain_.cells.i_begin
            ? 0.5 * (metrics_.J(left, j, k) + metrics_.J(right, j, k))
            : metrics_.Jx_face(left + 1, j, k);
        return jacobian * transport * 0.5 * (first.x + second.x);
    }

    double scalar_flux_y(const Field3D& scalar, int i, int south, int k,
                         bool use_horizontal, bool use_vertical) const
    {
        if (!use_horizontal || south < 0) return 0.0;
        const int north = domain_.next_j(south);
        if (north < 0) return 0.0;
        const Gradient first = selected_gradient(
            scalar, i, south, k, use_horizontal, use_vertical);
        const Gradient second = selected_gradient(
            scalar, i, north, k, use_horizontal, use_vertical);
        const double transport = face_transport(
            i, south, k, i, north, k);
        const double jacobian = north == domain_.cells.j_begin
            ? 0.5 * (metrics_.J(i, south, k) + metrics_.J(i, north, k))
            : metrics_.Jy_face(i, south + 1, k);
        return jacobian * transport * 0.5 * (first.y + second.y);
    }

    double scalar_flux_z(const Field3D& scalar, int i, int j, int lower,
                         bool use_horizontal, bool use_vertical) const
    {
        if (lower < domain_.cells.k_begin
            || lower + 1 >= domain_.cells.k_end)
            return 0.0;
        const Gradient first = selected_gradient(
            scalar, i, j, lower, use_horizontal, use_vertical);
        const Gradient second = selected_gradient(
            scalar, i, j, lower + 1, use_horizontal, use_vertical);
        const double transport = face_transport(
            i, j, lower, i, j, lower + 1);
        const double fx = transport * 0.5 * (first.x + second.x);
        const double fy = transport * 0.5 * (first.y + second.y);
        const double fz = transport * 0.5 * (first.z + second.z);
        const int k_face = lower + 1;
        return fz + metrics_.Gx_face(i, j, k_face) * fx
                  + metrics_.Gy_face(i, j, k_face) * fy;
    }

    PhysicalVector3 momentum_tendency(int target_i, int target_j,
                                      int target_k,
                                      const std::string& mode) const
    {
        PhysicalVector3 force;
        for_each_influenced_center(
            target_i, target_j, target_k,
            [&](int i, int j, int k)
            {
                Gradient coefficient = gradient_coefficient(
                    i, j, k, target_i, target_j, target_k);
                if (mode == "horizontal_only") coefficient.z = 0.0;
                else if (mode == "vertical_only")
                    coefficient.x = coefficient.y = 0.0;
                if (coefficient.x == 0.0 && coefficient.y == 0.0
                    && coefficient.z == 0.0)
                    return;
                const SymmetricStressTensor tensor = stress(i, j, k, mode);
                const double volume = metrics_.J(i, j, k) * metrics_.dx
                    * metrics_.dy * metrics_.dzeta;
                force.x -= volume * (
                    tensor.xx * coefficient.x
                  + tensor.xy * coefficient.y
                  + tensor.xz * coefficient.z);
                force.y -= volume * (
                    tensor.xy * coefficient.x
                  + tensor.yy * coefficient.y
                  + tensor.yz * coefficient.z);
                force.z -= volume * (
                    tensor.xz * coefficient.x
                  + tensor.yz * coefficient.y
                  + tensor.zz * coefficient.z);
            });
        const double target_mass = static_cast<double>(
            (*state_.rho)(target_i, target_j, target_k))
            * metrics_.J(target_i, target_j, target_k) * metrics_.dx
            * metrics_.dy * metrics_.dzeta;
        return {force.x / target_mass, force.y / target_mass,
                force.z / target_mass};
    }

    double scalar_tendency(const Field3D& scalar, int target_i,
                           int target_j, int target_k,
                           bool use_horizontal, bool use_vertical) const
    {
        double force = 0.0;
        for_each_influenced_center(
            target_i, target_j, target_k,
            [&](int i, int j, int k)
            {
                Gradient coefficient = gradient_coefficient(
                    i, j, k, target_i, target_j, target_k);
                if (!use_horizontal) coefficient.x = coefficient.y = 0.0;
                if (!use_vertical) coefficient.z = 0.0;
                if (coefficient.x == 0.0 && coefficient.y == 0.0
                    && coefficient.z == 0.0)
                    return;
                const Gradient value = selected_gradient(
                    scalar, i, j, k, use_horizontal, use_vertical);
                const double transport = static_cast<double>(
                    (*state_.rho)(i, j, k)) * coefficient_(i, j, k);
                const double volume = metrics_.J(i, j, k) * metrics_.dx
                    * metrics_.dy * metrics_.dzeta;
                force -= volume * transport * (
                    value.x * coefficient.x + value.y * coefficient.y
                    + value.z * coefficient.z);
            });
        const double target_mass = static_cast<double>(
            (*state_.rho)(target_i, target_j, target_k))
            * metrics_.J(target_i, target_j, target_k) * metrics_.dx
            * metrics_.dy * metrics_.dzeta;
        return force / target_mass;
    }

private:
    template <typename Callback>
    void for_each_influenced_center(int target_i, int target_j, int target_k,
                                    Callback&& callback) const
    {
        std::array<int, 3> i_candidates{
            domain_.previous_i(target_i), target_i,
            domain_.next_i(target_i)};
        std::array<int, 3> j_candidates{
            domain_.previous_j(target_j), target_j,
            domain_.next_j(target_j)};
        std::array<int, 3> k_candidates{
            target_k > domain_.cells.k_begin ? target_k - 1 : -1,
            target_k,
            target_k + 1 < domain_.cells.k_end ? target_k + 1 : -1};
        for (int i : i_candidates)
        {
            if (i < 0) continue;
            for (int j : j_candidates)
            {
                if (j < 0) continue;
                for (int k : k_candidates)
                {
                    if (k < 0) continue;
                    callback(i, j, k);
                }
            }
        }
    }

    Gradient gradient_coefficient(int center_i, int center_j, int center_k,
                                  int target_i, int target_j,
                                  int target_k) const
    {
        const auto vertical_coefficient = [&]()
        {
            if (target_i != center_i || target_j != center_j) return 0.0;
            const int lower = center_k == domain_.cells.k_begin
                ? center_k : center_k - 1;
            const int upper = center_k + 1 == domain_.cells.k_end
                ? center_k : center_k + 1;
            const double distance = metrics_.z(center_i, center_j, upper)
                                  - metrics_.z(center_i, center_j, lower);
            double value = 0.0;
            if (target_k == lower) value -= 1.0 / distance;
            if (target_k == upper) value += 1.0 / distance;
            return value;
        };
        const double cz = vertical_coefficient();
        double cx = -terrain_slope_x(center_i, center_j, center_k) * cz;
        if (target_j == center_j && target_k == center_k)
        {
            const int previous = domain_.previous_i(center_i);
            const int next = domain_.next_i(center_i);
            if (previous >= 0 && next >= 0)
            {
                if (target_i == previous) cx -= 0.5 / metrics_.dx;
                if (target_i == next) cx += 0.5 / metrics_.dx;
            }
            else if (next >= 0)
            {
                if (target_i == center_i) cx -= 1.0 / metrics_.dx;
                if (target_i == next) cx += 1.0 / metrics_.dx;
            }
            else
            {
                if (target_i == previous) cx -= 1.0 / metrics_.dx;
                if (target_i == center_i) cx += 1.0 / metrics_.dx;
            }
        }
        double cy = -terrain_slope_y(center_i, center_j, center_k) * cz;
        if (target_i == center_i && target_k == center_k)
        {
            const int previous = domain_.previous_j(center_j);
            const int next = domain_.next_j(center_j);
            if (previous >= 0 && next >= 0)
            {
                if (target_j == previous) cy -= 0.5 / metrics_.dy;
                if (target_j == next) cy += 0.5 / metrics_.dy;
            }
            else if (next >= 0)
            {
                if (target_j == center_j) cy -= 1.0 / metrics_.dy;
                if (target_j == next) cy += 1.0 / metrics_.dy;
            }
            else
            {
                if (target_j == previous) cy -= 1.0 / metrics_.dy;
                if (target_j == center_j) cy += 1.0 / metrics_.dy;
            }
        }
        return {cx, cy, cz};
    }

    Gradient selected_gradient(const Field3D& field, int i, int j, int k,
                               bool use_horizontal, bool use_vertical) const
    {
        Gradient result = gradient(field, i, j, k);
        if (!use_horizontal) result.x = result.y = 0.0;
        if (!use_vertical) result.z = 0.0;
        return result;
    }

    SymmetricStressTensor unit_strain(int i, int j, int k,
                                      const std::string& mode) const
    {
        Gradient gu = gradient(*state_.u, i, j, k);
        Gradient gv = gradient(*state_.v, i, j, k);
        Gradient gw = gradient(*state_.w, i, j, k);
        if (mode == "horizontal_only")
            gu.z = gv.z = gw.z = 0.0;
        else if (mode == "vertical_only")
        {
            gu.x = gu.y = gv.x = gv.y = gw.x = gw.y = 0.0;
        }
        return {
            2.0 * gu.x, 2.0 * gv.y, 2.0 * gw.z,
            gu.y + gv.x, gu.z + gw.x, gv.z + gw.y,
        };
    }

    static SymmetricStressTensor average(const SymmetricStressTensor& a,
                                         const SymmetricStressTensor& b)
    {
        return {
            0.5 * (a.xx + b.xx), 0.5 * (a.yy + b.yy),
            0.5 * (a.zz + b.zz), 0.5 * (a.xy + b.xy),
            0.5 * (a.xz + b.xz), 0.5 * (a.yz + b.yz),
        };
    }

    double gradient_z(const Field3D& field, int i, int j, int k) const
    {
        const auto& cells = domain_.cells;
        const int lower = k == cells.k_begin ? k : k - 1;
        const int upper = k + 1 == cells.k_end ? k : k + 1;
        return (field(i, j, upper) - field(i, j, lower))
            / (metrics_.z(i, j, upper) - metrics_.z(i, j, lower));
    }

    double computational_x(const Field3D& field,
                           int i, int j, int k) const
    {
        const int previous = domain_.previous_i(i);
        const int next = domain_.next_i(i);
        if (previous >= 0 && next >= 0)
            return (field(next, j, k) - field(previous, j, k))
                / (2.0 * metrics_.dx);
        if (next >= 0)
            return (field(next, j, k) - field(i, j, k)) / metrics_.dx;
        return (field(i, j, k) - field(previous, j, k)) / metrics_.dx;
    }

    double computational_y(const Field3D& field,
                           int i, int j, int k) const
    {
        const int previous = domain_.previous_j(j);
        const int next = domain_.next_j(j);
        if (previous >= 0 && next >= 0)
            return (field(i, next, k) - field(i, previous, k))
                / (2.0 * metrics_.dy);
        if (next >= 0)
            return (field(i, next, k) - field(i, j, k)) / metrics_.dy;
        return (field(i, j, k) - field(i, previous, k)) / metrics_.dy;
    }

    double terrain_slope_x(int i, int j, int k) const
    {
        const int previous = domain_.previous_i(i);
        const int next = domain_.next_i(i);
        if (previous >= 0 && next >= 0)
            return (metrics_.z(next, j, k) - metrics_.z(previous, j, k))
                / (2.0 * metrics_.dx);
        if (next >= 0)
            return (metrics_.z(next, j, k) - metrics_.z(i, j, k))
                / metrics_.dx;
        return (metrics_.z(i, j, k) - metrics_.z(previous, j, k))
            / metrics_.dx;
    }

    double terrain_slope_y(int i, int j, int k) const
    {
        const int previous = domain_.previous_j(j);
        const int next = domain_.next_j(j);
        if (previous >= 0 && next >= 0)
            return (metrics_.z(i, next, k) - metrics_.z(i, previous, k))
                / (2.0 * metrics_.dy);
        if (next >= 0)
            return (metrics_.z(i, next, k) - metrics_.z(i, j, k))
                / metrics_.dy;
        return (metrics_.z(i, j, k) - metrics_.z(i, previous, k))
            / metrics_.dy;
    }

    double face_transport(int i0, int j0, int k0,
                          int i1, int j1, int k1) const
    {
        return 0.5 * (
            static_cast<double>((*state_.rho)(i0, j0, k0))
                * coefficient_(i0, j0, k0)
          + static_cast<double>((*state_.rho)(i1, j1, k1))
                * coefficient_(i1, j1, k1));
    }

    const TurbulenceStateView& state_;
    const TerrainMetrics3D& metrics_;
    const Field3D& coefficient_;
    Domain domain_;
};

} // namespace

double directional_strain_sq(const TurbulenceStateView& state,
                             const GridMetrics& grid,
                             int i,
                             int j,
                             int k,
                             const std::string& mode,
                             bool periodic_x,
                             bool periodic_y)
{
    // Coefficient construction has already validated the full scheme state;
    // this local guard prevents a direct caller from indexing closure storage.
    const PhysicalCellDomain cells = physical_scalar_domain(
        state.NR, state.NTH, state.NZ, StaggerType::Collocated);
    if (i < cells.i_begin || i >= cells.i_end || j < cells.j_begin
        || j >= cells.j_end || k < cells.k_begin || k >= cells.k_end)
        throw std::out_of_range(
            "collocated strain index is outside the physical domain");
    Domain domain{cells, periodic_x, periodic_y};
    if (grid.terrain_metrics_active)
    {
        if (grid.terrain_metrics == nullptr || state.rho == nullptr)
            throw std::invalid_argument(
                "mapped collocated strain requires terrain metrics and density");
        terrain_transport::validate_geometry(*grid.terrain_metrics);
        MappedOperator op(state, grid, *state.rho, domain);
        return op.directional_strain_sq(i, j, k, mode);
    }
    FlatOperator op(state, grid, *state.rho, domain);
    return op.directional_strain_sq(i, j, k, mode);
}

double flat_shear_transfer(const TurbulenceStateView& state,
                           const GridMetrics& grid, const Field3D& viscosity,
                           int i, int j, int k, const std::string& mode)
{
    const Domain domain{physical_scalar_domain(state.NR, state.NTH, state.NZ,
        StaggerType::Collocated), state.periodic_x, state.periodic_y};
    const FlatOperator op(state, grid, viscosity, domain);
    return op.shear_transfer(i, j, k, mode);
}

void momentum_tendencies(const TurbulenceConfig& config,
                         const TurbulenceStateView& state,
                         const GridMetrics& grid,
                         const Field3D& viscosity,
                         Field3D& du_dt,
                         Field3D& dv_dt,
                         Field3D& dw_dt,
                         bool periodic_x,
                         bool periodic_y)
{
    const Domain domain = validate_contract(
        config, state, grid, viscosity, periodic_x, periodic_y);
    if (!matches(du_dt, state) || !matches(dv_dt, state)
        || !matches(dw_dt, state))
        throw std::invalid_argument(
            "collocated momentum tendencies must be pre-shaped");

    du_dt.fill(0.0f);
    dv_dt.fill(0.0f);
    dw_dt.fill(0.0f);
    if (grid.terrain_metrics_active)
    {
        const MappedOperator op(state, grid, viscosity, domain);
        const auto& cells = domain.cells;
        #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(cells.nx()) * cells.ny() * cells.nz() >= 32768)
        for (int i = cells.i_begin; i < cells.i_end; ++i)
            for (int j = cells.j_begin; j < cells.j_end; ++j)
                for (int k = cells.k_begin; k < cells.k_end; ++k)
                {
                    const PhysicalVector3 tendency = op.momentum_tendency(
                        i, j, k, config.mode);
                    du_dt(i, j, k) = static_cast<float>(tendency.x);
                    dv_dt(i, j, k) = static_cast<float>(tendency.y);
                    dw_dt(i, j, k) = static_cast<float>(tendency.z);
                }
        return;
    }
    const bool use_horizontal = config.mode != "vertical_only";
    const bool use_vertical = config.mode != "horizontal_only";
    const FlatOperator op(state, grid, viscosity, domain);
    const auto& cells = domain.cells;

    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(cells.nx()) * cells.ny() * cells.nz() >= 32768)
    for (int i = cells.i_begin; i < cells.i_end; ++i)
        for (int j = cells.j_begin; j < cells.j_end; ++j)
            for (int k = cells.k_begin; k < cells.k_end; ++k)
            {
                const auto east = op.x_stress(
                    i, j, k, use_horizontal, use_vertical);
                const auto west = op.x_stress(
                    domain.previous_i(i), j, k, use_horizontal, use_vertical);
                const auto north = op.y_stress(
                    i, j, k, use_horizontal, use_vertical);
                const auto south = op.y_stress(
                    i, domain.previous_j(j), k,
                    use_horizontal, use_vertical);
                const auto top = op.z_stress(
                    i, j, k, use_horizontal, use_vertical);
                const auto bottom = op.z_stress(
                    i, j, k - 1, use_horizontal, use_vertical);
                const double inverse_density = 1.0 / (*state.rho)(i, j, k);
                du_dt(i, j, k) = static_cast<float>(inverse_density * (
                    (east.x - west.x) / grid.dx
                    + (north.x - south.x) / grid.dy
                    + (top.x - bottom.x)
                        / grid.dz[static_cast<std::size_t>(k)]));
                dv_dt(i, j, k) = static_cast<float>(inverse_density * (
                    (east.y - west.y) / grid.dx
                    + (north.y - south.y) / grid.dy
                    + (top.y - bottom.y)
                        / grid.dz[static_cast<std::size_t>(k)]));
                dw_dt(i, j, k) = static_cast<float>(inverse_density * (
                    (east.z - west.z) / grid.dx
                    + (north.z - south.z) / grid.dy
                    + (top.z - bottom.z)
                        / grid.dz[static_cast<std::size_t>(k)]));
            }
}

void scalar_tendency(const TurbulenceConfig& config,
                     const TurbulenceStateView& state,
                     const GridMetrics& grid,
                     const Field3D& diffusivity,
                     const Field3D& scalar,
                     Field3D& tendency,
                     bool periodic_x,
                     bool periodic_y)
{
    const Domain domain = validate_contract(
        config, state, grid, diffusivity, periodic_x, periodic_y);
    if (!matches(scalar, state) || !matches(tendency, state))
        throw std::invalid_argument(
            "collocated scalar diffusion requires shape-matched scalar and tendency fields");
    for (int i = domain.cells.i_begin; i < domain.cells.i_end; ++i)
        for (int j = domain.cells.j_begin; j < domain.cells.j_end; ++j)
            for (int k = domain.cells.k_begin; k < domain.cells.k_end; ++k)
                if (!std::isfinite(static_cast<double>(scalar(i, j, k))))
                    throw std::invalid_argument(
                        "collocated scalar diffusion requires finite physical state");

    tendency.fill(0.0f);
    const bool use_horizontal = config.mode != "vertical_only";
    const bool use_vertical = config.mode != "horizontal_only";
    if (grid.terrain_metrics_active)
    {
        const MappedOperator op(state, grid, diffusivity, domain);
        const auto& cells = domain.cells;
        #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(cells.nx()) * cells.ny() * cells.nz() >= 32768)
        for (int i = cells.i_begin; i < cells.i_end; ++i)
            for (int j = cells.j_begin; j < cells.j_end; ++j)
                for (int k = cells.k_begin; k < cells.k_end; ++k)
                {
                    tendency(i, j, k) = static_cast<float>(
                        op.scalar_tendency(
                            scalar, i, j, k,
                            use_horizontal, use_vertical));
                }
        return;
    }
    const FlatOperator op(state, grid, diffusivity, domain);
    const auto& cells = domain.cells;
    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(cells.nx()) * cells.ny() * cells.nz() >= 32768)
    for (int i = cells.i_begin; i < cells.i_end; ++i)
        for (int j = cells.j_begin; j < cells.j_end; ++j)
            for (int k = cells.k_begin; k < cells.k_end; ++k)
            {
                double divergence = 0.0;
                if (use_horizontal)
                {
                    divergence += (op.scalar_x_flux(scalar, i, j, k)
                        - op.scalar_x_flux(
                            scalar, domain.previous_i(i), j, k)) / grid.dx;
                    divergence += (op.scalar_y_flux(scalar, i, j, k)
                        - op.scalar_y_flux(
                            scalar, i, domain.previous_j(j), k)) / grid.dy;
                }
                if (use_vertical)
                    divergence += (op.scalar_z_flux(scalar, i, j, k)
                        - op.scalar_z_flux(scalar, i, j, k - 1))
                        / grid.dz[static_cast<std::size_t>(k)];
                tendency(i, j, k) = static_cast<float>(
                    divergence / (*state.rho)(i, j, k));
            }
}

} // namespace tmv::collocated_eddy_viscosity
