#include "dynamics/cartesian_cgrid_momentum_flux.hpp"
#include "terrain/conservative_transport.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tmv::cartesian_cgrid_momentum_flux
{

double rho_x(const Field3D& rho, int i, int j, int k)
{
    return 0.5 * (static_cast<double>(rho[i][j][k]) + rho[i + 1][j][k]);
}

double rho_y(const Field3D& rho, int i, int j, int k)
{
    return 0.5 * (static_cast<double>(rho[i][j][k]) + rho[i][j + 1][k]);
}

double rho_z(const Field3D& rho, int i, int j, int k)
{
    return 0.5 * (static_cast<double>(rho[i][j][k]) + rho[i][j][k + 1]);
}

double x_normal_mass(const Field3D& rho, const Field3D& u,
                     int face, int j, int k, double frame_u_ms)
{
    const double velocity = 0.5 * (
        static_cast<double>(u[face - 1][j][k]) + u[face][j][k]);
    return static_cast<double>(rho[face][j][k]) * (velocity - frame_u_ms);
}

double y_normal_mass(const Field3D& rho, const Field3D& v,
                     int i, int face, int k, double frame_v_ms)
{
    const double velocity = 0.5 * (
        static_cast<double>(v[i][face - 1][k]) + v[i][face][k]);
    return static_cast<double>(rho[i][face][k]) * (velocity - frame_v_ms);
}

double z_normal_mass(const Field3D& rho, const Field3D& w,
                     int i, int j, int face)
{
    const double lower = face == 0 ? 0.0 : w[i][j][face - 1];
    const double upper = w[i][j][face];
    return static_cast<double>(rho[i][j][face]) * 0.5 * (lower + upper);
}

double x_flux_x(const Field3D& rho, const Field3D& u,
                int face, int j, int k, double frame_u_ms)
{
    const double carried = 0.5 * (
        static_cast<double>(u[face - 1][j][k]) + u[face][j][k]);
    return static_cast<double>(rho[face][j][k]) *
           (carried - frame_u_ms) * carried;
}

double x_flux_y(const Field3D& rho, const Field3D& u, const Field3D& v,
                int face, int j, int k, double frame_u_ms)
{
    const double density = 0.25 * (
        static_cast<double>(rho[face][j][k]) + rho[face + 1][j][k] +
        rho[face][j + 1][k] + rho[face + 1][j + 1][k]);
    const double transport = 0.5 * (
        static_cast<double>(u[face][j][k]) + u[face][j + 1][k]) -
        frame_u_ms;
    const double carried = 0.5 * (
        static_cast<double>(v[face][j][k]) + v[face + 1][j][k]);
    return density * transport * carried;
}

double x_flux_z(const Field3D& rho, const Field3D& u, const Field3D& w,
                int face, int j, int k, double frame_u_ms)
{
    const double density = 0.25 * (
        static_cast<double>(rho[face][j][k]) + rho[face + 1][j][k] +
        rho[face][j][k + 1] + rho[face + 1][j][k + 1]);
    const double transport = 0.5 * (
        static_cast<double>(u[face][j][k]) + u[face][j][k + 1]) -
        frame_u_ms;
    const double carried = 0.5 * (
        static_cast<double>(w[face][j][k]) + w[face + 1][j][k]);
    return density * transport * carried;
}

double y_flux_x(const Field3D& rho, const Field3D& u, const Field3D& v,
                int i, int face, int k, double frame_v_ms)
{
    const double density = 0.25 * (
        static_cast<double>(rho[i][face][k]) + rho[i + 1][face][k] +
        rho[i][face + 1][k] + rho[i + 1][face + 1][k]);
    const double transport = 0.5 * (
        static_cast<double>(v[i][face][k]) + v[i + 1][face][k]) -
        frame_v_ms;
    const double carried = 0.5 * (
        static_cast<double>(u[i][face][k]) + u[i][face + 1][k]);
    return density * transport * carried;
}

double y_flux_y(const Field3D& rho, const Field3D& v,
                int i, int face, int k, double frame_v_ms)
{
    const double carried = 0.5 * (
        static_cast<double>(v[i][face - 1][k]) + v[i][face][k]);
    return static_cast<double>(rho[i][face][k]) *
           (carried - frame_v_ms) * carried;
}

double y_flux_z(const Field3D& rho, const Field3D& v, const Field3D& w,
                int i, int face, int k, double frame_v_ms)
{
    const double density = 0.25 * (
        static_cast<double>(rho[i][face][k]) + rho[i][face + 1][k] +
        rho[i][face][k + 1] + rho[i][face + 1][k + 1]);
    const double transport = 0.5 * (
        static_cast<double>(v[i][face][k]) + v[i][face][k + 1]) -
        frame_v_ms;
    const double carried = 0.5 * (
        static_cast<double>(w[i][face][k]) + w[i][face + 1][k]);
    return density * transport * carried;
}

double z_flux_x(const Field3D& rho, const Field3D& u, const Field3D& w,
                int i, int j, int face)
{
    const double density = 0.25 * (
        static_cast<double>(rho[i][j][face]) + rho[i + 1][j][face] +
        rho[i][j][face + 1] + rho[i + 1][j][face + 1]);
    const double transport = 0.5 * (
        static_cast<double>(w[i][j][face]) + w[i + 1][j][face]);
    const double carried = 0.5 * (
        static_cast<double>(u[i][j][face]) + u[i][j][face + 1]);
    return density * transport * carried;
}

double z_flux_y(const Field3D& rho, const Field3D& v, const Field3D& w,
                int i, int j, int face)
{
    const double density = 0.25 * (
        static_cast<double>(rho[i][j][face]) + rho[i][j + 1][face] +
        rho[i][j][face + 1] + rho[i][j + 1][face + 1]);
    const double transport = 0.5 * (
        static_cast<double>(w[i][j][face]) + w[i][j + 1][face]);
    const double carried = 0.5 * (
        static_cast<double>(v[i][j][face]) + v[i][j][face + 1]);
    return density * transport * carried;
}

double z_flux_z(const Field3D& rho, const Field3D& w,
                int i, int j, int face)
{
    const double lower = face == 0 ? 0.0 : w[i][j][face - 1];
    const double upper = w[i][j][face];
    const double carried = 0.5 * (lower + upper);
    return static_cast<double>(rho[i][j][face]) * carried * carried;
}

} // namespace tmv::cartesian_cgrid_momentum_flux

namespace tmv
{

void CartesianCGridMomentumFluxRegister::configure(
    int nr, int nth, int nz, CartesianInterfaceFootprint footprint,
    int x_face_tangential_ratio, int y_face_tangential_ratio)
{
    if (nr < 4 || nth < 4 || nz < 2 || footprint.i_begin < 1 ||
        footprint.i_end >= nr || footprint.j_begin < 1 ||
        footprint.j_end >= nth || footprint.k_begin < 0 ||
        footprint.k_end >= nz || footprint.i_begin >= footprint.i_end ||
        footprint.j_begin >= footprint.j_end ||
        footprint.k_begin >= footprint.k_end ||
        x_face_tangential_ratio <= 0 || y_face_tangential_ratio <= 0 ||
        (footprint.j_end - footprint.j_begin) % x_face_tangential_ratio != 0 ||
        (footprint.i_end - footprint.i_begin) % y_face_tangential_ratio != 0)
        throw std::invalid_argument(
            "C-grid momentum flux register requires a non-empty interior dual footprint");

    CartesianCGridMomentumFluxRegister candidate;
    candidate.nr_ = nr;
    candidate.nth_ = nth;
    candidate.nz_ = nz;
    candidate.footprint_ = footprint;
    candidate.x_ratio_ = x_face_tangential_ratio;
    candidate.y_ratio_ = y_face_tangential_ratio;
    const std::size_t levels = static_cast<std::size_t>(
        footprint.k_end - footprint.k_begin);
    const std::size_t x_count = static_cast<std::size_t>(
        (footprint.j_end - footprint.j_begin) / candidate.x_ratio_) * levels;
    const std::size_t y_count = static_cast<std::size_t>(
        (footprint.i_end - footprint.i_begin) / candidate.y_ratio_) * levels;
    allocate(candidate.advective_, x_count, y_count);
    allocate(candidate.pressure_, x_count, y_count);
    allocate(candidate.staged_, x_count, y_count);
    candidate.configured_ = true;
    *this = std::move(candidate);
}

void CartesianCGridMomentumFluxRegister::reset_interval() noexcept
{
    clear(advective_);
    clear(pressure_);
    clear(staged_);
    staged_ready_ = false;
    staged_validated_ = false;
    staged_kind_ = StageKind::None;
    last_committed_kind_ = StageKind::None;
    last_committed_operator_impulse_ = {};
    pending_committed_kind_ = StageKind::None;
    pending_committed_operator_impulse_ = {};
}

void CartesianCGridMomentumFluxRegister::stage_advective(
    const Field3D& rho, const Field3D& u, const Field3D& v, const Field3D& w,
    double dt_s, double dx_m, double dy_m, double dz_m,
    double frame_u_ms, double frame_v_ms)
{
    require_fields(rho, u, v, w, dt_s, dx_m, dy_m, dz_m);
    clear(staged_);
    const double x_scale = dt_s * dy_m * dz_m;
    const double y_scale = dt_s * dx_m * dz_m;
    const int k_count = footprint_.k_end - footprint_.k_begin;
    using namespace cartesian_cgrid_momentum_flux;
    for (int j = footprint_.j_begin; j < footprint_.j_end; ++j)
        for (int k = footprint_.k_begin; k < footprint_.k_end; ++k)
        {
            const std::size_t bin = static_cast<std::size_t>(
                (j - footprint_.j_begin) / x_ratio_ * k_count +
                k - footprint_.k_begin);
            const int west_scalar = footprint_.i_begin;
            const int east_scalar = footprint_.i_end;
            const int west_face = footprint_.i_begin - 1;
            const int east_face = footprint_.i_end - 1;
            add(staged_.west, 0, bin, x_scale *
                x_flux_x(rho, u, west_scalar, j, k, frame_u_ms));
            add(staged_.east, 0, bin, x_scale *
                x_flux_x(rho, u, east_scalar, j, k, frame_u_ms));
            add(staged_.west, 1, bin, x_scale *
                x_flux_y(rho, u, v, west_face, j, k, frame_u_ms));
            add(staged_.east, 1, bin, x_scale *
                x_flux_y(rho, u, v, east_face, j, k, frame_u_ms));
            add(staged_.west, 2, bin, x_scale *
                x_flux_z(rho, u, w, west_face, j, k, frame_u_ms));
            add(staged_.east, 2, bin, x_scale *
                x_flux_z(rho, u, w, east_face, j, k, frame_u_ms));
        }
    for (int i = footprint_.i_begin; i < footprint_.i_end; ++i)
        for (int k = footprint_.k_begin; k < footprint_.k_end; ++k)
        {
            const std::size_t bin = static_cast<std::size_t>(
                (i - footprint_.i_begin) / y_ratio_ * k_count +
                k - footprint_.k_begin);
            const int south_scalar = footprint_.j_begin;
            const int north_scalar = footprint_.j_end;
            const int south_face = footprint_.j_begin - 1;
            const int north_face = footprint_.j_end - 1;
            add(staged_.south, 0, bin, y_scale *
                y_flux_x(rho, u, v, i, south_face, k, frame_v_ms));
            add(staged_.north, 0, bin, y_scale *
                y_flux_x(rho, u, v, i, north_face, k, frame_v_ms));
            add(staged_.south, 1, bin, y_scale *
                y_flux_y(rho, v, i, south_scalar, k, frame_v_ms));
            add(staged_.north, 1, bin, y_scale *
                y_flux_y(rho, v, i, north_scalar, k, frame_v_ms));
            add(staged_.south, 2, bin, y_scale *
                y_flux_z(rho, v, w, i, south_face, k, frame_v_ms));
            add(staged_.north, 2, bin, y_scale *
                y_flux_z(rho, v, w, i, north_face, k, frame_v_ms));
        }
    finish_stage(dt_s, StageKind::Advective);
}

void CartesianCGridMomentumFluxRegister::stage_native_fluxes(
    const float* pairs, std::size_t count, double dt_s, double dx_m, double dy_m, double dz_m, bool pressure)
{
    const int  nk       = footprint_.k_end - footprint_.k_begin;
    const int  width    = footprint_.i_end - footprint_.i_begin;
    const int  height   = footprint_.j_end - footprint_.j_begin;
    const auto expected = 12 * (std::size_t(width) + height) * nk;
    if (!configured_ || !pairs || count != expected || !std::isfinite(dt_s) || dt_s <= 0 || !std::isfinite(dx_m) ||
        dx_m <= 0 || !std::isfinite(dy_m) || dy_m <= 0 || !std::isfinite(dz_m) || dz_m <= 0)
        throw std::invalid_argument("native momentum interface strip descriptor is invalid");
    for (std::size_t n = 0; n < count; n += 2)
        if (!std::isfinite(pairs[n]) || !std::isfinite(pairs[n + 1]) ||
            std::abs(double(pairs[n + 1])) > std::abs(double(pairs[n])) * std::numeric_limits<float>::epsilon())
            throw std::invalid_argument("native momentum interface pair is invalid");
    auto candidate = *this;
    clear(candidate.staged_);
    const std::array<CartesianMomentumInterfaceSide*, 4> sides{
        &candidate.staged_.west, &candidate.staged_.east, &candidate.staged_.south, &candidate.staged_.north};
    std::size_t offset = 0;
    for (int side = 0; side < 4; ++side)
    {
        const int    tangents = side < 2 ? height : width;
        const int    ratio    = side < 2 ? x_ratio_ : y_ratio_;
        const double scale    = dt_s * (side < 2 ? dy_m : dx_m) * dz_m;
        for (int tangent = 0; tangent < tangents; ++tangent)
            for (int k = 0; k < nk; ++k)
                for (std::size_t component = 0; component < 3; ++component)
                {
                    const double flux = double(pairs[offset]) + pairs[offset + 1];
                    add(*sides[side], component, std::size_t(tangent / ratio) * nk + k, scale * flux);
                    offset += 2;
                }
    }
    candidate.finish_stage(dt_s, pressure ? StageKind::Pressure : StageKind::Advective);
    *this = std::move(candidate);
}

void CartesianCGridMomentumFluxRegister::stage_pressure(
    const Field3D& p, double dt_s, double dx_m, double dy_m, double dz_m)
{
    require_field(p, dt_s, dx_m, dy_m, dz_m);
    clear(staged_);
    const double x_scale = dt_s * dy_m * dz_m;
    const double y_scale = dt_s * dx_m * dz_m;
    const int k_count = footprint_.k_end - footprint_.k_begin;
    for (int j = footprint_.j_begin; j < footprint_.j_end; ++j)
        for (int k = footprint_.k_begin; k < footprint_.k_end; ++k)
        {
            const std::size_t bin = static_cast<std::size_t>(
                (j - footprint_.j_begin) / x_ratio_ * k_count +
                k - footprint_.k_begin);
            add(staged_.west, 0, bin, x_scale * p[footprint_.i_begin][j][k]);
            add(staged_.east, 0, bin, x_scale * p[footprint_.i_end][j][k]);
        }
    for (int i = footprint_.i_begin; i < footprint_.i_end; ++i)
        for (int k = footprint_.k_begin; k < footprint_.k_end; ++k)
        {
            const std::size_t bin = static_cast<std::size_t>(
                (i - footprint_.i_begin) / y_ratio_ * k_count +
                k - footprint_.k_begin);
            add(staged_.south, 1, bin, y_scale * p[i][footprint_.j_begin][k]);
            add(staged_.north, 1, bin, y_scale * p[i][footprint_.j_end][k]);
        }
    finish_stage(dt_s, StageKind::Pressure);
}

void CartesianCGridMomentumFluxRegister::stage_advective_mapped(
    const Field3D& rho, const Field3D& u,
    const Field3D& v, const Field3D& w,
    const TerrainMetrics3D& metrics, double dt_s,
    double frame_u_ms, double frame_v_ms)
{
    require_fields(
        rho, u, v, w, dt_s, metrics.dx, metrics.dy, metrics.dzeta);
    terrain_transport::validate_cgrid_shapes(metrics, rho, u, v, w);
    clear(staged_);
    const double x_scale = dt_s * metrics.dy * metrics.dzeta;
    const double y_scale = dt_s * metrics.dx * metrics.dzeta;
    const int k_count = footprint_.k_end - footprint_.k_begin;
    using namespace cartesian_cgrid_momentum_flux;
    for (int j = footprint_.j_begin; j < footprint_.j_end; ++j)
        for (int k = footprint_.k_begin; k < footprint_.k_end; ++k)
        {
            const std::size_t bin = static_cast<std::size_t>(
                (j - footprint_.j_begin) / x_ratio_ * k_count
                + k - footprint_.k_begin);
            const int west_scalar = footprint_.i_begin;
            const int east_scalar = footprint_.i_end;
            const int west_face = footprint_.i_begin - 1;
            const int east_face = footprint_.i_end - 1;
            const double west_metric = metrics.Jx_face(west_scalar, j, k);
            const double east_metric = metrics.Jx_face(east_scalar, j, k);
            add(staged_.west, 0, bin, x_scale * west_metric
                * x_flux_x(rho, u, west_scalar, j, k, frame_u_ms));
            add(staged_.east, 0, bin, x_scale * east_metric
                * x_flux_x(rho, u, east_scalar, j, k, frame_u_ms));
            add(staged_.west, 1, bin, x_scale * west_metric
                * x_flux_y(rho, u, v, west_face, j, k, frame_u_ms));
            add(staged_.east, 1, bin, x_scale * east_metric
                * x_flux_y(rho, u, v, east_face, j, k, frame_u_ms));
            add(staged_.west, 2, bin, x_scale * west_metric
                * x_flux_z(rho, u, w, west_face, j, k, frame_u_ms));
            add(staged_.east, 2, bin, x_scale * east_metric
                * x_flux_z(rho, u, w, east_face, j, k, frame_u_ms));
        }
    for (int i = footprint_.i_begin; i < footprint_.i_end; ++i)
        for (int k = footprint_.k_begin; k < footprint_.k_end; ++k)
        {
            const std::size_t bin = static_cast<std::size_t>(
                (i - footprint_.i_begin) / y_ratio_ * k_count
                + k - footprint_.k_begin);
            const int south_scalar = footprint_.j_begin;
            const int north_scalar = footprint_.j_end;
            const int south_face = footprint_.j_begin - 1;
            const int north_face = footprint_.j_end - 1;
            const double south_metric = metrics.Jy_face(i, south_scalar, k);
            const double north_metric = metrics.Jy_face(i, north_scalar, k);
            add(staged_.south, 0, bin, y_scale * south_metric
                * y_flux_x(rho, u, v, i, south_face, k, frame_v_ms));
            add(staged_.north, 0, bin, y_scale * north_metric
                * y_flux_x(rho, u, v, i, north_face, k, frame_v_ms));
            add(staged_.south, 1, bin, y_scale * south_metric
                * y_flux_y(rho, v, i, south_scalar, k, frame_v_ms));
            add(staged_.north, 1, bin, y_scale * north_metric
                * y_flux_y(rho, v, i, north_scalar, k, frame_v_ms));
            add(staged_.south, 2, bin, y_scale * south_metric
                * y_flux_z(rho, v, w, i, south_face, k, frame_v_ms));
            add(staged_.north, 2, bin, y_scale * north_metric
                * y_flux_z(rho, v, w, i, north_face, k, frame_v_ms));
        }
    finish_stage(dt_s, StageKind::Advective);
}

void CartesianCGridMomentumFluxRegister::stage_pressure_mapped(
    const Field3D& p, const TerrainMetrics3D& metrics, double dt_s)
{
    require_field(p, dt_s, metrics.dx, metrics.dy, metrics.dzeta);
    terrain_transport::validate_geometry(metrics);
    clear(staged_);
    const double x_scale = dt_s * metrics.dy * metrics.dzeta;
    const double y_scale = dt_s * metrics.dx * metrics.dzeta;
    const int k_count = footprint_.k_end - footprint_.k_begin;
    for (int j = footprint_.j_begin; j < footprint_.j_end; ++j)
        for (int k = footprint_.k_begin; k < footprint_.k_end; ++k)
        {
            const std::size_t bin = static_cast<std::size_t>(
                (j - footprint_.j_begin) / x_ratio_ * k_count
                + k - footprint_.k_begin);
            add(staged_.west, 0, bin, x_scale
                * metrics.Jx_face(footprint_.i_begin, j, k)
                * p[footprint_.i_begin][j][k]);
            add(staged_.east, 0, bin, x_scale
                * metrics.Jx_face(footprint_.i_end, j, k)
                * p[footprint_.i_end][j][k]);
        }
    for (int i = footprint_.i_begin; i < footprint_.i_end; ++i)
        for (int k = footprint_.k_begin; k < footprint_.k_end; ++k)
        {
            const std::size_t bin = static_cast<std::size_t>(
                (i - footprint_.i_begin) / y_ratio_ * k_count
                + k - footprint_.k_begin);
            add(staged_.south, 1, bin, y_scale
                * metrics.Jy_face(i, footprint_.j_begin, k)
                * p[i][footprint_.j_begin][k]);
            add(staged_.north, 1, bin, y_scale
                * metrics.Jy_face(i, footprint_.j_end, k)
                * p[i][footprint_.j_end][k]);
        }
    finish_stage(dt_s, StageKind::Pressure);
}

void CartesianCGridMomentumFluxRegister::stage_mapped_tendency_closure(
    const Field3D& rho, const Field3D& du_dt,
    const Field3D& dv_dt, const Field3D& dw_dt,
    const TerrainMetrics3D& metrics, double dt_s)
{
    if (!staged_ready_
        || (staged_kind_ != StageKind::Advective
            && staged_kind_ != StageKind::Pressure))
        throw std::logic_error(
            "mapped tendency closure requires a staged momentum operator");
    require_fields(
        rho, du_dt, dv_dt, dw_dt,
        dt_s, metrics.dx, metrics.dy, metrics.dzeta);
    std::array<double, 3> total{};
    const auto scalar_mass = [&](int i, int j, int k) {
        return static_cast<double>(rho[i][j][k])
            * metrics.J(i, j, k) * metrics.dx * metrics.dy
            * metrics.dzeta;
    };
    for (int face = footprint_.i_begin - 1;
         face < footprint_.i_end; ++face)
        for (int j = footprint_.j_begin; j < footprint_.j_end; ++j)
            for (int k = 0; k < nz_; ++k)
                total[0] += 0.5 * (
                    scalar_mass(face, j, k)
                    + scalar_mass(face + 1, j, k))
                    * du_dt[face][j][k] * dt_s;
    for (int i = footprint_.i_begin; i < footprint_.i_end; ++i)
        for (int face = footprint_.j_begin - 1;
             face < footprint_.j_end; ++face)
            for (int k = 0; k < nz_; ++k)
                total[1] += 0.5 * (
                    scalar_mass(i, face, k)
                    + scalar_mass(i, face + 1, k))
                    * dv_dt[i][face][k] * dt_s;
    for (int i = footprint_.i_begin; i < footprint_.i_end; ++i)
        for (int j = footprint_.j_begin; j < footprint_.j_end; ++j)
            for (int k = 0; k < nz_; ++k)
            {
                double mass = 0.5 * scalar_mass(i, j, k);
                if (k + 1 < nz_)
                    mass += 0.5 * scalar_mass(i, j, k + 1);
                total[2] += mass * dw_dt[i][j][k] * dt_s;
            }
    std::array<double, 3> transport{};
    for (std::size_t component = 0; component < 3; ++component)
    {
        for (std::size_t bin = 0; bin < staged_.west.component[component].size(); ++bin)
            transport[component] += staged_.west.component[component][bin]
                - staged_.east.component[component][bin];
        for (std::size_t bin = 0; bin < staged_.south.component[component].size(); ++bin)
            transport[component] += staged_.south.component[component][bin]
                - staged_.north.component[component][bin];
        total[component] -= transport[component];
    }
    if (staged_kind_ == StageKind::Advective)
        stage_source_impulse(total);
    else
        stage_pressure_source_impulse(total);
}

void CartesianCGridMomentumFluxRegister::stage_source_impulse(
    std::array<double, 3> impulse)
{
    if (!staged_ready_ || staged_kind_ != StageKind::Advective)
        throw std::logic_error(
            "momentum source impulse requires a staged flux contribution");
    for (std::size_t component = 0; component < impulse.size(); ++component)
    {
        if (!std::isfinite(impulse[component]) ||
            !std::isfinite(advective_.source_impulse[component] +
                           impulse[component]))
            throw std::runtime_error("momentum source impulse is non-finite");
        staged_.source_impulse[component] = impulse[component];
    }
    staged_validated_ = false;
}

void CartesianCGridMomentumFluxRegister::stage_pressure_source_impulse(
    std::array<double, 3> impulse)
{
    if (!staged_ready_ || staged_kind_ != StageKind::Pressure)
        throw std::logic_error(
            "momentum pressure source impulse requires a staged pressure contribution");
    for (std::size_t component = 0; component < impulse.size(); ++component)
    {
        if (!std::isfinite(impulse[component]) ||
            !std::isfinite(pressure_.source_impulse[component] +
                           impulse[component]))
            throw std::runtime_error(
                "momentum pressure source impulse is non-finite");
        staged_.source_impulse[component] = impulse[component];
    }
    staged_validated_ = false;
}

std::array<double, 3> CartesianCGridMomentumFluxRegister::native_dual_inventory(const Field3D& rho,
                                                                                const Field3D& u,
                                                                                const Field3D& v,
                                                                                const Field3D& w,
                                                                                double         dx_m,
                                                                                double         dy_m,
                                                                                double         dz_m) const
{
    for (const auto* field : {&u, &v, &w})
    {
        if (field->size_r() != rho.size_r() || field->size_th() != rho.size_th() || field->size_z() != rho.size_z())
        {
            throw std::invalid_argument("momentum inventory velocity shape differs from density");
        }
    }
    return native_dual_inventory_buffers(rho, {u.data(), v.data(), w.data()}, rho.size(), dx_m, dy_m, dz_m);
}

std::array<double, 3> CartesianCGridMomentumFluxRegister::native_dual_inventory_mapped(
    const Field3D& rho, const Field3D& u, const Field3D& v, const Field3D& w, const TerrainMetrics3D& metrics) const
{
    for (const auto* field : {&u, &v, &w})
    {
        if (field->size_r() != rho.size_r() || field->size_th() != rho.size_th() || field->size_z() != rho.size_z())
        {
            throw std::invalid_argument("momentum inventory velocity shape differs from density");
        }
    }
    return native_dual_inventory_buffers(
        rho, {u.data(), v.data(), w.data()}, rho.size(), metrics.dx, metrics.dy, metrics.dzeta, &metrics);
}

std::array<double, 3>
CartesianCGridMomentumFluxRegister::native_dual_inventory_buffers(const Field3D&              rho,
                                                                  std::array<const float*, 3> velocity,
                                                                  std::size_t                 cells,
                                                                  double                      dx_m,
                                                                  double                      dy_m,
                                                                  double                      dz_m,
                                                                  const TerrainMetrics3D*     terrain) const
{
    require_field(rho, 1.0, dx_m, dy_m, dz_m);
    if (cells != rho.size() || !velocity[0] || !velocity[1] || !velocity[2])
    {
        throw std::invalid_argument("momentum inventory requires complete velocity buffers");
    }
    for (std::size_t index = 0; index < cells; ++index)
    {
        if (rho.data()[index] <= 0 || !std::isfinite(velocity[0][index]) || !std::isfinite(velocity[1][index]) ||
            !std::isfinite(velocity[2][index]))
        {
            throw std::invalid_argument("momentum inventory requires positive density and finite velocities");
        }
    }
    const auto offset = [&](int i, int j, int k) { return (std::size_t(i) * nth_ + j) * nz_ + k; };
    if (terrain)
    {
        const auto& metrics = *terrain;
        if (dx_m != metrics.dx || dy_m != metrics.dy || dz_m != metrics.dzeta)
        {
            throw std::invalid_argument("momentum inventory metrics disagree with grid spacing");
        }
        terrain_transport::validate_cgrid_shapes(metrics, rho, rho, rho, rho);
        const auto scalar_mass = [&](int i, int j, int k)
        { return static_cast<double>(rho[i][j][k]) * metrics.J(i, j, k) * metrics.dx * metrics.dy * metrics.dzeta; };
        std::array<double, 3> result{};
        for (int face = footprint_.i_begin - 1; face < footprint_.i_end; ++face)
            for (int j = footprint_.j_begin; j < footprint_.j_end; ++j)
                for (int k = 0; k < nz_; ++k)
                    result[0] +=
                        0.5 * (scalar_mass(face, j, k) + scalar_mass(face + 1, j, k)) * velocity[0][offset(face, j, k)];
        for (int i = footprint_.i_begin; i < footprint_.i_end; ++i)
            for (int face = footprint_.j_begin - 1; face < footprint_.j_end; ++face)
                for (int k = 0; k < nz_; ++k)
                    result[1] +=
                        0.5 * (scalar_mass(i, face, k) + scalar_mass(i, face + 1, k)) * velocity[1][offset(i, face, k)];
        for (int i = footprint_.i_begin; i < footprint_.i_end; ++i)
            for (int j = footprint_.j_begin; j < footprint_.j_end; ++j)
                for (int k = 0; k < nz_; ++k)
                {
                    double mass = 0.5 * scalar_mass(i, j, k);
                    if (k + 1 < nz_)
                        mass += 0.5 * scalar_mass(i, j, k + 1);
                    result[2] += mass * velocity[2][offset(i, j, k)];
                }
        for (double value : result)
            if (!std::isfinite(value))
                throw std::runtime_error("mapped C-grid native-dual momentum inventory is non-finite");
        return result;
    }
    const double          volume = dx_m * dy_m * dz_m;
    const double          area   = dx_m * dy_m;
    std::array<double, 3> result{};
    for (int face = footprint_.i_begin - 1; face < footprint_.i_end; ++face)
        for (int j = footprint_.j_begin; j < footprint_.j_end; ++j)
            for (int k = 0; k < nz_; ++k)
                result[0] += 0.5 * volume *
                             (static_cast<double>(rho[face][j][k]) + static_cast<double>(rho[face + 1][j][k])) *
                             velocity[0][offset(face, j, k)];
    for (int i = footprint_.i_begin; i < footprint_.i_end; ++i)
        for (int face = footprint_.j_begin - 1; face < footprint_.j_end; ++face)
            for (int k = 0; k < nz_; ++k)
                result[1] += 0.5 * volume *
                             (static_cast<double>(rho[i][face][k]) + static_cast<double>(rho[i][face + 1][k])) *
                             velocity[1][offset(i, face, k)];
    for (int i = footprint_.i_begin; i < footprint_.i_end; ++i)
        for (int j = footprint_.j_begin; j < footprint_.j_end; ++j)
            for (int k = 0; k < nz_; ++k)
            {
                double mass = 0.5 * area * dz_m * rho[i][j][k];
                if (k + 1 < nz_)
                    mass += 0.5 * area * dz_m * rho[i][j][k + 1];
                result[2] += mass * velocity[2][offset(i, j, k)];
            }
    for (double value : result)
        if (!std::isfinite(value))
            throw std::runtime_error("C-grid native-dual momentum inventory is non-finite");
    return result;
}

std::array<double, 3> CartesianCGridMomentumFluxRegister::native_dual_tendency_impulse(const Field3D& rho,
                                                                                       const Field3D& du_dt,
                                                                                       const Field3D& dv_dt,
                                                                                       const Field3D& dw_dt,
                                                                                       double         dt_s,
                                                                                       double         dx_m,
                                                                                       double         dy_m,
                                                                                       double         dz_m) const
{
    require_fields(rho, du_dt, dv_dt, dw_dt, dt_s, dx_m, dy_m, dz_m);
    const double          volume = dx_m * dy_m * dz_m;
    const double          area   = dx_m * dy_m;
    std::array<double, 3> result{};
    for (int face = footprint_.i_begin - 1; face < footprint_.i_end; ++face)
        for (int j = footprint_.j_begin; j < footprint_.j_end; ++j)
            for (int k = 0; k < nz_; ++k)
                result[0] += 0.5 * volume * dt_s *
                             (static_cast<double>(rho[face][j][k]) + static_cast<double>(rho[face + 1][j][k])) *
                             du_dt[face][j][k];
    for (int i = footprint_.i_begin; i < footprint_.i_end; ++i)
        for (int face = footprint_.j_begin - 1;
             face < footprint_.j_end; ++face)
            for (int k = 0; k < nz_; ++k)
                result[1] += 0.5 * volume * dt_s *
                    (static_cast<double>(rho[i][face][k]) +
                     static_cast<double>(rho[i][face + 1][k])) *
                    dv_dt[i][face][k];
    for (int i = footprint_.i_begin; i < footprint_.i_end; ++i)
        for (int j = footprint_.j_begin; j < footprint_.j_end; ++j)
            for (int k = 0; k < nz_; ++k)
            {
                double mass = 0.5 * area * dz_m * rho[i][j][k];
                if (k + 1 < nz_)
                    mass += 0.5 * area * dz_m * rho[i][j][k + 1];
                result[2] += mass * dt_s * dw_dt[i][j][k];
            }
    for (double value : result)
        if (!std::isfinite(value))
            throw std::runtime_error(
                "C-grid native-dual momentum tendency impulse is non-finite");
    return result;
}

std::array<double, 3>
CartesianCGridMomentumFluxRegister::native_dual_tendency_impulse_mapped(
    const Field3D& rho, const Field3D& du_dt,
    const Field3D& dv_dt, const Field3D& dw_dt,
    double dt_s, const TerrainMetrics3D& metrics) const
{
    const auto inventory = native_dual_inventory_mapped(
        rho, du_dt, dv_dt, dw_dt, metrics);
    std::array<double, 3> result{};
    for (std::size_t component = 0; component < result.size(); ++component)
    {
        result[component] = inventory[component] * dt_s;
        if (!std::isfinite(result[component]))
            throw std::runtime_error(
                "mapped C-grid native-dual tendency impulse is non-finite");
    }
    return result;
}

std::array<double, 3>
CartesianCGridMomentumFluxRegister::native_dual_density_tendency_impulse(
    const Field3D& rho, const Field3D& drho_dt,
    const Field3D& u, const Field3D& v, const Field3D& w,
    double dt_s, double dx_m, double dy_m, double dz_m) const
{
    require_fields(rho, u, v, w, dt_s, dx_m, dy_m, dz_m);
    require_field(drho_dt, dt_s, dx_m, dy_m, dz_m);
    const double volume = dx_m * dy_m * dz_m;
    const double area = dx_m * dy_m;
    std::array<double, 3> result{};
    for (int face = footprint_.i_begin - 1;
         face < footprint_.i_end; ++face)
        for (int j = footprint_.j_begin; j < footprint_.j_end; ++j)
            for (int k = 0; k < nz_; ++k)
                result[0] += 0.5 * volume * dt_s *
                    (static_cast<double>(drho_dt[face][j][k]) +
                     static_cast<double>(drho_dt[face + 1][j][k])) *
                    u[face][j][k];
    for (int i = footprint_.i_begin; i < footprint_.i_end; ++i)
        for (int face = footprint_.j_begin - 1;
             face < footprint_.j_end; ++face)
            for (int k = 0; k < nz_; ++k)
                result[1] += 0.5 * volume * dt_s *
                    (static_cast<double>(drho_dt[i][face][k]) +
                     static_cast<double>(drho_dt[i][face + 1][k])) *
                    v[i][face][k];
    for (int i = footprint_.i_begin; i < footprint_.i_end; ++i)
        for (int j = footprint_.j_begin; j < footprint_.j_end; ++j)
            for (int k = 0; k < nz_; ++k)
            {
                double mass_rate = 0.5 * area * dz_m * drho_dt[i][j][k];
                if (k + 1 < nz_)
                    mass_rate += 0.5 * area * dz_m * drho_dt[i][j][k + 1];
                result[2] += mass_rate * dt_s * w[i][j][k];
            }
    for (double value : result)
        if (!std::isfinite(value))
            throw std::runtime_error(
                "C-grid native-dual density tendency impulse is non-finite");
    return result;
}

std::array<double, 3>
CartesianCGridMomentumFluxRegister::
native_dual_density_tendency_impulse_mapped(
    const Field3D& rho, const Field3D& drho_dt,
    const Field3D& u, const Field3D& v, const Field3D& w,
    double dt_s, const TerrainMetrics3D& metrics) const
{
    require_fields(
        rho, u, v, w, dt_s, metrics.dx, metrics.dy, metrics.dzeta);
    require_field(
        drho_dt, dt_s, metrics.dx, metrics.dy, metrics.dzeta);
    const auto scalar_rate = [&](int i, int j, int k) {
        return static_cast<double>(drho_dt[i][j][k])
            * metrics.J(i, j, k) * metrics.dx * metrics.dy
            * metrics.dzeta;
    };
    std::array<double, 3> result{};
    for (int face = footprint_.i_begin - 1;
         face < footprint_.i_end; ++face)
        for (int j = footprint_.j_begin; j < footprint_.j_end; ++j)
            for (int k = 0; k < nz_; ++k)
                result[0] += 0.5 * (
                    scalar_rate(face, j, k)
                    + scalar_rate(face + 1, j, k))
                    * u[face][j][k] * dt_s;
    for (int i = footprint_.i_begin; i < footprint_.i_end; ++i)
        for (int face = footprint_.j_begin - 1;
             face < footprint_.j_end; ++face)
            for (int k = 0; k < nz_; ++k)
                result[1] += 0.5 * (
                    scalar_rate(i, face, k)
                    + scalar_rate(i, face + 1, k))
                    * v[i][face][k] * dt_s;
    for (int i = footprint_.i_begin; i < footprint_.i_end; ++i)
        for (int j = footprint_.j_begin; j < footprint_.j_end; ++j)
            for (int k = 0; k < nz_; ++k)
            {
                double rate = 0.5 * scalar_rate(i, j, k);
                if (k + 1 < nz_)
                    rate += 0.5 * scalar_rate(i, j, k + 1);
                result[2] += rate * w[i][j][k] * dt_s;
            }
    for (double value : result)
        if (!std::isfinite(value))
            throw std::runtime_error(
                "mapped C-grid native-dual density impulse is non-finite");
    return result;
}

void CartesianCGridMomentumFluxRegister::accumulate_operator_source_impulse(
    const std::array<double, 3>& impulse)
{
    std::array<double, 3> candidate = advective_.source_impulse;
    for (std::size_t component = 0; component < candidate.size(); ++component)
    {
        if (!std::isfinite(impulse[component]) ||
            !std::isfinite(candidate[component] + impulse[component]))
            throw std::runtime_error(
                "C-grid momentum operator source impulse is non-finite");
        candidate[component] += impulse[component];
    }
    advective_.source_impulse = candidate;
}

std::array<double, 3>
CartesianCGridMomentumFluxRegister::last_committed_operator_impulse() const
{
    if (last_committed_kind_ == StageKind::None)
        throw std::logic_error(
            "C-grid momentum register has no committed operator contribution");
    return last_committed_operator_impulse_;
}

std::array<double, 3>
CartesianCGridMomentumFluxRegister::pending_committed_operator_impulse() const
{
    if (pending_committed_kind_ == StageKind::None)
        throw std::logic_error(
            "C-grid momentum register has no unapplied operator contribution");
    return pending_committed_operator_impulse_;
}

void CartesianCGridMomentumFluxRegister::accumulate_operator_storage_impulse(
    const std::array<double, 3>& impulse)
{
    if (pending_committed_kind_ == StageKind::None)
        throw std::logic_error(
            "C-grid momentum storage impulse has no committed operator");
    CartesianMomentumInterfaceFluxes& destination =
        pending_committed_kind_ == StageKind::Advective ? advective_ : pressure_;
    auto candidate = destination.storage_impulse;
    for (std::size_t component = 0; component < candidate.size(); ++component)
    {
        if (!std::isfinite(impulse[component]) ||
            !std::isfinite(candidate[component] + impulse[component]))
            throw std::overflow_error(
                "C-grid momentum storage impulse overflowed");
        candidate[component] += impulse[component];
    }
    destination.storage_impulse = candidate;
    pending_committed_kind_ = StageKind::None;
    pending_committed_operator_impulse_ = {};
}

void CartesianCGridMomentumFluxRegister::accumulate_boundary_source_impulse(
    const std::array<double, 3>& impulse)
{
    accumulate_operator_source_impulse(impulse);
}

void CartesianCGridMomentumFluxRegister::
accumulate_mapped_endpoint_storage_impulse(
    const std::array<double, 3>& impulse)
{
    auto candidate = advective_.storage_impulse;
    for (std::size_t component = 0; component < candidate.size(); ++component)
    {
        if (!std::isfinite(impulse[component])
            || !std::isfinite(candidate[component] + impulse[component]))
            throw std::overflow_error(
                "mapped C-grid endpoint storage impulse overflowed");
        candidate[component] += impulse[component];
    }
    advective_.storage_impulse = candidate;
}

void CartesianCGridMomentumFluxRegister::attribute_operator_commit(
    const std::array<double, 3>& requested_impulse,
    const std::array<double, 3>& committed_impulse)
{
    if (pending_committed_kind_ == StageKind::None)
        throw std::logic_error(
            "C-grid momentum commit attribution has no pending operator");
    CartesianMomentumInterfaceFluxes& destination =
        pending_committed_kind_ == StageKind::Advective ? advective_ : pressure_;
    auto source_candidate = destination.source_impulse;
    auto storage_candidate = destination.storage_impulse;
    for (std::size_t component = 0; component < 3; ++component)
    {
        const double split_coupling = requested_impulse[component] -
                                      pending_committed_operator_impulse_[component];
        const double storage = committed_impulse[component] -
                               requested_impulse[component];
        if (!std::isfinite(split_coupling) || !std::isfinite(storage) ||
            !std::isfinite(source_candidate[component] + split_coupling) ||
            !std::isfinite(storage_candidate[component] + storage))
            throw std::overflow_error(
                "C-grid momentum commit attribution overflowed");
        source_candidate[component] += split_coupling;
        storage_candidate[component] += storage;
    }
    destination.source_impulse = source_candidate;
    destination.storage_impulse = storage_candidate;
    pending_committed_kind_ = StageKind::None;
    pending_committed_operator_impulse_ = {};
}

void CartesianCGridMomentumFluxRegister::attribute_continuity_commit(
    const std::array<double, 3>& requested_impulse,
    const std::array<double, 3>& committed_impulse)
{
    auto source_candidate = advective_.source_impulse;
    auto storage_candidate = advective_.storage_impulse;
    for (std::size_t component = 0; component < 3; ++component)
    {
        const double storage = committed_impulse[component] -
                               requested_impulse[component];
        if (!std::isfinite(requested_impulse[component]) ||
            !std::isfinite(storage) ||
            !std::isfinite(source_candidate[component] +
                           requested_impulse[component]) ||
            !std::isfinite(storage_candidate[component] + storage))
            throw std::overflow_error(
                "C-grid momentum continuity attribution overflowed");
        source_candidate[component] += requested_impulse[component];
        storage_candidate[component] += storage;
    }
    advective_.source_impulse = source_candidate;
    advective_.storage_impulse = storage_candidate;
}

void CartesianCGridMomentumFluxRegister::validate_staged_commit()
{
    if (!staged_ready_ || staged_kind_ == StageKind::None)
        throw std::logic_error(
            "C-grid momentum flux commit has no staged contribution");
    const CartesianMomentumInterfaceFluxes& destination =
        staged_kind_ == StageKind::Advective ? advective_ : pressure_;
    const auto validate_side = [](
        const CartesianMomentumInterfaceSide& committed,
        const CartesianMomentumInterfaceSide& staged)
    {
        for (std::size_t component = 0; component < 3; ++component)
        {
            if (committed.component[component].size() !=
                staged.component[component].size())
                throw std::logic_error(
                    "C-grid momentum flux staged strip shape changed");
            for (std::size_t bin = 0;
                 bin < committed.component[component].size(); ++bin)
                if (!std::isfinite(committed.component[component][bin] +
                                   staged.component[component][bin]))
                    throw std::overflow_error(
                        "C-grid momentum flux cumulative strip overflowed");
        }
    };
    validate_side(destination.west, staged_.west);
    validate_side(destination.east, staged_.east);
    validate_side(destination.south, staged_.south);
    validate_side(destination.north, staged_.north);
    for (std::size_t component = 0; component < 3; ++component)
        if (!std::isfinite(destination.source_impulse[component] +
                           staged_.source_impulse[component]))
            throw std::overflow_error("C-grid momentum source impulse overflowed");
    if (!std::isfinite(destination.accumulated_dt_s + staged_.accumulated_dt_s))
        throw std::overflow_error("C-grid momentum flux cumulative duration overflowed");
    if (destination.contribution_count ==
        std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("C-grid momentum flux contribution count overflowed");
    if (pending_committed_kind_ != StageKind::None &&
        pending_committed_kind_ != staged_kind_)
        throw std::logic_error(
            "C-grid momentum operator contribution was not attributed before the next stage");
    std::array<double, 3> operator_impulse = staged_.source_impulse;
    for (std::size_t component = 0; component < 3; ++component)
    {
        for (std::size_t bin = 0;
             bin < staged_.west.component[component].size(); ++bin)
            operator_impulse[component] +=
                staged_.west.component[component][bin] -
                staged_.east.component[component][bin];
        for (std::size_t bin = 0;
             bin < staged_.south.component[component].size(); ++bin)
            operator_impulse[component] +=
                staged_.south.component[component][bin] -
                staged_.north.component[component][bin];
        if (!std::isfinite(pending_committed_operator_impulse_[component] +
                           operator_impulse[component]))
            throw std::overflow_error(
                "C-grid momentum pending operator impulse overflowed");
    }
    staged_validated_ = true;
}

void CartesianCGridMomentumFluxRegister::commit_staged() noexcept
{
    if (!staged_ready_ || !staged_validated_)
        return;
    CartesianMomentumInterfaceFluxes& destination =
        staged_kind_ == StageKind::Advective ? advective_ : pressure_;
    std::array<double, 3> operator_impulse = staged_.source_impulse;
    for (std::size_t component = 0; component < 3; ++component)
    {
        for (std::size_t bin = 0;
             bin < staged_.west.component[component].size(); ++bin)
            operator_impulse[component] +=
                staged_.west.component[component][bin] -
                staged_.east.component[component][bin];
        for (std::size_t bin = 0;
             bin < staged_.south.component[component].size(); ++bin)
            operator_impulse[component] +=
                staged_.south.component[component][bin] -
                staged_.north.component[component][bin];
    }
    add_side(destination.west, staged_.west);
    add_side(destination.east, staged_.east);
    add_side(destination.south, staged_.south);
    add_side(destination.north, staged_.north);
    for (std::size_t component = 0; component < 3; ++component)
        destination.source_impulse[component] += staged_.source_impulse[component];
    destination.accumulated_dt_s += staged_.accumulated_dt_s;
    ++destination.contribution_count;
    last_committed_kind_ = staged_kind_;
    last_committed_operator_impulse_ = operator_impulse;
    pending_committed_kind_ = staged_kind_;
    for (std::size_t component = 0; component < 3; ++component)
        pending_committed_operator_impulse_[component] +=
            operator_impulse[component];
    clear(staged_);
    staged_ready_ = false;
    staged_validated_ = false;
    staged_kind_ = StageKind::None;
}

bool CartesianCGridMomentumFluxRegister::configured() const noexcept
{
    return configured_;
}

const CartesianInterfaceFootprint&
CartesianCGridMomentumFluxRegister::footprint() const noexcept
{
    return footprint_;
}

const CartesianMomentumInterfaceFluxes&
CartesianCGridMomentumFluxRegister::advective_fluxes() const noexcept
{
    return advective_;
}

const CartesianMomentumInterfaceFluxes&
CartesianCGridMomentumFluxRegister::pressure_fluxes() const noexcept
{
    return pressure_;
}

void CartesianCGridMomentumFluxRegister::allocate(
    CartesianMomentumInterfaceFluxes& fluxes,
    std::size_t x_count, std::size_t y_count)
{
    for (std::size_t component = 0; component < 3; ++component)
    {
        fluxes.west.component[component].assign(x_count, 0.0);
        fluxes.east.component[component].assign(x_count, 0.0);
        fluxes.south.component[component].assign(y_count, 0.0);
        fluxes.north.component[component].assign(y_count, 0.0);
    }
    fluxes.source_impulse = {};
    fluxes.storage_impulse = {};
    fluxes.accumulated_dt_s = 0.0;
    fluxes.contribution_count = 0;
}

void CartesianCGridMomentumFluxRegister::clear(
    CartesianMomentumInterfaceFluxes& fluxes) noexcept
{
    for (std::size_t component = 0; component < 3; ++component)
    {
        std::fill(fluxes.west.component[component].begin(),
                  fluxes.west.component[component].end(), 0.0);
        std::fill(fluxes.east.component[component].begin(),
                  fluxes.east.component[component].end(), 0.0);
        std::fill(fluxes.south.component[component].begin(),
                  fluxes.south.component[component].end(), 0.0);
        std::fill(fluxes.north.component[component].begin(),
                  fluxes.north.component[component].end(), 0.0);
    }
    fluxes.source_impulse = {};
    fluxes.storage_impulse = {};
    fluxes.accumulated_dt_s = 0.0;
    fluxes.contribution_count = 0;
}

void CartesianCGridMomentumFluxRegister::add(
    CartesianMomentumInterfaceSide& side,
    std::size_t component, std::size_t bin, double value)
{
    if (!std::isfinite(value) ||
        !std::isfinite(side.component[component][bin] + value))
        throw std::runtime_error("C-grid momentum interface flux is non-finite");
    side.component[component][bin] += value;
}

void CartesianCGridMomentumFluxRegister::add_side(
    CartesianMomentumInterfaceSide& destination,
    const CartesianMomentumInterfaceSide& source) noexcept
{
    for (std::size_t component = 0; component < 3; ++component)
        for (std::size_t bin = 0;
             bin < destination.component[component].size(); ++bin)
            destination.component[component][bin] +=
                source.component[component][bin];
}

void CartesianCGridMomentumFluxRegister::require_field(
    const Field3D& field, double dt_s,
    double dx_m, double dy_m, double dz_m) const
{
    if (!configured_ || field.size_r() != nr_ || field.size_th() != nth_ ||
        field.size_z() != nz_ || !std::isfinite(dt_s) ||
        !std::isfinite(dx_m) || !std::isfinite(dy_m) ||
        !std::isfinite(dz_m) || dt_s <= 0.0 || dx_m <= 0.0 ||
        dy_m <= 0.0 || dz_m <= 0.0)
        throw std::invalid_argument(
            "C-grid momentum interface flux requires matching fields and finite metrics");
    for (std::size_t index = 0; index < field.size(); ++index)
        if (!std::isfinite(static_cast<double>(field.data()[index])))
            throw std::runtime_error("C-grid momentum interface input is non-finite");
}

void CartesianCGridMomentumFluxRegister::require_fields(
    const Field3D& rho, const Field3D& u, const Field3D& v, const Field3D& w,
    double dt_s, double dx_m, double dy_m, double dz_m) const
{
    require_field(rho, dt_s, dx_m, dy_m, dz_m);
    require_field(u, dt_s, dx_m, dy_m, dz_m);
    require_field(v, dt_s, dx_m, dy_m, dz_m);
    require_field(w, dt_s, dx_m, dy_m, dz_m);
    for (std::size_t index = 0; index < rho.size(); ++index)
        if (rho.data()[index] <= 0.0f)
            throw std::runtime_error(
                "C-grid momentum interface density must be positive");
}

void CartesianCGridMomentumFluxRegister::finish_stage(
    double dt_s, StageKind kind)
{
    const CartesianMomentumInterfaceFluxes& destination =
        kind == StageKind::Advective ? advective_ : pressure_;
    if (!std::isfinite(destination.accumulated_dt_s + dt_s))
        throw std::runtime_error("C-grid momentum interface duration is non-finite");
    staged_.accumulated_dt_s = dt_s;
    staged_.contribution_count = 1;
    staged_ready_ = true;
    staged_validated_ = false;
    staged_kind_ = kind;
}

} // namespace tmv
