// Shared eddy-viscosity and conservative SGS diffusion operators.

#include "eddy_viscosity.hpp"
#include "collocated_eddy_viscosity.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "terrain/conservative_transport.hpp"
#include "util/grid_metric_utils.hpp"
#include "core/runtime/runtime_config.hpp"
#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif


namespace tmv::eddy_viscosity
{

namespace
{
/**
 * @brief Boundary-aware finite difference in a horizontal dimension.
 *
 * Uses central differencing when not at a boundary, otherwise falls
 * back to one-sided (forward or backward) differencing.
 *
 * @param f_minus Value at index-1 (only valid when idx > 0).
 * @param f_center Value at current index.
 * @param f_plus Value at index+1 (only valid when idx < n-1).
 * @param dx Grid spacing.
 * @param idx Current index in this dimension.
 * @param n Total extent in this dimension.
 */
double horiz_diff(double f_minus, double f_center, double f_plus,
                  double dx, int idx, int n)
{
    if (idx > 0 && idx < n - 1)
    {
        return (f_plus - f_minus) / (2.0 * dx);
    }
    if (idx == 0)
    {
        return (f_plus - f_center) / dx;
    }
    return (f_center - f_minus) / dx;
}

/**
 * @brief Boundary-aware vertical finite difference.
 *
 * Uses centered differencing over the actual vertical span in the
 * interior, one-sided at top/bottom.
 */
double vert_diff(double f_minus, double f_center, double f_plus,
                 double dz_one_sided, double dz_centered,
                 int k, int nz)
{
    if (k > 0 && k < nz - 1)
    {
        return (f_plus - f_minus) / dz_centered;
    }
    if (k == 0)
    {
        return (f_plus - f_center) / dz_one_sided;
    }
    return (f_center - f_minus) / dz_one_sided;
}

int j_neighbor_prev(int j, int nth, bool periodic)
{
    if (!periodic)
    {
        return std::max(j - 1, 0);
    }
    // C-grid storage carries ghosts at 0 and nth-1. Periodicity wraps the
    // physical scalar rows 1..nth-2, never the allocated ghost rows.
    return (j <= 1) ? nth - 2 : std::min(j - 1, nth - 2);
}

int j_neighbor_next(int j, int nth, bool periodic)
{
    if (!periodic)
    {
        return std::min(j + 1, nth - 1);
    }
    return (j >= nth - 2) ? 1 : std::max(j + 1, 1);
}

int i_neighbor_prev(int i, int nr, bool periodic)
{
    if (!periodic)
    {
        return std::max(i - 1, 0);
    }
    return (i <= 1) ? nr - 2 : std::min(i - 1, nr - 2);
}

int i_neighbor_next(int i, int nr, bool periodic)
{
    if (!periodic)
    {
        return std::min(i + 1, nr - 1);
    }
    return (i >= nr - 2) ? 1 : std::max(i + 1, 1);
}

void require_field(const Field3D* field,
                   const TurbulenceStateView& state,
                   const char* label,
                   bool require_positive,
                   bool require_nonnegative)
{
    if (field == nullptr || field->size_r() != state.NR ||
        field->size_th() != state.NTH || field->size_z() != state.NZ)
    {
        throw std::invalid_argument(
            std::string("turbulence requires shape-matched field ") + label);
    }
    const PhysicalCellDomain domain = state.c_grid
        ? PhysicalCellDomain{0, state.NR, 0, state.NTH, 0, state.NZ}
        : physical_scalar_domain(
            state.NR, state.NTH, state.NZ, StaggerType::Collocated);
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
    {
        const double value = static_cast<double>((*field)(i, j, k));
        if (!std::isfinite(value) || (require_positive && value <= 0.0) ||
            (require_nonnegative && value < 0.0))
        {
            throw std::invalid_argument(
                std::string("turbulence field ") + label +
                " violates its finite physical domain");
        }
    }
}

void require_rho0(const TurbulenceStateView& state)
{
    if (state.rho0 == nullptr ||
        state.rho0->size() != static_cast<std::size_t>(state.NZ))
    {
        throw std::invalid_argument(
            "turbulence requires an exact rho0 base profile");
    }
    for (double density : *state.rho0)
    {
        if (!std::isfinite(density) || density <= 0.0)
        {
            throw std::invalid_argument(
                "turbulence rho0 base density must be positive and finite");
        }
    }
}

void require_output_shape(const Field3D& field,
                          const TurbulenceStateView& state,
                          const char* label)
{
    if (field.size_r() != state.NR || field.size_th() != state.NTH ||
        field.size_z() != state.NZ)
    {
        throw std::invalid_argument(
            std::string("turbulence output must be pre-shaped: ") + label);
    }
}

void require_flat_tensor_geometry(const GridMetrics& grid,
                                  const char* operation)
{
    if (grid.terrain_metrics_active)
    {
        throw std::invalid_argument(
            std::string(operation)
            + " is unavailable with terrain until the physical staggered "
              "tensor-stress divergence is implemented");
    }
}

double terrain_cgrid_strain_sq_centered(
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    int i, int j, int k,
    const std::string& mode)
{
    if (grid.terrain_metrics == nullptr)
        throw std::invalid_argument(
            "terrain C-grid strain requires complete mapped geometry");
    const TerrainMetrics3D& metrics = *grid.terrain_metrics;
    terrain_transport::validate_geometry_shape(metrics);
    const int nx = state.NR;
    const int ny = state.NTH;
    const int nz = state.NZ;
    if (i <= 0 || i >= nx - 1 || j <= 0 || j >= ny - 1
        || k < 0 || k >= nz)
        throw std::out_of_range(
            "terrain C-grid strain requires an interior scalar column");
    const auto shape_matches = [&](const Field3D* field)
    {
        return field != nullptr && field->size_r() == nx
            && field->size_th() == ny && field->size_z() == nz;
    };
    if (!shape_matches(state.u) || !shape_matches(state.v)
        || !shape_matches(state.w))
        throw std::invalid_argument(
            "terrain C-grid strain requires shape-matched velocity fields");

    const auto& u = *state.u;
    const auto& v = *state.v;
    const auto& w = *state.w;
    const bool wrap_x = periodic_x_topology(state);
    const bool wrap_y = periodic_y_topology(state);
    const int im = i_neighbor_prev(i, nx, wrap_x);
    const int jm = j_neighbor_prev(j, ny, wrap_y);

    const auto uf = [&](int ii, int jj, int kk)
    { return static_cast<double>(u[ii][jj][kk]); };
    const auto vf = [&](int ii, int jj, int kk)
    { return static_cast<double>(v[ii][jj][kk]); };
    const auto wf = [&](int ii, int jj, int kk)
    { return static_cast<double>(w[ii][jj][kk]); };
    const auto u_height = [&](int ii, int jj, int kk)
    {
        const int ii1 = i_neighbor_next(ii, nx, wrap_x);
        return 0.5 * (metrics.z(ii, jj, kk)
                    + metrics.z(ii1, jj, kk));
    };
    const auto v_height = [&](int ii, int jj, int kk)
    {
        const int jj1 = j_neighbor_next(jj, ny, wrap_y);
        return 0.5 * (metrics.z(ii, jj, kk)
                    + metrics.z(ii, jj1, kk));
    };
    const auto vertical_gradient =
        [&](const auto& value_at, const auto& height_at,
            int ii, int jj, int kk)
    {
        const int lower = kk == 0 ? 0 : kk - 1;
        const int upper = kk + 1 < nz ? kk + 1 : nz - 1;
        const double dz_physical =
            height_at(ii, jj, upper) - height_at(ii, jj, lower);
        if (!std::isfinite(dz_physical)
            || dz_physical <= terrain_constants::epsilon)
            throw std::invalid_argument(
                "terrain C-grid strain requires increasing component heights");
        return (value_at(ii, jj, upper) - value_at(ii, jj, lower))
            / dz_physical;
    };
    const auto gx_center = [&](int ii, int jj, int kk)
    { return metrics.J(ii, jj, kk) * metrics.mx(ii, jj, kk); };
    const auto gy_center = [&](int ii, int jj, int kk)
    { return metrics.J(ii, jj, kk) * metrics.my(ii, jj, kk); };
    const auto average4 = [](double a, double b, double c, double d)
    { return 0.25 * (a + b + c + d); };

    // Diagonal strains retain their natural C-grid face-to-center
    // differences. The metric correction removes the vertical change sampled
    // by a horizontal line of constant zeta.
    const double u_vertical = 0.5 * (
        vertical_gradient(uf, u_height, im, j, k)
      + vertical_gradient(uf, u_height, i, j, k));
    const double v_vertical = 0.5 * (
        vertical_gradient(vf, v_height, i, jm, k)
      + vertical_gradient(vf, v_height, i, j, k));
    const double s11 = (uf(i, j, k) - uf(im, j, k)) / metrics.dx
        + gx_center(i, j, k) * u_vertical;
    const double s22 = (vf(i, j, k) - vf(i, jm, k)) / metrics.dy
        + gy_center(i, j, k) * v_vertical;
    const double ground_w = k == 0
        ? terrain_transport::cgrid_ground_vertical_velocity(
              metrics, u, v, i, j,
              (state.earth_fixed_terrain ? 0.0 : state.frame_u_ms),
              (state.earth_fixed_terrain ? 0.0 : state.frame_v_ms))
        : wf(i, j, k - 1);
    const double layer_thickness =
        metrics.z_face(i, j, k + 1) - metrics.z_face(i, j, k);
    const double s33 = (wf(i, j, k) - ground_w) / layer_thickness;

    const auto s12_edge = [&](int ii, int jj)
    {
        const int ii1 = i_neighbor_next(ii, nx, wrap_x);
        const int jj1 = j_neighbor_next(jj, ny, wrap_y);
        const double gx = average4(
            gx_center(ii, jj, k), gx_center(ii1, jj, k),
            gx_center(ii, jj1, k), gx_center(ii1, jj1, k));
        const double gy = average4(
            gy_center(ii, jj, k), gy_center(ii1, jj, k),
            gy_center(ii, jj1, k), gy_center(ii1, jj1, k));
        const double du_dz = 0.5 * (
            vertical_gradient(uf, u_height, ii, jj, k)
          + vertical_gradient(uf, u_height, ii, jj1, k));
        const double dv_dz = 0.5 * (
            vertical_gradient(vf, v_height, ii, jj, k)
          + vertical_gradient(vf, v_height, ii1, jj, k));
        const double du_dy =
            (uf(ii, jj1, k) - uf(ii, jj, k)) / metrics.dy
            + gy * du_dz;
        const double dv_dx =
            (vf(ii1, jj, k) - vf(ii, jj, k)) / metrics.dx
            + gx * dv_dz;
        return 0.5 * (du_dy + dv_dx);
    };
    const auto w_vertical_at_face = [&](int ii, int jj, int face_owner)
    {
        const int lower = std::max(face_owner - 1, 0);
        const int upper = std::min(face_owner + 1, nz - 1);
        const double dz_physical =
            metrics.z_face(ii, jj, upper + 1)
          - metrics.z_face(ii, jj, lower + 1);
        return (wf(ii, jj, upper) - wf(ii, jj, lower)) / dz_physical;
    };
    const auto xz_edge_gradients = [&](int ii, int face_owner)
    {
        if (face_owner < 0 || face_owner >= nz - 1)
            return std::array<double, 2>{0.0, 0.0};
        const int ii1 = i_neighbor_next(ii, nx, wrap_x);
        const double lower_height = u_height(ii, j, face_owner);
        const double upper_height = u_height(ii, j, face_owner + 1);
        const double du_dz =
            (uf(ii, j, face_owner + 1) - uf(ii, j, face_owner))
            / (upper_height - lower_height);
        const int face = face_owner + 1;
        const double gx = 0.5 * (
            metrics.Gx_face(ii, j, face)
          + metrics.Gx_face(ii1, j, face));
        const double dw_dz = 0.5 * (
            w_vertical_at_face(ii, j, face_owner)
          + w_vertical_at_face(ii1, j, face_owner));
        const double dw_dx =
            (wf(ii1, j, face_owner) - wf(ii, j, face_owner))
                / metrics.dx
            + gx * dw_dz;
        return std::array<double, 2>{du_dz, dw_dx};
    };
    const auto yz_edge_gradients = [&](int jj, int face_owner)
    {
        if (face_owner < 0 || face_owner >= nz - 1)
            return std::array<double, 2>{0.0, 0.0};
        const int jj1 = j_neighbor_next(jj, ny, wrap_y);
        const double lower_height = v_height(i, jj, face_owner);
        const double upper_height = v_height(i, jj, face_owner + 1);
        const double dv_dz =
            (vf(i, jj, face_owner + 1) - vf(i, jj, face_owner))
            / (upper_height - lower_height);
        const int face = face_owner + 1;
        const double gy = 0.5 * (
            metrics.Gy_face(i, jj, face)
          + metrics.Gy_face(i, jj1, face));
        const double dw_dz = 0.5 * (
            w_vertical_at_face(i, jj, face_owner)
          + w_vertical_at_face(i, jj1, face_owner));
        const double dw_dy =
            (wf(i, jj1, face_owner) - wf(i, jj, face_owner))
                / metrics.dy
            + gy * dw_dz;
        return std::array<double, 2>{dv_dz, dw_dy};
    };

    const double s12a = s12_edge(im, jm);
    const double s12b = s12_edge(im, j);
    const double s12c = s12_edge(i, jm);
    const double s12d = s12_edge(i, j);
    const auto xza = xz_edge_gradients(im, k - 1);
    const auto xzb = xz_edge_gradients(im, k);
    const auto xzc = xz_edge_gradients(i, k - 1);
    const auto xzd = xz_edge_gradients(i, k);
    const auto yza = yz_edge_gradients(jm, k - 1);
    const auto yzb = yz_edge_gradients(jm, k);
    const auto yzc = yz_edge_gradients(j, k - 1);
    const auto yzd = yz_edge_gradients(j, k);
    const double s12_sq = average4(
        s12a * s12a, s12b * s12b, s12c * s12c, s12d * s12d);
    const auto average_gradient_sq = [&](const auto& a, const auto& b,
                                         const auto& c, const auto& d,
                                         std::size_t component)
    {
        return average4(
            a[component] * a[component], b[component] * b[component],
            c[component] * c[component], d[component] * d[component]);
    };
    double result = 0.0;
    if (mode == "horizontal_only")
    {
        result = 2.0 * (s11 * s11 + s22 * s22) + 4.0 * s12_sq
            + average_gradient_sq(xza, xzb, xzc, xzd, 1)
            + average_gradient_sq(yza, yzb, yzc, yzd, 1);
    }
    else if (mode == "vertical_only")
    {
        result = average_gradient_sq(xza, xzb, xzc, xzd, 0)
            + average_gradient_sq(yza, yzb, yzc, yzd, 0)
            + 2.0 * s33 * s33;
    }
    else
    {
        const auto symmetric_sq = [](const auto& gradient)
        {
            const double strain = 0.5 * (gradient[0] + gradient[1]);
            return strain * strain;
        };
        const double s13_sq = average4(
            symmetric_sq(xza), symmetric_sq(xzb),
            symmetric_sq(xzc), symmetric_sq(xzd));
        const double s23_sq = average4(
            symmetric_sq(yza), symmetric_sq(yzb),
            symmetric_sq(yzc), symmetric_sq(yzd));
        result = 2.0 * (s11 * s11 + s22 * s22 + s33 * s33)
            + 4.0 * (s12_sq + s13_sq + s23_sq);
    }
    if (!std::isfinite(result) || result < 0.0)
        throw std::runtime_error(
            "terrain C-grid strain produced an invalid norm");
    return result;
}

void terrain_cgrid_momentum_diffusion_tendencies(
    const TurbulenceConfig& cfg,
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    const Field3D& nu_t,
    Field3D& dudt_sgs,
    Field3D& dvdt_sgs,
    Field3D& dwdt_sgs,
    bool wrap_x,
    bool wrap_y,
    double* boundary_power_w)
{
    const TerrainMetrics3D& metrics = *grid.terrain_metrics;
    terrain_transport::validate_geometry(metrics);
    const int nx = state.NR;
    const int ny = state.NTH;
    const int nz = state.NZ;
    const bool use_h = cfg.mode != "vertical_only";
    const bool use_v = cfg.mode != "horizontal_only";
    const auto& u = *state.u;
    const auto& v = *state.v;
    const auto& w = *state.w;
    const auto& rho = *state.rho;

    // Complete the geometry preflight before mutating caller-owned outputs.
    // The strain stencil uses physical component heights, not nominal dzeta.
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
                if (metrics.z_face(i, j, k + 1)
                    - metrics.z_face(i, j, k)
                    <= terrain_constants::epsilon)
                    throw std::invalid_argument(
                        "terrain SGS requires strictly increasing physical faces");

    const auto uf = [&](int i, int j, int k)
    { return static_cast<double>(u[i][j][k]); };
    const auto vf = [&](int i, int j, int k)
    { return static_cast<double>(v[i][j][k]); };
    const auto wf = [&](int i, int j, int k)
    { return static_cast<double>(w[i][j][k]); };
    const auto rho_c = [&](int i, int j, int k)
    { return static_cast<double>(rho[i][j][k]); };
    const auto mu_c = [&](int i, int j, int k)
    {
        return rho_c(i, j, k)
             * static_cast<double>(nu_t[i][j][k]);
    };
    const auto gx_c = [&](int i, int j, int k)
    { return metrics.J(i, j, k) * metrics.mx(i, j, k); };
    const auto gy_c = [&](int i, int j, int k)
    { return metrics.J(i, j, k) * metrics.my(i, j, k); };
    const auto average4 = [](double a, double b, double c, double d)
    { return 0.25 * (a + b + c + d); };
    const auto u_height = [&](int i, int j, int k)
    {
        const int ip = i_neighbor_next(i, nx, wrap_x);
        return 0.5 * (metrics.z(i, j, k) + metrics.z(ip, j, k));
    };
    const auto v_height = [&](int i, int j, int k)
    {
        const int jp = j_neighbor_next(j, ny, wrap_y);
        return 0.5 * (metrics.z(i, j, k) + metrics.z(i, jp, k));
    };
    const auto vertical_gradient =
        [&](const auto& value_at, const auto& height_at,
            int i, int j, int k)
    {
        const int lower = k == 0 ? 0 : k - 1;
        const int upper = k + 1 < nz ? k + 1 : nz - 1;
        return (value_at(i, j, upper) - value_at(i, j, lower))
             / (height_at(i, j, upper) - height_at(i, j, lower));
    };
    const auto w_vertical_at_face = [&](int i, int j, int face_owner)
    {
        const int lower = std::max(face_owner - 1, 0);
        const int upper = std::min(face_owner + 1, nz - 1);
        return (wf(i, j, upper) - wf(i, j, lower))
             / (metrics.z_face(i, j, upper + 1)
              - metrics.z_face(i, j, lower + 1));
    };
    const auto s11 = [&](int i, int j, int k)
    {
        const int im = i_neighbor_prev(i, nx, wrap_x);
        const double du_dz = 0.5 * (
            vertical_gradient(uf, u_height, im, j, k)
          + vertical_gradient(uf, u_height, i, j, k));
        return (uf(i, j, k) - uf(im, j, k)) / metrics.dx
             + gx_c(i, j, k) * du_dz;
    };
    const auto s22 = [&](int i, int j, int k)
    {
        const int jm = j_neighbor_prev(j, ny, wrap_y);
        const double dv_dz = 0.5 * (
            vertical_gradient(vf, v_height, i, jm, k)
          + vertical_gradient(vf, v_height, i, j, k));
        return (vf(i, j, k) - vf(i, jm, k)) / metrics.dy
             + gy_c(i, j, k) * dv_dz;
    };
    const auto s33 = [&](int i, int j, int k)
    {
        const double lower_w = k == 0
            ? terrain_transport::cgrid_ground_vertical_velocity(
                  metrics, u, v, i, j,
                  (state.earth_fixed_terrain ? 0.0 : state.frame_u_ms),
                  (state.earth_fixed_terrain ? 0.0 : state.frame_v_ms))
            : wf(i, j, k - 1);
        return (wf(i, j, k) - lower_w)
             / (metrics.z_face(i, j, k + 1)
              - metrics.z_face(i, j, k));
    };
    const auto s12 = [&](int i, int j, int k)
    {
        const int ip = i_neighbor_next(i, nx, wrap_x);
        const int jp = j_neighbor_next(j, ny, wrap_y);
        const double gx = average4(
            gx_c(i, j, k), gx_c(ip, j, k),
            gx_c(i, jp, k), gx_c(ip, jp, k));
        const double gy = average4(
            gy_c(i, j, k), gy_c(ip, j, k),
            gy_c(i, jp, k), gy_c(ip, jp, k));
        const double du_dz = 0.5 * (
            vertical_gradient(uf, u_height, i, j, k)
          + vertical_gradient(uf, u_height, i, jp, k));
        const double dv_dz = 0.5 * (
            vertical_gradient(vf, v_height, i, j, k)
          + vertical_gradient(vf, v_height, ip, j, k));
        const double du_dy =
            (uf(i, jp, k) - uf(i, j, k)) / metrics.dy + gy * du_dz;
        const double dv_dx =
            (vf(ip, j, k) - vf(i, j, k)) / metrics.dx + gx * dv_dz;
        return 0.5 * (du_dy + dv_dx);
    };
    const auto xz_gradients = [&](int i, int j, int edge_k)
    {
        if (edge_k < 0 || edge_k >= nz - 1)
            return std::array<double, 2>{0.0, 0.0};
        const int ip = i_neighbor_next(i, nx, wrap_x);
        const double du_dz =
            (uf(i, j, edge_k + 1) - uf(i, j, edge_k))
          / (u_height(i, j, edge_k + 1) - u_height(i, j, edge_k));
        const int face = edge_k + 1;
        const double gx = 0.5 * (
            metrics.Gx_face(i, j, face)
          + metrics.Gx_face(ip, j, face));
        const double dw_dz = 0.5 * (
            w_vertical_at_face(i, j, edge_k)
          + w_vertical_at_face(ip, j, edge_k));
        const double dw_dx =
            (wf(ip, j, edge_k) - wf(i, j, edge_k)) / metrics.dx
          + gx * dw_dz;
        return std::array<double, 2>{du_dz, dw_dx};
    };
    const auto yz_gradients = [&](int i, int j, int edge_k)
    {
        if (edge_k < 0 || edge_k >= nz - 1)
            return std::array<double, 2>{0.0, 0.0};
        const int jp = j_neighbor_next(j, ny, wrap_y);
        const double dv_dz =
            (vf(i, j, edge_k + 1) - vf(i, j, edge_k))
          / (v_height(i, j, edge_k + 1) - v_height(i, j, edge_k));
        const int face = edge_k + 1;
        const double gy = 0.5 * (
            metrics.Gy_face(i, j, face)
          + metrics.Gy_face(i, jp, face));
        const double dw_dz = 0.5 * (
            w_vertical_at_face(i, j, edge_k)
          + w_vertical_at_face(i, jp, edge_k));
        const double dw_dy =
            (wf(i, jp, edge_k) - wf(i, j, edge_k)) / metrics.dy
          + gy * dw_dz;
        return std::array<double, 2>{dv_dz, dw_dy};
    };
    const auto mu_xy = [&](int i, int j, int k)
    {
        const int ip = i_neighbor_next(i, nx, wrap_x);
        const int jp = j_neighbor_next(j, ny, wrap_y);
        return average4(mu_c(i, j, k), mu_c(ip, j, k),
                        mu_c(i, jp, k), mu_c(ip, jp, k));
    };
    const auto mu_xz = [&](int i, int j, int edge_k)
    {
        const int ip = i_neighbor_next(i, nx, wrap_x);
        const double lower = 0.5 * (
            mu_c(i, j, edge_k) + mu_c(ip, j, edge_k));
        const double upper = 0.5 * (
            mu_c(i, j, edge_k + 1) + mu_c(ip, j, edge_k + 1));
        const double lower_dz = 0.5 * (
            metrics.z_face(i, j, edge_k + 1)
          - metrics.z_face(i, j, edge_k)
          + metrics.z_face(ip, j, edge_k + 1)
          - metrics.z_face(ip, j, edge_k));
        const double upper_dz = 0.5 * (
            metrics.z_face(i, j, edge_k + 2)
          - metrics.z_face(i, j, edge_k + 1)
          + metrics.z_face(ip, j, edge_k + 2)
          - metrics.z_face(ip, j, edge_k + 1));
        return (upper_dz * lower + lower_dz * upper)
             / (lower_dz + upper_dz);
    };
    const auto mu_yz = [&](int i, int j, int edge_k)
    {
        const int jp = j_neighbor_next(j, ny, wrap_y);
        const double lower = 0.5 * (
            mu_c(i, j, edge_k) + mu_c(i, jp, edge_k));
        const double upper = 0.5 * (
            mu_c(i, j, edge_k + 1) + mu_c(i, jp, edge_k + 1));
        const double lower_dz = 0.5 * (
            metrics.z_face(i, j, edge_k + 1)
          - metrics.z_face(i, j, edge_k)
          + metrics.z_face(i, jp, edge_k + 1)
          - metrics.z_face(i, jp, edge_k));
        const double upper_dz = 0.5 * (
            metrics.z_face(i, j, edge_k + 2)
          - metrics.z_face(i, j, edge_k + 1)
          + metrics.z_face(i, jp, edge_k + 2)
          - metrics.z_face(i, jp, edge_k + 1));
        return (upper_dz * lower + lower_dz * upper)
             / (lower_dz + upper_dz);
    };
    const auto t11 = [&](int i, int j, int k)
    { return 2.0 * mu_c(i, j, k) * s11(i, j, k); };
    const auto t22 = [&](int i, int j, int k)
    { return 2.0 * mu_c(i, j, k) * s22(i, j, k); };
    const auto t33 = [&](int i, int j, int k)
    { return 2.0 * mu_c(i, j, k) * s33(i, j, k); };
    const auto t12 = [&](int i, int j, int k)
    { return 2.0 * mu_xy(i, j, k) * s12(i, j, k); };
    const auto t13 = [&](int i, int j, int edge_k)
    {
        if (edge_k < 0 || edge_k >= nz - 1) return 0.0;
        const auto gradient = xz_gradients(i, j, edge_k);
        return mu_xz(i, j, edge_k)
            * (gradient[0] + (use_h ? gradient[1] : 0.0));
    };
    const auto t23 = [&](int i, int j, int edge_k)
    {
        if (edge_k < 0 || edge_k >= nz - 1) return 0.0;
        const auto gradient = yz_gradients(i, j, edge_k);
        return mu_yz(i, j, edge_k)
            * (gradient[0] + (use_h ? gradient[1] : 0.0));
    };
    const auto t31 = [&](int i, int j, int edge_k)
    {
        if (edge_k < 0 || edge_k >= nz - 1) return 0.0;
        const auto gradient = xz_gradients(i, j, edge_k);
        return mu_xz(i, j, edge_k)
            * (gradient[1] + (use_v ? gradient[0] : 0.0));
    };
    const auto t32 = [&](int i, int j, int edge_k)
    {
        if (edge_k < 0 || edge_k >= nz - 1) return 0.0;
        const auto gradient = yz_gradients(i, j, edge_k);
        return mu_yz(i, j, edge_k)
            * (gradient[1] + (use_v ? gradient[0] : 0.0));
    };

    // Interpolate natural stresses only in dimensions where their C-grid
    // locations differ from the requested mapped face.
    const auto t11_xz = [&](int i, int j, int edge_k)
    {
        const int ip = i_neighbor_next(i, nx, wrap_x);
        return average4(t11(i, j, edge_k), t11(ip, j, edge_k),
                        t11(i, j, edge_k + 1), t11(ip, j, edge_k + 1));
    };
    const auto t22_yz = [&](int i, int j, int edge_k)
    {
        const int jp = j_neighbor_next(j, ny, wrap_y);
        return average4(t22(i, j, edge_k), t22(i, jp, edge_k),
                        t22(i, j, edge_k + 1), t22(i, jp, edge_k + 1));
    };
    const auto t12_xz = [&](int i, int j, int edge_k)
    {
        const int jm = j_neighbor_prev(j, ny, wrap_y);
        return average4(t12(i, jm, edge_k), t12(i, j, edge_k),
                        t12(i, jm, edge_k + 1), t12(i, j, edge_k + 1));
    };
    const auto t12_yz = [&](int i, int j, int edge_k)
    {
        const int im = i_neighbor_prev(i, nx, wrap_x);
        return average4(t12(im, j, edge_k), t12(i, j, edge_k),
                        t12(im, j, edge_k + 1), t12(i, j, edge_k + 1));
    };
    const auto t31_center = [&](int i, int j, int k)
    {
        const int im = i_neighbor_prev(i, nx, wrap_x);
        return average4(t31(im, j, k - 1), t31(i, j, k - 1),
                        t31(im, j, k), t31(i, j, k));
    };
    const auto t32_center = [&](int i, int j, int k)
    {
        const int jm = j_neighbor_prev(j, ny, wrap_y);
        return average4(t32(i, jm, k - 1), t32(i, j, k - 1),
                        t32(i, jm, k), t32(i, j, k));
    };
    const auto seam_jx = [&](int face, int j, int k)
    {
        if (wrap_x && (face == 1 || face == nx - 1))
            return 0.5 * (
                metrics.J(nx - 2, j, k) + metrics.J(1, j, k));
        return metrics.Jx_face(face, j, k);
    };
    const auto seam_jy = [&](int i, int face, int k)
    {
        if (wrap_y && (face == 1 || face == ny - 1))
            return 0.5 * (
                metrics.J(i, ny - 2, k) + metrics.J(i, 1, k));
        return metrics.Jy_face(i, face, k);
    };

    dudt_sgs.fill(0.0f);
    dvdt_sgs.fill(0.0f);
    dwdt_sgs.fill(0.0f);
    const double inv_dx = 1.0 / metrics.dx;
    const double inv_dy = 1.0 / metrics.dy;
    const double inv_dzeta = 1.0 / metrics.dzeta;

    #pragma omp parallel if(static_cast<std::size_t>(nx) * ny * nz >= 32768)
    {
    #pragma omp for collapse(2) nowait
    for (int i = 1; i <= (wrap_x ? nx - 2 : nx - 3); ++i)
        for (int j = 1; j <= ny - 2; ++j)
        {
            const int ip = i_neighbor_next(i, nx, wrap_x);
            const int jm = j_neighbor_prev(j, ny, wrap_y);
            for (int k = 0; k < nz - 1; ++k)
            {
                const auto x_flux = [&](int center)
                {
                    return use_h
                        ? metrics.J(center, j, k) * t11(center, j, k)
                        : 0.0;
                };
                const auto y_flux = [&](int edge_j)
                {
                    if (!use_h) return 0.0;
                    const int face = edge_j + 1;
                    const double jacobian = 0.5 * (
                        seam_jy(i, face, k) + seam_jy(ip, face, k));
                    return jacobian * t12(i, edge_j, k);
                };
                const auto zeta_flux = [&](int edge_k)
                {
                    if (edge_k < 0 || edge_k >= nz - 1) return 0.0;
                    const int face = edge_k + 1;
                    const double gx = 0.5 * (
                        metrics.Gx_face(i, j, face)
                      + metrics.Gx_face(ip, j, face));
                    const double gy = 0.5 * (
                        metrics.Gy_face(i, j, face)
                      + metrics.Gy_face(ip, j, face));
                    return (use_v ? t13(i, j, edge_k) : 0.0)
                         + gx * (use_h ? t11_xz(i, j, edge_k) : 0.0)
                         + gy * (use_h ? t12_xz(i, j, edge_k) : 0.0);
                };
                const double divergence =
                    (x_flux(ip) - x_flux(i)) * inv_dx
                  + (y_flux(j) - y_flux(jm)) * inv_dy
                  + (zeta_flux(k) - zeta_flux(k - 1)) * inv_dzeta;
                const double mapped_mass = 0.5 * (
                    rho_c(i, j, k) * metrics.J(i, j, k)
                  + rho_c(ip, j, k) * metrics.J(ip, j, k));
                dudt_sgs[i][j][k] =
                    static_cast<float>(divergence / mapped_mass);
            }
        }

    #pragma omp for collapse(2) nowait
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j <= ny - 2; ++j)
        {
            if (!wrap_y && j >= ny - 2) continue;
            const int im = i_neighbor_prev(i, nx, wrap_x);
            const int jp = j_neighbor_next(j, ny, wrap_y);
            for (int k = 0; k < nz - 1; ++k)
            {
                const auto x_flux = [&](int edge_i)
                {
                    if (!use_h) return 0.0;
                    const int face = edge_i + 1;
                    const double jacobian = 0.5 * (
                        seam_jx(face, j, k) + seam_jx(face, jp, k));
                    return jacobian * t12(edge_i, j, k);
                };
                const auto y_flux = [&](int center)
                {
                    return use_h
                        ? metrics.J(i, center, k) * t22(i, center, k)
                        : 0.0;
                };
                const auto zeta_flux = [&](int edge_k)
                {
                    if (edge_k < 0 || edge_k >= nz - 1) return 0.0;
                    const int face = edge_k + 1;
                    const double gx = 0.5 * (
                        metrics.Gx_face(i, j, face)
                      + metrics.Gx_face(i, jp, face));
                    const double gy = 0.5 * (
                        metrics.Gy_face(i, j, face)
                      + metrics.Gy_face(i, jp, face));
                    return (use_v ? t23(i, j, edge_k) : 0.0)
                         + gx * (use_h ? t12_yz(i, j, edge_k) : 0.0)
                         + gy * (use_h ? t22_yz(i, j, edge_k) : 0.0);
                };
                const double divergence =
                    (x_flux(i) - x_flux(im)) * inv_dx
                  + (y_flux(jp) - y_flux(j)) * inv_dy
                  + (zeta_flux(k) - zeta_flux(k - 1)) * inv_dzeta;
                const double mapped_mass = 0.5 * (
                    rho_c(i, j, k) * metrics.J(i, j, k)
                  + rho_c(i, jp, k) * metrics.J(i, jp, k));
                dvdt_sgs[i][j][k] =
                    static_cast<float>(divergence / mapped_mass);
            }
        }

    #pragma omp for collapse(2) nowait
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz - 1; ++k)
            {
                const int im = i_neighbor_prev(i, nx, wrap_x);
                const int jm = j_neighbor_prev(j, ny, wrap_y);
                const auto x_flux = [&](int edge_i)
                {
                    if (!use_h) return 0.0;
                    const int face = edge_i + 1;
                    const double jacobian = 0.5 * (
                        seam_jx(face, j, k)
                      + seam_jx(face, j, k + 1));
                    return jacobian * t31(edge_i, j, k);
                };
                const auto y_flux = [&](int edge_j)
                {
                    if (!use_h) return 0.0;
                    const int face = edge_j + 1;
                    const double jacobian = 0.5 * (
                        seam_jy(i, face, k)
                      + seam_jy(i, face, k + 1));
                    return jacobian * t32(i, edge_j, k);
                };
                const auto zeta_flux = [&](int center_k)
                {
                    return (use_v ? t33(i, j, center_k) : 0.0)
                         + gx_c(i, j, center_k)
                            * (use_h ? t31_center(i, j, center_k) : 0.0)
                         + gy_c(i, j, center_k)
                            * (use_h ? t32_center(i, j, center_k) : 0.0);
                };
                const double divergence =
                    (x_flux(i) - x_flux(im)) * inv_dx
                  + (y_flux(j) - y_flux(jm)) * inv_dy
                  + (zeta_flux(k + 1) - zeta_flux(k)) * inv_dzeta;
                const double mapped_mass = 0.5 * (
                    rho_c(i, j, k) * metrics.J(i, j, k)
                  + rho_c(i, j, k + 1) * metrics.J(i, j, k + 1));
                dwdt_sgs[i][j][k] =
                    static_cast<float>(divergence / mapped_mass);
            }
    }

    if (boundary_power_w != nullptr)
    {
        const bool open_west = state.open_lateral[0];
        const bool open_east = state.open_lateral[1];
        const bool open_south = state.open_lateral[2];
        const bool open_north = state.open_lateral[3];
        const bool open_top =
            state.open_top;
        double boundary_power = 0.0;

        // Use the closure-plane velocity paired with each exterior mapped
        // traction. The adjacent closure-to-interior velocity jump remains in
        // the irreversible stress contraction, matching the flat C-grid
        // summation-by-parts convention used by the runtime guard.
        if (use_h && (open_west || open_east))
        {
            for (int j = 1; j < ny - 1; ++j)
                for (int k = 0; k < nz - 1; ++k)
                {
                    const double west_u = metrics.J(1, j, k)
                        * t11(1, j, k);
                    const double east_u = metrics.J(nx - 2, j, k)
                        * t11(nx - 2, j, k);
                    if (open_west)
                        boundary_power -= metrics.dy * metrics.dzeta
                            * static_cast<double>(u(0, j, k)) * west_u;
                    if (open_east)
                        boundary_power += metrics.dy * metrics.dzeta
                            * static_cast<double>(u(nx - 2, j, k)) * east_u;
                }

            for (int j = 1; j < ny - 2; ++j)
                for (int k = 0; k < nz - 1; ++k)
                {
                    const int jp = j_neighbor_next(j, ny, wrap_y);
                    const auto x_flux = [&](int edge_i)
                    {
                        const int face = edge_i + 1;
                        const double jacobian = 0.5 * (
                            seam_jx(face, j, k)
                          + seam_jx(face, jp, k));
                        return jacobian * t12(edge_i, j, k);
                    };
                    if (open_west)
                        boundary_power -= metrics.dy * metrics.dzeta
                            * static_cast<double>(v(0, j, k)) * x_flux(0);
                    if (open_east)
                        boundary_power += metrics.dy * metrics.dzeta
                            * static_cast<double>(v(nx - 1, j, k))
                            * x_flux(nx - 2);
                }

            for (int j = 1; j < ny - 1; ++j)
                for (int k = 0; k < nz - 1; ++k)
                {
                    const auto x_flux = [&](int edge_i)
                    {
                        const int face = edge_i + 1;
                        const double jacobian = 0.5 * (
                            seam_jx(face, j, k)
                          + seam_jx(face, j, k + 1));
                        return jacobian * t31(edge_i, j, k);
                    };
                    if (open_west)
                        boundary_power -= metrics.dy * metrics.dzeta
                            * static_cast<double>(w(0, j, k)) * x_flux(0);
                    if (open_east)
                        boundary_power += metrics.dy * metrics.dzeta
                            * static_cast<double>(w(nx - 1, j, k))
                            * x_flux(nx - 2);
                }
        }

        if (use_h && (open_south || open_north))
        {
            for (int i = 1; i < nx - 2; ++i)
                for (int k = 0; k < nz - 1; ++k)
                {
                    const int ip = i_neighbor_next(i, nx, wrap_x);
                    const auto y_flux = [&](int edge_j)
                    {
                        const int face = edge_j + 1;
                        const double jacobian = 0.5 * (
                            seam_jy(i, face, k)
                          + seam_jy(ip, face, k));
                        return jacobian * t12(i, edge_j, k);
                    };
                    if (open_south)
                        boundary_power -= metrics.dx * metrics.dzeta
                            * static_cast<double>(u(i, 0, k)) * y_flux(0);
                    if (open_north)
                        boundary_power += metrics.dx * metrics.dzeta
                            * static_cast<double>(u(i, ny - 1, k))
                            * y_flux(ny - 2);
                }

            for (int i = 1; i < nx - 1; ++i)
                for (int k = 0; k < nz - 1; ++k)
                {
                    const double south_v = metrics.J(i, 1, k)
                        * t22(i, 1, k);
                    const double north_v = metrics.J(i, ny - 2, k)
                        * t22(i, ny - 2, k);
                    if (open_south)
                        boundary_power -= metrics.dx * metrics.dzeta
                            * static_cast<double>(v(i, 0, k)) * south_v;
                    if (open_north)
                        boundary_power += metrics.dx * metrics.dzeta
                            * static_cast<double>(v(i, ny - 2, k)) * north_v;
                }

            for (int i = 1; i < nx - 1; ++i)
                for (int k = 0; k < nz - 1; ++k)
                {
                    const auto y_flux = [&](int edge_j)
                    {
                        const int face = edge_j + 1;
                        const double jacobian = 0.5 * (
                            seam_jy(i, face, k)
                          + seam_jy(i, face, k + 1));
                        return jacobian * t32(i, edge_j, k);
                    };
                    if (open_south)
                        boundary_power -= metrics.dx * metrics.dzeta
                            * static_cast<double>(w(i, 0, k)) * y_flux(0);
                    if (open_north)
                        boundary_power += metrics.dx * metrics.dzeta
                            * static_cast<double>(w(i, ny - 1, k))
                            * y_flux(ny - 2);
                }
        }
        if (open_top)
        {
            const int edge_k = nz - 2;
            const int face = edge_k + 1;
            for (int i = 1; i <= (wrap_x ? nx - 2 : nx - 3); ++i)
                for (int j = 1; j < ny - 1; ++j)
                {
                    const int ip = i_neighbor_next(i, nx, wrap_x);
                    const double gx = 0.5 * (
                        metrics.Gx_face(i, j, face)
                      + metrics.Gx_face(ip, j, face));
                    const double gy = 0.5 * (
                        metrics.Gy_face(i, j, face)
                      + metrics.Gy_face(ip, j, face));
                    const double traction =
                        (use_v ? t13(i, j, edge_k) : 0.0)
                      + gx * (use_h ? t11_xz(i, j, edge_k) : 0.0)
                      + gy * (use_h ? t12_xz(i, j, edge_k) : 0.0);
                    boundary_power += metrics.dx * metrics.dy
                        * static_cast<double>(u(i, j, nz - 1)) * traction;
                }
            for (int i = 1; i < nx - 1; ++i)
                for (int j = 1; j <= (wrap_y ? ny - 2 : ny - 3); ++j)
                {
                    const int jp = j_neighbor_next(j, ny, wrap_y);
                    const double gx = 0.5 * (
                        metrics.Gx_face(i, j, face)
                      + metrics.Gx_face(i, jp, face));
                    const double gy = 0.5 * (
                        metrics.Gy_face(i, j, face)
                      + metrics.Gy_face(i, jp, face));
                    const double traction =
                        (use_v ? t23(i, j, edge_k) : 0.0)
                      + gx * (use_h ? t12_yz(i, j, edge_k) : 0.0)
                      + gy * (use_h ? t22_yz(i, j, edge_k) : 0.0);
                    boundary_power += metrics.dx * metrics.dy
                        * static_cast<double>(v(i, j, nz - 1)) * traction;
                }
            for (int i = 1; i < nx - 1; ++i)
                for (int j = 1; j < ny - 1; ++j)
                {
                    const double traction =
                        (use_v ? t33(i, j, nz - 1) : 0.0)
                      + gx_c(i, j, nz - 1)
                            * (use_h ? t31_center(i, j, nz - 1) : 0.0)
                      + gy_c(i, j, nz - 1)
                            * (use_h ? t32_center(i, j, nz - 1) : 0.0);
                    boundary_power += metrics.dx * metrics.dy
                        * static_cast<double>(w(i, j, nz - 1)) * traction;
                }
        }
        if (!std::isfinite(boundary_power))
            throw std::runtime_error(
                "terrain SGS open-boundary power is non-finite");
        *boundary_power_w = boundary_power;
    }
}
} // anonymous namespace

// j-topology follows the ACTUAL y boundary (SCIENCE.md M16): the
// Cartesian y axis is periodic only when a y-face is configured Periodic.
// Before this, the diffusion Laplacian wrapped j unconditionally, mixing
// the north and south edges of every open-y domain. Strain, diffusion,
// and the GPU smagorinsky shaders (push constant) all share this rule.
bool periodic_y_topology()
{
    return global_lateral_bc_south
               == tmv::init::config::LateralBC::Periodic
        || global_lateral_bc_north
               == tmv::init::config::LateralBC::Periodic;
}

bool periodic_x_topology()
{
    return global_lateral_bc_west
               == tmv::init::config::LateralBC::Periodic
        || global_lateral_bc_east
               == tmv::init::config::LateralBC::Periodic;
}

void validate_scheme_inputs(const TurbulenceConfig& cfg,
                            const GridMetrics& grid,
                            const TurbulenceStateView& state,
                            bool require_tke)
{
    const int minimum_extent = state.c_grid ? 3 : 5;
    if (state.NR < minimum_extent || state.NTH < minimum_extent
        || state.NZ < (state.c_grid ? 2 : 5))
    {
        throw std::invalid_argument(
            "active turbulence schemes require at least a 3x3x2 grid");
    }
    if (cfg.mode != "3d" && cfg.mode != "horizontal_only" &&
        cfg.mode != "vertical_only")
    {
        throw std::invalid_argument("unsupported turbulence mode '" + cfg.mode + "'");
    }
    if (cfg.filter_width != "dx" && cfg.filter_width != "cubic_root" &&
        cfg.filter_width != "user")
    {
        throw std::invalid_argument(
            "unsupported turbulence filter width '" + cfg.filter_width + "'");
    }
    if (!known_turbulence_stability_model(cfg.stability_correction) ||
        !std::isfinite(cfg.Cs) || cfg.Cs < 0.0 ||
        !std::isfinite(cfg.Pr_t) || cfg.Pr_t <= 0.0 ||
        !std::isfinite(cfg.Sc_t) || cfg.Sc_t <= 0.0 ||
        !std::isfinite(cfg.nu_t_max) || cfg.nu_t_max < 0.0 ||
        !std::isfinite(cfg.K_max) || cfg.K_max < 0.0)
    {
        throw std::invalid_argument(
            "turbulence configuration violates the implemented coefficient contract");
    }
    if (!std::isfinite(cfg.user_filter_width_m) ||
        (cfg.filter_width == "user" ? cfg.user_filter_width_m <= 0.0
                                    : cfg.user_filter_width_m != 0.0))
    {
        throw std::invalid_argument(
            "turbulence user filter width violates the implemented metric contract");
    }
    if (require_tke && cfg.dynamic_Cs)
    {
        throw std::invalid_argument(
            "the TKE closure cannot use a dynamic Smagorinsky coefficient");
    }
    if (require_tke && (cfg.stability_correction == "ri" || cfg.stability_correction == kLillyRichardsonModel))
    {
        throw std::invalid_argument(
            "the TKE closure accepts only none or tke_based stability correction");
    }
    if (!require_tke && cfg.stability_correction == "tke_based")
    {
        throw std::invalid_argument(
            "tke_based stability correction requires the prognostic TKE closure");
    }
    if (!require_tke && cfg.dynamic_Cs &&
        cfg.stability_correction != "none")
    {
        throw std::invalid_argument(
            "dynamic Smagorinsky cannot use a separate stability correction");
    }
    if (cfg.stability_correction == kLillyRichardsonModel && cfg.mode != "3d")
        throw std::invalid_argument("Lilly equilibrium mixing requires the full 3-D strain tensor");

    grid_metric::sanitize_positive(grid.dx);
    grid_metric::sanitize_positive(grid.dy);
    grid_metric::validate_vertical_metrics(grid, state.NR, state.NTH, state.NZ);
    require_field(state.u, state, "u", false, false);
    require_field(state.v, state, "v", false, false);
    require_field(state.w, state, "w", false, false);
    require_field(state.theta, state, "theta", true, false);
    require_field(state.rho, state, "rho", true, false);
    if (state.qv != nullptr)
    {
        require_field(state.qv, state, "qv", false, true);
    }
    if (cfg.enable_moist_buoyancy && state.qv == nullptr)
    {
        throw std::invalid_argument(
            "moist turbulence buoyancy requires a water-vapor field");
    }
    if (require_tke)
    {
        require_field(state.tke, state, "tke", false, true);
    }

    require_rho0(state);
}

/**
 * @brief Computes the strain rate.
 */
StrainRate compute_strain_rate_3d(
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    int i, int j, int k
)
{
    require_flat_tensor_geometry(grid, "turbulence strain tensor");
    StrainRate S;

    const double dx = grid_metric::local_dx(grid, i, j, k);
    const double dy = grid_metric::local_dy(grid, i, j, k);
    const double dz_one_sided = k == state.NZ - 1
        ? grid_metric::vertical_center_distance(
            grid, i, j, k - 1, state.NZ)
        : grid_metric::vertical_center_distance(
            grid, i, j, k, state.NZ);
    const double dz_centered =
        grid_metric::centered_dz_span(grid, i, j, k, state.NZ);

    // Horizontal derivatives (radial direction, index i)
    const auto& u = *state.u;
    const auto& v = *state.v;
    const auto& w = *state.w;

    const bool wrap_x = periodic_x_topology(state);
    const int im = i_neighbor_prev(i, state.NR, wrap_x);
    const int ip = i_neighbor_next(i, state.NR, wrap_x);
    const int i_eff = wrap_x ? 1 : i;
    const int nr_eff = wrap_x ? 3 : state.NR;
    const double dudx = horiz_diff(
        u[im][j][k], u[i][j][k], u[ip][j][k], dx, i_eff, nr_eff);
    const double dvdx = horiz_diff(
        v[im][j][k], v[i][j][k], v[ip][j][k], dx, i_eff, nr_eff);
    const double dwdx = horiz_diff(
        w[im][j][k], w[i][j][k], w[ip][j][k], dx, i_eff, nr_eff);

    // Horizontal derivatives (azimuthal direction, index j). Neighbors
    // wrap on periodic-y topology (M16); horiz_diff still treats wrapped
    // points as a full centered stencil because j_eff masks the edge.
    const bool wrap_y = periodic_y_topology(state);
    const int jm = j_neighbor_prev(j, state.NTH, wrap_y);
    const int jp = j_neighbor_next(j, state.NTH, wrap_y);
    const int j_eff = wrap_y ? 1 : j;  // interior index when periodic
    const int nth_eff = wrap_y ? 3 : state.NTH;
    const double dudy = horiz_diff(u[i][jm][k], u[i][j][k], u[i][jp][k], dy, j_eff, nth_eff);
    const double dvdy = horiz_diff(v[i][jm][k], v[i][j][k], v[i][jp][k], dy, j_eff, nth_eff);
    const double dwdy = horiz_diff(w[i][jm][k], w[i][j][k], w[i][jp][k], dy, j_eff, nth_eff);

    // Vertical derivatives (index k)
    const double dudz = vert_diff(u[i][j][std::max(k-1,0)], u[i][j][k], u[i][j][std::min(k+1,state.NZ-1)], dz_one_sided, dz_centered, k, state.NZ);
    const double dvdz = vert_diff(v[i][j][std::max(k-1,0)], v[i][j][k], v[i][j][std::min(k+1,state.NZ-1)], dz_one_sided, dz_centered, k, state.NZ);
    const double dwdz = vert_diff(w[i][j][std::max(k-1,0)], w[i][j][k], w[i][j][std::min(k+1,state.NZ-1)], dz_one_sided, dz_centered, k, state.NZ);

    S.S11 = dudx;
    S.S12 = 0.5 * (dudy + dvdx);
    S.S13 = 0.5 * (dudz + dwdx);

    S.S21 = S.S12;
    S.S22 = dvdy;
    S.S23 = 0.5 * (dvdz + dwdy);

    S.S31 = S.S13;
    S.S32 = S.S23;
    S.S33 = dwdz;

    S.magnitude = std::sqrt(2.0 * (
        S.S11*S.S11 + S.S22*S.S22 + S.S33*S.S33 +
        2.0*(S.S12*S.S12 + S.S13*S.S13 + S.S23*S.S23)
    ));

    return S;
}

double compute_directional_strain_sq_centered(
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    int i, int j, int k,
    const std::string& mode
)
{
    if (mode != "3d" && mode != "horizontal_only"
        && mode != "vertical_only")
        throw std::invalid_argument(
            "unsupported turbulence mode '" + mode + "'");
    if (!state.c_grid)
    {
        return collocated_eddy_viscosity::directional_strain_sq(
            state, grid, i, j, k, mode,
            periodic_x_topology(state), periodic_y_topology(state));
    }
    if (grid.terrain_metrics_active)
    {
        return terrain_cgrid_strain_sq_centered(
            state, grid, i, j, k, mode);
    }
    const auto& u = *state.u;
    const auto& v = *state.v;
    const auto& w = *state.w;
    const int NX = state.NR, NY = state.NTH, NZ_l = state.NZ;

    const double dx = grid_metric::local_dx(grid, i, j, k);
    const double dy = grid_metric::local_dy(grid, i, j, k);
    const double dz_k = grid_metric::vertical_cell_thickness(
        grid, i, j, k, NZ_l);

    const bool wrap_x = periodic_x_topology(state);
    const bool wrap_y = periodic_y_topology(state);
    const int jm = j_neighbor_prev(j, NY, wrap_y);
    const int im = i_neighbor_prev(i, NX, wrap_x);

    // C-grid placements: u[i] is the x-face between centers i and i+1,
    // v[j] the y-face between j and j+1, w[k] the z-face between k and
    // k+1 (w[-1] = 0 is the implicit ground face).
    auto uf = [&](int ii, int jj, int kk) -> double
    { return static_cast<double>(u[ii][jj][kk]); };
    auto vf = [&](int ii, int jj, int kk) -> double
    { return static_cast<double>(v[ii][jj][kk]); };
    auto wf = [&](int ii, int jj, int kk) -> double
    { return (kk < 0) ? 0.0 : static_cast<double>(w[ii][jj][kk]); };

    // Diagonal components: natural face-to-center differences.
    const double S11 = (uf(i, j, k) - uf(im, j, k)) / dx;
    const double S22 = (vf(i, j, k) - vf(i, jm, k)) / dy;
    const double S33 = (wf(i, j, k) - wf(i, j, k - 1)) / dz_k;

    // Off-diagonal components at the four surrounding edges of each
    // pair; SQUARES averaged to the center. S12 lives on xy-edges
    // (x-face i' crossed with y-face j'):
    auto S12_edge = [&](int ii, int jj) -> double
    {
        // edge at (x-face ii, y-face jj): du/dy across y-face jj at
        // x-face ii, dv/dx across x-face ii at y-face jj.
        const int jj1 = j_neighbor_next(jj, NY, wrap_y);
        const int ii1 = i_neighbor_next(ii, NX, wrap_x);
        const double dudy = (uf(ii, jj1, k) - uf(ii, jj, k)) / dy;
        const double dvdx = (vf(ii1, jj, k) - vf(ii, jj, k)) / dx;
        return 0.5 * (dudy + dvdx);
    };
    auto xz_edge_gradients = [&](int ii, int kk)
    {
        // edge at (x-face ii, z-face kk); kk = -1 is the ground:
        // free-slip du/dz = 0 there and w(ground) = 0.
        if (kk < 0 || kk >= NZ_l - 1)
        {
            return std::array<double, 2>{0.0, 0.0};
        }
        const int ii1 = i_neighbor_next(ii, NX, wrap_x);
        const double dz_center = grid_metric::vertical_center_distance(
            grid, ii, j, kk, NZ_l);
        const double dudz = (uf(ii, j, kk + 1) - uf(ii, j, kk))
            / dz_center;
        const double dwdx = (wf(ii1, j, kk) - wf(ii, j, kk)) / dx;
        return std::array<double, 2>{dudz, dwdx};
    };
    auto yz_edge_gradients = [&](int jj, int kk)
    {
        if (kk < 0 || kk >= NZ_l - 1)
        {
            return std::array<double, 2>{0.0, 0.0};
        }
        const int jj1 = j_neighbor_next(jj, NY, wrap_y);
        const double dz_center = grid_metric::vertical_center_distance(
            grid, i, jj, kk, NZ_l);
        const double dvdz = (vf(i, jj, kk + 1) - vf(i, jj, kk))
            / dz_center;
        const double dwdy = (wf(i, jj1, kk) - wf(i, jj, kk)) / dy;
        return std::array<double, 2>{dvdz, dwdy};
    };

    const double s12a = S12_edge(im, jm), s12b = S12_edge(im, j);
    const double s12c = S12_edge(i, jm),  s12d = S12_edge(i, j);
    const auto xza = xz_edge_gradients(im, k - 1);
    const auto xzb = xz_edge_gradients(im, k);
    const auto xzc = xz_edge_gradients(i, k - 1);
    const auto xzd = xz_edge_gradients(i, k);
    const auto yza = yz_edge_gradients(jm, k - 1);
    const auto yzb = yz_edge_gradients(jm, k);
    const auto yzc = yz_edge_gradients(j, k - 1);
    const auto yzd = yz_edge_gradients(j, k);

    const double S12_sq = 0.25 * (s12a*s12a + s12b*s12b + s12c*s12c + s12d*s12d);
    const auto average_gradient_sq = [](const auto& a, const auto& b,
                                        const auto& c, const auto& d,
                                        std::size_t component)
    {
        return 0.25 * (
            a[component] * a[component] + b[component] * b[component]
            + c[component] * c[component] + d[component] * d[component]);
    };
    if (mode == "horizontal_only")
    {
        const double dwdx_sq = average_gradient_sq(
            xza, xzb, xzc, xzd, 1);
        const double dwdy_sq = average_gradient_sq(
            yza, yzb, yzc, yzd, 1);
        return 2.0 * (S11 * S11 + S22 * S22)
            + 4.0 * S12_sq + dwdx_sq + dwdy_sq;
    }
    if (mode == "vertical_only")
    {
        const double dudz_sq = average_gradient_sq(
            xza, xzb, xzc, xzd, 0);
        const double dvdz_sq = average_gradient_sq(
            yza, yzb, yzc, yzd, 0);
        return dudz_sq + dvdz_sq + 2.0 * S33 * S33;
    }
    const auto average_symmetric_sq = [](const auto& a, const auto& b,
                                         const auto& c, const auto& d)
    {
        const auto square = [](const auto& gradient)
        {
            const double strain = 0.5 * (gradient[0] + gradient[1]);
            return strain * strain;
        };
        return 0.25 * (square(a) + square(b) + square(c) + square(d));
    };
    const double S13_sq = average_symmetric_sq(xza, xzb, xzc, xzd);
    const double S23_sq = average_symmetric_sq(yza, yzb, yzc, yzd);
    return 2.0 * (S11 * S11 + S22 * S22 + S33 * S33)
        + 4.0 * (S12_sq + S13_sq + S23_sq);
}

double compute_strain_sq_centered(
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    int i, int j, int k)
{
    return compute_directional_strain_sq_centered(
        state, grid, i, j, k, "3d");
}

/**
 * @brief Computes the filter width.
 */
double compute_filter_width(
    const TurbulenceConfig& cfg,
    const GridMetrics& grid,
    int i, int j, int k
)
{
    if (grid.terrain_metrics_active && grid.terrain_metrics == nullptr)
        throw std::invalid_argument(
            "terrain filter width requires complete mapped geometry");
    int nz_hint = static_cast<int>(grid.dz.size());
    if (nz_hint <= 0 && grid.terrain_metrics != nullptr)
    {
        nz_hint = grid.terrain_metrics->z.size_z();
    }
    if (nz_hint <= 0)
    {
        throw std::invalid_argument(
            "turbulence filter width requires explicit vertical metrics");
    }

    // Mapped finite-volume filter size uses the computational horizontal
    // footprint and physical Jacobian thickness. Face-normal CFL distances
    // are not cell-volume dimensions and would inflate Delta on a slope.
    const double dx_local = grid.terrain_metrics_active
        ? grid.terrain_metrics->dx
        : grid_metric::local_dx(grid, i, j, k);
    const double dy_local = grid.terrain_metrics_active
        ? grid.terrain_metrics->dy
        : grid_metric::local_dy(grid, i, j, k);
    const double dz_local = grid.terrain_metrics_active
        ? grid.terrain_metrics->J(i, j, k)
            * grid.terrain_metrics->dzeta
        : grid_metric::vertical_cell_thickness(
            grid, i, j, k, nz_hint);

    if (cfg.filter_width == "dx")
    {
        return std::sqrt(dx_local * dy_local);
    }

    else if (cfg.filter_width == "cubic_root")
    {
        return std::cbrt(dx_local * dy_local * dz_local);
    }
    else if (cfg.filter_width == "user")
    {
        if (!std::isfinite(cfg.user_filter_width_m) ||
            cfg.user_filter_width_m <= 0.0)
            throw std::invalid_argument(
                "user turbulence filter width must be positive and finite");
        return cfg.user_filter_width_m;
    }
    throw std::invalid_argument(
        "unsupported turbulence filter width '" + cfg.filter_width + "'");
}

/**
 * @brief Computes the Smagorinsky viscosity.
 */

double compute_smagorinsky_viscosity(
    double Cs,
    double Delta,
    double strain_mag,
    double stability_factor
)
{
    if (!std::isfinite(Cs) || Cs < 0.0 ||
        !std::isfinite(Delta) || Delta <= 0.0 ||
        !std::isfinite(strain_mag) || strain_mag < 0.0 ||
        !std::isfinite(stability_factor) || stability_factor < 0.0)
    {
        throw std::invalid_argument(
            "Smagorinsky inputs must be finite with non-negative coefficients "
            "and positive filter width");
    }
    const double nu_t =
        Cs * Cs * Delta * Delta * strain_mag * stability_factor;
    if (!std::isfinite(nu_t))
    {
        throw std::overflow_error("Smagorinsky viscosity overflowed");
    }
    return nu_t;
}

/**
 * @brief Computes the eddy diffusivities.
 */
void compute_eddy_diffusivities(
    double nu_t,
    double Pr_t,
    double Sc_t,
    double& K_theta,
    double& K_q,
    double& K_tke
)
{
    if (!std::isfinite(nu_t) || nu_t < 0.0 ||
        !std::isfinite(Pr_t) || Pr_t <= 0.0 ||
        !std::isfinite(Sc_t) || Sc_t <= 0.0)
    {
        throw std::invalid_argument(
            "eddy diffusivity inputs require non-negative finite viscosity "
            "and positive finite Pr_t/Sc_t");
    }
    K_theta = nu_t / Pr_t;
    K_q = nu_t / Sc_t;
    K_tke = nu_t / Pr_t;
    if (!std::isfinite(K_theta) || !std::isfinite(K_q) ||
        !std::isfinite(K_tke))
    {
        throw std::overflow_error("eddy diffusivity overflowed");
    }
}

/**
 * @brief Computes the momentum diffusion tendencies.
 */
void compute_momentum_diffusion_tendencies(
    const TurbulenceConfig& cfg,
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    const Field3D& nu_t,
    Field3D& dudt_sgs,
    Field3D& dvdt_sgs,
    Field3D& dwdt_sgs,
    bool periodic_x,
    bool periodic_y,
    double* mapped_boundary_power_w
)
{
    if (mapped_boundary_power_w != nullptr)
        *mapped_boundary_power_w =
            std::numeric_limits<double>::quiet_NaN();
    const bool use_h = cfg.mode != "vertical_only";
    const bool use_v = cfg.mode != "horizontal_only";
    // Physical C-grid rows wrap only on genuinely periodic topology.
    const bool wrap_x = periodic_x;
    const bool wrap_y = periodic_y;
    const int NX = state.NR, NY = state.NTH, NZl = state.NZ;
    if (!state.c_grid)
    {
        collocated_eddy_viscosity::momentum_tendencies(
            cfg, state, grid, nu_t, dudt_sgs, dvdt_sgs, dwdt_sgs,
            wrap_x, wrap_y);
        if (grid.terrain_metrics_active
            && mapped_boundary_power_w != nullptr)
            *mapped_boundary_power_w = 0.0;
        return;
    }
    if (NX < 3 || NY < 3 || NZl < 2)
    {
        throw std::invalid_argument(
            "SGS momentum diffusion requires a C-grid of at least 3x3x2");
    }
    if (cfg.mode != "3d" && cfg.mode != "horizontal_only" &&
        cfg.mode != "vertical_only")
    {
        throw std::invalid_argument("unsupported turbulence mode '" + cfg.mode + "'");
    }
    grid_metric::sanitize_positive(grid.dx);
    grid_metric::sanitize_positive(grid.dy);
    grid_metric::validate_vertical_metrics(grid, NX, NY, NZl);
    require_field(state.u, state, "u", false, false);
    require_field(state.v, state, "v", false, false);
    require_field(state.w, state, "w", false, false);
    require_field(state.rho, state, "rho", true, false);
    require_field(&nu_t, state, "nu_t", false, true);
    require_output_shape(dudt_sgs, state, "dudt_sgs");
    require_output_shape(dvdt_sgs, state, "dvdt_sgs");
    require_output_shape(dwdt_sgs, state, "dwdt_sgs");
    if (grid.terrain_metrics_active)
    {
        if (grid.terrain_metrics == nullptr)
            throw std::invalid_argument(
                "terrain SGS momentum diffusion requires complete mapped geometry");
        terrain_cgrid_momentum_diffusion_tendencies(
            cfg, state, grid, nu_t, dudt_sgs, dvdt_sgs, dwdt_sgs,
            wrap_x, wrap_y, mapped_boundary_power_w);
        return;
    }

    const auto& u = *state.u;
    const auto& v = *state.v;
    const auto& w = *state.w;
    const auto& rho = *state.rho;

    // Conservative live-density stress divergence
    //   du_i/dt = (1/rho_face) d_j (2 rho_edge nu S_ij)
    // with each tau component formed where it naturally lives on the
    // C-grid (diagonals at centers, off-diagonals at edges with nu
    // 4-point-averaged). Live dry density is carried at each natural
    // center/face/edge location; using rho0 conserved the wrong momentum
    // currency as soon as the prognostic density departed from base state.
    // Replaces the local-nu Laplacian, which
    // conserved nothing where grad(nu) peaked (cloud/outflow edges) and
    // dropped the transpose term. The ground/lid closures are
    // ZERO-FLUX shear stresses (tau_13 = tau_23 = 0 at the ground and
    // lid faces - the free-slip M19 convention), so vertical mixing
    // conserves column momentum exactly.
    auto rho_c = [&](int ii, int jj, int kk) -> double
    { return static_cast<double>(rho[ii][jj][kk]); };
    auto rho_xface = [&](int ii, int jj, int kk) -> double
    {
        const int i1 = i_neighbor_next(ii, NX, wrap_x);
        return 0.5 * (rho_c(ii, jj, kk) + rho_c(i1, jj, kk));
    };
    auto rho_yface = [&](int ii, int jj, int kk) -> double
    {
        const int j1 = j_neighbor_next(jj, NY, wrap_y);
        return 0.5 * (rho_c(ii, jj, kk) + rho_c(ii, j1, kk));
    };
    auto nu_c = [&](int ii, int jj, int kk) -> double
    { return static_cast<double>(nu_t[ii][jj][kk]); };
    auto mu_c = [&](int ii, int jj, int kk) -> double
    { return rho_c(ii, jj, kk) * nu_c(ii, jj, kk); };
    auto mu_xyedge = [&](int ii, int jj, int kk) -> double
    {
        const int i1 = i_neighbor_next(ii, NX, wrap_x);
        const int j1 = j_neighbor_next(jj, NY, wrap_y);
        return 0.25 * (mu_c(ii, jj, kk) + mu_c(i1, jj, kk)
                     + mu_c(ii, j1, kk) + mu_c(i1, j1, kk));
    };
    auto mu_xzedge = [&](int ii, int jj, int kk) -> double
    {
        const int i1 = i_neighbor_next(ii, NX, wrap_x);
        const double lower = 0.5 * (
            mu_c(ii, jj, kk) + mu_c(i1, jj, kk));
        const double upper = 0.5 * (
            mu_c(ii, jj, kk + 1) + mu_c(i1, jj, kk + 1));
        const double lower_dz = grid_metric::vertical_cell_thickness(
            grid, ii, jj, kk, NZl);
        const double upper_dz = grid_metric::vertical_cell_thickness(
            grid, ii, jj, kk + 1, NZl);
        return (upper_dz * lower + lower_dz * upper) /
            (lower_dz + upper_dz);
    };
    auto mu_yzedge = [&](int ii, int jj, int kk) -> double
    {
        const int j1 = j_neighbor_next(jj, NY, wrap_y);
        const double lower = 0.5 * (
            mu_c(ii, jj, kk) + mu_c(ii, j1, kk));
        const double upper = 0.5 * (
            mu_c(ii, jj, kk + 1) + mu_c(ii, j1, kk + 1));
        const double lower_dz = grid_metric::vertical_cell_thickness(
            grid, ii, jj, kk, NZl);
        const double upper_dz = grid_metric::vertical_cell_thickness(
            grid, ii, jj, kk + 1, NZl);
        return (upper_dz * lower + lower_dz * upper) /
            (lower_dz + upper_dz);
    };
    auto wf = [&](int ii, int jj, int kk) -> double
    { return (kk < 0) ? 0.0 : static_cast<double>(w[ii][jj][kk]); };

    dudt_sgs.fill(0.0f);
    dvdt_sgs.fill(0.0f);
    dwdt_sgs.fill(0.0f);

    const double dx = grid.dx;
    const double dy = grid.dy;

    // The three components read immutable state/coefficient fields and write
    // disjoint outputs.  Retain one worker team and avoid component barriers.
    #pragma omp parallel if(static_cast<std::size_t>(NX) * NY * NZl >= 32768)
    {
    // ── du/dt at x-faces (u[i] between centers i and i+1) ──
    #pragma omp for collapse(2) nowait
    for (int i = 1; i <= (wrap_x ? NX - 2 : NX - 3); ++i)
    {
        for (int j = 1; j <= NY - 2; ++j)
        {
            const int jmm = j_neighbor_prev(j, NY, wrap_y);
            const int imm = i_neighbor_prev(i, NX, wrap_x);
            const int ipp = i_neighbor_next(i, NX, wrap_x);
            // T13(k) is the upper face stress of cell k and the lower face
            // stress of cell k+1. Retain that exact binary64 value while
            // walking the contiguous column instead of rebuilding it twice.
            // The ground value is the same literal zero returned by the
            // fail-closed edge helper for k=-1.
            double lower_T13 = 0.0;
            for (int k = 0; k < NZl - 1; ++k)
            {
                const double dz_k = grid_metric::vertical_cell_thickness(
                    grid, i, j, k, NZl);

                double div = 0.0;
                if (use_h)
                {
                    // T11 at the two centers flanking the face.
                    const double S11_r = (static_cast<double>(u[ipp][j][k])
                                        - static_cast<double>(u[i][j][k])) / dx;
                    const double S11_l = (static_cast<double>(u[i][j][k])
                                        - static_cast<double>(u[imm][j][k])) / dx;
                    const double T11_r = 2.0 * mu_c(ipp, j, k) * S11_r;
                    const double T11_l = 2.0 * mu_c(i, j, k) * S11_l;
                    div += (T11_r - T11_l) / dx;

                    // T12 at the xy-edges above (j) and below (j-1) the face.
                    auto T12_edge = [&](int je) -> double
                    {
                        const int i1 = i_neighbor_next(i, NX, wrap_x);
                        const int je1 = j_neighbor_next(je, NY, wrap_y);
                        const double S12 = 0.5 * (
                            (static_cast<double>(u[i][je1][k])
                           - static_cast<double>(u[i][je][k])) / dy
                          + (static_cast<double>(v[i1][je][k])
                           - static_cast<double>(v[i][je][k])) / dx);
                        return 2.0 * mu_xyedge(i, je, k) * S12;
                    };
                    div += (T12_edge(j) - T12_edge(jmm)) / dy;
                }
                if (use_v)
                {
                    // T13 at the xz-edges above (k) and below (k-1); the
                    // ground and lid edges carry zero stress.
                    auto T13_edge = [&](int ke) -> double
                    {
                        if (ke < 0 || ke >= NZl - 1)
                        {
                            return 0.0;
                        }
                        const double dz_center =
                            grid_metric::vertical_center_distance(
                                grid, i, j, ke, NZl);
                        const double dudz =
                            (static_cast<double>(u[i][j][ke + 1])
                           - static_cast<double>(u[i][j][ke])) / dz_center;
                        const double stress_gradient = use_h
                            ? dudz + (wf(ipp, j, ke) - wf(i, j, ke)) / dx
                            : dudz;
                        return mu_xzedge(i, j, ke) * stress_gradient;
                    };
#ifdef NIMBUS_RECOMPUTE_VERTICAL_SGS_STRESSES
                    div += (T13_edge(k) - T13_edge(k - 1)) / dz_k;
#else
                    const double upper_T13 = T13_edge(k);
                    div += (upper_T13 - lower_T13) / dz_k;
                    lower_T13 = upper_T13;
#endif
                }
                dudt_sgs[i][j][k] =
                    static_cast<float>(div / rho_xface(i, j, k));
            }
        }
    }

    // ── dv/dt at y-faces (v[j] between centers j and j+1) ──
    #pragma omp for collapse(2) nowait
    for (int i = 0; i < NX; ++i)
    {
        for (int j = 1; j <= NY - 2; ++j)
        {
            if (i == 0 || i == NX - 1)
            {
                continue;
            }
            // The y-face j sits between rows j and j+1: valid faces are
            // j in [0, NY-2] (wrap: all).
            if (!wrap_y && j >= NY - 2)
            {
                continue;
            }
            const int jmm = j_neighbor_prev(j, NY, wrap_y);
            const int jpp = j_neighbor_next(j, NY, wrap_y);
            // The same natural-edge T23 value borders consecutive vertical
            // control volumes. Reuse it without rounding or extra storage.
            double lower_T23 = 0.0;
            for (int k = 0; k < NZl - 1; ++k)
            {
                const double dz_k = grid_metric::vertical_cell_thickness(
                    grid, i, j, k, NZl);

                double div = 0.0;
                if (use_h)
                {
                    // T22 at the two centers flanking the face.
                    const double S22_n = (static_cast<double>(v[i][jpp][k])
                                        - static_cast<double>(v[i][j][k])) / dy;
                    const double S22_s = (static_cast<double>(v[i][j][k])
                                        - static_cast<double>(v[i][jmm][k])) / dy;
                    const double T22_n = 2.0 * mu_c(i, jpp, k) * S22_n;
                    const double T22_s = 2.0 * mu_c(i, j, k) * S22_s;
                    div += (T22_n - T22_s) / dy;

                    // T21 (= T12) at the xy-edges right (i) and left (i-1).
                    auto T21_edge = [&](int ie) -> double
                    {
                        const int ie1 = i_neighbor_next(ie, NX, wrap_x);
                        const int j1 = j_neighbor_next(j, NY, wrap_y);
                        const double S12 = 0.5 * (
                            (static_cast<double>(u[ie][j1][k])
                           - static_cast<double>(u[ie][j][k])) / dy
                          + (static_cast<double>(v[ie1][j][k])
                           - static_cast<double>(v[ie][j][k])) / dx);
                        return 2.0 * mu_xyedge(ie, j, k) * S12;
                    };
                    div += (T21_edge(i)
                          - T21_edge(i_neighbor_prev(i, NX, wrap_x))) / dx;
                }
                if (use_v)
                {
                    auto T23_edge = [&](int ke) -> double
                    {
                        if (ke < 0 || ke >= NZl - 1)
                        {
                            return 0.0;
                        }
                        const int j1 = j_neighbor_next(j, NY, wrap_y);
                        const double dz_center =
                            grid_metric::vertical_center_distance(
                                grid, i, j, ke, NZl);
                        const double dvdz =
                            (static_cast<double>(v[i][j][ke + 1])
                           - static_cast<double>(v[i][j][ke])) / dz_center;
                        const double stress_gradient = use_h
                            ? dvdz + (wf(i, j1, ke) - wf(i, j, ke)) / dy
                            : dvdz;
                        return mu_yzedge(i, j, ke) * stress_gradient;
                    };
#ifdef NIMBUS_RECOMPUTE_VERTICAL_SGS_STRESSES
                    div += (T23_edge(k) - T23_edge(k - 1)) / dz_k;
#else
                    const double upper_T23 = T23_edge(k);
                    div += (upper_T23 - lower_T23) / dz_k;
                    lower_T23 = upper_T23;
#endif
                }
                dvdt_sgs[i][j][k] =
                    static_cast<float>(div / rho_yface(i, j, k));
            }
        }
    }

    // ── dw/dt at z-faces (w[k] between centers k and k+1) ──
    #pragma omp for collapse(2) nowait
    for (int i = 0; i < NX; ++i)
    {
        for (int j = 1; j <= NY - 2; ++j)
        {
            if (i == 0 || i == NX - 1)
            {
                continue;
            }
#ifndef NIMBUS_RECOMPUTE_VERTICAL_SGS_STRESSES
            // T33 at center k+1 becomes the identical lower stress on the
            // next z-face. Seed the rolling value with the original rigid-
            // ground expression; all arithmetic remains binary64.
            double lower_T33 = 0.0;
            if (use_v)
            {
                const double ground_dz =
                    grid_metric::vertical_cell_thickness(
                        grid, i, j, 0, NZl);
                const double ground_S33 =
                    (wf(i, j, 0) - wf(i, j, -1)) / ground_dz;
                lower_T33 = 2.0 * mu_c(i, j, 0) * ground_S33;
            }
#endif
            for (int k = 0; k < NZl - 1; ++k)
            {
                const double dz_bottom =
                    grid_metric::vertical_cell_thickness(
                        grid, i, j, k, NZl);
                const double dz_top =
                    grid_metric::vertical_cell_thickness(
                        grid, i, j, k + 1, NZl);
                const double dz_dual =
                    grid_metric::vertical_center_distance(
                        grid, i, j, k, NZl);
                const double mass_per_area = 0.5 * (
                    rho_c(i, j, k) * dz_bottom +
                    rho_c(i, j, k + 1) * dz_top);

                double force_per_area = 0.0;
                if (use_v)
                {
                    // T33 at the two centers flanking the face; the
                    // ground enters via w[-1] = 0.
                    const double S33_t =
                        (wf(i, j, k + 1) - wf(i, j, k)) / dz_top;
#ifdef NIMBUS_RECOMPUTE_VERTICAL_SGS_STRESSES
                    const double S33_b =
                        (wf(i, j, k) - wf(i, j, k - 1)) / dz_bottom;
#endif
                    const double T33_t = 2.0 * mu_c(i, j, k + 1) * S33_t;
#ifdef NIMBUS_RECOMPUTE_VERTICAL_SGS_STRESSES
                    const double T33_b = 2.0 * mu_c(i, j, k) * S33_b;
                    force_per_area += T33_t - T33_b;
#else
                    force_per_area += T33_t - lower_T33;
                    lower_T33 = T33_t;
#endif
                }
                if (use_h)
                {
                    // T31 (= T13) at the xz-edges right (i) and left (i-1).
                    auto T31_edge = [&](int ie) -> double
                    {
                        const int ie1 = i_neighbor_next(ie, NX, wrap_x);
                        const double dwdx =
                            (wf(ie1, j, k) - wf(ie, j, k)) / dx;
                        const double stress_gradient = use_v
                            ? (static_cast<double>(u[ie][j][k + 1])
                               - static_cast<double>(u[ie][j][k])) / dz_dual
                                + dwdx
                            : dwdx;
                        return mu_xzedge(ie, j, k) * stress_gradient;
                    };
                    force_per_area += dz_dual * (T31_edge(i)
                          - T31_edge(i_neighbor_prev(i, NX, wrap_x))) / dx;

                    // T32 (= T23) at the yz-edges north (j) and south (j-1).
                    auto T32_edge = [&](int je) -> double
                    {
                        const int je1 = j_neighbor_next(je, NY, wrap_y);
                        const double dwdy =
                            (wf(i, je1, k) - wf(i, je, k)) / dy;
                        const double stress_gradient = use_v
                            ? (static_cast<double>(v[i][je][k + 1])
                               - static_cast<double>(v[i][je][k])) / dz_dual
                                + dwdy
                            : dwdy;
                        return mu_yzedge(i, je, k) * stress_gradient;
                    };
                    const int jmm = j_neighbor_prev(j, NY, wrap_y);
                    force_per_area += dz_dual *
                        (T32_edge(j) - T32_edge(jmm)) / dy;
                }
                dwdt_sgs[i][j][k] =
                    static_cast<float>(force_per_area / mass_per_area);
            }
        }
    }
    }
}

/**
 * @brief Computes the scalar diffusion tendencies.
 */
void compute_scalar_diffusion_tendencies(
    const TurbulenceConfig& cfg,
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    const Field3D& K_field,
    const Field3D& phi,
    Field3D& dphi_dt_sgs
)
{
    const bool use_h = cfg.mode != "vertical_only";
    const bool use_v = cfg.mode != "horizontal_only";
    // Conservative live-density flux form
    //   dphi/dt = (1/rho) d_j (rho_face K_face dphi/dx_j)
    // with K and live density averaged to faces, zero-flux ground and
    // lid closures, and the M16 j-topology rule. The old form was
    // K_center times a Laplacian - non-conservative wherever grad(K)
    // was nonzero, which is exactly the cloud and outflow edges.
    const bool wrap_x = periodic_x_topology(state);
    const bool wrap_y = periodic_y_topology(state);
    const int NX = state.NR, NY = state.NTH, NZl = state.NZ;
    if (!state.c_grid)
    {
        collocated_eddy_viscosity::scalar_tendency(
            cfg, state, grid, K_field, phi, dphi_dt_sgs,
            wrap_x, wrap_y);
        return;
    }
    if (NX < 3 || NY < 3 || NZl < 2)
    {
        throw std::invalid_argument(
            "SGS scalar diffusion requires a C-grid of at least 3x3x2");
    }
    if (cfg.mode != "3d" && cfg.mode != "horizontal_only" &&
        cfg.mode != "vertical_only")
    {
        throw std::invalid_argument("unsupported turbulence mode '" + cfg.mode + "'");
    }
    grid_metric::sanitize_positive(grid.dx);
    grid_metric::sanitize_positive(grid.dy);
    grid_metric::validate_vertical_metrics(grid, NX, NY, NZl);
    require_field(&K_field, state, "K_field", false, true);
    require_field(&phi, state, "scalar", false, false);
    require_field(state.rho, state, "rho", true, false);
    require_rho0(state);
    require_output_shape(dphi_dt_sgs, state, "dphi_dt_sgs");

    if (grid.terrain_metrics_active)
    {
        if (grid.terrain_metrics == nullptr)
        {
            throw std::invalid_argument(
                "terrain SGS scalar diffusion requires complete mapped geometry");
        }
        terrain_transport::cgrid_variable_scalar_diffusion_tendency(
            *grid.terrain_metrics, phi, *state.rho, K_field, K_field,
            use_h, use_v, wrap_x, wrap_y, dphi_dt_sgs);
        return;
    }

    const auto& rho = *state.rho;
    auto rho_c = [&](int ii, int jj, int kk) -> double
    { return static_cast<double>(rho[ii][jj][kk]); };
    auto K_at = [&](int ii, int jj, int kk) -> double
    { return static_cast<double>(K_field[ii][jj][kk]); };
    auto transport_at = [&](int ii, int jj, int kk) -> double
    { return rho_c(ii, jj, kk) * K_at(ii, jj, kk); };
    auto phi_at = [&](int ii, int jj, int kk) -> double
    { return static_cast<double>(phi[ii][jj][kk]); };

    const double dx = grid.dx;
    const double dy = grid.dy;

    dphi_dt_sgs.fill(0.0f);

    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(NX) * NY * NZl >= 32768)
    for (int i = 0; i < NX; ++i)
    {
        for (int j = 1; j <= NY - 2; ++j)
        {
            if (i == 0 || i == NX - 1)
            {
                continue;  // BC-owned cells (matches the i-edge skip)
            }
            const int jmm = !wrap_y && !state.open_lateral[2] && j == 1
                ? j : j_neighbor_prev(j, NY, wrap_y);
            const int jpp = !wrap_y && !state.open_lateral[3] && j == NY - 2
                ? j : j_neighbor_next(j, NY, wrap_y);
            const int imm = !wrap_x && !state.open_lateral[0] && i == 1
                ? i : i_neighbor_prev(i, NX, wrap_x);
            const int ipp = !wrap_x && !state.open_lateral[1] && i == NX - 2
                ? i : i_neighbor_next(i, NX, wrap_x);
            for (int k = 0; k < NZl; ++k)
            {
                const double dz_k = grid_metric::vertical_cell_thickness(
                    grid, i, j, k, NZl);

                double div = 0.0;
                if (use_h)
                {
                    const double Fx_e = 0.5 * (
                        transport_at(i, j, k) + transport_at(ipp, j, k))
                        * (phi_at(ipp, j, k) - phi_at(i, j, k)) / dx;
                    const double Fx_w = 0.5 * (
                        transport_at(imm, j, k) + transport_at(i, j, k))
                        * (phi_at(i, j, k) - phi_at(imm, j, k)) / dx;
                    const double Fy_n = 0.5 * (
                        transport_at(i, j, k) + transport_at(i, jpp, k))
                        * (phi_at(i, jpp, k) - phi_at(i, j, k)) / dy;
                    const double Fy_s = 0.5 * (
                        transport_at(i, jmm, k) + transport_at(i, j, k))
                        * (phi_at(i, j, k) - phi_at(i, jmm, k)) / dy;
                    div += (Fx_e - Fx_w) / dx + (Fy_n - Fy_s) / dy;
                }
                if (use_v)
                {
                    // Zero-flux ground (below k=0) and lid (above NZ-1).
                    const double Fz_t = (k < NZl - 1)
                        ? 0.5 * (transport_at(i, j, k) +
                                 transport_at(i, j, k + 1))
                          * (phi_at(i, j, k + 1) - phi_at(i, j, k))
                          / grid_metric::vertical_center_distance(
                              grid, i, j, k, NZl)
                        : 0.0;
                    const double Fz_b = (k > 0)
                        ? 0.5 * (transport_at(i, j, k - 1) +
                                 transport_at(i, j, k))
                          * (phi_at(i, j, k) - phi_at(i, j, k - 1))
                          / grid_metric::vertical_center_distance(
                              grid, i, j, k - 1, NZl)
                        : 0.0;
                    div += (Fz_t - Fz_b) / dz_k;
                }
                dphi_dt_sgs[i][j][k] = static_cast<float>(
                    div / rho_c(i, j, k));
            }
        }
    }
}

/**
 * @brief Applies Richardson-number stability correction to SGS coefficients.
 */
double stability_correction_ri(double Ri, double Ri_crit) {
    if (!std::isfinite(Ri) || !std::isfinite(Ri_crit) || Ri_crit <= 0.0)
    {
        throw std::invalid_argument(
            "Richardson stability correction requires finite Ri and positive Ri_crit");
    }
    if (Ri > 0.0) {
        return std::max(0.1, 1.0 - Ri / Ri_crit);
    } else {
        return 1.0 - 0.5 * Ri;
    }
}

double lilly_equilibrium_strain_rate(double strain_squared, double n2, double prandtl)
{
    if (!std::isfinite(strain_squared) || strain_squared < 0.0 || !std::isfinite(n2)
        || !std::isfinite(prandtl) || prandtl <= 0.0)
        throw std::invalid_argument("Lilly mixing requires finite strain, signed buoyancy and positive Prandtl number");
    double result;
    if (n2 >= 0.0)
    {
        if (strain_squared == 0.0 || n2 / strain_squared >= prandtl) return 0.0;
        result = std::sqrt(strain_squared) * std::sqrt(1.0 - (n2 / strain_squared) / prandtl);
    }
    else
        result = std::hypot(std::sqrt(strain_squared), std::sqrt(-n2) / std::sqrt(prandtl));
    if (!std::isfinite(result)) throw std::overflow_error("Lilly equilibrium strain overflowed");
    return result;
}

/**
 * @brief Computes the Brunt-Väisälä frequency.
 */
double compute_buoyancy_frequency_squared(
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    int i, int j, int k,
    bool moist
)
{
    if (!state.theta)
    {
        throw std::invalid_argument(
            "Brunt-Vaisala frequency requires potential temperature");
    }
    const PhysicalCellDomain domain = physical_scalar_domain(
        state.NR, state.NTH, state.NZ,
        state.c_grid ? StaggerType::CGrid : StaggerType::Collocated);
    if (k < domain.k_begin || k >= domain.k_end)
    {
        throw std::out_of_range("Brunt-Vaisala vertical index is outside the state");
    }
    if (domain.nz() == 1)
    {
        return 0.0;
    }

    const auto virtual_theta = [&](int level)
    {
        const double theta = static_cast<double>((*state.theta)[i][j][level]);
        if (!moist)
            return theta;
        if (state.qv == nullptr)
            throw std::invalid_argument(
                "moist Brunt-Vaisala frequency requires water vapor");
        const double vapor = static_cast<double>((*state.qv)[i][j][level]);
        if (!std::isfinite(vapor) || vapor < 0.0)
            throw std::invalid_argument(
                "moist Brunt-Vaisala frequency requires finite non-negative water vapor");
        return theta * (1.0 + 0.61 * vapor);
    };
    const double theta_k = virtual_theta(k);
    const int lower = k + 1 < domain.k_end ? k : k - 1;
    const int upper = lower + 1;
    const double theta_lower = virtual_theta(lower);
    const double theta_upper = virtual_theta(upper);
    if (!std::isfinite(theta_k) || !std::isfinite(theta_lower) || !std::isfinite(theta_upper) || theta_k <= 1.0e-6)
    {
        throw std::invalid_argument(
            "Brunt-Vaisala frequency requires finite positive potential temperature");
    }

    const double dz_local = grid_metric::vertical_center_distance(
        grid, i, j, lower, state.NZ);
    const double dtheta_dz = (theta_upper - theta_lower) / dz_local;

    const double N2 = (turbulence_constants::g / theta_k) * dtheta_dz;
    return N2;
}

double compute_brunt_vaisala_frequency(
    const TurbulenceStateView& state, const GridMetrics& grid,
    int i, int j, int k, bool moist)
{
    return std::sqrt(std::max(0.0,
        compute_buoyancy_frequency_squared(state, grid, i, j, k, moist)));
}

double cgrid_stability_factor_ri(
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    int i, int j, int k,
    bool moist)
{
    const double buoyancy_frequency_squared =
        compute_buoyancy_frequency_squared(state, grid, i, j, k, moist);
    const double strain_sq =
        compute_strain_sq_centered(state, grid, i, j, k);
    if (!std::isfinite(strain_sq) || strain_sq < 0.0)
        throw std::runtime_error(
            "Richardson correction produced invalid strain");
    const double richardson = strain_sq > 1.0e-20
        ? buoyancy_frequency_squared / strain_sq
        : 0.0;
    return stability_correction_ri(richardson);
}

double compute_brunt_vaisala_frequency(
    const TurbulenceStateView& state,
    int i, int j, int k
)
{
    (void)state;
    (void)i;
    (void)j;
    (void)k;
    throw std::invalid_argument(
        "Brunt-Vaisala frequency requires explicit grid metrics");
}

/**
 * @brief Applies the positivity limits.
 */
void apply_positivity_limits(
    Field3D& field,
    double min_value,
    double max_value
)
{
    if (!std::isfinite(min_value) || !std::isfinite(max_value) ||
        min_value > max_value)
    {
        throw std::invalid_argument(
            "positivity limits require finite ordered bounds");
    }
    for (std::size_t index = 0; index < field.size(); ++index)
    {
        if (!std::isfinite(static_cast<double>(field.data()[index])))
        {
            throw std::invalid_argument(
                "positivity limits cannot repair a non-finite field");
        }
    }
    #pragma omp parallel for collapse(2) if(field.size() >= 32768)
    for (int i = 0; i < field.size_r(); ++i)
    {
        for (int j = 0; j < field.size_th(); ++j)
        {
            for (int k = 0; k < field.size_z(); ++k)
            {
                const float old_value = static_cast<float>(field[i][j][k]);
                float new_value = old_value;
                new_value = std::max(static_cast<float>(min_value),
                    std::min(static_cast<float>(max_value), new_value));
                field[i][j][k] = new_value;
            }
        }
    }
}

/**
 * @brief Initializes the 3D field.
 */
void initialize_3d_field(
    std::vector<std::vector<std::vector<float>>>& field,
    int NR, int NTH, int NZ,
    float value
)
{
    field.assign(NR, std::vector<std::vector<float>>(
                NTH, std::vector<float>(NZ, value)));
}

} // namespace tmv::eddy_viscosity
