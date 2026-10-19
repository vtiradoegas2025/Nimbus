#include "terrain/conservative_transport.hpp"

#include "core/infra/physical_constants.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
void require_shape(const TerrainField3D& field, int nr, int nth, int nz,
                   const char* label)
{
    if (field.size_r() != nr || field.size_th() != nth || field.size_z() != nz)
    {
        throw std::invalid_argument(std::string(label) + " has the wrong shape");
    }
}

void require_finite(const TerrainField3D& field, const char* label)
{
    for (double value : field.values)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument(std::string(label) + " must be finite");
        }
    }
}
} // namespace

namespace tmv::terrain_transport
{

namespace
{
double rho_x_face(const Field3D& rho, int i, int j, int k)
{
    return 0.5 * (static_cast<double>(rho[i][j][k])
                + static_cast<double>(rho[i + 1][j][k]));
}

double rho_y_face(const Field3D& rho, int i, int j, int k)
{
    return 0.5 * (static_cast<double>(rho[i][j][k])
                + static_cast<double>(rho[i][j + 1][k]));
}

double rho_zeta_face(const Field3D& rho, int i, int j, int k)
{
    if (k + 1 < rho.size_z())
        return 0.5 * (static_cast<double>(rho[i][j][k])
                    + static_cast<double>(rho[i][j][k + 1]));
    return static_cast<double>(rho[i][j][k]);
}

double u_at_cell_center(const Field3D& u, int i, int j, int k)
{
    return 0.5 * (static_cast<double>(u[i - 1][j][k])
                + static_cast<double>(u[i][j][k]));
}

double v_at_cell_center(const Field3D& v, int i, int j, int k)
{
    return 0.5 * (static_cast<double>(v[i][j - 1][k])
                + static_cast<double>(v[i][j][k]));
}

double rho_u_at_cell_center(const Field3D& rho, const Field3D& u,
                            int i, int j, int k,
                            double frame_u_ms = 0.0)
{
    return 0.5 * (rho_x_face(rho, i - 1, j, k)
                    * (static_cast<double>(u[i - 1][j][k]) - frame_u_ms)
                + rho_x_face(rho, i, j, k)
                    * (static_cast<double>(u[i][j][k]) - frame_u_ms));
}

double rho_v_at_cell_center(const Field3D& rho, const Field3D& v,
                            int i, int j, int k,
                            double frame_v_ms = 0.0)
{
    return 0.5 * (rho_y_face(rho, i, j - 1, k)
                    * (static_cast<double>(v[i][j - 1][k]) - frame_v_ms)
                + rho_y_face(rho, i, j, k)
                    * (static_cast<double>(v[i][j][k]) - frame_v_ms));
}

template <typename AtCenter>
double interpolate_to_zeta_face(const AtCenter& at_center, int k, int nz)
{
    if (k + 1 < nz)
        return 0.5 * (at_center(k) + at_center(k + 1));
    return at_center(k);
}

void require_scalar_shape(const TerrainMetrics3D& metrics,
                          const Field3D& scalar,
                          const char* label)
{
    if (scalar.size_r() != metrics.J.size_r()
        || scalar.size_th() != metrics.J.size_th()
        || scalar.size_z() != metrics.J.size_z())
    {
        throw std::invalid_argument(std::string(label)
                                    + " must match terrain geometry");
    }
}

template <typename ValueAt>
double column_gradient_z(const TerrainMetrics3D& metrics,
                         const ValueAt& value_at,
                         int i, int j, int k)
{
    const int nz = metrics.J.size_z();
    if (nz < 2) return 0.0;

    const int lower = (k == 0) ? 0 : k - 1;
    const int upper = (k + 1 < nz) ? k + 1 : nz - 1;
    const double delta_z = metrics.z(i, j, upper)
                         - metrics.z(i, j, lower);
    if (!std::isfinite(delta_z)
        || std::abs(delta_z) <= terrain_constants::epsilon)
    {
        throw std::invalid_argument(
            "terrain scalar gradient requires distinct finite physical heights");
    }
    return (value_at(i, j, upper) - value_at(i, j, lower)) / delta_z;
}

template <typename ValueAt>
double physical_gradient_x(const TerrainMetrics3D& metrics,
                           const ValueAt& value_at,
                           int i, int j, int k)
{
    const double computational_gradient =
        (value_at(i + 1, j, k) - value_at(i, j, k)) / metrics.dx;
    const double slope = (metrics.z(i + 1, j, k) - metrics.z(i, j, k))
                       / metrics.dx;
    const double vertical_gradient = 0.5 * (
        column_gradient_z(metrics, value_at, i, j, k)
        + column_gradient_z(metrics, value_at, i + 1, j, k));
    return computational_gradient - slope * vertical_gradient;
}

template <typename ValueAt>
double physical_gradient_y(const TerrainMetrics3D& metrics,
                           const ValueAt& value_at,
                           int i, int j, int k)
{
    const double computational_gradient =
        (value_at(i, j + 1, k) - value_at(i, j, k)) / metrics.dy;
    const double slope = (metrics.z(i, j + 1, k) - metrics.z(i, j, k))
                       / metrics.dy;
    const double vertical_gradient = 0.5 * (
        column_gradient_z(metrics, value_at, i, j, k)
        + column_gradient_z(metrics, value_at, i, j + 1, k));
    return computational_gradient - slope * vertical_gradient;
}

template <typename ValueAt>
double physical_gradient_z_face(const TerrainMetrics3D& metrics,
                                const ValueAt& value_at,
                                int i, int j, int k)
{
    const double delta_z = metrics.z(i, j, k + 1)
                         - metrics.z(i, j, k);
    if (!std::isfinite(delta_z)
        || std::abs(delta_z) <= terrain_constants::epsilon)
    {
        throw std::invalid_argument(
            "terrain vertical gradient requires distinct finite physical heights");
    }
    return (value_at(i, j, k + 1) - value_at(i, j, k)) / delta_z;
}

void require_ground_surface_frame(const GroundSurfaceFrame& frame)
{
    const double values[] = {frame.slope_x, frame.slope_y,
                             frame.normal_x, frame.normal_y, frame.normal_z,
                             frame.area_ratio};
    for (double value : values)
    {
        if (!std::isfinite(value))
            throw std::invalid_argument("terrain ground surface frame must be finite");
    }
    if (frame.area_ratio < 1.0)
        throw std::invalid_argument("terrain ground surface area ratio must be at least one");

    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon()
                           * frame.area_ratio;
    const double normal_norm = std::hypot(
        std::hypot(frame.normal_x, frame.normal_y), frame.normal_z);
    if (std::abs(normal_norm - 1.0) > tolerance
        || std::abs(frame.normal_x * frame.area_ratio + frame.slope_x)
            > tolerance
        || std::abs(frame.normal_y * frame.area_ratio + frame.slope_y)
            > tolerance
        || std::abs(frame.normal_z * frame.area_ratio - 1.0) > tolerance)
    {
        throw std::invalid_argument(
            "terrain ground surface frame is geometrically inconsistent");
    }
}

void require_physical_stress(const SymmetricStressTensor& stress)
{
    const double values[] = {stress.xx, stress.yy, stress.zz,
                             stress.xy, stress.xz, stress.yz};
    for (double value : values)
    {
        if (!std::isfinite(value))
            throw std::invalid_argument("terrain physical stress must be finite");
    }
}
} // namespace

PhysicalVector3 physical_gradient_from_computational(
    double derivative_x_at_zeta,
    double derivative_y_at_zeta,
    double derivative_zeta,
    double jacobian,
    double metric_gx,
    double metric_gy)
{
    if (!std::isfinite(derivative_x_at_zeta)
        || !std::isfinite(derivative_y_at_zeta)
        || !std::isfinite(derivative_zeta)
        || !std::isfinite(jacobian)
        || jacobian <= terrain_constants::epsilon
        || !std::isfinite(metric_gx)
        || !std::isfinite(metric_gy))
    {
        throw std::invalid_argument(
            "terrain physical-gradient transform requires finite derivatives/metrics and a positive Jacobian");
    }
    const double physical_z = derivative_zeta / jacobian;
    return {derivative_x_at_zeta + metric_gx * physical_z,
            derivative_y_at_zeta + metric_gy * physical_z,
            physical_z};
}

GroundSurfaceFrame ground_surface_frame(const TerrainMetrics3D& metrics,
                                        int i, int j)
{
    validate_geometry_shape(metrics);
    const int nr = metrics.J.size_r();
    const int nth = metrics.J.size_th();
    if (i < 0 || i >= nr || j < 0 || j >= nth)
    {
        throw std::out_of_range(
            "terrain ground surface frame requires a mapped scalar column");
    }

    const double slope_x = -metrics.Gx_face(i, j, 0);
    const double slope_y = -metrics.Gy_face(i, j, 0);
    if (!std::isfinite(slope_x) || !std::isfinite(slope_y))
    {
        throw std::invalid_argument(
            "terrain ground surface slopes must be finite");
    }
    const double area_ratio = std::hypot(std::hypot(slope_x, slope_y), 1.0);
    if (!std::isfinite(area_ratio) || area_ratio < 1.0)
    {
        throw std::invalid_argument(
            "terrain ground surface area ratio must be finite and positive");
    }

    return {slope_x, slope_y,
            -slope_x / area_ratio,
            -slope_y / area_ratio,
            1.0 / area_ratio,
            area_ratio};
}

PhysicalVector3 ground_tangent_velocity(const GroundSurfaceFrame& frame,
                                        double physical_u,
                                        double physical_v)
{
    require_ground_surface_frame(frame);
    if (!std::isfinite(physical_u) || !std::isfinite(physical_v))
        throw std::invalid_argument("terrain ground velocity must be finite");
    return {physical_u, physical_v,
            physical_u * frame.slope_x + physical_v * frame.slope_y};
}

PhysicalVector3 project_to_ground_tangent(const GroundSurfaceFrame& frame,
                                          const PhysicalVector3& vector)
{
    require_ground_surface_frame(frame);
    if (!std::isfinite(vector.x) || !std::isfinite(vector.y)
        || !std::isfinite(vector.z))
    {
        throw std::invalid_argument("terrain surface vector must be finite");
    }
    const double normal_component = vector.x * frame.normal_x
                                  + vector.y * frame.normal_y
                                  + vector.z * frame.normal_z;
    return {vector.x - normal_component * frame.normal_x,
            vector.y - normal_component * frame.normal_y,
            vector.z - normal_component * frame.normal_z};
}

double mapped_ground_normal_flux(const GroundSurfaceFrame& frame,
                                 const PhysicalVector3& physical_flux)
{
    require_ground_surface_frame(frame);
    if (!std::isfinite(physical_flux.x) || !std::isfinite(physical_flux.y)
        || !std::isfinite(physical_flux.z))
    {
        throw std::invalid_argument("terrain ground flux must be finite");
    }
    return physical_flux.z
         - frame.slope_x * physical_flux.x
         - frame.slope_y * physical_flux.y;
}

PhysicalVector3 area_weighted_ground_traction(
    const GroundSurfaceFrame& frame,
    const PhysicalVector3& physical_traction)
{
    require_ground_surface_frame(frame);
    if (!std::isfinite(physical_traction.x)
        || !std::isfinite(physical_traction.y)
        || !std::isfinite(physical_traction.z))
    {
        throw std::invalid_argument("terrain ground traction must be finite");
    }
    return {frame.area_ratio * physical_traction.x,
            frame.area_ratio * physical_traction.y,
            frame.area_ratio * physical_traction.z};
}

PhysicalVector3 contravariant_stress_flux_x(
    double face_jacobian,
    const SymmetricStressTensor& physical_stress)
{
    require_physical_stress(physical_stress);
    if (!std::isfinite(face_jacobian)
        || face_jacobian <= terrain_constants::epsilon)
    {
        throw std::invalid_argument(
            "terrain x-face stress Jacobian must be positive and finite");
    }
    return {face_jacobian * physical_stress.xx,
            face_jacobian * physical_stress.xy,
            face_jacobian * physical_stress.xz};
}

PhysicalVector3 contravariant_stress_flux_y(
    double face_jacobian,
    const SymmetricStressTensor& physical_stress)
{
    require_physical_stress(physical_stress);
    if (!std::isfinite(face_jacobian)
        || face_jacobian <= terrain_constants::epsilon)
    {
        throw std::invalid_argument(
            "terrain y-face stress Jacobian must be positive and finite");
    }
    return {face_jacobian * physical_stress.xy,
            face_jacobian * physical_stress.yy,
            face_jacobian * physical_stress.yz};
}

PhysicalVector3 contravariant_stress_flux_zeta(
    double metric_gx,
    double metric_gy,
    const SymmetricStressTensor& physical_stress)
{
    require_physical_stress(physical_stress);
    if (!std::isfinite(metric_gx) || !std::isfinite(metric_gy))
    {
        throw std::invalid_argument(
            "terrain zeta-face stress metrics must be finite");
    }
    return {
        physical_stress.xz + metric_gx * physical_stress.xx
                           + metric_gy * physical_stress.xy,
        physical_stress.yz + metric_gx * physical_stress.xy
                           + metric_gy * physical_stress.yy,
        physical_stress.zz + metric_gx * physical_stress.xz
                           + metric_gy * physical_stress.yz};
}

PhysicalVector3 mapped_ground_stress_traction(
    const GroundSurfaceFrame& frame,
    const SymmetricStressTensor& physical_stress)
{
    require_ground_surface_frame(frame);
    // T dot (-dh/dx,-dh/dy,1). This is area_ratio*(T dot n), evaluated
    // without a normalization round-trip so it is the exact finite-volume
    // lower-face flux counterpart of the interior mapped tensor rows.
    return contravariant_stress_flux_zeta(
        -frame.slope_x, -frame.slope_y, physical_stress);
}

void validate_geometry_shape(const TerrainMetrics3D& metrics)
{
    const int nr = metrics.J.size_r();
    const int nth = metrics.J.size_th();
    const int nz = metrics.J.size_z();
    if (nr <= 0 || nth <= 0 || nz <= 0)
    {
        throw std::invalid_argument("terrain conservative geometry must be non-empty");
    }
    if (!std::isfinite(metrics.dx) || metrics.dx <= 0.0
        || !std::isfinite(metrics.dy) || metrics.dy <= 0.0
        || !std::isfinite(metrics.dzeta) || metrics.dzeta <= 0.0)
    {
        throw std::invalid_argument(
            "terrain computational spacings must be positive and finite");
    }

    require_shape(metrics.z, nr, nth, nz, "terrain physical height");
    require_shape(metrics.mx, nr, nth, nz, "terrain x point metric");
    require_shape(metrics.my, nr, nth, nz, "terrain y point metric");
    require_shape(metrics.zeta, nr, nth, nz, "terrain computational height");
    require_shape(metrics.z_face, nr, nth, nz + 1, "terrain z-face height");
    require_shape(metrics.Jx_face, nr + 1, nth, nz, "terrain x-face Jacobian");
    require_shape(metrics.Jy_face, nr, nth + 1, nz, "terrain y-face Jacobian");
    require_shape(metrics.Gx_face, nr, nth, nz + 1, "terrain z-face x metric");
    require_shape(metrics.Gy_face, nr, nth, nz + 1, "terrain z-face y metric");

}

void validate_geometry(const TerrainMetrics3D& metrics)
{
    validate_geometry_shape(metrics);
    require_finite(metrics.z, "terrain physical height");
    require_finite(metrics.mx, "terrain x point metric");
    require_finite(metrics.my, "terrain y point metric");
    require_finite(metrics.zeta, "terrain computational height");
    require_finite(metrics.z_face, "terrain z-face height");
    require_finite(metrics.Jx_face, "terrain x-face Jacobian");
    require_finite(metrics.Jy_face, "terrain y-face Jacobian");
    require_finite(metrics.Gx_face, "terrain z-face x metric");
    require_finite(metrics.Gy_face, "terrain z-face y metric");
    for (double jacobian : metrics.J.values)
    {
        if (!std::isfinite(jacobian) || jacobian <= terrain_constants::epsilon)
        {
            throw std::invalid_argument(
                "terrain cell Jacobian must be positive and finite");
        }
    }
    for (double jacobian : metrics.Jx_face.values)
    {
        if (jacobian <= terrain_constants::epsilon)
            throw std::invalid_argument("terrain x-face Jacobian must be positive");
    }
    for (double jacobian : metrics.Jy_face.values)
    {
        if (jacobian <= terrain_constants::epsilon)
            throw std::invalid_argument("terrain y-face Jacobian must be positive");
    }
}

TerrainContravariantFluxes transform_fluxes(
    const TerrainMetrics3D& metrics,
    const TerrainPhysicalFluxes& physical)
{
    validate_geometry(metrics);
    const int nr = metrics.J.size_r();
    const int nth = metrics.J.size_th();
    const int nz = metrics.J.size_z();
    require_shape(physical.x, nr + 1, nth, nz, "physical x flux");
    require_shape(physical.y, nr, nth + 1, nz, "physical y flux");
    require_shape(physical.z_x, nr, nth, nz + 1, "z-face physical x flux");
    require_shape(physical.z_y, nr, nth, nz + 1, "z-face physical y flux");
    require_shape(physical.z_vertical, nr, nth, nz + 1,
                  "z-face physical vertical flux");
    require_finite(physical.x, "physical x flux");
    require_finite(physical.y, "physical y flux");
    require_finite(physical.z_x, "z-face physical x flux");
    require_finite(physical.z_y, "z-face physical y flux");
    require_finite(physical.z_vertical, "z-face physical vertical flux");

    TerrainContravariantFluxes result;
    result.x.resize(nr + 1, nth, nz, 0.0);
    result.y.resize(nr, nth + 1, nz, 0.0);
    result.z.resize(nr, nth, nz + 1, 0.0);

    for (int i = 0; i <= nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                result.x(i, j, k) = metrics.Jx_face(i, j, k)
                    * physical.x(i, j, k);

    for (int i = 0; i < nr; ++i)
        for (int j = 0; j <= nth; ++j)
            for (int k = 0; k < nz; ++k)
                result.y(i, j, k) = metrics.Jy_face(i, j, k)
                    * physical.y(i, j, k);

    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k <= nz; ++k)
                result.z(i, j, k) = physical.z_vertical(i, j, k)
                    + metrics.Gx_face(i, j, k) * physical.z_x(i, j, k)
                    + metrics.Gy_face(i, j, k) * physical.z_y(i, j, k);

    return result;
}

TerrainField3D conservative_divergence(
    const TerrainMetrics3D& metrics,
    const TerrainContravariantFluxes& fluxes)
{
    validate_geometry(metrics);
    const int nr = metrics.J.size_r();
    const int nth = metrics.J.size_th();
    const int nz = metrics.J.size_z();
    require_shape(fluxes.x, nr + 1, nth, nz, "contravariant x flux");
    require_shape(fluxes.y, nr, nth + 1, nz, "contravariant y flux");
    require_shape(fluxes.z, nr, nth, nz + 1, "contravariant zeta flux");
    require_finite(fluxes.x, "contravariant x flux");
    require_finite(fluxes.y, "contravariant y flux");
    require_finite(fluxes.z, "contravariant zeta flux");

    TerrainField3D divergence;
    divergence.resize(nr, nth, nz, 0.0);
    const double inv_dx = 1.0 / metrics.dx;
    const double inv_dy = 1.0 / metrics.dy;
    const double inv_dzeta = 1.0 / metrics.dzeta;
    for (int i = 0; i < nr; ++i)
    {
        for (int j = 0; j < nth; ++j)
        {
            for (int k = 0; k < nz; ++k)
            {
                const double transformed_divergence =
                    (fluxes.x(i + 1, j, k) - fluxes.x(i, j, k)) * inv_dx
                    + (fluxes.y(i, j + 1, k) - fluxes.y(i, j, k)) * inv_dy
                    + (fluxes.z(i, j, k + 1) - fluxes.z(i, j, k))
                        * inv_dzeta;
                divergence(i, j, k) = transformed_divergence
                    / metrics.J(i, j, k);
            }
        }
    }
    return divergence;
}

void validate_cgrid_state(const TerrainMetrics3D& metrics,
                          const Field3D& rho,
                          const Field3D& u,
                          const Field3D& v,
                          const Field3D& w)
{
    validate_geometry(metrics);
    validate_cgrid_shapes(metrics, rho, u, v, w);
    const int nr = rho.size_r();
    const int nth = rho.size_th();
    const int nz = rho.size_z();
    for (int i = 1; i < nr - 1; ++i)
        for (int j = 1; j < nth - 1; ++j)
            for (int k = 0; k < nz; ++k)
            {
                if (!std::isfinite(rho[i][j][k]) || rho[i][j][k] <= 0.0f
                    || !std::isfinite(w[i][j][k]))
                    throw std::invalid_argument(
                        "terrain C-grid scalar/vertical owner is invalid at ("
                        + std::to_string(i) + "," + std::to_string(j)
                        + "," + std::to_string(k) + ")");
            }
    for (int face = 0; face < nr - 1; ++face)
        for (int j = 1; j < nth - 1; ++j)
            for (int k = 0; k < nz; ++k)
                if (!std::isfinite(u[face][j][k]))
                    throw std::invalid_argument(
                        "terrain C-grid x-face owner is invalid at ("
                        + std::to_string(face) + "," + std::to_string(j)
                        + "," + std::to_string(k) + ")");
    for (int i = 1; i < nr - 1; ++i)
        for (int face = 0; face < nth - 1; ++face)
            for (int k = 0; k < nz; ++k)
                if (!std::isfinite(v[i][face][k]))
                    throw std::invalid_argument(
                        "terrain C-grid y-face owner is invalid at ("
                        + std::to_string(i) + "," + std::to_string(face)
                        + "," + std::to_string(k) + ")");
}

void validate_cgrid_shapes(const TerrainMetrics3D& metrics,
                           const Field3D& rho,
                           const Field3D& u,
                           const Field3D& v,
                           const Field3D& w)
{
    validate_geometry_shape(metrics);
    const int nr = metrics.J.size_r();
    const int nth = metrics.J.size_th();
    const int nz = metrics.J.size_z();
    const auto matches = [nr, nth, nz](const Field3D& field)
    {
        return field.size_r() == nr && field.size_th() == nth
            && field.size_z() == nz;
    };
    if (!matches(rho) || !matches(u) || !matches(v) || !matches(w))
        throw std::invalid_argument(
            "terrain C-grid state fields must match the conservative geometry");
    if (nr < 3 || nth < 3 || nz < 1)
        throw std::invalid_argument(
            "terrain C-grid transport requires horizontal boundary storage");
}

double cgrid_mass_flux_x(const TerrainMetrics3D& metrics,
                         const Field3D& rho, const Field3D& u,
                         int i, int j, int k, double frame_u_ms)
{
    if (!std::isfinite(frame_u_ms))
        throw std::invalid_argument("terrain x-frame velocity must be finite");
    return metrics.Jx_face(i + 1, j, k) * rho_x_face(rho, i, j, k)
        * (static_cast<double>(u[i][j][k]) - frame_u_ms);
}

double cgrid_mass_flux_y(const TerrainMetrics3D& metrics,
                         const Field3D& rho, const Field3D& v,
                         int i, int j, int k, double frame_v_ms)
{
    if (!std::isfinite(frame_v_ms))
        throw std::invalid_argument("terrain y-frame velocity must be finite");
    return metrics.Jy_face(i, j + 1, k) * rho_y_face(rho, i, j, k)
        * (static_cast<double>(v[i][j][k]) - frame_v_ms);
}

double cgrid_mass_flux_zeta(const TerrainMetrics3D& metrics,
                            const Field3D& rho,
                            const Field3D& u,
                            const Field3D& v,
                            const Field3D& w,
                            int i, int j, int k,
                            double frame_u_ms, double frame_v_ms)
{
    if (!std::isfinite(frame_u_ms) || !std::isfinite(frame_v_ms))
        throw std::invalid_argument("terrain frame velocity must be finite");
    const int nz = rho.size_z();
    const double rho_w = rho_zeta_face(rho, i, j, k)
        * static_cast<double>(w[i][j][k]);
    const double rho_u = interpolate_to_zeta_face(
        [&](int level)
        {
            return rho_u_at_cell_center(
                rho, u, i, j, level, frame_u_ms);
        }, k, nz);
    const double rho_v = interpolate_to_zeta_face(
        [&](int level)
        {
            return rho_v_at_cell_center(
                rho, v, i, j, level, frame_v_ms);
        }, k, nz);
    const int face_k = k + 1;
    return rho_w + metrics.Gx_face(i, j, face_k) * rho_u
                 + metrics.Gy_face(i, j, face_k) * rho_v;
}

double cgrid_mass_divergence(const TerrainMetrics3D& metrics,
                             const Field3D& rho,
                             const Field3D& u,
                             const Field3D& v,
                             const Field3D& w,
                             int i, int j, int k,
                             double frame_u_ms, double frame_v_ms)
{
    const double x_positive = cgrid_mass_flux_x(
        metrics, rho, u, i, j, k, frame_u_ms);
    const double x_negative = cgrid_mass_flux_x(
        metrics, rho, u, i - 1, j, k, frame_u_ms);
    const double y_positive = cgrid_mass_flux_y(
        metrics, rho, v, i, j, k, frame_v_ms);
    const double y_negative = cgrid_mass_flux_y(
        metrics, rho, v, i, j - 1, k, frame_v_ms);
    const double z_positive = cgrid_mass_flux_zeta(
        metrics, rho, u, v, w, i, j, k, frame_u_ms, frame_v_ms);
    const double z_negative = (k > 0)
        ? cgrid_mass_flux_zeta(
            metrics, rho, u, v, w, i, j, k - 1,
            frame_u_ms, frame_v_ms)
        : 0.0;
    const double transformed = (x_positive - x_negative) / metrics.dx
        + (y_positive - y_negative) / metrics.dy
        + (z_positive - z_negative) / metrics.dzeta;
    return transformed / metrics.J(i, j, k);
}

double cgrid_velocity_divergence(const TerrainMetrics3D& metrics,
                                 const Field3D& u,
                                 const Field3D& v,
                                 const Field3D& w,
                                 int i, int j, int k,
                                 double frame_u_ms, double frame_v_ms)
{
    if (!std::isfinite(frame_u_ms) || !std::isfinite(frame_v_ms))
        throw std::invalid_argument("terrain frame velocity must be finite");
    const double x_positive = metrics.Jx_face(i + 1, j, k)
        * (static_cast<double>(u[i][j][k]) - frame_u_ms);
    const double x_negative = metrics.Jx_face(i, j, k)
        * (static_cast<double>(u[i - 1][j][k]) - frame_u_ms);
    const double y_positive = metrics.Jy_face(i, j + 1, k)
        * (static_cast<double>(v[i][j][k]) - frame_v_ms);
    const double y_negative = metrics.Jy_face(i, j, k)
        * (static_cast<double>(v[i][j - 1][k]) - frame_v_ms);
    const int nz = w.size_z();
    const auto zeta_flux = [&](int face_owner_k)
    {
        const double u_face = interpolate_to_zeta_face(
            [&](int level) { return u_at_cell_center(u, i, j, level); },
            face_owner_k, nz) - frame_u_ms;
        const double v_face = interpolate_to_zeta_face(
            [&](int level) { return v_at_cell_center(v, i, j, level); },
            face_owner_k, nz) - frame_v_ms;
        const int face_k = face_owner_k + 1;
        return static_cast<double>(w[i][j][face_owner_k])
            + metrics.Gx_face(i, j, face_k) * u_face
            + metrics.Gy_face(i, j, face_k) * v_face;
    };
    const double z_positive = zeta_flux(k);
    const double z_negative = (k > 0) ? zeta_flux(k - 1) : 0.0;
    const double transformed = (x_positive - x_negative) / metrics.dx
        + (y_positive - y_negative) / metrics.dy
        + (z_positive - z_negative) / metrics.dzeta;
    return transformed / metrics.J(i, j, k);
}

namespace
{
bool closed_lateral(tmv::init::config::LateralBC boundary)
{
    using tmv::init::config::LateralBC;
    return boundary == LateralBC::Rigid || boundary == LateralBC::RigidSlip;
}

int periodic_neighbor(int index, int offset, int extent, bool periodic)
{
    const int candidate = index + offset;
    if (!periodic) return candidate;
    if (candidate < 1) return extent - 2;
    if (candidate > extent - 2) return 1;
    return candidate;
}

template <typename ValueAt>
PhysicalVector3 agrid_gradient_center_impl(
    const TerrainMetrics3D& metrics,
    const ValueAt& value_at,
    int i, int j, int k,
    bool periodic_x,
    bool periodic_y)
{
    const int west = periodic_neighbor(
        i, -1, metrics.J.size_r(), periodic_x);
    const int east = periodic_neighbor(
        i, 1, metrics.J.size_r(), periodic_x);
    const int south = periodic_neighbor(
        j, -1, metrics.J.size_th(), periodic_y);
    const int north = periodic_neighbor(
        j, 1, metrics.J.size_th(), periodic_y);
    const double derivative_x =
        (value_at(east, j, k) - value_at(west, j, k))
        / (2.0 * metrics.dx);
    const double derivative_y =
        (value_at(i, north, k) - value_at(i, south, k))
        / (2.0 * metrics.dy);
    const double derivative_zeta =
        (value_at(i, j, k + 1) - value_at(i, j, k - 1))
        / (2.0 * metrics.dzeta);
    const double jacobian = metrics.J(i, j, k);
    return physical_gradient_from_computational(
        derivative_x, derivative_y, derivative_zeta, jacobian,
        jacobian * metrics.mx(i, j, k),
        jacobian * metrics.my(i, j, k));
}

void require_agrid_pressure_coupling(
    const mass_flux_detail::CollocatedPressureCouplingView& coupling,
    const TerrainMetrics3D& metrics)
{
    if (coupling.pressure == nullptr
        || coupling.reference_pressure == nullptr
        || coupling.pressure->size_r() != metrics.J.size_r()
        || coupling.pressure->size_th() != metrics.J.size_th()
        || coupling.pressure->size_z() != metrics.J.size_z()
        || coupling.reference_pressure->size()
            != static_cast<std::size_t>(metrics.J.size_z())
        || !std::isfinite(coupling.dt) || coupling.dt <= 0.0)
    {
        throw std::invalid_argument(
            "mapped A-grid pressure coupling is incomplete");
    }
}

double pressure_perturbation_at(
    const mass_flux_detail::CollocatedPressureCouplingView& coupling,
    const TerrainMetrics3D& metrics,
    int i, int j, int k)
{
    return static_cast<double>((*coupling.pressure)[i][j][k])
        - reference_profile_value_fp32(
            *coupling.reference_pressure, metrics.z(i, j, k),
            metrics.dzeta);
}

PhysicalVector3 pressure_gradient_center(
    const mass_flux_detail::CollocatedPressureCouplingView& coupling,
    const TerrainMetrics3D& metrics,
    int i, int j, int k,
    bool periodic_x,
    bool periodic_y)
{
    return agrid_gradient_center_impl(
        metrics,
        [&](int ii, int jj, int kk)
        {
            return pressure_perturbation_at(
                coupling, metrics, ii, jj, kk);
        },
        i, j, k, periodic_x, periodic_y);
}
} // namespace

PhysicalVector3 agrid_scalar_gradient_center(
    const TerrainMetrics3D& metrics,
    const Field3D& scalar,
    int i, int j, int k,
    bool periodic_x,
    bool periodic_y)
{
    validate_geometry_shape(metrics);
    require_scalar_shape(metrics, scalar, "mapped A-grid scalar");
    return agrid_gradient_center_impl(
        metrics,
        [&](int ii, int jj, int kk)
        {
            return static_cast<double>(scalar[ii][jj][kk]);
        },
        i, j, k, periodic_x, periodic_y);
}

PhysicalVector3 agrid_pressure_perturbation_gradient_center(
    const TerrainMetrics3D& metrics,
    const Field3D& pressure,
    const std::vector<double>& reference_pressure,
    int i, int j, int k,
    bool periodic_x,
    bool periodic_y)
{
    const mass_flux_detail::CollocatedPressureCouplingView coupling{
        &pressure, &reference_pressure, 1.0};
    require_agrid_pressure_coupling(coupling, metrics);
    return pressure_gradient_center(
        coupling, metrics, i, j, k, periodic_x, periodic_y);
}

double agrid_mass_flux_x(
    const TerrainMetrics3D& metrics,
    const Field3D& rho,
    const Field3D& u,
    int i, int j, int k,
    tmv::init::config::LateralBC west,
    tmv::init::config::LateralBC east,
    const mass_flux_detail::CollocatedPressureCouplingView* pressure_coupling,
    double frame_u_ms)
{
    using tmv::init::config::LateralBC;
    if (!std::isfinite(frame_u_ms))
        throw std::invalid_argument("mapped A-grid x-frame velocity must be finite");
    const int nx = metrics.J.size_r();
    if ((i == 0 && closed_lateral(west))
        || (i == nx - 2 && closed_lateral(east)))
        return 0.0;
    const bool periodic =
        (i == 0 && west == LateralBC::Periodic)
        || (i == nx - 2 && east == LateralBC::Periodic);
    const int lo = periodic ? nx - 2 : i;
    const int hi = periodic ? 1 : i + 1;
    const double density = 0.5 * (
        static_cast<double>(rho[lo][j][k])
        + static_cast<double>(rho[hi][j][k]));
    double velocity = 0.5 * (
        static_cast<double>(u[lo][j][k])
        + static_cast<double>(u[hi][j][k])) - frame_u_ms;
    if (pressure_coupling != nullptr && j > 0
        && j + 1 < metrics.J.size_th() && k > 0
        && k + 1 < metrics.J.size_z())
    {
        require_agrid_pressure_coupling(*pressure_coupling, metrics);
        const auto gradient_lo = pressure_gradient_center(
            *pressure_coupling, metrics, lo, j, k, periodic, false);
        const auto gradient_hi = pressure_gradient_center(
            *pressure_coupling, metrics, hi, j, k, periodic, false);
        const double rho_lo = static_cast<double>(rho[lo][j][k]);
        const double rho_hi = static_cast<double>(rho[hi][j][k]);
        const double vertical_gradient = 0.5 * (
            gradient_lo.z + gradient_hi.z);
        const double slope =
            (metrics.z(hi, j, k) - metrics.z(lo, j, k)) / metrics.dx;
        const double face_gradient =
            (pressure_perturbation_at(
                 *pressure_coupling, metrics, hi, j, k)
             - pressure_perturbation_at(
                 *pressure_coupling, metrics, lo, j, k)) / metrics.dx
            - slope * vertical_gradient;
        const double cells = -0.5 * (
            gradient_lo.x / rho_lo + gradient_hi.x / rho_hi);
        const double face = -face_gradient / density;
        velocity += pressure_coupling->dt * (face - cells);
    }
    const double jacobian = periodic
        ? 0.5 * (metrics.J(lo, j, k) + metrics.J(hi, j, k))
        : metrics.Jx_face(i + 1, j, k);
    return jacobian * density * velocity;
}

double agrid_mass_flux_y(
    const TerrainMetrics3D& metrics,
    const Field3D& rho,
    const Field3D& v,
    int i, int j, int k,
    tmv::init::config::LateralBC south,
    tmv::init::config::LateralBC north,
    const mass_flux_detail::CollocatedPressureCouplingView* pressure_coupling,
    double frame_v_ms)
{
    using tmv::init::config::LateralBC;
    if (!std::isfinite(frame_v_ms))
        throw std::invalid_argument("mapped A-grid y-frame velocity must be finite");
    const int ny = metrics.J.size_th();
    if ((j == 0 && closed_lateral(south))
        || (j == ny - 2 && closed_lateral(north)))
        return 0.0;
    const bool periodic =
        (j == 0 && south == LateralBC::Periodic)
        || (j == ny - 2 && north == LateralBC::Periodic);
    const int lo = periodic ? ny - 2 : j;
    const int hi = periodic ? 1 : j + 1;
    const double density = 0.5 * (
        static_cast<double>(rho[i][lo][k])
        + static_cast<double>(rho[i][hi][k]));
    double velocity = 0.5 * (
        static_cast<double>(v[i][lo][k])
        + static_cast<double>(v[i][hi][k])) - frame_v_ms;
    if (pressure_coupling != nullptr && i > 0
        && i + 1 < metrics.J.size_r() && k > 0
        && k + 1 < metrics.J.size_z())
    {
        require_agrid_pressure_coupling(*pressure_coupling, metrics);
        const auto gradient_lo = pressure_gradient_center(
            *pressure_coupling, metrics, i, lo, k, false, periodic);
        const auto gradient_hi = pressure_gradient_center(
            *pressure_coupling, metrics, i, hi, k, false, periodic);
        const double rho_lo = static_cast<double>(rho[i][lo][k]);
        const double rho_hi = static_cast<double>(rho[i][hi][k]);
        const double vertical_gradient = 0.5 * (
            gradient_lo.z + gradient_hi.z);
        const double slope =
            (metrics.z(i, hi, k) - metrics.z(i, lo, k)) / metrics.dy;
        const double face_gradient =
            (pressure_perturbation_at(
                 *pressure_coupling, metrics, i, hi, k)
             - pressure_perturbation_at(
                 *pressure_coupling, metrics, i, lo, k)) / metrics.dy
            - slope * vertical_gradient;
        const double cells = -0.5 * (
            gradient_lo.y / rho_lo + gradient_hi.y / rho_hi);
        const double face = -face_gradient / density;
        velocity += pressure_coupling->dt * (face - cells);
    }
    const double jacobian = periodic
        ? 0.5 * (metrics.J(i, lo, k) + metrics.J(i, hi, k))
        : metrics.Jy_face(i, j + 1, k);
    return jacobian * density * velocity;
}

double agrid_mass_flux_zeta(
    const TerrainMetrics3D& metrics,
    const Field3D& rho,
    const Field3D& u,
    const Field3D& v,
    const Field3D& w,
    int i, int j, int k,
    tmv::init::config::BottomBC bottom,
    tmv::init::config::TopBC top,
    const mass_flux_detail::CollocatedPressureCouplingView* pressure_coupling,
    double frame_u_ms,
    double frame_v_ms)
{
    using tmv::init::config::TopBC;
    (void)bottom;
    const int nz = metrics.J.size_z();
    if (k == 0 || (k == nz - 2 && top != TopBC::Open)) return 0.0;
    const int lo = k;
    const int hi = k + 1;
    const double density = 0.5 * (
        static_cast<double>(rho[i][j][lo])
        + static_cast<double>(rho[i][j][hi]));
    const double metric_x = metrics.Gx_face(i, j, k + 1);
    const double metric_y = metrics.Gy_face(i, j, k + 1);
    double normal_velocity = 0.5 * (
        static_cast<double>(w[i][j][lo])
        + static_cast<double>(w[i][j][hi]))
        + metric_x * (0.5 * (
            static_cast<double>(u[i][j][lo])
            + static_cast<double>(u[i][j][hi])) - frame_u_ms)
        + metric_y * (0.5 * (
            static_cast<double>(v[i][j][lo])
            + static_cast<double>(v[i][j][hi])) - frame_v_ms);
    if (pressure_coupling != nullptr && lo > 0 && hi + 1 < nz)
    {
        require_agrid_pressure_coupling(*pressure_coupling, metrics);
        const auto gradient_lo = pressure_gradient_center(
            *pressure_coupling, metrics, i, j, lo, false, false);
        const auto gradient_hi = pressure_gradient_center(
            *pressure_coupling, metrics, i, j, hi, false, false);
        const double rho_lo = static_cast<double>(rho[i][j][lo]);
        const double rho_hi = static_cast<double>(rho[i][j][hi]);
        const double face_vertical =
            (pressure_perturbation_at(
                 *pressure_coupling, metrics, i, j, hi)
             - pressure_perturbation_at(
                 *pressure_coupling, metrics, i, j, lo))
            / (metrics.z(i, j, hi) - metrics.z(i, j, lo));
        const double face_x = 0.5 * (gradient_lo.x + gradient_hi.x);
        const double face_y = 0.5 * (gradient_lo.y + gradient_hi.y);
        const double direct = -(
            face_vertical + metric_x * face_x + metric_y * face_y)
            / density;
        const double cells = -0.5 * (
            (gradient_lo.z + metric_x * gradient_lo.x
             + metric_y * gradient_lo.y) / rho_lo
          + (gradient_hi.z + metric_x * gradient_hi.x
             + metric_y * gradient_hi.y) / rho_hi);
        normal_velocity += pressure_coupling->dt * (direct - cells);
    }
    return density * normal_velocity;
}

double agrid_mass_divergence(
    const TerrainMetrics3D& metrics,
    const Field3D& rho,
    const Field3D& u,
    const Field3D& v,
    const Field3D& w,
    int i, int j, int k,
    tmv::init::config::LateralBC west,
    tmv::init::config::LateralBC east,
    tmv::init::config::LateralBC south,
    tmv::init::config::LateralBC north,
    tmv::init::config::BottomBC bottom,
    tmv::init::config::TopBC top,
    const mass_flux_detail::CollocatedPressureCouplingView* pressure_coupling,
    double frame_u_ms,
    double frame_v_ms)
{
    const double transformed =
        (agrid_mass_flux_x(
             metrics, rho, u, i, j, k, west, east,
             pressure_coupling, frame_u_ms)
         - agrid_mass_flux_x(
             metrics, rho, u, i - 1, j, k, west, east,
             pressure_coupling, frame_u_ms)) / metrics.dx
      + (agrid_mass_flux_y(
             metrics, rho, v, i, j, k, south, north,
             pressure_coupling, frame_v_ms)
         - agrid_mass_flux_y(
             metrics, rho, v, i, j - 1, k, south, north,
             pressure_coupling, frame_v_ms)) / metrics.dy
      + (agrid_mass_flux_zeta(
             metrics, rho, u, v, w, i, j, k, bottom, top,
             pressure_coupling, frame_u_ms, frame_v_ms)
         - agrid_mass_flux_zeta(
             metrics, rho, u, v, w, i, j, k - 1, bottom, top,
             pressure_coupling, frame_u_ms, frame_v_ms)) / metrics.dzeta;
    return transformed / metrics.J(i, j, k);
}

double agrid_velocity_divergence(
    const TerrainMetrics3D& metrics,
    const Field3D& u,
    const Field3D& v,
    const Field3D& w,
    int i, int j, int k,
    tmv::init::config::LateralBC west,
    tmv::init::config::LateralBC east,
    tmv::init::config::LateralBC south,
    tmv::init::config::LateralBC north,
    tmv::init::config::BottomBC bottom,
    tmv::init::config::TopBC top,
    double frame_u_ms,
    double frame_v_ms)
{
    using tmv::init::config::LateralBC;
    using tmv::init::config::TopBC;
    const int nx = metrics.J.size_r();
    const int ny = metrics.J.size_th();
    const int nz = metrics.J.size_z();
    const auto x_flux = [&](int owner)
    {
        if ((owner == 0 && closed_lateral(west))
            || (owner == nx - 2 && closed_lateral(east)))
            return 0.0;
        const bool periodic =
            (owner == 0 && west == LateralBC::Periodic)
            || (owner == nx - 2 && east == LateralBC::Periodic);
        const int lo = periodic ? nx - 2 : owner;
        const int hi = periodic ? 1 : owner + 1;
        const double jacobian = periodic
            ? 0.5 * (metrics.J(lo, j, k) + metrics.J(hi, j, k))
            : metrics.Jx_face(owner + 1, j, k);
        return jacobian * (0.5 * (
            static_cast<double>(u[lo][j][k])
            + static_cast<double>(u[hi][j][k])) - frame_u_ms);
    };
    const auto y_flux = [&](int owner)
    {
        if ((owner == 0 && closed_lateral(south))
            || (owner == ny - 2 && closed_lateral(north)))
            return 0.0;
        const bool periodic =
            (owner == 0 && south == LateralBC::Periodic)
            || (owner == ny - 2 && north == LateralBC::Periodic);
        const int lo = periodic ? ny - 2 : owner;
        const int hi = periodic ? 1 : owner + 1;
        const double jacobian = periodic
            ? 0.5 * (metrics.J(i, lo, k) + metrics.J(i, hi, k))
            : metrics.Jy_face(i, owner + 1, k);
        return jacobian * (0.5 * (
            static_cast<double>(v[i][lo][k])
            + static_cast<double>(v[i][hi][k])) - frame_v_ms);
    };
    const auto z_flux = [&](int owner)
    {
        (void)bottom;
        if (owner == 0 || (owner == nz - 2 && top != TopBC::Open))
            return 0.0;
        const int lo = owner;
        const int hi = owner + 1;
        return 0.5 * (
                   static_cast<double>(w[i][j][lo])
                   + static_cast<double>(w[i][j][hi]))
            + metrics.Gx_face(i, j, owner + 1) * (0.5 * (
                   static_cast<double>(u[i][j][lo])
                   + static_cast<double>(u[i][j][hi])) - frame_u_ms)
            + metrics.Gy_face(i, j, owner + 1) * (0.5 * (
                   static_cast<double>(v[i][j][lo])
                   + static_cast<double>(v[i][j][hi])) - frame_v_ms);
    };
    const double transformed = (x_flux(i) - x_flux(i - 1)) / metrics.dx
        + (y_flux(j) - y_flux(j - 1)) / metrics.dy
        + (z_flux(k) - z_flux(k - 1)) / metrics.dzeta;
    return transformed / metrics.J(i, j, k);
}

double reference_profile_value_fp32(const std::vector<double>& profile,
                                    double physical_height,
                                    double nominal_dz)
{
    if (profile.empty())
        throw std::invalid_argument("terrain reference profile must not be empty");
    if (!std::isfinite(physical_height) || !std::isfinite(nominal_dz)
        || nominal_dz <= 0.0)
    {
        throw std::invalid_argument(
            "terrain reference sampling requires finite height and positive spacing");
    }

    if (profile.size() == 1)
    {
        if (!std::isfinite(profile.front()) || !std::isfinite(static_cast<float>(profile.front())))
            throw std::invalid_argument("terrain reference profile must be finite");
        return static_cast<double>(static_cast<float>(profile.front()));
    }

    // Scalar reference entries live at (k+1/2)*dz.  Continue the first/last
    // segment outside that centre range; terrain columns can legitimately put
    // their highest centre above the flat column's highest centre.
    const double fractional_index = physical_height / nominal_dz - 0.5;
    if (!std::isfinite(fractional_index))
        throw std::invalid_argument("terrain reference interpolation index overflowed");
    const auto lower = static_cast<std::size_t>(std::clamp(
        std::floor(fractional_index), 0.0, static_cast<double>(profile.size() - 2)));
    const double weight = fractional_index - static_cast<double>(lower);
    const double lower_value = profile[static_cast<std::size_t>(lower)];
    const double upper_value = profile[static_cast<std::size_t>(lower + 1)];
    if (!std::isfinite(lower_value) || !std::isfinite(upper_value))
        throw std::invalid_argument("terrain reference profile must be finite");
    const double sampled = lower_value + weight * (upper_value - lower_value);
    if (!std::isfinite(sampled) || !std::isfinite(static_cast<float>(sampled)))
        throw std::invalid_argument("terrain reference interpolation overflowed");
    return static_cast<double>(static_cast<float>(sampled));
}

double reference_density_fp32(const std::vector<double>& reference_pressure,
                              const std::vector<double>& reference_theta,
                              double physical_height,
                              double nominal_dz)
{
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk /
        physical_constants::specific_heat_cp_jkgk;
    const double pressure = reference_profile_value_fp32(
        reference_pressure, physical_height, nominal_dz);
    const double theta = reference_profile_value_fp32(
        reference_theta, physical_height, nominal_dz);
    if (pressure <= 0.0 || theta <= 0.0)
        throw std::invalid_argument(
            "terrain EOS reference requires positive pressure and theta");
    const double temperature = theta * std::pow(
        pressure / physical_constants::reference_pressure_pa, kappa);
    const double density = pressure /
        (physical_constants::gas_constant_dry_air_jkgk * temperature);
    if (!std::isfinite(density) || !std::isfinite(static_cast<float>(density)) || static_cast<float>(density) <= 0.0f)
        throw std::invalid_argument(
            "terrain EOS reference produced invalid density");
    return static_cast<double>(static_cast<float>(density));
}

double cgrid_scalar_gradient_x(const TerrainMetrics3D& metrics,
                               const Field3D& scalar,
                               int i, int j, int k)
{
    validate_geometry_shape(metrics);
    require_scalar_shape(metrics, scalar, "terrain scalar");
    const auto value_at = [&](int ii, int jj, int kk)
    {
        return static_cast<double>(scalar[ii][jj][kk]);
    };
    return physical_gradient_x(metrics, value_at, i, j, k);
}

double cgrid_scalar_gradient_y(const TerrainMetrics3D& metrics,
                               const Field3D& scalar,
                               int i, int j, int k)
{
    validate_geometry_shape(metrics);
    require_scalar_shape(metrics, scalar, "terrain scalar");
    const auto value_at = [&](int ii, int jj, int kk)
    {
        return static_cast<double>(scalar[ii][jj][kk]);
    };
    return physical_gradient_y(metrics, value_at, i, j, k);
}

double cgrid_scalar_gradient_z(const TerrainMetrics3D& metrics,
                               const Field3D& scalar,
                               int i, int j, int k)
{
    validate_geometry_shape(metrics);
    require_scalar_shape(metrics, scalar, "terrain scalar");
    const auto value_at = [&](int ii, int jj, int kk)
    {
        return static_cast<double>(scalar[ii][jj][kk]);
    };
    return physical_gradient_z_face(metrics, value_at, i, j, k);
}

double cgrid_scalar_gradient_z_center(const TerrainMetrics3D& metrics,
                                      const Field3D& scalar,
                                      int i, int j, int k)
{
    validate_geometry_shape(metrics);
    require_scalar_shape(metrics, scalar, "terrain scalar");
    const auto value_at = [&](int ii, int jj, int kk)
    {
        return static_cast<double>(scalar[ii][jj][kk]);
    };
    return column_gradient_z(metrics, value_at, i, j, k);
}

namespace
{
void cgrid_scalar_diffusion_tendency_impl(
    const TerrainMetrics3D& metrics,
    const Field3D& scalar,
    const Field3D& rho,
    const Field3D* K_horizontal,
    const Field3D* K_vertical,
    double K_h,
    double K_v,
    bool use_horizontal,
    bool use_vertical,
    bool periodic_x,
    bool periodic_y,
    Field3D& tendency)
{
    validate_geometry(metrics);
    require_scalar_shape(metrics, scalar, "terrain diffused scalar");
    require_scalar_shape(metrics, rho, "terrain diffusion density");
    if (K_horizontal != nullptr)
        require_scalar_shape(metrics, *K_horizontal,
                             "terrain horizontal diffusivity");
    if (K_vertical != nullptr)
        require_scalar_shape(metrics, *K_vertical,
                             "terrain vertical diffusivity");
    if (!std::isfinite(K_h) || K_h < 0.0 || !std::isfinite(K_v) || K_v < 0.0)
        throw std::invalid_argument(
            "terrain scalar diffusivities must be finite and non-negative");

    const int nr = metrics.J.size_r();
    const int nth = metrics.J.size_th();
    const int nz = metrics.J.size_z();
    if (nr < 3 || nth < 3 || nz < 2)
        throw std::invalid_argument(
            "terrain scalar diffusion requires 3x3 lateral storage and two levels");
    if (tendency.size_r() != nr || tendency.size_th() != nth
        || tendency.size_z() != nz)
        tendency.resize(nr, nth, nz, 0.0f);
    else
        tendency.fill(0.0f);

    const auto value_at = [&](int i, int j, int k)
    { return static_cast<double>(scalar(i, j, k)); };
    const auto rho_at = [&](int i, int j, int k)
    { return static_cast<double>(rho(i, j, k)); };
    const auto kv_at = [&](int i, int j, int k)
    {
        return K_vertical ? static_cast<double>((*K_vertical)(i, j, k)) : K_v;
    };
    const auto kh_at = [&](int i, int j, int k)
    {
        return K_horizontal
            ? static_cast<double>((*K_horizontal)(i, j, k)) : K_h;
    };

    for (int i = 1; i < nr - 1; ++i)
        for (int j = 1; j < nth - 1; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const double value = static_cast<double>(scalar(i, j, k));
                const double density = static_cast<double>(rho(i, j, k));
                const double kh = kh_at(i, j, k);
                const double kv = kv_at(i, j, k);
                if (!std::isfinite(value) || !std::isfinite(density)
                    || density <= 0.0 || !std::isfinite(kh) || kh < 0.0
                    || !std::isfinite(kv) || kv < 0.0)
                    throw std::invalid_argument(
                        "terrain scalar diffusion requires finite physical state, positive density, and non-negative diffusivity");
            }
    for (int i = 1; i < nr - 1; ++i)
        for (int j = 1; j < nth - 1; ++j)
            for (int k = 1; k < nz; ++k)
                if (!std::isfinite(metrics.z(i, j, k))
                    || metrics.z(i, j, k) <= metrics.z(i, j, k - 1))
                    throw std::invalid_argument(
                        "terrain scalar diffusion requires increasing center heights");

    const auto x_face_gradient = [&](int face, int j, int k)
    {
        if (periodic_x && (face == 1 || face == nr - 1))
        {
            const int left = nr - 2;
            const int right = 1;
            const double computational_gradient =
                (value_at(right, j, k) - value_at(left, j, k)) / metrics.dx;
            const double slope =
                (metrics.z(right, j, k) - metrics.z(left, j, k)) / metrics.dx;
            const double vertical_gradient = 0.5 * (
                column_gradient_z(metrics, value_at, left, j, k)
              + column_gradient_z(metrics, value_at, right, j, k));
            return computational_gradient - slope * vertical_gradient;
        }
        return physical_gradient_x(metrics, value_at, face - 1, j, k);
    };
    const auto y_face_gradient = [&](int i, int face, int k)
    {
        if (periodic_y && (face == 1 || face == nth - 1))
        {
            const int south = nth - 2;
            const int north = 1;
            const double computational_gradient =
                (value_at(i, north, k) - value_at(i, south, k)) / metrics.dy;
            const double slope =
                (metrics.z(i, north, k) - metrics.z(i, south, k)) / metrics.dy;
            const double vertical_gradient = 0.5 * (
                column_gradient_z(metrics, value_at, i, south, k)
              + column_gradient_z(metrics, value_at, i, north, k));
            return computational_gradient - slope * vertical_gradient;
        }
        return physical_gradient_y(metrics, value_at, i, face - 1, k);
    };
    // Assemble the mapped operator as the negative weighted adjoint of the
    // exact face-gradient operators above:
    //
    //   M d(phi)/dt = -Dx^T Wx Dx phi - Dy^T Wy Dy phi - Dz^T Wz Dz phi.
    //
    // This mimetic form is conservative because every gradient annihilates a
    // constant, and it is energy dissipative by construction on arbitrary
    // valid slopes.  The earlier strong-form cross-flux interpolation was
    // conservative but not adjoint-consistent, so a vertically stratified
    // field over terrain could gain scalar variance.
    const auto column_gradient_coefficient =
        [&](int column_i, int column_j, int center_k, int target_k)
    {
        const int lower = center_k == 0 ? 0 : center_k - 1;
        const int upper = center_k + 1 < nz ? center_k + 1 : nz - 1;
        const double delta_z = metrics.z(column_i, column_j, upper)
                             - metrics.z(column_i, column_j, lower);
        double coefficient = 0.0;
        if (target_k == lower) coefficient -= 1.0 / delta_z;
        if (target_k == upper) coefficient += 1.0 / delta_z;
        return coefficient;
    };
    const auto x_face_cells = [&](int face)
    {
        const bool seam = periodic_x && face == 1;
        return std::array<int, 2>{
            seam ? nr - 2 : face - 1,
            seam ? 1 : face};
    };
    const auto y_face_cells = [&](int face)
    {
        const bool seam = periodic_y && face == 1;
        return std::array<int, 2>{
            seam ? nth - 2 : face - 1,
            seam ? 1 : face};
    };
    const auto x_face_active = [&](int face)
    {
        return (periodic_x && face == 1)
            || (face >= 2 && face <= nr - 2);
    };
    const auto y_face_active = [&](int face)
    {
        return (periodic_y && face == 1)
            || (face >= 2 && face <= nth - 2);
    };
    const auto x_gradient_coefficient =
        [&](int face, int face_j, int level,
            int target_i, int target_j, int target_k)
    {
        if (target_j != face_j) return 0.0;
        const auto cells = x_face_cells(face);
        const int left = cells[0];
        const int right = cells[1];
        double coefficient = 0.0;
        if (target_k == level)
        {
            if (target_i == left) coefficient -= 1.0 / metrics.dx;
            if (target_i == right) coefficient += 1.0 / metrics.dx;
        }
        const double slope = (metrics.z(right, face_j, level)
                            - metrics.z(left, face_j, level)) / metrics.dx;
        if (target_i == left || target_i == right)
            coefficient -= 0.5 * slope * column_gradient_coefficient(
                target_i, face_j, level, target_k);
        return coefficient;
    };
    const auto y_gradient_coefficient =
        [&](int face_i, int face, int level,
            int target_i, int target_j, int target_k)
    {
        if (target_i != face_i) return 0.0;
        const auto cells = y_face_cells(face);
        const int south = cells[0];
        const int north = cells[1];
        double coefficient = 0.0;
        if (target_k == level)
        {
            if (target_j == south) coefficient -= 1.0 / metrics.dy;
            if (target_j == north) coefficient += 1.0 / metrics.dy;
        }
        const double slope = (metrics.z(face_i, north, level)
                            - metrics.z(face_i, south, level)) / metrics.dy;
        if (target_j == south || target_j == north)
            coefficient -= 0.5 * slope * column_gradient_coefficient(
                face_i, target_j, level, target_k);
        return coefficient;
    };
    const auto x_weight = [&](int face, int j, int k)
    {
        const auto cells = x_face_cells(face);
        const int left = cells[0];
        const int right = cells[1];
        const bool seam = periodic_x && face == 1;
        const double jacobian = seam
            ? 0.5 * (metrics.J(left, j, k) + metrics.J(right, j, k))
            : metrics.Jx_face(face, j, k);
        const double transport = 0.5 * (
            rho_at(left, j, k) * kh_at(left, j, k)
          + rho_at(right, j, k) * kh_at(right, j, k));
        return metrics.dx * metrics.dy * metrics.dzeta
            * jacobian * transport;
    };
    const auto y_weight = [&](int i, int face, int k)
    {
        const auto cells = y_face_cells(face);
        const int south = cells[0];
        const int north = cells[1];
        const bool seam = periodic_y && face == 1;
        const double jacobian = seam
            ? 0.5 * (metrics.J(i, south, k) + metrics.J(i, north, k))
            : metrics.Jy_face(i, face, k);
        const double transport = 0.5 * (
            rho_at(i, south, k) * kh_at(i, south, k)
          + rho_at(i, north, k) * kh_at(i, north, k));
        return metrics.dx * metrics.dy * metrics.dzeta
            * jacobian * transport;
    };

    int invalid_tendencies = 0;
    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(nr) * nth * nz >= 32768) reduction(+:invalid_tendencies)
    for (int i = 1; i < nr - 1; ++i)
        for (int j = 1; j < nth - 1; ++j)
            for (int k = 0; k < nz; ++k)
            {
                double residual = 0.0;
                if (use_horizontal)
                {
                    std::array<int, 2> x_faces{i, i + 1};
                    if (periodic_x && x_faces[1] == nr - 1)
                        x_faces[1] = 1;
                    std::array<int, 2> y_faces{j, j + 1};
                    if (periodic_y && y_faces[1] == nth - 1)
                        y_faces[1] = 1;
                    const std::array<int, 3> levels{k - 1, k, k + 1};
                    for (int face_index = 0; face_index < 2; ++face_index)
                    {
                        const int face = x_faces[face_index];
                        if (!x_face_active(face)
                            || (face_index == 1 && face == x_faces[0]))
                            continue;
                        for (int level : levels)
                        {
                            if (level < 0 || level >= nz) continue;
                            const double coefficient = x_gradient_coefficient(
                                face, j, level, i, j, k);
                            if (coefficient != 0.0)
                                residual -= x_weight(face, j, level)
                                    * x_face_gradient(face, j, level)
                                    * coefficient;
                        }
                    }
                    for (int face_index = 0; face_index < 2; ++face_index)
                    {
                        const int face = y_faces[face_index];
                        if (!y_face_active(face)
                            || (face_index == 1 && face == y_faces[0]))
                            continue;
                        for (int level : levels)
                        {
                            if (level < 0 || level >= nz) continue;
                            const double coefficient = y_gradient_coefficient(
                                i, face, level, i, j, k);
                            if (coefficient != 0.0)
                                residual -= y_weight(i, face, level)
                                    * y_face_gradient(i, face, level)
                                    * coefficient;
                        }
                    }
                }
                if (use_vertical)
                {
                    for (int face : {k, k + 1})
                    {
                        if (face <= 0 || face >= nz) continue;
                        const int lower = face - 1;
                        const int upper = face;
                        const double delta_z = metrics.z(i, j, upper)
                                             - metrics.z(i, j, lower);
                        const double gradient =
                            (value_at(i, j, upper) - value_at(i, j, lower))
                            / delta_z;
                        const double transport = 0.5 * (
                            rho_at(i, j, lower) * kv_at(i, j, lower)
                          + rho_at(i, j, upper) * kv_at(i, j, upper));
                        const double coefficient =
                            (k == lower ? -1.0 : 1.0) / delta_z;
                        residual -= metrics.dx * metrics.dy * delta_z
                            * transport * gradient * coefficient;
                    }
                }
                const double cell_mass = rho_at(i, j, k)
                    * metrics.J(i, j, k) * metrics.dx * metrics.dy
                    * metrics.dzeta;
                const double specific = residual / cell_mass;
                const float stored = static_cast<float>(specific);
                if (!std::isfinite(stored))
                    ++invalid_tendencies;
                else
                    tendency(i, j, k) = stored;
            }
    if (invalid_tendencies > 0)
        throw std::runtime_error(
            "terrain scalar diffusion produced non-finite tendencies");
}
} // namespace

void cgrid_scalar_diffusion_tendency(
    const TerrainMetrics3D& metrics,
    const Field3D& scalar,
    const Field3D& rho,
    const Field3D* K_vertical,
    double K_h,
    double K_v,
    bool periodic_x,
    bool periodic_y,
    Field3D& tendency)
{
    cgrid_scalar_diffusion_tendency_impl(
        metrics, scalar, rho, nullptr, K_vertical, K_h, K_v,
        K_h > 0.0, K_vertical != nullptr || K_v > 0.0,
        periodic_x, periodic_y, tendency);
}

void cgrid_variable_scalar_diffusion_tendency(
    const TerrainMetrics3D& metrics,
    const Field3D& scalar,
    const Field3D& rho,
    const Field3D& K_horizontal,
    const Field3D& K_vertical,
    bool use_horizontal,
    bool use_vertical,
    bool periodic_x,
    bool periodic_y,
    Field3D& tendency)
{
    cgrid_scalar_diffusion_tendency_impl(
        metrics, scalar, rho, &K_horizontal, &K_vertical, 0.0, 0.0,
        use_horizontal, use_vertical, periodic_x, periodic_y, tendency);
}

double cgrid_scalar_diffusion_number(
    const TerrainMetrics3D& metrics,
    const Field3D& rho,
    const Field3D* K_vertical,
    double K_h,
    double K_v,
    double dt,
    bool periodic_x,
    bool periodic_y)
{
    validate_geometry(metrics);
    require_scalar_shape(metrics, rho, "terrain diffusion density");
    if (K_vertical != nullptr)
        require_scalar_shape(metrics, *K_vertical,
                             "terrain vertical diffusivity");
    if (!std::isfinite(K_h) || K_h < 0.0 || !std::isfinite(K_v) || K_v < 0.0
        || !std::isfinite(dt) || dt <= 0.0)
        throw std::invalid_argument(
            "terrain diffusion stability inputs are invalid");
    double rho_min = std::numeric_limits<double>::infinity();
    double rho_max = 0.0;
    double k_max = std::max(K_h, K_v);
    for (int i = 1; i < metrics.J.size_r() - 1; ++i)
        for (int j = 1; j < metrics.J.size_th() - 1; ++j)
            for (int k = 0; k < metrics.J.size_z(); ++k)
            {
                const double density = static_cast<double>(rho(i, j, k));
                const double kv = K_vertical
                    ? static_cast<double>((*K_vertical)(i, j, k)) : K_v;
                if (!std::isfinite(density) || density <= 0.0
                    || !std::isfinite(kv) || kv < 0.0)
                    throw std::invalid_argument(
                        "terrain diffusion stability requires positive physical density and non-negative diffusivity");
                rho_min = std::min(rho_min, density);
                rho_max = std::max(rho_max, density);
                k_max = std::max(k_max, kv);
            }
    if (k_max == 0.0) return 0.0;
    double dz_min = std::numeric_limits<double>::infinity();
    double slope_max = 0.0;
    double local_slope_x_max = 0.0;
    double local_slope_y_max = 0.0;
    for (int i = 0; i < metrics.J.size_r(); ++i)
        for (int j = 0; j < metrics.J.size_th(); ++j)
            for (int k = 0; k < metrics.J.size_z(); ++k)
            {
                dz_min = std::min(dz_min,
                    metrics.z_face(i, j, k + 1)
                  - metrics.z_face(i, j, k));
                const double slope_x =
                    std::abs(metrics.mx(i, j, k) * metrics.J(i, j, k));
                const double slope_y =
                    std::abs(metrics.my(i, j, k) * metrics.J(i, j, k));
                local_slope_x_max = std::max(local_slope_x_max, slope_x);
                local_slope_y_max = std::max(local_slope_y_max, slope_y);
                slope_max = std::max(slope_max, slope_x + slope_y);
            }
    double seam_slope_x_max = 0.0;
    if (periodic_x)
        for (int j = 1; j < metrics.J.size_th() - 1; ++j)
            for (int k = 0; k < metrics.J.size_z(); ++k)
                seam_slope_x_max = std::max(seam_slope_x_max,
                    std::abs(metrics.z(1, j, k)
                           - metrics.z(metrics.J.size_r() - 2, j, k))
                    / metrics.dx);
    double seam_slope_y_max = 0.0;
    if (periodic_y)
        for (int i = 1; i < metrics.J.size_r() - 1; ++i)
            for (int k = 0; k < metrics.J.size_z(); ++k)
                seam_slope_y_max = std::max(seam_slope_y_max,
                    std::abs(metrics.z(i, 1, k)
                           - metrics.z(i, metrics.J.size_th() - 2, k))
                    / metrics.dy);
    slope_max = std::max(slope_max,
        std::max(local_slope_x_max, seam_slope_x_max)
      + std::max(local_slope_y_max, seam_slope_y_max));
    if (!std::isfinite(dz_min) || dz_min <= 0.0)
        throw std::invalid_argument(
            "terrain diffusion stability requires positive layer thickness");
    const double metric_factor = 1.0 + slope_max;
    const double inverse_scale = 1.0 / (metrics.dx * metrics.dx)
        + 1.0 / (metrics.dy * metrics.dy)
        + 1.0 / (dz_min * dz_min);
    const double bound = 2.0 * dt * (rho_max / rho_min) * k_max
        * metric_factor * metric_factor * inverse_scale;
    if (!std::isfinite(bound))
        throw std::overflow_error(
            "terrain diffusion stability bound overflowed");
    return bound;
}

namespace
{
double require_jacobian(double jacobian)
{
    if (!std::isfinite(jacobian) || jacobian <= terrain_constants::epsilon)
        throw std::invalid_argument(
            "terrain contravariant velocity requires a positive finite Jacobian");
    return jacobian;
}

double zeta_velocity(double physical_u, double physical_v, double physical_w,
                     double slope_x, double slope_y, double jacobian)
{
    if (!std::isfinite(physical_u) || !std::isfinite(physical_v)
        || !std::isfinite(physical_w) || !std::isfinite(slope_x)
        || !std::isfinite(slope_y))
    {
        throw std::invalid_argument(
            "terrain contravariant velocity requires finite wind and slopes");
    }
    return (physical_w - physical_u * slope_x - physical_v * slope_y)
         / require_jacobian(jacobian);
}
} // namespace

double cgrid_zeta_velocity_at_x_face(const TerrainMetrics3D& metrics,
                                     double physical_u,
                                     double physical_v,
                                     double physical_w,
                                     int i, int j, int k)
{
    const double slope_x = (metrics.z(i + 1, j, k) - metrics.z(i, j, k))
                         / metrics.dx;
    const double z_u_north = 0.5 * (
        metrics.z(i, j + 1, k) + metrics.z(i + 1, j + 1, k));
    const double z_u_south = 0.5 * (
        metrics.z(i, j - 1, k) + metrics.z(i + 1, j - 1, k));
    const double slope_y = (z_u_north - z_u_south) / (2.0 * metrics.dy);
    return zeta_velocity(physical_u, physical_v, physical_w,
                         slope_x, slope_y,
                         metrics.Jx_face(i + 1, j, k));
}

double cgrid_zeta_velocity_at_y_face(const TerrainMetrics3D& metrics,
                                     double physical_u,
                                     double physical_v,
                                     double physical_w,
                                     int i, int j, int k)
{
    const double z_v_east = 0.5 * (
        metrics.z(i + 1, j, k) + metrics.z(i + 1, j + 1, k));
    const double z_v_west = 0.5 * (
        metrics.z(i - 1, j, k) + metrics.z(i - 1, j + 1, k));
    const double slope_x = (z_v_east - z_v_west) / (2.0 * metrics.dx);
    const double slope_y = (metrics.z(i, j + 1, k) - metrics.z(i, j, k))
                         / metrics.dy;
    return zeta_velocity(physical_u, physical_v, physical_w,
                         slope_x, slope_y,
                         metrics.Jy_face(i, j + 1, k));
}

double cgrid_zeta_velocity_at_z_face(const TerrainMetrics3D& metrics,
                                     double physical_u,
                                     double physical_v,
                                     double physical_w,
                                     int i, int j, int k)
{
    const int face_k = k + 1;
    const double slope_x = -metrics.Gx_face(i, j, face_k);
    const double slope_y = -metrics.Gy_face(i, j, face_k);
    const double jacobian = (metrics.z(i, j, k + 1)
                           - metrics.z(i, j, k)) / metrics.dzeta;
    return zeta_velocity(physical_u, physical_v, physical_w,
                         slope_x, slope_y, jacobian);
}

double cgrid_ground_vertical_velocity(const TerrainMetrics3D& metrics,
                                      const Field3D& u,
                                      const Field3D& v,
                                      int i, int j,
                                      double frame_u_ms,
                                      double frame_v_ms)
{
    if (!std::isfinite(frame_u_ms) || !std::isfinite(frame_v_ms))
        throw std::invalid_argument("terrain frame velocity must be finite");
    if (u.size_r() != metrics.J.size_r()
        || u.size_th() != metrics.J.size_th()
        || u.size_z() != metrics.J.size_z()
        || v.size_r() != metrics.J.size_r()
        || v.size_th() != metrics.J.size_th()
        || v.size_z() != metrics.J.size_z())
    {
        throw std::invalid_argument(
            "terrain ground velocity fields must match the geometry");
    }
    // Reconstruct the scalar-column horizontal velocity from the west and
    // south C-grid faces.  The upper closure columns are valid inputs: their
    // terrain metrics use the same one-sided, zero-normal-gradient domain
    // closure as the conservative face geometry.  The lower closure columns
    // have no stored west/south neighbour and must remain unavailable.
    if (i <= 0 || i >= metrics.J.size_r()
        || j <= 0 || j >= metrics.J.size_th())
    {
        throw std::out_of_range(
            "terrain ground velocity requires a scalar column with stored west and south faces");
    }
    // Resolve and validate the column before touching staggered neighbours.
    const GroundSurfaceFrame frame = ground_surface_frame(metrics, i, j);
    const double ground_u = 0.5 * (
        static_cast<double>(u[i - 1][j][0])
        + static_cast<double>(u[i][j][0]));
    const double ground_v = 0.5 * (
        static_cast<double>(v[i][j - 1][0])
        + static_cast<double>(v[i][j][0]));
    return ground_tangent_velocity(
        frame, ground_u - frame_u_ms, ground_v - frame_v_ms).z;
}

namespace
{
template <typename Gradient>
double pressure_perturbation_gradient(
    const TerrainMetrics3D& metrics,
    const Field3D& pressure,
    const std::vector<double>& reference_pressure,
    int i, int j, int k,
    const Gradient& gradient)
{
    validate_geometry_shape(metrics);
    require_scalar_shape(metrics, pressure, "terrain pressure");
    if (reference_pressure.size()
        != static_cast<std::size_t>(metrics.J.size_z()))
    {
        throw std::invalid_argument(
            "terrain reference pressure must span every vertical level");
    }
    const auto perturbation_at = [&](int ii, int jj, int kk)
    {
        const double reference = reference_profile_value_fp32(
            reference_pressure, metrics.z(ii, jj, kk), metrics.dzeta);
        return static_cast<double>(pressure[ii][jj][kk]) - reference;
    };
    return gradient(metrics, perturbation_at, i, j, k);
}
} // namespace

double cgrid_pressure_perturbation_gradient_x(
    const TerrainMetrics3D& metrics,
    const Field3D& pressure,
    const std::vector<double>& reference_pressure,
    int i, int j, int k)
{
    return pressure_perturbation_gradient(
        metrics, pressure, reference_pressure, i, j, k,
        [](const TerrainMetrics3D& local_metrics, const auto& value_at,
           int ii, int jj, int kk)
        {
            return physical_gradient_x(
                local_metrics, value_at, ii, jj, kk);
        });
}

double cgrid_pressure_perturbation_gradient_y(
    const TerrainMetrics3D& metrics,
    const Field3D& pressure,
    const std::vector<double>& reference_pressure,
    int i, int j, int k)
{
    return pressure_perturbation_gradient(
        metrics, pressure, reference_pressure, i, j, k,
        [](const TerrainMetrics3D& local_metrics, const auto& value_at,
           int ii, int jj, int kk)
        {
            return physical_gradient_y(
                local_metrics, value_at, ii, jj, kk);
        });
}

double cgrid_pressure_perturbation_gradient_z(
    const TerrainMetrics3D& metrics,
    const Field3D& pressure,
    const std::vector<double>& reference_pressure,
    int i, int j, int k)
{
    return pressure_perturbation_gradient(
        metrics, pressure, reference_pressure, i, j, k,
        [](const TerrainMetrics3D& local_metrics, const auto& value_at,
           int ii, int jj, int kk)
        {
            return physical_gradient_z_face(
                local_metrics, value_at, ii, jj, kk);
        });
}

} // namespace tmv::terrain_transport
