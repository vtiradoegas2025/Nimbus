#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/simulation_state.hpp"
/**
 * @file advection_cartesian.cpp
 * @brief Cartesian-grid scalar advection kernel implementations (Phase A.5).
 *
 * Implements the three helpers declared in
 * `include/numerics/advection_cartesian.hpp`:
 *
 *   - advect_scalar_1d_x_kernel_cartesian()
 *   - advect_scalar_1d_y_kernel_cartesian()
 *   - apply_diffusion_kernel_cartesian()
 *
 * The kernels use the standard first-order-upwind + Forward-Euler structure
 * with straight Cartesian indexing in (x, y, z):
 *
 *   1. The y kernel carries no `1/r` metric factor; the arc-length spacing
 *      is just `dy`.
 *
 *   2. The y kernel uses no periodic wraparound on j. The j = 0 and
 *      j = NTH − 1 faces are physical (zero-gradient) boundaries.
 *
 * The diffusion kernel is the standard 7-point Cartesian Laplacian.
 *
 * The vertical TVD scheme in `src/numerics/advection/schemes/tvd/tvd.cpp`
 * loops i, j, k and only operates on the z column at each (i, j) — it
 * touches no horizontal metric. The dispatcher reuses it for the z step.
 *
 * Boundary convention: every kernel here begins by copying `src` into
 * `dst` (a single contiguous memcpy), then overwrites the interior
 * (1..NR−2, 1..NTH−2 for the y kernel, 1..NZ−2). The six face planes
 * carry the source values through unchanged. This is the discrete
 * equivalent of `∂q/∂n = 0` on every face — the same zero-gradient
 * Cartesian BC the A.3 boundary helpers install for the dynamics fields.
 *
 * This file is linked by:
 *   - the production runtime via the standard SRCS list, and
 *   - the A.5 unit test via `tests/numerics/test_advection_cartesian.cpp`.
 *
 */

#include "numerics/advection/advection_cartesian.hpp"
#include "numerics/advection/advection_perf.hpp"
#include "numerics/advection/cartesian_mass_flux.hpp"
#include "numerics/advection/weno5_face.hpp" // weno5_face_value (single source; unit-tested)
#include "diagnostics/deterministic_reduction.hpp"
#include "dynamics/device_acoustic_energy_attribution.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/runtime/process_state.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/field/field_bounds.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "terrain/conservative_transport.hpp"
#include "compute/dispatch/terrain.hpp"
#include "compute/backend_runtime.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

/**
 * @brief Resizes `field` to (NR, NTH, NZ) if it doesn't already match.
 *
 * Duplicated from the inline helper in `src/numerics/advection/advection.cpp`
 * rather than shared so this file's link closure stays small.
 */
inline void ensure_field_shape(const SimulationStateView& execution_state, Field3D& field)
{
    if (field.size_r() != execution_state.grid.nr || field.size_th() != execution_state.grid.nth || field.size_z() != execution_state.grid.nz)
    {
        field.resize(execution_state.grid.nr, execution_state.grid.nth, execution_state.grid.nz, 0.0f);
    }
}

inline void require_global_field_shape(const SimulationStateView& execution_state, const Field3D& field, const char* label)
{
    if (execution_state.grid.nr <= 0 || execution_state.grid.nth <= 0 || execution_state.grid.nz <= 0 || field.size_r() != execution_state.grid.nr || field.size_th() != execution_state.grid.nth || field.size_z() != execution_state.grid.nz)
    {
        throw std::invalid_argument(std::string(label) + " shape does not match the Cartesian grid");
    }
}

inline double require_positive_finite(double value, const char* label)
{
    if (!std::isfinite(value) || value <= 0.0)
    {
        throw std::invalid_argument(std::string(label) + " must be positive and finite");
    }
    return value;
}

/**
 * @brief Computes the row-major flat index for a 3D cell.
 *
 * Matches the layout used by `Field3D::data()` — `(i * NTH + j) * NZ + k`.
 */
inline std::size_t idx3(const SimulationStateView& execution_state, int i, int j, int k)
{
    return (static_cast<std::size_t>(i) * static_cast<std::size_t>(execution_state.grid.nth) + static_cast<std::size_t>(j)) *
               static_cast<std::size_t>(execution_state.grid.nz) +
           static_cast<std::size_t>(k);
}

/**
 * @brief Seeds `dst` with the entire contents of `src`.
 *
 * The Cartesian advection and diffusion kernels only write the interior
 * (1..NR−2, 1..NTH−2, 1..NZ−2) and rely on the six face planes already
 * holding the source values for the zero-gradient boundary condition.
 * The seeding is a single contiguous `memcpy`.
 */
inline void seed_destination_from_source(const SimulationStateView& execution_state, const Field3D& src, Field3D& dst)
{
    ensure_field_shape(execution_state, dst);
    if (src.size() == 0 || src.data() == dst.data())
    {
        return;
    }
    std::memcpy(dst.data(), src.data(), src.size() * sizeof(float));
}

/**
 * @brief Three-argument minmod: common-signed minimum magnitude, else zero.
 */
inline double minmod3(double a, double b, double c)
{
    if (a > 0.0 && b > 0.0 && c > 0.0)
        return std::min(a, std::min(b, c));
    if (a < 0.0 && b < 0.0 && c < 0.0)
        return std::max(a, std::max(b, c));
    return 0.0;
}

/**
 * @brief MC (monotonized-central) limited cell slope from three cell values.
 *
 * @param qm Left neighbour (i-1), @param q0 centre (i), @param qp right (i+1).
 * @return Limited slope for the +/- 0.5 face reconstruction. Zero at a local
 *         extremum, so the reconstruction never manufactures a new extremum.
 *
 * Used only as the near-boundary fallback for weno5_face_value(): the MC
 * limiter clips its slope to zero at smooth extrema (e.g. a warm-bubble theta
 * peak), which degrades the scheme to 1st-order upwind there and numerically
 * diffuses the peak away. WENO5 is used everywhere the 5-point stencil fits.
 */
inline double mc_limited_slope(double qm, double q0, double qp)
{
    const double right = qp - q0;
    const double left  = q0 - qm;
    return minmod3(0.5 * (right + left), 2.0 * right, 2.0 * left);
}

// weno5_face_value (the upwind WENO5-JS face reconstruction) now lives in
// numerics/advection/weno5_face.hpp (included above): a single, unit-tested
// source shared with the GPU shader's literal mirror. Kept dependency-free so
// this file's link closure stays clear of the advection factory.

} // namespace

namespace tmv
{

namespace
{

std::array<double, 6> scalar_moist_energy_coefficients_v2(const SimulationStateView& execution_state, ScalarMoistEnergyKind     kind,
                                                          StaggerType               staggering,
                                                          const TerrainMetrics3D*   terrain_metrics,
                                                          const PhysicalCellDomain& domain,
                                                          int                       i,
                                                          int                       j,
                                                          int                       k)
{
    std::array<double, 6> coefficients{};
    if (kind == ScalarMoistEnergyKind::None)
        return coefficients;

    constexpr double cv_d = physical_constants::specific_heat_cp_jkgk - physical_constants::gas_constant_dry_air_jkgk;
    constexpr double cv_v =
        physical_constants::specific_heat_water_vapor_cp_jkgk - physical_constants::gas_constant_water_vapor_jkgk;
    constexpr double c_l   = physical_constants::specific_heat_liquid_water_jkgk;
    constexpr double c_i   = physical_constants::specific_heat_ice_jkgk;
    constexpr double t_ref = physical_constants::freezing_temperature_k;
    constexpr double vapor_reference =
        physical_constants::latent_heat_vaporization_jkg - physical_constants::gas_constant_water_vapor_jkgk * t_ref;
    constexpr double ice_reference = -physical_constants::latent_heat_fusion_jkg;
    constexpr double kappa = physical_constants::gas_constant_dry_air_jkgk / physical_constants::specific_heat_cp_jkgk;

    const std::size_t c           = idx3(execution_state, i, j, k);
    const double      pressure    = static_cast<double>(execution_state.fields.p.data()[c]);
    const double      exner       = std::pow(pressure / physical_constants::reference_pressure_pa, kappa);
    const double      temperature = static_cast<double>(execution_state.fields.theta.data()[c]) * exner;
    const double      delta_t     = temperature - t_ref;

    if (kind == ScalarMoistEnergyKind::PotentialTemperature)
    {
        const double q_liquid = static_cast<double>(execution_state.fields.qc.data()[c]) + static_cast<double>(execution_state.fields.qr.data()[c]);
        const double q_ice    = static_cast<double>(execution_state.fields.qi.data()[c]) + static_cast<double>(execution_state.fields.qs.data()[c]) +
                                static_cast<double>(execution_state.fields.qg.data()[c]) + static_cast<double>(execution_state.fields.qh.data()[c]);
        coefficients[0]       = cv_d * exner;
        coefficients[1]       = exner * (static_cast<double>(execution_state.fields.qv.data()[c]) * cv_v + q_liquid * c_l + q_ice * c_i);
        return coefficients;
    }

    if (kind == ScalarMoistEnergyKind::WaterVapor)
    {
        coefficients[1] = cv_v * delta_t;
        coefficients[2] = vapor_reference;
    }
    else if (kind == ScalarMoistEnergyKind::LiquidWater)
    {
        coefficients[1] = c_l * delta_t;
    }
    else
    {
        coefficients[1] = c_i * delta_t;
        coefficients[2] = ice_reference;
    }

    if (staggering == StaggerType::Collocated)
    {
        const double uc = static_cast<double>(execution_state.fields.u.data()[c]);
        const double vc = static_cast<double>(execution_state.fields.v.data()[c]);
        const double wc = static_cast<double>(execution_state.fields.w.data()[c]);
        coefficients[3] = 0.5 * (uc * uc + vc * vc + wc * wc);
    }
    else
    {
        const double u_w = static_cast<double>(execution_state.fields.u.data()[idx3(execution_state, i - 1, j, k)]);
        const double u_e = static_cast<double>(execution_state.fields.u.data()[c]);
        const double v_s = static_cast<double>(execution_state.fields.v.data()[idx3(execution_state, i, j - 1, k)]);
        const double v_n = static_cast<double>(execution_state.fields.v.data()[c]);
        double       w_b = 0.0;
        if (k != domain.k_begin)
        {
            w_b = static_cast<double>(execution_state.fields.w.data()[idx3(execution_state, i, j, k - 1)]);
        }
        else if (terrain_metrics != nullptr)
        {
            w_b = terrain_transport::cgrid_ground_vertical_velocity(*terrain_metrics,
                                                                    execution_state.fields.u,
                                                                    execution_state.fields.v,
                                                                    i,
                                                                    j,
                                                                    tmv::physical_grid_metrics::surface_velocity_ms(execution_state).u_ms,
                                                                    tmv::physical_grid_metrics::surface_velocity_ms(execution_state).v_ms);
        }
        const double w_t = static_cast<double>(execution_state.fields.w.data()[c]);
        coefficients[3]  = 0.25 * (u_w * u_w + u_e * u_e + v_s * v_s + v_n * v_n + w_b * w_b + w_t * w_t);
    }

    const double height = terrain_metrics != nullptr ? terrain_metrics->z(i, j, k)
                                                     : (execution_state.grid.geometry.z.size() == static_cast<std::size_t>(execution_state.grid.nz)
                                                            ? execution_state.grid.geometry.z[static_cast<std::size_t>(k)]
                                                            : (static_cast<double>(k) + 0.5) * execution_state.grid.dz);
    coefficients[4]     = physical_constants::gravity_ms2 * height;
    return coefficients;
}

double sum_energy_coefficients(const std::array<double, 6>& coefficients)
{
    double total = 0.0;
    for (double coefficient : coefficients)
        total += coefficient;
    return total;
}

float device_minmod3(float a, float b, float c)
{
    if (a > 0.0f && b > 0.0f && c > 0.0f)
        return std::min(a, std::min(b, c));
    if (a < 0.0f && b < 0.0f && c < 0.0f)
        return std::max(a, std::max(b, c));
    return 0.0f;
}

float device_mc_limited_slope(float qm, float q0, float qp)
{
    const float right = qp - q0;
    const float left  = q0 - qm;
    return device_minmod3(0.5f * (right + left), 2.0f * right, 2.0f * left);
}

/**
 * Literal host mirror of advect_conservative_scalar_flux_cartesian_cgrid.comp.
 *
 * This deliberately does not call the FP64 CPU WENO helper: attribution of a
 * retained device trajectory must use the FP32 request actually selected by
 * that trajectory, including the shader's 1e-30 all-zero floor.
 */
float device_weno5_face_offset(float a, float b, float c, float d, float e)
{
    return weno5_face_offset_fp32(a, b, c, d, e);
}

double carrier_kinetic_coefficient_v2(const CarrierKineticBoundaryInputs& inputs,
                                      StaggerType                         staggering,
                                      const TerrainMetrics3D*             terrain_metrics,
                                      const PhysicalCellDomain&           domain,
                                      const Field3D&                      u_state,
                                      const Field3D&                      v_state,
                                      const Field3D&                      w_state,
                                      int                                 i,
                                      int                                 j,
                                      int                                 k,
                                      double                              frame_u_ms,
                                      double                              frame_v_ms)
{
    double q_total = 0.0;
    for (const Field3D* field : inputs.water)
    {
        const double value = static_cast<double>((*field)[i][j][k]);
        if (!std::isfinite(value) || value < 0.0)
            return std::numeric_limits<double>::quiet_NaN();
        q_total += value;
    }

    double specific_kinetic = 0.0;
    if (staggering == StaggerType::Collocated)
    {
        const double uc  = static_cast<double>(u_state[i][j][k]);
        const double vc  = static_cast<double>(v_state[i][j][k]);
        const double wc  = static_cast<double>(w_state[i][j][k]);
        specific_kinetic = 0.5 * (uc * uc + vc * vc + wc * wc);
    }
    else
    {
        const double u_w = static_cast<double>(u_state[i - 1][j][k]);
        const double u_e = static_cast<double>(u_state[i][j][k]);
        const double v_s = static_cast<double>(v_state[i][j - 1][k]);
        const double v_n = static_cast<double>(v_state[i][j][k]);
        double       w_b = 0.0;
        if (k != domain.k_begin)
            w_b = static_cast<double>(w_state[i][j][k - 1]);
        else if (terrain_metrics != nullptr)
            w_b = terrain_transport::cgrid_ground_vertical_velocity(
                *terrain_metrics, u_state, v_state, i, j, frame_u_ms, frame_v_ms);
        const double w_t = static_cast<double>(w_state[i][j][k]);
        specific_kinetic = 0.25 * (u_w * u_w + u_e * u_e + v_s * v_s + v_n * v_n + w_b * w_b + w_t * w_t);
    }

    const double coefficient = (1.0 + q_total) * specific_kinetic;
    return std::isfinite(coefficient) ? coefficient : std::numeric_limits<double>::quiet_NaN();
}

} // namespace

void CartesianCGridTransportContext::begin(int                     nr,
                                           int                     nth,
                                           int                     nz,
                                           double                  dt_large,
                                           const TerrainMetrics3D* terrain_metrics,
                                           double                  frame_u_ms,
                                           double                  frame_v_ms)
{
    return begin(global_simulation_state(), nr, nth, nz, dt_large, terrain_metrics, frame_u_ms, frame_v_ms);
}

void CartesianCGridTransportContext::begin(const SimulationStateView& execution_state, int                     nr,
                                           int                     nth,
                                           int                     nz,
                                           double                  dt_large,
                                           const TerrainMetrics3D* terrain_metrics,
                                           double                  frame_u_ms,
                                           double                  frame_v_ms)
{
    if (nr <= 0 || nth <= 0 || nz <= 0)
    {
        throw std::invalid_argument("CartesianCGridTransportContext::begin requires a non-empty domain");
    }
    if (!std::isfinite(dt_large) || dt_large <= 0.0)
    {
        throw std::invalid_argument("CartesianCGridTransportContext::begin requires finite dt_large > 0");
    }
    if (!std::isfinite(frame_u_ms) || !std::isfinite(frame_v_ms))
        throw std::invalid_argument("CartesianCGridTransportContext::begin requires finite frame velocity");

    nr_                                  = nr;
    nth_                                 = nth;
    nz_                                  = nz;
    dt_large_                            = dt_large;
    accumulated_dt_                      = 0.0;
    substep_count_                       = 0;
    carrier_kinetic_substep_count_       = 0;
    active_                              = true;
    complete_                            = false;
    finite_                              = true;
    density_begin_captured_              = false;
    fp32_time_currency_                  = false;
    carrier_kinetic_tracking_            = false;
    boundary_flux_compatible_            = true;
    lateral_bc_west_                     = static_cast<int>(execution_state.configuration->lateral_bc_west);
    lateral_bc_east_                     = static_cast<int>(execution_state.configuration->lateral_bc_east);
    lateral_bc_south_                    = static_cast<int>(execution_state.configuration->lateral_bc_south);
    lateral_bc_north_                    = static_cast<int>(execution_state.configuration->lateral_bc_north);
    bottom_bc_                           = static_cast<int>(execution_state.configuration->bottom_bc);
    top_bc_                              = static_cast<int>(execution_state.configuration->top_bc);
    outward_carrier_kinetic_transport_j_ = 0.0;
    frame_u_ms_                          = frame_u_ms;
    frame_v_ms_                          = frame_v_ms;
    staggering_                          = execution_state.configuration->stagger_type;
    terrain_metrics_                     = terrain_metrics;
    jacobian_begin_.clear();
    if (terrain_metrics_ != nullptr)
    {
        terrain_transport::validate_geometry(*terrain_metrics_);
        if (terrain_metrics_->J.size_r() != nr_ || terrain_metrics_->J.size_th() != nth_ ||
            terrain_metrics_->J.size_z() != nz_)
            throw std::invalid_argument("terrain retained transport geometry does not match the domain");
    }

    auto reset = [nr, nth, nz](Field3D& field)
    {
        if (field.size_r() != nr || field.size_th() != nth || field.size_z() != nz)
        {
            field.resize(nr, nth, nz, 0.0f);
        }
        else
        {
            std::fill(field.data(), field.data() + field.size(), 0.0f);
        }
    };
    reset(mass_flux_x);
    reset(mass_flux_y);
    reset(mass_flux_z);
    reset(density_begin);
}

void CartesianCGridTransportContext::set_beginning_geometry(const TerrainMetrics3D& metrics)
{
    if (!active_ || complete_ || substep_count_ != 0 || terrain_metrics_ == nullptr)
        throw std::logic_error("beginning geometry requires an unaccumulated mapped transport context");
    terrain_transport::validate_geometry(metrics);
    if (metrics.J.nr != nr_ || metrics.J.nth != nth_ || metrics.J.nz != nz_
        || metrics.dx != terrain_metrics_->dx || metrics.dy != terrain_metrics_->dy
        || metrics.dzeta != terrain_metrics_->dzeta)
        throw std::invalid_argument("beginning transport geometry does not match its ending partition");
    jacobian_begin_ = metrics.J.values;
}

void CartesianCGridTransportContext::accumulate_integrated_moving_fluxes(
    const TerrainContravariantFluxes& flux, double interval_s)
{
    const double next_time = accumulated_dt_ + interval_s;
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0,dt_large_);
    if (!active_ || complete_ || !has_moving_geometry() || !density_begin_captured_
        || staggering_ != StaggerType::CGrid || carrier_kinetic_tracking_
        || !std::isfinite(interval_s) || interval_s <= 0.0
        || !std::isfinite(next_time) || next_time <= accumulated_dt_ || next_time > dt_large_ + tolerance
        || substep_count_ == std::numeric_limits<int>::max()
        || nr_ == std::numeric_limits<int>::max() || nth_ == std::numeric_limits<int>::max() || nz_ == std::numeric_limits<int>::max())
        throw std::invalid_argument("moving flux accumulation has an invalid context or interval");
    const auto shape = [](const TerrainField3D& field, int nx, int ny, int levels) {
        return field.nr == nx && field.nth == ny && field.nz == levels
            && field.values.size() == static_cast<std::size_t>(nx)*ny*levels;
    };
    const auto stored_shape = [&](const Field3D& field) {
        return field.size_r() == nr_ && field.size_th() == nth_ && field.size_z() == nz_;
    };
    if (!shape(flux.x,nr_+1,nth_,nz_) || !shape(flux.y,nr_,nth_+1,nz_) || !shape(flux.z,nr_,nth_,nz_+1)
        || !stored_shape(mass_flux_x) || !stored_shape(mass_flux_y) || !stored_shape(mass_flux_z))
        throw std::invalid_argument("moving face fluxes do not match retained transport storage");
    const double x_scale = dt_large_ * terrain_metrics_->dy * terrain_metrics_->dzeta;
    const double y_scale = dt_large_ * terrain_metrics_->dx * terrain_metrics_->dzeta;
    const double z_scale = dt_large_ * terrain_metrics_->dx * terrain_metrics_->dy;
    for (double scale : {x_scale,y_scale,z_scale})
        if (!std::isfinite(scale) || scale <= 0.0)
            throw std::overflow_error("moving face-flux units cannot be represented");
    Field3D next_x = mass_flux_x, next_y = mass_flux_y, next_z = mass_flux_z;
    for (int i=0; i<nr_; ++i)
        for (int j=0; j<nth_; ++j)
        {
            if (flux.z(i,j,0) != 0.0)
                throw std::invalid_argument("retained C-grid transport requires an impermeable ground");
            for (int k=0; k<nz_; ++k)
            {
                const double x = static_cast<double>(next_x(i,j,k)) + (i+1 < nr_ ? flux.x(i+1,j,k)/x_scale : 0.0);
                const double y = static_cast<double>(next_y(i,j,k)) + (j+1 < nth_ ? flux.y(i,j+1,k)/y_scale : 0.0);
                const double z = static_cast<double>(next_z(i,j,k)) + flux.z(i,j,k+1)/z_scale;
                if (!std::isfinite(static_cast<float>(x)) || !std::isfinite(static_cast<float>(y))
                    || !std::isfinite(static_cast<float>(z)))
                    throw std::runtime_error("moving flux accumulation cannot publish a non-finite value");
                next_x(i,j,k)=static_cast<float>(x);
                next_y(i,j,k)=static_cast<float>(y);
                next_z(i,j,k)=static_cast<float>(z);
            }
        }
    mass_flux_x=std::move(next_x); mass_flux_y=std::move(next_y); mass_flux_z=std::move(next_z);
    accumulated_dt_=next_time;
    ++substep_count_;
}

double CartesianCGridTransportContext::beginning_jacobian(int i, int j, int k) const
{
    if (i < 0 || j < 0 || k < 0 || i >= nr_ || j >= nth_ || k >= nz_)
        throw std::out_of_range("beginning-geometry index is outside the transport context");

    return jacobian_begin_.empty()
        ? (terrain_metrics_ != nullptr ? terrain_metrics_->J(i,j,k) : 1.0)
        : jacobian_begin_.at((static_cast<std::size_t>(i)*nth_ + j)*nz_ + k);
}

double CartesianCGridTransportContext::begin_density_in_current_geometry(int i, int j, int k) const
{
    if (i < 0 || j < 0 || k < 0 || i >= nr_ || j >= nth_ || k >= nz_)
        throw std::out_of_range("beginning-density index is outside the transport context");

    const double density = density_begin(i,j,k);
    return jacobian_begin_.empty() ? density
        : density * beginning_jacobian(i,j,k) / terrain_metrics_->J(i,j,k);
}

void CartesianCGridTransportContext::capture_density_begin(const Field3D& rho_state)
{
    if (!active_ || complete_)
    {
        throw std::logic_error(
            "CartesianCGridTransportContext::capture_density_begin requires an active, unsealed context");
    }
    if (rho_state.size_r() != nr_ || rho_state.size_th() != nth_ || rho_state.size_z() != nz_)
    {
        throw std::invalid_argument("CartesianCGridTransportContext::capture_density_begin field shape mismatch");
    }
    if (density_begin_captured_)
    {
        return;
    }

    std::copy(rho_state.data(), rho_state.data() + rho_state.size(), density_begin.data());
    for (std::size_t n = 0; n < density_begin.size(); ++n)
    {
        if (!std::isfinite(static_cast<double>(density_begin.data()[n])) || density_begin.data()[n] <= 0.0f)
        {
            finite_ = false;
            throw std::runtime_error("Cartesian transport beginning density must be finite and strictly positive");
        }
    }
    density_begin_captured_ = true;
}

void CartesianCGridTransportContext::accumulate(
    const Field3D&                                         rho_state,
    const Field3D&                                         u_state,
    const Field3D&                                         v_state,
    const Field3D&                                         w_state,
    double                                                 dt_small,
    const cartesian_mass_flux::CollocatedPressureCoupling* pressure_coupling,
    const CarrierKineticBoundaryInputs*                    carrier_kinetic_inputs,
    AcousticMassFluxPrecision precision)
{
    return accumulate(global_simulation_state(), rho_state, u_state, v_state, w_state, dt_small, pressure_coupling, carrier_kinetic_inputs, precision);
}

void CartesianCGridTransportContext::accumulate(const SimulationStateView& execution_state,
    const Field3D&                                         rho_state,
    const Field3D&                                         u_state,
    const Field3D&                                         v_state,
    const Field3D&                                         w_state,
    double                                                 dt_small,
    const cartesian_mass_flux::CollocatedPressureCoupling* pressure_coupling,
    const CarrierKineticBoundaryInputs*                    carrier_kinetic_inputs,
    AcousticMassFluxPrecision precision)
{
    accumulate_impl(execution_state, rho_state, u_state, v_state, w_state, dt_small,
        pressure_coupling, carrier_kinetic_inputs, precision, false);
}

void CartesianCGridTransportContext::capture_instantaneous(
    const Field3D& density, const Field3D& u, const Field3D& v, const Field3D& w,
    double dt, double frame_u_ms, double frame_v_ms)
{
    return capture_instantaneous(global_simulation_state(), density, u, v, w, dt, frame_u_ms, frame_v_ms);
}

void CartesianCGridTransportContext::capture_instantaneous(const SimulationStateView& execution_state,
    const Field3D& density, const Field3D& u, const Field3D& v, const Field3D& w,
    double dt, double frame_u_ms, double frame_v_ms)
{
    begin(execution_state, density.size_r(), density.size_th(), density.size_z(), dt,
        nullptr, frame_u_ms, frame_v_ms);
    accumulate_impl(execution_state, density, u, v, w, dt, nullptr, nullptr, AcousticMassFluxPrecision::Host, true);
    complete(execution_state);
}

void CartesianCGridTransportContext::accumulate_impl(
    const Field3D&                                         rho_state,
    const Field3D&                                         u_state,
    const Field3D&                                         v_state,
    const Field3D&                                         w_state,
    double                                                 dt_small,
    const cartesian_mass_flux::CollocatedPressureCoupling* pressure_coupling,
    const CarrierKineticBoundaryInputs*                    carrier_kinetic_inputs,
    AcousticMassFluxPrecision precision,
    bool instantaneous)
{
    return accumulate_impl(global_simulation_state(), rho_state, u_state, v_state, w_state, dt_small, pressure_coupling, carrier_kinetic_inputs, precision, instantaneous);
}

void CartesianCGridTransportContext::accumulate_impl(const SimulationStateView& execution_state,
    const Field3D&                                         rho_state,
    const Field3D&                                         u_state,
    const Field3D&                                         v_state,
    const Field3D&                                         w_state,
    double                                                 dt_small,
    const cartesian_mass_flux::CollocatedPressureCoupling* pressure_coupling,
    const CarrierKineticBoundaryInputs*                    carrier_kinetic_inputs,
    AcousticMassFluxPrecision precision,
    bool instantaneous)
{
    if (precision != AcousticMassFluxPrecision::Host && precision != AcousticMassFluxPrecision::NativeFloat &&
        precision != AcousticMassFluxPrecision::NativeCompensated)
        throw std::invalid_argument("invalid acoustic mass-flux precision");
    const bool compensated_device = precision == AcousticMassFluxPrecision::NativeCompensated;
    const bool native_flux = precision != AcousticMassFluxPrecision::Host;
    const auto native_x_flux = [&](int i, int j, int k)
    {
        return compensated_device
            ? tmv::device_acoustic_energy::x_face_mass_flux_precise(rho_state, u_state, i, j, k, static_cast<float>(frame_u_ms_))
            : static_cast<double>(tmv::device_acoustic_energy::x_face_mass_flux(rho_state, u_state, i, j, k, static_cast<float>(frame_u_ms_)));
    };
    const auto native_y_flux = [&](int i, int j, int k)
    {
        return compensated_device
            ? tmv::device_acoustic_energy::y_face_mass_flux_precise(rho_state, v_state, i, j, k, static_cast<float>(frame_v_ms_))
            : static_cast<double>(tmv::device_acoustic_energy::y_face_mass_flux(rho_state, v_state, i, j, k, static_cast<float>(frame_v_ms_)));
    };
    const auto native_z_flux = [&](int i, int j, int k)
    {
        return compensated_device
            ? tmv::device_acoustic_energy::z_face_mass_flux_precise(rho_state, w_state, nz_, i, j, k)
            : static_cast<double>(tmv::device_acoustic_energy::z_face_mass_flux(rho_state, w_state, nz_, i, j, k));
    };
    if (has_moving_geometry())
        throw std::logic_error("moving geometry requires preintegrated stage mass fluxes");

    if (!active_ || complete_)
    {
        throw std::logic_error("CartesianCGridTransportContext::accumulate requires an active, unsealed context");
    }
    if (!std::isfinite(dt_small) || dt_small <= 0.0)
    {
        throw std::invalid_argument("CartesianCGridTransportContext::accumulate requires finite dt_small > 0");
    }
    const auto matches = [this](const Field3D& field)
    { return field.size_r() == nr_ && field.size_th() == nth_ && field.size_z() == nz_; };
    if (!matches(rho_state) || !matches(u_state) || !matches(v_state) || !matches(w_state))
    {
        throw std::invalid_argument("CartesianCGridTransportContext::accumulate field shape mismatch");
    }
    if (carrier_kinetic_inputs != nullptr)
    {
        if (!carrier_kinetic_inputs->complete())
            throw std::invalid_argument("carrier-kinetic boundary attribution requires all seven water fields");
        for (const Field3D* field : carrier_kinetic_inputs->water)
            if (!matches(*field))
                throw std::invalid_argument("carrier-kinetic boundary attribution field shape mismatch");
        if (substep_count_ != carrier_kinetic_substep_count_)
            throw std::logic_error("carrier-kinetic boundary attribution cannot begin after an unattributed substep");
        carrier_kinetic_tracking_ = true;
    }
    else if (carrier_kinetic_tracking_)
    {
        throw std::logic_error("carrier-kinetic boundary attribution cannot omit a later substep");
    }
    if (staggering_ == StaggerType::Collocated && !instantaneous)
    {
        if (pressure_coupling == nullptr || pressure_coupling->pressure == nullptr ||
            pressure_coupling->base_pressure == nullptr || !matches(*pressure_coupling->pressure) ||
            static_cast<int>(pressure_coupling->base_pressure->size()) < nz_ || !std::isfinite(pressure_coupling->dt) ||
            pressure_coupling->dt <= 0.0 || pressure_coupling->dt != dt_small ||
            !std::isfinite(pressure_coupling->dx) || pressure_coupling->dx <= 0.0 ||
            !std::isfinite(pressure_coupling->dy) || pressure_coupling->dy <= 0.0 ||
            !std::isfinite(pressure_coupling->dz) || pressure_coupling->dz <= 0.0)
        {
            throw std::invalid_argument("collocated transport accumulation requires complete pressure-coupling "
                                        "geometry for the same acoustic substep");
        }
    }
    capture_density_begin(rho_state);

    if (terrain_metrics_ != nullptr)
    {
        terrain_transport::validate_geometry_shape(*terrain_metrics_);
        if (!matches(rho_state) || !matches(u_state)
            || !matches(v_state) || !matches(w_state))
            throw std::invalid_argument(
                "mapped Cartesian retained transport state has the wrong shape");
    }
    if (native_flux && (staggering_ != StaggerType::CGrid || terrain_metrics_ != nullptr ||
                                   pressure_coupling != nullptr))
    {
        throw std::invalid_argument("FP32 device mass-flux accumulation requires a flat C-grid");
    }
    fp32_time_currency_ = fp32_time_currency_ || native_flux;

    const double weight = dt_small / dt_large_;
    if (!std::isfinite(weight) || weight <= 0.0)
    {
        finite_ = false;
        throw std::invalid_argument("CartesianCGridTransportContext::accumulate produced an invalid substep weight");
    }

    float*     mfx   = mass_flux_x.data();
    float*     mfy   = mass_flux_y.data();
    float*     mfz   = mass_flux_z.data();
    const auto index = [this](int i, int j, int k)
    {
        return (static_cast<std::size_t>(i) * static_cast<std::size_t>(nth_) + static_cast<std::size_t>(j)) *
                   static_cast<std::size_t>(nz_) +
               static_cast<std::size_t>(k);
    };
    const terrain_transport::mass_flux_detail::CollocatedPressureCouplingView
        mapped_pressure_coupling{
            pressure_coupling != nullptr ? pressure_coupling->pressure : nullptr,
            pressure_coupling != nullptr ? pressure_coupling->base_pressure : nullptr,
            pressure_coupling != nullptr ? pressure_coupling->dt : 0.0};
    const auto* mapped_pressure = pressure_coupling != nullptr
        ? &mapped_pressure_coupling : nullptr;
    std::atomic<bool> all_finite{true};

// Each slot owns its positive-direction face. C-grid velocities already
// live on that face. Collocated velocities are reconstructed to the face,
// with the same explicit wall/periodic contract used by A-grid continuity.
// One worker team spans all three directional passes; the implicit
// barriers preserve the original ordering and floating-point accumulation.
#pragma omp parallel
    {
#pragma omp for collapse(2)
        for (int i = 0; i < nr_ - 1; ++i)
            for (int j = 0; j < nth_; ++j)
                for (int k = 0; k < nz_; ++k)
                {
                    const std::size_t f = index(i, j, k);
                    const double flux   = native_flux
                                              ? native_x_flux(i, j, k)
                                              : (terrain_metrics_ != nullptr
                                                     ? (staggering_ == StaggerType::Collocated
                                                           ? terrain_transport::agrid_mass_flux_x(
                                                                 *terrain_metrics_, rho_state, u_state,
                                                                 i, j, k, execution_state.configuration->lateral_bc_west,
                                                                 execution_state.configuration->lateral_bc_east, mapped_pressure,
                                                                 frame_u_ms_)
                                                           : terrain_transport::cgrid_mass_flux_x(
                                                                 *terrain_metrics_, rho_state, u_state,
                                                                 i, j, k, frame_u_ms_))
                                                     : cartesian_mass_flux::x_face(rho_state,
                                                                                   u_state,
                                                                                   i,
                                                                                   j,
                                                                                   k,
                                                                                   nr_,
                                                                                   staggering_,
                                                                                   execution_state.configuration->lateral_bc_west,
                                                                                   execution_state.configuration->lateral_bc_east,
                                                                                   pressure_coupling,
                                                                                   frame_u_ms_));
                    const double next = compensated_device
                                            ? std::fma(weight, flux, static_cast<double>(mfx[f]))
                                            : static_cast<double>(mfx[f]) + weight * flux;
                    const float  stored = static_cast<float>(next);
                    if (std::isfinite(next) && std::isfinite(stored))
                        mfx[f] = stored;
                    else
                        all_finite.store(false, std::memory_order_relaxed);
                }

#pragma omp for collapse(2)
        for (int i = 0; i < nr_; ++i)
            for (int j = 0; j < nth_ - 1; ++j)
                for (int k = 0; k < nz_; ++k)
                {
                    const std::size_t f = index(i, j, k);
                    const double flux   = native_flux
                                              ? native_y_flux(i, j, k)
                                              : (terrain_metrics_ != nullptr
                                                     ? (staggering_ == StaggerType::Collocated
                                                           ? terrain_transport::agrid_mass_flux_y(
                                                                 *terrain_metrics_, rho_state, v_state,
                                                                 i, j, k, execution_state.configuration->lateral_bc_south,
                                                                 execution_state.configuration->lateral_bc_north, mapped_pressure,
                                                                 frame_v_ms_)
                                                           : terrain_transport::cgrid_mass_flux_y(
                                                                 *terrain_metrics_, rho_state, v_state,
                                                                 i, j, k, frame_v_ms_))
                                                     : cartesian_mass_flux::y_face(rho_state,
                                                                                   v_state,
                                                                                   i,
                                                                                   j,
                                                                                   k,
                                                                                   nth_,
                                                                                   staggering_,
                                                                                   execution_state.configuration->lateral_bc_south,
                                                                                   execution_state.configuration->lateral_bc_north,
                                                                                   pressure_coupling,
                                                                                   frame_v_ms_));
                    const double next = compensated_device
                                            ? std::fma(weight, flux, static_cast<double>(mfy[f]))
                                            : static_cast<double>(mfy[f]) + weight * flux;
                    const float  stored = static_cast<float>(next);
                    if (std::isfinite(next) && std::isfinite(stored))
                        mfy[f] = stored;
                    else
                        all_finite.store(false, std::memory_order_relaxed);
                }

#pragma omp for collapse(2)
        for (int i = terrain_metrics_ != nullptr ? 1 : 0; i < (terrain_metrics_ != nullptr ? nr_ - 1 : nr_); ++i)
            for (int j = terrain_metrics_ != nullptr ? 1 : 0; j < (terrain_metrics_ != nullptr ? nth_ - 1 : nth_); ++j)
                for (int k = 0; k < (staggering_ == StaggerType::CGrid ? nz_ : nz_ - 1); ++k)
                {
                    const std::size_t f = index(i, j, k);
                    const double      flux =
                        native_flux
                            ? native_z_flux(i, j, k)
                            : (terrain_metrics_ != nullptr
                                   ? (staggering_ == StaggerType::Collocated
                                          ? terrain_transport::agrid_mass_flux_zeta(
                                                *terrain_metrics_, rho_state, u_state,
                                                v_state, w_state, i, j, k,
                                                execution_state.configuration->bottom_bc, execution_state.configuration->top_bc,
                                                mapped_pressure, frame_u_ms_, frame_v_ms_)
                                          : terrain_transport::cgrid_mass_flux_zeta(
                                                *terrain_metrics_, rho_state, u_state,
                                                v_state, w_state, i, j, k,
                                                frame_u_ms_, frame_v_ms_))
                                                           : cartesian_mass_flux::z_face(rho_state,
                                                                                         w_state,
                                                                                         i,
                                                                                         j,
                                                                                         k,
                                                                                         nz_,
                                                                                         staggering_,
                                                                                         execution_state.configuration->bottom_bc,
                                                                                         execution_state.configuration->top_bc,
                                                                                         pressure_coupling));
                    const double next = compensated_device
                                            ? std::fma(weight, flux, static_cast<double>(mfz[f]))
                                            : static_cast<double>(mfz[f]) + weight * flux;
                    const float  stored = static_cast<float>(next);
                    if (std::isfinite(next) && std::isfinite(stored))
                        mfz[f] = stored;
                    else
                        all_finite.store(false, std::memory_order_relaxed);
                }
    }

    finite_ = finite_ && all_finite.load(std::memory_order_relaxed);
    if (!finite_)
    {
        throw std::runtime_error("Cartesian transport accumulation produced a non-finite face mass flux");
    }

    if (carrier_kinetic_inputs != nullptr)
    {
        const PhysicalCellDomain domain = physical_scalar_domain(nr_, nth_, nz_, staggering_);
        if (domain.empty() || domain.i_begin < 1 || domain.j_begin < 1)
            throw std::runtime_error("carrier-kinetic boundary attribution has no valid physical domain");

        const double computational_dx  = terrain_metrics_ != nullptr ? terrain_metrics_->dx : execution_state.grid.dr;
        const double computational_dy  = terrain_metrics_ != nullptr ? terrain_metrics_->dy : execution_state.grid.dtheta;
        const double computational_dz  = terrain_metrics_ != nullptr ? terrain_metrics_->dzeta : execution_state.grid.dz;
        const double area_x            = computational_dy * computational_dz;
        const double area_y            = computational_dx * computational_dz;
        const double area_z            = computational_dx * computational_dy;
        const double geometry_values[] = {
            computational_dx,
            computational_dy,
            computational_dz,
            area_x,
            area_y,
            area_z,
        };
        if (std::any_of(std::begin(geometry_values),
                        std::end(geometry_values),
                        [](double value) { return !std::isfinite(value) || value <= 0.0; }))
        {
            throw std::runtime_error("carrier-kinetic boundary attribution has invalid geometry");
        }

        const auto x_flux = [&](int face, int j, int k)
        {
            return native_flux ? native_x_flux(face, j, k)
                                          : (terrain_metrics_ != nullptr
                                                 ? (staggering_ == StaggerType::Collocated
                                                       ? terrain_transport::agrid_mass_flux_x(
                                                             *terrain_metrics_, rho_state, u_state,
                                                             face, j, k, execution_state.configuration->lateral_bc_west,
                                                             execution_state.configuration->lateral_bc_east, mapped_pressure,
                                                             frame_u_ms_)
                                                       : terrain_transport::cgrid_mass_flux_x(
                                                             *terrain_metrics_, rho_state, u_state,
                                                             face, j, k, frame_u_ms_))
                                                 : cartesian_mass_flux::x_face(rho_state,
                                                                               u_state,
                                                                               face,
                                                                               j,
                                                                               k,
                                                                               nr_,
                                                                               staggering_,
                                                                               execution_state.configuration->lateral_bc_west,
                                                                               execution_state.configuration->lateral_bc_east,
                                                                               pressure_coupling,
                                                                               frame_u_ms_));
        };
        const auto y_flux = [&](int i, int face, int k)
        {
            return native_flux ? native_y_flux(i, face, k)
                                          : (terrain_metrics_ != nullptr
                                                 ? (staggering_ == StaggerType::Collocated
                                                       ? terrain_transport::agrid_mass_flux_y(
                                                             *terrain_metrics_, rho_state, v_state,
                                                             i, face, k, execution_state.configuration->lateral_bc_south,
                                                             execution_state.configuration->lateral_bc_north, mapped_pressure,
                                                             frame_v_ms_)
                                                       : terrain_transport::cgrid_mass_flux_y(
                                                             *terrain_metrics_, rho_state, v_state,
                                                             i, face, k, frame_v_ms_))
                                                 : cartesian_mass_flux::y_face(rho_state,
                                                                               v_state,
                                                                               i,
                                                                               face,
                                                                               k,
                                                                               nth_,
                                                                               staggering_,
                                                                               execution_state.configuration->lateral_bc_south,
                                                                               execution_state.configuration->lateral_bc_north,
                                                                               pressure_coupling,
                                                                               frame_v_ms_));
        };
        const auto z_flux = [&](int i, int j, int face)
        {
            return native_flux
                       ? native_z_flux(i, j, face)
                       : (terrain_metrics_ != nullptr
                              ? (staggering_ == StaggerType::Collocated
                                     ? terrain_transport::agrid_mass_flux_zeta(
                                           *terrain_metrics_, rho_state, u_state,
                                           v_state, w_state, i, j, face,
                                           execution_state.configuration->bottom_bc, execution_state.configuration->top_bc,
                                           mapped_pressure, frame_u_ms_, frame_v_ms_)
                                     : terrain_transport::cgrid_mass_flux_zeta(
                                           *terrain_metrics_, rho_state, u_state,
                                           v_state, w_state, i, j, face,
                                           frame_u_ms_, frame_v_ms_))
                                                      : cartesian_mass_flux::z_face(rho_state,
                                                                                    w_state,
                                                                                    i,
                                                                                    j,
                                                                                    face,
                                                                                    nz_,
                                                                                    staggering_,
                                                                                    execution_state.configuration->bottom_bc,
                                                                                    execution_state.configuration->top_bc,
                                                                                    pressure_coupling));
        };
        const auto surface_frame = execution_state.terrain_runtime != nullptr
            && execution_state.terrain_runtime->config.horizontal_placement_id == "earth_fixed"
            ? tmv::domain::HorizontalVelocity{}
            : tmv::domain::HorizontalVelocity{frame_u_ms_, frame_v_ms_};
        const auto coefficient = [&](int i, int j, int k)
        {
            return carrier_kinetic_coefficient_v2(*carrier_kinetic_inputs,
                                                  staggering_,
                                                  terrain_metrics_,
                                                  domain,
                                                  u_state,
                                                  v_state,
                                                  w_state,
                                                  i,
                                                  j,
                                                  k,
                                                  surface_frame.u_ms,
                                                  surface_frame.v_ms);
        };
        using LateralBC           = tmv::init::config::LateralBC;
        const bool west_physical  = execution_state.configuration->lateral_bc_west != LateralBC::Periodic;
        const bool east_physical  = execution_state.configuration->lateral_bc_east != LateralBC::Periodic;
        const bool south_physical = execution_state.configuration->lateral_bc_south != LateralBC::Periodic;
        const bool north_physical = execution_state.configuration->lateral_bc_north != LateralBC::Periodic;
        const int  west_face      = domain.i_begin - 1;
        const int  east_face      = domain.i_end - 1;
        const int  south_face     = domain.j_begin - 1;
        const int  north_face     = domain.j_end - 1;

        const auto x_sums = tmv::diagnostics::fixed_slab_reduce<2>(
            domain.j_begin,
            domain.j_end,
            [&](int j, std::array<double, 2>& partial)
            {
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    if (west_physical)
                    {
                        const double term = -area_x * x_flux(west_face, j, k) * coefficient(domain.i_begin, j, k);
                        if (std::isfinite(term))
                            partial[0] += term;
                        else
                            partial[1] += 1.0;
                    }
                    if (east_physical)
                    {
                        const double term = area_x * x_flux(east_face, j, k) * coefficient(domain.i_end - 1, j, k);
                        if (std::isfinite(term))
                            partial[0] += term;
                        else
                            partial[1] += 1.0;
                    }
                }
            });
        const auto y_sums = tmv::diagnostics::fixed_slab_reduce<2>(
            domain.i_begin,
            domain.i_end,
            [&](int i, std::array<double, 2>& partial)
            {
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    if (south_physical)
                    {
                        const double term = -area_y * y_flux(i, south_face, k) * coefficient(i, domain.j_begin, k);
                        if (std::isfinite(term))
                            partial[0] += term;
                        else
                            partial[1] += 1.0;
                    }
                    if (north_physical)
                    {
                        const double term = area_y * y_flux(i, north_face, k) * coefficient(i, domain.j_end - 1, k);
                        if (std::isfinite(term))
                            partial[0] += term;
                        else
                            partial[1] += 1.0;
                    }
                }
            });
        const auto z_sums = tmv::diagnostics::fixed_slab_reduce<2>(
            domain.i_begin,
            domain.i_end,
            [&](int i, std::array<double, 2>& partial)
            {
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                {
                    if (staggering_ == StaggerType::Collocated)
                    {
                        const int    bottom_face = domain.k_begin - 1;
                        const double bottom_term =
                            -area_z * z_flux(i, j, bottom_face) * coefficient(i, j, domain.k_begin);
                        if (std::isfinite(bottom_term))
                            partial[0] += bottom_term;
                        else
                            partial[1] += 1.0;
                    }
                    const int    top_face = domain.k_end - 1;
                    const double top_term = area_z * z_flux(i, j, top_face) * coefficient(i, j, domain.k_end - 1);
                    if (std::isfinite(top_term))
                        partial[0] += top_term;
                    else
                        partial[1] += 1.0;
                }
            });

        const double invalid             = x_sums[1] + y_sums[1] + z_sums[1];
        const double substep_transport_j = dt_small * (x_sums[0] + y_sums[0] + z_sums[0]);
        if (invalid != 0.0 || !std::isfinite(substep_transport_j))
        {
            finite_ = false;
            throw std::runtime_error("carrier-kinetic boundary attribution produced invalid work");
        }
        outward_carrier_kinetic_transport_j_ += substep_transport_j;
        if (!std::isfinite(outward_carrier_kinetic_transport_j_))
        {
            finite_ = false;
            throw std::overflow_error("carrier-kinetic boundary attribution overflowed");
        }
        ++carrier_kinetic_substep_count_;
    }
    accumulated_dt_ += dt_small;
    ++substep_count_;
}

CartesianCGridTransportContext::PreparedExternalAcousticLoop
CartesianCGridTransportContext::prepare_external_acoustic_loop(
    const Field3D& density, double total_dt, int substeps)
{
    if (!active_ || complete_ || !finite_ || substep_count_ != 0 || accumulated_dt_ != 0 ||
        density_begin_captured_ || carrier_kinetic_tracking_ || terrain_metrics_ ||
        staggering_ != StaggerType::CGrid || !std::isfinite(total_dt) || total_dt <= 0 ||
        substeps < 1 || substeps > 4096 ||
        std::abs(total_dt - dt_large_) > 4 * std::numeric_limits<float>::epsilon() * dt_large_)
        throw std::invalid_argument("external acoustic loop requires one fresh flat C-grid transport interval");
    for (const auto* field : std::array<const Field3D*, 4>{&density, &mass_flux_x, &mass_flux_y, &mass_flux_z})
        if (field->size_r() != nr_ || field->size_th() != nth_ || field->size_z() != nz_)
            throw std::invalid_argument("external acoustic loop transport shape mismatch");
    int valid = 1;
    #pragma omp parallel for simd reduction(&:valid) if(density.size() >= 65536)
    for (std::size_t n = 0; n < density.size(); ++n)
        valid &= std::isfinite(density.data()[n]) && density.data()[n] > 0;
    if (!valid)
        throw std::invalid_argument("external acoustic loop beginning density must be finite and positive");
    return PreparedExternalAcousticLoop(*this, density, total_dt, substeps);
}

void CartesianCGridTransportContext::PreparedExternalAcousticLoop::commit() noexcept
{
    if (!owner_) return;
    owner_->density_begin = std::move(density_);
    owner_->density_begin_captured_ = true;
    owner_->accumulated_dt_ = total_dt_;
    owner_->substep_count_ = substeps_;
    owner_->fp32_time_currency_ = true;
    owner_ = nullptr;
}

void CartesianCGridTransportContext::record_external_accumulation(double total_dt,
                                                                  int    substeps,
                                                                  bool   fp32_time_currency)
{
    if (!active_ || complete_)
    {
        throw std::logic_error(
            "CartesianCGridTransportContext::record_external_accumulation requires an active, unsealed context");
    }
    if (!std::isfinite(total_dt) || total_dt <= 0.0 || substeps <= 0)
    {
        finite_ = false;
        throw std::invalid_argument(
            "CartesianCGridTransportContext::record_external_accumulation received invalid metadata");
    }
    // Externally accumulated fluxes already include mapped geometry and the
    // configured grid-frame velocity; this method records their time currency
    // without reconstructing either on the host.

    long long       nonfinite_count = 0;
    const long long point_count     = static_cast<long long>(mass_flux_x.size());
    const float*    mfx             = mass_flux_x.data();
    const float*    mfy             = mass_flux_y.data();
    const float*    mfz             = mass_flux_z.data();
#pragma omp parallel for reduction(+ : nonfinite_count)
    for (long long n = 0; n < point_count; ++n)
    {
        if (!std::isfinite(static_cast<double>(mfx[n])))
            ++nonfinite_count;
        if (!std::isfinite(static_cast<double>(mfy[n])))
            ++nonfinite_count;
        if (!std::isfinite(static_cast<double>(mfz[n])))
            ++nonfinite_count;
    }
    if (nonfinite_count != 0)
    {
        finite_ = false;
        throw std::runtime_error("accelerator exported non-finite Cartesian C-grid mass fluxes");
    }
    accumulated_dt_ += total_dt;
    substep_count_ += substeps;
    fp32_time_currency_ = fp32_time_currency_ || fp32_time_currency;
}

void CartesianCGridTransportContext::record_external_carrier_kinetic_transport(double outward_j, int substeps)
{
    if (!active_ || complete_)
        throw std::logic_error("external carrier-kinetic attribution requires an active, unsealed context");
    if (substeps <= 0 || substeps > std::numeric_limits<int>::max() - carrier_kinetic_substep_count_ ||
        carrier_kinetic_substep_count_ != substep_count_)
        throw std::logic_error("external carrier-kinetic attribution is out of sequence");
    if (!std::isfinite(outward_j))
    {
        finite_ = false;
        throw std::invalid_argument("external carrier-kinetic attribution must be finite");
    }
    const double next = outward_carrier_kinetic_transport_j_ + outward_j;
    if (!std::isfinite(next))
    {
        finite_ = false;
        throw std::overflow_error("external carrier-kinetic attribution overflowed");
    }
    outward_carrier_kinetic_transport_j_ = next;
    carrier_kinetic_tracking_            = true;
    carrier_kinetic_substep_count_ += substeps;
}

void CartesianCGridTransportContext::complete()
{
    return complete(global_simulation_state());
}

void CartesianCGridTransportContext::complete(const SimulationStateView& execution_state)
{
    if (!active_)
    {
        throw std::logic_error("CartesianCGridTransportContext::complete called before begin");
    }
    if (static_cast<int>(execution_state.configuration->lateral_bc_west) != lateral_bc_west_ ||
        static_cast<int>(execution_state.configuration->lateral_bc_east) != lateral_bc_east_ ||
        static_cast<int>(execution_state.configuration->lateral_bc_south) != lateral_bc_south_ ||
        static_cast<int>(execution_state.configuration->lateral_bc_north) != lateral_bc_north_ ||
        static_cast<int>(execution_state.configuration->bottom_bc) != bottom_bc_ || static_cast<int>(execution_state.configuration->top_bc) != top_bc_)
    {
        boundary_flux_compatible_ = false;
        throw std::runtime_error("Cartesian transport boundary policy changed while retaining "
                                 "acoustic mass flux");
    }

    // A retained C-grid flux is the authority shared by continuity and every
    // scalar. Reject an impermeable face that carries nonzero mass rather
    // than letting the CPU and device scalar kernels interpret the
    // contradictory context differently. This scan is paid once when the
    // context is sealed, not once per transported scalar.
    if (staggering_ == StaggerType::CGrid)
    {
        using tmv::init::config::LateralBC;
        using tmv::init::config::TopBC;
        const auto lateral_is_rigid = [](int value)
        { return value == static_cast<int>(LateralBC::Rigid) || value == static_cast<int>(LateralBC::RigidSlip); };
        const auto reject_nonzero = [this](float value)
        {
            if (value != 0.0f)
            {
                boundary_flux_compatible_ = false;
                throw std::runtime_error("Cartesian retained mass flux crosses an impermeable "
                                         "C-grid boundary");
            }
        };

        if (lateral_is_rigid(lateral_bc_west_))
            for (int j = 1; j < nth_ - 1; ++j)
                for (int k = 0; k < nz_; ++k)
                    reject_nonzero(mass_flux_x[0][j][k]);
        if (lateral_is_rigid(lateral_bc_east_))
            for (int j = 1; j < nth_ - 1; ++j)
                for (int k = 0; k < nz_; ++k)
                    reject_nonzero(mass_flux_x[nr_ - 2][j][k]);
        if (lateral_is_rigid(lateral_bc_south_))
            for (int i = 1; i < nr_ - 1; ++i)
                for (int k = 0; k < nz_; ++k)
                    reject_nonzero(mass_flux_y[i][0][k]);
        if (lateral_is_rigid(lateral_bc_north_))
            for (int i = 1; i < nr_ - 1; ++i)
                for (int k = 0; k < nz_; ++k)
                    reject_nonzero(mass_flux_y[i][nth_ - 2][k]);
        if (top_bc_ == static_cast<int>(TopBC::Rigid) || top_bc_ == static_cast<int>(TopBC::RigidWithSponge))
        {
            for (int i = 1; i < nr_ - 1; ++i)
                for (int j = 1; j < nth_ - 1; ++j)
                    reject_nonzero(mass_flux_z[i][j][nz_ - 1]);
        }
    }
    complete_ = true;
}

DynamicsBoundaryMassReport CartesianCGridTransportContext::boundary_mass_report(const Field3D& density_end,
                                                                                double         dx_m,
                                                                                double         dy_m,
                                                                                double         dz_m) const
{
    return boundary_mass_report(global_simulation_state(), density_end, dx_m, dy_m, dz_m);
}

DynamicsBoundaryMassReport CartesianCGridTransportContext::boundary_mass_report(const SimulationStateView& execution_state, const Field3D& density_end,
                                                                                double         dx_m,
                                                                                double         dy_m,
                                                                                double         dz_m) const
{
    require_valid(execution_state, nr_, nth_, nz_, dt_large_);
    if (density_end.size_r() != nr_ || density_end.size_th() != nth_ || density_end.size_z() != nz_)
    {
        throw std::invalid_argument("Cartesian boundary mass report density shape mismatch");
    }
    require_positive_finite(dx_m, "Cartesian boundary mass report dx");
    require_positive_finite(dy_m, "Cartesian boundary mass report dy");
    require_positive_finite(dz_m, "Cartesian boundary mass report dz");

    const PhysicalCellDomain domain = physical_scalar_domain(nr_, nth_, nz_, staggering_);
    if (domain.empty())
    {
        throw std::runtime_error("Cartesian boundary mass report has an empty physical domain");
    }
    if (domain.i_begin < 1 || domain.j_begin < 1)
    {
        throw std::runtime_error("Cartesian boundary mass report requires horizontal boundary-face storage");
    }

    long double       outward_rate_kg_s = 0.0L;
    const double      computational_dx  = terrain_metrics_ != nullptr ? terrain_metrics_->dx : dx_m;
    const double      computational_dy  = terrain_metrics_ != nullptr ? terrain_metrics_->dy : dy_m;
    const double      computational_dz  = terrain_metrics_ != nullptr ? terrain_metrics_->dzeta : dz_m;
    const long double area_x            = static_cast<long double>(computational_dy) * computational_dz;
    const long double area_y            = static_cast<long double>(computational_dx) * computational_dz;
    const long double area_z            = static_cast<long double>(computational_dx) * computational_dy;

    const int west_face = domain.i_begin - 1;
    const int east_face = domain.i_end - 1;
    for (int j = domain.j_begin; j < domain.j_end; ++j)
        for (int k = domain.k_begin; k < domain.k_end; ++k)
            outward_rate_kg_s += area_x * (static_cast<long double>(mass_flux_x[east_face][j][k]) -
                                           static_cast<long double>(mass_flux_x[west_face][j][k]));

    const int south_face = domain.j_begin - 1;
    const int north_face = domain.j_end - 1;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int k = domain.k_begin; k < domain.k_end; ++k)
            outward_rate_kg_s += area_y * (static_cast<long double>(mass_flux_y[i][north_face][k]) -
                                           static_cast<long double>(mass_flux_y[i][south_face][k]));

    // A-grid density has vertical ghost cells, so its retained k=0 and
    // k=NZ-2 slots are the bottom/top physical faces. C-grid's lower face is
    // impermeable and implicit; its k=NZ-1 slot is the explicit top face.
    if (staggering_ == StaggerType::Collocated)
    {
        if (domain.k_begin < 1)
        {
            throw std::runtime_error(
                "collocated Cartesian boundary mass report requires vertical boundary-face storage");
        }
        const int bottom_face = domain.k_begin - 1;
        const int top_face    = domain.k_end - 1;
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                outward_rate_kg_s += area_z * (static_cast<long double>(mass_flux_z[i][j][top_face]) -
                                               static_cast<long double>(mass_flux_z[i][j][bottom_face]));
    }
    else
    {
        const int top_face = domain.k_end - 1;
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                outward_rate_kg_s += area_z * static_cast<long double>(mass_flux_z[i][j][top_face]);
    }

    long double       committed_change_kg = 0.0L;
    const long double computational_cell_volume =
        static_cast<long double>(computational_dx) * computational_dy * computational_dz;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double end   = static_cast<double>(density_end[i][j][k]);
                const double begin = static_cast<double>(density_begin[i][j][k]);
                if (!std::isfinite(end) || end <= 0.0)
                {
                    throw std::runtime_error("Cartesian boundary mass report requires finite positive ending density");
                }
                const long double jacobian =
                    terrain_metrics_ != nullptr ? static_cast<long double>(terrain_metrics_->J(i, j, k)) : 1.0L;
                committed_change_kg += has_moving_geometry()
                    ? computational_cell_volume * (jacobian * end - beginning_jacobian(i,j,k) * begin)
                    : computational_cell_volume * jacobian * (end - begin);
            }

    DynamicsBoundaryMassReport report;
    report.outward_boundary_mass_kg     = static_cast<double>(outward_rate_kg_s * static_cast<long double>(dt_large_));
    report.flux_implied_state_change_kg = -report.outward_boundary_mass_kg;
    report.committed_state_change_kg    = static_cast<double>(committed_change_kg);
    report.closure_residual_kg          = report.committed_state_change_kg - report.flux_implied_state_change_kg;
    if (!std::isfinite(report.outward_boundary_mass_kg) || !std::isfinite(report.committed_state_change_kg) ||
        !std::isfinite(report.closure_residual_kg))
    {
        throw std::runtime_error("Cartesian boundary mass report produced a non-finite ledger");
    }
    return report;
}

bool CartesianCGridTransportContext::validate(int nr, int nth, int nz, double dt_large, std::string* reason) const
{
    return validate(global_simulation_state(), nr, nth, nz, dt_large, reason);
}

bool CartesianCGridTransportContext::validate(const SimulationStateView& execution_state, int nr, int nth, int nz, double dt_large, std::string* reason) const
{
    const auto fail = [reason](const char* message)
    {
        if (reason != nullptr)
            *reason = message;
        return false;
    };
    if (!active_)
        return fail("context was not begun");
    if (!complete_)
        return fail("context was not sealed complete");
    if (!finite_)
        return fail("context contains a non-finite mass flux");
    if (!boundary_flux_compatible_)
        return fail("context mass flux contradicts an impermeable boundary");
    if (!density_begin_captured_)
        return fail("context beginning density was not captured");
    if (nr != nr_ || nth != nth_ || nz != nz_)
        return fail("context domain does not match scalar domain");
    if (staggering_ != execution_state.configuration->stagger_type)
        return fail("context staggering does not match active scalar staggering");
    if (static_cast<int>(execution_state.configuration->lateral_bc_west) != lateral_bc_west_ ||
        static_cast<int>(execution_state.configuration->lateral_bc_east) != lateral_bc_east_ ||
        static_cast<int>(execution_state.configuration->lateral_bc_south) != lateral_bc_south_ ||
        static_cast<int>(execution_state.configuration->lateral_bc_north) != lateral_bc_north_ ||
        static_cast<int>(execution_state.configuration->bottom_bc) != bottom_bc_ || static_cast<int>(execution_state.configuration->top_bc) != top_bc_)
        return fail("context boundary policy does not match active policy");
    if (!std::isfinite(dt_large) || dt_large <= 0.0)
        return fail("requested scalar timestep is invalid");
    const double dt_scale     = std::max({1.0, std::abs(dt_large_), std::abs(dt_large)});
    const double dt_tolerance = 64.0 * std::numeric_limits<double>::epsilon() * dt_scale;
    if (std::abs(dt_large - dt_large_) > dt_tolerance)
        return fail("context large timestep does not match scalar timestep");
    if (substep_count_ <= 0)
        return fail("context contains no acoustic substeps");
    // Device acoustic kernels consume one FP32 timestep per substep. The
    // retained flux is intentionally weighted by that executed duration so
    // scalar transport closes against the device continuity trajectory.
    // For N positive substeps whose exact durations sum to dt_large, the sum
    // of round-to-nearest FP32 conversions is bounded by roughly
    // 0.5*epsilon_float*dt_large. Use a factor of two for accumulation
    // roundoff while remaining many orders below one omitted CFL substep.
    const double coverage_tolerance =
        fp32_time_currency_
            ? std::max(dt_tolerance, 2.0 * static_cast<double>(std::numeric_limits<float>::epsilon()) * dt_scale)
            : dt_tolerance;
    if (!std::isfinite(accumulated_dt_) || std::abs(accumulated_dt_ - dt_large_) > coverage_tolerance)
        return fail("context acoustic substeps do not cover the large timestep");
    const auto matches = [this](const Field3D& field)
    { return field.size_r() == nr_ && field.size_th() == nth_ && field.size_z() == nz_; };
    if (!matches(density_begin) || !matches(mass_flux_x) || !matches(mass_flux_y) || !matches(mass_flux_z))
        return fail("context mass-flux field shape mismatch");
    if (!jacobian_begin_.empty() && (terrain_metrics_ == nullptr || jacobian_begin_.size() != density_begin.size()))
        return fail("context beginning geometry is inconsistent");
    if (terrain_metrics_ != nullptr)
    {
        try
        {
            terrain_transport::validate_geometry_shape(*terrain_metrics_);
        }
        catch (const std::exception&)
        {
            return fail("context terrain geometry is invalid");
        }
        if (terrain_metrics_->J.size_r() != nr_ || terrain_metrics_->J.size_th() != nth_ ||
            terrain_metrics_->J.size_z() != nz_)
            return fail("context terrain geometry shape mismatch");
    }
    if (reason != nullptr)
        reason->clear();
    return true;
}

void CartesianCGridTransportContext::require_valid(int nr, int nth, int nz, double dt_large) const
{
    return require_valid(global_simulation_state(), nr, nth, nz, dt_large);
}

void CartesianCGridTransportContext::require_valid(const SimulationStateView& execution_state, int nr, int nth, int nz, double dt_large) const
{
    std::string reason;
    if (!validate(execution_state, nr, nth, nz, dt_large, &reason))
    {
        throw std::runtime_error("invalid Cartesian C-grid transport context: " + reason);
    }
}

/**
 * @brief Cartesian first-order upwind advection in the x direction.
 */
void advect_scalar_1d_x_kernel_cartesian(const Field3D& src, Field3D& dst, double dt)
{
    return advect_scalar_1d_x_kernel_cartesian(global_simulation_state(), src, dst, dt);
}

void advect_scalar_1d_x_kernel_cartesian(const SimulationStateView& execution_state, const Field3D& src, Field3D& dst, double dt)
{
    const int NR  = execution_state.grid.nr;
    const int NTH = execution_state.grid.nth;
    const int NZ  = execution_state.grid.nz;
    require_global_field_shape(execution_state, src, "Cartesian x-advection source");
    require_global_field_shape(execution_state, execution_state.fields.u, "Cartesian x velocity");
    require_positive_finite(dt, "Cartesian x-advection timestep");
    const double dx = require_positive_finite(execution_state.grid.dr, "Cartesian dx");
    seed_destination_from_source(execution_state, src, dst);

    if (NR < 3 || NTH < 1 || NZ < 3)
    {
        // No interior to update — the seed already gives us the
        // zero-gradient extrapolation on every face. Do nothing on
        // degenerate grids.
        return;
    }

    const float* src_data = src.data();
    const float* u_data   = execution_state.fields.u.data();
    float*       dst_data = dst.data();

    const double inv_dx = 1.0 / dx;

#pragma omp parallel for collapse(2)
    for (int i = 1; i < NR - 1; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 1; k < NZ - 1; ++k)
            {
                const std::size_t c  = idx3(execution_state, i, j, k);
                const std::size_t im = idx3(execution_state, i - 1, j, k);
                const std::size_t ip = idx3(execution_state, i + 1, j, k);

                const double u_val = static_cast<double>(u_data[c]);
                double       dq_dx = 0.0;
                if (u_val > 0.0)
                {
                    dq_dx = (static_cast<double>(src_data[c]) - static_cast<double>(src_data[im])) * inv_dx;
                }
                else if (u_val < 0.0)
                {
                    dq_dx = (static_cast<double>(src_data[ip]) - static_cast<double>(src_data[c])) * inv_dx;
                }

                const double q = static_cast<double>(src_data[c]);
                dst_data[c]    = static_cast<float>(q - dt * u_val * dq_dx);
            }
        }
    }
}

/**
 * @brief Cartesian first-order upwind advection in the y direction.
 */
void advect_scalar_1d_y_kernel_cartesian(const Field3D& src, Field3D& dst, double dt)
{
    return advect_scalar_1d_y_kernel_cartesian(global_simulation_state(), src, dst, dt);
}

void advect_scalar_1d_y_kernel_cartesian(const SimulationStateView& execution_state, const Field3D& src, Field3D& dst, double dt)
{
    const int NR  = execution_state.grid.nr;
    const int NTH = execution_state.grid.nth;
    const int NZ  = execution_state.grid.nz;
    require_global_field_shape(execution_state, src, "Cartesian y-advection source");
    require_global_field_shape(execution_state, execution_state.fields.v, "Cartesian y velocity");
    require_positive_finite(dt, "Cartesian y-advection timestep");
    const double dy = require_positive_finite(execution_state.grid.dtheta, "Cartesian dy");
    seed_destination_from_source(execution_state, src, dst);

    if (NR < 1 || NTH < 3 || NZ < 3)
    {
        return;
    }

    const float* src_data = src.data();
    const float* v_data   = execution_state.fields.v.data();
    float*       dst_data = dst.data();

    const double inv_dy = 1.0 / dy;

#pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 1; j < NTH - 1; ++j)
        {
            for (int k = 1; k < NZ - 1; ++k)
            {
                const std::size_t c  = idx3(execution_state, i, j, k);
                const std::size_t jm = idx3(execution_state, i, j - 1, k);
                const std::size_t jp = idx3(execution_state, i, j + 1, k);

                const double v_val = static_cast<double>(v_data[c]);
                double       dq_dy = 0.0;
                if (v_val > 0.0)
                {
                    dq_dy = (static_cast<double>(src_data[c]) - static_cast<double>(src_data[jm])) * inv_dy;
                }
                else if (v_val < 0.0)
                {
                    dq_dy = (static_cast<double>(src_data[jp]) - static_cast<double>(src_data[c])) * inv_dy;
                }

                const double q = static_cast<double>(src_data[c]);
                dst_data[c]    = static_cast<float>(q - dt * v_val * dq_dy);
            }
        }
    }
}

/** Compatibility boundary for the removed transport-chained point filter. */
void apply_diffusion_kernel_cartesian(const Field3D& src, Field3D& dst, double dt, double kappa)
{
    return apply_diffusion_kernel_cartesian(global_simulation_state(), src, dst, dt, kappa);
}

void apply_diffusion_kernel_cartesian(const SimulationStateView& execution_state, const Field3D& src, Field3D& dst, double dt, double kappa)
{
    require_global_field_shape(execution_state, src, "Cartesian diffusion source");
    require_positive_finite(dt, "Cartesian diffusion timestep");
    if (!std::isfinite(kappa) || kappa < 0.0)
    {
        throw std::invalid_argument("Cartesian diffusion coefficient must be finite and non-negative");
    }
    if (kappa > 0.0)
    {
        throw std::invalid_argument("transport-chained Cartesian point diffusion is unavailable; "
                                    "use numerics.diffusion for live-density flux-form diffusion");
    }
    const double dx      = require_positive_finite(execution_state.grid.dr, "Cartesian dx");
    const double dy      = require_positive_finite(execution_state.grid.dtheta, "Cartesian dy");
    const double dz_safe = require_positive_finite(execution_state.grid.dz, "Cartesian dz");
    // The retained zero-kappa call boundary avoids churn in the transport
    // scheduler while making the former unweighted Laplacian unreachable.
    seed_destination_from_source(execution_state, src, dst);
    (void)dx;
    (void)dy;
    (void)dz_safe;
}

/**
 * @brief Conservative rho-weighted flux-form scalar transport (Cartesian C-grid).
 *
 * Advances rho*phi in flux form using the SAME face mass fluxes as the mass
 * continuity operator (derivative_operators.hpp::div_flux_mass_*). Standalone
 * calls recover against their predicted rho_check. Coupled calls consume the
 * acoustic loop's averaged flux context and captured rho_begin, then recover
 * against retained rho_end. Because the scalar face flux
 * is the mass flux times the limited upwind face value of phi, a spatially
 * constant phi is preserved to round-off and there is no spurious
 * -phi*div(u) source. The upwind face value phi_face is
 * reconstructed with WENO5 (5th-order, smooth-extremum-preserving) wherever the
 * 5-point stencil fits, falling back to the MC limiter in the two cells nearest
 * each boundary. WENO5 (unlike the MC/TVD limiter, which clips smooth peaks to
 * 1st-order) advects a resolved thermal without diffusing its peak away.
 *
 * C-grid face convention matches div_flux_mass_*. A coupled collocated call
 * consumes the already reconstructed A-grid face flux history. C-grid cells
 * update (1..NR-2, 1..NTH-2, all k); collocated cells update the full interior
 * (1..NR-2 in x/y/z), matching their respective continuity ownership exactly.
 *
 * Reads globals rho, u, v, w. Field-specific clamping (theta bounds, hydrometeor
 * non-negativity) is applied by the caller after this returns.
 */
struct AuthoredMappedScalar
{
    const float* scalar;
    const float* x;
    const float* y;
    const float* z;
    const float* limiter;
    const float* requested = nullptr;
    const float* requested_loss = nullptr;
    bool applied = false;
};

static void advect_scalar_conservative_3d_cartesian_impl(const SimulationStateView&            execution_state,
                                                         const Field3D&                        src,
                                                         Field3D&                              dst,
                                                         double                                dt,
                                                         bool                                  positive_definite,
                                                         const CartesianCGridTransportContext* context,
                                                         ScalarBoundaryTransportReport*        boundary_report,
                                                         ScalarMoistEnergyKind                 energy_kind,
                                                         ScalarMoistEnergyTransportReportV1*   energy_report,
                                                         bool                                  defer_report_inventories,
                                                         CartesianFaceReconstruction           reconstruction,
                                                         CartesianInterfaceFluxRegister*       interface_flux_register,
                                                         bool                        mass_tendency_only = false,
                                                         const AuthoredMappedScalar* authored           = nullptr)
{
    if (preserves_positive_support(reconstruction) && !positive_definite)
        throw std::invalid_argument("support-preserving reconstruction requires a positive scalar");
    if (context != nullptr && context->has_moving_geometry())
    {
        if (mass_tendency_only)
            throw std::invalid_argument("moving scalar tendencies require an explicit reference volume");
        if (execution_state.compute_runtime != nullptr &&
            execution_state.compute_runtime->active_kind != ComputeBackendKind::Cpu)
            throw std::invalid_argument("moving scalar transport requires CPU-owned execution");
    }
    const int NR  = execution_state.grid.nr;
    const int NTH = execution_state.grid.nth;
    const int NZ  = execution_state.grid.nz;
    require_global_field_shape(execution_state, src, "Cartesian conservative scalar source");
    require_global_field_shape(execution_state, execution_state.fields.rho, "Cartesian density");
    require_global_field_shape(execution_state, execution_state.fields.u, "Cartesian x velocity");
    require_global_field_shape(execution_state, execution_state.fields.v, "Cartesian y velocity");
    require_global_field_shape(execution_state, execution_state.fields.w, "Cartesian vertical velocity");
    require_positive_finite(dt, "Cartesian conservative-advection timestep");
    const double inv_dx = 1.0 / require_positive_finite(execution_state.grid.dr, "Cartesian dx");
    const double inv_dy = 1.0 / require_positive_finite(execution_state.grid.dtheta, "Cartesian dy");
    const double inv_dz = 1.0 / require_positive_finite(execution_state.grid.dz, "Cartesian dz");

    const bool coupled_to_retained_density = context != nullptr;
    if ((energy_report == nullptr) != (energy_kind == ScalarMoistEnergyKind::None))
    {
        throw std::invalid_argument("Cartesian scalar moist-energy reporting requires exactly one "
                                    "non-null report and non-None scalar energy kind");
    }
    if (boundary_report != nullptr)
    {
        *boundary_report = {};
        if (!coupled_to_retained_density)
        {
            throw std::invalid_argument("Cartesian scalar boundary reporting requires a retained mass-flux context");
        }
    }
    if (energy_report != nullptr)
    {
        *energy_report = {};
        if (!coupled_to_retained_density)
        {
            throw std::invalid_argument(
                "Cartesian scalar moist-energy reporting requires a retained mass-flux context");
        }
        require_global_field_shape(execution_state, execution_state.fields.p, "Cartesian pressure");
        require_global_field_shape(execution_state, execution_state.fields.theta, "Cartesian potential temperature");
        require_global_field_shape(execution_state, execution_state.fields.qv, "Cartesian water vapor");
        require_global_field_shape(execution_state, execution_state.fields.qc, "Cartesian cloud water");
        require_global_field_shape(execution_state, execution_state.fields.qr, "Cartesian rain water");
        require_global_field_shape(execution_state, execution_state.fields.qi, "Cartesian cloud ice");
        require_global_field_shape(execution_state, execution_state.fields.qs, "Cartesian snow");
        require_global_field_shape(execution_state, execution_state.fields.qg, "Cartesian graupel");
        require_global_field_shape(execution_state, execution_state.fields.qh, "Cartesian hail");
    }
    if (coupled_to_retained_density)
    {
        // Fail closed before touching any scalar output: a partial or stale
        // flux history would silently reintroduce the density time-level bug.
        context->require_valid(execution_state, NR, NTH, NZ, dt);
    }
    if (NR < 3 || NTH < 3 || NZ < 2)
    {
        if (boundary_report != nullptr || energy_report != nullptr)
        {
            throw std::runtime_error("Cartesian scalar reporting requires a non-empty physical domain");
        }
        seed_destination_from_source(execution_state, src, dst);
        return;
    }
    const StaggerType active_transport_staggering =
        coupled_to_retained_density ? context->staggering() : execution_state.configuration->stagger_type;
    const TerrainMetrics3D*  terrain_metrics = coupled_to_retained_density ? context->terrain_metrics() : nullptr;
    const bool               cgrid_open_top  = active_transport_staggering == StaggerType::CGrid &&
                                               execution_state.configuration->top_bc == tmv::init::config::TopBC::Open;
    const PhysicalCellDomain report_domain   = physical_scalar_domain(NR, NTH, NZ, active_transport_staggering);
    if (interface_flux_register != nullptr)
    {
        if (!coupled_to_retained_density || active_transport_staggering != StaggerType::CGrid)
        {
            throw std::invalid_argument("scalar interface flux capture requires retained C-grid transport");
        }
        interface_flux_register->require_compatible(NR, NTH, NZ);
    }

    // Energy coefficients are sampled at the owning scalar commit. Validate
    // the entire coefficient field before seeding or mutating the destination,
    // so malformed direct calls retain the batch's fail-before-mutation
    // contract.
    if (energy_report != nullptr)
    {
        const bool timing = advection_detail::perf_enabled();
        const auto started = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (report_domain.empty() || report_domain.i_begin < 1 || report_domain.j_begin < 1)
        {
            throw std::runtime_error("Cartesian scalar moist-energy report has an empty physical domain");
        }
        const auto valid_coefficients = [&](int i, int j, int k)
        {
            const auto coefficients = scalar_moist_energy_coefficients_v2(execution_state,
                                                                          energy_kind,
                                                                          active_transport_staggering,
                                                                          terrain_metrics,
                                                                          report_domain,
                                                                          i, j, k);
            return std::all_of(coefficients.begin(), coefficients.end(),
                               [](double value) { return std::isfinite(value); });
        };
        if (terrain_metrics == nullptr)
        {
            int valid = 1;
            // Flat coefficients cannot throw. Complete the read-only check
            // before any destination mutation; only the validity bit is reduced.
#pragma omp parallel for schedule(static) reduction(&:valid) if(src.size() >= 65536)
            for (int i = report_domain.i_begin; i < report_domain.i_end; ++i)
                for (int j = report_domain.j_begin; j < report_domain.j_end; ++j)
                    for (int k = report_domain.k_begin; k < report_domain.k_end; ++k)
                        valid &= valid_coefficients(i, j, k);
            if (!valid)
                throw std::runtime_error("Cartesian scalar moist-energy report rejected "
                                         "a non-finite operator-time coefficient");
        }
        else
        {
            for (int i = report_domain.i_begin; i < report_domain.i_end; ++i)
                for (int j = report_domain.j_begin; j < report_domain.j_end; ++j)
                    for (int k = report_domain.k_begin; k < report_domain.k_end; ++k)
                        if (!valid_coefficients(i, j, k))
                            throw std::runtime_error("Cartesian scalar moist-energy report rejected "
                                                     "a non-finite operator-time coefficient");
        }
        if (timing)
            advection_detail::perf_totals().host_coefficient_validation_s +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }
    const bool native_mapped =
        authored == nullptr && !mass_tendency_only && terrain_metrics != nullptr && execution_state.configuration &&
        (execution_state.configuration->admission.earth_fixed_terrain_motion ||
         execution_state.configuration->admission.two_way_accelerator_reference) && execution_state.compute_runtime &&
        execution_state.compute_runtime->active_kind != ComputeBackendKind::Cpu;
    if (native_mapped)
    {
        auto& compute = *execution_state.compute_runtime;
        auto* kernels = compute.backend ? compute.backend->terrain_kernels() : nullptr;
        if (!context || compute.fallback_active || !kernels || !kernels->supports_mapped_scalar_transport())
            throw std::runtime_error("native mapped scalar transport requires its own complete device path");
        Field3D            candidate(NR, NTH, NZ, 0);
        std::vector<float> jacobian(src.size());
        for (std::size_t n = 0; n < jacobian.size(); ++n)
            jacobian[n] = float(terrain_metrics->J.values[n]);
        struct Callback
        {
            const SimulationStateView*            state;
            const Field3D*                        src;
            Field3D*                              dst;
            double                                dt;
            bool                                  positive;
            const CartesianCGridTransportContext* context;
            ScalarBoundaryTransportReport*        boundary;
            ScalarMoistEnergyKind                 kind;
            ScalarMoistEnergyTransportReportV1*   energy;
            bool                                  defer;
            CartesianFaceReconstruction           reconstruction;
            CartesianInterfaceFluxRegister* interface;
        } callback{&execution_state,
                   &src,
                   &candidate,
                   dt,
                   positive_definite,
                   context,
                   boundary_report,
                   energy_kind,
                   energy_report,
                   defer_report_inventories,
                   reconstruction, interface_flux_register};
        MappedTerrainScalarTransportReport   report;
        MappedTerrainScalarTransportDispatch d;
        d.scalar_begin               = src.data();
        d.density_begin              = context->density_begin.data();
        d.density_end                = execution_state.fields.rho.data();
        d.mass_flux_x                = context->mass_flux_x.data();
        d.mass_flux_y                = context->mass_flux_y.data();
        d.mass_flux_zeta             = context->mass_flux_z.data();
        d.jacobian                   = jacobian.data();
        d.scalar_end                 = candidate.data();
        d.nx                         = NR;
        d.ny                         = NTH;
        d.nz                         = NZ;
        d.positive_definite          = positive_definite;
        d.weno                       = uses_weno(reconstruction);
        d.preserve_positive_support  = preserves_positive_support(reconstruction);
        d.dx                         = float(terrain_metrics->dx);
        d.dy                         = float(terrain_metrics->dy);
        d.dzeta                      = float(terrain_metrics->dzeta);
        d.dt                         = float(dt);
        d.density_floor              = density_min_kgm3;
        d.report                     = &report;
        d.applied_observer_context = &callback;
        d.applied_observer =
            +[](void* raw, const float* scalar, const float* x, const float* y, const float* z, const float* requested, const float* loss)
        {
            auto&                      c = *static_cast<Callback*>(raw);
            const AuthoredMappedScalar values{scalar, x, y, z, nullptr, requested, loss, true};
            advect_scalar_conservative_3d_cartesian_impl(*c.state,
                                                         *c.src,
                                                         *c.dst,
                                                         c.dt,
                                                         c.positive,
                                                         c.context,
                                                         c.boundary,
                                                         c.kind,
                                                         c.energy,
                                                         c.defer,
                                                         c.reconstruction,
                                                         c.interface,
                                                         false,
                                                         &values);
        };
        if (!kernels->dispatch_mapped_scalar_transport(d) || !report.complete)
            throw std::runtime_error("native mapped scalar transport failed");
        dst = std::move(candidate);
        return;
    }
    if (authored && !boundary_report && !energy_report && !interface_flux_register)
    {
        if (dst.size_r() != NR || dst.size_th() != NTH || dst.size_z() != NZ)
            dst.resize(NR, NTH, NZ, 0);
        std::copy_n(authored->scalar, src.size(), dst.data());
        return;
    }
    seed_destination_from_source(execution_state, src, dst);

    const double rho_floor = static_cast<double>(density_min_kgm3);

    const float* q         = src.data();
    const float* uf        = execution_state.fields.u.data();
    const float* vf        = execution_state.fields.v.data();
    const float* wf        = execution_state.fields.w.data();
    const float* rhf       = execution_state.fields.rho.data();
    const float* rho_begin = coupled_to_retained_density ? context->density_begin.data() : nullptr;

    // Face-flux scratch: mass flux (MF*) and scalar flux (SF*) stored in the
    // owning cell's slot (the flux through face i+1/2 lives at index i).
    static Field3D mfx, mfy, mfz, sfx, sfy, sfz, r_lim;
    Field3D*       scalar_scratch[] = {&sfx, &sfy, &sfz, &r_lim};
    if (!authored)
        for (Field3D* f : scalar_scratch)
        {
            if (f->size_r() != NR || f->size_th() != NTH || f->size_z() != NZ)
            {
                f->resize(NR, NTH, NZ, 0.0f);
            }
        }
    if (!coupled_to_retained_density)
    {
        Field3D* owned_mass_flux[] = {&mfx, &mfy, &mfz};
        for (Field3D* f : owned_mass_flux)
        {
            if (f->size_r() != NR || f->size_th() != NTH || f->size_z() != NZ)
            {
                f->resize(NR, NTH, NZ, 0.0f);
            }
        }
    }
    float*       MFX_owned = mfx.data();
    const float* SFX       = authored ? authored->x : sfx.data();
    float*       MFY_owned = mfy.data();
    const float* SFY       = authored ? authored->y : sfy.data();
    float*       MFZ_owned = mfz.data();
    const float* SFZ       = authored ? authored->z : sfz.data();
    const float* MFX       = coupled_to_retained_density ? context->mass_flux_x.data() : MFX_owned;
    const float* MFY       = coupled_to_retained_density ? context->mass_flux_y.data() : MFY_owned;
    const float* MFZ       = coupled_to_retained_density ? context->mass_flux_z.data() : MFZ_owned;

    if (!authored)
    {
// ---- x-faces (i+1/2), i in [0, NR-2] ----
#pragma omp parallel for collapse(2)
        for (int i = 0; i < NR - 1; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const std::size_t f     = idx3(execution_state, i, j, k);
                    const double      mflux = coupled_to_retained_density
                                                  ? static_cast<double>(MFX[f])
                                                  : 0.5 *
                                                        (static_cast<double>(rhf[f]) +
                                                         static_cast<double>(rhf[idx3(execution_state, i + 1, j, k)])) *
                                                        static_cast<double>(uf[f]);
                    double            phi_face;
                    const bool        collocated_periodic_boundary =
                        coupled_to_retained_density &&
                        (context->staggering() == StaggerType::Collocated || mass_tendency_only) &&
                        execution_state.configuration->lateral_bc_west == tmv::init::config::LateralBC::Periodic &&
                        execution_state.configuration->lateral_bc_east == tmv::init::config::LateralBC::Periodic &&
                        (i == 0 || i == NR - 2);
                    if (collocated_periodic_boundary)
                    {
                        phi_face = (mflux >= 0.0) ? static_cast<double>(q[idx3(execution_state, NR - 2, j, k)])
                                                  : static_cast<double>(q[idx3(execution_state, 1, j, k)]);
                    }
                    else if (mflux >= 0.0)
                    {
                        const double q0 = static_cast<double>(q[f]);
                        if (uses_weno(reconstruction) && i >= 2 && i + 2 <= NR - 1)
                        {
                            phi_face = weno5_face_value(static_cast<double>(q[idx3(execution_state, i - 2, j, k)]),
                                                        static_cast<double>(q[idx3(execution_state, i - 1, j, k)]),
                                                        q0,
                                                        static_cast<double>(q[idx3(execution_state, i + 1, j, k)]),
                                                        static_cast<double>(q[idx3(execution_state, i + 2, j, k)]));
                        }
                        else
                        {
                            const double sl =
                                (i >= 1) ? mc_limited_slope(static_cast<double>(q[idx3(execution_state, i - 1, j, k)]),
                                                            q0,
                                                            static_cast<double>(q[idx3(execution_state, i + 1, j, k)]))
                                         : 0.0;
                            phi_face = q0 + 0.5 * sl;
                        }
                    }
                    else
                    {
                        const double q0 = static_cast<double>(q[idx3(execution_state, i + 1, j, k)]);
                        if (uses_weno(reconstruction) && i - 1 >= 0 && i + 3 <= NR - 1)
                        {
                            phi_face = weno5_face_value(static_cast<double>(q[idx3(execution_state, i + 3, j, k)]),
                                                        static_cast<double>(q[idx3(execution_state, i + 2, j, k)]),
                                                        q0,
                                                        static_cast<double>(q[f]),
                                                        static_cast<double>(q[idx3(execution_state, i - 1, j, k)]));
                        }
                        else
                        {
                            const double sl =
                                (i + 1 <= NR - 2)
                                    ? mc_limited_slope(static_cast<double>(q[f]),
                                                       q0,
                                                       static_cast<double>(q[idx3(execution_state, i + 2, j, k)]))
                                    : 0.0;
                            phi_face = q0 - 0.5 * sl;
                        }
                    }
                    if (!coupled_to_retained_density)
                        MFX_owned[f] = static_cast<float>(mflux);
                    if (positive_definite && preserves_positive_support(reconstruction))
                        phi_face = support_preserving_face(mflux >= 0 ? q[f] : q[idx3(execution_state, i + 1, j, k)],
                                                           phi_face);
                    sfx.data()[f] = static_cast<float>(mflux * phi_face);
                }

// ---- y-faces (j+1/2), j in [0, NTH-2] ----
#pragma omp parallel for collapse(2)
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH - 1; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const std::size_t f     = idx3(execution_state, i, j, k);
                    const double      mflux = coupled_to_retained_density
                                                  ? static_cast<double>(MFY[f])
                                                  : 0.5 *
                                                        (static_cast<double>(rhf[f]) +
                                                         static_cast<double>(rhf[idx3(execution_state, i, j + 1, k)])) *
                                                        static_cast<double>(vf[f]);
                    double            phi_face;
                    const bool        collocated_periodic_boundary =
                        coupled_to_retained_density &&
                        (context->staggering() == StaggerType::Collocated || mass_tendency_only) &&
                        execution_state.configuration->lateral_bc_south == tmv::init::config::LateralBC::Periodic &&
                        execution_state.configuration->lateral_bc_north == tmv::init::config::LateralBC::Periodic &&
                        (j == 0 || j == NTH - 2);
                    if (collocated_periodic_boundary)
                    {
                        phi_face = (mflux >= 0.0) ? static_cast<double>(q[idx3(execution_state, i, NTH - 2, k)])
                                                  : static_cast<double>(q[idx3(execution_state, i, 1, k)]);
                    }
                    else if (mflux >= 0.0)
                    {
                        const double q0 = static_cast<double>(q[f]);
                        if (uses_weno(reconstruction) && j >= 2 && j + 2 <= NTH - 1)
                        {
                            phi_face = weno5_face_value(static_cast<double>(q[idx3(execution_state, i, j - 2, k)]),
                                                        static_cast<double>(q[idx3(execution_state, i, j - 1, k)]),
                                                        q0,
                                                        static_cast<double>(q[idx3(execution_state, i, j + 1, k)]),
                                                        static_cast<double>(q[idx3(execution_state, i, j + 2, k)]));
                        }
                        else
                        {
                            const double sl =
                                (j >= 1) ? mc_limited_slope(static_cast<double>(q[idx3(execution_state, i, j - 1, k)]),
                                                            q0,
                                                            static_cast<double>(q[idx3(execution_state, i, j + 1, k)]))
                                         : 0.0;
                            phi_face = q0 + 0.5 * sl;
                        }
                    }
                    else
                    {
                        const double q0 = static_cast<double>(q[idx3(execution_state, i, j + 1, k)]);
                        if (uses_weno(reconstruction) && j - 1 >= 0 && j + 3 <= NTH - 1)
                        {
                            phi_face = weno5_face_value(static_cast<double>(q[idx3(execution_state, i, j + 3, k)]),
                                                        static_cast<double>(q[idx3(execution_state, i, j + 2, k)]),
                                                        q0,
                                                        static_cast<double>(q[f]),
                                                        static_cast<double>(q[idx3(execution_state, i, j - 1, k)]));
                        }
                        else
                        {
                            const double sl =
                                (j + 1 <= NTH - 2)
                                    ? mc_limited_slope(static_cast<double>(q[f]),
                                                       q0,
                                                       static_cast<double>(q[idx3(execution_state, i, j + 2, k)]))
                                    : 0.0;
                            phi_face = q0 - 0.5 * sl;
                        }
                    }
                    if (!coupled_to_retained_density)
                        MFY_owned[f] = static_cast<float>(mflux);
                    if (positive_definite && preserves_positive_support(reconstruction))
                        phi_face = support_preserving_face(mflux >= 0 ? q[f] : q[idx3(execution_state, i, j + 1, k)],
                                                           phi_face);
                    sfy.data()[f] = static_cast<float>(mflux * phi_face);
                }

// ---- z-faces (k+1/2), normally k in [0, NZ-2].  C-grid open top owns
// one additional bulk boundary face at k=NZ-1. ----
#pragma omp parallel for collapse(2)
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < (cgrid_open_top ? NZ : NZ - 1); ++k)
                {
                    const std::size_t f = idx3(execution_state, i, j, k);
                    if (k == NZ - 1)
                    {
                        // Zero-gradient scalar exterior state.  This gives a
                        // defined inflow value and exports the top cell value on
                        // outflow using the exact retained continuity mass flux.
                        const double mflux = coupled_to_retained_density
                                                 ? static_cast<double>(MFZ[f])
                                                 : static_cast<double>(rhf[f]) * static_cast<double>(wf[f]);
                        if (!coupled_to_retained_density)
                            MFZ_owned[f] = static_cast<float>(mflux);
                        sfz.data()[f] = static_cast<float>(mflux * static_cast<double>(q[f]));
                        continue;
                    }
                    const double mflux = coupled_to_retained_density
                                             ? static_cast<double>(MFZ[f])
                                             : 0.5 *
                                                   (static_cast<double>(rhf[f]) +
                                                    static_cast<double>(rhf[idx3(execution_state, i, j, k + 1)])) *
                                                   static_cast<double>(wf[f]);
                    double       phi_face;
                    if (mflux >= 0.0)
                    {
                        const double q0 = static_cast<double>(q[f]);
                        if (uses_weno(reconstruction) && k >= 2 && k + 2 <= NZ - 1)
                        {
                            phi_face = weno5_face_value(static_cast<double>(q[idx3(execution_state, i, j, k - 2)]),
                                                        static_cast<double>(q[idx3(execution_state, i, j, k - 1)]),
                                                        q0,
                                                        static_cast<double>(q[idx3(execution_state, i, j, k + 1)]),
                                                        static_cast<double>(q[idx3(execution_state, i, j, k + 2)]));
                        }
                        else
                        {
                            const double sl =
                                (k >= 1) ? mc_limited_slope(static_cast<double>(q[idx3(execution_state, i, j, k - 1)]),
                                                            q0,
                                                            static_cast<double>(q[idx3(execution_state, i, j, k + 1)]))
                                         : 0.0;
                            phi_face = q0 + 0.5 * sl;
                        }
                    }
                    else
                    {
                        const double q0 = static_cast<double>(q[idx3(execution_state, i, j, k + 1)]);
                        if (uses_weno(reconstruction) && k - 1 >= 0 && k + 3 <= NZ - 1)
                        {
                            phi_face = weno5_face_value(static_cast<double>(q[idx3(execution_state, i, j, k + 3)]),
                                                        static_cast<double>(q[idx3(execution_state, i, j, k + 2)]),
                                                        q0,
                                                        static_cast<double>(q[f]),
                                                        static_cast<double>(q[idx3(execution_state, i, j, k - 1)]));
                        }
                        else
                        {
                            const double sl =
                                (k + 1 <= NZ - 2)
                                    ? mc_limited_slope(static_cast<double>(q[f]),
                                                       q0,
                                                       static_cast<double>(q[idx3(execution_state, i, j, k + 2)]))
                                    : 0.0;
                            phi_face = q0 - 0.5 * sl;
                        }
                    }
                    if (!coupled_to_retained_density)
                        MFZ_owned[f] = static_cast<float>(mflux);
                    if (positive_definite && preserves_positive_support(reconstruction))
                        phi_face = support_preserving_face(mflux >= 0 ? q[f] : q[idx3(execution_state, i, j, k + 1)],
                                                           phi_face);
                    sfz.data()[f] = static_cast<float>(mflux * phi_face);
                }
    }
    // ---- positive-definite flux renormalization (SCIENCE.md M5;
    // Skamarock-Weisman 2006) ----
    // For positive-definite scalars, limit each cell's OUTGOING scalar
    // fluxes so they cannot remove more than the cell holds in one step:
    //   r = min(1, (rho q / dt) / sum(outgoing flux divergence terms)).
    // Faces are then scaled by their DONOR cell's r in the divergence
    // below. Exactly conservative (fluxes are rescaled, not clipped);
    // free-stream preserving wherever the limiter is inactive (a uniform
    // positive field only trips it past CFL 1, which never occurs); and
    // it makes the post-advection moisture clamp a true no-op guard
    // instead of a silent mass source. Outgoing-ness is by FLUX SIGN, so
    // a negative reconstructed face value (WENO undershoot transporting
    // "negative mass" upstream) is correctly charged to its donor.
    const float* RL = authored ? authored->limiter : r_lim.data();
    if (positive_definite && !authored)
    {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const std::size_t c       = idx3(execution_state, i, j, k);
                    const double inv_jacobian = terrain_metrics != nullptr ? 1.0 / terrain_metrics->J(i, j, k) : 1.0;
                    const double inv_dx_cell =
                        (terrain_metrics != nullptr ? 1.0 / terrain_metrics->dx : inv_dx) * inv_jacobian;
                    const double inv_dy_cell =
                        (terrain_metrics != nullptr ? 1.0 / terrain_metrics->dy : inv_dy) * inv_jacobian;
                    const double inv_dz_cell =
                        (terrain_metrics != nullptr ? 1.0 / terrain_metrics->dzeta : inv_dz) * inv_jacobian;
                    double out_rate = 0.0;
                    // +x face (stored at c, exists for i <= NR-2)
                    if (i <= NR - 2 && SFX[c] > 0.0f)
                        out_rate += static_cast<double>(SFX[c]) * inv_dx_cell;
                    // -x face (stored at i-1), outgoing when negative
                    if (i >= 1 && SFX[idx3(execution_state, i - 1, j, k)] < 0.0f)
                        out_rate -= static_cast<double>(SFX[idx3(execution_state, i - 1, j, k)]) * inv_dx_cell;
                    if (j <= NTH - 2 && SFY[c] > 0.0f)
                        out_rate += static_cast<double>(SFY[c]) * inv_dy_cell;
                    if (j >= 1 && SFY[idx3(execution_state, i, j - 1, k)] < 0.0f)
                        out_rate -= static_cast<double>(SFY[idx3(execution_state, i, j - 1, k)]) * inv_dy_cell;
                    if ((k <= NZ - 2 || (cgrid_open_top && k == NZ - 1)) && SFZ[c] > 0.0f)
                        out_rate += static_cast<double>(SFZ[c]) * inv_dz_cell;
                    if (k >= 1 && SFZ[idx3(execution_state, i, j, k - 1)] < 0.0f)
                        out_rate -= static_cast<double>(SFZ[idx3(execution_state, i, j, k - 1)]) * inv_dz_cell;

                    double r             = 1.0;
                    bool   support_limit = false;
                    if (out_rate > 0.0)
                    {
                        const double rho_begin_eff = coupled_to_retained_density
                                                         ? context->begin_density_in_current_geometry(i, j, k)
                                                         : static_cast<double>(rhf[c]);
                        const double available     = rho_begin_eff * std::max(static_cast<double>(q[c]), 0.0) / dt;
                        r                          = std::min(1.0, available / out_rate);
                        support_limit = preserves_positive_support(reconstruction) && available <= out_rate;
                    }
                    float r_store = static_cast<float>(r);
                    // Apply the shared six-face FP32 operation-chain bound
                    // even on CPU so backend choice does not change the
                    // positive-definite transport equation.
                    if (r > 0.0 && (r < 1.0 || support_limit))
                    {
                        r_store = cartesian_advection_detail::conservative_pd_limiter_factor(r, support_limit);
                    }
                    r_lim.data()[c] = r_store;
                }
    }

    // A face's final transport flux after positive-definite donor limiting.
    // This helper is shared by the cell update and the boundary ledger so the
    // report cannot drift from the operator it describes.
    const auto limited_scalar_flux = [&](float raw, const std::size_t donor_lo, const std::size_t donor_hi) -> double
    {
        if ((authored && authored->applied) || !positive_definite || raw == 0.0f)
            return static_cast<double>(raw);
        const float r = (raw > 0.0f) ? RL[donor_lo] : RL[donor_hi];
        return static_cast<double>(raw) * static_cast<double>(r);
    };

    if (interface_flux_register != nullptr)
    {
        const double interface_dx = terrain_metrics != nullptr ? terrain_metrics->dx : execution_state.grid.dr;
        const double interface_dy = terrain_metrics != nullptr ? terrain_metrics->dy : execution_state.grid.dtheta;
        const double interface_dz = terrain_metrics != nullptr ? terrain_metrics->dzeta : execution_state.grid.dz;
        interface_flux_register->stage_scalar_flux(
            [&](int axis, int i, int j, int k)
            {
                const auto lo = idx3(execution_state, i, j, k);
                const auto hi = idx3(execution_state, i + (axis == 0), j + (axis == 1), k);
                return limited_scalar_flux(axis == 0 ? SFX[lo] : SFY[lo], lo, hi);
            },
            dt,
            interface_dx,
            interface_dy,
            interface_dz);
    }

    long double beginning_inventory = 0.0L;
    long double outward_rate        = 0.0L;
    long double energy_outward_rate = 0.0L;
    if (boundary_report != nullptr || energy_report != nullptr)
    {
        if (report_domain.empty() || report_domain.i_begin < 1 || report_domain.j_begin < 1)
        {
            throw std::runtime_error("Cartesian scalar boundary report has no owned physical boundary faces");
        }

        const double computational_dx = terrain_metrics != nullptr ? terrain_metrics->dx : execution_state.grid.dr;
        const double computational_dy = terrain_metrics != nullptr ? terrain_metrics->dy : execution_state.grid.dtheta;
        const double computational_dz = terrain_metrics != nullptr ? terrain_metrics->dzeta : execution_state.grid.dz;
        const long double area_x      = static_cast<long double>(computational_dy) * computational_dz;
        const long double area_y      = static_cast<long double>(computational_dx) * computational_dz;
        const long double area_z      = static_cast<long double>(computational_dx) * computational_dy;
        const long double computational_volume =
            static_cast<long double>(computational_dx) * computational_dy * computational_dz;

        if (boundary_report != nullptr && !defer_report_inventories)
        {
            for (int i = report_domain.i_begin; i < report_domain.i_end; ++i)
                for (int j = report_domain.j_begin; j < report_domain.j_end; ++j)
                    for (int k = report_domain.k_begin; k < report_domain.k_end; ++k)
                    {
                        const std::size_t c        = idx3(execution_state, i, j, k);
                        const long double jacobian = static_cast<long double>(context->beginning_jacobian(i, j, k));
                        beginning_inventory += computational_volume * jacobian *
                                               static_cast<long double>(rho_begin[c]) * static_cast<long double>(q[c]);
                    }
        }

        const bool periodic_x =
            active_transport_staggering == StaggerType::Collocated &&
            execution_state.configuration->lateral_bc_west == tmv::init::config::LateralBC::Periodic &&
            execution_state.configuration->lateral_bc_east == tmv::init::config::LateralBC::Periodic;
        const bool periodic_y =
            active_transport_staggering == StaggerType::Collocated &&
            execution_state.configuration->lateral_bc_south == tmv::init::config::LateralBC::Periodic &&
            execution_state.configuration->lateral_bc_north == tmv::init::config::LateralBC::Periodic;

        const int west_face = report_domain.i_begin - 1;
        const int east_face = report_domain.i_end - 1;
        for (int j = report_domain.j_begin; j < report_domain.j_end; ++j)
            for (int k = report_domain.k_begin; k < report_domain.k_end; ++k)
            {
                const std::size_t west    = idx3(execution_state, west_face, j, k);
                const std::size_t east    = idx3(execution_state, east_face, j, k);
                const std::size_t west_lo = periodic_x ? idx3(execution_state, NR - 2, j, k) : west;
                const std::size_t west_hi =
                    periodic_x ? idx3(execution_state, 1, j, k) : idx3(execution_state, west_face + 1, j, k);
                const std::size_t east_lo = periodic_x ? idx3(execution_state, NR - 2, j, k) : east;
                const std::size_t east_hi =
                    periodic_x ? idx3(execution_state, 1, j, k) : idx3(execution_state, east_face + 1, j, k);
                const double east_flux = limited_scalar_flux(SFX[east], east_lo, east_hi);
                const double west_flux = limited_scalar_flux(SFX[west], west_lo, west_hi);
                if (boundary_report != nullptr)
                    outward_rate += area_x * (east_flux - west_flux);
                if (energy_report != nullptr && !periodic_x)
                {
                    const double east_coefficient =
                        sum_energy_coefficients(scalar_moist_energy_coefficients_v2(execution_state,
                                                                                    energy_kind,
                                                                                    active_transport_staggering,
                                                                                    terrain_metrics,
                                                                                    report_domain,
                                                                                    report_domain.i_end - 1,
                                                                                    j,
                                                                                    k));
                    const double west_coefficient =
                        sum_energy_coefficients(scalar_moist_energy_coefficients_v2(execution_state,
                                                                                    energy_kind,
                                                                                    active_transport_staggering,
                                                                                    terrain_metrics,
                                                                                    report_domain,
                                                                                    report_domain.i_begin,
                                                                                    j,
                                                                                    k));
                    energy_outward_rate += area_x * (east_flux * east_coefficient - west_flux * west_coefficient);
                }
            }

        const int south_face = report_domain.j_begin - 1;
        const int north_face = report_domain.j_end - 1;
        for (int i = report_domain.i_begin; i < report_domain.i_end; ++i)
            for (int k = report_domain.k_begin; k < report_domain.k_end; ++k)
            {
                const std::size_t south    = idx3(execution_state, i, south_face, k);
                const std::size_t north    = idx3(execution_state, i, north_face, k);
                const std::size_t south_lo = periodic_y ? idx3(execution_state, i, NTH - 2, k) : south;
                const std::size_t south_hi =
                    periodic_y ? idx3(execution_state, i, 1, k) : idx3(execution_state, i, south_face + 1, k);
                const std::size_t north_lo = periodic_y ? idx3(execution_state, i, NTH - 2, k) : north;
                const std::size_t north_hi =
                    periodic_y ? idx3(execution_state, i, 1, k) : idx3(execution_state, i, north_face + 1, k);
                const double north_flux = limited_scalar_flux(SFY[north], north_lo, north_hi);
                const double south_flux = limited_scalar_flux(SFY[south], south_lo, south_hi);
                if (boundary_report != nullptr)
                    outward_rate += area_y * (north_flux - south_flux);
                if (energy_report != nullptr && !periodic_y)
                {
                    const double north_coefficient =
                        sum_energy_coefficients(scalar_moist_energy_coefficients_v2(execution_state,
                                                                                    energy_kind,
                                                                                    active_transport_staggering,
                                                                                    terrain_metrics,
                                                                                    report_domain,
                                                                                    i,
                                                                                    report_domain.j_end - 1,
                                                                                    k));
                    const double south_coefficient =
                        sum_energy_coefficients(scalar_moist_energy_coefficients_v2(execution_state,
                                                                                    energy_kind,
                                                                                    active_transport_staggering,
                                                                                    terrain_metrics,
                                                                                    report_domain,
                                                                                    i,
                                                                                    report_domain.j_begin,
                                                                                    k));
                    energy_outward_rate += area_y * (north_flux * north_coefficient - south_flux * south_coefficient);
                }
            }

        if (active_transport_staggering == StaggerType::Collocated)
        {
            if (report_domain.k_begin < 1)
            {
                throw std::runtime_error(
                    "collocated Cartesian scalar boundary report requires vertical boundary-face storage");
            }
            const int bottom_face = report_domain.k_begin - 1;
            const int top_face    = report_domain.k_end - 1;
            for (int i = report_domain.i_begin; i < report_domain.i_end; ++i)
                for (int j = report_domain.j_begin; j < report_domain.j_end; ++j)
                {
                    const std::size_t bottom = idx3(execution_state, i, j, bottom_face);
                    const std::size_t top    = idx3(execution_state, i, j, top_face);
                    const double      top_flux =
                        limited_scalar_flux(SFZ[top], top, idx3(execution_state, i, j, top_face + 1));
                    const double bottom_flux =
                        limited_scalar_flux(SFZ[bottom], bottom, idx3(execution_state, i, j, bottom_face + 1));
                    if (boundary_report != nullptr)
                        outward_rate += area_z * (top_flux - bottom_flux);
                    if (energy_report != nullptr)
                    {
                        const double top_coefficient =
                            sum_energy_coefficients(scalar_moist_energy_coefficients_v2(execution_state,
                                                                                        energy_kind,
                                                                                        active_transport_staggering,
                                                                                        terrain_metrics,
                                                                                        report_domain,
                                                                                        i,
                                                                                        j,
                                                                                        report_domain.k_end - 1));
                        const double bottom_coefficient =
                            sum_energy_coefficients(scalar_moist_energy_coefficients_v2(execution_state,
                                                                                        energy_kind,
                                                                                        active_transport_staggering,
                                                                                        terrain_metrics,
                                                                                        report_domain,
                                                                                        i,
                                                                                        j,
                                                                                        report_domain.k_begin));
                        energy_outward_rate += area_z * (top_flux * top_coefficient - bottom_flux * bottom_coefficient);
                    }
                }
        }
        else
        {
            const int top_face = report_domain.k_end - 1;
            for (int i = report_domain.i_begin; i < report_domain.i_end; ++i)
                for (int j = report_domain.j_begin; j < report_domain.j_end; ++j)
                {
                    const std::size_t top      = idx3(execution_state, i, j, top_face);
                    const double      top_flux = limited_scalar_flux(SFZ[top], top, top);
                    if (boundary_report != nullptr)
                        outward_rate += area_z * top_flux;
                    if (energy_report != nullptr)
                    {
                        const double top_coefficient =
                            sum_energy_coefficients(scalar_moist_energy_coefficients_v2(execution_state,
                                                                                        energy_kind,
                                                                                        active_transport_staggering,
                                                                                        terrain_metrics,
                                                                                        report_domain,
                                                                                        i,
                                                                                        j,
                                                                                        report_domain.k_end - 1));
                        energy_outward_rate += area_z * top_flux * top_coefficient;
                    }
                }
        }
    }

    // ---- assemble divergences; recover phi = (rho*phi) / rho_check ----
    float* out = dst.data();
    if (mass_tendency_only)
        dst.fill(0.0f);
    const int    k_begin = coupled_to_retained_density && context->staggering() == StaggerType::Collocated ? 1 : 0;
    const int    k_end = coupled_to_retained_density && context->staggering() == StaggerType::Collocated ? NZ - 1 : NZ;
    const double computational_dx     = terrain_metrics != nullptr ? terrain_metrics->dx : execution_state.grid.dr;
    const double computational_dy     = terrain_metrics != nullptr ? terrain_metrics->dy : execution_state.grid.dtheta;
    const double computational_dz     = terrain_metrics != nullptr ? terrain_metrics->dzeta : execution_state.grid.dz;
    const double computational_volume = computational_dx * computational_dy * computational_dz;
    const auto   update_scalar_slab   = [&](int i, std::array<double, 9>* energy_partial)
    {
        for (int j = 1; j < NTH - 1; ++j)
            for (int k = k_begin; k < k_end; ++k)
            {
                const std::size_t c            = idx3(execution_state, i, j, k);
                const double      old_phi      = static_cast<double>(q[c]);
                const double      inv_jacobian = terrain_metrics != nullptr ? 1.0 / terrain_metrics->J(i, j, k) : 1.0;
                const double      inv_dx_cell =
                    (terrain_metrics != nullptr ? 1.0 / terrain_metrics->dx : inv_dx) * inv_jacobian;
                const double inv_dy_cell =
                    (terrain_metrics != nullptr ? 1.0 / terrain_metrics->dy : inv_dy) * inv_jacobian;
                const double inv_dz_cell =
                    (terrain_metrics != nullptr ? 1.0 / terrain_metrics->dzeta : inv_dz) * inv_jacobian;

                const double mx =
                    static_cast<double>(MFX[c]) - static_cast<double>(MFX[idx3(execution_state, i - 1, j, k)]);
                const double my =
                    static_cast<double>(MFY[c]) - static_cast<double>(MFY[idx3(execution_state, i, j - 1, k)]);
                const bool   has_top_flux = k <= NZ - 2 || (cgrid_open_top && k == NZ - 1);
                const double mz_top       = has_top_flux ? static_cast<double>(MFZ[c]) : 0.0;
                const double mz_bot   = (k >= 1) ? static_cast<double>(MFZ[idx3(execution_state, i, j, k - 1)]) : 0.0;
                const double mass_div = mx * inv_dx_cell + my * inv_dy_cell + (mz_top - mz_bot) * inv_dz_cell;

                const std::size_t c_xm       = idx3(execution_state, i - 1, j, k);
                const std::size_t c_xp       = idx3(execution_state, i + 1, j, k);
                const std::size_t c_ym       = idx3(execution_state, i, j - 1, k);
                const std::size_t c_yp       = idx3(execution_state, i, j + 1, k);
                const bool        collocated = coupled_to_retained_density &&
                                               (context->staggering() == StaggerType::Collocated || mass_tendency_only);
                const bool        periodic_x =
                    collocated &&
                    execution_state.configuration->lateral_bc_west == tmv::init::config::LateralBC::Periodic &&
                    execution_state.configuration->lateral_bc_east == tmv::init::config::LateralBC::Periodic;
                const bool periodic_y =
                    collocated &&
                    execution_state.configuration->lateral_bc_south == tmv::init::config::LateralBC::Periodic &&
                    execution_state.configuration->lateral_bc_north == tmv::init::config::LateralBC::Periodic;
                const std::size_t periodic_x_lo = idx3(execution_state, NR - 2, j, k);
                const std::size_t periodic_x_hi = idx3(execution_state, 1, j, k);
                const std::size_t periodic_y_lo = idx3(execution_state, i, NTH - 2, k);
                const std::size_t periodic_y_hi = idx3(execution_state, i, 1, k);
                const double      sx_plus       = periodic_x && i == NR - 2
                                                      ? limited_scalar_flux(SFX[c], periodic_x_lo, periodic_x_hi)
                                                      : limited_scalar_flux(SFX[c], c, c_xp);
                const double      sx_minus      = periodic_x && i == 1
                                                      ? limited_scalar_flux(SFX[c_xm], periodic_x_lo, periodic_x_hi)
                                                      : limited_scalar_flux(SFX[c_xm], c_xm, c);
                const double      sy_plus       = periodic_y && j == NTH - 2
                                                      ? limited_scalar_flux(SFY[c], periodic_y_lo, periodic_y_hi)
                                                      : limited_scalar_flux(SFY[c], c, c_yp);
                const double      sy_minus      = periodic_y && j == 1
                                                      ? limited_scalar_flux(SFY[c_ym], periodic_y_lo, periodic_y_hi)
                                                      : limited_scalar_flux(SFY[c_ym], c_ym, c);
                const double      sz_plus =
                    has_top_flux ? limited_scalar_flux(SFZ[c], c, k < NZ - 1 ? idx3(execution_state, i, j, k + 1) : c)
                                 : 0.0;
                const double sz_minus = (k >= 1) ? limited_scalar_flux(SFZ[idx3(execution_state, i, j, k - 1)],
                                                                       idx3(execution_state, i, j, k - 1),
                                                                       c)
                                                 : 0.0;

                if (mass_tendency_only)
                {
                    out[c] =
                        static_cast<float>(-((sx_plus - sx_minus) * inv_dx_cell + (sy_plus - sy_minus) * inv_dy_cell +
                                             (sz_plus - sz_minus) * inv_dz_cell));
                    continue;
                }

                const double rho_end = static_cast<double>(rhf[c]);
                // Standalone transport predicts density from the same
                // instantaneous face flux. Coupled production transport has
                // already retained rho_end from dynamics and supplies the
                // exact beginning density paired with the unchanged scalar.
                const double rho_begin_eff =
                    coupled_to_retained_density ? context->begin_density_in_current_geometry(i, j, k) : rho_end;
                const double rho_denominator = coupled_to_retained_density ? rho_end : rho_end - dt * mass_div;
                double       rho_phi;
                if (positive_definite)
                {
                    // Accumulate the positive-definite update as a budget:
                    // first debit all outgoing shared face fluxes, then add
                    // incoming ones.  This is algebraically identical to
                    // -dt*div(SF), but it avoids cancellation between large
                    // signed directional divergences exactly where the donor
                    // limiter has exhausted a sparse cell.
                    double outgoing  = 0.0;
                    double incoming  = 0.0;
                    auto   plus_face = [&](double flux, double inv_d)
                    {
                        if (flux >= 0.0)
                            outgoing += flux * inv_d;
                        else
                            incoming -= flux * inv_d;
                    };
                    auto minus_face = [&](double flux, double inv_d)
                    {
                        if (flux <= 0.0)
                            outgoing -= flux * inv_d;
                        else
                            incoming += flux * inv_d;
                    };
                    plus_face(sx_plus, inv_dx_cell);
                    minus_face(sx_minus, inv_dx_cell);
                    plus_face(sy_plus, inv_dy_cell);
                    minus_face(sy_minus, inv_dy_cell);
                    if (has_top_flux)
                        plus_face(sz_plus, inv_dz_cell);
                    if (k >= 1)
                        minus_face(sz_minus, inv_dz_cell);

                    rho_phi = (rho_begin_eff * static_cast<double>(q[c]) - dt * outgoing) + dt * incoming;
                }
                else
                {
                    const double scal_div = (sx_plus - sx_minus) * inv_dx_cell + (sy_plus - sy_minus) * inv_dy_cell +
                                            (sz_plus - sz_minus) * inv_dz_cell;
                    rho_phi               = rho_begin_eff * static_cast<double>(q[c]) - dt * scal_div;
                }
                if (authored && authored->requested)
                    rho_phi = rho_begin_eff * static_cast<double>(q[c]) +
                              double(float(dt)) * (double(authored->requested[c]) - authored->requested_loss[c]);
                const double denom = (rho_denominator > rho_floor) ? rho_denominator : rho_floor;
                out[c]             = authored ? authored->scalar[c] : static_cast<float>(rho_phi / denom);

                if (energy_partial != nullptr)
                {
                    const auto   coefficients = scalar_moist_energy_coefficients_v2(execution_state,
                                                                                    energy_kind,
                                                                                    active_transport_staggering,
                                                                                    terrain_metrics,
                                                                                    report_domain,
                                                                                    i,
                                                                                    j,
                                                                                    k);
                    const double coefficient  = sum_energy_coefficients(coefficients);
                    const double jacobian     = terrain_metrics != nullptr ? terrain_metrics->J(i, j, k) : 1.0;
                    const double volume       = computational_volume * jacobian;
                    // This owner changes phi at the retained density and end
                    // geometry. Density/geometry work belongs to their commit.
                    const double committed_scalar_mass_change = rho_end * (static_cast<double>(out[c]) - old_phi);
                    for (std::size_t reservoir = 0; reservoir < coefficients.size(); ++reservoir)
                    {
                        (*energy_partial)[reservoir] += volume * committed_scalar_mass_change * coefficients[reservoir];
                    }
                    (*energy_partial)[6] += volume * coefficient * (rho_begin_eff - rho_end) * old_phi;
                    (*energy_partial)[7] += volume * coefficient * (rho_phi - rho_begin_eff * old_phi);
                    const double values[] = {
                        coefficient,
                        volume,
                        committed_scalar_mass_change,
                        (*energy_partial)[6],
                        (*energy_partial)[7],
                    };
                    if (std::any_of(
                            std::begin(values), std::end(values), [](double value) { return !std::isfinite(value); }))
                    {
                        (*energy_partial)[8] += 1.0;
                    }
                }
            }
    };

    std::array<double, 9> energy_sums{};
    if (energy_report != nullptr)
    {
        energy_sums = tmv::diagnostics::fixed_slab_reduce<9>(
            1, NR - 1, [&](int i, std::array<double, 9>& partial) { update_scalar_slab(i, &partial); });
    }
    else
    {
#pragma omp parallel for schedule(static)
        for (int i = 1; i < NR - 1; ++i)
            update_scalar_slab(i, nullptr);
    }

    if (energy_report != nullptr)
    {
        energy_report->dry_internal_change_j       = energy_sums[0];
        energy_report->water_sensible_change_j     = energy_sums[1];
        energy_report->phase_reference_change_j    = energy_sums[2];
        energy_report->carrier_kinetic_change_j    = energy_sums[3];
        energy_report->gravitational_change_j      = energy_sums[4];
        energy_report->subgrid_tke_change_j        = energy_sums[5];
        energy_report->density_history_remap_j     = energy_sums[6];
        energy_report->limited_face_state_change_j = energy_sums[7];
        energy_report->outward_boundary_energy_j =
            static_cast<double>(energy_outward_rate * static_cast<long double>(dt));
        energy_report->interior_conversion_j =
            energy_report->limited_face_state_change_j + energy_report->outward_boundary_energy_j;
        energy_report->requested_state_change_j =
            energy_report->density_history_remap_j + energy_report->limited_face_state_change_j;
        energy_report->committed_state_change_j =
            energy_report->dry_internal_change_j + energy_report->water_sensible_change_j +
            energy_report->phase_reference_change_j + energy_report->carrier_kinetic_change_j +
            energy_report->gravitational_change_j + energy_report->subgrid_tke_change_j;
        energy_report->storage_rounding_j =
            energy_report->committed_state_change_j - energy_report->requested_state_change_j;
        energy_report->closure_residual_j =
            energy_report->committed_state_change_j -
            (energy_report->requested_state_change_j + energy_report->storage_rounding_j);
        const double values[] = {
            energy_report->dry_internal_change_j,
            energy_report->water_sensible_change_j,
            energy_report->phase_reference_change_j,
            energy_report->carrier_kinetic_change_j,
            energy_report->gravitational_change_j,
            energy_report->subgrid_tke_change_j,
            energy_report->density_history_remap_j,
            energy_report->outward_boundary_energy_j,
            energy_report->interior_conversion_j,
            energy_report->limited_face_state_change_j,
            energy_report->requested_state_change_j,
            energy_report->storage_rounding_j,
            energy_report->committed_state_change_j,
            energy_report->closure_residual_j,
        };
        if (energy_sums[8] != 0.0 ||
            std::any_of(std::begin(values), std::end(values), [](double value) { return !std::isfinite(value); }))
        {
            *energy_report = {};
            throw std::runtime_error("Cartesian scalar transport produced a non-finite "
                                     "moist-energy ledger");
        }
        energy_report->complete = true;
    }

    if (boundary_report != nullptr)
    {
        boundary_report->outward_boundary_inventory = static_cast<double>(outward_rate * static_cast<long double>(dt));
        boundary_report->flux_implied_state_change  = -boundary_report->outward_boundary_inventory;
        if (!std::isfinite(boundary_report->outward_boundary_inventory))
        {
            *boundary_report = {};
            throw std::runtime_error("Cartesian scalar boundary transport produced a non-finite boundary flux ledger");
        }
        if (defer_report_inventories)
        {
            // The batch caller closes every requested scalar in two fused
            // volume passes. Leaving complete=false prevents a boundary-only
            // intermediate report from escaping as a valid ledger.
            return;
        }

        long double       ending_inventory = 0.0L;
        const double      computational_dx = terrain_metrics != nullptr ? terrain_metrics->dx : execution_state.grid.dr;
        const double      computational_dy = terrain_metrics != nullptr ? terrain_metrics->dy : execution_state.grid.dtheta;
        const double      computational_dz = terrain_metrics != nullptr ? terrain_metrics->dzeta : execution_state.grid.dz;
        const long double computational_volume =
            static_cast<long double>(computational_dx) * computational_dy * computational_dz;
        for (int i = report_domain.i_begin; i < report_domain.i_end; ++i)
            for (int j = report_domain.j_begin; j < report_domain.j_end; ++j)
                for (int k = report_domain.k_begin; k < report_domain.k_end; ++k)
                {
                    const std::size_t c = idx3(execution_state, i, j, k);
                    const long double jacobian =
                        terrain_metrics != nullptr ? static_cast<long double>(terrain_metrics->J(i, j, k)) : 1.0L;
                    ending_inventory += computational_volume * jacobian * static_cast<long double>(rhf[c]) *
                                        static_cast<long double>(out[c]);
                }

        boundary_report->committed_state_change = static_cast<double>(ending_inventory - beginning_inventory);
        boundary_report->closure_residual =
            boundary_report->committed_state_change - boundary_report->flux_implied_state_change;
        if (!std::isfinite(boundary_report->outward_boundary_inventory) ||
            !std::isfinite(boundary_report->committed_state_change) ||
            !std::isfinite(boundary_report->closure_residual))
        {
            *boundary_report = {};
            throw std::runtime_error("Cartesian scalar boundary transport produced a non-finite ledger");
        }
        boundary_report->complete = true;
    }
}

void attribute_flat_cgrid_device_scalar_commit(const float*                               scalar_begin,
                                               const float*                               scalar_end,
                                               const float*                               density_begin,
                                               const float*                               density_end,
                                               const float*                               mass_flux_x,
                                               const float*                               mass_flux_y,
                                               const float*                               mass_flux_z,
                                               int                                        nx,
                                               int                                        ny,
                                               int                                        nz,
                                               float                                      inv_dx,
                                               float                                      inv_dy,
                                               float                                      inv_dz,
                                               float                                      dt_step,
                                               bool                                       positive_definite,
                                               ScalarBoundaryTransportReport*             boundary_report,
                                               ScalarMoistEnergyKind                      energy_kind,
                                               ScalarMoistEnergyTransportReportV1*        energy_report,
                                               FlatCGridDeviceScalarAttributionWorkspace& workspace,
                                               const float*                               device_boundary_fluxes,
                                               const float*                               device_requested_update)
{
    return attribute_flat_cgrid_device_scalar_commit(global_simulation_state(),
                                                     scalar_begin,
                                                     scalar_end,
                                                     density_begin,
                                                     density_end,
                                                     mass_flux_x,
                                                     mass_flux_y,
                                                     mass_flux_z,
                                                     nx,
                                                     ny,
                                                     nz,
                                                     inv_dx,
                                                     inv_dy,
                                                     inv_dz,
                                                     dt_step,
                                                     positive_definite,
                                                     boundary_report,
                                                     energy_kind,
                                                     energy_report,
                                                     workspace,
                                                     device_boundary_fluxes,
                                                     device_requested_update);
}

void attribute_flat_cgrid_device_scalar_commit(const SimulationStateView&                 execution_state,
                                               const float*                               scalar_begin,
                                               const float*                               scalar_end,
                                               const float*                               density_begin,
                                               const float*                               density_end,
                                               const float*                               mass_flux_x,
                                               const float*                               mass_flux_y,
                                               const float*                               mass_flux_z,
                                               int                                        nx,
                                               int                                        ny,
                                               int                                        nz,
                                               float                                      inv_dx,
                                               float                                      inv_dy,
                                               float                                      inv_dz,
                                               float                                      dt_step,
                                               bool                                       positive_definite,
                                               ScalarBoundaryTransportReport*             boundary_report,
                                               ScalarMoistEnergyKind                      energy_kind,
                                               ScalarMoistEnergyTransportReportV1*        energy_report,
                                               FlatCGridDeviceScalarAttributionWorkspace& workspace,
                                               const float*                               device_boundary_fluxes,
                                               const float*                               device_requested_update)
{
    if ((device_boundary_fluxes == nullptr) != (device_requested_update == nullptr))
        throw std::invalid_argument("device scalar audit outputs must be provided together");
    if ((energy_report == nullptr) != (energy_kind == ScalarMoistEnergyKind::None))
    {
        throw std::invalid_argument("flat C-grid device scalar attribution received a partial energy descriptor");
    }
    if (boundary_report == nullptr && energy_report == nullptr)
        return;
    if (scalar_begin == nullptr || scalar_end == nullptr || density_begin == nullptr || density_end == nullptr ||
        mass_flux_x == nullptr || mass_flux_y == nullptr || mass_flux_z == nullptr)
    {
        throw std::invalid_argument("flat C-grid device scalar attribution received a null field");
    }
    if (nx != execution_state.grid.nr || ny != execution_state.grid.nth || nz != execution_state.grid.nz || nx < 3 ||
        ny < 3 || nz < 1 || execution_state.configuration->stagger_type != StaggerType::CGrid)
    {
        throw std::invalid_argument("flat C-grid device scalar attribution grid/layout mismatch");
    }
    require_positive_finite(inv_dx, "device scalar inverse dx");
    require_positive_finite(inv_dy, "device scalar inverse dy");
    require_positive_finite(inv_dz, "device scalar inverse dz");
    require_positive_finite(dt_step, "device scalar timestep");
    if (boundary_report != nullptr)
        *boundary_report = {};
    if (energy_report != nullptr)
    {
        *energy_report = {};
        require_global_field_shape(execution_state, execution_state.fields.p, "device scalar attribution pressure");
        require_global_field_shape(execution_state, execution_state.fields.theta, "device scalar attribution theta");
        require_global_field_shape(execution_state, execution_state.fields.qv, "device scalar attribution qv");
        require_global_field_shape(execution_state, execution_state.fields.qc, "device scalar attribution qc");
        require_global_field_shape(execution_state, execution_state.fields.qr, "device scalar attribution qr");
        require_global_field_shape(execution_state, execution_state.fields.qi, "device scalar attribution qi");
        require_global_field_shape(execution_state, execution_state.fields.qs, "device scalar attribution qs");
        require_global_field_shape(execution_state, execution_state.fields.qg, "device scalar attribution qg");
        require_global_field_shape(execution_state, execution_state.fields.qh, "device scalar attribution qh");
        require_global_field_shape(execution_state, execution_state.fields.u, "device scalar attribution u");
        require_global_field_shape(execution_state, execution_state.fields.v, "device scalar attribution v");
        require_global_field_shape(execution_state, execution_state.fields.w, "device scalar attribution w");
    }

    const PhysicalCellDomain domain = physical_scalar_domain(nx, ny, nz, StaggerType::CGrid);
    if (domain.empty() || domain.i_begin < 1 || domain.j_begin < 1)
    {
        throw std::runtime_error("flat C-grid device scalar attribution has no physical domain");
    }
#ifdef NIMBUS_SCALAR_ATTRIBUTION_REDUNDANT_COEFFICIENT_PREPASS
    // SCI-182 comparator only: reproduce the removed read-only traversal
    // without changing the authoritative reduction or committed result.
    if (energy_report != nullptr)
    {
        int invalid_coefficients = 0;
#pragma omp parallel for collapse(2) reduction(+ : invalid_coefficients)
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const auto coefficients =
                        scalar_moist_energy_coefficients_v2(energy_kind, StaggerType::CGrid, nullptr, domain, i, j, k);
                    if (std::any_of(coefficients.begin(),
                                    coefficients.end(),
                                    [](double value) { return !std::isfinite(value); }))
                    {
                        ++invalid_coefficients;
                    }
                }
        if (invalid_coefficients != 0)
        {
            throw std::runtime_error("flat C-grid device scalar attribution rejected a non-finite "
                                     "operator-time coefficient");
        }
    }
#endif
    const auto at = [ny, nz](int i, int j, int k) -> std::size_t
    {
        return (static_cast<std::size_t>(i) * static_cast<std::size_t>(ny) + static_cast<std::size_t>(j)) *
                   static_cast<std::size_t>(nz) +
               static_cast<std::size_t>(k);
    };

    float* sfx     = nullptr;
    float* sfy     = nullptr;
    float* sfz     = nullptr;
    float* limiter = nullptr;
    if (device_boundary_fluxes == nullptr)
    {
        Field3D* scratch_fields[] = {
            &workspace.scalar_flux_x,
            &workspace.scalar_flux_y,
            &workspace.scalar_flux_z,
            &workspace.donor_limiter,
        };
        for (Field3D* field : scratch_fields)
        {
            if (field->size_r() != nx || field->size_th() != ny || field->size_z() != nz)
            {
                field->resize(nx, ny, nz, 0.0f);
            }
        }
        sfx     = workspace.scalar_flux_x.data();
        sfy     = workspace.scalar_flux_y.data();
        sfz     = workspace.scalar_flux_z.data();
        limiter = workspace.donor_limiter.data();

        // Literal FP32 mirror of the three face values emitted by the shader.
        // Keep face construction and the dependent donor-limiter pass inside one
        // OpenMP team.  The implicit barrier after the first workshare preserves
        // the shader's phase boundary, while avoiding one team creation for every
        // reported scalar.  Theta never reads a donor limiter, so its second
        // volume traversal is omitted entirely.
#ifdef NIMBUS_SCALAR_ATTRIBUTION_COMBINED_OPENMP_EXPERIMENT
#pragma omp parallel
        {
#pragma omp for collapse(2)
#else
#pragma omp parallel for collapse(2)
#endif
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j)
                    for (int k = 0; k < nz; ++k)
                    {
                        const std::size_t c = at(i, j, k);

                        float sx = 0.0f;
                        if (i <= nx - 2)
                        {
                            const float mflux = mass_flux_x[c];
                            float       face  = 0.0f;
                            float       base  = 0.0f;
                            if (mflux >= 0.0f)
                            {
                                const float q0 = scalar_begin[c];
                                base           = q0;
                                if (i >= 2 && i + 2 <= nx - 1)
                                {
                                    face = device_weno5_face_offset(scalar_begin[at(i - 2, j, k)],
                                                                    scalar_begin[at(i - 1, j, k)],
                                                                    q0,
                                                                    scalar_begin[at(i + 1, j, k)],
                                                                    scalar_begin[at(i + 2, j, k)]);
                                }
                                else
                                {
                                    const float slope = i >= 1 ? device_mc_limited_slope(scalar_begin[at(i - 1, j, k)],
                                                                                         q0,
                                                                                         scalar_begin[at(i + 1, j, k)])
                                                               : 0.0f;
                                    face              = +0.5f * slope;
                                }
                            }
                            else
                            {
                                const float q0 = scalar_begin[at(i + 1, j, k)];
                                base           = q0;
                                if (i - 1 >= 0 && i + 3 <= nx - 1)
                                {
                                    face = device_weno5_face_offset(scalar_begin[at(i + 3, j, k)],
                                                                    scalar_begin[at(i + 2, j, k)],
                                                                    q0,
                                                                    scalar_begin[c],
                                                                    scalar_begin[at(i - 1, j, k)]);
                                }
                                else
                                {
                                    const float slope = i + 1 <= nx - 2
                                                            ? device_mc_limited_slope(
                                                                  scalar_begin[c], q0, scalar_begin[at(i + 2, j, k)])
                                                            : 0.0f;
                                    face              = -0.5f * slope;
                                }
                            }
                            sx = std::fma(mflux, base, mflux * face);
                        }
                        sfx[c] = sx;

                        float sy = 0.0f;
                        if (j <= ny - 2)
                        {
                            const float mflux = mass_flux_y[c];
                            float       face  = 0.0f;
                            float       base  = 0.0f;
                            if (mflux >= 0.0f)
                            {
                                const float q0 = scalar_begin[c];
                                base           = q0;
                                if (j >= 2 && j + 2 <= ny - 1)
                                {
                                    face = device_weno5_face_offset(scalar_begin[at(i, j - 2, k)],
                                                                    scalar_begin[at(i, j - 1, k)],
                                                                    q0,
                                                                    scalar_begin[at(i, j + 1, k)],
                                                                    scalar_begin[at(i, j + 2, k)]);
                                }
                                else
                                {
                                    const float slope = j >= 1 ? device_mc_limited_slope(scalar_begin[at(i, j - 1, k)],
                                                                                         q0,
                                                                                         scalar_begin[at(i, j + 1, k)])
                                                               : 0.0f;
                                    face              = +0.5f * slope;
                                }
                            }
                            else
                            {
                                const float q0 = scalar_begin[at(i, j + 1, k)];
                                base           = q0;
                                if (j - 1 >= 0 && j + 3 <= ny - 1)
                                {
                                    face = device_weno5_face_offset(scalar_begin[at(i, j + 3, k)],
                                                                    scalar_begin[at(i, j + 2, k)],
                                                                    q0,
                                                                    scalar_begin[c],
                                                                    scalar_begin[at(i, j - 1, k)]);
                                }
                                else
                                {
                                    const float slope = j + 1 <= ny - 2
                                                            ? device_mc_limited_slope(
                                                                  scalar_begin[c], q0, scalar_begin[at(i, j + 2, k)])
                                                            : 0.0f;
                                    face              = -0.5f * slope;
                                }
                            }
                            sy = std::fma(mflux, base, mflux * face);
                        }
                        sfy[c] = sy;

                        float sz = 0.0f;
                        if (k == nz - 1)
                        {
                            sz = mass_flux_z[c] * scalar_begin[c];
                        }
                        else
                        {
                            const float mflux = mass_flux_z[c];
                            float       face  = 0.0f;
                            float       base  = 0.0f;
                            if (mflux >= 0.0f)
                            {
                                const float q0 = scalar_begin[c];
                                base           = q0;
                                if (k >= 2 && k + 2 <= nz - 1)
                                {
                                    face = device_weno5_face_offset(scalar_begin[at(i, j, k - 2)],
                                                                    scalar_begin[at(i, j, k - 1)],
                                                                    q0,
                                                                    scalar_begin[at(i, j, k + 1)],
                                                                    scalar_begin[at(i, j, k + 2)]);
                                }
                                else
                                {
                                    const float slope = k >= 1 ? device_mc_limited_slope(scalar_begin[at(i, j, k - 1)],
                                                                                         q0,
                                                                                         scalar_begin[at(i, j, k + 1)])
                                                               : 0.0f;
                                    face              = +0.5f * slope;
                                }
                            }
                            else
                            {
                                const float q0 = scalar_begin[at(i, j, k + 1)];
                                base           = q0;
                                if (k - 1 >= 0 && k + 3 <= nz - 1)
                                {
                                    face = device_weno5_face_offset(scalar_begin[at(i, j, k + 3)],
                                                                    scalar_begin[at(i, j, k + 2)],
                                                                    q0,
                                                                    scalar_begin[c],
                                                                    scalar_begin[at(i, j, k - 1)]);
                                }
                                else
                                {
                                    const float slope = k + 1 <= nz - 2
                                                            ? device_mc_limited_slope(
                                                                  scalar_begin[c], q0, scalar_begin[at(i, j, k + 2)])
                                                            : 0.0f;
                                    face              = -0.5f * slope;
                                }
                            }
                            sz = std::fma(mflux, base, mflux * face);
                        }
                        sfz[c] = sz;
                    }

#ifdef NIMBUS_SCALAR_ATTRIBUTION_COMBINED_OPENMP_EXPERIMENT
            if (positive_definite)
            {
#pragma omp for collapse(2)
#else
#pragma omp parallel for collapse(2)
#endif
                for (int i = 0; i < nx; ++i)
                    for (int j = 0; j < ny; ++j)
                        for (int k = 0; k < nz; ++k)
                        {
                            const std::size_t c        = at(i, j, k);
                            float             out_rate = 0.0f;
                            if (i <= nx - 2 && sfx[c] > 0.0f)
                                out_rate += sfx[c] * inv_dx;
                            if (i >= 1)
                            {
                                const float flux = sfx[at(i - 1, j, k)];
                                if (flux < 0.0f)
                                    out_rate -= flux * inv_dx;
                            }
                            if (j <= ny - 2 && sfy[c] > 0.0f)
                                out_rate += sfy[c] * inv_dy;
                            if (j >= 1)
                            {
                                const float flux = sfy[at(i, j - 1, k)];
                                if (flux < 0.0f)
                                    out_rate -= flux * inv_dy;
                            }
                            if (sfz[c] > 0.0f)
                                out_rate += sfz[c] * inv_dz;
                            if (k >= 1)
                            {
                                const float flux = sfz[at(i, j, k - 1)];
                                if (flux < 0.0f)
                                    out_rate -= flux * inv_dz;
                            }

                            float ratio = 1.0f;
                            if (positive_definite && out_rate > 0.0f)
                            {
                                const float available = density_begin[c] * std::max(scalar_begin[c], 0.0f) / dt_step;
                                ratio                 = std::min(1.0f, available / out_rate);
                                ratio = cartesian_advection_detail::conservative_pd_limiter_factor(ratio);
                            }
                            limiter[c] = ratio;
                        }
#ifdef NIMBUS_SCALAR_ATTRIBUTION_COMBINED_OPENMP_EXPERIMENT
            }
        }
#endif
    }

    const auto limited_flux = [&](float raw, std::size_t donor_lo, std::size_t donor_hi) -> float
    {
        if (!positive_definite)
            return raw;
        return raw * (raw > 0.0f ? limiter[donor_lo] : limiter[donor_hi]);
    };

    const double      dx_m                = require_positive_finite(execution_state.grid.dr, "Cartesian dx");
    const double      dy_m                = require_positive_finite(execution_state.grid.dtheta, "Cartesian dy");
    const double      dz_m                = require_positive_finite(execution_state.grid.dz, "Cartesian dz");
    const double      volume              = dx_m * dy_m * dz_m;
    const long double area_x              = static_cast<long double>(dy_m) * static_cast<long double>(dz_m);
    const long double area_y              = static_cast<long double>(dx_m) * static_cast<long double>(dz_m);
    const long double area_z              = static_cast<long double>(dx_m) * static_cast<long double>(dy_m);
    long double       outward_rate        = 0.0L;
    long double       energy_outward_rate = 0.0L;

    const std::size_t x_faces   = std::size_t(ny - 2) * nz;
    const std::size_t y_faces   = std::size_t(nx - 2) * nz;
    const int         west_face = domain.i_begin - 1;
    const int         east_face = domain.i_end - 1;
    for (int j = domain.j_begin; j < domain.j_end; ++j)
        for (int k = domain.k_begin; k < domain.k_end; ++k)
        {
            const std::size_t west = at(west_face, j, k);
            const std::size_t east = at(east_face, j, k);
            const float west_flux  = device_boundary_fluxes ? device_boundary_fluxes[std::size_t(j - 1) * nz + k]
                                                            : limited_flux(sfx[west], west, at(west_face + 1, j, k));
            const float east_flux  = device_boundary_fluxes
                                         ? device_boundary_fluxes[x_faces + std::size_t(j - 1) * nz + k]
                                         : limited_flux(sfx[east], east, at(east_face + 1, j, k));
            if (boundary_report != nullptr)
                outward_rate += area_x * (static_cast<long double>(east_flux) - static_cast<long double>(west_flux));
            if (energy_report != nullptr)
            {
                const double west_coefficient = sum_energy_coefficients(scalar_moist_energy_coefficients_v2(
                    execution_state, energy_kind, StaggerType::CGrid, nullptr, domain, domain.i_begin, j, k));
                const double east_coefficient = sum_energy_coefficients(scalar_moist_energy_coefficients_v2(
                    execution_state, energy_kind, StaggerType::CGrid, nullptr, domain, domain.i_end - 1, j, k));
                energy_outward_rate += area_x * (static_cast<long double>(east_flux) * east_coefficient -
                                                 static_cast<long double>(west_flux) * west_coefficient);
            }
        }

    const int south_face = domain.j_begin - 1;
    const int north_face = domain.j_end - 1;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int k = domain.k_begin; k < domain.k_end; ++k)
        {
            const std::size_t south      = at(i, south_face, k);
            const std::size_t north      = at(i, north_face, k);
            const float       south_flux = device_boundary_fluxes
                                               ? device_boundary_fluxes[2 * x_faces + std::size_t(i - 1) * nz + k]
                                               : limited_flux(sfy[south], south, at(i, south_face + 1, k));
            const float north_flux = device_boundary_fluxes
                                         ? device_boundary_fluxes[2 * x_faces + y_faces + std::size_t(i - 1) * nz + k]
                                         : limited_flux(sfy[north], north, at(i, north_face + 1, k));
            if (boundary_report != nullptr)
                outward_rate += area_y * (static_cast<long double>(north_flux) - static_cast<long double>(south_flux));
            if (energy_report != nullptr)
            {
                const double south_coefficient = sum_energy_coefficients(scalar_moist_energy_coefficients_v2(
                    execution_state, energy_kind, StaggerType::CGrid, nullptr, domain, i, domain.j_begin, k));
                const double north_coefficient = sum_energy_coefficients(scalar_moist_energy_coefficients_v2(
                    execution_state, energy_kind, StaggerType::CGrid, nullptr, domain, i, domain.j_end - 1, k));
                energy_outward_rate += area_y * (static_cast<long double>(north_flux) * north_coefficient -
                                                 static_cast<long double>(south_flux) * south_coefficient);
            }
        }

    const int top_face = domain.k_end - 1;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            const std::size_t top = at(i, j, top_face);
            const float       top_flux =
                device_boundary_fluxes
                    ? device_boundary_fluxes[2 * (x_faces + y_faces) + std::size_t(i - 1) * (ny - 2) + j - 1]
                    : limited_flux(sfz[top], top, top);
            if (boundary_report != nullptr)
                outward_rate += area_z * static_cast<long double>(top_flux);
            if (energy_report != nullptr)
            {
                const double top_coefficient = sum_energy_coefficients(scalar_moist_energy_coefficients_v2(
                    execution_state, energy_kind, StaggerType::CGrid, nullptr, domain, i, j, domain.k_end - 1));
                energy_outward_rate += area_z * static_cast<long double>(top_flux) * top_coefficient;
            }
        }

    // 0..5: committed energy reservoirs; 6: density-history remap;
    // 7: requested limited-face change; 8/9: boundary begin/end inventory;
    // 10: invalid contribution count.
    const auto sums = tmv::diagnostics::fixed_slab_reduce<11>(
        domain.i_begin,
        domain.i_end,
        [&](int i, std::array<double, 11>& partial)
        {
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const std::size_t c               = at(i, j, k);
                    const double      old_scalar      = static_cast<double>(scalar_begin[c]);
                    const double      new_scalar      = static_cast<double>(scalar_end[c]);
                    const double      rho_begin_value = static_cast<double>(density_begin[c]);
                    const double      rho_end_value   = static_cast<double>(density_end[c]);

                    if (boundary_report != nullptr)
                    {
                        partial[8] += volume * rho_begin_value * old_scalar;
                        partial[9] += volume * rho_end_value * new_scalar;
                    }
                    if (energy_report == nullptr)
                        continue;

                    double requested_rho_scalar = 0;
                    if (device_requested_update)
                    {
                        requested_rho_scalar =
                            positive_definite
                                ? rho_begin_value * old_scalar + double(dt_step) * device_requested_update[c]
                                : rho_end_value * (old_scalar + double(device_requested_update[c]));
                    }
                    else
                    {
                        const std::size_t c_im = at(i - 1, j, k);
                        const std::size_t c_ip = at(i + 1, j, k);
                        const std::size_t c_jm = at(i, j - 1, k);
                        const std::size_t c_jp = at(i, j + 1, k);
                        const std::size_t c_km = k >= 1 ? at(i, j, k - 1) : c;
                        const std::size_t c_kp = k < nz - 1 ? at(i, j, k + 1) : c;

                        const float sx_plus  = limited_flux(sfx[c], c, c_ip);
                        const float sx_minus = limited_flux(sfx[c_im], c_im, c);
                        const float sy_plus  = limited_flux(sfy[c], c, c_jp);
                        const float sy_minus = limited_flux(sfy[c_jm], c_jm, c);
                        const float sz_plus  = limited_flux(sfz[c], c, c_kp);
                        const float sz_minus = k >= 1 ? limited_flux(sfz[c_km], c_km, c) : 0.0f;

                        if (positive_definite)
                        {
                            float outgoing = 0.0f;
                            float incoming = 0.0f;
                            if (sx_plus >= 0.0f)
                                outgoing += sx_plus * inv_dx;
                            else
                                incoming -= sx_plus * inv_dx;
                            if (sx_minus <= 0.0f)
                                outgoing -= sx_minus * inv_dx;
                            else
                                incoming += sx_minus * inv_dx;
                            if (sy_plus >= 0.0f)
                                outgoing += sy_plus * inv_dy;
                            else
                                incoming -= sy_plus * inv_dy;
                            if (sy_minus <= 0.0f)
                                outgoing -= sy_minus * inv_dy;
                            else
                                incoming += sy_minus * inv_dy;
                            if (sz_plus >= 0.0f)
                                outgoing += sz_plus * inv_dz;
                            else
                                incoming -= sz_plus * inv_dz;
                            if (sz_minus <= 0.0f)
                                outgoing -= sz_minus * inv_dz;
                            else
                                incoming += sz_minus * inv_dz;
                            requested_rho_scalar =
                                (density_begin[c] * scalar_begin[c] - dt_step * outgoing) + dt_step * incoming;
                        }
                        else
                        {
                            const float sx         = sx_plus - sx_minus;
                            const float sy         = sy_plus - sy_minus;
                            const float scalar_div = sx * inv_dx + sy * inv_dy + (sz_plus - sz_minus) * inv_dz;
                            const float increment =
                                ((density_begin[c] - density_end[c]) * scalar_begin[c] - dt_step * scalar_div) /
                                density_end[c];
                            requested_rho_scalar = rho_end_value * (old_scalar + static_cast<double>(increment));
                        }
                    }

                    const auto coefficients = scalar_moist_energy_coefficients_v2(
                        execution_state, energy_kind, StaggerType::CGrid, nullptr, domain, i, j, k);
                    const double coefficient                  = sum_energy_coefficients(coefficients);
                    const double committed_scalar_mass_change = rho_end_value * (new_scalar - old_scalar);
                    for (std::size_t reservoir = 0; reservoir < coefficients.size(); ++reservoir)
                    {
                        partial[reservoir] += volume * committed_scalar_mass_change * coefficients[reservoir];
                    }
                    partial[6] += volume * coefficient * (rho_begin_value - rho_end_value) * old_scalar;
                    partial[7] += volume * coefficient *
                                  (static_cast<double>(requested_rho_scalar) - rho_begin_value * old_scalar);
                    const double values[] = {
                        old_scalar,
                        new_scalar,
                        rho_begin_value,
                        rho_end_value,
                        coefficient,
                        static_cast<double>(requested_rho_scalar),
                    };
                    // This authoritative fixed-slab pass used to be preceded
                    // by a second full-volume traversal whose only purpose
                    // was to validate these same coefficients. Validate every
                    // component here before the report can commit. The sums,
                    // per-scalar transaction boundary, and fixed merge order
                    // are unchanged.
                    if (std::any_of(coefficients.begin(),
                                    coefficients.end(),
                                    [](double value) { return !std::isfinite(value); }) ||
                        std::any_of(
                            std::begin(values), std::end(values), [](double value) { return !std::isfinite(value); }))
                    {
                        partial[10] += 1.0;
                    }
                }
        });

    if (sums[10] != 0.0)
    {
        if (boundary_report != nullptr)
            *boundary_report = {};
        if (energy_report != nullptr)
            *energy_report = {};
        throw std::runtime_error("flat C-grid device scalar attribution produced a non-finite ledger");
    }

    if (boundary_report != nullptr)
    {
        boundary_report->outward_boundary_inventory =
            static_cast<double>(outward_rate * static_cast<long double>(dt_step));
        boundary_report->flux_implied_state_change = -boundary_report->outward_boundary_inventory;
        boundary_report->committed_state_change    = sums[9] - sums[8];
        boundary_report->closure_residual =
            boundary_report->committed_state_change - boundary_report->flux_implied_state_change;
        const double values[] = {
            boundary_report->outward_boundary_inventory,
            boundary_report->flux_implied_state_change,
            boundary_report->committed_state_change,
            boundary_report->closure_residual,
        };
        if (std::any_of(std::begin(values), std::end(values), [](double value) { return !std::isfinite(value); }))
        {
            *boundary_report = {};
            throw std::runtime_error("flat C-grid device scalar attribution produced a non-finite "
                                     "boundary ledger");
        }
        boundary_report->complete = true;
    }

    if (energy_report != nullptr)
    {
        energy_report->dry_internal_change_j       = sums[0];
        energy_report->water_sensible_change_j     = sums[1];
        energy_report->phase_reference_change_j    = sums[2];
        energy_report->carrier_kinetic_change_j    = sums[3];
        energy_report->gravitational_change_j      = sums[4];
        energy_report->subgrid_tke_change_j        = sums[5];
        energy_report->density_history_remap_j     = sums[6];
        energy_report->limited_face_state_change_j = sums[7];
        energy_report->outward_boundary_energy_j =
            static_cast<double>(energy_outward_rate * static_cast<long double>(dt_step));
        energy_report->interior_conversion_j =
            energy_report->limited_face_state_change_j + energy_report->outward_boundary_energy_j;
        energy_report->requested_state_change_j =
            energy_report->density_history_remap_j + energy_report->limited_face_state_change_j;
        energy_report->committed_state_change_j =
            energy_report->dry_internal_change_j + energy_report->water_sensible_change_j +
            energy_report->phase_reference_change_j + energy_report->carrier_kinetic_change_j +
            energy_report->gravitational_change_j + energy_report->subgrid_tke_change_j;
        energy_report->storage_rounding_j =
            energy_report->committed_state_change_j - energy_report->requested_state_change_j;
        energy_report->closure_residual_j =
            energy_report->committed_state_change_j -
            (energy_report->requested_state_change_j + energy_report->storage_rounding_j);
        const double values[] = {
            energy_report->dry_internal_change_j,
            energy_report->water_sensible_change_j,
            energy_report->phase_reference_change_j,
            energy_report->carrier_kinetic_change_j,
            energy_report->gravitational_change_j,
            energy_report->subgrid_tke_change_j,
            energy_report->density_history_remap_j,
            energy_report->outward_boundary_energy_j,
            energy_report->interior_conversion_j,
            energy_report->limited_face_state_change_j,
            energy_report->requested_state_change_j,
            energy_report->storage_rounding_j,
            energy_report->committed_state_change_j,
            energy_report->closure_residual_j,
        };
        if (std::any_of(std::begin(values), std::end(values), [](double value) { return !std::isfinite(value); }))
        {
            *energy_report = {};
            throw std::runtime_error("flat C-grid device scalar attribution produced a non-finite "
                                     "moist-energy ledger");
        }
        energy_report->complete = true;
    }
}

void compute_conservative_scalar_mass_tendency_cartesian(
    const Field3D& src, Field3D& tendency, double positivity_dt,
    bool positive_definite, const CartesianCGridTransportContext& context,
    CartesianFaceReconstruction reconstruction)
{
    return compute_conservative_scalar_mass_tendency_cartesian(global_simulation_state(), src, tendency, positivity_dt, positive_definite, context, reconstruction);
}

void compute_conservative_scalar_mass_tendency_cartesian(const SimulationStateView& execution_state,
    const Field3D& src, Field3D& tendency, double positivity_dt,
    bool positive_definite, const CartesianCGridTransportContext& context,
    CartesianFaceReconstruction reconstruction)
{
    if (src.data() == tendency.data())
        throw std::invalid_argument("scalar mass tendency cannot alias its input");
    advect_scalar_conservative_3d_cartesian_impl(execution_state,
        src, tendency, positivity_dt, positive_definite, &context,
        nullptr, ScalarMoistEnergyKind::None, nullptr, false,
        reconstruction, nullptr, true);
}

void advect_scalar_conservative_3d_cartesian(const Field3D&                        src,
                                             Field3D&                              dst,
                                             double                                dt,
                                             bool                                  positive_definite,
                                             const CartesianCGridTransportContext* context,
                                             ScalarBoundaryTransportReport*        boundary_report,
                                             ScalarMoistEnergyKind                 energy_kind,
                                             ScalarMoistEnergyTransportReportV1*   energy_report,
                                             CartesianFaceReconstruction           reconstruction,
                                             CartesianInterfaceFluxRegister*       interface_flux_register)
{
    return advect_scalar_conservative_3d_cartesian(global_simulation_state(), src, dst, dt, positive_definite, context, boundary_report, energy_kind, energy_report, reconstruction, interface_flux_register);
}

void advect_scalar_conservative_3d_cartesian(const SimulationStateView& execution_state, const Field3D&                        src,
                                             Field3D&                              dst,
                                             double                                dt,
                                             bool                                  positive_definite,
                                             const CartesianCGridTransportContext* context,
                                             ScalarBoundaryTransportReport*        boundary_report,
                                             ScalarMoistEnergyKind                 energy_kind,
                                             ScalarMoistEnergyTransportReportV1*   energy_report,
                                             CartesianFaceReconstruction           reconstruction,
                                             CartesianInterfaceFluxRegister*       interface_flux_register)
{
    try
    {
        advect_scalar_conservative_3d_cartesian_impl(execution_state,
            src, dst, dt, positive_definite, context, boundary_report,
            energy_kind, energy_report, /*defer_report_inventories=*/false,
            reconstruction, interface_flux_register);
        if (interface_flux_register != nullptr)
        {
            interface_flux_register->validate_staged_theta_commit();
            interface_flux_register->commit_staged_theta();
        }
    }
    catch (...)
    {
        if (interface_flux_register != nullptr)
            interface_flux_register->discard_staged_theta();
        throw;
    }
}

void advect_scalars_conservative_3d_cartesian(CartesianScalarTransport*             scalars,
                                              std::size_t                           count,
                                              double                                dt,
                                              const CartesianCGridTransportContext* context,
                                              CartesianFaceReconstruction           reconstruction)
{
    return advect_scalars_conservative_3d_cartesian(global_simulation_state(), scalars, count, dt, context, reconstruction);
}

void advect_scalars_conservative_3d_cartesian(const SimulationStateView& execution_state, CartesianScalarTransport*             scalars,
                                              std::size_t                           count,
                                              double                                dt,
                                              const CartesianCGridTransportContext* context,
                                              CartesianFaceReconstruction           reconstruction)
{
    if (count == 0)
    {
        return;
    }
    if (scalars == nullptr)
    {
        throw std::invalid_argument("Cartesian C-grid scalar batch requires a non-null descriptor array");
    }
    if (context == nullptr)
    {
        throw std::invalid_argument("Cartesian C-grid CPU batch requires retained mass-flux context");
    }

    const bool timing = advection_detail::perf_enabled();
    const auto batch_started = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    // Validate every caller-owned input before the first in-place update so a
    // malformed batch cannot leave only a prefix advanced.
    context->require_valid(execution_state, execution_state.grid.nr, execution_state.grid.nth, execution_state.grid.nz, dt);
    std::vector<CartesianInterfaceFluxRegister*> staged_interface_flux;
    for (std::size_t s = 0; s < count; ++s)
    {
        if (preserves_positive_support(reconstruction) && !scalars[s].positive_definite)
            throw std::invalid_argument("support-preserving scalar batch requires positive carriers");
        if (scalars[s].field == nullptr)
        {
            throw std::invalid_argument("Cartesian C-grid scalar batch contains a null field");
        }
        if (scalars[s].field->size_r() != execution_state.grid.nr ||
            scalars[s].field->size_th() != execution_state.grid.nth ||
            scalars[s].field->size_z() != execution_state.grid.nz)
        {
            throw std::invalid_argument("Cartesian C-grid scalar batch field shape mismatch");
        }
        if (!std::isfinite(scalars[s].kappa) || scalars[s].kappa != 0.0)
        {
            throw std::invalid_argument("Cartesian C-grid scalar batch requires kappa=0; use the "
                                        "live-density diffusion scheme for scalar mixing");
        }
        if ((scalars[s].energy_report == nullptr) != (scalars[s].energy_kind == ScalarMoistEnergyKind::None))
        {
            throw std::invalid_argument("Cartesian C-grid scalar batch energy descriptor is incomplete");
        }
        if (scalars[s].interface_flux_register != nullptr)
        {
            if (std::find(staged_interface_flux.begin(),
                          staged_interface_flux.end(),
                          scalars[s].interface_flux_register) != staged_interface_flux.end())
                throw std::invalid_argument("Cartesian scalar batch contains a duplicate interface register");
            staged_interface_flux.push_back(scalars[s].interface_flux_register);
        }
        for (std::size_t prior = 0; prior < s; ++prior)
        {
            if (scalars[prior].field == scalars[s].field)
            {
                throw std::invalid_argument("Cartesian C-grid scalar batch contains a duplicate field");
            }
            if (scalars[s].boundary_report != nullptr && scalars[prior].boundary_report == scalars[s].boundary_report)
            {
                throw std::invalid_argument("Cartesian C-grid scalar batch contains a duplicate boundary report");
            }
            if (scalars[s].energy_report != nullptr && scalars[prior].energy_report == scalars[s].energy_report)
            {
                throw std::invalid_argument("Cartesian C-grid scalar batch contains a duplicate "
                                            "moist-energy report");
            }
        }
    }

    // A production thermodynamic batch has at most eight ledgers. Keep a
    // generous fixed bound so reporting never allocates per step, while still
    // failing before mutation if a diagnostic caller exceeds the contract.
    constexpr std::size_t                             kMaxBatchBoundaryReports = 64;
    std::array<std::size_t, kMaxBatchBoundaryReports> report_scalar{};
    std::array<long double, kMaxBatchBoundaryReports> beginning_inventory{};
    std::array<long double, kMaxBatchBoundaryReports> ending_inventory{};
    std::size_t                                       report_count = 0;
    for (std::size_t s = 0; s < count; ++s)
    {
        if (scalars[s].boundary_report == nullptr)
            continue;
        if (report_count == kMaxBatchBoundaryReports)
        {
            throw std::invalid_argument("Cartesian C-grid scalar batch boundary-report count exceeds 64");
        }
        report_scalar[report_count++] = s;
    }

    const PhysicalCellDomain report_domain    = physical_scalar_domain(execution_state.grid.nr, execution_state.grid.nth, execution_state.grid.nz, context->staggering());
    const TerrainMetrics3D*  terrain_metrics  = context->terrain_metrics();
    const double             computational_dx = terrain_metrics != nullptr ? terrain_metrics->dx : execution_state.grid.dr;
    const double             computational_dy = terrain_metrics != nullptr ? terrain_metrics->dy : execution_state.grid.dtheta;
    const double             computational_dz = terrain_metrics != nullptr ? terrain_metrics->dzeta : execution_state.grid.dz;
    const long double        computational_volume =
        static_cast<long double>(computational_dx) * computational_dy * computational_dz;
    const auto inventory_started = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (report_count != 0)
    {
        if (report_domain.empty())
        {
            throw std::runtime_error("Cartesian C-grid scalar batch boundary report has an empty physical domain");
        }
        const float* rho_begin = context->density_begin.data();
        for (int i = report_domain.i_begin; i < report_domain.i_end; ++i)
            for (int j = report_domain.j_begin; j < report_domain.j_end; ++j)
                for (int k = report_domain.k_begin; k < report_domain.k_end; ++k)
                {
                    const std::size_t c = idx3(execution_state, i, j, k);
                    const long double weight =
                        computational_volume *
                        static_cast<long double>(context->beginning_jacobian(i,j,k)) *
                        static_cast<long double>(rho_begin[c]);
                    for (std::size_t r = 0; r < report_count; ++r)
                    {
                        beginning_inventory[r] +=
                            weight * static_cast<long double>(scalars[report_scalar[r]].field->data()[c]);
                    }
                }
    }

    if (timing)
        advection_detail::perf_totals().host_boundary_inventory_s +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - inventory_started).count();

    const bool native_transaction = terrain_metrics && execution_state.compute_runtime &&
        execution_state.compute_runtime->active_kind != ComputeBackendKind::Cpu;
    std::vector<Field3D> saved_fields;
    std::vector<ScalarBoundaryTransportReport> saved_boundaries;
    std::vector<ScalarMoistEnergyTransportReportV1> saved_energies;
    std::vector<CartesianInterfaceFluxRegister> saved_interfaces;
    if (native_transaction)
    {
        saved_fields.reserve(count);
        for (std::size_t s = 0; s < count; ++s)
        {
            saved_fields.push_back(*scalars[s].field);
            saved_boundaries.push_back(scalars[s].boundary_report ? *scalars[s].boundary_report : ScalarBoundaryTransportReport{});
            saved_energies.push_back(scalars[s].energy_report ? *scalars[s].energy_report : ScalarMoistEnergyTransportReportV1{});
        }
        for (auto* capture : staged_interface_flux)
            saved_interfaces.push_back(*capture);
    }

    // The conservative kernel finishes all face reconstructions before its
    // divergence pass, and that pass reads only the current cell's original
    // scalar plus face scratch. Exact in-place output is therefore safe and
    // removes the old full-volume destination/copy pair for every scalar.
    try
    {
        for (std::size_t s = 0; s < count; ++s)
        {
            CartesianScalarTransport& item = scalars[s];
            advect_scalar_conservative_3d_cartesian_impl(execution_state,
                *item.field, *item.field, dt, item.positive_definite,
                context, item.boundary_report, item.energy_kind,
                item.energy_report,
                /*defer_report_inventories=*/item.boundary_report != nullptr,
                reconstruction, item.interface_flux_register);
        }

        const auto inventory_finished = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (report_count != 0)
        {
            const float* rho_end = execution_state.fields.rho.data();
            for (int i = report_domain.i_begin; i < report_domain.i_end; ++i)
                for (int j = report_domain.j_begin; j < report_domain.j_end; ++j)
                    for (int k = report_domain.k_begin; k < report_domain.k_end; ++k)
                    {
                        const std::size_t c = idx3(execution_state, i, j, k);
                        const long double weight =
                            computational_volume *
                            (terrain_metrics != nullptr
                                 ? static_cast<long double>(terrain_metrics->J(i, j, k))
                                 : 1.0L) *
                            static_cast<long double>(rho_end[c]);
                        for (std::size_t r = 0; r < report_count; ++r)
                            ending_inventory[r] += weight *
                                static_cast<long double>(
                                    scalars[report_scalar[r]].field->data()[c]);
                    }

            for (std::size_t r = 0; r < report_count; ++r)
            {
                ScalarBoundaryTransportReport& report =
                    *scalars[report_scalar[r]].boundary_report;
                report.committed_state_change = static_cast<double>(
                    ending_inventory[r] - beginning_inventory[r]);
                report.closure_residual = report.committed_state_change -
                    report.flux_implied_state_change;
                if (!std::isfinite(report.outward_boundary_inventory) ||
                    !std::isfinite(report.committed_state_change) ||
                    !std::isfinite(report.closure_residual))
                {
                    report = {};
                    throw std::runtime_error(
                        "Cartesian scalar batch produced a non-finite boundary ledger");
                }
                report.complete = true;
            }
        }
        if (timing)
            advection_detail::perf_totals().host_boundary_inventory_s +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - inventory_finished).count();
        for (auto* capture : staged_interface_flux)
            capture->validate_staged_scalar_commit();
        for (auto* capture : staged_interface_flux)
            capture->commit_staged_scalar();
    }
    catch (...)
    {
        if (native_transaction)
        {
            for (std::size_t s = 0; s < count; ++s)
            {
                *scalars[s].field = std::move(saved_fields[s]);
                if (scalars[s].boundary_report)
                    *scalars[s].boundary_report = saved_boundaries[s];
                if (scalars[s].energy_report)
                    *scalars[s].energy_report = saved_energies[s];
            }
            for (std::size_t s = 0; s < staged_interface_flux.size(); ++s)
                *staged_interface_flux[s] = std::move(saved_interfaces[s]);
            throw;
        }
        for (auto* capture : staged_interface_flux)
            capture->discard_staged_scalar();
        throw;
    }
    if (timing)
        advection_detail::perf_totals().host_batch_s +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - batch_started).count();
}

} // namespace tmv
