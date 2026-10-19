/**
 * @file velocity.cpp
 * @brief Implementation for the radar module.
 *
 * Provides executable logic for the radar runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/radar subsystem.
 */

#include "velocity.hpp"
#include "core/infra/grid_coordinates.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace
{

void ensure_velocity_output(RadarOut& out, int nr, int nth, int nz)
{
    out.NR = nr;
    out.NTH = nth;
    out.NZ = nz;

    if (out.Vr.size_r() != nr || out.Vr.size_th() != nth || out.Vr.size_z() != nz)
    {
        out.Vr.resize(nr, nth, nz, 0.0f);
    }
    else
    {
        out.Vr.fill(0.0f);
    }
}

void physical_coordinate(const RadarStateView& state,
                         int i, int j, int k,
                         double& x, double& y, double& z)
{
    const int gi = state.geometry_i_offset + i;
    const int gj = state.geometry_j_offset + j;
    const int gk = state.geometry_k_offset + k;
    x = state.grid_origin_earth_x_m
        + state.geometry->r[static_cast<std::size_t>(gi)];
    y = state.grid_origin_earth_y_m
        + state.geometry->theta[static_cast<std::size_t>(gj)];
    z = state.terrain_metrics != nullptr
        ? state.terrain_metrics->z(gi, gj, gk)
        : state.geometry->z[static_cast<std::size_t>(gk)];
}

float sample_component(const Field3D* field,
                       int i,
                       int j,
                       int k,
                       int nr,
                       int nth,
                       int nz,
                       bool beam_volume,
                       StaggerType staggering)
{
    if (!field || nr <= 0 || nth <= 0 || nz <= 0)
    {
        return 0.0f;
    }

    const PhysicalCellDomain domain =
        physical_scalar_domain(nr, nth, nz, staggering);
    if (domain.empty())
    {
        return 0.0f;
    }
    const int ic = std::clamp(i, domain.i_begin, domain.i_end - 1);
    const int jc = std::clamp(j, domain.j_begin, domain.j_end - 1);
    const int kc = std::clamp(k, domain.k_begin, domain.k_end - 1);

    if (!beam_volume)
    {
        const float value = (*field)[ic][jc][kc];
        return std::isfinite(static_cast<double>(value)) ? value : 0.0f;
    }

    double sum = 0.0;
    int count = 0;
    for (int di = -1; di <= 1; ++di)
    {
        const int ii = ic + di;
        if (ii < domain.i_begin || ii >= domain.i_end)
        {
            continue;
        }
        for (int dj = -1; dj <= 1; ++dj)
        {
            const int jj = jc + dj;
            if (jj < domain.j_begin || jj >= domain.j_end)
            {
                continue;
            }
            for (int dk = -1; dk <= 1; ++dk)
            {
                const int kk = kc + dk;
                if (kk < domain.k_begin || kk >= domain.k_end)
                {
                    continue;
                }
                const float value = (*field)[ii][jj][kk];
                if (std::isfinite(static_cast<double>(value)))
                {
                    sum += static_cast<double>(value);
                    ++count;
                }
            }
        }
    }
    return (count > 0) ? static_cast<float>(sum / static_cast<double>(count)) : 0.0f;
}

enum class WindAxis
{
    X,
    Y,
    Z,
};

/**
 * @brief Returns one wind component at the scalar center of a grid cell.
 *
 * The C-grid formulas intentionally match core/output/stagger_interpolation:
 * u and v use the maintained west/south face one-sided at index zero, while
 * the surface w value averages the top face with the rigid, zero-flow ground
 * face. Collocated inputs pass through unchanged.
 */
float wind_at_scalar_center(const Field3D& field,
                            int i,
                            int j,
                            int k,
                            StaggerType staggering,
                            WindAxis axis)
{
    const float upper_face = field[i][j][k];
    if (staggering != StaggerType::CGrid)
    {
        return upper_face;
    }

    switch (axis)
    {
        case WindAxis::X:
            return (i == 0)
                ? upper_face
                : 0.5f * (field[i - 1][j][k] + upper_face);
        case WindAxis::Y:
            return (j == 0)
                ? upper_face
                : 0.5f * (field[i][j - 1][k] + upper_face);
        case WindAxis::Z:
            return (k == 0)
                ? 0.5f * upper_face
                : 0.5f * (field[i][j][k - 1] + upper_face);
    }
    return upper_face;
}

float sample_wind_component(const Field3D* field,
                            int i,
                            int j,
                            int k,
                            int nr,
                            int nth,
                            int nz,
                            bool beam_volume,
                            StaggerType staggering,
                            WindAxis axis)
{
    if (!field || nr <= 0 || nth <= 0 || nz <= 0)
    {
        return 0.0f;
    }

    const PhysicalCellDomain domain = physical_scalar_domain(nr, nth, nz, staggering);
    if (domain.empty())
    {
        return 0.0f;
    }
    const int ic = std::clamp(i, domain.i_begin, domain.i_end - 1);
    const int jc = std::clamp(j, domain.j_begin, domain.j_end - 1);
    const int kc = std::clamp(k, domain.k_begin, domain.k_end - 1);

    if (!beam_volume)
    {
        const float value = wind_at_scalar_center(*field, ic, jc, kc, staggering, axis);
        return std::isfinite(static_cast<double>(value)) ? value : 0.0f;
    }

    // Beam-volume averaging is performed over scalar-center winds, not over
    // raw faces. This keeps point and volume sampling on the same placement.
    double sum = 0.0;
    int count = 0;
    for (int di = -1; di <= 1; ++di)
    {
        const int ii = ic + di;
        if (ii < domain.i_begin || ii >= domain.i_end)
        {
            continue;
        }
        for (int dj = -1; dj <= 1; ++dj)
        {
            const int jj = jc + dj;
            if (jj < domain.j_begin || jj >= domain.j_end)
            {
                continue;
            }
            for (int dk = -1; dk <= 1; ++dk)
            {
                const int kk = kc + dk;
                if (kk < domain.k_begin || kk >= domain.k_end)
                {
                    continue;
                }
                const float value =
                    wind_at_scalar_center(*field, ii, jj, kk, staggering, axis);
                if (std::isfinite(static_cast<double>(value)))
                {
                    sum += static_cast<double>(value);
                    ++count;
                }
            }
        }
    }
    return (count > 0) ? static_cast<float>(sum / static_cast<double>(count)) : 0.0f;
}

}

namespace tmv
{

/**
 * @brief Initializes the velocity scheme.
 */
void VelocityScheme::initialize(const RadarConfig& config, int NR, int NTH, int NZ)
{
    NR_ = NR;
    NTH_ = NTH;
    NZ_ = NZ;

    radar_x_ = config.radar_x;
    radar_y_ = config.radar_y;
    radar_z_ = config.radar_z;

    tmv::log_info("Initialized velocity radar scheme (", NR, "x", NTH, "x", NZ, ")");
    tmv::log_info("Radar location: (", radar_x_, ", ", radar_y_, ", ", radar_z_, ")");
}

void VelocityScheme::invalidate_generation_state_after_domain_recycling() noexcept
{
    candidate_output_.invalidate_generation_values();
}

/**
 * @brief Computes the velocity scheme.
 */
void VelocityScheme::compute(const RadarConfig& config, const RadarStateView& state, RadarOut& out) 
{
    validate_radar_dimensions(state, NR_, NTH_, NZ_, "velocity");
    validate_radar_velocity_state(config, state);
    validate_radar_physical_geometry(state);
    ensure_velocity_output(candidate_output_, NR_, NTH_, NZ_);
    compute_radial_velocity(config, state, candidate_output_);

    if (!config.apply_scatterer_correction)
    {
        validate_radar_output_field(candidate_output_.Vr,
            NR_, NTH_, NZ_, -RadarConstants::velocity_abs_max_ms,
            RadarConstants::velocity_abs_max_ms, "Vr");
        out.NR = NR_;
        out.NTH = NTH_;
        out.NZ = NZ_;
        std::swap(out.Vr, candidate_output_.Vr);
        return;
    }

    const bool has_hydrometeors = config.has_qr || config.has_qs
        || config.has_qg || config.has_qh || config.has_qi;
    if (!has_hydrometeors)
        throw std::invalid_argument(
            "radar scatterer correction requires at least one configured hydrometeor field");

    apply_scatterer_correction(config, state, candidate_output_);
    validate_radar_output_field(candidate_output_.Vr,
        NR_, NTH_, NZ_, -RadarConstants::velocity_abs_max_ms,
        RadarConstants::velocity_abs_max_ms, "Vr");
    out.NR = NR_;
    out.NTH = NTH_;
    out.NZ = NZ_;
    std::swap(out.Vr, candidate_output_.Vr);
}

/**
 * @brief Computes the radial velocity.
 */
void VelocityScheme::compute_radial_velocity(const RadarConfig& config, const RadarStateView& state, RadarOut& out) 
{
    const bool beam_volume = (config.sampling == "beam_volume");

    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(NR_) * NTH_ * NZ_ >= 32768)
    for (int i = 0; i < NR_; ++i) 
    {
        for (int j = 0; j < NTH_; ++j) 
        {
            for (int k = 0; k < NZ_; ++k) 
            {
                const float u = sample_wind_component(
                    state.u, i, j, k, NR_, NTH_, NZ_, beam_volume,
                    state.staggering, WindAxis::X);
                const float v = sample_wind_component(
                    state.v, i, j, k, NR_, NTH_, NZ_, beam_volume,
                    state.staggering, WindAxis::Y);
                const float w = sample_wind_component(
                    state.w, i, j, k, NR_, NTH_, NZ_, beam_volume,
                    state.staggering, WindAxis::Z);

                double x = 0.0, y = 0.0, z = 0.0;
                physical_coordinate(state, i, j, k, x, y, z);

                double e_r_x, e_r_y, e_r_z, R;
                RadarGeometry::compute_line_of_sight(radar_x_, radar_y_, radar_z_,
                                                   x, y, z, e_r_x, e_r_y, e_r_z, R);

                float Vr = static_cast<float>(u * e_r_x + v * e_r_y + w * e_r_z);
                out.Vr[i][j][k] = Vr;
            }
        }
    }
}

/**
 * @brief Applies the scatterer correction.
 */
void VelocityScheme::apply_scatterer_correction(const RadarConfig& config, const RadarStateView& state, RadarOut& out) 
{
    const bool beam_volume = (config.sampling == "beam_volume");
    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(NR_) * NTH_ * NZ_ >= 32768)
    for (int i = 0; i < NR_; ++i) 
    {
        for (int j = 0; j < NTH_; ++j) 
        {
            for (int k = 0; k < NZ_; ++k) 
            {
                const float q_rain = sample_component(
                    config.has_qr ? state.qr : nullptr,
                    i, j, k, NR_, NTH_, NZ_, beam_volume, state.staggering);
                const float q_snow = sample_component(
                    config.has_qs ? state.qs : nullptr,
                    i, j, k, NR_, NTH_, NZ_, beam_volume, state.staggering);
                const float q_graupel = sample_component(
                    config.has_qg ? state.qg : nullptr,
                    i, j, k, NR_, NTH_, NZ_, beam_volume, state.staggering);
                const float q_hail = sample_component(
                    config.has_qh ? state.qh : nullptr,
                    i, j, k, NR_, NTH_, NZ_, beam_volume, state.staggering);
                const float q_ice = sample_component(
                    config.has_qi ? state.qi : nullptr,
                    i, j, k, NR_, NTH_, NZ_, beam_volume, state.staggering);

                float v_terminal = estimate_fall_speed(q_rain, q_snow, q_graupel, q_hail, q_ice);

                double x = 0.0, y = 0.0, z = 0.0;
                physical_coordinate(state, i, j, k, x, y, z);

                double e_r_x, e_r_y, e_r_z, R;
                RadarGeometry::compute_line_of_sight(radar_x_, radar_y_, radar_z_,
                                                   x, y, z, e_r_x, e_r_y, e_r_z, R);

                float correction = -v_terminal * static_cast<float>(e_r_z);
                float vr_corrected = out.Vr[i][j][k] + correction;
                out.Vr[i][j][k] = vr_corrected;
            }
        }
    }
}

/**
 * @brief Estimates the fall speed.
 */
float VelocityScheme::estimate_fall_speed(float q_rain, float q_snow, float q_graupel, float q_hail, float q_ice) 
{
    const float qr = std::max(0.0f, q_rain);
    const float qs = std::max(0.0f, q_snow);
    const float qg = std::max(0.0f, q_graupel);
    const float qh = std::max(0.0f, q_hail);
    const float qi = std::max(0.0f, q_ice);
    const float q_total = qr + qs + qg + qh + qi;
    if (q_total <= 0.0f)
    {
        return 0.0f;
    }

    auto bulk_speed = [](float q, float a, float b) -> float
    {
        if (q <= 0.0f)
        {
            return 0.0f;
        }
        return a * std::pow(q * 1000.0f, b);
    };

    const float v_rain = radar_bulk_rain_fall_speed_ms(qr);
    const float v_snow = bulk_speed(qs, 1.5f, 0.16f);
    const float v_graupel = bulk_speed(qg, 11.0f, 0.18f);
    const float v_hail = bulk_speed(qh, 15.0f, 0.16f);
    const float v_ice = bulk_speed(qi, 0.8f, 0.14f);

    const float v_weighted =
        (qr * v_rain + qs * v_snow + qg * v_graupel + qh * v_hail + qi * v_ice) / q_total;
    if (!std::isfinite(static_cast<double>(v_weighted)))
    {
        return 0.0f;
    }

    return std::clamp(v_weighted, 0.0f, 40.0f);
}

}  // namespace tmv
