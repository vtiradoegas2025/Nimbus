/**
 * @file radar_base.cpp
 * @brief Implementation for the radar module.
 *
 * Provides executable logic for the radar runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/radar subsystem.
 */

#include "radar/radar_base.hpp"
#include "core/field/field_bounds.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

/**
 * @brief Shared radar geometry and sampling utilities implementation
 */

namespace
{

/**
 * @brief Ensures a field has single-cell dimensions (1x1x1).
 */
void ensure_single_cell_shape(Field3D& out)
{
    if (out.size_r() != 1 || out.size_th() != 1 || out.size_z() != 1)
    {
        out.resize(1, 1, 1, 0.0f);
    }
}

/**
 * @brief Copies one sampled cell into a single-cell scratch field.
 */
Field3D* sample_to_single_cell(const Field3D* source, int i, int j, int k, Field3D& out)
{
    if (source == nullptr)
    {
        return nullptr;
    }

    ensure_single_cell_shape(out);
    out(0, 0, 0) = static_cast<float>((*source)[i][j][k]);
    return &out;
}

enum class WindAxis
{
    X,
    Y,
    Z,
};

/**
 * @brief Samples a wind at the scalar center into a single-cell field.
 */
Field3D* sample_wind_to_single_cell(const Field3D* source,
                                    int i,
                                    int j,
                                    int k,
                                    StaggerType staggering,
                                    WindAxis axis,
                                    Field3D& out)
{
    if (source == nullptr)
    {
        return nullptr;
    }

    ensure_single_cell_shape(out);
    const float upper_face = (*source)[i][j][k];
    if (staggering != StaggerType::CGrid)
    {
        out(0, 0, 0) = upper_face;
        return &out;
    }

    switch (axis)
    {
        case WindAxis::X:
            out(0, 0, 0) = (i == 0)
                ? upper_face
                : 0.5f * ((*source)[i - 1][j][k] + upper_face);
            break;
        case WindAxis::Y:
            out(0, 0, 0) = (j == 0)
                ? upper_face
                : 0.5f * ((*source)[i][j - 1][k] + upper_face);
            break;
        case WindAxis::Z:
            out(0, 0, 0) = (k == 0)
                ? 0.5f * upper_face
                : 0.5f * ((*source)[i][j][k - 1] + upper_face);
            break;
    }
    return &out;
}

/**
 * @brief Scratch buffers used for point-sampled radar state evaluation.
 */
struct RadarPointScratch
{
    Field3D u;
    Field3D v;
    Field3D w;
    Field3D qr;
    Field3D qs;
    Field3D qg;
    Field3D qh;
    Field3D qi;
    Field3D Nr;
    Field3D Ns;
    Field3D Ng;
    Field3D Nh;
    Field3D Ni;
    Field3D theta;
    Field3D p;
    Field3D rho;
};

/**
 * @brief Initializes all scratch fields for point-sample extraction.
 */
void initialize_scratch_fields(RadarPointScratch& scratch)
{
    ensure_single_cell_shape(scratch.u);
    ensure_single_cell_shape(scratch.v);
    ensure_single_cell_shape(scratch.w);
    ensure_single_cell_shape(scratch.qr);
    ensure_single_cell_shape(scratch.qs);
    ensure_single_cell_shape(scratch.qg);
    ensure_single_cell_shape(scratch.qh);
    ensure_single_cell_shape(scratch.qi);
    ensure_single_cell_shape(scratch.Nr);
    ensure_single_cell_shape(scratch.Ns);
    ensure_single_cell_shape(scratch.Ng);
    ensure_single_cell_shape(scratch.Nh);
    ensure_single_cell_shape(scratch.Ni);
    ensure_single_cell_shape(scratch.theta);
    ensure_single_cell_shape(scratch.p);
    ensure_single_cell_shape(scratch.rho);
}

}

namespace tmv
{

namespace
{

bool field_matches(const Field3D* field, const RadarStateView& state)
{
    return field != nullptr
        && field->size_r() == state.NR
        && field->size_th() == state.NTH
        && field->size_z() == state.NZ;
}

struct SpeciesInput
{
    bool configured;
    const Field3D* mixing_ratio;
    bool number_configured;
    const Field3D* number;
    const char* name;
};

std::array<SpeciesInput, 5> species_inputs(const RadarConfig& config,
                                           const RadarStateView& state)
{
    return {{
        {config.has_qr, state.qr, config.has_Nr, state.Nr, "rain"},
        {config.has_qs, state.qs, config.has_Ns, state.Ns, "snow"},
        {config.has_qg, state.qg, config.has_Ng, state.Ng, "graupel"},
        {config.has_qh, state.qh, config.has_Nh, state.Nh, "hail"},
        {config.has_qi, state.qi, config.has_Ni, state.Ni, "ice"},
    }};
}

} // namespace

void validate_radar_dimensions(const RadarStateView& state,
                               int nr, int nth, int nz,
                               const char* operator_name)
{
    if (nr <= 0 || nth <= 0 || nz <= 0
        || state.NR != nr || state.NTH != nth || state.NZ != nz)
    {
        throw std::invalid_argument(
            std::string(operator_name) +
            " radar state dimensions do not match the initialized operator");
    }
}

void validate_radar_hydrometeor_state(const RadarConfig& config,
                                      const RadarStateView& state,
                                      const char* operator_name,
                                      bool require_density)
{
    const auto species = species_inputs(config, state);
    bool any_configured = false;
    for (const SpeciesInput& input : species)
    {
        if (!input.configured) continue;
        any_configured = true;
        if (!field_matches(input.mixing_ratio, state))
            throw std::invalid_argument(
                std::string(operator_name) + " radar requires a matching "
                + input.name + " mixing-ratio field");
        if (input.number_configured && !field_matches(input.number, state))
            throw std::invalid_argument(
                std::string(operator_name) + " radar requires a matching "
                + input.name + " number-concentration field");
    }
    if (require_density && any_configured && !field_matches(state.rho, state))
        throw std::invalid_argument(
            std::string(operator_name) +
            " radar requires matching prognostic dry-air density");

    std::size_t invalid = 0;
    const std::size_t count = static_cast<std::size_t>(state.NR)
        * static_cast<std::size_t>(state.NTH)
        * static_cast<std::size_t>(state.NZ);
    #pragma omp parallel for if(count >= 32768) reduction(+:invalid)
    for (std::size_t index = 0; index < count; ++index)
    {
        if (require_density && any_configured)
        {
            const float density = state.rho->data()[index];
            if (!std::isfinite(static_cast<double>(density))
                || density < density_min_kgm3 || density > density_max_kgm3)
                ++invalid;
        }
        for (const SpeciesInput& input : species)
        {
            if (!input.configured) continue;
            const float q = input.mixing_ratio->data()[index];
            if (!std::isfinite(static_cast<double>(q))
                || q < 0.0f || q > hydrometeor_max_kgkg)
            {
                ++invalid;
                continue;
            }
            if (input.number_configured)
            {
                const float number = input.number->data()[index];
                if (!std::isfinite(static_cast<double>(number))
                    || number < 0.0f || (q > 0.0f && number < 1.0f))
                    ++invalid;
            }
        }
    }
    if (invalid != 0)
        throw std::invalid_argument(
            std::string(operator_name) + " radar rejected "
            + std::to_string(invalid)
            + " invalid density, hydrometeor, or number samples");
}

void validate_radar_velocity_state(const RadarConfig& config,
                                   const RadarStateView& state)
{
    if (config.sampling != "point" && config.sampling != "beam_volume")
        throw std::invalid_argument(
            "radar velocity sampling must be point or beam_volume");
    if (!std::isfinite(config.radar_x) || !std::isfinite(config.radar_y)
        || !std::isfinite(config.radar_z))
        throw std::invalid_argument(
            "radar velocity location must be finite");
    if (!field_matches(state.u, state) || !field_matches(state.v, state)
        || !field_matches(state.w, state))
        throw std::invalid_argument(
            "radar velocity requires matching u, v, and w fields");

    std::size_t invalid = 0;
    const std::size_t count = state.u->size();
    #pragma omp parallel for if(count >= 32768) reduction(+:invalid)
    for (std::size_t index = 0; index < count; ++index)
    {
        const float u = state.u->data()[index];
        const float v = state.v->data()[index];
        const float w = state.w->data()[index];
        if (!std::isfinite(static_cast<double>(u))
            || std::abs(u) > wind_horizontal_abs_max_ms
            || !std::isfinite(static_cast<double>(v))
            || std::abs(v) > wind_horizontal_abs_max_ms
            || !std::isfinite(static_cast<double>(w))
            || std::abs(w) > wind_vertical_abs_max_ms)
            ++invalid;
    }
    if (invalid != 0)
        throw std::invalid_argument(
            "radar velocity rejected " + std::to_string(invalid)
            + " invalid wind samples");

    if (config.apply_scatterer_correction)
        validate_radar_hydrometeor_state(
            config, state, "velocity", /*require_density=*/false);
}

void validate_radar_physical_geometry(const RadarStateView& state)
{
    if (!std::isfinite(state.grid_origin_earth_x_m)
        || !std::isfinite(state.grid_origin_earth_y_m))
        throw std::invalid_argument(
            "radar velocity requires a finite Earth-relative grid origin");
    if (state.geometry == nullptr
        || state.geometry->coord != CoordinateSystem::Cartesian
        || state.geometry_i_offset < 0
        || state.geometry_j_offset < 0
        || state.geometry_k_offset < 0
        || state.geometry_i_offset + state.NR
            > static_cast<int>(state.geometry->r.size())
        || state.geometry_j_offset + state.NTH
            > static_cast<int>(state.geometry->theta.size())
        || state.geometry_k_offset + state.NZ
            > static_cast<int>(state.geometry->z.size()))
        throw std::invalid_argument(
            "radar velocity requires matching caller-owned Cartesian coordinates");
    if (state.terrain_metrics != nullptr
        && (state.geometry_i_offset + state.NR
                > state.terrain_metrics->z.size_r()
            || state.geometry_j_offset + state.NTH
                > state.terrain_metrics->z.size_th()
            || state.geometry_k_offset + state.NZ
                > state.terrain_metrics->z.size_z()))
        throw std::invalid_argument(
            "radar velocity terrain coordinates do not match the supplied state");

    std::size_t invalid = 0;
    const std::size_t count = static_cast<std::size_t>(state.NR)
        * static_cast<std::size_t>(state.NTH)
        * static_cast<std::size_t>(state.NZ);
    #pragma omp parallel for if(count >= 32768) reduction(+:invalid)
    for (std::size_t index = 0; index < count; ++index)
    {
        const int k = static_cast<int>(index % static_cast<std::size_t>(state.NZ));
        const std::size_t horizontal = index / static_cast<std::size_t>(state.NZ);
        const int j = static_cast<int>(horizontal
            % static_cast<std::size_t>(state.NTH));
        const int i = static_cast<int>(horizontal
            / static_cast<std::size_t>(state.NTH));
        const int gi = state.geometry_i_offset + i;
        const int gj = state.geometry_j_offset + j;
        const int gk = state.geometry_k_offset + k;
        const double x = state.grid_origin_earth_x_m
            + state.geometry->r[static_cast<std::size_t>(gi)];
        const double y = state.grid_origin_earth_y_m
            + state.geometry->theta[static_cast<std::size_t>(gj)];
        const double z = state.terrain_metrics != nullptr
            ? state.terrain_metrics->z(gi, gj, gk)
            : state.geometry->z[static_cast<std::size_t>(gk)];
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
            ++invalid;
    }
    if (invalid != 0)
        throw std::invalid_argument(
            "radar velocity physical coordinates contain "
            + std::to_string(invalid) + " invalid samples");
}

void validate_radar_output_field(const Field3D& field,
                                 int nr, int nth, int nz,
                                 float minimum, float maximum,
                                 const char* field_name)
{
    if (field.size_r() != nr || field.size_th() != nth || field.size_z() != nz)
        throw std::runtime_error(
            std::string("radar candidate ") + field_name
            + " has the wrong shape");
    std::size_t invalid = 0;
    #pragma omp parallel for if(field.size() >= 32768) reduction(+:invalid)
    for (std::size_t index = 0; index < field.size(); ++index)
    {
        const float value = field.data()[index];
        if (!std::isfinite(static_cast<double>(value))
            || value < minimum || value > maximum)
            ++invalid;
    }
    if (invalid != 0)
        throw std::runtime_error(
            std::string("radar candidate ") + field_name + " contains "
            + std::to_string(invalid) + " invalid samples");
}

float radar_bulk_rain_fall_speed_ms(float q_rain)
{
    if (!std::isfinite(static_cast<double>(q_rain)) || q_rain < 0.0f
        || q_rain > hydrometeor_max_kgkg)
        throw std::invalid_argument(
            "radar rain fall speed requires finite in-range rain mixing ratio");
    if (q_rain == 0.0f)
        return 0.0f;
    const float speed = 9.0f * std::pow(q_rain * 1000.0f, 0.20f);
    if (!std::isfinite(static_cast<double>(speed)))
        throw std::runtime_error(
            "radar rain fall-speed evaluation produced a non-finite result");
    return std::min(speed, 40.0f);
}

float radar_bulk_rain_rate_mmph(float q_rain, float rho_dry)
{
    if (!std::isfinite(static_cast<double>(rho_dry))
        || rho_dry < density_min_kgm3 || rho_dry > density_max_kgm3)
        throw std::invalid_argument(
            "radar rain rate requires finite in-range dry-air density");
    const float rate = rho_dry * q_rain *
                       radar_bulk_rain_fall_speed_ms(q_rain) * 3600.0f;
    if (!std::isfinite(static_cast<double>(rate)))
        throw std::runtime_error(
            "radar rain-rate evaluation produced a non-finite result");
    return rate;
}

/**
 * @brief Computes radar line-of-sight unit vector and range.
 */
void RadarGeometry::compute_line_of_sight(double radar_x, double radar_y, double radar_z,
                                         double x, double y, double z,
                                         double& e_r_x, double& e_r_y, double& e_r_z,
                                         double& R) 
                                         
{
    double dx = x - radar_x;
    double dy = y - radar_y;
    double dz = z - radar_z;

    R = std::sqrt(dx*dx + dy*dy + dz*dz);

    if (R < 1e-6) {
        e_r_x = 0.0;
        e_r_y = 0.0;
        e_r_z = 1.0;
        return;
    }

    e_r_x = dx / R;
    e_r_y = dy / R;
    e_r_z = dz / R;
}

/**
 * @brief Samples full 3D state into a single-cell radar point view.
 */
void RadarGeometry::sample_state_point(const RadarStateView& state, int i, int j, int k,
                                      RadarStateView& point_state) 
{
    point_state = RadarStateView{};
    if (state.NR <= 0 || state.NTH <= 0 || state.NZ <= 0)
    {
        return;
    }

    const PhysicalCellDomain domain =
        physical_scalar_domain(state.NR, state.NTH, state.NZ, state.staggering);
    if (domain.empty())
    {
        return;
    }
    const int ic = std::clamp(i, domain.i_begin, domain.i_end - 1);
    const int jc = std::clamp(j, domain.j_begin, domain.j_end - 1);
    const int kc = std::clamp(k, domain.k_begin, domain.k_end - 1);

    thread_local RadarPointScratch scratch;
    initialize_scratch_fields(scratch);

    point_state.NR = 1;
    point_state.NTH = 1;
    point_state.NZ = 1;
    point_state.staggering = StaggerType::Collocated;
    point_state.geometry = state.geometry;
    point_state.terrain_metrics = state.terrain_metrics;
    point_state.geometry_i_offset = state.geometry_i_offset + ic;
    point_state.geometry_j_offset = state.geometry_j_offset + jc;
    point_state.geometry_k_offset = state.geometry_k_offset + kc;
    point_state.grid_origin_earth_x_m = state.grid_origin_earth_x_m;
    point_state.grid_origin_earth_y_m = state.grid_origin_earth_y_m;
    point_state.u = sample_wind_to_single_cell(
        state.u, ic, jc, kc, state.staggering, WindAxis::X, scratch.u);
    point_state.v = sample_wind_to_single_cell(
        state.v, ic, jc, kc, state.staggering, WindAxis::Y, scratch.v);
    point_state.w = sample_wind_to_single_cell(
        state.w, ic, jc, kc, state.staggering, WindAxis::Z, scratch.w);
    point_state.qr = sample_to_single_cell(state.qr, ic, jc, kc, scratch.qr);
    point_state.qs = sample_to_single_cell(state.qs, ic, jc, kc, scratch.qs);
    point_state.qg = sample_to_single_cell(state.qg, ic, jc, kc, scratch.qg);
    point_state.qh = sample_to_single_cell(state.qh, ic, jc, kc, scratch.qh);
    point_state.qi = sample_to_single_cell(state.qi, ic, jc, kc, scratch.qi);
    point_state.Nr = sample_to_single_cell(state.Nr, ic, jc, kc, scratch.Nr);
    point_state.Ns = sample_to_single_cell(state.Ns, ic, jc, kc, scratch.Ns);
    point_state.Ng = sample_to_single_cell(state.Ng, ic, jc, kc, scratch.Ng);
    point_state.Nh = sample_to_single_cell(state.Nh, ic, jc, kc, scratch.Nh);
    point_state.Ni = sample_to_single_cell(state.Ni, ic, jc, kc, scratch.Ni);
    point_state.theta = sample_to_single_cell(state.theta, ic, jc, kc, scratch.theta);
    point_state.p = sample_to_single_cell(state.p, ic, jc, kc, scratch.p);
    point_state.rho = sample_to_single_cell(state.rho, ic, jc, kc, scratch.rho);
}

}  // namespace tmv
