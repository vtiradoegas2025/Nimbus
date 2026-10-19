#include "explicit.hpp"
#include "compute/compute_backend.hpp"
#include "compute/compute_kernel_template.hpp"
#include "terrain/conservative_transport.hpp"
#include "turbulence/base/collocated_eddy_viscosity.hpp"
#include "turbulence/base/eddy_viscosity.hpp"
#include "util/grid_metric_utils.hpp"
#include "util/log.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
inline bool matches_shape(const Field3D* field, int nr, int nth, int nz)
{
    return field != nullptr &&
           field->size_r() == nr &&
           field->size_th() == nth &&
           field->size_z() == nz;
}

inline void ensure_tendency_shape(Field3D& field, int nr, int nth, int nz)
{
    if (field.size_r() != nr || field.size_th() != nth || field.size_z() != nz)
    {
        field.resize(nr, nth, nz, 0.0f);
    }
    else
    {
        field.fill(0.0f);
    }
}

struct HorizontalDomain
{
    int i_begin;
    int i_end;
    int j_begin;
    int j_end;
    int k_begin;
    int k_end;
};

inline HorizontalDomain scalar_horizontal_domain(
    int nr, int nth, int nz, bool c_grid)
{
    return {1, nr - 1, 1, nth - 1,
            c_grid ? 0 : 1, c_grid ? nz : nz - 1};
}

inline HorizontalDomain u_horizontal_domain(
    int nr, int nth, int nz, bool c_grid, bool periodic_x)
{
    if (!c_grid)
        return scalar_horizontal_domain(nr, nth, nz, false);
    // u[0] and, for non-periodic topology, u[nr-2] are boundary-normal
    // faces owned by the lateral boundary scheme. Under periodic topology
    // u[0] duplicates the prognostic seam face u[nr-2].
    return {1, periodic_x ? nr - 1 : nr - 2,
            1, nth - 1, 0, nz - 1};
}

inline HorizontalDomain v_horizontal_domain(
    int nr, int nth, int nz, bool c_grid, bool periodic_y)
{
    if (!c_grid)
        return scalar_horizontal_domain(nr, nth, nz, false);
    return {1, nr - 1,
            1, periodic_y ? nth - 1 : nth - 2, 0, nz - 1};
}

inline HorizontalDomain w_horizontal_domain(
    int nr, int nth, int nz, bool c_grid)
{
    if (!c_grid)
        return scalar_horizontal_domain(nr, nth, nz, false);
    // w[0] is prognostic; w[nz-1] is the boundary-owned lid face.
    return {1, nr - 1, 1, nth - 1, 0, nz - 1};
}

void validate_device_target(const Field3D& field,
                            const HorizontalDomain& domain,
                            const char* label)
{
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
                if (!std::isfinite(static_cast<double>(field(i, j, k))))
                    throw std::invalid_argument(
                        std::string("device explicit ") + label +
                        " diffusion requires finite prognostic state");
}

void validate_device_coefficient(const Field3D& field,
                                 const HorizontalDomain& domain,
                                 const char* label)
{
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double value = static_cast<double>(field(i, j, k));
                if (!std::isfinite(value) || value < 0.0)
                    throw std::invalid_argument(
                        std::string("device explicit ") + label +
                        " diffusion requires non-negative finite coefficients");
            }
}

template <typename DensityAt>
void add_density_weighted_horizontal_component(
    const Field3D& field,
    double K_h,
    const GridMetrics& grid,
    const HorizontalDomain& domain,
    bool periodic_x,
    bool periodic_y,
    DensityAt density_at,
    const char* label,
    Field3D& tendency)
{
    if (domain.i_begin >= domain.i_end ||
        domain.j_begin >= domain.j_end ||
        domain.k_begin >= domain.k_end)
        throw std::invalid_argument(
            std::string("horizontal explicit ") + label +
            " diffusion has no prognostic control volumes");

    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double value = static_cast<double>(field(i, j, k));
                const double density = density_at(i, j, k);
                if (!std::isfinite(value) || !std::isfinite(density) ||
                    density <= 0.0)
                    throw std::invalid_argument(
                        std::string("horizontal explicit ") + label +
                        " diffusion requires finite prognostic state and "
                        "positive finite live control-volume density");
            }

    const double inv_dx2 = 1.0 / (grid.dx * grid.dx);
    const double inv_dy2 = 1.0 / (grid.dy * grid.dy);
    #pragma omp parallel for collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const bool west_closed = i == domain.i_begin && !periodic_x;
                const bool east_closed = i == domain.i_end - 1 && !periodic_x;
                const bool south_closed = j == domain.j_begin && !periodic_y;
                const bool north_closed = j == domain.j_end - 1 && !periodic_y;
                const int iw = i == domain.i_begin && periodic_x
                    ? domain.i_end - 1 : i - 1;
                const int ie = i == domain.i_end - 1 && periodic_x
                    ? domain.i_begin : i + 1;
                const int js = j == domain.j_begin && periodic_y
                    ? domain.j_end - 1 : j - 1;
                const int jn = j == domain.j_end - 1 && periodic_y
                    ? domain.j_begin : j + 1;
                const double rho_c = density_at(i, j, k);
                const double rho_w = west_closed ? 0.0
                    : 0.5 * (rho_c + density_at(iw, j, k));
                const double rho_e = east_closed ? 0.0
                    : 0.5 * (rho_c + density_at(ie, j, k));
                const double rho_s = south_closed ? 0.0
                    : 0.5 * (rho_c + density_at(i, js, k));
                const double rho_n = north_closed ? 0.0
                    : 0.5 * (rho_c + density_at(i, jn, k));
                const double value = static_cast<double>(field(i, j, k));
                const double x_div = (
                    (east_closed ? 0.0 : rho_e *
                        (static_cast<double>(field(ie, j, k)) - value)) -
                    (west_closed ? 0.0 : rho_w *
                        (value - static_cast<double>(field(iw, j, k))))) * inv_dx2;
                const double y_div = (
                    (north_closed ? 0.0 : rho_n *
                        (static_cast<double>(field(i, jn, k)) - value)) -
                    (south_closed ? 0.0 : rho_s *
                        (value - static_cast<double>(field(i, js, k))))) * inv_dy2;
                tendency(i, j, k) = static_cast<float>(
                    static_cast<double>(tendency(i, j, k)) +
                    K_h * (x_div + y_div) / rho_c);
            }
}

template <typename DensityAt>
double horizontal_component_diffusion_number(
    double K_h,
    double dt_diffusion,
    const GridMetrics& grid,
    const HorizontalDomain& domain,
    bool periodic_x,
    bool periodic_y,
    DensityAt density_at)
{
    double maximum = 0.0;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const bool west_closed = i == domain.i_begin && !periodic_x;
                const bool east_closed = i == domain.i_end - 1 && !periodic_x;
                const bool south_closed = j == domain.j_begin && !periodic_y;
                const bool north_closed = j == domain.j_end - 1 && !periodic_y;
                const int iw = i == domain.i_begin && periodic_x
                    ? domain.i_end - 1 : i - 1;
                const int ie = i == domain.i_end - 1 && periodic_x
                    ? domain.i_begin : i + 1;
                const int js = j == domain.j_begin && periodic_y
                    ? domain.j_end - 1 : j - 1;
                const int jn = j == domain.j_end - 1 && periodic_y
                    ? domain.j_begin : j + 1;
                const double rho_c = density_at(i, j, k);
                const double rho_w = west_closed ? 0.0
                    : 0.5 * (rho_c + density_at(iw, j, k));
                const double rho_e = east_closed ? 0.0
                    : 0.5 * (rho_c + density_at(ie, j, k));
                const double rho_s = south_closed ? 0.0
                    : 0.5 * (rho_c + density_at(i, js, k));
                const double rho_n = north_closed ? 0.0
                    : 0.5 * (rho_c + density_at(i, jn, k));
                if (!std::isfinite(rho_c) || rho_c <= 0.0 ||
                    !std::isfinite(rho_w) || rho_w < 0.0 ||
                    !std::isfinite(rho_e) || rho_e < 0.0 ||
                    !std::isfinite(rho_s) || rho_s < 0.0 ||
                    !std::isfinite(rho_n) || rho_n < 0.0)
                    throw std::invalid_argument(
                        "horizontal explicit diffusion stability requires "
                        "positive finite live control-volume density");
                const double local = K_h * dt_diffusion / rho_c *
                    ((rho_w + rho_e) / (grid.dx * grid.dx) +
                     (rho_s + rho_n) / (grid.dy * grid.dy));
                maximum = std::max(maximum, local);
            }
    return maximum;
}

inline double vertical_point_distance(
    const GridMetrics& grid, int lower_k, bool c_grid_w)
{
    if (lower_k < 0 || lower_k + 1 >= static_cast<int>(grid.dz.size()))
        throw std::out_of_range(
            "vertical diffusion point distance is outside the grid");
    if (c_grid_w)
        return grid.dz[static_cast<std::size_t>(lower_k + 1)];
    return 0.5 * (
        grid.dz[static_cast<std::size_t>(lower_k)] +
        grid.dz[static_cast<std::size_t>(lower_k + 1)]);
}

template <typename ControlMassAt, typename TransportAt,
          typename FaceTransportAt>
void add_density_weighted_vertical_component(
    const Field3D& field,
    const GridMetrics& grid,
    const HorizontalDomain& domain,
    bool c_grid_w,
    ControlMassAt control_mass_at,
    TransportAt transport_at,
    FaceTransportAt face_transport_at,
    const char* label,
    Field3D& tendency)
{
    if (domain.i_begin >= domain.i_end ||
        domain.j_begin >= domain.j_end ||
        domain.k_begin >= domain.k_end)
        throw std::invalid_argument(
            std::string("explicit vertical ") + label +
            " diffusion has no prognostic control volumes");

    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double value = static_cast<double>(field(i, j, k));
                const double control_mass = control_mass_at(i, j, k);
                const double transport = transport_at(i, j, k);
                if (!std::isfinite(value) || !std::isfinite(control_mass) ||
                    control_mass <= 0.0 || !std::isfinite(transport) ||
                    transport < 0.0 ||
                    (k > domain.k_begin &&
                     (!std::isfinite(face_transport_at(i, j, k - 1)) ||
                      face_transport_at(i, j, k - 1) < 0.0)) ||
                    (k + 1 < domain.k_end &&
                     (!std::isfinite(face_transport_at(i, j, k)) ||
                      face_transport_at(i, j, k) < 0.0)))
                    throw std::invalid_argument(
                        std::string("explicit vertical ") + label +
                        " diffusion requires finite state, positive live "
                        "control-volume mass, and non-negative "
                        "finite density-weighted diffusivity");
            }

    #pragma omp parallel for collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const bool lower_closed = k == domain.k_begin;
                const bool upper_closed = k == domain.k_end - 1;
                const double control_mass = control_mass_at(i, j, k);
                const double value = static_cast<double>(field(i, j, k));
                double lower_flux = 0.0;
                double upper_flux = 0.0;
                if (!lower_closed)
                {
                    const double face_transport =
                        face_transport_at(i, j, k - 1);
                    const double distance = vertical_point_distance(
                        grid, k - 1, c_grid_w);
                    lower_flux = face_transport *
                        (value - static_cast<double>(field(i, j, k - 1))) /
                        distance;
                }
                if (!upper_closed)
                {
                    const double face_transport =
                        face_transport_at(i, j, k);
                    const double distance = vertical_point_distance(
                        grid, k, c_grid_w);
                    upper_flux = face_transport *
                        (static_cast<double>(field(i, j, k + 1)) - value) /
                        distance;
                }
                tendency(i, j, k) = static_cast<float>(
                    static_cast<double>(tendency(i, j, k)) +
                    (upper_flux - lower_flux) / control_mass);
            }
}

template <typename ControlMassAt, typename TransportAt,
          typename FaceTransportAt>
double vertical_component_diffusion_number(
    double dt_diffusion,
    const GridMetrics& grid,
    const HorizontalDomain& domain,
    bool c_grid_w,
    ControlMassAt control_mass_at,
    TransportAt transport_at,
    FaceTransportAt face_transport_at)
{
    double maximum = 0.0;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double control_mass = control_mass_at(i, j, k);
                const double transport = transport_at(i, j, k);
                double outgoing = 0.0;
                if (k > domain.k_begin)
                    outgoing += face_transport_at(i, j, k - 1) /
                        vertical_point_distance(grid, k - 1, c_grid_w);
                if (k + 1 < domain.k_end)
                    outgoing += face_transport_at(i, j, k) /
                        vertical_point_distance(grid, k, c_grid_w);
                if (!std::isfinite(control_mass) || control_mass <= 0.0 ||
                    !std::isfinite(transport) || transport < 0.0 ||
                    !std::isfinite(outgoing) || outgoing < 0.0)
                    throw std::invalid_argument(
                        "explicit vertical diffusion stability requires "
                        "positive finite control volumes and non-negative "
                        "finite density-weighted diffusivity");
                maximum = std::max(maximum,
                    dt_diffusion * outgoing / control_mass);
            }
    return maximum;
}

}

namespace tmv
{

ExplicitDiffusionScheme::ExplicitDiffusionScheme()
{
}

void ExplicitDiffusionScheme::initialize()
{
    initialize(DiffusionConfig{});
}

void ExplicitDiffusionScheme::initialize(const DiffusionConfig& cfg)
{
    if (cfg.operator_order != 2)
    {
        throw std::invalid_argument(
            "explicit diffusion implements only operator_order=2; got " +
            std::to_string(cfg.operator_order));
    }
    if (cfg.operator_type != "laplacian")
        throw std::invalid_argument(
            "explicit diffusion implements only the laplacian operator");
    if (!std::isfinite(cfg.K_h) || cfg.K_h < 0.0)
        throw std::invalid_argument(
            "explicit diffusion K_h must be non-negative and finite");
    if (!std::isfinite(cfg.K_v) || cfg.K_v < 0.0)
        throw std::invalid_argument(
            "explicit diffusion K_v must be non-negative and finite");
    if (!std::isfinite(cfg.dt_diffusion) || cfg.dt_diffusion <= 0.0)
        throw std::invalid_argument(
            "explicit diffusion dt_diffusion must be positive and finite");
    if (cfg.apply_to != "scalars" &&
        cfg.apply_to != "scalars_and_momentum" &&
        cfg.apply_to != "momentum" && cfg.apply_to != "all")
        throw std::invalid_argument(
            "explicit diffusion apply_to has an unknown target");
    if (cfg.implicit_dim != "none" && !cfg.implicit_dim.empty())
        throw std::invalid_argument(
            "explicit diffusion does not accept an implicit dimension");
    config_ = cfg;

    tmv::log_info("Initialized explicit diffusion scheme");
    tmv::log_info("  K_h = ", cfg.K_h, " m²/s, K_v = ", cfg.K_v, " m²/s");
}

void ExplicitDiffusionScheme::compute_diffusion_tendencies(
    const DiffusionConfig& cfg,
    const DiffusionStateView& state,
    DiffusionTendencies& tendencies,
    DiffusionDiagnostics* diag_opt
)
{
    if (!state.grid)
    {
        throw std::runtime_error("Invalid grid for explicit diffusion");
    }

    const auto* ref_field = state.theta ? state.theta : (state.u ? state.u : state.qv);
    if (ref_field == nullptr || ref_field->empty())
        throw std::invalid_argument(
            "explicit diffusion requires at least one non-empty requested field");
    const int nr = ref_field->size_r();
    const int nth = ref_field->size_th();
    const int nz = ref_field->size_z();

    grid_metric::validate_vertical_metrics(*state.grid, nr, nth, nz);
    if ((cfg.K_h > 0.0 || cfg.K_v > 0.0 || cfg.use_variable_K) &&
        !matches_shape(state.rho, nr, nth, nz))
        throw std::invalid_argument(
            "active explicit diffusion requires a matching live-density field");

    const Field3D* supplied[] = {
        state.theta, state.qv, state.u, state.v, state.w
    };
    for (const Field3D* field : supplied)
        if (field != nullptr && !matches_shape(field, nr, nth, nz))
            throw std::invalid_argument(
                "explicit diffusion target field shapes disagree");
    const int momentum_fields = static_cast<int>(state.u != nullptr) +
        static_cast<int>(state.v != nullptr) +
        static_cast<int>(state.w != nullptr);
    if (momentum_fields != 0 && momentum_fields != 3)
        throw std::invalid_argument(
            "explicit momentum diffusion requires u, v, and w together");
    const bool terrain_active = state.grid->terrain_metrics_active;
    if (terrain_active)
    {
        if (state.grid->terrain_metrics == nullptr)
            throw std::invalid_argument(
                "terrain diffusion requires bound conservative geometry");
    }

    if (cfg.use_variable_K && state.theta &&
        !matches_shape(state.K_scalar, nr, nth, nz))
        throw std::invalid_argument(
            "explicit diffusion requested variable scalar K with a missing or mismatched field");
    if (cfg.use_variable_K && state.u &&
        !matches_shape(state.K_momentum, nr, nth, nz))
        throw std::invalid_argument(
            "explicit diffusion requested variable momentum K with a missing or mismatched field");
    const bool has_variable_scalar_k = cfg.use_variable_K && matches_shape(state.K_scalar, nr, nth, nz);
    const bool has_variable_momentum_k = cfg.use_variable_K && matches_shape(state.K_momentum, nr, nth, nz);
    const Field3D* scalar_k = has_variable_scalar_k ? state.K_scalar : nullptr;
    const Field3D* momentum_k = has_variable_momentum_k ? state.K_momentum : nullptr;

    const double scalar_k_default = cfg.K_v;
    const double momentum_k_default = cfg.K_v;

    // Build the complete result privately. Validation and backend failures must
    // leave caller-owned tendencies byte-for-byte untouched.
    DiffusionTendencies candidate;
    ensure_tendency_shape(candidate.dudt_diff, nr, nth, nz);
    ensure_tendency_shape(candidate.dvdt_diff, nr, nth, nz);
    ensure_tendency_shape(candidate.dwdt_diff, nr, nth, nz);
    ensure_tendency_shape(candidate.dthetadt_diff, nr, nth, nz);
    ensure_tendency_shape(candidate.dqvdt_diff, nr, nth, nz);

    bool device_computed = false;
    const bool native_device_requested =
        state.compute_runtime != nullptr && active_compute_backend_kind(*state.compute_runtime) != ComputeBackendKind::Cpu &&
        !compute_backend_fallback_active(*state.compute_runtime);
    if (native_device_requested)
    {
        const bool device_eligible = !state.grid->terrain_metrics_active;
        if (!device_eligible && !state.compute_runtime->config.allow_fallback)
            throw std::runtime_error(
                "active accelerator diffusion requires a flat Cartesian grid; "
                "CPU substitution is disabled");

        if (device_eligible)
        {
            // The device kernel assumes the same validated positive-density
            // equation as the reference. check_stability performs the complete
            // component-density/transport scan even when diagnostics are off.
            (void)check_stability(cfg, state);
            device_dz_scratch_.resize(state.grid->dz.size());
            for (std::size_t k = 0; k < state.grid->dz.size(); ++k)
            {
                const double value = state.grid->dz[k];
                const float packed = static_cast<float>(value);
                if (!std::isfinite(packed) || packed <= 0.0f)
                    throw std::invalid_argument(
                        "device explicit diffusion cannot represent dz in float32");
                device_dz_scratch_[k] = packed;
            }
            const float dx_device = static_cast<float>(state.grid->dx);
            const float dy_device = static_cast<float>(state.grid->dy);
            const float kh_device = static_cast<float>(cfg.K_h);
            const float kv_device = static_cast<float>(cfg.K_v);
            if (!std::isfinite(dx_device) || dx_device <= 0.0f ||
                !std::isfinite(dy_device) || dy_device <= 0.0f ||
                !std::isfinite(kh_device) || kh_device < 0.0f ||
                !std::isfinite(kv_device) || kv_device < 0.0f)
                throw std::invalid_argument(
                    "device explicit diffusion metrics or coefficients are "
                    "not representable in float32");

            constexpr int kMaximumDiffusionTargets = 5;
            const float* device_fields[kMaximumDiffusionTargets]{};
            const float* device_vertical_k[kMaximumDiffusionTargets]{};
            float* device_tendencies[kMaximumDiffusionTargets]{};
            DiffusionFieldLocation device_locations[
                kMaximumDiffusionTargets]{};
            int device_target_count = 0;
            const auto append_target = [&](const Field3D* field,
                                           Field3D& tendency,
                                           DiffusionFieldLocation location,
                                           const HorizontalDomain& domain,
                                           const char* label,
                                           const Field3D* vertical_k)
            {
                if (field == nullptr) return;
                validate_device_target(*field, domain, label);
                if (vertical_k != nullptr)
                    validate_device_coefficient(*vertical_k, domain, label);
                device_fields[device_target_count] = field->data();
                device_vertical_k[device_target_count] =
                    vertical_k == nullptr ? nullptr : vertical_k->data();
                device_tendencies[device_target_count] = tendency.data();
                device_locations[device_target_count] = location;
                ++device_target_count;
            };

            if (state.u && state.v && state.w)
            {
                append_target(
                    state.u, candidate.dudt_diff,
                    DiffusionFieldLocation::UFace,
                    u_horizontal_domain(nr, nth, nz, state.c_grid,
                                        state.periodic_x), "u-momentum",
                    momentum_k);
                append_target(
                    state.v, candidate.dvdt_diff,
                    DiffusionFieldLocation::VFace,
                    v_horizontal_domain(nr, nth, nz, state.c_grid,
                                        state.periodic_y), "v-momentum",
                    momentum_k);
                append_target(
                    state.w, candidate.dwdt_diff,
                    DiffusionFieldLocation::WFace,
                    w_horizontal_domain(nr, nth, nz, state.c_grid),
                    "w-momentum", momentum_k);
            }
            append_target(
                state.theta, candidate.dthetadt_diff,
                DiffusionFieldLocation::Scalar,
                scalar_horizontal_domain(nr, nth, nz, state.c_grid), "scalar",
                scalar_k);
            append_target(
                state.qv, candidate.dqvdt_diff,
                DiffusionFieldLocation::Scalar,
                scalar_horizontal_domain(nr, nth, nz, state.c_grid), "scalar",
                scalar_k);

            const DiffusionBatchDispatch batch{
                device_fields, device_tendencies, device_locations,
                device_target_count,
                state.rho->data(), device_dz_scratch_.data(),
                nr, nth, nz,
                dx_device, dy_device, kh_device, kv_device,
                state.c_grid, state.periodic_x, state.periodic_y,
                cfg.use_variable_K ? device_vertical_k : nullptr};
            device_computed =
                device_target_count > 0 &&
                dispatch_diffusion_batch_backend(*state.compute_runtime, batch);

            if (!device_computed)
            {
                if (!state.compute_runtime->config.allow_fallback)
                    throw std::runtime_error(
                        "native accelerator finite-volume diffusion dispatch failed "
                        "and CPU substitution is disabled");
                ensure_tendency_shape(candidate.dudt_diff, nr, nth, nz);
                ensure_tendency_shape(candidate.dvdt_diff, nr, nth, nz);
                ensure_tendency_shape(candidate.dwdt_diff, nr, nth, nz);
                ensure_tendency_shape(candidate.dthetadt_diff, nr, nth, nz);
                ensure_tendency_shape(candidate.dqvdt_diff, nr, nth, nz);
                tmv::log_warn(
                    "[DIFFUSION][KERNEL] native accelerator dispatch failed; "
                    "using the CPU finite-volume reference");
            }
        }
    }

    if (!device_computed && state.u && state.v && state.w &&
        matches_shape(state.u, nr, nth, nz) &&
        matches_shape(state.v, nr, nth, nz) &&
        matches_shape(state.w, nr, nth, nz))
    {
        if (terrain_active)
        {
            TurbulenceStateView viscosity_state;
            static_cast<FlowExecutionInputs&>(viscosity_state) = state;
            viscosity_state.u = state.u;
            viscosity_state.v = state.v;
            viscosity_state.w = state.w;
            viscosity_state.rho = state.rho;
            viscosity_state.c_grid = state.c_grid;
            viscosity_state.NR = nr;
            viscosity_state.NTH = nth;
            viscosity_state.NZ = nz;
            const auto add_pass = [&](const Field3D& viscosity,
                                      const char* mode)
            {
                TurbulenceConfig tensor_cfg;
                tensor_cfg.mode = mode;
                Field3D du(nr, nth, nz, 0.0f);
                Field3D dv(nr, nth, nz, 0.0f);
                Field3D dw(nr, nth, nz, 0.0f);
                eddy_viscosity::compute_momentum_diffusion_tendencies(
                    tensor_cfg, viscosity_state, *state.grid, viscosity,
                    du, dv, dw, state.periodic_x, state.periodic_y);
                for (std::size_t index = 0; index < du.size(); ++index)
                {
                    candidate.dudt_diff.data()[index] += du.data()[index];
                    candidate.dvdt_diff.data()[index] += dv.data()[index];
                    candidate.dwdt_diff.data()[index] += dw.data()[index];
                }
            };
            if (cfg.K_h > 0.0)
            {
                Field3D horizontal_k(nr, nth, nz,
                                     static_cast<float>(cfg.K_h));
                add_pass(horizontal_k, "horizontal_only");
            }
            if (cfg.K_v > 0.0 || momentum_k != nullptr)
            {
                Field3D vertical_k(nr, nth, nz,
                    static_cast<float>(momentum_k_default));
                if (momentum_k != nullptr) vertical_k = *momentum_k;
                add_pass(vertical_k, "vertical_only");
            }
        }
        else
        {
            compute_momentum_diffusion(*state.u, *state.v, *state.w, *state.rho,
                                       momentum_k, momentum_k_default, *state.grid,
                                       state.c_grid, state.periodic_x,
                                       state.periodic_y,
                                       candidate.dudt_diff, candidate.dvdt_diff,
                                       candidate.dwdt_diff);
            if (cfg.K_h > 0.0)
                add_horizontal_momentum_diffusion(
                    *state.u, *state.v, *state.w, *state.rho, cfg.K_h,
                    *state.grid, state.c_grid, state.periodic_x, state.periodic_y,
                    candidate.dudt_diff, candidate.dvdt_diff,
                    candidate.dwdt_diff);
        }
    }

    if (!device_computed && state.theta && matches_shape(state.theta, nr, nth, nz))
    {
        if (terrain_active && state.c_grid)
            terrain_transport::cgrid_scalar_diffusion_tendency(
                *state.grid->terrain_metrics, *state.theta, *state.rho,
                scalar_k, cfg.K_h, scalar_k_default,
                state.periodic_x, state.periodic_y,
                candidate.dthetadt_diff);
        else if (terrain_active)
        {
            TurbulenceStateView scalar_state;
            static_cast<FlowExecutionInputs&>(scalar_state) = state;
            scalar_state.u = state.theta;
            scalar_state.v = state.theta;
            scalar_state.w = state.theta;
            scalar_state.rho = state.rho;
            scalar_state.c_grid = false;
            scalar_state.NR = nr;
            scalar_state.NTH = nth;
            scalar_state.NZ = nz;
            const auto add_pass = [&](const Field3D& coefficient,
                                      const char* mode)
            {
                TurbulenceConfig operator_config;
                operator_config.mode = mode;
                Field3D pass(nr, nth, nz, 0.0f);
                collocated_eddy_viscosity::scalar_tendency(
                    operator_config, scalar_state, *state.grid,
                    coefficient, *state.theta, pass,
                    state.periodic_x, state.periodic_y);
                for (std::size_t index = 0; index < pass.size(); ++index)
                    candidate.dthetadt_diff.data()[index] +=
                        pass.data()[index];
            };
            if (cfg.K_h > 0.0)
            {
                Field3D horizontal_k(
                    nr, nth, nz, static_cast<float>(cfg.K_h));
                add_pass(horizontal_k, "horizontal_only");
            }
            if (cfg.K_v > 0.0 || scalar_k != nullptr)
            {
                Field3D vertical_k(
                    nr, nth, nz, static_cast<float>(scalar_k_default));
                if (scalar_k != nullptr) vertical_k = *scalar_k;
                add_pass(vertical_k, "vertical_only");
            }
        }
        else
        {
            compute_scalar_diffusion(*state.theta, *state.rho, scalar_k,
                                     scalar_k_default, *state.grid, state.c_grid,
                                     candidate.dthetadt_diff);
            if (cfg.K_h > 0.0)
                add_horizontal_scalar_diffusion(
                    *state.theta, *state.rho, cfg.K_h, *state.grid,
                    state.c_grid,
                    state.periodic_x, state.periodic_y,
                    candidate.dthetadt_diff);
        }
    }

    if (!device_computed && state.qv && matches_shape(state.qv, nr, nth, nz))
    {
        if (terrain_active && state.c_grid)
            terrain_transport::cgrid_scalar_diffusion_tendency(
                *state.grid->terrain_metrics, *state.qv, *state.rho,
                scalar_k, cfg.K_h, scalar_k_default,
                state.periodic_x, state.periodic_y,
                candidate.dqvdt_diff);
        else if (terrain_active)
        {
            TurbulenceStateView scalar_state;
            static_cast<FlowExecutionInputs&>(scalar_state) = state;
            scalar_state.u = state.qv;
            scalar_state.v = state.qv;
            scalar_state.w = state.qv;
            scalar_state.rho = state.rho;
            scalar_state.c_grid = false;
            scalar_state.NR = nr;
            scalar_state.NTH = nth;
            scalar_state.NZ = nz;
            const auto add_pass = [&](const Field3D& coefficient,
                                      const char* mode)
            {
                TurbulenceConfig operator_config;
                operator_config.mode = mode;
                Field3D pass(nr, nth, nz, 0.0f);
                collocated_eddy_viscosity::scalar_tendency(
                    operator_config, scalar_state, *state.grid,
                    coefficient, *state.qv, pass,
                    state.periodic_x, state.periodic_y);
                for (std::size_t index = 0; index < pass.size(); ++index)
                    candidate.dqvdt_diff.data()[index] +=
                        pass.data()[index];
            };
            if (cfg.K_h > 0.0)
            {
                Field3D horizontal_k(
                    nr, nth, nz, static_cast<float>(cfg.K_h));
                add_pass(horizontal_k, "horizontal_only");
            }
            if (cfg.K_v > 0.0 || scalar_k != nullptr)
            {
                Field3D vertical_k(
                    nr, nth, nz, static_cast<float>(scalar_k_default));
                if (scalar_k != nullptr) vertical_k = *scalar_k;
                add_pass(vertical_k, "vertical_only");
            }
        }
        else
        {
            compute_scalar_diffusion(*state.qv, *state.rho, scalar_k,
                                     scalar_k_default, *state.grid, state.c_grid,
                                     candidate.dqvdt_diff);
            if (cfg.K_h > 0.0)
                add_horizontal_scalar_diffusion(
                    *state.qv, *state.rho, cfg.K_h, *state.grid,
                    state.c_grid,
                    state.periodic_x, state.periodic_y,
                    candidate.dqvdt_diff);
        }
    }

    if (diag_opt)
    {
        if (diag_opt->K_effective.size_r() != nr ||
            diag_opt->K_effective.size_th() != nth ||
            diag_opt->K_effective.size_z() != nz)
        {
            diag_opt->K_effective.resize(nr, nth, nz, 0.0f);
        }

        const Field3D* diagnostic_k = scalar_k ? scalar_k : momentum_k;
        if (diagnostic_k != nullptr)
        {
            diag_opt->K_effective.fill(0.0f);
            const HorizontalDomain domain = scalar_horizontal_domain(
                nr, nth, nz, state.c_grid);
            for (int i = domain.i_begin; i < domain.i_end; ++i)
            {
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                {
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        diag_opt->K_effective[i][j][k] = static_cast<float>(
                            std::max(cfg.K_h,
                                     static_cast<double>((*diagnostic_k)[i][j][k])));
                    }
                }
            }
        }
        else
        {
            // This legacy single-field diagnostic means the maximum active
            // directional coefficient, not an isotropic coefficient. That
            // keeps horizontal-only runs from falsely publishing K=0 while
            // preserving K_v for the common K_v >= K_h case.
            diag_opt->K_effective.fill(static_cast<float>(
                std::max(cfg.K_h, scalar_k_default)));
        }
        diag_opt->max_diffusion_number = check_stability(cfg, state);
    }

    tendencies = std::move(candidate);
}

double ExplicitDiffusionScheme::check_stability(
    const DiffusionConfig& cfg,
    const DiffusionStateView& state
)
{
    if (!state.grid) return 0.0;
    if (!std::isfinite(cfg.K_h) || cfg.K_h < 0.0 ||
        !std::isfinite(cfg.K_v) || cfg.K_v < 0.0 ||
        !std::isfinite(cfg.dt_diffusion) || cfg.dt_diffusion <= 0.0)
        throw std::invalid_argument(
            "explicit diffusion stability check received invalid configuration");
    if (!std::isfinite(state.grid->dx) || state.grid->dx <= 0.0 ||
        !std::isfinite(state.grid->dy) || state.grid->dy <= 0.0 ||
        state.grid->dz.empty() || !std::isfinite(state.grid->dz[0]) ||
        state.grid->dz[0] <= 0.0)
        throw std::invalid_argument(
            "explicit diffusion stability check requires positive finite grid metrics");
    for (double layer_dz : state.grid->dz)
        if (!std::isfinite(layer_dz) || layer_dz <= 0.0)
            throw std::invalid_argument(
                "explicit diffusion stability check requires positive finite layer thicknesses");
    const auto* ref_field = state.theta ? state.theta
        : (state.u ? state.u : state.qv);
    if (ref_field == nullptr || !matches_shape(
            state.rho, ref_field->size_r(), ref_field->size_th(),
            ref_field->size_z()))
        throw std::invalid_argument(
            "active explicit diffusion stability requires matching state and live density");
    const int nr = ref_field->size_r();
    const int nth = ref_field->size_th();
    const int nz = ref_field->size_z();
    if (nr < 3 || nth < 3)
        throw std::invalid_argument(
            "explicit diffusion requires at least 3x3 lateral storage");
    if (state.grid->terrain_metrics_active)
    {
        if (state.grid->terrain_metrics == nullptr)
            throw std::invalid_argument(
                "terrain diffusion stability requires bound conservative geometry");
        if (!state.c_grid)
        {
            const TerrainMetrics3D& terrain = *state.grid->terrain_metrics;
            terrain_transport::validate_geometry(terrain);
            double minimum_density = std::numeric_limits<double>::infinity();
            double maximum_density = 0.0;
            double maximum_coefficient = std::max(cfg.K_h, cfg.K_v);
            double minimum_vertical_distance =
                std::numeric_limits<double>::infinity();
            double maximum_metric_factor = 1.0;
            for (int i = 1; i < nr - 1; ++i)
                for (int j = 1; j < nth - 1; ++j)
                    for (int k = 1; k < nz - 1; ++k)
                    {
                        const double density = (*state.rho)(i, j, k);
                        if (!std::isfinite(density) || density <= 0.0)
                            throw std::invalid_argument(
                                "mapped collocated diffusion stability requires positive finite density");
                        minimum_density = std::min(minimum_density, density);
                        maximum_density = std::max(maximum_density, density);
                        if (cfg.use_variable_K)
                        {
                            for (const Field3D* coefficient :
                                 {state.K_scalar, state.K_momentum})
                            {
                                if (coefficient == nullptr) continue;
                                const double value = (*coefficient)(i, j, k);
                                if (!std::isfinite(value) || value < 0.0)
                                    throw std::invalid_argument(
                                        "mapped collocated diffusion stability requires non-negative finite coefficients");
                                maximum_coefficient = std::max(
                                    maximum_coefficient, value);
                            }
                        }
                        const double lower = terrain.z(i, j, k)
                            - terrain.z(i, j, k - 1);
                        const double upper = terrain.z(i, j, k + 1)
                            - terrain.z(i, j, k);
                        minimum_vertical_distance = std::min(
                            {minimum_vertical_distance, lower, upper});
                        const double gx = terrain.J(i, j, k)
                            * terrain.mx(i, j, k);
                        const double gy = terrain.J(i, j, k)
                            * terrain.my(i, j, k);
                        maximum_metric_factor = std::max(
                            maximum_metric_factor,
                            1.0 + gx * gx + gy * gy);
                    }
            if (!std::isfinite(minimum_vertical_distance)
                || minimum_vertical_distance <= 0.0)
                throw std::invalid_argument(
                    "mapped collocated diffusion stability requires increasing physical centers");
            const double density_ratio = maximum_density / minimum_density;
            const double scalar_bound = 2.0 * cfg.dt_diffusion
                * maximum_coefficient * density_ratio
                * maximum_metric_factor
                * (1.0 / (terrain.dx * terrain.dx)
                 + 1.0 / (terrain.dy * terrain.dy)
                 + 1.0 / (minimum_vertical_distance
                          * minimum_vertical_distance));
            const double bound = (state.u || state.v || state.w)
                ? 2.0 * scalar_bound : scalar_bound;
            if (!std::isfinite(bound) || bound < 0.0)
                throw std::overflow_error(
                    "mapped collocated diffusion stability bound overflowed");
            return bound;
        }
        double maximum = 0.0;
        if (state.theta || state.qv)
        {
            const Field3D* vertical_k = cfg.use_variable_K
                ? state.K_scalar : nullptr;
            maximum = terrain_transport::cgrid_scalar_diffusion_number(
                *state.grid->terrain_metrics, *state.rho, vertical_k,
                cfg.K_h, cfg.K_v, cfg.dt_diffusion,
                state.periodic_x, state.periodic_y);
        }
        if (state.u || state.v || state.w)
        {
            if (!(state.u && state.v && state.w))
                throw std::invalid_argument(
                    "terrain momentum diffusion stability requires u, v, and w together");
            const Field3D* vertical_k = cfg.use_variable_K
                ? state.K_momentum : nullptr;
            maximum = std::max(maximum, 2.0 *
                terrain_transport::cgrid_scalar_diffusion_number(
                    *state.grid->terrain_metrics, *state.rho, vertical_k,
                    cfg.K_h, cfg.K_v, cfg.dt_diffusion,
                    state.periodic_x, state.periodic_y));
        }
        return maximum;
    }

    const auto scalar_density = [&](int i, int j, int k)
    {
        return static_cast<double>((*state.rho)(i, j, k));
    };
    const auto scalar_k = [&](int i, int j, int k, bool momentum)
    {
        const Field3D* field = momentum ? state.K_momentum : state.K_scalar;
        if (cfg.use_variable_K)
        {
            if (!matches_shape(field, nr, nth, nz))
                throw std::invalid_argument(
                    "explicit variable diffusion stability requires a matching coefficient field");
            return static_cast<double>((*field)(i, j, k));
        }
        return cfg.K_v;
    };
    const auto scalar_transport = [&](int i, int j, int k)
    {
        return scalar_density(i, j, k) * scalar_k(i, j, k, false);
    };
    const auto u_density = [&](int i, int j, int k)
    {
        const int right = state.c_grid && state.periodic_x && i == nr - 2
            ? 1 : i + 1;
        return state.c_grid
            ? 0.5 * (scalar_density(i, j, k) +
                     scalar_density(right, j, k))
            : scalar_density(i, j, k);
    };
    const auto u_transport = [&](int i, int j, int k)
    {
        const int right = state.c_grid && state.periodic_x && i == nr - 2
            ? 1 : i + 1;
        return state.c_grid
            ? 0.5 * (scalar_density(i, j, k) * scalar_k(i, j, k, true) +
                     scalar_density(right, j, k) * scalar_k(right, j, k, true))
            : scalar_density(i, j, k) * scalar_k(i, j, k, true);
    };
    const auto v_density = [&](int i, int j, int k)
    {
        const int north = state.c_grid && state.periodic_y && j == nth - 2
            ? 1 : j + 1;
        return state.c_grid
            ? 0.5 * (scalar_density(i, j, k) +
                     scalar_density(i, north, k))
            : scalar_density(i, j, k);
    };
    const auto v_transport = [&](int i, int j, int k)
    {
        const int north = state.c_grid && state.periodic_y && j == nth - 2
            ? 1 : j + 1;
        return state.c_grid
            ? 0.5 * (scalar_density(i, j, k) * scalar_k(i, j, k, true) +
                     scalar_density(i, north, k) * scalar_k(i, north, k, true))
            : scalar_density(i, j, k) * scalar_k(i, j, k, true);
    };
    const auto w_density = [&](int i, int j, int k)
    {
        if (!state.c_grid) return scalar_density(i, j, k);
        const double lower_dz =
            state.grid->dz[static_cast<std::size_t>(k)];
        const double upper_dz =
            state.grid->dz[static_cast<std::size_t>(k + 1)];
        return (scalar_density(i, j, k) * lower_dz +
                scalar_density(i, j, k + 1) * upper_dz) /
            (lower_dz + upper_dz);
    };
    const auto w_transport = [&](int i, int j, int k)
    {
        return state.c_grid
            ? 0.5 * (scalar_density(i, j, k) * scalar_k(i, j, k, true) +
                     scalar_density(i, j, k + 1) * scalar_k(i, j, k + 1, true))
            : scalar_density(i, j, k) * scalar_k(i, j, k, true);
    };
    const auto scalar_mass = [&](int i, int j, int k)
    {
        return scalar_density(i, j, k) *
            state.grid->dz[static_cast<std::size_t>(k)];
    };
    const auto u_mass = [&](int i, int j, int k)
    {
        return u_density(i, j, k) *
            state.grid->dz[static_cast<std::size_t>(k)];
    };
    const auto v_mass = [&](int i, int j, int k)
    {
        return v_density(i, j, k) *
            state.grid->dz[static_cast<std::size_t>(k)];
    };
    const auto w_mass = [&](int i, int j, int k)
    {
        if (!state.c_grid)
            return w_density(i, j, k) *
                state.grid->dz[static_cast<std::size_t>(k)];
        return 0.5 * (
            scalar_density(i, j, k) *
                state.grid->dz[static_cast<std::size_t>(k)] +
            scalar_density(i, j, k + 1) *
                state.grid->dz[static_cast<std::size_t>(k + 1)]);
    };
    const auto scalar_face_transport = [&](int i, int j, int lower_k)
    {
        return 0.5 * (scalar_transport(i, j, lower_k) +
                      scalar_transport(i, j, lower_k + 1));
    };
    const auto u_face_transport = [&](int i, int j, int lower_k)
    {
        return 0.5 * (u_transport(i, j, lower_k) +
                      u_transport(i, j, lower_k + 1));
    };
    const auto v_face_transport = [&](int i, int j, int lower_k)
    {
        return 0.5 * (v_transport(i, j, lower_k) +
                      v_transport(i, j, lower_k + 1));
    };
    const auto w_face_transport = [&](int i, int j, int lower_k)
    {
        return state.c_grid
            ? scalar_density(i, j, lower_k + 1) *
                scalar_k(i, j, lower_k + 1, true)
            : 0.5 * (w_transport(i, j, lower_k) +
                     w_transport(i, j, lower_k + 1));
    };

    double vertical_max = 0.0;
    if (cfg.K_v > 0.0 || cfg.use_variable_K)
    {
        if (state.theta || state.qv)
            vertical_max = std::max(vertical_max,
                vertical_component_diffusion_number(
                    cfg.dt_diffusion, *state.grid,
                    scalar_horizontal_domain(nr, nth, nz, state.c_grid),
                    false, scalar_mass, scalar_transport,
                    scalar_face_transport));
        if (state.u && state.v && state.w)
        {
            vertical_max = std::max(vertical_max,
                vertical_component_diffusion_number(
                    cfg.dt_diffusion, *state.grid,
                    u_horizontal_domain(nr, nth, nz, state.c_grid,
                                        state.periodic_x),
                    false, u_mass, u_transport, u_face_transport));
            vertical_max = std::max(vertical_max,
                vertical_component_diffusion_number(
                    cfg.dt_diffusion, *state.grid,
                    v_horizontal_domain(nr, nth, nz, state.c_grid,
                                        state.periodic_y),
                    false, v_mass, v_transport, v_face_transport));
            vertical_max = std::max(vertical_max,
                vertical_component_diffusion_number(
                    cfg.dt_diffusion, *state.grid,
                    w_horizontal_domain(nr, nth, nz, state.c_grid),
                    state.c_grid, w_mass, w_transport,
                    w_face_transport));
        }
    }

    double horizontal_max = 0.0;
    if (cfg.K_h > 0.0)
    {
        if (state.theta || state.qv)
            horizontal_max = std::max(horizontal_max,
                horizontal_component_diffusion_number(
                    cfg.K_h, cfg.dt_diffusion, *state.grid,
                    scalar_horizontal_domain(nr, nth, nz, state.c_grid),
                    state.periodic_x, state.periodic_y, scalar_density));
        if (state.u && state.v && state.w)
        {
            horizontal_max = std::max(horizontal_max,
                horizontal_component_diffusion_number(
                    cfg.K_h, cfg.dt_diffusion, *state.grid,
                    u_horizontal_domain(nr, nth, nz, state.c_grid,
                                        state.periodic_x),
                    state.periodic_x, state.periodic_y, u_density));
            horizontal_max = std::max(horizontal_max,
                horizontal_component_diffusion_number(
                    cfg.K_h, cfg.dt_diffusion, *state.grid,
                    v_horizontal_domain(nr, nth, nz, state.c_grid,
                                        state.periodic_y),
                    state.periodic_x, state.periodic_y, v_density));
            horizontal_max = std::max(horizontal_max,
                horizontal_component_diffusion_number(
                    cfg.K_h, cfg.dt_diffusion, *state.grid,
                    w_horizontal_domain(nr, nth, nz, state.c_grid),
                    state.periodic_x, state.periodic_y, w_density));
        }
    }
    // The sum of independent directional maxima is a conservative sufficient
    // bound. It may be slightly stricter than the exact per-cell combined max.
    return vertical_max + horizontal_max;
}

void ExplicitDiffusionScheme::add_horizontal_scalar_diffusion(
    const Field3D& field,
    const Field3D& rho,
    double K_h,
    const GridMetrics& grid,
    bool c_grid,
    bool periodic_x,
    bool periodic_y,
    Field3D& tendency)
{
    if (K_h <= 0.0) return;
    const int nr = field.size_r();
    const int nth = field.size_th();
    const int nz = field.size_z();
    if (nr < 3 || nth < 3 || !std::isfinite(grid.dx) || grid.dx <= 0.0
        || !std::isfinite(grid.dy) || grid.dy <= 0.0)
        throw std::invalid_argument(
            "horizontal explicit diffusion requires positive Cartesian spacing and lateral closure storage");
    const auto density = [&](int i, int j, int k)
    {
        return static_cast<double>(rho(i, j, k));
    };
    add_density_weighted_horizontal_component(
        field, K_h, grid, scalar_horizontal_domain(nr, nth, nz, c_grid),
        periodic_x, periodic_y, density, "scalar", tendency);
}

void ExplicitDiffusionScheme::add_horizontal_momentum_diffusion(
    const Field3D& u,
    const Field3D& v,
    const Field3D& w,
    const Field3D& rho,
    double K_h,
    const GridMetrics& grid,
    bool c_grid,
    bool periodic_x,
    bool periodic_y,
    Field3D& du_dt,
    Field3D& dv_dt,
    Field3D& dw_dt)
{
    if (K_h <= 0.0) return;
    const int nr = u.size_r();
    const int nth = u.size_th();
    const int nz = u.size_z();
    if (nr < 4 || nth < 4 || nz < 3 ||
        !std::isfinite(grid.dx) || grid.dx <= 0.0 ||
        !std::isfinite(grid.dy) || grid.dy <= 0.0)
        throw std::invalid_argument(
            "horizontal explicit momentum diffusion requires positive "
            "Cartesian spacing and sufficient control-volume storage");

    const auto scalar_density = [&](int i, int j, int k)
    {
        return static_cast<double>(rho(i, j, k));
    };
    const auto u_density = [&](int i, int j, int k)
    {
        const int right = c_grid && periodic_x && i == nr - 2 ? 1 : i + 1;
        return c_grid
            ? 0.5 * (static_cast<double>(rho(i, j, k)) +
                     static_cast<double>(rho(right, j, k)))
            : scalar_density(i, j, k);
    };
    const auto v_density = [&](int i, int j, int k)
    {
        const int north = c_grid && periodic_y && j == nth - 2 ? 1 : j + 1;
        return c_grid
            ? 0.5 * (static_cast<double>(rho(i, j, k)) +
                     static_cast<double>(rho(i, north, k)))
            : scalar_density(i, j, k);
    };
    const auto w_density = [&](int i, int j, int k)
    {
        if (!c_grid) return scalar_density(i, j, k);
        const double lower_dz = grid.dz[static_cast<std::size_t>(k)];
        const double upper_dz = grid.dz[static_cast<std::size_t>(k + 1)];
        return (static_cast<double>(rho(i, j, k)) * lower_dz +
                static_cast<double>(rho(i, j, k + 1)) * upper_dz) /
            (lower_dz + upper_dz);
    };

    add_density_weighted_horizontal_component(
        u, K_h, grid,
        u_horizontal_domain(nr, nth, nz, c_grid, periodic_x),
        periodic_x, periodic_y, u_density, "u-momentum", du_dt);
    add_density_weighted_horizontal_component(
        v, K_h, grid,
        v_horizontal_domain(nr, nth, nz, c_grid, periodic_y),
        periodic_x, periodic_y, v_density, "v-momentum", dv_dt);
    add_density_weighted_horizontal_component(
        w, K_h, grid, w_horizontal_domain(nr, nth, nz, c_grid),
        periodic_x, periodic_y, w_density, "w-momentum", dw_dt);
}

void ExplicitDiffusionScheme::compute_scalar_diffusion(
    const Field3D& field,
    const Field3D& rho,
    const Field3D* K_field,
    double K_default,
    const GridMetrics& grid,
    bool c_grid,
    Field3D& tendency
)
{
    const int nr = field.size_r();
    const int nth = field.size_th();
    const int nz = field.size_z();
    if (K_default <= 0.0 && K_field == nullptr)
        return;
    const auto density = [&](int i, int j, int k)
    {
        return static_cast<double>(rho(i, j, k));
    };
    const auto transport = [&](int i, int j, int k)
    {
        const double diffusivity = K_field
            ? static_cast<double>((*K_field)(i, j, k)) : K_default;
        return density(i, j, k) * diffusivity;
    };
    const auto control_mass = [&](int i, int j, int k)
    {
        return density(i, j, k) * grid.dz[static_cast<std::size_t>(k)];
    };
    const auto face_transport = [&](int i, int j, int lower_k)
    {
        return 0.5 * (transport(i, j, lower_k) +
                      transport(i, j, lower_k + 1));
    };
    add_density_weighted_vertical_component(
        field, grid, scalar_horizontal_domain(nr, nth, nz, c_grid),
        false, control_mass, transport, face_transport, "scalar", tendency);
}

void ExplicitDiffusionScheme::compute_momentum_diffusion(
    const Field3D& u,
    const Field3D& v,
    const Field3D& w,
    const Field3D& rho,
    const Field3D* nu_t,
    double nu_default,
    const GridMetrics& grid,
    bool c_grid,
    bool periodic_x,
    bool periodic_y,
    Field3D& du_dt,
    Field3D& dv_dt,
    Field3D& dw_dt
)
{
    const int nr = u.size_r();
    const int nth = u.size_th();
    const int nz = u.size_z();
    if (nu_default <= 0.0 && nu_t == nullptr)
        return;
    const auto scalar_density = [&](int i, int j, int k)
    {
        return static_cast<double>(rho(i, j, k));
    };
    const auto scalar_transport = [&](int i, int j, int k)
    {
        const double diffusivity = nu_t
            ? static_cast<double>((*nu_t)(i, j, k)) : nu_default;
        return scalar_density(i, j, k) * diffusivity;
    };
    const auto u_density = [&](int i, int j, int k)
    {
        const int right = c_grid && periodic_x && i == nr - 2 ? 1 : i + 1;
        return c_grid ? 0.5 * (
            scalar_density(i, j, k) + scalar_density(right, j, k))
            : scalar_density(i, j, k);
    };
    const auto u_transport = [&](int i, int j, int k)
    {
        const int right = c_grid && periodic_x && i == nr - 2 ? 1 : i + 1;
        return c_grid ? 0.5 * (
            scalar_transport(i, j, k) + scalar_transport(right, j, k))
            : scalar_transport(i, j, k);
    };
    const auto v_density = [&](int i, int j, int k)
    {
        const int north = c_grid && periodic_y && j == nth - 2 ? 1 : j + 1;
        return c_grid ? 0.5 * (
            scalar_density(i, j, k) + scalar_density(i, north, k))
            : scalar_density(i, j, k);
    };
    const auto v_transport = [&](int i, int j, int k)
    {
        const int north = c_grid && periodic_y && j == nth - 2 ? 1 : j + 1;
        return c_grid ? 0.5 * (
            scalar_transport(i, j, k) + scalar_transport(i, north, k))
            : scalar_transport(i, j, k);
    };
    const auto w_density = [&](int i, int j, int k)
    {
        return c_grid ? 0.5 * (
            scalar_density(i, j, k) + scalar_density(i, j, k + 1))
            : scalar_density(i, j, k);
    };
    const auto w_transport = [&](int i, int j, int k)
    {
        return c_grid ? 0.5 * (
            scalar_transport(i, j, k) + scalar_transport(i, j, k + 1))
            : scalar_transport(i, j, k);
    };
    const auto u_mass = [&](int i, int j, int k)
    {
        return u_density(i, j, k) * grid.dz[static_cast<std::size_t>(k)];
    };
    const auto v_mass = [&](int i, int j, int k)
    {
        return v_density(i, j, k) * grid.dz[static_cast<std::size_t>(k)];
    };
    const auto w_mass = [&](int i, int j, int k)
    {
        if (!c_grid)
            return w_density(i, j, k) *
                grid.dz[static_cast<std::size_t>(k)];
        return 0.5 * (
            scalar_density(i, j, k) * grid.dz[static_cast<std::size_t>(k)] +
            scalar_density(i, j, k + 1) *
                grid.dz[static_cast<std::size_t>(k + 1)]);
    };
    const auto u_face_transport = [&](int i, int j, int lower_k)
    {
        return 0.5 * (u_transport(i, j, lower_k) +
                      u_transport(i, j, lower_k + 1));
    };
    const auto v_face_transport = [&](int i, int j, int lower_k)
    {
        return 0.5 * (v_transport(i, j, lower_k) +
                      v_transport(i, j, lower_k + 1));
    };
    const auto w_face_transport = [&](int i, int j, int lower_k)
    {
        return c_grid
            ? scalar_transport(i, j, lower_k + 1)
            : 0.5 * (w_transport(i, j, lower_k) +
                     w_transport(i, j, lower_k + 1));
    };

    add_density_weighted_vertical_component(
        u, grid, u_horizontal_domain(nr, nth, nz, c_grid, periodic_x),
        false, u_mass, u_transport, u_face_transport, "u-momentum", du_dt);
    add_density_weighted_vertical_component(
        v, grid, v_horizontal_domain(nr, nth, nz, c_grid, periodic_y),
        false, v_mass, v_transport, v_face_transport, "v-momentum", dv_dt);
    add_density_weighted_vertical_component(
        w, grid, w_horizontal_domain(nr, nth, nz, c_grid), c_grid,
        w_mass, w_transport, w_face_transport, "w-momentum", dw_dt);
}

} // namespace tmv
