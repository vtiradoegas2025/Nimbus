// Correlated stochastic multipliers across configured physics blocks.

#include "full_stochastic.hpp"
#include "chaos/schemes/initial_conditions/initial_conditions.hpp"
#include "chaos/base/perturbation_field.hpp"
#include "chaos/base/correlation_filter.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "turbulence/turbulence_base.hpp"
#include "util/log.hpp"

#include <chrono>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace {

bool portable_fields_share_shape(
    const std::vector<Field3D>& fields,
    std::size_t count)
{
    if (count == 0) return true;
    if (fields.size() < count) return false;
    const Field3D& reference = fields.front();
    for (std::size_t index = 0; index < count; ++index)
    {
        const Field3D& field = fields[index];
        if (!(field.empty() == reference.empty() &&
            (field.empty() || (field.size_r() == reference.size_r() &&
                               field.size_th() == reference.size_th() &&
                               field.size_z() == reference.size_z()))))
            return false;
    }
    return true;
}

bool same_shape(const Field3D& lhs, const Field3D& rhs)
{
    return lhs.empty() == rhs.empty()
        && (lhs.empty() || (lhs.size_r() == rhs.size_r()
            && lhs.size_th() == rhs.size_th()
            && lhs.size_z() == rhs.size_z()));
}

std::uint64_t encode_signed(std::int64_t value) noexcept
{
    std::uint64_t encoded = 0;
    std::memcpy(&encoded, &value, sizeof(encoded));
    return encoded;
}

std::int64_t decode_signed(std::uint64_t value) noexcept
{
    std::int64_t decoded = 0;
    std::memcpy(&decoded, &value, sizeof(decoded));
    return decoded;
}

bool bitwise_equal(const Field3D& lhs, const Field3D& rhs)
{
    return lhs.empty() == rhs.empty()
        && (lhs.empty() || (lhs.size_r() == rhs.size_r()
            && lhs.size_th() == rhs.size_th()
            && lhs.size_z() == rhs.size_z()
            && std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(float)) == 0));
}

/**
 * @brief Returns the first valid tendency field to infer active domain size.
 */
Field3D* first_available_tendency(chaos::ChaosTendencies& t)
{
    if (t.dtheta_micro_dt != nullptr) return t.dtheta_micro_dt;
    if (t.dqv_micro_dt != nullptr) return t.dqv_micro_dt;
    if (t.du_pbl_dt != nullptr) return t.du_pbl_dt;
    if (t.dtheta_pbl_dt != nullptr) return t.dtheta_pbl_dt;
    if (t.pbl_u_force_n != nullptr) return t.pbl_u_force_n;
    if (t.pbl_v_force_n != nullptr) return t.pbl_v_force_n;
    if (t.pbl_theta_inventory_rate_kgk_s != nullptr)
        return t.pbl_theta_inventory_rate_kgk_s;
    if (t.pbl_vapor_mass_rate_kg_s != nullptr)
        return t.pbl_vapor_mass_rate_kg_s;
    if (t.du_diff_dt != nullptr) return t.du_diff_dt;
    if (t.dtheta_diff_dt != nullptr) return t.dtheta_diff_dt;
    if (t.eddy_viscosity != nullptr) return t.eddy_viscosity;
    if (t.theta_diffusivity != nullptr) return t.theta_diffusivity;
    return nullptr;
}

/**
 * @brief Checks whether a noise field matches requested domain dimensions.
 */
bool noise_matches_domain(const Field3D& noise, int nr, int nth, int nz)
{
    return !noise.empty() &&
           noise.size_r() == nr &&
           noise.size_th() == nth &&
           noise.size_z() == nz;
}

/**
 * @brief Applies multiplicative stochastic scaling to one tendency block.
 */
void apply_block_multiplier(
    Field3D* tendency,
    const Field3D& xi,
    const PhysicalCellDomain& domain,
    double alpha,
    double& min_multiplier,
    double& max_multiplier,
    double& sum_multiplier,
    double& sum_sq_multiplier,
    size_t& sample_count
)
{
    if (tendency == nullptr || xi.empty())
    {
        return;
    }

    if (tendency->size_r() != xi.size_r() ||
        tendency->size_th() != xi.size_th() ||
        tendency->size_z() != xi.size_z())
    {
        return;
    }

    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                double multiplier = 1.0 + alpha *
                    static_cast<double>(xi(i, j, k));
                multiplier = std::clamp(multiplier, 0.0, 2.0);
                (*tendency)(i, j, k) = static_cast<float>(
                    static_cast<double>((*tendency)(i, j, k)) * multiplier);
                min_multiplier = std::min(min_multiplier, multiplier);
                max_multiplier = std::max(max_multiplier, multiplier);
                sum_multiplier += multiplier;
                sum_sq_multiplier += multiplier * multiplier;
                ++sample_count;
            }
}

void prepare_microphysics_column_multipliers(
    std::vector<float>& multipliers,
    const Field3D& xi,
    const PhysicalCellDomain& domain,
    double alpha,
    double& min_multiplier,
    double& max_multiplier,
    double& sum_multiplier,
    double& sum_sq_multiplier,
    size_t& sample_count)
{
    if (xi.empty())
    {
        multipliers.clear();
        return;
    }
    const std::size_t surface_size =
        static_cast<std::size_t>(xi.size_r())
        * static_cast<std::size_t>(xi.size_th());
    multipliers.assign(surface_size, 1.0f);
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            const double multiplier = std::clamp(
                1.0 + alpha * static_cast<double>(
                    xi[i][j][domain.k_begin]),
                0.0, 2.0);
            multipliers[static_cast<std::size_t>(i)
                            * static_cast<std::size_t>(xi.size_th())
                        + static_cast<std::size_t>(j)] =
                static_cast<float>(multiplier);
            min_multiplier = std::min(min_multiplier, multiplier);
            max_multiplier = std::max(max_multiplier, multiplier);
            sum_multiplier += multiplier;
            sum_sq_multiplier += multiplier * multiplier;
            ++sample_count;
        }
}

void require_pbl_source_shape(const Field3D* field,
                              const Field3D& xi,
                              const char* label)
{
    if (field != nullptr &&
        (field->size_r() != xi.size_r() ||
         field->size_th() != xi.size_th() ||
         field->size_z() != xi.size_z()))
    {
        throw std::invalid_argument(
            std::string("stochastic PBL ") + label +
            " shape does not match its noise field");
    }
}

/**
 * @brief Scales one complete PBL column and its boundary flux by one factor.
 *
 * Applying a height-varying factor to an already formed flux divergence is
 * not conservative.  The production representation therefore treats the
 * surface value of the correlated field as a column process-rate parameter
 * and applies it to the held force/inventory sources, the MYNN TKE source,
 * and the corresponding lower-boundary diagnostics together.
 */
void apply_conservative_pbl_column_multiplier(
    chaos::ChaosTendencies& tendencies,
    const Field3D& xi,
    const PhysicalCellDomain& domain,
    double alpha,
    double& min_multiplier,
    double& max_multiplier,
    double& sum_multiplier,
    double& sum_sq_multiplier,
    size_t& sample_count)
{
    Field3D* volume_fields[] = {
        tendencies.pbl_u_force_n,
        tendencies.pbl_v_force_n,
        tendencies.pbl_theta_inventory_rate_kgk_s,
        tendencies.pbl_vapor_mass_rate_kg_s,
        tendencies.pbl_tke_tendency};
    const char* labels[] = {
        "u-force", "v-force", "theta-inventory", "vapor-mass", "TKE"};
    for (std::size_t field = 0; field < 5; ++field)
        require_pbl_source_shape(volume_fields[field], xi, labels[field]);

    std::vector<float>* linear_surface_fields[] = {
        tendencies.pbl_tau_u,
        tendencies.pbl_tau_v,
        tendencies.pbl_sensible_heat_flux,
        tendencies.pbl_latent_heat_flux,
        tendencies.pbl_moisture_flux};
    const std::size_t surface_size =
        static_cast<std::size_t>(xi.size_r()) *
        static_cast<std::size_t>(xi.size_th());
    for (std::vector<float>* field : linear_surface_fields)
        if (field != nullptr && field->size() != surface_size)
            throw std::invalid_argument(
                "stochastic PBL surface diagnostic shape does not match domain");
    if (tendencies.pbl_ustar != nullptr &&
        tendencies.pbl_ustar->size() != surface_size)
        throw std::invalid_argument(
            "stochastic PBL friction-velocity shape does not match domain");

    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            const double multiplier = std::clamp(
                1.0 + alpha * static_cast<double>(
                    xi[i][j][domain.k_begin]), 0.0, 2.0);
            const float multiplier_f = static_cast<float>(multiplier);
            for (Field3D* field : volume_fields)
            {
                if (field == nullptr) continue;
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                    (*field)[i][j][k] *= multiplier_f;
            }
            const std::size_t surface_index =
                static_cast<std::size_t>(i) *
                static_cast<std::size_t>(xi.size_th()) +
                static_cast<std::size_t>(j);
            for (std::vector<float>* field : linear_surface_fields)
                if (field != nullptr)
                    (*field)[surface_index] *= multiplier_f;
            if (tendencies.pbl_ustar != nullptr)
                (*tendencies.pbl_ustar)[surface_index] *=
                    static_cast<float>(std::sqrt(multiplier));

            min_multiplier = std::min(min_multiplier, multiplier);
            max_multiplier = std::max(max_multiplier, multiplier);
            sum_multiplier += multiplier;
            sum_sq_multiplier += multiplier * multiplier;
            ++sample_count;
        }
}

}

namespace tmv::chaos
{

void FullStochasticScheme::configure_compute_backend(
    ComputeBackend* backend, bool required) noexcept
{
    native_backend_ = backend;
    native_generation_required_ = required;
    microphysics_noise_.configure_native_evolution(backend, required);
    pbl_noise_.configure_native_evolution(backend, required);
    diffusion_noise_.configure_native_evolution(backend, required);
}

PortableSchemeState FullStochasticScheme::portable_state() const
{
    PortableSchemeState state;
    // v3 records the Earth/control origin of every independent process.
    // v2 recorded only the microphysics control minimum and implicitly reused
    // it for PBL and diffusion. Different process keys have different lattice
    // phases, so that shortcut becomes invalid after cumulative recycle moves.
    state.version = 4;
    state.fields = {
        xi_microphysics_, xi_pbl_, xi_diffusion_,
        xi_micro_prev_, xi_pbl_prev_, xi_diff_prev_,
        microphysics_noise_.current_controls(),
        microphysics_noise_.previous_controls(),
        pbl_noise_.current_controls(), pbl_noise_.previous_controls(),
        diffusion_noise_.current_controls(),
        diffusion_noise_.previous_controls()
    };
    state.integers = {
        time_step_counter_,
        encode_signed(microphysics_noise_.earth_origin_i_cells()),
        encode_signed(microphysics_noise_.earth_origin_j_cells()),
        encode_signed(microphysics_noise_.control_i_min()),
        encode_signed(microphysics_noise_.control_j_min()),
        encode_signed(pbl_noise_.earth_origin_i_cells()),
        encode_signed(pbl_noise_.earth_origin_j_cells()),
        encode_signed(pbl_noise_.control_i_min()),
        encode_signed(pbl_noise_.control_j_min()),
        encode_signed(diffusion_noise_.earth_origin_i_cells()),
        encode_signed(diffusion_noise_.earth_origin_j_cells()),
        encode_signed(diffusion_noise_.control_i_min()),
        encode_signed(diffusion_noise_.control_j_min()),
        static_cast<std::uint64_t>(microphysics_noise_.nr()),
        static_cast<std::uint64_t>(microphysics_noise_.nth()),
        static_cast<std::uint64_t>(microphysics_noise_.nz())};
    state.reals = {
        cadence_dt_s_, microphysics_noise_.continuous_offset_x_m(),
        microphysics_noise_.continuous_offset_y_m()};
    return state;
}

bool FullStochasticScheme::restore_portable_state(
    const PortableSchemeState& state, std::string& error)
{
    const bool legacy_v2 = state.version == 2 && state.integers.size() == 8;
    const bool legacy_v3 = state.version == 3 && state.integers.size() == 16 &&
        state.reals.size() == 1;
    const bool current_v4 = state.version == 4 && state.integers.size() == 16 &&
        state.reals.size() == 3;
    const bool current_layout = legacy_v3 || current_v4;
    const std::size_t dimension_offset = current_layout ? 13U : 5U;
    if ((!legacy_v2 && !current_layout) || state.fields.size() != 12 ||
        !portable_fields_share_shape(state.fields, 6) ||
        !same_shape(state.fields[6], state.fields[7]) ||
        !same_shape(state.fields[8], state.fields[9]) ||
        !same_shape(state.fields[10], state.fields[11]) ||
        !std::isfinite(state.reals[0]) || state.reals[0] < 0.0 ||
        (current_v4 &&
         (!std::isfinite(state.reals[1]) || !std::isfinite(state.reals[2]))) ||
        state.integers[dimension_offset] > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        state.integers[dimension_offset + 1] > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        state.integers[dimension_offset + 2] > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    {
        error = "invalid full-stochastic chaos portable state";
        return false;
    }
    const int nr = static_cast<int>(state.integers[dimension_offset]);
    const int nth = static_cast<int>(state.integers[dimension_offset + 1]);
    const int nz = static_cast<int>(state.integers[dimension_offset + 2]);
    const std::array<std::size_t, 3> metadata_offsets =
        current_layout ? std::array<std::size_t, 3>{1U, 5U, 9U}
                   : std::array<std::size_t, 3>{1U, 1U, 1U};
    std::array<std::int64_t, 3> origin_i{};
    std::array<std::int64_t, 3> origin_j{};
    std::array<std::int64_t, 3> control_i{};
    std::array<std::int64_t, 3> control_j{};
    for (std::size_t process = 0; process < metadata_offsets.size(); ++process)
    {
        const std::size_t offset = metadata_offsets[process];
        origin_i[process] = decode_signed(state.integers[offset]);
        origin_j[process] = decode_signed(state.integers[offset + 1]);
        control_i[process] = decode_signed(state.integers[offset + 2]);
        control_j[process] = decode_signed(state.integers[offset + 3]);
    }
    if (current_layout && (origin_i[0] != origin_i[1]
                       || origin_i[0] != origin_i[2]
                       || origin_j[0] != origin_j[1]
                       || origin_j[0] != origin_j[2]))
    {
        error = "full-stochastic Earth-process windows disagree";
        return false;
    }
    if (nr > 0)
    {
        const double offset_x = current_v4 ? state.reals[1] : 0.0;
        const double offset_y = current_v4 ? state.reals[2] : 0.0;
        std::array<EarthCorrelatedNoise, 3> restored_noises{
            microphysics_noise_, pbl_noise_, diffusion_noise_};
        const int offsets[] = {6, 8, 10};
        for (int process = 0; process < 3; ++process)
        {
            EarthCorrelatedNoise& noise =
                restored_noises[static_cast<std::size_t>(process)];
            const bool restored = legacy_v2 && process > 0
                ? noise.restore_inferred_control_bounds(
                    state.fields[offsets[process]],
                    state.fields[offsets[process] + 1], nr, nth, nz,
                    origin_i[static_cast<std::size_t>(process)],
                    origin_j[static_cast<std::size_t>(process)], error)
                : noise.restore(
                    state.fields[offsets[process]],
                    state.fields[offsets[process] + 1], nr, nth, nz,
                    origin_i[static_cast<std::size_t>(process)],
                    origin_j[static_cast<std::size_t>(process)],
                    control_i[static_cast<std::size_t>(process)],
                    control_j[static_cast<std::size_t>(process)],
                    offset_x, offset_y, error);
            if (!restored)
                return false;
        }
        const Field3D* expected_current[] = {
            &state.fields[0], &state.fields[1], &state.fields[2]};
        const Field3D* expected_previous[] = {
            &state.fields[3], &state.fields[4], &state.fields[5]};
        for (int process = 0; process < 3; ++process)
        {
            Field3D rendered_current;
            Field3D rendered_previous;
            restored_noises[static_cast<std::size_t>(process)]
                .render_current(rendered_current);
            restored_noises[static_cast<std::size_t>(process)]
                .render_previous(rendered_previous);
            if (!bitwise_equal(rendered_current, *expected_current[process])
                || !bitwise_equal(
                    rendered_previous, *expected_previous[process]))
            {
                error = "full-stochastic fields disagree with Earth controls";
                return false;
            }
        }
        microphysics_noise_ = std::move(restored_noises[0]);
        pbl_noise_ = std::move(restored_noises[1]);
        diffusion_noise_ = std::move(restored_noises[2]);
    }
    else if (std::any_of(state.fields.begin(), state.fields.end(),
                         [](const Field3D& field) { return !field.empty(); }))
    {
        error = "invalid empty full-stochastic chaos portable state";
        return false;
    }
    else
    {
        EarthCorrelatedNoise* noises[] = {
            &microphysics_noise_, &pbl_noise_, &diffusion_noise_};
        for (std::size_t process = 0; process < std::size(noises); ++process)
        {
            EarthCorrelatedNoise* noise = noises[process];
            if (!noise->restore_uninitialized_origin(
                    origin_i[process], origin_j[process], error))
                return false;
        }
    }
    xi_microphysics_ = state.fields[0];
    xi_pbl_ = state.fields[1];
    xi_diffusion_ = state.fields[2];
    xi_micro_prev_ = state.fields[3];
    xi_pbl_prev_ = state.fields[4];
    xi_diff_prev_ = state.fields[5];
    time_step_counter_ = state.integers[0];
    cadence_dt_s_ = state.reals[0];
    if ((time_step_counter_ == 0 && cadence_dt_s_ != 0.0)
        || (time_step_counter_ > 0 && cadence_dt_s_ <= 0.0)
        || (nr == 0 && time_step_counter_ != 0))
    {
        error = "full-stochastic chaos cadence history is inconsistent";
        return false;
    }
    error.clear();
    return true;
}

/**
 * @brief Initializes RNG, filters, and internal stochastic state.
 */
void FullStochasticScheme::initialize(const ChaosConfig& cfg, const GridMetrics& grid) {
    (void)grid;
    tmv::log_info("Initialized chaos scheme: full_stochastic (evolving model error)");

    (void)create_correlation_filter(cfg.filter_id, cfg.Lx, cfg.Ly);
    microphysics_noise_ = EarthCorrelatedNoise{};
    pbl_noise_ = EarthCorrelatedNoise{};
    diffusion_noise_ = EarthCorrelatedNoise{};
    microphysics_noise_.configure_native_evolution(
        native_backend_, native_generation_required_);
    pbl_noise_.configure_native_evolution(
        native_backend_, native_generation_required_);
    diffusion_noise_.configure_native_evolution(
        native_backend_, native_generation_required_);
    microphysics_noise_.configure(
        cfg.seed, cfg.member_id, 4001ULL, 4101ULL,
        cfg.Lx, cfg.Ly, grid.dx, grid.dy);
    pbl_noise_.configure(
        cfg.seed, cfg.member_id, 4002ULL, 4102ULL,
        cfg.Lx, cfg.Ly, grid.dx, grid.dy);
    diffusion_noise_.configure(
        cfg.seed, cfg.member_id, 4003ULL, 4103ULL,
        cfg.Lx, cfg.Ly, grid.dx, grid.dy);
    tmv::log_info("  Spatial noise: ", EarthCorrelatedNoise::algorithm_id(),
                  " (legacy filter selection ", cfg.filter_id, ")");

    time_step_counter_ = 0;
    xi_microphysics_ = Field3D();
    xi_pbl_ = Field3D();
    xi_diffusion_ = Field3D();
    xi_micro_prev_ = Field3D();
    xi_pbl_prev_ = Field3D();
    xi_diff_prev_ = Field3D();
    cadence_dt_s_ = 0.0;
}

void FullStochasticScheme::apply_initial_conditions(
    const ChaosConfig& cfg,
    const GridMetrics& grid,
    SimulationState& state,
    ChaosDiagnostics* diag
) {
    InitialConditionsScheme initial_conditions;
    initial_conditions.configure_compute_backend(
        native_backend_, native_generation_required_);
    initial_conditions.apply_initial_conditions(cfg, grid, state, diag);
}

/**
 * @brief Applies stochastic multipliers to active model tendency blocks.
 */
void FullStochasticScheme::apply_tendencies(
    const ChaosConfig& cfg,
    const GridMetrics& grid,
    const ChaosStateView& state_view,
    ChaosTendencies& tendencies,
    ChaosDiagnostics* diag
) {
    auto t0 = std::chrono::high_resolution_clock::now();
    Field3D* ref = first_available_tendency(tendencies);
    if (ref == nullptr)
    {
        if (diag)
        {
            diag->warnings.push_back("Full stochastic scheme: no tendency fields available");
        }
        return;
    }

    const int nr = ref->size_r();
    const int nth = ref->size_th();
    const int nz = ref->size_z();
    if (nr <= 0 || nth <= 0 || nz <= 0)
    {
        return;
    }

    if (!noise_matches_domain(xi_microphysics_, nr, nth, nz) ||
        !noise_matches_domain(xi_pbl_, nr, nth, nz) ||
        !noise_matches_domain(xi_diffusion_, nr, nth, nz))
    {
        if (microphysics_noise_.initialized() || pbl_noise_.initialized()
            || diffusion_noise_.initialized())
            throw std::runtime_error(
                "full-stochastic chaos domain shape changed without an admitted Earth-window translation");
        microphysics_noise_.initialize_domain(nr, nth, nz);
        pbl_noise_.initialize_domain(nr, nth, nz);
        diffusion_noise_.initialize_domain(nr, nth, nz);
        microphysics_noise_.render_current(xi_microphysics_);
        microphysics_noise_.render_previous(xi_micro_prev_);
        pbl_noise_.render_current(xi_pbl_);
        pbl_noise_.render_previous(xi_pbl_prev_);
        diffusion_noise_.render_current(xi_diffusion_);
        diffusion_noise_.render_previous(xi_diff_prev_);
    }

    const bool has_micro_tendencies =
        tendencies.dtheta_micro_dt != nullptr || tendencies.dqv_micro_dt != nullptr ||
        tendencies.dqc_micro_dt != nullptr || tendencies.dqr_micro_dt != nullptr ||
        tendencies.dqi_micro_dt != nullptr || tendencies.dqs_micro_dt != nullptr ||
        tendencies.dqg_micro_dt != nullptr || tendencies.dqh_micro_dt != nullptr;
    const bool has_pbl_tendencies =
        tendencies.du_pbl_dt != nullptr || tendencies.dv_pbl_dt != nullptr ||
        tendencies.dtheta_pbl_dt != nullptr || tendencies.dqv_pbl_dt != nullptr ||
        tendencies.pbl_u_force_n != nullptr ||
        tendencies.pbl_v_force_n != nullptr ||
        tendencies.pbl_theta_inventory_rate_kgk_s != nullptr ||
        tendencies.pbl_vapor_mass_rate_kg_s != nullptr ||
        tendencies.pbl_tke_tendency != nullptr;
    const bool has_conservative_pbl_sources =
        tendencies.pbl_u_force_n != nullptr ||
        tendencies.pbl_v_force_n != nullptr ||
        tendencies.pbl_theta_inventory_rate_kgk_s != nullptr ||
        tendencies.pbl_vapor_mass_rate_kg_s != nullptr;
    const bool has_diffusion_tendencies =
        tendencies.du_diff_dt != nullptr || tendencies.dv_diff_dt != nullptr ||
        tendencies.dw_diff_dt != nullptr || tendencies.dtheta_diff_dt != nullptr ||
        tendencies.dqv_diff_dt != nullptr
        || tendencies.eddy_viscosity != nullptr
        || tendencies.theta_diffusivity != nullptr
        || tendencies.moisture_diffusivity != nullptr
        || tendencies.tke_diffusivity != nullptr;

    Field3D xi_micro_effective;
    if (has_micro_tendencies)
    {
        xi_micro_effective = xi_microphysics_;
        apply_vertical_taper(xi_micro_effective, grid, cfg.taper_id, cfg.taper_z1, cfg.taper_z2);
        bound_perturbation_field(xi_micro_effective, cfg.xi_max, true);
    }

    Field3D xi_pbl_effective;
    if (has_pbl_tendencies)
    {
        xi_pbl_effective = xi_pbl_;
        apply_vertical_taper(xi_pbl_effective, grid, cfg.taper_id, cfg.taper_z1, cfg.taper_z2);
        bound_perturbation_field(xi_pbl_effective, cfg.xi_max, true);
    }

    Field3D xi_diff_effective;
    if (has_diffusion_tendencies)
    {
        xi_diff_effective = xi_diffusion_;
        apply_vertical_taper(xi_diff_effective, grid, cfg.taper_id, cfg.taper_z1, cfg.taper_z2);
        bound_perturbation_field(xi_diff_effective, cfg.xi_max, true);
    }

    double alpha_micro = 0.3;
    double alpha_pbl = 0.2;
    double alpha_diff = 0.1;
    auto it_micro = cfg.alpha_tend.find("microphysics");
    if (it_micro != cfg.alpha_tend.end()) alpha_micro = it_micro->second;
    auto it_pbl = cfg.alpha_tend.find("pbl");
    if (it_pbl != cfg.alpha_tend.end()) alpha_pbl = it_pbl->second;
    auto it_diff = cfg.alpha_tend.find("diffusion");
    if (it_diff != cfg.alpha_tend.end()) alpha_diff = it_diff->second;

    double min_multiplier = 1.0;
    double max_multiplier = 1.0;
    double sum_multiplier = 0.0;
    double sum_sq_multiplier = 0.0;
    size_t sample_count = 0;
    const PhysicalCellDomain domain = grid.staggering == StaggerType::Collocated
        ? physical_scalar_domain(nr, nth, nz, StaggerType::Collocated)
        : PhysicalCellDomain{0, nr, 0, nth, 0, nz};

    if (tendencies.microphysics_column_multipliers != nullptr)
    {
        prepare_microphysics_column_multipliers(
            *tendencies.microphysics_column_multipliers,
            xi_micro_effective, domain, alpha_micro,
            min_multiplier, max_multiplier, sum_multiplier,
            sum_sq_multiplier, sample_count);
    }
    else
    {
        // Compatibility for extension/direct callers predating the
        // production column-process contract.
        apply_block_multiplier(tendencies.dtheta_micro_dt, xi_micro_effective, domain, alpha_micro, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
        apply_block_multiplier(tendencies.dqv_micro_dt, xi_micro_effective, domain, alpha_micro, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
        apply_block_multiplier(tendencies.dqc_micro_dt, xi_micro_effective, domain, alpha_micro, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
        apply_block_multiplier(tendencies.dqr_micro_dt, xi_micro_effective, domain, alpha_micro, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
        apply_block_multiplier(tendencies.dqi_micro_dt, xi_micro_effective, domain, alpha_micro, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
        apply_block_multiplier(tendencies.dqs_micro_dt, xi_micro_effective, domain, alpha_micro, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
        apply_block_multiplier(tendencies.dqg_micro_dt, xi_micro_effective, domain, alpha_micro, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
        apply_block_multiplier(tendencies.dqh_micro_dt, xi_micro_effective, domain, alpha_micro, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    }

    if (has_conservative_pbl_sources)
    {
        apply_conservative_pbl_column_multiplier(
            tendencies, xi_pbl_effective, domain, alpha_pbl,
            min_multiplier, max_multiplier, sum_multiplier,
            sum_sq_multiplier, sample_count);
    }
    else
    {
        apply_block_multiplier(tendencies.du_pbl_dt, xi_pbl_effective, domain, alpha_pbl, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
        apply_block_multiplier(tendencies.dv_pbl_dt, xi_pbl_effective, domain, alpha_pbl, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
        apply_block_multiplier(tendencies.dtheta_pbl_dt, xi_pbl_effective, domain, alpha_pbl, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
        apply_block_multiplier(tendencies.dqv_pbl_dt, xi_pbl_effective, domain, alpha_pbl, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    }

    apply_block_multiplier(tendencies.du_diff_dt, xi_diff_effective, domain, alpha_diff, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.dv_diff_dt, xi_diff_effective, domain, alpha_diff, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.dw_diff_dt, xi_diff_effective, domain, alpha_diff, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.dtheta_diff_dt, xi_diff_effective, domain, alpha_diff, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.dqv_diff_dt, xi_diff_effective, domain, alpha_diff, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.eddy_viscosity, xi_diff_effective, domain, alpha_diff, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.theta_diffusivity, xi_diff_effective, domain, alpha_diff, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.moisture_diffusivity, xi_diff_effective, domain, alpha_diff, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.tke_diffusivity, xi_diff_effective, domain, alpha_diff, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);

    if (diag)
    {
        if (sample_count > 0)
        {
            double mean_multiplier = sum_multiplier / static_cast<double>(sample_count);
            double var_multiplier = (sum_sq_multiplier / static_cast<double>(sample_count)) -
                                    (mean_multiplier * mean_multiplier);
            diag->mean_perturbation = mean_multiplier - 1.0;
            diag->variance_perturbation = var_multiplier;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        diag->time_apply_tend += std::chrono::duration<double>(t1 - t0).count();
    }
}

/**
 * @brief Advances AR(1) noise fields one model time step.
 */
void FullStochasticScheme::step_noise(
    const ChaosConfig& cfg,
    const GridMetrics& grid,
    double dt
) {
    if (xi_microphysics_.empty() || xi_pbl_.empty() || xi_diffusion_.empty())
    {
        return;
    }

    const double rho_t = compute_temporal_correlation(dt, cfg.tau_t);
    if (time_step_counter_ == 0)
        cadence_dt_s_ = dt;
    else if (std::abs(dt - cadence_dt_s_)
             > 1.0e-12 * std::max(1.0, cadence_dt_s_))
        throw std::invalid_argument(
            "full-stochastic chaos AR(1) cadence changed after initialization");
    EarthCorrelatedNoise* noises[] = {
        &microphysics_noise_, &pbl_noise_, &diffusion_noise_};
    Field3D* current[] = {&xi_microphysics_, &xi_pbl_, &xi_diffusion_};
    Field3D* previous[] = {&xi_micro_prev_, &xi_pbl_prev_, &xi_diff_prev_};
    if (time_step_counter_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error(
            "full-stochastic chaos AR(1) epoch overflow");
    for (int process = 0; process < 3; ++process)
    {
        noises[process]->advance(rho_t, time_step_counter_);
        noises[process]->render_current(*current[process]);
        noises[process]->render_previous(*previous[process]);
    }
    time_step_counter_++;
}

bool FullStochasticScheme::set_continuous_horizontal_earth_origin(
    double origin_x_m,
    double origin_y_m,
    const ChaosConfig& cfg,
    const GridMetrics& grid,
    std::string& error)
{
    try
    {
        (void)grid;
        EarthCorrelatedNoise* noises[] = {
            &microphysics_noise_, &pbl_noise_, &diffusion_noise_};
        Field3D* current[] = {
            &xi_microphysics_, &xi_pbl_, &xi_diffusion_};
        Field3D* previous[] = {
            &xi_micro_prev_, &xi_pbl_prev_, &xi_diff_prev_};
        for (std::size_t process = 0; process < std::size(noises); ++process)
            if (noises[process]->initialized() &&
                (!noises[process]->current_render_matches(*current[process]) ||
                 !noises[process]->previous_render_matches(*previous[process])))
                throw std::invalid_argument(
                    "full-stochastic fields disagree with their Earth controls");
        const double cadence =
            time_step_counter_ == 0 ? cfg.dt_chaos : cadence_dt_s_;
        const double rho = compute_temporal_correlation(cadence, cfg.tau_t);
        std::array<EarthCorrelatedNoise, 3> candidates{
            microphysics_noise_, pbl_noise_, diffusion_noise_};
        for (EarthCorrelatedNoise& candidate : candidates)
            candidate.set_continuous_earth_origin(
                origin_x_m, origin_y_m, time_step_counter_, rho);
        for (std::size_t process = 0; process < candidates.size(); ++process)
        {
            noises[process]->swap_window_state(candidates[process]);
            if (noises[process]->initialized())
            {
                noises[process]->render_current_preallocated(*current[process]);
                noises[process]->render_previous_preallocated(*previous[process]);
            }
        }
        error.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
        return false;
    }
}

bool FullStochasticScheme::translate_horizontal_stochastic_state(
    int shift_i_cells,
    int shift_j_cells,
    const ChaosConfig& cfg,
    const GridMetrics& grid,
    std::string& error)
{
    try
    {
        (void)grid;
        if (microphysics_noise_.initialized() || pbl_noise_.initialized() ||
            diffusion_noise_.initialized())
        {
            const EarthCorrelatedNoise* noises[] = {
                &microphysics_noise_, &pbl_noise_, &diffusion_noise_};
            const Field3D* current_fields[] = {
                &xi_microphysics_, &xi_pbl_, &xi_diffusion_};
            const Field3D* previous_fields[] = {
                &xi_micro_prev_, &xi_pbl_prev_, &xi_diff_prev_};
            for (std::size_t process = 0; process < std::size(noises);
                 ++process)
            {
                Field3D rendered_current;
                Field3D rendered_previous;
                noises[process]->render_current(rendered_current);
                noises[process]->render_previous(rendered_previous);
                if (!bitwise_equal(rendered_current,
                                   *current_fields[process]) ||
                    !bitwise_equal(rendered_previous,
                                   *previous_fields[process]))
                    throw std::invalid_argument(
                        "full-stochastic fields disagree with their Earth controls");
            }
        }
        const double cadence =
            time_step_counter_ == 0 ? cfg.dt_chaos : cadence_dt_s_;
        const double rho = compute_temporal_correlation(cadence, cfg.tau_t);
        EarthCorrelatedNoise* noises[] = {
            &microphysics_noise_, &pbl_noise_, &diffusion_noise_};
        Field3D* current[] = {
            &xi_microphysics_, &xi_pbl_, &xi_diffusion_};
        Field3D* previous[] = {
            &xi_micro_prev_, &xi_pbl_prev_, &xi_diff_prev_};
        for (int process = 0; process < 3; ++process)
        {
            noises[process]->translate(
                shift_i_cells, shift_j_cells, time_step_counter_, rho);
            if (noises[process]->initialized())
            {
                noises[process]->render_current(*current[process]);
                noises[process]->render_previous(*previous[process]);
            }
        }
        error.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
        return false;
    }
}

bool FullStochasticScheme::prepare_horizontal_stochastic_state_translation(
    int shift_i_cells,
    int shift_j_cells,
    const ChaosConfig& cfg,
    const GridMetrics& grid,
    std::unique_ptr<ChaosScheme>& workspace,
    HorizontalStochasticStatePhase& phase,
    std::string& error) const
{
    try
    {
        (void)grid;
        const EarthCorrelatedNoise* noises[] = {
            &microphysics_noise_, &pbl_noise_, &diffusion_noise_};
        const Field3D* current_fields[] = {
            &xi_microphysics_, &xi_pbl_, &xi_diffusion_};
        const Field3D* previous_fields[] = {
            &xi_micro_prev_, &xi_pbl_prev_, &xi_diff_prev_};
        for (std::size_t process = 0; process < std::size(noises); ++process)
            if (!noises[process]->current_render_matches(
                    *current_fields[process]) ||
                !noises[process]->previous_render_matches(
                    *previous_fields[process]))
                throw std::invalid_argument(
                    "full-stochastic fields disagree with their Earth controls");

        auto* candidate = dynamic_cast<FullStochasticScheme*>(workspace.get());
        if (candidate == nullptr)
        {
            workspace = std::make_unique<FullStochasticScheme>();
            candidate = static_cast<FullStochasticScheme*>(workspace.get());
        }
        const double cadence =
            time_step_counter_ == 0 ? cfg.dt_chaos : cadence_dt_s_;
        const double rho = compute_temporal_correlation(cadence, cfg.tau_t);
        EarthCorrelatedNoise* candidate_noises[] = {
            &candidate->microphysics_noise_, &candidate->pbl_noise_,
            &candidate->diffusion_noise_};
        for (std::size_t process = 0; process < std::size(noises); ++process)
            noises[process]->prepare_translated_window(
                *candidate_noises[process], shift_i_cells, shift_j_cells,
                time_step_counter_, rho);
        candidate->time_step_counter_ = time_step_counter_;
        candidate->cadence_dt_s_ = cadence_dt_s_;
        if (!candidate->horizontal_stochastic_state_phase(phase, error))
            return false;
        error.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
        return false;
    }
}

void FullStochasticScheme::
exchange_prepared_horizontal_stochastic_state_translation(
    std::unique_ptr<ChaosScheme>& workspace) noexcept
{
    auto& candidate = static_cast<FullStochasticScheme&>(*workspace);
    EarthCorrelatedNoise* noises[] = {
        &microphysics_noise_, &pbl_noise_, &diffusion_noise_};
    EarthCorrelatedNoise* candidate_noises[] = {
        &candidate.microphysics_noise_, &candidate.pbl_noise_,
        &candidate.diffusion_noise_};
    Field3D* current[] = {
        &xi_microphysics_, &xi_pbl_, &xi_diffusion_};
    Field3D* previous[] = {
        &xi_micro_prev_, &xi_pbl_prev_, &xi_diff_prev_};
    for (std::size_t process = 0; process < std::size(noises); ++process)
    {
        noises[process]->swap_window_state(*candidate_noises[process]);
        if (noises[process]->initialized())
        {
            noises[process]->render_current_preallocated(*current[process]);
            noises[process]->render_previous_preallocated(*previous[process]);
        }
    }
}

bool FullStochasticScheme::horizontal_stochastic_storage_identity(
    HorizontalStochasticStorageIdentity& identity,
    std::string& error) const
{
    identity = {};
    identity.process_count = 3;
    const EarthCorrelatedNoise* noises[] = {
        &microphysics_noise_, &pbl_noise_, &diffusion_noise_};
    const Field3D* current[] = {
        &xi_microphysics_, &xi_pbl_, &xi_diffusion_};
    const Field3D* previous[] = {
        &xi_micro_prev_, &xi_pbl_prev_, &xi_diff_prev_};
    for (std::size_t process = 0; process < std::size(noises); ++process)
    {
        identity.current_controls[process] =
            noises[process]->current_controls().data();
        identity.previous_controls[process] =
            noises[process]->previous_controls().data();
        identity.current_control_capacities[process] =
            noises[process]->current_controls().capacity();
        identity.previous_control_capacities[process] =
            noises[process]->previous_controls().capacity();
        identity.current_rendered[process] = current[process]->data();
        identity.previous_rendered[process] = previous[process]->data();
    }
    error.clear();
    return true;
}

bool FullStochasticScheme::horizontal_stochastic_state_phase(
    HorizontalStochasticStatePhase& phase,
    std::string& error) const
{
    const bool initialized = microphysics_noise_.initialized();
    if (pbl_noise_.initialized() != initialized ||
        diffusion_noise_.initialized() != initialized ||
        (time_step_counter_ == 0 && cadence_dt_s_ != 0.0) ||
        (time_step_counter_ > 0 &&
         (!std::isfinite(cadence_dt_s_) || cadence_dt_s_ <= 0.0)) ||
        (initialized &&
         (microphysics_noise_.nr() != pbl_noise_.nr() ||
          microphysics_noise_.nr() != diffusion_noise_.nr() ||
          microphysics_noise_.nth() != pbl_noise_.nth() ||
          microphysics_noise_.nth() != diffusion_noise_.nth() ||
          microphysics_noise_.nz() != pbl_noise_.nz() ||
          microphysics_noise_.nz() != diffusion_noise_.nz() ||
          microphysics_noise_.earth_origin_i_cells() !=
              pbl_noise_.earth_origin_i_cells() ||
          microphysics_noise_.earth_origin_i_cells() !=
              diffusion_noise_.earth_origin_i_cells() ||
          microphysics_noise_.earth_origin_j_cells() !=
              pbl_noise_.earth_origin_j_cells() ||
          microphysics_noise_.earth_origin_j_cells() !=
              diffusion_noise_.earth_origin_j_cells() ||
          microphysics_noise_.continuous_offset_x_m() !=
              pbl_noise_.continuous_offset_x_m() ||
          microphysics_noise_.continuous_offset_x_m() !=
              diffusion_noise_.continuous_offset_x_m() ||
          microphysics_noise_.continuous_offset_y_m() !=
              pbl_noise_.continuous_offset_y_m() ||
          microphysics_noise_.continuous_offset_y_m() !=
              diffusion_noise_.continuous_offset_y_m())))
    {
        error = "full-stochastic chaos process phases disagree";
        return false;
    }
    phase.initialized = initialized;
    phase.current_epoch = time_step_counter_;
    phase.cadence_dt_s = cadence_dt_s_;
    phase.earth_origin_i_cells =
        microphysics_noise_.earth_origin_i_cells();
    phase.earth_origin_j_cells =
        microphysics_noise_.earth_origin_j_cells();
    phase.continuous_offset_x_m =
        microphysics_noise_.continuous_offset_x_m();
    phase.continuous_offset_y_m =
        microphysics_noise_.continuous_offset_y_m();
    phase.nr = microphysics_noise_.nr();
    phase.nth = microphysics_noise_.nth();
    phase.nz = microphysics_noise_.nz();
    error.clear();
    return true;
}

bool FullStochasticScheme::initialize_nested_horizontal_stochastic_state(
    const ChaosScheme& parent_scheme,
    std::int64_t earth_origin_i_cells,
    std::int64_t earth_origin_j_cells,
    int nr,
    int nth,
    int nz,
    std::uint64_t target_epoch,
    double cadence_dt_s,
    const ChaosConfig& cfg,
    const GridMetrics& grid,
    std::string& error)
{
    try
    {
        (void)grid;
        const auto* parent =
            dynamic_cast<const FullStochasticScheme*>(&parent_scheme);
        if (parent == nullptr)
            throw std::invalid_argument(
                "nested full-stochastic chaos requires the same parent scheme");
        HorizontalStochasticStatePhase parent_phase;
        if (!parent->horizontal_stochastic_state_phase(parent_phase, error))
            return false;
        if (target_epoch > parent_phase.current_epoch ||
            !std::isfinite(cadence_dt_s) || cadence_dt_s < 0.0 ||
            (target_epoch == 0 && cadence_dt_s != 0.0) ||
            (target_epoch > 0 && cadence_dt_s != cfg.dt_chaos))
            throw std::invalid_argument(
                "nested full-stochastic chaos target phase is invalid");
        EarthCorrelatedNoise* child_noises[] = {
            &microphysics_noise_, &pbl_noise_, &diffusion_noise_};
        const EarthCorrelatedNoise* parent_noises[] = {
            &parent->microphysics_noise_, &parent->pbl_noise_,
            &parent->diffusion_noise_};
        Field3D* current[] = {
            &xi_microphysics_, &xi_pbl_, &xi_diffusion_};
        Field3D* previous[] = {
            &xi_micro_prev_, &xi_pbl_prev_, &xi_diff_prev_};
        if (!parent_phase.initialized)
        {
            if (target_epoch != 0)
                throw std::invalid_argument(
                    "uninitialized parent chaos cannot own an evolved child epoch");
            for (std::size_t process = 0; process < std::size(child_noises);
                 ++process)
            {
                if (!child_noises[process]->restore_uninitialized_origin(
                        earth_origin_i_cells, earth_origin_j_cells, error))
                    return false;
                *current[process] = Field3D{};
                *previous[process] = Field3D{};
            }
        }
        else
        {
            const double rho = compute_temporal_correlation(
                cfg.dt_chaos, cfg.tau_t);
            for (std::size_t process = 0; process < std::size(child_noises);
                 ++process)
            {
                child_noises[process]->initialize_domain_from(
                    *parent_noises[process], earth_origin_i_cells,
                    earth_origin_j_cells, nr, nth, nz,
                    parent_phase.current_epoch, target_epoch, rho);
                child_noises[process]->render_current(*current[process]);
                child_noises[process]->render_previous(*previous[process]);
            }
        }
        time_step_counter_ = target_epoch;
        cadence_dt_s_ = cadence_dt_s;
        error.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
        return false;
    }
}

}
