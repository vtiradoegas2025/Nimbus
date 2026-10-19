// Stochastic multipliers for deterministic boundary-layer tendencies.

#include "boundary_layer.hpp"
#include "chaos/base/perturbation_field.hpp"
#include "chaos/base/correlation_filter.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "turbulence/turbulence_base.hpp"
#include "util/log.hpp"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace {

bool same_shape(const Field3D& lhs, const Field3D& rhs)
{
    return lhs.empty() == rhs.empty() &&
        (lhs.empty() || (lhs.size_r() == rhs.size_r() &&
                         lhs.size_th() == rhs.size_th() &&
                         lhs.size_z() == rhs.size_z()));
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

Field3D* first_available_pbl_tendency(chaos::ChaosTendencies& tendencies)
{
    if (tendencies.dtheta_pbl_dt != nullptr) return tendencies.dtheta_pbl_dt;
    if (tendencies.du_pbl_dt != nullptr) return tendencies.du_pbl_dt;
    if (tendencies.dv_pbl_dt != nullptr) return tendencies.dv_pbl_dt;
    if (tendencies.dqv_pbl_dt != nullptr) return tendencies.dqv_pbl_dt;
    if (tendencies.pbl_u_force_n != nullptr)
        return tendencies.pbl_u_force_n;
    if (tendencies.pbl_v_force_n != nullptr)
        return tendencies.pbl_v_force_n;
    if (tendencies.pbl_theta_inventory_rate_kgk_s != nullptr)
        return tendencies.pbl_theta_inventory_rate_kgk_s;
    if (tendencies.pbl_vapor_mass_rate_kg_s != nullptr)
        return tendencies.pbl_vapor_mass_rate_kg_s;
    return nullptr;
}

bool noise_matches_domain(const Field3D& noise, int nr, int nth, int nz)
{
    return !noise.empty() &&
           noise.size_r() == nr &&
           noise.size_th() == nth &&
           noise.size_z() == nz;
}

bool bitwise_equal(const Field3D& lhs, const Field3D& rhs)
{
    return same_shape(lhs, rhs) && (lhs.empty() || std::memcmp(
        lhs.data(), rhs.data(), lhs.size() * sizeof(float)) == 0);
}

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

void apply_conservative_column_multiplier(
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
    for (Field3D* field : volume_fields)
        if (field != nullptr &&
            (field->size_r() != xi.size_r() ||
             field->size_th() != xi.size_th() ||
             field->size_z() != xi.size_z()))
            throw std::invalid_argument(
                "boundary-layer stochastic conserved-source shape mismatch");

    std::vector<float>* surface_fields[] = {
        tendencies.pbl_tau_u,
        tendencies.pbl_tau_v,
        tendencies.pbl_sensible_heat_flux,
        tendencies.pbl_latent_heat_flux,
        tendencies.pbl_moisture_flux};
    const std::size_t surface_size =
        static_cast<std::size_t>(xi.size_r()) *
        static_cast<std::size_t>(xi.size_th());
    for (std::vector<float>* field : surface_fields)
        if (field != nullptr && field->size() != surface_size)
            throw std::invalid_argument(
                "boundary-layer stochastic surface shape mismatch");
    if (tendencies.pbl_ustar != nullptr &&
        tendencies.pbl_ustar->size() != surface_size)
        throw std::invalid_argument(
            "boundary-layer stochastic friction-velocity shape mismatch");

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
            const std::size_t index =
                static_cast<std::size_t>(i) *
                static_cast<std::size_t>(xi.size_th()) +
                static_cast<std::size_t>(j);
            for (std::vector<float>* field : surface_fields)
                if (field != nullptr) (*field)[index] *= multiplier_f;
            if (tendencies.pbl_ustar != nullptr)
                (*tendencies.pbl_ustar)[index] *=
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

void BoundaryLayerScheme::configure_compute_backend(
    ComputeBackend* backend, bool required) noexcept
{
    native_backend_ = backend;
    native_generation_required_ = required;
    earth_noise_.configure_native_evolution(backend, required);
}

PortableSchemeState BoundaryLayerScheme::portable_state() const
{
    PortableSchemeState state;
    state.version = 3;
    state.fields = {
        xi_pbl_, xi_pbl_prev_,
        earth_noise_.current_controls(), earth_noise_.previous_controls()};
    state.integers = {
        time_step_counter_,
        encode_signed(earth_noise_.earth_origin_i_cells()),
        encode_signed(earth_noise_.earth_origin_j_cells()),
        encode_signed(earth_noise_.control_i_min()),
        encode_signed(earth_noise_.control_j_min()),
        static_cast<std::uint64_t>(earth_noise_.nr()),
        static_cast<std::uint64_t>(earth_noise_.nth()),
        static_cast<std::uint64_t>(earth_noise_.nz())};
    state.reals = {
        cadence_dt_s_, earth_noise_.continuous_offset_x_m(),
        earth_noise_.continuous_offset_y_m()};
    return state;
}

bool BoundaryLayerScheme::restore_portable_state(
    const PortableSchemeState& state, std::string& error)
{
    const bool legacy_v2 = state.version == 2 && state.reals.size() == 1;
    const bool current_v3 = state.version == 3 && state.reals.size() == 3;
    if ((!legacy_v2 && !current_v3) || state.fields.size() != 4 ||
        state.integers.size() != 8 ||
        !same_shape(state.fields[0], state.fields[1]) ||
        !same_shape(state.fields[2], state.fields[3]) ||
        !std::isfinite(state.reals[0]) || state.reals[0] < 0.0 ||
        (current_v3 &&
         (!std::isfinite(state.reals[1]) || !std::isfinite(state.reals[2]))) ||
        state.integers[5] > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        state.integers[6] > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        state.integers[7] > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    {
        error = "invalid boundary-layer chaos portable state";
        return false;
    }
    const int nr = static_cast<int>(state.integers[5]);
    const int nth = static_cast<int>(state.integers[6]);
    const int nz = static_cast<int>(state.integers[7]);
    if ((nr == 0 || nth == 0 || nz == 0) &&
        (!state.fields[0].empty() || !state.fields[2].empty()))
    {
        error = "invalid empty boundary-layer chaos portable state";
        return false;
    }
    const double offset_x = current_v3 ? state.reals[1] : 0.0;
    const double offset_y = current_v3 ? state.reals[2] : 0.0;
    if (nr > 0 && (!earth_noise_.restore(
            state.fields[2], state.fields[3], nr, nth, nz,
            decode_signed(state.integers[1]), decode_signed(state.integers[2]),
            decode_signed(state.integers[3]), decode_signed(state.integers[4]),
            offset_x, offset_y,
            error)))
        return false;
    if (nr == 0 && !earth_noise_.restore_uninitialized_origin(
            decode_signed(state.integers[1]), decode_signed(state.integers[2]),
            error))
        return false;
    Field3D rendered_current;
    Field3D rendered_previous;
    if (nr > 0)
    {
        earth_noise_.render_current(rendered_current);
        earth_noise_.render_previous(rendered_previous);
        if (!bitwise_equal(rendered_current, state.fields[0])
            || !bitwise_equal(rendered_previous, state.fields[1]))
        {
            error = "boundary-layer chaos fields disagree with Earth controls";
            return false;
        }
    }
    xi_pbl_ = state.fields[0];
    xi_pbl_prev_ = state.fields[1];
    time_step_counter_ = state.integers[0];
    cadence_dt_s_ = state.reals[0];
    if ((time_step_counter_ == 0 && cadence_dt_s_ != 0.0)
        || (time_step_counter_ > 0 && cadence_dt_s_ <= 0.0)
        || (nr == 0 && time_step_counter_ != 0))
    {
        error = "boundary-layer chaos cadence history is inconsistent";
        return false;
    }
    error.clear();
    return true;
}

/**
 * @brief Initializes the boundary layer scheme.
 */
void BoundaryLayerScheme::initialize(const ChaosConfig& cfg, const GridMetrics& grid)
{
    (void)grid;
    tmv::log_info("Initialized chaos scheme: boundary_layer (BL-focused perturbations)");

    // Validate the public filter vocabulary even though evolving schemes now
    // use the window-independent compact Earth lattice implementation.
    (void)create_correlation_filter(cfg.filter_id, cfg.Lx, cfg.Ly);
    earth_noise_ = EarthCorrelatedNoise{};
    earth_noise_.configure_native_evolution(
        native_backend_, native_generation_required_);
    earth_noise_.configure(
        cfg.seed, cfg.member_id, 3001ULL, 3101ULL,
        cfg.Lx, cfg.Ly, grid.dx, grid.dy);
    tmv::log_info("  Spatial noise: ", EarthCorrelatedNoise::algorithm_id(),
                  " (legacy filter selection ", cfg.filter_id, ")");

    xi_pbl_ = Field3D();
    xi_pbl_prev_ = Field3D();
    time_step_counter_ = 0;
    cadence_dt_s_ = 0.0;
}

/**
 * @brief Applies the initial conditions to the boundary layer scheme.
 */
void BoundaryLayerScheme::apply_initial_conditions(
    const ChaosConfig& cfg,
    const GridMetrics& grid,
    SimulationState& state,
    ChaosDiagnostics* diag
)
{
    if
    (diag)
    {
        diag->warnings.push_back("BL scheme: no IC perturbations applied (focus on tendencies)");
    }
}

/**
 * @brief Applies the tendencies to the boundary layer scheme.
 */
void BoundaryLayerScheme::apply_tendencies(
    const ChaosConfig& cfg,
    const GridMetrics& grid,
    const ChaosStateView& state_view,
    ChaosTendencies& tendencies,
    ChaosDiagnostics* diag
)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    Field3D* ref = first_available_pbl_tendency(tendencies);
    if (ref == nullptr)
    {
        if (diag)
        {
            diag->warnings.push_back("BL scheme: no PBL tendency fields available");
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

    if (!noise_matches_domain(xi_pbl_, nr, nth, nz))
    {
        if (earth_noise_.initialized())
            throw std::runtime_error(
                "boundary-layer chaos domain shape changed without an admitted Earth-window translation");
        earth_noise_.initialize_domain(nr, nth, nz);
        earth_noise_.render_current(xi_pbl_);
        earth_noise_.render_previous(xi_pbl_prev_);
    }

    Field3D xi_effective = xi_pbl_;
    apply_vertical_taper(xi_effective, grid, cfg.taper_id, cfg.taper_z1, cfg.taper_z2);
    bound_perturbation_field(xi_effective, cfg.xi_max, true);

    double alpha = 0.2;
    auto alpha_it = cfg.alpha_tend.find("pbl");
    if (alpha_it != cfg.alpha_tend.end())
    {
        alpha = alpha_it->second;
    }

    double min_multiplier = 1.0;
    double max_multiplier = 1.0;
    double sum_multiplier = 0.0;
    double sum_sq_multiplier = 0.0;
    size_t sample_count = 0;
    const PhysicalCellDomain domain = grid.staggering == StaggerType::Collocated
        ? physical_scalar_domain(nr, nth, nz, StaggerType::Collocated)
        : PhysicalCellDomain{0, nr, 0, nth, 0, nz};

    const bool conserved_sources =
        tendencies.pbl_u_force_n != nullptr ||
        tendencies.pbl_v_force_n != nullptr ||
        tendencies.pbl_theta_inventory_rate_kgk_s != nullptr ||
        tendencies.pbl_vapor_mass_rate_kg_s != nullptr;
    if (conserved_sources)
    {
        apply_conservative_column_multiplier(
            tendencies, xi_effective, domain, alpha, min_multiplier,
            max_multiplier, sum_multiplier, sum_sq_multiplier,
            sample_count);
    }
    else
    {
        apply_block_multiplier(tendencies.du_pbl_dt, xi_effective, domain, alpha,
            min_multiplier, max_multiplier, sum_multiplier,
            sum_sq_multiplier, sample_count);
        apply_block_multiplier(tendencies.dv_pbl_dt, xi_effective, domain, alpha,
            min_multiplier, max_multiplier, sum_multiplier,
            sum_sq_multiplier, sample_count);
        apply_block_multiplier(tendencies.dtheta_pbl_dt, xi_effective, domain, alpha,
            min_multiplier, max_multiplier, sum_multiplier,
            sum_sq_multiplier, sample_count);
        apply_block_multiplier(tendencies.dqv_pbl_dt, xi_effective, domain, alpha,
            min_multiplier, max_multiplier, sum_multiplier,
            sum_sq_multiplier, sample_count);
    }

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


void BoundaryLayerScheme::step_noise(
    const ChaosConfig& cfg,
    const GridMetrics& grid,
    double dt
) {
    if (xi_pbl_.empty())
    {
        return;
    }

    if (xi_pbl_prev_.empty())
    {
        xi_pbl_prev_ = xi_pbl_;
    }

    const double rho_t = compute_temporal_correlation(dt, cfg.tau_t);
    if (time_step_counter_ == 0)
        cadence_dt_s_ = dt;
    else if (std::abs(dt - cadence_dt_s_)
             > 1.0e-12 * std::max(1.0, cadence_dt_s_))
        throw std::invalid_argument(
            "boundary-layer chaos AR(1) cadence changed after initialization");
    if (time_step_counter_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error(
            "boundary-layer chaos AR(1) epoch overflow");
    earth_noise_.advance(rho_t, time_step_counter_);
    earth_noise_.render_current(xi_pbl_);
    earth_noise_.render_previous(xi_pbl_prev_);
    ++time_step_counter_;
}

bool BoundaryLayerScheme::set_continuous_horizontal_earth_origin(
    double origin_x_m,
    double origin_y_m,
    const ChaosConfig& cfg,
    const GridMetrics& grid,
    std::string& error)
{
    try
    {
        (void)grid;
        if (earth_noise_.initialized() &&
            (!earth_noise_.current_render_matches(xi_pbl_) ||
             !earth_noise_.previous_render_matches(xi_pbl_prev_)))
            throw std::invalid_argument(
                "boundary-layer stochastic fields disagree with their Earth controls");
        const double cadence =
            time_step_counter_ == 0 ? cfg.dt_chaos : cadence_dt_s_;
        const double rho = compute_temporal_correlation(cadence, cfg.tau_t);
        earth_noise_.set_continuous_earth_origin(
            origin_x_m, origin_y_m, time_step_counter_, rho);
        if (earth_noise_.initialized())
        {
            earth_noise_.render_current_preallocated(xi_pbl_);
            earth_noise_.render_previous_preallocated(xi_pbl_prev_);
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

bool BoundaryLayerScheme::translate_horizontal_stochastic_state(
    int shift_i_cells,
    int shift_j_cells,
    const ChaosConfig& cfg,
    const GridMetrics& grid,
    std::string& error)
{
    try
    {
        (void)grid;
        if (earth_noise_.initialized())
        {
            Field3D rendered_current;
            Field3D rendered_previous;
            earth_noise_.render_current(rendered_current);
            earth_noise_.render_previous(rendered_previous);
            if (!bitwise_equal(rendered_current, xi_pbl_) ||
                !bitwise_equal(rendered_previous, xi_pbl_prev_))
                throw std::invalid_argument(
                    "boundary-layer stochastic fields disagree with their Earth controls");
        }
        const double cadence =
            time_step_counter_ == 0 ? cfg.dt_chaos : cadence_dt_s_;
        const double rho = compute_temporal_correlation(cadence, cfg.tau_t);
        earth_noise_.translate(
            shift_i_cells, shift_j_cells, time_step_counter_, rho);
        if (earth_noise_.initialized())
        {
            earth_noise_.render_current(xi_pbl_);
            earth_noise_.render_previous(xi_pbl_prev_);
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

bool BoundaryLayerScheme::prepare_horizontal_stochastic_state_translation(
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
        if (!earth_noise_.current_render_matches(xi_pbl_) ||
            !earth_noise_.previous_render_matches(xi_pbl_prev_))
            throw std::invalid_argument(
                "boundary-layer stochastic fields disagree with their Earth controls");
        auto* candidate = dynamic_cast<BoundaryLayerScheme*>(workspace.get());
        if (candidate == nullptr)
        {
            workspace = std::make_unique<BoundaryLayerScheme>();
            candidate = static_cast<BoundaryLayerScheme*>(workspace.get());
        }
        const double cadence =
            time_step_counter_ == 0 ? cfg.dt_chaos : cadence_dt_s_;
        const double rho = compute_temporal_correlation(cadence, cfg.tau_t);
        earth_noise_.prepare_translated_window(
            candidate->earth_noise_, shift_i_cells, shift_j_cells,
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

void BoundaryLayerScheme::
exchange_prepared_horizontal_stochastic_state_translation(
    std::unique_ptr<ChaosScheme>& workspace) noexcept
{
    auto& candidate =
        static_cast<BoundaryLayerScheme&>(*workspace);
    earth_noise_.swap_window_state(candidate.earth_noise_);
    if (earth_noise_.initialized())
    {
        earth_noise_.render_current_preallocated(xi_pbl_);
        earth_noise_.render_previous_preallocated(xi_pbl_prev_);
    }
}

bool BoundaryLayerScheme::horizontal_stochastic_storage_identity(
    HorizontalStochasticStorageIdentity& identity,
    std::string& error) const
{
    identity = {};
    identity.process_count = 1;
    identity.current_controls[0] = earth_noise_.current_controls().data();
    identity.previous_controls[0] = earth_noise_.previous_controls().data();
    identity.current_control_capacities[0] =
        earth_noise_.current_controls().capacity();
    identity.previous_control_capacities[0] =
        earth_noise_.previous_controls().capacity();
    identity.current_rendered[0] = xi_pbl_.data();
    identity.previous_rendered[0] = xi_pbl_prev_.data();
    error.clear();
    return true;
}

bool BoundaryLayerScheme::horizontal_stochastic_state_phase(
    HorizontalStochasticStatePhase& phase,
    std::string& error) const
{
    if ((time_step_counter_ == 0 && cadence_dt_s_ != 0.0) ||
        (time_step_counter_ > 0 &&
         (!std::isfinite(cadence_dt_s_) || cadence_dt_s_ <= 0.0)) ||
        (earth_noise_.initialized() &&
         (earth_noise_.nr() <= 0 || earth_noise_.nth() <= 0 ||
          earth_noise_.nz() <= 0)))
    {
        error = "boundary-layer chaos phase is inconsistent";
        return false;
    }
    phase.initialized = earth_noise_.initialized();
    phase.current_epoch = time_step_counter_;
    phase.cadence_dt_s = cadence_dt_s_;
    phase.earth_origin_i_cells = earth_noise_.earth_origin_i_cells();
    phase.earth_origin_j_cells = earth_noise_.earth_origin_j_cells();
    phase.continuous_offset_x_m = earth_noise_.continuous_offset_x_m();
    phase.continuous_offset_y_m = earth_noise_.continuous_offset_y_m();
    phase.nr = earth_noise_.nr();
    phase.nth = earth_noise_.nth();
    phase.nz = earth_noise_.nz();
    error.clear();
    return true;
}

bool BoundaryLayerScheme::initialize_nested_horizontal_stochastic_state(
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
            dynamic_cast<const BoundaryLayerScheme*>(&parent_scheme);
        if (parent == nullptr)
            throw std::invalid_argument(
                "nested boundary-layer chaos requires the same parent scheme");
        HorizontalStochasticStatePhase parent_phase;
        if (!parent->horizontal_stochastic_state_phase(parent_phase, error))
            return false;
        if (target_epoch > parent_phase.current_epoch ||
            !std::isfinite(cadence_dt_s) || cadence_dt_s < 0.0 ||
            (target_epoch == 0 && cadence_dt_s != 0.0) ||
            (target_epoch > 0 && cadence_dt_s != cfg.dt_chaos))
            throw std::invalid_argument(
                "nested boundary-layer chaos target phase is invalid");
        if (!parent_phase.initialized)
        {
            if (target_epoch != 0)
                throw std::invalid_argument(
                    "uninitialized parent chaos cannot own an evolved child epoch");
            if (!earth_noise_.restore_uninitialized_origin(
                    earth_origin_i_cells, earth_origin_j_cells, error))
                return false;
            xi_pbl_ = Field3D{};
            xi_pbl_prev_ = Field3D{};
        }
        else
        {
            const double rho = compute_temporal_correlation(
                cfg.dt_chaos, cfg.tau_t);
            earth_noise_.initialize_domain_from(
                parent->earth_noise_, earth_origin_i_cells,
                earth_origin_j_cells, nr, nth, nz,
                parent_phase.current_epoch, target_epoch, rho);
            earth_noise_.render_current(xi_pbl_);
            earth_noise_.render_previous(xi_pbl_prev_);
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
