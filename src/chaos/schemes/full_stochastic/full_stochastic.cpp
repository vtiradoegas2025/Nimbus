/**
 * @file full_stochastic.cpp
 * @brief Implementation for the chaos module.
 *
 * Provides executable logic for the chaos runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/chaos subsystem.
 */

#include "full_stochastic.hpp"
#include "chaos/base/perturbation_field.hpp"
#include "chaos/base/correlation_filter.hpp"
#include "core/simulation.hpp"
#include "turbulence/turbulence_base.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>

namespace {

/**
 * @brief Ensures reusable 2D workspace dimensions for per-level filtering.
 */
void ensure_slice_workspace(std::vector<std::vector<double>>& slice, int nr, int nth)
{
    if (nr <= 0 || nth <= 0)
    {
        slice.clear();
        return;
    }

    if (slice.size() != static_cast<size_t>(nr) ||
        slice[0].size() != static_cast<size_t>(nth))
    {
        slice.assign(static_cast<size_t>(nr),
                     std::vector<double>(static_cast<size_t>(nth), 0.0));
    }
}

/**
 * @brief Applies horizontal spatial correlation independently on each z level.
 */
void apply_horizontal_correlation(
    Field3D& noise,
    chaos::CorrelationFilter* correlation_filter,
    double dx, double dy,
    std::vector<std::vector<double>>& slice
)
{
    if (correlation_filter == nullptr || noise.empty())
    {
        return;
    }

    const int nr = noise.size_r();
    const int nth = noise.size_th();
    const int nz = noise.size_z();
    if (nr <= 0 || nth <= 0 || nz <= 0)
    {
        return;
    }

    ensure_slice_workspace(slice, nr, nth);
    float* noise_data = noise.data();
    const size_t nz_stride = static_cast<size_t>(nz);

    for (int k = 0; k < nz; ++k)
    {
        for (int i = 0; i < nr; ++i)
        {
            for (int j = 0; j < nth; ++j)
            {
                const size_t idx = (static_cast<size_t>(i) * static_cast<size_t>(nth) +
                                    static_cast<size_t>(j)) * nz_stride + static_cast<size_t>(k);
                slice[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                    static_cast<double>(noise_data[idx]);
            }
        }
        correlation_filter->apply_2d(slice, dx, dy);
        for (int i = 0; i < nr; ++i)
        {
            for (int j = 0; j < nth; ++j)
            {
                const size_t idx = (static_cast<size_t>(i) * static_cast<size_t>(nth) +
                                    static_cast<size_t>(j)) * nz_stride + static_cast<size_t>(k);
                noise_data[idx] = static_cast<float>(slice[static_cast<size_t>(i)][static_cast<size_t>(j)]);
            }
        }
    }
}

/**
 * @brief Builds cell-centered height levels from grid metrics.
 */
std::vector<double> build_z_levels(const GridMetrics& grid, int nz)
{
    std::vector<double> z_levels(nz, 0.0);
    if (!grid.z_int.empty() && static_cast<int>(grid.z_int.size()) >= nz + 1)
    {
        for (int k = 0; k < nz; ++k)
        {
            z_levels[k] = 0.5 * (grid.z_int[k] + grid.z_int[k + 1]);
        }
        return z_levels;
    }

    double z = 0.0;
    for (int k = 0; k < nz; ++k)
    {
        double dz_k = (!grid.dz.empty() && k < static_cast<int>(grid.dz.size())) ? grid.dz[k] : 100.0;
        z_levels[k] = z + 0.5 * dz_k;
        z += dz_k;
    }
    return z_levels;
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
    if (t.du_diff_dt != nullptr) return t.du_diff_dt;
    if (t.dtheta_diff_dt != nullptr) return t.dtheta_diff_dt;
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
 * @brief Initializes correlated, normalized stochastic noise fields.
 */
void initialize_noise_field(
    Field3D& noise,
    Field3D& previous,
    chaos::ChaosRNG& rng,
    chaos::CorrelationFilter* filter,
    std::vector<std::vector<double>>& slice,
    int nr,
    int nth,
    int nz,
    uint64_t stream_key,
    const std::string& field_name,
    double dx,
    double dy
)
{
    noise = chaos::generate_white_noise_field3d(rng, nr, nth, nz, stream_key, field_name);
    apply_horizontal_correlation(noise, filter, dx, dy, slice);
    chaos::renormalize_to_unit_variance(noise);
    previous = noise;
}

/**
 * @brief Applies multiplicative stochastic scaling to one tendency block.
 */
void apply_block_multiplier(
    Field3D* tendency,
    const Field3D& xi,
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

    float* tendency_values = tendency->data();
    const float* noise_values = xi.data();
    const size_t n = tendency->size();
    for (size_t idx = 0; idx < n; ++idx)
    {
        double multiplier = 1.0 + alpha * static_cast<double>(noise_values[idx]);
        multiplier = std::clamp(multiplier, 0.0, 2.0);
        tendency_values[idx] = static_cast<float>(static_cast<double>(tendency_values[idx]) * multiplier);
        min_multiplier = std::min(min_multiplier, multiplier);
        max_multiplier = std::max(max_multiplier, multiplier);
        sum_multiplier += multiplier;
        sum_sq_multiplier += multiplier * multiplier;
        ++sample_count;
    }
}

}

namespace chaos {

/**
 * @brief Initializes RNG, filters, and internal stochastic state.
 */
void FullStochasticScheme::initialize(const ChaosConfig& cfg, const GridMetrics& grid) {
    std::cout << "Initialized chaos scheme: full_stochastic (evolving model error)" << std::endl;

    rng_ = ChaosRNG(cfg.seed, cfg.member_id);
    correlation_filter_ = create_correlation_filter(cfg.filter_id, cfg.Lx, cfg.Ly);

    time_step_counter_ = 0;
    xi_microphysics_ = Field3D();
    xi_pbl_ = Field3D();
    xi_diffusion_ = Field3D();
    xi_micro_prev_ = Field3D();
    xi_pbl_prev_ = Field3D();
    xi_diff_prev_ = Field3D();
    horizontal_slice_workspace_.clear();
}

/**
 * @brief Applies initial-condition perturbations (none for this scheme).
 */
void FullStochasticScheme::apply_initial_conditions(
    const ChaosConfig& cfg,
    const GridMetrics& grid,
    SimulationState& state,
    ChaosDiagnostics* diag
) {
    if (diag) {
        diag->warnings.push_back("Full stochastic scheme: no IC perturbations applied (focus on evolving tendencies)");
    }
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

    const double dx = (grid.dx > 0.0) ? grid.dx : 1.0;
    const double dy = (grid.dy > 0.0) ? grid.dy : dx;

    if (!noise_matches_domain(xi_microphysics_, nr, nth, nz) ||
        !noise_matches_domain(xi_pbl_, nr, nth, nz) ||
        !noise_matches_domain(xi_diffusion_, nr, nth, nz))
    {
        initialize_noise_field(xi_microphysics_, xi_micro_prev_, rng_, correlation_filter_.get(),
                               horizontal_slice_workspace_, nr, nth, nz, 4001ULL, "xi_micro", dx, dy);
        initialize_noise_field(xi_pbl_, xi_pbl_prev_, rng_, correlation_filter_.get(),
                               horizontal_slice_workspace_, nr, nth, nz, 4002ULL, "xi_pbl", dx, dy);
        initialize_noise_field(xi_diffusion_, xi_diff_prev_, rng_, correlation_filter_.get(),
                               horizontal_slice_workspace_, nr, nth, nz, 4003ULL, "xi_diff", dx, dy);
    }

    const auto z_levels = build_z_levels(grid, nz);
    const bool has_micro_tendencies =
        tendencies.dtheta_micro_dt != nullptr || tendencies.dqv_micro_dt != nullptr ||
        tendencies.dqc_micro_dt != nullptr || tendencies.dqr_micro_dt != nullptr ||
        tendencies.dqi_micro_dt != nullptr || tendencies.dqs_micro_dt != nullptr ||
        tendencies.dqg_micro_dt != nullptr || tendencies.dqh_micro_dt != nullptr;
    const bool has_pbl_tendencies =
        tendencies.du_pbl_dt != nullptr || tendencies.dv_pbl_dt != nullptr ||
        tendencies.dtheta_pbl_dt != nullptr || tendencies.dqv_pbl_dt != nullptr;
    const bool has_diffusion_tendencies =
        tendencies.du_diff_dt != nullptr || tendencies.dv_diff_dt != nullptr ||
        tendencies.dw_diff_dt != nullptr || tendencies.dtheta_diff_dt != nullptr ||
        tendencies.dqv_diff_dt != nullptr;

    Field3D xi_micro_effective;
    if (has_micro_tendencies)
    {
        xi_micro_effective = xi_microphysics_;
        apply_vertical_taper(xi_micro_effective, z_levels, cfg.taper_id, cfg.taper_z1, cfg.taper_z2);
        bound_perturbation_field(xi_micro_effective, cfg.xi_max, true);
    }

    Field3D xi_pbl_effective;
    if (has_pbl_tendencies)
    {
        xi_pbl_effective = xi_pbl_;
        apply_vertical_taper(xi_pbl_effective, z_levels, cfg.taper_id, cfg.taper_z1, cfg.taper_z2);
        bound_perturbation_field(xi_pbl_effective, cfg.xi_max, true);
    }

    Field3D xi_diff_effective;
    if (has_diffusion_tendencies)
    {
        xi_diff_effective = xi_diffusion_;
        apply_vertical_taper(xi_diff_effective, z_levels, cfg.taper_id, cfg.taper_z1, cfg.taper_z2);
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

    apply_block_multiplier(tendencies.dtheta_micro_dt, xi_micro_effective, alpha_micro, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.dqv_micro_dt, xi_micro_effective, alpha_micro, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.dqc_micro_dt, xi_micro_effective, alpha_micro, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.dqr_micro_dt, xi_micro_effective, alpha_micro, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.dqi_micro_dt, xi_micro_effective, alpha_micro, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.dqs_micro_dt, xi_micro_effective, alpha_micro, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.dqg_micro_dt, xi_micro_effective, alpha_micro, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.dqh_micro_dt, xi_micro_effective, alpha_micro, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);

    apply_block_multiplier(tendencies.du_pbl_dt, xi_pbl_effective, alpha_pbl, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.dv_pbl_dt, xi_pbl_effective, alpha_pbl, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.dtheta_pbl_dt, xi_pbl_effective, alpha_pbl, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.dqv_pbl_dt, xi_pbl_effective, alpha_pbl, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);

    apply_block_multiplier(tendencies.du_diff_dt, xi_diff_effective, alpha_diff, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.dv_diff_dt, xi_diff_effective, alpha_diff, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.dw_diff_dt, xi_diff_effective, alpha_diff, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.dtheta_diff_dt, xi_diff_effective, alpha_diff, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);
    apply_block_multiplier(tendencies.dqv_diff_dt, xi_diff_effective, alpha_diff, min_multiplier, max_multiplier, sum_multiplier, sum_sq_multiplier, sample_count);

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
    evolve_ar1_3d(xi_microphysics_, xi_micro_prev_, rho_t, rng_, 4101ULL, time_step_counter_);
    evolve_ar1_3d(xi_pbl_, xi_pbl_prev_, rho_t, rng_, 4102ULL, time_step_counter_);
    evolve_ar1_3d(xi_diffusion_, xi_diff_prev_, rho_t, rng_, 4103ULL, time_step_counter_);

    const double dx = (grid.dx > 0.0) ? grid.dx : 1.0;
    const double dy = (grid.dy > 0.0) ? grid.dy : dx;
    apply_horizontal_correlation(xi_microphysics_, correlation_filter_.get(), dx, dy, horizontal_slice_workspace_);
    apply_horizontal_correlation(xi_pbl_, correlation_filter_.get(), dx, dy, horizontal_slice_workspace_);
    apply_horizontal_correlation(xi_diffusion_, correlation_filter_.get(), dx, dy, horizontal_slice_workspace_);
    renormalize_to_unit_variance(xi_microphysics_);
    renormalize_to_unit_variance(xi_pbl_);
    renormalize_to_unit_variance(xi_diffusion_);

    xi_micro_prev_ = xi_microphysics_;
    xi_pbl_prev_ = xi_pbl_;
    xi_diff_prev_ = xi_diffusion_;
    time_step_counter_++;
}

}
