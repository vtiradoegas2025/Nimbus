/**
 * @file initial_conditions.cpp
 * @brief Implementation for the chaos module.
 *
 * Provides executable logic for the chaos runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/chaos subsystem.
 */

#include "initial_conditions.hpp"
#include "chaos/base/perturbation_field.hpp"
#include "chaos/base/correlation_filter.hpp"
#include "core/simulation.hpp"
#include "physics/turbulence_base.hpp"
#include <iostream>
#include <chrono>
#include <cmath>

namespace {

/**
 * @brief Resolves a state variable id to the corresponding mutable field.
 */
Field3D* resolve_state_field(SimulationState& state, const std::string& var)
{
    if (var == "u") return state.u;
    if (var == "v" || var == "v_theta") return state.v_theta;
    if (var == "w") return state.w;
    if (var == "theta") return state.theta;
    if (var == "qv") return state.qv;
    if (var == "qc") return state.qc;
    if (var == "qr") return state.qr;
    if (var == "qi") return state.qi;
    if (var == "qs") return state.qs;
    if (var == "qg") return state.qg;
    if (var == "qh") return state.qh;
    return nullptr;
}

/**
 * @brief Returns true for moisture-mixing-ratio state variables.
 */
bool is_moisture_var(const std::string& var)
{
    return var == "qv" || var == "qc" || var == "qr" ||
           var == "qi" || var == "qs" || var == "qg" || var == "qh";
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
    }
    else if (!grid.dz.empty())
    {
        double z = 0.0;
        for (int k = 0; k < nz; ++k)
        {
            double dz_k = (k < static_cast<int>(grid.dz.size())) ? grid.dz[k] : grid.dz.back();
            z_levels[k] = z + 0.5 * dz_k;
            z += dz_k;
        }
    }
    return z_levels;
}

/**
 * @brief Applies horizontal spatial correlation independently on each z level.
 */
void apply_horizontal_correlation(
    Field3D& noise,
    chaos::CorrelationFilter* correlation_filter,
    double dx,
    double dy
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

    thread_local std::vector<std::vector<double>> slice;
    if (slice.size() != static_cast<size_t>(nr) ||
        (nr > 0 && slice[0].size() != static_cast<size_t>(nth)))
    {
        slice.assign(static_cast<size_t>(nr), std::vector<double>(static_cast<size_t>(nth), 0.0));
    }

    for (int k = 0; k < nz; ++k)
    {
        for (int i = 0; i < nr; ++i)
        {
            for (int j = 0; j < nth; ++j)
            {
                slice[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                    static_cast<double>(noise(i, j, k));
            }
        }
        correlation_filter->apply_2d(slice, dx, dy);
        for (int i = 0; i < nr; ++i)
        {
            for (int j = 0; j < nth; ++j)
            {
                noise(i, j, k) = static_cast<float>(slice[static_cast<size_t>(i)][static_cast<size_t>(j)]);
            }
        }
    }
}

/**
 * @brief Hashes variable names into deterministic RNG stream keys.
 */
uint64_t hash_stream_key(const std::string& name)
{
    uint64_t key = 1469598103934665603ULL;
    for (unsigned char c : name)
    {
        key ^= static_cast<uint64_t>(c);
        key *= 1099511628211ULL;
    }
    return key;
}

}

namespace chaos {

/**
 * @brief Initializes RNG and configured spatial-correlation filter.
 */
void InitialConditionsScheme::initialize(const ChaosConfig& cfg, const GridMetrics& grid) {
    std::cout << "Initialized chaos scheme: initial_conditions" << std::endl;
    std::cout << "  Seed: " << cfg.seed << ", Member: " << cfg.member_id << std::endl;
    std::cout << "  Perturbed variables: ";
    for (const auto& var : cfg.apply_to_ic) {
        std::cout << var << " ";
    }
    std::cout << std::endl;

    rng_ = ChaosRNG(cfg.seed, cfg.member_id);

    correlation_filter_ = create_correlation_filter(cfg.filter_id, cfg.Lx, cfg.Ly);
}

/**
 * @brief Applies stochastic perturbations to configured initial-state fields.
 */
void InitialConditionsScheme::apply_initial_conditions(
    const ChaosConfig& cfg,
    const GridMetrics& grid,
    SimulationState& state,
    ChaosDiagnostics* diag
) {
    auto t0 = std::chrono::high_resolution_clock::now();

    if (!state.valid() || state.theta == nullptr)
    {
        if (diag)
        {
            diag->errors.push_back("IC scheme: invalid simulation state - no perturbations applied");
        }
        return;
    }

    const int nr = state.theta->size_r();
    const int nth = state.theta->size_th();
    const int nz = state.theta->size_z();
    if (nr <= 0 || nth <= 0 || nz <= 0)
    {
        if (diag)
        {
            diag->warnings.push_back("IC scheme: empty state grid - no perturbations applied");
        }
        return;
    }

    std::vector<std::string> vars = cfg.apply_to_ic;
    if (vars.empty())
    {
        vars = {"u", "v", "w", "theta", "qv"};
    }

    const auto z_levels = build_z_levels(grid, nz);
    const double dx = (grid.dx > 0.0) ? grid.dx : 1.0;
    const double dy = (grid.dy > 0.0) ? grid.dy : dx;

    double sum = 0.0;
    double sum_sq = 0.0;
    size_t sample_count = 0;

    for (const std::string& var : vars)
    {
        Field3D* target = resolve_state_field(state, var);
        if (target == nullptr)
        {
            if (diag)
            {
                diag->warnings.push_back("IC scheme: unknown/unavailable variable '" + var + "' - skipped");
            }
            continue;
        }

        double sigma = 0.0;
        auto sigma_it = cfg.sigma_ic.find(var);
        if (sigma_it != cfg.sigma_ic.end())
        {
            sigma = sigma_it->second;
        }
        if (sigma <= 0.0)
        {
            continue;
        }

        Field3D noise = generate_white_noise_field3d(
            rng_,
            nr,
            nth,
            nz,
            hash_stream_key(var),
            var
        );

        apply_horizontal_correlation(noise, correlation_filter_.get(), dx, dy);
        renormalize_to_unit_variance(noise);
        apply_vertical_taper(noise, z_levels, cfg.taper_id, cfg.taper_z1, cfg.taper_z2);
        scale_perturbation_field(noise, sigma);
        bound_perturbation_field(noise, cfg.xi_max * std::max(1.0, sigma), true);

        float* target_values = target->data();
        const float* noise_values = noise.data();
        const std::size_t count = noise.size();
        for (std::size_t idx = 0; idx < count; ++idx)
        {
            float updated = target_values[idx] + noise_values[idx];
            if (var == "theta")
            {
                updated = clamp_theta_k(updated);
            }
            else if (is_moisture_var(var))
            {
                updated = clamp_non_negative(updated);
                if (updated <= 0.0f && diag)
                {
                    diag->negative_moisture_detected = true;
                    ++diag->negative_moisture_count;
                }
            }

            target_values[idx] = updated;
            const double perturb = static_cast<double>(noise_values[idx]);
            sum += perturb;
            sum_sq += perturb * perturb;
            ++sample_count;
        }
    }

    if (diag)
    {
        if (sample_count > 0)
        {
            diag->mean_perturbation = sum / static_cast<double>(sample_count);
            diag->variance_perturbation = (sum_sq / static_cast<double>(sample_count)) -
                                          (diag->mean_perturbation * diag->mean_perturbation);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        diag->time_apply_ic = std::chrono::duration<double>(t1 - t0).count();
    }
}

/**
 * @brief Applies tendency perturbations (unused for IC-only scheme).
 */
void InitialConditionsScheme::apply_tendencies(
    const ChaosConfig& cfg,
    const GridMetrics& grid,
    const ChaosStateView& state_view,
    ChaosTendencies& tendencies,
    ChaosDiagnostics* diag
) {
    if (diag) {
        diag->warnings.push_back("IC scheme: no tendency perturbations applied");
    }
}

/**
 * @brief Advances internal noise state (no-op for IC-only scheme).
 */
void InitialConditionsScheme::step_noise(
    const ChaosConfig& cfg,
    const GridMetrics& grid,
    double dt
) {
}

}
