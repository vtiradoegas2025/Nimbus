// Spatially correlated stochastic initial-condition perturbations.

#include "initial_conditions.hpp"
#include "chaos/base/earth_correlated_noise.hpp"
#include "chaos/base/perturbation_field.hpp"
#include "chaos/base/correlation_filter.hpp"
#include "core/field/field_bounds.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "turbulence/turbulence_base.hpp"
#include "util/log.hpp"

#include <chrono>
#include <cmath>
#include <sstream>

namespace {

Field3D* resolve_state_field(SimulationState& state, const std::string& var)
{
    if (var == "u") return state.u;
    if (var == "v" || var == "v") return state.v;
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

bool is_moisture_var(const std::string& var)
{
    return var == "qv" || var == "qc" || var == "qr" ||
           var == "qi" || var == "qs" || var == "qg" || var == "qh";
}

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

namespace tmv::chaos
{

void InitialConditionsScheme::configure_compute_backend(
    ComputeBackend* backend, bool required) noexcept
{
    native_backend_ = backend;
    native_generation_required_ = required;
}

void InitialConditionsScheme::initialize(const ChaosConfig& cfg, const GridMetrics& grid) {
    tmv::log_info("Initialized chaos scheme: initial_conditions");
    tmv::log_info("  Seed: ", cfg.seed, ", Member: ", cfg.member_id);
    std::ostringstream variables;
    for (const auto& var : cfg.apply_to_ic) {
        variables << var << " ";
    }
    tmv::log_info("  Perturbed variables: ", variables.str());

    (void)create_correlation_filter(cfg.filter_id, cfg.Lx, cfg.Ly);
    tmv::log_info("  Spatial noise: ", EarthCorrelatedNoise::algorithm_id(),
                  " (legacy filter selection ", cfg.filter_id, ")");
}

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
        vars = {"u", "v", "w", "theta"};
    }

    const double dx = grid.dx;
    const double dy = grid.dy;

    double sum = 0.0;
    double sum_sq = 0.0;
    size_t sample_count = 0;
    const PhysicalCellDomain domain = physical_scalar_domain(
        nr, nth, nz, grid.staggering);
    if (domain.empty())
    {
        throw std::invalid_argument(
            "IC chaos has no physical cells for the selected staggering");
    }

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

        const std::uint64_t process_key = hash_stream_key(var);
        EarthCorrelatedNoise earth_noise;
        earth_noise.configure_native_evolution(
            native_backend_, native_generation_required_);
        earth_noise.configure(
            cfg.seed, cfg.member_id, process_key,
            process_key ^ 0xa0761d6478bd642fULL,
            cfg.Lx, cfg.Ly, dx, dy);
        earth_noise.initialize_domain(nr, nth, nz);
        Field3D noise;
        earth_noise.render_current(noise);
        apply_vertical_taper(noise, grid, cfg.taper_id, cfg.taper_z1, cfg.taper_z2);
        scale_perturbation_field(noise, sigma);
        bound_perturbation_field(noise, cfg.xi_max * std::max(1.0, sigma), true);

        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const float perturb_value = noise(i, j, k);
                    float updated = (*target)(i, j, k) + perturb_value;
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

                    (*target)(i, j, k) = updated;
                    const double perturb =
                        static_cast<double>(perturb_value);
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
