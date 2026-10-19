#include "util/physics_cadence.hpp"
#include "chaos/runtime_internal.hpp"

#include "boundary_layer/boundary_layer_base.hpp"
#include "chaos/config_contract.hpp"
#include "chaos/factory.hpp"
#include "compute/backend_interface.hpp"
#include "core/runtime/simulation_state.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace tmv::chaos
{

std::unique_ptr<ChaosScheme>& chaos_scheme = global_chaos_runtime_state.scheme;
ChaosConfig& global_chaos_config = global_chaos_runtime_state.config;
ChaosDiagnostics& global_chaos_diagnostics =
    global_chaos_runtime_state.diagnostics;

using runtime_detail::ChaosRuntimeView;
using runtime_detail::chaos_runtime_for;
using runtime_detail::grid_metrics_from_state;
using runtime_detail::invalidate_pbl_base_tendencies;
using runtime_detail::normalize_tendency_block_name;

namespace
{

void canonicalize_tendency_blocks(ChaosConfig& cfg)
{
    std::vector<std::string> canonical_blocks;
    canonical_blocks.reserve(cfg.apply_to_tendencies.size());
    std::unordered_set<std::string> seen;
    for (const std::string& raw_block : cfg.apply_to_tendencies)
    {
        const std::string canonical = normalize_tendency_block_name(raw_block);
        if (canonical != "microphysics" && canonical != "pbl" &&
            canonical != "diffusion")
        {
            throw std::invalid_argument(
                "Unknown chaos tendency block '" + raw_block +
                "'. Supported blocks: microphysics, pbl, diffusion");
        }
        if (seen.insert(canonical).second)
        {
            canonical_blocks.push_back(canonical);
        }
    }
    cfg.apply_to_tendencies = std::move(canonical_blocks);

    std::unordered_map<std::string, double> canonical_alpha;
    canonical_alpha.reserve(cfg.alpha_tend.size());
    for (const auto& kv : cfg.alpha_tend)
    {
        const std::string canonical = normalize_tendency_block_name(kv.first);
        if (canonical != "microphysics" && canonical != "pbl" &&
            canonical != "diffusion")
        {
            throw std::invalid_argument(
                "Unknown chaos alpha_tend block '" + kv.first + "'");
        }
        if (!std::isfinite(kv.second) || kv.second < 0.0)
        {
            throw std::invalid_argument(
                "Chaos alpha_tend." + kv.first + " must be non-negative and finite");
        }
        canonical_alpha[canonical] = kv.second;
    }
    cfg.alpha_tend = std::move(canonical_alpha);
}

void reset_noise_cadence(ChaosRuntimeView& runtime)
{
    runtime.noise_elapsed_s = 0.0;
}

} // namespace

void bind_chaos_compute_backend(const SimulationStateView& state)
{
    auto runtime = chaos_runtime_for(state);
    if (!runtime.scheme) throw std::runtime_error("chaos scheme is not initialized");
    ComputeBackend* chaos_backend = nullptr;
    if (runtime.config.native_generation_required && state.compute_runtime != nullptr)
        chaos_backend = state.compute_runtime->backend.get();
    if (runtime.config.native_generation_required &&
        (chaos_backend == nullptr || !chaos_backend->supports_earth_control_evolution()))
        throw std::runtime_error(
            "native chaos generation is required but the active backend does not provide Earth-control evolution");
    runtime.scheme->configure_compute_backend(
        chaos_backend, runtime.config.native_generation_required);
}

void initialize_chaos(const SimulationStateView& state, const ChaosConfig& cfg)
{
    try {
        ChaosRuntimeView runtime = chaos_runtime_for(state);
        runtime.config = cfg;
        if (runtime.config.scheme_id.empty())
        {
            runtime.config.scheme_id = "none";
        }
        std::transform(
            runtime.config.scheme_id.begin(),
            runtime.config.scheme_id.end(),
            runtime.config.scheme_id.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        runtime.scheme = create_chaos_scheme(runtime.config.scheme_id);
        runtime.config.scheme_id = runtime.scheme->name();
        bind_chaos_compute_backend(state);
        apply_canonical_chaos_target_defaults(runtime.config);
        canonicalize_tendency_blocks(runtime.config);
        if (!tmv::valid_physics_cadence(runtime.config.dt_chaos))
        {
            throw std::invalid_argument("Chaos dt_chaos must exceed the 1e-12 s scheduling tolerance");
        }
        if (!std::isfinite(runtime.config.tau_t) || runtime.config.tau_t < 0.0)
        {
            throw std::invalid_argument("Chaos tau_t must be non-negative and finite");
        }
        if (!std::isfinite(runtime.config.Lx) || runtime.config.Lx <= 0.0 ||
            !std::isfinite(runtime.config.Ly) || runtime.config.Ly <= 0.0)
        {
            throw std::invalid_argument(
                "Chaos correlation lengths Lx and Ly must be positive and finite");
        }
        if (!std::isfinite(runtime.config.xi_max) || runtime.config.xi_max <= 0.0)
        {
            throw std::invalid_argument("Chaos xi_max must be positive and finite");
        }
        if (!std::isfinite(runtime.config.taper_z1))
        {
            throw std::invalid_argument("Chaos taper_z1 must be finite");
        }
        if (!std::isfinite(runtime.config.taper_z2))
        {
            throw std::invalid_argument("Chaos taper_z2 must be finite");
        }
        if (runtime.config.taper_z2 <= runtime.config.taper_z1)
        {
            throw std::invalid_argument("Chaos taper_z2 must be greater than taper_z1");
        }
        if (runtime.config.taper_id != "none" &&
            runtime.config.taper_id != "pbl_only" &&
            runtime.config.taper_id != "cosine")
        {
            throw std::invalid_argument(
                "Unknown chaos taper_id '" + runtime.config.taper_id +
                "'. Supported tapers: none, pbl_only, cosine");
        }
        static const std::unordered_set<std::string> valid_ic_variables = {
            "u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh"};
        for (const auto& variable : runtime.config.apply_to_ic)
        {
            if (!valid_ic_variables.count(variable))
            {
                throw std::invalid_argument(
                    "Unknown chaos initial-condition variable '" + variable + "'");
            }
        }
        for (const auto& kv : runtime.config.sigma_ic)
        {
            if (!valid_ic_variables.count(kv.first))
            {
                throw std::invalid_argument(
                    "Unknown chaos sigma_ic variable '" + kv.first + "'");
            }
            if (!std::isfinite(kv.second) || kv.second < 0.0)
            {
                throw std::invalid_argument(
                    "Chaos sigma_ic." + kv.first + " must be non-negative and finite");
            }
        }
        reset_noise_cadence(runtime);
        invalidate_pbl_base_tendencies(runtime);
        runtime.diagnostics = ChaosDiagnostics{};

        const GridMetrics& init_grid = grid_metrics_from_state(state);
        runtime.scheme->initialize(runtime.config, init_grid);

        tmv::log_info("Initialized chaos scheme: ", runtime.config.scheme_id);
        if (runtime.config.scheme_id == "boundary_layer")
        {
            tmv::log_info("[CHAOS NOTE] 'boundary_layer' here means stochastic PBL tendency perturbations.\n"
                          "  It does not replace the physical PBL solver (boundary_layer.scheme=",
                          tmv::require_level_runtime(state.boundary_layer_runtime, "boundary_layer_runtime").config.scheme_id, ").");
        }

        if (runtime.config.scheme_id != "none")
        {
            tmv::log_info("  Perturbation amplitudes:");
            if (!runtime.config.apply_to_ic.empty())
            {
                std::ostringstream oss;
                oss << "    IC variables: ";
                for (const auto& var : runtime.config.apply_to_ic)
                {
                    auto it = runtime.config.sigma_ic.find(var);
                    if (it != runtime.config.sigma_ic.end())
                    {
                        oss << var << "(σ=" << it->second << ") ";
                    }
                }
                tmv::log_info(oss.str());
            }
            if (!runtime.config.apply_to_tendencies.empty())
            {
                std::ostringstream oss;
                oss << "    Tendency blocks: ";
                for (const auto& block : runtime.config.apply_to_tendencies)
                {
                    auto it = runtime.config.alpha_tend.find(block);
                    if (it != runtime.config.alpha_tend.end())
                    {
                        oss << block << "(α=" << it->second << ") ";
                    }
                }
                tmv::log_info(oss.str());
            }
            tmv::log_info("  Correlation: Lx=", runtime.config.Lx, "m, Ly=", runtime.config.Ly, "m");
            tmv::log_info("  Temporal decorrelation: τ=", runtime.config.tau_t, "s");
        }

    } catch (const std::exception& e) {
        tmv::log_error("Error initializing chaos: ", e.what());
        throw;
    }
}

void initialize_chaos(const ChaosConfig& cfg)
{
    initialize_chaos(global_simulation_state(), cfg);
}


} // namespace tmv::chaos
