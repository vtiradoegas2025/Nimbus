#pragma once

#include "chaos/config.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace tmv::chaos
{

inline const std::vector<std::string>&
canonical_chaos_initial_condition_targets(std::string_view scheme_id)
{
    static const std::vector<std::string> empty;
    static const std::vector<std::string> initial_conditions = {
        "u", "v", "w", "theta",
    };
    return scheme_id == "initial_conditions" || scheme_id == "full_stochastic"
        ? initial_conditions : empty;
}

inline const std::vector<std::string>&
canonical_chaos_tendency_targets(std::string_view scheme_id)
{
    static const std::vector<std::string> empty;
    static const std::vector<std::string> boundary_layer = {"pbl"};
    static const std::vector<std::string> full_stochastic = {
        "microphysics", "pbl", "diffusion",
    };
    if (scheme_id == "boundary_layer") return boundary_layer;
    if (scheme_id == "full_stochastic") return full_stochastic;
    return empty;
}

inline bool chaos_targets_are_canonical_or_implicit(
    std::string_view scheme_id,
    const std::vector<std::string>& initial_condition_targets,
    const std::vector<std::string>& tendency_targets)
{
    const auto& canonical_initial =
        canonical_chaos_initial_condition_targets(scheme_id);
    const auto& canonical_tendencies =
        canonical_chaos_tendency_targets(scheme_id);
    return (initial_condition_targets.empty() ||
            initial_condition_targets == canonical_initial) &&
        (tendency_targets.empty() ||
         tendency_targets == canonical_tendencies);
}

inline void apply_canonical_chaos_target_defaults(ChaosConfig& config)
{
    if (config.apply_to_ic.empty())
    {
        config.apply_to_ic =
            canonical_chaos_initial_condition_targets(config.scheme_id);
    }
    if (config.apply_to_tendencies.empty())
    {
        config.apply_to_tendencies =
            canonical_chaos_tendency_targets(config.scheme_id);
    }
    if (config.scheme_id == "initial_conditions")
    {
        config.apply_to_tendencies.clear();
    }
}

} // namespace tmv::chaos
