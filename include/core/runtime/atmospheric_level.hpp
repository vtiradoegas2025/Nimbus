#pragma once

#include "core/runtime/simulation_state.hpp"

#include <tuple>
#include <type_traits>

namespace tmv::state_ownership
{
inline auto& frame(SimulationStorageState& state) { return state.configuration.domain_frame; }
inline const auto& frame(const SimulationStorageState& state) { return state.configuration.domain_frame; }
inline auto& frame(AtmosphericLevelStorage& state) { return state.domain_frame; }
inline const auto& frame(const AtmosphericLevelStorage& state) { return state.domain_frame; }

template <class State>
auto level_members(State& state)
{
    using Runtime = std::conditional_t<std::is_const_v<State>,
        const AtmosphericLevelRuntimeState, AtmosphericLevelRuntimeState>;
    return std::tie(state.grid, state.process_grid_metrics, state.fields, state.base,
        state.tendencies, state.surface, state.diagnostics,
        static_cast<Runtime&>(state.runtime), frame(state), state.coriolis_f, state.time_s);
}

template <class Destination, class Source, class Function>
void visit_level_pair(Destination& destination, Source& source, Function&& function)
{
    auto targets = level_members(destination);
    auto sources = level_members(source);
    std::apply([&](auto&... target) {
        std::apply([&](auto&... input) { (function(target, input), ...); }, sources);
    }, targets);
}

template <class Left, class Right>
void exchange_levels(Left& left, Right& right) noexcept
{
    visit_level_pair(left, right, [](auto& target, auto& source) {
        using std::swap;
        static_assert(std::is_nothrow_swappable_v<std::remove_reference_t<decltype(target)>>);
        swap(target, source);
    });
}

inline void capture_level(AtmosphericLevelStorage& target, const AtmosphericLevelStorage& source)
{
    visit_level_pair(target, source, [](auto& destination, const auto& input) { destination = input; });
}

inline void capture_level(AtmosphericLevelStorage& target, const SimulationStorageState& source)
{
    visit_level_pair(target, source, [](auto& destination, const auto& input) { destination = input; });
}
}
