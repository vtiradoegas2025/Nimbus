/**
 * @file warm_bubble.cpp
 * @brief WarmBubbleTrigger: thin polymorphic wrapper around the existing
 *        Gaussian-bubble helper in initial_conditions_*.cpp.
 */

#include "init/trigger/warm_bubble.hpp"

#include "core/orchestration/dynamics/initial_conditions.hpp"

namespace tmv::init
{

void WarmBubbleTrigger::apply() const
{
    apply_cartesian_bubble_initialization();
}

void WarmBubbleTrigger::apply(const SimulationStateView& state) const
{
    apply_cartesian_bubble_initialization(state);
}

std::string WarmBubbleTrigger::describe() const
{
    return "warm_bubble";
}

}  // namespace tmv::init
