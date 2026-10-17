/**
 * @file none.cpp
 * @brief NoOpTrigger: applies nothing.
 */

#include "init/trigger/none.hpp"

namespace tmv::init
{

void NoOpTrigger::apply() const
{
}

void NoOpTrigger::apply(const SimulationStateView& /*state*/) const
{
}

std::string NoOpTrigger::describe() const
{
    return "none";
}

}  // namespace tmv::init
