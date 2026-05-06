/**
 * @file warm_bubble.cpp
 * @brief WarmBubbleTrigger: thin polymorphic wrapper around the existing
 *        Gaussian-bubble helpers in initial_conditions_*.cpp.
 */

#include "init/trigger/warm_bubble.hpp"

#include "core/infra/coordinate_system.hpp"
#include "core/orchestration/dynamics/initial_conditions.hpp"

extern CoordinateSystem global_coordinate_system;

namespace tmv::init
{

void WarmBubbleTrigger::apply() const
{
    if (global_coordinate_system == CoordinateSystem::Cartesian)
    {
        apply_cartesian_bubble_initialization();
    }
    else
    {
        apply_cylindrical_bubble_initialization();
    }
}

std::string WarmBubbleTrigger::describe() const
{
    return "warm_bubble";
}

}  // namespace tmv::init
