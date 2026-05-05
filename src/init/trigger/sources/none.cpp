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

std::string NoOpTrigger::describe() const
{
    return "none";
}

}  // namespace tmv::init
