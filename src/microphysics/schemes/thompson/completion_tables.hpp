#pragma once
#include "compute/thompson_completion.hpp"
namespace tmv::thompson
{
std::shared_ptr<const nimbus::compute::ThompsonCompletionTables> native_completion_tables();
}
