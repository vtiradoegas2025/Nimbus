#pragma once
#include "compute/thompson_deposition_tables.hpp"
namespace tmv::thompson
{
std::shared_ptr<const nimbus::compute::ThompsonDepositionTables> native_deposition_tables();
}
