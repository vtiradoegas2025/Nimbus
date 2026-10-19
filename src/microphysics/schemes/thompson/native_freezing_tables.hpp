#pragma once
#include "compute/thompson_freezing_tables.hpp"
namespace tmv::thompson
{
std::shared_ptr<const nimbus::compute::ThompsonFreezingTables> native_freezing_tables();
}
