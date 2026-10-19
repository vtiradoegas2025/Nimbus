#pragma once

#include "compute/thompson_cloud_tables.hpp"
#include "microphysics/schemes/thompson/activation_table.hpp"
#include "microphysics/schemes/thompson/cloud_evaporation.hpp"

namespace tmv::thompson
{
std::shared_ptr<const nimbus::compute::ThompsonCloudTables>
pack_native_cloud_tables(const ActivationTable& activation, const CloudEvaporationTable& evaporation);
}
