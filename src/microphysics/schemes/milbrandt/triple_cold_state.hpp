#pragma once

#include "triple_frozen_collection.hpp"

namespace tmv::milbrandt_detail
{

struct ColdRimingState
{
    double vapor_mass_per_kg = 0.0;
    double cloud_mass_per_kg = 0.0;
    double cloud_number_per_kg = 0.0;
    std::array<FrozenCollectionPopulation, 4> frozen{};
    FrozenCollectionPopulation rain;
};

struct ColdRimingCell
{
    double temperature_k = 0.0;
    double pressure_pa = 0.0;
    double dry_density_kg_m3 = 0.0;
    ColdRimingState initial;
};

} // namespace tmv::milbrandt_detail
