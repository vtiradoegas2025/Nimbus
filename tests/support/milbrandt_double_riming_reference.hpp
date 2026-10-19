#pragma once

#include "milbrandt_cold_cloud_diffusion_reference.hpp"

namespace milbrandt_double_riming_reference
{

namespace collection = milbrandt_cold_cloud_collection_reference;
namespace coupled = milbrandt_cold_cloud_diffusion_reference;

struct SnowDiffusion
{
    long double environmental = 0;
    long double heating = 0;
    long double feedback = 0;
};

inline tmv::milbrandt_detail::ColdCloudCollectionCell collection_source(
    const std::array<double, 16>& cell)
{
    tmv::milbrandt_detail::ColdCloudCollectionCell result;
    result.temperature_k = cell[0];
    result.pressure_pa = cell[1];
    result.dry_density_kg_m3 = cell[2];
    result.initial.cloud_mass_per_kg = cell[4];
    result.initial.cloud_number_per_kg = cell[10];
    for (std::size_t index = 0; index < 3; ++index)
    {
        result.initial.frozen[index] = collection::population(index, cell[index + 7], cell[index + 13], 0);
    }
    return result;
}

inline SnowDiffusion snow_diffusion(const std::array<double, 16>& cell,
                                    long double duration, long double admitted_cloud)
{
    SnowDiffusion result;
    const long double temperature = cell[0];
    const long double density = cell[2];
    const long double pressure = cell[1];
    const long double tc = temperature - 273.15L;
    const auto saturation = coupled::saturation(temperature, pressure);
    const long double conductivity = .023971L + .000078L * tc;
    const long double diffusivity = (2.2157e-5L + .0155e-5L * tc) * 1e5L / pressure;
    const long double viscosity = collection::dynamic_viscosity(temperature) / density;
    const long double resistance = coupled::latent_heat_sublimation * coupled::latent_heat_sublimation
        / (conductivity * coupled::vapor_gas_constant * temperature * temperature)
        + 1 / (density * saturation.mass * diffusivity);
    const auto population = collection::population(0, cell[7], cell[13], 0);
    const auto spectrum = collection::frozen::spectrum(population, 1, density);
    const long double ventilation = .78L * collection::frozen::moment(spectrum, 1)
        + .30L * std::cbrt(viscosity / diffusivity)
            * std::sqrt(std::sqrt(1.225L / density) * 11.72L / viscosity)
            * collection::frozen::moment(spectrum, 1.5L + .41L / 2);
    result.environmental = duration * .5L * 2 * std::acos(-1.0L) * cell[13]
        * ventilation / resistance * (static_cast<long double>(cell[3]) / saturation.mass - 1);
    result.feedback = coupled::latent_heat_sublimation * collection::latent_heat_fusion
        / (conductivity * coupled::vapor_gas_constant * temperature * temperature * resistance);
    result.heating = -result.feedback * admitted_cloud;
    return result;
}

} // namespace milbrandt_double_riming_reference
