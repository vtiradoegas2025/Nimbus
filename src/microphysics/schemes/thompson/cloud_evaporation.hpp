#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace tmv::thompson
{

struct CloudEvaporationMetadata
{
    static constexpr std::size_t diameters = 100;
    static constexpr std::size_t cloud_contents = 37;
    static constexpr std::size_t cloud_numbers = 100;
    static constexpr const char* upstream_release = "WRF v4.7.1";
    static constexpr const char* upstream_commit = "f52c197";
    static constexpr const char* upstream_source_sha256 =
        "fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802";
};

struct CloudEvaporationTable
{
    std::vector<double> evaporated_mass_m3;
    std::vector<double> evaporated_number_m3;
};

struct CloudEvaporationNumberExchange
{
    double cloud_number_kg_dry = 0.0;
    double water_aerosol_kg_dry = 0.0;
};

std::size_t cloud_evaporation_table_size();
CloudEvaporationTable generate_cloud_evaporation_table();
bool validate_cloud_evaporation_table(const CloudEvaporationTable& table,
                                      std::string& error);
std::size_t cloud_evaporation_index(std::size_t diameter,
                                    std::size_t cloud_content,
                                    std::size_t cloud_number);
std::size_t cloud_evaporation_diameter_index(double diameter_m);
std::size_t cloud_evaporation_content_index(double cloud_water_kg_m3);
std::size_t cloud_evaporation_number_index(double cloud_number_m3);
CloudEvaporationNumberExchange return_evaporated_cloud_number(
    const CloudEvaporationTable& table,
    double maximum_evaporated_diameter_m,
    double cloud_water_kg_m3,
    double cloud_number_kg_dry,
    double dry_air_density_kg_m3);
double maximum_fully_evaporated_cloud_diameter(
    double temperature_k,
    double pressure_pa,
    double dry_air_density_kg_m3,
    double vapor_kg_dry,
    double saturation_vapor_kg_dry,
    double dt_s);

} // namespace tmv::thompson
