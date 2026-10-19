#include "microphysics/schemes/thompson/native_cloud_tables.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tmv::thompson
{
namespace
{
template <class Locate>
double first_in_bin(double value, std::size_t bin, Locate locate)
{
    for (int step = 0; step < 64; ++step)
    {
        const auto index = locate(value);
        if (index < bin)
        {
            value = std::nextafter(value, std::numeric_limits<double>::infinity());
        }
        else
        {
            const double previous = std::nextafter(value, 0.0);
            if (locate(previous) < bin)
            {
                return value;
            }
            value = previous;
        }
    }
    throw std::runtime_error("Thompson native lookup edge is not representable");
}
} // namespace

std::shared_ptr<const nimbus::compute::ThompsonCloudTables>
pack_native_cloud_tables(const ActivationTable& activation, const CloudEvaporationTable& evaporation)
{
    std::string error;
    if (!validate_activation_table(activation, error) || !validate_cloud_evaporation_table(evaporation, error))
    {
        throw std::invalid_argument(error);
    }
    if (std::any_of(evaporation.evaporated_number_m3.begin(),
                    evaporation.evaporated_number_m3.end(),
                    [](double value) { return value > 3e9; }))
    {
        throw std::invalid_argument("Thompson evaporation number exceeds the table grid");
    }
    using Tables = nimbus::compute::ThompsonCloudTables;
    static_assert(Tables::activation_count == ActivationTableMetadata::element_count);
    static_assert(Tables::evaporation_count == CloudEvaporationMetadata::diameters *
                                                   CloudEvaporationMetadata::cloud_contents *
                                                   CloudEvaporationMetadata::cloud_numbers);
    std::array<double, Tables::edge_count> edges;
    edges[0] = 1e-6;
    for (std::size_t bin = 1; bin < 37; ++bin)
    {
        const double value = (1 + bin % 9) * std::pow(10.0, -6.0 + bin / 9);
        edges[bin]         = first_in_bin(value, bin, cloud_evaporation_content_index);
    }
    const double first = std::exp(std::log(3000.0) * 0.005) * 1e6;
    const double last  = std::exp(std::log(3000.0) * 0.995) * 1e6;
    const double span  = std::log(last / first);
    edges[37]          = first;
    for (std::size_t bin = 1; bin < 100; ++bin)
    {
        const double value = first * std::exp((bin - 0.5) * span / 100.0);
        edges[37 + bin]    = first_in_bin(value, bin, cloud_evaporation_number_index);
    }
    auto result = Tables::create(
        activation.fractions,
        std::vector<float>(evaporation.evaporated_number_m3.begin(), evaporation.evaporated_number_m3.end()),
        edges);
    if (!result)
    {
        throw std::runtime_error("Thompson native cloud tables are not representable");
    }
    return result;
}
} // namespace tmv::thompson
