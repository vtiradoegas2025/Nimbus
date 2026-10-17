#include "microphysics/schemes/thompson/native_freezing_tables.hpp"
#include "microphysics/schemes/thompson/freezing_factors.hpp"
#include <stdexcept>

namespace tmv::thompson
{
std::shared_ptr<const nimbus::compute::ThompsonFreezingTables> native_freezing_tables()
{
    static const auto tables = []()
    {
        const auto         factors = generate_freezing_factors();
        std::vector<float> values;
        values.reserve(nimbus::compute::ThompsonFreezingTables::float_count);
        const auto append = [&](const auto& source)
        {
            for (double value : source)
            {
                const float high = static_cast<float>(value);
                values.push_back(high);
                values.push_back(static_cast<float>(value - high));
            }
        };
        append(factors.water_content);
        append(factors.cloud_number);
        append(factors.rain_mass);
        append(factors.cloud_mass);
        append(factors.rain_bin_number);
        append(factors.cloud_bin_number);
        append(factors.rain_probability);
        append(factors.cloud_probability);
        append(factors.rain_intercept);
        append(factors.ice_nuclei);
        append(std::array<double, 2>{std::sqrt(std::exp(std::log(3000.0) / 100.0)) * 1e6,
                                     std::sqrt(3000.0 * std::exp(99.0 / 100.0 * std::log(3000.0))) * 1e6});
        auto result = nimbus::compute::ThompsonFreezingTables::create(std::move(values));
        if (!result)
            throw std::runtime_error("Thompson freezing factors are not representable");
        return result;
    }();
    return tables;
}
} // namespace tmv::thompson
