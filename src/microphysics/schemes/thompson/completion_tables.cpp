#include "microphysics/schemes/thompson/completion_tables.hpp"
#include "microphysics/schemes/thompson/cold_processes.hpp"
#include "microphysics/schemes/thompson/required_packaged_data.hpp"
#include <stdexcept>
namespace tmv::thompson
{
std::shared_ptr<const nimbus::compute::ThompsonCompletionTables> native_completion_tables()
{
    static const auto tables = []()
    {
        const auto         collection = require_packaged_collection_tables();
        const auto         cold       = require_packaged_cold_process_tables();
        std::vector<float> values;
        values.reserve(nimbus::compute::ThompsonCompletionTables::float_count);
        const auto append = [&](double v)
        {
            const float high = static_cast<float>(v);
            values.push_back(high);
            values.push_back(static_cast<float>(v - high));
        };
        for (const auto& c : collection->rain_graupel)
            for (double v : {c.graupel_mass_by_rain,
                             c.rain_mass_by_rain,
                             c.rain_mass_by_graupel,
                             c.graupel_number_by_rain,
                             c.rain_number_by_graupel})
                append(v);
        for (const auto& c : collection->rain_snow)
            for (double v : c.values)
                append(v);
        for (const auto& c : cold->ice_snow)
            for (double v : {c.snow_mass, c.ice_number, c.ice_deposition_fraction})
                append(v);
        for (double v : native_riming_coefficients())
            append(v);
        for (const auto axis : std::vector<std::pair<int, int>>{{-6, 37}, {6, 37}, {2, 37}, {-10, 64}, {0, 55}})
        {
            const int decades = (axis.second - 1) / 9;
            for (int decade = 0; decade < decades; ++decade)
                for (int digit = 1; digit <= 9; ++digit)
                    append(digit * std::pow(10.0, axis.first + decade));
            append(std::pow(10.0, axis.first + decades));
        }
        append(std::tgamma(.5 * (.89 + 5.0)));
        append(std::tgamma(3.89));
        append(std::tgamma(4.89));
        auto result = nimbus::compute::ThompsonCompletionTables::create(std::move(values));
        if (!result)
            throw std::runtime_error("Thompson completion coefficients are not representable");
        return result;
    }();
    return tables;
}
} // namespace tmv::thompson
