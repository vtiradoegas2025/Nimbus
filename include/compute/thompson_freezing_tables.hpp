#pragma once

#include <cmath>
#include <memory>
#include <limits>
#include <utility>
#include <vector>

namespace nimbus::compute
{
class ThompsonFreezingTables
{
public:
    static constexpr std::size_t water                = 0;
    static constexpr std::size_t cloud_number         = 37;
    static constexpr std::size_t rain_mass            = 137;
    static constexpr std::size_t cloud_mass           = 237;
    static constexpr std::size_t rain_bins            = 337;
    static constexpr std::size_t cloud_bins           = rain_bins + 37 * 37 * 100;
    static constexpr std::size_t rain_probability     = cloud_bins + 100 * 37 * 100;
    static constexpr std::size_t cloud_probability    = rain_probability + 55 * 45 * 100;
    static constexpr std::size_t rain_intercepts      = cloud_probability + 55 * 45 * 100;
    static constexpr std::size_t ice_nuclei           = rain_intercepts + 37;
    static constexpr std::size_t cloud_lookup_centers = ice_nuclei + 55;
    static constexpr std::size_t scalar_count         = cloud_lookup_centers + 2;
    static constexpr std::size_t float_count          = 2 * scalar_count;

    static std::shared_ptr<const ThompsonFreezingTables> create(std::vector<float> values)
    {
        if (values.size() != float_count)
            return {};
        for (std::size_t i = 0; i < scalar_count; ++i)
        {
            const float  high = values[2 * i], low = values[2 * i + 1];
            const double value = static_cast<double>(high) + low;
            const float  direction =
                low < 0 ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
            float neighbor = std::nextafter(high, direction);
            if (!std::isfinite(neighbor))
                neighbor = std::nextafter(high, -direction);
            if (std::abs(static_cast<double>(low)) > .5 * std::abs(static_cast<double>(neighbor) - high))
            {
                return {};
            }
            if (!std::isfinite(high) || !std::isfinite(low) || value < 0 ||
                ((i < rain_bins || i >= rain_intercepts) && value <= 0) ||
                (i >= rain_probability && i < rain_intercepts && value > 1))
                return {};
            if ((i > water && i < cloud_number) || (i > cloud_number && i < rain_mass) ||
                (i > rain_mass && i < cloud_mass) || (i > cloud_mass && i < rain_bins) ||
                (i > rain_intercepts && i < ice_nuclei) || (i > ice_nuclei && i < cloud_lookup_centers) ||
                i == cloud_lookup_centers + 1)
            {
                if (value <= static_cast<double>(values[2 * i - 2]) + values[2 * i - 1])
                    return {};
            }
        }
        return std::shared_ptr<const ThompsonFreezingTables>(new ThompsonFreezingTables(std::move(values)));
    }
    const std::vector<float>& values() const noexcept
    {
        return values_;
    }

private:
    explicit ThompsonFreezingTables(std::vector<float> values) : values_(std::move(values))
    {
    }
    std::vector<float> values_;
};
} // namespace nimbus::compute
