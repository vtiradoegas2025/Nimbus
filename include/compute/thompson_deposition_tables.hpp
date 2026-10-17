#pragma once

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace nimbus::compute
{
class ThompsonDepositionTables
{
public:
    static constexpr std::size_t fractions         = 3520;
    static constexpr std::size_t ice_contents      = fractions;
    static constexpr std::size_t ice_numbers       = ice_contents + 64;
    static constexpr std::size_t ventilation_gamma = ice_numbers + 55;
    static constexpr std::size_t float_count       = 2 * (ventilation_gamma + 1);

    static std::shared_ptr<const ThompsonDepositionTables> create(const std::vector<double>& fractions_data)
    {
        if (fractions_data.size() != fractions)
            return {};
        std::vector<float> values;
        values.reserve(float_count);
        const auto append = [&](double value)
        {
            const float high = static_cast<float>(value);
            values.push_back(high);
            values.push_back(static_cast<float>(value - high));
        };
        for (double value : fractions_data)
        {
            if (!std::isfinite(value) || value < 0 || value > 1 || (value > 0 && static_cast<float>(value) == 0))
                return {};
            append(value);
        }
        for (int first : {-10, 0})
        {
            const int decades = first == -10 ? 7 : 6;
            for (int decade = 0; decade < decades; ++decade)
            {
                const double scale = std::pow(10.0, first + decade);
                for (int digit = 1; digit <= 9; ++digit)
                    append(digit * scale);
            }
            append(std::pow(10.0, first + decades));
        }
        append(std::tgamma(.5 * (.89 + 5.0)));
        return std::shared_ptr<const ThompsonDepositionTables>(new ThompsonDepositionTables(std::move(values)));
    }
    const std::vector<float>& values() const noexcept
    {
        return values_;
    }

private:
    explicit ThompsonDepositionTables(std::vector<float> values) : values_(std::move(values))
    {
    }
    std::vector<float> values_;
};
} // namespace nimbus::compute
