#pragma once

#include <cmath>
#include <array>
#include <memory>
#include <utility>
#include <vector>

namespace nimbus::compute
{
class ThompsonCloudTables
{
public:
    static constexpr std::size_t activation_count  = 8820;
    static constexpr std::size_t evaporation_count = 370000;
    static constexpr std::size_t edge_count        = 137;

    static std::shared_ptr<const ThompsonCloudTables>
    create(std::vector<float> activation, std::vector<float> evaporation, const std::array<double, edge_count>& edges)
    {
        if (activation.size() != activation_count || evaporation.size() != evaporation_count)
        {
            return {};
        }
        for (float value : activation)
        {
            if (!std::isfinite(value) || value < 0 || value > 1)
            {
                return {};
            }
        }
        for (float value : evaporation)
        {
            if (!std::isfinite(value) || value < 0 || value > 3e9f)
            {
                return {};
            }
        }
        for (std::size_t i = 0; i < edges.size(); ++i)
        {
            if (!std::isfinite(edges[i]) || edges[i] <= 0 || edges[i] > 3e9 ||
                (i != 0 && i != 37 && edges[i] <= edges[i - 1]))
            {
                return {};
            }
            const float high = static_cast<float>(edges[i]);
            if (high <= 0)
            {
                return {};
            }
            evaporation.push_back(high);
            evaporation.push_back(static_cast<float>(edges[i] - high));
        }
        return std::shared_ptr<const ThompsonCloudTables>(
            new ThompsonCloudTables(std::move(activation), std::move(evaporation)));
    }

    const std::vector<float>& activation() const noexcept
    {
        return activation_;
    }

    const std::vector<float>& evaporation() const noexcept
    {
        return evaporation_;
    }

private:
    ThompsonCloudTables(std::vector<float> activation, std::vector<float> evaporation)
        : activation_(std::move(activation)), evaporation_(std::move(evaporation))
    {
    }

    std::vector<float> activation_;
    std::vector<float> evaporation_;
};
} // namespace nimbus::compute
