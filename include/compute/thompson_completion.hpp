#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace nimbus::compute
{
class ThompsonCompletionTables
{
public:
    static constexpr std::size_t                           rain_graupel = 0;
    static constexpr std::size_t                           rain_snow    = 1874161 * 5;
    static constexpr std::size_t                           ice_snow     = rain_snow + 455877 * 12;
    static constexpr std::size_t                           riming       = ice_snow + 3520 * 3;
    static constexpr std::size_t                           water_axis   = riming + 10002;
    static constexpr std::size_t                           rain_axis    = water_axis + 37;
    static constexpr std::size_t                           graupel_axis = rain_axis + 37;
    static constexpr std::size_t                           ice_axis     = graupel_axis + 37;
    static constexpr std::size_t                           number_axis  = ice_axis + 64;
    static constexpr std::size_t                           gamma        = number_axis + 55;
    static constexpr std::size_t                           scalar_count = gamma + 3;
    static constexpr std::size_t                           float_count  = 2 * scalar_count;
    static std::shared_ptr<const ThompsonCompletionTables> create(std::vector<float> values)
    {
        if (values.size() != float_count)
            return {};
        for (std::size_t i = 0; i < values.size(); i += 2)
        {
            if (!std::isfinite(values[i]) || !std::isfinite(values[i + 1]) ||
                static_cast<double>(values[i]) + values[i + 1] < 0)
                return {};
        }
        return std::shared_ptr<const ThompsonCompletionTables>(new ThompsonCompletionTables(std::move(values)));
    }
    const std::vector<float>& values() const noexcept
    {
        return values_;
    }

private:
    explicit ThompsonCompletionTables(std::vector<float> values) : values_(std::move(values))
    {
    }
    std::vector<float> values_;
};

inline constexpr std::size_t thompson_completion_input_stride  = 51;
inline constexpr std::size_t thompson_completion_output_stride = 18;
inline constexpr std::size_t thompson_completion_chunk_cells   = 8192;
struct ThompsonCompletionConstants
{
    int   cells;
    float dt_high;
    float dt_low;
};
static_assert(sizeof(ThompsonCompletionConstants) == 12);
} // namespace nimbus::compute

struct ThompsonCompletionDispatch
{
    const float*                                                     input  = nullptr;
    float*                                                           output = nullptr;
    std::size_t                                                      cells  = 0;
    double                                                           dt     = 0;
    std::shared_ptr<const nimbus::compute::ThompsonCompletionTables> tables;
};
