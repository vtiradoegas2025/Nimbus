#pragma once
#include "compute/thompson_completion.hpp"
#include "compute/thompson_cloud_contract.hpp"
namespace nimbus::compute::thompson_completion
{
inline bool validate(const ThompsonCompletionDispatch& d)
{
    if (!d.tables || !d.input || !d.output || d.cells == 0 || d.cells > thompson_completion_chunk_cells ||
        !std::isfinite(d.dt) || d.dt <= 0 || !std::isfinite(static_cast<float>(d.dt)) || static_cast<float>(d.dt) <= 0)
        return false;
    if (thompson_cloud::overlap(d.input, d.cells * thompson_completion_input_stride, d.output, d.cells * 18) ||
        thompson_cloud::overlap(d.tables->values().data(), d.tables->values().size(), d.output, d.cells * 18))
        return false;
    for (std::size_t n = 0; n < d.cells; ++n)
    {
        for (std::size_t f = 0; f < thompson_completion_input_stride; ++f)
            if (!std::isfinite(d.input[n * thompson_completion_input_stride + f]) ||
                (f < 15 && d.input[n * thompson_completion_input_stride + f] < 0))
                return false;
        if (d.input[n * thompson_completion_input_stride] <= 0 ||
            d.input[n * thompson_completion_input_stride + 1] <= 0 ||
            d.input[n * thompson_completion_input_stride + 2] <= 0)
            return false;
    }
    return true;
}
inline bool valid_candidate(const ThompsonCompletionDispatch& d, const float* output)
{
    for (std::size_t n = 0; n < d.cells; ++n)
    {
        for (std::size_t f = 0; f < 18; ++f)
            if (!std::isfinite(output[n * 18 + f]))
                return false;
        if (output[n * 18 + 17] != 0 || output[n * 18 + 13] < 1 || output[n * 18 + 13] > 1.5)
            return false;
        for (std::size_t f = 0; f < 5; ++f)
            if (d.input[n * thompson_completion_input_stride + 10 + f] + d.dt * output[n * 18 + 8 + f] < 0)
                return false;
        double before = 0, change = 0;
        for (std::size_t f = 0; f < 7; ++f)
        {
            const double value = d.input[n * thompson_completion_input_stride + 3 + f],
                         delta = d.dt * output[n * 18 + 1 + f];
            if (value + delta < -2e-7 * (value + std::abs(delta)) - 2e-12)
                return false;
            before += value;
            change += delta;
        }
        if (std::abs(change) > 2e-6 * before + 2e-12)
            return false;
    }
    return true;
}
inline ThompsonCompletionConstants constants(const ThompsonCompletionDispatch& d)
{
    return {static_cast<int>(d.cells), static_cast<float>(d.dt), thompson_cloud::timestep_low(d.dt)};
}
} // namespace nimbus::compute::thompson_completion
