#include "core/runtime/column_remap_internal.hpp"
#include "compute/dispatch/terrain.hpp"
#include "compute/column_remap_contract.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace tmv::held_remap_detail
{
namespace
{
void pack(double value, float& high, float& low)
{
    high = static_cast<float>(value);
    low  = static_cast<float>(value - high);
    if (!std::isfinite(high) || !std::isfinite(low) ||
        (value != 0 && std::abs(high) < std::numeric_limits<float>::min()))
        throw std::overflow_error("native held-column input exceeds compensated storage");
}
} // namespace
Columns remap_native(TerrainComputeKernels& kernels,
                     const Columns&         source,
                     int                    nx,
                     int                    ny,
                     int                    levels,
                     double                 sx,
                     double                 sy,
                     Storage                storage)
{
    if (!kernels.supports_column_remap())
        throw std::runtime_error("native held-column kernels are unavailable");
    const auto count = source.values[0].size();
    if (nx < 1 || ny < 1 || levels < 1 ||
        std::size_t(nx) > std::size_t(std::numeric_limits<int>::max()) / ColumnRemapDispatch::output_stride /
                              std::size_t(ny) / std::size_t(levels) ||
        count != std::size_t(nx) * ny * levels)
        throw std::invalid_argument("native held columns exceed dispatch limits");
    ColumnRemapDispatch d;
    auto&               p = d.constants;
    p.nx                  = nx;
    p.ny                  = ny;
    p.nz                  = levels;
    p.mode                = int(storage);
    pack(sx, p.sx_high, p.sx_low);
    pack(sy, p.sy_high, p.sy_low);
    std::vector<float> input(count * ColumnRemapDispatch::input_stride),
        output(count * ColumnRemapDispatch::output_stride);
    for (int channel = 0; channel < (storage == Storage::Slab ? 3 : 1); ++channel)
    {
        if (source.values[channel].size() != count)
            throw std::invalid_argument("native held-column channels differ in shape");
        for (std::size_t n = 0; n < count; ++n)
        {
            const double value = source.values[channel][n];
            if (!std::isfinite(value))
                throw std::invalid_argument("native held-column source is not finite");
            int          exponent = 0;
            const double fraction = std::frexp(value, &exponent);
            pack(fraction, input[9 * n + 2 * channel], input[9 * n + 2 * channel + 1]);
            input[9 * n + 6 + channel] = static_cast<float>(exponent);
        }
    }
    d.input  = input.data();
    d.output = output.data();
    if (!nimbus::compute::column_remap::validate(d))
        throw std::invalid_argument("native held columns violate the dispatch contract");
    if (!kernels.dispatch_column_remap(d) || !nimbus::compute::column_remap::finite_candidate(output.data(), d))
        throw std::runtime_error("native held-column dispatch failed");
    Columns result;
    for (int channel = 0; channel < (storage == Storage::Slab ? 3 : 1); ++channel)
    {
        result.values[channel].resize(count);
        for (std::size_t n = 0; n < count; ++n)
        {
            const double value =
                sx == 0 && sy == 0 ? source.values[channel][n]
                                   : std::ldexp(double(output[11 * n + 2 * channel]) + output[11 * n + 2 * channel + 1],
                                                static_cast<int>(output[11 * n + 8 + channel]));
            if (!std::isfinite(value))
                throw std::overflow_error("native held-column result exceeds host storage");
            result.values[channel][n] = value;
        }
    }
    if (storage == Storage::Float || storage == Storage::Rain)
    {
        result.rounded.resize(count);
        result.residual.resize(count);
        for (std::size_t n = 0; n < count; ++n)
        {
            result.rounded[n]  = output[11 * n + 6];
            result.residual[n] = output[11 * n + 7];
        }
    }
    return result;
}
} // namespace tmv::held_remap_detail
