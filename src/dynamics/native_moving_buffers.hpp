#pragma once
#include "compute/dispatch/terrain.hpp"
#include "compute/moving_momentum_contract.hpp"
#include "terrain/terrain_base.hpp"
#include "core/field/field3d.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tmv::moving_transport_detail
{
struct MomentumBuffers
{
    MovingMomentumDispatch dispatch;
    std::vector<float>     input, output;
    MomentumBuffers(TerrainComputeKernels& kernels, int nx, int ny, int nz)
    {
        if (!kernels.supports_moving_momentum())
            throw std::runtime_error("native moving-momentum kernels are unavailable");
        if (nx < 4 || ny < 4 || nz < 2 ||
            std::size_t(nx) + 1 > std::size_t(std::numeric_limits<int>::max()) / MovingMomentumDispatch::output_stride /
                                      (std::size_t(ny) + 1) / (std::size_t(nz) + 1))
            throw std::invalid_argument("native moving momentum received invalid dimensions");
        auto& p   = dispatch.constants;
        p.nx      = nx;
        p.ny      = ny;
        p.nz      = nz;
        p.dx_high = p.dy_high = p.dz_high = 1;
        const auto count                  = nimbus::compute::moving_momentum::cells(dispatch);
        input.assign(count * MovingMomentumDispatch::input_stride, 0);
        output.resize(count * MovingMomentumDispatch::output_stride);
        dispatch.input  = input.data();
        dispatch.output = output.data();
    }
    std::size_t index(int i, int j, int k) const
    {
        const auto& p = dispatch.constants;
        return (std::size_t(i) * (p.ny + 1) + j) * (p.nz + 1) + k;
    }
    static void pack(double value, float& high, float& low)
    {
        high = static_cast<float>(value);
        low  = static_cast<float>(value - high);
        if (!std::isfinite(value) || !std::isfinite(high) || !std::isfinite(low))
            throw std::overflow_error("native moving momentum exceeds compensated storage");
    }
    void field(const Field3D& source, int slot)
    {
        const auto& p = dispatch.constants;
        if (source.size_r() != p.nx || source.size_th() != p.ny || source.size_z() != p.nz ||
            source.size() != std::size_t(p.nx) * p.ny * p.nz)
            throw std::invalid_argument("native moving momentum received malformed state storage");
        for (int i = 0; i < p.nx; ++i)
            for (int j = 0; j < p.ny; ++j)
                for (int k = 0; k < p.nz; ++k)
                    input[index(i, j, k) * MovingMomentumDispatch::input_stride + slot] = source(i, j, k);
    }
    void field(const TerrainField3D& source, int slot, int nx, int ny, int nz)
    {
        if (source.nr != nx || source.nth != ny || source.nz != nz || source.size() != std::size_t(nx) * ny * nz)
            throw std::invalid_argument("native moving momentum received malformed geometry or flux storage");
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    auto* record = input.data() + index(i, j, k) * MovingMomentumDispatch::input_stride;
                    pack(source(i, j, k), record[slot], record[slot + 1]);
                }
    }
    void layer(const Field3D& source, int slot, int source_k, int target_k)
    {
        const auto& p = dispatch.constants;
        if (source.size_r() != p.nx || source.size_th() != p.ny || source_k < 0 || source_k >= source.size_z() ||
            target_k < 0 || target_k > p.nz)
            throw std::invalid_argument("native boundary layer does not match its record grid");
        for (int i = 0; i < p.nx; ++i)
            for (int j = 0; j < p.ny; ++j)
                input[index(i, j, target_k) * MovingMomentumDispatch::input_stride + slot] = source(i, j, source_k);
    }
    void layer(const TerrainField3D& source, int slot, int source_k, int target_k)
    {
        const auto& p = dispatch.constants;
        if (source.nr > p.nx + 1 || source.nth > p.ny + 1 || source_k < 0 || source_k >= source.nz || target_k < 0 ||
            target_k > p.nz)
            throw std::invalid_argument("native boundary metric layer does not match its record grid");
        for (int i = 0; i < source.nr; ++i)
            for (int j = 0; j < source.nth; ++j)
            {
                auto* record = input.data() + index(i, j, target_k) * MovingMomentumDispatch::input_stride;
                pack(source(i, j, source_k), record[slot], record[slot + 1]);
            }
    }
    void execute(TerrainComputeKernels& kernels)
    {
        if (!nimbus::compute::moving_momentum::validate(dispatch))
            throw std::invalid_argument("native moving-momentum input violates its dispatch contract");
        if (!kernels.dispatch_moving_momentum(dispatch))
            throw std::runtime_error("native moving-momentum dispatch failed");
        if (!nimbus::compute::moving_momentum::finite_candidate(output.data(), dispatch))
            throw std::overflow_error("native moving momentum returned an invalid candidate");
    }
    void flux(TerrainField3D& result, int slot, int nx, int ny, int nz) const
    {
        result.resize(nx, ny, nz, 0);
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const auto* record = output.data() + index(i, j, k) * MovingMomentumDispatch::output_stride;
                    result(i, j, k)    = double(record[slot]) + record[slot + 1];
                }
    }
};
} // namespace tmv::moving_transport_detail
