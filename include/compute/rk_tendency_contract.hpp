#pragma once
#include "compute/dispatch/rk_tendency.hpp"
#include "compute/rk_combination_contract.hpp"

namespace nimbus::compute::rk_tendency
{
inline std::size_t cells(const RkTendencyDispatch& d)
{
    return std::size_t(d.nx) * std::size_t(d.ny) * std::size_t(d.nz);
}
inline bool validate(const RkTendencyDispatch& d)
{
    if (d.nx < 3 || d.ny < 3 || d.nz < 2 ||
        std::size_t(d.nx) > std::size_t(std::numeric_limits<int>::max()) / std::size_t(d.ny) / std::size_t(d.nz) ||
        int(d.operation) < 0 || int(d.operation) > int(RkTendencyOperation::NonnegativeMixingRatio))
        return false;
    if (int(d.operation) >= int(RkTendencyOperation::ScalarHalo) &&
        int(d.operation) <= int(RkTendencyOperation::BasePerturbationHalo) &&
        (!d.periodic_x || !d.periodic_y || d.open_top))
        return false;
    if ((d.collocated && (d.nz < 3 || d.operation == RkTendencyOperation::Acoustic)) ||
        (!d.collocated && d.operation == RkTendencyOperation::BasePerturbationHalo))
        return false;
    if (d.collocated && d.operation == RkTendencyOperation::MassFlux && (!d.periodic_x || !d.periodic_y))
        return false;
    for (double v : {d.dx, d.dy, d.dz, d.dt, d.gamma})
        if (!std::isfinite(v) || v <= 0 || !std::isfinite(float(v)) || float(v) <= 0 ||
            !std::isfinite(float(1.0 / v)) || float(1.0 / v) <= 0)
            return false;
    const auto count = cells(d);
    for (auto p : d.input)
        if (!p)
            return false;
    for (std::size_t f = 0; f < d.output.size(); ++f)
    {
        if (!d.output[f])
            return false;
        for (auto p : d.input)
            if (rk_combination::overlap(p, d.output[f], count))
                return false;
        for (std::size_t g = 0; g < f; ++g)
            if (rk_combination::overlap(d.output[g], d.output[f], count))
                return false;
    }
    for (std::size_t n = 0; n < count; ++n)
    {
        for (auto p : d.input)
            if (!std::isfinite(p[n]))
                return false;
        if ((d.operation == RkTendencyOperation::Acoustic && d.input[3][n] <= 0) ||
            (d.operation == RkTendencyOperation::MassFlux && d.input[3][n] <= 0) ||
            ((d.operation == RkTendencyOperation::MixingRatio ||
              d.operation == RkTendencyOperation::NonnegativeMixingRatio ||
              d.operation == RkTendencyOperation::VolumeConcentrationHalo) &&
             d.input[1][n] <= 0) ||
            (d.operation == RkTendencyOperation::DonorLimiter && d.input[4][n] <= 0))
            return false;
        if (d.operation == RkTendencyOperation::LimitedDivergence && d.positive &&
            (d.input[5][n] < 0 || d.input[5][n] > 1))
            return false;
    }
    return true;
}
inline abi::RkTendencyPushConstants constants(const RkTendencyDispatch& d)
{
    abi::RkTendencyPushConstants c{};
    c.nx             = d.nx;
    c.ny             = d.ny;
    c.nz             = d.nz;
    c.operation      = int(d.operation);
    const auto split = [](double v, float& hi, float& lo)
    {
        hi = float(v);
        lo = float(v - hi);
    };
    split(1 / d.dx, c.idx_high, c.idx_low);
    split(1 / d.dy, c.idy_high, c.idy_low);
    split(1 / d.dz, c.idz_high, c.idz_low);
    split(d.dt, c.step_high, c.step_low);
    split(d.gamma, c.gamma_high, c.gamma_low);
    c.flags = int(d.periodic_x) | (int(d.periodic_y) << 1) | (int(d.open_top) << 2) | (int(d.weno) << 3) |
              (int(d.positive) << 4) | (int(d.collocated) << 5);
    return c;
}
inline bool finite_candidate(const std::array<const float*, 5>& outputs, std::size_t count)
{
    for (auto p : outputs)
        for (std::size_t n = 0; n < count; ++n)
            if (!std::isfinite(p[n]))
                return false;
    return true;
}
} // namespace nimbus::compute::rk_tendency
