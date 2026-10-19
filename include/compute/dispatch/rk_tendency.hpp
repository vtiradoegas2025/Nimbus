#pragma once
#include "compute/dispatch/rk_combination.hpp"

enum class RkTendencyOperation
{
    Acoustic,
    MassFlux,
    Divergence,
    ScalarFlux,
    DonorLimiter,
    LimitedDivergence,
    Add,
    MixingRatio,
    ScalarHalo,
    HorizontalWindHalo,
    VerticalWindHalo,
    VolumeConcentrationHalo,
    BasePerturbationHalo,
    NonnegativeInput,
    NonnegativeMixingRatio
};

struct RkTendencyDispatch
{
    // Acoustic: u/v/w/rho/p/p0 high/p0 low. Transport: q/fx/fy/fz/rho/limiter/unused.
    std::array<const float*, 7> input{};
    std::array<float*, 5>       output{};
    RkTendencyOperation         operation = RkTendencyOperation::Acoustic;
    int                         nx = 0, ny = 0, nz = 0;
    double                      dx = 1, dy = 1, dz = 1, dt = 1, gamma = 1;
    bool                        periodic_x = false, periodic_y = false, open_top = false;
    bool                        weno = false, positive = false, collocated = false;
};
