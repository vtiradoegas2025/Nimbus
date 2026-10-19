#pragma once

#include <array>
#include <cstddef>

enum class RkCombinationStatus
{
    Completed,
    Unavailable,
    InvalidInput,
    NonFinite
};

struct RkCombinationDispatch
{
    const float*                base    = nullptr;
    const float*                current = nullptr;
    std::array<const float*, 4> rhs{};
    // Required only for conserved mixing ratios; end density is the stored, admitted candidate.
    const float* base_density      = nullptr;
    const float* current_density   = nullptr;
    const float* candidate_density = nullptr;
    // Value high/low, then increment-scale high/low. Publication is atomic across all four fields.
    std::array<float*, 4> output{};
    std::size_t           cells                  = 0;
    int                   stages                 = 0;
    bool                  ssp                    = false;
    bool                  conserved_mixing_ratio = false;
    double                dt                     = 0;
    std::array<double, 4> weights{};
};
