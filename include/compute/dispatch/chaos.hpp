#pragma once

#include <cstdint>

enum class EarthControlEvolutionMode : std::uint32_t
{
    Initialize = 0,
    Advance = 1,
    Reconstruct = 2,
};

struct EarthControlEvolutionDispatch
{
    const float* previous = nullptr;
    float* destination = nullptr;
    std::uint64_t seed = 0;
    std::int64_t member_id = 0;
    std::uint64_t initial_process_key = 0;
    std::uint64_t innovation_process_key = 0;
    std::uint64_t innovation_epoch = 0;
    std::int64_t control_i_min = 0;
    std::int64_t control_j_min = 0;
    int control_i_count = 0;
    int control_j_count = 0;
    int nz = 0;
    double rho = 0.0;
    EarthControlEvolutionMode mode = EarthControlEvolutionMode::Initialize;
};
