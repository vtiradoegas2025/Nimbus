#pragma once

#include <vector>
#include <memory>
#include "boundary_conditions/boundary_conditions_base.hpp"

namespace tmv
{

// Time-integrated lateral-wave memory belongs to one atmospheric level. The
// vectors are transient continuation scratch: completed-step checkpoints may
// rebuild them from the live boundary faces before the next acoustic loop.
struct BoundaryConditionWaveState
{
    std::vector<float> radiative_u_west;
    std::vector<float> radiative_u_east;
    std::vector<float> radiative_v_south;
    std::vector<float> radiative_v_north;

    void clear() noexcept
    {
        radiative_u_west.clear();
        radiative_u_east.clear();
        radiative_v_south.clear();
        radiative_v_north.clear();
    }
};

struct BoundaryConditionRuntimeState : BoundaryConditionWaveState
{
    std::unique_ptr<BoundaryConditionScheme> execution_operator;
    BoundaryConditionRuntimeState() = default;
    BoundaryConditionRuntimeState(const BoundaryConditionRuntimeState& source)
        : BoundaryConditionWaveState(source) {}
    BoundaryConditionRuntimeState& operator=(const BoundaryConditionRuntimeState& source)
    {
        static_cast<BoundaryConditionWaveState&>(*this) = source;
        return *this;
    }
    BoundaryConditionRuntimeState(BoundaryConditionRuntimeState&&) noexcept = default;
    BoundaryConditionRuntimeState& operator=(BoundaryConditionRuntimeState&&) noexcept = default;
};

extern BoundaryConditionRuntimeState& global_boundary_condition_runtime_state;

} // namespace tmv

