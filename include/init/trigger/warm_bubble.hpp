#pragma once

#include "init/trigger/trigger_source.hpp"

namespace tmv::init
{

/**
 * @brief Today's default trigger: Gaussian theta perturbation.
 *
 * Dispatches by global_coordinate_system to apply_cartesian_bubble_initialization
 * (3D sphere) or apply_cylindrical_bubble_initialization (2D ring in r,z).
 * The bubble configuration (center, radius, dtheta) is read from the
 * legacy globals (global_bubble_*). This class is currently a thin wrapper
 * around the existing helpers; future refactors may absorb the math.
 *
 * Setting dtheta_k = 0 in the config produces a no-op equivalent to
 * NoOpTrigger and remains the legacy way to disable the trigger; the
 * recommended new way is `trigger.type: none`.
 */
class WarmBubbleTrigger final : public TriggerSource
{
public:
    void apply() const override;
    std::string describe() const override;
};

}  // namespace tmv::init
