#pragma once

#include <string>

namespace tmv::init
{

/**
 * @file trigger_source.hpp
 * @brief Abstract producer of an initial-condition trigger perturbation.
 *
 * A TriggerSource modifies the simulation field in place after the
 * SoundingSource column has been broadcast. Concrete sources:
 *
 *   - WarmBubbleTrigger : Gaussian Δθ patch, today's default. Cartesian
 *     uses a 3D sphere; cylindrical uses a 2D ring in (r, z).
 *   - NoOpTrigger       : applies nothing. For hydrostatic / equilibrium
 *     tests where any perturbation seeds startup transients.
 *   - (future) VortexSeedTrigger : Rankine vortex for tornado schemes
 *                ColdPoolTrigger  : negative Δθ patch
 *
 * Triggers operate on the global Field3D state declared in
 * `include/core/simulation.hpp`. They read their own configuration from
 * globals (e.g. global_bubble_*) so the existing YAML keys keep working.
 */

class TriggerSource
{
public:
    virtual ~TriggerSource() = default;

    /// Modifies the simulation state in place. Called once during
    /// initialize(), after the sounding column has been broadcast and
    /// the wind initialization has run, so the trigger can rely on a
    /// fully populated base state.
    virtual void apply() const = 0;

    /// Short identifier for logging (e.g. "warm_bubble", "none",
    /// "vortex_seed/rankine").
    virtual std::string describe() const = 0;
};

}  // namespace tmv::init
