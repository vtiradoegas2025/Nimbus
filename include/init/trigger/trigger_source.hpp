#pragma once

#include <string>

struct SimulationStateView;

namespace tmv::init
{

/**
 * @file trigger_source.hpp
 * @brief Abstract producer of an initial-condition trigger perturbation.
 *
 * A TriggerSource modifies the simulation field in place after the
 * SoundingSource column has been broadcast. Concrete sources:
 *
 *   - WarmBubbleTrigger : Gaussian Δθ patch (3D sphere), today's default.
 *   - NoOpTrigger       : applies nothing. For hydrostatic / equilibrium
 *     tests where any perturbation seeds startup transients.
 *   - VortexSeedTrigger : Rankine vortex seed for vortex spin-up.
 *   - (future) ColdPoolTrigger : negative Δθ patch
 *
 * The legacy virtual entry point still operates on the process-wide runtime
 * state. The state overload lets migrated callers route trigger writes through
 * the transitional SimulationStateView adapter without changing factory
 * dispatch.
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
    virtual void apply(const SimulationStateView& state) const = 0;

    /// Short identifier for logging (e.g. "warm_bubble", "none",
    /// "vortex_seed/rankine").
    virtual std::string describe() const = 0;
};

}  // namespace tmv::init
