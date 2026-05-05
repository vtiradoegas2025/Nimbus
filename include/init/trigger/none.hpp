#pragma once

#include "init/trigger/trigger_source.hpp"

namespace tmv::init
{

/**
 * @brief Trigger that applies nothing.
 *
 * Selected via `trigger.type: none` for hydrostatic / equilibrium tests
 * where any perturbation seeds startup transients. Equivalent to setting
 * `trigger.bubble.dtheta_k: 0` in the legacy YAML, but explicit.
 */
class NoOpTrigger final : public TriggerSource
{
public:
    void apply() const override;
    std::string describe() const override;
};

}  // namespace tmv::init
