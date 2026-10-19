#pragma once

#include "init/trigger/vortex_seed.hpp"

namespace tmv::init
{

/**
 * @brief Tagged config selecting which TriggerSource the runtime applies.
 *
 * The runtime reads `trigger.type` plus per-type knobs into this struct.
 * WarmBubble continues to source its parameters from the legacy
 * trigger.bubble.* globals (global_bubble_*); the VortexSeed branch is
 * the first trigger to receive a struct directly.
 */
struct TriggerSourceConfig
{
    enum class Type
    {
        WarmBubble,
        None,
        VortexSeed,
        // ColdPool     // future
    };

    Type type = Type::WarmBubble;
    VortexSeedParams vortex_seed;
};

}  // namespace tmv::init
