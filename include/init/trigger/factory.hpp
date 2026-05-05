#pragma once

#include "init/trigger/trigger_source.hpp"
#include "init/trigger/vortex_seed.hpp"

#include <memory>
#include <string>

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

bool parse_trigger_source_type(const std::string& s, TriggerSourceConfig::Type& out);
std::string trigger_source_type_name(TriggerSourceConfig::Type t);

/// Always returns a usable source. Default is WarmBubble (today's behavior).
std::unique_ptr<TriggerSource> make_trigger_source(const TriggerSourceConfig& cfg);

}  // namespace tmv::init
