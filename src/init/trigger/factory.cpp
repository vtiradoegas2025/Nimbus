/**
 * @file factory.cpp
 * @brief Dispatcher from TriggerSourceConfig to a concrete TriggerSource.
 */

#include "init/trigger/factory.hpp"
#include "init/trigger/none.hpp"
#include "init/trigger/vortex_seed.hpp"
#include "init/trigger/warm_bubble.hpp"

#include <cctype>
#include <stdexcept>

namespace tmv::init
{

namespace
{

std::string normalize(const std::string& s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s)
    {
        if (std::isspace(static_cast<unsigned char>(c)))
        {
            continue;
        }
        if (c == '-')
        {
            out.push_back('_');
            continue;
        }
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

}  // namespace

bool parse_trigger_source_type(const std::string& s, TriggerSourceConfig::Type& out)
{
    const std::string norm = normalize(s);
    if (norm == "warm_bubble" || norm == "warmbubble" || norm == "bubble")
    {
        out = TriggerSourceConfig::Type::WarmBubble;
        return true;
    }
    if (norm == "none" || norm == "off" || norm == "disabled")
    {
        out = TriggerSourceConfig::Type::None;
        return true;
    }
    if (norm == "vortex_seed" || norm == "vortexseed" || norm == "rankine"
        || norm == "vortex")
    {
        out = TriggerSourceConfig::Type::VortexSeed;
        return true;
    }
    return false;
}

std::string trigger_source_type_name(TriggerSourceConfig::Type t)
{
    switch (t)
    {
        case TriggerSourceConfig::Type::WarmBubble: return "warm_bubble";
        case TriggerSourceConfig::Type::None:       return "none";
        case TriggerSourceConfig::Type::VortexSeed: return "vortex_seed";
    }
    return "unknown";
}

std::unique_ptr<TriggerSource> make_trigger_source(const TriggerSourceConfig& cfg)
{
    switch (cfg.type)
    {
        case TriggerSourceConfig::Type::WarmBubble:
            return std::make_unique<WarmBubbleTrigger>();
        case TriggerSourceConfig::Type::None:
            return std::make_unique<NoOpTrigger>();
        case TriggerSourceConfig::Type::VortexSeed:
            return std::make_unique<VortexSeedTrigger>(cfg.vortex_seed);
    }
    throw std::logic_error("make_trigger_source: unhandled TriggerSourceConfig::Type");
}

}  // namespace tmv::init
