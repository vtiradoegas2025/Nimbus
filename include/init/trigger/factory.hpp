#pragma once

#include "init/trigger/trigger_source_config.hpp"
#include "init/trigger/trigger_source.hpp"

#include <memory>
#include <string>

namespace tmv::init
{

bool parse_trigger_source_type(const std::string& s, TriggerSourceConfig::Type& out);
std::string trigger_source_type_name(TriggerSourceConfig::Type t);

// An unset source preserves the historical WarmBubble default.
std::unique_ptr<TriggerSource> make_trigger_source(const TriggerSourceConfig& cfg);

}  // namespace tmv::init
