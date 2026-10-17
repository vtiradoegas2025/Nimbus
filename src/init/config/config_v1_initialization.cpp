#include "init/config/config_v1.hpp"

#include <algorithm>
#include <cctype>

namespace tmv::init::config
{

namespace
{

std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return s;
}

} // namespace

// ─────────────────────────────────────────────────────────────
// TriggerType
// ─────────────────────────────────────────────────────────────

bool parse_trigger_type(const std::string& s, TriggerType& out)
{
    const std::string v = to_lower(s);
    if (v == "none")
    {
        out = TriggerType::None;
        return true;
    }
    if (v == "bubble")
    {
        out = TriggerType::Bubble;
        return true;
    }
    if (v == "vortex_seed")
    {
        out = TriggerType::VortexSeed;
        return true;
    }
    return false;
}

const char* trigger_type_name(TriggerType v)
{
    switch (v)
    {
    case TriggerType::None:
        return "none";
    case TriggerType::Bubble:
        return "bubble";
    case TriggerType::VortexSeed:
        return "vortex_seed";
    }
    return "none";
}

// ─────────────────────────────────────────────────────────────
// SoundingType
// ─────────────────────────────────────────────────────────────

bool parse_sounding_type(const std::string& s, SoundingType& out)
{
    const std::string v = to_lower(s);
    if (v == "parametric_cape")
    {
        out = SoundingType::ParametricCAPE;
        return true;
    }
    if (v == "parametric_targets")
    {
        out = SoundingType::ParametricTargets;
        return true;
    }
    if (v == "file")
    {
        out = SoundingType::File;
        return true;
    }
    return false;
}

const char* sounding_type_name(SoundingType v)
{
    switch (v)
    {
    case SoundingType::ParametricCAPE:
        return "parametric_cape";
    case SoundingType::ParametricTargets:
        return "parametric_targets";
    case SoundingType::File:
        return "file";
    }
    return "parametric_cape";
}

// ─────────────────────────────────────────────────────────────
// HodographType
// ─────────────────────────────────────────────────────────────

bool parse_hodograph_type(const std::string& s, HodographType& out)
{
    const std::string v = to_lower(s);
    if (v == "auto")
    {
        out = HodographType::Auto;
        return true;
    }
    if (v == "custom")
    {
        out = HodographType::Custom;
        return true;
    }
    if (v == "wk_param")
    {
        out = HodographType::WKParam;
        return true;
    }
    return false;
}

const char* hodograph_type_name(HodographType v)
{
    switch (v)
    {
    case HodographType::Auto:
        return "auto";
    case HodographType::Custom:
        return "custom";
    case HodographType::WKParam:
        return "wk_param";
    }
    return "auto";
}

// ─────────────────────────────────────────────────────────────
} // namespace tmv::init::config
