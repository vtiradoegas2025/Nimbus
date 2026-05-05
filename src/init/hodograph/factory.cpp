/**
 * @file factory.cpp
 * @brief Dispatcher from HodographSourceConfig to a concrete HodographSource.
 */

#include "init/hodograph/factory.hpp"
#include "init/hodograph/wk_param.hpp"
#include "init/hodograph/zero.hpp"

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

bool parse_hodograph_source_type(const std::string& s, HodographSourceConfig::Type& out)
{
    const std::string norm = normalize(s);
    if (norm == "auto" || norm.empty())
    {
        out = HodographSourceConfig::Type::Auto;
        return true;
    }
    if (norm == "wk_param" || norm == "wkparam" || norm == "wk")
    {
        out = HodographSourceConfig::Type::WKParam;
        return true;
    }
    if (norm == "zero" || norm == "static" || norm == "calm")
    {
        out = HodographSourceConfig::Type::Zero;
        return true;
    }
    return false;
}

std::string hodograph_source_type_name(HodographSourceConfig::Type t)
{
    switch (t)
    {
        case HodographSourceConfig::Type::Auto:    return "auto";
        case HodographSourceConfig::Type::WKParam: return "wk_param";
        case HodographSourceConfig::Type::Zero:    return "zero";
    }
    return "unknown";
}

std::unique_ptr<HodographSource> make_hodograph_source(const HodographSourceConfig& cfg)
{
    switch (cfg.type)
    {
        case HodographSourceConfig::Type::Auto:
        case HodographSourceConfig::Type::WKParam:
            return std::make_unique<WKParamHodograph>(cfg.wk_anchors);
        case HodographSourceConfig::Type::Zero:
            return std::make_unique<ZeroHodograph>();
    }
    throw std::logic_error("make_hodograph_source: unhandled HodographSourceConfig::Type");
}

}  // namespace tmv::init
