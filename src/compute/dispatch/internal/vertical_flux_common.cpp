#include "compute/dispatch/internal/vertical_flux.hpp"

#include <cctype>

namespace nimbus::compute::dispatch_internal
{

std::string canonicalize_id(std::string value)
{
    std::string canonical;
    canonical.reserve(value.size());
    for (unsigned char c : value)
    {
        if (c == '_' || c == '-' || std::isspace(c))
        {
            continue;
        }
        canonical.push_back(static_cast<char>(std::tolower(c)));
    }
    return canonical;
}

bool parse_tvd_limiter_kind(const std::string& limiter_id, TVDLimiterKind& out_kind)
{
    const std::string canonical = canonicalize_id(limiter_id);
    if (canonical == "minmod")
    {
        out_kind = TVDLimiterKind::minmod;
        return true;
    }
    if (canonical == "vanleer")
    {
        out_kind = TVDLimiterKind::vanleer;
        return true;
    }
    if (canonical == "superbee")
    {
        out_kind = TVDLimiterKind::superbee;
        return true;
    }
    if (canonical == "mc" || canonical.empty())
    {
        out_kind = TVDLimiterKind::mc;
        return true;
    }
    if (canonical == "universal")
    {
        out_kind = TVDLimiterKind::universal;
        return true;
    }
    return false;
}

} // namespace nimbus::compute::dispatch_internal
