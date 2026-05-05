#pragma once

#include "init/hodograph/hodograph_source.hpp"
#include "init/hodograph/wk_param.hpp"

#include <memory>
#include <string>

namespace tmv::init
{

/**
 * @brief Tagged config selecting which HodographSource the runtime builds.
 *
 * The runtime reads `environment.hodograph.*` YAML keys into this struct.
 * Auto means "let the runtime pick at init time": use the SoundingSource's
 * winds when that source carries any (file-based path), otherwise fall
 * back to WKParam constructed from the legacy environment.hodograph.* WK
 * anchors. Explicit types always win over the source-supplied winds, so a
 * user can override file winds with `hodograph.type: zero` when they want
 * to test pressure-driven flow on a real-world thermo column.
 */
struct HodographSourceConfig
{
    enum class Type
    {
        Auto,
        WKParam,
        Zero,
    };

    Type type = Type::Auto;
    WKParamHodographAnchors wk_anchors;  // populated from environment.hodograph.* anchors
};

bool parse_hodograph_source_type(const std::string& s, HodographSourceConfig::Type& out);
std::string hodograph_source_type_name(HodographSourceConfig::Type t);

/**
 * @brief Constructs the configured HodographSource.
 *
 * Auto resolves to WKParam at construction time — the "use sounding winds
 * if available" behavior is enforced by the runtime, which checks the
 * Sounding column first and only calls into the hodograph source when the
 * sounding lacks winds. Build-time always returns a usable source; never
 * returns null.
 */
std::unique_ptr<HodographSource> make_hodograph_source(const HodographSourceConfig& cfg);

}  // namespace tmv::init
