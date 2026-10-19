#pragma once

#include "init/hodograph/wk_param.hpp"

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

}  // namespace tmv::init
