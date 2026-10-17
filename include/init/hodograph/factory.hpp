#pragma once

#include "init/hodograph/hodograph_source_config.hpp"
#include "init/hodograph/hodograph_source.hpp"
#include "init/sounding/sounding_source.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tmv::init
{

bool        parse_hodograph_source_type(const std::string& s, HodographSourceConfig::Type& out);
std::string hodograph_source_type_name(HodographSourceConfig::Type t);

// Auto constructs WKParam. Runtime sounding winds take precedence, and the
// hodograph source supplies winds only when the sounding has none.
std::unique_ptr<HodographSource> make_hodograph_source(const HodographSourceConfig& cfg);

/**
 * Resolve the exact wind column used by initialization.
 *
 * An explicit hodograph overrides file winds. Auto uses sounding winds when
 * present and otherwise builds the configured fallback hodograph. Keeping
 * this precedence in one function prevents preflight diagnostics from
 * reasoning about a different environment than the 3-D initializer.
 */
WindColumn
resolve_wind_column(const std::vector<double>& z_m, const Sounding& sounding, const HodographSourceConfig& config);

} // namespace tmv::init
