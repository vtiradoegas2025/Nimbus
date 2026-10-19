#include "terrain/factory.hpp"
#include "terrain/terrain_base.hpp"
#include "schemes/bell/bell.hpp"
#include "schemes/schar/schar.hpp"
#include "schemes/none/none.hpp"
#include "util/scheme_factory.hpp"

namespace tmv
{

namespace
{
const tmv::SchemeRegistry<TerrainSchemeBase> registry({
    {"none",  [] { return std::make_unique<NoneScheme>(); }},
    {"bell",  [] { return std::make_unique<BellScheme>(); }},
    {"schar", [] { return std::make_unique<ScharScheme>(); }},
}, get_terrain_scheme_descriptors());
}

std::unique_ptr<TerrainSchemeBase> create_terrain_scheme(const std::string& scheme_name)
{
    return registry.create("terrain", scheme_name);
}

std::vector<std::string> get_available_terrain_schemes()
{
    return registry.available_ids();
}

} // namespace tmv
