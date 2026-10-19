#include "radar/factory.hpp"
#include "radar/radar_base.hpp"
#include "schemes/reflectivity/reflectivity.hpp"
#include "schemes/velocity/velocity.hpp"
#include "schemes/zdr/zdr.hpp"
#include "util/scheme_factory.hpp"

namespace tmv
{

namespace
{
const tmv::SchemeRegistry<RadarSchemeBase> registry({
    {"reflectivity", [] { return std::make_unique<ReflectivityScheme>(); }},
    {"velocity",     [] { return std::make_unique<VelocityScheme>(); }},
    {"zdr",          [] { return std::make_unique<ZDRScheme>(); }},
}, get_radar_scheme_descriptors());
}

std::unique_ptr<RadarSchemeBase> create_radar_scheme(const std::string& scheme_name)
{
    return registry.create("radar", scheme_name);
}

std::vector<std::string> get_available_radar_schemes()
{
    return registry.available_ids();
}

} // namespace tmv
