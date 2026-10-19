#include "radiation/factory.hpp"
#include "radiation/radiation_base.hpp"
#include "schemes/simple_grey/simple_grey.hpp"
#include "schemes/rrtmg/rrtmg.hpp"
#include "schemes/none/none.hpp"
#include "util/scheme_factory.hpp"

#include <stdexcept>

namespace tmv
{

namespace
{
const tmv::SchemeRegistry<RadiationSchemeBase> registry(
    {
        {"simple_grey", [] { return std::make_unique<SimpleGreyScheme>(); }},
        {"rrtmg", [] { return std::make_unique<RrtmgScheme>(); }},
        {"none", [] { return std::make_unique<radiation::NoneScheme>(); }},
    },
    get_radiation_scheme_descriptors());
} // namespace

std::string canonicalize_radiation_scheme_id(const std::string& scheme_name)
{
    return canonicalize_scheme_id(get_radiation_scheme_descriptors(), scheme_name);
}

bool is_radiation_scheme_implemented(const std::string& scheme_name)
{
    const auto* descriptor = find_scheme_descriptor(get_radiation_scheme_descriptors(), scheme_name);
    return descriptor && descriptor->constructible;
}

std::unique_ptr<RadiationSchemeBase> create_radiation_scheme(const std::string& scheme_name)
{
    const std::string canonical = canonicalize_radiation_scheme_id(scheme_name);

    return registry.create("radiation", canonical);
}

std::vector<std::string> get_available_radiation_schemes()
{
    return registry.available_ids();
}

std::vector<std::string> get_known_radiation_schemes()
{
    std::vector<std::string> ids;
    const auto&              descriptors = get_radiation_scheme_descriptors();
    ids.reserve(descriptors.size());
    for (const auto& descriptor : descriptors)
    {
        ids.push_back(descriptor.id);
    }
    return ids;
}

} // namespace tmv
