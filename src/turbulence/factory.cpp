#include "turbulence/factory.hpp"
#include "turbulence/turbulence_base.hpp"
#include "schemes/smagorinsky/smagorinsky.hpp"
#include "schemes/tke/tke.hpp"
#include "schemes/none/none.hpp"
#include "util/scheme_factory.hpp"

namespace tmv
{

namespace
{
const SchemeRegistry<TurbulenceSchemeBase> registry({
    {"smagorinsky", [] { return std::make_unique<SmagorinskyScheme>(); }},
    {"tke",         [] { return std::make_unique<TKEScheme>(); }},
    {"none",        [] { return std::make_unique<turbulence::NoneScheme>(); }},
}, get_turbulence_scheme_descriptors());
}

std::unique_ptr<TurbulenceSchemeBase> create_turbulence_scheme(const std::string& scheme_name)
{
    return registry.create("turbulence", scheme_name);
}

std::vector<std::string> get_available_turbulence_schemes()
{
    return registry.available_ids();
}

} // namespace tmv
