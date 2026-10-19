#include "numerics/advection/factory.hpp"
#include "numerics/advection/advection_base.hpp"
#include "schemes/tvd/tvd.hpp"
#include "schemes/weno5/weno5.hpp"
#include "schemes/none/none.hpp"
#include "util/scheme_factory.hpp"

namespace tmv
{

namespace
{
const SchemeRegistry<AdvectionSchemeBase> registry({
    {"tvd",   [] { return std::make_unique<TVDScheme>(); }},
    {"weno5", [] { return std::make_unique<WENO5Scheme>(); }},
    {"none",  [] { return std::make_unique<NoAdvectionScheme>(); }},
}, get_advection_scheme_descriptors());
}

std::unique_ptr<AdvectionSchemeBase> create_advection_scheme(const std::string& scheme_name)
{
    return registry.create("advection", scheme_name);
}

std::vector<std::string> get_available_advection_schemes()
{
    return registry.available_ids();
}

} // namespace tmv
