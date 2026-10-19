#include "numerics/diffusion/factory.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "schemes/explicit/explicit.hpp"
#include "schemes/implicit/implicit.hpp"
#include "util/scheme_factory.hpp"

namespace tmv
{

namespace
{
const SchemeRegistry<DiffusionSchemeBase> registry({
    {"explicit", [] { return std::make_unique<ExplicitDiffusionScheme>(); }},
    {"implicit", [] { return std::make_unique<ImplicitDiffusionScheme>(); }},
}, get_diffusion_scheme_descriptors());
}

std::unique_ptr<DiffusionSchemeBase> create_diffusion_scheme(const std::string& scheme_name)
{
    return registry.create("diffusion", scheme_name);
}

std::vector<std::string> get_available_diffusion_schemes()
{
    return registry.available_ids();
}

} // namespace tmv
