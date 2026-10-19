#include "boundary_layer/factory.hpp"
#include "boundary_layer/boundary_layer_base.hpp"
#include "schemes/slab/slab.hpp"
#include "schemes/ysu/ysu.hpp"
#include "schemes/mynn/mynn.hpp"
#include "schemes/none/none.hpp"
#include "util/scheme_factory.hpp"

namespace tmv
{

namespace
{
const SchemeRegistry<BoundaryLayerSchemeBase> registry({
    {"slab", [] { return std::make_unique<SlabScheme>(); }},
    {"ysu",  [] { return std::make_unique<YSUScheme>(); }},
    {"mynn", [] { return std::make_unique<MYNNScheme>(); }},
    {"none", [] { return std::make_unique<boundary_layer::NoneScheme>(); }},
}, get_boundary_layer_scheme_descriptors());
}

std::unique_ptr<BoundaryLayerSchemeBase> create_boundary_layer_scheme(const std::string& scheme_name)
{
    return registry.create("boundary layer", scheme_name);
}

std::vector<std::string> get_available_boundary_layer_schemes()
{
    return registry.available_ids();
}

} // namespace tmv
