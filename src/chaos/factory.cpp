#include "chaos/factory.hpp"
#include "chaos/scheme.hpp"
#include "schemes/none/none.hpp"
#include "schemes/initial_conditions/initial_conditions.hpp"
#include "schemes/boundary_layer/boundary_layer.hpp"
#include "schemes/full_stochastic/full_stochastic.hpp"
#include "util/scheme_factory.hpp"

namespace tmv::chaos
{

namespace
{
const SchemeRegistry<ChaosScheme> registry({
    {"none",               [] { return std::make_unique<NoneScheme>(); }},
    {"initial_conditions", [] { return std::make_unique<InitialConditionsScheme>(); }},
    {"boundary_layer",     [] { return std::make_unique<BoundaryLayerScheme>(); }},
    {"full_stochastic",    [] { return std::make_unique<FullStochasticScheme>(); }},
}, get_chaos_scheme_descriptors());
}

std::unique_ptr<ChaosScheme> create_chaos_scheme(const std::string& scheme_name)
{
    return registry.create("chaos", scheme_name);
}

std::vector<std::string> get_available_chaos_schemes()
{
    return registry.available_ids();
}

} // namespace tmv::chaos
