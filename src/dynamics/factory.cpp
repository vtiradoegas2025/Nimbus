#include "dynamics/factory.hpp"
#include "core/runtime/simulation_state.hpp"
#include "dynamics/dynamics_base.hpp"
#include "init/config/config_v1.hpp"
#include "schemes/klemp_wilhelmson_split/cartesian_agrid.hpp"
#include "schemes/klemp_wilhelmson_split/cartesian_cgrid_3d.hpp"
#include "util/scheme_factory.hpp"

namespace tmv
{

namespace
{
// The supported Cartesian schemes differ only by staggering.
const SchemeRegistry<DynamicsScheme> registry({
    {kCartesianAGridDynamicsId, [] { return std::make_unique<CartesianAGridScheme>();   }},
    {kCartesianCGridDynamicsId, [] { return std::make_unique<CartesianCGrid3DScheme>(); }},
}, get_dynamics_scheme_descriptors());
}

std::unique_ptr<DynamicsScheme> create_dynamics_scheme(const std::string& scheme_name)
{
    return registry.create("dynamics", scheme_name);
}

std::unique_ptr<DynamicsScheme> create_dynamics_scheme(
    const std::string& scheme_name, const SimulationStateView& state)
{
    const auto* descriptor = find_scheme_descriptor(get_dynamics_scheme_descriptors(), scheme_name);
    if (!descriptor || !descriptor->constructible)
        throw std::invalid_argument("unknown dynamics scheme: " + scheme_name);
    if (descriptor->id == kCartesianAGridDynamicsId)
        return std::make_unique<CartesianAGridScheme>(state);
    if (descriptor->id == kCartesianCGridDynamicsId)
        return std::make_unique<CartesianCGrid3DScheme>(state);
    throw std::invalid_argument("unsupported dynamics scheme: " + scheme_name);
}

// The unqualified v1 ID selects its implementation from grid staggering.
std::unique_ptr<DynamicsScheme>
create_dynamics_scheme(const tmv::init::config::ConfigV1& config)
{
    const std::string& scheme = config.dynamics.scheme;
    if (scheme != "klemp_wilhelmson_split")
    {
        return create_dynamics_scheme(scheme);
    }

    const bool cgrid = (config.grid.staggering == StaggerType::CGrid);
    return create_dynamics_scheme(cgrid ? kCartesianCGridDynamicsId
                                        : kCartesianAGridDynamicsId);
}

std::vector<std::string> get_available_dynamics_schemes()
{
    return registry.available_ids();
}

} // namespace tmv
