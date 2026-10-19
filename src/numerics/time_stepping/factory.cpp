#include "numerics/time_stepping/factory.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "schemes/rk3/rk3.hpp"
#include "schemes/rk4/rk4.hpp"
#include "schemes/split_explicit/split_explicit.hpp"
#include "util/scheme_factory.hpp"

#include <stdexcept>

namespace tmv
{

namespace
{
const SchemeRegistry<TimeSteppingSchemeBase> registry({
    {"rk3", [] { return std::make_unique<RK3Scheme>(); }},
    {"rk4", [] { return std::make_unique<RK4Scheme>(); }},
    {"split_explicit", [] { return std::make_unique<SplitExplicitScheme>(); }},
}, get_time_stepping_scheme_descriptors());
}

std::unique_ptr<TimeSteppingSchemeBase> create_time_stepping_scheme(const std::string& scheme_name)
{
    return registry.create("time stepping", scheme_name);
}

void validate_runtime_time_stepping_contract(
    const TimeSteppingConfig& config,
    const TimeSteppingSchemeBase& scheme)
{
    if (!config.split_acoustic)
    {
        if (!scheme.supports_coupled_unsplit_rk())
        {
            throw std::runtime_error(
                "time-stepping scheme '" + scheme.name() +
                "' does not implement coupled unsplit Runge-Kutta stepping");
        }
        return;
    }
    if (!scheme.supports_split_acoustic())
    {
        throw std::runtime_error(
            "time-stepping scheme '" + scheme.name() +
            "' does not implement split-explicit acoustic stepping, but "
            "split_acoustic=true. Select scheme=split_explicit.");
    }
}

std::vector<std::string> get_available_time_stepping_schemes()
{
    return registry.available_ids();
}

} // namespace tmv
