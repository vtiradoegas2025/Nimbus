#pragma once

// Public dynamics factory surface.

#include "dynamics/scheme_descriptor.hpp"

#include <memory>
#include <string>
#include <vector>

struct SimulationStateView;

namespace tmv
{
class DynamicsScheme;
namespace init::config { struct ConfigV1; }

std::unique_ptr<DynamicsScheme>
create_dynamics_scheme(const std::string& scheme_name);
std::unique_ptr<DynamicsScheme>
create_dynamics_scheme(const init::config::ConfigV1& config);
std::unique_ptr<DynamicsScheme> create_dynamics_scheme(
    const std::string& scheme_name, const SimulationStateView& state);
std::vector<std::string> get_available_dynamics_schemes();
} // namespace tmv

using tmv::create_dynamics_scheme;
using tmv::get_available_dynamics_schemes;
using tmv::get_dynamics_scheme_descriptors;
