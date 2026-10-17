#pragma once

#include "radiation/radiation_base.hpp"

#include <string>

struct SimulationStateView;

namespace tmv
{
extern RadiationRuntimeState& global_radiation_runtime_state;

// Compatibility bindings into the process-owned runtime.
extern std::unique_ptr<RadiationSchemeBase>& radiation_scheme;
extern RadiationConfig& global_radiation_config;

void initialize_radiation(
    const std::string& scheme_name,
    const RadiationConfig& cfg = RadiationConfig{});
void initialize_radiation(const SimulationStateView& execution_state,
    const std::string& scheme_name,
    const RadiationConfig& cfg = RadiationConfig{});
void initialize_radiation(
    RadiationRuntimeState& runtime,
    const std::string& scheme_name,
    const RadiationConfig& cfg = RadiationConfig{});

// Terrain supports the physical-height, horizontal-area simple-grey longwave
// column. Shortwave remains guarded until slope-relative incidence and
// topographic shadow/transport are implemented.
void step_radiation(const SimulationStateView& state, double current_time);
void step_radiation(double current_time);

// Commit one successful shared-theta application to the held radiative
// boundary-power ledger. This is deliberately called only after every strict
// thermodynamic guard has accepted the prognostic update.
void commit_radiation_applied_heat(const SimulationStateView& state,
                                   double dt_s,
                                   double applied_radiation_heat_j);

void open_radiation_budget_csv(const std::string& path);
void close_radiation_budget_csv();
}

using tmv::global_radiation_runtime_state;
using tmv::radiation_scheme;
using tmv::global_radiation_config;
using tmv::initialize_radiation;
using tmv::step_radiation;
using tmv::commit_radiation_applied_heat;
using tmv::open_radiation_budget_csv;
using tmv::close_radiation_budget_csv;
