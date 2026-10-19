#pragma once

#include "turbulence/turbulence_base.hpp"

struct SimulationStateView;

namespace tmv
{
enum class TurbulencePerfStage
{
    SmagorinskyCoefficients,
    SmagorinskyMomentum,
    SmagorinskyTheta,
    SmagorinskyMoisture,
};

extern TurbulenceRuntimeState& global_turbulence_runtime_state;

// Compatibility bindings into the process-owned runtime.
extern std::unique_ptr<TurbulenceSchemeBase>& turbulence_scheme;
extern TurbulenceConfig& global_turbulence_config;

void initialize_turbulence(
    const std::string& scheme_name,
    const TurbulenceConfig& cfg = TurbulenceConfig{});
void initialize_turbulence(const SimulationStateView& execution_state,
    const std::string& scheme_name,
    const TurbulenceConfig& cfg = TurbulenceConfig{});
void initialize_turbulence(
    TurbulenceRuntimeState& runtime,
    const std::string& scheme_name,
    const TurbulenceConfig& cfg = TurbulenceConfig{});

// True only when the SGS tendency buffer was refreshed. Real Smagorinsky/TKE
// closures always refresh; cadence sampling is retained only for inactive or
// diagnostic/custom schemes and must not be used as an integration method.
bool step_turbulence(
    const SimulationStateView& state,
    double current_time,
    TurbulenceTendencies& tendencies);
bool step_turbulence(double current_time, TurbulenceTendencies& tendencies);

// Recompute from the live state regardless of the diagnostic cadence clock.
// Production explicit integration uses this after every accepted SGS
// substep; holding a nonlinear tendency across a changed state is unsafe.
void refresh_turbulence(
    const SimulationStateView& state,
    double current_time,
    TurbulenceTendencies& tendencies);

// Timing-only ownership for the production turbulence operators. Recording
// is inactive unless runtime performance timing is enabled.
void record_turbulence_perf(TurbulencePerfStage stage, double elapsed_s);
void reset_turbulence_perf_stats();
void log_turbulence_perf_summary();
}

using tmv::global_turbulence_runtime_state;
using tmv::turbulence_scheme;
using tmv::global_turbulence_config;
using tmv::initialize_turbulence;
using tmv::step_turbulence;
using tmv::refresh_turbulence;
using tmv::TurbulencePerfStage;
using tmv::record_turbulence_perf;
using tmv::reset_turbulence_perf_stats;
using tmv::log_turbulence_perf_summary;

