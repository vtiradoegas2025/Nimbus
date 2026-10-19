#pragma once

#include "diagnostics/pressure_projection.hpp"

struct SimulationStateView;

namespace tmv::diagnostics
{
struct ResolvedDiagnosticRequest
{
    bool vorticity             = false;
    bool angular_momentum      = false;
    bool pressure_perturbation = false;
    bool pressure_components   = false;
};

struct ResolvedDiagnosticReport
{
    PressureProjectionReport dynamic_pressure;
    PressureProjectionReport buoyancy_pressure;
};

void compute_resolved_diagnostics(const SimulationStateView&       state,
                                  const ResolvedDiagnosticRequest& request,
                                  ResolvedDiagnosticReport*        report = nullptr);
} // namespace tmv::diagnostics
