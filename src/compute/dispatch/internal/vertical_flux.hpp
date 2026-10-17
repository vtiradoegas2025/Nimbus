#pragma once

#include <string>

#include "numerics/advection/advection_base.hpp"

namespace nimbus::compute::dispatch_internal
{

enum class TVDLimiterKind
{
    minmod,
    vanleer,
    superbee,
    mc,
    universal
};

std::string canonicalize_id(std::string value);

bool parse_tvd_limiter_kind(
    const std::string& limiter_id,
    TVDLimiterKind& out_kind);

bool dispatch_tvd_vertical_flux_v1(
    const AdvectionConfig& cfg,
    const AdvectionStateView& state,
    AdvectionTendencies& tendencies,
    AdvectionDiagnostics* diag_opt);

bool dispatch_tvd_vertical_flux_v2(
    const AdvectionConfig& cfg,
    const AdvectionStateView& state,
    AdvectionTendencies& tendencies,
    AdvectionDiagnostics* diag_opt);

} // namespace nimbus::compute::dispatch_internal
