#pragma once

#include "core/orchestration/dynamics/sgs_apply.hpp"

// Core-only half of the SGS integration contract.  The caller must have
// computed and checked the realized stability report for this exact live
// state/tendency pair.  Public direct callers use apply_sgs_tendencies(),
// which performs admission itself.
TurbulenceMomentumWorkReport apply_admitted_sgs_substep(
    const SimulationStateView& state,
    double dt_sgs,
    const TurbulenceTendencies& turb_tend);
