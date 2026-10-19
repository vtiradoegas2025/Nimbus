/**
 * @file device_slow_energy_attribution.hpp
 * @brief Energy attribution for the completed device slow-dynamics commit.
 *
 * The slow-apply shader owns the FP32 velocity and pressure commit.  Its
 * observer receives the pre-commit host state and the authoritative
 * post-fence device buffers, allowing energy attribution without replaying the
 * writer or downloading additional snapshots.
 */

#pragma once

#include "boundary_layer/runtime.hpp"
#include "dynamics/slow_momentum_energy_report.hpp"

struct SimulationStateView;

namespace tmv::device_slow_energy
{

struct AttributionResult
{
    CartesianSlowMomentumEnergyReportV1 momentum;
    BoundaryLayerMomentumWorkReport pbl;
};

// Validate everything that can be checked before the device mutates state.
// The same validation is repeated by attribute_completed_commit so direct
// callers cannot bypass the contract.
void validate_request(
    const SimulationStateView& state,
    const Field3D& du_dt,
    const Field3D& dv_dt,
    const Field3D& dw_dt,
    const SlowMomentumAdvectionTendencyOutputs* advection,
    float dt_large,
    bool attribute_momentum,
    bool attribute_pbl);

// Attribute the authoritative completed device endpoint.  wrap_x/wrap_y
// describe ownership of the final periodic C-grid u/v faces; they are passed
// explicitly so this function is simulation-local rather than process-global.
[[nodiscard]] AttributionResult attribute_completed_commit(
    const SimulationStateView& state,
    const Field3D& du_dt,
    const Field3D& dv_dt,
    const Field3D& dw_dt,
    const SlowMomentumAdvectionTendencyOutputs* advection,
    float dt_large,
    bool attribute_momentum,
    bool attribute_pbl,
    bool wrap_x,
    bool wrap_y,
    const float* u_committed,
    const float* v_committed,
    const float* w_committed,
    const float* p_committed);

} // namespace tmv::device_slow_energy
