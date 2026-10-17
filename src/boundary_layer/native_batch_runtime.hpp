#pragma once

#include "boundary_layer/runtime.hpp"
#include "core/infra/physical_cell_domain.hpp"

class BoundaryLayerComputeKernels;
struct SimulationStateView;

namespace tmv
{

// References to the cadence transaction's candidate buffers. The executor
// fills these candidates but never commits them to SimulationStateView.
struct BoundaryLayerBatchPublication
{
    Field3D&                      dtheta;
    Field3D&                      dqv;
    Field3D&                      du;
    Field3D&                      dv;
    Field3D&                      dtke;
    Field3D&                      u_force;
    Field3D&                      v_force;
    Field3D&                      theta_rate;
    Field3D&                      vapor_rate;
    std::vector<float>&           ustar;
    std::vector<float>&           tau_u;
    std::vector<float>&           tau_v;
    std::vector<float>&           sensible;
    std::vector<float>&           latent;
    std::vector<float>&           moisture;
    std::vector<float>&           height;
    std::vector<float>&           skin_theta;
    std::vector<float>&           skin_qv;
    std::vector<SlabColumnState>& slab;
};

// Prepare and finalize columns in parallel around one serialized native
// submission per bounded chunk. Returns false only for a rejected dispatch;
// malformed science inputs continue to throw through the runtime contract.
bool evaluate_native_boundary_layer_batches(const SimulationStateView&    state,
                                            BoundaryLayerSchemeBase&      scheme,
                                            const BoundaryLayerConfig&    config,
                                            const SurfaceConfig&          surface_config,
                                            BoundaryLayerRefreshScratch&  scratch,
                                            const PhysicalCellDomain&     domain,
                                            bool                          stagger_x,
                                            bool                          stagger_y,
                                            bool                          capture_skin,
                                            double                        elapsed_time_s,
                                            double                        horizontal_cell_area,
                                            BoundaryLayerComputeKernels&  kernels,
                                            BoundaryLayerBatchPublication publication);

} // namespace tmv
