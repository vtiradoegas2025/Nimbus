#include "compute/acoustic_cgrid_dispatch_contract.hpp"
#include "compute/cartesian_scalar_capture.hpp"
#include "compute/kessler_sedimentation_contract.hpp"
#include "core/runtime/simulation_state.hpp"
#include "compute/thompson_sedimentation_contract.hpp"
#include "compute/thompson_cloud_contract.hpp"
#include "compute/thompson_warm_contract.hpp"
#include "compute/milbrandt_warm_contract.hpp"
#include "compute/milbrandt_cloud_contract.hpp"
#include "compute/rk_combination_contract.hpp"
#include "compute/rk_tendency_contract.hpp"
#include "compute/collocated_sgs_contract.hpp"
#include "compute/conservative_remap_contract.hpp"
#include "compute/earth_geometry_contract.hpp"
#include "compute/conservative_reduction_contract.hpp"
#include "compute/conservative_momentum_contract.hpp"
#include "compute/moving_momentum_contract.hpp"
#include "compute/moving_energy_contract.hpp"
#include "compute/reference_atmosphere_contract.hpp"
#include "compute/column_remap_contract.hpp"
#include "compute/held_sources_contract.hpp"
#include "compute/cgrid_boundary_contract.hpp"
#include "compute/simple_grey_contract.hpp"
#include "compute/slab_column_contract.hpp"
#include "compute/mapped_boundary_contract.hpp"
#include "compute/milbrandt_sedimentation_contract.hpp"
#include "compute/milbrandt_cold_contract.hpp"
#include "compute/milbrandt_melting_contract.hpp"
#include "compute/thompson_phase_contract.hpp"
#include "compute/thompson_freezing_contract.hpp"
#include "compute/thompson_deposition_contract.hpp"
#include "compute/thompson_completion_contract.hpp"
#include "compute/tke_dispatch_contract.hpp"
#include "compute/implicit_diffusion_contract.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute/metal/metal_compute_backend.hpp"
#include "compute/compute_backend_factory.hpp"

#include "compute/kernel_abi.hpp"
#include "compute/resident_field_store.hpp"
#include "core/infra/hardware_info.hpp"
#include "numerics/advection/advection_cartesian.hpp"

#include "internal/metal_source_catalog.hpp"
#include "internal/metal_pipeline_registry.hpp"
#include "internal/metal_telemetry.hpp"
#include "internal/metal_device_context.hpp"
#include "internal/metal_shared_allocations.hpp"
#include "internal/metal_resident_state.hpp"
#include "internal/metal_abi_contracts.hpp"
#include "internal/validation/common.hpp"
#include "internal/validation/domain.hpp"
#include "internal/validation/lin.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

// Keep the private Objective-C++ state and helpers in one translation unit,
// but divide implementation ownership by service/kernel family. This avoids
// exposing Metal objects in public headers while keeping each maintenance file
// small and purpose-specific.
namespace nimbus::metal
{

#include "backend/foundation.inc"
#include "backend/lifecycle.inc"
#include "backend/domain_recycling.inc"
#include "backend/chaos.inc"
#include "backend/advection.inc"
#include "backend/dynamics_acoustics.inc"
#include "backend/dynamics_slow.inc"
#include "backend/turbulence.inc"
#include "backend/microphysics_kessler.inc"
#include "backend/microphysics_lin.inc"
#include "backend/microphysics_thompson.inc"
#include "backend/microphysics_milbrandt.inc"
#include "backend/rk_combination.inc"
#include "backend/rk_tendency.inc"
#include "backend/collocated_sgs.inc"
#include "backend/conservative_remap.inc"
#include "backend/earth_geometry.inc"
#include "backend/conservative_reduction.inc"
#include "backend/conservative_momentum.inc"
#include "backend/moving_momentum.inc"
#include "backend/moving_energy.inc"
#include "backend/reference_atmosphere.inc"
#include "backend/column_remap.inc"
#include "backend/held_sources.inc"
#include "backend/cgrid_boundary.inc"
#include "backend/simple_grey.inc"
#include "backend/slab_column.inc"
#include "backend/diffusion.inc"
#include "backend/boundary_layer.inc"
#include "backend/radiation.inc"
#include "backend/terrain.inc"
#include "backend/services.inc"
#include "backend/shader_sources.inc"

} // namespace nimbus::metal

std::unique_ptr<ComputeBackend> create_metal_compute_backend()
{
    return std::make_unique<nimbus::metal::MetalComputeBackend>();
}
