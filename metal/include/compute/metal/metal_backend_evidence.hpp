#pragma once

/** Metal-only dispatch, benchmark, manifest, and qualification records. */

#include "compute/compute_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace nimbus::metal
{

// Metal-only qualification contract. The shared backend interface deliberately
// keeps the canonical Kessler phases separate; this wrapper proves their
// dependency-ordered, atomic execution without widening public configuration.
struct MetalKesslerFusedDispatch
{
    KesslerPointwiseDispatch    pointwise;
    KesslerSedimentationDispatch sedimentation;
};

struct MetalManifestCoverage
{
    std::string   canonical_manifest_hash;
    std::uint64_t manifest_kernels             = 0;
    std::uint64_t active_kernels               = 0;
    std::uint64_t ready_active_kernels         = 0;
    std::uint64_t hardware_limited_kernels     = 0;
    std::uint64_t deferred_kernels             = 0;
    std::uint64_t orphan_kernels               = 0;
    std::uint64_t unexpected_available_kernels = 0;
    std::uint64_t missing_artifact_hashes      = 0;
    bool          complete                     = false;
};

struct MetalStorageBenchmarkConfig
{
    int nr                = 128;
    int nth               = 128;
    int nz                = 32;
    int iterations        = 16;
    int warmups           = 2;
    int samples           = 7;
    int threadgroup_width = 64;
};

struct MetalStorageModeTiming
{
    double        median_device_milliseconds = 0.0;
    double        p95_device_milliseconds    = 0.0;
    double        median_wall_milliseconds   = 0.0;
    double        p95_wall_milliseconds      = 0.0;
    std::uint64_t gpu_upload_bytes           = 0;
    std::uint64_t gpu_download_bytes         = 0;
    std::uint64_t command_buffers            = 0;
};

struct MetalStorageBenchmarkResult
{
    std::size_t            field_bytes             = 0;
    std::size_t            private_heap_bytes      = 0;
    std::uint64_t          dispatches_per_sample   = 0;
    std::uint64_t          threadgroup_width       = 0;
    bool                   shared_bitwise_correct  = false;
    bool                   private_bitwise_correct = false;
    MetalStorageModeTiming shared;
    MetalStorageModeTiming private_heap;
};

struct MetalExecutionRecord
{
    std::string   api_name;
    std::string   device_name;
    std::string   operating_system;
    std::string   host_toolchain;
    std::string   msl_source_hash;
    std::string   smagorinsky_coefficients_msl_source_hash;
    std::string   smagorinsky_tendencies_msl_source_hash;
    std::string   sgs_apply_msl_source_hash;
    std::string   diffusion_msl_source_hash;
    std::string   kessler_pointwise_msl_source_hash;
    std::string   kessler_sedimentation_msl_source_hash;
    std::string   lin_pointwise_msl_source_hash;
    std::string   lin_sedimentation_msl_source_hash;
    std::string   microphysics_apply_msl_source_hash;
    std::string   advection_x_msl_source_hash;
    std::string   advection_y_msl_source_hash;
    std::string   tvd_vertical_flux_msl_source_hash;
    std::string   conservative_cgrid_transport_msl_source_hash;
    std::string   dynamics_apply_msl_source_hash;
    std::string   cartesian_cgrid_slow_tendencies_msl_source_hash;
    std::string   acoustic_cartesian_cgrid_msl_source_hash;
    std::string   acoustic_batched_cartesian_cgrid_msl_source_hash;
    std::string   cartesian_agrid_msl_source_hash;
    std::string   acoustic_agrid_msl_source_hash;
    std::string   acoustic_agrid_continuity_msl_source_hash;
    std::string   acoustic_agrid_pressure_eos_msl_source_hash;
    std::string   acoustic_agrid_native_momentum_msl_source_hash;
    std::string   acoustic_agrid_constraint_msl_source_hash;
    std::string   acoustic_agrid_boundary_msl_source_hash;
    std::string   canonical_kernel_manifest_hash;
    std::uint64_t manifest_kernels                                       = 0;
    std::uint64_t active_manifest_kernels                                = 0;
    std::uint64_t ready_active_manifest_kernels                          = 0;
    std::uint64_t hardware_limited_manifest_kernels                      = 0;
    std::uint64_t deferred_manifest_kernels                              = 0;
    std::uint64_t orphan_manifest_kernels                                = 0;
    std::uint64_t unexpected_available_manifest_kernels                  = 0;
    std::uint64_t missing_active_artifact_hashes                         = 0;
    bool          manifest_complete                                      = false;
    bool          unified_memory                                         = false;
    bool          timestamps_supported                          = false;
    std::uint64_t completed_command_buffers                     = 0;
    std::uint64_t completed_shift_dispatches                    = 0;
    std::uint64_t completed_shift_batches                       = 0;
    std::uint64_t resident_uploads                              = 0;
    std::uint64_t resident_reuses                               = 0;
    std::uint64_t completed_smagorinsky_coefficients_dispatches = 0;
    std::uint64_t completed_smagorinsky_tendencies_dispatches   = 0;
    std::uint64_t completed_smagorinsky_fused_dispatches        = 0;
    std::uint64_t completed_sgs_apply_dispatches                 = 0;
    std::uint64_t sgs_resident_tendency_reuses                   = 0;
    std::uint64_t completed_diffusion_dispatches                 = 0;
    std::uint64_t completed_diffusion_batches                    = 0;
    std::uint64_t completed_kessler_pointwise_dispatches          = 0;
    std::uint64_t completed_kessler_sedimentation_dispatches      = 0;
    std::uint64_t completed_kessler_fused_dispatches               = 0;
    std::uint64_t kessler_resident_tendency_writes                 = 0;
    std::uint64_t completed_lin_pointwise_dispatches                = 0;
    std::uint64_t completed_lin_sedimentation_dispatches            = 0;
    std::uint64_t completed_lin_fused_dispatches                     = 0;
    std::uint64_t lin_resident_tendency_writes                       = 0;
    std::uint64_t completed_microphysics_apply_dispatches           = 0;
    std::uint64_t microphysics_apply_resident_tendency_reuses       = 0;
    std::uint64_t completed_advection_x_dispatches                   = 0;
    std::uint64_t completed_advection_y_dispatches                   = 0;
    std::uint64_t completed_advection_pre_vertical_batches           = 0;
    std::uint64_t completed_advection_post_vertical_batches          = 0;
    std::uint64_t completed_vertical_flux_dispatches                 = 0;
    std::uint64_t completed_conservative_cgrid_batches               = 0;
    std::uint64_t completed_conservative_cgrid_scalars               = 0;
    std::uint64_t conservative_cgrid_resident_reuses                 = 0;
    std::uint64_t completed_state_relaxation_dispatches              = 0;
    std::uint64_t completed_slow_apply_dispatches                    = 0;
    std::uint64_t dynamics_state_resident_reuses                     = 0;
    std::uint64_t completed_cartesian_cgrid_slow_tendency_dispatches = 0;
    std::uint64_t slow_tendency_invariant_input_uploads               = 0;
    std::uint64_t slow_tendency_invariant_input_reuses                = 0;
    std::uint64_t slow_apply_resident_tendency_reuses                = 0;
    std::uint64_t completed_acoustic_pressure_dispatches              = 0;
    std::uint64_t completed_acoustic_momentum_dispatches              = 0;
    std::uint64_t completed_acoustic_fused_dispatches                 = 0;
    std::uint64_t acoustic_invariant_uploads                          = 0;
    std::uint64_t acoustic_invariant_reuses                           = 0;
    std::uint64_t completed_acoustic_batched_dispatches                = 0;
    std::uint64_t completed_acoustic_boundary_face_dispatches          = 0;
    std::uint64_t completed_acoustic_mass_flux_exports                 = 0;
    std::uint64_t completed_acoustic_mass_flux_pressure_dispatches      = 0;
    std::uint64_t completed_cartesian_agrid_tendency_dispatches          = 0;
    std::uint64_t completed_acoustic_agrid_pressure_dispatches           = 0;
    std::uint64_t completed_acoustic_agrid_momentum_dispatches           = 0;
    std::uint64_t completed_acoustic_agrid_fused_dispatches              = 0;
    std::uint64_t completed_acoustic_agrid_batched_dispatches            = 0;
    std::uint64_t completed_acoustic_agrid_continuity_dispatches         = 0;
    std::uint64_t completed_acoustic_agrid_pressure_eos_dispatches       = 0;
    std::uint64_t completed_acoustic_agrid_native_momentum_dispatches    = 0;
    std::uint64_t completed_acoustic_agrid_constraint_dispatches         = 0;
    std::uint64_t completed_acoustic_agrid_boundary_dispatches           = 0;
    double        last_device_milliseconds                      = 0.0;
    double        total_device_milliseconds                     = 0.0;
    double        total_submit_wait_milliseconds                = 0.0;
    std::uint64_t timestamped_command_buffers                   = 0;
};

} // namespace nimbus::metal
