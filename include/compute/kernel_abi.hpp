#pragma once

/**
 * @file kernel_abi.hpp
 * @brief Backend-neutral constants layouts shared by Vulkan and native Metal.
 *
 * These standard-layout fp32 blocks are the host half of KernelManifest's
 * constants ABI. Shader-language declarations must match their checked sizes.
 */

#include <cstdint>

namespace nimbus::compute::abi
{

struct MappedTerrainMassContinuityPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t stage;
    float dx;
    float dy;
    float dzeta;
    float gamma;
    float frame_u_ms;
    float frame_v_ms;
    float padding[6];
};
static_assert(sizeof(MappedTerrainMassContinuityPushConstants) == 64,
              "mapped terrain continuity constants must be 64 bytes");

struct MappedTerrainFastMomentumPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t sampled_reference;
    float dx;
    float dy;
    float nominal_dz;
    float padding1;
};
static_assert(sizeof(MappedTerrainFastMomentumPushConstants) == 32,
              "mapped terrain fast-momentum constants must be 32 bytes");

struct MappedTerrainSlowTendenciesPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t has_moisture;
    int32_t has_loading;
    int32_t sampled_reference;
    int32_t padding_i[2];
    float dx;
    float dy;
    float dzeta;
    float nominal_dz;
    float gravity;
    float padding_f[3];
};
static_assert(sizeof(MappedTerrainSlowTendenciesPushConstants) == 64,
              "mapped terrain slow-tendency constants must be 64 bytes");

struct MappedTerrainScalarUpdatePushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t positive_definite;
    float inv_dx;
    float inv_dy;
    float inv_dzeta;
    float dt;
    float density_floor;
    float padding[3];
};
static_assert(sizeof(MappedTerrainScalarUpdatePushConstants) == 48,
              "mapped terrain scalar-update constants must be 48 bytes");

struct MappedTerrainScalarLimiterPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t padding0;
    float inv_dx;
    float inv_dy;
    float inv_dzeta;
    float dt;
    float padding[4];
};
static_assert(sizeof(MappedTerrainScalarLimiterPushConstants) == 48,
              "mapped terrain scalar-limiter constants must be 48 bytes");

struct MappedTerrainEosPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t padding0;
};
static_assert(sizeof(MappedTerrainEosPushConstants) == 16,
              "mapped terrain EOS constants must be 16 bytes");

struct MappedTerrainBoundaryPushConstants
{
    int32_t nx, ny, nz;
    int32_t west, east, south, north, bottom, top;
    float frame_u, frame_v;
    int32_t padding;
};
static_assert(sizeof(MappedTerrainBoundaryPushConstants) == 48,
              "mapped terrain boundary constants must be 48 bytes");

struct MappedTerrainStateUpdatePushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    uint32_t mode;
    float dt;
    float padding[3];
};
static_assert(sizeof(MappedTerrainStateUpdatePushConstants) == 32,
              "mapped terrain state-update constants must be 32 bytes");

struct MappedTerrainFluxAccumulationPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t padding0;
    float time_weight;
    float padding[3];
};
static_assert(sizeof(MappedTerrainFluxAccumulationPushConstants) == 32,
              "mapped terrain flux-accumulation constants must be 32 bytes");

struct TvdPushConstants
{
    int32_t nr;
    int32_t nth;
    int32_t nz;
    int32_t limiter_id;
    int32_t positivity;
    float   positivity_dt;
    float   cfl_target;
    float   padding;
};
static_assert(sizeof(TvdPushConstants) == 32, "TVD push constants must be 32 bytes");

struct VerticalAdvectionCgridPushConstants
{
    int32_t nr;
    int32_t nth;
    int32_t nz;
    float   dz_val;
    float   dt;
    float   padding[3];
};
static_assert(sizeof(VerticalAdvectionCgridPushConstants) == 32,
              "vertical c-grid advection push constants must be 32 bytes");

struct DiffusionPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t location;
    int32_t c_grid;
    int32_t periodic_x;
    int32_t periodic_y;
    int32_t padding_i;
    float   inv_dx2;
    float   inv_dy2;
    float   K_h;
    float   K_v;
    float   padding[4];
};

struct HeldSourcesPushConstants
{
    int nx,ny,nz,i_begin,i_end,j_begin,j_end,k_begin,k_end;
    int staggered,periodic_x,periodic_y,pbl_enabled,radiation_enabled,cloud_enabled;
    float cp_high,cp_low,p0_high,p0_low,kappa_high,kappa_low;
};
static_assert(sizeof(HeldSourcesPushConstants)==84);

struct SlabColumnPushConstants
{
    int columns,layers,top_method,surface_method,apply_flux,padding0,padding1,padding2;
};
static_assert(sizeof(SlabColumnPushConstants)==32);

struct SimpleGreyPushConstants
{
    int columns,layers,longwave,shortwave,water,padding0,padding1,padding2;
};
static_assert(sizeof(SimpleGreyPushConstants)==32);

struct ColumnRemapPushConstants
{
    int nx,ny,nz,mode;
    float sx_high,sx_low,sy_high,sy_low;
};
static_assert(sizeof(ColumnRemapPushConstants)==32);

struct ReferenceAtmospherePushConstants
{
    int count, levels;
    float dz_high, dz_low, rd_high, rd_low, p0_high, p0_low, kappa_high, kappa_low;
    int theta_exponent, pressure_exponent, vapor_exponent, u_exponent, v_exponent, padding;
};
static_assert(sizeof(ReferenceAtmospherePushConstants) == 64);

struct MovingEnergyPushConstants
{
    int nx,ny,nz,operation;
    float rd_high,rd_low;
    float cv_high,cv_low;
    float cvv_high,cvv_low;
    float cl_high,cl_low;
    float ci_high,ci_low;
    float tref_high,tref_low;
    float ev_high,ev_low;
    float ei_high,ei_low;
    float gravity_high,gravity_low;
    float p0_high,p0_low;
    float kappa_high,kappa_low;
    float gamma_high,gamma_low;
    float theta_min_high,theta_min_low;
    float theta_max_high,theta_max_low;
};
static_assert(sizeof(MovingEnergyPushConstants)==128);

struct ConservativeReductionPushConstants
{
    int rows, terms;
};
static_assert(sizeof(ConservativeReductionPushConstants) == 8);

struct ConservativeMomentumPushConstants
{
    int nx, ny, nz, operation;
    int i_begin, i_end, j_begin, j_end, k_begin, k_end;
    float dx_high, dx_low, dy_high, dy_low, dz_high, dz_low;
    float gravity_high, gravity_low, coriolis_high, coriolis_low;
};
static_assert(sizeof(ConservativeMomentumPushConstants) == 80);

struct MovingMomentumPushConstants
{
    int   nx, ny, nz, component, include_boundary, operation;
    float dx_high, dx_low, dy_high, dy_low, dz_high, dz_low;
    float surface_u_high, surface_u_low, surface_v_high, surface_v_low;
};
static_assert(sizeof(MovingMomentumPushConstants) == 64);

struct EarthGeometryPushConstants
{
    int   nx, ny, nz, operation, scheme, smoothed;
    float dx_high, dx_low;
    float dy_high, dy_low;
    float dz_high, dz_low;
    float height_high, height_low;
    float width_high, width_low;
    float wavelength_high, wavelength_low;
    float decay_high, decay_low;
    float x_high, x_low;
    float y_high, y_low;
    float shift_x_high, shift_x_low;
    float shift_y_high, shift_y_low;
};
static_assert(sizeof(EarthGeometryPushConstants) == 112);

struct ConservativeRemapPushConstants
{
    int nx, ny, nz, mixing_ratio, store_float, lower_bound;
};
static_assert(sizeof(ConservativeRemapPushConstants) == 24);

struct RkTendencyPushConstants
{
    int   nx, ny, nz, operation;
    float idx_high, idx_low, idy_high, idy_low, idz_high, idz_low;
    float step_high, step_low, gamma_high, gamma_low;
    int   flags, padding;
};
static_assert(sizeof(RkTendencyPushConstants) == 64);

struct RkCombinationPushConstants
{
    int   cells, stages, ssp, conserved;
    float dt_high, dt_low;
    float weight0_high, weight0_low, weight1_high, weight1_low;
    float weight2_high, weight2_low, weight3_high, weight3_low;
    float fraction_high, fraction_low;
};
static_assert(sizeof(RkCombinationPushConstants) == 64);

struct MilbrandtSedimentationPushConstants
{
    int32_t nx, ny, nz;
    int32_t i_begin, i_end, j_begin, j_end, k_begin, k_end;
    int32_t maximum_substeps;
    float   dt_high, dt_low;
};
static_assert(sizeof(MilbrandtSedimentationPushConstants) == 48);

struct MilbrandtColdPushConstants
{
    int32_t nx, ny, nz;
    int32_t i_begin, i_end, j_begin, j_end, k_begin, k_end;
    int32_t hail_processes;
    float   dt_high, dt_low;
};
static_assert(sizeof(MilbrandtColdPushConstants) == 48);

struct MilbrandtCloudAdjustmentPushConstants
{
    int32_t nx, ny, nz;
    int32_t i_begin, i_end, j_begin, j_end, k_begin, k_end;
    int32_t face_staggered_velocity; // 0: scalar, 1: flat faces, 2: faces with a paired-FP32 ground tail.
    float   dt_high, dt_low;
};
static_assert(sizeof(MilbrandtCloudAdjustmentPushConstants) == 48, "Milbrandt cloud ABI must be 48 bytes");

struct ThompsonCloudAdjustmentPushConstants
{
    int32_t nx, ny, nz;
    int32_t i_begin, i_end, j_begin, j_end, k_begin, k_end;
    float   dt_high, dt_low;
};
static_assert(sizeof(ThompsonCloudAdjustmentPushConstants) == 44, "Thompson cloud adjustment ABI must be 44 bytes");

struct ThompsonCloudSourcePushConstants
{
    int32_t nx, ny, nz;
    int32_t i_begin, i_end, j_begin, j_end, k_begin, k_end;
    float   dt_high, dt_low;
    int32_t input_flags; // Bit 0: potential temperature; bit 1: timestep is exactly FP32.
};
static_assert(sizeof(ThompsonCloudSourcePushConstants) == 48, "Thompson cloud source ABI must be 48 bytes");

struct ThompsonMomentSedimentationPushConstants
{
    int32_t nx, ny, nz;
    int32_t i_begin, i_end, j_begin, j_end, k_begin, k_end;
    int32_t species, maximum_substeps;
    float   dt;
};
static_assert(sizeof(ThompsonMomentSedimentationPushConstants) == 48, "Thompson sedimentation ABI must be 48 bytes");

struct ImplicitDiffusionStencilPushConstants
{
    int32_t nx, ny, nz, location;
    int32_t c_grid, periodic_x, periodic_y, use_variable_K;
    float   inv_dx2, inv_dy2, K_h, K_v;
    float   dt;
    int32_t form_rhs;
    float   field_scale, density_scale, thickness_scale, padding0, padding1, padding2;
};
static_assert(sizeof(ImplicitDiffusionStencilPushConstants) == 80, "implicit stencil ABI must be 80 bytes");

struct ImplicitDiffusionVectorPushConstants
{
    int32_t nx, ny, nz, stage;
    float   alpha, beta, dt, field_scale;
};
static_assert(sizeof(ImplicitDiffusionVectorPushConstants) == 32, "implicit vector ABI must be 32 bytes");

struct CartesianTendenciesPushConstants
{
    int32_t nr;
    int32_t nth;
    int32_t nz;
    float   dx;
    float   dy;
    float   dz;
    float   g;
    float   gamma_val;
    float   coriolis_f;
    float   coriolis_beta;
    int32_t lateral_bc_west;
    int32_t lateral_bc_east;
    int32_t lateral_bc_south;
    int32_t lateral_bc_north;
    int32_t top_bc;
    int32_t has_moisture;
};
static_assert(sizeof(CartesianTendenciesPushConstants) == 64,
              "cartesian push constants must be 64 bytes");

struct AdvectionXPushConstants
{
    int32_t nr;
    int32_t nth;
    int32_t nz;
    float   dx;
    float   dt;
    float   padding[3];
};
static_assert(sizeof(AdvectionXPushConstants) == 32, "advection-x push constants must be 32 bytes");

struct AdvectionYPushConstants
{
    int32_t nr;
    int32_t nth;
    int32_t nz;
    float   dy;
    float   dt;
    float   padding[3];
};
static_assert(sizeof(AdvectionYPushConstants) == 32, "advection-y push constants must be 32 bytes");

struct TornadoCgridSlowTendenciesPushConstants
{
    int32_t nr;
    int32_t nth;
    int32_t nz;
    float   dr;
    float   dz_val;
    float   g;
    int32_t has_moisture;
    int32_t has_loading;
    float   padding0;
    float   padding1;
};
static_assert(sizeof(TornadoCgridSlowTendenciesPushConstants) == 40,
              "tornado_cgrid_slow push constants must be 40 bytes");

struct SupercellCgridSlowTendenciesPushConstants
{
    int32_t nr;
    int32_t nth;
    int32_t nz;
    float   dr;
    float   dtheta;
    float   dz_val;
    float   g;
    int32_t has_moisture;
    int32_t has_loading;
    float   padding0;
    float   padding1;
    float   coriolis_beta;
    float   frame_velocity_u_ms;
    float   frame_velocity_v_ms;
};
static_assert(sizeof(SupercellCgridSlowTendenciesPushConstants) == 56,
              "supercell_cgrid_slow push constants must be 56 bytes");

static_assert(sizeof(DiffusionPushConstants) == 64, "diffusion push constants must be 64 bytes");

struct KesslerPointwisePushConstants
{
    int32_t nr;
    int32_t nth;
    int32_t nz;
    int32_t i_begin;
    int32_t i_end;
    int32_t j_begin;
    int32_t j_end;
    int32_t k_begin;
    int32_t k_end;
    float   qc0;
    float   c_auto;
    float   c_accr;
    float   evaporation_efficiency;
    float   Lv_cp;
    float   T0;
    float   dt_step;
};
static_assert(sizeof(KesslerPointwisePushConstants) == 64,
              "kessler pointwise push constants must be 64 bytes");

struct KesslerSedimentationPushConstants
{
    int32_t nr;
    int32_t nth;
    int32_t nz;
    int32_t i_begin;
    int32_t i_end;
    int32_t j_begin;
    int32_t j_end;
    int32_t k_begin;
    int32_t k_end;
    float   dz_val;
    float   Vt_max_rain;
    float   vt_cap;
    int32_t have_rho_d;
    float   dt_step;
    int32_t have_layer_thickness;
    float   pad2;
};
static_assert(sizeof(KesslerSedimentationPushConstants) == 64,
              "kessler sedimentation push constants must be 64 bytes");

// Archived ABIs for the orphaned Thompson shaders. They remain represented so
// manifest validation can detect metadata drift without making those kernels
// dispatchable.
struct ArchivedLegacyThompsonLinPointwisePushConstants
{
    int32_t nr;
    int32_t nth;
    int32_t nz;
    float qc0;
    float c_auto;
    float c_evap;
    float c_dep;
    float c_subl;
    float c_melt;
    float Lv_cp;
    float Lf_cp;
    float Ls_cp;
    float T0;
    float ccn_conc;
    float in_conc;
    float padding;
};
static_assert(sizeof(ArchivedLegacyThompsonLinPointwisePushConstants) == 64,
              "archived legacy pointwise push constants must be 64 bytes");

struct ArchivedLegacyThompsonLinSedimentationPushConstants
{
    int32_t nr;
    int32_t nth;
    int32_t nz;
    float dz_val;
    float a_rain;
    float b_rain;
    float Vt_max_rain;
    float a_snow;
    float b_snow;
    float Vt_max_snow;
    float a_graupel;
    float b_graupel;
    float Vt_max_graupel;
    float a_hail;
    float b_hail;
    float Vt_max_hail;
};
static_assert(sizeof(ArchivedLegacyThompsonLinSedimentationPushConstants) == 64,
              "archived legacy sedimentation push constants must be 64 bytes");

struct AcousticPressurePushConstants
{
    int32_t nr;
    int32_t nth;
    int32_t nz;
    float   dr;
    float   dtheta;
    float   dz_val;
    float   gamma_val;
    float   dt_small;
    float   rho_floor;
    float   p_floor;
};
static_assert(sizeof(AcousticPressurePushConstants) == 40,
              "acoustic pressure push constants must be 40 bytes");

struct AcousticMomentumPushConstants
{
    int32_t nr;
    int32_t nth;
    int32_t nz;
    float   dr;
    float   dtheta;
    float   dz_val;
    float   dt_small;
    float   wind_clamp_h;
    float   wind_clamp_v;
    float   padding;
};
static_assert(sizeof(AcousticMomentumPushConstants) == 40,
              "acoustic momentum push constants must be 40 bytes");

// Cartesian C-grid acoustic substep (GPU audit Phase 3.1). The
// pressure pass carries the full clamp bounds because it fuses the
// Forward-Euler integration + sanitization + diagnostic closure.
struct AcousticPressureCartesianCgridPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float   dx;
    float   dy;
    float   dz_val;
    float   gamma_val;
    float   dt_small;
    float   rho_min;
    float   rho_max;
    float   p_min;
    float   p_max;
    float   p_ref_fallback;
    float   pad0;
    float   frame_velocity_u_ms;
    float   frame_velocity_v_ms;
    int32_t preserve_unrepaired;
    float gamma_low;
};
static_assert(sizeof(AcousticPressureCartesianCgridPushConstants) == 72,
              "cartesian c-grid acoustic pressure push constants must be 72 bytes");

struct AcousticMomentumCartesianCgridPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float   dx;
    float   dy;
    float   dz_val;
    float   dt_small;
    float   wind_clamp_h;
    float   wind_clamp_v;
    int32_t preserve_unrepaired;
    int32_t paired_base_gradient;
};
static_assert(sizeof(AcousticMomentumCartesianCgridPushConstants) == 44,
              "cartesian c-grid acoustic momentum push constants must be 44 bytes");

// Production collocated continuity with momentum-weighted pressure coupling
// and exact instantaneous face-flux export. Kept separate from the legacy
// 40-byte A-grid acoustic ABI so existing artifacts remain stable.
struct AcousticContinuityMassFluxCartesianAgridPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float   dx;
    float   dy;
    float   dz;
    float   dt_small;
    int32_t lateral_bc_west;
    int32_t lateral_bc_east;
    int32_t lateral_bc_south;
    int32_t lateral_bc_north;
    int32_t top_bc;
    int32_t accumulate_mass_flux;
    float   mass_flux_weight;
    float   frame_velocity_u_ms;
    float   frame_velocity_v_ms;
};
static_assert(sizeof(AcousticContinuityMassFluxCartesianAgridPushConstants) == 64,
              "cartesian A-grid continuity/mass-flux constants must be 64 bytes");

struct AcousticPressureEosCartesianAgridPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float   dt_small;
    float   reference_pressure;
    float   gas_constant_dry_air;
    float   gamma;
    float   rho_min;
    float   rho_max;
    float   p_min;
    float   p_max;
    float   padding;
};
static_assert(sizeof(AcousticPressureEosCartesianAgridPushConstants) == 48,
              "cartesian A-grid pressure/EOS constants must be 48 bytes");

struct AcousticMomentumCartesianAgridPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float   dx;
    float   dy;
    float   dz;
    float   dt_small;
    float   wind_clamp_h;
    float   wind_clamp_v;
    float   padding;
};
static_assert(sizeof(AcousticMomentumCartesianAgridPushConstants) == 40,
              "cartesian A-grid momentum constants must be 40 bytes");

struct AcousticLinePassCartesianAgridPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t pass;
};
static_assert(sizeof(AcousticLinePassCartesianAgridPushConstants) == 16,
              "cartesian A-grid line-pass constants must be 16 bytes");

// Conservative scalar-transport batch (GPU audit Phase 3.2). The
// mass-flux and scalar-flux passes need only the grid dims; the update
// pass carries the fp64-divided-then-rounded inverse spacings (the
// dp0_dz lesson, section 2.3); the diffusion pass carries the squared
// inverses and the per-scalar pre-multiplied dt*kappa.
struct AdvectFluxDimsCartesianCgridPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t pad0;
};
static_assert(sizeof(AdvectFluxDimsCartesianCgridPushConstants) == 16,
              "advect flux dims push constants must be 16 bytes");

// The standalone mass-flux producer needs the moving-grid velocity while the
// scalar-flux producer above does not. Keeping separate ABIs avoids widening
// every per-scalar dispatch merely to carry two mass-flux-only values.
struct AdvectMassFluxCartesianCgridPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t pad0;
    float   frame_velocity_u_ms;
    float   frame_velocity_v_ms;
};
static_assert(sizeof(AdvectMassFluxCartesianCgridPushConstants) == 24,
              "advect mass-flux push constants must be 24 bytes");

// Positive-definite flux renormalization (SCIENCE.md M5).
struct AdvectPdLimiterPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t use_coupled_mass;
    float   inv_dx;
    float   inv_dy;
    float   inv_dz;
    float   dt_step;
};
static_assert(sizeof(AdvectPdLimiterPushConstants) == 32,
              "pd limiter push constants must be 32 bytes");

struct AdvectUpdateCartesianCgridPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t use_pd;
    float   inv_dx;
    float   inv_dy;
    float   inv_dz;
    float   dt_step;
    float   rho_floor;
    int32_t use_coupled_mass;
    int32_t capture_offset;
    int32_t capture_i_begin;
    int32_t capture_i_end;
    int32_t capture_j_begin;
    int32_t capture_j_end;
    int32_t capture_k_begin;
    int32_t capture_k_end;
    int32_t report_offset;
    int32_t weno;
};
static_assert(sizeof(AdvectUpdateCartesianCgridPushConstants) == 76,
              "advect update push constants must be 76 bytes");

// Archived ABI for the orphaned, non-dispatchable transport point filter.
struct DiffuseCartesianPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t pad0;
    float   inv_dx2;
    float   inv_dy2;
    float   inv_dz2;
    float   dt_kappa;
};
static_assert(sizeof(DiffuseCartesianPushConstants) == 32, "cartesian diffuse push constants must be 32 bytes");

// Lin microphysics (GPU audit Phase 3.3). dt and inv_dt arrive
// separately, each rounded once from fp64 on the host.
struct LinPointwisePushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t pad0;
    float   dt_step;
    float   inv_dt;
    float   qc0;
    float   c_auto;
    float   c_accr;
    float   c_evap;
    float   c_ihom;
    float   c_rime;
    float   c_agg;
    float   c_melt_s;
    float   c_melt_g;
    float   c_melt_h;
    float   c_subl;
    float   Lv_cp;
    float   Lf_cp;
    float   Ls_cp;
    int32_t i_begin;
    int32_t i_end;
    int32_t j_begin;
    int32_t j_end;
    int32_t k_begin;
    int32_t k_end;
    int32_t pad_tail0;
    int32_t pad_tail1;
};
static_assert(sizeof(LinPointwisePushConstants) == 112, "Lin compute ABI must be 112 bytes");

struct LinSedimentationPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float   dz_val;
    float   vt_cap;
    int32_t have_rho_d;
    int32_t have_layer_thickness;
    int32_t pad1;
    int32_t i_begin;
    int32_t i_end;
    int32_t j_begin;
    int32_t j_end;
    int32_t k_begin;
    int32_t k_end;
    int32_t pad_tail0;
    int32_t pad_tail1;
};
static_assert(sizeof(LinSedimentationPushConstants) == 64, "Lin compute ABI must be 64 bytes");

struct CollocatedSgsPushConstants
{
    int   nx, ny, nz, phase, flags, filter_mode;
    float dx, dy, Cs, Pr, Sc, nu_max, K_max;
    int   padding0, padding1, padding2;
};
static_assert(sizeof(CollocatedSgsPushConstants) == 64);

struct TkeClosurePushConstants
{
    int32_t nx, ny, nz, filter_mode;
    int32_t use_horizontal, use_vertical, periodic_x, periodic_y;
    float dx, dy, Pr_t, Sc_t, nu_max, K_max;
    int32_t stability, moist_buoyancy;
};
static_assert(sizeof(TkeClosurePushConstants) == 64, "TKE closure constants must be 64 bytes");

struct TkeTendencyPushConstants
{
    int32_t nx, ny, nz, use_horizontal;
    int32_t use_vertical, periodic_x, periodic_y, closed_lateral;
    float dx, dy;
};
static_assert(sizeof(TkeTendencyPushConstants) == 40, "TKE tendency constants must be 40 bytes");

// Smagorinsky turbulence (GPU audit Phase 3.4).
struct SmagorinskyCoefficientsPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t filter_mode;
    int32_t use_horizontal;
    int32_t use_vertical;
    float   dx;
    float   dy;
    float   Cs;
    float   Pr_t;
    float   Sc_t;
    float   nu_t_max;
    float   K_max;
    int32_t periodic_y;  // j-topology (SCIENCE.md M16)
    int32_t periodic_x;
};
static_assert(sizeof(SmagorinskyCoefficientsPushConstants) == 60,
              "smagorinsky coefficients push constants must be 60 bytes");

struct SmagorinskyTendenciesPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t use_horizontal;
    int32_t use_vertical;
    int32_t has_qv;
    int32_t periodic_x;
    int32_t periodic_y;  // j-topology (SCIENCE.md M16)
    float   dx;
    float   dy;
    int32_t closed_lateral;
};
static_assert(sizeof(SmagorinskyTendenciesPushConstants) == 44,
              "smagorinsky tendencies push constants must be 44 bytes");

// Per-face acoustic boundary conditions (GPU audit Phase 4.1).
struct AcousticBcFacePushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t face;
    int32_t bc_west;
    int32_t bc_east;
    int32_t bc_south;
    int32_t bc_north;
    int32_t bc_bottom;
    int32_t bc_top;
    float   frame_velocity_u_ms;
    float   frame_velocity_v_ms;
    float   dt_small;
    float   inv_dx;
    float   inv_dy;
    float   pad2;
};

struct AcousticRetainedFluxFp64PushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t pad;
    float frame_u_ms;
    float frame_v_ms;
    uint32_t weight_low;
    uint32_t weight_high;
};
static_assert(sizeof(AcousticRetainedFluxFp64PushConstants) == 32);
static_assert(sizeof(AcousticBcFacePushConstants) == 64,
              "acoustic bc face push constants must be 64 bytes");

// Microphysics tendency application (GPU audit Phase 4.2).
struct MicrophysicsApplyPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t pad0;
    int32_t i_begin;
    int32_t i_end;
    int32_t j_begin;
    int32_t j_end;
    float   dt_step;
    float   theta_fallback;
    float   theta_min;
    float   theta_max;
    float   max_theta_step;
    float   qv_max;
    float   hydro_max;
    float   pad1;
};
static_assert(sizeof(MicrophysicsApplyPushConstants) == 64,
              "microphysics apply push constants must be 64 bytes");

// SGS turbulence tendency application (GPU audit Phase 4.2).
struct SgsApplyPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t has_qv;
    int32_t has_tke;
    float   dt_step;
    float   theta_fallback;
    float   theta_min;
    float   theta_max;
    float   qv_max;
    float   wind_h_max;
    float   wind_v_max;
    int32_t sanitize;
    int32_t collocated;
};
static_assert(sizeof(SgsApplyPushConstants) == 56, "sgs apply push constants must be 56 bytes");

// Slow-tendency application in the split-explicit step
// (GPU audit Phase 4.2).
struct SlowApplyPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t sanitize;
    float   dt_large;
    float   pad1;
    float   pad2;
    float   pad3;
};
static_assert(sizeof(SlowApplyPushConstants) == 32,
              "slow apply push constants must be 32 bytes");

// Fused end-of-step sponge relaxation. The retired EOS slots remain reserved
// so older cached pipeline layouts cannot be mistaken for a new ABI.
struct StateRelaxationPushConstants
{
    int32_t nx;
    int32_t ny;
    int32_t nz;
    int32_t lateral_width;
    int32_t reserved_zero;
    int32_t do_upper;
    int32_t do_w_only;
    int32_t do_lateral;
    float   dt_step;
    float   reserved1;
    float   reserved2;
    float   reserved3;
};
static_assert(sizeof(StateRelaxationPushConstants) == 48,
              "state relaxation push constants must be 48 bytes");

// Exact homogeneous recycling field shift. Source ownership is validated on the host;
// the shader needs only the retained destination and signed source offset.
struct HomogeneousDomainFieldShiftPushConstants
{
    int32_t nr;
    int32_t nth;
    int32_t nz;
    int32_t shift_i_cells;
    int32_t shift_j_cells;
    int32_t destination_i_begin;
    int32_t destination_i_end;
    int32_t destination_j_begin;
    int32_t destination_j_end;
    int32_t leading_profile_offset;
    int32_t leading_is_field;
};
static_assert(sizeof(HomogeneousDomainFieldShiftPushConstants) == 44,
              "domain-field shift push constants must be 44 bytes");

struct EarthControlEvolutionPushConstants
{
    uint64_t seed;
    uint64_t member_bits;
    uint64_t initial_process_key;
    uint64_t innovation_process_key;
    uint64_t innovation_epoch;
    int64_t control_i_min;
    int64_t control_j_min;
    int32_t control_i_count;
    int32_t control_j_count;
    int32_t nz;
    float rho;
    uint32_t mode;
    // Preserve innovation variance even when the high word rounds to one.
    float rho_low;
};
static_assert(sizeof(EarthControlEvolutionPushConstants) == 80,
              "Earth-control evolution push constants must be 80 bytes");

struct MynnLevel25StabilityPushConstants
{
    int32_t count;
    int32_t padding[3];
};
static_assert(sizeof(MynnLevel25StabilityPushConstants) == 16,
              "MYNN level-2.5 stability push constants must be 16 bytes");

struct RrtmgLongwaveTransferPushConstants
{
    int32_t layers;
    int32_t has_clouds;
    float surface_emissivity;
    float precipitable_water_cm;
    int32_t padding[4];
};
static_assert(sizeof(RrtmgLongwaveTransferPushConstants) == 32,
              "RRTMG longwave-transfer push constants must be 32 bytes");

struct RrtmgShortwaveTransferPushConstants
{
    int32_t layers;
    int32_t has_clouds;
    float cosine_zenith;
    float solar_irradiance_wm2;
    float surface_albedo;
    float source_sum;
    int32_t padding[2];
};
static_assert(sizeof(RrtmgShortwaveTransferPushConstants) == 32,
              "RRTMG shortwave-transfer push constants must be 32 bytes");

struct RrtmgLongwaveTransferBatchPushConstants
{
    int32_t layers;
    int32_t columns;
    int32_t has_clouds;
    int32_t padding[5];
};
static_assert(sizeof(RrtmgLongwaveTransferBatchPushConstants) == 32,
              "RRTMG longwave batch push constants must be 32 bytes");

struct RrtmgShortwaveTransferBatchPushConstants
{
    int32_t layers;
    int32_t columns;
    int32_t has_clouds;
    int32_t padding[5];
};
static_assert(sizeof(RrtmgShortwaveTransferBatchPushConstants) == 32,
              "RRTMG shortwave batch push constants must be 32 bytes");

struct RrtmgLongwaveColumnParameters
{
    float surface_emissivity;
    float precipitable_water_cm;
    float padding[2];
};
static_assert(sizeof(RrtmgLongwaveColumnParameters) == 16,
              "RRTMG longwave column parameters must be 16 bytes");

struct RrtmgShortwaveColumnParameters
{
    float cosine_zenith;
    float solar_irradiance_wm2;
    float surface_albedo;
    float source_sum;
};
static_assert(sizeof(RrtmgShortwaveColumnParameters) == 16,
              "RRTMG shortwave column parameters must be 16 bytes");

} // namespace nimbus::compute::abi
