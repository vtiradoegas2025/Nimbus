#pragma once

#include "core/infra/coordinate_system.hpp"
#include "microphysics/config.hpp"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @file config_v1.hpp
 * @brief Strongly-typed Nimbus v1 configuration schema.
 *
 * Mirrors the schema target in docs/CM1Mapping.md §20. Field defaults match
 * CM1 namelist defaults where CM1 has an established convention; deviations
 * carry an inline comment explaining the choice (per design principle §19.4).
 *
 * Section names, scheme keys, and parameter names follow CM1 namelist
 * conventions (§19.1) so a CM1 researcher recognizes the structure at sight.
 * Nimbus-specific extensions live in `ConfigV1::extensions` (§19.3).
 *
 * The struct is consumed by:
 *   - `v1_loader.cpp`            (Phase A2): populates from YAML
 *   - `v0_to_v1.cpp`             (Phase A3): produces from legacy v0 configs
 *   - `dynamics/factory.cpp`     (Phase B1b): config-aware scheme dispatch
 *   - downstream subsystem init  (Phase A4): drives sounding/hodograph/...
 */

namespace tmv::init::config
{

// ─────────────────────────────────────────────────────────────
// Enumerations
// ─────────────────────────────────────────────────────────────

/// Lateral boundary closure. Explicit values separate the collocated
/// zero-gradient closure from the C-grid KW78 radiative closure (SCI-017).
/// Numeric values 0..3 are stable because they are passed to Vulkan shaders.
enum class LateralBC
{
    Periodic     = 0,
    Radiative    = 1,
    Rigid        = 2,
    RigidSlip    = 3,
    ZeroGradient = 4,
    LegacyOpen   = 5,
};

/// CM1 &param3 bottom BC codes (bbc): 1=rigid no-slip, 2=rigid free-slip.
enum class BottomBC
{
    RigidNoSlip,
    RigidSlip,
};

/// CM1 &param3 top BC codes (tbc): rigid with optional sponge (irdamp).
enum class TopBC
{
    Rigid,
    RigidWithSponge,
    // Open: zero-gradient (Neumann) lid for u/v - NOT a true radiative/open
    // top. The lid w-face is not prognosed (effectively rigid for w) and the
    // upper Rayleigh sponge absorbs vertical outflow. The "open" label is a
    // naming convenience, not a non-reflecting boundary (science audit).
    Open,
};

/// CM1 irdamp: 1 = implicit, 2 = explicit.
enum class DampingMode
{
    Explicit,
    Implicit,
};

/// CM1 &param12 betaplane: 0 = f-plane, 1 = beta-plane.
enum class CoriolisMode
{
    FPlane,
    BetaPlane,
};

enum class TimeIntegrator
{
    SplitExplicit,
    SSPRK3,
    RK4,
};

/// Horizontal computational-domain motion. Runtime admission reports which
/// modes are implemented for the selected hardware/science combination.
enum class DomainMotionMode
{
    Stationary,
    ConstantVelocity,
    TrackedWindow,
    MovingNest,
};

/// Parent/child coupling direction.
enum class NestingFeedback
{
    OneWay,
    TwoWay,
};

/// Pre-allocation response when the requested storm-lifetime corridor does not
/// fit inside the configured horizontal domain.
enum class LifetimePlanningAction
{
    Warn,
    Error,
};

/// Source of the Earth-relative storm-motion vector used by launch preflight.
enum class StormMotionSource
{
    BunkersRight,
    BunkersLeft,
    Explicit,
};

/// Environmental discriminator used only after temporal continuity admits
/// multiple storm branches.
enum class StormBranchPreference
{
    Continuity,
    BunkersRight,
    BunkersLeft,
};

/// CM1 ptype codes for microphysics. Nimbus does not yet expose ptype=2 (GSR LFO).
enum class MicrophysicsScheme
{
    None,      ///< ptype = 0 (dry)
    Kessler,   ///< ptype = 1
    Lin,       ///< Lin-Farley-Orville
    Thompson,  ///< ptype = 3
    Milbrandt, ///< Nimbus Milbrandt-Yau double-moment closure; not Morrison equivalence
};

/// CM1 sgsmodel codes for SGS turbulence.
enum class TurbulenceScheme
{
    None,        ///< no SGS
    Smagorinsky, ///< sgsmodel = 1
    TKE,         ///< sgsmodel = 2 (Deardorff)
};

/// CM1 horizontal/vertical SGS toggle (dohturb + dovturb).
enum class TurbulenceMode
{
    TwoD,         ///< horizontal SGS only
    VerticalOnly, ///< vertical SGS only
    ThreeD,       ///< horizontal + vertical SGS
};

enum class TurbulenceFilterWidth
{
    CubicRoot,
    Horizontal,
    User,
};

enum class TurbulenceStabilityCorrection
{
    None,
    Richardson,
    TkeBased,
};

/// Scalar face reconstruction used by the runtime advection operator.
/// `Auto` preserves Nimbus's historical grid-specific behavior: TVD/MC on
/// the collocated A-grid and WENO5 on the conservative C-grid.
enum class AdvectionScheme
{
    Auto,
    None,
    TVD,
    WENO5,
};

/// CM1 idiff codes for explicit numerical diffusion.
enum class DiffusionScheme
{
    None,
    Explicit,
    Implicit,
};

/// What the diffusion operator is applied to.
enum class DiffusionTarget
{
    Scalars,
    ScalarsAndMomentum,
};

enum class DiffusionCoefficientSource
{
    Constant,
    LinearHeight,
};

/// Public surface-model selections. Monin-Obukhov similarity is owned by the
/// active PBL closures rather than exposed as an independent surface scheme.
enum class SurfaceScheme
{
    None,
    Slab,
};

/// Boundary-layer mixing schemes. Slab is Nimbus's prognostic mixed-layer
/// closure; YSU/MYNN are WRF-lineage PBL schemes (CM1 instead relies on LES).
enum class BoundaryLayerScheme
{
    None,
    Slab,
    YSU,
    MYNN,
};

enum class BoundaryLayerTopMethod
{
    Richardson,
    PotentialTemperature,
    TKE,
};

/// Public radiation selections implemented by Nimbus.
enum class RadiationScheme
{
    None,
    SimpleGrey,
    RRTMG,
};

/// Explicit solar forcing contract for idealized radiation.
enum class RadiationSolarForcing
{
    IdealizedDiurnal,
};

/// Terrain generator selected for the mapped Cartesian C-grid lower boundary.
enum class TerrainScheme
{
    None,
    Bell,
    Schar,
};

/// Vertical coordinate used above the configured lower-boundary topography.
enum class TerrainCoordinate
{
    BasicTerrainFollowing,
    Smoothed,
    Cartesian,
};

/// Horizontal frame in which analytic terrain features are evaluated.
enum class TerrainHorizontalPlacement
{
    GridAttached,
    EarthFixed,
};

/// CM1 iinit family of initial-condition triggers.
enum class TriggerType
{
    None,
    Bubble,
    VortexSeed,
};

/// Sounding source for the base-state column.
enum class SoundingType
{
    ParametricCAPE,
    ParametricTargets,
    File,
};

/// Hodograph source for the base-state wind profile.
enum class HodographType
{
    Auto,
    Custom,
    WKParam,
};

/// Output preset names for the complete published field set.
enum class OutputPreset
{
    Minimal,
    RadarReady,
    RadarZdrExploratory,
    RadarPolarimetricReady,
    Presentation,
    StudioFull,
    DynamicsFull,
    ScienceValidation,
};

/// Sparse, lossless diagnostic streams with a fixed scientific field contract.
enum class QualificationDiagnosticsPreset
{
    None,
    MovingReferenceV1,
};

/// Output file format.
enum class OutputFormat
{
    Npy3D,
    Netcdf,
    ZfpLow,
    ZfpMedium,
    ZfpHigh,
};

/// Compute backend (Nimbus extension; CM1 is CPU-only).
enum class ComputeBackend
{
    CPU,
    GPU,
    Metal,
};

/// Startup action when the phase-aware estimate exceeds the RAM budget.
enum class MemoryAdmissionPolicy
{
    Warn,
    Reject,
};

/// Chaos / ensemble perturbation scheme (Nimbus extension).
enum class ChaosScheme
{
    None,
    InitialConditions,
    BoundaryLayer,
    FullStochastic,
};

/// Field-contract validation mode.
enum class ValidationGuardMode
{
    Off,
    Warn,
    Error,
};

/// Scientific execution policy. Exploratory preserves disclosed recovery;
/// Qualified converts recovery/substitution into a fatal run error.
enum class ExecutionMode
{
    Exploratory,
    Qualified,
};

// ─────────────────────────────────────────────────────────────
// Section structs (mirror §20)
// ─────────────────────────────────────────────────────────────

/// CM1 &param0 / &param11 — Grid setup.
struct GridSection
{
    CoordinateSystem coordinate_system = CoordinateSystem::Cartesian; // CM1 is Cartesian-only
    StaggerType      staggering        = StaggerType::CGrid;          // CM1 is C-grid
    int              nx                = 0;
    int              ny                = 0;
    int              nz                = 0;
    double           dx                = 0.0;
    double           dy                = 0.0;
    double           dz                = 0.0;
    double           ztop              = 16000.0; // Nimbus pins explicitly; CM1 derives from nz*dz
};

/// CM1 &param1 / &param2 — Time control + output cadence.
struct TimeSection
{
    TimeIntegrator integrator       = TimeIntegrator::SplitExplicit;
    double run_time          = 0.0;   // CM1 &param1:run_time (s)
    double dt_large          = 0.0;   // CM1 &param1:dtl
    int    acoustic_substeps = 6;     // CM1 nsound; requested minimum (0 = auto)
    bool   adaptive_dt       = false; // CM1 &param1:adapt_dt
    double cfl_safety        = 0.7;   // CM1 cflmax-equivalent
    double output_interval   = 0.0;   // CM1 &param2:tapfrq (s)
};

/// Horizontal motion and storm-lifetime planning. The complete shape is typed
/// now so future moving-domain work cannot grow through unvalidated string
/// maps. Stationary defaults preserve the current engine exactly.
struct DomainMotionSection
{
    DomainMotionMode mode = DomainMotionMode::Stationary;

    struct FrameVelocity
    {
        double u_ms = 0.0;
        double v_ms = 0.0;
    } frame_velocity;

    struct LifetimePlanning
    {
        bool                   enabled               = false;
        LifetimePlanningAction action                = LifetimePlanningAction::Error;
        StormMotionSource      motion_source         = StormMotionSource::BunkersRight;
        double                 explicit_u_ms         = 0.0;
        double                 explicit_v_ms         = 0.0;
        double                 upstream_buffer_m     = 0.0;
        double                 downstream_buffer_m   = 0.0;
        double                 cross_track_buffer_m  = 0.0;
        double                 motion_uncertainty_ms = 0.0;
    } lifetime_planning;

    struct Tracking
    {
        bool enabled = false;
        // Explicit authority to commit a qualified recycling request to the
        // prognostic domain.  maximum_shift_cells alone remains a diagnostic
        // movement-policy limit so existing tracking configurations do not
        // acquire state mutation as an implicit side effect.
        bool                  recycle_domain                         = false;
        double                interval_s                             = 60.0;
        double                inner_margin_m                         = 0.0;
        int                   maximum_shift_cells                    = 0;
        double                cooldown_s                             = 300.0;
        double                minimum_confidence                     = 0.8;
        double                minimum_updraft_ms                     = 5.0;
        double                minimum_condensate_path_kgm2           = 0.1;
        double                updraft_helicity_bottom_m_agl          = 2000.0;
        double                updraft_helicity_top_m_agl             = 5000.0;
        double                minimum_absolute_updraft_helicity_m2s2 = 25.0;
        int                   minimum_component_cells                = 4;
        double                minimum_core_area_m2                   = 1.0e6;
        double                maximum_motion_speed_ms                = 75.0;
        double                maximum_motion_acceleration_ms2        = 0.25;
        int                   minimum_continuity_samples             = 3;
        int                   lost_after_samples                     = 3;
        int                   topology_confirmation_samples          = 2;
        StormBranchPreference branch_preference                      = StormBranchPreference::Continuity;
        double                branch_velocity_uncertainty_ms         = 5.0;
    } tracking;

    struct Nesting
    {
        bool            enabled                       = false;
        int             parent_ratio_x                = 3;
        int             parent_ratio_y                = 3;
        int             parent_time_step_ratio        = 3;
        int             child_origin_parent_cell_i    = 0;
        int             child_origin_parent_cell_j    = 0;
        int             child_cells_x                 = 0;
        int             child_cells_y                 = 0;
        int             relaxation_width_child_cells  = 0;
        int             boundary_stencil_parent_cells = 1;
        double          relaxation_e_folding_time_s   = 2.0;
        NestingFeedback feedback                      = NestingFeedback::OneWay;
    } nesting;
};

/// CM1 dynamics core (one solver; named for clarity).
struct DynamicsSection
{
    std::string scheme       = "klemp_wilhelmson_split";
    bool        axisymmetric = false; // NIMBUS EXTENSION: j=0 replicate optimization
};

/// CM1 &param3 — Boundary conditions.
struct BoundaryConditionsSection
{
    struct Lateral
    {
        // `open` was historically staggering-dependent. The loader resolves
        // this compatibility default to an explicit closure before use.
        LateralBC west  = LateralBC::LegacyOpen; // CM1 wbc
        LateralBC east  = LateralBC::LegacyOpen; // CM1 ebc
        LateralBC south = LateralBC::LegacyOpen; // CM1 sbc
        LateralBC north = LateralBC::LegacyOpen; // CM1 nbc
    } lateral;
    BottomBC bottom = BottomBC::RigidNoSlip;  // CM1 bbc
    TopBC    top    = TopBC::RigidWithSponge; // CM1 tbc
};

/// CM1 &param7 — Rayleigh damping (top sponge) + the Nimbus lateral
/// sponge (no CM1 analogue: CM1 pairs open BCs with radiation alone;
/// here a sin^2 band damps toward the environmental profile on every
/// non-periodic lateral face - SCIENCE.md M18 made it
/// configurable; width 0 or alpha 0 disables it).
struct DampingSection
{
    bool        enabled       = true;    // CM1 irdamp = 1
    double      z_d           = 12000.0; // CM1 zd (m)
    double      alpha         = 0.05;    // CM1 alph (s^-1)
    DampingMode mode          = DampingMode::Explicit;
    int         lateral_width = 20;   // grid points per face
    double      lateral_alpha = 0.05; // 1/s at the face
    // Master on/off for the lateral sponge band (promoted from the hidden
    // NIMBUS_DISABLE_LATERAL_SPONGE env knob - modularity cleanup 6a). false
    // skips the band entirely, regardless of lateral_width / lateral_alpha.
    bool lateral_sponge_enabled = true;
    // Additive w-only upper "lid" absorber (science audit: was always-on and
    // invisible in YAML). The top w_only_depth_cells get a sin^2-ramped
    // Rayleigh drag on w only, peaking at w_only_alpha_max (1/s) at the lid.
    bool   w_only_top_sponge  = true;
    int    w_only_depth_cells = 8;
    double w_only_alpha_max   = 0.20;
};

/// CM1 &param12 — Coriolis.
struct CoriolisSection
{
    bool         enabled      = false; // CM1 &param12:icor
    CoriolisMode mode         = CoriolisMode::FPlane;
    double       latitude_deg = 35.0; // CM1 lat0
};

/// CM1 &param4 — Moisture / Microphysics.
struct MicrophysicsSection
{
    MicrophysicsScheme      scheme = MicrophysicsScheme::None; // CM1 ptype = 0
    tmv::MicrophysicsConfig parameters;
};

/// CM1 &param5 — Turbulence (SGS).
struct TurbulenceSection
{
    TurbulenceScheme scheme = TurbulenceScheme::Smagorinsky;
    TurbulenceMode   mode   = TurbulenceMode::ThreeD;
    TurbulenceFilterWidth filter_width = TurbulenceFilterWidth::CubicRoot;
    TurbulenceStabilityCorrection stability_correction =
        TurbulenceStabilityCorrection::None;
    double user_filter_width_m = 0.0;
    bool dynamic_coefficient = false;
    double           Cs     = 0.18; // CM1 default
    // LES convention 1/3 (CM1 cloud-LES), not the RANS 0.7 (science audit
    // M-minor). Matches turbulence_constants::Pr_t_default; this is the
    // value that reaches the runtime via serialize_v1_to_v0_map when a
    // config omits the key.
    double Pr_t = 1.0 / 3.0;
    double Sc_t = 1.0 / 3.0;
};

/// Nimbus scalar-advection selection. This is separate from the
/// split-explicit dynamics integrator: it chooses transported-scalar face
/// reconstruction only.
struct AdvectionSection
{
    AdvectionScheme scheme = AdvectionScheme::Auto;
};

/// CM1 &param6 — Numerical diffusion.
struct DiffusionSection
{
    DiffusionScheme            scheme             = DiffusionScheme::Explicit;
    int                        operator_order     = 2;   // CM1 difforder; runtime currently implements only 2
    double                     K_h                = 0.0; // CM1 kdiff2 horizontal
    double                     K_v                = 0.0; // independent in Nimbus
    DiffusionTarget            apply_to           = DiffusionTarget::Scalars;
    DiffusionCoefficientSource coefficient_source = DiffusionCoefficientSource::Constant;
    double                     K_v_surface        = 0.0;
    double                     K_v_top            = 0.0;
};

/// CM1 &param9 — Surface.
struct SurfaceSection
{
    SurfaceScheme scheme       = SurfaceScheme::Slab;
    bool          apply_fluxes = true; // CM1 isfcflx = 1
    double        z0m_m        = 0.1;
    double        z0h_m        = 0.01;
    double        albedo       = 0.2;
    double        emissivity   = 0.95;
    // Prescribed surface (skin) state that drives surface fluxes (CM1 tsk0).
    // A sentinel value < 0 means "omitted": the runtime then holds the initial
    // surface-air state as the skin (neutral at t=0; flux develops as the
    // boundary layer evolves). tsk_k is an absolute temperature [K]; qsfc_kgkg
    // is a water-vapor mixing ratio [kg/kg].
    double tsk_k     = -1.0;
    double qsfc_kgkg = -1.0;
};

/// Boundary-layer mixing (Nimbus addition; not in CM1).
struct BoundaryLayerSection
{
    // Slab preserves Nimbus's historical/default runtime physics. `none` is
    // the canonical disabled value. Surface and boundary-layer selections
    // must be explicit and consistent; the loader never infers one from the
    // other.
    BoundaryLayerScheme scheme = BoundaryLayerScheme::Slab;
    BoundaryLayerTopMethod pbl_top_method =
        BoundaryLayerTopMethod::Richardson;
    double cadence_s = 60.0;
    bool clear_sky_only = true;
};

/// CM1 &param8 — Radiation plus explicit Nimbus forcing controls.
struct RadiationSection
{
    RadiationScheme       scheme                                   = RadiationScheme::None;
    RadiationSolarForcing solar_forcing                            = RadiationSolarForcing::IdealizedDiurnal;
    double                latitude_deg                             = 35.0;
    double                declination_deg                          = 15.0;
    double                local_solar_time_at_start_h              = 0.0;
    bool                  do_lw                                    = true;
    bool                  do_sw                                    = true;
    double                cadence_s                                = 300.0;
    bool                  clear_sky_only                           = true;
    std::string           simple_grey_optical_model                = "pressure_clear_sky_v1";
    bool                  use_subcycling                           = false;
    int                   subcycle_count                           = 1;
    bool                  enable_diagnostics                       = false;
    double                tau_lw_ref                               = 6.0;
    double                tau_sw_ref                               = 0.22;
    double                n_lw                                     = 4.0;
    double                n_sw                                     = 2.0;
    double                simple_grey_lw_vapor_absorption_m2kg     = 0.02;
    double                simple_grey_lw_liquid_absorption_m2kg    = 85.0;
    double                simple_grey_lw_ice_absorption_m2kg       = 45.0;
    double                simple_grey_sw_vapor_absorption_m2kg     = 0.001;
    double                simple_grey_sw_liquid_absorption_m2kg    = 20.0;
    double                simple_grey_sw_ice_absorption_m2kg       = 12.0;
    // RRTMG dry-air mole fractions. Water vapor is diagnosed from the model
    // state. Ozone has no hidden climatology and must be written explicitly
    // in a strict-v1 document whenever RRTMG is selected.
    double                rrtmg_ozone_vmr                           = 0.0;
    double                rrtmg_co2_vmr                             = 420.0e-6;
    double                rrtmg_ch4_vmr                             = 1.9e-6;
    double                rrtmg_n2o_vmr                             = 0.335e-6;
    double                rrtmg_co_vmr                              = 0.0;
    double                rrtmg_o2_vmr                              = 0.2095;
    double                rrtmg_n2_vmr                              = 0.7808;
    double                rrtmg_cfc11_vmr                           = 0.0;
    double                rrtmg_cfc12_vmr                           = 0.0;
    double                rrtmg_cfc22_vmr                           = 0.0;
    double                rrtmg_ccl4_vmr                            = 0.0;
    double                rrtmg_liquid_effective_radius_um          = 10.0;
    double                rrtmg_ice_effective_radius_um             = 30.0;
    int                   terrain_sw_cosine_nodes                  = 4;
    int                   terrain_sw_azimuth_nodes                 = 8;
    int                   terrain_sw_maximum_bounces               = 32;
    double                terrain_sw_relative_residual_tolerance   = 1.0e-10;
    double                terrain_sw_absolute_residual_tolerance_w = 1.0e-9;
};

/// Terrain lower-boundary and coordinate-mapping configuration.
/// `grid.ztop` remains the single source of truth for the domain top.
struct TerrainSection
{
    TerrainScheme     scheme              = TerrainScheme::None;
    TerrainCoordinate coordinate          = TerrainCoordinate::BasicTerrainFollowing;
    TerrainHorizontalPlacement horizontal_placement =
        TerrainHorizontalPlacement::GridAttached;
    bool              compute_derivatives = true;
    bool              compute_metrics     = true;
    double            smoothing_decay     = 0.1;
    double            feature_center_earth_x_m = 0.0;
    double            feature_center_earth_y_m = 0.0;

    struct Bell
    {
        double h0           = 1000.0;
        double a            = 5000.0;
        bool   axisymmetric = false;
    } bell;

    struct Schar
    {
        double H   = 1000.0;
        double a   = 5000.0;
        double ell = 4000.0;
    } schar;
};

/// Sub-block of initial_conditions: sounding source + parameters.
struct SoundingSubsection
{
    SoundingType type     = SoundingType::ParametricCAPE;
    double       cape_jkg = 2500.0;
    // Diagnostic-table targets. These are valid only for
    // type=parametric_targets; the strict loader rejects them for the
    // procedural CAPE and file branches.
    double cin_jkg = 50.0;
    double lcl_m   = 1200.0;
    double lfc_m   = 2000.0;
    double el_m    = 12000.0;
    // Procedural profile boundary. This is deliberately distinct from el_m:
    // the ParametricCAPE builder consumes a tropopause height, not a realized
    // equilibrium-level target.
    double tropopause_m    = 12000.0;
    double surface_theta_k = 300.0;
    // Optional because ParametricTargets derives qv from lcl_m when no
    // override is supplied. ParametricCAPE uses its 0.014 kg/kg default when
    // this key is omitted. The strict loader rejects an explicit LCL together
    // with an explicit qv override for ParametricTargets.
    std::optional<double> moisture_qv_kgkg;
    std::string           file_path; // populated when type == File
};

/// Sub-block of initial_conditions: hodograph source + 4-point custom layout.
struct HodographSubsection
{
    // Auto consumes winds from a file sounding when present, otherwise it
    // falls back to the configured three-point anchors. Custom always forces
    // those anchors, including over file-supplied winds.
    HodographType type     = HodographType::Auto;
    double        u_sfc_ms = 0.0;
    double        v_sfc_ms = 0.0;
    double        u_1km_ms = 0.0;
    double        v_1km_ms = 0.0;
    double        u_6km_ms = 0.0;
    double        v_6km_ms = 0.0;
};

/// Sub-block of initial_conditions: trigger type + per-type parameters.
struct TriggerSubsection
{
    TriggerType type = TriggerType::None;

    struct Bubble
    {
        double dtheta_k   = 2.0;
        double cloud_water_kgkg = 0.0;
        double radius_km  = 10.0;
        double center_x_m = 0.0;
        double center_y_m = 0.0;
        double center_z_m = 1500.0;
    } bubble;

    struct VortexSeed
    {
        double r_max_m    = 6000.0;
        double v_max_ms   = 40.0;
        double z_top_m    = 6000.0;
        double center_x_m = 0.0;
        double center_y_m = 0.0;
        double center_z_m = 3000.0;
    } vortex_seed;
};

/// CM1 iinit family — Initial conditions.
struct InitialConditionsSection
{
    SoundingSubsection  sounding;
    HodographSubsection hodograph;
    TriggerSubsection   trigger;
};

/// Output: CM1 &param2 cadence + Nimbus-specific format / preset / IO.
struct OutputSection
{
    OutputPreset preset   = OutputPreset::Minimal;
    OutputFormat format   = OutputFormat::Npy3D;
    bool         async_io = true;
    bool         live_shm = false;
    std::vector<std::string> live_shm_fields = {
        "w", "reflectivity_dbz", "vorticity_z"};
    double live_shm_interval_s = 1.0;
    bool live_shm_full_rate = false;

    // Independent planar qualification stream. MovingReferenceV1 publishes
    // exactly cold_pool_boundary and accumulated_rainfall as lossless NPY at
    // this cadence without changing ordinary history cadence or field preset.
    QualificationDiagnosticsPreset qualification_diagnostics_preset     = QualificationDiagnosticsPreset::None;
    int                            qualification_diagnostics_interval_s = 30;

    // Online Lagrangian parcel trajectories. These belong to output because
    // they produce a run artifact and are selectable by Studio.
    bool   trajectories           = false;
    double trajectory_seed_time_s = 0.0;
    int    trajectory_count       = 5;
    double trajectory_sample_interval_s = 0.0;
    int    trajectory_max_samples_per_parcel = 16384;

    // netCDF4/HDF5 history layout. These are serialized only when Netcdf is
    // selected, and map directly to the runtime OutputConfig policy.
    int  netcdf_deflate_level    = 2;
    bool netcdf_shuffle          = true;
    int  netcdf_target_chunk_kib = 1024;

    // ZFP compression tuning (Nimbus extension; consulted only when `format`
    // is a Zfp* tier). The tier name sets the default accuracy tolerance
    // (zfp_low=1e-2, zfp_medium=1e-3, zfp_high=1e-4); `zfp_tolerance`
    // overrides it when > 0. Defaults mirror OutputConfig (output_config.hpp).
    // keyframe_interval 0 = keyframe-only, so every frame stays randomly
    // seekable (required by Studio scrubbing; delta chains are opt-in).
    double                                  zfp_tolerance            = 0.0; // 0 = derive from the format tier
    std::string                             zfp_mode                 = "accuracy";
    int                                     zfp_precision_bits       = 16;
    int                                     zfp_rate_bps             = 8;
    int                                     zfp_keyframe_interval    = 0;
    double                                  zfp_sparse_threshold     = 0.0;
    bool                                    zfp_float16_prefilter    = false;
    bool                                    zfp_predictive_delta     = false;
    std::string                             zfp_field_execution      = "serial";
    int                                     zfp_field_workers        = 0;
    int                                     zfp_field_scratch_mib    = 64;
    bool                                    zfp_per_field_tolerances = false;
    double                                  zfp_tight_tolerance      = 1.0e-4;
    double                                  zfp_moderate_tolerance   = 1.0e-3;
    double                                  zfp_loose_tolerance      = 1.0e-2;
    std::unordered_map<std::string, double> zfp_field_tolerances;
    // Tiered write cadence (format-independent; works with npy too).
    bool tiered_write_cadence   = false;
    int  write_cadence_medium_s = 30;
    int  write_cadence_slow_s   = 60;
    // Zero retains every completed history frame and disables the free-space
    // reserve, preserving pre-retention behavior.
    int keep_last_frames       = 0;
    int minimum_free_space_mib = 0;
    int maximum_retained_history_files = 0;
};

struct RadarSection
{
    bool enabled = false;
    std::vector<std::string> products;
    std::string reflectivity_tier = "fast_da";
    std::string velocity_sampling = "point";
    std::string zdr_tier = "polarimetric_fo";
    double radar_x_m = 0.0;
    double radar_y_m = 0.0;
    double radar_z_m = 0.0;
    bool apply_scatterer_correction = true;
};

/// Nimbus extensions: compute backend.
struct ComputeExtension
{
    ComputeBackend        backend          = ComputeBackend::CPU;
    MemoryAdmissionPolicy memory_admission = MemoryAdmissionPolicy::Warn;
    /// Permit a requested accelerator to recover to the CPU when device
    /// initialization fails. Qualified execution overrides this to false in
    /// the effective runtime projection.
    bool allow_fallback = true;
    /// Runtime CPU-vs-GPU parity harness (GPU.md Phase 1):
    /// sampled GPU dispatches are replayed on the CPU reference path and
    /// compared with tiered tolerances; any breach fails the run.
    bool validate_parity = false;
    /// Optional device state-commit requests. They are no-ops on CPU and are
    /// admitted only when the shared interaction policy supports the selected
    /// scheme, geometry, maturity, and guard mode. Defaults preserve the
    /// established runtime profile; they are not blanket performance claims.
    bool gpu_apply_microphysics = true;
    bool gpu_apply_slow         = false;
    bool gpu_apply_sgs          = false;
    bool cuda_assist            = false;
};

/// Nimbus extensions: solver internals + research toggles, promoted from hidden
/// NIMBUS_* env knobs (modularity cleanup 6a). Every default reproduces the prior
/// env-unset runtime exactly; the experimental toggles stay off.
struct SolverExtension
{
    /// Heun (RK2) integration of the slow dynamics tendencies. true is the
    /// production default and the only stable pairing with the UP5 stencils;
    /// false restores Forward Euler for A/B comparison only (linearly unstable
    /// - not for science runs). Was NIMBUS_SLOW_RK2.
    bool slow_rk2 = true;
    /// Lid-BC dissipation coefficient (Smell #8 Option B experiment). A finite
    /// value in (0, 1] selects the base-state-anchored damping form; any other
    /// value (default -1) selects the linear hydrostatic form. Was
    /// NIMBUS_LID_BC_GAMMA.
    double lid_bc_gamma = -1.0;
    /// Fourier m=1 filter on the axis radial velocity u[0] (Smell #5 cheap
    /// path; experimental, default off). Was NIMBUS_AXIS_U_M1_FILTER.
    bool axis_u_m1_filter = false;
};

/// Nimbus extensions: chaos / ensemble perturbation infrastructure.
struct ChaosExtension
{
    ChaosScheme scheme    = ChaosScheme::None;
    int         seed      = 0;
    int         member_id = 0;
    bool        native_generation_required = false;

    // Optional process ownership. Empty lists retain each scheme's safe
    // public defaults; explicit lists are validated by the strict-v1 loader.
    std::vector<std::string> apply_to_ic;
    std::vector<std::string> apply_to_tendencies;

    // Per-target amplitudes. These defaults mirror ChaosConfig so projecting
    // a strict-v1 document into the runtime cannot silently change the
    // stochastic experiment.
    std::unordered_map<std::string, double> sigma_ic = {
        {"u", 0.5}, {"v", 0.5}, {"w", 0.1}, {"theta", 0.5},
        {"qv", 0.0005}, {"qc", 0.0}, {"qr", 0.0},
    };
    std::unordered_map<std::string, double> alpha_tend = {
        {"microphysics", 0.3}, {"pbl", 0.2}, {"diffusion", 0.1},
    };
};

/// Nimbus extensions: MMS, conservation gates, smell tracking.
struct ValidationExtension
{
    ValidationGuardMode guard_mode = ValidationGuardMode::Warn;
};

/// Nimbus extensions: declared scientific execution policy.
struct ExecutionExtension
{
    ExecutionMode mode = ExecutionMode::Exploratory;
    /// Deprecated compatibility input. Exact Ready profiles do not require a
    /// maturity acknowledgment; an explicit false value still fails closed so
    /// old configs cannot appear to opt out while selecting qualified mode.
    bool acknowledge_maturity_limits = false;
};

/// Nimbus extensions: portable local checkpoint lifecycle. These are
/// orchestration controls, not CM1 restart settings.
struct CheckpointExtension
{
    int         every_s                = 0;
    int         keep_last              = 0;
    int         minimum_free_space_mib = 0;
    std::string directory;
    std::string resume_from;
};

/// Nimbus extensions: scientific-diagnostic cadence.
///
/// `scientific_every_steps` uses an internal compatibility sentinel when the
/// key is omitted:
///   - -1: preserve the historical runtime policy (output/live/debug cadence,
///         or every step when periodic output is disabled)
///   -  0: disable conservation/scientific diagnostics
///   - >0: run them on that one-based step cadence
///
/// The strict v1 loader accepts only values >= 0 from YAML.  Users therefore
/// select compatibility behavior by omitting the key, not by spelling -1.
struct DiagnosticsExtension
{
    int  scientific_every_steps   = -1;
    bool deterministic_reductions = false;
};

/// Nimbus extensions: performance instrumentation.
///
/// A zero report cadence means "final summary only"; it does not disable
/// timing. `warmup_steps` are executed normally but excluded from all reported
/// timing totals.
struct PerformanceExtension
{
    // Presence is retained so an omitted v1 block does not erase an explicit
    // NIMBUS_PERF_* environment setting during the v1-to-v0 projection.
    std::optional<bool> timing;
    std::optional<int>  report_every_steps;
    std::optional<int>  warmup_steps;
};

/// Nimbus extensions (additive; not part of CM1).
struct ExtensionsSection
{
    ComputeExtension     compute;
    SolverExtension      solver;
    ChaosExtension       chaos;
    ExecutionExtension   execution;
    CheckpointExtension  checkpoint;
    ValidationExtension  validation;
    DiagnosticsExtension diagnostics;
    PerformanceExtension performance;
};

/**
 * @brief Full v1 configuration. Top-level shape mirrors docs/CM1Mapping.md §20.
 *
 * Populated by `load_v1_config` from a `schema: nimbus.v1` YAML file. The
 * `schema_version` field captures the version stamp read from the file
 * (currently the only valid value is "nimbus.v1") so downstream code can
 * route on it once future versions exist.
 */
struct ConfigV1
{
    std::string schema_version = "nimbus.v1";

    GridSection               grid;
    TimeSection               time;
    DomainMotionSection       domain_motion;
    DynamicsSection           dynamics;
    BoundaryConditionsSection boundary_conditions;
    DampingSection            damping;
    CoriolisSection           coriolis;
    MicrophysicsSection       microphysics;
    TurbulenceSection         turbulence;
    AdvectionSection          advection;
    DiffusionSection          diffusion;
    SurfaceSection            surface;
    BoundaryLayerSection      boundary_layer;
    RadiationSection          radiation;
    TerrainSection            terrain;
    InitialConditionsSection  initial_conditions;
    RadarSection              radar;
    OutputSection             output;
    ExtensionsSection         extensions;
};

// ─────────────────────────────────────────────────────────────
// Enum parse + name helpers
// ─────────────────────────────────────────────────────────────
//
// Each helper pair: `parse_X` returns false on unknown input (caller emits
// the line-numbered error); `X_name` returns the canonical YAML spelling
// for round-trip and error messages.

bool        parse_lateral_bc(const std::string& s, LateralBC& out);
const char* lateral_bc_name(LateralBC v);
LateralBC   resolve_legacy_lateral_bc(LateralBC v, StaggerType staggering);

bool        parse_bottom_bc(const std::string& s, BottomBC& out);
const char* bottom_bc_name(BottomBC v);

bool        parse_top_bc(const std::string& s, TopBC& out);
const char* top_bc_name(TopBC v);

bool        parse_damping_mode(const std::string& s, DampingMode& out);
const char* damping_mode_name(DampingMode v);

bool        parse_coriolis_mode(const std::string& s, CoriolisMode& out);
const char* coriolis_mode_name(CoriolisMode v);
bool        parse_time_integrator(const std::string& s, TimeIntegrator& out);
const char* time_integrator_name(TimeIntegrator v);

bool        parse_domain_motion_mode(const std::string& s, DomainMotionMode& out);
const char* domain_motion_mode_name(DomainMotionMode v);
void        set_internal_moving_nest_qualification(bool enabled);
bool        internal_moving_nest_qualification_enabled();

bool        parse_nesting_feedback(const std::string& s, NestingFeedback& out);
const char* nesting_feedback_name(NestingFeedback v);

bool        parse_lifetime_planning_action(const std::string& s, LifetimePlanningAction& out);
const char* lifetime_planning_action_name(LifetimePlanningAction v);

bool        parse_storm_motion_source(const std::string& s, StormMotionSource& out);
const char* storm_motion_source_name(StormMotionSource v);
bool        parse_storm_branch_preference(const std::string& s, StormBranchPreference& out);
const char* storm_branch_preference_name(StormBranchPreference v);

bool        parse_microphysics_scheme(const std::string& s, MicrophysicsScheme& out);
const char* microphysics_scheme_name(MicrophysicsScheme v);

bool        parse_turbulence_scheme(const std::string& s, TurbulenceScheme& out);
const char* turbulence_scheme_name(TurbulenceScheme v);

bool        parse_turbulence_mode(const std::string& s, TurbulenceMode& out);
const char* turbulence_mode_name(TurbulenceMode v);
bool parse_turbulence_filter_width(const std::string& s,
                                   TurbulenceFilterWidth& out);
const char* turbulence_filter_width_name(TurbulenceFilterWidth v);
bool parse_turbulence_stability_correction(
    const std::string& s, TurbulenceStabilityCorrection& out);
const char* turbulence_stability_correction_name(
    TurbulenceStabilityCorrection v);

bool        parse_advection_scheme(const std::string& s, AdvectionScheme& out);
const char* advection_scheme_name(AdvectionScheme v);
const char* resolved_advection_scheme_name(AdvectionScheme v, StaggerType staggering);

bool        parse_diffusion_scheme(const std::string& s, DiffusionScheme& out);
const char* diffusion_scheme_name(DiffusionScheme v);

bool        parse_diffusion_target(const std::string& s, DiffusionTarget& out);
const char* diffusion_target_name(DiffusionTarget v);

bool parse_diffusion_coefficient_source(
    const std::string& s, DiffusionCoefficientSource& out);
const char* diffusion_coefficient_source_name(DiffusionCoefficientSource v);

bool        parse_surface_scheme(const std::string& s, SurfaceScheme& out);
const char* surface_scheme_name(SurfaceScheme v);

bool        parse_boundary_layer_scheme(const std::string& s, BoundaryLayerScheme& out);
const char* boundary_layer_scheme_name(BoundaryLayerScheme v);
bool parse_boundary_layer_top_method(
    const std::string& s, BoundaryLayerTopMethod& out);
const char* boundary_layer_top_method_name(BoundaryLayerTopMethod v);

bool        parse_radiation_scheme(const std::string& s, RadiationScheme& out);
const char* radiation_scheme_name(RadiationScheme v);
bool        parse_radiation_solar_forcing(const std::string& s, RadiationSolarForcing& out);
const char* radiation_solar_forcing_name(RadiationSolarForcing v);

bool        parse_terrain_scheme(const std::string& s, TerrainScheme& out);
const char* terrain_scheme_name(TerrainScheme v);

bool        parse_terrain_coordinate(const std::string& s, TerrainCoordinate& out);
const char* terrain_coordinate_name(TerrainCoordinate v);

bool parse_terrain_horizontal_placement(
    const std::string& s, TerrainHorizontalPlacement& out);
const char* terrain_horizontal_placement_name(TerrainHorizontalPlacement v);

bool        parse_trigger_type(const std::string& s, TriggerType& out);
const char* trigger_type_name(TriggerType v);

bool        parse_sounding_type(const std::string& s, SoundingType& out);
const char* sounding_type_name(SoundingType v);

bool        parse_hodograph_type(const std::string& s, HodographType& out);
const char* hodograph_type_name(HodographType v);

bool        parse_output_preset(const std::string& s, OutputPreset& out);
const char* output_preset_name(OutputPreset v);
bool        parse_qualification_diagnostics_preset(const std::string& s, QualificationDiagnosticsPreset& out);
const char* qualification_diagnostics_preset_name(QualificationDiagnosticsPreset v);

bool        parse_output_format(const std::string& s, OutputFormat& out);
const char* output_format_name(OutputFormat v);

bool        parse_compute_backend(const std::string& s, ComputeBackend& out);
const char* compute_backend_name(ComputeBackend v);

bool        parse_memory_admission_policy(const std::string& s, MemoryAdmissionPolicy& out);
const char* memory_admission_policy_name(MemoryAdmissionPolicy v);

bool        parse_chaos_scheme(const std::string& s, ChaosScheme& out);
const char* chaos_scheme_name(ChaosScheme v);

bool        parse_validation_guard_mode(const std::string& s, ValidationGuardMode& out);
const char* validation_guard_mode_name(ValidationGuardMode v);

bool        parse_execution_mode(const std::string& s, ExecutionMode& out);
const char* execution_mode_name(ExecutionMode v);

} // namespace tmv::init::config
