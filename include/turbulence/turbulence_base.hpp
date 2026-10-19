#pragma once
#include "core/infra/flow_execution.hpp"
#include <array>

struct ComputeBackendRuntimeState;

#include <memory>
#include <cstdint>
#include <limits>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/field/field3d.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/infra/physical_constants.hpp"

// Subgrid-turbulence contracts and persistent runtime state.

struct SimulationStateView;

namespace tmv
{

struct TerrainMetrics3D;
struct Topography2D;

// The checkpointed selector distinguishes this balance from legacy linear Ri mixing.
inline constexpr const char* kLillyRichardsonModel = "ri_lilly_v1";

inline bool known_turbulence_stability_model(const std::string& model)
{
    return model == "none" || model == "ri" || model == kLillyRichardsonModel || model == "tke_based";
}

namespace turbulence_constants
{
inline constexpr double g = physical_constants::gravity_ms2;
inline constexpr double kappa = physical_constants::von_karman_constant;
inline constexpr double C_s_default = 0.18;
// LES convention (Deardorff; CM1's Smagorinsky likewise): the SGS
// turbulent Prandtl/Schmidt numbers are 1/3, not the RANS-style 0.7
// previously carried here (SCIENCE.md Tier 4 minor - heat and
// moisture mixed ~2x too weakly relative to momentum).
inline constexpr double Pr_t_default = 1.0 / 3.0;
inline constexpr double Sc_t_default = 1.0 / 3.0;
} // namespace turbulence_constants

struct TurbulenceConfig
{
    std::string scheme_id = "smagorinsky";
    // Maximum explicit SGS integration interval. Production may shorten it
    // from the realized operator bound and refreshes after every local commit.
    double dt_sgs = 1.0;
    std::string mode = "3d";
    std::string filter_width = "cubic_root";
    double user_filter_width_m = 0.0;
    double Cs = turbulence_constants::C_s_default;
    double Pr_t = turbulence_constants::Pr_t_default;
    double Sc_t = turbulence_constants::Sc_t_default;
    bool dynamic_Cs = false;
    std::string stability_correction = "none";
    bool enable_moist_buoyancy = false;
    bool enable_diagnostics = false;
    double nu_t_max = 1000.0;
    double K_max = 1000.0;
};

struct GridMetrics
{
    double dx = 1000.0;
    double dy = 1000.0;
    std::vector<double> dz;
    std::vector<double> z_int;
    bool terrain_metrics_active = false;
    const TerrainMetrics3D* terrain_metrics = nullptr;
    const Topography2D* terrain_topography = nullptr;
    // Storage ownership is part of the physical-grid contract. C-grid remains
    // the compatibility default for direct operator callers; runtime-owned
    // views publish their actual staggering explicitly.
    StaggerType staggering = StaggerType::CGrid;
};

enum class TurbulenceRequirements
{
    BASIC = 1,
    MOISTURE = 2,
    TKE = 4,
    STABILITY = 8
};

struct TurbulenceStateView : FlowExecutionInputs
{
    const Field3D* u = nullptr;
    const Field3D* v = nullptr;
    const Field3D* w = nullptr;
    const Field3D* rho = nullptr;
    const Field3D* theta = nullptr;
    const Field3D* qv = nullptr;
    const Field3D* tke = nullptr;
    const std::vector<double>* rho0 = nullptr;

    // Runtime binds the actual field placement. Collocated state uses a
    // separate physical-domain finite-volume stress operator on flat and
    // mapped CPU grids; native accelerator SGS remains profile-specific.
    bool c_grid = true;

    // A stochastic diffusion block requires host-current coefficient fields
    // even when the baseline closure executes on an accelerator.
    bool require_host_transport_coefficients = false;
    // The same block will rebuild tendencies after coefficient perturbation;
    // the baseline closure must not spend a second pass producing values that
    // are immediately discarded.
    bool defer_transport_tendency_build = false;
    // Qualified fallback-off accelerator profiles set this after live
    // capability admission. The scheme must then fail rather than silently
    // recomputing a rejected device dispatch on the CPU.
    bool require_native_dispatch = false;


    int NR = 0;
    int NTH = 0;
    int NZ = 0;
};

struct TurbulenceTendencies
{
    Field3D dudt_sgs;
    Field3D dvdt_sgs;
    Field3D dwdt_sgs;
    Field3D dthetadt_sgs;
    Field3D dqvdt_sgs;
    Field3D dtkedt_sgs;

    // Generation-bound provenance for the mapped C-grid stress operator.
    // Finite only when the same call that produced these tendencies evaluated
    // every active open exterior traction from the identical stress stencil.
    double mapped_boundary_power_w =
        std::numeric_limits<double>::quiet_NaN();

    // Producer/host-ownership hint for accelerator reuse. A GPU producer sets
    // this only after completion; every CPU writer clears it. Backends must
    // still validate model-step generation, pointer identity, and byte extent.
    bool resident_source_unchanged = false;
};

struct TurbulenceDiagnostics
{
    Field3D nu_t;
    Field3D K_theta;
    Field3D K_q;
    Field3D K_tke;

    Field3D shear_prod;
    Field3D buoy_prod;
    Field3D dissipation;

    Field3D Cs_eff;
    Field3D Pr_t_eff;
};

// Mutable coefficient view used by stochastic transport uncertainty.  A
// spatial multiplier belongs on the positive transport coefficient before
// conservative flux divergence; multiplying the already-diverged tendency
// would destroy momentum/scalar conservation and could masquerade as open-
// boundary work.
struct TurbulenceTransportFields
{
    Field3D* eddy_viscosity = nullptr;
    Field3D* theta_diffusivity = nullptr;
    Field3D* moisture_diffusivity = nullptr;
    Field3D* tke_diffusivity = nullptr;
};

// Exact resolved kinetic-energy attribution for one committed SGS update.
// The natural currency is C-grid face mass times face velocity squared.  The
// linear term is the work requested by the held stress-divergence tendency;
// the quadratic term is the finite-explicit-step correction; and the storage
// term records the difference introduced by the actual FP32 state commit.
struct TurbulenceMomentumWorkReport
{
    static constexpr std::uint32_t report_schema_version = 1;
    static constexpr std::uint32_t energy_basis_version = 2;

    double requested_linear_work_j = 0.0;
    double finite_step_correction_j = 0.0;
    double storage_rounding_j = 0.0;
    double committed_work_j = 0.0;
    // Exact changes in the remaining moist-mixture basis-v2 reservoirs
    // committed by the same SGS/PBL-TKE writer.  committed_energy_change_j
    // includes committed_work_j exactly once.
    double moist_dry_internal_change_j = 0.0;
    double moist_water_sensible_change_j = 0.0;
    double moist_phase_reference_change_j = 0.0;
    double moist_carrier_kinetic_change_j = 0.0;
    double moist_gravitational_change_j = 0.0;
    double moist_subgrid_tke_change_j = 0.0;
    double moist_committed_energy_change_j = 0.0;
    double moist_energy_closure_residual_j = 0.0;
    bool moist_energy_complete = false;
};

class TurbulenceSchemeBase
{
public:
    virtual ~TurbulenceSchemeBase() = default;
    virtual std::unique_ptr<TurbulenceSchemeBase> clone() const
    {
        return nullptr;
    }

    virtual std::string name() const = 0;

    // Bitmask of TurbulenceRequirements needed by compute().
    virtual int required_fields() const = 0;

    virtual void initialize(const TurbulenceConfig& cfg) = 0;

    // A successful call shapes and fully initializes every tendency field.
    virtual void compute(const TurbulenceConfig& cfg,
                         const GridMetrics& grid,
                         const TurbulenceStateView& state,
                         TurbulenceTendencies& tend,
                         TurbulenceDiagnostics* diag_opt = nullptr) = 0;

    // Host-current coefficient used to close realized stress work. Device
    // schemes return null unless the coefficient for the latest tendency was
    // actually downloaded; shape alone cannot prove generation identity.
    virtual const Field3D* realized_eddy_viscosity() const noexcept
    {
        return nullptr;
    }

    virtual bool mutable_realized_transport_fields(
        TurbulenceTransportFields&) noexcept
    {
        return false;
    }

    // Rebuild all scheme tendencies after an in-place, nonnegative
    // stochastic coefficient perturbation. Implementations must use their
    // ordinary conservative writers and leave no stale device-residency hint.
    virtual bool rebuild_from_realized_transport(
        const TurbulenceConfig&,
        const GridMetrics&,
        const TurbulenceStateView&,
        TurbulenceTendencies&)
    {
        return false;
    }

    // A spatially relocated level may preserve the scheme object only when it
    // can make every coefficient generated from the old field layout
    // unavailable until the next complete compute().
    virtual bool supports_spatial_generation_invalidation() const noexcept
    {
        return false;
    }
    virtual void invalidate_spatial_generation() noexcept
    {
    }
};

// Per-simulation scheme, configuration, and update cadence.
struct TurbulenceRuntimeState
{
    std::unique_ptr<TurbulenceSchemeBase> scheme;
    TurbulenceConfig config;
    double last_update_time_s = std::numeric_limits<double>::lowest();
    // Nominal sampling deadline, distinct from the actual dynamics time at
    // which a crossed deadline is serviced.
    double next_update_time_s = 0.0;
    // Most recently realized SGS tendencies.  Scheduled diagnostic callers
    // may inspect this buffer, but active production integration refreshes it
    // from the live state before every explicit substep and never advances a
    // changed state with a stale nonlinear tendency.
    TurbulenceTendencies held_tendencies;
    // Transactional writer storage. A successful refresh swaps this with the
    // held publication, retaining both field sets across nonlinear substeps;
    // copies/checkpoints intentionally omit this replaceable scratch.
    TurbulenceTendencies candidate_tendencies;
    // Optional scheme diagnostics use the same two-buffer publication model
    // as tendencies. They are derived state: runtime snapshots preserve the
    // last committed publication, while checkpoints may regenerate it from
    // the restored prognostic fields before the next output event.
    TurbulenceDiagnostics held_diagnostics;
    TurbulenceDiagnostics candidate_diagnostics;
    // True only when held_tendencies was produced from the live state by the
    // selected closure (and, if active, rebuilt from its stochastically
    // perturbed realized coefficients). This is transient generation
    // evidence, not checkpointed science state.
    bool realized_transport_tendency_current = false;
    bool stochastic_transport_applied_to_current = false;
    // Owned geometry scratch is used only when the process-global immutable
    // metrics cannot be borrowed (local states and mapped terrain).  Keeping
    // it here prevents rebuilding the vertical coordinate vectors on every
    // nonlinear SGS refresh; runtime copies intentionally omit this cache.
    GridMetrics grid_metrics_scratch;
    TurbulenceMomentumWorkReport last_momentum_work;
    TurbulenceMomentumWorkReport cumulative_moist_energy_work;
    double cumulative_requested_momentum_work_j = 0.0;
    double cumulative_finite_step_momentum_correction_j = 0.0;
    double cumulative_momentum_storage_rounding_j = 0.0;
    double cumulative_committed_momentum_work_j = 0.0;
    std::uint64_t momentum_work_step_count = 0;
    // The most restrictive accepted explicit SGS interval observed during
    // the previous outer model step.  Timestep selection uses it as a
    // predictive cap; the SGS-local integrator still re-evaluates the live
    // operator before every commit.
    double last_explicit_stability_limit_s =
        std::numeric_limits<double>::infinity();
    std::uint64_t last_explicit_substep_count = 0;
    // Exact realized open-stress identity integrated over the most recent
    // outer step and cumulatively over every such step:
    // requested linear work = boundary work - irreversible dissipation.
    // These are scientific provenance, not profiling counters.
    double last_open_boundary_work_j = 0.0;
    double last_irreversible_dissipation_j = 0.0;
    double last_stress_closure_residual_j = 0.0;
    double cumulative_open_boundary_work_j = 0.0;
    double cumulative_irreversible_dissipation_j = 0.0;
    double cumulative_stress_closure_residual_j = 0.0;
    std::uint64_t open_boundary_stability_step_count = 0;

    TurbulenceRuntimeState() = default;
    TurbulenceRuntimeState(const TurbulenceRuntimeState& other)
        : config(other.config), last_update_time_s(other.last_update_time_s),
          next_update_time_s(other.next_update_time_s),
          held_tendencies(other.held_tendencies),
          held_diagnostics(other.held_diagnostics),
          last_momentum_work(other.last_momentum_work),
          cumulative_moist_energy_work(other.cumulative_moist_energy_work),
          cumulative_requested_momentum_work_j(
              other.cumulative_requested_momentum_work_j),
          cumulative_finite_step_momentum_correction_j(
              other.cumulative_finite_step_momentum_correction_j),
          cumulative_momentum_storage_rounding_j(
              other.cumulative_momentum_storage_rounding_j),
          cumulative_committed_momentum_work_j(
              other.cumulative_committed_momentum_work_j),
          momentum_work_step_count(other.momentum_work_step_count),
          last_explicit_stability_limit_s(
              other.last_explicit_stability_limit_s),
          last_explicit_substep_count(other.last_explicit_substep_count),
          last_open_boundary_work_j(other.last_open_boundary_work_j),
          last_irreversible_dissipation_j(
              other.last_irreversible_dissipation_j),
          last_stress_closure_residual_j(
              other.last_stress_closure_residual_j),
          cumulative_open_boundary_work_j(
              other.cumulative_open_boundary_work_j),
          cumulative_irreversible_dissipation_j(
              other.cumulative_irreversible_dissipation_j),
          cumulative_stress_closure_residual_j(
              other.cumulative_stress_closure_residual_j),
          open_boundary_stability_step_count(
              other.open_boundary_stability_step_count)
    {
        if (other.scheme)
        {
            scheme = other.scheme->clone();
            if (!scheme)
            {
                throw std::runtime_error(
                    "turbulence scheme does not support runtime snapshots: "
                    + other.scheme->name());
            }
        }
    }
    TurbulenceRuntimeState& operator=(const TurbulenceRuntimeState& other)
    {
        if (this != &other)
        {
            TurbulenceRuntimeState copy(other);
            *this = std::move(copy);
        }
        return *this;
    }
    TurbulenceRuntimeState(TurbulenceRuntimeState&&) noexcept = default;
    TurbulenceRuntimeState& operator=(TurbulenceRuntimeState&&) noexcept = default;
};

using TurbulenceSchemeFactory = std::unique_ptr<TurbulenceSchemeBase> (*)(const TurbulenceConfig&);

} // namespace tmv

#include "turbulence/factory.hpp"
#include "turbulence/runtime.hpp"

namespace turbulence_constants = tmv::turbulence_constants;
using tmv::TerrainMetrics3D;
using tmv::Topography2D;
using tmv::TurbulenceConfig;
using tmv::GridMetrics;
using tmv::TurbulenceRequirements;
using tmv::TurbulenceStateView;
using tmv::TurbulenceTendencies;
using tmv::TurbulenceDiagnostics;
using tmv::TurbulenceMomentumWorkReport;
using tmv::TurbulenceSchemeBase;
using tmv::TurbulenceRuntimeState;
using tmv::TurbulenceSchemeFactory;
using tmv::create_turbulence_scheme;
using tmv::get_available_turbulence_schemes;
