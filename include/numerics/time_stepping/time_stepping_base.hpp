#pragma once
#include "numerics/time_stepping/native_rk.hpp"

struct SimulationStateView;

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "dynamics/acoustic_energy_report.hpp"
#include "dynamics/completed_step_energy_report.hpp"
#include "dynamics/eos_pressure.hpp"
#include "dynamics/final_boundary_energy_report.hpp"
#include "dynamics/slow_momentum_energy_report.hpp"
#include "dynamics/sponge_energy_report.hpp"
#include "numerics/numerics_base.hpp"

// Time-integration contracts and persistent runtime state.

namespace tmv
{

// The split-explicit callback table is rebuilt once per model step and is
// consumed synchronously before its caller returns.  std::function's small
// buffer is too small for several production closures, which formerly caused
// six heap allocations per step.  This bounded owning callable keeps the same
// safe temporary-lambda assignment semantics while making the known callback
// high-water mark explicit and allocation-free.
template <typename Signature, std::size_t Capacity = 128>
class InplaceCallback;

template <typename Return, std::size_t Capacity, typename... Args>
class InplaceCallback<Return(Args...), Capacity>
{
public:
    InplaceCallback() noexcept = default;
    InplaceCallback(std::nullptr_t) noexcept {}

    template <
        typename Callable,
        typename Stored = std::decay_t<Callable>,
        std::enable_if_t<
            !std::is_same_v<Stored, InplaceCallback>, int> = 0>
    InplaceCallback(Callable&& callable)
    {
        assign<Stored>(std::forward<Callable>(callable));
    }

    ~InplaceCallback() { reset(); }

    InplaceCallback(const InplaceCallback&) = delete;
    InplaceCallback& operator=(const InplaceCallback&) = delete;
    InplaceCallback(InplaceCallback&&) = delete;
    InplaceCallback& operator=(InplaceCallback&&) = delete;

    template <
        typename Callable,
        typename Stored = std::decay_t<Callable>,
        std::enable_if_t<
            !std::is_same_v<Stored, InplaceCallback>, int> = 0>
    InplaceCallback& operator=(Callable&& callable)
    {
        reset();
        assign<Stored>(std::forward<Callable>(callable));
        return *this;
    }

    InplaceCallback& operator=(std::nullptr_t) noexcept
    {
        reset();
        return *this;
    }

    explicit operator bool() const noexcept { return invoke_ != nullptr; }

    Return operator()(Args... args) const
    {
        if (invoke_ == nullptr)
            throw std::bad_function_call();
        return invoke_(
            const_cast<std::byte*>(storage_),
            std::forward<Args>(args)...);
    }

    void reset() noexcept
    {
        if (destroy_ != nullptr)
            destroy_(storage_);
        invoke_ = nullptr;
        destroy_ = nullptr;
    }

private:
    template <typename Stored, typename Callable>
    void assign(Callable&& callable)
    {
        static_assert(
            sizeof(Stored) <= Capacity,
            "split-explicit callback exceeds its allocation-free capacity");
        static_assert(
            alignof(Stored) <= alignof(std::max_align_t),
            "split-explicit callback requires unsupported over-alignment");
        static_assert(
            std::is_invocable_r_v<Return, Stored&, Args...>,
            "split-explicit callback has an incompatible signature");
        new (storage_) Stored(std::forward<Callable>(callable));
        invoke_ = [](void* storage, Args&&... values) -> Return
        {
            return std::invoke(
                *static_cast<Stored*>(storage),
                std::forward<Args>(values)...);
        };
        destroy_ = [](void* storage) noexcept
        {
            static_cast<Stored*>(storage)->~Stored();
        };
    }

    alignas(std::max_align_t) std::byte storage_[Capacity]{};
    Return (*invoke_)(void*, Args&&...) = nullptr;
    void (*destroy_)(void*) noexcept = nullptr;
};

// The scheme owns substep ordering; dynamics supplies field-specific work.
struct SplitExplicitCallbacks
{
    // Compute slow tendencies (advection + buoyancy).
    InplaceCallback<void()> compute_slow_tendencies;
    InplaceCallback<void(double dt)> apply_slow_tendencies;
    // Seed open-boundary history from the post-slow state.
    InplaceCallback<void()> seed_acoustic_radiation;
    InplaceCallback<void(double dt_small)> apply_fast_pressure;
    // Optional zero-time halo closure between forward pressure and backward
    // momentum. Collocated centered gradients require this pressure time level.
    InplaceCallback<void()> close_fast_pressure_boundaries;
    InplaceCallback<void(double dt_small)> apply_fast_momentum;
    // Optional optimized paths return true when they handle the work.
    InplaceCallback<bool(double dt_small)> apply_fast_fused;
    InplaceCallback<bool(double dt_small, int n_substeps)>
        apply_fast_batched;
    InplaceCallback<void(double dt_small)> acoustic_bcs;
    InplaceCallback<void()> begin_acoustic_loop_residency;
    // Releases backend storage whose lifetime is exactly the acoustic loop.
    InplaceCallback<void()> release_acoustic_loop_residency;
    bool batched_acoustic_boundary_complete = false;
};

struct UnsplitRkCallbacks
{
    InplaceCallback<void()> capture_base;
    InplaceCallback<void(std::size_t stage, double stage_time)>
        compute_tendencies;
    InplaceCallback<void(std::size_t stage_count,
                         const std::array<double, 4>& weights,
                         double dt)>
        publish_combination;
    InplaceCallback<void()> restore_base;
};

class UnsplitRkBoundError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

struct TimeSteppingConfig
{
    std::string scheme_id = "rk3";
    bool split_acoustic = false;
    // Requested minimum substep count (0 = fully automatic).  The runtime
    // updates max_acoustic_speed_ms from the resolved p/rho state before each
    // step and diagnoses the minimum physical face-normal spacing. Split-
    // explicit stepping promotes an under-resolved positive count instead of
    // allowing it to violate the acoustic CFL target.
    int n_acoustic_substeps = 1;
    double max_acoustic_speed_ms = 0.0;
    double min_acoustic_spacing_m = 0.0;
    std::string physics_splitting = "additive";
    double cfl_safety = numerics_constants::cfl_target;
    bool adaptive_dt = false;
    double dt_min = 1e-6;
    double dt_max = 100.0;
};

struct TimeSteppingState
{
    std::vector<NumericalState> fields;
    double time = 0.0;
    double dt = 1.0;
};

struct TimeSteppingDiagnostics
{
    int n_steps = 0;
    double max_cfl = 0.0;
    double cpu_time_per_step = 0.0;
    std::vector<double> dt_history;
    bool converged = true;

    void prepare_history_append()
    {
        if (dt_history.size() == dt_history.max_size())
            throw std::length_error("timestep history is full");
        if (dt_history.size() == dt_history.capacity())
        {
            const auto capacity = dt_history.capacity();
            const auto available = dt_history.max_size() - capacity;
            dt_history.reserve(capacity + std::min(
                available, std::max<std::size_t>(capacity, 16)));
        }
    }
};

class TimeSteppingSchemeBase : public NumericalSchemeBase
{
public:
    using NumericalSchemeBase::initialize;

    virtual std::unique_ptr<TimeSteppingSchemeBase> clone() const
    {
        return nullptr;
    }

    virtual void initialize(const TimeSteppingConfig& cfg, RHSFunction rhs_func) = 0;

    virtual void step(const TimeSteppingConfig& cfg,
                      TimeSteppingState& state,
                      TimeSteppingDiagnostics* diag_opt = nullptr) = 0;

    virtual double suggest_dt(const TimeSteppingConfig& cfg, const TimeSteppingState& state) = 0;

    // Split-explicit schemes own dt splitting and forward-backward ordering.
    virtual void step_split_acoustic(
        const TimeSteppingConfig&,
        double,
        const SplitExplicitCallbacks&) {}

    virtual int resolve_substep_count(const TimeSteppingConfig&, double)
    {
        throw std::runtime_error("time-stepping scheme has no acoustic substep resolver");
    }

    virtual bool supports_split_acoustic() const { return false; }

    virtual bool supports_coupled_unsplit_rk() const noexcept { return false; }

    virtual void step_coupled_unsplit_rk(
        const TimeSteppingConfig&,
        double,
        double,
        const UnsplitRkCallbacks&,
        TimeSteppingDiagnostics* = nullptr)
    {
        throw std::runtime_error(
            "time-stepping scheme does not implement coupled unsplit RK");
    }

    // Domain recycling occurs only at a completed-step boundary. Built-in
    // schemes use this hook to clear retained RK/fast-stage values without
    // releasing their allocation. Third-party schemes must opt in explicitly
    // before the recycling transaction can preserve their scheme object.
    virtual bool supports_domain_recycling_invalidation() const noexcept
    {
        return false;
    }
    virtual void invalidate_generation_state_after_domain_recycling() noexcept
    {
    }

    // Two callbacks cannot retain the base and intermediate states required by RK.
    virtual void step_unsplit(double dt,
                              std::function<void()> compute_tendencies,
                              std::function<void(double)> apply_tendencies)
    {
        if (supports_coupled_unsplit_rk())
            throw std::invalid_argument("Runge-Kutta integration requires the complete stage callback transaction");
        compute_tendencies();
        apply_tendencies(dt);
    }
};

using TimeSteppingSchemeFactory = std::unique_ptr<TimeSteppingSchemeBase> (*)(const TimeSteppingConfig&);

struct TimeSteppingSlowRk2Workspace
{
    Field3D u_save;
    Field3D v_save;
    Field3D w_save;
    Field3D p_save;
    Field3D du_dt_k1;
    Field3D dv_dt_k1;
    Field3D dw_dt_k1;
    Field3D dp_dt_k1;
    Field3D du_advection_dt_k1;
    Field3D dv_advection_dt_k1;
    Field3D dw_advection_dt_k1;
};

struct TimeSteppingUnsplitRkWorkspace
{
    NativeRkTendencyWorkspace                                            native_tendency;
    Field3D                                                              native_base_profile;
    static constexpr std::size_t                                         field_count            = 15;
    static constexpr std::size_t                                         stage_count            = 4;
    static constexpr std::size_t                                         private_field_capacity = 16;
    std::array<Field3D, field_count>                                     base;
    std::array<std::array<Field3D, field_count>, stage_count>            tendencies;
    std::array<Field3D, field_count>                                     candidate;
    std::array<Field3D, 4>                                               native_combination;
    std::array<Field3D, field_count>                                     retry_base;
    std::array<Field3D, 10>                                              scalar_mass_tendency;
    std::array<Field3D, private_field_capacity>                          private_base;
    std::array<std::array<Field3D, private_field_capacity>, stage_count> private_tendencies;
    std::array<Field3D, private_field_capacity>                          private_candidate;
    std::array<Field3D, private_field_capacity>                          private_retry_base;
    std::array<Field3D, private_field_capacity>                          private_mass_tendency;
};

/** Exact dry-mass accounting for one Cartesian split-explicit continuity step.
 *
 * Positive boundary mass is outward.  The flux-implied state change is its
 * negative, and closure_residual_kg is the committed FP32 inventory change
 * minus that flux-implied change.  Keeping all four terms prevents diagnostics
 * from hiding finite-precision loss inside a reconstructed end-state flux.
 */
struct DynamicsBoundaryMassReport
{
    double outward_boundary_mass_kg = 0.0;
    double flux_implied_state_change_kg = 0.0;
    double committed_state_change_kg = 0.0;
    double closure_residual_kg = 0.0;
};

struct TimeSteppingRuntimeState
{
    std::unique_ptr<TimeSteppingSchemeBase> scheme;
    TimeSteppingConfig config;
    TimeSteppingDiagnostics diagnostics;
    TimeSteppingSlowRk2Workspace slow_rk2;
    TimeSteppingUnsplitRkWorkspace unsplit_rk;
    DynamicsBoundaryMassReport last_boundary_mass;
    CartesianCGridAcousticEnergyReportV1 last_acoustic_energy;
    CartesianCGridAcousticEnergyReportV1 cumulative_acoustic_energy;
    std::uint64_t acoustic_energy_step_count = 0;
    CartesianSlowMomentumEnergyReportV1 last_slow_momentum_energy;
    CartesianSlowMomentumEnergyReportV1 cumulative_slow_momentum_energy;
    std::uint64_t slow_momentum_energy_step_count = 0;
    dynamics_eos::PressureProjectionEnergyReportV1
        last_eos_pressure_energy;
    dynamics_eos::PressureProjectionEnergyReportV1
        cumulative_eos_pressure_energy;
    std::uint64_t eos_pressure_energy_step_count = 0;
    SpongeEnergyReportV1 last_sponge_energy;
    SpongeEnergyReportV1 cumulative_sponge_energy;
    std::uint64_t sponge_energy_step_count = 0;
    FinalBoundaryEnergyReportV1 last_final_boundary_energy;
    FinalBoundaryEnergyReportV1 cumulative_final_boundary_energy;
    std::uint64_t final_boundary_energy_step_count = 0;
    CompletedStepEnergyReportV1 last_completed_step_energy;
    CompletedStepEnergyReportV1 cumulative_completed_step_energy;
    std::uint64_t completed_step_energy_count = 0;
    double cumulative_outward_boundary_mass_kg = 0.0;
    double cumulative_flux_implied_state_change_kg = 0.0;
    double cumulative_committed_state_change_kg = 0.0;
    double cumulative_boundary_mass_closure_residual_kg = 0.0;
    std::uint64_t boundary_mass_step_count = 0;

    TimeSteppingRuntimeState() = default;
    TimeSteppingRuntimeState(const TimeSteppingRuntimeState& other)
        : config(other.config),
          diagnostics(other.diagnostics),
          slow_rk2(other.slow_rk2),
          unsplit_rk(other.unsplit_rk),
          last_boundary_mass(other.last_boundary_mass),
          last_acoustic_energy(other.last_acoustic_energy),
          cumulative_acoustic_energy(other.cumulative_acoustic_energy),
          acoustic_energy_step_count(other.acoustic_energy_step_count),
          last_slow_momentum_energy(other.last_slow_momentum_energy),
          cumulative_slow_momentum_energy(
              other.cumulative_slow_momentum_energy),
          slow_momentum_energy_step_count(
              other.slow_momentum_energy_step_count),
          last_eos_pressure_energy(other.last_eos_pressure_energy),
          cumulative_eos_pressure_energy(
              other.cumulative_eos_pressure_energy),
          eos_pressure_energy_step_count(
              other.eos_pressure_energy_step_count),
          last_sponge_energy(other.last_sponge_energy),
          cumulative_sponge_energy(other.cumulative_sponge_energy),
          sponge_energy_step_count(other.sponge_energy_step_count),
          last_final_boundary_energy(other.last_final_boundary_energy),
          cumulative_final_boundary_energy(
              other.cumulative_final_boundary_energy),
          final_boundary_energy_step_count(
              other.final_boundary_energy_step_count),
          last_completed_step_energy(other.last_completed_step_energy),
          cumulative_completed_step_energy(
              other.cumulative_completed_step_energy),
          completed_step_energy_count(other.completed_step_energy_count),
          cumulative_outward_boundary_mass_kg(
              other.cumulative_outward_boundary_mass_kg),
          cumulative_flux_implied_state_change_kg(
              other.cumulative_flux_implied_state_change_kg),
          cumulative_committed_state_change_kg(
              other.cumulative_committed_state_change_kg),
          cumulative_boundary_mass_closure_residual_kg(
              other.cumulative_boundary_mass_closure_residual_kg),
          boundary_mass_step_count(other.boundary_mass_step_count)
    {
        if (other.scheme)
        {
            scheme = other.scheme->clone();
            if (!scheme)
            {
                throw std::runtime_error(
                    "time-stepping scheme does not support runtime snapshots: "
                    + other.scheme->name());
            }
        }
    }
    TimeSteppingRuntimeState& operator=(const TimeSteppingRuntimeState& other)
    {
        if (this != &other)
        {
            TimeSteppingRuntimeState copy(other);
            *this = std::move(copy);
        }
        return *this;
    }
    TimeSteppingRuntimeState(TimeSteppingRuntimeState&&) noexcept = default;
    TimeSteppingRuntimeState& operator=(TimeSteppingRuntimeState&&) noexcept = default;
};

extern TimeSteppingRuntimeState& global_time_stepping_runtime_state;

// Source-compatible names; the runtime object remains the sole owner.
extern std::unique_ptr<TimeSteppingSchemeBase>& time_stepping_scheme;
extern TimeSteppingConfig& global_time_stepping_config;

void initialize_time_stepping(const std::string& scheme_name,
                              const TimeSteppingConfig& cfg = TimeSteppingConfig{},
                              RHSFunction rhs_func = nullptr);
void initialize_time_stepping(const SimulationStateView& execution_state, const std::string& scheme_name,
                              const TimeSteppingConfig& cfg = TimeSteppingConfig{},
                              RHSFunction rhs_func = nullptr);
void initialize_time_stepping(TimeSteppingRuntimeState& runtime,
                              const std::string& scheme_name,
                              const TimeSteppingConfig& cfg = TimeSteppingConfig{},
                              RHSFunction rhs_func = nullptr);

} // namespace tmv
#include "numerics/time_stepping/factory.hpp"

using tmv::SplitExplicitCallbacks;
using tmv::UnsplitRkCallbacks;
using tmv::UnsplitRkBoundError;
using tmv::TimeSteppingConfig;
using tmv::TimeSteppingState;
using tmv::TimeSteppingDiagnostics;
using tmv::TimeSteppingSchemeBase;
using tmv::TimeSteppingRuntimeState;
using tmv::TimeSteppingUnsplitRkWorkspace;
using tmv::TimeSteppingSchemeFactory;
using tmv::global_time_stepping_runtime_state;
using tmv::time_stepping_scheme;
using tmv::global_time_stepping_config;
using tmv::create_time_stepping_scheme;
using tmv::get_available_time_stepping_schemes;
using tmv::initialize_time_stepping;
