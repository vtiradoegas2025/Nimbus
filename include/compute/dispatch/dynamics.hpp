#pragma once
#include "compute/dispatch/conservative_reduction.hpp"
#include "compute/dispatch/conservative_momentum.hpp"
#include "compute/dispatch/cgrid_boundary.hpp"

#include "compute/dispatch/common.hpp"

/**
 * @file dynamics.hpp
 * @brief Dynamics compute dispatch data and capability interface.
 */

enum class DynamicsKernelOp
{
    AcousticContinuityMassFluxCartesianAgrid,
    AcousticPressureEosCartesianAgrid,
    AcousticMomentumCartesianAgrid,
    AcousticNullModeConstraintCartesianAgrid,
    AcousticBoundaryCartesianAgrid,
    AcousticPressureCartesianCgrid,
    AcousticMomentumCartesianCgrid,
    AcousticSubstepFusedCartesianCgrid,
    AcousticSubstepsBatchedCartesianCgrid,
    StateRelaxationCartesian,
    SlowApply,
    CartesianCgridSlowTendencies,
    CartesianTendencies,
    AcousticPressure,
    AcousticMomentum,
    AcousticSubstepFused,
    AcousticSubstepsBatched,
    AcousticBoundaryCartesianCgrid,
    ConservativeMomentum,
    ConservativeReduction,
};

struct AcousticPressureEosCartesianAgridDispatch
{
    const float* rho_in = nullptr;
    const float* theta = nullptr;
    const float* drho_dt = nullptr;
    float* rho_out = nullptr;
    float* p_out = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float dt_small = 0.0f;
    float reference_pressure = 0.0f;
    float gas_constant_dry_air = 0.0f;
    float gamma = 0.0f;
    float rho_min = 0.0f;
    float rho_max = 0.0f;
    float p_min = 0.0f;
    float p_max = 0.0f;
    TwoFieldCompletedObserver completed_observer = nullptr;
    void* completed_observer_context = nullptr;
};

struct AcousticMomentumCartesianAgridDispatch
{
    const float* rho = nullptr;
    const float* p = nullptr;
    const float* p0_base_data = nullptr;
    int base_len = 0;
    const float* u_in = nullptr;
    const float* v_in = nullptr;
    const float* w_in = nullptr;
    float* u_out = nullptr;
    float* v_out = nullptr;
    float* w_out = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    float dt_small = 0.0f;
    float wind_clamp_h = 0.0f;
    float wind_clamp_v = 0.0f;
    ThreeFieldCompletedObserver completed_observer = nullptr;
    void* completed_observer_context = nullptr;
};

struct AcousticNullModeConstraintCartesianAgridDispatch
{
    const float* rho = nullptr;
    const float* qv = nullptr;
    const float* qc = nullptr;
    const float* qr = nullptr;
    const float* qi = nullptr;
    const float* qs = nullptr;
    const float* qg = nullptr;
    const float* qh = nullptr;
    const float* u_in = nullptr;
    const float* v_in = nullptr;
    const float* w_in = nullptr;
    float* u_out = nullptr;
    float* v_out = nullptr;
    float* w_out = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    ThreeFieldCompletedObserver completed_observer = nullptr;
    void* completed_observer_context = nullptr;
};

struct AcousticBoundaryCartesianAgridDispatch
{
    const float* p0_base_data = nullptr;
    const float* rho0_base_data = nullptr;
    const float* u0_base_data = nullptr;
    const float* v0_base_data = nullptr;
    int base_len = 0;
    float* u = nullptr;
    float* v = nullptr;
    float* w = nullptr;
    float* rho = nullptr;
    float* p = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    FiveFieldCompletedObserver completed_observer = nullptr;
    void* completed_observer_context = nullptr;
};

/**
 * Production collocated continuity operator.
 *
 * The outputs are tendencies plus the instantaneous positive-face mass fluxes
 * used to form drho_dt. Pressure/EOS integration remains owned by the shared
 * runtime so every backend uses the same nonlinear thermodynamic projection.
 */
struct AcousticContinuityMassFluxCartesianAgridDispatch
{
    const float* u = nullptr;
    const float* v = nullptr;
    const float* w = nullptr;
    const float* rho = nullptr;
    const float* p = nullptr;
    const float* p0_base_data = nullptr;
    int base_len = 0;
    float* drho_dt = nullptr;
    float* mass_flux_x = nullptr;
    float* mass_flux_y = nullptr;
    float* mass_flux_z = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    float dt_small = 0.0f;
    int lateral_bc_west = 0;
    int lateral_bc_east = 0;
    int lateral_bc_south = 0;
    int lateral_bc_north = 0;
    int top_bc = 0;
    bool accumulate_mass_flux = false;
    float mass_flux_weight = 0.0f;
    float frame_velocity_u_ms = 0.0f;
    float frame_velocity_v_ms = 0.0f;
};

struct AcousticPressureCartesianCgridDispatch
{
    const float* u = nullptr;
    const float* v = nullptr;
    const float* w = nullptr;
    const float* rho_in = nullptr;
    const float* p_in = nullptr;
    const float* rho0_base_data = nullptr;
    const float* p0_base_data = nullptr;
    int base_len = 0;
    const float* theta_data = nullptr;
    float* rho_out = nullptr;
    float* p_out = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    double gamma_val = 0.0;
    float dt_small = 0.0f;
    float rho_min = 0.0f;
    float rho_max = 0.0f;
    float p_min = 0.0f;
    float p_max = 0.0f;
    float p_ref_fallback = 0.0f;
    float frame_velocity_u_ms = 0.0f;
    float frame_velocity_v_ms = 0.0f;
    bool repair_primary_state = true;
};

struct AcousticMomentumCartesianCgridDispatch
{
    const float* rho_data = nullptr;
    const float* p_data = nullptr;
    const float* dp0_dz_data = nullptr;
    int base_len = 0;
    const float* u_in = nullptr;
    const float* v_in = nullptr;
    const float* w_in = nullptr;
    float* u_out = nullptr;
    float* v_out = nullptr;
    float* w_out = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    float dt_small = 0.0f;
    float wind_clamp_h = 0.0f;
    float wind_clamp_v = 0.0f;
    bool repair_primary_state = true;
    const float* dp0_dz_low_data = nullptr;
};

struct AcousticSubstepFusedCartesianCgridDispatch
{
    float* u = nullptr;
    float* v = nullptr;
    float* w = nullptr;
    float* rho = nullptr;
    float* p = nullptr;
    const float* theta_data = nullptr;
    const float* rho0_base_data = nullptr;
    const float* p0_base_data = nullptr;
    const float* dp0_dz_data = nullptr;
    int base_len = 0;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    float gamma_val = 0.0f;
    float dt_small = 0.0f;
    float rho_min = 0.0f;
    float rho_max = 0.0f;
    float p_min = 0.0f;
    float p_max = 0.0f;
    float p_ref_fallback = 0.0f;
    float wind_clamp_h = 0.0f;
    float wind_clamp_v = 0.0f;

    // Host-native views preserve the CPU solver's fp64 geometry/profile
    // contract without forcing compact-profile conversion or extra scratch.
    // Device backends ignore these optional fields.
    const double* rho0_base_host = nullptr;
    const double* p0_base_host = nullptr;
    double dx_host = 0.0;
    double dy_host = 0.0;
    double dz_host = 0.0;
    double dt_small_host = 0.0;
    double gamma_host = 0.0;
    float* drho_scratch = nullptr;

    // CPU dispatches can author this report directly. Native device dispatches
    // require a null pointer: orchestration owns their deterministic FP64
    // attribution using the exact input and committed field views.
    const float* qv_host = nullptr;
    const float* qc_host = nullptr;
    const float* qr_host = nullptr;
    const float* qi_host = nullptr;
    const float* qs_host = nullptr;
    const float* qg_host = nullptr;
    const float* qh_host = nullptr;
    const float* tke_host = nullptr;
    CartesianCGridAcousticEnergyReportV1* energy_report_host = nullptr;
    float frame_velocity_u_ms = 0.0f;
    float frame_velocity_v_ms = 0.0f;
    bool repair_primary_state = true;
    float gamma_low = 0.0f;
    const float* dp0_dz_low_data = nullptr;
};

struct AcousticSubstepsBatchedCartesianCgridDispatch
{
    float* u = nullptr;
    float* v = nullptr;
    float* w = nullptr;
    float* rho = nullptr;
    float* p = nullptr;
    const float* theta_data = nullptr;
    const float* rho0_base_data = nullptr;
    const float* p0_base_data = nullptr;
    const float* dp0_dz_data = nullptr;
    int base_len = 0;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    float gamma_val = 0.0f;
    float dt_small = 0.0f;
    int n_substeps = 0;
    float rho_min = 0.0f;
    float rho_max = 0.0f;
    float p_min = 0.0f;
    float p_max = 0.0f;
    float p_ref_fallback = 0.0f;
    float wind_clamp_h = 0.0f;
    float wind_clamp_v = 0.0f;
    int bc_west = 0;
    int bc_east = 0;
    int bc_south = 0;
    int bc_north = 0;
    int bc_bottom = 0;
    int bc_top = 0;
    float* mass_flux_x = nullptr;
    float* mass_flux_y = nullptr;
    float* mass_flux_z = nullptr;
    bool accumulate_mass_flux = false;
    float frame_velocity_u_ms = 0.0f;
    float frame_velocity_v_ms = 0.0f;
    bool repair_primary_state = true;
    float gamma_low = 0.0f;
    const float* dp0_dz_low_data = nullptr;
    double mass_flux_weight = 0.0;
    const float* wave_history = nullptr;
    float* wave_history_output = nullptr;
};


struct SlowApplyDispatch
{
    using CompletedObserver = FourFieldCompletedObserver;

    float* u = nullptr;
    float* v = nullptr;
    float* w = nullptr;
    float* p_data = nullptr;
    const float* du_dt = nullptr;
    const float* dv_dt = nullptr;
    const float* dw_dt = nullptr;
    const float* dp_dt = nullptr;
    const float* du_dt_pbl = nullptr;
    const float* dv_dt_pbl = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float dt_large = 0.0f;
    // Optional post-fence/pre-copy observer. At this point the four pointers
    // above still expose the old host state while these arguments expose the
    // actual committed device result. The observer must not retain them.
    CompletedObserver completed_observer = nullptr;
    void* completed_observer_context = nullptr;
    bool sanitize = true;
};

struct StateRelaxationDispatch
{
    float* u = nullptr;
    float* v = nullptr;
    float* w = nullptr;
    float* theta_data = nullptr;
    const float* profiles = nullptr;
    int profiles_len = 0;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int lateral_width = 0;
    bool do_upper = false;
    bool do_w_only = false;
    bool do_lateral = false;
    float dt_step = 0.0f;
    FourFieldCompletedObserver completed_observer = nullptr;
    void* completed_observer_context = nullptr;
};

struct CartesianCgridSlowTendenciesDispatch
{
    const float* u_data = nullptr;
    const float* v_data = nullptr;
    const float* w_data = nullptr;
    const float* rho_data = nullptr;
    const float* p_data = nullptr;
    const float* rho0_base_data = nullptr;
    int rho0_base_len = 0;
    const float* qv0_base_data = nullptr;
    int qv0_base_len = 0;
    const float* u0_base_data = nullptr;
    const float* v0_base_data = nullptr;
    int wind0_base_len = 0;
    const float* qv_data = nullptr;
    const float* loading_data = nullptr;
    float* du_dt_data = nullptr;
    float* dv_dt_data = nullptr;
    float* dw_dt_data = nullptr;
    float* drho_dt_data = nullptr;
    float* dp_dt_data = nullptr;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    float g = 0.0f;
    float coriolis_f_val = 0.0f;
    bool has_moisture = false;
    bool has_loading = false;
    bool slow_rk2_invariant_input_residency = false;
    // Optional operator-identical momentum-advection components. All three
    // must be supplied together; ordinary runs omit them and incur no extra
    // output allocations or writes.
    float* du_advection_dt_data = nullptr;
    float* dv_advection_dt_data = nullptr;
    float* dw_advection_dt_data = nullptr;
    // Prognostic winds remain Earth-relative. Only horizontal advecting
    // velocities subtract this constant frame translation.
    float frame_velocity_u_ms = 0.0f;
    float frame_velocity_v_ms = 0.0f;
    // df/dy in Earth coordinates. `coriolis_f_val` is f at grid y=0 for
    // this evaluation, so translating/recycled domains retain Earth-relative
    // beta-plane forcing without another device buffer.
    float coriolis_beta = 0.0f;
};

struct CartesianTendenciesDispatch
{
    const float* u_x_data = nullptr;
    const float* u_y_data = nullptr;
    const float* w_data = nullptr;
    const float* rho_data = nullptr;
    const float* p_data = nullptr;
    const float* theta_data = nullptr;
    const float* p0_base_data = nullptr;
    const float* rho0_base_data = nullptr;
    const float* loading_data = nullptr;
    const float* u0_base_data = nullptr;
    const float* v0_base_data = nullptr;
    const float* qv_data = nullptr;
    const float* qv0_base_data = nullptr;
    float* du_x_dt_data = nullptr;
    float* du_y_dt_data = nullptr;
    float* dw_dt_data = nullptr;
    float* drho_dt_data = nullptr;
    float* dp_dt_data = nullptr;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    float g = 0.0f;
    float gamma_val = 0.0f;
    float coriolis_f_val = 0.0f;
    int lateral_bc_west = 0;
    int lateral_bc_east = 0;
    int lateral_bc_south = 0;
    int lateral_bc_north = 0;
    int top_bc = 0;
    bool has_moisture = false;
    float coriolis_beta = 0.0f;
    bool split_slow_only = false;
};

struct AcousticPressureDispatch
{
    const float* u_data = nullptr;
    const float* v_data = nullptr;
    const float* w_data = nullptr;
    const float* rho_in = nullptr;
    const float* p_in = nullptr;
    float* rho_out = nullptr;
    float* p_out = nullptr;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    float dr = 0.0f;
    float dtheta = 0.0f;
    float dz = 0.0f;
    float gamma_val = 0.0f;
    float dt_small = 0.0f;
    float rho_floor = 0.0f;
    float p_floor = 0.0f;
};

struct AcousticMomentumDispatch
{
    const float* rho_data = nullptr;
    const float* p_data = nullptr;
    const float* u_in = nullptr;
    const float* v_in = nullptr;
    const float* w_in = nullptr;
    float* u_out = nullptr;
    float* v_out = nullptr;
    float* w_out = nullptr;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    float dr = 0.0f;
    float dtheta = 0.0f;
    float dz = 0.0f;
    float dt_small = 0.0f;
    float wind_clamp_h = 0.0f;
    float wind_clamp_v = 0.0f;
};

struct AcousticSubstepFusedDispatch
{
    float* u = nullptr;
    float* v = nullptr;
    float* w = nullptr;
    float* rho = nullptr;
    float* p = nullptr;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    float dr = 0.0f;
    float dtheta = 0.0f;
    float dz = 0.0f;
    float gamma_val = 0.0f;
    float dt_small = 0.0f;
    float rho_floor = 0.0f;
    float p_floor = 0.0f;
    float wind_clamp_h = 0.0f;
    float wind_clamp_v = 0.0f;
};

struct AcousticSubstepsBatchedDispatch
{
    float* u = nullptr;
    float* v = nullptr;
    float* w = nullptr;
    float* rho = nullptr;
    float* p = nullptr;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    float dr = 0.0f;
    float dtheta = 0.0f;
    float dz = 0.0f;
    float gamma_val = 0.0f;
    float dt_small = 0.0f;
    int n_substeps = 0;
    float rho_floor = 0.0f;
    float p_floor = 0.0f;
    float wind_clamp_h = 0.0f;
    float wind_clamp_v = 0.0f;
};


class DynamicsComputeKernels
{
public:
    virtual bool dispatch_conservative_reduction(const ConservativeReductionDispatch&)
    {
        return false;
    }

    virtual bool dispatch_conservative_momentum(const ConservativeMomentumDispatch&)
    {
        return false;
    }

    virtual bool dispatch_cgrid_boundary(const CgridBoundaryDispatch&)
    {
        return false;
    }

    virtual ~DynamicsComputeKernels() = default;

    /**
     * @brief Returns true when this backend can execute the requested dynamics operation.
     */
    virtual bool supports(DynamicsKernelOp /*op*/) const { return false; }

    /** Whether the batched C-grid acoustic loop exports averaged mass flux. */
    virtual bool supports_batched_cartesian_cgrid_mass_flux_export() const
    {
        return false;
    }

    virtual bool supports_unrepaired_acoustic_batched() const { return false; }

    /** Execute the production momentum-weighted A-grid continuity stencil. */
    virtual bool dispatch_acoustic_continuity_mass_flux_cartesian_agrid(
        const AcousticContinuityMassFluxCartesianAgridDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    virtual bool dispatch_acoustic_pressure_eos_cartesian_agrid(
        const AcousticPressureEosCartesianAgridDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    virtual bool dispatch_acoustic_momentum_cartesian_agrid(
        const AcousticMomentumCartesianAgridDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    virtual bool dispatch_acoustic_null_mode_constraint_cartesian_agrid(
        const AcousticNullModeConstraintCartesianAgridDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    virtual bool dispatch_acoustic_boundary_cartesian_agrid(
        const AcousticBoundaryCartesianAgridDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    // ── Cartesian C-grid acoustic dispatch (GPU audit Phase 3.1) ──

    /**
     * @brief Execute the Cartesian C-grid acoustic pressure substep on
     *        device: conservative mass continuity drho/dt = -div(rho u),
     *        Forward-Euler rho integration with sanitization and clamps,
     *        then the diagnostic pressure closure
     *        p = p0_base + cs0^2*(rho - rho0_base).
     *
     * Mirrors CartesianCGrid3DScheme::compute_fast_pressure_tendencies
     * (the drho part) plus the apply_fast_pressure integration/diagnosis
     * loop in dynamics.cpp, fused into one resident pass (compute +
     * integrate + clamp; GPU audit section 3 "resident-ready").
     *
     * @param rho0_base_data Reference density profile, 1D, length base_len.
     * @param p0_base_data   Reference pressure profile, 1D, length base_len.
     * @param base_len       Number of entries; must be >= nz.
     * @param p_ref_fallback Fallback pressure for degenerate diagnosis
     *                       (the dynamics p0 reference constant).
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_acoustic_pressure_cartesian_cgrid(
        const AcousticPressureCartesianCgridDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    /**
     * @brief Execute the Cartesian C-grid acoustic momentum substep on
     *        device: per-face pressure gradients (perturbation form in
     *        the vertical, GPU audit section 2.3) with Forward-Euler
     *        integration and magnitude clamps.
     *
     * Mirrors CartesianCGrid3DScheme::compute_fast_momentum_tendencies
     * plus the apply_fast_momentum integration loop in dynamics.cpp.
     *
     * @param dp0_dz_data Base-state vertical pressure gradient at z-faces,
     *                    1D, length base_len (entry nz-1 unused). Must be
     *                    computed in DOUBLE from the fp64 p0_base profile
     *                    and rounded once to fp32 (section 2.3: a broadcast
     *                    fp32 p0 loses the calm-state perturbation to
     *                    quantization).
     * @param base_len    Number of entries; must be >= nz.
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_acoustic_momentum_cartesian_cgrid(
        const AcousticMomentumCartesianCgridDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    /**
     * @brief Execute the fused Cartesian C-grid acoustic substep on
     *        device: pressure substep + compute-to-compute barrier +
     *        momentum substep in one command-buffer submission, all five
     *        state fields updated in place from the host's view.
     *
     * Equivalent to dispatch_acoustic_pressure_cartesian_cgrid followed
     * by dispatch_acoustic_momentum_cartesian_cgrid, with the momentum
     * pass reading the pressure pass's device-side outputs (no host
     * round-trip between them).
     *
     * @return True if dispatch succeeded.
     */
    // Native implementations publish all five fields only after validating
    // the staged result; a refused/failed dispatch leaves host fields intact.
    virtual bool supports_unrepaired_acoustic_fused() const { return false; }

    virtual bool dispatch_acoustic_substep_fused_cartesian_cgrid(
        const AcousticSubstepFusedCartesianCgridDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    /**
     * Theta/base profiles remain immutable until release. When explicitly
     * requested, velocity inputs also remain unchanged between each pressure
     * dispatch and its following momentum dispatch; boundary updates follow.
     * Pressure-output retention additionally requires that neither rho nor p
     * is modified on the host in that interval.
     */
    virtual void begin_acoustic_loop_residency(bool retain_velocity_inputs = false,
                                               bool retain_pressure_outputs = false)
    { (void)retain_velocity_inputs; (void)retain_pressure_outputs; }

    /**
     * Release storage retained only across observable acoustic substeps.
     *
     * Called exactly once after the acoustic loop, including when a substep
     * throws. Backends that do not retain loop-scoped inputs need no action.
     */
    virtual void release_acoustic_loop_residency() {}

    // ── Batched Cartesian C-grid acoustic loop (GPU audit 4.1) ──

    /**
     * @brief Execute ALL n_substeps Cartesian C-grid acoustic substeps in
     *        ONE submission: per substep, the diagnose-p pressure pass, the
     *        perturbation-p momentum pass, and the five per-face acoustic
     *        boundary-condition dispatches (in the CPU's face order), with
     *        a single fence wait at the end. Replaces n_substeps fused
     *        submissions plus n_substeps CPU BC passes.
     *
     * @param bc_* BC enum values as ints (LateralBC / BottomBC / TopBC
     *             encodings from config_v1.hpp).
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_acoustic_substeps_batched_cartesian_cgrid(
        const AcousticSubstepsBatchedCartesianCgridDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    // ── Fused state relaxation dispatch (GPU audit Phase 3.5) ───

    /**
     * @brief Execute upper Rayleigh, additive w-only lid, and lateral
     *        sponge updates in the CPU's call order for u/v/w/theta.
     *        The shared FP64 CPU projection owns EOS pressure.
     *
     * @param profiles Packed 7-row profile block, row stride nz:
     *        scalar upper alpha, w-only alpha, u_ref, v_ref, theta0,
     *        lateral alpha by boundary distance, and stagger-aware upper-w
     *        alpha.
     * @param profiles_len Number of floats in profiles; must be >= 7*nz.
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_state_relaxation_cartesian(const StateRelaxationDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
    // ── Slow-tendency application (GPU audit Phase 4.2) ─────────

    /**
     * @brief Apply the slow tendencies of the split-explicit step to
     *        u/v/w/p in place (non-finite sums integrate as zero; no
     *        clamps, matching the CPU callback).
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_slow_apply(const SlowApplyDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
    // ── Cartesian cgrid slow tendencies dispatch (GPU port of the
    //    Cartesian C-grid convection scheme) ──────────────────────────

    /**
     * @brief Execute Cartesian C-grid slow tendencies on device.
     *
     * 14-buffer SSBO layout (u/v/w/rho/p, rho0_base/qv0_base broadcast
     * 1D->3D, qv, loading, 5 tendency outputs) with a 56-byte push-constant
     * block (dr=dx, dtheta=dy, nr=NX, nth=NY). Momentum advection is
     * UPWIND-biased (Wicker-Skamarock). `loading` is the host-pre-summed
     * hydrometeor mixing ratio (qc+qr+qi+qs+qg+qh).
     *
     * @return True if dispatch completed. False leaves recovery or
     *         fail-closed handling to the caller's runtime policy.
     */
    virtual bool dispatch_cartesian_cgrid_slow_tendencies(
        const CartesianCgridSlowTendenciesDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    // ── Cartesian tendencies dispatch ──────────────────────────────

    /**
     * @brief Execute Cartesian momentum tendencies on device.
     *
     * Computes 5 tendency fields from 6 input state fields plus 2 reference-
     * state profiles (p0_base, rho0_base) using centered finite differences
     * in Cartesian coordinates. The vertical momentum equation uses reference-
     * state subtraction to eliminate the discrete hydrostatic imbalance.
     *
     * @param p0_base_data   Reference pressure profile, 1D [nz], float.
     * @param rho0_base_data Reference density profile, 1D [nz], float.
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_cartesian_tendencies(
        const CartesianTendenciesDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
    // ── Acoustic substep dispatch ─────────────────────────────────────

    /**
     * @brief Fused acoustic pressure substep: divergence -> rho/p integration.
     *
     * Reads u,v,w,rho,p; computes the velocity divergence; writes updated rho,p.
     * Boundary points get zero-gradient extrapolation.
     *
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_acoustic_pressure(
        const AcousticPressureDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    /**
     * @brief Fused acoustic momentum substep: pressure gradient -> u/v/w integration.
     *
     * Reads updated rho/p (from pressure substep), current u/v/w;
     * computes pressure gradient; writes updated u/v/w.
     * Boundary points get antisymmetric u (r-faces) or zero-gradient BCs.
     *
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_acoustic_momentum(
        const AcousticMomentumDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    // ── Fused acoustic substep dispatch ──────────────────────────────

    /**
     * @brief Fused acoustic substep: pressure + momentum in a single GPU submission.
     *
     * Performs both the divergence -> rho/p update (forward phase) and the
     * pressure gradient -> u/v/w update (backward phase) using one command
     * buffer with a compute-to-compute barrier between the two kernels.
     * All 5 fields are updated in-place.
     *
     * Eliminates one full H2D/D2H round-trip per acoustic substep compared
     * to calling dispatch_acoustic_pressure + dispatch_acoustic_momentum
     * separately.
     *
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_acoustic_substep_fused(
        const AcousticSubstepFusedDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    // ── Batched acoustic substeps (all N substeps in one submission) ──

    /**
     * @brief Batched acoustic substeps: all N substeps in one GPU submission.
     *
     * Records N iterations of (pressure dispatch -> barrier -> momentum dispatch
     * -> barrier) in a single command buffer. Fields stay GPU-resident across
     * all substeps. Single H2D upload at start, single D2H download at end.
     *
     * The GPU shaders handle boundary conditions internally (rigid-lid w=0,
     * antisymmetric u, zero-gradient v/rho, hydrostatic p extrapolation at
     * z-faces; antisymmetric u, symmetric v/w/rho/p at r-faces).
     *
     * @param n_substeps Number of acoustic substeps to execute.
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_acoustic_substeps_batched(
        const AcousticSubstepsBatchedDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
};
