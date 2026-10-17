#pragma once

#include "compute/dispatch/common.hpp"

struct SimulationStateView;

/**
 * @file advection.hpp
 * @brief Advection compute dispatch data and capability interface.
 */

/**
 * @brief Describes the result of a vertical flux GPU dispatch.
 *
 * Carries CFL diagnostics back to the caller so the adaptive timestepper
 * can adjust dt without re-scanning the tendency array on the CPU.
 */
struct VerticalFluxDispatchResult
{
    double max_cfl_z = 0.0;
    double suggested_dt = 1.0e30;
};

enum class AdvectionKernelOp
{
    VerticalFlux,
    ConservativeCartesianCgridTransport,
    AdvectionX,
    AdvectionY,
    BatchedAdvection
};

struct ScalarAdvectionDispatch
{
    const float* src = nullptr;
    const float* velocity = nullptr;
    float* dst = nullptr;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    float primary_spacing = 0.0f;
    float secondary_spacing = 0.0f;
    float dt = 0.0f;
};

struct ConservativeCartesianCgridTransportDispatch
{
    const float* u = nullptr;
    const float* v = nullptr;
    const float* w = nullptr;
    const float* rho_data = nullptr;
    const float* const* q_in = nullptr;
    float* const* q_out = nullptr;
    const float* dt_kappas = nullptr;
    const int* positive_definite = nullptr;
    int n_scalars = 0;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float inv_dx = 0.0f;
    float inv_dy = 0.0f;
    float inv_dz = 0.0f;
    float inv_dx2 = 0.0f;
    float inv_dy2 = 0.0f;
    float inv_dz2 = 0.0f;
    float dt_step = 0.0f;
    float rho_floor = 0.0f;
    // Explicit SCI-025 coupled mode. When true, these are the time-averaged
    // face mass fluxes accumulated by the split continuity solve. A backend
    // must not silently recompute instantaneous fluxes in this mode.
    const float* density_begin = nullptr;
    const float* mass_flux_x = nullptr;
    const float* mass_flux_y = nullptr;
    const float* mass_flux_z = nullptr;
    bool use_supplied_mass_flux = false;

    // Optional host-native views. The CPU family consumes these to preserve
    // Field3D ownership, fp64 kappa/dt, and the validated retained context;
    // device backends continue to use the raw ABI above.
    Field3D* const* host_fields = nullptr;
    const double* host_kappas = nullptr;
    const tmv::CartesianCGridTransportContext* host_context = nullptr;
    double dt_step_host = 0.0;

    // Optional operator-time science outputs. Device backends must populate
    // every non-null report before committing its corresponding host field.
    // Null entries carry no reporting cost. Energy kinds and reports are a
    // fail-closed pair for each scalar.
    tmv::ScalarBoundaryTransportReport* const* host_boundary_reports = nullptr;
    const tmv::ScalarMoistEnergyKind* host_energy_kinds = nullptr;
    tmv::ScalarMoistEnergyTransportReportV1* const*
        host_energy_reports = nullptr;

    // Earth-relative velocity of the translating grid. Only the standalone
    // mass-flux producer consumes these; supplied acoustic mass fluxes have
    // already been transformed into the grid frame.
    float frame_velocity_u_ms = 0.0f;
    float frame_velocity_v_ms = 0.0f;
    const SimulationStateView* execution_state     = nullptr;
    tmv::CartesianInterfaceFluxRegister* const* host_interface_registers = nullptr;
    bool                                        weno                      = true;
    bool                                        preserve_positive_support = false;
};

struct AdvectionBatchPreVerticalDispatch
{
    const float* scalar_in = nullptr;
    float* result_out = nullptr;
    const float* u_data = nullptr;
    const float* v_data = nullptr;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    float dr = 0.0f;
    float dtheta = 0.0f;
    float dt_half = 0.0f;
};

struct AdvectionBatchPostVerticalDispatch
{
    const float* scalar_in = nullptr;
    float* result_out = nullptr;
    const float* u_data = nullptr;
    const float* v_data = nullptr;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    float dr = 0.0f;
    float dtheta = 0.0f;
    float dz = 0.0f;
    float dt_half = 0.0f;
    float dt_full = 0.0f;
    float kappa = 0.0f;
};

struct VerticalFluxDispatch
{
    const float* q_data = nullptr;
    const float* w_data = nullptr;
    float* dqdt_data = nullptr;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    const double* dz_data = nullptr;
    int dz_count = 0;
    int limiter_id = 0;
    bool positivity = false;
    double positivity_dt = 0.0;
    double cfl_target = 0.0;
    VerticalFluxDispatchResult* result = nullptr;
};


class AdvectionComputeKernels
{
public:
    virtual ~AdvectionComputeKernels() = default;

    /**
     * @brief Returns true when this backend can execute the requested advection operation.
     */
    virtual bool supports(AdvectionKernelOp /*op*/) const { return false; }

    /**
     * @brief Whether coupled scalar transport can consume supplied averaged
     *        Cartesian C-grid face mass fluxes.
     *
     * Kept separate from the legacy operation capability so an older backend
     * cannot accept a coupled dispatch and silently use end-state rho/u.
     */
    virtual bool supports_supplied_cartesian_cgrid_mass_flux() const
    {
        return false;
    }

    /**
     * @brief Execute TVD vertical flux-divergence kernel on device.
     *
     * Accepts raw pointers and scalars to avoid pulling meteorology headers into
     * backend implementations. The caller marshals from domain types.
     *
     * @param q_data      Scalar field, row-major [nr][nth][nz], float.
     * @param w_data      Vertical velocity, same layout, float.
     * @param dqdt_data   Output tendency, same layout, float. Accumulated (+=).
     * @param nr          Radial dimension.
     * @param nth         Azimuthal dimension.
     * @param nz          Vertical dimension.
     * @param dz_data     Vertical spacing array, length nz, double.
     * @param dz_count    Length of dz_data (== nz).
     * @param limiter_id  TVD limiter: 0=minmod, 1=vanleer, 2=superbee, 3=mc, 4=universal.
     * @param positivity  Enable positivity limiting.
     * @param positivity_dt  Timestep used for positivity clamp.
     * @param cfl_target  Target CFL for suggested dt.
     * @param out_result  CFL diagnostics written here on success.
     * @return True if dispatch completed; false leaves recovery or fail-closed
     *         handling to the caller's runtime policy.
     */
    virtual bool dispatch_vertical_flux(const VerticalFluxDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    // ── Conservative scalar transport (GPU audit Phase 3.2) ─────

    /**
     * @brief Advect a batch of scalars with the conservative rho-weighted
     *        flux-form WENO5/MC transport on the Cartesian C-grid, in ONE
     *        GPU submission: the face mass-flux is computed once and shared
     *        by every scalar; each scalar then runs its face-flux,
     *        divergence+recovery. Scalar diffusion is a separate
     *        live-density finite-volume operator.
     *
     * Mirrors advect_scalar_conservative_3d_cartesian for each scalar (the
     * CPU recomputes the mass flux per scalar; the batch does not - same
     * arithmetic, shared inputs).
     *
     * @param q_in      Per-scalar input fields (n_scalars pointers).
     * @param q_out     Per-scalar output fields; q_out[s] may equal q_in[s].
     * @param dt_kappas Retained ABI slots; every entry must be exactly zero.
     *                  Nonzero transport-chained point diffusion is rejected.
     * @param inv_d*    1/spacing and 1/spacing^2, divided in double and
     *                  rounded once (GPU audit section 2.3).
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_advect_scalars_conservative_cartesian_cgrid(
        const ConservativeCartesianCgridTransportDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
    // ── Cartesian x-advection dispatch ──────────────────────────────

    virtual bool dispatch_advection_x(const ScalarAdvectionDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    // ── Cartesian y-advection dispatch ──────────────────────────────

    virtual bool dispatch_advection_y(const ScalarAdvectionDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
    // ── Batched advection dispatch ─────────────────────────────────

    /**
     * @brief Execute pre-vertical advection batch on device.
     *
     * Records radial(dt_half) → azimuthal(dt_half) in a single command
     * buffer submission, eliminating one intermediate GPU round-trip.
     *
     * @param scalar_in     Input scalar field, row-major [nr][nth][nz], float.
     * @param result_out    Output field after radial+azimuthal, same layout.
     * @param u_data        Radial velocity field, same layout.
     * @param v_data  Azimuthal velocity field, same layout.
     * @param nr            Radial dimension.
     * @param nth           Azimuthal dimension.
     * @param nz            Vertical dimension.
     * @param dr            Radial grid spacing.
     * @param dtheta        Azimuthal grid spacing.
     * @param dt_half       Half timestep.
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_advection_batch_pre_vertical(
        const AdvectionBatchPreVerticalDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    /**
     * @brief Execute post-vertical advection batch on device.
     *
     * Legacy unavailable hook retained for ABI compatibility.
     *
     * @param scalar_in     Input scalar field (output of vertical step).
     * @param result_out    Final output field after azimuthal+radial+diffusion.
     * @param u_data        Radial velocity field.
     * @param v_data  Azimuthal velocity field.
     * @param nr            Radial dimension.
     * @param nth           Azimuthal dimension.
     * @param nz            Vertical dimension.
     * @param dr            Radial grid spacing.
     * @param dtheta        Azimuthal grid spacing.
     * @param dz            Vertical grid spacing.
     * @param dt_half       Half timestep (for azimuthal+radial).
     * @param dt_full       Full timestep (for diffusion).
     * @param kappa         Must be zero if an implementation is restored.
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_advection_batch_post_vertical(
        const AdvectionBatchPostVerticalDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
};
