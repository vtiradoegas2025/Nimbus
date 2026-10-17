/**
 * @file advection_cartesian.hpp
 * @brief Cartesian-grid scalar advection helpers.
 *
 * This header declares three free functions implemented in
 * `src/numerics/advection/advection_cartesian.cpp` that compute a scalar
 * advection step — first-order upwind + Forward-Euler — with straight
 * Cartesian indexing in (x, y, z):
 *
 *   1. The y kernel carries no `1/r` metric factor; the y-flux uses the
 *      plain `dy` arc length.
 *   2. The y kernel uses no periodic `j ↔ j + NTH` wraparound. The
 *      j = 0 and j = NTH − 1 faces are physical boundaries.
 *
 * The dispatcher in `src/numerics/advection/advection.cpp::advect_scalar_3d`
 * calls these helpers in the Strang-like split sequence
 * (`x/2 → y/2 → z → y/2 → x/2 → diffusion`). The vertical TVD scheme
 * touches no horizontal metric and is reused verbatim.
 *
 * Wind field naming: the globals are:
 *   u  — horizontal-1 (zonal, x)
 *   v  — horizontal-2 (meridional, y)
 *   w  — vertical (z)
 *
 * Grid spacing note: the `dr`/`dtheta` geometry globals are retained as
 * Cartesian aliases (rename to `dx`/`dy` deferred). `dr` is dx and `dtheta`
 * is the physical y spacing in metres.
 */

#pragma once

#include <utility>

struct SimulationStateView;

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "core/field/field3d.hpp"
#include "core/infra/coordinate_system.hpp"
#include "numerics/advection/cartesian_interface_flux_register.hpp"
#include "numerics/advection/cartesian_mass_flux.hpp"
#include "numerics/advection/scalar_boundary_report.hpp"

namespace tmv
{
enum class AcousticMassFluxPrecision
{
    Host,
    NativeFloat,
    // Matches resident GPU flux accumulation even when host contraction is disabled.
    NativeCompensated
};

enum class CartesianFaceReconstruction
{
    MonotonicCentered,
    WENO5,
    MonotonicCenteredPositiveSupport,
    WENO5PositiveSupport,
};

inline bool uses_weno(CartesianFaceReconstruction reconstruction) noexcept
{
    return reconstruction == CartesianFaceReconstruction::WENO5 ||
           reconstruction == CartesianFaceReconstruction::WENO5PositiveSupport;
}

inline bool preserves_positive_support(CartesianFaceReconstruction reconstruction) noexcept
{
    return reconstruction == CartesianFaceReconstruction::MonotonicCenteredPositiveSupport ||
           reconstruction == CartesianFaceReconstruction::WENO5PositiveSupport;
}

inline double support_preserving_face(double donor, double reconstructed) noexcept
{
    return donor <= 0 ? 0 : (reconstructed <= 0 ? donor : reconstructed);
}

struct DynamicsBoundaryMassReport;
struct TerrainMetrics3D;
struct TerrainContravariantFluxes;

namespace cartesian_advection_detail
{

/**
 * Store an active positive-definite donor ratio as a strict upper bound.
 *
 * The caller supplies the FP64 ratio before conversion. A ratio just below
 * one can round to FP32 1.0, so activity must be decided from the FP64 value,
 * not from the rounded storage value. The shared six-face FP32 update has a
 * standard operation-chain bound below 32*epsilon_float; reserve that amount
 * and then move once toward zero. An exactly inactive ratio of one remains
 * one and preserves free streams.
 */
inline float conservative_pd_limiter_factor(double ratio, bool active_at_one = false) noexcept
{
    if (!(ratio > 0.0 && (ratio < 1.0 || (active_at_one && ratio == 1.0))))
        return static_cast<float>(ratio);
    constexpr float roundoff_factor = 1.0f - 32.0f * std::numeric_limits<float>::epsilon();
    const float     stored          = static_cast<float>(ratio);
    return std::nextafter(stored * roundoff_factor, 0.0f);
}

} // namespace cartesian_advection_detail

/** Optional moist-mixture fields needed only by SCI-150 boundary-energy
 * attribution. Ordinary retained scalar transport passes no object and pays
 * no validation or boundary-reduction cost. */
struct CarrierKineticBoundaryInputs
{
    std::array<const Field3D*, 7> water{};

    [[nodiscard]] bool complete() const noexcept
    {
        for (const Field3D* field : water)
            if (field == nullptr)
                return false;
        return true;
    }
};

/**
 * @brief Time-averaged Cartesian face mass fluxes from one split-explicit step.
 *
 * The acoustic continuity solve advances density over several small steps.  A
 * scalar transported after that solve must use the same time-integrated face
 * mass fluxes, rather than recomputing one flux from the retained end state.
 * This step-scoped context owns the exact beginning density and those three
 * averaged face-flux fields:
 *
 *   Mbar = sum_s (dt_s / dt_large) * (rho u)_s.
 *
 * `begin()` resets the context, the first `accumulate()` captures the density
 * paired with the unchanged scalar state, and every call samples one
 * pre-continuity acoustic state. `complete()` seals it. Coupled scalar
 * transport rejects a context unless `validate()` confirms its domain,
 * beginning density, timestep, and complete accumulated duration.
 */
class CartesianCGridTransportContext
{
public:
    class PreparedExternalAcousticLoop
    {
    public:
        PreparedExternalAcousticLoop(const PreparedExternalAcousticLoop&) = delete;
        PreparedExternalAcousticLoop& operator=(const PreparedExternalAcousticLoop&) = delete;
        PreparedExternalAcousticLoop(PreparedExternalAcousticLoop&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), density_(std::move(other.density_)),
              total_dt_(other.total_dt_), substeps_(other.substeps_) {}
        // The producer must finish and validate every flux before publication;
        // the owning context remains unchanged between preparation and commit.
        void commit() noexcept;

    private:
        friend class CartesianCGridTransportContext;
        PreparedExternalAcousticLoop(CartesianCGridTransportContext& owner, const Field3D& density,
                                     double total_dt, int substeps)
            : owner_(&owner), density_(density), total_dt_(total_dt), substeps_(substeps) {}
        CartesianCGridTransportContext* owner_;
        Field3D density_;
        double total_dt_;
        int substeps_;
    };

    PreparedExternalAcousticLoop prepare_external_acoustic_loop(
        const Field3D& density, double total_dt, int substeps);

    Field3D density_begin;
    Field3D mass_flux_x;
    Field3D mass_flux_y;
    Field3D mass_flux_z;

    void begin(int                     nr,
               int                     nth,
               int                     nz,
               double                  dt_large,
               const TerrainMetrics3D* terrain_metrics = nullptr,
               double                  frame_u_ms      = 0.0,
               double                  frame_v_ms      = 0.0);
    void begin(const SimulationStateView& execution_state, int                     nr,
               int                     nth,
               int                     nz,
               double                  dt_large,
               const TerrainMetrics3D* terrain_metrics = nullptr,
               double                  frame_u_ms      = 0.0,
               double                  frame_v_ms      = 0.0);

    void accumulate(const Field3D&                                         rho_state,
                    const Field3D&                                         u_state,
                    const Field3D&                                         v_state,
                    const Field3D&                                         w_state,
                    double                                                 dt_small,
                    const cartesian_mass_flux::CollocatedPressureCoupling* pressure_coupling      = nullptr,
                    const CarrierKineticBoundaryInputs*                    carrier_kinetic_inputs = nullptr,
                    AcousticMassFluxPrecision precision = AcousticMassFluxPrecision::Host);
    void accumulate(const SimulationStateView& execution_state, const Field3D&                                         rho_state,
                    const Field3D&                                         u_state,
                    const Field3D&                                         v_state,
                    const Field3D&                                         w_state,
                    double                                                 dt_small,
                    const cartesian_mass_flux::CollocatedPressureCoupling* pressure_coupling      = nullptr,
                    const CarrierKineticBoundaryInputs*                    carrier_kinetic_inputs = nullptr,
                    AcousticMassFluxPrecision precision = AcousticMassFluxPrecision::Host);

    /** Capture the density paired with the scalar state entering this step. */
    void capture_density_begin(const Field3D& rho_state);

    void set_beginning_geometry(const TerrainMetrics3D& metrics);
    void accumulate_integrated_moving_fluxes(const TerrainContravariantFluxes& fluxes, double interval_s);
    [[nodiscard]] bool has_moving_geometry() const noexcept { return !jacobian_begin_.empty(); }
    [[nodiscard]] double beginning_jacobian(int i, int j, int k) const;
    [[nodiscard]] double begin_density_in_current_geometry(int i, int j, int k) const;


    void capture_instantaneous(
        const Field3D& density, const Field3D& u, const Field3D& v, const Field3D& w,
        double dt, double frame_u_ms = 0.0, double frame_v_ms = 0.0);
    void capture_instantaneous(const SimulationStateView& execution_state,
        const Field3D& density, const Field3D& u, const Field3D& v, const Field3D& w,
        double dt, double frame_u_ms = 0.0, double frame_v_ms = 0.0);

    /**
     * Record duration/count metadata after an accelerator has accumulated
     * directly into mass_flux_x/y/z. Set `fp32_time_currency` when the
     * accelerator executed the substep duration after conversion to float;
     * validation will then admit only the bounded conversion roundoff.
     */
    void record_external_accumulation(double total_dt, int substeps, bool fp32_time_currency = false);

    void record_external_carrier_kinetic_transport(double outward_j, int substeps = 1);

    /** Seal the context after all acoustic substeps have contributed. */
    void complete();
    void complete(const SimulationStateView& execution_state);

    [[nodiscard]] bool validate(int nr, int nth, int nz, double dt_large, std::string* reason = nullptr) const;
    [[nodiscard]] bool validate(const SimulationStateView& execution_state, int nr, int nth, int nz, double dt_large, std::string* reason = nullptr) const;

    /** Throw std::runtime_error when validate() would fail. */
    void require_valid(int nr, int nth, int nz, double dt_large) const;
    void require_valid(const SimulationStateView& execution_state, int nr, int nth, int nz, double dt_large) const;

    /** Close the physical dry-mass inventory against this exact retained
     * acoustic face-flux history.  The context must already be sealed. */
    [[nodiscard]] DynamicsBoundaryMassReport
    boundary_mass_report(const Field3D& density_end, double dx_m, double dy_m, double dz_m) const;
    [[nodiscard]] DynamicsBoundaryMassReport
    boundary_mass_report(const SimulationStateView& execution_state, const Field3D& density_end, double dx_m, double dy_m, double dz_m) const;

    [[nodiscard]] bool active() const noexcept
    {
        return active_;
    }
    [[nodiscard]] bool is_complete() const noexcept
    {
        return complete_;
    }
    [[nodiscard]] double dt_large() const noexcept
    {
        return dt_large_;
    }
    [[nodiscard]] double accumulated_dt() const noexcept
    {
        return accumulated_dt_;
    }
    [[nodiscard]] int substep_count() const noexcept
    {
        return substep_count_;
    }
    [[nodiscard]] double outward_carrier_kinetic_transport_j() const noexcept
    {
        return outward_carrier_kinetic_transport_j_;
    }
    [[nodiscard]] bool carrier_kinetic_transport_complete() const noexcept
    {
        return carrier_kinetic_tracking_ && carrier_kinetic_substep_count_ == substep_count_;
    }
    [[nodiscard]] bool has_density_begin() const noexcept
    {
        return density_begin_captured_;
    }
    [[nodiscard]] StaggerType staggering() const noexcept
    {
        return staggering_;
    }
    [[nodiscard]] double frame_u_ms() const noexcept
    {
        return frame_u_ms_;
    }
    [[nodiscard]] double frame_v_ms() const noexcept
    {
        return frame_v_ms_;
    }
    [[nodiscard]] const TerrainMetrics3D* terrain_metrics() const noexcept
    {
        return terrain_metrics_;
    }

    /**
     * Clear one completed/prepared generation without releasing field
     * storage. Callers must reject an active incomplete transaction first.
     */
    void invalidate_generation_state_after_domain_recycling() noexcept
    {
        density_begin.fill(0.0f);
        mass_flux_x.fill(0.0f);
        mass_flux_y.fill(0.0f);
        mass_flux_z.fill(0.0f);
        nr_                                  = 0;
        nth_                                 = 0;
        nz_                                  = 0;
        dt_large_                            = 0.0;
        accumulated_dt_                      = 0.0;
        substep_count_                       = 0;
        carrier_kinetic_substep_count_       = 0;
        active_                              = false;
        complete_                            = false;
        finite_                              = true;
        density_begin_captured_              = false;
        fp32_time_currency_                  = false;
        carrier_kinetic_tracking_            = false;
        boundary_flux_compatible_            = true;
        lateral_bc_west_                     = 0;
        lateral_bc_east_                     = 0;
        lateral_bc_south_                    = 0;
        lateral_bc_north_                    = 0;
        bottom_bc_                           = 0;
        top_bc_                              = 0;
        outward_carrier_kinetic_transport_j_ = 0.0;
        frame_u_ms_                          = 0.0;
        frame_v_ms_                          = 0.0;
        staggering_                          = StaggerType::CGrid;
        terrain_metrics_                     = nullptr;
        jacobian_begin_.clear();
    }

private:
    void accumulate_impl(
        const Field3D& rho_state, const Field3D& u_state,
        const Field3D& v_state, const Field3D& w_state, double dt_small,
        const cartesian_mass_flux::CollocatedPressureCoupling* pressure_coupling,
        const CarrierKineticBoundaryInputs* carrier_kinetic_inputs,
        AcousticMassFluxPrecision precision, bool instantaneous);
    void accumulate_impl(const SimulationStateView& execution_state,
        const Field3D& rho_state, const Field3D& u_state,
        const Field3D& v_state, const Field3D& w_state, double dt_small,
        const cartesian_mass_flux::CollocatedPressureCoupling* pressure_coupling,
        const CarrierKineticBoundaryInputs* carrier_kinetic_inputs,
        AcousticMassFluxPrecision precision, bool instantaneous);
    std::vector<double> jacobian_begin_;
    int                     nr_                                  = 0;
    int                     nth_                                 = 0;
    int                     nz_                                  = 0;
    double                  dt_large_                            = 0.0;
    double                  accumulated_dt_                      = 0.0;
    int                     substep_count_                       = 0;
    int                     carrier_kinetic_substep_count_       = 0;
    bool                    active_                              = false;
    bool                    complete_                            = false;
    bool                    finite_                              = true;
    bool                    density_begin_captured_              = false;
    bool                    fp32_time_currency_                  = false;
    bool                    carrier_kinetic_tracking_            = false;
    bool                    boundary_flux_compatible_            = true;
    int                     lateral_bc_west_                     = 0;
    int                     lateral_bc_east_                     = 0;
    int                     lateral_bc_south_                    = 0;
    int                     lateral_bc_north_                    = 0;
    int                     bottom_bc_                           = 0;
    int                     top_bc_                              = 0;
    double                  outward_carrier_kinetic_transport_j_ = 0.0;
    double                  frame_u_ms_                          = 0.0;
    double                  frame_v_ms_                          = 0.0;
    StaggerType             staggering_                          = StaggerType::CGrid;
    const TerrainMetrics3D* terrain_metrics_                     = nullptr;
};

/**
 * @brief First-order upwind scalar advection in the x direction.
 *
 * Reads the global `u` field as the x velocity. Uses `dr` as the x cell
 * spacing. Steps the source field by `dt` with Forward Euler:
 *
 *   q^{n+1}[i][j][k] = q^n[i][j][k] − dt · u[i][j][k] · (dq/dx)_upwind
 *
 * where the upwind difference is selected per cell on the sign of `u`:
 *
 *   u > 0  →  dq/dx = (q[i  ] − q[i−1]) / dx
 *   u < 0  →  dq/dx = (q[i+1] − q[i  ]) / dx
 *   u = 0  →  dq/dx = 0
 *
 * Boundary handling (Cartesian, zero-gradient):
 *   - x faces (i = 0 and i = NR − 1): `dst` is seeded from `src`, no
 *     interior update is written, so the boundary value carries through
 *     unchanged. This is the discrete equivalent of `∂q/∂x = 0` at the
 *     wall.
 *   - y faces (j = 0 and j = NTH − 1): the i-loop walks all interior
 *     cells regardless of j, so j-boundary cells are written by this
 *     kernel. The y kernel that runs after will leave them in the
 *     zero-gradient state.
 *   - z faces (k = 0 and k = NZ − 1): the k-loop runs from 1 to NZ − 2,
 *     so the floor and lid are not modified.
 *
 * Preconditions:
 *   - `src.size_r() == NR`, `src.size_th() == NTH`, `src.size_z() == NZ`
 *   - `dst` will be resized to match `src` if it isn't already
 *   - `dr > 0`
 *   - `u` is sized (NR, NTH, NZ)
 *
 * @param src  Input scalar field (read-only).
 * @param dst  Output scalar field (overwritten).
 * @param dt   Forward-Euler step length in seconds.
 */
void advect_scalar_1d_x_kernel_cartesian(const Field3D& src, Field3D& dst, double dt);
    void advect_scalar_1d_x_kernel_cartesian(const SimulationStateView& execution_state, const Field3D& src, Field3D& dst, double dt);

/**
 * @brief First-order upwind scalar advection in the y direction.
 *
 * Reads the global `v` field as the y velocity (Phase-A field
 * aliasing — `v` carries `u_y` while the Cartesian backend is
 * active). Uses `dtheta` as the y cell spacing. Steps the source field by
 * `dt` with Forward Euler:
 *
 *   q^{n+1}[i][j][k] = q^n[i][j][k] − dt · v[i][j][k] · (dq/dy)_upwind
 *
 * **No `1/r` factor.** **No periodic `(j ± 1) % NTH` wraparound.** The
 * j = 0 and j = NTH − 1 faces are physical boundaries.
 *
 * Boundary handling:
 *   - y faces (j = 0 and j = NTH − 1): not written; carry the seeded
 *     `src` value (zero-gradient).
 *   - x faces (i = 0 and i = NR − 1): not written.
 *   - z faces (k = 0 and k = NZ − 1): not written.
 *
 * Preconditions:
 *   - `src.size_r() == NR`, `src.size_th() == NTH`, `src.size_z() == NZ`
 *   - `dst` will be resized to match `src` if it isn't already
 *   - `dtheta > 0`
 *   - `v` is sized (NR, NTH, NZ)
 *
 * @param src  Input scalar field (read-only).
 * @param dst  Output scalar field (overwritten).
 * @param dt   Forward-Euler step length in seconds.
 */
void advect_scalar_1d_y_kernel_cartesian(const Field3D& src, Field3D& dst, double dt);
    void advect_scalar_1d_y_kernel_cartesian(const SimulationStateView& execution_state, const Field3D& src, Field3D& dst, double dt);

/**
 * @brief Cartesian Laplacian-with-Forward-Euler diffusion kernel.
 *
 * Compatibility boundary for the removed transport-chained point filter.
 * `kappa == 0` copies `src` to `dst`; every nonzero or non-finite value is
 * rejected. Physical scalar diffusion must use the live-density flux-form
 * diffusion module.
 *
 * Preconditions:
 *   - `src.size_r() == NR`, `src.size_th() == NTH`, `src.size_z() == NZ`
 *   - `dst` will be resized to match `src` if it isn't already
 *   - `dr > 0`, `dtheta > 0`, `dz > 0`
 *
 * @param src    Input scalar field (read-only).
 * @param dst    Output scalar field (overwritten).
 * @param dt     Step length in seconds.
 * @param kappa  Must be exactly zero.
 */
void apply_diffusion_kernel_cartesian(const Field3D& src, Field3D& dst, double dt, double kappa);
    void apply_diffusion_kernel_cartesian(const SimulationStateView& execution_state, const Field3D& src, Field3D& dst, double dt, double kappa);

/**
 * @brief Conservative rho-weighted flux-form scalar transport on Cartesian grids.
 *
 * One unsplit 3D finite-volume update that advects rho*phi using the same face
 * mass fluxes as the mass-continuity operator. Standalone calls predict their
 * matching density denominator. Coupled calls consume the time-averaged flux
 * history and exact beginning density in `context`, advance rho_begin*phi with
 * those fluxes, and divide by retained rho_end without advancing density again.
 * Both forms are conservative and
 * free-stream preserving (a constant phi is preserved to round-off).
 *
 * Reads the globals rho, u, v, w, dr (x spacing), dtheta (y spacing), and dz.
 * C-grid velocities already occupy faces. A completed collocated context owns
 * the A-grid face reconstruction used by continuity. C-grid vertical cells and
 * A-grid interior vertical cells use their respective dynamics ownership
 * domains. The MC/WENO reconstruction is conservative on shared faces.
 *
 * @param src  Input scalar field (read-only).
 * @param dst  Output scalar field (overwritten). Exact in-place operation is
 *             supported; face fluxes are complete before cell values change.
 * @param dt   Step length in seconds.
 * @param context Optional completed split-continuity flux history. Null keeps
 *                the standalone legacy contract.
 * @param boundary_report Optional exact boundary/commit report. Reporting
 *                requires a completed retained context; requesting it for a
 *                standalone update fails before the output is touched.
 */
/// positive_definite enables the Skamarock-Weisman 2006 outgoing-flux
/// renormalization (exactly conservative, q_out >= 0 guaranteed); use it
/// for moisture and tracers, not theta.
void advect_scalar_conservative_3d_cartesian(
    const Field3D&                        src,
    Field3D&                              dst,
    double                                dt,
    bool                                  positive_definite = false,
    const CartesianCGridTransportContext* context           = nullptr,
    ScalarBoundaryTransportReport*        boundary_report   = nullptr,
    ScalarMoistEnergyKind                 energy_kind       = ScalarMoistEnergyKind::None,
    ScalarMoistEnergyTransportReportV1*   energy_report     = nullptr,
    CartesianFaceReconstruction           reconstruction    = CartesianFaceReconstruction::WENO5,
    CartesianInterfaceFluxRegister*       interface_flux_register = nullptr);
    void advect_scalar_conservative_3d_cartesian(const SimulationStateView& execution_state,
    const Field3D&                        src,
    Field3D&                              dst,
    double                                dt,
    bool                                  positive_definite = false,
    const CartesianCGridTransportContext* context           = nullptr,
    ScalarBoundaryTransportReport*        boundary_report   = nullptr,
    ScalarMoistEnergyKind                 energy_kind       = ScalarMoistEnergyKind::None,
    ScalarMoistEnergyTransportReportV1*   energy_report     = nullptr,
    CartesianFaceReconstruction           reconstruction    = CartesianFaceReconstruction::WENO5,
    CartesianInterfaceFluxRegister*       interface_flux_register = nullptr);

// Instantaneous conservative RHS d(rho*q)/dt. The timestep is used only
// to bound donor fluxes; density recovery belongs to the RK stage commit.
void compute_conservative_scalar_mass_tendency_cartesian(
    const Field3D& src, Field3D& tendency, double positivity_dt,
    bool positive_definite, const CartesianCGridTransportContext& context,
    CartesianFaceReconstruction reconstruction);
    void compute_conservative_scalar_mass_tendency_cartesian(const SimulationStateView& execution_state,
    const Field3D& src, Field3D& tendency, double positivity_dt,
    bool positive_definite, const CartesianCGridTransportContext& context,
    CartesianFaceReconstruction reconstruction);

/** One scalar in a caller-owned Cartesian CPU transport batch. */
struct CartesianScalarTransport
{
    Field3D*                            field             = nullptr;
    double                              kappa             = 0.0;
    bool                                positive_definite = false;
    ScalarBoundaryTransportReport*      boundary_report   = nullptr;
    ScalarMoistEnergyKind               energy_kind       = ScalarMoistEnergyKind::None;
    ScalarMoistEnergyTransportReportV1* energy_report     = nullptr;
    CartesianInterfaceFluxRegister*     interface_flux_register = nullptr;
};

/**
 * Reusable host workspace for attributing the retained flat-C-grid device
 * scalar commit. The four fields mirror the device's final scalar-face fluxes
 * and donor limiter. Capacity is retained after warm-up; no old-scalar or
 * final-state snapshot is stored.
 */
struct FlatCGridDeviceScalarAttributionWorkspace
{
    Field3D scalar_flux_x;
    Field3D scalar_flux_y;
    Field3D scalar_flux_z;
    Field3D donor_limiter;
};

/**
 * Attribute one completed flat Cartesian C-grid device scalar commit.
 *
 * `scalar_begin` is the still-unmodified host field and `scalar_end` is the
 * post-fence device result. The routine literally mirrors the FP32
 * reconstruction, limiter, and requested rho*scalar update used by the
 * Vulkan shaders, then reduces the ordinary schema-v1/basis-v2 reports with
 * the same operator-time thermodynamic coefficients as the CPU path.
 *
 * This function never mutates either scalar and performs no allocation after
 * `workspace` has reached the active grid shape.
 */
// Native audits provide unaggregated W/E/S/N/top fluxes and one value per cell:
// a donor-limited net scalar mass rate for positive fields, or the theta increment.
// The null pair retains the reference replay; device denormals cannot be reconstructed reliably on the host.
void attribute_flat_cgrid_device_scalar_commit(const float*                               scalar_begin,
                                               const float*                               scalar_end,
                                               const float*                               density_begin,
                                               const float*                               density_end,
                                               const float*                               mass_flux_x,
                                               const float*                               mass_flux_y,
                                               const float*                               mass_flux_z,
                                               int                                        nx,
                                               int                                        ny,
                                               int                                        nz,
                                               float                                      inv_dx,
                                               float                                      inv_dy,
                                               float                                      inv_dz,
                                               float                                      dt_step,
                                               bool                                       positive_definite,
                                               ScalarBoundaryTransportReport*             boundary_report,
                                               ScalarMoistEnergyKind                      energy_kind,
                                               ScalarMoistEnergyTransportReportV1*        energy_report,
                                               FlatCGridDeviceScalarAttributionWorkspace& workspace,
                                               const float* device_boundary_fluxes = nullptr,
                                               const float* device_requested_update = nullptr);
    void attribute_flat_cgrid_device_scalar_commit(const SimulationStateView& execution_state, const float*                               scalar_begin,
                                               const float*                               scalar_end,
                                               const float*                               density_begin,
                                               const float*                               density_end,
                                               const float*                               mass_flux_x,
                                               const float*                               mass_flux_y,
                                               const float*                               mass_flux_z,
                                               int                                        nx,
                                               int                                        ny,
                                               int                                        nz,
                                               float                                      inv_dx,
                                               float                                      inv_dy,
                                               float                                      inv_dz,
                                               float                                      dt_step,
                                               bool                                       positive_definite,
                                               ScalarBoundaryTransportReport*             boundary_report,
                                               ScalarMoistEnergyKind                      energy_kind,
                                               ScalarMoistEnergyTransportReportV1*        energy_report,
                                               FlatCGridDeviceScalarAttributionWorkspace& workspace,
                                               const float* device_boundary_fluxes = nullptr,
                                               const float* device_requested_update = nullptr);

/**
 * @brief Advance a set of independent scalars against one retained mass-flux
 *        context on the CPU.
 *
 * The completed split-continuity context supplies one authoritative set of
 * time-averaged MFX/MFY/MFZ fields for the whole batch. Each scalar is updated
 * in place using shared scalar-flux/limiter scratch; a single reusable field is
 * used only when a member requests chained diffusion. A null field, invalid
 * count, or missing/incomplete context fails before any scalar is modified.
 */
void advect_scalars_conservative_3d_cartesian(
    CartesianScalarTransport*             scalars,
    std::size_t                           count,
    double                                dt,
    const CartesianCGridTransportContext* context,
    CartesianFaceReconstruction           reconstruction = CartesianFaceReconstruction::WENO5);
    void advect_scalars_conservative_3d_cartesian(const SimulationStateView& execution_state,
    CartesianScalarTransport*             scalars,
    std::size_t                           count,
    double                                dt,
    const CartesianCGridTransportContext* context,
    CartesianFaceReconstruction           reconstruction = CartesianFaceReconstruction::WENO5);

} // namespace tmv

using tmv::advect_scalar_1d_x_kernel_cartesian;
using tmv::advect_scalar_1d_y_kernel_cartesian;
using tmv::advect_scalar_conservative_3d_cartesian;
using tmv::advect_scalars_conservative_3d_cartesian;
using tmv::apply_diffusion_kernel_cartesian;
using tmv::CartesianCGridTransportContext;
using tmv::CartesianFaceReconstruction;
using tmv::CartesianInterfaceFluxRegister;
using tmv::CartesianScalarTransport;
using tmv::FlatCGridDeviceScalarAttributionWorkspace;
using tmv::ScalarBoundaryTransportReport;
using tmv::ScalarMoistEnergyKind;
using tmv::ScalarMoistEnergyTransportReportV1;
