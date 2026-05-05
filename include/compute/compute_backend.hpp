#pragma once

#include "core/coordinate_system.hpp"

#include <string>
#include <vector>

struct HardwareInfo; // Forward declaration — defined in core/hardware_info.hpp

/**
 * @file compute_backend.hpp
 * @brief Runtime selection and lifecycle contract for simulation compute backends.
 *
 * This scaffold introduces backend indirection so heavy kernels can be moved
 * from CPU to accelerator implementations incrementally while preserving a
 * deterministic CPU fallback path.
 */

enum class ComputeBackendKind
{
    Cpu,
    Vulkan
};

struct ComputeBackendConfig
{
    std::string backend = "cpu";
    int device_index = -1;
    bool allow_fallback = true;
    bool validate_parity = false;
};

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

class ComputeBackend
{
public:
    virtual ~ComputeBackend() = default;

    /**
     * @brief Backend identifier used for logs and config echo.
     */
    virtual std::string name() const = 0;

    /**
     * @brief Initializes backend resources.
     * @param error Diagnostic on failure.
     * @return True on success.
     */
    virtual bool initialize(std::string& error) = 0;

    /**
     * @brief Releases backend resources.
     */
    virtual void shutdown() = 0;

    /**
     * @brief Sets the coordinate system for dispatch routing.
     *
     * Called once after initialization. Backends that have coordinate-system-
     * specific pipelines (e.g., Vulkan with cylindrical vs Cartesian acoustic
     * shaders) use this to select the correct shader variant at dispatch time.
     */
    virtual void set_coordinate_system(CoordinateSystem /*cs*/) {}

    /**
     * @brief Fills GPU-specific fields in HardwareInfo after initialization.
     *
     * CPU backend leaves fields at defaults. Vulkan backend populates
     * device name, queue count, etc. from the selected physical device.
     */
    virtual void populate_hardware_info(HardwareInfo& /*info*/) const {}

    /**
     * @brief Returns true when this backend can execute TVD vertical flux on device.
     */
    virtual bool supports_vertical_flux_dispatch() const { return false; }

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
     * @return True if dispatch succeeded; false means caller should fall back to CPU.
     */
    virtual bool dispatch_vertical_flux(
        const float* q_data, const float* w_data, float* dqdt_data,
        int nr, int nth, int nz,
        const double* dz_data, int dz_count,
        int limiter_id, bool positivity, double positivity_dt,
        double cfl_target,
        VerticalFluxDispatchResult& out_result)
    {
        (void)q_data; (void)w_data; (void)dqdt_data;
        (void)nr; (void)nth; (void)nz;
        (void)dz_data; (void)dz_count;
        (void)limiter_id; (void)positivity; (void)positivity_dt;
        (void)cfl_target; (void)out_result;
        return false;
    }

    // ── Radial advection dispatch ────────────────────────────────────

    virtual bool supports_radial_advection_dispatch() const { return false; }

    /**
     * @brief Execute first-order upwind radial advection on device.
     *
     * @param src       Source scalar field, row-major [nr][nth][nz], float.
     * @param u_data    Radial velocity field, same layout, float.
     * @param dst       Output field, same layout, float.
     * @param nr        Radial dimension.
     * @param nth       Azimuthal dimension.
     * @param nz        Vertical dimension.
     * @param dr        Radial grid spacing.
     * @param dt        Timestep.
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_radial_advection(
        const float* src, const float* u_data, float* dst,
        int nr, int nth, int nz,
        float dr, float dt)
    {
        (void)src; (void)u_data; (void)dst;
        (void)nr; (void)nth; (void)nz;
        (void)dr; (void)dt;
        return false;
    }

    // ── Radial advection on cylindrical Arakawa C-grid (Phase C.9) ────

    virtual bool supports_radial_advection_cgrid_dispatch() const { return false; }

    /**
     * @brief Execute TVD-MUSCL flux-form radial advection on the
     *        cylindrical Arakawa C-grid on device.
     *
     * Equivalent to advect_scalar_1d_r_kernel_cylindrical_cgrid in
     * src/numerics/advection/advection_cylindrical_cgrid.cpp. The face
     * velocity u_data is read from the cell-indexed buffer with the
     * understanding that u[i][j][k] is the right-face value at
     * r_face[i] = (i + 0.5) * dr.
     *
     * Boundary cells (i=0, i=NR-1, k=0, k=NZ-1) must be seeded from
     * src by the host before dispatch; the device only writes the
     * interior cells.
     *
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_radial_advection_cgrid(
        const float* src, const float* u_data, float* dst,
        int nr, int nth, int nz,
        float dr, float dt)
    {
        (void)src; (void)u_data; (void)dst;
        (void)nr; (void)nth; (void)nz;
        (void)dr; (void)dt;
        return false;
    }

    // ── Azimuthal advection dispatch ─────────────────────────────────

    virtual bool supports_azimuthal_advection_dispatch() const { return false; }

    /**
     * @brief Execute first-order upwind azimuthal advection on device.
     *
     * @param src       Source scalar field, row-major [nr][nth][nz], float.
     * @param v_data    Azimuthal velocity field, same layout, float.
     * @param dst       Output field, same layout, float.
     * @param nr        Radial dimension.
     * @param nth       Azimuthal dimension.
     * @param nz        Vertical dimension.
     * @param dr        Radial grid spacing (for 1/r geometry).
     * @param dtheta    Azimuthal grid spacing.
     * @param dt        Timestep.
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_azimuthal_advection(
        const float* src, const float* v_data, float* dst,
        int nr, int nth, int nz,
        float dr, float dtheta, float dt)
    {
        (void)src; (void)v_data; (void)dst;
        (void)nr; (void)nth; (void)nz;
        (void)dr; (void)dtheta; (void)dt;
        return false;
    }

    // ── Azimuthal advection on cylindrical Arakawa C-grid (Phase C.9) ──

    virtual bool supports_azimuthal_advection_cgrid_dispatch() const { return false; }

    /**
     * @brief Execute TVD-MUSCL flux-form azimuthal advection on the
     *        cylindrical Arakawa C-grid on device.
     *
     * Equivalent to advect_scalar_1d_theta_kernel_cylindrical_cgrid in
     * src/numerics/advection/advection_cylindrical_cgrid.cpp. The face
     * velocity v_data is read with the convention that v[i][j][k] is
     * the value at theta-face (between cells j and j+1). Theta is
     * periodic so every j is an interior point.
     *
     * Boundary cells in r and z (i=0, i=NR-1, k=0, k=NZ-1) must be
     * seeded from src by the host before dispatch; the device only
     * writes the interior cells.
     *
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_azimuthal_advection_cgrid(
        const float* src, const float* v_data, float* dst,
        int nr, int nth, int nz,
        float dr, float dtheta, float dt)
    {
        (void)src; (void)v_data; (void)dst;
        (void)nr; (void)nth; (void)nz;
        (void)dr; (void)dtheta; (void)dt;
        return false;
    }

    // ── Vertical advection on cylindrical Arakawa C-grid (Phase C.9) ──

    virtual bool supports_vertical_advection_cgrid_dispatch() const { return false; }

    /**
     * @brief Execute TVD-MUSCL flux-form vertical advection on the
     *        cylindrical Arakawa C-grid on device.
     *
     * Equivalent to advect_scalar_1d_z_kernel_cylindrical_cgrid in
     * src/numerics/advection/advection_cylindrical_cgrid.cpp. The face
     * velocity w_data is read with the convention that w[i][j][k] is
     * the value at z-face (between cells k and k+1).
     *
     * Boundary cells (i=0, i=NR-1, k=0, k=NZ-1) must be seeded from
     * src by the host before dispatch; the device only writes the
     * interior cells.
     *
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_vertical_advection_cgrid(
        const float* src, const float* w_data, float* dst,
        int nr, int nth, int nz,
        float dz, float dt)
    {
        (void)src; (void)w_data; (void)dst;
        (void)nr; (void)nth; (void)nz;
        (void)dz; (void)dt;
        return false;
    }

    // ── Acoustic pressure substep on cylindrical Arakawa C-grid (Phase C.9) ──

    virtual bool supports_acoustic_pressure_cgrid_dispatch() const { return false; }

    /**
     * @brief Execute fused acoustic pressure substep on the cylindrical
     *        Arakawa C-grid: flux-form divergence -> rho/p Forward-Euler
     *        integration with NaN sanitization and floor clamping.
     *
     * Equivalent to the chain
     *   SupercellCGridScheme::compute_fast_pressure_tendencies + the
     *   apply_fast_pressure integration loop in dynamics.cpp.
     *
     * Boundary cells (i=0, i=NR-1, k=0, k=NZ-1) are passed through with
     * zero tendency, matching the CPU C-grid semantics; this differs
     * from the collocated dispatch which extrapolates boundaries
     * hydrostatically.
     *
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_acoustic_pressure_cgrid(
        const float* u_data, const float* v_data, const float* w_data,
        const float* rho_in, const float* p_in,
        float* rho_out, float* p_out,
        int nr, int nth, int nz,
        float dr, float dtheta, float dz,
        float gamma_val, float dt_small,
        float rho_floor, float p_floor)
    {
        (void)u_data; (void)v_data; (void)w_data;
        (void)rho_in; (void)p_in; (void)rho_out; (void)p_out;
        (void)nr; (void)nth; (void)nz;
        (void)dr; (void)dtheta; (void)dz;
        (void)gamma_val; (void)dt_small;
        (void)rho_floor; (void)p_floor;
        return false;
    }

    // ── Acoustic momentum substep on cylindrical Arakawa C-grid (Phase C.9) ──

    virtual bool supports_acoustic_momentum_cgrid_dispatch() const { return false; }

    /**
     * @brief Execute fused acoustic momentum substep on the cylindrical
     *        Arakawa C-grid: per-face pressure gradient (with vertical
     *        reference-state subtraction) -> u/v/w Forward-Euler
     *        integration with NaN sanitization and magnitude clamps.
     *
     * Equivalent to the chain
     *   SupercellCGridScheme::compute_fast_momentum_tendencies + the
     *   apply_fast_momentum integration loop in dynamics.cpp.
     *
     * The vertical pressure gradient is computed in perturbation form
     * (dp/dz - dp0/dz) using the global hydrostatic reference profile
     * p0_base, so a hydrostatic state generates zero dw/dt up to float
     * roundoff. The collocated dispatch_acoustic_momentum has no
     * reference-state subtraction and would generate spurious g-scale
     * vertical accelerations on a hydrostatic state, which is why a
     * dedicated C-grid shader is required.
     *
     * Per-face valid index ranges (boundary cells get tendency = 0,
     * matching the CPU pre-zero + selective-write pattern):
     *   du: i in [0, NR-2],  j periodic,  k in [1, NZ-2]
     *   dv: i in [1, NR-2],  j periodic,  k in [1, NZ-2]
     *   dw: i in [1, NR-2],  j periodic,  k in [0, NZ-2]
     *
     * @param p0_base_data Reference pressure profile, 1D, length p0_base_len.
     * @param p0_base_len  Number of entries in p0_base_data; must equal nz.
     *
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_acoustic_momentum_cgrid(
        const float* rho_data, const float* p_data,
        const float* p0_base_data, int p0_base_len,
        const float* u_in, const float* v_in, const float* w_in,
        float* u_out, float* v_out, float* w_out,
        int nr, int nth, int nz,
        float dr, float dtheta, float dz,
        float dt_small,
        float wind_clamp_h, float wind_clamp_v)
    {
        (void)rho_data; (void)p_data;
        (void)p0_base_data; (void)p0_base_len;
        (void)u_in; (void)v_in; (void)w_in;
        (void)u_out; (void)v_out; (void)w_out;
        (void)nr; (void)nth; (void)nz;
        (void)dr; (void)dtheta; (void)dz;
        (void)dt_small;
        (void)wind_clamp_h; (void)wind_clamp_v;
        return false;
    }

    // ── Fused acoustic substep on cylindrical Arakawa C-grid (Phase C.9) ──

    virtual bool supports_acoustic_substep_fused_cgrid_dispatch() const { return false; }

    /**
     * @brief Execute the fused C-grid acoustic substep on device:
     *        pressure-substep + compute-to-compute barrier +
     *        momentum-substep, all 5 state fields in-place, in one
     *        command-buffer submission.
     *
     * Equivalent to calling dispatch_acoustic_pressure_cgrid followed
     * by dispatch_acoustic_momentum_cgrid back-to-back, but with a
     * single submit and a GPU-side barrier between the two compute
     * passes (no host round-trip). This halves the GPU dispatch
     * overhead per acoustic substep, which matters for
     * Klemp-Wilhelmson which does ~16 substeps per dynamics step.
     *
     * Reference-state subtraction in the vertical momentum equation
     * uses the global hydrostatic profile p0_base (broadcast 1D->3D
     * inside the dispatch); same constraints as
     * dispatch_acoustic_momentum_cgrid.
     *
     * @param p0_base_data Reference pressure profile, 1D, length p0_base_len.
     * @param p0_base_len  Number of entries; must equal nz.
     *
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_acoustic_substep_fused_cgrid(
        float* u, float* v, float* w,
        float* rho, float* p,
        const float* p0_base_data, int p0_base_len,
        int nr, int nth, int nz,
        float dr, float dtheta, float dz,
        float gamma_val, float dt_small,
        float rho_floor, float p_floor,
        float wind_clamp_h, float wind_clamp_v)
    {
        (void)u; (void)v; (void)w; (void)rho; (void)p;
        (void)p0_base_data; (void)p0_base_len;
        (void)nr; (void)nth; (void)nz;
        (void)dr; (void)dtheta; (void)dz;
        (void)gamma_val; (void)dt_small;
        (void)rho_floor; (void)p_floor;
        (void)wind_clamp_h; (void)wind_clamp_v;
        return false;
    }

    // ── Diffusion dispatch ───────────────────────────────────────────

    virtual bool supports_diffusion_dispatch() const { return false; }

    /**
     * @brief Execute cylindrical Laplacian diffusion on device.
     *
     * @param src       Source scalar field, row-major [nr][nth][nz], float.
     * @param dst       Output field, same layout, float.
     * @param nr        Radial dimension.
     * @param nth       Azimuthal dimension.
     * @param nz        Vertical dimension.
     * @param dr        Radial grid spacing.
     * @param dtheta    Azimuthal grid spacing.
     * @param dz        Vertical grid spacing.
     * @param dt        Timestep.
     * @param kappa     Diffusion coefficient.
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_diffusion(
        const float* src, float* dst,
        int nr, int nth, int nz,
        float dr, float dtheta, float dz,
        float dt, float kappa)
    {
        (void)src; (void)dst;
        (void)nr; (void)nth; (void)nz;
        (void)dr; (void)dtheta; (void)dz;
        (void)dt; (void)kappa;
        return false;
    }

    // ── Supercell tendencies dispatch ───────────────────────────────

    virtual bool supports_supercell_tendencies_dispatch() const { return false; }

    /**
     * @brief Execute supercell momentum tendencies on device.
     *
     * Computes 5 tendency fields from 6 input state fields using centered
     * finite differences in cylindrical coordinates. Only valid when terrain
     * metrics are NOT active (uniform grid spacing assumed).
     *
     * @param u_r_data      Radial velocity, row-major [nr][nth][nz], float.
     * @param u_theta_data  Azimuthal velocity, same layout, float.
     * @param u_z_data      Vertical velocity, same layout, float.
     * @param rho_data      Density, same layout, float.
     * @param p_data        Pressure, same layout, float.
     * @param theta_data    Potential temperature, same layout, float.
     * @param du_r_dt_data  Output: radial velocity tendency, float.
     * @param du_theta_dt_data Output: azimuthal velocity tendency, float.
     * @param du_z_dt_data  Output: vertical velocity tendency, float.
     * @param drho_dt_data  Output: density tendency, float.
     * @param dp_dt_data    Output: pressure tendency, float.
     * @param nr            Radial dimension.
     * @param nth           Azimuthal dimension.
     * @param nz            Vertical dimension.
     * @param dr            Radial grid spacing.
     * @param dtheta        Azimuthal grid spacing.
     * @param dz            Vertical grid spacing.
     * @param g             Gravitational acceleration.
     * @param gamma_val     Ratio of specific heats.
     * @param theta0        Reference potential temperature.
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_supercell_tendencies(
        const float* u_r_data, const float* u_theta_data, const float* u_z_data,
        const float* rho_data, const float* p_data, const float* theta_data,
        const float* loading_data,
        float* du_r_dt_data, float* du_theta_dt_data, float* du_z_dt_data,
        float* drho_dt_data, float* dp_dt_data,
        int nr, int nth, int nz,
        float dr, float dtheta, float dz,
        float g, float gamma_val, float theta0)
    {
        (void)u_r_data; (void)u_theta_data; (void)u_z_data;
        (void)rho_data; (void)p_data; (void)theta_data;
        (void)loading_data;
        (void)du_r_dt_data; (void)du_theta_dt_data; (void)du_z_dt_data;
        (void)drho_dt_data; (void)dp_dt_data;
        (void)nr; (void)nth; (void)nz;
        (void)dr; (void)dtheta; (void)dz;
        (void)g; (void)gamma_val; (void)theta0;
        return false;
    }

    // ── Cartesian tendencies dispatch ──────────────────────────────

    virtual bool supports_cartesian_tendencies_dispatch() const { return false; }

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
        const float* u_x_data, const float* u_y_data, const float* w_data,
        const float* rho_data, const float* p_data, const float* theta_data,
        const float* p0_base_data, const float* rho0_base_data,
        const float* loading_data,
        const float* u0_base_data, const float* v0_base_data,
        float* du_x_dt_data, float* du_y_dt_data, float* dw_dt_data,
        float* drho_dt_data, float* dp_dt_data,
        int nr, int nth, int nz,
        float dx, float dy, float dz,
        float g, float gamma_val, float coriolis_f_val)
    {
        (void)u_x_data; (void)u_y_data; (void)w_data;
        (void)rho_data; (void)p_data; (void)theta_data;
        (void)p0_base_data; (void)rho0_base_data;
        (void)loading_data;
        (void)u0_base_data; (void)v0_base_data;
        (void)du_x_dt_data; (void)du_y_dt_data; (void)dw_dt_data;
        (void)drho_dt_data; (void)dp_dt_data;
        (void)nr; (void)nth; (void)nz;
        (void)dx; (void)dy; (void)dz;
        (void)g; (void)gamma_val; (void)coriolis_f_val;
        return false;
    }

    // ── Cartesian x-advection dispatch ──────────────────────────────

    virtual bool supports_advection_x_dispatch() const { return false; }

    virtual bool dispatch_advection_x(
        const float* src, const float* u_data, float* dst,
        int nr, int nth, int nz,
        float dx, float dt)
    {
        (void)src; (void)u_data; (void)dst;
        (void)nr; (void)nth; (void)nz;
        (void)dx; (void)dt;
        return false;
    }

    // ── Cartesian y-advection dispatch ──────────────────────────────

    virtual bool supports_advection_y_dispatch() const { return false; }

    virtual bool dispatch_advection_y(
        const float* src, const float* v_data, float* dst,
        int nr, int nth, int nz,
        float dy, float dt)
    {
        (void)src; (void)v_data; (void)dst;
        (void)nr; (void)nth; (void)nz;
        (void)dy; (void)dt;
        return false;
    }

    // ── Tornado tendencies dispatch ─────────────────────────────────

    virtual bool supports_tornado_tendencies_dispatch() const { return false; }

    /**
     * @brief Execute tornado momentum tendencies on device.
     *
     * Similar to supercell but uses only radial and vertical derivatives
     * (no azimuthal). Includes vortex damping friction. Computes all (i,j,k)
     * interior points — the GPU version does not use azimuthal symmetry.
     *
     * @param u_r_data      Radial velocity, row-major [nr][nth][nz], float.
     * @param u_theta_data  Azimuthal velocity, same layout, float.
     * @param u_z_data      Vertical velocity, same layout, float.
     * @param rho_data      Density, same layout, float.
     * @param p_data        Pressure, same layout, float.
     * @param theta_data    Potential temperature, same layout, float.
     * @param du_r_dt_data  Output: radial velocity tendency, float.
     * @param du_theta_dt_data Output: azimuthal velocity tendency, float.
     * @param du_z_dt_data  Output: vertical velocity tendency, float.
     * @param drho_dt_data  Output: density tendency, float.
     * @param dp_dt_data    Output: pressure tendency, float.
     * @param nr            Radial dimension.
     * @param nth           Azimuthal dimension.
     * @param nz            Vertical dimension.
     * @param dr            Radial grid spacing.
     * @param dz            Vertical grid spacing.
     * @param g             Gravitational acceleration.
     * @param theta0        Reference potential temperature.
     * @param eps           Cylindrical singularity epsilon.
     * @param friction_coeff Vortex damping friction coefficient.
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_tornado_tendencies(
        const float* u_r_data, const float* u_theta_data, const float* u_z_data,
        const float* rho_data, const float* p_data, const float* theta_data,
        const float* loading_data,
        float* du_r_dt_data, float* du_theta_dt_data, float* du_z_dt_data,
        float* drho_dt_data, float* dp_dt_data,
        int nr, int nth, int nz,
        float dr, float dz,
        float g, float theta0, float eps, float friction_coeff)
    {
        (void)u_r_data; (void)u_theta_data; (void)u_z_data;
        (void)rho_data; (void)p_data; (void)theta_data;
        (void)loading_data;
        (void)du_r_dt_data; (void)du_theta_dt_data; (void)du_z_dt_data;
        (void)drho_dt_data; (void)dp_dt_data;
        (void)nr; (void)nth; (void)nz;
        (void)dr; (void)dz;
        (void)g; (void)theta0; (void)eps; (void)friction_coeff;
        return false;
    }

    // ── Kessler point-wise microphysics dispatch ────────────────────

    virtual bool supports_kessler_pointwise_dispatch() const { return false; }

    /**
     * @brief Execute fused Kessler warm rain + ice + melting on device.
     *
     * All three sub-processes are point-wise and operate on the same input
     * fields.  Fusing them avoids three separate GPU round-trips.
     *
     * @param temperature_data  Temperature field, row-major [nr][nth][nz], float.
     * @param p_data            Pressure field, same layout, float.
     * @param qv_data           Water vapor mixing ratio, same layout, float.
     * @param qc_data           Cloud water mixing ratio, same layout, float.
     * @param qr_data           Rain mixing ratio, same layout, float.
     * @param qg_data           Graupel mixing ratio, same layout, float.
     * @param qh_data           Hail mixing ratio, same layout, float.
     * @param dtheta_dt_data    Output: temperature tendency (pre-theta conversion).
     * @param dqv_dt_data       Output: water vapor tendency.
     * @param dqc_dt_data       Output: cloud water tendency.
     * @param dqr_dt_data       Output: rain tendency.
     * @param dqg_dt_data       Output: graupel tendency.
     * @param dqh_dt_data       Output: hail tendency.
     * @param nr                Radial dimension.
     * @param nth               Azimuthal dimension.
     * @param nz                Vertical dimension.
     * @param qc0               Autoconversion threshold.
     * @param c_auto            Autoconversion rate.
     * @param c_accr            Accretion rate.
     * @param c_evap            Evaporation rate.
     * @param c_freeze          Freezing rate.
     * @param c_rime            Riming rate.
     * @param c_melt            Melting rate.
     * @param c_subl            Sublimation rate.
     * @param Lv_cp             L_v / cp (precomputed).
     * @param Lf_cp             L_f / cp (precomputed).
     * @param Ls_cp             L_s / cp (precomputed).
     * @param T0                Freezing temperature (K).
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_kessler_pointwise(
        const float* temperature_data, const float* p_data,
        const float* qv_data,
        const float* qc_data, const float* qr_data,
        const float* qg_data, const float* qh_data,
        float* dtheta_dt_data, float* dqv_dt_data,
        float* dqc_dt_data, float* dqr_dt_data,
        float* dqg_dt_data, float* dqh_dt_data,
        int nr, int nth, int nz,
        float qc0, float c_auto, float c_accr, float c_evap,
        float c_freeze, float c_rime, float c_melt, float c_subl,
        float Lv_cp, float Lf_cp, float Ls_cp, float T0)
    {
        (void)temperature_data; (void)p_data; (void)qv_data;
        (void)qc_data; (void)qr_data;
        (void)qg_data; (void)qh_data;
        (void)dtheta_dt_data; (void)dqv_dt_data;
        (void)dqc_dt_data; (void)dqr_dt_data;
        (void)dqg_dt_data; (void)dqh_dt_data;
        (void)nr; (void)nth; (void)nz;
        (void)qc0; (void)c_auto; (void)c_accr; (void)c_evap;
        (void)c_freeze; (void)c_rime; (void)c_melt; (void)c_subl;
        (void)Lv_cp; (void)Lf_cp; (void)Ls_cp; (void)T0;
        return false;
    }

    // ── Batched advection dispatch ─────────────────────────────────

    virtual bool supports_batched_advection_dispatch() const { return false; }

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
        const float* scalar_in, float* result_out,
        const float* u_data, const float* v_data,
        int nr, int nth, int nz,
        float dr, float dtheta, float dt_half)
    {
        (void)scalar_in; (void)result_out;
        (void)u_data; (void)v_data;
        (void)nr; (void)nth; (void)nz;
        (void)dr; (void)dtheta; (void)dt_half;
        return false;
    }

    /**
     * @brief Execute post-vertical advection batch on device.
     *
     * Records azimuthal(dt_half) → radial(dt_half) → diffusion(dt_full)
     * in a single command buffer submission.
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
     * @param kappa         Diffusion coefficient.
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_advection_batch_post_vertical(
        const float* scalar_in, float* result_out,
        const float* u_data, const float* v_data,
        int nr, int nth, int nz,
        float dr, float dtheta, float dz,
        float dt_half, float dt_full, float kappa)
    {
        (void)scalar_in; (void)result_out;
        (void)u_data; (void)v_data;
        (void)nr; (void)nth; (void)nz;
        (void)dr; (void)dtheta; (void)dz;
        (void)dt_half; (void)dt_full; (void)kappa;
        return false;
    }

    // ── Kessler sedimentation dispatch ──────────────────────────────

    virtual bool supports_kessler_sedimentation_dispatch() const { return false; }

    /**
     * @brief Execute Kessler sedimentation on device (column-wise).
     *
     * Reads existing tendency values (from point-wise pass) and adds
     * sedimentation contributions in-place.
     *
     * @param qr_data           Rain mixing ratio (readonly), row-major [nr][nth][nz].
     * @param qg_data           Graupel mixing ratio (readonly).
     * @param qh_data           Hail mixing ratio (readonly).
     * @param dqr_dt_data       Rain tendency (read-write, accumulated).
     * @param dqg_dt_data       Graupel tendency (read-write, accumulated).
     * @param dqh_dt_data       Hail tendency (read-write, accumulated).
     * @param nr                Radial dimension.
     * @param nth               Azimuthal dimension.
     * @param nz                Vertical dimension.
     * @param dz_val            Vertical grid spacing.
     * @param a_rain            Rain terminal velocity coefficient a.
     * @param b_rain            Rain terminal velocity exponent b.
     * @param Vt_max_rain       Rain terminal velocity cap.
     * @param a_grau            Graupel terminal velocity coefficient a.
     * @param b_grau            Graupel terminal velocity exponent b.
     * @param Vt_max_grau       Graupel terminal velocity cap.
     * @param a_hail            Hail terminal velocity coefficient a.
     * @param b_hail            Hail terminal velocity exponent b.
     * @param Vt_max_hail       Hail terminal velocity cap.
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_kessler_sedimentation(
        const float* qr_data, const float* qg_data, const float* qh_data,
        float* dqr_dt_data, float* dqg_dt_data, float* dqh_dt_data,
        int nr, int nth, int nz,
        float dz_val,
        float a_rain, float b_rain, float Vt_max_rain,
        float a_grau, float b_grau, float Vt_max_grau,
        float a_hail, float b_hail, float Vt_max_hail)
    {
        (void)qr_data; (void)qg_data; (void)qh_data;
        (void)dqr_dt_data; (void)dqg_dt_data; (void)dqh_dt_data;
        (void)nr; (void)nth; (void)nz;
        (void)dz_val;
        (void)a_rain; (void)b_rain; (void)Vt_max_rain;
        (void)a_grau; (void)b_grau; (void)Vt_max_grau;
        (void)a_hail; (void)b_hail; (void)Vt_max_hail;
        return false;
    }

    // ── Thompson microphysics dispatch ─────────────────────────────────

    virtual bool supports_thompson_pointwise_dispatch() const { return false; }

    /**
     * @brief Execute fused Thompson point-wise microphysics on device.
     *
     * Includes saturation adjustment, warm rain, ice processes (freezing,
     * deposition, riming, ice-to-snow, sublimation), and melting.
     *
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_thompson_pointwise(
        const float* temperature_data, const float* p_data,
        const float* qv_data, const float* qc_data, const float* qr_data,
        const float* qi_data, const float* qs_data,
        const float* qg_data, const float* qh_data,
        float* dtheta_dt_data, float* dqv_dt_data,
        float* dqc_dt_data, float* dqr_dt_data,
        float* dqi_dt_data, float* dqs_dt_data,
        float* dqg_dt_data, float* dqh_dt_data,
        int nr, int nth, int nz,
        float qc0, float c_auto, float c_evap,
        float c_dep, float c_subl, float c_melt,
        float Lv_cp, float Lf_cp, float Ls_cp, float T0,
        float ccn_conc, float in_conc)
    {
        (void)temperature_data; (void)p_data;
        (void)qv_data; (void)qc_data; (void)qr_data;
        (void)qi_data; (void)qs_data;
        (void)qg_data; (void)qh_data;
        (void)dtheta_dt_data; (void)dqv_dt_data;
        (void)dqc_dt_data; (void)dqr_dt_data;
        (void)dqi_dt_data; (void)dqs_dt_data;
        (void)dqg_dt_data; (void)dqh_dt_data;
        (void)nr; (void)nth; (void)nz;
        (void)qc0; (void)c_auto; (void)c_evap;
        (void)c_dep; (void)c_subl; (void)c_melt;
        (void)Lv_cp; (void)Lf_cp; (void)Ls_cp; (void)T0;
        (void)ccn_conc; (void)in_conc;
        return false;
    }

    virtual bool supports_thompson_sedimentation_dispatch() const { return false; }

    /**
     * @brief Execute Thompson sedimentation on device (4 species: qr, qs, qg, qh).
     *
     * Adds sedimentation contributions to existing tendency values in-place.
     *
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_thompson_sedimentation(
        const float* qr_data, const float* qs_data,
        const float* qg_data, const float* qh_data,
        float* dqr_dt_data, float* dqs_dt_data,
        float* dqg_dt_data, float* dqh_dt_data,
        int nr, int nth, int nz,
        float dz_val,
        float a_rain, float b_rain, float Vt_max_rain,
        float a_snow, float b_snow, float Vt_max_snow,
        float a_grau, float b_grau, float Vt_max_grau,
        float a_hail, float b_hail, float Vt_max_hail)
    {
        (void)qr_data; (void)qs_data; (void)qg_data; (void)qh_data;
        (void)dqr_dt_data; (void)dqs_dt_data;
        (void)dqg_dt_data; (void)dqh_dt_data;
        (void)nr; (void)nth; (void)nz; (void)dz_val;
        (void)a_rain; (void)b_rain; (void)Vt_max_rain;
        (void)a_snow; (void)b_snow; (void)Vt_max_snow;
        (void)a_grau; (void)b_grau; (void)Vt_max_grau;
        (void)a_hail; (void)b_hail; (void)Vt_max_hail;
        return false;
    }

    // ── Acoustic substep dispatch ─────────────────────────────────────

    virtual bool supports_acoustic_pressure_dispatch() const { return false; }
    virtual bool supports_acoustic_momentum_dispatch() const { return false; }

    /**
     * @brief Fused acoustic pressure substep: divergence -> rho/p integration.
     *
     * Reads u,v,w,rho,p; computes cylindrical divergence; writes updated rho,p.
     * Boundary points get zero-gradient extrapolation.
     *
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_acoustic_pressure(
        const float* u_data, const float* v_data, const float* w_data,
        const float* rho_in, const float* p_in,
        float* rho_out, float* p_out,
        int nr, int nth, int nz,
        float dr_val, float dtheta_val, float dz_val,
        float gamma_val, float dt_small,
        float rho_floor, float p_floor)
    {
        (void)u_data; (void)v_data; (void)w_data;
        (void)rho_in; (void)p_in; (void)rho_out; (void)p_out;
        (void)nr; (void)nth; (void)nz;
        (void)dr_val; (void)dtheta_val; (void)dz_val;
        (void)gamma_val; (void)dt_small;
        (void)rho_floor; (void)p_floor;
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
        const float* rho_data, const float* p_data,
        const float* u_in, const float* v_in, const float* w_in,
        float* u_out, float* v_out, float* w_out,
        int nr, int nth, int nz,
        float dr_val, float dtheta_val, float dz_val,
        float dt_small,
        float wind_clamp_h, float wind_clamp_v)
    {
        (void)rho_data; (void)p_data;
        (void)u_in; (void)v_in; (void)w_in;
        (void)u_out; (void)v_out; (void)w_out;
        (void)nr; (void)nth; (void)nz;
        (void)dr_val; (void)dtheta_val; (void)dz_val;
        (void)dt_small;
        (void)wind_clamp_h; (void)wind_clamp_v;
        return false;
    }

    // ── Fused acoustic substep dispatch ──────────────────────────────

    virtual bool supports_acoustic_substep_fused_dispatch() const { return false; }

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
        float* u, float* v, float* w,
        float* rho, float* p,
        int nr, int nth, int nz,
        float dr, float dtheta, float dz,
        float gamma_val, float dt_small,
        float rho_floor, float p_floor,
        float wind_clamp_h, float wind_clamp_v)
    {
        (void)u; (void)v; (void)w; (void)rho; (void)p;
        (void)nr; (void)nth; (void)nz;
        (void)dr; (void)dtheta; (void)dz;
        (void)gamma_val; (void)dt_small;
        (void)rho_floor; (void)p_floor;
        (void)wind_clamp_h; (void)wind_clamp_v;
        return false;
    }

    // ── Batched acoustic substeps (all N substeps in one submission) ──

    virtual bool supports_acoustic_substeps_batched_dispatch() const { return false; }

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
        float* u, float* v, float* w,
        float* rho, float* p,
        int nr, int nth, int nz,
        float dr, float dtheta, float dz,
        float gamma_val, float dt_small, int n_substeps,
        float rho_floor, float p_floor,
        float wind_clamp_h, float wind_clamp_v)
    {
        (void)u; (void)v; (void)w; (void)rho; (void)p;
        (void)nr; (void)nth; (void)nz;
        (void)dr; (void)dtheta; (void)dz;
        (void)gamma_val; (void)dt_small; (void)n_substeps;
        (void)rho_floor; (void)p_floor;
        (void)wind_clamp_h; (void)wind_clamp_v;
        return false;
    }
};

extern ComputeBackendConfig global_compute_backend_config;

/**
 * @brief Converts backend kind enum to canonical id.
 */
const char* compute_backend_kind_name(ComputeBackendKind kind);

/**
 * @brief Parses backend id text into enum.
 * @param value Text identifier.
 * @param out_kind Parsed enum output.
 * @return True when parse succeeds.
 */
bool parse_compute_backend_kind(const std::string& value, ComputeBackendKind& out_kind);

/**
 * @brief Returns all currently supported backend ids.
 */
std::vector<std::string> get_available_compute_backends();

/**
 * @brief Initializes configured runtime compute backend with fallback behavior.
 * @param error Diagnostic on failure.
 * @return True when an active backend is ready.
 */
bool initialize_compute_backend_runtime(std::string& error);

/**
 * @brief Shuts down active runtime compute backend.
 */
void shutdown_compute_backend_runtime();

/**
 * @brief Returns active runtime compute backend, if initialized (const).
 */
const ComputeBackend* active_compute_backend();

/**
 * @brief Returns mutable active runtime compute backend for dispatch calls.
 */
ComputeBackend* mutable_compute_backend();

/**
 * @brief Returns active runtime backend kind currently executing kernels.
 */
ComputeBackendKind active_compute_backend_kind();

/**
 * @brief Returns configured backend id requested by numerics.compute.backend.
 */
const char* requested_compute_backend_name();

/**
 * @brief Returns true when runtime fell back from requested backend to CPU.
 */
bool compute_backend_fallback_active();
