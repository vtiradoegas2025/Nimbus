#pragma once

#include "compute/dispatch/common.hpp"
#include "compute/dispatch/collocated_sgs.hpp"
#include <array>

/**
 * @file turbulence.hpp
 * @brief Turbulence compute dispatch data and capability interface.
 */

enum class TurbulenceKernelOp
{
    Smagorinsky,
    SgsApply,
    Tke,
    CollocatedSgs
};

struct TkeClosureDispatch
{
    const float* u = nullptr;
    const float* v = nullptr;
    const float* w = nullptr;
    const float* theta = nullptr;
    const float* qv = nullptr;
    const float* tke = nullptr;
    const float* dz = nullptr;
    float* nu = nullptr;
    float* K_theta = nullptr;
    float* K_q = nullptr;
    float* K_tke = nullptr;
    float* shear = nullptr;
    float* buoyancy = nullptr;
    float* dissipation = nullptr;
    float* effective_prandtl = nullptr;
    int nx = 0, ny = 0, nz = 0, profile_length = 0;
    int filter_mode = 0;
    bool use_horizontal = true, use_vertical = true;
    bool periodic_x = false, periodic_y = false;
    bool stability = true, moist_buoyancy = false;
    float dx = 0, dy = 0, Pr_t = 0, Sc_t = 0, nu_max = 0, K_max = 0;
};

struct TkeTendencyDispatch
{
    const float* rho = nullptr;
    const float* tke = nullptr;
    const float* K_tke = nullptr;
    const float* shear = nullptr;
    const float* buoyancy = nullptr;
    const float* dissipation = nullptr;
    const float* dz = nullptr;
    float* tendency = nullptr;
    int nx = 0, ny = 0, nz = 0, profile_length = 0;
    bool use_horizontal = true, use_vertical = true;
    bool periodic_x = false, periodic_y = false;
    float dx = 0, dy = 0;
    std::array<bool, 4> open_lateral{{true, true, true, true}};
};

struct SgsApplyDispatch
{
    using CompletedObserver = SixFieldCompletedObserver;

    float* u = nullptr;
    float* v = nullptr;
    float* w = nullptr;
    float* theta_data = nullptr;
    float* qv_data = nullptr;
    float* tke_data = nullptr;
    const float* dudt = nullptr;
    const float* dvdt = nullptr;
    const float* dwdt = nullptr;
    const float* dthetadt = nullptr;
    const float* dqvdt = nullptr;
    const float* dtkedt = nullptr;
    const float* dtkedt_pbl = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    bool has_qv = false;
    bool has_tke = false;
    float dt_step = 0.0f;
    float theta_fallback = 0.0f;
    float theta_min = 0.0f;
    float theta_max = 0.0f;
    float qv_max_val = 0.0f;
    float wind_h_max = 0.0f;
    float wind_v_max = 0.0f;
    bool resident_tendencies_unchanged = false;
    bool sanitize = true;
    // Post-fence/pre-copy: host fields remain at the old time level while
    // the observer sees the six actual device results. Do not retain them.
    CompletedObserver completed_observer = nullptr;
    void* completed_observer_context = nullptr;
    bool collocated = false;
};

struct SmagorinskyCoefficientsDispatch
{
    const float* u = nullptr;
    const float* v = nullptr;
    const float* w = nullptr;
    const float* dz_prof = nullptr;
    const float* dzc_prof = nullptr;
    int prof_len = 0;
    float* nu_t = nullptr;
    float* K_theta = nullptr;
    float* K_q = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int filter_mode = 0;
    bool use_horizontal = true;
    bool use_vertical = true;
    bool periodic_x = false;
    bool periodic_y = false;
    float dx = 0.0f;
    float dy = 0.0f;
    float Cs = 0.0f;
    float Pr_t = 0.0f;
    float Sc_t = 0.0f;
    float nu_t_max = 0.0f;
    float K_max = 0.0f;
};

struct SmagorinskyTendenciesDispatch
{
    const float* u = nullptr;
    const float* v = nullptr;
    const float* w = nullptr;
    const float* theta_data = nullptr;
    const float* qv_data = nullptr;
    const float* nu_t = nullptr;
    const float* K_theta = nullptr;
    const float* K_q = nullptr;
    const float* dz_prof = nullptr;
    const float* rho = nullptr;
    int prof_len = 0;
    float* dudt_sgs = nullptr;
    float* dvdt_sgs = nullptr;
    float* dwdt_sgs = nullptr;
    float* dthetadt_sgs = nullptr;
    float* dqvdt_sgs = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    bool use_horizontal = false;
    bool use_vertical = false;
    bool has_qv = false;
    bool periodic_x = false;
    bool periodic_y = false;
    float dx = 0.0f;
    float dy = 0.0f;
    std::array<bool, 4> open_lateral{{true, true, true, true}};
};

struct SmagorinskyFusedDispatch
{
    const float* u_data = nullptr;
    const float* v_data = nullptr;
    const float* w_data = nullptr;
    const float* theta_data = nullptr;
    const float* qv_data = nullptr;
    const float* dz_prof = nullptr;
    const float* dzc_prof = nullptr;
    const float* rho = nullptr;
    int prof_len = 0;
    float* nu_t = nullptr;
    float* K_theta = nullptr;
    float* K_q = nullptr;
    float* dudt_sgs = nullptr;
    float* dvdt_sgs = nullptr;
    float* dwdt_sgs = nullptr;
    float* dthetadt_sgs = nullptr;
    float* dqvdt_sgs = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int filter_mode = 0;
    bool use_horizontal = false;
    bool use_vertical = false;
    bool has_qv = false;
    bool periodic_x = false;
    bool periodic_y = false;
    float dx_val = 0.0f;
    float dy_val = 0.0f;
    float Cs = 0.0f;
    float Pr_t = 0.0f;
    float Sc_t = 0.0f;
    float nu_t_max = 0.0f;
    float K_max = 0.0f;
    // Tendencies are always published; intermediate coefficients are needed
    // on the host only by diagnostics and sampled CPU parity.
    bool download_coefficients = true;
    std::array<bool, 4> open_lateral{{true, true, true, true}};
};

class TurbulenceComputeKernels
{
public:
    virtual ~TurbulenceComputeKernels() = default;

    /**
     * @brief Returns true when this backend can execute the requested turbulence operation.
     */
    virtual bool supports(TurbulenceKernelOp /*op*/) const { return false; }

    virtual bool dispatch_collocated_sgs(const CollocatedSgsDispatch&)
    {
        return false;
    }
    virtual bool dispatch_tke_closure(const TkeClosureDispatch&)
    {
        return false;
    }
    virtual bool dispatch_tke_tendency(const TkeTendencyDispatch&) { return false; }

    // ── Smagorinsky turbulence dispatch (GPU audit Phase 3.4) ───

    /**
     * @brief Compute the Smagorinsky eddy coefficients on device: per-cell
     *        strain magnitude -> nu_t (clamped) and the K_theta / K_q
     *        diffusivities. Flat-terrain, stability "none" only (the
     *        caller guards both).
     *
     * @param dz_prof / dzc_prof One-sided and centered vertical spacing
     *        profiles, 1D length >= nz, from the same grid_metric calls
     *        the CPU uses.
     * @param filter_mode 0 = cubic root of the cell volume, 1 = sqrt(dx*dy).
     */
    virtual bool dispatch_smagorinsky_coefficients(
        const SmagorinskyCoefficientsDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    /**
     * @brief Compute the conservative variable-nu SGS diffusion tendencies
     *        on device for u, v, w, theta and qv in one pass, reading the
     *        coefficient fields the coefficients dispatch produced
     *        (live-density flux form (1/rho) d_j(2 rho nu S_ij)).
     *
     * @param rho Live dry-density field, shape nx*ny*nz. It is required:
     *        conserving rho0-weighted momentum is not a valid substitute.
     */
    virtual bool dispatch_smagorinsky_tendencies(
        const SmagorinskyTendenciesDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
    // ── Fused Smagorinsky chain (GPU audit Phase 4.3c) ──────────

    /**
     * @brief Eddy coefficients + SGS tendencies in one submission; the
     *        coefficient fields stay device-resident between the passes
     *        and download once (diagnostics/parity). Same support gate as
     *        the split dispatches (both pipelines ready).
     * @return True if dispatch succeeded.
     * @param rho Live dry-density field (nx*ny*nz) for conservative
     *        prognostic momentum and scalar-mass weighting.
     */
    virtual bool dispatch_smagorinsky_fused(
        const SmagorinskyFusedDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    // ── SGS tendency application (GPU audit Phase 4.2) ──────────

    /**
     * @brief Apply the SGS turbulence tendencies to u/v/w/theta/qv/tke.
     *        sanitize=true uses the integrate loop's guard/clamp semantics;
     *        false publishes raw Warn/Strict candidates to the completed
     *        observer before any host mutation. has_qv/has_tke gate optional
     *        fields (their buffers must still be valid pointers).
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_sgs_apply(const SgsApplyDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
};

inline int turbulence_closed_lateral_mask(const std::array<bool, 4>& open_lateral) noexcept
{
    int mask = 0;
    for (int face = 0; face < 4; ++face)
    {
        if (!open_lateral[face])
            mask |= 1 << face;
    }
    return mask;
}
