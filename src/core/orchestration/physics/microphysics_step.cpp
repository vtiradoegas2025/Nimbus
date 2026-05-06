/**
 * @file microphysics_step.cpp
 * @brief Microphysics tendency driver.
 *
 * Computes microphysics tendencies via the active scheme and applies them
 * with Forward Euler integration, NaN guards, and physical-bounds clamping.
 * Also applies radiation and PBL tendencies to theta/qv.
 *
 * Extracted from src/core/equations.cpp.
 */

#include "core/runtime/simulation.hpp"
#include "util/simd_utils.hpp"
#include "util/log.hpp"
#include "microphysics/factory.hpp"

#include <algorithm>
#include <cmath>
#include <exception>

#ifdef _OPENMP
#include <omp.h>
#endif

extern std::unique_ptr<MicrophysicsScheme> microphysics_scheme;

void step_microphysics(double dt_micro)
{
    if (!std::isfinite(dt_micro) || dt_micro <= 0.0)
    {
        tmv::log_warn("[MICROPHYSICS GUARD] invalid microphysics timestep: ", dt_micro);
        return;
    }

    if (!microphysics_scheme)
    {
        tmv::log_warn("Microphysics scheme not initialized, using default Kessler");
        initialize_microphysics("kessler");
    }

    static Field3D dtheta_dt;
    static Field3D dqv_dt;
    static Field3D dqc_dt;
    static Field3D dqr_dt;
    static Field3D dqi_dt;
    static Field3D dqs_dt;
    static Field3D dqg_dt;
    static Field3D dqh_dt;

    auto ensure_shape = [](Field3D& f)
    {
        if (f.size_r() != NR || f.size_th() != NTH || f.size_z() != NZ)
        {
            f.resize(NR, NTH, NZ, 0.0f);
        }
    };
    ensure_shape(dtheta_dt);
    ensure_shape(dqv_dt);
    ensure_shape(dqc_dt);
    ensure_shape(dqr_dt);
    ensure_shape(dqi_dt);
    ensure_shape(dqs_dt);
    ensure_shape(dqg_dt);
    ensure_shape(dqh_dt);

    try
    {
        microphysics_scheme->compute_tendencies(p, theta, qv, qc, qr, qi, qs, qg, qh,
            dt_micro, dtheta_dt, dqv_dt, dqc_dt, dqr_dt, dqi_dt, dqs_dt, dqg_dt, dqh_dt);
    }
    catch (const std::exception& e)
    {
        tmv::log_warn("[MICROPHYSICS GUARD] tendency computation failed: ", e.what(),
                      ". Continuing with zero microphysics tendencies for this step.");
        dtheta_dt.fill(0.0f);
        dqv_dt.fill(0.0f);
        dqc_dt.fill(0.0f);
        dqr_dt.fill(0.0f);
        dqi_dt.fill(0.0f);
        dqs_dt.fill(0.0f);
        dqg_dt.fill(0.0f);
        dqh_dt.fill(0.0f);
    }

    apply_chaos_to_microphysics_tendencies(dtheta_dt, dqv_dt, dqc_dt, dqr_dt, dqi_dt, dqs_dt, dqg_dt, dqh_dt);

    auto sanitize_nonfinite_tendency = [](Field3D& field) -> int
    {
        if (field.empty()) { return 0; }
        float* const data = field.data();
        const int count = static_cast<int>(field.size());
        return simd_utils::sanitize_nonfinite(data, 0.0f, data, count);
    };

    const int non_finite_tendency_sanitized =
        sanitize_nonfinite_tendency(dtheta_dt) +
        sanitize_nonfinite_tendency(dqv_dt) +
        sanitize_nonfinite_tendency(dqc_dt) +
        sanitize_nonfinite_tendency(dqr_dt) +
        sanitize_nonfinite_tendency(dqi_dt) +
        sanitize_nonfinite_tendency(dqs_dt) +
        sanitize_nonfinite_tendency(dqg_dt) +
        sanitize_nonfinite_tendency(dqh_dt);

    constexpr float max_theta_step_change_k = 50.0f;
    int non_finite_theta_tendency_count = 0;
    int theta_tendency_limited_count = 0;
    int theta_bounds_clamp_count = 0;

    #pragma omp parallel for collapse(2) reduction(+:non_finite_theta_tendency_count, theta_tendency_limited_count, theta_bounds_clamp_count)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {

            for (int k = 0; k < NZ; ++k)
            {
                const double theta_old = static_cast<double>(theta[i][j][k]);
                double dtheta_total = (static_cast<double>(dtheta_dt[i][j][k])
                                     + static_cast<double>(dtheta_dt_rad[i][j][k])
                                     + static_cast<double>(dtheta_dt_pbl[i][j][k])) * dt_micro;
                if (!std::isfinite(dtheta_total))
                {
                    dtheta_total = 0.0;
                    ++non_finite_theta_tendency_count;
                }

                const double max_step = static_cast<double>(max_theta_step_change_k);
                double dtheta_limited = std::clamp(dtheta_total, -max_step, max_step);
                if (dtheta_limited != dtheta_total)
                {
                    ++theta_tendency_limited_count;
                }

                double theta_new = theta_old + dtheta_limited;
                float theta_bounded = clamp_theta_k(static_cast<float>(theta_new));
                if (theta_bounded != static_cast<float>(theta_new))
                {
                    ++theta_bounds_clamp_count;
                }
                theta[i][j][k] = theta_bounded;

                if (std::abs(dtheta_total) > 100.0 && i == 0 && j == 0 && k < 5)
                {
                    tmv::log_debug("[MICRO DEBUG] Large theta change at i=", i, ",j=", j, ",k=", k,
                                   ": ", theta_old, " -> ", theta[i][j][k],
                                   " (raw_delta=", dtheta_total,
                                   ", applied_delta=", dtheta_limited, ")");
                    tmv::log_debug("  dtheta_dt=", dtheta_dt[i][j][k],
                                   ", dtheta_dt_rad=", dtheta_dt_rad[i][j][k],
                                   ", dtheta_dt_pbl=", dtheta_dt_pbl[i][j][k]);
                }
                qv[i][j][k] = static_cast<float>(static_cast<double>(qv[i][j][k]) + (static_cast<double>(dqv_dt[i][j][k]) + static_cast<double>(dqv_dt_pbl[i][j][k])) * dt_micro);
                qc[i][j][k] = static_cast<float>(static_cast<double>(qc[i][j][k]) + static_cast<double>(dqc_dt[i][j][k]) * dt_micro);
                qr[i][j][k] = static_cast<float>(static_cast<double>(qr[i][j][k]) + static_cast<double>(dqr_dt[i][j][k]) * dt_micro);
                qi[i][j][k] = static_cast<float>(static_cast<double>(qi[i][j][k]) + static_cast<double>(dqi_dt[i][j][k]) * dt_micro);
                qs[i][j][k] = static_cast<float>(static_cast<double>(qs[i][j][k]) + static_cast<double>(dqs_dt[i][j][k]) * dt_micro);
                qg[i][j][k] = static_cast<float>(static_cast<double>(qg[i][j][k]) + static_cast<double>(dqg_dt[i][j][k]) * dt_micro);
                qh[i][j][k] = static_cast<float>(static_cast<double>(qh[i][j][k]) + static_cast<double>(dqh_dt[i][j][k]) * dt_micro);

                qv[i][j][k] = clamp_qv_kgkg(qv[i][j][k]);
                qc[i][j][k] = clamp_hydrometeor_kgkg(qc[i][j][k]);
                qr[i][j][k] = clamp_hydrometeor_kgkg(qr[i][j][k]);
                qi[i][j][k] = clamp_hydrometeor_kgkg(qi[i][j][k]);
                qs[i][j][k] = clamp_hydrometeor_kgkg(qs[i][j][k]);
                qg[i][j][k] = clamp_hydrometeor_kgkg(qg[i][j][k]);
                qh[i][j][k] = clamp_hydrometeor_kgkg(qh[i][j][k]);
            }
        }
    }

    if (non_finite_theta_tendency_count > 0 ||
        non_finite_tendency_sanitized > 0 ||
        theta_tendency_limited_count > 0 ||
        theta_bounds_clamp_count > 0)
    {
        tmv::log_warn("[MICROPHYSICS GUARD] non_finite_dtheta=",
                      non_finite_theta_tendency_count,
                      ", non_finite_tendency_sanitized=", non_finite_tendency_sanitized,
                      ", limited_dtheta=", theta_tendency_limited_count,
                      ", theta_bounds_clamped=", theta_bounds_clamp_count);
    }
}
