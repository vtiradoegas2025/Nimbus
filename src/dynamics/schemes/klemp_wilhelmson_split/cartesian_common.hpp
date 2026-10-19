#pragma once

#include "core/field/field_bounds.hpp"

#include "core/field/field3d.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace nimbus::dynamics::cartesian
{

inline double safe_rho(double rho)
{
    if (std::isfinite(rho) && rho > 1.0e-6)
    {
        return rho;
    }
    if (!std::isfinite(rho))
    {
        tmv::record_numerical_nonfinite_incidents();
    }
    return primary_state_repairs_enabled ? 1.0 : rho;
}

inline double finite_or_zero(double value)
{
    if (!std::isfinite(value))
    {
        tmv::record_numerical_nonfinite_incidents();
    }
    return (std::isfinite(value) || !primary_state_repairs_enabled)
        ? value : 0.0;
}

inline double stored_base_value(const std::vector<double>& profile, int k)
{
    return static_cast<double>(
        static_cast<float>(profile[static_cast<std::size_t>(k)]));
}

template <typename... Fields>
inline void zero_fields(int nx, int ny, int nz, Fields&... fields)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                ((fields[i][j][k] = 0.0f), ...);
            }
}

inline double hydrometeor_mixing_ratio_at(
    const Field3D& qc,
    const Field3D& qr,
    const Field3D& qi,
    const Field3D& qs,
    const Field3D& qg,
    const Field3D& qh,
    int i,
    int j,
    int k)
{
    if (qc.empty()) return 0.0;
    return static_cast<double>(qc[i][j][k])
         + static_cast<double>(qr[i][j][k])
         + static_cast<double>(qi[i][j][k])
         + static_cast<double>(qs[i][j][k])
         + static_cast<double>(qg[i][j][k])
         + static_cast<double>(qh[i][j][k]);
}

inline double hydrometeor_mixing_ratio_z_face(
    const Field3D& qc,
    const Field3D& qr,
    const Field3D& qi,
    const Field3D& qs,
    const Field3D& qg,
    const Field3D& qh,
    int i,
    int j,
    int k)
{
    if (qc.empty()) return 0.0;
    return 0.5 * (hydrometeor_mixing_ratio_at(qc, qr, qi, qs, qg, qh, i, j, k)
                + hydrometeor_mixing_ratio_at(qc, qr, qi, qs, qg, qh, i, j, k + 1));
}

inline double moisture_buoyancy_at(
    const Field3D& qv,
    const std::vector<double>& qv0_base,
    double gravity,
    int i,
    int j,
    int k)
{
    if (qv.empty() || qv0_base.empty()) return 0.0;
    const double qv_val = static_cast<double>(qv[i][j][k]);
    return gravity * 0.608 * (
        qv_val - stored_base_value(qv0_base, k));
}

inline double moisture_buoyancy_z_face(
    const Field3D& qv,
    const std::vector<double>& qv0_base,
    double gravity,
    int i,
    int j,
    int k)
{
    if (qv.empty() || qv0_base.empty()) return 0.0;
    const double qv_face = 0.5 * (static_cast<double>(qv[i][j][k])
                                + static_cast<double>(qv[i][j][k + 1]));
    const double qv0_face = 0.5 * (
        stored_base_value(qv0_base, k)
        + stored_base_value(qv0_base, k + 1));
    return gravity * 0.608 * (qv_face - qv0_face);
}

inline bool fill_hydrometeor_loading_buffer(
    const Field3D& qc,
    const Field3D& qr,
    const Field3D& qi,
    const Field3D& qs,
    const Field3D& qg,
    const Field3D& qh,
    std::size_t n_total,
    bool compact_empty,
    std::vector<float>& loading)
{
    if (qc.empty())
    {
        loading.assign(compact_empty ? 1 : n_total, 0.0f);
        return false;
    }

    loading.assign(n_total, 0.0f);
    const float* qc_p = qc.data();
    const float* qr_p = qr.data();
    const float* qi_p = qi.data();
    const float* qs_p = qs.data();
    const float* qg_p = qg.data();
    const float* qh_p = qh.data();
    for (std::size_t idx = 0; idx < n_total; ++idx)
    {
        loading[idx] = qc_p[idx] + qr_p[idx] + qi_p[idx]
                     + qs_p[idx] + qg_p[idx] + qh_p[idx];
    }
    return true;
}

}  // namespace nimbus::dynamics::cartesian
