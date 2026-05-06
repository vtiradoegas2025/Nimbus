/**
 * @file field_sanitization.cpp
 * @brief Field-level NaN/Inf sanitization and physical-bounds enforcement.
 *
 * Extracted from src/core/dynamics.cpp to give field sanitization its own
 * compilation unit. These utilities are called after each physics stage
 * (dynamics, microphysics, turbulence, diffusion, boundary conditions) to
 * ensure no non-finite or out-of-range values propagate into the next stage.
 */

#include "core/field/field_sanitization.hpp"
#include "core/runtime/simulation.hpp"
#include "diagnostics/field_contract.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

int sanitize_field_nonfinite_and_bounds(Field3D& field, float min_value, float max_value)
{
    if (field.empty())
    {
        return 0;
    }

    int sanitized = 0;
    float* const data = field.data();
    const std::size_t count = field.size();
    #pragma omp parallel for reduction(+:sanitized)
    for (long long idx = 0; idx < static_cast<long long>(count); ++idx)
    {
        const float old_value = data[idx];
        float new_value = old_value;
        if (!std::isfinite(static_cast<double>(new_value)))
        {
            new_value = 0.0f;
        }
        new_value = std::clamp(new_value, min_value, max_value);
        if (new_value != old_value)
        {
            ++sanitized;
        }
        data[idx] = new_value;
    }
    return sanitized;
}

int sanitize_field_nonfinite_and_contract_bounds(Field3D& field, const char* field_id)
{
    const tmv::FieldContract* contract = tmv::find_field_contract(field_id);
    const bool has_min = (contract != nullptr) && contract->default_bounds.has_min;
    const bool has_max = (contract != nullptr) && contract->default_bounds.has_max;
    const float min_value = has_min
        ? static_cast<float>(contract->default_bounds.min_value)
        : -std::numeric_limits<float>::infinity();
    const float max_value = has_max
        ? static_cast<float>(contract->default_bounds.max_value)
        : std::numeric_limits<float>::infinity();

    if (!has_min && !has_max)
    {
        if (field.empty())
        {
            return 0;
        }

        int sanitized = 0;
        float* const data = field.data();
        const std::size_t count = field.size();
        #pragma omp parallel for reduction(+:sanitized)
        for (long long idx = 0; idx < static_cast<long long>(count); ++idx)
        {
            const float old_value = data[idx];
            if (!std::isfinite(static_cast<double>(old_value)))
            {
                data[idx] = 0.0f;
                ++sanitized;
            }
        }
        return sanitized;
    }

    return sanitize_field_nonfinite_and_bounds(field, min_value, max_value);
}

int enforce_primary_state_bounds(const char* stage)
{
    if (rho.empty() || p.empty() || theta.empty() || qv.empty() ||
        qc.empty() || qr.empty() || qi.empty() || qs.empty() || qg.empty() || qh.empty())
    {
        return 0;
    }

    const std::size_t count = rho.size();
    if (p.size() != count || theta.size() != count ||
        qv.size() != count || qc.size() != count || qr.size() != count ||
        qi.size() != count || qs.size() != count || qg.size() != count || qh.size() != count)
    {
        return 0;
    }

    float* rho_data = rho.data();
    float* p_data = p.data();
    float* theta_data = theta.data();
    float* qv_data = qv.data();
    float* qc_data = qc.data();
    float* qr_data = qr.data();
    float* qi_data = qi.data();
    float* qs_data = qs.data();
    float* qg_data = qg.data();
    float* qh_data = qh.data();

    int corrected = 0;
    #pragma omp parallel for reduction(+:corrected)
    for (long long idx = 0; idx < static_cast<long long>(count); ++idx)
    {
        auto apply = [&](float& value, float bounded)
        {
            if (bounded != value)
            {
                ++corrected;
                value = bounded;
            }
        };

        apply(rho_data[idx], clamp_density_kgm3(rho_data[idx]));
        apply(p_data[idx], clamp_pressure_pa(p_data[idx]));
        apply(theta_data[idx], clamp_theta_k(theta_data[idx]));
        apply(qv_data[idx], clamp_qv_kgkg(qv_data[idx]));
        apply(qc_data[idx], clamp_hydrometeor_kgkg(qc_data[idx]));
        apply(qr_data[idx], clamp_hydrometeor_kgkg(qr_data[idx]));
        apply(qi_data[idx], clamp_hydrometeor_kgkg(qi_data[idx]));
        apply(qs_data[idx], clamp_hydrometeor_kgkg(qs_data[idx]));
        apply(qg_data[idx], clamp_hydrometeor_kgkg(qg_data[idx]));
        apply(qh_data[idx], clamp_hydrometeor_kgkg(qh_data[idx]));
    }

    if (corrected > 0)
    {
        tmv::log_warn("[PHYSICS GUARD] stage=", stage,
                      " corrected_primary_state_samples=", corrected);
    }
    return corrected;
}
