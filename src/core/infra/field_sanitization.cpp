#include "core/field/field_inventory.hpp"
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
#include "core/field/field_bounds.hpp"
#include "core/field/numerical_integrity.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/runtime/runtime_config.hpp"
#include "diagnostics/field_contract.hpp"
#include "diagnostics/field_validation.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

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
            tmv::record_numerical_nonfinite_incidents();
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
                tmv::record_numerical_nonfinite_incidents();
                data[idx] = 0.0f;
                ++sanitized;
            }
        }
        return sanitized;
    }

    return sanitize_field_nonfinite_and_bounds(field, min_value, max_value);
}

bool primary_state_verification_due(const tmv::ValidationPolicy& policy,
                                    bool scientific_diagnostics_due,
                                    bool debug_logging_enabled) noexcept
{
    if (policy.mode == tmv::GuardMode::Off)
    {
        return false;
    }
    if (policy.mode == tmv::GuardMode::Warn ||
        policy.mode == tmv::GuardMode::Strict)
    {
        return true;
    }
    return scientific_diagnostics_due || debug_logging_enabled;
}

int enforce_primary_state_bounds(const SimulationStateView& state, const char* stage)
{
    auto& fields = state.fields;
    const auto mode = global_validation_policy.mode;
    set_primary_state_repair_policy(
        mode == tmv::GuardMode::Sanitize || mode == tmv::GuardMode::Off,
        mode == tmv::GuardMode::Strict);
    if (mode == tmv::GuardMode::Off)
    {
        return 0;
    }

    struct GuardedField
    {
        const char* id;
        Field3D* field;
    };
    const auto guarded = std::apply([&](const auto&... member) {
        return std::array<GuardedField, sizeof...(member)>{{
            {member.name, &member.get(fields)}...}};
    }, tmv::fields::prognostic);

    const std::size_t expected_count = static_cast<std::size_t>(state.grid.nr)
        * static_cast<std::size_t>(state.grid.nth)
        * static_cast<std::size_t>(state.grid.nz);
    for (const auto& item : guarded)
    {
        if (!item.field->empty() && item.field->size() != expected_count)
        {
            throw std::runtime_error(
                std::string("primary-state guard shape mismatch for field '")
                + item.id + "' at stage '" + stage + "'");
        }
    }

    if (mode == tmv::GuardMode::Warn || mode == tmv::GuardMode::Strict)
    {
        std::size_t observed = 0;
        bool failed = false;
        for (const auto& item : guarded)
        {
            if (item.field->empty())
            {
                continue;
            }
            const tmv::FieldContract* contract = tmv::find_field_contract(item.id);
            if (contract == nullptr)
            {
                continue;
            }
            const auto result = tmv::validate_field3d_inplace(
                *item.field, *contract, global_validation_policy, false);
            const std::size_t field_observed =
                result.stats.nan_count + result.stats.inf_count
              + result.stats.below_min_count + result.stats.above_max_count;
            observed += field_observed;
            if (field_observed > 0)
            {
                tmv::log_warn(
                    "[PHYSICS GUARD FIELD] stage=", stage,
                    " field=", item.id,
                    " nan=", result.stats.nan_count,
                    " inf=", result.stats.inf_count,
                    " below_min=", result.stats.below_min_count,
                    " above_max=", result.stats.above_max_count,
                    " observed_min=", result.stats.min_value,
                    " observed_max=", result.stats.max_value);
            }
            failed = failed || result.failed;
        }
        if (observed > 0)
        {
            tmv::log_warn("[PHYSICS GUARD] stage=", stage,
                          " observed_primary_state_violations=", observed);
        }
        if (failed)
        {
            throw std::runtime_error(
                std::string("strict primary-state validation failed at stage '")
                + stage + "'");
        }
        return 0;
    }

    int corrected = 0;
    const auto sanitize = [&](Field3D& field, auto clamp)
    {
        if (field.empty())
        {
            return;
        }
        float* data = field.data();
        int field_corrected = 0;
        #pragma omp parallel for reduction(+:field_corrected)
        for (long long idx = 0; idx < static_cast<long long>(field.size()); ++idx)
        {
            const float old_value = data[idx];
            const float bounded = clamp(old_value);
            if (bounded != old_value)
            {
                ++field_corrected;
                data[idx] = bounded;
            }
        }
        corrected += field_corrected;
    };

    sanitize(fields.rho, clamp_density_kgm3);
    sanitize(fields.p, clamp_pressure_pa);
    sanitize(fields.theta, clamp_theta_k);
    sanitize(fields.u, clamp_wind_horizontal_ms);
    sanitize(fields.v, clamp_wind_horizontal_ms);
    sanitize(fields.w, clamp_wind_vertical_ms);
    sanitize(fields.qv, clamp_qv_kgkg);
    sanitize(fields.qc, clamp_hydrometeor_kgkg);
    sanitize(fields.qr, clamp_hydrometeor_kgkg);
    sanitize(fields.qi, clamp_hydrometeor_kgkg);
    sanitize(fields.qs, clamp_hydrometeor_kgkg);
    sanitize(fields.qg, clamp_hydrometeor_kgkg);
    sanitize(fields.qh, clamp_hydrometeor_kgkg);
    corrected += sanitize_field_nonfinite_and_contract_bounds(fields.tracer, "tracer");
    corrected += sanitize_field_nonfinite_and_contract_bounds(fields.tke, "tke");

    if (corrected > 0)
    {
        tmv::log_warn("[PHYSICS GUARD] stage=", stage,
                      " corrected_primary_state_samples=", corrected);
    }
    return corrected;
}

int enforce_primary_state_bounds(const char* stage)
{
    return enforce_primary_state_bounds(global_simulation_state(), stage);
}
