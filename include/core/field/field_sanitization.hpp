/**
 * @file field_sanitization.hpp
 * @brief Field-level NaN/Inf sanitization and physical-bounds enforcement.
 *
 * Provides utilities for clamping prognostic fields into valid physical
 * ranges and replacing non-finite values. These are safety-net operations
 * called after each physics stage in the dynamics orchestrator.
 */

#pragma once

#include "core/field/field3d.hpp"
#include "diagnostics/validation_policy.hpp"

struct SimulationStateView;

/**
 * @brief Replaces non-finite values with zero and clamps to [min_value, max_value].
 * @param field     The field to sanitize in-place.
 * @param min_value Lower bound for clamping.
 * @param max_value Upper bound for clamping.
 * @return Number of corrected samples.
 */
int sanitize_field_nonfinite_and_bounds(Field3D& field, float min_value, float max_value);

/**
 * @brief Sanitizes a field using bounds from its registered field contract.
 *
 * Looks up the field contract by `field_id`. If the contract defines min/max
 * bounds, delegates to `sanitize_field_nonfinite_and_bounds`. If no bounds
 * are registered, only replaces non-finite values with zero.
 *
 * @param field    The field to sanitize in-place.
 * @param field_id Contract identifier (e.g., "vorticity_r", "p_prime").
 * @return Number of corrected samples.
 */
int sanitize_field_nonfinite_and_contract_bounds(Field3D& field, const char* field_id);

/**
 * @brief Returns whether an independent primary-state verification sweep is due.
 *
 * Writer/application kernels own the normal sanitize-mode bounds contract.
 * Independent full-volume verification is unconditional for warn/strict runs
 * so observation cannot disappear when diagnostics and debug logging are off.
 * Sanitize follows the scientific-diagnostic cadence or debug logging, while
 * GuardMode::Off remains scan-free.
 */
bool primary_state_verification_due(const tmv::ValidationPolicy& policy,
                                    bool scientific_diagnostics_due,
                                    bool debug_logging_enabled) noexcept;

/**
 * @brief Applies the configured guard policy to all primary prognostic fields.
 *
 * Sanitize mode mutates invalid samples. Warn mode observes without mutation,
 * strict mode throws without mutation according to `guard_fail_on`, and off is
 * a no-op. The state includes thermodynamics, velocity, tracer, and TKE.
 *
 * @param state Runtime state view whose fields will be clamped.
 * @param stage Label used in guard diagnostics (e.g., "dynamics", "microphysics").
 * @return Number of corrected samples (zero for non-mutating modes).
 */
int enforce_primary_state_bounds(const SimulationStateView& state, const char* stage);

/**
 * @brief Compatibility wrapper that clamps the process-wide simulation state.
 */
int enforce_primary_state_bounds(const char* stage);
