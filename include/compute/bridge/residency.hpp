#pragma once

struct ComputeBackendRuntimeState;

/**
 * @brief Marks a model-step boundary for the named resident-slot layer
 *        (GPU audit Phase 4.3d): within-step device-resident reuse
 *        windows open here and expire at the next call.
 */
void begin_resident_step_backend();
void begin_resident_step_backend(ComputeBackendRuntimeState& compute);

void invalidate_resident_field_backend(const char* field_id) noexcept;
void invalidate_resident_field_backend(ComputeBackendRuntimeState& compute, const char* field_id) noexcept;

void invalidate_all_resident_fields_backend() noexcept;
void invalidate_all_resident_fields_backend(ComputeBackendRuntimeState& compute) noexcept;
