#pragma once

struct ComputeBackendRuntimeState;

#include <string>
#include <vector>

#include "numerics/advection/advection_base.hpp"

struct VerticalFluxTemplateDescriptor
{
    std::string id;
    std::string summary;
    bool backend_dispatch_ready = false;
};

/**
 * @brief Returns active vertical-flux template id.
 */
const std::string& active_vertical_flux_template_id();
const std::string& active_vertical_flux_template_id(ComputeBackendRuntimeState& compute);

/**
 * @brief Sets the active vertical-flux template id.
 *
 * Unknown ids are ignored to preserve a safe default template.
 */
void set_active_vertical_flux_template_id(const std::string& template_id);
void set_active_vertical_flux_template_id(ComputeBackendRuntimeState& compute, const std::string& template_id);

/**
 * @brief Returns true if a vertical-flux template id is available.
 */
bool has_vertical_flux_template(const std::string& template_id);

/**
 * @brief Lists available vertical-flux templates.
 */
std::vector<VerticalFluxTemplateDescriptor> list_vertical_flux_templates();
std::vector<VerticalFluxTemplateDescriptor> list_vertical_flux_templates(ComputeBackendRuntimeState& compute);

/**
 * @brief Dispatches active vertical-flux template using backend-managed path.
 *
 * Returns true only after the selected native or direct CPU path completes.
 * False leaves the caller outputs uncommitted and does not imply that fallback
 * is permitted.
 */
bool dispatch_vertical_flux_template_backend(
    const AdvectionConfig& cfg,
    const AdvectionStateView& state,
    AdvectionTendencies& tendencies,
    AdvectionDiagnostics* diag_opt = nullptr);
bool dispatch_vertical_flux_template_backend(ComputeBackendRuntimeState& compute,
    const AdvectionConfig& cfg,
    const AdvectionStateView& state,
    AdvectionTendencies& tendencies,
    AdvectionDiagnostics* diag_opt = nullptr);
