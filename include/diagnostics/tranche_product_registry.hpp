#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/infra/physical_cell_domain.hpp"
#include "core/output/tranche_writer.hpp"
#include "core/runtime/simulation_state.hpp"

/**
 * @file tranche_product_registry.hpp
 * @brief Diagnostic tranche product registration from runtime state.
 */

namespace tmv
{
namespace diagnostics
{

struct TrancheProductRegistryContext
{
    std::size_t                            slice_size           = 0;
    float                                  reflectivity_dbz_min = -30.0f;
    float                                  reflectivity_dbz_max = 120.0f;
    const std::vector<float>&              precipitation_rate_surface_mm_h;
    std::vector<float>&                    accumulated_rainfall_surface_mm;
    const std::vector<float>&              accumulated_rainfall_residual_surface_mm;
    double&                                accumulated_rainfall_last_update_s;
    const std::vector<float>&              storm_u_level_mean;
    const std::vector<float>&              storm_v_level_mean;
    const std::unordered_set<std::string>& active_output_fields;
    SimulationStateView                    simulation_state;
    PhysicalCellDomain                     physical_domain;
    const Field3D&                         centered_u;
    const Field3D&                         centered_v;
    const Field3D&                         centered_w;
};

bool with_diagnostic_tranche_registry(TrancheProductRegistryContext& context,
                                      std::string&                   error_message,
                                      const std::function<bool(const std::vector<RegisteredDiagnosticProduct>&)>& body);

} // namespace diagnostics
} // namespace tmv
