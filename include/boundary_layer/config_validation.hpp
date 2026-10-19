#pragma once

#include <string_view>

namespace tmv
{

struct BoundaryLayerConfig;
struct SurfaceConfig;

/**
 * Validate the complete configuration shared by live initialization and
 * portable restart. `canonical_scheme_id` must be a factory-canonical name.
 * Throws std::invalid_argument without mutating runtime state.
 */
void validate_boundary_layer_configuration(
    const BoundaryLayerConfig& config,
    const SurfaceConfig& surface,
    std::string_view canonical_scheme_id);

} // namespace tmv
