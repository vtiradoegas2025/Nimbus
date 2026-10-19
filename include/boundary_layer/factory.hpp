#pragma once

// Public boundary-layer factory surface.

#include "boundary_layer/scheme_descriptor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tmv
{
class BoundaryLayerSchemeBase;
struct BoundaryLayerConfig;

using BoundaryLayerSchemeFactory =
    std::unique_ptr<BoundaryLayerSchemeBase> (*)(const BoundaryLayerConfig&);

std::unique_ptr<BoundaryLayerSchemeBase>
create_boundary_layer_scheme(const std::string& scheme_name);
std::vector<std::string> get_available_boundary_layer_schemes();
} // namespace tmv

using tmv::create_boundary_layer_scheme;
using tmv::get_available_boundary_layer_schemes;
using tmv::get_boundary_layer_scheme_descriptors;
