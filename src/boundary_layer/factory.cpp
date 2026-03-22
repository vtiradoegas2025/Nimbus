/**
 * @file factory.cpp
 * @brief Implementation for the boundary_layer module.
 *
 * Provides executable logic for the boundary_layer runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/boundary_layer subsystem.
 */

#include "factory.hpp"
#include "schemes/slab/slab.hpp"
#include "schemes/ysu/ysu.hpp"
#include "schemes/mynn/mynn.hpp"
#include "util/scheme_factory.hpp"

namespace
{
const tmv::SchemeRegistry<BoundaryLayerSchemeBase> registry({
    {"slab", [] { return std::make_unique<SlabScheme>(); }},
    {"ysu",  [] { return std::make_unique<YSUScheme>(); }},
    {"mynn", [] { return std::make_unique<MYNNScheme>(); }},
}, {
    {"yonsei",                        "ysu"},
    {"yonsei_university",             "ysu"},
    {"mixed_layer",                   "slab"},
    {"mixed-layer",                   "slab"},
    {"mynn2",                         "mynn"},
    {"mellor_yamada_nakanishi_niino", "mynn"},
});
}

/**
 * @brief Creates a boundary layer scheme based on the scheme name.
 */
std::unique_ptr<BoundaryLayerSchemeBase> create_boundary_layer_scheme(const std::string& scheme_name)
{
    return registry.create("boundary layer", scheme_name);
}

/**
 * @brief Returns the available boundary layer schemes.
 */
std::vector<std::string> get_available_boundary_layer_schemes()
{
    return registry.available_ids();
}
