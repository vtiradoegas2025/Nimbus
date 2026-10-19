#pragma once

#include "util/scheme_descriptor.hpp"

#include <string>
#include <vector>

namespace tmv::runtime
{

enum class StudioSelectionMode
{
    Form,
    ProductSetV1,
    FixedV1,
    UnavailableV1,
};

/** A selectable module and the descriptor catalog owned by that module. */
struct SchemeModuleCatalog
{
    std::string                          id;
    std::string                          selection_path;
    const std::vector<SchemeDescriptor>* schemes = nullptr;
    StudioSelectionMode                  studio_selection_mode = StudioSelectionMode::UnavailableV1;
    /** Canonical descriptor identity selected by the strict-v1 bridge. */
    std::string                          fixed_v1_scheme_id;
    /** Required evidence for fixed/unavailable policies; empty for Form. */
    std::string                          studio_selection_reason;
};

/** All descriptor-backed selectable modules compiled into this binary. */
const std::vector<SchemeModuleCatalog>& scheme_module_catalogs();

/**
 * Logical fields/resources supplied by the normal SimulationState runtime.
 *
 * This is deliberately an inventory of concrete base-runtime bindings, not a
 * union of descriptor requirements or optional products. Exact-binary
 * preflight consumers use it to fail closed when a selected scheme requires a
 * resource the runtime does not provide.
 */
const std::vector<std::string>& runtime_state_fields();

/** Stable machine-readable catalog consumed by Nimbus Studio. */
std::string serialize_scheme_catalog_json();

/** Native audit surface for the complete state-commit ownership matrix. */
std::string serialize_compute_ownership_matrix_json();

/** Native audit surface for descriptor-backed solver-process ownership. */
std::string serialize_compute_process_ownership_matrix_json();

} // namespace tmv::runtime
