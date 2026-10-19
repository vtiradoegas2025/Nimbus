#pragma once

#include "util/scheme_descriptor.hpp"

#include <string>
#include <vector>

namespace tmv::chaos
{

namespace descriptor_detail
{
inline SchemeCapabilities capabilities(bool initial_conditions,
                                       bool tendencies)
{
    SchemeCapabilities value;
    value.declared = true;
    declare_initialization_not_applicable(value);
    value.coordinate_systems = {"cartesian"};
    // Initial state and evolving tendency multipliers consume the shared
    // physical-cell ownership contract. On the collocated grid every lateral
    // and vertical closure plane is excluded; C-grid behavior retains its
    // established full-storage stochastic field for evidence compatibility.
    value.staggerings = {"collocated", "c_grid"};
    // Generic scheme coverage stays host-composed. Exact profile-specific
    // native ownership is admitted separately by strict configuration.
    value.backends = {"cpu", "gpu"};
    if (initial_conditions)
    {
        value.required_state_fields.insert(
            value.required_state_fields.end(),
            {"u", "v", "w", "theta", "qv", "qc", "qr", "grid_metrics"});
        value.produced_state_fields.push_back("perturbed_initial_state");
    }
    if (tendencies)
    {
        value.required_state_fields.insert(
            value.required_state_fields.end(),
            {"physics_tendencies", "grid_metrics", "pbl_height"});
        value.produced_state_fields.push_back("stochastically_scaled_tendencies");
    }
    value.optional_products = {
        "noise_field", "realized_variance", "multiplier_extrema",
    };
    if (!initial_conditions && !tendencies)
    {
        value.storage.coverage = SchemeCoverage::NotApplicable;
        value.storage.persistent_full_volume_fields = 0;
        value.storage.peak_scratch_full_volume_fields = 0;
        value.storage.note = "No-op scheme owns no storage";
    }
    else
    {
        value.storage.coverage = SchemeCoverage::Partial;
        value.storage.note =
            "Earth-control storage is an explicit function of domain shape and Lx/Ly (bounded below by quarter-cell control spacing); initial-condition fields are processed sequentially, boundary-layer owns one current/previous process, and full-stochastic owns three";
    }
    value.config_validation_contract =
        "chaos.scheme+apply_to_fields+amplitude_maps";
    value.cpu_reference_coverage = initial_conditions || tendencies
        ? SchemeCoverage::Covered
        : SchemeCoverage::NotApplicable;
    value.gpu_implementation_coverage = initial_conditions || tendencies
        ? SchemeCoverage::None
        : SchemeCoverage::NotApplicable;
    value.cpu_gpu_parity_coverage = initial_conditions || tendencies
        ? SchemeCoverage::None
        : SchemeCoverage::NotApplicable;
    return value;
}
} // namespace descriptor_detail

inline const std::vector<SchemeDescriptor>& get_chaos_scheme_descriptors()
{
    using descriptor_detail::capabilities;
    static const std::vector<SchemeDescriptor> descriptors = {
        {
            "none", {}, SchemeMaturity::Qualified, true,
            capabilities(false, false),
        },
        {
            "initial_conditions", {"ic"}, SchemeMaturity::Qualified, true,
            capabilities(true, false),
            "Earth-addressed initial-condition perturbations",
            "Qualified on flat collocated A-grid and flat or mapped C-grid CPU state for explicit physical-cell targets, including deterministic threading, closure-plane preservation, physical-height tapering, EOS closure, and transactional fail-closed validation. Moisture perturbations require a profile-appropriate amplitude. The native accelerator Ready scope is limited to the exact full-stochastic domain-recycling profile.",
        },
        {
            "boundary_layer",
            {
                "pbl", "pbl_perturbation", "pbl_perturbations",
                "bl_perturbation", "bl_perturbations",
                "boundary_layer_perturbation", "boundary_layer_perturbations",
            },
            SchemeMaturity::Qualified,
            true,
            capabilities(false, true),
            "Earth-addressed stochastic boundary-layer perturbations",
            "Qualified for conservative held-source perturbation on stationary or constant-velocity flat Cartesian A-grid and flat or mapped Cartesian C-grid profiles, deterministic threading, closure preservation, restart, continuous subcell Earth reconstruction, and whole-cell Earth-window translation. The native accelerator Ready scope is limited to the exact full-stochastic domain-recycling profile.",
        },
        {
            "full_stochastic", {"full"}, SchemeMaturity::Qualified, true,
            capabilities(true, true),
            "Earth-addressed stochastic physics",
            "Qualified for conservative microphysics, PBL, and Smagorinsky-coefficient perturbations on stationary or constant-velocity flat Cartesian A-grid, plus the established microphysics/PBL/SGS profile on flat and mapped Cartesian C-grid, including deterministic threading, closure preservation, restart, continuous subcell Earth reconstruction, and whole-cell Earth-window translation. Exact fallback-off Vulkan and Metal domain-recycling profiles own Earth-control initialization, AR evolution, and reconstruction natively; rendering and application remain host-owned.",
        },
    };
    return descriptors;
}

} // namespace tmv::chaos
