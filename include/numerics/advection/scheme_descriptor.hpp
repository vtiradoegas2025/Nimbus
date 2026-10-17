#pragma once

#include "util/scheme_descriptor.hpp"

#include <string>
#include <utility>
#include <vector>

namespace tmv
{

namespace advection_descriptor_detail
{
inline SchemeCapabilities capabilities(bool active,
                                       int memory_delta,
                                       std::vector<std::string> backends,
                                       SchemeCoverage gpu,
                                       SchemeCoverage parity)
{
    SchemeCapabilities value;
    value.declared = true;
    declare_initialization_not_applicable(value);
    value.coordinate_systems = active
        ? std::vector<std::string>{"cartesian"}
        : std::vector<std::string>{"not_applicable"};
    value.staggerings = active
        ? std::vector<std::string>{"collocated", "c_grid"}
        : std::vector<std::string>{"not_applicable"};
    value.backends = std::move(backends);
    if (active)
    {
        // This factory selects the legacy vertical reconstruction path. The
        // production C-grid transport pipeline owns its multidimensional fluxes.
        value.required_state_fields = {"q", "w", "grid_metrics"};
        value.produced_state_fields = {"dqdt_adv", "flux_z"};
        value.optional_products = {"cfl", "flux_diagnostics"};
    }
    value.storage.coverage = active
        ? SchemeCoverage::Covered
        : SchemeCoverage::NotApplicable;
    value.storage.persistent_full_volume_fields = 0;
    value.storage.peak_scratch_full_volume_fields = 0;
    value.storage.note = active
        ? "Vertical reconstruction uses column-local work; tendencies are caller-owned"
        : "Planning sentinel owns no runtime storage";
    value.solver_high_water_delta_fields = memory_delta;
    value.memory_delta_calibrated = true;
    value.config_validation_contract = active
        ? "numerics.advection.scheme"
        : "legacy-memory-profile-only";
    value.cpu_reference_coverage = active
        ? SchemeCoverage::Covered
        : SchemeCoverage::NotApplicable;
    value.gpu_implementation_coverage = gpu;
    value.cpu_gpu_parity_coverage = parity;
    return value;
}
} // namespace advection_descriptor_detail

inline const std::vector<SchemeDescriptor>& get_advection_scheme_descriptors()
{
    using advection_descriptor_detail::capabilities;
    static const std::vector<SchemeDescriptor> descriptors = {
        {
            "tvd", {}, SchemeMaturity::Qualified, true,
            // The vertical factory is host-owned; native RK owns its separate
            // conservative MC face-flux and mass-tendency implementation.
            capabilities(true, 0, {"cpu", "gpu"}, SchemeCoverage::None,
                         SchemeCoverage::None), {},
            "The vertical factory remains host-owned. The exact canonical dry flat C-grid qualified Metal scope uses native conservative MC scalar transport with retained mass flux and strict transactional publication; other native RK and nested paths declare their own owners.",
            {}, true,
        },
        {
            "weno5", {"weno"}, SchemeMaturity::Qualified, true,
            capabilities(true, 0, {"cpu", "gpu"}, SchemeCoverage::Covered,
                         SchemeCoverage::Covered), {},
            "Coupled RK applies WENO5 to unbounded potential temperature and conservative TVD/MC to bounded vapor, tracer, hydrometeor, and TKE fields.",
            {}, true,
        },
        {
            "none", {"off", "disabled"}, SchemeMaturity::Qualified, true,
            capabilities(true, -9, {"cpu", "gpu"}, SchemeCoverage::NotApplicable,
                         SchemeCoverage::NotApplicable),
            "None (hold transported scalars)",
            "Disables resolved transport of theta, water species, passive tracer, TKE, and private microphysics moments while retaining the selected dynamics momentum, continuity, pressure, and physics-source equations.",
            {},
            true,
        },
    };
    return descriptors;
}

} // namespace tmv
