#pragma once

#include "util/scheme_descriptor.hpp"

#include <string>
#include <vector>

namespace tmv
{

namespace terrain_descriptor_detail
{
inline SchemeCapabilities capabilities(
    bool active, bool native = false, bool collocated = false)
{
    SchemeCapabilities value;
    value.declared = true;
    declare_initialization_not_applicable(value);
    value.coordinate_systems = {"cartesian"};
    value.staggerings = active && !collocated
        ? std::vector<std::string>{"c_grid"}
        : std::vector<std::string>{"collocated", "c_grid"};
    value.backends = native || !active
        ? std::vector<std::string>{"cpu", "gpu"}
        : std::vector<std::string>{"cpu"};
    value.required_state_fields = {"grid_geometry"};
    value.produced_state_fields = active
        ? std::vector<std::string>{"topography", "terrain_metrics"}
        : std::vector<std::string>{"flat_topography", "identity_metrics"};
    value.optional_products = {"height", "slope", "jacobian_diagnostics"};
    value.storage.coverage = SchemeCoverage::Covered;
    value.storage.persistent_full_volume_fields = 0;
    value.storage.peak_scratch_full_volume_fields = 0;
    value.storage.note =
        "Topography and metric volumes are explicit caller-owned outputs";
    value.config_validation_contract =
        "terrain.scheme+dynamics_supports_terrain";
    value.cpu_reference_coverage = SchemeCoverage::Covered;
    value.gpu_implementation_coverage = native
        ? SchemeCoverage::Covered : SchemeCoverage::NotApplicable;
    value.cpu_gpu_parity_coverage = native
        ? SchemeCoverage::Covered : SchemeCoverage::NotApplicable;
    return value;
}
} // namespace terrain_descriptor_detail

inline const std::vector<SchemeDescriptor>& get_terrain_scheme_descriptors()
{
    using terrain_descriptor_detail::capabilities;
    static const std::vector<SchemeDescriptor> descriptors = {
        {
            "none", {"flat"}, SchemeMaturity::Qualified, true,
            capabilities(false),
            {}, {}, {}, true,
        },
        {
            "bell", {}, SchemeMaturity::Qualified, true,
            capabilities(true, true, true),
            "Bell terrain",
            "Qualified selection includes the structural mapped C-grid CPU all-physics family with simple-grey or 112-point RRTMG shortwave, the stationary/constant-velocity PBL/implicit-scalar family, the dry collocated CPU reference, and exact dry stationary fallback-off Vulkan/Metal mapped cores, plus the maintained Earth-fixed Bell Kessler-or-Lin/slab/simple-grey/full-stochastic motion profile on Vulkan and Metal.",
            {},
            true,
        },
        {
            "schar", {"schaer"}, SchemeMaturity::Qualified, true,
            capabilities(true, true, true),
            "Schar terrain",
            "Qualified selection includes the structural mapped C-grid CPU all-physics family with simple-grey or 112-point RRTMG shortwave, the stationary/constant-velocity PBL/implicit-scalar family, the exact maximal active stationary collocated CPU family, and the exact dry stationary fallback-off Vulkan/Metal mapped cores.",
            {},
            true,
        },
    };
    return descriptors;
}

} // namespace tmv
