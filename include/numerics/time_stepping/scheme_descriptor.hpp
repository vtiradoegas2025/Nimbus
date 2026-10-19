#pragma once

#include "util/scheme_descriptor.hpp"

#include <string>
#include <utility>
#include <vector>

namespace tmv
{

namespace time_stepping_descriptor_detail
{
inline SchemeCapabilities capabilities(bool split_explicit,
                                       std::vector<std::string> backends,
                                       SchemeCoverage gpu,
                                       SchemeCoverage parity,
                                       int fixed_persistent_fields,
                                       int persistent_fields_per_state)
{
    SchemeCapabilities value;
    value.declared = true;
    declare_initialization_not_applicable(value);
    value.coordinate_systems = split_explicit
        ? std::vector<std::string>{"any"}
        : std::vector<std::string>{"cartesian"};
    value.staggerings = split_explicit
        ? std::vector<std::string>{"any"}
        : std::vector<std::string>{"collocated", "c_grid"};
    value.backends = std::move(backends);
    value.required_state_fields = split_explicit
        ? std::vector<std::string>{
            "slow_tendency_callbacks", "fast_acoustic_callbacks",
            "boundary_condition_callback",
        }
        : std::vector<std::string>{"time_stepping_state", "rhs_callback"};
    value.produced_state_fields = {"advanced_prognostic_state"};
    value.optional_products = {"step_count", "max_cfl", "dt_history"};
    value.storage.coverage = SchemeCoverage::Covered;
    value.storage.persistent_full_volume_fields = fixed_persistent_fields;
    value.storage.peak_scratch_full_volume_fields = 0;
    value.storage.persistent_fields_per_integrated_field =
        persistent_fields_per_state;
    value.storage.note = split_explicit
        ? "Five reusable fast-tendency buffers; callback storage is caller-owned"
        : "Persistent stage and RHS buffers scale with integrated field count";
    value.config_validation_contract = split_explicit
        ? "numerics.time_stepping.scheme+split_acoustic=true"
        : "flat Cartesian CPU families and periodic stationary native C-grid/collocated RK with no-advection/TVD/mixed-WENO public/private staging, explicit/implicit constant diffusion, and admitted physics compositions";
    value.cpu_reference_coverage = SchemeCoverage::Covered;
    value.gpu_implementation_coverage = gpu;
    value.cpu_gpu_parity_coverage = parity;
    return value;
}
} // namespace time_stepping_descriptor_detail

inline const std::vector<SchemeDescriptor>& get_time_stepping_scheme_descriptors()
{
    using time_stepping_descriptor_detail::capabilities;
    static const std::vector<SchemeDescriptor> descriptors = {
        {
            "rk3", {"ssprk3"}, SchemeMaturity::Qualified, true,
            capabilities(false, {"cpu", "gpu"}, SchemeCoverage::Covered,
                         SchemeCoverage::Covered, 0, 4),
            "SSPRK3 coupled runtime integration supports maintained flat CPU families and periodic native C-grid/collocated profiles with staged public/private state and admitted physics compositions.",
            "Other retained boundary, moving-domain, terrain and nesting combinations remain tracked for engineering completion.",
            {},
            true,
        },
        {
            "rk4", {}, SchemeMaturity::Qualified, true,
            capabilities(false, {"cpu", "gpu"}, SchemeCoverage::Covered,
                         SchemeCoverage::Covered, 0, 7),
            "RK4 coupled runtime integration supports maintained flat CPU families and periodic native C-grid/collocated profiles with staged public/private state and admitted physics compositions.",
            "Other retained boundary, moving-domain, terrain and nesting combinations remain tracked for engineering completion.",
            {},
            true,
        },
        {
            "split_explicit", {"split_explicit_rk3", "klemp_wilhelmson"},
            SchemeMaturity::Qualified, true,
            capabilities(true, {"cpu", "gpu"}, SchemeCoverage::Covered,
                         SchemeCoverage::Covered, 5, 0), {}, {}, {}, true,
        },
    };
    return descriptors;
}

} // namespace tmv
