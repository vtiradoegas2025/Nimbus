#pragma once

#include "util/scheme_descriptor.hpp"

#include <string>
#include <utility>
#include <vector>

namespace tmv
{

namespace diffusion_descriptor_detail
{
inline SchemeCapabilities capabilities(std::vector<std::string> backends,
                                       SchemeCoverage gpu,
                                       SchemeCoverage parity,
                                       int peak_scratch_fields)
{
    SchemeCapabilities value;
    value.declared = true;
    declare_initialization_not_applicable(value);
    value.coordinate_systems = {"cartesian"};
    value.staggerings = {"collocated", "c_grid"};
    value.backends = std::move(backends);
    value.required_state_fields = {
        "grid_metrics", "u", "v", "w", "theta", "qv",
        "live_rho_when_diffusion_active",
    };
    value.produced_state_fields = {
        "du_dt_diff", "dv_dt_diff", "dw_dt_diff", "dtheta_dt_diff",
        "dqv_dt_diff",
    };
    value.optional_products = {
        "K_momentum", "K_scalar",
        "prescribed_vertical_K(linear_height_scalar_centers)",
        "K_effective=max(K_h,K_v)",
        "max_diffusion_number",
    };
    value.storage.coverage = SchemeCoverage::Covered;
    value.storage.persistent_full_volume_fields = 0;
    value.storage.peak_scratch_full_volume_fields = peak_scratch_fields;
    value.storage.note = peak_scratch_fields > 1
                             ? "Bounded linear-solver vectors, atomic target staging and device transfer buffers"
                             : "Reusable transactional updated-state storage only for selected fields; tendencies and "
                               "optional diagnostics are caller-owned";
    value.config_validation_contract =
        "numerics.diffusion.scheme+operator_order=2; explicit and implicit K_h/K_v use the same live-density conservative finite-volume scalar and staggered-momentum equations; implicit advances them with backward Euler; explicit linear_height owns prescribed vertical K at physical scalar-cell centers with K_v=0";
    value.cpu_reference_coverage = SchemeCoverage::Covered;
    value.gpu_implementation_coverage = gpu;
    value.cpu_gpu_parity_coverage = parity;
    return value;
}
} // namespace diffusion_descriptor_detail

inline const std::vector<SchemeDescriptor>& get_diffusion_scheme_descriptors()
{
    using diffusion_descriptor_detail::capabilities;
    static const std::vector<SchemeDescriptor> descriptors = {
        {
            "explicit",
            {},
            SchemeMaturity::Qualified,
            true,
            capabilities({"cpu", "gpu"}, SchemeCoverage::Covered, SchemeCoverage::Covered, 1),
            {},
            {},
            {},
            true,
        },
        {
            "implicit",
            {},
            SchemeMaturity::Qualified,
            true,
            capabilities({"cpu", "gpu"}, SchemeCoverage::Covered, SchemeCoverage::Covered, 64),
            {},
            "Native selection is Ready for the exact flat C-grid implicit/RRTMG references; broader native couplings "
            "remain engineering incomplete",
            {},
            true,
        },
    };
    return descriptors;
}

} // namespace tmv
