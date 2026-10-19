#pragma once

#include "util/scheme_descriptor.hpp"

#include <string>
#include <utility>
#include <vector>

namespace tmv
{

inline constexpr const char* kCartesianAGridDynamicsId = "klemp_wilhelmson_split_cartesian_agrid";
inline constexpr const char* kCartesianCGridDynamicsId = "klemp_wilhelmson_split_cartesian_cgrid";

namespace dynamics_descriptor_detail
{
inline SchemeCapabilities capabilities(std::string    staggering,
                                       SchemeCoverage parity,
                                       int            peak_scratch_fields,
                                       bool           gpu_selectable = true,
                                       SchemeCoverage gpu_coverage   = SchemeCoverage::Covered)
{
    SchemeCapabilities value;
    value.declared           = true;
    value.coordinate_systems = {"cartesian"};
    value.staggerings        = {std::move(staggering)};
    value.backends = gpu_selectable ? std::vector<std::string>{"cpu", "gpu"} : std::vector<std::string>{"cpu"};
    value.required_state_fields = {"u", "v", "w", "rho", "p", "theta"};
    value.produced_state_fields = {
        "du_dt",
        "dv_dt",
        "dw_dt",
        "drho_dt",
        "dp_dt",
    };
    value.optional_products = {
        "vorticity_budget",
        "angular_momentum",
        "pressure_decomposition",
        "split_explicit_acoustic_substeps",
    };
    value.accepted_sounding_sources = {
        "parametric_cape",
        "file",
        "parametric_targets",
    };
    value.accepted_hodograph_sources              = {"auto", "wk_param", "zero"};
    value.accepted_trigger_sources                = {"warm_bubble", "none", "vortex_seed"};
    value.recommended_trigger_source              = "warm_bubble";
    value.storage.coverage                        = SchemeCoverage::Covered;
    value.storage.persistent_full_volume_fields   = 0;
    value.storage.peak_scratch_full_volume_fields = peak_scratch_fields;
    value.storage.note = "The dynamics object owns no persistent volumes; its declared scratch count excludes "
                         "orchestration-owned tendencies and acoustic buffers budgeted by their runtime owners";
    value.config_validation_contract  = "dynamics.scheme+grid.staggering";
    value.cpu_reference_coverage      = SchemeCoverage::Covered;
    value.gpu_implementation_coverage = gpu_selectable ? gpu_coverage : SchemeCoverage::None;
    value.cpu_gpu_parity_coverage     = gpu_selectable ? parity : SchemeCoverage::None;
    return value;
}
} // namespace dynamics_descriptor_detail

// Both discretizations expose only their explicitly declared qualified scope.
inline const std::vector<SchemeDescriptor>& get_dynamics_scheme_descriptors()
{
    using dynamics_descriptor_detail::capabilities;
    static const std::vector<SchemeDescriptor> descriptors = {
        {
            kCartesianAGridDynamicsId,
            {"cartesian", "cart", "cartesian_cpu"},
            SchemeMaturity::Qualified,
            true,
            capabilities("collocated", SchemeCoverage::Covered, 0, true, SchemeCoverage::Covered),
            "Cartesian A-grid (qualified flat and mapped CPU/native flat cores)",
            "Qualified for the flat collocated CPU family, including supported boundary pairs, Kessler, canonical Smagorinsky/TKE, slab/simple-grey, and YSU or MYNN/RRTMG column ownership; the stationary-Bell mapped CPU core; and the exact flat core's fallback-off Vulkan and Metal counterparts. The "
            "native profiles own slow tendencies, momentum-weighted continuity, nonlinear EOS, momentum, null-mode "
            "constraint, rigid boundaries, retained mass fluxes, and report-bearing transactional commit. The "
            "TKE and other retained active-physics closures are not part of these profiles; radiative lateral boundaries remain outside the declared A-grid scope.",
            {},
            true,
        },
        {
            kCartesianCGridDynamicsId,
            {},
            SchemeMaturity::Qualified,
            true,
            capabilities("c_grid", SchemeCoverage::Covered, 5),
        },
    };
    return descriptors;
}

inline std::string canonicalize_dynamics_scheme_id(const std::string& raw_id)
{
    return canonicalize_scheme_id(get_dynamics_scheme_descriptors(), raw_id);
}

} // namespace tmv
