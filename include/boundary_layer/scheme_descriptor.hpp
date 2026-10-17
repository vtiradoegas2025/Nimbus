#pragma once

#include "util/scheme_descriptor.hpp"

#include <string>
#include <utility>
#include <vector>

namespace tmv
{

namespace boundary_layer_descriptor_detail
{
inline SchemeCapabilities capabilities(bool active, int memory_delta,
                                       bool native_accelerator = false,
                                       bool cloud_aware = false)
{
    SchemeCapabilities value;
    value.declared = true;
    declare_initialization_not_applicable(value);
    value.coordinate_systems = {"cartesian"};
    // The runtime presents only the owned PhysicalCellDomain to a column
    // closure. C-grid columns therefore begin at stored k=0, while collocated
    // columns exclude their bottom/top closure planes and rebase the first
    // physical level to column index zero.
    value.staggerings = {"collocated", "c_grid"};
    // `backends` describes globally composable execution. The separate
    // implementation/parity fields below identify whether the column closure
    // itself has a native accelerator path.
    value.backends = {"cpu", "gpu"};
    if (active)
    {
        value.required_state_fields = {
            "u", "v", "theta", "qv", "p", "rho", "z_interfaces",
            "surface_state",
        };
        value.produced_state_fields = {
            "du_dt_pbl", "dv_dt_pbl", "dtheta_dt_pbl", "dqv_dt_pbl",
            "dtke_dt_pbl",
        };
        value.optional_products = {"pbl_height", "surface_fluxes"};
        if (cloud_aware)
        {
            value.required_state_fields.insert(
                value.required_state_fields.end(),
                {"qc", "qi", "dtheta_dt_rad"});
            value.produced_state_fields.insert(
                value.produced_state_fields.end(),
                {"dqc_dt_pbl", "dqi_dt_pbl"});
            value.optional_products.push_back(
                "cloud_top_radiative_mixing");
        }
    }
    value.storage.coverage = active
        ? SchemeCoverage::Covered
        : SchemeCoverage::NotApplicable;
    value.storage.persistent_full_volume_fields = active
        ? (cloud_aware ? 6 : 4) : 0;
    value.storage.peak_scratch_full_volume_fields = active
        ? (cloud_aware ? 13 : 9) : 0;
    value.storage.note = active
        ? (cloud_aware
            ? "Six conserved held-source volumes persist; a transactional refresh owns seven tendency and six source candidates plus O(NZ) column work"
            : "Four conserved held-source volumes persist; a transactional refresh owns five tendency and four source candidates plus O(NZ) column work")
        : "No-op scheme owns no storage";
    value.solver_high_water_delta_fields = memory_delta;
    value.memory_delta_calibrated = true;
    value.config_validation_contract = "boundary_layer.scheme+surface";
    value.cpu_reference_coverage = SchemeCoverage::Covered;
    value.gpu_implementation_coverage = active
        ? (native_accelerator ? SchemeCoverage::Covered : SchemeCoverage::None)
        : SchemeCoverage::NotApplicable;
    value.cpu_gpu_parity_coverage = active
        ? (native_accelerator ? SchemeCoverage::Covered : SchemeCoverage::None)
        : SchemeCoverage::NotApplicable;
    return value;
}
} // namespace boundary_layer_descriptor_detail

inline const std::vector<SchemeDescriptor>& get_boundary_layer_scheme_descriptors()
{
    using boundary_layer_descriptor_detail::capabilities;
    static const std::vector<SchemeDescriptor> descriptors = {
        {
            "slab", {"mixed_layer", "mixed-layer"},
            SchemeMaturity::Qualified, true, capabilities(true, 31, true),
            {}, {}, {}, true,
        },
        {
            "ysu", {"yonsei", "yonsei_university"},
            SchemeMaturity::Qualified, true,
            capabilities(true, 31, false, true),
            "YSU",
            "Hong-Noh-Dudhia nonlocal K-profile with countergradient transport, explicit inversion entrainment, Richardson-dependent outer mixing, and an exact CPU surface-coupled Lock cloud-top radiative lane transporting linearized liquid-water potential temperature and qv/qc/qi; elevated decoupled clouds and campaigns remain separate.",
            {}, true,
        },
        {
            "mynn", {"mynn2", "mellor_yamada_nakanishi_niino"},
            SchemeMaturity::Qualified, true, capabilities(true, 31, true),
            "MYNN",
            "Nakanishi-Niino level-2.5 local closure with bounded coupled, restart, conservation, transactional, "
            "threaded, and accelerator coverage; sustained campaigns and external review remain.",
            {}, true,
        },
        {
            "none", {"off", "disabled"}, SchemeMaturity::Qualified, true,
            capabilities(false, 0),
            {}, {}, {}, true,
        },
    };
    return descriptors;
}

} // namespace tmv
