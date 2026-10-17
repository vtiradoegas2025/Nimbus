#pragma once

#include "util/scheme_descriptor.hpp"

#include <string>
#include <utility>
#include <vector>

namespace tmv
{

namespace turbulence_descriptor_detail
{
inline SchemeCapabilities capabilities(bool active,
                                       int memory_delta,
                                       std::vector<std::string> backends,
                                       SchemeCoverage gpu,
                                       SchemeCoverage parity,
                                       bool collocated = false)
{
    SchemeCapabilities value;
    value.declared = true;
    declare_initialization_not_applicable(value);
    value.coordinate_systems = {"cartesian"};
    // Flat collocated state has its own physical-domain finite-volume stress
    // operator; terrain and native accelerator SGS remain C-grid-only through
    // the aggregate interaction policy.
    value.staggerings = active && !collocated
        ? std::vector<std::string>{"c_grid"}
        : std::vector<std::string>{"collocated", "c_grid"};
    value.backends = std::move(backends);
    if (active)
    {
        value.required_state_fields = {"u", "v", "w", "theta", "qv", "rho"};
        if (memory_delta != 0)
            value.required_state_fields.push_back("tke");
        value.produced_state_fields = {
            "du_dt", "dv_dt", "dw_dt", "dtheta_dt", "dqv_dt", "dtke_dt",
        };
        value.optional_products = {"eddy_viscosity", "mixing_length"};
    }
    value.storage.persistent_full_volume_fields = 0;
    value.storage.peak_scratch_full_volume_fields = 0;
    if (!active)
    {
        value.storage.coverage = SchemeCoverage::NotApplicable;
        value.storage.note = "No-op scheme owns no storage";
    }
    else if (memory_delta == 0)
    {
        value.storage.coverage = SchemeCoverage::Covered;
        value.storage.persistent_full_volume_fields = 3;
        value.storage.peak_scratch_full_volume_fields = 3;
        value.storage.note =
            "Three closure coefficients; parity sampling may copy all three";
    }
    else
    {
        value.storage.coverage = SchemeCoverage::Covered;
        value.storage.persistent_full_volume_fields = 5;
        value.storage.peak_scratch_full_volume_fields = 2;
        value.storage.note =
            "TKE state plus four coefficients and two pooled production fields";
    }
    value.solver_high_water_delta_fields = memory_delta;
    value.memory_delta_calibrated = true;
    value.config_validation_contract = !active
        ? "turbulence.scheme"
        : memory_delta == 0
            ? "turbulence.scheme+turbulence.mode+turbulence.filter_width+"
              "turbulence.dynamic_coefficient+turbulence.stability_correction"
            : "turbulence.scheme+turbulence.mode+turbulence.filter_width+"
              "turbulence.stability_correction+turbulence.Pr_t+turbulence.Sc_t";
    value.cpu_reference_coverage = SchemeCoverage::Covered;
    value.gpu_implementation_coverage = gpu;
    value.cpu_gpu_parity_coverage = parity;
    return value;
}
} // namespace turbulence_descriptor_detail

inline const std::vector<SchemeDescriptor>& get_turbulence_scheme_descriptors()
{
    using turbulence_descriptor_detail::capabilities;
    static const std::vector<SchemeDescriptor> descriptors = {
        {
            "smagorinsky",
            {"smag", "smagorinsky_lilly", "smagorinsky-lilly"},
            SchemeMaturity::Qualified,
            true,
            capabilities(true, 0, {"cpu", "gpu"}, SchemeCoverage::Covered,
                         SchemeCoverage::Covered, true), {},
            "Dynamic Germano-Lilly coefficients are Ready only for the exact "
            "flat periodic C-grid CPU profile; Richardson selection uses the versioned Lilly-equilibrium balance in its exact flat CPU reference profile", {}, true,
        },
        {
            "tke", {"1.5", "1.5order", "1.5-order"},
            SchemeMaturity::Qualified, true,
            capabilities(true, 5, {"cpu", "gpu"}, SchemeCoverage::Covered,
                         SchemeCoverage::Covered, true),
            "Deardorff-Moeng prognostic TKE",
            "The tke_based stability correction is Ready in the declared "
            "flat C-grid, A-grid, and mapped CPU families and the exact flat "
            "Vulkan/Metal TKE/RRTMG reference; broader native combinations remain engineering incomplete", {}, true,
        },
        {
            "none", {"off", "disabled"}, SchemeMaturity::Qualified, true,
            capabilities(false, -9, {"cpu", "gpu"}, SchemeCoverage::NotApplicable,
                         SchemeCoverage::NotApplicable), {}, {}, {}, true,
        },
    };
    return descriptors;
}

} // namespace tmv
