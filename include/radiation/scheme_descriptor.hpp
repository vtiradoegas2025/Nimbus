#pragma once

#include "util/scheme_descriptor.hpp"

#include <string>
#include <vector>

namespace tmv
{

namespace radiation_descriptor_detail
{
inline SchemeCapabilities capabilities(bool active, bool implemented,
                                       bool requires_moist_composition = false,
                                       bool native_accelerator = false,
                                       bool requires_rrtmg_composition = false)
{
    SchemeCapabilities value;
    value.declared = true;
    declare_initialization_not_applicable(value);
    value.coordinate_systems =
        implemented ? std::vector<std::string>{"cartesian"} : std::vector<std::string>{"not_applicable"};
    value.staggerings =
        implemented ? std::vector<std::string>{"collocated", "c_grid"} : std::vector<std::string>{"not_applicable"};
    // `backends` describes which globally selected orchestration profiles can
    // compose this scheme, not where this individual operator executes.
    // Native column coverage is declared separately from global orchestration.
    value.backends = implemented ? std::vector<std::string>{"cpu", "gpu"} : std::vector<std::string>{"not_implemented"};
    if (active)
    {
        value.required_state_fields = {
            "temperature",
            "p",
            "p_interfaces",
            "rho",
            "z_interfaces",
            "dz",
            "solar_geometry",
            "surface_radiative_state",
        };
        if (requires_moist_composition)
        {
            value.required_state_fields.push_back("qv");
            value.required_state_fields.push_back("qc");
            value.required_state_fields.push_back("qi");
        }
        value.produced_state_fields = {"dT_dt_rad", "dT_dt_lw", "dT_dt_sw"};
        value.optional_products     = {"longwave_fluxes", "shortwave_fluxes"};
    }
    value.storage.coverage                      = implemented ? SchemeCoverage::Covered : SchemeCoverage::NotApplicable;
    value.storage.persistent_full_volume_fields = active && implemented ? 1 : 0;
    value.storage.peak_scratch_full_volume_fields = active && implemented ? 2 : 0;
    value.storage.note = implemented
                             ? (active ? "One held per-cell atmospheric-power field plus two reusable transactional "
                                         "full-volume candidates and caller-owned O(NZ) column work; mapped shortwave "
                                         "adds configuration-sized spectral/angular workspaces and a bounded geometry cache "
                                         "through the runtime memory policy"
                                       : "No-op radiation owns no persistent field")
                             : "Unimplemented scheme allocates no storage";
    value.solver_high_water_delta_fields = active && implemented ? 3 : 0;
    value.memory_delta_calibrated        = implemented;
    value.config_validation_contract =
        implemented ? (requires_moist_composition
                           ? (requires_rrtmg_composition
                                  ? "radiation.scheme+explicit_idealized_solar_geometry+lw_sw_options+rrtmg_composition"
                                  : "radiation.scheme+versioned_absorbing_water_cloud_optics+explicit_mass_absorption_coefficients+subcycle_and_diagnostic_contract")
                           : "radiation.scheme+explicit_idealized_solar_geometry+lw_sw_options")
                    : "known-but-unimplemented; fail-closed";
    value.cpu_reference_coverage     = active && implemented ? SchemeCoverage::Covered : SchemeCoverage::NotApplicable;
    value.gpu_implementation_coverage = active && implemented
                                            ? (native_accelerator ? SchemeCoverage::Covered : SchemeCoverage::None)
                                            : SchemeCoverage::NotApplicable;
    value.cpu_gpu_parity_coverage = active && implemented
                                        ? (native_accelerator ? SchemeCoverage::Covered : SchemeCoverage::None)
                                        : SchemeCoverage::NotApplicable;
    return value;
}
} // namespace radiation_descriptor_detail

inline const std::vector<SchemeDescriptor>& get_radiation_scheme_descriptors()
{
    using radiation_descriptor_detail::capabilities;
    static const std::vector<SchemeDescriptor> descriptors = {
        {
            "simple_grey",
            {"simple-grey", "simplegray", "simplegrey", "gray", "grey"},
            SchemeMaturity::Qualified,
            true,
            capabilities(true, true, true, true),
            "Pressure-coordinate and absorbing water/cloud-path simple-grey radiation",
            "Qualified pressure-coordinate clear-sky and versioned pure-absorption cloudy lanes with explicit vapor, liquid, and ice "
            "mass-path coefficients, retained interface fluxes, and cadence subcycling. It excludes cloud scattering "
            "and does not claim observational all-sky accuracy.",
            {},
            true,
        },
        {
            "rrtmg",
            {"rrtmg_lw", "rrtmg_sw", "rrtmg_lw_sw"},
            SchemeMaturity::Qualified,
            true,
            capabilities(true, true, true, true, true),
            "RRTMG",
            "Native major/minor-gas, continuum, resolved liquid/ice cloud optics, and longwave/shortwave transfer pass "
            "official 52-interface clear and overcast MLS reference gates. Bounded coupled, restart, conservation, "
            "transactional, threaded, and accelerator tests pass. CPU mapped terrain additionally owns a conservative "
            "112-point transport-corrected scattering and sloped-reflection solve; sustained campaigns and external review remain.",
            {},
            true,
        },
        {
            "none",
            {"off", "disabled"},
            SchemeMaturity::Qualified,
            true,
            capabilities(false, true),
            {}, {}, {}, true,
        },
    };
    return descriptors;
}

} // namespace tmv
