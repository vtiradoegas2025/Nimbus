#pragma once

#include "util/scheme_descriptor.hpp"

#include <string>
#include <utility>
#include <vector>

namespace tmv
{

namespace microphysics_descriptor_detail
{
inline SchemeCapabilities
capabilities(bool active, int memory_delta, std::vector<std::string> backends,
             SchemeCoverage gpu, SchemeCoverage parity,
             bool supports_collocated = false)
{
    SchemeCapabilities value;
    value.declared = true;
    declare_initialization_not_applicable(value);
    value.coordinate_systems    = {"cartesian"};
    value.staggerings = active && !supports_collocated
        ? std::vector<std::string>{"c_grid"}
        : std::vector<std::string>{"collocated", "c_grid"};
    value.backends              = std::move(backends);
    if (active)
    {
        if (memory_delta == 0)
        {
            value.required_state_fields = {
                "p", "theta", "rho", "qv", "qc", "qr",
            };
            value.produced_state_fields = {
                "dtheta_dt", "dqv_dt", "dqc_dt", "dqr_dt",
            };
        }
        else
        {
            value.required_state_fields = {
                "p", "theta", "rho", "qv", "qc", "qr", "qi", "qs",
                "qg", "qh",
            };
            value.produced_state_fields = {
                "dtheta_dt", "dqv_dt", "dqc_dt", "dqr_dt", "dqi_dt",
                "dqs_dt", "dqg_dt", "dqh_dt",
            };
        }
    }
    value.storage.coverage                        = SchemeCoverage::Covered;
    value.storage.persistent_full_volume_fields   = 0;
    value.storage.peak_scratch_full_volume_fields = 0;
    if (memory_delta == 0)
    {
        value.storage.peak_scratch_full_volume_fields = 3;
        value.storage.note                            = "Kessler uses three pooled CPU work fields";
    }
    else if (memory_delta == 4)
    {
        value.storage.peak_scratch_full_volume_fields = 19;
        value.storage.note = "Lin reserves eight transactional tendencies, eight readback or parity fields, and three layer-thickness equivalents";
    }
    else if (memory_delta == 8)
    {
        value.storage.persistent_full_volume_fields   = 11;
        value.storage.peak_scratch_full_volume_fields = 384;
        value.storage.note = "Thompson owns five number moments, five pending tendencies and a surface ledger; "
                             "the conservative scratch reserve includes transactional process state, native "
                             "freezing and sedimentation staging, and device work buffers";
    }
    else if (memory_delta == 12)
    {
        value.storage.persistent_full_volume_fields   = 12;
        value.storage.peak_scratch_full_volume_fields = 256;
        value.storage.note =
            "Milbrandt owns six number moments and six pending tendencies; the conservative scratch reserve includes "
            "native process staging, packed cold inputs, column transport work, and transactional host assembly";
    }
    else
    {
        value.storage.coverage = SchemeCoverage::NotApplicable;
        value.storage.note     = "No-op scheme owns no storage";
    }
    value.solver_high_water_delta_fields = (memory_delta == 8 || memory_delta == 12) ? 96 : memory_delta;
    value.memory_delta_calibrated        = true;
    value.config_validation_contract     = "microphysics.scheme";
    value.cpu_reference_coverage         = SchemeCoverage::Covered;
    value.gpu_implementation_coverage    = gpu;
    value.cpu_gpu_parity_coverage        = parity;
    return value;
}

inline SchemeParameterDescriptor number_parameter(
    std::string path, std::string display_name, double default_value,
    double step, std::string unit, std::string description)
{
    SchemeParameterDescriptor value;
    value.path           = std::move(path);
    value.display_name   = std::move(display_name);
    value.default_number = default_value;
    value.minimum        = 0.0;
    value.step           = step;
    value.unit           = std::move(unit);
    value.description    = std::move(description);
    return value;
}

inline SchemeParameterDescriptor boolean_parameter(
    std::string path, std::string display_name, bool default_value,
    std::string description)
{
    SchemeParameterDescriptor value;
    value.path            = std::move(path);
    value.display_name    = std::move(display_name);
    value.kind            = SchemeParameterKind::Boolean;
    value.default_boolean = default_value;
    value.description     = std::move(description);
    return value;
}

inline std::vector<SchemeParameterDescriptor> parameters(const std::string& scheme)
{
    std::vector<SchemeParameterDescriptor> values;
    if (scheme == "none") return values;
    if (scheme != "milbrandt")
    {
        values.push_back(number_parameter(
            "microphysics.shared.cloud_water_threshold_kgkg",
            "Cloud-water conversion threshold", 1.0e-3, 1.0e-4, "kg/kg",
            "Cloud-water mixing ratio above which autoconversion can begin."));
        values.push_back(number_parameter(
            "microphysics.shared.autoconversion_rate_s_inv",
            "Autoconversion rate", 1.0e-3, 1.0e-4, "s⁻¹",
            "Rate coefficient converting cloud water above the threshold to rain."));
    }
    if (scheme == "kessler")
    {
        values.push_back(number_parameter(
            "microphysics.kessler.accretion_coefficient",
            "Rain accretion coefficient", 2.2, 0.1, "",
            "Coefficient for rain collecting cloud water."));
        values.push_back(number_parameter(
            "microphysics.kessler.evaporation_efficiency",
            "Rain evaporation efficiency", 1.0, 0.05, "",
            "Dimensionless multiplier on the published Kessler diffusion/ventilation evaporation kernel; 1 is the reference equation."));
    }
    else if (scheme == "lin")
    {
        values.push_back(number_parameter(
            "microphysics.lin.accretion_coefficient",
            "Rain accretion coefficient", 2.2, 0.1, "",
            "Coefficient for rain collecting cloud water in the Lin process chain."));
        values.push_back(number_parameter(
            "microphysics.lin.evaporation_rate_s_inv",
            "Rain evaporation rate", 1.0e-3, 1.0e-4, "s⁻¹",
            "Subsaturation-driven rain evaporation rate coefficient."));
    }
    else if (scheme == "thompson")
    {
        values.push_back(number_parameter(
            "microphysics.thompson.evaporation_rate_s_inv",
            "Compatibility evaporation rate", 1.0e-3, 1.0e-4, "s⁻¹",
            "Canonical compatibility parameter retained in portable identity; initialized Thompson–Eidhammer execution uses the pinned diffusion and ventilation equation."));
        values.push_back(number_parameter(
            "microphysics.thompson.water_friendly_aerosol_surface_m3",
            "Surface water-friendly aerosol number", 350.0e6, 10.0e6, "m⁻³",
            "Total surface endpoint of the fixed aerosol-aware Thompson CCN profile."));
        values.push_back(number_parameter(
            "microphysics.thompson.ice_friendly_aerosol_surface_m3",
            "Surface ice-friendly aerosol number", 2.0e6, 0.1e6, "m⁻³",
            "Total surface endpoint of the fixed aerosol-aware Thompson IN profile."));
    }
    else if (scheme == "milbrandt")
    {
        values.push_back(boolean_parameter(
            "microphysics.milbrandt.hail_processes", "New hail production", true,
            "Enable production of hail; any existing hail remains physically "
            "treated by melting, sublimation, collection, and sedimentation."));
    }
    return values;
}
} // namespace microphysics_descriptor_detail

inline const std::vector<SchemeDescriptor>& get_microphysics_scheme_descriptors()
{
    using microphysics_descriptor_detail::capabilities;
    using microphysics_descriptor_detail::parameters;
    static const std::vector<SchemeDescriptor> descriptors = {
        {
            "kessler",
            {},
            SchemeMaturity::Qualified,
            true,
            capabilities(true, 0, {"cpu", "gpu"}, SchemeCoverage::Covered, SchemeCoverage::Covered, true),
            "Kessler (warm rain)",
            "Qualified canonical qv/qc/qr warm rain only inside the exact admitted stationary and moving nested "
            "profiles; native accelerator process kernels retain the FP64 CPU state commit, while noncanonical "
            "parameters and other cross-products remain exploratory.",
            parameters("kessler"),
        },
        {
            "lin",
            {},
            SchemeMaturity::Qualified,
            true,
            capabilities(true, 4, {"cpu", "gpu"}, SchemeCoverage::Covered, SchemeCoverage::Covered, true),
            "Lin (single-moment ice)",
            "Qualified canonical single-moment mixed-phase Lin process chain on flat C-grid CPU/Vulkan/Metal and "
            "collocated profiles. Native pointwise/fallout honors physical-cell ownership and layer thickness; the strict host "
            "transaction retains state commit and FP64 ledgers. Additional coupled profiles remain tracked for completion.",
            parameters("lin"),
        },
        {
            "thompson",
            {},
            SchemeMaturity::Qualified,
            true,
            capabilities(true, 8, {"cpu", "gpu"}, SchemeCoverage::Covered, SchemeCoverage::Covered, true),
            "Thompson–Eidhammer (aerosol-aware)",
            "Thompson–Eidhammer CPU profiles and native C-grid process/sedimentation reference with transported Nc, "
            "Nr, Ni, Nwfa, and Nifa. The host retains the explicit FP64 transactional state commit and ledgers; long "
            "campaigns and independent scientific review remain pending.",
            parameters("thompson"),
        },
        {
            "milbrandt",
            {},
            SchemeMaturity::Qualified,
            true,
            capabilities(true, 12, {"cpu", "gpu"}, SchemeCoverage::Covered, SchemeCoverage::Covered, true),
            "Milbrandt-Yau (double moment)",
            "Milbrandt-Yau CPU profiles and a native C-grid process/sedimentation reference with six transported "
            "number "
            "moments. The host retains FP64 transfer assembly, transactional number-state commit and ledgers; "
            "long campaigns and independent scientific review remain pending.",
            parameters("milbrandt"),
            true,
        },
        {
            "none",
            {},
            SchemeMaturity::Qualified,
            true,
            capabilities(false, -10, {"cpu", "gpu"}, SchemeCoverage::NotApplicable, SchemeCoverage::NotApplicable),
            "None (dry dynamics)",
            "Disables cloud and precipitation microphysics.",
            parameters("none"),
        },
    };
    return descriptors;
}

} // namespace tmv
