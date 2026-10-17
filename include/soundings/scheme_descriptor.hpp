#pragma once

#include "util/scheme_descriptor.hpp"

#include <string>
#include <utility>
#include <vector>

namespace tmv
{

inline const std::vector<SchemeDescriptor>& get_sounding_scheme_descriptors()
{
    SchemeCapabilities automatic;
    automatic.declared = true;
    declare_initialization_not_applicable(automatic);
    automatic.coordinate_systems = {"any"};
    automatic.staggerings = {"any"};
    automatic.backends = {"cpu"};
    automatic.required_state_fields = {"sounding_file"};
    automatic.produced_state_fields = {
        "height", "pressure", "temperature", "moisture", "wind_profile",
    };
    automatic.optional_products = {"quality_flags", "derived_thermodynamics"};
    automatic.storage.coverage = SchemeCoverage::NotApplicable;
    automatic.storage.persistent_full_volume_fields = 0;
    automatic.storage.peak_scratch_full_volume_fields = 0;
    automatic.storage.note = "Format detection and sounding profiles use host vectors";
    automatic.config_validation_contract = "init.sounding.file+content_detected_reader";
    automatic.cpu_reference_coverage = SchemeCoverage::Covered;
    automatic.gpu_implementation_coverage = SchemeCoverage::NotApplicable;
    automatic.cpu_gpu_parity_coverage = SchemeCoverage::NotApplicable;

    SchemeCapabilities cm1 = automatic;
    cm1.config_validation_contract = "cm1.input_sounding.native_text";
    cm1.storage.note = "Native CM1 profile uses host vectors, not grid volumes";

    SchemeCapabilities sharpy;
    sharpy.declared = true;
    declare_initialization_not_applicable(sharpy);
    sharpy.coordinate_systems = {"any"};
    sharpy.staggerings = {"any"};
    sharpy.backends = {"cpu"};
    sharpy.required_state_fields = {"sounding_file"};
    sharpy.produced_state_fields = {
        "height", "pressure", "temperature", "moisture", "wind_profile",
    };
    sharpy.optional_products = {"quality_flags", "derived_thermodynamics"};
    sharpy.storage.coverage = SchemeCoverage::NotApplicable;
    sharpy.storage.persistent_full_volume_fields = 0;
    sharpy.storage.peak_scratch_full_volume_fields = 0;
    sharpy.storage.note = "Sounding profiles use host vectors, not grid volumes";
    sharpy.config_validation_contract = "init.sounding.file+reader_backend";
    sharpy.cpu_reference_coverage = SchemeCoverage::Covered;
    sharpy.gpu_implementation_coverage = SchemeCoverage::NotApplicable;
    sharpy.cpu_gpu_parity_coverage = SchemeCoverage::NotApplicable;

    static const std::vector<SchemeDescriptor> descriptors = {
        {
            "auto", {"detect"}, SchemeMaturity::Qualified, true,
            std::move(automatic),
            {}, {}, {}, true,
        },
        {
            "cm1", {"cm1_text", "input_sounding"}, SchemeMaturity::Qualified,
            true, std::move(cm1), {}, {}, {}, true,
        },
        {
            "sharpy", {}, SchemeMaturity::Qualified, true, std::move(sharpy),
            {}, {}, {}, true,
        },
    };
    return descriptors;
}

} // namespace tmv

using tmv::get_sounding_scheme_descriptors;
