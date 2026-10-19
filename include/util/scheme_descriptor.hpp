#pragma once

#include "util/string_utils.hpp"

#include <optional>
#include <string>
#include <vector>

namespace tmv
{

enum class SchemeMaturity
{
    Qualified,
    Compatibility,
    Experimental,
    Deferred,
};

inline const char* scheme_maturity_name(SchemeMaturity maturity)
{
    switch (maturity)
    {
    case SchemeMaturity::Qualified:
        return "qualified";
    case SchemeMaturity::Compatibility:
        return "compatibility";
    case SchemeMaturity::Experimental:
        return "experimental";
    case SchemeMaturity::Deferred:
        return "deferred";
    }
    return "unknown";
}

enum class SchemeCoverage
{
    Unspecified,
    NotApplicable,
    None,
    Partial,
    Covered,
};

struct SchemeStorageEstimate
{
    // Counts use one active-grid Field3D as the unit. Per-integrated-field
    // terms cover algorithms such as RK stages whose storage scales with the
    // number of state variables. A negative count means not yet quantified.
    SchemeCoverage coverage                                 = SchemeCoverage::Unspecified;
    int            persistent_full_volume_fields            = -1;
    int            peak_scratch_full_volume_fields          = -1;
    int            persistent_fields_per_integrated_field   = 0;
    int            peak_scratch_fields_per_integrated_field = 0;
    std::string    note;
};

struct SchemeCapabilities
{
    bool                     declared = false;
    std::vector<std::string> coordinate_systems;
    std::vector<std::string> staggerings;
    std::vector<std::string> backends;
    std::vector<std::string> required_state_fields;
    std::vector<std::string> produced_state_fields;
    std::vector<std::string> optional_products;
    std::vector<std::string> accepted_sounding_sources;
    std::vector<std::string> accepted_hodograph_sources;
    std::vector<std::string> accepted_trigger_sources;
    std::string              recommended_trigger_source;
    SchemeStorageEstimate    storage;

    // Calibrated delta from the Kessler/Smagorinsky/WENO CPU memory baseline.
    int  solver_high_water_delta_fields = 0;
    bool memory_delta_calibrated        = false;

    std::string    config_validation_contract;
    SchemeCoverage cpu_reference_coverage      = SchemeCoverage::Unspecified;
    SchemeCoverage gpu_implementation_coverage = SchemeCoverage::Unspecified;
    SchemeCoverage cpu_gpu_parity_coverage     = SchemeCoverage::Unspecified;
};

enum class SchemeParameterKind
{
    Float,
    Boolean,
};

struct SchemeParameterDescriptor
{
    std::string           path;
    std::string           display_name;
    SchemeParameterKind   kind = SchemeParameterKind::Float;
    double                default_number = 0.0;
    bool                  default_boolean = false;
    std::optional<double> minimum;
    std::optional<double> maximum;
    std::optional<double> step;
    std::string           unit;
    std::string           description;
};

inline void declare_initialization_not_applicable(SchemeCapabilities& capabilities)
{
    capabilities.accepted_sounding_sources  = {"not_applicable"};
    capabilities.accepted_hodograph_sources = {"not_applicable"};
    capabilities.accepted_trigger_sources   = {"not_applicable"};
    capabilities.recommended_trigger_source = "not_applicable";
}

struct SchemeDescriptor
{
    std::string              id;
    std::vector<std::string> aliases;
    SchemeMaturity           maturity      = SchemeMaturity::Experimental;
    bool                     constructible = true;
    SchemeCapabilities       capabilities;
    // Optional presentation metadata for catalog consumers such as Studio.
    // Empty values deliberately fall back to the canonical id / capability
    // metadata rather than creating a second required registry.
    std::string display_name;
    std::string caveat;
    std::vector<SchemeParameterDescriptor> parameters;
    // Factory availability is intentionally separate from user-run
    // availability: development schemes may remain directly constructible for
    // focused tests while the simulation runtime fails closed.
    bool runtime_selectable = true;
};

inline bool has_complete_scheme_capabilities(const SchemeDescriptor& descriptor)
{
    const auto& capabilities = descriptor.capabilities;
    const auto& storage      = capabilities.storage;
    const bool  quantified_storage =
        storage.coverage == SchemeCoverage::Partial ||
        ((storage.coverage == SchemeCoverage::Covered || storage.coverage == SchemeCoverage::NotApplicable) &&
         storage.persistent_full_volume_fields >= 0 && storage.peak_scratch_full_volume_fields >= 0 &&
         storage.persistent_fields_per_integrated_field >= 0 && storage.peak_scratch_fields_per_integrated_field >= 0);
    return capabilities.declared && !capabilities.coordinate_systems.empty() && !capabilities.staggerings.empty() &&
           !capabilities.backends.empty() && !capabilities.accepted_sounding_sources.empty() &&
           !capabilities.accepted_hodograph_sources.empty() && !capabilities.accepted_trigger_sources.empty() &&
           !capabilities.recommended_trigger_source.empty() && quantified_storage && !storage.note.empty() &&
           !capabilities.config_validation_contract.empty() &&
           capabilities.cpu_reference_coverage != SchemeCoverage::Unspecified &&
           capabilities.gpu_implementation_coverage != SchemeCoverage::Unspecified &&
           capabilities.cpu_gpu_parity_coverage != SchemeCoverage::Unspecified;
}

inline const SchemeDescriptor* find_scheme_descriptor(const std::vector<SchemeDescriptor>& descriptors,
                                                      const std::string&                   raw_id)
{
    const std::string normalized = strutil::trim_and_lower(raw_id);
    for (const auto& descriptor : descriptors)
    {
        if (descriptor.id == normalized)
        {
            return &descriptor;
        }
        for (const auto& alias : descriptor.aliases)
        {
            if (alias == normalized)
            {
                return &descriptor;
            }
        }
    }
    return nullptr;
}

inline std::string canonicalize_scheme_id(const std::vector<SchemeDescriptor>& descriptors, const std::string& raw_id)
{
    const auto* descriptor = find_scheme_descriptor(descriptors, raw_id);
    return descriptor ? descriptor->id : strutil::trim_and_lower(raw_id);
}

} // namespace tmv
