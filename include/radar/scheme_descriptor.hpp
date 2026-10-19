#pragma once

#include "util/scheme_descriptor.hpp"

#include <string>
#include <utility>
#include <vector>

namespace tmv
{

namespace radar_descriptor_detail
{
inline SchemeCapabilities capabilities(std::vector<std::string> required,
                                       std::vector<std::string> products)
{
    SchemeCapabilities value;
    value.declared = true;
    declare_initialization_not_applicable(value);
    value.coordinate_systems = {"cartesian"};
    value.staggerings = {"collocated", "c_grid"};
    value.backends = {"cpu"};
    value.required_state_fields = std::move(required);
    value.produced_state_fields = std::move(products);
    value.optional_products = {"point_sampling", "scatterer_correction"};
    value.storage.coverage = SchemeCoverage::Covered;
    value.storage.persistent_full_volume_fields = 0;
    value.storage.peak_scratch_full_volume_fields = 0;
    value.storage.note = "Radar volumes are explicit caller-owned products";
    value.config_validation_contract = "radar.products+operator_tier+species_flags";
    value.cpu_reference_coverage = SchemeCoverage::Covered;
    value.gpu_implementation_coverage = SchemeCoverage::None;
    value.cpu_gpu_parity_coverage = SchemeCoverage::None;
    return value;
}
} // namespace radar_descriptor_detail

inline const std::vector<SchemeDescriptor>& get_radar_scheme_descriptors()
{
    using radar_descriptor_detail::capabilities;
    static const std::vector<SchemeDescriptor> descriptors = {
        {
            "reflectivity", {"dbz", "ref", "z", "reflectivity_dbz"},
            SchemeMaturity::Qualified, true,
            capabilities(
                {"rho", "qr", "qs", "qg", "qh", "qi"},
                {"Z_dBZ", "Ze_linear", "species_reflectivity"}),
        },
        {
            "velocity", {"vel", "vr", "radial_velocity"},
            SchemeMaturity::Qualified, true,
            capabilities(
                {"u", "v", "w", "radar_location"}, {"Vr"}),
        },
        {
            "zdr", {"kdp", "rhohv"}, SchemeMaturity::Qualified, true,
            capabilities(
                {"rho", "qr", "qs", "qg", "qh", "qi"},
                {"ZDR_dB", "ZH_dBZ", "ZV_dBZ"}),
        },
    };
    return descriptors;
}

} // namespace tmv
