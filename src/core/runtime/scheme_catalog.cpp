#include "core/runtime/scheme_catalog.hpp"
#include "core/runtime/readiness.hpp"
#include "core/runtime/generated_metal_scope_v1.hpp"
#include "core/runtime/flat_mixed_metal_scope_v1.hpp"
#include "core/runtime/generated_metal_output_scope_v1.hpp"
#include "core/runtime/generated_metal_output_scope_v2.hpp"
#include "core/runtime/generated_metal_output_scope_v3.hpp"
#include "core/runtime/generated_metal_output_scope_nozfp_v2.hpp"
#include "core/runtime/generated_metal_east_npy_scope_v1.hpp"
#include "core/runtime/generated_metal_zfp_typed_scope_v1.hpp"
#include "core/runtime/science_profile.hpp"

#include "boundary_layer/scheme_descriptor.hpp"
#include "chaos/config_contract.hpp"
#include "chaos/scheme_descriptor.hpp"
#include "core/output/output_config.hpp"
#include "dynamics/scheme_descriptor.hpp"
#include "init/config/config_v1.hpp"
#include "microphysics/scheme_descriptor.hpp"
#include "microphysics/milbrandt_triple_scope.hpp"
#include "numerics/advection/scheme_descriptor.hpp"
#include "numerics/diffusion/scheme_descriptor.hpp"
#include "numerics/time_stepping/scheme_descriptor.hpp"
#include "radar/scheme_descriptor.hpp"
#include "radiation/scheme_descriptor.hpp"
#include "soundings/scheme_descriptor.hpp"
#include "terrain/scheme_descriptor.hpp"
#include "turbulence/scheme_descriptor.hpp"

#include <algorithm>
#include <array>
#include <iomanip>
#include <sstream>
#include <string_view>

namespace tmv::runtime
{
namespace
{

const char* coverage_name(SchemeCoverage coverage)
{
    switch (coverage)
    {
    case SchemeCoverage::Unspecified:
        return "unspecified";
    case SchemeCoverage::NotApplicable:
        return "not_applicable";
    case SchemeCoverage::None:
        return "none";
    case SchemeCoverage::Partial:
        return "partial";
    case SchemeCoverage::Covered:
        return "covered";
    }
    return "unknown";
}

const char* studio_selection_mode_name(StudioSelectionMode mode)
{
    switch (mode)
    {
    case StudioSelectionMode::Form:
        return "form";
    case StudioSelectionMode::ProductSetV1:
        return "product_set_v1";
    case StudioSelectionMode::FixedV1:
        return "fixed_v1";
    case StudioSelectionMode::UnavailableV1:
        return "unavailable_v1";
    }
    return "unavailable_v1";
}

void append_json_string(std::ostringstream& out, std::string_view value)
{
    out << '"';
    for (const unsigned char c : value)
    {
        switch (c)
        {
        case '"':
            out << "\\\"";
            break;
        case '\\':
            out << "\\\\";
            break;
        case '\b':
            out << "\\b";
            break;
        case '\f':
            out << "\\f";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (c < 0x20)
            {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec
                    << std::setfill(' ');
            }
            else
            {
                out << static_cast<char>(c);
            }
            break;
        }
    }
    out << '"';
}

void append_string_array(std::ostringstream& out, const std::vector<std::string>& values)
{
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (i != 0)
        {
            out << ',';
        }
        append_json_string(out, values[i]);
    }
    out << ']';
}

template <std::size_t N>
void append_string_view_array(std::ostringstream& out,
                              const std::array<std::string_view, N>& values)
{
    out << '[';
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0)
            out << ',';
        append_json_string(out, values[index]);
    }
    out << ']';
}

template <std::size_t N>
void append_int_array(std::ostringstream& out, const std::array<int, N>& values)
{
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (i != 0)
        {
            out << ',';
        }
        out << values[i];
    }
    out << ']';
}

void append_output_format_capability(std::ostringstream& out,
                                     std::string_view id,
                                     bool available,
                                     std::string_view reason,
                                     std::string_view implementation)
{
    out << "{\"id\":";
    append_json_string(out, id);
    out << ",\"available\":" << (available ? "true" : "false");
    out << ",\"reason\":";
    append_json_string(out, reason);
    out << ",\"implementation\":";
    append_json_string(out, implementation);
    out << '}';
}

void append_runtime_capabilities(std::ostringstream& out)
{
    out << "{\"outputFormats\":[";
    append_output_format_capability(out, "npy_3d", true, "", "builtin_npy");
    out << ',';
#ifdef HAVE_NETCDF
    append_output_format_capability(out, "netcdf", true, "", "libnetcdf");
#else
    append_output_format_capability(out,
                                    "netcdf",
                                    false,
                                    "This Nimbus binary was built without libnetcdf history-output support.",
                                    "");
#endif
    out << ',';
#ifdef HAVE_ZFP
    append_output_format_capability(out, "zfp_low", true, "", "libzfp");
    out << ',';
    append_output_format_capability(out, "zfp_medium", true, "", "libzfp");
    out << ',';
    append_output_format_capability(out, "zfp_high", true, "", "libzfp");
#else
    constexpr std::string_view zfp_reason =
        "This Nimbus binary was built without libzfp output support.";
    append_output_format_capability(out, "zfp_low", false, zfp_reason, "");
    out << ',';
    append_output_format_capability(out, "zfp_medium", false, zfp_reason, "");
    out << ',';
    append_output_format_capability(out, "zfp_high", false, zfp_reason, "");
#endif
    out << "],\"interactionPolicy\":{\"version\":14,\"turbulenceDynamicCoefficientDefault\":"
        << (tmv::init::config::TurbulenceSection{}.dynamic_coefficient ? "true" : "false");
#ifdef HAVE_ZFP
    out << ",\"generatedMetalCanonicalScope\":{\"version\":4,\"backend\":\"metal\","
           "\"executionMode\":\"qualified\",\"tupleCount\":"
        << tmv::runtime::kGeneratedMetalOutputScopeV3TupleCount
        << ",\"distinctConfigCount\":"
        << tmv::runtime::kGeneratedMetalOutputScopeV3DistinctConfigCount
        << ",\"outputFormats\":[\"npy_3d\",\"netcdf\",\"zfp_high\"],"
           "\"writeCadenceSecondsByFormat\":{\"npy_3d\":";
    append_int_array(out, tmv::runtime::kGeneratedMetalNpyWriteCadenceSeconds);
    out << ",\"netcdf\":";
    append_int_array(out, tmv::runtime::kGeneratedMetalNetcdfWriteCadenceSeconds);
    out << ",\"zfp_high\":";
    append_int_array(out, tmv::runtime::kGeneratedMetalZfpHighWriteCadenceSeconds);
    out << "},\"scalarTransportOwner\":\"native_conservative_mc\",\"baseManifestSha256\":";
    append_json_string(out, tmv::runtime::kGeneratedMetalScopeManifestSha256);
    out << ",\"npyNetcdfManifestSha256\":";
    append_json_string(out, tmv::runtime::kGeneratedMetalOutputScopeManifestSha256);
    out << ",\"zfpTypedOutputManifestSha256\":";
    append_json_string(out, tmv::runtime::kGeneratedMetalZfpTypedScopeManifestSha256);
    out << ",\"manifestSha256\":";
    append_json_string(out, tmv::runtime::kGeneratedMetalOutputScopeV3ManifestSha256);
#else
    out << ",\"generatedMetalCanonicalScope\":{\"version\":3,\"backend\":\"metal\","
           "\"executionMode\":\"qualified\",\"tupleCount\":"
        << tmv::runtime::kGeneratedMetalOutputScopeNoZfpV2TupleCount
        << ",\"distinctConfigCount\":"
        << tmv::runtime::kGeneratedMetalOutputScopeNoZfpV2DistinctConfigCount
        << ",\"outputFormats\":[\"npy_3d\",\"netcdf\"],"
           "\"writeCadenceSecondsByFormat\":{\"npy_3d\":";
    append_int_array(out, tmv::runtime::kGeneratedMetalNpyWriteCadenceSeconds);
    out << ",\"netcdf\":";
    append_int_array(out, tmv::runtime::kGeneratedMetalNetcdfWriteCadenceSeconds);
    out << "},\"scalarTransportOwner\":\"native_conservative_mc\",\"baseManifestSha256\":";
    append_json_string(out, tmv::runtime::kGeneratedMetalScopeManifestSha256);
    out << ",\"npyNetcdfManifestSha256\":";
    append_json_string(out, tmv::runtime::kGeneratedMetalOutputScopeManifestSha256);
    out << ",\"manifestSha256\":";
    append_json_string(out, tmv::runtime::kGeneratedMetalOutputScopeNoZfpV2ManifestSha256);
#endif
    out << ",\"additionalBoundaryNpyScope\":{\"version\":1,\"profileCount\":1,\"familyId\":";
    append_json_string(out, tmv::runtime::kGeneratedMetalEastNpyFamilyId);
    out << ",\"strictCanonicalYamlSha256\":";
    append_json_string(out, tmv::runtime::kGeneratedMetalEastNpyCanonicalYamlSha256);
    out << ",\"outputFormats\":[\"npy_3d\"],\"writeCadenceSeconds\":[1],\"manifestSha256\":";
    append_json_string(out, tmv::runtime::kGeneratedMetalEastNpyScopeManifestSha256);
    out << '}';
    out << '}';
    out << ",\"qualifiedFlatMixedMetalNpyScope\":"
        << tmv::runtime::kFlatMixedMetalCadence1ScopeJson.substr(
               0, tmv::runtime::kFlatMixedMetalCadence1ScopeJson.size() - 1)
        << ",\"manifestSha256\":";
    append_json_string(out, tmv::runtime::kFlatMixedMetalCadence1ScopeManifestSha256);
    out << '}';
    out << ",\"inactiveSchemeId\":";
    append_json_string(out, ScienceInteractionPolicy::inactive_scheme_id());
    out << ",\"horizontalOnlyTurbulenceModes\":";
    append_string_array(
        out, ScienceInteractionPolicy::horizontal_only_turbulence_modes());
    out << ",\"knownSurfaceSchemes\":";
    append_string_array(out, ScienceInteractionPolicy::known_surface_schemes());
    out << ",\"activeSurfaceSchemeId\":";
    append_json_string(out, ScienceInteractionPolicy::active_surface_scheme_id());
    out << ",\"terrainRadiationSchemes\":";
    append_string_array(
        out, ScienceInteractionPolicy::terrain_radiation_schemes());
    out << ",\"terrainLongwaveRadiationSchemeId\":";
    append_json_string(
        out, ScienceInteractionPolicy::terrain_longwave_radiation_scheme_id());
    out << ",\"terrainShortwaveAllowed\":true";
    out << ",\"terrainBoundaryLayerSchemes\":";
    append_string_array(
        out, ScienceInteractionPolicy::terrain_boundary_layer_schemes());
    out << ",\"terrainTurbulenceSchemes\":";
    append_string_array(
        out, ScienceInteractionPolicy::terrain_turbulence_schemes());
    out << ",\"terrainChaosSchemes\":";
    append_string_array(
        out, ScienceInteractionPolicy::terrain_chaos_schemes());
    out << ",\"chaosCanonicalTargets\":{";
    const auto& chaos_descriptors = chaos::get_chaos_scheme_descriptors();
    for (std::size_t index = 0; index < chaos_descriptors.size(); ++index)
    {
        if (index != 0)
            out << ',';
        const auto& scheme_id = chaos_descriptors[index].id;
        append_json_string(out, scheme_id);
        out << ":{\"initialConditions\":";
        append_string_array(
            out, chaos::canonical_chaos_initial_condition_targets(scheme_id));
        out << ",\"tendencies\":";
        append_string_array(
            out, chaos::canonical_chaos_tendency_targets(scheme_id));
        out << ",\"nativeGenerationRequired\":false}";
    }
    out << '}';
    out << ",\"terrainActiveDiffusionSchemes\":";
    append_string_array(
        out, ScienceInteractionPolicy::terrain_active_diffusion_schemes());
    out << ",\"terrainActiveDiffusionTargets\":";
    append_string_array(
        out, ScienceInteractionPolicy::terrain_active_diffusion_targets());
    out << ",\"qualifiedZfpTerrainSchemes\":";
    append_string_array(
        out, ScienceInteractionPolicy::qualified_zfp_terrain_schemes());
    out << ",\"qualifiedZfpPresetIds\":";
    append_string_array(out, qualified_zfp_preset_ids());
    out << ",\"qualifiedZfpPresetFields\":{";
    const auto& zfp_preset_fields = qualified_zfp_preset_fields();
    const auto& zfp_preset_ids = qualified_zfp_preset_ids();
    for (std::size_t index = 0; index < zfp_preset_ids.size(); ++index)
    {
        if (index != 0)
            out << ',';
        append_json_string(out, zfp_preset_ids[index]);
        out << ':';
        append_string_array(out, zfp_preset_fields.at(zfp_preset_ids[index]));
    }
    out << '}';
    out << ",\"qualifiedZfpMode\":";
    append_json_string(out, qualified_zfp_mode());
    out << ",\"qualifiedZfpFieldExecutionPolicies\":";
    append_string_array(out, qualified_zfp_field_execution_policies());
    out << ",\"qualifiedZfpMaxFieldWorkers\":"
        << qualified_zfp_max_field_workers();
    out << ",\"qualifiedZfpMinFieldScratchMib\":"
        << qualified_zfp_min_field_scratch_mib();
    out << ",\"qualifiedZfpMaxFieldScratchMib\":"
        << qualified_zfp_max_field_scratch_mib();
    out << ",\"qualifiedZfpRequiresPerFieldTolerances\":"
        << (qualified_zfp_requires_per_field_tolerances() ? "true" : "false");
    out << ",\"qualifiedZfpMinimumKeepLastFrames\":1";
    out << ",\"qualifiedZfpMinimumFreeSpaceMib\":1";
    out << ",\"qualifiedZfpRequiresRetainedFileCeiling\":true";
    out << ",\"qualifiedExecutionRequiresMaturityAcknowledgment\":false";
    out << ",\"qualifiedZfpMaxSparseThreshold\":"
        << std::setprecision(17) << qualified_zfp_max_sparse_threshold();
    out << ",\"qualifiedZfpAllowsFloat16Prefilter\":"
        << (qualified_zfp_allows_float16_prefilter() ? "true" : "false");
    out << ",\"qualifiedZfpFieldBudgets\":{";
    std::vector<std::string> budget_fields;
    budget_fields.reserve(qualified_zfp_field_budgets().size());
    for (const auto& [field, budget] : qualified_zfp_field_budgets())
    {
        (void)budget;
        budget_fields.push_back(field);
    }
    std::sort(budget_fields.begin(), budget_fields.end());
    for (std::size_t index = 0; index < budget_fields.size(); ++index)
    {
        if (index != 0)
            out << ',';
        const auto& field = budget_fields[index];
        append_json_string(out, field);
        out << ':' << std::setprecision(17)
            << qualified_zfp_field_budgets().at(field)
                   .max_configured_absolute_error;
    }
    out << '}';
    out << ",\"qualifiedZfpFieldUnits\":{";
    for (std::size_t index = 0; index < budget_fields.size(); ++index)
    {
        if (index != 0)
            out << ',';
        const auto& field = budget_fields[index];
        append_json_string(out, field);
        out << ':';
        append_json_string(out, qualified_zfp_field_budgets().at(field).units);
    }
    out << '}';
    out << ",\"qualifiedZfpReversibleRadarV4\":{\"version\":"
        << nimbus::output::kQualifiedRadarZfpV4ScopeVersion
        << ",\"admitted\":"
        << (nimbus::output::kQualifiedRadarZfpV4PublicAdmissionOpen ? "true" : "false")
        << ",\"productIds\":";
    append_string_view_array(out, nimbus::output::kQualifiedRadarZfpV4Products);
    out << ",\"nonemptyProductSubsets\":31,\"coreFields\":";
    append_string_view_array(out, nimbus::output::kQualifiedRadarZfpV4CoreFields);
    out << ",\"presets\":";
    append_string_view_array(out, nimbus::output::kQualifiedRadarZfpV4Presets);
    out << ",\"namedPresetProductSets\":{\"radar_ready\":[\"reflectivity_dbz\",\"radial_velocity\"],"
           "\"radar_polarimetric_ready\":[\"zdr\",\"kdp\",\"rhohv\"]},"
           "\"polarimetricExtraFields\":[\"qc\"],\"outputFormatTiers\":";
    append_string_view_array(out, nimbus::output::kQualifiedRadarZfpV4FormatTiers);
    out << ",\"backend\":\"cpu\",\"coordinateSystem\":\"cartesian\","
           "\"staggering\":\"c_grid\",\"terrain\":\"none\","
           "\"domainMotion\":\"stationary\",\"nesting\":false,"
           "\"reviewedSourceProfileContract\":\"nimbus.qualified-radar-source-profile.v2\","
           "\"reviewedSourceProfileOwnerSymbol\":\"is_public_qualified_radar_profile\","
           "\"reviewedSourceProfileFamilies\":[\"lin_cloudy_radar_ready_v1\","
           "\"kessler_warm_cloud_radar_v1\"],\"asyncIoModes\":[false,true],"
           "\"gridShape\":[12,12,48],\"gridSpacingMeters\":[1000,1000,250],"
           "\"dtLargeSeconds\":0.2,\"acousticSubsteps\":6,"
           "\"writeCadenceSeconds\":";
    append_int_array(out, nimbus::output::kQualifiedRadarZfpV4WriteCadenceSeconds);
    out << ",\"nonRadarMode\":\"accuracy\",\"nonRadarContainerVersion\":3,"
           "\"nonRadarBudgetOwner\":\"qualifiedZfpFieldBudgets\","
           "\"radarMode\":\"reversible\",\"radarContainerVersion\":4,"
           "\"radarKeyframeEveryExport\":true,\"radarTolerancePolicy\":\"forbidden\","
           "\"manifestContract\":\"nimbus.zfp.per_field.v2\","
           "\"sourceFieldBoundsContract\":\"registered_field_contract_v1\"}";
    out << ",\"qualifiedVerticalOnlySgs\":{\"version\":1,\"admitted\":true,"
           "\"scheme\":\"smagorinsky\",\"mode\":\"vertical_only\","
           "\"dynamicCoefficient\":false,\"executionMode\":\"qualified\","
           "\"backend\":\"cpu\",\"coordinateSystem\":\"cartesian\","
           "\"staggering\":\"c_grid\",\"terrain\":\"none\","
           "\"domainMotion\":\"stationary\",\"nesting\":false,"
           "\"lateralBoundaryMode\":\"periodic\",\"activeBoundaryLayerAllowed\":false,"
           "\"gridShape\":[12,12,16],\"gridSpacingMeters\":[1000,1000,250],"
           "\"dtLargeSeconds\":0.2,\"acousticSubsteps\":6,"
           "\"outputFormat\":\"npy_3d\",\"writeCadenceSeconds\":[1],"
           "\"sourceProfileContract\":\"nimbus.static-vertical-sgs-cpu.v1\","
           "\"sourceProfileOwnerSymbol\":\"is_transactional_static_vertical_only_cpu_candidate\","
           "\"wholeStepRollback\":\"state_and_six_science_ledgers\","
           "\"dynamicNonThreeDAllowed\":false}";
    out << ",\"milbrandtTripleMoment\":{\"version\":"
        << tmv::microphysics::kMilbrandtTripleMomentPolicyVersion
        << ",\"retained\":"
        << (tmv::microphysics::kMilbrandtTripleMomentRetained ? "true" : "false")
        << ",\"admitted\":"
        << (tmv::microphysics::kMilbrandtTripleMomentPublicAdmissionOpen ? "true" : "false")
        << ",\"implementationState\":\"engineering_incomplete\","
           "\"selector\":\"microphysics.milbrandt.triple_moment\","
           "\"strictV1ExplicitKeyAdmitted\":"
        << (tmv::microphysics::kMilbrandtTripleMomentStrictV1ExplicitKeyAdmitted ? "true" : "false")
        << ",\"scheme\":\"milbrandt\",\"baselineState\":[\"q\",\"N\"],"
           "\"additionalState\":[\"Z\"],\"additionalMomentOrder\":6,"
           "\"additionalMomentSpecies\":[\"rain\",\"ice\",\"snow\",\"graupel\",\"hail\"],"
           "\"privateStorageUnits\":\"m^6/kg_dry\"}";
    out << ",\"lateralBoundaryModes\":";
    append_string_array(
        out, ScienceInteractionPolicy::lateral_boundary_modes());
    out << ",\"radiativeLateralStaggerings\":";
    append_string_array(
        out, ScienceInteractionPolicy::radiative_lateral_staggerings());
    out << ",\"legacyOpenCgridResolution\":";
    append_json_string(
        out, ScienceInteractionPolicy::legacy_open_cgrid_resolution());
    out << ",\"legacyOpenCollocatedResolution\":";
    append_json_string(
        out, ScienceInteractionPolicy::legacy_open_collocated_resolution());
    out << ",\"periodicLateralPairsRequired\":true";
    out << ",\"bottomBoundaryModes\":";
    append_string_array(
        out, ScienceInteractionPolicy::bottom_boundary_modes());
    out << ",\"topBoundaryModes\":";
    append_string_array(
        out, ScienceInteractionPolicy::top_boundary_modes());
    out << ",\"scientificDiagnosticsOmittedSentinel\":-1";
    out << ",\"scientificDiagnosticsMinimumCadence\":0";
    out << ",\"gpuApplyControlsNoOpOnCpu\":true";
    out << ",\"gpuApplyMicrophysicsSchemes\":";
    append_string_array(
        out, ScienceInteractionPolicy::gpu_apply_microphysics_schemes());
    out << ",\"gpuApplySgsSchemes\":";
    append_string_array(out, ScienceInteractionPolicy::gpu_apply_sgs_schemes());
    out << ",\"gpuApplySlowDynamicsSchemes\":";
    append_string_array(
        out, ScienceInteractionPolicy::gpu_apply_slow_dynamics_schemes());
    out << ",\"qualifiedGpuApplySlowDynamicsSchemes\":";
    append_string_array(
        out,
        ScienceInteractionPolicy::
            qualified_gpu_apply_slow_dynamics_schemes());
    out << ",\"gpuApplyMicrophysicsDispatchGuardModes\":";
    append_string_array(
        out, ScienceInteractionPolicy::gpu_apply_mutating_dispatch_guard_modes());
    out << ",\"gpuApplyMicrophysicsHostCommitGuardModes\":";
    append_string_array(
        out, ScienceInteractionPolicy::gpu_apply_host_commit_guard_modes());
    out << ",\"gpuApplySlowDispatchGuardModes\":";
    append_string_array(
        out, ScienceInteractionPolicy::gpu_apply_mutating_dispatch_guard_modes());
    out << ",\"gpuApplySlowHostCommitGuardModes\":";
    append_string_array(
        out, ScienceInteractionPolicy::gpu_apply_host_commit_guard_modes());
    out << ",\"gpuApplySgsDispatchGuardModes\":";
    append_string_array(
        out, ScienceInteractionPolicy::gpu_apply_nonmutating_dispatch_guard_modes());
    out << "}}";
}

void append_optional_number(std::ostringstream& out,
                            const std::optional<double>& value)
{
    if (value.has_value()) out << std::setprecision(17) << *value;
    else out << "null";
}

void append_parameters(
    std::ostringstream& out,
    const std::vector<SchemeParameterDescriptor>& parameters)
{
    out << '[';
    for (std::size_t i = 0; i < parameters.size(); ++i)
    {
        if (i != 0) out << ',';
        const auto& parameter = parameters[i];
        out << "{\"path\":";
        append_json_string(out, parameter.path);
        out << ",\"displayName\":";
        append_json_string(out, parameter.display_name);
        out << ",\"kind\":";
        append_json_string(
            out, parameter.kind == SchemeParameterKind::Boolean ? "boolean" : "float");
        out << ",\"default\":";
        if (parameter.kind == SchemeParameterKind::Boolean)
            out << (parameter.default_boolean ? "true" : "false");
        else
            out << std::setprecision(17) << parameter.default_number;
        out << ",\"minimum\":";
        append_optional_number(out, parameter.minimum);
        out << ",\"maximum\":";
        append_optional_number(out, parameter.maximum);
        out << ",\"step\":";
        append_optional_number(out, parameter.step);
        out << ",\"unit\":";
        append_json_string(out, parameter.unit);
        out << ",\"description\":";
        append_json_string(out, parameter.description);
        out << '}';
    }
    out << ']';
}

void append_descriptor(std::ostringstream& out, const SchemeDescriptor& descriptor)
{
    const auto& capabilities = descriptor.capabilities;
    const auto& storage      = capabilities.storage;
    out << '{';
    out << "\"id\":";
    append_json_string(out, descriptor.id);
    out << ",\"aliases\":";
    append_string_array(out, descriptor.aliases);
    out << ",\"displayName\":";
    append_json_string(out, descriptor.display_name.empty() ? descriptor.id : descriptor.display_name);
    out << ",\"maturity\":";
    append_json_string(out, scheme_maturity_name(descriptor.maturity));
    out << ",\"constructible\":" << (descriptor.constructible ? "true" : "false");
    out << ",\"runtimeSelectable\":" << (descriptor.runtime_selectable ? "true" : "false");
    const bool descriptor_ready = descriptor.constructible &&
        descriptor.runtime_selectable &&
        descriptor.maturity == SchemeMaturity::Qualified &&
        has_complete_scheme_capabilities(descriptor);
    out << ",\"readinessState\":";
    append_json_string(
        out,
        descriptor_ready ? "ready" : "engineering_incomplete");
    out << ",\"caveat\":";
    append_json_string(out, descriptor.caveat);
    out << ",\"parameters\":";
    append_parameters(out, descriptor.parameters);
    out << ",\"capabilities\":{";
    out << "\"declared\":" << (capabilities.declared ? "true" : "false");
    out << ",\"coordinateSystems\":";
    append_string_array(out, capabilities.coordinate_systems);
    out << ",\"staggerings\":";
    append_string_array(out, capabilities.staggerings);
    out << ",\"backends\":";
    append_string_array(out, capabilities.backends);
    out << ",\"requiredStateFields\":";
    append_string_array(out, capabilities.required_state_fields);
    out << ",\"producedStateFields\":";
    append_string_array(out, capabilities.produced_state_fields);
    out << ",\"optionalProducts\":";
    append_string_array(out, capabilities.optional_products);
    out << ",\"acceptedSoundingSources\":";
    append_string_array(out, capabilities.accepted_sounding_sources);
    out << ",\"acceptedHodographSources\":";
    append_string_array(out, capabilities.accepted_hodograph_sources);
    out << ",\"acceptedTriggerSources\":";
    append_string_array(out, capabilities.accepted_trigger_sources);
    out << ",\"recommendedTriggerSource\":";
    append_json_string(out, capabilities.recommended_trigger_source);
    out << ",\"configValidationContract\":";
    append_json_string(out, capabilities.config_validation_contract);
    out << ",\"cpuReferenceCoverage\":";
    append_json_string(out, coverage_name(capabilities.cpu_reference_coverage));
    out << ",\"gpuImplementationCoverage\":";
    append_json_string(out, coverage_name(capabilities.gpu_implementation_coverage));
    out << ",\"cpuGpuParityCoverage\":";
    append_json_string(out, coverage_name(capabilities.cpu_gpu_parity_coverage));
    out << ",\"storage\":{";
    out << "\"coverage\":";
    append_json_string(out, coverage_name(storage.coverage));
    out << ",\"persistentFullVolumeFields\":" << storage.persistent_full_volume_fields;
    out << ",\"peakScratchFullVolumeFields\":" << storage.peak_scratch_full_volume_fields;
    out << ",\"persistentFieldsPerIntegratedField\":" << storage.persistent_fields_per_integrated_field;
    out << ",\"peakScratchFieldsPerIntegratedField\":" << storage.peak_scratch_fields_per_integrated_field;
    out << ",\"note\":";
    append_json_string(out, storage.note);
    out << "}}}";
}

void append_readiness_items(std::ostringstream& out)
{
    out << "{\"version\":1,\"items\":[";
    const auto& items = tracked_readiness_items();
    for (std::size_t index = 0; index < items.size(); ++index)
    {
        if (index != 0)
            out << ',';
        const auto& item = items[index];
        out << "{\"id\":";
        append_json_string(out, item.id);
        out << ",\"area\":";
        append_json_string(out, item.area);
        out << ",\"scope\":";
        append_json_string(out, item.scope);
        out << ",\"state\":";
        append_json_string(out, readiness_state_name(item.state));
        out << ",\"boundedEvidence\":";
        append_json_string(out, item.bounded_evidence);
        out << ",\"campaignEvidence\":";
        append_json_string(out, item.campaign_evidence);
        out << ",\"externalReviewEvidence\":";
        append_json_string(out, item.external_review_evidence);
        out << ",\"blocker\":";
        append_json_string(out, item.blocker);
        out << '}';
    }
    out << "]}";
}

} // namespace

const std::vector<SchemeModuleCatalog>& scheme_module_catalogs()
{
    static const std::vector<SchemeModuleCatalog> modules = {
        {"dynamics", "dynamics.scheme", &get_dynamics_scheme_descriptors(), StudioSelectionMode::Form, "", ""},
        {"advection", "advection.scheme", &get_advection_scheme_descriptors(),
         StudioSelectionMode::Form, "", ""},
        {"diffusion", "diffusion.scheme", &get_diffusion_scheme_descriptors(), StudioSelectionMode::Form, "", ""},
        {"time_stepping", "time.integrator", &get_time_stepping_scheme_descriptors(),
         StudioSelectionMode::Form, "", ""},
        {"microphysics", "microphysics.scheme", &get_microphysics_scheme_descriptors(), StudioSelectionMode::Form, "", ""},
        {"radiation", "radiation.scheme", &get_radiation_scheme_descriptors(), StudioSelectionMode::Form, "", ""},
        {"boundary_layer", "boundary_layer.scheme", &get_boundary_layer_scheme_descriptors(), StudioSelectionMode::Form, "", ""},
        {"turbulence", "turbulence.scheme", &get_turbulence_scheme_descriptors(), StudioSelectionMode::Form, "", ""},
        {"chaos", "extensions.chaos.scheme", &chaos::get_chaos_scheme_descriptors(), StudioSelectionMode::Form, "", ""},
        {"terrain", "terrain.scheme", &get_terrain_scheme_descriptors(),
         StudioSelectionMode::Form, "", ""},
        {"radar", "radar.products", &get_radar_scheme_descriptors(),
         StudioSelectionMode::ProductSetV1, "",
         "Strict v1 and Studio expose the descriptor-backed radar products as an ordered product set."},
        {"soundings", "initial_conditions.sounding.file.scheme_id", &get_sounding_scheme_descriptors(),
         StudioSelectionMode::FixedV1, "auto",
         "Strict v1 selects file sounding input and content-detects native CM1, TMV SHARPy, HDF5, or netCDF."},
    };
    return modules;
}

const std::vector<std::string>& runtime_state_fields()
{
    // PrognosticFieldStorage is allocated in resize_fields(). The remaining
    // names are concrete GridStateView / SimulationStateView resources built
    // by the normal orchestration path. Optional diagnostic products and
    // scheme-owned scratch fields are intentionally absent.
    static const std::vector<std::string> fields = {
        "rho", "p", "u", "v", "w", "tracer", "theta", "qv",
        "qc", "qr", "qi", "qs", "qh", "qg", "tke",
        "grid_metrics", "grid_geometry", "z_interfaces", "dz",
        "surface_state", "surface_radiative_state", "solar_geometry",
        "physics_tendencies", "pbl_height", "q", "temperature", "p_interfaces",
        "dtheta_dt_rad", "live_rho_when_diffusion_active",
        "slow_tendency_callbacks", "fast_acoustic_callbacks",
        "boundary_condition_callback", "time_stepping_state",
        "rhs_callback", "radar_location", "sounding_file",
    };
    return fields;
}

std::string serialize_scheme_catalog_json()
{
    std::ostringstream out;
    out << "{\"schema\":\"nimbus.scheme-catalog.v9\",\"catalogVersion\":9,\"runtimeStateFields\":";
    append_string_array(out, runtime_state_fields());
    out << ",\"readiness\":";
    append_readiness_items(out);
    out << ",\"runtimeCapabilities\":";
    append_runtime_capabilities(out);
    out << ",\"modules\":[";
    const auto& modules = scheme_module_catalogs();
    for (std::size_t module_index = 0; module_index < modules.size(); ++module_index)
    {
        if (module_index != 0)
        {
            out << ',';
        }
        const auto& module = modules[module_index];
        out << "{\"id\":";
        append_json_string(out, module.id);
        out << ",\"selectionPath\":";
        append_json_string(out, module.selection_path);
        out << ",\"studioSelection\":{\"mode\":";
        append_json_string(out, studio_selection_mode_name(module.studio_selection_mode));
        out << ",\"fixedSchemeId\":";
        append_json_string(out, module.fixed_v1_scheme_id);
        out << ",\"reason\":";
        append_json_string(out, module.studio_selection_reason);
        out << '}';
        out << ",\"schemes\":[";
        if (module.schemes != nullptr)
        {
            for (std::size_t scheme_index = 0; scheme_index < module.schemes->size(); ++scheme_index)
            {
                if (scheme_index != 0)
                {
                    out << ',';
                }
                append_descriptor(out, (*module.schemes)[scheme_index]);
            }
        }
        out << "]}";
    }
    out << "]}\n";
    return out.str();
}

std::string serialize_compute_ownership_matrix_json()
{
    const std::array<std::string, 2> backends = {"cpu", "gpu"};
    const std::array<std::string, 2> dynamics = {
        kCartesianCGridDynamicsId,
        kCartesianAGridDynamicsId,
    };
    const std::array<std::string, 5> microphysics = {
        "none", "kessler", "lin", "thompson", "milbrandt",
    };
    const std::array<std::string, 3> turbulence = {
        "none", "smagorinsky", "tke",
    };

    std::ostringstream records;
    std::size_t evaluated   = 0;
    std::size_t supported   = 0;
    std::size_t unsupported = 0;
    for (const auto& backend : backends)
        for (const bool qualified : {false, true})
            for (const auto& dynamics_id : dynamics)
                for (const auto& microphysics_id : microphysics)
                    for (const auto& turbulence_id : turbulence)
                        for (const bool apply_microphysics : {false, true})
                            for (const bool apply_slow : {false, true})
                                for (const bool apply_sgs : {false, true})
                                {
                                    ScienceInteractionSelection selection;
                                    selection.backend                = backend;
                                    selection.require_qualified      = qualified;
                                    selection.dynamics_id            = dynamics_id;
                                    selection.microphysics_id        = microphysics_id;
                                    selection.turbulence_id          = turbulence_id;
                                    selection.gpu_apply_microphysics = apply_microphysics;
                                    selection.gpu_apply_slow         = apply_slow;
                                    selection.gpu_apply_sgs          = apply_sgs;
                                    const auto decision = resolve_compute_ownership(selection);
                                    if (evaluated != 0)
                                        records << ',';
                                    records << "{\"index\":" << evaluated << ",\"backend\":";
                                    append_json_string(records, backend);
                                    records << ",\"requireQualified\":" << (qualified ? "true" : "false")
                                            << ",\"dynamics\":";
                                    append_json_string(records, dynamics_id);
                                    records << ",\"microphysics\":";
                                    append_json_string(records, microphysics_id);
                                    records << ",\"turbulence\":";
                                    append_json_string(records, turbulence_id);
                                    records << ",\"apply\":{\"microphysics\":"
                                            << (apply_microphysics ? "true" : "false")
                                            << ",\"slow\":" << (apply_slow ? "true" : "false")
                                            << ",\"sgs\":" << (apply_sgs ? "true" : "false") << '}';
                                    records << ",\"owners\":{\"microphysicsStateCommit\":";
                                    append_json_string(
                                        records, compute_commit_owner_name(decision.microphysics_state_commit));
                                    records << ",\"slowStateCommit\":";
                                    append_json_string(records, compute_commit_owner_name(decision.slow_state_commit));
                                    records << ",\"sgsStateCommit\":";
                                    append_json_string(records, compute_commit_owner_name(decision.sgs_state_commit));
                                    records << "},\"valid\":" << (decision.errors.empty() ? "true" : "false")
                                            << ",\"errors\":";
                                    append_string_array(records, decision.errors);
                                    records << '}';
                                    ++evaluated;
                                    if (decision.errors.empty())
                                        ++supported;
                                    else
                                        ++unsupported;
                                }

    std::ostringstream out;
    out << "{\"schema\":\"nimbus.compute-ownership-matrix.v1\",\"version\":1,"
           "\"scope\":\"state_commit_controls\",\"records\":["
        << records.str() << "],\"summary\":{\"evaluated\":" << evaluated
        << ",\"supported\":" << supported << ",\"unsupported\":" << unsupported << "}}\n";
    return out.str();
}

std::string serialize_compute_process_ownership_matrix_json()
{
    std::ostringstream records;
    std::size_t evaluated   = 0;
    std::size_t supported   = 0;
    std::size_t unsupported = 0;
    for (const auto& module : scheme_module_catalogs())
    {
        if (module.id == "radar" || module.id == "soundings" || module.schemes == nullptr)
            continue;
        for (const auto& descriptor : *module.schemes)
            for (const std::string& backend : {std::string("cpu"), std::string("gpu")})
            {
                const auto owner = resolve_compute_process_owner(descriptor, backend);
                if (evaluated != 0)
                    records << ',';
                records << "{\"index\":" << evaluated << ",\"module\":";
                append_json_string(records, module.id);
                records << ",\"scheme\":";
                append_json_string(records, descriptor.id);
                records << ",\"backend\":";
                append_json_string(records, backend);
                records << ",\"owner\":";
                append_json_string(records, compute_commit_owner_name(owner));
                records << ",\"valid\":" << (owner == ComputeCommitOwner::Unsupported ? "false" : "true")
                        << ",\"gpuImplementationCoverage\":";
                append_json_string(records, coverage_name(descriptor.capabilities.gpu_implementation_coverage));
                records << '}';
                ++evaluated;
                if (owner == ComputeCommitOwner::Unsupported)
                    ++unsupported;
                else
                    ++supported;
            }
    }

    std::ostringstream out;
    out << "{\"schema\":\"nimbus.compute-process-ownership-matrix.v1\",\"version\":1,"
           "\"scope\":\"descriptor_backed_solver_processes\",\"records\":["
        << records.str() << "],\"summary\":{\"evaluated\":" << evaluated
        << ",\"supported\":" << supported << ",\"unsupported\":" << unsupported << "}}\n";
    return out.str();
}

} // namespace tmv::runtime
