#include "catch2/catch.hpp"
#include "core/runtime/scheme_catalog.hpp"
#include "core/runtime/readiness.hpp"
#include "core/runtime/science_profile.hpp"
#include "core/runtime/generated_metal_output_scope_v2.hpp"
#include "core/runtime/flat_mixed_metal_scope_v1.hpp"
#include "core/infra/sha256.hpp"
#include "core/runtime/generated_metal_output_scope_v3.hpp"
#include "core/runtime/generated_metal_output_scope_nozfp_v2.hpp"
#include "core/runtime/generated_metal_east_npy_scope_v1.hpp"
#include "core/runtime/generated_metal_zfp_typed_scope_v1.hpp"
#include "boundary_layer/scheme_descriptor.hpp"
#include "chaos/config_contract.hpp"
#include "chaos/scheme_descriptor.hpp"
#include "dynamics/scheme_descriptor.hpp"
#include "microphysics/scheme_descriptor.hpp"
#include "numerics/advection/scheme_descriptor.hpp"
#include "numerics/diffusion/scheme_descriptor.hpp"
#include "numerics/time_stepping/scheme_descriptor.hpp"
#include "radar/scheme_descriptor.hpp"
#include "radiation/scheme_descriptor.hpp"
#include "soundings/scheme_descriptor.hpp"
#include "terrain/scheme_descriptor.hpp"
#include "turbulence/scheme_descriptor.hpp"

#include <map>
#include <array>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

TEST_CASE("runtime scheme catalog publishes every descriptor-backed module", "[core][runtime][scheme-catalog]")
{
    const auto& modules = tmv::runtime::scheme_module_catalogs();
    REQUIRE(modules.size() == 12);
    const std::map<std::string, std::pair<tmv::runtime::StudioSelectionMode, std::string>> expected = {
        {"dynamics", {tmv::runtime::StudioSelectionMode::Form, ""}},
        {"advection", {tmv::runtime::StudioSelectionMode::Form, ""}},
        {"diffusion", {tmv::runtime::StudioSelectionMode::Form, ""}},
        {"time_stepping", {tmv::runtime::StudioSelectionMode::Form, ""}},
        {"microphysics", {tmv::runtime::StudioSelectionMode::Form, ""}},
        {"radiation", {tmv::runtime::StudioSelectionMode::Form, ""}},
        {"boundary_layer", {tmv::runtime::StudioSelectionMode::Form, ""}},
        {"turbulence", {tmv::runtime::StudioSelectionMode::Form, ""}},
        {"chaos", {tmv::runtime::StudioSelectionMode::Form, ""}},
        {"terrain", {tmv::runtime::StudioSelectionMode::Form, ""}},
        {"radar", {tmv::runtime::StudioSelectionMode::ProductSetV1, ""}},
        {"soundings", {tmv::runtime::StudioSelectionMode::FixedV1, "auto"}},
    };

    std::set<std::string> ids;
    std::size_t           form_modules        = 0;
    std::size_t           product_set_modules = 0;
    std::size_t           fixed_modules       = 0;
    std::size_t           unavailable_modules = 0;
    for (const auto& module : modules)
    {
        REQUIRE_FALSE(module.id.empty());
        REQUIRE_FALSE(module.selection_path.empty());
        REQUIRE(module.schemes != nullptr);
        REQUIRE_FALSE(module.schemes->empty());
        REQUIRE(ids.insert(module.id).second);
        const auto expected_policy = expected.find(module.id);
        REQUIRE(expected_policy != expected.end());
        REQUIRE(module.studio_selection_mode == expected_policy->second.first);
        REQUIRE(module.fixed_v1_scheme_id == expected_policy->second.second);
        if (module.studio_selection_mode == tmv::runtime::StudioSelectionMode::Form)
        {
            ++form_modules;
            REQUIRE(module.fixed_v1_scheme_id.empty());
            REQUIRE(module.studio_selection_reason.empty());
        }
        else
        {
            REQUIRE_FALSE(module.studio_selection_reason.empty());
            if (module.studio_selection_mode == tmv::runtime::StudioSelectionMode::FixedV1)
            {
                ++fixed_modules;
                const auto* fixed = tmv::find_scheme_descriptor(*module.schemes, module.fixed_v1_scheme_id);
                REQUIRE(fixed != nullptr);
                REQUIRE(fixed->constructible);
                REQUIRE(fixed->id == module.fixed_v1_scheme_id);
            }
            else if (module.studio_selection_mode ==
                     tmv::runtime::StudioSelectionMode::ProductSetV1)
            {
                ++product_set_modules;
                REQUIRE(module.id == "radar");
                REQUIRE(module.selection_path == "radar.products");
                REQUIRE(module.fixed_v1_scheme_id.empty());
            }
            else
            {
                ++unavailable_modules;
                REQUIRE(module.fixed_v1_scheme_id.empty());
            }
        }
    }
    REQUIRE(form_modules == 10);
    REQUIRE(product_set_modules == 1);
    REQUIRE(fixed_modules == 1);
    REQUIRE(unavailable_modules == 0);
    REQUIRE(ids.count("microphysics") == 1);
    REQUIRE(ids.count("dynamics") == 1);
    REQUIRE(ids.count("soundings") == 1);
}

TEST_CASE("runtime scheme catalog JSON exposes canonical microphysics truth", "[core][runtime][scheme-catalog]")
{
    const std::string json = tmv::runtime::serialize_scheme_catalog_json();
    REQUIRE(json.find("\"schema\":\"nimbus.scheme-catalog.v9\"") != std::string::npos);
    REQUIRE(json.find("\"catalogVersion\":9") != std::string::npos);
    REQUIRE(json.find("\"readiness\":{\"version\":1,\"items\":[") != std::string::npos);
    REQUIRE(json.find("\"interactionPolicy\":{\"version\":14,\"turbulenceDynamicCoefficientDefault\":false") !=
            std::string::npos);
    REQUIRE(json.find("\"id\":\"radar\",\"selectionPath\":\"radar.products\",\"studioSelection\":{\"mode\":\"product_set_v1\"") != std::string::npos);
    REQUIRE(json.find("\"terrainShortwaveAllowed\":true") != std::string::npos);
    REQUIRE(json.find(
        "\"chaosCanonicalTargets\":{\"none\":{\"initialConditions\":[],"
        "\"tendencies\":[],\"nativeGenerationRequired\":false},"
        "\"initial_conditions\":{\"initialConditions\":[\"u\",\"v\",\"w\","
        "\"theta\"],\"tendencies\":[],\"nativeGenerationRequired\":false}")
        != std::string::npos);
    REQUIRE(json.find("\"qualifiedZfpTerrainSchemes\":[\"none\",\"bell\",\"schar\"]") != std::string::npos);
    REQUIRE(json.find("\"qualifiedZfpPresetIds\":[\"minimal\",\"presentation\",\"studio_full\",\"dynamics_full\",\"science_validation\"]") != std::string::npos);
    REQUIRE(json.find("\"qualifiedZfpPresetFields\":{\"minimal\":[\"p\",\"qr\",\"qv\",\"rho\",\"theta\",\"u\",\"v\",\"w\"]") != std::string::npos);
    REQUIRE(json.find("\"qualifiedZfpMode\":\"accuracy\"") != std::string::npos);
    REQUIRE(json.find("\"qualifiedZfpFieldBudgets\":{") != std::string::npos);
    REQUIRE(json.find("\"qualifiedZfpReversibleRadarV4\":{\"version\":1,\"admitted\":true") !=
            std::string::npos);
    REQUIRE(json.find("\"qualifiedVerticalOnlySgs\":{\"version\":1,\"admitted\":true") !=
            std::string::npos);
    REQUIRE(json.find("\"dynamicCoefficient\":false,\"executionMode\":\"qualified\",\"backend\":\"cpu\"") !=
            std::string::npos);
    REQUIRE(json.find("\"dynamicNonThreeDAllowed\":false") != std::string::npos);
    REQUIRE(json.find("\"milbrandtTripleMoment\":{\"version\":1,\"retained\":true,\"admitted\":false,") !=
            std::string::npos);
    REQUIRE(json.find("\"strictV1ExplicitKeyAdmitted\":false") != std::string::npos);
    REQUIRE(json.find("\"baselineState\":[\"q\",\"N\"],\"additionalState\":[\"Z\"]") !=
            std::string::npos);
    REQUIRE(json.find("\"additionalMomentSpecies\":[\"rain\",\"ice\",\"snow\",\"graupel\",\"hail\"]") !=
            std::string::npos);
    REQUIRE(json.find("\"privateStorageUnits\":\"m^6/kg_dry\"") != std::string::npos);
    REQUIRE(json.find("\"id\":\"microphysics.milbrandt_triple_moment\"") != std::string::npos);
    REQUIRE(json.find("\"nonemptyProductSubsets\":31") != std::string::npos);
    REQUIRE(json.find("\"outputFormatTiers\":[\"zfp_low\",\"zfp_medium\",\"zfp_high\"]") !=
            std::string::npos);
    REQUIRE(json.find("\"radarTolerancePolicy\":\"forbidden\"") != std::string::npos);
    REQUIRE(json.find("\"reviewedSourceProfileContract\":\"nimbus.qualified-radar-source-profile.v2\"") !=
            std::string::npos);
    REQUIRE(json.find("\"reviewedSourceProfileOwnerSymbol\":\"is_public_qualified_radar_profile\"") !=
            std::string::npos);
    REQUIRE(json.find("\"asyncIoModes\":[false,true]") != std::string::npos);
    REQUIRE(json.find("\"p\":0.050000000000000003") != std::string::npos);
    REQUIRE(json.find("\"potential_vorticity\":1e-08") != std::string::npos);
    REQUIRE(json.find("\"lateralBoundaryModes\":[\"periodic\",\"radiative\",\"zero_gradient\",\"open\",\"rigid\","
                      "\"rigid_slip\"]") != std::string::npos);
    REQUIRE(json.find("\"legacyOpenCgridResolution\":\"radiative\"") != std::string::npos);
    REQUIRE(json.find("\"legacyOpenCollocatedResolution\":\"zero_gradient\"") != std::string::npos);
    REQUIRE(json.find("\"scientificDiagnosticsOmittedSentinel\":-1") != std::string::npos);
    REQUIRE(json.find("\"gpuApplyControlsNoOpOnCpu\":true") != std::string::npos);
    REQUIRE(json.find("\"gpuApplyMicrophysicsSchemes\":[\"kessler\",\"lin\",\"thompson\",\"milbrandt\"]") !=
            std::string::npos);
    REQUIRE(json.find("\"gpuApplySgsSchemes\":[\"smagorinsky\",\"tke\"]") != std::string::npos);
    REQUIRE(json.find("\"gpuApplyMicrophysicsDispatchGuardModes\":[\"off\"]") != std::string::npos);
    REQUIRE(json.find("\"gpuApplyMicrophysicsHostCommitGuardModes\":[\"warn\",\"error\"]") != std::string::npos);
    REQUIRE(json.find("\"gpuApplySlowDispatchGuardModes\":[\"off\"]") != std::string::npos);
    REQUIRE(json.find("\"gpuApplySlowHostCommitGuardModes\":[\"warn\",\"error\"]") != std::string::npos);
    REQUIRE(json.find("\"gpuApplySgsDispatchGuardModes\":[\"off\",\"warn\",\"error\"]") != std::string::npos);
    REQUIRE(json.find("\"qualifiedGpuApplySlowDynamicsSchemes\":[\"klemp_wilhelmson_split_cartesian_agrid\",\"klemp_wilhelmson_split_cartesian_cgrid\"]") != std::string::npos);
    REQUIRE(json.find("\"runtimeStateFields\":[\"rho\",\"p\",\"u\",\"v\",\"w\"") != std::string::npos);
    REQUIRE(json.find("\"grid_metrics\"") != std::string::npos);
    REQUIRE(json.find("\"id\":\"microphysics\"") != std::string::npos);
    REQUIRE(json.find("\"selectionPath\":\"microphysics.scheme\"") != std::string::npos);
    REQUIRE(json.find("\"studioSelection\":{\"mode\":\"form\",\"fixedSchemeId\":\"\",\"reason\":\"\"}") !=
            std::string::npos);
    REQUIRE(json.find("\"id\":\"time_stepping\",\"selectionPath\":\"time.integrator\",\"studioSelection\":{\"mode\":\"form\"") != std::string::npos);
    REQUIRE(json.find("\"mode\":\"product_set_v1\",\"fixedSchemeId\":\"\"") != std::string::npos);
    REQUIRE(json.find("\"id\":\"kessler\"") != std::string::npos);
    REQUIRE(json.find("\"id\":\"milbrandt\"") != std::string::npos);
    REQUIRE(json.find("\"morrison\"") == std::string::npos);
    REQUIRE(json.find("\"runtimeSelectable\":false") == std::string::npos);
    REQUIRE(json.find("\"displayName\":\"Milbrandt-Yau (double moment)\"") != std::string::npos);
    REQUIRE(json.find("\"path\":\"microphysics.shared.cloud_water_threshold_kgkg\"") != std::string::npos);
    REQUIRE(json.find("\"path\":\"microphysics.milbrandt.triple_moment\"") == std::string::npos);
    REQUIRE(json.find("\"path\":\"microphysics.thompson.evaporation_rate_s_inv\"") != std::string::npos);
    REQUIRE(json.find("\"path\":\"microphysics.thompson.water_friendly_aerosol_surface_m3\"")
            != std::string::npos);
    REQUIRE(json.find("\"path\":\"microphysics.thompson.ice_friendly_aerosol_surface_m3\"")
            != std::string::npos);
    REQUIRE(json.find("\"path\":\"microphysics.milbrandt.evaporation_rate_s_inv\"") == std::string::npos);
    REQUIRE(json.find("\"path\":\"microphysics.milbrandt.hail_processes\"") != std::string::npos);
#ifdef HAVE_NETCDF
    REQUIRE(json.find("\"id\":\"netcdf\",\"available\":true,\"reason\":\"\",\"implementation\":\"libnetcdf\"") !=
            std::string::npos);
#else
    REQUIRE(json.find("\"id\":\"netcdf\",\"available\":false") != std::string::npos);
    REQUIRE(json.find("built without libnetcdf history-output support") != std::string::npos);
#endif
    REQUIRE(json.back() == '\n');
}

TEST_CASE("retained Milbrandt triple-moment readiness has an engineering blocker, not campaign evidence",
          "[core][runtime][scheme-catalog][milbrandt-triple]")
{
    const auto& items = tmv::runtime::tracked_readiness_items();
    const auto found = std::find_if(items.begin(), items.end(), [](const auto& item)
    {
        return item.id == "microphysics.milbrandt_triple_moment";
    });
    REQUIRE(found != items.end());
    CHECK(found->state == tmv::runtime::ReadinessState::EngineeringIncomplete);
    CHECK(found->campaign_evidence.empty());
    CHECK(found->external_review_evidence.empty());
    CHECK(found->blocker.find("Five reflectivity process") != std::string_view::npos);
}

TEST_CASE("generated Metal output scope seals ZFP and per-format write cadence",
          "[core][runtime][scheme-catalog][generated-metal-zfp]")
{
    using namespace tmv::runtime;
    REQUIRE(admits_generated_metal_output_write_cadence("npy_3d", 1));
    REQUIRE(admits_generated_metal_output_write_cadence("npy_3d", 301));
    REQUIRE(admits_generated_metal_output_write_cadence("netcdf", 1));
    REQUIRE(admits_generated_metal_output_write_cadence("netcdf", 301));
    REQUIRE(admits_generated_metal_output_write_cadence("zfp_high", 1));
    REQUIRE_FALSE(admits_generated_metal_output_write_cadence("zfp_high", 0));
    REQUIRE_FALSE(admits_generated_metal_output_write_cadence("zfp_high", 301));
    REQUIRE_FALSE(admits_generated_metal_output_write_cadence("zfp_medium", 1));
    REQUIRE_FALSE(admits_generated_metal_output_write_cadence("npy_3d", 2));
    const std::string json = tmv::runtime::serialize_scheme_catalog_json();
#ifdef HAVE_ZFP
    REQUIRE(json.find("\"generatedMetalCanonicalScope\":{\"version\":4,\"backend\":\"metal\",") !=
            std::string::npos);
    REQUIRE(json.find("\"tupleCount\":6049,\"distinctConfigCount\":4537") != std::string::npos);
    REQUIRE(json.find("\"outputFormats\":[\"npy_3d\",\"netcdf\",\"zfp_high\"]") !=
            std::string::npos);
    REQUIRE(json.find("\"writeCadenceSecondsByFormat\":{\"npy_3d\":[1,301],\"netcdf\":[1,301],"
                      "\"zfp_high\":[1]}") != std::string::npos);
    REQUIRE(json.find("\"zfpTypedOutputManifestSha256\":\"" +
                      std::string(kGeneratedMetalZfpTypedScopeManifestSha256) + "\"") !=
            std::string::npos);
    REQUIRE(json.find("\"manifestSha256\":\"" +
                      std::string(kGeneratedMetalOutputScopeV3ManifestSha256) + "\"") !=
            std::string::npos);
#else
    REQUIRE(json.find("\"generatedMetalCanonicalScope\":{\"version\":3,\"backend\":\"metal\",") !=
            std::string::npos);
    REQUIRE(json.find("\"tupleCount\":4033,\"distinctConfigCount\":3025") != std::string::npos);
    REQUIRE(json.find("\"manifestSha256\":\"" +
                      std::string(kGeneratedMetalOutputScopeNoZfpV2ManifestSha256) + "\"") !=
            std::string::npos);
#endif
    REQUIRE(json.find("\"additionalBoundaryNpyScope\":{\"version\":1,\"profileCount\":1,") !=
            std::string::npos);
    REQUIRE(json.find("\"familyId\":\"" + std::string(kGeneratedMetalEastNpyFamilyId) + "\"") !=
            std::string::npos);
    REQUIRE(json.find("\"strictCanonicalYamlSha256\":\"" +
                      std::string(kGeneratedMetalEastNpyCanonicalYamlSha256) + "\"") !=
            std::string::npos);
    REQUIRE(json.find("\"outputFormats\":[\"npy_3d\"],\"writeCadenceSeconds\":[1],") !=
            std::string::npos);
    REQUIRE(json.find("\"manifestSha256\":\"" +
                      std::string(kGeneratedMetalEastNpyScopeManifestSha256) + "\"") !=
            std::string::npos);
}

TEST_CASE("mixed-open Metal catalog scope is independent exact and names its host owners",
          "[core][runtime][scheme-catalog][flat-mixed-metal]")
{
    const std::string json = tmv::runtime::serialize_scheme_catalog_json();
    const auto scope = tmv::runtime::kFlatMixedMetalCadence1ScopeJson;
    const auto expected = "\"qualifiedFlatMixedMetalNpyScope\":" +
        std::string(scope.substr(0, scope.size() - 1)) + ",\"manifestSha256\":\"" +
        std::string(tmv::runtime::kFlatMixedMetalCadence1ScopeManifestSha256) + "\"}";
    REQUIRE(json.find(expected) != std::string::npos);
    REQUIRE(json.find("\"scalarTransportOwner\":\"metal_native_conservative_mc\"") != std::string::npos);
    REQUIRE(json.find("\"initialConditionChaosOwner\":\"host\",\"lateralDampingOwner\":\"host\"") !=
        std::string::npos);
    REQUIRE(json.find("\"effectiveWOnlySponge\":false") != std::string::npos);
    std::ifstream fixture("tests/fixtures/reference/catalog/flat_mixed_metal_cadence1_scope_v1.json");
    REQUIRE(fixture.good());
    const std::string source((std::istreambuf_iterator<char>(fixture)), std::istreambuf_iterator<char>());
    REQUIRE(nimbus::crypto::sha256_hex(source) ==
        tmv::runtime::kFlatMixedMetalCadence1ScopeManifestSha256);
}

TEST_CASE("Lin Thompson and Milbrandt publish collocated CPU support",
          "[core][runtime][scheme-catalog][microphysics][agrid]")
{
    const auto& schemes = tmv::get_microphysics_scheme_descriptors();
    const auto* milbrandt = tmv::find_scheme_descriptor(schemes, "milbrandt");
    const auto* lin = tmv::find_scheme_descriptor(schemes, "lin");
    const auto* thompson = tmv::find_scheme_descriptor(schemes, "thompson");
    REQUIRE(milbrandt != nullptr);
    REQUIRE(lin != nullptr);
    REQUIRE(thompson != nullptr);
    REQUIRE(std::find(
        milbrandt->capabilities.staggerings.begin(),
        milbrandt->capabilities.staggerings.end(), "collocated") !=
        milbrandt->capabilities.staggerings.end());
    REQUIRE(milbrandt->capabilities.backends == std::vector<std::string>{"cpu", "gpu"});
    REQUIRE(std::find(
        lin->capabilities.staggerings.begin(),
        lin->capabilities.staggerings.end(), "collocated") !=
        lin->capabilities.staggerings.end());
    REQUIRE(std::find(
        thompson->capabilities.staggerings.begin(),
        thompson->capabilities.staggerings.end(), "collocated") !=
        thompson->capabilities.staggerings.end());
}

TEST_CASE("readiness registry enforces evidence progression", "[core][runtime][scheme-catalog][readiness]")
{
    using State       = tmv::runtime::ReadinessState;
    const auto& items = tmv::runtime::tracked_readiness_items();
    REQUIRE(items.size() >= 35);

    std::set<std::string_view> ids;
    std::size_t                incomplete         = 0;
    std::size_t                ready              = 0;
    std::size_t                campaign_validated = 0;
    std::size_t                fully_qualified    = 0;
    for (const auto& item : items)
    {
        INFO("readiness item=" << item.id);
        REQUIRE_FALSE(item.id.empty());
        REQUIRE_FALSE(item.area.empty());
        REQUIRE_FALSE(item.scope.empty());
        REQUIRE(ids.insert(item.id).second);
        REQUIRE(item.id.find(' ') == std::string_view::npos);

        if (item.state == State::EngineeringIncomplete)
        {
            ++incomplete;
            REQUIRE_FALSE(item.blocker.empty());
        }
        else
        {
            ++ready;
            REQUIRE(item.blocker.empty());
            REQUIRE_FALSE(item.bounded_evidence.empty());

            std::string evidence(item.bounded_evidence);
            std::size_t begin = 0;
            while (begin < evidence.size())
            {
                const std::size_t end = evidence.find(';', begin);
                std::string path = evidence.substr(begin, end - begin);
                const std::size_t first = path.find_first_not_of(" \t");
                const std::size_t last = path.find_last_not_of(" \t");
                REQUIRE(first != std::string::npos);
                path = path.substr(first, last - first + 1);
                INFO("readiness evidence path=" << path);
                REQUIRE(std::filesystem::exists(path));
                if (end == std::string::npos)
                    break;
                begin = end + 1;
            }
        }
        if (item.state == State::CampaignValidated || item.state == State::ExternallyReviewed ||
            item.state == State::FullyQualified)
        {
            ++campaign_validated;
            REQUIRE_FALSE(item.campaign_evidence.empty());
        }
        if (item.state == State::ExternallyReviewed || item.state == State::FullyQualified)
            REQUIRE_FALSE(item.external_review_evidence.empty());
        if (item.state == State::FullyQualified)
            ++fully_qualified;
    }

    REQUIRE(incomplete + ready == items.size());
    REQUIRE(ready > 0);
    REQUIRE(campaign_validated == 0);
    REQUIRE(fully_qualified == 0);
    REQUIRE(ids.count("agrid.complete_vulkan") == 1);
    REQUIRE(ids.count("agrid.complete_metal") == 1);
    REQUIRE(ids.count("terrain.earth_fixed_remap") == 1);
    REQUIRE(ids.count("terrain.earth_fixed_continuous_cpu") == 1);
    REQUIRE(ids.count("terrain.earth_fixed_native") == 1);
    REQUIRE(ids.count("physics.native_remaining") == 1);
    REQUIRE(ids.count("nesting.moist_two_way") == 1);
    REQUIRE(ids.count("turbulence.richardson_public") == 1);
    REQUIRE(ids.count("recycling.nested_grid") == 1);
    REQUIRE(ids.count("recycling.terrain") == 1);
    REQUIRE(ids.count("domain_motion.tracked_window") == 1);
    REQUIRE(ids.count("radiation.rrtmg") == 1);
    REQUIRE(ids.count("boundary_layer.mynn") == 1);
    REQUIRE(ids.count("boundary_layer.pbl_top_theta") == 1);
    REQUIRE(ids.count("boundary_layer.pbl_top_tke") == 1);
    REQUIRE(ids.count("boundary_layer.cloud_top_radiative_ysu") == 1);
    REQUIRE(ids.count("turbulence.user_filter_width") == 1);
    REQUIRE(ids.count("turbulence.dynamic_smagorinsky") == 1);
    REQUIRE(ids.count("turbulence.tke_stability_correction") == 1);
}

TEST_CASE("deferred and nonconstructible descriptors cannot advertise runtime selection",
          "[core][runtime][scheme-catalog][truth]")
{
    const std::vector<const std::vector<tmv::SchemeDescriptor>*> registries = {
        &tmv::get_dynamics_scheme_descriptors(),
        &tmv::get_advection_scheme_descriptors(),
        &tmv::get_diffusion_scheme_descriptors(),
        &tmv::get_time_stepping_scheme_descriptors(),
        &tmv::get_microphysics_scheme_descriptors(),
        &tmv::get_radiation_scheme_descriptors(),
        &tmv::get_boundary_layer_scheme_descriptors(),
        &tmv::get_turbulence_scheme_descriptors(),
        &tmv::chaos::get_chaos_scheme_descriptors(),
        &tmv::get_terrain_scheme_descriptors(),
        &tmv::get_radar_scheme_descriptors(),
        &tmv::get_sounding_scheme_descriptors(),
    };
    for (const auto* registry : registries)
        for (const auto& descriptor : *registry)
        {
            INFO("scheme=" << descriptor.id);
            if (!descriptor.constructible || descriptor.maturity == tmv::SchemeMaturity::Deferred)
                REQUIRE_FALSE(descriptor.runtime_selectable);
        }
}

TEST_CASE("chaos catalog identities have one canonical public target contract",
          "[core][runtime][scheme-catalog][chaos][config]")
{
    const auto& descriptors = tmv::chaos::get_chaos_scheme_descriptors();
    REQUIRE(descriptors.size() == 4);
    for (const auto& descriptor : descriptors)
    {
        CAPTURE(descriptor.id);
        const auto& initial =
            tmv::chaos::canonical_chaos_initial_condition_targets(
                descriptor.id);
        const auto& tendencies =
            tmv::chaos::canonical_chaos_tendency_targets(descriptor.id);
        REQUIRE(tmv::chaos::chaos_targets_are_canonical_or_implicit(
            descriptor.id, {}, {}));
        REQUIRE(tmv::chaos::chaos_targets_are_canonical_or_implicit(
            descriptor.id, initial, tendencies));
        if (descriptor.id == "none")
        {
            REQUIRE(initial.empty());
            REQUIRE(tendencies.empty());
        }
        else if (descriptor.id == "initial_conditions")
        {
            REQUIRE(initial ==
                    std::vector<std::string>{"u", "v", "w", "theta"});
            REQUIRE(tendencies.empty());
        }
        else if (descriptor.id == "boundary_layer")
        {
            REQUIRE(initial.empty());
            REQUIRE(tendencies == std::vector<std::string>{"pbl"});
        }
        else if (descriptor.id == "full_stochastic")
        {
            REQUIRE(initial ==
                    std::vector<std::string>{"u", "v", "w", "theta"});
            REQUIRE(tendencies == std::vector<std::string>{
                "microphysics", "pbl", "diffusion"});
        }
        else
        {
            FAIL("unhandled public chaos descriptor");
        }
    }
}

TEST_CASE("science profile rejects unavailable and immature selections", "[core][runtime][science-profile]")
{
    using tmv::runtime::SelectedScheme;

    SECTION("qualified C-grid dynamics is accepted")
    {
        const std::vector<SelectedScheme> selected = {
            {"dynamics.scheme", tmv::kCartesianCGridDynamicsId, &tmv::get_dynamics_scheme_descriptors()},
            {"microphysics.scheme", "none", &tmv::get_microphysics_scheme_descriptors()},
            {"radiation.scheme", "none", &tmv::get_radiation_scheme_descriptors()},
            {"boundary_layer.scheme", "none", &tmv::get_boundary_layer_scheme_descriptors()},
            {"turbulence.scheme", "none", &tmv::get_turbulence_scheme_descriptors()},
            {"numerics.advection.scheme", "tvd", &tmv::get_advection_scheme_descriptors()},
            {"numerics.diffusion.scheme", "explicit", &tmv::get_diffusion_scheme_descriptors()},
            {"terrain.scheme", "none", &tmv::get_terrain_scheme_descriptors()},
            {"numerics.time_stepping.scheme", "split_explicit", &tmv::get_time_stepping_scheme_descriptors()},
            {"chaos.scheme", "none", &tmv::chaos::get_chaos_scheme_descriptors()},
        };
        REQUIRE(tmv::runtime::validate_science_profile(selected, true, "cartesian", "c_grid", "cpu").empty());
    }

    SECTION("A-grid flat CPU and native cores are qualified within their declared scope")
    {
        const std::vector<SelectedScheme> selected = {
            {"dynamics.scheme", tmv::kCartesianAGridDynamicsId, &tmv::get_dynamics_scheme_descriptors()},
        };
        REQUIRE(tmv::runtime::validate_science_profile(selected, false, "cartesian", "collocated", "cpu").empty());
        REQUIRE(tmv::runtime::validate_science_profile(selected, true, "cartesian", "collocated", "cpu").empty());
        REQUIRE(tmv::runtime::validate_science_profile(selected, false, "cartesian", "collocated", "gpu").empty());
        const auto gpu_errors =
            tmv::runtime::validate_science_profile(selected, true, "cartesian", "collocated", "gpu");
        REQUIRE(gpu_errors.empty());
        const auto* descriptor =
            tmv::find_scheme_descriptor(tmv::get_dynamics_scheme_descriptors(), tmv::kCartesianAGridDynamicsId);
        REQUIRE(descriptor != nullptr);
        REQUIRE(descriptor->capabilities.gpu_implementation_coverage == tmv::SchemeCoverage::Covered);
        REQUIRE(descriptor->capabilities.cpu_gpu_parity_coverage == tmv::SchemeCoverage::Covered);
    }

    SECTION("qualified Kessler exposes its runtime-scoped reference profile")
    {
        const std::vector<SelectedScheme> selected = {
            {"microphysics.scheme", "kessler", &tmv::get_microphysics_scheme_descriptors()},
        };
        const auto errors = tmv::runtime::validate_science_profile(selected, true, "cartesian", "c_grid", "cpu");
        REQUIRE(errors.empty());
        const auto* descriptor = tmv::find_scheme_descriptor(tmv::get_microphysics_scheme_descriptors(), "kessler");
        REQUIRE(descriptor != nullptr);
        REQUIRE(descriptor->caveat.find("exact admitted stationary and moving nested profiles") != std::string::npos);
    }

    SECTION("qualified TKE is restricted to its bounded reference profiles")
    {
        tmv::runtime::QualifiedTkeSelection selection{
            "tke",  "cpu",      "cartesian", "c_grid",   "3d",       "none",     "none",     "none",       "none",
            "none", "explicit", "scalars",   "periodic", "periodic", "periodic", "periodic", "rigid_slip", "rigid",
            false,  false,      false,       false,      false,      1.0 / 3.0,  1.0 / 3.0, "tke_based"};
        REQUIRE(tmv::runtime::is_exact_qualified_tke_profile(selection));
        std::vector<std::string> errors;
        tmv::runtime::append_qualified_tke_scope_error(errors, true, selection);
        REQUIRE(errors.empty());

        selection.stability_correction = "none";
        REQUIRE_FALSE(tmv::runtime::is_exact_qualified_tke_profile(selection));
        selection.stability_correction = "tke_based";

        selection.backend = "gpu";
        REQUIRE(tmv::runtime::is_exact_qualified_tke_profile(selection));
        tmv::runtime::append_qualified_tke_scope_error(errors, true, selection);
        REQUIRE(errors.empty());
        selection.backend = "metal";
        REQUIRE(tmv::runtime::is_exact_qualified_tke_profile(selection));

        selection.microphysics = "thompson";
        REQUIRE(tmv::runtime::is_exact_qualified_tke_profile(selection));
        selection.microphysics = "milbrandt";
        REQUIRE_FALSE(tmv::runtime::is_exact_qualified_tke_profile(selection));
        selection.microphysics = "none";

        selection.diffusion          = "implicit";
        selection.diffusion_target   = "scalars_and_momentum";
        selection.physical_diffusion = true;
        for (const auto* backend : {"cpu", "gpu", "metal"})
        {
            selection.backend = backend;
            REQUIRE(tmv::runtime::is_exact_qualified_tke_profile(selection));
        }
        selection.diffusion_target = "scalars";
        REQUIRE_FALSE(tmv::runtime::is_exact_qualified_tke_profile(selection));
        selection.diffusion_target   = "scalars_and_momentum";
        selection.physical_diffusion = false;
        REQUIRE_FALSE(tmv::runtime::is_exact_qualified_tke_profile(selection));
        selection.physical_diffusion = true;

        errors.clear();
        selection.backend        = "cpu";
        selection.lateral_sponge = true;
        REQUIRE_FALSE(tmv::runtime::is_exact_qualified_tke_profile(selection));
        tmv::runtime::append_qualified_tke_scope_error(errors, true, selection);
        REQUIRE(errors.size() == 1);
    }

    SECTION("qualified WENO5 is restricted to its coupled CPU reference profile")
    {
        tmv::runtime::QualifiedWeno5Selection selection{
            "weno5",      "cpu",      "cartesian", "c_grid",     "klemp_wilhelmson_split_cartesian_cgrid",
            "none",       "none",     "none",      "none",       "rrtmg",
            "none",       "none",     "explicit",  "scalars",    "periodic",
            "periodic",   "periodic", "periodic",  "rigid_slip", "rigid",
            "stationary", false,      false,       false,        false,
            false,        false};
        REQUIRE(tmv::runtime::is_exact_qualified_weno5_profile(selection));
        std::vector<std::string> errors;
        tmv::runtime::append_qualified_weno5_scope_error(errors, true, selection);
        REQUIRE(errors.empty());

        selection.radiation = "none";
        REQUIRE_FALSE(tmv::runtime::is_exact_qualified_weno5_profile(selection));
        tmv::runtime::append_qualified_weno5_scope_error(errors, true, selection);
        REQUIRE(errors.size() == 1);
        REQUIRE(errors.front().find("exact flat periodic C-grid CPU RRTMG") != std::string::npos);
    }

    SECTION("qualified implicit diffusion exposes its structural CPU envelope")
    {
        tmv::runtime::QualifiedImplicitDiffusionSelection selection{
            "implicit",   "cpu",      "cartesian", "c_grid",     "klemp_wilhelmson_split_cartesian_cgrid",
            "tvd",        "none",     "none",      "none",       "none",
            "rrtmg",      "none",     "none",      "scalars",    "periodic",
            "periodic",   "periodic", "periodic",  "rigid_slip", "rigid",
            "stationary", false,      false,       false,        false,
            false,        false,      0.0,         20.0};
        REQUIRE(tmv::runtime::is_exact_qualified_implicit_diffusion_profile(selection));
        REQUIRE(tmv::runtime::is_structurally_ready_implicit_diffusion_profile(selection));
        std::vector<std::string> errors;
        tmv::runtime::append_qualified_implicit_diffusion_scope_error(errors, true, selection);
        REQUIRE(errors.empty());

        selection.vertical_coefficient = 19.0;
        REQUIRE_FALSE(tmv::runtime::is_exact_qualified_implicit_diffusion_profile(selection));
        tmv::runtime::append_qualified_implicit_diffusion_scope_error(errors, true, selection);
        REQUIRE(errors.empty());
        selection.backend = "vulkan";
        REQUIRE_FALSE(tmv::runtime::is_structurally_ready_implicit_diffusion_profile(selection));
        tmv::runtime::append_qualified_implicit_diffusion_scope_error(errors, true, selection);
        REQUIRE(errors.size() == 1);
        REQUIRE(errors.front().find("Cartesian CPU") != std::string::npos);
        errors.clear();
        selection.backend = "cpu";
        selection.staggering = "collocated";
        selection.dynamics = "klemp_wilhelmson_split_cartesian_agrid";
        selection.target = "scalars_and_momentum";
        REQUIRE(tmv::runtime::is_structurally_ready_implicit_diffusion_profile(selection));
        tmv::runtime::append_qualified_implicit_diffusion_scope_error(
            errors, true, selection);
        REQUIRE(errors.empty());
    }

    SECTION("qualified Milbrandt and bell terrain expose their scoped CPU profiles")
    {
        const std::vector<SelectedScheme> milbrandt = {
            {"microphysics.scheme", "milbrandt", &tmv::get_microphysics_scheme_descriptors()},
        };
        REQUIRE(tmv::runtime::validate_science_profile(milbrandt, true, "cartesian", "c_grid", "cpu").empty());
        REQUIRE(
            tmv::runtime::validate_science_profile(milbrandt, true, "cartesian", "collocated", "cpu").empty());
        REQUIRE(tmv::runtime::validate_science_profile(milbrandt, false, "cartesian", "c_grid", "gpu").empty());

        const std::vector<SelectedScheme> terrain = {
            {"terrain.scheme", "bell", &tmv::get_terrain_scheme_descriptors()},
        };
        REQUIRE(tmv::runtime::validate_science_profile(terrain, false, "cartesian", "c_grid", "cpu").empty());
        REQUIRE(tmv::runtime::validate_science_profile(terrain, true, "cartesian", "c_grid", "cpu").empty());

        const std::vector<SelectedScheme> schar = {
            {"terrain.scheme", "schar", &tmv::get_terrain_scheme_descriptors()},
        };
        REQUIRE(tmv::runtime::validate_science_profile(schar, false, "cartesian", "c_grid", "cpu").empty());
        REQUIRE(tmv::runtime::validate_science_profile(schar, true, "cartesian", "c_grid", "cpu").empty());
    }

    SECTION("coupled Runge-Kutta classes publish their bounded flat CPU and native capability")
    {
        const std::vector<SelectedScheme> selected = {
            {"numerics.time_stepping.scheme", "rk3", &tmv::get_time_stepping_scheme_descriptors()},
        };
        REQUIRE(tmv::runtime::validate_science_profile(
                    selected, true, "cartesian", "c_grid", "cpu").empty());
        REQUIRE(tmv::runtime::validate_science_profile(
                    selected, true, "cartesian", "c_grid", "gpu").empty());
        REQUIRE(tmv::runtime::validate_science_profile(
                    selected, true, "cartesian", "collocated", "cpu").empty());
    }

    SECTION("qualified file readers are accepted for explicit file sources")
    {
        for (const std::string reader : {"auto", "cm1", "sharpy"})
        {
            const std::vector<SelectedScheme> selected = {
                {"environment.sounding.file.scheme_id", reader, &tmv::get_sounding_scheme_descriptors()},
            };
            REQUIRE(tmv::runtime::validate_science_profile(selected, true, "cartesian", "c_grid", "cpu").empty());
        }
    }

    SECTION("native radiation declares its column implementation")
    {
        const std::vector<SelectedScheme> selected = {
            {"radiation.scheme", "simple_grey", &tmv::get_radiation_scheme_descriptors()},
        };
        REQUIRE(tmv::runtime::validate_science_profile(selected, false, "cartesian", "c_grid", "gpu").empty());
        const auto* descriptor = tmv::find_scheme_descriptor(tmv::get_radiation_scheme_descriptors(), "simple_grey");
        REQUIRE(descriptor != nullptr);
        REQUIRE(descriptor->capabilities.gpu_implementation_coverage == tmv::SchemeCoverage::Covered);
        REQUIRE(descriptor->capabilities.cpu_gpu_parity_coverage == tmv::SchemeCoverage::Covered);
    }

    SECTION("PBL descriptors distinguish native and host column implementations")
    {
        for (const std::string scheme : {"slab", "ysu"})
        {
            const std::vector<SelectedScheme> selected = {
                {"boundary_layer.scheme", scheme, &tmv::get_boundary_layer_scheme_descriptors()},
            };
            REQUIRE(tmv::runtime::validate_science_profile(selected, false, "cartesian", "c_grid", "gpu").empty());
            const auto* descriptor = tmv::find_scheme_descriptor(tmv::get_boundary_layer_scheme_descriptors(), scheme);
            REQUIRE(descriptor != nullptr);
            REQUIRE(descriptor->capabilities.gpu_implementation_coverage == (scheme == "slab" ? tmv::SchemeCoverage::Covered : tmv::SchemeCoverage::None));
            REQUIRE(descriptor->capabilities.cpu_gpu_parity_coverage == (scheme == "slab" ? tmv::SchemeCoverage::Covered : tmv::SchemeCoverage::None));
        }
    }

    SECTION("host-owned chaos composes with a GPU-led profile honestly")
    {
        for (const std::string scheme : {"initial_conditions", "boundary_layer", "full_stochastic"})
        {
            const std::vector<SelectedScheme> selected = {
                {"chaos.scheme", scheme, &tmv::chaos::get_chaos_scheme_descriptors()},
            };
            REQUIRE(tmv::runtime::validate_science_profile(selected, false, "cartesian", "c_grid", "gpu").empty());
            const auto* descriptor = tmv::find_scheme_descriptor(tmv::chaos::get_chaos_scheme_descriptors(), scheme);
            REQUIRE(descriptor != nullptr);
            REQUIRE(descriptor->capabilities.gpu_implementation_coverage == tmv::SchemeCoverage::None);
            REQUIRE(descriptor->capabilities.cpu_gpu_parity_coverage == tmv::SchemeCoverage::None);
        }
    }

    SECTION("A-grid admits physical-domain initial and evolving chaos")
    {
        const auto validate = [](const std::string& scheme)
        {
            const std::vector<SelectedScheme> selected = {
                {"chaos.scheme", scheme, &tmv::chaos::get_chaos_scheme_descriptors()},
            };
            return tmv::runtime::validate_science_profile(selected, true, "cartesian", "collocated", "cpu");
        };
        REQUIRE(validate("initial_conditions").empty());
        REQUIRE(validate("boundary_layer").empty());
        REQUIRE(validate("full_stochastic").empty());
    }
}

TEST_CASE("physics interaction profile rejects overlapping vertical closures",
          "[core][runtime][science-profile][coupling]")
{
    using tmv::runtime::validate_physics_interactions;
    REQUIRE(validate_physics_interactions("slab", "smagorinsky", "2d").empty());
    REQUIRE(validate_physics_interactions("ysu", "smagorinsky", "horizontal_only").empty());
    REQUIRE(validate_physics_interactions("none", "smagorinsky", "3d").empty());
    REQUIRE(validate_physics_interactions("slab", "none", "3d").empty());

    for (const std::string& pbl : {"slab", "ysu", "mynn"})
    {
        for (const std::string& sgs : {"smagorinsky", "tke"})
        {
            const auto errors = validate_physics_interactions(pbl, sgs, "3d");
            REQUIRE(errors.size() == 1);
            REQUIRE(errors.front().find("both own vertical") != std::string::npos);
        }
    }
}

TEST_CASE("surface interaction profile preserves explicit operator ownership",
          "[core][runtime][science-profile][coupling][surface]")
{
    using tmv::runtime::validate_surface_interactions;
    REQUIRE(validate_surface_interactions("none", false, "none").empty());
    REQUIRE(validate_surface_interactions("none", false, "ysu").empty());
    REQUIRE(validate_surface_interactions("slab", false, "slab").empty());
    REQUIRE(validate_surface_interactions("slab", true, "ysu").empty());
    REQUIRE_FALSE(validate_surface_interactions("none", true, "slab").empty());
    REQUIRE_FALSE(validate_surface_interactions("slab", true, "none").empty());
    REQUIRE_FALSE(validate_surface_interactions("slab", false, "none").empty());
    REQUIRE_FALSE(validate_surface_interactions("unknown", false, "none").empty());
}

TEST_CASE("terrain interaction profile matches the mapped runtime scope",
          "[core][runtime][science-profile][terrain][coupling]")
{
    using tmv::runtime::validate_terrain_interactions;
    REQUIRE(validate_terrain_interactions(
                "none", "simple_grey", true, "ysu", "tke", "full_stochastic", "implicit", "all", true)
                .empty());
    REQUIRE(validate_terrain_interactions("bell", "none", false, "none", "none", "none", "explicit", "scalars", false)
                .empty());
    REQUIRE(validate_terrain_interactions(
                "bell", "simple_grey", false, "none", "smagorinsky", "none", "explicit", "scalars", true)
                .empty());

    REQUIRE(validate_terrain_interactions(
                "bell", "simple_grey", true, "none", "smagorinsky", "none", "explicit", "scalars", false)
                .empty());
    REQUIRE(validate_terrain_interactions(
                "bell", "rrtmg", false, "none", "smagorinsky", "none", "explicit", "scalars", false)
                .empty());
    REQUIRE(validate_terrain_interactions(
                "bell", "rrtmg", true, "none", "smagorinsky", "none", "explicit", "scalars", false)
                .empty());
    REQUIRE(validate_terrain_interactions(
                "bell", "simple_grey", false, "ysu", "smagorinsky", "none", "explicit", "scalars", false)
                .empty());
    REQUIRE(validate_terrain_interactions("bell", "none", false, "none", "tke", "none", "explicit", "scalars", false)
                .empty());
    REQUIRE(validate_terrain_interactions(
                "bell", "none", false, "none", "none", "full_stochastic", "explicit", "scalars", false)
                .empty());
    REQUIRE(validate_terrain_interactions(
                "bell", "none", false, "none", "none", "none", "explicit", "scalars_and_momentum", true)
                .empty());
    REQUIRE(validate_terrain_interactions("bell", "none", false, "none", "none", "none", "implicit", "scalars", true)
                .empty());
    REQUIRE(validate_terrain_interactions("bell", "none", false, "none", "none", "none", "implicit", "momentum", true)
                .empty());
}

TEST_CASE("aggregate science interaction profile owns every value-dependent gate",
          "[core][runtime][science-profile][coupling][aggregate][sci120]")
{
    using tmv::runtime::ScienceInteractionSelection;
    using tmv::runtime::validate_science_interactions;

    const ScienceInteractionSelection valid{
        "none",
        "smagorinsky",
        "3d",
        "none",
        false,
        "bell",
        "simple_grey",
        false,
        "none",
        "explicit",
        "scalars",
        true,
        "npy_3d",
        true,
    };
    REQUIRE(validate_science_interactions(valid).empty());

    auto invalid                        = valid;
    invalid.boundary_layer_id           = "slab";
    invalid.surface_id                  = "none";
    invalid.apply_surface_fluxes        = true;
    invalid.radiation_shortwave_enabled = true;
    invalid.chaos_id                    = "full_stochastic";
    invalid.diffusion_target            = "scalars_and_momentum";
    const auto errors                   = validate_science_interactions(invalid);
    REQUIRE(errors.size() == 2);
    REQUIRE(std::any_of(errors.begin(),
                        errors.end(),
                        [](const std::string& error) { return error.find("both own vertical") != std::string::npos; }));
    REQUIRE(std::any_of(errors.begin(),
                        errors.end(),
                        [](const std::string& error)
                        { return error.find("surface.apply_fluxes") != std::string::npos; }));
    REQUIRE(std::none_of(errors.begin(),
                         errors.end(),
                         [](const std::string& error) { return error.find("terrain=bell") != std::string::npos; }));

    auto qualified_zfp_terrain          = valid;
    qualified_zfp_terrain.output_format = "zfp_high";
    REQUIRE(validate_science_interactions(qualified_zfp_terrain).empty());
    qualified_zfp_terrain.require_qualified = false;
    REQUIRE(validate_science_interactions(qualified_zfp_terrain).empty());
}

TEST_CASE("generated GPU apply-control matrix is fail-closed", "[core][runtime][science-profile][gpu][cross-product]")
{
    using tmv::runtime::ComputeCommitOwner;
    using tmv::runtime::ScienceInteractionSelection;
    using tmv::runtime::resolve_compute_ownership;
    using tmv::runtime::validate_compute_interactions;

    const std::array<std::string, 2> backends = {"cpu", "gpu"};
    const std::array<std::string, 2> dynamics = {
        tmv::kCartesianCGridDynamicsId,
        tmv::kCartesianAGridDynamicsId,
    };
    const std::array<std::string, 5> microphysics = {
        "none",
        "kessler",
        "lin",
        "thompson",
        "milbrandt",
    };
    const std::array<std::string, 3> turbulence = {
        "none",
        "smagorinsky",
        "tke",
    };

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

                                    const auto ownership = resolve_compute_ownership(selection);
                                    const auto errors    = validate_compute_interactions(selection);
                                    INFO("backend="
                                         << backend << " qualified=" << qualified << " dynamics=" << dynamics_id
                                         << " microphysics=" << microphysics_id << " turbulence=" << turbulence_id
                                         << " apply=" << apply_microphysics << '/' << apply_slow << '/' << apply_sgs);
                                    REQUIRE(errors == ownership.errors);
                                    if (ownership.errors.empty())
                                        ++supported;
                                    else
                                        ++unsupported;
                                    ++evaluated;
                                }
    REQUIRE(evaluated == 2 * 2 * 2 * 5 * 3 * 2 * 2 * 2);
    REQUIRE(supported == 960);
    REQUIRE(unsupported == 0);

    ScienceInteractionSelection device_owned;
    device_owned.backend                = "gpu";
    device_owned.dynamics_id            = tmv::kCartesianAGridDynamicsId;
    device_owned.microphysics_id        = "kessler";
    device_owned.turbulence_id          = "smagorinsky";
    device_owned.gpu_apply_microphysics = true;
    device_owned.gpu_apply_slow         = true;
    device_owned.gpu_apply_sgs          = true;
    const auto device_decision = resolve_compute_ownership(device_owned);
    REQUIRE(device_decision.errors.empty());
    REQUIRE(device_decision.microphysics_state_commit == ComputeCommitOwner::Device);
    REQUIRE(device_decision.slow_state_commit == ComputeCommitOwner::Device);
    REQUIRE(device_decision.sgs_state_commit == ComputeCommitOwner::Device);

    auto host_owned                     = device_owned;
    host_owned.backend                  = "cpu";
    host_owned.microphysics_id          = "thompson";
    host_owned.turbulence_id            = "tke";
    const auto host_decision = resolve_compute_ownership(host_owned);
    REQUIRE(host_decision.errors.empty());
    REQUIRE(host_decision.microphysics_state_commit == ComputeCommitOwner::Host);
    REQUIRE(host_decision.slow_state_commit == ComputeCommitOwner::Host);
    REQUIRE(host_decision.sgs_state_commit == ComputeCommitOwner::Host);

    auto rejected                = device_owned;
    rejected.microphysics_id     = "unknown_microphysics";
    rejected.turbulence_id       = "tke";
    const auto rejected_decision = resolve_compute_ownership(rejected);
    REQUIRE(rejected_decision.errors.size() == 1);
    REQUIRE(rejected_decision.microphysics_state_commit == ComputeCommitOwner::Unsupported);
    REQUIRE(rejected_decision.slow_state_commit == ComputeCommitOwner::Device);
    REQUIRE(rejected_decision.sgs_state_commit == ComputeCommitOwner::Device);

    ScienceInteractionSelection unknown;
    unknown.backend = "metal";
    const auto unknown_decision = resolve_compute_ownership(unknown);
    REQUIRE(unknown_decision.errors.size() == 1);
    REQUIRE(unknown_decision.microphysics_state_commit == ComputeCommitOwner::Inactive);
    REQUIRE(unknown_decision.slow_state_commit == ComputeCommitOwner::Unsupported);
    REQUIRE(unknown_decision.sgs_state_commit == ComputeCommitOwner::Inactive);
}

TEST_CASE("native compute ownership matrix is complete and machine readable",
          "[core][runtime][science-profile][gpu][ownership]")
{
    const std::string json = tmv::runtime::serialize_compute_ownership_matrix_json();
    REQUIRE(json.find("\"schema\":\"nimbus.compute-ownership-matrix.v1\"") != std::string::npos);
    REQUIRE(json.find("\"scope\":\"state_commit_controls\"") != std::string::npos);
    REQUIRE(json.find("\"summary\":{\"evaluated\":960,\"supported\":960,\"unsupported\":0}") != std::string::npos);
    REQUIRE(json.find("\"microphysicsStateCommit\":\"device\"") != std::string::npos);
    REQUIRE(json.find("\"slowStateCommit\":\"host\"") != std::string::npos);
    REQUIRE(json.find("\"sgsStateCommit\":\"unsupported\"") == std::string::npos);

    std::size_t records = 0;
    std::size_t offset  = 0;
    while ((offset = json.find("\"index\":", offset)) != std::string::npos)
    {
        ++records;
        offset += 8;
    }
    REQUIRE(records == 960);
}

TEST_CASE("native solver process ownership matrix is descriptor exact",
          "[core][runtime][science-profile][gpu][process-ownership]")
{
    const std::string json = tmv::runtime::serialize_compute_process_ownership_matrix_json();
    REQUIRE(json.find("\"schema\":\"nimbus.compute-process-ownership-matrix.v1\"") != std::string::npos);
    REQUIRE(json.find("\"scope\":\"descriptor_backed_solver_processes\"") != std::string::npos);
    REQUIRE(json.find("\"summary\":{\"evaluated\":64,\"supported\":64,\"unsupported\":0}") != std::string::npos);
    REQUIRE(json.find("\"module\":\"radiation\",\"scheme\":\"rrtmg\",\"backend\":\"gpu\",\"owner\":\"device\"") !=
            std::string::npos);
    REQUIRE(json.find("\"module\":\"radiation\",\"scheme\":\"simple_grey\",\"backend\":\"gpu\",\"owner\":\"device\"") !=
            std::string::npos);
    REQUIRE(json.find("\"module\":\"microphysics\",\"scheme\":\"thompson\",\"backend\":\"gpu\",\"owner\":\"device\"") !=
            std::string::npos);
    REQUIRE(json.find("\"module\":\"terrain\",\"scheme\":\"none\",\"backend\":\"gpu\",\"owner\":\"inactive\"") !=
            std::string::npos);

    std::size_t records = 0;
    std::size_t offset  = 0;
    while ((offset = json.find("\"index\":", offset)) != std::string::npos)
    {
        ++records;
        offset += 8;
    }
    REQUIRE(records == 64);
}

TEST_CASE("solver backend admission excludes host preflight inputs",
          "[core][runtime][science-profile][gpu][process-ownership]")
{
    using tmv::runtime::SelectedScheme;
    const auto& terrain = tmv::get_terrain_scheme_descriptors();
    const auto& soundings = tmv::get_sounding_scheme_descriptors();

    const auto terrain_errors = tmv::runtime::validate_science_profile(
        {SelectedScheme{"terrain.scheme", "schar", &terrain}},
        false,
        "cartesian",
        "c_grid",
        "gpu");
    REQUIRE(terrain_errors.empty());

    const auto sounding_errors = tmv::runtime::validate_science_profile(
        {SelectedScheme{"environment.sounding.file.scheme_id", "auto", &soundings}},
        false,
        "cartesian",
        "c_grid",
        "gpu");
    REQUIRE(sounding_errors.empty());
}

TEST_CASE("generated boundary interaction matrix owns topology and staggering",
          "[core][runtime][science-profile][boundary][cross-product][sci137]")
{
    using tmv::runtime::ScienceInteractionPolicy;
    using tmv::runtime::validate_boundary_interactions;

    std::size_t cases = 0;
    const auto& modes = ScienceInteractionPolicy::lateral_boundary_modes();
    for (const std::string& staggering : std::array<std::string, 2>{"c_grid", "collocated"})
        for (const auto& west : modes)
            for (const auto& east : modes)
                for (const auto& south : modes)
                    for (const auto& north : modes)
                    {
                        const auto errors = validate_boundary_interactions(
                            staggering, west, east, south, north, "rigid_noslip", "rigid_with_sponge");
                        const bool radiative_on_collocated =
                            staggering == "collocated" && (west == "radiative" || east == "radiative" ||
                                                           south == "radiative" || north == "radiative");
                        const bool unpaired_periodic = (west == "periodic") != (east == "periodic") ||
                                                       (south == "periodic") != (north == "periodic");
                        INFO("staggering=" << staggering << " west=" << west << " east=" << east << " south=" << south
                                           << " north=" << north);
                        REQUIRE(errors.empty() == (!radiative_on_collocated && !unpaired_periodic));
                        ++cases;
                    }
    REQUIRE(cases == 2 * 6 * 6 * 6 * 6);

    REQUIRE(tmv::runtime::validate_diagnostic_interactions(-1).empty());
    REQUIRE(tmv::runtime::validate_diagnostic_interactions(0).empty());
    REQUIRE(tmv::runtime::validate_diagnostic_interactions(17).empty());
    REQUIRE(tmv::runtime::validate_diagnostic_interactions(-2).size() == 1);
    REQUIRE(validate_boundary_interactions(
                "c_grid", "zero_gradient", "zero_gradient", "rigid_slip", "rigid_slip", "open", "periodic")
                .size() == 2);
}

TEST_CASE("generated collocated physics cross-product is fail-closed",
          "[core][runtime][science-profile][agrid][cross-product]")
{
    using tmv::runtime::SelectedScheme;
    const auto& microphysics   = tmv::get_microphysics_scheme_descriptors();
    const auto& radiation      = tmv::get_radiation_scheme_descriptors();
    const auto& boundary_layer = tmv::get_boundary_layer_scheme_descriptors();
    const auto& turbulence     = tmv::get_turbulence_scheme_descriptors();
    const auto& chaos          = tmv::chaos::get_chaos_scheme_descriptors();
    const auto& terrain        = tmv::get_terrain_scheme_descriptors();
    const auto& advection      = tmv::get_advection_scheme_descriptors();
    const auto& dynamics       = tmv::get_dynamics_scheme_descriptors();
    const auto& diffusion      = tmv::get_diffusion_scheme_descriptors();
    const auto& time_stepping  = tmv::get_time_stepping_scheme_descriptors();
    const auto* agrid_dynamics = tmv::find_scheme_descriptor(dynamics, tmv::kCartesianAGridDynamicsId);
    const auto* explicit_diffusion = tmv::find_scheme_descriptor(diffusion, "explicit");
    const auto* split_explicit = tmv::find_scheme_descriptor(time_stepping, "split_explicit");
    REQUIRE(agrid_dynamics != nullptr);
    REQUIRE(explicit_diffusion != nullptr);
    REQUIRE(split_explicit != nullptr);
    const std::array<std::string, 2> backends = {"cpu", "gpu"};
    const std::array<std::string, 3> turbulence_modes = {"2d", "vertical_only", "3d"};
    const auto supports = [](const std::vector<std::string>& values, const std::string& requested)
    {
        return values.end() != std::find(values.begin(), values.end(), requested) ||
               values.end() != std::find(values.begin(), values.end(), "any") ||
               values.end() != std::find(values.begin(), values.end(), "not_applicable");
    };
    const auto accepts_backend = [&](const tmv::SchemeDescriptor& descriptor, const std::string& backend)
    {
        const bool active = !descriptor.capabilities.required_state_fields.empty() ||
                            !descriptor.capabilities.produced_state_fields.empty();
        return !active || supports(descriptor.capabilities.backends, backend);
    };

    std::size_t accepted = 0;
    std::size_t rejected = 0;
    for (const auto& mp : microphysics)
        for (const auto& rad : radiation)
            for (const auto& pbl : boundary_layer)
                for (const auto& sgs : turbulence)
                    for (const auto& stochastic : chaos)
                        for (const auto& terrain_scheme : terrain)
                            for (const auto& transport : advection)
                                for (const auto& backend : backends)
                                    for (const auto& mode : turbulence_modes)
                        {
                            const std::vector<SelectedScheme> selections = {
                                {"microphysics.scheme", mp.id, &microphysics},
                                {"radiation.scheme", rad.id, &radiation},
                                {"boundary_layer.scheme", pbl.id, &boundary_layer},
                                {"turbulence.scheme", sgs.id, &turbulence},
                                {"chaos.scheme", stochastic.id, &chaos},
                                {"terrain.scheme", terrain_scheme.id, &terrain},
                                {"advection.scheme", transport.id, &advection},
                                {"dynamics.scheme", agrid_dynamics->id, &dynamics},
                                {"diffusion.scheme", explicit_diffusion->id, &diffusion},
                                {"time.integrator", split_explicit->id, &time_stepping},
                            };
                            auto errors = tmv::runtime::validate_science_profile(
                                selections, false, "cartesian", "collocated", backend);
                            tmv::runtime::ScienceInteractionSelection interaction;
                            interaction.boundary_layer_id = pbl.id;
                            interaction.turbulence_id = sgs.id;
                            interaction.turbulence_mode = mode;
                            interaction.surface_id = "none";
                            interaction.terrain_id = terrain_scheme.id;
                            interaction.radiation_id = rad.id;
                            interaction.chaos_id = stochastic.id;
                            interaction.diffusion_id = explicit_diffusion->id;
                            interaction.diffusion_target = "scalars";
                            interaction.output_format = "npy_3d";
                            interaction.staggering = "collocated";
                            interaction.lateral_west = "rigid";
                            interaction.lateral_east = "rigid";
                            interaction.lateral_south = "rigid";
                            interaction.lateral_north = "rigid";
                            interaction.bottom_boundary = "rigid_slip";
                            interaction.top_boundary = "rigid";
                            interaction.backend = backend;
                            interaction.dynamics_id = tmv::kCartesianAGridDynamicsId;
                            interaction.microphysics_id = mp.id;
                            const auto interaction_errors =
                                tmv::runtime::validate_science_interactions(interaction);
                            errors.insert(errors.end(), interaction_errors.begin(), interaction_errors.end());
                            const bool flat_supported =
                                                   (mp.id == "none" || mp.id == "kessler" ||
                                                    mp.id == "lin" || mp.id == "thompson" ||
                                                    mp.id == "milbrandt") &&
                                                   (sgs.id == "none" || sgs.id == "smagorinsky" || sgs.id == "tke") &&
                                                   terrain_scheme.id == "none" &&
                                                   // Radiation operates on scalar thermodynamic columns. Both
                                                   // implemented schemes therefore support the collocated path;
                                                   // maturity still keeps RRTMG exploratory rather than qualified.
                                                   (rad.id == "none" || rad.id == "simple_grey" || rad.id == "rrtmg");
                            const bool mapped_dry_supported =
                                mp.id == "none" && rad.id == "none" &&
                                pbl.id == "none" && sgs.id == "none" &&
                                stochastic.id == "none" &&
                                (terrain_scheme.id == "bell" ||
                                 terrain_scheme.id == "schar");
                            const bool backend_supported = std::all_of(
                                selections.begin(),
                                selections.end(),
                                [&](const SelectedScheme& selection)
                                {
                                    const auto* descriptor = tmv::find_scheme_descriptor(
                                        *selection.descriptors, selection.id);
                                    return descriptor != nullptr && accepts_backend(*descriptor, backend);
                                });
                            const bool supported = (flat_supported || mapped_dry_supported) && backend_supported &&
                                                   interaction_errors.empty();
                            if (supported)
                            {
                                INFO("unexpected rejection for radiation=" << rad.id << " backend=" << backend);
                                REQUIRE(errors.empty());
                                const auto commit_ownership = tmv::runtime::resolve_compute_ownership(interaction);
                                REQUIRE(commit_ownership.errors.empty());
                                REQUIRE(commit_ownership.microphysics_state_commit !=
                                        tmv::runtime::ComputeCommitOwner::Unsupported);
                                REQUIRE(commit_ownership.slow_state_commit !=
                                        tmv::runtime::ComputeCommitOwner::Unsupported);
                                REQUIRE(commit_ownership.sgs_state_commit !=
                                        tmv::runtime::ComputeCommitOwner::Unsupported);
                                for (const auto& selection : selections)
                                {
                                    REQUIRE(selection.descriptors != nullptr);
                                    const auto* descriptor =
                                        tmv::find_scheme_descriptor(*selection.descriptors, selection.id);
                                    REQUIRE(descriptor != nullptr);
                                    REQUIRE(tmv::runtime::resolve_compute_process_owner(*descriptor, backend) !=
                                            tmv::runtime::ComputeCommitOwner::Unsupported);
                                }
                                ++accepted;
                            }
                            else
                            {
                                INFO("unexpected acceptance for mp=" << mp.id << " rad=" << rad.id << " pbl=" << pbl.id
                                                                     << " sgs=" << sgs.id << " chaos=" << stochastic.id
                                                                     << " terrain=" << terrain_scheme.id
                                                                     << " backend=" << backend);
                                REQUIRE_FALSE(errors.empty());
                                ++rejected;
                            }
                        }
    REQUIRE(accepted == 8676);
    REQUIRE(rejected == microphysics.size() * radiation.size() * boundary_layer.size() * turbulence.size() *
                                chaos.size() * terrain.size() * advection.size() * backends.size() *
                                turbulence_modes.size() -
                            accepted);
}

TEST_CASE("generated C-grid engine cross-product matches the shared capability contract",
          "[core][runtime][science-profile][cgrid][cross-product]")
{
    using tmv::runtime::SelectedScheme;
    const auto& microphysics   = tmv::get_microphysics_scheme_descriptors();
    const auto& radiation      = tmv::get_radiation_scheme_descriptors();
    const auto& boundary_layer = tmv::get_boundary_layer_scheme_descriptors();
    const auto& turbulence     = tmv::get_turbulence_scheme_descriptors();
    const auto& chaos          = tmv::chaos::get_chaos_scheme_descriptors();
    const auto& diffusion      = tmv::get_diffusion_scheme_descriptors();
    const auto& terrain        = tmv::get_terrain_scheme_descriptors();
    const auto& soundings      = tmv::get_sounding_scheme_descriptors();
    const auto& dynamics       = tmv::get_dynamics_scheme_descriptors();
    const auto& advection      = tmv::get_advection_scheme_descriptors();
    const auto& time_stepping  = tmv::get_time_stepping_scheme_descriptors();
    const auto* automatic      = tmv::find_scheme_descriptor(soundings, "auto");
    const auto* cm1            = tmv::find_scheme_descriptor(soundings, "cm1");
    const auto* sharpy         = tmv::find_scheme_descriptor(soundings, "sharpy");
    REQUIRE(automatic != nullptr);
    REQUIRE(cm1 != nullptr);
    REQUIRE(sharpy != nullptr);
    const auto* cgrid_dynamics = tmv::find_scheme_descriptor(dynamics, tmv::kCartesianCGridDynamicsId);
    const auto* tvd             = tmv::find_scheme_descriptor(advection, "tvd");
    const auto* split_explicit  = tmv::find_scheme_descriptor(time_stepping, "split_explicit");
    REQUIRE(cgrid_dynamics != nullptr);
    REQUIRE(tvd != nullptr);
    REQUIRE(split_explicit != nullptr);
    const std::array<std::string, 2>                  backends         = {"cpu", "gpu"};
    const std::array<std::string, 3>                  turbulence_modes = {"2d", "vertical_only", "3d"};
    const std::array<std::pair<std::string, bool>, 4> surface_profiles = {{
        {"none", false},
        {"none", true},
        {"slab", false},
        {"slab", true},
    }};
    const std::array<std::string, 2>                  output_formats   = {
        "npy_3d",
        "zfp_high",
    };
    // null means an in-process parametric source with no external reader.
    // Every fixed/explicit file reader participates in the same generated
    // capability cross-product.
    const std::array<const tmv::SchemeDescriptor*, 4> initialization_profiles = {
        nullptr,
        automatic,
        cm1,
        sharpy,
    };

    const auto supports = [](const std::vector<std::string>& values, const std::string& requested)
    {
        return values.end() != std::find(values.begin(), values.end(), requested) ||
               values.end() != std::find(values.begin(), values.end(), "any") ||
               values.end() != std::find(values.begin(), values.end(), "not_applicable");
    };
    const auto descriptor_accepts =
        [&](const tmv::SchemeDescriptor& descriptor,
            bool                         qualified,
            const std::string&           backend,
            bool                         host_preflight_input)
    {
        if (!descriptor.constructible || !descriptor.runtime_selectable)
            return false;
        if (qualified && descriptor.maturity != tmv::SchemeMaturity::Qualified)
            return false;
        if (!supports(descriptor.capabilities.coordinate_systems, "cartesian") ||
            !supports(descriptor.capabilities.staggerings, "c_grid"))
            return false;
        const bool active = !descriptor.capabilities.required_state_fields.empty() ||
                            !descriptor.capabilities.produced_state_fields.empty();
        return !active || host_preflight_input || supports(descriptor.capabilities.backends, backend);
    };

    std::size_t evaluated = 0;
    std::size_t accepted  = 0;
    for (const auto& mp : microphysics)
        for (const auto& rad : radiation)
            for (const auto& pbl : boundary_layer)
                for (const auto& sgs : turbulence)
                    for (const auto& stochastic : chaos)
                        for (const auto& diff : diffusion)
                            for (const auto& terrain_scheme : terrain)
                                for (const auto& backend : backends)
                                    for (const bool qualified : {false, true})
                                        for (const auto& mode : turbulence_modes)
                                            for (const auto& surface : surface_profiles)
                                                for (const auto* sounding : initialization_profiles)
                                                    for (const auto& output_format : output_formats)
                                                    {
                                                        std::vector<SelectedScheme> selections = {
                                                            {"microphysics.scheme", mp.id, &microphysics},
                                                            {"radiation.scheme", rad.id, &radiation},
                                                            {"boundary_layer.scheme", pbl.id, &boundary_layer},
                                                            {"turbulence.scheme", sgs.id, &turbulence},
                                                            {"chaos.scheme", stochastic.id, &chaos},
                                                            {"numerics.diffusion.scheme", diff.id, &diffusion},
                                                            {"terrain.scheme", terrain_scheme.id, &terrain},
                                                            {"dynamics.scheme", cgrid_dynamics->id, &dynamics},
                                                            {"advection.scheme", tvd->id, &advection},
                                                            {"time.integrator", split_explicit->id, &time_stepping},
                                                        };
                                                        if (sounding != nullptr)
                                                        {
                                                            selections.push_back({
                                                                "environment.sounding.file.scheme_id",
                                                                sounding->id,
                                                                &soundings,
                                                            });
                                                        }
                                                        auto errors = tmv::runtime::validate_science_profile(
                                                            selections, qualified, "cartesian", "c_grid", backend);
                                                        const tmv::runtime::ScienceInteractionSelection interactions{
                                                            pbl.id,
                                                            sgs.id,
                                                            mode,
                                                            surface.first,
                                                            surface.second,
                                                            terrain_scheme.id,
                                                            rad.id,
                                                            false,
                                                            stochastic.id,
                                                            diff.id,
                                                            "scalars",
                                                            true,
                                                            output_format,
                                                            qualified,
                                                        };
                                                        auto ownership_interactions = interactions;
                                                        ownership_interactions.backend = backend;
                                                        ownership_interactions.dynamics_id =
                                                            tmv::kCartesianCGridDynamicsId;
                                                        ownership_interactions.microphysics_id = mp.id;
                                                        const auto interaction_errors =
                                                            tmv::runtime::validate_science_interactions(
                                                                ownership_interactions);
                                                        errors.insert(errors.end(),
                                                                      interaction_errors.begin(),
                                                                      interaction_errors.end());

                                                        const bool descriptors_accept = std::all_of(
                                                            selections.begin(),
                                                            selections.end(),
                                                            [&](const SelectedScheme& selection)
                                                            {
                                                                const auto* descriptor = tmv::find_scheme_descriptor(
                                                                    *selection.descriptors, selection.id);
                                                                return descriptor != nullptr && descriptor_accepts(
                                                                    *descriptor,
                                                                    qualified,
                                                                    backend,
                                                                    selection.component ==
                                                                        "environment.sounding.file.scheme_id");
                                                            });
                                                        const bool expected =
                                                            descriptors_accept && interaction_errors.empty();
                                                        INFO("mp="
                                                             << mp.id << " rad=" << rad.id << " pbl=" << pbl.id
                                                             << " sgs=" << sgs.id << " chaos=" << stochastic.id
                                                             << " diffusion=" << diff.id << " terrain="
                                                             << terrain_scheme.id << " backend=" << backend
                                                             << " qualified=" << qualified << " mode=" << mode
                                                             << " surface=" << surface.first << "/" << surface.second
                                                             << " output=" << output_format << " initialization="
                                                             << (sounding == nullptr ? "parametric" : sounding->id));
                                                        REQUIRE(errors.empty() == expected);
                                                        if (expected)
                                                        {
                                                            const auto commit_ownership =
                                                                tmv::runtime::resolve_compute_ownership(
                                                                    ownership_interactions);
                                                            REQUIRE(commit_ownership.errors.empty());
                                                            REQUIRE(commit_ownership.microphysics_state_commit !=
                                                                    tmv::runtime::ComputeCommitOwner::Unsupported);
                                                            REQUIRE(commit_ownership.slow_state_commit !=
                                                                    tmv::runtime::ComputeCommitOwner::Unsupported);
                                                            REQUIRE(commit_ownership.sgs_state_commit !=
                                                                    tmv::runtime::ComputeCommitOwner::Unsupported);
                                                            for (const auto& selection : selections)
                                                            {
                                                                if (selection.component ==
                                                                    "environment.sounding.file.scheme_id")
                                                                    continue;
                                                                REQUIRE(selection.descriptors != nullptr);
                                                                const auto* descriptor = tmv::find_scheme_descriptor(
                                                                    *selection.descriptors, selection.id);
                                                                REQUIRE(descriptor != nullptr);
                                                                REQUIRE(tmv::runtime::resolve_compute_process_owner(
                                                                            *descriptor, backend) !=
                                                                        tmv::runtime::ComputeCommitOwner::Unsupported);
                                                            }
                                                        }
                                                        ++evaluated;
                                                        accepted += errors.empty() ? 1u : 0u;
                                                    }

    REQUIRE(evaluated == 1658880);
    REQUIRE(accepted == 622080);
    REQUIRE(accepted < evaluated);
}

TEST_CASE("runtime state inventory is concrete and duplicate free", "[core][runtime][scheme-catalog]")
{
    const auto& fields = tmv::runtime::runtime_state_fields();
    REQUIRE_FALSE(fields.empty());
    const std::set<std::string> unique(fields.begin(), fields.end());
    REQUIRE(unique.size() == fields.size());
    REQUIRE(unique.count("rho") == 1);
    REQUIRE(unique.count("grid_metrics") == 1);
    for (const auto& module : tmv::runtime::scheme_module_catalogs())
        for (const auto& descriptor : *module.schemes)
            for (const auto& required : descriptor.capabilities.required_state_fields)
            {
                INFO("module=" << module.id << " scheme=" << descriptor.id
                               << " required=" << required);
                REQUIRE(unique.count(required) == 1);
            }
    REQUIRE(unique.count("reflectivity") == 0); // optional product, not base state
    REQUIRE(unique.count("dtheta_dt") == 0);    // scheme product, not base state
}
