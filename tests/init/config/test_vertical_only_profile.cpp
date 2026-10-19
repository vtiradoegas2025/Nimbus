#include "catch2/catch.hpp"
#include "init/config/config_v1_serialization_common.hpp"
#include "init/config/v1_loader.hpp"

using namespace tmv::init::config;

TEST_CASE("static vertical-only candidate is exact while public selection stays guarded",
          "[init][vertical_only][guard]")
{
    const auto loaded = load_v1_config(
        "tests/fixtures/configs/qualification/cgrid_dynamic_smagorinsky_ready.yaml");
    REQUIRE(loaded.ok());
    const auto original = *loaded.config;
    REQUIRE(is_public_qualified_dynamic_smagorinsky_profile(original));
    REQUIRE_NOTHROW(require_runtime_supported_enum_selections(original));

    auto candidate = original;
    candidate.turbulence.mode = TurbulenceMode::VerticalOnly;
    candidate.turbulence.dynamic_coefficient = false;
    REQUIRE(is_transactional_static_vertical_only_cpu_candidate(candidate));
    REQUIRE_NOTHROW(require_runtime_supported_enum_selections(candidate));
    candidate.extensions.execution.mode = ExecutionMode::Exploratory;
    REQUIRE(is_transactional_static_vertical_only_cpu_candidate(candidate));
    REQUIRE_NOTHROW(require_runtime_supported_enum_selections(candidate));

    auto invalid = candidate;
    invalid.grid.nx = 13;
    REQUIRE_FALSE(is_transactional_static_vertical_only_cpu_candidate(invalid));
    invalid.extensions.execution.mode = ExecutionMode::Qualified;
    REQUIRE_THROWS_WITH(require_runtime_supported_enum_selections(invalid),
                        Catch::Matchers::Contains("guarded pending whole-step failure atomicity"));
    invalid = candidate;
    invalid.extensions.compute.backend = ComputeBackend::Metal;
    REQUIRE_FALSE(is_transactional_static_vertical_only_cpu_candidate(invalid));
    invalid.extensions.execution.mode = ExecutionMode::Qualified;
    REQUIRE_THROWS_WITH(require_runtime_supported_enum_selections(invalid),
                        Catch::Matchers::Contains("guarded pending whole-step failure atomicity"));
    invalid = candidate;
    invalid.boundary_layer.scheme = BoundaryLayerScheme::Slab;
    REQUIRE_FALSE(is_transactional_static_vertical_only_cpu_candidate(invalid));
    invalid.extensions.execution.mode = ExecutionMode::Qualified;
    REQUIRE_THROWS_WITH(require_runtime_supported_enum_selections(invalid),
                        Catch::Matchers::Contains("guarded pending whole-step failure atomicity"));
    invalid = candidate;
    invalid.output.format = OutputFormat::ZfpHigh;
    REQUIRE_FALSE(is_transactional_static_vertical_only_cpu_candidate(invalid));
    invalid.extensions.execution.mode = ExecutionMode::Qualified;
    REQUIRE_THROWS_WITH(require_runtime_supported_enum_selections(invalid),
                        Catch::Matchers::Contains("guarded pending whole-step failure atomicity"));
    invalid = candidate;
    invalid.turbulence.mode = TurbulenceMode::ThreeD;
    REQUIRE_FALSE(is_transactional_static_vertical_only_cpu_candidate(invalid));
}

TEST_CASE("dynamic Smagorinsky rejects non-3D modes before allocation in every execution mode",
          "[init][vertical_only][dynamic_guard]")
{
    const auto loaded = load_v1_config(
        "tests/fixtures/configs/qualification/cgrid_dynamic_smagorinsky_ready.yaml");
    REQUIRE(loaded.ok());
    for (const auto mode : {ExecutionMode::Qualified, ExecutionMode::Exploratory})
    {
        auto candidate = *loaded.config;
        candidate.extensions.execution.mode = mode;
        candidate.turbulence.mode = TurbulenceMode::VerticalOnly;
        REQUIRE_THROWS_WITH(require_runtime_supported_enum_selections(candidate),
                            Catch::Matchers::Contains("requires turbulence.mode=3d"));
        candidate.turbulence.mode = TurbulenceMode::TwoD;
        REQUIRE_THROWS_WITH(require_runtime_supported_enum_selections(candidate),
                            Catch::Matchers::Contains("requires turbulence.mode=3d"));
    }
}
