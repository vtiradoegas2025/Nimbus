#include "catch2/catch.hpp"
#include "core/runtime/yaml_helpers.hpp"
#include "init/config/config_v1_serialization_common.hpp"
#include "init/config/v0_to_v1.hpp"
#include "init/config/v1_loader.hpp"
#include "microphysics/milbrandt_triple_scope.hpp"

using namespace tmv::init::config;

TEST_CASE("retained Milbrandt triple selector remains fail-closed in strict and typed v1",
          "[init][milbrandt][triple_scope]")
{
    constexpr const char* fixture =
        "tests/fixtures/configs/qualification/cgrid_milbrandt_cpu_ready.yaml";
    const auto loaded = load_v1_config(fixture);
    REQUIRE(loaded.ok());
    REQUIRE_FALSE(tmv::microphysics::kMilbrandtTripleMomentPublicAdmissionOpen);
    REQUIRE_FALSE(tmv::microphysics::kMilbrandtTripleMomentStrictV1ExplicitKeyAdmitted);
    REQUIRE_NOTHROW(require_runtime_supported_enum_selections(*loaded.config));

    const auto parsed = parse_yaml_with_lines(fixture);
    REQUIRE(parsed.has_value());
    for (const char* value : {"false", "true"})
    {
        auto fields = *parsed;
        fields["microphysics.milbrandt.triple_moment"] = {value, 1};
        const auto rejected = load_v1_config_fields(fixture, std::move(fields));
        REQUIRE_FALSE(rejected.ok());
        REQUIRE_THAT(format_errors(rejected.errors),
                     Catch::Matchers::Contains("engineering incomplete and not admitted"));
    }

    for (const auto mode : {ExecutionMode::Qualified, ExecutionMode::Exploratory})
    {
        auto candidate = *loaded.config;
        candidate.extensions.execution.mode = mode;
        candidate.microphysics.parameters.milbrandt.triple_moment = true;
        REQUIRE_THROWS_WITH(require_runtime_supported_enum_selections(candidate),
                            Catch::Matchers::Contains("retained but engineering incomplete"));
        REQUIRE_THROWS_WITH(serialize_v1_to_yaml(candidate),
                            Catch::Matchers::Contains("retained but engineering incomplete"));
        candidate.microphysics.scheme = MicrophysicsScheme::Kessler;
        REQUIRE_THROWS_WITH(require_runtime_supported_enum_selections(candidate),
                            Catch::Matchers::Contains("retained but engineering incomplete"));
        REQUIRE_THROWS_WITH(serialize_v1_to_yaml(candidate),
                            Catch::Matchers::Contains("retained but engineering incomplete"));
        candidate.microphysics.parameters.milbrandt.triple_moment = false;
        candidate.extensions.execution.mode = ExecutionMode::Exploratory;
        REQUIRE_NOTHROW(require_runtime_supported_enum_selections(candidate));
    }
}
