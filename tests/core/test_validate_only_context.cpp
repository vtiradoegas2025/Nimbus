#include <catch2/catch.hpp>

#include "core/infra/sha256.hpp"
#include "core/runtime/validate_only_context.hpp"
#include "core/runtime/yaml_helpers.hpp"
#include "init/config/v1_loader.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace
{
std::string fixture_bytes()
{
    std::ifstream input("tests/fixtures/reference/catalog/flat_mixed_boundary_metal_exploratory_v1.yaml");
    REQUIRE(input.good());
    return tmv::runtime::read_validate_only_source_bytes(input);
}

std::filesystem::path temporary_source()
{
    return std::filesystem::temp_directory_path() / ("nimbus-context-parser-" +
        std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + ".yaml");
}
}

TEST_CASE("native context serializes actual typed scope and preserves source ownership", "[core][validate-context]")
{
    const auto bytes = fixture_bytes();
    auto loaded = tmv::init::config::load_v1_config_fields("original.yaml", parse_yaml_text_with_lines(bytes));
    REQUIRE(loaded.ok());
    auto& config = *loaded.config;
    const auto digest = nimbus::crypto::sha256_hex(bytes);
    const auto exploratory = tmv::runtime::serialize_validate_only_context_v1(config, digest, "none", {"w", "rho"});
    REQUIRE(exploratory.find("\"executionMode\":\"exploratory\"") != std::string::npos);
    REQUIRE(exploratory.find("\"backend\":\"metal\"") != std::string::npos);
    REQUIRE(exploratory.find("\"sourceSha256\":\"" + digest + "\"") != std::string::npos);
    REQUIRE(exploratory.find("\"availableStateFields\":[\"rho\",\"w\"]") != std::string::npos);
    REQUIRE(exploratory.find("\"lateralSpongeEnabled\":true,\"lateralWidth\":20") != std::string::npos);
    REQUIRE(exploratory.find("\"slowRk2\":true") != std::string::npos);
    config.extensions.execution.mode = tmv::init::config::ExecutionMode::Qualified;
    const auto qualified = tmv::runtime::serialize_validate_only_context_v1(config, digest, "mixed_npy", {"w", "rho"});
    REQUIRE(qualified.find("\"executionMode\":\"qualified\"") != std::string::npos);
    REQUIRE(qualified.find("\"generatedMetalScopeKind\":\"mixed_npy\"") != std::string::npos);
    REQUIRE(qualified.size() < tmv::runtime::kValidateOnlyContextLimitBytes);
    config.output.zfp_mode = "accuracy\"\\\n\t";
    const auto escaped = tmv::runtime::serialize_validate_only_context_v1(config, digest, "mixed_npy", {});
    REQUIRE(escaped.find("accuracy\\\"\\\\\\u000a\\u0009") != std::string::npos);
    REQUIRE(config.output.zfp_mode == "accuracy\"\\\n\t");
}

TEST_CASE("context byte reader enforces its local bound and preserves exact byte forms", "[core][validate-context]")
{
    for (const auto& bytes : {std::string{}, std::string("schema: nimbus.v1\r\n"), std::string("schema: nimbus.v1")})
    {
        std::istringstream input(bytes);
        REQUIRE(tmv::runtime::read_validate_only_source_bytes(input) == bytes);
    }
    std::istringstream fitting(std::string(tmv::runtime::kValidateOnlySourceLimitBytes, ' '));
    REQUIRE(tmv::runtime::read_validate_only_source_bytes(fitting).size() == tmv::runtime::kValidateOnlySourceLimitBytes);
    std::istringstream oversized(std::string(tmv::runtime::kValidateOnlySourceLimitBytes + 1, ' '));
    REQUIRE_THROWS_WITH(tmv::runtime::read_validate_only_source_bytes(oversized), "preflight context source exceeds 8 MiB");
    std::istringstream broken;
    broken.setstate(std::ios::badbit);
    REQUIRE_THROWS_WITH(tmv::runtime::read_validate_only_source_bytes(broken), "preflight context source read failed");
    broken.setstate(std::ios::eofbit);
    REQUIRE_THROWS_WITH(tmv::runtime::read_validate_only_source_bytes(broken), "preflight context source read failed");
}

TEST_CASE("file and immutable text use identical YAML parsing and line semantics", "[core][validate-context][yaml]")
{
    for (const auto& bytes : {std::string{}, std::string("schema: nimbus.v1"),
             std::string("schema: nimbus.v1\r\n# comment\r\nextensions:\r\n  execution:\r\n    mode: qualified\r\n"),
             std::string("schema: wrong\nschema: nimbus.v1\nextensions:\n  execution:\n    mode: 'qualified' # comment\n"),
             std::string("schema: nimbus.v2\ngrid:\n  nx: invalid\n")})
    {
        const auto path = temporary_source();
        {
            std::ofstream output(path, std::ios::binary);
            output << bytes;
        }
        const auto file = parse_yaml_with_lines(path.string());
        REQUIRE(file.has_value());
        const auto text = parse_yaml_text_with_lines(bytes);
        REQUIRE(file->size() == text.size());
        for (const auto& [key, value] : *file)
        {
            REQUIRE(text.at(key).value == value.value);
            REQUIRE(text.at(key).line_number == value.line_number);
        }
        const auto file_result = tmv::init::config::load_v1_config(path.string());
        const auto text_result = tmv::init::config::load_v1_config_fields(path.string(), text);
        REQUIRE(file_result.ok() == text_result.ok());
        REQUIRE(tmv::init::config::format_errors(file_result.errors) ==
            tmv::init::config::format_errors(text_result.errors));
        std::filesystem::remove(path);
    }
}

TEST_CASE("receipt input identity follows immutable parsed bytes after original file changes", "[core][validate-context][provenance]")
{
    const auto original = fixture_bytes();
    const auto path = temporary_source();
    {
        std::ofstream output(path, std::ios::binary);
        output << original;
    }
    std::ifstream input(path, std::ios::binary);
    const auto captured = tmv::runtime::read_validate_only_source_bytes(input);
    input.close();
    {
        std::ofstream changed(path, std::ios::binary | std::ios::trunc);
        changed << "schema: wrong\n";
    }
    const auto parsed = tmv::init::config::load_v1_config_fields(path.string(), parse_yaml_text_with_lines(captured));
    REQUIRE(parsed.ok());
    REQUIRE_FALSE(tmv::init::config::load_v1_config(path.string()).ok());
    const auto receipt = tmv::runtime::serialize_validate_only_context_v1(*parsed.config,
        nimbus::crypto::sha256_hex(captured), "none", {});
    REQUIRE(receipt.find(nimbus::crypto::sha256_hex(original)) != std::string::npos);
    REQUIRE(receipt.find(nimbus::crypto::sha256_hex("schema: wrong\n")) == std::string::npos);
    std::filesystem::remove(path);
}

TEST_CASE("native context fails closed on invalid identity nonfinite or unbounded facts", "[core][validate-context][invalid]")
{
    const auto parsed = tmv::init::config::load_v1_config_fields("original.yaml", parse_yaml_text_with_lines(fixture_bytes()));
    REQUIRE(parsed.ok());
    auto config = *parsed.config;
    const std::string digest(64, 'a');
    REQUIRE_THROWS(tmv::runtime::serialize_validate_only_context_v1(config, "wrong", "none", {}));
    REQUIRE_THROWS(tmv::runtime::serialize_validate_only_context_v1(config, digest, "unknown", {}));
    REQUIRE_THROWS(tmv::runtime::serialize_validate_only_context_v1(config, digest, "none", {"w", "w"}));
    config.time.dt_large = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS(tmv::runtime::serialize_validate_only_context_v1(config, digest, "none", {}));
    config = *parsed.config;
    config.output.zfp_mode.assign(8193, 'x');
    REQUIRE_THROWS(tmv::runtime::serialize_validate_only_context_v1(config, digest, "none", {}));
    config = *parsed.config;
    std::vector<std::string> large_inventory;
    for (int index = 0; index < 100; ++index)
    {
        large_inventory.push_back(std::to_string(index) + std::string(1024, '\n'));
    }
    REQUIRE_THROWS(tmv::runtime::serialize_validate_only_context_v1(config, digest, "none", large_inventory));
    std::ostringstream stream;
    {
        tmv::runtime::validate_only_detail::JsonObject object(stream);
        stream.setstate(std::ios::badbit);
        REQUIRE_THROWS_WITH(object.field("value", 1), "preflight context serialization stream failed");
    }
}
