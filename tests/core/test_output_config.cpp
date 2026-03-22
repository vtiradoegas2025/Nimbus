/**
 * @file test_output_config.cpp
 * @brief Unit tests for OutputConfig parsing, field presets, and disk budget.
 */
#include "catch2/catch.hpp"
#include "core/output_config.hpp"

TEST_CASE("OutputConfig default values", "[core][output_config]")
{
    OutputConfig cfg;
    REQUIRE(cfg.format == OutputFormat::npy_2d_slices);
    REQUIRE(cfg.preset == FieldPreset::all);
    REQUIRE(cfg.async_io == false);
    REQUIRE(cfg.zfp_tolerance == Approx(1.0e-4));
    REQUIRE(cfg.zfp_keyframe_interval == 0);
}

TEST_CASE("resolve_output_fields with minimal preset", "[core][output_config]")
{
    OutputConfig cfg;
    cfg.preset = FieldPreset::minimal;
    resolve_output_fields(cfg);

    REQUIRE_FALSE(cfg.resolved_fields.empty());
    // Minimal should have core fields like u, w, theta
    REQUIRE(cfg.resolved_fields.count("u") == 1);
    REQUIRE(cfg.resolved_fields.count("w") == 1);
    REQUIRE(cfg.resolved_fields.count("theta") == 1);
}

TEST_CASE("resolve_output_fields with presentation preset", "[core][output_config]")
{
    OutputConfig cfg;
    cfg.preset = FieldPreset::presentation;
    resolve_output_fields(cfg);

    // Presentation should be larger than minimal
    OutputConfig minimal_cfg;
    minimal_cfg.preset = FieldPreset::minimal;
    resolve_output_fields(minimal_cfg);

    REQUIRE(cfg.resolved_fields.size() > minimal_cfg.resolved_fields.size());
}

TEST_CASE("resolve_output_fields with all preset", "[core][output_config]")
{
    OutputConfig cfg;
    cfg.preset = FieldPreset::all;
    resolve_output_fields(cfg);

    const auto& all_names = get_all_field_names();
    REQUIRE(cfg.resolved_fields.size() == all_names.size());
}

TEST_CASE("resolve_output_fields with custom fields", "[core][output_config]")
{
    OutputConfig cfg;
    cfg.preset = FieldPreset::custom;
    cfg.custom_fields = {"w", "theta", "reflectivity_dbz"};
    resolve_output_fields(cfg);

    REQUIRE(cfg.resolved_fields.size() == 3);
    REQUIRE(cfg.resolved_fields.count("w") == 1);
    REQUIRE(cfg.resolved_fields.count("theta") == 1);
    REQUIRE(cfg.resolved_fields.count("reflectivity_dbz") == 1);
}

TEST_CASE("parse_output_config reads format", "[core][output_config]")
{
    std::unordered_map<std::string, std::string> kv;
    kv["output.format"] = "npy_3d";
    auto cfg = parse_output_config(kv);
    REQUIRE(cfg.format == OutputFormat::npy_3d);
}

TEST_CASE("parse_output_config reads zfp settings", "[core][output_config]")
{
    std::unordered_map<std::string, std::string> kv;
    kv["output.format"] = "zfp";
    kv["output.zfp_tolerance"] = "1.0e-5";
    kv["output.zfp_mode"] = "accuracy";
    kv["output.zfp_keyframe_interval"] = "10";

    auto cfg = parse_output_config(kv);
    REQUIRE(cfg.format == OutputFormat::zfp);
    REQUIRE(cfg.zfp_tolerance == Approx(1.0e-5));
    REQUIRE(cfg.zfp_mode == "accuracy");
    REQUIRE(cfg.zfp_keyframe_interval == 10);
}

TEST_CASE("parse_output_config reads async_io", "[core][output_config]")
{
    std::unordered_map<std::string, std::string> kv;
    kv["output.async_io"] = "true";
    auto cfg = parse_output_config(kv);
    REQUIRE(cfg.async_io == true);
}

TEST_CASE("parse_output_config reads csv format", "[core][output_config]")
{
    std::unordered_map<std::string, std::string> kv;
    kv["output.format"] = "csv";
    auto cfg = parse_output_config(kv);
    REQUIRE(cfg.format == OutputFormat::csv);
}

TEST_CASE("estimate_disk_budget populates estimates", "[core][output_config]")
{
    OutputConfig cfg;
    cfg.format = OutputFormat::npy_3d;
    cfg.preset = FieldPreset::minimal;
    resolve_output_fields(cfg);

    estimate_disk_budget(cfg, 64, 64, 32, 3600, 60);

    REQUIRE(cfg.estimated_bytes_per_export > 0);
    REQUIRE(cfg.estimated_total_bytes > 0);
    REQUIRE(cfg.estimated_total_bytes >= cfg.estimated_bytes_per_export);
}

TEST_CASE("output_format_name returns valid strings", "[core][output_config]")
{
    REQUIRE(std::string(output_format_name(OutputFormat::npy_3d)) != "");
    REQUIRE(std::string(output_format_name(OutputFormat::zfp)) != "");
    REQUIRE(std::string(output_format_name(OutputFormat::csv)) != "");
}

TEST_CASE("field_preset_name returns valid strings", "[core][output_config]")
{
    REQUIRE(std::string(field_preset_name(FieldPreset::all)) != "");
    REQUIRE(std::string(field_preset_name(FieldPreset::minimal)) != "");
    REQUIRE(std::string(field_preset_name(FieldPreset::presentation)) != "");
}

TEST_CASE("get_all_field_names is non-empty", "[core][output_config]")
{
    const auto& names = get_all_field_names();
    REQUIRE(names.size() > 20);
}
