/**
 * @file test_output_config.cpp
 * @brief Unit tests for OutputConfig parsing, field presets, and disk budget.
 */
#include "catch2/catch.hpp"
#include "core/output/output_config.hpp"

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

// -- Tier 2: Per-field ZFP compression tiers ----------------------------------

TEST_CASE("apply_default_compression_tiers assigns all resolved fields",
          "[core][output_config][zfp_tiers]")
{
    OutputConfig cfg;
    cfg.preset = FieldPreset::all;
    cfg.zfp_per_field_tolerances = true;
    cfg.zfp_tight_tolerance = 1.0e-4;
    cfg.zfp_moderate_tolerance = 1.0e-3;
    cfg.zfp_loose_tolerance = 1.0e-2;
    resolve_output_fields(cfg);
    apply_default_compression_tiers(cfg);

    // Every resolved field must have a tolerance entry
    for (const auto& name : cfg.resolved_fields)
    {
        REQUIRE(cfg.zfp_field_tolerances.count(name) == 1);
        REQUIRE(cfg.zfp_field_tolerances.at(name) > 0.0);
    }
}

TEST_CASE("apply_default_compression_tiers assigns correct tiers",
          "[core][output_config][zfp_tiers]")
{
    OutputConfig cfg;
    cfg.preset = FieldPreset::all;
    cfg.zfp_per_field_tolerances = true;
    cfg.zfp_tight_tolerance = 1.0e-4;
    cfg.zfp_moderate_tolerance = 1.0e-3;
    cfg.zfp_loose_tolerance = 1.0e-2;
    resolve_output_fields(cfg);
    apply_default_compression_tiers(cfg);

    // Core dynamics → tight
    REQUIRE(cfg.zfp_field_tolerances.at("u") == Approx(1.0e-4));
    REQUIRE(cfg.zfp_field_tolerances.at("w") == Approx(1.0e-4));
    REQUIRE(cfg.zfp_field_tolerances.at("rho") == Approx(1.0e-4));
    REQUIRE(cfg.zfp_field_tolerances.at("vorticity_z") == Approx(1.0e-4));

    // Moisture / thermo → moderate
    REQUIRE(cfg.zfp_field_tolerances.at("qv") == Approx(1.0e-3));
    REQUIRE(cfg.zfp_field_tolerances.at("temperature") == Approx(1.0e-3));
    REQUIRE(cfg.zfp_field_tolerances.at("relative_humidity") == Approx(1.0e-3));

    // Diagnostics → loose
    REQUIRE(cfg.zfp_field_tolerances.at("reflectivity_dbz") == Approx(1.0e-2));
    REQUIRE(cfg.zfp_field_tolerances.at("composite_reflectivity") == Approx(1.0e-2));
    REQUIRE(cfg.zfp_field_tolerances.at("cape") == Approx(1.0e-2));
}

TEST_CASE("apply_default_compression_tiers is no-op when disabled",
          "[core][output_config][zfp_tiers]")
{
    OutputConfig cfg;
    cfg.preset = FieldPreset::minimal;
    cfg.zfp_per_field_tolerances = false;
    resolve_output_fields(cfg);
    apply_default_compression_tiers(cfg);

    REQUIRE(cfg.zfp_field_tolerances.empty());
}

TEST_CASE("per-field YAML overrides take precedence over tier defaults",
          "[core][output_config][zfp_tiers]")
{
    OutputConfig cfg;
    cfg.preset = FieldPreset::minimal;
    cfg.zfp_per_field_tolerances = true;
    cfg.zfp_tight_tolerance = 1.0e-4;
    // Pre-populate an explicit override for "u" (normally tight = 1e-4)
    cfg.zfp_field_tolerances["u"] = 5.0e-6;
    resolve_output_fields(cfg);
    apply_default_compression_tiers(cfg);

    // Override preserved — not overwritten by tight tier
    REQUIRE(cfg.zfp_field_tolerances.at("u") == Approx(5.0e-6));
}

TEST_CASE("parse_output_config reads per-field tolerance settings",
          "[core][output_config][zfp_tiers]")
{
    std::unordered_map<std::string, std::string> kv;
    kv["output.format"] = "zfp";
    kv["output.zfp_per_field_tolerances"] = "true";
    kv["output.zfp_tight_tolerance"] = "2.0e-5";
    kv["output.zfp_moderate_tolerance"] = "5.0e-4";
    kv["output.zfp_loose_tolerance"] = "5.0e-3";
    kv["output.zfp_field_tolerance.theta"] = "1.0e-6";

    auto cfg = parse_output_config(kv);
    REQUIRE(cfg.zfp_per_field_tolerances == true);
    REQUIRE(cfg.zfp_tight_tolerance == Approx(2.0e-5));
    REQUIRE(cfg.zfp_moderate_tolerance == Approx(5.0e-4));
    REQUIRE(cfg.zfp_loose_tolerance == Approx(5.0e-3));
    REQUIRE(cfg.zfp_field_tolerances.count("theta") == 1);
    REQUIRE(cfg.zfp_field_tolerances.at("theta") == Approx(1.0e-6));
}

// -- Tier 2b: Tiered write cadence -------------------------------------------

TEST_CASE("get_field_write_cadence_tier returns correct tiers",
          "[core][output_config][cadence]")
{
    // Core dynamics → fast
    REQUIRE(get_field_write_cadence_tier("u") == WriteCadenceTier::fast);
    REQUIRE(get_field_write_cadence_tier("w") == WriteCadenceTier::fast);
    REQUIRE(get_field_write_cadence_tier("rho") == WriteCadenceTier::fast);
    REQUIRE(get_field_write_cadence_tier("vorticity_z") == WriteCadenceTier::fast);
    REQUIRE(get_field_write_cadence_tier("p_prime") == WriteCadenceTier::fast);

    // Moisture / thermo → medium
    REQUIRE(get_field_write_cadence_tier("qv") == WriteCadenceTier::medium);
    REQUIRE(get_field_write_cadence_tier("temperature") == WriteCadenceTier::medium);
    REQUIRE(get_field_write_cadence_tier("relative_humidity") == WriteCadenceTier::medium);
    REQUIRE(get_field_write_cadence_tier("tracer") == WriteCadenceTier::medium);

    // Diagnostics / viz → slow
    REQUIRE(get_field_write_cadence_tier("reflectivity_dbz") == WriteCadenceTier::slow);
    REQUIRE(get_field_write_cadence_tier("composite_reflectivity") == WriteCadenceTier::slow);
    REQUIRE(get_field_write_cadence_tier("cape") == WriteCadenceTier::slow);
    REQUIRE(get_field_write_cadence_tier("u10") == WriteCadenceTier::slow);
}

TEST_CASE("parse_output_config reads tiered cadence settings",
          "[core][output_config][cadence]")
{
    std::unordered_map<std::string, std::string> kv;
    kv["output.tiered_write_cadence"] = "true";
    kv["output.write_cadence_medium_s"] = "20";
    kv["output.write_cadence_slow_s"] = "45";

    auto cfg = parse_output_config(kv);
    REQUIRE(cfg.tiered_write_cadence == true);
    REQUIRE(cfg.write_cadence_medium_s == 20);
    REQUIRE(cfg.write_cadence_slow_s == 45);
}

TEST_CASE("tiered cadence defaults are sane", "[core][output_config][cadence]")
{
    OutputConfig cfg;
    REQUIRE(cfg.tiered_write_cadence == false);
    REQUIRE(cfg.write_cadence_medium_s == 30);
    REQUIRE(cfg.write_cadence_slow_s == 60);
}

TEST_CASE("cadence tiers are consistent with compression tiers",
          "[core][output_config][cadence]")
{
    // Fields in tight compression tier should be fast cadence,
    // moderate → medium, loose → slow. Verify a sample.
    OutputConfig cfg;
    cfg.preset = FieldPreset::all;
    cfg.zfp_per_field_tolerances = true;
    cfg.zfp_tight_tolerance = 1.0e-4;
    cfg.zfp_moderate_tolerance = 1.0e-3;
    cfg.zfp_loose_tolerance = 1.0e-2;
    resolve_output_fields(cfg);
    apply_default_compression_tiers(cfg);

    for (const auto& name : cfg.resolved_fields)
    {
        WriteCadenceTier cadence = get_field_write_cadence_tier(name);
        double tolerance = cfg.zfp_field_tolerances.at(name);

        if (cadence == WriteCadenceTier::fast)
        {
            REQUIRE(tolerance == Approx(cfg.zfp_tight_tolerance));
        }
        else if (cadence == WriteCadenceTier::medium)
        {
            REQUIRE(tolerance == Approx(cfg.zfp_moderate_tolerance));
        }
        else
        {
            REQUIRE(tolerance == Approx(cfg.zfp_loose_tolerance));
        }
    }
}

// -- Tier 3: Advanced compression features ------------------------------------

TEST_CASE("parse_output_config reads Tier 3 settings",
          "[core][output_config][tier3]")
{
    std::unordered_map<std::string, std::string> kv;
    kv["output.zfp_predictive_delta"] = "true";
    kv["output.zfp_sparse_threshold"] = "1.0e-10";
    kv["output.zfp_float16_prefilter"] = "true";

    auto cfg = parse_output_config(kv);
    REQUIRE(cfg.zfp_predictive_delta == true);
    REQUIRE(cfg.zfp_sparse_threshold == Approx(1.0e-10f));
    REQUIRE(cfg.zfp_float16_prefilter == true);
}

TEST_CASE("Tier 3 defaults are disabled", "[core][output_config][tier3]")
{
    OutputConfig cfg;
    REQUIRE(cfg.zfp_predictive_delta == false);
    REQUIRE(cfg.zfp_sparse_threshold == 0.0f);
    REQUIRE(cfg.zfp_float16_prefilter == false);
}

TEST_CASE("is_sparse_eligible identifies hydrometeors",
          "[core][output_config][tier3]")
{
    // Sparse fields
    REQUIRE(is_sparse_eligible("qi"));
    REQUIRE(is_sparse_eligible("qs"));
    REQUIRE(is_sparse_eligible("qh"));
    REQUIRE(is_sparse_eligible("qg"));
    REQUIRE(is_sparse_eligible("qc"));
    REQUIRE(is_sparse_eligible("qr"));
    REQUIRE(is_sparse_eligible("radar"));
    REQUIRE(is_sparse_eligible("reflectivity_dbz"));

    // Non-sparse fields
    REQUIRE_FALSE(is_sparse_eligible("u"));
    REQUIRE_FALSE(is_sparse_eligible("theta"));
    REQUIRE_FALSE(is_sparse_eligible("rho"));
    REQUIRE_FALSE(is_sparse_eligible("temperature"));
}

TEST_CASE("is_float16_eligible identifies bounded diagnostics",
          "[core][output_config][tier3]")
{
    // Float16-eligible fields
    REQUIRE(is_float16_eligible("relative_humidity"));
    REQUIRE(is_float16_eligible("cape"));
    REQUIRE(is_float16_eligible("stp"));
    REQUIRE(is_float16_eligible("u10"));
    REQUIRE(is_float16_eligible("composite_reflectivity"));

    // Not eligible (core dynamics need full precision)
    REQUIRE_FALSE(is_float16_eligible("u"));
    REQUIRE_FALSE(is_float16_eligible("theta"));
    REQUIRE_FALSE(is_float16_eligible("rho"));
    REQUIRE_FALSE(is_float16_eligible("qv"));
}
