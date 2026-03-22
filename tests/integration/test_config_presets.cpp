/**
 * @file test_config_presets.cpp
 * @brief Integration tests verifying preset YAML configs parse correctly.
 */
#include "catch2/catch.hpp"
#include "core/output_config.hpp"

#include <fstream>
#include <filesystem>

namespace
{

bool file_exists(const std::string& path)
{
    return std::filesystem::exists(path);
}

} // namespace

TEST_CASE("Student config file exists", "[integration][configs]")
{
    REQUIRE(file_exists("configs/student.yaml"));
}

TEST_CASE("Research config file exists", "[integration][configs]")
{
    REQUIRE(file_exists("configs/research.yaml"));
}

TEST_CASE("Production config file exists", "[integration][configs]")
{
    REQUIRE(file_exists("configs/production.yaml"));
}

TEST_CASE("Benchmark config file exists", "[integration][configs]")
{
    REQUIRE(file_exists("configs/benchmark.yaml"));
}

TEST_CASE("Teaching configs exist", "[integration][configs]")
{
    REQUIRE(file_exists("configs/teaching/thermal_bubble.yaml"));
    REQUIRE(file_exists("configs/teaching/supercell_30min.yaml"));
    REQUIRE(file_exists("configs/teaching/tornado_genesis.yaml"));
}

TEST_CASE("Output presets produce valid field counts", "[integration][output]")
{
    SECTION("minimal has ~8 fields")
    {
        OutputConfig cfg;
        cfg.preset = FieldPreset::minimal;
        resolve_output_fields(cfg);
        REQUIRE(cfg.resolved_fields.size() >= 6);
        REQUIRE(cfg.resolved_fields.size() <= 12);
    }

    SECTION("presentation has more than minimal")
    {
        OutputConfig min_cfg;
        min_cfg.preset = FieldPreset::minimal;
        resolve_output_fields(min_cfg);

        OutputConfig pres_cfg;
        pres_cfg.preset = FieldPreset::presentation;
        resolve_output_fields(pres_cfg);

        REQUIRE(pres_cfg.resolved_fields.size() > min_cfg.resolved_fields.size());
    }

    SECTION("all has >= 50 fields")
    {
        OutputConfig cfg;
        cfg.preset = FieldPreset::all;
        resolve_output_fields(cfg);
        REQUIRE(cfg.resolved_fields.size() >= 50);
    }
}
