/**
 * @file test_config_presets.cpp
 * @brief Integration tests verifying preset YAML configs parse correctly.
 */
#include "catch2/catch.hpp"
#include "core/output/output_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <utility>
#include <vector>

namespace
{

bool file_exists(const std::string& path)
{
    return std::filesystem::exists(path);
}

struct CapturedCommand
{
    int         status = -1;
    std::string output;
};

CapturedCommand run_and_capture(const std::string& command, const std::string& log_name)
{
    namespace fs             = std::filesystem;
    const fs::path  log_path = fs::temp_directory_path() / log_name;
    CapturedCommand result;
    result.status = std::system((command + " >" + log_path.string() + " 2>&1").c_str());

    std::ifstream log_in(log_path);
    result.output.assign(std::istreambuf_iterator<char>(log_in), std::istreambuf_iterator<char>());
    std::error_code ec;
    fs::remove(log_path, ec);
    return result;
}

CapturedCommand run_config_text(const std::string& yaml, const std::string& stem, const std::string& extra_args = "")
{
    namespace fs               = std::filesystem;
    const fs::path config_path = fs::temp_directory_path() / (stem + ".yaml");
    {
        std::ofstream config_out(config_path);
        config_out << yaml;
    }

    CapturedCommand result =
        run_and_capture("bin/nimbus_sim --headless --config " + config_path.string() + extra_args, stem + ".log");
    std::error_code ec;
    fs::remove(config_path, ec);
    return result;
}

std::string replace_once(std::string text, const std::string& needle, const std::string& replacement)
{
    const std::size_t offset = text.find(needle);
    REQUIRE(offset != std::string::npos);
    text.replace(offset, needle.size(), replacement);
    return text;
}

struct WaterClosureSummary
{
    int    total_step_rows              = 0;
    double initial_inventory            = 0.0;
    double cumulative_change            = 0.0;
    double maximum_step_relative_change = 0.0;
    double dynamics_change              = 0.0;
    double advection_change             = 0.0;
};

WaterClosureSummary read_water_closure(const std::filesystem::path& csv_path)
{
    WaterClosureSummary summary;
    std::ifstream       input(csv_path);
    std::string         line;
    std::getline(input, line); // header
    while (std::getline(input, line))
    {
        std::vector<std::string> columns;
        std::istringstream       row(line);
        std::string              value;
        while (std::getline(row, value, ','))
            columns.push_back(value);
        if (columns.size() < 11)
            continue;

        const std::string& stage    = columns[1];
        const double       change   = std::stod(columns[4]);
        const double       relative = std::abs(std::stod(columns[10]));
        if (stage == "dynamics")
            summary.dynamics_change += change;
        if (stage == "advection")
            summary.advection_change += change;
        if (stage != "total_step")
            continue;

        ++summary.total_step_rows;
        summary.cumulative_change += change;
        summary.maximum_step_relative_change = std::max(summary.maximum_step_relative_change, relative);
        if (summary.initial_inventory == 0.0 && relative > 0.0)
        {
            summary.initial_inventory = std::abs(change) / relative;
        }
    }
    return summary;
}

std::string top_level_scalar(const std::string& yaml,
                             const std::string& section,
                             const std::string& key,
                             const std::string& fallback)
{
    std::istringstream input(yaml);
    std::string        line;
    bool               in_section     = false;
    const std::string  section_header = section + ":";
    const std::string  key_prefix     = "  " + key + ":";
    while (std::getline(input, line))
    {
        if (!line.empty() && line.front() != ' ' && line.front() != '#')
        {
            in_section = line == section_header;
            continue;
        }
        if (!in_section || line.rfind(key_prefix, 0) != 0)
        {
            continue;
        }
        std::string value   = line.substr(key_prefix.size());
        const auto  comment = value.find('#');
        if (comment != std::string::npos)
            value.resize(comment);
        const auto begin = value.find_first_not_of(" \t\r");
        const auto end   = value.find_last_not_of(" \t\r");
        return begin == std::string::npos ? fallback : value.substr(begin, end - begin + 1);
    }
    return fallback;
}

} // namespace

TEST_CASE("Public resolution catalog has complete coupled physics matrix",
          "[integration][configs]")
{
    const std::vector<std::string> resolutions = {
        "1000", "500", "450", "400", "350", "300",
        "250", "200", "150", "100", "50",
    };
    const std::vector<std::string> variants = {
        "dry", "kessler", "lin", "thompson", "milbrandt",
    };
    const std::vector<std::string> profiles = {
        "baseline", "radiation", "ysu", "ysu_radiation",
        "les", "les_radiation", "tke", "tke_radiation",
    };
    for (const auto& resolution : resolutions)
    {
        for (const auto& variant : variants)
        {
            for (const auto& profile : profiles)
            {
                const std::string path = "configs/" + resolution
                    + "meterResolution/" + variant + "/" + profile + ".yaml";
                REQUIRE(file_exists(path));
            }
        }
    }
}

TEST_CASE("SCI-017 shipped strict configs name lateral algorithms explicitly",
          "[integration][configs][boundary_conditions][SCI-017]")
{
    namespace fs               = std::filesystem;
    std::size_t strict_configs = 0;
    for (const auto& entry : fs::recursive_directory_iterator("configs"))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".yaml")
        {
            continue;
        }
        std::ifstream     input(entry.path());
        const std::string yaml((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        if (yaml.find("schema: nimbus.v1") == std::string::npos)
        {
            continue;
        }
        ++strict_configs;
        CAPTURE(entry.path().string());
        REQUIRE(yaml.find("    west: open") == std::string::npos);
        REQUIRE(yaml.find("    east: open") == std::string::npos);
        REQUIRE(yaml.find("    south: open") == std::string::npos);
        REQUIRE(yaml.find("    north: open") == std::string::npos);
    }
    REQUIRE(strict_configs >= 440);
}

TEST_CASE("shipped strict configs do not double-count PBL and vertical SGS",
          "[integration][configs][physics-coupling][AG-007]")
{
    namespace fs                   = std::filesystem;
    std::size_t active_pbl_configs = 0;
    for (const auto& entry : fs::recursive_directory_iterator("configs"))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".yaml")
        {
            continue;
        }
        std::ifstream     input(entry.path());
        const std::string yaml((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        if (yaml.find("schema: nimbus.v1") == std::string::npos)
        {
            continue;
        }
        const std::string pbl = top_level_scalar(yaml, "boundary_layer", "scheme", "none");
        if (pbl == "none")
        {
            continue;
        }
        ++active_pbl_configs;
        const std::string sgs  = top_level_scalar(yaml, "turbulence", "scheme", "none");
        const std::string mode = top_level_scalar(yaml, "turbulence", "mode", "3d");
        CAPTURE(entry.path().string(), pbl, sgs, mode);
        REQUIRE((sgs == "none" || mode == "2d"));
        REQUIRE(yaml.find("    backend: gpu") == std::string::npos);
    }
    REQUIRE(active_pbl_configs > 0);
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

    SECTION("studio_full covers presentation without selecting everything")
    {
        OutputConfig pres_cfg;
        pres_cfg.preset = FieldPreset::presentation;
        resolve_output_fields(pres_cfg);

        OutputConfig studio_cfg;
        studio_cfg.preset = FieldPreset::studio_full;
        resolve_output_fields(studio_cfg);

        OutputConfig all_cfg;
        all_cfg.preset = FieldPreset::all;
        resolve_output_fields(all_cfg);

        REQUIRE(studio_cfg.resolved_fields.size() > pres_cfg.resolved_fields.size());
        REQUIRE(studio_cfg.resolved_fields.size() < all_cfg.resolved_fields.size());
    }

    SECTION("all has >= 50 fields")
    {
        OutputConfig cfg;
        cfg.preset = FieldPreset::all;
        resolve_output_fields(cfg);
        REQUIRE(cfg.resolved_fields.size() >= 50);
    }
}

TEST_CASE("SCI-025 production C-grid order conserves retained-density water", "[integration][conservation][SCI-025]")
{
    namespace fs = std::filesystem;
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping SCI-025 integration test");
        return;
    }

    const fs::path  output_dir = fs::temp_directory_path() / "nimbus_sci025_conservation";
    std::error_code ec;
    fs::remove_all(output_dir, ec);

    const CapturedCommand result = run_and_capture("bin/nimbus_sim --headless "
                                                   "--config tests/fixtures/configs/test/acceptance/sci025_cgrid_conservation.yaml "
                                                   "--compute-backend=cpu --log-profile=quiet --no-timing "
                                                   "--outdir=" +
                                                       output_dir.string(),
                                                   "nimbus_sci025_conservation.log");

    const fs::path            budget_path = output_dir / "budget.csv";
    const bool                have_budget = fs::exists(budget_path);
    const WaterClosureSummary closure     = have_budget ? read_water_closure(budget_path) : WaterClosureSummary{};
    fs::remove_all(output_dir, ec);

    REQUIRE(result.status != -1);
    REQUIRE(WIFEXITED(result.status));
    REQUIRE(WEXITSTATUS(result.status) == 0);
    REQUIRE(have_budget);
    REQUIRE(closure.total_step_rows == 100);
    REQUIRE(closure.initial_inventory > 1.0e10);

    // The bubble must actually exercise divergent transport; a calm/no-op
    // fixture could otherwise pass a conservation assertion trivially.
    REQUIRE(std::abs(closure.dynamics_change) > 1.0e7);
    REQUIRE(std::abs(closure.advection_change) > 1.0e7);

    const double cumulative_relative_error = std::abs(closure.cumulative_change) / closure.initial_inventory;
    INFO("SCI-025 cumulative relative water error="
         << cumulative_relative_error << " max step=" << closure.maximum_step_relative_change
         << " dynamics=" << closure.dynamics_change << " advection=" << closure.advection_change);
    REQUIRE(closure.maximum_step_relative_change < 5.0e-8);
    REQUIRE(cumulative_relative_error < 2.0e-7);
}

TEST_CASE("SCI-026 strict sounding source reaches runtime truth reporting", "[integration][configs][sounding][SCI-026]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping SCI-026 runtime test");
        return;
    }

    const CapturedCommand result = run_and_capture("bin/nimbus_sim --headless "
                                                   "--config tests/fixtures/configs/test/acceptance/sci026_parametric_targets.yaml "
                                                   "--duration=0 --write-every=0 --log-profile=normal --no-timing",
                                                   "nimbus_sci026_parametric_targets.log");

    REQUIRE(result.status != -1);
    REQUIRE(WIFEXITED(result.status));
    REQUIRE(WEXITSTATUS(result.status) == 0);
    REQUIRE(result.output.find("Sounding source: parametric_targets") != std::string::npos);
    REQUIRE(result.output.find("Base state initialized via parametric_targets") != std::string::npos);

    // Startup must publish the user's complete target request beside the
    // realized diagnostic table and its explicit resolution-aware policy.
    REQUIRE(result.output.find("source=parametric_targets requested={cape_jkg:2500,cin_jkg:50") != std::string::npos);
    REQUIRE(result.output.find("lcl_m:1100m,lfc_m:1500,el_m:12000") != std::string::npos);
    REQUIRE(result.output.find("realized={cape_jkg:") != std::string::npos);
    REQUIRE(result.output.find("tolerances={cape_jkg:125,cin_jkg:75,height_m:250}") != std::string::npos);
    REQUIRE(result.output.find("within_tolerance=") != std::string::npos);

    // The same target family must not be accepted under parametric_cape,
    // where CIN/LCL/LFC/EL would be inert. Exercise the executable boundary,
    // not only the typed loader unit.
    std::ifstream fixture("tests/fixtures/configs/test/acceptance/sci026_parametric_targets.yaml");
    REQUIRE(fixture.good());
    std::string       invalid_yaml((std::istreambuf_iterator<char>(fixture)), std::istreambuf_iterator<char>());
    const std::string selected  = "    type: parametric_targets";
    const std::size_t selection = invalid_yaml.find(selected);
    REQUIRE(selection != std::string::npos);
    invalid_yaml.replace(selection, selected.size(), "    type: parametric_cape");

    const CapturedCommand rejected = run_config_text(invalid_yaml, "nimbus_sci026_inert_targets");
    REQUIRE(rejected.status != -1);
    REQUIRE(WIFEXITED(rejected.status));
    REQUIRE(WEXITSTATUS(rejected.status) == 1);
    REQUIRE(rejected.output.find("initial_conditions.sounding.cin_jkg") != std::string::npos);
    REQUIRE(rejected.output.find("not valid for sounding.type=parametric_cape") != std::string::npos);

    // The maintained target fixture intentionally misses its LFC/EL tolerance.
    // That remains useful exploratory evidence, but it must not be possible to
    // relabel the same initialized atmosphere as qualified science.
    std::ifstream qualified_fixture("tests/fixtures/configs/test/acceptance/sci026_parametric_targets.yaml");
    REQUIRE(qualified_fixture.good());
    std::string qualified_yaml((std::istreambuf_iterator<char>(qualified_fixture)), std::istreambuf_iterator<char>());
    const std::string extensions     = "extensions:\n";
    const std::size_t extensions_pos = qualified_yaml.find(extensions);
    REQUIRE(extensions_pos != std::string::npos);
    qualified_yaml.insert(extensions_pos + extensions.size(),
                          "  execution:\n"
                          "    mode: qualified\n");

    const CapturedCommand target_rejected = run_config_text(qualified_yaml,
                                                            "nimbus_sci026_qualified_target_mismatch",
                                                            " --duration=0 --write-every=0 --log-profile=normal");
    REQUIRE(target_rejected.status != -1);
    REQUIRE(WIFEXITED(target_rejected.status));
    REQUIRE(WEXITSTATUS(target_rejected.status) == 1);
    REQUIRE(target_rejected.output.find("within_tolerance=no") != std::string::npos);
    REQUIRE(target_rejected.output.find("qualified initialization rejected parametric_targets") != std::string::npos);
}

// ─────────────────────────────────────────────────────────────
// Config load failure exit codes (regression: pre-fix, an
// unloadable config logged the error and then ran the built-in
// DEFAULT simulation, segfaulting downstream with exit 139)
// ─────────────────────────────────────────────────────────────

TEST_CASE("nimbus_sim fails cleanly on unloadable configs", "[integration][configs]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping config-failure exit-code test");
        return;
    }

    SECTION("missing config file exits non-zero (no default-config run)")
    {
        const int rc = std::system("bin/nimbus_sim --config /nonexistent/nimbus_missing.yaml --headless "
                                   ">/dev/null 2>&1");
        REQUIRE(rc != -1);
        REQUIRE(WIFEXITED(rc));
        REQUIRE(WEXITSTATUS(rc) == 1);
    }

    SECTION("v1 config with an invalid enum exits non-zero without crashing")
    {
        namespace fs       = std::filesystem;
        const fs::path tmp = fs::temp_directory_path() / "nimbus_bad_backend_test.yaml";
        {
            std::ofstream f(tmp);
            f << "schema: nimbus.v1\n"
                 "grid:\n"
                 "  nx: 16\n  ny: 16\n  nz: 16\n"
                 "  dx: 250.0\n  dy: 250.0\n  dz: 250.0\n"
                 "time:\n"
                 "  run_time: 10.0\n  dt_large: 2.0\n  output_interval: 0\n"
                 "extensions:\n"
                 "  compute:\n"
                 "    backend: cuda\n";
        }
        const std::string cmd = "bin/nimbus_sim --config " + tmp.string() + " --headless >/dev/null 2>&1";
        const int         rc  = std::system(cmd.c_str());
        std::error_code   ec;
        fs::remove(tmp, ec);
        REQUIRE(rc != -1);
        // A signal exit (the old segfault, 139) fails WIFEXITED.
        REQUIRE(WIFEXITED(rc));
        REQUIRE(WEXITSTATUS(rc) == 1);
    }
}

TEST_CASE("nimbus_sim command-line boundary is explicit and non-crashing", "[integration][cli]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping CLI boundary test");
        return;
    }

    SECTION("help exits before configuration or initialization")
    {
        const CapturedCommand result = run_and_capture("bin/nimbus_sim --help", "nimbus_cli_help.log");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 0);
        REQUIRE(result.output.find("Usage:") != std::string::npos);
        REQUIRE(result.output.find("--config PATH") != std::string::npos);
        REQUIRE(result.output.find("--checkpoint-every=SECONDS") != std::string::npos);
        REQUIRE(result.output.find("--resume=PATH") != std::string::npos);
        REQUIRE(result.output.find("--acoustic-substeps COUNT") != std::string::npos);
        REQUIRE(result.output.find("[RUN SETTINGS]") == std::string::npos);
    }

    SECTION("version exits before configuration or initialization")
    {
        const CapturedCommand result = run_and_capture("bin/nimbus_sim --version", "nimbus_cli_version.log");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 0);
        REQUIRE(result.output.find("config schema nimbus.v1") != std::string::npos);
    }

    SECTION("no arguments reports the required input instead of entering initialization")
    {
        const CapturedCommand result = run_and_capture("bin/nimbus_sim", "nimbus_cli_no_args.log");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find("Missing required --config PATH") != std::string::npos);
    }

    SECTION("unknown options fail instead of being silently ignored")
    {
        const CapturedCommand result =
            run_and_capture("bin/nimbus_sim --definitely-not-a-nimbus-option", "nimbus_cli_unknown.log");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find("Unknown argument") != std::string::npos);
    }

    SECTION("options requiring a separate value report a missing value")
    {
        const CapturedCommand result = run_and_capture("bin/nimbus_sim --config", "nimbus_cli_missing_value.log");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find("Missing value for --config") != std::string::npos);
    }

    SECTION("trajectory numerics reject malformed values")
    {
        const CapturedCommand result = run_and_capture(
            "bin/nimbus_sim --config tests/fixtures/configs/test/legacy/base_cartesian.yaml --trajectory-seed-time=nan",
            "nimbus_cli_bad_trajectory.log");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find("Invalid --trajectory-seed-time") != std::string::npos);
    }

    SECTION("acoustic substeps reject non-positive values")
    {
        const CapturedCommand result =
            run_and_capture("bin/nimbus_sim --acoustic-substeps=0", "nimbus_cli_bad_acoustic_substeps.log");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find("Invalid --acoustic-substeps") != std::string::npos);
    }

    SECTION("ordinary binary rejects internal qualification state injection")
    {
        const CapturedCommand result =
            run_and_capture("bin/nimbus_sim --qualification-hydrometeor-pulse", "nimbus_cli_internal_hydrometeor.log");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find("only in bin/nimbus_sim_agrid_qualification") != std::string::npos);
    }

    SECTION("ordinary binary rejects the internal moving-nest campaign")
    {
        const CapturedCommand result = run_and_capture(
            "bin/nimbus_sim --qualification-moving-nest",
            "nimbus_cli_internal_moving_nest.log");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find("only in the internal qualification binary") !=
                std::string::npos);
    }

    SECTION("large timestep rejects non-positive values")
    {
        const CapturedCommand result = run_and_capture("bin/nimbus_sim --dt-large=0", "nimbus_cli_bad_dt_large.log");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find("Invalid --dt-large") != std::string::npos);
    }

    SECTION("ordinary binary rejects every internal acoustic-state injection")
    {
        const std::vector<std::pair<std::string, std::string>> hooks = {
            {"--qualification-acoustic-wave", "wave"},
            {"--qualification-acoustic-standing-wave", "standing"},
            {"--qualification-acoustic-pulse", "pulse"},
        };
        for (const auto& [argument, name] : hooks)
        {
            const CapturedCommand result =
                run_and_capture("bin/nimbus_sim " + argument, "nimbus_cli_internal_acoustic_" + name + ".log");
            REQUIRE(result.status != -1);
            REQUIRE(WIFEXITED(result.status));
            REQUIRE(WEXITSTATUS(result.status) == 1);
            REQUIRE(result.output.find("only in bin/nimbus_sim_agrid_qualification") != std::string::npos);
        }
    }

    SECTION("explicit live transport rejects unsupported fields at startup")
    {
        const CapturedCommand result =
            run_and_capture("bin/nimbus_sim --headless "
                            "--config tests/fixtures/configs/test/acceptance/t0_cart_calm.yaml --duration=0 "
                            "--live-shm --live-shm-fields=definitely_not_a_live_field",
                            "nimbus_cli_bad_live_field.log");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find("unsupported live shared-memory field") != std::string::npos);
        REQUIRE(result.output.find("refusing to start") != std::string::npos);
    }

    SECTION("live transport rejects unsafe endpoint names")
    {
        const CapturedCommand result = run_and_capture(
            "bin/nimbus_sim --live-shm-name=/bad/name",
            "nimbus_cli_bad_live_shm_name.log");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find("Invalid --live-shm-name") != std::string::npos);
    }
}

TEST_CASE("configured scientific schemes fail closed", "[integration][configs][schemes]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping scheme-selection test");
        return;
    }

    std::ifstream input("tests/fixtures/configs/test/legacy/base_cartesian.yaml");
    REQUIRE(input.good());
    const std::string base_yaml((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());

    SECTION("unknown microphysics is rejected instead of becoming Kessler")
    {
        std::string       yaml       = base_yaml;
        const std::string configured = "microphysics:\n  scheme: kessler";
        const std::size_t pos        = yaml.find(configured);
        REQUIRE(pos != std::string::npos);
        yaml.replace(pos, configured.size(), "microphysics:\n  scheme: definitely_not_microphysics");

        const CapturedCommand result = run_config_text(yaml, "nimbus_invalid_microphysics");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find("Invalid microphysics scheme") != std::string::npos);
        REQUIRE(result.output.find("Falling back to Kessler") == std::string::npos);
    }

    SECTION("native RRTMG is selected without substitution")
    {
        std::ifstream rrtmg_input("tests/fixtures/configs/qualification/agrid_radiation_coupled.yaml");
        REQUIRE(rrtmg_input.good());
        std::string yaml((std::istreambuf_iterator<char>(rrtmg_input)), std::istreambuf_iterator<char>());
        const std::string configured = "radiation:\n  scheme: simple_grey";
        const std::size_t pos        = yaml.find(configured);
        REQUIRE(pos != std::string::npos);
        yaml.replace(pos, configured.size(), "radiation:\n  scheme: rrtmg\n  rrtmg_ozone_vmr: 4e-6");

        const CapturedCommand result = run_config_text(
            yaml, "nimbus_native_rrtmg", " --duration=0 --write-every=0 --log-profile=normal");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 0);
        REQUIRE(result.output.find("Initialized radiation scheme: rrtmg") != std::string::npos);
        REQUIRE(result.output.find("RRTMG coefficient") == std::string::npos);
        REQUIRE(result.output.find("Using default: simple_grey") == std::string::npos);
    }
}

TEST_CASE("SCI-021 turbulence and diffusion choices reach an exact runtime contract",
          "[integration][configs][schemes][sci021]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping SCI-021 runtime-boundary test");
        return;
    }

    SECTION("strict-v1 2d turbulence initializes as horizontal-only")
    {
        const CapturedCommand result = run_and_capture("bin/nimbus_sim --headless "
                                                       "--config tests/fixtures/configs/test/acceptance/t6_cart_smag2d.yaml "
                                                       "--duration=0 --write-every=0 --log-profile=normal",
                                                       "nimbus_sci021_turbulence_2d.log");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 0);
        REQUIRE(result.output.find("Mode = horizontal_only") != std::string::npos);
        REQUIRE(result.output.find("using '3d'") == std::string::npos);
    }

    std::ifstream input("tests/fixtures/configs/test/legacy/base_cartesian.yaml");
    REQUIRE(input.good());
    const std::string base_yaml((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());

    SECTION("raw legacy diffusion none is rejected during config loading")
    {
        const std::string     yaml   = base_yaml + R"yaml(
numerics:
  diffusion:
    scheme: none
    operator_order: 2
)yaml";
        const CapturedCommand result = run_config_text(yaml, "nimbus_sci021_diffusion_none");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find("Invalid numerics.diffusion.scheme 'none'") != std::string::npos);
    }

    SECTION("raw legacy sixth-order diffusion is rejected during config loading")
    {
        const std::string     yaml   = base_yaml + R"yaml(
numerics:
  diffusion:
    scheme: explicit
    operator_order: 6
)yaml";
        const CapturedCommand result = run_config_text(yaml, "nimbus_sci021_diffusion_order6");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find("numerics.diffusion.operator_order must be the implemented") != std::string::npos);
    }

    SECTION("raw legacy invalid turbulence mode is rejected instead of falling back")
    {
        const std::string     yaml   = base_yaml + R"yaml(
turbulence:
  scheme: smagorinsky
  mode: definitely_not_a_mode
)yaml";
        const CapturedCommand result = run_config_text(yaml, "nimbus_sci021_turbulence_invalid");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find("Invalid turbulence.mode") != std::string::npos);
        REQUIRE(result.output.find("using '3d'") == std::string::npos);
    }

    SECTION("raw legacy invalid turbulence cadence is rejected instead of retaining a default")
    {
        const std::string     yaml   = base_yaml + R"yaml(
turbulence:
  scheme: smagorinsky
  dt_sgs: -2
)yaml";
        const CapturedCommand result = run_config_text(yaml, "nimbus_sci083_turbulence_invalid_cadence");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find("Invalid turbulence.dt_sgs") != std::string::npos);
    }

    SECTION("strict-v1 NetCDF output follows the exact binary capability without fallback")
    {
        std::ifstream strict_input("tests/fixtures/configs/test/acceptance/t0_cart_calm.yaml");
        REQUIRE(strict_input.good());
        std::string       yaml((std::istreambuf_iterator<char>(strict_input)), std::istreambuf_iterator<char>());
        const std::string configured = "  format: npy_3d";
        const std::size_t pos        = yaml.find(configured);
        REQUIRE(pos != std::string::npos);
        yaml.replace(pos, configured.size(), "  format: netcdf");

        const CapturedCommand result =
            run_config_text(yaml, "nimbus_sci021_output_netcdf", " --duration=0 --write-every=0 --log-profile=normal");
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
#ifdef HAVE_NETCDF
        REQUIRE(WEXITSTATUS(result.status) == 0);
        REQUIRE(result.output.find("[OUTPUT] Format: netcdf") != std::string::npos);
#else
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find("compiled without libnetcdf") != std::string::npos);
#endif
        REQUIRE(result.output.find("using npy_3d") == std::string::npos);
    }
}

TEST_CASE("legacy science policy enums fail closed instead of retaining defaults",
          "[integration][configs][config-truth][science]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping legacy policy rejection test");
        return;
    }

    std::ifstream input("tests/fixtures/configs/test/legacy/base_cartesian.yaml");
    REQUIRE(input.good());
    const std::string base_yaml((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());

    const std::vector<std::pair<std::string, std::string>> invalid_values = {
        {"environment:\n  sounding:\n    interpolation_method: cubic_magic\n",
         "Invalid environment.sounding.interpolation_method"},
        {"validation:\n  guard_mode: maybe_repair\n", "Invalid validation.guard_mode"},
        {"validation:\n  guard_fail_on: sometimes\n", "Invalid validation.guard_fail_on"},
        {"validation:\n  guard_scope: arbitrary_cells\n", "Invalid validation.guard_scope"},
    };

    std::size_t case_index = 0;
    for (const auto& [fragment, expected] : invalid_values)
    {
        const CapturedCommand result = run_config_text(base_yaml + "\n" + fragment,
                                                       "nimbus_invalid_science_policy_" + std::to_string(case_index++));
        INFO(result.output);
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find(expected) != std::string::npos);
    }
}

TEST_CASE("ordinary binary exposes the qualified flat A-grid CPU profile",
          "[integration][configs][agrid][runtime-selectable]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping ordinary A-grid startup test");
        return;
    }
    const std::string config_path = "tests/fixtures/configs/qualification/agrid_production_integrator_calm.yaml";
    std::ifstream     config_in(config_path);
    REQUIRE(config_in.good());
    const std::string base_yaml{std::istreambuf_iterator<char>(config_in), std::istreambuf_iterator<char>()};
    const std::string qualified_config_path =
        "tests/fixtures/configs/qualification/agrid_flat_cpu_core_ready.yaml";
    std::ifstream qualified_config_in(qualified_config_path);
    REQUIRE(qualified_config_in.good());
    const std::string qualified_yaml{
        std::istreambuf_iterator<char>(qualified_config_in),
        std::istreambuf_iterator<char>()};

    SECTION("flat exploratory CPU startup is accepted")
    {
        const CapturedCommand result = run_and_capture("bin/nimbus_sim --headless --config " + config_path +
                                                           " --duration=0 --write-every=0 --log-profile=normal",
                                                       "nimbus_agrid_ordinary_startup.log");
        INFO(result.output);
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 0);
        REQUIRE(result.output.find("development-only and unavailable") == std::string::npos);
        REQUIRE(result.output.find("[INTERNAL QUALIFICATION]") == std::string::npos);
    }

    SECTION("qualified flat CPU core is accepted")
    {
        const CapturedCommand result = run_and_capture(
            "bin/nimbus_sim --headless --config " + qualified_config_path +
                " --duration=0 --write-every=0 --log-profile=quiet",
            "nimbus_agrid_ordinary_qualified.log");
        INFO(result.output);
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 0);
        REQUIRE(result.output.find("maturity=compatibility") == std::string::npos);
    }

    SECTION("domain-aware Kessler is accepted on collocated CPU")
    {
        const std::string yaml =
            replace_once(base_yaml, "microphysics:\n  scheme: none", "microphysics:\n  scheme: kessler");
        const CapturedCommand result = run_config_text(
            yaml, "nimbus_agrid_ordinary_microphysics", " --duration=2 --write-every=0 --log-profile=quiet");
        INFO(result.output);
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 0);
        REQUIRE(result.output.find("does not support staggering=collocated") == std::string::npos);
    }

    SECTION("domain-aware Kessler composes with declared clear-sky radiation")
    {
        std::ifstream radiation_in("tests/fixtures/configs/qualification/agrid_radiation_coupled.yaml");
        REQUIRE(radiation_in.good());
        const std::string radiation_yaml{std::istreambuf_iterator<char>(radiation_in),
                                         std::istreambuf_iterator<char>()};
        const std::string yaml =
            replace_once(radiation_yaml, "microphysics:\n  scheme: none", "microphysics:\n  scheme: kessler");
        const CapturedCommand result = run_config_text(
            yaml, "nimbus_agrid_kessler_radiation", " --duration=2 --write-every=0 --log-profile=quiet");
        INFO(result.output);
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 0);
    }

    SECTION("physical-domain Smagorinsky is accepted on collocated CPU")
    {
        const std::string yaml = replace_once(
            base_yaml, "turbulence:\n  scheme: none",
            "turbulence:\n  scheme: smagorinsky");
        const CapturedCommand result = run_config_text(
            yaml, "nimbus_agrid_smagorinsky",
            " --duration=1 --write-every=0 --checkpoint-every=0"
            " --log-profile=quiet");
        INFO(result.output);
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 0);
        REQUIRE(result.output.find("does not support staggering=collocated")
                == std::string::npos);
        REQUIRE(result.output.find("[ERROR]") == std::string::npos);
    }

    SECTION("A-grid Smagorinsky composes with stochastic transport coefficients")
    {
        std::string yaml = replace_once(
            base_yaml, "turbulence:\n  scheme: none",
            "turbulence:\n  scheme: smagorinsky");
        yaml = replace_once(
            std::move(yaml),
            "  compute:\n    backend: cpu\n",
            "  chaos:\n"
            "    scheme: full_stochastic\n"
            "    seed: 161803\n"
            "    member_id: 2\n"
            "    apply_to_tendencies: [diffusion]\n"
            "    alpha_tend:\n"
            "      diffusion: 0.1\n"
            "  compute:\n    backend: cpu\n");
        const CapturedCommand result = run_config_text(
            yaml, "nimbus_agrid_smagorinsky_stochastic",
            " --duration=1 --write-every=0 --checkpoint-every=0"
            " --log-profile=quiet");
        INFO(result.output);
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 0);
        REQUIRE(result.output.find("does not support staggering=collocated")
                == std::string::npos);
        REQUIRE(result.output.find("[ERROR]") == std::string::npos);
    }

    SECTION("physical-column PBL schemes are accepted on collocated CPU")
    {
        for (const std::string& scheme : {
                 std::string("slab"), std::string("ysu"),
                 std::string("mynn")})
        {
            DYNAMIC_SECTION("scheme=" << scheme)
            {
                std::string yaml = replace_once(
                    base_yaml,
                    "surface:\n  scheme: none\n  apply_fluxes: false",
                    "surface:\n"
                    "  scheme: slab\n"
                    "  apply_fluxes: true\n"
                    "  z0m_m: 0.1\n"
                    "  z0h_m: 0.01\n"
                    "  albedo: 0.2\n"
                    "  emissivity: 0.96\n"
                    "  tsk_k: 300");
                yaml = replace_once(
                    std::move(yaml),
                    "boundary_layer:\n  scheme: none",
                    "boundary_layer:\n  scheme: " + scheme);
                yaml = replace_once(
                    std::move(yaml),
                    "radiation:\n  scheme: none",
                    scheme == "slab"
                        ? "radiation:\n"
                          "  scheme: simple_grey\n"
                          "  solar_forcing: idealized_diurnal\n"
                          "  latitude_deg: 35\n"
                          "  declination_deg: 15\n"
                          "  local_solar_time_at_start_h: 12\n"
                          "  do_lw: true\n"
                          "  do_sw: false\n"
                          "  cadence_s: 1\n"
                          "  clear_sky_only: true"
                        : "radiation:\n"
                          "  scheme: rrtmg\n"
                          "  solar_forcing: idealized_diurnal\n"
                          "  latitude_deg: 35\n"
                          "  declination_deg: 15\n"
                          "  local_solar_time_at_start_h: 12\n"
                          "  do_lw: true\n"
                          "  do_sw: true\n"
                          "  cadence_s: 1\n"
                          "  clear_sky_only: false\n"
                          "  rrtmg_ozone_vmr: 0.000004");
                const CapturedCommand result = run_config_text(
                    yaml, "nimbus_agrid_pbl_" + scheme,
                    " --duration=1 --write-every=0 --checkpoint-every=0"
                    " --log-profile=quiet");
                INFO(result.output);
                REQUIRE(result.status != -1);
                REQUIRE(WIFEXITED(result.status));
                REQUIRE(WEXITSTATUS(result.status) == 0);
                REQUIRE(result.output.find(
                            "does not support staggering=collocated") ==
                        std::string::npos);
                REQUIRE(result.output.find("[ERROR]") == std::string::npos);
            }
        }
    }

    SECTION("qualified A-grid initial-condition chaos is in the structural CPU family")
    {
        const std::string yaml = replace_once(
            qualified_yaml,
            "  chaos:\n    scheme: none\n",
            "  chaos:\n"
            "    scheme: initial_conditions\n"
            "    seed: 314159\n"
            "    member_id: 8\n"
            "    apply_to_ic: [theta]\n"
            "    sigma_ic:\n"
            "      theta: 0.25\n");
        const CapturedCommand result = run_config_text(
            yaml, "nimbus_agrid_qualified_chaos",
            " --duration=1 --write-every=0 --checkpoint-every=0"
            " --log-profile=normal");
        INFO(result.output);
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 0);
        REQUIRE(result.output.find(
                    "Initialized chaos scheme: initial_conditions") !=
                std::string::npos);
        REQUIRE(result.output.find("IC variables: theta(") !=
                std::string::npos);
    }

    SECTION("A-grid evolving chaos composes with physical-column PBL")
    {
        for (const std::string& chaos_scheme : {
                 std::string("boundary_layer"),
                 std::string("full_stochastic")})
        {
            DYNAMIC_SECTION("scheme=" << chaos_scheme)
            {
                std::string yaml = replace_once(
                    base_yaml,
                    "surface:\n  scheme: none\n  apply_fluxes: false",
                    "surface:\n"
                    "  scheme: slab\n"
                    "  apply_fluxes: true\n"
                    "  z0m_m: 0.1\n"
                    "  z0h_m: 0.01\n"
                    "  albedo: 0.2\n"
                    "  emissivity: 0.96\n"
                    "  tsk_k: 300");
                yaml = replace_once(
                    std::move(yaml),
                    "boundary_layer:\n  scheme: none",
                    "boundary_layer:\n  scheme: slab");
                yaml = replace_once(
                    std::move(yaml),
                    "radiation:\n  scheme: none",
                    "radiation:\n"
                    "  scheme: simple_grey\n"
                    "  solar_forcing: idealized_diurnal\n"
                    "  latitude_deg: 35\n"
                    "  declination_deg: 15\n"
                    "  local_solar_time_at_start_h: 12\n"
                    "  do_lw: true\n"
                    "  do_sw: false\n"
                    "  cadence_s: 1\n"
                    "  clear_sky_only: true");
                yaml = replace_once(
                    std::move(yaml),
                    "  compute:\n    backend: cpu\n",
                    "  chaos:\n"
                    "    scheme: " + chaos_scheme + "\n"
                    "    seed: 271828\n"
                    "    member_id: 6\n"
                    "    apply_to_tendencies: [pbl]\n"
                    "    alpha_tend:\n"
                    "      pbl: 0.2\n"
                    "  compute:\n    backend: cpu\n");
                const CapturedCommand result = run_config_text(
                    yaml, "nimbus_agrid_evolving_chaos_" + chaos_scheme,
                    " --duration=1 --write-every=0 --checkpoint-every=0"
                    " --log-profile=quiet");
                INFO(result.output);
                REQUIRE(result.status != -1);
                REQUIRE(WIFEXITED(result.status));
                REQUIRE(WEXITSTATUS(result.status) == 0);
                REQUIRE(result.output.find(
                            "does not support staggering=collocated") ==
                        std::string::npos);
                REQUIRE(result.output.find("[ERROR]") == std::string::npos);
            }
        }
    }

    SECTION("exploratory A-grid GPU request follows ordinary backend policy")
    {
        const std::string exploratory_yaml = replace_once(
            base_yaml,
            "  execution:\n    mode: qualified",
            "  execution:\n    mode: exploratory");
        const CapturedCommand result = run_config_text(
            exploratory_yaml, "nimbus_agrid_ordinary_gpu",
            " --duration=0 --write-every=0 --compute-backend=gpu"
            " --log-profile=quiet");
        INFO(result.output);
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        if (WEXITSTATUS(result.status) != 0)
        {
            REQUIRE(WEXITSTATUS(result.status) == 1);
            REQUIRE(result.output.find(
                        "compute backend pre-allocation initialization failed") !=
                    std::string::npos);
        }
        REQUIRE(result.output.find("does not support backend=gpu") == std::string::npos);
        REQUIRE(result.output.find("fallback=active") == std::string::npos);
    }

    SECTION("validate-only evaluates the effective CLI backend")
    {
        const std::string exploratory_yaml = replace_once(
            base_yaml,
            "  execution:\n    mode: qualified",
            "  execution:\n    mode: exploratory");
        const CapturedCommand result = run_config_text(
            exploratory_yaml, "nimbus_agrid_validate_only_gpu",
            " --validate-only --compute-backend=gpu");
        INFO(result.output);
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 0);
        REQUIRE(result.output.find("[VALIDATE] PASS") != std::string::npos);
        REQUIRE(result.output.find("does not support backend=gpu") == std::string::npos);
    }
}

TEST_CASE("CLI cannot weaken qualified state guards", "[integration][configs][qualified][guard]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping qualified CLI guard test");
        return;
    }
    const std::string base = "bin/nimbus_sim --headless --config "
                             "tests/fixtures/configs/qualification/hydrostatic_cgrid_calm_1h.yaml "
                             "--duration=0 --write-every=0 --log-profile=normal ";

    SECTION("guard mode")
    {
        const CapturedCommand result =
            run_and_capture(base + "--guard-mode=off", "nimbus_qualified_cli_guard_mode.log");
        INFO(result.output);
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find("requires --guard-mode=strict") != std::string::npos);
    }

    SECTION("failure class")
    {
        const CapturedCommand result =
            run_and_capture(base + "--guard-fail-on=nonfinite", "nimbus_qualified_cli_guard_fail_on.log");
        INFO(result.output);
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find("requires --guard-fail-on=both") != std::string::npos);
    }
}

TEST_CASE("validate-only and runtime share qualified Kessler scope",
          "[integration][configs][qualified][kessler][preflight]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping qualified Kessler scope test");
        return;
    }

    std::ifstream source("tests/fixtures/configs/qualification/constant_velocity_cgrid.yaml");
    REQUIRE(source.good());
    std::string yaml((std::istreambuf_iterator<char>(source)),
                     std::istreambuf_iterator<char>());
    yaml = replace_once(
        std::move(yaml), "    mode: exploratory", "    mode: qualified");
    yaml = replace_once(
        std::move(yaml),
        "    acknowledge_maturity_limits: false",
        "    acknowledge_maturity_limits: true");

    const std::string expected =
        "microphysics.scheme=kessler is qualified only in an exact public reference profile";
    const CapturedCommand preflight = run_config_text(
        yaml, "nimbus_qualified_flat_kessler_preflight", " --validate-only");
    INFO(preflight.output);
    REQUIRE(preflight.status != -1);
    REQUIRE(WIFEXITED(preflight.status));
    REQUIRE(WEXITSTATUS(preflight.status) == 2);
    REQUIRE(preflight.output.find(expected) != std::string::npos);
    REQUIRE(preflight.output.find("Initialized dynamics scheme") ==
            std::string::npos);

    const CapturedCommand runtime = run_config_text(
        yaml, "nimbus_qualified_flat_kessler_runtime",
        " --duration=0 --write-every=0 --log-profile=quiet");
    INFO(runtime.output);
    REQUIRE(runtime.status != -1);
    REQUIRE(WIFEXITED(runtime.status));
    REQUIRE(WEXITSTATUS(runtime.status) == 1);
    REQUIRE(runtime.output.find(expected) != std::string::npos);
    REQUIRE(runtime.output.find("Initialized dynamics scheme") ==
            std::string::npos);

    std::ifstream nested_source(
        "tests/fixtures/configs/qualification/"
        "stationary_nested_vulkan_kessler_smagorinsky.yaml");
    REQUIRE(nested_source.good());
    const std::string canonical_nested(
        (std::istreambuf_iterator<char>(nested_source)),
        std::istreambuf_iterator<char>());
    const CapturedCommand canonical_preflight = run_config_text(
        canonical_nested, "nimbus_qualified_exact_kessler_preflight",
        " --validate-only");
    INFO(canonical_preflight.output);
    REQUIRE(canonical_preflight.status != -1);
    REQUIRE(WIFEXITED(canonical_preflight.status));
    REQUIRE(WEXITSTATUS(canonical_preflight.status) == 0);

    const std::string noncanonical_nested = replace_once(
        canonical_nested,
        "    cloud_water_threshold_kgkg: 0.001",
        "    cloud_water_threshold_kgkg: 0.0011");
    const CapturedCommand nested_preflight = run_config_text(
        noncanonical_nested,
        "nimbus_qualified_noncanonical_kessler_preflight",
        " --validate-only");
    INFO(nested_preflight.output);
    REQUIRE(nested_preflight.status != -1);
    REQUIRE(WIFEXITED(nested_preflight.status));
    REQUIRE(WEXITSTATUS(nested_preflight.status) == 2);
    REQUIRE(nested_preflight.output.find(expected) != std::string::npos);

    const CapturedCommand nested_runtime = run_config_text(
        noncanonical_nested,
        "nimbus_qualified_noncanonical_kessler_runtime",
        " --duration=0 --write-every=0 --log-profile=quiet");
    INFO(nested_runtime.output);
    REQUIRE(nested_runtime.status != -1);
    REQUIRE(WIFEXITED(nested_runtime.status));
    REQUIRE(WEXITSTATUS(nested_runtime.status) == 1);
    REQUIRE(nested_runtime.output.find(expected) != std::string::npos);
    REQUIRE(nested_runtime.output.find("Initialized dynamics scheme") ==
            std::string::npos);
}

TEST_CASE("validate-only admits only the exact qualified nested radiation profile",
          "[integration][configs][qualified][radiation][preflight]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping qualified radiation preflight test");
        return;
    }

    std::ifstream source(
        "tests/fixtures/configs/qualification/stationary_nested_vulkan_radiation.yaml");
    REQUIRE(source.good());
    const std::string canonical(
        (std::istreambuf_iterator<char>(source)),
        std::istreambuf_iterator<char>());

    const CapturedCommand accepted = run_config_text(
        canonical, "nimbus_qualified_exact_radiation_preflight",
        " --validate-only");
    INFO(accepted.output);
    REQUIRE(accepted.status != -1);
    REQUIRE(WIFEXITED(accepted.status));
    REQUIRE(WEXITSTATUS(accepted.status) == 0);

    const struct
    {
        const char* name;
        const char* from;
        const char* to;
    } neighbors[] = {
        {"optical_depth", "  tau_lw_ref: 2.5", "  tau_lw_ref: 2.6"},
        {"shortwave", "  do_sw: false", "  do_sw: true"},
        {"surface_emissivity", "  emissivity: 0.96", "  emissivity: 0.95"},
        {"radiation_cadence", "  cadence_s: 2", "  cadence_s: 3"},
        {"fallback", "    allow_fallback: false", "    allow_fallback: true"},
        {"turbulence", "  scheme: none\n  mode: 3d", "  scheme: smagorinsky\n  mode: 3d"},
    };
    const std::string expected =
        "domain_motion.nesting is limited to the exact stationary "
        "simple_grey longwave reference profile";
    for (const auto& neighbor : neighbors)
    {
        CAPTURE(neighbor.name);
        const std::string changed = replace_once(
            canonical, neighbor.from, neighbor.to);
        const CapturedCommand rejected = run_config_text(
            changed,
            std::string("nimbus_qualified_radiation_neighbor_") +
                neighbor.name,
            " --validate-only");
        INFO(rejected.output);
        REQUIRE(rejected.status != -1);
        REQUIRE(WIFEXITED(rejected.status));
        REQUIRE(WEXITSTATUS(rejected.status) == 2);
        REQUIRE(rejected.output.find(expected) != std::string::npos);
    }
}

TEST_CASE("qualified flat constant-K diffusion admits native Vulkan preflight",
          "[integration][configs][qualified][diffusion][vulkan][preflight]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping qualified Vulkan diffusion preflight test");
        return;
    }

    std::ifstream source(
        "tests/fixtures/configs/qualification/hydrostatic_cgrid_calm_1h.yaml");
    REQUIRE(source.good());
    std::string yaml((std::istreambuf_iterator<char>(source)),
                     std::istreambuf_iterator<char>());
    yaml = replace_once(std::move(yaml), "  K_h: 0\n", "  K_h: 1\n");

    const CapturedCommand accepted = run_config_text(
        yaml, "nimbus_qualified_vulkan_active_diffusion_preflight",
        " --validate-only --compute-backend=vulkan");
    INFO(accepted.output);
    REQUIRE(accepted.status != -1);
    REQUIRE(WIFEXITED(accepted.status));
    REQUIRE(WEXITSTATUS(accepted.status) == 0);
    REQUIRE(accepted.output.find("physical-device qualification pending") ==
            std::string::npos);
}

TEST_CASE("CLI sounding replacement participates in pre-allocation validation", "[integration][configs][sounding][cli]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping CLI sounding preflight test");
        return;
    }
    const std::string sounding = "tests/fixtures/configs/qualification/soundings/dry_isothermal_profile.txt";

    SECTION("validate-only builds the effective file column without a config argument")
    {
        const CapturedCommand result = run_and_capture("bin/nimbus_sim --validate-only --sounding " + sounding,
                                                       "nimbus_cli_sounding_validate_only.log");
        INFO(result.output);
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 0);
        REQUIRE(result.output.find("file/auto initial-condition column built") != std::string::npos);
        REQUIRE(result.output.find("Auto-detected sounding reader: sharpy") != std::string::npos);
        REQUIRE(result.output.find("no 3-D simulation fields allocated") != std::string::npos);
    }

    SECTION("ordinary A-grid startup records the overridden file source")
    {
        const CapturedCommand result =
            run_and_capture("bin/nimbus_sim --headless --config "
                            "tests/fixtures/configs/qualification/agrid_production_integrator_calm.yaml "
                            "--sounding " +
                                sounding + " --duration=0 --write-every=0 --log-profile=normal",
                            "nimbus_cli_sounding_agrid.log");
        INFO(result.output);
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 0);
        REQUIRE(result.output.find("Sounding source: file") != std::string::npos);
        REQUIRE(result.output.find("override preflighted as type=file") != std::string::npos);
    }

    SECTION("missing file fails before configuration completion")
    {
        const CapturedCommand result = run_and_capture("bin/nimbus_sim --headless --config "
                                                       "tests/fixtures/configs/qualification/agrid_production_integrator_calm.yaml "
                                                       "--sounding /private/tmp/nimbus-definitely-missing-sounding.txt "
                                                       "--duration=0 --write-every=0 --log-profile=normal",
                                                       "nimbus_cli_sounding_missing.log");
        INFO(result.output);
        REQUIRE(result.status != -1);
        REQUIRE(WIFEXITED(result.status));
        REQUIRE(WEXITSTATUS(result.status) == 1);
        REQUIRE(result.output.find("is not a regular file") != std::string::npos);
        REQUIRE(result.output.find("CONFIGURATION LOADED") == std::string::npos);
        REQUIRE(result.output.find("Initialized dynamics scheme") == std::string::npos);
    }
}

TEST_CASE("Strict YAML directly preflights native CM1 input_sounding",
          "[integration][configs][sounding][cm1][reference]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping native CM1 YAML preflight");
        return;
    }

    const CapturedCommand result = run_and_capture("bin/nimbus_sim --validate-only --config "
                                                   "tests/init/fixtures/validate_only_cm1_sounding.yaml",
                                                   "nimbus_native_cm1_sounding_validate_only.log");
    INFO(result.output);
    REQUIRE(result.status != -1);
    REQUIRE(WIFEXITED(result.status));
    REQUIRE(WEXITSTATUS(result.status) == 0);
    REQUIRE(result.output.find("Auto-detected sounding reader: cm1") != std::string::npos);
    REQUIRE(result.output.find("file/auto initial-condition column built") != std::string::npos);
    REQUIRE(result.output.find("[VALIDATE] PASS") != std::string::npos);
    REQUIRE(result.output.find("no 3-D simulation fields allocated") != std::string::npos);

    const CapturedCommand runtime = run_and_capture("bin/nimbus_sim --headless --config "
                                                    "tests/init/fixtures/validate_only_cm1_sounding.yaml "
                                                    "--duration=0 --write-every=0 --log-profile=normal",
                                                    "nimbus_native_cm1_sounding_runtime.log");
    INFO(runtime.output);
    REQUIRE(runtime.status != -1);
    REQUIRE(WIFEXITED(runtime.status));
    REQUIRE(WEXITSTATUS(runtime.status) == 0);
    REQUIRE(runtime.output.find("Base state initialized via file/auto") != std::string::npos);
    REQUIRE(runtime.output.find("NaN count: 0, Inf count: 0") != std::string::npos);
    REQUIRE(runtime.output.find("Initialized dynamics scheme: klemp_wilhelmson_split_cartesian_cgrid") !=
            std::string::npos);
}

TEST_CASE("legacy runtime uses the shared surface ownership preflight",
          "[integration][configs][legacy][surface][preflight]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping legacy surface preflight test");
        return;
    }
    const std::string     yaml = R"yaml(
grid:
  coordinate_system: cartesian
  staggering: c_grid
  nx: 16
  ny: 16
  nz: 16
  dx: 1000
  dy: 1000
  dz: 500
  dt: 0.2
duration_s: 0
dynamics:
  scheme: klemp_wilhelmson_split_cartesian_cgrid
microphysics:
  scheme: none
radiation:
  scheme: none
boundary_layer:
  scheme: slab
  apply_surface_fluxes: true
surface:
  scheme: none
turbulence:
  scheme: none
numerics:
  advection:
    scheme: tvd
  diffusion:
    scheme: explicit
    K_h: 0
    K_v: 0
  time_stepping:
    scheme: split_explicit
  compute:
    backend: cpu
terrain:
  scheme: none
chaos:
  scheme: none
environment:
  sounding:
    type: parametric_cape
)yaml";
    const CapturedCommand result =
        run_config_text(yaml, "nimbus_legacy_surface_owner", " --write-every=0 --log-profile=quiet");
    INFO(result.output);
    REQUIRE(result.status != -1);
    REQUIRE(WIFEXITED(result.status));
    REQUIRE(WEXITSTATUS(result.status) == 1);
    REQUIRE(result.output.find("surface.apply_fluxes cannot be true when surface.scheme=none") != std::string::npos);
    REQUIRE(result.output.find("Initialized dynamics scheme") == std::string::npos);
}

TEST_CASE("SCI-114: strict active C-grid terrain survives projection and preflight",
          "[integration][configs][terrain][preflight][sci094]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping strict terrain preflight test");
        return;
    }
    const std::string     yaml   = R"yaml(
schema: nimbus.v1
grid:
  coordinate_system: cartesian
  staggering: c_grid
  nx: 16
  ny: 16
  nz: 16
  dx: 1000
  dy: 1000
  dz: 500
  ztop: 8000
time:
  run_time: 1
  dt_large: 0.2
  output_interval: 0
terrain:
  scheme: bell
  coordinate: btf
  compute_derivatives: true
  compute_metrics: true
  bell:
    h0: 900
    a: 4000
boundary_layer:
  scheme: none
turbulence:
  scheme: smagorinsky
  mode: 3d
radiation:
  scheme: none
surface:
  scheme: none
  apply_fluxes: false
)yaml";
    const CapturedCommand result = run_config_text(yaml, "nimbus_strict_terrain_preflight", " --validate-only");
    INFO(result.output);
    REQUIRE(result.status != -1);
    REQUIRE(WIFEXITED(result.status));
    REQUIRE(WEXITSTATUS(result.status) == 0);
    REQUIRE(result.output.find("[VALIDATE] PASS") != std::string::npos);
    REQUIRE(result.output.find("Initialized dynamics scheme") == std::string::npos);
}

TEST_CASE("SCI-275: strict validation and runtime admit mapped terrain shortwave",
          "[integration][configs][terrain][preflight][radiation][sci275]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping terrain interaction parity test");
        return;
    }

    namespace fs          = std::filesystem;
    const fs::path source = "tests/fixtures/configs/qualification/cgrid_terrain_smagorinsky_coupled.yaml";
    REQUIRE(fs::exists(source));

    std::ifstream input(source);
    std::string   yaml((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    yaml = replace_once(std::move(yaml), "  do_sw: false", "  do_sw: true");
    yaml = replace_once(
        std::move(yaml),
        "  local_solar_time_at_start_h: 0",
        "  local_solar_time_at_start_h: 12");

    const CapturedCommand validate_only =
        run_config_text(yaml, "nimbus_strict_terrain_shortwave_validate", " --validate-only");
    INFO(validate_only.output);
    REQUIRE(validate_only.status != -1);
    REQUIRE(WIFEXITED(validate_only.status));
    REQUIRE(WEXITSTATUS(validate_only.status) == 0);
    REQUIRE(validate_only.output.find("[VALIDATE] PASS") != std::string::npos);
    REQUIRE(validate_only.output.find("Initialized dynamics scheme") == std::string::npos);

    const CapturedCommand runtime = run_config_text(
        yaml, "nimbus_strict_terrain_shortwave_runtime", " --duration=1 --write-every=0 --log-profile=quiet");
    INFO(runtime.output);
    REQUIRE(runtime.status != -1);
    REQUIRE(WIFEXITED(runtime.status));
    REQUIRE(WEXITSTATUS(runtime.status) == 0);
    REQUIRE(runtime.output.find("[ERROR]") == std::string::npos);
}

TEST_CASE("translating mapped C-grid PBL traction composes with open horizontal SGS",
          "[integration][configs][terrain][boundary-layer][sgs][domain-motion]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping mapped PBL engineering smoke");
        return;
    }
    const std::string config =
        "tests/fixtures/configs/test/cgrid_terrain_pbl_engineering.yaml";
    REQUIRE(file_exists(config));

    for (int threads : {1, 4})
    {
        const CapturedCommand runtime = run_and_capture(
            "bin/nimbus_sim --headless --config " + config
                + " --duration=1 --write-every=0 --checkpoint-every=0"
                  " --log-profile=quiet --threads "
                + std::to_string(threads),
            "nimbus_mapped_pbl_t" + std::to_string(threads) + ".log");
        INFO("threads=" << threads << "\n" << runtime.output);
        REQUIRE(runtime.status != -1);
        REQUIRE(WIFEXITED(runtime.status));
        REQUIRE(WEXITSTATUS(runtime.status) == 0);
        REQUIRE(runtime.output.find("[ERROR]") == std::string::npos);
    }
}

TEST_CASE("SCI-121: native CM1 sounding composes with the retained terrain physics stack",
          "[integration][configs][terrain][sounding][cm1][sci121]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping coupled CM1 terrain test");
        return;
    }
    REQUIRE(file_exists("tests/fixtures/configs/qualification/cgrid_terrain_smagorinsky_coupled.yaml"));
    REQUIRE(file_exists("tests/fixtures/soundings/cm1/HIGHCAPENEW_QUARTERCIRCLE"));

    const CapturedCommand result =
        run_and_capture("bin/nimbus_sim --headless "
                        "--config tests/fixtures/configs/qualification/cgrid_terrain_smagorinsky_coupled.yaml "
                        "--sounding tests/fixtures/soundings/cm1/HIGHCAPENEW_QUARTERCIRCLE "
                        "--duration=1 --write-every=0 --checkpoint-every=0 "
                        "--log-profile=quiet --guard-mode=strict --guard-fail-on=both",
                        "nimbus_cm1_terrain_coupled.log");
    INFO(result.output);
    REQUIRE(result.status != -1);
    REQUIRE(WIFEXITED(result.status));
    REQUIRE(WEXITSTATUS(result.status) == 0);
}

TEST_CASE("mapped terrain composes with Earth-addressed stochastic physics",
          "[integration][configs][terrain][chaos][coupling]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping terrain chaos coupling test");
        return;
    }

    std::ifstream input(
        "tests/fixtures/configs/qualification/cgrid_terrain_smagorinsky_coupled.yaml");
    REQUIRE(input.good());
    std::string yaml(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    yaml = replace_once(
        std::move(yaml), "    mode: qualified\n", "    mode: exploratory\n");
    yaml = replace_once(
        std::move(yaml),
        "  checkpoint:\n",
        "  chaos:\n    scheme: full_stochastic\n    seed: 324\n"
        "    member_id: 2\n"
        "  checkpoint:\n");

    const CapturedCommand validate_only = run_config_text(
        yaml, "nimbus_terrain_chaos_validate", " --validate-only");
    INFO(validate_only.output);
    REQUIRE(validate_only.status != -1);
    REQUIRE(WIFEXITED(validate_only.status));
    REQUIRE(WEXITSTATUS(validate_only.status) == 0);
    REQUIRE(validate_only.output.find("[VALIDATE] PASS") != std::string::npos);

    const CapturedCommand runtime = run_config_text(
        yaml, "nimbus_terrain_chaos_runtime",
        " --duration=1 --write-every=0 --checkpoint-every=0 --log-profile=normal");
    INFO(runtime.output);
    REQUIRE(runtime.status != -1);
    REQUIRE(WIFEXITED(runtime.status));
    REQUIRE(WEXITSTATUS(runtime.status) == 0);
    REQUIRE(runtime.output.find("Initialized chaos scheme: full_stochastic")
            != std::string::npos);
    REQUIRE(runtime.output.find("[ERROR]") == std::string::npos);
}

TEST_CASE("mapped terrain runs active explicit scalar and momentum diffusion",
          "[integration][configs][terrain][diffusion][explicit][momentum]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping terrain explicit diffusion test");
        return;
    }
    std::ifstream input(
        "tests/fixtures/configs/qualification/cgrid_terrain_smagorinsky_coupled.yaml");
    REQUIRE(input.good());
    std::string yaml(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    yaml = replace_once(
        std::move(yaml), "    mode: qualified\n", "    mode: exploratory\n");
    yaml = replace_once(std::move(yaml), "  K_h: 0\n", "  K_h: 2\n");
    yaml = replace_once(std::move(yaml), "  K_v: 0\n", "  K_v: 3\n");
    yaml = replace_once(
        std::move(yaml), "  apply_to: scalars\n",
        "  apply_to: scalars_and_momentum\n");

    const CapturedCommand validate = run_config_text(
        yaml, "nimbus_terrain_explicit_momentum_validate", " --validate-only");
    INFO(validate.output);
    REQUIRE(validate.status != -1);
    REQUIRE(WIFEXITED(validate.status));
    REQUIRE(WEXITSTATUS(validate.status) == 0);
    REQUIRE(validate.output.find("[VALIDATE] PASS") != std::string::npos);

    const CapturedCommand runtime = run_config_text(
        yaml, "nimbus_terrain_explicit_momentum_runtime",
        " --duration=1 --write-every=0 --checkpoint-every=0"
        " --log-profile=quiet");
    INFO(runtime.output);
    REQUIRE(runtime.status != -1);
    REQUIRE(WIFEXITED(runtime.status));
    REQUIRE(WEXITSTATUS(runtime.status) == 0);
    REQUIRE(runtime.output.find("[ERROR]") == std::string::npos);
}

TEST_CASE("mapped terrain runs active implicit scalar diffusion",
          "[integration][configs][terrain][diffusion][implicit]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping terrain implicit diffusion test");
        return;
    }
    std::ifstream input(
        "tests/fixtures/configs/qualification/cgrid_terrain_smagorinsky_coupled.yaml");
    REQUIRE(input.good());
    std::string yaml(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    yaml = replace_once(
        std::move(yaml), "    mode: qualified\n", "    mode: exploratory\n");
    yaml = replace_once(
        std::move(yaml), "diffusion:\n  scheme: explicit",
        "diffusion:\n  scheme: implicit");
    yaml = replace_once(std::move(yaml), "  K_v: 0\n", "  K_v: 4\n");

    const CapturedCommand validate = run_config_text(
        yaml, "nimbus_terrain_implicit_validate", " --validate-only");
    INFO(validate.output);
    REQUIRE(validate.status != -1);
    REQUIRE(WIFEXITED(validate.status));
    REQUIRE(WEXITSTATUS(validate.status) == 0);
    REQUIRE(validate.output.find("[VALIDATE] PASS") != std::string::npos);

    const CapturedCommand runtime = run_config_text(
        yaml, "nimbus_terrain_implicit_runtime",
        " --duration=1 --write-every=0 --checkpoint-every=0"
        " --log-profile=quiet");
    INFO(runtime.output);
    REQUIRE(runtime.status != -1);
    REQUIRE(WIFEXITED(runtime.status));
    REQUIRE(WEXITSTATUS(runtime.status) == 0);
    REQUIRE(runtime.output.find("[ERROR]") == std::string::npos);
}

TEST_CASE("qualified execution admits every scoped stochastic scheme",
          "[integration][configs][qualified][chaos][preflight]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping qualified chaos admission test");
        return;
    }

    std::ifstream input(
        "tests/fixtures/configs/qualification/hydrostatic_cgrid_calm_1h.yaml");
    REQUIRE(input.good());
    std::string base_yaml(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    for (const std::string& scheme : {
             std::string("initial_conditions"),
             std::string("boundary_layer"),
             std::string("full_stochastic")})
    {
        DYNAMIC_SECTION("scheme=" << scheme)
        {
            const std::string yaml = replace_once(
                base_yaml,
                "  compute:\n    backend: cpu\n",
                "  chaos:\n    scheme: " + scheme
                    + "\n    seed: 7813\n    member_id: 4\n"
                      "  compute:\n    backend: cpu\n");

            const CapturedCommand validate_only = run_config_text(
                yaml, "nimbus_qualified_chaos_" + scheme + "_validate",
                " --validate-only");
            INFO(validate_only.output);
            REQUIRE(validate_only.status != -1);
            REQUIRE(WIFEXITED(validate_only.status));
            REQUIRE(WEXITSTATUS(validate_only.status) == 0);
            REQUIRE(validate_only.output.find("[VALIDATE] PASS")
                    != std::string::npos);
            REQUIRE(validate_only.output.find("maturity=experimental")
                    == std::string::npos);

            const CapturedCommand runtime = run_config_text(
                yaml, "nimbus_qualified_chaos_" + scheme + "_runtime",
                " --duration=1 --write-every=0 --checkpoint-every=0"
                " --log-profile=normal");
            INFO(runtime.output);
            REQUIRE(runtime.status != -1);
            REQUIRE(WIFEXITED(runtime.status));
            REQUIRE(WEXITSTATUS(runtime.status) == 0);
            REQUIRE(runtime.output.find(
                        "Initialized chaos scheme: " + scheme)
                    != std::string::npos);
            if (scheme == "initial_conditions")
            {
                REQUIRE(runtime.output.find(
                            "IC variables: u(σ=0.5) v(σ=0.5) w(σ=0.1) "
                            "theta(σ=0.5)") != std::string::npos);
                REQUIRE(runtime.output.find("qv(σ=") == std::string::npos);
            }
            REQUIRE(runtime.output.find("[ERROR]") == std::string::npos);
        }
    }
}

TEST_CASE("nimbus_sim rejects an unsupported time-stepping contract before initialization",
          "[integration][configs][timestepping]")
{
    if (!file_exists("bin/nimbus_sim"))
    {
        WARN("bin/nimbus_sim not built — skipping initialization-failure test");
        return;
    }

    std::ifstream input(
        "tests/fixtures/configs/qualification/hydrostatic_cgrid_calm_1h.yaml");
    REQUIRE(input.good());
    const std::string source(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    const std::string yaml = replace_once(
        source, "time:\n", "time:\n  integrator: rk3\n");

    const CapturedCommand result = run_config_text(
        yaml, "nimbus_bad_timestep_contract",
        " --duration=0 --write-every=0 --log-profile=quiet");
    INFO(result.output);
    REQUIRE(result.status != -1);
    REQUIRE(WIFEXITED(result.status));
    REQUIRE(WEXITSTATUS(result.status) == 1);
    REQUIRE(result.output.find(
                "qualified coupled RK integration requires a ready flat CPU "
                "C-grid or collocated A-grid profile") != std::string::npos);
    REQUIRE(result.output.find("Initialized dynamics scheme") ==
            std::string::npos);
}
