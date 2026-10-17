#include "catch2/catch.hpp"

#include "boundary_conditions/runtime.hpp"
#include "boundary_layer/runtime.hpp"
#include "chaos/runtime.hpp"
#include "compute/compute_backend.hpp"
#include "core/orchestration/dynamics/thermodynamic_closure.hpp"
#include "core/output/step_manifest.hpp"
#include "core/output/zfp_reader.hpp"
#include "core/runtime/headless_runtime.hpp"
#include "core/runtime/configuration_compiler.hpp"
#include "core/runtime/process_state.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation_state.hpp"
#include "diagnostics/conservation_budget.hpp"
#include "diagnostics/energy_budget.hpp"
#include "diagnostics/momentum_budget.hpp"
#include "diagnostics/moisture_budget.hpp"
#include "dynamics/dynamics_base.hpp"
#include "init/config/v0_to_v1.hpp"
#include "init/config/v1_loader.hpp"
#include "microphysics/runtime.hpp"
#include "numerics/runtime.hpp"
#include "radar/runtime.hpp"
#include "radiation/runtime.hpp"
#include "terrain/factory.hpp"
#include "turbulence/runtime.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <omp.h>

namespace
{

std::string read_text(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    REQUIRE(stream.good());
    return {std::istreambuf_iterator<char>(stream), {}};
}

std::vector<float> read_npy_f32_3d(const std::filesystem::path& path,
                                   std::array<int, 3>& shape)
{
    std::ifstream stream(path, std::ios::binary);
    REQUIRE(stream.good());
    char magic[6];
    stream.read(magic, sizeof(magic));
    REQUIRE(std::memcmp(magic, "\x93NUMPY", sizeof(magic)) == 0);
    unsigned char version[2];
    stream.read(reinterpret_cast<char*>(version), sizeof(version));
    REQUIRE(version[0] == 1);
    unsigned char length[2];
    stream.read(reinterpret_cast<char*>(length), sizeof(length));
    const std::size_t header_length = length[0] | (length[1] << 8);
    REQUIRE(header_length < 4096);
    std::string header(header_length, '\0');
    stream.read(header.data(), static_cast<std::streamsize>(header.size()));
    REQUIRE(header.find("'<f4'") != std::string::npos);
    const auto position = header.find("shape");
    REQUIRE(position != std::string::npos);
    const auto shape_start = header.find('(', position);
    REQUIRE(shape_start != std::string::npos);
    REQUIRE(std::sscanf(header.c_str() + shape_start, "(%d, %d, %d)",
                        &shape[0], &shape[1], &shape[2]) == 3);
    REQUIRE(shape[0] > 0);
    REQUIRE(shape[1] > 0);
    REQUIRE(shape[2] > 0);
    const std::size_t count = static_cast<std::size_t>(shape[0]) * shape[1] * shape[2];
    REQUIRE(count < 1U * 1024U * 1024U);
    std::vector<float> values(count);
    stream.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(count * sizeof(float)));
    REQUIRE(stream.good());
    REQUIRE(stream.peek() == std::char_traits<char>::eof());
    return values;
}

void close_ledgers()
{
    close_budget_csv();
    close_energy_csv();
    close_moisture_csv();
    close_momentum_csv();
    close_radiation_budget_csv();
    close_thermodynamic_source_budget_csv();
}

int run_fixture(const std::string& fixture,
                const std::filesystem::path& outdir,
                bool zfp,
                const std::string& execution,
                int workers,
                std::function<void(SimulationStorageState&)> hook = {},
                int scratch_mib = 64,
                bool before_initial_output = false,
                int requested_duration = 4,
                int requested_write_every = 4,
                int checkpoint_every = 0,
                const std::filesystem::path& resume_checkpoint = {},
                int keep_last_frames = 3,
                int minimum_free_space_mib = 4096,
                int maximum_retained_files = 256,
                const std::string& zfp_tier = "zfp_high",
                const std::string& typed_zfp_fixture = {})
{
    close_ledgers();
    shutdown_compute_backend_runtime();
    global_simulation_storage = SimulationStorageState{};
    int duration = requested_duration;
    int write_every = requested_write_every;
    std::string output_path = outdir.string();
    OutputConfig output;
    load_config(fixture, duration, write_every, output_path, &output,
                nullptr, nullptr, requested_write_every, -1, nullptr,
                false, nullptr, nullptr, requested_duration);
    if (zfp)
    {
        if (!typed_zfp_fixture.empty())
        {
            const auto loaded = tmv::init::config::load_v1_config(typed_zfp_fixture);
            REQUIRE(loaded.ok());
            RuntimeConfigurationState controls_runtime;
            output = tmv::configuration::compile_controls(
                *loaded.config, controls_runtime).output;
            REQUIRE(output.zfp_requested_format_tier == zfp_tier);
        }
        else
        {
            output.format = OutputFormat::zfp;
            output.async_io = execution == "field_parallel";
            output.zfp_mode = "accuracy";
            output.zfp_requested_format_tier = zfp_tier;
            output.zfp_tolerance = zfp_tier == "zfp_low" ? 1.0e-2 :
                zfp_tier == "zfp_medium" ? 1.0e-3 : 1.0e-4;
            output.zfp_keyframe_interval = 4;
            output.zfp_predictive_delta = true;
            output.zfp_per_field_tolerances = true;
            output.zfp_sparse_threshold = 0.0f;
            output.zfp_float16_prefilter = false;
            output.zfp_field_execution = execution;
            output.zfp_field_workers = workers;
            output.zfp_field_scratch_mib = scratch_mib;
            output.keep_last_frames = keep_last_frames;
            output.minimum_free_space_mib = minimum_free_space_mib;
            output.maximum_retained_history_files = maximum_retained_files;
        }
        resolve_output_fields(output);
        std::vector<std::string> selected_radar;
        if (global_radar_product_selection.reflectivity)
            selected_radar.push_back("reflectivity_dbz");
        if (global_radar_product_selection.radial_velocity)
            selected_radar.push_back("radial_velocity");
        if (global_radar_product_selection.zdr)
            selected_radar.push_back("zdr");
        if (global_radar_product_selection.kdp)
            selected_radar.push_back("kdp");
        if (global_radar_product_selection.rhohv)
            selected_radar.push_back("rhohv");
        merge_selected_radar_output_fields(output, selected_radar);
        for (const auto& field : output.resolved_fields)
        {
            const auto budget = qualified_zfp_field_budgets().find(field);
            if (typed_zfp_fixture.empty() &&
                budget != qualified_zfp_field_budgets().end())
                output.zfp_field_tolerances[field] =
                    budget->second.max_configured_absolute_error;
        }
    }
    initialize_microphysics(global_microphysics_scheme, global_microphysics_config);
    initialize_radar_products(global_radar_product_selection);
    initialize_terrain(global_terrain_config.scheme_id, global_terrain_config);
    initialize();
    initialize_numerics();
    tmv::chaos::initialize_chaos(global_simulation_state(),
                                  tmv::chaos::global_chaos_config);
    tmv::chaos::apply_chaos_initial_conditions(global_simulation_state());
    apply_initial_thermodynamic_closure(global_simulation_state());
    tmv::initialize_dynamics(global_dynamics_scheme_name);
    initialize_radiation(global_radiation_config.scheme_id, global_radiation_config);
    initialize_boundary_layer(global_boundary_layer_config.scheme_id,
                              global_boundary_layer_config, global_surface_config);
    initialize_turbulence(global_turbulence_config.scheme_id,
                          global_turbulence_config);
    apply_boundary_conditions(global_simulation_state());
    if (before_initial_output && hook)
        hook(global_simulation_storage);
    HeadlessRunOptions options;
    options.duration_s = duration;
    options.write_every_s = write_every;
    options.scientific_diagnostics_every_steps = 1;
    options.checkpoint_every_s = checkpoint_every;
    options.resume_checkpoint = resume_checkpoint;
    options.outdir = output_path;
    options.output_config = output;
    if (!before_initial_output)
        options.before_time_loop_test_hook = std::move(hook);
    const int status = run_headless_simulation(options);
    close_ledgers();
    return status;
}

std::set<std::string> published_manifest_fields(const std::filesystem::path& path)
{
    const auto document = read_text(path);
    const auto name = document.find("\"published_fields\"");
    REQUIRE(name != std::string::npos);
    const auto start = document.find('[', name);
    const auto end = document.find(']', start);
    REQUIRE(start != std::string::npos);
    REQUIRE(end != std::string::npos);
    const std::string array = document.substr(start, end - start + 1);
    const std::regex field_pattern("\"([^\"]+)\"");
    std::set<std::string> result;
    for (auto it = std::sregex_iterator(array.begin(), array.end(), field_pattern);
         it != std::sregex_iterator(); ++it)
        REQUIRE(result.insert((*it)[1].str()).second);
    return result;
}

std::vector<std::string> selected_products(const std::filesystem::path& config)
{
    const auto source = read_text(config);
    const auto name = source.find("  products: [");
    REQUIRE(name != std::string::npos);
    const auto start = source.find('[', name);
    const auto end = source.find(']', start);
    REQUIRE(start != std::string::npos);
    REQUIRE(end != std::string::npos);
    std::stringstream stream(source.substr(start + 1, end - start - 1));
    std::vector<std::string> products;
    std::string product;
    while (std::getline(stream, product, ','))
    {
        product.erase(std::remove_if(product.begin(), product.end(),
            [](unsigned char value) { return std::isspace(value); }), product.end());
        REQUIRE(is_reversible_zfp_radar_field_id(product));
        products.push_back(product);
    }
    REQUIRE_FALSE(products.empty());
    return products;
}

std::filesystem::path step_at_time(const std::filesystem::path& outdir,
                                   double requested_time)
{
    const std::regex time_pattern(R"("simulation_time_s": ([0-9.eE+-]+))");
    for (const auto& entry : std::filesystem::directory_iterator(outdir))
    {
        if (!entry.is_directory() ||
            entry.path().filename().string().rfind("step_", 0) != 0 ||
            !std::filesystem::is_regular_file(entry.path() / "manifest.json"))
            continue;
        const auto manifest = read_text(entry.path() / "manifest.json");
        std::smatch match;
        if (std::regex_search(manifest, match, time_pattern) &&
            std::abs(std::stod(match[1].str()) - requested_time) < 1.0e-9)
            return entry.path();
    }
    return {};
}

std::vector<float> decode_zfp_at_time(const std::filesystem::path& outdir,
                                      const std::string& field,
                                      double requested_time)
{
    std::vector<std::filesystem::path> steps;
    for (const auto& entry : std::filesystem::directory_iterator(outdir))
        if (entry.is_directory() &&
            entry.path().filename().string().rfind("step_", 0) == 0 &&
            std::filesystem::is_regular_file(entry.path() / "manifest.json"))
            steps.push_back(entry.path());
    std::sort(steps.begin(), steps.end());
    ZfpFrameReader reader;
    for (const auto& step : steps)
    {
        Zfp3dHeader header;
        std::vector<float> data;
        std::string error;
        REQUIRE(reader.read_frame(field, (step / (field + ".zfp3d")).string(),
                                  header, data, error));
        if (step == step_at_time(outdir, requested_time))
            return data;
    }
    return {};
}

std::string last_csv_row(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    REQUIRE(stream.good());
    std::string row;
    std::string last;
    while (std::getline(stream, row))
        if (!row.empty()) last = row;
    return last;
}

std::string contract_generation(const std::string& document)
{
    const std::string key = "\"publication_generation\": \"";
    const auto begin = document.find(key);
    if (begin == std::string::npos) return {};
    const auto first = begin + key.size();
    const auto end = document.find('"', first);
    return end == std::string::npos ? std::string{} : document.substr(first, end - first);
}

void set_history_index_failure(bool enabled)
{
#if defined(_WIN32)
    _putenv_s("NIMBUS_TEST_HISTORY_INDEX_FAIL_AFTER_MANIFEST", enabled ? "1" : "");
#else
    if (enabled) setenv("NIMBUS_TEST_HISTORY_INDEX_FAIL_AFTER_MANIFEST", "1", 1);
    else unsetenv("NIMBUS_TEST_HISTORY_INDEX_FAIL_AFTER_MANIFEST");
#endif
}

} // namespace

TEST_CASE("typed ZFP tier projection matches strict v1 serialized output controls",
          "[core][zfp][v4][projection]")
{
    for (const auto& [format, token] :
         {std::pair<tmv::init::config::OutputFormat, const char*>{
              tmv::init::config::OutputFormat::ZfpLow, "zfp_low"},
          {tmv::init::config::OutputFormat::ZfpMedium, "zfp_medium"},
          {tmv::init::config::OutputFormat::ZfpHigh, "zfp_high"}})
    {
        tmv::init::config::ConfigV1 config;
        config.time.run_time = 4.0;
        config.time.output_interval = 4.0;
        config.output.format = format;
        config.output.async_io = false;
        config.output.zfp_tolerance = 2.0e-5;
        config.output.zfp_mode = "accuracy";
        config.output.zfp_keyframe_interval = 4;
        config.output.zfp_predictive_delta = true;
        config.output.zfp_field_execution = "field_parallel";
        config.output.zfp_field_workers = 4;
        config.output.zfp_field_scratch_mib = 64;
        config.output.zfp_per_field_tolerances = true;
        config.output.zfp_field_tolerances = {{"theta", 1.0e-3}};
        config.output.keep_last_frames = 3;
        config.output.minimum_free_space_mib = 4096;
        config.output.maximum_retained_history_files = 256;
        RuntimeConfigurationState runtime;
        const auto typed = tmv::configuration::compile_controls(config, runtime).output;
        const auto serialized = tmv::init::config::serialize_v1_to_v0_map(config);
        const auto strict = parse_output_config(serialized);
        REQUIRE(typed.format == OutputFormat::zfp);
        REQUIRE(strict.format == typed.format);
        REQUIRE(typed.zfp_requested_format_tier == token);
        REQUIRE(strict.zfp_requested_format_tier == typed.zfp_requested_format_tier);
        REQUIRE(strict.zfp_tolerance == typed.zfp_tolerance);
        REQUIRE(strict.async_io == typed.async_io);
        REQUIRE(strict.zfp_mode == typed.zfp_mode);
        REQUIRE(strict.zfp_keyframe_interval == typed.zfp_keyframe_interval);
        REQUIRE(strict.zfp_predictive_delta == typed.zfp_predictive_delta);
        REQUIRE(strict.zfp_field_execution == typed.zfp_field_execution);
        REQUIRE(strict.zfp_field_workers == typed.zfp_field_workers);
        REQUIRE(strict.zfp_field_scratch_mib == typed.zfp_field_scratch_mib);
        REQUIRE(strict.zfp_per_field_tolerances == typed.zfp_per_field_tolerances);
        REQUIRE(strict.zfp_field_tolerances == typed.zfp_field_tolerances);
        REQUIRE(strict.keep_last_frames == typed.keep_last_frames);
        REQUIRE(strict.minimum_free_space_mib == typed.minimum_free_space_mib);
        REQUIRE(strict.maximum_retained_history_files ==
                typed.maximum_retained_history_files);
    }
}

TEST_CASE("guarded typed all-five radar ZFP headless composition matches NPY",
          "[core][zfp][v4][typed_headless]")
{
    const char* npy_env = std::getenv("NIMBUS_TEST_RADAR_V4_TYPED_NPY_CONFIG");
    const char* zfp_env = std::getenv("NIMBUS_TEST_RADAR_V4_TYPED_ZFP_CONFIG");
    const char* root_env = std::getenv("NIMBUS_TEST_RADAR_V4_TYPED_ROOT");
    if (!npy_env && !zfp_env && !root_env)
    {
        return;
    }
    REQUIRE(npy_env != nullptr);
    REQUIRE(zfp_env != nullptr);
    REQUIRE(root_env != nullptr);
    const std::filesystem::path root(root_env);
    REQUIRE(std::filesystem::create_directory(root));
    const auto selected = selected_products(zfp_env);
    REQUIRE(selected.size() == 5);
    omp_set_num_threads(1);
    const auto npy = root / "npy";
    const auto zfp = root / "typed_zfp_high";
    REQUIRE(run_fixture(npy_env, npy, false, "serial", 1) == 0);
    REQUIRE(run_fixture(npy_env, zfp, true, "serial", 1,
                        {}, 64, false, 4, 4, 0, {}, 3, 4096,
                        256, "zfp_high", zfp_env) == 0);
    for (const auto* ledger : {"budget.csv", "momentum_budget.csv", "energy_budget.csv",
                               "radiation_budget.csv", "thermodynamic_source_budget.csv",
                               "moisture_budget.csv"})
    {
        REQUIRE(read_text(npy / ledger) == read_text(zfp / ledger));
    }
    std::map<std::string, ZfpFrameReader> readers;
    for (const auto* step : {"step_000000", "step_000001"})
    {
        const auto npy_step = npy / step;
        const auto zfp_step = zfp / step;
        REQUIRE(published_manifest_fields(npy_step / "manifest.json") ==
                published_manifest_fields(zfp_step / "manifest.json"));
        for (const auto& field : published_manifest_fields(npy_step / "manifest.json"))
        {
            std::array<int, 3> shape{};
            const auto reference = read_npy_f32_3d(npy_step / (field + ".npy"), shape);
            Zfp3dHeader header;
            std::vector<float> decoded;
            std::string error;
            REQUIRE(readers[field].read_frame(
                field, (zfp_step / (field + ".zfp3d")).string(),
                header, decoded, error));
            REQUIRE(decoded.size() == reference.size());
            INFO(std::string(step) + ":" + field);
            if (is_reversible_zfp_radar_field_id(field))
            {
                REQUIRE(header.version == 4);
                REQUIRE(std::memcmp(decoded.data(), reference.data(),
                                    decoded.size() * sizeof(float)) == 0);
            }
            else
            {
                REQUIRE(header.version == 3);
                REQUIRE(qualified_zfp_field_budgets().count(field) == 1);
                double maximum_error = 0.0;
                for (std::size_t value = 0; value < decoded.size(); ++value)
                {
                    maximum_error = std::max(maximum_error,
                        std::abs(static_cast<double>(decoded[value]) - reference[value]));
                }
                REQUIRE(maximum_error <= qualified_zfp_field_budgets().at(field)
                    .max_configured_absolute_error);
            }
        }
    }
    REQUIRE(read_text(zfp / "nimbus_output.json").find(
                "\"publication_state\": \"ready\"") != std::string::npos);
}

TEST_CASE("nonpublic radar v4 headless candidate preserves NPY science and six ledgers",
          "[core][zfp][v4][integration]")
{
    const char* root_env = std::getenv("NIMBUS_TEST_RADAR_V4_ROOT");
    const auto root = root_env
        ? std::filesystem::path(root_env)
        : std::filesystem::temp_directory_path() /
            ("nimbus_radar_v4_integrated_" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
    REQUIRE(std::filesystem::create_directory(root));
    std::cout << "RADAR_V4_INTEGRATION_ROOT=" << root << '\n';
    const std::array<const char*, 2> fixture_names{
        "cgrid_radar_ready", "cgrid_radar_polarimetric_ready"};
    const std::array<const char*, 6> ledgers{
        "budget.csv", "momentum_budget.csv", "energy_budget.csv",
        "radiation_budget.csv", "thermodynamic_source_budget.csv",
        "moisture_budget.csv"};
    for (const auto* name : fixture_names)
    {
        const std::string fixture = std::string(
            "tests/fixtures/configs/qualification/") + name + ".yaml";
        const auto npy = root / (std::string(name) + "_npy");
        const auto serial = root / (std::string(name) + "_zfp_serial");
        const auto parallel = root / (std::string(name) + "_zfp_parallel");
        REQUIRE(run_fixture(fixture, npy, false, "serial", 1) == 0);
        REQUIRE(run_fixture(fixture, serial, true, "serial", 1) == 0);
        REQUIRE(run_fixture(fixture, parallel, true, "field_parallel", 4) == 0);
        for (const auto* ledger : ledgers)
        {
            REQUIRE(read_text(npy / ledger) == read_text(serial / ledger));
            REQUIRE(read_text(npy / ledger) == read_text(parallel / ledger));
            REQUIRE(read_text(npy / ledger).find('\n') != std::string::npos);
        }
        const auto& budgets = qualified_zfp_field_budgets();
        const std::array<const char*, 2> steps{"step_000000", "step_000001"};
        std::map<std::string, ZfpFrameReader> readers;
        for (const auto* step : steps)
        {
            const auto manifest = read_text(serial / step / "manifest.json");
            REQUIRE(manifest.find("\"compression_contract\": \"nimbus.zfp.per_field.v2\"") !=
                    std::string::npos);
            REQUIRE(manifest.find("\"lossy_compression_contract\"") == std::string::npos);
            REQUIRE(manifest.find("\"publication_state\": \"complete\"") != std::string::npos);
            for (const auto& entry : std::filesystem::directory_iterator(npy / step))
            {
                if (entry.path().extension() != ".npy")
                    continue;
                const std::string field = entry.path().stem().string();
                std::array<int, 3> shape{};
                const auto reference = read_npy_f32_3d(entry.path(), shape);
                Zfp3dHeader header;
                std::vector<float> decoded;
                std::string error;
                const auto serial_path = serial / step / (field + ".zfp3d");
                const auto parallel_path = parallel / step / (field + ".zfp3d");
                REQUIRE(readers[field].read_frame(
                    field, serial_path.string(), header, decoded, error));
                REQUIRE(decoded.size() == reference.size());
                REQUIRE(read_text(serial_path) == read_text(parallel_path));
                if (is_reversible_zfp_radar_field_id(field))
                {
                    REQUIRE(header.version == 4);
                    REQUIRE(header.mode == 3);
                    REQUIRE(header.flags == 0);
                    REQUIRE(std::memcmp(decoded.data(), reference.data(),
                                        decoded.size() * sizeof(float)) == 0);
                }
                else
                {
                    REQUIRE(header.version == 3);
                    REQUIRE(header.mode == 0);
                    REQUIRE(budgets.count(field) == 1);
                    double maximum_error = 0.0;
                    for (std::size_t value = 0; value < decoded.size(); ++value)
                        maximum_error = std::max(maximum_error,
                            std::abs(static_cast<double>(decoded[value]) - reference[value]));
                    INFO(std::string(name) + ":" + step + ":" + field);
                    REQUIRE(maximum_error <= budgets.at(field).max_configured_absolute_error);
                }
            }
        }
        for (const auto& outdir : {npy, serial, parallel})
            REQUIRE(read_text(outdir / "nimbus_output.json").find(
                        "\"publication_state\": \"ready\"") != std::string::npos);
    }

    const auto emitted = root / "cgrid_radar_ready_zfp_serial" /
        "step_000000" / "reflectivity_dbz.zfp3d";
    const auto corrupt = root / "corrupt_reflectivity.zfp3d";
    REQUIRE(std::filesystem::copy_file(emitted, corrupt));
    {
        std::fstream stream(corrupt, std::ios::binary | std::ios::in | std::ios::out);
        REQUIRE(stream.good());
        stream.seekg(44);
        char payload_byte = 0;
        stream.get(payload_byte);
        REQUIRE(stream.good());
        stream.seekp(44);
        stream.put(static_cast<char>(payload_byte ^ 1));
        REQUIRE(stream.good());
    }
    Zfp3dHeader corrupt_header;
    corrupt_header.version = 99;
    std::vector<float> untouched{123.0f};
    std::string corrupt_error;
    REQUIRE_FALSE(read_zfp_3d(corrupt.string(), corrupt_header,
                              untouched, corrupt_error));
    REQUIRE(corrupt_error.find("digest mismatch") != std::string::npos);
    REQUIRE(corrupt_header.version == 99);
    REQUIRE(untouched == std::vector<float>{123.0f});

    const std::string ready_fixture =
        "tests/fixtures/configs/qualification/cgrid_radar_ready.yaml";
    const auto initial_invalid = root / "invalid_initial_radar_source";
    const auto invalid_state = [](SimulationStorageState& state)
    {
        state.fields.u.fill(10000.0f);
    };
    REQUIRE(run_fixture(ready_fixture, initial_invalid, true, "serial", 1,
                        invalid_state, 64, true) != 0);
    const auto initial_failed_contract = read_text(initial_invalid / "nimbus_output.json");
    REQUIRE(initial_failed_contract.find("\"publication_state\": \"failed\"") !=
            std::string::npos);
    REQUIRE_FALSE(std::filesystem::exists(initial_invalid / "step_000000" / "manifest.json"));
    REQUIRE_FALSE(std::filesystem::exists(initial_invalid / "history_index.json"));

    const auto invalid = root / "invalid_late_radar_source";
    std::string ready_contract;
    const auto injected_invalid_source = [&](SimulationStorageState& state)
    {
        ready_contract = read_text(invalid / "nimbus_output.json");
        state.fields.u.fill(10000.0f);
    };
    REQUIRE(run_fixture(ready_fixture, invalid, true, "serial", 1,
                        injected_invalid_source) != 0);
    const auto failed_contract = read_text(invalid / "nimbus_output.json");
    REQUIRE(ready_contract.find("\"publication_state\": \"ready\"") != std::string::npos);
    REQUIRE(failed_contract.find("\"publication_state\": \"failed\"") != std::string::npos);
    const auto generation = [](const std::string& json)
    {
        const std::string key = "\"publication_generation\": \"";
        const auto begin = json.find(key);
        if (begin == std::string::npos) return std::string{};
        const auto first = begin + key.size();
        return json.substr(first, json.find('"', first) - first);
    };
    REQUIRE_FALSE(generation(ready_contract).empty());
    REQUIRE(generation(ready_contract) == generation(failed_contract));
    REQUIRE(std::filesystem::exists(invalid / "step_000000" / "manifest.json"));
    REQUIRE(std::filesystem::exists(invalid / "history_index.json"));
    REQUIRE_FALSE(std::filesystem::exists(invalid / "step_000001" / "manifest.json"));

    const auto insufficient = root / "insufficient_reversible_scratch";
    REQUIRE(run_fixture(ready_fixture, insufficient, true, "serial", 1,
                        {}, 0) != 0);
    REQUIRE_FALSE(std::filesystem::exists(insufficient / "nimbus_output.json"));
    REQUIRE_FALSE(std::filesystem::exists(insufficient / "step_000000" / "manifest.json"));
    REQUIRE_FALSE(std::filesystem::exists(insufficient / "history_index.json"));
}

TEST_CASE("nonpublic radar v4 covers every exact selected product subset",
          "[core][zfp][v4][subsets]")
{
    const char* config_env = std::getenv("NIMBUS_TEST_RADAR_V4_SUBSET_CONFIG_ROOT");
    const char* root_env = std::getenv("NIMBUS_TEST_RADAR_V4_SUBSET_RUN_ROOT");
    if (!config_env && !root_env)
        return;
    REQUIRE(config_env != nullptr);
    REQUIRE(root_env != nullptr);
    const std::filesystem::path configs(config_env);
    const std::filesystem::path root(root_env);
    REQUIRE(std::filesystem::is_directory(configs));
    REQUIRE(std::filesystem::create_directory(root));
    const auto started = std::chrono::steady_clock::now();
    const std::set<std::string> core{"u", "v", "w", "theta", "p", "rho", "qv", "qr"};
    const std::array<const char*, 6> ledgers{
        "budget.csv", "momentum_budget.csv", "energy_budget.csv",
        "radiation_budget.csv", "thermodynamic_source_budget.csv",
        "moisture_budget.csv"};
    const auto& budgets = qualified_zfp_field_budgets();
    std::array<std::filesystem::path, 2> all_five{};
    for (int index = 0; index < 31; ++index)
    {
        REQUIRE(std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count() < 240.0);
        std::ostringstream label;
        label << "products_" << std::setw(2) << std::setfill('0') << index;
        const auto config = configs / (label.str() + ".yaml");
        const auto selected = selected_products(config);
        std::set<std::string> expected = core;
        expected.insert(selected.begin(), selected.end());
        REQUIRE(expected.size() == core.size() + selected.size());
        const auto npy = root / (label.str() + "_npy");
        const auto zfp = root / (label.str() + "_zfp_serial");
        omp_set_num_threads(1);
        REQUIRE(run_fixture(config.string(), npy, false, "serial", 1) == 0);
        REQUIRE(run_fixture(config.string(), zfp, true, "serial", 1,
                            {}, 64, false) == 0);
        for (const auto* ledger : ledgers)
            REQUIRE(read_text(npy / ledger) == read_text(zfp / ledger));
        std::map<std::string, ZfpFrameReader> readers;
        std::set<std::string> active_radar;
        for (const auto* step : {"step_000000", "step_000001"})
        {
            const auto npy_step = npy / step;
            const auto zfp_step = zfp / step;
            REQUIRE(published_manifest_fields(npy_step / "manifest.json") == expected);
            REQUIRE(published_manifest_fields(zfp_step / "manifest.json") == expected);
            std::set<std::string> npy_artifacts;
            std::set<std::string> zfp_artifacts;
            for (const auto& entry : std::filesystem::directory_iterator(npy_step))
                if (entry.path().extension() == ".npy")
                    npy_artifacts.insert(entry.path().stem().string());
            for (const auto& entry : std::filesystem::directory_iterator(zfp_step))
                if (entry.path().extension() == ".zfp3d")
                    zfp_artifacts.insert(entry.path().stem().string());
            REQUIRE(npy_artifacts == expected);
            REQUIRE(zfp_artifacts == expected);
            for (const auto& field : expected)
            {
                std::array<int, 3> shape{};
                const auto reference = read_npy_f32_3d(npy_step / (field + ".npy"), shape);
                Zfp3dHeader header;
                std::vector<float> decoded;
                std::string error;
                REQUIRE(readers[field].read_frame(
                    field, (zfp_step / (field + ".zfp3d")).string(),
                    header, decoded, error));
                REQUIRE(decoded.size() == reference.size());
                INFO(label.str() + ":" + step + ":" + field);
                if (is_reversible_zfp_radar_field_id(field))
                {
                    REQUIRE(header.version == 4);
                    REQUIRE(header.mode == 3);
                    REQUIRE(header.flags == 0);
                    REQUIRE(std::memcmp(decoded.data(), reference.data(),
                                        decoded.size() * sizeof(float)) == 0);
                    if (std::any_of(reference.begin(), reference.end(),
                            [](float value) { return value != 0.0f; }))
                        active_radar.insert(field);
                }
                else
                {
                    REQUIRE(header.version == 3);
                    REQUIRE(header.mode == 0);
                    REQUIRE(budgets.count(field) == 1);
                    double maximum_error = 0.0;
                    for (std::size_t value = 0; value < decoded.size(); ++value)
                        maximum_error = std::max(maximum_error,
                            std::abs(static_cast<double>(decoded[value]) - reference[value]));
                    REQUIRE(maximum_error <=
                        budgets.at(field).max_configured_absolute_error);
                }
            }
        }
        REQUIRE(read_text(zfp / "nimbus_output.json").find(
                    "\"publication_state\": \"ready\"") != std::string::npos);
        REQUIRE(active_radar == std::set<std::string>(selected.begin(), selected.end()));
        if (selected.size() == 5)
            all_five = {npy, zfp};
        std::size_t raw_bytes = 0;
        std::size_t files = 0;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(root))
            if (entry.is_regular_file())
            {
                ++files;
                raw_bytes += entry.file_size();
            }
        REQUIRE(files < 4000);
        REQUIRE(raw_bytes < 128U * 1024U * 1024U);
    }
    REQUIRE_FALSE(all_five[0].empty());
    const auto four = root / "all_five_zfp_parallel_four";
    omp_set_num_threads(4);
    REQUIRE(run_fixture((configs / "products_30.yaml").string(), four,
                        true, "field_parallel", 4, {}, 64, false) == 0);
    for (const auto* ledger : ledgers)
        REQUIRE(read_text(all_five[0] / ledger) == read_text(four / ledger));
    for (const auto* step : {"step_000000", "step_000001"})
        for (const auto& field :
             std::set<std::string>{"u", "v", "w", "theta", "p", "rho", "qv", "qr",
                                   "reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"})
            REQUIRE(read_text(all_five[1] / step / (field + ".zfp3d")) ==
                    read_text(four / step / (field + ".zfp3d")));
    REQUIRE(std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count() < 240.0);
}

TEST_CASE("reversible radar output is byte-identical across all ZFP tier aliases",
          "[core][zfp][v4][tiers]")
{
    const char* config_env = std::getenv("NIMBUS_TEST_RADAR_V4_TIER_CONFIG");
    const char* root_env = std::getenv("NIMBUS_TEST_RADAR_V4_TIER_ROOT");
    if (!config_env && !root_env)
    {
        return;
    }
    REQUIRE(config_env != nullptr);
    REQUIRE(root_env != nullptr);
    const std::filesystem::path config(config_env);
    const std::filesystem::path root(root_env);
    REQUIRE(std::filesystem::is_regular_file(config));
    REQUIRE(selected_products(config).size() == 5);
    REQUIRE(std::filesystem::create_directory(root));
    omp_set_num_threads(1);
    const auto npy = root / "npy_control";
    REQUIRE(run_fixture(config.string(), npy, false, "serial", 1) == 0);
    const std::array<const char*, 13> fields{
        "u", "v", "w", "theta", "p", "rho", "qv", "qr",
        "reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"};
    const std::array<const char*, 6> ledgers{
        "budget.csv", "momentum_budget.csv", "energy_budget.csv",
        "radiation_budget.csv", "thermodynamic_source_budget.csv",
        "moisture_budget.csv"};
    const auto baseline = root / "zfp_high_serial";
    for (const auto* tier : {"zfp_high", "zfp_medium", "zfp_low"})
    {
        for (const auto& [policy, workers] :
             {std::pair<std::string, int>{"serial", 1},
              {"field_parallel", 4}})
        {
            omp_set_num_threads(workers);
            const auto outdir = root / (std::string(tier) + "_" + policy);
            REQUIRE(run_fixture(config.string(), outdir, true, policy, workers,
                                {}, 64, false, 4, 4, 0, {}, 3, 4096,
                                256, tier) == 0);
            for (const auto* ledger : ledgers)
            {
                REQUIRE(read_text(npy / ledger) == read_text(outdir / ledger));
            }
            std::map<std::string, ZfpFrameReader> readers;
            for (const auto* step : {"step_000000", "step_000001"})
            {
                for (const auto* field : fields)
                {
                    const auto payload = outdir / step / (std::string(field) + ".zfp3d");
                    if (outdir != baseline)
                    {
                        REQUIRE(read_text(payload) ==
                                read_text(baseline / step / (std::string(field) + ".zfp3d")));
                    }
                    std::array<int, 3> shape{};
                    const auto reference = read_npy_f32_3d(
                        npy / step / (std::string(field) + ".npy"), shape);
                    Zfp3dHeader header;
                    std::vector<float> decoded;
                    std::string error;
                    REQUIRE(readers[field].read_frame(field, payload.string(),
                                                      header, decoded, error));
                    REQUIRE(decoded.size() == reference.size());
                    INFO(std::string(tier) + ":" + policy + ":" + step + ":" + field);
                    if (is_reversible_zfp_radar_field_id(field))
                    {
                        REQUIRE(header.version == 4);
                        REQUIRE(std::memcmp(decoded.data(), reference.data(),
                                            decoded.size() * sizeof(float)) == 0);
                    }
                    else
                    {
                        REQUIRE(header.version == 3);
                        const double budget = qualified_zfp_field_budgets().at(field)
                            .max_configured_absolute_error;
                        double maximum_error = 0.0;
                        for (std::size_t value = 0; value < decoded.size(); ++value)
                        {
                            maximum_error = std::max(maximum_error,
                                std::abs(static_cast<double>(decoded[value]) - reference[value]));
                        }
                        REQUIRE(maximum_error <= budget);
                    }
                }
            }
        }
    }
}

TEST_CASE("nonpublic radar v4 restart retention and terminal failure boundaries",
          "[core][zfp][v4][continuation]")
{
    const char* config_env = std::getenv("NIMBUS_TEST_RADAR_V4_CONTINUATION_CONFIG");
    const char* root_env = std::getenv("NIMBUS_TEST_RADAR_V4_CONTINUATION_ROOT");
    if (!config_env && !root_env)
        return;
    REQUIRE(config_env != nullptr);
    REQUIRE(root_env != nullptr);
    const std::filesystem::path config(config_env);
    const std::filesystem::path root(root_env);
    REQUIRE(std::filesystem::is_regular_file(config));
    REQUIRE(selected_products(config).size() == 5);
    REQUIRE(std::filesystem::create_directory(root));
    const auto started = std::chrono::steady_clock::now();
    const std::array<const char*, 13> fields{
        "u", "v", "w", "theta", "p", "rho", "qv", "qr",
        "reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"};
    const std::array<const char*, 6> ledgers{
        "budget.csv", "momentum_budget.csv", "energy_budget.csv",
        "radiation_budget.csv", "thermodynamic_source_budget.csv",
        "moisture_budget.csv"};
    const auto run = [&](const std::filesystem::path& outdir,
                         const std::string& execution, int workers,
                         int duration, int write_every, int checkpoint_every,
                         const std::filesystem::path& resume,
                         int keep_last, int free_mib, int max_files,
                         std::function<void(SimulationStorageState&)> hook,
                         bool before_initial)
    {
        omp_set_num_threads(workers);
        return run_fixture(config.string(), outdir, true, execution, workers,
                           std::move(hook), 64, before_initial,
                           duration, write_every, checkpoint_every, resume,
                           keep_last, free_mib, max_files);
    };
    const auto continuous_serial = root / "continuous_serial";
    const auto continuous_parallel = root / "continuous_parallel";
    const auto npy_control = root / "npy_continuous";
    omp_set_num_threads(1);
    REQUIRE(run_fixture(config.string(), npy_control, false, "serial", 1,
                        {}, 64, false, 4, 1, 1) == 0);
    const auto npy_at_four = step_at_time(npy_control, 4.0);
    REQUIRE_FALSE(npy_at_four.empty());
    const auto parent_serial = root / "parent_serial";
    const auto parent_parallel = root / "parent_parallel";
    const auto resumed_serial = root / "resumed_serial";
    const auto resumed_parallel = root / "resumed_parallel";
    for (const auto& [policy, workers, continuous, parent, resumed] :
         {std::tuple<std::string, int, std::filesystem::path,
                     std::filesystem::path, std::filesystem::path>{
              "serial", 1, continuous_serial, parent_serial, resumed_serial},
          {"field_parallel", 4, continuous_parallel, parent_parallel,
           resumed_parallel}})
    {
        REQUIRE(run(continuous, policy, workers, 4, 1, 1, {}, 5, 4096, 256,
                    {}, false) == 0);
        REQUIRE(run(parent, policy, workers, 2, 1, 1, {}, 5, 4096, 256,
                    {}, false) == 0);
        std::vector<std::filesystem::path> checkpoints;
        for (const auto& entry : std::filesystem::directory_iterator(parent / "checkpoints"))
            if (entry.path().extension() == ".nimbus")
                checkpoints.push_back(entry.path());
        std::sort(checkpoints.begin(), checkpoints.end());
        REQUIRE(checkpoints.size() >= 2);
        REQUIRE(std::filesystem::is_regular_file(
            checkpoints.front().string() + ".json"));
        REQUIRE(read_text(checkpoints.front().string() + ".json").find(
                    "\"simulation_time_s\": 1") != std::string::npos);
        REQUIRE(run(resumed, policy, workers, 4, 1, 0,
                    checkpoints.front(), 5, 4096, 256, {}, false) == 0);
        REQUIRE_FALSE(step_at_time(continuous, 4.0).empty());
        REQUIRE_FALSE(step_at_time(resumed, 4.0).empty());
        for (const auto* field : fields)
        {
            const auto control = decode_zfp_at_time(continuous, field, 4.0);
            const auto continuation = decode_zfp_at_time(resumed, field, 4.0);
            std::array<int, 3> shape{};
            const auto reference = read_npy_f32_3d(
                npy_at_four / (std::string(field) + ".npy"), shape);
            REQUIRE_FALSE(control.empty());
            REQUIRE(continuation.size() == control.size());
            REQUIRE(reference.size() == control.size());
            INFO(policy + ":" + field);
            if (is_reversible_zfp_radar_field_id(field))
            {
                REQUIRE(std::memcmp(control.data(), reference.data(),
                                    control.size() * sizeof(float)) == 0);
                REQUIRE(std::memcmp(continuation.data(), reference.data(),
                                    control.size() * sizeof(float)) == 0);
            }
            else
            {
                const double budget = qualified_zfp_field_budgets().at(field)
                    .max_configured_absolute_error;
                double continuous_error = 0.0;
                double resumed_error = 0.0;
                for (std::size_t value = 0; value < control.size(); ++value)
                {
                    continuous_error = std::max(continuous_error,
                        std::abs(static_cast<double>(control[value]) - reference[value]));
                    resumed_error = std::max(resumed_error,
                        std::abs(static_cast<double>(continuation[value]) - reference[value]));
                }
                REQUIRE(continuous_error <= budget);
                REQUIRE(resumed_error <= budget);
            }
        }
        for (const auto* ledger : ledgers)
        {
            REQUIRE(last_csv_row(continuous / ledger) ==
                    last_csv_row(npy_control / ledger));
            REQUIRE(last_csv_row(resumed / ledger) ==
                    last_csv_row(npy_control / ledger));
        }
        REQUIRE(read_text(resumed / "nimbus_output.json").find(
                    "\"publication_state\": \"ready\"") != std::string::npos);
    }
    for (const auto& [policy, workers] :
         {std::pair<std::string, int>{"serial", 1},
          {"field_parallel", 4}})
    {
        const auto retention = root / (policy + "_retention");
        REQUIRE(run(retention, policy, workers, 4, 1, 0, {}, 1, 4096, 256,
                    {}, false) == 0);
        const auto retained_latest = step_at_time(retention, 4.0);
        REQUIRE_FALSE(retained_latest.empty());
        REQUIRE(std::filesystem::is_regular_file(retention / "history_index.json"));
        for (const auto* field : fields)
            REQUIRE_FALSE(decode_zfp_at_time(retention, field, 4.0).empty());
        for (const auto* radar : {"reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"})
        {
            Zfp3dHeader header;
            std::string error;
            REQUIRE(read_zfp_3d_header((retained_latest / (std::string(radar) + ".zfp3d")).string(),
                                       header, error));
            REQUIRE(header.version == 4);
            REQUIRE(header.flags == 0);
        }

        const auto invalid_initial = root / (policy + "_invalid_initial");
        const auto invalid_state = [](SimulationStorageState& state)
        { state.fields.u.fill(10000.0f); };
        REQUIRE(run(invalid_initial, policy, workers, 2, 1, 0, {}, 3, 4096,
                    256, invalid_state, true) != 0);
        REQUIRE(read_text(invalid_initial / "nimbus_output.json").find(
                    "\"publication_state\": \"failed\"") != std::string::npos);
        REQUIRE_FALSE(std::filesystem::exists(
            invalid_initial / "step_000000" / "manifest.json"));
        REQUIRE_FALSE(std::filesystem::exists(invalid_initial / "history_index.json"));

        const auto invalid_late = root / (policy + "_invalid_late");
        std::string ready_contract;
        const auto late_hook = [&](SimulationStorageState& state)
        {
            ready_contract = read_text(invalid_late / "nimbus_output.json");
            state.fields.u.fill(10000.0f);
        };
        REQUIRE(run(invalid_late, policy, workers, 2, 1, 0, {}, 3, 4096,
                    256, late_hook, false) != 0);
        const auto failed_contract = read_text(invalid_late / "nimbus_output.json");
        REQUIRE(ready_contract.find("\"publication_state\": \"ready\"") !=
                std::string::npos);
        REQUIRE(failed_contract.find("\"publication_state\": \"failed\"") !=
                std::string::npos);
        REQUIRE(contract_generation(ready_contract) == contract_generation(failed_contract));
        REQUIRE(std::filesystem::is_regular_file(
            invalid_late / "step_000000" / "manifest.json"));
        REQUIRE_FALSE(std::filesystem::exists(
            invalid_late / "step_000001" / "manifest.json"));

        const auto index_failure = root / (policy + "_index_failure");
        int index_status = 0;
        {
            struct ResetHistoryFailure
            {
                ~ResetHistoryFailure() { set_history_index_failure(false); }
            } reset;
            set_history_index_failure(true);
            index_status = run(index_failure, policy, workers, 1, 1, 0,
                               {}, 3, 4096, 256, {}, false);
        }
        REQUIRE(index_status != 0);
        REQUIRE(read_text(index_failure / "nimbus_output.json").find(
                    "\"publication_state\": \"failed\"") != std::string::npos);
        REQUIRE(std::filesystem::is_regular_file(
            index_failure / "step_000000" / "manifest.json"));
        REQUIRE_FALSE(std::filesystem::exists(index_failure / "history_index.json"));

        const auto disk_failure = root / (policy + "_free_space_failure");
        REQUIRE(run(disk_failure, policy, workers, 1, 1, 0, {}, 3,
                    std::numeric_limits<int>::max(), 256, {}, false) != 0);
        REQUIRE(read_text(disk_failure / "nimbus_output.json").find(
                    "\"publication_state\": \"failed\"") != std::string::npos);
        REQUIRE_FALSE(std::filesystem::exists(
            disk_failure / "step_000000" / "manifest.json"));
    }
    const auto file_limit = root / "file_limit_pre_ready";
    REQUIRE(run(file_limit, "serial", 1, 1, 1, 0, {}, 3, 4096, 1,
                {}, false) != 0);
    REQUIRE_FALSE(std::filesystem::exists(file_limit / "nimbus_output.json"));
    REQUIRE(std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count() < 120.0);
    std::size_t files = 0;
    std::size_t bytes = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root))
        if (entry.is_regular_file())
        {
            ++files;
            bytes += entry.file_size();
        }
    REQUIRE(files < 1500);
    REQUIRE(bytes < 64U * 1024U * 1024U);
}

TEST_CASE("terminal deadline extension matches NPY on the same step schedule",
          "[core][zfp][v4][terminal_matched]")
{
    const char* zfp_env = std::getenv("NIMBUS_TEST_RADAR_V4_TERMINAL_ZFP_ROOT");
    const char* npy_env = std::getenv("NIMBUS_TEST_RADAR_V4_TERMINAL_NPY_ROOT");
    if (!zfp_env && !npy_env)
        return;
    REQUIRE(zfp_env != nullptr);
    REQUIRE(npy_env != nullptr);
    const std::filesystem::path zfp_root(zfp_env);
    const std::filesystem::path npy_root(npy_env);
    const std::array<const char*, 13> fields{
        "u", "v", "w", "theta", "p", "rho", "qv", "qr",
        "reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"};
    const std::array<const char*, 6> ledgers{
        "budget.csv", "momentum_budget.csv", "energy_budget.csv",
        "radiation_budget.csv", "thermodynamic_source_budget.csv",
        "moisture_budget.csv"};
    for (const auto& [zfp, npy, time] :
         {std::tuple<std::filesystem::path, std::filesystem::path, double>{
              zfp_root / "runs" / "parent_serial", npy_root / "parent", 2.0},
          {zfp_root / "runs" / "resumed_serial", npy_root / "resumed", 4.0}})
    {
        const auto npy_step = step_at_time(npy, time);
        const auto zfp_step = step_at_time(zfp, time);
        REQUIRE_FALSE(npy_step.empty());
        REQUIRE_FALSE(zfp_step.empty());
        REQUIRE(read_text(zfp_step / "manifest.json").find(
                    "\"compression_contract\": \"nimbus.zfp.per_field.v2\"") !=
                std::string::npos);
        for (const auto* field : fields)
        {
            std::array<int, 3> shape{};
            const auto reference = read_npy_f32_3d(
                npy_step / (std::string(field) + ".npy"), shape);
            const auto decoded = decode_zfp_at_time(zfp, field, time);
            REQUIRE(decoded.size() == reference.size());
            INFO(std::string("terminal matched time=") + std::to_string(time) +
                 " field=" + field);
            if (is_reversible_zfp_radar_field_id(field))
                REQUIRE(std::memcmp(decoded.data(), reference.data(),
                                    decoded.size() * sizeof(float)) == 0);
            else
            {
                const double budget = qualified_zfp_field_budgets().at(field)
                    .max_configured_absolute_error;
                double maximum_error = 0.0;
                for (std::size_t value = 0; value < decoded.size(); ++value)
                    maximum_error = std::max(maximum_error,
                        std::abs(static_cast<double>(decoded[value]) - reference[value]));
                REQUIRE(maximum_error <= budget);
            }
        }
        for (const auto* ledger : ledgers)
            REQUIRE(read_text(zfp / ledger) == read_text(npy / ledger));
        REQUIRE(read_text(zfp / "nimbus_output.json").find(
                    "\"publication_state\": \"ready\"") != std::string::npos);
    }
}

TEST_CASE("public radar ZFP subset artifacts match ordinary NPY controls",
          "[core][zfp][v4][public_readback]")
{
    const char* config_env = std::getenv("NIMBUS_TEST_RADAR_V4_PUBLIC_CONFIG_ROOT");
    const char* root_env = std::getenv("NIMBUS_TEST_RADAR_V4_PUBLIC_RUN_ROOT");
    if (!config_env && !root_env)
    {
        return;
    }
    REQUIRE(config_env != nullptr);
    REQUIRE(root_env != nullptr);
    const std::filesystem::path configs(config_env);
    const std::filesystem::path root(root_env);
    REQUIRE(std::filesystem::is_directory(configs));
    REQUIRE(std::filesystem::is_directory(root));
    const char* index_env = std::getenv("NIMBUS_TEST_RADAR_V4_PUBLIC_INDEX");
    const int selected_index = index_env ? std::stoi(index_env) : -1;
    REQUIRE(selected_index >= -1);
    REQUIRE(selected_index < 31);
    const std::set<std::string> core{"u", "v", "w", "theta", "p", "rho", "qv", "qr"};
    const std::array<const char*, 6> ledgers{
        "budget.csv", "momentum_budget.csv", "energy_budget.csv",
        "radiation_budget.csv", "thermodynamic_source_budget.csv",
        "moisture_budget.csv"};
    const auto& budgets = qualified_zfp_field_budgets();
    for (int index = 0; index < 31; ++index)
    {
        if (selected_index >= 0 && index != selected_index)
        {
            continue;
        }
        std::ostringstream label;
        label << "products_" << std::setw(2) << std::setfill('0') << index;
        const auto name = label.str();
        const auto selected = selected_products(configs / (name + "_zfp_high.yaml"));
        std::set<std::string> expected = core;
        expected.insert(selected.begin(), selected.end());
        REQUIRE(expected.size() == core.size() + selected.size());
        const auto npy = root / (name + "_npy");
        const auto zfp = root / (name + "_zfp_high");
        for (const auto* ledger : ledgers)
        {
            REQUIRE(read_text(npy / ledger) == read_text(zfp / ledger));
        }
        REQUIRE(read_text(npy / "nimbus_output.json").find(
                    "\"publication_state\": \"ready\"") != std::string::npos);
        REQUIRE(read_text(zfp / "nimbus_output.json").find(
                    "\"publication_state\": \"ready\"") != std::string::npos);
        std::map<std::string, ZfpFrameReader> readers;
        std::set<std::string> active_radar;
        for (const auto* step : {"step_000000", "step_000001"})
        {
            const auto npy_step = npy / step;
            const auto zfp_step = zfp / step;
            REQUIRE(published_manifest_fields(npy_step / "manifest.json") == expected);
            REQUIRE(published_manifest_fields(zfp_step / "manifest.json") == expected);
            const auto manifest = read_text(zfp_step / "manifest.json");
            REQUIRE(manifest.find("\"compression_contract\": \"nimbus.zfp.per_field.v2\"") !=
                    std::string::npos);
            REQUIRE(manifest.find("\"publication_state\": \"complete\"") !=
                    std::string::npos);
            std::set<std::string> npy_artifacts;
            std::set<std::string> zfp_artifacts;
            for (const auto& entry : std::filesystem::directory_iterator(npy_step))
            {
                if (entry.path().extension() == ".npy")
                {
                    npy_artifacts.insert(entry.path().stem().string());
                }
            }
            for (const auto& entry : std::filesystem::directory_iterator(zfp_step))
            {
                if (entry.path().extension() == ".zfp3d")
                {
                    zfp_artifacts.insert(entry.path().stem().string());
                }
            }
            REQUIRE(npy_artifacts == expected);
            REQUIRE(zfp_artifacts == expected);
            for (const auto& field : expected)
            {
                std::array<int, 3> shape{};
                const auto reference = read_npy_f32_3d(npy_step / (field + ".npy"), shape);
                Zfp3dHeader header;
                std::vector<float> decoded;
                std::string error;
                REQUIRE(readers[field].read_frame(
                    field, (zfp_step / (field + ".zfp3d")).string(),
                    header, decoded, error));
                REQUIRE(decoded.size() == reference.size());
                INFO(name + ":" + step + ":" + field);
                if (is_reversible_zfp_radar_field_id(field))
                {
                    REQUIRE(header.version == 4);
                    REQUIRE(header.mode == 3);
                    REQUIRE(header.flags == 0);
                    REQUIRE(std::memcmp(decoded.data(), reference.data(),
                                        decoded.size() * sizeof(float)) == 0);
                    if (std::any_of(reference.begin(), reference.end(),
                            [](float value) { return value != 0.0f; }))
                    {
                        active_radar.insert(field);
                    }
                }
                else
                {
                    REQUIRE(header.version == 3);
                    REQUIRE(header.mode == 0);
                    REQUIRE(budgets.count(field) == 1);
                    double maximum_error = 0.0;
                    for (std::size_t value = 0; value < decoded.size(); ++value)
                    {
                        maximum_error = std::max(maximum_error,
                            std::abs(static_cast<double>(decoded[value]) - reference[value]));
                    }
                    REQUIRE(maximum_error <=
                            budgets.at(field).max_configured_absolute_error);
                }
            }
        }
        REQUIRE(active_radar == std::set<std::string>(selected.begin(), selected.end()));
    }
}

TEST_CASE("public radar ZFP named and tier artifacts match ordinary NPY controls",
          "[core][zfp][v4][public_pair_readback]")
{
    const char* config_env = std::getenv("NIMBUS_TEST_RADAR_V4_PUBLIC_PAIR_CONFIG");
    const char* npy_env = std::getenv("NIMBUS_TEST_RADAR_V4_PUBLIC_PAIR_NPY");
    const char* zfp_env = std::getenv("NIMBUS_TEST_RADAR_V4_PUBLIC_PAIR_ZFP");
    if (!config_env && !npy_env && !zfp_env)
    {
        return;
    }
    REQUIRE(config_env != nullptr);
    REQUIRE(npy_env != nullptr);
    REQUIRE(zfp_env != nullptr);
    const std::filesystem::path config(config_env);
    const std::filesystem::path npy(npy_env);
    const std::filesystem::path zfp(zfp_env);
    const auto selected = selected_products(config);
    const std::set<std::string> selected_set(selected.begin(), selected.end());
    const auto& budgets = qualified_zfp_field_budgets();
    const std::array<const char*, 6> ledgers{
        "budget.csv", "momentum_budget.csv", "energy_budget.csv",
        "radiation_budget.csv", "thermodynamic_source_budget.csv",
        "moisture_budget.csv"};
    for (const auto* ledger : ledgers)
    {
        REQUIRE(read_text(npy / ledger) == read_text(zfp / ledger));
    }
    REQUIRE(read_text(zfp / "nimbus_output.json").find(
                "\"publication_state\": \"ready\"") != std::string::npos);
    std::map<std::string, ZfpFrameReader> readers;
    for (const auto* step : {"step_000000", "step_000001"})
    {
        const auto npy_step = npy / step;
        const auto zfp_step = zfp / step;
        const auto fields = published_manifest_fields(npy_step / "manifest.json");
        REQUIRE(published_manifest_fields(zfp_step / "manifest.json") == fields);
        REQUIRE(read_text(zfp_step / "manifest.json").find(
                    "\"compression_contract\": \"nimbus.zfp.per_field.v2\"") !=
                std::string::npos);
        std::set<std::string> published_radar;
        for (const auto& field : fields)
        {
            std::array<int, 3> shape{};
            const auto reference = read_npy_f32_3d(npy_step / (field + ".npy"), shape);
            Zfp3dHeader header;
            std::vector<float> decoded;
            std::string error;
            REQUIRE(readers[field].read_frame(
                field, (zfp_step / (field + ".zfp3d")).string(),
                header, decoded, error));
            REQUIRE(decoded.size() == reference.size());
            INFO(std::string(step) + ":" + field);
            if (is_reversible_zfp_radar_field_id(field))
            {
                published_radar.insert(field);
                REQUIRE(header.version == 4);
                REQUIRE(header.mode == 3);
                REQUIRE(header.flags == 0);
                REQUIRE(std::memcmp(decoded.data(), reference.data(),
                                    decoded.size() * sizeof(float)) == 0);
            }
            else
            {
                REQUIRE(header.version == 3);
                REQUIRE(header.mode == 0);
                REQUIRE(budgets.count(field) == 1);
                double maximum_error = 0.0;
                for (std::size_t value = 0; value < decoded.size(); ++value)
                {
                    maximum_error = std::max(maximum_error,
                        std::abs(static_cast<double>(decoded[value]) - reference[value]));
                }
                REQUIRE(maximum_error <=
                        budgets.at(field).max_configured_absolute_error);
            }
        }
        REQUIRE(published_radar == selected_set);
    }
}

TEST_CASE("public radar ZFP nonterminal checkpoint branches match continuous NPY",
          "[core][zfp][v4][public_nonterminal]")
{
    const char* root_env = std::getenv("NIMBUS_TEST_RADAR_V4_PUBLIC_CONT_ROOT");
    if (root_env == nullptr)
    {
        return;
    }
    const std::filesystem::path root(root_env);
    const auto npy_continuous = root / "npy_continuous";
    const auto npy_resumed = root / "npy_resumed";
    const auto npy_at_four = step_at_time(npy_continuous, 4.0);
    const auto npy_resumed_at_four = step_at_time(npy_resumed, 4.0);
    REQUIRE_FALSE(npy_at_four.empty());
    REQUIRE_FALSE(npy_resumed_at_four.empty());
    const auto expected = published_manifest_fields(npy_at_four / "manifest.json");
    REQUIRE(published_manifest_fields(npy_resumed_at_four / "manifest.json") == expected);
    for (const auto* policy : {"serial", "field_parallel"})
    {
        const auto continuous = root / (std::string("zfp_continuous_") + policy);
        const auto resumed = root / (std::string("zfp_resumed_") + policy);
        const auto continuous_at_four = step_at_time(continuous, 4.0);
        const auto resumed_at_four = step_at_time(resumed, 4.0);
        REQUIRE_FALSE(continuous_at_four.empty());
        REQUIRE_FALSE(resumed_at_four.empty());
        REQUIRE(published_manifest_fields(continuous_at_four / "manifest.json") == expected);
        REQUIRE(published_manifest_fields(resumed_at_four / "manifest.json") == expected);
        for (const auto& field : expected)
        {
            std::array<int, 3> shape{};
            const auto npy = read_npy_f32_3d(npy_at_four / (field + ".npy"), shape);
            const auto npy_branch = read_npy_f32_3d(
                npy_resumed_at_four / (field + ".npy"), shape);
            const auto zfp = decode_zfp_at_time(continuous, field, 4.0);
            const auto zfp_branch = decode_zfp_at_time(resumed, field, 4.0);
            REQUIRE(npy == npy_branch);
            REQUIRE(zfp.size() == npy.size());
            REQUIRE(zfp_branch.size() == npy.size());
            INFO(std::string(policy) + ":" + field);
            if (is_reversible_zfp_radar_field_id(field))
            {
                REQUIRE(std::memcmp(zfp.data(), npy.data(),
                                    zfp.size() * sizeof(float)) == 0);
                REQUIRE(std::memcmp(zfp_branch.data(), npy.data(),
                                    zfp_branch.size() * sizeof(float)) == 0);
            }
            else
            {
                REQUIRE(qualified_zfp_field_budgets().count(field) == 1);
                double continuous_error = 0.0;
                double resumed_error = 0.0;
                for (std::size_t index = 0; index < npy.size(); ++index)
                {
                    continuous_error = std::max(continuous_error,
                        std::abs(static_cast<double>(zfp[index]) - npy[index]));
                    resumed_error = std::max(resumed_error,
                        std::abs(static_cast<double>(zfp_branch[index]) - npy[index]));
                }
                const double budget = qualified_zfp_field_budgets().at(field)
                    .max_configured_absolute_error;
                REQUIRE(continuous_error <= budget);
                REQUIRE(resumed_error <= budget);
            }
        }
        for (const auto* ledger : {"budget.csv", "momentum_budget.csv",
                                   "energy_budget.csv", "radiation_budget.csv",
                                   "thermodynamic_source_budget.csv", "moisture_budget.csv"})
        {
            REQUIRE(last_csv_row(continuous / ledger) ==
                    last_csv_row(npy_continuous / ledger));
            REQUIRE(last_csv_row(resumed / ledger) ==
                    last_csv_row(npy_resumed / ledger));
        }
        REQUIRE(read_text(resumed / "nimbus_output.json").find(
                    "\"publication_state\": \"ready\"") != std::string::npos);
    }
}

TEST_CASE("public radar ZFP retained keyframes remain readable at t4",
          "[core][zfp][v4][public_retention]")
{
    const char* root_env = std::getenv("NIMBUS_TEST_RADAR_V4_PUBLIC_CONT_ROOT");
    if (root_env == nullptr)
    {
        return;
    }
    const std::filesystem::path root(root_env);
    const auto npy_at_four = step_at_time(root / "npy_continuous", 4.0);
    REQUIRE_FALSE(npy_at_four.empty());
    const auto expected = published_manifest_fields(npy_at_four / "manifest.json");
    for (const auto* policy : {"serial", "field_parallel"})
    {
        const auto retained = root / (std::string("zfp_retention_") + policy);
        const auto retained_at_four = step_at_time(retained, 4.0);
        REQUIRE_FALSE(retained_at_four.empty());
        REQUIRE(std::filesystem::is_regular_file(retained / "history_index.json"));
        REQUIRE(published_manifest_fields(retained_at_four / "manifest.json") == expected);
        for (const auto& field : expected)
        {
            std::array<int, 3> shape{};
            const auto reference = read_npy_f32_3d(
                npy_at_four / (field + ".npy"), shape);
            const auto decoded = decode_zfp_at_time(retained, field, 4.0);
            REQUIRE(decoded.size() == reference.size());
            Zfp3dHeader header;
            std::string error;
            REQUIRE(read_zfp_3d_header(
                (retained_at_four / (field + ".zfp3d")).string(),
                header, error));
            INFO(std::string(policy) + ":" + field);
            if (is_reversible_zfp_radar_field_id(field))
            {
                REQUIRE(header.version == 4);
                REQUIRE(header.flags == 0);
                REQUIRE(std::memcmp(decoded.data(), reference.data(),
                                    decoded.size() * sizeof(float)) == 0);
            }
            else
            {
                REQUIRE(header.version == 3);
                REQUIRE(qualified_zfp_field_budgets().count(field) == 1);
                double maximum_error = 0.0;
                for (std::size_t index = 0; index < decoded.size(); ++index)
                {
                    maximum_error = std::max(maximum_error,
                        std::abs(static_cast<double>(decoded[index]) - reference[index]));
                }
                REQUIRE(maximum_error <= qualified_zfp_field_budgets().at(field)
                    .max_configured_absolute_error);
            }
        }
        REQUIRE(read_text(retained / "nimbus_output.json").find(
                    "\"publication_state\": \"ready\"") != std::string::npos);
    }
}
