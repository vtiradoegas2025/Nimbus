/**
 * @file test_v1_loader.cpp
 * @brief Unit tests for the Nimbus v1 config loader (Phase A2).
 *
 * Covers the A2 exit criteria from docs/CM1Mapping.md §21:
 *   - A malformed v1 config produces an error message identifying file,
 *     line, key, and reason.
 *   - A well-formed v1 config loads and round-trips through a unit test.
 *
 * Plus the §19.6 "no silent runs" decision: unknown top-level keys are
 * hard errors, not warnings.
 */

#include "catch2/catch.hpp"
#include "chaos/config_contract.hpp"
#include "core/runtime/flat_lin_storm_profile.hpp"
#include "init/config/v0_to_v1.hpp"
#include "init/config/v1_loader.hpp"
#include "init/config/config_v1_serialization_common.hpp"
#include "init/config/qualified_terrain_profile.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <utility>

using namespace tmv::init::config;

TEST_CASE("flat Lin storm admission follows physics rather than machine paths or grid size",
          "[init][config][lin-storm]")
{
    const auto sounding = GENERATE("01_classic_supercell", "02_pulse_decay", "03_moist_supercell");
    const auto loaded = load_v1_config(std::string("tests/fixtures/configs/lin_storm/") + sounding + ".yaml");
    REQUIRE(loaded.ok());
    auto config = *loaded.config;
    config.extensions.execution.mode = ExecutionMode::Qualified;
    const auto backend = GENERATE(ComputeBackend::CPU, ComputeBackend::GPU, ComputeBackend::Metal);
    config.extensions.compute.backend = backend;
    config.extensions.compute.gpu_apply_microphysics = backend != ComputeBackend::CPU;
    REQUIRE(tmv::runtime::is_flat_lin_storm_profile(config));
    REQUIRE_NOTHROW(serialize_v1_to_v0_map(config));
    for (int size : {6, 48, 128, 640, 1024})
    {
        auto resized = config;
        resized.grid.nx = size;
        resized.grid.ny = size + 1;
        resized.grid.dx = 37.5;
        resized.grid.dy = 62.5;
        resized.initial_conditions.sounding.file_path = "/research/another-user/storm.txt";
        REQUIRE(tmv::runtime::is_flat_lin_storm_profile(resized));
    }
    for (int change = 0; change < 10; ++change)
    {
        CAPTURE(change);
        auto neighbor = config;
        switch (change)
        {
            case 0: neighbor.extensions.compute.allow_fallback = true; break;
            case 1: neighbor.extensions.validation.guard_mode = ValidationGuardMode::Off; break;
            case 2: neighbor.terrain.scheme = TerrainScheme::Bell; break;
            case 3: neighbor.radiation.scheme = RadiationScheme::RRTMG; break;
            case 4: neighbor.boundary_conditions.lateral.east = LateralBC::Periodic; break;
            case 5: neighbor.extensions.compute.gpu_apply_slow = true; break;
            case 6: neighbor.microphysics.parameters.lin.accretion_coefficient = 3.0; break;
            case 7: neighbor.time.integrator = TimeIntegrator::RK4; break;
            case 8: neighbor.extensions.diagnostics.scientific_every_steps = 0; break;
            case 9: neighbor.grid.dx = std::numeric_limits<double>::quiet_NaN(); break;
        }
        if (change == 5 && backend == ComputeBackend::CPU)
            REQUIRE(tmv::runtime::is_flat_lin_storm_profile(neighbor));
        else
            REQUIRE_FALSE(tmv::runtime::is_flat_lin_storm_profile(neighbor));
    }
}

TEST_CASE("CUDA assist is explicit portable configuration and rejects incompatible backends",
          "[init][config][cuda-assist]")
{
    const auto loaded = load_v1_config("tests/fixtures/configs/lin_storm/01_classic_supercell.yaml");
    REQUIRE(loaded.ok());
    auto config = *loaded.config;
    REQUIRE_FALSE(config.extensions.compute.cuda_assist);
    REQUIRE(serialize_v1_to_v0_map(config).count("numerics.compute.cuda_assist") == 0);
    config.extensions.compute.cuda_assist = true;
    REQUIRE(serialize_v1_to_v0_map(config).at("numerics.compute.cuda_assist") == "true");
    REQUIRE(serialize_v1_to_yaml(config).find("cuda_assist: true") != std::string::npos);
    for (const auto backend : {ComputeBackend::CPU, ComputeBackend::Metal})
    {
        config.extensions.compute.backend = backend;
        REQUIRE_THROWS(serialize_v1_to_v0_map(config));
    }
}

namespace
{

/**
 * @brief RAII wrapper around a temp YAML file written for one test case.
 *
 * Each instance writes the provided contents to a unique path under the
 * system temp directory and deletes the file on destruction. Tests use
 * this to drive `load_v1_config(path)` with concrete inputs.
 */
struct TempYaml
{
    std::filesystem::path path;

    explicit TempYaml(const std::string& contents)
    {
        // Use a randomised suffix so parallel test runs don't collide on
        // the same path. PID alone is not enough because Catch2 sections
        // can write multiple files within one process.
        std::random_device                      rd;
        std::mt19937                            rng(rd());
        std::uniform_int_distribution<uint64_t> dist;
        const uint64_t                          suffix = dist(rng);

        path = std::filesystem::temp_directory_path() / ("nimbus_v1_test_" + std::to_string(suffix) + ".yaml");
        std::ofstream f(path);
        f << contents;
    }

    ~TempYaml()
    {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

/// Minimal v1 config with the required keys set so loader returns ok().
/// Tests that want to exercise specific fields extend this.
const char* kMinimalValidV1 = R"yaml(
schema: nimbus.v1
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 250.0
  dy: 250.0
  dz: 250.0
time:
  run_time: 7200.0
  dt_large: 2.0
  output_interval: 30.0
)yaml";

bool has_error(const LoadResult& result, const std::string& key, const std::string& message_fragment = {})
{
    return std::any_of(result.errors.begin(),
                       result.errors.end(),
                       [&](const LoadError& error)
                       {
                           return error.key == key && (message_fragment.empty() ||
                                                       error.message.find(message_fragment) != std::string::npos);
                       });
}

std::string serialized_output_block(const ConfigV1& config)
{
    const std::string serialized = serialize_v1_to_yaml(config);
    const auto begin = serialized.find("output:\n");
    const auto end = serialized.find("\n# Nimbus extensions", begin);
    REQUIRE(begin != std::string::npos);
    REQUIRE(end != std::string::npos);
    return serialized.substr(begin, end - begin);
}

TEST_CASE("Generated Metal canonical YAML preserves its serialized identity",
          "[init][config][v1_loader][generated-metal]")
{
    const std::string path = "tests/fixtures/reference/catalog/generated_metal_canonical_v1.yaml";
    std::ifstream input(path);
    REQUIRE(input.good());
    const std::string source{std::istreambuf_iterator<char>(input),
                             std::istreambuf_iterator<char>()};
    const auto loaded = load_v1_config(path);
    REQUIRE(loaded.ok());
    REQUIRE(serialize_v1_to_yaml(*loaded.config) == source);
}

TEST_CASE("East-radiative generated Metal NPY scope retains raw output and boundary identity",
          "[init][config][v1_loader][generated-metal-east-npy]")
{
    const std::string path =
        "tests/fixtures/reference/catalog/flat_east_radiative_metal_npy_candidate_v1.yaml";
    std::ifstream input(path);
    REQUIRE(input.good());
    const std::string source{std::istreambuf_iterator<char>(input),
                             std::istreambuf_iterator<char>()};
    const auto loaded = load_v1_config(path);
    REQUIRE(loaded.ok());
    REQUIRE(serialize_v1_to_yaml(*loaded.config) == source);
    REQUIRE(loaded.config->output.format == OutputFormat::Npy3D);
    REQUIRE_FALSE(loaded.config->output.async_io);
    REQUIRE(loaded.config->boundary_conditions.lateral.east == LateralBC::Radiative);
    auto netcdf = *loaded.config;
    netcdf.output.format = OutputFormat::Netcdf;
    REQUIRE(serialize_v1_to_yaml(netcdf) != source);
    const auto zfp = load_v1_config(
        "tests/fixtures/reference/catalog/generated_metal_zfp_candidate_v1.yaml");
    REQUIRE(zfp.ok());
    auto compressed = *loaded.config;
    compressed.output = zfp.config->output;
    REQUIRE(serialize_v1_to_yaml(compressed) != source);
    auto async = *loaded.config;
    async.output.async_io = true;
    REQUIRE(serialize_v1_to_yaml(async) != source);
    auto adjacent = *loaded.config;
    adjacent.boundary_conditions.lateral.east = LateralBC::ZeroGradient;
    REQUIRE(serialize_v1_to_yaml(adjacent) != source);
}

TEST_CASE("Generated Metal netCDF output normalizes to the same solver profile",
          "[init][config][v1_loader][generated-metal]")
{
    const std::string base_path = "tests/fixtures/reference/catalog/generated_metal_canonical_v1.yaml";
    const std::string netcdf_path = "tests/fixtures/reference/catalog/generated_metal_netcdf_v1.yaml";
    std::ifstream input(base_path);
    REQUIRE(input.good());
    const std::string source{std::istreambuf_iterator<char>(input),
                             std::istreambuf_iterator<char>()};
    const auto loaded = load_v1_config(netcdf_path);
    REQUIRE(loaded.ok());
    REQUIRE(loaded.config->output.format == OutputFormat::Netcdf);
    auto canonical = *loaded.config;
    canonical.output.format = OutputFormat::Npy3D;
    REQUIRE(serialize_v1_to_yaml(canonical) == source);
}

TEST_CASE("Generated Metal ZFP typed output block round-trips and seals active owners",
          "[init][config][v1_loader][generated-metal-zfp-typed-scope]")
{
    const auto loaded = load_v1_config(
        "tests/fixtures/reference/catalog/generated_metal_zfp_candidate_v1.yaml");
    const auto base = load_v1_config(
        "tests/fixtures/reference/catalog/generated_metal_canonical_v1.yaml");
    REQUIRE(loaded.ok());
    REQUIRE(base.ok());
    const std::string serialized = serialize_v1_to_yaml(*loaded.config);
    TempYaml round_trip(serialized);
    const auto loaded_again = load_v1_config(round_trip.path.string());
    REQUIRE(loaded_again.ok());
    REQUIRE(serialize_v1_to_yaml(*loaded_again.config) == serialized);

    auto normalized = *loaded.config;
    normalized.output = OutputSection{};
    normalized.output.async_io = false;
    normalized.extensions.execution.mode = ExecutionMode::Qualified;
    REQUIRE(serialize_v1_to_yaml(normalized) == serialize_v1_to_yaml(*base.config));

    const std::string exact_block = serialized_output_block(*loaded.config);
    if (std::getenv("NIMBUS_EMIT_TYPED_ZFP_SCOPE") != nullptr)
    {
        std::ostringstream encoded;
        encoded << std::hex << std::setfill('0');
        for (unsigned char byte : exact_block)
        {
            encoded << std::setw(2) << static_cast<unsigned int>(byte);
        }
        std::cout << "NIMBUS_TYPED_ZFP_OUTPUT_HEX " << encoded.str() << '\n';
    }

    const auto changed_or_rejected = [&](auto mutate)
    {
        auto candidate = *loaded.config;
        mutate(candidate.output);
        try
        {
            CHECK(serialized_output_block(candidate) != exact_block);
        }
        catch (const std::exception&)
        {
            SUCCEED();
        }
    };
    changed_or_rejected([](auto& o) { o.preset = OutputPreset::Presentation; });
    changed_or_rejected([](auto& o) { o.format = OutputFormat::ZfpMedium; });
    changed_or_rejected([](auto& o) { o.async_io = false; });
    changed_or_rejected([](auto& o) { o.live_shm = true; });
    changed_or_rejected([](auto& o) { o.live_shm_fields = {"u"}; });
    changed_or_rejected([](auto& o) { o.live_shm_interval_s = 2.0; });
    changed_or_rejected([](auto& o) { o.live_shm_full_rate = true; });
    changed_or_rejected([](auto& o) {
        o.qualification_diagnostics_preset = QualificationDiagnosticsPreset::MovingReferenceV1;
    });
    changed_or_rejected([](auto& o) { o.trajectories = true; });
    changed_or_rejected([](auto& o) { o.trajectory_seed_time_s = 1.0; });
    changed_or_rejected([](auto& o) { o.trajectory_count = 6; });
    changed_or_rejected([](auto& o) { o.trajectory_sample_interval_s = 1.0; });
    changed_or_rejected([](auto& o) { o.trajectory_max_samples_per_parcel = 100; });
    changed_or_rejected([](auto& o) { o.zfp_tolerance = 1.0e-5; });
    changed_or_rejected([](auto& o) { o.zfp_mode = "rate"; });
    changed_or_rejected([](auto& o) { o.zfp_precision_bits = 17; });
    changed_or_rejected([](auto& o) { o.zfp_rate_bps = 9; });
    changed_or_rejected([](auto& o) { o.zfp_keyframe_interval = 3; });
    changed_or_rejected([](auto& o) { o.zfp_sparse_threshold = 1.0e-8; });
    changed_or_rejected([](auto& o) { o.zfp_float16_prefilter = true; });
    changed_or_rejected([](auto& o) { o.zfp_predictive_delta = false; });
    changed_or_rejected([](auto& o) { o.zfp_field_execution = "field_parallel"; });
    changed_or_rejected([](auto& o) { o.zfp_field_workers = 1; });
    changed_or_rejected([](auto& o) { o.zfp_field_scratch_mib = 65; });
    changed_or_rejected([](auto& o) { o.zfp_per_field_tolerances = false; });
    changed_or_rejected([](auto& o) { o.zfp_tight_tolerance = 2.0e-4; });
    changed_or_rejected([](auto& o) { o.zfp_moderate_tolerance = 2.0e-3; });
    changed_or_rejected([](auto& o) { o.zfp_loose_tolerance = 2.0e-2; });
    changed_or_rejected([](auto& o) { o.zfp_field_tolerances["u"] = 2.0e-4; });
    changed_or_rejected([](auto& o) { o.zfp_field_tolerances["qc"] = 1.0e-8; });
    changed_or_rejected([](auto& o) { o.tiered_write_cadence = true; });
    changed_or_rejected([](auto& o) { o.keep_last_frames = 2; });
    changed_or_rejected([](auto& o) { o.minimum_free_space_mib = 1; });
    changed_or_rejected([](auto& o) { o.maximum_retained_history_files = 128; });

    auto active_diagnostics = *loaded.config;
    active_diagnostics.output.qualification_diagnostics_preset =
        QualificationDiagnosticsPreset::MovingReferenceV1;
    const std::string diagnostics_block = serialized_output_block(active_diagnostics);
    active_diagnostics.output.qualification_diagnostics_interval_s = 31;
    CHECK(serialized_output_block(active_diagnostics) != diagnostics_block);

    auto active_tiered = *loaded.config;
    active_tiered.output.tiered_write_cadence = true;
    const std::string tiered_block = serialized_output_block(active_tiered);
    active_tiered.output.write_cadence_medium_s = 31;
    CHECK(serialized_output_block(active_tiered) != tiered_block);
    active_tiered.output.write_cadence_medium_s = 30;
    active_tiered.output.write_cadence_slow_s = 61;
    CHECK(serialized_output_block(active_tiered) != tiered_block);

    auto inactive = *loaded.config;
    inactive.output.netcdf_deflate_level = 9;
    inactive.output.netcdf_shuffle = false;
    inactive.output.netcdf_target_chunk_kib = 2048;
    inactive.output.write_cadence_medium_s = 31;
    inactive.output.write_cadence_slow_s = 61;
    inactive.output.qualification_diagnostics_interval_s = 31;
    CHECK(serialized_output_block(inactive) == exact_block);
}

} // namespace

TEST_CASE("load_v1_config: invalid boolean fails closed", "[init][config][v1_loader][config-truth]")
{
    TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
time:
  adaptive_dt: sometimes
)yaml");
    const LoadResult result = load_v1_config(tmp.path.string());
    REQUIRE_FALSE(result.ok());
    REQUIRE(has_error(result, "time.adaptive_dt", "cannot parse"));
}

TEST_CASE("strict v1 user SGS filter width is typed and fail closed",
          "[init][config][v1_loader][turbulence][sci357]")
{
    SECTION("explicit positive metre width projects to runtime")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
turbulence:
  scheme: smagorinsky
  mode: 3d
  filter_width: user
  user_filter_width_m: 125.0
)yaml");
        const auto loaded = load_v1_config(tmp.path.string());
        REQUIRE(loaded.ok());
        REQUIRE(loaded.config->turbulence.filter_width ==
                TurbulenceFilterWidth::User);
        REQUIRE(loaded.config->turbulence.user_filter_width_m == 125.0);
        const auto projected = serialize_v1_to_v0_map(*loaded.config);
        REQUIRE(projected.at("turbulence.filter_width") == "user");
        REQUIRE(std::stod(projected.at("turbulence.user_filter_width_m")) ==
                125.0);
    }

    SECTION("user selection requires a positive width")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
turbulence:
  filter_width: user
)yaml");
        const auto loaded = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(loaded.ok());
        REQUIRE(has_error(loaded, "turbulence.user_filter_width_m",
                          "positive and finite"));
    }

    SECTION("metric-derived selection rejects a user width")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
turbulence:
  filter_width: cubic_root
  user_filter_width_m: 125.0
)yaml");
        const auto loaded = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(loaded.ok());
        REQUIRE(has_error(loaded, "turbulence.user_filter_width_m",
                          "only valid"));
    }
}

TEST_CASE("strict v1 turbulence closure selectors are typed and fail closed",
          "[init][config][v1_loader][turbulence][sci359]")
{
    SECTION("dynamic Smagorinsky projects to the runtime selector")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
turbulence:
  scheme: smagorinsky
  dynamic_coefficient: true
  stability_correction: none
)yaml");
        const auto loaded = load_v1_config(tmp.path.string());
        REQUIRE(loaded.ok());
        REQUIRE(loaded.config->turbulence.dynamic_coefficient);
        REQUIRE(loaded.config->turbulence.stability_correction ==
                TurbulenceStabilityCorrection::None);
        const auto projected = serialize_v1_to_v0_map(*loaded.config);
        REQUIRE(projected.at("turbulence.dynamic_Cs") == "true");
        REQUIRE(projected.at("turbulence.stability_correction") == "none");
    }

    SECTION("TKE stability projects to the runtime selector")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
turbulence:
  scheme: tke
  stability_correction: tke_based
)yaml");
        const auto loaded = load_v1_config(tmp.path.string());
        REQUIRE(loaded.ok());
        REQUIRE(loaded.config->turbulence.stability_correction ==
                TurbulenceStabilityCorrection::TkeBased);
        const auto projected = serialize_v1_to_v0_map(*loaded.config);
        REQUIRE(projected.at("turbulence.stability_correction") ==
                "tke_based");
    }

    SECTION("dynamic coefficient requires Smagorinsky")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
turbulence:
  scheme: tke
  dynamic_coefficient: true
)yaml");
        const auto loaded = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(loaded.ok());
        REQUIRE(has_error(loaded, "turbulence.dynamic_coefficient",
                          "requires scheme=smagorinsky"));
    }

    SECTION("TKE stability requires the TKE closure")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
turbulence:
  scheme: smagorinsky
  stability_correction: tke_based
)yaml");
        const auto loaded = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(loaded.ok());
        REQUIRE(has_error(loaded, "turbulence.stability_correction",
                          "requires scheme=tke"));
    }

    SECTION("dynamic coefficient cannot stack another stability model")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
turbulence:
  scheme: smagorinsky
  dynamic_coefficient: true
  stability_correction: ri
)yaml");
        const auto loaded = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(loaded.ok());
        REQUIRE(has_error(loaded, "turbulence.stability_correction",
                          "cannot use a separate"));
    }
}

TEST_CASE("load_v1_config: advection reconstruction is explicit and typed",
          "[init][config][v1_loader][advection][config-truth]")
{
    SECTION("omitted selection preserves grid-specific auto resolution")
    {
        TempYaml tmp(kMinimalValidV1);
        const LoadResult result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->advection.scheme == AdvectionScheme::Auto);
        REQUIRE(std::string(resolved_advection_scheme_name(
                    result.config->advection.scheme,
                    StaggerType::CGrid)) == "weno5");
        REQUIRE(std::string(resolved_advection_scheme_name(
                    result.config->advection.scheme,
                    StaggerType::Collocated)) == "tvd");
    }

    SECTION("canonical WENO5 selection")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
advection:
  scheme: weno5
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->advection.scheme == AdvectionScheme::WENO5);
    }

    SECTION("unknown selection fails closed")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
advection:
  scheme: magic
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "advection.scheme", "expected one of"));
    }
}

TEST_CASE("load_v1_config: coupled Runge-Kutta selection projects without substitution",
          "[init][config][v1_loader][time_stepping]")
{
    for (const auto& selection :
         std::array<std::pair<const char*, TimeIntegrator>, 2>{{
             {"rk3", TimeIntegrator::SSPRK3},
             {"rk4", TimeIntegrator::RK4},
         }})
    {
        std::string yaml = kMinimalValidV1;
        const std::string anchor = "time:\n";
        const auto position = yaml.find(anchor);
        REQUIRE(position != std::string::npos);
        yaml.insert(position + anchor.size(),
                    "  integrator: " + std::string(selection.first) + "\n");
        TempYaml tmp(yaml);
        const LoadResult result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->time.integrator == selection.second);
        const auto projected = serialize_v1_to_v0_map(*result.config);
        REQUIRE(projected.at("numerics.time_stepping.scheme") ==
                selection.first);
        REQUIRE(projected.at("numerics.time_stepping.split_acoustic") ==
                "false");
    }

    std::string yaml = kMinimalValidV1;
    const std::string anchor = "time:\n";
    yaml.insert(yaml.find(anchor) + anchor.size(),
                "  integrator: leapfrog\n");
    TempYaml invalid(yaml);
    const LoadResult result = load_v1_config(invalid.path.string());
    REQUIRE_FALSE(result.ok());
    REQUIRE(has_error(result, "time.integrator", "expected one of"));
}

TEST_CASE("load_v1_config: Studio preset and trajectories are first-class output config",
          "[init][config][v1_loader][output][studio][trajectories]")
{
    TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
output:
  preset: studio_full
  trajectories: true
  trajectory_seed_time_s: 90.5
  trajectory_count: 7
  trajectory_sample_interval_s: 2.5
  trajectory_max_samples_per_parcel: 1200
)yaml");
    const LoadResult result = load_v1_config(tmp.path.string());
    INFO(format_errors(result.errors));
    REQUIRE(result.ok());
    REQUIRE(result.config->output.preset == OutputPreset::StudioFull);
    REQUIRE(result.config->output.trajectories);
    REQUIRE(result.config->output.trajectory_seed_time_s == Approx(90.5));
    REQUIRE(result.config->output.trajectory_count == 7);
    REQUIRE(result.config->output.trajectory_sample_interval_s == Approx(2.5));
    REQUIRE(result.config->output.trajectory_max_samples_per_parcel == 1200);
}

TEST_CASE("load_v1_config: live shared memory has an explicit field and cadence contract",
          "[init][config][v1_loader][output][shm]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
output:
  live_shm: true
  live_shm_fields: [w, theta]
  live_shm_interval_s: 2.5
  live_shm_full_rate: false
)yaml");
    const LoadResult result = load_v1_config(tmp.path.string());
    INFO(format_errors(result.errors));
    REQUIRE(result.ok());
    REQUIRE(result.config->output.live_shm);
    REQUIRE(result.config->output.live_shm_fields ==
            std::vector<std::string>{"w", "theta"});
    REQUIRE(result.config->output.live_shm_interval_s == Approx(2.5));
    REQUIRE_FALSE(result.config->output.live_shm_full_rate);

    TempYaml duplicate(std::string(kMinimalValidV1) + R"yaml(
output:
  live_shm_fields: [w, w]
)yaml");
    const LoadResult duplicate_result = load_v1_config(duplicate.path.string());
    REQUIRE_FALSE(duplicate_result.ok());
    REQUIRE(has_error(duplicate_result, "output.live_shm_fields", "duplicate"));

    TempYaml invalid_interval(std::string(kMinimalValidV1) + R"yaml(
output:
  live_shm_interval_s: 0
)yaml");
    const LoadResult interval_result = load_v1_config(invalid_interval.path.string());
    REQUIRE_FALSE(interval_result.ok());
    REQUIRE(has_error(interval_result, "output.live_shm_interval_s"));
}

TEST_CASE("load_v1_config: ZFP field-parallel policy is explicit and bounded",
          "[init][config][v1_loader][output][zfp][parallel]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
output:
  format: zfp_high
  zfp_field_execution: field_parallel
  zfp_field_workers: 4
  zfp_field_scratch_mib: 128
)yaml");
    const LoadResult result = load_v1_config(tmp.path.string());
    INFO(format_errors(result.errors));
    REQUIRE(result.ok());
    REQUIRE(result.config->output.zfp_field_execution == "field_parallel");
    REQUIRE(result.config->output.zfp_field_workers == 4);
    REQUIRE(result.config->output.zfp_field_scratch_mib == 128);

    TempYaml invalid(std::string(kMinimalValidV1) + R"yaml(
output:
  zfp_field_execution: shared_stream
)yaml");
    const LoadResult invalid_result = load_v1_config(invalid.path.string());
    REQUIRE_FALSE(invalid_result.ok());
    REQUIRE(has_error(invalid_result, "output.zfp_field_execution"));
}

TEST_CASE("load_v1_config: science-validation output preset is first-class",
          "[init][config][v1_loader][output][science_validation]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
output:
  preset: science_validation
)yaml");
    const LoadResult result = load_v1_config(tmp.path.string());
    INFO(format_errors(result.errors));
    REQUIRE(result.ok());
    REQUIRE(result.config->output.preset == OutputPreset::ScienceValidation);
}

TEST_CASE("load_v1_config: radar products and instrument geometry are typed",
          "[init][config][v1_loader][radar]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
radar:
  enabled: true
  products: [reflectivity_dbz, radial_velocity]
  reflectivity_tier: fast_da
  velocity_sampling: point
  location:
    x_m: -1500
    y_m: 750
    z_m: 250
  apply_scatterer_correction: true
output:
  preset: radar_ready
)yaml");
    const LoadResult result = load_v1_config(tmp.path.string());
    INFO(format_errors(result.errors));
    REQUIRE(result.ok());
    REQUIRE(result.config->radar.enabled);
    REQUIRE(result.config->radar.products ==
            std::vector<std::string>{"reflectivity_dbz", "radial_velocity"});
    REQUIRE(result.config->radar.radar_x_m == Approx(-1500.0));
    REQUIRE(result.config->radar.radar_y_m == Approx(750.0));
    REQUIRE(result.config->radar.radar_z_m == Approx(250.0));
    REQUIRE(result.config->output.preset == OutputPreset::RadarReady);
}

TEST_CASE("load_v1_config: omitted radar is disabled in unrelated qualified GPU profiles",
          "[init][config][v1_loader][radar][qualification]")
{
    for (const std::string path : {
             "tests/fixtures/configs/qualification/cgrid_variable_diffusion_vulkan_ready.yaml",
             "tests/fixtures/configs/qualification/cgrid_variable_diffusion_metal_ready.yaml"})
    {
        CAPTURE(path);
        const LoadResult result = load_v1_config(path);
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE_FALSE(result.config->radar.enabled);
        REQUIRE(result.config->radar.products.empty());
    }
}

TEST_CASE("load_v1_config: maintained polarimetric products require maintained tier",
          "[init][config][v1_loader][radar][strict]")
{
    for (const std::string product : {"kdp", "rhohv"})
    {
        CAPTURE(product);
        TempYaml tmp(std::string(kMinimalValidV1) +
                     "radar:\n  enabled: true\n  products: [" + product + "]\n");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "radar.zdr_tier", "require zhang2021_rain_mp"));
    }
}

TEST_CASE("load_v1_config: exact polarimetric profile changes only its declared scope",
          "[init][config][radar][qualification]")
{
    const LoadResult lin = load_v1_config(
        "tests/fixtures/configs/qualification/cgrid_lin_rrtmg_ready.yaml");
    const LoadResult polar = load_v1_config(
        "tests/fixtures/configs/qualification/cgrid_radar_polarimetric_ready.yaml");
    REQUIRE(lin.ok());
    REQUIRE(polar.ok());

    ConfigV1 expected = *lin.config;
    ConfigV1 normalized = *polar.config;
    normalized.microphysics = expected.microphysics;
    normalized.radiation = expected.radiation;
    normalized.initial_conditions.sounding =
        expected.initial_conditions.sounding;
    normalized.radar = expected.radar;
    normalized.output = expected.output;
    REQUIRE(is_public_qualified_lin_rrtmg_profile(expected));
    REQUIRE(is_public_qualified_lin_rrtmg_profile(normalized));
    REQUIRE(is_public_qualified_radar_profile(*polar.config));
    expected.extensions.execution.mode = ExecutionMode::Exploratory;
    normalized.extensions.execution.mode = ExecutionMode::Exploratory;
    REQUIRE(serialize_v1_to_yaml(normalized) == serialize_v1_to_yaml(expected));
}

TEST_CASE("load_v1_config: exact user filter-width profile is publicly selectable",
          "[init][config][turbulence][qualification][sci357]")
{
    const LoadResult result = load_v1_config(
        "tests/fixtures/configs/qualification/cgrid_user_filter_width_ready.yaml");
    INFO(format_errors(result.errors));
    REQUIRE(result.ok());
    REQUIRE(is_public_qualified_user_filter_width_profile(*result.config));
    REQUIRE(result.config->turbulence.filter_width ==
            TurbulenceFilterWidth::User);
    REQUIRE(result.config->turbulence.user_filter_width_m == 500.0);

    ConfigV1 neighbor = *result.config;
    neighbor.turbulence.user_filter_width_m = 250.0;
    REQUIRE_FALSE(is_public_qualified_user_filter_width_profile(neighbor));
    neighbor = *result.config;
    neighbor.extensions.compute.backend = ComputeBackend::Metal;
    REQUIRE_FALSE(is_public_qualified_user_filter_width_profile(neighbor));
}

TEST_CASE("load_v1_config: SCI-359 turbulence profiles are exact public selections",
          "[init][config][turbulence][qualification][sci359]")
{
    const LoadResult dynamic = load_v1_config(
        "tests/fixtures/configs/qualification/cgrid_dynamic_smagorinsky_ready.yaml");
    INFO(format_errors(dynamic.errors));
    REQUIRE(dynamic.ok());
    REQUIRE(is_public_qualified_dynamic_smagorinsky_profile(
        *dynamic.config));
    REQUIRE_FALSE(is_public_qualified_tke_stability_profile(
        *dynamic.config));
    ConfigV1 neighbor = *dynamic.config;
    neighbor.turbulence.Cs = 0.18;
    REQUIRE_FALSE(is_public_qualified_dynamic_smagorinsky_profile(neighbor));
    neighbor = *dynamic.config;
    neighbor.extensions.compute.backend = ComputeBackend::Metal;
    REQUIRE_FALSE(is_public_qualified_dynamic_smagorinsky_profile(neighbor));

    const LoadResult tke = load_v1_config(
        "tests/fixtures/configs/qualification/cgrid_tke_rrtmg_ready.yaml");
    INFO(format_errors(tke.errors));
    REQUIRE(tke.ok());
    REQUIRE(is_public_qualified_tke_stability_profile(*tke.config));
    REQUIRE_FALSE(is_public_qualified_dynamic_smagorinsky_profile(
        *tke.config));
    neighbor = *tke.config;
    neighbor.turbulence.stability_correction =
        TurbulenceStabilityCorrection::None;
    REQUIRE_FALSE(is_public_qualified_tke_stability_profile(neighbor));
    neighbor = *tke.config;
    neighbor.time.dt_large = 0.1;
    REQUIRE_FALSE(is_public_qualified_tke_stability_profile(neighbor));
}

TEST_CASE("load_v1_config: radar-derived output requires its operator product",
          "[init][config][v1_loader][radar][output]")
{
    SECTION("exploratory ZDR is explicitly backed")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
radar:
  enabled: true
  products: [zdr]
output:
  preset: radar_zdr_exploratory
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
    }
    SECTION("missing ZDR ownership fails")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
output:
  preset: radar_zdr_exploratory
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "output.preset", "does not select"));
    }
    SECTION("maintained polarimetric preset is fully backed")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
radar:
  enabled: true
  products: [zdr, kdp, rhohv]
  zdr_tier: zhang2021_rain_mp
output:
  preset: radar_polarimetric_ready
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
    }
}

TEST_CASE("load_v1_config: moving-reference diagnostics are a typed sparse cadence",
          "[init][config][v1_loader][output][qualification]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  tracking:
    enabled: true
    interval_s: 30
    inner_margin_m: 1000
output:
  qualification_diagnostics:
    preset: moving_reference_v1
    interval_s: 30
)yaml");
    const LoadResult result = load_v1_config(tmp.path.string());
    INFO(format_errors(result.errors));
    REQUIRE(result.ok());
    REQUIRE(result.config->output.qualification_diagnostics_preset ==
            QualificationDiagnosticsPreset::MovingReferenceV1);
    REQUIRE(result.config->output.qualification_diagnostics_interval_s == 30);
    REQUIRE(result.config->time.output_interval == Approx(30.0));
}

TEST_CASE("load_v1_config: moving-reference sparse cadence fails closed",
          "[init][config][v1_loader][output][qualification]")
{
    SECTION("interval without a typed preset")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
output:
  qualification_diagnostics:
    interval_s: 30
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result,
                          "output.qualification_diagnostics.interval_s",
                          "requires preset"));
    }
    SECTION("tracker cadence mismatch")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  tracking:
    enabled: true
    interval_s: 60
    inner_margin_m: 1000
output:
  qualification_diagnostics:
    preset: moving_reference_v1
    interval_s: 30
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result,
                          "output.qualification_diagnostics.preset",
                          "same interval_s"));
    }
    SECTION("adaptive time stepping cannot promise exact publication times")
    {
        std::string yaml = kMinimalValidV1;
        const std::string marker = "  output_interval: 30.0\n";
        yaml.replace(yaml.find(marker), marker.size(),
                     "  adaptive_dt: true\n" + marker);
        yaml += R"yaml(
domain_motion:
  tracking:
    enabled: true
    interval_s: 30
    inner_margin_m: 1000
output:
  qualification_diagnostics:
    preset: moving_reference_v1
    interval_s: 30
)yaml";
        TempYaml tmp(yaml);
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result,
                          "output.qualification_diagnostics.preset",
                          "fixed time stepping"));
    }
    SECTION("comparison-level dt must land exactly on cadence")
    {
        std::string yaml = kMinimalValidV1;
        const std::string dt = "  dt_large: 2.0\n";
        yaml.replace(yaml.find(dt), dt.size(), "  dt_large: 7.0\n");
        yaml += R"yaml(
domain_motion:
  tracking:
    enabled: true
    interval_s: 30
    inner_margin_m: 1000
output:
  qualification_diagnostics:
    preset: moving_reference_v1
    interval_s: 30
)yaml";
        TempYaml tmp(yaml);
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result,
                          "output.qualification_diagnostics.preset",
                          "exactly reaches interval_s"));
    }
    SECTION("child cadence cannot hide an unreachable root publication deadline")
    {
        std::string yaml = kMinimalValidV1;
        const std::string dt = "  dt_large: 2.0\n";
        yaml.replace(yaml.find(dt), dt.size(), "  dt_large: 12.0\n");
        yaml += R"yaml(
domain_motion:
  tracking:
    enabled: true
    interval_s: 30
    inner_margin_m: 1000
  nesting:
    enabled: true
    parent_ratio_x: 2
    parent_ratio_y: 2
    parent_time_step_ratio: 2
    child_cells_x: 24
    child_cells_y: 16
    relaxation_width_child_cells: 2
    boundary_stencil_parent_cells: 1
    relaxation_e_folding_time_s: 3
    feedback: one_way
output:
  qualification_diagnostics:
    preset: moving_reference_v1
    interval_s: 30
)yaml";
        TempYaml tmp(yaml);
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result,
                          "output.qualification_diagnostics.preset",
                          "root macrostep"));
    }
}

TEST_CASE("load_v1_config: v5 oversized reference is an admitted phase-aligned comparison domain",
          "[init][config][v1_loader][output][qualification][production_reference]")
{
    const LoadResult reference = load_v1_config(
        "tests/fixtures/configs/qualification/moving_nested_supercell_reference_v5.yaml");
    INFO(format_errors(reference.errors));
    REQUIRE(reference.ok());

    const ConfigV1& ref = *reference.config;
    constexpr int moving_parent_nx = 135;
    constexpr int moving_parent_ny = 108;
    constexpr double moving_parent_dx = 1200.0;
    constexpr double moving_parent_dy = 1200.0;
    constexpr double moving_parent_dt = 1.5;
    constexpr int moving_child_time_ratio = 2;
    constexpr int moving_child_origin_i = 2;
    constexpr int moving_child_origin_j = 1;
    REQUIRE(ref.grid.coordinate_system == CoordinateSystem::Cartesian);
    REQUIRE(ref.grid.staggering == StaggerType::CGrid);
    REQUIRE(ref.grid.nx == 269);
    REQUIRE(ref.grid.ny == 215);
    REQUIRE(ref.grid.nz == 50);
    REQUIRE(ref.grid.dx == Approx(600.0));
    REQUIRE(ref.grid.dy == Approx(600.0));
    REQUIRE(ref.time.run_time == Approx(4500.0));
    REQUIRE(ref.time.dt_large == Approx(
        moving_parent_dt / moving_child_time_ratio));
    REQUIRE_FALSE(ref.time.adaptive_dt);
    REQUIRE(ref.time.output_interval == Approx(450.0));
    REQUIRE(ref.domain_motion.mode == DomainMotionMode::Stationary);
    REQUIRE(ref.domain_motion.lifetime_planning.enabled);
    REQUIRE(ref.domain_motion.lifetime_planning.action ==
            LifetimePlanningAction::Error);
    REQUIRE(ref.domain_motion.tracking.enabled);
    REQUIRE(ref.domain_motion.tracking.interval_s == Approx(30.0));
    REQUIRE_FALSE(ref.domain_motion.nesting.enabled);
    REQUIRE(ref.output.preset == OutputPreset::ScienceValidation);
    REQUIRE(ref.output.format == OutputFormat::Npy3D);
    REQUIRE(ref.output.qualification_diagnostics_preset ==
            QualificationDiagnosticsPreset::MovingReferenceV1);
    REQUIRE(ref.output.qualification_diagnostics_interval_s == 30);

    // C-grid scalar ownership is [1, n-1). The finer reference begins one
    // 600 m cell farther west/south and reaches the moving parent's exact far
    // physical faces, so every admitted child trusted window is covered.
    REQUIRE(ref.grid.dx <= moving_parent_dx);
    REQUIRE(ref.grid.dy <= moving_parent_dy);
    REQUIRE((ref.grid.nx - 1) * ref.grid.dx == Approx(
            (moving_parent_nx - 1) * moving_parent_dx));
    REQUIRE((ref.grid.ny - 1) * ref.grid.dy == Approx(
            (moving_parent_ny - 1) * moving_parent_dy));
    REQUIRE(std::fmod(
                moving_child_origin_i * moving_parent_dx,
                ref.grid.dx) == Approx(0.0));
    REQUIRE(std::fmod(
                moving_child_origin_j * moving_parent_dy,
                ref.grid.dy) == Approx(0.0));
}

TEST_CASE("load_v1_config: trajectory controls reject unphysical values",
          "[init][config][v1_loader][output][trajectories][config-truth]")
{
    SECTION("negative seed time")
    {
        TempYaml         tmp(std::string(kMinimalValidV1) + "output:\n  trajectory_seed_time_s: -1\n");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "output.trajectory_seed_time_s"));
    }

    SECTION("zero seed count")
    {
        TempYaml         tmp(std::string(kMinimalValidV1) + "output:\n  trajectory_count: 0\n");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "output.trajectory_count"));
    }
}

TEST_CASE("load_v1_config: explicit horizontal diffusion is preserved",
          "[init][config][v1_loader][diffusion][config-truth]")
{
    TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
diffusion:
  scheme: explicit
  operator_order: 2
  K_h: 5.0
  K_v: 0.0
  apply_to: scalars
)yaml");
    const LoadResult result = load_v1_config(tmp.path.string());
    REQUIRE(result.ok());
    REQUIRE(result.config->diffusion.K_h == Approx(5.0));
}

TEST_CASE("SCI-094: strict v1 preserves an explicit active terrain request",
          "[init][config][v1_loader][terrain][config-truth][sci094]")
{
    TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
grid:
  ztop: 8000
terrain:
  scheme: bell
  coordinate: smoothed
  compute_derivatives: true
  compute_metrics: true
  smoothing_decay: 0.7
  bell:
    h0: 900
    a: 4000
    axisymmetric: true
)yaml");
    const LoadResult result = load_v1_config(tmp.path.string());
    INFO(format_errors(result.errors));
    REQUIRE(result.ok());
    REQUIRE(result.config->terrain.scheme == TerrainScheme::Bell);
    REQUIRE(result.config->terrain.coordinate == TerrainCoordinate::Smoothed);
    REQUIRE(result.config->terrain.smoothing_decay == Approx(0.7));
    REQUIRE(result.config->terrain.bell.h0 == Approx(900.0));
    REQUIRE(result.config->terrain.bell.a == Approx(4000.0));
    REQUIRE(result.config->terrain.bell.axisymmetric);
    REQUIRE(result.config->terrain.horizontal_placement ==
            TerrainHorizontalPlacement::GridAttached);
}

TEST_CASE("strict v1 terrain placement owns explicit Earth feature identity",
          "[init][config][v1_loader][terrain][earth_fixed]")
{
    SECTION("Earth-fixed analytic terrain round-trips without losing its center")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
grid:
  ztop: 8000
terrain:
  scheme: bell
  coordinate: smoothed
  horizontal_placement: earth_fixed
  feature_center_earth_x_m: 12500.25
  feature_center_earth_y_m: -8400.5
  bell:
    h0: 900
    a: 4000
    axisymmetric: true
)yaml");
        const auto loaded = load_v1_config(tmp.path.string());
        INFO(format_errors(loaded.errors));
        REQUIRE(loaded.ok());
        REQUIRE(loaded.config->terrain.horizontal_placement ==
                TerrainHorizontalPlacement::EarthFixed);
        REQUIRE(loaded.config->terrain.feature_center_earth_x_m ==
                Approx(12500.25));
        REQUIRE(loaded.config->terrain.feature_center_earth_y_m ==
                Approx(-8400.5));

        const auto projected = serialize_v1_to_v0_map(*loaded.config);
        REQUIRE(projected.at("terrain.horizontal_placement") == "earth_fixed");
        REQUIRE(projected.at("terrain.feature_center_earth_x_m") == "12500.25");
        REQUIRE(projected.at("terrain.feature_center_earth_y_m") == "-8400.5");

        TempYaml round_trip(serialize_v1_to_yaml(*loaded.config));
        const auto restored = load_v1_config(round_trip.path.string());
        INFO(format_errors(restored.errors));
        REQUIRE(restored.ok());
        REQUIRE(restored.config->terrain.horizontal_placement ==
                TerrainHorizontalPlacement::EarthFixed);
        REQUIRE(restored.config->terrain.feature_center_earth_x_m ==
                loaded.config->terrain.feature_center_earth_x_m);
        REQUIRE(restored.config->terrain.feature_center_earth_y_m ==
                loaded.config->terrain.feature_center_earth_y_m);
    }

    SECTION("Earth-fixed selection requires both explicit center coordinates")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
terrain:
  scheme: bell
  horizontal_placement: earth_fixed
  feature_center_earth_x_m: 1200
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result,
                          "terrain.feature_center_earth_y_m",
                          "required key missing"));
    }

    SECTION("Grid-attached selection rejects ignored Earth centers")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
terrain:
  scheme: none
  horizontal_placement: grid_attached
  feature_center_earth_x_m: 0
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result,
                          "terrain.feature_center_earth_x_m",
                          "is ignored"));
    }

    SECTION("Earth-fixed selection rejects unsupported scheme and metrics")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
terrain:
  scheme: none
  coordinate: cartesian
  horizontal_placement: earth_fixed
  feature_center_earth_x_m: 0
  feature_center_earth_y_m: 0
  compute_derivatives: false
  compute_metrics: false
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "terrain.scheme", "only bell or schar"));
        REQUIRE(has_error(result,
                          "terrain.horizontal_placement",
                          "complete derivative and metric"));
        REQUIRE(has_error(result,
                          "terrain.coordinate",
                          "only btf or smoothed metrics"));
    }
}

TEST_CASE("qualified mapped-terrain predicates remain grid-attached",
          "[init][config][terrain][qualification]")
{
    const auto loaded = load_v1_config(
        "tests/fixtures/configs/qualification/"
        "cgrid_stationary_bell_pbl_implicit_ready.yaml");
    INFO(format_errors(loaded.errors));
    REQUIRE(loaded.ok());
    REQUIRE(is_public_qualified_mapped_terrain_profile(*loaded.config));

    ConfigV1 earth_fixed = *loaded.config;
    earth_fixed.terrain.horizontal_placement =
        TerrainHorizontalPlacement::EarthFixed;
    earth_fixed.terrain.feature_center_earth_x_m = 3000.0;
    earth_fixed.terrain.feature_center_earth_y_m = 3000.0;
    REQUIRE_FALSE(is_public_qualified_mapped_terrain_profile(earth_fixed));
    REQUIRE_FALSE(
        is_public_qualified_native_mapped_terrain_profile(earth_fixed));
}

TEST_CASE("mapped active PBL admits horizontal prognostic TKE",
          "[init][config][terrain][qualification][pbl][tke]")
{
    const auto loaded = load_v1_config(
        "tests/fixtures/configs/qualification/"
        "cgrid_terrain_smagorinsky_coupled.yaml");
    INFO(format_errors(loaded.errors));
    REQUIRE(loaded.ok());

    ConfigV1 ysu = *loaded.config;
    ysu.surface.scheme = SurfaceScheme::Slab;
    ysu.surface.apply_fluxes = true;
    ysu.boundary_layer.scheme = BoundaryLayerScheme::YSU;
    ysu.turbulence.scheme = TurbulenceScheme::TKE;
    ysu.turbulence.mode = TurbulenceMode::TwoD;
    ysu.turbulence.stability_correction =
        TurbulenceStabilityCorrection::TkeBased;
    ysu.turbulence.Pr_t = 1.0 / 3.0;
    ysu.turbulence.Sc_t = 1.0 / 3.0;
    REQUIRE(is_public_qualified_mapped_terrain_profile(ysu));
    REQUIRE_NOTHROW(serialize_v1_to_yaml(ysu));

    ConfigV1 mynn = ysu;
    mynn.boundary_layer.scheme = BoundaryLayerScheme::MYNN;
    mynn.boundary_layer.pbl_top_method = BoundaryLayerTopMethod::TKE;
    mynn.radiation.scheme = RadiationScheme::RRTMG;
    mynn.radiation.do_lw = true;
    mynn.radiation.do_sw = false;
    mynn.radiation.clear_sky_only = false;
    REQUIRE(is_public_qualified_mapped_terrain_profile(mynn));
    REQUIRE_NOTHROW(serialize_v1_to_yaml(mynn));

    ConfigV1 duplicate_vertical_owner = mynn;
    duplicate_vertical_owner.turbulence.mode = TurbulenceMode::ThreeD;
    REQUIRE_FALSE(is_public_qualified_mapped_terrain_profile(
        duplicate_vertical_owner));

    ConfigV1 cloudy_simple_grey = ysu;
    cloudy_simple_grey.radiation.clear_sky_only = false;
    cloudy_simple_grey.radiation.simple_grey_optical_model =
        "absorbing_water_cloud_path_v1";
    REQUIRE(is_public_qualified_mapped_terrain_profile(
        cloudy_simple_grey));
    REQUIRE_NOTHROW(serialize_v1_to_yaml(cloudy_simple_grey));
}

TEST_CASE("Private recycling admission preserves qualified reconstruction and physics constraints",
          "[init][config][nesting][private][qualification]")
{
    const auto loaded = load_v1_config(
        "tests/fixtures/configs/qualification/nested_grid_recycling_thompson_two_way_tvd_cpu_ready.yaml");
    INFO(format_errors(loaded.errors));
    REQUIRE(loaded.ok());
    const auto config = *loaded.config;
    REQUIRE(is_public_ready_private_two_way_profile(config));
    REQUIRE(is_public_qualified_accelerated_two_way_profile(config));
    REQUIRE(is_public_qualified_nested_grid_recycling_profile(config));

    auto weno = config;
    weno.advection.scheme = AdvectionScheme::WENO5;
    CHECK(is_public_ready_private_two_way_profile(weno));
    CHECK(is_public_qualified_accelerated_two_way_profile(weno));
    CHECK(is_public_qualified_nested_grid_recycling_profile(weno));
    auto unsupported = weno;
    unsupported.microphysics.parameters.thompson.evaporation_rate_s_inv = .002;
    CHECK_FALSE(is_public_ready_private_two_way_profile(unsupported));
    CHECK_FALSE(is_public_qualified_accelerated_two_way_profile(unsupported));
    CHECK_FALSE(is_public_qualified_nested_grid_recycling_profile(unsupported));
}

TEST_CASE("mapped coupled family admits evidenced microphysics owners",
          "[init][config][terrain][qualification][microphysics]")
{
    const auto loaded = load_v1_config(
        "tests/fixtures/configs/qualification/"
        "cgrid_terrain_smagorinsky_coupled.yaml");
    INFO(format_errors(loaded.errors));
    REQUIRE(loaded.ok());

    for (const auto scheme : {
             MicrophysicsScheme::None,
             MicrophysicsScheme::Kessler,
             MicrophysicsScheme::Lin,
             MicrophysicsScheme::Thompson,
             MicrophysicsScheme::Milbrandt})
    {
        ConfigV1 config = *loaded.config;
        config.microphysics.scheme = scheme;
        CAPTURE(microphysics_scheme_name(scheme));
        REQUIRE(is_public_qualified_mapped_terrain_profile(config));
        REQUIRE_NOTHROW(serialize_v1_to_yaml(config));
    }
}

TEST_CASE("mapped coupled family admits every retained chaos owner",
          "[init][config][terrain][qualification][chaos]")
{
    const auto loaded = load_v1_config(
        "tests/fixtures/configs/qualification/"
        "cgrid_terrain_smagorinsky_coupled.yaml");
    INFO(format_errors(loaded.errors));
    REQUIRE(loaded.ok());
    for (const auto scheme : {
             ChaosScheme::None,
             ChaosScheme::InitialConditions,
             ChaosScheme::BoundaryLayer,
             ChaosScheme::FullStochastic})
    {
        ConfigV1 config = *loaded.config;
        config.extensions.chaos.scheme = scheme;
        config.extensions.chaos.seed = 324;
        config.extensions.chaos.member_id = 2;
        CAPTURE(chaos_scheme_name(scheme));
        REQUIRE(is_public_qualified_mapped_terrain_profile(config));
        REQUIRE_NOTHROW(serialize_v1_to_yaml(config));
    }
}

TEST_CASE("exact active collocated terrain admits the canonical chaos axis",
          "[init][config][terrain][qualification][chaos][collocated]")
{
    const std::array<const char*, 2> fixtures = {
        "tests/fixtures/configs/qualification/"
        "agrid_mapped_full_stochastic_ready.yaml",
        "tests/fixtures/configs/qualification/"
        "agrid_mapped_full_stochastic_schar_thompson_ready.yaml",
    };
    const std::array<MicrophysicsScheme, 5> microphysics = {
        MicrophysicsScheme::None,
        MicrophysicsScheme::Kessler,
        MicrophysicsScheme::Lin,
        MicrophysicsScheme::Thompson,
        MicrophysicsScheme::Milbrandt,
    };
    const std::array<ChaosScheme, 4> chaos_schemes = {
        ChaosScheme::None,
        ChaosScheme::InitialConditions,
        ChaosScheme::BoundaryLayer,
        ChaosScheme::FullStochastic,
    };

    for (const char* fixture : fixtures)
    {
        const auto loaded = load_v1_config(fixture);
        INFO(format_errors(loaded.errors));
        REQUIRE(loaded.ok());
        for (const MicrophysicsScheme microphysics_scheme : microphysics)
        {
            for (const ChaosScheme chaos_scheme : chaos_schemes)
            {
                CAPTURE(
                    fixture, microphysics_scheme_name(microphysics_scheme),
                    chaos_scheme_name(chaos_scheme));
                ConfigV1 implicit_targets = *loaded.config;
                implicit_targets.microphysics.scheme = microphysics_scheme;
                implicit_targets.extensions.chaos.scheme = chaos_scheme;
                implicit_targets.extensions.chaos.apply_to_ic.clear();
                implicit_targets.extensions.chaos.apply_to_tendencies.clear();
                REQUIRE(is_public_qualified_collocated_mapped_terrain_profile(
                    implicit_targets));
                REQUIRE_NOTHROW(serialize_v1_to_yaml(implicit_targets));

                ConfigV1 explicit_targets = implicit_targets;
                const std::string scheme_id = chaos_scheme_name(chaos_scheme);
                explicit_targets.extensions.chaos.apply_to_ic =
                    tmv::chaos::canonical_chaos_initial_condition_targets(
                        scheme_id);
                explicit_targets.extensions.chaos.apply_to_tendencies =
                    tmv::chaos::canonical_chaos_tendency_targets(scheme_id);
                REQUIRE(is_public_qualified_collocated_mapped_terrain_profile(
                    explicit_targets));
                REQUIRE_NOTHROW(serialize_v1_to_yaml(explicit_targets));
            }
        }
    }

    const auto loaded = load_v1_config(fixtures.front());
    REQUIRE(loaded.ok());
    ConfigV1 noncanonical = *loaded.config;
    noncanonical.extensions.chaos.scheme = ChaosScheme::InitialConditions;
    noncanonical.extensions.chaos.apply_to_ic = {"theta"};
    noncanonical.extensions.chaos.apply_to_tendencies.clear();
    REQUIRE_FALSE(is_public_qualified_collocated_mapped_terrain_profile(
        noncanonical));
    REQUIRE_THROWS(serialize_v1_to_yaml(noncanonical));

    noncanonical = *loaded.config;
    noncanonical.extensions.chaos.scheme = ChaosScheme::BoundaryLayer;
    noncanonical.extensions.chaos.apply_to_ic.clear();
    noncanonical.extensions.chaos.apply_to_tendencies = {"diffusion"};
    REQUIRE_FALSE(is_public_qualified_collocated_mapped_terrain_profile(
        noncanonical));
    REQUIRE_THROWS(serialize_v1_to_yaml(noncanonical));

    noncanonical = *loaded.config;
    noncanonical.extensions.chaos.scheme = ChaosScheme::FullStochastic;
    noncanonical.extensions.chaos.apply_to_tendencies = {"pbl"};
    REQUIRE_FALSE(is_public_qualified_collocated_mapped_terrain_profile(
        noncanonical));
    REQUIRE_THROWS(serialize_v1_to_yaml(noncanonical));

    noncanonical = *loaded.config;
    noncanonical.extensions.chaos.native_generation_required = true;
    REQUIRE_FALSE(is_public_qualified_collocated_mapped_terrain_profile(
        noncanonical));
    REQUIRE_THROWS(serialize_v1_to_yaml(noncanonical));

    const auto inactive = load_v1_config(
        "tests/fixtures/configs/qualification/"
        "agrid_mapped_none_bell_thompson_ready.yaml");
    INFO(format_errors(inactive.errors));
    REQUIRE(inactive.ok());
    REQUIRE(is_public_qualified_mapped_terrain_profile(*inactive.config));
}

TEST_CASE("exact active collocated terrain admits the public advection axis",
          "[init][config][terrain][qualification][advection][collocated]")
{
    const std::array<const char*, 2> fixtures = {
        "tests/fixtures/configs/qualification/"
        "agrid_mapped_full_stochastic_ready.yaml",
        "tests/fixtures/configs/qualification/"
        "agrid_mapped_full_stochastic_schar_thompson_ready.yaml",
    };
    const std::array<AdvectionScheme, 3> public_schemes = {
        AdvectionScheme::None,
        AdvectionScheme::TVD,
        AdvectionScheme::WENO5,
    };

    for (const char* fixture : fixtures)
    {
        const auto loaded = load_v1_config(fixture);
        INFO(format_errors(loaded.errors));
        REQUIRE(loaded.ok());
        for (const AdvectionScheme scheme : public_schemes)
        {
            CAPTURE(fixture, advection_scheme_name(scheme));
            ConfigV1 config = *loaded.config;
            config.advection.scheme = scheme;
            REQUIRE(is_public_qualified_collocated_mapped_terrain_profile(
                config));
            REQUIRE_NOTHROW(serialize_v1_to_yaml(config));
        }
    }

    const auto loaded = load_v1_config(fixtures.front());
    REQUIRE(loaded.ok());
    ConfigV1 neighbor = *loaded.config;
    neighbor.advection.scheme = AdvectionScheme::Auto;
    REQUIRE_FALSE(is_public_qualified_collocated_mapped_terrain_profile(
        neighbor));
    REQUIRE_THROWS(serialize_v1_to_yaml(neighbor));

    neighbor = *loaded.config;
    neighbor.extensions.compute.backend = ComputeBackend::Metal;
    REQUIRE_FALSE(is_public_qualified_collocated_mapped_terrain_profile(
        neighbor));
    REQUIRE_THROWS(serialize_v1_to_yaml(neighbor));

    neighbor = *loaded.config;
    neighbor.grid.dx = 501.0;
    REQUIRE_FALSE(is_public_qualified_collocated_mapped_terrain_profile(
        neighbor));
    REQUIRE_THROWS(serialize_v1_to_yaml(neighbor));
}

TEST_CASE("mapped coupled family admits vertical implicit scalar diffusion",
          "[init][config][terrain][qualification][diffusion][implicit]")
{
    const auto loaded = load_v1_config(
        "tests/fixtures/configs/qualification/"
        "cgrid_terrain_smagorinsky_coupled.yaml");
    INFO(format_errors(loaded.errors));
    REQUIRE(loaded.ok());
    ConfigV1 config = *loaded.config;
    config.diffusion.scheme = DiffusionScheme::Implicit;
    config.diffusion.K_h = 0.0;
    config.diffusion.K_v = 160.0;
    config.diffusion.apply_to = DiffusionTarget::Scalars;
    REQUIRE(is_public_qualified_mapped_terrain_profile(config));
    REQUIRE_NOTHROW(serialize_v1_to_yaml(config));

    config.diffusion.K_h = 1.0;
    REQUIRE(is_public_qualified_mapped_terrain_profile(config));
    config.diffusion.apply_to = DiffusionTarget::ScalarsAndMomentum;
    REQUIRE(is_public_qualified_mapped_terrain_profile(config));
    config.diffusion.K_h = -1.0;
    REQUIRE_FALSE(is_public_qualified_mapped_terrain_profile(config));
}

TEST_CASE("Earth-fixed terrain recycling has one exact qualified predicate",
          "[init][config][terrain][domain_recycling][qualification]")
{
    const auto loaded = load_v1_config(
        "tests/fixtures/configs/qualification/"
        "earth_fixed_terrain_recycling_ready.yaml");
    INFO(format_errors(loaded.errors));
    REQUIRE(loaded.ok());
    REQUIRE(is_public_qualified_earth_fixed_terrain_recycling_profile(
        *loaded.config));
    REQUIRE_FALSE(is_public_qualified_domain_recycling_profile(
        *loaded.config));
    REQUIRE_FALSE(is_public_qualified_mapped_terrain_profile(
        *loaded.config));
    REQUIRE_NOTHROW(require_runtime_supported_domain_motion(*loaded.config));

    ConfigV1 translating = *loaded.config;
    translating.extensions.execution.mode = ExecutionMode::Exploratory;
    translating.domain_motion.mode = DomainMotionMode::ConstantVelocity;
    translating.domain_motion.frame_velocity.u_ms = 5.0;
    REQUIRE_THROWS_WITH(require_runtime_supported_domain_motion(translating),
        Catch::Matchers::Contains("maintained Bell remapping profile"));

    const auto reject = [&](const auto& mutate)
    {
        ConfigV1 changed = *loaded.config;
        mutate(changed);
        REQUIRE_FALSE(
            is_public_qualified_earth_fixed_terrain_recycling_profile(
                changed));
    };
    reject([](ConfigV1& config) {
        config.domain_motion.tracking.maximum_shift_cells = 7;
    });
    reject([](ConfigV1& config) {
        config.terrain.feature_center_earth_x_m += 1.0;
    });
    reject([](ConfigV1& config) {
        config.radiation.cadence_s = 3.0;
    });
    reject([](ConfigV1& config) {
        config.boundary_layer.scheme = BoundaryLayerScheme::YSU;
    });
    reject([](ConfigV1& config) {
        config.extensions.chaos.seed += 1;
    });
    reject([](ConfigV1& config) {
        config.extensions.chaos.alpha_tend["pbl"] = 0.21;
    });
}

TEST_CASE("nested-grid recycling has one exact qualified predicate",
          "[init][config][domain_recycling][nested][qualification]")
{
    const auto loaded = load_v1_config(
        "tests/fixtures/configs/qualification/"
        "nested_grid_recycling_ready.yaml");
    INFO(format_errors(loaded.errors));
    REQUIRE(loaded.ok());
    REQUIRE(is_public_qualified_nested_grid_recycling_profile(
        *loaded.config));
    REQUIRE_FALSE(is_public_qualified_domain_recycling_profile(
        *loaded.config));
    REQUIRE_FALSE(is_public_qualified_two_way_nested_profile(*loaded.config));
    REQUIRE_NOTHROW(require_runtime_supported_domain_motion(*loaded.config));

    const auto reject = [&](const auto& mutate)
    {
        ConfigV1 changed = *loaded.config;
        mutate(changed);
        REQUIRE_FALSE(is_public_qualified_nested_grid_recycling_profile(
            changed));
    };
    reject([](ConfigV1& config) {
        config.domain_motion.nesting.child_origin_parent_cell_i = 10;
    });
    reject([](ConfigV1& config) {
        config.domain_motion.nesting.feedback = NestingFeedback::TwoWay;
    });
    reject([](ConfigV1& config) {
        config.domain_motion.tracking.maximum_shift_cells = 5;
    });
    reject([](ConfigV1& config) {
        config.extensions.chaos.seed += 1;
    });
}

TEST_CASE("qualified native mapped-terrain predicates distinguish Vulkan and Metal",
          "[init][config][terrain][qualification][native]")
{
    const auto loaded = load_v1_config(
        "tests/fixtures/configs/qualification/"
        "cgrid_stationary_bell_vulkan_complete_ready.yaml");
    INFO(format_errors(loaded.errors));
    REQUIRE(loaded.ok());
    REQUIRE(is_public_qualified_native_mapped_terrain_profile(*loaded.config));
    REQUIRE(is_public_qualified_vulkan_mapped_terrain_profile(*loaded.config));
    REQUIRE_FALSE(is_public_qualified_metal_mapped_terrain_profile(*loaded.config));

    ConfigV1 metal = *loaded.config;
    metal.extensions.compute.backend = ComputeBackend::Metal;
    REQUIRE(is_public_qualified_mapped_terrain_profile(metal));
    REQUIRE(is_public_qualified_native_mapped_terrain_profile(metal));
    REQUIRE_FALSE(is_public_qualified_vulkan_mapped_terrain_profile(metal));
    REQUIRE(is_public_qualified_metal_mapped_terrain_profile(metal));
}

TEST_CASE("SCI-094: strict v1 rejects nonphysical active terrain contracts",
          "[init][config][v1_loader][terrain][config-truth][sci094]")
{
    SECTION("active terrain cannot suppress metrics or derivatives")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
grid:
  ztop: 8000
terrain:
  scheme: schar
  compute_derivatives: false
  compute_metrics: false
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "terrain.compute_derivatives", "active terrain requires"));
        REQUIRE(has_error(result, "terrain.compute_metrics", "active terrain requires"));
    }

    SECTION("active terrain cannot use a flat coordinate")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
grid:
  ztop: 8000
terrain:
  scheme: bell
  coordinate: cartesian
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "terrain.coordinate", "would ignore topography"));
    }

    SECTION("active terrain top must match the computational grid")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
terrain:
  scheme: bell
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "grid.ztop", "grid.nz * grid.dz"));
    }

    SECTION("terrain must stay below the lid")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
grid:
  ztop: 8000
terrain:
  scheme: schar
  schar:
    H: 8000
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "terrain.schar.H", "below grid.ztop"));
    }

}

TEST_CASE("load_v1_config: surface and boundary-layer ownership is explicit",
          "[init][config][v1_loader][surface][pbl][config-truth]")
{
    SECTION("surface fluxes cannot be active without a surface scheme")
    {
        TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
surface:
  scheme: none
  apply_fluxes: true
boundary_layer:
  scheme: slab
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "surface.apply_fluxes", "cannot be true when surface.scheme=none"));
    }

    SECTION("surface scheme cannot silently select a PBL closure")
    {
        TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
surface:
  scheme: slab
  apply_fluxes: true
boundary_layer:
  scheme: none
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "boundary_layer.scheme", "select boundary_layer.scheme=slab explicitly"));
    }

    SECTION("explicit slab surface and PBL selection remains accepted")
    {
        TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
surface:
  scheme: slab
  apply_fluxes: true
boundary_layer:
  scheme: slab
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE(result.ok());
    }

    SECTION("potential-temperature PBL-top diagnosis is typed")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
boundary_layer:
  scheme: slab
  pbl_top_method: theta
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->boundary_layer.pbl_top_method ==
                BoundaryLayerTopMethod::PotentialTemperature);
    }

    SECTION("TKE PBL-top diagnosis requires MYNN")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
boundary_layer:
  scheme: slab
  pbl_top_method: tke
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(
            result, "boundary_layer.pbl_top_method", "requires"));
    }

    SECTION("MYNN accepts TKE PBL-top diagnosis")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
boundary_layer:
  scheme: mynn
  pbl_top_method: tke
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->boundary_layer.pbl_top_method ==
                BoundaryLayerTopMethod::TKE);
    }

    SECTION("YSU accepts typed cloud-top mixing cadence")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
boundary_layer:
  scheme: ysu
  cadence_s: 0.25
  clear_sky_only: false
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->boundary_layer.cadence_s == Approx(0.25));
        REQUIRE_FALSE(result.config->boundary_layer.clear_sky_only);
    }

    SECTION("cloud-top mixing rejects non-YSU ownership")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
boundary_layer:
  scheme: slab
  clear_sky_only: false
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(
            result, "boundary_layer.clear_sky_only", "requires"));
    }

    SECTION("boundary-layer cadence must be positive and finite")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
boundary_layer:
  scheme: ysu
  cadence_s: 0
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "boundary_layer.cadence_s", "scheduling tolerance"));
    }
}

TEST_CASE("load_v1_config: implicit horizontal scalar diffusion is preserved",
          "[init][config][v1_loader][diffusion][config-truth]")
{
    TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
diffusion:
  scheme: implicit
  operator_order: 2
  K_h: 5.0
  K_v: 0.0
  apply_to: scalars
)yaml");
    const LoadResult result = load_v1_config(tmp.path.string());
    REQUIRE(result.ok());
    REQUIRE(result.config->diffusion.scheme == DiffusionScheme::Implicit);
    REQUIRE(result.config->diffusion.K_h == 5.0);
    REQUIRE(result.config->diffusion.apply_to == DiffusionTarget::Scalars);
}

TEST_CASE("load_v1_config: implicit horizontal momentum diffusion is preserved",
          "[init][config][v1_loader][diffusion][config-truth]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
diffusion:
  scheme: implicit
  operator_order: 2
  K_h: 5.0
  K_v: 0.0
  apply_to: scalars_and_momentum
)yaml");
    const LoadResult result = load_v1_config(tmp.path.string());
    REQUIRE(result.ok());
    REQUIRE(result.config->diffusion.scheme == DiffusionScheme::Implicit);
    REQUIRE(result.config->diffusion.K_h == 5.0);
    REQUIRE(result.config->diffusion.apply_to ==
            DiffusionTarget::ScalarsAndMomentum);
}

TEST_CASE("load_v1_config: explicit horizontal momentum diffusion is preserved",
          "[init][config][v1_loader][diffusion][config-truth]")
{
    TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
diffusion:
  scheme: explicit
  operator_order: 2
  K_h: 5.0
  K_v: 0.0
  apply_to: scalars_and_momentum
)yaml");
    const LoadResult result = load_v1_config(tmp.path.string());
    REQUIRE(result.ok());
    REQUIRE(result.config->diffusion.K_h == Approx(5.0));
    REQUIRE(result.config->diffusion.apply_to == DiffusionTarget::ScalarsAndMomentum);
}

TEST_CASE("load_v1_config: linear-height diffusion has an explicit sealed contract",
          "[init][config][v1_loader][diffusion][variable-k]")
{
    SECTION("complete profile is accepted")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
diffusion:
  scheme: explicit
  operator_order: 2
  K_h: 7
  K_v: 0
  apply_to: scalars_and_momentum
  coefficient_source: linear_height
  K_v_surface: 4
  K_v_top: 20
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->diffusion.coefficient_source ==
                DiffusionCoefficientSource::LinearHeight);
        REQUIRE(result.config->diffusion.K_v_surface == 4.0);
        REQUIRE(result.config->diffusion.K_v_top == 20.0);
    }

    SECTION("constant K_v cannot remain active")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
diffusion:
  scheme: explicit
  K_v: 1
  coefficient_source: linear_height
  K_v_surface: 4
  K_v_top: 20
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "diffusion.K_v", "must be zero"));
    }

    SECTION("both endpoints are required")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
diffusion:
  scheme: explicit
  coefficient_source: linear_height
  K_v_surface: 4
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "diffusion.K_v_top", "required"));
    }

    SECTION("profile endpoints cannot decorate constant diffusion")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
diffusion:
  scheme: explicit
  coefficient_source: constant
  K_v_surface: 4
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "diffusion.K_v_surface", "linear_height"));
    }
}

TEST_CASE("all shipped schema-v1 YAML presets pass the strict loader", "[init][config][v1_loader][presets]")
{
    namespace fs        = std::filesystem;
    const fs::path root = "configs";
    if (!fs::exists(root))
    {
        WARN("configs directory not available from this test working directory");
        return;
    }

    int checked = 0;
    for (const auto& entry : fs::recursive_directory_iterator(root))
    {
        if (!entry.is_regular_file() || (entry.path().extension() != ".yaml" && entry.path().extension() != ".yml"))
        {
            continue;
        }

        std::ifstream     input(entry.path());
        const std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        if (contents.find("schema: nimbus.v1") == std::string::npos)
        {
            continue;
        }

        ++checked;
        const auto result = load_v1_config(entry.path().string());
        INFO("preset: " << entry.path().string());
        INFO(format_errors(result.errors));
#ifndef HAVE_ZFP
        if (contents.find("format: zfp") != std::string::npos)
        {
            REQUIRE_FALSE(result.ok());
            REQUIRE(has_error(result, "output.format", "unavailable"));
            continue;
        }
#endif
        REQUIRE(result.ok());
    }
    REQUIRE(checked > 0);
}

// ─────────────────────────────────────────────────────────────
// Happy path: minimal valid v1 config
// ─────────────────────────────────────────────────────────────

TEST_CASE("load_v1_config: minimal valid config loads with CM1 defaults", "[init][config][v1_loader]")
{
    TempYaml tmp(kMinimalValidV1);

    auto result = load_v1_config(tmp.path.string());

    REQUIRE(result.ok());
    REQUIRE(result.errors.empty());
    REQUIRE(result.config.has_value());

    const auto& cfg = *result.config;
    REQUIRE(cfg.schema_version == "nimbus.v1");

    // Required values present
    REQUIRE(cfg.grid.nx == 64);
    REQUIRE(cfg.grid.ny == 64);
    REQUIRE(cfg.grid.nz == 32);
    REQUIRE(cfg.grid.dx == Approx(250.0));
    REQUIRE(cfg.time.run_time == Approx(7200.0));
    REQUIRE(cfg.time.dt_large == Approx(2.0));

    // CM1 defaults preserved for omitted keys
    REQUIRE(cfg.grid.ztop == Approx(16000.0));
    REQUIRE(cfg.time.acoustic_substeps == 6);
    REQUIRE(cfg.time.cfl_safety == Approx(0.7));
    REQUIRE(cfg.domain_motion.mode == DomainMotionMode::Stationary);
    REQUIRE(cfg.domain_motion.frame_velocity.u_ms == Approx(0.0));
    REQUIRE(cfg.domain_motion.frame_velocity.v_ms == Approx(0.0));
    REQUIRE_FALSE(cfg.domain_motion.lifetime_planning.enabled);
    REQUIRE_FALSE(cfg.domain_motion.tracking.enabled);
    REQUIRE(cfg.dynamics.scheme == "klemp_wilhelmson_split");
    REQUIRE(cfg.dynamics.axisymmetric == false);
    REQUIRE(cfg.damping.enabled == true);
    REQUIRE(cfg.damping.alpha == Approx(0.05));
    REQUIRE(cfg.damping.lateral_width == 20);
    REQUIRE(cfg.damping.lateral_alpha == Approx(0.05));
    REQUIRE(cfg.turbulence.scheme == TurbulenceScheme::Smagorinsky);
    REQUIRE(cfg.turbulence.Cs == Approx(0.18));
    REQUIRE(cfg.coriolis.enabled == false);
    REQUIRE(cfg.microphysics.scheme == MicrophysicsScheme::None);
    REQUIRE(cfg.radiation.scheme == RadiationScheme::None);
    REQUIRE(cfg.boundary_layer.scheme == BoundaryLayerScheme::Slab);
    REQUIRE(cfg.boundary_conditions.lateral.west == LateralBC::Radiative);
    REQUIRE(cfg.boundary_conditions.bottom == BottomBC::RigidNoSlip);
    REQUIRE(cfg.extensions.compute.backend == ComputeBackend::CPU);
    REQUIRE(cfg.extensions.compute.memory_admission == MemoryAdmissionPolicy::Warn);
    REQUIRE(cfg.extensions.diagnostics.scientific_every_steps == -1);
    REQUIRE_FALSE(cfg.extensions.diagnostics.deterministic_reductions);
    REQUIRE_FALSE(cfg.extensions.performance.timing.has_value());
    REQUIRE_FALSE(cfg.extensions.performance.report_every_steps.has_value());
    REQUIRE_FALSE(cfg.extensions.performance.warmup_steps.has_value());
}

TEST_CASE("domain-motion contract is typed and unsupported execution fails closed",
          "[init][config][v1_loader][domain_motion]")
{
    SECTION("explicit stationary defaults are accepted")
    {
        TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  mode: stationary
  frame_velocity:
    u_ms: 0
    v_ms: 0
  lifetime_planning:
    enabled: false
    action: error
    motion_source: bunkers_left
    explicit_u_ms: 11
    explicit_v_ms: -3
    upstream_buffer_m: 20000
    downstream_buffer_m: 30000
    cross_track_buffer_m: 25000
    motion_uncertainty_ms: 5
  tracking:
    enabled: false
    recycle_domain: false
    interval_s: 60
    inner_margin_m: 20000
    maximum_shift_cells: 8
    cooldown_s: 300
    minimum_confidence: 0.8
    minimum_updraft_ms: 6
    minimum_condensate_path_kgm2: 0.2
    updraft_helicity_bottom_m_agl: 1500
    updraft_helicity_top_m_agl: 4500
    minimum_absolute_updraft_helicity_m2s2: 35
    minimum_component_cells: 6
    minimum_core_area_m2: 2500000
    maximum_motion_speed_ms: 65
    maximum_motion_acceleration_ms2: 0.2
    minimum_continuity_samples: 4
    lost_after_samples: 5
    topology_confirmation_samples: 3
    branch_preference: bunkers_left
    branch_velocity_uncertainty_ms: 4.5
  nesting:
    parent_ratio_x: 3
    parent_ratio_y: 2
    parent_time_step_ratio: 4
    feedback: one_way
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->domain_motion.mode == DomainMotionMode::Stationary);
        REQUIRE(result.config->domain_motion.lifetime_planning.motion_source == StormMotionSource::BunkersLeft);
        REQUIRE(result.config->domain_motion.lifetime_planning.upstream_buffer_m == Approx(20000.0));
        REQUIRE(result.config->domain_motion.tracking.maximum_shift_cells == 8);
        REQUIRE_FALSE(result.config->domain_motion.tracking.recycle_domain);
        REQUIRE(result.config->domain_motion.tracking.minimum_updraft_ms == Approx(6.0));
        REQUIRE(result.config->domain_motion.tracking.updraft_helicity_bottom_m_agl == Approx(1500.0));
        REQUIRE(result.config->domain_motion.tracking.updraft_helicity_top_m_agl == Approx(4500.0));
        REQUIRE(result.config->domain_motion.tracking.minimum_absolute_updraft_helicity_m2s2 == Approx(35.0));
        REQUIRE(result.config->domain_motion.tracking.minimum_component_cells == 6);
        REQUIRE(result.config->domain_motion.tracking.minimum_core_area_m2 == Approx(2500000.0));
        REQUIRE(result.config->domain_motion.tracking.maximum_motion_speed_ms == Approx(65.0));
        REQUIRE(result.config->domain_motion.tracking.maximum_motion_acceleration_ms2 == Approx(0.2));
        REQUIRE(result.config->domain_motion.tracking.minimum_continuity_samples == 4);
        REQUIRE(result.config->domain_motion.tracking.lost_after_samples == 5);
        REQUIRE(result.config->domain_motion.tracking.topology_confirmation_samples == 3);
        REQUIRE(result.config->domain_motion.tracking.branch_preference == StormBranchPreference::BunkersLeft);
        REQUIRE(result.config->domain_motion.tracking.branch_velocity_uncertainty_ms == Approx(4.5));
        REQUIRE(result.config->domain_motion.nesting.parent_ratio_x == 3);
        REQUIRE(result.config->domain_motion.nesting.parent_ratio_y == 2);
        REQUIRE(result.config->domain_motion.nesting.parent_time_step_ratio == 4);
        REQUIRE(result.config->domain_motion.nesting.feedback == NestingFeedback::OneWay);
    }

    SECTION("constant translation is accepted and retained")
    {
        TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  mode: constant_velocity
  frame_velocity:
    u_ms: 12
    v_ms: -4
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->domain_motion.mode == DomainMotionMode::ConstantVelocity);
        REQUIRE(result.config->domain_motion.frame_velocity.u_ms == Approx(12.0));
        REQUIRE(result.config->domain_motion.frame_velocity.v_ms == Approx(-4.0));
    }

    SECTION("adaptive moving-window execution is retained as discrete recycling")
    {
        TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  mode: tracked_window
  tracking:
    enabled: true
    recycle_domain: true
    inner_margin_m: 5000
    maximum_shift_cells: 8
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->domain_motion.mode == DomainMotionMode::TrackedWindow);
        REQUIRE(result.config->domain_motion.tracking.recycle_domain);
    }

    SECTION("tracked window fails closed without live recycling authority")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  mode: tracked_window
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "domain_motion.tracking.enabled", "requires"));
        REQUIRE(has_error(result, "domain_motion.tracking.recycle_domain", "requires"));
    }

    SECTION("tracked window qualification is decided by full-profile admission")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  mode: tracked_window
  tracking:
    enabled: true
    recycle_domain: true
    inner_margin_m: 5000
    maximum_shift_cells: 8
extensions:
  execution:
    mode: qualified
    acknowledge_maturity_limits: true
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->domain_motion.mode ==
                DomainMotionMode::TrackedWindow);
        REQUIRE(result.config->extensions.execution.mode ==
                ExecutionMode::Qualified);
    }

    SECTION("stationary mode cannot hide a non-zero frame velocity")
    {
        TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  mode: stationary
  frame_velocity:
    u_ms: 12
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "domain_motion.frame_velocity.u_ms", "requires zero frame velocity"));
    }

    SECTION("lifetime enforcement is accepted with a physical envelope")
    {
        TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  lifetime_planning:
    enabled: true
    motion_source: explicit
    explicit_u_ms: 12
    explicit_v_ms: -4
    upstream_buffer_m: 10000
    downstream_buffer_m: 20000
    cross_track_buffer_m: 15000
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->domain_motion.lifetime_planning.enabled);
        REQUIRE(result.config->domain_motion.lifetime_planning.motion_source == StormMotionSource::Explicit);
        REQUIRE(result.config->domain_motion.lifetime_planning.explicit_u_ms == Approx(12.0));
    }

    SECTION("enabled lifetime enforcement rejects a point-sized envelope")
    {
        TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  lifetime_planning:
    enabled: true
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(
            has_error(result, "domain_motion.lifetime_planning.upstream_buffer_m", "not a storm-lifetime envelope"));
    }

    SECTION("qualified lifetime enforcement cannot be warning-only")
    {
        TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  lifetime_planning:
    enabled: true
    action: warn
    motion_source: explicit
    upstream_buffer_m: 10000
    downstream_buffer_m: 20000
    cross_track_buffer_m: 15000
extensions:
  execution:
    mode: qualified
    acknowledge_maturity_limits: true
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(
            has_error(result, "domain_motion.lifetime_planning.action", "qualified execution requires action=error"));
    }

    SECTION("diagnostic storm tracking is accepted explicitly")
    {
        TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  tracking:
    enabled: true
    inner_margin_m: 15000
    minimum_updraft_ms: 7
    minimum_condensate_path_kgm2: 0.25
    updraft_helicity_bottom_m_agl: 1000
    updraft_helicity_top_m_agl: 4000
    minimum_absolute_updraft_helicity_m2s2: 30
    minimum_component_cells: 9
    maximum_motion_speed_ms: 70
    maximum_motion_acceleration_ms2: 0.15
    minimum_continuity_samples: 4
    lost_after_samples: 6
    topology_confirmation_samples: 2
    branch_preference: bunkers_right
    branch_velocity_uncertainty_ms: 3
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->domain_motion.tracking.enabled);
        REQUIRE(result.config->domain_motion.tracking.inner_margin_m == Approx(15000.0));
        REQUIRE(result.config->domain_motion.tracking.minimum_updraft_ms == Approx(7.0));
        REQUIRE(result.config->domain_motion.tracking.updraft_helicity_bottom_m_agl == Approx(1000.0));
        REQUIRE(result.config->domain_motion.tracking.updraft_helicity_top_m_agl == Approx(4000.0));
        REQUIRE(result.config->domain_motion.tracking.minimum_absolute_updraft_helicity_m2s2 == Approx(30.0));
        REQUIRE(result.config->domain_motion.tracking.minimum_component_cells == 9);
        REQUIRE(result.config->domain_motion.tracking.maximum_motion_speed_ms == Approx(70.0));
        REQUIRE(result.config->domain_motion.tracking.maximum_motion_acceleration_ms2 == Approx(0.15));
        REQUIRE(result.config->domain_motion.tracking.minimum_continuity_samples == 4);
        REQUIRE(result.config->domain_motion.tracking.lost_after_samples == 6);
        REQUIRE(result.config->domain_motion.tracking.topology_confirmation_samples == 2);
        REQUIRE(result.config->domain_motion.tracking.branch_preference == StormBranchPreference::BunkersRight);
        REQUIRE(result.config->domain_motion.tracking.branch_velocity_uncertainty_ms == Approx(3.0));
    }

    SECTION("updraft-helicity AGL layer must have positive ordered depth")
    {
        TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  tracking:
    updraft_helicity_bottom_m_agl: 5000
    updraft_helicity_top_m_agl: 2000
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(
            result, "domain_motion.tracking.updraft_helicity_top_m_agl", "must exceed updraft_helicity_bottom_m_agl"));
    }

    SECTION("continuity history length outside the fixed-capacity range is rejected")
    {
        TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  tracking:
    minimum_continuity_samples: 1
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "domain_motion.tracking.minimum_continuity_samples", "integer from 2 through 5"));
    }

    SECTION("diagnostic tracker is available in qualified execution")
    {
        TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  tracking:
    enabled: true
    inner_margin_m: 15000
extensions:
  execution:
    mode: qualified
    acknowledge_maturity_limits: true
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->domain_motion.tracking.enabled);
    }

    SECTION("branch controls are strict")
    {
        TempYaml         invalid_preference(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  tracking:
    branch_preference: helicity_sign
)yaml");
        const LoadResult preference_result = load_v1_config(invalid_preference.path.string());
        REQUIRE_FALSE(preference_result.ok());
        REQUIRE(has_error(
            preference_result, "domain_motion.tracking.branch_preference", "continuity, bunkers_right, bunkers_left"));

        TempYaml         invalid_confirmation(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  tracking:
    topology_confirmation_samples: 0
)yaml");
        const LoadResult confirmation_result = load_v1_config(invalid_confirmation.path.string());
        REQUIRE_FALSE(confirmation_result.ok());
        REQUIRE(has_error(confirmation_result, "domain_motion.tracking.topology_confirmation_samples", "must be > 0"));
    }

    SECTION("confidence and buffer bounds are enforced")
    {
        TempYaml         tmp(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  lifetime_planning:
    upstream_buffer_m: -1
  tracking:
    minimum_confidence: 1.1
)yaml");
        const LoadResult result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "domain_motion.lifetime_planning.upstream_buffer_m", "must be >= 0"));
        REQUIRE(has_error(result, "domain_motion.tracking.minimum_confidence", "must be in [0, 1]"));
    }

    SECTION("nesting ratios and feedback are strict")
    {
        TempYaml invalid_ratio(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  nesting:
    parent_ratio_x: 4
    parent_ratio_y: 1
    parent_time_step_ratio: 3
)yaml");
        const LoadResult ratio_result = load_v1_config(invalid_ratio.path.string());
        REQUIRE_FALSE(ratio_result.ok());
        REQUIRE(has_error(
            ratio_result, "domain_motion.nesting.parent_ratio_y", "at least 2"));
        REQUIRE(has_error(
            ratio_result,
            "domain_motion.nesting.parent_time_step_ratio",
            "at least the largest horizontal parent ratio"));

        TempYaml two_way_feedback(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  nesting:
    enabled: true
    parent_ratio_x: 2
    parent_ratio_y: 2
    parent_time_step_ratio: 2
    child_cells_x: 8
    child_cells_y: 8
    relaxation_width_child_cells: 1
    feedback: two_way
)yaml");
        const LoadResult feedback_result =
            load_v1_config(two_way_feedback.path.string());
        REQUIRE(feedback_result.ok());
        REQUIRE(feedback_result.config->domain_motion.nesting.feedback ==
                NestingFeedback::TwoWay);
        REQUIRE_THROWS_WITH(
            require_runtime_supported_domain_motion(*feedback_result.config),
            Catch::Matchers::Contains("domain_motion.nesting.feedback=two_way is admitted only"));
    }

    SECTION("stationary one-way nesting selection is explicit and complete")
    {
        TempYaml valid(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  nesting:
    enabled: true
    parent_ratio_x: 3
    parent_ratio_y: 2
    parent_time_step_ratio: 3
    child_origin_parent_cell_i: 4
    child_origin_parent_cell_j: 5
    child_cells_x: 24
    child_cells_y: 16
    relaxation_width_child_cells: 2
    boundary_stencil_parent_cells: 1
    relaxation_e_folding_time_s: 3.5
    feedback: one_way
)yaml");
        const LoadResult result = load_v1_config(valid.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        const auto& nesting = result.config->domain_motion.nesting;
        REQUIRE(nesting.enabled);
        REQUIRE(nesting.child_origin_parent_cell_i == 4);
        REQUIRE(nesting.child_origin_parent_cell_j == 5);
        REQUIRE(nesting.child_cells_x == 24);
        REQUIRE(nesting.child_cells_y == 16);
        REQUIRE(nesting.relaxation_width_child_cells == 2);
        REQUIRE(nesting.boundary_stencil_parent_cells == 1);
        REQUIRE(nesting.relaxation_e_folding_time_s == Approx(3.5));

        TempYaml invalid(std::string(kMinimalValidV1) + R"yaml(
domain_motion:
  nesting:
    enabled: true
    parent_ratio_x: 3
    parent_ratio_y: 2
    parent_time_step_ratio: 3
    child_cells_x: 25
    child_cells_y: 4
    relaxation_width_child_cells: 2
)yaml");
        const LoadResult invalid_result = load_v1_config(invalid.path.string());
        REQUIRE_FALSE(invalid_result.ok());
        REQUIRE(has_error(invalid_result,
                          "domain_motion.nesting.child_cells_x",
                          "whole number of parent cells"));
        REQUIRE(has_error(invalid_result,
                          "domain_motion.nesting.relaxation_width_child_cells",
                          "non-empty child interior"));
    }
}

TEST_CASE("simple-grey radiation requires explicit physical solar inputs", "[init][config][radiation]")
{
    SECTION("missing solar contract is rejected")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + "\nradiation:\n  scheme: simple_grey\n");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "radiation.solar_forcing", "required"));
        REQUIRE(has_error(result, "radiation.latitude_deg", "required"));
        REQUIRE(has_error(result, "radiation.declination_deg", "required"));
        REQUIRE(has_error(result, "radiation.local_solar_time_at_start_h", "required"));
        REQUIRE(has_error(result, "surface.tsk_k", "required"));
    }

    SECTION("unphysical geometry and incomplete cloud optics are rejected")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
radiation:
  scheme: simple_grey
  solar_forcing: idealized_diurnal
  latitude_deg: 91
  declination_deg: 24
  local_solar_time_at_start_h: 24
  clear_sky_only: false
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "radiation.latitude_deg", "must be in"));
        REQUIRE(has_error(result, "radiation.declination_deg", "must be in"));
        REQUIRE(has_error(result, "radiation.local_solar_time_at_start_h", "[0, 24)"));
        REQUIRE(has_error(result, "radiation.simple_grey_optical_model",
                          "absorbing_water_cloud_path_v1"));
    }

    SECTION("absorbing water and cloud paths are explicit and unit-bearing")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
radiation:
  scheme: simple_grey
  solar_forcing: idealized_diurnal
  latitude_deg: 0
  declination_deg: 0
  local_solar_time_at_start_h: 12
  clear_sky_only: false
  simple_grey_optical_model: absorbing_water_cloud_path_v1
  use_subcycling: true
  subcycle_count: 2
  enable_diagnostics: true
  simple_grey_lw_vapor_absorption_m2kg: 0.02
  simple_grey_lw_liquid_absorption_m2kg: 85
  simple_grey_lw_ice_absorption_m2kg: 45
  simple_grey_sw_vapor_absorption_m2kg: 0.001
  simple_grey_sw_liquid_absorption_m2kg: 20
  simple_grey_sw_ice_absorption_m2kg: 12
surface:
  tsk_k: 290
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        const auto& radiation = result.config->radiation;
        CHECK(radiation.simple_grey_optical_model ==
              "absorbing_water_cloud_path_v1");
        CHECK(radiation.use_subcycling);
        CHECK(radiation.subcycle_count == 2);
        CHECK(radiation.enable_diagnostics);
        CHECK(radiation.simple_grey_lw_liquid_absorption_m2kg == Approx(85.0));
        CHECK(radiation.simple_grey_sw_ice_absorption_m2kg == Approx(12.0));
    }

    SECTION("simple-grey subcycles and absorption coefficients fail closed")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
radiation:
  scheme: simple_grey
  solar_forcing: idealized_diurnal
  latitude_deg: 0
  declination_deg: 0
  local_solar_time_at_start_h: 12
  clear_sky_only: false
  simple_grey_optical_model: absorbing_water_cloud_path_v1
  use_subcycling: true
  subcycle_count: 1
  simple_grey_lw_liquid_absorption_m2kg: -1
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        CHECK(has_error(result, "radiation.subcycle_count", "[2, 64]"));
        CHECK(has_error(result,
                        "radiation.simple_grey_lw_liquid_absorption_m2kg",
                        ">= 0"));
    }

    SECTION("simple-grey-only controls cannot leak into RRTMG")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
radiation:
  scheme: rrtmg
  simple_grey_optical_model: absorbing_water_cloud_path_v1
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        CHECK(has_error(result, "radiation.simple_grey_optical_model",
                        "inactive"));
    }

    SECTION("shortwave-only radiation does not require an unused skin temperature")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
radiation:
  scheme: simple_grey
  solar_forcing: idealized_diurnal
  latitude_deg: 35
  declination_deg: 15
  local_solar_time_at_start_h: 12
  do_lw: false
  do_sw: true
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->surface.tsk_k < 0.0);
    }

    SECTION("native RRTMG is accepted as an explicit exploratory radiation choice")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
radiation:
  scheme: rrtmg
  solar_forcing: idealized_diurnal
  latitude_deg: 35
  declination_deg: 15
  local_solar_time_at_start_h: 12
  do_lw: false
  do_sw: true
  clear_sky_only: false
  rrtmg_ozone_vmr: 4e-6
  rrtmg_co2_vmr: 421e-6
  rrtmg_ch4_vmr: 1.91e-6
  rrtmg_n2o_vmr: 0.336e-6
  rrtmg_co_vmr: 0.1e-6
  rrtmg_o2_vmr: 0.2094
  rrtmg_n2_vmr: 0.7807
  rrtmg_cfc11_vmr: 2e-10
  rrtmg_cfc12_vmr: 5e-10
  rrtmg_cfc22_vmr: 3e-10
  rrtmg_ccl4_vmr: 1e-10
  rrtmg_liquid_effective_radius_um: 11
  rrtmg_ice_effective_radius_um: 35
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->radiation.scheme == RadiationScheme::RRTMG);
        REQUIRE(std::string(radiation_scheme_name(result.config->radiation.scheme)) == "rrtmg");
        REQUIRE_FALSE(result.config->radiation.clear_sky_only);
        REQUIRE(result.config->radiation.rrtmg_ozone_vmr == Approx(4.0e-6));
        REQUIRE(result.config->radiation.rrtmg_co2_vmr == Approx(421.0e-6));
        REQUIRE(result.config->radiation.rrtmg_ch4_vmr == Approx(1.91e-6));
        REQUIRE(result.config->radiation.rrtmg_n2o_vmr == Approx(0.336e-6));
        REQUIRE(result.config->radiation.rrtmg_co_vmr == Approx(0.1e-6));
        REQUIRE(result.config->radiation.rrtmg_o2_vmr == Approx(0.2094));
        REQUIRE(result.config->radiation.rrtmg_n2_vmr == Approx(0.7807));
        REQUIRE(result.config->radiation.rrtmg_cfc11_vmr == Approx(2.0e-10));
        REQUIRE(result.config->radiation.rrtmg_cfc12_vmr == Approx(5.0e-10));
        REQUIRE(result.config->radiation.rrtmg_cfc22_vmr == Approx(3.0e-10));
        REQUIRE(result.config->radiation.rrtmg_ccl4_vmr == Approx(1.0e-10));
        REQUIRE(result.config->radiation.rrtmg_liquid_effective_radius_um == Approx(11.0));
        REQUIRE(result.config->radiation.rrtmg_ice_effective_radius_um == Approx(35.0));
    }

    SECTION("RRTMG requires an explicit ozone value instead of a hidden climatology")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
radiation:
  scheme: rrtmg
  solar_forcing: idealized_diurnal
  latitude_deg: 35
  declination_deg: 15
  local_solar_time_at_start_h: 12
  do_lw: false
  do_sw: true
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "radiation.rrtmg_ozone_vmr", "required"));
    }

    SECTION("RRTMG gas and cloud optics bounds fail closed")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
radiation:
  scheme: rrtmg
  solar_forcing: idealized_diurnal
  latitude_deg: 35
  declination_deg: 15
  local_solar_time_at_start_h: 12
  do_lw: false
  do_sw: true
  rrtmg_ozone_vmr: 0.02
  rrtmg_o2_vmr: 1.1
  rrtmg_liquid_effective_radius_um: 2
  rrtmg_ice_effective_radius_um: 131
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "radiation.rrtmg_ozone_vmr", "no larger"));
        REQUIRE(has_error(result, "radiation.rrtmg_o2_vmr", "no larger"));
        REQUIRE(has_error(result, "radiation.rrtmg_liquid_effective_radius_um", "must be in"));
        REQUIRE(has_error(result, "radiation.rrtmg_ice_effective_radius_um", "must be in"));
    }

    SECTION("RRTMG-only controls are rejected for other radiation schemes")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
radiation:
  scheme: simple_grey
  solar_forcing: idealized_diurnal
  latitude_deg: 35
  declination_deg: 15
  local_solar_time_at_start_h: 12
  do_lw: false
  do_sw: true
  rrtmg_ozone_vmr: 4e-6
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "radiation.rrtmg_ozone_vmr", "inactive"));
    }
}

TEST_CASE("load_v1_config: diagnostics and performance controls are typed",
          "[init][config][v1_loader][extensions][performance]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
extensions:
  compute:
    memory_admission: reject
  diagnostics:
    scientific_every_steps: 0
    deterministic_reductions: true
  performance:
    timing: true
    report_every_steps: 25
    warmup_steps: 10
)yaml");

    const auto result = load_v1_config(tmp.path.string());
    INFO(format_errors(result.errors));
    REQUIRE(result.ok());
    REQUIRE(result.config->extensions.diagnostics.scientific_every_steps == 0);
    REQUIRE(result.config->extensions.compute.memory_admission == MemoryAdmissionPolicy::Reject);
    REQUIRE(result.config->extensions.diagnostics.deterministic_reductions);
    REQUIRE(result.config->extensions.performance.timing == true);
    REQUIRE(result.config->extensions.performance.report_every_steps == 25);
    REQUIRE(result.config->extensions.performance.warmup_steps == 10);
}

TEST_CASE("load_v1_config: negative runtime cadences fail closed", "[init][config][v1_loader][extensions][performance]")
{
    SECTION("scientific diagnostics")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
extensions:
  diagnostics:
    scientific_every_steps: -1
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE_FALSE(result.errors.empty());
    }

    SECTION("performance warmup")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
extensions:
  performance:
    warmup_steps: -2
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE_FALSE(result.errors.empty());
    }

    SECTION("memory admission policy")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
extensions:
  compute:
    memory_admission: maybe
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE_FALSE(result.errors.empty());
    }
}

TEST_CASE("load_v1_config: 2d turbulence is a supported horizontal-only mode",
          "[init][config][v1_loader][turbulence][sci021]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
turbulence:
  scheme: smagorinsky
  mode: 2d
)yaml");

    const auto result = load_v1_config(tmp.path.string());
    INFO(format_errors(result.errors));
    REQUIRE(result.ok());
    REQUIRE(result.config->turbulence.mode == TurbulenceMode::TwoD);
}

TEST_CASE("load_v1_config: vertical-only turbulence is a selectable mode",
          "[init][config][v1_loader][turbulence][sci235]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
turbulence:
  scheme: smagorinsky
  mode: vertical_only
)yaml");

    const auto result = load_v1_config(tmp.path.string());
    INFO(format_errors(result.errors));
    REQUIRE(result.ok());
    REQUIRE(result.config->turbulence.mode == TurbulenceMode::VerticalOnly);
}

TEST_CASE("load_v1_config: unsupported diffusion selections fail at schema load",
          "[init][config][v1_loader][diffusion][sci021]")
{
    SECTION("none has no registered runtime scheme")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
diffusion:
  scheme: none
  operator_order: 2
)yaml");

        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE_FALSE(result.config.has_value());
        REQUIRE(std::any_of(result.errors.begin(),
                            result.errors.end(),
                            [](const LoadError& error)
                            {
                                return error.key == "diffusion.scheme" &&
                                       error.message.find("no registered runtime") != std::string::npos;
                            }));
    }

    SECTION("sixth order is not implemented")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
diffusion:
  scheme: explicit
  operator_order: 6
)yaml");

        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE_FALSE(result.config.has_value());
        REQUIRE(std::any_of(result.errors.begin(),
                            result.errors.end(),
                            [](const LoadError& error)
                            {
                                return error.key == "diffusion.operator_order" &&
                                       error.message.find("runtime order 2") != std::string::npos;
                            }));
    }

    SECTION("explicit second order remains accepted")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
diffusion:
  scheme: explicit
  operator_order: 2
)yaml");

        const auto result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->diffusion.scheme == DiffusionScheme::Explicit);
        REQUIRE(result.config->diffusion.operator_order == 2);
    }

    SECTION("implicit second order reaches the registered runtime path")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
diffusion:
  scheme: implicit
  operator_order: 2
)yaml");

        const auto result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->diffusion.scheme == DiffusionScheme::Implicit);
        REQUIRE(result.config->diffusion.operator_order == 2);
    }
}

TEST_CASE("load_v1_config: matched periodic lateral pairs are accepted",
          "[init][config][v1_loader][boundary_conditions]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
boundary_conditions:
  lateral:
    west: periodic
    east: periodic
    south: periodic
    north: periodic
)yaml");

    auto result = load_v1_config(tmp.path.string());

    INFO(format_errors(result.errors));
    REQUIRE(result.ok());
    REQUIRE(result.config->boundary_conditions.lateral.west == LateralBC::Periodic);
    REQUIRE(result.config->boundary_conditions.lateral.east == LateralBC::Periodic);
    REQUIRE(result.config->boundary_conditions.lateral.south == LateralBC::Periodic);
    REQUIRE(result.config->boundary_conditions.lateral.north == LateralBC::Periodic);
}

TEST_CASE("load_v1_config: lateral open algorithms are explicit after loading",
          "[init][config][v1_loader][boundary_conditions][SCI-017]")
{
    SECTION("C-grid accepts both explicit closures")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
boundary_conditions:
  lateral:
    west: radiative
    east: zero_gradient
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->boundary_conditions.lateral.west == LateralBC::Radiative);
        REQUIRE(result.config->boundary_conditions.lateral.east == LateralBC::ZeroGradient);
    }

    SECTION("legacy open projects from declared staggering")
    {
        std::string yaml     = kMinimalValidV1;
        const auto  grid_pos = yaml.find("grid:\n");
        REQUIRE(grid_pos != std::string::npos);
        yaml.insert(grid_pos + std::string("grid:\n").size(), "  staggering: collocated\n");
        yaml += R"yaml(
boundary_conditions:
  lateral:
    west: open
)yaml";
        TempYaml   tmp(yaml);
        const auto result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->boundary_conditions.lateral.west == LateralBC::ZeroGradient);
        REQUIRE(result.config->boundary_conditions.lateral.east == LateralBC::ZeroGradient);
    }

    SECTION("collocated radiative fails closed")
    {
        std::string yaml     = kMinimalValidV1;
        const auto  grid_pos = yaml.find("grid:\n");
        REQUIRE(grid_pos != std::string::npos);
        yaml.insert(grid_pos + std::string("grid:\n").size(), "  staggering: collocated\n");
        yaml += R"yaml(
boundary_conditions:
  lateral:
    west: radiative
)yaml";
        TempYaml   tmp(yaml);
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "boundary_conditions.lateral.west", "require grid.staggering=c_grid"));
    }
}

TEST_CASE("load_v1_config: one-sided periodic lateral boundaries are rejected",
          "[init][config][v1_loader][boundary_conditions]")
{
    const char* lateral_yaml = nullptr;
    const char* periodic_key = nullptr;
    const char* opposite_key = nullptr;

    SECTION("west periodic without east")
    {
        lateral_yaml = R"yaml(
boundary_conditions:
  lateral:
    west: periodic
)yaml";
        periodic_key = "boundary_conditions.lateral.west";
        opposite_key = "boundary_conditions.lateral.east";
    }
    SECTION("east periodic without west")
    {
        lateral_yaml = R"yaml(
boundary_conditions:
  lateral:
    west: rigid
    east: periodic
)yaml";
        periodic_key = "boundary_conditions.lateral.east";
        opposite_key = "boundary_conditions.lateral.west";
    }
    SECTION("south periodic without north")
    {
        lateral_yaml = R"yaml(
boundary_conditions:
  lateral:
    south: periodic
    north: rigid_slip
)yaml";
        periodic_key = "boundary_conditions.lateral.south";
        opposite_key = "boundary_conditions.lateral.north";
    }
    SECTION("north periodic without south")
    {
        lateral_yaml = R"yaml(
boundary_conditions:
  lateral:
    south: open
    north: periodic
)yaml";
        periodic_key = "boundary_conditions.lateral.north";
        opposite_key = "boundary_conditions.lateral.south";
    }

    TempYaml tmp(std::string(kMinimalValidV1) + lateral_yaml);
    auto     result = load_v1_config(tmp.path.string());

    INFO(format_errors(result.errors));
    REQUIRE_FALSE(result.ok());
    REQUIRE_FALSE(result.config.has_value());

    bool found_pair_error = false;
    for (const auto& e : result.errors)
    {
        if (e.key == periodic_key && e.message.find("opposing pairs") != std::string::npos)
        {
            REQUIRE(e.line > 0);
            REQUIRE(e.message.find(opposite_key) != std::string::npos);
            found_pair_error = true;
        }
    }
    REQUIRE(found_pair_error);
}

TEST_CASE("load_v1_config: slab is an explicit boundary-layer scheme", "[init][config][v1_loader][pbl]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
surface:
  scheme: slab
  apply_fluxes: true
  z0m_m: 0.15
  z0h_m: 0.015
  albedo: 0.17
  emissivity: 0.93
  tsk_k: 301
  qsfc_kgkg: 0.014
boundary_layer:
  scheme: slab
)yaml");

    auto result = load_v1_config(tmp.path.string());
    INFO(format_errors(result.errors));
    REQUIRE(result.ok());
    REQUIRE(result.config->surface.scheme == SurfaceScheme::Slab);
    REQUIRE(result.config->boundary_layer.scheme == BoundaryLayerScheme::Slab);
    REQUIRE(std::string(boundary_layer_scheme_name(result.config->boundary_layer.scheme)) == "slab");
}

// ─────────────────────────────────────────────────────────────
// modularity cleanup 6a: behavioral knobs promoted from NIMBUS_* env vars to the
// schema. Defaults must reproduce the prior env-unset runtime; explicit values
// must parse into damping / extensions.{compute,solver}.
// ─────────────────────────────────────────────────────────────

TEST_CASE("load_v1_config: 6a promoted knobs default to the prior env-unset runtime",
          "[init][config][v1_loader][extensions]")
{
    TempYaml tmp(kMinimalValidV1);
    auto     result = load_v1_config(tmp.path.string());
    INFO(format_errors(result.errors));
    REQUIRE(result.ok());
    const auto& cfg = *result.config;

    REQUIRE(cfg.damping.lateral_sponge_enabled == true);
    REQUIRE(cfg.extensions.compute.allow_fallback == true);
    REQUIRE(cfg.extensions.compute.validate_parity == false);
    REQUIRE(cfg.extensions.compute.gpu_apply_microphysics == true);
    REQUIRE(cfg.extensions.compute.gpu_apply_slow == false);
    REQUIRE(cfg.extensions.compute.gpu_apply_sgs == false);
    REQUIRE(cfg.extensions.solver.slow_rk2 == true);
    REQUIRE(cfg.extensions.solver.lid_bc_gamma == Approx(-1.0));
    REQUIRE(cfg.extensions.solver.axis_u_m1_filter == false);
}

TEST_CASE("load_v1_config: 6a promoted knobs parse explicit values", "[init][config][v1_loader][extensions]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
damping:
  lateral_sponge_enabled: false
extensions:
  compute:
    allow_fallback: false
    validate_parity: true
    gpu_apply_microphysics: false
    gpu_apply_slow: true
    gpu_apply_sgs: true
  solver:
    slow_rk2: false
    lid_bc_gamma: 0.6
    axis_u_m1_filter: true
)yaml");
    auto     result = load_v1_config(tmp.path.string());
    INFO(format_errors(result.errors));
    REQUIRE(result.ok());
    const auto& cfg = *result.config;

    REQUIRE(cfg.damping.lateral_sponge_enabled == false);
    REQUIRE(cfg.extensions.compute.allow_fallback == false);
    REQUIRE(cfg.extensions.compute.validate_parity == true);
    REQUIRE(cfg.extensions.compute.gpu_apply_microphysics == false);
    REQUIRE(cfg.extensions.compute.gpu_apply_slow == true);
    REQUIRE(cfg.extensions.compute.gpu_apply_sgs == true);
    REQUIRE(cfg.extensions.solver.slow_rk2 == false);
    REQUIRE(cfg.extensions.solver.lid_bc_gamma == Approx(0.6));
    REQUIRE(cfg.extensions.solver.axis_u_m1_filter == true);
}

TEST_CASE("load_v1_config: compute fallback policy rejects invalid booleans",
          "[init][config][v1_loader][extensions][compute]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
extensions:
  compute:
    allow_fallback: sometimes
)yaml");
    const auto result = load_v1_config(tmp.path.string());
    REQUIRE_FALSE(result.ok());
    REQUIRE(has_error(result, "extensions.compute.allow_fallback", "as boolean"));
}

TEST_CASE("strict v1 rejects recognized enum selections without runtime semantics",
          "[init][config][v1_loader][sci021][completeness]")
{
    SECTION("implicit damping")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
damping:
  mode: implicit
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE(result.ok());
        REQUIRE(result.config->damping.mode == DampingMode::Implicit);
    }

    SECTION("beta-plane Coriolis")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
coriolis:
  mode: beta_plane
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE(result.ok());
        REQUIRE(result.config->coriolis.mode == CoriolisMode::BetaPlane);
    }

    SECTION("NetCDF output")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
output:
  format: netcdf
  netcdf_deflate_level: 4
  netcdf_shuffle: false
  netcdf_target_chunk_kib: 256
)yaml");
        const auto result = load_v1_config(tmp.path.string());
#ifdef HAVE_NETCDF
        REQUIRE(result.ok());
        REQUIRE(result.config.has_value());
        REQUIRE(result.config->output.format == OutputFormat::Netcdf);
        REQUIRE(result.config->output.netcdf_deflate_level == 4);
        REQUIRE_FALSE(result.config->output.netcdf_shuffle);
        REQUIRE(result.config->output.netcdf_target_chunk_kib == 256);
#else
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "output.format", "unavailable"));
#endif
    }
}

// ─────────────────────────────────────────────────────────────
// Microphysics scheme naming: Milbrandt and Morrison are distinct schemes.
// ─────────────────────────────────────────────────────────────

TEST_CASE("load_v1_config: Milbrandt is canonical and Morrison is not an alias",
          "[init][config][v1_loader][microphysics]")
{
    SECTION("canonical token: milbrandt")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
microphysics:
  scheme: milbrandt
)yaml");
        auto     result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->microphysics.scheme == MicrophysicsScheme::Milbrandt);
    }

    SECTION("scientifically distinct token: morrison")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
microphysics:
  scheme: morrison
)yaml");
        auto     result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "microphysics.scheme", "unknown value"));
    }
}

TEST_CASE("load_v1_config: active microphysics parameters are typed and inactive blocks fail",
          "[init][config][v1_loader][microphysics][parameters]")
{
    SECTION("Lin parameters load")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
microphysics:
  scheme: lin
  shared:
    cloud_water_threshold_kgkg: 0.002
    autoconversion_rate_s_inv: 0.004
  lin:
    accretion_coefficient: 1.7
    evaporation_rate_s_inv: 0.0008
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        const auto& parameters = result.config->microphysics.parameters;
        REQUIRE(parameters.shared.cloud_water_threshold_kgkg == Approx(0.002));
        REQUIRE(parameters.shared.autoconversion_rate_s_inv == Approx(0.004));
        REQUIRE(parameters.lin.accretion_coefficient == Approx(1.7));
        REQUIRE(parameters.lin.evaporation_rate_s_inv == Approx(0.0008));
    }

    SECTION("inactive Kessler block is rejected for Lin")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
microphysics:
  scheme: lin
  kessler:
    accretion_coefficient: 1.7
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(
            has_error(result, "microphysics.kessler.accretion_coefficient", "inactive for microphysics.scheme=lin"));
    }

    SECTION("Thompson evaporation parameter loads at its lower boundary")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
microphysics:
  scheme: thompson
  thompson:
    evaporation_rate_s_inv: 0
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->microphysics.parameters.thompson.evaporation_rate_s_inv == Approx(0.0));
    }

    SECTION("Thompson aerosol profile endpoints load as finite positive values")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
microphysics:
  scheme: thompson
  thompson:
    water_friendly_aerosol_surface_m3: 325000000
    ice_friendly_aerosol_surface_m3: 1750000
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->microphysics.parameters.thompson
                    .water_friendly_aerosol_surface_m3 == Approx(325.0e6));
        REQUIRE(result.config->microphysics.parameters.thompson
                    .ice_friendly_aerosol_surface_m3 == Approx(1.75e6));
    }

    SECTION("Thompson aerosol profile endpoints reject zero and nonfinite values")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
microphysics:
  scheme: thompson
  thompson:
    water_friendly_aerosol_surface_m3: 0
    ice_friendly_aerosol_surface_m3: .nan
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(
            result,
            "microphysics.thompson.water_friendly_aerosol_surface_m3",
            "must be > 0"));
        REQUIRE(has_error(
            result,
            "microphysics.thompson.ice_friendly_aerosol_surface_m3",
            "cannot parse"));
    }

    SECTION("Thompson surface aerosol totals must exceed the fixed backgrounds")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
microphysics:
  scheme: thompson
  thompson:
    water_friendly_aerosol_surface_m3: 50000000
    ice_friendly_aerosol_surface_m3: 500000
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(
            result,
            "microphysics.thompson.water_friendly_aerosol_surface_m3",
            "must exceed"));
        REQUIRE(has_error(
            result,
            "microphysics.thompson.ice_friendly_aerosol_surface_m3",
            "must exceed"));
    }

    SECTION("obsolete fixed-rate Milbrandt evaporation is rejected")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
microphysics:
  scheme: milbrandt
  milbrandt:
    evaporation_rate_s_inv: 0.0006
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "microphysics.milbrandt.evaporation_rate_s_inv", "inactive"));
    }

    SECTION("Thompson parameter is rejected for Milbrandt")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
microphysics:
  scheme: milbrandt
  thompson:
    evaporation_rate_s_inv: 0.001
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "microphysics.thompson.evaporation_rate_s_inv", "inactive"));
    }

    SECTION("Thompson aerosol controls are rejected for non-Thompson schemes")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
microphysics:
  scheme: milbrandt
  thompson:
    water_friendly_aerosol_surface_m3: 300000000
    ice_friendly_aerosol_surface_m3: 1500000
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(
            result,
            "microphysics.thompson.water_friendly_aerosol_surface_m3",
            "inactive"));
        REQUIRE(has_error(
            result,
            "microphysics.thompson.ice_friendly_aerosol_surface_m3",
            "inactive"));
    }

    SECTION("negative Thompson evaporation is rejected")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
microphysics:
  scheme: thompson
  thompson:
    evaporation_rate_s_inv: -0.0001
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "microphysics.thompson.evaporation_rate_s_inv", "must be >= 0"));
    }

    SECTION("zero is the accepted numeric lower boundary")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
microphysics:
  scheme: kessler
  shared:
    cloud_water_threshold_kgkg: 0
    autoconversion_rate_s_inv: 0
  kessler:
    accretion_coefficient: 0
    evaporation_efficiency: 0
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        const auto& parameters = result.config->microphysics.parameters;
        REQUIRE(parameters.shared.cloud_water_threshold_kgkg == Approx(0.0));
        REQUIRE(parameters.shared.autoconversion_rate_s_inv == Approx(0.0));
        REQUIRE(parameters.kessler.accretion_coefficient == Approx(0.0));
        REQUIRE(parameters.kessler.evaporation_efficiency == Approx(0.0));
    }

    SECTION("obsolete Kessler first-order evaporation rate is rejected")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
microphysics:
  scheme: kessler
  kessler:
    evaporation_rate_s_inv: 0.003
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "microphysics.kessler.evaporation_rate_s_inv", "unrecognized v1 key"));
    }

    SECTION("dry dynamics rejects wet-scheme shared parameters")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
microphysics:
  scheme: none
  shared:
    autoconversion_rate_s_inv: 0.001
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(
            result, "microphysics.shared.autoconversion_rate_s_inv", "inactive for microphysics.scheme=none"));
    }
}

// ─────────────────────────────────────────────────────────────
// Surface skin state (CM1 tsk0): optional surface.tsk_k / surface.qsfc_kgkg
// parse into the v1 surface section; omitting them leaves the < 0 'unset'
// sentinel so the runtime holds the initial surface-air state as the skin.
// ─────────────────────────────────────────────────────────────

TEST_CASE("load_v1_config: surface skin tsk_k/qsfc_kgkg parse; omitted -> unset sentinel",
          "[init][config][v1_loader][surface]")
{
    SECTION("prescribed skin state parses")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
surface:
  scheme: slab
  apply_fluxes: true
  tsk_k: 305.0
  qsfc_kgkg: 0.018
)yaml");
        auto     result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->surface.tsk_k == Approx(305.0));
        REQUIRE(result.config->surface.qsfc_kgkg == Approx(0.018));
    }

    SECTION("omitted skin state keeps the < 0 'unset' sentinel")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
surface:
  scheme: slab
  apply_fluxes: true
)yaml");
        auto     result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->surface.tsk_k < 0.0);
        REQUIRE(result.config->surface.qsfc_kgkg < 0.0);
    }

    SECTION("radiative and exchange surface properties are range checked")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
surface:
  scheme: slab
  apply_fluxes: true
  z0m_m: 0.1
  z0h_m: 0.01
  albedo: 1.1
  emissivity: -0.1
  tsk_k: 500
  qsfc_kgkg: 0.2
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "surface.albedo", "must be in"));
        REQUIRE(has_error(result, "surface.emissivity", "must be in"));
        REQUIRE(has_error(result, "surface.tsk_k", "must be in"));
        REQUIRE(has_error(result, "surface.qsfc_kgkg", "must be in"));
    }
}

// ─────────────────────────────────────────────────────────────
// ZFP output tuning (Nimbus extension) — formerly fatal unknown keys
// ─────────────────────────────────────────────────────────────

TEST_CASE("load_v1_config: ZFP format + tuning keys load cleanly", "[init][config][v1_loader]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
output:
  format: zfp_high
  zfp_tolerance: 1.0e-3
  zfp_mode: precision
  zfp_precision_bits: 22
  zfp_keyframe_interval: 8
  zfp_float16_prefilter: true
  zfp_per_field_tolerances: true
  zfp_field_tolerance:
    rho: 1.0e-6
    qv: 1.0e-8
  tiered_write_cadence: true
  write_cadence_medium_s: 45
)yaml");

    auto result = load_v1_config(tmp.path.string());

    // Headline: the ZFP tuning keys are consumed, not reported as
    // unrecognized v1 keys (§19.6 no-silent-runs).
#ifndef HAVE_ZFP
    REQUIRE_FALSE(result.ok());
    REQUIRE(has_error(result, "output.format", "unavailable"));
    return;
#endif
    REQUIRE(result.ok());
    REQUIRE(result.errors.empty());
    REQUIRE(result.config.has_value());

    const auto& o = result.config->output;
    REQUIRE(o.format == OutputFormat::ZfpHigh);
    REQUIRE(o.zfp_tolerance == Approx(1.0e-3));
    REQUIRE(o.zfp_mode == "precision");
    REQUIRE(o.zfp_precision_bits == 22);
    REQUIRE(o.zfp_keyframe_interval == 8);
    REQUIRE(o.zfp_float16_prefilter == true);
    REQUIRE(o.zfp_per_field_tolerances == true);
    REQUIRE(o.zfp_field_tolerances.at("rho") == Approx(1.0e-6));
    REQUIRE(o.zfp_field_tolerances.at("qv") == Approx(1.0e-8));
    REQUIRE(o.tiered_write_cadence == true);
    REQUIRE(o.write_cadence_medium_s == 45);
}

TEST_CASE("load_v1_config: invalid ZFP mode and precision fail closed", "[init][config][v1_loader][zfp]")
{
    SECTION("unknown mode")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
output:
  format: zfp_high
  zfp_mode: magic
)yaml");
        auto     result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(format_errors(result.errors).find("output.zfp_mode") != std::string::npos);
    }
    SECTION("precision exceeds float32")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
output:
  format: zfp_high
  zfp_mode: precision
  zfp_precision_bits: 33
)yaml");
        auto     result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(format_errors(result.errors).find("output.zfp_precision_bits") != std::string::npos);
    }
}

// ─────────────────────────────────────────────────────────────
// Schema stamp errors
// ─────────────────────────────────────────────────────────────

TEST_CASE("load_v1_config: missing schema stamp is rejected", "[init][config][v1_loader]")
{
    TempYaml tmp(R"yaml(
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 250
  dy: 250
  dz: 250
)yaml");

    auto result = load_v1_config(tmp.path.string());

    REQUIRE_FALSE(result.ok());
    REQUIRE_FALSE(result.config.has_value());
    REQUIRE(result.errors.size() == 1);
    REQUIRE(result.errors[0].key == "schema");
    REQUIRE(result.errors[0].message.find("missing required v1 schema stamp") != std::string::npos);
}

TEST_CASE("load_v1_config: wrong schema version is rejected with line number", "[init][config][v1_loader]")
{
    TempYaml tmp(R"yaml(schema: nimbus.v0
grid:
  nx: 1
)yaml");

    auto result = load_v1_config(tmp.path.string());

    REQUIRE_FALSE(result.ok());
    REQUIRE(result.errors.size() == 1);
    REQUIRE(result.errors[0].key == "schema");
    REQUIRE(result.errors[0].line == 1);
    REQUIRE(result.errors[0].message.find("unsupported schema version") != std::string::npos);
    REQUIRE(result.errors[0].message.find("nimbus.v0") != std::string::npos);
}

TEST_CASE("load_v1_config: missing file produces clean error", "[init][config][v1_loader]")
{
    auto result = load_v1_config("/tmp/nonexistent_nimbus_test_path_xyz123.yaml");

    REQUIRE_FALSE(result.ok());
    REQUIRE(result.errors.size() == 1);
    REQUIRE(result.errors[0].message.find("could not open config file") != std::string::npos);
}

// ─────────────────────────────────────────────────────────────
// Required-key validation
// ─────────────────────────────────────────────────────────────

TEST_CASE("load_v1_config: missing required key is reported", "[init][config][v1_loader]")
{
    // grid.nx omitted from an otherwise-valid config
    TempYaml tmp(R"yaml(
schema: nimbus.v1
grid:
  ny: 64
  nz: 32
  dx: 250
  dy: 250
  dz: 250
time:
  run_time: 100
  dt_large: 1
  output_interval: 10
)yaml");

    auto result = load_v1_config(tmp.path.string());

    REQUIRE_FALSE(result.ok());
    REQUIRE_FALSE(result.errors.empty());

    bool found_grid_nx_missing = false;
    for (const auto& e : result.errors)
    {
        if (e.key == "grid.nx" && e.message.find("required key missing") != std::string::npos)
        {
            found_grid_nx_missing = true;
            break;
        }
    }
    REQUIRE(found_grid_nx_missing);
}

TEST_CASE("load_v1_config: multiple errors collected in one pass", "[init][config][v1_loader]")
{
    // Both grid.nx and time.run_time missing — loader should report both,
    // not bail on the first.
    TempYaml tmp(R"yaml(
schema: nimbus.v1
grid:
  ny: 64
  nz: 32
  dx: 250
  dy: 250
  dz: 250
time:
  dt_large: 1
  output_interval: 10
)yaml");

    auto result = load_v1_config(tmp.path.string());

    REQUIRE_FALSE(result.ok());
    REQUIRE(result.errors.size() >= 2);

    bool nx_missing       = false;
    bool run_time_missing = false;
    for (const auto& e : result.errors)
    {
        if (e.key == "grid.nx")
            nx_missing = true;
        if (e.key == "time.run_time")
            run_time_missing = true;
    }
    REQUIRE(nx_missing);
    REQUIRE(run_time_missing);
}

// ─────────────────────────────────────────────────────────────
// Value validation: enums, types, ranges
// ─────────────────────────────────────────────────────────────

TEST_CASE("load_v1_config: bad enum value reports line and allowed set", "[init][config][v1_loader]")
{
    TempYaml tmp(R"yaml(schema: nimbus.v1
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 250
  dy: 250
  dz: 250
time:
  run_time: 100
  dt_large: 1
  output_interval: 10
microphysics:
  scheme: kessler_typo
)yaml");

    auto result = load_v1_config(tmp.path.string());

    REQUIRE_FALSE(result.ok());

    bool found = false;
    for (const auto& e : result.errors)
    {
        if (e.key == "microphysics.scheme")
        {
            // Line points at the offending value
            REQUIRE(e.line > 0);
            REQUIRE(e.message.find("kessler_typo") != std::string::npos);
            REQUIRE(e.message.find("expected one of") != std::string::npos);
            REQUIRE(e.message.find("kessler") != std::string::npos);
            found = true;
            break;
        }
    }
    REQUIRE(found);
}

TEST_CASE("load_v1_config: non-numeric value rejected with line number", "[init][config][v1_loader]")
{
    TempYaml tmp(R"yaml(schema: nimbus.v1
grid:
  nx: sixty-four
  ny: 64
  nz: 32
  dx: 250
  dy: 250
  dz: 250
time:
  run_time: 100
  dt_large: 1
  output_interval: 10
)yaml");

    auto result = load_v1_config(tmp.path.string());

    REQUIRE_FALSE(result.ok());

    bool found = false;
    for (const auto& e : result.errors)
    {
        if (e.key == "grid.nx")
        {
            REQUIRE(e.line == 3); // grid.nx is on line 3 of the YAML
            REQUIRE(e.message.find("cannot parse") != std::string::npos);
            REQUIRE(e.message.find("sixty-four") != std::string::npos);
            found = true;
            break;
        }
    }
    REQUIRE(found);
}

TEST_CASE("load_v1_config: negative grid size rejected", "[init][config][v1_loader]")
{
    TempYaml tmp(R"yaml(schema: nimbus.v1
grid:
  nx: -1
  ny: 64
  nz: 32
  dx: 250
  dy: 250
  dz: 250
time:
  run_time: 100
  dt_large: 1
  output_interval: 10
)yaml");

    auto result = load_v1_config(tmp.path.string());

    REQUIRE_FALSE(result.ok());

    bool found = false;
    for (const auto& e : result.errors)
    {
        if (e.key == "grid.nx" && e.message.find("must be > 0") != std::string::npos)
        {
            REQUIRE(e.line == 3);
            found = true;
            break;
        }
    }
    REQUIRE(found);
}

TEST_CASE("load_v1_config: runtime seconds reject fractional values", "[init][config][v1_loader][runtime]")
{
    TempYaml tmp(R"yaml(schema: nimbus.v1
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 250
  dy: 250
  dz: 250
time:
  run_time: 100.5
  dt_large: 1
  output_interval: 10.25
)yaml");

    const auto result = load_v1_config(tmp.path.string());
    REQUIRE_FALSE(result.ok());

    bool duration_rejected = false;
    bool output_rejected   = false;
    for (const auto& error : result.errors)
    {
        if (error.key == "time.run_time")
        {
            duration_rejected = error.message.find("whole number of seconds") != std::string::npos;
        }
        if (error.key == "time.output_interval")
        {
            output_rejected = error.message.find("whole number of seconds") != std::string::npos;
        }
    }
    REQUIRE(duration_rejected);
    REQUIRE(output_rejected);
}

TEST_CASE("load_v1_config: coriolis latitude out of range rejected", "[init][config][v1_loader]")
{
    TempYaml tmp(R"yaml(schema: nimbus.v1
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 250
  dy: 250
  dz: 250
time:
  run_time: 100
  dt_large: 1
  output_interval: 10
coriolis:
  latitude_deg: 120
)yaml");

    auto result = load_v1_config(tmp.path.string());

    REQUIRE_FALSE(result.ok());

    bool found = false;
    for (const auto& e : result.errors)
    {
        if (e.key == "coriolis.latitude_deg")
        {
            REQUIRE(e.message.find("must be in [-90, 90]") != std::string::npos);
            found = true;
            break;
        }
    }
    REQUIRE(found);
}

// ─────────────────────────────────────────────────────────────
// Unknown-key strictness (§19.6 no silent runs)
// ─────────────────────────────────────────────────────────────

TEST_CASE("load_v1_config: unknown top-level key is hard error", "[init][config][v1_loader]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
microhpysics:
  scheme: kessler
)yaml");

    auto result = load_v1_config(tmp.path.string());

    REQUIRE_FALSE(result.ok());

    bool found_typo = false;
    for (const auto& e : result.errors)
    {
        if (e.key == "microhpysics.scheme" && e.message.find("unrecognized v1 key") != std::string::npos)
        {
            REQUIRE(e.line > 0);
            found_typo = true;
            break;
        }
    }
    REQUIRE(found_typo);
}

// ─────────────────────────────────────────────────────────────
// Sounding type=file requires file_path
// ─────────────────────────────────────────────────────────────

TEST_CASE("load_v1_config: sounding type=file without file_path is rejected", "[init][config][v1_loader]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
initial_conditions:
  sounding:
    type: file
)yaml");

    auto result = load_v1_config(tmp.path.string());

    REQUIRE_FALSE(result.ok());

    bool found = false;
    for (const auto& e : result.errors)
    {
        if (e.key == "initial_conditions.sounding.file_path")
        {
            found = true;
            break;
        }
    }
    REQUIRE(found);
}

TEST_CASE("file sounding defaults to automatic use of supplied winds", "[init][config][v1_loader][sounding][hodograph]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
initial_conditions:
  sounding:
    type: file
    file_path: profile.txt
)yaml");

    const auto result = load_v1_config(tmp.path.string());
    INFO(format_errors(result.errors));
    REQUIRE(result.ok());
    REQUIRE(result.config->initial_conditions.hodograph.type == HodographType::Auto);
    REQUIRE(std::string(hodograph_type_name(result.config->initial_conditions.hodograph.type)) == "auto");
}

TEST_CASE("SCI-026: strict v1 reaches the parametric-targets source", "[init][config][v1_loader][sounding][sci026]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
initial_conditions:
  sounding:
    type: parametric_targets
    cape_jkg: 3200
    cin_jkg: 75
    lcl_m: 1050
    lfc_m: 1700
    el_m: 12500
    surface_theta_k: 302
)yaml");

    const auto result = load_v1_config(tmp.path.string());
    INFO(format_errors(result.errors));
    REQUIRE(result.ok());
    const auto& sounding = result.config->initial_conditions.sounding;
    REQUIRE(sounding.type == SoundingType::ParametricTargets);
    REQUIRE(std::string(sounding_type_name(sounding.type)) == "parametric_targets");
    REQUIRE(sounding.cape_jkg == Approx(3200.0));
    REQUIRE(sounding.cin_jkg == Approx(75.0));
    REQUIRE(sounding.lcl_m == Approx(1050.0));
    REQUIRE(sounding.lfc_m == Approx(1700.0));
    REQUIRE(sounding.el_m == Approx(12500.0));
    REQUIRE_FALSE(sounding.moisture_qv_kgkg.has_value());
}

TEST_CASE("SCI-026: parametric-cape accepts only effective source keys", "[init][config][v1_loader][sounding][sci026]")
{
    SECTION("truthful tropopause and active qv load")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
initial_conditions:
  sounding:
    type: parametric_cape
    cape_jkg: 2500
    tropopause_m: 11500
    surface_theta_k: 300
    moisture_qv_kgkg: 0.004
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->initial_conditions.sounding.tropopause_m == Approx(11500.0));
        REQUIRE(result.config->initial_conditions.sounding.moisture_qv_kgkg.has_value());
        REQUIRE(*result.config->initial_conditions.sounding.moisture_qv_kgkg == Approx(0.004));
    }

    SECTION("diagnostic targets and file path are rejected")
    {
        const std::vector<std::pair<std::string, std::string>> rejected = {
            {"cin_jkg", "25"},
            {"lcl_m", "1000"},
            {"lfc_m", "1500"},
            {"el_m", "12000"},
            {"file_path", "profile.txt"},
        };
        for (const auto& [leaf, value] : rejected)
        {
            TempYaml   tmp(std::string(kMinimalValidV1) +
                           "initial_conditions:\n  sounding:\n"
                           "    type: parametric_cape\n    " +
                           leaf + ": " + value + "\n");
            const auto result = load_v1_config(tmp.path.string());
            INFO("key: " << leaf);
            INFO(format_errors(result.errors));
            REQUIRE_FALSE(result.ok());
            REQUIRE(has_error(
                result, "initial_conditions.sounding." + leaf, "not valid for sounding.type=parametric_cape"));
        }
    }

    SECTION("values that the builder would clamp are rejected")
    {
        const std::vector<std::pair<std::string, std::string>> rejected = {
            {"surface_theta_k", "249.9"},
            {"tropopause_m", "7999"},
            {"moisture_qv_kgkg", "0.0039"},
            {"moisture_qv_kgkg", "0.0241"},
        };
        for (const auto& [leaf, value] : rejected)
        {
            TempYaml   tmp(std::string(kMinimalValidV1) +
                           "initial_conditions:\n  sounding:\n"
                           "    type: parametric_cape\n    " +
                           leaf + ": " + value + "\n");
            const auto result = load_v1_config(tmp.path.string());
            INFO("key: " << leaf << " value: " << value);
            INFO(format_errors(result.errors));
            REQUIRE_FALSE(result.ok());
            REQUIRE(has_error(result, "initial_conditions.sounding." + leaf, "otherwise clamps"));
        }
    }
}

TEST_CASE("SCI-026: parametric-target fields are effective or fail closed",
          "[init][config][v1_loader][sounding][sci026]")
{
    SECTION("qv override without LCL is accepted and retained")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
initial_conditions:
  sounding:
    type: parametric_targets
    moisture_qv_kgkg: 0.012
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->initial_conditions.sounding.moisture_qv_kgkg.has_value());
        REQUIRE(*result.config->initial_conditions.sounding.moisture_qv_kgkg == Approx(0.012));
    }

    SECTION("an explicit qv override cannot silently bypass an explicit LCL")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + R"yaml(
initial_conditions:
  sounding:
    type: parametric_targets
    lcl_m: 1000
    moisture_qv_kgkg: 0.012
)yaml");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "initial_conditions.sounding.moisture_qv_kgkg", "would bypass the LCL target"));
    }

    SECTION("keys owned by other sources are rejected")
    {
        const std::vector<std::pair<std::string, std::string>> rejected = {
            {"tropopause_m", "12000"},
            {"file_path", "profile.txt"},
        };
        for (const auto& [leaf, value] : rejected)
        {
            TempYaml   tmp(std::string(kMinimalValidV1) +
                           "initial_conditions:\n  sounding:\n"
                           "    type: parametric_targets\n    " +
                           leaf + ": " + value + "\n");
            const auto result = load_v1_config(tmp.path.string());
            INFO(format_errors(result.errors));
            REQUIRE_FALSE(result.ok());
            REQUIRE(has_error(
                result, "initial_conditions.sounding." + leaf, "not valid for sounding.type=parametric_targets"));
        }
    }

    SECTION("translated values that would saturate or clamp are rejected")
    {
        const std::vector<std::pair<std::string, std::string>> rejected = {
            {"cin_jkg", "301"},
            {"lcl_m", "0"},
            {"lcl_m", "50000"},
            {"lfc_m", "49"},
            {"el_m", "7999"},
            {"surface_theta_k", "249"},
            {"moisture_qv_kgkg", "0.003"},
            {"moisture_qv_kgkg", "0.025"},
        };
        for (const auto& [leaf, value] : rejected)
        {
            TempYaml   tmp(std::string(kMinimalValidV1) +
                           "initial_conditions:\n  sounding:\n"
                           "    type: parametric_targets\n    " +
                           leaf + ": " + value + "\n");
            const auto result = load_v1_config(tmp.path.string());
            INFO("key: " << leaf << " value: " << value);
            INFO(format_errors(result.errors));
            REQUIRE_FALSE(result.ok());
            REQUIRE(has_error(result, "initial_conditions.sounding." + leaf));
        }

        TempYaml   over_moist_lcl(std::string(kMinimalValidV1) + R"yaml(
initial_conditions:
  sounding:
    type: parametric_targets
    lcl_m: 1
    surface_theta_k: 340
)yaml");
        const auto over_moist_result = load_v1_config(over_moist_lcl.path.string());
        REQUIRE_FALSE(over_moist_result.ok());
        REQUIRE(has_error(over_moist_result, "initial_conditions.sounding.lcl_m", "would clamp it"));

        TempYaml   lfc_too_high(std::string(kMinimalValidV1) + R"yaml(
initial_conditions:
  sounding:
    type: parametric_targets
    lfc_m: 11500
    el_m: 12000
)yaml");
        const auto result = load_v1_config(lfc_too_high.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "initial_conditions.sounding.lfc_m", "at least 1000 m below el_m"));
    }
}

TEST_CASE("SCI-026: file sounding rejects every procedural source key", "[init][config][v1_loader][sounding][sci026]")
{
    const std::vector<std::pair<std::string, std::string>> rejected = {
        {"cape_jkg", "2500"},
        {"cin_jkg", "50"},
        {"lcl_m", "1200"},
        {"lfc_m", "2000"},
        {"el_m", "12000"},
        {"tropopause_m", "12000"},
        {"surface_theta_k", "300"},
        {"moisture_qv_kgkg", "0.014"},
    };
    for (const auto& [leaf, value] : rejected)
    {
        TempYaml   tmp(std::string(kMinimalValidV1) +
                       "initial_conditions:\n  sounding:\n"
                       "    type: file\n    file_path: profile.txt\n    " +
                       leaf + ": " + value + "\n");
        const auto result = load_v1_config(tmp.path.string());
        INFO("key: " << leaf);
        INFO(format_errors(result.errors));
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "initial_conditions.sounding." + leaf, "not valid for sounding.type=file"));
    }
}

// ─────────────────────────────────────────────────────────────
// Full v1 config end-to-end
// ─────────────────────────────────────────────────────────────

TEST_CASE("load_v1_config: full §20-shaped config round-trips through loader", "[init][config][v1_loader]")
{
    TempYaml tmp(R"yaml(
schema: nimbus.v1
grid:
  coordinate_system: cartesian
  staggering: c_grid
  nx: 96
  ny: 32
  nz: 40
  dx: 100.0
  dy: 100.0
  dz: 250.0
  ztop: 12000.0
time:
  run_time: 3600.0
  dt_large: 1.5
  acoustic_substeps: 8
  adaptive_dt: false
  cfl_safety: 0.6
  output_interval: 60.0
dynamics:
  scheme: klemp_wilhelmson_split
  axisymmetric: true
boundary_conditions:
  lateral:
    west: periodic
    east: periodic
    south: rigid
    north: rigid_slip
  bottom: rigid_noslip
  top: rigid_with_sponge
damping:
  enabled: true
  z_d: 10000.0
  alpha: 0.03
  mode: explicit
  lateral_width: 12
  lateral_alpha: 0.02
  w_only_top_sponge: false
  w_only_depth_cells: 6
  w_only_alpha_max: 0.10
coriolis:
  enabled: true
  mode: f_plane
  latitude_deg: 42.5
microphysics:
  scheme: kessler
turbulence:
  scheme: smagorinsky
  mode: 3d
  Cs: 0.20
  Pr_t: 0.7
  Sc_t: 0.7
diffusion:
  scheme: explicit
  operator_order: 2
  K_h: 0.0
  K_v: 25.0
  apply_to: scalars
surface:
  scheme: slab
  apply_fluxes: true
  z0m_m: 0.15
  z0h_m: 0.015
  albedo: 0.17
  emissivity: 0.93
  tsk_k: 301
  qsfc_kgkg: 0.014
boundary_layer:
  scheme: ysu
radiation:
  scheme: simple_grey
  solar_forcing: idealized_diurnal
  latitude_deg: 42.5
  declination_deg: -10
  local_solar_time_at_start_h: 6
  do_lw: true
  do_sw: true
  cadence_s: 120
  clear_sky_only: true
  tau_lw_ref: 5
  tau_sw_ref: 0.18
  n_lw: 3
  n_sw: 1.5
  terrain_sw_cosine_nodes: 12
  terrain_sw_azimuth_nodes: 24
  terrain_sw_maximum_bounces: 48
  terrain_sw_relative_residual_tolerance: 2e-11
  terrain_sw_absolute_residual_tolerance_w: 3e-8
initial_conditions:
  sounding:
    type: parametric_targets
    cape_jkg: 3000
    cin_jkg: 25
    lcl_m: 1000
    lfc_m: 1800
    el_m: 13000
    surface_theta_k: 303
  hodograph:
    type: custom
    u_sfc_ms: 5
    v_sfc_ms: 0
    u_1km_ms: 12
    v_1km_ms: 4
    u_6km_ms: 25
    v_6km_ms: 8
  trigger:
    type: bubble
    bubble:
      dtheta_k: 2.5
      cloud_water_kgkg: 0.002
      radius_km: 8.0
      center_x_m: 0.0
      center_y_m: 0.0
      center_z_m: 1500.0
output:
  preset: presentation
  format: npy_3d
  async_io: true
  live_shm: false
extensions:
  compute:
    backend: gpu
  chaos:
    scheme: initial_conditions
    seed: 12345
    member_id: 3
  validation:
    guard_mode: error
)yaml");

    auto result = load_v1_config(tmp.path.string());

    INFO(format_errors(result.errors));
    REQUIRE(result.ok());

    const auto& c = *result.config;

    REQUIRE(c.grid.coordinate_system == CoordinateSystem::Cartesian);
    REQUIRE(c.grid.staggering == StaggerType::CGrid);
    REQUIRE(c.grid.nx == 96);
    REQUIRE(c.grid.ztop == Approx(12000.0));

    REQUIRE(c.time.acoustic_substeps == 8);
    REQUIRE(c.time.cfl_safety == Approx(0.6));

    REQUIRE(c.dynamics.axisymmetric == true);

    REQUIRE(c.boundary_conditions.lateral.west == LateralBC::Periodic);
    REQUIRE(c.boundary_conditions.lateral.south == LateralBC::Rigid);
    REQUIRE(c.boundary_conditions.lateral.north == LateralBC::RigidSlip);

    REQUIRE(c.damping.z_d == Approx(10000.0));
    REQUIRE(c.damping.alpha == Approx(0.03));
    REQUIRE(c.damping.lateral_width == 12);
    REQUIRE(c.damping.lateral_alpha == Approx(0.02));
    REQUIRE(c.damping.w_only_top_sponge == false);
    REQUIRE(c.damping.w_only_depth_cells == 6);
    REQUIRE(c.damping.w_only_alpha_max == Approx(0.10));

    REQUIRE(c.coriolis.enabled == true);
    REQUIRE(c.coriolis.latitude_deg == Approx(42.5));

    REQUIRE(c.microphysics.scheme == MicrophysicsScheme::Kessler);

    REQUIRE(c.turbulence.Cs == Approx(0.20));

    REQUIRE(c.diffusion.K_h == Approx(0.0));
    REQUIRE(c.diffusion.K_v == Approx(25.0));

    REQUIRE(c.surface.scheme == SurfaceScheme::Slab);
    REQUIRE(c.surface.z0m_m == Approx(0.15));
    REQUIRE(c.surface.z0h_m == Approx(0.015));
    REQUIRE(c.surface.albedo == Approx(0.17));
    REQUIRE(c.surface.emissivity == Approx(0.93));
    REQUIRE(c.surface.tsk_k == Approx(301.0));
    REQUIRE(c.surface.qsfc_kgkg == Approx(0.014));
    REQUIRE(c.boundary_layer.scheme == BoundaryLayerScheme::YSU);
    REQUIRE(c.radiation.scheme == RadiationScheme::SimpleGrey);
    REQUIRE(c.radiation.solar_forcing == RadiationSolarForcing::IdealizedDiurnal);
    REQUIRE(c.radiation.latitude_deg == Approx(42.5));
    REQUIRE(c.radiation.declination_deg == Approx(-10.0));
    REQUIRE(c.radiation.local_solar_time_at_start_h == Approx(6.0));
    REQUIRE(c.radiation.cadence_s == Approx(120.0));
    REQUIRE(c.radiation.tau_lw_ref == Approx(5.0));
    REQUIRE(c.radiation.terrain_sw_cosine_nodes == 12);
    REQUIRE(c.radiation.terrain_sw_azimuth_nodes == 24);
    REQUIRE(c.radiation.terrain_sw_maximum_bounces == 48);
    REQUIRE(c.radiation.terrain_sw_relative_residual_tolerance
            == Approx(2.0e-11));
    REQUIRE(c.radiation.terrain_sw_absolute_residual_tolerance_w
            == Approx(3.0e-8));

    REQUIRE(c.initial_conditions.sounding.type == SoundingType::ParametricTargets);
    REQUIRE(c.initial_conditions.sounding.cape_jkg == Approx(3000.0));
    REQUIRE(c.initial_conditions.hodograph.u_6km_ms == Approx(25.0));
    REQUIRE(c.initial_conditions.trigger.type == TriggerType::Bubble);
    REQUIRE(c.initial_conditions.trigger.bubble.dtheta_k == Approx(2.5));
    REQUIRE(c.initial_conditions.trigger.bubble.cloud_water_kgkg
            == Approx(0.002));

    REQUIRE(c.output.preset == OutputPreset::Presentation);

    REQUIRE(c.extensions.compute.backend == ComputeBackend::GPU);
    REQUIRE(c.extensions.chaos.scheme == ChaosScheme::InitialConditions);
    REQUIRE(c.extensions.chaos.seed == 12345);
    REQUIRE(c.extensions.validation.guard_mode == ValidationGuardMode::Error);
}

TEST_CASE("load_v1_config: bubble cloud-water seed is non-negative",
          "[init][config][v1_loader][trigger][microphysics]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(
initial_conditions:
  trigger:
    type: bubble
    bubble:
      cloud_water_kgkg: -0.001
)yaml");
    const auto result = load_v1_config(tmp.path.string());
    REQUIRE_FALSE(result.ok());
    REQUIRE(has_error(
        result,
        "initial_conditions.trigger.bubble.cloud_water_kgkg",
        "must be >= 0"));
}

// ─────────────────────────────────────────────────────────────
// Compute backend vocabulary
// ─────────────────────────────────────────────────────────────

TEST_CASE("load_v1_config: compute backend accepts both vocabularies", "[init][config][v1_loader]")
{
    auto config_with_backend = [](const char* backend)
    { return std::string(kMinimalValidV1) + "extensions:\n  compute:\n    backend: " + backend + "\n"; };

    SECTION("vulkan parses as the GPU backend")
    {
        TempYaml tmp(config_with_backend("vulkan"));
        auto     result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->extensions.compute.backend == ComputeBackend::GPU);
    }

    SECTION("vk parses as the GPU backend")
    {
        TempYaml tmp(config_with_backend("vk"));
        auto     result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->extensions.compute.backend == ComputeBackend::GPU);
    }

    SECTION("metal preserves the native Metal backend")
    {
        TempYaml tmp(config_with_backend("metal"));
        auto     result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->extensions.compute.backend == ComputeBackend::Metal);
    }

    SECTION("mtl is a Metal alias")
    {
        TempYaml tmp(config_with_backend("mtl"));
        auto     result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->extensions.compute.backend == ComputeBackend::Metal);
    }

    SECTION("an unknown backend is a load error, not a silent default")
    {
        TempYaml tmp(config_with_backend("cuda"));
        auto     result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE_FALSE(result.errors.empty());
    }
}

TEST_CASE("load_v1_config: execution policy is explicit and strict", "[init][config][v1_loader][execution]")
{
    SECTION("qualified is retained without a maturity gate")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) +
                       "extensions:\n  execution:\n    mode: qualified\n");
        const auto result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->extensions.execution.mode == ExecutionMode::Qualified);
        REQUIRE_FALSE(result.config->extensions.execution.acknowledge_maturity_limits);
    }

    SECTION("qualified rejects an explicit false legacy acknowledgment")
    {
        TempYaml tmp(std::string(kMinimalValidV1) +
                     "extensions:\n  execution:\n    mode: qualified\n"
                     "    acknowledge_maturity_limits: false\n");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(
            result, "extensions.execution.acknowledge_maturity_limits", "explicit false"));
    }

    SECTION("exploratory rejects a stale maturity acknowledgment")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) +
                       "extensions:\n  execution:\n    mode: exploratory\n    acknowledge_maturity_limits: true\n");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(
            result, "extensions.execution.acknowledge_maturity_limits", "active only for qualified execution"));
    }

    SECTION("exploratory is the compatibility default")
    {
        TempYaml   tmp(kMinimalValidV1);
        const auto result = load_v1_config(tmp.path.string());
        INFO(format_errors(result.errors));
        REQUIRE(result.ok());
        REQUIRE(result.config->extensions.execution.mode == ExecutionMode::Exploratory);
    }

    SECTION("unknown recovery policy fails closed")
    {
        TempYaml   tmp(std::string(kMinimalValidV1) + "extensions:\n  execution:\n    mode: best_effort\n");
        const auto result = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_error(result, "extensions.execution.mode", "exploratory, qualified"));
    }
}

TEST_CASE("load_v1_config: checkpoint lifecycle is strict and Nimbus-only", "[init][config][v1_loader][checkpoint]")
{
    TempYaml tmp(std::string(kMinimalValidV1) + "extensions:\n"
                                                "  checkpoint:\n"
                                                "    every_s: 300\n"
                                                "    keep_last: 4\n"
                                                "    minimum_free_space_mib: 2048\n"
                                                "    directory: /research/checkpoints\n"
                                                "    resume_from: /research/parent/checkpoint_step_000000100.nimbus\n");
    const auto result = load_v1_config(tmp.path.string());
    INFO(format_errors(result.errors));
    REQUIRE(result.ok());
    REQUIRE(result.config->extensions.checkpoint.every_s == 300);
    REQUIRE(result.config->extensions.checkpoint.keep_last == 4);
    REQUIRE(result.config->extensions.checkpoint.minimum_free_space_mib == 2048);
    REQUIRE(result.config->extensions.checkpoint.directory == "/research/checkpoints");
    REQUIRE(result.config->extensions.checkpoint.resume_from.find("checkpoint_step_") != std::string::npos);
}

TEST_CASE("load_v1_config: chaos targets and amplitudes fail closed",
          "[init][config][v1_loader][chaos]")
{
    SECTION("known targets and non-negative amplitudes load")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(extensions:
  chaos:
    scheme: full_stochastic
    apply_to_ic: [u, theta, qv]
    apply_to_tendencies: [microphysics, pbl, diffusion]
    sigma_ic:
      theta: 0.75
      qv: 0.0002
    alpha_tend:
      microphysics: 0.4
      pbl: 0.25
)yaml");
        const auto loaded = load_v1_config(tmp.path.string());
        INFO(format_errors(loaded.errors));
        REQUIRE(loaded.ok());
        REQUIRE(loaded.config->extensions.chaos.apply_to_ic ==
                std::vector<std::string>{"u", "theta", "qv"});
        REQUIRE(loaded.config->extensions.chaos.apply_to_tendencies ==
                std::vector<std::string>{"microphysics", "pbl", "diffusion"});
        REQUIRE(loaded.config->extensions.chaos.sigma_ic.at("theta") ==
                Approx(0.75));
        REQUIRE(loaded.config->extensions.chaos.alpha_tend.at("pbl") ==
                Approx(0.25));
    }

    SECTION("unknown targets, duplicates, and negative amplitudes reject")
    {
        TempYaml tmp(std::string(kMinimalValidV1) + R"yaml(extensions:
  chaos:
    apply_to_ic: [theta, theta, pressure]
    apply_to_tendencies: [radiation]
    sigma_ic:
      pressure: 1.0
      qv: -0.1
    alpha_tend:
      radiation: 0.2
)yaml");
        const auto loaded = load_v1_config(tmp.path.string());
        REQUIRE_FALSE(loaded.ok());
        REQUIRE(has_error(
            loaded, "extensions.chaos.apply_to_ic", "duplicate"));
        REQUIRE(has_error(
            loaded, "extensions.chaos.apply_to_ic", "unsupported"));
        REQUIRE(has_error(
            loaded, "extensions.chaos.apply_to_tendencies", "unsupported"));
        REQUIRE(has_error(
            loaded, "extensions.chaos.sigma_ic.pressure", "unknown"));
        REQUIRE(has_error(
            loaded, "extensions.chaos.sigma_ic.qv", "must be >= 0"));
        REQUIRE(has_error(
            loaded, "extensions.chaos.alpha_tend.radiation", "unknown"));
    }
}

TEST_CASE("load_v1_config: output retention controls are non-negative",
          "[init][config][v1_loader][output][retention]")
{
    TempYaml valid(std::string(kMinimalValidV1) + "output:\n"
                                                  "  keep_last_frames: 12\n"
                                                  "  minimum_free_space_mib: 4096\n"
                                                  "  maximum_retained_history_files: 512\n");
    const auto loaded = load_v1_config(valid.path.string());
    INFO(format_errors(loaded.errors));
    REQUIRE(loaded.ok());
    REQUIRE(loaded.config->output.keep_last_frames == 12);
    REQUIRE(loaded.config->output.minimum_free_space_mib == 4096);
    REQUIRE(loaded.config->output.maximum_retained_history_files == 512);

    TempYaml invalid(std::string(kMinimalValidV1) + "output:\n  keep_last_frames: -1\n");
    const auto rejected = load_v1_config(invalid.path.string());
    REQUIRE_FALSE(rejected.ok());
    REQUIRE(has_error(rejected, "output.keep_last_frames", "non-negative"));
}

// ─────────────────────────────────────────────────────────────
// Error formatting
// ─────────────────────────────────────────────────────────────

TEST_CASE("format_errors emits file:line: key: message", "[init][config][v1_loader]")
{
    std::vector<LoadError> errs = {
        {"foo.yaml", 12, "grid.nx", "must be > 0; got -1"},
        {"foo.yaml", 0, "time.run_time", "required key missing"},
    };
    const std::string formatted = format_errors(errs);
    REQUIRE(formatted.find("foo.yaml:12: grid.nx: must be > 0") != std::string::npos);
    REQUIRE(formatted.find("foo.yaml: time.run_time: required key missing") != std::string::npos);
}

TEST_CASE("Configuration serializers preserve double round trips", "[init][config][precision]")
{
    TempYaml source(kMinimalValidV1);
    auto loaded = load_v1_config(source.path.string());
    REQUIRE(loaded.ok());
    auto config = *loaded.config;
    config.grid.dx = std::nextafter(250.0, 251.0);
    config.time.cfl_safety = std::nextafter(0.7, 1.0);
    const auto projected = serialize_v1_to_v0_map(config);
    REQUIRE(std::stod(projected.at("grid.dx")) == config.grid.dx);
    REQUIRE(std::stod(projected.at("numerics.time_stepping.cfl_safety")) == config.time.cfl_safety);
    TempYaml serialized(serialize_v1_to_yaml(config));
    const auto restored = load_v1_config(serialized.path.string());
    REQUIRE(restored.ok());
    REQUIRE(restored.config->grid.dx == config.grid.dx);
    REQUIRE(restored.config->time.cfl_safety == config.time.cfl_safety);
}
