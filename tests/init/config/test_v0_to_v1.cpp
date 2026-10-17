/**
 * @file test_v0_to_v1.cpp
 * @brief Unit tests for the v0→v1 config migrator (Phase A3).
 *
 * The exit criterion from docs/CM1Mapping.md §21:
 *   "Every current tests/fixtures/configs/test/<name>.yaml round-trips through the migrator
 *    and produces a v1 config that loads cleanly via A2."
 *
 * This file enforces that with:
 *   - A directory sweep over `tests/fixtures/configs/test/` that migrates each file,
 *     serializes the result, and feeds it back into `load_v1_config`.
 *   - Targeted cases covering the trickier translations: scheme-name
 *     decomposition, km→m unit conversion, env→initial_conditions move,
 *     terrain.ztop → grid.ztop, output.fields → output.preset, etc.
 *   - A negative case rejecting v1 files (so `--migrate-config` is
 *     guaranteed to be a one-way function).
 */

#include "catch2/catch.hpp"
#include "init/config/v0_to_v1.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <random>
#include <sstream>
#include <string>

using namespace tmv::init::config;

namespace
{

struct TempYaml
{
    std::filesystem::path path;

    explicit TempYaml(const std::string& contents)
    {
        std::random_device                      rd;
        std::mt19937                            rng(rd());
        std::uniform_int_distribution<uint64_t> dist;
        path = std::filesystem::temp_directory_path() / ("nimbus_v0v1_test_" + std::to_string(dist(rng)) + ".yaml");
        std::ofstream f(path);
        f << contents;
    }

    ~TempYaml()
    {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

/// Migrate v0 → serialize v1 → re-parse via load_v1_config. Returns the
/// LoadResult of the final parse. This is the round-trip the §21 exit
/// criterion specifies.
LoadResult round_trip(const std::string& v0_path)
{
    auto migration = migrate_v0_to_v1(v0_path);
    if (!migration.ok())
    {
        // Surface the errors for diagnostic visibility before the test fails.
        INFO("migration errors for " << v0_path);
        for (const auto& e : migration.errors)
        {
            INFO(e.file << ":" << e.line << ": " << e.key << ": " << e.message);
        }
        return LoadResult{
            std::nullopt,
            {LoadError{v0_path,
                       0,
                       "",
                       "migration produced no config: " + std::to_string(migration.errors.size()) + " errors"}}};
    }
    const std::string v1_yaml = serialize_v1_to_yaml(*migration.config);
    TempYaml          tmp(v1_yaml);
    return load_v1_config(tmp.path.string());
}

[[maybe_unused]] bool has_load_error(const LoadResult& result,
                                     const std::string& key,
                                     const std::string& message_fragment)
{
    return std::any_of(result.errors.begin(),
                       result.errors.end(),
                       [&](const LoadError& error)
                       { return error.key == key && error.message.find(message_fragment) != std::string::npos; });
}

template <typename Enum>
void require_canonical_enum_round_trip(std::initializer_list<Enum> values,
                                       bool (*parse)(const std::string&, Enum&),
                                       const char* (*name)(Enum))
{
    for (const Enum expected : values)
    {
        const std::string token = name(expected);
        INFO("canonical enum token: " << token);
        REQUIRE_FALSE(token.empty());
        Enum parsed{};
        REQUIRE(parse(token, parsed));
        REQUIRE(parsed == expected);
    }
}

} // namespace

TEST_CASE("v0 migrator rejects invalid boolean instead of selecting false", "[init][config][migration][config-truth]")
{
    TempYaml              tmp(R"yaml(
grid:
  nr: 16
  nth: 16
  nz: 8
  dr: 250
  dz: 250
numerics:
  time_stepping:
    adaptive_dt: sometimes
)yaml");
    const MigrationResult result = migrate_v0_to_v1(tmp.path.string());
    REQUIRE_FALSE(result.ok());
    REQUIRE(std::any_of(result.errors.begin(),
                        result.errors.end(),
                        [](const LoadError& error)
                        {
                            return error.key == "numerics.time_stepping.adaptive_dt" &&
                                   error.message.find("cannot parse") != std::string::npos;
                        }));
}

TEST_CASE("scientific enum acceptance surface has complete canonical projections",
          "[init][config][sci021][completeness]")
{
    require_canonical_enum_round_trip<LateralBC>({LateralBC::Periodic,
                                                  LateralBC::Radiative,
                                                  LateralBC::Rigid,
                                                  LateralBC::RigidSlip,
                                                  LateralBC::ZeroGradient,
                                                  LateralBC::LegacyOpen},
                                                 parse_lateral_bc,
                                                 lateral_bc_name);
    require_canonical_enum_round_trip<BottomBC>(
        {BottomBC::RigidNoSlip, BottomBC::RigidSlip}, parse_bottom_bc, bottom_bc_name);
    require_canonical_enum_round_trip<TopBC>(
        {TopBC::Rigid, TopBC::RigidWithSponge, TopBC::Open}, parse_top_bc, top_bc_name);
    require_canonical_enum_round_trip<DampingMode>(
        {DampingMode::Explicit, DampingMode::Implicit}, parse_damping_mode, damping_mode_name);
    require_canonical_enum_round_trip<CoriolisMode>(
        {CoriolisMode::FPlane, CoriolisMode::BetaPlane}, parse_coriolis_mode, coriolis_mode_name);
    require_canonical_enum_round_trip<MicrophysicsScheme>({MicrophysicsScheme::None,
                                                           MicrophysicsScheme::Kessler,
                                                           MicrophysicsScheme::Lin,
                                                           MicrophysicsScheme::Thompson,
                                                           MicrophysicsScheme::Milbrandt},
                                                          parse_microphysics_scheme,
                                                          microphysics_scheme_name);
    require_canonical_enum_round_trip<TurbulenceScheme>(
        {TurbulenceScheme::None, TurbulenceScheme::Smagorinsky, TurbulenceScheme::TKE},
        parse_turbulence_scheme,
        turbulence_scheme_name);
    require_canonical_enum_round_trip<TurbulenceMode>(
        {TurbulenceMode::TwoD, TurbulenceMode::VerticalOnly, TurbulenceMode::ThreeD},
        parse_turbulence_mode,
        turbulence_mode_name);
    require_canonical_enum_round_trip<AdvectionScheme>(
        {AdvectionScheme::Auto, AdvectionScheme::None,
         AdvectionScheme::TVD, AdvectionScheme::WENO5},
        parse_advection_scheme,
        advection_scheme_name);
    require_canonical_enum_round_trip<DiffusionScheme>(
        {DiffusionScheme::None, DiffusionScheme::Explicit, DiffusionScheme::Implicit},
        parse_diffusion_scheme,
        diffusion_scheme_name);
    require_canonical_enum_round_trip<DiffusionTarget>(
        {DiffusionTarget::Scalars, DiffusionTarget::ScalarsAndMomentum}, parse_diffusion_target, diffusion_target_name);
    require_canonical_enum_round_trip<SurfaceScheme>(
        {SurfaceScheme::None, SurfaceScheme::Slab}, parse_surface_scheme, surface_scheme_name);
    require_canonical_enum_round_trip<BoundaryLayerScheme>(
        {BoundaryLayerScheme::None, BoundaryLayerScheme::Slab, BoundaryLayerScheme::YSU, BoundaryLayerScheme::MYNN},
        parse_boundary_layer_scheme,
        boundary_layer_scheme_name);
    require_canonical_enum_round_trip<BoundaryLayerTopMethod>(
        {BoundaryLayerTopMethod::Richardson,
         BoundaryLayerTopMethod::PotentialTemperature,
         BoundaryLayerTopMethod::TKE},
        parse_boundary_layer_top_method,
        boundary_layer_top_method_name);
    require_canonical_enum_round_trip<RadiationScheme>(
        {RadiationScheme::None, RadiationScheme::SimpleGrey}, parse_radiation_scheme, radiation_scheme_name);
    require_canonical_enum_round_trip<TriggerType>(
        {TriggerType::None, TriggerType::Bubble, TriggerType::VortexSeed}, parse_trigger_type, trigger_type_name);
    require_canonical_enum_round_trip<SoundingType>(
        {SoundingType::ParametricCAPE, SoundingType::ParametricTargets, SoundingType::File},
        parse_sounding_type,
        sounding_type_name);
    require_canonical_enum_round_trip<HodographType>(
        {HodographType::Auto, HodographType::Custom, HodographType::WKParam},
        parse_hodograph_type,
        hodograph_type_name);
    require_canonical_enum_round_trip<OutputPreset>({OutputPreset::Minimal,
                                                     OutputPreset::Presentation,
                                                     OutputPreset::StudioFull,
                                                     OutputPreset::DynamicsFull,
                                                     OutputPreset::ScienceValidation},
                                                    parse_output_preset,
                                                    output_preset_name);
    require_canonical_enum_round_trip<OutputFormat>({OutputFormat::Npy3D,
                                                     OutputFormat::Netcdf,
                                                     OutputFormat::ZfpLow,
                                                     OutputFormat::ZfpMedium,
                                                     OutputFormat::ZfpHigh},
                                                    parse_output_format,
                                                    output_format_name);
    require_canonical_enum_round_trip<ComputeBackend>(
        {ComputeBackend::CPU, ComputeBackend::GPU, ComputeBackend::Metal}, parse_compute_backend, compute_backend_name);
    require_canonical_enum_round_trip<ChaosScheme>(
        {ChaosScheme::None, ChaosScheme::InitialConditions, ChaosScheme::BoundaryLayer, ChaosScheme::FullStochastic},
        parse_chaos_scheme,
        chaos_scheme_name);
    require_canonical_enum_round_trip<ValidationGuardMode>(
        {ValidationGuardMode::Off, ValidationGuardMode::Warn, ValidationGuardMode::Error},
        parse_validation_guard_mode,
        validation_guard_mode_name);
    require_canonical_enum_round_trip<ExecutionMode>(
        {ExecutionMode::Exploratory, ExecutionMode::Qualified}, parse_execution_mode, execution_mode_name);
}

TEST_CASE("qualified execution projects fail-closed runtime policy", "[init][config][execution][qualified]")
{
    ConfigV1 config;
    config.extensions.execution.mode                        = ExecutionMode::Qualified;
    config.extensions.execution.acknowledge_maturity_limits = true;
    config.extensions.validation.guard_mode                 = ValidationGuardMode::Warn;
    config.extensions.compute.allow_fallback                = true;

    const auto projected = serialize_v1_to_v0_map(config);
    REQUIRE(projected.at("execution.mode") == "qualified");
    REQUIRE(projected.count("execution.acknowledge_maturity_limits") == 0);
    REQUIRE(projected.at("validation.guard_mode") == "error");
    REQUIRE(projected.at("numerics.compute.allow_fallback") == "false");
    REQUIRE(projected.at("environment.sounding.use_fallback_profiles") == "false");

    const std::string yaml = serialize_v1_to_yaml(config);
    REQUIRE(yaml.find("    allow_fallback: true\n") != std::string::npos);
    REQUIRE(yaml.find("  execution:\n    mode: qualified\n") != std::string::npos);
    REQUIRE(yaml.find("acknowledge_maturity_limits") == std::string::npos);
}

TEST_CASE("advection selection round-trips and auto preserves historical grid behavior",
          "[init][config][advection][serializer][projection][migration]")
{
    ConfigV1 config;
    config.grid.nx = 16;
    config.grid.ny = 16;
    config.grid.nz = 8;
    config.grid.dx = 250.0;
    config.grid.dy = 250.0;
    config.grid.dz = 250.0;
    config.time.run_time = 60.0;
    config.time.dt_large = 1.0;
    config.grid.staggering = StaggerType::CGrid;
    REQUIRE(serialize_v1_to_v0_map(config).at("numerics.advection.scheme")
            == "weno5");

    config.grid.staggering = StaggerType::Collocated;
    REQUIRE(serialize_v1_to_v0_map(config).at("numerics.advection.scheme")
            == "tvd");

    config.advection.scheme = AdvectionScheme::WENO5;
    const std::string yaml = serialize_v1_to_yaml(config);
    REQUIRE(yaml.find("advection:\n  scheme: weno5\n") != std::string::npos);

    TempYaml strict(yaml);
    const auto loaded = load_v1_config(strict.path.string());
    INFO(format_errors(loaded.errors));
    REQUIRE(loaded.ok());
    REQUIRE(loaded.config->advection.scheme == AdvectionScheme::WENO5);

    TempYaml legacy(R"yaml(
grid:
  nr: 16
  nth: 16
  nz: 8
  dr: 250
  dz: 250
numerics:
  advection:
    scheme: weno
)yaml");
    const auto migrated = migrate_v0_to_v1(legacy.path.string());
    INFO(format_errors(migrated.errors));
    REQUIRE(migrated.ok());
    REQUIRE(migrated.config->advection.scheme == AdvectionScheme::WENO5);
}

TEST_CASE("compute controls serialize, load, and project without provenance loss",
          "[init][config][compute][serializer][projection]")
{
    ConfigV1 config;
    config.grid.nx                                   = 16;
    config.grid.ny                                   = 12;
    config.grid.nz                                   = 8;
    config.grid.dx                                   = 500.0;
    config.grid.dy                                   = 750.0;
    config.grid.dz                                   = 250.0;
    config.time.run_time                             = 60.0;
    config.time.dt_large                             = 1.0;
    config.time.output_interval                      = 10.0;
    config.extensions.compute.backend                = ComputeBackend::GPU;
    config.extensions.compute.memory_admission       = MemoryAdmissionPolicy::Reject;
    config.extensions.compute.allow_fallback         = false;
    config.extensions.compute.validate_parity        = true;
    config.extensions.compute.gpu_apply_microphysics = false;
    config.extensions.compute.gpu_apply_slow         = true;
    config.extensions.compute.gpu_apply_sgs          = true;

    const std::string yaml = serialize_v1_to_yaml(config);
    REQUIRE(yaml.find("    backend: gpu\n") != std::string::npos);
    REQUIRE(yaml.find("    memory_admission: reject\n") != std::string::npos);
    REQUIRE(yaml.find("    allow_fallback: false\n") != std::string::npos);
    REQUIRE(yaml.find("    validate_parity: true\n") != std::string::npos);
    REQUIRE(yaml.find("    gpu_apply_microphysics: false\n") != std::string::npos);
    REQUIRE(yaml.find("    gpu_apply_slow: true\n") != std::string::npos);
    REQUIRE(yaml.find("    gpu_apply_sgs: true\n") != std::string::npos);

    TempYaml   tmp(yaml);
    const auto loaded = load_v1_config(tmp.path.string());
    INFO(format_errors(loaded.errors));
    REQUIRE(loaded.ok());
    const auto& compute = loaded.config->extensions.compute;
    REQUIRE(compute.backend == ComputeBackend::GPU);
    REQUIRE(compute.memory_admission == MemoryAdmissionPolicy::Reject);
    REQUIRE_FALSE(compute.allow_fallback);
    REQUIRE(compute.validate_parity);
    REQUIRE_FALSE(compute.gpu_apply_microphysics);
    REQUIRE(compute.gpu_apply_slow);
    REQUIRE(compute.gpu_apply_sgs);

    const auto projected = serialize_v1_to_v0_map(*loaded.config);
    REQUIRE(projected.at("numerics.compute.backend") == "gpu");
    REQUIRE(projected.at("hardware.memory_admission") == "reject");
    REQUIRE(projected.at("numerics.compute.allow_fallback") == "false");
    REQUIRE(projected.at("numerics.compute.validate_parity") == "true");
    REQUIRE(projected.at("numerics.compute.gpu_apply_microphysics") == "false");
    REQUIRE(projected.at("numerics.compute.gpu_apply_slow") == "true");
    REQUIRE(projected.at("numerics.compute.gpu_apply_sgs") == "true");
}

TEST_CASE("chaos targets and amplitudes serialize, load, and project without loss",
          "[init][config][chaos][serializer][projection]")
{
    ConfigV1 config;
    config.grid.nx = 16;
    config.grid.ny = 12;
    config.grid.nz = 8;
    config.grid.dx = 500.0;
    config.grid.dy = 750.0;
    config.grid.dz = 250.0;
    config.time.run_time = 60.0;
    config.time.dt_large = 1.0;
    config.time.output_interval = 10.0;
    auto& chaos = config.extensions.chaos;
    chaos.scheme = ChaosScheme::FullStochastic;
    chaos.seed = 12345;
    chaos.member_id = 7;
    chaos.apply_to_ic = {"u", "theta", "qv"};
    chaos.apply_to_tendencies = {"microphysics", "diffusion"};
    chaos.sigma_ic["theta"] = 0.75;
    chaos.sigma_ic["qv"] = 0.0002;
    chaos.alpha_tend["microphysics"] = 0.45;
    chaos.alpha_tend["diffusion"] = 0.15;

    const std::string yaml = serialize_v1_to_yaml(config);
    REQUIRE(yaml.find("apply_to_ic: [u, theta, qv]") != std::string::npos);
    REQUIRE(yaml.find("apply_to_tendencies: [microphysics, diffusion]") !=
            std::string::npos);
    REQUIRE(yaml.find("sigma_ic:\n") != std::string::npos);
    REQUIRE(yaml.find("alpha_tend:\n") != std::string::npos);

    TempYaml tmp(yaml);
    const auto loaded = load_v1_config(tmp.path.string());
    INFO(format_errors(loaded.errors));
    REQUIRE(loaded.ok());
    const auto& loaded_chaos = loaded.config->extensions.chaos;
    REQUIRE(loaded_chaos.apply_to_ic == chaos.apply_to_ic);
    REQUIRE(loaded_chaos.apply_to_tendencies == chaos.apply_to_tendencies);
    REQUIRE(loaded_chaos.sigma_ic.at("theta") == Approx(0.75));
    REQUIRE(loaded_chaos.sigma_ic.at("qv") == Approx(0.0002));
    REQUIRE(loaded_chaos.alpha_tend.at("microphysics") == Approx(0.45));
    REQUIRE(loaded_chaos.alpha_tend.at("diffusion") == Approx(0.15));

    const auto projected = serialize_v1_to_v0_map(*loaded.config);
    REQUIRE(projected.at("chaos.apply_to_ic") == "u,theta,qv");
    REQUIRE(projected.at("chaos.apply_to_tendencies") ==
            "microphysics,diffusion");
    REQUIRE(projected.at("chaos.sigma_ic.theta") == "0.75");
    REQUIRE(projected.at("chaos.sigma_ic.qv") == "0.0002");
    REQUIRE(projected.at("chaos.alpha_tend.microphysics") == "0.45");
    REQUIRE(projected.at("chaos.alpha_tend.diffusion") == "0.15");
}

TEST_CASE("v0 compute fallback policy migrates into the typed v1 contract", "[init][config][compute][migration]")
{
    TempYaml   tmp(R"yaml(
duration_s: 10
grid:
  nx: 8
  ny: 8
  nz: 8
  dx: 500
  dy: 500
  dz: 250
  dt: 1
dynamics:
  scheme: cartesian
output:
  write_every_s: 0
numerics:
  compute:
    allow_fallback: false
)yaml");
    const auto migration = migrate_v0_to_v1(tmp.path.string());
    REQUIRE(migration.ok());
    REQUIRE_FALSE(migration.config->extensions.compute.allow_fallback);

    const std::string yaml = serialize_v1_to_yaml(*migration.config);
    REQUIRE(yaml.find("    allow_fallback: false\n") != std::string::npos);
    TempYaml   reloaded_yaml(yaml);
    const auto reloaded = load_v1_config(reloaded_yaml.path.string());
    INFO(format_errors(reloaded.errors));
    REQUIRE(reloaded.ok());
    REQUIRE_FALSE(reloaded.config->extensions.compute.allow_fallback);
}

TEST_CASE("output trajectory config serializes and projects without losing provenance",
          "[init][config][output][trajectories][serializer]")
{
    ConfigV1 config;
    config.output.preset                 = OutputPreset::StudioFull;
    config.output.trajectories           = true;
    config.output.trajectory_seed_time_s = 75.25;
    config.output.trajectory_count       = 9;
    config.output.trajectory_sample_interval_s = 2.5;
    config.output.trajectory_max_samples_per_parcel = 1200;

    const auto flat = serialize_v1_to_v0_map(config);
    REQUIRE(flat.at("output.fields") == "studio_full");
    REQUIRE(flat.at("output.trajectories") == "true");
    REQUIRE(flat.at("output.trajectory_seed_time_s") == "75.25");
    REQUIRE(flat.at("output.trajectory_count") == "9");
    REQUIRE(flat.at("output.trajectory_sample_interval_s") == "2.5");
    REQUIRE(flat.at("output.trajectory_max_samples_per_parcel") == "1200");

    const std::string yaml = serialize_v1_to_yaml(config);
    REQUIRE(yaml.find("  preset: studio_full\n") != std::string::npos);
    REQUIRE(yaml.find("  trajectories: true\n") != std::string::npos);
    REQUIRE(yaml.find("  trajectory_seed_time_s: 75.25\n") != std::string::npos);
    REQUIRE(yaml.find("  trajectory_count: 9\n") != std::string::npos);
    REQUIRE(yaml.find("  trajectory_sample_interval_s: 2.5\n") !=
            std::string::npos);
    REQUIRE(yaml.find("  trajectory_max_samples_per_parcel: 1200\n") !=
            std::string::npos);
}

TEST_CASE("live shared-memory contract serializes and projects canonically",
          "[init][config][output][shm][serializer]")
{
    ConfigV1 config;
    config.output.live_shm = true;
    config.output.live_shm_fields = {"w", "theta"};
    config.output.live_shm_interval_s = 2.5;
    config.output.live_shm_full_rate = true;

    const auto flat = serialize_v1_to_v0_map(config);
    REQUIRE(flat.at("output.live_shm") == "true");
    REQUIRE(flat.at("output.live_shm_fields") == "w,theta");
    REQUIRE(flat.at("output.live_shm_interval_s") == "2.5");
    REQUIRE(flat.at("output.live_shm_full_rate") == "true");

    const std::string yaml = serialize_v1_to_yaml(config);
    REQUIRE(yaml.find("  live_shm_fields: [w, theta]\n") !=
            std::string::npos);
    REQUIRE(yaml.find("  live_shm_interval_s: 2.5\n") !=
            std::string::npos);
    REQUIRE(yaml.find("  live_shm_full_rate: true\n") !=
            std::string::npos);
}

TEST_CASE("ZFP field execution policy serializes and projects canonically",
          "[init][config][output][zfp][parallel][serializer]")
{
    ConfigV1 config;
    config.output.format = OutputFormat::ZfpHigh;
    config.output.zfp_field_execution = "field_parallel";
    config.output.zfp_field_workers = 4;
    config.output.zfp_field_scratch_mib = 128;
    const auto flat = serialize_v1_to_v0_map(config);
    REQUIRE(flat.at("output.zfp_field_execution") == "field_parallel");
    REQUIRE(flat.at("output.zfp_field_workers") == "4");
    REQUIRE(flat.at("output.zfp_field_scratch_mib") == "128");
    const std::string yaml = serialize_v1_to_yaml(config);
    REQUIRE(yaml.find("  zfp_field_execution: field_parallel\n") !=
            std::string::npos);
    REQUIRE(yaml.find("  zfp_field_workers: 4\n") != std::string::npos);
    REQUIRE(yaml.find("  zfp_field_scratch_mib: 128\n") !=
            std::string::npos);
}

TEST_CASE("science-validation output preset serializes and projects canonically",
          "[init][config][output][science_validation][serializer]")
{
    ConfigV1 config;
    config.output.preset = OutputPreset::ScienceValidation;

    const auto flat = serialize_v1_to_v0_map(config);
    REQUIRE(flat.at("output.preset") == "science_validation");
    REQUIRE(flat.at("output.fields") == "science_validation");

    const std::string yaml = serialize_v1_to_yaml(config);
    REQUIRE(yaml.find("  preset: science_validation\n") != std::string::npos);
}

TEST_CASE("moving-reference qualification diagnostics serialize and project canonically",
          "[init][config][output][qualification][serializer]")
{
    ConfigV1 config;
    config.output.qualification_diagnostics_preset     = QualificationDiagnosticsPreset::MovingReferenceV1;
    config.output.qualification_diagnostics_interval_s = 30;

    const auto flat = serialize_v1_to_v0_map(config);
    REQUIRE(flat.at("output.qualification_diagnostics.preset") == "moving_reference_v1");
    REQUIRE(flat.at("output.qualification_diagnostics.interval_s") == "30");

    const std::string yaml = serialize_v1_to_yaml(config);
    REQUIRE(yaml.find("  qualification_diagnostics:\n"
                      "    preset: moving_reference_v1\n"
                      "    interval_s: 30\n") != std::string::npos);

    ConfigV1   defaults;
    const auto default_flat = serialize_v1_to_v0_map(defaults);
    REQUIRE(default_flat.count("output.qualification_diagnostics.preset") == 0);
    REQUIRE(serialize_v1_to_yaml(defaults).find("qualification_diagnostics") == std::string::npos);
}

TEST_CASE("SCI-017 serialization removes the legacy open token", "[init][config][boundary_conditions][sci017]")
{
    ConfigV1 config;

    SECTION("C-grid defaults serialize as radiative")
    {
        config.grid.staggering = StaggerType::CGrid;
        const std::string yaml = serialize_v1_to_yaml(config);
        REQUIRE(yaml.find("    west: radiative") != std::string::npos);
        REQUIRE(yaml.find("    east: radiative") != std::string::npos);
        REQUIRE(yaml.find("    west: open") == std::string::npos);
    }

    SECTION("collocated defaults serialize as zero-gradient")
    {
        config.grid.staggering = StaggerType::Collocated;
        const std::string yaml = serialize_v1_to_yaml(config);
        REQUIRE(yaml.find("    west: zero_gradient") != std::string::npos);
        REQUIRE(yaml.find("    east: zero_gradient") != std::string::npos);
        REQUIRE(yaml.find("    west: open") == std::string::npos);
    }
}

// ─────────────────────────────────────────────────────────────
// EXIT CRITERION: round-trip every tests/fixtures/configs/test/*.yaml
// ─────────────────────────────────────────────────────────────

TEST_CASE("A3 exit criterion: every v0 config round-trips through migrator", "[init][config][v0_to_v1][exit_criterion]")
{
    namespace fs = std::filesystem;
    // A4: live tests/fixtures/configs/test/*.yaml are now v1; the v0 originals live under
    // tests/fixtures/configs/test/legacy/ for this regression sweep.
    const fs::path configs_dir{"tests/fixtures/configs/test/legacy"};
    REQUIRE(fs::exists(configs_dir));

    std::vector<fs::path> yaml_files;
    for (const auto& entry : fs::directory_iterator(configs_dir))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".yaml")
        {
            yaml_files.push_back(entry.path());
        }
    }
    REQUIRE_FALSE(yaml_files.empty());

    int                      passed = 0;
    int                      failed = 0;
    std::vector<std::string> failures;
    for (const auto& path : yaml_files)
    {
        auto result = round_trip(path.string());
        if (result.ok())
        {
            ++passed;
        }
        else
        {
            ++failed;
            std::ostringstream oss;
            oss << path.filename().string() << ":\n";
            for (const auto& e : result.errors)
            {
                oss << "    " << e.file << ":" << e.line << " " << e.key << ": " << e.message << "\n";
            }
            failures.push_back(oss.str());
        }
    }

    if (failed > 0)
    {
        std::ostringstream oss;
        oss << failed << " of " << yaml_files.size() << " configs failed round-trip:\n";
        for (const auto& f : failures)
            oss << f;
        FAIL(oss.str());
    }
    REQUIRE(passed == static_cast<int>(yaml_files.size()));
}

// ─────────────────────────────────────────────────────────────
// Targeted: dynamics scheme decomposition
// ─────────────────────────────────────────────────────────────

TEST_CASE("migrate: cartesian scheme decomposes to non-axisym + cart + agrid", "[init][config][v0_to_v1]")
{
    TempYaml tmp(R"yaml(
duration_s: 100
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 2000
  dy: 2000
  dz: 500
  dt: 2
dynamics:
  scheme: cartesian
microphysics:
  scheme: kessler
output:
  write_every_s: 30
)yaml");

    auto m = migrate_v0_to_v1(tmp.path.string());
    REQUIRE(m.ok());

    const auto& c = *m.config;
    REQUIRE(c.dynamics.axisymmetric == false);
    REQUIRE(c.grid.coordinate_system == CoordinateSystem::Cartesian);
    REQUIRE(c.grid.staggering == StaggerType::Collocated);
    REQUIRE(c.microphysics.scheme == MicrophysicsScheme::Kessler);
}

TEST_CASE("migrate: diagnostics and performance metadata are preserved", "[init][config][v0_to_v1][performance]")
{
    TempYaml tmp(R"yaml(
duration_s: 100
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 2000
  dy: 2000
  dz: 500
  dt: 2
dynamics:
  scheme: cartesian
output:
  write_every_s: 0
hardware:
  memory_admission: reject
diagnostics:
  scientific_every_steps: 0
  deterministic_reductions: true
performance:
  timing: true
  report_every_steps: 25
  warmup_steps: 10
)yaml");

    const auto migration = migrate_v0_to_v1(tmp.path.string());
    INFO(format_warnings(tmp.path.string(), migration.warnings));
    REQUIRE(migration.ok());
    REQUIRE(migration.config->extensions.diagnostics.scientific_every_steps == 0);
    REQUIRE(migration.config->extensions.diagnostics.deterministic_reductions);
    REQUIRE(migration.config->extensions.compute.memory_admission == MemoryAdmissionPolicy::Reject);
    REQUIRE(migration.config->extensions.performance.timing == true);
    REQUIRE(migration.config->extensions.performance.report_every_steps == 25);
    REQUIRE(migration.config->extensions.performance.warmup_steps == 10);

    const std::string yaml = serialize_v1_to_yaml(*migration.config);
    REQUIRE(yaml.find("scientific_every_steps: 0") != std::string::npos);
    REQUIRE(yaml.find("deterministic_reductions: true") != std::string::npos);
    REQUIRE(yaml.find("memory_admission: reject") != std::string::npos);
    REQUIRE(yaml.find("timing: true") != std::string::npos);
    REQUIRE(yaml.find("report_every_steps: 25") != std::string::npos);
    REQUIRE(yaml.find("warmup_steps: 10") != std::string::npos);

    const auto flat = serialize_v1_to_v0_map(*migration.config);
    REQUIRE(flat.at("diagnostics.scientific_every_steps") == "0");
    REQUIRE(flat.at("diagnostics.deterministic_reductions") == "true");
    REQUIRE(flat.at("hardware.memory_admission") == "reject");
    REQUIRE(flat.at("performance.timing") == "true");
    REQUIRE(flat.at("performance.report_every_steps") == "25");
    REQUIRE(flat.at("performance.warmup_steps") == "10");
}

TEST_CASE("migrate: fractional runtime seconds fail closed", "[init][config][v0_to_v1][runtime]")
{
    TempYaml tmp(R"yaml(
duration_s: 100.5
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 2000
  dy: 2000
  dz: 500
  dt: 2
dynamics:
  scheme: cartesian
output:
  write_every_s: 10.25
)yaml");

    const auto migration = migrate_v0_to_v1(tmp.path.string());
    REQUIRE_FALSE(migration.ok());

    bool duration_rejected = false;
    bool output_rejected   = false;
    for (const auto& error : migration.errors)
    {
        if (error.key == "duration_s")
        {
            duration_rejected = error.message.find("whole number of seconds") != std::string::npos;
        }
        if (error.key == "output.write_every_s")
        {
            output_rejected = error.message.find("whole number of seconds") != std::string::npos;
        }
    }
    REQUIRE(duration_rejected);
    REQUIRE(output_rejected);
}

TEST_CASE("serialize_v1_to_v0_map preserves absent performance overrides", "[init][config][v0_to_v1][performance]")
{
    ConfigV1   config;
    const auto flat = serialize_v1_to_v0_map(config);
    REQUIRE(flat.count("performance.timing") == 0);
    REQUIRE(flat.count("performance.report_every_steps") == 0);
    REQUIRE(flat.count("performance.warmup_steps") == 0);
    REQUIRE(flat.count("diagnostics.scientific_every_steps") == 0);
    REQUIRE(flat.count("diagnostics.deterministic_reductions") == 0);
}

TEST_CASE("domain-motion intent is projected exactly or rejected", "[init][config][v0_to_v1][domain_motion]")
{
    ConfigV1 config;

    SECTION("constant translation projects without loss")
    {
        config.domain_motion.mode                = DomainMotionMode::ConstantVelocity;
        config.domain_motion.frame_velocity.u_ms = 12.0;
        config.domain_motion.frame_velocity.v_ms = -4.0;
        const auto flat                          = serialize_v1_to_v0_map(config);
        REQUIRE(flat.at("domain_motion.mode") == "constant_velocity");
        REQUIRE(flat.at("domain_motion.frame_velocity.u_ms") == "12");
        REQUIRE(flat.at("domain_motion.frame_velocity.v_ms") == "-4");
        const auto yaml = serialize_v1_to_yaml(config);
        REQUIRE(yaml.find("mode: constant_velocity") != std::string::npos);
        REQUIRE(yaml.find("u_ms: 12") != std::string::npos);
        REQUIRE(yaml.find("v_ms: -4") != std::string::npos);
    }

    SECTION("one-way nesting relationship controls project without loss")
    {
        config.domain_motion.nesting.enabled                       = true;
        config.domain_motion.nesting.parent_ratio_x                = 4;
        config.domain_motion.nesting.parent_ratio_y                = 2;
        config.domain_motion.nesting.parent_time_step_ratio        = 5;
        config.domain_motion.nesting.child_origin_parent_cell_i    = 3;
        config.domain_motion.nesting.child_origin_parent_cell_j    = 2;
        config.domain_motion.nesting.child_cells_x                 = 16;
        config.domain_motion.nesting.child_cells_y                 = 12;
        config.domain_motion.nesting.relaxation_width_child_cells  = 2;
        config.domain_motion.nesting.boundary_stencil_parent_cells = 1;
        config.domain_motion.nesting.relaxation_e_folding_time_s   = 4.0;
        config.domain_motion.nesting.feedback                      = NestingFeedback::OneWay;
        const auto flat                                            = serialize_v1_to_v0_map(config);
        REQUIRE(flat.at("domain_motion.nesting.enabled") == "true");
        REQUIRE(flat.at("domain_motion.nesting.parent_ratio_x") == "4");
        REQUIRE(flat.at("domain_motion.nesting.parent_ratio_y") == "2");
        REQUIRE(flat.at("domain_motion.nesting.parent_time_step_ratio") == "5");
        REQUIRE(flat.at("domain_motion.nesting.child_origin_parent_cell_i") == "3");
        REQUIRE(flat.at("domain_motion.nesting.child_origin_parent_cell_j") == "2");
        REQUIRE(flat.at("domain_motion.nesting.child_cells_x") == "16");
        REQUIRE(flat.at("domain_motion.nesting.child_cells_y") == "12");
        REQUIRE(flat.at("domain_motion.nesting.relaxation_width_child_cells") == "2");
        REQUIRE(flat.at("domain_motion.nesting.boundary_stencil_parent_cells") == "1");
        REQUIRE(flat.at("domain_motion.nesting.relaxation_e_folding_time_s") == "4");
        REQUIRE(flat.at("domain_motion.nesting.feedback") == "one_way");
        const auto yaml = serialize_v1_to_yaml(config);
        REQUIRE(yaml.find("parent_ratio_x: 4") != std::string::npos);
        REQUIRE(yaml.find("parent_ratio_y: 2") != std::string::npos);
        REQUIRE(yaml.find("parent_time_step_ratio: 5") != std::string::npos);
        REQUIRE(yaml.find("child_cells_x: 16") != std::string::npos);
        REQUIRE(yaml.find("relaxation_e_folding_time_s: 4") != std::string::npos);
        REQUIRE(yaml.find("feedback: one_way") != std::string::npos);
    }

    SECTION("nesting time ratio cannot weaken the horizontal CFL floor")
    {
        config.domain_motion.nesting.parent_ratio_x         = 4;
        config.domain_motion.nesting.parent_time_step_ratio = 3;
        REQUIRE_THROWS_WITH(serialize_v1_to_v0_map(config),
                            Catch::Matchers::Contains("at least the largest horizontal parent ratio"));
        REQUIRE_THROWS_WITH(serialize_v1_to_yaml(config),
                            Catch::Matchers::Contains("at least the largest horizontal parent ratio"));
    }

    SECTION("adaptive moving-window mode projects its discrete recycling authority")
    {
        config.domain_motion.mode                         = DomainMotionMode::TrackedWindow;
        config.domain_motion.tracking.enabled             = true;
        config.domain_motion.tracking.recycle_domain      = true;
        config.domain_motion.tracking.inner_margin_m      = 5000.0;
        config.domain_motion.tracking.maximum_shift_cells = 8;
        const auto flat = serialize_v1_to_v0_map(config);
        REQUIRE(flat.at("domain_motion.mode") == "tracked_window");
        REQUIRE(flat.at("domain_motion.tracking.recycle_domain") == "true");
        REQUIRE(serialize_v1_to_yaml(config).find("mode: tracked_window")
                != std::string::npos);
    }

    SECTION("tracked window cannot omit recycling authority")
    {
        config.domain_motion.mode = DomainMotionMode::TrackedWindow;
        REQUIRE_THROWS_WITH(serialize_v1_to_v0_map(config),
                            Catch::Matchers::Contains("requires enabled tracking"));
    }

    SECTION("hidden velocity in stationary mode")
    {
        config.domain_motion.frame_velocity.v_ms = -4.0;
        REQUIRE_THROWS_AS(serialize_v1_to_v0_map(config), std::invalid_argument);
        REQUIRE_THROWS_AS(serialize_v1_to_yaml(config), std::invalid_argument);
    }

    SECTION("connected lifetime enforcement is projected without loss")
    {
        config.domain_motion.lifetime_planning.enabled              = true;
        config.domain_motion.lifetime_planning.motion_source        = StormMotionSource::Explicit;
        config.domain_motion.lifetime_planning.explicit_u_ms        = 12.0;
        config.domain_motion.lifetime_planning.explicit_v_ms        = -4.0;
        config.domain_motion.lifetime_planning.upstream_buffer_m    = 10000.0;
        config.domain_motion.lifetime_planning.downstream_buffer_m  = 20000.0;
        config.domain_motion.lifetime_planning.cross_track_buffer_m = 15000.0;
        const auto flat                                             = serialize_v1_to_v0_map(config);
        REQUIRE(flat.at("domain_motion.lifetime_planning.enabled") == "true");
        REQUIRE(flat.at("domain_motion.lifetime_planning.motion_source") == "explicit");
        REQUIRE(flat.at("domain_motion.lifetime_planning.explicit_u_ms") == "12");
        REQUIRE(serialize_v1_to_yaml(config).find("motion_source: explicit") != std::string::npos);
    }

    SECTION("qualified lifetime evidence cannot be warning-only")
    {
        config.extensions.execution.mode                            = ExecutionMode::Qualified;
        config.domain_motion.lifetime_planning.enabled              = true;
        config.domain_motion.lifetime_planning.action               = LifetimePlanningAction::Warn;
        config.domain_motion.lifetime_planning.upstream_buffer_m    = 10000.0;
        config.domain_motion.lifetime_planning.downstream_buffer_m  = 20000.0;
        config.domain_motion.lifetime_planning.cross_track_buffer_m = 15000.0;
        REQUIRE_THROWS_WITH(serialize_v1_to_v0_map(config), Catch::Matchers::Contains("action=error"));
        REQUIRE_THROWS_WITH(serialize_v1_to_yaml(config), Catch::Matchers::Contains("action=error"));
    }

    SECTION("diagnostic tracking projects without loss")
    {
        config.domain_motion.tracking.enabled                         = true;
        config.domain_motion.tracking.inner_margin_m                  = 12000.0;
        config.domain_motion.tracking.minimum_updraft_ms              = 7.0;
        config.domain_motion.tracking.minimum_condensate_path_kgm2    = 0.25;
        config.domain_motion.tracking.minimum_component_cells         = 9;
        config.domain_motion.tracking.minimum_core_area_m2            = 2.5e6;
        config.domain_motion.tracking.maximum_motion_speed_ms         = 70.0;
        config.domain_motion.tracking.maximum_motion_acceleration_ms2 = 0.15;
        config.domain_motion.tracking.minimum_continuity_samples      = 4;
        config.domain_motion.tracking.lost_after_samples              = 6;
        const auto flat                                               = serialize_v1_to_v0_map(config);
        REQUIRE(flat.at("domain_motion.tracking.enabled") == "true");
        REQUIRE(flat.at("domain_motion.tracking.recycle_domain") == "false");
        REQUIRE(flat.at("domain_motion.tracking.inner_margin_m") == "12000");
        REQUIRE(flat.at("domain_motion.tracking.minimum_updraft_ms") == "7");
        REQUIRE(flat.at("domain_motion.tracking.minimum_condensate_path_kgm2") == "0.25");
        REQUIRE(flat.at("domain_motion.tracking.minimum_component_cells") == "9");
        REQUIRE(flat.at("domain_motion.tracking.minimum_core_area_m2") == "2500000");
        REQUIRE(flat.at("domain_motion.tracking.maximum_motion_speed_ms") == "70");
        REQUIRE(flat.at("domain_motion.tracking.maximum_motion_acceleration_ms2") == "0.15");
        REQUIRE(flat.at("domain_motion.tracking.minimum_continuity_samples") == "4");
        REQUIRE(flat.at("domain_motion.tracking.lost_after_samples") == "6");
        REQUIRE(serialize_v1_to_yaml(config).find("minimum_component_cells: 9") != std::string::npos);
        REQUIRE(serialize_v1_to_yaml(config).find("minimum_core_area_m2: 2500000") != std::string::npos);
        REQUIRE(serialize_v1_to_yaml(config).find("minimum_continuity_samples: 4") != std::string::npos);
    }

    SECTION("live recycling authority projects explicitly")
    {
        config.domain_motion.tracking.enabled             = true;
        config.domain_motion.tracking.recycle_domain      = true;
        config.domain_motion.tracking.inner_margin_m      = 12000.0;
        config.domain_motion.tracking.maximum_shift_cells = 8;
        const auto flat                                   = serialize_v1_to_v0_map(config);
        REQUIRE(flat.at("domain_motion.tracking.recycle_domain") == "true");
        REQUIRE(serialize_v1_to_yaml(config).find("recycle_domain: true") != std::string::npos);
    }

    SECTION("live recycling authority requires a nonzero shift policy")
    {
        config.domain_motion.tracking.enabled        = true;
        config.domain_motion.tracking.recycle_domain = true;
        config.domain_motion.tracking.inner_margin_m = 12000.0;
        REQUIRE_THROWS_WITH(serialize_v1_to_v0_map(config), Catch::Matchers::Contains("maximum_shift_cells > 0"));
    }

    SECTION("diagnostic tracking cannot claim qualification")
    {
        config.extensions.execution.mode             = ExecutionMode::Qualified;
        config.domain_motion.tracking.enabled        = true;
        config.domain_motion.tracking.inner_margin_m = 12000.0;
        REQUIRE_THROWS_WITH(serialize_v1_to_v0_map(config), Catch::Matchers::Contains("not reference-qualified"));
    }
}

TEST_CASE("serialize: SCI-021 values are either exact at runtime or rejected", "[init][config][v0_to_v1][sci021]")
{
    SECTION("2d turbulence projects to the runtime horizontal-only mode")
    {
        ConfigV1 config;
        config.turbulence.mode          = TurbulenceMode::TwoD;
        config.diffusion.scheme         = DiffusionScheme::Explicit;
        config.diffusion.operator_order = 2;

        const auto flat = serialize_v1_to_v0_map(config);
        REQUIRE(flat.at("turbulence.mode") == "horizontal_only");
        REQUIRE(flat.at("numerics.diffusion.scheme") == "explicit");
        REQUIRE(flat.at("numerics.diffusion.operator_order") == "2");
    }

    SECTION("programmatic none selection cannot bypass strict loading")
    {
        ConfigV1 config;
        config.diffusion.scheme = DiffusionScheme::None;
        REQUIRE_THROWS(serialize_v1_to_v0_map(config));
        REQUIRE_THROWS(serialize_v1_to_yaml(config));
    }

    SECTION("implicit diffusion projects to the registered runtime factory id")
    {
        ConfigV1 config;
        config.diffusion.scheme         = DiffusionScheme::Implicit;
        config.diffusion.operator_order = 2;
        const auto flat                 = serialize_v1_to_v0_map(config);
        REQUIRE(flat.at("numerics.diffusion.scheme") == "implicit");
    }

    SECTION("programmatic sixth-order selection cannot be dropped")
    {
        ConfigV1 config;
        config.diffusion.operator_order = 6;
        REQUIRE_THROWS(serialize_v1_to_v0_map(config));
        REQUIRE_THROWS(serialize_v1_to_yaml(config));
    }

    SECTION("programmatic implicit damping stays exact in every projection")
    {
        ConfigV1 config;
        config.damping.mode = DampingMode::Implicit;
        REQUIRE(serialize_v1_to_v0_map(config).at("damping.mode")
                == "implicit");
        REQUIRE(serialize_v1_to_yaml(config).find("mode: implicit")
                != std::string::npos);
    }

    SECTION("programmatic beta-plane remains exact in every projection")
    {
        ConfigV1 config;
        config.coriolis.mode = CoriolisMode::BetaPlane;
        REQUIRE(serialize_v1_to_v0_map(config).at("coriolis.mode")
                == "beta_plane");
        REQUIRE(serialize_v1_to_yaml(config).find("mode: beta_plane")
                != std::string::npos);

        config.domain_motion.nesting.enabled = true;
        config.domain_motion.nesting.child_cells_x = 6;
        config.domain_motion.nesting.child_cells_y = 6;
        config.domain_motion.nesting.relaxation_width_child_cells = 1;
        REQUIRE(serialize_v1_to_v0_map(config).at("coriolis.mode")
                == "beta_plane");
        REQUIRE(serialize_v1_to_yaml(config).find("mode: beta_plane")
                != std::string::npos);
    }

    SECTION("programmatic NetCDF cannot fall back to NPY")
    {
        ConfigV1 config;
        config.output.format = OutputFormat::Netcdf;
#ifdef HAVE_NETCDF
        REQUIRE(serialize_v1_to_v0_map(config).at("output.format") == "netcdf");
        REQUIRE(serialize_v1_to_yaml(config).find("format: netcdf") != std::string::npos);
#else
        REQUIRE_THROWS_AS(serialize_v1_to_v0_map(config), std::invalid_argument);
        REQUIRE_THROWS_AS(serialize_v1_to_yaml(config), std::invalid_argument);
#endif
    }
}

TEST_CASE("migrate: NetCDF output follows exact-binary capability", "[init][config][v0_to_v1][sci021][output]")
{
    TempYaml   tmp(R"yaml(
grid:
  nx: 8
  ny: 8
  nz: 8
  dr: 100
  dtheta: 100
  dz: 100
  dt: 1
duration_s: 10
output:
  format: netcdf
)yaml");
    const auto result = migrate_v0_to_v1(tmp.path.string());
#ifdef HAVE_NETCDF
    REQUIRE(result.ok());
    REQUIRE(result.config.has_value());
    REQUIRE(result.config->output.format == OutputFormat::Netcdf);
#else
    REQUIRE_FALSE(result.ok());
    REQUIRE(std::any_of(
        result.errors.begin(),
        result.errors.end(),
        [](const LoadError& error)
        { return error.key == "output.format" && error.message.find("unavailable") != std::string::npos; }));
#endif
}

TEST_CASE("SCI-026: sounding projection emits only the selected source branch",
          "[init][config][v0_to_v1][sounding][sci026]")
{
    SECTION("parametric CAPE projects a tropopause, never an EL target")
    {
        ConfigV1 config;
        auto&    sounding         = config.initial_conditions.sounding;
        sounding.type             = SoundingType::ParametricCAPE;
        sounding.cape_jkg         = 2800.0;
        sounding.tropopause_m     = 11500.0;
        sounding.surface_theta_k  = 302.0;
        sounding.moisture_qv_kgkg = 0.013;

        const auto flat = serialize_v1_to_v0_map(config);
        REQUIRE(flat.at("environment.sounding.type") == "parametric_cape");
        REQUIRE(flat.at("environment.cape_target_jkg") == "2800");
        REQUIRE(flat.at("environment.tropopause_z_m") == "11500");
        REQUIRE(flat.at("environment.sfc_theta_k") == "302");
        REQUIRE(flat.at("environment.sfc_qv_kgkg") == "0.013");
        REQUIRE(flat.count("environment.sounding.targets.el_m") == 0);
        REQUIRE(flat.count("environment.sounding.targets.cin_jkg") == 0);
        REQUIRE(flat.count("environment.sounding.file_path") == 0);

        const std::string yaml = serialize_v1_to_yaml(config);
        REQUIRE(yaml.find("    type: parametric_cape") != std::string::npos);
        REQUIRE(yaml.find("    tropopause_m: 11500") != std::string::npos);
        REQUIRE(yaml.find("    el_m:") == std::string::npos);
        REQUIRE(yaml.find("    cin_jkg:") == std::string::npos);
    }

    SECTION("parametric targets with LCL project the complete target branch")
    {
        ConfigV1 config;
        auto&    sounding        = config.initial_conditions.sounding;
        sounding.type            = SoundingType::ParametricTargets;
        sounding.cape_jkg        = 3200.0;
        sounding.cin_jkg         = 80.0;
        sounding.lcl_m           = 1000.0;
        sounding.lfc_m           = 1700.0;
        sounding.el_m            = 12500.0;
        sounding.surface_theta_k = 302.0;
        sounding.moisture_qv_kgkg.reset();

        const auto flat = serialize_v1_to_v0_map(config);
        REQUIRE(flat.at("environment.sounding.type") == "parametric_targets");
        REQUIRE(flat.at("environment.sounding.targets.cape_jkg") == "3200");
        REQUIRE(flat.at("environment.sounding.targets.cin_jkg") == "80");
        REQUIRE(flat.at("environment.sounding.targets.lcl_m") == "1000");
        REQUIRE(flat.at("environment.sounding.targets.lfc_m") == "1700");
        REQUIRE(flat.at("environment.sounding.targets.el_m") == "12500");
        REQUIRE(flat.at("environment.sounding.targets.surface_theta_k") == "302");
        REQUIRE(flat.count("environment.sounding.targets.moisture_qv_kgkg") == 0);
        REQUIRE(flat.count("environment.tropopause_z_m") == 0);
        REQUIRE(flat.count("environment.cape_target_jkg") == 0);
        REQUIRE(flat.count("environment.sounding.file_path") == 0);

        const std::string yaml = serialize_v1_to_yaml(config);
        REQUIRE(yaml.find("    type: parametric_targets") != std::string::npos);
        REQUIRE(yaml.find("    lcl_m: 1000") != std::string::npos);
        REQUIRE(yaml.find("    tropopause_m:") == std::string::npos);
        REQUIRE(yaml.find("    moisture_qv_kgkg:") == std::string::npos);
    }

    SECTION("parametric target qv override replaces rather than shadows LCL")
    {
        ConfigV1 config;
        auto&    sounding         = config.initial_conditions.sounding;
        sounding.type             = SoundingType::ParametricTargets;
        sounding.moisture_qv_kgkg = 0.012;

        const auto flat = serialize_v1_to_v0_map(config);
        REQUIRE(flat.at("environment.sounding.targets.moisture_qv_kgkg") == "0.012");
        REQUIRE(flat.count("environment.sounding.targets.lcl_m") == 0);

        const std::string yaml = serialize_v1_to_yaml(config);
        REQUIRE(yaml.find("    moisture_qv_kgkg: 0.012") != std::string::npos);
        REQUIRE(yaml.find("    lcl_m:") == std::string::npos);
    }

    SECTION("file projection contains no procedural sounding values")
    {
        ConfigV1 config;
        auto&    sounding  = config.initial_conditions.sounding;
        sounding.type      = SoundingType::File;
        sounding.file_path = "profile.txt";

        const auto flat = serialize_v1_to_v0_map(config);
        REQUIRE(flat.at("environment.sounding.type") == "file");
        REQUIRE(flat.at("environment.sounding.file_path") == "profile.txt");
        REQUIRE(flat.count("environment.sounding.targets.cape_jkg") == 0);
        REQUIRE(flat.count("environment.cape_target_jkg") == 0);
        REQUIRE(flat.count("environment.sfc_theta_k") == 0);
        REQUIRE(flat.count("environment.tropopause_z_m") == 0);

        const std::string yaml = serialize_v1_to_yaml(config);
        REQUIRE(yaml.find("    type: file") != std::string::npos);
        REQUIRE(yaml.find("    file_path: profile.txt") != std::string::npos);
        REQUIRE(yaml.find("    cape_jkg:") == std::string::npos);
        REQUIRE(yaml.find("    surface_theta_k:") == std::string::npos);
    }
}

TEST_CASE("strict custom hodograph projects to the explicit anchor source", "[init][config][v0_to_v1][hodograph]")
{
    ConfigV1   automatic;
    const auto automatic_flat = serialize_v1_to_v0_map(automatic);
    REQUIRE(automatic.initial_conditions.hodograph.type == HodographType::Auto);
    REQUIRE(automatic_flat.at("environment.hodograph.type") == "auto");
    REQUIRE(serialize_v1_to_yaml(automatic).find("    type: auto") != std::string::npos);

    ConfigV1 config;
    auto&    hodograph = config.initial_conditions.hodograph;
    hodograph.type     = HodographType::Custom;
    hodograph.u_sfc_ms = 4.0;
    hodograph.v_sfc_ms = -1.0;
    hodograph.u_1km_ms = 10.0;
    hodograph.v_1km_ms = 3.0;
    hodograph.u_6km_ms = 24.0;
    hodograph.v_6km_ms = 8.0;

    const auto flat = serialize_v1_to_v0_map(config);
    REQUIRE(flat.at("environment.hodograph.type") == "wk_param");
    REQUIRE(flat.at("environment.hodograph.u_sfc_ms") == "4");
    REQUIRE(flat.at("environment.hodograph.v_sfc_ms") == "-1");
    REQUIRE(flat.at("environment.hodograph.u_6km_ms") == "24");
    REQUIRE(flat.at("environment.hodograph.v_6km_ms") == "8");

    // The public v1 representation remains source-oriented and round-trips
    // as "custom"; only the private runtime bridge uses "wk_param".
    const std::string yaml = serialize_v1_to_yaml(config);
    REQUIRE(yaml.find("    type: custom") != std::string::npos);
}

TEST_CASE("SCI-026: programmatic sounding configs cannot bypass clamp guards",
          "[init][config][v0_to_v1][sounding][sci026]")
{
    ConfigV1 config;

    SECTION("CAPE qv floor")
    {
        config.initial_conditions.sounding.moisture_qv_kgkg = 0.001;
        REQUIRE_THROWS_AS(serialize_v1_to_v0_map(config), std::invalid_argument);
        REQUIRE_THROWS_AS(serialize_v1_to_yaml(config), std::invalid_argument);
    }

    SECTION("target CIN saturation")
    {
        config.initial_conditions.sounding.type    = SoundingType::ParametricTargets;
        config.initial_conditions.sounding.cin_jkg = 301.0;
        REQUIRE_THROWS_AS(serialize_v1_to_v0_map(config), std::invalid_argument);
        REQUIRE_THROWS_AS(serialize_v1_to_yaml(config), std::invalid_argument);
    }

    SECTION("target LCL cannot translate through the qv floor")
    {
        config.initial_conditions.sounding.type  = SoundingType::ParametricTargets;
        config.initial_conditions.sounding.lcl_m = 50000.0;
        REQUIRE_THROWS_AS(serialize_v1_to_v0_map(config), std::invalid_argument);
        REQUIRE_THROWS_AS(serialize_v1_to_yaml(config), std::invalid_argument);
    }

    SECTION("file path is required")
    {
        config.initial_conditions.sounding.type = SoundingType::File;
        REQUIRE_THROWS_AS(serialize_v1_to_v0_map(config), std::invalid_argument);
        REQUIRE_THROWS_AS(serialize_v1_to_yaml(config), std::invalid_argument);
    }
}

TEST_CASE("migrate: SCI-021 legacy diffusion choices fail closed", "[init][config][v0_to_v1][sci021]")
{
    const std::string prefix = R"yaml(
duration_s: 100
grid:
  nx: 16
  ny: 16
  nz: 16
  dx: 500
  dy: 500
  dz: 250
  dt: 1
dynamics:
  scheme: cartesian
output:
  write_every_s: 0
)yaml";

    SECTION("none is rejected during migration")
    {
        TempYaml   tmp(prefix + R"yaml(
numerics:
  diffusion:
    scheme: none
    operator_order: 2
)yaml");
        const auto migration = migrate_v0_to_v1(tmp.path.string());
        REQUIRE_FALSE(migration.ok());
        REQUIRE(std::any_of(migration.errors.begin(),
                            migration.errors.end(),
                            [](const LoadError& error)
                            {
                                return error.key == "numerics.diffusion.scheme" &&
                                       error.message.find("no registered runtime") != std::string::npos;
                            }));
    }

    SECTION("sixth order is rejected during migration")
    {
        TempYaml   tmp(prefix + R"yaml(
numerics:
  diffusion:
    scheme: explicit
    operator_order: 6
)yaml");
        const auto migration = migrate_v0_to_v1(tmp.path.string());
        REQUIRE_FALSE(migration.ok());
        REQUIRE(std::any_of(migration.errors.begin(),
                            migration.errors.end(),
                            [](const LoadError& error)
                            {
                                return error.key == "numerics.diffusion.operator_order" &&
                                       error.message.find("runtime order 2") != std::string::npos;
                            }));
    }

    SECTION("unwired variable coefficients are rejected during migration")
    {
        TempYaml   tmp(prefix + R"yaml(
numerics:
  diffusion:
    scheme: implicit
    operator_order: 2
    use_variable_K: true
)yaml");
        const auto migration = migrate_v0_to_v1(tmp.path.string());
        REQUIRE_FALSE(migration.ok());
        REQUIRE(std::any_of(migration.errors.begin(),
                            migration.errors.end(),
                            [](const LoadError& error)
                            {
                                return error.key == "numerics.diffusion.use_variable_K" &&
                                       error.message.find("unavailable") != std::string::npos;
                            }));
    }

    SECTION("2d turbulence retains horizontal-only runtime semantics")
    {
        TempYaml   tmp(prefix + R"yaml(
turbulence:
  scheme: smagorinsky
  mode: 2d
numerics:
  diffusion:
    scheme: explicit
    operator_order: 2
)yaml");
        const auto migration = migrate_v0_to_v1(tmp.path.string());
        INFO(format_warnings(tmp.path.string(), migration.warnings));
        REQUIRE(migration.ok());
        REQUIRE(migration.config->turbulence.mode == TurbulenceMode::TwoD);
        REQUIRE(migration.config->diffusion.operator_order == 2);
        const auto flat = serialize_v1_to_v0_map(*migration.config);
        REQUIRE(flat.at("turbulence.mode") == "horizontal_only");
        REQUIRE(flat.at("numerics.diffusion.operator_order") == "2");
    }
}

TEST_CASE("migrate: Cartesian canonical dynamics schemes infer their staggering", "[init][config][v0_to_v1]")
{
    SECTION("canonical C-grid infers c_grid")
    {
        TempYaml tmp(R"yaml(
duration_s: 100
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 2000
  dy: 2000
  dz: 500
  dt: 2
dynamics:
  scheme: klemp_wilhelmson_split_cartesian_cgrid
microphysics:
  scheme: kessler
output:
  write_every_s: 30
)yaml");

        auto m = migrate_v0_to_v1(tmp.path.string());
        REQUIRE(m.ok());
        REQUIRE(m.config->dynamics.scheme == "klemp_wilhelmson_split_cartesian_cgrid");
        REQUIRE(m.config->dynamics.axisymmetric == false);
        REQUIRE(m.config->grid.coordinate_system == CoordinateSystem::Cartesian);
        REQUIRE(m.config->grid.staggering == StaggerType::CGrid);
    }

    SECTION("canonical A-grid infers collocated")
    {
        TempYaml tmp(R"yaml(
duration_s: 100
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 2000
  dy: 2000
  dz: 500
  dt: 2
dynamics:
  scheme: klemp_wilhelmson_split_cartesian_agrid
microphysics:
  scheme: kessler
output:
  write_every_s: 30
)yaml");

        auto m = migrate_v0_to_v1(tmp.path.string());
        REQUIRE(m.ok());
        REQUIRE(m.config->dynamics.scheme == "klemp_wilhelmson_split_cartesian_agrid");
        REQUIRE(m.config->dynamics.axisymmetric == false);
        REQUIRE(m.config->grid.coordinate_system == CoordinateSystem::Cartesian);
        REQUIRE(m.config->grid.staggering == StaggerType::Collocated);
    }
}

TEST_CASE("migrate: bare Klemp-Wilhelmson preserves scheme and derives from staggering", "[init][config][v0_to_v1]")
{
    TempYaml tmp(R"yaml(
duration_s: 100
grid:
  staggering: collocated
  nx: 64
  ny: 64
  nz: 32
  dx: 2000
  dy: 2000
  dz: 500
  dt: 2
dynamics:
  scheme: klemp_wilhelmson_split
microphysics:
  scheme: kessler
output:
  write_every_s: 30
)yaml");

    auto m = migrate_v0_to_v1(tmp.path.string());
    REQUIRE(m.ok());
    REQUIRE(m.config->dynamics.scheme == "klemp_wilhelmson_split");
    REQUIRE(m.config->dynamics.axisymmetric == false);
    REQUIRE(m.config->grid.coordinate_system == CoordinateSystem::Cartesian);
    REQUIRE(m.config->grid.staggering == StaggerType::Collocated);

    const auto kv = serialize_v1_to_v0_map(*m.config);
    REQUIRE(kv.at("dynamics.scheme") == "klemp_wilhelmson_split_cartesian_agrid");
}

TEST_CASE("serialize: bare Klemp-Wilhelmson bridge preserves both Cartesian paths", "[init][config][v0_to_v1]")
{
    SECTION("C-grid staggering emits the C-grid runtime id")
    {
        ConfigV1 c;
        c.dynamics.scheme = "klemp_wilhelmson_split";
        c.grid.staggering = StaggerType::CGrid;
        const auto kv     = serialize_v1_to_v0_map(c);
        REQUIRE(kv.at("dynamics.scheme") == "klemp_wilhelmson_split_cartesian_cgrid");
    }

    SECTION("collocated staggering emits the A-grid runtime id")
    {
        ConfigV1 c;
        c.dynamics.scheme = "klemp_wilhelmson_split";
        c.grid.staggering = StaggerType::Collocated;
        const auto kv     = serialize_v1_to_v0_map(c);
        REQUIRE(kv.at("dynamics.scheme") == "klemp_wilhelmson_split_cartesian_agrid");
    }
}

TEST_CASE("migrate: a v0 coordinate_system: cylindrical is rejected", "[init][config][v0_to_v1]")
{
    // Cylindrical coordinates are removed (Cartesian-only build). A legacy v0
    // config that still names `coordinate_system: cylindrical` must fail the
    // migration loudly rather than silently mapping to Cartesian.
    TempYaml tmp(R"yaml(
duration_s: 100
coordinate_system: cylindrical
grid:
  staggering: c_grid
  nx: 64
  ny: 64
  nz: 32
  dx: 250
  dy: 250
  dz: 250
  dt: 1
dynamics:
  scheme: cartesian
microphysics:
  scheme: none
output:
  write_every_s: 10
)yaml");

    auto m = migrate_v0_to_v1(tmp.path.string());
    REQUIRE_FALSE(m.ok());

    bool found = false;
    for (const auto& e : m.errors)
    {
        if (e.key == "coordinate_system" && e.message.find("unknown value 'cylindrical'") != std::string::npos)
        {
            found = true;
            break;
        }
    }
    REQUIRE(found);
}

// ─────────────────────────────────────────────────────────────
// Targeted: trigger km → m unit conversion
// ─────────────────────────────────────────────────────────────

TEST_CASE("migrate: bubble center km → m conversion", "[init][config][v0_to_v1]")
{
    TempYaml tmp(R"yaml(
duration_s: 100
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 2000
  dy: 2000
  dz: 500
  dt: 2
dynamics:
  scheme: cartesian
microphysics:
  scheme: none
trigger:
  bubble:
    dtheta_k: 2.5
    radius_km: 10.0
    center_x_km: 50.0
    center_y_km: 50.0
    center_z_km: 1.5
output:
  write_every_s: 30
)yaml");

    auto m = migrate_v0_to_v1(tmp.path.string());
    REQUIRE(m.ok());

    const auto& b = m.config->initial_conditions.trigger.bubble;
    REQUIRE(b.dtheta_k == Approx(2.5));
    REQUIRE(b.radius_km == Approx(10.0));
    REQUIRE(b.center_x_m == Approx(50000.0)); // 50 km → 50000 m
    REQUIRE(b.center_y_m == Approx(50000.0));
    REQUIRE(b.center_z_m == Approx(1500.0)); // 1.5 km → 1500 m

    // Trigger.type was implicit (bubble keys present, no type) — should
    // synthesize to Bubble per v0 default convention.
    REQUIRE(m.config->initial_conditions.trigger.type == TriggerType::Bubble);
}

// ─────────────────────────────────────────────────────────────
// Targeted: section renames (env → initial_conditions; terrain → grid)
// ─────────────────────────────────────────────────────────────

TEST_CASE("migrate: environment.* → initial_conditions.* renames", "[init][config][v0_to_v1]")
{
    TempYaml tmp(R"yaml(
duration_s: 100
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 250
  dy: 250
  dz: 250
  dt: 1
environment:
  sfc_theta_k: 302.0
  sfc_qv_kgkg: 0.013
  cape_target_jkg: 2800
  tropopause_z_m: 11500
  hodograph:
    type: custom
    u_sfc_ms: 3
    v_sfc_ms: 1
    u_1km_ms: 12
    v_1km_ms: 4
    u_6km_ms: 25
    v_6km_ms: 8
dynamics:
  scheme: cartesian
microphysics:
  scheme: none
output:
  write_every_s: 30
)yaml");

    auto m = migrate_v0_to_v1(tmp.path.string());
    REQUIRE(m.ok());

    const auto& s = m.config->initial_conditions.sounding;
    REQUIRE(s.surface_theta_k == Approx(302.0));
    REQUIRE(s.moisture_qv_kgkg.has_value());
    REQUIRE(*s.moisture_qv_kgkg == Approx(0.013));
    REQUIRE(s.cape_jkg == Approx(2800.0));
    REQUIRE(s.tropopause_m == Approx(11500.0));

    const auto& h = m.config->initial_conditions.hodograph;
    REQUIRE(h.u_sfc_ms == Approx(3.0));
    REQUIRE(h.u_6km_ms == Approx(25.0));
}

TEST_CASE("migrate: terrain.ztop → grid.ztop cross-section move", "[init][config][v0_to_v1]")
{
    TempYaml tmp(R"yaml(
duration_s: 100
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 250
  dy: 250
  dz: 250
  dt: 1
dynamics:
  scheme: cartesian
microphysics:
  scheme: none
terrain:
  scheme: flat
  ztop: 14000
output:
  write_every_s: 30
)yaml");

    auto m = migrate_v0_to_v1(tmp.path.string());
    REQUIRE(m.ok());
    REQUIRE(m.config->grid.ztop == Approx(14000.0));
    REQUIRE(m.config->terrain.scheme == TerrainScheme::None);
}

TEST_CASE("SCI-094: legacy terrain selection survives migration, YAML, and runtime projection",
          "[init][config][v0_to_v1][terrain][config-truth][sci094]")
{
    TempYaml tmp(R"yaml(
duration_s: 100
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 250
  dy: 250
  dz: 250
  dt: 1
dynamics:
  scheme: cartesian
microphysics:
  scheme: none
terrain:
  scheme: schaer
  coord_id: smoothed
  ztop: 8000
  compute_derivatives: true
  compute_metrics: true
  smoothing_decay: 0.65
  schar:
    H: 900
    a: 5000
    ell: 4000
output:
  write_every_s: 30
)yaml");

    const auto migration = migrate_v0_to_v1(tmp.path.string());
    INFO(format_errors(migration.errors));
    REQUIRE(migration.ok());
    REQUIRE(migration.config->terrain.scheme == TerrainScheme::Schar);
    REQUIRE(migration.config->terrain.coordinate == TerrainCoordinate::Smoothed);
    REQUIRE(migration.config->terrain.smoothing_decay == Approx(0.65));
    REQUIRE(migration.config->terrain.schar.H == Approx(900.0));

    const auto projected = serialize_v1_to_v0_map(*migration.config);
    REQUIRE(projected.at("terrain.scheme") == "schar");
    REQUIRE(projected.at("terrain.coord_id") == "smoothed");
    REQUIRE(projected.at("terrain.schar.H") == "900");

    TempYaml   strict(serialize_v1_to_yaml(*migration.config));
    const auto reloaded = load_v1_config(strict.path.string());
    INFO(format_errors(reloaded.errors));
    REQUIRE(reloaded.ok());
    REQUIRE(reloaded.config->terrain.scheme == TerrainScheme::Schar);
    REQUIRE(reloaded.config->terrain.coordinate == TerrainCoordinate::Smoothed);
}

TEST_CASE("SCI-094: legacy migration rejects a nonphysical active terrain contract",
          "[init][config][v0_to_v1][terrain][config-truth][sci094]")
{
    TempYaml   tmp(R"yaml(
duration_s: 100
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 250
  dy: 250
  dz: 250
  dt: 1
dynamics:
  scheme: cartesian
terrain:
  scheme: bell
  coord_id: cartesian
  ztop: 9000
  compute_metrics: false
  bell:
    h0: 10000
output:
  write_every_s: 30
)yaml");
    const auto migration = migrate_v0_to_v1(tmp.path.string());
    REQUIRE_FALSE(migration.ok());
    const std::string errors = format_errors(migration.errors);
    REQUIRE(errors.find("active terrain requires terrain-following metrics") != std::string::npos);
    REQUIRE(errors.find("active terrain requires btf or smoothed coordinates") != std::string::npos);
    REQUIRE(errors.find("grid.nz * grid.dz") != std::string::npos);
    REQUIRE(errors.find("terrain height must remain below") != std::string::npos);
}

// ─────────────────────────────────────────────────────────────
// Targeted: output.fields → output.preset rename
// ─────────────────────────────────────────────────────────────

TEST_CASE("migrate: output.fields → output.preset rename", "[init][config][v0_to_v1]")
{
    TempYaml tmp(R"yaml(
duration_s: 100
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 250
  dy: 250
  dz: 250
  dt: 1
dynamics:
  scheme: cartesian
microphysics:
  scheme: none
output:
  write_every_s: 30
  fields: presentation
  format: npy_3d
)yaml");

    auto m = migrate_v0_to_v1(tmp.path.string());
    REQUIRE(m.ok());
    REQUIRE(m.config->output.preset == OutputPreset::Presentation);
    REQUIRE(m.config->output.format == OutputFormat::Npy3D);
}

// ─────────────────────────────────────────────────────────────
// Targeted: ZFP tuning + tiered cadence survive migration and the
// v1→v0 bridge (previously dropped, which made ZFP unreachable)
// ─────────────────────────────────────────────────────────────

TEST_CASE("migrate: ZFP tuning + tiered cadence preserved and bridged", "[init][config][v0_to_v1]")
{
    TempYaml tmp(R"yaml(
duration_s: 100
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 250
  dy: 250
  dz: 250
  dt: 1
dynamics:
  scheme: cartesian
microphysics:
  scheme: none
output:
  write_every_s: 30
  format: zfp_high
  zfp_tolerance: 0.0005
  zfp_keyframe_interval: 12
  zfp_predictive_delta: true
  zfp_sparse_threshold: 1.0e-6
  zfp_per_field_tolerances: true
  zfp_field_tolerance:
    rho: 1.0e-6
    qv: 1.0e-8
  tiered_write_cadence: true
  write_cadence_slow_s: 120
)yaml");

    auto m = migrate_v0_to_v1(tmp.path.string());
    REQUIRE(m.ok());

    // (T4) Keys survive migration into the typed v1 config.
    const auto& o = m.config->output;
    REQUIRE(o.format == OutputFormat::ZfpHigh);
    REQUIRE(o.zfp_tolerance == Approx(0.0005));
    REQUIRE(o.zfp_keyframe_interval == 12);
    REQUIRE(o.zfp_predictive_delta == true);
    REQUIRE(o.zfp_sparse_threshold == Approx(1.0e-6));
    REQUIRE(o.zfp_per_field_tolerances == true);
    REQUIRE(o.zfp_field_tolerances.at("rho") == Approx(1.0e-6));
    REQUIRE(o.zfp_field_tolerances.at("qv") == Approx(1.0e-8));
    REQUIRE(o.tiered_write_cadence == true);
    REQUIRE(o.write_cadence_slow_s == 120);

    // (T3) The v1→v0 bridge re-emits the exact keys parse_output_config
    // consumes on the writer side.
    auto kv = serialize_v1_to_v0_map(*m.config);
    REQUIRE(kv.at("output.format") == "zfp_high");
    REQUIRE(kv.count("output.zfp_tolerance") == 1);
    REQUIRE(kv.at("output.zfp_keyframe_interval") == "12");
    REQUIRE(kv.at("output.zfp_predictive_delta") == "true");
    REQUIRE(kv.at("output.zfp_field_tolerance.rho") == "1e-06");
    REQUIRE(kv.at("output.zfp_field_tolerance.qv") == "1e-08");
    REQUIRE(kv.at("output.tiered_write_cadence") == "true");
    REQUIRE(kv.at("output.write_cadence_slow_s") == "120");

    // The human-readable serializer keeps them too (config→yaml→config).
    auto yaml = serialize_v1_to_yaml(*m.config);
    REQUIRE(yaml.find("zfp_keyframe_interval: 12") != std::string::npos);
    REQUIRE(yaml.find("zfp_field_tolerance:") != std::string::npos);
    REQUIRE(yaml.find("    qv: 1e-08") != std::string::npos);
    REQUIRE(yaml.find("    rho: 1e-06") != std::string::npos);
    REQUIRE(yaml.find("tiered_write_cadence: true") != std::string::npos);

    TempYaml   roundtrip(yaml);
    const auto reloaded = load_v1_config(roundtrip.path.string());
    INFO(format_errors(reloaded.errors));
#ifdef HAVE_ZFP
    REQUIRE(reloaded.ok());
    REQUIRE(reloaded.config->output.zfp_field_tolerances.at("rho") == Approx(1.0e-6));
    REQUIRE(reloaded.config->output.zfp_field_tolerances.at("qv") == Approx(1.0e-8));
#else
    REQUIRE_FALSE(reloaded.ok());
    REQUIRE(has_load_error(reloaded, "output.format", "unavailable"));
#endif
}

// Non-ZFP, non-tiered output must NOT gain ZFP keys in the bridge map
// (keeps existing configs like beast_xl byte-identical through migration).
TEST_CASE("migrate: non-ZFP output omits ZFP keys from the v0 bridge map", "[init][config][v0_to_v1]")
{
    TempYaml tmp(R"yaml(
duration_s: 100
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 250
  dy: 250
  dz: 250
  dt: 1
dynamics:
  scheme: cartesian
microphysics:
  scheme: none
output:
  write_every_s: 30
  format: npy_3d
)yaml");

    auto m = migrate_v0_to_v1(tmp.path.string());
    REQUIRE(m.ok());
    auto kv = serialize_v1_to_v0_map(*m.config);
    REQUIRE(kv.at("output.format") == "npy_3d");
    REQUIRE(kv.count("output.zfp_tolerance") == 0);
    REQUIRE(kv.count("output.zfp_keyframe_interval") == 0);
    REQUIRE(kv.count("output.tiered_write_cadence") == 0);
}

// ─────────────────────────────────────────────────────────────
// Targeted: surface + PBL → single v0 boundary_layer.scheme slot.
// Regression for the PBL-selection bug: an explicit YSU/MYNN must NOT
// be silently overridden by surface:slab. The runtime has one slot and
// each scheme self-computes surface fluxes, so the PBL wins the slot and
// surface.scheme only gates apply_surface_fluxes.
// ─────────────────────────────────────────────────────────────

TEST_CASE("serialize: explicit PBL scheme wins the v0 boundary_layer slot", "[init][config][v0_to_v1][pbl]")
{
    SECTION("surface:slab + boundary_layer:slab -> slab mixed-layer")
    {
        ConfigV1 c;
        c.surface.scheme        = SurfaceScheme::Slab;
        c.surface.apply_fluxes  = true;
        c.boundary_layer.scheme = BoundaryLayerScheme::Slab;
        auto kv                 = serialize_v1_to_v0_map(c);
        REQUIRE(kv.at("surface.scheme") == "slab");
        REQUIRE(kv.at("boundary_layer.scheme") == "slab");
        REQUIRE(kv.at("boundary_layer.apply_surface_fluxes") == "true");
    }

    SECTION("surface:slab + boundary_layer:ysu -> ysu (not silently slab)")
    {
        ConfigV1 c;
        c.surface.scheme        = SurfaceScheme::Slab;
        c.surface.apply_fluxes  = true;
        c.boundary_layer.scheme = BoundaryLayerScheme::YSU;
        c.boundary_layer.cadence_s = 0.25;
        c.boundary_layer.clear_sky_only = false;
        auto kv                 = serialize_v1_to_v0_map(c);
        REQUIRE(kv.at("surface.scheme") == "slab");
        REQUIRE(kv.at("boundary_layer.scheme") == "ysu");
        REQUIRE(kv.at("boundary_layer.dt_pbl") == "0.25");
        REQUIRE(kv.at("boundary_layer.clear_sky_only") == "false");
        REQUIRE(kv.at("boundary_layer.apply_surface_fluxes") == "true");
    }

    SECTION("surface:slab + boundary_layer:mynn -> mynn")
    {
        ConfigV1 c;
        c.surface.scheme        = SurfaceScheme::Slab;
        c.surface.apply_fluxes  = true;
        c.boundary_layer.scheme = BoundaryLayerScheme::MYNN;
        auto kv                 = serialize_v1_to_v0_map(c);
        REQUIRE(kv.at("surface.scheme") == "slab");
        REQUIRE(kv.at("boundary_layer.scheme") == "mynn");
        REQUIRE(kv.at("boundary_layer.enable_tke") == "true");
        REQUIRE(kv.at("boundary_layer.apply_surface_fluxes") == "true");
    }

    SECTION("surface:none + boundary_layer:ysu -> ysu with fluxes OFF")
    {
        ConfigV1 c;
        c.surface.scheme        = SurfaceScheme::None;
        c.surface.apply_fluxes  = false;
        c.boundary_layer.scheme = BoundaryLayerScheme::YSU;
        auto kv                 = serialize_v1_to_v0_map(c);
        REQUIRE(kv.at("surface.scheme") == "none");
        REQUIRE(kv.at("boundary_layer.scheme") == "ysu");
        REQUIRE(kv.at("boundary_layer.enable_tke") == "false");
        REQUIRE(kv.at("boundary_layer.apply_surface_fluxes") == "false");
    }

    SECTION("surface:slab plus a disabled boundary layer is rejected")
    {
        ConfigV1 c;
        c.surface.scheme        = SurfaceScheme::Slab;
        c.surface.apply_fluxes  = true;
        c.boundary_layer.scheme = BoundaryLayerScheme::None;
        REQUIRE_THROWS_WITH(serialize_v1_to_v0_map(c), Catch::Contains("select boundary_layer.scheme=slab explicitly"));
    }

    SECTION("surface:none + boundary_layer:none -> none")
    {
        ConfigV1 c;
        c.surface.scheme        = SurfaceScheme::None;
        c.surface.apply_fluxes  = false;
        c.boundary_layer.scheme = BoundaryLayerScheme::None;
        auto kv                 = serialize_v1_to_v0_map(c);
        REQUIRE(kv.at("surface.scheme") == "none");
        REQUIRE(kv.at("boundary_layer.scheme") == "none");
        REQUIRE(kv.at("boundary_layer.apply_surface_fluxes") == "false");
    }

    SECTION("surface:slab + apply_fluxes:false + boundary_layer:ysu -> fluxes OFF")
    {
        ConfigV1 c;
        c.surface.scheme        = SurfaceScheme::Slab;
        c.surface.apply_fluxes  = false;
        c.boundary_layer.scheme = BoundaryLayerScheme::YSU;
        auto kv                 = serialize_v1_to_v0_map(c);
        REQUIRE(kv.at("surface.scheme") == "slab");
        REQUIRE(kv.at("boundary_layer.scheme") == "ysu");
        REQUIRE(kv.at("boundary_layer.apply_surface_fluxes") == "false");
    }
}

TEST_CASE("migrate: v0 slab becomes an explicit v1 mixed-layer selection", "[init][config][v0_to_v1][pbl]")
{
    TempYaml tmp(R"yaml(
duration_s: 100
grid:
  nx: 16
  ny: 16
  nz: 16
  dx: 500
  dy: 500
  dz: 250
  dt: 1
dynamics:
  scheme: cartesian
microphysics:
  scheme: none
boundary_layer:
  scheme: slab
  apply_surface_fluxes: true
output:
  write_every_s: 10
)yaml");

    auto m = migrate_v0_to_v1(tmp.path.string());
    INFO(format_errors(m.errors));
    REQUIRE(m.ok());
    REQUIRE(m.config->surface.scheme == SurfaceScheme::Slab);
    REQUIRE(m.config->surface.apply_fluxes == true);
    REQUIRE(m.config->boundary_layer.scheme == BoundaryLayerScheme::Slab);

    const auto yaml = serialize_v1_to_yaml(*m.config);
    REQUIRE(yaml.find("boundary_layer:\n  scheme: slab") != std::string::npos);
    const auto kv = serialize_v1_to_v0_map(*m.config);
    REQUIRE(kv.at("boundary_layer.scheme") == "slab");
}

TEST_CASE("migrate: PBL-top diagnosis survives v0 and v1 projection",
          "[init][config][v0_to_v1][pbl]")
{
    TempYaml tmp(R"yaml(
duration_s: 100
grid:
  nx: 16
  ny: 16
  nz: 16
  dx: 500
  dy: 500
  dz: 250
  dt: 1
dynamics:
  scheme: cartesian
microphysics:
  scheme: none
boundary_layer:
  scheme: mynn
  pbl_top_method: tke
output:
  write_every_s: 10
)yaml");

    const auto migrated = migrate_v0_to_v1(tmp.path.string());
    INFO(format_errors(migrated.errors));
    REQUIRE(migrated.ok());
    REQUIRE(migrated.config->boundary_layer.pbl_top_method ==
            BoundaryLayerTopMethod::TKE);

    const std::string yaml = serialize_v1_to_yaml(*migrated.config);
    REQUIRE(yaml.find("  pbl_top_method: tke\n") != std::string::npos);
    const auto projected = serialize_v1_to_v0_map(*migrated.config);
    REQUIRE(projected.at("boundary_layer.pbl_top_method") == "tke");
}

// ─────────────────────────────────────────────────────────────
// Targeted: prescribed surface (skin) state (CM1 tsk0). v1 surface.tsk_k /
// surface.qsfc_kgkg serialize to v0 surface.tsfc / surface.qsfc ONLY when set;
// omitting them leaves the runtime to hold the initial surface-air state as the
// skin. A legacy v0 surface.tsfc round-trips back into v1 surface.tsk_k.
// ─────────────────────────────────────────────────────────────

TEST_CASE("serialize: prescribed skin state emits surface.tsfc/qsfc only when set", "[init][config][v0_to_v1][surface]")
{
    SECTION("tsk_k + qsfc_kgkg set -> surface.tsfc + surface.qsfc emitted")
    {
        ConfigV1 c;
        c.surface.scheme       = SurfaceScheme::Slab;
        c.surface.apply_fluxes = true;
        c.surface.tsk_k        = 305.0;
        c.surface.qsfc_kgkg    = 0.018;
        auto kv                = serialize_v1_to_v0_map(c);
        REQUIRE(kv.count("surface.tsfc") == 1);
        REQUIRE(kv.count("surface.qsfc") == 1);
        REQUIRE(std::stod(kv.at("surface.tsfc")) == Approx(305.0));
        REQUIRE(std::stod(kv.at("surface.qsfc")) == Approx(0.018));
    }

    SECTION("skin unset (sentinel) -> neither key emitted")
    {
        ConfigV1 c;
        c.surface.scheme       = SurfaceScheme::Slab;
        c.surface.apply_fluxes = true;
        // tsk_k / qsfc_kgkg keep their < 0 'unset' sentinel.
        auto kv = serialize_v1_to_v0_map(c);
        REQUIRE(kv.count("surface.tsfc") == 0);
        REQUIRE(kv.count("surface.qsfc") == 0);
    }

    SECTION("qsfc_kgkg = 0 is a valid (dry) skin and IS emitted")
    {
        ConfigV1 c;
        c.surface.qsfc_kgkg = 0.0;
        auto kv             = serialize_v1_to_v0_map(c);
        REQUIRE(kv.count("surface.qsfc") == 1);
        REQUIRE(std::stod(kv.at("surface.qsfc")) == Approx(0.0));
    }
}

TEST_CASE("surface exchange and radiative properties project without hidden defaults",
          "[init][config][v0_to_v1][surface][radiation]")
{
    ConfigV1 c;
    c.grid.nx            = 8;
    c.grid.ny            = 8;
    c.grid.nz            = 8;
    c.grid.dx            = 1000.0;
    c.grid.dy            = 1000.0;
    c.grid.dz            = 500.0;
    c.time.run_time      = 10.0;
    c.time.dt_large      = 1.0;
    c.surface.z0m_m      = 0.23;
    c.surface.z0h_m      = 0.017;
    c.surface.albedo     = 0.31;
    c.surface.emissivity = 0.88;
    const auto flat      = serialize_v1_to_v0_map(c);
    REQUIRE(std::stod(flat.at("surface.z0m")) == Approx(0.23));
    REQUIRE(std::stod(flat.at("surface.z0h")) == Approx(0.017));
    REQUIRE(std::stod(flat.at("surface.albedo")) == Approx(0.31));
    REQUIRE(std::stod(flat.at("surface.emissivity")) == Approx(0.88));

    const std::string yaml = serialize_v1_to_yaml(c);
    TempYaml          tmp(yaml);
    const auto        loaded = load_v1_config(tmp.path.string());
    INFO(format_errors(loaded.errors));
    REQUIRE(loaded.ok());
    REQUIRE(loaded.config->surface.z0m_m == Approx(0.23));
    REQUIRE(loaded.config->surface.z0h_m == Approx(0.017));
    REQUIRE(loaded.config->surface.albedo == Approx(0.31));
    REQUIRE(loaded.config->surface.emissivity == Approx(0.88));
}

TEST_CASE("migrate: v0 surface.tsfc/qsfc -> v1 surface.tsk_k/qsfc_kgkg", "[init][config][v0_to_v1][surface]")
{
    TempYaml tmp(R"yaml(
duration_s: 100
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 250
  dy: 250
  dz: 250
  dt: 1
environment:
  sfc_theta_k: 302.0
  sfc_qv_kgkg: 0.013
dynamics:
  scheme: cartesian
microphysics:
  scheme: none
surface:
  tsfc: 305.0
  qsfc: 0.018
output:
  write_every_s: 30
)yaml");

    auto m = migrate_v0_to_v1(tmp.path.string());
    REQUIRE(m.ok());
    REQUIRE(m.config->surface.tsk_k == Approx(305.0));
    REQUIRE(m.config->surface.qsfc_kgkg == Approx(0.018));
}

// ─────────────────────────────────────────────────────────────
// Targeted: microphysics milbrandt is the canonical token. The v0 migrator
// alone recognizes historical "morrison" spelling and emits a semantic warning.
// ─────────────────────────────────────────────────────────────

TEST_CASE("serialize: microphysics milbrandt is the canonical v0 token", "[init][config][v0_to_v1][microphysics]")
{
    ConfigV1 c;
    c.microphysics.scheme = MicrophysicsScheme::Milbrandt;
    auto kv               = serialize_v1_to_v0_map(c);
    REQUIRE(kv.at("microphysics.scheme") == "milbrandt");
}

// ─────────────────────────────────────────────────────────────
// Targeted: chaos.* → extensions.chaos.* rename
// ─────────────────────────────────────────────────────────────

TEST_CASE("migrate: chaos.* → extensions.chaos.* rename", "[init][config][v0_to_v1]")
{
    TempYaml tmp(R"yaml(
duration_s: 100
grid:
  nx: 64
  ny: 64
  nz: 32
  dx: 250
  dy: 250
  dz: 250
  dt: 1
dynamics:
  scheme: cartesian
microphysics:
  scheme: none
chaos:
  scheme: initial_conditions
  seed: 12345
  member_id: 7
output:
  write_every_s: 30
)yaml");

    auto m = migrate_v0_to_v1(tmp.path.string());
    REQUIRE(m.ok());
    REQUIRE(m.config->extensions.chaos.scheme == ChaosScheme::InitialConditions);
    REQUIRE(m.config->extensions.chaos.seed == 12345);
    REQUIRE(m.config->extensions.chaos.member_id == 7);
}

// ─────────────────────────────────────────────────────────────
// Negative: reject v1 inputs (migrator is one-way)
// ─────────────────────────────────────────────────────────────

TEST_CASE("migrate: refuses configs that already have schema stamp", "[init][config][v0_to_v1]")
{
    TempYaml tmp(R"yaml(
schema: nimbus.v1
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
)yaml");

    auto m = migrate_v0_to_v1(tmp.path.string());

    REQUIRE_FALSE(m.ok());
    REQUIRE_FALSE(m.errors.empty());
    REQUIRE(m.errors[0].key == "schema");
    REQUIRE(m.errors[0].message.find("already has a 'schema:' stamp") != std::string::npos);
}

// ─────────────────────────────────────────────────────────────
// Negative: missing v0 file produces clean error
// ─────────────────────────────────────────────────────────────

TEST_CASE("migrate: missing file produces clean error", "[init][config][v0_to_v1]")
{
    auto m = migrate_v0_to_v1("/tmp/nonexistent_migrator_test_xyz999.yaml");
    REQUIRE_FALSE(m.ok());
    REQUIRE(m.errors.size() == 1);
    REQUIRE(m.errors[0].message.find("could not open") != std::string::npos);
}

// ─────────────────────────────────────────────────────────────
// Serializer: round-trip identity for the default ConfigV1
// ─────────────────────────────────────────────────────────────

TEST_CASE("serialize_v1_to_yaml: default config round-trips identity", "[init][config][v0_to_v1][serializer]")
{
    ConfigV1 in;
    in.grid.nx              = 64;
    in.grid.ny              = 32;
    in.grid.nz              = 24;
    in.grid.dx              = 500;
    in.grid.dy              = 500;
    in.grid.dz              = 250;
    in.time.run_time        = 1200;
    in.time.dt_large        = 1.5;
    in.time.output_interval = 30;

    const std::string yaml = serialize_v1_to_yaml(in);
    TempYaml          tmp(yaml);
    auto              result = load_v1_config(tmp.path.string());
    INFO(format_errors(result.errors));
    REQUIRE(result.ok());

    REQUIRE(result.config->grid.nx == 64);
    REQUIRE(result.config->grid.ny == 32);
    REQUIRE(result.config->time.dt_large == Approx(1.5));
}

TEST_CASE("RRTMG composition round-trips and projects to the runtime map",
          "[init][config][radiation][rrtmg][serializer]")
{
    ConfigV1 config;
    config.grid.nx                                   = 8;
    config.grid.ny                                   = 8;
    config.grid.nz                                   = 8;
    config.grid.dx                                   = 500.0;
    config.grid.dy                                   = 500.0;
    config.grid.dz                                   = 250.0;
    config.time.run_time                             = 60.0;
    config.time.dt_large                             = 1.0;
    config.radiation.scheme                           = RadiationScheme::RRTMG;
    config.radiation.do_lw                            = false;
    config.radiation.rrtmg_ozone_vmr                  = 4.0e-6;
    config.radiation.rrtmg_co2_vmr                    = 421.0e-6;
    config.radiation.rrtmg_ch4_vmr                    = 1.91e-6;
    config.radiation.rrtmg_n2o_vmr                    = 0.336e-6;
    config.radiation.rrtmg_co_vmr                     = 0.1e-6;
    config.radiation.rrtmg_o2_vmr                     = 0.2094;
    config.radiation.rrtmg_n2_vmr                     = 0.7807;
    config.radiation.rrtmg_cfc11_vmr                  = 2.0e-10;
    config.radiation.rrtmg_cfc12_vmr                  = 5.0e-10;
    config.radiation.rrtmg_cfc22_vmr                  = 3.0e-10;
    config.radiation.rrtmg_ccl4_vmr                   = 1.0e-10;
    config.radiation.rrtmg_liquid_effective_radius_um = 11.0;
    config.radiation.rrtmg_ice_effective_radius_um    = 35.0;

    const auto flat = serialize_v1_to_v0_map(config);
    REQUIRE(std::stod(flat.at("radiation.rrtmg_ozone_vmr")) == Approx(4.0e-6));
    REQUIRE(std::stod(flat.at("radiation.rrtmg_co2_vmr")) == Approx(421.0e-6));
    REQUIRE(std::stod(flat.at("radiation.rrtmg_ch4_vmr")) == Approx(1.91e-6));
    REQUIRE(std::stod(flat.at("radiation.rrtmg_n2o_vmr")) == Approx(0.336e-6));
    REQUIRE(std::stod(flat.at("radiation.rrtmg_co_vmr")) == Approx(0.1e-6));
    REQUIRE(std::stod(flat.at("radiation.rrtmg_o2_vmr")) == Approx(0.2094));
    REQUIRE(std::stod(flat.at("radiation.rrtmg_n2_vmr")) == Approx(0.7807));
    REQUIRE(std::stod(flat.at("radiation.rrtmg_cfc11_vmr")) == Approx(2.0e-10));
    REQUIRE(std::stod(flat.at("radiation.rrtmg_cfc12_vmr")) == Approx(5.0e-10));
    REQUIRE(std::stod(flat.at("radiation.rrtmg_cfc22_vmr")) == Approx(3.0e-10));
    REQUIRE(std::stod(flat.at("radiation.rrtmg_ccl4_vmr")) == Approx(1.0e-10));
    REQUIRE(std::stod(flat.at("radiation.rrtmg_liquid_effective_radius_um")) == Approx(11.0));
    REQUIRE(std::stod(flat.at("radiation.rrtmg_ice_effective_radius_um")) == Approx(35.0));

    TempYaml strict(serialize_v1_to_yaml(config));
    const auto loaded = load_v1_config(strict.path.string());
    INFO(format_errors(loaded.errors));
    REQUIRE(loaded.ok());
    REQUIRE(loaded.config->radiation.rrtmg_ozone_vmr == Approx(4.0e-6));
    REQUIRE(loaded.config->radiation.rrtmg_liquid_effective_radius_um == Approx(11.0));
    REQUIRE(loaded.config->radiation.rrtmg_ice_effective_radius_um == Approx(35.0));
}

TEST_CASE("microphysics parameters serialize only the selected variant and project to runtime",
          "[init][config][microphysics][parameters]")
{
    ConfigV1 config;
    config.grid.nx                                                   = 8;
    config.grid.ny                                                   = 8;
    config.grid.nz                                                   = 8;
    config.grid.dx                                                   = 500.0;
    config.grid.dy                                                   = 500.0;
    config.grid.dz                                                   = 250.0;
    config.time.run_time                                             = 60.0;
    config.time.dt_large                                             = 1.0;
    config.microphysics.scheme                                       = MicrophysicsScheme::Lin;
    config.microphysics.parameters.shared.cloud_water_threshold_kgkg = 0.002;
    config.microphysics.parameters.shared.autoconversion_rate_s_inv  = 0.004;
    config.microphysics.parameters.lin.accretion_coefficient         = 1.7;
    config.microphysics.parameters.lin.evaporation_rate_s_inv        = 0.0008;

    const std::string yaml = serialize_v1_to_yaml(config);
    REQUIRE(yaml.find("  shared:\n") != std::string::npos);
    REQUIRE(yaml.find("  lin:\n") != std::string::npos);
    REQUIRE(yaml.find("  kessler:\n") == std::string::npos);
    REQUIRE(yaml.find("  milbrandt:\n") == std::string::npos);

    TempYaml   tmp(yaml);
    const auto loaded = load_v1_config(tmp.path.string());
    INFO(format_errors(loaded.errors));
    REQUIRE(loaded.ok());
    REQUIRE(loaded.config->microphysics.parameters.lin.accretion_coefficient == Approx(1.7));

    const auto flat = serialize_v1_to_v0_map(config);
    REQUIRE(flat.at("microphysics.shared.cloud_water_threshold_kgkg") == "0.002");
    REQUIRE(flat.at("microphysics.lin.accretion_coefficient") == "1.7");
    REQUIRE(flat.count("microphysics.kessler.accretion_coefficient") == 0);
}

TEST_CASE("Thompson controls serialize and obsolete Milbrandt rate is dropped",
          "[init][config][microphysics][parameters][migration]")
{
    SECTION("strict v1 serialization and projection select Thompson only")
    {
        ConfigV1 config;
        config.grid.nx = 8;
        config.grid.ny = 8;
        config.grid.nz = 8;
        config.grid.dx = 500.0;
        config.grid.dy = 500.0;
        config.grid.dz = 250.0;
        config.time.run_time = 60.0;
        config.time.dt_large = 1.0;
        config.microphysics.scheme                                     = MicrophysicsScheme::Thompson;
        config.microphysics.parameters.thompson.evaporation_rate_s_inv = 0.0007;
        config.microphysics.parameters.thompson
            .water_friendly_aerosol_surface_m3 = 375.0e6;
        config.microphysics.parameters.thompson
            .ice_friendly_aerosol_surface_m3 = 2.25e6;
        const std::string yaml                                         = serialize_v1_to_yaml(config);
        REQUIRE(yaml.find("  thompson:\n") != std::string::npos);
        REQUIRE(yaml.find("evaporation_rate_s_inv: 0.0007") != std::string::npos);
        REQUIRE(yaml.find("water_friendly_aerosol_surface_m3: 375000000")
                != std::string::npos);
        REQUIRE(yaml.find("ice_friendly_aerosol_surface_m3: 2250000")
                != std::string::npos);
        REQUIRE(yaml.find("  milbrandt:\n") == std::string::npos);
        const auto flat = serialize_v1_to_v0_map(config);
        REQUIRE(flat.at("microphysics.thompson.evaporation_rate_s_inv") == "0.0007");
        REQUIRE(flat.at("microphysics.thompson.water_friendly_aerosol_surface_m3")
                == "375000000");
        REQUIRE(flat.at("microphysics.thompson.ice_friendly_aerosol_surface_m3")
                == "2250000");
        REQUIRE(flat.count("microphysics.milbrandt.evaporation_rate_s_inv") == 0);

        TempYaml roundtrip(yaml);
        const auto loaded = load_v1_config(roundtrip.path.string());
        INFO(format_errors(loaded.errors));
        REQUIRE(loaded.ok());
        REQUIRE(loaded.config->microphysics.parameters.thompson
                    .water_friendly_aerosol_surface_m3 == Approx(375.0e6));
        REQUIRE(loaded.config->microphysics.parameters.thompson
                    .ice_friendly_aerosol_surface_m3 == Approx(2.25e6));
    }

    SECTION("legacy migration drops inactive Thompson and obsolete Milbrandt controls")
    {
        TempYaml   source(R"yaml(
microphysics:
  scheme: milbrandt
  thompson:
    evaporation_rate_s_inv: 0.0007
  milbrandt:
    evaporation_rate_s_inv: 0.0006
)yaml");
        const auto migrated = migrate_v0_to_v1(source.path.string());
        REQUIRE(migrated.ok());
        REQUIRE(migrated.config->microphysics.parameters.milbrandt.evaporation_rate_s_inv == Approx(1.0e-3));
        REQUIRE(migrated.config->microphysics.parameters.thompson.evaporation_rate_s_inv == Approx(1.0e-3));
        REQUIRE(std::any_of(migrated.warnings.begin(),
                            migrated.warnings.end(),
                            [](const MigrationWarning& warning)
                            { return warning.key == "microphysics.thompson.evaporation_rate_s_inv"; }));
        REQUIRE(std::any_of(migrated.warnings.begin(),
                            migrated.warnings.end(),
                            [](const MigrationWarning& warning)
                            { return warning.key == "microphysics.milbrandt.evaporation_rate_s_inv"; }));
    }
}

TEST_CASE("Kessler evaporation efficiency serializes and obsolete rates migrate safely",
          "[init][config][microphysics][parameters][migration][kessler]")
{
    SECTION("strict v1 serialization and projection preserve the dimensionless control")
    {
        ConfigV1 config;
        config.microphysics.scheme                                    = MicrophysicsScheme::Kessler;
        config.microphysics.parameters.kessler.evaporation_efficiency = 0.75;
        const std::string yaml                                        = serialize_v1_to_yaml(config);
        REQUIRE(yaml.find("  kessler:\n") != std::string::npos);
        REQUIRE(yaml.find("evaporation_efficiency: 0.75") != std::string::npos);
        REQUIRE(yaml.find("kessler:\n    evaporation_rate_s_inv:") == std::string::npos);
        const auto flat = serialize_v1_to_v0_map(config);
        REQUIRE(flat.at("microphysics.kessler.evaporation_efficiency") == "0.75");
        REQUIRE(flat.count("microphysics.kessler.evaporation_rate_s_inv") == 0);
    }

    SECTION("legacy first-order rate is consumed but cannot be converted dimensionally")
    {
        TempYaml   source(R"yaml(
microphysics:
  scheme: kessler
  kessler:
    evaporation_rate_s_inv: 0.003
)yaml");
        const auto migrated = migrate_v0_to_v1(source.path.string());
        INFO(format_errors(migrated.errors));
        REQUIRE(migrated.ok());
        REQUIRE(migrated.config->microphysics.parameters.kessler.evaporation_efficiency == Approx(1.0));
        REQUIRE(std::any_of(migrated.warnings.begin(),
                            migrated.warnings.end(),
                            [](const MigrationWarning& warning)
                            { return warning.key == "microphysics.kessler.evaporation_rate_s_inv"; }));
    }

    SECTION("legacy files may already provide the new efficiency")
    {
        TempYaml   source(R"yaml(
microphysics:
  scheme: kessler
  kessler:
    evaporation_efficiency: 0.6
)yaml");
        const auto migrated = migrate_v0_to_v1(source.path.string());
        INFO(format_errors(migrated.errors));
        REQUIRE(migrated.ok());
        REQUIRE(migrated.config->microphysics.parameters.kessler.evaporation_efficiency == Approx(0.6));
    }
}

TEST_CASE("checkpoint lifecycle round-trips and projects to the runtime map", "[init][config][checkpoint]")
{
    ConfigV1 config;
    config.grid.nx = config.grid.ny = config.grid.nz = 8;
    config.grid.dx = config.grid.dy = config.grid.dz    = 250.0;
    config.time.run_time                                = 120.0;
    config.time.dt_large                                = 1.0;
    config.extensions.checkpoint.every_s                = 30;
    config.extensions.checkpoint.keep_last              = 3;
    config.extensions.checkpoint.minimum_free_space_mib = 1024;
    config.extensions.checkpoint.directory              = "/research/checkpoints";
    config.extensions.checkpoint.resume_from            = "/research/parent/checkpoint.nimbus";

    const std::string yaml = serialize_v1_to_yaml(config);
    REQUIRE(yaml.find("  checkpoint:\n") != std::string::npos);
    TempYaml   tmp(yaml);
    const auto loaded = load_v1_config(tmp.path.string());
    INFO(format_errors(loaded.errors));
    REQUIRE(loaded.ok());
    REQUIRE(loaded.config->extensions.checkpoint.every_s == 30);
    REQUIRE(loaded.config->extensions.checkpoint.keep_last == 3);
    REQUIRE(loaded.config->extensions.checkpoint.minimum_free_space_mib == 1024);

    const auto flat = serialize_v1_to_v0_map(config);
    REQUIRE(flat.at("checkpoint.every_s") == "30");
    REQUIRE(flat.at("checkpoint.keep_last") == "3");
    REQUIRE(flat.at("checkpoint.minimum_free_space_mib") == "1024");
    REQUIRE(flat.at("checkpoint.directory") == "/research/checkpoints");
    REQUIRE(flat.at("checkpoint.resume_from") == "/research/parent/checkpoint.nimbus");
}

// ─────────────────────────────────────────────────────────────
// Warning formatter
// ─────────────────────────────────────────────────────────────

TEST_CASE("format_warnings emits file:line: key: action (reason)", "[init][config][v0_to_v1]")
{
    std::vector<MigrationWarning> ws = {
        {"case_name", 3, "dropped", "metadata not part of v1 schema"},
        {"trigger.bubble.center_x_km",
         12,
         "renamed to initial_conditions.trigger.bubble.center_x_m (km → m (×1000))",
         "value unit-converted during rename"},
    };
    const std::string s = format_warnings("foo.yaml", ws);
    REQUIRE(s.find("foo.yaml:3: case_name: dropped") != std::string::npos);
    REQUIRE(s.find("foo.yaml:12: trigger.bubble.center_x_km: renamed") != std::string::npos);
    REQUIRE(s.find("km → m") != std::string::npos);
}

TEST_CASE("migrate: legacy Richardson equations cannot change silently", "[init][config][richardson][migration]")
{
    TempYaml tmp("turbulence:\n  scheme: smagorinsky\n  stability_correction: ri\n");
    const auto result=migrate_v0_to_v1(tmp.path.string());
    REQUIRE_FALSE(result.ok());
    REQUIRE(std::any_of(result.errors.begin(),result.errors.end(),[](const auto& error) {
        return error.key=="turbulence.stability_correction" && error.message.find("lossless")!=std::string::npos;
    }));
}
