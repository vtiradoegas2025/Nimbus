#include <catch2/catch.hpp>

#include "core/runtime/flat_mixed_metal_admission.hpp"
#include "core/runtime/qualified_generated_metal_cli_scope.hpp"
#include "init/config/v1_loader.hpp"

#include <functional>
#include <limits>
#include <string>
#include <vector>

using namespace tmv::init::config;

namespace
{
ConfigV1 retained_profile()
{
    const auto loaded = load_v1_config(
        "tests/fixtures/reference/catalog/flat_mixed_boundary_metal_exploratory_v1.yaml");
    REQUIRE(loaded.ok());
    auto config = *loaded.config;
    config.extensions.execution.mode = ExecutionMode::Qualified;
    return config;
}
}

TEST_CASE("mixed-open proposed admission is pure and binds the exact retained bytes",
          "[core][flat_mixed][admission]")
{
    auto config = retained_profile();
    const auto before = serialize_v1_to_yaml(config);
    REQUIRE(nimbus::crypto::sha256_hex(before) ==
        tmv::runtime::kFlatMixedMetalRetainedQualifiedYamlSha256);
    REQUIRE(tmv::runtime::matches_flat_mixed_metal_physics(config));
    REQUIRE(tmv::runtime::matches_flat_mixed_metal_cadence1_output(config));
    REQUIRE(tmv::runtime::is_proposed_exact_flat_mixed_metal_cadence1_profile(config));
    REQUIRE(tmv::runtime::is_public_qualified_flat_mixed_metal_cadence1_profile(config));
    REQUIRE(serialize_v1_to_yaml(config) == before);
    config.extensions.compute.memory_admission = MemoryAdmissionPolicy::Reject;
    const auto reject_before = serialize_v1_to_yaml(config);
    REQUIRE(tmv::runtime::is_proposed_exact_flat_mixed_metal_cadence1_profile(config));
    REQUIRE(serialize_v1_to_yaml(config) == reject_before);
}

TEST_CASE("mixed-open CLI scope accepts only its reviewed cadence and unchanged canonical controls",
          "[core][flat_mixed][admission][cli]")
{
    tmv::runtime::QualifiedGeneratedMetalCliOverrides cli;
    const auto error = [&](bool east, bool mixed, std::string_view format)
    {
        return tmv::runtime::qualified_generated_metal_cli_scope_error(cli, east, mixed, format);
    };
    REQUIRE(error(false, true, "npy_3d").empty());
    cli.dt_large = true;
    cli.dt_large_s = 0.1;
    cli.acoustic_substeps = true;
    cli.acoustic_substeps_value = 6;
    cli.write_every = true;
    cli.write_every_s = 1;
    cli.guard_mode = true;
    cli.guard_fail_on = true;
    cli.guard_scope = true;
    REQUIRE(error(false, true, "npy_3d").empty());
    REQUIRE(error(true, false, "npy_3d").empty());
    REQUIRE(error(false, false, "npy_3d").empty());
    REQUIRE(error(false, false, "netcdf").empty());
    REQUIRE(error(false, false, "zfp_high").empty());
    cli.write_every_s = 301;
    REQUIRE(error(false, true, "npy_3d") ==
        "qualified mixed-open Metal NPY scope requires --write-every=1");
    REQUIRE(error(true, false, "npy_3d") ==
        "qualified east-radiative Metal NPY scope requires --write-every=1");
    REQUIRE(error(false, false, "npy_3d").empty());
    REQUIRE(error(false, false, "netcdf").empty());
    REQUIRE_FALSE(error(false, false, "zfp_high").empty());
    for (const int cadence : {0, 2, 301})
    {
        cli.write_every_s = cadence;
        REQUIRE(error(false, true, "npy_3d") ==
            "qualified mixed-open Metal NPY scope requires --write-every=1");
    }
    cli.write_every_s = 1;
    for (const auto format : {"netcdf", "zfp_high", "zfp_medium"})
    {
        REQUIRE(error(false, true, format) ==
            "qualified mixed-open Metal scope requires synchronous minimal NPY");
    }
}

TEST_CASE("mixed-open effective CLI cannot weaken timestep guard or publication owners",
          "[core][flat_mixed][admission][cli]")
{
    tmv::runtime::QualifiedGeneratedMetalCliOverrides cli;
    const auto error = [&]()
    {
        return tmv::runtime::qualified_generated_metal_cli_scope_error(cli, false, true, "npy_3d");
    };
    cli.dt_large = true;
    cli.dt_large_s = 0.2;
    REQUIRE(error() == "qualified generated Metal scope requires --dt-large=0.1");
    cli = {};
    cli.acoustic_substeps = true;
    cli.acoustic_substeps_value = 5;
    REQUIRE(error() == "qualified generated Metal scope requires --acoustic-substeps=6");
    cli = {};
    cli.guard_mode = true;
    cli.guard_mode_value = tmv::GuardMode::Off;
    REQUIRE(error().find("--guard-mode=strict") != std::string_view::npos);
    cli = {};
    cli.guard_fail_on = true;
    cli.guard_fail_on_value = tmv::GuardFailOn::NonFinite;
    REQUIRE(error().find("--guard-fail-on=both") != std::string_view::npos);
    cli = {};
    cli.guard_scope = true;
    cli.guard_scope_value = tmv::StrictGuardScope::ExportedNow;
    REQUIRE(error() == "qualified generated Metal scope requires --guard-scope=required");
    cli = {};
    cli.live_shm = true;
    REQUIRE(error() == "qualified generated Metal scope does not admit live-SHM CLI overrides");
    cli = {};
    cli.trajectories = true;
    REQUIRE(error() == "qualified generated Metal scope does not admit trajectory CLI overrides");
}

TEST_CASE("mixed-open proposal rejects cadence guard and publication mutations",
          "[core][flat_mixed][admission][output]")
{
    const auto baseline = retained_profile();
    for (const auto cadence : {301.0, 2.0, 0.0, 0.5,
             std::numeric_limits<double>::quiet_NaN()})
    {
        auto config = baseline;
        config.time.output_interval = cadence;
        REQUIRE_FALSE(tmv::runtime::matches_flat_mixed_metal_cadence1_output(config));
        REQUIRE_FALSE(tmv::runtime::is_proposed_exact_flat_mixed_metal_cadence1_profile(config));
    }
    const std::vector<std::pair<std::string, std::function<void(ConfigV1&)>>> mutations = {
        {"dt", [](auto& c) { c.time.dt_large = 0.2; }},
        {"substeps", [](auto& c) { c.time.acoustic_substeps = 5; }},
        {"auto substeps", [](auto& c) { c.time.acoustic_substeps = 0; }},
        {"guard off", [](auto& c) { c.extensions.validation.guard_mode = ValidationGuardMode::Off; }},
        {"guard warn", [](auto& c) { c.extensions.validation.guard_mode = ValidationGuardMode::Warn; }},
        {"live SHM", [](auto& c) { c.output.live_shm = true; }},
        {"trajectory", [](auto& c) { c.output.trajectories = true; }},
        {"async", [](auto& c) { c.output.async_io = true; }},
        {"netCDF", [](auto& c) { c.output.format = OutputFormat::Netcdf; }},
        {"ZFP", [](auto& c) { c.output.format = OutputFormat::ZfpHigh; }},
        {"preset", [](auto& c) { c.output.preset = OutputPreset::DynamicsFull; }},
        {"tiered", [](auto& c) { c.output.tiered_write_cadence = true; }},
        {"retention", [](auto& c) { c.output.keep_last_frames = 1; }},
        {"history ceiling", [](auto& c) { c.output.maximum_retained_history_files = 1; }},
        {"free space", [](auto& c) { c.output.minimum_free_space_mib = 1; }},
        {"diagnostic stream", [](auto& c) { c.output.qualification_diagnostics_preset = QualificationDiagnosticsPreset::MovingReferenceV1; }},
        {"science cadence", [](auto& c) { c.extensions.diagnostics.scientific_every_steps = 2; }},
        {"reduction", [](auto& c) { c.extensions.diagnostics.deterministic_reductions = false; }},
        {"CPU backend", [](auto& c) { c.extensions.compute.backend = ComputeBackend::CPU; }},
        {"Vulkan backend", [](auto& c) { c.extensions.compute.backend = ComputeBackend::GPU; }},
        {"fallback", [](auto& c) { c.extensions.compute.allow_fallback = true; }},
        {"parity harness", [](auto& c) { c.extensions.compute.validate_parity = true; }},
        {"memory policy", [](auto& c) { c.extensions.compute.memory_admission = static_cast<MemoryAdmissionPolicy>(99); }},
        {"exploratory", [](auto& c) { c.extensions.execution.mode = ExecutionMode::Exploratory; }},
    };
    for (const auto& [name, mutate] : mutations)
    {
        INFO(name);
        auto config = baseline;
        mutate(config);
        REQUIRE_FALSE(tmv::runtime::is_proposed_exact_flat_mixed_metal_cadence1_profile(config));
    }
}

TEST_CASE("mixed-open proposal rejects adjacent topology and active-owner mutations",
          "[core][flat_mixed][admission][physics]")
{
    const auto baseline = retained_profile();
    const std::vector<std::pair<std::string, std::function<void(ConfigV1&)>>> mutations = {
        {"east rigid", [](auto& c) { c.boundary_conditions.lateral.east = LateralBC::Rigid; }},
        {"east radiative", [](auto& c) { c.boundary_conditions.lateral.east = LateralBC::Radiative; }},
        {"unpaired periodic", [](auto& c) { c.boundary_conditions.lateral.west = LateralBC::Periodic; }},
        {"bottom slip", [](auto& c) { c.boundary_conditions.bottom = BottomBC::RigidSlip; }},
        {"top rigid", [](auto& c) { c.boundary_conditions.top = TopBC::Rigid; }},
        {"adaptive", [](auto& c) { c.time.adaptive_dt = true; }},
        {"grid extent", [](auto& c) { c.grid.nx = 20; }},
        {"grid spacing", [](auto& c) { c.grid.dx = 501.0; }},
        {"nonfinite grid", [](auto& c) { c.grid.dx = std::numeric_limits<double>::quiet_NaN(); }},
        {"integrator", [](auto& c) { c.time.integrator = TimeIntegrator::SSPRK3; }},
        {"transport", [](auto& c) { c.advection.scheme = AdvectionScheme::WENO5; }},
        {"microphysics", [](auto& c) { c.microphysics.scheme = MicrophysicsScheme::Kessler; }},
        {"SGS", [](auto& c) { c.turbulence.scheme = TurbulenceScheme::Smagorinsky; }},
        {"PBL", [](auto& c) { c.boundary_layer.scheme = BoundaryLayerScheme::Slab; }},
        {"radiation", [](auto& c) { c.radiation.scheme = RadiationScheme::SimpleGrey; }},
        {"surface", [](auto& c) { c.surface.apply_fluxes = true; }},
        {"diffusion", [](auto& c) { c.diffusion.K_h = 1.0; }},
        {"top damping", [](auto& c) { c.damping.enabled = true; }},
        {"lateral damping default", [](auto& c) { c.damping.lateral_sponge_enabled = false; }},
        {"lid damping default", [](auto& c) { c.damping.w_only_top_sponge = false; }},
        {"lateral damping width", [](auto& c) { c.damping.lateral_width = 0; }},
        {"lateral damping strength", [](auto& c) { c.damping.lateral_alpha = 0.0; }},
        {"Coriolis", [](auto& c) { c.coriolis.enabled = true; }},
        {"motion", [](auto& c) { c.domain_motion.mode = DomainMotionMode::ConstantVelocity; }},
        {"velocity", [](auto& c) { c.domain_motion.frame_velocity.u_ms = 1.0; }},
        {"planner", [](auto& c) { c.domain_motion.lifetime_planning.enabled = true; }},
        {"tracking", [](auto& c) { c.domain_motion.tracking.enabled = true; }},
        {"recycling", [](auto& c) { c.domain_motion.tracking.recycle_domain = true; }},
        {"nesting", [](auto& c) { c.domain_motion.nesting.enabled = true; }},
        {"disabled two-way", [](auto& c) { c.domain_motion.nesting.feedback = NestingFeedback::TwoWay; }},
        {"bubble cloud", [](auto& c) { c.initial_conditions.trigger.bubble.cloud_water_kgkg = 0.001; }},
        {"bubble amplitude", [](auto& c) { c.initial_conditions.trigger.bubble.dtheta_k = 3.0; }},
        {"chaos seed", [](auto& c) { c.extensions.chaos.seed = 1; }},
        {"chaos amplitude", [](auto& c) { c.extensions.chaos.sigma_ic["theta"] = 0.6; }},
        {"chaos target", [](auto& c) { c.extensions.chaos.apply_to_ic = {"theta"}; }},
        {"native chaos", [](auto& c) { c.extensions.chaos.native_generation_required = true; }},
        {"solver", [](auto& c) { c.extensions.solver.slow_rk2 = false; }},
    };
    for (const auto& [name, mutate] : mutations)
    {
        INFO(name);
        auto config = baseline;
        mutate(config);
        REQUIRE_FALSE(tmv::runtime::is_proposed_exact_flat_mixed_metal_cadence1_profile(config));
    }
}

TEST_CASE("inactive constants and unreviewed operational aliases retain exact hash guards",
          "[core][flat_mixed][admission][aliases]")
{
    const auto baseline = retained_profile();
    const std::vector<std::pair<std::string, std::function<void(ConfigV1&)>>> mutations = {
        {"inactive radiation cadence", [](auto& c) { c.radiation.cadence_s = 1.0; }},
        {"inactive PBL cadence", [](auto& c) { c.boundary_layer.cadence_s = 1.0; }},
        {"inactive SGS coefficient", [](auto& c) { c.turbulence.Cs = 0.2; }},
        {"inactive terrain shape", [](auto& c) { c.terrain.bell.h0 = 2000.0; }},
        {"inactive one-way ratio", [](auto& c) { c.domain_motion.nesting.parent_ratio_x = 4; }},
        {"inactive trajectory count", [](auto& c) { c.output.trajectory_count = 6; }},
        {"inactive SHM fields", [](auto& c) { c.output.live_shm_fields = {"theta"}; }},
        {"checkpoint cadence", [](auto& c) { c.extensions.checkpoint.every_s = 0; }},
        {"duration", [](auto& c) { c.time.run_time = 1.0; }},
        {"performance", [](auto& c) { c.extensions.performance.timing = false; }},
        {"inactive codec tolerance", [](auto& c) { c.output.zfp_tolerance = 0.01; }},
        {"inactive netCDF layout", [](auto& c) { c.output.netcdf_target_chunk_kib = 512; }},
        {"inactive tier cadence", [](auto& c) { c.output.write_cadence_slow_s = 120; }},
        {"inactive optical model", [](auto& c) { c.radiation.simple_grey_optical_model = "changed"; }},
    };
    for (const auto& [name, mutate] : mutations)
    {
        INFO(name);
        auto config = baseline;
        mutate(config);
        REQUIRE(tmv::runtime::matches_flat_mixed_metal_physics(config));
        REQUIRE(tmv::runtime::matches_flat_mixed_metal_cadence1_output(config));
        REQUIRE_FALSE(tmv::runtime::is_proposed_exact_flat_mixed_metal_cadence1_profile(config));
    }
}
