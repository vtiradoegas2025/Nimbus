#include "catch2/catch.hpp"
#include "core/runtime/configuration_compiler.hpp"

#include <tuple>

TEST_CASE("Typed compiler canonicalizes Vulkan backend spelling for runtime admission",
          "[init][config][compute][compilation]")
{
    for (const std::string spelling : {"gpu", "vulkan", "vk"})
    {
        tmv::init::config::ConfigV1 config;
        REQUIRE(tmv::init::config::parse_compute_backend(spelling,config.extensions.compute.backend));
        const auto compiled=tmv::configuration::compile_physics(config);
        REQUIRE(compiled.compute.backend=="vulkan");
    }
}

TEST_CASE("Typed compiler preserves explicit CUDA selection and qualified fallback policy",
          "[init][config][cuda-assist][compilation]")
{
    for (const bool enabled : {false, true})
    {
        tmv::init::config::ConfigV1 config;
        config.extensions.compute.backend = tmv::init::config::ComputeBackend::GPU;
        config.extensions.compute.cuda_assist = enabled;
        config.extensions.compute.allow_fallback = true;
        config.extensions.execution.mode = tmv::init::config::ExecutionMode::Qualified;
        auto compiled = tmv::configuration::compile_physics(config);
        REQUIRE(compiled.compute.cuda_assist == enabled);
        REQUIRE_FALSE(compiled.compute.allow_fallback);
        SimulationStorageState state;
        std::move(compiled).apply_to(state);
        REQUIRE(state.runtime.compute.config.cuda_assist == enabled);
    }
}

TEST_CASE("Strict Richardson selection projects the versioned Lilly model",
          "[init][config][richardson][compilation]")
{
    tmv::init::config::ConfigV1 config;
    config.turbulence.stability_correction=tmv::init::config::TurbulenceStabilityCorrection::Richardson;
    CHECK(tmv::configuration::compile_physics(config).turbulence.stability_correction==tmv::kLillyRichardsonModel);
}

TEST_CASE("Typed ZFP controls retain the exact public tier token",
          "[init][config][zfp][compilation]")
{
    using tmv::init::config::OutputFormat;
    for (const auto& [format, token, tolerance] :
         {std::tuple<OutputFormat, const char*, double>{OutputFormat::ZfpLow, "zfp_low", 1.0e-2},
          {OutputFormat::ZfpMedium, "zfp_medium", 1.0e-3},
          {OutputFormat::ZfpHigh, "zfp_high", 1.0e-4}})
    {
        tmv::init::config::ConfigV1 config;
        config.time.run_time = 4.0;
        config.time.output_interval = 4.0;
        config.output.format = format;
        RuntimeConfigurationState runtime;
        const auto default_controls = tmv::configuration::compile_controls(config, runtime);
        REQUIRE(default_controls.output.format == ::OutputFormat::zfp);
        REQUIRE(default_controls.output.zfp_requested_format_tier == token);
        REQUIRE(default_controls.output.zfp_tolerance == tolerance);

        config.output.zfp_tolerance = 2.0e-5;
        const auto explicit_controls = tmv::configuration::compile_controls(config, runtime);
        REQUIRE(explicit_controls.output.zfp_requested_format_tier == token);
        REQUIRE(explicit_controls.output.zfp_tolerance == 2.0e-5);
    }

    tmv::init::config::ConfigV1 npy;
    RuntimeConfigurationState runtime;
    const auto controls = tmv::configuration::compile_controls(npy, runtime);
    REQUIRE(controls.output.zfp_requested_format_tier.empty());
}
