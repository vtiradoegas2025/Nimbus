#include "catch2/catch.hpp"
#include "compute/vulkan_pipeline_cache_signature.hpp"
#include "core/runtime/configuration_compiler.hpp"
#include "init/config/v1_loader.hpp"

TEST_CASE("Retained mixed dry YAML seals its Vulkan startup plan in the typed compiler",
          "[vulkan][startup-plan][loader]")
{
    const auto loaded = tmv::init::config::load_v1_config(
        "tests/fixtures/configs/startup/flat_mixed_vulkan.yaml");
    for (const auto& error : loaded.errors)
    {
        INFO(error.key << ": " << error.message);
    }
    REQUIRE(loaded.ok());
    REQUIRE(nimbus::compute::is_flat_dry_cgrid18_vulkan_startup_profile(*loaded.config));
    const auto compiled = tmv::configuration::compile_physics(*loaded.config);
    REQUIRE(compiled.compute.backend == "vulkan");
    REQUIRE_FALSE(compiled.compute.allow_fallback);
    REQUIRE(compiled.compute.startup_plan);
    REQUIRE(compiled.compute.startup_plan->size() == 15);
    std::string error;
    REQUIRE(compiled.compute.startup_plan->validate(nimbus::compute::kernel_manifest(), error));
    REQUIRE(compiled.compute.startup_plan->contains(nimbus::compute::KernelId::StateRelaxationCartesian));
    REQUIRE(loaded.config->damping.lateral_sponge_enabled);
    REQUIRE(loaded.config->damping.lateral_width == 20);
    REQUIRE(loaded.config->damping.lateral_alpha == 0.05);
    const auto scoped = tmv_vulkan::pipeline_cache_manifest_signature(nimbus::compute::kernel_manifest(),
                                                                      compiled.compute.startup_plan.get());
    REQUIRE(scoped != "eefb18304f5c2a4b");

    auto continuation                              = *loaded.config;
    continuation.time.run_time                     = 4;
    continuation.extensions.checkpoint.resume_from = "checkpoint_step_000000010.nimbus";
    const auto resumed                             = tmv::configuration::compile_physics(continuation);
    REQUIRE(resumed.compute.startup_plan);
    REQUIRE(resumed.compute.startup_plan->cache_identity() == compiled.compute.startup_plan->cache_identity());
    REQUIRE(tmv_vulkan::pipeline_cache_manifest_signature(nimbus::compute::kernel_manifest(),
                                                          resumed.compute.startup_plan.get()) == scoped);

    auto native_chaos                                        = *loaded.config;
    native_chaos.extensions.chaos.native_generation_required = true;
    REQUIRE_FALSE(tmv::configuration::compile_physics(native_chaos).compute.startup_plan);
    auto nested                          = *loaded.config;
    nested.domain_motion.nesting.enabled = true;
    REQUIRE_FALSE(tmv::configuration::compile_physics(nested).compute.startup_plan);
}
