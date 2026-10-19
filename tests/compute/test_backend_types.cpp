/**
 * @file test_backend_types.cpp
 * @brief Backend-neutral compute selection value-type tests.
 */

#include "catch2/catch.hpp"
#include "compute/backend_types.hpp"

#include <string>

TEST_CASE("Compute backend value types have conservative defaults",
          "[compute][backend][types]")
{
    const ComputeBackendConfig config;
    CHECK(config.backend == "cpu");
    CHECK(config.device_index == -1);
    CHECK(config.allow_fallback);
    CHECK_FALSE(config.validate_parity);

    const ComputeMemoryProperties memory;
    CHECK_FALSE(memory.accelerator);
    CHECK_FALSE(memory.unified_host_device_memory);
    CHECK(memory.device_local_heap_bytes == 0);
    CHECK(memory.max_storage_buffer_bytes == 0);
}

TEST_CASE("Compute backend names cover valid and invalid enum values",
          "[compute][backend][types]")
{
    CHECK(std::string(compute_backend_kind_name(ComputeBackendKind::Cpu)) ==
          "cpu");
    CHECK(std::string(compute_backend_kind_name(ComputeBackendKind::Vulkan)) ==
          "vulkan");
    CHECK(std::string(compute_backend_kind_name(ComputeBackendKind::Metal)) ==
          "metal");

    constexpr auto invalid_kind = static_cast<ComputeBackendKind>(-1);
    CHECK(std::string(compute_backend_kind_name(invalid_kind)) == "unknown");
}

TEST_CASE("Compute backend parser recognizes stable names and aliases",
          "[compute][backend][types]")
{
    ComputeBackendKind kind = ComputeBackendKind::Cpu;

    REQUIRE(parse_compute_backend_kind("CPU", kind));
    CHECK(kind == ComputeBackendKind::Cpu);
    REQUIRE(parse_compute_backend_kind("vulkan", kind));
    CHECK(kind == ComputeBackendKind::Vulkan);
    REQUIRE(parse_compute_backend_kind("vk", kind));
    CHECK(kind == ComputeBackendKind::Vulkan);
    REQUIRE(parse_compute_backend_kind("gpu", kind));
    CHECK(kind == ComputeBackendKind::Vulkan);
    REQUIRE(parse_compute_backend_kind("metal", kind));
    CHECK(kind == ComputeBackendKind::Metal);
    REQUIRE(parse_compute_backend_kind("mtl", kind));
    CHECK(kind == ComputeBackendKind::Metal);
}

TEST_CASE("Compute backend parser fails without changing the selection",
          "[compute][backend][types]")
{
    ComputeBackendKind kind = ComputeBackendKind::Metal;
    CHECK_FALSE(parse_compute_backend_kind("cuda", kind));
    CHECK(kind == ComputeBackendKind::Metal);
}
