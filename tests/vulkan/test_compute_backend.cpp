/**
 * @file test_compute_backend.cpp
 * @brief Unit tests for compute backend configuration and kind parsing.
 */
#include "catch2/catch.hpp"
#include "compute/compute_backend.hpp"

TEST_CASE("parse_compute_backend_kind recognizes cpu", "[vulkan][backend]")
{
    ComputeBackendKind kind;
    REQUIRE(parse_compute_backend_kind("cpu", kind));
    REQUIRE(kind == ComputeBackendKind::Cpu);
}

TEST_CASE("parse_compute_backend_kind recognizes vulkan", "[vulkan][backend]")
{
    ComputeBackendKind kind;
    REQUIRE(parse_compute_backend_kind("vulkan", kind));
    REQUIRE(kind == ComputeBackendKind::Vulkan);
}

TEST_CASE("parse_compute_backend_kind rejects unknown", "[vulkan][backend]")
{
    ComputeBackendKind kind;
    REQUIRE_FALSE(parse_compute_backend_kind("cuda", kind));
}

TEST_CASE("compute_backend_kind_name returns non-empty", "[vulkan][backend]")
{
    REQUIRE(std::string(compute_backend_kind_name(ComputeBackendKind::Cpu)) != "");
    REQUIRE(std::string(compute_backend_kind_name(ComputeBackendKind::Vulkan)) != "");
}

TEST_CASE("ComputeBackendConfig defaults", "[vulkan][backend]")
{
    ComputeBackendConfig cfg;
    REQUIRE(cfg.backend == "cpu");
    REQUIRE(cfg.allow_fallback == true);
}
