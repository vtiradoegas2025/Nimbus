/**
 * @file test_memory_admission.cpp
 * @brief Backend-neutral accelerator memory-admission input contracts.
 */

#include "catch2/catch.hpp"
#include "compute/compute_memory_admission.hpp"

#include <array>
#include <limits>

namespace
{

ComputeMemoryProperties accelerator_properties()
{
    ComputeMemoryProperties properties;
    properties.accelerator = true;
    return properties;
}

} // namespace

TEST_CASE("Compute memory hierarchy rejects a missing non-empty level array",
          "[compute][memory][admission]")
{
    const auto result = assess_compute_memory_hierarchy(
        accelerator_properties(), nullptr, 1);

    CHECK(result.accelerator);
    CHECK(result.resident_level_count == 1);
    CHECK(result.invalid_input);
    CHECK_FALSE(result.arithmetic_saturated);
    CHECK(result.field_bytes == 0);
    CHECK(result.device_residency_bytes == 0);
    CHECK(compute_memory_admission_should_reject(result, false));
    CHECK(compute_memory_admission_should_reject(result, true));
}

TEST_CASE("Compute memory hierarchy rejects every nonpositive dimension",
          "[compute][memory][admission]")
{
    const std::array<ComputeMemoryLevelDimensions, 6> invalid_levels{{
        {-1, 8, 8},
        {8, -1, 8},
        {8, 8, -1},
        {0, 8, 8},
        {8, 0, 8},
        {8, 8, 0},
    }};

    for (const auto& level : invalid_levels)
    {
        const auto result = assess_compute_memory_hierarchy(
            accelerator_properties(), &level, 1);
        CHECK(result.invalid_input);
        CHECK_FALSE(result.arithmetic_saturated);
        CHECK(result.resident_level_count == 1);
        CHECK(result.field_bytes == 0);
        CHECK(compute_memory_admission_should_reject(result, false));
    }
}

TEST_CASE("Compute memory hierarchy permits an explicitly empty hierarchy",
          "[compute][memory][admission]")
{
    const auto result = assess_compute_memory_hierarchy(
        accelerator_properties(), nullptr, 0);

    CHECK(result.accelerator);
    CHECK(result.resident_level_count == 0);
    CHECK_FALSE(result.invalid_input);
    CHECK_FALSE(result.arithmetic_saturated);
    CHECK(result.field_bytes == 0);
    CHECK(result.host_residency_bytes == 0);
    CHECK(result.device_residency_bytes == 0);
    CHECK_FALSE(compute_memory_admission_should_reject(result, true));
}

TEST_CASE("Compute memory hierarchy rejects saturated working-set arithmetic",
          "[compute][memory][admission]")
{
    constexpr int maximum = std::numeric_limits<int>::max();
    const ComputeMemoryLevelDimensions level{maximum, maximum, maximum};
    const auto result = assess_compute_memory_hierarchy(
        accelerator_properties(), &level, 1);

    CHECK_FALSE(result.invalid_input);
    CHECK(result.arithmetic_saturated);
    CHECK(result.field_bytes == std::numeric_limits<std::uint64_t>::max());
    CHECK(result.host_residency_bytes ==
          std::numeric_limits<std::uint64_t>::max());
    CHECK(result.device_residency_bytes ==
          std::numeric_limits<std::uint64_t>::max());
    CHECK(compute_memory_admission_should_reject(result, false));
    CHECK(compute_memory_admission_should_reject(result, true));
}

TEST_CASE("Compute memory hierarchy retains valid budget behavior",
          "[compute][memory][admission]")
{
    auto properties = accelerator_properties();
    properties.device_local_heap_bytes = 10'000;
    properties.max_storage_buffer_bytes = 1'000;
    const ComputeMemoryLevelDimensions level{8, 8, 1};
    const auto result = assess_compute_memory_hierarchy(
        properties, &level, 1, 4, 80);

    CHECK_FALSE(result.invalid_input);
    CHECK_FALSE(result.arithmetic_saturated);
    CHECK(result.field_bytes == 256);
    CHECK(result.device_residency_bytes == 1'024);
    CHECK(result.device_budget_bytes == 8'000);
    CHECK_FALSE(result.exceeds_storage_buffer_range);
    CHECK_FALSE(result.exceeds_device_budget);
    CHECK_FALSE(compute_memory_admission_should_reject(result, true));
}

TEST_CASE("Live accelerator budget admits against available memory before allocation",
          "[compute][memory][admission]")
{
    auto properties = accelerator_properties();
    properties.device_local_heap_bytes = 10'000;
    properties.device_budget_is_live = true;
    properties.device_available_budget_bytes = 1'000;
    const auto result = assess_compute_memory(properties, 8, 8, 1, 4);
    CHECK(result.device_budget_is_live);
    CHECK(result.device_heap_known);
    CHECK(result.device_budget_bytes == 800);
    CHECK(result.device_residency_bytes == 1'024);
    CHECK(result.exceeds_device_budget);
    CHECK(compute_memory_admission_should_reject(result, true));
    CHECK_FALSE(compute_memory_admission_should_reject(result, false));

    properties.device_available_budget_bytes = 2'000;
    const auto freed = assess_compute_memory(properties, 8, 8, 1, 4);
    CHECK(freed.device_budget_bytes == 1'600);
    CHECK_FALSE(compute_memory_admission_should_reject(freed, true));
}

TEST_CASE("Zero live accelerator budget is known exhaustion rather than unknown capacity",
          "[compute][memory][admission]")
{
    auto properties = accelerator_properties();
    properties.device_budget_is_live = true;
    const auto result = assess_compute_memory(properties, 1, 1, 1);
    CHECK(result.device_heap_known);
    CHECK(result.device_budget_bytes == 0);
    CHECK(result.exceeds_device_budget);
    CHECK(compute_memory_admission_should_reject(result, true));
    properties.device_budget_is_live = false;
    CHECK_FALSE(assess_compute_memory(properties, 1, 1, 1).device_heap_known);
}

TEST_CASE("Live accelerator budget never enlarges physical capacity and counts all levels",
          "[compute][memory][admission]")
{
    auto properties = accelerator_properties();
    properties.device_local_heap_bytes = 1'000;
    properties.device_budget_is_live = true;
    properties.device_available_budget_bytes = std::numeric_limits<std::uint64_t>::max();
    const ComputeMemoryLevelDimensions levels[] = {{8, 8, 1}, {8, 8, 1}};
    const auto result = assess_compute_memory_hierarchy(properties, levels, 2, 2);
    CHECK(result.device_budget_bytes == 800);
    CHECK(result.device_residency_bytes == 1'024);
    CHECK(compute_memory_admission_should_reject(result, true));
    properties.device_local_heap_bytes = 0;
    properties.device_available_budget_bytes = 1'280;
    const auto exact = assess_compute_memory_hierarchy(properties, levels, 2, 2);
    CHECK(exact.device_budget_bytes == 1'024);
    CHECK_FALSE(compute_memory_admission_should_reject(exact, true));
}
