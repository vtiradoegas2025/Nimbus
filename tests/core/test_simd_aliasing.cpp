/**
 * @file test_simd_aliasing.cpp
 * @brief Optimized-build regression tests for supported exact SIMD aliasing.
 */

#include "catch2/catch.hpp"
#include "util/simd_utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace
{

std::vector<float> sequence(int count, float offset)
{
    std::vector<float> values(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        values[static_cast<std::size_t>(i)] = offset + static_cast<float>(i) * 0.25f;
    }
    return values;
}

void require_values(const std::vector<float>& actual,
                    const std::vector<float>& expected)
{
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        REQUIRE(actual[i] == Approx(expected[i]));
    }
}

} // namespace

TEST_CASE("SIMD implementation selection maps compiler capabilities honestly",
          "[core][simd][selection][cpu025]")
{
    using simd_utils::SIMDType;
    using simd_utils::select_simd_implementation;
    using simd_utils::select_runtime_simd;
    using simd_utils::simd_implementation_width;
    using simd_utils::SIMDFeatures;

    REQUIRE(select_simd_implementation(true, false, false, false) ==
            SIMDType::AVX);
    REQUIRE(select_simd_implementation(true, true, true, false) ==
            SIMDType::AVX);
    REQUIRE(select_simd_implementation(false, true, true, false) ==
            SIMDType::AVX);
    REQUIRE(select_simd_implementation(false, false, true, false) ==
            SIMDType::SSE);
    REQUIRE(select_simd_implementation(false, false, false, true) ==
            SIMDType::NEON);
    REQUIRE(select_simd_implementation(false, false, false, false) ==
            SIMDType::NONE);

    // AVX-512 compiler capability currently selects the tested AVX-width
    // implementation, so hardware reporting must advertise eight active lanes.
    REQUIRE(simd_implementation_width(
                select_simd_implementation(true, false, false, false)) == 8);
    REQUIRE(simd_implementation_width(simd_utils::get_available_simd()) ==
            SIMD_WIDTH);

    const SIMDFeatures all{true, true, true, true};
    REQUIRE(select_runtime_simd(all, {false, true, true, false}) ==
            SIMDType::AVX);
    REQUIRE(select_runtime_simd(all, {false, false, true, false}) ==
            SIMDType::SSE);
    REQUIRE(select_runtime_simd(all, {false, false, false, true}) ==
            SIMDType::NEON);
    REQUIRE(select_runtime_simd(
                {false, false, true, false},
                {true, true, true, false}) == SIMDType::SSE);
    REQUIRE(select_runtime_simd(
                {false, true, false, false},
                {false, false, true, false}) == SIMDType::NONE);

    const SIMDFeatures compiled = simd_utils::get_compiled_simd_features();
    const SIMDFeatures runtime = simd_utils::get_runtime_simd_features();
    REQUIRE(simd_utils::get_available_simd() ==
            select_runtime_simd(compiled, runtime));
}

TEST_CASE("SIMD subtraction supports every exact in-place alias pattern",
          "[core][simd][aliasing]")
{
    constexpr int count = 37; // exercise vector lanes and the scalar tail
    const std::vector<float> original_a = sequence(count, 3.0f);
    const std::vector<float> original_b = sequence(count, -2.0f);
    std::vector<float> expected(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        expected[static_cast<std::size_t>(i)] =
            original_a[static_cast<std::size_t>(i)]
            - original_b[static_cast<std::size_t>(i)];
    }

    SECTION("result aliases the left input")
    {
        std::vector<float> a = original_a;
        simd_utils::subtract_vectors(a.data(), original_b.data(), a.data(), count);
        require_values(a, expected);
    }

    SECTION("result aliases the right input")
    {
        std::vector<float> b = original_b;
        simd_utils::subtract_vectors(original_a.data(), b.data(), b.data(), count);
        require_values(b, expected);
    }

    SECTION("both inputs and result are identical")
    {
        std::vector<float> values = original_a;
        simd_utils::subtract_vectors(values.data(), values.data(), values.data(), count);
        for (float value : values)
        {
            REQUIRE(value == 0.0f);
        }
    }
}

TEST_CASE("SIMD scale supports exact in-place aliasing",
          "[core][simd][aliasing]")
{
    constexpr int count = 37;
    std::vector<float> values = sequence(count, -4.0f);
    std::vector<float> expected = values;
    for (float& value : expected)
    {
        value *= -1.75f;
    }

    simd_utils::scale_vectors(-1.75f, values.data(), values.data(), count);
    require_values(values, expected);
}

TEST_CASE("SIMD sanitization supports exact in-place aliasing",
          "[core][simd][aliasing]")
{
    std::vector<float> values = sequence(19, 1.0f);
    values[0] = std::numeric_limits<float>::quiet_NaN();
    values[7] = std::numeric_limits<float>::infinity();
    values[18] = -std::numeric_limits<float>::infinity();

    const int replaced = simd_utils::sanitize_nonfinite(
        values.data(), -9.5f, values.data(), static_cast<int>(values.size()));

    REQUIRE(replaced == 3);
    REQUIRE(values[0] == -9.5f);
    REQUIRE(values[7] == -9.5f);
    REQUIRE(values[18] == -9.5f);
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        REQUIRE(std::isfinite(values[i]));
    }
}
