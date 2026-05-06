/**
 * @file test_field_validation.cpp
 * @brief Unit tests for field validation with sanitized value correctness.
 *
 * Sanitize mode:
 *   NaN/Inf → replaced with clamp(0, bounds.min, bounds.max)
 *   Out-of-bounds → clamped to [bounds.min, bounds.max]
 */
#include "catch2/catch.hpp"
#include "diagnostics/field_validation.hpp"
#include "diagnostics/field_contract.hpp"
#include "core/field/field3d.hpp"

#include <cmath>
#include <limits>

namespace
{

tmv::FieldContract make_contract(double lo, double hi)
{
    tmv::FieldContract c;
    c.id = "test_field";
    c.default_bounds.has_min = true;
    c.default_bounds.has_max = true;
    c.default_bounds.min_value = lo;
    c.default_bounds.max_value = hi;
    c.status = tmv::FieldImplementationStatus::ExportedNow;
    c.severity.check_nonfinite = true;
    c.severity.check_bounds = true;
    return c;
}

} // namespace

// ---- NaN/Inf detection ----

TEST_CASE("validate detects NaN count", "[diagnostics][validation][analytical]")
{
    Field3D f(4, 4, 4, 5.0f);
    f(0, 0, 0) = std::numeric_limits<float>::quiet_NaN();
    f(1, 1, 1) = std::numeric_limits<float>::quiet_NaN();
    f(2, 2, 2) = std::numeric_limits<float>::quiet_NaN();

    auto contract = make_contract(0.0, 100.0);
    tmv::ValidationPolicy policy;
    policy.mode = tmv::GuardMode::Sanitize;

    auto result = tmv::validate_field3d_inplace(f, contract, policy, false);
    REQUIRE(result.stats.nan_count == 3);
    REQUIRE(result.stats.sanitized_nonfinite_count == 3);
}

TEST_CASE("validate detects Inf count", "[diagnostics][validation][analytical]")
{
    Field3D f(4, 4, 4, 5.0f);
    f(0, 0, 0) = std::numeric_limits<float>::infinity();
    f(1, 1, 1) = -std::numeric_limits<float>::infinity();

    auto contract = make_contract(0.0, 100.0);
    tmv::ValidationPolicy policy;
    policy.mode = tmv::GuardMode::Sanitize;

    auto result = tmv::validate_field3d_inplace(f, contract, policy, false);
    REQUIRE(result.stats.inf_count == 2);
}

// ---- Sanitize mode: NaN replacement value ----

TEST_CASE("sanitize replaces NaN with clamped zero", "[diagnostics][validation][analytical]")
{
    // With bounds [0, 100]: replacement = clamp(0, 0, 100) = 0
    Field3D f(2, 2, 2, 50.0f);
    f(0, 0, 0) = std::numeric_limits<float>::quiet_NaN();

    auto contract = make_contract(0.0, 100.0);
    tmv::ValidationPolicy policy;
    policy.mode = tmv::GuardMode::Sanitize;

    tmv::validate_field3d_inplace(f, contract, policy, false);

    REQUIRE(std::isfinite(f(0, 0, 0)));
    REQUIRE(f(0, 0, 0) == Approx(0.0f).margin(1e-6));
    // Other values untouched
    REQUIRE(f(1, 1, 1) == 50.0f);
}

TEST_CASE("sanitize replaces NaN with min when 0 < min", "[diagnostics][validation][analytical]")
{
    // With bounds [10, 100]: replacement = clamp(0, 10, 100) = 10
    Field3D f(2, 2, 2, 50.0f);
    f(0, 0, 0) = std::numeric_limits<float>::quiet_NaN();

    auto contract = make_contract(10.0, 100.0);
    tmv::ValidationPolicy policy;
    policy.mode = tmv::GuardMode::Sanitize;

    tmv::validate_field3d_inplace(f, contract, policy, false);

    REQUIRE(std::isfinite(f(0, 0, 0)));
    REQUIRE(f(0, 0, 0) >= 10.0f);
    REQUIRE(f(0, 0, 0) <= 100.0f);
}

// ---- Sanitize mode: out-of-bounds clamping ----

TEST_CASE("sanitize clamps values above max", "[diagnostics][validation][analytical]")
{
    Field3D f(2, 2, 2, 50.0f);
    f(0, 0, 0) = 200.0f;  // above max

    auto contract = make_contract(0.0, 100.0);
    tmv::ValidationPolicy policy;
    policy.mode = tmv::GuardMode::Sanitize;

    auto result = tmv::validate_field3d_inplace(f, contract, policy, false);

    REQUIRE(f(0, 0, 0) == Approx(100.0f).margin(1e-6));
    REQUIRE(result.stats.sanitized_bounds_count >= 1);
}

TEST_CASE("sanitize clamps values below min", "[diagnostics][validation][analytical]")
{
    Field3D f(2, 2, 2, 50.0f);
    f(0, 0, 0) = -50.0f;  // below min

    auto contract = make_contract(0.0, 100.0);
    tmv::ValidationPolicy policy;
    policy.mode = tmv::GuardMode::Sanitize;

    auto result = tmv::validate_field3d_inplace(f, contract, policy, false);

    REQUIRE(f(0, 0, 0) == Approx(0.0f).margin(1e-6));
    REQUIRE(result.stats.sanitized_bounds_count >= 1);
}

// ---- Clean data ----

TEST_CASE("validate clean data has zero violations", "[diagnostics][validation][analytical]")
{
    Field3D f(4, 4, 4, 50.0f);
    auto contract = make_contract(0.0, 100.0);
    tmv::ValidationPolicy policy;
    policy.mode = tmv::GuardMode::Off;

    auto result = tmv::validate_field3d_inplace(f, contract, policy, false);
    REQUIRE(result.stats.nan_count == 0);
    REQUIRE(result.stats.inf_count == 0);
    REQUIRE(result.stats.below_min_count == 0);
    REQUIRE(result.stats.above_max_count == 0);
    REQUIRE(result.stats.finite_count == 64);
    REQUIRE(result.stats.total_count == 64);
}

// ---- Statistics ----

TEST_CASE("validate computes correct min/max/mean", "[diagnostics][validation][analytical]")
{
    Field3D f(2, 2, 2, 0.0f);
    f(0, 0, 0) = 10.0f;
    f(1, 1, 1) = 30.0f;
    // Other 6 values are 0

    auto contract = make_contract(-1000.0, 1000.0);
    tmv::ValidationPolicy policy;
    policy.mode = tmv::GuardMode::Off;

    auto result = tmv::validate_field3d_inplace(f, contract, policy, false);
    REQUIRE(result.stats.min_value == Approx(0.0));
    REQUIRE(result.stats.max_value == Approx(30.0));
    // Mean: (10 + 30 + 0*6) / 8 = 5.0
    REQUIRE(result.stats.mean_value == Approx(5.0).margin(0.01));
}

// ---- Buffer API ----

TEST_CASE("validate_buffer_inplace counts out-of-bounds correctly", "[diagnostics][validation][analytical]")
{
    std::vector<float> data = {-10.0f, 5.0f, 50.0f, 105.0f, 200.0f};

    auto contract = make_contract(0.0, 100.0);
    tmv::ValidationPolicy policy;
    policy.mode = tmv::GuardMode::Sanitize;

    auto result = tmv::validate_buffer_inplace(data.data(), data.size(), contract, policy, false);

    // -10 is below min, 105 and 200 are above max
    REQUIRE(result.stats.below_min_count >= 1);
    REQUIRE(result.stats.above_max_count >= 2);

    // After sanitize: values should be clamped
    REQUIRE(data[0] == Approx(0.0f).margin(1e-6));
    REQUIRE(data[1] == 5.0f);  // untouched
    REQUIRE(data[2] == 50.0f); // untouched
    REQUIRE(data[3] == Approx(100.0f).margin(1e-6));
    REQUIRE(data[4] == Approx(100.0f).margin(1e-6));
}

// ---- Guard mode parsing ----

TEST_CASE("parse_guard_mode round-trips all modes", "[diagnostics][validation]")
{
    tmv::GuardMode mode;

    REQUIRE(tmv::parse_guard_mode("sanitize", mode));
    REQUIRE(mode == tmv::GuardMode::Sanitize);

    REQUIRE(tmv::parse_guard_mode("off", mode));
    REQUIRE(mode == tmv::GuardMode::Off);

    REQUIRE(tmv::parse_guard_mode("strict", mode));
    REQUIRE(mode == tmv::GuardMode::Strict);
}

TEST_CASE("to_string returns non-empty for all guard modes", "[diagnostics][validation]")
{
    REQUIRE(std::string(tmv::to_string(tmv::GuardMode::Sanitize)) != "");
    REQUIRE(std::string(tmv::to_string(tmv::GuardMode::Off)) != "");
    REQUIRE(std::string(tmv::to_string(tmv::GuardMode::Strict)) != "");
}
