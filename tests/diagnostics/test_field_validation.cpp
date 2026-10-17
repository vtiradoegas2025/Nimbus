/**
 * @file test_field_validation.cpp
 * @brief Unit tests for field validation with sanitized value correctness.
 *
 * Sanitize mode:
 *   NaN/Inf → replaced with clamp(0, bounds.min, bounds.max)
 *   Out-of-bounds → clamped to [bounds.min, bounds.max]
 */
#include "catch2/catch.hpp"
#include "core/field/field_bounds.hpp"
#include "core/field/numerical_integrity.hpp"
#include "core/field/field_sanitization.hpp"
#include "core/field/field3d.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/runtime/runtime_config.hpp"
#include "diagnostics/field_validation.hpp"
#include "diagnostics/field_contract.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

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

TEST_CASE("primary state verification policy separates fast and independent modes",
          "[diagnostics][validation][policy]")
{
    tmv::ValidationPolicy policy;

    policy.mode = tmv::GuardMode::Sanitize;
    REQUIRE_FALSE(primary_state_verification_due(policy, false, false));
    REQUIRE(primary_state_verification_due(policy, true, false));
    REQUIRE(primary_state_verification_due(policy, false, true));

    policy.mode = tmv::GuardMode::Warn;
    REQUIRE(primary_state_verification_due(policy, false, false));
    REQUIRE(primary_state_verification_due(policy, true, false));
    REQUIRE(primary_state_verification_due(policy, false, true));

    policy.mode = tmv::GuardMode::Strict;
    REQUIRE(primary_state_verification_due(policy, false, false));

    policy.mode = tmv::GuardMode::Off;
    REQUIRE_FALSE(primary_state_verification_due(policy, true, true));
}

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

TEST_CASE("non-finite report verdict is independent of guard mode",
          "[diagnostics][validation][integrity]")
{
    for (const tmv::GuardMode mode : {tmv::GuardMode::Off,
                                      tmv::GuardMode::Warn,
                                      tmv::GuardMode::Sanitize,
                                      tmv::GuardMode::Strict})
    {
        Field3D field(1, 1, 1, std::numeric_limits<float>::quiet_NaN());
        const auto contract = make_contract(0.0, 100.0);
        tmv::ValidationPolicy policy;
        policy.mode = mode;

        tmv::ValidationReport report;
        tmv::FieldValidationReport field_report;
        field_report.field_id = contract.id;
        field_report.result =
            tmv::validate_field3d_inplace(field, contract, policy, false);
        report.fields.push_back(std::move(field_report));

        INFO("guard mode=" << tmv::to_string(mode));
        REQUIRE(tmv::validation_report_has_nonfinite(report));
    }

    tmv::ValidationReport finite_report;
    tmv::FieldValidationReport finite_field;
    finite_field.result.stats.finite_count = 1;
    finite_report.fields.push_back(finite_field);
    REQUIRE_FALSE(tmv::validation_report_has_nonfinite(finite_report));
}

TEST_CASE("writer-owned non-finite repairs leave a numerical-integrity incident",
          "[diagnostics][validation][integrity]")
{
    tmv::reset_numerical_nonfinite_incidents();
    set_primary_state_repair_policy(true, false);

    REQUIRE(clamp_theta_k(900.0f) == theta_max_k);
    REQUIRE(tmv::numerical_nonfinite_incidents() == 0);

    REQUIRE(clamp_theta_k(std::numeric_limits<float>::quiet_NaN()) ==
            Approx(static_cast<float>(theta0)));
    REQUIRE(clamp_qv_kgkg(std::numeric_limits<float>::infinity()) == 0.0f);
    REQUIRE(tmv::numerical_nonfinite_incidents() == 2);
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

TEST_CASE("primary state bounds can sanitize an explicit local state", "[diagnostics][validation][state]")
{
    int local_nr = 1;
    int local_nth = 1;
    int local_nz = 1;
    double local_dr = 1.0;
    double local_dz = 1.0;
    double local_dt = 0.1;
    double local_dtheta = 1.0;
    GridGeometry local_geometry;

    Field3D local_rho(1, 1, 1, std::numeric_limits<float>::quiet_NaN());
    Field3D local_p(1, 1, 1, 10.0f);
    Field3D local_u(1, 1, 1, 0.0f);
    Field3D local_w(1, 1, 1, 0.0f);
    Field3D local_v(1, 1, 1, 0.0f);
    Field3D local_tracer(1, 1, 1, 0.0f);
    Field3D local_theta(1, 1, 1, 900.0f);
    Field3D local_qv(1, 1, 1, 0.2f);
    Field3D local_qc(1, 1, 1, -0.1f);
    Field3D local_qr(1, 1, 1, 0.1f);
    Field3D local_qi(1, 1, 1, 0.0f);
    Field3D local_qs(1, 1, 1, 0.0f);
    Field3D local_qh(1, 1, 1, 0.1f);
    Field3D local_qg(1, 1, 1, std::numeric_limits<float>::infinity());
    Field3D local_tke(1, 1, 1, 0.0f);
    Field3D local_dtheta_dt_rad(1, 1, 1, 0.0f);
    Field3D local_dtheta_dt_pbl(1, 1, 1, 0.0f);
    Field3D local_dqv_dt_pbl(1, 1, 1, 0.0f);
    Field3D local_du_dt_pbl(1, 1, 1, 0.0f);
    Field3D local_dv_dt_pbl(1, 1, 1, 0.0f);
    Field3D local_dtke_dt_pbl(1, 1, 1, 0.0f);

    std::vector<double> local_rho0;
    std::vector<double> local_p0;
    std::vector<double> local_qv0;
    std::vector<double> local_theta0;
    std::vector<double> local_u0;
    std::vector<double> local_v0;
    double local_coriolis = 0.0;
    double local_time = 0.0;

    SimulationStateView local_state{
        {local_nr, local_nth, local_nz, local_dr, local_dz, local_dt, local_dtheta,
         local_geometry},
        {local_rho, local_p, local_u, local_w, local_v, local_tracer,
         local_theta, local_qv, local_qc, local_qr, local_qi, local_qs,
         local_qh, local_qg, local_tke},
        {local_rho0, local_p0, local_qv0, local_theta0, local_u0, local_v0},
        {local_dtheta_dt_rad, local_dtheta_dt_pbl, local_dqv_dt_pbl,
         local_du_dt_pbl, local_dv_dt_pbl, local_dtke_dt_pbl},
        local_coriolis,
        local_time};

    const int corrected = enforce_primary_state_bounds(local_state, "test");

    REQUIRE(corrected == 8);
    REQUIRE(local_rho(0, 0, 0) == Approx(1.0f));
    REQUIRE(local_p(0, 0, 0) == Approx(pressure_min_pa));
    REQUIRE(local_theta(0, 0, 0) == Approx(theta_max_k));
    REQUIRE(local_qv(0, 0, 0) == Approx(qv_max_kgkg));
    REQUIRE(local_qc(0, 0, 0) == Approx(0.0f));
    REQUIRE(local_qr(0, 0, 0) == Approx(hydrometeor_max_kgkg));
    REQUIRE(local_qg(0, 0, 0) == Approx(0.0f));
    REQUIRE(local_qh(0, 0, 0) == Approx(hydrometeor_max_kgkg));
}

TEST_CASE("warn and strict primary-state verification never mutate",
          "[diagnostics][validation][state][warn][strict]")
{
    int nr = 1, nth = 1, nz = 1;
    double dx = 1.0, dz_local = 1.0, dt_local = 0.1, dy = 1.0;
    GridGeometry geometry;
    Field3D rho_local(1, 1, 1, 1.0f), p_local(1, 1, 1, 90000.0f);
    Field3D u_local(1, 1, 1, 0.0f), w_local(1, 1, 1, 0.0f), v_local(1, 1, 1, 0.0f);
    Field3D tracer_local(1, 1, 1, 0.0f), theta_local(1, 1, 1, 900.0f);
    Field3D qv_local(1, 1, 1, 0.01f), zero(1, 1, 1, 0.0f), tke_local(1, 1, 1, -1.0f);
    Field3D rad(1, 1, 1, 0.0f), pbl(1, 1, 1, 0.0f), qv_pbl(1, 1, 1, 0.0f);
    Field3D u_pbl(1, 1, 1, 0.0f), v_pbl(1, 1, 1, 0.0f), tke_pbl(1, 1, 1, 0.0f);
    std::vector<double> base;
    double coriolis = 0.0, time = 0.0;
    SimulationStateView state{
        {nr, nth, nz, dx, dz_local, dt_local, dy, geometry},
        {rho_local, p_local, u_local, w_local, v_local, tracer_local,
         theta_local, qv_local, zero, zero, zero, zero, zero, zero, tke_local},
        {base, base, base, base, base, base},
        {rad, pbl, qv_pbl, u_pbl, v_pbl, tke_pbl},
        coriolis, time};

    const tmv::ValidationPolicy saved = global_validation_policy;
    SECTION("warn observes even without diagnostics or debug logging")
    {
        global_validation_policy.mode = tmv::GuardMode::Warn;
        global_validation_policy.fail_on = tmv::GuardFailOn::Both;
        REQUIRE(primary_state_verification_due(
            global_validation_policy, false, false));
        REQUIRE(enforce_primary_state_bounds(state, "warn-test") == 0);
        REQUIRE(theta_local[0][0][0] == 900.0f);
        REQUIRE(tke_local[0][0][0] == -1.0f);
        REQUIRE(clamp_theta_k(900.0f) == 900.0f);
    }

    SECTION("strict rejects")
    {
        global_validation_policy.mode = tmv::GuardMode::Strict;
        global_validation_policy.fail_on = tmv::GuardFailOn::Both;
        REQUIRE_THROWS_AS(
            enforce_primary_state_bounds(state, "strict-test"),
            std::runtime_error);
        REQUIRE(theta_local[0][0][0] == 900.0f);
        REQUIRE(tke_local[0][0][0] == -1.0f);
        REQUIRE(clamp_theta_k(900.0f) == 900.0f);
    }
    global_validation_policy = saved;
    set_primary_state_repair_policy(
        saved.mode == tmv::GuardMode::Sanitize || saved.mode == tmv::GuardMode::Off,
        saved.mode == tmv::GuardMode::Strict);
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

TEST_CASE("validation run summary aggregates report counts",
          "[diagnostics][validation][summary]")
{
    tmv::ValidationPolicy policy;
    policy.mode = tmv::GuardMode::Sanitize;
    policy.fail_on = tmv::GuardFailOn::Both;
    policy.strict_scope = tmv::StrictGuardScope::RequiredOnly;

    tmv::FieldValidationReport field_report;
    field_report.field_id = "theta";
    field_report.result.stats.nan_count = 2;
    field_report.result.stats.inf_count = 1;
    field_report.result.stats.below_min_count = 3;
    field_report.result.stats.above_max_count = 4;
    field_report.result.stats.sanitized_nonfinite_count = 3;
    field_report.result.stats.sanitized_bounds_count = 7;
    field_report.result.violations.push_back({"theta", "nonfinite", 3, true});

    tmv::ValidationReport clean_report;
    clean_report.context = "pre_export";

    tmv::ValidationReport failed_report;
    failed_report.context = "pre_export";
    failed_report.failed = true;
    failed_report.fields.push_back(field_report);

    const std::string json = tmv::validation_run_summary_to_json(
        {clean_report, failed_report},
        policy);

    REQUIRE(json.find("\"context\": \"run_summary\"") != std::string::npos);
    REQUIRE(json.find("\"guard_mode\": \"sanitize\"") != std::string::npos);
    REQUIRE(json.find("\"report_count\": 2") != std::string::npos);
    REQUIRE(json.find("\"failed_report_count\": 1") != std::string::npos);
    REQUIRE(json.find("\"total_violations\": 1") != std::string::npos);
    REQUIRE(json.find("\"total_nonfinite\": 3") != std::string::npos);
    REQUIRE(json.find("\"total_bounds\": 7") != std::string::npos);
    REQUIRE(json.find("\"total_sanitized_nonfinite\": 3") != std::string::npos);
    REQUIRE(json.find("\"total_sanitized_bounds\": 7") != std::string::npos);
    REQUIRE(json.find("\"failed\": true") != std::string::npos);
}

TEST_CASE("missing pressure diagnostics are distinct from invalid model state",
          "[diagnostics][validation][missing]")
{
    const auto* optional = tmv::find_field_contract("lifted_index");
    const auto* required = tmv::find_field_contract("theta");
    REQUIRE(optional != nullptr);
    REQUIRE(required != nullptr);
    tmv::ValidationPolicy policy;
    policy.mode = tmv::GuardMode::Strict;
    policy.strict_scope = tmv::StrictGuardScope::ExportedNow;
    float values[] = {1.0f, std::numeric_limits<float>::quiet_NaN()};
    const auto missing = tmv::validate_buffer_inplace(values, 2, *optional, policy, false);
    REQUIRE_FALSE(missing.failed);
    REQUIRE(missing.stats.missing_count == 1);
    REQUIRE(missing.stats.nan_count == 0);
    REQUIRE(tmv::validate_buffer_inplace(values, 2, *required, policy, false).failed);
    values[1] = std::numeric_limits<float>::infinity();
    REQUIRE(tmv::validate_buffer_inplace(values, 2, *optional, policy, false).failed);
    values[1] = std::numeric_limits<float>::quiet_NaN();
    policy.mode = tmv::GuardMode::Sanitize;
    const auto sanitized = tmv::validate_buffer_inplace(values, 2, *optional, policy, false);
    REQUIRE(sanitized.stats.sanitized_nonfinite_count == 0);
    REQUIRE(std::isnan(values[1]));
}
