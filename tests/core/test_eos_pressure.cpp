#include "catch2/catch.hpp"
#include "core/field/field_bounds.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/simulation_state.hpp"
#include "dynamics/eos_pressure.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace
{

std::uint32_t float_bits(float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

template <typename Evaluate>
double time_eos(const std::vector<float>& rho,
                const std::vector<float>& theta,
                Evaluate evaluate,
                volatile double& checksum)
{
    constexpr int repeats = 8;
    const auto started = std::chrono::steady_clock::now();
    double sum = 0.0;
    for (int repeat = 0; repeat < repeats; ++repeat)
    {
        for (std::size_t i = 0; i < rho.size(); ++i)
        {
            sum += static_cast<double>(evaluate(rho[i], theta[i]));
        }
    }
    checksum = sum;
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
}

} // namespace

TEST_CASE("CPU-028 fp32 EOS experiment has a measured stored-pressure error bound",
          "[core][eos][cpu028]")
{
    SECTION("ordinary builds retain the fp64 production closure")
    {
        REQUIRE(tmv::dynamics_eos::pressure_selected(1.2f, 300.0f) ==
                tmv::dynamics_eos::pressure_fp64(1.2f, 300.0f));
        REQUIRE(tmv::dynamics_eos::pressure_selected(0.4f, 375.0f) ==
                tmv::dynamics_eos::pressure_fp64(0.4f, 375.0f));
    }

    constexpr std::size_t count = 1u << 20;
    std::vector<float> rho(count);
    std::vector<float> theta(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        const double fraction = static_cast<double>(i) /
                                static_cast<double>(count - 1);
        rho[i] = static_cast<float>(0.08 + 1.17 * fraction);
        // Decorrelate theta from rho while covering the model's deep-column
        // range; both inputs match the production fields' stored precision.
        const std::size_t permuted = (i * 65537u) & (count - 1u);
        const double theta_fraction = static_cast<double>(permuted) /
                                      static_cast<double>(count - 1);
        theta[i] = static_cast<float>(280.0 + 220.0 * theta_fraction);
    }

    std::uint32_t max_ulp = 0;
    double max_absolute_pa = 0.0;
    double max_relative = 0.0;
    std::size_t exact_stored_matches = 0;
    bool all_reference_finite = true;
    bool all_candidate_finite = true;
    for (std::size_t i = 0; i < count; ++i)
    {
        const float reference = static_cast<float>(
            tmv::dynamics_eos::pressure_fp64(rho[i], theta[i]));
        const float candidate = tmv::dynamics_eos::pressure_fp32(
            rho[i], theta[i]);
        all_reference_finite = all_reference_finite && std::isfinite(reference);
        all_candidate_finite = all_candidate_finite && std::isfinite(candidate);
        const std::uint32_t reference_bits = float_bits(reference);
        const std::uint32_t candidate_bits = float_bits(candidate);
        const std::uint32_t ulp = reference_bits > candidate_bits
            ? reference_bits - candidate_bits
            : candidate_bits - reference_bits;
        max_ulp = std::max(max_ulp, ulp);
        if (ulp == 0) ++exact_stored_matches;
        const double absolute = std::abs(
            static_cast<double>(candidate) - static_cast<double>(reference));
        max_absolute_pa = std::max(max_absolute_pa, absolute);
        max_relative = std::max(max_relative,
            absolute / std::max(1.0, std::abs(static_cast<double>(reference))));
    }

    volatile double reference_checksum = 0.0;
    volatile double candidate_checksum = 0.0;
    const double reference_s = time_eos(
        rho, theta, tmv::dynamics_eos::pressure_fp64, reference_checksum);
    const double candidate_s = time_eos(
        rho, theta, tmv::dynamics_eos::pressure_fp32, candidate_checksum);

    std::cout << "[CPU-028 EOS] samples=" << count
              << ", exact_stored=" << exact_stored_matches
              << ", max_ulp=" << max_ulp
              << ", max_abs_pa=" << max_absolute_pa
              << ", max_relative=" << max_relative
              << ", fp64_s=" << reference_s
              << ", fp32_s=" << candidate_s
              << ", isolated_speedup=" << reference_s / candidate_s
              << std::endl;

    REQUIRE(reference_checksum > 0.0);
    REQUIRE(candidate_checksum > 0.0);
    REQUIRE(all_reference_finite);
    REQUIRE(all_candidate_finite);
    // Qualification gate for proceeding to end-to-end experiments. This is
    // deliberately a stored-float bound, not permission to enable fp32 EOS.
    REQUIRE(max_ulp <= 4);
    REQUIRE(max_relative <= 5.0e-7);
}

TEST_CASE("EOS pressure projection commits the FP64 closure and exact basis-v2 energy",
          "[core][eos][energy][SCI-154]")
{
    constexpr int nr = 5;
    constexpr int nth = 5;
    constexpr int nz = 4;
    SimulationStorageState storage{{nr, nth, nz, 120.0, 80.0, 0.5, 60.0}};
    storage.grid.geometry.initialize(
        nr, nth, nz, 120.0, 80.0, 60.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto initialize = [&](Field3D& field, float value)
    {
        field.resize(nr, nth, nz, value);
    };
    initialize(storage.fields.u, 2.0f);
    initialize(storage.fields.v, -1.0f);
    initialize(storage.fields.w, 0.5f);
    initialize(storage.fields.rho, 1.05f);
    initialize(storage.fields.p, 90000.0f);
    initialize(storage.fields.theta, 305.0f);
    initialize(storage.fields.qv, 0.012f);
    initialize(storage.fields.qc, 0.001f);
    initialize(storage.fields.qr, 0.0005f);
    initialize(storage.fields.qi, 0.0004f);
    initialize(storage.fields.qs, 0.0003f);
    initialize(storage.fields.qg, 0.0002f);
    initialize(storage.fields.qh, 0.0001f);
    initialize(storage.fields.tke, 0.25f);

    SimulationStateView state = make_simulation_state_view(storage);
    const auto domain = physical_scalar_domain(state.grid);
    const tmv::physical_grid_metrics::ControlVolumeMassView masses(state);
    constexpr double cv_d =
        physical_constants::specific_heat_cp_jkgk
        - physical_constants::gas_constant_dry_air_jkgk;
    constexpr double cv_v =
        physical_constants::specific_heat_water_vapor_cp_jkgk
        - physical_constants::gas_constant_water_vapor_jkgk;
    constexpr double c_l =
        physical_constants::specific_heat_liquid_water_jkgk;
    constexpr double c_i =
        physical_constants::specific_heat_ice_jkgk;
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk
        / physical_constants::specific_heat_cp_jkgk;
    double expected_dry = 0.0;
    double expected_water = 0.0;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const float rho = storage.fields.rho[i][j][k];
                const float theta = storage.fields.theta[i][j][k];
                const float pressure_new = clamp_pressure_pa(
                    static_cast<float>(
                        tmv::dynamics_eos::pressure_fp64(rho, theta)));
                const double old_temperature =
                    static_cast<double>(theta) * std::pow(
                        90000.0
                            / physical_constants::reference_pressure_pa,
                        kappa);
                const double new_temperature =
                    static_cast<double>(theta) * std::pow(
                        static_cast<double>(pressure_new)
                            / physical_constants::reference_pressure_pa,
                        kappa);
                const double mass =
                    masses.scalar_mass_kg(storage.fields.rho, i, j, k);
                const double delta_temperature =
                    new_temperature - old_temperature;
                expected_dry += mass * cv_d * delta_temperature;
                expected_water += mass * delta_temperature
                    * (0.012 * cv_v + 0.0015 * c_l + 0.001 * c_i);
            }

    const auto report =
        tmv::dynamics_eos::apply_pressure_projection(state);
    REQUIRE(report.complete);
    CHECK(report.dry_internal_change_j == Approx(expected_dry));
    CHECK(report.water_sensible_change_j == Approx(expected_water));
    CHECK(report.phase_reference_change_j == 0.0);
    CHECK(report.carrier_kinetic_change_j == 0.0);
    CHECK(report.gravitational_change_j == 0.0);
    CHECK(report.subgrid_tke_change_j == 0.0);
    CHECK(report.committed_energy_change_j
          == Approx(expected_dry + expected_water));
    CHECK(report.closure_residual_j == 0.0);

    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const float expected = clamp_pressure_pa(
                    static_cast<float>(
                        tmv::dynamics_eos::pressure_fp64(
                            storage.fields.rho[i][j][k],
                            storage.fields.theta[i][j][k])));
                CHECK(storage.fields.p[i][j][k] == expected);
            }
}

TEST_CASE("EOS rejection preserves every pressure value", "[core][eos][transaction]")
{
    SimulationStorageState storage{{5, 5, 4, 120.0, 80.0, 0.5, 60.0}};
    storage.grid.geometry.initialize(5, 5, 4, 120.0, 80.0, 60.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& f = storage.fields;
    for (Field3D* field : {&f.u, &f.v, &f.w, &f.rho, &f.p, &f.theta,
            &f.qv, &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh, &f.tke})
        field->resize(5, 5, 4, 0.0f);
    f.rho.fill(1.0f);
    f.p.fill(90000.0f);
    f.theta.fill(300.0f);
    SECTION("invalid density in a later slab") { f.rho[3][3][3] = -1.0f; }
    SECTION("invalid attribution after pressure evaluation")
    { f.qv[3][3][3] = std::numeric_limits<float>::quiet_NaN(); }
    SECTION("invalid old pressure") { f.p[3][3][3] = -1.0f; }
    SECTION("pressure cannot be represented by storage")
    { f.theta[3][3][3] = std::numeric_limits<float>::max(); }
    const Field3D before = f.p;
    REQUIRE_THROWS(tmv::dynamics_eos::apply_pressure_projection(
        make_simulation_state_view(storage)));
    REQUIRE(std::memcmp(before.data(), f.p.data(), before.size() * sizeof(float)) == 0);
}
