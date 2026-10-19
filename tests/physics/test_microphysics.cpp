#include "microphysics/schemes/milbrandt/coupled_storage.hpp"
/**
 * @file test_microphysics.cpp
 * @brief Unit tests for microphysics schemes with analytical value verification.
 *
 * Tests verify actual computed tendency values against the Kessler formulas:
 *   production: finite-step implicit accretion plus autoconversion
 *   evaporation: published diffusion/ventilation kernel, mass/saturation bounded
 *   total water:    sum(dq_x/dt) should balance (mass conservation)
 */
#include "catch2/catch.hpp"
#include "microphysics/microphysics_base.hpp"
#include "microphysics/factory.hpp"
#include "microphysics/schemes/lin/lin.hpp"
#include "microphysics/base/thermodynamics.hpp"
#include "microphysics/base/fall_speeds.hpp"
#include "microphysics/schemes/milbrandt/milbrandt.hpp"
#include "microphysics/schemes/thompson/cold_processes.hpp"
#include "microphysics/schemes/thompson/collection_processes.hpp"
#include "microphysics/schemes/thompson/frozen_melting.hpp"
#include "microphysics/schemes/thompson/nucleation.hpp"
#include "microphysics/schemes/thompson/thermodynamic_helpers.hpp"
#include "microphysics/schemes/thompson/warm_rain_processes.hpp"
#include "core/runtime/simulation.hpp"
#include "compute/compute_backend.hpp"

#include <cmath>
#include <cstring>
#include <numeric>
#include <limits>
#include <iterator>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

// Kessler constants (from simulation.hpp / microphysics_base.hpp)
constexpr double kQc0 = 1.0e-3;
constexpr double kCauto = 1.0e-3;
constexpr double kCaccr = 2.2;
constexpr double kCaccrExponent = 0.875;

// Test conditions: theta=300K, p=100kPa -> T=300K
constexpr double kTestT = 300.0;
constexpr double kTestP = 100000.0;

double float_ulp(float value)
{
    const float upward = std::nextafter(
        value, std::numeric_limits<float>::infinity());
    const float downward = std::nextafter(
        value, -std::numeric_limits<float>::infinity());
    return std::max(
        std::abs(static_cast<double>(upward) - value),
        std::abs(static_cast<double>(value) - downward));
}

void setup_grid()
{
    NR = 8;
    NTH = 8;
    NZ = 8;
    dr = 1000.0;
    dz = 500.0;
    dt = 1.0;
    dtheta = 2.0 * 3.14159265358979323846 / NTH;
}

void setup_warm_state()
{
    // Warm atmosphere above freezing — only warm-rain processes active
    theta.resize(NR, NTH, NZ, 300.0f);
    p.resize(NR, NTH, NZ, 100000.0f);
    rho.resize(NR, NTH, NZ, 1.2f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
    rho0_base.assign(NZ, 1.2);
}

struct Tendencies
{
    Field3D dtheta, dqv, dqc, dqr, dqi, dqs, dqg, dqh;

    void resize(int nr, int nth, int nz)
    {
        dtheta.resize(nr, nth, nz, 0.0f);
        dqv.resize(nr, nth, nz, 0.0f);
        dqc.resize(nr, nth, nz, 0.0f);
        dqr.resize(nr, nth, nz, 0.0f);
        dqi.resize(nr, nth, nz, 0.0f);
        dqs.resize(nr, nth, nz, 0.0f);
        dqg.resize(nr, nth, nz, 0.0f);
        dqh.resize(nr, nth, nz, 0.0f);
    }
};

class PartialLinBackend final : public ComputeBackend,
                                public MicrophysicsComputeKernels
{
public:
    explicit PartialLinBackend(bool atomic = false, bool throws = false)
        : atomic_(atomic), throws_(throws) {}
    int pointwise_calls = 0;
    const float* fused_destination = nullptr;
    std::string name() const override { return "partial-lin"; }
    bool initialize(std::string&) override { return true; }
    void shutdown() override {}
    MicrophysicsComputeKernels* microphysics_kernels() override { return this; }

    bool supports(MicrophysicsKernelOp op) const override
    {
        return op == MicrophysicsKernelOp::LinPointwise ||
            op == MicrophysicsKernelOp::LinSedimentation || (atomic_ && op == MicrophysicsKernelOp::LinFused);
    }

    bool supports_atomic_lin_fused() const override { return atomic_; }
    bool dispatch_lin_fused(const LinFusedDispatch& d) override
    {
        fused_destination = d.pointwise.dtheta_dt;
        if (throws_) throw std::runtime_error("injected fused Lin failure");
        return false;
    }

    bool dispatch_lin_pointwise(const LinPointwiseDispatch& dispatch) override
    {
        ++pointwise_calls;
        const std::size_t cells = static_cast<std::size_t>(dispatch.nx) *
            static_cast<std::size_t>(dispatch.ny) *
            static_cast<std::size_t>(dispatch.nz);
        for (float* output : {
                 dispatch.dtheta_dt, dispatch.dqv_dt, dispatch.dqc_dt,
                 dispatch.dqr_dt, dispatch.dqi_dt, dispatch.dqs_dt,
                 dispatch.dqg_dt, dispatch.dqh_dt})
            std::fill_n(output, cells, 17.0f);
        return true;
    }

    bool dispatch_lin_sedimentation(
        const LinSedimentationDispatch&) override
    {
        return false;
    }
private:
    bool atomic_;
    bool throws_;
};

struct ComputeRuntimeReset
{
    ~ComputeRuntimeReset()
    {
        global_compute_backend_runtime_state.reset();
    }
};

void require_same_field(const Field3D& expected, const Field3D& actual)
{
    REQUIRE(expected.size_r() == actual.size_r());
    REQUIRE(expected.size_th() == actual.size_th());
    REQUIRE(expected.size_z() == actual.size_z());
    REQUIRE(expected.size() == actual.size());
    REQUIRE(std::memcmp(expected.data(), actual.data(),
                        expected.size() * sizeof(float)) == 0);
}

void compute(MicrophysicsScheme& scheme, double step_dt, Tendencies& t)
{
    t.resize(NR, NTH, NZ);
    scheme.set_vertical_spacing_m(dz);
    scheme.compute_tendencies(p, theta, rho, qv, qc, qr, qi, qs, qg, qh,
                              step_dt,
                              t.dtheta, t.dqv, t.dqc, t.dqr,
                              t.dqi, t.dqs, t.dqg, t.dqh);
}

struct LiteralGammaOracle
{
    double inverse_lambda = 0.0;
    double intercept = 0.0;
    double mean_mass_diameter = 0.0;
};

LiteralGammaOracle literal_gamma_oracle(
    double density, double mixing_ratio, double number_mixing_ratio,
    double alpha, double mass_coefficient, double mass_exponent)
{
    LiteralGammaOracle result;
    const double number_volume = density * number_mixing_ratio;
    result.inverse_lambda = std::pow(
        density * mixing_ratio * std::tgamma(1.0 + alpha)
            / (mass_coefficient * number_volume
               * std::tgamma(1.0 + alpha + mass_exponent)),
        1.0 / mass_exponent);
    result.intercept = number_volume
        / (std::tgamma(1.0 + alpha)
           * std::pow(result.inverse_lambda, 1.0 + alpha));
    result.mean_mass_diameter = std::pow(
        density * mixing_ratio
            / (mass_coefficient * number_volume),
        1.0 / mass_exponent);
    return result;
}

double literal_gamma_moment(
    const LiteralGammaOracle& distribution,
    double alpha, double exponent)
{
    return distribution.intercept
        * std::tgamma(1.0 + alpha + exponent)
        * std::pow(
            distribution.inverse_lambda,
            1.0 + alpha + exponent);
}

double literal_mass_weighted_speed(
    const LiteralGammaOracle& distribution,
    double density, double alpha, double mass_exponent,
    double fall_coefficient, double fall_exponent)
{
    return std::sqrt(1.225 / density) * fall_coefficient
        * std::pow(distribution.inverse_lambda, fall_exponent)
        * std::tgamma(
            1.0 + alpha + mass_exponent + fall_exponent)
        / std::tgamma(1.0 + alpha + mass_exponent);
}

struct LiteralPairCollisionOracle
{
    double first_mass = 0.0;
    double second_mass = 0.0;
    double first_number = 0.0;
    double second_number = 0.0;
};

LiteralPairCollisionOracle literal_pair_collision_oracle(
    double density, double step_dt, double efficiency,
    double relative_speed,
    double first_q, double first_n,
    const LiteralGammaOracle& first,
    double first_alpha, double first_mass_coefficient,
    double first_mass_exponent,
    double second_q, double second_n,
    const LiteralGammaOracle& second,
    double second_alpha, double second_mass_coefficient,
    double second_mass_exponent)
{
    constexpr double pi_over_four =
        0.785398163397448309616;
    const double common =
        step_dt * pi_over_four * efficiency
        * relative_speed / density;
    const auto first_moment =
        [&](double exponent)
        {
            return literal_gamma_moment(
                first, first_alpha, exponent);
        };
    const auto second_moment =
        [&](double exponent)
        {
            return literal_gamma_moment(
                second, second_alpha, exponent);
        };
    const double collision_number_volume =
        common * density
        * (first_moment(2.0) * second_moment(0.0)
           + 2.0 * first_moment(1.0) * second_moment(1.0)
           + first_moment(0.0) * second_moment(2.0));

    LiteralPairCollisionOracle result;
    result.first_mass = std::clamp(
        common * first_mass_coefficient
            * (first_moment(first_mass_exponent + 2.0)
                 * second_moment(0.0)
               + 2.0
                 * first_moment(first_mass_exponent + 1.0)
                 * second_moment(1.0)
               + first_moment(first_mass_exponent)
                 * second_moment(2.0)),
        0.0, first_q);
    result.second_mass = std::clamp(
        common * second_mass_coefficient
            * (second_moment(second_mass_exponent + 2.0)
                 * first_moment(0.0)
               + 2.0
                 * second_moment(second_mass_exponent + 1.0)
                 * first_moment(1.0)
               + second_moment(second_mass_exponent)
                 * first_moment(2.0)),
        0.0, second_q);
    result.first_number = std::clamp(
        std::min(
            collision_number_volume / density,
            result.first_mass * first_n / first_q),
        0.0, first_n);
    result.second_number = std::clamp(
        std::min(
            collision_number_volume / density,
            result.second_mass * second_n / second_q),
        0.0, second_n);
    return result;
}

struct LiteralCloudCollectionOracle
{
    double mass = 0.0;
    double cloud_number = 0.0;
};

LiteralCloudCollectionOracle literal_cloud_collection_oracle(
    double density, double temperature_k, double step_dt,
    double cloud, double cloud_number_mixing,
    double collector_q, double collector_number_mixing,
    double collector_alpha, double collector_mass_coefficient,
    double collector_mass_exponent,
    double collector_fall_coefficient,
    double collector_fall_exponent,
    int collector_category)
{
    constexpr double pi_over_four =
        0.785398163397448309616;
    constexpr double cloud_alpha = 1.0;
    constexpr double cloud_mu = 3.0;
    constexpr double liquid_mass_coefficient = 523.598775598;
    const double cloud_number_volume =
        density * cloud_number_mixing;
    const double cloud_gamma_number =
        std::tgamma(1.0 + cloud_alpha);
    const double cloud_inverse_lambda = std::cbrt(
        density * cloud / cloud_number_volume
        / (liquid_mass_coefficient
           * std::tgamma(
               1.0 + cloud_alpha + 3.0 / cloud_mu)
           / cloud_gamma_number));
    const LiteralGammaOracle collector = literal_gamma_oracle(
        density, collector_q, collector_number_mixing,
        collector_alpha, collector_mass_coefficient,
        collector_mass_exponent);
    const double collector_speed = literal_mass_weighted_speed(
        collector, density, collector_alpha,
        collector_mass_exponent, collector_fall_coefficient,
        collector_fall_exponent);

    double efficiency = 0.0;
    if (collector_category == 0)
    {
        efficiency =
            std::min(collector.mean_mass_diameter, 1.0e-3) * 1.0e3;
        efficiency = std::min(
            std::cbrt(
                density * cloud
                / (liquid_mass_coefficient * cloud_number_volume)),
            30.0e-6) * 3.333e4 * std::sqrt(efficiency);
    }
    else if (collector_category == 1)
    {
        const double dynamic_viscosity =
            1.72e-5 * (393.0 / (temperature_k + 120.0))
            * std::pow(
                temperature_k
                    / microphysics_constants::T0,
                1.5);
        const double cloud_diameter = std::cbrt(
            density * cloud
            / (liquid_mass_coefficient * cloud_number_volume));
        double stokes =
            microphysics_constants::rho_w
            * collector_speed * cloud_diameter * cloud_diameter
            / (9.0 * dynamic_viscosity
               * collector.mean_mass_diameter);
        stokes = std::clamp(stokes, 1.5, 10.0);
        efficiency =
            0.55 * std::log10(2.51 * stokes);
    }
    else
    {
        const double cloud_diameter = std::cbrt(
            density * cloud
            / (liquid_mass_coefficient * cloud_number_volume));
        efficiency = std::exp(
            -8.68e-7 * std::pow(cloud_diameter, -1.6)
            * collector.mean_mass_diameter);
    }
    efficiency = std::clamp(efficiency, 0.0, 1.0);

    const double ilc = cloud_inverse_lambda;
    const double ilx = collector.inverse_lambda;
    const double gc11 = std::tgamma(
        1.0 / cloud_mu + 1.0 + cloud_alpha);
    const double gc12 = std::tgamma(
        2.0 / cloud_mu + 1.0 + cloud_alpha);
    const double gc13 = std::tgamma(
        3.0 / cloud_mu + 1.0 + cloud_alpha);
    const double gc14 = std::tgamma(
        4.0 / cloud_mu + 1.0 + cloud_alpha);
    const double gc15 = std::tgamma(
        5.0 / cloud_mu + 1.0 + cloud_alpha);
    const double gx1 = std::tgamma(1.0 + collector_alpha);
    const double common =
        step_dt * std::sqrt(1.225 / density)
        * collector_fall_coefficient * pi_over_four
        * efficiency * cloud_number_volume
        * (density * collector_number_mixing)
        / (cloud_gamma_number * gx1);

    LiteralCloudCollectionOracle result;
    result.mass =
        common * liquid_mass_coefficient / density
        * (gc13 * std::tgamma(3.0 + collector_alpha + collector_fall_exponent)
             * std::pow(ilc, 3.0)
             * std::pow(ilx, collector_fall_exponent + 2.0)
           + 2.0 * gc14
             * std::tgamma(2.0 + collector_alpha + collector_fall_exponent)
             * std::pow(ilc, 4.0)
             * std::pow(ilx, collector_fall_exponent + 1.0)
           + gc15 * std::tgamma(1.0 + collector_alpha + collector_fall_exponent) * std::pow(ilc, 5.0)
             * std::pow(ilx, collector_fall_exponent));
    result.cloud_number =
        common
        * (cloud_gamma_number
             * std::tgamma(3.0 + collector_alpha + collector_fall_exponent)
             * std::pow(ilx, collector_fall_exponent + 2.0)
           + 2.0 * gc11
             * std::tgamma(2.0 + collector_alpha + collector_fall_exponent) * ilc
             * std::pow(ilx, collector_fall_exponent + 1.0)
           + gc12 * std::tgamma(1.0 + collector_alpha + collector_fall_exponent) * ilc * ilc
             * std::pow(ilx, collector_fall_exponent));
    if (collector_category == 0)
    {
        result.mass *= 0.6366;
        result.cloud_number *= 0.6366;
    }
    result.mass = std::clamp(result.mass, 0.0, cloud);
    result.cloud_number = std::clamp(
        result.cloud_number / density,
        0.0, cloud_number_mixing);
    return result;
}

} // namespace

TEST_CASE("SCI-024 thermodynamic density helpers use dry-air mixing ratios",
          "[physics][microphysics][thermodynamics][SCI-024]")
{
    constexpr double p_test = 90000.0;
    constexpr double T_test = 285.0;
    constexpr double qv_test = 0.018;
    constexpr double qc_test = 0.002;
    constexpr double qr_test = 0.001;
    constexpr double epsilon = microphysics_constants::R_d
                             / microphysics_constants::R_v;

    const double expected_rho_d = p_test /
        (microphysics_constants::R_d * T_test);
    const double expected_Tv = T_test * (1.0 + qv_test / epsilon) /
        (1.0 + qv_test + qc_test + qr_test);

    REQUIRE(thermodynamics::dry_air_density(p_test, T_test) ==
            Approx(expected_rho_d).epsilon(1e-13));
    REQUIRE(thermodynamics::air_density(p_test, T_test, qv_test) ==
            Approx(expected_rho_d).epsilon(1e-13));
    REQUIRE(thermodynamics::air_density(p_test, T_test, 0.0) ==
            Approx(expected_rho_d).epsilon(1e-13));
    REQUIRE(thermodynamics::virtual_temperature(
                T_test, qv_test, qc_test, qr_test) ==
            Approx(expected_Tv).epsilon(1e-13));

    REQUIRE(thermodynamics::virtual_temperature(T_test, qv_test) > T_test);
    REQUIRE(thermodynamics::virtual_temperature(
                T_test, qv_test, qc_test, qr_test) <
            thermodynamics::virtual_temperature(T_test, qv_test));
}

TEST_CASE("Liquid saturation adjustment solves equilibrium and latent energy",
          "[physics][microphysics][thermodynamics][saturation]")
{
    constexpr double pressure = 90000.0;
    constexpr double initial_temperature = 290.0;
    constexpr double latent_factor =
        microphysics_constants::L_v / microphysics_constants::cp;

    SECTION("supersaturated vapor condenses to the warmed equilibrium")
    {
        const double initial_vapor =
            thermodynamics::saturation_mixing_ratio_water(
                initial_temperature, pressure) + 0.01;
        double vapor = initial_vapor;
        double cloud = 0.0;
        const double adjusted_temperature =
            thermodynamics::saturation_adjustment(
                initial_temperature, pressure, vapor, cloud);

        REQUIRE(vapor + cloud
                == Approx(initial_vapor).epsilon(2.0e-14));
        REQUIRE(vapor
                == Approx(
                    thermodynamics::saturation_mixing_ratio_water(
                        adjusted_temperature, pressure))
                    .epsilon(2.0e-13));
        REQUIRE(adjusted_temperature - initial_temperature
                == Approx(latent_factor * cloud).epsilon(2.0e-13));

        const double once_temperature = adjusted_temperature;
        const double once_vapor = vapor;
        const double once_cloud = cloud;
        const double twice_temperature =
            thermodynamics::saturation_adjustment(
                once_temperature, pressure, vapor, cloud);
        REQUIRE(twice_temperature
                == Approx(once_temperature).epsilon(2.0e-14));
        REQUIRE(vapor == Approx(once_vapor).epsilon(2.0e-14));
        REQUIRE(cloud == Approx(once_cloud).epsilon(2.0e-14));
    }

    SECTION("cloud evaporates to a cooled equilibrium when sufficient")
    {
        const double initial_vapor =
            0.4 * thermodynamics::saturation_mixing_ratio_water(
                initial_temperature, pressure);
        constexpr double initial_cloud = 0.02;
        double vapor = initial_vapor;
        double cloud = initial_cloud;
        const double adjusted_temperature =
            thermodynamics::saturation_adjustment(
                initial_temperature, pressure, vapor, cloud);
        const double evaporated = initial_cloud - cloud;

        REQUIRE(vapor + cloud
                == Approx(initial_vapor + initial_cloud)
                    .epsilon(2.0e-14));
        REQUIRE(vapor
                == Approx(
                    thermodynamics::saturation_mixing_ratio_water(
                        adjusted_temperature, pressure))
                    .epsilon(2.0e-13));
        REQUIRE(initial_temperature - adjusted_temperature
                == Approx(latent_factor * evaporated)
                    .epsilon(2.0e-13));
    }

    SECTION("finite cloud evaporates completely without inventing water")
    {
        constexpr double initial_vapor = 1.0e-4;
        constexpr double initial_cloud = 2.0e-4;
        double vapor = initial_vapor;
        double cloud = initial_cloud;
        const double adjusted_temperature =
            thermodynamics::saturation_adjustment(
                initial_temperature, pressure, vapor, cloud);

        REQUIRE(cloud == Approx(0.0).margin(1.0e-18));
        REQUIRE(vapor
                == Approx(initial_vapor + initial_cloud)
                    .epsilon(2.0e-14));
        REQUIRE(initial_temperature - adjusted_temperature
                == Approx(latent_factor * initial_cloud)
                    .epsilon(2.0e-13));
        REQUIRE(vapor
                < thermodynamics::saturation_mixing_ratio_water(
                    adjusted_temperature, pressure));
    }
}

TEST_CASE("Microphysics factory creates kessler scheme", "[physics][microphysics]")
{
    setup_grid();
    auto scheme = create_microphysics_scheme("kessler");
    REQUIRE(scheme != nullptr);
    REQUIRE(scheme->get_scheme_name() == "kessler");
}

TEST_CASE("Microphysics factory publishes module-qualified available ids",
          "[physics][microphysics][factory][mod01]")
{
    const auto ids = get_available_microphysics_schemes();
    REQUIRE(ids == std::vector<std::string>{
        "kessler", "lin", "thompson", "milbrandt", "none"});
}

TEST_CASE("Microphysics factory creates lin scheme", "[physics][microphysics]")
{
    setup_grid();
    auto scheme = create_microphysics_scheme("lin");
    REQUIRE(scheme != nullptr);
}

TEST_CASE("Microphysics factory creates thompson scheme", "[physics][microphysics]")
{
    setup_grid();
    auto scheme = create_microphysics_scheme("thompson");
    REQUIRE(scheme != nullptr);
}

TEST_CASE("Microphysics factory applies only runtime-consumed variant parameters",
          "[physics][microphysics][config]")
{
    tmv::MicrophysicsConfig config;
    config.shared.cloud_water_threshold_kgkg = 0.004;
    config.shared.autoconversion_rate_s_inv = 0.007;
    config.kessler.accretion_coefficient = 1.5;
    config.kessler.evaporation_efficiency = 0.5;

    auto kessler = create_microphysics_scheme("kessler", config);
    const auto state = kessler->portable_state();
    REQUIRE(state.version == 2);
    REQUIRE(state.reals.size() == 5);
    REQUIRE(state.reals[0] == Approx(0.004));
    REQUIRE(state.reals[1] == Approx(0.007));
    REQUIRE(state.reals[2] == Approx(1.5));
    REQUIRE(state.reals[3] == Approx(0.5));

    PortableSchemeState legacy_kessler;
    legacy_kessler.version = 1;
    legacy_kessler.reals = {
        0.005, 0.006, 1.7, 0.0015,
        0.001, 1.0, 0.001, 0.001, 0.001,
        18.0, 40.0, 15.0,
    };
    std::string kessler_restore_error;
    REQUIRE(kessler->restore_portable_state(
        legacy_kessler, kessler_restore_error));
    REQUIRE(kessler_restore_error.empty());
    REQUIRE(kessler->portable_state().version == 2);
    REQUIRE(kessler->portable_state().reals ==
            std::vector<double>{0.005, 0.006, 1.7, 1.0, 18.0});

    config.milbrandt.triple_moment = false;
    config.milbrandt.hail_processes = false;
    config.milbrandt.evaporation_rate_s_inv = 0.0006;
    auto milbrandt = create_microphysics_scheme("milbrandt", config);
    REQUIRE(milbrandt->get_num_prognostic_vars() == 13);
    REQUIRE(milbrandt->portable_state().integers ==
            std::vector<std::uint64_t>{0});
    REQUIRE(milbrandt->portable_state().version == 4);
    REQUIRE(milbrandt->portable_state().reals.size() == 25);

    config.thompson.evaporation_rate_s_inv = 0.0007;
    config.thompson.water_friendly_aerosol_surface_m3 = 375.0e6;
    config.thompson.ice_friendly_aerosol_surface_m3 = 2.25e6;
    auto thompson = create_microphysics_scheme("thompson", config);
    REQUIRE(thompson->portable_state().version == 3);
    REQUIRE(thompson->portable_state().fields.size() == 10);
    REQUIRE(thompson->portable_state().reals[2] == Approx(0.0007));
    REQUIRE(thompson->portable_state().reals[3] == Approx(375.0e6));
    REQUIRE(thompson->portable_state().reals[4] == Approx(2.25e6));

    auto thompson_v1 = thompson->portable_state();
    thompson_v1.version = 1;
    thompson_v1.reals = {0.002, 0.003, 100e6, 100e3};
    thompson_v1.fields.clear();
    thompson_v1.integers.clear();
    std::string restore_error;
    REQUIRE(thompson->restore_portable_state(thompson_v1, restore_error));
    REQUIRE(thompson->portable_state().reals[2] == Approx(1.0e-3));

    config.shared.autoconversion_rate_s_inv = -1.0;
    REQUIRE_THROWS_AS(create_microphysics_scheme("lin", config),
                      std::invalid_argument);

    config.shared.autoconversion_rate_s_inv = 0.001;
    config.kessler.accretion_coefficient = -1.0;
    REQUIRE_NOTHROW(create_microphysics_scheme("lin", config));

    config.thompson.evaporation_rate_s_inv = -1.0;
    REQUIRE_THROWS_AS(create_microphysics_scheme("thompson", config),
                      std::invalid_argument);

    config.thompson.evaporation_rate_s_inv = 0.001;
    config.thompson.water_friendly_aerosol_surface_m3 = 0.0;
    REQUIRE_THROWS_AS(create_microphysics_scheme("thompson", config),
                      std::invalid_argument);
    config.thompson.water_friendly_aerosol_surface_m3 = 350.0e6;
    config.thompson.ice_friendly_aerosol_surface_m3 =
        std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(create_microphysics_scheme("thompson", config),
                      std::invalid_argument);
}

TEST_CASE("Microphysics hot-path parameter identity is exact and field free",
          "[physics][microphysics][identity]")
{
    const std::array<const char*, 5> scheme_names = {
        "none", "kessler", "lin", "thompson", "milbrandt"};
    for (const char* scheme_name : scheme_names)
    {
        const auto scheme = create_microphysics_scheme(scheme_name);
        const auto identity = scheme->parameter_identity();
        const auto portable = scheme->portable_parameter_state();

        REQUIRE(portable.fields.empty());
        REQUIRE(identity.version == portable.version);
        REQUIRE(identity.integer_count == portable.integers.size());
        REQUIRE(identity.real_count == portable.reals.size());
        for (std::size_t n = 0; n < identity.integer_count; ++n)
            REQUIRE(identity.integers[n] == portable.integers[n]);
        for (std::size_t n = 0; n < identity.real_count; ++n)
            REQUIRE(identity.reals[n] == portable.reals[n]);
        REQUIRE(identity == scheme->parameter_identity());
    }

    tmv::MicrophysicsConfig baseline;
    tmv::MicrophysicsConfig changed = baseline;

    changed.kessler.evaporation_efficiency = 0.5;
    REQUIRE(!(create_microphysics_scheme("kessler", baseline)
                  ->parameter_identity() ==
              create_microphysics_scheme("kessler", changed)
                  ->parameter_identity()));

    changed = baseline;
    changed.lin.evaporation_rate_s_inv = 2.0e-3;
    REQUIRE(!(create_microphysics_scheme("lin", baseline)
                  ->parameter_identity() ==
              create_microphysics_scheme("lin", changed)
                  ->parameter_identity()));

    changed = baseline;
    changed.thompson.evaporation_rate_s_inv = 2.0e-3;
    REQUIRE(!(create_microphysics_scheme("thompson", baseline)
                  ->parameter_identity() ==
              create_microphysics_scheme("thompson", changed)
                  ->parameter_identity()));

    changed = baseline;
    changed.thompson.water_friendly_aerosol_surface_m3 = 375.0e6;
    REQUIRE(!(create_microphysics_scheme("thompson", baseline)
                  ->parameter_identity() ==
              create_microphysics_scheme("thompson", changed)
                  ->parameter_identity()));

    changed = baseline;
    changed.thompson.ice_friendly_aerosol_surface_m3 = 2.25e6;
    REQUIRE(!(create_microphysics_scheme("thompson", baseline)
                  ->parameter_identity() ==
              create_microphysics_scheme("thompson", changed)
                  ->parameter_identity()));

    changed = baseline;
    changed.milbrandt.hail_processes = false;
    REQUIRE(!(create_microphysics_scheme("milbrandt", baseline)
                  ->parameter_identity() ==
              create_microphysics_scheme("milbrandt", changed)
                  ->parameter_identity()));

    tmv::MicrophysicsParameterIdentity malformed;
    malformed.integer_count =
        tmv::MicrophysicsParameterIdentity::maximum_integers + 1;
    REQUIRE(!(malformed == malformed));
}

TEST_CASE("Thompson condensation caller activates CCN with a conservative private transfer",
          "[physics][microphysics][thompson][activation][transaction]")
{
    setup_grid();
    setup_warm_state();
    const double saturation =
        thermodynamics::saturation_mixing_ratio_water(kTestT, kTestP);
    qv.resize(NR, NTH, NZ, static_cast<float>(saturation + 1.0e-6));
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    Field3D vertical_velocity(NR, NTH, NZ, 1.0f);

    auto scheme = create_microphysics_scheme("thompson");
    scheme->set_vertical_spacing_m(dz);
    scheme->set_vertical_velocity_field(vertical_velocity, false);
    scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
    const PortableSchemeState before = scheme->portable_state();
    Tendencies tendencies;
    compute(*scheme, dt, tendencies);
    const PortableSchemeState prepared = scheme->portable_state();

    REQUIRE(tendencies.dqc[4][4][4] > 0.0f);
    REQUIRE(tendencies.dqr[4][4][4] == 0.0f);
    REQUIRE(prepared.fields[5][4][4][4] > 0.0f);
    REQUIRE(prepared.fields[5][4][4][4]
            == -prepared.fields[8][4][4][4]);
    REQUIRE(prepared.fields[0][4][4][4]
            == before.fields[0][4][4][4]);
    REQUIRE(prepared.fields[3][4][4][4]
            == before.fields[3][4][4][4]);
    std::string error;
    REQUIRE(scheme->validate_prognostic_update(dt, error));
}

TEST_CASE("Thompson cold-cloud caller applies gated aerosol ice nucleation",
          "[physics][microphysics][thompson][nucleation][transaction]")
{
    setup_grid();
    setup_warm_state();
    theta.resize(NR, NTH, NZ, 235.0f);
    const double saturation =
        thermodynamics::saturation_mixing_ratio_water(235.0, kTestP);
    qv.resize(NR, NTH, NZ, static_cast<float>(saturation + 1.0e-4));
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    Field3D vertical_velocity(NR, NTH, NZ, 0.5f);

    auto scheme = create_microphysics_scheme("thompson");
    scheme->set_vertical_spacing_m(dz);
    scheme->set_vertical_velocity_field(vertical_velocity, false);
    scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
    Tendencies tendencies;
    compute(*scheme, dt, tendencies);
    const PortableSchemeState prepared = scheme->portable_state();

    const float cloud_number = prepared.fields[5][4][4][4];
    const float ice_number = prepared.fields[7][4][4][4];
    const float water_aerosol = prepared.fields[8][4][4][4];
    const float ice_aerosol = prepared.fields[9][4][4][4];
    REQUIRE(ice_number > 0.0f);
    REQUIRE(water_aerosol <= 0.0f);
    REQUIRE(ice_aerosol <= 0.0f);
    const double hydrometeor_number = static_cast<double>(cloud_number)
        + static_cast<double>(ice_number);
    const double aerosol_number = -static_cast<double>(water_aerosol)
        - static_cast<double>(ice_aerosol);
    const double prior_relative_envelope = 2.0e-6
        * std::abs(aerosol_number);
    const double cloud_freezing_storage_envelope =
        3.0 * float_ulp(cloud_number)
        + 3.0 * float_ulp(ice_number)
        + 4.0 * float_ulp(water_aerosol)
        + 2.0 * float_ulp(ice_aerosol);
    const double private_storage_envelope = prior_relative_envelope
        + cloud_freezing_storage_envelope;
    REQUIRE(hydrometeor_number
            <= aerosol_number + private_storage_envelope);
    REQUIRE(hydrometeor_number >= 0.0);
    REQUIRE(aerosol_number - hydrometeor_number > 0.0);
    std::string error;
    REQUIRE(scheme->validate_prognostic_update(dt, error));
}

TEST_CASE("Thompson official cloud freezing closes water heat and private moments",
          "[physics][microphysics][thompson][freezing][conservation][thread]")
{
    setup_grid();
    setup_warm_state();
    constexpr double temperature = 258.15;
    theta.fill(static_cast<float>(thermodynamics::temperature_to_theta(
        temperature, kTestP)));
    qv.resize(NR, NTH, NZ, static_cast<float>(
        thermodynamics::saturation_mixing_ratio_water(
            temperature, kTestP)));
    qc.resize(NR, NTH, NZ, 2.0e-6f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.fill(0.0f);
    qs.fill(0.0f);
    qg.fill(0.0f);
    qh.fill(0.0f);

    Tendencies reference;
    PortableSchemeState reference_private;
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
#endif
    for (const int thread_count : {1, 4})
    {
#ifdef _OPENMP
        omp_set_num_threads(thread_count);
#else
        static_cast<void>(thread_count);
#endif
        auto scheme = create_microphysics_scheme("thompson");
        scheme->set_vertical_spacing_m(dz);
        scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
        Tendencies actual;
        compute(*scheme, dt, actual);
        const PortableSchemeState prepared = scheme->portable_state();

        constexpr int i = 4;
        constexpr int j = 4;
        constexpr int k = 4;
        const double water_rate = actual.dqv[i][j][k]
            + actual.dqc[i][j][k] + actual.dqr[i][j][k]
            + actual.dqi[i][j][k] + actual.dqs[i][j][k]
            + actual.dqg[i][j][k] + actual.dqh[i][j][k];
        const double water_storage_envelope = 2.0e-12
            + 2.0 * (float_ulp(actual.dqv[i][j][k])
                     + float_ulp(actual.dqc[i][j][k])
                     + float_ulp(actual.dqr[i][j][k])
                     + float_ulp(actual.dqi[i][j][k])
                     + float_ulp(actual.dqs[i][j][k])
                     + float_ulp(actual.dqg[i][j][k])
                     + float_ulp(actual.dqh[i][j][k]));
        REQUIRE(std::abs(water_rate) <= water_storage_envelope);

        const double vapor_to_condensed = -actual.dqv[i][j][k];
        const double frozen_gain = actual.dqi[i][j][k]
            + actual.dqs[i][j][k] + actual.dqg[i][j][k];
        const double liquid_to_frozen = frozen_gain
            - vapor_to_condensed;
        const double expected_temperature_rate =
            microphysics_constants::L_s / microphysics_constants::cp
                * vapor_to_condensed
            + microphysics_constants::L_f / microphysics_constants::cp
                * liquid_to_frozen;
        const double expected_theta_rate =
            thermodynamics::temperature_tendency_to_theta(
                expected_temperature_rate, p[i][j][k]);
        REQUIRE(actual.dtheta[i][j][k]
                == Approx(expected_theta_rate).epsilon(3.0e-5)
                    .margin(2.0e-10));

        const float cloud_number = prepared.fields[5][i][j][k];
        const float rain_number = prepared.fields[6][i][j][k];
        const float ice_number = prepared.fields[7][i][j][k];
        const float water_aerosol = prepared.fields[8][i][j][k];
        const float ice_aerosol = prepared.fields[9][i][j][k];
        REQUIRE(ice_number > 0.0f);
        REQUIRE(rain_number == 0.0f);
        const double moment_residual = static_cast<double>(cloud_number)
            + static_cast<double>(rain_number)
            + static_cast<double>(ice_number)
            + static_cast<double>(water_aerosol)
            + static_cast<double>(ice_aerosol);
        const double moment_storage_envelope =
            3.0 * float_ulp(cloud_number)
            + 2.0 * float_ulp(rain_number)
            + 3.0 * float_ulp(ice_number)
            + 4.0 * float_ulp(water_aerosol)
            + 2.0 * float_ulp(ice_aerosol);
        REQUIRE(std::abs(moment_residual) <= moment_storage_envelope);

        if (thread_count == 1)
        {
            reference = actual;
            reference_private = prepared;
        }
        else
        {
            require_same_field(reference.dtheta, actual.dtheta);
            require_same_field(reference.dqv, actual.dqv);
            require_same_field(reference.dqc, actual.dqc);
            require_same_field(reference.dqr, actual.dqr);
            require_same_field(reference.dqi, actual.dqi);
            require_same_field(reference.dqs, actual.dqs);
            require_same_field(reference.dqg, actual.dqg);
            require_same_field(reference.dqh, actual.dqh);
            REQUIRE(prepared.fields.size() == reference_private.fields.size());
            for (std::size_t field = 0; field < prepared.fields.size(); ++field)
                require_same_field(
                    reference_private.fields[field], prepared.fields[field]);
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
}

TEST_CASE("Thompson official ice deposition closes water heat and Ni",
          "[physics][microphysics][thompson][deposition][conservation][thread]")
{
    setup_grid();
    setup_warm_state();
    constexpr double temperature = 258.15;
    theta.fill(static_cast<float>(thermodynamics::temperature_to_theta(
        temperature, kTestP)));
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.fill(1.0e-6f);
    qs.fill(0.0f);
    qg.fill(0.0f);
    qh.fill(0.0f);
    tmv::thompson::ColdProcessTableStore store;
    std::string load_error;
    REQUIRE(store.publish_packaged_file(
        tmv::thompson::ColdProcessTableMetadata::packaged_path,
        load_error,
        tmv::thompson::checked_cold_payload_bytes()));
    const auto tables = store.snapshot();

#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
#endif
    for (const double saturation : {1.1, 0.8})
    {
        Tendencies reference;
        PortableSchemeState reference_private;
        for (const int thread_count : {1, 4})
        {
#ifdef _OPENMP
            omp_set_num_threads(thread_count);
#else
            static_cast<void>(thread_count);
#endif
            const double caller_temperature =
                thermodynamics::theta_to_temperature(
                    theta[4][4][4], p[4][4][4]);
            const double qvsi =
                thermodynamics::saturation_mixing_ratio_ice(
                    static_cast<float>(caller_temperature),
                    p[4][4][4]);
            qv.resize(
                NR, NTH, NZ, static_cast<float>(saturation * qvsi));
            auto scheme = create_microphysics_scheme("thompson");
            scheme->set_vertical_spacing_m(dz);
            scheme->prepare_prognostic_state(
                rho, qc, qr, qi, qs, qg, qh);
            const double density = rho[4][4][4];
            auto private_scalars = scheme->prognostic_scalars_for_transport();
            private_scalars[2].field->fill(
                static_cast<float>(1.0e3 / density));
            const PortableSchemeState initialized = scheme->portable_state();
            const auto expected = tmv::thompson::ice_deposition_rates(
                *tables,
                {p[4][4][4], caller_temperature, density, qv[4][4][4],
                 density * qi[4][4][4],
                 density * initialized.fields[2][4][4][4]},
                dt);
            const auto conversion = tmv::thompson::ice_to_snow_rates(
                *tables,
                {density * qi[4][4][4]
                     + dt * expected.ice_kg_m3_s,
                 density * initialized.fields[2][4][4][4]
                     + dt * expected.ice_number_m3_s},
                dt);

            Tendencies actual;
            compute(*scheme, dt, actual);
            const PortableSchemeState prepared = scheme->portable_state();
            constexpr int i = 4;
            constexpr int j = 4;
            constexpr int k = 4;
            REQUIRE(((saturation > 1.0 && actual.dqv[i][j][k] < 0.0f)
                     || (saturation < 1.0 && actual.dqv[i][j][k] > 0.0f)));
            REQUIRE(conversion.snow_kg_m3_s > 0.0);
            const double water_rate = actual.dqv[i][j][k]
                + actual.dqc[i][j][k] + actual.dqr[i][j][k]
                + actual.dqi[i][j][k] + actual.dqs[i][j][k]
                + actual.dqg[i][j][k] + actual.dqh[i][j][k];
            REQUIRE(water_rate == Approx(0.0).margin(2.0e-12));
            const double expected_theta_rate =
                thermodynamics::temperature_tendency_to_theta(
                    -microphysics_constants::L_s
                        / microphysics_constants::cp
                        * actual.dqv[i][j][k],
                    p[i][j][k]);
            REQUIRE(actual.dtheta[i][j][k]
                    == Approx(expected_theta_rate)
                        .epsilon(3.0e-5).margin(2.0e-10));
            const float ni_tendency = prepared.fields[7][i][j][k];
            REQUIRE(initialized.fields[2][i][j][k]
                        + dt * static_cast<double>(ni_tendency)
                    >= 0.0);

            if (thread_count == 1)
            {
                reference = actual;
                reference_private = prepared;
            }
            else
            {
                require_same_field(reference.dtheta, actual.dtheta);
                require_same_field(reference.dqv, actual.dqv);
                require_same_field(reference.dqc, actual.dqc);
                require_same_field(reference.dqr, actual.dqr);
                require_same_field(reference.dqi, actual.dqi);
                require_same_field(reference.dqs, actual.dqs);
                require_same_field(reference.dqg, actual.dqg);
                require_same_field(reference.dqh, actual.dqh);
                REQUIRE(prepared.fields.size()
                        == reference_private.fields.size());
                for (std::size_t field = 0;
                     field < prepared.fields.size(); ++field)
                    require_same_field(
                        reference_private.fields[field],
                        prepared.fields[field]);
            }
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
}

TEST_CASE("Thompson official snow riming closes water heat and Nc donors",
          "[physics][microphysics][thompson][riming][conservation][thread]")
{
    setup_grid();
    setup_warm_state();
    dz = 1.0e9;
    constexpr double temperature = 258.15;
    theta.fill(static_cast<float>(thermodynamics::temperature_to_theta(
        temperature, kTestP)));
    qv.resize(NR, NTH, NZ, static_cast<float>(
        thermodynamics::saturation_mixing_ratio_water(
            temperature, kTestP)));
    qc.resize(NR, NTH, NZ, 2.0e-4f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.fill(0.0f);
    qs.fill(5.0e-4f);
    qg.fill(5.0e-4f);
    qh.fill(0.0f);

    Tendencies reference;
    PortableSchemeState reference_private;
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
#endif
    for (const int thread_count : {1, 4})
    {
#ifdef _OPENMP
        omp_set_num_threads(thread_count);
#else
        static_cast<void>(thread_count);
#endif
        auto scheme = create_microphysics_scheme("thompson");
        scheme->set_vertical_spacing_m(dz);
        scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
        const PortableSchemeState initialized = scheme->portable_state();
        Tendencies actual;
        compute(*scheme, dt, actual);
        const PortableSchemeState prepared = scheme->portable_state();
        constexpr int i = 4;
        constexpr int j = 4;
        constexpr int k = 4;
        REQUIRE(actual.dqc[i][j][k] < 0.0f);
        REQUIRE(actual.dqs[i][j][k] > 0.0f);
        REQUIRE(actual.dqg[i][j][k] > 0.0f);
        const double water_rate = actual.dqv[i][j][k]
            + actual.dqc[i][j][k] + actual.dqr[i][j][k]
            + actual.dqi[i][j][k] + actual.dqs[i][j][k]
            + actual.dqg[i][j][k] + actual.dqh[i][j][k];
        const double source_storage_envelope =
            (float_ulp(qv[i][j][k]) + float_ulp(qc[i][j][k])
             + float_ulp(qr[i][j][k]) + float_ulp(qi[i][j][k])
             + float_ulp(qs[i][j][k]) + float_ulp(qg[i][j][k])
             + float_ulp(qh[i][j][k])) / dt;
        const double tendency_storage_envelope =
            float_ulp(actual.dqv[i][j][k])
            + float_ulp(actual.dqc[i][j][k])
            + float_ulp(actual.dqr[i][j][k])
            + float_ulp(actual.dqi[i][j][k])
            + float_ulp(actual.dqs[i][j][k])
            + float_ulp(actual.dqg[i][j][k])
            + float_ulp(actual.dqh[i][j][k]);
        REQUIRE(std::abs(water_rate)
                <= 2.0e-12 + source_storage_envelope
                    + tendency_storage_envelope);
        const double vapor_to_condensed = -actual.dqv[i][j][k];
        const double frozen_gain = actual.dqi[i][j][k]
            + actual.dqs[i][j][k] + actual.dqg[i][j][k];
        const double liquid_to_frozen = frozen_gain
            - vapor_to_condensed;
        const double expected_temperature_rate =
            microphysics_constants::L_s / microphysics_constants::cp
                * vapor_to_condensed
            + microphysics_constants::L_f / microphysics_constants::cp
                * liquid_to_frozen;
        const double expected_theta_rate =
            thermodynamics::temperature_tendency_to_theta(
                expected_temperature_rate, p[i][j][k]);
        REQUIRE(actual.dtheta[i][j][k]
                == Approx(expected_theta_rate).epsilon(3.0e-5)
                    .margin(2.0e-10));
        const float nc_tendency = prepared.fields[5][i][j][k];
        REQUIRE(initialized.fields[0][i][j][k]
                    + dt * static_cast<double>(nc_tendency)
                >= 0.0);

        if (thread_count == 1)
        {
            reference = actual;
            reference_private = prepared;
        }
        else
        {
            require_same_field(reference.dtheta, actual.dtheta);
            require_same_field(reference.dqv, actual.dqv);
            require_same_field(reference.dqc, actual.dqc);
            require_same_field(reference.dqr, actual.dqr);
            require_same_field(reference.dqi, actual.dqi);
            require_same_field(reference.dqs, actual.dqs);
            require_same_field(reference.dqg, actual.dqg);
            require_same_field(reference.dqh, actual.dqh);
            REQUIRE(prepared.fields.size() == reference_private.fields.size());
            for (std::size_t field = 0; field < prepared.fields.size(); ++field)
                require_same_field(
                    reference_private.fields[field], prepared.fields[field]);
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
}

TEST_CASE("Thompson late candidate failure preserves public and private destinations",
          "[physics][microphysics][thompson][atomic][invalid]")
{
    setup_grid();
    setup_warm_state();
    constexpr double cold_temperature = 258.15;
    theta.fill(static_cast<float>(thermodynamics::temperature_to_theta(
        cold_temperature, kTestP)));
    const double saturation =
        thermodynamics::saturation_mixing_ratio_water(
            cold_temperature, kTestP);
    qv.resize(NR, NTH, NZ, static_cast<float>(saturation));
    qc.resize(NR, NTH, NZ, 1.0e-4f);
    qr.resize(NR, NTH, NZ, 1.0e-3f);
    qi.fill(1.0e-6f);
    qs.fill(5.0e-4f);
    qg.fill(5.0e-4f);
    Field3D vertical_velocity(NR, NTH, NZ, 1.0f);

    auto scheme = create_microphysics_scheme("thompson");
    scheme->set_vertical_spacing_m(dz);
    scheme->set_vertical_velocity_field(vertical_velocity, false);
    scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
    Tendencies accepted;
    compute(*scheme, dt, accepted);
    const double retained_sink = scheme->bottom_sedimentation_sink_rate(
        rho, qr, qs, qg, qh, 4, 4, dt);
    REQUIRE(retained_sink > 0.0);
    qv.fill(static_cast<float>(saturation + 1.0e-3));
    const PortableSchemeState private_before = scheme->portable_state();

    Tendencies destination;
    destination.resize(NR, NTH, NZ);
    for (Field3D* field : {
             &destination.dtheta, &destination.dqv, &destination.dqc,
             &destination.dqr, &destination.dqi, &destination.dqs,
             &destination.dqg, &destination.dqh})
        field->fill(17.0f);

    REQUIRE_THROWS_AS(
        scheme->compute_tendencies(
            p, theta, rho, qv, qc, qr, qi, qs, qg, qh,
            std::numeric_limits<double>::denorm_min(),
            destination.dtheta, destination.dqv, destination.dqc,
            destination.dqr, destination.dqi, destination.dqs,
            destination.dqg, destination.dqh),
        std::runtime_error);
    for (const Field3D* field : {
             &destination.dtheta, &destination.dqv, &destination.dqc,
             &destination.dqr, &destination.dqi, &destination.dqs,
             &destination.dqg, &destination.dqh})
        for (std::size_t index = 0; index < field->size(); ++index)
            REQUIRE(field->data()[index] == 17.0f);

    const PortableSchemeState private_after = scheme->portable_state();
    REQUIRE(private_after.fields.size() == private_before.fields.size());
    for (std::size_t field = 0; field < private_before.fields.size(); ++field)
        REQUIRE(std::memcmp(private_before.fields[field].data(),
                            private_after.fields[field].data(),
                            private_before.fields[field].size()
                                * sizeof(float)) == 0);
    REQUIRE(scheme->bottom_sedimentation_sink_rate(
                rho, qr, qs, qg, qh, 4, 4, dt) == retained_sink);

    qv[4][4][4] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(
        scheme->compute_tendencies(
            p, theta, rho, qv, qc, qr, qi, qs, qg, qh, dt,
            destination.dtheta, destination.dqv, destination.dqc,
            destination.dqr, destination.dqi, destination.dqs,
            destination.dqg, destination.dqh),
        std::invalid_argument);
    for (const Field3D* field : {
             &destination.dtheta, &destination.dqv, &destination.dqc,
             &destination.dqr, &destination.dqi, &destination.dqs,
             &destination.dqg, &destination.dqh})
        REQUIRE(field->data()[0] == 17.0f);
    REQUIRE(scheme->bottom_sedimentation_sink_rate(
                rho, qr, qs, qg, qh, 4, 4, dt) == retained_sink);
}

TEST_CASE("Thompson retained surface sink owns cloud and ice sedimentation",
          "[physics][microphysics][thompson][sedimentation][conservation]")
{
    setup_grid();
    setup_warm_state();
    theta.fill(static_cast<float>(microphysics_constants::T0));
    qv.fill(static_cast<float>(thermodynamics::saturation_mixing_ratio_water(
        microphysics_constants::T0, kTestP)));
    qc.fill(2.0e-4f);
    qr.fill(0.0f);
    qi.fill(1.0e-4f);
    qs.fill(0.0f);
    qg.fill(0.0f);
    qh.fill(0.0f);
    Field3D vertical_velocity(NR, NTH, NZ, 0.0f);

    auto scheme = create_microphysics_scheme("thompson");
    scheme->set_vertical_spacing_m(dz);
    scheme->set_vertical_velocity_field(vertical_velocity, false);
    scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
    Tendencies tendencies;
    compute(*scheme, dt, tendencies);

    constexpr int i = 4;
    constexpr int j = 4;
    const double sink = scheme->bottom_sedimentation_sink_rate(
        rho, qr, qs, qg, qh, i, j, dt);
    REQUIRE(sink > 0.0);
    double column_water_rate = 0.0;
    for (int k = 0; k < NZ; ++k)
        column_water_rate += rho(i, j, k) * dz
            * (tendencies.dqv(i, j, k) + tendencies.dqc(i, j, k)
               + tendencies.dqr(i, j, k) + tendencies.dqi(i, j, k)
               + tendencies.dqs(i, j, k) + tendencies.dqg(i, j, k)
               + tendencies.dqh(i, j, k));
    REQUIRE(column_water_rate
            == Approx(-rho(i, j, 0) * sink * dz).epsilon(3.0e-5));
}

TEST_CASE("Thompson official sedimentation substeps high-CFL snow transactionally",
          "[physics][microphysics][thompson][sedimentation][substep][thread]")
{
    setup_grid();
    setup_warm_state();
    constexpr double temperature = 260.0;
    constexpr double step_dt = 20.0;
    dz = 0.1;
    theta.fill(static_cast<float>(thermodynamics::temperature_to_theta(
        temperature, kTestP)));
    qv.resize(NR, NTH, NZ, static_cast<float>(
        thermodynamics::saturation_mixing_ratio_ice(
            temperature, kTestP)));
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.fill(0.0f);
    qs.fill(0.0f);
    qg.fill(0.0f);
    qh.fill(0.0f);
    qs(4, 4, NZ - 1) = 1.0e-3f;

    Tendencies reference;
    PortableSchemeState reference_private;
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
#endif
    for (const int thread_count : {1, 4})
    {
#ifdef _OPENMP
        omp_set_num_threads(thread_count);
#else
        static_cast<void>(thread_count);
#endif
        auto scheme = create_microphysics_scheme("thompson");
        scheme->set_vertical_spacing_m(dz);
        scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
        Tendencies actual;
        compute(*scheme, step_dt, actual);
        const auto prepared = scheme->portable_state();
        const double sink = scheme->bottom_sedimentation_sink_rate(
            rho, qr, qs, qg, qh, 4, 4, step_dt);
        REQUIRE(sink > 0.0);
        REQUIRE(qs[4][4][NZ - 1]
                    + step_dt * actual.dqs[4][4][NZ - 1]
                == Approx(0.0).margin(2.0e-10));
        double column_change = 0.0;
        for (int k = 0; k < NZ; ++k)
            column_change += rho[4][4][k] * actual.dqs[4][4][k] * dz;
        REQUIRE(column_change
                == Approx(-rho[4][4][0] * sink * dz).epsilon(4.0e-5));

        if (thread_count == 1)
        {
            reference = actual;
            reference_private = prepared;
        }
        else
        {
            require_same_field(reference.dqs, actual.dqs);
            REQUIRE(prepared.fields.size() == reference_private.fields.size());
            for (std::size_t field = 0; field < prepared.fields.size(); ++field)
                require_same_field(
                    reference_private.fields[field], prepared.fields[field]);
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
}

TEST_CASE("Thompson differential ice sedimentation owns Ni independently",
          "[physics][microphysics][thompson][sedimentation][moments][thread]")
{
    setup_grid();
    setup_warm_state();
    constexpr double temperature = 260.0;
    constexpr double step_dt = 10.0;
    dz = 0.1;
    theta.fill(static_cast<float>(thermodynamics::temperature_to_theta(
        temperature, kTestP)));
    qv.resize(NR, NTH, NZ, static_cast<float>(
        thermodynamics::saturation_mixing_ratio_ice(
            temperature, kTestP)));
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.fill(0.0f);
    qs.fill(0.0f);
    qg.fill(0.0f);
    qh.fill(0.0f);
    qi(4, 4, 2) = 1.0e-4f;

    Tendencies reference;
    PortableSchemeState reference_private;
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
#endif
    for (const int thread_count : {1, 4})
    {
#ifdef _OPENMP
        omp_set_num_threads(thread_count);
#else
        static_cast<void>(thread_count);
#endif
        auto scheme = create_microphysics_scheme("thompson");
        scheme->set_vertical_spacing_m(dz);
        scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
        auto scalars = scheme->prognostic_scalars_for_transport();
        scalars[2].field->fill(0.0f);
        (*scalars[2].field)(4, 4, 2) = static_cast<float>(1.0e9 / rho[4][4][2]);
        const auto initialized = scheme->portable_state();
        Tendencies actual;
        compute(*scheme, step_dt, actual);
        const auto prepared = scheme->portable_state();
        double initial_mass = 0.0;
        double final_mass = 0.0;
        double initial_number = 0.0;
        double final_number = 0.0;
        for (int k = 0; k < NZ; ++k)
        {
            initial_mass += rho[4][4][k] * qi[4][4][k] * dz;
            final_mass += rho[4][4][k]
                * (qi[4][4][k] + step_dt * actual.dqi[4][4][k]) * dz;
            initial_number += rho[4][4][k]
                * initialized.fields[2][4][4][k] * dz;
            final_number += rho[4][4][k]
                * (initialized.fields[2][4][4][k]
                   + step_dt * prepared.fields[7][4][4][k]) * dz;
        }
        const double mass_fraction = (initial_mass - final_mass) / initial_mass;
        const double number_fraction =
            (initial_number - final_number) / initial_number;
        REQUIRE(mass_fraction > 0.0);
        REQUIRE(number_fraction >= 0.0);
        REQUIRE(mass_fraction > number_fraction);
        REQUIRE(initialized.fields[2][4][4][2]
                    + step_dt * prepared.fields[7][4][4][2]
                >= 0.0);

        if (thread_count == 1)
        {
            reference = actual;
            reference_private = prepared;
        }
        else
        {
            require_same_field(reference.dqi, actual.dqi);
            for (std::size_t field = 0; field < prepared.fields.size(); ++field)
                require_same_field(
                    reference_private.fields[field], prepared.fields[field]);
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
}

TEST_CASE("Thompson caller returns evaporated rain number and scavenges aerosol",
          "[physics][microphysics][thompson][aerosol][transaction]")
{
    setup_grid();
    setup_warm_state();
    qv.fill(0.0f);
    qc.fill(0.0f);
    qr.fill(1.0e-3f);
    qi.fill(0.0f);
    qs.fill(0.0f);
    qg.fill(0.0f);
    qh.fill(0.0f);
    tmv::MicrophysicsConfig evaporation_config;
    evaporation_config.thompson.evaporation_rate_s_inv = 10.0;
    auto evaporation = create_microphysics_scheme(
        "thompson", evaporation_config);
    evaporation->set_vertical_spacing_m(dz);
    evaporation->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
    Tendencies tendencies;
    compute(*evaporation, dt, tendencies);
    const auto evaporated_state = evaporation->portable_state();
    REQUIRE(evaporated_state.fields[6][4][4][4] < 0.0f);
    REQUIRE(evaporated_state.fields[8][4][4][4] > 0.0f);
    REQUIRE(-evaporated_state.fields[6][4][4][4]
            > evaporated_state.fields[8][4][4][4]);

    theta.fill(260.0f);
    qv.fill(static_cast<float>(
        thermodynamics::saturation_mixing_ratio_water(260.0, kTestP)));
    qr.fill(1.0e-3f);
    qs.fill(8.0e-4f);
    qg.fill(6.0e-4f);
    auto scavenging = create_microphysics_scheme("thompson");
    scavenging->set_vertical_spacing_m(dz);
    scavenging->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
    compute(*scavenging, dt, tendencies);
    const auto scavenged_state = scavenging->portable_state();
    REQUIRE(scavenged_state.fields[8][4][4][4] < 0.0f);
    REQUIRE(scavenged_state.fields[9][4][4][4] < 0.0f);
}

TEST_CASE("Thompson caller applies pinned collection tables identically across threads",
          "[physics][microphysics][thompson][collection][thread]")
{
    setup_grid();
    setup_warm_state();
    theta.fill(278.15f);
    const float saturated = static_cast<float>(
        thermodynamics::saturation_mixing_ratio_water(278.15, kTestP));
    qv.resize(NR, NTH, NZ, saturated);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 1.0e-4f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 1.0e-4f);
    qg.resize(NR, NTH, NZ, 1.0e-4f);
    qh.resize(NR, NTH, NZ, 0.0f);

    tmv::thompson::CollectionTableStore store;
    std::string load_error;
    REQUIRE(store.publish_packaged_file(
        tmv::thompson::CollectionTableMetadata::packaged_path,
        load_error,
        tmv::thompson::checked_payload_bytes()));
    const auto tables = store.snapshot();

    Tendencies reference;
    PortableSchemeState reference_private;
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
#endif
    for (const int thread_count : {1, 4})
    {
#ifdef _OPENMP
        omp_set_num_threads(thread_count);
#else
        (void)thread_count;
#endif
        auto scheme = create_microphysics_scheme("thompson");
        scheme->set_vertical_spacing_m(dz);
        scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
        const auto initialized = scheme->portable_state();
        const double density = rho[4][4][4];
        const double temperature = thermodynamics::theta_to_temperature(
            theta[4][4][4], p[4][4][4]);
        const auto warm = tmv::thompson::warm_rain_conversion_rates(
            temperature, density, 0.0, density * qr[4][4][4], 0.0,
            density * initialized.fields[1][4][4][4], dt);
        const auto evaporation = tmv::thompson::rain_evaporation_rates(
            p[4][4][4], temperature, density, qv[4][4][4],
            density * qr[4][4][4] + dt * warm.rain_water_kg_m3_s,
            density * initialized.fields[1][4][4][4]
                + dt * warm.rain_number_m3_s,
            dt);
        const auto melting = tmv::thompson::frozen_melting_rates(
            p[4][4][4], temperature, density,
            qv[4][4][4] + dt * evaporation.vapor_kg_m3_s / density,
            density * qs[4][4][4], density * qg[4][4][4], dt);
        const auto expected = tmv::thompson::collection_process_rates(
            *tables,
            density * qr[4][4][4]
                + dt * (warm.rain_water_kg_m3_s
                        + evaporation.rain_water_kg_m3_s
                        + melting.rain_kg_m3_s),
            density * initialized.fields[1][4][4][4]
                + dt * (warm.rain_number_m3_s
                        + evaporation.rain_number_m3_s
                        + melting.rain_number_m3_s),
            density * qs[4][4][4] + dt * melting.snow_kg_m3_s,
            density * qg[4][4][4] + dt * melting.graupel_kg_m3_s,
            tmv::thompson::wet_bulb_temperature(
                p[4][4][4], temperature,
                qv[4][4][4] + dt * evaporation.vapor_kg_m3_s / density),
            dt);

        Tendencies actual;
        compute(*scheme, dt, actual);
        const auto prepared = scheme->portable_state();
        REQUIRE(actual.dqr[4][4][4]
                == Approx((warm.rain_water_kg_m3_s
                           + evaporation.rain_water_kg_m3_s
                           + melting.rain_kg_m3_s
                           + expected.rain_kg_m3_s) / density)
                    .epsilon(2.0e-6).margin(1.0e-10));
        REQUIRE(actual.dqs[4][4][4]
                == Approx((melting.snow_kg_m3_s
                           + expected.snow_kg_m3_s) / density)
                    .epsilon(2.0e-6).margin(1.0e-10));
        REQUIRE(actual.dqg[4][4][4]
                == Approx((melting.graupel_kg_m3_s
                           + expected.graupel_kg_m3_s) / density)
                    .epsilon(2.0e-6).margin(1.0e-10));
        REQUIRE(prepared.fields[6][4][4][4]
                == Approx((warm.rain_number_m3_s
                           + evaporation.rain_number_m3_s
                           + melting.rain_number_m3_s
                           + expected.rain_number_m3_s) / density)
                    .epsilon(2.0e-6).margin(3.0e-4));
        REQUIRE(actual.dqv[4][4][4] + actual.dqc[4][4][4]
                    + actual.dqr[4][4][4] + actual.dqi[4][4][4]
                    + actual.dqs[4][4][4] + actual.dqg[4][4][4]
                == Approx(0.0).margin(2.0e-12));

        if (thread_count == 1)
        {
            reference = actual;
            reference_private = prepared;
        }
        else
        {
            require_same_field(reference.dtheta, actual.dtheta);
            require_same_field(reference.dqv, actual.dqv);
            require_same_field(reference.dqc, actual.dqc);
            require_same_field(reference.dqr, actual.dqr);
            require_same_field(reference.dqi, actual.dqi);
            require_same_field(reference.dqs, actual.dqs);
            require_same_field(reference.dqg, actual.dqg);
            require_same_field(reference.dqh, actual.dqh);
            REQUIRE(prepared.fields.size() == reference_private.fields.size());
            for (std::size_t field = 0; field < prepared.fields.size(); ++field)
                require_same_field(reference_private.fields[field],
                                   prepared.fields[field]);
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
}

TEST_CASE("Thompson forms unseeded warm rain and preserves number state across steps",
          "[physics][microphysics][thompson][autoconversion][transaction]")
{
    setup_grid();
    setup_warm_state();
    const float saturation = static_cast<float>(thermodynamics::saturation_mixing_ratio_water(300.0, 100000.0));
    qv.resize(NR, NTH, NZ, saturation);
    qc.resize(NR, NTH, NZ, 0.005f);
    qr.resize(NR, NTH, NZ, 0.0f);
    Field3D velocity(NR, NTH, NZ, 0.0f);
    auto    scheme = create_microphysics_scheme("thompson");
    scheme->set_vertical_spacing_m(dz);
    scheme->set_vertical_velocity_field(velocity, false);
    scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
    for (int step = 0; step < 3; ++step)
    {
        INFO(step);
        Tendencies tendencies;
        REQUIRE_NOTHROW(compute(*scheme, 1.0, tendencies));
        std::string error;
        REQUIRE(scheme->validate_prognostic_update(1.0, error));
        if (step == 0)
        {
            REQUIRE(tendencies.dqr(4, 4, 4) > 0.0f);
            REQUIRE(scheme->portable_state().fields[6](4, 4, 4) > 0.0f);
        }
        scheme->commit_prognostic_update(1.0);
        for (const auto& pair : {std::make_pair(&theta, &tendencies.dtheta),
                                 std::make_pair(&qv, &tendencies.dqv),
                                 std::make_pair(&qc, &tendencies.dqc),
                                 std::make_pair(&qr, &tendencies.dqr)})
        {
            for (std::size_t n = 0; n < pair.first->size(); ++n)
            {
                pair.first->data()[n] += pair.second->data()[n];
                REQUIRE(std::isfinite(pair.first->data()[n]));
                REQUIRE(pair.first->data()[n] >= 0.0f);
            }
        }
    }
}

TEST_CASE("Thompson caller applies pinned moment-based frozen melting",
          "[physics][microphysics][thompson][melting][transaction]")
{
    setup_grid();
    setup_warm_state();
    constexpr double pressure = 90000.0;
    constexpr double temperature = 278.15;
    constexpr double step_dt = 0.01;
    p.fill(static_cast<float>(pressure));
    theta.fill(static_cast<float>(
        thermodynamics::temperature_to_theta(temperature, pressure)));
    qv.resize(NR, NTH, NZ, 0.004f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 2.0e-4f);
    qg.resize(NR, NTH, NZ, 3.0e-4f);
    qh.resize(NR, NTH, NZ, 0.0f);

    auto scheme = create_microphysics_scheme("thompson");
    scheme->set_vertical_spacing_m(dz);
    scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
    const double density = rho[4][4][4];
    const double caller_temperature = thermodynamics::theta_to_temperature(
        theta[4][4][4], p[4][4][4]);
    const auto expected = tmv::thompson::frozen_melting_rates(
        p[4][4][4], caller_temperature, density, qv[4][4][4],
        density * qs[4][4][4], density * qg[4][4][4], step_dt);
    Tendencies actual;
    compute(*scheme, step_dt, actual);
    const auto prepared = scheme->portable_state();

    REQUIRE(actual.dqr[4][4][4]
            == Approx(expected.rain_kg_m3_s / density).epsilon(2.0e-5));
    REQUIRE(actual.dqs[4][4][4]
            == Approx(expected.snow_kg_m3_s / density).epsilon(2.0e-5));
    REQUIRE(actual.dqg[4][4][4]
            == Approx(expected.graupel_kg_m3_s / density).epsilon(2.0e-5));
    REQUIRE(prepared.fields[6][4][4][4]
            == Approx(expected.rain_number_m3_s / density).epsilon(2.0e-5));
    REQUIRE(actual.dqr[4][4][4] + actual.dqs[4][4][4]
                + actual.dqg[4][4][4]
            == Approx(0.0).margin(2.0e-12));
}

TEST_CASE("Thompson invalid derived temperature fails atomically across threads",
          "[physics][microphysics][thompson][thread][invalid][atomic]")
{
    setup_grid();
    setup_warm_state();
    const double saturation =
        thermodynamics::saturation_mixing_ratio_water(kTestT, kTestP);
    qv.fill(static_cast<float>(saturation));
    qc.fill(1.0e-4f);
    qr.fill(1.0e-3f);

#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
#endif
    for (const int thread_count : {1, 4})
    {
#ifdef _OPENMP
        omp_set_num_threads(thread_count);
#else
        (void)thread_count;
#endif
        auto scheme = create_microphysics_scheme("thompson");
        scheme->set_vertical_spacing_m(dz);
        scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
        Tendencies accepted;
        compute(*scheme, dt, accepted);
        const PortableSchemeState private_before = scheme->portable_state();
        const double sink_before = scheme->bottom_sedimentation_sink_rate(
            rho, qr, qs, qg, qh, 4, 4, dt);

        theta[4][4][4] = -1.0f;
        Tendencies destination;
        destination.resize(NR, NTH, NZ);
        for (Field3D* field : {
                 &destination.dtheta, &destination.dqv, &destination.dqc,
                 &destination.dqr, &destination.dqi, &destination.dqs,
                 &destination.dqg, &destination.dqh})
            field->fill(23.0f);

        REQUIRE_THROWS_AS(
            scheme->compute_tendencies(
                p, theta, rho, qv, qc, qr, qi, qs, qg, qh, dt,
                destination.dtheta, destination.dqv, destination.dqc,
                destination.dqr, destination.dqi, destination.dqs,
                destination.dqg, destination.dqh),
            std::invalid_argument);
        for (const Field3D* field : {
                 &destination.dtheta, &destination.dqv, &destination.dqc,
                 &destination.dqr, &destination.dqi, &destination.dqs,
                 &destination.dqg, &destination.dqh})
            for (std::size_t index = 0; index < field->size(); ++index)
                REQUIRE(field->data()[index] == 23.0f);
        const PortableSchemeState private_after = scheme->portable_state();
        for (std::size_t field = 0; field < private_before.fields.size(); ++field)
            REQUIRE(std::memcmp(private_before.fields[field].data(),
                                private_after.fields[field].data(),
                                private_before.fields[field].size()
                                    * sizeof(float)) == 0);
        REQUIRE(scheme->bottom_sedimentation_sink_rate(
                    rho, qr, qs, qg, qh, 4, 4, dt) == sink_before);
        theta[4][4][4] = static_cast<float>(kTestT);
    }
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
}

TEST_CASE("Thompson extreme private currency rejection is failure atomic",
          "[physics][microphysics][thompson][aerosol][invalid][atomic]")
{
    setup_grid();
    setup_warm_state();
    const double saturation =
        thermodynamics::saturation_mixing_ratio_water(kTestT, kTestP);
    qv.fill(static_cast<float>(saturation));
    qc.fill(0.0f);
    qr.fill(1.0e-3f);
    auto scheme = create_microphysics_scheme("thompson");
    scheme->set_vertical_spacing_m(dz);
    scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
    Tendencies accepted;
    compute(*scheme, dt, accepted);
    const double sink_before = scheme->bottom_sedimentation_sink_rate(
        rho, qr, qs, qg, qh, 4, 4, dt);
    auto scalars = scheme->prognostic_scalars_for_transport();
    scalars[3].field->fill(std::numeric_limits<float>::max());

    Tendencies destination;
    destination.resize(NR, NTH, NZ);
    for (Field3D* field : {
             &destination.dtheta, &destination.dqv, &destination.dqc,
             &destination.dqr, &destination.dqi, &destination.dqs,
             &destination.dqg, &destination.dqh})
        field->fill(19.0f);
    REQUIRE_THROWS_AS(
        scheme->compute_tendencies(
            p, theta, rho, qv, qc, qr, qi, qs, qg, qh, dt,
            destination.dtheta, destination.dqv, destination.dqc,
            destination.dqr, destination.dqi, destination.dqs,
            destination.dqg, destination.dqh),
        std::invalid_argument);
    for (const Field3D* field : {
             &destination.dtheta, &destination.dqv, &destination.dqc,
             &destination.dqr, &destination.dqi, &destination.dqs,
             &destination.dqg, &destination.dqh})
        for (std::size_t index = 0; index < field->size(); ++index)
            REQUIRE(field->data()[index] == 19.0f);
    REQUIRE(scheme->bottom_sedimentation_sink_rate(
                rho, qr, qs, qg, qh, 4, 4, dt) == sink_before);
}

TEST_CASE("Thompson evaporation control drives bounded tendencies",
          "[physics][microphysics][config][evaporation]")
{
    setup_grid();
    setup_warm_state();
    qv.resize(NR, NTH, NZ, 0.0f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 1.0e-3f);

    struct Result
    {
        double dqv;
        double dqr;
        double dtheta;
    };
    const auto run = [](const char* name, double coefficient)
    {
        tmv::MicrophysicsConfig config;
        config.thompson.evaporation_rate_s_inv = coefficient;
        config.milbrandt.evaporation_rate_s_inv = coefficient;
        auto scheme = create_microphysics_scheme(name, config);
        Tendencies tendencies;
        compute(*scheme, dt, tendencies);
        constexpr int i = 4, j = 4, k = 4;
        return Result{
            static_cast<double>(tendencies.dqv(i, j, k)),
            static_cast<double>(tendencies.dqr(i, j, k)),
            static_cast<double>(tendencies.dtheta(i, j, k))};
    };

    const Result disabled = run("thompson", 0.0);
    REQUIRE(disabled.dqv == Approx(0.0).margin(1.0e-12));

    const Result base = run("thompson", 5.0e-4);
    const Result doubled = run("thompson", 1.0e-3);
    REQUIRE(base.dqv > 0.0);
    REQUIRE(base.dtheta < 0.0);
    REQUIRE(doubled.dqv == Approx(2.0 * base.dqv).epsilon(2.0e-5));
    REQUIRE(doubled.dtheta == Approx(2.0 * base.dtheta).epsilon(2.0e-5));

    const Result bounded = run("thompson", 2.0);
    REQUIRE(bounded.dqv == Approx(1.0e-3 / dt).margin(1.0e-9));
    REQUIRE(1.0e-3 + dt * bounded.dqr >= -1.0e-9);
}

TEST_CASE("Milbrandt rain evaporation is the published diffusion kernel",
          "[physics][microphysics][milbrandt][evaporation]")
{
    setup_grid();
    setup_warm_state();
    qv.resize(NR, NTH, NZ, 0.0f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 1.0e-3f);

    const auto run = [](double legacy_coefficient)
    {
        tmv::MicrophysicsConfig config;
        config.milbrandt.evaporation_rate_s_inv = legacy_coefficient;
        auto scheme = create_microphysics_scheme("milbrandt", config);
        Tendencies tendencies;
        compute(*scheme, dt, tendencies);
        return std::array<double, 3>{
            tendencies.dqv(4, 4, 4),
            tendencies.dqr(4, 4, 4),
            tendencies.dtheta(4, 4, 4)};
    };

    const auto zero = run(0.0);
    const auto arbitrary = run(2.0);
    REQUIRE(zero == arbitrary);
    REQUIRE(zero[0] > 0.0);
    REQUIRE(zero[1] < 0.0);
    REQUIRE(zero[2] < 0.0);
    REQUIRE(1.0e-3 + dt * zero[1] >= -1.0e-9);
}

TEST_CASE("Milbrandt pristine-ice deposition matches the published ventilation equation",
          "[physics][microphysics][milbrandt][ice][reference]")
{
    setup_grid();
    constexpr double temperature_k = 263.15;
    constexpr double pressure_pa = 80000.0;
    constexpr double density = 1.0;
    constexpr double ice = 2.0e-4;
    theta.resize(
        NR, NTH, NZ,
        static_cast<float>(
            thermodynamics::temperature_to_theta(
                temperature_k, pressure_pa)));
    p.resize(NR, NTH, NZ, static_cast<float>(pressure_pa));
    rho.resize(NR, NTH, NZ, static_cast<float>(density));
    const double qsi =
        thermodynamics::saturation_mixing_ratio_ice(
            temperature_k, pressure_pa);
    qv.resize(NR, NTH, NZ, static_cast<float>(1.02 * qsi));
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, static_cast<float>(ice));
    qs.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);

    auto scheme = create_microphysics_scheme("milbrandt");
    scheme->set_vertical_spacing_m(1.0e12);
    Tendencies tendencies;
    tendencies.resize(NR, NTH, NZ);
    scheme->compute_tendencies(
        p, theta, rho, qv, qc, qr, qi, qs, qg, qh, dt,
        tendencies.dtheta, tendencies.dqv, tendencies.dqc,
        tendencies.dqr, tendencies.dqi, tendencies.dqs,
        tendencies.dqg, tendencies.dqh);

    constexpr double pi = 3.14159265358979323846;
    constexpr double c_i = 440.0;
    constexpr double seed_diameter = 1.0e-4;
    constexpr double fall_a = 71.34;
    constexpr double fall_b = 0.6635;
    const double number_mixing =
        ice / (c_i * std::pow(seed_diameter, 3.0));
    const double number_volume = density * number_mixing;
    const double inverse_lambda = std::cbrt(
        density * ice
        / (c_i * number_volume * std::tgamma(4.0)));
    const double intercept = number_volume / inverse_lambda;
    const double temperature_c =
        temperature_k - microphysics_constants::T0;
    const double diffusivity =
        (2.2157e-5 + 0.0155e-5 * temperature_c)
        * 1.0e5 / pressure_pa;
    const double dynamic_viscosity =
        1.72e-5 * (393.0 / (temperature_k + 120.0))
        * std::pow(
            temperature_k / microphysics_constants::T0, 1.5);
    const double kinematic_viscosity =
        dynamic_viscosity / density;
    const double schmidt_one_third =
        std::cbrt(kinematic_viscosity / diffusivity);
    const double conductivity =
        2.3971e-2 + 0.0078e-2 * temperature_c;
    const double ventilation =
        0.78 * std::tgamma(2.0)
            * inverse_lambda * inverse_lambda
        + 0.30 * schmidt_one_third
            * std::sqrt(
                std::sqrt(1.225 / density)
                * fall_a / kinematic_viscosity)
            * std::tgamma(2.5 + 0.5 * fall_b)
            * std::pow(
                inverse_lambda, 2.5 + 0.5 * fall_b);
    const double inverse_resistance = 1.0 / (
        microphysics_constants::L_s * microphysics_constants::L_s
            / (conductivity * microphysics_constants::R_v
               * temperature_k * temperature_k)
        + 1.0 / (density * qsi * diffusivity));
    const double expected_increment =
        dt / density * 0.5 * inverse_resistance * 2.0 * pi
        * 0.02 * intercept * ventilation;
    const double expected_theta_rate =
        thermodynamics::temperature_tendency_to_theta(
            microphysics_constants::L_s
                / microphysics_constants::cp
                * expected_increment / dt,
            pressure_pa);

    CAPTURE(expected_increment, tendencies.dqv(4, 4, 4),
            tendencies.dqi(4, 4, 4), tendencies.dqs(4, 4, 4),
            tendencies.dtheta(4, 4, 4));
    REQUIRE(tendencies.dqv(4, 4, 4)
            == Approx(-expected_increment / dt).epsilon(2.0e-3));
    REQUIRE(tendencies.dtheta(4, 4, 4)
            == Approx(expected_theta_rate).epsilon(2.0e-3));
    REQUIRE(tendencies.dqi(4, 4, 4)
            + tendencies.dqs(4, 4, 4)
            == Approx(expected_increment / dt).epsilon(2.0e-3));
}

TEST_CASE("Milbrandt snow melting matches the published heat-transfer equation",
          "[physics][microphysics][milbrandt][melting][reference]")
{
    setup_grid();
    constexpr double temperature_k = 278.15;
    constexpr double pressure_pa = 100000.0;
    constexpr double density = 1.0;
    constexpr double snow = 1.0e-3;
    theta.resize(NR, NTH, NZ, static_cast<float>(temperature_k));
    p.resize(NR, NTH, NZ, static_cast<float>(pressure_pa));
    rho.resize(NR, NTH, NZ, static_cast<float>(density));
    qv.resize(
        NR, NTH, NZ,
        static_cast<float>(
            thermodynamics::saturation_mixing_ratio_water(
                microphysics_constants::T0, pressure_pa)));
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, static_cast<float>(snow));
    qg.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);

    auto scheme = create_microphysics_scheme("milbrandt");
    scheme->set_vertical_spacing_m(1.0e12);
    Tendencies tendencies;
    tendencies.resize(NR, NTH, NZ);
    scheme->compute_tendencies(
        p, theta, rho, qv, qc, qr, qi, qs, qg, qh, dt,
        tendencies.dtheta, tendencies.dqv, tendencies.dqc,
        tendencies.dqr, tendencies.dqi, tendencies.dqs,
        tendencies.dqg, tendencies.dqh);

    constexpr double pi = 3.14159265358979323846;
    constexpr double c_s = 0.1597;
    constexpr double d_s = 2.078;
    constexpr double seed_diameter = 1.0e-3;
    constexpr double c_spherical = pi / 6.0 * 100.0;
    constexpr double fall_a = 11.72;
    constexpr double fall_b = 0.41;
    const double number_mixing =
        snow / (c_s * std::pow(seed_diameter, d_s));
    const double number_volume = density * number_mixing;
    const double inverse_lambda = std::cbrt(
        density * snow
        / (c_spherical * number_volume * std::tgamma(4.0)));
    const double intercept = number_volume / inverse_lambda;
    const double temperature_c =
        temperature_k - microphysics_constants::T0;
    const double diffusivity =
        (2.2157e-5 + 0.0155e-5 * temperature_c)
        * 1.0e5 / pressure_pa;
    const double dynamic_viscosity =
        1.72e-5 * (393.0 / (temperature_k + 120.0))
        * std::pow(
            temperature_k / microphysics_constants::T0, 1.5);
    const double kinematic_viscosity =
        dynamic_viscosity / density;
    const double schmidt_one_third =
        std::cbrt(kinematic_viscosity / diffusivity);
    const double conductivity =
        2.3971e-2 + 0.0078e-2 * temperature_c;
    const double ventilation =
        0.78 * inverse_lambda * inverse_lambda
        + 0.30 * schmidt_one_third
            * std::sqrt(
                std::sqrt(1.225 / density)
                * fall_a / kinematic_viscosity)
            * std::tgamma(2.5 + 0.5 * fall_b)
            * std::pow(
                inverse_lambda, 2.5 + 0.5 * fall_b);
    const double expected_increment =
        dt * 2.0 * pi
        / (density * microphysics_constants::L_f)
        * intercept * ventilation
        * conductivity * temperature_c;
    const double expected_theta_rate =
        -microphysics_constants::L_f
        / microphysics_constants::cp
        * expected_increment / dt;

    REQUIRE(tendencies.dtheta(4, 4, 4)
            == Approx(expected_theta_rate).epsilon(2.0e-5));
    REQUIRE(tendencies.dqr(4, 4, 4)
            == Approx(expected_increment / dt).epsilon(2.0e-5));
    REQUIRE(tendencies.dqs(4, 4, 4)
            == Approx(-expected_increment / dt).epsilon(2.0e-5));
}

TEST_CASE("Milbrandt rain-ice collision freezes liquid and closes all moments",
          "[physics][microphysics][milbrandt][ice][collision][conservation]")
{
    setup_grid();
    constexpr double temperature_k = 265.15;
    constexpr double pressure_pa = 80000.0;
    constexpr double density = 1.0;
    constexpr double collision_dt = 0.01;
    theta.resize(
        NR, NTH, NZ,
        static_cast<float>(
            thermodynamics::temperature_to_theta(
                temperature_k, pressure_pa)));
    p.resize(NR, NTH, NZ, static_cast<float>(pressure_pa));
    rho.resize(NR, NTH, NZ, static_cast<float>(density));
    const float water_saturation = static_cast<float>(
        thermodynamics::saturation_mixing_ratio_water(
            temperature_k, pressure_pa));
    qv.resize(
        NR, NTH, NZ,
        std::nextafter(
            water_saturation,
            std::numeric_limits<float>::infinity()));
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 8.0e-4f);
    qi.resize(NR, NTH, NZ, 3.0e-4f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);

    auto scheme = create_microphysics_scheme("milbrandt");
    scheme->set_vertical_spacing_m(1.0e12);
    Tendencies tendencies;
    tendencies.resize(NR, NTH, NZ);
    scheme->compute_tendencies(
        p, theta, rho, qv, qc, qr, qi, qs, qg, qh, collision_dt,
        tendencies.dtheta, tendencies.dqv, tendencies.dqc,
        tendencies.dqr, tendencies.dqi, tendencies.dqs,
        tendencies.dqg, tendencies.dqh);
    Field3D no_ice = qi;
    no_ice.fill(0.0f);
    auto no_collision_scheme =
        create_microphysics_scheme("milbrandt");
    no_collision_scheme->set_vertical_spacing_m(1.0e12);
    Tendencies no_collision;
    no_collision.resize(NR, NTH, NZ);
    no_collision_scheme->compute_tendencies(
        p, theta, rho, qv, qc, qr, no_ice, qs, qg, qh,
        collision_dt,
        no_collision.dtheta, no_collision.dqv,
        no_collision.dqc, no_collision.dqr,
        no_collision.dqi, no_collision.dqs,
        no_collision.dqg, no_collision.dqh);

    constexpr int i = 4, j = 4, k = 4;
    const double total_water_rate =
        static_cast<double>(tendencies.dqv(i, j, k))
        + static_cast<double>(tendencies.dqc(i, j, k))
        + static_cast<double>(tendencies.dqr(i, j, k))
        + static_cast<double>(tendencies.dqi(i, j, k))
        + static_cast<double>(tendencies.dqs(i, j, k))
        + static_cast<double>(tendencies.dqg(i, j, k))
        + static_cast<double>(tendencies.dqh(i, j, k));
    REQUIRE(tendencies.dqr(i, j, k) < 0.0f);
    REQUIRE(tendencies.dqi(i, j, k) < 0.0f);
    REQUIRE(tendencies.dqg(i, j, k) > 0.0f);
    REQUIRE(tendencies.dqh(i, j, k) == 0.0f);
    REQUIRE(tendencies.dtheta(i, j, k) > 0.0f);
    REQUIRE(total_water_rate == Approx(0.0).margin(1.0e-9));

    // Literal generalized-gamma collision oracle. It intentionally calls no
    // Milbrandt distribution, fall-speed, or collection helper.
    constexpr double pi = 3.14159265358979323846;
    constexpr double c_r = 523.598775598;
    constexpr double c_i = 440.0;
    constexpr double d_r = 3.0;
    constexpr double d_i = 3.0;
    constexpr double rain_seed_diameter = 5.0e-4;
    constexpr double ice_seed_diameter = 1.0e-4;
    const double rain = static_cast<double>(qr(i, j, k));
    const double ice = static_cast<double>(qi(i, j, k));
    const double rain_number =
        rain / (c_r * std::pow(rain_seed_diameter, d_r));
    const double ice_number =
        ice / (c_i * std::pow(ice_seed_diameter, d_i));
    const double rain_inverse_lambda = std::cbrt(
        rain / (c_r * rain_number * std::tgamma(4.0)));
    const double ice_inverse_lambda = std::cbrt(
        ice / (c_i * ice_number * std::tgamma(4.0)));
    const double rain_intercept =
        rain_number / rain_inverse_lambda;
    const double ice_intercept =
        ice_number / ice_inverse_lambda;
    const auto moment = [](double intercept,
                           double inverse_lambda,
                           double exponent)
    {
        return intercept * std::tgamma(1.0 + exponent)
            * std::pow(inverse_lambda, 1.0 + exponent);
    };
    const double density_factor = std::sqrt(1.225 / density);
    const double rain_speed =
        density_factor * 149.1
        * std::pow(rain_inverse_lambda, 0.5)
        * std::tgamma(4.5) / std::tgamma(4.0);
    const double ice_speed =
        density_factor * 71.34
        * std::pow(ice_inverse_lambda, 0.6635)
        * std::tgamma(4.6635) / std::tgamma(4.0);
    const double relative_speed = std::sqrt(
        (rain_speed - ice_speed) * (rain_speed - ice_speed)
        + 0.04 * rain_speed * ice_speed);
    const double common =
        collision_dt * (pi / 4.0) * relative_speed;
    const double collision_number =
        common
        * (moment(ice_intercept, ice_inverse_lambda, 2.0)
             * moment(rain_intercept, rain_inverse_lambda, 0.0)
           + 2.0
             * moment(ice_intercept, ice_inverse_lambda, 1.0)
             * moment(rain_intercept, rain_inverse_lambda, 1.0)
           + moment(ice_intercept, ice_inverse_lambda, 0.0)
             * moment(rain_intercept, rain_inverse_lambda, 2.0));
    const double collected_ice = std::min(
        ice,
        common * c_i
        * (moment(ice_intercept, ice_inverse_lambda, 5.0)
             * moment(rain_intercept, rain_inverse_lambda, 0.0)
           + 2.0
             * moment(ice_intercept, ice_inverse_lambda, 4.0)
             * moment(rain_intercept, rain_inverse_lambda, 1.0)
           + moment(ice_intercept, ice_inverse_lambda, 3.0)
             * moment(rain_intercept, rain_inverse_lambda, 2.0)));
    const double collected_rain = std::min(
        rain,
        common * c_r
        * (moment(rain_intercept, rain_inverse_lambda, 5.0)
             * moment(ice_intercept, ice_inverse_lambda, 0.0)
           + 2.0
             * moment(rain_intercept, rain_inverse_lambda, 4.0)
             * moment(ice_intercept, ice_inverse_lambda, 1.0)
           + moment(rain_intercept, rain_inverse_lambda, 3.0)
             * moment(ice_intercept, ice_inverse_lambda, 2.0)));
    const double collided_ice_number = std::min(
        {ice_number, collision_number,
         collected_ice * ice_number / ice});
    const double collided_rain_number = std::min(
        {rain_number, collision_number,
         collected_rain * rain_number / rain});
    const double combined_mass = collected_ice + collected_rain;
    const double combined_number =
        combined_mass
        / (400.0 * (pi / 6.0)
           * std::pow(rain_seed_diameter, 3.0));
    const double produced_graupel_number = std::min(
        {combined_number,
         collided_ice_number, collided_rain_number});

    REQUIRE(tendencies.dqr(i, j, k)
                - no_collision.dqr(i, j, k)
            == Approx(-collected_rain / collision_dt).epsilon(3.0e-4));
    REQUIRE(tendencies.dqg(i, j, k)
            == Approx(combined_mass / collision_dt).epsilon(3.0e-4));
    const PortableSchemeState prepared = scheme->portable_state();
    const PortableSchemeState no_collision_prepared =
        no_collision_scheme->portable_state();
    REQUIRE(prepared.fields[7](i, j, k)
                - no_collision_prepared.fields[7](i, j, k)
            == Approx(
                -collided_rain_number / collision_dt)
                .epsilon(3.0e-4));
    REQUIRE(prepared.fields[8](i, j, k) < 0.0f);  // dNi/dt
    REQUIRE(prepared.fields[10](i, j, k)
            == Approx(
                produced_graupel_number / collision_dt)
                .epsilon(3.0e-4));
    REQUIRE(prepared.fields[11](i, j, k) == 0.0f); // dNh/dt
    std::string error;
    REQUIRE(scheme->validate_prognostic_update(collision_dt, error));
    REQUIRE(error.empty());
}

TEST_CASE("Milbrandt rain-snow collision matches the literal pair oracle",
          "[physics][microphysics][milbrandt][ice][collision][rain-snow]")
{
    setup_grid();
    constexpr double temperature_k = 264.15;
    constexpr double pressure_pa = 80000.0;
    constexpr double collision_dt = 0.01;
    theta.resize(
        NR, NTH, NZ,
        static_cast<float>(
            thermodynamics::temperature_to_theta(
                temperature_k, pressure_pa)));
    p.resize(NR, NTH, NZ, static_cast<float>(pressure_pa));
    rho.resize(NR, NTH, NZ, 1.0f);
    qv.resize(
        NR, NTH, NZ,
        static_cast<float>(
            thermodynamics::saturation_mixing_ratio_water(
                temperature_k, pressure_pa)));
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 8.0e-4f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 4.0e-4f);
    qg.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);

    const auto run =
        [&](const Field3D& rain_field,
            const Field3D& snow_field)
        {
            auto result_scheme =
                create_microphysics_scheme("milbrandt");
            result_scheme->set_vertical_spacing_m(1.0e12);
            Tendencies result;
            result.resize(NR, NTH, NZ);
            result_scheme->compute_tendencies(
                p, theta, rho, qv, qc, rain_field, qi,
                snow_field, qg, qh, collision_dt,
                result.dtheta, result.dqv, result.dqc,
                result.dqr, result.dqi, result.dqs,
                result.dqg, result.dqh);
            return std::make_pair(
                std::move(result_scheme), std::move(result));
        };

    Field3D zero_rain = qr;
    zero_rain.fill(0.0f);
    Field3D zero_snow = qs;
    zero_snow.fill(0.0f);
    auto [scheme, tendencies] = run(qr, qs);
    auto [no_snow_scheme, no_snow] = run(qr, zero_snow);
    auto [no_rain_scheme, no_rain] = run(zero_rain, qs);

    constexpr int i = 4, j = 4, k = 4;
    constexpr double pi = 3.14159265358979323846;
    constexpr double c_r = 523.598775598;
    constexpr double c_s = 0.1597;
    constexpr double d_s = 2.078;
    const double rain = static_cast<double>(qr(i, j, k));
    const double snow = static_cast<double>(qs(i, j, k));
    const double rain_number =
        rain / (c_r * std::pow(5.0e-4, 3.0));
    const double snow_number =
        snow / (c_s * std::pow(1.0e-3, d_s));
    const LiteralGammaOracle rain_psd =
        literal_gamma_oracle(
            1.0, rain, rain_number, 0.0, c_r, 3.0);
    const LiteralGammaOracle snow_psd =
        literal_gamma_oracle(
            1.0, snow, snow_number, 0.0, c_s, d_s);
    const double rain_speed = literal_mass_weighted_speed(
        rain_psd, 1.0, 0.0, 3.0, 149.1, 0.5);
    const double snow_speed = literal_mass_weighted_speed(
        snow_psd, 1.0, 0.0, d_s, 11.72, 0.41);
    const double relative_speed = std::sqrt(
        (rain_speed - snow_speed) * (rain_speed - snow_speed)
        + 0.04 * rain_speed * snow_speed);
    const LiteralPairCollisionOracle collision =
        literal_pair_collision_oracle(
            1.0, collision_dt, 1.0, relative_speed,
            snow, snow_number, snow_psd, 0.0, c_s, d_s,
            rain, rain_number, rain_psd, 0.0, c_r, 3.0);

    const double snow_bulk_density = std::min(
        500.0,
        c_s / (pi / 6.0)
            * std::pow(
                snow_psd.mean_mass_diameter, d_s - 3.0));
    const double maximum_diameter = std::max(
        snow_psd.mean_mass_diameter,
        rain_psd.mean_mass_diameter);
    const double combined_density =
        (snow_bulk_density
             * std::pow(snow_psd.mean_mass_diameter, 3.0)
         + microphysics_constants::rho_w
             * std::pow(rain_psd.mean_mass_diameter, 3.0))
        / std::pow(maximum_diameter, 3.0);
    REQUIRE(
        combined_density
        > 0.5 * (snow_bulk_density + 400.0));
    const double total_collision_mass =
        collision.first_mass + collision.second_mass;
    const double produced_graupel_number = std::min(
        {total_collision_mass
             / (400.0 * (pi / 6.0)
                * std::pow(maximum_diameter, 3.0)),
         collision.first_number, collision.second_number});

    REQUIRE(tendencies.dqg(i, j, k)
            == Approx(total_collision_mass / collision_dt)
                .epsilon(4.0e-4));
    REQUIRE(tendencies.dqr(i, j, k)
                - no_snow.dqr(i, j, k)
            == Approx(-collision.second_mass / collision_dt)
                .epsilon(4.0e-4));
    REQUIRE(tendencies.dqs(i, j, k)
                - no_rain.dqs(i, j, k)
            == Approx(-collision.first_mass / collision_dt)
                .epsilon(4.0e-4));

    const PortableSchemeState prepared = scheme->portable_state();
    const PortableSchemeState no_snow_prepared =
        no_snow_scheme->portable_state();
    const PortableSchemeState no_rain_prepared =
        no_rain_scheme->portable_state();
    REQUIRE(prepared.fields[7](i, j, k)
                - no_snow_prepared.fields[7](i, j, k)
            == Approx(-collision.second_number / collision_dt)
                .epsilon(4.0e-4));
    REQUIRE(prepared.fields[9](i, j, k)
                - no_rain_prepared.fields[9](i, j, k)
            == Approx(-collision.first_number / collision_dt)
                .epsilon(4.0e-4));
    REQUIRE(prepared.fields[10](i, j, k)
            == Approx(produced_graupel_number / collision_dt)
                .epsilon(4.0e-4));
    REQUIRE(prepared.fields[11](i, j, k) == 0.0f);
}

TEST_CASE("Milbrandt large-rain graupel collision routes to hail",
          "[physics][microphysics][milbrandt][ice][collision][rain-graupel][hail]")
{
    setup_grid();
    constexpr double temperature_k = 264.15;
    constexpr double pressure_pa = 80000.0;
    constexpr double collision_dt = 0.005;
    constexpr double c_r = 523.598775598;
    constexpr double c_g = 209.439510239;
    constexpr double rain_diameter = 3.0e-3;
    constexpr double graupel_diameter = 1.0e-3;
    theta.resize(
        NR, NTH, NZ,
        static_cast<float>(
            thermodynamics::temperature_to_theta(
                temperature_k, pressure_pa)));
    p.resize(NR, NTH, NZ, static_cast<float>(pressure_pa));
    rho.resize(NR, NTH, NZ, 1.0f);
    qv.resize(
        NR, NTH, NZ,
        static_cast<float>(
            thermodynamics::saturation_mixing_ratio_water(
                temperature_k, pressure_pa)));
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 1.0e-3f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 1.0e-3f);
    qh.resize(NR, NTH, NZ, 0.0f);

    const double rain_number =
        static_cast<double>(qr(4, 4, 4))
        / (c_r * std::pow(rain_diameter, 3.0));
    const auto run =
        [&](const Field3D& rain_field,
            const Field3D& graupel_field)
        {
            auto result_scheme =
                create_microphysics_scheme("milbrandt");
            result_scheme->set_vertical_spacing_m(1.0e12);
            Tendencies initialization;
            initialization.resize(NR, NTH, NZ);
            result_scheme->compute_tendencies(
                p, theta, rho, qv, qc, rain_field, qi, qs,
                graupel_field, qh, collision_dt,
                initialization.dtheta, initialization.dqv,
                initialization.dqc, initialization.dqr,
                initialization.dqi, initialization.dqs,
                initialization.dqg, initialization.dqh);
            PortableSchemeState initialized =
                result_scheme->portable_state();
            if (rain_field(4, 4, 4) > 0.0f)
                initialized.fields[1].fill(
                    static_cast<float>(rain_number));
            std::string restore_error;
            REQUIRE(result_scheme->restore_portable_state(
                initialized, restore_error));
            REQUIRE(restore_error.empty());

            Tendencies result;
            result.resize(NR, NTH, NZ);
            result_scheme->compute_tendencies(
                p, theta, rho, qv, qc, rain_field, qi, qs,
                graupel_field, qh, collision_dt,
                result.dtheta, result.dqv, result.dqc,
                result.dqr, result.dqi, result.dqs,
                result.dqg, result.dqh);
            return std::make_pair(
                std::move(result_scheme), std::move(result));
        };

    Field3D zero_rain = qr;
    zero_rain.fill(0.0f);
    Field3D zero_graupel = qg;
    zero_graupel.fill(0.0f);
    auto [scheme, tendencies] = run(qr, qg);
    auto [no_graupel_scheme, no_graupel] =
        run(qr, zero_graupel);
    auto [no_rain_scheme, no_rain] =
        run(zero_rain, qg);

    constexpr int i = 4, j = 4, k = 4;
    constexpr double pi = 3.14159265358979323846;
    const double rain = static_cast<double>(qr(i, j, k));
    const double graupel =
        static_cast<double>(qg(i, j, k));
    const double graupel_number =
        graupel / (c_g * std::pow(graupel_diameter, 3.0));
    const LiteralGammaOracle rain_psd =
        literal_gamma_oracle(
            1.0, rain, rain_number, 0.0, c_r, 3.0);
    const LiteralGammaOracle graupel_psd =
        literal_gamma_oracle(
            1.0, graupel, graupel_number,
            0.0, c_g, 3.0);
    const double rain_speed = literal_mass_weighted_speed(
        rain_psd, 1.0, 0.0, 3.0, 149.1, 0.5);
    const double graupel_speed = literal_mass_weighted_speed(
        graupel_psd, 1.0, 0.0, 3.0, 19.3, 0.37);
    double adjusted_temperature = static_cast<double>(
        static_cast<float>(
            thermodynamics::theta_to_temperature(
                theta(i, j, k), p(i, j, k))));
    double adjusted_vapor = static_cast<double>(qv(i, j, k));
    double adjusted_cloud = 0.0;
    adjusted_temperature = thermodynamics::saturation_adjustment(
        adjusted_temperature, p(i, j, k),
        adjusted_vapor, adjusted_cloud);
    adjusted_temperature =
        static_cast<double>(static_cast<float>(adjusted_temperature));
    const double dynamic_viscosity =
        1.72e-5 * (393.0 / (adjusted_temperature + 120.0))
        * std::pow(
            adjusted_temperature / microphysics_constants::T0,
            1.5);
    double stokes =
        microphysics_constants::rho_w
        * std::abs(graupel_speed - rain_speed)
        * rain_psd.mean_mass_diameter
        * rain_psd.mean_mass_diameter
        / (9.0 * dynamic_viscosity
           * graupel_psd.mean_mass_diameter);
    stokes = std::clamp(stokes, 1.5, 10.0);
    const double efficiency = std::clamp(
        0.55 * std::log10(2.51 * stokes), 0.0, 1.0);
    const double relative_speed = std::sqrt(
        (graupel_speed - rain_speed)
            * (graupel_speed - rain_speed)
        + 0.04 * graupel_speed * rain_speed);
    const LiteralPairCollisionOracle collision =
        literal_pair_collision_oracle(
            1.0, collision_dt, efficiency, relative_speed,
            graupel, graupel_number, graupel_psd,
            0.0, c_g, 3.0,
            rain, rain_number, rain_psd, 0.0, c_r, 3.0);

    const double maximum_diameter = std::max(
        graupel_psd.mean_mass_diameter,
        rain_psd.mean_mass_diameter);
    const double combined_density =
        (400.0 * std::pow(
             graupel_psd.mean_mass_diameter, 3.0)
         + microphysics_constants::rho_w
             * std::pow(rain_psd.mean_mass_diameter, 3.0))
        / std::pow(maximum_diameter, 3.0);
    REQUIRE(rain_psd.mean_mass_diameter > 2.5e-3);
    REQUIRE(combined_density >= 650.0);
    const double total_collision_mass =
        collision.first_mass + collision.second_mass;
    const double produced_hail_number = std::min(
        {total_collision_mass
             / (900.0 * (pi / 6.0)
                * std::pow(maximum_diameter, 3.0)),
         collision.first_number, collision.second_number});

    REQUIRE(tendencies.dqh(i, j, k)
            == Approx(total_collision_mass / collision_dt)
                .epsilon(5.0e-4));
    REQUIRE(tendencies.dqr(i, j, k)
                - no_graupel.dqr(i, j, k)
            == Approx(-collision.second_mass / collision_dt)
                .epsilon(5.0e-4));
    REQUIRE(tendencies.dqg(i, j, k)
                - no_rain.dqg(i, j, k)
            == Approx(-collision.first_mass / collision_dt)
                .epsilon(5.0e-4));

    const PortableSchemeState prepared = scheme->portable_state();
    const PortableSchemeState no_graupel_prepared =
        no_graupel_scheme->portable_state();
    const PortableSchemeState no_rain_prepared =
        no_rain_scheme->portable_state();
    REQUIRE(prepared.fields[7](i, j, k)
                - no_graupel_prepared.fields[7](i, j, k)
            == Approx(-collision.second_number / collision_dt)
                .epsilon(5.0e-4));
    REQUIRE(prepared.fields[10](i, j, k)
                - no_rain_prepared.fields[10](i, j, k)
            == Approx(-collision.first_number / collision_dt)
                .epsilon(5.0e-4));
    REQUIRE(prepared.fields[11](i, j, k)
            == Approx(produced_hail_number / collision_dt)
                .epsilon(5.0e-4));
}

TEST_CASE("Milbrandt riming converts snow to graupel conservatively",
          "[physics][microphysics][milbrandt][ice][riming][conservation]")
{
    setup_grid();
    constexpr double temperature_k = 265.15;
    constexpr double pressure_pa = 80000.0;
    theta.resize(
        NR, NTH, NZ,
        static_cast<float>(
            thermodynamics::temperature_to_theta(
                temperature_k, pressure_pa)));
    p.resize(NR, NTH, NZ, static_cast<float>(pressure_pa));
    rho.resize(NR, NTH, NZ, 1.0f);
    qv.resize(
        NR, NTH, NZ,
        static_cast<float>(
            thermodynamics::saturation_mixing_ratio_water(
                temperature_k, pressure_pa)));
    qc.resize(NR, NTH, NZ, 2.0e-3f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 1.0e-4f);
    qg.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);

    auto scheme = create_microphysics_scheme("milbrandt");
    scheme->set_vertical_spacing_m(1.0e12);
    Tendencies tendencies;
    tendencies.resize(NR, NTH, NZ);
    scheme->compute_tendencies(
        p, theta, rho, qv, qc, qr, qi, qs, qg, qh, dt,
        tendencies.dtheta, tendencies.dqv, tendencies.dqc,
        tendencies.dqr, tendencies.dqi, tendencies.dqs,
        tendencies.dqg, tendencies.dqh);

    constexpr int i = 4, j = 4, k = 4;
    const double total_water_rate =
        static_cast<double>(tendencies.dqv(i, j, k))
        + static_cast<double>(tendencies.dqc(i, j, k))
        + static_cast<double>(tendencies.dqr(i, j, k))
        + static_cast<double>(tendencies.dqi(i, j, k))
        + static_cast<double>(tendencies.dqs(i, j, k))
        + static_cast<double>(tendencies.dqg(i, j, k))
        + static_cast<double>(tendencies.dqh(i, j, k));
    REQUIRE(tendencies.dqc(i, j, k) < 0.0f);
    REQUIRE(tendencies.dqg(i, j, k) > 0.0f);
    REQUIRE(tendencies.dqh(i, j, k) == 0.0f);
    REQUIRE(total_water_rate == Approx(0.0).margin(3.0e-10));

    const PortableSchemeState prepared = scheme->portable_state();
    REQUIRE(prepared.fields[10](i, j, k) > 0.0f); // dNg/dt

    // Independent literal oracle for the published cloud-snow collection
    // integral followed by the riming conversion QCNsg. The oracle uses only
    // constants and the initialized dry-air moments, never production
    // distribution or collection helpers.
    constexpr double c_s = 0.1597;
    constexpr double d_s = 2.078;
    const double cloud =
        static_cast<double>(qc(i, j, k));
    const double snow =
        static_cast<double>(qs(i, j, k));
    const double snow_number =
        snow / (c_s * std::pow(1.0e-3, d_s));
    const LiteralCloudCollectionOracle cloud_snow =
        literal_cloud_collection_oracle(
            1.0, temperature_k, dt,
            cloud, 2.0e8, snow, snow_number,
            0.0, c_s, d_s, 11.72, 0.41, 0);
    const double expected_snow_to_graupel = std::min(
        snow + cloud_snow.mass,
        (400.0 / (400.0 - 100.0)) * cloud_snow.mass);
    const double expected_snow_number_to_graupel = std::min(
        snow_number,
        snow_number * expected_snow_to_graupel
            / (snow + cloud_snow.mass));

    REQUIRE(tendencies.dqg(i, j, k)
            == Approx(expected_snow_to_graupel / dt)
                .epsilon(4.0e-4));
    REQUIRE(prepared.fields[10](i, j, k)
            == Approx(expected_snow_number_to_graupel / dt)
                .epsilon(4.0e-4));
}

TEST_CASE("Milbrandt donor admission does not throttle independent ice",
          "[physics][microphysics][milbrandt][ice][admission]")
{
    setup_grid();
    constexpr double temperature_k = 258.15;
    constexpr double pressure_pa = 75000.0;
    constexpr double process_dt = 900.0;
    theta.resize(
        NR, NTH, NZ,
        static_cast<float>(
            thermodynamics::temperature_to_theta(
                temperature_k, pressure_pa)));
    p.resize(NR, NTH, NZ, static_cast<float>(pressure_pa));
    rho.resize(NR, NTH, NZ, 1.0f);
    const float water_saturation = static_cast<float>(
        thermodynamics::saturation_mixing_ratio_water(
            temperature_k, pressure_pa));
    qv.resize(
        NR, NTH, NZ,
        std::nextafter(
            water_saturation,
            std::numeric_limits<float>::infinity()));
    qc.resize(NR, NTH, NZ, 2.0e-7f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 2.0e-4f);
    qs.resize(NR, NTH, NZ, 1.5e-3f);
    qg.resize(NR, NTH, NZ, 1.5e-3f);
    qh.resize(NR, NTH, NZ, 0.0f);

    auto coupled_scheme = create_microphysics_scheme("milbrandt");
    coupled_scheme->set_vertical_spacing_m(1.0e12);
    Tendencies coupled;
    coupled.resize(NR, NTH, NZ);
    coupled_scheme->compute_tendencies(
        p, theta, rho, qv, qc, qr, qi, qs, qg, qh, process_dt,
        coupled.dtheta, coupled.dqv, coupled.dqc, coupled.dqr,
        coupled.dqi, coupled.dqs, coupled.dqg, coupled.dqh);

    Field3D no_cloud = qc;
    no_cloud.fill(0.0f);
    auto independent_scheme = create_microphysics_scheme("milbrandt");
    independent_scheme->set_vertical_spacing_m(1.0e12);
    Tendencies independent;
    independent.resize(NR, NTH, NZ);
    independent_scheme->compute_tendencies(
        p, theta, rho, qv, no_cloud, qr, qi, qs, qg, qh,
        process_dt,
        independent.dtheta, independent.dqv, independent.dqc,
        independent.dqr, independent.dqi, independent.dqs,
        independent.dqg, independent.dqh);

    constexpr int i = 4, j = 4, k = 4;
    const double remaining_cloud =
        static_cast<double>(qc(i, j, k))
        + process_dt * static_cast<double>(coupled.dqc(i, j, k));
    CAPTURE(
        remaining_cloud, coupled.dqc(i, j, k),
        coupled.dqi(i, j, k), independent.dqi(i, j, k),
        coupled.dqg(i, j, k), independent.dqg(i, j, k));
    REQUIRE(coupled.dqc(i, j, k) < 0.0f);
    REQUIRE(remaining_cloud >= -2.0e-12);
    REQUIRE(remaining_cloud <= 2.0e-10);
    REQUIRE(coupled.dqi(i, j, k)
            == Approx(independent.dqi(i, j, k)).margin(2.0e-10));

    const PortableSchemeState coupled_state =
        coupled_scheme->portable_state();
    const PortableSchemeState independent_state =
        independent_scheme->portable_state();
    REQUIRE(coupled_state.fields[8](i, j, k)
            == Approx(independent_state.fields[8](i, j, k))
                .margin(2.0e-7));

    const double total_water_rate =
        static_cast<double>(coupled.dqv(i, j, k))
        + static_cast<double>(coupled.dqc(i, j, k))
        + static_cast<double>(coupled.dqr(i, j, k))
        + static_cast<double>(coupled.dqi(i, j, k))
        + static_cast<double>(coupled.dqs(i, j, k))
        + static_cast<double>(coupled.dqg(i, j, k))
        + static_cast<double>(coupled.dqh(i, j, k));
    REQUIRE(total_water_rate == Approx(0.0).margin(2.0e-10));
}

TEST_CASE("Milbrandt Schumann-Ludlam tail converts rimed graupel to hail",
          "[physics][microphysics][milbrandt][ice][hail][conservation]")
{
    setup_grid();
    constexpr double temperature_k = 272.15;
    constexpr double pressure_pa = 90000.0;
    theta.resize(
        NR, NTH, NZ,
        static_cast<float>(
            thermodynamics::temperature_to_theta(
                temperature_k, pressure_pa)));
    p.resize(NR, NTH, NZ, static_cast<float>(pressure_pa));
    rho.resize(NR, NTH, NZ, 1.0f);
    qv.resize(
        NR, NTH, NZ,
        static_cast<float>(
            thermodynamics::saturation_mixing_ratio_water(
                temperature_k, pressure_pa)));
    qc.resize(NR, NTH, NZ, 1.0e-2f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 1.0e-3f);
    qh.resize(NR, NTH, NZ, 0.0f);

    auto scheme = create_microphysics_scheme("milbrandt");
    scheme->set_vertical_spacing_m(1.0e12);
    Tendencies tendencies;
    tendencies.resize(NR, NTH, NZ);
    scheme->compute_tendencies(
        p, theta, rho, qv, qc, qr, qi, qs, qg, qh, dt,
        tendencies.dtheta, tendencies.dqv, tendencies.dqc,
        tendencies.dqr, tendencies.dqi, tendencies.dqs,
        tendencies.dqg, tendencies.dqh);

    constexpr int i = 4, j = 4, k = 4;
    const double total_water_rate =
        static_cast<double>(tendencies.dqv(i, j, k))
        + static_cast<double>(tendencies.dqc(i, j, k))
        + static_cast<double>(tendencies.dqr(i, j, k))
        + static_cast<double>(tendencies.dqi(i, j, k))
        + static_cast<double>(tendencies.dqs(i, j, k))
        + static_cast<double>(tendencies.dqg(i, j, k))
        + static_cast<double>(tendencies.dqh(i, j, k));
    REQUIRE(tendencies.dqh(i, j, k) > 0.0f);
    REQUIRE(total_water_rate == Approx(0.0).margin(4.0e-10));

    const PortableSchemeState prepared = scheme->portable_state();
    REQUIRE(prepared.fields[10](i, j, k) < 0.0f); // dNg/dt
    REQUIRE(prepared.fields[11](i, j, k) > 0.0f); // dNh/dt

    // Literal Schumann-Ludlam tail oracle. Cloud collection first changes
    // the diagnosed graupel distribution; only the analytical tail beyond
    // the wet-growth size limit changes category.
    constexpr double c_g = 209.439510239;
    constexpr double d_g = 3.0;
    const double cloud =
        static_cast<double>(qc(i, j, k));
    const double graupel =
        static_cast<double>(qg(i, j, k));
    const double graupel_number =
        graupel / (c_g * std::pow(1.0e-3, d_g));
    const LiteralCloudCollectionOracle cloud_graupel =
        literal_cloud_collection_oracle(
            1.0, temperature_k, dt,
            cloud, 2.0e8, graupel, graupel_number,
            0.0, c_g, d_g, 19.3, 0.37, 1);
    const double temperature_c =
        temperature_k - microphysics_constants::T0;
    const double liquid_loading =
        std::max(1.0, 1.1e4 * cloud + 1.0);
    const double size_limit = std::clamp(
        0.02
            * (std::exp(
                   std::min(
                       20.0, -temperature_c / liquid_loading))
               - 1.0),
        1.0e-4, 1.0);
    const double post_riming_graupel =
        graupel + cloud_graupel.mass;
    const LiteralGammaOracle post_riming =
        literal_gamma_oracle(
            1.0, post_riming_graupel, graupel_number,
            0.0, c_g, d_g);
    const double exponential_tail =
        std::exp(-size_limit / post_riming.inverse_lambda);
    const double tail_number =
        post_riming.intercept * post_riming.inverse_lambda
        * exponential_tail;
    const double d = size_limit;
    const double tail_mass =
        c_g * post_riming.intercept * exponential_tail
        * (d * d * d * post_riming.inverse_lambda
           + 3.0 * d * d
             * post_riming.inverse_lambda
             * post_riming.inverse_lambda
           + 6.0 * d
             * std::pow(post_riming.inverse_lambda, 3.0)
           + 6.0
             * std::pow(post_riming.inverse_lambda, 4.0));
    const double expected_graupel_to_hail =
        tail_number > 1.0
        ? std::min(graupel, tail_mass)
        : 0.0;
    const double expected_graupel_number_to_hail =
        tail_number > 1.0
        ? std::min(graupel_number, tail_number)
        : 0.0;

    REQUIRE(tendencies.dqh(i, j, k)
            == Approx(expected_graupel_to_hail / dt)
                .epsilon(4.0e-4));
    REQUIRE(prepared.fields[11](i, j, k)
            == Approx(expected_graupel_number_to_hail / dt)
                .epsilon(4.0e-4));
}

TEST_CASE("Milbrandt hail wet growth retains bounded rain and closes water",
          "[physics][microphysics][milbrandt][ice][hail][wet-growth]")
{
    setup_grid();
    constexpr double temperature_k = 268.15;
    constexpr double pressure_pa = 85000.0;
    theta.resize(
        NR, NTH, NZ,
        static_cast<float>(
            thermodynamics::temperature_to_theta(
                temperature_k, pressure_pa)));
    p.resize(NR, NTH, NZ, static_cast<float>(pressure_pa));
    rho.resize(NR, NTH, NZ, 1.0f);
    qv.resize(
        NR, NTH, NZ,
        static_cast<float>(
            thermodynamics::saturation_mixing_ratio_water(
                temperature_k, pressure_pa)));
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 1.0e-2f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 5.0e-3f);

    auto scheme = create_microphysics_scheme("milbrandt");
    scheme->set_vertical_spacing_m(1.0e12);
    Tendencies tendencies;
    tendencies.resize(NR, NTH, NZ);
    scheme->compute_tendencies(
        p, theta, rho, qv, qc, qr, qi, qs, qg, qh, dt,
        tendencies.dtheta, tendencies.dqv, tendencies.dqc,
        tendencies.dqr, tendencies.dqi, tendencies.dqs,
        tendencies.dqg, tendencies.dqh);

    Field3D no_rain = qr;
    no_rain.fill(0.0f);
    auto hail_only_scheme =
        create_microphysics_scheme("milbrandt");
    hail_only_scheme->set_vertical_spacing_m(1.0e12);
    Tendencies hail_only;
    hail_only.resize(NR, NTH, NZ);
    hail_only_scheme->compute_tendencies(
        p, theta, rho, qv, qc, no_rain, qi, qs, qg, qh, dt,
        hail_only.dtheta, hail_only.dqv, hail_only.dqc,
        hail_only.dqr, hail_only.dqi, hail_only.dqs,
        hail_only.dqg, hail_only.dqh);

    Field3D no_hail = qh;
    no_hail.fill(0.0f);
    auto rain_only_scheme =
        create_microphysics_scheme("milbrandt");
    rain_only_scheme->set_vertical_spacing_m(1.0e12);
    Tendencies rain_only;
    rain_only.resize(NR, NTH, NZ);
    rain_only_scheme->compute_tendencies(
        p, theta, rho, qv, qc, qr, qi, qs, qg, no_hail, dt,
        rain_only.dtheta, rain_only.dqv, rain_only.dqc,
        rain_only.dqr, rain_only.dqi, rain_only.dqs,
        rain_only.dqg, rain_only.dqh);

    constexpr int i = 4, j = 4, k = 4;
    const double total_water_rate =
        static_cast<double>(tendencies.dqv(i, j, k))
        + static_cast<double>(tendencies.dqc(i, j, k))
        + static_cast<double>(tendencies.dqr(i, j, k))
        + static_cast<double>(tendencies.dqi(i, j, k))
        + static_cast<double>(tendencies.dqs(i, j, k))
        + static_cast<double>(tendencies.dqg(i, j, k))
        + static_cast<double>(tendencies.dqh(i, j, k));
    REQUIRE(tendencies.dqr(i, j, k) < 0.0f);
    REQUIRE(tendencies.dqh(i, j, k) > 0.0f);
    REQUIRE(tendencies.dtheta(i, j, k) > 0.0f);
    REQUIRE(total_water_rate == Approx(0.0).margin(4.0e-10));

    const PortableSchemeState prepared = scheme->portable_state();
    REQUIRE(prepared.fields[7](i, j, k) < 0.0f); // dNr/dt
    REQUIRE(prepared.fields[11](i, j, k) == 0.0f); // collector Nh unchanged

    constexpr double pi = 3.14159265358979323846;
    constexpr double c_r = 523.598775598;
    constexpr double c_h = 471.238898038;
    const double rain =
        static_cast<double>(qr(i, j, k));
    const double hail =
        static_cast<double>(qh(i, j, k));
    const PortableSchemeState rain_only_prepared =
        rain_only_scheme->portable_state();
    const double rain_number =
        static_cast<double>(rain_only_prepared.fields[1](i, j, k))
        + dt * static_cast<double>(
            rain_only_prepared.fields[7](i, j, k));
    const double hail_number =
        hail / (c_h * std::pow(5.0e-3, 3.0));
    const LiteralGammaOracle rain_psd =
        literal_gamma_oracle(
            1.0, rain, rain_number, 0.0, c_r, 3.0);
    const LiteralGammaOracle hail_psd =
        literal_gamma_oracle(
            1.0, hail, hail_number, 0.0, c_h, 3.0);
    const double rain_speed = literal_mass_weighted_speed(
        rain_psd, 1.0, 0.0, 3.0, 149.1, 0.5);
    const double hail_speed = literal_mass_weighted_speed(
        hail_psd, 1.0, 0.0, 3.0, 206.89, 0.6384);
    const double relative_speed = std::sqrt(
        (rain_speed - hail_speed) * (rain_speed - hail_speed)
        + 0.04 * rain_speed * hail_speed);
    const LiteralPairCollisionOracle rain_hail =
        literal_pair_collision_oracle(
            1.0, dt, 1.0, relative_speed,
            rain, rain_number, rain_psd, 0.0, c_r, 3.0,
            hail, hail_number, hail_psd, 0.0, c_h, 3.0);

    double adjusted_temperature = static_cast<double>(
        static_cast<float>(
            thermodynamics::theta_to_temperature(
                theta(i, j, k), p(i, j, k))));
    double vapor = static_cast<double>(qv(i, j, k));
    double adjusted_cloud = static_cast<double>(qc(i, j, k));
    adjusted_temperature = thermodynamics::saturation_adjustment(
        adjusted_temperature, static_cast<double>(p(i, j, k)),
        vapor, adjusted_cloud);
    adjusted_temperature =
        static_cast<double>(static_cast<float>(adjusted_temperature));
    vapor = static_cast<double>(static_cast<float>(vapor));
    const double temperature_c =
        adjusted_temperature - microphysics_constants::T0;
    const double vapor_diffusivity =
        (2.2157e-5 + 0.0155e-5 * temperature_c)
        * 1.0e5 / pressure_pa;
    const double dynamic_viscosity =
        1.72e-5 * (393.0 / (adjusted_temperature + 120.0))
        * std::pow(
            adjusted_temperature / microphysics_constants::T0, 1.5);
    const double kinematic_viscosity = dynamic_viscosity;
    const double schmidt_one_third =
        std::cbrt(kinematic_viscosity / vapor_diffusivity);
    const double thermal_conductivity =
        2.3971e-2 + 0.0078e-2 * temperature_c;
    const double hail_ventilation =
        0.78 * std::tgamma(2.0)
            * std::pow(hail_psd.inverse_lambda, 2.0)
        + 0.30 * schmidt_one_third
            * std::sqrt(
                std::sqrt(1.225) * 206.89
                / kinematic_viscosity)
            * std::tgamma(2.5 + 0.5 * 0.6384)
            * std::pow(
                hail_psd.inverse_lambda,
                2.5 + 0.5 * 0.6384);
    constexpr double liquid_heat_capacity = 4218.0;
    const double wet_growth_denominator =
        microphysics_constants::L_f
        + liquid_heat_capacity * temperature_c;
    const double water_saturation_at_freezing =
        thermodynamics::saturation_mixing_ratio_water(
            microphysics_constants::T0, pressure_pa);
    const double wet_growth_capacity = std::max(
        0.0,
        dt * 2.0 * pi
            * (microphysics_constants::L_v
                   * vapor_diffusivity
                   * (water_saturation_at_freezing - vapor)
               - thermal_conductivity * temperature_c)
            * hail_psd.intercept
            / wet_growth_denominator
            * hail_ventilation);
    const double retained_rain =
        temperature_c > -40.0
            && wet_growth_capacity < rain_hail.first_mass
        ? std::clamp(
              wet_growth_capacity, 0.0, rain_hail.first_mass)
        : rain_hail.first_mass;
    const double shed_rain =
        rain_hail.first_mass - retained_rain;
    const double shed_rain_number =
        shed_rain / (c_r * std::pow(1.0e-3, 3.0));

    CAPTURE(
        rain_hail.first_mass, rain_hail.first_number,
        wet_growth_capacity, retained_rain, shed_rain,
        adjusted_temperature, vapor,
        1.0e9 * (static_cast<double>(tendencies.dqh(i, j, k))
                 - static_cast<double>(hail_only.dqh(i, j, k))),
        1.0e9 * retained_rain,
        tendencies.dqh(i, j, k), hail_only.dqh(i, j, k),
        tendencies.dqr(i, j, k), rain_only.dqr(i, j, k));
    REQUIRE(
        tendencies.dqh(i, j, k) - hail_only.dqh(i, j, k)
        == Approx(retained_rain / dt).epsilon(5.0e-4));
    REQUIRE(
        tendencies.dqr(i, j, k) - rain_only.dqr(i, j, k)
        == Approx(-retained_rain / dt).epsilon(5.0e-4));
    REQUIRE(
        prepared.fields[7](i, j, k)
            - rain_only_prepared.fields[7](i, j, k)
        == Approx(
            (-rain_hail.first_number + shed_rain_number) / dt)
            .epsilon(5.0e-4));
}

TEST_CASE("Milbrandt always sediments existing hail when production is disabled",
          "[physics][microphysics][milbrandt][hail][sedimentation][config]")
{
    setup_grid();
    constexpr double temperature_k = 268.15;
    constexpr double pressure_pa = 85000.0;
    theta.resize(
        NR, NTH, NZ,
        static_cast<float>(
            thermodynamics::temperature_to_theta(
                temperature_k, pressure_pa)));
    p.resize(NR, NTH, NZ, static_cast<float>(pressure_pa));
    rho.resize(NR, NTH, NZ, 1.0f);
    qv.resize(
        NR, NTH, NZ,
        static_cast<float>(
            thermodynamics::saturation_mixing_ratio_ice(
                temperature_k, pressure_pa)));
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
    constexpr int i = 4, j = 4, k = 0;
    qh(i, j, k) = 2.0e-3f;

    tmv::MicrophysicsConfig config;
    config.milbrandt.hail_processes = false;
    auto scheme = create_microphysics_scheme("milbrandt", config);
    Tendencies tendencies;
    compute(*scheme, dt, tendencies);

    REQUIRE(tendencies.dqh(i, j, k) < 0.0f);
    REQUIRE(scheme->bottom_sedimentation_sink_rate(
                rho, qr, qs, qg, qh, i, j, dt) > 0.0);
    const double total_water_rate =
        static_cast<double>(tendencies.dqv(i, j, k))
        + static_cast<double>(tendencies.dqc(i, j, k))
        + static_cast<double>(tendencies.dqr(i, j, k))
        + static_cast<double>(tendencies.dqi(i, j, k))
        + static_cast<double>(tendencies.dqs(i, j, k))
        + static_cast<double>(tendencies.dqg(i, j, k))
        + static_cast<double>(tendencies.dqh(i, j, k));
    REQUIRE(-total_water_rate
            == Approx(
                scheme->bottom_sedimentation_sink_rate(
                    rho, qr, qs, qg, qh, i, j, dt))
                .epsilon(2.0e-6));
}

TEST_CASE("Milbrandt and Morrison remain distinct scheme identities",
          "[physics][microphysics][descriptor]")
{
    const auto& descriptors = get_microphysics_scheme_descriptors();
    const auto* milbrandt = tmv::find_scheme_descriptor(descriptors, "milbrandt");
    REQUIRE(milbrandt != nullptr);
    REQUIRE(milbrandt->id == "milbrandt");
    REQUIRE(milbrandt->maturity == tmv::SchemeMaturity::Qualified);
    REQUIRE(milbrandt->runtime_selectable);
    REQUIRE(tmv::find_scheme_descriptor(descriptors, "morrison") == nullptr);
    REQUIRE_THROWS(create_microphysics_scheme("MORRISON"));

    for (const auto& descriptor : descriptors)
    {
        REQUIRE(tmv::has_complete_scheme_capabilities(descriptor));
        if (descriptor.id == "none" || descriptor.id == "kessler" ||
            descriptor.id == "lin" || descriptor.id == "thompson" ||
            descriptor.id == "milbrandt")
        {
            REQUIRE(descriptor.capabilities.staggerings
                    == std::vector<std::string>{"collocated", "c_grid"});
        }
        else
        {
            REQUIRE(descriptor.capabilities.staggerings
                    == std::vector<std::string>{"c_grid"});
        }
    }
}

TEST_CASE("Kessler honors the collocated physical-cell domain and surface",
          "[physics][microphysics][kessler][agrid][ownership][sci104]")
{
    constexpr int nr = 4;
    constexpr int nth = 4;
    constexpr int nz = 5;
    constexpr float poison = std::numeric_limits<float>::quiet_NaN();
    const PhysicalCellDomain domain{1, 3, 1, 3, 1, 4};

    Field3D p(nr, nth, nz, poison);
    Field3D theta(nr, nth, nz, poison);
    Field3D rho_d(nr, nth, nz, poison);
    Field3D qv(nr, nth, nz, poison);
    Field3D qc(nr, nth, nz, poison);
    Field3D qr(nr, nth, nz, poison);
    Field3D qi(nr, nth, nz, poison);
    Field3D qs(nr, nth, nz, poison);
    Field3D qg(nr, nth, nz, poison);
    Field3D qh(nr, nth, nz, poison);
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                p(i, j, k) = 90000.0f;
                theta(i, j, k) = 300.0f;
                rho_d(i, j, k) = 1.0f;
                qv(i, j, k) = 0.010f;
                qc(i, j, k) = 0.0f;
                qr(i, j, k) = k == domain.k_begin ? 1.0e-3f : 0.0f;
                qi(i, j, k) = 0.0f;
                qs(i, j, k) = 0.0f;
                qg(i, j, k) = 0.0f;
                qh(i, j, k) = 0.0f;
            }

    auto scheme = create_microphysics_scheme("kessler");
    scheme->set_vertical_spacing_m(100.0);
    scheme->set_physical_cell_domain(nr, nth, nz, domain);
    Field3D dtheta, dqv, dqc, dqr, dqi, dqs, dqg, dqh;
    REQUIRE_NOTHROW(scheme->compute_tendencies(
        p, theta, rho_d, qv, qc, qr, qi, qs, qg, qh, 1.0,
        dtheta, dqv, dqc, dqr, dqi, dqs, dqg, dqh));

    const std::array<const Field3D*, 8> tendencies{{
        &dtheta, &dqv, &dqc, &dqr, &dqi, &dqs, &dqg, &dqh}};
    for (const Field3D* tendency : tendencies)
    {
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nth; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const bool physical = i >= domain.i_begin && i < domain.i_end
                        && j >= domain.j_begin && j < domain.j_end
                        && k >= domain.k_begin && k < domain.k_end;
                    if (physical)
                        REQUIRE(std::isfinite((*tendency)(i, j, k)));
                    else
                        REQUIRE((*tendency)(i, j, k) == 0.0f);
                }
    }
    REQUIRE(dqr(1, 1, domain.k_begin) < 0.0f);
    REQUIRE(dqr(1, 1, 0) == 0.0f);
    REQUIRE(scheme->bottom_sedimentation_sink_rate(
        rho_d, qr, qs, qg, qh, 1, 1, 1.0) > 0.0);
}

TEST_CASE("Implemented microphysics schemes use descriptors for qualification scope",
          "[physics][microphysics]")
{
    setup_grid();

    REQUIRE(create_microphysics_scheme("none")->capability_caveat().empty());
    REQUIRE(create_microphysics_scheme("kessler")->capability_caveat().empty());
    REQUIRE(create_microphysics_scheme("lin")->capability_caveat().empty());
    REQUIRE(create_microphysics_scheme("milbrandt")
                ->capability_caveat().empty());
    REQUIRE(create_microphysics_scheme("thompson")
                ->capability_caveat().empty());
}

TEST_CASE("qualified Kessler rejects a missing native pointwise operator",
          "[physics][microphysics][kessler][qualified][fallback]")
{
    setup_grid();
    setup_warm_state();
    auto scheme = create_microphysics_scheme("kessler");
    scheme->bind_execution_owners(global_compute_backend_runtime_state, nullptr);
    scheme->set_require_native_dispatch(true);
    Field3D dtheta, dqv, dqc, dqr, dqi, dqs, dqg, dqh;

    REQUIRE_THROWS_WITH(
        scheme->compute_tendencies(
            p, theta, rho, qv, qc, qr, qi, qs, qg, qh, 1.0,
            dtheta, dqv, dqc, dqr, dqi, dqs, dqg, dqh),
        "qualified accelerator Kessler requires the native pointwise operator");
}

TEST_CASE("qualified Lin rejects missing native process ownership",
          "[physics][microphysics][lin][qualified][fallback]")
{
    setup_grid();
    setup_warm_state();
    qv.resize(NR, NTH, NZ, 0.012f);
    qc.resize(NR, NTH, NZ, 0.002f);
    qr.resize(NR, NTH, NZ, 0.001f);
    auto scheme = create_microphysics_scheme("lin");
    scheme->set_vertical_spacing_m(dz);
    scheme->bind_execution_owners(global_compute_backend_runtime_state, nullptr);
    scheme->set_require_native_dispatch(true);
    Field3D dtheta(NR, NTH, NZ, 1.0f), dqv(NR, NTH, NZ, 2.0f),
        dqc(NR, NTH, NZ, 3.0f), dqr(NR, NTH, NZ, 4.0f),
        dqi(NR, NTH, NZ, 5.0f), dqs(NR, NTH, NZ, 6.0f),
        dqg(NR, NTH, NZ, 7.0f), dqh(NR, NTH, NZ, 8.0f);
    const std::array<Field3D, 8> before{
        dtheta, dqv, dqc, dqr, dqi, dqs, dqg, dqh};

    REQUIRE_THROWS_WITH(scheme->compute_tendencies(
                            p, theta, rho, qv, qc, qr, qi, qs, qg, qh, 1.0, dtheta, dqv, dqc, dqr, dqi, dqs, dqg, dqh),
                        "qualified accelerator Lin requires an explicit compute owner");

    for (const Field3D* field : {&theta, &qv, &qc, &qr, &qi, &qs, &qg, &qh})
    {
        for (std::size_t n = 0; n < field->size(); ++n)
            REQUIRE(std::isfinite(field->data()[n]));
    }
    REQUIRE(theta(1, 1, 1) == 300.0f);
    REQUIRE(qv(1, 1, 1) == 0.012f);
    REQUIRE(qc(1, 1, 1) == 0.002f);
    REQUIRE(qr(1, 1, 1) == 0.001f);
    const std::array<const Field3D*, 8> after{
        &dtheta, &dqv, &dqc, &dqr, &dqi, &dqs, &dqg, &dqh};
    for (std::size_t n = 0; n < before.size(); ++n)
        require_same_field(before[n], *after[n]);
}

TEST_CASE("Lin sedimentation validation propagates parallel errors deterministically",
          "[physics][microphysics][lin][thread][validation]")
{
#ifdef _OPENMP
    struct RestoreWorkers
    {
        int threads = omp_get_max_threads();
        int dynamic = omp_get_dynamic();
        ~RestoreWorkers() { omp_set_num_threads(threads); omp_set_dynamic(dynamic); }
    } restore;
    omp_set_dynamic(0);
    omp_set_num_threads(GENERATE(1, 4));
#endif
    struct Probe : tmv::LinScheme
    {
        using MicrophysicsScheme::require_stable_sedimentation;
    } scheme;
    scheme.set_vertical_spacing_m(100.0);
    Field3D density(12, 10, 7, 1.0f), rain(12, 10, 7, .001f);
    const std::array<const Field3D*, 4> species{&rain, nullptr, nullptr, nullptr};
    REQUIRE_NOTHROW(scheme.require_stable_sedimentation(density, species, 1.0));
    density[1][1][2] = -1;
    rain[10][8][3] = -1;
    const auto density_before = density, rain_before = rain;
    REQUIRE_THROWS_WITH(scheme.require_stable_sedimentation(density, species, 1.0),
                       "sedimentation density must be positive and finite");
    require_same_field(density_before, density);
    require_same_field(rain_before, rain);
    density[1][1][2] = 1;
    rain[10][8][3] = .05f;
    REQUIRE_THROWS_WITH(scheme.require_stable_sedimentation(density, species, 10.0),
                       "sedimentation timestep exceeds the physical terminal-velocity CFL limit");
    rain[10][8][3] = .001f;
    REQUIRE_NOTHROW(scheme.require_stable_sedimentation(density, species, 1.0));
}

TEST_CASE("qualified Lin split dispatch is failure atomic",
          "[physics][microphysics][lin][qualified][fallback]")
{
    setup_grid();
    setup_warm_state();
    qv.resize(NR, NTH, NZ, 0.012f);
    qc.resize(NR, NTH, NZ, 0.002f);
    qr.resize(NR, NTH, NZ, 0.001f);
    const std::array<Field3D, 8> primary_before{
        theta, qv, qc, qr, qi, qs, qg, qh};

    ComputeRuntimeReset reset;
    global_compute_backend_runtime_state.backend =
        std::make_unique<PartialLinBackend>();
    global_compute_backend_runtime_state.active_kind = ComputeBackendKind::Metal;
    global_compute_backend_runtime_state.requested_kind = ComputeBackendKind::Metal;
    global_compute_backend_runtime_state.fallback_active = false;

    auto scheme = create_microphysics_scheme("lin");
    scheme->set_vertical_spacing_m(dz);
    scheme->bind_execution_owners(global_compute_backend_runtime_state, nullptr);
    scheme->set_require_native_dispatch(true);
    Field3D dtheta(NR, NTH, NZ, 1.0f), dqv(NR, NTH, NZ, 2.0f),
        dqc(NR, NTH, NZ, 3.0f), dqr(NR, NTH, NZ, 4.0f),
        dqi(NR, NTH, NZ, 5.0f), dqs(NR, NTH, NZ, 6.0f),
        dqg(NR, NTH, NZ, 7.0f), dqh(NR, NTH, NZ, 8.0f);
    const std::array<Field3D, 8> tendency_before{
        dtheta, dqv, dqc, dqr, dqi, dqs, dqg, dqh};

    REQUIRE_THROWS_WITH(
        scheme->compute_tendencies(
            p, theta, rho, qv, qc, qr, qi, qs, qg, qh, 1.0e9,
            dtheta, dqv, dqc, dqr, dqi, dqs, dqg, dqh),
        "sedimentation timestep exceeds the physical terminal-velocity CFL limit");

    REQUIRE_THROWS_WITH(
        scheme->compute_tendencies(
            p, theta, rho, qv, qc, qr, qi, qs, qg, qh, 1.0,
            dtheta, dqv, dqc, dqr, dqi, dqs, dqg, dqh),
        "qualified accelerator Lin requires the native sedimentation operator");

    const std::array<const Field3D*, 8> primary_after{
        &theta, &qv, &qc, &qr, &qi, &qs, &qg, &qh};
    const std::array<const Field3D*, 8> tendency_after{
        &dtheta, &dqv, &dqc, &dqr, &dqi, &dqs, &dqg, &dqh};
    for (std::size_t n = 0; n < primary_before.size(); ++n)
    {
        require_same_field(primary_before[n], *primary_after[n]);
        require_same_field(tendency_before[n], *tendency_after[n]);
    }
}

TEST_CASE("Native Lin direct fused refusal preserves caller outputs without split retry",
          "[physics][microphysics][lin][qualified][atomicity]")
{
    const bool throws = GENERATE(false, true);
    const int mode = GENERATE(0, 1, 2, 3);
    CAPTURE(mode, throws);
    setup_grid();
    setup_warm_state();
    ComputeRuntimeReset reset;
    auto backend = std::make_unique<PartialLinBackend>(true, throws);
    auto* observed = backend.get();
    global_compute_backend_runtime_state.backend = std::move(backend);
    global_compute_backend_runtime_state.active_kind = ComputeBackendKind::Metal;
    global_compute_backend_runtime_state.requested_kind = ComputeBackendKind::Metal;
    global_compute_backend_runtime_state.config.validate_parity = mode == 1;
    auto scheme = create_microphysics_scheme("lin");
    scheme->bind_execution_owners(global_compute_backend_runtime_state, nullptr);
    scheme->set_vertical_spacing_m(dz);
    scheme->set_require_native_dispatch(true);
    Tendencies rates;
    rates.resize(NR, NTH, NZ);
    rates.dtheta.fill(9);
    if (mode == 2) rates.dtheta.resize(NR + 1, NTH, NZ, 9);
    Field3D& target = mode == 3 ? theta : rates.dtheta;
    const auto before = target;
    const auto original = target.data();
    const char* expected_error = throws ? "injected fused Lin failure"
        : mode == 0 ? "native Lin fused transaction failed before output commit"
                    : "qualified accelerator Lin requires the native sedimentation operator";
    REQUIRE_THROWS_WITH(scheme->compute_tendencies(p, theta, rho, qv, qc, qr, qi, qs, qg, qh, .1,
        target, rates.dqv, rates.dqc, rates.dqr, rates.dqi, rates.dqs, rates.dqg, rates.dqh), expected_error);
    CHECK((observed->fused_destination == original) == (mode == 0));
    CHECK(observed->pointwise_calls == ((mode == 0 || throws) ? 0 : 1));
    require_same_field(before, target);
}

TEST_CASE("Kessler autoconversion: verify dqr/dt = c_auto*(qc - qc0)", "[physics][microphysics][analytical]")
{
    setup_grid();
    setup_warm_state();

    // Isolate autoconversion: set qc above threshold, no rain (no accretion),
    // saturated air (no evaporation)
    const float qc_val = 0.003f;
    const float qvsat = static_cast<float>(
        thermodynamics::saturation_mixing_ratio_water(kTestT, kTestP));
    qv.resize(NR, NTH, NZ, qvsat);  // saturated at test conditions
    qc.resize(NR, NTH, NZ, qc_val);
    qr.resize(NR, NTH, NZ, 0.0f);

    auto scheme = create_microphysics_scheme("kessler");
    Tendencies t;
    compute(*scheme, dt, t);

    // Expected autoconversion rate: c_auto * (qc - qc0)
    double expected_dqr = kCauto * (qc_val - kQc0);

    // Check a representative interior point
    float actual_dqr = t.dqr(4, 4, 4);
    REQUIRE(actual_dqr > 0.0f);

    // The rain tendency should include autoconversion contribution.
    // With qr=0 and saturated air, autoconversion is the dominant source.
    // Allow for additional small contributions from other processes.
    REQUIRE(static_cast<double>(actual_dqr) == Approx(expected_dqr).margin(1.0e-6));
}

TEST_CASE("Kessler production matches the reference finite-step law",
          "[physics][microphysics][analytical][reference]")
{
    setup_grid();
    setup_warm_state();

    // Cloud and rain are both present and qc exceeds the threshold, so this
    // exercises implicit accretion and autoconversion together. A deliberately
    // long process step distinguishes the finite-step equation from an
    // explicit Euler source while uniform rain cancels sedimentation here.
    const double step_dt = 37.0;
    const float qc_val = 0.0025f;
    const float qr_val = 0.001f;
    const float qvsat = static_cast<float>(
        thermodynamics::saturation_mixing_ratio_water(kTestT, kTestP));
    qv.resize(NR, NTH, NZ, qvsat);  // saturated at test conditions
    qc.resize(NR, NTH, NZ, qc_val);
    qr.resize(NR, NTH, NZ, qr_val);

    auto scheme = create_microphysics_scheme("kessler");
    Tendencies t;
    compute(*scheme, step_dt, t);

    // WRF/COMMAS Kessler production:
    // [qc*(1-factor) + factor*c1*dt*max(qc-qc0,0)] / dt.
    const double factor = 1.0 /
        (1.0 + kCaccr * step_dt
         * std::pow(qr_val, kCaccrExponent));
    const double expected_production =
        (static_cast<double>(qc_val) * (1.0 - factor)
         + factor * kCauto * step_dt
           * (static_cast<double>(qc_val) - kQc0)) / step_dt;

    float actual_dqr = t.dqr(4, 4, 4);
    // Rain gains from accretion; sedimentation may also contribute
    // but accretion should be the dominant pointwise tendency
    REQUIRE(actual_dqr > 0.0f);

    // Cloud water should decrease by accretion
    float actual_dqc = t.dqc(4, 4, 4);
    REQUIRE(actual_dqc < 0.0f);
    REQUIRE(static_cast<double>(actual_dqc)
            == Approx(-expected_production).margin(2.0e-10));
    REQUIRE(-static_cast<double>(actual_dqc) * step_dt
            <= static_cast<double>(qc_val));

    // Accretion transfers mass: |dqc/dt| should match dqr/dt from accretion
    // (within tolerance for sedimentation contribution)
    REQUIRE(std::abs(actual_dqc + actual_dqr) < 0.01 * std::abs(actual_dqr) + 1.0e-8);
}

TEST_CASE("Kessler evaporation matches the reference ventilation kernel",
          "[physics][microphysics][analytical][reference][evaporation]")
{
    setup_grid();
    setup_warm_state();

    const float qvsat = static_cast<float>(
        thermodynamics::saturation_mixing_ratio_water(kTestT, kTestP));
    const float qv_val = 0.9f * qvsat;
    const float qr_val = 1.0e-3f;
    qv.resize(NR, NTH, NZ, qv_val);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, qr_val);

    const double rcgs = 1.0e-3 * 1.2;
    const double rain_content = rcgs * static_cast<double>(qr_val);
    const double ventilation =
        (1.6 + 124.9 * std::pow(rain_content, 0.2046))
        * std::pow(rain_content, 0.525);
    const double resistance =
        2.55e8 / (kTestP * static_cast<double>(qvsat)) + 5.4e5;
    const double expected_rate =
        (ventilation / resistance)
        * ((static_cast<double>(qvsat) - static_cast<double>(qv_val))
           / (rcgs * static_cast<double>(qvsat)));

    const auto run = [&](double efficiency)
    {
        tmv::MicrophysicsConfig config;
        config.kessler.evaporation_efficiency = efficiency;
        auto scheme = create_microphysics_scheme("kessler", config);
        Tendencies tendencies;
        compute(*scheme, dt, tendencies);
        return tendencies;
    };

    const Tendencies disabled = run(0.0);
    REQUIRE(disabled.dqv(4, 4, 4) == 0.0f);
    REQUIRE(disabled.dtheta(4, 4, 4) == 0.0f);

    const Tendencies reference = run(1.0);
    REQUIRE(static_cast<double>(reference.dqv(4, 4, 4))
            == Approx(expected_rate).epsilon(3.0e-5));
    REQUIRE(static_cast<double>(reference.dqr(4, 4, 4))
            == Approx(-expected_rate).epsilon(3.0e-5));
    REQUIRE(static_cast<double>(reference.dtheta(4, 4, 4))
            == Approx(
                -(microphysics_constants::L_v
                  / microphysics_constants::cp) * expected_rate)
                   .epsilon(3.0e-5));

    const Tendencies half = run(0.5);
    REQUIRE(half.dqv(4, 4, 4)
            == Approx(0.5f * reference.dqv(4, 4, 4)).epsilon(3.0e-5));

    // The kernel must consume live dry density, not a fixed initialization
    // profile or fallback value.
    constexpr double low_density = 0.8;
    rho.resize(NR, NTH, NZ, static_cast<float>(low_density));
    const double low_rcgs = 1.0e-3 * low_density;
    const double low_rain_content = low_rcgs * static_cast<double>(qr_val);
    const double low_ventilation =
        (1.6 + 124.9 * std::pow(low_rain_content, 0.2046))
        * std::pow(low_rain_content, 0.525);
    const double expected_low_density_rate =
        (low_ventilation / resistance)
        * ((static_cast<double>(qvsat) - static_cast<double>(qv_val))
           / (low_rcgs * static_cast<double>(qvsat)));
    const Tendencies live_density = run(1.0);
    REQUIRE(static_cast<double>(live_density.dqv(4, 4, 4))
            == Approx(expected_low_density_rate).epsilon(3.0e-5));
    REQUIRE(live_density.dqv(4, 4, 4) != reference.dqv(4, 4, 4));
}

TEST_CASE("Kessler declares runtime ownership for combined evaporation and sedimentation donor bounds",
          "[physics][microphysics][kessler][positivity][sedimentation]")
{
    setup_grid();
    setup_warm_state();

    qv.resize(NR, NTH, NZ, 0.0f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    constexpr int donor_k = 4;
    constexpr float rain = 1.0e-6f;
    constexpr double step_dt = 1.5;
    qr(4, 4, donor_k) = rain;

    auto scheme = create_microphysics_scheme("kessler");
    REQUIRE(scheme->requires_runtime_column_donor_admission());
    Tendencies tendencies;
    compute(*scheme, step_dt, tendencies);

    REQUIRE(tendencies.dqv(4, 4, donor_k) > 0.0f);
    REQUIRE(tendencies.dqr(4, 4, donor_k - 1) > 0.0f);
    const double committed_rain =
        static_cast<double>(rain)
        + step_dt * static_cast<double>(tendencies.dqr(4, 4, donor_k));
    REQUIRE(committed_rain >= 0.0);
    REQUIRE(committed_rain <= static_cast<double>(rain));
}

TEST_CASE("Kessler total water mass conservation (warm rain)", "[physics][microphysics][conservation]")
{
    setup_grid();
    setup_warm_state();

    qv.resize(NR, NTH, NZ, 0.012f);
    qc.resize(NR, NTH, NZ, 0.002f);
    qr.resize(NR, NTH, NZ, 0.001f);

    auto scheme = create_microphysics_scheme("kessler");
    Tendencies t;
    compute(*scheme, dt, t);

    // Conservation currency is water mass, not an unweighted sum of mixing
    // ratios. Internal sedimentation telescopes after weighting by dry mass;
    // the only external sink is the diagnosed bottom rain flux.
    double water_rate = 0.0;
    double surface_flux = 0.0;
    double scale = 0.0;
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                const double dry_mass_per_area =
                    static_cast<double>(rho(i, j, k)) * dz;
                const double dqv_rate = t.dqv(i, j, k);
                const double dqc_rate = t.dqc(i, j, k);
                const double dqr_rate = t.dqr(i, j, k);
                water_rate += dry_mass_per_area
                    * (dqv_rate + dqc_rate + dqr_rate);
                scale += dry_mass_per_area
                    * (std::abs(dqv_rate) + std::abs(dqc_rate)
                       + std::abs(dqr_rate));
            }
            const double sink = scheme->bottom_sedimentation_sink_rate(
                rho, qr, qs, qg, qh, i, j, dt);
            const double bottom_dry_mass_per_area =
                static_cast<double>(rho(i, j, 0)) * dz;
            surface_flux += bottom_dry_mass_per_area * sink;
            scale += bottom_dry_mass_per_area * std::abs(sink);
        }

    REQUIRE(std::abs(water_rate + surface_flux)
            / std::max(scale, 1.0e-30) < 2.0e-6);
}

TEST_CASE("Kessler dry state produces exactly zero tendencies", "[physics][microphysics][analytical]")
{
    setup_grid();
    setup_warm_state();

    qv.fill(0.0f);
    qc.fill(0.0f);
    qr.fill(0.0f);

    auto scheme = create_microphysics_scheme("kessler");
    Tendencies t;
    compute(*scheme, dt, t);

    // With all hydrometeors zero, every tendency must be exactly zero
    for (size_t n = 0; n < t.dqr.size(); ++n)
    {
        REQUIRE(t.dqv.data()[n] == 0.0f);
        REQUIRE(t.dqc.data()[n] == 0.0f);
        REQUIRE(t.dqr.data()[n] == 0.0f);
    }
}

TEST_CASE("Kessler CPU tendencies are bit-exact across thread counts",
          "[physics][microphysics][CPU-019]")
{
#ifdef _OPENMP
    setup_grid();
    setup_warm_state();
    qv.resize(NR, NTH, NZ, 0.0f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const float pattern = static_cast<float>((13 * i + 7 * j + 3 * k) % 17);
                p[i][j][k] = 78000.0f + 125.0f * pattern;
                theta[i][j][k] = 258.0f + 3.1f * static_cast<float>(k)
                               + 0.07f * pattern;
                rho[i][j][k] = 0.8f + 0.015f * pattern;
                qv[i][j][k] = 0.001f + 0.0004f * pattern;
                qc[i][j][k] = 0.0002f * static_cast<float>((i + j + k) % 9);
                qr[i][j][k] = 0.0001f * static_cast<float>((2 * i + j + k) % 7);
                qg[i][j][k] = 0.00008f * static_cast<float>((i + 2 * j + k) % 6);
                qh[i][j][k] = 0.00005f * static_cast<float>((i + j + 2 * k) % 5);
            }

    const int saved_dynamic = omp_get_dynamic();
    const int saved_threads = omp_get_max_threads();
    omp_set_dynamic(0);

    auto scheme = create_microphysics_scheme("kessler");
    Tendencies single_thread;
    Tendencies eight_threads;
    omp_set_num_threads(1);
    compute(*scheme, dt, single_thread);
    omp_set_num_threads(8);
    compute(*scheme, dt, eight_threads);

    omp_set_num_threads(saved_threads);
    omp_set_dynamic(saved_dynamic);

    const Field3D* single_fields[] = {
        &single_thread.dtheta, &single_thread.dqv, &single_thread.dqc,
        &single_thread.dqr, &single_thread.dqi, &single_thread.dqs,
        &single_thread.dqg, &single_thread.dqh};
    const Field3D* parallel_fields[] = {
        &eight_threads.dtheta, &eight_threads.dqv, &eight_threads.dqc,
        &eight_threads.dqr, &eight_threads.dqi, &eight_threads.dqs,
        &eight_threads.dqg, &eight_threads.dqh};
    for (std::size_t field = 0; field < std::size(single_fields); ++field)
        for (std::size_t n = 0; n < single_fields[field]->size(); ++n)
            REQUIRE(single_fields[field]->data()[n] ==
                    parallel_fields[field]->data()[n]);
#else
    SUCCEED("OpenMP is unavailable in this build");
#endif
}

TEST_CASE("Milbrandt mixed-phase tendencies are bit-exact across thread counts",
          "[physics][microphysics][milbrandt][thread][determinism]")
{
#ifdef _OPENMP
    setup_grid();
    setup_warm_state();
    qv.resize(NR, NTH, NZ, 0.0f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const float pattern = static_cast<float>(
                    (13 * i + 7 * j + 3 * k) % 17);
                p[i][j][k] = 76000.0f + 160.0f * pattern;
                theta[i][j][k] =
                    254.0f + 2.3f * static_cast<float>(k)
                    + 0.05f * pattern;
                rho[i][j][k] = 0.75f + 0.018f * pattern;
                qv[i][j][k] = 0.0015f + 0.00035f * pattern;
                qc[i][j][k] = 0.00012f
                    * static_cast<float>((i + j + k) % 9);
                qr[i][j][k] = 0.00009f
                    * static_cast<float>((2 * i + j + k) % 7);
                qi[i][j][k] = 0.00004f
                    * static_cast<float>((i + 3 * j + k) % 6);
                qs[i][j][k] = 0.00007f
                    * static_cast<float>((i + j + 2 * k) % 8);
                qg[i][j][k] = 0.00006f
                    * static_cast<float>((i + 2 * j + k) % 6);
                qh[i][j][k] = 0.00003f
                    * static_cast<float>((i + j + 2 * k) % 5);
            }

    const int saved_dynamic = omp_get_dynamic();
    const int saved_threads = omp_get_max_threads();
    omp_set_dynamic(0);

    auto scheme = create_microphysics_scheme("milbrandt");
    Tendencies single_thread;
    Tendencies eight_threads;
    omp_set_num_threads(1);
    compute(*scheme, dt, single_thread);
    const PortableSchemeState single_private =
        scheme->portable_state();
    omp_set_num_threads(8);
    compute(*scheme, dt, eight_threads);
    const PortableSchemeState parallel_private =
        scheme->portable_state();

    omp_set_num_threads(saved_threads);
    omp_set_dynamic(saved_dynamic);

    const Field3D* single_fields[] = {
        &single_thread.dtheta, &single_thread.dqv,
        &single_thread.dqc, &single_thread.dqr,
        &single_thread.dqi, &single_thread.dqs,
        &single_thread.dqg, &single_thread.dqh};
    const Field3D* parallel_fields[] = {
        &eight_threads.dtheta, &eight_threads.dqv,
        &eight_threads.dqc, &eight_threads.dqr,
        &eight_threads.dqi, &eight_threads.dqs,
        &eight_threads.dqg, &eight_threads.dqh};
    for (std::size_t field = 0;
         field < std::size(single_fields); ++field)
        for (std::size_t n = 0;
             n < single_fields[field]->size(); ++n)
            REQUIRE(single_fields[field]->data()[n] ==
                    parallel_fields[field]->data()[n]);

    REQUIRE(single_private.fields.size()
            == parallel_private.fields.size());
    for (std::size_t field = 0;
         field < single_private.fields.size(); ++field)
        for (std::size_t n = 0;
             n < single_private.fields[field].size(); ++n)
            REQUIRE(single_private.fields[field].data()[n] ==
                    parallel_private.fields[field].data()[n]);
#else
    SUCCEED("OpenMP is unavailable in this build");
#endif
}

TEST_CASE("Kessler sub-threshold cloud water produces no autoconversion", "[physics][microphysics][analytical]")
{
    setup_grid();
    setup_warm_state();

    // qc below threshold, no rain, saturated -- nothing should happen
    const float qvsat = static_cast<float>(
        thermodynamics::saturation_mixing_ratio_water(kTestT, kTestP));
    qv.resize(NR, NTH, NZ, qvsat);
    qc.resize(NR, NTH, NZ, 0.0005f);  // < qc0 = 1e-3
    qr.resize(NR, NTH, NZ, 0.0f);

    auto scheme = create_microphysics_scheme("kessler");
    Tendencies t;
    compute(*scheme, dt, t);

    // No autoconversion, no accretion (qr=0), no evaporation (saturated)
    // All moisture tendencies should be zero
    for (size_t n = 0; n < t.dqr.size(); ++n)
    {
        REQUIRE(std::abs(t.dqr.data()[n]) < 1.0e-10f);
    }
}

TEST_CASE("Kessler produces finite tendencies for all schemes", "[physics][microphysics]")
{
    setup_grid();
    setup_warm_state();
    qv.resize(NR, NTH, NZ, 0.012f);
    qc.resize(NR, NTH, NZ, 0.001f);
    qr.resize(NR, NTH, NZ, 0.0005f);

    for (const char* name : {"kessler", "lin", "thompson"})
    {
        auto scheme = create_microphysics_scheme(name);
        Tendencies t;
        compute(*scheme, dt, t);

        for (size_t n = 0; n < t.dtheta.size(); ++n)
        {
            REQUIRE(std::isfinite(t.dtheta.data()[n]));
            REQUIRE(std::isfinite(t.dqv.data()[n]));
            REQUIRE(std::isfinite(t.dqc.data()[n]));
            REQUIRE(std::isfinite(t.dqr.data()[n]));
        }
    }
}

TEST_CASE("Latent heating magnitude: saturation adjustment closes mass and energy",
          "[physics][microphysics][analytical][heating]")
{
    // SCIENCE.md C3 + M13 regression. Before the fix every heating
    // term carried a spurious /T (condensing 1 g/kg warmed ~0.009 K
    // instead of ~2.5 K - the storm's CAPE engine disconnected), and the
    // saturation adjustment condensed the raw excess instead of
    // excess/(1 + gamma). This pins the full chain quantitatively for
    // both production schemes. At p = p0 the Exner factor is exactly 1,
    // so dtheta == dT and the expectation is clean.
    setup_grid();
    setup_warm_state();

    const double qvs =
        thermodynamics::saturation_mixing_ratio_water(kTestT, kTestP);
    const double excess = 1.0e-3;  // 1 g/kg supersaturation
    qv.resize(NR, NTH, NZ, static_cast<float>(qvs + excess));
    qc.resize(NR, NTH, NZ, 0.0f);  // post-adjustment qc stays below the
    qr.resize(NR, NTH, NZ, 0.0f);  // autoconversion threshold, so ONLY
                                   // the saturation adjustment fires

    // Qualified explicitly: simulation.hpp's global-scope physics
    // constants collide with microphysics_constants in shared TUs.
    const double L_v_c = microphysics_constants::L_v;
    const double cp_c = microphysics_constants::cp;
    const double R_v_c = microphysics_constants::R_v;
    const double gamma_sat =
        (L_v_c * L_v_c * qvs) / (cp_c * R_v_c * kTestT * kTestT);
    const double dq_cond = excess / (1.0 + gamma_sat);

    // Literal nonlinear liquid-saturation solve used as the independent
    // expectation for schemes on the shared coupled adjuster. Lin retains
    // its separately mirrored first-order CPU/device equation.
    const auto literal_qvs =
        [](double temperature, double pressure)
        {
            const double temperature_c =
                temperature - microphysics_constants::T0;
            const double vapor_pressure =
                611.21 * std::exp(
                    (18.678 - temperature_c / 234.5)
                    * temperature_c / (257.14 + temperature_c));
            return std::clamp(
                0.62198 * vapor_pressure
                    / std::max(pressure - vapor_pressure, 1.0),
                0.0, 0.2);
        };
    const double initial_vapor =
        static_cast<double>(qv(4, 4, 4));
    double lower = 0.0;
    double upper = initial_vapor;
    for (int iteration = 0; iteration < 52; ++iteration)
    {
        const double midpoint = 0.5 * (lower + upper);
        const double adjusted_temperature =
            kTestT + (L_v_c / cp_c) * midpoint;
        const double residual =
            initial_vapor - midpoint
            - literal_qvs(adjusted_temperature, kTestP);
        if (residual > 0.0)
            lower = midpoint;
        else
            upper = midpoint;
    }
    const double nonlinear_condensation =
        0.5 * (lower + upper);

    for (const char* name : {"lin", "kessler", "milbrandt", "thompson"})
    {
        INFO("scheme = " << name);
        auto scheme = create_microphysics_scheme(name);
        Tendencies t;
        compute(*scheme, dt, t);

        const double actual_heating = t.dtheta(4, 4, 4);
        const double actual_dqc = t.dqc(4, 4, 4);
        const bool linearized = std::string(name) == "lin";
        const double expected_condensation =
            linearized ? dq_cond : nonlinear_condensation;
        const double expected_dtheta_dt =
            (L_v_c / cp_c) * expected_condensation / dt;
        // 2% covers the fp32 theta->T->qvs chain differences between the
        // schemes; the C3 bug was a factor-of-300 miss.
        REQUIRE(actual_heating == Approx(expected_dtheta_dt).epsilon(0.02));
        REQUIRE(actual_dqc
                == Approx(expected_condensation / dt).epsilon(0.02));
        // Conservation: vapor loss mirrors cloud gain.
        REQUIRE(static_cast<double>(t.dqv(4, 4, 4))
                == Approx(-actual_dqc).epsilon(1e-6));
    }
}

TEST_CASE("Sedimentation conserves column rain mass in stratified air",
          "[physics][microphysics][conservation][sedimentation]")
{
    // SCIENCE.md M11 + SCI-024: the falling MASS rho_d*q*Vt is the
    // conserved quantity. With a horizontally perturbed live rho_d field,
    // the column integral sum_k rho_d[k] * dq_dt[k] * dz must exactly balance
    // the live-density surface sink; the old
    // unweighted update multiplied water mass by the density ratio at
    // every level of fall (~7x integrated from a 16 km anvil).
    setup_grid();
    setup_warm_state();

    // Stratified, horizontally perturbed live state. Keep rho0 deliberately
    // unrelated so this regression fails if sedimentation falls back to it.
    for (int k = 0; k < NZ; ++k)
    {
        rho0_base[static_cast<size_t>(k)] = 0.75;
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                rho(i, j, k) = static_cast<float>(
                    1.2 * std::exp(-(k * dz) / 8000.0)
                    * (1.0 + 0.03 * std::sin(0.4 * i) * std::cos(0.3 * j)));
    }

    // Saturated, cloud-free, warm: condensation, evaporation, and every
    // ice process are inactive, so rain sedimentation is the only
    // process that touches dqr_dt.
    const double qvs =
        thermodynamics::saturation_mixing_ratio_water(kTestT, kTestP);
    qv.resize(NR, NTH, NZ, static_cast<float>(qvs));
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 1.0e-3f);

    for (const char* name : {"lin", "kessler", "thompson", "milbrandt"})
    {
        INFO("scheme = " << name);
        std::fill(rho0_base.begin(), rho0_base.end(), 0.75);
        auto scheme = create_microphysics_scheme(name);
        Tendencies t;
        compute(*scheme, dt, t);

        for (const auto& column : {std::pair<int, int>{2, 3}, {5, 1}})
        {
            const int i = column.first;
            const int j = column.second;
            double column_mass_rate = 0.0;
            for (int k = 0; k < NZ; ++k)
            {
                column_mass_rate +=
                    static_cast<double>(rho(i, j, k))
                    * static_cast<double>(t.dqr(i, j, k)) * dz;
            }
            const double sink_rate = scheme->bottom_sedimentation_sink_rate(
                rho, qr, qs, qg, qh, i, j, dt);
            const double surface_flux = static_cast<double>(rho(i, j, 0))
                                      * sink_rate * dz;

            REQUIRE(sink_rate > 0.0);
            // fp32 tendency storage rounds each cell at ~1e-7 relative.
            REQUIRE(column_mass_rate
                    == Approx(-surface_flux).epsilon(1.0e-5));
        }

        // Changing only rho0 must not change any sedimentation tendency.
        for (int k = 0; k < NZ; ++k)
            rho0_base[static_cast<std::size_t>(k)] = 1.8 - 0.1 * k;
        auto rebased_scheme = create_microphysics_scheme(name);
        Tendencies rebased;
        compute(*rebased_scheme, dt, rebased);
        for (int k = 0; k < NZ; ++k)
            REQUIRE(rebased.dqr(2, 3, k) == t.dqr(2, 3, k));
    }
}

TEST_CASE("Sedimentation conserves mass across nonuniform terrain layers",
          "[physics][microphysics][conservation][sedimentation][terrain][TE-005]")
{
    setup_grid();
    setup_warm_state();
    const double qvs =
        thermodynamics::saturation_mixing_ratio_water(kTestT, kTestP);
    qv.resize(NR, NTH, NZ, static_cast<float>(qvs));
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 1.0e-3f);

    std::vector<double> layer_thickness(
        static_cast<std::size_t>(NR) * NTH * NZ);
    auto thickness = [&](int i, int j, int k) -> double&
    {
        return layer_thickness[
            (static_cast<std::size_t>(i) * NTH + j) * NZ + k];
    };
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                thickness(i, j, k) = 250.0 + 17.0 * i + 9.0 * j
                    + 31.0 * k;
                // One checked column is deliberately below the former
                // 1e-6 m substitution threshold. Positive geometry is an
                // exact caller contract; changing unequal layers to a common
                // floor breaks the rho*q*dz sedimentation inventory.
                if (i == 5 && j == 1)
                    thickness(i, j, k) *= 1.0e-9;
                rho(i, j, k) = static_cast<float>(
                    1.2 * std::exp(-0.08 * k) * (1.0 + 0.01 * i));
            }

    for (const char* name : {"lin", "kessler", "thompson", "milbrandt"})
    {
        INFO("scheme = " << name);
        auto scheme = create_microphysics_scheme(name);
        scheme->set_vertical_spacing_m(
            NR, NTH, NZ, layer_thickness);
        Tendencies t;
        t.resize(NR, NTH, NZ);
        double step = dt;
        if (std::string(name) == "lin" || std::string(name) == "kessler" || std::string(name) == "thompson")
        {
            t.dqr.fill(17.0f);
            REQUIRE_THROWS_WITH(scheme->compute_tendencies(p,
                                                           theta,
                                                           rho,
                                                           qv,
                                                           qc,
                                                           qr,
                                                           qi,
                                                           qs,
                                                           qg,
                                                           qh,
                                                           dt,
                                                           t.dtheta,
                                                           t.dqv,
                                                           t.dqc,
                                                           t.dqr,
                                                           t.dqi,
                                                           t.dqs,
                                                           t.dqg,
                                                           t.dqh),
                                Catch::Matchers::Contains(std::string(name) == "thompson"
                                                              ? "Thompson sedimentation failed"
                                                              : "terminal-velocity CFL"));
            REQUIRE(t.dqr[5][1][0] == 17.0f);
            step = 1.0e-8;
        }
        scheme->compute_tendencies(
            p, theta, rho, qv, qc, qr, qi, qs, qg, qh, step,
            t.dtheta, t.dqv, t.dqc, t.dqr,
            t.dqi, t.dqs, t.dqg, t.dqh);

        for (const auto& column : {std::pair<int, int>{2, 3}, {5, 1}})
        {
            const int i = column.first;
            const int j = column.second;
            CAPTURE(i, j, thickness(i, j, 0));
            double column_mass_rate_per_area = 0.0;
            for (int k = 0; k < NZ; ++k)
            {
                const double total_tendency =
                    static_cast<double>(t.dqv(i, j, k))
                    + static_cast<double>(t.dqc(i, j, k))
                    + static_cast<double>(t.dqr(i, j, k))
                    + static_cast<double>(t.dqi(i, j, k))
                    + static_cast<double>(t.dqs(i, j, k))
                    + static_cast<double>(t.dqg(i, j, k))
                    + static_cast<double>(t.dqh(i, j, k));
                column_mass_rate_per_area +=
                    static_cast<double>(rho(i, j, k)) * total_tendency
                    * thickness(i, j, k);
            }
            const double sink_rate = scheme->bottom_sedimentation_sink_rate(
                rho, qr, qs, qg, qh, i, j, step);
            const double surface_flux = static_cast<double>(rho(i, j, 0))
                * sink_rate * thickness(i, j, 0);
            REQUIRE(sink_rate > 0.0);
            REQUIRE(column_mass_rate_per_area
                    == Approx(-surface_flux).epsilon(3.0e-5));
        }
    }
}

TEST_CASE("Milbrandt surface fallout includes every precipitating species",
          "[physics][microphysics][conservation][sedimentation][milbrandt]")
{
    setup_grid();
    setup_warm_state();

    const double qvs =
        thermodynamics::saturation_mixing_ratio_water(kTestT, kTestP);
    qv.resize(NR, NTH, NZ, static_cast<float>(qvs));
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);

    const int i = 3;
    const int j = 4;
    qr(i, j, 0) = 1.0e-3f;
    qs(i, j, 0) = 8.0e-4f;
    qg(i, j, 0) = 6.0e-4f;
    qh(i, j, 0) = 4.0e-4f;

    auto scheme = create_microphysics_scheme("milbrandt");
    Tendencies t;
    compute(*scheme, dt, t);
    auto* const milbrandt = dynamic_cast<MilbrandtScheme*>(scheme.get());
    REQUIRE(milbrandt != nullptr);
    const std::size_t worker_count =
        milbrandt->retained_sedimentation_worker_count();
    REQUIRE(worker_count >= 1);
    std::vector<std::array<const double*, 6>> first_worker_buffers(
        worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker)
    {
        const auto& workspace =
            milbrandt->retained_sedimentation_workspace(worker);
        first_worker_buffers[worker] = {
            workspace.q_work.data(), workspace.n_work.data(),
            workspace.q_next.data(), workspace.n_next.data(),
            workspace.velocity_mass.data(),
            workspace.velocity_number.data()};
        for (const double* buffer : first_worker_buffers[worker])
            REQUIRE(buffer != nullptr);
    }

    const auto copied_scheme = scheme->clone();
    const auto* const copied_milbrandt =
        dynamic_cast<const MilbrandtScheme*>(copied_scheme.get());
    REQUIRE(copied_milbrandt != nullptr);
    REQUIRE(copied_milbrandt->retained_sedimentation_worker_count() == 0);

    const double sink_rate = scheme->bottom_sedimentation_sink_rate(
        rho, qr, qs, qg, qh, i, j, dt);
    REQUIRE(sink_rate > 0.0);

    double column_water_mass_rate = 0.0;
    for (int k = 0; k < NZ; ++k)
    {
        const double total_water_tendency =
            static_cast<double>(t.dqv(i, j, k))
          + static_cast<double>(t.dqc(i, j, k))
          + static_cast<double>(t.dqr(i, j, k))
          + static_cast<double>(t.dqi(i, j, k))
          + static_cast<double>(t.dqs(i, j, k))
          + static_cast<double>(t.dqg(i, j, k))
          + static_cast<double>(t.dqh(i, j, k));
        column_water_mass_rate += static_cast<double>(rho(i, j, k))
                                * total_water_tendency * dz;
    }
    const double surface_flux = static_cast<double>(rho(i, j, 0))
                              * sink_rate * dz;
    REQUIRE(column_water_mass_rate
            == Approx(-surface_flux).epsilon(2.0e-5));

    // A later dry call must clear the cached rate instead of reporting the
    // previous step's precipitation to the runtime conservation gate.
    qr.fill(0.0f);
    qs.fill(0.0f);
    qg.fill(0.0f);
    qh.fill(0.0f);
    Tendencies dry;
    compute(*scheme, dt, dry);
    REQUIRE(milbrandt->retained_sedimentation_worker_count()
            == worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker)
    {
        const auto& workspace =
            milbrandt->retained_sedimentation_workspace(worker);
        const std::array<const double*, 6> reused_buffers{
            workspace.q_work.data(), workspace.n_work.data(),
            workspace.q_next.data(), workspace.n_next.data(),
            workspace.velocity_mass.data(),
            workspace.velocity_number.data()};
        REQUIRE(reused_buffers == first_worker_buffers[worker]);
    }
    REQUIRE(scheme->bottom_sedimentation_sink_rate(
                rho, qr, qs, qg, qh, i, j, dt) == 0.0);
}

TEST_CASE("Milbrandt portable state preserves private moment history",
          "[physics][microphysics][runtime][restart][mod03]")
{
    setup_grid();
    setup_warm_state();
    qv.resize(NR, NTH, NZ, 0.012f);
    qc.resize(NR, NTH, NZ, 0.0015f);
    qr.resize(NR, NTH, NZ, 0.0008f);
    qi.resize(NR, NTH, NZ, 0.0002f);
    qs.resize(NR, NTH, NZ, 0.0003f);
    qg.resize(NR, NTH, NZ, 0.0001f);
    qh.resize(NR, NTH, NZ, 0.00005f);

#ifdef _OPENMP
    const int saved_dynamic = omp_get_dynamic();
    const int saved_threads = omp_get_max_threads();
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
    auto uninterrupted = create_microphysics_scheme("milbrandt");
    Tendencies spinup;
    compute(*uninterrupted, dt, spinup);
    const PortableSchemeState checkpoint = uninterrupted->portable_state();
    const PortableSchemeState parameters =
        uninterrupted->portable_parameter_state();
    REQUIRE(parameters.fields.empty());
    REQUIRE(parameters.version == checkpoint.version);
    REQUIRE(parameters.integers == checkpoint.integers);
    REQUIRE(parameters.reals == checkpoint.reals);
    auto restarted = create_microphysics_scheme("milbrandt");
    REQUIRE(restarted != nullptr);
    std::string restore_error;
    REQUIRE(restarted->restore_portable_state(checkpoint, restore_error));
    REQUIRE(restore_error.empty());

    Tendencies expected;
    Tendencies actual;
    compute(*uninterrupted, dt, expected);
#ifdef _OPENMP
    omp_set_num_threads(8);
#endif
    compute(*restarted, dt, actual);
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
    omp_set_dynamic(saved_dynamic);
#endif

    const Field3D* expected_fields[] = {
        &expected.dtheta, &expected.dqv, &expected.dqc, &expected.dqr,
        &expected.dqi, &expected.dqs, &expected.dqg, &expected.dqh};
    const Field3D* actual_fields[] = {
        &actual.dtheta, &actual.dqv, &actual.dqc, &actual.dqr,
        &actual.dqi, &actual.dqs, &actual.dqg, &actual.dqh};
    for (std::size_t field = 0; field < std::size(expected_fields); ++field)
    {
        REQUIRE(expected_fields[field]->size() == actual_fields[field]->size());
        for (std::size_t idx = 0; idx < expected_fields[field]->size(); ++idx)
        {
            REQUIRE(expected_fields[field]->data()[idx] ==
                    actual_fields[field]->data()[idx]);
        }
    }
}

TEST_CASE("Milbrandt exposes all six number moments as transported dry-air mixing ratios",
          "[physics][microphysics][milbrandt][transport][moments]")
{
    setup_grid();
    setup_warm_state();
    qv.resize(NR, NTH, NZ, 0.010f);
    qc.resize(NR, NTH, NZ, 1.0e-3f);
    qr.resize(NR, NTH, NZ, 8.0e-4f);

    auto scheme = create_microphysics_scheme("milbrandt");
    scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
    const auto scalars = scheme->prognostic_scalars_for_transport();
    REQUIRE(scalars.size() == 6);
    const std::array<const char*, 6> expected_names{{
        "milbrandt.Nc_kg-1", "milbrandt.Nr_kg-1",
        "milbrandt.Ni_kg-1", "milbrandt.Ns_kg-1",
        "milbrandt.Ng_kg-1", "milbrandt.Nh_kg-1"}};
    const std::array<tmv::MicrophysicsMassSpecies, 6> expected_mass{{
        tmv::MicrophysicsMassSpecies::Cloud, tmv::MicrophysicsMassSpecies::Rain,
        tmv::MicrophysicsMassSpecies::Ice, tmv::MicrophysicsMassSpecies::Snow,
        tmv::MicrophysicsMassSpecies::Graupel, tmv::MicrophysicsMassSpecies::Hail}};
    for (std::size_t n = 0; n < scalars.size(); ++n)
    {
        REQUIRE(std::string(scalars[n].name) == expected_names[n]);
        REQUIRE(scalars[n].mass_species == expected_mass[n]);
        REQUIRE(scalars[n].field != nullptr);
        REQUIRE(scalars[n].field->size_r() == NR);
        REQUIRE(scalars[n].field->size_th() == NTH);
        REQUIRE(scalars[n].field->size_z() == NZ);
        REQUIRE(scalars[n].positive_definite);
        REQUIRE(scalars[n].storage
                == tmv::MicrophysicsScalarStorage::DryAirMixingRatio);
    }

    constexpr int i = 4, j = 4, k = 4;
    REQUIRE((*scalars[0].field)(i, j, k)
            == Approx(2.0e8 / rho(i, j, k)).epsilon(2.0e-7));
    REQUIRE((*scalars[1].field)(i, j, k)
            == Approx(qr(i, j, k)
                      / (523.598775598
                         * std::pow(5.0e-4, 3.0))).epsilon(2.0e-6));
    for (std::size_t n = 2; n < scalars.size(); ++n)
        REQUIRE((*scalars[n].field)(i, j, k) == 0.0f);
}

TEST_CASE("Milbrandt private moments commit only after transaction acceptance",
          "[physics][microphysics][milbrandt][transaction][moments]")
{
    setup_grid();
    setup_warm_state();
    const double qvs =
        thermodynamics::saturation_mixing_ratio_water(kTestT, kTestP);
    qv.resize(NR, NTH, NZ, static_cast<float>(qvs));
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qr(4, 4, 4) = 1.0e-3f;

    auto scheme = create_microphysics_scheme("milbrandt");
    scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
    const PortableSchemeState before = scheme->portable_state();
    Tendencies tendencies;
    compute(*scheme, dt, tendencies);
    const PortableSchemeState prepared = scheme->portable_state();

    REQUIRE(before.fields.size() == prepared.fields.size());
    for (std::size_t moment = 0; moment < 6; ++moment)
    {
        REQUIRE(before.fields[moment].size() == prepared.fields[moment].size());
        for (std::size_t index = 0;
             index < before.fields[moment].size(); ++index)
        {
            REQUIRE(before.fields[moment].data()[index]
                    == prepared.fields[moment].data()[index]);
        }
    }
    bool has_private_tendency = false;
    for (std::size_t tendency = 6; tendency < 12; ++tendency)
        for (std::size_t index = 0;
             index < prepared.fields[tendency].size(); ++index)
            has_private_tendency =
                has_private_tendency
                || prepared.fields[tendency].data()[index] != 0.0f;
    REQUIRE(has_private_tendency);

    std::string error;
    REQUIRE(scheme->validate_prognostic_update(dt, error));
    REQUIRE(error.empty());
    scheme->commit_prognostic_update(dt);
    const PortableSchemeState committed = scheme->portable_state();
    bool moment_changed = false;
    for (std::size_t moment = 0; moment < 6; ++moment)
        for (std::size_t index = 0;
             index < committed.fields[moment].size(); ++index)
            moment_changed = moment_changed
                || committed.fields[moment].data()[index]
                   != before.fields[moment].data()[index];
    REQUIRE(moment_changed);
}

TEST_CASE("Milbrandt stochastic column control scales private number moments with mass tendencies",
          "[physics][microphysics][milbrandt][chaos][moments]")
{
    setup_grid();
    setup_warm_state();
    const double qvs =
        thermodynamics::saturation_mixing_ratio_water(kTestT, kTestP);
    qv.resize(NR, NTH, NZ, static_cast<float>(qvs));
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qr(4, 4, 4) = 1.0e-3f;

    auto scheme = create_microphysics_scheme("milbrandt");
    scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
    Tendencies tendencies;
    compute(*scheme, dt, tendencies);
    const PortableSchemeState deterministic = scheme->portable_state();
    REQUIRE(deterministic.fields.size() >= 12);

    std::vector<float> multipliers(
        static_cast<std::size_t>(NR) * static_cast<std::size_t>(NTH),
        0.5f);
    std::string error;
    REQUIRE(scheme->apply_stochastic_column_multipliers(
        dt, rho, multipliers, error));
    REQUIRE(error.empty());
    const PortableSchemeState stochastic = scheme->portable_state();
    for (std::size_t moment = 0; moment < 6; ++moment)
        for (std::size_t index = 0;
             index < deterministic.fields[moment].size(); ++index)
        {
            REQUIRE(stochastic.fields[moment].data()[index]
                    == deterministic.fields[moment].data()[index]);
            REQUIRE(stochastic.fields[moment + 6].data()[index]
                    == Approx(0.5f
                              * deterministic.fields[moment + 6]
                                    .data()[index]));
        }
    REQUIRE(scheme->validate_prognostic_update(dt, error));
    scheme->commit_prognostic_update(dt);
    const PortableSchemeState committed = scheme->portable_state();
    for (std::size_t moment = 0; moment < 6; ++moment)
        for (std::size_t index = 0;
             index < committed.fields[moment].size(); ++index)
        {
            const double expected =
                static_cast<double>(deterministic.fields[moment].data()[index])
                + dt * static_cast<double>(
                    stochastic.fields[moment + 6].data()[index]);
            REQUIRE(committed.fields[moment].data()[index]
                    == Approx(expected).margin(1.0e-5
                        * std::max(1.0, std::abs(expected))));
        }
}

TEST_CASE("Milbrandt portable state rejects malformed history transactionally",
          "[physics][microphysics][checkpoint][mod03]")
{
    const auto require_transactional_rejection = [](PortableSchemeState malformed)
    {
        auto scheme = create_microphysics_scheme("milbrandt");
        const PortableSchemeState original = scheme->portable_state();
        std::string error;
        REQUIRE_FALSE(scheme->restore_portable_state(malformed, error));
        REQUIRE_FALSE(error.empty());
        const PortableSchemeState after = scheme->portable_state();
        REQUIRE(after.version == original.version);
        REQUIRE(after.fields.size() == original.fields.size());
        REQUIRE(after.reals == original.reals);
    };

    PortableSchemeState malformed =
        create_microphysics_scheme("milbrandt")->portable_state();
    malformed.version = 999;
    require_transactional_rejection(malformed);

    malformed = create_microphysics_scheme("milbrandt")->portable_state();
    malformed.reals[2] = -1.0e-3;
    require_transactional_rejection(malformed);

    malformed = create_microphysics_scheme("milbrandt")->portable_state();
    malformed.reals[5] = 0.0;
    require_transactional_rejection(malformed);

    malformed = create_microphysics_scheme("milbrandt")->portable_state();
    malformed.reals[16] = 3.0;
    require_transactional_rejection(malformed);
}

TEST_CASE("Milbrandt rejects initialized checkpoints with obsolete volumetric moments",
          "[physics][microphysics][milbrandt][checkpoint][compatibility]")
{
    setup_grid();
    setup_warm_state();
    qv.resize(NR, NTH, NZ, 0.010f);
    qc.resize(NR, NTH, NZ, 1.0e-3f);
    qr.resize(NR, NTH, NZ, 1.0e-3f);

    auto source = create_microphysics_scheme("milbrandt");
    source->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
    const PortableSchemeState current = source->portable_state();
    PortableSchemeState legacy;
    legacy.version = 2;
    legacy.integers = {0, current.integers[0]};
    legacy.reals = {1.0e-3, 1.0e-3, 6.0e-4};
    legacy.reals.insert(
        legacy.reals.end(), current.reals.begin(), current.reals.end());
    legacy.fields = {
        current.fields[1], current.fields[2], current.fields[3],
        current.fields[4], current.fields[5],
        Field3D{}, Field3D{}, Field3D{}, Field3D{}, Field3D{},
        current.fields[7], current.fields[8], current.fields[9],
        current.fields[10], current.fields[11],
        Field3D{}, Field3D{}, Field3D{}, Field3D{}, Field3D{},
        current.fields[12]};

    auto target = create_microphysics_scheme("milbrandt");
    std::string error;
    REQUIRE_FALSE(target->restore_portable_state(legacy, error));
    REQUIRE(error.find("volumetric number units") != std::string::npos);
}

TEST_CASE("Thompson restored aerosol fields retain their profile identity",
          "[physics][microphysics][thompson][checkpoint][atomic]")
{
    setup_grid();
    setup_warm_state();
    qc.resize(NR, NTH, NZ, 1.0e-4f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);

    tmv::MicrophysicsConfig profile_a;
    profile_a.thompson.water_friendly_aerosol_surface_m3 = 375.0e6;
    profile_a.thompson.ice_friendly_aerosol_surface_m3 = 2.25e6;
    auto source = create_microphysics_scheme("thompson", profile_a);
    source->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
    const PortableSchemeState checkpoint = source->portable_state();

    auto restored = create_microphysics_scheme("thompson");
    std::string error;
    REQUIRE(restored->restore_portable_state(checkpoint, error));
    const PortableSchemeState bound = restored->portable_state();

    tmv::MicrophysicsConfig profile_b = profile_a;
    profile_b.thompson.water_friendly_aerosol_surface_m3 = 425.0e6;
    restored->configure(profile_b);
    REQUIRE_THROWS_AS(
        restored->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh),
        std::invalid_argument);
    const PortableSchemeState rejected = restored->portable_state();
    REQUIRE(rejected.fields.size() == bound.fields.size());
    for (std::size_t field = 0; field < bound.fields.size(); ++field)
        REQUIRE(std::memcmp(bound.fields[field].data(),
                            rejected.fields[field].data(),
                            bound.fields[field].size() * sizeof(float)) == 0);

    restored->configure(profile_a);
    REQUIRE_NOTHROW(
        restored->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh));

    auto malformed = checkpoint;
    malformed.reals[3] = 50.0e6;
    const PortableSchemeState before_malformed = restored->portable_state();
    REQUIRE_FALSE(restored->restore_portable_state(malformed, error));
    REQUIRE(restored->portable_state().reals == before_malformed.reals);
}

TEST_CASE("Milbrandt uninitialized v3 state migrates to the double-moment layout",
          "[physics][microphysics][checkpoint][compatibility]")
{
    tmv::MicrophysicsConfig configured;
    configured.milbrandt.hail_processes = false;
    const PortableSchemeState current =
        create_microphysics_scheme("milbrandt", configured)->portable_state();
    PortableSchemeState legacy;
    legacy.version = 3;
    legacy.integers = {0, 0};
    legacy.reals = {1.0e-3, 1.0e-3, 6.0e-4};
    legacy.reals.insert(
        legacy.reals.end(), current.reals.begin(), current.reals.end());
    legacy.fields.resize(23);

    auto restored = create_microphysics_scheme("milbrandt", configured);
    std::string error;
    REQUIRE(restored->restore_portable_state(legacy, error));
    REQUIRE(error.empty());
    REQUIRE(restored->portable_state().version == 4);
    REQUIRE(restored->portable_state().reals == current.reals);
    REQUIRE(restored->portable_state().integers
            == std::vector<std::uint64_t>{0});
}

TEST_CASE("Sedimentation fall speed carries the density correction aloft",
          "[physics][microphysics][sedimentation]")
{
    // SCIENCE.md M11 + M12. Rain's mass-weighted LFO83 speed is
    // pref * lambda^-0.8 * sqrt(rho_ref/rho); lambda carries rho^-1/4, so
    // at fixed q the altitude dependence is (rho0/rho)^0.5 * rho^0.2
    // = (rho0/rho)^0.3. The top cell only loses mass (nothing falls in
    // from above), so its loss rate over the surface sink rate isolates
    // the factor.
    setup_grid();
    setup_warm_state();
    for (int k = 0; k < NZ; ++k)
    {
        rho0_base[static_cast<size_t>(k)] = 0.75;
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                rho(i, j, k) = static_cast<float>(
                    1.2 * std::exp(-(k * dz) / 8000.0));
    }
    const double qvs =
        thermodynamics::saturation_mixing_ratio_water(kTestT, kTestP);
    qv.resize(NR, NTH, NZ, static_cast<float>(qvs));
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 1.0e-3f);

    for (const char* name : {"lin", "thompson"})
    {
        INFO("scheme = " << name);
        auto scheme = create_microphysics_scheme(name);
        Tendencies t;
        compute(*scheme, dt, t);

        const int i = 2, j = 3, k_top = NZ - 1;
        const double top_loss_rate = -static_cast<double>(t.dqr(i, j, k_top));
        const double sink_rate = scheme->bottom_sedimentation_sink_rate(
            rho, qr, qs, qg, qh, i, j, dt);
        const double expected_factor = std::pow(
            static_cast<double>(rho(i, j, 0)) /
                static_cast<double>(rho(i, j, k_top)), 0.3);

        REQUIRE(sink_rate > 0.0);
        if (std::string(name) == "lin")
            REQUIRE(top_loss_rate / sink_rate
                    == Approx(expected_factor).epsilon(1.0e-4));
        else
            REQUIRE(top_loss_rate / sink_rate
                    == Approx(1.2445200532).epsilon(1.0e-4));
    }
}

TEST_CASE("Melting rate scales with warm excess per species",
          "[physics][microphysics][melting]")
{
    // SCIENCE.md M10: the old constant 1e-3 1/s melt rate let
    // frozen species survive kilometers below the melting level. The
    // rate is now c_melt_X * (T - T0) * q with heat-transfer-derived
    // per-species coefficients, so snow melts an order of magnitude
    // faster than graupel, which melts faster than hail.
    setup_grid();
    setup_warm_state();

    // T = T0 + 2 at p = p0 (Exner = 1 -> theta = T): melting is the only
    // process touching the frozen species. Saturated, cloud-free air
    // keeps condensation/evaporation inert; uniform rho0 + uniform q
    // make interior-cell sedimentation a net no-op.
    const double T_warm = microphysics_constants::T0 + 2.0;
    theta.resize(NR, NTH, NZ, static_cast<float>(T_warm));
    const double qvs = thermodynamics::saturation_mixing_ratio_water(
        T_warm, kTestP);
    qv.resize(NR, NTH, NZ, static_cast<float>(qvs));
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 1.0e-3f);
    qg.resize(NR, NTH, NZ, 1.0e-3f);
    qh.resize(NR, NTH, NZ, 1.0e-3f);

    const double warm_excess = 2.0;
    const double q0 = 1.0e-3;
    const int i = 4, j = 4, k = 4;  // interior: sedimentation cancels

    SECTION("lin melts all three frozen species")
    {
        auto scheme = create_microphysics_scheme("lin");
        Tendencies t;
        compute(*scheme, dt, t);

        const double expect_s =
            microphysics_constants::c_melt_snow_per_K * warm_excess * q0;
        const double expect_g =
            microphysics_constants::c_melt_graupel_per_K * warm_excess * q0;
        const double expect_h =
            microphysics_constants::c_melt_hail_per_K * warm_excess * q0;

        REQUIRE(static_cast<double>(t.dqs(i, j, k))
                == Approx(-expect_s).epsilon(1e-4));
        REQUIRE(static_cast<double>(t.dqg(i, j, k))
                == Approx(-expect_g).epsilon(1e-4));
        REQUIRE(static_cast<double>(t.dqh(i, j, k))
                == Approx(-expect_h).epsilon(1e-4));
        REQUIRE(static_cast<double>(t.dqr(i, j, k))
                == Approx(expect_s + expect_g + expect_h).epsilon(1e-4));
        REQUIRE(t.dtheta(i, j, k) < 0.0f);
    }

    SECTION("Thompson mp=28 rejects hail rather than relabeling it")
    {
        auto scheme = create_microphysics_scheme("thompson");
        Tendencies t;
        REQUIRE_THROWS_AS(compute(*scheme, dt, t), std::invalid_argument);

        qh.fill(0.0f);
        REQUIRE_NOTHROW(compute(*scheme, dt, t));
        REQUIRE(t.dqs(i, j, k) < 0.0f);
        REQUIRE(t.dqg(i, j, k) < 0.0f);
        REQUIRE(t.dqh(i, j, k) == 0.0f);
    }

    SECTION("kessler leaves every frozen species inert")
    {
        auto scheme = create_microphysics_scheme("kessler");
        Tendencies t;
        compute(*scheme, dt, t);

        REQUIRE(t.dqi(i, j, k) == 0.0f);
        REQUIRE(t.dqs(i, j, k) == 0.0f);
        REQUIRE(t.dqg(i, j, k) == 0.0f);
        REQUIRE(t.dqh(i, j, k) == 0.0f);
        REQUIRE(t.dqr(i, j, k) == 0.0f);
        REQUIRE(t.dtheta(i, j, k) == Approx(0.0f).margin(3.0e-6));
    }

    SECTION("no melting below freezing")
    {
        const double T_cold = microphysics_constants::T0 - 1.0;
        theta.resize(NR, NTH, NZ, static_cast<float>(T_cold));
        // Ice-saturated so deposition/sublimation stay inert; qi = 0 and
        // qc = 0 keep the cold-side sources off (lin sublimates qs only
        // when subsaturated wrt ice).
        const double qvs_i = thermodynamics::saturation_mixing_ratio_ice(
            T_cold, kTestP);
        qv.resize(NR, NTH, NZ, static_cast<float>(qvs_i));

        auto scheme = create_microphysics_scheme("lin");
        Tendencies t;
        compute(*scheme, dt, t);

        REQUIRE(static_cast<double>(t.dqg(i, j, k)) == Approx(0.0).margin(1e-12));
        REQUIRE(static_cast<double>(t.dqh(i, j, k)) == Approx(0.0).margin(1e-12));
        REQUIRE(static_cast<double>(t.dqr(i, j, k)) == Approx(0.0).margin(1e-12));
    }
}

TEST_CASE("Supercooled saturation split restores the WBF band",
          "[physics][microphysics][wbf]")
{
    // SCIENCE.md M7: saturation_mixing_ratio_water used the Buck
    // ICE equation below 0 C, so water and ice saturation coincided and
    // the Wegener-Bergeron-Findeisen band (qvs_ice < qv < qvs_water)
    // had zero width - every vapor-to-ice deposition guard was
    // unreachable dead code.
    setup_grid();
    setup_warm_state();
    // setup_warm_state sizes theta/p/rho/qi/qs/qg/qh only.
    qv.resize(NR, NTH, NZ, 0.0f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    theta.resize(NR, NTH, NZ, static_cast<float>(microphysics_constants::T0));
    qv.fill(static_cast<float>(thermodynamics::saturation_mixing_ratio_water(
        microphysics_constants::T0, kTestP)));

    const double T_cold = microphysics_constants::T0 - 10.0;
    const double qvs_w =
        thermodynamics::saturation_mixing_ratio_water(T_cold, kTestP);
    const double qvs_i =
        thermodynamics::saturation_mixing_ratio_ice(T_cold, kTestP);

    SECTION("water saturation exceeds ice saturation below freezing")
    {
        // Literature e_sw/e_si at -10 C is ~1.10.
        REQUIRE(qvs_w / qvs_i > 1.08);
        REQUIRE(qvs_w / qvs_i < 1.13);
    }

    SECTION("deposition fires inside the band with no condensation")
    {
        // theta = T at p = p0 (Exner = 1). Vapor mid-band, ice present,
        // no liquid: the ONLY active process is vapor-to-ice deposition
        // (satadj inert below water saturation; sublimation inert above
        // ice saturation). Before the fix, satadj clamped qv down to the
        // collapsed qvs and deposition never ran.
        theta.resize(NR, NTH, NZ, static_cast<float>(T_cold));
        const double qv_band = 0.5 * (qvs_i + qvs_w);
        qv.resize(NR, NTH, NZ, static_cast<float>(qv_band));
        qc.resize(NR, NTH, NZ, 0.0f);
        qi.resize(NR, NTH, NZ, 1.0e-4f);

        const int i = 4, j = 4, k = 4;
        for (const char* name : {"lin", "thompson"})
        {
            INFO("scheme = " << name);
            auto scheme = create_microphysics_scheme(name);
            Tendencies t;
            compute(*scheme, dt, t);

            // Both schemes deposit at 1e-3 * (qv - qvs_i) (lin via
            // c_subl, thompson hardcoded). The fp32 state round-trip
            // dominates the error band.
            const double expect_dep = 1.0e-3 * (qv_band - qvs_i);
            REQUIRE(static_cast<double>(t.dqi(i, j, k))
                    == Approx(expect_dep).epsilon(5e-3));
            REQUIRE(static_cast<double>(t.dqv(i, j, k))
                    == Approx(-expect_dep).epsilon(5e-3));
            // No liquid appears mid-band (the pre-fix failure mode was
            // condensation against the collapsed saturation).
            REQUIRE(static_cast<double>(t.dqc(i, j, k))
                    == Approx(0.0).margin(1e-12));
            // Deposition releases L_s: warming.
            REQUIRE(t.dtheta(i, j, k) > 0.0f);
        }
    }

    SECTION("supercooled cloud evaporates while ice deposits (WBF)")
    {
        theta.resize(NR, NTH, NZ, static_cast<float>(T_cold));
        const double qv_band = 0.5 * (qvs_i + qvs_w);
        qv.resize(NR, NTH, NZ, static_cast<float>(qv_band));
        qc.resize(NR, NTH, NZ, 5.0e-4f);
        qi.resize(NR, NTH, NZ, 1.0e-4f);

        const int i = 4, j = 4, k = 4;
        auto scheme = create_microphysics_scheme("lin");
        Tendencies t;
        compute(*scheme, dt, t);

        // Liquid shrinks (evaporation toward water saturation plus
        // freezing/riming sinks), ice grows (deposition + freezing):
        // the WBF transfer direction.
        REQUIRE(t.dqc(i, j, k) < 0.0f);
        REQUIRE(t.dqi(i, j, k) > 0.0f);
    }
}

TEST_CASE("Cloud freezing ramps with supercooling to a homogeneous cutoff",
          "[physics][microphysics][freezing]")
{
    // SCIENCE.md M8: the old constant freezing rate glaciated
    // -0.5 C cloud exactly as fast as -60 C cloud. The rate now ramps
    // linearly with supercooling and drains the whole cloud in one step
    // at the homogeneous-nucleation threshold T0 - 38.
    setup_grid();

    const int i = 4, j = 4, k = 4;
    const float qc_init = 5.0e-4f;  // below qc0: no autoconversion

    // Saturated wrt water at each temperature: the saturation
    // adjustment stays inert and freezing is the only first-order
    // cloud sink (riming feeds on the freshly frozen ice at second
    // order only).
    auto run_at = [&](double T_target, const char* scheme_name,
                      Tendencies& t)
    {
        setup_warm_state();
        theta.resize(NR, NTH, NZ, static_cast<float>(T_target));
        const double qvs_w =
            thermodynamics::saturation_mixing_ratio_water(T_target, kTestP);
        qv.resize(NR, NTH, NZ, static_cast<float>(qvs_w));
        qc.resize(NR, NTH, NZ, qc_init);
        qr.resize(NR, NTH, NZ, 0.0f);
        auto scheme = create_microphysics_scheme(scheme_name);
        compute(*scheme, dt, t);
    };

    SECTION("rate scales linearly with supercooling")
    {
        // Both mixed-phase schemes use base coefficient 1e-3 (lin c_ihom
        // and Thompson's retained constant). Each leg is pinned to
        // the analytic law with an absolute margin: fp32-cast
        // saturation dust feeds O(1e-9) adjustment tendencies that are
        // visible against the tiny near-freezing signal (a ratio pin
        // at dT = 1 K is ~10% noisy for that reason).
        const double c_base = 1.0e-3;
        const double margin = 2.5e-9;
        for (const char* name : {"lin", "thompson"})
        {
            INFO("scheme = " << name);
            for (double dT : {1.0, 20.0})
            {
                INFO("supercooling = " << dT);
                Tendencies t;
                run_at(microphysics_constants::T0 - dT, name, t);
                const double expect = c_base
                    * (dT / microphysics_constants::homogeneous_freezing_dT)
                    * static_cast<double>(qc_init);
                REQUIRE(static_cast<double>(t.dqc(i, j, k))
                        == Approx(-expect).margin(margin).epsilon(1e-3));
            }
        }
    }

    SECTION("homogeneous threshold drains the cloud in one step")
    {
        for (const char* name : {"lin", "thompson"})
        {
            INFO("scheme = " << name);
            Tendencies t;
            run_at(microphysics_constants::T0 - 40.0, name, t);

            // dt = 1: the full mixing ratio freezes this step.
            REQUIRE(static_cast<double>(t.dqc(i, j, k))
                    == Approx(-static_cast<double>(qc_init) / dt)
                           .epsilon(1e-4));
            // Freezing releases L_f: warming.
            REQUIRE(t.dtheta(i, j, k) > 0.0f);
        }
    }

    SECTION("Kessler never manufactures frozen-species physics")
    {
        Tendencies t;
        setup_warm_state();
        const double T_cold = microphysics_constants::T0 - 20.0;
        theta.resize(NR, NTH, NZ, static_cast<float>(T_cold));
        qv.resize(
            NR, NTH, NZ,
            static_cast<float>(
                thermodynamics::saturation_mixing_ratio_water(
                    T_cold, kTestP)));
        qc.resize(NR, NTH, NZ, qc_init);
        qr.resize(NR, NTH, NZ, 0.0f);
        qi.resize(NR, NTH, NZ, 1.0e-3f);
        qs.resize(NR, NTH, NZ, 1.0e-3f);
        qg.resize(NR, NTH, NZ, 1.0e-3f);
        qh.resize(NR, NTH, NZ, 1.0e-3f);

        auto scheme = create_microphysics_scheme("kessler");
        compute(*scheme, dt, t);

        REQUIRE(t.dqi(i, j, k) == 0.0f);
        REQUIRE(t.dqs(i, j, k) == 0.0f);
        REQUIRE(t.dqg(i, j, k) == 0.0f);
        REQUIRE(t.dqh(i, j, k) == 0.0f);
        REQUIRE(t.dtheta(i, j, k) == Approx(0.0f).margin(3.0e-6));
        REQUIRE(t.dqc(i, j, k) == Approx(0.0f).margin(3.0e-9));
    }
}

TEST_CASE("Fall speeds match the gamma-integrated literature forms",
          "[physics][microphysics][fallspeed]")
{
    // SCIENCE.md M12: the old power laws gave 27 m/s rain at
    // 1 g/kg (physical ~5-6) with an 11.6 m/s MINIMUM on trace rain
    // from the max(q, 1e-6) floor, and graupel/hail shared one
    // coefficient set. Reference values below are hand-computed from
    // the LFO83 mass-weighted forms at q = 1 g/kg, rho_air = 1.2
    // (see fall_speeds.hpp). The surface sink rate is Vt q / dz, so
    // Vt = sink dz / q probes the speed through the public interface.
    setup_grid();
    setup_warm_state();
    qv.resize(NR, NTH, NZ, 0.0f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);

    const int i = 4, j = 4;
    const float q0 = 1.0e-3f;

    auto probe = [&](const char* scheme_name, Field3D& species)
    {
        species.resize(NR, NTH, NZ, q0);
        auto scheme = create_microphysics_scheme(scheme_name);
        scheme->set_vertical_spacing_m(dz);
        if (std::string(scheme_name) == "thompson")
        {
            Tendencies t;
            compute(*scheme, dt, t);
        }
        const double sink = scheme->bottom_sedimentation_sink_rate(
            rho, qr, qs, qg, qh, i, j, dt);
        species.resize(NR, NTH, NZ, 0.0f);
        return sink * dz / static_cast<double>(q0);
    };

    SECTION("Lin uses the LFO83 mass-weighted speeds")
    {
        REQUIRE(probe("lin", qr) == Approx(5.420).epsilon(2e-3));
        REQUIRE(probe("lin", qs) == Approx(1.206).epsilon(2e-3));
        REQUIRE(probe("lin", qg) == Approx(5.825).epsilon(2e-3));
        REQUIRE(probe("lin", qh) == Approx(10.584).epsilon(2e-3));
    }

    SECTION("Thompson uses its mp=28 particle distributions")
    {
        REQUIRE(probe("thompson", qr) == Approx(4.05584).epsilon(2e-3));
        REQUIRE(probe("thompson", qs) == Approx(1.46947).epsilon(2e-3));
        REQUIRE(probe("thompson", qg) == Approx(6.07095).epsilon(2e-3));
        qh.fill(q0);
        auto scheme = create_microphysics_scheme("thompson");
        Tendencies t;
        REQUIRE_THROWS_AS(compute(*scheme, dt, t), std::invalid_argument);
        qh.fill(0.0f);
    }

    SECTION("kessler rain uses the classic Kessler speed")
    {
        REQUIRE(probe("kessler", qr) == Approx(5.801).epsilon(2e-3));
        REQUIRE(probe("kessler", qg) == 0.0);
        REQUIRE(probe("kessler", qh) == 0.0);
    }

    SECTION("trace rain falls slowly (no speed floor)")
    {
        // The old max(q, 1e-6) floor imposed a minimum 11.6 m/s on any
        // trace of rain; the gamma form scales as q^0.2.
        qr.resize(NR, NTH, NZ, 1.0e-7f);
        auto scheme = create_microphysics_scheme("lin");
        scheme->set_vertical_spacing_m(dz);
        const double sink = scheme->bottom_sedimentation_sink_rate(
            rho, qr, qs, qg, qh, i, j, dt);
        const double Vt = sink * dz / 1.0e-7;
        REQUIRE(Vt > 0.0);
        REQUIRE(Vt < 1.0);
    }
}

TEST_CASE("Lin honors a shifted physical domain and its true fallout surface",
          "[physics][microphysics][lin][physical-domain][collocated]")
{
    setup_grid();
    const float poison = std::numeric_limits<float>::quiet_NaN();
    theta.resize(NR, NTH, NZ, poison);
    p.resize(NR, NTH, NZ, poison);
    rho.resize(NR, NTH, NZ, poison);
    qv.resize(NR, NTH, NZ, poison);
    qc.resize(NR, NTH, NZ, poison);
    qr.resize(NR, NTH, NZ, poison);
    qi.resize(NR, NTH, NZ, poison);
    qs.resize(NR, NTH, NZ, poison);
    qg.resize(NR, NTH, NZ, poison);
    qh.resize(NR, NTH, NZ, poison);
    const PhysicalCellDomain domain{1, NR - 1, 1, NTH - 1, 1, NZ - 1};
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                theta(i, j, k) = 300.0f;
                p(i, j, k) = 100000.0f;
                rho(i, j, k) = 1.2f;
                qv(i, j, k) = 0.004f;
                qc(i, j, k) = 0.0f;
                qr(i, j, k) = k == domain.k_begin ? 1.0e-3f : 0.0f;
                qi(i, j, k) = 0.0f;
                qs(i, j, k) = 0.0f;
                qg(i, j, k) = 0.0f;
                qh(i, j, k) = 0.0f;
            }

    auto scheme = create_microphysics_scheme("lin");
    scheme->set_physical_cell_domain(NR, NTH, NZ, domain);
    Tendencies tendencies;
    compute(*scheme, dt, tendencies);
    REQUIRE(tendencies.dqr(2, 2, domain.k_begin) < 0.0f);
    REQUIRE(scheme->bottom_sedimentation_sink_rate(
        rho, qr, qs, qg, qh, 2, 2, dt) > 0.0);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const bool physical = i >= domain.i_begin && i < domain.i_end
                    && j >= domain.j_begin && j < domain.j_end
                    && k >= domain.k_begin && k < domain.k_end;
                if (!physical)
                {
                    REQUIRE(tendencies.dtheta(i, j, k) == 0.0f);
                    REQUIRE(tendencies.dqv(i, j, k) == 0.0f);
                    REQUIRE(tendencies.dqc(i, j, k) == 0.0f);
                    REQUIRE(tendencies.dqr(i, j, k) == 0.0f);
                    REQUIRE(tendencies.dqi(i, j, k) == 0.0f);
                    REQUIRE(tendencies.dqs(i, j, k) == 0.0f);
                    REQUIRE(tendencies.dqg(i, j, k) == 0.0f);
                    REQUIRE(tendencies.dqh(i, j, k) == 0.0f);
                }
            }
}

TEST_CASE("Thompson honors a shifted physical domain including private moments",
          "[physics][microphysics][thompson][physical-domain][collocated]")
{
    setup_grid();
    const float poison = std::numeric_limits<float>::quiet_NaN();
    theta.resize(NR, NTH, NZ, poison);
    p.resize(NR, NTH, NZ, poison);
    rho.resize(NR, NTH, NZ, poison);
    qv.resize(NR, NTH, NZ, poison);
    qc.resize(NR, NTH, NZ, poison);
    qr.resize(NR, NTH, NZ, poison);
    qi.resize(NR, NTH, NZ, poison);
    qs.resize(NR, NTH, NZ, poison);
    qg.resize(NR, NTH, NZ, poison);
    qh.resize(NR, NTH, NZ, poison);
    Field3D vertical_velocity(NR, NTH, NZ, poison);
    const PhysicalCellDomain domain{1, NR - 1, 1, NTH - 1, 1, NZ - 1};
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                theta(i, j, k) = 300.0f;
                p(i, j, k) = 100000.0f;
                rho(i, j, k) = 1.2f;
                qv(i, j, k) = 0.004f;
                qc(i, j, k) = 0.0f;
                qr(i, j, k) = k == domain.k_begin ? 1.0e-3f : 0.0f;
                qi(i, j, k) = 0.0f;
                qs(i, j, k) = 0.0f;
                qg(i, j, k) = 0.0f;
                qh(i, j, k) = 0.0f;
                vertical_velocity(i, j, k) = 0.0f;
            }

    auto scheme = create_microphysics_scheme("thompson");
    scheme->set_physical_cell_domain(NR, NTH, NZ, domain);
    scheme->set_vertical_spacing_m(dz);
    scheme->set_vertical_velocity_field(vertical_velocity, false);
    REQUIRE_NOTHROW(scheme->prepare_prognostic_state(
        rho, qc, qr, qi, qs, qg, qh));
    const PortableSchemeState initialized = scheme->portable_state();
    REQUIRE(initialized.fields.size() == 10);
    for (const Field3D& field : initialized.fields)
        REQUIRE(field(0, 0, 0) == 0.0f);

    Tendencies tendencies;
    REQUIRE_NOTHROW(compute(*scheme, dt, tendencies));
    REQUIRE(tendencies.dqr(2, 2, domain.k_begin) < 0.0f);
    REQUIRE(scheme->bottom_sedimentation_sink_rate(
        rho, qr, qs, qg, qh, 2, 2, dt) > 0.0);
    REQUIRE_NOTHROW([&] {
        std::string error;
        REQUIRE(scheme->validate_prognostic_update(dt, error));
        scheme->commit_prognostic_update(dt);
    }());
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const bool physical = i >= domain.i_begin && i < domain.i_end
                    && j >= domain.j_begin && j < domain.j_end
                    && k >= domain.k_begin && k < domain.k_end;
                if (!physical)
                {
                    REQUIRE(tendencies.dtheta(i, j, k) == 0.0f);
                    REQUIRE(tendencies.dqv(i, j, k) == 0.0f);
                    REQUIRE(tendencies.dqc(i, j, k) == 0.0f);
                    REQUIRE(tendencies.dqr(i, j, k) == 0.0f);
                    REQUIRE(tendencies.dqi(i, j, k) == 0.0f);
                    REQUIRE(tendencies.dqs(i, j, k) == 0.0f);
                    REQUIRE(tendencies.dqg(i, j, k) == 0.0f);
                    REQUIRE(tendencies.dqh(i, j, k) == 0.0f);
                }
            }
    const PortableSchemeState committed = scheme->portable_state();
    for (const Field3D& field : committed.fields)
        REQUIRE(field(0, 0, 0) == 0.0f);
}

TEST_CASE("Rain freezing and snow riming pathways are live",
          "[physics][microphysics][pathways]")
{
    // SCIENCE.md M9: the 6-class scheme had no rain-freezing
    // pathway, hail had NO source process (it could only melt or fall),
    // and snow never rimed. Bigg stochastic freezing of supercooled
    // rain now seeds hail (LFO83 routes frozen drops to the 913-density
    // class) and LFO83 PSACW grows snow on supercooled cloud.
    setup_grid();

    SECTION("rate helpers match hand-computed LFO83 values")
    {
        // q = 1 g/kg, rho_air = 1.2; Bigg at 15 K supercooling.
        REQUIRE(fall_speeds::rain_bigg_freezing_rate(1.0e-3, 1.2, 15.0)
                == Approx(1.278e-5).epsilon(2e-2));
        // PSACW per unit cloud mixing ratio.
        REQUIRE(fall_speeds::snow_cloud_collection_rate(1.0e-3, 1.2)
                == Approx(6.268e-3).epsilon(2e-2));
        // No freezing without supercooling.
        REQUIRE(fall_speeds::rain_bigg_freezing_rate(1.0e-3, 1.2, 0.0)
                == 0.0);
    }

    SECTION("supercooled rain freezes into hail (the hail source)")
    {
        setup_warm_state();
        const double T_cold = microphysics_constants::T0 - 15.0;
        theta.resize(NR, NTH, NZ, static_cast<float>(T_cold));
        const double qvs_w =
            thermodynamics::saturation_mixing_ratio_water(T_cold, kTestP);
        qv.resize(NR, NTH, NZ, static_cast<float>(qvs_w));
        qc.resize(NR, NTH, NZ, 0.0f);
        qr.resize(NR, NTH, NZ, 1.0e-3f);

        auto scheme = create_microphysics_scheme("lin");
        Tendencies t;
        compute(*scheme, dt, t);

        const int i = 4, j = 4, k = 4;
        const double rho_air =
            kTestP / (microphysics_constants::R_d * T_cold);
        const double expect = fall_speeds::rain_bigg_freezing_rate(
            1.0e-3, rho_air, 15.0);
        REQUIRE(static_cast<double>(t.dqh(i, j, k))
                == Approx(expect).epsilon(1e-3));
        REQUIRE(static_cast<double>(t.dqr(i, j, k))
                == Approx(-expect).epsilon(1e-3));
        // Freezing releases L_f: warming.
        REQUIRE(t.dtheta(i, j, k) > 0.0f);
    }

    SECTION("snow rimes supercooled cloud")
    {
        setup_warm_state();
        const double T_cold = microphysics_constants::T0 - 10.0;
        theta.resize(NR, NTH, NZ, static_cast<float>(T_cold));
        const double qvs_w =
            thermodynamics::saturation_mixing_ratio_water(T_cold, kTestP);
        qv.resize(NR, NTH, NZ, static_cast<float>(qvs_w));
        qc.resize(NR, NTH, NZ, 5.0e-4f);  // below qc0: no autoconversion
        qr.resize(NR, NTH, NZ, 0.0f);
        qs.resize(NR, NTH, NZ, 1.0e-3f);

        auto scheme = create_microphysics_scheme("lin");
        Tendencies t;
        compute(*scheme, dt, t);

        const int i = 4, j = 4, k = 4;
        const double rho_air =
            kTestP / (microphysics_constants::R_d * T_cold);
        // M8 cloud freezing removes 2.6e-4 of qc first; 1% margin
        // covers it plus the trace ice interactions it seeds.
        const double expect = fall_speeds::snow_cloud_collection_rate(
            1.0e-3, rho_air) * 5.0e-4;
        REQUIRE(static_cast<double>(t.dqs(i, j, k))
                == Approx(expect).epsilon(1e-2));
        REQUIRE(t.dtheta(i, j, k) > 0.0f);
    }
}

TEST_CASE("Fall-speed primitives retain exact positive air density",
          "[physics][microphysics][sedimentation][density][config-truth]")
{
    constexpr double q = 1.0e-3;
    constexpr double rho_ref = 1.0e-3;
    constexpr double rho_low = 5.0e-4;
    const double ratio = rho_low / rho_ref;

    REQUIRE(fall_speeds::rain_mass_weighted(q, rho_low) /
                fall_speeds::rain_mass_weighted(q, rho_ref)
            == Approx(std::pow(ratio, fall_speeds::kRainB / 4.0))
                .epsilon(2.0e-13));
    REQUIRE(fall_speeds::graupel_mass_weighted(q, rho_low) /
                fall_speeds::graupel_mass_weighted(q, rho_ref)
            == Approx(std::pow(ratio, -3.0 / 8.0))
                .epsilon(2.0e-13));
    REQUIRE(fall_speeds::kessler_rain(q, rho_low) /
                fall_speeds::kessler_rain(q, rho_ref)
            == Approx(std::pow(ratio, fall_speeds::kKesslerRainB))
                .epsilon(2.0e-13));
}

TEST_CASE("Theta clamp bound is reconciled to 700 K across layers (science audit)",
          "[theta][robustness]")
{
    using thermodynamics::theta_to_temperature;

    // State clamp (simulation.hpp): legitimate stratospheric theta near a
    // 16 km top (~400-430 K) must NOT be clipped - the cap is theta_max_k
    // (700 K), never 500.
    REQUIRE(clamp_theta_k(650.0f) == Approx(650.0f));
    REQUIRE(clamp_theta_k(800.0f) == Approx(theta_max_k));   // 700
    REQUIRE(clamp_theta_k(150.0f) == Approx(theta_min_k));   // 200
    REQUIRE(std::isfinite(clamp_theta_k(
        std::numeric_limits<float>::quiet_NaN())));          // finite terminal

    // Conversion layer must agree with the state cap: theta in (500, 700]
    // is no longer silently treated as 500 (the audit's cross-layer
    // disagreement). At p = p0 the Exner factor is 1, so T == theta_safe.
    const double p0 = microphysics_constants::p0;
    const double T500 = theta_to_temperature(500.0, p0);
    const double T600 = theta_to_temperature(600.0, p0);
    REQUIRE(T500 == Approx(500.0).margin(1.0));
    REQUIRE(T600 == Approx(600.0).margin(1.0));   // would be ~500 under the old cap
    REQUIRE(T600 > T500 + 50.0);
}

TEST_CASE("NaN-theta fallback keeps the pre-step value, not a surface cold bomb (science audit)",
          "[theta][robustness]")
{
    // Executable spec of the inline NaN-theta fallback shared by the apply
    // paths - dynamics.cpp (SGS apply + ghosts), diffusion_step.cpp,
    // microphysics_step.cpp, and the sgs_apply / microphysics_apply shaders.
    // A non-finite integrated theta keeps the cell's pre-step theta (its own
    // altitude's value); the scalar reference theta0 is only the terminal
    // backstop when the pre-step value is itself non-finite. Before the fix
    // the fallback was always theta0 (~300 K) - a large cold anomaly at
    // altitude (a 370 K cell dropped ~70 K, seeding spurious buoyancy).
    auto nan_theta_fallback = [](double theta_new, double theta_old) -> double
    {
        if (std::isfinite(theta_new)) return theta_new;
        return std::isfinite(theta_old) ? theta_old
                                        : static_cast<double>(theta0);
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double strat_theta = 370.0;  // ~10-12 km

    REQUIRE(nan_theta_fallback(nan, strat_theta) == Approx(strat_theta));
    REQUIRE(nan_theta_fallback(nan, strat_theta)
            != Approx(static_cast<double>(theta0)));   // not the ~300 K cold bomb
    REQUIRE(nan_theta_fallback(372.5, strat_theta) == Approx(372.5));  // finite untouched
    REQUIRE(std::isfinite(nan_theta_fallback(nan, nan)));
    REQUIRE(nan_theta_fallback(nan, nan)
            == Approx(static_cast<double>(theta0)));   // terminal backstop
}

TEST_CASE("Milbrandt refuses a native request before touching CPU state", "[physics][microphysics][milbrandt][native]")
{
    setup_grid();
    setup_warm_state();
    qv.resize(NR, NTH, NZ, .008f);
    qc.resize(NR, NTH, NZ, .001f);
    qr.resize(NR, NTH, NZ, 0);
    auto scheme = create_microphysics_scheme("milbrandt");
    scheme->set_require_native_dispatch(true);
    Tendencies rates;
    rates.resize(NR, NTH, NZ);
    rates.dtheta.fill(17);
    const auto before = scheme->portable_state();
    REQUIRE_THROWS_WITH(scheme->compute_tendencies(p,
                                                   theta,
                                                   rho,
                                                   qv,
                                                   qc,
                                                   qr,
                                                   qi,
                                                   qs,
                                                   qg,
                                                   qh,
                                                   dt,
                                                   rates.dtheta,
                                                   rates.dqv,
                                                   rates.dqc,
                                                   rates.dqr,
                                                   rates.dqi,
                                                   rates.dqs,
                                                   rates.dqg,
                                                   rates.dqh),
                        Catch::Matchers::Contains("native Milbrandt process ownership"));
    CHECK(std::all_of(rates.dtheta.data(), rates.dtheta.data() + rates.dtheta.size(), [](float x) { return x == 17; }));
    const auto after = scheme->portable_state();
    for (std::size_t n = 0; n < before.fields.size(); ++n)
        CHECK(after.fields[n].size() == before.fields[n].size());
}

TEST_CASE("Milbrandt number updates reject layout and FP32 overflow atomically",
          "[physics][microphysics][milbrandt][state][atomic]")
{
    auto scheme  = create_microphysics_scheme("milbrandt");
    auto initial = scheme->portable_state();
    for (std::size_t n = 0; n < 12; ++n)
        initial.fields[n].resize(8, 8, 8, n < 6 ? 4.0f : -1.0f);
    initial.fields[12].resize(8, 8, 1, 0);
    std::string error;
    REQUIRE(scheme->restore_portable_state(initial, error));
    const auto same_state = [&](const PortableSchemeState& expected)
    {
        const auto actual = scheme->portable_state();
        REQUIRE(actual.fields.size() == expected.fields.size());
        for (std::size_t n = 0; n < actual.fields.size(); ++n)
            require_same_field(expected.fields[n], actual.fields[n]);
    };
    std::vector<float> multipliers(64, 10);
    multipliers.back()              = -1;
    const auto original_multipliers = multipliers;
    REQUIRE_FALSE(scheme->apply_stochastic_column_multipliers(1, rho, multipliers, error));
    CHECK(multipliers == original_multipliers);
    same_state(initial);

    auto overflow = initial;
    overflow.fields[0].fill(std::numeric_limits<float>::max());
    overflow.fields[6].fill(std::numeric_limits<float>::max());
    REQUIRE(scheme->restore_portable_state(overflow, error));
    REQUIRE_FALSE(scheme->validate_prognostic_update(1, error));
    multipliers.assign(64, 1);
    REQUIRE_FALSE(scheme->apply_stochastic_column_multipliers(1, rho, multipliers, error));
    CHECK(std::all_of(multipliers.begin(), multipliers.end(), [](float x) { return x == 1; }));
    same_state(overflow);

    overflow.fields[0].fill(4);
    REQUIRE(scheme->restore_portable_state(overflow, error));
    multipliers.assign(64, 2);
    REQUIRE_FALSE(scheme->apply_stochastic_column_multipliers(1, rho, multipliers, error));
    CHECK(std::all_of(multipliers.begin(), multipliers.end(), [](float x) { return x == 2; }));
    same_state(overflow);

    REQUIRE(scheme->restore_portable_state(initial, error));
    auto moments = scheme->prognostic_scalars_for_transport();
    moments[5].field->resize(4, 8, 16, 4);
    REQUIRE_FALSE(scheme->validate_prognostic_update(1, error));
    REQUIRE_FALSE(scheme->apply_stochastic_column_multipliers(1, rho, multipliers, error));
    CHECK(std::all_of(multipliers.begin(), multipliers.end(), [](float x) { return x == 2; }));

    REQUIRE(scheme->restore_portable_state(initial, error));
    multipliers.assign(64, 10);
    REQUIRE(scheme->apply_stochastic_column_multipliers(1, rho, multipliers, error));
    REQUIRE(scheme->validate_prognostic_update(1, error));
    scheme->commit_prognostic_update(1);
    for (const auto& moment : scheme->prognostic_scalars_for_transport())
        CHECK(std::all_of(moment.field->data(),
                          moment.field->data() + moment.field->size(),
                          [](float x) { return std::isfinite(x) && x >= 0 && x < 1e-5f; }));
}

TEST_CASE("Milbrandt coupled cloud cancellation preserves water and latent heat without widening guards",
          "[physics][microphysics][milbrandt][storage]")
{
    const std::array<float, 4> initial{-2.9103830456733704e-11f, -0.00048533038352616131f, 0.0f, 0.0f};
    auto                       rates = initial;
    const double               dt    = 1.0 / 6.0;
    REQUIRE(tmv::milbrandt_detail::finalize_cloud_storage(
        0, 7.8204502642620355e-5f, .0003375302767381072f, 1, 0, dt, rates));
    CHECK(rates[0] == 0);
    CHECK(double(rates[0]) + rates[1] == double(initial[0]) + initial[1]);
    const double condensation = double(rates[0]) - initial[0];
    CHECK(rates[2] == Approx(condensation * microphysics_constants::L_v / microphysics_constants::cp).epsilon(2e-7));
    for (float cloud : {0.0f, 1e-12f, 1e-8f})
    {
        auto invalid        = initial;
        invalid[0]          = -1e-3f;
        const auto retained = invalid;
        CHECK_FALSE(tmv::milbrandt_detail::finalize_cloud_storage(cloud, 7.8e-5f, .001f, 1, 0, dt, invalid));
        CHECK(invalid == retained);
    }
    auto rejected = initial;
    CHECK_FALSE(tmv::milbrandt_detail::finalize_cloud_storage(0, 7.8e-5f, 0, 1, 0, dt, rejected));
    CHECK(rejected == initial);
    auto positive       = initial;
    positive[0]         = 1e-5f;
    positive[3]         = 100;
    const auto retained = positive;
    REQUIRE(tmv::milbrandt_detail::finalize_cloud_storage(0, 7.8e-5f, .001f, 1, 0, dt, positive));
    CHECK(positive == retained);
    auto overflowing = initial;
    overflowing[0]   = std::numeric_limits<float>::max();
    const auto saved = overflowing;
    CHECK_FALSE(
        tmv::milbrandt_detail::finalize_cloud_storage(0, 1, 1, 1, 0, std::numeric_limits<double>::max(), overflowing));
    CHECK(overflowing == saved);
}

TEST_CASE("Milbrandt complete cloud removal cannot retain liquid without particles",
          "[physics][microphysics][milbrandt][storage][population-support]")
{
    for (double dt : {.5, .2, 1.0 / 6.0})
    {
        constexpr float cloud = .001f, number = 1000;
        float           mass_rate = static_cast<float>(-double(cloud) / dt);
        while (cloud + dt * mass_rate <= 0)
            mass_rate = std::nextafter(mass_rate, std::numeric_limits<float>::infinity());
        float number_rate = static_cast<float>(-double(number) / dt);
        if (number + dt * number_rate > 0)
            number_rate = std::nextafter(number_rate, -std::numeric_limits<float>::infinity());
        std::array<float, 4> rates{mass_rate, 0, 0, number_rate};
        const auto           before = rates;
        REQUIRE(tmv::milbrandt_detail::finalize_cloud_storage(cloud, cloud, .01f, 1, number, dt, rates));
        const double remaining = cloud + dt * rates[0];
        CHECK(remaining >= 0);
        CHECK((remaining == 0 || number + dt * rates[3] > 0));
        CHECK(std::abs(double(rates[0]) + rates[1] - before[0]) <=
              4 * std::numeric_limits<float>::epsilon() * std::abs(double(before[0])));
        CHECK(rates[2] ==
              Approx((double(rates[0]) - before[0]) * microphysics_constants::L_v / microphysics_constants::cp)
                  .margin(1e-14));
        CHECK(std::abs(dt * (double(rates[3]) - before[3])) <= 8 * std::numeric_limits<float>::epsilon() * number);
    }
    std::array<float, 4> unresolved{-2.5986888592127853e-27f, 0, 0, 0};
    const auto           unresolved_before = unresolved;
    REQUIRE(tmv::milbrandt_detail::finalize_cloud_storage(
        4.3311482591821375e-28f, 1.6049416681536841e-35f, .005f, 1, 0, 1.0 / 6.0, unresolved));
    CHECK(unresolved == unresolved_before);
    std::array<float, 4> inconsistent{0, 0, 0, -2000};
    const auto           before = inconsistent;
    CHECK_FALSE(tmv::milbrandt_detail::finalize_cloud_storage(.001f, .001f, .01f, 1, 1000, .5, inconsistent));
    CHECK(inconsistent == before);
}

TEST_CASE("Thompson complete evaporation returns its entire droplet population",
          "[physics][microphysics][thompson][evaporation]")
{
    setup_grid();
    setup_warm_state();
    qv.resize(NR, NTH, NZ, .001f);
    qc.resize(NR, NTH, NZ, 1e-6f);
    qr.resize(NR, NTH, NZ, 0.0f);
    auto scheme = create_microphysics_scheme("thompson");
    scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
    const auto before = scheme->portable_state();
    Tendencies rates;
    compute(*scheme, 1.0, rates);
    const auto after = scheme->portable_state();
    const int i = 4, j = 4, k = 4;
    CHECK(qc(i,j,k) + rates.dqc(i,j,k) == Approx(0).margin(1e-13));
    CHECK(before.fields[0](i,j,k) + after.fields[5](i,j,k) <= 8 * std::numeric_limits<float>::epsilon() * before.fields[0](i,j,k));
    CHECK(after.fields[5](i,j,k) == -after.fields[8](i,j,k));
}

TEST_CASE("Thompson stochastic hook preserves evaporation mass and particle support",
          "[physics][microphysics][thompson][stochastic]")
{
    auto scheme = create_microphysics_scheme("thompson");
    auto state = scheme->portable_state();
    const std::array<float, 5> initial{1048576, 64, 32, 0, 16};
    const std::array<float, 5> rates{-1048576, -32, -16, 1048576, -4};
    for (std::size_t f = 0; f < 5; ++f)
    {
        state.fields[f].resize(8, 8, 8, initial[f]);
        state.fields[f + 5].resize(8, 8, 8, rates[f]);
    }
    std::string error;
    REQUIRE(scheme->restore_portable_state(state, error));
    const Field3D density(8, 8, 8, 1);
    const float requested = std::nextafter(1.0f, 0.0f);
    std::vector<float> factors(64, requested);
    REQUIRE(scheme->apply_stochastic_column_multipliers(1, density, factors, error));
    CHECK(factors[4 * 8 + 4] == requested);
    const auto scaled = scheme->portable_state();
    for (std::size_t f = 0; f < 5; ++f)
        CHECK(scaled.fields[f + 5](4, 4, 4) == float(rates[f] * requested));
    REQUIRE(scheme->validate_prognostic_update(1, error));
    scheme->commit_prognostic_update(1);
    const auto final = scheme->portable_state();
    const double cloud = std::ldexp(1.0, -15);
    const double retained_cloud = cloud + double(float(-float(cloud) * requested));
    CHECK(retained_cloud == std::ldexp(1.0, -39));
    CHECK(final.fields[0](4, 4, 4) == .0625f);
    CHECK(final.fields[0](4, 4, 4) / retained_cloud == initial[0] / cloud);
    CHECK(double(final.fields[0](4, 4, 4)) + final.fields[3](4, 4, 4) == double(initial[0]));
}

TEST_CASE("Thompson diagnoses bounded rain before evaluating breakup",
          "[physics][microphysics][thompson][rain-entry]")
{
    for (float number : {0.0f, 1e-20f, 1e-11f})
    {
        CAPTURE(number);
        setup_grid();
        setup_warm_state();
        theta.fill(280);
        rho.fill(1);
        qv.resize(NR, NTH, NZ, float(tmv::thermodynamics::saturation_mixing_ratio_water(280, 100000)));
        qc.resize(NR, NTH, NZ, 0);
        qr.resize(NR, NTH, NZ, 1.3387687788486869e-12f);
        auto scheme = create_microphysics_scheme("thompson");
        scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
        auto state = scheme->portable_state();
        state.fields[1].fill(number);
        std::string error;
        REQUIRE(scheme->restore_portable_state(state, error));
        Tendencies rates;
        REQUIRE_NOTHROW(compute(*scheme, .25, rates));
        REQUIRE(scheme->validate_prognostic_update(.25, error));
        scheme->commit_prognostic_update(.25);
        const auto result = scheme->portable_state();
        const double rain = double(qr(4, 4, 4)) + .25 * rates.dqr(4, 4, 4);
        const double particles = result.fields[1](4, 4, 4);
        CHECK(rain > 0);
        CHECK(std::isfinite(particles));
        REQUIRE(particles > 0);
        const double diameter = 3.672 / std::cbrt(3.1415926536 * 1000 * particles / rain);
        CHECK(diameter >= 37.5e-6 * (1 - 1e-6));
        CHECK(diameter <= 2.5e-3 * (1 + 1e-6));
    }
}

TEST_CASE("Thompson cloud activation consumes only retained droplets",
          "[physics][microphysics][thompson][activation][conservation]")
{
    setup_grid();
    setup_warm_state();
    theta.fill(305.29608154296875f);
    p.fill(81549.5078125f);
    rho.fill(.9865851402282715f);
    qv.resize(NR, NTH, NZ, .013157349079847336f);
    qc.resize(NR, NTH, NZ, 4.304698542068053e-10f);
    qr.resize(NR, NTH, NZ, 0);
    Field3D velocity(NR, NTH, NZ, 2.137484312057495f);
    auto    scheme = create_microphysics_scheme("thompson");
    scheme->set_vertical_velocity_field(velocity, false);
    scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
    auto initial = scheme->portable_state();
    initial.fields[0].fill(111067.1015625f);
    initial.fields[3].fill(209394096.0f);
    std::string error;
    REQUIRE(scheme->restore_portable_state(initial, error));
    Tendencies rates;
    compute(*scheme, .25, rates);
    const auto   after        = scheme->portable_state();
    const double droplet_rate = after.fields[5](4, 4, 4);
    const double aerosol_rate = after.fields[8](4, 4, 4);
    REQUIRE(rates.dqc(4, 4, 4) > 0);
    REQUIRE(rates.dqr(4, 4, 4) == 0);
    REQUIRE(droplet_rate > 0);
    CHECK(std::abs(droplet_rate + aerosol_rate) <= 8 * std::numeric_limits<float>::epsilon() * std::abs(droplet_rate));
    REQUIRE(scheme->validate_prognostic_update(.25, error));
    scheme->commit_prognostic_update(.25);
    const auto   final            = scheme->portable_state();
    const double before_particles = double(initial.fields[0](4, 4, 4)) + initial.fields[3](4, 4, 4);
    const double after_particles  = double(final.fields[0](4, 4, 4)) + final.fields[3](4, 4, 4);
    CHECK(std::abs(after_particles - before_particles) <= 8 * std::numeric_limits<float>::epsilon() * before_particles);
}
