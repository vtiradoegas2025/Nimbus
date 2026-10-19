/**
 * @file test_kessler_reference_column.cpp
 * @brief Independent full-column qualification for Kessler warm rain.
 *
 * The oracle below deliberately does not call Nimbus thermodynamic, fall-speed,
 * or Kessler process helpers.  It is a direct transcription of the published
 * Kessler/COMMAS equations and Nimbus's documented finite-volume sedimentation
 * contract using literal constants.  This separation is important: the test is
 * intended to catch a common-mode change in production code, not merely prove
 * that two production helpers agree with each other.
 */

#include "catch2/catch.hpp"
#include "core/runtime/simulation.hpp"
#include "microphysics/schemes/kessler/kessler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace
{

constexpr int kLevels = 6;
constexpr double kDt = 12.0;

// Independent reference constants.  Do not replace these with production
// constants: literal duplication is intentional for this qualification.
constexpr double kRd = 287.0;
constexpr double kRv = 461.5;
constexpr double kCp = 1004.0;
constexpr double kLv = 2.5e6;
constexpr double kP0 = 100000.0;
constexpr double kT0 = 273.15;
constexpr double kQc0 = 1.0e-3;
constexpr double kCauto = 1.0e-3;
constexpr double kCaccr = 2.2;
constexpr double kRhoReference = 1.2;
constexpr double kFallA = 36.34;
constexpr double kFallB = 0.1346;
constexpr double kFallSpeedMax = 20.0;

struct ColumnInput
{
    std::array<float, kLevels> pressure_pa{};
    std::array<float, kLevels> theta_k{};
    std::array<float, kLevels> rho_d_kgm3{};
    std::array<float, kLevels> qv{};
    std::array<float, kLevels> qc{};
    std::array<float, kLevels> qr{};
    std::array<double, kLevels> dz_m{};
};

struct ColumnTendencies
{
    std::array<float, kLevels> theta{};
    std::array<float, kLevels> qv{};
    std::array<float, kLevels> qc{};
    std::array<float, kLevels> qr{};
    double surface_rain_sink_rate = 0.0;
};

double reference_saturation_mixing_ratio(double temperature_k,
                                         double pressure_pa)
{
    const double temperature = std::clamp(temperature_k, 150.0, 400.0);
    const double pressure = std::clamp(pressure_pa, 100.0, 200000.0);
    const double tc = temperature - kT0;
    const double es = 611.21 * std::exp(
        (18.678 - tc / 234.5) * tc / (257.14 + tc));
    return std::clamp(0.62198 * es / std::max(pressure - es, 1.0),
                      0.0, 0.2);
}

double reference_saturation_adjustment(double temperature_k,
                                       double pressure_pa,
                                       double& vapor,
                                       double& cloud)
{
    double temperature = std::clamp(temperature_k, 150.0, 400.0);
    vapor = std::max(0.0, vapor);
    cloud = std::max(0.0, cloud);
    const double saturation = reference_saturation_mixing_ratio(
        temperature, pressure_pa);
    constexpr int iterations = 52;
    constexpr double latent_temperature_factor = kLv / kCp;

    if (vapor > saturation)
    {
        const double maximum = std::min(
            vapor,
            std::max(
                0.0,
                (400.0 - temperature)
                    / latent_temperature_factor));
        const auto residual =
            [&](double condensed)
            {
                return vapor - condensed
                    - reference_saturation_mixing_ratio(
                        temperature
                            + latent_temperature_factor * condensed,
                        pressure_pa);
            };
        double condensed = maximum;
        if (residual(maximum) <= 0.0)
        {
            double lower = 0.0;
            double upper = maximum;
            for (int iteration = 0;
                 iteration < iterations; ++iteration)
            {
                const double midpoint = 0.5 * (lower + upper);
                if (residual(midpoint) > 0.0)
                    lower = midpoint;
                else
                    upper = midpoint;
            }
            condensed = 0.5 * (lower + upper);
        }
        vapor -= condensed;
        cloud += condensed;
        temperature += latent_temperature_factor * condensed;
    }
    else if (vapor < saturation && cloud > 0.0)
    {
        const double maximum = std::min(
            cloud,
            std::max(
                0.0,
                (temperature - 150.0)
                    / latent_temperature_factor));
        const auto residual =
            [&](double evaporated)
            {
                return vapor + evaporated
                    - reference_saturation_mixing_ratio(
                        temperature
                            - latent_temperature_factor * evaporated,
                        pressure_pa);
            };
        double evaporated = maximum;
        if (residual(maximum) >= 0.0)
        {
            double lower = 0.0;
            double upper = maximum;
            for (int iteration = 0;
                 iteration < iterations; ++iteration)
            {
                const double midpoint = 0.5 * (lower + upper);
                if (residual(midpoint) < 0.0)
                    lower = midpoint;
                else
                    upper = midpoint;
            }
            evaporated = 0.5 * (lower + upper);
        }
        vapor += evaporated;
        cloud -= evaporated;
        temperature -= latent_temperature_factor * evaporated;
    }
    return temperature;
}

float reference_temperature(float theta_k, float pressure_pa)
{
    return static_cast<float>(
        static_cast<double>(theta_k)
        * std::pow(static_cast<double>(pressure_pa) / kP0, kRd / kCp));
}

double reference_fall_speed(float rain, float rho_d, double dz_m)
{
    if (rain < 1.0e-9f) return 0.0;
    const double rho = static_cast<double>(rho_d);
    const double base = kFallA * std::pow(
        1.0e-3 * rho * static_cast<double>(rain), kFallB);
    const double altitude_corrected = base * std::sqrt(kRhoReference / rho);
    return std::min({altitude_corrected, kFallSpeedMax,
                     0.9 * dz_m / kDt});
}

ColumnTendencies reference_column(const ColumnInput& input)
{
    ColumnTendencies out;
    std::array<float, kLevels> adjusted_qv{};
    std::array<float, kLevels> adjusted_qc{};
    std::array<float, kLevels> temperature{};

    // Nonlinear liquid saturation adjustment, including the exactly paired
    // latent temperature change consumed by all subsequent processes.
    for (int k = 0; k < kLevels; ++k)
    {
        temperature[k] = reference_temperature(
            input.theta_k[k], input.pressure_pa[k]);
        double qv_value = static_cast<double>(input.qv[k]);
        double qc_value = static_cast<double>(input.qc[k]);
        temperature[k] = static_cast<float>(
            reference_saturation_adjustment(
                static_cast<double>(temperature[k]),
                static_cast<double>(input.pressure_pa[k]),
                qv_value, qc_value));
        adjusted_qv[k] = static_cast<float>(std::max(qv_value, 0.0));
        adjusted_qc[k] = static_cast<float>(std::max(qc_value, 0.0));

        const double adjustment_rate =
            (static_cast<double>(adjusted_qc[k])
             - static_cast<double>(input.qc[k])) / kDt;
        out.qc[k] += static_cast<float>(adjustment_rate);
        out.qv[k] -= static_cast<float>(adjustment_rate);
        out.theta[k] += static_cast<float>((kLv / kCp) * adjustment_rate);
    }

    // Finite-step implicit accretion/autoconversion followed by the published
    // density-dependent Kessler diffusion/ventilation evaporation kernel.
    for (int k = 0; k < kLevels; ++k)
    {
        const float cloud = adjusted_qc[k];
        const float rain = input.qr[k];
        float production_rate = 0.0f;
        if (cloud > 0.0f)
        {
            const double factor = 1.0 /
                (1.0 + kCaccr * kDt
                 * std::pow(std::max(0.0f, rain), 0.875));
            const double amount =
                static_cast<double>(cloud) * (1.0 - factor)
                + factor * kCauto * kDt
                  * std::max(static_cast<double>(cloud) - kQc0, 0.0);
            production_rate = static_cast<float>(
                std::clamp(amount, 0.0, static_cast<double>(cloud)) / kDt);
            out.qc[k] -= production_rate;
            out.qr[k] += production_rate;
        }

        if (rain <= 0.0f && production_rate <= 0.0f) continue;
        const double pressure = static_cast<double>(input.pressure_pa[k]);
        const double rho_d = static_cast<double>(input.rho_d_kgm3[k]);
        const double vapor = static_cast<double>(adjusted_qv[k]);
        const double qvs = reference_saturation_mixing_ratio(
            static_cast<double>(temperature[k]), pressure);
        if (!(qvs > 0.0 && vapor < qvs && rho_d > 0.0 && pressure > 0.0))
            continue;

        const double available_rain = static_cast<double>(rain)
            + static_cast<double>(production_rate) * kDt;
        const double rcgs = 1.0e-3 * rho_d;
        const double rain_content = rcgs * std::max(available_rain, 0.0);
        const double ventilation =
            (1.6 + 124.9 * std::pow(rain_content, 0.2046))
            * std::pow(rain_content, 0.525);
        const double resistance = 2.55e8 / (pressure * qvs) + 5.4e5;
        const double raw_rate = (ventilation / resistance)
            * ((qvs - vapor) / (rcgs * qvs));
        const double gamma = kLv * kLv * qvs
            / (kCp * kRv
               * std::max(static_cast<double>(temperature[k])
                          * static_cast<double>(temperature[k]), 1.0));
        const double saturation_limit = (qvs - vapor) / (1.0 + gamma);
        const double evaporation_rate = std::clamp(
            raw_rate, 0.0,
            std::min(saturation_limit / kDt, available_rain / kDt));
        out.qv[k] += static_cast<float>(evaporation_rate);
        out.qr[k] -= static_cast<float>(evaporation_rate);
        out.theta[k] -= static_cast<float>((kLv / kCp) * evaporation_rate);
    }

    // Conservative donor-cell sedimentation over nonuniform layers.  The
    // transported currency is rho_d*q*dz, so the receiving mixing-ratio rate
    // carries both density and layer-thickness ratios.
    for (int k = 0; k < kLevels; ++k)
    {
        if (input.qr[k] <= 0.0f) continue;
        const double speed = reference_fall_speed(
            input.qr[k], input.rho_d_kgm3[k], input.dz_m[k]);
        const float flux = static_cast<float>(speed) * input.qr[k];
        out.qr[k] -= static_cast<float>(
            static_cast<double>(flux) / input.dz_m[k]);
        if (k > 0)
        {
            const double receiver_weight =
                static_cast<double>(input.rho_d_kgm3[k])
                / static_cast<double>(input.rho_d_kgm3[k - 1]);
            out.qr[k - 1] += static_cast<float>(
                receiver_weight * static_cast<double>(flux)
                / input.dz_m[k - 1]);
        }
    }

    const double surface_speed = reference_fall_speed(
        input.qr[0], input.rho_d_kgm3[0], input.dz_m[0]);
    out.surface_rain_sink_rate =
        static_cast<double>(static_cast<float>(surface_speed) * input.qr[0])
        / input.dz_m[0];

    // Nimbus prognoses potential temperature.  Process heat above is in K/s.
    for (int k = 0; k < kLevels; ++k)
    {
        out.theta[k] = static_cast<float>(
            static_cast<double>(out.theta[k])
            * std::pow(kP0 / static_cast<double>(input.pressure_pa[k]),
                       kRd / kCp));
    }
    return out;
}

ColumnInput qualification_column()
{
    ColumnInput input;
    input.pressure_pa = {98000.0f, 91000.0f, 84000.0f,
                         77000.0f, 70000.0f, 63000.0f};
    const std::array<double, kLevels> temperature_k =
        {298.0, 295.0, 292.0, 289.0, 286.0, 283.0};
    for (int k = 0; k < kLevels; ++k)
    {
        input.theta_k[k] = static_cast<float>(
            temperature_k[k]
            * std::pow(kP0 / static_cast<double>(input.pressure_pa[k]),
                       kRd / kCp));
    }
    input.rho_d_kgm3 = {1.16f, 1.08f, 1.00f, 0.92f, 0.84f, 0.76f};
    input.qv = {0.0240f, 0.0080f, 0.0200f,
                0.0060f, 0.0100f, 0.0030f};
    input.qc = {0.0003f, 0.0f, 0.0015f,
                0.0020f, 0.0025f, 0.0f};
    input.qr = {0.0005f, 0.0012f, 0.0002f,
                0.0f, 0.0008f, 0.0018f};
    input.dz_m = {170.0, 230.0, 310.0, 420.0, 560.0, 740.0};
    return input;
}

} // namespace

TEST_CASE("Kessler full nonuniform column matches an independent reference",
          "[qualification][physics][microphysics][kessler][reference-column][SCI-106]")
{
    const ColumnInput input = qualification_column();
    const ColumnTendencies expected = reference_column(input);

    NR = 1;
    NTH = 1;
    NZ = kLevels;
    Field3D pressure(NR, NTH, NZ);
    Field3D potential_temperature(NR, NTH, NZ);
    Field3D dry_density(NR, NTH, NZ);
    Field3D vapor(NR, NTH, NZ);
    Field3D cloud(NR, NTH, NZ);
    Field3D rain(NR, NTH, NZ);
    Field3D zero(NR, NTH, NZ, 0.0f);
    for (int k = 0; k < kLevels; ++k)
    {
        pressure(0, 0, k) = input.pressure_pa[k];
        potential_temperature(0, 0, k) = input.theta_k[k];
        dry_density(0, 0, k) = input.rho_d_kgm3[k];
        vapor(0, 0, k) = input.qv[k];
        cloud(0, 0, k) = input.qc[k];
        rain(0, 0, k) = input.qr[k];
    }

    tmv::KesslerScheme scheme;
    scheme.set_physical_cell_domain(
        NR, NTH, NZ, PhysicalCellDomain{0, NR, 0, NTH, 0, NZ});
    std::vector<double> spacing(input.dz_m.begin(), input.dz_m.end());
    scheme.set_vertical_spacing_m(NR, NTH, NZ, std::move(spacing));

    Field3D dtheta, dqv, dqc, dqr, dqi, dqs, dqg, dqh;
    scheme.compute_tendencies(
        pressure, potential_temperature, dry_density,
        vapor, cloud, rain, zero, zero, zero, zero, kDt,
        dtheta, dqv, dqc, dqr, dqi, dqs, dqg, dqh);

    bool saw_condensation = false;
    bool saw_evaporation = false;
    bool saw_cloud_to_rain = false;
    for (int k = 0; k < kLevels; ++k)
    {
        INFO("level=" << k);
        REQUIRE(static_cast<double>(dqv(0, 0, k)) ==
                Approx(expected.qv[k]).margin(3.0e-10));
        REQUIRE(static_cast<double>(dqc(0, 0, k)) ==
                Approx(expected.qc[k]).margin(3.0e-10));
        REQUIRE(static_cast<double>(dqr(0, 0, k)) ==
                Approx(expected.qr[k]).margin(3.0e-10));
        REQUIRE(static_cast<double>(dtheta(0, 0, k)) ==
                Approx(expected.theta[k]).margin(8.0e-7));
        REQUIRE(dqi(0, 0, k) == 0.0f);
        REQUIRE(dqs(0, 0, k) == 0.0f);
        REQUIRE(dqg(0, 0, k) == 0.0f);
        REQUIRE(dqh(0, 0, k) == 0.0f);

        saw_condensation = saw_condensation || dqv(0, 0, k) < 0.0f;
        saw_evaporation = saw_evaporation || dqv(0, 0, k) > 0.0f;
        saw_cloud_to_rain = saw_cloud_to_rain || dqc(0, 0, k) < 0.0f;
    }
    REQUIRE(saw_condensation);
    REQUIRE(saw_evaporation);
    REQUIRE(saw_cloud_to_rain);

    const double actual_sink = scheme.bottom_sedimentation_sink_rate(
        dry_density, rain, zero, zero, zero, 0, 0, kDt);
    REQUIRE(actual_sink == Approx(expected.surface_rain_sink_rate)
                           .epsilon(2.0e-7));

    // Column water changes only through the independently predicted surface
    // rain flux.  This also verifies the rho_d*dz receiver weights.
    double water_rate = 0.0;
    double scale = 0.0;
    for (int k = 0; k < kLevels; ++k)
    {
        const double cell_mass = static_cast<double>(input.rho_d_kgm3[k])
            * input.dz_m[k];
        const double local_rate = static_cast<double>(dqv(0, 0, k))
            + static_cast<double>(dqc(0, 0, k))
            + static_cast<double>(dqr(0, 0, k));
        water_rate += cell_mass * local_rate;
        scale += cell_mass * (std::abs(static_cast<double>(dqv(0, 0, k)))
            + std::abs(static_cast<double>(dqc(0, 0, k)))
            + std::abs(static_cast<double>(dqr(0, 0, k))));
    }
    const double surface_flux = static_cast<double>(input.rho_d_kgm3[0])
        * input.dz_m[0] * actual_sink;
    REQUIRE(std::abs(water_rate + surface_flux)
            / std::max(scale + std::abs(surface_flux), 1.0e-30) < 2.0e-6);

    // Latent heat is pointwise paired with vapor phase change; conversion to
    // potential temperature must not alter that temperature-space identity.
    for (int k = 0; k < kLevels; ++k)
    {
        const double exner_inverse = std::pow(
            kP0 / static_cast<double>(input.pressure_pa[k]), kRd / kCp);
        const double temperature_rate =
            static_cast<double>(dtheta(0, 0, k)) / exner_inverse;
        const double latent_reference =
            -(kLv / kCp) * static_cast<double>(dqv(0, 0, k));
        INFO("latent level=" << k);
        REQUIRE(temperature_rate == Approx(latent_reference).margin(1.0e-6));
    }
}
