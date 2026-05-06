/**
 * @file test_file_sounding.cpp
 * @brief Verifies the column-build helper used by FileSoundingSource.
 *
 * Tests target sounding_from_data() rather than FileSoundingSource::build()
 * so we don't need a fixture file on disk. The helper takes an already-
 * interpolated SoundingData and turns it into a Sounding with hydrostatically
 * re-integrated pressure, theta/rho/qv consistency, and Cartesian winds.
 *
 * Coverage:
 *   - Hydrostatic re-integration starts from the file's surface pressure
 *     when present; falls back to p0 otherwise.
 *   - theta = T * (p0/p)^(R_d/cp), rho = p / (R_d T) at every level.
 *   - qv prefers mixing_ratio_kgkg, falls back to dewpoint_k, capped at
 *     0.95 * qvsat.
 *   - Wind extraction uses meteorological convention
 *     (u_x = -speed * sin(dir_rad), u_y = -speed * cos(dir_rad)).
 *   - Missing T column throws.
 *   - Missing winds with require_winds=true throws; with require_winds=false
 *     leaves Sounding::u_ms / v_ms empty (caller falls back to hodograph).
 */

#include "catch2/catch.hpp"
#include "core/infra/physical_constants.hpp"
#include "init/sounding/file_sounding.hpp"

#include <cmath>
#include <vector>

namespace
{

std::vector<double> linspace(double lo, double hi, std::size_t n)
{
    std::vector<double> out(n);
    if (n == 1)
    {
        out[0] = lo;
        return out;
    }
    const double step = (hi - lo) / static_cast<double>(n - 1);
    for (std::size_t i = 0; i < n; ++i)
    {
        out[i] = lo + step * static_cast<double>(i);
    }
    return out;
}

SoundingData make_synthetic_isothermal_data(const std::vector<double>& z_m,
                                            double T_const_k,
                                            double surface_p_hpa,
                                            double mixing_ratio_kgkg,
                                            double wind_speed_ms,
                                            double wind_dir_deg)
{
    SoundingData d;
    const std::size_t n = z_m.size();
    d.height_m = z_m;
    d.temperature_k.assign(n, T_const_k);
    d.pressure_hpa.assign(n, surface_p_hpa);  // unused except for [0]
    d.mixing_ratio_kgkg.assign(n, mixing_ratio_kgkg);
    d.wind_speed_ms.assign(n, wind_speed_ms);
    d.wind_direction_deg.assign(n, wind_dir_deg);
    return d;
}

}  // namespace

TEST_CASE("sounding_from_data builds a self-consistent column",
          "[init][sounding][file]")
{
    constexpr std::size_t NZ = 32;
    constexpr double DZ = 250.0;
    const auto z = linspace(0.0, DZ * (NZ - 1), NZ);
    const auto data = make_synthetic_isothermal_data(
        z, /*T*/ 280.0, /*p_sfc_hpa*/ 1000.0, /*qv*/ 0.008,
        /*speed*/ 10.0, /*dir*/ 270.0);

    const auto s = tmv::init::sounding_from_data(data, z, DZ, /*require_winds=*/true);

    REQUIRE(s.size() == NZ);
    REQUIRE_NOTHROW(s.verify_self_consistent());

    SECTION("surface boundary uses file pressure when valid")
    {
        REQUIRE(s.p_pa[0] == Approx(100000.0).margin(1.0e-9));
        REQUIRE(s.T_k[0] == Approx(280.0).margin(1.0e-12));
    }

    SECTION("pressure decreases monotonically (hydrostatic re-integration)")
    {
        for (std::size_t k = 1; k < NZ; ++k)
        {
            REQUIRE(s.p_pa[k] < s.p_pa[k - 1]);
        }
    }

    SECTION("isothermal hydrostatic profile matches analytical p(z)")
    {
        // For constant T, p(z) = p0 * exp(-g*z / (R_d T)) within float
        // precision of the integration. The numeric integration uses
        // T_avg = T at every step, so the analytical exponential is exact.
        for (std::size_t k = 0; k < NZ; ++k)
        {
            const double p_expected = 100000.0 * std::exp(-g * z[k] / (R_d * 280.0));
            REQUIRE(s.p_pa[k] == Approx(p_expected).epsilon(1.0e-12));
        }
    }

    SECTION("theta = T * (p0/p)^kappa")
    {
        const double kappa = R_d / cp;
        for (std::size_t k = 0; k < NZ; ++k)
        {
            const double th = 280.0 * std::pow(p0 / s.p_pa[k], kappa);
            REQUIRE(s.theta_k[k] == Approx(th).epsilon(1.0e-12));
        }
    }

    SECTION("winds: speed=10 m/s from 270 deg = westerly, u_x=+10, u_y=0")
    {
        REQUIRE(s.has_winds());
        // Direction 270 = wind FROM the west, blowing TOWARD the east (+x).
        // u_x = -10 * sin(270 deg) = -10 * -1 = +10
        // u_y = -10 * cos(270 deg) = -10 *  0 = 0
        for (std::size_t k = 0; k < NZ; ++k)
        {
            REQUIRE(s.u_ms[k] == Approx(10.0).margin(1.0e-12));
            REQUIRE(s.v_ms[k] == Approx(0.0).margin(1.0e-12));
        }
    }
}

TEST_CASE("sounding_from_data: northerly direction = 0 deg gives v=-speed",
          "[init][sounding][file]")
{
    // Direction 0 = wind from the north, blowing TOWARD the south (-y).
    // u_x = -speed * sin(0)   = 0
    // u_y = -speed * cos(0)   = -speed
    const std::vector<double> z = {0.0, 250.0, 500.0};
    const auto data = make_synthetic_isothermal_data(z, 280.0, 1000.0, 0.005, 7.5, 0.0);
    const auto s = tmv::init::sounding_from_data(data, z, 250.0, true);
    REQUIRE(s.has_winds());
    for (std::size_t k = 0; k < z.size(); ++k)
    {
        REQUIRE(s.u_ms[k] == Approx(0.0).margin(1.0e-12));
        REQUIRE(s.v_ms[k] == Approx(-7.5).margin(1.0e-12));
    }
}

TEST_CASE("sounding_from_data: missing winds with require_winds=false",
          "[init][sounding][file]")
{
    const std::vector<double> z = {0.0, 250.0, 500.0};
    SoundingData d;
    d.height_m = z;
    d.temperature_k = {290.0, 288.0, 286.0};
    d.pressure_hpa = {1010.0, 980.0, 950.0};
    d.mixing_ratio_kgkg = {0.012, 0.010, 0.008};
    // wind_speed / wind_direction intentionally absent

    const auto s = tmv::init::sounding_from_data(d, z, 250.0, /*require_winds=*/false);
    REQUIRE(s.has_winds() == false);
    REQUIRE(s.u_ms.empty());
    REQUIRE(s.v_ms.empty());
    REQUIRE_NOTHROW(s.verify_self_consistent());
}

TEST_CASE("sounding_from_data: missing winds with require_winds=true throws",
          "[init][sounding][file]")
{
    const std::vector<double> z = {0.0, 250.0};
    SoundingData d;
    d.height_m = z;
    d.temperature_k = {290.0, 288.0};
    d.pressure_hpa = {1010.0, 980.0};
    d.mixing_ratio_kgkg = {0.012, 0.010};

    REQUIRE_THROWS_AS(
        tmv::init::sounding_from_data(d, z, 250.0, /*require_winds=*/true),
        std::invalid_argument);
}

TEST_CASE("sounding_from_data: missing temperature column throws",
          "[init][sounding][file]")
{
    const std::vector<double> z = {0.0, 250.0};
    SoundingData d;
    d.height_m = z;
    // temperature_k missing
    d.pressure_hpa = {1010.0, 980.0};
    d.mixing_ratio_kgkg = {0.012, 0.010};

    REQUIRE_THROWS_AS(
        tmv::init::sounding_from_data(d, z, 250.0, /*require_winds=*/false),
        std::invalid_argument);
}

TEST_CASE("sounding_from_data: dewpoint fallback when mixing ratio missing",
          "[init][sounding][file]")
{
    const std::vector<double> z = {0.0, 500.0};
    SoundingData d;
    d.height_m = z;
    d.temperature_k = {290.0, 286.0};
    d.pressure_hpa = {1010.0, 950.0};
    // No mixing_ratio_kgkg, but dewpoint_k present (T_d < T to avoid super)
    d.dewpoint_k = {285.0, 281.0};
    d.wind_speed_ms = {5.0, 6.0};
    d.wind_direction_deg = {180.0, 200.0};

    const auto s = tmv::init::sounding_from_data(d, z, 500.0, true);
    REQUIRE(s.size() == 2);
    REQUIRE(s.qv_kgkg[0] > 0.0);
    REQUIRE(s.qv_kgkg[1] > 0.0);
    REQUIRE_NOTHROW(s.verify_self_consistent());
}

TEST_CASE("sounding_from_data: surface p falls back to p0 when file is invalid",
          "[init][sounding][file]")
{
    const std::vector<double> z = {0.0, 500.0};
    SoundingData d;
    d.height_m = z;
    d.temperature_k = {290.0, 286.0};
    // pressure_hpa intentionally absent — should fall back to p0=1e5 Pa
    d.mixing_ratio_kgkg = {0.012, 0.010};
    d.wind_speed_ms = {5.0, 6.0};
    d.wind_direction_deg = {180.0, 200.0};

    const auto s = tmv::init::sounding_from_data(d, z, 500.0, true);
    REQUIRE(s.p_pa[0] == Approx(p0).margin(1.0e-9));
}

TEST_CASE("sounding_from_data: qv is bounded by 0.95 * qvsat",
          "[init][sounding][file]")
{
    // Cold column with very high mixing ratio — saturation cap must clamp.
    const std::vector<double> z = {0.0, 500.0, 1000.0};
    SoundingData d;
    d.height_m = z;
    d.temperature_k = {250.0, 248.0, 246.0};        // well below freezing
    d.pressure_hpa = {800.0, 760.0, 720.0};
    d.mixing_ratio_kgkg = {0.05, 0.05, 0.05};       // physically absurd
    d.wind_speed_ms = {0.0, 0.0, 0.0};
    d.wind_direction_deg = {0.0, 0.0, 0.0};

    const auto s = tmv::init::sounding_from_data(d, z, 500.0, true);
    REQUIRE_NOTHROW(s.verify_self_consistent());
    for (std::size_t k = 0; k < z.size(); ++k)
    {
        // qv must be far below 0.05 — the saturation cap rules.
        REQUIRE(s.qv_kgkg[k] < 0.01);
        REQUIRE(s.qv_kgkg[k] >= 0.0);
    }
}

TEST_CASE("FileSoundingSource constructor rejects empty path / scheme",
          "[init][sounding][file]")
{
    tmv::init::FileSoundingParams p;
    p.scheme_id = "sharpy";
    REQUIRE_THROWS_AS(tmv::init::FileSoundingSource(p), std::invalid_argument);

    p.file_path = "/tmp/foo";
    p.scheme_id = "";
    REQUIRE_THROWS_AS(tmv::init::FileSoundingSource(p), std::invalid_argument);

    p.scheme_id = "none";
    REQUIRE_THROWS_AS(tmv::init::FileSoundingSource(p), std::invalid_argument);
}

TEST_CASE("FileSoundingSource describe() reports scheme_id",
          "[init][sounding][file]")
{
    tmv::init::FileSoundingParams p;
    p.file_path = "/tmp/x.h5";
    p.scheme_id = "sharpy";
    tmv::init::FileSoundingSource src(p);
    REQUIRE(src.describe() == "file/sharpy");
}
