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
 *   - qv prefers mixing_ratio_kgkg and falls back to dewpoint_k; a supplied
 *     supersaturation is preserved for microphysics rather than hidden here.
 *   - Wind extraction uses meteorological convention
 *     (u_x = -speed * sin(dir_rad), u_y = -speed * cos(dir_rad)).
 *   - Missing T column throws.
 *   - Missing winds with require_winds=true throws; with require_winds=false
 *     leaves Sounding::u_ms / v_ms empty (caller falls back to hodograph).
 */

#include "catch2/catch.hpp"
#include "core/infra/physical_constants.hpp"
#include "init/sounding/diagnostics.hpp"
#include "init/sounding/file_sounding.hpp"
#include "soundings/soundings.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <string>
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

    SECTION("density satisfies the dry ideal-gas equation exactly")
    {
        for (std::size_t k = 0; k < NZ; ++k)
        {
            REQUIRE(s.rho_kgm3[k]
                    == Approx(s.p_pa[k] / (R_d * s.T_k[k])).epsilon(1.0e-12));
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

TEST_CASE("sounding_from_data never discards a malformed primary-theta contract",
          "[init][sounding][file][theta][guard]")
{
    const std::vector<double> z = {0.0, 500.0};
    SoundingData d;
    d.height_m = z;
    d.temperature_k = {290.0, 288.0};
    d.pressure_hpa = {1000.0, 950.0};
    d.mixing_ratio_kgkg = {0.010, 0.008};
    d.potential_temperature_is_primary = true;

    SECTION("missing native theta")
    {
        REQUIRE_THROWS_WITH(
            tmv::init::sounding_from_data(d, z, 500.0, false),
            Catch::Matchers::Contains("marks potential temperature as primary"));
    }

    SECTION("non-physical native theta")
    {
        d.potential_temperature_k = {300.0, std::numeric_limits<double>::quiet_NaN()};
        REQUIRE_THROWS_WITH(
            tmv::init::sounding_from_data(d, z, 500.0, false),
            Catch::Matchers::Contains("incomplete or non-physical"));
    }
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

TEST_CASE("sounding_from_data: missing surface pressure uses the documented p0 boundary",
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

TEST_CASE("sounding_from_data rejects supplied invalid pressure",
          "[init][sounding][file]")
{
    const std::vector<double> z = {0.0, 500.0};
    SoundingData d;
    d.height_m = z;
    d.temperature_k = {290.0, 286.0};
    d.pressure_hpa = {-1.0, 900.0};
    d.mixing_ratio_kgkg = {0.012, 0.010};

    REQUIRE_THROWS_AS(
        tmv::init::sounding_from_data(d, z, 500.0, false),
        std::invalid_argument);
}

TEST_CASE("sounding_from_data fallback integrates surface pressure to first center",
          "[init][sounding][file][coordinates]")
{
    constexpr double DZ = 500.0;
    const std::vector<double> z = {250.0, 750.0, 1250.0};
    SoundingData d;
    d.height_m = z;
    d.temperature_k = {280.0, 280.0, 280.0};
    d.mixing_ratio_kgkg = {0.005, 0.005, 0.005};

    const auto s = tmv::init::sounding_from_data(d, z, DZ, false);
    for (std::size_t k = 0; k < z.size(); ++k)
    {
        const double expected = p0 * std::exp(-g * z[k] / (R_d * 280.0));
        REQUIRE(s.p_pa[k] == Approx(expected).epsilon(1.0e-12));
    }
    REQUIRE(s.p_pa[0] < p0);
}

TEST_CASE("sounding_from_data preserves valid supplied supersaturation",
          "[init][sounding][file]")
{
    // Condensation belongs to microphysics, not to a hidden reader clamp.
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
        REQUIRE(s.qv_kgkg[k] == Approx(0.05));
    }
}

TEST_CASE("sounding_from_data does not floor upper-level density",
          "[init][sounding][file][regression]")
{
    const std::vector<double> z = {0.0, 30000.0};
    SoundingData d;
    d.height_m = z;
    d.temperature_k = {220.0, 220.0};
    d.pressure_hpa = {1000.0, 10.0};
    d.mixing_ratio_kgkg = {0.004, 0.004};

    const auto s = tmv::init::sounding_from_data(d, z, 30000.0, false);
    REQUIRE(s.rho_kgm3[1] < 0.1);
    REQUIRE(s.rho_kgm3[1]
            == Approx(s.p_pa[1] / (R_d * s.T_k[1])).epsilon(1.0e-12));
    REQUIRE_NOTHROW(s.verify_self_consistent());
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

TEST_CASE("sounding reader runtimes are isolated", "[init][sounding][runtime]")
{
    reset_soundings();

    SoundingConfig first_config;
    first_config.scheme_id = "sharpy";
    first_config.file_path = "/tmp/first_sounding.h5";

    SoundingConfig second_config = first_config;
    second_config.file_path = "/tmp/second_sounding.h5";

    SoundingRuntimeState first;
    SoundingRuntimeState second;
    initialize_soundings(first, first_config);
    initialize_soundings(second, second_config);

    REQUIRE(is_soundings_initialized(first));
    REQUIRE(is_soundings_initialized(second));
    REQUIRE(first.scheme.get() != second.scheme.get());
    REQUIRE(get_sounding_config(first).file_path == first_config.file_path);
    REQUIRE(get_sounding_config(second).file_path == second_config.file_path);
    REQUIRE_FALSE(is_soundings_initialized());

    reset_soundings(first);
    REQUIRE_FALSE(is_soundings_initialized(first));
    REQUIRE(is_soundings_initialized(second));
}

TEST_CASE("FileSoundingSource builds directly from a real CM1 input_sounding",
          "[init][sounding][file][cm1][reference]")
{
    tmv::init::FileSoundingParams params;
    params.file_path = "tests/fixtures/soundings/cm1/HIGHCAPENEW_QUARTERCIRCLE";
    params.scheme_id = "auto";
    params.require_winds = true;
    tmv::init::FileSoundingSource source(params);

    const std::vector<double> z_m = {25.0, 75.0, 125.0, 175.0, 225.0};
    const auto sounding = source.build(z_m, 50.0);
    REQUIRE_NOTHROW(sounding.verify_self_consistent());
    REQUIRE(sounding.has_winds());
    REQUIRE(sounding.theta_k[0]
            == Approx(0.5 * (306.465 + 306.494)).margin(1.0e-12));
    REQUIRE(sounding.qv_kgkg[0]
            == Approx(0.5 * (14.7394 + 14.710) / 1000.0).margin(1.0e-12));
    REQUIRE(sounding.u_ms[0]
            == Approx(0.5 * (-15.798 - 15.388)).margin(1.0e-12));
    REQUIRE(sounding.v_ms[0]
            == Approx(0.5 * (0.0 + 0.848)).margin(1.0e-12));
}

TEST_CASE("FileSoundingSource auto-detects the retained SHARPY text profile",
          "[init][sounding][file][sharpy][reference]")
{
    tmv::init::FileSoundingParams params;
    params.file_path = "tests/init/fixtures/test_supercell.tmv";
    params.scheme_id = "auto";
    params.require_winds = true;
    tmv::init::FileSoundingSource source(params);

    const std::vector<double> z_m = {250.0, 750.0, 1250.0, 1750.0, 2250.0};
    const auto sounding = source.build(z_m, 500.0);
    REQUIRE(sounding.p_pa.size() == z_m.size());
    REQUIRE(sounding.u_ms.size() == z_m.size());
    REQUIRE(sounding.v_ms.size() == z_m.size());
    REQUIRE(sounding.qv_kgkg.size() == z_m.size());
    REQUIRE_NOTHROW(sounding.verify_self_consistent());
    REQUIRE(sounding.u_ms.front() > 0.0);
    REQUIRE(sounding.v_ms.front() > 0.0);
}

TEST_CASE("FileSoundingSource cannot retry rejected HDF5 metadata through NetCDF",
          "[init][sounding][file][sharpy][hdf5][guard]")
{
    tmv::init::FileSoundingParams params;
    params.file_path = "tests/init/fixtures/sharpy_hdf5_public_safe.h5";
    params.scheme_id = "auto";
    params.require_winds = false;
    const tmv::init::FileSoundingSource safe(params);
    const auto sounding = safe.build({250.0, 750.0}, 500.0);
    REQUIRE(sounding.p_pa.size() == 2);
    REQUIRE_NOTHROW(sounding.verify_self_consistent());

    params.file_path = "tests/init/fixtures/sharpy_hdf5_public_oversized.h5";
    const tmv::init::FileSoundingSource oversized(params);
    REQUIRE_THROWS_WITH(oversized.build({250.0, 750.0}, 500.0),
                        Catch::Matchers::Contains("1023-byte fixed metadata contract"));

    params.file_path = "tests/init/fixtures/sharpy_hdf5_numeric_oversized_100001.h5";
    const tmv::init::FileSoundingSource oversized_levels(params);
    REQUIRE_THROWS_WITH(oversized_levels.build({250.0, 750.0}, 500.0),
                        Catch::Matchers::Contains("100,000-level safety limit"));

    params.file_path = "tests/init/fixtures/sharpy_netcdf_public_safe.nc";
    params.require_winds = true;
    const tmv::init::FileSoundingSource netcdf_safe(params);
    const auto netcdf_column = netcdf_safe.build({250.0, 750.0}, 500.0);
    REQUIRE(netcdf_column.p_pa.size() == 2);
    REQUIRE(netcdf_column.u_ms.size() == 2);
    REQUIRE_NOTHROW(netcdf_column.verify_self_consistent());

    params.file_path = "tests/init/fixtures/sharpy_netcdf_numeric_oversized_100001.nc";
    const tmv::init::FileSoundingSource netcdf_oversized_levels(params);
    REQUIRE_THROWS_WITH(netcdf_oversized_levels.build({250.0, 750.0}, 500.0),
                        Catch::Matchers::Contains("100,000-level safety limit"));
}

TEST_CASE("FileSoundingSource rejects multi-value NetCDF coordinates before HDF5 fallback",
          "[init][sounding][file][sharpy][netcdf][metadata][guard]")
{
    tmv::init::FileSoundingParams params;
    params.file_path = "tests/init/fixtures/sharpy_netcdf_metadata_public_scalar.nc";
    params.scheme_id = "auto";
    params.require_winds = true;
    const tmv::init::FileSoundingSource scalar(params);
    const auto sounding = scalar.build({250.0, 750.0}, 500.0);
    REQUIRE(sounding.p_pa.size() == 2);
    REQUIRE(sounding.u_ms.size() == 2);
    REQUIRE_NOTHROW(sounding.verify_self_consistent());

    params.file_path = "tests/init/fixtures/sharpy_netcdf_metadata_public_array.nc";
    const tmv::init::FileSoundingSource array(params);
    REQUIRE_THROWS_WITH(array.build({250.0, 750.0}, 500.0),
                        Catch::Matchers::Contains("numeric metadata must be scalar"));
}

TEST_CASE("FileSoundingSource rejects multi-value NetCDF station and timestamp metadata",
          "[init][sounding][file][sharpy][netcdf][string][guard]")
{
    tmv::init::FileSoundingParams params;
    params.scheme_id = "auto";
    params.require_winds = true;
    for (const char* path : {
             "tests/init/fixtures/sharpy_netcdf_string_public_station_scalar.nc",
             "tests/init/fixtures/sharpy_netcdf_string_public_timestamp_scalar.nc"})
    {
        params.file_path = path;
        const tmv::init::FileSoundingSource source(params);
        const auto sounding = source.build({250.0, 750.0}, 500.0);
        REQUIRE(sounding.p_pa.size() == 2);
        REQUIRE(sounding.u_ms.size() == 2);
        REQUIRE_NOTHROW(sounding.verify_self_consistent());
    }
    for (const char* path : {
             "tests/init/fixtures/sharpy_netcdf_string_public_station_array.nc",
             "tests/init/fixtures/sharpy_netcdf_string_public_timestamp_array.nc"})
    {
        params.file_path = path;
        const tmv::init::FileSoundingSource source(params);
        REQUIRE_THROWS_WITH(source.build({250.0, 750.0}, 500.0),
                            Catch::Matchers::Contains("string metadata must be scalar"));
    }
}

TEST_CASE("FileSoundingSource applies reviewed NetCDF metadata byte boundary",
          "[init][sounding][file][sharpy][netcdf][bytes][guard]")
{
    tmv::init::FileSoundingParams params;
    params.scheme_id = "auto";
    params.require_winds = true;
    for (const char* path : {
             "tests/init/fixtures/sharpy_netcdf_char_public_station_1023.nc",
             "tests/init/fixtures/sharpy_netcdf_string_public_station_1023.nc",
             "tests/init/fixtures/sharpy_netcdf_string_public_timestamp_1023.nc"})
    {
        params.file_path = path;
        const tmv::init::FileSoundingSource source(params);
        const auto sounding = source.build({250.0, 750.0}, 500.0);
        REQUIRE(sounding.p_pa.size() == 2);
        REQUIRE_NOTHROW(sounding.verify_self_consistent());
    }
    for (const char* path : {
             "tests/init/fixtures/sharpy_netcdf_char_public_station_1024.nc",
             "tests/init/fixtures/sharpy_netcdf_string_public_station_1024.nc",
             "tests/init/fixtures/sharpy_netcdf_string_public_timestamp_1024.nc"})
    {
        params.file_path = path;
        const tmv::init::FileSoundingSource source(params);
        REQUIRE_THROWS_WITH(source.build({250.0, 750.0}, 500.0),
                            Catch::Matchers::Contains("1023-byte limit"));
    }
}

TEST_CASE("FileSoundingSource builds complete model columns from every supplied CM1 sounding",
          "[init][sounding][file][cm1][reference][coverage]")
{
    const std::array<std::string, 4> paths = {
        "tests/fixtures/soundings/cm1/HIGHCAPENEW_QUARTERCIRCLE",
        "tests/fixtures/soundings/cm1/HIGHCAPENEW_SEMICIRCLE_02",
        "tests/fixtures/soundings/cm1/MEDCAPENEW_QUARTERCIRCLE",
        "tests/fixtures/soundings/cm1/MEDCAPENEW_SEMICIRCLE_06",
    };
    const auto z_m = linspace(250.0, 15750.0, 32);

    for (const auto& path : paths)
    {
        CAPTURE(path);
        tmv::init::FileSoundingParams params;
        params.file_path = path;
        params.scheme_id = "auto";
        params.require_winds = true;
        const tmv::init::FileSoundingSource source(params);

        const auto sounding = source.build(z_m, 500.0);
        REQUIRE_NOTHROW(sounding.verify_self_consistent());
        REQUIRE(sounding.size() == z_m.size());
        REQUIRE(sounding.has_winds());
        for (std::size_t k = 0; k < sounding.size(); ++k)
        {
            REQUIRE(std::isfinite(sounding.theta_k[k]));
            REQUIRE(std::isfinite(sounding.qv_kgkg[k]));
            REQUIRE(std::isfinite(sounding.u_ms[k]));
            REQUIRE(std::isfinite(sounding.v_ms[k]));
            REQUIRE(sounding.qv_kgkg[k] >= 0.0);
            if (k > 0)
            {
                REQUIRE(sounding.p_pa[k] < sounding.p_pa[k - 1]);
            }
        }
    }
}

TEST_CASE("CM1 high- and medium-CAPE families retain their physical ordering",
          "[init][sounding][file][cm1][diagnostics][reference]")
{
    const auto z_m = linspace(250.0, 15750.0, 32);
    const auto diagnostics_for = [&](const std::string& path)
    {
        tmv::init::FileSoundingParams params;
        params.file_path = path;
        params.scheme_id = "auto";
        params.require_winds = true;
        const tmv::init::FileSoundingSource source(params);
        return tmv::init::compute_sounding_diagnostics(
            source.build(z_m, 500.0));
    };

    const auto high = diagnostics_for(
        "tests/fixtures/soundings/cm1/HIGHCAPENEW_QUARTERCIRCLE");
    const auto medium = diagnostics_for(
        "tests/fixtures/soundings/cm1/MEDCAPENEW_QUARTERCIRCLE");

    REQUIRE(high.cape_jkg > medium.cape_jkg + 1000.0);
    REQUIRE(high.cape_jkg == Approx(5985.0).margin(1.0));
    REQUIRE(medium.cape_jkg == Approx(4650.0).margin(1.0));
    REQUIRE(high.lcl_m == Approx(medium.lcl_m).margin(1.0e-9));
    REQUIRE(high.lfc_m == Approx(medium.lfc_m).margin(1.0e-9));
    REQUIRE(high.el_m > medium.el_m);
}

TEST_CASE("FileSoundingSource rejects an invalid target grid before file I/O",
          "[init][sounding][file][grid][guard]")
{
    tmv::init::FileSoundingParams params;
    params.file_path = "/path/that/must/not/be/opened";
    params.scheme_id = "auto";
    const tmv::init::FileSoundingSource source(params);

    REQUIRE_THROWS_WITH(
        source.build({}, 500.0),
        Catch::Matchers::Contains("model-height column is empty"));
    REQUIRE_THROWS_WITH(
        source.build({250.0}, 0.0),
        Catch::Matchers::Contains("vertical spacing"));
    REQUIRE_THROWS_WITH(
        source.build({250.0, 200.0}, 500.0),
        Catch::Matchers::Contains("increase strictly"));
}

TEST_CASE("FileSoundingSource requires coverage through the model top interface",
          "[init][sounding][file][coverage]")
{
    tmv::init::FileSoundingParams params;
    params.file_path = "tests/fixtures/soundings/cm1/HIGHCAPENEW_QUARTERCIRCLE";
    params.scheme_id = "cm1";
    params.require_winds = true;
    tmv::init::FileSoundingSource source(params);

    REQUIRE_THROWS_WITH(
        source.build({19500.0}, 500.0),
        Catch::Matchers::Contains("exceeds sounding top"));

    // The scalar center is still inside the supplied profile, but its upper
    // half-cell is not. Accepting this would make "model-top coverage" depend
    // on storage staggering rather than the represented physical domain.
    REQUIRE_THROWS_WITH(
        source.build({18800.0}, 500.0),
        Catch::Matchers::Contains("upper half-cell"));
}

TEST_CASE("FileSoundingSource propagates strict NetCDF metadata types without native fallback",
          "[init][sounding][file][sharpy][netcdf-metadata-contract]")
{
    tmv::init::FileSoundingParams params;
    params.scheme_id = "auto";
    params.require_winds = true;
    const auto path = [](const char* name)
    {
        return std::string("tests/init/fixtures/sharpy_netcdf_metadata_types_") + name + ".nc";
    };
    for (const char* name : {"absent", "station_char", "station_string", "geo_alias"})
    {
        params.file_path = path(name);
        const tmv::init::FileSoundingSource source(params);
        const auto sounding = source.build({250.0, 750.0}, 500.0);
        REQUIRE(sounding.p_pa.size() == 2);
        REQUIRE(sounding.u_ms.size() == 2);
        REQUIRE_NOTHROW(sounding.verify_self_consistent());
    }
    for (const char* name : {"station_int", "timestamp_double", "latitude_char", "longitude_string",
                            "elevation_empty", "invalid_primary_alias"})
    {
        params.file_path = path(name);
        const tmv::init::FileSoundingSource source(params);
        INFO(name);
        REQUIRE_THROWS_WITH(source.build({250.0, 750.0}, 500.0), Catch::Matchers::Contains("NetCDF"));
    }
}

TEST_CASE("FileSoundingSource streaming NetCDF preserves candidate priority and group traversal",
          "[init][sounding][file][sharpy][netcdf-selected-enumeration]")
{
    tmv::init::FileSoundingParams params;
    params.scheme_id = "auto";
    params.require_winds = true;
    params.file_path = "tests/init/fixtures/sharpy_netcdf_public_safe.nc";
    const auto reference = tmv::init::FileSoundingSource(params).build({250.0, 750.0}, 500.0);
    for (const char* name : {"dense", "priority", "depth_first", "normalized"})
    {
        params.file_path = std::string("tests/init/fixtures/sharpy_netcdf_selected_") + name + ".nc";
        const auto selected = tmv::init::FileSoundingSource(params).build({250.0, 750.0}, 500.0);
        INFO(name);
        REQUIRE(selected.p_pa == reference.p_pa);
        REQUIRE(selected.theta_k == reference.theta_k);
        REQUIRE(selected.qv_kgkg == reference.qv_kgkg);
        REQUIRE(selected.u_ms == reference.u_ms);
        REQUIRE(selected.v_ms == reference.v_ms);
        REQUIRE_NOTHROW(selected.verify_self_consistent());
    }
    params.file_path = "tests/init/fixtures/sharpy_netcdf_selected_selected_oversized.nc";
    REQUIRE_THROWS_WITH(tmv::init::FileSoundingSource(params).build({250.0, 750.0}, 500.0),
                        Catch::Matchers::Contains("100,000-level safety limit"));
}

TEST_CASE("FileSoundingSource iterative NetCDF groups retain deep wide classic and alias-priority columns",
          "[init][sounding][file][sharpy][netcdf-iterative-groups]")
{
    tmv::init::FileSoundingParams params;
    params.scheme_id = "auto";
    params.require_winds = true;
    params.file_path = "tests/init/fixtures/sharpy_netcdf_public_safe.nc";
    const auto reference = tmv::init::FileSoundingSource(params).build({250.0, 750.0}, 500.0);
    for (const char* name : {"deep", "priority", "wide", "classic"})
    {
        params.file_path = std::string("tests/init/fixtures/sharpy_netcdf_groups_") + name + ".nc";
        const auto actual = tmv::init::FileSoundingSource(params).build({250.0, 750.0}, 500.0);
        INFO(name);
        REQUIRE(actual.p_pa == reference.p_pa);
        REQUIRE(actual.theta_k == reference.theta_k);
        REQUIRE(actual.qv_kgkg == reference.qv_kgkg);
        REQUIRE(actual.u_ms == reference.u_ms);
        REQUIRE(actual.v_ms == reference.v_ms);
        REQUIRE_NOTHROW(actual.verify_self_consistent());
    }
}

TEST_CASE("FileSoundingSource bounded scalar string validation preserves empty and full metadata contracts",
          "[init][sounding][file][sharpy][netcdf-bounded-string-scan]")
{
    tmv::init::FileSoundingParams params;
    params.scheme_id = "auto";
    params.require_winds = true;
    params.file_path = "tests/init/fixtures/sharpy_netcdf_public_safe.nc";
    const auto reference = tmv::init::FileSoundingSource(params).build({250.0, 750.0}, 500.0);
    for (const char* name : {"empty", "limit"})
    {
        params.file_path = std::string("tests/init/fixtures/sharpy_netcdf_bounded_string_") + name + ".nc";
        const auto actual = tmv::init::FileSoundingSource(params).build({250.0, 750.0}, 500.0);
        REQUIRE(actual.p_pa == reference.p_pa);
        REQUIRE(actual.theta_k == reference.theta_k);
        REQUIRE(actual.qv_kgkg == reference.qv_kgkg);
        REQUIRE(actual.u_ms == reference.u_ms);
        REQUIRE(actual.v_ms == reference.v_ms);
        REQUIRE_NOTHROW(actual.verify_self_consistent());
    }
    for (const char* name : {"station_1024", "station_65536", "timestamp_65536"})
    {
        params.file_path = std::string("tests/init/fixtures/sharpy_netcdf_bounded_string_") + name + ".nc";
        REQUIRE_THROWS_WITH(tmv::init::FileSoundingSource(params).build({250.0, 750.0}, 500.0),
            Catch::Matchers::Contains("string metadata exceeds the 1023-byte limit"));
    }
}
