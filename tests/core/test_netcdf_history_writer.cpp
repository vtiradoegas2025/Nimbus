/**
 * @file test_netcdf_history_writer.cpp
 * @brief Qualification tests for the optional Nimbus netCDF4 history writer.
 */

#include "catch2/catch.hpp"
#include "core/output/netcdf_history_writer.hpp"
#include "core/output/step_manifest.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <filesystem>
#include <string>
#include <vector>

#ifdef HAVE_NETCDF
#include <netcdf.h>
#endif

namespace
{

std::filesystem::path temporary_step(const char* label)
{
    const auto serial = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto path =
        std::filesystem::temp_directory_path() / (std::string("nimbus_netcdf_") + label + "_" + std::to_string(serial));
    std::filesystem::create_directories(path);
    return path;
}

ExportSnapshot qualified_snapshot(const std::filesystem::path& step)
{
    ExportSnapshot snapshot;
    snapshot.export_index           = 4;
    snapshot.simulation_time_s      = 75.5;
    snapshot.step_dir               = step;
    snapshot.grid.nx                = 2;
    snapshot.grid.ny                = 3;
    snapshot.grid.nz                = 2;
    snapshot.grid.dx_m              = 1000.0;
    snapshot.grid.dy_m              = 750.0;
    snapshot.grid.dz_m              = 250.0;
    snapshot.grid.coordinate_system = "cartesian";
    snapshot.grid.native_staggering = "c_grid";
    snapshot.grid.domain_motion_mode = "constant_velocity";
    snapshot.grid.grid_origin_t0_earth_x_m = 100.0;
    snapshot.grid.grid_origin_t0_earth_y_m = -200.0;
    snapshot.grid.frame_velocity_u_ms = 10.0;
    snapshot.grid.frame_velocity_v_ms = -5.0;
    snapshot.grid.grid_origin_earth_x_m = 855.0;
    snapshot.grid.grid_origin_earth_y_m = -577.5;

    FieldSnapshotEntry theta;
    theta.name  = "theta";
    theta.dim0  = 2;
    theta.dim1  = 3;
    theta.dim2  = 2;
    theta.is_3d = true;
    theta.data  = {300.0f, 301.0f, 302.0f, 303.0f, 304.0f, 305.0f, 310.0f, 311.0f, 312.0f, 313.0f, 314.0f, 315.0f};
    snapshot.fields.push_back(theta);

    FieldSnapshotEntry rain;
    rain.name  = "accumulated_rainfall";
    rain.dim0  = 2;
    rain.dim1  = 3;
    rain.dim2  = 0;
    rain.is_3d = false;
    rain.data  = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    snapshot.fields.push_back(rain);
    return snapshot;
}

#ifdef HAVE_NETCDF
void require_nc(int status)
{
    INFO(nc_strerror(status));
    REQUIRE(status == NC_NOERR);
}
#endif

} // namespace

TEST_CASE("netCDF history capability reports the exact compiled binary", "[core][output][netcdf][capability]")
{
    const NetcdfHistoryCapability capability = netcdf_history_capability();
#ifdef HAVE_NETCDF
    REQUIRE(capability.available);
    REQUIRE(capability.implementation.find("libnetcdf") != std::string::npos);
    REQUIRE(capability.reason.empty());
#else
    REQUIRE_FALSE(capability.available);
    REQUIRE(capability.reason.find("without libnetcdf") != std::string::npos);
#endif
}

TEST_CASE("netCDF history rejects inconsistent grids before publication", "[core][output][netcdf][fail-closed]")
{
    const auto     step          = temporary_step("invalid");
    ExportSnapshot snapshot      = qualified_snapshot(step);
    snapshot.fields.front().dim2 = 3;
    NetcdfHistoryWriteResult result;
    std::string              error;
    REQUIRE_FALSE(write_netcdf4_history(snapshot, {}, result, error));
    REQUIRE_FALSE(error.empty());
    REQUIRE_FALSE(std::filesystem::exists(step / "history.nc"));
    REQUIRE_FALSE(std::filesystem::exists(step / "history.nc.tmp"));
    std::filesystem::remove_all(step);
}

#ifdef HAVE_NETCDF
TEST_CASE("netCDF4 history roundtrip preserves coordinates metadata and fields", "[core][output][netcdf][roundtrip]")
{
    const auto           step     = temporary_step("roundtrip");
    ExportSnapshot snapshot = qualified_snapshot(step);
    const auto expected_theta = snapshot.fields.front().data;
    const auto expected_rain = snapshot.fields.back().data;
    const bool borrowed = GENERATE(false, true);
    if (borrowed)
    {
        snapshot.fields.front().data.clear();
        snapshot.fields.front().borrow_for_synchronous_write(expected_theta.data(), expected_theta.size());
        snapshot.fields.back().data.clear();
        snapshot.fields.back().borrow_for_synchronous_write(expected_rain.data(), expected_rain.size());
        REQUIRE(snapshot.owned_bytes() == 0);
    }
    NetcdfHistoryPolicy  policy;
    policy.deflate_level      = 3;
    policy.shuffle            = true;
    policy.target_chunk_bytes = 32;
    NetcdfHistoryWriteResult result;
    std::string              error;
    REQUIRE(write_netcdf4_history(snapshot, policy, result, error));
    REQUIRE(error.empty());
    REQUIRE(result.path == step / "history.nc");
    REQUIRE(result.bytes_written > 0);
    REQUIRE_FALSE(std::filesystem::exists(step / "history.nc.tmp"));

    int ncid = -1;
    require_nc(nc_open(result.path.string().c_str(), NC_NOWRITE, &ncid));

    int format = 0;
    require_nc(nc_inq_format(ncid, &format));
    REQUIRE((format == NC_FORMAT_NETCDF4 || format == NC_FORMAT_NETCDF4_CLASSIC));

    int version = 0;
    require_nc(nc_get_att_int(ncid, NC_GLOBAL, "nimbus_manifest_version", &version));
    REQUIRE(version == kNimbusOutputManifestVersion);
    signed char restart_complete = 1;
    require_nc(nc_get_att_schar(ncid, NC_GLOBAL, "restart_complete", &restart_complete));
    REQUIRE(restart_complete == 0);

    int time_dim = -1, x_dim = -1, y_dim = -1, z_dim = -1;
    require_nc(nc_inq_dimid(ncid, "time", &time_dim));
    require_nc(nc_inq_dimid(ncid, "x", &x_dim));
    require_nc(nc_inq_dimid(ncid, "y", &y_dim));
    require_nc(nc_inq_dimid(ncid, "z", &z_dim));
    std::size_t length = 0;
    require_nc(nc_inq_dimlen(ncid, time_dim, &length));
    REQUIRE(length == 1);
    require_nc(nc_inq_dimlen(ncid, x_dim, &length));
    REQUIRE(length == 2);
    require_nc(nc_inq_dimlen(ncid, y_dim, &length));
    REQUIRE(length == 3);
    require_nc(nc_inq_dimlen(ncid, z_dim, &length));
    REQUIRE(length == 2);

    int x_var = -1;
    require_nc(nc_inq_varid(ncid, "x", &x_var));
    std::vector<double> x(2);
    require_nc(nc_get_var_double(ncid, x_var, x.data()));
    REQUIRE(x == std::vector<double>{1355.0, 2355.0});

    std::vector<double> frame_origin(2);
    require_nc(nc_get_att_double(
        ncid, NC_GLOBAL, "grid_origin_earth_m", frame_origin.data()));
    REQUIRE(frame_origin == std::vector<double>{855.0, -577.5});
    std::vector<double> frame_velocity(2);
    require_nc(nc_get_att_double(
        ncid, NC_GLOBAL, "frame_velocity_earth_ms", frame_velocity.data()));
    REQUIRE(frame_velocity == std::vector<double>{10.0, -5.0});
    std::size_t mode_length = 0;
    require_nc(nc_inq_attlen(
        ncid, NC_GLOBAL, "domain_motion_mode", &mode_length));
    std::string motion_mode(mode_length, '\0');
    require_nc(nc_get_att_text(
        ncid, NC_GLOBAL, "domain_motion_mode", motion_mode.data()));
    REQUIRE(motion_mode == "constant_velocity");

    int time_var = -1;
    require_nc(nc_inq_varid(ncid, "time", &time_var));
    double time = 0.0;
    require_nc(nc_get_var_double(ncid, time_var, &time));
    REQUIRE(time == 75.5);

    int theta_var = -1;
    require_nc(nc_inq_varid(ncid, "theta", &theta_var));
    int         storage   = 0;
    std::size_t chunks[4] = {};
    require_nc(nc_inq_var_chunking(ncid, theta_var, &storage, chunks));
    REQUIRE(storage == NC_CHUNKED);
    REQUIRE(chunks[0] == 1);
    int shuffle = 0, deflate = 0, level = 0;
    require_nc(nc_inq_var_deflate(ncid, theta_var, &shuffle, &deflate, &level));
    REQUIRE(shuffle == 1);
    REQUIRE(deflate == 1);
    REQUIRE(level == 3);

    std::vector<float> theta(expected_theta.size());
    require_nc(nc_get_var_float(ncid, theta_var, theta.data()));
    REQUIRE(theta == expected_theta);

    std::size_t units_length = 0;
    require_nc(nc_inq_attlen(ncid, theta_var, "units", &units_length));
    std::string units(units_length, '\0');
    require_nc(nc_get_att_text(ncid, theta_var, "units", units.data()));
    REQUIRE(units == "K");

    int rain_var = -1;
    require_nc(nc_inq_varid(ncid, "accumulated_rainfall", &rain_var));
    int ndims = 0;
    require_nc(nc_inq_varndims(ncid, rain_var, &ndims));
    REQUIRE(ndims == 3);
    std::vector<float> rain(expected_rain.size());
    require_nc(nc_get_var_float(ncid, rain_var, rain.data()));
    REQUIRE(rain == expected_rain);

    require_nc(nc_close(ncid));
    std::filesystem::remove_all(step);
}

TEST_CASE("netCDF4 history atomically replaces an existing frame", "[core][output][netcdf][atomic]")
{
    const auto     step     = temporary_step("replace");
    ExportSnapshot snapshot = qualified_snapshot(step);
    NetcdfHistoryWriteResult result;
    std::string              error;
    REQUIRE(write_netcdf4_history(snapshot, {}, result, error));
    snapshot.fields.front().data.front() = 999.0f;
    REQUIRE(write_netcdf4_history(snapshot, {}, result, error));
    REQUIRE_FALSE(std::filesystem::exists(step / "history.nc.tmp"));

    int ncid = -1;
    require_nc(nc_open(result.path.string().c_str(), NC_NOWRITE, &ncid));
    int theta_var = -1;
    require_nc(nc_inq_varid(ncid, "theta", &theta_var));
    const std::size_t start[4] = {0, 0, 0, 0};
    float             first    = 0.0f;
    require_nc(nc_get_var1_float(ncid, theta_var, start, &first));
    REQUIRE(first == 999.0f);
    require_nc(nc_close(ncid));
    std::filesystem::remove_all(step);
}

TEST_CASE("netCDF4 accepts and identifies nested child history", "[core][output][netcdf][nesting]")
{
    const auto     step     = temporary_step("nested_child");
    ExportSnapshot snapshot = qualified_snapshot(step);
    snapshot.data_product_kind = "nested_child_history";
    NetcdfHistoryWriteResult result;
    std::string              error;
    REQUIRE(write_netcdf4_history(snapshot, {}, result, error));
    REQUIRE(error.empty());

    int ncid = -1;
    require_nc(nc_open(result.path.string().c_str(), NC_NOWRITE, &ncid));
    std::size_t kind_length = 0;
    require_nc(nc_inq_attlen(ncid, NC_GLOBAL, "nimbus_data_product", &kind_length));
    std::string kind(kind_length, '\0');
    require_nc(nc_get_att_text(ncid, NC_GLOBAL, "nimbus_data_product", kind.data()));
    REQUIRE(kind == "nested_child_history");
    require_nc(nc_close(ncid));
    std::filesystem::remove_all(step);
}

TEST_CASE("netCDF terrain history preserves exact 3-D physical heights",
          "[core][output][netcdf][terrain][TE-005]")
{
    const auto step = temporary_step("terrain");
    ExportSnapshot snapshot = qualified_snapshot(step);
    snapshot.grid.terrain_following = true;
    snapshot.grid.physical_z_center_m = {
        40.0, 170.0, 45.0, 185.0, 50.0, 200.0,
        90.0, 260.0, 95.0, 275.0, 100.0, 290.0,
    };
    snapshot.grid.physical_z_interface_m = {
        0.0, 80.0, 260.0, 0.0, 90.0, 280.0, 0.0, 100.0, 300.0,
        50.0, 130.0, 390.0, 50.0, 140.0, 410.0, 50.0, 150.0, 430.0,
    };

    NetcdfHistoryWriteResult result;
    std::string error;
    REQUIRE(write_netcdf4_history(snapshot, {}, result, error));
    REQUIRE(error.empty());

    int ncid = -1;
    require_nc(nc_open(result.path.string().c_str(), NC_NOWRITE, &ncid));
    int interface_dim = -1;
    require_nc(nc_inq_dimid(ncid, "z_interface", &interface_dim));
    std::size_t length = 0;
    require_nc(nc_inq_dimlen(ncid, interface_dim, &length));
    REQUIRE(length == 3);

    int center_var = -1, interface_var = -1;
    require_nc(nc_inq_varid(ncid, "physical_z_center_m", &center_var));
    require_nc(nc_inq_varid(ncid, "physical_z_interface_m", &interface_var));
    std::vector<double> centers(snapshot.grid.physical_z_center_m.size());
    std::vector<double> interfaces(snapshot.grid.physical_z_interface_m.size());
    require_nc(nc_get_var_double(ncid, center_var, centers.data()));
    require_nc(nc_get_var_double(ncid, interface_var, interfaces.data()));
    REQUIRE(centers == snapshot.grid.physical_z_center_m);
    REQUIRE(interfaces == snapshot.grid.physical_z_interface_m);

    int z_var = -1;
    require_nc(nc_inq_varid(ncid, "z", &z_var));
    std::vector<double> computational_z(2);
    require_nc(nc_get_var_double(ncid, z_var, computational_z.data()));
    REQUIRE(computational_z == std::vector<double>{0.0, 1.0});
    std::size_t units_length = 0;
    require_nc(nc_inq_attlen(ncid, z_var, "units", &units_length));
    std::string units(units_length, '\0');
    require_nc(nc_get_att_text(ncid, z_var, "units", units.data()));
    REQUIRE(units == "1");

    int theta_var = -1;
    require_nc(nc_inq_varid(ncid, "theta", &theta_var));
    std::size_t coordinates_length = 0;
    require_nc(nc_inq_attlen(ncid, theta_var, "coordinates", &coordinates_length));
    std::string coordinates(coordinates_length, '\0');
    require_nc(nc_get_att_text(ncid, theta_var, "coordinates", coordinates.data()));
    REQUIRE(coordinates == "time x y physical_z_center_m");

    require_nc(nc_close(ncid));
    std::filesystem::remove_all(step);
}
#endif

#ifdef HAVE_NETCDF
TEST_CASE("netCDF pressure diagnostics preserve and declare missing values",
          "[core][output][netcdf][missing]")
{
    const auto step = temporary_step("missing_pressure");
    auto snapshot = qualified_snapshot(step);
    auto& field = snapshot.fields.back();
    field.name = "lifted_index";
    field.data[1] = std::numeric_limits<float>::quiet_NaN();
    NetcdfHistoryWriteResult result;
    std::string error;
    REQUIRE(write_netcdf4_history(snapshot, {}, result, error));
    int file = -1, variable = -1;
    require_nc(nc_open(result.path.string().c_str(), NC_NOWRITE, &file));
    require_nc(nc_inq_varid(file, "lifted_index", &variable));
    float fill = 0.0f;
    require_nc(nc_get_att_float(file, variable, "_FillValue", &fill));
    REQUIRE(std::isnan(fill));
    std::vector<float> values(6);
    require_nc(nc_get_var_float(file, variable, values.data()));
    REQUIRE(std::isnan(values[1]));
    REQUIRE(values[2] == field.data[2]);
    require_nc(nc_close(file));
    std::filesystem::remove_all(step);
}
#endif
