/**
 * @file netcdf_history_writer.cpp
 * @brief Optional netCDF4 history writer for Cartesian scalar-cell output.
 */

#include "core/output/netcdf_history_writer.hpp"

#include "core/output/step_manifest.hpp"
#include "diagnostics/field_contract.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <sstream>
#include <vector>

#ifdef HAVE_NETCDF
#include <netcdf.h>
#endif

namespace
{

constexpr const char* kHistoryFilename = "history.nc";

bool checked_product(std::size_t lhs, std::size_t rhs, std::size_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool validate_snapshot(const ExportSnapshot& snapshot, const NetcdfHistoryPolicy& policy, std::string& error)
{
    if (snapshot.step_dir.empty())
    {
        error = "netCDF history output requires a step directory";
        return false;
    }
    if (snapshot.grid.coordinate_system != "cartesian")
    {
        error = "netCDF history output currently supports only Cartesian grids";
        return false;
    }
    if (snapshot.grid.nx <= 0 || snapshot.grid.ny <= 0 || snapshot.grid.nz <= 0)
    {
        error = "netCDF history output requires positive x/y/z dimensions";
        return false;
    }
    if (!std::isfinite(snapshot.grid.dx_m) || snapshot.grid.dx_m <= 0.0 || !std::isfinite(snapshot.grid.dy_m) ||
        snapshot.grid.dy_m <= 0.0 || !std::isfinite(snapshot.grid.dz_m) || snapshot.grid.dz_m <= 0.0)
    {
        error = "netCDF history output requires finite positive x/y/z spacing";
        return false;
    }
    if (!std::isfinite(snapshot.simulation_time_s))
    {
        error = "netCDF history output requires a finite simulation time";
        return false;
    }
    const double horizontal_frame_values[] = {
        snapshot.grid.grid_origin_t0_earth_x_m,
        snapshot.grid.grid_origin_t0_earth_y_m,
        snapshot.grid.grid_origin_earth_x_m,
        snapshot.grid.grid_origin_earth_y_m,
        snapshot.grid.frame_velocity_u_ms,
        snapshot.grid.frame_velocity_v_ms,
    };
    if (snapshot.grid.domain_motion_mode.empty()
        || snapshot.grid.horizontal_frame_contract.empty()
        || std::any_of(
            std::begin(horizontal_frame_values),
            std::end(horizontal_frame_values),
            [](double value) { return !std::isfinite(value); }))
    {
        error = "netCDF history output requires complete finite horizontal-frame provenance";
        return false;
    }
    if (policy.deflate_level < 0 || policy.deflate_level > 9)
    {
        error = "netCDF deflate level must be in [0, 9]";
        return false;
    }
    if (policy.target_chunk_bytes < sizeof(float))
    {
        error = "netCDF target chunk size is smaller than one float";
        return false;
    }
    if (snapshot.fields.empty())
    {
        error = "netCDF history output refuses to publish an empty field set";
        return false;
    }
    const bool nested_child_history =
        snapshot.data_product_kind == "nested_child_history";
    const bool retiring_tile =
        snapshot.data_product_kind == "retiring_swath_tile";
    if ((snapshot.data_product_kind != "history" && !nested_child_history && !retiring_tile)
        || (retiring_tile
            && (snapshot.domain_recycling_transaction_id.empty()
                || snapshot.domain_recycling_tile_ordinal < 0)))
    {
        error = "netCDF history output has invalid data-product semantics";
        return false;
    }
    if (snapshot.grid.terrain_following)
    {
        const std::size_t center_count =
            static_cast<std::size_t>(snapshot.grid.nx)
            * static_cast<std::size_t>(snapshot.grid.ny)
            * static_cast<std::size_t>(snapshot.grid.nz);
        const std::size_t interface_count =
            static_cast<std::size_t>(snapshot.grid.nx)
            * static_cast<std::size_t>(snapshot.grid.ny)
            * static_cast<std::size_t>(snapshot.grid.nz + 1);
        if (snapshot.grid.physical_z_center_m.size() != center_count
            || snapshot.grid.physical_z_interface_m.size() != interface_count)
        {
            error = "terrain netCDF output requires complete center/interface height arrays";
            return false;
        }
        for (int i = 0; i < snapshot.grid.nx; ++i)
            for (int j = 0; j < snapshot.grid.ny; ++j)
                for (int k = 0; k < snapshot.grid.nz; ++k)
                {
                    const std::size_t center_index =
                        (static_cast<std::size_t>(i) * snapshot.grid.ny + j)
                        * snapshot.grid.nz + k;
                    const std::size_t face_index =
                        (static_cast<std::size_t>(i) * snapshot.grid.ny + j)
                        * static_cast<std::size_t>(snapshot.grid.nz + 1) + k;
                    const double center = snapshot.grid.physical_z_center_m[center_index];
                    const double lower = snapshot.grid.physical_z_interface_m[face_index];
                    const double upper = snapshot.grid.physical_z_interface_m[face_index + 1];
                    if (!std::isfinite(center) || !std::isfinite(lower)
                        || !std::isfinite(upper) || !(upper > lower)
                        || !(center > lower) || !(center < upper))
                    {
                        error = "terrain netCDF output contains malformed physical heights";
                        return false;
                    }
                }
    }

    std::set<std::string> names;
    for (const FieldSnapshotEntry& field : snapshot.fields)
    {
        if (field.name.empty() || !names.insert(field.name).second)
        {
            error = field.name.empty() ? "netCDF history field name is empty"
                                       : "netCDF history field name is duplicated: " + field.name;
            return false;
        }

        const bool volume = field.is_3d;
        if (field.dim0 != snapshot.grid.nx ||
            (volume && (field.dim1 != snapshot.grid.ny || field.dim2 != snapshot.grid.nz)) ||
            (!volume && ((field.dim1 > 0 ? field.dim1 : field.dim2) != snapshot.grid.ny)))
        {
            error = "netCDF history field dimensions do not match the declared grid: " + field.name;
            return false;
        }

        std::size_t expected = static_cast<std::size_t>(snapshot.grid.nx);
        if (!checked_product(expected, static_cast<std::size_t>(snapshot.grid.ny), expected) ||
            (volume && !checked_product(expected, static_cast<std::size_t>(snapshot.grid.nz), expected)))
        {
            error = "netCDF history field element count overflow: " + field.name;
            return false;
        }
        if (field.values().size() != expected)
        {
            error = "netCDF history field payload size mismatch: " + field.name;
            return false;
        }
    }
    return true;
}

#ifdef HAVE_NETCDF

bool nc_ok(int status, const std::string& operation, std::string& error)
{
    if (status == NC_NOERR)
    {
        return true;
    }
    error = operation + ": " + nc_strerror(status);
    return false;
}

bool put_text_attribute(int ncid, int varid, const char* name, const std::string& value, std::string& error)
{
    return nc_ok(nc_put_att_text(ncid, varid, name, value.size(), value.data()),
                 std::string("failed to define attribute '") + name + "'",
                 error);
}

bool define_coordinate(int          ncid,
                       const char*  name,
                       const char*  standard_name,
                       const char*  axis,
                       int          dimension,
                       int&         variable,
                       std::string& error)
{
    if (!nc_ok(nc_def_var(ncid, name, NC_DOUBLE, 1, &dimension, &variable),
               std::string("failed to define coordinate '") + name + "'",
               error) ||
        !put_text_attribute(ncid, variable, "standard_name", standard_name, error) ||
        !put_text_attribute(
            ncid, variable, "long_name", std::string(name) + " coordinate of scalar cell center", error) ||
        !put_text_attribute(ncid, variable, "units", "m", error) ||
        !put_text_attribute(ncid, variable, "axis", axis, error))
    {
        return false;
    }
    return true;
}

std::vector<std::size_t>
field_chunks(const FieldSnapshotEntry& field, const ExportGridMetadata& grid, std::size_t target_bytes)
{
    std::size_t       cx            = static_cast<std::size_t>(grid.nx);
    std::size_t       cy            = static_cast<std::size_t>(grid.ny);
    std::size_t       cz            = field.is_3d ? static_cast<std::size_t>(grid.nz) : 1U;
    const std::size_t target_values = std::max<std::size_t>(1, target_bytes / sizeof(float));
    while (cx * cy > target_values / cz)
    {
        if (cx >= cy && cx >= cz && cx > 1)
        {
            cx = (cx + 1U) / 2U;
        }
        else if (cy >= cz && cy > 1)
        {
            cy = (cy + 1U) / 2U;
        }
        else if (cz > 1)
        {
            cz = (cz + 1U) / 2U;
        }
        else
        {
            break;
        }
    }
    if (field.is_3d)
    {
        return {1U, cx, cy, cz};
    }
    return {1U, cx, cy};
}

struct OpenNetcdfFile
{
    int id = -1;
    ~OpenNetcdfFile()
    {
        if (id >= 0)
        {
            nc_abort(id);
        }
    }
};

#endif

} // namespace

NetcdfHistoryCapability netcdf_history_capability()
{
#ifdef HAVE_NETCDF
    return {true, std::string("libnetcdf ") + nc_inq_libvers(), {}};
#else
    return {false, {}, "this Nimbus binary was built without libnetcdf"};
#endif
}

bool write_netcdf4_history(const ExportSnapshot&      snapshot,
                           const NetcdfHistoryPolicy& policy,
                           NetcdfHistoryWriteResult&  result,
                           std::string&               error)
{
    result = {};
    error.clear();
    const NetcdfHistoryCapability capability = netcdf_history_capability();
    if (!capability.available)
    {
        error = "netCDF4 history output is unavailable: " + capability.reason;
        return false;
    }
    if (!validate_snapshot(snapshot, policy, error))
    {
        return false;
    }

#ifndef HAVE_NETCDF
    return false;
#else
    const std::filesystem::path final_path     = snapshot.step_dir / kHistoryFilename;
    const std::filesystem::path temporary_path = final_path.string() + ".tmp";
    std::error_code             ec;
    std::filesystem::remove(temporary_path, ec);
    ec.clear();

    OpenNetcdfFile file;
    if (!nc_ok(nc_create(temporary_path.string().c_str(), NC_NETCDF4 | NC_CLOBBER, &file.id),
               "failed to create temporary netCDF4 history file",
               error))
    {
        return false;
    }

    int time_dim = -1, x_dim = -1, y_dim = -1, z_dim = -1;
    int z_interface_dim = -1;
    if (!nc_ok(nc_def_dim(file.id, "time", NC_UNLIMITED, &time_dim), "failed to define time dimension", error) ||
        !nc_ok(nc_def_dim(file.id, "x", static_cast<std::size_t>(snapshot.grid.nx), &x_dim),
               "failed to define x dimension",
               error) ||
        !nc_ok(nc_def_dim(file.id, "y", static_cast<std::size_t>(snapshot.grid.ny), &y_dim),
               "failed to define y dimension",
               error) ||
        !nc_ok(nc_def_dim(file.id, "z", static_cast<std::size_t>(snapshot.grid.nz), &z_dim),
               "failed to define z dimension",
               error))
    {
        return false;
    }
    if (snapshot.grid.terrain_following
        && !nc_ok(nc_def_dim(file.id, "z_interface",
                             static_cast<std::size_t>(snapshot.grid.nz + 1),
                             &z_interface_dim),
                  "failed to define terrain interface dimension", error))
        return false;

    int time_var = -1, x_var = -1, y_var = -1, z_var = -1;
    if (!nc_ok(nc_def_var(file.id, "time", NC_DOUBLE, 1, &time_dim, &time_var),
               "failed to define time coordinate",
               error) ||
        !put_text_attribute(file.id, time_var, "standard_name", "time", error) ||
        !put_text_attribute(file.id, time_var, "long_name", "simulation time", error) ||
        !put_text_attribute(file.id, time_var, "units", "s", error) ||
        !put_text_attribute(file.id, time_var, "axis", "T", error) ||
        !define_coordinate(file.id, "x", "projection_x_coordinate", "X", x_dim, x_var, error) ||
        !define_coordinate(file.id, "y", "projection_y_coordinate", "Y", y_dim, y_var, error))
    {
        return false;
    }
    if (snapshot.grid.terrain_following)
    {
        if (!nc_ok(nc_def_var(file.id, "z", NC_DOUBLE, 1, &z_dim, &z_var),
                   "failed to define terrain computational z coordinate", error)
            || !put_text_attribute(file.id, z_var, "long_name",
                                   "terrain-following computational layer coordinate", error)
            || !put_text_attribute(file.id, z_var, "units", "1", error)
            || !put_text_attribute(file.id, z_var, "axis", "Z", error)
            || !put_text_attribute(file.id, z_var, "positive", "up", error))
            return false;
    }
    else if (!define_coordinate(file.id, "z", "height", "Z", z_dim, z_var, error)
             || !put_text_attribute(file.id, z_var, "positive", "up", error))
    {
        return false;
    }

    int physical_z_center_var = -1;
    int physical_z_interface_var = -1;
    if (snapshot.grid.terrain_following)
    {
        const int center_dims[3] = {x_dim, y_dim, z_dim};
        const int interface_dims[3] = {x_dim, y_dim, z_interface_dim};
        if (!nc_ok(nc_def_var(file.id, "physical_z_center_m", NC_DOUBLE, 3,
                              center_dims, &physical_z_center_var),
                   "failed to define terrain center heights", error)
            || !nc_ok(nc_def_var(file.id, "physical_z_interface_m", NC_DOUBLE, 3,
                                 interface_dims, &physical_z_interface_var),
                      "failed to define terrain interface heights", error)
            || !put_text_attribute(file.id, physical_z_center_var,
                                   "standard_name", "altitude", error)
            || !put_text_attribute(file.id, physical_z_center_var,
                                   "long_name", "physical scalar-cell height", error)
            || !put_text_attribute(file.id, physical_z_center_var, "units", "m", error)
            || !put_text_attribute(file.id, physical_z_center_var, "positive", "up", error)
            || !put_text_attribute(file.id, physical_z_interface_var,
                                   "long_name", "physical layer-interface height", error)
            || !put_text_attribute(file.id, physical_z_interface_var, "units", "m", error)
            || !put_text_attribute(file.id, physical_z_interface_var, "positive", "up", error))
            return false;
    }

    const int         manifest_version = kNimbusOutputManifestVersion;
    const signed char restart_complete = 0;
    const double grid_origin_t0_earth_m[2] = {
        snapshot.grid.grid_origin_t0_earth_x_m,
        snapshot.grid.grid_origin_t0_earth_y_m,
    };
    const double grid_origin_earth_m[2] = {
        snapshot.grid.grid_origin_earth_x_m,
        snapshot.grid.grid_origin_earth_y_m,
    };
    const double frame_velocity_earth_ms[2] = {
        snapshot.grid.frame_velocity_u_ms,
        snapshot.grid.frame_velocity_v_ms,
    };
    if (!put_text_attribute(file.id, NC_GLOBAL, "Conventions", "CF-1.10", error) ||
        !put_text_attribute(file.id, NC_GLOBAL, "title", "Nimbus Cartesian history frame", error) ||
        !put_text_attribute(file.id, NC_GLOBAL, "source", "Nimbus supercell simulation engine", error) ||
        !put_text_attribute(file.id, NC_GLOBAL, "nimbus_schema", "nimbus.step", error) ||
        !nc_ok(nc_put_att_int(file.id, NC_GLOBAL, "nimbus_manifest_version", NC_INT, 1, &manifest_version),
               "failed to define manifest version",
               error) ||
        !put_text_attribute(file.id, NC_GLOBAL, "nimbus_data_product",
                            snapshot.data_product_kind, error) ||
        !nc_ok(nc_put_att_schar(file.id, NC_GLOBAL, "restart_complete", NC_BYTE, 1, &restart_complete),
               "failed to define restart status",
               error) ||
        !put_text_attribute(file.id, NC_GLOBAL, "coordinate_system", snapshot.grid.coordinate_system, error) ||
        !put_text_attribute(file.id, NC_GLOBAL, "axis_order", "x y z", error) ||
        !put_text_attribute(file.id, NC_GLOBAL, "native_model_staggering", snapshot.grid.native_staggering, error) ||
        !put_text_attribute(file.id, NC_GLOBAL, "publication_staggering", "cell_centered", error) ||
        !put_text_attribute(file.id, NC_GLOBAL, "horizontal_frame_contract",
                            snapshot.grid.horizontal_frame_contract, error) ||
        !put_text_attribute(file.id, NC_GLOBAL, "domain_motion_mode",
                            snapshot.grid.domain_motion_mode, error) ||
        !put_text_attribute(file.id, NC_GLOBAL, "published_horizontal_wind",
                            "earth_relative", error) ||
        !put_text_attribute(file.id, NC_GLOBAL, "horizontal_transport_velocity",
                            "grid_relative", error) ||
        !nc_ok(nc_put_att_double(file.id, NC_GLOBAL,
                                 "grid_origin_t0_earth_m", NC_DOUBLE, 2,
                                 grid_origin_t0_earth_m),
               "failed to define t0 Earth-grid origin", error) ||
        !nc_ok(nc_put_att_double(file.id, NC_GLOBAL,
                                 "grid_origin_earth_m", NC_DOUBLE, 2,
                                 grid_origin_earth_m),
               "failed to define publication Earth-grid origin", error) ||
        !nc_ok(nc_put_att_double(file.id, NC_GLOBAL,
                                 "frame_velocity_earth_ms", NC_DOUBLE, 2,
                                 frame_velocity_earth_ms),
               "failed to define Earth-frame velocity", error) ||
        !put_text_attribute(file.id, NC_GLOBAL, "vertical_geometry",
                            snapshot.grid.terrain_following ? "terrain_following" : "uniform", error) ||
        !put_text_attribute(file.id,
                            NC_GLOBAL,
                            "compatibility_note",
                            "Nimbus history contract; not a CM1 restart or binary-compatible CM1 history file",
                            error))
    {
        return false;
    }
    if (snapshot.data_product_kind == "retiring_swath_tile")
    {
        const int tile_ordinal = snapshot.domain_recycling_tile_ordinal;
        if (!put_text_attribute(
                file.id, NC_GLOBAL, "domain_recycling_transaction_id",
                snapshot.domain_recycling_transaction_id, error)
            || !put_text_attribute(
                file.id, NC_GLOBAL, "domain_recycling_publication_phase",
                "captured_precommit", error)
            || !nc_ok(
                nc_put_att_int(file.id, NC_GLOBAL,
                               "domain_recycling_tile_ordinal", NC_INT, 1,
                               &tile_ordinal),
                "failed to define domain-recycling tile ordinal", error))
        {
            return false;
        }
    }

    std::vector<int> field_variables;
    field_variables.reserve(snapshot.fields.size());
    for (const FieldSnapshotEntry& field : snapshot.fields)
    {
        const int volume_dims[4]  = {time_dim, x_dim, y_dim, z_dim};
        const int surface_dims[3] = {time_dim, x_dim, y_dim};
        int       variable        = -1;
        if (!nc_ok(nc_def_var(file.id,
                              field.name.c_str(),
                              NC_FLOAT,
                              field.is_3d ? 4 : 3,
                              field.is_3d ? volume_dims : surface_dims,
                              &variable),
                   "failed to define netCDF field '" + field.name + "'",
                   error))
        {
            return false;
        }

        const std::vector<std::size_t> chunks = field_chunks(field, snapshot.grid, policy.target_chunk_bytes);
        if (!nc_ok(nc_def_var_chunking(file.id, variable, NC_CHUNKED, chunks.data()),
                   "failed to configure chunks for field '" + field.name + "'",
                   error) ||
            !nc_ok(
                nc_def_var_deflate(
                    file.id, variable, policy.shuffle ? 1 : 0, policy.deflate_level > 0 ? 1 : 0, policy.deflate_level),
                "failed to configure compression for field '" + field.name + "'",
                error))
        {
            return false;
        }

        const tmv::FieldContract* contract    = tmv::find_field_contract(field.name);
        if (contract && contract->allows_missing_values)
        {
            const float missing = std::numeric_limits<float>::quiet_NaN();
            if (!nc_ok(nc_def_var_fill(file.id, variable, 0, &missing),
                       "failed to declare missing pressure diagnostic values", error))
                return false;
        }
        const std::string         units       = contract ? contract->units : "1";
        const std::string         description = contract ? contract->description : field.name;
        const std::string         grid_location =
            contract ? tmv::to_string(contract->grid_location) : (field.is_3d ? "cell_center" : "surface_cell_center");
        if (!put_text_attribute(file.id, variable, "units", units, error) ||
            !put_text_attribute(file.id, variable, "long_name", description, error) ||
            !put_text_attribute(file.id, variable, "nimbus_field_id", field.name, error) ||
            !put_text_attribute(file.id, variable, "grid_location", grid_location, error) ||
            !put_text_attribute(file.id, variable, "coordinates",
                                field.is_3d && snapshot.grid.terrain_following
                                    ? "time x y physical_z_center_m"
                                    : (field.is_3d ? "time x y z" : "time x y"),
                                error))
        {
            return false;
        }
        field_variables.push_back(variable);
    }

    if (!nc_ok(nc_enddef(file.id), "failed to finalize netCDF metadata", error))
    {
        return false;
    }

    std::vector<double> x(static_cast<std::size_t>(snapshot.grid.nx));
    std::vector<double> y(static_cast<std::size_t>(snapshot.grid.ny));
    std::vector<double> z(static_cast<std::size_t>(snapshot.grid.nz));
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = snapshot.grid.grid_origin_earth_x_m
             + (static_cast<double>(i) + 0.5) * snapshot.grid.dx_m;
    for (std::size_t j = 0; j < y.size(); ++j)
        y[j] = snapshot.grid.grid_origin_earth_y_m
             + (static_cast<double>(j) + 0.5) * snapshot.grid.dy_m;
    for (std::size_t k = 0; k < z.size(); ++k)
        z[k] = snapshot.grid.terrain_following
            ? static_cast<double>(k)
            : (static_cast<double>(k) + 0.5) * snapshot.grid.dz_m;
    if (!nc_ok(nc_put_var_double(file.id, x_var, x.data()), "failed to write x coordinate", error) ||
        !nc_ok(nc_put_var_double(file.id, y_var, y.data()), "failed to write y coordinate", error) ||
        !nc_ok(nc_put_var_double(file.id, z_var, z.data()), "failed to write z coordinate", error))
    {
        return false;
    }
    if (snapshot.grid.terrain_following
        && (!nc_ok(nc_put_var_double(file.id, physical_z_center_var,
                                     snapshot.grid.physical_z_center_m.data()),
                   "failed to write terrain center heights", error)
            || !nc_ok(nc_put_var_double(file.id, physical_z_interface_var,
                                        snapshot.grid.physical_z_interface_m.data()),
                      "failed to write terrain interface heights", error)))
        return false;
    const std::size_t time_start[1] = {0U};
    const std::size_t time_count[1] = {1U};
    if (!nc_ok(nc_put_vara_double(file.id, time_var, time_start, time_count, &snapshot.simulation_time_s),
               "failed to write time coordinate",
               error))
    {
        return false;
    }

    for (std::size_t index = 0; index < snapshot.fields.size(); ++index)
    {
        const FieldSnapshotEntry& field    = snapshot.fields[index];
        const std::size_t         start[4] = {0U, 0U, 0U, 0U};
        const std::size_t         count[4] = {1U,
                                              static_cast<std::size_t>(snapshot.grid.nx),
                                              static_cast<std::size_t>(snapshot.grid.ny),
                                              field.is_3d ? static_cast<std::size_t>(snapshot.grid.nz) : 1U};
        if (!nc_ok(nc_put_vara_float(file.id, field_variables[index], start, count, field.values().data()),
                   "failed to write netCDF field '" + field.name + "'",
                   error))
        {
            return false;
        }
    }

    const int close_status = nc_close(file.id);
    file.id                = -1;
    if (!nc_ok(close_status, "failed to close netCDF history file", error))
    {
        std::filesystem::remove(temporary_path, ec);
        return false;
    }

    // Never unlink the prior frame first: that creates a visibility gap.  The
    // shared helper uses rename on POSIX and MoveFileEx(REPLACE_EXISTING) on
    // Windows, preserving one atomic publication contract on every host.
    if (!nimbus::output::atomic_replace_file(temporary_path, final_path, error))
    {
        std::filesystem::remove(temporary_path, ec);
        return false;
    }

    result.path          = final_path;
    result.bytes_written = std::filesystem::file_size(final_path, ec);
    if (ec)
    {
        error = "failed to measure netCDF history file '" + final_path.string() + "': " + ec.message();
        return false;
    }
    return true;
#endif
}
