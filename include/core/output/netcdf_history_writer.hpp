#pragma once

/**
 * @file netcdf_history_writer.hpp
 * @brief Optional, standards-aware netCDF4 history publication.
 */

#include "core/field/field_snapshot.hpp"

#include <cstddef>
#include <filesystem>
#include <string>

struct NetcdfHistoryCapability
{
    bool        available = false;
    std::string implementation;
    std::string reason;
};

/** Runtime truth for the exact binary: false unless linked with libnetcdf. */
NetcdfHistoryCapability netcdf_history_capability();

struct NetcdfHistoryPolicy
{
    /// HDF5 deflate level. Zero disables deflate; valid range is [0, 9].
    int  deflate_level = 2;
    bool shuffle       = true;

    /// Target upper bound used when deriving x/y/z chunks for a field.
    std::size_t target_chunk_bytes = 1024U * 1024U;
};

struct NetcdfHistoryWriteResult
{
    std::filesystem::path path;
    std::size_t           bytes_written = 0;
};

/**
 * Write one manifest-v3 history frame to `history.nc` in snapshot.step_dir.
 *
 * The final filename is published only after nc_close succeeds.  The file is
 * a visualization/analysis history product with one time record; it is not a
 * restart/checkpoint and makes no CM1 binary-compatibility claim.
 */
bool write_netcdf4_history(const ExportSnapshot&      snapshot,
                           const NetcdfHistoryPolicy& policy,
                           NetcdfHistoryWriteResult&  result,
                           std::string&               error);
