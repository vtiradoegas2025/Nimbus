#pragma once

#include "soundings/soundings_base.hpp"

#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tmv::soundings::sharpy
{

class Hdf5MetadataContractError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

class Hdf5ProfileResourceError : public Hdf5MetadataContractError
{
public:
    using Hdf5MetadataContractError::Hdf5MetadataContractError;
};

class NetcdfInputContractError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

class NetcdfProfileResourceError : public NetcdfInputContractError
{
public:
    using NetcdfInputContractError::NetcdfInputContractError;
};

class NetcdfMetadataContractError : public NetcdfInputContractError
{
public:
    using NetcdfInputContractError::NetcdfInputContractError;
};

struct NativeProfileExtraction
{
    std::vector<std::vector<double>> rows;
    std::string station_id;
    std::string timestamp_utc;
    double latitude_deg = std::numeric_limits<double>::quiet_NaN();
    double longitude_deg = std::numeric_limits<double>::quiet_NaN();
    double elevation_m = std::numeric_limits<double>::quiet_NaN();
};

struct NativeReadAttempt
{
    std::string backend;
    std::string detail;
};

NativeProfileExtraction parse_netcdf_classic_profile_native(const std::string& file_path);
NativeProfileExtraction parse_netcdf_dynamic_profile_native(const std::string& file_path);
NativeProfileExtraction parse_hdf5_profile_native(const std::string& file_path);

void append_native_attempt_failure(std::vector<NativeReadAttempt>& attempts,
                                   std::string backend,
                                   const std::exception& error);

[[noreturn]] void throw_native_ingest_failure(const std::string& file_path,
                                              const std::vector<NativeReadAttempt>& attempts);

void apply_native_metadata(SoundingData& data, const NativeProfileExtraction& native);

}  // namespace tmv::soundings::sharpy
