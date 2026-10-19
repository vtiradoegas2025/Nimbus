#pragma once

#include "init/sounding/sounding_source_config.hpp"
#include "init/sounding/sounding_source.hpp"

#include <memory>
#include <string>

namespace tmv::init
{

// False leaves path-aware error reporting to the configuration loader.
bool parse_sounding_source_type(const std::string& s,
                                SoundingSourceConfig::Type& out);

std::string sounding_source_type_name(SoundingSourceConfig::Type t);

// An unset source preserves the historical ParametricCAPE default.
std::unique_ptr<SoundingSource> make_sounding_source(const SoundingSourceConfig& cfg);

}  // namespace tmv::init
