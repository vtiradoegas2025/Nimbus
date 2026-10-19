#pragma once

#include "soundings/soundings_base.hpp"

#include <string>
#include <vector>

namespace tmv::soundings::sharpy
{

SoundingData read_sharpy_text_file(const std::string& file_path);
SoundingData parse_sharpy_profile_rows(const std::vector<std::vector<double>>& profile_data);

}  // namespace tmv::soundings::sharpy
