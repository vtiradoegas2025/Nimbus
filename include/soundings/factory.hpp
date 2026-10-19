#pragma once

// Public legacy sounding-reader backend factory surface.

#include "soundings/scheme_descriptor.hpp"

#include <memory>
#include <string>

class SoundingScheme;

std::unique_ptr<SoundingScheme> create_sharpy_sounding_scheme();
std::unique_ptr<SoundingScheme> create_sounding_scheme(const std::string& scheme_id);
