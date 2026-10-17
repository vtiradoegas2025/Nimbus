#pragma once

// Public stochastic-scheme factory surface.

#include "chaos/scheme_descriptor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tmv::chaos
{
class ChaosScheme;

std::unique_ptr<ChaosScheme> create_chaos_scheme(const std::string& scheme_name);
std::vector<std::string> get_available_chaos_schemes();
} // namespace tmv::chaos

using tmv::chaos::create_chaos_scheme;
using tmv::chaos::get_available_chaos_schemes;
using tmv::chaos::get_chaos_scheme_descriptors;
