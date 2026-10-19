#pragma once

#include "init/config/config_v1.hpp"

namespace tmv::init::config
{

bool is_public_qualified_mapped_terrain_profile(
    const ConfigV1& config) noexcept;

bool is_public_qualified_collocated_mapped_terrain_profile(
    const ConfigV1& config) noexcept;

bool is_public_qualified_native_mapped_terrain_profile(
    const ConfigV1& config) noexcept;

bool is_public_qualified_vulkan_mapped_terrain_profile(
    const ConfigV1& config) noexcept;

bool is_public_qualified_metal_mapped_terrain_profile(
    const ConfigV1& config) noexcept;

} // namespace tmv::init::config
