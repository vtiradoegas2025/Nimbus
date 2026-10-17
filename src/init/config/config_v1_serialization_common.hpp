#pragma once

#include "init/config/config_v1.hpp"

namespace tmv::init::config
{
void require_runtime_supported_configuration(const ConfigV1& config);
bool is_public_ready_native_weno5_profile(const ConfigV1& config);

bool is_public_qualified_native_implicit_diffusion_profile(const ConfigV1& config);

void   require_runtime_supported_diffusion(const ConfigV1& config);
void   require_runtime_supported_enum_selections(const ConfigV1& config);
void   require_runtime_supported_domain_motion(const ConfigV1& config);
bool   is_public_qualified_stationary_simple_grey_profile(
           const ConfigV1& config) noexcept;
bool   is_public_qualified_simple_grey_absorbing_water_cloud_profile(
           const ConfigV1& config) noexcept;
bool   is_public_qualified_domain_recycling_profile(
           const ConfigV1& config) noexcept;
bool is_public_qualified_nested_grid_recycling_profile(
    const ConfigV1& config) noexcept;
bool is_public_qualified_earth_fixed_terrain_motion_profile(const ConfigV1& config) noexcept;
bool is_public_qualified_earth_fixed_terrain_recycling_profile(
    const ConfigV1& config) noexcept;
bool   is_public_qualified_moving_nest_profile(
           const ConfigV1& config) noexcept;
bool is_public_ready_private_two_way_profile(const ConfigV1& config) noexcept;
bool is_public_qualified_accelerated_two_way_profile(const ConfigV1& config) noexcept;
bool is_public_qualified_two_way_nested_profile(
    const ConfigV1& config) noexcept;
bool is_public_qualified_active_terrain_nested_profile(
    const ConfigV1& config) noexcept;
bool   is_public_qualified_lin_rrtmg_profile(
           const ConfigV1& config) noexcept;
bool is_public_ready_milbrandt_reference_profile(const ConfigV1& config);
bool is_public_ready_thompson_native_profile(const ConfigV1& config);

bool is_public_qualified_thompson_cpu_profile(
    const ConfigV1& config) noexcept;
bool   is_public_qualified_cloud_top_radiative_ysu_profile(
           const ConfigV1& config) noexcept;
bool   is_public_qualified_radar_profile(
           const ConfigV1& config) noexcept;
bool is_public_qualified_user_filter_width_profile(
    const ConfigV1& config) noexcept;
bool is_public_qualified_richardson_smagorinsky_profile(const ConfigV1& config) noexcept;

bool is_public_qualified_dynamic_smagorinsky_profile(
    const ConfigV1& config) noexcept;
bool is_transactional_static_vertical_only_cpu_candidate(
    const ConfigV1& config);
bool is_public_qualified_tke_stability_profile(
    const ConfigV1& config) noexcept;
bool is_public_qualified_coupled_rk_profile(
    const ConfigV1& config) noexcept;
double sounding_qv_or_default(const SoundingSubsection& sounding);
void   require_runtime_supported_sounding(const ConfigV1& config);

} // namespace tmv::init::config
