#include "milbrandt.hpp"
#include "triple_warm_adapter.hpp"
#include "triple_internal_thermal.hpp"

#include "core/runtime/simulation_state.hpp"
#include "core/runtime/runtime_config.hpp"
#include "compute/backend_runtime.hpp"
#include "boundary_layer/runtime.hpp"
#include "chaos/runtime.hpp"
#include "radiation/radiation_base.hpp"
#include "terrain/terrain_base.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tmv
{
namespace
{

bool matches(const Field3D& reference, const Field3D& field) noexcept
{
    return field.size_r() == reference.size_r()
        && field.size_th() == reference.size_th()
        && field.size_z() == reference.size_z();
}

bool zero_field(const Field3D& field) noexcept
{
    for (std::size_t index = 0; index < field.size(); ++index)
    {
        if (field.data()[index] != 0.0f)
        {
            return false;
        }
    }
    return true;
}

bool zero_optional(const Field3D* field) noexcept
{
    return field == nullptr || zero_field(*field);
}

} // namespace

bool MilbrandtScheme::validate_internal_triple_source_profile(
    const SimulationStateView& state,
    bool rain_freezing,
    std::string& error,
    std::optional<milbrandt_detail::TripleMomentSpecies> frozen_species,
    bool joint_frozen,
    bool melting_only,
    bool ice_melting_only,
    bool snow_melting_only,
    bool warm_primary_only,
    bool cloud_adjustment_only,
    bool warm_cloud_primary_only,
    bool dynamic_warm_cloud_primary_only,
    bool subcycled_warm_cloud_primary_only,
    bool frozen_collection_primary_only,
    bool cold_cloud_collection_primary_only,
    bool cold_cloud_diffusion_primary_only,
    bool cold_riming_primary_only, bool cold_rain_primary_only)
{
    if (cold_rain_primary_only && !cold_riming_primary_only)
    {
        error = "internal Milbrandt rain collection requires the cold composition owner";
        return false;
    }
    if (cold_riming_primary_only &&
        (rain_freezing || frozen_species || joint_frozen || melting_only || ice_melting_only ||
         snow_melting_only || warm_primary_only || cloud_adjustment_only || warm_cloud_primary_only ||
         dynamic_warm_cloud_primary_only || subcycled_warm_cloud_primary_only ||
         frozen_collection_primary_only || cold_cloud_collection_primary_only || cold_cloud_diffusion_primary_only))
    {
        error = "internal Milbrandt ColdRimingPrimaryOnly selected category is invalid";
        return false;
    }
    if (cold_cloud_diffusion_primary_only &&
        (rain_freezing || frozen_species || joint_frozen || melting_only || ice_melting_only ||
         snow_melting_only || warm_primary_only || cloud_adjustment_only ||
         warm_cloud_primary_only || dynamic_warm_cloud_primary_only || subcycled_warm_cloud_primary_only ||
         frozen_collection_primary_only || cold_cloud_collection_primary_only))
    {
        error = "internal Milbrandt ColdCloudDiffusionPrimaryOnly selected category is invalid";
        return false;
    }
    cold_cloud_collection_primary_only = cold_cloud_collection_primary_only ||
        cold_cloud_diffusion_primary_only || cold_riming_primary_only;
    const char* cold_cloud_label = cold_riming_primary_only
        ? "internal Milbrandt ColdRimingPrimaryOnly "
        : cold_cloud_diffusion_primary_only ? "internal Milbrandt ColdCloudDiffusionPrimaryOnly "
        : "internal Milbrandt ColdCloudCollectionPrimaryOnly ";
    if (cold_cloud_collection_primary_only &&
        (rain_freezing || frozen_species || joint_frozen || melting_only || ice_melting_only ||
         snow_melting_only || warm_primary_only || cloud_adjustment_only ||
         warm_cloud_primary_only || dynamic_warm_cloud_primary_only || subcycled_warm_cloud_primary_only ||
         frozen_collection_primary_only))
    {
        error = "internal Milbrandt ColdCloudCollectionPrimaryOnly selected category is invalid";
        return false;
    }
    if (frozen_collection_primary_only &&
        (rain_freezing || frozen_species || joint_frozen || melting_only || ice_melting_only ||
         snow_melting_only || warm_primary_only || cloud_adjustment_only ||
         warm_cloud_primary_only || dynamic_warm_cloud_primary_only || subcycled_warm_cloud_primary_only))
    {
        error = "internal Milbrandt FrozenCollectionPrimaryOnly selected category is invalid";
        return false;
    }
    const char* warm_cloud_label = subcycled_warm_cloud_primary_only
        ? "internal Milbrandt SubcycledWarmCloudPrimaryOnly "
        : dynamic_warm_cloud_primary_only ? "internal Milbrandt DynamicWarmCloudPrimaryOnly "
        : "internal Milbrandt WarmCloudPrimaryOnly ";
    if (static_cast<int>(warm_cloud_primary_only) + static_cast<int>(dynamic_warm_cloud_primary_only)
        + static_cast<int>(subcycled_warm_cloud_primary_only) > 1)
    {
        error = std::string(warm_cloud_label) + "selected category is invalid";
        return false;
    }
    warm_cloud_primary_only = warm_cloud_primary_only || dynamic_warm_cloud_primary_only
        || subcycled_warm_cloud_primary_only;
    if (warm_cloud_primary_only &&
        (rain_freezing || frozen_species || joint_frozen || melting_only || ice_melting_only ||
         snow_melting_only || warm_primary_only || cloud_adjustment_only))
    {
        error = std::string(warm_cloud_label) + "selected category is invalid";
        return false;
    }
    if (cloud_adjustment_only &&
        (rain_freezing || frozen_species || joint_frozen || melting_only || ice_melting_only || snow_melting_only || warm_primary_only))
    {
        error = "internal Milbrandt CloudAdjustmentOnly selected category is invalid";
        return false;
    }
    if (warm_primary_only && (rain_freezing || frozen_species || joint_frozen || melting_only || ice_melting_only || snow_melting_only))
    {
        error = "internal Milbrandt WarmPrimaryOnly selected category is invalid";
        return false;
    }
    if (snow_melting_only && (rain_freezing || frozen_species || joint_frozen || melting_only || ice_melting_only))
    {
        error = "internal Milbrandt SnowMeltingOnly selected category is invalid";
        return false;
    }
    if (ice_melting_only && (rain_freezing || frozen_species || joint_frozen || melting_only))
    {
        error = "internal Milbrandt IceMeltingOnly selected category is invalid";
        return false;
    }
    if (melting_only && (rain_freezing || frozen_species || joint_frozen))
    {
        error = "internal Milbrandt MeltingOnly selected category is invalid";
        return false;
    }
    if (joint_frozen && (rain_freezing || frozen_species))
    {
        error = "internal Milbrandt FrozenCompositionOnly selected category is invalid";
        return false;
    }
    const bool frozen_diffusion = frozen_species.has_value();
    if (frozen_diffusion &&
        (rain_freezing ||
         (*frozen_species != milbrandt_detail::TripleMomentSpecies::Graupel &&
          *frozen_species != milbrandt_detail::TripleMomentSpecies::Hail)))
    {
        error = "internal Milbrandt FrozenDiffusionOnly selected category is invalid";
        return false;
    }
    if (!state.storage_owner || !state.configuration ||
        !state.compute_runtime || !state.boundary_layer_runtime ||
        !state.radiation_runtime || !state.chaos_runtime ||
        !state.terrain_runtime ||
        state.grid.geometry.coord != CoordinateSystem::Cartesian ||
        state.grid.geometry.staggered ||
        state.grid.nr <= 0 || state.grid.nth <= 0 || state.grid.nz <= 0 ||
        state.grid.nz > 4096 ||
        state.compute_runtime->active_kind != ComputeBackendKind::Cpu ||
        state.compute_runtime->requested_kind != ComputeBackendKind::Cpu ||
        state.compute_runtime->fallback_active ||
        state.configuration->microphysics_scheme != "milbrandt" ||
        state.configuration->microphysics_config.milbrandt.triple_moment ||
        ((joint_frozen || melting_only || ice_melting_only || snow_melting_only || warm_primary_only || cloud_adjustment_only || warm_cloud_primary_only) &&
         !state.configuration->microphysics_config.milbrandt.hail_processes) ||
        state.configuration->validation_policy.mode != GuardMode::Strict ||
        state.configuration->validation_policy.fail_on != GuardFailOn::Both)
    {
        error = cold_cloud_collection_primary_only
            ? std::string(cold_cloud_label) + "requires exact flat strict CPU profile"
            : frozen_collection_primary_only
            ? "internal Milbrandt FrozenCollectionPrimaryOnly requires exact flat strict CPU profile"
            : warm_cloud_primary_only
            ? std::string(warm_cloud_label) + "requires exact flat strict CPU profile"
            : cloud_adjustment_only
            ? "internal Milbrandt CloudAdjustmentOnly requires exact flat strict CPU profile"
            : warm_primary_only
            ? "internal Milbrandt WarmPrimaryOnly requires exact flat strict CPU profile"
            : snow_melting_only
            ? "internal Milbrandt SnowMeltingOnly requires exact flat strict CPU profile"
            : ice_melting_only
            ? "internal Milbrandt IceMeltingOnly requires exact flat strict CPU profile"
            : melting_only
            ? "internal Milbrandt MeltingOnly requires exact flat strict CPU profile"
            : joint_frozen
            ? "internal Milbrandt FrozenCompositionOnly requires exact flat strict CPU profile"
            : frozen_diffusion
            ? "internal Milbrandt FrozenDiffusionOnly requires exact flat strict CPU profile"
            : rain_freezing
            ? "internal Milbrandt RainFreezingOnly requires exact flat strict CPU profile"
            : "internal Milbrandt WarmOnly requires exact flat strict CPU profile";
        return false;
    }
    const std::size_t nr = static_cast<std::size_t>(state.grid.nr);
    const std::size_t nth = static_cast<std::size_t>(state.grid.nth);
    const std::size_t nz = static_cast<std::size_t>(state.grid.nz);
    if (nr > 4096 / nth || nr * nth > 4096 / nz ||
        state.boundary_layer_runtime->config.scheme_id != "none" ||
        state.boundary_layer_runtime->scheme != nullptr ||
        state.radiation_runtime->config.scheme_id != "none" ||
        state.radiation_runtime->scheme != nullptr ||
        state.chaos_runtime->config.scheme_id != "none" ||
        state.chaos_runtime->scheme != nullptr ||
        state.terrain_runtime->config.scheme_id != "none" ||
        state.terrain_runtime->scheme != nullptr ||
        state.terrain_runtime->earth_fixed_placement.active ||
        !state.terrain_runtime->metrics.z.values.empty() ||
        state.radiation_runtime->held_atmospheric_power_w != 0.0 ||
        state.radiation_runtime->held_shortwave_incoming_power_w != 0.0 ||
        state.radiation_runtime->held_shortwave_atmospheric_absorbed_power_w != 0.0 ||
        state.radiation_runtime->held_surface_absorbed_power_w != 0.0 ||
        state.radiation_runtime->held_top_escape_power_w != 0.0 ||
        state.radiation_runtime->held_lateral_escape_power_w != 0.0 ||
        state.radiation_runtime->held_unresolved_reflected_power_w != 0.0 ||
        !zero_field(state.radiation_runtime->held_cell_power_w) ||
        !zero_field(state.tendencies.dtheta_dt_rad) ||
        !zero_field(state.tendencies.dtheta_dt_pbl) ||
        !zero_field(state.tendencies.dqv_dt_pbl) ||
        !zero_optional(state.tendencies.dqc_dt_pbl) ||
        !zero_optional(state.tendencies.dqi_dt_pbl) ||
        !zero_field(state.boundary_layer_runtime->held_u_force_n) ||
        !zero_field(state.boundary_layer_runtime->held_v_force_n) ||
        !zero_field(state.boundary_layer_runtime->held_theta_inventory_rate_kgk_s) ||
        !zero_field(state.boundary_layer_runtime->held_vapor_mass_rate_kg_s) ||
        !zero_field(state.boundary_layer_runtime->held_cloud_liquid_mass_rate_kg_s) ||
        !zero_field(state.boundary_layer_runtime->held_cloud_ice_mass_rate_kg_s) ||
        state.chaos_runtime->noise_elapsed_s != 0.0 ||
        state.chaos_runtime->pbl_base_tendencies_valid ||
        !zero_field(state.chaos_runtime->base_du_dt_pbl) ||
        !zero_field(state.chaos_runtime->base_dv_dt_pbl) ||
        !zero_field(state.chaos_runtime->base_dtheta_dt_pbl) ||
        !zero_field(state.chaos_runtime->base_dqv_dt_pbl) ||
        !zero_field(state.chaos_runtime->base_dtke_dt_pbl) ||
        (state.domain_frame &&
         (state.domain_frame->velocity_ms.u_ms != 0.0 ||
          state.domain_frame->velocity_ms.v_ms != 0.0)))
    {
        error = cold_cloud_collection_primary_only
            ? std::string(cold_cloud_label) + "has active held or mapped source"
            : frozen_collection_primary_only
            ? "internal Milbrandt FrozenCollectionPrimaryOnly has active held or mapped source"
            : warm_cloud_primary_only
            ? std::string(warm_cloud_label) + "has active held or mapped source"
            : cloud_adjustment_only
            ? "internal Milbrandt CloudAdjustmentOnly has active held or mapped source"
            : warm_primary_only
            ? "internal Milbrandt WarmPrimaryOnly has active held or mapped source"
            : snow_melting_only
            ? "internal Milbrandt SnowMeltingOnly has active held or mapped source"
            : ice_melting_only
            ? "internal Milbrandt IceMeltingOnly has active held or mapped source"
            : melting_only
            ? "internal Milbrandt MeltingOnly has active held or mapped source"
            : joint_frozen
            ? "internal Milbrandt FrozenCompositionOnly has active held or mapped source"
            : frozen_diffusion
            ? "internal Milbrandt FrozenDiffusionOnly has active held or mapped source"
            : rain_freezing
            ? "internal Milbrandt RainFreezingOnly has active held or mapped source"
            : "internal Milbrandt WarmOnly has active held or mapped source";
        return false;
    }
    const auto& f = state.fields;
    if (f.rho.size_r() != state.grid.nr ||
        f.rho.size_th() != state.grid.nth ||
        f.rho.size_z() != state.grid.nz)
    {
        error = cold_cloud_collection_primary_only
            ? std::string(cold_cloud_label) + "field shape differs from its grid"
            : frozen_collection_primary_only
            ? "internal Milbrandt FrozenCollectionPrimaryOnly field shape differs from its grid"
            : warm_cloud_primary_only
            ? std::string(warm_cloud_label) + "field shape differs from its grid"
            : cloud_adjustment_only
            ? "internal Milbrandt CloudAdjustmentOnly field shape differs from its grid"
            : warm_primary_only
            ? "internal Milbrandt WarmPrimaryOnly field shape differs from its grid"
            : snow_melting_only
            ? "internal Milbrandt SnowMeltingOnly field shape differs from its grid"
            : ice_melting_only
            ? "internal Milbrandt IceMeltingOnly field shape differs from its grid"
            : melting_only
            ? "internal Milbrandt MeltingOnly field shape differs from its grid"
            : joint_frozen
            ? "internal Milbrandt FrozenCompositionOnly field shape differs from its grid"
            : "internal Milbrandt process field shape differs from its grid";
        return false;
    }
    for (const Field3D* field : {&f.p, &f.theta, &f.qv, &f.qc,
                                &f.qr, &f.qi, &f.qs, &f.qg, &f.qh})
    {
        if (!matches(f.rho, *field))
        {
            error = cold_cloud_collection_primary_only
                ? std::string(cold_cloud_label) + "field shape differs from density"
                : frozen_collection_primary_only
                ? "internal Milbrandt FrozenCollectionPrimaryOnly field shape differs from density"
                : warm_cloud_primary_only
                ? std::string(warm_cloud_label) + "field shape differs from density"
                : cloud_adjustment_only
                ? "internal Milbrandt CloudAdjustmentOnly field shape differs from density"
                : warm_primary_only
                ? "internal Milbrandt WarmPrimaryOnly field shape differs from density"
                : snow_melting_only
                ? "internal Milbrandt SnowMeltingOnly field shape differs from density"
                : ice_melting_only
                ? "internal Milbrandt IceMeltingOnly field shape differs from density"
                : melting_only
                ? "internal Milbrandt MeltingOnly field shape differs from density"
                : joint_frozen
                ? "internal Milbrandt FrozenCompositionOnly field shape differs from density"
                : frozen_diffusion
                ? "internal Milbrandt FrozenDiffusionOnly field shape differs from density"
                : rain_freezing
                ? "internal Milbrandt RainFreezingOnly field shape differs from density"
                : "internal Milbrandt WarmOnly field shape differs from density";
            return false;
        }
    }
    if (cloud_adjustment_only || warm_cloud_primary_only)
    {
        if (cloud_adjustment_only && !zero_field(f.qr))
        {
            error = "internal Milbrandt CloudAdjustmentOnly has an unowned rain category";
            return false;
        }
        const auto canonical = make_simulation_state_view(*state.storage_owner);
        if (&f.w != &canonical.fields.w || !matches(f.rho, f.w))
        {
            error = warm_cloud_primary_only
                ? std::string(warm_cloud_label) + "wind owner or shape differs"
                : "internal Milbrandt CloudAdjustmentOnly wind owner or shape differs";
            return false;
        }
        for (std::size_t index = 0; index < f.w.size(); ++index)
        {
            if (!std::isfinite(f.w.data()[index]))
            {
                error = warm_cloud_primary_only
                    ? std::string(warm_cloud_label) + "wind is invalid"
                    : "internal Milbrandt CloudAdjustmentOnly wind is invalid";
                return false;
            }
        }
    }
    if (cold_cloud_collection_primary_only)
    {
        if ((!cold_rain_primary_only && !zero_field(f.qr)) || (!cold_riming_primary_only && !zero_field(f.qi)))
        {
            error = std::string(cold_cloud_label) + "has an unowned rain or ice category";
            return false;
        }
    }
    else if (frozen_collection_primary_only)
    {
        if (!zero_field(f.qc) || !zero_field(f.qr))
        {
            error = "internal Milbrandt FrozenCollectionPrimaryOnly has an unowned liquid category";
            return false;
        }
    }
    else if (snow_melting_only)
    {
        if (!zero_field(f.qc) || !zero_field(f.qi) || !zero_field(f.qg) || !zero_field(f.qh))
        {
            error = "internal Milbrandt SnowMeltingOnly has an unowned water category";
            return false;
        }
    }
    else if (ice_melting_only)
    {
        if (!zero_field(f.qc) || !zero_field(f.qs) || !zero_field(f.qg) || !zero_field(f.qh))
        {
            error = "internal Milbrandt IceMeltingOnly has an unowned water category";
            return false;
        }
    }
    else if (melting_only)
    {
        if (!zero_field(f.qc) || !zero_field(f.qi) || !zero_field(f.qs))
        {
            error = "internal Milbrandt MeltingOnly has an unowned water category";
            return false;
        }
    }
    else if (joint_frozen)
    {
        if (!zero_field(f.qc) || !zero_field(f.qr) ||
            !zero_field(f.qi) || !zero_field(f.qs))
        {
            error = "internal Milbrandt FrozenCompositionOnly has an unowned water category";
            return false;
        }
    }
    else if (frozen_diffusion)
    {
        const Field3D& other_frozen =
            *frozen_species == milbrandt_detail::TripleMomentSpecies::Graupel
                ? f.qh : f.qg;
        if (!zero_field(f.qc) || !zero_field(f.qr) ||
            !zero_field(f.qi) || !zero_field(f.qs) ||
            !zero_field(other_frozen))
        {
            error = "internal Milbrandt FrozenDiffusionOnly has an unowned water category";
            return false;
        }
    }
    else if ((rain_freezing && !zero_field(f.qc)) ||
        !zero_field(f.qi) || !zero_field(f.qs) ||
        !zero_field(f.qg) || !zero_field(f.qh))
    {
        error = warm_cloud_primary_only
            ? std::string(warm_cloud_label) + "has another precipitating category"
            : cloud_adjustment_only
            ? "internal Milbrandt CloudAdjustmentOnly has another precipitating category"
            : warm_primary_only
            ? "internal Milbrandt WarmPrimaryOnly has another precipitating category"
            : rain_freezing
            ? "internal Milbrandt RainFreezingOnly has an unowned cloud or frozen category"
            : "internal Milbrandt WarmOnly has another precipitating category";
        return false;
    }
    for (std::size_t index = 0; index < f.rho.size(); ++index)
    {
        const double pressure = f.p.data()[index];
        const double raw_theta = f.theta.data()[index];
        double temperature = 0.0;
        if (!milbrandt_detail::internal_triple_raw_temperature(raw_theta, pressure, temperature) ||
            !std::isfinite(pressure) || pressure < 100.0 ||
            pressure > 200000.0 || !std::isfinite(temperature) ||
            ((melting_only || ice_melting_only || snow_melting_only) ? temperature < microphysics_constants::T0
                : (rain_freezing || frozen_diffusion || joint_frozen || frozen_collection_primary_only || cold_cloud_collection_primary_only)
                ? temperature < 150.0 || temperature >= microphysics_constants::T0
                : temperature <= microphysics_constants::T0) ||
            ((cloud_adjustment_only || warm_cloud_primary_only) && temperature >= 400.0) ||
            !std::isfinite(f.rho.data()[index]) ||
            f.rho.data()[index] <= 0.0f)
        {
            error = cold_cloud_collection_primary_only
                ? std::string(cold_cloud_label) + "pressure, temperature or density is invalid"
                : frozen_collection_primary_only
                ? "internal Milbrandt FrozenCollectionPrimaryOnly pressure, temperature or density is invalid"
                : warm_cloud_primary_only
                ? std::string(warm_cloud_label) + "pressure, temperature or density is invalid"
                : cloud_adjustment_only
                ? "internal Milbrandt CloudAdjustmentOnly pressure, temperature or density is invalid"
                : warm_primary_only
                ? "internal Milbrandt WarmPrimaryOnly pressure, temperature or density is invalid"
                : snow_melting_only
                ? "internal Milbrandt SnowMeltingOnly pressure, temperature or density is invalid"
                : ice_melting_only
                ? "internal Milbrandt IceMeltingOnly pressure, temperature or density is invalid"
                : melting_only
                ? "internal Milbrandt MeltingOnly pressure, temperature or density is invalid"
                : joint_frozen
                ? "internal Milbrandt FrozenCompositionOnly pressure, temperature or density is invalid"
                : frozen_diffusion
                ? "internal Milbrandt FrozenDiffusionOnly pressure, temperature or density is invalid"
                : rain_freezing
                ? "internal Milbrandt RainFreezingOnly pressure, temperature or density is invalid"
                : "internal Milbrandt WarmOnly pressure, temperature or density is invalid";
            return false;
        }
        for (const Field3D* water : {&f.qv, &f.qc, &f.qr})
        {
            if (!std::isfinite(water->data()[index]) ||
                water->data()[index] < 0.0f)
            {
                error = cold_cloud_collection_primary_only
                    ? std::string(cold_cloud_label) + "water input is invalid"
                    : frozen_collection_primary_only
                    ? "internal Milbrandt FrozenCollectionPrimaryOnly water input is invalid"
                    : warm_cloud_primary_only
                    ? std::string(warm_cloud_label) + "water input is invalid"
                    : cloud_adjustment_only
                    ? "internal Milbrandt CloudAdjustmentOnly water input is invalid"
                    : warm_primary_only
                    ? "internal Milbrandt WarmPrimaryOnly water input is invalid"
                    : snow_melting_only
                    ? "internal Milbrandt SnowMeltingOnly water input is invalid"
                    : ice_melting_only
                    ? "internal Milbrandt IceMeltingOnly water input is invalid"
                    : melting_only
                    ? "internal Milbrandt MeltingOnly water input is invalid"
                    : joint_frozen
                    ? "internal Milbrandt FrozenCompositionOnly water input is invalid"
                    : frozen_diffusion
                    ? "internal Milbrandt FrozenDiffusionOnly water input is invalid"
                    : rain_freezing
                    ? "internal Milbrandt RainFreezingOnly water input is invalid"
                    : "internal Milbrandt WarmOnly water input is invalid";
                return false;
            }
        }
        if (cold_cloud_collection_primary_only)
        {
            for (const Field3D* water : {&f.qi, &f.qs, &f.qg, &f.qh})
            {
                if (!std::isfinite(water->data()[index]) || water->data()[index] < 0.0f)
                {
                    error = std::string(cold_cloud_label) + "water input is invalid";
                    return false;
                }
            }
        }
        if (frozen_collection_primary_only)
        {
            for (const Field3D* water : {&f.qi, &f.qs, &f.qg, &f.qh})
            {
                if (!std::isfinite(water->data()[index]) || water->data()[index] < 0.0f)
                {
                    error = "internal Milbrandt FrozenCollectionPrimaryOnly water input is invalid";
                    return false;
                }
            }
        }
        if (joint_frozen || melting_only)
        {
            for (const Field3D* water : {&f.qg, &f.qh})
            {
                if (!std::isfinite(water->data()[index]) ||
                    water->data()[index] < 0.0f)
                {
                    error = melting_only
                        ? "internal Milbrandt MeltingOnly water input is invalid"
                        : "internal Milbrandt FrozenCompositionOnly water input is invalid";
                    return false;
                }
            }
        }
        if (ice_melting_only &&
            (!std::isfinite(f.qi.data()[index]) || f.qi.data()[index] < 0.0f))
        {
            error = "internal Milbrandt IceMeltingOnly water input is invalid";
            return false;
        }
        if (snow_melting_only &&
            (!std::isfinite(f.qs.data()[index]) || f.qs.data()[index] < 0.0f))
        {
            error = "internal Milbrandt SnowMeltingOnly water input is invalid";
            return false;
        }
        if (frozen_diffusion)
        {
            const Field3D& selected =
                *frozen_species == milbrandt_detail::TripleMomentSpecies::Graupel
                    ? f.qg : f.qh;
            if (!std::isfinite(selected.data()[index]) ||
                selected.data()[index] < 0.0f)
            {
                error = "internal Milbrandt FrozenDiffusionOnly water input is invalid";
                return false;
            }
        }
    }
    error.clear();
    return true;
}

namespace
{

bool within_two_ulps(float actual, float expected) noexcept
{
    if (!std::isfinite(actual) || !std::isfinite(expected))
    {
        return false;
    }
    if (actual == expected)
    {
        return true;
    }
    const auto ulp = [](float value)
    {
        return std::abs(static_cast<double>(
            std::nextafter(value, std::numeric_limits<float>::infinity()))
            - static_cast<double>(value));
    };
    return std::abs(static_cast<double>(actual) - expected)
        <= 2.0 * std::max(ulp(actual), ulp(expected));
}

} // namespace

bool MilbrandtScheme::enable_internal_triple_warm_for_test(
    const SimulationStateView& state,
    const Field3D& cloud_number,
    const Field3D& rain_number,
    const Field3D& rain_reflectivity,
    std::string& error)
{
    return enable_internal_triple_warm_mode_for_test(
        state, cloud_number, rain_number, rain_reflectivity,
        InternalTripleMode::WarmOnly, error);
}

bool MilbrandtScheme::enable_internal_triple_warm_rain_sedimentation_for_test(
    const SimulationStateView& state,
    const Field3D& cloud_number,
    const Field3D& rain_number,
    const Field3D& rain_reflectivity,
    std::string& error)
{
    return enable_internal_triple_warm_mode_for_test(
        state, cloud_number, rain_number, rain_reflectivity,
        InternalTripleMode::WarmRainSedimentation, error);
}

bool MilbrandtScheme::enable_internal_triple_warm_mode_for_test(
    const SimulationStateView& state,
    const Field3D& cloud_number,
    const Field3D& rain_number,
    const Field3D& rain_reflectivity,
    InternalTripleMode mode,
    std::string& error)
{
    if (internal_triple_ ||
        !validate_internal_triple_source_profile(state, false, error))
    {
        if (error.empty())
        {
            error = "internal Milbrandt triple modes are mutually exclusive";
        }
        return false;
    }
    const auto& f = state.fields;
    if (mode == InternalTripleMode::WarmRainSedimentation)
    {
        const std::size_t levels = static_cast<std::size_t>(state.grid.nz);
        const std::size_t substeps = std::min<std::size_t>(
            4096, 65536 / levels);
        if (f.rho.size() > 2097152 / (5 * substeps))
        {
            error = "internal Milbrandt warm-rain sedimentation exceeds aggregate work budget";
            return false;
        }
    }
    if (!matches(f.rho, cloud_number) ||
        !matches(f.rho, rain_number) ||
        !matches(f.rho, rain_reflectivity) ||
        alpha_r_ != 0.0 || c_r_ != 523.598775598 ||
        d_r_ != 3.0 || a_r_ != 149.1 || b_r_ != 0.5)
    {
        error = "internal Milbrandt WarmOnly moment shape or rain law differs";
        return false;
    }
    for (const Field3D* other : {&Ni_, &Ns_, &Ng_, &Nh_})
    {
        if (!other->empty() &&
            (!matches(f.rho, *other) || !zero_field(*other)))
        {
            error = "internal Milbrandt WarmOnly has retained other number moments";
            return false;
        }
    }
    for (std::size_t index = 0; index < f.rho.size(); ++index)
    {
        const double nc = cloud_number.data()[index];
        const double nr = rain_number.data()[index];
        const double zr = rain_reflectivity.data()[index];
        if (!std::isfinite(nc) || nc < 0.0 ||
            (f.qc.data()[index] > 0.0f && nc <= 0.0))
        {
            error = "internal Milbrandt WarmOnly cloud number is invalid";
            return false;
        }
        if (f.qr.data()[index] == 0.0f && nr == 0.0 && zr == 0.0)
        {
            continue;
        }
        double alpha = 0.0;
        double lambda = 0.0;
        if (!milbrandt_detail::diagnose_triple_gamma_shape(
                f.qr.data()[index], nr, zr,
                {c_r_, d_r_}, alpha, lambda, error))
        {
            return false;
        }
    }
    MilbrandtScheme candidate(*this);
    candidate.Nc_ = cloud_number;
    candidate.Nr_ = rain_number;
    for (Field3D* other : {&candidate.Ni_, &candidate.Ns_,
                           &candidate.Ng_, &candidate.Nh_})
    {
        other->resize(f.rho.size_r(), f.rho.size_th(),
                      f.rho.size_z(), 0.0f);
    }
    for (Field3D* tendency : {&candidate.dNc_dt_, &candidate.dNr_dt_,
                              &candidate.dNi_dt_, &candidate.dNs_dt_,
                              &candidate.dNg_dt_, &candidate.dNh_dt_})
    {
        tendency->resize(f.rho.size_r(), f.rho.size_th(),
                         f.rho.size_z(), 0.0f);
    }
    candidate.surface_sedimentation_sink_rate_.resize(
        f.rho.size_r(), f.rho.size_th(), 1, 0.0f);
    InternalTripleCandidateState internal;
    internal.mode = mode;
    internal.warm_storage_owner = state.storage_owner;
    internal.reflectivity[0] = rain_reflectivity;
    for (std::size_t species = 1; species < 5; ++species)
    {
        internal.reflectivity[species].resize(
            f.rho.size_r(), f.rho.size_th(), f.rho.size_z(), 0.0f);
    }
    candidate.internal_triple_ = std::move(internal);
    *this = std::move(candidate);
    error.clear();
    return true;
}

const Field3D& MilbrandtScheme::internal_triple_cloud_number_for_test() const
{
    if (!internal_triple_ ||
        internal_triple_->mode == InternalTripleMode::SedimentationOnly)
    {
        throw std::out_of_range("internal Milbrandt warm cloud number is unavailable");
    }
    return Nc_;
}

void MilbrandtScheme::compute_internal_triple_warm_tendencies(
    const Field3D& p,
    const Field3D& theta,
    const Field3D& density,
    const Field3D& vapor,
    const Field3D& cloud,
    const std::array<const Field3D*, 5>& mass,
    double dt,
    const std::array<Field3D*, 8>& tendencies)
{
    auto& internal = *internal_triple_;
    internal.candidate_valid = false;
    internal.prepared = false;
    if ((internal.mode != InternalTripleMode::WarmOnly &&
         internal.mode != InternalTripleMode::WarmRainSedimentation) ||
        internal.poisoned || !internal.warm_storage_owner ||
        !std::isfinite(dt) || dt <= 0.0)
    {
        throw std::runtime_error(
            "internal Milbrandt WarmOnly candidate is unavailable");
    }
    auto live = make_simulation_state_view(*internal.warm_storage_owner);
    std::string error;
    if (&live.fields.p != &p || &live.fields.theta != &theta ||
        &live.fields.rho != &density || &live.fields.qv != &vapor ||
        &live.fields.qc != &cloud || &live.fields.qr != mass[0] ||
        &live.fields.qi != mass[1] || &live.fields.qs != mass[2] ||
        &live.fields.qg != mass[3] || &live.fields.qh != mass[4] ||
        !validate_internal_triple_source_profile(live, false, error))
    {
        throw std::runtime_error(
            "internal Milbrandt WarmOnly runtime owner changed: " + error);
    }
    if (!matches(density, Nc_) || !matches(density, Nr_) ||
        !matches(density, internal.reflectivity[0]))
    {
        throw std::runtime_error(
            "internal Milbrandt WarmOnly private moment shape changed");
    }
    for (const Field3D* other : {&Ni_, &Ns_, &Ng_, &Nh_,
                                 &internal.reflectivity[1],
                                 &internal.reflectivity[2],
                                 &internal.reflectivity[3],
                                 &internal.reflectivity[4]})
    {
        if (!matches(density, *other) || !zero_field(*other))
        {
            throw std::runtime_error(
                "internal Milbrandt WarmOnly gained another precipitating moment");
        }
    }
    std::array<Field3D, 8> candidate_rates;
    for (auto& field : candidate_rates)
    {
        field.resize(density.size_r(), density.size_th(),
                     density.size_z(), 0.0f);
    }
    std::array<Field3D, 5> mass_after;
    std::array<Field3D, 5> number_after;
    std::array<Field3D, 5> reflectivity_after;
    const std::array<const Field3D*, 5> number{
        &Nr_, &Ni_, &Ns_, &Ng_, &Nh_};
    for (std::size_t species = 0; species < 5; ++species)
    {
        mass_after[species] = *mass[species];
        number_after[species] = *number[species];
        reflectivity_after[species] = internal.reflectivity[species];
    }
    Field3D cloud_number_after(Nc_);
    Field3D vapor_after(vapor);
    Field3D cloud_after(cloud);
    Field3D density_before(density);
    const auto parameters = milbrandt_detail::warm_rain_parameters(
        alpha_r_, c_r_, d_r_, a_r_, b_r_);
    const auto domain = physical_cell_domain(
        density.size_r(), density.size_th(), density.size_z());
    const int levels = domain.k_end - domain.k_begin;
    std::size_t maximum_substeps = 0;
    if (internal.mode == InternalTripleMode::WarmRainSedimentation)
    {
        if (levels <= 0 || levels > 4096)
        {
            throw std::runtime_error(
                "internal Milbrandt warm-rain sedimentation column extent is invalid");
        }
        maximum_substeps = std::min<std::size_t>(
            4096, 65536 / static_cast<std::size_t>(levels));
        if (density.size() > 2097152 / (5 * maximum_substeps))
        {
            throw std::runtime_error(
                "internal Milbrandt warm-rain sedimentation exceeds aggregate work budget");
        }
    }
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double temperature =
                    thermodynamics::theta_to_temperature(
                        theta(i, j, k), p(i, j, k));
                const milbrandt_detail::WarmRainCell cell{
                    static_cast<float>(temperature), p(i, j, k),
                    density(i, j, k), vapor(i, j, k), cloud(i, j, k),
                    (*mass[0])(i, j, k), Nc_(i, j, k), Nr_(i, j, k)};
                const milbrandt_detail::RainTripleMomentState initial{
                    (*mass[0])(i, j, k), Nr_(i, j, k),
                    internal.reflectivity[0](i, j, k)};
                milbrandt_detail::WarmRainTendencies rates{};
                milbrandt_detail::RainTripleMomentState result{};
                std::string cell_error;
                if (!milbrandt_detail::evaluate_triple_only_warm_cell(
                        parameters, cell, dt, {}, initial,
                        rates, result, cell_error))
                {
                    throw std::runtime_error(
                        "internal Milbrandt WarmOnly cell rejected: " +
                        cell_error);
                }
                const double qv_next = static_cast<double>(vapor(i, j, k))
                    + dt * rates.vapor;
                const double qc_next = static_cast<double>(cloud(i, j, k))
                    + dt * rates.cloud;
                const double qr_next = static_cast<double>((*mass[0])(i, j, k))
                    + dt * rates.rain;
                const double nc_next = static_cast<double>(Nc_(i, j, k))
                    + dt * rates.cloud_number;
                const double dtheta =
                    thermodynamics::temperature_tendency_to_theta(
                        rates.temperature, p(i, j, k));
                const double final_temperature = temperature
                    + dt * static_cast<double>(rates.temperature);
                const double theta_unrounded = static_cast<double>(theta(i, j, k))
                    + dt * static_cast<float>(dtheta);
                const float theta_stored = static_cast<float>(theta_unrounded);
                double unrounded_temperature = 0.0;
                double stored_temperature = 0.0;
                if (!std::isfinite(final_temperature) ||
                    final_temperature <= microphysics_constants::T0 ||
                    !milbrandt_detail::internal_triple_raw_temperature_pair(theta_unrounded, theta_stored,
                        p(i, j, k), unrounded_temperature, stored_temperature) ||
                    unrounded_temperature <= microphysics_constants::T0 ||
                    stored_temperature <= microphysics_constants::T0)
                {
                    throw std::runtime_error(
                        "internal Milbrandt WarmOnly final temperature exits the warm regime");
                }
                if (!std::isfinite(qv_next) || qv_next < 0.0 ||
                    !std::isfinite(qc_next) || qc_next < 0.0 ||
                    !std::isfinite(qr_next) || qr_next < 0.0 ||
                    !std::isfinite(nc_next) || nc_next < 0.0 ||
                    !std::isfinite(dtheta) ||
                    std::abs(dtheta) > std::numeric_limits<float>::max() ||
                    qv_next > std::numeric_limits<float>::max() ||
                    qc_next > std::numeric_limits<float>::max() ||
                    qr_next > std::numeric_limits<float>::max() ||
                    nc_next > std::numeric_limits<float>::max() ||
                    (qc_next > 0.0 && nc_next <= 0.0) ||
                    !within_two_ulps(
                        static_cast<float>(qr_next), result.mass_per_kg))
                {
                    throw std::runtime_error(
                        "internal Milbrandt WarmOnly rounded q/N/theta candidate is invalid");
                }
                double alpha = 0.0;
                double lambda = 0.0;
                if (!(qr_next == 0.0 &&
                      result.number_per_kg == 0.0f &&
                      result.reflectivity_per_kg == 0.0f) &&
                    !milbrandt_detail::diagnose_triple_gamma_shape(
                        static_cast<float>(qr_next),
                        result.number_per_kg,
                        result.reflectivity_per_kg,
                        {c_r_, d_r_}, alpha, lambda, cell_error))
                {
                    throw std::runtime_error(
                        "internal Milbrandt WarmOnly rounded rain moment is invalid: "
                        + cell_error);
                }
                candidate_rates[0](i, j, k) = static_cast<float>(dtheta);
                candidate_rates[1](i, j, k) = rates.vapor;
                candidate_rates[2](i, j, k) = rates.cloud;
                candidate_rates[3](i, j, k) = rates.rain;
                vapor_after(i, j, k) = static_cast<float>(qv_next);
                cloud_after(i, j, k) = static_cast<float>(qc_next);
                mass_after[0](i, j, k) = static_cast<float>(qr_next);
                cloud_number_after(i, j, k) = static_cast<float>(nc_next);
                number_after[0](i, j, k) = result.number_per_kg;
                reflectivity_after[0](i, j, k) =
                    result.reflectivity_per_kg;
            }
        }
    }
    Field3D surface_sink(
        density.size_r(), density.size_th(), 1, 0.0f);
    std::vector<std::array<milbrandt_detail::TripleMomentSurfaceInventory, 5>>
        surface(static_cast<std::size_t>(density.size_r()) * density.size_th());
    if (internal.mode == InternalTripleMode::WarmRainSedimentation)
    {
        std::array<Field3D, 3> after_sedimentation{
            mass_after[0], number_after[0], reflectivity_after[0]};
        std::vector<double> thickness(static_cast<std::size_t>(levels));
        auto sedimentation =
            milbrandt_detail::current_nimbus_triple_sedimentation_parameters(
                milbrandt_detail::TripleMomentSpecies::Rain);
        sedimentation.maximum_substeps =
            static_cast<int>(maximum_substeps);
        for (int i = domain.i_begin; i < domain.i_end; ++i)
        {
            for (int j = domain.j_begin; j < domain.j_end; ++j)
            {
                const std::size_t column = static_cast<std::size_t>(i)
                    * density.size_th() + static_cast<std::size_t>(j);
                const std::size_t base = column * density.size_z()
                    + static_cast<std::size_t>(domain.k_begin);
                for (int level = 0; level < levels; ++level)
                {
                    thickness[static_cast<std::size_t>(level)] =
                        vertical_spacing_m(i, j, domain.k_begin + level);
                }
                std::string column_error;
                if (!milbrandt_detail::sediment_triple_moment_column(
                        levels,
                        mass_after[0].data() + base,
                        number_after[0].data() + base,
                        reflectivity_after[0].data() + base,
                        density.data() + base, thickness.data(),
                        dt, sedimentation,
                        after_sedimentation[0].data() + base,
                        after_sedimentation[1].data() + base,
                        after_sedimentation[2].data() + base,
                        surface[column][0], column_error))
                {
                    throw std::runtime_error(
                        "internal Milbrandt warm-rain sedimentation rejected column: "
                        + column_error);
                }
                const double denominator =
                    static_cast<double>(density(i, j, domain.k_begin))
                    * thickness.front() * dt;
                const double sink = surface[column][0].mass / denominator;
                if (!std::isfinite(denominator) || denominator <= 0.0 ||
                    !std::isfinite(sink) || sink < 0.0 ||
                    sink > std::numeric_limits<float>::max())
                {
                    throw std::runtime_error(
                        "internal Milbrandt warm-rain surface mass rate is invalid");
                }
                surface_sink(i, j, 0) = static_cast<float>(sink);
            }
        }
        for (std::size_t index = 0; index < density.size(); ++index)
        {
            const double rate =
                (static_cast<double>(after_sedimentation[0].data()[index])
                 - mass[0]->data()[index]) / dt;
            if (!std::isfinite(rate) ||
                std::abs(rate) > std::numeric_limits<float>::max())
            {
                throw std::runtime_error(
                    "internal Milbrandt warm-rain total mass rate is invalid");
            }
            candidate_rates[3].data()[index] = static_cast<float>(rate);
        }
        mass_after[0] = std::move(after_sedimentation[0]);
        number_after[0] = std::move(after_sedimentation[1]);
        reflectivity_after[0] = std::move(after_sedimentation[2]);
    }
    for (std::size_t index = 0; index < candidate_rates.size(); ++index)
    {
        *tendencies[index] = std::move(candidate_rates[index]);
    }
    internal.mass_after = std::move(mass_after);
    internal.number_after = std::move(number_after);
    internal.reflectivity_after = std::move(reflectivity_after);
    internal.cloud_number_after = std::move(cloud_number_after);
    internal.vapor_after = std::move(vapor_after);
    internal.cloud_after = std::move(cloud_after);
    internal.density_before = std::move(density_before);
    internal.surface_sink_after = std::move(surface_sink);
    internal.surface_after = std::move(surface);
    internal.candidate_dt = dt;
    internal.candidate_valid = true;
}

} // namespace tmv
