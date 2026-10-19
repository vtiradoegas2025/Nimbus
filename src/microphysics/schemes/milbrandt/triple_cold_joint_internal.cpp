#include "milbrandt.hpp"
#include "triple_cold_diffusion.hpp"
#include "triple_cold_riming.hpp"
#include "triple_internal_thermal.hpp"

#include "core/runtime/simulation_state.hpp"
#include "core/runtime/runtime_config.hpp"
#include "diagnostics/diagnostic_cell_volume.hpp"
#include "microphysics/milbrandt_cold_parameters.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tmv
{
namespace
{

std::array<const Field3D*, 10> primary_fields(const SimulationStateView& state)
{
    const auto& f = state.fields;
    return {&f.rho, &f.p, &f.theta, &f.qv, &f.qc,
            &f.qr, &f.qi, &f.qs, &f.qg, &f.qh};
}

bool matches(const Field3D& reference, const Field3D& field) noexcept
{
    return field.size_r() == reference.size_r()
        && field.size_th() == reference.size_th()
        && field.size_z() == reference.size_z();
}

bool same_bytes(const Field3D& left, const Field3D& right) noexcept
{
    return matches(left, right) &&
        (left.empty() || std::memcmp(left.data(), right.data(),
                                    left.size() * sizeof(float)) == 0);
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

bool canonical_primary_fields(const SimulationStateView& state)
{
    if (!state.storage_owner)
    {
        return false;
    }
    const auto fields = primary_fields(state);
    if (fields != primary_fields(make_simulation_state_view(*state.storage_owner)))
    {
        return false;
    }
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        for (std::size_t other = index + 1; other < fields.size(); ++other)
        {
            if (fields[index] == fields[other])
            {
                return false;
            }
        }
    }
    return true;
}

bool same_record(const MicrophysicsPrimaryEndpointCandidate& left,
                 const MicrophysicsPrimaryEndpointCandidate& right) noexcept
{
    if (left.owner != right.owner || left.generation != right.generation ||
        left.duration_s != right.duration_s || left.nr != right.nr ||
        left.nth != right.nth || left.nz != right.nz ||
        left.source_fields != right.source_fields ||
        left.theta_increment.size() != right.theta_increment.size())
    {
        return false;
    }
    for (std::size_t species = 0; species < left.water_after.size(); ++species)
    {
        const auto& a = left.water_after[species];
        const auto& b = right.water_after[species];
        if (a.size() != b.size() ||
            (!a.empty() && std::memcmp(a.data(), b.data(),
                                      a.size() * sizeof(double)) != 0))
        {
            return false;
        }
    }
    return left.theta_increment.empty() ||
        std::memcmp(left.theta_increment.data(), right.theta_increment.data(),
                    left.theta_increment.size() * sizeof(double)) == 0;
}

bool representable_water(double value) noexcept
{
    return std::isfinite(value) && value >= 0.0 &&
        value <= std::numeric_limits<float>::max() &&
        (value == 0.0 || value >= std::numeric_limits<float>::denorm_min());
}

bool valid_triple(const Field3D& mass,
                  const Field3D& number,
                  const Field3D& reflectivity,
                  milbrandt_detail::TripleMomentSpecies species,
                  std::string& error,
                  const milbrandt_detail::ColdSpectrumDomain* spectrum_domain = nullptr)
{
    const auto law = milbrandt_detail::current_nimbus_triple_sedimentation_parameters(
        species).mass_law;
    for (std::size_t index = 0; index < mass.size(); ++index)
    {
        const double q = mass.data()[index];
        const double n = number.data()[index];
        const double z = reflectivity.data()[index];
        if (q == 0.0 && n == 0.0 && z == 0.0)
        {
            continue;
        }
        if (spectrum_domain)
        {
            if (!milbrandt_detail::validate_cold_riming_source_population({q, n, z}, species, *spectrum_domain, error))
            {
                return false;
            }
            continue;
        }
        double alpha = 0.0;
        double lambda = 0.0;
        if (!milbrandt_detail::diagnose_triple_gamma_shape(
                q, n, z, law, alpha, lambda, error))
        {
            return false;
        }
    }
    return true;
}

bool endpoint_attribution_matches(
    const SimulationStateView& state,
    PhysicalCellDomain domain,
    const MicrophysicsPrimaryEndpointCandidate& candidate,
    const std::array<Field3D, 10>& source,
    const MicrophysicsWaterRepairReport& water,
    const MicrophysicsThermodynamicRepairReport& heat,
    const char* label,
    std::string& error)
{
    std::array<double, 7> rounding{};
    std::array<double, 7> requested{};
    double requested_heat = 0.0;
    double theta_rounding = 0.0;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        std::array<double, 7> slab_rounding{};
        std::array<double, 7> slab_requested{};
        double slab_heat = 0.0;
        double slab_theta_rounding = 0.0;
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const std::size_t index = (static_cast<std::size_t>(i) * state.grid.nth + j)
                    * state.grid.nz + k;
                const double volume = diagnostic_cell_volume(state, i, j, k);
                const double density = source[0].data()[index];
                const double dry_mass = density * volume;
                const double exner = std::pow(
                    static_cast<double>(source[1].data()[index]) / microphysics_constants::p0,
                    microphysics_constants::R_d / microphysics_constants::cp);
                const double heat_factor = density * microphysics_constants::cp * exner * volume;
                if (!std::isfinite(volume) || volume <= 0.0 ||
                    !std::isfinite(dry_mass) || dry_mass <= 0.0 ||
                    !std::isfinite(heat_factor) || heat_factor <= 0.0)
                {
                    error = std::string(label) + "attribution geometry is invalid";
                    return false;
                }
                for (std::size_t species = 0; species < requested.size(); ++species)
                {
                    const double endpoint = candidate.water_after[species][index];
                    slab_rounding[species] +=
                        (static_cast<double>(static_cast<float>(endpoint)) - endpoint) * dry_mass;
                    slab_requested[species] +=
                        (endpoint - source[species + 3].data()[index]) * dry_mass;
                }
                const double increment = candidate.theta_increment[index];
                const double theta_unrounded = static_cast<double>(source[2].data()[index]) + increment;
                slab_heat += increment * heat_factor;
                slab_theta_rounding +=
                    (static_cast<double>(static_cast<float>(theta_unrounded)) - theta_unrounded) * heat_factor;
            }
        }
        for (std::size_t species = 0; species < requested.size(); ++species)
        {
            rounding[species] += slab_rounding[species];
            requested[species] += slab_requested[species];
        }
        requested_heat += slab_heat;
        theta_rounding += slab_theta_rounding;
    }
    double requested_total = 0.0;
    for (std::size_t species = 0; species < requested.size(); ++species)
    {
        const auto& repairs = water.species[species];
        if (!std::isfinite(rounding[species]) || !std::isfinite(requested[species]) ||
            repairs.storage_rounding_kg != rounding[species] ||
            repairs.lower_bound_correction_kg != 0.0 ||
            repairs.lower_bound_roundoff_correction_kg != 0.0 ||
            repairs.upper_bound_correction_kg != 0.0 ||
            repairs.nonfinite_correction_kg != 0.0 ||
            heat.microphysics_water_tendency_kg_by_species[species] != requested[species])
        {
            error = std::string(label) + "water attribution differs";
            return false;
        }
        requested_total += requested[species];
    }
    if (!std::isfinite(requested_heat) || !std::isfinite(theta_rounding) ||
        !std::isfinite(requested_total) || heat.microphysics_heat_j != requested_heat ||
        heat.theta_storage_rounding_j != theta_rounding ||
        heat.microphysics_water_tendency_kg != requested_total ||
        heat.radiation_heat_j != 0.0 || heat.boundary_layer_heat_j != 0.0 ||
        heat.theta_tendency_limit_correction_j != 0.0 ||
        heat.theta_nonfinite_correction_j != 0.0 || heat.theta_bound_correction_j != 0.0 ||
        heat.theta_storage_residual_release_j != 0.0 || heat.theta_storage_residual_j != 0.0 ||
        heat.boundary_layer_vapor_tendency_kg != 0.0 ||
        heat.boundary_layer_cloud_liquid_tendency_kg != 0.0 ||
        heat.boundary_layer_cloud_ice_tendency_kg != 0.0 || !heat.moist_energy_complete)
    {
        error = std::string(label) + "heat or source attribution differs";
        return false;
    }
    for (double value : {heat.moist_dry_internal_change_j, heat.moist_water_sensible_change_j,
                         heat.moist_phase_reference_change_j, heat.moist_carrier_kinetic_change_j,
                         heat.moist_gravitational_change_j, heat.moist_subgrid_tke_change_j,
                         heat.moist_committed_energy_change_j, heat.moist_energy_closure_residual_j})
    {
        if (!std::isfinite(value))
        {
            error = std::string(label) + "moist-energy attribution is invalid";
            return false;
        }
    }
    return true;
}

} // namespace

milbrandt_detail::ColdSpectrumDomain MilbrandtScheme::internal_triple_spectrum_domain(InternalTripleMode mode) noexcept
{
    if (mode == InternalTripleMode::ColdSllPrimaryOnly || mode == InternalTripleMode::ColdCollectedSllPrimaryOnly ||
        mode == InternalTripleMode::ColdRainSllPrimaryOnly ||
        mode == InternalTripleMode::ColdCollisionSllPrimaryOnly || mode == InternalTripleMode::ColdWetGrowthSllPrimaryOnly)
    {
        return milbrandt_detail::ColdSpectrumDomain::GraupelAndHail512;
    }
    if (mode == InternalTripleMode::ColdGraupelConversionsPrimaryOnly ||
        mode == InternalTripleMode::ColdHomogeneousPrimaryOnly)
    {
        return milbrandt_detail::ColdSpectrumDomain::SnowConvertedGraupel512;
    }
    return milbrandt_detail::ColdSpectrumDomain::Base200;
}

const char* MilbrandtScheme::internal_triple_primary_label() const noexcept
{
    if (internal_triple_ && internal_triple_->mode == InternalTripleMode::ColdWetGrowthSllPrimaryOnly)
    {
        return "internal Milbrandt ColdWetGrowthSllPrimaryOnly ";
    }
    if (internal_triple_ && internal_triple_->mode == InternalTripleMode::ColdCollisionSllPrimaryOnly)
    {
        return "internal Milbrandt ColdCollisionSllPrimaryOnly ";
    }
    if (internal_triple_ && internal_triple_->mode == InternalTripleMode::ColdRainSllPrimaryOnly)
    {
        return "internal Milbrandt ColdRainSllPrimaryOnly ";
    }
    if (internal_triple_ && internal_triple_->mode == InternalTripleMode::ColdCollectedSllPrimaryOnly)
    {
        return "internal Milbrandt ColdCollectedSllPrimaryOnly ";
    }
    if (internal_triple_ && internal_triple_->mode == InternalTripleMode::ColdSllPrimaryOnly)
    {
        return "internal Milbrandt ColdSllPrimaryOnly ";
    }
    if (internal_triple_ && internal_triple_->mode == InternalTripleMode::ColdHomogeneousPrimaryOnly)
    {
        return "internal Milbrandt ColdHomogeneousPrimaryOnly ";
    }
    if (internal_triple_ && internal_triple_->mode == InternalTripleMode::ColdGraupelConversionsPrimaryOnly)
    {
        return "internal Milbrandt ColdGraupelConversionsPrimaryOnly ";
    }
    if (internal_triple_ && internal_triple_->mode == InternalTripleMode::ColdIceGraupelPrimaryOnly)
    {
        return "internal Milbrandt ColdIceGraupelPrimaryOnly ";
    }
    if (internal_triple_ && internal_triple_->mode == InternalTripleMode::ColdRimingPrimaryOnly)
    {
        return "internal Milbrandt ColdRimingPrimaryOnly ";
    }
    if (internal_triple_ && internal_triple_->mode == InternalTripleMode::ColdCloudDiffusionPrimaryOnly)
    {
        return "internal Milbrandt ColdCloudDiffusionPrimaryOnly ";
    }
    if (internal_triple_ && internal_triple_->mode == InternalTripleMode::ColdCloudCollectionPrimaryOnly)
    {
        return "internal Milbrandt ColdCloudCollectionPrimaryOnly ";
    }
    if (internal_triple_ && internal_triple_->mode == InternalTripleMode::FrozenCollectionPrimaryOnly)
    {
        return "internal Milbrandt FrozenCollectionPrimaryOnly ";
    }
    if (internal_triple_ && internal_triple_->mode == InternalTripleMode::SubcycledWarmCloudPrimaryOnly)
    {
        return "internal Milbrandt SubcycledWarmCloudPrimaryOnly ";
    }
    if (internal_triple_ && internal_triple_->mode == InternalTripleMode::DynamicWarmCloudPrimaryOnly)
    {
        return "internal Milbrandt DynamicWarmCloudPrimaryOnly ";
    }
    if (internal_triple_ && internal_triple_->mode == InternalTripleMode::WarmCloudPrimaryOnly)
    {
        return "internal Milbrandt WarmCloudPrimaryOnly ";
    }
    if (internal_triple_ && internal_triple_->mode == InternalTripleMode::CloudAdjustmentOnly)
    {
        return "internal Milbrandt CloudAdjustmentOnly ";
    }
    if (internal_triple_ && internal_triple_->mode == InternalTripleMode::WarmPrimaryOnly)
    {
        return "internal Milbrandt WarmPrimaryOnly ";
    }
    if (internal_triple_ && internal_triple_->mode == InternalTripleMode::SnowMeltingOnly)
    {
        return "internal Milbrandt SnowMeltingOnly ";
    }
    if (internal_triple_ && internal_triple_->mode == InternalTripleMode::IceMeltingOnly)
    {
        return "internal Milbrandt IceMeltingOnly ";
    }
    return internal_triple_ && internal_triple_->mode == InternalTripleMode::MeltingOnly
        ? "internal Milbrandt MeltingOnly " : "internal Milbrandt FrozenCompositionOnly ";
}

bool MilbrandtScheme::internal_triple_primary_parameters_match(bool frozen_collection) const noexcept
{
    const std::array<double, 25> parameters{
        a_g_, a_h_, a_i_, a_r_, a_s_,
        alpha_g_, alpha_h_, alpha_i_, alpha_r_, alpha_s_,
        b_g_, b_h_, b_i_, b_r_, b_s_,
        c_g_, c_h_, c_i_, c_r_, c_s_,
        d_g_, d_h_, d_i_, d_r_, d_s_};
    return (hail_processes_ || frozen_collection ||
            (internal_triple_ && (internal_triple_->mode == InternalTripleMode::FrozenCollectionPrimaryOnly ||
                                 internal_triple_->mode == InternalTripleMode::ColdCloudCollectionPrimaryOnly ||
                                 internal_triple_->mode == InternalTripleMode::ColdCloudDiffusionPrimaryOnly ||
                                 is_cold_riming_mode(internal_triple_->mode))))
        && parameters == milbrandt_detail::default_cold_parameters;
}

bool MilbrandtScheme::enable_internal_triple_frozen_composition_for_test(
    const SimulationStateView& state,
    const Field3D& graupel_number,
    const Field3D& graupel_reflectivity,
    const Field3D& hail_number,
    const Field3D& hail_reflectivity,
    std::string& error)
{
    return enable_internal_triple_primary_mode_for_test(
        state, nullptr, nullptr, graupel_number, graupel_reflectivity,
        hail_number, hail_reflectivity, InternalTripleMode::FrozenCompositionOnly, error);
}

bool MilbrandtScheme::enable_internal_triple_primary_mode_for_test(
    const SimulationStateView& state,
    const Field3D* rain_number,
    const Field3D* rain_reflectivity,
    const Field3D& graupel_number,
    const Field3D& graupel_reflectivity,
    const Field3D& hail_number,
    const Field3D& hail_reflectivity,
    InternalTripleMode mode,
    std::string& error)
{
    return enable_internal_triple_primary_moments_for_test(state,
        {rain_number, nullptr, nullptr, &graupel_number, &hail_number},
        {rain_reflectivity, nullptr, nullptr, &graupel_reflectivity, &hail_reflectivity},
        mode, error);
}

bool MilbrandtScheme::enable_internal_triple_primary_moments_for_test(
    const SimulationStateView& state,
    const std::array<const Field3D*, 5>& supplied_number,
    const std::array<const Field3D*, 5>& supplied_reflectivity,
    InternalTripleMode mode,
    std::string& error,
    const Field3D* cloud_number)
{
    const bool melting = mode == InternalTripleMode::MeltingOnly;
    const bool ice_melting = mode == InternalTripleMode::IceMeltingOnly;
    const bool snow_melting = mode == InternalTripleMode::SnowMeltingOnly;
    const bool warm_primary = mode == InternalTripleMode::WarmPrimaryOnly;
    const bool cloud_adjustment = mode == InternalTripleMode::CloudAdjustmentOnly;
    const bool dynamic_warm_cloud_primary = mode == InternalTripleMode::DynamicWarmCloudPrimaryOnly;
    const bool subcycled_warm_cloud_primary = mode == InternalTripleMode::SubcycledWarmCloudPrimaryOnly;
    const bool frozen_collection = mode == InternalTripleMode::FrozenCollectionPrimaryOnly;
    const bool cold_cloud_diffusion = mode == InternalTripleMode::ColdCloudDiffusionPrimaryOnly;
    const bool cold_riming = is_cold_riming_mode(mode);
    const bool cold_rain = mode == InternalTripleMode::ColdRainSllPrimaryOnly || mode == InternalTripleMode::ColdCollisionSllPrimaryOnly || mode == InternalTripleMode::ColdWetGrowthSllPrimaryOnly;
    const auto spectrum_domain = internal_triple_spectrum_domain(mode);
    const bool cold_cloud_collection = mode == InternalTripleMode::ColdCloudCollectionPrimaryOnly || cold_cloud_diffusion || cold_riming;
    const bool warm_cloud_primary = mode == InternalTripleMode::WarmCloudPrimaryOnly
        || dynamic_warm_cloud_primary || subcycled_warm_cloud_primary;
    const bool cloud_owned = warm_primary || cloud_adjustment || warm_cloud_primary || cold_cloud_collection;
    const bool cloud_velocity = cloud_adjustment || warm_cloud_primary;
    const bool warm = melting || ice_melting || snow_melting || (cloud_owned && !cold_cloud_collection);
    const char* label = mode == InternalTripleMode::ColdWetGrowthSllPrimaryOnly ? "internal Milbrandt ColdWetGrowthSllPrimaryOnly "
        : mode == InternalTripleMode::ColdCollisionSllPrimaryOnly ? "internal Milbrandt ColdCollisionSllPrimaryOnly "
        : cold_rain ? "internal Milbrandt ColdRainSllPrimaryOnly "
        : mode == InternalTripleMode::ColdCollectedSllPrimaryOnly
        ? "internal Milbrandt ColdCollectedSllPrimaryOnly "
        : mode == InternalTripleMode::ColdSllPrimaryOnly ? "internal Milbrandt ColdSllPrimaryOnly "
        : mode == InternalTripleMode::ColdHomogeneousPrimaryOnly ? "internal Milbrandt ColdHomogeneousPrimaryOnly "
        : mode == InternalTripleMode::ColdGraupelConversionsPrimaryOnly ? "internal Milbrandt ColdGraupelConversionsPrimaryOnly "
        : mode == InternalTripleMode::ColdIceGraupelPrimaryOnly ? "internal Milbrandt ColdIceGraupelPrimaryOnly "
        : cold_riming ? "internal Milbrandt ColdRimingPrimaryOnly "
        : cold_cloud_diffusion ? "internal Milbrandt ColdCloudDiffusionPrimaryOnly "
        : cold_cloud_collection ? "internal Milbrandt ColdCloudCollectionPrimaryOnly "
        : frozen_collection ? "internal Milbrandt FrozenCollectionPrimaryOnly "
        : subcycled_warm_cloud_primary ? "internal Milbrandt SubcycledWarmCloudPrimaryOnly "
        : dynamic_warm_cloud_primary ? "internal Milbrandt DynamicWarmCloudPrimaryOnly "
        : warm_cloud_primary ? "internal Milbrandt WarmCloudPrimaryOnly "
        : cloud_adjustment ? "internal Milbrandt CloudAdjustmentOnly "
        : warm_primary ? "internal Milbrandt WarmPrimaryOnly "
        : snow_melting ? "internal Milbrandt SnowMeltingOnly "
        : ice_melting ? "internal Milbrandt IceMeltingOnly "
        : melting ? "internal Milbrandt MeltingOnly " : "internal Milbrandt FrozenCompositionOnly ";
    const std::array<bool, 5> owned{warm || cold_rain, ice_melting || frozen_collection || cold_riming,
                                  snow_melting || frozen_collection || cold_cloud_collection,
                                  !ice_melting && !snow_melting && (!cloud_owned || cold_cloud_collection),
                                  !ice_melting && !snow_melting && (!cloud_owned || cold_cloud_collection)};
    if (internal_triple_)
    {
        error = "internal Milbrandt triple modes are mutually exclusive";
        return false;
    }
    if (mode != InternalTripleMode::FrozenCompositionOnly && !warm && !frozen_collection && !cold_cloud_collection)
    {
        error = std::string(label) + "moment selection is invalid";
        return false;
    }
    if (cloud_owned ? !cloud_number : cloud_number != nullptr)
    {
        error = std::string(label) + "cloud moment selection is invalid";
        return false;
    }
    for (std::size_t species = 0; species < owned.size(); ++species)
    {
        if (owned[species] ? !supplied_number[species] || !supplied_reflectivity[species]
                           : supplied_number[species] || supplied_reflectivity[species])
        {
            error = std::string(label) + "moment selection is invalid";
            return false;
        }
    }
    if (!validate_internal_triple_source_profile(
            state, false, error, std::nullopt, !warm && !frozen_collection && !cold_cloud_collection, melting, ice_melting, snow_melting,
            warm_primary, cloud_adjustment,
            warm_cloud_primary && !dynamic_warm_cloud_primary && !subcycled_warm_cloud_primary,
            dynamic_warm_cloud_primary, subcycled_warm_cloud_primary, frozen_collection,
            cold_cloud_collection && !cold_cloud_diffusion && !cold_riming, cold_cloud_diffusion, cold_riming, cold_rain))
    {
        return false;
    }
    const auto& f = state.fields;
    if (!canonical_primary_fields(state) ||
        !internal_triple_primary_parameters_match(frozen_collection || cold_cloud_collection) ||
        ((frozen_collection || cold_cloud_collection) && state.configuration->microphysics_config.milbrandt.hail_processes != hail_processes_) ||
        (cloud_owned && !matches(f.rho, *cloud_number)))
    {
        error = std::string(label) + "owner, moment shape or cold law differs";
        return false;
    }
    for (std::size_t species = 0; species < owned.size(); ++species)
    {
        if (owned[species] &&
            (!matches(f.rho, *supplied_number[species]) || !matches(f.rho, *supplied_reflectivity[species])))
        {
            error = std::string(label) + "owner, moment shape or cold law differs";
            return false;
        }
    }
    const std::array<const Field3D*, 6> retained{&Nc_, &Nr_, &Ni_, &Ns_, &Ng_, &Nh_};
    for (std::size_t index = 0; index < retained.size(); ++index)
    {
        if ((index == 0 && cloud_owned) ||
            (index > 0 && owned[index - 1] && !(cloud_adjustment && index == 1)))
        {
            continue;
        }
        if (!retained[index]->empty() &&
            (!matches(f.rho, *retained[index]) || !zero_field(*retained[index])))
        {
            error = std::string(label) + "has an unowned retained number moment";
            return false;
        }
    }
    if (cloud_owned)
    {
        for (std::size_t index = 0; index < f.rho.size(); ++index)
        {
            const float nc = cloud_number->data()[index];
            if (!std::isfinite(nc) || nc < 0.0f || (f.qc.data()[index] > 0.0f && nc <= 0.0f))
            {
                error = std::string(label) + "cloud number is invalid";
                return false;
            }
        }
    }
    const std::array<const Field3D*, 5> mass{&f.qr, &f.qi, &f.qs, &f.qg, &f.qh};
    for (std::size_t species = 0; species < owned.size(); ++species)
    {
        if (!owned[species])
        {
            continue;
        }
        if (!valid_triple(*mass[species], *supplied_number[species], *supplied_reflectivity[species],
                          static_cast<milbrandt_detail::TripleMomentSpecies>(species), error,
                          cold_riming ? &spectrum_domain : nullptr))
        {
            return false;
        }
    }
    MilbrandtScheme candidate(*this);
    if (cloud_owned)
    {
        candidate.Nc_ = *cloud_number;
    }
    else
    {
        candidate.Nc_.resize(f.rho.size_r(), f.rho.size_th(), f.rho.size_z(), 0.0f);
    }
    const std::array<Field3D*, 5> number{&candidate.Nr_, &candidate.Ni_, &candidate.Ns_,
                                       &candidate.Ng_, &candidate.Nh_};
    for (std::size_t species = 0; species < owned.size(); ++species)
    {
        if (owned[species])
        {
            *number[species] = *supplied_number[species];
        }
        else
        {
            number[species]->resize(f.rho.size_r(), f.rho.size_th(), f.rho.size_z(), 0.0f);
        }
    }
    for (Field3D* field : {&candidate.dNc_dt_, &candidate.dNr_dt_, &candidate.dNi_dt_,
                           &candidate.dNs_dt_, &candidate.dNg_dt_, &candidate.dNh_dt_})
    {
        field->resize(f.rho.size_r(), f.rho.size_th(), f.rho.size_z(), 0.0f);
    }
    candidate.surface_sedimentation_sink_rate_.resize(
        f.rho.size_r(), f.rho.size_th(), 1, 0.0f);
    InternalTripleCandidateState internal;
    internal.mode = mode;
    internal.warm_storage_owner = state.storage_owner;
    if (cloud_velocity)
    {
        const auto canonical = make_simulation_state_view(*state.storage_owner);
        candidate.set_vertical_velocity_field(canonical.fields.w, false);
        internal.primary_velocity_source = &canonical.fields.w;
    }
    for (std::size_t species = 0; species < owned.size(); ++species)
    {
        if (owned[species])
        {
            internal.reflectivity[species] = *supplied_reflectivity[species];
        }
        else
        {
            internal.reflectivity[species].resize(
                f.rho.size_r(), f.rho.size_th(), f.rho.size_z(), 0.0f);
        }
    }
    candidate.internal_triple_ = std::move(internal);
    *this = std::move(candidate);
    error.clear();
    return true;
}

bool MilbrandtScheme::internal_triple_primary_private_source_matches(
    std::string& error) const
{
    const auto& internal = *internal_triple_;
    const std::array<const Field3D*, 6> number{&Nc_, &Nr_, &Ni_, &Ns_, &Ng_, &Nh_};
    for (std::size_t index = 0; index < number.size(); ++index)
    {
        if (!same_bytes(*number[index], internal.number_source_before[index]))
        {
            error = std::string(internal_triple_primary_label()) + "private number source changed";
            return false;
        }
    }
    for (std::size_t index = 0; index < internal.reflectivity.size(); ++index)
    {
        if (!same_bytes(internal.reflectivity[index], internal.reflectivity_source_before[index]))
        {
            error = std::string(internal_triple_primary_label()) + "private reflectivity source changed";
            return false;
        }
    }
    return true;
}

bool MilbrandtScheme::internal_triple_cloud_velocity_source_matches() const noexcept
{
    if (!internal_triple_ ||
        (internal_triple_->mode != InternalTripleMode::CloudAdjustmentOnly &&
         internal_triple_->mode != InternalTripleMode::WarmCloudPrimaryOnly &&
         internal_triple_->mode != InternalTripleMode::DynamicWarmCloudPrimaryOnly &&
         internal_triple_->mode != InternalTripleMode::SubcycledWarmCloudPrimaryOnly))
    {
        return false;
    }
    const auto& internal = *internal_triple_;
    const auto* source = internal.primary_velocity_source;
    if (!source || vertical_velocity_field() != source || vertical_velocity_is_face_staggered() ||
        !lower_boundary_vertical_velocity().empty() ||
        !same_bytes(*source, internal.primary_velocity_before))
    {
        return false;
    }
    for (std::size_t index = 0; index < source->size(); ++index)
    {
        if (!std::isfinite(source->data()[index]))
        {
            return false;
        }
    }
    return true;
}

void MilbrandtScheme::compute_internal_triple_primary_tendencies(
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
    const bool melting = internal.mode == InternalTripleMode::MeltingOnly;
    const bool ice_melting = internal.mode == InternalTripleMode::IceMeltingOnly;
    const bool snow_melting = internal.mode == InternalTripleMode::SnowMeltingOnly;
    const bool warm_primary = internal.mode == InternalTripleMode::WarmPrimaryOnly;
    const bool cloud_adjustment = internal.mode == InternalTripleMode::CloudAdjustmentOnly;
    const bool dynamic_warm_cloud_primary = internal.mode == InternalTripleMode::DynamicWarmCloudPrimaryOnly;
    const bool subcycled_warm_cloud_primary = internal.mode == InternalTripleMode::SubcycledWarmCloudPrimaryOnly;
    const bool frozen_collection = internal.mode == InternalTripleMode::FrozenCollectionPrimaryOnly;
    const bool cold_cloud_diffusion = internal.mode == InternalTripleMode::ColdCloudDiffusionPrimaryOnly;
    const bool cold_riming = is_cold_riming_mode(internal.mode);
    const bool cold_rain = internal.mode == InternalTripleMode::ColdRainSllPrimaryOnly || internal.mode == InternalTripleMode::ColdCollisionSllPrimaryOnly || internal.mode == InternalTripleMode::ColdWetGrowthSllPrimaryOnly;
    const auto spectrum_domain = internal_triple_spectrum_domain(internal.mode);
    const bool cold_cloud_collection = internal.mode == InternalTripleMode::ColdCloudCollectionPrimaryOnly || cold_cloud_diffusion || cold_riming;
    const bool warm_cloud_primary = internal.mode == InternalTripleMode::WarmCloudPrimaryOnly
        || dynamic_warm_cloud_primary || subcycled_warm_cloud_primary;
    const bool cloud_owned = warm_primary || cloud_adjustment || warm_cloud_primary || cold_cloud_collection;
    const bool cloud_velocity = cloud_adjustment || warm_cloud_primary;
    const bool warm = melting || ice_melting || snow_melting || (cloud_owned && !cold_cloud_collection);
    const std::array<bool, 5> owned{warm || cold_rain, ice_melting || frozen_collection || cold_riming,
                                  snow_melting || frozen_collection || cold_cloud_collection,
                                  !ice_melting && !snow_melting && (!cloud_owned || cold_cloud_collection),
                                  !ice_melting && !snow_melting && (!cloud_owned || cold_cloud_collection)};
    const char* label = internal_triple_primary_label();
    internal.candidate_valid = false;
    internal.prepared = false;
    internal.primary_endpoints.owner = nullptr;
    if ((!warm && !frozen_collection && !cold_cloud_collection && internal.mode != InternalTripleMode::FrozenCompositionOnly) || internal.poisoned ||
        !internal.warm_storage_owner || !std::isfinite(dt) || dt <= 0.0 ||
        internal.candidate_generation == std::numeric_limits<std::uint64_t>::max())
    {
        throw std::runtime_error(std::string(label) + "candidate is unavailable");
    }
    ++internal.candidate_generation;
    const auto live = make_simulation_state_view(*internal.warm_storage_owner);
    const std::array<const Field3D*, 10> sources{
        &density, &p, &theta, &vapor, &cloud,
        mass[0], mass[1], mass[2], mass[3], mass[4]};
    std::string error;
    if (!canonical_primary_fields(live) || sources != primary_fields(live) ||
        !internal_triple_primary_parameters_match() ||
        ((frozen_collection || cold_cloud_collection) && live.configuration->microphysics_config.milbrandt.hail_processes != hail_processes_) ||
        !validate_internal_triple_source_profile(live, false, error, std::nullopt,
                                                !warm && !frozen_collection && !cold_cloud_collection, melting, ice_melting, snow_melting,
                                                warm_primary, cloud_adjustment,
                                                warm_cloud_primary && !dynamic_warm_cloud_primary && !subcycled_warm_cloud_primary,
                                                dynamic_warm_cloud_primary, subcycled_warm_cloud_primary, frozen_collection,
                                                cold_cloud_collection && !cold_cloud_diffusion && !cold_riming, cold_cloud_diffusion, cold_riming, cold_rain))
    {
        throw std::runtime_error(
            std::string(label) + "runtime owner changed: " + error);
    }
    Field3D velocity_before;
    if (cloud_velocity)
    {
        if (internal.primary_velocity_source != &live.fields.w ||
            vertical_velocity_field() != &live.fields.w || vertical_velocity_is_face_staggered() ||
            !lower_boundary_vertical_velocity().empty())
        {
            throw std::runtime_error(std::string(label) + "wind context changed");
        }
        velocity_before = live.fields.w;
    }
    const std::array<const Field3D*, 6> number{&Nc_, &Nr_, &Ni_, &Ns_, &Ng_, &Nh_};
    for (std::size_t index = 0; index < number.size(); ++index)
    {
        if (!matches(density, *number[index]) ||
            ((index == 0 ? !cloud_owned : !owned[index - 1]) && !zero_field(*number[index])))
        {
            throw std::runtime_error(
                std::string(label) + "private number shape or owner changed");
        }
    }
    if (cloud_owned)
    {
        for (std::size_t index = 0; index < density.size(); ++index)
        {
            const float nc = Nc_.data()[index];
            if (!std::isfinite(nc) || nc < 0.0f || (cloud.data()[index] > 0.0f && nc <= 0.0f))
            {
                throw std::runtime_error(std::string(label) + "cloud number source is invalid");
            }
        }
    }
    for (std::size_t index = 0; index < internal.reflectivity.size(); ++index)
    {
        if (!matches(density, internal.reflectivity[index]) ||
            (!owned[index] && !zero_field(internal.reflectivity[index])))
        {
            throw std::runtime_error(
                std::string(label) + "private reflectivity shape or owner changed");
        }
    }
    for (std::size_t species = 0; species < owned.size(); ++species)
    {
        if (owned[species] && !valid_triple(*mass[species], *number[species + 1],
                internal.reflectivity[species], static_cast<milbrandt_detail::TripleMomentSpecies>(species), error,
                cold_riming ? &spectrum_domain : nullptr))
        {
            throw std::runtime_error(std::string(label) + "source moment rejected: " + error);
        }
    }
    std::array<Field3D, 10> source_before;
    std::array<Field3D, 6> number_before;
    for (std::size_t index = 0; index < sources.size(); ++index)
    {
        source_before[index] = *sources[index];
    }
    for (std::size_t index = 0; index < number.size(); ++index)
    {
        number_before[index] = *number[index];
    }
    auto reflectivity_before = internal.reflectivity;
    std::array<Field3D, 8> candidate_rates;
    for (auto& field : candidate_rates)
    {
        field.resize(density.size_r(), density.size_th(), density.size_z(), 0.0f);
    }
    std::array<Field3D, 5> mass_after;
    std::array<Field3D, 5> number_after;
    auto reflectivity_after = internal.reflectivity;
    for (std::size_t species = 0; species < mass.size(); ++species)
    {
        mass_after[species] = *mass[species];
        number_after[species] = *number[species + 1];
    }
    Field3D cloud_number_after(Nc_);
    Field3D vapor_after(vapor);
    Field3D cloud_after(cloud);
    Field3D theta_after(theta);
    Field3D pressure_before(p);
    Field3D density_before(density);
    MicrophysicsPrimaryEndpointCandidate endpoints;
    endpoints.owner = this;
    endpoints.generation = internal.candidate_generation;
    endpoints.duration_s = dt;
    endpoints.nr = density.size_r();
    endpoints.nth = density.size_th();
    endpoints.nz = density.size_z();
    endpoints.source_fields = sources;
    for (std::size_t species = 0; species < endpoints.water_after.size(); ++species)
    {
        endpoints.water_after[species].assign(sources[species + 3]->data(),
                                             sources[species + 3]->data() + density.size());
    }
    endpoints.theta_increment.assign(density.size(), 0.0);
    const auto domain = physical_cell_domain(
        density.size_r(), density.size_th(), density.size_z());
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const std::size_t index = (static_cast<std::size_t>(i) * density.size_th() + j)
                    * density.size_z() + k;
                const double temperature = thermodynamics::theta_to_temperature(theta(i, j, k), p(i, j, k));
                const std::array<double, 16> cell{
                    temperature, p(i, j, k), density(i, j, k),
                    vapor(i, j, k), cloud(i, j, k), (*mass[0])(i, j, k),
                    (*mass[1])(i, j, k), (*mass[2])(i, j, k),
                    (*mass[3])(i, j, k), (*mass[4])(i, j, k),
                    Nc_(i, j, k), Nr_(i, j, k), Ni_(i, j, k),
                    Ns_(i, j, k), Ng_(i, j, k), Nh_(i, j, k)};
                InternalTriplePrimaryCellCandidate cell_candidate;
                const std::array<float, 5> cell_reflectivity{
                    internal.reflectivity[0](i, j, k), internal.reflectivity[1](i, j, k),
                    internal.reflectivity[2](i, j, k), internal.reflectivity[3](i, j, k),
                    internal.reflectivity[4](i, j, k)};
                if (cold_riming)
                {
                    if (!evaluate_internal_triple_cold_riming_primary_cell(
                            cell, cell_reflectivity, dt, cell_candidate, error))
                    {
                        throw std::runtime_error(std::string(label) + "cell rejected: " + error);
                    }
                }
                else if (cold_cloud_diffusion)
                {
                    if (!evaluate_internal_triple_cold_cloud_diffusion_primary_cell(
                            cell, cell_reflectivity, dt, cell_candidate, error))
                    {
                        throw std::runtime_error(std::string(label) + "cell rejected: " + error);
                    }
                }
                else if (cold_cloud_collection)
                {
                    if (!evaluate_internal_triple_cold_cloud_collection_primary_cell(
                            cell, cell_reflectivity, dt, cell_candidate, error))
                    {
                        throw std::runtime_error(std::string(label) + "cell rejected: " + error);
                    }
                }
                else if (frozen_collection)
                {
                    if (!evaluate_internal_triple_frozen_collection_primary_cell(
                            cell, cell_reflectivity, dt, cell_candidate, error))
                    {
                        throw std::runtime_error(std::string(label) + "cell rejected: " + error);
                    }
                }
                else if (subcycled_warm_cloud_primary)
                {
                    if (!evaluate_internal_triple_subcycled_warm_cloud_primary_cell(
                            cell, cell_reflectivity, live.fields.w(i, j, k), dt, cell_candidate, error))
                    {
                        throw std::runtime_error(std::string(label) + "cell rejected: " + error);
                    }
                }
                else if (dynamic_warm_cloud_primary)
                {
                    if (!evaluate_internal_triple_dynamic_warm_cloud_primary_cell(
                            cell, cell_reflectivity, live.fields.w(i, j, k), dt, cell_candidate, error))
                    {
                        throw std::runtime_error(std::string(label) + "cell rejected: " + error);
                    }
                }
                else if (warm_cloud_primary)
                {
                    if (!evaluate_internal_triple_warm_cloud_primary_cell(
                            cell, cell_reflectivity, live.fields.w(i, j, k), dt, cell_candidate, error))
                    {
                        throw std::runtime_error(std::string(label) + "cell rejected: " + error);
                    }
                }
                else if (cloud_adjustment)
                {
                    if (!evaluate_internal_triple_cloud_adjustment_cell(
                            cell, cell_reflectivity, live.fields.w(i, j, k), dt, cell_candidate, error))
                    {
                        throw std::runtime_error(std::string(label) + "cell rejected: " + error);
                    }
                }
                else if (warm_primary)
                {
                    if (!evaluate_internal_triple_warm_primary_cell(
                            cell, cell_reflectivity, dt, cell_candidate, error))
                    {
                        throw std::runtime_error(std::string(label) + "cell rejected: " + error);
                    }
                }
                else if (snow_melting)
                {
                    if (!evaluate_internal_triple_snow_melting_cell(
                            cell, cell_reflectivity, dt, cell_candidate, error))
                    {
                        throw std::runtime_error(std::string(label) + "cell rejected: " + error);
                    }
                }
                else if (ice_melting)
                {
                    if (!evaluate_internal_triple_ice_melting_cell(
                            cell, cell_reflectivity, dt, cell_candidate, error))
                    {
                        throw std::runtime_error(std::string(label) + "cell rejected: " + error);
                    }
                }
                else if (melting)
                {
                    if (!evaluate_internal_triple_melting_cell(
                            cell, cell_reflectivity, dt, cell_candidate, error))
                    {
                        throw std::runtime_error(std::string(label) + "cell rejected: " + error);
                    }
                }
                else
                {
                    const milbrandt_detail::ColdJointFrozenDiffusionTripleState initial{
                        {(*mass[3])(i, j, k), Ng_(i, j, k), internal.reflectivity[3](i, j, k)},
                        {(*mass[4])(i, j, k), Nh_(i, j, k), internal.reflectivity[4](i, j, k)}};
                    std::array<float, 14> rates{};
                    milbrandt_detail::ColdJointFrozenDiffusionTripleResult result;
                    if (!milbrandt_detail::evaluate_published_joint_triple_frozen_diffusion_cell(
                            cell, dt, initial, rates, result, error))
                    {
                        throw std::runtime_error(
                            "internal Milbrandt FrozenCompositionOnly cell rejected: " + error);
                    }
                    for (std::size_t species = 0; species < cell_candidate.water_after.size(); ++species)
                    {
                        cell_candidate.water_after[species] = cell[species + 3];
                    }
                    for (std::size_t species = 0; species < cell_candidate.number_after.size(); ++species)
                    {
                        cell_candidate.number_after[species] = static_cast<float>(cell[species + 11]);
                    }
                    cell_candidate.reflectivity_after = cell_reflectivity;
                    const double deposition = result.diffusion_events[0].admitted_mass_increment_per_kg
                        + result.diffusion_events[1].admitted_mass_increment_per_kg;
                    cell_candidate.water_after[0] = static_cast<double>(vapor(i, j, k)) - deposition;
                    cell_candidate.water_after[5] = result.unrounded.graupel.mass_per_kg;
                    cell_candidate.water_after[6] = result.unrounded.hail.mass_per_kg;
                    cell_candidate.temperature_increment = microphysics_constants::L_s
                        / microphysics_constants::cp * deposition;
                    cell_candidate.number_after[3] = result.graupel.number_per_kg;
                    cell_candidate.number_after[4] = result.hail.number_per_kg;
                    cell_candidate.reflectivity_after[3] = result.graupel.reflectivity_per_kg;
                    cell_candidate.reflectivity_after[4] = result.hail.reflectivity_per_kg;
                    cell_candidate.rates[1] = rates[6];
                    cell_candidate.rates[6] = rates[4];
                    cell_candidate.rates[7] = rates[5];
                }
                const double qv_next = cell_candidate.water_after[0];
                const double theta_increment = thermodynamics::temperature_tendency_to_theta(
                    cell_candidate.temperature_increment, p(i, j, k));
                const double theta_unrounded = static_cast<double>(theta(i, j, k)) + theta_increment;
                if (!representable_water(qv_next) || !std::isfinite(theta_increment) ||
                    !std::isfinite(theta_unrounded) ||
                    std::abs(theta_unrounded) > std::numeric_limits<float>::max() ||
                    !milbrandt_detail::published_joint_frozen_diffusion_rate_representable(theta_increment / dt))
                {
                    throw std::runtime_error(
                        std::string(label) + "water or theta endpoint is unrepresentable");
                }
                const float theta_rounded = static_cast<float>(theta_unrounded);
                double unrounded_temperature = 0.0;
                double rounded_temperature = 0.0;
                if (!milbrandt_detail::internal_triple_raw_temperature_pair(theta_unrounded, theta_rounded,
                        p(i, j, k), unrounded_temperature, rounded_temperature) ||
                    !std::isfinite(theta_rounded) ||
                    !std::isfinite(unrounded_temperature) ||
                    ((cloud_owned && !cold_cloud_collection) ? unrounded_temperature <= microphysics_constants::T0
                        : warm ? unrounded_temperature < microphysics_constants::T0
                             : unrounded_temperature < 150.0 || unrounded_temperature >= microphysics_constants::T0) ||
                    !std::isfinite(rounded_temperature) ||
                    ((cloud_owned && !cold_cloud_collection) ? rounded_temperature <= microphysics_constants::T0
                        : warm ? rounded_temperature < microphysics_constants::T0
                             : rounded_temperature < 150.0 || rounded_temperature >= microphysics_constants::T0) ||
                    (cloud_velocity && (unrounded_temperature >= 400.0 || rounded_temperature >= 400.0)))
                {
                    throw std::runtime_error(
                        std::string(label) + "water or final heat is invalid");
                }
                for (std::size_t species = 0; species < endpoints.water_after.size(); ++species)
                {
                    endpoints.water_after[species][index] = cell_candidate.water_after[species];
                }
                endpoints.theta_increment[index] = theta_increment;
                vapor_after(i, j, k) = static_cast<float>(qv_next);
                cloud_after(i, j, k) = static_cast<float>(cell_candidate.water_after[1]);
                if (cloud_owned)
                {
                    cloud_number_after(i, j, k) = static_cast<float>(cell_candidate.cloud_number_after);
                }
                for (std::size_t species = 0; species < mass_after.size(); ++species)
                {
                    mass_after[species](i, j, k) = static_cast<float>(cell_candidate.water_after[species + 2]);
                    number_after[species](i, j, k) = cell_candidate.number_after[species];
                    reflectivity_after[species](i, j, k) = cell_candidate.reflectivity_after[species];
                }
                theta_after(i, j, k) = theta_rounded;
                cell_candidate.rates[0] = static_cast<float>(theta_increment / dt);
                for (std::size_t species = 0; species < candidate_rates.size(); ++species)
                {
                    candidate_rates[species](i, j, k) = cell_candidate.rates[species];
                }
            }
        }
    }
    Field3D surface_sink(density.size_r(), density.size_th(), 1, 0.0f);
    std::vector<std::array<milbrandt_detail::TripleMomentSurfaceInventory, 5>> surface(
        static_cast<std::size_t>(density.size_r()) * density.size_th());
    for (std::size_t index = 0; index < tendencies.size(); ++index)
    {
        *tendencies[index] = std::move(candidate_rates[index]);
    }
    internal.mass_after = std::move(mass_after);
    internal.number_after = std::move(number_after);
    internal.reflectivity_after = std::move(reflectivity_after);
    internal.cloud_number_after = std::move(cloud_number_after);
    internal.vapor_after = std::move(vapor_after);
    internal.cloud_after = std::move(cloud_after);
    internal.theta_after = std::move(theta_after);
    internal.pressure_before = std::move(pressure_before);
    internal.density_before = std::move(density_before);
    internal.primary_source_before = std::move(source_before);
    internal.number_source_before = std::move(number_before);
    internal.reflectivity_source_before = std::move(reflectivity_before);
    if (cloud_velocity)
    {
        internal.primary_velocity_before = std::move(velocity_before);
    }
    internal.primary_endpoints = std::move(endpoints);
    internal.surface_sink_after = std::move(surface_sink);
    internal.surface_after = std::move(surface);
    internal.candidate_dt = dt;
    internal.candidate_valid = true;
}

bool MilbrandtScheme::owns_primary_endpoint_commit() const noexcept
{
    return internal_triple_ &&
        (internal_triple_->mode == InternalTripleMode::FrozenCompositionOnly ||
         internal_triple_->mode == InternalTripleMode::MeltingOnly ||
         internal_triple_->mode == InternalTripleMode::IceMeltingOnly ||
         internal_triple_->mode == InternalTripleMode::SnowMeltingOnly ||
         internal_triple_->mode == InternalTripleMode::WarmPrimaryOnly ||
         internal_triple_->mode == InternalTripleMode::CloudAdjustmentOnly ||
         internal_triple_->mode == InternalTripleMode::WarmCloudPrimaryOnly ||
         internal_triple_->mode == InternalTripleMode::DynamicWarmCloudPrimaryOnly ||
         internal_triple_->mode == InternalTripleMode::SubcycledWarmCloudPrimaryOnly ||
         internal_triple_->mode == InternalTripleMode::FrozenCollectionPrimaryOnly ||
         internal_triple_->mode == InternalTripleMode::ColdCloudCollectionPrimaryOnly ||
         internal_triple_->mode == InternalTripleMode::ColdCloudDiffusionPrimaryOnly ||
         is_cold_riming_mode(internal_triple_->mode));
}

const Field3D* MilbrandtScheme::protected_primary_endpoint_wind_source() const noexcept
{
    return internal_triple_ &&
        (internal_triple_->mode == InternalTripleMode::CloudAdjustmentOnly ||
         internal_triple_->mode == InternalTripleMode::WarmCloudPrimaryOnly ||
         internal_triple_->mode == InternalTripleMode::DynamicWarmCloudPrimaryOnly ||
         internal_triple_->mode == InternalTripleMode::SubcycledWarmCloudPrimaryOnly)
        ? internal_triple_->primary_velocity_source : nullptr;
}

bool MilbrandtScheme::validate_primary_endpoint_commit(
    const SimulationStateView& state,
    double dt,
    const MicrophysicsPrimaryEndpointCandidate& candidate,
    std::string& error) const
{
    if (!owns_primary_endpoint_commit() || internal_triple_->poisoned ||
        !internal_triple_->candidate_valid || !std::isfinite(dt) || dt <= 0.0 ||
        dt != internal_triple_->candidate_dt ||
        state.storage_owner != internal_triple_->warm_storage_owner ||
        !canonical_primary_fields(state) ||
        !internal_triple_primary_parameters_match())
    {
        error = std::string(internal_triple_primary_label()) + "has no matching primary candidate";
        return false;
    }
    const auto& internal = *internal_triple_;
    const bool melting = internal.mode == InternalTripleMode::MeltingOnly;
    const bool ice_melting = internal.mode == InternalTripleMode::IceMeltingOnly;
    const bool snow_melting = internal.mode == InternalTripleMode::SnowMeltingOnly;
    const bool warm_primary = internal.mode == InternalTripleMode::WarmPrimaryOnly;
    const bool cloud_adjustment = internal.mode == InternalTripleMode::CloudAdjustmentOnly;
    const bool dynamic_warm_cloud_primary = internal.mode == InternalTripleMode::DynamicWarmCloudPrimaryOnly;
    const bool subcycled_warm_cloud_primary = internal.mode == InternalTripleMode::SubcycledWarmCloudPrimaryOnly;
    const bool frozen_collection = internal.mode == InternalTripleMode::FrozenCollectionPrimaryOnly;
    const bool cold_cloud_diffusion = internal.mode == InternalTripleMode::ColdCloudDiffusionPrimaryOnly;
    const bool cold_riming = is_cold_riming_mode(internal.mode);
    const bool cold_rain = internal.mode == InternalTripleMode::ColdRainSllPrimaryOnly || internal.mode == InternalTripleMode::ColdCollisionSllPrimaryOnly || internal.mode == InternalTripleMode::ColdWetGrowthSllPrimaryOnly;
    const bool cold_cloud_collection = internal.mode == InternalTripleMode::ColdCloudCollectionPrimaryOnly || cold_cloud_diffusion || cold_riming;
    const bool warm_cloud_primary = internal.mode == InternalTripleMode::WarmCloudPrimaryOnly
        || dynamic_warm_cloud_primary || subcycled_warm_cloud_primary;
    const auto sources = primary_fields(state);
    const auto& expected = internal.primary_endpoints;
    if (expected.owner != this || expected.generation != internal.candidate_generation ||
        expected.duration_s != dt || expected.nr != state.grid.nr ||
        expected.nth != state.grid.nth || expected.nz != state.grid.nz ||
        expected.source_fields != sources || !same_record(candidate, expected) ||
        expected.theta_increment.size() != state.fields.rho.size())
    {
        error = std::string(internal_triple_primary_label()) + "primary record differs";
        return false;
    }
    for (const auto& water : expected.water_after)
    {
        if (water.size() != state.fields.rho.size())
        {
            error = std::string(internal_triple_primary_label()) + "primary water record shape changed";
            return false;
        }
    }
    for (std::size_t index = 0; index < sources.size(); ++index)
    {
        if (!same_bytes(*sources[index], internal.primary_source_before[index]))
        {
            error = std::string(internal_triple_primary_label()) + "primary source changed";
            return false;
        }
    }
    if ((cloud_adjustment || warm_cloud_primary) &&
        (internal.primary_velocity_source != &state.fields.w || !internal_triple_cloud_velocity_source_matches()))
    {
        error = std::string(internal_triple_primary_label()) + "wind source or context changed";
        return false;
    }
    if (((frozen_collection || cold_cloud_collection) && (!state.configuration ||
            state.configuration->microphysics_config.milbrandt.hail_processes != hail_processes_)) ||
        !validate_internal_triple_source_profile(state, false, error, std::nullopt,
                                                !melting && !ice_melting && !snow_melting && !warm_primary && !cloud_adjustment && !warm_cloud_primary && !frozen_collection && !cold_cloud_collection,
                                                melting, ice_melting, snow_melting, warm_primary, cloud_adjustment,
                                                warm_cloud_primary && !dynamic_warm_cloud_primary && !subcycled_warm_cloud_primary,
                                                dynamic_warm_cloud_primary, subcycled_warm_cloud_primary, frozen_collection,
                                                cold_cloud_collection && !cold_cloud_diffusion && !cold_riming, cold_cloud_diffusion, cold_riming, cold_rain) ||
        !internal_triple_primary_private_source_matches(error))
    {
        return false;
    }
    error.clear();
    return true;
}

bool MilbrandtScheme::prepare_primary_endpoint_commit(
    const SimulationStateView& state,
    double dt,
    MicrophysicsPrimaryEndpointCandidate& candidate,
    std::string& error) const
{
    if (!owns_primary_endpoint_commit() ||
        !MilbrandtScheme::validate_primary_endpoint_commit(
            state, dt, internal_triple_->primary_endpoints, error))
    {
        if (error.empty())
        {
            error = std::string(internal_triple_primary_label()) + "primary candidate is unavailable";
        }
        return false;
    }
    auto staged = internal_triple_->primary_endpoints;
    staged.owner = this;
    candidate = std::move(staged);
    error.clear();
    return true;
}

bool MilbrandtScheme::prepare_internal_triple_primary_private_commit(
    double dt,
    const MicrophysicsAcceptedPrimaryFields& accepted,
    std::string& error)
{
    auto& internal = *internal_triple_;
    const bool melting = internal.mode == InternalTripleMode::MeltingOnly;
    const bool ice_melting = internal.mode == InternalTripleMode::IceMeltingOnly;
    const bool snow_melting = internal.mode == InternalTripleMode::SnowMeltingOnly;
    const bool warm_primary = internal.mode == InternalTripleMode::WarmPrimaryOnly;
    const bool cloud_adjustment = internal.mode == InternalTripleMode::CloudAdjustmentOnly;
    const bool dynamic_warm_cloud_primary = internal.mode == InternalTripleMode::DynamicWarmCloudPrimaryOnly;
    const bool subcycled_warm_cloud_primary = internal.mode == InternalTripleMode::SubcycledWarmCloudPrimaryOnly;
    const bool frozen_collection = internal.mode == InternalTripleMode::FrozenCollectionPrimaryOnly;
    const bool cold_cloud_diffusion = internal.mode == InternalTripleMode::ColdCloudDiffusionPrimaryOnly;
    const bool cold_riming = is_cold_riming_mode(internal.mode);
    const bool cold_rain = internal.mode == InternalTripleMode::ColdRainSllPrimaryOnly || internal.mode == InternalTripleMode::ColdCollisionSllPrimaryOnly || internal.mode == InternalTripleMode::ColdWetGrowthSllPrimaryOnly;
    const bool cold_cloud_collection = internal.mode == InternalTripleMode::ColdCloudCollectionPrimaryOnly || cold_cloud_diffusion || cold_riming;
    const bool warm_cloud_primary = internal.mode == InternalTripleMode::WarmCloudPrimaryOnly
        || dynamic_warm_cloud_primary || subcycled_warm_cloud_primary;
    const bool cloud_owned = warm_primary || cloud_adjustment || warm_cloud_primary || cold_cloud_collection;
    const char* label = internal_triple_primary_label();
    internal.prepared = false;
    const auto* candidate = accepted.primary_endpoints;
    if (internal.poisoned || !internal.candidate_valid || !internal.warm_storage_owner ||
        !std::isfinite(dt) || dt <= 0.0 || dt != internal.candidate_dt || !candidate ||
        internal.primary_endpoints.owner != this ||
        internal.primary_endpoints.generation != internal.candidate_generation ||
        !same_record(*candidate, internal.primary_endpoints) ||
        !internal_triple_primary_parameters_match())
    {
        error = std::string(label) + "accepted primary record differs";
        return false;
    }
    const auto live = make_simulation_state_view(*internal.warm_storage_owner);
    const std::array<const Field3D*, 7> water{
        &accepted.qv, &accepted.qc, &accepted.qr, &accepted.qi,
        &accepted.qs, &accepted.qg, &accepted.qh};
    const auto sources = primary_fields(live);
    if ((cloud_adjustment || warm_cloud_primary) &&
        (internal.primary_velocity_source != &live.fields.w || !internal_triple_cloud_velocity_source_matches()))
    {
        error = std::string(label) + "wind source or context changed";
        return false;
    }
    if (!canonical_primary_fields(live) || sources != candidate->source_fields ||
        &accepted.rho_d != sources[0] ||
        !same_bytes(accepted.rho_d, internal.primary_source_before[0]) ||
        !same_bytes(*sources[1], internal.primary_source_before[1]) ||
        !same_bytes(*sources[2], internal.theta_after) ||
        candidate->nr != live.grid.nr || candidate->nth != live.grid.nth ||
        candidate->nz != live.grid.nz || candidate->duration_s != dt ||
        candidate->theta_increment.size() != accepted.rho_d.size())
    {
        error = std::string(label) + "accepted source or heat owner differs";
        return false;
    }
    for (std::size_t species = 0; species < water.size(); ++species)
    {
        if (water[species] != sources[species + 3] ||
            !matches(accepted.rho_d, *water[species]) ||
            candidate->water_after[species].size() != accepted.rho_d.size())
        {
            error = std::string(label) + "accepted water owner differs";
            return false;
        }
        for (std::size_t index = 0; index < accepted.rho_d.size(); ++index)
        {
            const float expected = static_cast<float>(candidate->water_after[species][index]);
            if (std::memcmp(water[species]->data() + index, &expected, sizeof(float)) != 0)
            {
                error = std::string(label) + "accepted water endpoint differs";
                return false;
            }
        }
    }
    const auto* repairs = accepted.water_repairs;
    const auto* thermal = accepted.thermodynamic_repairs;
    if (!repairs || !thermal || repairs->nonfinite_tendency_count != 0 ||
        thermal->theta_tendency_limit_count != 0 || thermal->theta_nonfinite_count != 0 ||
        thermal->theta_lower_bound_count != 0 || thermal->theta_upper_bound_count != 0)
    {
        error = std::string(label) + "rejected accepted-primary repair";
        return false;
    }
    for (const auto& species : repairs->species)
    {
        if (species.lower_bound_count != 0 || species.lower_bound_roundoff_count != 0 ||
            species.upper_bound_count != 0 || species.nonfinite_count != 0)
        {
            error = std::string(label) + "rejected accepted-primary water repair";
            return false;
        }
    }
    if (((frozen_collection || cold_cloud_collection) && live.configuration->microphysics_config.milbrandt.hail_processes != hail_processes_) ||
        !validate_internal_triple_source_profile(live, false, error, std::nullopt,
                                                !melting && !ice_melting && !snow_melting && !warm_primary && !cloud_adjustment && !warm_cloud_primary && !frozen_collection && !cold_cloud_collection,
                                                melting, ice_melting, snow_melting, warm_primary, cloud_adjustment,
                                                warm_cloud_primary && !dynamic_warm_cloud_primary && !subcycled_warm_cloud_primary,
                                                dynamic_warm_cloud_primary, subcycled_warm_cloud_primary, frozen_collection,
                                                cold_cloud_collection && !cold_cloud_diffusion && !cold_riming, cold_cloud_diffusion, cold_riming, cold_rain) ||
        !internal_triple_primary_private_source_matches(error) ||
        !matches(accepted.rho_d, internal.cloud_number_after) ||
        (!cloud_owned && !zero_field(internal.cloud_number_after)) ||
        internal.surface_sink_after.size_r() != live.grid.nr ||
        internal.surface_sink_after.size_th() != live.grid.nth ||
        internal.surface_sink_after.size_z() != 1 || !zero_field(internal.surface_sink_after) ||
        internal.surface_after.size() != static_cast<std::size_t>(live.grid.nr) * live.grid.nth)
    {
        if (error.empty())
        {
            error = std::string(label) + "private or surface owner changed";
        }
        return false;
    }
    if (cloud_owned)
    {
        for (std::size_t index = 0; index < accepted.rho_d.size(); ++index)
        {
            const float nc = internal.cloud_number_after.data()[index];
            if (!std::isfinite(nc) || nc < 0.0f || (accepted.qc.data()[index] > 0.0f && nc <= 0.0f))
            {
                error = std::string(label) + "cloud number endpoint is invalid";
                return false;
            }
        }
    }
    const std::array<const Field3D*, 5> mass{
        &accepted.qr, &accepted.qi, &accepted.qs, &accepted.qg, &accepted.qh};
    const auto spectrum_domain = internal_triple_spectrum_domain(internal.mode);
    for (std::size_t species = 0; species < mass.size(); ++species)
    {
        if (!same_bytes(*mass[species], internal.mass_after[species]) ||
            !matches(accepted.rho_d, internal.number_after[species]) ||
            !matches(accepted.rho_d, internal.reflectivity_after[species]) ||
            !valid_triple(*mass[species], internal.number_after[species],
                          internal.reflectivity_after[species],
                          static_cast<milbrandt_detail::TripleMomentSpecies>(species), error,
                          cold_riming ? &spectrum_domain : nullptr))
        {
            if (error.empty())
            {
                error = std::string(label) + "private endpoint differs";
            }
            return false;
        }
    }
    for (const auto& column : internal.surface_after)
    {
        for (const auto& inventory : column)
        {
            if (inventory.mass != 0.0 || inventory.number != 0.0 || inventory.reflectivity != 0.0)
            {
                error = std::string(label) + "observed unowned fallout";
                return false;
            }
        }
    }
    if (!endpoint_attribution_matches(live,
            physical_cell_domain(live.grid.nr, live.grid.nth, live.grid.nz),
            *candidate, internal.primary_source_before, *repairs, *thermal, label, error))
    {
        return false;
    }
    internal.prepared = true;
    error.clear();
    return true;
}

} // namespace tmv
