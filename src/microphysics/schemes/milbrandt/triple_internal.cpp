#include "milbrandt.hpp"

#include "core/runtime/simulation_state.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tmv
{
namespace
{

constexpr std::size_t kInternalTripleWorkBudget = 2097152;

bool same_shape(const Field3D& a, const Field3D& b) noexcept
{
    return a.size_r() == b.size_r() && a.size_th() == b.size_th()
        && a.size_z() == b.size_z();
}

bool same_within_two_ulps(float actual, float expected) noexcept
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

bool MilbrandtScheme::enable_internal_triple_sedimentation_for_test(
    const Field3D& density,
    const std::array<const Field3D*, 5>& mass,
    const std::array<const Field3D*, 5>& number,
    const std::array<const Field3D*, 5>& reflectivity,
    std::string& error)
{
    if (internal_triple_ || density.empty()
        || density.size_r() <= 0 || density.size_th() <= 0
        || density.size_z() <= 0 || density.size_z() > 4096)
    {
        error = "internal Milbrandt triple sedimentation activation is invalid";
        return false;
    }
    const std::size_t nr = static_cast<std::size_t>(density.size_r());
    const std::size_t nth = static_cast<std::size_t>(density.size_th());
    const std::size_t nz = static_cast<std::size_t>(density.size_z());
    if (nr > 4096 / nth || nr * nth > 4096 / nz || !hail_processes_)
    {
        error = "internal Milbrandt triple sedimentation exceeds its grid or source cap";
        return false;
    }
    const std::size_t substeps = std::min<std::size_t>(4096, 65536 / nz);
    if (nr * nth * nz > kInternalTripleWorkBudget / (5 * substeps))
    {
        error = "internal Milbrandt triple sedimentation exceeds aggregate work budget";
        return false;
    }
    const std::array<double, 5> c{c_r_, c_i_, c_s_, c_g_, c_h_};
    const std::array<double, 5> d{d_r_, d_i_, d_s_, d_g_, d_h_};
    const std::array<double, 5> a{a_r_, a_i_, a_s_, a_g_, a_h_};
    const std::array<double, 5> b{b_r_, b_i_, b_s_, b_g_, b_h_};
    for (std::size_t species = 0; species < 5; ++species)
    {
        const auto p = milbrandt_detail::current_nimbus_triple_sedimentation_parameters(
            static_cast<milbrandt_detail::TripleMomentSpecies>(species));
        if (!mass[species] || !number[species] || !reflectivity[species]
            || !same_shape(density, *mass[species])
            || !same_shape(density, *number[species])
            || !same_shape(density, *reflectivity[species])
            || c[species] != p.mass_law.coefficient
            || d[species] != p.mass_law.exponent
            || a[species] != p.fall_coefficient
            || b[species] != p.fall_exponent)
        {
            error = "internal Milbrandt triple sedimentation field or physical parameters differ";
            return false;
        }
    }
    for (std::size_t index = 0; index < density.size(); ++index)
    {
        if (!std::isfinite(density.data()[index]) || density.data()[index] <= 0.0f)
        {
            error = "internal Milbrandt triple sedimentation density is invalid";
            return false;
        }
        for (std::size_t species = 0; species < 5; ++species)
        {
            const double q = mass[species]->data()[index];
            const double n = number[species]->data()[index];
            const double z = reflectivity[species]->data()[index];
            if (q == 0.0 && n == 0.0 && z == 0.0)
            {
                continue;
            }
            double alpha = 0.0;
            double lambda = 0.0;
            const auto p = milbrandt_detail::current_nimbus_triple_sedimentation_parameters(
                static_cast<milbrandt_detail::TripleMomentSpecies>(species));
            if (!milbrandt_detail::diagnose_triple_gamma_shape(
                    q, n, z, p.mass_law, alpha, lambda, error))
            {
                return false;
            }
        }
    }

    MilbrandtScheme candidate(*this);
    candidate.Nc_.resize(density.size_r(), density.size_th(), density.size_z(), 0.0f);
    const std::array<Field3D*, 5> owned_number{
        &candidate.Nr_, &candidate.Ni_, &candidate.Ns_, &candidate.Ng_, &candidate.Nh_};
    const std::array<Field3D*, 6> number_tendencies{
        &candidate.dNc_dt_, &candidate.dNr_dt_, &candidate.dNi_dt_,
        &candidate.dNs_dt_, &candidate.dNg_dt_, &candidate.dNh_dt_};
    InternalTripleCandidateState internal;
    for (std::size_t species = 0; species < 5; ++species)
    {
        *owned_number[species] = *number[species];
        internal.reflectivity[species] = *reflectivity[species];
    }
    for (Field3D* tendency : number_tendencies)
    {
        tendency->resize(density.size_r(), density.size_th(), density.size_z(), 0.0f);
    }
    candidate.surface_sedimentation_sink_rate_.resize(
        density.size_r(), density.size_th(), 1, 0.0f);
    candidate.internal_triple_ = std::move(internal);
    *this = std::move(candidate);
    error.clear();
    return true;
}

const Field3D& MilbrandtScheme::internal_triple_reflectivity_for_test(
    std::size_t species) const
{
    if (!internal_triple_ || species >= 5)
    {
        throw std::out_of_range("internal Milbrandt triple reflectivity is unavailable");
    }
    return internal_triple_->reflectivity[species];
}

const Field3D& MilbrandtScheme::internal_triple_number_for_test(
    std::size_t species) const
{
    if (!internal_triple_ || species >= 5)
    {
        throw std::out_of_range("internal Milbrandt triple number is unavailable");
    }
    const std::array<const Field3D*, 5> number{
        &Nr_, &Ni_, &Ns_, &Ng_, &Nh_};
    return *number[species];
}

milbrandt_detail::TripleMomentSurfaceInventory
MilbrandtScheme::internal_triple_surface_for_test(
    int i, int j, std::size_t species) const
{
    if (!internal_triple_ || i < 0 || j < 0
        || species >= 5 || i >= Nr_.size_r() || j >= Nr_.size_th())
    {
        throw std::out_of_range("internal Milbrandt triple surface inventory is unavailable");
    }
    const std::size_t column = static_cast<std::size_t>(i)
        * static_cast<std::size_t>(Nr_.size_th()) + static_cast<std::size_t>(j);
    const auto& committed = internal_triple_->surface_committed;
    if (column >= committed.size())
    {
        throw std::out_of_range("internal Milbrandt triple surface inventory is not committed");
    }
    return committed[column][species];
}

void MilbrandtScheme::compute_internal_triple_sedimentation_tendencies(
    const Field3D& density,
    const std::array<const Field3D*, 5>& mass,
    double dt,
    const std::array<Field3D*, 5>& mass_tendencies,
    Field3D& theta_tendency,
    Field3D& vapor_tendency,
    Field3D& cloud_tendency)
{
    auto& internal = *internal_triple_;
    internal.candidate_valid = false;
    internal.prepared = false;
    if (internal.poisoned || !std::isfinite(dt) || dt <= 0.0
        || !same_shape(density, Nr_))
    {
        throw std::runtime_error(
            "internal Milbrandt triple sedimentation input or owner is invalid");
    }
    const std::array<const Field3D*, 5> number{
        &Nr_, &Ni_, &Ns_, &Ng_, &Nh_};
    std::array<Field3D, 5> after_mass;
    std::array<Field3D, 5> after_number;
    std::array<Field3D, 5> after_reflectivity;
    std::array<Field3D, 5> rates;
    for (std::size_t species = 0; species < 5; ++species)
    {
        if (!same_shape(density, *mass[species])
            || !same_shape(density, *number[species])
            || !same_shape(density, internal.reflectivity[species]))
        {
            throw std::runtime_error(
                "internal Milbrandt triple sedimentation shape changed");
        }
        after_mass[species] = *mass[species];
        after_number[species] = *number[species];
        after_reflectivity[species] = internal.reflectivity[species];
        rates[species].resize(density.size_r(), density.size_th(), density.size_z(), 0.0f);
    }
    const auto domain = physical_cell_domain(
        density.size_r(), density.size_th(), density.size_z());
    const int levels = domain.k_end - domain.k_begin;
    if (levels <= 0 || levels > 4096)
    {
        throw std::runtime_error(
            "internal Milbrandt triple sedimentation physical column is invalid");
    }
    const std::size_t maximum_substeps = std::min<std::size_t>(
        4096, 65536 / static_cast<std::size_t>(levels));
    if (density.size() > kInternalTripleWorkBudget
            / (5 * maximum_substeps))
    {
        throw std::runtime_error(
            "internal Milbrandt triple sedimentation exceeds aggregate work budget");
    }
    std::vector<double> thickness(static_cast<std::size_t>(levels));
    Field3D surface_sink(density.size_r(), density.size_th(), 1, 0.0f);
    std::vector<std::array<milbrandt_detail::TripleMomentSurfaceInventory, 5>> surface(
        static_cast<std::size_t>(density.size_r()) * density.size_th());
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            const std::size_t column = static_cast<std::size_t>(i)
                * density.size_th() + static_cast<std::size_t>(j);
            const std::size_t base = column * density.size_z() + domain.k_begin;
            for (int level = 0; level < levels; ++level)
            {
                thickness[static_cast<std::size_t>(level)] =
                    vertical_spacing_m(i, j, domain.k_begin + level);
            }
            double surface_mass = 0.0;
            for (std::size_t species = 0; species < 5; ++species)
            {
                auto parameters =
                    milbrandt_detail::current_nimbus_triple_sedimentation_parameters(
                        static_cast<milbrandt_detail::TripleMomentSpecies>(species));
                parameters.maximum_substeps = static_cast<int>(maximum_substeps);
                std::string error;
                if (!milbrandt_detail::sediment_triple_moment_column(
                        levels, mass[species]->data() + base,
                        number[species]->data() + base,
                        internal.reflectivity[species].data() + base,
                        density.data() + base, thickness.data(), dt, parameters,
                        after_mass[species].data() + base,
                        after_number[species].data() + base,
                        after_reflectivity[species].data() + base,
                        surface[column][species], error))
                {
                    throw std::runtime_error(
                        "internal Milbrandt triple sedimentation rejected species "
                        + std::to_string(species) + ": " + error);
                }
                surface_mass += surface[column][species].mass;
            }
            const double denominator = density(i, j, domain.k_begin)
                * thickness.front() * dt;
            const double sink = surface_mass / denominator;
            if (!std::isfinite(sink) || sink < 0.0
                || sink > std::numeric_limits<float>::max())
            {
                throw std::runtime_error(
                    "internal Milbrandt triple sedimentation surface mass rate is invalid");
            }
            surface_sink(i, j, 0) = static_cast<float>(sink);
        }
    }
    for (std::size_t species = 0; species < 5; ++species)
    {
        for (std::size_t index = 0; index < rates[species].size(); ++index)
        {
            const double rate = (static_cast<double>(after_mass[species].data()[index])
                - mass[species]->data()[index]) / dt;
            if (!std::isfinite(rate) ||
                std::abs(rate) > std::numeric_limits<float>::max())
            {
                throw std::runtime_error(
                    "internal Milbrandt triple sedimentation mass rate is invalid");
            }
            rates[species].data()[index] = static_cast<float>(rate);
        }
    }
    Field3D zero(density.size_r(), density.size_th(), density.size_z(), 0.0f);
    Field3D zero_vapor(zero);
    Field3D zero_cloud(zero);
    Field3D density_before(density);
    for (std::size_t species = 0; species < 5; ++species)
    {
        *mass_tendencies[species] = std::move(rates[species]);
    }
    theta_tendency = std::move(zero);
    vapor_tendency = std::move(zero_vapor);
    cloud_tendency = std::move(zero_cloud);
    internal.mass_after = std::move(after_mass);
    internal.number_after = std::move(after_number);
    internal.reflectivity_after = std::move(after_reflectivity);
    internal.density_before = std::move(density_before);
    internal.surface_sink_after = std::move(surface_sink);
    internal.surface_after = std::move(surface);
    internal.candidate_dt = dt;
    internal.candidate_valid = true;
}

bool MilbrandtScheme::prepare_post_primary_private_commit(
    double dt,
    const MicrophysicsAcceptedPrimaryFields& accepted,
    std::string& error)
{
    if (internal_triple_ &&
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
         is_cold_riming_mode(internal_triple_->mode)))
    {
        return prepare_internal_triple_primary_private_commit(
            dt, accepted, error);
    }
    if (!internal_triple_)
    {
        error.clear();
        return true;
    }
    auto& internal = *internal_triple_;
    internal.prepared = false;
    const bool frozen_diffusion =
        internal.mode == InternalTripleMode::FrozenDiffusionOnly;
    if (internal.poisoned || !internal.candidate_valid
        || dt != internal.candidate_dt
        || !std::isfinite(dt) || dt <= 0.0
        || !same_shape(accepted.rho_d, internal.density_before)
        || internal.surface_sink_after.size_r() != accepted.rho_d.size_r()
        || internal.surface_sink_after.size_th() != accepted.rho_d.size_th()
        || internal.surface_sink_after.size_z() != 1
        || internal.surface_after.size() != static_cast<std::size_t>(
            accepted.rho_d.size_r()) * accepted.rho_d.size_th())
    {
        error = "internal Milbrandt triple mode has no matching candidate";
        return false;
    }
    const std::array<const Field3D*, 5> accepted_mass{
        &accepted.qr, &accepted.qi, &accepted.qs, &accepted.qg, &accepted.qh};
    for (std::size_t species = 0; species < 5; ++species)
    {
        if (!same_shape(accepted.rho_d, *accepted_mass[species])
            || !same_shape(accepted.rho_d, internal.mass_after[species])
            || !same_shape(accepted.rho_d, internal.number_after[species])
            || !same_shape(accepted.rho_d, internal.reflectivity_after[species]))
        {
            error = "internal Milbrandt triple candidate shape changed";
            return false;
        }
    }
    if (internal.mode != InternalTripleMode::SedimentationOnly)
    {
        const auto* water = accepted.water_repairs;
        const auto* thermal = accepted.thermodynamic_repairs;
        bool water_repaired = !water;
        if (water)
        {
            water_repaired = water->nonfinite_tendency_count != 0;
            for (const auto& species : water->species)
            {
                water_repaired = water_repaired ||
                    species.lower_bound_count != 0 ||
                    species.lower_bound_roundoff_count != 0 ||
                    species.upper_bound_count != 0 ||
                    species.nonfinite_count != 0;
            }
        }
        if (water_repaired || !thermal ||
            thermal->theta_tendency_limit_count != 0 ||
            thermal->theta_lower_bound_count != 0 ||
            thermal->theta_upper_bound_count != 0 ||
            thermal->theta_nonfinite_count != 0)
        {
            error = frozen_diffusion
                ? "internal Milbrandt FrozenDiffusionOnly rejected accepted-primary repair"
                : "internal Milbrandt WarmOnly rejected accepted-primary repair";
            return false;
        }
        if (!same_shape(accepted.rho_d, internal.vapor_after) ||
            !same_shape(accepted.rho_d, internal.cloud_after) ||
            !same_shape(accepted.rho_d, internal.cloud_number_after) ||
            (frozen_diffusion &&
             (!same_shape(accepted.rho_d, accepted.qv) ||
              !same_shape(accepted.rho_d, accepted.qc))))
        {
            error = frozen_diffusion
                ? "internal Milbrandt FrozenDiffusionOnly qv/qc/Nc candidate shape changed"
                : "internal Milbrandt WarmOnly qv/qc/Nc candidate shape changed";
            return false;
        }
        for (std::size_t index = 0; index < accepted.rho_d.size(); ++index)
        {
            const float qv = accepted.qv.data()[index];
            const float qc = accepted.qc.data()[index];
            const float nc = internal.cloud_number_after.data()[index];
            if (qv < 0.0f || qc < 0.0f ||
                !same_within_two_ulps(
                    qv, internal.vapor_after.data()[index]) ||
                !same_within_two_ulps(
                    qc, internal.cloud_after.data()[index]) ||
                !std::isfinite(nc) || nc < 0.0f ||
                (qc > 0.0f && nc <= 0.0f))
            {
                error = frozen_diffusion
                    ? "internal Milbrandt FrozenDiffusionOnly accepted qv/qc/Nc differs"
                    : "internal Milbrandt WarmOnly accepted qv/qc/Nc differs";
                return false;
            }
        }
    }
    if (internal.mode == InternalTripleMode::RainFreezingOnly || frozen_diffusion)
    {
        if (!internal.warm_storage_owner ||
            !same_shape(accepted.rho_d, internal.theta_after) ||
            !same_shape(accepted.rho_d, internal.pressure_before))
        {
            error = frozen_diffusion
                ? "internal Milbrandt FrozenDiffusionOnly heat owner changed"
                : "internal Milbrandt RainFreezingOnly heat owner changed";
            return false;
        }
        const auto live = make_simulation_state_view(*internal.warm_storage_owner);
        if (&live.fields.rho != &accepted.rho_d ||
            &live.fields.qr != &accepted.qr ||
            &live.fields.qg != &accepted.qg ||
            &live.fields.qh != &accepted.qh ||
            !same_shape(accepted.rho_d, live.fields.theta) ||
            !same_shape(accepted.rho_d, live.fields.p) ||
            (frozen_diffusion &&
             (&live.fields.qv != &accepted.qv ||
              &live.fields.qc != &accepted.qc ||
              &live.fields.qi != &accepted.qi ||
              &live.fields.qs != &accepted.qs ||
              !same_shape(accepted.rho_d, accepted.qv) ||
              !same_shape(accepted.rho_d, accepted.qc))))
        {
            error = frozen_diffusion
                ? "internal Milbrandt FrozenDiffusionOnly accepted owner differs"
                : "internal Milbrandt RainFreezingOnly accepted owner differs";
            return false;
        }
        if (frozen_diffusion &&
            !validate_internal_triple_source_profile(
                live, false, error, internal.selected_frozen_species))
        {
            return false;
        }
        for (std::size_t index = 0; index < accepted.rho_d.size(); ++index)
        {
            if (live.fields.p.data()[index] != internal.pressure_before.data()[index] ||
                !same_within_two_ulps(live.fields.theta.data()[index],
                                     internal.theta_after.data()[index]))
            {
                error = frozen_diffusion
                    ? "internal Milbrandt FrozenDiffusionOnly accepted heat differs"
                    : "internal Milbrandt RainFreezingOnly accepted heat differs";
                return false;
            }
        }
    }
    const auto domain = physical_cell_domain(
        accepted.rho_d.size_r(), accepted.rho_d.size_th(),
        accepted.rho_d.size_z());
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            const std::size_t column = static_cast<std::size_t>(i)
                * accepted.rho_d.size_th() + static_cast<std::size_t>(j);
            double mass = 0.0;
            for (const auto& inventory : internal.surface_after[column])
            {
                if (!std::isfinite(inventory.mass) || inventory.mass < 0.0
                    || !std::isfinite(inventory.number) || inventory.number < 0.0
                    || !std::isfinite(inventory.reflectivity)
                    || inventory.reflectivity < 0.0)
                {
                    error = "internal Milbrandt triple surface inventory is invalid";
                    return false;
                }
                mass += inventory.mass;
            }
            const double denominator = static_cast<double>(
                accepted.rho_d(i, j, domain.k_begin))
                * vertical_spacing_m(i, j, domain.k_begin) * dt;
            const double expected_rate = mass / denominator;
            if (!std::isfinite(denominator) || denominator <= 0.0
                || !std::isfinite(expected_rate) || expected_rate < 0.0
                || expected_rate > std::numeric_limits<float>::max()
                || !same_within_two_ulps(
                    internal.surface_sink_after(i, j, 0),
                    static_cast<float>(expected_rate)))
            {
                error = "internal Milbrandt triple surface conversion changed";
                return false;
            }
        }
    }
    for (std::size_t index = 0; index < accepted.rho_d.size(); ++index)
    {
        if (accepted.rho_d.data()[index] != internal.density_before.data()[index])
        {
            error = "internal Milbrandt triple density changed before commit";
            return false;
        }
        for (std::size_t species = 0; species < 5; ++species)
        {
            const float q = accepted_mass[species]->data()[index];
            const float expected_q = internal.mass_after[species].data()[index];
            const float n = internal.number_after[species].data()[index];
            const float z = internal.reflectivity_after[species].data()[index];
            if (!same_within_two_ulps(q, expected_q))
            {
                error = "internal Milbrandt triple accepted mass differs";
                return false;
            }
            if (q == 0.0f && n == 0.0f && z == 0.0f)
            {
                continue;
            }
            double alpha = 0.0;
            double lambda = 0.0;
            const auto parameters =
                milbrandt_detail::current_nimbus_triple_sedimentation_parameters(
                    static_cast<milbrandt_detail::TripleMomentSpecies>(species));
            if (!milbrandt_detail::diagnose_triple_gamma_shape(
                    q, n, z, parameters.mass_law, alpha, lambda, error))
            {
                return false;
            }
        }
    }
    internal.prepared = true;
    error.clear();
    return true;
}

bool MilbrandtScheme::owns_post_primary_private_commit() const noexcept
{
    return internal_triple_.has_value();
}

bool MilbrandtScheme::commit_prepared_private_candidate() noexcept
{
    if (!internal_triple_ ||
        !internal_triple_->prepared ||
        !internal_triple_->candidate_valid ||
        internal_triple_->poisoned)
    {
        return false;
    }
    auto& internal = *internal_triple_;
    if ((internal.mode == InternalTripleMode::CloudAdjustmentOnly ||
         internal.mode == InternalTripleMode::WarmCloudPrimaryOnly ||
         internal.mode == InternalTripleMode::DynamicWarmCloudPrimaryOnly ||
         internal.mode == InternalTripleMode::SubcycledWarmCloudPrimaryOnly) &&
        !internal_triple_cloud_velocity_source_matches())
    {
        return false;
    }
    if (internal.mode != InternalTripleMode::SedimentationOnly)
    {
        Nc_ = std::move(internal.cloud_number_after);
    }
    const std::array<Field3D*, 5> number{
        &Nr_, &Ni_, &Ns_, &Ng_, &Nh_};
    for (std::size_t species = 0; species < 5; ++species)
    {
        *number[species] = std::move(internal.number_after[species]);
        internal.reflectivity[species] =
            std::move(internal.reflectivity_after[species]);
    }
    surface_sedimentation_sink_rate_ = std::move(internal.surface_sink_after);
    internal.surface_committed = std::move(internal.surface_after);
    internal.candidate_valid = false;
    internal.prepared = false;
    if (internal.mode == InternalTripleMode::FrozenCompositionOnly ||
        internal.mode == InternalTripleMode::MeltingOnly ||
        internal.mode == InternalTripleMode::IceMeltingOnly ||
        internal.mode == InternalTripleMode::SnowMeltingOnly ||
        internal.mode == InternalTripleMode::WarmPrimaryOnly ||
        internal.mode == InternalTripleMode::CloudAdjustmentOnly ||
        internal.mode == InternalTripleMode::WarmCloudPrimaryOnly ||
        internal.mode == InternalTripleMode::DynamicWarmCloudPrimaryOnly ||
        internal.mode == InternalTripleMode::SubcycledWarmCloudPrimaryOnly ||
        internal.mode == InternalTripleMode::FrozenCollectionPrimaryOnly ||
        internal.mode == InternalTripleMode::ColdCloudCollectionPrimaryOnly ||
        internal.mode == InternalTripleMode::ColdCloudDiffusionPrimaryOnly ||
        is_cold_riming_mode(internal.mode))
    {
        internal.primary_endpoints.owner = nullptr;
    }
    return true;
}

} // namespace tmv
