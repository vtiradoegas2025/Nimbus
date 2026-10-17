#include "milbrandt.hpp"
#include "triple_cold_diffusion.hpp"
#include "triple_internal_thermal.hpp"

#include "core/runtime/simulation_state.hpp"
#include "microphysics/milbrandt_cold_parameters.hpp"

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

bool MilbrandtScheme::enable_internal_triple_frozen_diffusion_for_test(
    const SimulationStateView& state,
    milbrandt_detail::TripleMomentSpecies species,
    const Field3D& supplied_number,
    const Field3D& supplied_reflectivity,
    std::string& error)
{
    if (internal_triple_)
    {
        error = "internal Milbrandt triple modes are mutually exclusive";
        return false;
    }
    if (!validate_internal_triple_source_profile(state, false, error, species))
    {
        return false;
    }
    const auto& f = state.fields;
    const std::array<double, 25> physical_parameters{
        a_g_, a_h_, a_i_, a_r_, a_s_,
        alpha_g_, alpha_h_, alpha_i_, alpha_r_, alpha_s_,
        b_g_, b_h_, b_i_, b_r_, b_s_,
        c_g_, c_h_, c_i_, c_r_, c_s_,
        d_g_, d_h_, d_i_, d_r_, d_s_};
    if (physical_parameters != milbrandt_detail::default_cold_parameters ||
        !hail_processes_ ||
        !matches(f.rho, supplied_number) ||
        !matches(f.rho, supplied_reflectivity))
    {
        error = "internal Milbrandt FrozenDiffusionOnly moment shape or cold law differs";
        return false;
    }
    const std::size_t selected = static_cast<std::size_t>(species);
    const std::array<const Field3D*, 6> retained_number{
        &Nc_, &Nr_, &Ni_, &Ns_, &Ng_, &Nh_};
    for (std::size_t index = 0; index < retained_number.size(); ++index)
    {
        if (index != selected + 1 && !retained_number[index]->empty() &&
            (!matches(f.rho, *retained_number[index]) ||
             !zero_field(*retained_number[index])))
        {
            error = "internal Milbrandt FrozenDiffusionOnly has an unowned retained number moment";
            return false;
        }
    }
    const Field3D& selected_mass =
        species == milbrandt_detail::TripleMomentSpecies::Graupel ? f.qg : f.qh;
    const auto law = milbrandt_detail::current_nimbus_triple_sedimentation_parameters(
        species).mass_law;
    for (std::size_t index = 0; index < f.rho.size(); ++index)
    {
        const double q = selected_mass.data()[index];
        const double n = supplied_number.data()[index];
        const double z = supplied_reflectivity.data()[index];
        if (q == 0.0 && n == 0.0 && z == 0.0)
        {
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
    MilbrandtScheme candidate(*this);
    const std::array<Field3D*, 6> number{
        &candidate.Nc_, &candidate.Nr_, &candidate.Ni_,
        &candidate.Ns_, &candidate.Ng_, &candidate.Nh_};
    for (std::size_t index = 0; index < number.size(); ++index)
    {
        if (index == selected + 1)
        {
            *number[index] = supplied_number;
        }
        else
        {
            number[index]->resize(f.rho.size_r(), f.rho.size_th(),
                                  f.rho.size_z(), 0.0f);
        }
    }
    for (Field3D* rate : {&candidate.dNc_dt_, &candidate.dNr_dt_,
                          &candidate.dNi_dt_, &candidate.dNs_dt_,
                          &candidate.dNg_dt_, &candidate.dNh_dt_})
    {
        rate->resize(f.rho.size_r(), f.rho.size_th(), f.rho.size_z(), 0.0f);
    }
    candidate.surface_sedimentation_sink_rate_.resize(
        f.rho.size_r(), f.rho.size_th(), 1, 0.0f);
    InternalTripleCandidateState internal;
    internal.mode = InternalTripleMode::FrozenDiffusionOnly;
    internal.selected_frozen_species = species;
    internal.warm_storage_owner = state.storage_owner;
    for (std::size_t index = 0; index < internal.reflectivity.size(); ++index)
    {
        if (index == selected)
        {
            internal.reflectivity[index] = supplied_reflectivity;
        }
        else
        {
            internal.reflectivity[index].resize(
                f.rho.size_r(), f.rho.size_th(), f.rho.size_z(), 0.0f);
        }
    }
    candidate.internal_triple_ = std::move(internal);
    *this = std::move(candidate);
    error.clear();
    return true;
}

void MilbrandtScheme::compute_internal_triple_frozen_diffusion_tendencies(
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
    if (internal.mode != InternalTripleMode::FrozenDiffusionOnly ||
        internal.poisoned || !internal.warm_storage_owner ||
        !std::isfinite(dt) || dt <= 0.0)
    {
        throw std::runtime_error(
            "internal Milbrandt FrozenDiffusionOnly candidate is unavailable");
    }
    auto live = make_simulation_state_view(*internal.warm_storage_owner);
    const auto species = internal.selected_frozen_species;
    std::string error;
    if (&live.fields.p != &p || &live.fields.theta != &theta ||
        &live.fields.rho != &density || &live.fields.qv != &vapor ||
        &live.fields.qc != &cloud || &live.fields.qr != mass[0] ||
        &live.fields.qi != mass[1] || &live.fields.qs != mass[2] ||
        &live.fields.qg != mass[3] || &live.fields.qh != mass[4] ||
        !validate_internal_triple_source_profile(live, false, error, species))
    {
        throw std::runtime_error(
            "internal Milbrandt FrozenDiffusionOnly runtime owner changed: " + error);
    }
    const std::array<double, 25> physical_parameters{
        a_g_, a_h_, a_i_, a_r_, a_s_,
        alpha_g_, alpha_h_, alpha_i_, alpha_r_, alpha_s_,
        b_g_, b_h_, b_i_, b_r_, b_s_,
        c_g_, c_h_, c_i_, c_r_, c_s_,
        d_g_, d_h_, d_i_, d_r_, d_s_};
    if (physical_parameters != milbrandt_detail::default_cold_parameters ||
        !hail_processes_)
    {
        throw std::runtime_error(
            "internal Milbrandt FrozenDiffusionOnly cold law changed");
    }
    const std::size_t selected = static_cast<std::size_t>(species);
    const std::array<const Field3D*, 5> number{
        &Nr_, &Ni_, &Ns_, &Ng_, &Nh_};
    if (!matches(density, Nc_) || !zero_field(Nc_))
    {
        throw std::runtime_error(
            "internal Milbrandt FrozenDiffusionOnly gained an unowned cloud number");
    }
    for (std::size_t index = 0; index < number.size(); ++index)
    {
        if (!matches(density, *number[index]) ||
            !matches(density, internal.reflectivity[index]) ||
            (index != selected &&
             (!zero_field(*number[index]) ||
              !zero_field(internal.reflectivity[index]))))
        {
            throw std::runtime_error(
                "internal Milbrandt FrozenDiffusionOnly private moment shape or owner changed");
        }
    }
    const auto law = milbrandt_detail::current_nimbus_triple_sedimentation_parameters(
        species).mass_law;
    for (std::size_t index = 0; index < density.size(); ++index)
    {
        const double q = mass[selected]->data()[index];
        const double n = number[selected]->data()[index];
        const double z = internal.reflectivity[selected].data()[index];
        if (q == 0.0 && n == 0.0 && z == 0.0)
        {
            continue;
        }
        double alpha = 0.0;
        double lambda = 0.0;
        if (!milbrandt_detail::diagnose_triple_gamma_shape(
                q, n, z, law, alpha, lambda, error))
        {
            throw std::runtime_error(
                "internal Milbrandt FrozenDiffusionOnly source moment rejected: " + error);
        }
    }
    std::array<Field3D, 8> candidate_rates;
    for (auto& field : candidate_rates)
    {
        field.resize(density.size_r(), density.size_th(), density.size_z(), 0.0f);
    }
    std::array<Field3D, 5> mass_after;
    std::array<Field3D, 5> number_after;
    std::array<Field3D, 5> reflectivity_after;
    for (std::size_t index = 0; index < number.size(); ++index)
    {
        mass_after[index] = *mass[index];
        number_after[index] = *number[index];
        reflectivity_after[index] = internal.reflectivity[index];
    }
    Field3D cloud_number_after(Nc_);
    Field3D vapor_after(vapor);
    Field3D cloud_after(cloud);
    Field3D theta_after(theta);
    Field3D pressure_before(p);
    Field3D density_before(density);
    const auto diffusion_species = species == milbrandt_detail::TripleMomentSpecies::Graupel
        ? milbrandt_detail::ColdFrozenDiffusionSpecies::Graupel
        : milbrandt_detail::ColdFrozenDiffusionSpecies::Hail;
    const auto domain = physical_cell_domain(
        density.size_r(), density.size_th(), density.size_z());
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double temperature = thermodynamics::theta_to_temperature(
                    theta(i, j, k), p(i, j, k));
                const std::array<double, 16> cell{
                    temperature, p(i, j, k), density(i, j, k),
                    vapor(i, j, k), cloud(i, j, k), (*mass[0])(i, j, k),
                    (*mass[1])(i, j, k), (*mass[2])(i, j, k),
                    (*mass[3])(i, j, k), (*mass[4])(i, j, k),
                    Nc_(i, j, k), Nr_(i, j, k), Ni_(i, j, k),
                    Ns_(i, j, k), Ng_(i, j, k), Nh_(i, j, k)};
                std::array<float, 14> rates{};
                milbrandt_detail::ColdFrozenDiffusionTripleEndpoint result;
                if (!milbrandt_detail::evaluate_published_triple_frozen_diffusion_cell(
                        cell, dt, diffusion_species,
                        {(*mass[selected])(i, j, k), (*number[selected])(i, j, k),
                         internal.reflectivity[selected](i, j, k)},
                        rates, result, error))
                {
                    throw std::runtime_error(
                        "internal Milbrandt FrozenDiffusionOnly cell rejected: " + error);
                }
                for (std::size_t index = 0; index < rates.size(); ++index)
                {
                    if (!std::isfinite(rates[index]) ||
                        (index != selected + 1 && index != 6 && index != 7 &&
                         index != selected + 9 && rates[index] != 0.0f))
                    {
                        throw std::runtime_error(
                            "internal Milbrandt FrozenDiffusionOnly rate or channel is invalid");
                    }
                }
                const double q_next = static_cast<double>((*mass[selected])(i, j, k))
                    + dt * rates[selected + 1];
                const double qv_next = static_cast<double>(vapor(i, j, k))
                    + dt * rates[6];
                const double dtheta =
                    thermodynamics::temperature_tendency_to_theta(rates[7], p(i, j, k));
                const double final_temperature = temperature + dt * rates[7];
                if (!std::isfinite(q_next) || q_next < 0.0 ||
                    q_next > std::numeric_limits<float>::max() ||
                    !std::isfinite(qv_next) || qv_next < 0.0 ||
                    qv_next > std::numeric_limits<float>::max() ||
                    !std::isfinite(dtheta) ||
                    std::abs(dtheta) > std::numeric_limits<float>::max() ||
                    !within_two_ulps(static_cast<float>(q_next), result.mass_per_kg))
                {
                    throw std::runtime_error(
                        "internal Milbrandt FrozenDiffusionOnly rounded q/vapor/theta candidate is invalid");
                }
                candidate_rates[0](i, j, k) = static_cast<float>(dtheta);
                candidate_rates[1](i, j, k) = rates[6];
                candidate_rates[selected + 3](i, j, k) = rates[selected + 1];
                const double theta_unrounded = static_cast<double>(theta(i, j, k))
                    + dt * candidate_rates[0](i, j, k);
                theta_after(i, j, k) = static_cast<float>(theta_unrounded);
                double unrounded_temperature = 0.0;
                double rounded_temperature = 0.0;
                if (!std::isfinite(final_temperature) || final_temperature < 150.0 ||
                    final_temperature >= microphysics_constants::T0 ||
                    !milbrandt_detail::internal_triple_raw_temperature_pair(theta_unrounded, theta_after(i, j, k),
                        p(i, j, k), unrounded_temperature, rounded_temperature) ||
                    unrounded_temperature < 150.0 || unrounded_temperature >= microphysics_constants::T0 ||
                    !std::isfinite(rounded_temperature) || rounded_temperature < 150.0 ||
                    rounded_temperature >= microphysics_constants::T0)
                {
                    throw std::runtime_error(
                        "internal Milbrandt FrozenDiffusionOnly final heat exits the cold regime");
                }
                const float rounded_q = static_cast<float>(q_next);
                if (!(rounded_q == 0.0f && result.number_per_kg == 0.0f &&
                      result.reflectivity_per_kg == 0.0f))
                {
                    double alpha = 0.0;
                    double lambda = 0.0;
                    if (!milbrandt_detail::diagnose_triple_gamma_shape(
                            rounded_q, result.number_per_kg,
                            result.reflectivity_per_kg, law, alpha, lambda, error))
                    {
                        throw std::runtime_error(
                            "internal Milbrandt FrozenDiffusionOnly rounded moment is invalid: " + error);
                    }
                }
                vapor_after(i, j, k) = static_cast<float>(qv_next);
                mass_after[selected](i, j, k) = rounded_q;
                number_after[selected](i, j, k) = result.number_per_kg;
                reflectivity_after[selected](i, j, k) = result.reflectivity_per_kg;
            }
        }
    }
    Field3D surface_sink(density.size_r(), density.size_th(), 1, 0.0f);
    std::vector<std::array<milbrandt_detail::TripleMomentSurfaceInventory, 5>>
        surface(static_cast<std::size_t>(density.size_r()) * density.size_th());
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
    internal.surface_sink_after = std::move(surface_sink);
    internal.surface_after = std::move(surface);
    internal.candidate_dt = dt;
    internal.candidate_valid = true;
}

} // namespace tmv
