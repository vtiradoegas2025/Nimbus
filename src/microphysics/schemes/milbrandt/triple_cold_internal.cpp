#include "milbrandt.hpp"
#include "cold.hpp"
#include "triple_cold_freezing.hpp"
#include "triple_internal_thermal.hpp"

#include "core/runtime/simulation_state.hpp"
#include "microphysics/milbrandt_cold_parameters.hpp"

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

milbrandt_detail::ColdParameters freezing_parameters(bool hail)
{
    const auto& p = milbrandt_detail::default_cold_parameters;
    return {p[0], p[1], p[2], p[3], p[4],
            p[5], p[6], p[7], p[8], p[9],
            p[10], p[11], p[12], p[13], p[14],
            p[15], p[16], p[17], p[18], p[19],
            p[20], p[21], p[22], p[23], p[24], hail};
}

bool pure_freezing_rates(const std::array<float, 14>& rates,
                         const milbrandt_detail::ColdRainFreezingAcceptedEvent& event,
                         bool hail,
                         double dt) noexcept
{
    std::array<float, 14> expected{};
    const float mass_rate = static_cast<float>(
        event.admitted_mass_increment_per_kg / dt);
    const float number_rate = static_cast<float>(
        event.admitted_number_increment_per_kg / dt);
    expected[1] = -mass_rate;
    expected[hail ? 5 : 4] = mass_rate;
    expected[7] = static_cast<float>(
        microphysics_constants::L_f * event.admitted_mass_increment_per_kg
        / (microphysics_constants::cp * dt));
    expected[9] = -number_rate;
    expected[hail ? 13 : 12] = number_rate;
    return rates == expected;
}

} // namespace

bool MilbrandtScheme::enable_internal_triple_rain_freezing_for_test(
    const SimulationStateView& state,
    const Field3D& rain_number,
    const Field3D& rain_reflectivity,
    std::string& error)
{
    if (internal_triple_ ||
        !validate_internal_triple_source_profile(state, true, error))
    {
        if (error.empty())
        {
            error = "internal Milbrandt triple modes are mutually exclusive";
        }
        return false;
    }
    const std::array<double, 25> physical_parameters{
        a_g_, a_h_, a_i_, a_r_, a_s_,
        alpha_g_, alpha_h_, alpha_i_, alpha_r_, alpha_s_,
        b_g_, b_h_, b_i_, b_r_, b_s_,
        c_g_, c_h_, c_i_, c_r_, c_s_,
        d_g_, d_h_, d_i_, d_r_, d_s_};
    const auto& f = state.fields;
    if (physical_parameters != milbrandt_detail::default_cold_parameters ||
        !matches(f.rho, rain_number) ||
        !matches(f.rho, rain_reflectivity))
    {
        error = "internal Milbrandt RainFreezingOnly moment shape or cold law differs";
        return false;
    }
    for (const Field3D* other : {&Nc_, &Ni_, &Ns_, &Ng_, &Nh_})
    {
        if (!other->empty() &&
            (!matches(f.rho, *other) || !zero_field(*other)))
        {
            error = "internal Milbrandt RainFreezingOnly has an unowned retained number moment";
            return false;
        }
    }
    for (std::size_t index = 0; index < f.rho.size(); ++index)
    {
        const double temperature = thermodynamics::theta_to_temperature(
            f.theta.data()[index], f.p.data()[index]);
        if (temperature < microphysics_constants::T0 - 5.0 &&
            f.qv.data()[index] / std::max(
                thermodynamics::saturation_mixing_ratio_ice(
                    temperature, f.p.data()[index]), 1.0e-30) > 1.08)
        {
            error = "internal Milbrandt RainFreezingOnly has unowned primary ice nucleation";
            return false;
        }
        const double q = f.qr.data()[index];
        const double n = rain_number.data()[index];
        const double z = rain_reflectivity.data()[index];
        if (q == 0.0 && n == 0.0 && z == 0.0)
        {
            continue;
        }
        double alpha = 0.0;
        double lambda = 0.0;
        if (!milbrandt_detail::diagnose_triple_gamma_shape(
                q, n, z, {c_r_, d_r_}, alpha, lambda, error))
        {
            return false;
        }
    }
    MilbrandtScheme candidate(*this);
    candidate.Nr_ = rain_number;
    for (Field3D* other : {&candidate.Nc_, &candidate.Ni_,
                           &candidate.Ns_, &candidate.Ng_, &candidate.Nh_,
                           &candidate.dNc_dt_, &candidate.dNr_dt_,
                           &candidate.dNi_dt_, &candidate.dNs_dt_,
                           &candidate.dNg_dt_, &candidate.dNh_dt_})
    {
        other->resize(f.rho.size_r(), f.rho.size_th(), f.rho.size_z(), 0.0f);
    }
    candidate.surface_sedimentation_sink_rate_.resize(
        f.rho.size_r(), f.rho.size_th(), 1, 0.0f);
    InternalTripleCandidateState internal;
    internal.mode = InternalTripleMode::RainFreezingOnly;
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

void MilbrandtScheme::compute_internal_triple_rain_freezing_tendencies(
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
    if (internal.mode != InternalTripleMode::RainFreezingOnly ||
        internal.poisoned || !internal.warm_storage_owner ||
        !std::isfinite(dt) || dt <= 0.0)
    {
        throw std::runtime_error(
            "internal Milbrandt RainFreezingOnly candidate is unavailable");
    }
    auto live = make_simulation_state_view(*internal.warm_storage_owner);
    std::string error;
    if (&live.fields.p != &p || &live.fields.theta != &theta ||
        &live.fields.rho != &density || &live.fields.qv != &vapor ||
        &live.fields.qc != &cloud || &live.fields.qr != mass[0] ||
        &live.fields.qi != mass[1] || &live.fields.qs != mass[2] ||
        &live.fields.qg != mass[3] || &live.fields.qh != mass[4] ||
        !validate_internal_triple_source_profile(live, true, error))
    {
        throw std::runtime_error(
            "internal Milbrandt RainFreezingOnly runtime owner changed: " + error);
    }
    if (!matches(density, Nr_) ||
        !matches(density, internal.reflectivity[0]))
    {
        throw std::runtime_error(
            "internal Milbrandt RainFreezingOnly private rain shape changed");
    }
    for (const Field3D* other : {&Nc_, &Ni_, &Ns_, &Ng_, &Nh_,
                                 &internal.reflectivity[1],
                                 &internal.reflectivity[2],
                                 &internal.reflectivity[3],
                                 &internal.reflectivity[4]})
    {
        if (!matches(density, *other) || !zero_field(*other))
        {
            throw std::runtime_error(
                "internal Milbrandt RainFreezingOnly gained an unowned moment");
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
    const std::array<const Field3D*, 5> number{
        &Nr_, &Ni_, &Ns_, &Ng_, &Nh_};
    for (std::size_t species = 0; species < 5; ++species)
    {
        mass_after[species] = *mass[species];
        number_after[species] = *number[species];
        reflectivity_after[species] = internal.reflectivity[species];
    }
    Field3D theta_after(theta);
    Field3D pressure_before(p);
    Field3D density_before(density);
    Field3D cloud_number_after(Nc_);
    Field3D vapor_after(vapor);
    Field3D cloud_after(cloud);
    const auto parameters = freezing_parameters(hail_processes_);
    const std::size_t destination = hail_processes_ ? 4 : 3;
    const double frozen_c = hail_processes_ ? c_h_ : c_g_;
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
                milbrandt_detail::ColdRainFreezingAcceptedEventBuffer observer;
                milbrandt_detail::cold_cell(parameters, cell, dt, rates, &observer);
                const auto event = observer.size == 1 ? observer.events[0]
                    : milbrandt_detail::ColdRainFreezingAcceptedEvent{
                        hail_processes_
                            ? milbrandt_detail::ColdRainFreezingDestination::Hail
                            : milbrandt_detail::ColdRainFreezingDestination::Graupel,
                        cell[5], cell[11], 0.0, 0.0, 0.0, cell[2], dt};
                if (observer.size > 1 ||
                    !pure_freezing_rates(rates, event, hail_processes_, dt))
                {
                    throw std::runtime_error(
                        "internal Milbrandt RainFreezingOnly observed an unowned cold channel");
                }
                milbrandt_detail::ColdFreezingTripleMomentResult result;
                if (!milbrandt_detail::apply_equal_fraction_rain_freezing(
                        {cell[5], cell[11], internal.reflectivity[0](i, j, k)},
                        {cell[hail_processes_ ? 9 : 8],
                         cell[hail_processes_ ? 15 : 14],
                         internal.reflectivity[destination](i, j, k)},
                        cell[2], dt, event, {c_r_, d_r_}, {frozen_c, 3.0},
                        result, error))
                {
                    throw std::runtime_error(
                        "internal Milbrandt RainFreezingOnly transfer rejected: " + error);
                }
                const double dtheta =
                    thermodynamics::temperature_tendency_to_theta(rates[7], cell[1]);
                const double final_temperature = temperature + dt * rates[7];
                const double theta_unrounded = static_cast<double>(theta(i, j, k))
                    + dt * static_cast<float>(dtheta);
                const float theta_stored = static_cast<float>(theta_unrounded);
                double unrounded_temperature = 0.0;
                double stored_temperature = 0.0;
                if (!std::isfinite(dtheta) ||
                    std::abs(dtheta) > std::numeric_limits<float>::max() ||
                    !std::isfinite(final_temperature) ||
                    final_temperature >= microphysics_constants::T0 ||
                    !milbrandt_detail::internal_triple_raw_temperature_pair(theta_unrounded, theta_stored,
                        cell[1], unrounded_temperature, stored_temperature) ||
                    unrounded_temperature < 150.0 || unrounded_temperature >= microphysics_constants::T0 ||
                    stored_temperature < 150.0 || stored_temperature >= microphysics_constants::T0)
                {
                    throw std::runtime_error(
                        "internal Milbrandt RainFreezingOnly final heat exits the cold regime");
                }
                candidate_rates[0](i, j, k) = static_cast<float>(dtheta);
                for (const std::size_t species : {std::size_t{0}, destination})
                {
                    const auto& endpoint = species == 0 ? result.rain : result.frozen;
                    const double rate = (static_cast<double>(endpoint.mass_per_kg)
                        - (*mass[species])(i, j, k)) / dt;
                    if (!std::isfinite(rate) ||
                        std::abs(rate) > std::numeric_limits<float>::max())
                    {
                        throw std::runtime_error(
                            "internal Milbrandt RainFreezingOnly mass rate is invalid");
                    }
                    candidate_rates[species + 3](i, j, k) = static_cast<float>(rate);
                    mass_after[species](i, j, k) = endpoint.mass_per_kg;
                    number_after[species](i, j, k) = endpoint.number_per_kg;
                    reflectivity_after[species](i, j, k) = endpoint.reflectivity_per_kg;
                }
                theta_after(i, j, k) = static_cast<float>(
                    static_cast<double>(theta(i, j, k))
                    + dt * candidate_rates[0](i, j, k));
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
