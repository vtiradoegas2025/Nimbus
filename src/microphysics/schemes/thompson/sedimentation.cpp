#include "microphysics/schemes/thompson/thompson.hpp"

#include "microphysics/schemes/thompson/moment_initialization.hpp"
#include "microphysics/schemes/thompson/particle_distributions.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tmv
{

void ThompsonScheme::compute_sedimentation(
    const Field3D& p,
    const Field3D& theta,
    const Field3D& rho_d,
    const Field3D& qc,
    const Field3D& qr,
    const Field3D& qi,
    const Field3D& qs,
    const Field3D& qg,
    Field3D& dqc_dt,
    Field3D& dqr_dt,
    Field3D& dqi_dt,
    Field3D& dqs_dt,
    Field3D& dqg_dt,
    const Field3D& snow_fall_speed_multiplier,
    thompson::PrognosticState* state,
    Field3D& surface_sink_candidate,
    double dt_micro)
{
    const int nr = qr.size_r();
    const int nth = qr.size_th();
    const int nz = qr.size_z();
    const PhysicalCellDomain domain = physical_cell_domain(nr, nth, nz);
    const int levels = domain.k_end - domain.k_begin;
    if (snow_fall_speed_multiplier.size_r() != nr
        || snow_fall_speed_multiplier.size_th() != nth
        || snow_fall_speed_multiplier.size_z() != nz)
        throw std::invalid_argument(
            "invalid Thompson sedimentation multiplier shape");

    if (state != nullptr && try_native_sedimentation(p,
                                                     theta,
                                                     rho_d,
                                                     {&qc, &qr, &qi, &qs, &qg},
                                                     {&dqc_dt, &dqr_dt, &dqi_dt, &dqs_dt, &dqg_dt},
                                                     snow_fall_speed_multiplier,
                                                     *state,
                                                     surface_sink_candidate,
                                                     dt_micro))
    {
        return;
    }

    std::atomic<bool> sedimentation_failed{false};
    #pragma omp parallel for collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            if (sedimentation_failed.load(std::memory_order_relaxed))
                continue;
            try
            {
                std::vector<double> density(static_cast<std::size_t>(levels));
                std::vector<double> spacing(static_cast<std::size_t>(levels));
                std::vector<double> temperature(static_cast<std::size_t>(levels));
                for (int local = 0; local < levels; ++local)
                {
                    const int k = domain.k_begin + local;
                    density[static_cast<std::size_t>(local)] = rho_d[i][j][k];
                    spacing[static_cast<std::size_t>(local)] =
                        vertical_spacing_m(i, j, k);
                    temperature[static_cast<std::size_t>(local)] =
                        thermodynamics::theta_to_temperature(
                            theta[i][j][k], p[i][j][k]);
                    if (!std::isfinite(density[static_cast<std::size_t>(local)])
                        || density[static_cast<std::size_t>(local)] <= 0.0
                        || !std::isfinite(spacing[static_cast<std::size_t>(local)])
                        || spacing[static_cast<std::size_t>(local)] <= 0.0)
                        throw std::invalid_argument(
                            "invalid Thompson sedimentation column");
                }

                double surface_precipitation_kg_m2 = 0.0;
                const auto transport = [&](const Field3D& source,
                                           Field3D& tendency,
                                           std::vector<double> mass_speed,
                                           const Field3D* number,
                                           Field3D* number_tendency,
                                           std::vector<double> number_speed,
                                           bool extend_speed,
                                           bool number_controls_cfl) {
                    std::vector<double> mass(static_cast<std::size_t>(levels));
                    std::vector<double> initial_mass(
                        static_cast<std::size_t>(levels));
                    std::vector<double> number_values(
                        static_cast<std::size_t>(levels));
                    std::vector<double> initial_number(
                        static_cast<std::size_t>(levels));
                    for (int local = 0; local < levels; ++local)
                    {
                        const int k = domain.k_begin + local;
                        const auto index = static_cast<std::size_t>(local);
                        const float source_value = source[i][j][k];
                        const float tendency_value = tendency[i][j][k];
                        const double mixing = source_value
                            + dt_micro * tendency_value;
                        const double storage_envelope = 2.0e-12 + 2.0 * (
                            static_cast<double>(std::nextafter(
                                source_value,
                                std::numeric_limits<float>::infinity()))
                                - static_cast<double>(source_value)
                            + std::abs(dt_micro) * (
                                static_cast<double>(std::nextafter(
                                    tendency_value,
                                    std::numeric_limits<float>::infinity()))
                                - static_cast<double>(tendency_value)));
                        if (!std::isfinite(mixing)
                            || mixing < -storage_envelope)
                            throw std::runtime_error(
                                "invalid Thompson sedimentation mass donor");
                        mass[index] = density[index] * std::max(0.0, mixing);
                        initial_mass[index] = mass[index];
                        if (number && number_tendency)
                        {
                            const double number_mixing = (*number)[i][j][k] + dt_micro * (*number_tendency)[i][j][k];
                            if (!std::isfinite(number_mixing) || number_mixing < -1.0e-8)
                                throw std::runtime_error("invalid Thompson sedimentation number donor");
                            number_values[index] =
                                density[index] * thompson::transport_number_candidate(
                                                     (*number)[i][j][k], (*number_tendency)[i][j][k], dt_micro);
                            initial_number[index] = number_values[index];
                        }
                    }
                    if (extend_speed)
                        for (int local = levels - 2; local >= 0; --local)
                        {
                            const auto index = static_cast<std::size_t>(local);
                            if (mass[index] <= 1.0e-12)
                            {
                                mass_speed[index] = mass_speed[index + 1];
                                if (number && number_tendency)
                                    number_speed[index] =
                                        number_speed[index + 1];
                            }
                        }
                    int substeps = 1;
                    if (extend_speed)
                        for (int local = 0; local < levels; ++local)
                        {
                            const auto index = static_cast<std::size_t>(local);
                            double speed = mass_speed[index];
                            if (number_controls_cfl)
                                speed = std::max(speed, number_speed[index]);
                            if (speed > 1.0e-3)
                            {
                                const double courant = dt_micro * speed / spacing[index];
                                if (!std::isfinite(courant) || courant >= 4096.0)
                                {
                                    throw std::runtime_error("Thompson sedimentation substep budget exceeded");
                                }
                                substeps = std::max(substeps, static_cast<int>(courant) + 1);
                            }
                        }
                    const double substep_dt = dt_micro / substeps;
                    std::vector<double> mass_flux(
                        static_cast<std::size_t>(levels));
                    std::vector<double> number_flux(
                        static_cast<std::size_t>(levels));
                    std::vector<double> next_mass(
                        static_cast<std::size_t>(levels));
                    std::vector<double> next_number(
                        static_cast<std::size_t>(levels));
                    for (int step = 0; step < substeps; ++step)
                    {
                        for (int local = 0; local < levels; ++local)
                        {
                            const auto index = static_cast<std::size_t>(local);
                            mass_flux[index] = mass_speed[index] * mass[index];
                            if (number && number_tendency)
                                number_flux[index] = number_speed[index]
                                    * number_values[index];
                        }
                        next_mass = mass;
                        if (number && number_tendency)
                            next_number = number_values;
                        for (int local = 0; local < levels; ++local)
                        {
                            const auto index = static_cast<std::size_t>(local);
                            const double removed = mass_flux[index]
                                * substep_dt / spacing[index];
                            if (removed > mass[index] + 1.0e-12)
                                throw std::runtime_error(
                                    "Thompson sedimentation CFL failure");
                            next_mass[index] = std::max(
                                0.0, next_mass[index] - removed);
                            if (local > 0)
                                next_mass[index - 1] += mass_flux[index]
                                    * substep_dt / spacing[index - 1];
                            else
                                surface_precipitation_kg_m2 +=
                                    mass_flux[index] * substep_dt;
                            if (number && number_tendency)
                            {
                                const double removed_number =
                                    number_flux[index] * substep_dt
                                    / spacing[index];
                                if (removed_number
                                    > number_values[index] + 1.0e-5)
                                    throw std::runtime_error(
                                        "Thompson number sedimentation CFL failure");
                                next_number[index] = std::max(
                                    0.0,
                                    next_number[index] - removed_number);
                                if (local > 0)
                                    next_number[index - 1] +=
                                        number_flux[index] * substep_dt
                                        / spacing[index - 1];
                            }
                        }
                        mass.swap(next_mass);
                        if (number && number_tendency)
                            number_values.swap(next_number);
                    }
                    for (int local = 0; local < levels; ++local)
                    {
                        const int k = domain.k_begin + local;
                        const auto index = static_cast<std::size_t>(local);
                        tendency[i][j][k] += static_cast<float>(
                            (mass[index] - initial_mass[index])
                            / density[index] / dt_micro);
                        if (number && number_tendency)
                            (*number_tendency)[i][j][k]
                                += static_cast<float>(
                                    (number_values[index]
                                     - initial_number[index])
                                    / density[index] / dt_micro);
                    }
                };

                const Field3D* cloud_number = state
                    ? &state->cloud_number() : nullptr;
                const Field3D* rain_number = state
                    ? &state->rain_number() : nullptr;
                const Field3D* ice_number = state
                    ? &state->ice_number() : nullptr;
                Field3D* cloud_number_tendency = state
                    ? &state->cloud_number_tendency() : nullptr;
                Field3D* rain_number_tendency = state
                    ? &state->rain_number_tendency() : nullptr;
                Field3D* ice_number_tendency = state
                    ? &state->ice_number_tendency() : nullptr;

                std::vector<double> cloud_speed(static_cast<std::size_t>(levels));
                std::vector<double> cloud_number_speed(
                    static_cast<std::size_t>(levels));
                std::vector<double> rain_speed(static_cast<std::size_t>(levels));
                std::vector<double> rain_number_speed(
                    static_cast<std::size_t>(levels));
                std::vector<double> ice_speed(static_cast<std::size_t>(levels));
                std::vector<double> ice_number_speed(
                    static_cast<std::size_t>(levels));
                std::vector<double> snow_speed(static_cast<std::size_t>(levels));
                std::vector<double> graupel_speed(
                    static_cast<std::size_t>(levels));
                double height = 0.0;
                for (int local = 0; local < levels; ++local)
                {
                    const int k = domain.k_begin + local;
                    const auto index = static_cast<std::size_t>(local);
                    const double cloud_mass = density[index] * std::max(
                        0.0, static_cast<double>(qc[i][j][k])
                            + dt_micro * dqc_dt[i][j][k]);
                    const double rain_mass = density[index] * std::max(
                        0.0, static_cast<double>(qr[i][j][k])
                            + dt_micro * dqr_dt[i][j][k]);
                    const double ice_mass = density[index] * std::max(
                        0.0, static_cast<double>(qi[i][j][k])
                            + dt_micro * dqi_dt[i][j][k]);
                    const double snow_mass = density[index] * std::max(
                        0.0, static_cast<double>(qs[i][j][k])
                            + dt_micro * dqs_dt[i][j][k]);
                    const double graupel_mass = density[index] * std::max(
                        0.0, static_cast<double>(qg[i][j][k])
                            + dt_micro * dqg_dt[i][j][k]);
                    const double cloud_number_m3 =
                        state ? density[index] * thompson::transport_number_candidate((*cloud_number)[i][j][k],
                                                                                      (*cloud_number_tendency)[i][j][k],
                                                                                      dt_micro)
                              : 0.0;
                    const double rain_number_m3 =
                        state ? density[index] * thompson::transport_number_candidate((*rain_number)[i][j][k],
                                                                                      (*rain_number_tendency)[i][j][k],
                                                                                      dt_micro)
                              : 0.0;
                    const double ice_number_m3 =
                        state ? density[index] * thompson::transport_number_candidate(
                                                     (*ice_number)[i][j][k], (*ice_number_tendency)[i][j][k], dt_micro)
                              : 0.0;
                    if (state && height <= 500.0 && vertical_velocity_at_scalar(i, j, k) < 0.1)
                    {
                        const auto speeds  = thompson::cloud_fall_speeds(cloud_mass, cloud_number_m3, density[index]);
                        cloud_speed[index] = speeds.mass_m_s;
                        cloud_number_speed[index] = speeds.number_m_s;
                    }
                    if (state)
                    {
                        const auto rain   = thompson::rain_fall_speeds(rain_mass, rain_number_m3, density[index]);
                        rain_speed[index] = rain.mass_m_s;
                        rain_number_speed[index] = rain.number_m_s;
                        const auto ice = thompson::ice_fall_speeds(
                            ice_mass, ice_number_m3, density[index]);
                        ice_speed[index] = ice.mass_m_s;
                        ice_number_speed[index] = ice.number_m_s;
                    }
                    else
                    {
                        const double diagnosed = rain_mass > 1.0e-12
                            ? thompson::normalize_rain_number_m3(
                                rain_mass, 0.0) : 0.0;
                        rain_speed[index] = thompson::rain_fall_speeds(
                            rain_mass, diagnosed, density[index]).mass_m_s;
                    }
                    snow_speed[index] = thompson::snow_distribution(
                        snow_mass, temperature[index], density[index])
                        .mass_fall_speed_m_s
                        * snow_fall_speed_multiplier[i][j][k];
                    graupel_speed[index] = thompson::graupel_fall_speeds(
                        graupel_mass, density[index]).mass_m_s;
                    height += spacing[index];
                }
                std::vector<double> no_number_speed(
                    static_cast<std::size_t>(levels));
                if (state)
                    transport(qc, dqc_dt, cloud_speed,
                              cloud_number, cloud_number_tendency,
                              cloud_number_speed, false, false);
                transport(qr, dqr_dt, rain_speed,
                          rain_number, rain_number_tendency,
                          state ? rain_number_speed : no_number_speed,
                          true, state);
                if (state)
                    transport(qi, dqi_dt, ice_speed,
                              ice_number, ice_number_tendency,
                              ice_number_speed, true, false);
                transport(qs, dqs_dt, snow_speed,
                          nullptr, nullptr, no_number_speed, true, false);
                transport(qg, dqg_dt, graupel_speed,
                          nullptr, nullptr, no_number_speed, true, false);
                surface_sink_candidate[i][j][0] = static_cast<float>(
                    surface_precipitation_kg_m2
                    / (dt_micro * density.front() * spacing.front()));
            }
            catch (...)
            {
                sedimentation_failed.store(true, std::memory_order_relaxed);
            }
        }
    if (sedimentation_failed.load(std::memory_order_relaxed))
        throw std::runtime_error("Thompson sedimentation failed");
}

double ThompsonScheme::bottom_sedimentation_sink_rate(
    const Field3D& rho_d,
    const Field3D& qr,
    const Field3D& qs,
    const Field3D& qg,
    const Field3D& qh,
    int i,
    int j,
    double dt_micro) const
{
    (void)rho_d;
    (void)qr;
    (void)qs;
    (void)qg;
    (void)qh;
    (void)dt_micro;
    if (i < 0 || j < 0
        || i >= surface_sedimentation_sink_rate_.size_r()
        || j >= surface_sedimentation_sink_rate_.size_th()
        || surface_sedimentation_sink_rate_.size_z() == 0)
        return 0.0;
    return std::max(
        0.0, static_cast<double>(surface_sedimentation_sink_rate_[i][j][0]));
}

} // namespace tmv
