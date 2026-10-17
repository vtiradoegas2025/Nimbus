#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace tmv
{
struct MilbrandtSedimentationWorkspace
{
    std::vector<double> q_work;
    std::vector<double> n_work;
    std::vector<double> q_next;
    std::vector<double> n_next;
    std::vector<double> velocity_mass;
    std::vector<double> velocity_number;
    std::vector<double> thickness;

    void resize(std::size_t levels)
    {
        q_work.resize(levels);
        n_work.resize(levels);
        q_next.resize(levels);
        n_next.resize(levels);
        velocity_mass.resize(levels);
        velocity_number.resize(levels);
        thickness.resize(levels);
    }
};
namespace milbrandt_detail
{
inline bool sediment_column(const std::array<double, 7>&     parameters,
                            int                              levels,
                            const float*                     mass,
                            const float*                     number,
                            const float*                     dry_density,
                            const double*                    thickness,
                            double                           dt,
                            MilbrandtSedimentationWorkspace& workspace,
                            float*                           mass_rate,
                            float*                           number_rate,
                            double&                          surface_loss)
{
    const double alpha            = parameters[0];
    const double c_mass           = parameters[1];
    const double d_mass           = parameters[2];
    const double a_fall           = parameters[3];
    const double b_fall           = parameters[4];
    const double maximum_speed    = parameters[5];
    const double maximum_diameter = parameters[6];
    bool         succeeded        = true;
    auto&        q_work           = workspace.q_work;
    auto&        n_work           = workspace.n_work;
    auto&        q_next           = workspace.q_next;
    auto&        n_next           = workspace.n_next;
    auto&        velocity_mass    = workspace.velocity_mass;
    auto&        velocity_number  = workspace.velocity_number;

    for (int level = 0; level < levels; ++level)
    {
        const int k = 0 + level;
        q_work[static_cast<std::size_t>(level)] =
            std::max(0.0, static_cast<double>(mass[k]) + dt * static_cast<double>(mass_rate[k]));
        n_work[static_cast<std::size_t>(level)] =
            std::max(0.0, static_cast<double>(number[k]) + dt * static_cast<double>(number_rate[k]));

        if (q_work[static_cast<std::size_t>(level)] <= 1.0e-14)
        {
            q_work[static_cast<std::size_t>(level)] = 0.0;
            n_work[static_cast<std::size_t>(level)] = 0.0;
        }
        else if (n_work[static_cast<std::size_t>(level)] <= 0.0)
        {
            // A mass-bearing category must have a realizable
            // distribution. Seed the smallest number compatible
            // with its qualified maximum mean-mass diameter.
            n_work[static_cast<std::size_t>(level)] =
                q_work[static_cast<std::size_t>(level)] / (c_mass * std::pow(maximum_diameter, d_mass));
        }
    }

    surface_loss     = 0.0;
    double remaining = dt;
    int    substeps  = 0;
    while (remaining > 1.0e-12 * dt)
    {
        double stable_step      = remaining;
        bool   any_falling_mass = false;
        for (int level = 0; level < levels; ++level)
        {
            const std::size_t index = static_cast<std::size_t>(level);
            velocity_mass[index]    = 0.0;
            velocity_number[index]  = 0.0;
            if (!(q_work[index] > 0.0) || !(n_work[index] > 0.0))
            {
                continue;
            }

            const int    k              = 0 + level;
            const double density        = dry_density[k];
            const double number_volume  = density * n_work[index];
            const double i_lambda       = std::pow(density * q_work[index] * std::tgamma(1.0 + alpha) /
                                                       (c_mass * number_volume * std::tgamma(1.0 + alpha + d_mass)),
                                                   1.0 / d_mass);
            const double density_factor = std::sqrt(1.225 / density);
            const double common         = a_fall * density_factor * std::pow(std::max(i_lambda, 1.0e-10), b_fall);
            velocity_mass[index]        = std::min(
                maximum_speed, common * std::tgamma(1.0 + alpha + d_mass + b_fall) / std::tgamma(1.0 + alpha + d_mass));
            velocity_number[index] =
                std::min(maximum_speed, common * std::tgamma(1.0 + alpha + b_fall) / std::tgamma(1.0 + alpha));
            const double maximum_velocity = std::max(velocity_mass[index], velocity_number[index]);
            if (maximum_velocity > 0.0)
            {
                any_falling_mass = true;
                stable_step      = std::min(stable_step, 0.8 * thickness[k] / maximum_velocity);
            }
        }

        if (!any_falling_mass)
        {
            break;
        }
        if (stable_step < remaining / 4096.0)
        {
            // Extremely thin terrain layers can imply millions of
            // explicit CFL passes. Conservatively remap each
            // donor's rho*scalar*dz parcel across the complete
            // fall distance for the remaining interval. This
            // branch is positive, permits multi-layer transit,
            // and retains exact live-density/layer-volume mass.
            const auto remap_parcels = [&](const std::vector<double>& source,
                                           const std::vector<double>& velocity,
                                           std::vector<double>&       destination,
                                           bool                       record_surface)
            {
                std::fill(destination.begin(), destination.end(), 0.0);
                for (int source_level = 0; source_level < levels; ++source_level)
                {
                    const std::size_t source_index     = static_cast<std::size_t>(source_level);
                    const int         source_k         = 0 + source_level;
                    const double      source_dz        = thickness[source_k];
                    const double      source_rho       = dry_density[source_k];
                    const double      courant_distance = velocity[source_index] * remaining;
                    const double      moving_fraction  = std::clamp(courant_distance / source_dz, 0.0, 1.0);
                    const double      inventory        = source_rho * source[source_index] * source_dz;
                    const double      moving           = moving_fraction * inventory;
                    const double      retained         = inventory - moving;
                    destination[source_index] += retained / (source_rho * source_dz);
                    if (!(moving > 0.0))
                    {
                        continue;
                    }

                    double residual_distance = std::max(0.0, courant_distance - source_dz);
                    int    destination_level = source_level - 1;
                    while (destination_level >= 0)
                    {
                        const int    destination_k  = 0 + destination_level;
                        const double destination_dz = thickness[destination_k];
                        if (residual_distance < destination_dz)
                        {
                            break;
                        }
                        residual_distance -= destination_dz;
                        --destination_level;
                    }
                    if (destination_level >= 0)
                    {
                        const int destination_k = 0 + destination_level;
                        destination[static_cast<std::size_t>(destination_level)] +=
                            moving / (static_cast<double>(dry_density[destination_k]) * thickness[destination_k]);
                    }
                    else if (record_surface)
                    {
                        surface_loss += moving / (static_cast<double>(dry_density[0]) * thickness[0]);
                    }
                }
            };

            remap_parcels(q_work, velocity_mass, q_next, true);
            remap_parcels(n_work, velocity_number, n_next, false);
            q_work.swap(q_next);
            n_work.swap(n_next);
            remaining = 0.0;
            break;
        }
        if (!(stable_step > 0.0) || !std::isfinite(stable_step) || ++substeps > 100000)
        {
            succeeded = false;
            break;
        }

        q_next = q_work;
        n_next = n_work;
        for (int level = 0; level < levels; ++level)
        {
            const std::size_t index = static_cast<std::size_t>(level);
            if (!(q_work[index] > 0.0) || !(n_work[index] > 0.0))
            {
                continue;
            }
            const int    k        = 0 + level;
            const double dz_donor = thickness[k];
            const double q_out    = q_work[index] * velocity_mass[index] * stable_step / dz_donor;
            const double n_out    = n_work[index] * velocity_number[index] * stable_step / dz_donor;
            q_next[index] -= q_out;
            n_next[index] -= n_out;

            if (level > 0)
            {
                const int    receiver_k      = k - 1;
                const double receiver_weight = static_cast<double>(dry_density[k]) /
                                               static_cast<double>(dry_density[receiver_k]) * dz_donor /
                                               thickness[receiver_k];
                q_next[index - 1] += receiver_weight * q_out;
                n_next[index - 1] += receiver_weight * n_out;
            }
            else
            {
                surface_loss += q_out;
            }
        }

        for (int level = 0; level < levels; ++level)
        {
            const std::size_t index = static_cast<std::size_t>(level);
            q_work[index]           = std::max(0.0, q_next[index]);
            n_work[index]           = std::max(0.0, n_next[index]);
        }
        remaining -= stable_step;
    }

    for (int level = 0; level < levels; ++level)
    {
        const int k = 0 + level;
        mass_rate[k] =
            static_cast<float>((q_work[static_cast<std::size_t>(level)] - static_cast<double>(mass[k])) / dt);
        number_rate[k] =
            static_cast<float>((n_work[static_cast<std::size_t>(level)] - static_cast<double>(number[k])) / dt);
    }

    return succeeded;
}
} // namespace milbrandt_detail
} // namespace tmv
