#include "microphysics/schemes/thompson/thompson.hpp"
#include "compute/backend_runtime.hpp"
#include "microphysics/schemes/thompson/moment_initialization.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace tmv
{
bool ThompsonScheme::try_native_sedimentation(const Field3D&                       p,
                                              const Field3D&                       theta,
                                              const Field3D&                       rho_d,
                                              const std::array<const Field3D*, 5>& mass,
                                              const std::array<Field3D*, 5>&       tendency,
                                              const Field3D&                       snow_fall_speed_multiplier,
                                              thompson::PrognosticState&           state,
                                              Field3D&                             surface_sink_candidate,
                                              double                               dt)
{
    auto* runtime = compute_runtime();
    auto* kernels = runtime != nullptr && runtime->backend ? runtime->backend->microphysics_kernels() : nullptr;
    if (kernels == nullptr || !kernels->supports(MicrophysicsKernelOp::ThompsonMomentSedimentation))
    {
        return false;
    }
    const int  nr = p.size_r(), nth = p.size_th(), nz = p.size_z();
    const auto domain = physical_cell_domain(nr, nth, nz);
    Field3D    temperature(nr, nth, nz, 0.0f), thickness(nr, nth, nz, 0.0f), velocity(nr, nth, nz, 0.0f);
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                temperature(i, j, k) = thermodynamics::theta_to_temperature(theta(i, j, k), p(i, j, k));
                thickness(i, j, k)   = static_cast<float>(vertical_spacing_m(i, j, k));
                velocity(i, j, k)    = static_cast<float>(vertical_velocity_at_scalar(i, j, k));
            }
        }
    }
    const std::array<const Field3D*, 3> number{&state.cloud_number(), &state.rain_number(), &state.ice_number()};
    const std::array<Field3D*, 3>       number_tendency{
        &state.cloud_number_tendency(), &state.rain_number_tendency(), &state.ice_number_tendency()};
    std::array<Field3D, 5> mass_changes;
    std::array<Field3D, 3> number_changes;
    Field3D donor_mass(nr, nth, nz, 0.0f), donor_number(nr, nth, nz, 0.0f), unused_number(nr, nth, nz, 0.0f);
    Field3D surface(nr, nth, 1, 0.0f);
    std::vector<double>                 precipitation(surface.size(), 0.0);
    ThompsonMomentSedimentationDispatch dispatch;
    dispatch.mass                  = donor_mass.data();
    dispatch.number                = donor_number.data();
    dispatch.density               = rho_d.data();
    dispatch.temperature           = temperature.data();
    dispatch.thickness             = thickness.data();
    dispatch.vertical_velocity     = velocity.data();
    dispatch.snow_speed_multiplier = snow_fall_speed_multiplier.data();
    dispatch.surface_mass          = surface.data();
    dispatch.nx                    = nr;
    dispatch.ny                    = nth;
    dispatch.nz                    = nz;
    dispatch.i_begin               = domain.i_begin;
    dispatch.i_end                 = domain.i_end;
    dispatch.j_begin               = domain.j_begin;
    dispatch.j_end                 = domain.j_end;
    dispatch.k_begin               = domain.k_begin;
    dispatch.k_end                 = domain.k_end;
    dispatch.dt                    = static_cast<float>(dt);
    for (std::size_t species = 0; species < mass.size(); ++species)
    {
        for (int i = domain.i_begin; i < domain.i_end; ++i)
        {
            for (int j = domain.j_begin; j < domain.j_end; ++j)
            {
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    // The process owner has validated cancellation envelopes before this packing step.
                    donor_mass(i, j, k) = static_cast<float>(std::max(
                        0.0, static_cast<double>((*mass[species])(i, j, k)) + dt * (*tendency[species])(i, j, k)));
                    donor_number(i, j, k) =
                        species < 3 ? static_cast<float>(thompson::transport_number_candidate(
                                          (*number[species])(i, j, k), (*number_tendency[species])(i, j, k), dt))
                                    : 0.0f;
                }
            }
        }
        mass_changes[species] = Field3D(nr, nth, nz, 0.0f);
        if (species < 3)
        {
            number_changes[species] = Field3D(nr, nth, nz, 0.0f);
        }
        dispatch.species         = static_cast<ThompsonSedimentationSpecies>(species);
        dispatch.mass_tendency   = mass_changes[species].data();
        dispatch.number_tendency = species < 3 ? number_changes[species].data() : unused_number.data();
        if (!kernels->dispatch_thompson_moment_sedimentation(dispatch))
        {
            throw std::runtime_error("native Thompson moment sedimentation failed");
        }
        for (std::size_t n = 0; n < precipitation.size(); ++n)
        {
            precipitation[n] += surface.data()[n];
        }
    }
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                for (std::size_t species = 0; species < mass.size(); ++species)
                {
                    (*tendency[species])(i, j, k) += mass_changes[species](i, j, k);
                    if (species < 3)
                    {
                        (*number_tendency[species])(i, j, k) += number_changes[species](i, j, k);
                    }
                }
            }
            surface_sink_candidate(i, j, 0) =
                static_cast<float>(precipitation[static_cast<std::size_t>(i) * nth + j] /
                                   (dt * rho_d(i, j, domain.k_begin) * vertical_spacing_m(i, j, domain.k_begin)));
        }
    }
    return true;
}
} // namespace tmv
