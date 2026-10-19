#include "microphysics/schemes/thompson/thompson.hpp"
#include "microphysics/schemes/thompson/cloud_phase.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/thompson_phase_contract.hpp"

#include <stdexcept>

namespace tmv
{
void ThompsonScheme::complete_cloud_phase(const Field3D&             p,
                                          const Field3D&             theta,
                                          const Field3D&             rho,
                                          const Field3D&             vapor,
                                          const Field3D&             cloud,
                                          const Field3D&             ice,
                                          const Field3D&             vapor_tendency,
                                          Field3D&                   cloud_tendency,
                                          Field3D&                   ice_tendency,
                                          Field3D&                   temperature_tendency,
                                          thompson::PrognosticState* state,
                                          double                     dt)
{
    const int  nx = p.size_r(), ny = p.size_th(), nz = p.size_z();
    const auto domain = physical_cell_domain(nx, ny, nz);
    Field3D    temperature(nx, ny, nz, 0), current_vapor(nx, ny, nz, 0);
    Field3D    zero;
    if (!state)
        zero = Field3D(nx, ny, nz, 0);
    auto*      runtime = compute_runtime();
    auto*      kernels = runtime && runtime->backend ? runtime->backend->microphysics_kernels() : nullptr;
    const bool native  = kernels && kernels->supports(MicrophysicsKernelOp::ThompsonCloudPhase);
    if (require_native_dispatch() && !native)
        throw std::runtime_error("native Thompson phase ownership is unavailable");
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                temperature(i, j, k) =
                    static_cast<float>(thermodynamics::theta_to_temperature(theta(i, j, k), p(i, j, k)) +
                                       dt * temperature_tendency(i, j, k));
                current_vapor(i, j, k) =
                    static_cast<float>(std::max(0.0, vapor(i, j, k) + dt * vapor_tendency(i, j, k)));
            }
    ThompsonCloudPhaseDispatch d;
    d.temperature          = temperature.data();
    d.vapor                = current_vapor.data();
    d.density              = rho.data();
    d.cloud                = cloud.data();
    d.ice                  = ice.data();
    d.cloud_number         = state ? state->cloud_number().data() : zero.data();
    d.ice_number           = state ? state->ice_number().data() : zero.data();
    d.cloud_tendency       = cloud_tendency.data();
    d.ice_tendency         = ice_tendency.data();
    d.cloud_number_source  = state ? state->cloud_number_tendency().data() : zero.data();
    d.ice_number_source    = state ? state->ice_number_tendency().data() : zero.data();
    d.temperature_tendency = temperature_tendency.data();
    d.nx                   = nx;
    d.ny                   = ny;
    d.nz                   = nz;
    d.i_begin              = domain.i_begin;
    d.i_end                = domain.i_end;
    d.j_begin              = domain.j_begin;
    d.j_end                = domain.j_end;
    d.k_begin              = domain.k_begin;
    d.k_end                = domain.k_end;
    d.dt                   = dt;
    std::array<Field3D, 6> result;
    for (auto& f : result)
        f = Field3D(nx, ny, nz, 0);
    d.next_cloud_tendency       = result[0].data();
    d.next_ice_tendency         = result[1].data();
    d.next_cloud_number_source  = result[2].data();
    d.next_ice_number_source    = result[3].data();
    d.next_temperature_tendency = result[4].data();
    d.status                    = result[5].data();
    if (native)
    {
        if (!kernels->dispatch_thompson_cloud_phase(d))
        {
            throw std::runtime_error("native Thompson cloud-phase completion failed");
        }
    }
    else
    {
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const auto n     = (static_cast<std::size_t>(i) * ny + j) * nz + k;
                    const auto value = thompson::cloud_phase_tendencies({d.temperature[n],
                                                                         d.vapor[n],
                                                                         d.density[n],
                                                                         d.cloud[n],
                                                                         d.ice[n],
                                                                         d.cloud_number[n],
                                                                         d.ice_number[n],
                                                                         {d.cloud_tendency[n],
                                                                          d.ice_tendency[n],
                                                                          d.cloud_number_source[n],
                                                                          d.ice_number_source[n],
                                                                          d.temperature_tendency[n]}},
                                                                        dt);
                    for (std::size_t f = 0; f < value.size(); ++f)
                        result[f].data()[n] = value[f];
                }
    }
    cloud_tendency = std::move(result[0]);
    ice_tendency   = std::move(result[1]);
    if (state)
    {
        state->cloud_number_tendency() = std::move(result[2]);
        state->ice_number_tendency()   = std::move(result[3]);
    }
    temperature_tendency = std::move(result[4]);
}
} // namespace tmv
