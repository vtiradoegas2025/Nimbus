#include "microphysics/schemes/thompson/thompson.hpp"
#include "microphysics/schemes/thompson/native_cloud_tables.hpp"
#include "compute/backend_runtime.hpp"

#include <stdexcept>

namespace tmv
{
bool ThompsonScheme::try_native_cloud_adjustment(const Field3D&         p,
                                                 const Field3D&         theta,
                                                 const Field3D&         rho,
                                                 const Field3D&         vapor,
                                                 const Field3D&         cloud,
                                                 double                 dt,
                                                 NativeCloudAdjustment& result)
{
    auto* runtime = compute_runtime();
    auto* kernels = runtime != nullptr && runtime->backend ? runtime->backend->microphysics_kernels() : nullptr;
    if (!kernels || !kernels->supports(MicrophysicsKernelOp::ThompsonCloudAdjustment))
    {
        return false;
    }
    if (!native_cloud_tables_)
    {
        native_cloud_tables_ = thompson::pack_native_cloud_tables(*activation_table_, *cloud_evaporation_table_);
    }
    const int  nx = p.size_r(), ny = p.size_th(), nz = p.size_z();
    const auto domain = physical_cell_domain(nx, ny, nz);
    Field3D    temperature(nx, ny, nz, 0.0f), velocity(nx, ny, nz, 0.0f);
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                temperature(i, j, k) =
                    static_cast<float>(thermodynamics::theta_to_temperature(theta(i, j, k), p(i, j, k)));
                velocity(i, j, k) = static_cast<float>(vertical_velocity_at_scalar(i, j, k));
            }
        }
    }
    for (auto* field : {&result.vapor_change,
                        &result.vapor_change_low,
                        &result.temperature_change,
                        &result.temperature_change_low,
                        &result.cloud_number_tendency,
                        &result.aerosol_tendency,
                        &result.lookup_index})
    {
        *field = Field3D(nx, ny, nz, 0.0f);
    }
    ThompsonCloudAdjustmentDispatch d;
    d.temperature            = temperature.data();
    d.potential_temperature  = theta.data();
    d.pressure               = p.data();
    d.density                = rho.data();
    d.vapor                  = vapor.data();
    d.cloud                  = cloud.data();
    d.cloud_number           = prognostic_state_.cloud_number().data();
    d.water_aerosol          = prognostic_state_.water_aerosol().data();
    d.velocity               = velocity.data();
    d.tables                 = native_cloud_tables_;
    d.vapor_change           = result.vapor_change.data();
    d.vapor_change_low       = result.vapor_change_low.data();
    d.temperature_change     = result.temperature_change.data();
    d.temperature_change_low = result.temperature_change_low.data();
    d.cloud_number_tendency  = result.cloud_number_tendency.data();
    d.aerosol_tendency       = result.aerosol_tendency.data();
    d.lookup_index           = result.lookup_index.data();
    d.nx                     = nx;
    d.ny                     = ny;
    d.nz                     = nz;
    d.i_begin                = domain.i_begin;
    d.i_end                  = domain.i_end;
    d.j_begin                = domain.j_begin;
    d.j_end                  = domain.j_end;
    d.k_begin                = domain.k_begin;
    d.k_end                  = domain.k_end;
    d.dt                     = dt;
    if (!kernels->dispatch_thompson_cloud_adjustment(d))
    {
        throw std::runtime_error("native Thompson cloud adjustment failed");
    }
    result.temperature = std::move(temperature);
    return true;
}
} // namespace tmv
