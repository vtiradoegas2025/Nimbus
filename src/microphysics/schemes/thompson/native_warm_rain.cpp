#include "microphysics/schemes/thompson/thompson.hpp"
#include "microphysics/schemes/thompson/warm_rain_processes.hpp"
#include "compute/backend_runtime.hpp"

#include <stdexcept>

namespace tmv
{
bool ThompsonScheme::try_native_warm_rain(const Field3D&               p,
                                          const Field3D&               rho,
                                          const Field3D&               vapor,
                                          const Field3D&               cloud,
                                          const Field3D&               rain,
                                          double                       dt,
                                          const NativeCloudAdjustment& cloud_result,
                                          std::array<Field3D, 10>&     result)
{
    auto* runtime = compute_runtime();
    auto* kernels = runtime && runtime->backend ? runtime->backend->microphysics_kernels() : nullptr;
    if (!kernels || !kernels->supports(MicrophysicsKernelOp::ThompsonWarmRain))
    {
        return false;
    }
    const int  nx = p.size_r(), ny = p.size_th(), nz = p.size_z();
    const auto domain = physical_cell_domain(nx, ny, nz);
    for (auto& field : result)
    {
        field = Field3D(nx, ny, nz, 0.0f);
    }
    ThompsonWarmRainDispatch d;
    d.coefficients           = thompson::native_warm_rain_coefficients();
    d.temperature            = cloud_result.temperature.data();
    d.pressure               = p.data();
    d.density                = rho.data();
    d.vapor                  = vapor.data();
    d.cloud                  = cloud.data();
    d.rain                   = rain.data();
    d.cloud_number           = prognostic_state_.cloud_number().data();
    d.rain_number            = prognostic_state_.rain_number().data();
    d.water_aerosol          = prognostic_state_.water_aerosol().data();
    d.cloud_vapor_change     = cloud_result.vapor_change.data();
    d.cloud_vapor_change_low = cloud_result.vapor_change_low.data();
    d.cloud_number_source    = cloud_result.cloud_number_tendency.data();
    d.cloud_aerosol_source   = cloud_result.aerosol_tendency.data();
    d.cloud_to_rain_high     = result[0].data();
    d.cloud_to_rain_low      = result[1].data();
    d.rain_to_vapor_high     = result[2].data();
    d.rain_to_vapor_low      = result[3].data();
    d.temperature_high       = result[4].data();
    d.temperature_low        = result[5].data();
    d.cloud_number_tendency  = result[6].data();
    d.rain_number_tendency   = result[7].data();
    d.aerosol_tendency       = result[8].data();
    d.status                 = result[9].data();
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
    if (!kernels->dispatch_thompson_warm_rain(d))
    {
        throw std::runtime_error("native Thompson warm-rain processes failed");
    }
    return true;
}
} // namespace tmv
