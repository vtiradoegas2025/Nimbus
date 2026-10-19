#include "microphysics/schemes/thompson/thompson.hpp"
#include "microphysics/schemes/thompson/native_freezing_tables.hpp"
#include "compute/backend_runtime.hpp"
#include <stdexcept>

namespace tmv
{
bool ThompsonScheme::try_native_freezing(const Field3D&                 rho,
                                         const Field3D&                 cloud,
                                         const Field3D&                 rain,
                                         double                         dt,
                                         const NativeCloudAdjustment&   cloud_result,
                                         const std::array<Field3D, 10>& warm_result,
                                         std::array<Field3D, 13>&       result)
{
    auto* runtime = compute_runtime();
    auto* kernels = runtime && runtime->backend ? runtime->backend->microphysics_kernels() : nullptr;
    if (!kernels || !kernels->supports(MicrophysicsKernelOp::ThompsonFreezing))
        return false;
    const int  nx = rho.size_r(), ny = rho.size_th(), nz = rho.size_z();
    const auto domain = physical_cell_domain(nx, ny, nz);
    for (auto& f : result)
        f = Field3D(nx, ny, nz, 0);
    ThompsonFreezingDispatch d;
    d.temperature              = cloud_result.temperature.data();
    d.density                  = rho.data();
    d.cloud                    = cloud.data();
    d.rain                     = rain.data();
    d.cloud_number             = prognostic_state_.cloud_number().data();
    d.rain_number              = prognostic_state_.rain_number().data();
    d.ice_aerosol              = prognostic_state_.ice_aerosol().data();
    d.cloud_vapor_change       = cloud_result.vapor_change.data();
    d.cloud_vapor_change_low   = cloud_result.vapor_change_low.data();
    d.conversion_high          = warm_result[0].data();
    d.conversion_low           = warm_result[1].data();
    d.evaporation_high         = warm_result[2].data();
    d.evaporation_low          = warm_result[3].data();
    d.cloud_number_source      = warm_result[6].data();
    d.rain_number_source       = warm_result[7].data();
    d.tables                   = thompson::native_freezing_tables();
    d.cloud_frozen_high        = result[0].data();
    d.cloud_frozen_low         = result[1].data();
    d.rain_ice_high            = result[2].data();
    d.rain_ice_low             = result[3].data();
    d.rain_graupel_high        = result[4].data();
    d.rain_graupel_low         = result[5].data();
    d.next_cloud_number_source = result[6].data();
    d.next_rain_number_source  = result[7].data();
    d.rain_ice_number_source   = result[8].data();
    d.cloud_ice_number_source  = result[9].data();
    d.rain_lookup              = result[10].data();
    d.cloud_lookup             = result[11].data();
    d.status                   = result[12].data();
    d.nx                       = nx;
    d.ny                       = ny;
    d.nz                       = nz;
    d.i_begin                  = domain.i_begin;
    d.i_end                    = domain.i_end;
    d.j_begin                  = domain.j_begin;
    d.j_end                    = domain.j_end;
    d.k_begin                  = domain.k_begin;
    d.k_end                    = domain.k_end;
    d.dt                       = dt;
    if (!kernels->dispatch_thompson_freezing(d))
        throw std::runtime_error("native Thompson freezing failed");
    return true;
}
} // namespace tmv
