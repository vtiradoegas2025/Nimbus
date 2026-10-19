#include "microphysics/schemes/thompson/thompson.hpp"
#include "microphysics/schemes/thompson/native_deposition_tables.hpp"
#include "compute/backend_runtime.hpp"
#include <stdexcept>

namespace tmv
{
bool ThompsonScheme::try_native_deposition(const Field3D&                 p,
                                           const Field3D&                 rho,
                                           const Field3D&                 vapor,
                                           const Field3D&                 ice,
                                           const Field3D&                 snow,
                                           const Field3D&                 graupel,
                                           double                         dt,
                                           const NativeCloudAdjustment&   cloud_result,
                                           const std::array<Field3D, 10>& warm_result,
                                           std::array<Field3D, 13>&       result)
{
    auto* runtime = compute_runtime();
    auto* kernels = runtime && runtime->backend ? runtime->backend->microphysics_kernels() : nullptr;
    if (!kernels || !kernels->supports(MicrophysicsKernelOp::ThompsonDeposition))
        return false;
    const int  nx = p.size_r(), ny = p.size_th(), nz = p.size_z();
    const auto domain = physical_cell_domain(nx, ny, nz);
    for (auto& f : result)
        f = Field3D(nx, ny, nz, 0);
    ThompsonDepositionDispatch d;
    d.temperature             = cloud_result.temperature.data();
    d.pressure                = p.data();
    d.density                 = rho.data();
    d.vapor                   = vapor.data();
    d.ice                     = ice.data();
    d.snow                    = snow.data();
    d.graupel                 = graupel.data();
    d.ice_number              = prognostic_state_.ice_number().data();
    d.cloud_vapor_change      = cloud_result.vapor_change.data();
    d.cloud_vapor_change_low  = cloud_result.vapor_change_low.data();
    d.evaporation_high        = warm_result[2].data();
    d.evaporation_low         = warm_result[3].data();
    d.tables                  = thompson::native_deposition_tables();
    d.ice_change_high         = result[0].data();
    d.ice_change_low          = result[1].data();
    d.snow_change_high        = result[2].data();
    d.snow_change_low         = result[3].data();
    d.graupel_change_high     = result[4].data();
    d.graupel_change_low      = result[5].data();
    d.temperature_change_high = result[6].data();
    d.temperature_change_low  = result[7].data();
    d.ice_number_tendency     = result[8].data();
    d.snow_vapor_rate_high    = result[9].data();
    d.snow_vapor_rate_low     = result[10].data();
    d.lookup_index            = result[11].data();
    d.status                  = result[12].data();
    d.nx                      = nx;
    d.ny                      = ny;
    d.nz                      = nz;
    d.i_begin                 = domain.i_begin;
    d.i_end                   = domain.i_end;
    d.j_begin                 = domain.j_begin;
    d.j_end                   = domain.j_end;
    d.k_begin                 = domain.k_begin;
    d.k_end                   = domain.k_end;
    d.dt                      = dt;
    if (!kernels->dispatch_thompson_deposition(d))
        throw std::runtime_error("native Thompson deposition failed");
    return true;
}
} // namespace tmv
