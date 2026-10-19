#include "none.hpp"

namespace tmv::radiation
{

void NoneScheme::compute_column(const RadiationConfig& /*cfg*/,
                                const RadiationColumnStateView& col,
                                RadiationColumnTendencies& tend,
                                RadiationColumnFluxes* fluxes_opt)
{
    const std::size_t nz = (col.p != nullptr) ? col.p->size() : 0;

    tend.dTdt_rad.assign(nz, 0.0);
    tend.dTdt_lw.assign(nz, 0.0);
    tend.dTdt_sw.assign(nz, 0.0);
    tend.atmospheric_power_lw_wm2 = 0.0;
    tend.atmospheric_power_sw_wm2 = 0.0;
    tend.atmospheric_power_total_wm2 = 0.0;

    if (fluxes_opt != nullptr)
    {
        fluxes_opt->Fup_lw.assign(nz + 1, 0.0);
        fluxes_opt->Fdn_lw.assign(nz + 1, 0.0);
        fluxes_opt->Fup_sw.assign(nz + 1, 0.0);
        fluxes_opt->Fdn_sw.assign(nz + 1, 0.0);
        fluxes_opt->Fnet_lw.assign(nz + 1, 0.0);
        fluxes_opt->Fnet_sw.assign(nz + 1, 0.0);
        fluxes_opt->Fnet_total.assign(nz + 1, 0.0);
    }
}

}  // namespace tmv::radiation
