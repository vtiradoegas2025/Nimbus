#include "none.hpp"

namespace tmv::boundary_layer
{

void NoneScheme::compute_column(const BoundaryLayerConfig& /*cfg*/,
                                const SurfaceConfig& /*sfc*/,
                                const BoundaryLayerColumnStateView& col,
                                BoundaryLayerTendencies& tend,
                                BoundaryLayerDiagnostics* diag_opt)
{
    const std::size_t nz = (col.theta != nullptr) ? col.theta->size() : 0;

    tend.dudt_pbl.assign(nz, 0.0);
    tend.dvdt_pbl.assign(nz, 0.0);
    tend.dthetadt_pbl.assign(nz, 0.0);
    tend.dqvdt_pbl.assign(nz, 0.0);
    tend.dqcdt_pbl.assign(nz, 0.0);
    tend.dqidt_pbl.assign(nz, 0.0);
    tend.dtkedt_pbl.assign(nz, 0.0);

    if (diag_opt != nullptr)
    {
        if (diag_opt->include_diffusivity_profiles)
        {
            diag_opt->K_m.assign(nz, 0.0);
            diag_opt->K_h.assign(nz, 0.0);
            diag_opt->K_e.assign(nz, 0.0);
        }
        diag_opt->pbl_height = 0.0;
        diag_opt->ustar = 0.0;
        diag_opt->tau_u = 0.0;
        diag_opt->tau_v = 0.0;
        diag_opt->H = 0.0;
        diag_opt->E = 0.0;
        diag_opt->L = 0.0;
        diag_opt->rib_bulk = 0.0;
        diag_opt->Cd = 0.0;
        diag_opt->Ch = 0.0;
        diag_opt->Ce = 0.0;
    }
}

}  // namespace tmv::boundary_layer
