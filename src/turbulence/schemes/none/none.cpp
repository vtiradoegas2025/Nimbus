#include "none.hpp"

namespace tmv::turbulence {

namespace
{

void resize_and_zero(Field3D& f, int NR, int NTH, int NZ)
{
    if (f.size_r() != NR || f.size_th() != NTH || f.size_z() != NZ)
    {
        f.resize(NR, NTH, NZ, 0.0f);
    }
    else
    {
        f.fill(0.0f);
    }
}

}  // anonymous namespace

void NoneScheme::compute(const TurbulenceConfig& /*cfg*/,
                         const GridMetrics& /*grid*/,
                         const TurbulenceStateView& state,
                         TurbulenceTendencies& tend,
                         TurbulenceDiagnostics* diag_opt)
{
    tend.resident_source_unchanged = false;
    tend.mapped_boundary_power_w =
        std::numeric_limits<double>::quiet_NaN();
    const int NR = state.NR;
    const int NTH = state.NTH;
    const int NZ = state.NZ;

    if (NR <= 0 || NTH <= 0 || NZ <= 0) return;

    resize_and_zero(tend.dudt_sgs, NR, NTH, NZ);
    resize_and_zero(tend.dvdt_sgs, NR, NTH, NZ);
    resize_and_zero(tend.dwdt_sgs, NR, NTH, NZ);
    resize_and_zero(tend.dthetadt_sgs, NR, NTH, NZ);
    resize_and_zero(tend.dqvdt_sgs, NR, NTH, NZ);
    resize_and_zero(tend.dtkedt_sgs, NR, NTH, NZ);

    if (diag_opt != nullptr)
    {
        resize_and_zero(diag_opt->nu_t, NR, NTH, NZ);
        resize_and_zero(diag_opt->K_theta, NR, NTH, NZ);
        resize_and_zero(diag_opt->K_q, NR, NTH, NZ);
        resize_and_zero(diag_opt->K_tke, NR, NTH, NZ);
        resize_and_zero(diag_opt->shear_prod, NR, NTH, NZ);
        resize_and_zero(diag_opt->buoy_prod, NR, NTH, NZ);
        resize_and_zero(diag_opt->dissipation, NR, NTH, NZ);
        resize_and_zero(diag_opt->Cs_eff, NR, NTH, NZ);
        resize_and_zero(diag_opt->Pr_t_eff, NR, NTH, NZ);
    }
}

}  // namespace tmv::turbulence
