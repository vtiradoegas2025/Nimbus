#include "none.hpp"

namespace microphysics 
{

void NoneScheme::compute_tendencies(const Field3D& p,
                                    const Field3D& /*theta*/,
                                    const Field3D& /*qv*/,
                                    const Field3D& /*qc*/,
                                    const Field3D& /*qr*/,
                                    const Field3D& /*qi*/,
                                    const Field3D& /*qs*/,
                                    const Field3D& /*qg*/,
                                    const Field3D& /*qh*/,
                                    double /*dt*/,
                                    Field3D& dtheta_dt,
                                    Field3D& dqv_dt,
                                    Field3D& dqc_dt,
                                    Field3D& dqr_dt,
                                    Field3D& dqi_dt,
                                    Field3D& dqs_dt,
                                    Field3D& dqg_dt,
                                    Field3D& dqh_dt)
{
    const int NR = p.size_r();
    if (NR == 0) return;
    const int NTH = p.size_th();
    if (NTH == 0) return;
    const int NZ = p.size_z();
    if (NZ == 0) return;
    init_tendency_fields(NR, NTH, NZ,
                         dtheta_dt, dqv_dt, dqc_dt, dqr_dt,
                         dqi_dt, dqs_dt, dqg_dt, dqh_dt);
}

}  // namespace microphysics
