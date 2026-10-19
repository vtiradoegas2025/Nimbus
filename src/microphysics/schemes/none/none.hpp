#pragma once

#include "microphysics/microphysics_base.hpp"

namespace tmv::microphysics {

/**
 * @brief No-op microphysics scheme.
 *
 * Returns zero tendencies for every prognostic variable. Used by the
 * hydrostatic-equilibrium integration test where microphysics must be
 * fully out of the loop so the test exercises only dynamics. Also useful
 * for dry-dynamics smoke runs and CFD-style benchmarks.
 *
 * Wrapped in `namespace microphysics` to avoid colliding with
 * `::NoneScheme` in the terrain module.
 */
class NoneScheme : public MicrophysicsScheme
{
public:
    std::unique_ptr<MicrophysicsScheme> clone() const override
    {
        return std::make_unique<NoneScheme>(*this);
    }

    void compute_tendencies(const Field3D& p,
                            const Field3D& theta,
                            const Field3D& rho_d,
                            const Field3D& qv,
                            const Field3D& qc,
                            const Field3D& qr,
                            const Field3D& qi,
                            const Field3D& qs,
                            const Field3D& qg,
                            const Field3D& qh,
                            double dt,
                            Field3D& dtheta_dt,
                            Field3D& dqv_dt,
                            Field3D& dqc_dt,
                            Field3D& dqr_dt,
                            Field3D& dqi_dt,
                            Field3D& dqs_dt,
                            Field3D& dqg_dt,
                            Field3D& dqh_dt) override;

    std::string get_scheme_name() const override { return "none"; }

    int get_num_prognostic_vars() const override { return 0; }
};

}  // namespace tmv::microphysics

#ifndef NIMBUS_MICROPHYSICS_NAMESPACE_ALIAS
#define NIMBUS_MICROPHYSICS_NAMESPACE_ALIAS
namespace microphysics = tmv::microphysics;
#endif
