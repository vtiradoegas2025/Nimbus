/**
 * @file sounding_source.cpp
 * @brief Definitions for SoundingSource utilities. Currently hosts only
 *        Sounding::verify_self_consistent(); future cross-source helpers
 *        (e.g. CAPE/CIN/SRH diagnostics computed once over the final
 *        column) will land here too.
 */

#include "init/sounding/sounding_source.hpp"
#include "core/physical_constants.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace tmv::init
{

namespace
{

void check(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::logic_error("Sounding::verify_self_consistent: " + message);
    }
}

}  // namespace

void Sounding::verify_self_consistent() const
{
    const std::size_t nz = z_m.size();
    check(nz > 0, "z_m is empty");
    check(T_k.size() == nz, "T_k length mismatch");
    check(theta_k.size() == nz, "theta_k length mismatch");
    check(qv_kgkg.size() == nz, "qv_kgkg length mismatch");
    check(p_pa.size() == nz, "p_pa length mismatch");
    check(rho_kgm3.size() == nz, "rho_kgm3 length mismatch");

    const double kappa = R_d / cp;
    for (std::size_t k = 0; k < nz; ++k)
    {
        check(std::isfinite(T_k[k]) && T_k[k] > 0.0,
              "T_k[" + std::to_string(k) + "] is non-positive or non-finite");
        check(std::isfinite(p_pa[k]) && p_pa[k] > 0.0,
              "p_pa[" + std::to_string(k) + "] is non-positive or non-finite");
        check(std::isfinite(rho_kgm3[k]) && rho_kgm3[k] > 0.0,
              "rho_kgm3[" + std::to_string(k) + "] is non-positive or non-finite");
        check(qv_kgkg[k] >= 0.0,
              "qv_kgkg[" + std::to_string(k) + "] is negative");

        // theta = T * (p0/p)^kappa, allow 0.5% slack for the round-trip.
        const double theta_expected = T_k[k] * std::pow(p0 / p_pa[k], kappa);
        const double theta_rel_err = std::abs(theta_k[k] - theta_expected)
                                     / std::max(theta_expected, 1.0);
        check(theta_rel_err < 5.0e-3,
              "theta_k[" + std::to_string(k) + "] inconsistent with T,p");

        // rho = p / (R_d T), allow 0.5% slack (rho_floor pushes some entries off
        // exact equality at the highest model levels — that is intentional).
        const double rho_expected = p_pa[k] / (R_d * T_k[k]);
        const double rho_rel_err = std::abs(rho_kgm3[k] - rho_expected)
                                   / std::max(rho_expected, 0.01);
        const bool rho_floor_active = rho_kgm3[k] > rho_expected;
        check(rho_rel_err < 5.0e-3 || rho_floor_active,
              "rho_kgm3[" + std::to_string(k) + "] inconsistent with p,T");
    }
}

}  // namespace tmv::init
