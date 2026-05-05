/**
 * @file diagnostics.cpp
 * @brief CAPE/CIN/LCL/LFC/EL computation via a surface-parcel lift over
 *        the final Sounding column.
 *
 * Uses Bolton (1980) for the LCL temperature and a stepwise moist
 * adiabatic lapse rate integration from LCL upward. Approximate but
 * faithful to the SHARPpy reference within a few percent for typical
 * supercell soundings — the right level of accuracy for an initialization
 * diagnostic the user reads at startup.
 */

#include "init/sounding/diagnostics.hpp"

#include "core/physical_constants.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace tmv::init
{

namespace
{

constexpr double R_v = 461.5;                       // J/(kg·K), water-vapor gas constant
constexpr double L_v = physical_constants::latent_heat_vaporization_jkg;

/// Saturation vapor pressure (Pa) over water — same Magnus form used elsewhere.
double e_sat_pa(double T_k)
{
    const double T_c = T_k - physical_constants::freezing_temperature_k;
    if (T_k >= physical_constants::freezing_temperature_k)
    {
        return 611.21 * std::exp((18.678 - T_c / 234.5) * T_c / (257.14 + T_c));
    }
    return 611.15 * std::exp((23.036 - T_c / 333.7) * T_c / (279.82 + T_c));
}

double q_sat(double T_k, double p_pa)
{
    const double e = e_sat_pa(T_k);
    return 0.622 * e / std::max(p_pa - e, 1.0);
}

/// Inverse Magnus: dewpoint from a given vapor pressure (Pa).
double T_d_from_e(double e_pa)
{
    if (e_pa <= 0.0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // Use Magnus over water (Bolton 1980 form).
    const double a = 17.625;
    const double b = 243.04;
    const double ln_e = std::log(e_pa / 611.21);
    const double T_c = (b * ln_e) / (a - ln_e);
    return T_c + physical_constants::freezing_temperature_k;
}

double virtual_temperature(double T_k, double qv_kgkg)
{
    return T_k * (1.0 + 0.608 * qv_kgkg);
}

/// Bolton (1980) eq. 22: temperature at the lifting condensation level.
double T_lcl_bolton(double T_sfc_k, double T_d_k)
{
    if (T_d_k >= T_sfc_k)
    {
        return T_sfc_k;
    }
    const double inv = 1.0 / (T_d_k - 56.0)
                     + std::log(T_sfc_k / T_d_k) / 800.0;
    return 1.0 / inv + 56.0;
}

/// Moist adiabatic lapse rate (K/m) at the saturated parcel state (T, p).
/// Standard form: Γ_m = g · (1 + L_v q_s / (R_d T)) / (c_p + L_v² q_s / (R_v T²)).
double moist_adiabatic_lapse(double T_k, double p_pa)
{
    const double q_s = q_sat(T_k, p_pa);
    const double num = g * (1.0 + L_v * q_s / (R_d * T_k));
    const double den = cp + (L_v * L_v) * q_s / (R_v * T_k * T_k);
    return num / den;
}

double linear_interp(const std::vector<double>& z, const std::vector<double>& y,
                     double z_target)
{
    const std::size_t n = z.size();
    if (n == 0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (z_target <= z.front())
    {
        return y.front();
    }
    if (z_target >= z.back())
    {
        return y.back();
    }
    for (std::size_t k = 1; k < n; ++k)
    {
        if (z_target <= z[k])
        {
            const double f = (z_target - z[k - 1]) / (z[k] - z[k - 1]);
            return y[k - 1] + f * (y[k] - y[k - 1]);
        }
    }
    return y.back();
}

}  // namespace

SoundingDiagnostics compute_sounding_diagnostics(const Sounding& s)
{
    SoundingDiagnostics d;
    const std::size_t nz = s.size();
    if (nz < 2)
    {
        return d;
    }

    // Surface parcel state.
    const double T_sfc = s.T_k[0];
    const double qv_sfc = s.qv_kgkg[0];
    const double p_sfc = s.p_pa[0];
    const double e_sfc = qv_sfc * p_sfc / (0.622 + qv_sfc);
    const double T_d_sfc = T_d_from_e(e_sfc);

    // LCL via Bolton (1980). Convert T_LCL to height z_LCL via dry-adiabatic
    // ascent: z_LCL ≈ (cp / g) * (T_sfc - T_LCL).
    const double T_lcl = T_lcl_bolton(T_sfc, T_d_sfc);
    d.lcl_m = std::max(0.0, (cp / g) * (T_sfc - T_lcl));

    // Lift the surface parcel level by level over the model grid. Below the
    // LCL the parcel is dry; from the LCL onward, integrate the moist
    // adiabatic lapse rate stepwise. We evaluate buoyancy at every level
    // using virtual temperature so moisture loading is included.
    std::vector<double> T_parcel(nz, 0.0);
    std::vector<double> qv_parcel(nz, 0.0);
    std::vector<double> buoyancy(nz, 0.0);

    T_parcel[0] = T_sfc;
    qv_parcel[0] = qv_sfc;
    buoyancy[0] = 0.0;

    bool reached_lcl = (s.z_m[0] >= d.lcl_m);
    double T_running = T_sfc;
    double qv_running = qv_sfc;

    for (std::size_t k = 1; k < nz; ++k)
    {
        const double dz = s.z_m[k] - s.z_m[k - 1];
        if (!reached_lcl && s.z_m[k] >= d.lcl_m)
        {
            // Cross the LCL inside this layer. Step part-way dry, part-way
            // moist for second-order accuracy of the LCL transition.
            const double dz_dry = d.lcl_m - s.z_m[k - 1];
            const double dz_moist = s.z_m[k] - d.lcl_m;
            T_running = T_running - (g / cp) * dz_dry;
            const double Gm = moist_adiabatic_lapse(T_running, s.p_pa[k]);
            T_running = T_running - Gm * dz_moist;
            qv_running = q_sat(T_running, s.p_pa[k]);
            reached_lcl = true;
        }
        else if (!reached_lcl)
        {
            // Dry adiabatic ascent.
            T_running = T_running - (g / cp) * dz;
            // qv conserved while subsaturated.
        }
        else
        {
            // Moist adiabatic ascent: lapse evaluated at the lower-level
            // parcel state, which is good enough at the model dz.
            const double Gm = moist_adiabatic_lapse(T_running, s.p_pa[k]);
            T_running = T_running - Gm * dz;
            qv_running = q_sat(T_running, s.p_pa[k]);
        }

        T_parcel[k] = T_running;
        qv_parcel[k] = reached_lcl ? qv_running : qv_sfc;

        const double T_v_p = virtual_temperature(T_parcel[k], qv_parcel[k]);
        const double T_v_e = virtual_temperature(s.T_k[k], s.qv_kgkg[k]);
        buoyancy[k] = g * (T_v_p - T_v_e) / std::max(T_v_e, 1.0);
    }

    // Find LFC = first level where buoyancy turns positive (above the LCL).
    // EL = first level above LFC where buoyancy turns negative again.
    std::size_t k_lfc = nz;
    for (std::size_t k = 1; k < nz; ++k)
    {
        if (s.z_m[k] < d.lcl_m)
        {
            continue;
        }
        if (buoyancy[k] > 0.0)
        {
            k_lfc = k;
            break;
        }
    }
    if (k_lfc < nz)
    {
        d.lfc_m = s.z_m[k_lfc];
        std::size_t k_el = nz;
        for (std::size_t k = k_lfc + 1; k < nz; ++k)
        {
            if (buoyancy[k] <= 0.0)
            {
                k_el = k;
                break;
            }
        }
        if (k_el < nz)
        {
            d.el_m = s.z_m[k_el];
        }
        else
        {
            d.el_m = s.z_m.back();
        }

        // CAPE: integrate positive buoyancy between LFC and EL.
        for (std::size_t k = k_lfc; k + 1 < nz && s.z_m[k + 1] <= d.el_m; ++k)
        {
            const double dz = s.z_m[k + 1] - s.z_m[k];
            const double b_avg = 0.5 * (buoyancy[k] + buoyancy[k + 1]);
            if (b_avg > 0.0)
            {
                d.cape_jkg += b_avg * dz;
            }
        }
    }
    else
    {
        d.lfc_m = std::numeric_limits<double>::quiet_NaN();
        d.el_m = std::numeric_limits<double>::quiet_NaN();
    }

    // CIN: negative buoyancy from surface up to LFC. When no LFC is found
    // the air column is convectively suppressed and the CIN concept loses
    // its standard meaning (it would otherwise integrate to model top
    // and produce an arbitrarily large number that misleads users).
    // Report 0 in that case and rely on the "no LFC" warning to convey
    // the situation.
    if (std::isfinite(d.lfc_m))
    {
        for (std::size_t k = 0; k + 1 < nz; ++k)
        {
            if (s.z_m[k + 1] > d.lfc_m)
            {
                break;
            }
            const double dz = s.z_m[k + 1] - s.z_m[k];
            const double b_avg = 0.5 * (buoyancy[k] + buoyancy[k + 1]);
            if (b_avg < 0.0)
            {
                d.cin_jkg += -b_avg * dz;
            }
        }
    }

    // Precipitable water: ∫ ρ_v dz ≈ ∫ ρ_air q_v dz, in mm of liquid water.
    // 1 kg/m^2 of water column equals 1 mm of equivalent depth.
    for (std::size_t k = 0; k + 1 < nz; ++k)
    {
        const double dz = s.z_m[k + 1] - s.z_m[k];
        const double rho_qv_avg = 0.5 * (s.rho_kgm3[k] * s.qv_kgkg[k]
                                       + s.rho_kgm3[k + 1] * s.qv_kgkg[k + 1]);
        d.pwat_mm += rho_qv_avg * dz;
    }

    return d;
}

SoundingDiagnostics compute_sounding_diagnostics(const Sounding& s,
                                                 const WindColumn& winds)
{
    SoundingDiagnostics d = compute_sounding_diagnostics(s);
    if (winds.is_consistent_with(s.z_m))
    {
        const double u_sfc = winds.u_ms.front();
        const double v_sfc = winds.v_ms.front();
        const double u_6km = linear_interp(s.z_m, winds.u_ms, 6000.0);
        const double v_6km = linear_interp(s.z_m, winds.v_ms, 6000.0);
        const double du = u_6km - u_sfc;
        const double dv = v_6km - v_sfc;
        d.bulk_shear_0_6km_ms = std::sqrt(du * du + dv * dv);
        d.has_kinematic = true;
    }
    return d;
}

}  // namespace tmv::init
