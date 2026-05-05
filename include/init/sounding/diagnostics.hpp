#pragma once

#include "init/hodograph/hodograph_source.hpp"
#include "init/sounding/sounding_source.hpp"

namespace tmv::init
{

/**
 * @brief Convective and kinematic diagnostics computed once over a final
 *        Sounding column.
 *
 * Closes the loop on "what did I actually load?": a user supplying
 * parametric_targets sees whether CAPE/CIN/LCL came out at the requested
 * targets, a user loading a real-world sounding sees the headline
 * convective parameters before the run begins.
 *
 * Values not computable from the inputs are NaN. LFC and EL are NaN when
 * the surface parcel never becomes positively buoyant (no convection
 * possible — exactly the "nothing happens" case from the architecture
 * discussion).
 */
struct SoundingDiagnostics
{
    double cape_jkg = 0.0;
    double cin_jkg = 0.0;
    double lcl_m = 0.0;
    double lfc_m = 0.0;     // NaN if no LFC (capped column)
    double el_m = 0.0;      // NaN if no EL
    double pwat_mm = 0.0;   // precipitable water column

    // Kinematic — populated only when a non-empty WindColumn is supplied.
    double bulk_shear_0_6km_ms = 0.0;  // |V(6 km) - V(sfc)|

    bool has_kinematic = false;
};

/**
 * @brief Runs a surface-parcel lift over the column and returns CAPE, CIN,
 *        LCL/LFC/EL, and precipitable water.
 *
 *  - LCL via Bolton (1980): T_LCL = 1/(1/(T_d - 56) + ln(T_sfc/T_d)/800) + 56
 *  - Below LCL: dry adiabatic ascent (lapse = g/cp).
 *  - Above LCL: stepwise integration of the moist adiabatic lapse rate
 *    Γ_m = g (1 + L_v q_s / (R_d T)) / (c_p + L_v^2 q_s / (R_v T^2))
 *    from the model grid level to level. Approximate but matches SHARPpy's
 *    parcel lift to within a few percent for typical sounding profiles.
 *  - Buoyancy uses virtual temperature (B = g (T_v_p - T_v_e) / T_v_e).
 *  - CAPE = sum of positive buoyancy times dz between LFC and EL.
 *  - CIN  = sum of negative buoyancy times dz from sfc to LFC.
 *
 * If the parcel never becomes positively buoyant, lfc_m / el_m are set to
 * NaN and CAPE = 0. CIN may still be > 0 (the cap that prevented the
 * parcel from ever lifting freely).
 */
SoundingDiagnostics compute_sounding_diagnostics(const Sounding& s);

/**
 * @brief Same as compute_sounding_diagnostics(s) but additionally fills the
 *        kinematic fields from a Cartesian wind column. Bulk shear 0-6 km
 *        is the magnitude of the wind difference between the cell closest
 *        to z = 6000 m and the surface cell.
 */
SoundingDiagnostics compute_sounding_diagnostics(const Sounding& s,
                                                 const WindColumn& winds);

}  // namespace tmv::init
