#pragma once

#include <memory>
#include <string>

#include "core/field3d.hpp"
#include "core/physical_constants.hpp"

/**
 * @file microphysics_base.hpp
 * @brief Abstract interface for microphysics parameterization schemes.
 *
 * Declares the core tendency and diagnostic hooks used by runtime
 * microphysics modules.
 * Also centralizes shared constants and factory construction.
 */

class MicrophysicsScheme
{
public:
    /**
     * @brief Virtual destructor for polymorphic cleanup.
     */
    virtual ~MicrophysicsScheme() = default;

    /**
     * @brief Computes thermodynamic and hydrometeor tendencies.
     * @param p Pressure field.
     * @param theta Potential temperature field.
     * @param qv Water vapor mixing ratio field.
     * @param qc Cloud water mixing ratio field.
     * @param qr Rain mixing ratio field.
     * @param qi Cloud ice mixing ratio field.
     * @param qs Snow mixing ratio field.
     * @param qg Graupel mixing ratio field.
     * @param qh Hail mixing ratio field.
     * @param dt Time step in seconds.
     * @param dtheta_dt Potential temperature tendency.
     * @param dqv_dt Water vapor tendency.
     * @param dqc_dt Cloud water tendency.
     * @param dqr_dt Rain tendency.
     * @param dqi_dt Cloud ice tendency.
     * @param dqs_dt Snow tendency.
     * @param dqg_dt Graupel tendency.
     * @param dqh_dt Hail tendency.
     */
    virtual void compute_tendencies(const Field3D& p,
                                    const Field3D& theta,
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
                                    Field3D& dqh_dt) = 0;

    /**
     * @brief Computes radar reflectivity diagnostics from hydrometeors.
     * @param qc Cloud water mixing ratio field.
     * @param qr Rain mixing ratio field.
     * @param qi Cloud ice mixing ratio field.
     * @param qs Snow mixing ratio field.
     * @param qg Graupel mixing ratio field.
     * @param qh Hail mixing ratio field.
     * @param reflectivity_dbz Output reflectivity in dBZ.
     */
    virtual void compute_radar_reflectivity(const Field3D& qc,
                                            const Field3D& qr,
                                            const Field3D& qi,
                                            const Field3D& qs,
                                            const Field3D& qg,
                                            const Field3D& qh,
                                            Field3D& reflectivity_dbz)
    {
    }

    /**
     * @brief Computes surface precipitation-rate diagnostics.
     * @param qr Rain mixing ratio field.
     * @param qs Snow mixing ratio field.
     * @param qg Graupel mixing ratio field.
     * @param qh Hail mixing ratio field.
     * @param precip_rate_rain Output rain rate.
     * @param precip_rate_snow Output snow rate.
     * @param precip_rate_grau Output graupel rate.
     * @param precip_rate_hail Output hail rate.
     */
    virtual void compute_precipitation_rates(const Field3D& qr,
                                             const Field3D& qs,
                                             const Field3D& qg,
                                             const Field3D& qh,
                                             Field3D& precip_rate_rain,
                                             Field3D& precip_rate_snow,
                                             Field3D& precip_rate_grau,
                                             Field3D& precip_rate_hail)
    {
    }

    /**
     * @brief Returns the scheme identifier.
     * @return Scheme name.
     */
    virtual std::string get_scheme_name() const = 0;

    /**
     * @brief Returns the number of prognostic microphysics variables.
     * @return Count of prognostic microphysics variables.
     */
    virtual int get_num_prognostic_vars() const = 0;

protected:
    /**
     * @brief Resizes all 8 tendency output fields to (NR, NTH, NZ) and zeros them.
     *
     * Call at the top of compute_tendencies() to eliminate the identical
     * 8-line resize block duplicated across every scheme implementation.
     */
    static void init_tendency_fields(int NR, int NTH, int NZ,
                                     Field3D& dtheta_dt,
                                     Field3D& dqv_dt,
                                     Field3D& dqc_dt,
                                     Field3D& dqr_dt,
                                     Field3D& dqi_dt,
                                     Field3D& dqs_dt,
                                     Field3D& dqg_dt,
                                     Field3D& dqh_dt)
    {
        dtheta_dt.resize(NR, NTH, NZ, 0.0f);
        dqv_dt.resize(NR, NTH, NZ, 0.0f);
        dqc_dt.resize(NR, NTH, NZ, 0.0f);
        dqr_dt.resize(NR, NTH, NZ, 0.0f);
        dqi_dt.resize(NR, NTH, NZ, 0.0f);
        dqs_dt.resize(NR, NTH, NZ, 0.0f);
        dqg_dt.resize(NR, NTH, NZ, 0.0f);
        dqh_dt.resize(NR, NTH, NZ, 0.0f);
    }
};

/**
 * @brief Creates a microphysics scheme by name.
 * @param scheme_name Scheme identifier.
 * @return Owning pointer to a scheme instance.
 */
std::unique_ptr<MicrophysicsScheme> create_microphysics_scheme(const std::string& scheme_name);

namespace microphysics_constants
{
constexpr double T0 = physical_constants::freezing_temperature_k;
constexpr double L_v = physical_constants::latent_heat_vaporization_jkg;
constexpr double L_f = physical_constants::latent_heat_fusion_jkg;
constexpr double L_s = L_v + L_f;
constexpr double cp = physical_constants::specific_heat_cp_jkgk;
constexpr double R_v = 461.5;
constexpr double R_d = physical_constants::gas_constant_dry_air_jkgk;
constexpr double p0 = physical_constants::reference_pressure_pa;
constexpr double rho_w = 1000.0;
constexpr double rho_i = 917.0;
} // namespace microphysics_constants
