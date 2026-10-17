/**
 * @file thompson.hpp
 * @brief Declarations for the microphysics module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the microphysics runtime and scheme implementations.
 * This file is part of the src/microphysics subsystem.
 */

#pragma once
#include "microphysics/base/thermodynamics.hpp"
#include "microphysics/microphysics_base.hpp"
#include "core/field/field3d.hpp"
#include "microphysics/schemes/thompson/prognostic_state.hpp"
#include "microphysics/schemes/thompson/activation_table.hpp"
#include "microphysics/schemes/thompson/cloud_evaporation.hpp"
#include "microphysics/schemes/thompson/collection_tables.hpp"
#include "microphysics/schemes/thompson/cold_process_tables.hpp"
#include "compute/thompson_cloud_tables.hpp"

#include <memory>

namespace tmv
{

/**
 * @brief Thompson bulk microphysics parameterization with prognostic ice number.
 */
class ThompsonScheme : public MicrophysicsScheme 
{
private:
    double qc0_;
    double c_auto_;
    double c_evap_;

    double ccn_conc_;
    double in_conc_;
    thompson::PrognosticState prognostic_state_;
    std::shared_ptr<const thompson::ActivationTable> activation_table_;
    std::shared_ptr<const thompson::CloudEvaporationTable>
        cloud_evaporation_table_;
    std::shared_ptr<const thompson::CollectionTables> collection_tables_;
    std::shared_ptr<const thompson::ColdProcessTables> cold_process_tables_;
    Field3D surface_sedimentation_sink_rate_;
    std::shared_ptr<const nimbus::compute::ThompsonCloudTables> native_cloud_tables_;

    struct NativeCloudAdjustment
    {
        Field3D vapor_change;
        Field3D vapor_change_low;
        Field3D temperature_change;
        Field3D temperature_change_low;
        Field3D cloud_number_tendency;
        Field3D aerosol_tendency;
        Field3D lookup_index;
        Field3D temperature;
    };

    void ensure_activation_table();
    bool try_native_completion(const Field3D&                       p,
                               const Field3D&                       rho,
                               const std::array<const Field3D*, 7>& water,
                               double                               dt,
                               const NativeCloudAdjustment&         cloud,
                               const std::array<Field3D, 10>&       warm,
                               const std::array<Field3D, 13>&       deposition,
                               const std::array<Field3D, 13>&       freezing,
                               thompson::PrognosticState&           state,
                               const std::array<Field3D*, 8>&       tendency,
                               Field3D&                             snow_speed);

    bool try_native_deposition(const Field3D&                 p,
                               const Field3D&                 rho,
                               const Field3D&                 vapor,
                               const Field3D&                 ice,
                               const Field3D&                 snow,
                               const Field3D&                 graupel,
                               double                         dt,
                               const NativeCloudAdjustment&   cloud_result,
                               const std::array<Field3D, 10>& warm_result,
                               std::array<Field3D, 13>&       result);

    bool try_native_freezing(const Field3D&                 rho,
                             const Field3D&                 cloud,
                             const Field3D&                 rain,
                             double                         dt,
                             const NativeCloudAdjustment&   cloud_result,
                             const std::array<Field3D, 10>& warm_result,
                             std::array<Field3D, 13>&       result);

    bool try_native_warm_rain(const Field3D&               p,
                              const Field3D&               rho,
                              const Field3D&               vapor,
                              const Field3D&               cloud,
                              const Field3D&               rain,
                              double                       dt,
                              const NativeCloudAdjustment& cloud_result,
                              std::array<Field3D, 10>&     result);

    bool try_native_cloud_adjustment(const Field3D&         p,
                                     const Field3D&         theta,
                                     const Field3D&         rho,
                                     const Field3D&         vapor,
                                     const Field3D&         cloud,
                                     double                 dt,
                                     NativeCloudAdjustment& result);

public:
    std::unique_ptr<MicrophysicsScheme> clone() const override
    {
        return std::make_unique<ThompsonScheme>(*this);
    }
    std::unique_ptr<MicrophysicsScheme> clone_parameters() const override
    {
        auto result               = std::make_unique<ThompsonScheme>(qc0_, c_auto_, c_evap_, ccn_conc_, in_conc_);
        result->activation_table_ = activation_table_;
        result->cloud_evaporation_table_ = cloud_evaporation_table_;
        result->collection_tables_       = collection_tables_;
        result->cold_process_tables_     = cold_process_tables_;
        result->native_cloud_tables_     = native_cloud_tables_;
        return result;
    }

/**
 * @brief Initializes the Thompson scheme with default parameters.
 */
    ThompsonScheme(
        double qc0 = 1.0e-3,
        double c_auto = 1.0e-3,
        double c_evap = 1.0e-3,
        double ccn_conc = 350e6,
        double in_conc = 2.0e6
    );

    void configure(const MicrophysicsConfig& config) override;
    PortableSchemeState portable_state() const override;
    PortableSchemeState portable_parameter_state() const override;
    MicrophysicsParameterIdentity parameter_identity() const noexcept override;
    bool restore_portable_state(const PortableSchemeState& state,
                                std::string& error) override;
    void                             prepare_prognostic_state(const Field3D& rho_d,
                                                              const Field3D& qc,
                                                              const Field3D& qr,
                                                              const Field3D& qi,
                                                              const Field3D& qs,
                                                              const Field3D& qg,
                                                              const Field3D& qh) override;
    MicrophysicsPrognosticScalarList prognostic_scalars_for_transport() override;
    bool                             validate_prognostic_update(double dt, std::string& error) const override;
    bool                             apply_stochastic_column_multipliers(double              dt,
                                                                         const Field3D&      density,
                                                                         std::vector<float>& column_multipliers,
                                                                         std::string&        error) override;
    void                             commit_prognostic_update(double dt) noexcept override;
    void                             invalidate_transient_state_after_domain_recycling() noexcept override;

    /**
     * @brief Computes the tendencies for the Thompson scheme.
     */
    void compute_tendencies(
        const Field3D& p,
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
        Field3D& dqh_dt
    ) override;

/**
 * @brief Computes the radar reflectivity for the Thompson scheme.
 */
    double bottom_sedimentation_sink_rate(
        const Field3D& rho_d,
        const Field3D& qr, const Field3D& qs,
        const Field3D& qg, const Field3D& qh,
        int i, int j, double dt_micro) const override;

    std::string get_scheme_name() const override { return "thompson"; }
    int get_num_prognostic_vars() const override { return 11; }

private:
    void complete_cloud_phase(const Field3D&             p,
                              const Field3D&             theta,
                              const Field3D&             rho,
                              const Field3D&             vapor,
                              const Field3D&             cloud,
                              const Field3D&             ice,
                              const Field3D&             vapor_tendency,
                              Field3D&                   cloud_tendency,
                              Field3D&                   ice_tendency,
                              Field3D&                   temperature_tendency,
                              thompson::PrognosticState* state,
                              double                     dt);

    bool try_native_sedimentation(
        const Field3D& p, const Field3D& theta, const Field3D& rho_d,
        const std::array<const Field3D*, 5>& mass,
        const std::array<Field3D*, 5>& tendency,
        const Field3D& snow_fall_speed_multiplier,
        thompson::PrognosticState& state,
        Field3D& surface_sink_candidate,
        double dt);

    /**
 * @brief Computes the sedimentation for the Thompson scheme.
 */
    void compute_sedimentation(
        const Field3D& p,
        const Field3D& theta,
        const Field3D& rho_d,
        const Field3D& qc,
        const Field3D& qr,
        const Field3D& qi,
        const Field3D& qs,
        const Field3D& qg,
        Field3D& dqc_dt,
        Field3D& dqr_dt,
        Field3D& dqi_dt,
        Field3D& dqs_dt,
        Field3D& dqg_dt,
        const Field3D& snow_fall_speed_multiplier,
        thompson::PrognosticState* state,
        Field3D& surface_sink_candidate,
        double dt_micro
    );

    /**
     * @brief Computes collection efficiency for mixed-phase interactions.
     */
    double collection_efficiency(double T_celsius);
};

} // namespace tmv

using tmv::ThompsonScheme;
