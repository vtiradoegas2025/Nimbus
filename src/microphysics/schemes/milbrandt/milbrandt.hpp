// Milbrandt-Yau-style multi-moment bulk microphysics.

#pragma once

#include "sedimentation.hpp"
#include "triple_sedimentation.hpp"
#include "triple_cold_spectrum_domain.hpp"
#include "microphysics/base/thermodynamics.hpp"
#include "microphysics/microphysics_base.hpp"
#include "core/field/field3d.hpp"

#include <array>
#include <optional>
#include <stdexcept>
#include <vector>

struct SimulationStateView;
struct SimulationStorageState;

namespace tmv
{



// Milbrandt-Yau double-moment scheme. Number moments use the operational
// prognostic convention #/kg_dry and are converted to #/m3 only inside local
// collection/distribution integrals using live dry-air density.
class MilbrandtScheme : public MicrophysicsScheme 
{
private:
    double alpha_r_;
    double alpha_i_;
    double alpha_s_;
    double alpha_g_;
    double alpha_h_;

    double c_r_, d_r_;
    double c_i_, d_i_;
    double c_s_, d_s_;
    double c_g_, d_g_;
    double c_h_, d_h_;

    double a_r_, b_r_;
    double a_i_, b_i_;
    double a_s_, b_s_;
    double a_g_, b_g_;
    double a_h_, b_h_;

    bool hail_processes_;

    Field3D Nc_;
    Field3D Nr_;
    Field3D Ni_;
    Field3D Ns_;
    Field3D Ng_;
    Field3D Nh_;

    Field3D dNc_dt_, dNr_dt_, dNi_dt_, dNs_dt_, dNg_dt_, dNh_dt_;

    // Exact total mixing-ratio rate removed through k=0 during the most
    // recent tendency evaluation.  Milbrandt fall speeds depend on the
    // scheme-private number moments, which are advanced after sedimentation,
    // so the runtime conservation gate cannot reconstruct this rate later
    // from the updated moments.
    Field3D surface_sedimentation_sink_rate_;
    enum class InternalTripleMode
    {
        SedimentationOnly,
        WarmOnly,
        WarmRainSedimentation,
        RainFreezingOnly,
        FrozenDiffusionOnly,
        FrozenCompositionOnly,
        MeltingOnly,
        IceMeltingOnly,
        SnowMeltingOnly,
        WarmPrimaryOnly,
        CloudAdjustmentOnly,
        WarmCloudPrimaryOnly,
        DynamicWarmCloudPrimaryOnly,
        SubcycledWarmCloudPrimaryOnly,
        FrozenCollectionPrimaryOnly,
        ColdCloudCollectionPrimaryOnly,
        ColdCloudDiffusionPrimaryOnly,
        ColdRimingPrimaryOnly,
        ColdIceGraupelPrimaryOnly,
        ColdGraupelConversionsPrimaryOnly,
        ColdHomogeneousPrimaryOnly,
        ColdSllPrimaryOnly,
        ColdCollectedSllPrimaryOnly,
        ColdRainSllPrimaryOnly,
        ColdCollisionSllPrimaryOnly,
        ColdWetGrowthSllPrimaryOnly
    };
    static bool is_cold_riming_mode(InternalTripleMode mode) noexcept
    {
        return mode == InternalTripleMode::ColdRimingPrimaryOnly ||
            mode == InternalTripleMode::ColdIceGraupelPrimaryOnly ||
            mode == InternalTripleMode::ColdGraupelConversionsPrimaryOnly ||
            mode == InternalTripleMode::ColdHomogeneousPrimaryOnly ||
            mode == InternalTripleMode::ColdSllPrimaryOnly ||
            mode == InternalTripleMode::ColdCollectedSllPrimaryOnly ||
            mode == InternalTripleMode::ColdRainSllPrimaryOnly ||
            mode == InternalTripleMode::ColdCollisionSllPrimaryOnly ||
            mode == InternalTripleMode::ColdWetGrowthSllPrimaryOnly;
    }
    static milbrandt_detail::ColdSpectrumDomain internal_triple_spectrum_domain(InternalTripleMode mode) noexcept;
    struct InternalTripleCandidateState
    {
        InternalTripleMode mode = InternalTripleMode::SedimentationOnly;
        milbrandt_detail::TripleMomentSpecies selected_frozen_species =
            milbrandt_detail::TripleMomentSpecies::Graupel;
        std::array<Field3D, 5> reflectivity;
        std::array<Field3D, 5> mass_after;
        std::array<Field3D, 5> number_after;
        std::array<Field3D, 5> reflectivity_after;
        Field3D cloud_number_after;
        Field3D vapor_after;
        Field3D cloud_after;
        Field3D theta_after;
        Field3D pressure_before;
        SimulationStorageState* warm_storage_owner = nullptr;
        Field3D density_before;
        std::array<Field3D, 10> primary_source_before;
        std::array<Field3D, 6> number_source_before;
        std::array<Field3D, 5> reflectivity_source_before;
        const Field3D* primary_velocity_source = nullptr;
        Field3D primary_velocity_before;
        MicrophysicsPrimaryEndpointCandidate primary_endpoints;
        std::uint64_t candidate_generation = 0;
        Field3D surface_sink_after;
        std::vector<std::array<milbrandt_detail::TripleMomentSurfaceInventory, 5>>
            surface_after;
        std::vector<std::array<milbrandt_detail::TripleMomentSurfaceInventory, 5>>
            surface_committed;
        double candidate_dt = 0.0;
        bool candidate_valid = false;
        bool prepared = false;
        bool poisoned = false;
    };
    struct InternalTriplePrimaryCellCandidate
    {
        std::array<double, 7> water_after{};
        std::array<float, 5> number_after{};
        std::array<float, 5> reflectivity_after{};
        std::array<float, 8> rates{};
        double temperature_increment = 0.0;
        double cloud_number_after = 0.0;
    };
    std::optional<InternalTripleCandidateState> internal_triple_;
    // Replaceable execution scratch: deliberately omitted from scheme copies
    // and portable state so checkpoints contain committed microphysics only.
    std::vector<MilbrandtSedimentationWorkspace>
        sedimentation_workspaces_;
    std::vector<float> stochastic_multiplier_workspace_;

public:
    std::unique_ptr<MicrophysicsScheme> clone() const override
    {
        return std::make_unique<MilbrandtScheme>(*this);
    }
    std::unique_ptr<MicrophysicsScheme> clone_parameters() const override
    {
        if (internal_triple_)
        {
            throw std::logic_error(
                "internal Milbrandt triple sedimentation has no parameter-only clone");
        }
        return std::make_unique<MilbrandtScheme>(alpha_r_,
                                                 alpha_i_,
                                                 alpha_s_,
                                                 alpha_g_,
                                                 alpha_h_,
                                                 c_r_,
                                                 d_r_,
                                                 c_i_,
                                                 d_i_,
                                                 c_s_,
                                                 d_s_,
                                                 c_g_,
                                                 d_g_,
                                                 c_h_,
                                                 d_h_,
                                                 a_r_,
                                                 b_r_,
                                                 a_i_,
                                                 b_i_,
                                                 a_s_,
                                                 b_s_,
                                                 a_g_,
                                                 b_g_,
                                                 a_h_,
                                                 b_h_,
                                                 hail_processes_);
    }

    MilbrandtScheme(const MilbrandtScheme& other);
    MilbrandtScheme& operator=(const MilbrandtScheme& other);
    MilbrandtScheme(MilbrandtScheme&&) noexcept = default;
    MilbrandtScheme& operator=(MilbrandtScheme&&) noexcept = default;

    MilbrandtScheme(
        double alpha_r = 0.0, double alpha_i = 0.0, double alpha_s = 0.0,
        double alpha_g = 0.0, double alpha_h = 0.0,
        double c_r = 523.598775598, double d_r = 3.0,
        double c_i = 440.0, double d_i = 3.0,
        double c_s = 0.1597, double d_s = 2.078,
        double c_g = 209.439510239, double d_g = 3.0,
        double c_h = 471.238898038, double d_h = 3.0,
        double a_r = 149.1, double b_r = 0.5,
        double a_i = 71.34, double b_i = 0.6635,
        double a_s = 11.72, double b_s = 0.41,
        double a_g = 19.3, double b_g = 0.37,
        double a_h = 206.89, double b_h = 0.6384,
        bool hail_processes = true
    );

    void configure(const MicrophysicsConfig& config) override;

    bool enable_internal_triple_sedimentation_for_test(
        const Field3D& density,
        const std::array<const Field3D*, 5>& mass,
        const std::array<const Field3D*, 5>& number,
        const std::array<const Field3D*, 5>& reflectivity,
        std::string& error);
    bool enable_internal_triple_warm_for_test(
        const SimulationStateView& state,
        const Field3D& cloud_number,
        const Field3D& rain_number,
        const Field3D& rain_reflectivity,
        std::string& error);
    bool enable_internal_triple_warm_rain_sedimentation_for_test(
        const SimulationStateView& state,
        const Field3D& cloud_number,
        const Field3D& rain_number,
        const Field3D& rain_reflectivity,
        std::string& error);
    bool enable_internal_triple_rain_freezing_for_test(
        const SimulationStateView& state,
        const Field3D& rain_number,
        const Field3D& rain_reflectivity,
        std::string& error);
    bool enable_internal_triple_frozen_diffusion_for_test(
        const SimulationStateView& state,
        milbrandt_detail::TripleMomentSpecies species,
        const Field3D& supplied_number,
        const Field3D& supplied_reflectivity,
        std::string& error);
    bool enable_internal_triple_frozen_composition_for_test(
        const SimulationStateView& state,
        const Field3D& graupel_number,
        const Field3D& graupel_reflectivity,
        const Field3D& hail_number,
        const Field3D& hail_reflectivity,
        std::string& error);
    bool enable_internal_triple_melting_for_test(
        const SimulationStateView& state,
        const Field3D& rain_number,
        const Field3D& rain_reflectivity,
        const Field3D& graupel_number,
        const Field3D& graupel_reflectivity,
        const Field3D& hail_number,
        const Field3D& hail_reflectivity,
        std::string& error);
    bool enable_internal_triple_ice_melting_for_test(
        const SimulationStateView& state,
        const Field3D& rain_number,
        const Field3D& rain_reflectivity,
        const Field3D& ice_number,
        const Field3D& ice_reflectivity,
        std::string& error);
    bool enable_internal_triple_snow_melting_for_test(
        const SimulationStateView& state,
        const Field3D& rain_number,
        const Field3D& rain_reflectivity,
        const Field3D& snow_number,
        const Field3D& snow_reflectivity,
        std::string& error);
    bool enable_internal_triple_warm_primary_for_test(
        const SimulationStateView& state,
        const Field3D& cloud_number,
        const Field3D& rain_number,
        const Field3D& rain_reflectivity,
        std::string& error);
    bool enable_internal_triple_cloud_adjustment_for_test(
        const SimulationStateView& state,
        const Field3D& cloud_number,
        std::string& error);
    bool enable_internal_triple_warm_cloud_primary_for_test(
        const SimulationStateView& state,
        const Field3D& cloud_number,
        const Field3D& rain_number,
        const Field3D& rain_reflectivity,
        std::string& error);
    bool enable_internal_triple_dynamic_warm_cloud_primary_for_test(
        const SimulationStateView& state,
        const Field3D& cloud_number,
        const Field3D& rain_number,
        const Field3D& rain_reflectivity,
        std::string& error);
    bool enable_internal_triple_subcycled_warm_cloud_primary_for_test(
        const SimulationStateView& state,
        const Field3D& cloud_number,
        const Field3D& rain_number,
        const Field3D& rain_reflectivity,
        std::string& error);
    bool enable_internal_triple_frozen_collection_for_test(
        const SimulationStateView& state,
        const Field3D& ice_number,
        const Field3D& ice_reflectivity,
        const Field3D& snow_number,
        const Field3D& snow_reflectivity,
        const Field3D& graupel_number,
        const Field3D& graupel_reflectivity,
        const Field3D& hail_number,
        const Field3D& hail_reflectivity,
        std::string& error);
    bool enable_internal_triple_cold_cloud_collection_for_test(
        const SimulationStateView& state,
        const Field3D& cloud_number,
        const Field3D& snow_number,
        const Field3D& snow_reflectivity,
        const Field3D& graupel_number,
        const Field3D& graupel_reflectivity,
        const Field3D& hail_number,
        const Field3D& hail_reflectivity,
        std::string& error);
    bool enable_internal_triple_cold_cloud_diffusion_for_test(
        const SimulationStateView& state,
        const Field3D& cloud_number,
        const Field3D& snow_number,
        const Field3D& snow_reflectivity,
        const Field3D& graupel_number,
        const Field3D& graupel_reflectivity,
        const Field3D& hail_number,
        const Field3D& hail_reflectivity,
        std::string& error);
    bool enable_internal_triple_cold_riming_for_test(
        const SimulationStateView& state,
        const Field3D& cloud_number,
        const std::array<const Field3D*, 4>& frozen_number,
        const std::array<const Field3D*, 4>& frozen_reflectivity,
        std::string& error);
    bool enable_internal_triple_cold_ice_graupel_for_test(
        const SimulationStateView& state,
        const Field3D& cloud_number,
        const std::array<const Field3D*, 4>& frozen_number,
        const std::array<const Field3D*, 4>& frozen_reflectivity,
        std::string& error);
    bool enable_internal_triple_cold_graupel_conversions_for_test(
        const SimulationStateView& state,
        const Field3D& cloud_number,
        const std::array<const Field3D*, 4>& frozen_number,
        const std::array<const Field3D*, 4>& frozen_reflectivity,
        std::string& error);
    bool enable_internal_triple_cold_homogeneous_for_test(
        const SimulationStateView& state,
        const Field3D& cloud_number,
        const std::array<const Field3D*, 4>& frozen_number,
        const std::array<const Field3D*, 4>& frozen_reflectivity,
        std::string& error);
    bool enable_internal_triple_cold_sll_for_test(
        const SimulationStateView& state,
        const Field3D& cloud_number,
        const std::array<const Field3D*, 4>& frozen_number,
        const std::array<const Field3D*, 4>& frozen_reflectivity,
        std::string& error);
    bool enable_internal_triple_cold_collected_sll_for_test(
        const SimulationStateView& state,
        const Field3D& cloud_number,
        const std::array<const Field3D*, 4>& frozen_number,
        const std::array<const Field3D*, 4>& frozen_reflectivity,
        std::string& error);
    bool enable_internal_triple_cold_rain_sll_for_test(
        const SimulationStateView& state,
        const Field3D& cloud_number,
        const std::array<const Field3D*, 5>& number,
        const std::array<const Field3D*, 5>& reflectivity,
        std::string& error);
    bool enable_internal_triple_cold_collision_sll_for_test(
        const SimulationStateView& state,
        const Field3D& cloud_number,
        const std::array<const Field3D*, 5>& number,
        const std::array<const Field3D*, 5>& reflectivity,
        std::string& error);
    bool enable_internal_triple_cold_wet_growth_sll_for_test(
        const SimulationStateView& state,
        const Field3D& cloud_number,
        const std::array<const Field3D*, 5>& number,
        const std::array<const Field3D*, 5>& reflectivity,
        std::string& error);
    const Field3D& internal_triple_cloud_number_for_test() const;
    const Field3D& internal_triple_reflectivity_for_test(
        std::size_t species) const;
    const Field3D& internal_triple_number_for_test(
        std::size_t species) const;
    milbrandt_detail::TripleMomentSurfaceInventory
    internal_triple_surface_for_test(int i, int j, std::size_t species) const;

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

    std::string get_scheme_name() const override { return "milbrandt"; }

    [[nodiscard]] std::size_t retained_sedimentation_worker_count() const
        noexcept
    {
        return sedimentation_workspaces_.size();
    }
    [[nodiscard]] const MilbrandtSedimentationWorkspace&
    retained_sedimentation_workspace(std::size_t worker) const
    {
        return sedimentation_workspaces_.at(worker);
    }

    PortableSchemeState portable_state() const override;
    PortableSchemeState portable_parameter_state() const override;
    MicrophysicsParameterIdentity parameter_identity() const noexcept override;
    bool restore_portable_state(const PortableSchemeState& state,
                                std::string& error) override;

    void prepare_prognostic_state(
        const Field3D& rho_d,
        const Field3D& qc, const Field3D& qr,
        const Field3D& qi, const Field3D& qs,
        const Field3D& qg, const Field3D& qh) override;
    MicrophysicsPrognosticScalarList
    prognostic_scalars_for_transport() override;
    bool validate_prognostic_update(double dt, std::string& error) const override;
    bool prepare_post_primary_private_commit(
        double dt,
        const MicrophysicsAcceptedPrimaryFields& accepted,
        std::string& error) override;
    bool owns_primary_endpoint_commit() const noexcept override;
    const Field3D* protected_primary_endpoint_wind_source() const noexcept override;
    bool prepare_primary_endpoint_commit(
        const SimulationStateView& state,
        double dt,
        MicrophysicsPrimaryEndpointCandidate& candidate,
        std::string& error) const override;
    bool validate_primary_endpoint_commit(
        const SimulationStateView& state,
        double dt,
        const MicrophysicsPrimaryEndpointCandidate& candidate,
        std::string& error) const override;
    bool owns_post_primary_private_commit() const noexcept override;
    bool commit_prepared_private_candidate() noexcept override;
    bool apply_stochastic_column_multipliers(double              dt,
                                             const Field3D&      density,
                                             std::vector<float>& column_multipliers,
                                             std::string&        error) override;
    void commit_prognostic_update(double dt) noexcept override;
    void invalidate_transient_state_after_domain_recycling() noexcept override;

    double bottom_sedimentation_sink_rate(
        const Field3D& rho_d,
        const Field3D& qr, const Field3D& qs,
        const Field3D& qg, const Field3D& qh,
        int i, int j, double dt_micro) const override;

    double explicit_terminal_velocity_mps(
        std::size_t species, double mixing_ratio,
        double dry_density) const override
    {
        (void)mixing_ratio;
        (void)dry_density;
        return internal_triple_ &&
            internal_triple_->mode == InternalTripleMode::WarmRainSedimentation &&
            species == 0 ? 16.0 : 0.0;
    }

    int get_num_prognostic_vars() const override
    {
        return internal_triple_ ? 18 : 13;
    }

    std::string capability_caveat() const override
    {
        return {};
    }

private:
    static bool validate_internal_triple_source_profile(
        const SimulationStateView& state,
        bool rain_freezing,
        std::string& error,
        std::optional<milbrandt_detail::TripleMomentSpecies> frozen_species =
            std::nullopt,
        bool joint_frozen = false,
        bool melting_only = false,
        bool ice_melting_only = false,
        bool snow_melting_only = false,
        bool warm_primary_only = false,
        bool cloud_adjustment_only = false,
        bool warm_cloud_primary_only = false,
        bool dynamic_warm_cloud_primary_only = false,
        bool subcycled_warm_cloud_primary_only = false,
        bool frozen_collection_primary_only = false,
        bool cold_cloud_collection_primary_only = false,
        bool cold_cloud_diffusion_primary_only = false,
        bool cold_riming_primary_only = false, bool cold_rain_primary_only = false);
    const char* internal_triple_primary_label() const noexcept;
    bool internal_triple_primary_parameters_match(bool frozen_collection = false) const noexcept;
    bool internal_triple_primary_private_source_matches(
        std::string& error) const;
    bool internal_triple_cloud_velocity_source_matches() const noexcept;
    bool enable_internal_triple_primary_mode_for_test(
        const SimulationStateView& state,
        const Field3D* rain_number,
        const Field3D* rain_reflectivity,
        const Field3D& graupel_number,
        const Field3D& graupel_reflectivity,
        const Field3D& hail_number,
        const Field3D& hail_reflectivity,
        InternalTripleMode mode,
        std::string& error);
    bool enable_internal_triple_primary_moments_for_test(
        const SimulationStateView& state,
        const std::array<const Field3D*, 5>& number,
        const std::array<const Field3D*, 5>& reflectivity,
        InternalTripleMode mode,
        std::string& error,
        const Field3D* cloud_number = nullptr);
    bool evaluate_internal_triple_melting_cell(
        const std::array<double, 16>& cell,
        const std::array<float, 5>& reflectivity,
        double dt,
        InternalTriplePrimaryCellCandidate& result,
        std::string& error) const;
    bool evaluate_internal_triple_ice_melting_cell(
        const std::array<double, 16>& cell,
        const std::array<float, 5>& reflectivity,
        double dt,
        InternalTriplePrimaryCellCandidate& result,
        std::string& error) const;
    bool evaluate_internal_triple_snow_melting_cell(
        const std::array<double, 16>& cell,
        const std::array<float, 5>& reflectivity,
        double dt,
        InternalTriplePrimaryCellCandidate& result,
        std::string& error) const;
    bool evaluate_internal_triple_warm_primary_cell(
        const std::array<double, 16>& cell,
        const std::array<float, 5>& reflectivity,
        double dt,
        InternalTriplePrimaryCellCandidate& result,
        std::string& error) const;
    bool evaluate_internal_triple_cloud_adjustment_cell(
        const std::array<double, 16>& cell,
        const std::array<float, 5>& reflectivity,
        double velocity,
        double dt,
        InternalTriplePrimaryCellCandidate& result,
        std::string& error) const;
    bool evaluate_internal_triple_warm_cloud_primary_cell(
        const std::array<double, 16>& cell,
        const std::array<float, 5>& reflectivity,
        double velocity,
        double dt,
        InternalTriplePrimaryCellCandidate& result,
        std::string& error) const;
    bool evaluate_internal_triple_dynamic_warm_cloud_primary_cell(
        const std::array<double, 16>& cell,
        const std::array<float, 5>& reflectivity,
        double velocity,
        double dt,
        InternalTriplePrimaryCellCandidate& result,
        std::string& error) const;
    bool evaluate_internal_triple_subcycled_warm_cloud_primary_cell(
        const std::array<double, 16>& cell,
        const std::array<float, 5>& reflectivity,
        double velocity,
        double dt,
        InternalTriplePrimaryCellCandidate& result,
        std::string& error) const;
    bool evaluate_internal_triple_frozen_collection_primary_cell(
        const std::array<double, 16>& cell,
        const std::array<float, 5>& reflectivity,
        double dt,
        InternalTriplePrimaryCellCandidate& result,
        std::string& error) const;
    bool evaluate_internal_triple_cold_cloud_collection_primary_cell(
        const std::array<double, 16>& cell,
        const std::array<float, 5>& reflectivity,
        double dt,
        InternalTriplePrimaryCellCandidate& result,
        std::string& error) const;
    bool evaluate_internal_triple_cold_cloud_diffusion_primary_cell(
        const std::array<double, 16>& cell,
        const std::array<float, 5>& reflectivity,
        double dt,
        InternalTriplePrimaryCellCandidate& result,
        std::string& error) const;
    bool evaluate_internal_triple_cold_riming_primary_cell(
        const std::array<double, 16>& cell,
        const std::array<float, 5>& reflectivity,
        double dt,
        InternalTriplePrimaryCellCandidate& result,
        std::string& error) const;
    bool prepare_internal_triple_primary_private_commit(
        double dt,
        const MicrophysicsAcceptedPrimaryFields& accepted,
        std::string& error);
    void compute_internal_triple_primary_tendencies(
        const Field3D& p,
        const Field3D& theta,
        const Field3D& density,
        const Field3D& vapor,
        const Field3D& cloud,
        const std::array<const Field3D*, 5>& mass,
        double dt,
        const std::array<Field3D*, 8>& tendencies);
    void compute_internal_triple_frozen_diffusion_tendencies(
        const Field3D& p,
        const Field3D& theta,
        const Field3D& density,
        const Field3D& vapor,
        const Field3D& cloud,
        const std::array<const Field3D*, 5>& mass,
        double dt,
        const std::array<Field3D*, 8>& tendencies);
    void compute_internal_triple_rain_freezing_tendencies(
        const Field3D& p,
        const Field3D& theta,
        const Field3D& density,
        const Field3D& vapor,
        const Field3D& cloud,
        const std::array<const Field3D*, 5>& mass,
        double dt,
        const std::array<Field3D*, 8>& tendencies);
    void compute_internal_triple_sedimentation_tendencies(
        const Field3D& density,
        const std::array<const Field3D*, 5>& mass,
        double dt,
        const std::array<Field3D*, 5>& mass_tendencies,
        Field3D& theta_tendency,
        Field3D& vapor_tendency,
        Field3D& cloud_tendency);
    void compute_internal_triple_warm_tendencies(
        const Field3D& p,
        const Field3D& theta,
        const Field3D& density,
        const Field3D& vapor,
        const Field3D& cloud,
        const std::array<const Field3D*, 5>& mass,
        double dt,
        const std::array<Field3D*, 8>& tendencies);
    bool enable_internal_triple_warm_mode_for_test(
        const SimulationStateView& state,
        const Field3D& cloud_number,
        const Field3D& rain_number,
        const Field3D& rain_reflectivity,
        InternalTripleMode mode,
        std::string& error);
    bool native_execution_selected() const noexcept;
    void compute_native_tendencies(const std::array<const Field3D*, 10>& state,
                                   double                                dt,
                                   const std::array<Field3D*, 8>&        output);

    // Seeds scheme-private prognostic moments for a new grid shape.
    void initialize_prognostic_fields(
        const Field3D& rho_d,
        const Field3D& qc, const Field3D& qr,
        const Field3D& qi, const Field3D& qs,
        const Field3D& qg, const Field3D& qh);

    void compute_warm_rain_processes(
        const Field3D& temperature,
        const Field3D& p,
        const Field3D& rho,
        const Field3D& qv,
        const Field3D& qc,
        const Field3D& qr,
        Field3D& dqc_dt,
        Field3D& dqr_dt,
        Field3D& dqv_dt,
        Field3D& dtheta_dt,
        Field3D& dNc_dt,
        Field3D& dNr_dt,
        double dt
    );

    void compute_ice_processes(
        const Field3D& temperature,
        const Field3D& p,
        const Field3D& rho,
        const Field3D& qv,
        const Field3D& qc,
        const Field3D& qr,
        const Field3D& qi,
        const Field3D& qs,
        const Field3D& qg,
        const Field3D& qh,
        Field3D& dqc_dt,
        Field3D& dqr_dt,
        Field3D& dqi_dt,
        Field3D& dqs_dt,
        Field3D& dqg_dt,
        Field3D& dqh_dt,
        Field3D& dqv_dt,
        Field3D& dtheta_dt,
        Field3D& dNc_dt,
        Field3D& dNr_dt,
        Field3D& dNi_dt,
        Field3D& dNs_dt,
        Field3D& dNg_dt,
        Field3D& dNh_dt,
        double dt
    );

    void compute_melting_processes(
        const Field3D& temperature,
        const Field3D& p,
        const Field3D& rho,
        const Field3D& qv,
        const Field3D& qi,
        const Field3D& qs,
        const Field3D& qg,
        const Field3D& qh,
        Field3D& dqi_dt,
        Field3D& dqs_dt,
        Field3D& dqg_dt,
        Field3D& dqh_dt,
        Field3D& dqr_dt,
        Field3D& dtheta_dt,
        Field3D& dNr_dt,
        Field3D& dNi_dt,
        Field3D& dNs_dt,
        Field3D& dNg_dt,
        Field3D& dNh_dt,
        double dt
    );

    void compute_sedimentation(
        const Field3D& rho_d,
        const Field3D& qr,
        const Field3D& qi,
        const Field3D& qs,
        const Field3D& qg,
        const Field3D& qh,
        const Field3D& Nr,
        const Field3D& Ni,
        const Field3D& Ns,
        const Field3D& Ng,
        const Field3D& Nh,
        Field3D& dqr_dt,
        Field3D& dqi_dt,
        Field3D& dqs_dt,
        Field3D& dqg_dt,
        Field3D& dqh_dt,
        Field3D& dNr_dt,
        Field3D& dNi_dt,
        Field3D& dNs_dt,
        Field3D& dNg_dt,
        Field3D& dNh_dt,
        double dt
    );
};

} // namespace tmv

using tmv::MilbrandtScheme;
