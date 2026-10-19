#pragma once

#include "microphysics/milbrandt_melting_parameters.hpp"
#include "microphysics/milbrandt_cold_parameters.hpp"

#include <array>

#include "compute/thompson_cloud_tables.hpp"
#include "compute/thompson_freezing_tables.hpp"
#include "compute/thompson_deposition_tables.hpp"
#include "compute/thompson_completion.hpp"

/**
 * @file microphysics.hpp
 * @brief Microphysics compute dispatch data and capability interface.
 */

enum class MicrophysicsKernelOp
{
    LinFused,
    LinPointwise,
    LinSedimentation,
    MicrophysicsApply,
    KesslerPointwise,
    KesslerSedimentation,
    ArchivedLegacyThompsonLinPointwise,
    ArchivedLegacyThompsonLinSedimentation,
    ThompsonMomentSedimentation,
    ThompsonCloudAdjustment,
    ThompsonWarmRain,
    ThompsonCloudPhase,
    ThompsonFreezing,
    ThompsonDeposition,
    ThompsonCompletion,
    MilbrandtWarmRain,
    MilbrandtCloudAdjustment,
    MilbrandtMelting,
    MilbrandtCold,
    MilbrandtSedimentation
};

struct MilbrandtSedimentationDispatch
{
    // Mass, number/kg dry air, dry density, split layer thickness, mass rate, number rate.
    std::array<const float*, 7> state{};
    // Completed mass/number rates and surface-loss rate at k=0 for each column.
    std::array<float*, 3> output{};
    int                   species          = 0;
    int                   maximum_substeps = 100000;
    int                   nx = 0, ny = 0, nz = 0;
    int                   i_begin = 0, i_end = 0, j_begin = 0, j_end = 0, k_begin = 0, k_end = 0;
    double                dt = 0;
};

struct MilbrandtColdDispatch
{
    // T,p,dry density,qv,qc,qr,qi,qs,qg,qh,Nc,Nr,Ni,Ns,Ng,Nh; number per kg dry air.
    std::array<const float*, 16> state{};
    // qc,qr,qi,qs,qg,qh,qv,T,Nc,Nr,Ni,Ns,Ng,Nh tendencies; T in K/s.
    std::array<const float*, 14> rates{};
    std::array<float*, 14>       output{};
    std::array<double, 25>       parameters     = tmv::milbrandt_detail::default_cold_parameters;
    bool                         hail_processes = true;
    int                          nx = 0, ny = 0, nz = 0;
    int                          i_begin = 0, i_end = 0, j_begin = 0, j_end = 0, k_begin = 0, k_end = 0;
    double                       dt = 0;
};

struct MilbrandtMeltingDispatch
{
    // T (K), p (Pa), dry density, vapor, ice/snow/graupel/hail mass, then their number per kg dry air.
    std::array<const float*, 12> state{};
    // Ice/snow/graupel/hail/rain mass rates, temperature rate, rain/ice/snow/graupel/hail number rates.
    std::array<float*, 11> output{};
    std::array<double, 13> parameters = tmv::milbrandt_detail::default_melting_parameters;
    int                    nx = 0, ny = 0, nz = 0;
    int                    i_begin = 0, i_end = 0, j_begin = 0, j_end = 0, k_begin = 0, k_end = 0;
    double                 dt = 0;
};

struct MilbrandtCloudAdjustmentDispatch
{
    // Temperature (K), pressure, dry density, vapor, cloud, Nc/kg, vertical velocity.
    std::array<const float*, 7> state{};
    // Adjusted T/qv/qc, their three tendencies, and the Nc/kg tendency.
    std::array<float*, 7> output{};
    int                   nx = 0, ny = 0, nz = 0;
    int                   i_begin = 0, i_end = 0, j_begin = 0, j_end = 0, k_begin = 0, k_end = 0;
    bool                  face_staggered_velocity = true;
    const double*         lower_boundary_vertical_velocity = nullptr;
    double                dt                      = 0;
};

struct MilbrandtWarmRainDispatch
{
    // Temperature, pressure, dry density, vapor, cloud, rain, Nc/kg, Nr/kg.
    std::array<const float*, 8> state{};
    // Cloud, rain, vapor, temperature, Nc/kg, Nr/kg tendencies, in this order.
    std::array<const float*, 6> rates{};
    std::array<float*, 6>       output{};
    std::array<double, 5>       parameters{0.0, 523.598775598, 3.0, 149.1, 0.5};
    int                         nx = 0, ny = 0, nz = 0;
    int                         i_begin = 0, i_end = 0, j_begin = 0, j_end = 0, k_begin = 0, k_end = 0;
    double                      dt = 0;
};

struct ThompsonDepositionDispatch
{
    const float*                                                     temperature        = nullptr;
    const float*                                                     pressure           = nullptr;
    const float*                                                     density            = nullptr;
    const float*                                                     vapor              = nullptr;
    const float*                                                     ice                = nullptr;
    const float*                                                     snow               = nullptr;
    const float*                                                     graupel            = nullptr;
    const float*                                                     ice_number         = nullptr;
    const float*                                                     cloud_vapor_change = nullptr;
    const float*                                                     cloud_vapor_change_low = nullptr;
    const float*                                                     evaporation_high   = nullptr;
    const float*                                                     evaporation_low    = nullptr;
    std::shared_ptr<const nimbus::compute::ThompsonDepositionTables> tables;
    float*                                                           ice_change_high         = nullptr;
    float*                                                           ice_change_low          = nullptr;
    float*                                                           snow_change_high        = nullptr;
    float*                                                           snow_change_low         = nullptr;
    float*                                                           graupel_change_high     = nullptr;
    float*                                                           graupel_change_low      = nullptr;
    float*                                                           temperature_change_high = nullptr;
    float*                                                           temperature_change_low  = nullptr;
    float*                                                           ice_number_tendency     = nullptr;
    float*                                                           snow_vapor_rate_high    = nullptr;
    float*                                                           snow_vapor_rate_low     = nullptr;
    float*                                                           lookup_index            = nullptr;
    float*                                                           status                  = nullptr;
    int                                                              nx                      = 0;
    int                                                              ny                      = 0;
    int                                                              nz                      = 0;
    int                                                              i_begin                 = 0;
    int                                                              i_end                   = 0;
    int                                                              j_begin                 = 0;
    int                                                              j_end                   = 0;
    int                                                              k_begin                 = 0;
    int                                                              k_end                   = 0;
    double                                                           dt                      = 0;
};

struct ThompsonFreezingDispatch
{
    const float*                                                   temperature         = nullptr;
    const float*                                                   density             = nullptr;
    const float*                                                   cloud               = nullptr;
    const float*                                                   rain                = nullptr;
    const float*                                                   cloud_number        = nullptr;
    const float*                                                   rain_number         = nullptr;
    const float*                                                   ice_aerosol         = nullptr;
    const float*                                                   cloud_vapor_change  = nullptr;
    const float*                                                   cloud_vapor_change_low = nullptr;
    const float*                                                   conversion_high     = nullptr;
    const float*                                                   conversion_low      = nullptr;
    const float*                                                   evaporation_high    = nullptr;
    const float*                                                   evaporation_low     = nullptr;
    const float*                                                   cloud_number_source = nullptr;
    const float*                                                   rain_number_source  = nullptr;
    std::shared_ptr<const nimbus::compute::ThompsonFreezingTables> tables;
    float*                                                         cloud_frozen_high        = nullptr;
    float*                                                         cloud_frozen_low         = nullptr;
    float*                                                         rain_ice_high            = nullptr;
    float*                                                         rain_ice_low             = nullptr;
    float*                                                         rain_graupel_high        = nullptr;
    float*                                                         rain_graupel_low         = nullptr;
    float*                                                         next_cloud_number_source = nullptr;
    float*                                                         next_rain_number_source  = nullptr;
    float*                                                         rain_ice_number_source   = nullptr;
    float*                                                         cloud_ice_number_source  = nullptr;
    float*                                                         rain_lookup              = nullptr;
    float*                                                         cloud_lookup             = nullptr;
    float*                                                         status                   = nullptr;
    int                                                            nx                       = 0;
    int                                                            ny                       = 0;
    int                                                            nz                       = 0;
    int                                                            i_begin                  = 0;
    int                                                            i_end                    = 0;
    int                                                            j_begin                  = 0;
    int                                                            j_end                    = 0;
    int                                                            k_begin                  = 0;
    int                                                            k_end                    = 0;
    double                                                         dt                       = 0;
};

struct ThompsonCloudPhaseDispatch
{
    const float* temperature               = nullptr;
    const float* vapor                     = nullptr;
    const float* density                   = nullptr;
    const float* cloud                     = nullptr;
    const float* ice                       = nullptr;
    const float* cloud_number              = nullptr;
    const float* ice_number                = nullptr;
    const float* cloud_tendency            = nullptr;
    const float* ice_tendency              = nullptr;
    const float* cloud_number_source       = nullptr;
    const float* ice_number_source         = nullptr;
    const float* temperature_tendency      = nullptr;
    float*       next_cloud_tendency       = nullptr;
    float*       next_ice_tendency         = nullptr;
    float*       next_cloud_number_source  = nullptr;
    float*       next_ice_number_source    = nullptr;
    float*       next_temperature_tendency = nullptr;
    float*       status                    = nullptr;
    int          nx                        = 0;
    int          ny                        = 0;
    int          nz                        = 0;
    int          i_begin                   = 0;
    int          i_end                     = 0;
    int          j_begin                   = 0;
    int          j_end                     = 0;
    int          k_begin                   = 0;
    int          k_end                     = 0;
    double       dt                        = 0;
};

struct ThompsonWarmRainDispatch
{
    std::shared_ptr<const std::vector<float>> coefficients;
    const float*                              temperature           = nullptr;
    const float*                              pressure              = nullptr;
    const float*                              density               = nullptr;
    const float*                              vapor                 = nullptr;
    const float*                              cloud                 = nullptr;
    const float*                              rain                  = nullptr;
    const float*                              cloud_number          = nullptr;
    const float*                              rain_number           = nullptr;
    const float*                              water_aerosol         = nullptr;
    const float*                              cloud_vapor_change    = nullptr;
    const float*                              cloud_vapor_change_low = nullptr;
    const float*                              cloud_number_source   = nullptr;
    const float*                              cloud_aerosol_source  = nullptr;
    float*                                    cloud_to_rain_high    = nullptr;
    float*                                    cloud_to_rain_low     = nullptr;
    float*                                    rain_to_vapor_high    = nullptr;
    float*                                    rain_to_vapor_low     = nullptr;
    float*                                    temperature_high      = nullptr;
    float*                                    temperature_low       = nullptr;
    float*                                    cloud_number_tendency = nullptr;
    float*                                    rain_number_tendency  = nullptr;
    float*                                    aerosol_tendency      = nullptr;
    float*                                    status                = nullptr;
    int                                       nx                    = 0;
    int                                       ny                    = 0;
    int                                       nz                    = 0;
    int                                       i_begin               = 0;
    int                                       i_end                 = 0;
    int                                       j_begin               = 0;
    int                                       j_end                 = 0;
    int                                       k_begin               = 0;
    int                                       k_end                 = 0;
    double                                    dt                    = 0.0;
};

struct ThompsonCloudAdjustmentDispatch
{
    const float*                                                temperature   = nullptr;
    const float*                                                pressure      = nullptr;
    const float*                                                density       = nullptr;
    const float*                                                vapor         = nullptr;
    const float*                                                cloud         = nullptr;
    const float*                                                cloud_number  = nullptr;
    const float*                                                water_aerosol = nullptr;
    const float*                                                velocity      = nullptr;
    std::shared_ptr<const nimbus::compute::ThompsonCloudTables> tables;
    float*                                                      vapor_change          = nullptr;
    float*                                                      temperature_change    = nullptr;
    float*                                                      cloud_number_tendency = nullptr;
    float*                                                      aerosol_tendency      = nullptr;
    float*                                                      lookup_index          = nullptr;
    float*                                                      vapor_change_low       = nullptr;
    float*                                                      temperature_change_low = nullptr;
    int                                                         nx                    = 0;
    int                                                         ny                    = 0;
    int                                                         nz                    = 0;
    int                                                         i_begin               = 0;
    int                                                         i_end                 = 0;
    int                                                         j_begin               = 0;
    int                                                         j_end                 = 0;
    int                                                         k_begin               = 0;
    int                                                         k_end                 = 0;
    double                                                      dt                    = 0.0;
    const float*                                                potential_temperature = nullptr;
};

enum class ThompsonSedimentationSpecies : int
{
    Cloud,
    Rain,
    Ice,
    Snow,
    Graupel
};

struct ThompsonMomentSedimentationDispatch
{
    const float*                 mass                  = nullptr;
    const float*                 number                = nullptr;
    const float*                 density               = nullptr;
    const float*                 temperature           = nullptr;
    const float*                 thickness             = nullptr;
    const float*                 vertical_velocity     = nullptr;
    const float*                 snow_speed_multiplier = nullptr;
    float*                       mass_tendency         = nullptr;
    float*                       number_tendency       = nullptr;
    float*                       surface_mass          = nullptr;
    int                          nx                    = 0;
    int                          ny                    = 0;
    int                          nz                    = 0;
    int                          i_begin               = 0;
    int                          i_end                 = 0;
    int                          j_begin               = 0;
    int                          j_end                 = 0;
    int                          k_begin               = 0;
    int                          k_end                 = 0;
    float                        dt                    = 0.0f;
    ThompsonSedimentationSpecies species               = ThompsonSedimentationSpecies::Rain;
    int                          maximum_substeps      = 4096;
};

struct MicrophysicsApplyDispatch
{
    float* theta_data = nullptr;
    float* qv_data = nullptr;
    float* qc_data = nullptr;
    float* qr_data = nullptr;
    float* qi_data = nullptr;
    float* qs_data = nullptr;
    float* qg_data = nullptr;
    float* qh_data = nullptr;
    const float* dtheta = nullptr;
    const float* dqv = nullptr;
    const float* dqc = nullptr;
    const float* dqr = nullptr;
    const float* dqi = nullptr;
    const float* dqs = nullptr;
    const float* dqg = nullptr;
    const float* dqh = nullptr;
    const float* dtheta_rad = nullptr;
    const float* dtheta_pbl = nullptr;
    const float* dqv_pbl = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int i_begin = 0;
    int i_end = 0;
    int j_begin = 0;
    int j_end = 0;
    float dt_step = 0.0f;
    float theta_fallback = 0.0f;
    float theta_min = 0.0f;
    float theta_max = 0.0f;
    float max_theta_step = 0.0f;
    float qv_max_val = 0.0f;
    float hydro_max_val = 0.0f;
    unsigned int* stats_out = nullptr;
    int stats_len = 0;
};

struct LinPointwiseDispatch
{
    const float* theta_data = nullptr;
    const float* p_data = nullptr;
    const float* qv = nullptr;
    const float* qc = nullptr;
    const float* qr = nullptr;
    const float* qi         = nullptr;
    const float* qs         = nullptr;
    const float* qg         = nullptr;
    const float* qh         = nullptr;
    float*       dtheta_dt  = nullptr;
    float*       dqv_dt     = nullptr;
    float*       dqc_dt     = nullptr;
    float*       dqr_dt     = nullptr;
    float*       dqi_dt     = nullptr;
    float*       dqs_dt     = nullptr;
    float*       dqg_dt     = nullptr;
    float*       dqh_dt     = nullptr;
    int          nx         = 0;
    int          ny         = 0;
    int          nz         = 0;
    float        dt_step    = 0.0f;
    float        inv_dt     = 0.0f;
    float        qc0        = 0.0f;
    float        c_auto     = 0.0f;
    float        c_accr     = 0.0f;
    float        c_evap     = 0.0f;
    float        c_ihom     = 0.0f;
    float        c_rime     = 0.0f;
    float        c_agg      = 0.0f;
    float        c_melt_s   = 0.0f;
    float        c_melt_g   = 0.0f;
    float        c_melt_h   = 0.0f;
    float        c_subl     = 0.0f;
    float        Lv_cp      = 0.0f;
    float        Lf_cp      = 0.0f;
    float        Ls_cp      = 0.0f;
    int          i_begin = 0, i_end = -1;
    int          j_begin = 0, j_end = -1;
    int          k_begin = 0, k_end = -1;
};

struct LinSedimentationDispatch
{
    const float* qr      = nullptr;
    const float* qs      = nullptr;
    const float* qg      = nullptr;
    const float* qh      = nullptr;
    const float* rho_d   = nullptr;
    float*       dqr_dt  = nullptr;
    float*       dqs_dt  = nullptr;
    float*       dqg_dt  = nullptr;
    float*       dqh_dt  = nullptr;
    int          nx      = 0;
    int          ny      = 0;
    int          nz      = 0;
    float        dz      = 0.0f;
    float        vt_cap  = 0.0f;
    int          i_begin = 0, i_end = -1;
    int          j_begin = 0, j_end = -1;
    int          k_begin = 0, k_end = -1;
    const float* layer_thickness_m = nullptr;
};

struct LinFusedDispatch
{
    LinPointwiseDispatch     pointwise;
    LinSedimentationDispatch sedimentation;
};

struct KesslerPointwiseDispatch
{
    const float* temperature_data       = nullptr;
    const float* p_data = nullptr;
    const float* qv_data = nullptr;
    const float* qc_data = nullptr;
    const float* qr_data = nullptr;
    const float* rho_d_data = nullptr;
    const float* qh_data = nullptr;
    float* dtheta_dt_data = nullptr;
    float* dqv_dt_data = nullptr;
    float* dqc_dt_data = nullptr;
    float* dqr_dt_data = nullptr;
    float* dqg_dt_data = nullptr;
    float* dqh_dt_data = nullptr;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    int i_begin = 0;
    int i_end = 0;
    int j_begin = 0;
    int j_end = 0;
    int k_begin = 0;
    int k_end = 0;
    float qc0 = 0.0f;
    float c_auto = 0.0f;
    float c_accr = 0.0f;
    float evaporation_efficiency = 0.0f;
    float Lv_cp = 0.0f;
    float T0 = 0.0f;
    float dt_step = 0.0f;
};

struct KesslerSedimentationDispatch
{
    const float* qr_data = nullptr;
    const float* qg_data = nullptr;
    const float* qh_data = nullptr;
    const float* rho_d_data = nullptr;
    float* dqr_dt_data = nullptr;
    float* dqg_dt_data = nullptr;
    float* dqh_dt_data = nullptr;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    int i_begin = 0;
    int i_end = 0;
    int j_begin = 0;
    int j_end = 0;
    int k_begin = 0;
    int k_end = 0;
    float dz_val = 0.0f;
    float Vt_max_rain = 0.0f;
    float vt_cap = 0.0f;
    float dt_step = 0.0f;
    const float* layer_thickness_m = nullptr;
};

struct ArchivedLegacyThompsonLinPointwiseDispatch
{
    const float* temperature_data = nullptr;
    const float* p_data = nullptr;
    const float* qv_data = nullptr;
    const float* qc_data = nullptr;
    const float* qr_data = nullptr;
    const float* qi_data = nullptr;
    const float* qs_data = nullptr;
    const float* qg_data = nullptr;
    const float* qh_data = nullptr;
    float* dtheta_dt_data = nullptr;
    float* dqv_dt_data = nullptr;
    float* dqc_dt_data = nullptr;
    float* dqr_dt_data = nullptr;
    float* dqi_dt_data = nullptr;
    float* dqs_dt_data = nullptr;
    float* dqg_dt_data = nullptr;
    float* dqh_dt_data = nullptr;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    float qc0 = 0.0f;
    float c_auto = 0.0f;
    float c_evap = 0.0f;
    float c_dep = 0.0f;
    float c_subl = 0.0f;
    float c_melt = 0.0f;
    float Lv_cp = 0.0f;
    float Lf_cp = 0.0f;
    float Ls_cp = 0.0f;
    float T0 = 0.0f;
    float ccn_conc = 0.0f;
    float in_conc = 0.0f;
};

struct ArchivedLegacyThompsonLinSedimentationDispatch
{
    const float* qr_data = nullptr;
    const float* qs_data = nullptr;
    const float* qg_data = nullptr;
    const float* qh_data = nullptr;
    float* dqr_dt_data = nullptr;
    float* dqs_dt_data = nullptr;
    float* dqg_dt_data = nullptr;
    float* dqh_dt_data = nullptr;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    float dz_val = 0.0f;
    float a_rain = 0.0f;
    float b_rain = 0.0f;
    float Vt_max_rain = 0.0f;
    float a_snow = 0.0f;
    float b_snow = 0.0f;
    float Vt_max_snow = 0.0f;
    float a_grau = 0.0f;
    float b_grau = 0.0f;
    float Vt_max_grau = 0.0f;
    float a_hail = 0.0f;
    float b_hail = 0.0f;
    float Vt_max_hail = 0.0f;
};


class MicrophysicsComputeKernels
{
public:
    virtual ~MicrophysicsComputeKernels() = default;

    virtual bool dispatch_thompson_completion(const ThompsonCompletionDispatch&)
    {
        return false;
    }

    virtual bool dispatch_thompson_deposition(const ThompsonDepositionDispatch&)
    {
        return false;
    }

    virtual bool dispatch_thompson_freezing(const ThompsonFreezingDispatch&)
    {
        return false;
    }

    virtual bool dispatch_thompson_cloud_phase(const ThompsonCloudPhaseDispatch&)
    {
        return false;
    }

    virtual bool dispatch_milbrandt_sedimentation(const MilbrandtSedimentationDispatch&)
    {
        return false;
    }

    virtual bool dispatch_milbrandt_cold(const MilbrandtColdDispatch&)
    {
        return false;
    }

    virtual bool dispatch_milbrandt_melting(const MilbrandtMeltingDispatch&)
    {
        return false;
    }

    virtual bool dispatch_milbrandt_cloud_adjustment(const MilbrandtCloudAdjustmentDispatch&)
    {
        return false;
    }

    virtual bool dispatch_milbrandt_warm_rain(const MilbrandtWarmRainDispatch&)
    {
        return false;
    }

    virtual bool dispatch_thompson_warm_rain(const ThompsonWarmRainDispatch&)
    {
        return false;
    }

    virtual bool dispatch_thompson_cloud_adjustment(const ThompsonCloudAdjustmentDispatch&)
    {
        return false;
    }

    virtual bool dispatch_thompson_moment_sedimentation(const ThompsonMomentSedimentationDispatch&)
    {
        return false;
    }

    /**
     * @brief Returns true when this backend can execute the requested microphysics operation.
     */
    virtual bool supports(MicrophysicsKernelOp /*op*/) const { return false; }

    // Direct native publication requires unchanged host outputs on refusal
    // or exception, including failures in residency bookkeeping.
    virtual bool supports_atomic_lin_fused() const { return false; }

    /** Execute Lin pointwise physics and column sedimentation in one
     * dependency-ordered submission, publishing only final tendencies. */
    virtual bool dispatch_lin_fused(const LinFusedDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    // ── Lin microphysics dispatch (GPU audit Phase 3.3) ─────────

    /**
     * @brief Execute the fused point-wise Lin microphysics on device: the
     *        full mass-limited process chain of LinScheme::compute_tendencies
     *        including the theta->temperature conversion and the Exner
     *        back-conversion of the heating tendency.
     *
     * @param dt_step / inv_dt  The step and its reciprocal, each rounded
     *                          once from fp64 on the host.
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_lin_pointwise(const LinPointwiseDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    /**
     * @brief Execute Lin column sedimentation on device: per-(i,j)-column
     *        terminal-velocity fallout for rain/snow/graupel/hail,
     *        accumulating onto the point-wise tendencies in place.
     *
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_lin_sedimentation(const LinSedimentationDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
    // ── Microphysics tendency application (GPU audit Phase 4.2) ──

    /**
     * @brief Apply microphysics (+ radiation/PBL) tendencies to the eight
     *        thermodynamic state fields in place, with the driver's exact
     *        guard semantics: per-read non-finite tendency zeroing, the
     *        +-max_theta_step limiter, and the physical bounds clamps.
     *
     * @param stats_out Guard telemetry, length >= 10 + nz + nx:
     *        [0] non-finite tendency reads, [1] non-finite dtheta totals,
     *        [2] limited dtheta totals, [3] theta bounds clamps,
     *        [4] first-clamp flag, [5] first-clamp flat index,
     *        [6..9] first-clamp detail (theta_new, theta_bounded,
     *        dtheta_total, dtheta_limited as float bits),
     *        [10..10+nz) per-k and [10+nz..10+nz+nx) per-i clamp counts.
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_microphysics_apply(const MicrophysicsApplyDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
    // ── Kessler point-wise microphysics dispatch ────────────────────

    /**
     * @brief Execute Kessler three-species warm rain on device.
     *
     * Frozen-species arguments are retained only as inert ABI compatibility
     * slots and their tendencies are written as exact zero.
     *
     * @param temperature_data  Temperature field, row-major [nr][nth][nz], float.
     * @param p_data            Pressure field, same layout, float.
     * @param qv_data           Water vapor mixing ratio, same layout, float.
     * @param qc_data           Cloud water mixing ratio, same layout, float.
     * @param qr_data           Rain mixing ratio, same layout, float.
     * @param rho_d_data        Live dry-air density for rain evaporation.
     * @param qh_data           Inert compatibility slot.
     * @param dtheta_dt_data    Output: temperature tendency (pre-theta conversion).
     * @param dqv_dt_data       Output: water vapor tendency.
     * @param dqc_dt_data       Output: cloud water tendency.
     * @param dqr_dt_data       Output: rain tendency.
     * @param dqg_dt_data       Output: exact zero.
     * @param dqh_dt_data       Output: exact zero.
     * @param nr                Radial dimension.
     * @param nth               Azimuthal dimension.
     * @param nz                Vertical dimension.
     * @param qc0               Autoconversion threshold.
     * @param c_auto            Autoconversion rate.
     * @param c_accr            Accretion rate.
     * @param evaporation_efficiency Dimensionless multiplier on the
     *                         reference Kessler evaporation kernel.
     * @param Lv_cp             L_v / cp (precomputed).
     * @param T0                Freezing temperature (K).
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_kessler_pointwise(const KesslerPointwiseDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
    // ── Kessler sedimentation dispatch ──────────────────────────────

    /**
     * @brief Execute Kessler sedimentation on device (column-wise).
     *
     * Reads existing tendency values (from point-wise pass) and adds
     * sedimentation contributions in-place.
     *
     * @param qr_data           Rain mixing ratio (readonly), row-major [nr][nth][nz].
     * @param qg_data           Inert compatibility slot.
     * @param qh_data           Inert compatibility slot.
     * @param dqr_dt_data       Rain tendency (read-write, accumulated).
     * @param dqg_dt_data       Inert compatibility slot.
     * @param dqh_dt_data       Inert compatibility slot.
     * @param nr                Radial dimension.
     * @param nth               Azimuthal dimension.
     * @param nz                Vertical dimension.
     * @param dz_val            Vertical grid spacing.
     * @param Vt_max_rain       Rain terminal velocity cap.
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_kessler_sedimentation(
        const KesslerSedimentationDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    // ── Archived legacy Thompson-named Lin-class dispatch ──────────────

    /**
     * @brief Retained ABI hook for an orphaned Lin-class shader.
     *
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_archived_legacy_thompson_lin_pointwise(const ArchivedLegacyThompsonLinPointwiseDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    /**
     * @brief Retained ABI hook for orphaned Lin-class sedimentation.
     *
     * @return True if dispatch succeeded.
     */
    virtual bool dispatch_archived_legacy_thompson_lin_sedimentation(
        const ArchivedLegacyThompsonLinSedimentationDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }
};
