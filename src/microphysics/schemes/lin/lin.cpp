#include "compute/backend_runtime.hpp"
#include "compute/backend_interface.hpp"
// Lin bulk microphysics with coupled pointwise and sedimentation kernels.

#include "lin.hpp"
#include "compute/compute_kernel_template.hpp"
#include "compute/parity_check.hpp"
#include "microphysics/base/fall_speeds.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace tmv
{

/**
 * @brief Initializes Lin scheme coefficients and hydrometeor parameters.
 */
LinScheme::LinScheme(
    double qc0, double c_auto, double c_accr, double c_evap,
    double c_ihom, double c_rime, double c_agg,
    double c_melt_s, double c_melt_g, double c_melt_h, double c_subl
) :
    qc0_(qc0), c_auto_(c_auto), c_accr_(c_accr), c_evap_(c_evap),
    c_ihom_(c_ihom), c_rime_(c_rime), c_agg_(c_agg),
    c_melt_s_(c_melt_s), c_melt_g_(c_melt_g), c_melt_h_(c_melt_h),
    c_subl_(c_subl) {}

void LinScheme::configure(const MicrophysicsConfig& config)
{
    qc0_ = config.shared.cloud_water_threshold_kgkg;
    c_auto_ = config.shared.autoconversion_rate_s_inv;
    c_accr_ = config.lin.accretion_coefficient;
    c_evap_ = config.lin.evaporation_rate_s_inv;
}

PortableSchemeState LinScheme::portable_state() const
{
    PortableSchemeState state;
    state.version = 1;
    state.reals = {qc0_, c_auto_, c_accr_, c_evap_, c_ihom_, c_rime_,
                   c_agg_, c_melt_s_, c_melt_g_, c_melt_h_, c_subl_};
    return state;
}

MicrophysicsParameterIdentity LinScheme::parameter_identity() const noexcept
{
    MicrophysicsParameterIdentity identity;
    identity.version = 1;
    identity.real_count = 11;
    identity.reals = {
        qc0_, c_auto_, c_accr_, c_evap_, c_ihom_, c_rime_, c_agg_,
        c_melt_s_, c_melt_g_, c_melt_h_, c_subl_};
    return identity;
}

bool LinScheme::restore_portable_state(const PortableSchemeState& state,
                                       std::string& error)
{
    if (state.version != 1 || !state.fields.empty() ||
        !state.integers.empty() || state.reals.size() != 11)
    {
        error = "invalid Lin portable-state layout";
        return false;
    }
    for (double value : state.reals)
    {
        if (!std::isfinite(value) || value < 0.0)
        {
            error = "invalid Lin portable-state parameter";
            return false;
        }
    }
    std::size_t i = 0;
    qc0_ = state.reals[i++];
    c_auto_ = state.reals[i++];
    c_accr_ = state.reals[i++];
    c_evap_ = state.reals[i++];
    c_ihom_ = state.reals[i++];
    c_rime_ = state.reals[i++];
    c_agg_ = state.reals[i++];
    c_melt_s_ = state.reals[i++];
    c_melt_g_ = state.reals[i++];
    c_melt_h_ = state.reals[i++];
    c_subl_ = state.reals[i];
    error.clear();
    return true;
}

/**
 * @brief Computes the tendencies for the Lin scheme.
 */
double LinScheme::explicit_terminal_velocity_mps(
    std::size_t species, double mixing_ratio, double dry_density) const
{
    switch (species)
    {
    case 0: return fall_speeds::rain_mass_weighted(mixing_ratio, dry_density) *
        std::sqrt(fall_speeds::kFallbackAirDensity / dry_density);
    case 1: return fall_speeds::snow_mass_weighted(mixing_ratio, dry_density) *
        std::sqrt(fall_speeds::kFallbackAirDensity / dry_density);
    case 2: return fall_speeds::graupel_mass_weighted(mixing_ratio, dry_density);
    case 3: return fall_speeds::hail_mass_weighted(mixing_ratio, dry_density);
    default: return 0.0;
    }
}

void LinScheme::compute_tendencies(
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
)
{
    int NR = p.size_r();
    if (NR == 0) return;
    if (require_native_dispatch() && !lin_parity_replay_ && compute_runtime() == nullptr)
        throw std::runtime_error("native Lin execution requires an explicit compute owner");
    const std::array<Field3D*, 8> targets{
        &dtheta_dt, &dqv_dt, &dqc_dt, &dqr_dt, &dqi_dt, &dqs_dt, &dqg_dt, &dqh_dt};
    bool native_atomic_outputs = require_native_dispatch() && !lin_parity_replay_ && !lin_native_staging_ &&
        compute_runtime() != nullptr && compute_runtime()->active_kind != ComputeBackendKind::Cpu &&
        !compute_runtime()->config.validate_parity;
    if (native_atomic_outputs)
    {
        auto* backend = compute_runtime()->backend.get();
        auto* kernels = backend != nullptr ? backend->microphysics_kernels() : nullptr;
        native_atomic_outputs = kernels != nullptr && kernels->supports_atomic_lin_fused();
        const std::array<const Field3D*, 10> sources{&p, &theta, &rho_d, &qv, &qc, &qr, &qi, &qs, &qg, &qh};
        for (std::size_t i = 0; i < targets.size(); ++i)
        {
            native_atomic_outputs &= targets[i]->size_r() == NR &&
                targets[i]->size_th() == p.size_th() && targets[i]->size_z() == p.size_z();
            for (const auto* source : sources) native_atomic_outputs &= targets[i] != source;
            for (std::size_t j = 0; j < i; ++j) native_atomic_outputs &= targets[i] != targets[j];
        }
    }
    if (require_native_dispatch() && !lin_parity_replay_ &&
        !lin_native_staging_ && !native_atomic_outputs)
    {
        lin_native_staging_ = true;
        try
        {
            compute_tendencies(
                p, theta, rho_d, qv, qc, qr, qi, qs, qg, qh, dt,
                native_tendency_staging_[0], native_tendency_staging_[1],
                native_tendency_staging_[2], native_tendency_staging_[3],
                native_tendency_staging_[4], native_tendency_staging_[5],
                native_tendency_staging_[6], native_tendency_staging_[7]);
        }
        catch (...)
        {
            lin_native_staging_ = false;
            throw;
        }
        lin_native_staging_ = false;
        std::swap(dtheta_dt, native_tendency_staging_[0]);
        std::swap(dqv_dt, native_tendency_staging_[1]);
        std::swap(dqc_dt, native_tendency_staging_[2]);
        std::swap(dqr_dt, native_tendency_staging_[3]);
        std::swap(dqi_dt, native_tendency_staging_[4]);
        std::swap(dqs_dt, native_tendency_staging_[5]);
        std::swap(dqg_dt, native_tendency_staging_[6]);
        std::swap(dqh_dt, native_tendency_staging_[7]);
        return;
    }

    if (native_atomic_outputs)
        for (auto& field : native_tendency_staging_)
            if (!field.empty()) field = Field3D{};

    require_stable_sedimentation(rho_d, {&qr, &qs, &qg, &qh}, dt);

    int NTH = p.size_th();
    int NZ = p.size_z();

    const PhysicalCellDomain configured_domain = physical_cell_domain(
        NR, NTH, NZ);
    const PhysicalCellDomain domain = configured_domain;

    // GPU path (GPU audit Phase 3.3): the fused pointwise shader runs
    // the whole process chain INCLUDING the theta->temperature conversion
    // and the Exner back-conversion, so no temperature scratch field is
    // built; sedimentation follows as a per-column dispatch. Skipped during
    // a parity replay, which must take the CPU path below on the same
    // inputs (the inputs are pure - tendencies are the only outputs - so
    // the replay needs no state snapshots).
    if (compute_runtime() != nullptr && compute_runtime()->active_kind != ComputeBackendKind::Cpu &&
        !lin_parity_replay_)
    {
        // The native pointwise operator overwrites all cells, including halos.
        for (auto* field : {&dtheta_dt, &dqv_dt, &dqc_dt, &dqr_dt,
                            &dqi_dt, &dqs_dt, &dqg_dt, &dqh_dt})
            if (field->size_r() != NR || field->size_th() != NTH || field->size_z() != NZ)
                field->resize(NR, NTH, NZ);
        const double inv_dt_d = (dt > 0.0) ? 1.0 / dt : 0.0;
        const float  Lv_cp_f  = static_cast<float>(microphysics_constants::L_v / microphysics_constants::cp);
        const float  Lf_cp_f  = static_cast<float>(microphysics_constants::L_f / microphysics_constants::cp);
        const float  Ls_cp_f  = static_cast<float>(microphysics_constants::L_s / microphysics_constants::cp);

        const bool parity_sample =
            parity_runtime() != nullptr &&
            nimbus::parity::sample(*parity_runtime(), compute_runtime()->config.validate_parity, "lin_microphysics");

        const double             dz_local = vertical_spacing_m();
        const float              vt_cap_f = std::numeric_limits<float>::max();
        LinPointwiseDispatch     pointwise_dispatch{theta.data(),
                                                    p.data(),
                                                    qv.data(),
                                                    qc.data(),
                                                    qr.data(),
                                                    qi.data(),
                                                    qs.data(),
                                                    qg.data(),
                                                    qh.data(),
                                                    dtheta_dt.data(),
                                                    dqv_dt.data(),
                                                    dqc_dt.data(),
                                                    dqr_dt.data(),
                                                    dqi_dt.data(),
                                                    dqs_dt.data(),
                                                    dqg_dt.data(),
                                                    dqh_dt.data(),
                                                    NR,
                                                    NTH,
                                                    NZ,
                                                    static_cast<float>(dt),
                                                    static_cast<float>(inv_dt_d),
                                                    static_cast<float>(qc0_),
                                                    static_cast<float>(c_auto_),
                                                    static_cast<float>(c_accr_),
                                                    static_cast<float>(c_evap_),
                                                    static_cast<float>(c_ihom_),
                                                    static_cast<float>(c_rime_),
                                                    static_cast<float>(c_agg_),
                                                    static_cast<float>(c_melt_s_),
                                                    static_cast<float>(c_melt_g_),
                                                    static_cast<float>(c_melt_h_),
                                                    static_cast<float>(c_subl_),
                                                    Lv_cp_f,
                                                    Lf_cp_f,
                                                    Ls_cp_f};
        LinSedimentationDispatch sedimentation_dispatch{qr.data(),
                                                        qs.data(),
                                                        qg.data(),
                                                        qh.data(),
                                                        rho_d.data(),
                                                        dqr_dt.data(),
                                                        dqs_dt.data(),
                                                        dqg_dt.data(),
                                                        dqh_dt.data(),
                                                        NR,
                                                        NTH,
                                                        NZ,
                                                        static_cast<float>(dz_local),
                                                        vt_cap_f};

        std::vector<float> layer_thickness;
        if (has_nonuniform_vertical_spacing())
        {
            layer_thickness.resize(rho_d.size());
            for (int i = domain.i_begin; i < domain.i_end; ++i)
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                        layer_thickness[(std::size_t(i) * NTH + j) * NZ + k] = float(vertical_spacing_m(i, j, k));
            sedimentation_dispatch.layer_thickness_m = layer_thickness.data();
        }
        pointwise_dispatch.i_begin = sedimentation_dispatch.i_begin = domain.i_begin;
        pointwise_dispatch.i_end = sedimentation_dispatch.i_end = domain.i_end;
        pointwise_dispatch.j_begin = sedimentation_dispatch.j_begin = domain.j_begin;
        pointwise_dispatch.j_end = sedimentation_dispatch.j_end = domain.j_end;
        pointwise_dispatch.k_begin = sedimentation_dispatch.k_begin = domain.k_begin;
        pointwise_dispatch.k_end = sedimentation_dispatch.k_end = domain.k_end;
        bool gpu_chain_ok = dispatch_lin_fused_backend(*compute_runtime(), pointwise_dispatch, sedimentation_dispatch);
        if (native_atomic_outputs && !gpu_chain_ok)
            throw std::runtime_error("native Lin fused transaction failed before output commit");
        if (!gpu_chain_ok && dispatch_lin_pointwise_backend(*compute_runtime(), pointwise_dispatch))
        {
            if (!dispatch_lin_sedimentation_backend(*compute_runtime(), sedimentation_dispatch))
            {
                if (require_native_dispatch())
                {
                    throw std::runtime_error("qualified accelerator Lin requires the native "
                                             "sedimentation operator");
                }
                compute_sedimentation(rho_d, qr, qs, qg, qh, dqr_dt, dqs_dt, dqg_dt, dqh_dt, dt);
            }
            gpu_chain_ok = true;
        }

        if (gpu_chain_ok)
        {
            if (parity_sample)
            {
                // Parity replay (GPU audit Phase 1): rerun the full CPU
                // path into scratch tendencies and compare; the GPU result
                // stays production.
                Field3D r_dtheta, r_dqv, r_dqc, r_dqr, r_dqi, r_dqs, r_dqg, r_dqh;
                lin_parity_replay_ = true;
                try
                {
                    compute_tendencies(p,
                                       theta,
                                       rho_d,
                                       qv,
                                       qc,
                                       qr,
                                       qi,
                                       qs,
                                       qg,
                                       qh,
                                       dt,
                                       r_dtheta,
                                       r_dqv,
                                       r_dqc,
                                       r_dqr,
                                       r_dqi,
                                       r_dqs,
                                       r_dqg,
                                       r_dqh);
                }
                catch (...)
                {
                    lin_parity_replay_ = false;
                    throw;
                }
                lin_parity_replay_ = false;

                using nimbus::parity::Tier;
                const char* kn = "lin_microphysics";
                nimbus::parity::check_field(*parity_runtime(),
                                            kn,
                                            "dtheta_dt",
                                            r_dtheta.data(),
                                            dtheta_dt.data(),
                                            NR,
                                            NTH,
                                            NZ,
                                            Tier::MicrophysicsHeating,
                                            nimbus::parity::ComparisonDomain::StrictInterior);
                nimbus::parity::check_field(*parity_runtime(),
                                            kn,
                                            "dqv_dt",
                                            r_dqv.data(),
                                            dqv_dt.data(),
                                            NR,
                                            NTH,
                                            NZ,
                                            Tier::Microphysics,
                                            nimbus::parity::ComparisonDomain::StrictInterior);
                nimbus::parity::check_field(*parity_runtime(), kn, "dqc_dt",
                    r_dqc.data(), dqc_dt.data(), NR, NTH, NZ, Tier::Microphysics, nimbus::parity::ComparisonDomain::StrictInterior);
                nimbus::parity::check_field(*parity_runtime(), kn, "dqr_dt",
                    r_dqr.data(), dqr_dt.data(), NR, NTH, NZ, Tier::Microphysics, nimbus::parity::ComparisonDomain::StrictInterior);
                nimbus::parity::check_field(*parity_runtime(), kn, "dqi_dt",
                    r_dqi.data(), dqi_dt.data(), NR, NTH, NZ, Tier::Microphysics, nimbus::parity::ComparisonDomain::StrictInterior);
                nimbus::parity::check_field(*parity_runtime(), kn, "dqs_dt",
                    r_dqs.data(), dqs_dt.data(), NR, NTH, NZ, Tier::Microphysics, nimbus::parity::ComparisonDomain::StrictInterior);
                nimbus::parity::check_field(*parity_runtime(), kn, "dqg_dt",
                    r_dqg.data(), dqg_dt.data(), NR, NTH, NZ, Tier::Microphysics, nimbus::parity::ComparisonDomain::StrictInterior);
                nimbus::parity::check_field(*parity_runtime(), kn, "dqh_dt",
                    r_dqh.data(), dqh_dt.data(), NR, NTH, NZ, Tier::Microphysics, nimbus::parity::ComparisonDomain::StrictInterior);
            }
            return;
        }

        if (require_native_dispatch())
        {
            throw std::runtime_error(
                "qualified accelerator Lin requires native pointwise and "
                "sedimentation operators");
        }
    }
    else if (require_native_dispatch() && !lin_parity_replay_)
    {
        throw std::runtime_error(
            "qualified accelerator Lin requires an explicit compute owner");
    }

    init_tendency_fields(NR, NTH, NZ, dtheta_dt, dqv_dt, dqc_dt, dqr_dt,
                         dqi_dt, dqs_dt, dqg_dt, dqh_dt);
    const double inv_dt = (dt > 0.0) ? 1.0 / dt : 0.0;
    const double Lv_cp = microphysics_constants::L_v / microphysics_constants::cp;
    const double Lf_cp = microphysics_constants::L_f / microphysics_constants::cp;
    const double Ls_cp = microphysics_constants::L_s / microphysics_constants::cp;
    const double T0 = microphysics_constants::T0;

    // Point-wise processes on per-cell working copies. Every conversion is
    // mass-limited via transfer() -- a process can never move more mass than the
    // source currently holds -- so the scheme is conservative by construction:
    // no species goes negative and no water is manufactured by the downstream
    // non-negativity clamp (the prior cause of the condensate runaway). Latent
    // heat is released ONLY for genuine phase changes (zero for the
    // liquid->liquid autoconversion/accretion and the ice->ice aggregation).
    #pragma omp parallel for collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                // Preserve the historical fp32 workspace rounding without
                // materializing a full temperature volume.
                const double T = static_cast<double>(static_cast<float>(
                    thermodynamics::theta_to_temperature(
                        static_cast<double>(theta[i][j][k]),
                        static_cast<double>(p[i][j][k]))));
                const double P = static_cast<double>(p[i][j][k]);
                const double T_safe = std::max(T, 1.0);
                const double Tc = T - T0;
                // Local dry-air density for the spectrum integrals
                // (M9/M12); the shader mirrors this exactly.
                const double rho_air =
                    P / (microphysics_constants::R_d * T_safe);

                double wqv = static_cast<double>(qv[i][j][k]);
                double wqc = static_cast<double>(qc[i][j][k]);
                double wqr = static_cast<double>(qr[i][j][k]);
                double wqi = static_cast<double>(qi[i][j][k]);
                double wqs = static_cast<double>(qs[i][j][k]);
                double wqg = static_cast<double>(qg[i][j][k]);
                double wqh = static_cast<double>(qh[i][j][k]);
                double dT_sum = 0.0;  // accumulated temperature change (K)

                // Move `amount` (kg/kg) from src to dst, capped at the available
                // src mass; release latent heat L_over_cp per unit (signed).
                // dT_sum is a TEMPERATURE increment: dT = (L/cp) dq. The
                // Exner conversion at the end of compute_tendencies turns it
                // into dtheta. (SCIENCE.md C3: a spurious /T here made
                // every heating term ~T x too weak - the /T belongs only to
                // a d(ln theta) formulation, which this is not.)
                auto transfer = [&](double& src, double& dst, double amount, double L_over_cp)
                {
                    if (amount > src) amount = src;
                    if (amount <= 0.0) return;
                    src -= amount;
                    dst += amount;
                    dT_sum += L_over_cp * amount;
                };

                // --- Saturation adjustment (qv <-> qc, latent L_v) ---
                // Implicit form : divide the raw
                // excess/deficit by 1 + gamma, gamma = L^2 qvs/(cp Rv T^2),
                // accounting for the qvs shift the heating itself causes.
                const double qvsat_w =
                    thermodynamics::saturation_mixing_ratio_water(static_cast<float>(T), static_cast<float>(P));
                const double gamma_sat =
                    (microphysics_constants::L_v * microphysics_constants::L_v * qvsat_w)
                    / (microphysics_constants::cp * microphysics_constants::R_v
                       * std::max(T_safe * T_safe, 1.0));
                if (wqv > qvsat_w)
                {
                    transfer(wqv, wqc, (wqv - qvsat_w) / (1.0 + gamma_sat), Lv_cp);
                }
                else if (wqc > 0.0)
                {
                    transfer(wqc, wqv, (qvsat_w - wqv) / (1.0 + gamma_sat), -Lv_cp);
                }

                // --- Warm rain (liquid->liquid: NO latent heat) ---
                if (wqc > qc0_)
                {
                    transfer(wqc, wqr, c_auto_ * (wqc - qc0_) * dt, 0.0);
                }
                if (wqc > 0.0 && wqr > 0.0)
                {
                    transfer(wqc, wqr, c_accr_ * wqc * wqr * dt, 0.0);
                }
                if (wqr > 0.0)
                {
                    const double RH = (wqv > 0.0) ? wqv / std::max(qvsat_w, 1.0e-12) : 0.0;
                    if (RH < 1.0)
                    {
                        transfer(wqr, wqv, c_evap_ * (1.0 - RH) * wqr * dt, -Lv_cp);
                    }
                }

                // --- Ice / mixed phase ---
                if (T < T0)
                {
                    // Homogeneous freezing of cloud -> ice (release L_f).
                    if (wqc > 0.0)
                    {
                        // Ramp with supercooling toward instantaneous
                        // homogeneous freezing at T0 - 38 (SCIENCE.md
                        // M8; the transfer is mass-limited regardless).
                        const double supercooling = T0 - T;
                        const double freeze_frac_per_s =
                            (supercooling
                             >= microphysics_constants::homogeneous_freezing_dT)
                                ? 1.0
                                : c_ihom_ * (supercooling
                                    / microphysics_constants::homogeneous_freezing_dT);
                        transfer(wqc, wqi, freeze_frac_per_s * wqc * dt, Lf_cp);
                    }
                    // Bigg stochastic freezing of supercooled rain -> hail
                    // embryos (SCIENCE.md M9; LFO83 routes frozen
                    // drops to the 913-density class). This is the hail
                    // SOURCE - qh previously could only melt or fall.
                    if (wqr > 0.0)
                    {
                        const double pgfr = fall_speeds::rain_bigg_freezing_rate(
                            wqr, rho_air, T0 - T);
                        transfer(wqr, wqh, pgfr * dt, Lf_cp);
                    }
                    const double qvsat_i = thermodynamics::saturation_mixing_ratio_ice(
                        static_cast<float>(T), static_cast<float>(P));
                    // Vapor deposition onto ice (release L_s).
                    if (wqi > 0.0 && wqv > qvsat_i)
                    {
                        transfer(wqv, wqi, c_subl_ * (wqv - qvsat_i) * dt, Ls_cp);
                    }
                    // Riming: ice collects cloud water -> graupel (release L_f).
                    // FIX: bulk collection rate c_rime*E*qc*qi. The old form
                    // (pi/4 * 0.1 * E * Ni * qc) used the ice NUMBER concentration
                    // Ni~1e3, making the rate ~1000x too large -> graupel runaway.
                    if (wqc > 0.0 && wqi > 0.0)
                    {
                        const double E_ac = collection_efficiency(Tc);
                        transfer(wqc, wqg, c_rime_ * E_ac * wqc * wqi * dt, Lf_cp);
                    }
                    // Snow rimes supercooled cloud (SCIENCE.md M9;
                    // LFO83 PSACW continuous collection - light riming
                    // grows snow in place, releasing L_f).
                    if (wqc > 0.0 && wqs > 0.0)
                    {
                        const double psacw_per_qc =
                            fall_speeds::snow_cloud_collection_rate(wqs, rho_air);
                        transfer(wqc, wqs, psacw_per_qc * wqc * dt, Lf_cp);
                    }
                    // Lin83 ice -> snow autoconversion/aggregation (no latent:
                    // ice<->ice): threshold-gated and temperature-dependent. The
                    // old extra c_auto*qi^2 term had no Lin83 analog and
                    // double-counted ice->snow below the threshold (science audit)
                    // - removed; ice->snow now flows only through this form.
                    const double q_I0 = 1.0e-3;  // Lin83 ice autoconversion threshold (kg/kg)
                    if (wqi > q_I0)
                    {
                        transfer(wqi, wqs, c_agg_ * (wqi - q_I0) * std::exp(0.025 * Tc) * dt, 0.0);
                    }
                    // Sublimation when subsaturated wrt ice (absorb L_s).
                    const double RH_i = (wqv > 0.0) ? wqv / std::max(qvsat_i, 1.0e-12) : 0.0;
                    if (RH_i < 1.0)
                    {
                        if (wqi > 0.0)
                        {
                            transfer(wqi, wqv, c_subl_ * (1.0 - RH_i) * wqi * dt, -Ls_cp);
                        }
                        if (wqs > 0.0)
                        {
                            transfer(wqs, wqv, c_subl_ * (1.0 - RH_i) * wqs * dt, -Ls_cp);
                        }
                    }
                }
                else
                {
                    // --- Melting (T > T0): snow/graupel/hail -> rain (absorb
                    // L_f). Rate scales with the warm excess Tc and per-species
                    // heat-transfer coefficients (SCIENCE.md M10): snow
                    // clears within the bright band, large hail survives to
                    // the surface.
                    if (wqs > 0.0) transfer(wqs, wqr, c_melt_s_ * Tc * wqs * dt, -Lf_cp);
                    if (wqg > 0.0) transfer(wqg, wqr, c_melt_g_ * Tc * wqg * dt, -Lf_cp);
                    if (wqh > 0.0) transfer(wqh, wqr, c_melt_h_ * Tc * wqh * dt, -Lf_cp);
                }

                // Net per-species tendencies (sum is zero within the cell:
                // conservative by construction).
                dqv_dt[i][j][k] = static_cast<float>((wqv - static_cast<double>(qv[i][j][k])) * inv_dt);
                dqc_dt[i][j][k] = static_cast<float>((wqc - static_cast<double>(qc[i][j][k])) * inv_dt);
                dqr_dt[i][j][k] = static_cast<float>((wqr - static_cast<double>(qr[i][j][k])) * inv_dt);
                dqi_dt[i][j][k] = static_cast<float>((wqi - static_cast<double>(qi[i][j][k])) * inv_dt);
                dqs_dt[i][j][k] = static_cast<float>((wqs - static_cast<double>(qs[i][j][k])) * inv_dt);
                dqg_dt[i][j][k] = static_cast<float>((wqg - static_cast<double>(qg[i][j][k])) * inv_dt);
                dqh_dt[i][j][k] = static_cast<float>((wqh - static_cast<double>(qh[i][j][k])) * inv_dt);
                dtheta_dt[i][j][k] = static_cast<float>(dT_sum * inv_dt);
            }
        }
    }

    // Sedimentation (inter-cell vertical flux). CFL-stable at the realistic
    // hydrometeor amounts the conservative local step now produces.
    compute_sedimentation(rho_d, qr, qs, qg, qh,
                          dqr_dt, dqs_dt, dqg_dt, dqh_dt, dt);

    // Temperature tendency -> potential-temperature tendency (Exner).
    #pragma omp parallel for collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                dtheta_dt[i][j][k] = static_cast<float>(
                    thermodynamics::temperature_tendency_to_theta(
                        static_cast<float>(dtheta_dt[i][j][k]),
                        static_cast<float>(p[i][j][k])));
            }
        }
    }
}


/**
 * @brief Computes the sedimentation for the Lin scheme.
 */
void LinScheme::compute_sedimentation(
    const Field3D& rho_d,
    const Field3D& qr,
    const Field3D& qs,
    const Field3D& qg,
    const Field3D& qh,
    Field3D& dqr_dt,
    Field3D& dqs_dt,
    Field3D& dqg_dt,
    Field3D& dqh_dt,
    double dt_micro
)
{
    int NR = qr.size_r();
    int NTH = qr.size_th();
    int NZ = qr.size_z();
    const PhysicalCellDomain configured_domain = physical_cell_domain(
        NR, NTH, NZ);
    const PhysicalCellDomain domain = configured_domain;

    // sedimentation in stratified air.
    // (1) Fall speeds carry the Foote-du Toit density correction
    //     sqrt(rho_ref / rho_d(k)): the power-law coefficients are
    //     surface-calibrated and falling hydrometeors speed up
    //     aloft (~2.7x at a 16 km anvil).
    // (2) The receiving cell gains mixing ratio weighted by
    //     rho_d(k)/rho_d(k-1): the FALLING MASS rho_d q Vt is what
    //     conserves; without the weight each level of fall
    //     multiplied the water mass by the density ratio (~7x
    //     integrated from 16 km to the surface).
    // The caller admits the timestep against these physical fall speeds.
    // rho_ref is the fixed near-surface reference used by the terminal-
    // velocity parameterization; transport weighting uses live rho_d.
    constexpr double rho_ref = fall_speeds::kFallbackAirDensity;
    #pragma omp parallel for collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double dz_donor = vertical_spacing_m(i, j, k);
                const double dz_receiver = k > domain.k_begin
                    ? vertical_spacing_m(i, j, k - 1)
                    : dz_donor;
                // Sedimentation flux falls one cell down (k -> k-1). At the
                // physical surface the hydrometeor leaves the domain as
                // precipitation -- there is no receiver cell, so the flux is
                // a pure sink. The previous storage-index guard skipped k=0 entirely,
                // so surface hydrometeors were never removed and accumulated
                // (re-evaporating into vapor and feeding a condensate loop).
                const double rho_d_k =
                    static_cast<double>(rho_d[i][j][k]);
                const double rho_d_km = (k > domain.k_begin)
                    ? static_cast<double>(rho_d[i][j][k - 1])
                    : rho_d_k;
                const double fall_factor = std::sqrt(rho_ref / rho_d_k);
                const double recv_weight = rho_d_k / rho_d_km;
                const double rho_air_k = rho_d_k;

                // gamma-integrated LFO83 speeds.
                // Rain/snow carry the Foote-du Toit altitude factor; the
                // Wisner graupel/hail forms have the air density explicit,
                // so adding it would double-count.
                float qr_val = static_cast<float>(qr[i][j][k]);
                if (qr_val > 0.0f)
                {
                    const double Vt = fall_speeds::rain_mass_weighted(qr_val, rho_air_k) * fall_factor;
                    float sed_flux = static_cast<float>(Vt) * qr_val;
                    dqr_dt[i][j][k] -= sed_flux / dz_donor;
                    if (k > domain.k_begin) dqr_dt[i][j][k - 1] += static_cast<float>(
                        recv_weight * sed_flux / dz_receiver);
                }

                float qs_val = static_cast<float>(qs[i][j][k]);
                if (qs_val > 0.0f)
                {
                    const double Vt = fall_speeds::snow_mass_weighted(qs_val, rho_air_k) * fall_factor;
                    float sed_flux = static_cast<float>(Vt) * qs_val;
                    dqs_dt[i][j][k] -= sed_flux / dz_donor;
                    if (k > domain.k_begin) dqs_dt[i][j][k - 1] += static_cast<float>(
                        recv_weight * sed_flux / dz_receiver);
                }

                float qg_val = static_cast<float>(qg[i][j][k]);
                if (qg_val > 0.0f)
                {
                    const double Vt = fall_speeds::graupel_mass_weighted(qg_val, rho_air_k);
                    float sed_flux = static_cast<float>(Vt) * qg_val;
                    dqg_dt[i][j][k] -= sed_flux / dz_donor;
                    if (k > domain.k_begin) dqg_dt[i][j][k - 1] += static_cast<float>(
                        recv_weight * sed_flux / dz_receiver);
                }

                float qh_val = static_cast<float>(qh[i][j][k]);
                if (qh_val > 0.0f)
                {
                    const double Vt = fall_speeds::hail_mass_weighted(qh_val, rho_air_k);
                    float sed_flux = static_cast<float>(Vt) * qh_val;
                    dqh_dt[i][j][k] -= sed_flux / dz_donor;
                    if (k > domain.k_begin) dqh_dt[i][j][k - 1] += static_cast<float>(
                        recv_weight * sed_flux / dz_receiver);
                }
            }
        }
    }
}


double LinScheme::bottom_sedimentation_sink_rate(
    const Field3D& rho_d,
    const Field3D& qr, const Field3D& qs,
    const Field3D& qg, const Field3D& qh,
    int i, int j, double dt_micro) const
{
    const PhysicalCellDomain domain = physical_cell_domain(
        qr.size_r(), qr.size_th(), qr.size_z());
    const int k_surface = domain.k_begin;
    // Mirrors compute_sedimentation at the physical surface exactly: same speed laws,
    // same fp32 flux arithmetic and the same live-density
    // fall-speed correction.
    const double dz_local = vertical_spacing_m(i, j, k_surface);
    const double rho_air_0 = static_cast<double>(rho_d[i][j][k_surface]);
    const double fall_factor = std::sqrt(
        fall_speeds::kFallbackAirDensity / rho_air_0);
    double sink = 0.0;
    const float qr_val = qr[i][j][k_surface];
    if (qr_val > 0.0f)
    {
        const double Vt = fall_speeds::rain_mass_weighted(qr_val, rho_air_0) * fall_factor;
        sink += static_cast<double>(static_cast<float>(Vt) * qr_val)
                / dz_local;
    }
    const float qs_val = qs[i][j][k_surface];
    if (qs_val > 0.0f)
    {
        const double Vt = fall_speeds::snow_mass_weighted(qs_val, rho_air_0) * fall_factor;
        sink += static_cast<double>(static_cast<float>(Vt) * qs_val)
                / dz_local;
    }
    const float qg_val = qg[i][j][k_surface];
    if (qg_val > 0.0f)
    {
        const double Vt = fall_speeds::graupel_mass_weighted(qg_val, rho_air_0);
        sink += static_cast<double>(static_cast<float>(Vt) * qg_val)
                / dz_local;
    }
    const float qh_val = qh[i][j][k_surface];
    if (qh_val > 0.0f)
    {
        const double Vt = fall_speeds::hail_mass_weighted(qh_val, rho_air_0);
        sink += static_cast<double>(static_cast<float>(Vt) * qh_val)
                / dz_local;
    }
    return sink;
}

/**
 * @brief Computes the collection efficiency for the Lin scheme.
 */
double LinScheme::collection_efficiency(double T_celsius)
{
    if (T_celsius <= -5.0)
    {
        return 1.0;
    }
    else if (T_celsius >= 0.0)
    {
        return 0.0;
    }
    else
    {
        return std::clamp((-T_celsius) / 5.0, 0.0, 1.0);
    }
}

// ice_threshold_temperature removed (science audit): it was a misnamed stub
// returning the constant Lin83 q_I0 = 1e-3 (its qi argument and the
// exp(0.1*0.0) were dead). The constant is now inlined at the aggregation site.

} // namespace tmv
