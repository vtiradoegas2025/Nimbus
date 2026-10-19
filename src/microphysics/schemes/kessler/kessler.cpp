// Three-species Kessler warm-rain process rates.

#include "kessler.hpp"
#include "core/field/field_pool.hpp"
#include "microphysics/base/fall_speeds.hpp"
#include "compute/compute_backend.hpp"
#include "compute/compute_kernel_template.hpp"
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
 * @brief The constructor for the KesslerScheme class.
 */
KesslerScheme::KesslerScheme(
    double qc0, double c_auto, double c_accr, double evaporation_efficiency,
    double Vt_max
) :
    qc0_(qc0), c_auto_(c_auto), c_accr_(c_accr),
    evaporation_efficiency_(evaporation_efficiency),
    Vt_max_(Vt_max) {}

void KesslerScheme::configure(const MicrophysicsConfig& config)
{
    qc0_ = config.shared.cloud_water_threshold_kgkg;
    c_auto_ = config.shared.autoconversion_rate_s_inv;
    c_accr_ = config.kessler.accretion_coefficient;
    evaporation_efficiency_ = config.kessler.evaporation_efficiency;
}

double KesslerScheme::explicit_terminal_velocity_mps(
    std::size_t species, double mixing_ratio, double dry_density) const
{
    return species == 0 ? std::min(Vt_max_,
        fall_speeds::kessler_rain(mixing_ratio, dry_density) *
            std::sqrt(fall_speeds::kFallbackAirDensity / dry_density)) : 0.0;
}

PortableSchemeState KesslerScheme::portable_state() const
{
    PortableSchemeState state;
    state.version = 2;
    state.reals = {
        qc0_, c_auto_, c_accr_, evaporation_efficiency_, Vt_max_};
    return state;
}

MicrophysicsParameterIdentity KesslerScheme::parameter_identity() const noexcept
{
    MicrophysicsParameterIdentity identity;
    identity.version = 2;
    identity.real_count = 5;
    identity.reals = {
        qc0_, c_auto_, c_accr_, evaporation_efficiency_, Vt_max_};
    return identity;
}

bool KesslerScheme::restore_portable_state(const PortableSchemeState& state,
                                           std::string& error)
{
    const bool current_layout =
        state.version == 2 && state.reals.size() == 5;
    const bool legacy_layout =
        state.version == 1 && state.reals.size() == 12;
    if ((!current_layout && !legacy_layout) || !state.fields.empty() ||
        !state.integers.empty())
    {
        error = "invalid Kessler portable-state layout";
        return false;
    }
    for (double value : state.reals)
    {
        if (!std::isfinite(value) || value < 0.0)
        {
            error = "invalid Kessler portable-state parameter";
            return false;
        }
    }
    std::size_t i = 0;
    qc0_ = state.reals[i++];
    c_auto_ = state.reals[i++];
    c_accr_ = state.reals[i++];
    evaporation_efficiency_ = legacy_layout ? 1.0 : state.reals[i++];
    // v1 carried seven hidden frozen-species parameters. They are ignored
    // deliberately: Kessler is now the advertised qv/qc/qr warm-rain scheme.
    Vt_max_ = legacy_layout ? state.reals[9] : state.reals[i];
    error.clear();
    return true;
}


/**
 * @brief Computes the tendencies for the Kessler scheme.
 */
void KesslerScheme::compute_tendencies(
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
    (void)qi;
    (void)qs;
    (void)qg;
    (void)qh;
    int NR = p.size_r();
    if (NR == 0) return;
    int NTH = p.size_th();
    if (NTH == 0) return;
    int NZ = p.size_z();
    require_stable_sedimentation(rho_d, {&qr, nullptr, nullptr, nullptr}, dt);
    const PhysicalCellDomain domain = physical_cell_domain(NR, NTH, NZ);

    auto& pool = FieldPool::instance();
    auto temp_guard = pool.scoped_acquire_for_overwrite(NR, NTH, NZ);
    Field3D& temperature = temp_guard.field;

    init_tendency_fields(NR, NTH, NZ, dtheta_dt, dqv_dt, dqc_dt, dqr_dt,
                         dqi_dt, dqs_dt, dqg_dt, dqh_dt);
    if (!(dt > 0.0) || !std::isfinite(dt)) return;

    // Saturation adjustment: convert supersaturated vapor to cloud water
    // before processing warm-rain physics. Without this, Kessler only acts
    // on existing hydrometeors and never initiates condensation. We operate
    // on copies so the original qv/qc are preserved and the adjustment
    // appears as tendencies (consistent with Lin/Thompson).
    auto qv_adj_guard = pool.scoped_acquire_for_overwrite(NR, NTH, NZ);
    auto qc_adj_guard = pool.scoped_acquire_for_overwrite(NR, NTH, NZ);
    Field3D& qv_adj = qv_adj_guard.field;
    Field3D& qc_adj = qc_adj_guard.field;
    const bool gpu_pointwise_candidate =
        compute_runtime() != nullptr && active_compute_backend_supports(*compute_runtime(),
            MicrophysicsKernelOp::KesslerPointwise);
    if (require_native_dispatch() && !gpu_pointwise_candidate)
    {
        throw std::runtime_error(
            "qualified accelerator Kessler requires the native pointwise operator");
    }
    const int compute_i_begin = domain.i_begin;
    const int compute_i_end = domain.i_end;
    const int compute_j_begin = domain.j_begin;
    const int compute_j_end = domain.j_end;
    const int compute_k_begin = domain.k_begin;
    const int compute_k_end = domain.k_end;
    const double inv_dt = (dt > 0.0) ? 1.0 / dt : 0.0;
    const double Lv_cp_d =
        microphysics_constants::L_v / microphysics_constants::cp;
    #pragma omp parallel for collapse(2)
    for (int i = compute_i_begin; i < compute_i_end; ++i)
    {
        for (int j = compute_j_begin; j < compute_j_end; ++j)
        {
            for (int k = compute_k_begin; k < compute_k_end; ++k)
            {
                const float temperature_value = static_cast<float>(
                    thermodynamics::theta_to_temperature(
                        static_cast<double>(theta[i][j][k]),
                        static_cast<double>(p[i][j][k])));
                temperature[i][j][k] = temperature_value;
                double T_val = static_cast<double>(temperature_value);
                double p_val = static_cast<double>(p[i][j][k]);
                double qv_val = static_cast<double>(qv[i][j][k]);
                double qc_val = static_cast<double>(qc[i][j][k]);
                T_val = thermodynamics::saturation_adjustment(
                    T_val, p_val, qv_val, qc_val);
                temperature[i][j][k] = static_cast<float>(T_val);
                qv_adj[i][j][k] = static_cast<float>(qv_val);
                qc_adj[i][j][k] = static_cast<float>(qc_val);
                if (!gpu_pointwise_candidate)
                {
                    const double dqc_adj =
                        static_cast<double>(qc_adj[i][j][k])
                      - static_cast<double>(qc[i][j][k]);
                    if (dqc_adj != 0.0)
                    {
                        const double rate = dqc_adj * inv_dt;
                        dqc_dt[i][j][k] += static_cast<float>(rate);
                        dqv_dt[i][j][k] -= static_cast<float>(rate);
                        dtheta_dt[i][j][k] +=
                            static_cast<float>(Lv_cp_d * rate);
                    }
                }
            }
        }
    }

    // The retained device ABI still has compatibility slots for qg/qh and
    // their tendencies. The shader writes those tendencies as exact zero.
    const float Lv_cp = static_cast<float>(microphysics_constants::L_v / microphysics_constants::cp);

    bool gpu_pointwise_ok = gpu_pointwise_candidate
        && compute_runtime() != nullptr && dispatch_kessler_pointwise_backend(*compute_runtime(),
        temperature.data(), p.data(),
        qv_adj.data(), qc_adj.data(), qr.data(), rho_d.data(), qh.data(),
        dtheta_dt.data(), dqv_dt.data(), dqc_dt.data(), dqr_dt.data(),
        dqg_dt.data(), dqh_dt.data(),
        NR, NTH, NZ,
        domain.i_begin, domain.i_end,
        domain.j_begin, domain.j_end,
        domain.k_begin, domain.k_end,
        static_cast<float>(qc0_), static_cast<float>(c_auto_),
        static_cast<float>(c_accr_),
        static_cast<float>(evaporation_efficiency_),
        Lv_cp, static_cast<float>(microphysics_constants::T0),
        static_cast<float>(dt));
    if (require_native_dispatch() && !gpu_pointwise_ok)
    {
        throw std::runtime_error(
            "qualified accelerator Kessler pointwise dispatch failed");
    }

    // Commit the saturation adjustment (vapor <-> cloud) as a CONSERVING
    // tendency, INCLUDING its latent heat. This is the dominant buoyancy
    // source. Runs on BOTH backends: the GPU pointwise pass consumes the
    // adjusted fields as inputs, but the adjustment's delta against the
    // prognostic qv/qc must still be booked as tendencies + heating.
    // Before 2026-06-11 this lived on the CPU branch only, so
    // backend=gpu ran Kessler with no condensation at all
    // (SCIENCE.md C5).
    if (gpu_pointwise_candidate)
    {
        #pragma omp parallel for collapse(2)
        for (int i = compute_i_begin; i < compute_i_end; ++i)
            for (int j = compute_j_begin; j < compute_j_end; ++j)
                for (int k = compute_k_begin; k < compute_k_end; ++k)
                {
                    const double dqc_adj = static_cast<double>(qc_adj[i][j][k])
                                         - static_cast<double>(qc[i][j][k]);
                    if (dqc_adj != 0.0)
                    {
                        const double rate = dqc_adj * inv_dt;
                        dqc_dt[i][j][k] += static_cast<float>(rate);
                        dqv_dt[i][j][k] -= static_cast<float>(rate);
                        dtheta_dt[i][j][k] += static_cast<float>(Lv_cp_d * rate);
                    }
                }
    }

    if (gpu_pointwise_ok)
    {
        // Try GPU sedimentation (operates on tendencies from point-wise pass)
        const double dz_local = vertical_spacing_m();
        // SCI-024: sedimentation consumes the live prognostic dry density,
        // not the horizontally uniform initialization profile.
        const float        vt_cap_f = std::numeric_limits<float>::max();
        std::vector<float> layers;
        if (has_nonuniform_vertical_spacing())
        {
            layers.resize(qr.size());
            for (int i = 0; i < NR; ++i)
                for (int j = 0; j < NTH; ++j)
                    for (int k = 0; k < NZ; ++k)
                        layers[(std::size_t(i) * NTH + j) * NZ + k] = static_cast<float>(vertical_spacing_m(i, j, k));
        }
        bool gpu_sed_ok = compute_runtime() != nullptr &&
                          dispatch_kessler_sedimentation_backend(*compute_runtime(),
                                                                 qr.data(),
                                                                 qg.data(),
                                                                 qh.data(),
                                                                 rho_d.data(),
                                                                 dqr_dt.data(),
                                                                 dqg_dt.data(),
                                                                 dqh_dt.data(),
                                                                 NR,
                                                                 NTH,
                                                                 NZ,
                                                                 domain.i_begin,
                                                                 domain.i_end,
                                                                 domain.j_begin,
                                                                 domain.j_end,
                                                                 domain.k_begin,
                                                                 domain.k_end,
                                                                 static_cast<float>(dz_local),
                                                                 static_cast<float>(Vt_max_),
                                                                 vt_cap_f,
                                                                 static_cast<float>(dt),
                                                                 layers.empty() ? nullptr : layers.data());

        if (require_native_dispatch() && !gpu_sed_ok)
        {
            throw std::runtime_error("qualified accelerator Kessler sedimentation dispatch failed");
        }

        if (!gpu_sed_ok)
        {
            compute_sedimentation(rho_d, qr, dqr_dt, dt);
        }
    }
    else
    {
        // CPU fallback for the liquid-only pointwise process.
        #pragma omp parallel
        {
            compute_warm_rain_processes(
                temperature, p, rho_d, qv_adj, qc_adj, qr,
                dqc_dt, dqr_dt, dqv_dt, dtheta_dt, dt);
        }
        compute_sedimentation(rho_d, qr, dqr_dt, dt);
    }

    #pragma omp parallel for collapse(2)
    for (int i = compute_i_begin; i < compute_i_end; ++i)
    {
        for (int j = compute_j_begin; j < compute_j_end; ++j)
        {
            for (int k = compute_k_begin; k < compute_k_end; ++k)
            {
                dtheta_dt[i][j][k] = static_cast<float>(
                    thermodynamics::temperature_tendency_to_theta(
                        static_cast<float>(dtheta_dt[i][j][k]), static_cast<float>(p[i][j][k])
                    )
                );
            }
        }
    }
}

} // namespace tmv

/**
 * @brief Computes the warm rain processes for the Kessler scheme.
 */
void KesslerScheme::compute_warm_rain_processes(
    const Field3D& temperature,
    const Field3D& p,
    const Field3D& rho_d,
    const Field3D& qv,
    const Field3D& qc,
    const Field3D& qr,
    Field3D& dqc_dt,
    Field3D& dqr_dt,
    Field3D& dqv_dt,
    Field3D& dtheta_dt,
    double dt_micro
)
{
    int NR = temperature.size_r();
    int NTH = temperature.size_th();
    int NZ = temperature.size_z();
    const PhysicalCellDomain domain = physical_cell_domain(NR, NTH, NZ);

    // Orphaned worksharing loop; its implicit barrier preserves process order.
    #pragma omp for collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                float qc_val = static_cast<float>(qc[i][j][k]);
                float qr_val = static_cast<float>(qr[i][j][k]);
                float qv_val = static_cast<float>(qv[i][j][k]);
                float T = static_cast<float>(temperature[i][j][k]);

                // Kessler/COMMAS finite-step cloud-to-rain production:
                // implicit accretion with the canonical qr^0.875 law plus
                // threshold autoconversion. The amount form is bounded by
                // available cloud water before conversion to a tendency.
                float production_rate = 0.0f;
                if (dt_micro > 0.0 && qc_val > 0.0f)
                {
                    const double factor = 1.0 /
                        (1.0 + c_accr_ * dt_micro
                         * std::pow(std::max(0.0f, qr_val), 0.875));
                    const double amount =
                        static_cast<double>(qc_val) * (1.0 - factor)
                        + factor * c_auto_ * dt_micro
                          * std::max(
                              static_cast<double>(qc_val) - qc0_, 0.0);
                    production_rate = static_cast<float>(
                        std::clamp(amount, 0.0,
                                   static_cast<double>(qc_val)) / dt_micro);
                    dqc_dt[i][j][k] -= production_rate;
                    dqr_dt[i][j][k] += production_rate;
                }

                if (qr_val > 0.0f || production_rate > 0.0f)
                {
                    const double p_val =
                        static_cast<double>(p[i][j][k]);
                    const double rho_val =
                        static_cast<double>(rho_d[i][j][k]);
                    const double qvsat =
                        thermodynamics::saturation_mixing_ratio_water(
                            static_cast<double>(T), p_val);

                    if (qvsat > 0.0 &&
                        static_cast<double>(qv_val) < qvsat &&
                        rho_val > 0.0 && p_val > 0.0)
                    {
                        const double qr_available =
                            static_cast<double>(qr_val)
                            + static_cast<double>(production_rate) * dt_micro;
                        // WRF/COMMAS Kessler diffusion/ventilation kernel.
                        // rcgs converts rho [kg m-3] to g cm-3 per kg kg-1.
                        const double rcgs = 1.0e-3 * rho_val;
                        const double rain_content =
                            rcgs * std::max(qr_available, 0.0);
                        const double ventilation =
                            (1.6 + 124.9 * std::pow(rain_content, 0.2046))
                            * std::pow(rain_content, 0.525);
                        const double resistance =
                            2.55e8 / (p_val * qvsat) + 5.4e5;
                        const double raw_rate =
                            evaporation_efficiency_
                            * (ventilation / resistance)
                            * ((qvsat - static_cast<double>(qv_val))
                               / (rcgs * qvsat));

                        // Saturation adjustment has already evaporated all
                        // available cloud. Bound rain evaporation by the
                        // remaining implicit saturation deficit and rain mass.
                        const double gamma_sat =
                            (microphysics_constants::L_v
                             * microphysics_constants::L_v * qvsat)
                            / (microphysics_constants::cp
                               * microphysics_constants::R_v
                               * std::max(
                                   static_cast<double>(T)
                                   * static_cast<double>(T), 1.0));
                        const double saturation_bound =
                            (qvsat - static_cast<double>(qv_val))
                            / (1.0 + gamma_sat);
                        const double evap_rate = std::clamp(
                            raw_rate, 0.0,
                            std::min(
                                saturation_bound / dt_micro,
                                qr_available / dt_micro));
                        dqv_dt[i][j][k] += evap_rate;
                        dqr_dt[i][j][k] -= evap_rate;
                        dtheta_dt[i][j][k] -= microphysics_constants::L_v / microphysics_constants::cp * evap_rate;
                    }
                }
            }
        }
    }
}

/**
 * @brief Computes the sedimentation for the Kessler scheme.
 */
void KesslerScheme::compute_sedimentation(
    const Field3D& rho_d,
    const Field3D& qr,
    Field3D& dqr_dt,
    double dt_micro)
{
    int NR = qr.size_r();
    int NTH = qr.size_th();
    int NZ = qr.size_z();
    const PhysicalCellDomain domain = physical_cell_domain(NR, NTH, NZ);

    // SCIENCE.md M11 (see LinScheme::compute_sedimentation):
    // density-corrected fall speeds, mass-conserving receiver
    // weighting, and the physical rain-speed cap.
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
                // At the lowest physical cell the hydrometeor leaves the
                // domain as precipitation. Receiver transport must stop at
                // this boundary rather than entering closure storage.
                const double rho_d_k =
                    static_cast<double>(rho_d[i][j][k]);
                const double rho_d_km = (k > domain.k_begin)
                    ? static_cast<double>(rho_d[i][j][k - 1])
                    : rho_d_k;
                const double fall_factor = std::sqrt(rho_ref / rho_d_k);
                const double recv_weight = rho_d_k / rho_d_km;
                const double rho_air_k = rho_d_k;

                // SCIENCE.md M12: rain uses the classic Kessler speed
                // (Soong-Ogura 1973, as in CM1) with the altitude factor.
                // Sedimentation transports the pre-process rain field, as do
                // the other qualified bulk schemes. The later column donor
                // admission owns the finite-step combined process/fallout
                // bound. Using dqr_dt here would make the transported bottom
                // flux differ from bottom_sedimentation_sink_rate(), breaking
                // the conservative column ledger whenever conversion or
                // evaporation is active.
                const double qr_val = std::max(
                    0.0, static_cast<double>(qr[i][j][k]));
                if (qr_val > 0.0)
                {
                    double Vt = std::min(
                        fall_speeds::kessler_rain(qr_val, rho_air_k)
                            * fall_factor,
                        static_cast<double>(Vt_max_));
                    const float sed_flux =
                        static_cast<float>(Vt * qr_val);
                    dqr_dt[i][j][k] -=
                        static_cast<float>(sed_flux / dz_donor);
                    if (k > domain.k_begin) dqr_dt[i][j][k - 1] += static_cast<float>(
                        recv_weight * sed_flux / dz_receiver);
                }

            }
        }
    }
}

double KesslerScheme::bottom_sedimentation_sink_rate(
    const Field3D& rho_d,
    const Field3D& qr, const Field3D& qs,
    const Field3D& qg, const Field3D& qh,
    int i, int j, double dt_micro) const
{
    const PhysicalCellDomain domain = physical_cell_domain(
        qr.size_r(), qr.size_th(), qr.size_z());
    const int k_surface = domain.k_begin;
    // Mirrors compute_sedimentation at the lowest physical cell exactly.
    (void)qs;
    (void)qg;
    (void)qh;
    const double dz_local = vertical_spacing_m(i, j, k_surface);
    const double rho_air_0 = static_cast<double>(rho_d[i][j][k_surface]);
    const double fall_factor = std::sqrt(
        fall_speeds::kFallbackAirDensity / rho_air_0);
    double sink = 0.0;
    const float qr_val = qr[i][j][k_surface];
    if (qr_val > 0.0f)
    {
        double Vt = std::min(fall_speeds::kessler_rain(qr_val, rho_air_0)
                                 * fall_factor,
                             static_cast<double>(Vt_max_));
        sink += static_cast<double>(static_cast<float>(Vt) * qr_val)
                / dz_local;
    }
    return sink;
}
