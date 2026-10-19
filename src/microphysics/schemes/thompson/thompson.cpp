// Thompson-inspired conservative bulk microphysics.

#include "thompson.hpp"
#include "compute/backend_runtime.hpp"
#include "microphysics/schemes/thompson/aerosol_scavenging.hpp"
#include "microphysics/schemes/thompson/collection_processes.hpp"
#include "microphysics/schemes/thompson/cold_processes.hpp"
#include "microphysics/schemes/thompson/frozen_melting.hpp"
#include "microphysics/schemes/thompson/nucleation.hpp"
#include "microphysics/schemes/thompson/required_packaged_data.hpp"
#include "microphysics/schemes/thompson/moment_initialization.hpp"
#include "microphysics/schemes/thompson/thermodynamic_helpers.hpp"
#include "microphysics/schemes/thompson/warm_rain_processes.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <sstream>
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace tmv
{
namespace
{

float stored_number_tendency(double initial, double candidate, double timestep)
{
    float tendency = static_cast<float>((candidate - initial) / timestep);
    if (!std::isfinite(tendency))
    {
        throw std::overflow_error("Thompson number tendency is not representable");
    }
    if (initial + timestep * tendency < 0.0)
    {
        tendency = std::nextafter(tendency, std::numeric_limits<float>::infinity());
    }
    return tendency;
}

float projected_cloud_tendency(double initial, double content, double density, float rate, double dt)
{
    const double candidate = initial + dt * rate;
    const double normalized =
        content > 1e-12 ? thompson::normalize_cloud_number_m3(content, density * std::max(0.0, candidate)) / density
                        : std::max(0.0, candidate);
    const double rounding =
        8 * std::numeric_limits<float>::epsilon() * std::max({1.0, std::abs(candidate), std::abs(normalized)});
    return candidate < 0 || density * candidate > 1999e6 || std::abs(normalized - candidate) > rounding
               ? stored_number_tendency(initial, normalized, dt)
               : rate;
}

std::shared_ptr<const thompson::CollectionTables> packaged_collection_tables()
{
    static const auto tables = thompson::require_packaged_collection_tables();
    return tables;
}

std::shared_ptr<const thompson::ColdProcessTables> packaged_cold_tables()
{
    static const auto tables =
        thompson::require_packaged_cold_process_tables();
    return tables;
}

}

/**
 * @brief Initializes the Thompson scheme with default parameters.
 */
ThompsonScheme::ThompsonScheme(
    double qc0, double c_auto, double c_evap, double ccn_conc, double in_conc
) : qc0_(qc0), c_auto_(c_auto), c_evap_(c_evap), ccn_conc_(ccn_conc), in_conc_(in_conc) {}

void ThompsonScheme::ensure_activation_table()
{
    if (!activation_table_)
        activation_table_ = thompson::require_packaged_activation_table();
    if (!cloud_evaporation_table_)
    {
        auto candidate = std::make_shared<thompson::CloudEvaporationTable>(
            thompson::generate_cloud_evaporation_table());
        std::string error;
        if (!thompson::validate_cloud_evaporation_table(*candidate, error))
            throw std::runtime_error(error);
        cloud_evaporation_table_ = std::move(candidate);
    }
    if (!collection_tables_)
        collection_tables_ = packaged_collection_tables();
    if (!cold_process_tables_)
        cold_process_tables_ = packaged_cold_tables();
}

void ThompsonScheme::configure(const MicrophysicsConfig& config)
{
    qc0_ = config.shared.cloud_water_threshold_kgkg;
    c_auto_ = config.shared.autoconversion_rate_s_inv;
    c_evap_ = config.thompson.evaporation_rate_s_inv;
    ccn_conc_ = config.thompson.water_friendly_aerosol_surface_m3;
    in_conc_ = config.thompson.ice_friendly_aerosol_surface_m3;
    surface_sedimentation_sink_rate_ = Field3D{};
    ensure_activation_table();
}

/**
 * @brief Computes the tendencies for the Thompson scheme.
 */
void ThompsonScheme::compute_tendencies(
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
    Field3D& destination_dtheta_dt,
    Field3D& destination_dqv_dt,
    Field3D& destination_dqc_dt,
    Field3D& destination_dqr_dt,
    Field3D& destination_dqi_dt,
    Field3D& destination_dqs_dt,
    Field3D& destination_dqg_dt,
    Field3D& destination_dqh_dt
) 
{

    if (require_native_dispatch())
    {
        auto* runtime = compute_runtime();
        auto* kernels = runtime && runtime->backend ? runtime->backend->microphysics_kernels() : nullptr;
        if (!kernels || prognostic_state_.cloud_number().empty())
            throw std::runtime_error("native Thompson requires prepared state and native process ownership");
        for (auto op : {MicrophysicsKernelOp::ThompsonCloudAdjustment,
                        MicrophysicsKernelOp::ThompsonWarmRain,
                        MicrophysicsKernelOp::ThompsonDeposition,
                        MicrophysicsKernelOp::ThompsonFreezing,
                        MicrophysicsKernelOp::ThompsonCompletion,
                        MicrophysicsKernelOp::ThompsonMomentSedimentation,
                        MicrophysicsKernelOp::ThompsonCloudPhase})
            if (!kernels->supports(op))
                throw std::runtime_error("native Thompson process ownership is unavailable");
    }

    int NR = p.size_r();
    if (NR == 0) return;
    int NTH = p.size_th();
    int NZ = p.size_z();
    const PhysicalCellDomain domain = physical_cell_domain(NR, NTH, NZ);

    const bool prognostic_ready = !prognostic_state_.cloud_number().empty();
    if (prognostic_ready)
    {
        std::string error;
        if (!prognostic_state_.validate_process_state(p, domain, error))
            throw std::invalid_argument(error);
        ensure_activation_table();
    }

    const auto same_shape = [&](const Field3D& field) {
        return field.size_r() == NR && field.size_th() == NTH
            && field.size_z() == NZ;
    };
    if (!std::isfinite(dt) || dt <= 0.0
        || !same_shape(theta) || !same_shape(rho_d) || !same_shape(qv)
        || !same_shape(qc) || !same_shape(qr) || !same_shape(qi)
        || !same_shape(qs) || !same_shape(qg) || !same_shape(qh))
    {
        throw std::invalid_argument("invalid Thompson process field contract");
    }
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
        const std::size_t index =
            (static_cast<std::size_t>(i) * NTH + static_cast<std::size_t>(j))
                * NZ + static_cast<std::size_t>(k);
        if (!std::isfinite(p.data()[index]) || p.data()[index] <= 0.0f
            || !std::isfinite(theta.data()[index])
            || theta.data()[index] <= 0.0f
            || !std::isfinite(rho_d.data()[index])
            || rho_d.data()[index] <= 0.0f
            || !std::isfinite(qv.data()[index]) || qv.data()[index] < 0.0f
            || !std::isfinite(qc.data()[index]) || qc.data()[index] < 0.0f
            || !std::isfinite(qr.data()[index]) || qr.data()[index] < 0.0f
            || !std::isfinite(qi.data()[index]) || qi.data()[index] < 0.0f
            || !std::isfinite(qs.data()[index]) || qs.data()[index] < 0.0f
            || !std::isfinite(qg.data()[index]) || qg.data()[index] < 0.0f
            || !std::isfinite(qh.data()[index]) || qh.data()[index] != 0.0f)
        {
            throw std::invalid_argument("invalid Thompson process state");
        }
        const double temperature = thermodynamics::theta_to_temperature(
            theta.data()[index], p.data()[index]);
        const double saturation_water =
            thermodynamics::saturation_mixing_ratio_water(
                static_cast<float>(temperature), p.data()[index]);
        if (!std::isfinite(temperature) || temperature <= 0.0
            || !std::isfinite(saturation_water) || saturation_water < 0.0)
            throw std::invalid_argument(
                "invalid Thompson thermodynamic process state");
        if (temperature < microphysics_constants::T0)
        {
            const double saturation_ice =
                thermodynamics::saturation_mixing_ratio_ice(
                    static_cast<float>(temperature), p.data()[index]);
            if (!std::isfinite(saturation_ice) || saturation_ice < 0.0)
                throw std::invalid_argument(
                    "invalid Thompson thermodynamic process state");
        }
            }
    if (prognostic_ready)
    {
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
            const std::size_t index =
                (static_cast<std::size_t>(i) * NTH
                 + static_cast<std::size_t>(j)) * NZ
                + static_cast<std::size_t>(k);
            const double density = rho_d.data()[index];
            const double cloud_number =
                prognostic_state_.cloud_number().data()[index] * density;
            const double rain_number =
                prognostic_state_.rain_number().data()[index] * density;
            const double ice_number =
                prognostic_state_.ice_number().data()[index] * density;
            const double water_aerosol =
                prognostic_state_.water_aerosol().data()[index] * density;
            const double ice_aerosol =
                prognostic_state_.ice_aerosol().data()[index] * density;
            if (cloud_number > 1999.0e6 || rain_number > 1.0e12
                || ice_number > 1.0e12 || water_aerosol > 9999.0e6
                || ice_aerosol > 9999.0e6)
                throw std::invalid_argument(
                    "Thompson prognostic number exceeds mp=28 bounds");
                }
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                    if (!std::isfinite(vertical_velocity_at_scalar(i, j, k)))
                        throw std::invalid_argument(
                            "invalid Thompson activation velocity");
    }

    Field3D dtheta_dt, dqv_dt, dqc_dt, dqr_dt;
    Field3D dqi_dt, dqs_dt, dqg_dt, dqh_dt;
    thompson::PrognosticState staged_state = prognostic_state_;

    init_tendency_fields(NR, NTH, NZ, dtheta_dt, dqv_dt, dqc_dt, dqr_dt,
                         dqi_dt, dqs_dt, dqg_dt, dqh_dt);
    Field3D snow_fall_speed_multiplier(NR, NTH, NZ, 1.0f);
    staged_state.invalidate_transient();
    NativeCloudAdjustment  native_cloud;
    const bool             cloud_on_device =
        prognostic_ready && try_native_cloud_adjustment(p, theta, rho_d, qv, qc, dt, native_cloud);
    std::array<Field3D, 10> native_warm;
    const bool              warm_on_device =
        cloud_on_device && try_native_warm_rain(p, rho_d, qv, qc, qr, dt, native_cloud, native_warm);
    std::array<Field3D, 13> native_deposition;
    const bool              deposition_on_device =
        warm_on_device &&
        try_native_deposition(p, rho_d, qv, qi, qs, qg, dt, native_cloud, native_warm, native_deposition);
    std::array<Field3D, 13> native_freezing;
    const bool              freezing_on_device =
        warm_on_device && try_native_freezing(rho_d, qc, qr, dt, native_cloud, native_warm, native_freezing);

    const double inv_dt = (dt > 0.0) ? 1.0 / dt : 0.0;
    const double Lv_cp = microphysics_constants::L_v / microphysics_constants::cp;
    const double Lf_cp = microphysics_constants::L_f / microphysics_constants::cp;
    const double Ls_cp = microphysics_constants::L_s / microphysics_constants::cp;
    const double T0 = microphysics_constants::T0;

    const bool pointwise_native =
        freezing_on_device && deposition_on_device &&
        try_native_completion(p,
                              rho_d,
                              {&qv, &qc, &qr, &qi, &qs, &qg, &qh},
                              dt,
                              native_cloud,
                              native_warm,
                              native_deposition,
                              native_freezing,
                              staged_state,
                              {&dtheta_dt, &dqv_dt, &dqc_dt, &dqr_dt, &dqi_dt, &dqs_dt, &dqg_dt, &dqh_dt},
                              snow_fall_speed_multiplier);
    if (require_native_dispatch() && !pointwise_native)
        throw std::runtime_error("native Thompson pointwise completion failed");
    if (!pointwise_native)
    {
        std::atomic<bool>  pointwise_failed{false};
        std::exception_ptr pointwise_error;
#pragma omp parallel for collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                if (pointwise_failed.load(std::memory_order_relaxed))
                    continue;
                try
                {
                // Preserve the historical fp32 workspace rounding without
                // materializing a full temperature volume.
                const double T  = cloud_on_device
                                      ? native_cloud.temperature(i, j, k)
                                      : static_cast<double>(static_cast<float>(thermodynamics::theta_to_temperature(
                                            static_cast<double>(theta[i][j][k]), static_cast<double>(p[i][j][k]))));
                const double P = static_cast<double>(p[i][j][k]);
                const double Tc = T - T0;

                double wqv = static_cast<double>(qv[i][j][k]);
                double wqc = static_cast<double>(qc[i][j][k]);
                double wqr = static_cast<double>(qr[i][j][k]);
                double wqi = static_cast<double>(qi[i][j][k]);
                double wqs = static_cast<double>(qs[i][j][k]);
                double wqg = static_cast<double>(qg[i][j][k]);
                double wqh = static_cast<double>(qh[i][j][k]);
                double dT_sum = 0.0;

                auto transfer = [&](double& src, double& dst, double amount, double L_over_cp)
                {
                    if (amount > src) amount = src;
                    if (amount <= 0.0) return;
                    src -= amount;
                    dst += amount;
                    dT_sum += L_over_cp * amount;
                };

                // Saturation adjustment (qv <-> qc, latent L_v).
                const double qvsat_w = thermodynamics::saturation_mixing_ratio_water(
                    static_cast<float>(T), static_cast<float>(P));
                double condensed_cloud = 0.0;
                double evaporated_cloud = 0.0;
                double evaporated_rain = 0.0;
                double rain_before_evaporation = 0.0;
                if (cloud_on_device)
                {
                    wqv += double(native_cloud.vapor_change(i, j, k)) + native_cloud.vapor_change_low(i, j, k);
                    wqc -= double(native_cloud.vapor_change(i, j, k)) + native_cloud.vapor_change_low(i, j, k);
                    dT_sum +=
                        double(native_cloud.temperature_change(i, j, k)) + native_cloud.temperature_change_low(i, j, k);
                    staged_state.cloud_number_tendency()(i, j, k)  = native_cloud.cloud_number_tendency(i, j, k);
                    staged_state.water_aerosol_tendency()(i, j, k) = native_cloud.aerosol_tendency(i, j, k);
                }
                else
                {
                    const double old_cloud = wqc;
                    const auto   resolution =
                        thompson::liquid_saturation_resolution_from_theta(theta(i, j, k), float(P), float(wqv));
                    const double adjusted_temperature = resolution.condensation || resolution.evaporation
                                                            ? thermodynamics::saturation_adjustment(T, P, wqv, wqc)
                                                            : T;
                    dT_sum += adjusted_temperature - T;
                    condensed_cloud  = std::max(0.0, wqc - old_cloud);
                    evaporated_cloud = std::max(0.0, old_cloud - wqc);

                    if (prognostic_ready && evaporated_cloud > 0 && rho_d[i][j][k] * old_cloud > 1e-12 &&
                        staged_state.cloud_number()(i, j, k) <= 0)
                    {
                        std::ostringstream message;
                        message << std::setprecision(17)
                                << "resolved evaporating cloud requires its droplet population at (" << i << ',' << j
                                << ',' << k << ") in " << NR << 'x' << NTH << 'x' << NZ << ": cloud=" << old_cloud
                                << ", density=" << rho_d[i][j][k];
                        throw std::invalid_argument(message.str());
                    }
                    const double exhausted_temperature = T - Lv_cp * old_cloud;
                    const bool   exhausted_resolved    = evaporated_cloud > 0 && exhausted_temperature >= 150 &&
                                                         !thompson::liquid_saturation_resolution_from_theta(
                                                              theta(i, j, k), float(P), qv(i, j, k), float(old_cloud))
                                                              .condensation;
                    const bool   fully_evaporated =
                        evaporated_cloud > 0 && (rho_d[i][j][k] * wqc <= 1e-12 || exhausted_resolved);
                    if (fully_evaporated)
                    {
                        transfer(wqc, wqv, wqc, -Lv_cp);
                    }
                    if (prognostic_ready && fully_evaporated)
                    {
                        const float rate = stored_number_tendency(staged_state.cloud_number()(i, j, k), 0, dt);
                        staged_state.cloud_number_tendency()(i, j, k)  = rate;
                        staged_state.water_aerosol_tendency()(i, j, k) = -rate;
                    }
                    else if (prognostic_ready && condensed_cloud > 1.0e-15 && resolution.condensation)
                    {
                        const std::size_t index    = (static_cast<std::size_t>(i) * static_cast<std::size_t>(NTH) +
                                                      static_cast<std::size_t>(j)) *
                                                         static_cast<std::size_t>(NZ) +
                                                     static_cast<std::size_t>(k);
                        const auto        exchange = thompson::activate_cloud_droplets(
                            *activation_table_,
                            T,
                            vertical_velocity_at_scalar(i, j, k),
                            static_cast<double>(rho_d.data()[index]),
                            static_cast<double>(staged_state.cloud_number().data()[index]),
                            static_cast<double>(staged_state.water_aerosol().data()[index]),
                            rho_d.data()[index] * wqc);
                        staged_state.cloud_number_tendency().data()[index] =
                            static_cast<float>(exchange.hydrometeor_kg_dry * inv_dt);
                        staged_state.water_aerosol_tendency().data()[index] =
                            static_cast<float>(exchange.aerosol_kg_dry * inv_dt);
                    }
                    else if (prognostic_ready && evaporated_cloud > 1.0e-15 && resolution.evaporation &&
                             static_cast<double>(qv[i][j][k]) / qvsat_w - 1.0 < -1.0e-6)
                    {
                        const std::size_t index    = (static_cast<std::size_t>(i) * static_cast<std::size_t>(NTH) +
                                                      static_cast<std::size_t>(j)) *
                                                         static_cast<std::size_t>(NZ) +
                                                     static_cast<std::size_t>(k);
                        const double      density  = rho_d.data()[index];
                        const double      diameter = thompson::maximum_fully_evaporated_cloud_diameter(
                            T, P, density, qv.data()[index], qvsat_w, dt);
                        const auto exchange =
                            thompson::return_evaporated_cloud_number(*cloud_evaporation_table_,
                                                                     diameter,
                                                                     static_cast<double>(qc.data()[index]) * density,
                                                                     staged_state.cloud_number().data()[index],
                                                                     density);
                        staged_state.cloud_number_tendency().data()[index] =
                            static_cast<float>(exchange.cloud_number_kg_dry * inv_dt);
                        staged_state.water_aerosol_tendency().data()[index] =
                            static_cast<float>(exchange.water_aerosol_kg_dry * inv_dt);
                    }
                }

                if (warm_on_device)
                {
                    const double converted  = static_cast<double>(native_warm[0](i, j, k)) + native_warm[1](i, j, k);
                    const double evaporated = static_cast<double>(native_warm[2](i, j, k)) + native_warm[3](i, j, k);
                    transfer(wqc, wqr, converted, 0.0);
                    transfer(wqr, wqv, evaporated, 0.0);
                    dT_sum += static_cast<double>(native_warm[4](i, j, k)) + native_warm[5](i, j, k);
                    staged_state.cloud_number_tendency()(i, j, k)  = native_warm[6](i, j, k);
                    staged_state.rain_number_tendency()(i, j, k)   = native_warm[7](i, j, k);
                    staged_state.water_aerosol_tendency()(i, j, k) = native_warm[8](i, j, k);
                }
                else if (prognostic_ready)
                {
                    const std::size_t index =
                        (static_cast<std::size_t>(i) * static_cast<std::size_t>(NTH) + static_cast<std::size_t>(j)) *
                            static_cast<std::size_t>(NZ) +
                        static_cast<std::size_t>(k);
                    const double density              = rho_d.data()[index];
                    double       current_cloud_number = staged_state.cloud_number().data()[index] +
                                                        dt * staged_state.cloud_number_tendency().data()[index];
                    double       current_rain_number  = staged_state.rain_number().data()[index] +
                                                        dt * staged_state.rain_number_tendency().data()[index];
                    if (density * wqr > 1e-12)
                    {
                        current_rain_number =
                            thompson::normalize_rain_number_m3(density * wqr, density * current_rain_number) / density;
                    }
                    const auto   conversion      = thompson::warm_rain_conversion_rates(T,
                                                                                        density,
                                                                                        density * wqc,
                                                                                        density * wqr,
                                                                                        density * current_cloud_number,
                                                                                        density * current_rain_number,
                                                                                        dt);
                    const double converted_cloud = std::min(wqc, dt * conversion.rain_water_kg_m3_s / density);
                    transfer(wqc, wqr, converted_cloud, 0.0);
                    staged_state.cloud_number_tendency().data()[index] +=
                        static_cast<float>(conversion.cloud_number_m3_s / density);
                    staged_state.rain_number_tendency().data()[index] +=
                        static_cast<float>(conversion.rain_number_m3_s / density);
                    const double rain_candidate         = staged_state.rain_number().data()[index] +
                                                          dt * staged_state.rain_number_tendency().data()[index];
                    const double normalized_rain        = density * wqr > 1.0e-12
                                                              ? thompson::normalize_rain_number_m3(
                                                                    density * wqr, density * std::max(0.0, rain_candidate)) /
                                                                    density
                                                              : std::max(0.0, rain_candidate);
                    const float  unprojected_cloud_rate = staged_state.cloud_number_tendency().data()[index];
                    staged_state.cloud_number_tendency().data()[index] = projected_cloud_tendency(
                        staged_state.cloud_number().data()[index], density * wqc, density, unprojected_cloud_rate, dt);
                    if (cloud_on_device ? double(native_cloud.vapor_change.data()[index]) +
                                                  native_cloud.vapor_change_low.data()[index] >
                                              0
                                        : evaporated_cloud > 0)
                    {
                        const float baseline_rate = static_cast<float>(conversion.cloud_number_m3_s / density);
                        const float projected_baseline_rate = projected_cloud_tendency(
                            staged_state.cloud_number().data()[index],
                            density * std::max(0.0, double(qc.data()[index]) - converted_cloud),
                            density,
                            baseline_rate,
                            dt);
                        const double baseline_loss = std::max(0.0, double(baseline_rate) - projected_baseline_rate);
                        const double actual_loss   = std::max(
                            0.0, double(unprojected_cloud_rate) - staged_state.cloud_number_tendency().data()[index]);
                        staged_state.water_aerosol_tendency().data()[index] =
                            static_cast<float>(double(staged_state.water_aerosol_tendency().data()[index]) +
                                               std::max(0.0, actual_loss - baseline_loss));
                    }
                    staged_state.rain_number_tendency().data()[index] =
                        stored_number_tendency(staged_state.rain_number().data()[index], normalized_rain, dt);
                    current_rain_number    = staged_state.rain_number().data()[index] +
                                             dt * staged_state.rain_number_tendency().data()[index];
                    const auto evaporation = thompson::rain_evaporation_rates(
                        P, T, density, wqv, density * wqr, density * current_rain_number, dt);
                    transfer(wqr, wqv, dt * evaporation.vapor_kg_m3_s / density, 0.0);
                    dT_sum += dt * evaporation.temperature_k_s;
                    staged_state.rain_number_tendency().data()[index] =
                        stored_number_tendency(staged_state.rain_number().data()[index],
                                               current_rain_number + dt * evaporation.rain_number_m3_s / density,
                                               dt);
                    staged_state.water_aerosol_tendency().data()[index] +=
                        static_cast<float>(evaporation.water_aerosol_m3_s / density);
                }
                else
                {
                    if (wqc > qc0_)
                        transfer(wqc, wqr,
                                 c_auto_ * (wqc - qc0_) * dt, 0.0);
                    if (wqc > 0.0 && wqr > 0.0)
                        transfer(wqc, wqr, 2.2 * wqc * wqr * dt, 0.0);
                    if (wqr > 0.0)
                    {
                        const double RH = wqv > 0.0
                            ? wqv / std::max(qvsat_w, 1.0e-12) : 0.0;
                        if (RH < 1.0)
                        {
                            rain_before_evaporation = wqr;
                            transfer(wqr, wqv,
                                     c_evap_ * (1.0 - RH) * wqr * dt,
                                     -Lv_cp);
                            evaporated_rain = rain_before_evaporation - wqr;
                        }
                    }
                }

                // Ice / mixed phase.
                if (T < T0)
                {
                    const double qvsat_i =
                        thermodynamics::saturation_mixing_ratio_ice(
                            static_cast<float>(T), static_cast<float>(P));
                    double snow_vapor_rate_kg_m3_s = 0.0;
                    if (prognostic_ready)
                    {
                        const std::size_t index =
                            (static_cast<std::size_t>(i)
                             * static_cast<std::size_t>(NTH)
                             + static_cast<std::size_t>(j))
                                * static_cast<std::size_t>(NZ)
                            + static_cast<std::size_t>(k);
                        const double density = rho_d.data()[index];
                        const auto   transfer_deposition = [&](double& solid, double amount)
                        {
                            if (amount >= 0)
                                transfer(wqv, solid, amount, 0.0);
                            else
                                transfer(solid, wqv, -amount, 0.0);
                        };
                        if (deposition_on_device)
                        {
                            transfer_deposition(wqi,
                                                static_cast<double>(native_deposition[0].data()[index]) +
                                                    native_deposition[1].data()[index]);
                            transfer_deposition(wqs,
                                                static_cast<double>(native_deposition[2].data()[index]) +
                                                    native_deposition[3].data()[index]);
                            transfer_deposition(wqg,
                                                static_cast<double>(native_deposition[4].data()[index]) +
                                                    native_deposition[5].data()[index]);
                            dT_sum += static_cast<double>(native_deposition[6].data()[index]) +
                                      native_deposition[7].data()[index];
                            staged_state.ice_number_tendency().data()[index] = native_deposition[8].data()[index];
                            snow_vapor_rate_kg_m3_s = static_cast<double>(native_deposition[9].data()[index]) +
                                                      native_deposition[10].data()[index];
                        }
                        else
                        {
                            const double initial_ice_number =
                                std::max(0.0,
                                         staged_state.ice_number().data()[index] +
                                             dt * staged_state.ice_number_tendency().data()[index]);
                            const auto deposition = thompson::ice_deposition_rates(*cold_process_tables_,
                                                                                   {P,
                                                                                    T,
                                                                                    density,
                                                                                    wqv,
                                                                                    density * wqi,
                                                                                    density * initial_ice_number,
                                                                                    density * wqs,
                                                                                    density * wqg},
                                                                                   dt);
                            transfer_deposition(wqi, dt * deposition.ice_kg_m3_s / density);
                            transfer_deposition(wqs, dt * deposition.snow_kg_m3_s / density);
                            transfer_deposition(wqg, dt * deposition.graupel_kg_m3_s / density);
                            dT_sum += dt * deposition.temperature_k_s;
                            staged_state.ice_number_tendency().data()[index] = stored_number_tendency(
                                staged_state.ice_number().data()[index],
                                std::max(0.0, initial_ice_number + dt * deposition.ice_number_m3_s / density),
                                dt);
                            snow_vapor_rate_kg_m3_s = deposition.snow_vapor_kg_m3_s;
                        }
                        if (freezing_on_device)
                        {
                            transfer(wqr,
                                     wqi,
                                     static_cast<double>(native_freezing[2].data()[index]) +
                                         native_freezing[3].data()[index],
                                     Lf_cp);
                            transfer(wqr,
                                     wqg,
                                     static_cast<double>(native_freezing[4].data()[index]) +
                                         native_freezing[5].data()[index],
                                     Lf_cp);
                            transfer(wqc,
                                     wqi,
                                     static_cast<double>(native_freezing[0].data()[index]) +
                                         native_freezing[1].data()[index],
                                     Lf_cp);
                            staged_state.cloud_number_tendency().data()[index] = native_freezing[6].data()[index];
                            staged_state.rain_number_tendency().data()[index]  = native_freezing[7].data()[index];
                            staged_state.ice_number_tendency().data()[index] += native_freezing[8].data()[index];
                            staged_state.ice_number_tendency().data()[index] += native_freezing[9].data()[index];
                        }
                        else
                        {
                            const double rain_number = std::max(
                                0.0,
                                static_cast<double>(staged_state.rain_number().data()[index]) +
                                    dt * static_cast<double>(staged_state.rain_number_tendency().data()[index]));
                            const double ice_aerosol = std::max(
                                0.0,
                                static_cast<double>(staged_state.ice_aerosol().data()[index]) +
                                    dt * static_cast<double>(staged_state.ice_aerosol_tendency().data()[index]));
                            const double nuclei = thompson::demott_ice_number_m3(Tc, density, density * ice_aerosol);
                            const auto   frozen = thompson::rain_freezing_exchange(
                                *cold_process_tables_, {T, density * wqr, density * rain_number, nuclei});
                            transfer(wqr, wqi, frozen.ice_kg_m3 / density, Lf_cp);
                            transfer(wqr, wqg, frozen.graupel_kg_m3 / density, Lf_cp);
                            staged_state.rain_number_tendency().data()[index] =
                                stored_number_tendency(staged_state.rain_number().data()[index],
                                                       std::max(0.0, rain_number + frozen.rain_number_m3 / density),
                                                       dt);
                            staged_state.ice_number_tendency().data()[index] +=
                                static_cast<float>(frozen.ice_number_m3 * inv_dt / density);
                            const double cloud_number = std::max(
                                0.0,
                                static_cast<double>(staged_state.cloud_number().data()[index]) +
                                    dt * static_cast<double>(staged_state.cloud_number_tendency().data()[index]));
                            const auto cloud_frozen = thompson::cloud_freezing_exchange(
                                *cold_process_tables_, {T, density * wqc, density * cloud_number, nuclei});
                            transfer(wqc, wqi, cloud_frozen.ice_kg_m3 / density, Lf_cp);
                            staged_state.cloud_number_tendency().data()[index] = stored_number_tendency(
                                staged_state.cloud_number().data()[index],
                                std::max(0.0, cloud_number + cloud_frozen.cloud_number_m3 / density),
                                dt);
                            staged_state.ice_number_tendency().data()[index] +=
                                static_cast<float>(cloud_frozen.ice_number_m3 * inv_dt / density);
                        }
                    }

                    // Immersion + homogeneous freezing of cloud -> ice (release L_f).
                    // FIX: the old freezing_probability saturated to 1.0 (qc_vol
                    // scaled by a 1e-15 droplet volume), glaciating all cloud in a
                    // single step. Use a gradual immersion rate that ramps with
                    // supercooling and approaches rapid homogeneous freezing near
                    // -38 C (the transfer is mass-limited regardless).
                    const double supercooling = T0 - T;
                    const double freeze_frac_per_s =
                        (supercooling
                         >= microphysics_constants::homogeneous_freezing_dT)
                            ? 1.0
                            : 1.0e-3 * (supercooling
                                / microphysics_constants::homogeneous_freezing_dT);
                    if (!prognostic_ready && wqc > 0.0
                        && freeze_frac_per_s > 0.0)
                    {
                        transfer(wqc, wqi, wqc * freeze_frac_per_s * dt, Lf_cp);
                    }
                    if (prognostic_ready)
                    {
                        const std::size_t index =
                            (static_cast<std::size_t>(i)
                             * static_cast<std::size_t>(NTH)
                             + static_cast<std::size_t>(j))
                                * static_cast<std::size_t>(NZ)
                            + static_cast<std::size_t>(k);
                        const double available_water_aerosol = std::max(
                            0.0,
                            static_cast<double>(
                                staged_state.water_aerosol().data()[index])
                                + dt * static_cast<double>(
                                    staged_state
                                        .water_aerosol_tendency()
                                        .data()[index]));
                        const double snow_number_for_koop =
                            wqs <= 1.0e-12 ? 0.0 : 1.0e6;
                        const double current_ice_number = std::max(
                            0.0,
                            static_cast<double>(
                                staged_state.ice_number().data()[index])
                                + dt * static_cast<double>(
                                    staged_state
                                        .ice_number_tendency()
                                        .data()[index]));
                        const auto nucleated = thompson::nucleate_aerosol_ice(
                            T,
                            wqv,
                            qvsat_w,
                            qvsat_i,
                            static_cast<double>(rho_d.data()[index]),
                            current_ice_number,
                            snow_number_for_koop,
                            available_water_aerosol,
                            static_cast<double>(
                                staged_state.ice_aerosol().data()[index]),
                            dt);
                        transfer(
                            wqv, wqi, nucleated.ice_mass_kg_dry, Ls_cp);
                        staged_state.ice_number_tendency().data()[index]
                            += static_cast<float>(
                                nucleated.ice_number_kg_dry * inv_dt);
                        staged_state.water_aerosol_tendency().data()[index] = stored_number_tendency(
                            staged_state.water_aerosol().data()[index],
                            std::max(0.0, available_water_aerosol + nucleated.water_aerosol_kg_dry),
                            dt);
                        staged_state.ice_aerosol_tendency().data()[index] = stored_number_tendency(
                            staged_state.ice_aerosol().data()[index],
                            std::max(0.0, staged_state.ice_aerosol().data()[index] + nucleated.ice_aerosol_kg_dry),
                            dt);
                    }
                    // Vapor deposition onto ice (release L_s).
                    if (!prognostic_ready && wqi > 0.0 && wqv > qvsat_i)
                    {
                        transfer(wqv, wqi, 1.0e-3 * (wqv - qvsat_i) * dt, Ls_cp);
                    }
                    if (prognostic_ready)
                    {
                        const std::size_t index =
                            (static_cast<std::size_t>(i)
                             * static_cast<std::size_t>(NTH)
                             + static_cast<std::size_t>(j))
                                * static_cast<std::size_t>(NZ)
                            + static_cast<std::size_t>(k);
                        const double density = rho_d.data()[index];
                        const double cloud_number = std::max(
                            0.0,
                            static_cast<double>(
                                staged_state.cloud_number().data()[index])
                                + dt * static_cast<double>(
                                    staged_state
                                        .cloud_number_tendency()
                                        .data()[index]));
                        const auto riming = thompson::snow_riming_rates(
                            {T, density, density * wqc,
                             density * cloud_number, density * wqs,
                             density * wqg, snow_vapor_rate_kg_m3_s},
                            dt);
                        transfer(wqc, wqi, dt * riming.ice_kg_m3_s / density, 0.0);
                        transfer(wqc, wqs, dt * riming.snow_kg_m3_s / density, 0.0);
                        transfer(wqc, wqg, dt * riming.graupel_kg_m3_s / density, 0.0);
                        dT_sum += dt * riming.temperature_k_s;
                        staged_state.cloud_number_tendency().data()[index] = stored_number_tendency(
                            staged_state.cloud_number().data()[index],
                            std::max(0.0, cloud_number + dt * riming.cloud_number_m3_s / density),
                            dt);
                        staged_state.ice_number_tendency().data()[index]
                            += static_cast<float>(
                                riming.ice_number_m3_s / density);
                        snow_fall_speed_multiplier.data()[index] =
                            static_cast<float>(
                                riming.snow_fall_speed_multiplier);
                    }
                    else if (wqc > 0.0 && wqi > 0.0)
                    {
                        const double E_rime = collection_efficiency(Tc);
                        transfer(wqc, wqg, E_rime * wqc * wqi * dt, Lf_cp);
                    }
                    if (prognostic_ready)
                    {
                        const std::size_t index =
                            (static_cast<std::size_t>(i)
                             * static_cast<std::size_t>(NTH)
                             + static_cast<std::size_t>(j))
                                * static_cast<std::size_t>(NZ)
                            + static_cast<std::size_t>(k);
                        const double density = rho_d.data()[index];
                        const double ice_number = std::max(
                            0.0,
                            static_cast<double>(
                                staged_state.ice_number().data()[index])
                                + dt * static_cast<double>(
                                    staged_state
                                        .ice_number_tendency()
                                        .data()[index]));
                        const auto conversion =
                            thompson::ice_to_snow_rates(
                                *cold_process_tables_,
                                {density * wqi, density * ice_number},
                                dt);
                        transfer(wqi, wqs, dt * conversion.snow_kg_m3_s / density, 0.0);
                        staged_state.ice_number_tendency().data()[index] = stored_number_tendency(
                            staged_state.ice_number().data()[index],
                            std::max(0.0, ice_number + dt * conversion.ice_number_m3_s / density),
                            dt);
                    }
                    else if (wqi > 5.0e-4)
                    {
                        transfer(
                            wqi, wqs, (wqi - 5.0e-4) * 1.0e-3 * dt, 0.0);
                    }
                    // Sublimation when subsaturated wrt ice (absorb L_s).
                    const double RH_i = (wqv > 0.0) ? wqv / std::max(qvsat_i, 1.0e-12) : 0.0;
                    if (RH_i < 1.0)
                    {
                        if (!prognostic_ready && wqi > 0.0)
                            transfer(wqi, wqv,
                                     1.0e-3 * (1.0 - RH_i) * wqi * dt,
                                     -Ls_cp);
                        if (!prognostic_ready && wqs > 0.0)
                            transfer(wqs, wqv,
                                     1.0e-3 * (1.0 - RH_i) * wqs * dt,
                                     -Ls_cp);
                    }
                }
                else
                {
                    if (prognostic_ready)
                    {
                        const std::size_t index =
                            (static_cast<std::size_t>(i)
                             * static_cast<std::size_t>(NTH)
                             + static_cast<std::size_t>(j))
                                * static_cast<std::size_t>(NZ)
                            + static_cast<std::size_t>(k);
                        const double density = rho_d.data()[index];
                        const auto melting = thompson::frozen_melting_rates(
                            P, T, density, wqv, density * wqs,
                            density * wqg, dt);
                        transfer(wqs, wqr, -dt * melting.snow_kg_m3_s / density, 0.0);
                        transfer(wqg, wqr, -dt * melting.graupel_kg_m3_s / density, 0.0);
                        dT_sum += dt * melting.temperature_k_s;
                        staged_state.rain_number_tendency().data()[index]
                            += static_cast<float>(
                                melting.rain_number_m3_s / density);
                    }
                    else
                    {
                        if (wqs > 0.0)
                            transfer(
                                wqs, wqr,
                                microphysics_constants::c_melt_snow_per_K
                                    * Tc * wqs * dt,
                                -Lf_cp);
                        if (wqg > 0.0)
                            transfer(
                                wqg, wqr,
                                microphysics_constants::c_melt_graupel_per_K
                                    * Tc * wqg * dt,
                                -Lf_cp);
                        if (wqh > 0.0)
                            transfer(
                                wqh, wqr,
                                microphysics_constants::c_melt_hail_per_K
                                    * Tc * wqh * dt,
                                -Lf_cp);
                    }
                }

                if (prognostic_ready)
                {
                    const std::size_t index =
                        (static_cast<std::size_t>(i)
                         * static_cast<std::size_t>(NTH)
                         + static_cast<std::size_t>(j))
                            * static_cast<std::size_t>(NZ)
                        + static_cast<std::size_t>(k);
                    const double density = rho_d.data()[index];
                    double current_rain_number =
                        staged_state.rain_number().data()[index]
                        + dt * staged_state.rain_number_tendency().data()[index];
                    const auto collection = thompson::collection_process_rates(
                        *collection_tables_, density * wqr,
                        density * current_rain_number, density * wqs,
                        density * wqg,
                        thompson::wet_bulb_temperature(P, T, wqv), dt);
                    const double                rain_before_collection = wqr;
                    const std::array<double, 3> collection_rates{
                        collection.rain_kg_m3_s, collection.snow_kg_m3_s, collection.graupel_kg_m3_s};
                    const std::array<double*, 3> collection_water{&wqr, &wqs, &wqg};
                    double                       positive = 0;
                    for (double rate : collection_rates)
                        positive += std::max(0.0, rate);
                    for (std::size_t donor = 0; donor < 3; ++donor)
                    {
                        if (collection_rates[donor] >= 0)
                            continue;
                        double remaining = std::min(*collection_water[donor], -dt * collection_rates[donor] / density);
                        double denominator = positive;
                        for (std::size_t receiver = 0; receiver < 3; ++receiver)
                        {
                            if (collection_rates[receiver] <= 0)
                                continue;
                            const double amount = remaining * collection_rates[receiver] / denominator;
                            transfer(*collection_water[donor], *collection_water[receiver], amount, 0.0);
                            remaining -= amount;
                            denominator -= collection_rates[receiver];
                        }
                    }
                    dT_sum -= Lf_cp * (wqr - rain_before_collection);
                    current_rain_number =
                        std::max(0.0, current_rain_number + dt * collection.rain_number_m3_s / density);
                    staged_state.rain_number_tendency().data()[index] =
                        stored_number_tendency(staged_state.rain_number().data()[index], current_rain_number, dt);
                    if (evaporated_rain > 0.0
                        && rain_before_evaporation > 0.0
                        && current_rain_number > 0.0)
                    {
                        const double returned = std::min(
                            0.99 * current_rain_number,
                            evaporated_rain * current_rain_number
                                / rain_before_evaporation);
                        staged_state.rain_number_tendency().data()[index]
                            -= static_cast<float>(returned * inv_dt);
                        staged_state.water_aerosol_tendency().data()[index]
                            += static_cast<float>(returned * inv_dt);
                        current_rain_number -= returned;
                    }
                    const double current_water_aerosol =
                        staged_state.water_aerosol().data()[index]
                        + dt * staged_state.water_aerosol_tendency().data()[index];
                    const double current_ice_aerosol =
                        staged_state.ice_aerosol().data()[index]
                        + dt * staged_state.ice_aerosol_tendency().data()[index];
                    const auto scavenging =
                        thompson::aerosol_scavenging_tendencies(
                            T, density, wqr,
                            current_rain_number, wqs, wqg,
                            current_water_aerosol, current_ice_aerosol, dt);
                    staged_state.water_aerosol_tendency().data()[index] = stored_number_tendency(
                        staged_state.water_aerosol().data()[index],
                        std::max(0.0, current_water_aerosol + dt * scavenging.water_aerosol_kg_dry_s),
                        dt);
                    staged_state.ice_aerosol_tendency().data()[index] = stored_number_tendency(
                        staged_state.ice_aerosol().data()[index],
                        std::max(0.0, current_ice_aerosol + dt * scavenging.ice_aerosol_kg_dry_s),
                        dt);
                }

                if (prognostic_ready)
                {
                    const double density = rho_d(i, j, k);
                    const double number  = std::max(
                        0.0, staged_state.ice_number()(i, j, k) + dt * staged_state.ice_number_tendency()(i, j, k));
                    const double target =
                        density * wqi > 1e-12
                            ? std::min(999e3, thompson::normalize_ice_number_m3(density * wqi, density * number)) /
                                  density
                            : number;
                    staged_state.ice_number_tendency()(i, j, k) =
                        stored_number_tendency(staged_state.ice_number()(i, j, k), target, dt);
                }
                dqv_dt[i][j][k]    = static_cast<float>((wqv - static_cast<double>(qv[i][j][k])) * inv_dt);
                dqc_dt[i][j][k] = static_cast<float>((wqc - static_cast<double>(qc[i][j][k])) * inv_dt);
                dqr_dt[i][j][k] = static_cast<float>((wqr - static_cast<double>(qr[i][j][k])) * inv_dt);
                dqi_dt[i][j][k] = static_cast<float>((wqi - static_cast<double>(qi[i][j][k])) * inv_dt);
                dqs_dt[i][j][k] = static_cast<float>((wqs - static_cast<double>(qs[i][j][k])) * inv_dt);
                dqg_dt[i][j][k] = static_cast<float>((wqg - static_cast<double>(qg[i][j][k])) * inv_dt);
                dqh_dt[i][j][k] = static_cast<float>((wqh - static_cast<double>(qh[i][j][k])) * inv_dt);
                dtheta_dt[i][j][k] = static_cast<float>(dT_sum * inv_dt);
                }
                catch (...)
                {
                    if (!pointwise_failed.exchange(true, std::memory_order_relaxed))
                    {
                        pointwise_error = std::current_exception();
                    }
                }
            }
        }
    }
    if (pointwise_failed.load(std::memory_order_relaxed))
    {
        try
        {
            std::rethrow_exception(pointwise_error);
        }
        catch (const std::exception& error)
        {
            throw std::runtime_error(std::string("Thompson pointwise process failed: ") + error.what());
        }
        catch (...)
        {
            throw std::runtime_error("Thompson pointwise process failed");
        }
    }
    }

    native_deposition = {};
    native_freezing = {};
    native_warm  = {};
    native_cloud = {};

    struct SedimentationDonor
    {
        const char* name;
        const Field3D* state;
        const Field3D* tendency;
    };
    for (const auto& donor : std::array<SedimentationDonor, 5>{
             SedimentationDonor{"qc", &qc, &dqc_dt},
             SedimentationDonor{"qr", &qr, &dqr_dt},
             SedimentationDonor{"qi", &qi, &dqi_dt},
             SedimentationDonor{"qs", &qs, &dqs_dt},
             SedimentationDonor{"qg", &qg, &dqg_dt}})
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
            const float source = (*donor.state)(i, j, k);
            const float tendency = (*donor.tendency)(i, j, k);
            const double candidate = source + dt * tendency;
            const double storage_envelope = 2.0e-12 + 2.0 * (
                static_cast<double>(std::nextafter(
                    source, std::numeric_limits<float>::infinity()))
                    - static_cast<double>(source)
                + std::abs(dt) * (
                    static_cast<double>(std::nextafter(
                        tendency, std::numeric_limits<float>::infinity()))
                    - static_cast<double>(tendency)));
            if (!std::isfinite(candidate)
                || candidate < -storage_envelope)
                throw std::runtime_error(std::string(
                    "invalid Thompson post-process sedimentation donor ")
                    + donor.name + " at cell (" + std::to_string(i) + ","
                    + std::to_string(j) + "," + std::to_string(k) + ")"
                    + ": " + std::to_string(candidate));
                }
    if (prognostic_ready)
    {
        std::string error;
        if (!staged_state.validate_update(dt, domain, error))
            throw std::runtime_error(error);
    }
    Field3D surface_sink_candidate(NR, NTH, 1, 0.0f);
    compute_sedimentation(p, theta, rho_d, qc, qr, qi, qs, qg,
                          dqc_dt, dqr_dt, dqi_dt, dqs_dt, dqg_dt,
                          snow_fall_speed_multiplier,
                          prognostic_ready ? &staged_state : nullptr,
                          surface_sink_candidate, dt);
    complete_cloud_phase(
        p, theta, rho_d, qv, qc, qi, dqv_dt, dqc_dt, dqi_dt, dtheta_dt, prognostic_ready ? &staged_state : nullptr, dt);

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

    for (const Field3D* field : {
             &dtheta_dt, &dqv_dt, &dqc_dt, &dqr_dt,
             &dqi_dt, &dqs_dt, &dqg_dt, &dqh_dt})
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                    if (!std::isfinite((*field)(i, j, k)))
                        throw std::runtime_error(
                            "nonfinite Thompson process candidate");
    if (prognostic_ready)
    {
        std::string error;
        if (!staged_state.validate_update(dt, domain, error))
            throw std::runtime_error(error);
    }

    destination_dtheta_dt = std::move(dtheta_dt);
    destination_dqv_dt = std::move(dqv_dt);
    destination_dqc_dt = std::move(dqc_dt);
    destination_dqr_dt = std::move(dqr_dt);
    destination_dqi_dt = std::move(dqi_dt);
    destination_dqs_dt = std::move(dqs_dt);
    destination_dqg_dt = std::move(dqg_dt);
    destination_dqh_dt = std::move(dqh_dt);
    prognostic_state_ = std::move(staged_state);
    surface_sedimentation_sink_rate_ = std::move(surface_sink_candidate);
}

/**
 * @brief Computes the collection efficiency for the Thompson scheme.
 */
double ThompsonScheme::collection_efficiency(double T_celsius)
{
    if (T_celsius < -5.0) return 0.8;
    else if (T_celsius < 0.0) return 0.6;
    else return 0.0;
}

} // namespace tmv
