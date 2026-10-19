#include "diagnostics/parcel_thermodynamics.hpp"
#include "diagnostics/physical_reference.hpp"
/**
 * @file derived_slice_compute.cpp
 * @brief Derived diagnostic slice computation.
 */

#include "diagnostics/derived_slice_compute.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "core/infra/physical_constants.hpp"
#include "diagnostics/cartesian_horizontal_derivatives.hpp"
#include "diagnostics/derived_field_math.hpp"
#include "diagnostics/physical_derivative_kernel.hpp"

using tmv::derived::linear_z_to_dbz;
using tmv::derived::okubo_weiss_parameter;

void compute_derived_slices_for_theta(DerivedSliceScratch&                  derived_slices,
                                      const DerivedSliceComputationContext& context)
{
    const int              NR                       = context.nr;
    const int              NTH                      = context.nth;
    const int              NZ                       = context.nz;
    const double           dr                       = context.dr;
    const double           dtheta                   = context.dtheta;
    const double           dz                       = context.dz;
    const double           dt                       = context.dt;
    const bool             radar_diagnostic_available =
        output_field_has_dependency(context.dependencies, OutputFieldDependency::RadarDiagnostic);
    const bool vorticity_diagnostics_available =
        output_field_has_dependency(context.dependencies, OutputFieldDependency::VorticityDiagnostics);
    const bool angular_momentum_diagnostics_available =
        output_field_has_dependency(context.dependencies, OutputFieldDependency::AngularMomentumDiagnostics);
    const int      th                            = context.theta_index;
    const float    reflectivity_dbz_min          = context.reflectivity_dbz_min;
    const float    reflectivity_dbz_max          = context.reflectivity_dbz_max;
    const auto&    storm_u_level_mean            = context.storm_u_level_mean;
    const auto&    storm_v_level_mean            = context.storm_v_level_mean;
    const Field3D& rho                           = context.rho;
    const Field3D& p                             = context.p;
    const Field3D& u                             = context.u;
    const Field3D& v                             = context.v;
    const Field3D& w                             = context.w;
    const Field3D& theta                         = context.theta;
    const Field3D& qv                            = context.qv;
    const Field3D& qc                            = context.qc;
    const Field3D& qr                            = context.qr;
    const Field3D& qi                            = context.qi;
    const Field3D& qs                            = context.qs;
    const Field3D& qg                            = context.qg;
    const Field3D& qh                            = context.qh;
    const Field3D& radar_reflectivity            = context.radar_reflectivity;
    const Field3D& radar_radial_velocity         = context.radar_radial_velocity;
    const Field3D& radar_zdr                     = context.radar_zdr;
    const Field3D& radar_kdp                     = context.radar_kdp;
    const Field3D& radar_rhohv                   = context.radar_rhohv;
    const Field3D& vorticity_r                   = context.vorticity_r;
    const Field3D& vorticity_theta               = context.vorticity_theta;
    const Field3D& vorticity_z                   = context.vorticity_z;
    const Field3D& angular_momentum              = context.angular_momentum;
    const auto&    theta0_base                   = context.theta0_base;
    auto&          temperature_slice             = derived_slices.writable_buffer("temperature");
    auto&          theta_prime_slice             = derived_slices.writable_buffer("theta_prime");
    auto&          theta_v_slice                 = derived_slices.writable_buffer("theta_v");
    auto&          theta_e_slice                 = derived_slices.writable_buffer("theta_e");
    auto&          dewpoint_slice                = derived_slices.writable_buffer("dewpoint");
    auto&          relative_humidity_slice       = derived_slices.writable_buffer("relative_humidity");
    auto&          saturation_mixing_ratio_slice = derived_slices.writable_buffer("saturation_mixing_ratio");
    auto&          total_condensate_slice        = derived_slices.writable_buffer("total_condensate");
    auto&          cloud_condensate_slice        = derived_slices.writable_buffer("cloud_condensate");
    auto&          cloud_liquid_condensate_slice = derived_slices.writable_buffer("cloud_liquid_condensate");
    auto&          cloud_frozen_condensate_slice = derived_slices.writable_buffer("cloud_frozen_condensate");
    auto&          reflectivity_dbz_slice        = derived_slices.writable_buffer("reflectivity_dbz");
    auto&          radial_velocity_slice         = derived_slices.writable_buffer("radial_velocity");
    auto&          vorticity_magnitude_slice     = derived_slices.writable_buffer("vorticity_magnitude");
    auto&          divergence_slice              = derived_slices.writable_buffer("divergence");
    auto&          potential_vorticity_slice     = derived_slices.writable_buffer("potential_vorticity");
    auto&          buoyancy_slice                = derived_slices.writable_buffer("buoyancy");
    auto& horizontal_vorticity_streamwise_slice  = derived_slices.writable_buffer("horizontal_vorticity_streamwise");
    auto& horizontal_vorticity_crosswise_slice   = derived_slices.writable_buffer("horizontal_vorticity_crosswise");
    auto& pressure_gradient_force_x_slice        = derived_slices.writable_buffer("pressure_gradient_force_x");
    auto& pressure_gradient_force_y_slice        = derived_slices.writable_buffer("pressure_gradient_force_y");
    auto& pressure_gradient_force_z_slice        = derived_slices.writable_buffer("pressure_gradient_force_z");
    auto& storm_relative_winds_slice             = derived_slices.writable_buffer("storm_relative_winds");
    auto& helicity_density_slice                 = derived_slices.writable_buffer("helicity_density");
    auto& okubo_weiss_slice                      = derived_slices.writable_buffer("okubo_weiss");
    auto& theta_w_slice                          = derived_slices.writable_buffer("theta_w");
    auto& zdr_slice                              = derived_slices.writable_buffer("zdr");
    auto& kdp_slice                              = derived_slices.writable_buffer("kdp");
    auto& rhohv_slice                            = derived_slices.writable_buffer("rhohv");
    auto& streamlines_slice                      = derived_slices.writable_buffer("streamlines");
    auto& trajectory_paths_slice                 = derived_slices.writable_buffer("trajectory_paths");
    auto& q_vectors_slice                        = derived_slices.writable_buffer("q_vectors");
    auto& turbulent_diffusion_term_slice         = derived_slices.writable_buffer("turbulent_diffusion_term");
    auto& cross_section_slice                    = derived_slices.writable_buffer("cross_section");
    auto& rhi_slice_diag                         = derived_slices.writable_buffer("rhi_slice");
    auto& hodograph_aligned_cross_section_slice  = derived_slices.writable_buffer("hodograph_aligned_cross_section");
    auto& forward_trajectories_slice             = derived_slices.writable_buffer("forward_trajectories");
    auto& backward_trajectories_slice            = derived_slices.writable_buffer("backward_trajectories");
    auto& parcel_buoyancy_trajectory_slice       = derived_slices.writable_buffer("parcel_buoyancy_trajectory");
    auto& vorticity_trajectory_slice             = derived_slices.writable_buffer("vorticity_trajectory");
    auto& circulation_material_surface_slice     = derived_slices.writable_buffer("circulation_material_surface");

    const float latent_heat_v   = 2.5e6f;
    const float cp_f            = static_cast<float>(cp);
    const auto  optional_sample = [](const Field3D& field, bool available, int i, int j, int k) -> float
    { return available ? static_cast<float>(field[i][j][k]) : std::numeric_limits<float>::quiet_NaN(); };

    const auto theta_reference = [&](int i, int k) {
        return tmv::diagnostics::reference_at_cell(theta0_base, context.terrain_metrics, dz, i, th, k);
    };
    const auto physical_buoyancy = [&](int i, int k) {
        if (!context.rho0_base || !context.qv0_base || !context.p0_base)
            throw std::invalid_argument("buoyancy requires the level's density, vapor and pressure reference profiles");
        const double reference_density = context.terrain_metrics
            ? tmv::terrain_transport::reference_density_fp32(*context.p0_base, theta0_base,
                context.terrain_metrics->z(i, th, k), dz)
            : tmv::diagnostics::reference_at_cell(*context.rho0_base, nullptr, dz, i, th, k);
        const double reference_vapor = tmv::diagnostics::reference_at_cell(
            *context.qv0_base, context.terrain_metrics, dz, i, th, k);
        double loading = 0.0;
        for (const Field3D* species : {&qc, &qr, &qi, &qs, &qg, &qh})
        {
            const double value = (*species)[i][th][k];
            if (!std::isfinite(value) || value < 0.0) return std::numeric_limits<float>::quiet_NaN();
            loading += value;
        }
        return static_cast<float>(tmv::diagnostics::density_buoyancy(
            rho[i][th][k], reference_density, qv[i][th][k], reference_vapor, loading));
    };

    // The presentation path and many custom exports request only independent,
    // pointwise products. Keep those frames out of the general diagnostic
    // kernel below: that kernel also evaluates gradients, wet-bulb
    // trigonometry, trajectories, polarimetric proxies, and other products
    // whose discard-sink results are not consumed.
    const bool        need_temperature         = derived_slices.requested("temperature");
    const bool        need_theta_prime         = derived_slices.requested("theta_prime");
    const bool        need_total_condensate    = derived_slices.requested("total_condensate");
    const bool        need_cloud_condensate    = derived_slices.requested("cloud_condensate");
    const bool        need_cloud_liquid        = derived_slices.requested("cloud_liquid_condensate");
    const bool        need_cloud_frozen        = derived_slices.requested("cloud_frozen_condensate");
    const bool        need_reflectivity_dbz    = derived_slices.requested("reflectivity_dbz");
    const bool        need_radial_velocity     = derived_slices.requested("radial_velocity");
    const bool        need_vorticity_magnitude = derived_slices.requested("vorticity_magnitude");
    const bool        need_buoyancy            = derived_slices.requested("buoyancy");
    const std::size_t simple_product_count =
        static_cast<std::size_t>(need_temperature) + static_cast<std::size_t>(need_theta_prime) +
        static_cast<std::size_t>(need_total_condensate) + static_cast<std::size_t>(need_cloud_condensate) +
        static_cast<std::size_t>(need_cloud_liquid) + static_cast<std::size_t>(need_cloud_frozen) +
        static_cast<std::size_t>(need_reflectivity_dbz) + static_cast<std::size_t>(need_vorticity_magnitude) +
        static_cast<std::size_t>(need_buoyancy) + static_cast<std::size_t>(need_radial_velocity);

    if (simple_product_count > 0 && derived_slices.requests_only_simple_pointwise_products())
    {
        std::size_t simple_idx = 0;
        for (int k = 0; k < NZ; ++k)
        {
            for (int i = 0; i < NR; ++i)
            {
                const float       theta_value = static_cast<float>(theta[i][th][k]);

                if (need_temperature)
                {
                    temperature_slice[simple_idx] =
                        tmv::derived::temperature_from_theta_and_pressure(theta_value, static_cast<float>(p[i][th][k]));
                }
                if (need_theta_prime)
                {
                    theta_prime_slice[simple_idx] = theta_value - static_cast<float>(theta_reference(i, k));
                }

                float qc_value = 0.0f;
                float qi_value = 0.0f;
                float qs_value = 0.0f;
                if (need_total_condensate || need_cloud_condensate || need_cloud_liquid || need_cloud_frozen)
                {
                    qc_value = std::max(0.0f, static_cast<float>(qc[i][th][k]));
                    qi_value = std::max(0.0f, static_cast<float>(qi[i][th][k]));
                    qs_value = std::max(0.0f, static_cast<float>(qs[i][th][k]));
                }
                if (need_total_condensate)
                {
                    const float qr_value = std::max(0.0f, static_cast<float>(qr[i][th][k]));
                    const float qg_value = std::max(0.0f, static_cast<float>(qg[i][th][k]));
                    const float qh_value = std::max(0.0f, static_cast<float>(qh[i][th][k]));
                    total_condensate_slice[simple_idx] =
                        qc_value + qr_value + qi_value + qs_value + qg_value + qh_value;
                }
                if (need_cloud_condensate)
                {
                    cloud_condensate_slice[simple_idx] = qc_value + qi_value + qs_value;
                }
                if (need_cloud_liquid)
                {
                    cloud_liquid_condensate_slice[simple_idx] = qc_value;
                }
                if (need_cloud_frozen)
                {
                    cloud_frozen_condensate_slice[simple_idx] = qi_value + qs_value;
                }
                if (need_reflectivity_dbz)
                {
                    const float z_linear = optional_sample(radar_reflectivity, radar_diagnostic_available, i, th, k);
                    reflectivity_dbz_slice[simple_idx] =
                        linear_z_to_dbz(z_linear, reflectivity_dbz_min, reflectivity_dbz_max);
                }
                if (need_radial_velocity)
                {
                    const bool available =
                        radar_radial_velocity.size_r() == NR &&
                        radar_radial_velocity.size_th() == NTH &&
                        radar_radial_velocity.size_z() == NZ;
                    radial_velocity_slice[simple_idx] =
                        optional_sample(radar_radial_velocity, available, i, th, k);
                }
                if (need_vorticity_magnitude)
                {
                    const float vort_r = optional_sample(vorticity_r, vorticity_diagnostics_available, i, th, k);
                    const float vort_theta =
                        optional_sample(vorticity_theta, vorticity_diagnostics_available, i, th, k);
                    const float vort_z = optional_sample(vorticity_z, vorticity_diagnostics_available, i, th, k);
                    vorticity_magnitude_slice[simple_idx] =
                        (std::isfinite(vort_r) && std::isfinite(vort_theta) && std::isfinite(vort_z))
                            ? std::sqrt((vort_r * vort_r) + (vort_theta * vort_theta) + (vort_z * vort_z))
                            : std::numeric_limits<float>::quiet_NaN();
                }
                if (need_buoyancy)
                {
                    buoyancy_slice[simple_idx] = physical_buoyancy(i, k);
                }
                ++simple_idx;
            }
        }
        return;
    }

    // Complex/custom combinations share the general loop, but each invariant
    // group below runs only when its result or an explicit intermediate is
    // allocated. This keeps one traversal without restoring the former
    // compute-everything discard-sink behavior.
    const bool compute_temperature             = derived_slices.has_buffer("temperature");
    const bool compute_theta_prime             = derived_slices.has_buffer("theta_prime");
    const bool compute_theta_v                 = derived_slices.has_buffer("theta_v");
    const bool compute_theta_e                 = derived_slices.has_buffer("theta_e");
    const bool compute_dewpoint                = derived_slices.has_buffer("dewpoint");
    const bool compute_relative_humidity       = derived_slices.has_buffer("relative_humidity");
    const bool compute_saturation_mixing_ratio = derived_slices.has_buffer("saturation_mixing_ratio");
    const bool compute_thermodynamic_scalars =
        compute_temperature || compute_dewpoint || compute_relative_humidity || compute_saturation_mixing_ratio;
    const bool compute_total_condensate        = derived_slices.has_buffer("total_condensate");
    const bool compute_cloud_condensate        = derived_slices.has_buffer("cloud_condensate");
    const bool compute_cloud_liquid            = derived_slices.has_buffer("cloud_liquid_condensate");
    const bool compute_cloud_frozen            = derived_slices.has_buffer("cloud_frozen_condensate");
    const bool compute_reflectivity_dbz        = derived_slices.has_buffer("reflectivity_dbz");
    const bool compute_radial_velocity         = derived_slices.has_buffer("radial_velocity");
    const bool compute_vorticity_magnitude     = derived_slices.has_buffer("vorticity_magnitude");
    const bool compute_divergence              = derived_slices.has_buffer("divergence");
    const bool compute_potential_vorticity     = derived_slices.has_buffer("potential_vorticity");
    const bool compute_buoyancy                = derived_slices.has_buffer("buoyancy");
    const bool compute_streamwise              = derived_slices.has_buffer("horizontal_vorticity_streamwise");
    const bool compute_crosswise               = derived_slices.has_buffer("horizontal_vorticity_crosswise");
    const bool compute_pressure_x              = derived_slices.has_buffer("pressure_gradient_force_x");
    const bool compute_pressure_y              = derived_slices.has_buffer("pressure_gradient_force_y");
    const bool compute_pressure_z              = derived_slices.has_buffer("pressure_gradient_force_z");
    const bool compute_storm_relative          = derived_slices.has_buffer("storm_relative_winds");
    const bool compute_helicity                = derived_slices.has_buffer("helicity_density");
    const bool compute_okubo_weiss             = derived_slices.has_buffer("okubo_weiss");
    const bool compute_theta_w                 = derived_slices.has_buffer("theta_w");
    const bool compute_zdr                     = derived_slices.has_buffer("zdr");
    const bool compute_kdp                     = derived_slices.has_buffer("kdp");
    const bool compute_rhohv                   = derived_slices.has_buffer("rhohv");
    const bool compute_streamlines             = derived_slices.has_buffer("streamlines");
    const bool compute_trajectory_paths        = derived_slices.has_buffer("trajectory_paths");
    const bool compute_q_vectors               = derived_slices.has_buffer("q_vectors");
    const bool compute_turbulent_diffusion     = derived_slices.has_buffer("turbulent_diffusion_term");
    const bool compute_cross_section           = derived_slices.has_buffer("cross_section");
    const bool compute_rhi                     = derived_slices.has_buffer("rhi_slice");
    const bool compute_hodograph_cross_section = derived_slices.has_buffer("hodograph_aligned_cross_section");
    const bool compute_forward_trajectories    = derived_slices.has_buffer("forward_trajectories");
    const bool compute_backward_trajectories   = derived_slices.has_buffer("backward_trajectories");
    const bool compute_parcel_trajectory       = derived_slices.has_buffer("parcel_buoyancy_trajectory");
    const bool compute_vorticity_trajectory    = derived_slices.has_buffer("vorticity_trajectory");
    const bool compute_circulation             = derived_slices.has_buffer("circulation_material_surface");
    const bool compute_hydrometeor_values      = compute_total_condensate || compute_cloud_condensate ||
                                                 compute_cloud_liquid || compute_cloud_frozen;
    const bool compute_vorticity_values  = compute_vorticity_magnitude || compute_streamwise || compute_crosswise ||
                                           compute_helicity || compute_okubo_weiss || compute_potential_vorticity;
    const bool compute_pressure_gradient = compute_pressure_x || compute_pressure_y || compute_pressure_z;
    const bool compute_motion_trajectory = compute_streamlines || compute_trajectory_paths;
    const bool compute_theta_gradients   = compute_q_vectors || compute_turbulent_diffusion;

    if (compute_potential_vorticity || compute_divergence || compute_pressure_gradient || compute_okubo_weiss || compute_theta_gradients)
    {
        tmv::diagnostics::validate_physical_derivative_geometry(
            NR, NTH, NZ, dr, dtheta, dz, context.terrain_metrics, context.z_center_m);
    }

    const auto gradient = [&](const Field3D& field, int i, int k) {
        return tmv::diagnostics::physical_gradient_at(
            [&](int ii, int jj, int kk) { return static_cast<double>(field[ii][jj][kk]); },
            i, th, k, NR, NTH, NZ, dr, dtheta, dz, context.terrain_metrics, context.z_center_m);
    };

    std::size_t idx = 0;
    for (int k = 0; k < NZ; ++k)
    {
        for (int i = 0; i < NR; ++i)
        {
            const float p_pa          = static_cast<float>(p[i][th][k]);
            const float theta_value   = static_cast<float>(theta[i][th][k]);
            const float qv_value      = static_cast<float>(qv[i][th][k]);
            float       temperature_k = std::numeric_limits<float>::quiet_NaN();
            if (compute_thermodynamic_scalars)
            {
                const auto thermo = tmv::derived::compute_thermodynamic_derived_scalars(theta_value, p_pa, qv_value);
                temperature_k     = thermo.temperature_k;
                if (compute_temperature)
                    temperature_slice[idx] = thermo.temperature_k;
                if (compute_saturation_mixing_ratio)
                    saturation_mixing_ratio_slice[idx] = thermo.saturation_mixing_ratio_kgkg;
                if (compute_relative_humidity)
                    relative_humidity_slice[idx] = thermo.relative_humidity_pct;
                if (compute_dewpoint)
                    dewpoint_slice[idx] = thermo.dewpoint_k;
            }
            if (compute_theta_prime)
            {
                theta_prime_slice[idx] = theta_value - static_cast<float>(theta_reference(i, k));
            }

            const float invalid = std::numeric_limits<float>::quiet_NaN();
            float qc_value = invalid;
            float qr_value = invalid;
            float qi_value = invalid;
            float qs_value = invalid;
            float qg_value = invalid;
            float qh_value = invalid;
            if (compute_hydrometeor_values)
            {
                const auto physical_mixing_ratio = [invalid](float value)
                {
                    return std::isfinite(value) && value >= 0.0f ? value : invalid;
                };
                qc_value = physical_mixing_ratio(static_cast<float>(qc[i][th][k]));
                qr_value = physical_mixing_ratio(static_cast<float>(qr[i][th][k]));
                qi_value = physical_mixing_ratio(static_cast<float>(qi[i][th][k]));
                qs_value = physical_mixing_ratio(static_cast<float>(qs[i][th][k]));
                qg_value = physical_mixing_ratio(static_cast<float>(qg[i][th][k]));
                qh_value = physical_mixing_ratio(static_cast<float>(qh[i][th][k]));
            }

            if (compute_total_condensate)
            {
                total_condensate_slice[idx] = qc_value + qr_value + qi_value + qs_value + qg_value + qh_value;
            }
            // Suspended cloud only (cloud water + cloud ice + snow): the
            // VISIBLE cloud, excluding the falling precip (rain/graupel/
            // hail) that the reflectivity floor already shows. Drives the
            // Studio supercell render so cloud is drawn where the parcel
            // condenses, not where precip falls.
            if (compute_cloud_condensate)
            {
                cloud_condensate_slice[idx] = qc_value + qi_value + qs_value;
            }
            if (compute_cloud_liquid)
            {
                cloud_liquid_condensate_slice[idx] = qc_value;
            }
            if (compute_cloud_frozen)
            {
                cloud_frozen_condensate_slice[idx] = qi_value + qs_value;
            }

            const float qv_nonnegative =
                std::isfinite(qv_value) && qv_value >= 0.0f ? qv_value : invalid;

            if (compute_theta_v && std::isfinite(theta_value) && std::isfinite(qv_nonnegative) &&
                std::isfinite(total_condensate_slice[idx]))
            {
                theta_v_slice[idx] = theta_value * (1.0f + (0.61f * qv_nonnegative) - total_condensate_slice[idx]);
            }
            else if (compute_theta_v)
            {
                theta_v_slice[idx] = std::numeric_limits<float>::quiet_NaN();
            }

            if (compute_theta_e && std::isfinite(theta_value) && std::isfinite(qv_nonnegative) &&
                std::isfinite(temperature_k) && temperature_k > 0.0f)
            {
                const float exponent = (latent_heat_v * qv_nonnegative) / (cp_f * temperature_k);
                theta_e_slice[idx]   = theta_value * std::exp(exponent);
            }
            else if (compute_theta_e)
            {
                theta_e_slice[idx] = std::numeric_limits<float>::quiet_NaN();
            }

            if (compute_reflectivity_dbz)
            {
                const float z_linear        = optional_sample(radar_reflectivity, radar_diagnostic_available, i, th, k);
                reflectivity_dbz_slice[idx] = linear_z_to_dbz(z_linear, reflectivity_dbz_min, reflectivity_dbz_max);
            }
            if (compute_radial_velocity)
            {
                const bool available =
                    radar_radial_velocity.size_r() == NR &&
                    radar_radial_velocity.size_th() == NTH &&
                    radar_radial_velocity.size_z() == NZ;
                radial_velocity_slice[idx] =
                    optional_sample(radar_radial_velocity, available, i, th, k);
            }

            float vort_r     = std::numeric_limits<float>::quiet_NaN();
            float vort_theta = std::numeric_limits<float>::quiet_NaN();
            float vort_z     = std::numeric_limits<float>::quiet_NaN();
            if (compute_vorticity_values)
            {
                vort_r     = optional_sample(vorticity_r, vorticity_diagnostics_available, i, th, k);
                vort_theta = optional_sample(vorticity_theta, vorticity_diagnostics_available, i, th, k);
                vort_z     = optional_sample(vorticity_z, vorticity_diagnostics_available, i, th, k);
            }
            if (compute_vorticity_magnitude && std::isfinite(vort_r) && std::isfinite(vort_theta) &&
                std::isfinite(vort_z))
            {
                vorticity_magnitude_slice[idx] =
                    std::sqrt((vort_r * vort_r) + (vort_theta * vort_theta) + (vort_z * vort_z));
            }
            else if (compute_vorticity_magnitude)
            {
                vorticity_magnitude_slice[idx] = std::numeric_limits<float>::quiet_NaN();
            }

            if (compute_potential_vorticity)
            {
                const auto theta_at = [&](int ii, int jj, int kk)
                {
                    return static_cast<double>(theta[ii][jj][kk]);
                };
                const auto theta_gradient = tmv::diagnostics::physical_gradient_at(
                    theta_at, i, th, k, NR, NTH, NZ,
                    dr, dtheta, dz, context.terrain_metrics, context.z_center_m);
                potential_vorticity_slice[idx] = static_cast<float>(
                    tmv::diagnostics::ertel_potential_vorticity_si(
                        theta_gradient,
                        static_cast<double>(vort_r),
                        static_cast<double>(vort_theta),
                        static_cast<double>(vort_z),
                        context.coriolis_f,
                        static_cast<double>(rho[i][th][k])));
            }

            const float u_center = u[i][th][k];
            const float v_center = v[i][th][k];
            const float w_center = w[i][th][k];

            // Plain Cartesian divergence du/dx + dv/dy + dw/dz; the
            // dr/dtheta slots carry dx/dy in metres.
            if (compute_divergence)
            {
                divergence_slice[idx] = static_cast<float>(gradient(u, i, k).x + gradient(v, i, k).y + gradient(w, i, k).z);
            }

            if (compute_buoyancy) buoyancy_slice[idx] = physical_buoyancy(i, k);

            if ((compute_streamwise || compute_crosswise) && std::isfinite(vort_r) && std::isfinite(vort_theta) &&
                std::isfinite(u_center) && std::isfinite(v_center))
            {
                const float speed_h = std::sqrt((u_center * u_center) + (v_center * v_center));
                if (speed_h > 1.0e-6f)
                {
                    if (compute_streamwise)
                        horizontal_vorticity_streamwise_slice[idx] =
                            ((vort_r * u_center) + (vort_theta * v_center)) / speed_h;
                    if (compute_crosswise)
                        horizontal_vorticity_crosswise_slice[idx] =
                            ((-vort_r * v_center) + (vort_theta * u_center)) / speed_h;
                }
                else
                {
                    if (compute_streamwise)
                        horizontal_vorticity_streamwise_slice[idx] = 0.0f;
                    if (compute_crosswise)
                        horizontal_vorticity_crosswise_slice[idx] = 0.0f;
                }
            }
            else if (compute_streamwise || compute_crosswise)
            {
                if (compute_streamwise)
                    horizontal_vorticity_streamwise_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                if (compute_crosswise)
                    horizontal_vorticity_crosswise_slice[idx] = std::numeric_limits<float>::quiet_NaN();
            }

            if (compute_storm_relative)
            {
                const float storm_u_mean = storm_u_level_mean[static_cast<std::size_t>(k)];
                const float storm_v_mean = storm_v_level_mean[static_cast<std::size_t>(k)];
                if (std::isfinite(u_center) && std::isfinite(v_center) && std::isfinite(storm_u_mean) &&
                    std::isfinite(storm_v_mean))
                {
                    const float u_sr                = u_center - storm_u_mean;
                    const float v_sr                = v_center - storm_v_mean;
                    storm_relative_winds_slice[idx] = std::sqrt((u_sr * u_sr) + (v_sr * v_sr));
                }
                else
                {
                    storm_relative_winds_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                }
            }

            if (compute_helicity && std::isfinite(u_center) && std::isfinite(v_center) && std::isfinite(w_center) &&
                std::isfinite(vort_r) && std::isfinite(vort_theta) && std::isfinite(vort_z))
            {
                helicity_density_slice[idx] = (u_center * vort_r) + (v_center * vort_theta) + (w_center * vort_z);
            }
            else if (compute_helicity)
            {
                helicity_density_slice[idx] = std::numeric_limits<float>::quiet_NaN();
            }

            if (compute_okubo_weiss)
            {
                const auto gu = gradient(u, i, k), gv = gradient(v, i, k);
                okubo_weiss_slice[idx] = okubo_weiss_parameter(
                    gu.x, gv.y, gv.x, gu.y, gv.x - gu.y);
            }

            if (compute_pressure_gradient)
            {
                const auto pressure_gradient = gradient(p, i, k);
                const double density = rho[i][th][k];
                const double inverse_density = std::isfinite(density) && density > 0.0
                    ? 1.0 / density : std::numeric_limits<double>::quiet_NaN();
                if (compute_pressure_x) pressure_gradient_force_x_slice[idx] = static_cast<float>(-pressure_gradient.x * inverse_density);
                if (compute_pressure_y) pressure_gradient_force_y_slice[idx] = static_cast<float>(-pressure_gradient.y * inverse_density);
                if (compute_pressure_z) pressure_gradient_force_z_slice[idx] = static_cast<float>(-pressure_gradient.z * inverse_density);
            }

            if (compute_theta_w)
                theta_w_slice[idx] = static_cast<float>(
                    tmv::diagnostics::wet_bulb_potential_temperature(
                        temperature_k, p_pa, qv[i][th][k]));

            if (compute_zdr)
            {
                const bool available =
                    radar_zdr.size_r() == NR && radar_zdr.size_th() == NTH &&
                    radar_zdr.size_z() == NZ;
                zdr_slice[idx] = optional_sample(radar_zdr, available, i, th, k);
            }
            if (compute_kdp)
            {
                const bool available = radar_kdp.size_r() == NR &&
                    radar_kdp.size_th() == NTH && radar_kdp.size_z() == NZ;
                kdp_slice[idx] = optional_sample(radar_kdp, available, i, th, k);
            }
            if (compute_rhohv)
            {
                const bool available = radar_rhohv.size_r() == NR &&
                    radar_rhohv.size_th() == NTH && radar_rhohv.size_z() == NZ;
                rhohv_slice[idx] = optional_sample(radar_rhohv, available, i, th, k);
            }

            const float dt_f = std::isfinite(static_cast<float>(dt)) && dt >= 0.0
                ? static_cast<float>(dt) : invalid;
            if (compute_motion_trajectory)
            {
                const float stream_speed =
                    (std::isfinite(u_center) && std::isfinite(v_center) && std::isfinite(w_center))
                        ? std::sqrt((u_center * u_center) + (v_center * v_center) + (w_center * w_center))
                        : invalid;
                if (compute_streamlines)
                    streamlines_slice[idx] = stream_speed;
                if (compute_trajectory_paths)
                    trajectory_paths_slice[idx] = stream_speed * dt_f;
            }
            if (compute_forward_trajectories)
                forward_trajectories_slice[idx] = trajectory_paths_slice[idx];
            if (compute_backward_trajectories)
                backward_trajectories_slice[idx] = -trajectory_paths_slice[idx];
            if (compute_parcel_trajectory)
                parcel_buoyancy_trajectory_slice[idx] =
                    std::isfinite(buoyancy_slice[idx]) ? (buoyancy_slice[idx] * dt_f) : 0.0f;
            if (compute_vorticity_trajectory)
                vorticity_trajectory_slice[idx] =
                    std::isfinite(vorticity_magnitude_slice[idx]) ? (vorticity_magnitude_slice[idx] * dt_f) : 0.0f;

            if (compute_circulation)
            {
                const float ang_momentum =
                    optional_sample(angular_momentum, angular_momentum_diagnostics_available, i, th, k);
                circulation_material_surface_slice[idx] = std::isfinite(ang_momentum) ? ang_momentum : 0.0f;
            }
            if (compute_cross_section)
                cross_section_slice[idx] = std::isfinite(theta_value) ? theta_value : 0.0f;
            if (compute_rhi)
                rhi_slice_diag[idx] =
                    std::isfinite(reflectivity_dbz_slice[idx]) ? reflectivity_dbz_slice[idx] : reflectivity_dbz_min;
            if (compute_hodograph_cross_section)
                hodograph_aligned_cross_section_slice[idx] =
                    std::isfinite(storm_relative_winds_slice[idx]) ? storm_relative_winds_slice[idx] : 0.0f;

            if (compute_theta_gradients)
            {
                const auto gt = gradient(theta, i, k);
                if (compute_q_vectors)
                {
                    const auto gu = gradient(u, i, k), gv = gradient(v, i, k);
                    const double qx = -(gu.x * gt.x + gv.x * gt.y);
                    const double qy = -(gv.x * gt.x - gu.x * gt.y);
                    const double qz = -gu.z * gt.z;
                    q_vectors_slice[idx] = static_cast<float>(std::sqrt(qx * qx + qy * qy + qz * qz));
                }
                if (compute_turbulent_diffusion)
                {
                    const auto gradient_component = [&](int axis) {
                        return tmv::diagnostics::physical_gradient_at(
                            [&](int ii, int jj, int kk) {
                                const auto first = tmv::diagnostics::physical_gradient_at(
                                    [&](int x, int y, int z) { return static_cast<double>(theta[x][y][z]); },
                                    ii, jj, kk, NR, NTH, NZ, dr, dtheta, dz, context.terrain_metrics, context.z_center_m);
                                return axis == 0 ? first.x : axis == 1 ? first.y : first.z;
                            }, i, th, k, NR, NTH, NZ, dr, dtheta, dz, context.terrain_metrics, context.z_center_m);
                    };
                    turbulent_diffusion_term_slice[idx] = static_cast<float>(
                        gradient_component(0).x + gradient_component(1).y + gradient_component(2).z);
                }
            }

            ++idx;
        }
    }
}
