#include "diagnostics/column_sampling.hpp"
#include "diagnostics/column_integrals.hpp"
#include "diagnostics/tranche_product_registry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/infra/physical_constants.hpp"
#include "core/infra/grid_coordinates.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/output/field_registry.hpp"
#include "diagnostics/column_water_paths.hpp"
#include "boundary_layer/runtime.hpp"
#include "diagnostics/derived_field_math.hpp"
#include "terrain/conservative_transport.hpp"

namespace tmv
{
namespace diagnostics
{

using tmv::derived::dewpoint_from_temp_qv_pressure;
using tmv::derived::fill_slice_from_radius;
using tmv::derived::lcl_height_from_temperature_dewpoint;
using tmv::derived::linear_z_to_dbz;
using tmv::derived::mean_level_component;
using tmv::derived::sample_field_at_height;
using tmv::derived::saturation_mixing_ratio_from_temp_pressure;
using tmv::derived::temperature_from_theta_and_pressure;

namespace
{

struct SurfaceColumnDiagnosticCache
{
    explicit SurfaceColumnDiagnosticCache(std::size_t radius_size)
        : surface_pressure_mean_by_radius(radius_size, std::numeric_limits<float>::quiet_NaN()),
          surface_sensible_heat_flux_by_radius(radius_size, 0.0f),
          surface_latent_heat_flux_by_radius(radius_size, 0.0f), surface_moisture_flux_by_radius(radius_size, 0.0f),
          skin_temperature_by_radius(radius_size, 273.15f),
          cloud_top_height_by_radius(radius_size, 0.0f), cloud_base_height_by_radius(radius_size, 0.0f),
          lcl_by_radius(radius_size, 0.0f), lfc_by_radius(radius_size, 0.0f), el_by_radius(radius_size, 0.0f),
          cape_by_radius(radius_size, 0.0f), cin_by_radius(radius_size, 0.0f),
          lifted_index_by_radius(radius_size, 0.0f), k_index_by_radius(radius_size, 0.0f),
          showalter_index_by_radius(radius_size, 0.0f), total_totals_by_radius(radius_size, 0.0f),
          ehi_by_radius(radius_size, 0.0f), scp_by_radius(radius_size, 0.0f), stp_by_radius(radius_size, 0.0f)
    {
    }

    std::vector<float> surface_pressure_mean_by_radius;
    std::vector<float> surface_sensible_heat_flux_by_radius;
    std::vector<float> surface_latent_heat_flux_by_radius;
    std::vector<float> surface_moisture_flux_by_radius;
    std::vector<float> skin_temperature_by_radius;
    std::vector<float> cloud_top_height_by_radius;
    std::vector<float> cloud_base_height_by_radius;
    std::vector<float> lcl_by_radius;
    std::vector<float> lfc_by_radius;
    std::vector<float> el_by_radius;
    std::vector<float> cape_by_radius;
    std::vector<float> cin_by_radius;
    std::vector<float> lifted_index_by_radius;
    std::vector<float> k_index_by_radius;
    std::vector<float> showalter_index_by_radius;
    std::vector<float> total_totals_by_radius;
    std::vector<float> ehi_by_radius;
    std::vector<float> scp_by_radius;
    std::vector<float> stp_by_radius;
    int                cached_surface_column_theta = -1;
};

} // namespace

bool with_diagnostic_tranche_registry(TrancheProductRegistryContext& context,
                                      std::string&                   error_message,
                                      const std::function<bool(const std::vector<RegisteredDiagnosticProduct>&)>& body)
{
    const auto& state = context.simulation_state;
    const int NR = state.grid.nr;
    const int NTH = state.grid.nth;
    const int NZ = state.grid.nz;
    const double dr = state.grid.dr;
    const double dz = state.grid.dz;
    const double simulation_time = state.time_s;
    const auto& rho = state.fields.rho;
    const auto& p = state.fields.p;
    const auto& theta = state.fields.theta;
    const auto& qv = state.fields.qv;
    const auto& qc = state.fields.qc;
    const auto& qr = state.fields.qr;
    const auto& qi = state.fields.qi;
    const auto& qs = state.fields.qs;
    const auto& qg = state.fields.qg;
    const auto& qh = state.fields.qh;
    const auto& diagnostics = tmv::require_level_runtime(state.diagnostics, "diagnostics");
    const auto& radar_reflectivity = diagnostics.radar_reflectivity;
    const auto& vorticity_r = diagnostics.vorticity_r;
    const auto& vorticity_theta = diagnostics.vorticity_theta;
    const auto& vorticity_z = diagnostics.vorticity_z;
    const std::vector<float> empty_surface;
    const auto& sfc_sensible_heat_flux = state.surface.sfc_sensible_heat_flux
        ? *state.surface.sfc_sensible_heat_flux : empty_surface;
    const auto& sfc_latent_heat_flux = state.surface.sfc_latent_heat_flux
        ? *state.surface.sfc_latent_heat_flux : empty_surface;
    const auto& sfc_moisture_flux = state.surface.sfc_moisture_flux
        ? *state.surface.sfc_moisture_flux : empty_surface;
    const std::size_t slice_size                               = context.slice_size;
    const float       reflectivity_dbz_min                     = context.reflectivity_dbz_min;
    const float       reflectivity_dbz_max                     = context.reflectivity_dbz_max;
    const float       p0f                                      = static_cast<float>(p0);
    const float       g_f                                      = static_cast<float>(g);
    const auto&       precipitation_rate_surface_mm_h          = context.precipitation_rate_surface_mm_h;
    auto&             accumulated_rainfall_surface_mm          = context.accumulated_rainfall_surface_mm;
    const auto&       accumulated_rainfall_residual_surface_mm = context.accumulated_rainfall_residual_surface_mm;
    double&           accumulated_rainfall_last_update_s       = context.accumulated_rainfall_last_update_s;
    const auto&       storm_u_level_mean                       = context.storm_u_level_mean;
    const auto&       storm_v_level_mean                       = context.storm_v_level_mean;
    const PhysicalCellDomain domain                            = context.physical_domain;
    if (domain.empty() || domain.i_begin < 0 || domain.i_end > NR || domain.j_begin < 0 || domain.j_end > NTH ||
        domain.k_begin < 0 || domain.k_end > NZ)
    {
        error_message = "diagnostic tranche registry received an invalid physical-cell domain";
        return false;
    }
    const int      i_begin    = domain.i_begin;
    const int      i_end      = domain.i_end;
    const int      j_begin    = domain.j_begin;
    const int      j_end      = domain.j_end;
    const int      k_begin    = domain.k_begin;
    const int      k_end      = domain.k_end;
    const int      k_surface  = k_begin;
    const Field3D& u_centered = context.centered_u;
    const Field3D& v_centered = context.centered_v;
    const Field3D& w_centered = context.centered_w;

    const std::size_t polar_size = static_cast<std::size_t>(NR) * static_cast<std::size_t>(NTH);
    if (!precipitation_rate_surface_mm_h.empty() && precipitation_rate_surface_mm_h.size() != polar_size)
    {
        error_message = "surface precipitation-rate state does not match the diagnostic grid";
        return false;
    }
    if (accumulated_rainfall_surface_mm.size() != polar_size)
    {
        accumulated_rainfall_surface_mm.assign(polar_size, 0.0f);
        accumulated_rainfall_last_update_s = simulation_time;
    }
    if (!accumulated_rainfall_residual_surface_mm.empty() &&
        accumulated_rainfall_residual_surface_mm.size() != polar_size)
    {
        error_message = "surface rainfall compensation state does not match the diagnostic grid";
        return false;
    }

    const ColumnWaterPathSelection water_path_selection{
        context.active_output_fields.count("precipitable_water") != 0,
        context.active_output_fields.count("cloud_water_path") != 0,
        context.active_output_fields.count("liquid_water_path") != 0,
        context.active_output_fields.count("vil") != 0,
    };
    std::unique_ptr<ColumnWaterPathCache> water_path_cache;
    if (water_path_selection.any())
    {
        try
        {
            water_path_cache =
                std::make_unique<ColumnWaterPathCache>(context.simulation_state,
                                                       ColumnWaterPathFields{rho, qv, qc, qr, qi, qs, qg, qh},
                                                       water_path_selection,
                                                       k_begin,
                                                       k_end);
        }
        catch (const std::exception& error)
        {
            error_message = "column water-path geometry is invalid: " + std::string(error.what());
            return false;
        }
    }

    const auto polar_index = [&](int i, int th) -> std::size_t
    { return static_cast<std::size_t>(i) * static_cast<std::size_t>(NTH) + static_cast<std::size_t>(th); };
    const auto slice_index = [&](int i, int k) -> std::size_t
    { return static_cast<std::size_t>(k) * static_cast<std::size_t>(NR) + static_cast<std::size_t>(i); };

    const auto center_height = [&](int i, int j, int k) {
        return tmv::physical_grid_metrics::scalar_height(state, i, j, k);
    };
    const auto face_height = [&](int i, int j, int k) {
        return tmv::physical_grid_metrics::interface_height(state, i, j, k);
    };
    const auto height_agl = [&](int i, int j, int k) {
        return center_height(i, j, k) - face_height(i, j, 0);
    };
    const auto sample_column_field = [&](const Field3D& field, int i, int th, float target) -> float
    {
        const auto bracket = bracket_column(k_begin, k_end, target,
            [&](int k) { return height_agl(i, th, k); }, true);
        return static_cast<float>(bracket.sample([&](int k) { return field(i, th, k); }));
    };
    const auto lcl_height = [](float temperature_k, float dewpoint_k) -> float
    {
        return 125.0f * std::max(0.0f, temperature_k - dewpoint_k);
    };
    const auto fill_radius_slice = [&](std::vector<float>& out, const auto& value_at_radius)
    { fill_slice_from_radius(out, NR, NZ, value_at_radius); };

    auto safe_reflectivity_dbz = [&](float z_linear) -> float
    { return linear_z_to_dbz(z_linear, reflectivity_dbz_min, reflectivity_dbz_max); };

    const float storm_u_motion = mean_level_component(storm_u_level_mean, 6000.0f, NZ, dz, k_begin, k_end);
    const float storm_v_motion = mean_level_component(storm_v_level_mean, 6000.0f, NZ, dz, k_begin, k_end);

    auto integrate_srh = [&](int i, int th, float depth_m) -> float
    {
        const double ground = face_height(i, th, 0);
        return static_cast<float>(integrate_storm_relative_helicity(
            k_begin, k_end, ground, ground + depth_m, storm_u_motion, storm_v_motion,
            [&](int k) { return center_height(i, th, k); },
            [&](int k) { return static_cast<double>(u_centered[i][th][k]); },
            [&](int k) { return static_cast<double>(v_centered[i][th][k]); }));
    };

    SurfaceColumnDiagnosticCache surface_column_cache(static_cast<std::size_t>(NR));
    auto&                        surface_pressure_mean_by_radius = surface_column_cache.surface_pressure_mean_by_radius;
    auto& surface_sensible_heat_flux_by_radius = surface_column_cache.surface_sensible_heat_flux_by_radius;
    auto& surface_latent_heat_flux_by_radius   = surface_column_cache.surface_latent_heat_flux_by_radius;
    auto& surface_moisture_flux_by_radius      = surface_column_cache.surface_moisture_flux_by_radius;
    auto& skin_temperature_by_radius           = surface_column_cache.skin_temperature_by_radius;
    auto& cloud_top_height_by_radius           = surface_column_cache.cloud_top_height_by_radius;
    auto& cloud_base_height_by_radius          = surface_column_cache.cloud_base_height_by_radius;
    auto& lcl_by_radius                        = surface_column_cache.lcl_by_radius;
    auto& lfc_by_radius                        = surface_column_cache.lfc_by_radius;
    auto& el_by_radius                         = surface_column_cache.el_by_radius;
    auto& cape_by_radius                       = surface_column_cache.cape_by_radius;
    auto& cin_by_radius                        = surface_column_cache.cin_by_radius;
    auto& lifted_index_by_radius               = surface_column_cache.lifted_index_by_radius;
    auto& k_index_by_radius                    = surface_column_cache.k_index_by_radius;
    auto& showalter_index_by_radius            = surface_column_cache.showalter_index_by_radius;
    auto& total_totals_by_radius               = surface_column_cache.total_totals_by_radius;
    auto& ehi_by_radius                        = surface_column_cache.ehi_by_radius;
    auto& scp_by_radius                        = surface_column_cache.scp_by_radius;
    auto& stp_by_radius                        = surface_column_cache.stp_by_radius;
    auto& cached_surface_column_theta          = surface_column_cache.cached_surface_column_theta;

    for (int i = 0; i < NR; ++i)
    {
        double sum   = 0.0;
        int    count = 0;
        for (int th = j_begin; th < j_end; ++th)
        {
            const float p_surface = static_cast<float>(p[i][th][k_surface]);
            if (std::isfinite(p_surface))
            {
                sum += static_cast<double>(p_surface);
                ++count;
            }
        }
        if (count > 0)
        {
            surface_pressure_mean_by_radius[static_cast<std::size_t>(i)] =
                static_cast<float>(sum / static_cast<double>(count));
        }
    }

    auto compute_precip_rate_mmh = [&](int i, int th) -> float
    {
        if (precipitation_rate_surface_mm_h.empty())
            return 0.0f;
        const float rate = precipitation_rate_surface_mm_h[polar_index(i, th)];
        return std::isfinite(rate) && rate >= 0.0f ? rate : std::numeric_limits<float>::quiet_NaN();
    };

    std::vector<RegisteredDiagnosticProduct> tranche_registry;
    tranche_registry.reserve(40);
    bool tranche_registry_valid = true;

    auto register_tranche_product =
        [&](const char* field_id, std::function<bool(int, std::vector<float>&)> compute_slice)
    {
        const OutputFieldDescriptor* field = find_output_field(field_id);
        if (field == nullptr || !output_field_is_tranche_product(*field))
        {
            error_message = "missing tranche output registry metadata for field='" + std::string(field_id) + "'";
            tranche_registry_valid = false;
            return;
        }
        tranche_registry.push_back({field_id, std::move(compute_slice)});
    };

    auto finite_or = [](float value, float fallback) -> float { return std::isfinite(value) ? value : fallback; };

    auto ensure_surface_column_diagnostics = [&](int th)
    {
        if (cached_surface_column_theta == th)
        {
            return;
        }
        cached_surface_column_theta = th;

        constexpr float kDryLapseKPerM        = 9.8e-3f;
        constexpr float kMoistLapseKPerM      = 6.0e-3f;
        constexpr float kCloudMixingThreshold = 1.0e-5f;
        for (int i = 0; i < NR; ++i)
        {
            const std::size_t idx           = static_cast<std::size_t>(i);
            const double ground = face_height(i, th, 0);
            const float z_top = static_cast<float>(center_height(i, th, k_end - 1) - ground);
            const float parcel_origin = static_cast<float>(height_agl(i, th, k_surface));
            const float       p_surface     = finite_or(static_cast<float>(p[i][th][k_surface]), p0f);
            const float       theta_surface = finite_or(static_cast<float>(theta[i][th][k_surface]), 300.0f);
            const float       t_air = finite_or(temperature_from_theta_and_pressure(theta_surface, p_surface), 300.0f);
            const float       qv_surface = std::max(0.0f, finite_or(static_cast<float>(qv[i][th][k_surface]), 0.0f));

            const float theta2 = sample_column_field(theta, i, th, 2.0f);
            const float p2     = sample_column_field(p, i, th, 2.0f);
            const float t2     = finite_or(temperature_from_theta_and_pressure(theta2, p2), t_air);
            const float qv2    = std::max(0.0f, finite_or(sample_column_field(qv, i, th, 2.0f), qv_surface));
            const float td2    = finite_or(dewpoint_from_temp_qv_pressure(t2, qv2, finite_or(p2, p_surface)), t2);

            const float u10_local = finite_or(sample_column_field(u_centered, i, th, 10.0f), 0.0f);
            const float v10_local = finite_or(sample_column_field(v_centered, i, th, 10.0f), 0.0f);
            const std::size_t sfc_polar =
                static_cast<std::size_t>(i) * static_cast<std::size_t>(NTH) + static_cast<std::size_t>(th);
            const float sensible_flux =
                (sfc_polar < sfc_sensible_heat_flux.size()) ? sfc_sensible_heat_flux[sfc_polar] : 0.0f;
            const float moisture_flux = (sfc_polar < sfc_moisture_flux.size()) ? sfc_moisture_flux[sfc_polar] : 0.0f;
            const float latent_flux =
                (sfc_polar < sfc_latent_heat_flux.size()) ? sfc_latent_heat_flux[sfc_polar] : 0.0f;
            float skin_temperature = t_air;
            if (state.boundary_layer_runtime)
            {
                const auto& pbl = *state.boundary_layer_runtime;
                if (pbl.surface_config.Tsfc > 0.0)
                    skin_temperature = static_cast<float>(pbl.surface_config.Tsfc);
                else if (pbl.skin_captured && sfc_polar < pbl.skin_theta_2d.size())
                    skin_temperature = temperature_from_theta_and_pressure(
                        pbl.skin_theta_2d[sfc_polar], p_surface);
            }
            skin_temperature_by_radius[idx] = skin_temperature;
            surface_sensible_heat_flux_by_radius[idx] = finite_or(sensible_flux, 0.0f);
            surface_moisture_flux_by_radius[idx]      = finite_or(moisture_flux, 0.0f);
            surface_latent_heat_flux_by_radius[idx]   = finite_or(latent_flux, 0.0f);

            float cloud_base  = 0.0f;
            float cloud_top   = 0.0f;
            bool  cloud_found = false;
            float cape        = 0.0f;
            float cin         = 0.0f;

            const float td_surface = finite_or(dewpoint_from_temp_qv_pressure(t_air, qv_surface, p_surface), td2);
            const float lcl        = parcel_origin + lcl_height(t_air, td_surface);
            float       lfc        = 0.0f;
            float       el         = 0.0f;
            bool        lfc_found  = false;
            bool        el_found   = false;

            auto env_temperature_at_k = [&](int k) -> float
            {
                const float pk     = finite_or(static_cast<float>(p[i][th][k]), p_surface);
                const float thetak = finite_or(static_cast<float>(theta[i][th][k]), theta_surface);
                return finite_or(temperature_from_theta_and_pressure(thetak, pk), t_air);
            };

            auto env_dewpoint_at_k = [&](int k, float temp_k) -> float
            {
                const float pk  = finite_or(static_cast<float>(p[i][th][k]), p_surface);
                const float qvk = std::max(0.0f, finite_or(static_cast<float>(qv[i][th][k]), qv_surface));
                return finite_or(dewpoint_from_temp_qv_pressure(temp_k, qvk, pk), temp_k);
            };

            auto parcel_temp_from_surface = [&](float z_m) -> float
            {
                const float dry_depth   = std::max(0.0f, std::min(z_m, lcl) - parcel_origin);
                const float moist_depth = std::max(0.0f, z_m - lcl);
                const float parcel_t    = t_air - (kDryLapseKPerM * dry_depth) - (kMoistLapseKPerM * moist_depth);
                return std::max(150.0f, parcel_t);
            };

            for (int k = k_begin; k < k_end; ++k)
            {
                const float z_m = static_cast<float>(height_agl(i, th, k));
                const float layer_depth = static_cast<float>(face_height(i, th, k + 1) - face_height(i, th, k));
                const float env_t    = env_temperature_at_k(k);
                const float parcel_t = parcel_temp_from_surface(z_m);
                const float buoyancy = g_f * (parcel_t - env_t) / std::max(150.0f, env_t);

                if (std::isfinite(buoyancy))
                {
                    if (!el_found && buoyancy > 0.0f && z_m >= lcl)
                    {
                        cape += buoyancy * layer_depth;
                        if (!lfc_found)
                        {
                            lfc       = z_m;
                            lfc_found = true;
                        }
                    }
                    else if (buoyancy < 0.0f)
                    {
                        if (!lfc_found)
                            cin += buoyancy * layer_depth;
                        if (lfc_found && !el_found)
                        {
                            el       = z_m;
                            el_found = true;
                        }
                    }
                }

                const float qc_k       = std::max(0.0f, finite_or(static_cast<float>(qc[i][th][k]), 0.0f));
                const float qr_k       = std::max(0.0f, finite_or(static_cast<float>(qr[i][th][k]), 0.0f));
                const float qi_k       = std::max(0.0f, finite_or(static_cast<float>(qi[i][th][k]), 0.0f));
                const float qs_k       = std::max(0.0f, finite_or(static_cast<float>(qs[i][th][k]), 0.0f));
                const float qg_k       = std::max(0.0f, finite_or(static_cast<float>(qg[i][th][k]), 0.0f));
                const float qh_k       = std::max(0.0f, finite_or(static_cast<float>(qh[i][th][k]), 0.0f));
                const float condensate = qc_k + qr_k + qi_k + qs_k + qg_k + qh_k;
                if (condensate > kCloudMixingThreshold)
                {
                    if (!cloud_found)
                    {
                        cloud_base  = z_m;
                        cloud_found = true;
                    }
                    cloud_top = z_m;
                }
            }

            if (!lfc_found)
            {
                lfc = 0.0f;
            }
            if (!el_found)
            {
                el = lfc_found ? z_top : 0.0f;
            }
            if (!cloud_found)
            {
                cloud_base = 0.0f;
                cloud_top  = 0.0f;
            }

            const auto sample_pressure_level = [&](double target)
            {
                return bracket_column(k_begin, k_end, -std::log(target),
                    [&](int k) { return -std::log(static_cast<double>(p(i, th, k))); });
            };
            const auto level850 = sample_pressure_level(85000.0);
            const auto level700 = sample_pressure_level(70000.0);
            const auto level500 = sample_pressure_level(50000.0);
            const float z850 = static_cast<float>(level850.sample(
                [&](int k) { return height_agl(i, th, k); }));
            const float z500 = static_cast<float>(level500.sample(
                [&](int k) { return height_agl(i, th, k); }));
            const float t850 = static_cast<float>(level850.sample(env_temperature_at_k));
            const float t700 = static_cast<float>(level700.sample(env_temperature_at_k));
            const float t500 = static_cast<float>(level500.sample(env_temperature_at_k));
            const float td850 = static_cast<float>(level850.sample(
                [&](int k) { return env_dewpoint_at_k(k, env_temperature_at_k(k)); }));
            const float td700 = static_cast<float>(level700.sample(
                [&](int k) { return env_dewpoint_at_k(k, env_temperature_at_k(k)); }));
            const float parcel_t500_surface = parcel_temp_from_surface(z500);

            const float lcl_850         = lcl_height(t850, td850);
            const float z_rel_500       = std::max(0.0f, z500 - z850);
            const float dry_depth_850   = std::min(z_rel_500, lcl_850);
            const float moist_depth_850 = std::max(0.0f, z_rel_500 - lcl_850);
            const float parcel_t500_850 =
                std::max(150.0f, t850 - (kDryLapseKPerM * dry_depth_850) - (kMoistLapseKPerM * moist_depth_850));

            const float srh01          = integrate_srh(i, th, 1000.0f);
            const float srh03          = integrate_srh(i, th, 3000.0f);
            const float u6             = finite_or(sample_column_field(u_centered, i, th, 6000.0f), u10_local);
            const float v6             = finite_or(sample_column_field(v_centered, i, th, 6000.0f), v10_local);
            const float bulk_shear_0_6 = std::hypot(u6 - u10_local, v6 - v10_local);
            const float cin_mag        = std::abs(std::min(0.0f, cin));

            const float lifted_index    = level500.valid ? t500 - parcel_t500_surface
                : std::numeric_limits<float>::quiet_NaN();
            const float k_index         = (t850 - t500) + td850 - (t700 - td700);
            const float showalter_index = level850.valid && level500.valid ? t500 - parcel_t500_850
                : std::numeric_limits<float>::quiet_NaN();
            const float total_totals    = (t850 - t500) + (td850 - t500);
            const float ehi             = std::max(0.0f, cape) * std::max(0.0f, srh01) / 160000.0f;
            const float scp             = std::max(0.0f, cape / 1000.0f) * std::max(0.0f, srh03 / 100.0f) *
                                          std::max(0.0f, bulk_shear_0_6 / 20.0f);
            const float stp = std::max(0.0f, cape / 1500.0f) * std::clamp((2000.0f - lcl) / 1000.0f, 0.0f, 2.0f) *
                              std::max(0.0f, srh01 / 150.0f) * std::clamp((200.0f - cin_mag) / 200.0f, 0.0f, 1.0f);

            cloud_top_height_by_radius[idx]  = finite_or(cloud_top, 0.0f);
            cloud_base_height_by_radius[idx] = finite_or(cloud_base, 0.0f);
            lcl_by_radius[idx]               = finite_or(lcl, 0.0f);
            lfc_by_radius[idx]               = finite_or(lfc, 0.0f);
            el_by_radius[idx]                = finite_or(el, 0.0f);
            cape_by_radius[idx]              = finite_or(std::max(0.0f, cape), 0.0f);
            cin_by_radius[idx]               = finite_or(std::min(0.0f, cin), 0.0f);
            lifted_index_by_radius[idx]      = lifted_index;
            k_index_by_radius[idx]           = k_index;
            showalter_index_by_radius[idx]   = showalter_index;
            total_totals_by_radius[idx]      = total_totals;
            ehi_by_radius[idx]               = finite_or(std::max(0.0f, ehi), 0.0f);
            scp_by_radius[idx]               = finite_or(std::max(0.0f, scp), 0.0f);
            stp_by_radius[idx]               = finite_or(std::max(0.0f, stp), 0.0f);
        }
    };

    auto register_cached_radius_field = [&](const char* field_id, const std::vector<float>* values_by_radius)
    {
        const std::vector<float>* values = values_by_radius;
        register_tranche_product(field_id,
                                 [&, values](int th, std::vector<float>& out)
                                 {
                                     ensure_surface_column_diagnostics(th);
                                     fill_radius_slice(out,
                                                       [&](int i) { return (*values)[static_cast<std::size_t>(i)]; });
                                     return true;
                                 });
    };

    auto register_cached_column_path = [&](const char* field_id, ColumnWaterPathKind kind)
    {
        register_tranche_product(field_id,
                                 [&, kind](int th, std::vector<float>& out)
                                 {
                                     const std::vector<float>& values = water_path_cache->values(kind);
                                     if (!water_path_cache->valid())
                                         return false;
                                     fill_radius_slice(out, [&](int i) { return values[polar_index(i, th)]; });
                                     return true;
                                 });
    };

    register_tranche_product("u10",
                             [&](int th, std::vector<float>& out)
                             {
                                 fill_radius_slice(
                                     out, [&](int i) { return sample_column_field(u_centered, i, th, 10.0f); });
                                 return true;
                             });
    register_tranche_product("v10",
                             [&](int th, std::vector<float>& out)
                             {
                                 fill_radius_slice(
                                     out, [&](int i) { return sample_column_field(v_centered, i, th, 10.0f); });
                                 return true;
                             });
    register_tranche_product("t2",
                             [&](int th, std::vector<float>& out)
                             {
                                 fill_radius_slice(out,
                                                   [&](int i)
                                                   {
                                                       const float theta2 = sample_column_field(theta, i, th, 2.0f);
                                                       const float p2     = sample_column_field(p, i, th, 2.0f);
                                                       return temperature_from_theta_and_pressure(theta2, p2);
                                                   });
                                 return true;
                             });
    register_tranche_product("td2",
                             [&](int th, std::vector<float>& out)
                             {
                                 fill_radius_slice(out,
                                                   [&](int i)
                                                   {
                                                       const float theta2 = sample_column_field(theta, i, th, 2.0f);
                                                       const float p2     = sample_column_field(p, i, th, 2.0f);
                                                       const float t2 = temperature_from_theta_and_pressure(theta2, p2);
                                                       const float qv2 = sample_column_field(qv, i, th, 2.0f);
                                                       return dewpoint_from_temp_qv_pressure(t2, qv2, p2);
                                                   });
                                 return true;
                             });
    register_tranche_product("surface_pressure_perturbation",
                             [&](int th, std::vector<float>& out)
                             {
                                 fill_radius_slice(out,
                                                   [&](int i)
                                                   {
                                                       const float p_surface = static_cast<float>(p[i][th][k_surface]);
                                                       const float p_mean =
                                                           surface_pressure_mean_by_radius[static_cast<std::size_t>(i)];
                                                       if (!std::isfinite(p_surface) || !std::isfinite(p_mean))
                                                       {
                                                           return std::numeric_limits<float>::quiet_NaN();
                                                       }
                                                       return p_surface - p_mean;
                                                   });
                                 return true;
                             });
    // surface_sensible_heat_flux / surface_latent_heat_flux /
    // surface_moisture_flux were previously written by this tranche
    // path as per-theta NR×NZ axisymmetric slabs derived from a
    // toy bulk formula that diverged from the PBL scheme's actual
    // fluxes. They are now written as true 2D (NR×NTH) fields
    // sourced from BoundaryLayerDiagnostics — see the "3D mode:
    // 2D surface-flux fields" block earlier in this function. The
    // tranche registrations are intentionally NOT made here.
    register_cached_radius_field("skin_temperature", &skin_temperature_by_radius);
    if (context.active_output_fields.count("cold_pool_boundary") != 0)
    {
        if (state.grid.geometry.coord != CoordinateSystem::Cartesian)
        {
            error_message =
                "cold_pool_boundary is currently qualified only for Cartesian grids";
            return false;
        }
        const TerrainMetrics3D* const terrain_metrics =
            tmv::physical_grid_metrics::active_terrain_metrics(state);
        if ((terrain_metrics == nullptr &&
             (state.base.theta0.size() <= static_cast<std::size_t>(k_surface) ||
              state.base.qv0.size() <= static_cast<std::size_t>(k_surface))) ||
            (terrain_metrics != nullptr &&
             (state.base.theta0.empty() || state.base.qv0.empty())))
        {
            error_message =
                "cold_pool_boundary requires level-matched theta0 and qv0 base profiles";
            return false;
        }
        const float base_theta = static_cast<float>(
            state.base.theta0[static_cast<std::size_t>(k_surface)]);
        const float base_qv = static_cast<float>(
            state.base.qv0[static_cast<std::size_t>(k_surface)]);
        register_tranche_product(
            "cold_pool_boundary",
            [&, base_theta, base_qv, terrain_metrics](
                int th, std::vector<float>& out)
            {
                if (terrain_metrics != nullptr)
                {
                    std::vector<float> reference_theta(
                        static_cast<std::size_t>(NR));
                    std::vector<float> reference_qv(
                        static_cast<std::size_t>(NR));
                    try
                    {
                        for (int i = 0; i < NR; ++i)
                        {
                            const double physical_height =
                                terrain_metrics->z(i, th, k_surface);
                            reference_theta[static_cast<std::size_t>(i)] =
                                static_cast<float>(terrain_transport::
                                    reference_profile_value_fp32(
                                        state.base.theta0, physical_height,
                                        terrain_metrics->dzeta));
                            reference_qv[static_cast<std::size_t>(i)] =
                                static_cast<float>(terrain_transport::
                                    reference_profile_value_fp32(
                                        state.base.qv0, physical_height,
                                        terrain_metrics->dzeta));
                        }
                    }
                    catch (const std::exception& error)
                    {
                        error_message = std::string(
                            "cold_pool_boundary terrain reference sampling failed: ")
                            + error.what();
                        return false;
                    }
                    return tmv::derived::
                        fill_density_potential_temperature_deficit_surface_slice(
                            state.fields.theta,
                            state.fields.qv,
                            state.fields.qc,
                            state.fields.qr,
                            state.fields.qi,
                            state.fields.qs,
                            state.fields.qg,
                            state.fields.qh,
                            NR, NTH, NZ, th, k_surface,
                            reference_theta, reference_qv,
                            out, error_message);
                }
                return tmv::derived::
                    fill_density_potential_temperature_deficit_surface_slice(
                        state.fields.theta,
                        state.fields.qv,
                        state.fields.qc,
                        state.fields.qr,
                        state.fields.qi,
                        state.fields.qs,
                        state.fields.qg,
                        state.fields.qh,
                        NR,
                        NTH,
                        NZ,
                        th,
                        k_surface,
                        base_theta,
                        base_qv,
                        out,
                        error_message);
            });
    }
    register_tranche_product("precip_rate",
                             [&](int th, std::vector<float>& out)
                             {
                                 fill_radius_slice(out, [&](int i) { return compute_precip_rate_mmh(i, th); });
                                 return true;
                             });
    register_tranche_product("accumulated_rainfall",
                             [&](int th, std::vector<float>& out)
                             {
                                 fill_radius_slice(
                                     out,
                                     [&](int i)
                                     {
                                         const std::size_t idx = polar_index(i, th);
                                         const double      residual =
                                             accumulated_rainfall_residual_surface_mm.empty()
                                                 ? 0.0
                                                 : static_cast<double>(accumulated_rainfall_residual_surface_mm[idx]);
                                         const double exact_total =
                                             static_cast<double>(accumulated_rainfall_surface_mm[idx]) + residual;
                                         if (!std::isfinite(exact_total) || exact_total < 0.0)
                                         {
                                             return std::numeric_limits<float>::quiet_NaN();
                                         }
                                         return static_cast<float>(exact_total);
                                     });
                                 return true;
                             });
    register_tranche_product("composite_reflectivity",
                             [&](int th, std::vector<float>& out)
                             {
                                 fill_radius_slice(out,
                                                   [&](int i)
                                                   {
                                                       float max_dbz = reflectivity_dbz_min;
                                                       for (int k = k_begin; k < k_end; ++k)
                                                       {
                                                           max_dbz = std::max(max_dbz,
                                                                              safe_reflectivity_dbz(static_cast<float>(
                                                                                  radar_reflectivity[i][th][k])));
                                                       }
                                                       return max_dbz;
                                                   });
                                 return true;
                             });
    register_tranche_product("column_max_w",
                             [&](int th, std::vector<float>& out)
                             {
                                 fill_radius_slice(out,
                                                   [&](int i)
                                                   {
                                                       float max_w       = -std::numeric_limits<float>::infinity();
                                                       bool  has_samples = false;
                                                       for (int k = k_begin; k < k_end; ++k)
                                                       {
                                                           if (height_agl(i, th, k) > 1000.0) continue;
                                                           const float sample =
                                                               static_cast<float>(w_centered[i][th][k]);
                                                           if (std::isfinite(sample))
                                                           {
                                                               max_w       = std::max(max_w, sample);
                                                               has_samples = true;
                                                           }
                                                       }
                                                       if (!has_samples)
                                                       {
                                                           return std::numeric_limits<float>::quiet_NaN();
                                                       }
                                                       return max_w;
                                                   });
                                 return true;
                             });
    register_tranche_product(
        "column_max_vorticity",
        [&](int th, std::vector<float>& out)
        {
            fill_radius_slice(out,
                              [&](int i)
                              {
                                  float max_vort    = 0.0f;
                                  bool  has_samples = false;
                                  for (int k = k_begin; k < k_end; ++k)
                                  {
                                      const float vort_r  = static_cast<float>(vorticity_r[i][th][k]);
                                      const float vort_th = static_cast<float>(vorticity_theta[i][th][k]);
                                      const float vort_z  = static_cast<float>(vorticity_z[i][th][k]);
                                      if (std::isfinite(vort_r) && std::isfinite(vort_th) && std::isfinite(vort_z))
                                      {
                                          const float mag =
                                              std::sqrt((vort_r * vort_r) + (vort_th * vort_th) + (vort_z * vort_z));
                                          max_vort    = std::max(max_vort, mag);
                                          has_samples = true;
                                      }
                                  }
                                  if (!has_samples)
                                  {
                                      return std::numeric_limits<float>::quiet_NaN();
                                  }
                                  return max_vort;
                              });
            return true;
        });
    // These four paths share one request-aware traversal of the dry-density
    // state. VIL retains its existing total-condensate composition, but now
    // uses exact physical layer thickness rather than nominal grid.dz.
    register_cached_column_path("vil", ColumnWaterPathKind::TotalCondensate);
    register_cached_column_path("precipitable_water", ColumnWaterPathKind::PrecipitableWater);
    register_cached_column_path("liquid_water_path", ColumnWaterPathKind::LiquidWater);
    register_cached_column_path("cloud_water_path", ColumnWaterPathKind::CloudWater);
    register_cached_radius_field("cloud_top_height", &cloud_top_height_by_radius);
    register_cached_radius_field("cloud_base_height", &cloud_base_height_by_radius);
    register_cached_radius_field("lcl", &lcl_by_radius);
    register_cached_radius_field("lfc", &lfc_by_radius);
    register_cached_radius_field("el", &el_by_radius);
    register_cached_radius_field("cape", &cape_by_radius);
    register_cached_radius_field("cin", &cin_by_radius);
    register_cached_radius_field("lifted_index", &lifted_index_by_radius);
    register_cached_radius_field("k_index", &k_index_by_radius);
    register_cached_radius_field("showalter_index", &showalter_index_by_radius);
    register_cached_radius_field("total_totals", &total_totals_by_radius);
    register_tranche_product("srh_0_1km",
                             [&](int th, std::vector<float>& out)
                             {
                                 fill_radius_slice(out, [&](int i) { return integrate_srh(i, th, 1000.0f); });
                                 return true;
                             });
    register_tranche_product("srh_0_3km",
                             [&](int th, std::vector<float>& out)
                             {
                                 fill_radius_slice(out, [&](int i) { return integrate_srh(i, th, 3000.0f); });
                                 return true;
                             });
    // Updraft helicity: vertical integral of w * vertical vorticity over the
    // 2-5 km AGL layer (Kain et al. 2008) — the standard mesocyclone-rotation
    // / supercell-detection proxy. Signed (cyclonic > 0). w and vorticity_z
    // are in scope here (used by the column tranches above).
    auto integrate_uh = [&](int i, int th) -> float
    {
        const double ground = face_height(i, th, 0);
        const auto integral = integrate_cell_layer(
            k_begin, k_end, ground + 2000.0, ground + 5000.0,
            [&](int k) { return face_height(i, th, k); },
            [&](int k) { return static_cast<double>(w_centered[i][th][k])
                * static_cast<double>(vorticity_z[i][th][k]); });
        return integral.valid ? static_cast<float>(integral.value) : std::numeric_limits<float>::quiet_NaN();
    };
    register_tranche_product("updraft_helicity",
                             [&](int th, std::vector<float>& out)
                             {
                                 fill_radius_slice(out, [&](int i) { return integrate_uh(i, th); });
                                 return true;
                             });
    register_cached_radius_field("ehi", &ehi_by_radius);
    register_cached_radius_field("scp", &scp_by_radius);
    register_cached_radius_field("stp", &stp_by_radius);

    const float ppi_elevation_rad        = 0.5f * static_cast<float>(3.14159265358979323846 / 180.0);

    register_tranche_product(
        "ppi_sweep",
        [&](int th, std::vector<float>& out)
        {
            fill_radius_slice(
                out,
                [&](int i)
                {
                    const float r_m      = static_cast<float>(i) * static_cast<float>(dr);
                    const float z_target = r_m * std::tan(ppi_elevation_rad);
                    const int   k_ppi    = std::clamp(
                        static_cast<int>(std::lround(grid_coordinates::scalar_fractional_index(z_target, dz))),
                        k_begin,
                        k_end - 1);
                    return safe_reflectivity_dbz(static_cast<float>(radar_reflectivity[i][th][k_ppi]));
                });
            return true;
        });
    register_tranche_product(
        "rhi_sweep",
        [&](int th, std::vector<float>& out)
        {
            out.assign(slice_size, reflectivity_dbz_min);
            for (int k = k_begin; k < k_end; ++k)
            {
                for (int i = i_begin; i < i_end; ++i)
                {
                    float sum   = 0.0f;
                    int   count = 0;
                    for (int dk = -1; dk <= 1; ++dk)
                    {
                        const int kk = k + dk;
                        if (kk < k_begin || kk >= k_end)
                        {
                            continue;
                        }
                        for (int di = -1; di <= 1; ++di)
                        {
                            const int ii = i + di;
                            if (ii < i_begin || ii >= i_end)
                            {
                                continue;
                            }
                            const float dbz = safe_reflectivity_dbz(static_cast<float>(radar_reflectivity[ii][th][kk]));
                            if (std::isfinite(dbz))
                            {
                                sum += dbz;
                                ++count;
                            }
                        }
                    }
                    out[slice_index(i, k)] = (count > 0) ? (sum / static_cast<float>(count)) : reflectivity_dbz_min;
                }
            }
            return true;
        });
    register_tranche_product(
        "bwer",
        [&](int th, std::vector<float>& out)
        {
            fill_radius_slice(
                out,
                [&](int i)
                {
                    float low_max   = reflectivity_dbz_min;
                    float mid_min   = 120.0f;
                    float upper_max = reflectivity_dbz_min;
                    for (int k = k_begin; k < k_end; ++k)
                    {
                        const double height = height_agl(i, th, k);
                        const float value = safe_reflectivity_dbz(radar_reflectivity(i, th, k));
                        if (height <= 2000.0) low_max = std::max(low_max, value);
                        if (height >= 3000.0 && height <= 6000.0) mid_min = std::min(mid_min, value);
                        if (height >= 6000.0 && height <= 10000.0) upper_max = std::max(upper_max, value);
                    }
                    if (low_max >= 45.0f && mid_min <= 20.0f && upper_max >= 45.0f)
                    {
                        return 1.0f;
                    }
                    return 0.0f;
                });
            return true;
        });
    register_tranche_product(
        "mesocyclone_diagnostic",
        [&](int th, std::vector<float>& out)
        {
            fill_radius_slice(out,
                              [&](int i)
                              {
                                  float low_vort_max       = 0.0f;
                                  float column_max_updraft = -std::numeric_limits<float>::infinity();
                                  float low_level_vrot     = 0.0f;
                                  bool  has_updraft        = false;
                                  for (int k = k_begin; k < k_end; ++k)
                                  {
                                      const float w_sample = static_cast<float>(w_centered[i][th][k]);
                                      if (std::isfinite(w_sample))
                                      {
                                          column_max_updraft = std::max(column_max_updraft, w_sample);
                                          has_updraft        = true;
                                      }
                                      if (height_agl(i, th, k) <= 3000.0)
                                      {
                                          const float vort_z = std::abs(static_cast<float>(vorticity_z[i][th][k]));
                                          if (std::isfinite(vort_z))
                                          {
                                              low_vort_max = std::max(low_vort_max, vort_z);
                                          }
                                      }
                                      if (height_agl(i, th, k) <= 1000.0)
                                      {
                                          const float vrot = std::abs(static_cast<float>(v_centered[i][th][k]));
                                          if (std::isfinite(vrot))
                                          {
                                              low_level_vrot = std::max(low_level_vrot, vrot);
                                          }
                                      }
                                  }
                                  if (has_updraft && column_max_updraft >= 10.0f && low_vort_max >= 0.01f &&
                                      low_level_vrot >= 20.0f)
                                  {
                                      return 1.0f;
                                  }
                                  return 0.0f;
                              });
            return true;
        });
    register_tranche_product("vrot",
                             [&](int th, std::vector<float>& out)
                             {
                                 fill_radius_slice(out,
                                                   [&](int i)
                                                   {
                                                       float max_vrot = 0.0f;
                                                       for (int k = k_begin; k < k_end; ++k)
                                                       {
                                                           if (height_agl(i, th, k) > 1000.0) continue;
                                                           const float sample =
                                                               std::abs(static_cast<float>(v_centered[i][th][k]));
                                                           if (std::isfinite(sample))
                                                           {
                                                               max_vrot = std::max(max_vrot, sample);
                                                           }
                                                       }
                                                       return max_vrot;
                                                   });
                                 return true;
                             });

    if (!tranche_registry_valid)
    {
        return false;
    }
    return body(tranche_registry);
}

} // namespace diagnostics
} // namespace tmv
