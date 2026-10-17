#include "v0_to_v1_internal.hpp"

#include "init/sounding/parametric_target_math.hpp"
#include "turbulence/turbulence_base.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tmv::init::config::detail
{

void V0ToV1Migrator::migrate_dynamics()
{
    std::string scheme;
    if (!take_string("dynamics.scheme", scheme))
        return;

    config_.dynamics.scheme = scheme; // factory aliases (B1a) resolve the name

    const std::string lower = to_lower(scheme);

    // Decode the implicit axes from scheme names the runtime already accepts.
    bool             axisymmetric = false;
    CoordinateSystem coord        = config_.grid.coordinate_system;
    StaggerType      stag         = config_.grid.staggering;
    bool             saw_known    = false;

    if (lower == "cartesian" || lower == "cart" || lower == "cartesian_cpu" ||
        lower == "klemp_wilhelmson_split_cartesian_agrid")
    {
        axisymmetric = false;
        coord        = CoordinateSystem::Cartesian;
        stag         = StaggerType::Collocated;
        saw_known    = true;
    }
    else if (lower == "klemp_wilhelmson_split_cartesian_cgrid")
    {
        axisymmetric = false;
        coord        = CoordinateSystem::Cartesian;
        stag         = StaggerType::CGrid;
        saw_known    = true;
    }
    else if (lower == "klemp_wilhelmson_split")
    {
        axisymmetric = false;
        coord        = CoordinateSystem::Cartesian;
        // The unqualified solver is intentionally derived from the explicit
        // staggering axis. If absent, ConfigV1's CM1/C-grid default applies.
        saw_known = true;
    }

    if (saw_known)
    {
        config_.dynamics.axisymmetric = axisymmetric;
        // Only override grid axes when the v0 file did NOT set them explicitly.
        // (Explicit v0 settings beat inferred ones — they are the user's intent.)
        if (!has("coordinate_system") && !has("grid.coordinate_system"))
        {
            config_.grid.coordinate_system = coord;
            warn_synth("dynamics.scheme",
                       std::string("inferred grid.coordinate_system=") + coordinate_system_name(coord),
                       "v0 scheme name '" + scheme + "' implies coordinate system");
        }
        if (!has("grid.staggering"))
        {
            config_.grid.staggering = stag;
            warn_synth("dynamics.scheme",
                       std::string("inferred grid.staggering=") + stagger_type_name(stag),
                       "v0 scheme name '" + scheme + "' implies grid staggering");
        }
        warn_synth("dynamics.scheme",
                   "set dynamics.axisymmetric=" + std::string(axisymmetric ? "true" : "false"),
                   "v0 scheme name '" + scheme + "' encodes axisymmetric flag implicitly");
    }
    else
    {
        warn_synth("dynamics.scheme",
                   "preserved as-is",
                   "v0 scheme '" + scheme +
                       "' is not a known legacy name; "
                       "B1a factory aliases will attempt to resolve it");
    }
}

// ─────────────────────────────────────────────────────────────
// microphysics.scheme — enum mapping
// ─────────────────────────────────────────────────────────────

void V0ToV1Migrator::migrate_microphysics()
{
    std::string scheme;
    if (!take_string("microphysics.scheme", scheme))
        return;
    MicrophysicsScheme m = MicrophysicsScheme::None;
    if (scheme == "morrison")
    {
        config_.microphysics.scheme = MicrophysicsScheme::Milbrandt;
        warn_synth(
            "microphysics.scheme",
            "migrated to 'milbrandt'",
            "legacy Morrison spelling selected Nimbus's historical "
            "Milbrandt slot; Morrison and Milbrandt-Yau are distinct "
            "physical schemes");
    }
    else if (parse_microphysics_scheme(scheme, m))
    {
        config_.microphysics.scheme = m;
    }
    else
    {
        error(line_of("microphysics.scheme"), "microphysics.scheme", "unknown microphysics scheme '" + scheme + "'");
    }

    const auto migrate_non_negative = [&](const char* key, double& destination, bool active)
    {
        double value = 0.0;
        if (!take_double(key, value)) return;
        if (!active)
        {
            warn_drop(key, "parameter is inactive for the selected microphysics scheme");
            return;
        }
        if (!std::isfinite(value) || value < 0.0)
        {
            error(line_of(key), key, "must be a finite non-negative number");
            return;
        }
        destination = value;
    };
    const bool wet = config_.microphysics.scheme != MicrophysicsScheme::None;
    migrate_non_negative("microphysics.shared.cloud_water_threshold_kgkg",
                         config_.microphysics.parameters.shared.cloud_water_threshold_kgkg, wet);
    migrate_non_negative("microphysics.shared.autoconversion_rate_s_inv",
                         config_.microphysics.parameters.shared.autoconversion_rate_s_inv, wet);
    migrate_non_negative("microphysics.kessler.accretion_coefficient",
                         config_.microphysics.parameters.kessler.accretion_coefficient,
                         config_.microphysics.scheme == MicrophysicsScheme::Kessler);
    migrate_non_negative("microphysics.kessler.evaporation_efficiency",
                         config_.microphysics.parameters.kessler.evaporation_efficiency,
                         config_.microphysics.scheme == MicrophysicsScheme::Kessler);
    double obsolete_kessler_evaporation_rate = 0.0;
    if (take_double("microphysics.kessler.evaporation_rate_s_inv",
                    obsolete_kessler_evaporation_rate))
    {
        warn_drop(
            "microphysics.kessler.evaporation_rate_s_inv",
            "obsolete first-order rate cannot be converted dimensionally to "
            "the reference-kernel evaporation_efficiency; using 1.0");
    }
    migrate_non_negative("microphysics.lin.accretion_coefficient",
                         config_.microphysics.parameters.lin.accretion_coefficient,
                         config_.microphysics.scheme == MicrophysicsScheme::Lin);
    migrate_non_negative("microphysics.lin.evaporation_rate_s_inv",
                         config_.microphysics.parameters.lin.evaporation_rate_s_inv,
                         config_.microphysics.scheme == MicrophysicsScheme::Lin);
    migrate_non_negative("microphysics.thompson.evaporation_rate_s_inv",
                         config_.microphysics.parameters.thompson.evaporation_rate_s_inv,
                         config_.microphysics.scheme == MicrophysicsScheme::Thompson);
    double obsolete_milbrandt_evaporation = 0.0;
    if (take_double(
            "microphysics.milbrandt.evaporation_rate_s_inv",
            obsolete_milbrandt_evaporation))
    {
        warn_drop(
            "microphysics.milbrandt.evaporation_rate_s_inv",
            "the fixed-rate control was never part of the published "
            "Milbrandt-Yau evaporation equation");
    }
}

// ─────────────────────────────────────────────────────────────
// radiation.* — scheme kept, everything else dropped
// ─────────────────────────────────────────────────────────────

void V0ToV1Migrator::migrate_radiation()
{
    std::string scheme;
    if (take_string("radiation.scheme", scheme))
    {
        RadiationScheme r = RadiationScheme::None;
        if (parse_radiation_scheme(scheme, r))
        {
            config_.radiation.scheme = r;
        }
        else
        {
            error(line_of("radiation.scheme"), "radiation.scheme", "unknown radiation scheme '" + scheme + "'");
        }
    }
    const char* dropped[] = {
        "radiation.dt",
        "radiation.do_lw",
        "radiation.do_sw",
        "radiation.lw",
        "radiation.sw",
        "radiation.clear_sky_only",
        "radiation.use_subcycling",
        "radiation.enable_diagnostics",
        "radiation.tau_lw_ref",
        "radiation.tau_sw_ref",
        "radiation.n_lw",
        "radiation.n_sw",
        "radiation.terrain_sw_cosine_nodes",
        "radiation.terrain_sw_azimuth_nodes",
        "radiation.terrain_sw_maximum_bounces",
        "radiation.terrain_sw_relative_residual_tolerance",
        "radiation.terrain_sw_absolute_residual_tolerance_w",
    };
    for (const char* k : dropped)
    {
        if (has(k))
            warn_drop(k, "radiation scheme-specific knob; not exposed in v1 §20");
    }
}

// ─────────────────────────────────────────────────────────────
// boundary_layer.* + surface.* — scheme kept; flux toggle moves to surface
// ─────────────────────────────────────────────────────────────

void V0ToV1Migrator::migrate_boundary_layer_and_surface()
{
    std::string scheme;
    if (take_string("boundary_layer.scheme", scheme))
    {
        BoundaryLayerScheme bl = BoundaryLayerScheme::None;
        if (parse_boundary_layer_scheme(scheme, bl))
        {
            config_.boundary_layer.scheme = bl;
            // v0 boundary_layer.scheme=none meant "no PBL AND no surface
            // fluxes" (v0 slab was the only surface-flux path, and it ran
            // iff boundary_layer.scheme=slab). The v1 schema default of
            // surface.scheme=Slab would otherwise round-trip a v0 PBL=none
            // config into a v0 boundary_layer.scheme=slab — silently
            // enabling fluxes the v0 author had explicitly disabled. Force
            // surface=None to preserve intent.
            if (bl == BoundaryLayerScheme::None)
            {
                config_.surface.scheme       = SurfaceScheme::None;
                config_.surface.apply_fluxes = false;
            }
            else if (bl == BoundaryLayerScheme::Slab)
            {
                // v0 bundled the slab mixed-layer closure and its surface
                // model under one id. v1 represents both axes explicitly.
                config_.surface.scheme = SurfaceScheme::Slab;
            }
        }
        else
        {
            // Preserve the runtime factory's historical mixed-layer aliases,
            // but serialize them back to the canonical v1 spelling `slab`.
            const std::string lower = to_lower(scheme);
            if (lower == "mixed_layer" || lower == "mixed-layer")
            {
                config_.surface.scheme        = SurfaceScheme::Slab;
                config_.boundary_layer.scheme = BoundaryLayerScheme::Slab;
                warn_synth("boundary_layer.scheme",
                           "canonicalized mixed-layer alias → boundary_layer.scheme=slab",
                           "v1 uses `slab` for the prognostic mixed-layer closure; "
                           "the old alias remains accepted only by the legacy runtime factory");
            }
            else
            {
                error(line_of("boundary_layer.scheme"),
                      "boundary_layer.scheme",
                      "unknown boundary_layer scheme '" + scheme + "'");
            }
        }
    }

    // apply_surface_fluxes / surface_fluxes both map to surface.apply_fluxes
    bool b;
    if (take_bool("boundary_layer.apply_surface_fluxes", b))
    {
        config_.surface.apply_fluxes = b;
        warn_rename("boundary_layer.apply_surface_fluxes", "surface.apply_fluxes");
    }
    if (take_bool("boundary_layer.surface_fluxes", b))
    {
        config_.surface.apply_fluxes = b;
        warn_rename("boundary_layer.surface_fluxes", "surface.apply_fluxes");
    }

    // Prescribed skin state (CM1 tsk0): v0 surface.tsfc/Tsfc -> v1 surface.tsk_k,
    // v0 surface.qsfc/qv_sfc -> v1 surface.qsfc_kgkg.
    double d;
    bool   got_tsk = take_double("surface.tsfc", d);
    if (!got_tsk)
        got_tsk = take_double("surface.Tsfc", d);
    if (got_tsk)
    {
        config_.surface.tsk_k = d;
        warn_rename("surface.tsfc", "surface.tsk_k");
    }
    bool got_qsfc = take_double("surface.qsfc", d);
    if (!got_qsfc)
        got_qsfc = take_double("surface.qv_sfc", d);
    if (got_qsfc)
    {
        config_.surface.qsfc_kgkg = d;
        warn_rename("surface.qsfc", "surface.qsfc_kgkg");
    }
    if (take_double("surface.z0m", d))
    {
        config_.surface.z0m_m = d;
        warn_rename("surface.z0m", "surface.z0m_m");
    }
    if (take_double("surface.z0h", d))
    {
        config_.surface.z0h_m = d;
        warn_rename("surface.z0h", "surface.z0h_m");
    }
    take_double("surface.albedo", config_.surface.albedo);
    take_double("surface.emissivity", config_.surface.emissivity);

    std::string top_method;
    if (take_string("boundary_layer.pbl_top_method", top_method))
    {
        BoundaryLayerTopMethod parsed = BoundaryLayerTopMethod::Richardson;
        if (parse_boundary_layer_top_method(top_method, parsed))
            config_.boundary_layer.pbl_top_method = parsed;
        else
            error(line_of("boundary_layer.pbl_top_method"),
                  "boundary_layer.pbl_top_method",
                  "expected one of ri, theta, tke");
    }

    if (take_bool("boundary_layer.clear_sky_only", b))
    {
        config_.boundary_layer.clear_sky_only = b;
    }
    if (take_double("boundary_layer.dt_pbl", d))
    {
        config_.boundary_layer.cadence_s = d;
        warn_rename("boundary_layer.dt_pbl", "boundary_layer.cadence_s");
    }
    else if (take_double("boundary_layer.dt", d))
    {
        config_.boundary_layer.cadence_s = d;
        warn_rename("boundary_layer.dt", "boundary_layer.cadence_s");
    }

    const char* dropped[] = {
        "boundary_layer.enable_nonlocal_transport",
        "boundary_layer.enable_tke",
        "boundary_layer.surface_layer",
        "boundary_layer.pbl_max_height",
        "boundary_layer.min_ustar",
    };
    for (const char* k : dropped)
    {
        if (has(k))
            warn_drop(k, "scheme-specific knob; not exposed in v1 §20");
    }
}

// ─────────────────────────────────────────────────────────────
// turbulence.* — scheme + mode + Cs/Pr_t/Sc_t preserved
// ─────────────────────────────────────────────────────────────

void V0ToV1Migrator::migrate_turbulence()
{
    std::string scheme;
    if (take_string("turbulence.scheme", scheme))
    {
        TurbulenceScheme t = TurbulenceScheme::None;
        if (parse_turbulence_scheme(scheme, t))
        {
            config_.turbulence.scheme = t;
        }
        else
        {
            error(line_of("turbulence.scheme"), "turbulence.scheme", "unknown turbulence scheme '" + scheme + "'");
        }
    }

    std::string mode;
    if (take_string("turbulence.mode", mode))
    {
        TurbulenceMode m = TurbulenceMode::ThreeD;
        if (parse_turbulence_mode(mode, m))
        {
            config_.turbulence.mode = m;
        }
        else
        {
            error(line_of("turbulence.mode"), "turbulence.mode", "unknown turbulence mode '" + mode + "'");
        }
    }

    take_double("turbulence.Cs", config_.turbulence.Cs);
    take_double("turbulence.Pr_t", config_.turbulence.Pr_t);
    take_double("turbulence.Sc_t", config_.turbulence.Sc_t);

    std::string filter_width;
    if (take_string("turbulence.filter_width", filter_width))
    {
        TurbulenceFilterWidth value = TurbulenceFilterWidth::CubicRoot;
        if (parse_turbulence_filter_width(filter_width, value))
            config_.turbulence.filter_width = value;
        else
            error(line_of("turbulence.filter_width"),
                  "turbulence.filter_width",
                  "unknown turbulence filter width '" + filter_width + "'");
    }
    take_double("turbulence.user_filter_width_m",
                config_.turbulence.user_filter_width_m);
    take_bool("turbulence.dynamic_Cs",
              config_.turbulence.dynamic_coefficient);
    std::string stability_correction;
    if (take_string("turbulence.stability_correction", stability_correction))
    {
        TurbulenceStabilityCorrection value =
            TurbulenceStabilityCorrection::None;
        if (to_lower(stability_correction) == tmv::kLillyRichardsonModel)
            config_.turbulence.stability_correction = TurbulenceStabilityCorrection::Richardson;
        else if (parse_turbulence_stability_correction(stability_correction, value))
        {
            if (value == TurbulenceStabilityCorrection::Richardson)
                error(line_of("turbulence.stability_correction"), "turbulence.stability_correction",
                      "legacy linear Ri has no lossless v1 equivalent; select the Lilly model explicitly for a new run");
            else
                config_.turbulence.stability_correction = value;
        }
        else
            error(line_of("turbulence.stability_correction"),
                  "turbulence.stability_correction",
                  "unknown turbulence stability correction '" +
                      stability_correction + "'");
    }

    const char* dropped[] = {
        "turbulence.dt",
        "turbulence.nu_t_max",
        "turbulence.K_max",
        "turbulence.enable_moist_buoyancy",
        "turbulence.enable_diagnostics",
    };
    for (const char* k : dropped)
    {
        if (has(k))
            warn_drop(k, "scheme-specific knob; not exposed in v1 §20");
    }
}

// ─────────────────────────────────────────────────────────────
// numerics.* — split across time / diffusion / extensions.compute
// ─────────────────────────────────────────────────────────────

} // namespace tmv::init::config::detail
