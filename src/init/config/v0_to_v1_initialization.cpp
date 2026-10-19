#include "v0_to_v1_internal.hpp"

#include "init/sounding/parametric_target_math.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tmv::init::config::detail
{

void V0ToV1Migrator::migrate_top_level()
{
    std::string coord;
    if (take_string("coordinate_system", coord))
    {
        CoordinateSystem cs = CoordinateSystem::Cartesian;
        if (parse_coordinate_system(coord, cs))
        {
            config_.grid.coordinate_system = cs;
            warn_rename("coordinate_system", "grid.coordinate_system");
        }
        else
        {
            error(line_of("coordinate_system"), "coordinate_system", "unknown value '" + coord + "'");
        }
    }

    double dur = 0.0;
    if (take_double("duration_s", dur))
    {
        if (!valid_runtime_seconds(dur, false))
        {
            error(line_of("duration_s"),
                  "duration_s",
                  "must be a positive whole number of seconds within the runtime integer range");
        }
        else
        {
            config_.time.run_time = dur;
            warn_rename("duration_s", "time.run_time");
        }
    }

    if (has("case_name"))
    {
        warn_drop("case_name", "metadata not part of v1 schema");
    }
}

// ─────────────────────────────────────────────────────────────
// grid.* and the cross-section move grid.dt → time.dt_large
// ─────────────────────────────────────────────────────────────

void V0ToV1Migrator::migrate_grid_and_time()
{
    std::string stag;
    if (take_string("grid.staggering", stag))
    {
        StaggerType st = StaggerType::CGrid;
        if (parse_stagger_type(stag, st))
        {
            config_.grid.staggering = st;
        }
        else
        {
            error(line_of("grid.staggering"), "grid.staggering", "unknown value '" + stag + "'");
        }
    }

    take_int("grid.nx", config_.grid.nx);
    take_int("grid.ny", config_.grid.ny);
    take_int("grid.nz", config_.grid.nz);
    take_double("grid.dx", config_.grid.dx);
    take_double("grid.dy", config_.grid.dy);
    take_double("grid.dz", config_.grid.dz);

    double dt = 0.0;
    if (take_double("grid.dt", dt))
    {
        config_.time.dt_large = dt;
        warn_rename("grid.dt", "time.dt_large");
    }
}

// ─────────────────────────────────────────────────────────────
// environment.* — split into sounding + hodograph + dropped extras
// ─────────────────────────────────────────────────────────────

void V0ToV1Migrator::migrate_environment()
{
    auto& s = config_.initial_conditions.sounding;

    // Top-level environment scalars feeding sounding defaults
    double v;
    if (take_double("environment.sfc_theta_k", v))
    {
        s.surface_theta_k = v;
        warn_rename("environment.sfc_theta_k", "initial_conditions.sounding.surface_theta_k");
    }
    if (take_double("environment.sfc_qv_kgkg", v))
    {
        s.moisture_qv_kgkg = v;
        warn_rename("environment.sfc_qv_kgkg", "initial_conditions.sounding.moisture_qv_kgkg");
    }
    if (take_double("environment.cape_target_jkg", v))
    {
        s.cape_jkg = v;
        warn_rename("environment.cape_target_jkg", "initial_conditions.sounding.cape_jkg");
    }
    if (take_double("environment.tropopause_z_m", v))
    {
        s.tropopause_m = v;
        warn_rename("environment.tropopause_z_m", "initial_conditions.sounding.tropopause_m");
    }
    if (has("environment.shear_0_6km_ms"))
    {
        warn_drop("environment.shear_0_6km_ms",
                  "v1 hodograph is 4-point custom or WK-parametric; "
                  "no scalar shear input. Set initial_conditions.hodograph.u_6km_ms manually.");
    }
    if (has("environment.midlevel_drying"))
    {
        warn_drop("environment.midlevel_drying",
                  "no v1 equivalent (Nimbus extension; remove or open an enhancement request)");
    }

    migrate_sounding_subsection();
    migrate_hodograph_subsection();
}

void V0ToV1Migrator::migrate_sounding_subsection()
{
    auto& s = config_.initial_conditions.sounding;

    std::string type_str;
    if (take_string("environment.sounding.type", type_str))
    {
        SoundingType t = SoundingType::ParametricCAPE;
        if (parse_sounding_type(type_str, t))
        {
            s.type = t;
            warn_rename("environment.sounding.type", "initial_conditions.sounding.type");
        }
        else
        {
            // Some v0 configs use "parametric" or other variants — best guess.
            const std::string lower = to_lower(type_str);
            if (lower.find("parametric") != std::string::npos)
            {
                s.type = SoundingType::ParametricCAPE;
                warn_synth("environment.sounding.type",
                           "best-guess: parametric_cape",
                           "v0 value '" + type_str + "' normalized to v1 parametric_cape");
            }
            else if (lower.find("file") != std::string::npos)
            {
                s.type = SoundingType::File;
                warn_synth("environment.sounding.type",
                           "best-guess: file",
                           "v0 value '" + type_str + "' normalized to v1 file");
            }
            else
            {
                error(line_of("environment.sounding.type"),
                      "environment.sounding.type",
                      "unknown sounding type '" + type_str + "'");
            }
        }
    }

    std::string fp;
    if (take_string("environment.sounding.file_path", fp))
    {
        if (!fp.empty())
        {
            s.file_path = fp;
            warn_rename("environment.sounding.file_path", "initial_conditions.sounding.file_path");
        }
        // Empty file_path in v0 was a no-op convention. Just consume.
    }

    // environment.sounding.targets.* override the top-level environment.*
    // scalars when both are present (sounding's targets are the more precise
    // statement of intent in v0 — they go straight into the sounding source).
    double v;
    if (take_double("environment.sounding.targets.cape_jkg", v))
    {
        s.cape_jkg = v;
        warn_rename("environment.sounding.targets.cape_jkg", "initial_conditions.sounding.cape_jkg");
    }
    if (take_double("environment.sounding.targets.cin_jkg", v))
    {
        s.cin_jkg = v;
        warn_rename("environment.sounding.targets.cin_jkg", "initial_conditions.sounding.cin_jkg");
    }
    if (take_double("environment.sounding.targets.lcl_m", v))
    {
        s.lcl_m = v;
        warn_rename("environment.sounding.targets.lcl_m", "initial_conditions.sounding.lcl_m");
    }
    if (take_double("environment.sounding.targets.lfc_m", v))
    {
        s.lfc_m = v;
        warn_rename("environment.sounding.targets.lfc_m", "initial_conditions.sounding.lfc_m");
    }
    if (take_double("environment.sounding.targets.el_m", v))
    {
        s.el_m = v;
        warn_rename("environment.sounding.targets.el_m", "initial_conditions.sounding.el_m");
    }
    if (take_double("environment.sounding.targets.surface_theta_k", v))
    {
        s.surface_theta_k = v;
        warn_rename("environment.sounding.targets.surface_theta_k", "initial_conditions.sounding.surface_theta_k");
    }
    if (take_double("environment.sounding.targets.moisture_qv_kgkg", v))
    {
        s.moisture_qv_kgkg = v;
        warn_rename("environment.sounding.targets.moisture_qv_kgkg", "initial_conditions.sounding.moisture_qv_kgkg");
    }

    // Drop the Nimbus-specific sounding plumbing knobs.
    const char* dropped_sounding[] = {
        "environment.sounding.enabled",
        "environment.sounding.use_fallback_profiles",
        "environment.sounding.interpolation_method",
        "environment.sounding.extrapolate_below_ground",
        "environment.sounding.extrapolate_above_top",
        "environment.sounding.min_pressure_hpa",
        "environment.sounding.max_pressure_hpa",
        "environment.sounding.min_temperature_k",
        "environment.sounding.max_temperature_k",
        "environment.sounding.allow_placeholder_profiles",
        "environment.sounding.require_winds",
    };
    for (const char* k : dropped_sounding)
    {
        if (has(k))
        {
            warn_drop(k, "Nimbus-specific sounding-loader knob; no v1 equivalent");
        }
    }
}

void V0ToV1Migrator::migrate_hodograph_subsection()
{
    auto& h = config_.initial_conditions.hodograph;

    std::string type_str;
    if (take_string("environment.hodograph.type", type_str))
    {
        HodographType t = HodographType::Custom;
        if (parse_hodograph_type(type_str, t))
        {
            h.type = t;
            warn_rename("environment.hodograph.type", "initial_conditions.hodograph.type");
        }
        else
        {
            // v0 used "auto" / "zero" / "wk_param" / "custom" — map best-fit.
            const std::string lower = to_lower(type_str);
            if (lower == "auto")
            {
                h.type = HodographType::Auto;
                warn_synth("environment.hodograph.type",
                           "auto",
                           "v0 auto preserves file-sounding winds and falls "
                           "back to the configured anchors");
            }
            else if (lower == "wk_param")
            {
                h.type = HodographType::WKParam;
                warn_synth("environment.hodograph.type",
                           "best-guess: wk_param",
                           "v0 value '" + type_str + "' normalized to v1 wk_param");
            }
            else if (lower == "zero" || lower == "custom")
            {
                h.type = HodographType::Custom;
                warn_synth("environment.hodograph.type",
                           "best-guess: custom",
                           "v0 value '" + type_str +
                               "' normalized to v1 custom "
                               "(zero hodograph = custom with all wind keys = 0)");
            }
            else
            {
                error(line_of("environment.hodograph.type"),
                      "environment.hodograph.type",
                      "unknown hodograph type '" + type_str + "'");
            }
        }
    }

    const char* wind_keys[6][2] = {
        {"environment.hodograph.u_sfc_ms", "initial_conditions.hodograph.u_sfc_ms"},
        {"environment.hodograph.v_sfc_ms", "initial_conditions.hodograph.v_sfc_ms"},
        {"environment.hodograph.u_1km_ms", "initial_conditions.hodograph.u_1km_ms"},
        {"environment.hodograph.v_1km_ms", "initial_conditions.hodograph.v_1km_ms"},
        {"environment.hodograph.u_6km_ms", "initial_conditions.hodograph.u_6km_ms"},
        {"environment.hodograph.v_6km_ms", "initial_conditions.hodograph.v_6km_ms"},
    };
    double* wind_targets[6] = {
        &h.u_sfc_ms,
        &h.v_sfc_ms,
        &h.u_1km_ms,
        &h.v_1km_ms,
        &h.u_6km_ms,
        &h.v_6km_ms,
    };
    for (int i = 0; i < 6; ++i)
    {
        double v;
        if (take_double(wind_keys[i][0], v))
        {
            *wind_targets[i] = v;
            warn_rename(wind_keys[i][0], wind_keys[i][1]);
        }
    }
}

// ─────────────────────────────────────────────────────────────
// trigger.* — type + per-type sub-blocks; km→m conversion
// ─────────────────────────────────────────────────────────────

void V0ToV1Migrator::migrate_trigger()
{
    auto& tr = config_.initial_conditions.trigger;

    std::string type_str;
    bool        had_type = take_string("trigger.type", type_str);
    if (had_type)
    {
        // Normalize legacy names.
        const std::string lower = to_lower(type_str);
        if (lower == "warm_bubble" || lower == "warm")
        {
            tr.type = TriggerType::Bubble;
            warn_synth("trigger.type",
                       "renamed to initial_conditions.trigger.type (value 'bubble')",
                       "v0 'warm_bubble'/'warm' aliases unified to v1 'bubble'");
        }
        else
        {
            TriggerType t = TriggerType::None;
            if (parse_trigger_type(type_str, t))
            {
                tr.type = t;
                warn_rename("trigger.type", "initial_conditions.trigger.type");
            }
            else
            {
                error(line_of("trigger.type"), "trigger.type", "unknown trigger type '" + type_str + "'");
            }
        }
    }

    // Bubble sub-block. v0 uses km for center coords; v1 uses m. Multiply.
    double v;
    if (take_double("trigger.bubble.dtheta_k", v))
    {
        tr.bubble.dtheta_k = v;
        warn_rename("trigger.bubble.dtheta_k", "initial_conditions.trigger.bubble.dtheta_k");
    }
    if (take_double("trigger.bubble.cloud_water_kgkg", v))
    {
        tr.bubble.cloud_water_kgkg = v;
        warn_rename(
            "trigger.bubble.cloud_water_kgkg",
            "initial_conditions.trigger.bubble.cloud_water_kgkg");
    }
    if (take_double("trigger.bubble.radius_km", v))
    {
        tr.bubble.radius_km = v;
        warn_rename("trigger.bubble.radius_km", "initial_conditions.trigger.bubble.radius_km");
    }
    if (take_double("trigger.bubble.center_x_km", v))
    {
        tr.bubble.center_x_m = v * 1000.0;
        warn_unit("trigger.bubble.center_x_km", "initial_conditions.trigger.bubble.center_x_m", "km → m (×1000)");
    }
    if (take_double("trigger.bubble.center_y_km", v))
    {
        tr.bubble.center_y_m = v * 1000.0;
        warn_unit("trigger.bubble.center_y_km", "initial_conditions.trigger.bubble.center_y_m", "km → m (×1000)");
    }
    if (take_double("trigger.bubble.center_z_km", v))
    {
        tr.bubble.center_z_m = v * 1000.0;
        warn_unit("trigger.bubble.center_z_km", "initial_conditions.trigger.bubble.center_z_m", "km → m (×1000)");
    }

    // Vortex seed sub-block — center is km in v0, m in v1.
    if (take_double("trigger.vortex_seed.r_max_m", v))
    {
        tr.vortex_seed.r_max_m = v;
        warn_rename("trigger.vortex_seed.r_max_m", "initial_conditions.trigger.vortex_seed.r_max_m");
    }
    if (take_double("trigger.vortex_seed.v_max_ms", v))
    {
        tr.vortex_seed.v_max_ms = v;
        warn_rename("trigger.vortex_seed.v_max_ms", "initial_conditions.trigger.vortex_seed.v_max_ms");
    }
    if (take_double("trigger.vortex_seed.z_top_m", v))
    {
        tr.vortex_seed.z_top_m = v;
        warn_rename("trigger.vortex_seed.z_top_m", "initial_conditions.trigger.vortex_seed.z_top_m");
    }
    if (take_double("trigger.vortex_seed.center_x_km", v))
    {
        tr.vortex_seed.center_x_m = v * 1000.0;
        warn_unit(
            "trigger.vortex_seed.center_x_km", "initial_conditions.trigger.vortex_seed.center_x_m", "km → m (×1000)");
    }
    if (take_double("trigger.vortex_seed.center_y_km", v))
    {
        tr.vortex_seed.center_y_m = v * 1000.0;
        warn_unit(
            "trigger.vortex_seed.center_y_km", "initial_conditions.trigger.vortex_seed.center_y_m", "km → m (×1000)");
    }
    if (take_double("trigger.vortex_seed.center_z_km", v))
    {
        tr.vortex_seed.center_z_m = v * 1000.0;
        warn_unit(
            "trigger.vortex_seed.center_z_km", "initial_conditions.trigger.vortex_seed.center_z_m", "km → m (×1000)");
    }

    // v0 default: if no trigger.type and trigger.bubble.* present, type=warm_bubble.
    // Match this convention so configs that rely on the default produce sane v1 output.
    if (!had_type && (has("trigger.bubble.dtheta_k") ||
                      has("trigger.bubble.cloud_water_kgkg") ||
                      has("trigger.bubble.radius_km")))
    {
        tr.type = TriggerType::Bubble;
        warn_synth("trigger",
                   "synthesized trigger.type=bubble",
                   "v0 default: bubble sub-keys present without explicit type → "
                   "treated as warm_bubble in v0; explicitly set to 'bubble' in v1");
    }
}

// ─────────────────────────────────────────────────────────────
// dynamics.scheme — decompose into scheme + axisymmetric + grid axes
// ─────────────────────────────────────────────────────────────

} // namespace tmv::init::config::detail
