#include "v0_to_v1_internal.hpp"

#include "init/sounding/parametric_target_math.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tmv::init::config::detail
{

void V0ToV1Migrator::warn_drop(const std::string& v0_key, const std::string& reason)
{
    warnings_.push_back({v0_key, line_of(v0_key), "dropped", reason});
    consumed_.insert(v0_key);
}

void V0ToV1Migrator::warn_rename(const std::string& v0_key, const std::string& v1_key)
{
    warnings_.push_back(
        {v0_key, line_of(v0_key), "renamed to " + v1_key, "v0 key path differs from v1; value preserved"});
}

void V0ToV1Migrator::warn_unit(const std::string& v0_key, const std::string& v1_key, const std::string& conversion)
{
    warnings_.push_back({v0_key,
                         line_of(v0_key),
                         "renamed to " + v1_key + " (" + conversion + ")",
                         "value unit-converted during rename"});
}

void V0ToV1Migrator::warn_synth(const std::string& v0_key, const std::string& action, const std::string& reason)
{
    warnings_.push_back({v0_key, line_of(v0_key), action, reason});
}

void V0ToV1Migrator::error(int line, const std::string& key, const std::string& message)
{
    errors_.push_back({file_, line, key, message});
}

void V0ToV1Migrator::consume_silent(const std::string& v0_key)
{
    if (has(v0_key))
        consumed_.insert(v0_key);
}

bool V0ToV1Migrator::take_string(const std::string& v0_key, std::string& out)
{
    auto it = yaml_.find(v0_key);
    if (it == yaml_.end())
        return false;
    consumed_.insert(v0_key);
    out = it->second.value;
    return true;
}

bool V0ToV1Migrator::take_int(const std::string& v0_key, int& out)
{
    auto it = yaml_.find(v0_key);
    if (it == yaml_.end())
        return false;
    consumed_.insert(v0_key);
    int parsed = 0;
    if (!try_parse_int_value(it->second.value, parsed))
    {
        error(it->second.line_number, v0_key, "cannot parse '" + it->second.value + "' as integer");
        return false;
    }
    out = parsed;
    return true;
}

bool V0ToV1Migrator::take_double(const std::string& v0_key, double& out)
{
    auto it = yaml_.find(v0_key);
    if (it == yaml_.end())
        return false;
    consumed_.insert(v0_key);
    double parsed = 0.0;
    if (!try_parse_double_value(it->second.value, parsed))
    {
        error(it->second.line_number, v0_key, "cannot parse '" + it->second.value + "' as number");
        return false;
    }
    out = parsed;
    return true;
}

bool V0ToV1Migrator::take_bool(const std::string& v0_key, bool& out)
{
    auto it = yaml_.find(v0_key);
    if (it == yaml_.end())
        return false;
    consumed_.insert(v0_key);
    bool parsed = false;
    if (!try_parse_bool_value(it->second.value, parsed))
    {
        error(it->second.line_number, v0_key,
              "cannot parse '" + it->second.value +
              "' as boolean; expected one of: true, false, yes, no, on, off, 1, 0");
        return false;
    }
    out = parsed;
    return true;
}

// ─────────────────────────────────────────────────────────────
// Top-level: coordinate_system, duration_s, case_name
// ─────────────────────────────────────────────────────────────

// Unmapped-key sweep
// ─────────────────────────────────────────────────────────────

void V0ToV1Migrator::report_unmapped_keys()
{
    for (const auto& [key, field] : yaml_)
    {
        if (consumed_.count(key) == 0)
        {
            warnings_.push_back({key,
                                 field.line_number,
                                 "dropped",
                                 "v0 key not recognized by migrator; please file a bug "
                                 "if this is intentional configuration"});
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Top-level driver
// ─────────────────────────────────────────────────────────────

MigrationResult V0ToV1Migrator::migrate(const std::string& v0_path)
{
    file_ = v0_path;

    auto parsed = parse_yaml_with_lines(v0_path);
    if (!parsed.has_value())
    {
        error(0, "", "could not open config file");
        return MigrationResult{std::nullopt, {}, errors_};
    }
    yaml_ = std::move(*parsed);

    // A v0 file must NOT have `schema:` — that would indicate it is already
    // v1 (or some unknown format). Refuse rather than mangling.
    if (has("schema"))
    {
        error(line_of("schema"),
              "schema",
              "config file already has a 'schema:' stamp — looks like v1 or "
              "newer; migrator only handles v0 (legacy, unstamped) configs");
        return MigrationResult{std::nullopt, warnings_, errors_};
    }

    migrate_top_level();
    migrate_grid_and_time();
    migrate_environment();
    migrate_trigger();
    migrate_dynamics();
    migrate_microphysics();
    migrate_radiation();
    migrate_boundary_layer_and_surface();
    migrate_turbulence();
    migrate_numerics();
    migrate_terrain();
    migrate_output();
    migrate_chaos();
    migrate_validation();
    migrate_metadata_sections();

    // C2: v0 has no boundary_conditions.* keys, so ConfigV1 would default
    // to CM1 (RigidNoSlip + RigidWithSponge). Override to the legacy
    // Nimbus Cartesian behavior (RigidSlip + Rigid) so migrated configs
    // produce identical simulator output to their v0 originals. New v1
    // configs written from scratch still get the CM1 defaults — only the
    // migrator preserves legacy behavior.
    config_.boundary_conditions.bottom = BottomBC::RigidSlip;
    config_.boundary_conditions.top    = TopBC::Rigid;
    const LateralBC legacy_lateral     = resolve_legacy_lateral_bc(LateralBC::LegacyOpen, config_.grid.staggering);
    config_.boundary_conditions.lateral.west  = legacy_lateral;
    config_.boundary_conditions.lateral.east  = legacy_lateral;
    config_.boundary_conditions.lateral.south = legacy_lateral;
    config_.boundary_conditions.lateral.north = legacy_lateral;
    warn_synth("boundary_conditions",
               "preserved legacy Cartesian behavior (bottom=rigid_slip, top=rigid)",
               "v0 had no boundary_conditions.* keys; ConfigV1 defaults to CM1 "
               "(noslip+sponge) but the legacy Nimbus Cartesian backend uses "
               "slip+no-sponge. Override locks behavior; flip explicitly for CM1 parity.");

    // C3: v0 had no damping.* keys; ConfigV1 defaults to CM1 (z_d=12000m,
    // alpha=0.05). The legacy Nimbus sponge used depth_fraction=0.3 of
    // NZ*dz (so z_d varies per config: ztop=8000 → z_d=5600; ztop=16000 →
    // z_d=11200). Override z_d per-config to preserve legacy behavior.
    // Other knobs (enabled, alpha, mode) already match defaults.
    const double z_top = static_cast<double>(config_.grid.nz) * config_.grid.dz;
    if (z_top > 0.0)
    {
        config_.damping.z_d = z_top * 0.7;
        warn_synth("damping",
                   "set damping.z_d=" + std::to_string(config_.damping.z_d) +
                       "m (NZ*dz*0.7 — preserves legacy depth_fraction=0.3)",
                   "v0 sponge was top 30% of NZ*dz; v1 schema CM1 default of "
                   "z_d=12000 would land above the lid for some configs and "
                   "silently disable the sponge. Per-config override locks behavior.");
    }

    // C4: v0 had no coriolis.* keys; ConfigV1 defaults to CM1 (enabled=false).
    // Legacy v0 hardcoded latitude=35°N and applied unconditionally. Override
    // to enabled=true so re-migrated configs preserve the Weisman-Klemp f≈8.36e-5.
    config_.coriolis.enabled      = true;
    config_.coriolis.mode         = CoriolisMode::FPlane;
    config_.coriolis.latitude_deg = 35.0;
    warn_synth("coriolis",
               "set coriolis.enabled=true, latitude_deg=35.0",
               "v0 hardcoded Weisman-Klemp 35°N f-plane Coriolis applied "
               "unconditionally; v1 schema CM1 default is disabled. Override "
               "preserves legacy behavior. Set coriolis.enabled=false explicitly "
               "to drop Coriolis (recommended for short tornado-scale runs).");

    report_unmapped_keys();

    if (!errors_.empty())
    {
        return MigrationResult{std::nullopt, warnings_, errors_};
    }
    return MigrationResult{std::move(config_), warnings_, {}};
}

} // namespace tmv::init::config::detail

namespace tmv::init::config
{

MigrationResult migrate_v0_to_v1(const std::string& v0_path)
{
    detail::V0ToV1Migrator m;
    return m.migrate(v0_path);
}

} // namespace tmv::init::config
