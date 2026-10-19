#include "util/physics_cadence.hpp"
#include "init/config/v1_loader.hpp"

#include "core/runtime/yaml_helpers.hpp"
#include "core/output/output_config.hpp"
#include "diagnostics/storm_tracker.hpp"
#include "init/sounding/parametric_target_math.hpp"
#include "radiation/rrtmg_config_limits.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace tmv::init::config
{

namespace
{

/**
 * @brief Internal stateful loader. One instance per `load_v1_config()` call.
 *
 * Holds the parsed YAML map, accumulating error list, set of consumed keys,
 * and the ConfigV1 being built. Per-section `take_*` methods pull their
 * fields and record errors against the file/line metadata captured at parse
 * time. After all sections are walked, any keys not in `consumed_` become
 * "unrecognized v1 key" errors (§19.6 no-silent-runs principle).
 */
class V1Loader
{
public:
    LoadResult load(const std::string& path, std::unordered_map<std::string, YamlField> fields);

private:
    std::string                                file_;
    std::unordered_map<std::string, YamlField> yaml_;
    std::unordered_set<std::string>            consumed_;
    std::vector<LoadError>                     errors_;
    ConfigV1                                   config_;

    // ─── error reporting ─────────────────────────────────────
    void add_error(int line, const std::string& key, const std::string& message);
    void add_required_missing(const std::string& key);

    // ─── primitive consumers ─────────────────────────────────
    // take_* returns true iff the key was present (regardless of validation).
    // require_* additionally records a "required key missing" error when absent.
    bool take_string(const std::string& key, std::string& out);
    bool require_string(const std::string& key, std::string& out);

    bool take_int(const std::string& key, int& out);
    bool require_int(const std::string& key, int& out);

    bool take_positive_int(const std::string& key, int& out);
    bool require_positive_int(const std::string& key, int& out);

    bool take_double(const std::string& key, double& out);
    bool require_double(const std::string& key, double& out);

    bool take_positive_double(const std::string& key, double& out);
    bool require_positive_double(const std::string& key, double& out);

    bool take_non_negative_double(const std::string& key, double& out);
    bool require_non_negative_double(const std::string& key, double& out);

    // The current runtime schedules duration/output cadence with integer
    // seconds. Accept YAML spellings such as `30.0`, but reject fractional or
    // out-of-range values here instead of silently falling back later.
    bool take_runtime_seconds(const std::string& key, double& out, bool allow_zero);
    bool require_runtime_seconds(const std::string& key, double& out, bool allow_zero);

    bool take_non_negative_int(const std::string& key, int& out);

    bool take_bool(const std::string& key, bool& out);

    template <typename E>
    bool take_enum(const std::string& key, E& out, bool (*parse_fn)(const std::string&, E&), const char* allowed_csv);

    template <typename E>
    bool
    require_enum(const std::string& key, E& out, bool (*parse_fn)(const std::string&, E&), const char* allowed_csv);

    // ─── coordinate-system / stagger parsers (use existing helpers) ──
    bool take_coordinate_system(const std::string& key, CoordinateSystem& out);
    bool take_stagger_type(const std::string& key, StaggerType& out);

    // ─── section walkers ─────────────────────────────────────
    void take_schema_stamp();
    void take_grid();
    void take_time();
    void take_domain_motion();
    void take_dynamics();
    void take_boundary_conditions();
    void take_damping();
    void take_coriolis();
    void take_microphysics();
    void take_turbulence();
    void take_advection();
    void take_diffusion();
    void take_surface();
    void take_boundary_layer();
    void take_radiation();
    void take_terrain();
    void take_initial_conditions();
    void take_radar();
    void take_output();
    void take_extensions();

    void report_unknown_keys();
};

// ─────────────────────────────────────────────────────────────
// Error reporting
// ─────────────────────────────────────────────────────────────

void V1Loader::add_error(int line, const std::string& key, const std::string& message)
{
    errors_.push_back(LoadError{file_, line, key, message});
}

void V1Loader::add_required_missing(const std::string& key)
{
    add_error(0, key, "required key missing");
}

// ─────────────────────────────────────────────────────────────
// String / number / bool consumers
// ─────────────────────────────────────────────────────────────

bool V1Loader::take_string(const std::string& key, std::string& out)
{
    auto it = yaml_.find(key);
    if (it == yaml_.end())
        return false;
    consumed_.insert(key);
    out = it->second.value;
    return true;
}

bool V1Loader::require_string(const std::string& key, std::string& out)
{
    if (take_string(key, out))
        return true;
    add_required_missing(key);
    return false;
}

bool V1Loader::take_int(const std::string& key, int& out)
{
    auto it = yaml_.find(key);
    if (it == yaml_.end())
        return false;
    consumed_.insert(key);
    int parsed = 0;
    if (!try_parse_int_value(it->second.value, parsed))
    {
        add_error(it->second.line_number, key, "cannot parse '" + it->second.value + "' as integer");
        return true; // present (just invalid)
    }
    out = parsed;
    return true;
}

bool V1Loader::require_int(const std::string& key, int& out)
{
    if (take_int(key, out))
        return true;
    add_required_missing(key);
    return false;
}

bool V1Loader::take_positive_int(const std::string& key, int& out)
{
    auto it = yaml_.find(key);
    if (it == yaml_.end())
        return false;
    consumed_.insert(key);
    int parsed = 0;
    if (!try_parse_int_value(it->second.value, parsed))
    {
        add_error(it->second.line_number, key, "cannot parse '" + it->second.value + "' as integer");
        return true;
    }
    if (parsed <= 0)
    {
        add_error(it->second.line_number, key, "must be > 0; got " + std::to_string(parsed));
        return true;
    }
    out = parsed;
    return true;
}

bool V1Loader::require_positive_int(const std::string& key, int& out)
{
    if (take_positive_int(key, out))
        return true;
    add_required_missing(key);
    return false;
}

bool V1Loader::take_double(const std::string& key, double& out)
{
    auto it = yaml_.find(key);
    if (it == yaml_.end())
        return false;
    consumed_.insert(key);
    double parsed = 0.0;
    if (!try_parse_double_value(it->second.value, parsed))
    {
        add_error(it->second.line_number, key, "cannot parse '" + it->second.value + "' as number");
        return true;
    }
    out = parsed;
    return true;
}

bool V1Loader::require_double(const std::string& key, double& out)
{
    if (take_double(key, out))
        return true;
    add_required_missing(key);
    return false;
}

bool V1Loader::take_positive_double(const std::string& key, double& out)
{
    auto it = yaml_.find(key);
    if (it == yaml_.end())
        return false;
    consumed_.insert(key);
    double parsed = 0.0;
    if (!try_parse_double_value(it->second.value, parsed))
    {
        add_error(it->second.line_number, key, "cannot parse '" + it->second.value + "' as number");
        return true;
    }
    if (!(parsed > 0.0))
    {
        std::ostringstream oss;
        oss << "must be > 0; got " << parsed;
        add_error(it->second.line_number, key, oss.str());
        return true;
    }
    out = parsed;
    return true;
}

bool V1Loader::require_positive_double(const std::string& key, double& out)
{
    if (take_positive_double(key, out))
        return true;
    add_required_missing(key);
    return false;
}

bool V1Loader::take_non_negative_double(const std::string& key, double& out)
{
    auto it = yaml_.find(key);
    if (it == yaml_.end())
        return false;
    consumed_.insert(key);
    double parsed = 0.0;
    if (!try_parse_double_value(it->second.value, parsed))
    {
        add_error(it->second.line_number, key, "cannot parse '" + it->second.value + "' as number");
        return true;
    }
    if (parsed < 0.0)
    {
        std::ostringstream oss;
        oss << "must be >= 0; got " << parsed;
        add_error(it->second.line_number, key, oss.str());
        return true;
    }
    out = parsed;
    return true;
}

bool V1Loader::require_non_negative_double(const std::string& key, double& out)
{
    if (take_non_negative_double(key, out))
        return true;
    add_required_missing(key);
    return false;
}

bool V1Loader::take_runtime_seconds(const std::string& key, double& out, bool allow_zero)
{
    auto it = yaml_.find(key);
    if (it == yaml_.end())
        return false;
    consumed_.insert(key);

    double parsed = 0.0;
    if (!try_parse_double_value(it->second.value, parsed))
    {
        add_error(it->second.line_number, key, "cannot parse '" + it->second.value + "' as number");
        return true;
    }
    if ((allow_zero && parsed < 0.0) || (!allow_zero && parsed <= 0.0))
    {
        std::ostringstream oss;
        oss << (allow_zero ? "must be >= 0; got " : "must be > 0; got ") << parsed;
        add_error(it->second.line_number, key, oss.str());
        return true;
    }
    if (std::trunc(parsed) != parsed || parsed > static_cast<double>(std::numeric_limits<int>::max()))
    {
        add_error(it->second.line_number,
                  key,
                  "must be a whole number of seconds within the runtime integer range; got '" + it->second.value + "'");
        return true;
    }
    out = parsed;
    return true;
}

bool V1Loader::require_runtime_seconds(const std::string& key, double& out, bool allow_zero)
{
    if (take_runtime_seconds(key, out, allow_zero))
        return true;
    add_required_missing(key);
    return false;
}

bool V1Loader::take_non_negative_int(const std::string& key, int& out)
{
    auto it = yaml_.find(key);
    if (it == yaml_.end())
        return false;
    consumed_.insert(key);
    int parsed = 0;
    if (!try_parse_non_negative_int_value(it->second.value, parsed))
    {
        add_error(it->second.line_number, key, "cannot parse '" + it->second.value + "' as non-negative integer");
        return true;
    }
    out = parsed;
    return true;
}

bool V1Loader::take_bool(const std::string& key, bool& out)
{
    auto it = yaml_.find(key);
    if (it == yaml_.end())
        return false;
    consumed_.insert(key);
    bool parsed = false;
    if (!try_parse_bool_value(it->second.value, parsed))
    {
        add_error(it->second.line_number,
                  key,
                  "cannot parse '" + it->second.value +
                      "' as boolean; expected one of: true, false, yes, no, on, off, 1, 0");
        return true;
    }
    out = parsed;
    return true;
}

template <typename E>
bool V1Loader::take_enum(const std::string& key,
                         E&                 out,
                         bool (*parse_fn)(const std::string&, E&),
                         const char* allowed_csv)
{
    auto it = yaml_.find(key);
    if (it == yaml_.end())
        return false;
    consumed_.insert(key);
    E parsed{};
    if (!parse_fn(it->second.value, parsed))
    {
        add_error(
            it->second.line_number, key, "unknown value '" + it->second.value + "'; expected one of: " + allowed_csv);
        return true;
    }
    out = parsed;
    return true;
}

template <typename E>
bool V1Loader::require_enum(const std::string& key,
                            E&                 out,
                            bool (*parse_fn)(const std::string&, E&),
                            const char* allowed_csv)
{
    if (take_enum(key, out, parse_fn, allowed_csv))
        return true;
    add_required_missing(key);
    return false;
}

bool V1Loader::take_coordinate_system(const std::string& key, CoordinateSystem& out)
{
    auto it = yaml_.find(key);
    if (it == yaml_.end())
        return false;
    consumed_.insert(key);
    CoordinateSystem parsed = CoordinateSystem::Cartesian;
    if (!parse_coordinate_system(it->second.value, parsed))
    {
        add_error(it->second.line_number, key, "unknown value '" + it->second.value + "'; expected: cartesian");
        return true;
    }
    out = parsed;
    return true;
}

bool V1Loader::take_stagger_type(const std::string& key, StaggerType& out)
{
    auto it = yaml_.find(key);
    if (it == yaml_.end())
        return false;
    consumed_.insert(key);
    StaggerType parsed = StaggerType::CGrid;
    if (!parse_stagger_type(it->second.value, parsed))
    {
        add_error(it->second.line_number,
                  key,
                  "unknown value '" + it->second.value + "'; expected one of: collocated, c_grid");
        return true;
    }
    out = parsed;
    return true;
}

// ─────────────────────────────────────────────────────────────
// Section walkers
// ─────────────────────────────────────────────────────────────

void V1Loader::take_schema_stamp()
{
    auto it = yaml_.find("schema");
    if (it == yaml_.end())
    {
        add_error(0, "schema", "missing required v1 schema stamp; expected `schema: nimbus.v1`");
        return;
    }
    consumed_.insert("schema");
    if (it->second.value != "nimbus.v1")
    {
        add_error(it->second.line_number,
                  "schema",
                  "unsupported schema version '" + it->second.value + "'; expected 'nimbus.v1'");
        return;
    }
    config_.schema_version = it->second.value;
}

void V1Loader::take_grid()
{
    auto& g = config_.grid;
    take_coordinate_system("grid.coordinate_system", g.coordinate_system);
    take_stagger_type("grid.staggering", g.staggering);
    require_positive_int("grid.nx", g.nx);
    require_positive_int("grid.ny", g.ny);
    require_positive_int("grid.nz", g.nz);
    require_positive_double("grid.dx", g.dx);
    require_positive_double("grid.dy", g.dy);
    require_positive_double("grid.dz", g.dz);
    take_positive_double("grid.ztop", g.ztop);
}

void V1Loader::take_time()
{
    auto& t = config_.time;
    take_enum("time.integrator",
              t.integrator,
              parse_time_integrator,
              "split_explicit, rk3, rk4");
    require_runtime_seconds("time.run_time", t.run_time, false);
    require_positive_double("time.dt_large", t.dt_large);
    // 0 means fully automatic. Positive values are requested minima; runtime
    // CFL enforcement may promote them when the current sound speed requires
    // more acoustic substeps.
    take_non_negative_int("time.acoustic_substeps", t.acoustic_substeps);
    take_bool("time.adaptive_dt", t.adaptive_dt);
    take_positive_double("time.cfl_safety", t.cfl_safety);
    // output_interval == 0 means "no periodic output" (CM1 tapfrq=0 convention).
    require_runtime_seconds("time.output_interval", t.output_interval, true);
}

void V1Loader::take_domain_motion()
{
    auto& motion = config_.domain_motion;
    take_enum("domain_motion.mode",
              motion.mode,
              parse_domain_motion_mode,
              "stationary, constant_velocity, tracked_window, moving_nest");
    take_double("domain_motion.frame_velocity.u_ms", motion.frame_velocity.u_ms);
    take_double("domain_motion.frame_velocity.v_ms", motion.frame_velocity.v_ms);

    auto& planning = motion.lifetime_planning;
    take_bool("domain_motion.lifetime_planning.enabled", planning.enabled);
    take_enum("domain_motion.lifetime_planning.action", planning.action, parse_lifetime_planning_action, "warn, error");
    take_enum("domain_motion.lifetime_planning.motion_source",
              planning.motion_source,
              parse_storm_motion_source,
              "bunkers_right, bunkers_left, explicit");
    take_double("domain_motion.lifetime_planning.explicit_u_ms", planning.explicit_u_ms);
    take_double("domain_motion.lifetime_planning.explicit_v_ms", planning.explicit_v_ms);
    take_non_negative_double("domain_motion.lifetime_planning.upstream_buffer_m", planning.upstream_buffer_m);
    take_non_negative_double("domain_motion.lifetime_planning.downstream_buffer_m", planning.downstream_buffer_m);
    take_non_negative_double("domain_motion.lifetime_planning.cross_track_buffer_m", planning.cross_track_buffer_m);
    take_non_negative_double("domain_motion.lifetime_planning.motion_uncertainty_ms", planning.motion_uncertainty_ms);

    auto& tracking = motion.tracking;
    take_bool("domain_motion.tracking.enabled", tracking.enabled);
    take_bool("domain_motion.tracking.recycle_domain", tracking.recycle_domain);
    take_positive_double("domain_motion.tracking.interval_s", tracking.interval_s);
    take_non_negative_double("domain_motion.tracking.inner_margin_m", tracking.inner_margin_m);
    take_non_negative_int("domain_motion.tracking.maximum_shift_cells", tracking.maximum_shift_cells);
    take_non_negative_double("domain_motion.tracking.cooldown_s", tracking.cooldown_s);
    take_non_negative_double("domain_motion.tracking.minimum_confidence", tracking.minimum_confidence);
    take_positive_double("domain_motion.tracking.minimum_updraft_ms", tracking.minimum_updraft_ms);
    take_positive_double("domain_motion.tracking.minimum_condensate_path_kgm2", tracking.minimum_condensate_path_kgm2);
    take_non_negative_double("domain_motion.tracking.updraft_helicity_bottom_m_agl",
                             tracking.updraft_helicity_bottom_m_agl);
    take_positive_double("domain_motion.tracking.updraft_helicity_top_m_agl", tracking.updraft_helicity_top_m_agl);
    take_positive_double("domain_motion.tracking.minimum_absolute_updraft_helicity_m2s2",
                         tracking.minimum_absolute_updraft_helicity_m2s2);
    take_positive_int("domain_motion.tracking.minimum_component_cells", tracking.minimum_component_cells);
    take_positive_double("domain_motion.tracking.minimum_core_area_m2", tracking.minimum_core_area_m2);
    take_positive_double("domain_motion.tracking.maximum_motion_speed_ms", tracking.maximum_motion_speed_ms);
    take_non_negative_double("domain_motion.tracking.maximum_motion_acceleration_ms2",
                             tracking.maximum_motion_acceleration_ms2);
    take_positive_int("domain_motion.tracking.minimum_continuity_samples", tracking.minimum_continuity_samples);
    take_positive_int("domain_motion.tracking.lost_after_samples", tracking.lost_after_samples);
    take_positive_int("domain_motion.tracking.topology_confirmation_samples", tracking.topology_confirmation_samples);
    take_enum("domain_motion.tracking.branch_preference",
              tracking.branch_preference,
              parse_storm_branch_preference,
              "continuity, bunkers_right, bunkers_left");
    take_positive_double("domain_motion.tracking.branch_velocity_uncertainty_ms",
                         tracking.branch_velocity_uncertainty_ms);
    take_positive_int("domain_motion.nesting.parent_ratio_x", motion.nesting.parent_ratio_x);
    take_positive_int("domain_motion.nesting.parent_ratio_y", motion.nesting.parent_ratio_y);
    take_positive_int("domain_motion.nesting.parent_time_step_ratio", motion.nesting.parent_time_step_ratio);
    take_bool("domain_motion.nesting.enabled", motion.nesting.enabled);
    take_non_negative_int("domain_motion.nesting.child_origin_parent_cell_i",
                          motion.nesting.child_origin_parent_cell_i);
    take_non_negative_int("domain_motion.nesting.child_origin_parent_cell_j",
                          motion.nesting.child_origin_parent_cell_j);
    take_non_negative_int("domain_motion.nesting.child_cells_x", motion.nesting.child_cells_x);
    take_non_negative_int("domain_motion.nesting.child_cells_y", motion.nesting.child_cells_y);
    take_non_negative_int("domain_motion.nesting.relaxation_width_child_cells",
                          motion.nesting.relaxation_width_child_cells);
    take_positive_int("domain_motion.nesting.boundary_stencil_parent_cells",
                      motion.nesting.boundary_stencil_parent_cells);
    take_positive_double("domain_motion.nesting.relaxation_e_folding_time_s",
                         motion.nesting.relaxation_e_folding_time_s);
    take_enum("domain_motion.nesting.feedback", motion.nesting.feedback,
              parse_nesting_feedback, "one_way, two_way");

    if (motion.nesting.parent_ratio_x < 2 || motion.nesting.parent_ratio_y < 2)
    {
        const char* key = motion.nesting.parent_ratio_x < 2 ? "domain_motion.nesting.parent_ratio_x"
                                                            : "domain_motion.nesting.parent_ratio_y";
        const auto  it  = yaml_.find(key);
        add_error(it != yaml_.end() ? it->second.line_number : 0, key, "must be at least 2 for a refined child");
    }
    if (motion.nesting.parent_time_step_ratio < std::max(motion.nesting.parent_ratio_x, motion.nesting.parent_ratio_y))
    {
        const auto it = yaml_.find("domain_motion.nesting.parent_time_step_ratio");
        add_error(it != yaml_.end() ? it->second.line_number : 0,
                  "domain_motion.nesting.parent_time_step_ratio",
                  "must be at least the largest horizontal parent ratio");
    }
    if (motion.nesting.enabled)
    {
        const auto& nesting     = motion.nesting;
        const char* invalid_key = nesting.child_cells_x <= 0   ? "domain_motion.nesting.child_cells_x"
                                  : nesting.child_cells_y <= 0 ? "domain_motion.nesting.child_cells_y"
                                  : nesting.relaxation_width_child_cells <= 0
                                      ? "domain_motion.nesting.relaxation_width_child_cells"
                                      : nullptr;
        if (invalid_key != nullptr)
        {
            const auto it = yaml_.find(invalid_key);
            add_error(it != yaml_.end() ? it->second.line_number : 0,
                      invalid_key,
                      "must be positive when nesting is enabled");
        }
        if (nesting.child_cells_x > 0 && nesting.child_cells_x % nesting.parent_ratio_x != 0)
        {
            const auto it = yaml_.find("domain_motion.nesting.child_cells_x");
            add_error(it != yaml_.end() ? it->second.line_number : 0,
                      "domain_motion.nesting.child_cells_x",
                      "must span a whole number of parent cells");
        }
        if (nesting.child_cells_y > 0 && nesting.child_cells_y % nesting.parent_ratio_y != 0)
        {
            const auto it = yaml_.find("domain_motion.nesting.child_cells_y");
            add_error(it != yaml_.end() ? it->second.line_number : 0,
                      "domain_motion.nesting.child_cells_y",
                      "must span a whole number of parent cells");
        }
        if (nesting.relaxation_width_child_cells > 0 &&
            2LL * nesting.relaxation_width_child_cells >= std::min(nesting.child_cells_x, nesting.child_cells_y))
        {
            const auto it = yaml_.find("domain_motion.nesting.relaxation_width_child_cells");
            add_error(it != yaml_.end() ? it->second.line_number : 0,
                      "domain_motion.nesting.relaxation_width_child_cells",
                      "must leave a non-empty child interior");
        }
    }

    const auto require_finite = [this](const char* key, double value)
    {
        if (std::isfinite(value))
            return;
        const auto it = yaml_.find(key);
        add_error(it != yaml_.end() ? it->second.line_number : 0, key, "must be finite");
    };
    require_finite("domain_motion.frame_velocity.u_ms", motion.frame_velocity.u_ms);
    require_finite("domain_motion.frame_velocity.v_ms", motion.frame_velocity.v_ms);
    require_finite("domain_motion.lifetime_planning.upstream_buffer_m", planning.upstream_buffer_m);
    require_finite("domain_motion.lifetime_planning.downstream_buffer_m", planning.downstream_buffer_m);
    require_finite("domain_motion.lifetime_planning.cross_track_buffer_m", planning.cross_track_buffer_m);
    require_finite("domain_motion.lifetime_planning.motion_uncertainty_ms", planning.motion_uncertainty_ms);
    require_finite("domain_motion.lifetime_planning.explicit_u_ms", planning.explicit_u_ms);
    require_finite("domain_motion.lifetime_planning.explicit_v_ms", planning.explicit_v_ms);
    require_finite("domain_motion.tracking.interval_s", tracking.interval_s);
    require_finite("domain_motion.tracking.inner_margin_m", tracking.inner_margin_m);
    require_finite("domain_motion.tracking.cooldown_s", tracking.cooldown_s);
    require_finite("domain_motion.tracking.minimum_confidence", tracking.minimum_confidence);
    require_finite("domain_motion.tracking.minimum_updraft_ms", tracking.minimum_updraft_ms);
    require_finite("domain_motion.tracking.minimum_condensate_path_kgm2", tracking.minimum_condensate_path_kgm2);
    require_finite("domain_motion.tracking.updraft_helicity_bottom_m_agl", tracking.updraft_helicity_bottom_m_agl);
    require_finite("domain_motion.tracking.updraft_helicity_top_m_agl", tracking.updraft_helicity_top_m_agl);
    require_finite("domain_motion.tracking.minimum_absolute_updraft_helicity_m2s2",
                   tracking.minimum_absolute_updraft_helicity_m2s2);
    require_finite("domain_motion.tracking.minimum_core_area_m2", tracking.minimum_core_area_m2);
    require_finite("domain_motion.tracking.maximum_motion_speed_ms", tracking.maximum_motion_speed_ms);
    require_finite("domain_motion.tracking.maximum_motion_acceleration_ms2", tracking.maximum_motion_acceleration_ms2);
    require_finite("domain_motion.tracking.branch_velocity_uncertainty_ms", tracking.branch_velocity_uncertainty_ms);

    if (tracking.minimum_confidence > 1.0 || (tracking.enabled && tracking.minimum_confidence <= 0.0))
    {
        const auto it = yaml_.find("domain_motion.tracking.minimum_confidence");
        add_error(it != yaml_.end() ? it->second.line_number : 0,
                  "domain_motion.tracking.minimum_confidence",
                  tracking.enabled ? "must be in (0, 1] when tracking is enabled" : "must be in [0, 1]");
    }
    if (tracking.updraft_helicity_top_m_agl <= tracking.updraft_helicity_bottom_m_agl)
    {
        const auto it = yaml_.find("domain_motion.tracking.updraft_helicity_top_m_agl");
        add_error(it != yaml_.end() ? it->second.line_number : 0,
                  "domain_motion.tracking.updraft_helicity_top_m_agl",
                  "must exceed updraft_helicity_bottom_m_agl");
    }
    if (tracking.minimum_continuity_samples < 2 ||
        tracking.minimum_continuity_samples > static_cast<int>(tmv::diagnostics::storm_continuity_history_capacity))
    {
        const auto it = yaml_.find("domain_motion.tracking.minimum_continuity_samples");
        add_error(it != yaml_.end() ? it->second.line_number : 0,
                  "domain_motion.tracking.minimum_continuity_samples",
                  "must be an integer from 2 through 5");
    }
    if (tracking.minimum_continuity_samples > static_cast<int>(tmv::diagnostics::storm_continuity_history_capacity))
    {
        const auto it = yaml_.find("domain_motion.tracking.minimum_continuity_samples");
        add_error(it != yaml_.end() ? it->second.line_number : 0,
                  "domain_motion.tracking.minimum_continuity_samples",
                  "must not exceed the fixed continuity history capacity of 5");
    }

    // Moving-nest admission depends on the complete science, execution, and
    // compute tuple, so it is validated after every section has been parsed.
    if (motion.mode == DomainMotionMode::TrackedWindow)
    {
        const auto add_tracked_error = [&](const char* key,
                                           const char* message)
        {
            const auto it = yaml_.find(key);
            add_error(it != yaml_.end() ? it->second.line_number : 0,
                      key, message);
        };
        if (!tracking.enabled)
            add_tracked_error("domain_motion.tracking.enabled",
                              "tracked_window requires tracking.enabled=true");
        if (!tracking.recycle_domain)
            add_tracked_error("domain_motion.tracking.recycle_domain",
                              "tracked_window requires recycle_domain=true");
        if (tracking.maximum_shift_cells <= 0)
            add_tracked_error("domain_motion.tracking.maximum_shift_cells",
                              "tracked_window requires a positive shift limit");
        if (motion.frame_velocity.u_ms != 0.0
            || motion.frame_velocity.v_ms != 0.0)
            add_tracked_error("domain_motion.frame_velocity.u_ms",
                              "tracked_window uses discrete recycling and requires zero continuous frame velocity");
    }
    if (motion.mode == DomainMotionMode::Stationary &&
        (motion.frame_velocity.u_ms != 0.0 || motion.frame_velocity.v_ms != 0.0))
    {
        const char* key = motion.frame_velocity.u_ms != 0.0 ? "domain_motion.frame_velocity.u_ms"
                                                            : "domain_motion.frame_velocity.v_ms";
        const auto  it  = yaml_.find(key);
        add_error(it != yaml_.end() ? it->second.line_number : 0, key, "stationary mode requires zero frame velocity");
    }
    if (planning.enabled && (planning.upstream_buffer_m <= 0.0 || planning.downstream_buffer_m <= 0.0 ||
                             planning.cross_track_buffer_m <= 0.0))
    {
        const char* key = planning.upstream_buffer_m <= 0.0 ? "domain_motion.lifetime_planning.upstream_buffer_m"
                          : planning.downstream_buffer_m <= 0.0
                              ? "domain_motion.lifetime_planning.downstream_buffer_m"
                              : "domain_motion.lifetime_planning.cross_track_buffer_m";
        const auto  it  = yaml_.find(key);
        add_error(it != yaml_.end() ? it->second.line_number : 0,
                  key,
                  "must be > 0 when lifetime planning is enabled; a zero-width "
                  "point trajectory is not a storm-lifetime envelope");
    }
    if (tracking.enabled && tracking.inner_margin_m <= 0.0)
    {
        const auto it = yaml_.find("domain_motion.tracking.inner_margin_m");
        add_error(it != yaml_.end() ? it->second.line_number : 0,
                  "domain_motion.tracking.inner_margin_m",
                  "must be > 0 when diagnostic storm tracking is enabled");
    }
    if (tracking.recycle_domain && !tracking.enabled)
    {
        const auto it = yaml_.find("domain_motion.tracking.recycle_domain");
        add_error(it != yaml_.end() ? it->second.line_number : 0,
                  "domain_motion.tracking.recycle_domain",
                  "requires domain_motion.tracking.enabled=true");
    }
    if (tracking.recycle_domain && tracking.maximum_shift_cells <= 0)
    {
        const auto it = yaml_.find("domain_motion.tracking.maximum_shift_cells");
        add_error(it != yaml_.end() ? it->second.line_number : 0,
                  "domain_motion.tracking.maximum_shift_cells",
                  "must be > 0 when live domain recycling is enabled");
    }
}

void V1Loader::take_dynamics()
{
    auto& d = config_.dynamics;
    // `dynamics.scheme` is a string (not enum) — factory.cpp owns the
    // alias resolution including legacy v0 names. Loader records whatever
    // is supplied; downstream factory rejects unknown names.
    take_string("dynamics.scheme", d.scheme);
    take_bool("dynamics.axisymmetric", d.axisymmetric);
}

void V1Loader::take_boundary_conditions()
{
    auto&       bc              = config_.boundary_conditions;
    const char* lateral_allowed = "periodic, radiative, zero_gradient, open (legacy), rigid, rigid_slip";
    take_enum("boundary_conditions.lateral.west", bc.lateral.west, parse_lateral_bc, lateral_allowed);
    take_enum("boundary_conditions.lateral.east", bc.lateral.east, parse_lateral_bc, lateral_allowed);
    take_enum("boundary_conditions.lateral.south", bc.lateral.south, parse_lateral_bc, lateral_allowed);
    take_enum("boundary_conditions.lateral.north", bc.lateral.north, parse_lateral_bc, lateral_allowed);

    // SCI-017: `open` used to select two different algorithms depending on
    // staggering. Resolve that compatibility token now so no runtime enum is
    // ambiguous. An explicitly radiative closure cannot run on the collocated
    // scheme; reject it instead of silently degrading to zero-gradient.
    const auto resolve_lateral = [this](const char* key, LateralBC& value)
    {
        value = resolve_legacy_lateral_bc(value, config_.grid.staggering);
        if (value != LateralBC::Radiative || config_.grid.staggering == StaggerType::CGrid)
        {
            return;
        }
        const auto it   = yaml_.find(key);
        const int  line = (it != yaml_.end()) ? it->second.line_number : 0;
        add_error(line,
                  key,
                  "radiative lateral boundaries require grid.staggering=c_grid; "
                  "use zero_gradient for a collocated grid");
    };
    resolve_lateral("boundary_conditions.lateral.west", bc.lateral.west);
    resolve_lateral("boundary_conditions.lateral.east", bc.lateral.east);
    resolve_lateral("boundary_conditions.lateral.south", bc.lateral.south);
    resolve_lateral("boundary_conditions.lateral.north", bc.lateral.north);

    // Periodicity is a domain-topology property, not an independent face
    // closure.  Both opposing faces must therefore opt into it together.
    // Enforce that invariant here so the A-grid implementation (which wraps
    // a whole pair when either face is periodic) and the C-grid implementation
    // (which dispatches each face separately) cannot interpret the same valid
    // v1 configuration differently.
    const auto require_periodic_pair =
        [this](const char* first_key, LateralBC first, const char* second_key, LateralBC second)
    {
        const bool first_periodic  = first == LateralBC::Periodic;
        const bool second_periodic = second == LateralBC::Periodic;
        if (first_periodic == second_periodic)
            return;

        // Attribute the error to the face that requested periodicity.  A
        // missing opposite key intentionally retains its default `open`,
        // so one explicit periodic face is still rejected with a useful
        // source line.
        const char* periodic_key = first_periodic ? first_key : second_key;
        const auto  it           = yaml_.find(periodic_key);
        const int   line         = (it != yaml_.end()) ? it->second.line_number : 0;
        add_error(line,
                  periodic_key,
                  std::string("periodic lateral boundaries must be configured ") + "as opposing pairs; '" + first_key +
                      "' and '" + second_key + "' must either both be periodic or both " + "be non-periodic");
    };

    require_periodic_pair(
        "boundary_conditions.lateral.west", bc.lateral.west, "boundary_conditions.lateral.east", bc.lateral.east);
    require_periodic_pair(
        "boundary_conditions.lateral.south", bc.lateral.south, "boundary_conditions.lateral.north", bc.lateral.north);

    take_enum("boundary_conditions.bottom", bc.bottom, parse_bottom_bc, "rigid_noslip, rigid_slip");
    take_enum("boundary_conditions.top", bc.top, parse_top_bc, "rigid, rigid_with_sponge, open");
}

void V1Loader::take_damping()
{
    auto& d = config_.damping;
    take_bool("damping.enabled", d.enabled);
    take_non_negative_double("damping.z_d", d.z_d);
    take_non_negative_double("damping.alpha", d.alpha);
    take_enum("damping.mode", d.mode, parse_damping_mode, "explicit, implicit");
    take_non_negative_int("damping.lateral_width", d.lateral_width);
    take_non_negative_double("damping.lateral_alpha", d.lateral_alpha);
    take_bool("damping.lateral_sponge_enabled", d.lateral_sponge_enabled);
    take_bool("damping.w_only_top_sponge", d.w_only_top_sponge);
    take_non_negative_int("damping.w_only_depth_cells", d.w_only_depth_cells);
    take_non_negative_double("damping.w_only_alpha_max", d.w_only_alpha_max);
}

void V1Loader::take_coriolis()
{
    auto& c = config_.coriolis;
    take_bool("coriolis.enabled", c.enabled);
    take_enum("coriolis.mode", c.mode, parse_coriolis_mode, "f_plane, beta_plane");
    take_double("coriolis.latitude_deg", c.latitude_deg);
    if (c.latitude_deg < -90.0 || c.latitude_deg > 90.0)
    {
        auto               it   = yaml_.find("coriolis.latitude_deg");
        int                line = (it != yaml_.end()) ? it->second.line_number : 0;
        std::ostringstream oss;
        oss << "must be in [-90, 90]; got " << c.latitude_deg;
        add_error(line, "coriolis.latitude_deg", oss.str());
    }
}

void V1Loader::take_microphysics()
{
    auto& microphysics = config_.microphysics;
    take_enum("microphysics.scheme",
              microphysics.scheme,
              parse_microphysics_scheme,
              "none, kessler, lin, thompson, milbrandt");

    const bool shared_threshold = take_non_negative_double("microphysics.shared.cloud_water_threshold_kgkg",
                                                           microphysics.parameters.shared.cloud_water_threshold_kgkg);
    const bool shared_autoconversion = take_non_negative_double(
        "microphysics.shared.autoconversion_rate_s_inv", microphysics.parameters.shared.autoconversion_rate_s_inv);
    const bool kessler_accretion    = take_non_negative_double("microphysics.kessler.accretion_coefficient",
                                                               microphysics.parameters.kessler.accretion_coefficient);
    const bool kessler_evaporation  = take_non_negative_double("microphysics.kessler.evaporation_efficiency",
                                                               microphysics.parameters.kessler.evaporation_efficiency);
    const bool lin_accretion        = take_non_negative_double("microphysics.lin.accretion_coefficient",
                                                               microphysics.parameters.lin.accretion_coefficient);
    const bool lin_evaporation      = take_non_negative_double("microphysics.lin.evaporation_rate_s_inv",
                                                               microphysics.parameters.lin.evaporation_rate_s_inv);
    const bool thompson_evaporation = take_non_negative_double("microphysics.thompson.evaporation_rate_s_inv",
                                                               microphysics.parameters.thompson.evaporation_rate_s_inv);
    const bool thompson_water_aerosol = take_positive_double(
        "microphysics.thompson.water_friendly_aerosol_surface_m3",
        microphysics.parameters.thompson.water_friendly_aerosol_surface_m3);
    const bool thompson_ice_aerosol = take_positive_double(
        "microphysics.thompson.ice_friendly_aerosol_surface_m3",
        microphysics.parameters.thompson.ice_friendly_aerosol_surface_m3);
    if (microphysics.scheme == MicrophysicsScheme::Thompson
        && thompson_water_aerosol
        && microphysics.parameters.thompson.water_friendly_aerosol_surface_m3 <= 50.0e6)
    {
        const auto it = yaml_.find(
            "microphysics.thompson.water_friendly_aerosol_surface_m3");
        add_error(it->second.line_number,
                  it->first,
                  "must exceed the fixed 50000000 m^-3 free-troposphere background");
    }
    if (microphysics.scheme == MicrophysicsScheme::Thompson
        && thompson_ice_aerosol
        && microphysics.parameters.thompson.ice_friendly_aerosol_surface_m3 <= 0.5e6)
    {
        const auto it = yaml_.find(
            "microphysics.thompson.ice_friendly_aerosol_surface_m3");
        add_error(it->second.line_number,
                  it->first,
                  "must exceed the fixed 500000 m^-3 free-troposphere background");
    }
    const bool milbrandt_evaporation = take_non_negative_double(
        "microphysics.milbrandt.evaporation_rate_s_inv", microphysics.parameters.milbrandt.evaporation_rate_s_inv);
    const bool milbrandt_triple =
        take_bool("microphysics.milbrandt.triple_moment", microphysics.parameters.milbrandt.triple_moment);
    const bool milbrandt_hail =
        take_bool("microphysics.milbrandt.hail_processes", microphysics.parameters.milbrandt.hail_processes);

    const auto reject_inactive = [&](bool present, const char* key, bool active, const char* expected)
    {
        if (!present || active)
            return;
        const auto it = yaml_.find(key);
        add_error(it != yaml_.end() ? it->second.line_number : 0,
                  key,
                  std::string("parameter is inactive for microphysics.scheme=") +
                      microphysics_scheme_name(microphysics.scheme) + "; select " + expected + " or remove the key");
    };
    const bool shared_parameter_scheme =
        microphysics.scheme != MicrophysicsScheme::None && microphysics.scheme != MicrophysicsScheme::Milbrandt;
    reject_inactive(shared_threshold,
                    "microphysics.shared.cloud_water_threshold_kgkg",
                    shared_parameter_scheme,
                    "kessler, lin, or thompson");
    reject_inactive(shared_autoconversion,
                    "microphysics.shared.autoconversion_rate_s_inv",
                    shared_parameter_scheme,
                    "kessler, lin, or thompson");
    reject_inactive(kessler_accretion,
                    "microphysics.kessler.accretion_coefficient",
                    microphysics.scheme == MicrophysicsScheme::Kessler,
                    "kessler");
    reject_inactive(kessler_evaporation,
                    "microphysics.kessler.evaporation_efficiency",
                    microphysics.scheme == MicrophysicsScheme::Kessler,
                    "kessler");
    reject_inactive(
        lin_accretion, "microphysics.lin.accretion_coefficient", microphysics.scheme == MicrophysicsScheme::Lin, "lin");
    reject_inactive(lin_evaporation,
                    "microphysics.lin.evaporation_rate_s_inv",
                    microphysics.scheme == MicrophysicsScheme::Lin,
                    "lin");
    reject_inactive(thompson_evaporation,
                    "microphysics.thompson.evaporation_rate_s_inv",
                    microphysics.scheme == MicrophysicsScheme::Thompson,
                    "thompson");
    reject_inactive(thompson_water_aerosol,
                    "microphysics.thompson.water_friendly_aerosol_surface_m3",
                    microphysics.scheme == MicrophysicsScheme::Thompson,
                    "thompson");
    reject_inactive(thompson_ice_aerosol,
                    "microphysics.thompson.ice_friendly_aerosol_surface_m3",
                    microphysics.scheme == MicrophysicsScheme::Thompson,
                    "thompson");
    reject_inactive(milbrandt_evaporation,
                    "microphysics.milbrandt.evaporation_rate_s_inv",
                    false,
                    "no scheme (obsolete fixed-rate Milbrandt control)");
    if (milbrandt_triple)
    {
        const auto it = yaml_.find("microphysics.milbrandt.triple_moment");
        add_error(it != yaml_.end() ? it->second.line_number : 0,
                  "microphysics.milbrandt.triple_moment",
                  "retained Milbrandt triple-moment selection is engineering incomplete and not admitted");
    }
    reject_inactive(milbrandt_hail,
                    "microphysics.milbrandt.hail_processes",
                    microphysics.scheme == MicrophysicsScheme::Milbrandt,
                    "milbrandt");
}

void V1Loader::take_turbulence()
{
    auto& t = config_.turbulence;
    take_enum("turbulence.scheme", t.scheme, parse_turbulence_scheme, "none, smagorinsky, tke");
    take_enum("turbulence.mode", t.mode, parse_turbulence_mode, "2d, vertical_only, 3d");
    const bool has_filter = take_enum(
        "turbulence.filter_width", t.filter_width,
        parse_turbulence_filter_width, "cubic_root, dx, user");
    take_enum(
        "turbulence.stability_correction", t.stability_correction,
        parse_turbulence_stability_correction, "none, ri, tke_based");
    take_bool("turbulence.dynamic_coefficient", t.dynamic_coefficient);
    const bool has_user_width = take_non_negative_double(
        "turbulence.user_filter_width_m", t.user_filter_width_m);
    if (t.filter_width == TurbulenceFilterWidth::User)
    {
        if (!(t.user_filter_width_m > 0.0) ||
            !std::isfinite(t.user_filter_width_m))
            add_error(has_user_width
                          ? yaml_.at("turbulence.user_filter_width_m").line_number
                          : (has_filter
                                 ? yaml_.at("turbulence.filter_width").line_number
                                 : 0),
                "turbulence.user_filter_width_m",
                "turbulence.user_filter_width_m must be positive and finite "
                "when turbulence.filter_width=user");
    }
    else if (t.user_filter_width_m != 0.0)
    {
        add_error(yaml_.at("turbulence.user_filter_width_m").line_number,
            "turbulence.user_filter_width_m",
            "turbulence.user_filter_width_m is only valid when "
            "turbulence.filter_width=user");
    }
    // Cs == 0 is a valid "disable Smagorinsky" pattern used in minimal-physics
    // configs (scheme name kept, coefficient zeroed). Pr_t / Sc_t must be > 0
    // (zero turbulent Prandtl/Schmidt is physically meaningless).
    take_non_negative_double("turbulence.Cs", t.Cs);
    take_positive_double("turbulence.Pr_t", t.Pr_t);
    take_positive_double("turbulence.Sc_t", t.Sc_t);

    const auto line_of = [&](const char* key)
    {
        const auto it = yaml_.find(key);
        return it == yaml_.end() ? 0 : it->second.line_number;
    };
    if (t.dynamic_coefficient && t.scheme != TurbulenceScheme::Smagorinsky)
    {
        add_error(line_of("turbulence.dynamic_coefficient"),
            "turbulence.dynamic_coefficient",
            "turbulence.dynamic_coefficient requires scheme=smagorinsky");
    }
    if (t.dynamic_coefficient && t.stability_correction !=
            TurbulenceStabilityCorrection::None)
    {
        add_error(line_of("turbulence.stability_correction"),
            "turbulence.stability_correction",
            "dynamic Smagorinsky cannot use a separate stability correction");
    }
    if (t.stability_correction == TurbulenceStabilityCorrection::TkeBased &&
        t.scheme != TurbulenceScheme::TKE)
    {
        add_error(line_of("turbulence.stability_correction"),
            "turbulence.stability_correction",
            "tke_based stability correction requires scheme=tke");
    }
    if (t.stability_correction ==
            TurbulenceStabilityCorrection::Richardson &&
        t.scheme != TurbulenceScheme::Smagorinsky)
    {
        add_error(line_of("turbulence.stability_correction"),
            "turbulence.stability_correction",
            "ri stability correction requires scheme=smagorinsky");
    }
}

void V1Loader::take_advection()
{
    take_enum("advection.scheme", config_.advection.scheme,
              parse_advection_scheme, "auto, none, tvd, weno5");
}

void V1Loader::take_diffusion()
{
    auto& d = config_.diffusion;
    take_enum("diffusion.scheme", d.scheme, parse_diffusion_scheme, "explicit, implicit");
    if (d.scheme == DiffusionScheme::None)
    {
        auto      it   = yaml_.find("diffusion.scheme");
        const int line = (it != yaml_.end()) ? it->second.line_number : 0;
        add_error(line,
                  "diffusion.scheme",
                  "no registered runtime diffusion scheme exists for 'none'; "
                  "use 'explicit' with K_h: 0 and K_v: 0 to disable "
                  "numerical diffusion");
    }
    take_positive_int("diffusion.operator_order", d.operator_order);
    if (d.operator_order != 2)
    {
        auto      it   = yaml_.find("diffusion.operator_order");
        const int line = (it != yaml_.end()) ? it->second.line_number : 0;
        add_error(line,
                  "diffusion.operator_order",
                  "only the implemented runtime order 2 is supported; got " + std::to_string(d.operator_order));
    }
    take_non_negative_double("diffusion.K_h", d.K_h);
    take_non_negative_double("diffusion.K_v", d.K_v);
    take_enum("diffusion.apply_to", d.apply_to, parse_diffusion_target, "scalars, scalars_and_momentum");
    take_enum("diffusion.coefficient_source", d.coefficient_source,
              parse_diffusion_coefficient_source,
              "constant, linear_height");
    const bool surface_present =
        take_non_negative_double("diffusion.K_v_surface", d.K_v_surface);
    const bool top_present =
        take_non_negative_double("diffusion.K_v_top", d.K_v_top);
    if (d.coefficient_source == DiffusionCoefficientSource::LinearHeight)
    {
        if (d.scheme != DiffusionScheme::Explicit)
        {
            const auto it = yaml_.find("diffusion.coefficient_source");
            add_error(it == yaml_.end() ? 0 : it->second.line_number,
                      "diffusion.coefficient_source",
                      "linear_height is implemented only by explicit diffusion");
        }
        if (!surface_present)
            add_required_missing("diffusion.K_v_surface");
        if (!top_present)
            add_required_missing("diffusion.K_v_top");
        if (d.K_v != 0.0)
        {
            const auto it = yaml_.find("diffusion.K_v");
            add_error(it == yaml_.end() ? 0 : it->second.line_number,
                      "diffusion.K_v",
                      "must be zero when coefficient_source is linear_height");
        }
    }
    else if (surface_present || top_present)
    {
        const std::string key = surface_present
            ? "diffusion.K_v_surface" : "diffusion.K_v_top";
        const auto it = yaml_.find(key);
        add_error(it == yaml_.end() ? 0 : it->second.line_number,
                  key,
                  "is valid only when coefficient_source is linear_height");
    }
}

void V1Loader::take_surface()
{
    auto& s = config_.surface;
    take_enum("surface.scheme", s.scheme, parse_surface_scheme, "none, slab");
    take_bool("surface.apply_fluxes", s.apply_fluxes);
    take_positive_double("surface.z0m_m", s.z0m_m);
    take_positive_double("surface.z0h_m", s.z0h_m);
    const bool albedo_present     = take_double("surface.albedo", s.albedo);
    const bool emissivity_present = take_double("surface.emissivity", s.emissivity);
    // Optional prescribed skin state (CM1 tsk0). Omit to inherit the initial
    // surface-air state as the skin. Consuming the keys here also keeps them
    // from being reported as unrecognized.
    const bool tsk_present  = take_double("surface.tsk_k", s.tsk_k);
    const bool qsfc_present = take_double("surface.qsfc_kgkg", s.qsfc_kgkg);

    const auto bounded = [this](const char* key, double value, double minimum, double maximum)
    {
        if (value >= minimum && value <= maximum)
            return;
        const auto it = yaml_.find(key);
        add_error(it == yaml_.end() ? 0 : it->second.line_number,
                  key,
                  "must be in [" + std::to_string(minimum) + ", " + std::to_string(maximum) + "]");
    };
    if (albedo_present)
        bounded("surface.albedo", s.albedo, 0.0, 1.0);
    if (emissivity_present)
        bounded("surface.emissivity", s.emissivity, 0.0, 1.0);
    if (tsk_present && s.tsk_k != -1.0)
        bounded("surface.tsk_k", s.tsk_k, 150.0, 400.0);
    if (qsfc_present && s.qsfc_kgkg != -1.0)
        bounded("surface.qsfc_kgkg", s.qsfc_kgkg, 0.0, 0.1);
}

void V1Loader::take_boundary_layer()
{
    take_enum(
        "boundary_layer.scheme", config_.boundary_layer.scheme, parse_boundary_layer_scheme, "none, slab, ysu, mynn");
    take_enum(
        "boundary_layer.pbl_top_method",
        config_.boundary_layer.pbl_top_method,
        parse_boundary_layer_top_method,
        "ri, theta, tke");
    take_double(
        "boundary_layer.cadence_s", config_.boundary_layer.cadence_s);
    take_bool(
        "boundary_layer.clear_sky_only",
        config_.boundary_layer.clear_sky_only);

    if (!tmv::valid_physics_cadence(config_.boundary_layer.cadence_s))
    {
        const auto it = yaml_.find("boundary_layer.cadence_s");
        add_error(
            it == yaml_.end() ? 0 : it->second.line_number,
            "boundary_layer.cadence_s", "must exceed the 1e-12 s scheduling tolerance");
    }
    if (!config_.boundary_layer.clear_sky_only &&
        config_.boundary_layer.scheme != BoundaryLayerScheme::YSU)
    {
        const auto it = yaml_.find("boundary_layer.clear_sky_only");
        add_error(
            it == yaml_.end() ? 0 : it->second.line_number,
            "boundary_layer.clear_sky_only",
            "false requires boundary_layer.scheme=ysu");
    }

    if (config_.boundary_layer.pbl_top_method == BoundaryLayerTopMethod::TKE &&
        config_.boundary_layer.scheme != BoundaryLayerScheme::MYNN)
    {
        const auto it = yaml_.find("boundary_layer.pbl_top_method");
        add_error(
            it == yaml_.end() ? 0 : it->second.line_number,
            "boundary_layer.pbl_top_method",
            "tke requires boundary_layer.scheme=mynn");
    }

    // Surface forcing has no independent runtime slot.  Accept only choices
    // whose requested meaning survives projection to the boundary-layer
    // runtime; warning and silently enabling/disabling a closure changes the
    // selected physics.
    if (config_.surface.scheme == SurfaceScheme::None && config_.surface.apply_fluxes)
    {
        const auto it = yaml_.find("surface.apply_fluxes");
        add_error(it == yaml_.end() ? 0 : it->second.line_number,
                  "surface.apply_fluxes",
                  "cannot be true when surface.scheme=none; select "
                  "surface.scheme=slab or disable surface fluxes");
    }
    if (config_.surface.scheme == SurfaceScheme::Slab && config_.boundary_layer.scheme == BoundaryLayerScheme::None)
    {
        const auto it = yaml_.find("boundary_layer.scheme");
        add_error(it == yaml_.end() ? 0 : it->second.line_number,
                  "boundary_layer.scheme",
                  "cannot be none when surface.scheme=slab because the "
                  "surface model has no independent runtime operator; select "
                  "boundary_layer.scheme=slab explicitly");
    }
}

void V1Loader::take_radiation()
{
    auto& r = config_.radiation;
    take_enum("radiation.scheme", r.scheme, parse_radiation_scheme, "none, simple_grey, rrtmg");
    const bool forcing_present =
        take_enum("radiation.solar_forcing", r.solar_forcing, parse_radiation_solar_forcing, "idealized_diurnal");
    const bool latitude_present    = take_double("radiation.latitude_deg", r.latitude_deg);
    const bool declination_present = take_double("radiation.declination_deg", r.declination_deg);
    const bool start_time_present = take_double("radiation.local_solar_time_at_start_h", r.local_solar_time_at_start_h);
    take_bool("radiation.do_lw", r.do_lw);
    take_bool("radiation.do_sw", r.do_sw);
    take_positive_double("radiation.cadence_s", r.cadence_s);
    if (!tmv::valid_physics_cadence(r.cadence_s))
        add_error(0, "radiation.cadence_s", "must exceed the 1e-12 s scheduling tolerance");
    take_bool("radiation.clear_sky_only", r.clear_sky_only);
    const bool simple_grey_model_present = take_string(
        "radiation.simple_grey_optical_model",
        r.simple_grey_optical_model);
    const bool subcycling_present = take_bool(
        "radiation.use_subcycling", r.use_subcycling);
    const bool subcycle_count_present = take_positive_int(
        "radiation.subcycle_count", r.subcycle_count);
    const bool diagnostics_present = take_bool(
        "radiation.enable_diagnostics", r.enable_diagnostics);
    take_non_negative_double("radiation.tau_lw_ref", r.tau_lw_ref);
    take_non_negative_double("radiation.tau_sw_ref", r.tau_sw_ref);
    take_positive_double("radiation.n_lw", r.n_lw);
    take_positive_double("radiation.n_sw", r.n_sw);
    const bool lw_vapor_absorption_present = take_non_negative_double(
        "radiation.simple_grey_lw_vapor_absorption_m2kg",
        r.simple_grey_lw_vapor_absorption_m2kg);
    const bool lw_liquid_absorption_present = take_non_negative_double(
        "radiation.simple_grey_lw_liquid_absorption_m2kg",
        r.simple_grey_lw_liquid_absorption_m2kg);
    const bool lw_ice_absorption_present = take_non_negative_double(
        "radiation.simple_grey_lw_ice_absorption_m2kg",
        r.simple_grey_lw_ice_absorption_m2kg);
    const bool sw_vapor_absorption_present = take_non_negative_double(
        "radiation.simple_grey_sw_vapor_absorption_m2kg",
        r.simple_grey_sw_vapor_absorption_m2kg);
    const bool sw_liquid_absorption_present = take_non_negative_double(
        "radiation.simple_grey_sw_liquid_absorption_m2kg",
        r.simple_grey_sw_liquid_absorption_m2kg);
    const bool sw_ice_absorption_present = take_non_negative_double(
        "radiation.simple_grey_sw_ice_absorption_m2kg",
        r.simple_grey_sw_ice_absorption_m2kg);
    const bool ozone_present = take_non_negative_double("radiation.rrtmg_ozone_vmr", r.rrtmg_ozone_vmr);
    const bool co2_present   = take_non_negative_double("radiation.rrtmg_co2_vmr", r.rrtmg_co2_vmr);
    const bool ch4_present   = take_non_negative_double("radiation.rrtmg_ch4_vmr", r.rrtmg_ch4_vmr);
    const bool n2o_present   = take_non_negative_double("radiation.rrtmg_n2o_vmr", r.rrtmg_n2o_vmr);
    const bool co_present    = take_non_negative_double("radiation.rrtmg_co_vmr", r.rrtmg_co_vmr);
    const bool o2_present    = take_non_negative_double("radiation.rrtmg_o2_vmr", r.rrtmg_o2_vmr);
    const bool n2_present    = take_non_negative_double("radiation.rrtmg_n2_vmr", r.rrtmg_n2_vmr);
    const bool cfc11_present = take_non_negative_double("radiation.rrtmg_cfc11_vmr", r.rrtmg_cfc11_vmr);
    const bool cfc12_present = take_non_negative_double("radiation.rrtmg_cfc12_vmr", r.rrtmg_cfc12_vmr);
    const bool cfc22_present = take_non_negative_double("radiation.rrtmg_cfc22_vmr", r.rrtmg_cfc22_vmr);
    const bool ccl4_present  = take_non_negative_double("radiation.rrtmg_ccl4_vmr", r.rrtmg_ccl4_vmr);
    const bool liquid_radius_present =
        take_positive_double("radiation.rrtmg_liquid_effective_radius_um", r.rrtmg_liquid_effective_radius_um);
    const bool ice_radius_present =
        take_positive_double("radiation.rrtmg_ice_effective_radius_um", r.rrtmg_ice_effective_radius_um);
    take_positive_int("radiation.terrain_sw_cosine_nodes", r.terrain_sw_cosine_nodes);
    take_positive_int("radiation.terrain_sw_azimuth_nodes", r.terrain_sw_azimuth_nodes);
    take_positive_int("radiation.terrain_sw_maximum_bounces", r.terrain_sw_maximum_bounces);
    take_non_negative_double("radiation.terrain_sw_relative_residual_tolerance",
                             r.terrain_sw_relative_residual_tolerance);
    take_non_negative_double("radiation.terrain_sw_absolute_residual_tolerance_w",
                             r.terrain_sw_absolute_residual_tolerance_w);

    const auto range_error = [this](const char* key, double value, double minimum, double maximum)
    {
        if (value >= minimum && value <= maximum)
            return;
        const auto it = yaml_.find(key);
        add_error(it == yaml_.end() ? 0 : it->second.line_number,
                  key,
                  "must be in [" + std::to_string(minimum) + ", " + std::to_string(maximum) + "]");
    };
    if (latitude_present)
        range_error("radiation.latitude_deg", r.latitude_deg, -90.0, 90.0);
    if (declination_present)
        range_error("radiation.declination_deg", r.declination_deg, -23.5, 23.5);
    if (start_time_present && (r.local_solar_time_at_start_h < 0.0 || r.local_solar_time_at_start_h >= 24.0))
    {
        const auto it = yaml_.find("radiation.local_solar_time_at_start_h");
        add_error(it == yaml_.end() ? 0 : it->second.line_number,
                  "radiation.local_solar_time_at_start_h",
                  "must be in [0, 24)");
    }
    const int minimum_cosine_nodes =
        r.scheme == RadiationScheme::RRTMG ? 1 : 4;
    const int minimum_azimuth_nodes =
        r.scheme == RadiationScheme::RRTMG ? 4 : 8;
    if (r.terrain_sw_cosine_nodes < minimum_cosine_nodes ||
        r.terrain_sw_cosine_nodes > 32)
        add_error(0, "radiation.terrain_sw_cosine_nodes",
                  "is outside the selected radiation scheme's mapped shortwave range");
    if (r.terrain_sw_azimuth_nodes < minimum_azimuth_nodes ||
        r.terrain_sw_azimuth_nodes > 128 ||
        r.terrain_sw_azimuth_nodes % 4 != 0)
        add_error(0, "radiation.terrain_sw_azimuth_nodes",
                  "must be a supported multiple of four for the selected radiation scheme");
    if (r.terrain_sw_maximum_bounces > 256)
        add_error(0, "radiation.terrain_sw_maximum_bounces", "must be in [1, 256]");
    if (r.terrain_sw_relative_residual_tolerance > 1.0e-6)
        add_error(0, "radiation.terrain_sw_relative_residual_tolerance", "must be no larger than 1e-6");

    const auto maximum_error = [this](const char* key, double value, double maximum)
    {
        if (value <= maximum)
            return;
        const auto it = yaml_.find(key);
        add_error(it == yaml_.end() ? 0 : it->second.line_number,
                  key,
                  "must be no larger than " + std::to_string(maximum));
    };
    using namespace tmv::rrtmg_config_limits;
    maximum_error("radiation.rrtmg_ozone_vmr", r.rrtmg_ozone_vmr, ozone_vmr_max);
    maximum_error("radiation.rrtmg_co2_vmr", r.rrtmg_co2_vmr, co2_vmr_max);
    maximum_error("radiation.rrtmg_ch4_vmr", r.rrtmg_ch4_vmr, ch4_vmr_max);
    maximum_error("radiation.rrtmg_n2o_vmr", r.rrtmg_n2o_vmr, n2o_vmr_max);
    maximum_error("radiation.rrtmg_co_vmr", r.rrtmg_co_vmr, co_vmr_max);
    maximum_error("radiation.rrtmg_o2_vmr", r.rrtmg_o2_vmr, o2_vmr_max);
    maximum_error("radiation.rrtmg_n2_vmr", r.rrtmg_n2_vmr, n2_vmr_max);
    maximum_error("radiation.rrtmg_cfc11_vmr", r.rrtmg_cfc11_vmr, halocarbon_vmr_max);
    maximum_error("radiation.rrtmg_cfc12_vmr", r.rrtmg_cfc12_vmr, halocarbon_vmr_max);
    maximum_error("radiation.rrtmg_cfc22_vmr", r.rrtmg_cfc22_vmr, halocarbon_vmr_max);
    maximum_error("radiation.rrtmg_ccl4_vmr", r.rrtmg_ccl4_vmr, halocarbon_vmr_max);
    range_error("radiation.rrtmg_liquid_effective_radius_um",
                r.rrtmg_liquid_effective_radius_um,
                liquid_effective_radius_min_um,
                liquid_effective_radius_max_um);
    range_error("radiation.rrtmg_ice_effective_radius_um",
                r.rrtmg_ice_effective_radius_um,
                ice_effective_radius_min_um,
                ice_effective_radius_max_um);

    const auto reject_inactive_rrtmg = [this, &r](bool present, const char* key)
    {
        if (!present || r.scheme == RadiationScheme::RRTMG)
            return;
        const auto it = yaml_.find(key);
        add_error(it == yaml_.end() ? 0 : it->second.line_number,
                  key,
                  std::string("parameter is inactive for radiation.scheme=") + radiation_scheme_name(r.scheme) +
                      "; select rrtmg or remove the key");
    };
    reject_inactive_rrtmg(ozone_present, "radiation.rrtmg_ozone_vmr");
    reject_inactive_rrtmg(co2_present, "radiation.rrtmg_co2_vmr");
    reject_inactive_rrtmg(ch4_present, "radiation.rrtmg_ch4_vmr");
    reject_inactive_rrtmg(n2o_present, "radiation.rrtmg_n2o_vmr");
    reject_inactive_rrtmg(co_present, "radiation.rrtmg_co_vmr");
    reject_inactive_rrtmg(o2_present, "radiation.rrtmg_o2_vmr");
    reject_inactive_rrtmg(n2_present, "radiation.rrtmg_n2_vmr");
    reject_inactive_rrtmg(cfc11_present, "radiation.rrtmg_cfc11_vmr");
    reject_inactive_rrtmg(cfc12_present, "radiation.rrtmg_cfc12_vmr");
    reject_inactive_rrtmg(cfc22_present, "radiation.rrtmg_cfc22_vmr");
    reject_inactive_rrtmg(ccl4_present, "radiation.rrtmg_ccl4_vmr");
    reject_inactive_rrtmg(liquid_radius_present, "radiation.rrtmg_liquid_effective_radius_um");
    reject_inactive_rrtmg(ice_radius_present, "radiation.rrtmg_ice_effective_radius_um");

    const auto reject_inactive_simple_grey = [this, &r](bool present,
                                                        const char* key)
    {
        if (!present || r.scheme == RadiationScheme::SimpleGrey)
            return;
        const auto it = yaml_.find(key);
        add_error(it == yaml_.end() ? 0 : it->second.line_number,
                  key,
                  std::string("parameter is inactive for radiation.scheme=") +
                      radiation_scheme_name(r.scheme) +
                      "; select simple_grey or remove the key");
    };
    reject_inactive_simple_grey(simple_grey_model_present,
                                "radiation.simple_grey_optical_model");
    reject_inactive_simple_grey(subcycling_present,
                                "radiation.use_subcycling");
    reject_inactive_simple_grey(subcycle_count_present,
                                "radiation.subcycle_count");
    reject_inactive_simple_grey(diagnostics_present,
                                "radiation.enable_diagnostics");
    reject_inactive_simple_grey(lw_vapor_absorption_present,
                                "radiation.simple_grey_lw_vapor_absorption_m2kg");
    reject_inactive_simple_grey(lw_liquid_absorption_present,
                                "radiation.simple_grey_lw_liquid_absorption_m2kg");
    reject_inactive_simple_grey(lw_ice_absorption_present,
                                "radiation.simple_grey_lw_ice_absorption_m2kg");
    reject_inactive_simple_grey(sw_vapor_absorption_present,
                                "radiation.simple_grey_sw_vapor_absorption_m2kg");
    reject_inactive_simple_grey(sw_liquid_absorption_present,
                                "radiation.simple_grey_sw_liquid_absorption_m2kg");
    reject_inactive_simple_grey(sw_ice_absorption_present,
                                "radiation.simple_grey_sw_ice_absorption_m2kg");

    if (r.scheme != RadiationScheme::None)
    {
        if (!forcing_present)
            add_required_missing("radiation.solar_forcing");
        if (!latitude_present)
            add_required_missing("radiation.latitude_deg");
        if (!declination_present)
            add_required_missing("radiation.declination_deg");
        if (!start_time_present)
            add_required_missing("radiation.local_solar_time_at_start_h");
        if (r.do_lw && yaml_.find("surface.tsk_k") == yaml_.end())
            add_required_missing("surface.tsk_k");
        else if (r.do_lw && config_.surface.tsk_k < 150.0)
            add_error(0, "surface.tsk_k", "active longwave requires an explicit skin temperature in [150, 400] K");
        if (!r.do_lw && !r.do_sw)
            add_error(0, "radiation", "active radiation requires do_lw or do_sw");
        if (r.scheme == RadiationScheme::SimpleGrey)
        {
            const bool clear_model = r.simple_grey_optical_model ==
                "pressure_clear_sky_v1";
            const bool absorbing_model = r.simple_grey_optical_model ==
                "absorbing_water_cloud_path_v1";
            if ((!clear_model && !absorbing_model) ||
                (clear_model && !r.clear_sky_only) ||
                (absorbing_model && r.clear_sky_only))
                add_error(0, "radiation.simple_grey_optical_model",
                          "must be pressure_clear_sky_v1 with clear_sky_only=true or absorbing_water_cloud_path_v1 with clear_sky_only=false");
            if ((!r.use_subcycling && r.subcycle_count != 1) ||
                (r.use_subcycling &&
                 (r.subcycle_count < 2 || r.subcycle_count > 64)))
                add_error(0, "radiation.subcycle_count",
                          "must be one when subcycling is disabled or in [2, 64] when enabled");
        }
        if (r.scheme == RadiationScheme::RRTMG && !ozone_present)
            add_required_missing("radiation.rrtmg_ozone_vmr");
    }
}

void V1Loader::take_terrain()
{
    auto& t = config_.terrain;
    take_enum("terrain.scheme", t.scheme, parse_terrain_scheme, "none, bell, schar");
    take_enum("terrain.coordinate", t.coordinate, parse_terrain_coordinate, "btf, smoothed, cartesian");
    take_enum("terrain.horizontal_placement",
              t.horizontal_placement,
              parse_terrain_horizontal_placement,
              "grid_attached, earth_fixed");
    take_bool("terrain.compute_derivatives", t.compute_derivatives);
    take_bool("terrain.compute_metrics", t.compute_metrics);
    take_non_negative_double("terrain.smoothing_decay", t.smoothing_decay);
    const bool center_x_present = take_double(
        "terrain.feature_center_earth_x_m", t.feature_center_earth_x_m);
    const bool center_y_present = take_double(
        "terrain.feature_center_earth_y_m", t.feature_center_earth_y_m);
    take_non_negative_double("terrain.bell.h0", t.bell.h0);
    take_positive_double("terrain.bell.a", t.bell.a);
    take_bool("terrain.bell.axisymmetric", t.bell.axisymmetric);
    take_non_negative_double("terrain.schar.H", t.schar.H);
    take_positive_double("terrain.schar.a", t.schar.a);
    take_positive_double("terrain.schar.ell", t.schar.ell);

    const auto error_at = [this](const char* key, const std::string& message)
    {
        const auto it = yaml_.find(key);
        add_error(it == yaml_.end() ? 0 : it->second.line_number, key, message);
    };
    if (t.horizontal_placement == TerrainHorizontalPlacement::GridAttached)
    {
        if (center_x_present)
            error_at("terrain.feature_center_earth_x_m",
                     "is ignored for horizontal_placement=grid_attached; remove the key");
        if (center_y_present)
            error_at("terrain.feature_center_earth_y_m",
                     "is ignored for horizontal_placement=grid_attached; remove the key");
    }
    else
    {
        const auto& velocity = config_.domain_motion.frame_velocity;
        if ((velocity.u_ms != 0.0 || velocity.v_ms != 0.0)
            && (config_.domain_motion.mode != DomainMotionMode::ConstantVelocity
                || config_.grid.staggering != StaggerType::CGrid
                || config_.domain_motion.tracking.enabled || config_.domain_motion.nesting.enabled))
            error_at("terrain.horizontal_placement",
                     "continuous Earth-fixed motion requires an unnested constant-velocity C-grid");
        if (!center_x_present)
            add_required_missing("terrain.feature_center_earth_x_m");
        if (!center_y_present)
            add_required_missing("terrain.feature_center_earth_y_m");
        if (t.scheme != TerrainScheme::Bell && t.scheme != TerrainScheme::Schar)
            error_at("terrain.scheme",
                     "horizontal_placement=earth_fixed supports only bell or schar analytic terrain");
        if (!t.compute_derivatives || !t.compute_metrics)
            error_at("terrain.horizontal_placement",
                     "earth-fixed terrain requires complete derivative and metric fields");
        if (t.coordinate != TerrainCoordinate::BasicTerrainFollowing
            && t.coordinate != TerrainCoordinate::Smoothed)
            error_at("terrain.coordinate",
                     "earth-fixed terrain supports only btf or smoothed metrics");
    }

    if (t.scheme == TerrainScheme::None)
        return;

    if (!t.compute_derivatives)
    {
        error_at("terrain.compute_derivatives",
                 "active terrain requires derivative fields for metric-consistent dynamics");
    }
    if (!t.compute_metrics)
    {
        error_at("terrain.compute_metrics", "active terrain requires terrain-following metrics");
    }
    if (t.coordinate == TerrainCoordinate::Cartesian)
    {
        error_at("terrain.coordinate",
                 "active terrain requires btf or smoothed coordinates; cartesian would ignore topography");
    }

    const double computational_top = static_cast<double>(config_.grid.nz) * config_.grid.dz;
    const double scale             = std::max({1.0, std::abs(computational_top), std::abs(config_.grid.ztop)});
    const double tolerance         = 64.0 * std::numeric_limits<double>::epsilon() * scale;
    if (std::abs(config_.grid.ztop - computational_top) > tolerance)
    {
        error_at("grid.ztop",
                 "active terrain requires grid.ztop == grid.nz * grid.dz so the computational and physical domain tops "
                 "agree");
    }
    const double max_height = t.scheme == TerrainScheme::Bell ? t.bell.h0 : t.schar.H;
    if (max_height >= config_.grid.ztop)
    {
        error_at(t.scheme == TerrainScheme::Bell ? "terrain.bell.h0" : "terrain.schar.H",
                 "terrain height must remain below grid.ztop to prevent coordinate folding");
    }
}

void V1Loader::take_initial_conditions()
{
    auto& s = config_.initial_conditions.sounding;
    take_enum(
        "initial_conditions.sounding.type", s.type, parse_sounding_type, "parametric_cape, parametric_targets, file");

    constexpr const char* prefix         = "initial_conditions.sounding.";
    const std::string     cape_key       = std::string(prefix) + "cape_jkg";
    const std::string     cin_key        = std::string(prefix) + "cin_jkg";
    const std::string     lcl_key        = std::string(prefix) + "lcl_m";
    const std::string     lfc_key        = std::string(prefix) + "lfc_m";
    const std::string     el_key         = std::string(prefix) + "el_m";
    const std::string     tropopause_key = std::string(prefix) + "tropopause_m";
    const std::string     theta_key      = std::string(prefix) + "surface_theta_k";
    const std::string     qv_key         = std::string(prefix) + "moisture_qv_kgkg";
    const std::string     file_key       = std::string(prefix) + "file_path";

    const bool cape_present       = take_non_negative_double(cape_key, s.cape_jkg);
    const bool cin_present        = take_non_negative_double(cin_key, s.cin_jkg);
    const bool lcl_present        = take_non_negative_double(lcl_key, s.lcl_m);
    const bool lfc_present        = take_non_negative_double(lfc_key, s.lfc_m);
    const bool el_present         = take_non_negative_double(el_key, s.el_m);
    const bool tropopause_present = take_non_negative_double(tropopause_key, s.tropopause_m);
    const bool theta_present      = take_positive_double(theta_key, s.surface_theta_k);

    double     qv         = 0.014;
    const bool qv_present = take_non_negative_double(qv_key, qv);
    if (qv_present)
    {
        s.moisture_qv_kgkg = qv;
    }
    const bool file_present = take_string(file_key, s.file_path);

    const auto reject_present = [this](const std::string& key, bool present, const char* source)
    {
        if (!present)
            return;
        const auto it   = yaml_.find(key);
        const int  line = (it != yaml_.end()) ? it->second.line_number : 0;
        add_error(line, key, "is not valid for sounding.type=" + std::string(source));
    };

    const auto require_at_least = [this](const std::string& key, double value, double minimum)
    {
        if (value >= minimum)
            return;
        const auto         it   = yaml_.find(key);
        const int          line = (it != yaml_.end()) ? it->second.line_number : 0;
        std::ostringstream oss;
        oss << "must be >= " << minimum << " because the selected sounding source otherwise clamps it; got " << value;
        add_error(line, key, oss.str());
    };

    const auto require_closed_range = [this](const std::string& key, double value, double minimum, double maximum)
    {
        if (value >= minimum && value <= maximum)
            return;
        const auto         it   = yaml_.find(key);
        const int          line = (it != yaml_.end()) ? it->second.line_number : 0;
        std::ostringstream oss;
        oss << "must be in [" << minimum << ", " << maximum
            << "] because the selected sounding source otherwise clamps it; got " << value;
        add_error(line, key, oss.str());
    };

    switch (s.type)
    {
    case SoundingType::ParametricCAPE:
        // The procedural CAPE source has no CIN/LCL/LFC/EL target
        // solver. In particular, el_m must never be reinterpreted as its
        // tropopause boundary; that boundary has a truthful, separate key.
        reject_present(cin_key, cin_present, "parametric_cape");
        reject_present(lcl_key, lcl_present, "parametric_cape");
        reject_present(lfc_key, lfc_present, "parametric_cape");
        reject_present(el_key, el_present, "parametric_cape");
        reject_present(file_key, file_present, "parametric_cape");
        require_at_least(theta_key, s.surface_theta_k, 250.0);
        require_at_least(tropopause_key, s.tropopause_m, 8000.0);
        if (qv_present)
        {
            if (qv != 0.0)
                require_closed_range(qv_key, qv, 0.004, 0.024);
        }
        break;

    case SoundingType::ParametricTargets:
        reject_present(tropopause_key, tropopause_present, "parametric_targets");
        reject_present(file_key, file_present, "parametric_targets");
        if (lcl_present && qv_present)
        {
            const auto it   = yaml_.find(qv_key);
            const int  line = (it != yaml_.end()) ? it->second.line_number : 0;
            add_error(line,
                      qv_key,
                      "cannot be specified together with lcl_m for "
                      "sounding.type=parametric_targets; qv is an override "
                      "that would bypass the LCL target");
        }
        require_at_least(theta_key, s.surface_theta_k, 250.0);
        require_at_least(el_key, s.el_m, 8000.0);
        require_at_least(lfc_key, s.lfc_m, 50.0);
        if (s.cin_jkg > 300.0)
        {
            const auto         it   = yaml_.find(cin_key);
            const int          line = (it != yaml_.end()) ? it->second.line_number : 0;
            std::ostringstream oss;
            oss << "must be <= 300 because the parametric_targets CIN "
                   "translation saturates above that value; got "
                << s.cin_jkg;
            add_error(line, cin_key, oss.str());
        }
        if (s.lfc_m > s.el_m - 1000.0)
        {
            const auto         it   = yaml_.find(lfc_key);
            const int          line = (it != yaml_.end()) ? it->second.line_number : 0;
            std::ostringstream oss;
            oss << "must be at least 1000 m below el_m because the "
                   "underlying profile otherwise clamps the mixed-layer top; got lfc_m="
                << s.lfc_m << " and el_m=" << s.el_m;
            add_error(line, lfc_key, oss.str());
        }
        if (!qv_present)
        {
            require_at_least(lcl_key, s.lcl_m, 1.0);
            if (s.lcl_m >= 1.0 && s.surface_theta_k >= 250.0)
            {
                const double derived_qv =
                    tmv::init::parametric_target_qv_from_lcl(s.lcl_m, s.surface_theta_k, 100000.0);
                if (derived_qv < 0.004 || derived_qv > 0.024)
                {
                    const auto         it   = yaml_.find(lcl_key);
                    const int          line = (it != yaml_.end()) ? it->second.line_number : 0;
                    std::ostringstream oss;
                    oss << "implies surface qv=" << derived_qv
                        << " kg/kg outside [0.004, 0.024]; the delegated "
                           "parametric_cape source would clamp it";
                    add_error(line, lcl_key, oss.str());
                }
            }
        }
        else
        {
            require_closed_range(qv_key, qv, 0.004, 0.024);
        }
        break;

    case SoundingType::File:
        reject_present(cape_key, cape_present, "file");
        reject_present(cin_key, cin_present, "file");
        reject_present(lcl_key, lcl_present, "file");
        reject_present(lfc_key, lfc_present, "file");
        reject_present(el_key, el_present, "file");
        reject_present(tropopause_key, tropopause_present, "file");
        reject_present(theta_key, theta_present, "file");
        reject_present(qv_key, qv_present, "file");
        if (!file_present)
        {
            add_required_missing(file_key);
        }
        else if (s.file_path.empty())
        {
            const auto it   = yaml_.find(file_key);
            const int  line = (it != yaml_.end()) ? it->second.line_number : 0;
            add_error(line, file_key, "must not be empty for sounding.type=file");
        }
        break;
    }

    auto& h = config_.initial_conditions.hodograph;
    take_enum("initial_conditions.hodograph.type", h.type, parse_hodograph_type, "auto, custom, wk_param");
    take_double("initial_conditions.hodograph.u_sfc_ms", h.u_sfc_ms);
    take_double("initial_conditions.hodograph.v_sfc_ms", h.v_sfc_ms);
    take_double("initial_conditions.hodograph.u_1km_ms", h.u_1km_ms);
    take_double("initial_conditions.hodograph.v_1km_ms", h.v_1km_ms);
    take_double("initial_conditions.hodograph.u_6km_ms", h.u_6km_ms);
    take_double("initial_conditions.hodograph.v_6km_ms", h.v_6km_ms);

    auto& tr = config_.initial_conditions.trigger;
    take_enum("initial_conditions.trigger.type", tr.type, parse_trigger_type, "none, bubble, vortex_seed");

    take_double("initial_conditions.trigger.bubble.dtheta_k", tr.bubble.dtheta_k);
    take_non_negative_double(
        "initial_conditions.trigger.bubble.cloud_water_kgkg",
        tr.bubble.cloud_water_kgkg);
    take_positive_double("initial_conditions.trigger.bubble.radius_km", tr.bubble.radius_km);
    take_double("initial_conditions.trigger.bubble.center_x_m", tr.bubble.center_x_m);
    take_double("initial_conditions.trigger.bubble.center_y_m", tr.bubble.center_y_m);
    take_double("initial_conditions.trigger.bubble.center_z_m", tr.bubble.center_z_m);

    take_positive_double("initial_conditions.trigger.vortex_seed.r_max_m", tr.vortex_seed.r_max_m);
    take_double("initial_conditions.trigger.vortex_seed.v_max_ms", tr.vortex_seed.v_max_ms);
    take_positive_double("initial_conditions.trigger.vortex_seed.z_top_m", tr.vortex_seed.z_top_m);
    take_double("initial_conditions.trigger.vortex_seed.center_x_m", tr.vortex_seed.center_x_m);
    take_double("initial_conditions.trigger.vortex_seed.center_y_m", tr.vortex_seed.center_y_m);
    take_double("initial_conditions.trigger.vortex_seed.center_z_m", tr.vortex_seed.center_z_m);
}

void V1Loader::take_output()
{
    auto& o = config_.output;
    take_enum("output.preset",
              o.preset,
              parse_output_preset,
              "minimal, radar_ready, radar_zdr_exploratory, radar_polarimetric_ready, science_validation, presentation, studio_full, dynamics_full");
    take_enum("output.format", o.format, parse_output_format, "npy_3d, netcdf, zfp_low, zfp_medium, zfp_high");
    if (o.format == OutputFormat::Netcdf)
    {
#ifndef HAVE_NETCDF
        const auto it   = yaml_.find("output.format");
        const int  line = (it != yaml_.end()) ? it->second.line_number : 0;
        add_error(
            line, "output.format", "netcdf output is unavailable in this binary; rebuild with libnetcdf or use npy_3d");
#endif
    }
    take_bool("output.async_io", o.async_io);
    take_bool("output.live_shm", o.live_shm);
    std::string live_shm_fields;
    if (take_string("output.live_shm_fields", live_shm_fields))
        o.live_shm_fields = parse_string_list(live_shm_fields);
    take_positive_double("output.live_shm_interval_s",
                         o.live_shm_interval_s);
    take_bool("output.live_shm_full_rate", o.live_shm_full_rate);
    std::string live_field_error;
    if (!validate_live_shm_fields(o.live_shm_fields, live_field_error))
    {
        const auto it = yaml_.find("output.live_shm_fields");
        add_error(it != yaml_.end() ? it->second.line_number : 0,
                  "output.live_shm_fields", live_field_error);
    }
    take_enum("output.qualification_diagnostics.preset",
              o.qualification_diagnostics_preset,
              parse_qualification_diagnostics_preset,
              "none, moving_reference_v1");
    take_positive_int("output.qualification_diagnostics.interval_s", o.qualification_diagnostics_interval_s);
    take_bool("output.trajectories", o.trajectories);
    take_non_negative_double("output.trajectory_seed_time_s", o.trajectory_seed_time_s);
    take_positive_int("output.trajectory_count", o.trajectory_count);
    take_non_negative_double("output.trajectory_sample_interval_s",
                             o.trajectory_sample_interval_s);
    take_positive_int("output.trajectory_max_samples_per_parcel",
                      o.trajectory_max_samples_per_parcel);
    take_non_negative_int("output.netcdf_deflate_level", o.netcdf_deflate_level);
    if (o.netcdf_deflate_level > 9)
    {
        const auto it = yaml_.find("output.netcdf_deflate_level");
        add_error(it != yaml_.end() ? it->second.line_number : 0, "output.netcdf_deflate_level", "must be in [0, 9]");
    }
    take_bool("output.netcdf_shuffle", o.netcdf_shuffle);
    take_positive_int("output.netcdf_target_chunk_kib", o.netcdf_target_chunk_kib);

    // ZFP compression tuning (consulted only when format is a Zfp* tier) +
    // tiered write cadence. Consuming these here also keeps a Zfp config from
    // tripping report_unknown_keys (§19.6 no-silent-runs).
    take_non_negative_double("output.zfp_tolerance", o.zfp_tolerance);
    take_string("output.zfp_mode", o.zfp_mode);
    take_positive_int("output.zfp_precision_bits", o.zfp_precision_bits);
    take_positive_int("output.zfp_rate_bps", o.zfp_rate_bps);
    take_non_negative_int("output.zfp_keyframe_interval", o.zfp_keyframe_interval);
    take_non_negative_double("output.zfp_sparse_threshold", o.zfp_sparse_threshold);
    take_bool("output.zfp_float16_prefilter", o.zfp_float16_prefilter);
    take_bool("output.zfp_predictive_delta", o.zfp_predictive_delta);
    take_string("output.zfp_field_execution", o.zfp_field_execution);
    if (o.zfp_field_execution != "serial" &&
        o.zfp_field_execution != "field_parallel")
    {
        const auto it = yaml_.find("output.zfp_field_execution");
        add_error(it != yaml_.end() ? it->second.line_number : 0,
                  "output.zfp_field_execution",
                  "must be serial or field_parallel");
    }
    take_non_negative_int("output.zfp_field_workers", o.zfp_field_workers);
    take_positive_int("output.zfp_field_scratch_mib", o.zfp_field_scratch_mib);
    if (o.zfp_field_workers > 16)
    {
        const auto it = yaml_.find("output.zfp_field_workers");
        add_error(it != yaml_.end() ? it->second.line_number : 0,
                  "output.zfp_field_workers", "must be in [0, 16]");
    }
    if (o.zfp_field_scratch_mib > 4096)
    {
        const auto it = yaml_.find("output.zfp_field_scratch_mib");
        add_error(it != yaml_.end() ? it->second.line_number : 0,
                  "output.zfp_field_scratch_mib", "must be in [1, 4096]");
    }
    take_bool("output.zfp_per_field_tolerances", o.zfp_per_field_tolerances);
    take_non_negative_double("output.zfp_tight_tolerance", o.zfp_tight_tolerance);
    take_non_negative_double("output.zfp_moderate_tolerance", o.zfp_moderate_tolerance);
    take_non_negative_double("output.zfp_loose_tolerance", o.zfp_loose_tolerance);
    const std::string        field_tolerance_prefix = "output.zfp_field_tolerance.";
    std::vector<std::string> field_tolerance_keys;
    for (const auto& [key, field] : yaml_)
    {
        (void)field;
        if (key.rfind(field_tolerance_prefix, 0) == 0)
            field_tolerance_keys.push_back(key);
    }
    for (const std::string& key : field_tolerance_keys)
    {
        const std::string field = key.substr(field_tolerance_prefix.size());
        if (field.empty())
        {
            const auto it = yaml_.find(key);
            consumed_.insert(key);
            add_error(it == yaml_.end() ? 0 : it->second.line_number, key, "field name must not be empty");
            continue;
        }
        double tolerance = 0.0;
        if (take_positive_double(key, tolerance))
            o.zfp_field_tolerances[field] = tolerance;
    }
    take_bool("output.tiered_write_cadence", o.tiered_write_cadence);
    take_positive_int("output.write_cadence_medium_s", o.write_cadence_medium_s);
    take_positive_int("output.write_cadence_slow_s", o.write_cadence_slow_s);
    take_non_negative_int("output.keep_last_frames", o.keep_last_frames);
    take_non_negative_int("output.maximum_retained_history_files",
                          o.maximum_retained_history_files);
    take_non_negative_int("output.minimum_free_space_mib", o.minimum_free_space_mib);

    const bool is_zfp =
        o.format == OutputFormat::ZfpLow || o.format == OutputFormat::ZfpMedium || o.format == OutputFormat::ZfpHigh;
#ifndef HAVE_ZFP
    if (is_zfp)
    {
        const auto it = yaml_.find("output.format");
        add_error(it != yaml_.end() ? it->second.line_number : 0,
                  "output.format",
                  "ZFP output is unavailable in this binary; rebuild with ZFP=1 or use npy_3d/netcdf");
    }
#endif
    if (is_zfp && o.zfp_mode != "accuracy" && o.zfp_mode != "precision" && o.zfp_mode != "rate")
    {
        const auto it = yaml_.find("output.zfp_mode");
        add_error(it != yaml_.end() ? it->second.line_number : 0,
                  "output.zfp_mode",
                  "must be one of: accuracy, precision, rate");
    }
    if (o.zfp_precision_bits < 1 || o.zfp_precision_bits > 32)
    {
        const auto it = yaml_.find("output.zfp_precision_bits");
        add_error(it != yaml_.end() ? it->second.line_number : 0, "output.zfp_precision_bits", "must be in [1, 32]");
    }
    if (o.preset == OutputPreset::RadarReady &&
        (!config_.radar.enabled || config_.radar.products !=
            std::vector<std::string>{"reflectivity_dbz", "radial_velocity"}))
    {
        const auto it = yaml_.find("output.preset");
        add_error(it == yaml_.end() ? 0 : it->second.line_number,
                  "output.preset",
                  "radar_ready requires radar products [reflectivity_dbz, radial_velocity]");
    }
    if (o.preset == OutputPreset::RadarPolarimetricReady &&
        (!config_.radar.enabled || config_.radar.products !=
            std::vector<std::string>{"zdr", "kdp", "rhohv"} ||
         config_.radar.zdr_tier != "zhang2021_rain_mp"))
    {
        const auto it = yaml_.find("output.preset");
        add_error(it == yaml_.end() ? 0 : it->second.line_number,
                  "output.preset",
                  "radar_polarimetric_ready requires products [zdr, kdp, rhohv] and zhang2021_rain_mp");
    }
    const auto radar_selected = [&](const char* product)
    {
        return std::find(
            config_.radar.products.begin(), config_.radar.products.end(),
            product) != config_.radar.products.end();
    };
    const bool output_reflectivity = o.preset == OutputPreset::RadarReady;
    const bool output_velocity = o.preset == OutputPreset::RadarReady;
    const bool polarimetric_ready =
        o.preset == OutputPreset::RadarPolarimetricReady;
    const bool output_zdr = o.preset == OutputPreset::RadarZdrExploratory ||
        polarimetric_ready;
    for (const auto& requirement : {
             std::pair<bool, const char*>{output_reflectivity, "reflectivity_dbz"},
             {output_velocity, "radial_velocity"},
             {output_zdr, "zdr"},
             {polarimetric_ready, "kdp"},
             {polarimetric_ready, "rhohv"}})
    {
        if (requirement.first &&
            (!config_.radar.enabled || !radar_selected(requirement.second)))
        {
            const auto it = yaml_.find("output.preset");
            add_error(it == yaml_.end() ? 0 : it->second.line_number,
                      "output.preset",
                      std::string("requests ") + requirement.second +
                          " but radar.products does not select it");
        }
    }
}

void V1Loader::take_radar()
{
    auto& radar = config_.radar;
    take_bool("radar.enabled", radar.enabled);
    std::string products;
    const bool products_present = take_string("radar.products", products);
    if (products_present)
        radar.products = parse_string_list(products);
    take_string("radar.reflectivity_tier", radar.reflectivity_tier);
    take_string("radar.velocity_sampling", radar.velocity_sampling);
    take_string("radar.zdr_tier", radar.zdr_tier);
    take_double("radar.location.x_m", radar.radar_x_m);
    take_double("radar.location.y_m", radar.radar_y_m);
    take_double("radar.location.z_m", radar.radar_z_m);
    take_bool("radar.apply_scatterer_correction", radar.apply_scatterer_correction);

    const std::unordered_set<std::string> allowed = {
        "reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"};
    std::unordered_set<std::string> seen;
    for (const std::string& product : radar.products)
    {
        if (allowed.count(product) == 0 || !seen.insert(product).second)
        {
            const auto it = yaml_.find("radar.products");
            add_error(it == yaml_.end() ? 0 : it->second.line_number,
                      "radar.products",
                      allowed.count(product) == 0
                          ? "unknown radar product '" + product + "'"
                          : "duplicate radar product '" + product + "'");
        }
    }
    if (radar.enabled && radar.products.empty())
    {
        const auto it = yaml_.find("radar.enabled");
        add_error(it == yaml_.end() ? 0 : it->second.line_number,
                  "radar.products", "must select at least one product when radar.enabled=true");
    }
    if (!radar.enabled && !radar.products.empty())
    {
        const auto it = yaml_.find("radar.products");
        add_error(it == yaml_.end() ? 0 : it->second.line_number,
                  "radar.products", "requires radar.enabled=true");
    }
    if (radar.reflectivity_tier != "fast_da" && radar.reflectivity_tier != "psd_moment")
    {
        const auto it = yaml_.find("radar.reflectivity_tier");
        add_error(it == yaml_.end() ? 0 : it->second.line_number,
                  "radar.reflectivity_tier", "must be fast_da or psd_moment");
    }
    if (radar.velocity_sampling != "point" && radar.velocity_sampling != "beam_volume")
    {
        const auto it = yaml_.find("radar.velocity_sampling");
        add_error(it == yaml_.end() ? 0 : it->second.line_number,
                  "radar.velocity_sampling", "must be point or beam_volume");
    }
    if (radar.zdr_tier != "simple" && radar.zdr_tier != "polarimetric_fo" &&
        radar.zdr_tier != "zhang2021_rain_mp")
    {
        const auto it = yaml_.find("radar.zdr_tier");
        add_error(it == yaml_.end() ? 0 : it->second.line_number,
                  "radar.zdr_tier", "must be simple, polarimetric_fo, or zhang2021_rain_mp");
    }
    const bool selects_kdp_or_rhohv =
        seen.count("kdp") != 0 || seen.count("rhohv") != 0;
    if (selects_kdp_or_rhohv && radar.zdr_tier != "zhang2021_rain_mp")
    {
        const auto it = yaml_.find("radar.zdr_tier");
        add_error(it == yaml_.end() ? 0 : it->second.line_number,
                  "radar.zdr_tier",
                  "kdp and rhohv require zhang2021_rain_mp");
    }
}

void V1Loader::take_extensions()
{
    auto& e = config_.extensions;
    take_enum(
        "extensions.compute.backend", e.compute.backend, parse_compute_backend, "cpu, gpu, vulkan, vk, metal, mtl");
    take_enum("extensions.compute.memory_admission",
              e.compute.memory_admission,
              parse_memory_admission_policy,
              "warn, reject");
    take_bool("extensions.compute.allow_fallback", e.compute.allow_fallback);
    take_bool("extensions.compute.validate_parity", e.compute.validate_parity);
    take_bool("extensions.compute.cuda_assist", e.compute.cuda_assist);
    if (e.compute.cuda_assist && e.compute.backend != ComputeBackend::GPU)
        add_error(0, "extensions.compute.cuda_assist", "CUDA assist requires the Vulkan backend on a compatible NVIDIA GPU");
    take_bool("extensions.compute.gpu_apply_microphysics", e.compute.gpu_apply_microphysics);
    take_bool("extensions.compute.gpu_apply_slow", e.compute.gpu_apply_slow);
    take_bool("extensions.compute.gpu_apply_sgs", e.compute.gpu_apply_sgs);
    take_bool("extensions.solver.slow_rk2", e.solver.slow_rk2);
    take_double("extensions.solver.lid_bc_gamma", e.solver.lid_bc_gamma);
    take_bool("extensions.solver.axis_u_m1_filter", e.solver.axis_u_m1_filter);
    take_enum("extensions.chaos.scheme",
              e.chaos.scheme,
              parse_chaos_scheme,
              "none, initial_conditions, boundary_layer, full_stochastic");
    take_int("extensions.chaos.seed", e.chaos.seed);
    take_int("extensions.chaos.member_id", e.chaos.member_id);
    take_bool("extensions.chaos.native_generation_required",
              e.chaos.native_generation_required);
    std::string chaos_targets;
    if (take_string("extensions.chaos.apply_to_ic", chaos_targets))
        e.chaos.apply_to_ic = parse_string_list(chaos_targets);
    if (take_string("extensions.chaos.apply_to_tendencies", chaos_targets))
        e.chaos.apply_to_tendencies = parse_string_list(chaos_targets);

    const auto validate_targets = [this](
        const std::string& key,
        const std::vector<std::string>& targets,
        const std::unordered_set<std::string>& allowed)
    {
        std::unordered_set<std::string> seen;
        for (const std::string& target : targets)
        {
            if (target.empty() || allowed.count(target) == 0)
            {
                const auto it = yaml_.find(key);
                add_error(it == yaml_.end() ? 0 : it->second.line_number,
                          key, "contains unsupported chaos target '" + target + "'");
            }
            else if (!seen.insert(target).second)
            {
                const auto it = yaml_.find(key);
                add_error(it == yaml_.end() ? 0 : it->second.line_number,
                          key, "contains duplicate chaos target '" + target + "'");
            }
        }
    };
    validate_targets(
        "extensions.chaos.apply_to_ic", e.chaos.apply_to_ic,
        {"u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh"});
    validate_targets(
        "extensions.chaos.apply_to_tendencies", e.chaos.apply_to_tendencies,
        {"microphysics", "pbl", "diffusion"});

    const auto take_amplitude_map = [this](
        const std::string& prefix,
        std::unordered_map<std::string, double>& values,
        const std::unordered_set<std::string>& allowed)
    {
        std::vector<std::string> keys;
        for (const auto& [key, field] : yaml_)
        {
            (void)field;
            if (key.rfind(prefix, 0) == 0) keys.push_back(key);
        }
        for (const std::string& key : keys)
        {
            const std::string target = key.substr(prefix.size());
            if (target.empty() || allowed.count(target) == 0)
            {
                const auto it = yaml_.find(key);
                consumed_.insert(key);
                add_error(it == yaml_.end() ? 0 : it->second.line_number,
                          key, "unknown chaos amplitude target '" + target + "'");
                continue;
            }
            double amplitude = 0.0;
            if (take_non_negative_double(key, amplitude))
                values[target] = amplitude;
        }
    };
    take_amplitude_map(
        "extensions.chaos.sigma_ic.", e.chaos.sigma_ic,
        {"u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh"});
    take_amplitude_map(
        "extensions.chaos.alpha_tend.", e.chaos.alpha_tend,
        {"microphysics", "pbl", "diffusion"});
    take_enum("extensions.execution.mode", e.execution.mode, parse_execution_mode, "exploratory, qualified");
    const bool maturity_ack_present =
        take_bool("extensions.execution.acknowledge_maturity_limits", e.execution.acknowledge_maturity_limits);
    if (e.execution.mode == ExecutionMode::Qualified &&
        maturity_ack_present && !e.execution.acknowledge_maturity_limits)
    {
        add_error(
            0,
            "extensions.execution.acknowledge_maturity_limits",
            "an explicit false maturity acknowledgment cannot select qualified execution; remove the obsolete key");
    }
    else if (e.execution.mode == ExecutionMode::Exploratory && e.execution.acknowledge_maturity_limits)
    {
        add_error(0,
                  "extensions.execution.acknowledge_maturity_limits",
                  "maturity acknowledgment is active only for qualified execution; set it to false or remove the key");
    }
    take_non_negative_int("extensions.checkpoint.every_s", e.checkpoint.every_s);
    take_non_negative_int("extensions.checkpoint.keep_last", e.checkpoint.keep_last);
    take_non_negative_int("extensions.checkpoint.minimum_free_space_mib", e.checkpoint.minimum_free_space_mib);
    take_string("extensions.checkpoint.directory", e.checkpoint.directory);
    take_string("extensions.checkpoint.resume_from", e.checkpoint.resume_from);
    take_enum(
        "extensions.validation.guard_mode", e.validation.guard_mode, parse_validation_guard_mode, "off, warn, error");
    // Omitted diagnostics cadence retains the ConfigV1 compatibility sentinel
    // (-1). Explicit YAML values are deliberately non-negative: 0 is off and
    // positive values are a one-based step cadence.
    take_non_negative_int("extensions.diagnostics.scientific_every_steps", e.diagnostics.scientific_every_steps);
    take_bool("extensions.diagnostics.deterministic_reductions", e.diagnostics.deterministic_reductions);
    bool performance_timing = false;
    if (take_bool("extensions.performance.timing", performance_timing))
    {
        e.performance.timing = performance_timing;
    }
    int report_every_steps = 0;
    if (take_non_negative_int("extensions.performance.report_every_steps", report_every_steps))
    {
        e.performance.report_every_steps = report_every_steps;
    }
    int warmup_steps = 0;
    if (take_non_negative_int("extensions.performance.warmup_steps", warmup_steps))
    {
        e.performance.warmup_steps = warmup_steps;
    }
}

void V1Loader::report_unknown_keys()
{
    for (const auto& [key, field] : yaml_)
    {
        if (consumed_.count(key) == 0)
        {
            add_error(field.line_number, key, "unrecognized v1 key");
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Top-level load
// ─────────────────────────────────────────────────────────────

LoadResult V1Loader::load(const std::string& path, std::unordered_map<std::string, YamlField> fields)
{
    file_ = path;
    yaml_ = std::move(fields);

    take_schema_stamp();
    // If the schema stamp is missing or wrong, bail before walking sections —
    // continuing would produce a wall of misleading "unknown key" errors on
    // what is actually a v0 (or unknown-format) config.
    if (!errors_.empty())
    {
        return LoadResult{std::nullopt, errors_};
    }

    take_grid();
    take_time();
    take_domain_motion();
    take_dynamics();
    take_boundary_conditions();
    take_damping();
    take_coriolis();
    take_microphysics();
    take_turbulence();
    take_advection();
    take_diffusion();
    take_surface();
    take_boundary_layer();
    take_radiation();
    take_terrain();
    take_initial_conditions();
    take_radar();
    take_output();
    take_extensions();

    if (config_.output.qualification_diagnostics_preset == QualificationDiagnosticsPreset::MovingReferenceV1)
    {
        const auto add_qualification_error = [&](const std::string& message)
        {
            const auto it = yaml_.find("output.qualification_diagnostics.preset");
            add_error(
                it != yaml_.end() ? it->second.line_number : 0, "output.qualification_diagnostics.preset", message);
        };
        const double interval = static_cast<double>(config_.output.qualification_diagnostics_interval_s);
        const int    level_time_ratio =
            config_.domain_motion.nesting.enabled ? config_.domain_motion.nesting.parent_time_step_ratio : 1;
        const double effective_dt =
            level_time_ratio > 0 ? config_.time.dt_large / static_cast<double>(level_time_ratio) : 0.0;
        const bool root_cadence_reachable =
            qualification_diagnostics_cadence_reachable(interval, config_.time.dt_large);
        const bool comparison_cadence_reachable = qualification_diagnostics_cadence_reachable(interval, effective_dt);
        if (config_.grid.coordinate_system != CoordinateSystem::Cartesian ||
            config_.terrain.scheme != TerrainScheme::None)
        {
            add_qualification_error("moving_reference_v1 requires flat Cartesian diagnostics");
        }
        if (!config_.domain_motion.tracking.enabled || config_.domain_motion.tracking.interval_s != interval)
        {
            add_qualification_error("moving_reference_v1 requires enabled storm tracking at the same interval_s");
        }
        if (config_.time.adaptive_dt || !root_cadence_reachable || !comparison_cadence_reachable)
        {
            add_qualification_error("moving_reference_v1 requires fixed time stepping where each of root macrostep dt "
                                    "and comparison-level dt exactly reaches interval_s");
        }
        if (config_.time.run_time <= 0.0 || std::fmod(config_.time.run_time, interval) != 0.0)
        {
            add_qualification_error("moving_reference_v1 requires positive run_time exactly divisible by interval_s");
        }
    }
    else if (yaml_.count("output.qualification_diagnostics.interval_s") != 0)
    {
        const auto& field = yaml_.at("output.qualification_diagnostics.interval_s");
        add_error(
            field.line_number, "output.qualification_diagnostics.interval_s", "requires preset=moving_reference_v1");
    }

    if (config_.extensions.execution.mode == ExecutionMode::Qualified &&
        config_.domain_motion.lifetime_planning.enabled &&
        config_.domain_motion.lifetime_planning.action != LifetimePlanningAction::Error)
    {
        const auto it = yaml_.find("domain_motion.lifetime_planning.action");
        add_error(it != yaml_.end() ? it->second.line_number : 0,
                  "domain_motion.lifetime_planning.action",
                  "qualified execution requires action=error; a predicted "
                  "truncated storm lifetime cannot be warning-only evidence");
    }
    report_unknown_keys();

    if (!errors_.empty())
    {
        return LoadResult{std::nullopt, errors_};
    }
    return LoadResult{std::move(config_), {}};
}

} // namespace

LoadResult load_v1_config(const std::string& path)
{
    auto fields = parse_yaml_with_lines(path);
    if (!fields) return LoadResult{std::nullopt, {{path, 0, "", "could not open config file"}}};
    return load_v1_config_fields(path, std::move(*fields));
}

LoadResult load_v1_config_fields(const std::string& path,
    std::unordered_map<std::string, YamlField> fields)
{
    V1Loader loader;
    return loader.load(path, std::move(fields));
}

std::string format_errors(const std::vector<LoadError>& errors)
{
    std::ostringstream oss;
    for (size_t i = 0; i < errors.size(); ++i)
    {
        const auto& e = errors[i];
        oss << e.file;
        if (e.line > 0)
            oss << ':' << e.line;
        oss << ": ";
        if (!e.key.empty())
            oss << e.key << ": ";
        oss << e.message;
        if (i + 1 < errors.size())
            oss << '\n';
    }
    return oss.str();
}

} // namespace tmv::init::config
