#include "diagnostics/resolved_diagnostics.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "diagnostics/energy_budget.hpp"
#include "diagnostics/momentum_budget.hpp"
#include "diagnostics/moisture_budget.hpp"
#include "core/infra/nested_scalar_feedback.hpp"
#include "core/runtime/earth_fixed_step_transaction.hpp"
#include "diagnostics/physical_reference.hpp"
/**
 * @file headless_runtime.cpp
 * @brief Core runtime implementation for the tornado model.
 *
 * Provides simulation orchestration and subsystem integration
 * for dynamics, numerics, physics, and runtime execution paths.
 * This file belongs to the primary src/core execution layer.
 */

#include "core/runtime/headless_runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "numerics/advection/advection.hpp"
#include "compute/parity_check.hpp"
#include "diagnostics/conservation_budget.hpp"
#include "diagnostics/derived_field_math.hpp"
#include "diagnostics/derived_slice_compute.hpp"
#include "diagnostics/momentum_budget.hpp"
#include "diagnostics/energy_budget.hpp"
#include "diagnostics/moisture_budget.hpp"
#include "diagnostics/storm_rotation.hpp"
#include "diagnostics/storm_tracker.hpp"
#include "diagnostics/field_contract.hpp"
#include "diagnostics/field_validation.hpp"
#include "diagnostics/level_budget_csv.hpp"
#include "diagnostics/tranche_product_registry.hpp"
#include "dynamics/dynamics_base.hpp"
#include "microphysics/runtime.hpp"
#include "boundary_conditions/runtime.hpp"
#include "boundary_layer/runtime.hpp"
#include "core/field/numerical_integrity.hpp"
#include "core/field/field_snapshot.hpp"
#include "core/infra/hardware_info.hpp"
#include "core/infra/domain_motion.hpp"
#include "core/infra/domain_recycling_state.hpp"
#include "core/infra/nested_execution.hpp"
#include "core/infra/nested_feedback.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/infra/sha256.hpp"
#include "core/output/core_volume_writer.hpp"
#include "core/output/derived_slice_scratch.hpp"
#include "core/output/derived_volume_writer.hpp"
#include "core/output/direct_surface_writer.hpp"
#include "core/output/field_registry.hpp"
#include "core/output/npy_writer.hpp"
#include "core/output/output_writer.hpp"
#include "core/output/retention_policy.hpp"
#include "core/output/domain_recycling_swath.hpp"
#include "core/output/stagger_interpolation.hpp"
#include "core/output/velocity_staging.hpp"
#include "core/output/step_manifest.hpp"
#include "core/output/zfp_field_storage.hpp"
#include "core/output/tranche_writer.hpp"
// hardware_info.hpp: non-mutating memory preflight runs in load_config()
// before field allocation; requested dimensions and spacing are never changed.
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/flat_native_tvd_step_transaction.hpp"
#include "core/runtime/runtime_cadence.hpp"
#include "core/runtime/simulation_checkpoint.hpp"
#include "core/runtime/termination_signal.hpp"
#ifdef NIMBUS_ALLOCATION_TELEMETRY
#include "core/infra/allocation_telemetry.hpp"
#endif
#include "core/runtime/simulation_state.hpp"
#include "core/output/shm_writer.hpp"
#include "core/runtime/process_state.hpp"
#include "radiation/runtime.hpp"
#include "radar/runtime.hpp"
#include "terrain/conservative_transport.hpp"
#include "compute/compute_backend.hpp"
#include "compute/kernel_manifest.hpp"
#include "util/string_utils.hpp"
#include "util/log.hpp"

namespace
{
using PerfClock = std::chrono::steady_clock;
using tmv::derived::fill_reflectivity_dbz_field;
using tmv::derived::reflectivity_dbz_bounds;

template <class Runtime>
void configure_two_way_level_flux_capture(Runtime&                         runtime,
                                          const GridStorageState&          grid,
                                          tmv::CartesianInterfaceFootprint footprint,
                                          int                              tangential_ratio,
                                          bool                             moist = false)
{
    auto& advection = runtime.advection;
    advection.reset_for_shape(grid.nr, grid.nth, grid.nz);
    advection.theta_interface_flux_register.configure(
        grid.nr, grid.nth, grid.nz, footprint, tangential_ratio, tangential_ratio);
    advection.dry_mass_interface_flux_register.configure(
        grid.nr, grid.nth, grid.nz, footprint, tangential_ratio, tangential_ratio);
    tmv::CartesianInterfaceFootprint momentum_footprint = footprint;
    momentum_footprint.k_end                            = grid.nz - 1;
    runtime.dynamics.momentum_interface_flux_register.configure(
        grid.nr, grid.nth, grid.nz, momentum_footprint, tangential_ratio, tangential_ratio);
    if (moist)
        advection.moist_interface.configure(grid.nr, grid.nth, grid.nz, footprint, tangential_ratio);
    else
        advection.moist_interface = {};
    if (!runtime.microphysics.scheme)
        throw std::runtime_error("two-way flux capture requires its microphysics owner");
    advection.private_interface.configure(grid.nr,
                                          grid.nth,
                                          grid.nz,
                                          footprint,
                                          tangential_ratio,
                                          runtime.microphysics.scheme->prognostic_scalars_for_transport());
    advection.theta_interface_flux_capture_enabled           = true;
    advection.dry_mass_interface_flux_capture_enabled        = true;
    runtime.dynamics.momentum_interface_flux_capture_enabled = true;
}

std::array<double, 5> nested_two_way_child_inventory(const NestedExecutionStorage& nested)
{
    return nested.active_terrain.ready
               ? nested_two_way_child_inventory_mapped(nested.active_terrain.geometry, nested.child.fields)
               : nested_two_way_child_inventory_flat(
                     nested.relationship, nested.child.process_grid_metrics.dz, nested.child.fields);
}

template <class Runtime>
void reset_two_way_level_flux_capture(Runtime& runtime) noexcept
{
    runtime.advection.theta_interface_flux_register.reset_interval();
    runtime.advection.moist_interface.reset_interval();
    runtime.advection.private_interface.reset_interval();
    runtime.advection.dry_mass_interface_flux_register.reset_interval();
    runtime.dynamics.momentum_interface_flux_register.reset_interval();
}

std::array<double, 5> two_way_interval_interface_source(
    const NestedChildRelaxationReport& begin,
    const NestedChildRelaxationReport& end)
{
    return {
        end.theta.committed_dry_mass_change_kg -
            begin.theta.committed_dry_mass_change_kg,
        end.theta.committed_carrier_change -
            begin.theta.committed_carrier_change,
        end.momentum.committed_x_change_kgms -
            begin.momentum.committed_x_change_kgms,
        end.momentum.committed_y_change_kgms -
            begin.momentum.committed_y_change_kgms,
        end.momentum.committed_z_change_kgms -
            begin.momentum.committed_z_change_kgms};
}

struct CheckpointSidecar
{
    std::string   run_id;
    std::string   parent_run_id;
    std::uint64_t completed_steps   = 0;
    double        simulation_time_s = 0.0;
    std::uint64_t checkpoint_bytes  = 0;
    std::string   checkpoint_fnv1a64;
    std::string   checkpoint_sha256;
};

std::string hex_u64(std::uint64_t value)
{
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

std::filesystem::path checkpoint_sidecar_path(const std::filesystem::path& checkpoint)
{
    return checkpoint.string() + ".json";
}

bool load_checkpoint_sidecar(const std::filesystem::path& checkpoint, CheckpointSidecar& sidecar, std::string& error)
{
    std::ifstream input(checkpoint_sidecar_path(checkpoint), std::ios::binary);
    if (!input)
    {
        error = "cannot read checkpoint sidecar '" + checkpoint_sidecar_path(checkpoint).string() + "'";
        return false;
    }
    const std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const auto        string_value = [&](const char* key, std::string& value)
    {
        const std::regex pattern(std::string("\\\"") + key + R"(\"\s*:\s*\"([^\"]*)\")");
        std::smatch      match;
        if (!std::regex_search(json, match, pattern))
            return false;
        value = match[1].str();
        return true;
    };
    const auto uint_value = [&](const char* key, std::uint64_t& value)
    {
        const std::regex pattern(std::string("\\\"") + key + R"(\"\s*:\s*([0-9]+))");
        std::smatch      match;
        if (!std::regex_search(json, match, pattern))
            return false;
        try
        {
            value = std::stoull(match[1].str());
        }
        catch (...)
        {
            return false;
        }
        return true;
    };
    const auto double_value = [&](const char* key, double& value)
    {
        const std::regex pattern(std::string("\\\"") + key +
                                 R"(\"\s*:\s*([-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?))");
        std::smatch      match;
        if (!std::regex_search(json, match, pattern))
            return false;
        try
        {
            value = std::stod(match[1].str());
        }
        catch (...)
        {
            return false;
        }
        return true;
    };
    std::string   schema;
    std::string   checkpoint_file;
    std::uint64_t schema_version = 0;
    if (!string_value("schema", schema) || schema != "nimbus.checkpoint.lifecycle" ||
        !uint_value("schema_version", schema_version) || schema_version != 1 ||
        !string_value("checkpoint_file", checkpoint_file) || checkpoint_file != checkpoint.filename().string() ||
        !string_value("run_id", sidecar.run_id) || !string_value("parent_run_id", sidecar.parent_run_id) ||
        !uint_value("completed_steps", sidecar.completed_steps) ||
        !double_value("simulation_time_s", sidecar.simulation_time_s) ||
        !uint_value("checkpoint_bytes", sidecar.checkpoint_bytes) ||
        !string_value("checkpoint_fnv1a64", sidecar.checkpoint_fnv1a64))
    {
        error = "checkpoint sidecar identity/schema mismatch";
        return false;
    }
    if (sidecar.run_id.empty() || !std::isfinite(sidecar.simulation_time_s) || sidecar.simulation_time_s < 0.0 ||
        sidecar.checkpoint_fnv1a64.size() != 16)
    {
        error = "checkpoint sidecar contains invalid lifecycle metadata";
        return false;
    }
    // SHA-256 was added additively to lifecycle v1. Legacy sidecars remain
    // readable; every newly committed sidecar carries and verifies it.
    string_value("checkpoint_sha256", sidecar.checkpoint_sha256);
    const auto valid_hex = [](std::string_view value)
    {
        return std::all_of(value.begin(), value.end(), [](unsigned char c)
        {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        });
    };
    if (!valid_hex(sidecar.checkpoint_fnv1a64) ||
        (!sidecar.checkpoint_sha256.empty() &&
         (sidecar.checkpoint_sha256.size() != 64 ||
          !valid_hex(sidecar.checkpoint_sha256))))
    {
        error = "checkpoint sidecar contains an invalid integrity digest";
        return false;
    }

    return true;
}

bool path_is_within(const std::filesystem::path& child, const std::filesystem::path& parent)
{
    const auto normalized_child  = std::filesystem::absolute(child).lexically_normal();
    const auto normalized_parent = std::filesystem::absolute(parent).lexically_normal();
    auto       child_it          = normalized_child.begin();
    for (auto parent_it = normalized_parent.begin(); parent_it != normalized_parent.end(); ++parent_it, ++child_it)
    {
        if (child_it == normalized_child.end() || *child_it != *parent_it)
            return false;
    }
    return true;
}

struct StormTrackResumePlan
{
    std::filesystem::path        tracking_directory;
    bool                         preserved_prefix     = false;
    std::size_t                  history_record_count = 0;
    std::optional<std::uint64_t> last_history_step;
};

bool read_file_bytes(const std::filesystem::path& path, std::string& bytes, std::string& error)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        error = "cannot read storm-track artifact '" + path.string() + "'";
        return false;
    }
    bytes.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    if (input.bad())
    {
        error = "failed reading storm-track artifact '" + path.string() + "'";
        return false;
    }
    return true;
}

bool inspect_storm_track_resume_prefix(const std::filesystem::path& outdir,
                                       std::uint64_t                checkpoint_step,
                                       StormTrackResumePlan&        plan,
                                       std::string&                 error)
{
    plan = {};
    plan.tracking_directory = outdir / "storm_track";
    const std::filesystem::path latest_path = outdir / "storm_track_latest.json";
    std::error_code ec;
    const bool directory_exists = std::filesystem::exists(plan.tracking_directory, ec);
    if (ec)
    {
        error = "cannot inspect storm-track directory: " + ec.message();
        return false;
    }
    const bool latest_exists = std::filesystem::exists(latest_path, ec);
    if (ec)
    {
        error = "cannot inspect storm-track latest artifact: " + ec.message();
        return false;
    }
    if (!directory_exists)
    {
        if (latest_exists)
        {
            error = "storm_track_latest.json exists without its history directory";
            return false;
        }
        return true;
    }
    if (!std::filesystem::is_directory(plan.tracking_directory, ec) || ec)
    {
        error = "storm-track history path is not a readable directory";
        return false;
    }

    static const std::regex history_pattern(R"(^track_step_([0-9]{9})\.json$)");
    std::string last_record_bytes;
    std::uint64_t last_step = 0;
    for (std::filesystem::directory_iterator iterator(plan.tracking_directory, ec), end;
         !ec && iterator != end; iterator.increment(ec))
    {
        const auto& entry = *iterator;
        const std::string filename = entry.path().filename().string();
        std::smatch match;
        if (!std::regex_match(filename, match, history_pattern))
        {
            if (filename.rfind("track_step_", 0) == 0)
            {
                error = "malformed storm-track history filename '" + filename + "'";
                return false;
            }
            continue;
        }
        if (!entry.is_regular_file(ec) || ec)
        {
            error = "storm-track history entry is not a regular file: " + entry.path().string();
            return false;
        }
        std::uint64_t filename_step = 0;
        try
        {
            filename_step = std::stoull(match[1].str());
        }
        catch (...)
        {
            error = "invalid storm-track history step in '" + filename + "'";
            return false;
        }
        std::string record_bytes;
        if (!read_file_bytes(entry.path(), record_bytes, error))
            return false;
        const std::regex completed_step_pattern(R"("completed_step"\s*:\s*([0-9]+))");
        std::smatch completed_step_match;
        if (!std::regex_search(record_bytes, completed_step_match, completed_step_pattern))
        {
            error = "storm-track history record is missing completed_step: " + entry.path().string();
            return false;
        }
        std::uint64_t embedded_step = 0;
        try
        {
            embedded_step = std::stoull(completed_step_match[1].str());
        }
        catch (...)
        {
            error = "invalid completed_step in storm-track history record: " + entry.path().string();
            return false;
        }
        if (embedded_step != filename_step)
        {
            error = "storm-track history filename and completed_step disagree: " + entry.path().string();
            return false;
        }
        if (embedded_step > checkpoint_step)
        {
            error = "storm-track history extends beyond the resume checkpoint";
            return false;
        }
        ++plan.history_record_count;
        if (!plan.last_history_step || embedded_step > last_step)
        {
            last_step = embedded_step;
            last_record_bytes = std::move(record_bytes);
            plan.last_history_step = embedded_step;
        }
    }
    if (ec)
    {
        error = "failed while inspecting storm-track history: " + ec.message();
        return false;
    }
    if (!plan.last_history_step)
    {
        if (latest_exists)
        {
            error = "storm_track_latest.json exists without a history record";
            return false;
        }
        return true;
    }
    if (!latest_exists)
    {
        error = "storm-track history is missing storm_track_latest.json";
        return false;
    }
    std::string latest_bytes;
    if (!read_file_bytes(latest_path, latest_bytes, error))
        return false;
    if (latest_bytes != last_record_bytes)
    {
        error = "storm_track_latest.json does not match the last preserved history record";
        return false;
    }
    plan.preserved_prefix = true;
    return true;
}

bool publish_storm_track_resume_marker(const StormTrackResumePlan& plan,
                                       const CheckpointSidecar&     sidecar,
                                       const std::filesystem::path& parent_checkpoint,
                                       std::string_view             branch_run_id,
                                       std::string&                 error)
{
    std::error_code ec;
    std::filesystem::create_directories(plan.tracking_directory, ec);
    if (ec)
    {
        error = "cannot create storm-track resume directory: " + ec.message();
        return false;
    }
    std::ostringstream filename;
    filename << "resume_step_" << std::setfill('0') << std::setw(9)
             << sidecar.completed_steps << "_" << branch_run_id << ".json";
    std::ostringstream json;
    json << std::setprecision(17) << "{\n"
         << "  \"schema\": \"nimbus.storm_track.resume\",\n"
         << "  \"schema_version\": 1,\n"
         << "  \"branch_run_id\": \"" << tmv::strutil::json_escape(std::string(branch_run_id)) << "\",\n"
         << "  \"parent_run_id\": \"" << tmv::strutil::json_escape(sidecar.run_id) << "\",\n"
         << "  \"parent_checkpoint\": \""
         << tmv::strutil::json_escape(parent_checkpoint.string()) << "\",\n"
         << "  \"parent_checkpoint_fnv1a64\": \"" << sidecar.checkpoint_fnv1a64 << "\",\n"
         << "  \"parent_checkpoint_sha256\": ";
    if (sidecar.checkpoint_sha256.empty())
        json << "null,\n";
    else
        json << "\"" << sidecar.checkpoint_sha256 << "\",\n";
    json
         << "  \"resumed_step\": " << sidecar.completed_steps << ",\n"
         << "  \"resumed_simulation_time_s\": " << sidecar.simulation_time_s << ",\n"
         << "  \"history_mode\": \""
         << (plan.preserved_prefix ? "preserved_prefix" : "new_branch") << "\",\n"
         << "  \"preserved_history_records\": " << plan.history_record_count << ",\n"
         << "  \"preserved_last_completed_step\": ";
    if (plan.last_history_step)
        json << *plan.last_history_step;
    else
        json << "null";
    json << "\n}\n";
    return publish_output_json_atomically(plan.tracking_directory / filename.str(), json.str(), error);
}

double next_cadence_after(double time_s, int cadence_s)
{
    if (cadence_s <= 0)
        return -1.0;
    // A completed step that lands within the runtime's deadline tolerance is
    // on that boundary. Without the same tolerance here, 34.999999999999993 s
    // after a nominal 35-s checkpoint was scheduled for 35 s again and the
    // following 35.7-s step emitted a duplicate checkpoint.
    constexpr double deadline_tolerance_s = 1.0e-9;
    return (std::floor((time_s + deadline_tolerance_s) / static_cast<double>(cadence_s)) + 1.0) * cadence_s;
}

double next_cadence_after(double time_s, double cadence_s)
{
    if (!std::isfinite(cadence_s) || cadence_s <= 0.0)
        return -1.0;
    constexpr double deadline_tolerance_s = 1.0e-9;
    return (std::floor((time_s + deadline_tolerance_s) / cadence_s) + 1.0) * cadence_s;
}

void log_memory_milestone(const char* phase)
{
    if (!log_normal_enabled())
        return;
    tmv::log_info("[MEMORY MILESTONE] phase=", phase, ", peak_rss_bytes=", process_peak_rss_bytes());
}

/**
 * @brief Aggregated runtime timing totals for headless simulation profiling.
 */
struct PerfTotals
{
    uint64_t steps               = 0;
    double   simulated_s         = 0.0;
    double   total_step_s        = 0.0;
    double   radiation_s         = 0.0;
    double   boundary_layer_s    = 0.0;
    double   chaos_noise_s       = 0.0;
    double   chaos_tendency_s    = 0.0;
    double   dynamics_s          = 0.0;
    double   output_submission_s = 0.0;
};

struct LiveDomainRecyclingResult
{
    bool                                      committed = false;
    std::string                               transaction_id;
    std::filesystem::path                     event_directory;
    tmv::domain::HorizontalDomainFrame        post_shift_frame;
    tmv::domain::HomogeneousWindowShiftReport shift;
    int                                       shift_i_cells = 0;
    int                                       shift_j_cells = 0;
    double                                    elapsed_ms    = 0.0;
};

#ifdef NIMBUS_ALLOCATION_TELEMETRY
void accumulate_allocation_delta(tmv::allocation_telemetry::Snapshot&       total,
                                 const tmv::allocation_telemetry::Snapshot& before,
                                 const tmv::allocation_telemetry::Snapshot& after) noexcept
{
    total.heap_allocations += after.heap_allocations - before.heap_allocations;
    total.heap_requested_bytes += after.heap_requested_bytes - before.heap_requested_bytes;
    total.physics_field_allocations += after.physics_field_allocations - before.physics_field_allocations;
    total.physics_field_allocated_bytes += after.physics_field_allocated_bytes - before.physics_field_allocated_bytes;
}
#endif

template <typename FieldPtr>
std::array<CoreVolumeFieldBinding<FieldPtr>, 27>
core_field_binding_array(const SimulationStateView& execution_state, FieldPtr u_field, FieldPtr v_field, FieldPtr w_field)
{
    return {{
        {"u", u_field},
        {"v", v_field},
        {"w", w_field},
        {"rho", &execution_state.fields.rho},
        {"p", &execution_state.fields.p},
        {"theta", &execution_state.fields.theta},
        {"qv", &execution_state.fields.qv},
        {"qc", &execution_state.fields.qc},
        {"qr", &execution_state.fields.qr},
        {"qh", &execution_state.fields.qh},
        {"qg", &execution_state.fields.qg},
        {"radar", &tmv::require_level_runtime(execution_state.diagnostics, "diagnostics").radar_reflectivity},
        {"tracer", &execution_state.fields.tracer},
        {"tke", &execution_state.fields.tke},
        {"qi", &execution_state.fields.qi},
        {"qs", &execution_state.fields.qs},
        {"vorticity_r", &tmv::require_level_runtime(execution_state.diagnostics, "diagnostics").vorticity_r},
        {"vorticity_theta", &tmv::require_level_runtime(execution_state.diagnostics, "diagnostics").vorticity_theta},
        {"vorticity_z", &tmv::require_level_runtime(execution_state.diagnostics, "diagnostics").vorticity_z},
        {"stretching_term", &tmv::require_level_runtime(execution_state.diagnostics, "diagnostics").stretching_term},
        {"tilting_term", &tmv::require_level_runtime(execution_state.diagnostics, "diagnostics").tilting_term},
        {"baroclinic_term", &tmv::require_level_runtime(execution_state.diagnostics, "diagnostics").baroclinic_term},
        {"p_prime", &tmv::require_level_runtime(execution_state.diagnostics, "diagnostics").p_prime},
        {"dynamic_pressure", &tmv::require_level_runtime(execution_state.diagnostics, "diagnostics").dynamic_pressure},
        {"buoyancy_pressure", &tmv::require_level_runtime(execution_state.diagnostics, "diagnostics").buoyancy_pressure},
        {"angular_momentum", &tmv::require_level_runtime(execution_state.diagnostics, "diagnostics").angular_momentum},
        {"angular_momentum_tendency", &tmv::require_level_runtime(execution_state.diagnostics, "diagnostics").angular_momentum_tendency},
    }};
}

template <typename FieldPtr>
std::vector<CoreVolumeFieldBinding<FieldPtr>> core_field_bindings(const SimulationStateView& execution_state, FieldPtr u_field, FieldPtr v_field, FieldPtr w_field)
{
    const auto fields = core_field_binding_array(execution_state, u_field, v_field, w_field);
    return {fields.begin(), fields.end()};
}

class CheckpointLifecycle
{
public:
    const std::string& run_id() const noexcept { return run_id_; }
    const std::string& parent_run_id() const noexcept { return parent_run_id_; }
    const std::filesystem::path& parent_path() const noexcept { return parent_path_; }
    const std::filesystem::path& last_path() const noexcept { return last_path_; }
    bool committed_step(std::uint64_t step) const noexcept
    { return last_step_ == step && !last_path_.empty(); }
    void schedule(double time, int cadence) { next_time_s_ = next_cadence_after(time, cadence); }
    void resume(std::string parent_id, std::filesystem::path path, double time, int cadence)
    {
        parent_run_id_ = std::move(parent_id);
        parent_path_ = std::move(path);
        schedule(time, cadence);
    }
    int publish(const HeadlessRunOptions& options, SimulationStorageState& storage,
                const tmv::TrajectorySet& trajectories, int steps, bool final_checkpoint);

private:
    const std::string run_id_ = make_output_publication_generation();
    std::string parent_run_id_;
    std::filesystem::path parent_path_;
    double next_time_s_ = -1.0;
    std::uint64_t last_step_ = std::numeric_limits<std::uint64_t>::max();
    std::filesystem::path last_path_;
};

int CheckpointLifecycle::publish(const HeadlessRunOptions& options, SimulationStorageState& storage,
                                 const tmv::TrajectorySet& trajectories, int steps, bool final_checkpoint)
{
    if (options.checkpoint_every_s <= 0)
        return 0;
    const std::uint64_t completed_steps = static_cast<std::uint64_t>(std::max(0, steps));
    const bool cadence_due = next_time_s_ >= 0.0 && storage.time_s + 1.0e-9 >= next_time_s_;
    if ((!cadence_due && !final_checkpoint) || completed_steps == last_step_)
        return 0;

    const std::filesystem::path directory =
        options.checkpoint_dir.empty() ? std::filesystem::path(options.outdir) / "checkpoints" : options.checkpoint_dir;
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec)
    {
        tmv::log_error("[CHECKPOINT] cannot create checkpoint directory '", directory.string(), "': ", ec.message());
        return 2;
    }

    const std::uintmax_t checkpoint_reserve =
        static_cast<std::uintmax_t>(options.checkpoint_minimum_free_space_mib) * 1024U * 1024U;
    std::string error;
    if (!nimbus::output::require_free_space(directory, checkpoint_reserve, error))
    {
        tmv::log_error("[CHECKPOINT] ", error);
        return 2;
    }

    std::ostringstream filename;
    filename << "checkpoint_step_" << std::setfill('0') << std::setw(9) << completed_steps << ".nimbus";
    const std::filesystem::path checkpoint_path = directory / filename.str();
    log_memory_milestone("checkpoint_assembly_begin");
    const tmv::TrajectorySet* checkpoint_trajectories =
        options.trajectory.enabled ? &trajectories : nullptr;
    tmv::checkpoint::SimulationCheckpointIntegrity checkpoint_integrity;
    const auto checkpoint_started = global_perf_timing_enabled ? PerfClock::now() : PerfClock::time_point{};
    if (!tmv::checkpoint::write_simulation_checkpoint(
            checkpoint_path, storage,
            checkpoint_trajectories, checkpoint_integrity, error,
            static_cast<std::uint64_t>(checkpoint_reserve)))
    {
        tmv::log_error("[CHECKPOINT] write failed: ", error);
        return 2;
    }
    if (global_perf_timing_enabled)
    {
        const double seconds = std::chrono::duration<double>(PerfClock::now() - checkpoint_started).count();
        tmv::log_info("[CHECKPOINT PERF] completed_step=", completed_steps,
                      ", bytes=", checkpoint_integrity.bytes, ", artifact_publication_s=", seconds,
                      ", mib_per_s=", static_cast<double>(checkpoint_integrity.bytes) / (1024.0 * 1024.0 * std::max(seconds, 1e-12)));
    }
    log_memory_milestone("checkpoint_write_complete");

    const std::uint64_t checkpoint_bytes = checkpoint_integrity.bytes;
    const std::string&  fnv1a64          = checkpoint_integrity.fnv1a64;
    const std::string&  sha256            = checkpoint_integrity.sha256;
    std::ostringstream sidecar;
    const auto&        recycling = storage.runtime.domain_recycling_provenance;
    sidecar << std::setprecision(17) << "{\n"
            << "  \"schema\": \"nimbus.checkpoint.lifecycle\",\n"
            << "  \"schema_version\": 1,\n"
            << "  \"checkpoint_file\": \"" << tmv::strutil::json_escape(checkpoint_path.filename().string()) << "\",\n"
            << "  \"checkpoint_bytes\": " << checkpoint_bytes << ",\n"
            << "  \"checkpoint_fnv1a64\": \"" << fnv1a64 << "\",\n"
            << "  \"checkpoint_sha256\": \"" << sha256 << "\",\n"
            << "  \"run_id\": \"" << tmv::strutil::json_escape(run_id_) << "\",\n"
            << "  \"parent_run_id\": \"" << tmv::strutil::json_escape(parent_run_id_) << "\",\n"
            << "  \"parent_checkpoint\": \"" << tmv::strutil::json_escape(parent_path_.string()) << "\",\n"
            << "  \"completed_steps\": " << completed_steps << ",\n"
            << "  \"simulation_time_s\": " << storage.time_s << ",\n"
            << "  \"boundary\": \"completed_timestep\",\n"
            << "  \"portable_across_compute_memory_topology\": true,\n"
            << "  \"trajectory_state\": ";
    if (!options.trajectory.enabled)
    {
        sidecar << "null,\n";
    }
    else
    {
        sidecar << "{\"seeded\": " << (trajectories.seeded() ? "true" : "false")
                << ", \"parcels\": " << trajectories.count()
                << ", \"samples\": " << trajectories.total_samples()
                << ", \"sample_interval_s\": "
                << trajectories.configuration().sample_interval_s
                << ", \"max_samples_per_parcel\": "
                << trajectories.configuration().max_samples_per_parcel
                << ", \"encoded_sample_bytes\": "
                << trajectories.total_samples() * sizeof(float) *
                       static_cast<std::size_t>(tmv::kTrajectoryVarCount)
                << "},\n";
    }
    sidecar
            << "  \"domain_recycling\": ";
    if (!recycling.available)
    {
        sidecar << "null\n";
    }
    else
    {
        sidecar << "{\"algorithm\": \"" << tmv::domain::domain_recycling_transaction_provenance_algorithm_id()
                << "\", \"shift_algorithm\": \""
                << tmv::strutil::json_escape(
                       std::string(recycling.shift_algorithm.view()))
                << "\", \"owner_algorithms\": {\"boundary_layer\": \""
                << tmv::domain::boundary_layer_domain_recycling_algorithm_id()
                << "\", \"radiation\": \""
                << tmv::domain::radiation_domain_recycling_algorithm_id()
                << "\", \"microphysics\": \""
                << tmv::domain::microphysics_domain_recycling_algorithm_id()
                << "\", \"chaos\": \""
                << tmv::domain::chaos_domain_recycling_algorithm_id()
                << "\"";
        if (storage.nested_execution.relationship_ready)
            sidecar << ", \"nested_grid\": \""
                    << tmv::domain::nested_grid_domain_recycling_algorithm_id()
                    << "\"";
        sidecar << "}"
                << ", \"transaction_id\": \""
                << tmv::strutil::json_escape(std::string(recycling.transaction_id.view()))
                << "\", \"publication_generation\": \""
                << tmv::strutil::json_escape(std::string(recycling.publication_generation.view()))
                << "\", \"sequence\": " << recycling.sequence
                << ", \"transaction_simulation_time_s\": " << recycling.simulation_time_s
                << ", \"event_manifest_bytes\": " << recycling.event_manifest_bytes
                << ", \"event_manifest_fnv1a64\": \"" << hex_u64(recycling.event_manifest_fnv1a64)
                << "\", \"previous_chain_fnv1a64\": \"" << hex_u64(recycling.previous_chain_fnv1a64)
                << "\", \"committed_chain_fnv1a64\": \"" << hex_u64(recycling.committed_chain_fnv1a64)
                << "\", \"control_volume_exchange_complete\": "
                << (recycling.control_volume_exchange_complete ? "true" : "false");
        if (recycling.control_volume_exchange_complete)
            sidecar << ", \"control_volume_exchange_basis\": \""
                    << tmv::strutil::json_escape(std::string(recycling.control_volume_exchange_basis.view())) << "\"";
        sidecar << ", \"private_inventory_complete\": " << (recycling.private_inventory_complete ? "true" : "false");
        if (recycling.private_inventory_complete)
        {
            sidecar << ", \"private_inventory\": [";
            for (std::size_t f = 0; f < recycling.retiring_private_inventory.count; ++f)
            {
                if (f)
                    sidecar << ',';
                sidecar << "{\"name\":\""
                        << tmv::strutil::json_escape(
                               std::string(recycling.retiring_private_inventory.entries[f].name.view()))
                        << "\",\"retiring\":" << recycling.retiring_private_inventory.entries[f].inventory
                        << ",\"admitted\":" << recycling.admitted_private_inventory.entries[f].inventory << '}';
            }
            sidecar << ']';
        }
        sidecar << "}\n";
    }
    sidecar << "}\n";
    if (!publish_output_json_atomically(checkpoint_sidecar_path(checkpoint_path), sidecar.str(), error))
    {
        tmv::log_error("[CHECKPOINT] sidecar publication failed: ", error);
        return 2;
    }
    last_step_   = completed_steps;
    last_path_   = checkpoint_path;
    next_time_s_ = next_cadence_after(storage.time_s, options.checkpoint_every_s);
    tmv::log_info("[CHECKPOINT] committed ",
                  checkpoint_path.string(),
                  " at step=",
                  completed_steps,
                  ", time_s=",
                  storage.time_s);
    nimbus::output::RetentionResult retention;
    if (!nimbus::output::enforce_checkpoint_retention(
            directory,
            static_cast<std::size_t>(options.checkpoint_keep_last),
            options.resume_checkpoint,
            retention,
            error))
    {
        tmv::log_error("[CHECKPOINT] retention failed after committing the newest checkpoint: ", error);
        return 2;
    }
    if (retention.removed_items > 0)
    {
        tmv::log_info("[CHECKPOINT] retention removed ",
                      retention.removed_items,
                      " old checkpoint(s), reclaimed_bytes=",
                      retention.reclaimed_bytes);
    }
    return 0;
}

class FieldValidation
{
public:
    bool validate_core_fields(const SimulationStateView&   execution_state,
                              const std::string&           context,
                              int                          step_index,
                              bool                         include_percentiles,
                              const std::filesystem::path* report_path,
                              bool                         persist_report);
#ifdef EXPORT_NPY
    bool validate_derived_export_slice(const SimulationStateView& execution_state,
                                       const char*                field_id,
                                       std::vector<float>&        values,
                                       int                        theta_index,
                                       int                        frame_export_index);
#endif
    std::vector<tmv::ValidationReport> export_validation_reports;
    std::string                        validation_error_message;
};

class OutputPublisher
{
public:
    OutputPublisher(const HeadlessRunOptions& input,
                    SimulationStorageState&   source,
                    int&                      completed_steps,
                    bool&                     step_timing,
                    PerfTotals&               timing,
                    FieldValidation&          validator)
        : options(input), run(source), steps(completed_steps), perf_step_active(step_timing), perf_totals(timing),
          validation(validator), export_ms(input.export_ms), write_every_s(input.write_every_s), outdir(input.outdir),
          output_config(input.output_config),
          export_diagnostics_enabled(log_debug_enabled() || std::getenv("NIMBUS_EXPORT_DIAGNOSTICS") != nullptr ||
                                     std::getenv("NIMBUS_DEBUG_EXPORTS") != nullptr)
    {

        qualification_output_config.format = OutputFormat::npy_3d;
        qualification_output_config.preset = FieldPreset::custom;
        qualification_output_config.custom_fields =
            qualification_diagnostics_fields(output_config.qualification_diagnostics_preset);
        qualification_output_config.async_io               = false;
        qualification_output_config.estimate_disk_budget   = false;
        qualification_output_config.keep_last_frames       = output_config.keep_last_frames;
        qualification_output_config.minimum_free_space_mib = output_config.minimum_free_space_mib;
    }
    template <class Fn>
    void timed_call(double& accumulator, Fn&& function)
    {
        if (!run.configuration.perf_timing_enabled || !perf_step_active)
        {
            function();
            return;
        }
        const auto start = PerfClock::now();
        function();
        accumulator += std::chrono::duration<double>(PerfClock::now() - start).count();
    }
    const HeadlessRunOptions& options;
    SimulationStorageState&   run;
    int&                      steps;
    bool&                     perf_step_active;
    PerfTotals&               perf_totals;
    FieldValidation&          validation;
    const int                 export_ms;
    const int                 write_every_s;
    const std::string&        outdir;
#ifdef EXPORT_NPY
    struct WriteFrameContext
    {
        WriteFrameContext(const SimulationStateView& level, int index, AsyncOutputWriter& output_writer)
            : state(level), frame_export_index(index), writer(output_writer),
              assembly_started(std::chrono::steady_clock::now()),
              slice_size(static_cast<std::size_t>(level.grid.nr) * static_cast<std::size_t>(level.grid.nz))
        {
        }

        SimulationStateView                                 state;
        int                                                 frame_export_index;
        AsyncOutputWriter&                                  writer;
        std::chrono::steady_clock::time_point               assembly_started;
        std::filesystem::path                               step_path;
        std::string                                         step_str;
        std::filesystem::path                               report_path;
        std::size_t                                         slice_size;
        float                                               reflectivity_dbz_min = -30.0f;
        float                                               reflectivity_dbz_max = 120.0f;
        bool                                                use_3d               = false;
        bool                                                use_async            = false;
        ExportSnapshot                                      snapshot;
        Field3D                                             u_center_buf;
        Field3D                                             v_center_buf;
        Field3D                                             w_center_buf;
        const Field3D*                                      u_for_output = nullptr;
        const Field3D*                                      v_for_output = nullptr;
        const Field3D*                                      w_for_output = nullptr;
        std::unordered_map<std::string, std::vector<float>> derived_3d_bufs;
        std::vector<float>                                  tranche_slice_buffer;
        bool                                                need_derived_volume_slices = false;
    };
#endif
    bool qualification_diagnostics_enabled() const
    {
        return output_config.qualification_diagnostics_preset != QualificationDiagnosticsPreset::none;
    }
    const std::vector<double>* refresh_cgrid_ground_vertical_velocity();
    const std::vector<double>* refresh_cgrid_ground_vertical_velocity(const SimulationStateView& execution_state);
    void refresh_output_diagnostics(const SimulationStateView& state, OutputFieldDependency dependencies);
    tmv_shm::ShmHorizontalFrame live_shm_frame() const;
    bool                        shm_update(bool force = false);
#ifdef EXPORT_NPY
    bool prepare_write_frame(WriteFrameContext& frame);
#endif
#ifdef EXPORT_NPY
    void prepare_core_output_fields(WriteFrameContext& frame);
#endif
#ifdef EXPORT_NPY
    void write_3d_core_fields(WriteFrameContext& frame);
#endif
#ifdef EXPORT_NPY
    bool write_3d_surface_fields(WriteFrameContext& frame);
#endif
#ifdef EXPORT_NPY
    bool write_3d_column_fields(WriteFrameContext&                              frame,
                                const std::vector<RegisteredDiagnosticProduct>& tranche_registry);
#endif
#ifdef EXPORT_NPY
    std::string build_step_manifest_json(const WriteFrameContext& frame) const;
#endif
#ifdef EXPORT_NPY
    bool write_step_manifest(const WriteFrameContext& frame, const std::filesystem::path& manifest_path) const;
#endif
#ifdef EXPORT_NPY
    void compute_theta_azimuth_mean(std::vector<float>& theta_azimuth_mean) const;
#endif
#ifdef EXPORT_NPY
    void compute_theta_azimuth_mean(const SimulationStateView& execution_state,
                                    std::vector<float>&        theta_azimuth_mean) const;
#endif
#ifdef EXPORT_NPY
    void compute_storm_level_means(std::vector<float>& storm_u_level_mean,
                                   std::vector<float>& storm_v_level_mean) const;
#endif
#ifdef EXPORT_NPY
    void compute_storm_level_means(const SimulationStateView& execution_state,
                                   std::vector<float>&        storm_u_level_mean,
                                   std::vector<float>&        storm_v_level_mean) const;
#endif
#ifdef EXPORT_NPY
    void prepare_derived_output_buffers(WriteFrameContext& frame);
#endif
#ifdef EXPORT_NPY
    void write_core_slice_fields(WriteFrameContext& frame, int theta_index);
#endif
#ifdef EXPORT_NPY
    bool write_derived_slice_outputs(WriteFrameContext&         frame,
                                     const DerivedSliceBinding* bindings,
                                     std::size_t                binding_count,
                                     int                        theta_index);
#endif
#ifdef EXPORT_NPY
    bool write_tranche_slice_outputs(WriteFrameContext&                              frame,
                                     const std::vector<RegisteredDiagnosticProduct>& tranche_registry,
                                     int                                             theta_index,
                                     std::vector<float>&                             tranche_slice_buffer);
#endif
#ifdef EXPORT_NPY
    bool finish_write_frame(WriteFrameContext& frame);
    bool fail_qualified_zfp_output_contract();
#endif
#ifdef EXPORT_NPY
    bool write_all_fields(int frame_export_index);
#endif
#ifdef EXPORT_NPY
    bool write_all_fields(const SimulationStateView& execution_state, int frame_export_index);
#endif
    bool enforce_output_retention();
    bool fail_terminal_run_contract() noexcept;
#ifdef EXPORT_NPY
    bool write_qualification_diagnostics_frame(int frame_export_index);
#endif
    int          maybe_write_output();
    OutputConfig output_config;
    OutputConfig qualification_output_config;
    // Mirrors the writer's decoded-reference state so each manifest can name
    // the exact delta parents before the async snapshot is handed off.
    mutable std::unordered_map<std::string, std::vector<int>> zfp_manifest_history;
    // A nested child is a separate time series. Its compressed frames must
    // never name a root-grid frame as a delta parent.
    mutable std::unordered_map<std::string, std::vector<int>> nested_child_zfp_manifest_history;
    bool                                                      publishing_nested_child              = false;
    bool                                                      publishing_qualification_diagnostics = false;
    static constexpr int                                      thetaIndex                           = 0;
    // NIMBUS_DEBUG_EXPORTS is retained as a compatibility alias for existing
    // diagnostic scripts. NIMBUS_EXPORT_DIAGNOSTICS is the descriptive name.
    const bool                            export_diagnostics_enabled;
    std::optional<AsyncOutputWriter>      async_writer;
    double                                snapshot_assembly_total_s   = 0.0;
    double                                snapshot_assembly_max_s     = 0.0;
    std::size_t                           snapshot_assembly_max_bytes = 0;
    std::size_t                           snapshot_assembly_max_owned_bytes = 0;
    std::size_t                           snapshot_assembly_frames    = 0;
    ShmWriter                             shm_writer;
    double                                next_shm_publish_time_s = -1.0;
    std::uint64_t                         live_shm_frames         = 0;
    std::uint64_t                         live_shm_payload_bytes  = 0;
    std::uint64_t                         live_shm_skipped_steps  = 0;
    std::uint64_t                         live_shm_failed_frames  = 0;
    double                                live_shm_publish_time_s = 0.0;
    std::unordered_set<std::string>       active_export_fields;
    std::string                           output_publication_generation;
    bool                                  run_contract_ready = false;
    std::chrono::steady_clock::time_point lastGuiExport;
    int                                   export_index                                 = 0;
    int                                   qualification_export_index                   = 0;
    double                                next_field_export_time_s                     = -1.0;
    double                                next_medium_export_time_s                    = -1.0;
    double                                next_slow_export_time_s                      = -1.0;
    double                                next_qualification_diagnostics_export_time_s = -1.0;
    std::vector<double>                   cgrid_ground_vertical_velocity_ms;
    Field3D                               shm_u_center_buf;
    Field3D                               shm_v_center_buf;
    Field3D                               shm_w_center_buf;
    Field3D                               shm_reflectivity_dbz_buf;
};

class StormTrackingController
{
public:
    StormTrackingController(const HeadlessRunOptions& input,
                            SimulationStorageState&   source,
                            int&                      completed_steps,
                            OutputPublisher&          publisher,
                            FieldValidation&          validator,
                            CheckpointLifecycle&      lifecycle,
                            tmv::TrajectorySet&       parcels)
        : options(input), run(source), steps(completed_steps), output(publisher), validation(validator),
          checkpoints(lifecycle), trajectories(parcels), outdir(input.outdir), write_every_s(input.write_every_s)
    {
    }
    const HeadlessRunOptions& options;
    SimulationStorageState&   run;
    int&                      steps;
    OutputPublisher&          output;
    FieldValidation&          validation;
    CheckpointLifecycle&      checkpoints;
    tmv::TrajectorySet&       trajectories;
    const std::string&        outdir;
    const int                 write_every_s;
    LiveDomainRecyclingResult perform_domain_recycling(const tmv::domain::StormWindowMovementDecision& decision);
    int                       maybe_track_storm();
    double                    next_storm_tracking_time_s = -1.0;
    std::uint64_t             domain_recycling_attempt   = 0;
    tmv::diagnostics::StormFeatureDetector          storm_feature_detector;
    tmv::diagnostics::StormBranchAssociationContext storm_branch_association;
    bool                                            storm_tracking_uses_nested_child = false;
    std::vector<double>                             storm_maximum_updraft_ms;
    std::vector<double>                             storm_condensate_path_kgm2;
    std::vector<double>                             storm_updraft_helicity_m2s2;
};

struct HeadlessDriver
{
    int                     steps            = 0;
    bool                    perf_step_active = false;
    PerfTotals              perf_totals;
    FieldValidation         validation;
    CheckpointLifecycle     checkpoints;
    tmv::TrajectorySet      trajectories;
    OutputPublisher         output;
    StormTrackingController tracking;

    explicit HeadlessDriver(const HeadlessRunOptions& run_options)
        : output(run_options, global_simulation_storage, steps, perf_step_active, perf_totals, validation),
          tracking(run_options, global_simulation_storage, steps, output, validation, checkpoints, trajectories),
          options(run_options), export_ms(run_options.export_ms), duration_s(run_options.duration_s),
          write_every_s(run_options.write_every_s),
          scientific_diagnostics_every_steps(run_options.scientific_diagnostics_every_steps), outdir(run_options.outdir)
    {
    }

    int setup();
    int build_initial_conditions();
    int restore_resume_checkpoint();
    int maybe_write_checkpoint(bool final_checkpoint = false);

    bool capture_termination_request();
    bool publish_termination_record();
    int  run_time_loop();
    int  advance_one_step();
    int  validate_configured_nesting() const;
    int  advance_configured_nested_child();

    void record_step_timing(const PerfClock::time_point& step_t0);
    bool should_stop() const;

    int finalize();
    int run();
    template <class Fn>
    void timed_call(double& accumulator, Fn&& fn)
    {
        if (!global_perf_timing_enabled || !perf_step_active)
        {
            fn();
            return;
        }
        const auto t0 = PerfClock::now();
        fn();
        const auto t1 = PerfClock::now();
        accumulator += std::chrono::duration<double>(t1 - t0).count();
    }

    const HeadlessRunOptions& options;
    const int                 export_ms;
    const int                 duration_s;
    const int                 write_every_s;
    const int                 scientific_diagnostics_every_steps;
    const std::string&        outdir;

    int termination_signal_number = 0;

    PrognosticFieldStorage             nested_previous_parent_fields;
    NestedParentMicrophysicsSnapshotV1 nested_previous_parent_microphysics;
    std::array<std::uint64_t, 3>       nested_parent_smagorinsky_dispatches{};
    std::array<std::uint64_t, 3>       nested_child_smagorinsky_dispatches{};
    std::array<std::uint64_t, 3>       nested_parent_kessler_dispatches{};
    std::array<std::uint64_t, 3>       nested_child_kessler_dispatches{};
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    tmv::allocation_telemetry::Snapshot allocation_snapshot;
    tmv::allocation_telemetry::Snapshot radiation_allocation_snapshot;
    tmv::allocation_telemetry::Snapshot boundary_layer_allocation_snapshot;
    tmv::allocation_telemetry::Snapshot dynamics_allocation_snapshot;
    bool                                allocation_measurement_started = false;
#endif

    std::optional<tmv::TrajectoryCheckpointState> pending_trajectory_restore;
};

int HeadlessDriver::validate_configured_nesting() const
{
    const auto& config = global_simulation_storage.configuration.nesting;
    const bool active = global_simulation_storage.nested_execution.relationship_ready;
    if (!config.enabled)
    {
        if (active)
        {
            tmv::log_error(
                "[NESTING] checkpoint contains an active child but domain_motion.nesting.enabled is false");
            return 2;
        }
        return 0;
    }

    const SimulationStateView state = global_simulation_state();
    const auto physical = physical_scalar_domain(state.grid);
    if (physical.empty())
    {
        tmv::log_error("[NESTING] parent physical scalar domain is empty");
        return 2;
    }
    if (global_domain_motion_mode ==
            tmv::init::config::DomainMotionMode::MovingNest &&
        global_storm_tracking_config.enabled)
    {
        std::size_t minimum_recenter_shift_cells = 0;
        try
        {
            minimum_recenter_shift_cells =
                tmv::domain::minimum_storm_window_emergency_shift_cells(
                    global_storm_tracking_config.inner_margin_m,
                    state.grid.dr, state.grid.dtheta);
        }
        catch (const std::exception& e)
        {
            tmv::log_error(
                "[NESTING] invalid moving-nest release geometry: ", e.what());
            return 2;
        }
        if (static_cast<std::size_t>(
                global_storm_tracking_config.maximum_shift_cells) <
            minimum_recenter_shift_cells)
        {
            tmv::log_error(
                "[NESTING] moving-nest maximum_shift_cells cannot restore the "
                "configured release margin from a contained emergency boundary approach "
                "on the parent grid: require at least ",
                minimum_recenter_shift_cells,
                ", received ",
                global_storm_tracking_config.maximum_shift_cells);
            return 2;
        }
    }
    try
    {
        const ComputeBackend* backend =
            state.compute_runtime != nullptr
                ? state.compute_runtime->backend.get()
                : nullptr;
        const bool native_dry_cgrid_ready =
            backend != nullptr &&
            backend->supports_dynamics(
                DynamicsKernelOp::CartesianCgridSlowTendencies) &&
            backend->supports_dynamics(
                DynamicsKernelOp::AcousticSubstepFusedCartesianCgrid) &&
            backend->supports_dynamics(
                DynamicsKernelOp::AcousticSubstepsBatchedCartesianCgrid) &&
            backend->supports_batched_cartesian_cgrid_mass_flux_export() &&
            backend->supports_advection(
                AdvectionKernelOp::ConservativeCartesianCgridTransport) &&
            backend->supports_supplied_cartesian_cgrid_mass_flux();
        const bool native_smagorinsky_ready =
            backend != nullptr &&
            backend->supports_turbulence(TurbulenceKernelOp::Smagorinsky) &&
            backend->supports_turbulence(TurbulenceKernelOp::SgsApply);
        const bool native_kessler_ready =
            backend != nullptr &&
            backend->supports_microphysics(
                MicrophysicsKernelOp::KesslerPointwise) &&
            backend->supports_microphysics(
                MicrophysicsKernelOp::KesslerSedimentation);
        const StationaryNestingLiveExecution live{
            state.compute_runtime != nullptr &&
                backend != nullptr,
            state.compute_runtime != nullptr
                ? compute_backend_kind_name(state.compute_runtime->active_kind)
                : std::string{},
            state.compute_runtime != nullptr &&
                state.compute_runtime->fallback_active,
            state.terrain_runtime != nullptr &&
                (state.terrain_runtime->config.scheme_id == "none"
                 || ((state.terrain_runtime->config.scheme_id == "bell" ||
                      state.terrain_runtime->config.scheme_id == "schar")
                     && state.terrain_runtime->metrics.J.size_r()
                         == state.grid.nr
                     && state.terrain_runtime->metrics.J.size_th()
                         == state.grid.nth
                     && state.terrain_runtime->metrics.J.size_z()
                         == state.grid.nz)),
            native_dry_cgrid_ready,
            native_smagorinsky_ready,
            native_kessler_ready,
        };
        validate_stationary_nesting_execution(&live);
        tmv::domain::OneWayNestRelationshipInput input;
        input.parent_cells_x = physical.nx();
        input.parent_cells_y = physical.ny();
        input.parent_levels_z = state.grid.nz;
        input.parent_dx_m = state.grid.dr;
        input.parent_dy_m = state.grid.dtheta;
        const auto& time = global_simulation_storage.runtime.time_stepping.config;
        input.parent_dt_s = std::max(
            global_simulation_storage.configuration.configured_dt_large_s,
            time.adaptive_dt ? time.dt_max : state.grid.dt);
        input.parent_ratio_x = config.parent_ratio_x;
        input.parent_ratio_y = config.parent_ratio_y;
        input.parent_time_step_ratio = config.parent_time_step_ratio;
        input.child_origin_parent_cell_i = config.child_origin_parent_cell_i;
        input.child_origin_parent_cell_j = config.child_origin_parent_cell_j;
        input.child_cells_x = config.child_cells_x;
        input.child_cells_y = config.child_cells_y;
        input.child_levels_z = state.grid.nz;
        input.relaxation_width_child_cells =
            config.relaxation_width_child_cells;
        input.boundary_stencil_parent_cells =
            config.boundary_stencil_parent_cells;
        const auto expected =
            tmv::domain::prepare_one_way_nest_relationship(input);
        if (active)
        {
            const auto& actual =
                global_simulation_storage.nested_execution.relationship;
            const auto& nested =
                global_simulation_storage.nested_execution;
            if (actual.parent_cells_x != expected.parent_cells_x ||
                actual.parent_cells_y != expected.parent_cells_y ||
                actual.levels_z != expected.levels_z ||
                actual.parent_ratio_x != expected.parent_ratio_x ||
                actual.parent_ratio_y != expected.parent_ratio_y ||
                actual.parent_time_step_ratio != expected.parent_time_step_ratio ||
                nested.initial_parent_origin_i != expected.child_origin_parent_cell_i ||
                nested.initial_parent_origin_j != expected.child_origin_parent_cell_j ||
                actual.child_cells_x != expected.child_cells_x ||
                actual.child_cells_y != expected.child_cells_y ||
                actual.relaxation_width_child_cells != expected.relaxation_width_child_cells ||
                actual.boundary_stencil_parent_cells != expected.boundary_stencil_parent_cells ||
                actual.parent_dx_m != expected.parent_dx_m ||
                actual.parent_dy_m != expected.parent_dy_m ||
                actual.parent_dt_s != expected.parent_dt_s)
            {
                tmv::log_error(
                    "[NESTING] restored child relationship does not match the requested YAML contract");
                return 2;
            }
            require_nested_child_level_runtime_ready(
                global_simulation_storage);
        }
    }
    catch (const std::exception& error)
    {
        tmv::log_error("[NESTING] admission failed: ", error.what());
        return 2;
    }
    return 0;
}

int HeadlessDriver::advance_configured_nested_child()
{
    auto& storage = global_simulation_storage;
    const auto& config = storage.configuration.nesting;
    if (!config.enabled)
        return 0;
    const bool two_way =
        config.feedback == tmv::init::config::NestingFeedback::TwoWay;
    std::optional<NestedExecutionStorage> two_way_rollback;

    try
    {
        if (two_way)
            two_way_rollback.emplace(storage.nested_execution);
        auto& nested = storage.nested_execution;
        if (!nested.relationship_ready)
        {
            const auto physical = physical_scalar_domain(storage.grid.nr,
                storage.grid.nth, storage.grid.nz, StaggerType::CGrid);
            const auto& time = storage.runtime.time_stepping.config;
            tmv::domain::OneWayNestRelationshipInput input;
            input.parent_cells_x = physical.nx();
            input.parent_cells_y = physical.ny();
            input.parent_levels_z = storage.grid.nz;
            input.parent_dx_m = storage.grid.dx;
            input.parent_dy_m = storage.grid.dy;
            input.parent_dt_s = std::max(
                storage.configuration.configured_dt_large_s,
                time.adaptive_dt ? time.dt_max : dt);
            input.parent_ratio_x = config.parent_ratio_x;
            input.parent_ratio_y = config.parent_ratio_y;
            input.parent_time_step_ratio = config.parent_time_step_ratio;
            input.child_origin_parent_cell_i = config.child_origin_parent_cell_i;
            input.child_origin_parent_cell_j = config.child_origin_parent_cell_j;
            input.child_cells_x = config.child_cells_x;
            input.child_cells_y = config.child_cells_y;
            input.child_levels_z = storage.grid.nz;
            input.relaxation_width_child_cells =
                config.relaxation_width_child_cells;
            input.boundary_stencil_parent_cells =
                config.boundary_stencil_parent_cells;
            prepare_nested_execution_relationship(
                storage,
                tmv::domain::prepare_one_way_nest_relationship(input),
                simulation_time - dt);
            initialize_nested_shared_level_reference_state(storage);
            NestedChildInitializationReport initialization;
            initialize_nested_child_physical_primary_state(
                storage, nested_previous_parent_fields, initialization);
            NestedChildRelaxationReport boundary;
            apply_nested_child_physical_relaxation(
                storage, nested_previous_parent_fields,
                {config.relaxation_e_folding_time_s},
                storage.nested_execution.child.grid.dt, boundary);
            if (two_way)
            {
                nested.two_way_interval_source_begin =
                    nested.interface_source.cumulative;
                nested.two_way_child_interval_begin_inventory =
                    nested_two_way_child_inventory(nested);
            }
            initialize_nested_child_level_runtime(storage);
            if (two_way)
            {
                configure_two_way_level_flux_capture(
                    nested.child.runtime, nested.child.grid,
                    {1, nested.relationship.child_cells_x + 1,
                     1, nested.relationship.child_cells_y + 1,
                     0, nested.relationship.levels_z},
                    nested.relationship.parent_ratio_x, storage.configuration.microphysics_scheme != "none");
            }
            if (!nested_previous_parent_microphysics.available)
                throw std::runtime_error(
                    "nested child initialization is missing the previous parent microphysics endpoint");
            initialize_nested_child_microphysics_private_state(
                storage, nested_previous_parent_fields,
                nested_previous_parent_microphysics, initialization);
            tmv::log_info(
                "[NESTING] initialized stationary ",
                two_way ? "two-way" : "one-way",
                " child: physical_cells=",
                config.child_cells_x, "x", config.child_cells_y, "x",
                storage.grid.nz, ", refinement=", config.parent_ratio_x,
                "x", config.parent_ratio_y, ", time_ratio=",
                config.parent_time_step_ratio);
        }
        else
        {
            roll_nested_parent_interval(storage);
        }

        const bool moist_feedback = two_way && nested.child.runtime.advection.moist_interface.enabled;
        const auto water_begin = moist_feedback ? nested_thermodynamic_inventory(nested_child_execution_state(storage))
                                                : std::array<double, 8>{};
        std::array<tmv::diagnostics::CompensatedSum, 7> water_source, water_storage;
        const auto                                      private_begin =
            two_way ? nested_private_inventory(nested_child_execution_state(storage)) : std::vector<double>{};
        std::vector<tmv::diagnostics::CompensatedSum> private_source(private_begin.size()),
            private_storage(private_begin.size());
        const auto                                      water_sources = [](const NestedChildRelaxationReport& report)
        {
            return std::array<double, 7>{report.qv.committed_carrier_change,
                                         report.qc.committed_carrier_change,
                                         report.qr.committed_carrier_change,
                                         report.qi.committed_carrier_change,
                                         report.qs.committed_carrier_change,
                                         report.qg.committed_carrier_change,
                                         report.qh.committed_carrier_change};
        };
        while (storage.nested_execution.parent_time_bracket.completed_child_substeps <
               storage.nested_execution.relationship.parent_time_step_ratio)
        {
            std::array<double, 5> debug_inventory_before{};
            std::array<double, 5> debug_interface_before{};
            std::array<double, 3> debug_transport_before{};
            std::array<double, 3> debug_source_before{};
            std::array<double, 3> debug_storage_before{};
            std::array<std::array<double, 4>, 3> debug_sides_before{};
            const auto debug_momentum_totals = [](const auto& advective,
                                                  const auto& pressure,
                                                  std::array<double, 3>& transport,
                                                  std::array<double, 3>& source,
                                                  std::array<double, 3>& stored,
                                                  auto& sides) {
                for (std::size_t component = 0; component < 3; ++component)
                {
                    for (std::size_t bin = 0;
                         bin < advective.west.component[component].size(); ++bin)
                    {
                        sides[component][0] +=
                            advective.west.component[component][bin] +
                            pressure.west.component[component][bin];
                        sides[component][1] +=
                            advective.east.component[component][bin] +
                            pressure.east.component[component][bin];
                        transport[component] +=
                            advective.west.component[component][bin] +
                            pressure.west.component[component][bin] -
                            advective.east.component[component][bin] -
                            pressure.east.component[component][bin];
                    }
                    for (std::size_t bin = 0;
                         bin < advective.south.component[component].size(); ++bin)
                    {
                        sides[component][2] +=
                            advective.south.component[component][bin] +
                            pressure.south.component[component][bin];
                        sides[component][3] +=
                            advective.north.component[component][bin] +
                            pressure.north.component[component][bin];
                        transport[component] +=
                            advective.south.component[component][bin] +
                            pressure.south.component[component][bin] -
                            advective.north.component[component][bin] -
                            pressure.north.component[component][bin];
                    }
                    source[component] = advective.source_impulse[component] +
                                        pressure.source_impulse[component];
                    stored[component] = advective.storage_impulse[component] +
                                        pressure.storage_impulse[component];
                }
            };
            if (two_way)
            {
                debug_inventory_before = nested_two_way_child_inventory(nested);
                debug_interface_before = two_way_interval_interface_source(
                    {}, nested.interface_source.cumulative);
                const auto& momentum = nested.child.runtime.dynamics
                                           .momentum_interface_flux_register;
                debug_momentum_totals(momentum.advective_fluxes(),
                                      momentum.pressure_fluxes(),
                                      debug_transport_before,
                                      debug_source_before,
                                      debug_storage_before,
                                      debug_sides_before);
            }
            NestedChildSubstepReport report;
            advance_nested_child_dynamics_substep(
                storage, nested_previous_parent_fields,
                {config.relaxation_e_folding_time_s}, report,
                &nested_previous_parent_microphysics);
            if (!private_begin.empty())
            {
                const auto& source  = report.interface_source.microphysics_private;
                const auto& capture = nested.child.runtime.advection.private_interface.fields;
                if (source.scalar_count != private_begin.size() || capture.size() != private_begin.size())
                    throw std::runtime_error("private child substep ledger layout changed");
                for (std::size_t f = 0; f < private_begin.size(); ++f)
                {
                    if (!capture[f].boundary.complete || capture[f].name != source.name[f].data())
                        throw std::runtime_error("private child substep omitted transport provenance");
                    private_source[f].add(source.committed_carrier_change[f]);
                    private_storage[f].add(capture[f].boundary.closure_residual);
                }
            }
            if (moist_feedback)
            {
                const auto source = water_sources(report.interface_source);
                for (std::size_t f = 0; f < source.size(); ++f)
                {
                    water_source[f].add(source[f]);
                    water_storage[f].add(
                        nested.child.runtime.advection.thermodynamic_boundary_reports[f + 1].closure_residual);
                }
            }
            if (two_way)
            {
                const auto debug_inventory_after =
                    nested_two_way_child_inventory(nested);
                const auto debug_interface_after =
                    two_way_interval_interface_source(
                        {}, nested.interface_source.cumulative);
                std::array<double, 3> transport_after{};
                std::array<double, 3> source_after{};
                std::array<double, 3> storage_after{};
                std::array<std::array<double, 4>, 3> sides_after{};
                const auto& momentum = nested.child.runtime.dynamics
                                           .momentum_interface_flux_register;
                debug_momentum_totals(momentum.advective_fluxes(),
                                      momentum.pressure_fluxes(),
                                      transport_after, source_after,
                                      storage_after, sides_after);
                const std::array<double, 3> relaxation_change = {
                    report.interface_source.momentum.committed_x_change_kgms,
                    report.interface_source.momentum.committed_y_change_kgms,
                    report.interface_source.momentum.committed_z_change_kgms};
                if (nested.active_terrain.ready)
                {
                    std::array<double, 3> endpoint_storage{};
                    for (std::size_t component = 0; component < 3; ++component)
                        endpoint_storage[component] =
                            (debug_inventory_after[2 + component]
                             - debug_inventory_before[2 + component])
                            - ((transport_after[component]
                                - debug_transport_before[component])
                               + (debug_interface_after[2 + component]
                                  - debug_interface_before[2 + component])
                               + (source_after[component]
                                  - debug_source_before[component])
                               + (storage_after[component]
                                  - debug_storage_before[component]));
                    nested.child.runtime.dynamics
                        .momentum_interface_flux_register
                        .accumulate_mapped_endpoint_storage_impulse(
                            endpoint_storage);
                    for (std::size_t component = 0; component < 3; ++component)
                        storage_after[component] += endpoint_storage[component];
                }
                for (std::size_t component = 0; component < 3; ++component)
                {
                    const double total_change =
                        debug_inventory_after[2 + component] -
                        debug_inventory_before[2 + component];
                    tmv::log_info(
                        "[NESTING TWO-WAY SUBSTEP] component=", component,
                        " change=", total_change,
                        " transport=", transport_after[component] -
                                           debug_transport_before[component],
                        " interface_source=",
                        debug_interface_after[2 + component] -
                            debug_interface_before[2 + component],
                        " dynamics_source=", source_after[component] -
                                               debug_source_before[component],
                        " storage=", storage_after[component] -
                                      debug_storage_before[component],
                        " dynamics_change=",
                        total_change - relaxation_change[component],
                        " relaxation_change=",
                        relaxation_change[component],
                        " west=", sides_after[component][0] -
                                      debug_sides_before[component][0],
                        " east=", sides_after[component][1] -
                                      debug_sides_before[component][1],
                        " south=", sides_after[component][2] -
                                       debug_sides_before[component][2],
                        " north=", sides_after[component][3] -
                                       debug_sides_before[component][3]);
                }
            }
        }
        require_valid_nested_execution_relationship(storage);

        const auto& child = storage.nested_execution.child.fields;
        const Field3D* primary[] = {
            &child.rho, &child.p, &child.u, &child.v, &child.w,
            &child.theta, &child.qv, &child.qc, &child.qr, &child.qi,
            &child.qs, &child.qg, &child.qh, &child.tracer, &child.tke};
        for (const Field3D* field : primary)
            for (std::size_t index = 0; index < field->size(); ++index)
                if (!std::isfinite(static_cast<double>(field->data()[index])))
                    throw std::runtime_error(
                        "nested child produced a non-finite primary value");

        if (two_way)
        {
            if (nested.two_way_feedback_application_count ==
                std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error(
                    "nested two-way feedback application counter overflow");
            NestedTwoWayFeedbackReportV1 feedback;
            const auto& parent_advection = storage.runtime.advection;
            const auto& child_advection = nested.child.runtime.advection;
            const auto& parent_momentum =
                storage.runtime.dynamics.momentum_interface_flux_register;
            const auto& child_momentum =
                nested.child.runtime.dynamics.momentum_interface_flux_register;
            const auto interface_source = two_way_interval_interface_source(
                nested.two_way_interval_source_begin,
                nested.interface_source.cumulative);
            const std::array<double, 5> dynamics_source = {
                0.0, child_advection.moist_interface.enabled ? child_advection.moist_interface.committed_physics[0] : 0.0,
                child_momentum.advective_fluxes().source_impulse[0] +
                    child_momentum.pressure_fluxes().source_impulse[0],
                child_momentum.advective_fluxes().source_impulse[1] +
                    child_momentum.pressure_fluxes().source_impulse[1],
                child_momentum.advective_fluxes().source_impulse[2] +
                    child_momentum.pressure_fluxes().source_impulse[2]};
            const std::array<double, 5> dynamics_storage = {
                nested.child.runtime.time_stepping
                        .cumulative_boundary_mass_closure_residual_kg -
                    nested.two_way_dry_mass_storage_begin,
                nested.child.runtime.advection
                        .cumulative_thermodynamic_boundary_reports[
                            tmv::theta_boundary_report_index]
                        .closure_residual -
                    nested.two_way_theta_storage_begin,
                child_momentum.advective_fluxes().storage_impulse[0] +
                    child_momentum.pressure_fluxes().storage_impulse[0],
                child_momentum.advective_fluxes().storage_impulse[1] +
                    child_momentum.pressure_fluxes().storage_impulse[1],
                child_momentum.advective_fluxes().storage_impulse[2] +
                    child_momentum.pressure_fluxes().storage_impulse[2]};
            const auto next_child_interval_begin =
                nested_two_way_child_inventory(nested);
            NestedMoistFeedbackInputs water_inputs;
            if (moist_feedback)
            {
                for (std::size_t f = 0; f < 7; ++f)
                {
                    water_inputs.parent_flux[f]      = &parent_advection.moist_interface.water[f].scalar();
                    water_inputs.child_flux[f]       = &child_advection.moist_interface.water[f].scalar();
                    water_inputs.child_begin[f]      = water_begin[f + 1];
                    water_inputs.interface_source[f] = water_source[f].value();
                    water_inputs.physics_source[f]   = child_advection.moist_interface.committed_physics[f + 1];
                    water_inputs.transport_storage[f] = water_storage[f].value();
                }
            }
            const auto parent_view          = make_simulation_state_view(storage);
            const bool feedback_diagnostics = step_diagnostics_enabled();
            const auto mass_before =
                feedback_diagnostics ? compute_conservation_budget(parent_view) : ConservationBudget{};
            const auto water_before    = feedback_diagnostics ? compute_moisture_budget(parent_view) : MoistureBudget{};
            const auto momentum_before = feedback_diagnostics ? compute_momentum_budget(parent_view) : MomentumBudget{};
            const auto energy_before   = feedback_diagnostics ? compute_energy_diagnostic_snapshot_v1(parent_view)
                                                              : EnergyDiagnosticSnapshotV1{};
            PrognosticFieldStorage parent_candidate = storage.fields;
            std::array<Field3D, 7> child_candidates;
            if (nested.active_terrain.ready)
                apply_nested_two_way_feedback_mapped(nested.active_terrain.geometry,
                                                     parent_advection.dry_mass_interface_flux_register.fluxes(),
                                                     child_advection.dry_mass_interface_flux_register.fluxes(),
                                                     parent_advection.theta_interface_flux_register.theta(),
                                                     child_advection.theta_interface_flux_register.theta(),
                                                     parent_momentum.advective_fluxes(),
                                                     parent_momentum.pressure_fluxes(),
                                                     child_momentum.advective_fluxes(),
                                                     child_momentum.pressure_fluxes(),
                                                     interface_source,
                                                     dynamics_source,
                                                     dynamics_storage,
                                                     nested.two_way_child_interval_begin_inventory,
                                                     parent_candidate,
                                                     nested.child.fields,
                                                     feedback,
                                                     moist_feedback ? &water_inputs : nullptr,
                                                     moist_feedback ? &child_candidates : nullptr, &storage.runtime.compute);
            else
                apply_nested_two_way_feedback_flat(nested.relationship,
                                                   nested.child.process_grid_metrics.dz,
                                                   parent_advection.dry_mass_interface_flux_register.fluxes(),
                                                   child_advection.dry_mass_interface_flux_register.fluxes(),
                                                   parent_advection.theta_interface_flux_register.theta(),
                                                   child_advection.theta_interface_flux_register.theta(),
                                                   parent_momentum.advective_fluxes(),
                                                   parent_momentum.pressure_fluxes(),
                                                   child_momentum.advective_fluxes(),
                                                   child_momentum.pressure_fluxes(),
                                                   interface_source,
                                                   dynamics_source,
                                                   dynamics_storage,
                                                   nested.two_way_child_interval_begin_inventory,
                                                   parent_candidate,
                                                   nested.child.fields,
                                                   feedback,
                                                   moist_feedback ? &water_inputs : nullptr,
                                                   moist_feedback ? &child_candidates : nullptr, &storage.runtime.compute);
            if (!feedback.complete)
                throw std::runtime_error("nested two-way feedback report is incomplete");
            const auto parent_private = storage.runtime.microphysics.scheme->prognostic_scalars_for_transport();
            const auto child_private  = nested.child.runtime.microphysics.scheme->prognostic_scalars_for_transport();
            parent_advection.private_interface.require_layout(parent_private);
            child_advection.private_interface.require_layout(child_private);
            if (parent_private.size() != child_private.size() || child_private.size() != private_begin.size())
                throw std::runtime_error("private two-way feedback layouts differ");
            std::vector<NestedPrivateScalarFeedbackInput> private_inputs;
            for (std::size_t f = 0; f < private_begin.size(); ++f)
            {
                const auto& coarse = parent_advection.private_interface.fields[f];
                const auto& fine   = child_advection.private_interface.fields[f];
                if (coarse.name != fine.name)
                    throw std::runtime_error("parent and child private carrier identities differ");
                private_inputs.push_back({coarse.name,
                                          parent_private[f].field,
                                          child_private[f].field,
                                          &coarse.flux.scalar(),
                                          &fine.flux.scalar(),
                                          private_begin[f],
                                          private_source[f].value(),
                                          fine.committed_physics,
                                          private_storage[f].value()});
                if (parent_private[f].mass_species != child_private[f].mass_species)
                    throw std::runtime_error("parent and child private mass bindings differ");
                const auto species = parent_private[f].mass_species;
                if (species != tmv::MicrophysicsMassSpecies::None)
                {
                    const std::array<const Field3D*, 6> parent_before{&storage.fields.qc,
                                                                      &storage.fields.qr,
                                                                      &storage.fields.qi,
                                                                      &storage.fields.qs,
                                                                      &storage.fields.qg,
                                                                      &storage.fields.qh};
                    const std::array<const Field3D*, 6> parent_after{&parent_candidate.qc,
                                                                     &parent_candidate.qr,
                                                                     &parent_candidate.qi,
                                                                     &parent_candidate.qs,
                                                                     &parent_candidate.qg,
                                                                     &parent_candidate.qh};
                    const std::array<const Field3D*, 6> child_mass{&nested.child.fields.qc,
                                                                   &nested.child.fields.qr,
                                                                   &nested.child.fields.qi,
                                                                   &nested.child.fields.qs,
                                                                   &nested.child.fields.qg,
                                                                   &nested.child.fields.qh};
                    const auto index = static_cast<std::size_t>(species) - 1;
                    if (index >= parent_before.size())
                        throw std::runtime_error("private mass species is invalid");
                    auto& binding              = private_inputs.back();
                    binding.parent_mass_before = parent_before[index];
                    binding.parent_mass_after  = parent_after[index];
                    binding.child_mass =
                        child_candidates[index + 1].empty() ? child_mass[index] : &child_candidates[index + 1];
                }
            }
            const auto prepare_private = [&](const Field3D* returned)
            {
                return prepare_nested_private_feedback(nested.relationship,
                                                       nested.child.process_grid_metrics.dz,
                                                       nested.active_terrain.ready ? &nested.active_terrain.geometry
                                                                                   : nullptr,
                                                       storage.fields.rho,
                                                       parent_candidate.rho,
                                                       nested.child.fields.rho,
                                                       private_inputs,
                                                       &storage.runtime.compute,
                                                       returned);
            };
            auto private_candidate = [&]()
            {
                try
                {
                    return prepare_private(nullptr);
                }
                catch (const NestedScalarInventoryExhausted&)
                {
                    const auto returned = tmv::completed_cloud_evaporation_return(
                        parent_view, parent_advection.dry_mass_interface_flux_register.fluxes().accumulated_dt_s);
                    if (returned.empty())
                    {
                        throw;
                    }
                    return prepare_private(&returned);
                }
            }();
            feedback.private_scalars = std::move(private_candidate.reports);

            const std::uint64_t          next_application_count = nested.two_way_feedback_application_count + 1;
            static constexpr const char* currencies[]           = {
                "dry_mass", "rho_theta", "x_momentum", "y_momentum", "z_momentum"};
            for (std::size_t currency = 0; currency < 5; ++currency)
                tmv::log_info("[NESTING TWO-WAY CLOSURE] currency=",
                              currencies[currency],
                              " mismatch=",
                              feedback.interface_mismatch[currency],
                              " reflux=",
                              feedback.applied_reflux[currency],
                              " child_transport=",
                              feedback.child_interface_transport[currency],
                              " interface_source=",
                              feedback.child_interface_source[currency],
                              " dynamics_source=",
                              feedback.child_dynamics_source[currency],
                              " dynamics_storage=",
                              feedback.child_dynamics_storage[currency],
                              " interval_change=",
                              feedback.child_interval_change[currency],
                              " residual=",
                              feedback.child_composite_residual[currency],
                              " tolerance=",
                              feedback.child_composite_tolerance[currency],
                              " normalized_residual=",
                              std::abs(feedback.child_composite_residual[currency]) /
                                  feedback.child_composite_tolerance[currency]);
            if (feedback.moist)
            {
                static constexpr const char* names[]{"qv", "qc", "qr", "qi", "qs", "qg", "qh"};
                for (std::size_t f = 0; f < 7; ++f)
                {
                    const auto& item = feedback.water[f];
                    tmv::log_info("[NESTING TWO-WAY WATER] field=",
                                  names[f],
                                  ", reflux=",
                                  item.parent.applied_reflux,
                                  ", redistributed=",
                                  item.parent.redistributed_inventory,
                                  ", redistributed_cells=",
                                  item.parent.redistributed_cells,
                                  ", physics_source=",
                                  item.physics_source,
                                  ", interface_source=",
                                  item.interface_source,
                                  ", residual=",
                                  item.closure_residual,
                                  ", tolerance=",
                                  item.tolerance,
                                  ", normalized_residual=",
                                  item.tolerance ? std::abs(item.closure_residual) / item.tolerance : 0.0,
                                  ", global_redistributed=",
                                  item.parent.global_redistributed_inventory,
                                  ", global_redistributed_cells=",
                                  item.parent.global_redistributed_cells,
                                  ", fine_transfer=",
                                  item.parent.fine_requested_transfer,
                                  ", fine_storage=",
                                  item.parent.fine_storage_rounding,
                                  ", phase_exchange=",
                                  item.parent.phase_exchange_inventory,
                                  ", child_synchronized=",
                                  item.parent.child_synchronized_inventory);
                }
            }
            for (const auto& scalar : feedback.private_scalars)
                tmv::log_info("[NESTING TWO-WAY PRIVATE] field=",
                              scalar.name,
                              ", reflux=",
                              scalar.carrier.parent.applied_reflux,
                              ", physics_source=",
                              scalar.carrier.physics_source,
                              ", interface_source=",
                              scalar.carrier.interface_source,
                              ", transport_storage=",
                              scalar.carrier.transport_storage,
                              ", residual=",
                              scalar.carrier.closure_residual,
                              ", tolerance=",
                              scalar.carrier.tolerance,
                              " phase_exchange=",
                              scalar.carrier.parent.phase_exchange_inventory);
            if (feedback.phase_projection_cells)
                tmv::log_info("[NESTING TWO-WAY PHASE] cells=", feedback.phase_projection_cells,
                    ", rho_theta_change=", feedback.phase_theta_inventory_change,
                    ", thermal_storage=", feedback.phase_thermal_roundoff_j,
                    ", thermal_scale=", feedback.phase_thermal_scale_j);
            SimulationStateView candidate_view{
                parent_view.grid,
                std::apply([&](const auto&... member) { return PrognosticFieldsView{member.get(parent_candidate)...}; },
                           tmv::fields::prognostic),
                parent_view.base,
                parent_view.tendencies,
                parent_view.coriolis_f,
                parent_view.time_s,
                parent_view.surface};
            storage.runtime.bind(candidate_view);
            candidate_view.configuration        = parent_view.configuration;
            candidate_view.domain_frame         = parent_view.domain_frame;
            candidate_view.process_grid_metrics = parent_view.process_grid_metrics;
            const auto mass_after =
                feedback_diagnostics ? compute_conservation_budget(candidate_view) : ConservationBudget{};
            const auto water_after = feedback_diagnostics ? compute_moisture_budget(candidate_view) : MoistureBudget{};
            const auto momentum_after =
                feedback_diagnostics ? compute_momentum_budget(candidate_view) : MomentumBudget{};
            const auto energy_after = feedback_diagnostics ? compute_energy_diagnostic_snapshot_v1(candidate_view)
                                                           : EnergyDiagnosticSnapshotV1{};
            if (feedback_diagnostics)
            {
                double              report_time = nested.parent_time_bracket.previous_parent_time_s;
                const double        interval    = nested.parent_time_bracket.current_parent_time_s - report_time;
                SimulationStateView report_view{parent_view.grid,
                                                candidate_view.fields,
                                                parent_view.base,
                                                parent_view.tendencies,
                                                parent_view.coriolis_f,
                                                report_time,
                                                parent_view.surface};
                storage.runtime.bind(report_view);
                report_view.configuration        = parent_view.configuration;
                report_view.domain_frame         = parent_view.domain_frame;
                report_view.process_grid_metrics = parent_view.process_grid_metrics;
                report_budget_transition(report_view, "nested_feedback", mass_before, mass_after, interval);
                report_moisture_transition(report_view, "nested_feedback", water_before, water_after, interval);
                report_momentum_transition(report_view, "nested_feedback", momentum_before, momentum_after, interval);
                report_energy_transition(report_view, "nested_feedback", energy_before, energy_after, interval);
            }
            using std::swap;
            swap(storage.fields, parent_candidate);
            const std::array<Field3D*, 7> child_water{&nested.child.fields.qv,
                                                      &nested.child.fields.qc,
                                                      &nested.child.fields.qr,
                                                      &nested.child.fields.qi,
                                                      &nested.child.fields.qs,
                                                      &nested.child.fields.qg,
                                                      &nested.child.fields.qh};
            for (std::size_t f = 0; f < child_water.size(); ++f)
                if (!child_candidates[f].empty())
                    swap(*child_water[f], child_candidates[f]);
            for (std::size_t f = 0; f < private_candidate.parent.size(); ++f)
            {
                swap(*parent_private[f].field, private_candidate.parent[f]);
                if (!private_candidate.child[f].empty())
                    swap(*child_private[f].field, private_candidate.child[f]);
            }
            nested.child.runtime.advection.invalidate_hydrometeor_activity();
            tmv::invalidate_microphysics_reaction_snapshot(storage.runtime.microphysics);
            if (storage.runtime.compute.backend)
                storage.runtime.compute.backend->invalidate_all_resident_fields();
            nested.last_two_way_feedback                  = std::move(feedback);
            nested.two_way_feedback_application_count     = next_application_count;
            nested.two_way_child_interval_begin_inventory = next_child_interval_begin;
            nested.two_way_interval_source_begin          = nested.interface_source.cumulative;
            nested.two_way_dry_mass_storage_begin =
                nested.child.runtime.time_stepping.cumulative_boundary_mass_closure_residual_kg;
            nested.two_way_theta_storage_begin =
                nested.child.runtime.advection
                    .cumulative_thermodynamic_boundary_reports[
                        tmv::theta_boundary_report_index]
                    .closure_residual;
            reset_two_way_level_flux_capture(storage.runtime);
            reset_two_way_level_flux_capture(nested.child.runtime);
        }
        two_way_rollback.reset();
    }
    catch (const std::exception& error)
    {
        if (two_way_rollback)
        {
            storage.nested_execution = std::move(*two_way_rollback);
            if (storage.runtime.compute.backend)
                storage.runtime.compute.backend->invalidate_all_resident_fields();
        }
        tmv::log_error("[NESTING] child advance failed: ", error.what());
        return 2;
    }
    return 0;
}



bool FieldValidation::validate_core_fields(const SimulationStateView& execution_state, const std::string&           context,
                                          int                          step_index,
                                          bool                         include_percentiles,
                                          const std::filesystem::path* report_path,
                                          bool                         persist_report)
{
    const auto fields = core_field_binding_array(execution_state, &execution_state.fields.u, &execution_state.fields.v, &execution_state.fields.w);
    const bool finite_only =
        !include_percentiles && report_path == nullptr && !persist_report &&
        (tmv::require_run_configuration(execution_state).validation_policy.mode == tmv::GuardMode::Off || tmv::require_run_configuration(execution_state).validation_policy.mode == tmv::GuardMode::Warn);
    if (finite_only)
    {
        int nonfinite_found = 0;
#pragma omp parallel for reduction(| : nonfinite_found)
        for (std::size_t field_index = 0; field_index < fields.size(); ++field_index)
        {
            const auto&               binding  = fields[field_index];
            const tmv::FieldContract* contract = tmv::find_field_contract(binding.name);
            if (contract == nullptr || contract->status == tmv::FieldImplementationStatus::NotImplemented)
            {
                continue;
            }
            const float*      values = binding.field->data();
            const std::size_t count  = binding.field->size();
            for (std::size_t value_index = 0; value_index < count; ++value_index)
            {
                nonfinite_found |= !std::isfinite(static_cast<double>(values[value_index]));
            }
        }
        if (nonfinite_found == 0)
            return true;

        validation_error_message =
            "non-finite primary state in context='" + context + "' step=" + std::to_string(step_index);
        tmv::log_error("[VALIDATION] ", validation_error_message);
        return false;
    }

    tmv::ValidationReport report;
    report.context       = context;
    report.step_index    = step_index;
    report.guard_mode    = tmv::to_string(tmv::require_run_configuration(execution_state).validation_policy.mode);
    report.guard_fail_on = tmv::to_string(tmv::require_run_configuration(execution_state).validation_policy.fail_on);
    report.guard_scope   = tmv::to_string(tmv::require_run_configuration(execution_state).validation_policy.strict_scope);

    for (const auto& binding : fields)
    {
        const tmv::FieldContract* contract = tmv::find_field_contract(binding.name);
        if (contract == nullptr)
        {
            report.missing_required.push_back(binding.name);
            continue;
        }
        if (contract->status == tmv::FieldImplementationStatus::NotImplemented)
        {
            if (contract->requirement == tmv::FieldRequirementTier::RequiredNow)
            {
                report.missing_required.push_back(binding.name);
            }
            continue;
        }

        tmv::FieldValidationReport field_report;
        field_report.field_id    = contract->id;
        field_report.status      = tmv::to_string(contract->status);
        field_report.requirement = tmv::to_string(contract->requirement);
        field_report.result =
            tmv::validate_field3d_inplace(*binding.field, *contract, tmv::require_run_configuration(execution_state).validation_policy, include_percentiles);

        if (field_report.result.failed)
        {
            report.failed = true;
        }
        report.fields.push_back(std::move(field_report));
    }

    if (persist_report)
    {
        const auto not_implemented = tmv::contracts_with_status(tmv::FieldImplementationStatus::NotImplemented);
        report.known_not_implemented.reserve(not_implemented.size());
        for (const tmv::FieldContract* contract : not_implemented)
        {
            report.known_not_implemented.push_back(contract->id);
            if (contract->requirement == tmv::FieldRequirementTier::RequiredNow)
            {
                report.missing_not_implemented.push_back(contract->id);
            }
        }
    }

    if (!report.missing_required.empty() && tmv::require_run_configuration(execution_state).validation_policy.mode == tmv::GuardMode::Strict)
    {
        report.failed = true;
    }

    // Bounds policy remains mode-dependent, but a primary field containing
    // NaN/Inf is never a successful science result. In particular, Warn is an
    // observation policy rather than permission to advance and publish an
    // undefined state. validate_* reports the pre-sanitize counts, so this
    // also catches a non-finite discovered while running in Sanitize mode.
    const bool fatal_nonfinite = tmv::validation_report_has_nonfinite(report);
    report.failed              = report.failed || fatal_nonfinite;

    if (report_path != nullptr)
    {
        std::string write_error;
        if (!tmv::write_validation_report_json(report, *report_path, write_error))
        {
            validation_error_message = write_error;
            return false;
        }
    }

    if (persist_report)
    {
        export_validation_reports.push_back(report);
    }

    if (fatal_nonfinite)
    {
        validation_error_message =
            "non-finite primary state in context='" + context + "' step=" + std::to_string(step_index);
        tmv::log_error("[VALIDATION] ", validation_error_message);
        return false;
    }

    if (report.failed && tmv::require_run_configuration(execution_state).validation_policy.mode == tmv::GuardMode::Strict)
    {
        validation_error_message =
            "strict guard failure in context='" + context + "' step=" + std::to_string(step_index);
        tmv::log_error("[VALIDATION] strict guard failure in context='", context, "' step=", step_index);
        return false;
    }

    return true;
}

const std::vector<double>* OutputPublisher::refresh_cgrid_ground_vertical_velocity()
{
    return refresh_cgrid_ground_vertical_velocity(make_simulation_state_view(run));
}

const std::vector<double>* OutputPublisher::refresh_cgrid_ground_vertical_velocity(const SimulationStateView& execution_state)
{
    const SimulationStateView state = execution_state;
    if (tmv::require_run_configuration(execution_state).stagger_type != StaggerType::CGrid || !tmv::physical_grid_metrics::terrain_is_active(state))
    {
        return nullptr;
    }

    const TerrainMetrics3D* metrics  = tmv::physical_grid_metrics::active_terrain_metrics(state);
    const std::size_t       expected = static_cast<std::size_t>(execution_state.grid.nr) * static_cast<std::size_t>(execution_state.grid.nth);
    if (metrics == nullptr || execution_state.grid.nr < 2 || execution_state.grid.nth < 2
        || cgrid_ground_vertical_velocity_ms.capacity() < expected)
    {
        throw std::runtime_error(
            "terrain C-grid output ground-velocity workspace is unavailable: grid="
            + std::to_string(execution_state.grid.nr) + "x" + std::to_string(execution_state.grid.nth)
            + ", workspace="
            + std::to_string(cgrid_ground_vertical_velocity_ms.size())
            + ", metrics=" + (metrics != nullptr ? "active" : "missing"));
    }
    cgrid_ground_vertical_velocity_ms.resize(expected);

    // The stored lower x/y closure columns do not have both staggered
    // neighbours. Their published value follows the nearest physical scalar
    // column, consistent with the zero-normal-gradient terrain closure. Valid
    // physical cells and the upper closure columns use their own geometry.
    for (int i = 0; i < execution_state.grid.nr; ++i)
        for (int j = 0; j < execution_state.grid.nth; ++j)
        {
            const int sample_i = std::max(i, 1);
            const int sample_j = std::max(j, 1);
            cgrid_ground_vertical_velocity_ms[static_cast<std::size_t>(i) * static_cast<std::size_t>(execution_state.grid.nth) +
                                              static_cast<std::size_t>(j)] =
                tmv::terrain_transport::cgrid_ground_vertical_velocity(
                    *metrics, state.fields.u, state.fields.v,
                    sample_i, sample_j,
                    tmv::physical_grid_metrics::surface_velocity_ms(state).u_ms,
                    tmv::physical_grid_metrics::surface_velocity_ms(state).v_ms);
        }

    return &cgrid_ground_vertical_velocity_ms;
}

tmv_shm::ShmHorizontalFrame OutputPublisher::live_shm_frame() const
{
    tmv_shm::ShmHorizontalFrame frame;
    frame.motion_mode =
        global_domain_motion_mode ==
                tmv::init::config::DomainMotionMode::ConstantVelocity
            ? tmv_shm::ShmDomainMotionMode::ConstantVelocity
            : tmv_shm::ShmDomainMotionMode::Stationary;
    frame.grid_origin_t0_earth_m[0] =
        global_domain_frame.earth_origin_at_t0_m.x_m;
    frame.grid_origin_t0_earth_m[1] =
        global_domain_frame.earth_origin_at_t0_m.y_m;
    const auto current_origin =
        global_domain_frame.earth_origin_at(simulation_time);
    frame.grid_origin_earth_m[0] = current_origin.x_m;
    frame.grid_origin_earth_m[1] = current_origin.y_m;
    frame.frame_velocity_earth_ms[0] = global_domain_frame.velocity_ms.u_ms;
    frame.frame_velocity_earth_ms[1] = global_domain_frame.velocity_ms.v_ms;
    return frame;
}

void OutputPublisher::refresh_output_diagnostics(const SimulationStateView& state, OutputFieldDependency dependencies)
{
    const bool vorticity = output_field_has_dependency(dependencies, OutputFieldDependency::VorticityDiagnostics);
    const bool angular   = output_field_has_dependency(dependencies, OutputFieldDependency::AngularMomentumDiagnostics);
    const bool pressure  = output_field_has_dependency(dependencies, OutputFieldDependency::PressureDiagnostics);
    const bool components = output_field_has_dependency(dependencies, OutputFieldDependency::PressureComponents);
    // Feedback and relocation can change fields after the step diagnostics were computed.
    if (output_field_has_dependency(dependencies, OutputFieldDependency::RadarDiagnostic))
        calculate_radar_observables(state);
    tmv::diagnostics::compute_resolved_diagnostics(state, {vorticity, angular, pressure, components});
}

bool OutputPublisher::shm_update(bool force)
{
    if (!shm_writer.is_open())
    {
        return true;
    }
    if (!force &&
        !tmv::runtime::live_shm_publish_due(true, options.live_shm_full_rate, simulation_time, next_shm_publish_time_s))
    {
        ++live_shm_skipped_steps;
        return true;
    }
    const auto  publish_start = PerfClock::now();
    const auto& shm_fields    = options.live_shm_fields;
    refresh_output_diagnostics(make_simulation_state_view(run), output_dependencies_for_fields(shm_fields));

    // C-grid output interpolation (Phase C.8): on the C-grid the
    // prognostic u, v, w live at faces; the live SHM viewer expects
    // cell-center velocities so all streamed fields share the same
    // coordinate convention. Stage only components required by the selected
    // live fields, once per SHM tick.
    const Field3D* u_shm = &u;
    const Field3D* v_shm = &v;
    const Field3D* w_shm = &w;
    if (global_stagger_type == StaggerType::CGrid)
    {
        const CenteredVelocityStagingPlan staging =
            centered_velocity_staging_plan(true, output_dependencies_for_fields(shm_fields));
        if (staging.u)
        {
            interpolate_u_face_to_center(u, shm_u_center_buf);
            u_shm = &shm_u_center_buf;
        }
        if (staging.v)
        {
            interpolate_v_face_to_center(v, shm_v_center_buf);
            v_shm = &shm_v_center_buf;
        }
        if (staging.w)
        {
            if (const std::vector<double>* ground_w = refresh_cgrid_ground_vertical_velocity())
            {
                interpolate_w_face_to_center(w, shm_w_center_buf, *ground_w);
            }
            else
            {
                interpolate_w_face_to_center(w, shm_w_center_buf);
            }
            w_shm = &shm_w_center_buf;
        }
    }

    // "reflectivity_dbz" must be published as dBZ (10*log10 Z), not the raw
    // linear radar_reflectivity the field holds; build a converted buffer on
    // demand so a live viewer asking for reflectivity_dbz sees dBZ (matching
    // the npy / Studio "reflectivity_dbz" field). "radar" stays linear.
    const Field3D* reflectivity_dbz_shm = &radar_reflectivity;
    const bool     want_reflectivity_dbz =
        std::find(shm_fields.begin(), shm_fields.end(), std::string("reflectivity_dbz")) != shm_fields.end();
    if (want_reflectivity_dbz)
    {
        float dbz_min = -30.0f;
        float dbz_max = 120.0f;
        reflectivity_dbz_bounds(dbz_min, dbz_max);
        fill_reflectivity_dbz_field(
            radar_reflectivity, shm_reflectivity_dbz_buf,
            NR, NTH, NZ, dbz_min, dbz_max);
        reflectivity_dbz_shm = &shm_reflectivity_dbz_buf;
    }

    // Reuse the core export registry so the configured live-field validation
    // and actual publisher cannot drift apart. reflectivity_dbz is the one
    // derived live-only binding added after its conversion above.
    auto core_map = core_field_bindings(make_simulation_state_view(run), u_shm, v_shm, w_shm);
    core_map.push_back({"reflectivity_dbz", reflectivity_dbz_shm});
    bool wrote_all_fields = true;
    for (int fi = 0; fi < static_cast<int>(shm_fields.size()); ++fi)
    {
        bool found_field = false;
        for (const auto& binding : core_map)
        {
            if (shm_fields[fi] == binding.name)
            {
                found_field      = true;
                wrote_all_fields = shm_writer.write_field(fi, *binding.field) && wrote_all_fields;
                break;
            }
        }
        if (!found_field)
        {
            wrote_all_fields = false;
            tmv::log_error("[SHM] configured field has no live publisher: ", shm_fields[fi]);
        }
    }
    const bool committed =
        shm_writer.commit(simulation_time, live_shm_frame());
    if (!wrote_all_fields || !committed)
    {
        ++live_shm_failed_frames;
        tmv::log_error("[SHM] incomplete live field bundle was not published");
        return false;
    }
    ++live_shm_frames;
    live_shm_payload_bytes +=
        static_cast<std::uint64_t>(options.live_shm_fields.size()) *
        static_cast<std::uint64_t>(NR) * static_cast<std::uint64_t>(NTH) *
        static_cast<std::uint64_t>(NZ) * sizeof(float);
    live_shm_publish_time_s += std::chrono::duration<double>(
        PerfClock::now() - publish_start).count();
    if (!force && !options.live_shm_full_rate)
        next_shm_publish_time_s =
            tmv::runtime::next_live_shm_publish_time(
                simulation_time, options.live_shm_interval_s);
    return true;
}

#ifdef EXPORT_NPY


bool FieldValidation::validate_derived_export_slice(const SimulationStateView& execution_state, const char*         field_id,
                                                   std::vector<float>& values,
                                                   int                 theta_index,
                                                   int                 frame_export_index)
{
    const tmv::FieldContract* contract = tmv::find_field_contract(field_id);
    if (contract == nullptr || contract->status != tmv::FieldImplementationStatus::ExportedNow)
    {
        return true;
    }

    tmv::FieldValidationResult result =
        tmv::validate_buffer_inplace(values.data(), values.size(), *contract, tmv::require_run_configuration(execution_state).validation_policy, false);

    if (result.failed && tmv::require_run_configuration(execution_state).validation_policy.mode == tmv::GuardMode::Strict)
    {
        validation_error_message = "strict guard failure for derived export field='" + std::string(field_id) +
                                   "' step=" + std::to_string(frame_export_index) +
                                   " theta=" + std::to_string(theta_index);
        tmv::log_error("[VALIDATION] strict guard failure for derived export field='",
                       field_id,
                       "' step=",
                       frame_export_index,
                       " theta=",
                       theta_index);
        return false;
    }

    return true;
}

bool OutputPublisher::prepare_write_frame(WriteFrameContext& frame)
{
    if (export_diagnostics_enabled && frame.frame_export_index == 0)
    {
        float theta_min = 1e10, theta_max = -1e10;
        float u_min = 1e10, u_max = -1e10;
        int   nan_count = 0;

        for (int i = 0; i < frame.state.grid.nr && i < 10; ++i)
        {
            for (int j = 0; j < frame.state.grid.nth && j < 3; ++j)
            {
                for (int k = 0; k < frame.state.grid.nz && k < 3; ++k)
                {
                    if (std::isnan(frame.state.fields.theta[i][j][k]))
                        nan_count++;
                    if (frame.state.fields.theta[i][j][k] < theta_min)
                        theta_min = frame.state.fields.theta[i][j][k];
                    if (frame.state.fields.theta[i][j][k] > theta_max)
                        theta_max = frame.state.fields.theta[i][j][k];
                    if (frame.state.fields.u[i][j][k] < u_min)
                        u_min = frame.state.fields.u[i][j][k];
                    if (frame.state.fields.u[i][j][k] > u_max)
                        u_max = frame.state.fields.u[i][j][k];
                }
            }
        }
        tmv::log_info("[EXPORT DIAGNOSTIC] frame=", frame.frame_export_index,
                      " time_s=", frame.state.time_s,
                      " sample_theta_min_k=", theta_min,
                      " sample_theta_max_k=", theta_max,
                      " sample_u_min_ms=", u_min,
                      " sample_u_max_ms=", u_max,
                      " sample_nan=", nan_count);

        if (theta_min < 0 || theta_max > 700) // matches simulation.hpp theta_max_k
        {
            tmv::log_error("[EXPORT DIAGNOSTIC] theta sample is outside the runtime contract");
        }
    }

    std::filesystem::path publication_root(outdir);
    if (publishing_nested_child)
        publication_root /= std::filesystem::path("nests") / "child_0000";
    if (publishing_qualification_diagnostics)
        publication_root /= "qualification_diagnostics";
    const std::uintmax_t reserve_bytes =
        static_cast<std::uintmax_t>(output_config.minimum_free_space_mib) * 1024U * 1024U;
    const std::uintmax_t estimated_frame_bytes = publishing_qualification_diagnostics
        ? output_config.qualification_diagnostics_estimated_bytes_per_export
        : output_config.estimated_bytes_per_export;
    const std::uintmax_t required_bytes =
        estimated_frame_bytes > std::numeric_limits<std::uintmax_t>::max() - reserve_bytes
        ? std::numeric_limits<std::uintmax_t>::max()
        : reserve_bytes + estimated_frame_bytes;
    std::string free_space_error;
    if (output_config.minimum_free_space_mib > 0 &&
        !nimbus::output::require_free_space(publication_root, required_bytes, free_space_error))
    {
        validation.validation_error_message = free_space_error;
        tmv::log_error("[OUTPUT] ", free_space_error);
        return false;
    }
    std::ostringstream stepdir;
    stepdir << publication_root.string() << "/step_" << std::setfill('0') << std::setw(6)
            << frame.frame_export_index;
    frame.step_str  = stepdir.str();
    frame.step_path = std::filesystem::path(frame.step_str);
    std::error_code remove_ec;
    std::filesystem::remove_all(frame.step_path, remove_ec);
    std::filesystem::create_directories(frame.step_path);

    frame.report_path = frame.step_path / "validation_report.json";
    if (!tmv::require_run_configuration(frame.state).validation_report_path.empty())
    {
        std::filesystem::path report_root(tmv::require_run_configuration(frame.state).validation_report_path);
        std::ostringstream    report_name;
        if (publishing_nested_child)
            report_name << "child_0000_";
        if (publishing_qualification_diagnostics)
            report_name << "qualification_diagnostics_";
        report_name << "step_" << std::setfill('0') << std::setw(6) << frame.frame_export_index
                    << "_validation_report.json";
        frame.report_path = report_root / report_name.str();
    }

    const char* validation_context = publishing_qualification_diagnostics
        ? (publishing_nested_child
               ? "pre_export_nested_child_qualification_diagnostics"
               : "pre_export_qualification_diagnostics")
        : (publishing_nested_child ? "pre_export_nested_child" : "pre_export");
    if (!validation.validate_core_fields(frame.state, validation_context,
                              frame.frame_export_index,
                              true,
                              &frame.report_path,
                              true))
    {
        return false;
    }

    reflectivity_dbz_bounds(frame.reflectivity_dbz_min, frame.reflectivity_dbz_max);
    frame.use_3d = (output_config.format == OutputFormat::npy_3d || output_config.format == OutputFormat::netcdf ||
                    output_config.format == OutputFormat::csv || output_config.format == OutputFormat::zfp);
    // netCDF and ZFP are always assembled as one publication snapshot. The
    // writer still runs synchronously when async_io=false; despite its legacy
    // name, this flag selects snapshot assembly instead of the direct NPY path.
    // Sending synchronous ZFP through that path would otherwise publish plain
    // .npy payloads under a manifest that falsely claimed ZFP storage.
    frame.use_async = frame.use_3d && requires_snapshot_assembly(output_config.format, output_config.async_io);
    const auto simulation_state  = frame.state;
    const bool terrain_following = tmv::physical_grid_metrics::terrain_is_active(simulation_state);
    if (terrain_following && !frame.use_3d)
    {
        tmv::log_error("[EXPORT] terrain-following output requires a 3-D volume format; "
                       "npy_2d_slices cannot publish complete physical coordinates");
        return false;
    }
    // Terrain geometry sidecars require snapshot ownership even for a
    // synchronous NPY/CSV writer. This keeps every field and its coordinates
    // in the same atomic publication transaction.
    frame.use_async                        = frame.use_async || terrain_following;
    frame.snapshot.export_index            = frame.frame_export_index;
    frame.snapshot.simulation_time_s       = frame.state.time_s;
    frame.snapshot.step_dir                = frame.step_path;
    frame.snapshot.data_product_kind = publishing_qualification_diagnostics
        ? (publishing_nested_child
               ? "nested_child_qualification_diagnostics"
               : "qualification_diagnostics")
        : (publishing_nested_child ? "nested_child_history" : "history");
    frame.snapshot.grid.nx                 = frame.state.grid.nr;
    frame.snapshot.grid.ny                 = frame.state.grid.nth;
    frame.snapshot.grid.nz                 = frame.state.grid.nz;
    frame.snapshot.grid.dx_m               = frame.state.grid.dr;
    frame.snapshot.grid.dy_m               = frame.state.grid.dtheta;
    frame.snapshot.grid.dz_m               = frame.state.grid.dz;
    frame.snapshot.grid.coordinate_system  = coordinate_system_name(tmv::require_run_configuration(frame.state).coordinate_system);
    frame.snapshot.grid.native_staggering  = stagger_type_name(tmv::require_run_configuration(frame.state).stagger_type);
    frame.snapshot.grid.domain_motion_mode = tmv::init::config::domain_motion_mode_name(tmv::require_run_configuration(frame.state).domain_motion_mode);
    frame.snapshot.grid.grid_origin_t0_earth_x_m          = tmv::require_level_runtime(frame.state.domain_frame, "domain frame").earth_origin_at_t0_m.x_m;
    frame.snapshot.grid.grid_origin_t0_earth_y_m          = tmv::require_level_runtime(frame.state.domain_frame, "domain frame").earth_origin_at_t0_m.y_m;
    const tmv::domain::HorizontalPoint publication_origin = tmv::require_level_runtime(frame.state.domain_frame, "domain frame").earth_origin_at(frame.state.time_s);
    frame.snapshot.grid.grid_origin_earth_x_m             = publication_origin.x_m;
    frame.snapshot.grid.grid_origin_earth_y_m             = publication_origin.y_m;
    frame.snapshot.grid.frame_velocity_u_ms               = tmv::require_level_runtime(frame.state.domain_frame, "domain frame").velocity_ms.u_ms;
    frame.snapshot.grid.frame_velocity_v_ms               = tmv::require_level_runtime(frame.state.domain_frame, "domain frame").velocity_ms.v_ms;
    frame.snapshot.grid.terrain_following                 = terrain_following;
    if (terrain_following)
    {
        tmv::physical_grid_metrics::validate(simulation_state);
        const TerrainMetrics3D* metrics         = tmv::physical_grid_metrics::active_terrain_metrics(simulation_state);
        frame.snapshot.grid.terrain_geometry_publication_generation =
            output_publication_generation;
        frame.snapshot.grid.terrain_geometry_sha256 =
            terrain_geometry_content_sha256(
                frame.state.grid.nr, frame.state.grid.nth, frame.state.grid.nz, metrics->z.values, metrics->z_face.values);
        const PhysicalCellDomain domain =
            physical_scalar_domain(frame.state.grid.nr, frame.state.grid.nth, frame.state.grid.nz, tmv::require_run_configuration(frame.state).stagger_type);
        frame.snapshot.grid.terrain_min_z_m =
            std::numeric_limits<double>::infinity();
        frame.snapshot.grid.terrain_max_z_m =
            -std::numeric_limits<double>::infinity();
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
            {
                const std::size_t base =
                    (static_cast<std::size_t>(i) * frame.state.grid.nth + j) *
                    static_cast<std::size_t>(frame.state.grid.nz + 1);
                frame.snapshot.grid.terrain_min_z_m = std::min(
                    frame.snapshot.grid.terrain_min_z_m,
                    metrics->z_face.values[base + domain.k_begin]);
                frame.snapshot.grid.terrain_max_z_m = std::max(
                    frame.snapshot.grid.terrain_max_z_m,
                    metrics->z_face.values[base + domain.k_end]);
            }
        frame.snapshot.grid.physical_z_center_m = metrics->z.values;
        frame.snapshot.grid.physical_z_interface_m = metrics->z_face.values;
    }
    return true;
}

void OutputPublisher::prepare_core_output_fields(WriteFrameContext& frame)
{
    // C-grid output interpolation (Phase C.8): the prognostic u, v, w
    // live at faces on the C-grid (u at r-face, v at theta-face,
    // w at z-face). Output consumers (npy writer, slice writer, the
    // SHM live channel) expect cell-center velocities so all fields
    // share the same coordinate convention. Build cell-center
    // buffers once per export step and dispatch core_bindings to
    // them when on C-grid.
    frame.u_for_output = &frame.state.fields.u;
    frame.v_for_output = &frame.state.fields.v;
    frame.w_for_output = &frame.state.fields.w;
    if (tmv::require_run_configuration(frame.state).stagger_type == StaggerType::CGrid)
    {
        // Stage only the centered components required by the selected core or
        // derived products.
        const CenteredVelocityStagingPlan staging =
            centered_velocity_staging_plan(true, output_dependencies_for_fields(active_export_fields));
        if (staging.u)
        {
            interpolate_u_face_to_center(frame.state.fields.u, frame.u_center_buf);
            frame.u_for_output = &frame.u_center_buf;
        }
        if (staging.v)
        {
            interpolate_v_face_to_center(frame.state.fields.v, frame.v_center_buf);
            frame.v_for_output = &frame.v_center_buf;
        }
        if (staging.w)
        {
            if (const std::vector<double>* ground_w = refresh_cgrid_ground_vertical_velocity(frame.state))
            {
                interpolate_w_face_to_center(frame.state.fields.w, frame.w_center_buf, *ground_w);
            }
            else
            {
                interpolate_w_face_to_center(frame.state.fields.w, frame.w_center_buf);
            }
            frame.w_for_output = &frame.w_center_buf;
        }
    }

}

void OutputPublisher::write_3d_core_fields(WriteFrameContext& frame)
{
    if (!frame.use_3d)
    {
        return;
    }

    const auto core_bindings = core_field_bindings(frame.state, frame.u_for_output, frame.v_for_output, frame.w_for_output);
    write_or_snapshot_core_volume_fields(core_bindings,
                                         active_export_fields,
                                         frame.snapshot,
                                         frame.use_async,
                                         output_config.format,
                                         frame.state.grid.nr,
                                         frame.state.grid.nth,
                                         frame.state.grid.nz,
                                         frame.step_str,
                                         !output_config.async_io);
}

const std::vector<float>* direct_surface_field_buffer(const SimulationStateView& state, std::string_view field_id)
{
    if (field_id == "surface_sensible_heat_flux")
    {
        return state.surface.sfc_sensible_heat_flux;
    }
    if (field_id == "surface_latent_heat_flux")
    {
        return state.surface.sfc_latent_heat_flux;
    }
    if (field_id == "surface_moisture_flux")
    {
        return state.surface.sfc_moisture_flux;
    }
    if (field_id == "surface_friction_velocity")
    {
        return state.surface.sfc_ustar;
    }
    if (field_id == "surface_stress_u")
    {
        return state.surface.sfc_tau_u;
    }
    if (field_id == "surface_stress_v")
    {
        return state.surface.sfc_tau_v;
    }
    if (field_id == "pbl_height")
    {
        return state.surface.pbl_height_2d;
    }
    return nullptr;
}

bool OutputPublisher::write_3d_surface_fields(WriteFrameContext& frame)
{
    if (!frame.use_3d)
    {
        return true;
    }

    return write_or_snapshot_direct_surface_fields(active_export_fields,
                                                   [&](std::string_view field_id) { return direct_surface_field_buffer(frame.state, field_id); },
                                                   validation.validation_error_message,
                                                   frame.snapshot,
                                                   frame.use_async,
                                                   frame.state.grid.nr,
                                                   frame.state.grid.nth,
                                                   frame.step_str,
                                                   !output_config.async_io);
}

bool OutputPublisher::write_3d_column_fields(WriteFrameContext&                              frame,
                                            const std::vector<RegisteredDiagnosticProduct>& tranche_registry)
{
    if (!frame.use_3d)
    {
        return true;
    }

    return write_or_snapshot_column_tranche_fields(
        tranche_registry,
        active_export_fields,
        [&](const char* field_id, std::vector<float>& values, int theta_index)
        { return validation.validate_derived_export_slice(frame.state, field_id, values, theta_index, frame.frame_export_index); },
        validation.validation_error_message,
        frame.snapshot,
        frame.use_async,
        frame.state.grid.nr,
        frame.state.grid.nth,
        frame.frame_export_index,
        frame.step_str);
}

std::string OutputPublisher::build_step_manifest_json(const WriteFrameContext& frame) const
{
    OutputStepManifestContext context;
    context.step_index                     = frame.frame_export_index;
    context.simulation_time_s              = frame.state.time_s;
    context.step_dir                       = frame.step_path.filename().string();
    context.publication_generation         = output_publication_generation;
    context.field_brick_hierarchies        = frame.use_async;
    context.storage_format                 = output_format_name(output_config.format);
    context.zfp_field_execution            = output_config.zfp_field_execution;
    context.zfp_field_workers              = output_config.zfp_resolved_field_workers;
    context.zfp_field_scratch_mib          = output_config.zfp_field_scratch_mib;
    context.data_product_kind = publishing_qualification_diagnostics
        ? (publishing_nested_child
               ? "nested_child_qualification_diagnostics"
               : "qualification_diagnostics")
        : (publishing_nested_child ? "nested_child_history" : "history");
    context.grid.nr                        = frame.state.grid.nr;
    context.grid.nth                       = frame.state.grid.nth;
    context.grid.nz                        = frame.state.grid.nz;
    context.grid.coordinate_system         = coordinate_system_name(tmv::require_run_configuration(frame.state).coordinate_system);
    context.grid.staggering                = stagger_type_name(tmv::require_run_configuration(frame.state).stagger_type);
    context.grid.dr_m                      = frame.state.grid.dr;
    context.grid.dy_m                      = frame.state.grid.dtheta;
    context.grid.dtheta_rad                = frame.state.grid.dtheta;
    context.grid.dz_m                      = frame.state.grid.dz;
    context.grid.domain_motion_mode        = frame.snapshot.grid.domain_motion_mode;
    context.grid.horizontal_frame_contract = frame.snapshot.grid.horizontal_frame_contract;
    context.grid.grid_origin_t0_earth_x_m  = frame.snapshot.grid.grid_origin_t0_earth_x_m;
    context.grid.grid_origin_t0_earth_y_m  = frame.snapshot.grid.grid_origin_t0_earth_y_m;
    context.grid.grid_origin_earth_x_m     = frame.snapshot.grid.grid_origin_earth_x_m;
    context.grid.grid_origin_earth_y_m     = frame.snapshot.grid.grid_origin_earth_y_m;
    context.grid.frame_velocity_u_ms       = frame.snapshot.grid.frame_velocity_u_ms;
    context.grid.frame_velocity_v_ms       = frame.snapshot.grid.frame_velocity_v_ms;
    context.grid.terrain_following         = frame.snapshot.grid.terrain_following;
    context.grid.terrain_geometry_sha256 =
        frame.snapshot.grid.terrain_geometry_sha256;
    context.grid.terrain_geometry_publication_generation =
        frame.snapshot.grid.terrain_geometry_publication_generation;
    if (frame.state.terrain_runtime != nullptr && frame.state.terrain_runtime->earth_fixed_placement.active)
    {
        const bool averaged=frame.state.terrain_runtime->earth_fixed_placement.sampling
            == tmv::EarthFixedTerrainSampling::CellAverageBilinear;
        context.grid.terrain_sampling=averaged ? "cell_average_bilinear_earth_v1" : "point";
        if (averaged)
            context.grid.coordinate_remap_method="eulerian_step_conservative_total_energy_remap_v1";
    }
    if (context.grid.terrain_following)
    {
        context.grid.terrain_min_z_m = frame.snapshot.grid.terrain_min_z_m;
        context.grid.terrain_max_z_m = frame.snapshot.grid.terrain_max_z_m;
    }
    context.soundings.enabled   = tmv::require_run_configuration(frame.state).sounding_enabled;
    context.soundings.scheme    = tmv::require_run_configuration(frame.state).sounding_config.scheme_id;
    context.soundings.file_path = tmv::require_run_configuration(frame.state).sounding_config.file_path;
    context.soundings.interpolation_method =
        sounding_interpolation_method_name(tmv::require_run_configuration(frame.state).sounding_config.interpolation_method);
    if ((frame.use_async || output_config.format == OutputFormat::netcdf) && output_config.format != OutputFormat::csv)
    {
        context.published_fields.reserve(frame.snapshot.fields.size());
        for (const auto& entry : frame.snapshot.fields)
        {
            context.published_fields.push_back(entry.name);
        }
    }
    else if (output_config.format != OutputFormat::csv)
    {
        // The synchronous volume and theta-slice paths have already closed all
        // payload files by the time the completion manifest is built. Capture
        // the exact visible inventory so a reader whose directory scan raced
        // publication cannot mistake a partial field list for a full frame.
        for (const auto& entry : std::filesystem::directory_iterator(frame.step_path))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }
            const std::string extension = entry.path().extension().string();
            if (extension == ".npy" || extension == ".zfp3d")
            {
                context.published_fields.push_back(entry.path().stem().string());
            }
        }
    }
    std::sort(context.published_fields.begin(), context.published_fields.end());
    context.published_fields.erase(std::unique(context.published_fields.begin(), context.published_fields.end()),
                                   context.published_fields.end());

    const auto publishes = [&](const char* field)
    {
        return std::binary_search(
            context.published_fields.begin(), context.published_fields.end(),
            std::string(field));
    };
    const auto& radar_selection =
        tmv::require_level_runtime(frame.state.radar_runtime, "radar_runtime").product_selection;
    if (publishes("reflectivity_dbz") || publishes("radial_velocity") ||
        publishes("zdr") || publishes("kdp") || publishes("rhohv"))
    {
        context.radar.enabled = true;
        if (publishes("reflectivity_dbz"))
        {
            context.radar.products.push_back("reflectivity_dbz");
            context.radar.reflectivity_operator =
                "marshall_palmer_lfo83_smith1984_fast_da_v1";
        }
        if (publishes("radial_velocity"))
        {
            context.radar.products.push_back("radial_velocity");
            context.radar.velocity_operator =
                "earth_relative_cgrid_line_of_sight_point_v1";
        }
        if (publishes("zdr"))
        {
            context.radar.products.push_back("zdr");
            context.radar.zdr_operator = radar_selection.zdr_tier ==
                    "zhang2021_rain_mp"
                ? "zhang_gao_du_2021_rain_mp_v1"
                : "exploratory_bulk_axis_ratio_polarimetric_fo_v1";
        }
        if (publishes("kdp"))
            context.radar.products.push_back("kdp");
        if (publishes("rhohv"))
            context.radar.products.push_back("rhohv");
        if (publishes("zdr") || publishes("kdp") || publishes("rhohv"))
            context.radar.polarimetric_operator =
                radar_selection.zdr_tier == "zhang2021_rain_mp"
                ? "zhang_gao_du_2021_rain_mp_v1"
                : "exploratory_bulk_axis_ratio_polarimetric_fo_v1";
        context.radar.radar_x_m = radar_selection.radar_x;
        context.radar.radar_y_m = radar_selection.radar_y;
        context.radar.radar_z_m = radar_selection.radar_z;
        context.radar.apply_scatterer_correction =
            radar_selection.apply_scatterer_correction;
    }

    if (output_config.format == OutputFormat::zfp)
    {
        auto& manifest_history = publishing_nested_child
            ? nested_child_zfp_manifest_history
            : zfp_manifest_history;
        for (const std::string& field_id : context.published_fields)
        {
            bool zfp_payload = std::filesystem::is_regular_file(frame.step_path / (field_id + ".zfp3d"));
            if (frame.use_async)
            {
                const auto entry =
                    std::find_if(frame.snapshot.fields.begin(),
                                 frame.snapshot.fields.end(),
                                 [&](const FieldSnapshotEntry& candidate) { return candidate.name == field_id; });
                if (entry == frame.snapshot.fields.end())
                {
                    throw std::runtime_error("ZFP manifest field is missing from its snapshot: " + field_id);
                }
                const auto storage = nimbus::output::classify_zfp_field_storage(*entry);
                if (storage == nimbus::output::ZfpFieldStorage::InvalidShape)
                {
                    throw std::runtime_error("ZFP manifest field has inconsistent dimensions: " + field_id);
                }
                zfp_payload = storage == nimbus::output::ZfpFieldStorage::Compressed;
            }
            if (!zfp_payload)
            {
                manifest_history.erase(field_id);
                continue;
            }

            if (output_config.zfp_reversible_radar_v4 &&
                std::find(context.radar.products.begin(),
                          context.radar.products.end(), field_id) !=
                    context.radar.products.end())
            {
                context.compression.push_back(
                    make_reversible_zfp_manifest_compression(field_id));
                manifest_history.erase(field_id);
                continue;
            }

            OutputStepManifestCompression compression;
            compression.field_id = field_id;
            compression.mode     = output_config.zfp_mode;
            if (output_config.zfp_mode == "accuracy")
            {
                compression.mode_parameter       =
                    effective_zfp_accuracy_tolerance(output_config, field_id);
                compression.mode_parameter_units = "absolute_field_units";
                compression.error_contract       = "zfp_absolute_accuracy_before_prefilters";
            }
            else if (output_config.zfp_mode == "precision")
            {
                compression.mode_parameter       = output_config.zfp_precision_bits;
                compression.mode_parameter_units = "significant_binary_bits";
                compression.error_contract       = "visualization_only_no_absolute_bound";
            }
            else
            {
                compression.mode_parameter       = output_config.zfp_rate_bps;
                compression.mode_parameter_units = "bits_per_value";
                compression.error_contract       = "visualization_only_no_absolute_bound";
            }

            compression.sparse_threshold    = is_sparse_eligible(field_id) ? output_config.zfp_sparse_threshold : 0.0;
            compression.float16_prefilter   = output_config.zfp_float16_prefilter && is_float16_eligible(field_id);
            compression.parent_step_indices = manifest_history[field_id];
            const bool scheduled_keyframe   = output_config.zfp_keyframe_interval <= 0 ||
                                              frame.frame_export_index % output_config.zfp_keyframe_interval == 0;
            if (scheduled_keyframe || compression.parent_step_indices.empty())
            {
                compression.frame_role = "keyframe";
                compression.parent_step_indices.clear();
            }
            else if (output_config.zfp_predictive_delta && compression.parent_step_indices.size() >= 2)
            {
                compression.frame_role = "predictive_delta";
            }
            else
            {
                compression.frame_role = "simple_delta";
                compression.parent_step_indices.resize(1);
            }
            auto& history = manifest_history[field_id];
            if (compression.frame_role == "keyframe")
            {
                history = {frame.frame_export_index};
            }
            else
            {
                history.insert(history.begin(), frame.frame_export_index);
                if (history.size() > 2)
                    history.resize(2);
            }
            context.compression.push_back(std::move(compression));
        }
    }
    return build_output_step_manifest_json(context);
}

bool OutputPublisher::write_step_manifest(const WriteFrameContext&     frame,
                                         const std::filesystem::path& manifest_path) const
{
    std::string error;
    const std::string manifest_json = build_step_manifest_json(frame);
    const bool published = nimbus::output::publish_manifest_after_durable_payloads(
        frame.step_path,
        [&]
        {
            return publish_output_json_atomically(
                manifest_path, manifest_json, error);
        },
        error);
    if (!published)
    {
        tmv::log_error("[EXPORT] ", error);
    }
    return published;
}

void OutputPublisher::compute_theta_azimuth_mean(std::vector<float>& theta_azimuth_mean) const
{
    return compute_theta_azimuth_mean(make_simulation_state_view(run), theta_azimuth_mean);
}

void OutputPublisher::compute_theta_azimuth_mean(const SimulationStateView& execution_state, std::vector<float>& theta_azimuth_mean) const
{
    theta_azimuth_mean.assign(static_cast<std::size_t>(execution_state.grid.nr) * static_cast<std::size_t>(execution_state.grid.nz), 0.0f);
    const PhysicalCellDomain domain = physical_scalar_domain(execution_state.grid.nr, execution_state.grid.nth, execution_state.grid.nz, tmv::require_run_configuration(execution_state).stagger_type);
    for (int i = 0; i < execution_state.grid.nr; ++i)
    {
        for (int k = 0; k < execution_state.grid.nz; ++k)
        {
            double      sum   = 0.0;
            std::size_t count = 0;
            for (int j = domain.j_begin; j < domain.j_end; ++j)
            {
                const double sample = static_cast<double>(execution_state.fields.theta[i][j][k]);
                if (std::isfinite(sample))
                {
                    sum += sample;
                    ++count;
                }
            }
            theta_azimuth_mean[static_cast<std::size_t>(i) * static_cast<std::size_t>(execution_state.grid.nz) +
                               static_cast<std::size_t>(k)] = count > 0
                                                                  ? static_cast<float>(sum / static_cast<double>(count))
                                                                  : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void OutputPublisher::compute_storm_level_means(std::vector<float>& storm_u_level_mean,
                                               std::vector<float>& storm_v_level_mean) const
{
    return compute_storm_level_means(make_simulation_state_view(run), storm_u_level_mean, storm_v_level_mean);
}

void OutputPublisher::compute_storm_level_means(const SimulationStateView& execution_state, std::vector<float>& storm_u_level_mean,
                                               std::vector<float>& storm_v_level_mean) const
{
    storm_u_level_mean.assign(static_cast<std::size_t>(execution_state.grid.nz), 0.0f);
    storm_v_level_mean.assign(static_cast<std::size_t>(execution_state.grid.nz), 0.0f);
    const PhysicalCellDomain domain = physical_scalar_domain(execution_state.grid.nr, execution_state.grid.nth, execution_state.grid.nz, tmv::require_run_configuration(execution_state).stagger_type);
    for (int k = 0; k < execution_state.grid.nz; ++k)
    {
        double      u_sum = 0.0;
        double      v_sum = 0.0;
        std::size_t count = 0;

        for (int th = domain.j_begin; th < domain.j_end; ++th)
        {
            for (int i = domain.i_begin; i < domain.i_end; ++i)
            {
                const float u_sample =
                    tmv::require_run_configuration(execution_state).stagger_type == StaggerType::CGrid
                        ? 0.5f * (static_cast<float>(execution_state.fields.u[i - 1][th][k]) + static_cast<float>(execution_state.fields.u[i][th][k]))
                        : static_cast<float>(execution_state.fields.u[i][th][k]);
                const float v_sample =
                    tmv::require_run_configuration(execution_state).stagger_type == StaggerType::CGrid
                        ? 0.5f * (static_cast<float>(execution_state.fields.v[i][th - 1][k]) + static_cast<float>(execution_state.fields.v[i][th][k]))
                        : static_cast<float>(execution_state.fields.v[i][th][k]);
                if (std::isfinite(u_sample) && std::isfinite(v_sample))
                {
                    u_sum += static_cast<double>(u_sample);
                    v_sum += static_cast<double>(v_sample);
                    ++count;
                }
            }
        }
        if (count > 0)
        {
            const double denom                              = static_cast<double>(count);
            storm_u_level_mean[static_cast<std::size_t>(k)] = static_cast<float>(u_sum / denom);
            storm_v_level_mean[static_cast<std::size_t>(k)] = static_cast<float>(v_sum / denom);
        }
        else
        {
            storm_u_level_mean[static_cast<std::size_t>(k)] = std::numeric_limits<float>::quiet_NaN();
            storm_v_level_mean[static_cast<std::size_t>(k)] = std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void OutputPublisher::prepare_derived_output_buffers(WriteFrameContext& frame)
{
    const auto vol_size = static_cast<std::size_t>(frame.state.grid.nr) * static_cast<std::size_t>(frame.state.grid.nth) * static_cast<std::size_t>(frame.state.grid.nz);
    prepare_derived_volume_output_buffers(active_export_fields, frame.use_3d, vol_size, frame.derived_3d_bufs);
    frame.need_derived_volume_slices = active_output_fields_need_derived_volume_slices(active_export_fields);
}

void OutputPublisher::write_core_slice_fields(WriteFrameContext& frame, int theta_index)
{
    if (frame.use_3d)
    {
        return;
    }

    const auto core_bindings = core_field_bindings(frame.state, frame.u_for_output, frame.v_for_output, frame.w_for_output);
    write_core_field_slices(core_bindings, active_export_fields, theta_index, frame.step_str);
}

bool OutputPublisher::write_derived_slice_outputs(WriteFrameContext&         frame,
                                                 const DerivedSliceBinding* bindings,
                                                 std::size_t                binding_count,
                                                 int                        theta_index)
{
    return write_or_scatter_derived_slice_fields(
        bindings,
        binding_count,
        active_export_fields,
        frame.derived_3d_bufs,
        [&](const char* field_id, std::vector<float>& values)
        { return validation.validate_derived_export_slice(frame.state, field_id, values, theta_index, frame.frame_export_index); },
        frame.use_3d,
        theta_index,
        frame.state.grid.nr,
        frame.state.grid.nth,
        frame.state.grid.nz,
        frame.step_str);
}

bool OutputPublisher::write_tranche_slice_outputs(WriteFrameContext&                              frame,
                                                 const std::vector<RegisteredDiagnosticProduct>& tranche_registry,
                                                 int                                             theta_index,
                                                 std::vector<float>&                             tranche_slice_buffer)
{
    return write_tranche_slice_fields(
        tranche_registry,
        active_export_fields,
        [&](const char* field_id, std::vector<float>& values, int validation_theta_index)
        { return validation.validate_derived_export_slice(frame.state, field_id, values, validation_theta_index, frame.frame_export_index); },
        validation.validation_error_message,
        frame.use_3d,
        theta_index,
        frame.frame_export_index,
        frame.state.grid.nr,
        frame.state.grid.nz,
        frame.step_str,
        tranche_slice_buffer);
}

bool OutputPublisher::fail_qualified_zfp_output_contract()
{
    if (global_execution_mode != tmv::init::config::ExecutionMode::Qualified ||
        output_config.format != OutputFormat::zfp)
    {
        return true;
    }
    std::string error;
    if (!publish_output_json_atomically(
            std::filesystem::path(outdir) / kNimbusRunOutputContractFilename,
            build_output_run_contract_json(output_publication_generation, "failed",
                                           run.configuration.nesting.enabled,
                                           qualification_diagnostics_enabled(),
                                           write_every_s > 0),
            error))
    {
        tmv::log_error("[OUTPUT] failed to invalidate qualified ZFP run contract: ", error);
        return false;
    }
    return true;
}

bool OutputPublisher::finish_write_frame(WriteFrameContext& frame)
{
    if (frame.use_async)
    {
        const double assembly_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - frame.assembly_started).count();
        const std::size_t assembly_bytes = frame.snapshot.total_bytes();
        snapshot_assembly_total_s += assembly_s;
        snapshot_assembly_max_s = std::max(snapshot_assembly_max_s, assembly_s);
        snapshot_assembly_max_bytes = std::max(snapshot_assembly_max_bytes, assembly_bytes);
        snapshot_assembly_max_owned_bytes = std::max(
            snapshot_assembly_max_owned_bytes, frame.snapshot.owned_bytes());
        ++snapshot_assembly_frames;
        // Serialize manifest and submit snapshot for background writing.
        frame.snapshot.publication_generation = output_publication_generation;
        frame.snapshot.manifest_json = build_step_manifest_json(frame);
        if (!frame.writer.submit(std::move(frame.snapshot)))
        {
            tmv::log_error("[EXPORT] Async writer error: ", frame.writer.error_message());
            fail_qualified_zfp_output_contract();
            return false;
        }
        // A qualified ZFP frame is a science publication boundary: its
        // worker must acknowledge the commit before another solver step can
        // append ledgers or checkpoints.
        if (global_execution_mode == tmv::init::config::ExecutionMode::Qualified &&
            output_config.format == OutputFormat::zfp &&
            !frame.writer.wait_until_ready())
        {
            tmv::log_error("[EXPORT] Qualified ZFP writer error before science continuation: ",
                           frame.writer.error_message());
            fail_qualified_zfp_output_contract();
            return false;
        }
    }
    else
    {
        const std::filesystem::path manifest_path = frame.step_path / "manifest.json";
        if (!write_step_manifest(frame, manifest_path))
        {
            tmv::log_error("[EXPORT] failed to write step manifest: ", manifest_path);
            return false;
        }
    }
    log_memory_milestone("output_submission_complete");
    return true;
}

bool OutputPublisher::write_all_fields(int frame_export_index)
{
    return write_all_fields(make_simulation_state_view(run), frame_export_index);
}

bool OutputPublisher::write_all_fields(const SimulationStateView& execution_state, int frame_export_index)
{
    if (output_config.async_io &&
        (output_config.format == OutputFormat::npy_3d || output_config.format == OutputFormat::netcdf ||
         output_config.format == OutputFormat::csv || output_config.format == OutputFormat::zfp) &&
        !async_writer->wait_until_ready())
    {
        tmv::log_error("[EXPORT] Async writer error before frame assembly: ", async_writer->error_message());
        fail_qualified_zfp_output_contract();
        return false;
    }
    if (!enforce_output_retention())
        return false;
    log_memory_milestone("output_assembly_begin");
    refresh_output_diagnostics(execution_state, output_dependencies_for_fields(active_export_fields));
    WriteFrameContext frame(execution_state, frame_export_index, *async_writer);
    if (!prepare_write_frame(frame))
    {
        return false;
    }

    // Core/direct-surface frames do not need the 39 derived slice arrays,
    // azimuthal/storm means, or the diagnostic tranche registry. Keep this
    // path allocation-free apart from the selected output/snapshot buffers.
    if (!active_output_fields_need_derived_slices(active_export_fields))
    {
        prepare_core_output_fields(frame);
        write_3d_core_fields(frame);
        if (!write_3d_surface_fields(frame))
        {
            return false;
        }
        for (int th = 0; th < execution_state.grid.nth; ++th)
        {
            write_core_slice_fields(frame, th);
        }
        return finish_write_frame(frame);
    }

    const std::size_t slice_size           = frame.slice_size;
    const float       reflectivity_dbz_min = frame.reflectivity_dbz_min;
    const float       reflectivity_dbz_max = frame.reflectivity_dbz_max;

    const bool need_derived_volume_slices = active_output_fields_need_derived_volume_slices(active_export_fields);
    const bool need_tranche_products      = active_output_fields_need_tranche_products(active_export_fields);
    std::optional<DerivedSliceScratch> derived_slices;
    std::vector<DerivedSliceBinding>   derived_bindings;
    if (need_derived_volume_slices)
    {
        derived_slices.emplace(slice_size, active_export_fields);
        if (!build_derived_slice_bindings(*derived_slices, derived_bindings, validation.validation_error_message))
        {
            return false;
        }
    }

    // Derived diagnostics consume publication-grid velocities. On the C-grid
    // this stages face winds to scalar centres before any column products or
    // storm-motion estimates are evaluated.
    prepare_core_output_fields(frame);

    std::vector<float> theta_azimuth_mean;
    if (derived_slices.has_value() &&
        (derived_slices->has_buffer("theta_prime") || derived_slices->has_buffer("buoyancy")))
    {
        compute_theta_azimuth_mean(execution_state, theta_azimuth_mean);
    }

    std::vector<float> storm_u_level_mean;
    std::vector<float> storm_v_level_mean;
    if (need_tranche_products || (derived_slices.has_value() && derived_slices->has_buffer("storm_relative_winds")))
    {
        compute_storm_level_means(execution_state, storm_u_level_mean, storm_v_level_mean);
    }

    const auto write_with_tranche_registry =
        [&](const std::vector<RegisteredDiagnosticProduct>& tranche_registry) -> bool
    {
        // --- 3D mode: write core fields as full volumes directly from Field3D ---
        write_3d_core_fields(frame);

        // --- 3D mode: write 2D surface-flux fields harvested from the PBL scheme ---
        // These are dimensioned [NR][NTH] (one value per surface column).
        // Populated by src/boundary_layer/runtime.cpp from
        // the active BoundaryLayerScheme's BoundaryLayerDiagnostics. Zero when
        // no PBL scheme is active or boundary_layer == "none".
        if (!write_3d_surface_fields(frame))
        {
            return false;
        }

        // --- 3D mode: write 2D COLUMN/SURFACE tranches as (NR, NTH) maps ---
        // npy_3d scatters the 3D-derived fields into volumes and writes the
        // surface 2D fields above, but the column tranche diagnostics
        // (composite_reflectivity, srh, cape, vil, updraft_helicity, ...) are
        // inherently 2D and were previously emitted only in the per-theta
        // npy_2d_slices path. Studio reads step_N/<field>.npy, so without this
        // they are never visible. Each column tranche's compute_slice broadcasts
        // the per-(i, th) value across k, so column_scratch[i] == value(i, th);
        // assemble the (NR, NTH) map and write it like the surface fields above.
        // Surface tranches are assembled by the same planar path. Their state
        // updates occur in the owning physics/runtime step, not during export.
        if (!write_3d_column_fields(frame, tranche_registry))
        {
            return false;
        }

        prepare_derived_output_buffers(frame);

        for (int th = 0; th < execution_state.grid.nth; ++th)
        {
            write_core_slice_fields(frame, th);

            if (frame.need_derived_volume_slices)
            {
                const DerivedSliceComputationContext derived_context{
                    execution_state.grid.nr,
                    execution_state.grid.nth,
                    execution_state.grid.nz,
                    execution_state.grid.dr,
                    execution_state.grid.dtheta,
                    execution_state.grid.dz,
                    execution_state.grid.dt,
                    tmv::require_run_configuration(execution_state).coordinate_system,
                    output_dependencies_for_fields(active_export_fields),
                    th,
                    reflectivity_dbz_min,
                    reflectivity_dbz_max,
                    theta_azimuth_mean,
                    storm_u_level_mean,
                    storm_v_level_mean,
                    execution_state.fields.rho,
                    execution_state.fields.p,
                    *frame.u_for_output,
                    *frame.v_for_output,
                    *frame.w_for_output,
                    execution_state.fields.theta,
                    execution_state.fields.qv,
                    execution_state.fields.qc,
                    execution_state.fields.qr,
                    execution_state.fields.qi,
                    execution_state.fields.qs,
                    execution_state.fields.qg,
                    execution_state.fields.qh,
                    tmv::require_level_runtime(execution_state.diagnostics, "diagnostics").radar_reflectivity,
                    tmv::require_level_runtime(execution_state.radar_runtime, "radar_runtime").selected_output.Vr,
                    tmv::require_level_runtime(execution_state.radar_runtime, "radar_runtime").selected_output.ZDR_dB,
                    tmv::require_level_runtime(execution_state.radar_runtime, "radar_runtime").selected_output.KDP_deg_km,
                    tmv::require_level_runtime(execution_state.radar_runtime, "radar_runtime").selected_output.rhohv,
                    tmv::require_level_runtime(execution_state.diagnostics, "diagnostics").vorticity_r,
                    tmv::require_level_runtime(execution_state.diagnostics, "diagnostics").vorticity_theta,
                    tmv::require_level_runtime(execution_state.diagnostics, "diagnostics").vorticity_z,
                    tmv::require_level_runtime(execution_state.diagnostics, "diagnostics").angular_momentum,
                    execution_state.base.theta0,
                    tmv::physical_grid_metrics::active_terrain_metrics(
                        execution_state),
                    execution_state.coriolis_f,
                    &execution_state.base.rho0,
                    &execution_state.base.qv0,
                    &execution_state.base.p0,
                    &execution_state.grid.geometry.z,
                };
                compute_derived_slices_for_theta(*derived_slices, derived_context);

                if (!write_derived_slice_outputs(frame, derived_bindings.data(), derived_bindings.size(), th))
                {
                    return false;
                }
            }

            if (!write_tranche_slice_outputs(frame, tranche_registry, th, frame.tranche_slice_buffer))
            {
                return false;
            }
        }

        if (frame.use_3d)
        {
            write_or_snapshot_derived_volume_fields(frame.derived_3d_bufs,
                                                    frame.snapshot,
                                                    frame.use_async,
                                                    output_config.format,
                                                    execution_state.grid.nr,
                                                    execution_state.grid.nth,
                                                    execution_state.grid.nz,
                                                    frame.step_str);
        }
        return finish_write_frame(frame);
    };

    if (!need_tranche_products)
    {
        static const std::vector<RegisteredDiagnosticProduct> empty_registry;
        return write_with_tranche_registry(empty_registry);
    }

    tmv::diagnostics::TrancheProductRegistryContext tranche_context{
        frame.slice_size,
        frame.reflectivity_dbz_min,
        frame.reflectivity_dbz_max,
        tmv::require_level_runtime(execution_state.surface.precipitation_rate_mm_h, "surface.precipitation_rate_mm_h"),
        tmv::require_level_runtime(execution_state.surface.accumulated_rainfall_mm, "surface.accumulated_rainfall_mm"),
        tmv::require_level_runtime(execution_state.surface.accumulated_rainfall_residual_mm, "surface.accumulated_rainfall_residual_mm"),
        tmv::require_level_runtime(execution_state.surface.accumulated_rainfall_last_update_s, "surface.accumulated_rainfall_last_update_s"),
        storm_u_level_mean,
        storm_v_level_mean,
        active_export_fields,
        execution_state,
        physical_scalar_domain(execution_state.grid.nr, execution_state.grid.nth, execution_state.grid.nz, tmv::require_run_configuration(execution_state).stagger_type),
        *frame.u_for_output,
        *frame.v_for_output,
        *frame.w_for_output,
    };
    return tmv::diagnostics::with_diagnostic_tranche_registry(
        tranche_context, validation.validation_error_message, write_with_tranche_registry);
}
#endif

bool OutputPublisher::fail_terminal_run_contract() noexcept
{
    if (!run_contract_ready)
    {
        return true;
    }
    const auto log_secondary_failure = [](const char* reason) noexcept
    {
        try
        {
            tmv::log_error("[OUTPUT] failed to mark terminal run failure: ", reason);
        }
        catch (...)
        {
        }
    };
    try
    {
        std::string error;
        if (!publish_output_json_atomically(
                std::filesystem::path(outdir) / kNimbusRunOutputContractFilename,
                build_output_run_contract_json(output_publication_generation, "failed",
                                               run.configuration.nesting.enabled,
                                               qualification_diagnostics_enabled(),
                                               write_every_s > 0),
                error))
        {
            log_secondary_failure(error.c_str());
            return false;
        }
        run_contract_ready = false;
        return true;
    }
    catch (const std::exception& failure)
    {
        log_secondary_failure(failure.what());
        return false;
    }
    catch (...)
    {
        log_secondary_failure("unknown secondary exception");
        return false;
    }
}

int HeadlessDriver::setup()
{
    output.output_publication_generation = make_output_publication_generation();
    if (!options.resume_checkpoint.empty() && path_is_within(options.resume_checkpoint, outdir))
    {
        tmv::log_error("[CHECKPOINT] branch output directory contains the source checkpoint; choose a new --outdir so "
                       "the parent run cannot be overwritten");
        return 1;
    }
    // Resolve field selection, compression tiers, and estimate disk budget
    resolve_output_fields(output.output_config);
    const auto& radar_selection =
        tmv::global_radar_runtime_state.product_selection;
    std::vector<std::string> selected_radar_products;
    if (radar_selection.reflectivity)
        selected_radar_products.push_back("reflectivity_dbz");
    if (radar_selection.radial_velocity)
        selected_radar_products.push_back("radial_velocity");
    if (radar_selection.zdr)
        selected_radar_products.push_back("zdr");
    if (radar_selection.kdp)
        selected_radar_products.push_back("kdp");
    if (radar_selection.rhohv)
        selected_radar_products.push_back("rhohv");
    merge_selected_radar_output_fields(output.output_config, selected_radar_products);
    output.output_config.zfp_reversible_radar_v4 =
        global_execution_mode == tmv::init::config::ExecutionMode::Qualified &&
        output.output_config.format == OutputFormat::zfp &&
        !selected_radar_products.empty();
    if (output.output_config.zfp_reversible_radar_v4)
    {
        nimbus::output::QualifiedRadarZfpV4Context context;
        context.qualified = true;
        context.cpu = global_simulation_storage.runtime.compute.active_kind ==
            ComputeBackendKind::Cpu;
        context.fallback_off = !global_compute_backend_config.allow_fallback;
        context.cartesian = global_coordinate_system == CoordinateSystem::Cartesian;
        context.c_grid = global_stagger_type == StaggerType::CGrid;
        context.flat = !tmv::physical_grid_metrics::terrain_is_active(
            global_simulation_state());
        context.stationary = global_domain_motion_mode ==
                tmv::init::config::DomainMotionMode::Stationary &&
            global_domain_frame.velocity_ms.u_ms == 0.0 &&
            global_domain_frame.velocity_ms.v_ms == 0.0;
        context.no_nesting = !global_simulation_storage.configuration.nesting.enabled;
        context.no_tracking = !global_storm_tracking_config.enabled &&
            !global_storm_tracking_config.recycle_domain;
        context.no_side_stream = !output.qualification_diagnostics_enabled();
        context.no_tiered_cadence = !output.output_config.tiered_write_cadence;
        context.no_live_shm = !options.live_shm;
        context.no_trajectories = !options.trajectory.enabled &&
            !output.output_config.trajectories;
        context.deterministic_reductions =
            tmv::diagnostics::deterministic_reductions_enabled();
        context.reviewed_source_profile =
            global_simulation_storage.configuration.admission.radar;
        context.selected_radar_products = selected_radar_products;
        context.format_tier = output.output_config.zfp_requested_format_tier;
        context.nx = global_simulation_storage.grid.nr;
        context.ny = global_simulation_storage.grid.nth;
        context.nz = global_simulation_storage.grid.nz;
        context.dx_m = global_simulation_storage.grid.dx;
        context.dy_m = global_simulation_storage.grid.dy;
        context.dz_m = global_simulation_storage.grid.dz;
        context.dt_large_s =
            global_simulation_storage.configuration.configured_dt_large_s;
        context.acoustic_substeps = global_time_stepping_config.n_acoustic_substeps;
        context.write_every_s = write_every_s;
        const auto errors = qualified_radar_zfp_v4_candidate_errors(
            output.output_config, context);
        if (!errors.empty())
        {
            tmv::log_error("[OUTPUT] ", errors.front());
            return 1;
        }
    }
    if (output.output_config.zfp_reversible_radar_v4 &&
        (output.output_config.preset != FieldPreset::radar_ready &&
         output.output_config.preset != FieldPreset::radar_polarimetric_ready &&
         output.output_config.preset != FieldPreset::minimal))
    {
        tmv::log_error("[OUTPUT] reversible radar ZFP requires a retained ready radar preset");
        return 1;
    }
    if (output.output_config.zfp_reversible_radar_v4 &&
        (global_coordinate_system != CoordinateSystem::Cartesian ||
         tmv::physical_grid_metrics::terrain_is_active(global_simulation_state()) ||
         global_simulation_storage.configuration.nesting.enabled ||
         global_simulation_storage.runtime.compute.active_kind != ComputeBackendKind::Cpu ||
         output.output_config.tiered_write_cadence ||
         output.qualification_diagnostics_enabled()))
    {
        tmv::log_error("[OUTPUT] reversible radar ZFP candidate requires flat CPU untiered ordinary history");
        return 1;
    }
    if (!radar_selection.reflectivity && !radar_selection.radial_velocity &&
        !radar_selection.zdr && !radar_selection.kdp &&
        !radar_selection.rhohv)
    {
        for (const auto& field : output_field_registry())
        {
            if (output_field_has_dependency(
                    field.dependencies,
                    OutputFieldDependency::RadarDiagnostic))
                output.output_config.resolved_fields.erase(field.field_id);
        }
    }
    apply_default_compression_tiers(output.output_config);
    if (output.qualification_diagnostics_enabled())
    {
        resolve_output_fields(output.qualification_output_config);
        const std::unordered_set<std::string> expected = {
            "cold_pool_boundary", "accumulated_rainfall"};
        const auto& nesting =
            global_simulation_storage.configuration.nesting;
        const int level_time_ratio = nesting.enabled
            ? nesting.parent_time_step_ratio
            : 1;
        const double effective_dt = level_time_ratio > 0
            ? global_simulation_storage.configuration.configured_dt_large_s /
                  static_cast<double>(level_time_ratio)
            : 0.0;
        const double interval_s = static_cast<double>(
            output.output_config.qualification_diagnostics_interval_s);
        const bool root_cadence_reachable =
            qualification_diagnostics_cadence_reachable(
                interval_s,
                global_simulation_storage.configuration.configured_dt_large_s);
        const bool comparison_cadence_reachable =
            qualification_diagnostics_cadence_reachable(
                interval_s, effective_dt);
        if (output.qualification_output_config.resolved_fields != expected ||
            output.qualification_output_config.format != OutputFormat::npy_3d ||
            output.qualification_output_config.async_io ||
            global_coordinate_system != CoordinateSystem::Cartesian ||
            tmv::physical_grid_metrics::terrain_is_active(
                global_simulation_state()) ||
            !global_storm_tracking_config.enabled ||
            output.output_config.qualification_diagnostics_interval_s <= 0 ||
            global_simulation_storage.runtime.time_stepping.config.adaptive_dt ||
            !root_cadence_reachable ||
            !comparison_cadence_reachable ||
            global_storm_tracking_config.interval_s !=
                static_cast<double>(
                    output.output_config.qualification_diagnostics_interval_s) ||
            duration_s <= 0 ||
            duration_s % output.output_config.qualification_diagnostics_interval_s != 0)
        {
            tmv::log_error(
                "[OUTPUT] moving_reference_v1 qualification diagnostics lost "
                "their exact flat-Cartesian, tracker-aligned, lossless "
                "two-planar-field cadence contract");
            return 1;
        }
    }
    tmv::log_info("[CONFIG] coordinate_system=",
                  coordinate_system_name(global_coordinate_system),
                  ", staggering=",
                  stagger_type_name(global_stagger_type));
    tmv::log_info("[OUTPUT] Format: ",
                  output_format_name(output.output_config.format),
                  ", fields: ",
                  field_preset_name(output.output_config.preset),
                  " (",
                  output.output_config.resolved_fields.size(),
                  " fields)",
                  ", async_io: ",
                  output.output_config.async_io ? "on" : "off");
    if (output.output_config.keep_last_frames > 0)
    {
        tmv::log_info("[OUTPUT] Retention: newest ",
                      output.output_config.keep_last_frames,
                      " completed frame(s) per series plus compression dependencies");
    }
    if (output.output_config.minimum_free_space_mib > 0)
    {
        tmv::log_info("[OUTPUT] Free-space reserve: ", output.output_config.minimum_free_space_mib, " MiB");
    }
    if (output.output_config.maximum_retained_history_files > 0)
    {
        tmv::log_info("[OUTPUT] Retained file ceiling: ",
                      output.output_config.maximum_retained_history_files,
                      " per series; conservative estimate=",
                      output.output_config.estimated_retained_history_files);
    }
    if (options.checkpoint_keep_last > 0)
    {
        tmv::log_info("[CHECKPOINT] Retention: newest ", options.checkpoint_keep_last, " complete checkpoint(s)");
    }
    if (options.checkpoint_minimum_free_space_mib > 0)
    {
        tmv::log_info("[CHECKPOINT] Free-space reserve: ",
                      options.checkpoint_minimum_free_space_mib,
                      " MiB");
    }
    if (output.output_config.tiered_write_cadence)
    {
        tmv::log_info("[OUTPUT] Tiered write cadence: fast=",
                      write_every_s,
                      "s, medium=",
                      output.output_config.write_cadence_medium_s,
                      "s, slow=",
                      output.output_config.write_cadence_slow_s,
                      "s");
    }
    if (output.qualification_diagnostics_enabled())
    {
        tmv::log_info(
            "[OUTPUT] Qualification diagnostics: preset=",
            qualification_diagnostics_preset_name(
                output.output_config.qualification_diagnostics_preset),
            ", interval_s=",
            output.output_config.qualification_diagnostics_interval_s,
            ", format=npy_3d, fields=2");
    }

    // Exact-binary configuration truth: an unavailable requested codec is a
    // launch error, never permission to publish a different storage format.
    if (output.output_config.format == OutputFormat::zfp)
    {
#ifndef HAVE_ZFP
        tmv::log_error("[OUTPUT] ZFP compression requested but binary was built "
                       "without ZFP support (rebuild with ZFP=1 or select npy_3d/netcdf).");
        return 1;
#endif
    }

    estimate_disk_budget(output.output_config, NR, NTH, NZ, duration_s, write_every_s);
    if (output.qualification_diagnostics_enabled() &&
        global_simulation_storage.configuration.nesting.enabled)
    {
        const auto& nesting =
            global_simulation_storage.configuration.nesting;
        estimate_qualification_diagnostics_disk_budget(
            output.output_config,
            nesting.child_cells_x + 2,
            nesting.child_cells_y + 2,
            duration_s);
    }

    if (output.output_config.format == OutputFormat::zfp &&
        output.output_config.zfp_field_execution == "field_parallel")
    {
        const unsigned hardware_threads =
            std::max(1U, std::thread::hardware_concurrency());
        const int solver_threads = std::max(1, omp_get_max_threads());
        const bool accelerator =
            global_simulation_storage.runtime.compute.active_kind !=
            ComputeBackendKind::Cpu;
        const int resolved_workers = resolve_zfp_field_workers(
            output.output_config.zfp_field_workers,
            output.output_config.zfp_field_scratch_mib,
            std::max(1, static_cast<int>(output.output_config.resolved_fields.size())),
            accelerator, solver_threads,
            static_cast<int>(hardware_threads));
        if (resolved_workers < 1)
        {
            tmv::log_error(
                "[OUTPUT] ZFP field-parallel worker request exceeds its field/scratch budget");
            return 1;
        }
        output.output_config.zfp_resolved_field_workers = resolved_workers;
        tmv::log_info(
            "[OUTPUT] ZFP field execution=field_parallel, workers=",
            output.output_config.zfp_resolved_field_workers,
            ", scratch_budget_mib=",
            output.output_config.zfp_field_scratch_mib,
            ", backend=", accelerator ? "accelerator" : "cpu",
            ", solver_threads=", solver_threads,
            ", hardware_threads=", hardware_threads);
    }

    // Do not create an idle writer thread for an output-disabled run. The old
    // unconditional construction perturbed the CPU solver benchmark even
    // though no snapshot could ever be submitted.
    if (write_every_s > 0 || output.qualification_diagnostics_enabled())
    {
        // Async mode serializes snapshots on a background thread; sync mode
        // writes directly on the caller without the snapshot-copy queue.
        output.async_writer.emplace(output.output_config);
    }

    // Live shared-memory transport for zero-I/O visualization
    if (options.live_shm)
    {
        std::string live_field_error;
        const bool valid_live_fields =
            validate_live_shm_fields(options.live_shm_fields,
                                     live_field_error) &&
            tmv_shm::valid_logical_shm_name(options.live_shm_name) &&
            std::isfinite(options.live_shm_interval_s) &&
            options.live_shm_interval_s > 0.0;

        // Viewer expects [Z][TH][X] layout; ShmWriter handles transposition.
        // Dimensions passed as viewer-space (X=NR, Y=NTH, Z=NZ).
        if (!valid_live_fields)
        {
            tmv::log_error(
                "[SHM] live transport configuration is invalid; refusing to start: ",
                live_field_error.empty()
                    ? "endpoint name or publish interval is invalid"
                    : live_field_error);
            return 1;
        }
        else
        {
            const PhysicalCellDomain         domain = physical_scalar_domain(NR, NTH, NZ, global_stagger_type);
            const tmv_shm::ShmPhysicalDomain live_domain{
                {domain.i_begin, domain.j_begin, domain.k_begin},
                {domain.i_end - 1, domain.j_end - 1, domain.k_end - 1},
            };
            const tmv_shm::ShmGridSpacing live_spacing{dr, dtheta, dz};
            if (!output.shm_writer.open(
                    NR, NTH, NZ, options.live_shm_fields, live_domain,
                    live_spacing, output.live_shm_frame(),
                    options.live_shm_name.c_str(),
                    output.output_publication_generation))
            {
                tmv::log_error("[SHM] live transport was explicitly requested but could not be initialized");
                return 1;
            }
        }
    }

    if (write_every_s > 0 || output.qualification_diagnostics_enabled() ||
        options.trajectory.enabled || options.live_shm ||
        options.checkpoint_every_s > 0 ||
        !options.resume_checkpoint.empty() || global_storm_tracking_config.enabled)
    {
        std::error_code ec;
        std::filesystem::create_directories(outdir, ec);
        if (ec)
        {
            tmv::log_error("[OUTPUT] failed to create output directory '", outdir, "': ", ec.message());
            return 1;
        }
        for (const char* artifact : {"nimbus_termination.json",
                                     "nimbus_termination.json.tmp",
                                     "nimbus_live_shm.json",
                                     "nimbus_live_shm.json.tmp"})
        {
            std::filesystem::remove(std::filesystem::path(outdir) / artifact, ec);
            if (ec)
            {
                tmv::log_error("[RUNTIME] failed to remove stale termination record: ", ec.message());
                return 1;
            }
        }
    }

    if (global_storm_tracking_config.enabled)
    {
        const std::filesystem::path tracking_directory = std::filesystem::path(outdir) / "storm_track";
        std::error_code             ec;
        if (options.resume_checkpoint.empty())
        {
            std::filesystem::remove_all(tracking_directory, ec);
            if (ec)
            {
                tmv::log_error("[STORM TRACK] failed to clear stale diagnostic directory '",
                               tracking_directory.string(),
                               "': ",
                               ec.message());
                return 1;
            }
            std::filesystem::create_directories(tracking_directory, ec);
            if (ec)
            {
                tmv::log_error("[STORM TRACK] failed to create diagnostic directory '",
                               tracking_directory.string(),
                               "': ",
                               ec.message());
                return 1;
            }
            std::filesystem::remove(std::filesystem::path(outdir) / "storm_track_latest.json", ec);
            if (ec)
            {
                tmv::log_error("[STORM TRACK] failed to clear stale latest diagnostic: ", ec.message());
                return 1;
            }
        }
        std::filesystem::remove(
            std::filesystem::path(outdir) /
                "moving_nest_scientific_validity.json",
            ec);
        if (ec)
        {
            tmv::log_error(
                "[STORM TRACK] failed to clear stale moving-nest scientific-validity artifact: ",
                ec.message());
            return 1;
        }
    }

    // A trajectory-only run has no run marker/step cleanup block below. Clear
    // its run-scoped artifacts up front so a no-seed or failed-new-data run
    // cannot leave a prior run's trajectories looking current.
    if (options.trajectory.enabled && write_every_s <= 0)
    {
        static constexpr const char* trajectory_artifacts[] = {
            "trajectories.npy",
            "trajectories.npy.tmp",
            "trajectories.json",
        };
        for (const char* artifact : trajectory_artifacts)
        {
            const std::filesystem::path artifact_path = std::filesystem::path(outdir) / artifact;
            std::error_code             ec;
            std::filesystem::remove(artifact_path, ec);
            if (ec)
            {
                tmv::log_error(
                    "[TRAJECTORIES] failed to remove stale artifact '", artifact_path.string(), "': ", ec.message());
                return 1;
            }
        }
    }

    if (write_every_s > 0 || output.qualification_diagnostics_enabled())
    {
        // Publish a fresh generation in the initializing state before touching
        // old step directories. This is the atomic invalidation barrier: a
        // concurrent Studio scan cannot accept old manifests under the new run,
        // even while cleanup is in progress. Only publish `ready` after every
        // prior step directory has been removed successfully.
        const std::filesystem::path contract_path = std::filesystem::path(outdir) / kNimbusRunOutputContractFilename;
        std::string                 contract_error;
        if (!publish_output_json_atomically(
                contract_path,
                build_output_run_contract_json(output.output_publication_generation,
                                               "initializing",
                                               global_simulation_storage.configuration.nesting.enabled,
                                               output.qualification_diagnostics_enabled(),
                                               write_every_s > 0),
                contract_error))
        {
            tmv::log_error("[OUTPUT] ", contract_error);
            return 1;
        }

        static const std::regex            step_pattern(R"(^step_[0-9]{6}$)");
        std::vector<std::filesystem::path> stale_step_directories;
        for (const auto& entry : std::filesystem::directory_iterator(outdir))
        {
            if (!entry.is_directory())
            {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if (std::regex_match(name, step_pattern))
            {
                stale_step_directories.push_back(entry.path());
            }
        }
        // Do not mutate the directory while its iterator is live: some
        // standard-library/filesystem combinations may otherwise skip entries.
        for (const auto& step_path : stale_step_directories)
        {
            std::error_code ec;
            std::filesystem::remove_all(step_path, ec);
            if (ec)
            {
                tmv::log_error(
                    "[OUTPUT] failed to remove stale step directory '", step_path.string(), "': ", ec.message());
                return 1;
            }
        }

        {
            const std::filesystem::path stale_qualification =
                std::filesystem::path(outdir) /
                "qualification_diagnostics";
            std::error_code ec;
            std::filesystem::remove_all(stale_qualification, ec);
            if (ec)
            {
                tmv::log_error(
                    "[OUTPUT] failed to remove stale qualification diagnostics '",
                    stale_qualification.string(), "': ", ec.message());
                return 1;
            }
        }

        // Nested histories are run-scoped just like root step directories.
        // Remove the entire prior hierarchy while the new generation remains
        // unavailable, so a reader cannot combine children from two runs.
        {
            const std::filesystem::path stale_nests = std::filesystem::path(outdir) / "nests";
            std::error_code             ec;
            std::filesystem::remove_all(stale_nests, ec);
            if (ec)
            {
                tmv::log_error(
                    "[OUTPUT] failed to remove stale nested history '", stale_nests.string(), "': ", ec.message());
                return 1;
            }
        }

        // Run-scoped products are not contained by step directories. Remove
        // every known product while the new generation is still
        // `initializing`; otherwise a same-directory restart without the
        // corresponding feature enabled could advertise data from the prior
        // run. Budget files are recreated below and trajectories are published
        // atomically at finalization.
        static constexpr const char* stale_run_artifacts[] = {
            "history_index.json",
            "history_index.json.tmp",
            "history_index.pending.json",
            "history_index.pending.json.tmp",
            "budget.csv",
            "momentum_budget.csv",
            "energy_budget.csv",
            "radiation_budget.csv",
            "thermodynamic_source_budget.csv",
            "moisture_budget.csv",
            "validation_summary.json",
            "trajectories.npy",
            "trajectories.npy.tmp",
            "trajectories.json",
        };
        for (const char* artifact : stale_run_artifacts)
        {
            const std::filesystem::path artifact_path = std::filesystem::path(outdir) / artifact;
            std::error_code             ec;
            std::filesystem::remove(artifact_path, ec);
            if (ec)
            {
                tmv::log_error(
                    "[OUTPUT] failed to remove stale run artifact '", artifact_path.string(), "': ", ec.message());
                return 1;
            }
        }
        {
            const std::filesystem::path stale_index_segments =
                std::filesystem::path(outdir) / "history_index";
            std::error_code ec;
            std::filesystem::remove_all(stale_index_segments, ec);
            if (ec)
            {
                tmv::log_error("[OUTPUT] failed to remove stale history index segments: ", ec.message());
                return 1;
            }
        }

        if (!publish_output_json_atomically(
                contract_path,
                build_output_run_contract_json(output.output_publication_generation,
                                               "ready",
                                               global_simulation_storage.configuration.nesting.enabled,
                                               output.qualification_diagnostics_enabled(),
                                               write_every_s > 0),
                contract_error))
        {
            tmv::log_error("[OUTPUT] ", contract_error);
            return 1;
        }
        output.run_contract_ready = true;

        // Open the conservation-budget CSV time series in the run output
        // directory. report_budget_transition will append one row per
        // physics stage per macro step. close_budget_csv() runs at end.
        open_budget_csv((std::filesystem::path(outdir) / "budget.csv").string());

        // F.5b momentum/energy/moisture budgets: one total_step closure row
        // per macro step. Closed alongside the mass budget at shutdown.
        open_momentum_csv((std::filesystem::path(outdir) / "momentum_budget.csv").string());
        open_energy_csv((std::filesystem::path(outdir) / "energy_budget.csv").string());
        if (global_terrain_config.horizontal_placement_id == "earth_fixed"
            && (global_domain_frame.velocity_ms.u_ms != 0.0 || global_domain_frame.velocity_ms.v_ms != 0.0))
            tmv::open_earth_fixed_remap_csv((std::filesystem::path(outdir) / "earth_fixed_remap.csv").string());
        open_radiation_budget_csv((std::filesystem::path(outdir) / "radiation_budget.csv").string());
        open_thermodynamic_source_budget_csv(
            (std::filesystem::path(outdir) / "thermodynamic_source_budget.csv").string());
        open_moisture_csv((std::filesystem::path(outdir) / "moisture_budget.csv").string());
    }

    output.lastGuiExport     = std::chrono::steady_clock::now();
    ::simulation_time = 0.0;
    return 0;
}

int HeadlessDriver::restore_resume_checkpoint()
{
    if (options.resume_checkpoint.empty())
        return 0;

    CheckpointSidecar sidecar;
    std::string       error;
    if (!load_checkpoint_sidecar(options.resume_checkpoint, sidecar, error))
    {
        tmv::log_error("[CHECKPOINT] ", error);
        return 1;
    }

    SimulationStorageState restored;
    tmv::checkpoint::SimulationCheckpointAttachments attachments;
    tmv::checkpoint::SimulationCheckpointIntegrity checkpoint_integrity;
    const auto restore_started = global_perf_timing_enabled ? PerfClock::now() : PerfClock::time_point{};
    if (!tmv::checkpoint::read_simulation_checkpoint(
            options.resume_checkpoint, restored, attachments,
            checkpoint_integrity, error))
    {
        tmv::log_error("[CHECKPOINT] cannot restore checkpoint: ", error);
        return 1;
    }
    if (global_perf_timing_enabled)
        tmv::log_info("[CHECKPOINT READ PERF] bytes=", checkpoint_integrity.bytes,
                      ", load_validate_decode_s=", std::chrono::duration<double>(PerfClock::now() - restore_started).count(),
                      ", sha256_implementation=", nimbus::crypto::sha256_implementation());
    if (checkpoint_integrity.bytes != sidecar.checkpoint_bytes ||
        checkpoint_integrity.fnv1a64 != sidecar.checkpoint_fnv1a64 ||
        (!sidecar.checkpoint_sha256.empty() &&
         checkpoint_integrity.sha256 != sidecar.checkpoint_sha256))
    {
        tmv::log_error("[CHECKPOINT] checkpoint sidecar integrity mismatch");
        return 1;
    }
    if (options.trajectory.enabled && !attachments.trajectories.has_value())
    {
        tmv::log_error("[CHECKPOINT] trajectory-enabled resume requires parcel state in the source checkpoint");
        return 1;
    }
    if (options.trajectory.enabled)
        pending_trajectory_restore = std::move(attachments.trajectories);
    if (!tmv::checkpoint::simulation_checkpoint_compatible(global_simulation_storage, restored, error))
    {
        tmv::log_error("[CHECKPOINT] incompatible resume request: ", error);
        return 1;
    }
    if (std::abs(restored.time_s - sidecar.simulation_time_s) > 1.0e-9)
    {
        tmv::log_error("[CHECKPOINT] sidecar time does not match checkpoint state");
        return 1;
    }
    if (duration_s >= 0 && restored.time_s + 1.0e-9 >= static_cast<double>(duration_s))
    {
        tmv::log_error(
            "[CHECKPOINT] requested duration must be later than checkpoint simulation time ", restored.time_s, "s");
        return 1;
    }

    StormTrackResumePlan storm_track_resume;
    if (global_storm_tracking_config.enabled &&
        !inspect_storm_track_resume_prefix(
            std::filesystem::path(outdir), sidecar.completed_steps,
            storm_track_resume, error))
    {
        tmv::log_error("[STORM TRACK] cannot preserve resume history: ", error);
        return 1;
    }

    // The serialized scientific state is portable; the target run's backend
    // request belongs to the new branch and is reinitialized locally.
    const RuntimeConfigurationState requested_configuration =
        global_simulation_storage.configuration;
    const ComputeBackendConfig requested_compute = global_compute_backend_config;
    const bool requested_native_chaos = global_simulation_storage.runtime.chaos.config.native_generation_required;
    const bool requested_gpu_apply_microphysics =
        global_gpu_apply_microphysics;
    const bool requested_gpu_apply_slow = global_gpu_apply_slow;
    const bool requested_gpu_apply_sgs = global_gpu_apply_sgs;
    const RadarProductSelection requested_radar_products =
        global_radar_product_selection;
    shutdown_compute_backend_runtime();
    global_simulation_storage     = std::move(restored);
    tmv::checkpoint::adopt_checkpoint_branch_configuration(
        requested_configuration, global_simulation_storage.configuration);
    global_compute_backend_config = requested_compute;
    global_simulation_storage.runtime.chaos.config.native_generation_required = requested_native_chaos;
    if (global_simulation_storage.nested_execution.relationship_ready)
        global_simulation_storage.nested_execution.child.runtime.chaos.config.native_generation_required = requested_native_chaos;
    global_simulation_storage.runtime.radar.product_selection =
        requested_radar_products;
    if (global_simulation_storage.nested_execution.relationship_ready)
        global_simulation_storage.nested_execution.child.runtime.radar
            .product_selection = requested_radar_products;
    // Checkpoints carry the source branch's ownership tuple for provenance,
    // but a topology-portable resume must execute the target branch's
    // admitted backend-local commit policy.
    global_gpu_apply_microphysics = requested_gpu_apply_microphysics;
    global_gpu_apply_slow = requested_gpu_apply_slow;
    global_gpu_apply_sgs = requested_gpu_apply_sgs;
    try
    {
        // Derived metric caches and terrain-member pointers are deliberately
        // not serialized. Rebuild them only after restored storage becomes
        // the live process owner so stretched flat levels and mapped terrain
        // both bind to their actual checkpoint geometry.
        refresh_grid_metrics_from_terrain();
        const std::string dynamics_name = global_dynamics_scheme_name.empty() ? "klemp_wilhelmson_split_cartesian_cgrid"
                                                                              : global_dynamics_scheme_name;
        // Dynamics scheme ownership and its full-grid execution scratch are
        // deliberately omitted from durable checkpoints. Reconstruct both
        // after the restored aggregate becomes live, before the first resumed
        // step can observe a null operator or empty tendency buffer.
        initialize_dynamics(dynamics_name);
        if (global_simulation_storage.nested_execution.relationship_ready)
            rehydrate_nested_child_transient_operators(
                global_simulation_storage);
    }
    catch (const std::exception& exception)
    {
        tmv::log_error("[CHECKPOINT] could not reconstruct dynamics runtime: ", exception.what());
        return 1;
    }
    if (!initialize_compute_backend_runtime(error))
    {
        tmv::log_error("[CHECKPOINT] restored state but could not initialize requested compute backend: ", error);
        return 1;
    }

    try
    {
        tmv::chaos::bind_chaos_compute_backend(global_simulation_state());
        if (global_simulation_storage.nested_execution.relationship_ready)
            tmv::chaos::bind_chaos_compute_backend(nested_child_execution_state(global_simulation_storage));
    }
    catch (const std::exception& exception)
    {
        tmv::log_error("[CHECKPOINT] could not bind restored stochastic execution: ", exception.what());
        return 1;
    }

    steps                    = sidecar.completed_steps > static_cast<std::uint64_t>(std::numeric_limits<int>::max())
                                   ? std::numeric_limits<int>::max()
                                   : static_cast<int>(sidecar.completed_steps);
    checkpoints.resume(sidecar.run_id, std::filesystem::absolute(options.resume_checkpoint).lexically_normal(), simulation_time, options.checkpoint_every_s);

    std::ostringstream provenance;
    provenance << std::setprecision(17) << "{\n"
               << "  \"schema\": \"nimbus.checkpoint.branch\",\n"
               << "  \"schema_version\": 1,\n"
               << "  \"run_id\": \"" << tmv::strutil::json_escape(checkpoints.run_id()) << "\",\n"
               << "  \"parent_run_id\": \"" << tmv::strutil::json_escape(checkpoints.parent_run_id()) << "\",\n"
               << "  \"parent_checkpoint\": \"" << tmv::strutil::json_escape(checkpoints.parent_path().string()) << "\",\n"
               << "  \"parent_checkpoint_sha256\": ";
    if (sidecar.checkpoint_sha256.empty())
        provenance << "null,\n";
    else
        provenance << "\"" << sidecar.checkpoint_sha256 << "\",\n";
    provenance
               << "  \"resumed_step\": " << sidecar.completed_steps << ",\n"
               << "  \"resumed_simulation_time_s\": " << simulation_time << "\n"
               << "}\n";
    if (!publish_output_json_atomically(std::filesystem::path(outdir) / "nimbus_branch.json", provenance.str(), error))
    {
        tmv::log_error("[CHECKPOINT] could not publish branch provenance: ", error);
        return 1;
    }
    if (global_storm_tracking_config.enabled &&
        !publish_storm_track_resume_marker(
            storm_track_resume, sidecar, checkpoints.parent_path(),
            checkpoints.run_id(), error))
    {
        tmv::log_error("[STORM TRACK] could not publish resume provenance: ", error);
        return 1;
    }
    tmv::log_info("[CHECKPOINT] resumed step=",
                  sidecar.completed_steps,
                  ", time_s=",
                  simulation_time,
                  ", parent_run_id=",
                  checkpoints.parent_run_id(),
                  ", branch_run_id=",
                  checkpoints.run_id());
    return 0;
}

int HeadlessDriver::maybe_write_checkpoint(bool final_checkpoint)
{
    return checkpoints.publish(options, global_simulation_storage, trajectories, steps, final_checkpoint);
}

bool OutputPublisher::enforce_output_retention()
{
    if (output_config.keep_last_frames <= 0)
        return true;
    const std::filesystem::path root(outdir);
    const std::array<std::filesystem::path, 4> series = {
        root,
        root / "nests" / "child_0000",
        root / "qualification_diagnostics",
        root / "nests" / "child_0000" / "qualification_diagnostics",
    };
    for (const auto& directory : series)
    {
        nimbus::output::RetentionResult result;
        std::string                     error;
        if (!nimbus::output::enforce_step_retention(
                directory, static_cast<std::size_t>(output_config.keep_last_frames), result, error))
        {
            validation.validation_error_message = "output retention failed: " + error;
            tmv::log_error("[OUTPUT] ", validation.validation_error_message);
            return false;
        }
        if (result.removed_items > 0 || result.removed_geometry_versions > 0)
        {
            tmv::log_info("[OUTPUT] retention series=",
                          directory.string(),
                          ", removed_frames=",
                          result.removed_items,
                          ", removed_geometry_versions=",
                          result.removed_geometry_versions,
                          ", dependency_frames_preserved=",
                          result.dependency_frames,
                          ", reclaimed_bytes=",
                          result.reclaimed_bytes);
        }
    }
    return true;
}

int HeadlessDriver::build_initial_conditions()
{
    output.next_field_export_time_s   = next_cadence_after(simulation_time, write_every_s);
    output.next_medium_export_time_s  = (output.output_config.tiered_write_cadence && write_every_s > 0)
                                     ? next_cadence_after(simulation_time, output.output_config.write_cadence_medium_s)
                                     : -1.0;
    output.next_slow_export_time_s    = (output.output_config.tiered_write_cadence && write_every_s > 0)
                                     ? next_cadence_after(simulation_time, output.output_config.write_cadence_slow_s)
                                     : -1.0;
    output.next_qualification_diagnostics_export_time_s =
        output.qualification_diagnostics_enabled()
            ? next_cadence_after(
                  simulation_time,
                  output.output_config.qualification_diagnostics_interval_s)
            : -1.0;
    checkpoints.schedule(simulation_time, options.checkpoint_every_s);
    output.next_shm_publish_time_s = options.live_shm_full_rate
        ? -1.0
        : tmv::runtime::next_live_shm_publish_time(
              simulation_time, options.live_shm_interval_s);
    tracking.next_storm_tracking_time_s = global_storm_tracking_config.enabled
                                     ? next_cadence_after(simulation_time, global_storm_tracking_config.interval_s)
                                     : -1.0;

    OutputFieldDependency terrain_output_dependencies =
        write_every_s > 0 ? output_dependencies_for_fields(output.output_config.resolved_fields) : OutputFieldDependency::None;
    if (options.live_shm)
    {
        terrain_output_dependencies |= output_dependencies_for_fields(options.live_shm_fields);
    }
    const bool terrain_ground_velocity_required =
        global_storm_tracking_config.enabled ||
        output_field_has_dependency(terrain_output_dependencies, OutputFieldDependency::CenteredW);
    const SimulationStateView initial_state = global_simulation_state();
    if (const int rc = validate_configured_nesting(); rc != 0)
        return rc;
    if (global_storm_tracking_config.recycle_domain)
    {
        const PhysicalCellDomain recycling_domain = physical_scalar_domain(initial_state.grid);
        const bool radiative_laterals = global_lateral_bc_west == tmv::init::config::LateralBC::Radiative &&
                                        global_lateral_bc_east == tmv::init::config::LateralBC::Radiative &&
                                        global_lateral_bc_south == tmv::init::config::LateralBC::Radiative &&
                                        global_lateral_bc_north == tmv::init::config::LateralBC::Radiative;
        const bool terrain_active =
            tmv::physical_grid_metrics::terrain_is_active(initial_state);
        const bool terrain_recycling_supported = !terrain_active
            || (initial_state.terrain_runtime != nullptr
                && initial_state.terrain_runtime->config.horizontal_placement_id
                    == "earth_fixed"
                && initial_state.terrain_runtime->earth_fixed_placement.active);
        if (!global_storm_tracking_config.enabled || global_storm_tracking_config.maximum_shift_cells <= 0 ||
            (global_domain_motion_mode != tmv::init::config::DomainMotionMode::ConstantVelocity &&
             global_domain_motion_mode != tmv::init::config::DomainMotionMode::TrackedWindow) ||
            global_coordinate_system != CoordinateSystem::Cartesian || global_stagger_type != StaggerType::CGrid ||
            !terrain_recycling_supported || initial_state.compute_runtime == nullptr ||
            (initial_state.compute_runtime->active_kind != ComputeBackendKind::Cpu &&
             (initial_state.compute_runtime->backend == nullptr ||
              !initial_state.compute_runtime->backend->supports_homogeneous_domain_field_shift())) ||
            global_simulation_storage.nested.config.enabled || !radiative_laterals ||
            global_storm_tracking_config.maximum_shift_cells >= std::min(recycling_domain.nx(), recycling_domain.ny()))
        {
            tmv::log_error("[DOMAIN RECYCLING] live recycling currently requires enabled "
                           "tracking, tracked_window or compatible constant_velocity mode, flat or Earth-fixed Cartesian C-grid CPU or "
                           "native accelerator execution, no active nest, four radiative lateral "
                           "boundaries, and maximum_shift_cells in [1, min(nx, ny)) for "
                           "physical horizontal dimensions");
            return 2;
        }
    }
    if (tmv::physical_grid_metrics::terrain_is_active(initial_state)
        && global_stagger_type == StaggerType::CGrid
        && (terrain_ground_velocity_required || write_every_s > 0
            || options.live_shm))
    {
        std::size_t columns = static_cast<std::size_t>(NR) * static_cast<std::size_t>(NTH);
        const auto& nesting = global_simulation_storage.configuration.nesting;
        if (nesting.enabled)
            columns = std::max(columns, (std::size_t(nesting.child_cells_x) + 2) *
                                       (std::size_t(nesting.child_cells_y) + 2));
        output.cgrid_ground_vertical_velocity_ms.resize(columns);
    }
    else
    {
        output.cgrid_ground_vertical_velocity_ms.clear();
    }

    if (global_storm_tracking_config.enabled)
    {
        tracking.storm_tracking_uses_nested_child =
            global_domain_motion_mode ==
                tmv::init::config::DomainMotionMode::MovingNest &&
            global_simulation_storage.configuration.nesting.enabled;
        const bool child_tracking_state_ready =
            tracking.storm_tracking_uses_nested_child &&
            global_simulation_storage.nested_execution.relationship_ready;
        const SimulationStateView state = child_tracking_state_ready
            ? make_simulation_state_view(
                  global_simulation_storage.nested_execution.child,
                  global_simulation_storage.configuration,
                  &global_simulation_storage.runtime.compute,
                  &global_simulation_storage.runtime.parity)
            : global_simulation_state();
        tmv::physical_grid_metrics::validate(state);
        if (tracking.storm_tracking_uses_nested_child &&
            tmv::physical_grid_metrics::terrain_is_active(state) &&
            !global_simulation_storage.configuration.admission.active_terrain_nested)
        {
            tmv::log_error(
                "[STORM TRACK] refined-child terrain tracking requires admitted mapped geometry");
            return 2;
        }
        const PhysicalCellDomain domain = physical_scalar_domain(state.grid);
        if (domain.empty())
        {
            tmv::log_error("[STORM TRACK] physical scalar domain is empty");
            return 2;
        }
        const std::size_t tracking_nx = tracking.storm_tracking_uses_nested_child
            ? static_cast<std::size_t>(
                  global_simulation_storage.configuration.nesting.child_cells_x)
            : static_cast<std::size_t>(domain.nx());
        const std::size_t tracking_ny = tracking.storm_tracking_uses_nested_child
            ? static_cast<std::size_t>(
                  global_simulation_storage.configuration.nesting.child_cells_y)
            : static_cast<std::size_t>(domain.ny());
        if (tracking_nx == 0 || tracking_ny == 0 ||
            tracking_nx > std::numeric_limits<std::size_t>::max() / tracking_ny)
        {
            tmv::log_error("[STORM TRACK] configured tracking workspace is invalid");
            return 2;
        }
        const std::size_t column_count = tracking_nx * tracking_ny;
        tracking.storm_feature_detector.reserve(tracking_nx, tracking_ny);
        tracking.storm_maximum_updraft_ms.resize(column_count);
        tracking.storm_condensate_path_kgm2.resize(column_count);
        tracking.storm_updraft_helicity_m2s2.resize(column_count);
        tracking.storm_branch_association = {};
        tracking.storm_branch_association.environmental_velocity_uncertainty_ms =
            global_storm_tracking_config.branch_velocity_uncertainty_ms;
        switch (global_storm_tracking_config.branch_preference)
        {
        case tmv::init::config::StormBranchPreference::Continuity:
            tracking.storm_branch_association.preference = tmv::diagnostics::StormBranchPreference::Continuity;
            break;
        case tmv::init::config::StormBranchPreference::BunkersRight:
            tracking.storm_branch_association.preference = tmv::diagnostics::StormBranchPreference::BunkersRight;
            break;
        case tmv::init::config::StormBranchPreference::BunkersLeft:
            tracking.storm_branch_association.preference = tmv::diagnostics::StormBranchPreference::BunkersLeft;
            break;
        }
        try
        {
            const auto right = tmv::domain::estimate_bunkers_storm_motion(
                state.grid.geometry.z, state.base.u0, state.base.v0, tmv::domain::StormMotionMethod::BunkersRight);
            const auto left = tmv::domain::estimate_bunkers_storm_motion(
                state.grid.geometry.z, state.base.u0, state.base.v0, tmv::domain::StormMotionMethod::BunkersLeft);
            const tmv::domain::HorizontalVelocity right_detector =
                tracking.storm_tracking_uses_nested_child
                    ? right.velocity_ms
                    : global_domain_frame.earth_to_grid(right.velocity_ms);
            const tmv::domain::HorizontalVelocity left_detector =
                tracking.storm_tracking_uses_nested_child
                    ? left.velocity_ms
                    : global_domain_frame.earth_to_grid(left.velocity_ms);
            tracking.storm_branch_association.environmental_motion_available = true;
            tracking.storm_branch_association.bunkers_right_velocity_x_ms    = right_detector.u_ms;
            tracking.storm_branch_association.bunkers_right_velocity_y_ms    = right_detector.v_ms;
            tracking.storm_branch_association.bunkers_left_velocity_x_ms     = left_detector.u_ms;
            tracking.storm_branch_association.bunkers_left_velocity_y_ms     = left_detector.v_ms;
        }
        catch (const std::exception& error)
        {
            if (tracking.storm_branch_association.preference != tmv::diagnostics::StormBranchPreference::Continuity)
            {
                tmv::log_error("[STORM TRACK] configured Bunkers branch preference is unavailable: ", error.what());
                return 2;
            }
            tmv::log_info("[STORM TRACK] environmental right/left classification unavailable; "
                          "temporal continuity remains active: ",
                          error.what());
        }
        bool rotation_layer_covered = true;
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
            {
                const double ground_m = tmv::physical_grid_metrics::interface_height(state, i, j, 0);
                const double physical_bottom_m =
                    tmv::physical_grid_metrics::interface_height(state, i, j, domain.k_begin);
                const double physical_top_m = tmv::physical_grid_metrics::interface_height(state, i, j, domain.k_end);
                const double requested_bottom_m = ground_m + global_storm_tracking_config.updraft_helicity_bottom_m_agl;
                const double requested_top_m    = ground_m + global_storm_tracking_config.updraft_helicity_top_m_agl;
                rotation_layer_covered = rotation_layer_covered && std::isfinite(ground_m) &&
                                         std::isfinite(physical_bottom_m) && std::isfinite(physical_top_m) &&
                                         physical_bottom_m <= requested_bottom_m && physical_top_m >= requested_top_m;
            }
        if (!rotation_layer_covered)
        {
            tmv::log_error("[STORM TRACK] requested updraft-helicity AGL layer is not completely represented "
                           "by every physical model column");
            return 2;
        }
        tmv::log_info("[STORM TRACK] snapshot detector enabled: cadence=",
                      global_storm_tracking_config.interval_s,
                      "s, algorithm=",
                      tmv::diagnostics::storm_feature_detector_algorithm_id(),
                      ", rotation_algorithm=",
                      tmv::diagnostics::storm_updraft_helicity_algorithm_id(),
                      ", physical_columns=",
                      column_count,
                      ", tracking_source=",
                      tracking.storm_tracking_uses_nested_child
                          ? "synchronized_nested_child"
                          : "root_grid",
                      ", branch_preference=",
                      tmv::diagnostics::storm_branch_preference_name(tracking.storm_branch_association.preference),
                      ", topology_confirmation_samples=",
                      global_storm_tracking_config.topology_confirmation_samples,
                      ", movement_mode=",
                      global_storm_tracking_config.recycle_domain
                          ? (tmv::physical_grid_metrics::terrain_is_active(
                                 initial_state)
                                 ? "live_earth_fixed_terrain_cgrid_recycling"
                                 : "live_flat_cgrid_recycling")
                          : (tracking.storm_tracking_uses_nested_child
                                 ? "live_one_way_nested_child"
                                 : (global_storm_tracking_config.maximum_shift_cells > 0
                                        ? "diagnostic_request_only"
                                        : "diagnostic_only")));
    }

    if (global_perf_timing_enabled)
    {
        reset_advection_perf_stats();
        reset_dynamics_perf_stats();
        reset_turbulence_perf_stats();
    }

    if (output.export_diagnostics_enabled)
    {
        float theta_min = 1e10, theta_max = -1e10;
        float u_min = 1e10, u_max = -1e10;
        int   nan_count = 0;
        for (int i = 0; i < NR && i < 10; ++i)
        {
            for (int j = 0; j < NTH && j < 5; ++j)
            {
                for (int k = 0; k < NZ && k < 5; ++k)
                {
                    if (std::isnan(theta[i][j][k]))
                        nan_count++;
                    if (theta[i][j][k] < theta_min)
                        theta_min = theta[i][j][k];
                    if (theta[i][j][k] > theta_max)
                        theta_max = theta[i][j][k];
                    if (u[i][j][k] < u_min)
                        u_min = u[i][j][k];
                    if (u[i][j][k] > u_max)
                        u_max = u[i][j][k];
                }
            }
        }
        tmv::log_info("[STEP DIAGNOSTIC] step=0 time_s=0",
                      " sample_theta_min_k=", theta_min,
                      " sample_theta_max_k=", theta_max,
                      " sample_u_min_ms=", u_min,
                      " sample_u_max_ms=", u_max,
                      " sample_nan=", nan_count,
                      " cadence_units=simulation_seconds");
        if (theta_min < 0 || theta_max > 700)
        { // matches simulation.hpp theta_max_k
            tmv::log_error("[STEP DIAGNOSTIC] initial theta sample is outside the runtime contract");
        }
    }

    bool initial_radar_diagnostics_ready = false;
    bool initial_vorticity_diagnostics_ready = false;
    bool initial_angular_momentum_diagnostics_ready = false;
    bool initial_pressure_diagnostics_ready = false;
    bool initial_pressure_components_ready = false;

#ifdef EXPORT_NPY
    if (write_every_s > 0)
    {
        // Initial export writes all fields regardless of cadence tier
        output.active_export_fields = output.output_config.resolved_fields;

        // Diagnostics are lazy. Allocate and refresh every requested derived
        // field before publishing t=0; otherwise the first frame would either
        // dereference empty storage or advertise allocation defaults as science.
        const OutputFieldDependency initial_output_dependencies = output_dependencies_for_fields(output.active_export_fields);
        if (output_field_has_dependency(initial_output_dependencies, OutputFieldDependency::RadarDiagnostic))
        {
            calculate_radar_observables();
            initial_radar_diagnostics_ready = true;
        }
        if (output_field_has_dependency(initial_output_dependencies, OutputFieldDependency::VorticityDiagnostics) ||
            output_field_has_dependency(initial_output_dependencies,
                                        OutputFieldDependency::AngularMomentumDiagnostics) ||
            output_field_has_dependency(initial_output_dependencies, OutputFieldDependency::PressureDiagnostics))
        {
            const bool initial_vorticity_required =
                output_field_has_dependency(initial_output_dependencies, OutputFieldDependency::VorticityDiagnostics);
            tmv::diagnostics::compute_resolved_diagnostics(
                global_simulation_state(),
                {initial_vorticity_required,
                output_field_has_dependency(initial_output_dependencies,
                                            OutputFieldDependency::AngularMomentumDiagnostics),
                output_field_has_dependency(initial_output_dependencies, OutputFieldDependency::PressureDiagnostics),
                output_field_has_dependency(initial_output_dependencies, OutputFieldDependency::PressureComponents)});
            initial_vorticity_diagnostics_ready = initial_vorticity_required;
            initial_angular_momentum_diagnostics_ready =
                output_field_has_dependency(
                    initial_output_dependencies,
                    OutputFieldDependency::AngularMomentumDiagnostics);
            initial_pressure_diagnostics_ready =
                output_field_has_dependency(
                    initial_output_dependencies,
                    OutputFieldDependency::PressureDiagnostics);
            initial_pressure_components_ready =
                output_field_has_dependency(initial_output_dependencies, OutputFieldDependency::PressureComponents);
        }

        // Initial output happens before the measured solver-step window. Keep
        // it out of the solver benchmark instead of publishing a permanently
        // zero/dead timing bucket.
        const bool initial_export_ok = output.write_all_fields(output.export_index++);
        if (!initial_export_ok)
        {
            tmv::log_error("[VALIDATION] ", validation.validation_error_message);
            return 2;
        }
    }
#endif

    if (options.live_shm)
    {
        const OutputFieldDependency dependencies =
            output_dependencies_for_fields(options.live_shm_fields);
        const bool need_radar = output_field_has_dependency(
            dependencies, OutputFieldDependency::RadarDiagnostic);
        const bool need_vorticity = output_field_has_dependency(
            dependencies, OutputFieldDependency::VorticityDiagnostics);
        const bool need_angular = output_field_has_dependency(
            dependencies, OutputFieldDependency::AngularMomentumDiagnostics);
        const bool need_pressure = output_field_has_dependency(
            dependencies, OutputFieldDependency::PressureDiagnostics);
        const bool need_pressure_components = output_field_has_dependency(
            dependencies, OutputFieldDependency::PressureComponents);
        if (need_radar && !initial_radar_diagnostics_ready)
            calculate_radar_observables();
        if ((need_vorticity && !initial_vorticity_diagnostics_ready) ||
            (need_angular && !initial_angular_momentum_diagnostics_ready) ||
            (need_pressure && !initial_pressure_diagnostics_ready) ||
            (need_pressure_components && !initial_pressure_components_ready))
        {
            tmv::diagnostics::compute_resolved_diagnostics(
                global_simulation_state(),
                {need_vorticity && !initial_vorticity_diagnostics_ready,
                need_angular && !initial_angular_momentum_diagnostics_ready,
                need_pressure && !initial_pressure_diagnostics_ready,
                need_pressure_components && !initial_pressure_components_ready});
        }
        if (!output.shm_update(true))
            return 2;
    }

    // --- Lagrangian trajectories: construct, configure, and seed if due ---
    if (options.trajectory.enabled)
    {
        trajectories.configure(options.trajectory,
                               global_grid_geometry.coord,
                               global_stagger_type == StaggerType::CGrid,
                               NR,
                               NTH,
                               NZ,
                               dr,
                               dtheta,
                               dz,
                               global_domain_frame,
                               tmv::physical_grid_metrics::active_terrain_metrics(
                                   global_simulation_state()),
                               global_terrain_config.horizontal_placement_id == "earth_fixed");
        if (pending_trajectory_restore.has_value())
        {
            std::string restore_error;
            if (!trajectories.restore_checkpoint_state(
                    std::move(*pending_trajectory_restore), simulation_time,
                    restore_error))
            {
                tmv::log_error("[CHECKPOINT] cannot restore trajectories: ",
                               restore_error);
                return 1;
            }
            pending_trajectory_restore.reset();
            tmv::log_info("[CHECKPOINT] restored ", trajectories.count(),
                          " trajectories with ", trajectories.total_samples(),
                          " historical samples");
        }
        else if (trajectories.seed_time_s() <= simulation_time)
        {
            const bool online_sampling_supported =
                global_grid_geometry.coord == CoordinateSystem::Cartesian;
            const bool automatic_center =
                std::isnan(options.trajectory.center_x_m) || std::isnan(options.trajectory.center_y_m);
            const bool needs_gridded_diagnostics = tmv::runtime::trajectory_requires_gridded_diagnostics(
                true, false, true, automatic_center, online_sampling_supported);
            if (needs_gridded_diagnostics && !initial_vorticity_diagnostics_ready)
            {
                compute_dynamics_diagnostics(true, false, false);
            }
            trajectories.seed(simulation_time, u, v, w, rho, p, vorticity_z);
        }
        tmv::log_info("[TRAJECTORIES] enabled: seeding ",
                      (trajectories.seeded() ? trajectories.count() : 0),
                      " parcels (seed_time=",
                      trajectories.seed_time_s(),
                      "s)");
    }
    log_memory_milestone("startup_ready");
    return 0;
}

int HeadlessDriver::run_time_loop()
{
    // A zero-duration run publishes its initialized state without applying a
    // physics step. Resumed runs at/past their requested endpoint are rejected
    // earlier, but this precondition also keeps the loop boundary self-owned.
    if (should_stop())
    {
        return 0;
    }
    if (capture_termination_request())
        return 0;
    while (true)
    {
        perf_step_active = global_perf_timing_enabled && steps >= global_perf_warmup_steps;
        if (global_perf_timing_enabled && steps == global_perf_warmup_steps)
        {
            // Internal advection totals are process-global and begin collecting
            // as soon as timing is enabled. Reset exactly at the warmup
            // boundary so the final breakdown covers the same measured steps
            // as the runtime totals below.
            reset_advection_perf_stats();
            reset_dynamics_perf_stats();
            reset_turbulence_perf_stats();
#ifdef NIMBUS_ALLOCATION_TELEMETRY
            tmv::allocation_telemetry::begin_measurement();
            allocation_measurement_started = true;
#endif
        }
        if (global_perf_timing_enabled)
        {
            // The kernel snapshot is genuinely per-step only when reset before
            // each advance. Previously the "KERNEL STEP" log was cumulative.
            reset_advection_kernel_step_timing();
        }

        const bool native_flat_step = global_simulation_storage.configuration.admission.native_dry_tvd &&
            global_simulation_storage.runtime.compute.active_kind == ComputeBackendKind::Metal &&
            !global_simulation_storage.runtime.compute.fallback_active;
        const bool static_vertical_cpu =
            global_simulation_storage.configuration.admission.static_vertical_only_sgs_cpu &&
            global_simulation_storage.runtime.compute.active_kind == ComputeBackendKind::Cpu &&
            !global_simulation_storage.runtime.compute.fallback_active;
        std::optional<tmv::runtime::FlatCgridStepTransaction> state_transaction;
        std::optional<tmv::BudgetPublicationTransaction> budget_transaction;
        if (native_flat_step || static_vertical_cpu)
        {
            state_transaction.emplace(global_simulation_storage);
            budget_transaction.emplace();
        }

        PerfClock::time_point step_t0{};
        if (perf_step_active)
        {
            // Start before choose_runtime_timestep(): the CFL scan is a
            // full-domain solver cost and must be inside a solver-step metric.
            step_t0 = PerfClock::now();
        }
        if (const int rc = advance_one_step(); rc != 0)
        {
            return rc;
        }
        if (state_transaction)
        {
            state_transaction->commit();
            state_transaction.reset();
        }
        // Solver timing deliberately stops before snapshot/output work. Export
        // time is tracked independently, while the benchmark runner measures
        // whole-process end-to-end wall time.
        record_step_timing(step_t0);
#ifdef NIMBUS_ALLOCATION_TELEMETRY
        // Match allocation accounting to the same solver-step boundary.
        // The benchmark's mandatory final checkpoint is endpoint proof, not
        // solver work, and must not be charged as per-step physics churn.
        if (allocation_measurement_started && should_stop())
        {
            allocation_snapshot            = tmv::allocation_telemetry::end_measurement();
            allocation_measurement_started = false;
        }
#endif
        if (const int rc = output.maybe_write_output(); rc != 0)
        {
            return rc;
        }
        if (budget_transaction)
        {
            budget_transaction->commit();
            budget_transaction.reset();
        }

        ++steps;
        if (should_stop())
        {
            break;
        }
        if (capture_termination_request())
        {
            break;
        }
        if (const int rc = maybe_write_checkpoint(); rc != 0)
        {
            return rc;
        }
    }
    return 0;
}

int HeadlessDriver::advance_one_step()
{
    const std::uint64_t nonfinite_incidents_before_step = tmv::numerical_nonfinite_incidents();
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    tmv::allocation_telemetry::ScopedComponent timestep_allocations(
        tmv::allocation_telemetry::Component::RuntimeTimestep);
#endif
    const double runtime_dt = choose_runtime_timestep();
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    timestep_allocations.restore();
#endif
    if (std::isfinite(runtime_dt) && runtime_dt > 0.0)
    {
        dt = runtime_dt;
    }

    if (duration_s >= 0)
    {
        dt = tmv::runtime::clamp_step_to_deadline(
            simulation_time, dt, static_cast<double>(duration_s), static_cast<std::uint64_t>(steps));
    }

    const double next_simulation_time = simulation_time + dt;
    const bool   output_deadline_due  = write_every_s > 0 && next_simulation_time + 1.0e-9 >= output.next_field_export_time_s;
    const bool qualification_diagnostics_deadline_due =
        output.qualification_diagnostics_enabled() &&
        next_simulation_time + 1.0e-9 >=
            output.next_qualification_diagnostics_export_time_s;
    const bool live_shm_deadline_due =
        tmv::runtime::live_shm_publish_due(
            options.live_shm, options.live_shm_full_rate,
            next_simulation_time, output.next_shm_publish_time_s);
    const bool   final_step_due = duration_s >= 0 && next_simulation_time + 1.0e-9 >= static_cast<double>(duration_s);
    const bool   compatibility_scientific_due = write_every_s <= 0 ||
                                                live_shm_deadline_due ||
                                                log_debug_enabled() || output_deadline_due ||
                                                qualification_diagnostics_deadline_due || final_step_due;

    OutputFieldDependency output_dependencies =
        output_deadline_due ? output_dependencies_for_fields(output.output_config.resolved_fields)
                            : OutputFieldDependency::None;
    if (qualification_diagnostics_deadline_due)
    {
        output_dependencies |= output_dependencies_for_fields(output.qualification_output_config.resolved_fields);
    }
    const OutputFieldDependency live_dependencies =
        live_shm_deadline_due ? output_dependencies_for_fields(options.live_shm_fields) : OutputFieldDependency::None;
    const bool radar_diagnostics_required =
        output_field_has_dependency(output_dependencies, OutputFieldDependency::RadarDiagnostic) ||
        output_field_has_dependency(live_dependencies, OutputFieldDependency::RadarDiagnostic);
    bool vorticity_diagnostics_required =
        output_field_has_dependency(output_dependencies, OutputFieldDependency::VorticityDiagnostics) ||
        output_field_has_dependency(live_dependencies, OutputFieldDependency::VorticityDiagnostics);
    const bool angular_momentum_diagnostics_required =
        output_field_has_dependency(output_dependencies, OutputFieldDependency::AngularMomentumDiagnostics) ||
        output_field_has_dependency(live_dependencies, OutputFieldDependency::AngularMomentumDiagnostics);
    const bool pressure_diagnostics_required =
        output_field_has_dependency(output_dependencies, OutputFieldDependency::PressureDiagnostics) ||
        output_field_has_dependency(live_dependencies, OutputFieldDependency::PressureDiagnostics);
    const bool trajectory_seed_due = options.trajectory.enabled && !trajectories.seeded() &&
                                     next_simulation_time + 1.0e-9 >= trajectories.seed_time_s();
    const bool trajectory_automatic_center =
        std::isnan(options.trajectory.center_x_m) || std::isnan(options.trajectory.center_y_m);
    const bool trajectory_online_sampling_supported = global_grid_geometry.coord == CoordinateSystem::Cartesian;
    const bool trajectory_requires_derived =
        tmv::runtime::trajectory_requires_gridded_diagnostics(options.trajectory.enabled,
                                                              trajectories.seeded(),
                                                              trajectory_seed_due,
                                                              trajectory_automatic_center,
                                                              trajectory_online_sampling_supported);
    vorticity_diagnostics_required = vorticity_diagnostics_required || trajectory_requires_derived;
    const bool derived_required    = radar_diagnostics_required || vorticity_diagnostics_required ||
                                     angular_momentum_diagnostics_required || pressure_diagnostics_required;
    const tmv::runtime::DiagnosticStepPlan diagnostic_plan =
        tmv::runtime::make_diagnostic_step_plan(static_cast<std::uint64_t>(steps) + 1,
                                                scientific_diagnostics_every_steps,
                                                compatibility_scientific_due,
                                                derived_required);
    set_step_diagnostics_enabled(diagnostic_plan.scientific);
    // Publication owns field diagnostics; online trajectories may need vorticity before export.
    set_step_derived_diagnostic_requirements(
        false, diagnostic_plan.derived && trajectory_requires_derived, false, false);

    // GPU audit Phase 3.6 (extended 2026-06-10): the five per-stage
    // validations below sweep ~25 full fields each. In Warn mode their
    // result struct is discarded (nothing is persisted or logged outside
    // the export-step report), and in Off mode validation is contractually
    // disabled - profiling showed the sweeps were 47% of beast step wall
    // time. Observe-only modes therefore follow the diagnostic-step
    // cadence (Warn) or skip stage sweeps entirely (Off). Sanitize
    // mutates state and Strict aborts the run - both are load-bearing
    // physics guards and keep running every step. The pre_export
    // validation (report artifact) is unaffected in all modes.
    const bool stage_validation_active =
        global_validation_policy.mode == tmv::GuardMode::Sanitize ||
        global_validation_policy.mode == tmv::GuardMode::Strict ||
        (global_validation_policy.mode == tmv::GuardMode::Warn && step_diagnostics_enabled());

    if (global_simulation_storage.configuration.nesting.enabled)
    {
        nested_previous_parent_fields = global_simulation_storage.fields;
        capture_nested_parent_microphysics_snapshot(
            global_simulation_storage,
            nested_previous_parent_microphysics);
        if (global_simulation_storage.configuration.nesting.feedback ==
            tmv::init::config::NestingFeedback::TwoWay)
        {
            global_simulation_storage.nested_execution
                .two_way_interval_source_begin =
                global_simulation_storage.nested_execution
                    .interface_source.cumulative;
            if (global_simulation_storage.nested_execution
                    .relationship_ready)
                global_simulation_storage.nested_execution
                    .two_way_child_interval_begin_inventory =
                    nested_two_way_child_inventory(
                        global_simulation_storage.nested_execution);
            const auto& config =
                global_simulation_storage.configuration.nesting;
            tmv::domain::OneWayNestRelationshipV1 relationship;
            if (global_simulation_storage.nested_execution.relationship_ready)
                relationship =
                    global_simulation_storage.nested_execution.relationship;
            else
            {
                const auto physical = physical_scalar_domain(
                    global_simulation_storage.grid.nr,
                    global_simulation_storage.grid.nth,
                    global_simulation_storage.grid.nz,
                    StaggerType::CGrid);
                tmv::domain::OneWayNestRelationshipInput input;
                input.parent_cells_x = physical.nx();
                input.parent_cells_y = physical.ny();
                input.parent_levels_z = global_simulation_storage.grid.nz;
                input.parent_dx_m = global_simulation_storage.grid.dx;
                input.parent_dy_m = global_simulation_storage.grid.dy;
                input.parent_dt_s = dt;
                input.parent_ratio_x = config.parent_ratio_x;
                input.parent_ratio_y = config.parent_ratio_y;
                input.parent_time_step_ratio = config.parent_time_step_ratio;
                input.child_origin_parent_cell_i =
                    config.child_origin_parent_cell_i;
                input.child_origin_parent_cell_j =
                    config.child_origin_parent_cell_j;
                input.child_cells_x = config.child_cells_x;
                input.child_cells_y = config.child_cells_y;
                input.child_levels_z = global_simulation_storage.grid.nz;
                input.relaxation_width_child_cells =
                    config.relaxation_width_child_cells;
                input.boundary_stencil_parent_cells =
                    config.boundary_stencil_parent_cells;
                relationship =
                    tmv::domain::prepare_one_way_nest_relationship(input);
            }
            configure_two_way_level_flux_capture(
                global_simulation_storage.runtime,
                global_simulation_storage.grid,
                {relationship.child_origin_parent_cell_i + 1,
                 relationship.child_origin_parent_cell_i +
                     relationship.child_parent_span_x + 1,
                 relationship.child_origin_parent_cell_j + 1,
                 relationship.child_origin_parent_cell_j +
                     relationship.child_parent_span_y + 1,
                 0, relationship.levels_z},
                1, global_simulation_storage.configuration.microphysics_scheme != "none");
            if (global_simulation_storage.nested_execution.relationship_ready)
            {
                configure_two_way_level_flux_capture(
                    global_simulation_storage.nested_execution.child.runtime,
                    global_simulation_storage.nested_execution.child.grid,
                    {1, relationship.child_cells_x + 1,
                     1, relationship.child_cells_y + 1,
                     0, relationship.levels_z},
                    relationship.parent_ratio_x, global_simulation_storage.configuration.microphysics_scheme != "none");
            }
        }
    }

    std::unique_ptr<tmv::EarthFixedStepTransaction> earth_step;
    if (global_terrain_config.scheme_id != "none"
        && global_terrain_config.horizontal_placement_id == "earth_fixed"
        && (global_simulation_storage.runtime.compute.active_kind == ComputeBackendKind::Cpu
            || global_simulation_storage.configuration.admission.earth_fixed_terrain_motion)
        && !global_simulation_storage.configuration.nesting.enabled)
        earth_step = std::make_unique<tmv::EarthFixedStepTransaction>(global_simulation_storage, dt);
    const auto timestep_state = earth_step ? earth_step->state() : global_simulation_state();

#ifdef NIMBUS_ALLOCATION_TELEMETRY
    const auto radiation_allocations_before = tmv::allocation_telemetry::current_snapshot();
#endif
    if (!earth_step && options.trajectory.enabled && trajectories.seeded())
        trajectories.prepare_advance(simulation_time, dt, u, v, w);
    timed_call(perf_totals.radiation_s, [&] { step_radiation(timestep_state, simulation_time); });
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    accumulate_allocation_delta(
        radiation_allocation_snapshot, radiation_allocations_before, tmv::allocation_telemetry::current_snapshot());
#endif
    if (steps == 0)
        log_memory_milestone("first_radiation");
    if (stage_validation_active && !validation.validate_core_fields(timestep_state, "after_radiation", steps, false, nullptr, false))
    {
        tmv::log_error("[VALIDATION] ", validation.validation_error_message);
        return 2;
    }
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    const auto boundary_layer_allocations_before = tmv::allocation_telemetry::current_snapshot();
#endif
    timed_call(perf_totals.boundary_layer_s, [&] { step_boundary_layer(timestep_state, simulation_time); });
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    accumulate_allocation_delta(boundary_layer_allocation_snapshot,
                                boundary_layer_allocations_before,
                                tmv::allocation_telemetry::current_snapshot());
#endif
    if (steps == 0)
        log_memory_milestone("first_boundary_layer");
    if (stage_validation_active && !validation.validate_core_fields(timestep_state, "after_boundary_layer", steps, false, nullptr, false))
    {
        tmv::log_error("[VALIDATION] ", validation.validation_error_message);
        return 2;
    }
    timed_call(perf_totals.chaos_noise_s, [&] { step_chaos_noise(timestep_state, dt); });
    if (stage_validation_active && !validation.validate_core_fields(timestep_state, "after_chaos_noise", steps, false, nullptr, false))
    {
        tmv::log_error("[VALIDATION] ", validation.validation_error_message);
        return 2;
    }
    timed_call(perf_totals.chaos_tendency_s, [&] { apply_chaos_tendencies(timestep_state); });
    if (stage_validation_active && !validation.validate_core_fields(timestep_state, "after_chaos_tendency", steps, false, nullptr, false))
    {
        tmv::log_error("[VALIDATION] ", validation.validation_error_message);
        return 2;
    }
    if (step_diagnostics_enabled())
    {
        // Loud instability telemetry (SCIENCE.md M3): the wind
        // clamps inside the acoustic loop are silent, so the CFL
        // scan's max wind is the visible blow-up signal.
        tmv::log_info("[FLOW MAX] t=", simulation_time, " max_wind_ms=", last_max_flow_speed());
    }
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    const auto dynamics_allocations_before = tmv::allocation_telemetry::current_snapshot();
#endif
    const auto smagorinsky_dispatch_counts = []
    {
        std::array<std::uint64_t, 3> counts{};
        const auto& compute = global_simulation_storage.runtime.compute;
        if (!compute.backend)
            return counts;
        constexpr std::array<nimbus::compute::KernelId, 3> kernels{
            nimbus::compute::KernelId::SmagorinskyCoefficients,
            nimbus::compute::KernelId::SmagorinskyTendencies,
            nimbus::compute::KernelId::SgsApply};
        for (std::size_t index = 0; index < counts.size(); ++index)
            counts[index] = compute.backend->successful_kernel_dispatch_count(
                kernels[index]);
        return counts;
    };
    const bool track_nested_kessler_dispatches =
        global_simulation_storage.configuration.nesting.enabled &&
        global_simulation_storage.runtime.compute.active_kind !=
            ComputeBackendKind::Cpu &&
        global_simulation_storage.runtime.microphysics.scheme &&
        global_simulation_storage.runtime.microphysics.scheme->get_scheme_name() ==
            "kessler";
    const auto kessler_dispatch_counts = [track_nested_kessler_dispatches]
    {
        std::array<std::uint64_t, 3> counts{};
        const auto& compute = global_simulation_storage.runtime.compute;
        if (!track_nested_kessler_dispatches || !compute.backend)
            return counts;
        constexpr std::array<nimbus::compute::KernelId, 3> kernels{
            nimbus::compute::KernelId::KesslerPointwise,
            nimbus::compute::KernelId::KesslerSedimentation,
            nimbus::compute::KernelId::MicrophysicsApply};
        for (std::size_t index = 0; index < counts.size(); ++index)
            counts[index] = compute.backend->successful_kernel_dispatch_count(
                kernels[index]);
        return counts;
    };
    const auto before_parent_smagorinsky = smagorinsky_dispatch_counts();
    const auto before_parent_kessler = kessler_dispatch_counts();
    timed_call(perf_totals.dynamics_s, [&] { step_dynamics(timestep_state, simulation_time); });
    const auto after_parent_smagorinsky = smagorinsky_dispatch_counts();
    const auto after_parent_kessler = kessler_dispatch_counts();
    for (std::size_t index = 0;
         index < nested_parent_smagorinsky_dispatches.size(); ++index)
    {
        if (after_parent_smagorinsky[index] <
            before_parent_smagorinsky[index])
        {
            throw std::runtime_error(
                "Smagorinsky device dispatch counter regressed during parent execution");
        }
        nested_parent_smagorinsky_dispatches[index] +=
            after_parent_smagorinsky[index] - before_parent_smagorinsky[index];
    }
    for (std::size_t index = 0;
         index < nested_parent_kessler_dispatches.size(); ++index)
    {
        if (after_parent_kessler[index] < before_parent_kessler[index])
        {
            throw std::runtime_error(
                "Kessler device dispatch counter regressed during parent execution");
        }
        nested_parent_kessler_dispatches[index] +=
            after_parent_kessler[index] - before_parent_kessler[index];
    }
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    accumulate_allocation_delta(
        dynamics_allocation_snapshot, dynamics_allocations_before, tmv::allocation_telemetry::current_snapshot());
#endif
    if (steps == 0)
        log_memory_milestone("first_dynamics");
    const std::uint64_t nonfinite_incidents_after_step = tmv::numerical_nonfinite_incidents();
    if (nonfinite_incidents_after_step > nonfinite_incidents_before_step)
    {
        tmv::log_error("[NUMERICAL INTEGRITY] step=",
                       steps,
                       " detected_nonfinite_incidents=",
                       nonfinite_incidents_after_step - nonfinite_incidents_before_step,
                       "; guard mode cannot convert an undefined numerical result into a successful step");
        return 2;
    }
    // A mode-independent post-step sweep is load-bearing: Off and Warn may
    // skip the more expensive per-physics-stage checks, but neither may turn
    // a non-finite primary state into a successful run.
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    tmv::allocation_telemetry::ScopedComponent integrity_allocations(
        tmv::allocation_telemetry::Component::IntegrityValidation);
#endif
    if (earth_step) earth_step->finish_evolution();
    if (!validation.validate_core_fields(timestep_state, "after_dynamics", steps, false, nullptr, false))
    {
        tmv::log_error("[VALIDATION] ", validation.validation_error_message);
        return 2;
    }
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    integrity_allocations.restore();
#endif
    if (earth_step)
    {
        earth_step->prepare_commit();
        if (options.trajectory.enabled && trajectories.seeded())
            trajectories.prepare_advance(simulation_time, dt, u, v, w);
        if (!earth_step->publish())
            throw std::runtime_error("Earth-fixed timestep publication became stale");
        earth_step->publish_reports();
    }
    else
        simulation_time += dt;

    if (const int rc = advance_configured_nested_child(); rc != 0)
    {
        return rc;
    }
    const auto after_child_smagorinsky = smagorinsky_dispatch_counts();
    const auto after_child_kessler = kessler_dispatch_counts();
    for (std::size_t index = 0;
         index < nested_child_smagorinsky_dispatches.size(); ++index)
    {
        if (after_child_smagorinsky[index] <
            after_parent_smagorinsky[index])
        {
            throw std::runtime_error(
                "Smagorinsky device dispatch counter regressed during child execution");
        }
        nested_child_smagorinsky_dispatches[index] +=
            after_child_smagorinsky[index] - after_parent_smagorinsky[index];
    }
    for (std::size_t index = 0;
         index < nested_child_kessler_dispatches.size(); ++index)
    {
        if (after_child_kessler[index] < after_parent_kessler[index])
        {
            throw std::runtime_error(
                "Kessler device dispatch counter regressed during child execution");
        }
        nested_child_kessler_dispatches[index] +=
            after_child_kessler[index] - after_parent_kessler[index];
    }

    // Complete parcel integration before recycling changes the grid frame. Seed
    // on the first step that reaches the configured seed time.
    if (options.trajectory.enabled)
    {
        if (!trajectories.seeded() && simulation_time + 1.0e-9 >= trajectories.seed_time_s())
        {
            trajectories.seed(simulation_time, u, v, w, rho, p, vorticity_z);
        }
        else if (trajectories.seeded())
        {
            trajectories.finish_advance(simulation_time, dt, u, v, w, rho, p, vorticity_z);
        }
    }

    if (const int rc = tracking.maybe_track_storm(); rc != 0)
    {
        return rc;
    }

    if (perf_step_active)
    {
        perf_totals.simulated_s += dt;
    }

    if (!output.shm_update())
        return 2;
    return 0;
}

LiveDomainRecyclingResult
StormTrackingController::perform_domain_recycling(const tmv::domain::StormWindowMovementDecision& decision)
{
    const auto start = PerfClock::now();
    // The environmental reservoir must match initialization, including the
    // exactly zero inactive TKE owned by two-way feedback profiles.
    const bool inactive_feedback_tke =
        run.configuration.nesting.enabled &&
        run.configuration.nesting.feedback == tmv::init::config::NestingFeedback::TwoWay &&
        run.runtime.turbulence.config.scheme_id != "tke";
    const double        environmental_tke_m2s2 = inactive_feedback_tke ? 0.0 : 0.1;
    std::vector<double> environmental_tke(static_cast<std::size_t>(run.grid.nz), environmental_tke_m2s2);
    auto prepared = tmv::domain::prepare_flat_homogeneous_domain_recycling(run, decision, environmental_tke);

    const double displacement_x_m = static_cast<double>(prepared->plan.shift_i_cells) * run.grid.dx;
    const double displacement_y_m = static_cast<double>(prepared->plan.shift_j_cells) * run.grid.dy;
    if (options.trajectory.enabled)
        trajectories.preflight_domain_recycling(displacement_x_m, displacement_y_m, prepared->post_shift_frame);

    static constexpr std::array<const char*, 8>        swath_names{{"u", "v", "w", "theta", "p", "rho", "qv", "qr"}};
    std::array<std::vector<float>, swath_names.size()> swath_values;
    for (auto& values : swath_values)
        values.reserve(prepared->plan.trailing_cells);
    const auto& plan = prepared->plan;
    for (int i = plan.physical_i_begin; i < plan.physical_i_end; ++i)
        for (int j = plan.physical_j_begin; j < plan.physical_j_end; ++j)
        {
            const bool retained =
                i >= plan.source_i_begin && i < plan.source_i_end && j >= plan.source_j_begin && j < plan.source_j_end;
            if (retained)
                continue;
            for (int k = 0; k < plan.nz; ++k)
            {
                swath_values[0].push_back(0.5f * (run.fields.u[i - 1][j][k] + run.fields.u[i][j][k]));
                swath_values[1].push_back(0.5f * (run.fields.v[i][j - 1][k] + run.fields.v[i][j][k]));
                swath_values[2].push_back(k == 0 ? 0.5f * run.fields.w[i][j][k] : 0.5f * (run.fields.w[i][j][k - 1] + run.fields.w[i][j][k]));
                swath_values[3].push_back(run.fields.theta[i][j][k]);
                swath_values[4].push_back(run.fields.p[i][j][k]);
                swath_values[5].push_back(run.fields.rho[i][j][k]);
                swath_values[6].push_back(run.fields.qv[i][j][k]);
                swath_values[7].push_back(run.fields.qr[i][j][k]);
            }
        }
    for (const auto& values : swath_values)
        if (values.size() != plan.trailing_cells)
            throw std::runtime_error("D5 centered retiring-swath capture count mismatch");

    OutputConfig swath_output = output.output_config;
    if (swath_output.format != OutputFormat::npy_3d && swath_output.format != OutputFormat::netcdf &&
        swath_output.format != OutputFormat::zfp)
        swath_output.format = OutputFormat::npy_3d;
    swath_output.preset = FieldPreset::minimal;
    swath_output.custom_fields.clear();
    swath_output.resolved_fields.clear();
    for (const char* name : swath_names)
        swath_output.resolved_fields.insert(name);
    apply_default_compression_tiers(swath_output);

    const auto& previous = run.runtime.domain_recycling_provenance;
    if (previous.sequence == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("D5 transaction sequence overflow");
    const std::uint64_t sequence       = previous.sequence + 1;
    const std::uint64_t previous_chain = previous.available ? previous.committed_chain_fnv1a64 : 0;
    if (domain_recycling_attempt == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("D5 transaction-attempt sequence overflow");
    ++domain_recycling_attempt;
    std::ostringstream transaction_name;
    transaction_name << "d5_" << checkpoints.run_id() << "_" << domain_recycling_attempt;

    nimbus::output::RetiringSwathPublication publication;
    publication.swath_root                       = std::filesystem::path(outdir) / "domain_recycling_swath";
    publication.transaction_id                   = transaction_name.str();
    publication.publication_generation           = checkpoints.run_id();
    publication.transaction_sequence             = sequence;
    publication.previous_committed_chain_fnv1a64 = previous_chain;
    publication.simulation_time_s                = run.time_s;
    publication.dx_m                             = run.grid.dx;
    publication.dy_m                             = run.grid.dy;
    publication.dz_m                             = run.grid.dz;
    publication.domain_motion_mode = tmv::init::config::domain_motion_mode_name(run.configuration.domain_motion_mode);
    publication.native_staggering  = "c_grid";
    publication.pre_shift_frame    = prepared->pre_shift_frame;
    publication.plan               = plan;
    publication.output             = swath_output;
    for (std::size_t index = 0; index < swath_names.size(); ++index)
        publication.fields.push_back({swath_names[index], &swath_values[index], true});

    nimbus::output::RetiringSwathPublicationResult publication_result;
    std::string                                    publication_error;
    if (!nimbus::output::publish_retiring_swath_precommit(publication, publication_result, publication_error))
        throw std::runtime_error("D5 retiring-swath publication failed: " + publication_error);

    tmv::domain::append_flat_homogeneous_domain_recycling_provenance(*prepared,
                                                                     publication.transaction_id,
                                                                     publication.publication_generation,
                                                                     output_format_name(publication.output.format),
                                                                     nimbus::output::retiring_swath_algorithm_id(),
                                                                     run.time_s,
                                                                     sequence,
                                                                     previous_chain,
                                                                     publication_result.event_manifest_bytes,
                                                                     publication_result.event_manifest_fnv1a64);
    const auto shift_report = tmv::domain::commit_flat_homogeneous_domain_recycling(*prepared);

    // Everything below is post-commit and either non-throwing or fail-stop.
    // Device copies must become ineligible before any post-shift validation.
    SimulationStateView state = make_simulation_state_view(run);
    if (state.compute_runtime != nullptr && state.compute_runtime->backend != nullptr)
        state.compute_runtime->backend->invalidate_all_resident_fields();
    if (options.trajectory.enabled)
        trajectories.commit_domain_recycling(displacement_x_m, displacement_y_m, prepared->post_shift_frame);
    tmv::domain::verify_flat_homogeneous_domain_recycling_after_commit(*prepared);
    // Recycling translates the current Earth-coordinate chaos realization and shifts
    // its deterministic PBL cache in the same transaction.  Reapply that
    // already-current realization before publishing density-divided
    // tendencies so newly admitted columns cannot remain temporarily
    // deterministic while retained columns remain stochastic.  This does not
    // advance the chaos clock or draw a new realization.
    apply_chaos_tendencies(state);
    tmv::rebuild_boundary_layer_tendencies_after_domain_recycling(state);
    tmv::rebuild_boundary_conditions_after_domain_recycling(state);

    if (step_radar_diagnostics_enabled())
        calculate_radar_observables();
    if (step_vorticity_diagnostics_enabled() || step_angular_momentum_diagnostics_enabled() ||
        step_pressure_diagnostics_enabled())
        compute_dynamics_diagnostics(step_vorticity_diagnostics_enabled(),
                                     step_angular_momentum_diagnostics_enabled(),
                                     step_pressure_diagnostics_enabled());
    if (!validation.validate_core_fields(make_simulation_state_view(run), "after_domain_recycling", steps, false, nullptr, false))
        throw std::runtime_error(validation.validation_error_message);

    LiveDomainRecyclingResult result;
    result.committed        = true;
    result.transaction_id   = publication.transaction_id;
    result.event_directory  = publication_result.event_directory;
    result.post_shift_frame = prepared->post_shift_frame;
    result.shift            = shift_report;
    result.shift_i_cells    = plan.shift_i_cells;
    result.shift_j_cells    = plan.shift_j_cells;
    result.elapsed_ms       = std::chrono::duration<double, std::milli>(PerfClock::now() - start).count();
    tmv::log_info("[DOMAIN RECYCLING] committed transaction=",
                  result.transaction_id,
                  ", shift_cells=(",
                  plan.shift_i_cells,
                  ",",
                  plan.shift_j_cells,
                  "), fields=",
                  shift_report.volumes.fields_shifted,
                  ", surfaces=",
                  shift_report.surfaces_shifted,
                  ", invalidations=",
                  shift_report.runtime_invalidations_committed,
                  ", runtime_commits=",
                  shift_report.runtime_commits_committed,
                  ", elapsed_ms=",
                  result.elapsed_ms);
    return result;
}

int StormTrackingController::maybe_track_storm()
{
    const auto& config = run.configuration.storm_tracking;
    if (!config.enabled || next_storm_tracking_time_s < 0.0 || run.time_s + 1.0e-9 < next_storm_tracking_time_s)
    {
        return 0;
    }

    const double scheduled_time_s = next_storm_tracking_time_s;
    next_storm_tracking_time_s    = next_cadence_after(run.time_s, config.interval_s);
    const auto compute_start      = PerfClock::now();

    try
    {
        const SimulationStateView state = storm_tracking_uses_nested_child
            ? make_simulation_state_view(
                  run.nested_execution.child,
                  run.configuration,
                  &run.runtime.compute,
                  &run.runtime.parity)
            : make_simulation_state_view(run);
        if (storm_tracking_uses_nested_child &&
            state.time_s != run.time_s)
            throw std::runtime_error(
                "storm tracker requires a synchronized nested-child endpoint");
        const PhysicalCellDomain  domain       = physical_scalar_domain(state.grid);
        const std::size_t         nx           = static_cast<std::size_t>(domain.nx());
        const std::size_t         ny           = static_cast<std::size_t>(domain.ny());
        const std::size_t         column_count = nx * ny;
        if (column_count == 0 || storm_maximum_updraft_ms.size() != column_count ||
            storm_condensate_path_kgm2.size() != column_count || storm_updraft_helicity_m2s2.size() != column_count)
        {
            throw std::runtime_error("storm tracker workspace does not match the physical domain");
        }
        if (state.domain_frame == nullptr)
            throw std::runtime_error("storm tracker coordinate frame is unavailable");
        const tmv::domain::HorizontalDomainFrame tracking_snapshot_frame =
            *state.domain_frame;
        const tmv::domain::HorizontalPoint earth_origin =
            tracking_snapshot_frame.earth_origin_at(run.time_s);
        const bool detector_coordinates_are_earth =
            storm_tracking_uses_nested_child;
        const double detector_west_face_m =
            static_cast<double>(domain.i_begin) * state.grid.dr +
            (detector_coordinates_are_earth ? earth_origin.x_m : 0.0);
        const double detector_south_face_m =
            static_cast<double>(domain.j_begin) * state.grid.dtheta +
            (detector_coordinates_are_earth ? earth_origin.y_m : 0.0);

        const bool                 terrain_active   = tmv::physical_grid_metrics::terrain_is_active(state);
        const TerrainMetrics3D*    terrain_metrics  = tmv::physical_grid_metrics::active_terrain_metrics(state);
        const std::vector<double>* terrain_ground_w = nullptr;
        if (terrain_active && run.configuration.stagger_type == StaggerType::CGrid)
        {
            terrain_ground_w = output.refresh_cgrid_ground_vertical_velocity(state);
            if (terrain_metrics == nullptr || terrain_ground_w == nullptr)
            {
                throw std::runtime_error("storm tracker terrain workspace is unavailable");
            }
        }

        const tmv::physical_grid_metrics::ControlVolumeMassView mass_view(state);
        const double                                            horizontal_area_m2 = state.grid.dr * state.grid.dtheta;
        const std::vector<double>* flat_z_center = state.grid.geometry.z.empty() ? nullptr : &state.grid.geometry.z;
        const std::vector<double>* flat_z_interface =
            state.grid.geometry.z_faces.empty() ? nullptr : &state.grid.geometry.z_faces;
        tmv::diagnostics::compute_storm_updraft_helicity_columns(
            tmv::diagnostics::StormRotationGridView{
                &state.fields.u,
                &state.fields.v,
                &state.fields.w,
                domain,
                run.configuration.stagger_type,
                state.grid.dr,
                state.grid.dtheta,
                state.grid.dz,
                terrain_metrics,
                flat_z_center,
                flat_z_interface,
                terrain_ground_w,
            },
            tmv::diagnostics::StormRotationLayer{
                config.updraft_helicity_bottom_m_agl,
                config.updraft_helicity_top_m_agl,
            },
            storm_updraft_helicity_m2s2);

#pragma omp parallel for collapse(2) if (column_count >= 4096)
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
            {
                const std::size_t index =
                    static_cast<std::size_t>(i - domain.i_begin) * ny + static_cast<std::size_t>(j - domain.j_begin);
                double maximum_updraft_ms   = -std::numeric_limits<double>::infinity();
                double condensate_path_kgm2 = 0.0;
                bool   valid_column         = true;
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    double centered_w_ms = state.fields.w(i, j, k);
                    if (run.configuration.stagger_type == StaggerType::CGrid)
                    {
                        const double lower_w_ms =
                            k == 0 ? (terrain_active ? (*terrain_ground_w)[static_cast<std::size_t>(i) *
                                                                               static_cast<std::size_t>(state.grid.nth) +
                                                                           static_cast<std::size_t>(j)]
                                                     : 0.0)
                                   : static_cast<double>(state.fields.w(i, j, k - 1));
                        centered_w_ms = 0.5 * (lower_w_ms + static_cast<double>(state.fields.w(i, j, k)));
                    }

                    const double species[] = {
                        state.fields.qc(i, j, k),
                        state.fields.qr(i, j, k),
                        state.fields.qi(i, j, k),
                        state.fields.qs(i, j, k),
                        state.fields.qh(i, j, k),
                        state.fields.qg(i, j, k),
                    };
                    double condensate_kgkg = 0.0;
                    for (double mixing_ratio : species)
                    {
                        valid_column = valid_column && std::isfinite(mixing_ratio) && mixing_ratio >= 0.0;
                        condensate_kgkg += mixing_ratio;
                    }
                    const double density_kgm3 = state.fields.rho(i, j, k);
                    valid_column       = valid_column && std::isfinite(centered_w_ms) && std::isfinite(density_kgm3) &&
                                         density_kgm3 > 0.0;
                    maximum_updraft_ms = std::max(maximum_updraft_ms, centered_w_ms);
                    condensate_path_kgm2 +=
                        density_kgm3 * condensate_kgkg * mass_view.scalar_volume_m3(i, j, k) / horizontal_area_m2;
                }
                storm_maximum_updraft_ms[index] =
                    valid_column ? maximum_updraft_ms : std::numeric_limits<double>::quiet_NaN();
                storm_condensate_path_kgm2[index] =
                    valid_column ? condensate_path_kgm2 : std::numeric_limits<double>::quiet_NaN();
            }

        const tmv::diagnostics::StormFeatureThresholds thresholds{
            config.minimum_updraft_ms,
            config.minimum_condensate_path_kgm2,
            config.minimum_absolute_updraft_helicity_m2s2,
            static_cast<std::size_t>(config.minimum_component_cells),
            config.minimum_core_area_m2,
            config.minimum_confidence,
        };
        const tmv::diagnostics::StormColumnSignalView signals{
            nx,
            ny,
            detector_west_face_m,
            detector_south_face_m,
            state.grid.dr,
            state.grid.dtheta,
            &storm_maximum_updraft_ms,
            &storm_condensate_path_kgm2,
            &storm_updraft_helicity_m2s2,
        };
        auto&       partition_runtime  = run.runtime;
        if (partition_runtime.storm_selected_partition_available &&
            (partition_runtime.storm_selected_partition_nx != nx ||
             partition_runtime.storm_selected_partition_ny != ny))
            throw std::runtime_error(
                "stored storm partition does not match the authoritative tracking grid");
        const auto* previous_partition = partition_runtime.storm_selected_partition_available
                                             ? &partition_runtime.storm_selected_partition
                                             : nullptr;
        const tmv::diagnostics::StormFeatureDetection detection =
            storm_feature_detector.detect(signals, thresholds, previous_partition);
        const tmv::diagnostics::StormContinuityThresholds continuity_thresholds{
            static_cast<std::size_t>(config.minimum_continuity_samples),
            static_cast<std::size_t>(config.lost_after_samples),
            config.maximum_motion_speed_ms,
            config.maximum_motion_acceleration_ms2,
            std::hypot(state.grid.dr, state.grid.dtheta) / std::sqrt(12.0),
            static_cast<std::size_t>(config.topology_confirmation_samples),
        };
        tmv::diagnostics::StormContinuityState& continuity = run.runtime.storm_tracking;
        const tmv::diagnostics::StormContinuityState prior_continuity = continuity;
        tmv::diagnostics::update_storm_continuity(
            continuity, detection, run.time_s, continuity_thresholds, storm_branch_association);
        const tmv::diagnostics::StormEnvelopeFaceMotion envelope_face_motion =
            tmv::diagnostics::derive_storm_envelope_face_motion(
                prior_continuity, detection, continuity, run.time_s);
        if (continuity.selected_candidate_available)
        {
            storm_feature_detector.copy_partition(continuity.selected_component_label,
                                                  partition_runtime.storm_selected_partition);
            partition_runtime.storm_selected_partition_available = true;
            partition_runtime.storm_selected_partition_nx        = nx;
            partition_runtime.storm_selected_partition_ny        = ny;
        }
        else if (continuity.status == tmv::diagnostics::StormContinuityStatus::Lost ||
                 continuity.status == tmv::diagnostics::StormContinuityStatus::Uninitialized)
        {
            partition_runtime.storm_selected_partition_available = false;
            partition_runtime.storm_selected_partition_nx        = 0;
            partition_runtime.storm_selected_partition_ny        = 0;
        }
        const double compute_ms = std::chrono::duration<double, std::milli>(PerfClock::now() - compute_start).count();
        const bool   continuity_available = continuity.sample_count > 0;
        const bool   motion_available     = continuity.motion_available;
        const bool   selected_reported_candidate =
            continuity.selected_candidate_available &&
            continuity.selected_candidate_rank < detection.reported_branch_candidate_count;
        const auto* selected_branch =
            selected_reported_candidate ? &detection.branch_candidates[continuity.selected_candidate_rank] : nullptr;
        const bool   has_feature = selected_branch != nullptr;
        const double envelope_west_m =
            selected_branch != nullptr ? selected_branch->envelope_west_m : detection.envelope_west_m;
        const double envelope_east_m =
            selected_branch != nullptr ? selected_branch->envelope_east_m : detection.envelope_east_m;
        const double envelope_south_m =
            selected_branch != nullptr ? selected_branch->envelope_south_m : detection.envelope_south_m;
        const double envelope_north_m =
            selected_branch != nullptr ? selected_branch->envelope_north_m : detection.envelope_north_m;
        const double reported_core_x_m =
            selected_branch != nullptr ? selected_branch->core_centroid_x_m : detection.core_centroid_x_m;
        const double reported_core_y_m =
            selected_branch != nullptr ? selected_branch->core_centroid_y_m : detection.core_centroid_y_m;
        const std::size_t reported_component_cells =
            selected_branch != nullptr ? selected_branch->component_cells : detection.selected_component_cells;
        const std::size_t reported_core_cells =
            selected_branch != nullptr ? selected_branch->core_cells : detection.selected_core_cells;
        const std::size_t reported_rotating_core_cells =
            selected_branch != nullptr ? selected_branch->rotating_core_cells : detection.selected_rotating_core_cells;
        const double reported_maximum_updraft_ms =
            selected_branch != nullptr ? selected_branch->maximum_updraft_ms : detection.maximum_updraft_ms;
        const double reported_condensate_mass_kg =
            selected_branch != nullptr ? selected_branch->condensate_mass_kg : detection.condensate_mass_kg;
        const double reported_centroid_x_m =
            selected_branch != nullptr ? selected_branch->centroid_x_m : detection.centroid_x_m;
        const double reported_centroid_y_m =
            selected_branch != nullptr ? selected_branch->centroid_y_m : detection.centroid_y_m;
        const double reported_maximum_absolute_uh = selected_branch != nullptr
                                                        ? selected_branch->maximum_absolute_updraft_helicity_m2s2
                                                        : detection.maximum_absolute_updraft_helicity_m2s2;
        const double reported_signed_mean_uh = selected_branch != nullptr
                                                   ? selected_branch->signed_rotating_core_mean_updraft_helicity_m2s2
                                                   : detection.signed_core_mean_updraft_helicity_m2s2;
        const double distance_west_m =
            selected_branch != nullptr ? selected_branch->distance_west_m : detection.distance_west_m;
        const double distance_east_m =
            selected_branch != nullptr ? selected_branch->distance_east_m : detection.distance_east_m;
        const double distance_south_m =
            selected_branch != nullptr ? selected_branch->distance_south_m : detection.distance_south_m;
        const double distance_north_m =
            selected_branch != nullptr ? selected_branch->distance_north_m : detection.distance_north_m;
        const double nearest_boundary_m =
            has_feature ? std::min({distance_west_m, distance_east_m, distance_south_m, distance_north_m}) : 0.0;
        const bool          boundary_contact      = has_feature && nearest_boundary_m <= 0.0;
        const bool          inner_margin_breached = has_feature && nearest_boundary_m < config.inner_margin_m;
        const std::uint64_t completed_step        = static_cast<std::uint64_t>(std::max(0, steps)) + 1;
        const double        prediction_horizon_s  = std::max(0.0, next_storm_tracking_time_s - run.time_s);
        const double accepted_x_m = continuity_available ? continuity.samples[continuity.sample_count - 1].x_m : 0.0;
        const double accepted_y_m = continuity_available ? continuity.samples[continuity.sample_count - 1].y_m : 0.0;
        const double predicted_x_m =
            motion_available ? accepted_x_m + continuity.velocity_x_ms * prediction_horizon_s : 0.0;
        const double predicted_y_m =
            motion_available ? accepted_y_m + continuity.velocity_y_ms * prediction_horizon_s : 0.0;
        const tmv::domain::HorizontalPoint predicted_earth_origin =
            tracking_snapshot_frame.earth_origin_at(
                run.time_s + prediction_horizon_s);
        const double earth_coordinate_offset_x_m =
            detector_coordinates_are_earth ? 0.0 : earth_origin.x_m;
        const double earth_coordinate_offset_y_m =
            detector_coordinates_are_earth ? 0.0 : earth_origin.y_m;
        const double grid_coordinate_offset_x_m =
            detector_coordinates_are_earth ? earth_origin.x_m : 0.0;
        const double grid_coordinate_offset_y_m =
            detector_coordinates_are_earth ? earth_origin.y_m : 0.0;
        const double predicted_grid_coordinate_offset_x_m =
            detector_coordinates_are_earth ? predicted_earth_origin.x_m : 0.0;
        const double predicted_grid_coordinate_offset_y_m =
            detector_coordinates_are_earth ? predicted_earth_origin.y_m : 0.0;
        const double earth_velocity_x_ms = continuity.velocity_x_ms +
            (detector_coordinates_are_earth
                 ? 0.0
                 : tracking_snapshot_frame.velocity_ms.u_ms);
        const double earth_velocity_y_ms = continuity.velocity_y_ms +
            (detector_coordinates_are_earth
                 ? 0.0
                 : tracking_snapshot_frame.velocity_ms.v_ms);
        const double grid_velocity_x_ms = continuity.velocity_x_ms -
            (detector_coordinates_are_earth
                 ? tracking_snapshot_frame.velocity_ms.u_ms
                 : 0.0);
        const double grid_velocity_y_ms = continuity.velocity_y_ms -
            (detector_coordinates_are_earth
                 ? tracking_snapshot_frame.velocity_ms.v_ms
                 : 0.0);
        const double earth_speed_ms      = std::hypot(earth_velocity_x_ms, earth_velocity_y_ms);
        const bool   moving_west         = motion_available && grid_velocity_x_ms < 0.0;
        const bool   moving_east         = motion_available && grid_velocity_x_ms > 0.0;
        const bool   moving_south        = motion_available && grid_velocity_y_ms < 0.0;
        const bool   moving_north        = motion_available && grid_velocity_y_ms > 0.0;
        const double time_west_s    = moving_west ? std::max(0.0, distance_west_m) / -grid_velocity_x_ms : 0.0;
        const double time_east_s    = moving_east ? std::max(0.0, distance_east_m) / grid_velocity_x_ms : 0.0;
        const double time_south_s   = moving_south ? std::max(0.0, distance_south_m) / -grid_velocity_y_ms : 0.0;
        const double time_north_s   = moving_north ? std::max(0.0, distance_north_m) / grid_velocity_y_ms : 0.0;
        double       nearest_time_s = std::numeric_limits<double>::infinity();
        if (moving_west)
            nearest_time_s = std::min(nearest_time_s, time_west_s);
        if (moving_east)
            nearest_time_s = std::min(nearest_time_s, time_east_s);
        if (moving_south)
            nearest_time_s = std::min(nearest_time_s, time_south_s);
        if (moving_north)
            nearest_time_s = std::min(nearest_time_s, time_north_s);
        const bool nearest_time_available = std::isfinite(nearest_time_s);

        constexpr std::size_t movement_confirmation_samples = 2;
        const bool            movement_authority_enabled    = config.maximum_shift_cells > 0;
        const bool nested_movement_authority =
            run.configuration.domain_motion_mode ==
                tmv::init::config::DomainMotionMode::MovingNest &&
            run.configuration.nesting.enabled;
        const double movement_dx_m = nested_movement_authority
            ? run.nested_execution.relationship.parent_dx_m
            : state.grid.dr;
        const double movement_dy_m = nested_movement_authority
            ? run.nested_execution.relationship.parent_dy_m
            : state.grid.dtheta;
        const double movement_release_margin_m =
            tmv::domain::storm_window_release_margin_m(
                config.inner_margin_m, movement_dx_m, movement_dy_m);
        auto&        movement_state            = run.runtime.storm_window_movement;
        const tmv::domain::StormWindowMovementState movement_state_before =
            movement_state;
        std::optional<tmv::domain::StormWindowMovementDecision> movement_decision;
        std::optional<tmv::domain::StormWindowMovementEvidence>
            nested_movement_evidence;
        NestedChildMovementAuthorityReportV1 nested_movement_result;
        double nested_movement_elapsed_ms = 0.0;
        double nested_relaxation_width_x_m = 0.0;
        double nested_relaxation_width_y_m = 0.0;
        bool nested_trusted_interior_available = false;
        bool nested_trusted_interior = false;
        bool nested_scientific_invalid = false;
        bool nested_scientific_pending = false;
        const char* nested_scientific_reason = "not_applicable";
        if (movement_authority_enabled)
        {
            const tmv::domain::StormWindowMovementPolicy movement_policy{
                config.inner_margin_m,
                movement_release_margin_m,
                movement_dx_m,
                movement_dy_m,
                static_cast<std::size_t>(config.maximum_shift_cells),
                config.cooldown_s,
                config.minimum_confidence,
                movement_confirmation_samples,
                nested_movement_authority
                    ? run.nested_execution.relationship
                          .relaxation_width_child_cells *
                          run.nested_execution.relationship
                              .child_dx_m
                    : 0.0,
                nested_movement_authority
                    ? run.nested_execution.relationship
                          .relaxation_width_child_cells *
                          run.nested_execution.relationship
                              .child_dy_m
                    : 0.0,
            };
            if (nested_movement_authority)
            {
                const NestedTrackedEnvelopeV1 nested_envelope =
                    detector_coordinates_are_earth
                        ? prepare_nested_tracked_envelope_earth(
                              detection, continuity, envelope_face_motion,
                              run.time_s)
                        : prepare_nested_tracked_envelope(
                              detection, continuity, envelope_face_motion,
                              run.time_s,
                              earth_origin.x_m, earth_origin.y_m,
                              tracking_snapshot_frame.velocity_ms.u_ms,
                              tracking_snapshot_frame.velocity_ms.v_ms);
                nested_movement_evidence =
                    prepare_nested_child_movement_evidence(
                        run, nested_envelope);
                const bool authoritative_track_interrupted =
                    prior_continuity.status ==
                        tmv::diagnostics::StormContinuityStatus::Tracking &&
                    (!nested_movement_evidence->selected_feature_available ||
                     !nested_movement_evidence->continuity_authoritative);
                if (authoritative_track_interrupted)
                {
                    nested_scientific_invalid = true;
                    nested_scientific_reason =
                        "authoritative_track_interrupted";
                }
                const auto& relationship = run
                    .nested_execution.relationship;
                nested_relaxation_width_x_m =
                    relationship.relaxation_width_child_cells *
                    relationship.child_dx_m;
                nested_relaxation_width_y_m =
                    relationship.relaxation_width_child_cells *
                    relationship.child_dy_m;
                if (nested_movement_evidence->selected_feature_available)
                {
                    nested_trusted_interior_available = true;
                    const bool negative_clearance =
                        nested_movement_evidence->distance_west_m < 0.0 ||
                        nested_movement_evidence->distance_east_m < 0.0 ||
                        nested_movement_evidence->distance_south_m < 0.0 ||
                        nested_movement_evidence->distance_north_m < 0.0;
                    nested_trusted_interior =
                        nested_movement_evidence->distance_west_m >=
                            nested_relaxation_width_x_m &&
                        nested_movement_evidence->distance_east_m >=
                            nested_relaxation_width_x_m &&
                        nested_movement_evidence->distance_south_m >=
                            nested_relaxation_width_y_m &&
                        nested_movement_evidence->distance_north_m >=
                            nested_relaxation_width_y_m;
                    if (negative_clearance || !nested_trusted_interior)
                    {
                        nested_scientific_invalid = true;
                        nested_scientific_reason = negative_clearance
                            ? "negative_clearance"
                            : "relaxation_zone_entry";
                    }
                }
                const auto movement_start = PerfClock::now();
                auto diagnostic_authority = movement_state;
                const auto diagnostic_decision =
                    tmv::domain::evaluate_storm_window_movement(
                        diagnostic_authority,
                        *nested_movement_evidence,
                        movement_policy);
                if (!nested_scientific_invalid &&
                    diagnostic_decision.reason ==
                        tmv::domain::StormWindowMovementReason::EmergencyContainment)
                {
                    nested_scientific_invalid = true;
                    nested_scientific_reason =
                        "emergency_containment_diagnostic_only";
                }
                if (nested_scientific_invalid)
                {
                    nested_movement_result.decision = diagnostic_decision;
                }
                else
                {
                    evaluate_and_move_nested_child_from_tracker(
                        run, nested_envelope,
                        movement_policy,
                        {run.configuration.nesting
                             .relaxation_e_folding_time_s},
                        nested_movement_result);
                }
                nested_movement_elapsed_ms =
                    std::chrono::duration<double, std::milli>(
                        PerfClock::now() - movement_start).count();
                movement_decision = nested_movement_result.decision;
            }
            else
            {
                tmv::domain::StormWindowMovementEvidence movement_evidence{
                    run.time_s,
                    has_feature,
                    continuity.status == tmv::diagnostics::StormContinuityStatus::Tracking,
                    motion_available,
                    continuity.status == tmv::diagnostics::StormContinuityStatus::Ambiguous ||
                        continuity.reason == tmv::diagnostics::StormContinuityReason::SpatialAmbiguity,
                    detection.branch_candidate_overflow ||
                        continuity.reason == tmv::diagnostics::StormContinuityReason::BranchCandidateOverflow,
                    continuity.topology_reorganization,
                    continuity.continuity_confidence,
                    distance_west_m,
                    distance_east_m,
                    distance_south_m,
                    distance_north_m,
                    continuity.velocity_x_ms,
                    continuity.velocity_y_ms,
                    envelope_face_motion.available,
                    envelope_face_motion.interval_s,
                    envelope_face_motion.west_coordinate_velocity_ms,
                    envelope_face_motion.east_coordinate_velocity_ms,
                    envelope_face_motion.south_coordinate_velocity_ms,
                    envelope_face_motion.north_coordinate_velocity_ms,
                };
                if (config.recycle_domain && run.nested_execution.relationship_ready)
                {
                    movement_evidence = tmv::domain::with_nested_parent_recycling_bounds(
                        run.nested_execution.relationship, movement_evidence);
                    nested_movement_evidence = movement_evidence;
                }
                movement_decision = tmv::domain::evaluate_storm_window_movement(
                    movement_state, movement_evidence, movement_policy);
            }
        }
        LiveDomainRecyclingResult recycling_result;
        if (!nested_movement_authority && config.recycle_domain && movement_decision &&
            movement_decision->status == tmv::domain::StormWindowMovementStatus::Requested)
            recycling_result = perform_domain_recycling(*movement_decision);
        const bool movement_committed = recycling_result.committed ||
            nested_movement_result.movement_committed;
        if (nested_movement_authority && nested_movement_evidence &&
            nested_movement_evidence->selected_feature_available &&
            !nested_scientific_invalid && movement_decision)
        {
            const bool breached =
                nested_movement_evidence->distance_west_m <
                    config.inner_margin_m ||
                nested_movement_evidence->distance_east_m <
                    config.inner_margin_m ||
                nested_movement_evidence->distance_south_m <
                    config.inner_margin_m ||
                nested_movement_evidence->distance_north_m <
                    config.inner_margin_m;
            if (movement_decision->reason ==
                tmv::domain::StormWindowMovementReason::EmergencyContainment)
            {
                nested_scientific_invalid = true;
                nested_scientific_reason =
                    "emergency_containment_diagnostic_only";
            }
            else if (breached)
            {
                const bool first_bounded_confirmation =
                    tmv::domain::storm_window_started_new_confirmation(
                        movement_state_before, *movement_decision);
                const bool completed_ordinary_relocation =
                    movement_decision->status ==
                        tmv::domain::StormWindowMovementStatus::Requested &&
                    (movement_decision->reason ==
                         tmv::domain::StormWindowMovementReason::RequestReady ||
                     movement_decision->reason ==
                         tmv::domain::StormWindowMovementReason::ProjectedRelaxationRisk ||
                     movement_decision->reason ==
                         tmv::domain::StormWindowMovementReason::TrustedInteriorContainment) &&
                    nested_movement_result.movement_committed &&
                    movement_state.armed;
                const bool trusted_interior_contained =
                    movement_decision->status ==
                        tmv::domain::StormWindowMovementStatus::Ineligible &&
                    (movement_decision->reason ==
                         tmv::domain::StormWindowMovementReason::TrustedInteriorContained ||
                     movement_decision->reason ==
                         tmv::domain::StormWindowMovementReason::ParentStencilLimitedContained);
                if (first_bounded_confirmation)
                {
                    nested_scientific_pending = true;
                    nested_scientific_reason =
                        "confirmation_pending_outside_relaxation";
                }
                else if (!completed_ordinary_relocation &&
                         !trusted_interior_contained)
                {
                    nested_scientific_invalid = true;
                    nested_scientific_reason =
                        "unresolved_breached_face";
                }
            }
            else if (nested_movement_result.movement_committed &&
                     !movement_state.armed)
            {
                nested_scientific_invalid = true;
                nested_scientific_reason =
                    "post_shift_release_not_recovered";
            }
        }
        if (nested_movement_authority && !nested_scientific_invalid &&
            !nested_scientific_pending)
            nested_scientific_reason = nested_trusted_interior_available
                ? "none"
                : "no_selected_feature";

        const auto json_optional_number = [](std::ostringstream& out, bool available, double value)
        {
            if (available)
                out << value;
            else
                out << "null";
        };
        const auto json_optional_count = [](std::ostringstream& out, bool available, std::size_t value)
        {
            if (available)
                out << value;
            else
                out << "null";
        };
        const auto json_optional_bool = [](std::ostringstream& out, bool available, bool value)
        {
            if (available)
                out << (value ? "true" : "false");
            else
                out << "null";
        };
        std::ostringstream json;
        json << std::setprecision(17) << "{\n"
             << "  \"schema\": \"nimbus.storm_feature\",\n"
             << "  \"schema_version\": 12,\n"
             << "  \"algorithm\": \"" << tmv::diagnostics::storm_feature_detector_algorithm_id() << "\",\n"
             << "  \"updraft_helicity_algorithm\": \"" << tmv::diagnostics::storm_updraft_helicity_algorithm_id()
             << "\",\n"
             << "  \"continuity_algorithm\": \"" << tmv::diagnostics::storm_continuity_algorithm_id() << "\",\n"
             << "  \"mode\": \""
             << (nested_movement_authority
                     ? "tracking_with_one_way_moving_nest"
                     : (config.recycle_domain
                            ? (tmv::physical_grid_metrics::terrain_is_active(
                                   state)
                                   ? "tracking_with_earth_fixed_terrain_cgrid_recycling"
                                   : "tracking_with_flat_cgrid_recycling")
                            : "tracking_with_diagnostic_movement_requests"))
             << "\",\n"
             << "  \"tracking_source\": \""
             << (storm_tracking_uses_nested_child
                     ? "synchronized_nested_child"
                     : "root_grid")
             << "\",\n"
             << "  \"detector_coordinate_frame\": \""
             << (detector_coordinates_are_earth
                     ? "earth_absolute"
                     : "grid_relative")
             << "\",\n"
             << "  \"coordinate_frame\": \"earth_with_grid_frame_v2\",\n"
             << "  \"coordinate_origin\": \""
             << (detector_coordinates_are_earth
                     ? "earth_absolute_detector_coordinates"
                     : "earth_grid_origin_at_simulation_time")
             << "\",\n"
             << "  \"horizontal_frame\": {\"contract\": "
             << (detector_coordinates_are_earth
                     ? "\"integer_parent_cell_relocation_earth_detector_v1\", "
                     : "\"constant_translation_earth_wind_grid_transport_v1\", ")
             << "\"motion_mode\": \""
             << tmv::init::config::domain_motion_mode_name(run.configuration.domain_motion_mode)
             << "\", \"grid_origin_t0_earth_m\": {\"x\": " << tracking_snapshot_frame.earth_origin_at_t0_m.x_m
             << ", \"y\": " << tracking_snapshot_frame.earth_origin_at_t0_m.y_m
             << "}, \"grid_origin_earth_m\": {\"x\": " << earth_origin.x_m << ", \"y\": " << earth_origin.y_m
             << "}, \"frame_velocity_earth_ms\": {\"x\": " << tracking_snapshot_frame.velocity_ms.u_ms
             << ", \"y\": " << tracking_snapshot_frame.velocity_ms.v_ms << "}},\n"
             << "  \"state_mutation\": " << (movement_committed ? "true" : "false") << ",\n"
             << "  \"prognostic_state_mutation\": " << (movement_committed ? "true" : "false") << ",\n"
             << "  \"diagnostic_history_mutation\": true,\n"
             << "  \"motion_available\": " << (motion_available ? "true" : "false") << ",\n"
             << "  \"continuity_available\": " << (continuity_available ? "true" : "false") << ",\n"
             << "  \"completed_step\": " << completed_step << ",\n"
             << "  \"simulation_time_s\": " << run.time_s << ",\n"
             << "  \"scheduled_time_s\": " << scheduled_time_s << ",\n"
             << "  \"sampling_lag_s\": " << std::max(0.0, run.time_s - scheduled_time_s) << ",\n"
             << "  \"compute_ms\": " << compute_ms << ",\n"
             << "  \"status\": \"" << tmv::diagnostics::storm_detection_status_name(detection.status) << "\",\n"
             << "  \"snapshot_primary_confidence\": " << detection.confidence << ",\n"
             << "  \"confidence\": ";
        json_optional_number(json, has_feature, has_feature ? selected_branch->score_fraction : 0.0);
        json << ",\n  \"envelope_component_count\": " << detection.envelope_component_count << ",\n"
             << "  \"candidate_component_count\": " << detection.candidate_component_count << ",\n"
             << "  \"reported_branch_candidate_count\": " << detection.reported_branch_candidate_count << ",\n"
             << "  \"branch_candidate_overflow\": " << (detection.branch_candidate_overflow ? "true" : "false") << ",\n"
             << "  \"previous_selected_partition_available\": "
             << (detection.previous_partition_available ? "true" : "false") << ",\n"
             << "  \"selected_feature_available\": " << (has_feature ? "true" : "false") << ",\n"
             << "  \"selected_feature_role\": " << (has_feature ? "\"continuity_selected_candidate\"" : "null") << ",\n"
             << "  \"branch_candidates\": [";
        for (std::size_t rank = 0; rank < detection.reported_branch_candidate_count; ++rank)
        {
            const auto& candidate = detection.branch_candidates[rank];
            if (rank > 0)
                json << ",";
            json << "\n    {\"rank\": " << rank << ", \"component_label\": " << candidate.component_label
                 << ", \"component_cells\": " << candidate.component_cells
                 << ", \"core_cells\": " << candidate.core_cells
                 << ", \"rotating_core_cells\": " << candidate.rotating_core_cells
                 << ", \"score_fraction\": " << candidate.score_fraction
                 << ", \"previous_partition_overlap_fraction\": " << candidate.previous_partition_overlap_fraction
                 << ", \"maximum_updraft_ms\": " << candidate.maximum_updraft_ms
                 << ", \"condensate_mass_kg\": " << candidate.condensate_mass_kg
                 << ", \"centroid_m\": {\"x\": " << candidate.centroid_x_m + earth_coordinate_offset_x_m
                 << ", \"y\": " << candidate.centroid_y_m + earth_coordinate_offset_y_m << "}"
                 << ", \"grid_centroid_m\": {\"x\": " << candidate.centroid_x_m - grid_coordinate_offset_x_m << ", \"y\": " << candidate.centroid_y_m - grid_coordinate_offset_y_m
                 << "}"
                 << ", \"core_centroid_m\": {\"x\": " << candidate.core_centroid_x_m + earth_coordinate_offset_x_m
                 << ", \"y\": " << candidate.core_centroid_y_m + earth_coordinate_offset_y_m << "}"
                 << ", \"grid_core_centroid_m\": {\"x\": " << candidate.core_centroid_x_m - grid_coordinate_offset_x_m
                 << ", \"y\": " << candidate.core_centroid_y_m - grid_coordinate_offset_y_m << "}"
                 << ", \"separation_from_primary_m\": " << candidate.separation_from_primary_m
                 << ", \"envelope_faces_m\": {\"west\": " << candidate.envelope_west_m + earth_coordinate_offset_x_m
                 << ", \"east\": " << candidate.envelope_east_m + earth_coordinate_offset_x_m
                 << ", \"south\": " << candidate.envelope_south_m + earth_coordinate_offset_y_m
                 << ", \"north\": " << candidate.envelope_north_m + earth_coordinate_offset_y_m << "}"
                 << ", \"grid_envelope_faces_m\": {\"west\": " << candidate.envelope_west_m - grid_coordinate_offset_x_m
                 << ", \"east\": " << candidate.envelope_east_m - grid_coordinate_offset_x_m << ", \"south\": " << candidate.envelope_south_m - grid_coordinate_offset_y_m
                 << ", \"north\": " << candidate.envelope_north_m - grid_coordinate_offset_y_m << "}"
                 << ", \"distance_to_physical_boundary_m\": {\"west\": " << candidate.distance_west_m
                 << ", \"east\": " << candidate.distance_east_m << ", \"south\": " << candidate.distance_south_m
                 << ", \"north\": " << candidate.distance_north_m << "}"
                 << ", \"rotation_class\": \"" << tmv::diagnostics::storm_rotation_class_name(candidate.rotation_class)
                 << "\""
                 << ", \"maximum_absolute_updraft_helicity_m2s2\": " << candidate.maximum_absolute_updraft_helicity_m2s2
                 << ", \"signed_rotating_core_mean_updraft_helicity_m2s2\": ";
            json_optional_number(
                json, candidate.rotating_core_cells > 0, candidate.signed_rotating_core_mean_updraft_helicity_m2s2);
            json << "}";
        }
        if (detection.reported_branch_candidate_count > 0)
            json << "\n  ";
        json << "],\n"
             << "  \"selected_component_cells\": ";
        json_optional_count(json, has_feature, reported_component_cells);
        json << ",\n  \"selected_core_cells\": ";
        json_optional_count(json, has_feature, reported_core_cells);
        json << ",\n  \"selected_rotating_core_cells\": ";
        json_optional_count(json, has_feature, reported_rotating_core_cells);
        json << ",\n  \"maximum_updraft_ms\": ";
        json_optional_number(json, has_feature, reported_maximum_updraft_ms);
        json << ",\n  \"rotation\": {\n"
             << "    \"definition\": \"integral_w_times_vertical_vorticity_dz\",\n"
             << "    \"layer_reference\": \"above_ground_level\",\n"
             << "    \"layer_bottom_m_agl\": " << config.updraft_helicity_bottom_m_agl << ",\n"
             << "    \"layer_top_m_agl\": " << config.updraft_helicity_top_m_agl << ",\n"
             << "    \"sign_retained\": true,\n"
             << "    \"maximum_absolute_updraft_helicity_m2s2\": ";
        json_optional_number(json, has_feature, reported_maximum_absolute_uh);
        json << ",\n    \"signed_rotating_core_mean_updraft_helicity_m2s2\": ";
        json_optional_number(json, has_feature && reported_rotating_core_cells > 0, reported_signed_mean_uh);
        json << "\n  }";
        json << ",\n  \"condensate_mass_kg\": ";
        json_optional_number(json, has_feature, reported_condensate_mass_kg);
        json << ",\n  \"centroid_m\": {\"x\": ";
        json_optional_number(json, has_feature, reported_centroid_x_m + earth_coordinate_offset_x_m);
        json << ", \"y\": ";
        json_optional_number(json, has_feature, reported_centroid_y_m + earth_coordinate_offset_y_m);
        json << "},\n  \"grid_centroid_m\": {\"x\": ";
        json_optional_number(json, has_feature, reported_centroid_x_m - grid_coordinate_offset_x_m);
        json << ", \"y\": ";
        json_optional_number(json, has_feature, reported_centroid_y_m - grid_coordinate_offset_y_m);
        json << "},\n  \"core_centroid_m\": {\"x\": ";
        json_optional_number(json, has_feature, reported_core_x_m + earth_coordinate_offset_x_m);
        json << ", \"y\": ";
        json_optional_number(json, has_feature, reported_core_y_m + earth_coordinate_offset_y_m);
        json << "},\n  \"grid_core_centroid_m\": {\"x\": ";
        json_optional_number(json, has_feature, reported_core_x_m - grid_coordinate_offset_x_m);
        json << ", \"y\": ";
        json_optional_number(json, has_feature, reported_core_y_m - grid_coordinate_offset_y_m);
        json << "},\n  \"envelope_faces_m\": {\"west\": ";
        json_optional_number(json, has_feature, envelope_west_m + earth_coordinate_offset_x_m);
        json << ", \"east\": ";
        json_optional_number(json, has_feature, envelope_east_m + earth_coordinate_offset_x_m);
        json << ", \"south\": ";
        json_optional_number(json, has_feature, envelope_south_m + earth_coordinate_offset_y_m);
        json << ", \"north\": ";
        json_optional_number(json, has_feature, envelope_north_m + earth_coordinate_offset_y_m);
        json << "},\n  \"envelope_face_motion\": {\"available\": "
             << (envelope_face_motion.available ? "true" : "false")
             << ", \"interval_s\": ";
        json_optional_number(
            json, envelope_face_motion.available,
            envelope_face_motion.interval_s);
        json << ", \"west_coordinate_velocity_ms\": ";
        json_optional_number(
            json, envelope_face_motion.available,
            envelope_face_motion.west_coordinate_velocity_ms);
        json << ", \"east_coordinate_velocity_ms\": ";
        json_optional_number(
            json, envelope_face_motion.available,
            envelope_face_motion.east_coordinate_velocity_ms);
        json << ", \"south_coordinate_velocity_ms\": ";
        json_optional_number(
            json, envelope_face_motion.available,
            envelope_face_motion.south_coordinate_velocity_ms);
        json << ", \"north_coordinate_velocity_ms\": ";
        json_optional_number(
            json, envelope_face_motion.available,
            envelope_face_motion.north_coordinate_velocity_ms);
        json << "},\n  \"grid_envelope_faces_m\": {\"west\": ";
        json_optional_number(json, has_feature, envelope_west_m - grid_coordinate_offset_x_m);
        json << ", \"east\": ";
        json_optional_number(json, has_feature, envelope_east_m - grid_coordinate_offset_x_m);
        json << ", \"south\": ";
        json_optional_number(json, has_feature, envelope_south_m - grid_coordinate_offset_y_m);
        json << ", \"north\": ";
        json_optional_number(json, has_feature, envelope_north_m - grid_coordinate_offset_y_m);
        json << "},\n  \"distance_to_physical_boundary_m\": {\"west\": ";
        json_optional_number(json, has_feature, distance_west_m);
        json << ", \"east\": ";
        json_optional_number(json, has_feature, distance_east_m);
        json << ", \"south\": ";
        json_optional_number(json, has_feature, distance_south_m);
        json << ", \"north\": ";
        json_optional_number(json, has_feature, distance_north_m);
        json << ", \"nearest\": ";
        json_optional_number(json, has_feature, nearest_boundary_m);
        json << "},\n  \"boundary_contact\": ";
        json_optional_bool(json, has_feature, boundary_contact);
        json << ",\n  \"inner_margin_breached\": ";
        json_optional_bool(json, has_feature, inner_margin_breached);
        json << ",\n  \"moving_nest_scientific_validity\": {\n"
             << "    \"owner\": \"physical_child_trusted_interior_v1\",\n"
             << "    \"status\": \"";
        if (!nested_movement_authority)
            json << "not_applicable";
        else if (nested_scientific_invalid)
            json << "invalid";
        else if (nested_scientific_pending)
            json << "valid_pending_relocation";
        else if (!nested_trusted_interior_available)
            json << "not_assessed";
        else
            json << "valid";
        json << "\",\n"
             << "    \"reason\": \"" << nested_scientific_reason << "\",\n"
             << "    \"trusted_interior\": ";
        json_optional_bool(
            json, nested_trusted_interior_available,
            nested_trusted_interior);
        json << ",\n    \"relaxation_width_m\": {\"x\": ";
        json_optional_number(
            json, nested_movement_authority,
            nested_relaxation_width_x_m);
        json << ", \"y\": ";
        json_optional_number(
            json, nested_movement_authority,
            nested_relaxation_width_y_m);
        json << "},\n"
             << "    \"emergency_diagnostic_only\": "
             << (nested_movement_authority && movement_decision &&
                         movement_decision->reason ==
                             tmv::domain::StormWindowMovementReason::EmergencyContainment
                     ? "true"
                     : "false")
             << "\n  },\n  \"movement_authority\": {\n";
        json << "    \"algorithm\": \"" << tmv::domain::storm_window_movement_algorithm_id() << "\",\n"
             << "    \"mode\": \""
             << (nested_movement_authority
                     ? "live_one_way_nested_child"
                     : (config.recycle_domain
                            ? (tmv::physical_grid_metrics::terrain_is_active(
                                   state)
                                   ? "live_earth_fixed_terrain_cgrid_recycling"
                                   : "live_flat_cgrid_recycling")
                            : "diagnostic_request_only"))
             << "\",\n"
             << "    \"enabled\": " << (movement_authority_enabled ? "true" : "false") << ",\n"
             << "    \"recycling_enabled\": " << (config.recycle_domain ? "true" : "false") << ",\n"
             << "    \"nested_movement_enabled\": "
             << (nested_movement_authority ? "true" : "false") << ",\n"
             << "    \"prognostic_state_mutation\": " << (movement_committed ? "true" : "false") << ",\n"
             << "    \"commit_performed\": " << (movement_committed ? "true" : "false") << ",\n"
             << "    \"transaction_id\": ";
        if (recycling_result.committed)
            json << "\"" << tmv::strutil::json_escape(recycling_result.transaction_id) << "\"";
        else
            json << "null";
        json << ",\n    \"transaction_elapsed_ms\": ";
        json_optional_number(
            json, movement_committed,
            recycling_result.committed ? recycling_result.elapsed_ms
                                       : nested_movement_elapsed_ms);
        json << ",\n    \"event_directory\": ";
        if (recycling_result.committed)
            json << "\"" << tmv::strutil::json_escape(recycling_result.event_directory.generic_string()) << "\"";
        else
            json << "null";
        json << ",\n    \"post_shift_grid_origin_t0_earth_m\": {\"x\": ";
        json_optional_number(
            json, recycling_result.committed, recycling_result.post_shift_frame.earth_origin_at_t0_m.x_m);
        json << ", \"y\": ";
        json_optional_number(
            json, recycling_result.committed, recycling_result.post_shift_frame.earth_origin_at_t0_m.y_m);
        json << "},\n    \"post_shift_child_grid_origin_t0_earth_m\": {\"x\": ";
        json_optional_number(
            json, nested_movement_result.movement_committed,
            run.nested_execution.child.domain_frame
                .earth_origin_at_t0_m.x_m);
        json << ", \"y\": ";
        json_optional_number(
            json, nested_movement_result.movement_committed,
            run.nested_execution.child.domain_frame
                .earth_origin_at_t0_m.y_m);
        json << "},\n    \"committed_shift_cells\": {\"i\": ";
        if (recycling_result.committed)
            json << recycling_result.shift_i_cells;
        else if (nested_movement_result.movement_committed)
            json << nested_movement_result.movement.parent_shift_i;
        else
            json << "null";
        json << ", \"j\": ";
        if (recycling_result.committed)
            json << recycling_result.shift_j_cells;
        else if (nested_movement_result.movement_committed)
            json << nested_movement_result.movement.parent_shift_j;
        else
            json << "null";
        json << "},\n    \"nested_movement_sequence\": ";
        if (nested_movement_result.movement_committed)
            json << nested_movement_result.movement.movement_sequence;
        else
            json << "null";
        json << ",\n    \"two_way_inventory\": ";
        const auto& moved_inventory = nested_movement_result.movement.two_way_inventory;
        if (nested_movement_result.movement_committed && moved_inventory.complete)
        {
            static constexpr const char* names[]{"dry_mass",
                                                 "rho_theta",
                                                 "qv",
                                                 "qc",
                                                 "qr",
                                                 "qi",
                                                 "qs",
                                                 "qg",
                                                 "qh",
                                                 "physical_x_momentum",
                                                 "physical_y_momentum",
                                                 "physical_z_momentum",
                                                 "dry_internal_energy",
                                                 "water_sensible_energy",
                                                 "phase_reference_energy",
                                                 "resolved_kinetic_energy",
                                                 "gravitational_energy",
                                                 "subgrid_tke_energy",
                                                 "total_energy"};
            static_assert(std::size(names) == std::tuple_size_v<decltype(moved_inventory.before)>);
            json << "{\"schema_version\":1,\"feedback_application_count\":"
                 << moved_inventory.feedback_application_count << ",\"currencies\":{";
            for (std::size_t f = 0; f < moved_inventory.before.size(); ++f)
            {
                if (f != 0)
                    json << ',';
                json << '\"' << names[f] << "\":{\"before\":" << moved_inventory.before[f]
                     << ",\"retired\":" << moved_inventory.retired[f] << ",\"admitted\":" << moved_inventory.admitted[f]
                     << ",\"boundary_source\":" << moved_inventory.boundary_source[f]
                     << ",\"after\":" << moved_inventory.after[f]
                     << ",\"arithmetic_scale\":" << moved_inventory.arithmetic_scale[f] << '}';
            }
            json << "}}";
        }
        else
            json << "null";
        json << ",\n    \"precipitation_relocation\": ";
        if (nested_movement_result.movement_committed)
        {
            const auto& precipitation =
                nested_movement_result.movement.precipitation;
            json << "{\"owner\": \"earth_fixed_surface_history_v1\", "
                 << "\"schema_version\": "
                 << NestedChildPrecipitationMovementReportV1::schema_version
                 << ", \"active\": "
                 << (precipitation.active ? "true" : "false")
                 << ", \"sample_time_s\": "
                 << precipitation.sample_time_s
                 << ", \"newly_admitted_columns\": "
                 << precipitation.newly_admitted_columns
                 << ", \"previous_child_accumulated_water_m3\": "
                 << precipitation.previous_child_accumulated_water_m3
                 << ", \"discarded_child_accumulated_water_m3\": "
                 << precipitation.discarded_child_accumulated_water_m3
                 << ", \"retained_child_accumulated_water_m3\": "
                 << precipitation.retained_child_accumulated_water_m3
                 << ", \"newly_admitted_parent_accumulated_water_m3\": "
                 << precipitation.newly_admitted_parent_accumulated_water_m3
                 << ", \"committed_child_accumulated_water_m3\": "
                 << precipitation.committed_child_accumulated_water_m3
                 << ", \"previous_child_precipitation_volume_rate_m3s\": "
                 << precipitation.previous_child_precipitation_volume_rate_m3s
                 << ", \"discarded_child_precipitation_volume_rate_m3s\": "
                 << precipitation.discarded_child_precipitation_volume_rate_m3s
                 << ", \"retained_child_precipitation_volume_rate_m3s\": "
                 << precipitation.retained_child_precipitation_volume_rate_m3s
                 << ", \"newly_admitted_parent_precipitation_volume_rate_m3s\": "
                 << precipitation.newly_admitted_parent_precipitation_volume_rate_m3s
                 << ", \"committed_child_precipitation_volume_rate_m3s\": "
                 << precipitation.committed_child_precipitation_volume_rate_m3s
                 << "}";
        }
        else
        {
            json << "null";
        }
        json << ",\n    \"status\": ";
        if (movement_decision)
            json << "\"" << tmv::domain::storm_window_movement_status_name(movement_decision->status) << "\"";
        else
            json << "\"disabled\"";
        json << ",\n    \"reason\": ";
        if (movement_decision)
            json << "\"" << tmv::domain::storm_window_movement_reason_name(movement_decision->reason) << "\"";
        else
            json << "\"maximum_shift_cells_zero\"";
        json << ",\n    \"request_ready\": "
             << (movement_decision && movement_decision->status == tmv::domain::StormWindowMovementStatus::Requested
                     ? "true"
                     : "false")
             << ",\n    \"requested_shift_cells\": {\"i\": ";
        if (movement_decision)
            json << movement_decision->requested_i_shift_cells;
        else
            json << "null";
        json << ", \"j\": ";
        if (movement_decision)
            json << movement_decision->requested_j_shift_cells;
        else
            json << "null";
        json << "},\n    \"confirmation_count\": ";
        if (movement_decision)
            json << movement_decision->confirmation_count;
        else
            json << "null";
        json << ",\n    \"confirmation_samples_required\": " << movement_confirmation_samples << ",\n"
             << "    \"inner_margin_m\": " << config.inner_margin_m << ",\n"
             << "    \"release_margin_m\": " << movement_release_margin_m << ",\n"
             << "    \"maximum_shift_cells\": " << config.maximum_shift_cells << ",\n"
             << "    \"legal_parent_shift_bounds_cells\": {\"available\": "
             << (nested_movement_evidence &&
                         nested_movement_evidence->legal_shift_bounds_available
                     ? "true"
                     : "false")
             << ", \"minimum_i\": ";
        if (nested_movement_evidence &&
            nested_movement_evidence->legal_shift_bounds_available)
            json << nested_movement_evidence->minimum_i_shift_cells;
        else
            json << "null";
        json << ", \"maximum_i\": ";
        if (nested_movement_evidence &&
            nested_movement_evidence->legal_shift_bounds_available)
            json << nested_movement_evidence->maximum_i_shift_cells;
        else
            json << "null";
        json << ", \"minimum_j\": ";
        if (nested_movement_evidence &&
            nested_movement_evidence->legal_shift_bounds_available)
            json << nested_movement_evidence->minimum_j_shift_cells;
        else
            json << "null";
        json << ", \"maximum_j\": ";
        if (nested_movement_evidence &&
            nested_movement_evidence->legal_shift_bounds_available)
            json << nested_movement_evidence->maximum_j_shift_cells;
        else
            json << "null";
        json << "},\n"
             << "    \"cooldown_s\": " << config.cooldown_s << ",\n"
             << "    \"armed\": " << (movement_state.armed ? "true" : "false") << ",\n"
             << "    \"committed_move_available\": " << (movement_state.committed_move_available ? "true" : "false")
             << ",\n"
             << "    \"cumulative_committed_shift_cells\": {\"i\": " << movement_state.cumulative_i_shift_cells
             << ", \"j\": " << movement_state.cumulative_j_shift_cells << "}\n"
             << "  },\n"
             << "  \"continuity\": {\n"
             << "    \"status\": \"" << tmv::diagnostics::storm_continuity_status_name(continuity.status) << "\",\n"
             << "    \"reason\": \"" << tmv::diagnostics::storm_continuity_reason_name(continuity.reason) << "\",\n"
             << "    \"tracking_point\": \"updraft_core_centroid\",\n"
             << "    \"branch_id\": " << continuity.branch_id << ",\n"
             << "    \"selected_candidate_available\": " << (continuity.selected_candidate_available ? "true" : "false")
             << ",\n"
             << "    \"selected_candidate_rank\": ";
        json_optional_count(json, continuity.selected_candidate_available, continuity.selected_candidate_rank);
        json << ",\n    \"selected_component_label\": ";
        json_optional_count(json, continuity.selected_candidate_available, continuity.selected_component_label);
        json << ",\n"
             << "    \"topology\": \"" << tmv::diagnostics::storm_branch_topology_name(continuity.topology) << "\",\n"
             << "    \"pending_topology\": \""
             << tmv::diagnostics::storm_branch_topology_name(continuity.pending_topology) << "\",\n"
             << "    \"pending_topology_samples\": " << continuity.pending_topology_samples << ",\n"
             << "    \"topology_event\": \""
             << tmv::diagnostics::storm_branch_topology_event_name(continuity.topology_event) << "\",\n"
             << "    \"topology_event_id\": " << continuity.topology_event_id << ",\n"
             << "    \"topology_reorganization\": " << (continuity.topology_reorganization ? "true" : "false") << ",\n"
             << "    \"reorganization_overlap_basis\": "
             << (detection.previous_partition_available ? "\"exact_watershed_partition_cells\""
                                                        : "\"envelope_bbox_fallback\"")
             << ",\n"
             << "    \"reorganization_overlap_fraction\": " << continuity.reorganization_overlap_fraction << ",\n"
             << "    \"history_samples\": " << continuity.sample_count << ",\n"
             << "    \"missed_samples\": " << continuity.missed_samples << ",\n"
             << "    \"confidence\": " << continuity.continuity_confidence << ",\n"
             << "    \"innovation_distance_m\": " << continuity.innovation_distance_m << ",\n"
             << "    \"association_gate_radius_m\": " << continuity.association_gate_radius_m << "\n"
             << "  },\n"
             << "  \"environmental_branch_association\": {\n"
             << "    \"algorithm\": \"bunkers_2000_height_integrated_v1\",\n"
             << "    \"selection_uses_updraft_helicity_sign\": false,\n"
             << "    \"available\": " << (storm_branch_association.environmental_motion_available ? "true" : "false")
             << ",\n"
             << "    \"preference\": \""
             << tmv::diagnostics::storm_branch_preference_name(storm_branch_association.preference) << "\",\n"
             << "    \"selected_motion_class\": \""
             << tmv::diagnostics::storm_branch_motion_class_name(continuity.selected_motion_class) << "\",\n"
             << "    \"velocity_frame\": \"earth_relative\",\n"
             << "    \"velocity_uncertainty_ms\": " << storm_branch_association.environmental_velocity_uncertainty_ms
             << ",\n"
             << "    \"bunkers_right_velocity_ms\": {\"x\": ";
        json_optional_number(json,
                             storm_branch_association.environmental_motion_available,
                             storm_branch_association.bunkers_right_velocity_x_ms +
                                 (detector_coordinates_are_earth
                                      ? 0.0
                                      : tracking_snapshot_frame.velocity_ms.u_ms));
        json << ", \"y\": ";
        json_optional_number(json,
                             storm_branch_association.environmental_motion_available,
                             storm_branch_association.bunkers_right_velocity_y_ms +
                                 (detector_coordinates_are_earth
                                      ? 0.0
                                      : tracking_snapshot_frame.velocity_ms.v_ms));
        json << "},\n    \"bunkers_left_velocity_ms\": {\"x\": ";
        json_optional_number(json,
                             storm_branch_association.environmental_motion_available,
                             storm_branch_association.bunkers_left_velocity_x_ms +
                                 (detector_coordinates_are_earth
                                      ? 0.0
                                      : tracking_snapshot_frame.velocity_ms.u_ms));
        json << ", \"y\": ";
        json_optional_number(json,
                             storm_branch_association.environmental_motion_available,
                             storm_branch_association.bunkers_left_velocity_y_ms +
                                 (detector_coordinates_are_earth
                                      ? 0.0
                                      : tracking_snapshot_frame.velocity_ms.v_ms));
        json << "},\n    \"selected_bunkers_right_mismatch_ms\": ";
        json_optional_number(json,
                             continuity.selected_candidate_available &&
                                 storm_branch_association.environmental_motion_available,
                             continuity.selected_bunkers_right_mismatch_ms);
        json << ",\n    \"selected_bunkers_left_mismatch_ms\": ";
        json_optional_number(json,
                             continuity.selected_candidate_available &&
                                 storm_branch_association.environmental_motion_available,
                             continuity.selected_bunkers_left_mismatch_ms);
        json << "\n  },\n"
             << "  \"motion\": {\n"
             << "    \"velocity_frame\": \"earth_relative\",\n"
             << "    \"velocity_ms\": {\"x\": ";
        json_optional_number(json, motion_available, earth_velocity_x_ms);
        json << ", \"y\": ";
        json_optional_number(json, motion_available, earth_velocity_y_ms);
        json << "},\n    \"speed_ms\": ";
        json_optional_number(json, motion_available, earth_speed_ms);
        json << ",\n    \"grid_velocity_ms\": {\"x\": ";
        json_optional_number(json, motion_available, grid_velocity_x_ms);
        json << ", \"y\": ";
        json_optional_number(json, motion_available, grid_velocity_y_ms);
        json << "},\n    \"grid_speed_ms\": ";
        json_optional_number(
            json, motion_available,
            std::hypot(grid_velocity_x_ms, grid_velocity_y_ms));
        json << ",\n    \"regression_uncertainty_ms\": ";
        json_optional_number(json, motion_available, continuity.velocity_uncertainty_ms);
        json << ",\n    \"prediction_horizon_s\": ";
        json_optional_number(json, motion_available, prediction_horizon_s);
        json << ",\n    \"predicted_core_centroid_m\": {\"x\": ";
        json_optional_number(
            json, motion_available,
            predicted_x_m +
                (detector_coordinates_are_earth
                     ? 0.0
                     : predicted_earth_origin.x_m));
        json << ", \"y\": ";
        json_optional_number(
            json, motion_available,
            predicted_y_m +
                (detector_coordinates_are_earth
                     ? 0.0
                     : predicted_earth_origin.y_m));
        json << "},\n    \"grid_predicted_core_centroid_m\": {\"x\": ";
        json_optional_number(
            json, motion_available,
            predicted_x_m - predicted_grid_coordinate_offset_x_m);
        json << ", \"y\": ";
        json_optional_number(
            json, motion_available,
            predicted_y_m - predicted_grid_coordinate_offset_y_m);
        json << "},\n    \"nominal_time_to_boundary_s\": {\"west\": ";
        json_optional_number(json, moving_west, time_west_s);
        json << ", \"east\": ";
        json_optional_number(json, moving_east, time_east_s);
        json << ", \"south\": ";
        json_optional_number(json, moving_south, time_south_s);
        json << ", \"north\": ";
        json_optional_number(json, moving_north, time_north_s);
        json << ", \"nearest\": ";
        json_optional_number(json, nearest_time_available, nearest_time_s);
        json << "}\n"
             << "  },\n"
             << "  \"thresholds\": {\n"
             << "    \"minimum_updraft_ms\": " << config.minimum_updraft_ms << ",\n"
             << "    \"minimum_condensate_path_kgm2\": " << config.minimum_condensate_path_kgm2 << ",\n"
             << "    \"minimum_absolute_updraft_helicity_m2s2\": " << config.minimum_absolute_updraft_helicity_m2s2
             << ",\n"
             << "    \"minimum_component_cells\": " << config.minimum_component_cells << ",\n"
             << "    \"minimum_core_area_m2\": " << config.minimum_core_area_m2 << ",\n"
             << "    \"minimum_confidence\": " << config.minimum_confidence << ",\n"
             << "    \"inner_margin_m\": " << config.inner_margin_m << ",\n"
             << "    \"maximum_motion_speed_ms\": " << config.maximum_motion_speed_ms << ",\n"
             << "    \"maximum_motion_acceleration_ms2\": " << config.maximum_motion_acceleration_ms2 << ",\n"
             << "    \"minimum_continuity_samples\": " << config.minimum_continuity_samples << ",\n"
             << "    \"lost_after_samples\": " << config.lost_after_samples << ",\n"
             << "    \"topology_confirmation_samples\": " << config.topology_confirmation_samples << ",\n"
             << "    \"branch_velocity_uncertainty_ms\": " << config.branch_velocity_uncertainty_ms << "\n"
             << "  }\n"
             << "}\n";

        std::ostringstream filename;
        filename << "track_step_" << std::setfill('0') << std::setw(9) << completed_step << ".json";
        const std::filesystem::path history_path = std::filesystem::path(outdir) / "storm_track" / filename.str();
        std::string                 error;
        if (!publish_output_json_atomically(history_path, json.str(), error) ||
            !publish_output_json_atomically(
                std::filesystem::path(outdir) / "storm_track_latest.json", json.str(), error))
        {
            tmv::log_error("[STORM TRACK] ", error);
            return 2;
        }

        tmv::log_info("[STORM TRACK] status=",
                      tmv::diagnostics::storm_detection_status_name(detection.status),
                      ", time_s=",
                      run.time_s,
                      ", confidence=",
                      detection.confidence,
                      ", continuity=",
                      tmv::diagnostics::storm_continuity_status_name(continuity.status),
                      ", speed_ms=",
                      motion_available ? continuity.speed_ms : -1.0,
                      ", nearest_boundary_m=",
                      has_feature ? nearest_boundary_m : -1.0,
                      ", inner_margin_breached=",
                      inner_margin_breached ? "true" : "false",
                      ", compute_ms=",
                      compute_ms);
        if (nested_scientific_invalid)
        {
            std::string invalidation_error;
            if ((write_every_s > 0 || output.qualification_diagnostics_enabled()) &&
                !publish_output_json_atomically(
                    std::filesystem::path(outdir) /
                        kNimbusRunOutputContractFilename,
                        build_output_run_contract_json(
                            output.output_publication_generation, "failed",
                            run.configuration.nesting.enabled,
                            output.qualification_diagnostics_enabled(),
                            write_every_s > 0),
                    invalidation_error))
            {
                tmv::log_error(
                    "[NESTING SCIENTIFIC VALIDITY] could not invalidate the run output contract: ",
                    invalidation_error);
                return 2;
            }
            std::ostringstream invalidation;
            invalidation << std::setprecision(17)
                         << "{\n"
                         << "  \"schema\": \"nimbus.moving_nest.scientific_validity\",\n"
                         << "  \"schema_version\": 1,\n"
                         << "  \"status\": \"invalid\",\n"
                         << "  \"owner\": \"physical_child_trusted_interior_v1\",\n"
                         << "  \"reason\": \"" << nested_scientific_reason << "\",\n"
                         << "  \"simulation_time_s\": " << run.time_s << ",\n"
                         << "  \"completed_step\": " << completed_step << ",\n"
                         << "  \"tracker_record\": \"storm_track/"
                         << filename.str() << "\",\n"
                         << "  \"publication_generation\": ";
            if (output.output_publication_generation.empty())
                invalidation << "null";
            else
                invalidation << "\""
                             << tmv::strutil::json_escape(
                                    output.output_publication_generation)
                             << "\"";
            invalidation << "\n}\n";
            if (!publish_output_json_atomically(
                    std::filesystem::path(outdir) /
                        "moving_nest_scientific_validity.json",
                    invalidation.str(), invalidation_error))
            {
                tmv::log_error(
                    "[NESTING SCIENTIFIC VALIDITY] could not publish the invalidity artifact: ",
                    invalidation_error);
                return 2;
            }
            tmv::log_error(
                "[NESTING SCIENTIFIC VALIDITY] moving child failed closed at time_s=",
                run.time_s,
                ", reason=",
                nested_scientific_reason,
                "; the diagnostic tracker record was published and no invalid-step "
                "history or checkpoint is permitted");
            return 2;
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        tmv::log_error(
            "[STORM TRACK] tracking/recycling failed closed at time_s=", run.time_s, ": ", error.what());
        return 2;
    }
}

#ifdef EXPORT_NPY
bool OutputPublisher::write_qualification_diagnostics_frame(
    int frame_export_index)
{
    const bool nesting_configured =
        run.configuration.nesting.enabled;
    const bool child_relationship_ready =
        run.nested_execution.relationship_ready;
    if (!qualification_diagnostics_child_ownership_consistent(
            nesting_configured, child_relationship_ready))
    {
        validation.validation_error_message =
            "sparse qualification diagnostics configured nesting and live "
            "child ownership disagree";
        return false;
    }

    const OutputConfig saved_output_config = output_config;
    const auto saved_active_fields = active_export_fields;
    const bool saved_nested = publishing_nested_child;
    const bool saved_qualification = publishing_qualification_diagnostics;
    const auto restore = [&]
    {
        output_config = saved_output_config;
        active_export_fields = saved_active_fields;
        publishing_nested_child = saved_nested;
        publishing_qualification_diagnostics = saved_qualification;
    };

    output_config = qualification_output_config;
    active_export_fields = qualification_output_config.resolved_fields;
    publishing_qualification_diagnostics = true;
    publishing_nested_child = false;
    try
    {
        if (child_relationship_ready)
        {
            bool child_ok = false;
            publishing_nested_child = true;
            {
                ScopedNestedChildResidency child_residency(
                    run);
                child_ok = write_all_fields(nested_child_execution_state(run), frame_export_index);
            }
            publishing_nested_child = false;
            if (!child_ok)
            {
                restore();
                return false;
            }
        }
        else if (!write_all_fields(frame_export_index))
        {
            restore();
            return false;
        }
    }
    catch (...)
    {
        restore();
        throw;
    }
    restore();
    return true;
}
#endif

int OutputPublisher::maybe_write_output()
{
    if (export_diagnostics_enabled && steps % 10 == 0)
    {
        float theta_min = 1e10, theta_max = -1e10;
        float u_min = 1e10, u_max = -1e10;
        int   nan_count = 0, inf_count = 0;
        for (int i = 0; i < NR && i < 10; ++i)
        {
            for (int j = 0; j < NTH && j < 5; ++j)
            {
                for (int k = 0; k < NZ && k < 5; ++k)
                {
                    if (std::isnan(theta[i][j][k]))
                        nan_count++;
                    if (std::isinf(theta[i][j][k]))
                        inf_count++;
                    if (theta[i][j][k] < theta_min)
                        theta_min = theta[i][j][k];
                    if (theta[i][j][k] > theta_max)
                        theta_max = theta[i][j][k];
                    if (u[i][j][k] < u_min)
                        u_min = u[i][j][k];
                    if (u[i][j][k] > u_max)
                        u_max = u[i][j][k];
                }
            }
        }
        // Full-field w_max scan (cheap: one pass, no allocation)
        float w_global_min = 1e10f, w_global_max = -1e10f;
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const float wv = static_cast<float>(w[i][j][k]);
                    if (wv < w_global_min)
                        w_global_min = wv;
                    if (wv > w_global_max)
                        w_global_max = wv;
                }
        tmv::log_info("[STEP DIAGNOSTIC] step=", steps,
                      " time_s=", simulation_time,
                      " sample_theta_min_k=", theta_min,
                      " sample_theta_max_k=", theta_max,
                      " sample_u_min_ms=", u_min,
                      " sample_u_max_ms=", u_max,
                      " sample_nan=", nan_count,
                      " sample_inf=", inf_count,
                      " w_min_ms=", w_global_min,
                      " w_max_ms=", w_global_max);

        if (theta_min < 0 || theta_max > 700 || std::abs(u_min) > 150 || std::abs(u_max) > 150)
        {
            tmv::log_warn("[STEP DIAGNOSTIC] sampled state exceeded diagnostic bounds at step ", steps);
        }
    }
#ifdef EXPORT_NPY
    if (export_ms > 0)
    {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastGuiExport).count() >= export_ms)
        {
            lastGuiExport   = now;
            std::string tmp = std::string("data/.tracer_slice_th0.npy.tmp");
            std::string fin = std::string("data/tracer_slice_th0.npy");
            npy::write_field_slice(tracer, thetaIndex, tmp);
            std::string publish_error;
            if (!nimbus::output::atomic_replace_file(tmp, fin, publish_error))
                tmv::log_error("[GUI EXPORT] ", publish_error);
        }
    }
    if (write_every_s > 0)
    {
        while (simulation_time + 1.0e-9 >= next_field_export_time_s)
        {
            // Populate active_export_fields for this tick.
            // When tiered cadence is disabled, all resolved fields are active.
            active_export_fields.clear();
            const bool medium_due =
                !output_config.tiered_write_cadence || (simulation_time + 1.0e-9 >= next_medium_export_time_s);
            const bool slow_due =
                !output_config.tiered_write_cadence || (simulation_time + 1.0e-9 >= next_slow_export_time_s);

            for (const auto& name : output_config.resolved_fields)
            {
                if (!output_config.tiered_write_cadence)
                {
                    active_export_fields.insert(name);
                }
                else
                {
                    const WriteCadenceTier tier = get_field_write_cadence_tier(name);
                    if (tier == WriteCadenceTier::fast || (tier == WriteCadenceTier::medium && medium_due) ||
                        (tier == WriteCadenceTier::slow && slow_due))
                    {
                        active_export_fields.insert(name);
                    }
                }
            }

            const int frame_export_index = export_index++;
            bool      export_ok          = true;
            timed_call(perf_totals.output_submission_s,
                       [&] { export_ok = write_all_fields(frame_export_index); });
            if (!export_ok)
            {
                tmv::log_error("[VALIDATION] ", validation.validation_error_message);
                return 2;
            }

            if (run.nested_execution.relationship_ready)
            {
                bool child_export_ok = false;
                publishing_nested_child = true;
                try
                {
                    {
                        ScopedNestedChildResidency child_residency(run);
                        timed_call(perf_totals.output_submission_s,
                                   [&] { child_export_ok = write_all_fields(nested_child_execution_state(run), frame_export_index); });
                    }
                    publishing_nested_child = false;
                }
                catch (...)
                {
                    publishing_nested_child = false;
                    throw;
                }
                if (!child_export_ok)
                {
                    tmv::log_error("[VALIDATION] nested child output failed: ", validation.validation_error_message);
                    return 2;
                }
            }
            next_field_export_time_s += static_cast<double>(write_every_s);
            if (medium_due && output_config.tiered_write_cadence)
            {
                next_medium_export_time_s += static_cast<double>(output_config.write_cadence_medium_s);
            }
            if (slow_due && output_config.tiered_write_cadence)
            {
                next_slow_export_time_s += static_cast<double>(output_config.write_cadence_slow_s);
            }
        }
    }
    if (qualification_diagnostics_enabled())
    {
        while (simulation_time + 1.0e-9 >=
               next_qualification_diagnostics_export_time_s)
        {
            const int frame_export_index = qualification_export_index++;
            bool export_ok = true;
            timed_call(
                perf_totals.output_submission_s,
                [&]
                {
                    export_ok = write_qualification_diagnostics_frame(
                        frame_export_index);
                });
            if (!export_ok)
            {
                tmv::log_error(
                    "[VALIDATION] sparse qualification diagnostics failed: ",
                    validation.validation_error_message);
                return 2;
            }
            next_qualification_diagnostics_export_time_s +=
                static_cast<double>(
                    output_config.qualification_diagnostics_interval_s);
        }
    }
#endif
    return 0;
}

void HeadlessDriver::record_step_timing(const PerfClock::time_point& step_t0)
{
    if (global_perf_timing_enabled && perf_step_active)
    {
        const auto   step_t1     = PerfClock::now();
        const double step_wall_s = std::chrono::duration<double>(step_t1 - step_t0).count();
        perf_totals.total_step_s += step_wall_s;
        ++perf_totals.steps;

        // Periodic reporting is intentionally sparse. A cadence of zero means
        // final summary only, not one log write per step.
        const bool periodic_report_due =
            global_perf_report_every_steps > 0 &&
            (perf_totals.steps % static_cast<uint64_t>(global_perf_report_every_steps) == 0);
        if (periodic_report_due)
        {
            const AdvectionKernelStepTiming kt = current_advection_kernel_step_timing();
            tmv::log_info("[KERNEL SAMPLE PERF] measured_step=",
                          perf_totals.steps,
                          ", model_step=",
                          (steps + 1),
                          ", wall_ms=",
                          (step_wall_s * 1000.0),
                          ", kernel_total_s=",
                          kt.kernel_total_s,
                          ", backend=",
                          kt.backend,
                          ", fallback=",
                          kt.fallback_active ? "active" : "off",
                          ", cpu_calls=",
                          kt.cpu_calls,
                          ", gpu_calls=",
                          kt.gpu_calls);
            const double step_ms = (perf_totals.total_step_s / std::max<uint64_t>(1, perf_totals.steps)) * 1000.0;
            const double sim_seconds_per_wall_second =
                perf_totals.simulated_s / std::max(1e-9, perf_totals.total_step_s);
            tmv::log_info("[PERF] measured_steps=",
                          perf_totals.steps,
                          ", warmup_steps=",
                          global_perf_warmup_steps,
                          ", avg_step_ms=",
                          step_ms,
                          ", sim_s_per_wall_s=",
                          sim_seconds_per_wall_second);
        }
    }
}

bool HeadlessDriver::should_stop() const
{
    return duration_s >= 0 && simulation_time + 1.0e-9 >= static_cast<double>(duration_s);
}

bool HeadlessDriver::capture_termination_request()
{
    const int requested = tmv::runtime::requested_termination_signal();
    if (requested == 0)
        return false;
    if (termination_signal_number == 0)
    {
        termination_signal_number = requested;
        tmv::log_warn("[RUNTIME] ",
                      tmv::runtime::termination_signal_name(requested),
                      " received; stopping after completed step=",
                      steps,
                      ", simulation_time_s=",
                      simulation_time,
                      " and committing final output/checkpoint state");
    }
    return true;
}

bool HeadlessDriver::publish_termination_record()
{
    if (termination_signal_number == 0)
        return true;
    std::error_code ec;
    std::filesystem::create_directories(outdir, ec);
    if (ec)
    {
        tmv::log_error("[RUNTIME] cannot create termination-record directory '", outdir, "': ", ec.message());
        return false;
    }
    const bool checkpoint_committed =
        checkpoints.committed_step(static_cast<std::uint64_t>(std::max(0, steps)));
    std::ostringstream record;
    record << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"nimbus.termination\",\n"
           << "  \"schema_version\": 1,\n"
           << "  \"reason\": \"signal\",\n"
           << "  \"signal\": \""
           << tmv::runtime::termination_signal_name(termination_signal_number) << "\",\n"
           << "  \"signal_number\": " << termination_signal_number << ",\n"
           << "  \"exit_status\": " << 128 + termination_signal_number << ",\n"
           << "  \"boundary\": \"completed_timestep\",\n"
           << "  \"completed_steps\": " << std::max(0, steps) << ",\n"
           << "  \"simulation_time_s\": " << simulation_time << ",\n"
           << "  \"output_drained\": true,\n"
           << "  \"checkpoint_committed\": " << (checkpoint_committed ? "true" : "false") << ",\n"
           << "  \"checkpoint_file\": ";
    if (checkpoint_committed)
        record << "\"" << tmv::strutil::json_escape(checkpoints.last_path().string()) << "\"\n";
    else
        record << "null\n";
    record << "}\n";
    std::string error;
    if (!publish_output_json_atomically(
            std::filesystem::path(outdir) / "nimbus_termination.json", record.str(), error))
    {
        tmv::log_error("[RUNTIME] failed to publish termination record: ", error);
        return false;
    }
    tmv::log_warn("[RUNTIME] graceful termination committed at step=",
                  steps,
                  ", time_s=",
                  simulation_time,
                  ", checkpoint=",
                  checkpoint_committed ? checkpoints.last_path().string() : std::string("disabled"),
                  ", exit_status=",
                  128 + termination_signal_number);
    return true;
}

int HeadlessDriver::finalize()
{
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    if (allocation_measurement_started)
    {
        allocation_snapshot            = tmv::allocation_telemetry::end_measurement();
        allocation_measurement_started = false;
    }
#endif
    if (global_simulation_storage.nested_execution.relationship_ready &&
        global_simulation_storage.runtime.radiation.config.scheme_id ==
            "simple_grey")
    {
        struct RadiationActivity
        {
            double        absolute_power_w = 0.0;
            std::uint64_t nonzero_cells = 0;
            std::uint64_t nonfinite_cells = 0;
        };
        const auto activity = [](
            const tmv::RadiationRuntimeState& radiation,
            const GridStorageState& grid)
        {
            RadiationActivity result;
            const PhysicalCellDomain physical = physical_scalar_domain(
                grid.nr, grid.nth, grid.nz, StaggerType::CGrid);
            for (int i = physical.i_begin; i < physical.i_end; ++i)
                for (int j = physical.j_begin; j < physical.j_end; ++j)
                    for (int k = physical.k_begin; k < physical.k_end; ++k)
                    {
                        const double value = static_cast<double>(
                            radiation.held_cell_power_w[i][j][k]);
                        if (!std::isfinite(value))
                        {
                            ++result.nonfinite_cells;
                            continue;
                        }
                        result.absolute_power_w += std::abs(value);
                        if (value != 0.0)
                            ++result.nonzero_cells;
                    }
            return result;
        };
        const auto& parent =
            global_simulation_storage.runtime.radiation;
        const auto& child = global_simulation_storage.nested_execution.child
                                .runtime.radiation;
        const RadiationActivity parent_activity = activity(
            parent, global_simulation_storage.grid);
        const RadiationActivity child_activity = activity(
            child, global_simulation_storage.nested_execution.child.grid);
        tmv::log_info(
            "[NESTING RADIATION] parent_owner=host, parent_last_update_time_s=",
            parent.last_update_time_s,
            ", parent_next_update_time_s=", parent.next_update_time_s,
            ", parent_applied_steps=", parent.applied_step_count,
            ", parent_held_power_w=", parent.held_atmospheric_power_w,
            ", parent_held_abs_power_w=", parent_activity.absolute_power_w,
            ", parent_nonzero_cells=", parent_activity.nonzero_cells,
            ", parent_nonfinite_cells=", parent_activity.nonfinite_cells,
            ", parent_last_expected_energy_j=",
            parent.last_expected_atmospheric_energy_j,
            ", parent_last_applied_energy_j=",
            parent.last_applied_radiation_heat_j,
            ", parent_expected_energy_j=",
            parent.accumulated_expected_atmospheric_energy_j,
            ", parent_applied_energy_j=",
            parent.accumulated_applied_radiation_heat_j,
            ", parent_energy_residual_j=",
            parent.accumulated_applied_radiation_heat_j -
                parent.accumulated_expected_atmospheric_energy_j,
            ", child_owner=host, child_last_update_time_s=",
            child.last_update_time_s,
            ", child_next_update_time_s=", child.next_update_time_s,
            ", child_applied_steps=", child.applied_step_count,
            ", child_held_power_w=", child.held_atmospheric_power_w,
            ", child_held_abs_power_w=", child_activity.absolute_power_w,
            ", child_nonzero_cells=", child_activity.nonzero_cells,
            ", child_nonfinite_cells=", child_activity.nonfinite_cells,
            ", child_last_expected_energy_j=",
            child.last_expected_atmospheric_energy_j,
            ", child_last_applied_energy_j=",
            child.last_applied_radiation_heat_j,
            ", child_expected_energy_j=",
            child.accumulated_expected_atmospheric_energy_j,
            ", child_applied_energy_j=",
            child.accumulated_applied_radiation_heat_j,
            ", child_energy_residual_j=",
            child.accumulated_applied_radiation_heat_j -
                child.accumulated_expected_atmospheric_energy_j);
    }
    if (global_simulation_storage.nested_execution.relationship_ready &&
        global_simulation_storage.runtime.turbulence.config.scheme_id ==
            "smagorinsky" &&
        global_simulation_storage.runtime.compute.active_kind !=
            ComputeBackendKind::Cpu &&
        global_simulation_storage.runtime.compute.backend)
    {
        const ComputeBackend& backend =
            *global_simulation_storage.runtime.compute.backend;
        tmv::log_info(
            "[NESTING DEVICE] smagorinsky_coefficients_dispatches=",
            backend.successful_kernel_dispatch_count(
                nimbus::compute::KernelId::SmagorinskyCoefficients),
            ", smagorinsky_tendencies_dispatches=",
            backend.successful_kernel_dispatch_count(
                nimbus::compute::KernelId::SmagorinskyTendencies),
            ", sgs_apply_dispatches=",
            backend.successful_kernel_dispatch_count(
                nimbus::compute::KernelId::SgsApply));
        tmv::log_info(
            "[NESTING DEVICE LEVELS] parent_coefficients=",
            nested_parent_smagorinsky_dispatches[0],
            ", parent_tendencies=",
            nested_parent_smagorinsky_dispatches[1],
            ", parent_sgs_apply=",
            nested_parent_smagorinsky_dispatches[2],
            ", child_coefficients=",
            nested_child_smagorinsky_dispatches[0],
            ", child_tendencies=",
            nested_child_smagorinsky_dispatches[1],
            ", child_sgs_apply=",
            nested_child_smagorinsky_dispatches[2]);
        const auto& parent_turbulence =
            global_simulation_storage.runtime.turbulence;
        const auto& child_turbulence =
            global_simulation_storage.nested_execution.child.runtime.turbulence;
        tmv::log_info(
            "[NESTING SGS] parent_work_steps=",
            parent_turbulence.momentum_work_step_count,
            ", parent_committed_work_j=",
            parent_turbulence.cumulative_committed_momentum_work_j,
            ", parent_stress_steps=",
            parent_turbulence.open_boundary_stability_step_count,
            ", parent_open_boundary_work_j=",
            parent_turbulence.cumulative_open_boundary_work_j,
            ", parent_irreversible_dissipation_j=",
            parent_turbulence.cumulative_irreversible_dissipation_j,
            ", parent_stress_residual_j=",
            parent_turbulence.cumulative_stress_closure_residual_j,
            ", child_work_steps=",
            child_turbulence.momentum_work_step_count,
            ", child_committed_work_j=",
            child_turbulence.cumulative_committed_momentum_work_j,
            ", child_stress_steps=",
            child_turbulence.open_boundary_stability_step_count,
            ", child_open_boundary_work_j=",
            child_turbulence.cumulative_open_boundary_work_j,
            ", child_irreversible_dissipation_j=",
            child_turbulence.cumulative_irreversible_dissipation_j,
            ", child_stress_residual_j=",
            child_turbulence.cumulative_stress_closure_residual_j);
    }
    if (global_simulation_storage.nested_execution.relationship_ready &&
        global_simulation_storage.runtime.microphysics.scheme &&
        global_simulation_storage.runtime.microphysics.scheme->get_scheme_name() ==
            "kessler" &&
        global_simulation_storage.runtime.compute.active_kind !=
            ComputeBackendKind::Cpu &&
        global_simulation_storage.runtime.compute.backend)
    {
        const ComputeBackend& backend =
            *global_simulation_storage.runtime.compute.backend;
        tmv::log_info(
            "[NESTING DEVICE MICROPHYSICS] kessler_pointwise_dispatches=",
            backend.successful_kernel_dispatch_count(
                nimbus::compute::KernelId::KesslerPointwise),
            ", kessler_sedimentation_dispatches=",
            backend.successful_kernel_dispatch_count(
                nimbus::compute::KernelId::KesslerSedimentation),
            ", microphysics_apply_dispatches=",
            backend.successful_kernel_dispatch_count(
                nimbus::compute::KernelId::MicrophysicsApply));
        tmv::log_info(
            "[NESTING DEVICE MICROPHYSICS LEVELS] parent_pointwise=",
            nested_parent_kessler_dispatches[0],
            ", parent_sedimentation=",
            nested_parent_kessler_dispatches[1],
            ", parent_apply=",
            nested_parent_kessler_dispatches[2],
            ", child_pointwise=",
            nested_child_kessler_dispatches[0],
            ", child_sedimentation=",
            nested_child_kessler_dispatches[1],
            ", child_apply=",
            nested_child_kessler_dispatches[2]);
        const auto& parent_microphysics =
            global_simulation_storage.runtime.microphysics;
        const auto& child_microphysics =
            global_simulation_storage.nested_execution.child.runtime.microphysics;
        const auto& parent_conservation =
            parent_microphysics.last_conservation_report;
        const auto& child_conservation =
            child_microphysics.last_conservation_report;
        const auto& parent_heat =
            parent_microphysics.last_thermodynamic_repair_report;
        const auto& child_heat =
            child_microphysics.last_thermodynamic_repair_report;
        const auto water_activity_kg = [](
            const tmv::MicrophysicsThermodynamicRepairReport& report,
            const tmv::MicrophysicsConservationReport& conservation)
        {
            double activity = std::abs(conservation.surface_precipitation_kg);
            for (double value : report.microphysics_water_tendency_kg_by_species)
                activity += std::abs(value);
            return activity;
        };
        const auto energy_activity_j = [](
            const tmv::MicrophysicsThermodynamicRepairReport& report)
        {
            return std::abs(report.moist_dry_internal_change_j)
                 + std::abs(report.moist_water_sensible_change_j)
                 + std::abs(report.moist_phase_reference_change_j)
                 + std::abs(report.moist_carrier_kinetic_change_j)
                 + std::abs(report.moist_gravitational_change_j)
                 + std::abs(report.moist_subgrid_tke_change_j);
        };
        const auto material_water_repair_kg = [](
            const tmv::MicrophysicsWaterRepairReport& report)
        {
            return std::abs(report.lower_bound_correction_kg())
                 + std::abs(report.upper_bound_correction_kg())
                 + std::abs(report.nonfinite_correction_kg());
        };
        const auto material_water_repair_count = [](
            const tmv::MicrophysicsWaterRepairReport& report)
        {
            return report.lower_bound_count() + report.upper_bound_count()
                 + report.nonfinite_count();
        };
        const auto theta_repair_count = [](
            const tmv::MicrophysicsThermodynamicRepairReport& report)
        {
            return report.theta_tendency_limit_count
                 + report.theta_nonfinite_count
                 + report.theta_lower_bound_count
                 + report.theta_upper_bound_count;
        };
        const auto& parent_water_repair =
            parent_microphysics.last_water_repair_report;
        const auto& child_water_repair =
            child_microphysics.last_water_repair_report;
        tmv::log_info(
            "[NESTING MICROPHYSICS] parent_active=",
            parent_conservation.active ? 1 : 0,
            ", parent_relative_imbalance=",
            parent_conservation.relative_imbalance,
            ", parent_imbalance_kg=",
            parent_conservation.tendency_imbalance_kg,
            ", parent_baseline_water_kg=",
            parent_conservation.baseline_q_kg,
            ", parent_water_activity_kg=",
            water_activity_kg(parent_heat, parent_conservation),
            ", parent_material_water_repair_kg=",
            material_water_repair_kg(parent_water_repair),
            ", parent_material_water_repair_count=",
            material_water_repair_count(parent_water_repair),
            ", parent_surface_precipitation_kg=",
            parent_conservation.surface_precipitation_kg,
            ", parent_heat_j=",
            parent_heat.microphysics_heat_j,
            ", parent_theta_repair_j=",
            std::abs(parent_heat.net_theta_repair_j()),
            ", parent_theta_repair_count=",
            theta_repair_count(parent_heat),
            ", parent_moist_energy_complete=",
            parent_heat.moist_energy_complete ? 1 : 0,
            ", parent_moist_energy_residual_j=",
            parent_heat.moist_energy_closure_residual_j,
            ", parent_energy_activity_j=",
            energy_activity_j(parent_heat),
            ", child_active=",
            child_conservation.active ? 1 : 0,
            ", child_relative_imbalance=",
            child_conservation.relative_imbalance,
            ", child_imbalance_kg=",
            child_conservation.tendency_imbalance_kg,
            ", child_baseline_water_kg=",
            child_conservation.baseline_q_kg,
            ", child_water_activity_kg=",
            water_activity_kg(child_heat, child_conservation),
            ", child_material_water_repair_kg=",
            material_water_repair_kg(child_water_repair),
            ", child_material_water_repair_count=",
            material_water_repair_count(child_water_repair),
            ", child_surface_precipitation_kg=",
            child_conservation.surface_precipitation_kg,
            ", child_heat_j=",
            child_heat.microphysics_heat_j,
            ", child_theta_repair_j=",
            std::abs(child_heat.net_theta_repair_j()),
            ", child_theta_repair_count=",
            theta_repair_count(child_heat),
            ", child_moist_energy_complete=",
            child_heat.moist_energy_complete ? 1 : 0,
            ", child_moist_energy_residual_j=",
            child_heat.moist_energy_closure_residual_j,
            ", child_energy_activity_j=",
            energy_activity_j(child_heat));
    }
    if (const int rc = maybe_write_checkpoint(true); rc != 0)
    {
        return rc;
    }

    if (options.live_shm)
    {
        std::ostringstream telemetry;
        telemetry << std::setprecision(17)
                  << "{\n"
                  << "  \"schema\": \"nimbus.live_shm.telemetry\",\n"
                  << "  \"schema_version\": 1,\n"
                  << "  \"publication_generation\": \""
                  << tmv::strutil::json_escape(output.output_publication_generation)
                  << "\",\n"
                  << "  \"endpoint_name\": \""
                  << tmv::strutil::json_escape(options.live_shm_name)
                  << "\",\n"
                  << "  \"cadence_mode\": \""
                  << (options.live_shm_full_rate ? "every_model_step"
                                                : "simulation_time")
                  << "\",\n"
                  << "  \"interval_s\": "
                  << options.live_shm_interval_s << ",\n"
                  << "  \"fields\": [";
        for (std::size_t index = 0;
             index < options.live_shm_fields.size(); ++index)
        {
            if (index != 0) telemetry << ", ";
            telemetry << "\""
                      << tmv::strutil::json_escape(
                             options.live_shm_fields[index])
                      << "\"";
        }
        telemetry << "],\n"
                  << "  \"frames_published\": " << output.live_shm_frames << ",\n"
                  << "  \"final_sequence\": "
                  << output.shm_writer.sequence_number() << ",\n"
                  << "  \"payload_bytes_published\": "
                  << output.live_shm_payload_bytes << ",\n"
                  << "  \"cadence_skipped_steps\": "
                  << output.live_shm_skipped_steps << ",\n"
                  << "  \"failed_frames\": " << output.live_shm_failed_frames
                  << ",\n"
                  << "  \"publish_time_s\": "
                  << output.live_shm_publish_time_s << "\n"
                  << "}\n";
        std::string telemetry_error;
        if (!publish_output_json_atomically(
                std::filesystem::path(outdir) / "nimbus_live_shm.json",
                telemetry.str(), telemetry_error))
        {
            tmv::log_error("[SHM] failed to publish telemetry: ",
                           telemetry_error);
            return 2;
        }
        tmv::log_info(
            "[SHM TELEMETRY] frames=", output.live_shm_frames,
            ", payload_bytes=", output.live_shm_payload_bytes,
            ", cadence_skipped_steps=", output.live_shm_skipped_steps,
            ", failed_frames=", output.live_shm_failed_frames,
            ", publish_time_s=", output.live_shm_publish_time_s);
    }

    if ((write_every_s > 0 || output.qualification_diagnostics_enabled()) &&
        !validation.export_validation_reports.empty())
    {
        std::filesystem::path summary_path = std::filesystem::path(outdir) / "validation_summary.json";
        if (!global_validation_report_path.empty())
        {
            summary_path = std::filesystem::path(global_validation_report_path) / "validation_summary.json";
        }

        std::string write_error;
        if (!tmv::write_validation_run_summary_json(
                validation.export_validation_reports, global_validation_policy, summary_path, write_error))
        {
            tmv::log_error("[VALIDATION] ", write_error);
            return 2;
        }
    }

    if (global_perf_timing_enabled && perf_totals.steps > 0)
    {
        const double step_total = std::max(perf_totals.total_step_s, 1e-9);
        const auto   pct_step   = [&](double component) { return 100.0 * component / step_total; };
        tmv::log_info(
            "[PERF SUMMARY] measured_steps=",
            perf_totals.steps,
            ", warmup_steps=",
            global_perf_warmup_steps,
            ", scope=model_step_including_cfl_and_enabled_consumers_excluding_periodic_file_output_submission",
            ", step_wall_s=",
            step_total,
            ", avg_step_ms=",
            (1000.0 * step_total / perf_totals.steps),
            ", sim_s_per_wall_s=",
            (perf_totals.simulated_s / step_total),
            ", measured_simulated_s=",
            perf_totals.simulated_s);
        tmv::log_info("  radiation_s=", perf_totals.radiation_s, " (", pct_step(perf_totals.radiation_s), "%)");
        tmv::log_info(
            "  boundary_layer_s=", perf_totals.boundary_layer_s, " (", pct_step(perf_totals.boundary_layer_s), "%)");
        tmv::log_info("  chaos_noise_s=", perf_totals.chaos_noise_s, " (", pct_step(perf_totals.chaos_noise_s), "%)");
        tmv::log_info(
            "  chaos_tendency_s=", perf_totals.chaos_tendency_s, " (", pct_step(perf_totals.chaos_tendency_s), "%)");
        tmv::log_info("  dynamics_s=", perf_totals.dynamics_s, " (", pct_step(perf_totals.dynamics_s), "%)");
        tmv::log_info("  output_preparation_submission_s=",
                      perf_totals.output_submission_s,
                      " (outside solver-step denominator; async writer flush time excluded)");
        log_dynamics_perf_summary(step_total);
        log_turbulence_perf_summary();
        log_advection_perf_summary();
#ifdef NIMBUS_ALLOCATION_TELEMETRY
        tmv::log_info("[ALLOCATION SUMMARY] heap_allocations=",
                      allocation_snapshot.heap_allocations,
                      ", heap_requested_bytes=",
                      allocation_snapshot.heap_requested_bytes,
                      ", physics_field_allocations=",
                      allocation_snapshot.physics_field_allocations,
                      ", physics_field_allocated_bytes=",
                      allocation_snapshot.physics_field_allocated_bytes);
        const auto log_component_allocations = [](const char* name, const tmv::allocation_telemetry::Snapshot& snapshot)
        {
            tmv::log_info("[ALLOCATION COMPONENT] name=",
                          name,
                          ", heap_allocations=",
                          snapshot.heap_allocations,
                          ", heap_requested_bytes=",
                          snapshot.heap_requested_bytes,
                          ", physics_field_allocations=",
                          snapshot.physics_field_allocations,
                          ", physics_field_allocated_bytes=",
                          snapshot.physics_field_allocated_bytes);
        };
        log_component_allocations("radiation", radiation_allocation_snapshot);
        log_component_allocations("boundary_layer", boundary_layer_allocation_snapshot);
        log_component_allocations("dynamics", dynamics_allocation_snapshot);
        const auto log_scoped_component_allocations =
            [](tmv::allocation_telemetry::Component component, const tmv::allocation_telemetry::Counters& counters)
        {
            tmv::log_info("[ALLOCATION COMPONENT] name=",
                          tmv::allocation_telemetry::component_name(component),
                          ", heap_allocations=",
                          counters.heap_allocations,
                          ", heap_requested_bytes=",
                          counters.heap_requested_bytes,
                          ", physics_field_allocations=",
                          counters.physics_field_allocations,
                          ", physics_field_allocated_bytes=",
                          counters.physics_field_allocated_bytes);
        };
        for (std::size_t component = 0;
             component < static_cast<std::size_t>(tmv::allocation_telemetry::Component::Count);
             ++component)
        {
            log_scoped_component_allocations(static_cast<tmv::allocation_telemetry::Component>(component),
                                             allocation_snapshot.components[component]);
        }
        for (std::size_t component = 0;
             component < static_cast<std::size_t>(tmv::allocation_telemetry::Component::Count);
             ++component)
        {
            const auto allocation_component = static_cast<tmv::allocation_telemetry::Component>(component);
            for (std::size_t bytes = 1; bytes <= tmv::allocation_telemetry::maximum_exact_heap_size_bytes; ++bytes)
            {
                const auto count = tmv::allocation_telemetry::heap_size_count(allocation_component, bytes);
                if (count == 0)
                    continue;
                tmv::log_info("[ALLOCATION SIZE] component=",
                              tmv::allocation_telemetry::component_name(allocation_component),
                              ", bytes=",
                              bytes,
                              ", allocations=",
                              count);
            }
        }
#endif
    }

    if (output.async_writer.has_value())
    {
        AsyncOutputWriter& writer = *output.async_writer;

        // Drain the async writer and report I/O statistics.
        if (!writer.flush())
        {
            tmv::log_error("[OUTPUT] Async writer error during flush: ", writer.error_message());
#ifdef EXPORT_NPY
            output.fail_qualified_zfp_output_contract();
#endif
            return 2;
        }
        if (!output.enforce_output_retention())
            return 2;
        log_memory_milestone("output_drain_complete");
        if (writer.snapshots_written() > 0)
        {
            const double      write_time  = writer.total_write_time_s();
            const std::size_t total_bytes = writer.total_bytes_written();
            const double      mb          = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
            tmv::log_info("[OUTPUT] ",
                          writer.snapshots_written(),
                          " snapshots written",
                          ", ",
                          mb,
                          " MB total",
                          ", ",
                          write_time,
                          "s write time",
                          ", max_snapshot_write_s=",
                          writer.maximum_snapshot_write_time_s(),
                          ", max_snapshot_bytes=",
                          writer.maximum_snapshot_bytes_written(),
                          ", producer_backpressure_wait_s=",
                          writer.total_backpressure_wait_time_s(),
                          ", ",
                          (write_time > 0.0 ? mb / write_time : 0.0),
                          " MB/s");
            tmv::log_info("[OUTPUT] Snapshot assembly: frames=",
                          output.snapshot_assembly_frames,
                          ", total_s=",
                          output.snapshot_assembly_total_s,
                          ", max_s=",
                          output.snapshot_assembly_max_s,
                          ", max_bytes=",
                          output.snapshot_assembly_max_bytes,
                          ", max_owned_bytes=",
                          output.snapshot_assembly_max_owned_bytes);
#ifdef HAVE_ZFP
            if (writer.zfp_codec_workspace_initializations() > 0)
            {
                tmv::log_info("[OUTPUT] ZFP codec workspace: initializations=",
                              writer.zfp_codec_workspace_initializations(),
                              ", bitstream_bindings=",
                              writer.zfp_bitstream_bindings(),
                              ", bitstream_capacity_bytes=",
                              writer.zfp_bitstream_capacity_bytes());
                tmv::log_info("[OUTPUT] ZFP codec provenance: version=",
                              writer.zfp_codec_version(),
                              ", libzfp_execution_policy=",
                              writer.zfp_execution_policy_id() == 0 ? "serial" : "unexpected");
                tmv::log_info(
                    "[OUTPUT] ZFP field execution: policy=",
                    writer.zfp_field_execution_policy(),
                    ", workers=", writer.zfp_resolved_field_workers(),
                    ", scratch_budget_mib=",
                    output.output_config.zfp_field_scratch_mib,
                    ", scratch_high_water_bytes=",
                    writer.zfp_parallel_scratch_high_water_bytes(),
                    ", tasks_completed=",
                    writer.zfp_parallel_tasks_completed(),
                    ", parallel_wall_s=",
                    writer.zfp_parallel_wall_time_s());
                if (writer.zfp_control_time_s() > 0.0 || writer.zfp_codec_time_s() > 0.0 ||
                    writer.zfp_container_io_time_s() > 0.0)
                {
                    tmv::log_info("[OUTPUT] ZFP instrumented phases: control_s=",
                                  writer.zfp_control_time_s(),
                                  ", codec_s=",
                                  writer.zfp_codec_time_s(),
                                  ", container_io_s=",
                                  writer.zfp_container_io_time_s());
                }
            }
#endif
        }
    }

    close_budget_csv();
    close_momentum_csv();
    close_energy_csv();
    tmv::close_earth_fixed_remap_csv();
    close_radiation_budget_csv();
    close_thermodynamic_source_budget_csv();
    close_moisture_csv();

    // Write Lagrangian parcel paths for the whole run (one file, not per-step).
    if (options.trajectory.enabled && trajectories.seeded())
    {
        const std::filesystem::path traj_path      = std::filesystem::path(outdir) / "trajectories.npy";
        const std::filesystem::path temporary_path = std::filesystem::path(outdir) / "trajectories.npy.tmp";
        std::error_code             ec;
        std::filesystem::remove(temporary_path, ec);
        ec.clear();
        if (trajectories.write(temporary_path.string()))
        {
            std::string publish_error;
            if (!nimbus::output::atomic_replace_file(
                    temporary_path, traj_path, publish_error))
            {
                tmv::log_error("[TRAJECTORIES] failed to publish '",
                               traj_path.string(), "': ", publish_error);
                std::filesystem::remove(temporary_path, ec);
                return 2;
            }

            std::size_t maximum_samples = 0;
            for (const auto& parcel : trajectories.parcels())
                maximum_samples = std::max(maximum_samples, parcel.path.size());
            std::ostringstream trajectory_manifest;
            trajectory_manifest << std::setprecision(17) << "{\n"
                                << "  \"schema\": \"nimbus.trajectories\",\n"
                                << "  \"schema_version\": 2,\n"
                                << "  \"artifact\": \"trajectories.npy\",\n"
                                << "  \"coordinate_frame\": \"earth_relative_cartesian_m\",\n"
                                << "  \"vertical_coordinate\": \"physical_geometric_height_msl\",\n"
                                << "  \"terrain_sampling\": \""
                                << (trajectories.terrain_aware() ? "mapped_physical_height"
                                                                : "uniform_cartesian_height")
                                << "\",\n"
                                << "  \"integration\": \"rk2_time_interpolated_midpoint\",\n"
                                << "  \"integration_cadence\": \"every_model_timestep\",\n"
                                << "  \"sample_interval_s\": "
                                << trajectories.configuration().sample_interval_s << ",\n"
                                << "  \"sample_interval_zero_semantics\": \"every_model_timestep\",\n"
                                << "  \"retention\": \"latest_samples_per_parcel\",\n"
                                << "  \"max_samples_per_parcel\": "
                                << trajectories.configuration().max_samples_per_parcel << ",\n"
                                << "  \"seed_time_s\": " << options.trajectory.seed_time_s << ",\n"
                                << "  \"requested_seed_shape\": [" << options.trajectory.nx << ", "
                                << options.trajectory.ny << ", " << options.trajectory.nz << "],\n"
                                << "  \"requested_half_extents_m\": [" << options.trajectory.half_x_m << ", "
                                << options.trajectory.half_y_m << ", " << options.trajectory.half_z_m << "],\n"
                                << "  \"realized_shape\": [" << trajectories.count() << ", " << maximum_samples << ", "
                                << tmv::kTrajectoryVarCount << "],\n"
                                << "  \"total_samples\": " << trajectories.total_samples() << ",\n"
                                << "  \"columns\": [\"t_s\", \"earth_x_m\", \"earth_y_m\", \"z_m\", "
                                   "\"w_ms\", \"vorticity_z_s-1\", \"tilting_s-2\", \"stretching_s-2\", "
                                   "\"baroclinic_s-2\"]\n"
                                << "}\n";
            std::string manifest_error;
            if (!publish_output_json_atomically(
                    std::filesystem::path(outdir) / "trajectories.json", trajectory_manifest.str(), manifest_error))
            {
                tmv::log_error("[TRAJECTORIES] failed to publish trajectory provenance: ", manifest_error);
                std::filesystem::remove(traj_path, ec);
                return 2;
            }
            tmv::log_info("[TRAJECTORIES] wrote ",
                          trajectories.count(),
                          " parcel paths (",
                          trajectories.total_samples(),
                          " samples) to ",
                          traj_path.string());
        }
        else
        {
            tmv::log_error("[TRAJECTORIES] failed to write parcel data");
            std::filesystem::remove(temporary_path, ec);
            return 2;
        }
    }

    // Parity harness verdict (GPU.md Phase 1): print the
    // CPU-vs-GPU divergence table and fail the run on any tolerance breach
    // so shader drift is a CI failure, not a silent skew.
    if (nimbus::parity::enabled())
    {
        nimbus::parity::print_report();
        if (nimbus::parity::any_breach())
        {
            tmv::log_error("[PARITY] FAIL: GPU-vs-CPU divergence exceeded "
                           "tolerance (see table above).");
            return 2;
        }
    }
    const auto& compute_runtime = global_simulation_storage.runtime.compute;
    if (compute_runtime.active_kind != ComputeBackendKind::Cpu &&
        compute_runtime.backend)
    {
        std::uint64_t kernel_dispatches = 0;
        std::ostringstream completed_kernels;
        bool first = true;
        for (std::size_t index = 0; index < nimbus::compute::kernel_count(); ++index)
        {
            const auto kernel = static_cast<nimbus::compute::KernelId>(index);
            const std::uint64_t count =
                compute_runtime.backend->successful_kernel_dispatch_count(kernel);
            if (count == 0)
                continue;
            kernel_dispatches += count;
            if (!first)
                completed_kernels << ',';
            first = false;
            completed_kernels
                << nimbus::compute::kernel_manifest_entry(kernel).kernel_name
                << ':' << count;
        }
        tmv::log_info(
            "[COMPUTE DISPATCH] backend=",
            compute_backend_kind_name(compute_runtime.active_kind),
            ", completed_submissions=",
            compute_runtime.backend->successful_dispatch_count(),
            ", kernel_dispatches=", kernel_dispatches,
            ", resident_uploads=",
            compute_runtime.backend->resident_upload_count(),
            ", resident_reuses=",
            compute_runtime.backend->resident_reuse_count(),
            ", kernels={", completed_kernels.str(), "}");
    }
    if (!publish_termination_record())
        return 2;
    return 0;
}

int HeadlessDriver::run()
{
    const auto terminal_status = [&](int status)
    {
        if (status != 0)
        {
            output.fail_terminal_run_contract();
        }
        return status;
    };
    try
    {
        if (const int rc = setup(); rc != 0)
        {
            return terminal_status(rc);
        }
        if (const int rc = restore_resume_checkpoint(); rc != 0)
        {
            return terminal_status(rc);
        }
        if (const int rc = build_initial_conditions(); rc != 0)
        {
            return terminal_status(rc);
        }
        if (options.before_time_loop_test_hook)
        {
            options.before_time_loop_test_hook(global_simulation_storage);
        }
        if (const int rc = run_time_loop(); rc != 0)
        {
            return terminal_status(rc);
        }
        if (const int rc = finalize(); rc != 0)
        {
            return terminal_status(rc);
        }
        return terminal_status(termination_signal_number == 0 ? 0 : 128 + termination_signal_number);
    }
    catch (...)
    {
        output.fail_terminal_run_contract();
        throw;
    }
}

} // namespace

/**
 * @brief Executes the simulation loop in headless mode with periodic exports.
 * @param options Headless runtime options from CLI/config integration.
 * @return Zero on success, non-zero on runtime or export failure.
 */
int run_headless_simulation(const HeadlessRunOptions& options)
{
    try
    {
        return HeadlessDriver(options).run();
    }
    catch (const std::exception& error)
    {
        tmv::log_error("[RUNTIME] headless simulation failed: ", error.what());
        return 2;
    }
    catch (...)
    {
        tmv::log_error("[RUNTIME] headless simulation failed with an unknown exception");
        return 2;
    }
}
