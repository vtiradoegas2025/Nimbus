#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "core/output/output_config.hpp"
#include "core/output/shm_transport.hpp"
#include "diagnostics/trajectories.hpp"

struct SimulationStorageState;

/**
 * @file headless_runtime.hpp
 * @brief Command-line runtime options for non-GUI simulation execution.
 *
 * Defines the option bundle consumed by the headless runner and
 * exposes the entry point used by tests, scripts, and batch jobs.
 * This header is intentionally minimal to keep startup dependencies low.
 */

struct HeadlessRunOptions
{
    int export_ms = 0;
    int duration_s = -1;
    int write_every_s = 0;
    /// -1 preserves the historical output/live/debug cadence, 0 disables
    /// scientific budgets, and positive values select a one-based step cadence.
    int scientific_diagnostics_every_steps = -1;
    std::string outdir = "data/exports";
    OutputConfig output_config;

    /// Portable Nimbus checkpoint cadence in simulation seconds. Zero disables
    /// checkpoint publication. Checkpoints are committed only between completed
    /// timesteps and are independent of history-output cadence.
    int checkpoint_every_s = 0;

    /// Number of newest complete checkpoint pairs to retain. Zero is unlimited.
    int checkpoint_keep_last = 0;

    /// Filesystem reserve checked before checkpoint publication. Zero disables.
    int checkpoint_minimum_free_space_mib = 0;

    /// Directory for checkpoint binaries and their integrity/provenance
    /// sidecars. An empty path resolves to <outdir>/checkpoints.
    std::filesystem::path checkpoint_dir;

    /// Resume from a Nimbus portable checkpoint. The matching .json sidecar is
    /// mandatory so completed-step and branch provenance cannot be guessed.
    std::filesystem::path resume_checkpoint;

    /// Enable live shared-memory transport for zero-I/O visualization.
    /// When true, selected fields are streamed to a POSIX SHM region
    /// that the Vulkan viewer reads in real time.
    bool live_shm = false;

    /// Fields to stream via SHM (default: w, reflectivity_dbz, vorticity_z).
    /// Ignored if live_shm is false.
    std::vector<std::string> live_shm_fields = {"w", "reflectivity_dbz", "vorticity_z"};

    /// Logical endpoint shared with the viewer. The default preserves the
    /// standard Studio connection; explicit names isolate concurrent runs.
    std::string live_shm_name = tmv_shm::kDefaultShmName;

    /// Positive simulation-time cadence used unless full-rate is explicit.
    double live_shm_interval_s = 1.0;
    bool live_shm_full_rate = false;

    /// Online Lagrangian parcel-trajectory integration. Disabled by default;
    /// when enabled, parcels are seeded and advanced every step and the paths
    /// are written to <outdir>/trajectories.npy.
    tmv::TrajectoryConfig trajectory;

    // In-process tests may replace the initialized backend before stepping;
    // this is never populated by CLI or configuration selection.
    std::function<void(SimulationStorageState&)> before_time_loop_test_hook;

};

/**
 * @brief Runs the simulation in headless mode.
 * @param options Runtime options for export cadence and output paths.
 * @return Process-style status code where zero indicates success.
 */
int run_headless_simulation(const HeadlessRunOptions& options);
