#![recursion_limit = "256"]

mod commands;
mod error;
mod npy;
mod platform;
mod state;
mod zfp3d;

use state::AppState;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tauri::Manager;

/// Old enough that no in-flight recording could own it, short enough that a
/// leaked 4K session is not left on disk indefinitely.
const STALE_ENCODE_SESSION_AGE: Duration = Duration::from_secs(6 * 60 * 60);

fn bundled_simulator_path(resource_dir: &Path) -> Option<PathBuf> {
    let candidate = resource_dir
        .join("bin")
        .join(platform::simulator_executable_name());
    candidate.is_file().then_some(candidate)
}

/*
 * Nimbus Studio — Tauri shell.
 *
 * Responsibilities:
 *   - Manage the desktop window + lifecycle.
 *   - Expose the command surface to the React frontend — config I/O,
 *     runner, outputs, status, hardware, plugins, and movie encode (see
 *     the invoke_handler list below and the commands/ modules).
 *   - Hold the in-memory run registry across the session.
 *
 * The Rust side is intentionally thin: no business logic, no schema
 * validation, no plotting. Those live in the React layer.
 */

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::default())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // Reap encode sessions a previous quit/crash left behind.
            commands::movie::sweep_stale_encode_sessions(STALE_ENCODE_SESSION_AGE);
            // A packaged Studio owns the simulator it was built and qualified
            // with. This makes Finder/AppImage launches work without inheriting
            // a developer shell's cwd or NIMBUS_SIM_BIN. Settings remains an
            // explicit higher-priority override for advanced workflows.
            if let Ok(resource_dir) = app.path().resource_dir() {
                if let Some(simulator) = bundled_simulator_path(&resource_dir) {
                    if let Ok(mut cached) = app.state::<AppState>().simulator_binary.lock() {
                        *cached = Some(simulator);
                    }
                }
                #[cfg(windows)]
                {
                    // MSYS2 netCDF filter plugins are bundled beside the
                    // native DLL graph. Point HDF5 at that application-owned
                    // directory so compressed CM1/netCDF products retain the
                    // same readable formats as macOS/Linux installations.
                    let native_runtime = resource_dir.join("bin");
                    if native_runtime.is_dir() {
                        std::env::set_var("HDF5_PLUGIN_PATH", native_runtime);
                    }
                }
            }
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            if commands::visualization_qualification::visualization_qualification_request()
                .is_some()
            {
                if let Some(window) = app.get_webview_window("main") {
                    window.show()?;
                    window.set_focus()?;
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::config::read_config,
            commands::config::write_config,
            commands::config::list_configs,
            commands::config::project_root,
            commands::config::user_config_dir,
            commands::runner::spawn_sim,
            commands::remote_compute::probe_remote_compute,
            commands::remote_workspace::connect_remote_workspace,
            commands::remote_workspace::remote_workspace_status,
            commands::remote_workspace::disconnect_remote_workspace,
            commands::remote_workspace::open_remote_workspace,
            commands::remote_receive::receive_remote_results,
            commands::remote_receive::cancel_remote_receive,
            commands::runner::scheme_catalog,
            commands::runner::migrate_config,
            commands::runner::validate_config_contents,
            commands::runner::abort_sim,
            commands::runner::delete_run,
            commands::runner::list_runs,
            commands::runner::poll_run,
            commands::run_library::discover_run_library,
            commands::provenance::copy_run_provenance,
            commands::status::simulator_status,
            commands::hardware::hardware_info,
            commands::io_budget::configure_native_io_budget,
            commands::io_budget::native_io_budget_status,
            commands::worker_profile::profile_loading_workers,
            commands::worker_profile::loading_worker_profile_status,
            commands::worker_profile::cancel_loading_worker_profile,
            commands::hardware::process_memory_info,
            commands::datasets::list_netcdf_files,
            commands::datasets::inspect_cm1_case_folder,
            commands::cm1_namelist::inspect_cm1_namelist,
            commands::cm1_sounding::inspect_cm1_sounding,
            commands::cm1_sounding::save_cm1_sounding_projection,
            commands::cm1_sounding::validate_cm1_sounding_project,
            commands::checkpoints::inspect_checkpoint,
            commands::datasets::inspect_netcdf_metadata,
            commands::datasets::read_netcdf_field,
            commands::datasets::read_netcdf_values,
            commands::datasets::read_netcdf_field_window,
            commands::datasets::cancel_netcdf_field_read,
            commands::config_preflight::preflight_config,
            commands::outputs::list_step_dirs,
            commands::outputs::list_nested_history_levels,
            commands::outputs::read_field_bytes,
            commands::outputs::read_field_brick_hierarchy,
            commands::outputs::read_field_brick_window,
            commands::outputs::read_field_window,
            commands::outputs::cancel_field_read,
            commands::outputs::native_field_read_capabilities,
            commands::outputs::reduce_field,
            commands::outputs::reduce_field_axis,
            commands::outputs::read_trajectories,
            commands::outputs::run_has_trajectories,
            commands::outputs::read_storm_track,
            commands::plugins::list_plugins,
            commands::plugins::plugins_dir,
            commands::plugins::reveal_path,
            commands::movie::ffmpeg_detect,
            commands::movie::prepare_encode_session,
            commands::movie::append_encode_frame,
            commands::movie::encode_movie_dir,
            commands::movie::cleanup_encode_session,
            commands::visualization_qualification::visualization_qualification_request,
            commands::visualization_qualification::visualization_capacity_qualification,
            commands::visualization_qualification::record_visualization_qualification_progress,
            commands::visualization_qualification::create_visualization_brick_qualification_fixture,
            commands::visualization_qualification::remove_visualization_brick_qualification_fixture,
            commands::visualization_qualification::finish_visualization_qualification,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            // Closing the window must not leave nimbus_sim running headless.
            if matches!(event, tauri::RunEvent::Exit) {
                commands::remote_workspace::shutdown();
                commands::remote_receive::cancel_all_receives();
                commands::runner::terminate_all_runs(&app.state::<AppState>());
            }
        });
}

#[cfg(test)]
mod distribution_tests {
    use super::bundled_simulator_path;

    #[test]
    fn packaged_simulator_resolves_only_when_the_resource_exists() {
        let root =
            std::env::temp_dir().join(format!("nimbus-studio-resource-{}", std::process::id()));
        let bin = root.join("bin");
        let simulator = bin.join(crate::platform::simulator_executable_name());
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&bin).expect("create resource fixture");
        assert_eq!(bundled_simulator_path(&root), None);
        std::fs::write(&simulator, b"fixture").expect("create simulator fixture");
        assert_eq!(bundled_simulator_path(&root), Some(simulator));
        std::fs::remove_dir_all(root).expect("remove resource fixture");
    }
}
