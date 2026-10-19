/**
 * Deterministic cross-tool fixture produced by the real Nimbus writers.
 * This is intentionally a small odd/non-cubic Cartesian frame so accidental
 * axis swaps cannot hide behind equal dimensions.
 */

#include "core/output/netcdf_history_writer.hpp"
#include "core/output/npy_writer.hpp"
#include "core/output/step_manifest.hpp"
#include "core/output/history_index.hpp"

#include <filesystem>
#include <iostream>
#include <string>

namespace
{

constexpr const char* kGeneration = "0123456789abcdef0123456789abcdef";

ExportSnapshot fixture_snapshot(const std::filesystem::path& step, bool terrain)
{
    ExportSnapshot snapshot;
    snapshot.export_index           = 4;
    snapshot.simulation_time_s      = 75.5;
    snapshot.step_dir               = step;
    snapshot.grid.nx                = 2;
    snapshot.grid.ny                = 3;
    snapshot.grid.nz                = 2;
    snapshot.grid.dx_m              = 1000.0;
    snapshot.grid.dy_m              = 750.0;
    snapshot.grid.dz_m              = 250.0;
    snapshot.grid.coordinate_system = "cartesian";
    snapshot.grid.native_staggering = "c_grid";
    if (terrain)
    {
        snapshot.grid.terrain_following = true;
        snapshot.grid.physical_z_center_m = {
            40.0, 170.0, 45.0, 185.0, 50.0, 200.0,
            90.0, 260.0, 95.0, 275.0, 100.0, 290.0,
        };
        snapshot.grid.physical_z_interface_m = {
            0.0, 80.0, 260.0, 0.0, 90.0, 280.0, 0.0, 100.0, 300.0,
            50.0, 130.0, 390.0, 50.0, 140.0, 410.0, 50.0, 150.0, 430.0,
        };
    }

    FieldSnapshotEntry theta;
    theta.name  = "theta";
    theta.dim0  = 2;
    theta.dim1  = 3;
    theta.dim2  = 2;
    theta.is_3d = true;
    theta.data  = {300.0f, 301.0f, 302.0f, 303.0f, 304.0f, 305.0f,
                   310.0f, 311.0f, 312.0f, 313.0f, 314.0f, 315.0f};
    snapshot.fields.push_back(theta);

    FieldSnapshotEntry rain;
    rain.name  = "accumulated_rainfall";
    rain.dim0  = 2;
    rain.dim1  = 3;
    rain.dim2  = 0;
    rain.is_3d = false;
    rain.data  = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    snapshot.fields.push_back(rain);
    return snapshot;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--terrain"))
    {
        std::cerr << "usage: netcdf_history_fixture OUTPUT_RUN_DIR [--terrain]\n";
        return 2;
    }
    const bool terrain = argc == 3;
    const std::filesystem::path run  = argv[1];
    const std::filesystem::path step = run / "step_000004";
    const std::filesystem::path npy  = run / "npy_reference";
    std::error_code             ec;
    std::filesystem::create_directories(step, ec);
    if (ec)
    {
        std::cerr << "cannot create fixture step: " << ec.message() << '\n';
        return 1;
    }
    std::filesystem::create_directories(npy, ec);
    if (ec)
    {
        std::cerr << "cannot create NPY reference directory: " << ec.message() << '\n';
        return 1;
    }

    const ExportSnapshot snapshot = fixture_snapshot(step, terrain);
    NetcdfHistoryWriteResult result;
    std::string              error;
    if (!write_netcdf4_history(snapshot, {}, result, error))
    {
        std::cerr << error << '\n';
        return 1;
    }
    if (!npy::write_3d(snapshot.fields[0].data.data(), 2, 3, 2, (npy / "theta.npy").string()) ||
        !npy::write_2d(snapshot.fields[1].data, 2, 3, (npy / "accumulated_rainfall.npy").string()))
    {
        std::cerr << "failed to write NPY parity references\n";
        return 1;
    }

    OutputStepManifestContext context;
    context.step_index             = 4;
    context.simulation_time_s      = snapshot.simulation_time_s;
    context.step_dir               = "step_000004";
    context.publication_generation = kGeneration;
    context.storage_format         = "netcdf";
    context.grid.nr                = snapshot.grid.nx;
    context.grid.nth               = snapshot.grid.ny;
    context.grid.nz                = snapshot.grid.nz;
    context.grid.coordinate_system = snapshot.grid.coordinate_system;
    context.grid.staggering        = snapshot.grid.native_staggering;
    context.grid.dr_m              = snapshot.grid.dx_m;
    context.grid.dy_m              = snapshot.grid.dy_m;
    context.grid.dtheta_rad        = snapshot.grid.dy_m;
    context.grid.dz_m              = snapshot.grid.dz_m;
    context.grid.terrain_following = terrain;
    if (terrain)
    {
        context.grid.terrain_min_z_m = 0.0;
        context.grid.terrain_max_z_m = 430.0;
    }
    context.published_fields       = {"accumulated_rainfall", "theta"};
    const std::string manifest_json = build_output_step_manifest_json(context);
    if (!nimbus::output::prepare_history_index_update(run, manifest_json, error) ||
        !publish_output_json_atomically(step / "manifest.json", manifest_json, error) ||
        !nimbus::output::publish_history_index_after_manifest(run, manifest_json, error) ||
        !publish_output_json_atomically(
            run / kNimbusRunOutputContractFilename, build_output_run_contract_json(kGeneration, "ready"), error))
    {
        std::cerr << error << '\n';
        return 1;
    }
    if (std::filesystem::exists(step / "history.nc.tmp") ||
        std::filesystem::exists(step / "manifest.json.tmp") ||
        std::filesystem::exists(run / "history_index.json.tmp") ||
        std::filesystem::exists(run / "nimbus_output.json.tmp"))
    {
        std::cerr << "temporary publication artifact remained visible\n";
        return 1;
    }
    std::cout << run.string() << '\n';
    return 0;
}
