/**
 * @file test_output_writer.cpp
 * @brief Tests for AsyncOutputWriter sync mode and NPY field roundtrip.
 *
 * Verifies:
 *   - Sync mode writes files to disk immediately
 *   - Written NPY files contain exact field values
 *   - Multi-field snapshot produces one file per field
 *   - Statistics (bytes written, snapshot count) are accurate
 */
#include "catch2/catch.hpp"
#include "core/output/direct_surface_writer.hpp"
#include "core/output/core_volume_writer.hpp"
#include "core/output/output_writer.hpp"
#include "core/output/history_index.hpp"
#include "core/output/field_brick_hierarchy.hpp"
#include "core/output/retention_policy.hpp"
#include "core/output/step_manifest.hpp"
#include "core/output/zfp_reader.hpp"
#include "core/output/zfp_field_storage.hpp"
#include "core/output/output_config.hpp"
#include "core/output/tranche_writer.hpp"
#include "core/field/field_snapshot.hpp"
#include "core/field/field3d.hpp"
#include "diagnostics/tranche_product_registry.hpp"
#include "diagnostics/derived_field_math.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace
{

std::string make_tmp_dir(const std::string& name)
{
    auto dir = std::filesystem::temp_directory_path() / ("tmv_test_writer_" + name);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir.string();
}

void cleanup_dir(const std::string& dir)
{
    std::filesystem::remove_all(dir);
}

std::vector<unsigned char> read_file_bytes(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void set_history_index_failure_injection(bool enabled)
{
#if defined(_WIN32)
    _putenv_s("NIMBUS_TEST_HISTORY_INDEX_FAIL_AFTER_MANIFEST",
              enabled ? "1" : "");
#else
    if (enabled)
        setenv("NIMBUS_TEST_HISTORY_INDEX_FAIL_AFTER_MANIFEST", "1", 1);
    else
        unsetenv("NIMBUS_TEST_HISTORY_INDEX_FAIL_AFTER_MANIFEST");
#endif
}

std::string indexed_manifest(const std::string& generation, int index)
{
    std::ostringstream out;
    out << "{\"publication_state\": \"complete\", \"publication_generation\": \""
        << generation << "\", \"step_index\": " << index
        << ", \"step_dir\": \"step_" << std::setfill('0') << std::setw(6) << index
        << "\", \"parent_step_indices\": []}\n";
    return out.str();
}

std::string current_history_segment(const std::filesystem::path& directory)
{
    std::ifstream head_input(directory / "history_index.json");
    const std::string head((std::istreambuf_iterator<char>(head_input)), std::istreambuf_iterator<char>());
    const std::string key = "\"segment\": \"";
    const auto begin = head.find(key);
    REQUIRE(begin != std::string::npos);
    const auto value_begin = begin + key.size();
    const auto end = head.find('"', value_begin);
    REQUIRE(end != std::string::npos);
    std::ifstream segment_input(directory / head.substr(value_begin, end - value_begin));
    return {std::istreambuf_iterator<char>(segment_input), std::istreambuf_iterator<char>()};
}

std::string all_history_segments(const std::filesystem::path& directory)
{
    std::string contents;
    for (const auto& entry : std::filesystem::directory_iterator(directory / "history_index"))
    {
        if (!entry.is_regular_file())
            continue;
        std::ifstream input(entry.path());
        contents.append(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }
    return contents;
}

ExportSnapshot make_snapshot(const std::string& step_dir, int index, int nr, int nth, int nz)
{
    ExportSnapshot snap;
    snap.export_index = index;
    snap.simulation_time_s = index * 10.0;
    snap.step_dir = step_dir;
    snap.grid.nx = nr;
    snap.grid.ny = nth;
    snap.grid.nz = nz;

    // Create a field with known pattern: value = i*1000 + j*10 + k + 0.5
    FieldSnapshotEntry entry;
    entry.name = "test_field";
    entry.dim0 = nr;
    entry.dim1 = nth;
    entry.dim2 = nz;
    entry.is_3d = true;
    entry.data.resize(static_cast<size_t>(nr) * nth * nz);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                size_t idx = i * nth * nz + j * nz + k;
                entry.data[idx] = static_cast<float>(i * 1000 + j * 10 + k) + 0.5f;
            }
    snap.fields.push_back(std::move(entry));
    return snap;
}

void bind_terrain_geometry_identity(ExportSnapshot& snapshot,
                                    const std::string& generation = "0123456789abcdef0123456789abcdef")
{
    snapshot.grid.terrain_geometry_publication_generation = generation;
    snapshot.grid.terrain_geometry_sha256 =
        terrain_geometry_content_sha256(snapshot.grid);
}

TEST_CASE("field brick hierarchy publishes exact extrema checksums and edge topology",
          "[core][output_writer][brick-hierarchy]")
{
    auto snapshot = make_snapshot("unused", 23, 65, 3, 33);
    snapshot.publication_generation = "0123456789abcdef0123456789abcdef";
    snapshot.grid.dx_m = 100.0;
    snapshot.grid.dy_m = 200.0;
    snapshot.grid.dz_m = 50.0;
    snapshot.grid.grid_origin_earth_x_m = -1000.0;
    snapshot.grid.grid_origin_earth_y_m = 500.0;
    const std::string first = nimbus::output::build_field_brick_hierarchy_json(
        snapshot.fields.front(), snapshot.grid, snapshot.publication_generation, true, false, 0.0);
    REQUIRE(first.find("\"schema\": \"nimbus.field-brick-hierarchy\"") != std::string::npos);
    REQUIRE(first.find("\"sourceShape\": [65, 3, 33]") != std::string::npos);
    REQUIRE(first.find("\"id\": \"l0:1:0:1\"") != std::string::npos);
    REQUIRE(first.find("\"extent\": [1, 3, 1]") != std::string::npos);
    REQUIRE(first.find("\"lower\": [5400, 500, 1600]") != std::string::npos);
    REQUIRE(first.find("\"upper\": [5500, 1100, 1650]") != std::string::npos);
    REQUIRE(first.find("\"cullingSafe\": true") != std::string::npos);
    const std::string repeated = nimbus::output::build_field_brick_hierarchy_json(
        snapshot.fields.front(), snapshot.grid, snapshot.publication_generation, true, false, 0.0);
    REQUIRE(first == repeated);
    const std::string padded = nimbus::output::build_field_brick_hierarchy_json(
        snapshot.fields.front(), snapshot.grid, snapshot.publication_generation,
        true, false, 0.25);
    REQUIRE(padded.find("\"extremaSemantics\": \"decoded_value_bounds\"") !=
            std::string::npos);
    REQUIRE(padded.find("\"extremaAbsolutePadding\": 0.25") !=
            std::string::npos);
    REQUIRE(padded.find("\"minimum\": 0.25") != std::string::npos);
    snapshot.fields.front().data.back() += 1.0f;
    const std::string changed = nimbus::output::build_field_brick_hierarchy_json(
        snapshot.fields.front(), snapshot.grid, snapshot.publication_generation, true, false, 0.0);
    REQUIRE(first != changed);
}

TEST_CASE("parallel visualization hierarchy preserves every payload byte and record order",
          "[core][output_writer][brick-hierarchy][parallel]")
{
    auto snapshot = make_snapshot("unused", 3, 129, 67, 65);
    const bool terrain = GENERATE(false, true);
    snapshot.grid.dx_m = 37.5;
    snapshot.grid.dy_m = 62.5;
    snapshot.grid.dz_m = 100.0;
    snapshot.grid.terrain_following = terrain;
    auto& entry = snapshot.fields.front();
    for (std::size_t n = 0; n < entry.data.size(); ++n)
        entry.data[n] = static_cast<float>(int(n % 37) - 18) / 17.0f;
    entry.data[65537] = std::numeric_limits<float>::quiet_NaN();
    entry.data[12489] = -0.0f;
    if (terrain)
    {
        snapshot.grid.physical_z_center_m.resize(entry.data.size());
        snapshot.grid.physical_z_interface_m.resize(129 * 67 * 66);
        for (int i = 0; i < 129; ++i)
            for (int j = 0; j < 67; ++j)
                for (int k = 0; k < 66; ++k)
                {
                    const double height = 0.3 * i + 0.7 * j + 37.0 * k + 0.125 * k * k;
                    snapshot.grid.physical_z_interface_m[(i * 67 + j) * 66 + k] = height;
                    if (k < 65)
                        snapshot.grid.physical_z_center_m[(i * 67 + j) * 65 + k] =
                            height + 0.5 * (37.125 + 0.25 * k);
                }
    }
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
    omp_set_num_threads(4);
#endif
    std::array<std::vector<std::string>, 2> payloads;
    std::array<std::string, 2> json;
    for (int pass = 0; pass < 2; ++pass)
    {
        const auto publish = [&](int level, const std::array<int, 3>& shape,
                                 const std::vector<float>& values,
                                 const nimbus::output::FieldBrickLodGeometry* geometry)
        {
            auto& bytes = payloads[pass].emplace_back(
                reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
            nimbus::output::FieldBrickLodPayloadIdentity identity;
            identity.path = "field.lod" + std::to_string(level) + ".npy";
            if (geometry)
            {
                identity.geometry.emplace();
                const std::array<const std::vector<double>*, 4> components{
                    &geometry->x_m, &geometry->y_m, &geometry->z_m, &geometry->volume_m3};
                for (std::size_t index = 0; index < components.size(); ++index)
                {
                    const auto& component = *components[index];
                    bytes.append(reinterpret_cast<const char*>(component.data()), component.size() * sizeof(double));
                    identity.geometry->paths[index] = "geometry" + std::to_string(index) + ".npy";
                    identity.geometry->content_sha256[index] = std::string(64, 'a');
                }
            }
            nimbus::crypto::Sha256 hash;
            hash.update(bytes.data(), bytes.size());
            identity.content_sha256 = hash.hex_digest();
            return identity;
        };
        json[pass] = nimbus::output::build_field_brick_hierarchy_json(entry, snapshot.grid,
            "0123456789abcdef0123456789abcdef", true, false, 0.0, publish, pass == 0 ? 1 : 4);
    }
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
    REQUIRE(payloads[0].size() >= 2);
    REQUIRE(payloads[0] == payloads[1]);
    REQUIRE(json[0] == json[1]);
}

TEST_CASE("ZFP decoded extrema padding is fail closed by compression contract",
          "[core][output_writer][brick-hierarchy][zfp]")
{
    const auto accuracy = nimbus::output::zfp_decoded_extrema_padding(
        "accuracy", 0.001, 0.0001, false);
    REQUIRE(accuracy.has_value());
    REQUIRE(std::abs(*accuracy - 0.0011) < 1e-12);
    REQUIRE_FALSE(nimbus::output::zfp_decoded_extrema_padding(
        "precision", 16.0, 0.0, false).has_value());
    REQUIRE_FALSE(nimbus::output::zfp_decoded_extrema_padding(
        "rate", 8.0, 0.0, false).has_value());
    REQUIRE_FALSE(nimbus::output::zfp_decoded_extrema_padding(
        "accuracy", 0.001, 0.0, true).has_value());
}

TEST_CASE("field brick hierarchy publishes durable recursive LOD identities",
          "[core][output_writer][brick-hierarchy][lod]")
{
    auto snapshot = make_snapshot("unused", 23, 130, 70, 35);
    snapshot.publication_generation = "0123456789abcdef0123456789abcdef";
    snapshot.grid.dx_m = 100.0;
    snapshot.grid.dy_m = 100.0;
    snapshot.grid.dz_m = 100.0;
    std::vector<std::array<int, 3>> shapes;
    std::vector<float> first_lod;
    const auto publish = [&](int level,
                             const std::array<int, 3>& shape,
                             const std::vector<float>& values,
                             const nimbus::output::FieldBrickLodGeometry*)
    {
        shapes.push_back(shape);
        if (level == 1) first_lod = values;
        return nimbus::output::FieldBrickLodPayloadIdentity{
            "test_field.lod" + std::to_string(level) + ".npy",
            std::string(64, static_cast<char>('a' + level - 1)),
        };
    };
    const std::string json = nimbus::output::build_field_brick_hierarchy_json(
        snapshot.fields.front(), snapshot.grid, snapshot.publication_generation,
        true, false, 0.0, publish);
    REQUIRE(shapes == std::vector<std::array<int, 3>>{{65, 35, 18}, {33, 18, 9}});
    REQUIRE(first_lod.size() == static_cast<std::size_t>(65 * 35 * 18));
    REQUIRE(first_lod.front() == Approx(506.0f));
    REQUIRE(json.find("\"schemaVersion\": 2") != std::string::npos);
    REQUIRE(json.find("\"level\": 2") != std::string::npos);
    REQUIRE(json.find("\"stride\": [4, 4, 4]") != std::string::npos);
    REQUIRE(json.find("\"path\": \"test_field.lod1.npy\"") != std::string::npos);
    REQUIRE(json.find("\"aggregation\": \"recursive_cell_mean_2x2x2\"") !=
            std::string::npos);
}

TEST_CASE("terrain brick LODs publish volume-weighted physical geometry identities",
          "[core][output_writer][brick-hierarchy][lod][terrain]")
{
    auto snapshot = make_snapshot("unused", 23, 65, 3, 33);
    snapshot.publication_generation = "0123456789abcdef0123456789abcdef";
    snapshot.grid.dx_m = 100.0;
    snapshot.grid.dy_m = 120.0;
    snapshot.grid.dz_m = 50.0;
    snapshot.grid.terrain_following = true;
    snapshot.grid.physical_z_center_m.resize(static_cast<std::size_t>(65 * 3 * 33));
    snapshot.grid.physical_z_interface_m.resize(static_cast<std::size_t>(65 * 3 * 34));
    for (int i = 0; i < 65; ++i)
        for (int j = 0; j < 3; ++j)
        {
            double height = 2.0 * i + 3.0 * j;
            const std::size_t interface_base =
                (static_cast<std::size_t>(i) * 3 + j) * 34;
            snapshot.grid.physical_z_interface_m[interface_base] = height;
            for (int k = 0; k < 33; ++k)
            {
                const double thickness = k % 2 == 0 ? 40.0 : 60.0;
                const std::size_t center = (static_cast<std::size_t>(i) * 3 + j) * 33 + k;
                snapshot.grid.physical_z_center_m[center] = height + 0.5 * thickness;
                height += thickness;
                snapshot.grid.physical_z_interface_m[interface_base + k + 1] = height;
            }
    snapshot.fields.front().data[0] = std::numeric_limits<float>::quiet_NaN();
        }
    bool geometry_seen = false;
    const auto publish = [&](int level,
                             const std::array<int, 3>& shape,
                             const std::vector<float>& values,
                             const nimbus::output::FieldBrickLodGeometry* geometry)
    {
        REQUIRE(level == 1);
        REQUIRE(shape == std::array<int, 3>{33, 2, 17});
        REQUIRE(geometry != nullptr);
        REQUIRE(geometry->x_m.size() == static_cast<std::size_t>(33 * 2 * 17));
        REQUIRE(std::isnan(values.front()));
        REQUIRE(std::isfinite(geometry->z_m.front()));
        REQUIRE(geometry->volume_m3.front() == Approx(100.0 * 120.0 * 400.0));
        geometry_seen = true;
        nimbus::output::FieldBrickLodPayloadIdentity identity{
            "test_field.lod1.npy", std::string(64, 'a'), std::nullopt};
        nimbus::output::FieldBrickLodPayloadIdentity::Geometry physical;
        physical.paths = {
            "physical_x_center_m.lod1.npy", "physical_y_center_m.lod1.npy",
            "physical_z_center_m.lod1.npy", "physical_volume_m3.lod1.npy"};
        physical.content_sha256 = {
            std::string(64, 'b'), std::string(64, 'c'),
            std::string(64, 'd'), std::string(64, 'e')};
        identity.geometry = physical;
        return identity;
    };
    const std::string json = nimbus::output::build_field_brick_hierarchy_json(
        snapshot.fields.front(), snapshot.grid, snapshot.publication_generation,
        true, false, 0.0, publish);
    REQUIRE(geometry_seen);
    REQUIRE(json.find("\"schemaVersion\": 3") != std::string::npos);
    REQUIRE(json.find("\"kind\": \"physical_volume_centroids\"") !=
            std::string::npos);
    REQUIRE(json.find("\"physical_volume_m3.lod1.npy\"") != std::string::npos);
}

TEST_CASE("completion manifest is published after durable field brick hierarchy",
          "[core][output_writer][brick-hierarchy][atomic]")
{
    const auto dir = make_tmp_dir("brick_hierarchy_atomic");
    const std::string generation = "0123456789abcdef0123456789abcdef";
    auto snapshot = make_snapshot(dir + "/step_000023", 23, 4, 3, 2);
    snapshot.publication_generation = generation;
    snapshot.data_product_kind = "retiring_swath_tile";
    snapshot.grid.dx_m = 100.0;
    snapshot.grid.dy_m = 100.0;
    snapshot.grid.dz_m = 50.0;
    snapshot.manifest_json =
        "{\"publication_state\":\"complete\",\"publication_generation\":\"" +
        generation + "\",\"step_index\":23,\"step_dir\":\"step_000023\","
        "\"brick_hierarchy\":{\"content_sha256\":\"" +
        field_brick_hierarchy_sha256_token("test_field") + "\"}}\n";
    OutputConfig config;
    config.format = OutputFormat::npy_3d;
    config.async_io = false;
    AsyncOutputWriter writer(config);
    REQUIRE(writer.submit(std::move(snapshot)));
    const auto hierarchy = std::filesystem::path(dir) / "step_000023" /
        "test_field.bricks.json";
    REQUIRE(std::filesystem::is_regular_file(hierarchy));
    REQUIRE(std::filesystem::is_regular_file(
        std::filesystem::path(dir) / "step_000023" / "manifest.json"));
    std::ifstream input(hierarchy);
    const std::string json{
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    REQUIRE(json.find("\"finiteCount\": 24") != std::string::npos);
    REQUIRE(json.find("\"nonFiniteCount\": 0") != std::string::npos);
    REQUIRE(json.find("\"extremaSemantics\": \"stored_values\"") != std::string::npos);
    std::ifstream manifest_input(
        std::filesystem::path(dir) / "step_000023" / "manifest.json");
    const std::string manifest{
        std::istreambuf_iterator<char>(manifest_input),
        std::istreambuf_iterator<char>()};
    REQUIRE(manifest.find("nimbus-pending-field-brick-sha256:") == std::string::npos);
    REQUIRE(manifest.find("\"content_sha256\":\"") != std::string::npos);
    cleanup_dir(dir);
}

TEST_CASE("completion manifest follows every durable reduced brick payload",
          "[core][output_writer][brick-hierarchy][lod][atomic]")
{
    const auto dir = make_tmp_dir("brick_hierarchy_lod_atomic");
    const std::string generation = "0123456789abcdef0123456789abcdef";
    auto snapshot = make_snapshot(dir + "/step_000023", 23, 130, 70, 35);
    snapshot.publication_generation = generation;
    snapshot.data_product_kind = "retiring_swath_tile";
    snapshot.grid.dx_m = 100.0;
    snapshot.grid.dy_m = 100.0;
    snapshot.grid.dz_m = 50.0;
    snapshot.manifest_json =
        "{\"publication_state\":\"complete\",\"publication_generation\":\"" +
        generation + "\",\"step_index\":23,\"step_dir\":\"step_000023\","
        "\"brick_hierarchy\":{\"content_sha256\":\"" +
        field_brick_hierarchy_sha256_token("test_field") + "\"}}\n";
    OutputConfig config;
    config.format = OutputFormat::npy_3d;
    config.async_io = false;
    AsyncOutputWriter writer(config);
    REQUIRE(writer.submit(std::move(snapshot)));
    const auto step = std::filesystem::path(dir) / "step_000023";
    REQUIRE(std::filesystem::is_regular_file(step / "test_field.lod1.npy"));
    REQUIRE(std::filesystem::is_regular_file(step / "test_field.lod2.npy"));
    std::ifstream hierarchy_input(step / "test_field.bricks.json");
    const std::string hierarchy{
        std::istreambuf_iterator<char>(hierarchy_input),
        std::istreambuf_iterator<char>()};
    REQUIRE(hierarchy.find("\"schemaVersion\": 2") != std::string::npos);
    REQUIRE(hierarchy.find("\"shape\": [65, 35, 18]") != std::string::npos);
    REQUIRE(hierarchy.find("\"shape\": [33, 18, 9]") != std::string::npos);
    REQUIRE(hierarchy.find("\"path\": \"test_field.lod2.npy\"") !=
            std::string::npos);
    REQUIRE(hierarchy.find("nimbus-pending") == std::string::npos);
    REQUIRE(std::filesystem::is_regular_file(step / "manifest.json"));
    cleanup_dir(dir);
}

TEST_CASE("committed history steps are immutable across writer restart",
          "[core][output_writer][publication][restart][failure-atomicity]")
{
    const auto dir = make_tmp_dir("committed_step_immutable");
    const auto step = std::filesystem::path(dir) / "step_000023";
    auto first = make_snapshot(step.string(), 23, 4, 3, 2);
    first.manifest_json = "{\"publication_state\":\"complete\"}\n";
    OutputConfig config;
    config.format = OutputFormat::npy_3d;
    config.async_io = false;
    {
        AsyncOutputWriter writer(config);
        REQUIRE(writer.submit(std::move(first)));
    }
    const auto field_path = step / "test_field.npy";
    const auto manifest_path = step / "manifest.json";
    const auto read_bytes = [](const std::filesystem::path& path)
    {
        std::ifstream input(path, std::ios::binary);
        return std::vector<char>{
            std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    };
    const auto field_before = read_bytes(field_path);
    const auto manifest_before = read_bytes(manifest_path);

    auto replacement = make_snapshot(step.string(), 23, 4, 3, 2);
    replacement.fields.front().data.front() = 999.0f;
    replacement.manifest_json = "{\"publication_state\":\"complete\",\"replacement\":true}\n";
    AsyncOutputWriter restarted(config);
    REQUIRE_FALSE(restarted.submit(std::move(replacement)));
    REQUIRE(restarted.has_error());
    REQUIRE(restarted.error_message().find("Refusing to overwrite committed history step") !=
            std::string::npos);
    const auto field_after = read_bytes(field_path);
    const auto manifest_after = read_bytes(manifest_path);
    REQUIRE(field_after == field_before);
    REQUIRE(manifest_after == manifest_before);
    cleanup_dir(dir);
}

#ifdef HAVE_NETCDF
TEST_CASE("netCDF history publishes a checksummed durable brick hierarchy and lossless LODs",
          "[core][output_writer][netcdf][brick-hierarchy][lod][atomic]")
{
    const auto dir = make_tmp_dir("netcdf_brick_hierarchy_lod_atomic");
    const std::string generation = "0123456789abcdef0123456789abcdef";
    auto snapshot = make_snapshot(dir + "/step_000023", 23, 130, 70, 35);
    snapshot.fields.front().name = "w";
    snapshot.publication_generation = generation;
    snapshot.grid.dx_m = 100.0;
    snapshot.grid.dy_m = 100.0;
    snapshot.grid.dz_m = 50.0;
    snapshot.grid.coordinate_system = "cartesian";
    snapshot.grid.native_staggering = "c_grid";
    snapshot.grid.domain_motion_mode = "stationary";
    snapshot.manifest_json =
        "{\"publication_state\":\"complete\",\"publication_generation\":\"" +
        generation + "\",\"step_index\":23,\"step_dir\":\"step_000023\","
        "\"brick_hierarchy\":{\"content_sha256\":\"" +
        field_brick_hierarchy_sha256_token("w") + "\"}}\n";

    OutputConfig config;
    config.format = OutputFormat::netcdf;
    config.async_io = false;
    config.netcdf_deflate_level = 1;
    config.netcdf_shuffle = true;
    config.netcdf_target_chunk_kib = 64;
    const bool borrow = GENERATE(false, true);
    std::vector<float> source;
    if (borrow)
    {
        source = std::move(snapshot.fields.front().data);
        snapshot.fields.front().borrow_for_synchronous_write(source.data(), source.size());
    }
    AsyncOutputWriter writer(config);
    REQUIRE(writer.submit(std::move(snapshot)));
    REQUIRE_FALSE(writer.has_error());

    const auto step = std::filesystem::path(dir) / "step_000023";
    REQUIRE(std::filesystem::is_regular_file(step / "history.nc"));
    REQUIRE(std::filesystem::is_regular_file(step / "w.lod1.npy"));
    REQUIRE(std::filesystem::is_regular_file(step / "w.bricks.json"));
    REQUIRE(std::filesystem::is_regular_file(step / "manifest.json"));
    REQUIRE_FALSE(std::filesystem::exists(step / "history.nc.tmp"));

    std::ifstream hierarchy_input(step / "w.bricks.json");
    const std::string hierarchy{
        std::istreambuf_iterator<char>(hierarchy_input),
        std::istreambuf_iterator<char>()};
    REQUIRE(hierarchy.find("\"schemaVersion\": 2") != std::string::npos);
    REQUIRE(hierarchy.find("\"source\": {\"kind\": \"primary_field\"") !=
            std::string::npos);
    REQUIRE(hierarchy.find("\"source\": {\"kind\": \"npy_lod\"") !=
            std::string::npos);
    REQUIRE(hierarchy.find("\"sourceShape\": [128, 68, 35]") != std::string::npos);
    REQUIRE(hierarchy.find("\"shape\": [64, 34, 18]") != std::string::npos);
    REQUIRE(hierarchy.find("\"path\": \"w.lod1.npy\"") != std::string::npos);
    REQUIRE(hierarchy.find("\"extremaSemantics\": \"stored_values\"") !=
            std::string::npos);

    std::ifstream manifest_input(step / "manifest.json");
    const std::string manifest{
        std::istreambuf_iterator<char>(manifest_input),
        std::istreambuf_iterator<char>()};
    REQUIRE(manifest.find("nimbus-pending-field-brick-sha256:") == std::string::npos);
    REQUIRE(manifest.find("\"content_sha256\":\"") != std::string::npos);
    REQUIRE(writer.snapshots_written() == 1);
    REQUIRE(writer.total_bytes_written() >
            std::filesystem::file_size(step / "history.nc"));
    cleanup_dir(dir);
}

TEST_CASE("terrain netCDF history publishes hierarchy-v3 volume-weighted LOD geometry",
          "[core][output_writer][netcdf][brick-hierarchy][lod][terrain][atomic]")
{
    const auto dir = make_tmp_dir("netcdf_terrain_brick_hierarchy");
    const std::string generation = "0123456789abcdef0123456789abcdef";
    auto snapshot = make_snapshot(dir + "/step_000023", 23, 67, 3, 33);
    snapshot.fields.front().name = "w";
    snapshot.publication_generation = generation;
    snapshot.grid.dx_m = 100.0;
    snapshot.grid.dy_m = 120.0;
    snapshot.grid.dz_m = 50.0;
    snapshot.grid.coordinate_system = "cartesian";
    snapshot.grid.native_staggering = "c_grid";
    snapshot.grid.domain_motion_mode = "stationary";
    snapshot.grid.terrain_following = true;
    snapshot.grid.physical_z_center_m.resize(static_cast<std::size_t>(67 * 3 * 33));
    snapshot.grid.physical_z_interface_m.resize(static_cast<std::size_t>(67 * 3 * 34));
    for (int i = 0; i < 67; ++i)
        for (int j = 0; j < 3; ++j)
        {
            double height = 2.0 * i + 3.0 * j;
            const std::size_t interface_base =
                (static_cast<std::size_t>(i) * 3 + j) * 34;
            snapshot.grid.physical_z_interface_m[interface_base] = height;
            for (int k = 0; k < 33; ++k)
            {
                const double thickness = k % 2 == 0 ? 40.0 : 60.0;
                const std::size_t center =
                    (static_cast<std::size_t>(i) * 3 + j) * 33 + k;
                snapshot.grid.physical_z_center_m[center] = height + 0.5 * thickness;
                height += thickness;
                snapshot.grid.physical_z_interface_m[interface_base + k + 1] = height;
            }
        }
    bind_terrain_geometry_identity(snapshot, generation);
    snapshot.manifest_json =
        "{\"publication_state\":\"complete\",\"publication_generation\":\"" +
        generation + "\",\"step_index\":23,\"step_dir\":\"step_000023\","
        "\"brick_hierarchy\":{\"content_sha256\":\"" +
        field_brick_hierarchy_sha256_token("w") + "\"}}\n";

    OutputConfig config;
    config.format = OutputFormat::netcdf;
    config.async_io = false;
    const bool borrow = GENERATE(false, true);
    std::vector<float> source;
    if (borrow)
    {
        source = std::move(snapshot.fields.front().data);
        snapshot.fields.front().borrow_for_synchronous_write(source.data(), source.size());
    }
    AsyncOutputWriter writer(config);
    REQUIRE(writer.submit(std::move(snapshot)));
    const auto step = std::filesystem::path(dir) / "step_000023";
    for (const std::string& name : {
             "history.nc", "w.lod1.npy", "physical_x_center_m.lod1.npy",
             "physical_y_center_m.lod1.npy", "physical_z_center_m.lod1.npy",
             "physical_volume_m3.lod1.npy", "w.bricks.json", "manifest.json"})
        REQUIRE(std::filesystem::is_regular_file(step / name));
    std::ifstream hierarchy_input(step / "w.bricks.json");
    const std::string hierarchy{
        std::istreambuf_iterator<char>(hierarchy_input),
        std::istreambuf_iterator<char>()};
    REQUIRE(hierarchy.find("\"schemaVersion\": 3") != std::string::npos);
    REQUIRE(hierarchy.find("\"kind\": \"physical_volume_centroids\"") !=
            std::string::npos);
    REQUIRE(hierarchy.find("\"physical_volume_m3.lod1.npy\"") !=
            std::string::npos);
    cleanup_dir(dir);
}
#endif

TEST_CASE("completion manifest follows shared durable terrain LOD geometry",
          "[core][output_writer][brick-hierarchy][lod][terrain][atomic]")
{
    const auto dir = make_tmp_dir("terrain_brick_lod_atomic");
    const std::string generation = "0123456789abcdef0123456789abcdef";
    auto snapshot = make_snapshot(dir + "/step_000023", 23, 65, 2, 33);
    snapshot.publication_generation = generation;
    snapshot.data_product_kind = "retiring_swath_tile";
    snapshot.grid.dx_m = 100.0;
    snapshot.grid.dy_m = 120.0;
    snapshot.grid.dz_m = 50.0;
    snapshot.grid.terrain_following = true;
    snapshot.grid.physical_z_center_m.resize(static_cast<std::size_t>(65 * 2 * 33));
    snapshot.grid.physical_z_interface_m.resize(static_cast<std::size_t>(65 * 2 * 34));
    for (int i = 0; i < 65; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 34; ++k)
            {
                const double interface = 2.0 * i + 3.0 * j + 50.0 * k;
                const std::size_t face = (static_cast<std::size_t>(i) * 2 + j) * 34 + k;
                snapshot.grid.physical_z_interface_m[face] = interface;
                if (k < 33)
                {
                    const std::size_t center = (static_cast<std::size_t>(i) * 2 + j) * 33 + k;
                    snapshot.grid.physical_z_center_m[center] = interface + 25.0;
                }
            }
    bind_terrain_geometry_identity(snapshot, generation);
    snapshot.manifest_json =
        "{\"publication_state\":\"complete\",\"publication_generation\":\"" +
        generation + "\",\"step_index\":23,\"step_dir\":\"step_000023\","
        "\"brick_hierarchy\":{\"content_sha256\":\"" +
        field_brick_hierarchy_sha256_token("test_field") + "\"}}\n";
    OutputConfig config;
    config.format = OutputFormat::npy_3d;
    config.async_io = false;
    AsyncOutputWriter writer(config);
    REQUIRE(writer.submit(std::move(snapshot)));
    const auto step = std::filesystem::path(dir) / "step_000023";
    for (const std::string& name : {
             "test_field.lod1.npy", "physical_x_center_m.lod1.npy",
             "physical_y_center_m.lod1.npy", "physical_z_center_m.lod1.npy",
             "physical_volume_m3.lod1.npy"})
        REQUIRE(std::filesystem::is_regular_file(step / name));
    std::ifstream hierarchy_input(step / "test_field.bricks.json");
    const std::string hierarchy{
        std::istreambuf_iterator<char>(hierarchy_input),
        std::istreambuf_iterator<char>()};
    REQUIRE(hierarchy.find("\"schemaVersion\": 3") != std::string::npos);
    REQUIRE(hierarchy.find("\"kind\": \"physical_volume_centroids\"") !=
            std::string::npos);
    REQUIRE(std::filesystem::is_regular_file(step / "manifest.json"));
    cleanup_dir(dir);
}

[[maybe_unused]] std::vector<double> read_npy_f64_3d(
    const std::filesystem::path& path,
    int expected_d0,
    int expected_d1,
    int expected_d2)
{
    std::ifstream file(path, std::ios::binary);
    REQUIRE(file.is_open());
    char magic[6] = {};
    file.read(magic, sizeof(magic));
    REQUIRE(std::string(magic + 1, magic + 6) == "NUMPY");
    char version[2] = {};
    file.read(version, sizeof(version));
    std::uint16_t header_length = 0;
    file.read(reinterpret_cast<char*>(&header_length), sizeof(header_length));
    std::string header(header_length, '\0');
    file.read(header.data(), static_cast<std::streamsize>(header.size()));
    REQUIRE(header.find("'descr': '<f8'") != std::string::npos);
    const std::string expected_shape = "(" + std::to_string(expected_d0) + ", "
        + std::to_string(expected_d1) + ", " + std::to_string(expected_d2) + ")";
    REQUIRE(header.find(expected_shape) != std::string::npos);
    std::vector<double> values(static_cast<std::size_t>(expected_d0)
                               * expected_d1 * expected_d2);
    file.read(reinterpret_cast<char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(double)));
    REQUIRE(file.good());
    return values;
}

} // namespace

TEST_CASE("AsyncOutputWriter sync mode writes files", "[core][output_writer][analytical]")
{
    auto dir = make_tmp_dir("sync");
    auto step_dir = dir + "/step_000000";
    std::filesystem::create_directories(step_dir);

    OutputConfig cfg;
    cfg.format = OutputFormat::npy_3d;
    cfg.async_io = false;

    AsyncOutputWriter writer(cfg);

    auto snap = make_snapshot(step_dir, 0, 4, 6, 8);
    REQUIRE(writer.submit(std::move(snap)));

    // File should exist
    auto npy_path = step_dir + "/test_field.npy";
    REQUIRE(std::filesystem::exists(npy_path));

    // File size: header + 4*6*8*4 = header + 768 bytes
    auto file_size = std::filesystem::file_size(npy_path);
    REQUIRE(file_size > 768);  // data + header overhead
    REQUIRE(file_size < 2000); // but not absurdly large

    REQUIRE(writer.snapshots_written() == 1);
    REQUIRE(writer.total_bytes_written() > 0);
    REQUIRE_FALSE(writer.has_error());

    cleanup_dir(dir);
}

TEST_CASE("AsyncOutputWriter atomically publishes the final step manifest",
          "[core][output_writer][manifest][atomic]")
{
    auto dir = make_tmp_dir("atomic_manifest");
    auto step_dir = dir + "/step_000000";
    std::filesystem::create_directories(step_dir);

    OutputConfig cfg;
    cfg.format = OutputFormat::npy_3d;
    cfg.async_io = false;
    AsyncOutputWriter writer(cfg);

    auto snap = make_snapshot(step_dir, 0, 2, 2, 2);
    snap.manifest_json = "{\"publication_state\":\"complete\"}\n";
    REQUIRE(writer.submit(std::move(snap)));

    const auto manifest = std::filesystem::path(step_dir) / "manifest.json";
    REQUIRE(std::filesystem::is_regular_file(manifest));
    REQUIRE_FALSE(std::filesystem::exists(manifest.string() + ".tmp"));
    std::ifstream in(manifest);
    const std::string contents((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
    REQUIRE(contents == "{\"publication_state\":\"complete\"}\n");

    cleanup_dir(dir);
}

TEST_CASE("history index recovers a manifest-last crash and rejects corrupt current state",
          "[core][output_writer][history_index][durability]")
{
    const auto dir = std::filesystem::path(make_tmp_dir("history_index"));
    const std::string generation = "0123456789abcdef0123456789abcdef";
    OutputConfig cfg;
    cfg.format = OutputFormat::npy_3d;
    cfg.async_io = false;
    AsyncOutputWriter writer(cfg);

    auto first = make_snapshot((dir / "step_000000").string(), 0, 2, 2, 2);
    first.manifest_json = indexed_manifest(generation, 0);
    REQUIRE(writer.submit(std::move(first)));
    const auto index_path = dir / "history_index.json";
    REQUIRE(std::filesystem::is_regular_file(index_path));
    REQUIRE_FALSE(std::filesystem::exists(index_path.string() + ".tmp"));

    // Simulate a crash after manifest-last publication but before its index
    // replacement. The next commit rebuilds from durable manifests.
    std::filesystem::create_directories(dir / "step_000001");
    std::string error;
    const std::string orphan_manifest = indexed_manifest(generation, 1);
    REQUIRE(nimbus::output::prepare_history_index_update(dir, orphan_manifest, error));
    std::ofstream(dir / "step_000001" / "manifest.json") << orphan_manifest;
    auto third = make_snapshot((dir / "step_000002").string(), 2, 2, 2, 2);
    third.manifest_json = indexed_manifest(generation, 2);
    REQUIRE(writer.submit(std::move(third)));
    std::ifstream input(index_path);
    const std::string index((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const std::string segment = all_history_segments(dir);
    REQUIRE(index.find("\"sequence\": 3") != std::string::npos);
    REQUIRE(segment.find("\"step_index\": 0") != std::string::npos);
    REQUIRE(segment.find("\"step_index\": 1") != std::string::npos);
    REQUIRE(segment.find("\"step_index\": 2") != std::string::npos);
    REQUIRE(index.find("\"checksum\":") != std::string::npos);

    std::ofstream(index_path, std::ios::trunc) << "{\"corrupt\":true}\n";
    auto fourth = make_snapshot((dir / "step_000003").string(), 3, 2, 2, 2);
    fourth.manifest_json = indexed_manifest(generation, 3);
    REQUIRE_FALSE(writer.submit(std::move(fourth)));
    REQUIRE(writer.error_message().find("history index") != std::string::npos);
    cleanup_dir(dir.string());
}

TEST_CASE("history retention atomically reindexes only surviving frames",
          "[core][output_writer][history_index][retention]")
{
    const auto dir = std::filesystem::path(make_tmp_dir("history_index_retention"));
    const std::string generation = "abcdef0123456789abcdef0123456789";
    OutputConfig cfg;
    cfg.format = OutputFormat::npy_3d;
    cfg.async_io = false;
    AsyncOutputWriter writer(cfg);
    for (int index = 0; index < 4; ++index)
    {
        auto snapshot = make_snapshot((dir / ("step_00000" + std::to_string(index))).string(), index, 2, 2, 2);
        snapshot.manifest_json = indexed_manifest(generation, index);
        REQUIRE(writer.submit(std::move(snapshot)));
    }
    nimbus::output::RetentionResult result;
    std::string error;
    REQUIRE(nimbus::output::enforce_step_retention(dir, 2, result, error));
    REQUIRE(result.removed_items == 2);
    std::ifstream input(dir / "history_index.json");
    const std::string index((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const std::string segment = current_history_segment(dir);
    REQUIRE(index.find("\"sequence\": 5") != std::string::npos);
    REQUIRE(segment.find("\"step_index\": 0") == std::string::npos);
    REQUIRE(segment.find("\"step_index\": 1") == std::string::npos);
    REQUIRE(segment.find("\"step_index\": 2") != std::string::npos);
    REQUIRE(segment.find("\"step_index\": 3") != std::string::npos);
    REQUIRE_FALSE(std::filesystem::exists(dir / "history_index.json.tmp"));
    cleanup_dir(dir.string());
}

TEST_CASE("history index rolls bounded append batches without rereading prior manifests",
          "[core][output_writer][history_index][rollover]")
{
    const auto dir = std::filesystem::path(make_tmp_dir("history_index_rollover"));
    const std::string generation = "123456789abcdef0123456789abcdef0";
    std::string error;
    for (int index = 0; index < 260; ++index)
    {
        const std::string manifest = indexed_manifest(generation, index);
        const auto step = dir / ("step_" + [&] {
            std::ostringstream value;
            value << std::setfill('0') << std::setw(6) << index;
            return value.str();
        }());
        std::filesystem::create_directories(step);
        std::ofstream(step / "manifest.json") << manifest;
        REQUIRE(nimbus::output::publish_history_index_after_manifest(dir, manifest, error));
        if (index == 0)
            std::ofstream(step / "manifest.json", std::ios::trunc) << "{corrupt-prior}";
    }
    std::size_t segment_files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir / "history_index"))
        if (entry.is_regular_file())
            ++segment_files;
    REQUIRE(segment_files <= 3);
    std::ifstream head_input(dir / "history_index.json");
    const std::string head((std::istreambuf_iterator<char>(head_input)), std::istreambuf_iterator<char>());
    REQUIRE(head.find("\"sequence\": 260") != std::string::npos);
    cleanup_dir(dir.string());
}

TEST_CASE("AsyncOutputWriter NPY roundtrip preserves exact field values", "[core][output_writer][analytical]")
{
    auto dir = make_tmp_dir("roundtrip");
    auto step_dir = dir + "/step_000000";
    std::filesystem::create_directories(step_dir);

    OutputConfig cfg;
    cfg.format = OutputFormat::npy_3d;
    cfg.async_io = false;

    AsyncOutputWriter writer(cfg);

    const int nr = 4, nth = 6, nz = 8;
    auto snap = make_snapshot(step_dir, 0, nr, nth, nz);

    // Save the original data for comparison
    std::vector<float> original = snap.fields[0].data;

    REQUIRE(writer.submit(std::move(snap)));

    // Read back the NPY file
    auto npy_path = step_dir + "/test_field.npy";
    std::ifstream f(npy_path, std::ios::binary);
    REQUIRE(f.is_open());

    // Skip NPY header (magic + version + header_len + header dict)
    char magic[6];
    f.read(magic, 6);
    REQUIRE(magic[0] == '\x93');

    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    uint16_t header_len;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    f.seekg(header_len, std::ios::cur);

    // Read data
    std::vector<float> readback(nr * nth * nz);
    f.read(reinterpret_cast<char*>(readback.data()), readback.size() * sizeof(float));
    REQUIRE(f.good());

    // Verify every value matches exactly (float32 roundtrip is lossless)
    for (size_t i = 0; i < original.size(); ++i)
    {
        REQUIRE(readback[i] == original[i]);
    }

    cleanup_dir(dir);
}

TEST_CASE("AsyncOutputWriter multi-field snapshot creates all files", "[core][output_writer][analytical]")
{
    auto dir = make_tmp_dir("multi");
    auto step_dir = dir + "/step_000000";
    std::filesystem::create_directories(step_dir);

    OutputConfig cfg;
    cfg.format = OutputFormat::npy_3d;
    cfg.async_io = false;

    AsyncOutputWriter writer(cfg);

    ExportSnapshot snap;
    snap.export_index = 0;
    snap.simulation_time_s = 0.0;
    snap.step_dir = step_dir;

    // Add 3 fields with different data
    for (const auto& name : {"u", "w", "theta"})
    {
        FieldSnapshotEntry entry;
        entry.name = name;
        entry.dim0 = 4;
        entry.dim1 = 4;
        entry.dim2 = 4;
        entry.is_3d = true;
        entry.data.assign(64, 1.0f);
        snap.fields.push_back(std::move(entry));
    }

    REQUIRE(writer.submit(std::move(snap)));

    // All 3 files should exist
    REQUIRE(std::filesystem::exists(step_dir + "/u.npy"));
    REQUIRE(std::filesystem::exists(step_dir + "/w.npy"));
    REQUIRE(std::filesystem::exists(step_dir + "/theta.npy"));

    REQUIRE(writer.snapshots_written() == 1);

    cleanup_dir(dir);
}

TEST_CASE("AsyncOutputWriter CSV mode writes valid CSV", "[core][output_writer][analytical]")
{
    auto dir = make_tmp_dir("csv");
    auto step_dir = dir + "/step_000000";
    std::filesystem::create_directories(step_dir);

    OutputConfig cfg;
    cfg.format = OutputFormat::csv;
    cfg.async_io = false;

    AsyncOutputWriter writer(cfg);

    ExportSnapshot snap;
    snap.export_index = 0;
    snap.simulation_time_s = 0.0;
    snap.step_dir = step_dir;

    FieldSnapshotEntry entry;
    entry.name = "test_csv";
    entry.dim0 = 2;
    entry.dim1 = 2;
    entry.dim2 = 2;
    entry.is_3d = true;
    entry.data = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 42.0f};
    snap.fields.push_back(std::move(entry));

    REQUIRE(writer.submit(std::move(snap)));

    auto csv_path = step_dir + "/test_csv.csv";
    REQUIRE(std::filesystem::exists(csv_path));

    // Read CSV and verify the non-zero value is present
    std::ifstream f(csv_path);
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    REQUIRE(content.find("42") != std::string::npos);
    REQUIRE(content.find("i,j,k,value") != std::string::npos);
    CHECK(writer.total_bytes_written() ==
          std::filesystem::file_size(csv_path));

    cleanup_dir(dir);
}

TEST_CASE("AsyncOutputWriter multiple snapshots accumulate statistics", "[core][output_writer][analytical]")
{
    auto dir = make_tmp_dir("multi_snap");

    OutputConfig cfg;
    cfg.format = OutputFormat::npy_3d;
    cfg.async_io = false;

    AsyncOutputWriter writer(cfg);

    for (int s = 0; s < 3; ++s)
    {
        auto step_dir = dir + "/step_" + std::to_string(s);
        std::filesystem::create_directories(step_dir);
        auto snap = make_snapshot(step_dir, s, 4, 4, 4);
        REQUIRE(writer.submit(std::move(snap)));
    }

    REQUIRE(writer.snapshots_written() == 3);
    REQUIRE(writer.total_bytes_written() > 0);

    // Each snapshot writes one 4x4x4 field = 256 bytes of data + header
    // Total bytes should be roughly 3 * (256 + header)
    REQUIRE(writer.total_bytes_written() >= 3 * 256);

    cleanup_dir(dir);
}

TEST_CASE("AsyncOutputWriter readiness drains the one-frame slot",
          "[core][output_writer][async][cpu016]")
{
    auto dir = make_tmp_dir("bounded_async");
    auto step_dir = dir + "/step_000000";
    std::filesystem::create_directories(step_dir);

    OutputConfig cfg;
    cfg.format = OutputFormat::npy_3d;
    cfg.async_io = true;
    AsyncOutputWriter writer(cfg);

    auto snap = make_snapshot(step_dir, 0, 32, 32, 16);
    REQUIRE(writer.submit(std::move(snap)));
    REQUIRE(writer.wait_until_ready());
    REQUIRE(writer.snapshots_written() == 1);
    REQUIRE(std::filesystem::is_regular_file(step_dir + "/test_field.npy"));
    REQUIRE(writer.flush());

    cleanup_dir(dir);
}

TEST_CASE("synchronous borrowed core fields preserve payloads and visualization sidecars",
          "[core][output_writer][borrowed]")
{
#ifdef HAVE_ZFP
    const auto format = GENERATE(OutputFormat::npy_3d, OutputFormat::csv, OutputFormat::zfp);
#else
    const auto format = GENERATE(OutputFormat::npy_3d, OutputFormat::csv);
#endif
    const auto dir = std::filesystem::path(make_tmp_dir("borrowed_payloads"));
    OutputConfig config;
    config.format = format;
    config.async_io = false;
    config.zfp_keyframe_interval = 4;
    config.zfp_predictive_delta = GENERATE(false, true);
    AsyncOutputWriter owned_writer(config);
    AsyncOutputWriter borrowed_writer(config);
    Field3D field(35, 9, 7);
    for (int frame = 0; frame < 4; ++frame)
    {
        for (std::size_t index = 0; index < field.size(); ++index)
            field.data()[index] = 300.0f + static_cast<float>((index * 13 + frame * 7) % 113) / 32.0f;
        std::ostringstream step;
        step << "step_" << std::setfill('0') << std::setw(6) << frame;
        auto owned = make_snapshot((dir / "owned" / step.str()).string(), frame, 35, 9, 7);
        owned.fields.clear();
        owned.publication_generation = "0123456789abcdef0123456789abcdef";
        owned.grid.dx_m = 50.0;
        owned.grid.dy_m = 50.0;
        owned.grid.dz_m = 100.0;
        owned.manifest_json = indexed_manifest(owned.publication_generation, frame);
        owned.manifest_json.insert(owned.manifest_json.rfind('}'),
            ",\"brick_hierarchy\":{\"content_sha256\":\"" +
            field_brick_hierarchy_sha256_token("theta") + "\"}");
        auto borrowed = owned;
        borrowed.step_dir = dir / "borrowed" / step.str();
        const std::vector<CoreVolumeFieldBinding<const Field3D*>> bindings{{"theta", &field}};
        write_or_snapshot_core_volume_fields(bindings, {"theta"}, owned, true,
            format, 35, 9, 7, owned.step_dir.string());
        write_or_snapshot_core_volume_fields(bindings, {"theta"}, borrowed, true,
            format, 35, 9, 7, borrowed.step_dir.string(), true);
        REQUIRE(borrowed.fields.front().values().data() == field.data());
        REQUIRE(borrowed.owned_bytes() == 0);
        REQUIRE(borrowed.total_bytes() == owned.total_bytes());
        REQUIRE(owned.owned_bytes() == field.size() * sizeof(float));
        const auto expected = owned.fields.front().data;
        REQUIRE(owned_writer.submit(std::move(owned)));
        REQUIRE(borrowed_writer.submit(std::move(borrowed)));
        REQUIRE(std::memcmp(field.data(), expected.data(), expected.size() * sizeof(float)) == 0);
        for (const auto& file : std::filesystem::directory_iterator(dir / "owned" / step.str()))
        {
            const auto counterpart = dir / "borrowed" / step.str() / file.path().filename();
            INFO(counterpart.string());
            REQUIRE(std::filesystem::is_regular_file(counterpart));
            REQUIRE(read_file_bytes(file.path()) == read_file_bytes(counterpart));
        }
        REQUIRE(std::filesystem::is_regular_file(dir / "borrowed" / step.str() / "theta.bricks.json"));
    }
    REQUIRE(owned_writer.snapshots_written() == 4);
    REQUIRE(borrowed_writer.snapshots_written() == 4);
    cleanup_dir(dir.string());
}

TEST_CASE("background output rejects borrowed fields before publication",
          "[core][output_writer][borrowed][async][atomic]")
{
    const auto dir = make_tmp_dir("borrowed_rejected");
    auto snapshot = make_snapshot(dir + "/never_created", 0, 4, 3, 2);
    auto storage = std::move(snapshot.fields.front().data);
    snapshot.fields.front().borrow_for_synchronous_write(storage.data(), storage.size());
    OutputConfig config;
    config.async_io = true;
    config.format = OutputFormat::npy_3d;
    AsyncOutputWriter writer(config);
    REQUIRE_FALSE(writer.submit(std::move(snapshot)));
    REQUIRE_FALSE(writer.flush());
    REQUIRE(writer.snapshots_written() == 0);
    REQUIRE(writer.error_message().find("requires owned field storage") != std::string::npos);
    REQUIRE_FALSE(std::filesystem::exists(dir + "/never_created"));
    cleanup_dir(dir);
}

TEST_CASE("borrowed field views retain ownership semantics through moves and copies",
          "[core][output_writer][borrowed]")
{
    FieldSnapshotEntry invalid;
    REQUIRE_THROWS_AS(invalid.borrow_for_synchronous_write(nullptr, 2), std::invalid_argument);
    invalid.data = {1.0f};
    REQUIRE_THROWS_AS(invalid.borrow_for_synchronous_write(invalid.data.data(), 1), std::invalid_argument);
    const std::vector<float> storage{1.25f, -0.0f, 3.5f};
    FieldSnapshotEntry borrowed;
    borrowed.borrow_for_synchronous_write(storage.data(), storage.size());
    auto copied = borrowed;
    auto moved = std::move(copied);
    REQUIRE(moved.is_borrowed());
    REQUIRE(moved.values().data() == storage.data());
    REQUIRE(moved.values().size() == storage.size());
    FieldSnapshotEntry owned;
    owned.data = storage;
    auto owned_copy = owned;
    REQUIRE_FALSE(owned_copy.is_borrowed());
    REQUIRE(owned_copy.values().data() != owned.values().data());
    REQUIRE(owned_copy.data == owned.data);
}

TEST_CASE("invalid borrowed output shape is rejected before any files are written",
          "[core][output_writer][borrowed][atomic]")
{
    const auto dir = make_tmp_dir("borrowed_shape");
    auto snapshot = make_snapshot(dir + "/never_created", 0, 4, 3, 2);
    auto storage = std::move(snapshot.fields.front().data);
    snapshot.fields.front().borrow_for_synchronous_write(storage.data(), storage.size() - 1);
    OutputConfig config;
    config.async_io = false;
    config.format = OutputFormat::npy_3d;
    AsyncOutputWriter writer(config);
    REQUIRE_FALSE(writer.submit(std::move(snapshot)));
    REQUIRE(writer.snapshots_written() == 0);
    REQUIRE_FALSE(std::filesystem::exists(dir + "/never_created"));
    cleanup_dir(dir);
}

TEST_CASE("surface snapshots borrow only when explicitly synchronous",
          "[core][output_writer][borrowed]")
{
    const bool borrow = GENERATE(false, true);
    std::vector<float> source(12, 0.125f);
    ExportSnapshot snapshot;
    std::string error;
    REQUIRE(write_or_snapshot_direct_surface_fields({"surface_sensible_heat_flux"},
        [&](std::string_view) { return &source; }, error, snapshot, true, 4, 3, "unused", borrow));
    REQUIRE(snapshot.fields.size() == 1);
    REQUIRE(snapshot.fields.front().is_borrowed() == borrow);
    REQUIRE(snapshot.owned_bytes() == (borrow ? 0 : 12 * sizeof(float)));
    source.front() = 1.0f;
    REQUIRE(snapshot.fields.front().values()[0] == (borrow ? 1.0f : 0.125f));
}

TEST_CASE("output retention removes only completed frames and preserves the newest tail",
          "[core][output][retention]")
{
    const auto dir = std::filesystem::path(make_tmp_dir("retention_tail"));
    OutputConfig cfg;
    cfg.format   = OutputFormat::npy_3d;
    cfg.async_io = false;
    AsyncOutputWriter writer(cfg);
    for (int index = 0; index < 4; ++index)
    {
        std::ostringstream name;
        name << "step_" << std::setfill('0') << std::setw(6) << index;
        auto snapshot = make_snapshot((dir / name.str()).string(), index, 2, 2, 2);
        snapshot.manifest_json = "{\"parent_step_indices\": []}\n";
        REQUIRE(writer.submit(std::move(snapshot)));
        std::ofstream(dir / name.str() / "w.lod1.npy", std::ios::binary)
            << "retained hierarchy dependency " << index;
    }
    const auto incomplete = dir / "step_000004";
    std::filesystem::create_directories(incomplete);
    std::ofstream(incomplete / "payload.tmp") << "not committed";
    const auto unrelated = dir / "notes";
    std::filesystem::create_directories(unrelated);

    nimbus::output::RetentionResult result;
    std::string error;
    REQUIRE(nimbus::output::enforce_step_retention(dir, 2, result, error));
    REQUIRE(error.empty());
    REQUIRE(result.removed_items == 2);
    REQUIRE_FALSE(std::filesystem::exists(dir / "step_000000"));
    REQUIRE_FALSE(std::filesystem::exists(dir / "step_000001"));
    REQUIRE(std::filesystem::is_directory(dir / "step_000002"));
    REQUIRE(std::filesystem::is_directory(dir / "step_000003"));
    REQUIRE(std::filesystem::is_regular_file(dir / "step_000002" / "w.lod1.npy"));
    REQUIRE(std::filesystem::is_regular_file(dir / "step_000003" / "w.lod1.npy"));
    REQUIRE(std::filesystem::is_directory(incomplete));
    REQUIRE(std::filesystem::is_directory(unrelated));
    cleanup_dir(dir.string());
}

TEST_CASE("output retention preserves manifest-declared compression dependency closure",
          "[core][output][retention][zfp]")
{
    const auto dir = std::filesystem::path(make_tmp_dir("retention_dependencies"));
    const std::array<const char*, 5> manifests = {
        "{\"parent_step_indices\": []}\n",
        "{\"parent_step_indices\": [0]}\n",
        "{\"parent_step_indices\": [1, 0]}\n",
        "{\"parent_step_indices\": []}\n",
        "{\"parent_step_indices\": [3]}\n",
    };
    for (int index = 0; index < 5; ++index)
    {
        std::ostringstream name;
        name << "step_" << std::setfill('0') << std::setw(6) << index;
        const auto step = dir / name.str();
        std::filesystem::create_directories(step);
        std::ofstream(step / "manifest.json") << manifests[static_cast<std::size_t>(index)];
        std::ofstream(step / "field.zfp3d", std::ios::binary) << "payload";
    }

    nimbus::output::RetentionResult result;
    std::string error;
    REQUIRE(nimbus::output::enforce_step_retention(dir, 1, result, error));
    REQUIRE(result.removed_items == 3);
    REQUIRE(result.dependency_frames == 1);
    REQUIRE_FALSE(std::filesystem::exists(dir / "step_000000"));
    REQUIRE_FALSE(std::filesystem::exists(dir / "step_000001"));
    REQUIRE_FALSE(std::filesystem::exists(dir / "step_000002"));
    REQUIRE(std::filesystem::is_directory(dir / "step_000003"));
    REQUIRE(std::filesystem::is_directory(dir / "step_000004"));
    cleanup_dir(dir.string());
}

TEST_CASE("output retention removes unreferenced immutable terrain geometry versions",
          "[core][output][retention][terrain]")
{
    const auto dir = std::filesystem::path(make_tmp_dir("retention_terrain_geometry"));
    const std::string generation = "0123456789abcdef0123456789abcdef";
    for (int index = 0; index < 3; ++index)
    {
        std::ostringstream step_name;
        step_name << "step_" << std::setfill('0') << std::setw(6) << index;
        const auto step = dir / step_name.str();
        std::filesystem::create_directories(step);
        const std::string digest(64, static_cast<char>('a' + index));
        std::ofstream(step / "manifest.json")
            << "{\"parent_step_indices\":[],\"vertical_geometry\":{"
            << "\"storage\":\"versioned_npy_geometry\","
            << "\"content_sha256\":\"" << digest << "\","
            << "\"publication_generation\":\"" << generation << "\"}}";
        const auto geometry = dir / "geometry" / generation / digest;
        std::filesystem::create_directories(geometry);
        std::ofstream(geometry / "manifest.json") << "geometry-" << index;
    }

    nimbus::output::RetentionResult result;
    std::string error;
    REQUIRE(nimbus::output::enforce_step_retention(dir, 1, result, error));
    REQUIRE(result.removed_items == 2);
    REQUIRE(result.removed_geometry_versions == 2);
    REQUIRE(std::filesystem::is_directory(
        dir / "geometry" / generation / std::string(64, 'c')));
    REQUIRE_FALSE(std::filesystem::exists(
        dir / "geometry" / generation / std::string(64, 'a')));
    cleanup_dir(dir.string());
}

TEST_CASE("output retention fails closed instead of traversing symbolic links",
          "[core][output][retention][symlink]")
{
    const auto dir = std::filesystem::path(make_tmp_dir("retention_symlink"));
    for (int index = 0; index < 2; ++index)
    {
        std::ostringstream name;
        name << "step_" << std::setfill('0') << std::setw(6) << index;
        const auto step = dir / name.str();
        std::filesystem::create_directories(step);
        std::ofstream(step / "manifest.json") << "{\"parent_step_indices\": []}\n";
    }
    std::filesystem::create_directory_symlink(dir / "step_000001", dir / "step_000000" / "unsafe_link");

    nimbus::output::RetentionResult result;
    std::string error;
    REQUIRE_FALSE(nimbus::output::enforce_step_retention(dir, 1, result, error));
    REQUIRE(error.find("symbolic link") != std::string::npos);
    REQUIRE(std::filesystem::is_directory(dir / "step_000000"));
    cleanup_dir(dir.string());
}

TEST_CASE("checkpoint retention keeps newest pairs and a protected branch parent",
          "[core][checkpoint][retention]")
{
    const auto dir = std::filesystem::path(make_tmp_dir("checkpoint_retention"));
    std::array<std::filesystem::path, 3> checkpoints;
    for (int index = 0; index < 3; ++index)
    {
        std::ostringstream name;
        name << "checkpoint_step_" << std::setfill('0') << std::setw(9) << index << ".nimbus";
        checkpoints[static_cast<std::size_t>(index)] = dir / name.str();
        std::ofstream(checkpoints[static_cast<std::size_t>(index)], std::ios::binary) << "checkpoint";
        std::ofstream(checkpoints[static_cast<std::size_t>(index)].string() + ".json") << "{}\n";
    }

    nimbus::output::RetentionResult result;
    std::string error;
    REQUIRE(nimbus::output::enforce_checkpoint_retention(dir, 1, checkpoints[0], result, error));
    REQUIRE(result.removed_items == 1);
    REQUIRE(std::filesystem::is_regular_file(checkpoints[0]));
    REQUIRE_FALSE(std::filesystem::exists(checkpoints[1]));
    REQUIRE_FALSE(std::filesystem::exists(checkpoints[1].string() + ".json"));
    REQUIRE(std::filesystem::is_regular_file(checkpoints[2]));
    cleanup_dir(dir.string());
}

TEST_CASE("free-space reserve guard reports an impossible request", "[core][output][retention]")
{
    const auto dir = std::filesystem::path(make_tmp_dir("free_space_guard"));
    std::error_code ec;
    const auto available = std::filesystem::space(dir, ec).available;
    REQUIRE_FALSE(ec);
    std::string error;
    REQUIRE(nimbus::output::require_free_space(dir, 1, error));
    REQUIRE_FALSE(nimbus::output::require_free_space(dir, available + 1, error));
    REQUIRE(error.find("insufficient free space") != std::string::npos);
    cleanup_dir(dir.string());
}

#ifdef HAVE_ZFP
TEST_CASE("keyframe-only ZFP does not allocate full-field scratch",
          "[core][output_writer][zfp][memory]")
{
    const auto dir = make_tmp_dir("zfp_keyframe_scratch");
    OutputConfig cfg;
    cfg.format = OutputFormat::zfp;
    cfg.async_io = false;
    cfg.zfp_keyframe_interval = 0;
    cfg.zfp_sparse_threshold = 0.0f;
    cfg.zfp_float16_prefilter = false;
    AsyncOutputWriter writer(cfg);

    auto snapshot = make_snapshot(dir + "/step_000000", 0, 8, 7, 6);
    REQUIRE(writer.submit(std::move(snapshot)));
    CHECK(writer.zfp_full_field_scratch_capacity_bytes() == 0);
    REQUIRE(writer.flush());
    cleanup_dir(dir);
}

TEST_CASE("accuracy ZFP publishes conservative decoded-value brick bounds without decoded field scratch",
          "[core][output_writer][zfp][brick-hierarchy]")
{
    const auto dir = make_tmp_dir("zfp_brick_bounds");
    const std::string generation = "0123456789abcdef0123456789abcdef";
    OutputConfig cfg;
    cfg.format = OutputFormat::zfp;
    cfg.async_io = false;
    cfg.zfp_keyframe_interval = 0;
    cfg.zfp_mode = "accuracy";
    cfg.zfp_tolerance = 0.001;
    cfg.zfp_sparse_threshold = 0.0f;
    cfg.zfp_float16_prefilter = false;
    AsyncOutputWriter writer(cfg);
    auto snapshot = make_snapshot(dir + "/step_000000", 0, 8, 7, 6);
    snapshot.fields.front().name = "w";
    snapshot.publication_generation = generation;
    snapshot.data_product_kind = "retiring_swath_tile";
    snapshot.grid.dx_m = 100.0;
    snapshot.grid.dy_m = 100.0;
    snapshot.grid.dz_m = 50.0;
    snapshot.manifest_json =
        "{\"publication_state\":\"complete\",\"publication_generation\":\"" +
        generation + "\",\"step_index\":0,\"step_dir\":\"step_000000\","
        "\"brick_hierarchy\":{\"content_sha256\":\"" +
        field_brick_hierarchy_sha256_token("w") + "\"}}\n";
    REQUIRE(writer.submit(std::move(snapshot)));
    REQUIRE(writer.zfp_full_field_scratch_capacity_bytes() == 0);
    std::ifstream hierarchy(std::filesystem::path(dir) / "step_000000" /
                            "w.bricks.json");
    const std::string json{
        std::istreambuf_iterator<char>(hierarchy),
        std::istreambuf_iterator<char>()};
    REQUIRE(json.find("\"extremaSemantics\": \"decoded_value_bounds\"") !=
            std::string::npos);
    REQUIRE(json.find("\"extremaAbsolutePadding\": 0.001") !=
            std::string::npos);
    REQUIRE(json.find("\"cullingSafe\": true") != std::string::npos);
    Zfp3dHeader header;
    std::vector<float> decoded;
    std::string decode_error;
    REQUIRE(read_zfp_3d(
        (std::filesystem::path(dir) / "step_000000" / "w.zfp3d").string(),
        header, decoded, decode_error));
    REQUIRE(decoded.size() == 8U * 7U * 6U);
    for (float value : decoded)
    {
        REQUIRE(value >= 0.5f - 0.001f);
        REQUIRE(value <= 7065.5f + 0.001f);
    }
    cleanup_dir(dir);
}

TEST_CASE("parallel ZFP preserves a manifest-committed frame when index append fails",
          "[core][output_writer][zfp][parallel][failure-atomicity][history_index]")
{
    const auto dir = make_tmp_dir("zfp_parallel_post_manifest_failure");
    const auto step_dir = dir + "/step_000000";
    OutputConfig cfg;
    cfg.format = OutputFormat::zfp;
    cfg.async_io = false;
    cfg.zfp_field_execution = "field_parallel";
    cfg.zfp_field_workers = 2;
    cfg.zfp_resolved_field_workers = 2;
    cfg.zfp_field_scratch_mib = 16;
    AsyncOutputWriter writer(cfg);
    auto snapshot = make_snapshot(step_dir, 0, 8, 7, 6);
    snapshot.manifest_json = indexed_manifest(
        "0123456789abcdef0123456789abcdef", 0);
    std::vector<float> source;
    if (GENERATE(false, true))
    {
        source = std::move(snapshot.fields.front().data);
        snapshot.fields.front().borrow_for_synchronous_write(source.data(), source.size());
    }
    set_history_index_failure_injection(true);
    const bool submitted = writer.submit(std::move(snapshot));
    set_history_index_failure_injection(false);
    REQUIRE_FALSE(submitted);
    REQUIRE(writer.has_error());
    CHECK(std::filesystem::is_regular_file(
        std::filesystem::path(step_dir) / "test_field.zfp3d"));
    CHECK(std::filesystem::is_regular_file(
        std::filesystem::path(step_dir) / "manifest.json"));
    CHECK(std::filesystem::is_regular_file(
        std::filesystem::path(dir) / "history_index.pending.json"));
    cleanup_dir(dir);
}

TEST_CASE("ZFP pre-manifest index failure leaves no completed publication or ledger change",
          "[core][output_writer][zfp][failure-atomicity][pre-manifest]")
{
    const auto dir = make_tmp_dir("zfp_pre_manifest_index_failure");
    const auto series = std::filesystem::path(dir);
    std::filesystem::create_symlink(
        "history_index.pending.json", series / "history_index.pending.json");
    const auto ledger = series / "budget.csv";
    {
        std::ofstream output(ledger);
        output << "preexisting-science-ledger\n";
    }
    const auto ledger_before = read_file_bytes(ledger);

    OutputConfig cfg;
    cfg.format = OutputFormat::zfp;
    cfg.async_io = true;
    cfg.zfp_mode = "accuracy";
    cfg.zfp_keyframe_interval = 4;
    cfg.zfp_predictive_delta = true;
    cfg.zfp_per_field_tolerances = true;
    cfg.zfp_field_execution = "serial";
    cfg.zfp_field_tolerances["w"] = 1.0e-4;
    cfg.keep_last_frames = 3;
    cfg.minimum_free_space_mib = 4096;
    cfg.maximum_retained_history_files = 256;
    AsyncOutputWriter writer(cfg);

    auto snapshot = make_snapshot(
        (series / "step_000000").string(), 0, 8, 7, 6);
    snapshot.fields.front().name = "w";
    snapshot.manifest_json = indexed_manifest(
        "0123456789abcdef0123456789abcdef", 0);
    REQUIRE(writer.submit(std::move(snapshot)));
    REQUIRE_FALSE(writer.flush());
    REQUIRE(writer.has_error());
    REQUIRE_FALSE(writer.error_message().empty());
    CHECK(writer.snapshots_written() == 0);
    CHECK_FALSE(std::filesystem::exists(series / "step_000000/manifest.json"));
    CHECK_FALSE(std::filesystem::exists(series / "history_index.json"));
    CHECK(read_file_bytes(ledger) == ledger_before);
    CHECK_FALSE(writer.submit(make_snapshot(
        (series / "step_000001").string(), 1, 8, 7, 6)));
    cleanup_dir(dir);
}

TEST_CASE("accuracy ZFP delta reconstruction remains inside published decoded brick bounds",
          "[core][output_writer][zfp][brick-hierarchy][delta]")
{
    const auto dir = make_tmp_dir("zfp_delta_brick_bounds");
    const std::string generation = "0123456789abcdef0123456789abcdef";
    OutputConfig cfg;
    cfg.format = OutputFormat::zfp;
    cfg.async_io = false;
    cfg.zfp_keyframe_interval = 8;
    cfg.zfp_predictive_delta = true;
    cfg.zfp_mode = "accuracy";
    cfg.zfp_tolerance = 0.001;
    AsyncOutputWriter writer(cfg);
    const auto make_frame = [&](int index, float offset)
    {
        auto snapshot = make_snapshot(
            dir + "/step_" + (index == 0 ? std::string("000000") : std::string("000001")),
            index, 8, 7, 6);
        snapshot.fields.front().name = "w";
        for (float& value : snapshot.fields.front().data) value += offset;
        snapshot.publication_generation = generation;
        snapshot.data_product_kind = "retiring_swath_tile";
        snapshot.grid.dx_m = 100.0;
        snapshot.grid.dy_m = 100.0;
        snapshot.grid.dz_m = 50.0;
        snapshot.manifest_json =
            "{\"publication_state\":\"complete\",\"publication_generation\":\"" +
            generation + "\",\"step_index\":" + std::to_string(index) +
            ",\"step_dir\":\"step_" + (index == 0 ? std::string("000000") : std::string("000001")) +
            "\",\"brick_hierarchy\":{\"content_sha256\":\"" +
            field_brick_hierarchy_sha256_token("w") + "\"}}\n";
        return snapshot;
    };
    REQUIRE(writer.submit(make_frame(0, 0.0f)));
    REQUIRE(writer.submit(make_frame(1, 1.0f)));
    ZfpFrameReader reader;
    std::vector<float> first;
    std::vector<float> second;
    std::string error;
    REQUIRE(reader.read_frame(
        "w", (std::filesystem::path(dir) / "step_000000" / "w.zfp3d").string(),
        first, error));
    REQUIRE(reader.read_frame(
        "w", (std::filesystem::path(dir) / "step_000001" / "w.zfp3d").string(),
        second, error));
    REQUIRE(second.size() == 8U * 7U * 6U);
    for (std::size_t index = 0; index < second.size(); ++index)
    {
        REQUIRE(first[index] >= 0.5f - 0.001f);
        REQUIRE(first[index] <= 7065.5f + 0.001f);
        REQUIRE(second[index] >= 1.5f - 0.001f);
        REQUIRE(second[index] <= 7066.5f + 0.001f);
    }
    std::ifstream second_hierarchy(
        std::filesystem::path(dir) / "step_000001" / "w.bricks.json");
    const std::string json{
        std::istreambuf_iterator<char>(second_hierarchy),
        std::istreambuf_iterator<char>()};
    REQUIRE(json.find("\"extremaSemantics\": \"decoded_value_bounds\"") !=
            std::string::npos);
    cleanup_dir(dir);
}

TEST_CASE("terrain history keeps exact physical geometry lossless beside ZFP fields",
          "[core][output_writer][terrain][zfp][TE-005]")
{
    const auto dir = make_tmp_dir("terrain_geometry");
    const auto step_dir = dir + "/step_000000";
    std::filesystem::create_directories(step_dir);

    OutputConfig cfg;
    cfg.format = OutputFormat::zfp;
    cfg.async_io = false;
    AsyncOutputWriter writer(cfg);

    auto snapshot = make_snapshot(step_dir, 0, 2, 2, 2);
    snapshot.grid.terrain_following = true;
    snapshot.grid.physical_z_center_m = {
        40.0, 170.0, 45.0, 185.0,
        90.0, 260.0, 95.0, 275.0,
    };
    snapshot.grid.physical_z_interface_m = {
        0.0, 80.0, 260.0, 0.0, 90.0, 280.0,
        50.0, 130.0, 390.0, 50.0, 140.0, 410.0,
    };
    const auto expected_centers = snapshot.grid.physical_z_center_m;
    const auto expected_interfaces = snapshot.grid.physical_z_interface_m;
    bind_terrain_geometry_identity(snapshot);
    const auto geometry_dir = std::filesystem::path(dir) / "geometry" /
        snapshot.grid.terrain_geometry_publication_generation /
        snapshot.grid.terrain_geometry_sha256;
    snapshot.manifest_json = "{\"publication_state\":\"complete\"}\n";

    REQUIRE(writer.submit(std::move(snapshot)));
    REQUIRE(std::filesystem::is_regular_file(std::filesystem::path(step_dir) / "test_field.zfp3d"));
    REQUIRE_FALSE(std::filesystem::exists(geometry_dir / "physical_z_center_m.zfp3d"));
    REQUIRE(read_npy_f64_3d(geometry_dir / "physical_z_center_m.npy", 2, 2, 2)
            == expected_centers);
    REQUIRE(read_npy_f64_3d(geometry_dir / "physical_z_interface_m.npy", 2, 2, 3)
            == expected_interfaces);
    REQUIRE(std::filesystem::is_regular_file(geometry_dir / "manifest.json"));
    REQUIRE(std::filesystem::is_regular_file(std::filesystem::path(step_dir) / "manifest.json"));
    cleanup_dir(dir);
}
#endif

TEST_CASE("terrain history rejects malformed geometry before completion publication",
          "[core][output_writer][terrain][fail-closed][TE-005]")
{
    const auto dir = make_tmp_dir("terrain_bad_geometry");
    const auto step_dir = dir + "/step_000000";
    OutputConfig cfg;
    cfg.format = OutputFormat::npy_3d;
    cfg.async_io = false;
    AsyncOutputWriter writer(cfg);

    auto snapshot = make_snapshot(step_dir, 0, 1, 1, 2);
    snapshot.grid.terrain_following = true;
    snapshot.grid.physical_z_center_m = {50.0, 125.0};
    snapshot.grid.physical_z_interface_m = {0.0, 100.0, 90.0};
    bind_terrain_geometry_identity(snapshot);
    snapshot.manifest_json = "{\"publication_state\":\"complete\"}\n";
    REQUIRE_FALSE(writer.submit(std::move(snapshot)));
    REQUIRE(writer.has_error());
    REQUIRE(writer.error_message().find("malformed") != std::string::npos);
    REQUIRE_FALSE(std::filesystem::exists(std::filesystem::path(step_dir) / "manifest.json"));
    cleanup_dir(dir);
}

TEST_CASE("terrain history reuses a static immutable geometry version and publishes changes",
          "[core][output_writer][terrain][versioned]")
{
    const auto dir = std::filesystem::path(make_tmp_dir("terrain_version_reuse"));
    OutputConfig cfg;
    cfg.format = OutputFormat::npy_3d;
    cfg.async_io = false;
    AsyncOutputWriter writer(cfg);

    auto first = make_snapshot((dir / "step_000000").string(), 0, 1, 1, 2);
    first.grid.terrain_following = true;
    first.grid.physical_z_center_m = {50.0, 150.0};
    first.grid.physical_z_interface_m = {0.0, 100.0, 200.0};
    bind_terrain_geometry_identity(first);
    const auto generation = first.grid.terrain_geometry_publication_generation;
    const auto static_identity = first.grid.terrain_geometry_sha256;
    first.manifest_json = "{\"publication_state\":\"complete\"}\n";
    REQUIRE(writer.submit(std::move(first)));
    const auto after_first = writer.total_bytes_written();

    auto second = make_snapshot((dir / "step_000001").string(), 1, 1, 1, 2);
    second.grid.terrain_following = true;
    second.grid.terrain_geometry_publication_generation = generation;
    second.grid.terrain_geometry_sha256 = static_identity;
    second.manifest_json = "{\"publication_state\":\"complete\"}\n";
    REQUIRE(second.total_bytes() == second.fields[0].data.size() * sizeof(float));
    REQUIRE(writer.submit(std::move(second)));
    const auto static_dir = dir / "geometry" / generation / static_identity;
    REQUIRE(std::filesystem::is_directory(static_dir));
    REQUIRE(std::distance(std::filesystem::directory_iterator(dir / "geometry" / generation),
                          std::filesystem::directory_iterator{}) == 1);
    REQUIRE(writer.total_bytes_written() - after_first < after_first);

    auto moved = make_snapshot((dir / "step_000002").string(), 2, 1, 1, 2);
    moved.grid.terrain_following = true;
    moved.grid.physical_z_center_m = {55.0, 155.0};
    moved.grid.physical_z_interface_m = {5.0, 105.0, 205.0};
    bind_terrain_geometry_identity(moved, generation);
    const auto moved_identity = moved.grid.terrain_geometry_sha256;
    REQUIRE(moved_identity != static_identity);
    moved.manifest_json = "{\"publication_state\":\"complete\"}\n";
    REQUIRE(writer.submit(std::move(moved)));
    REQUIRE(std::filesystem::is_directory(dir / "geometry" / generation / moved_identity));
    REQUIRE(std::distance(std::filesystem::directory_iterator(dir / "geometry" / generation),
                          std::filesystem::directory_iterator{}) == 2);
    cleanup_dir(dir.string());
}

TEST_CASE("terrain history rejects a corrupt reused geometry version before frame completion",
          "[core][output_writer][terrain][versioned][corruption]")
{
    const auto dir = std::filesystem::path(make_tmp_dir("terrain_version_corrupt"));
    OutputConfig cfg;
    cfg.format = OutputFormat::npy_3d;
    cfg.async_io = false;
    AsyncOutputWriter writer(cfg);
    auto first = make_snapshot((dir / "step_000000").string(), 0, 1, 1, 1);
    first.grid.terrain_following = true;
    first.grid.physical_z_center_m = {50.0};
    first.grid.physical_z_interface_m = {0.0, 100.0};
    bind_terrain_geometry_identity(first);
    const auto generation = first.grid.terrain_geometry_publication_generation;
    const auto identity = first.grid.terrain_geometry_sha256;
    first.manifest_json = "{\"publication_state\":\"complete\"}\n";
    REQUIRE(writer.submit(std::move(first)));
    std::ofstream(dir / "geometry" / generation / identity / "manifest.json",
                  std::ios::trunc) << "{corrupt";

    auto second = make_snapshot((dir / "step_000001").string(), 1, 1, 1, 1);
    second.grid.terrain_following = true;
    second.grid.terrain_geometry_publication_generation = generation;
    second.grid.terrain_geometry_sha256 = identity;
    second.manifest_json = "{\"publication_state\":\"complete\"}\n";
    REQUIRE_FALSE(writer.submit(std::move(second)));
    REQUIRE(writer.error_message().find("manifest is corrupt") != std::string::npos);
    REQUIRE_FALSE(std::filesystem::exists(dir / "step_000001" / "manifest.json"));
    cleanup_dir(dir.string());
}

TEST_CASE("ZFP delta histories are independent across root and nested child products",
          "[core][output_writer][zfp][nesting]")
{
    const auto dir = make_tmp_dir("zfp_nested_streams");
    OutputConfig cfg;
    cfg.format                = OutputFormat::zfp;
    cfg.async_io              = false;
    cfg.zfp_keyframe_interval = 8;
    cfg.zfp_predictive_delta  = true;
    AsyncOutputWriter writer(cfg);

    auto root0 = make_snapshot(dir + "/step_000000", 0, 4, 3, 2);
    root0.data_product_kind = "history";
    REQUIRE(writer.submit(std::move(root0)));

    // A different native shape and product must establish its own keyframe,
    // never attempt a delta against the root field with the same field name.
    auto child0 = make_snapshot(dir + "/nests/child_0000/step_000001", 1, 6, 5, 2);
    child0.data_product_kind = "nested_child_history";
    REQUIRE(writer.submit(std::move(child0)));

    auto root1 = make_snapshot(dir + "/step_000001", 1, 4, 3, 2);
    root1.data_product_kind = "history";
    REQUIRE(writer.submit(std::move(root1)));

    auto child1 = make_snapshot(dir + "/nests/child_0000/step_000002", 2, 6, 5, 2);
    child1.data_product_kind = "nested_child_history";
    REQUIRE(writer.submit(std::move(child1)));
    REQUIRE(writer.snapshots_written() == 4);
    REQUIRE(writer.flush());

    cleanup_dir(dir);
}

TEST_CASE("synchronous direct surface output fails closed when its NPY cannot be written",
          "[core][output_writer][surface][fail-closed]")
{
    const auto dir = make_tmp_dir("surface_write_failure");
    const std::string missing_step = dir + "/missing/step_000000";
    const std::unordered_set<std::string> active_fields = {
        "surface_sensible_heat_flux"};
    const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
    ExportSnapshot snapshot;
    std::string error_message;

    const bool ok = write_or_snapshot_direct_surface_fields(
        active_fields,
        [&](std::string_view field_id) -> const std::vector<float>*
        {
            return field_id == "surface_sensible_heat_flux" ? &values : nullptr;
        },
        error_message,
        snapshot,
        false,
        2,
        2,
        missing_step);

    REQUIRE_FALSE(ok);
    REQUIRE(error_message.find("failed to write") != std::string::npos);
    REQUIRE(error_message.find("surface_sensible_heat_flux") != std::string::npos);
    REQUIRE_FALSE(std::filesystem::exists(
        std::filesystem::path(missing_step) /
        "surface_sensible_heat_flux.npy"));
    cleanup_dir(dir);
}

TEST_CASE("synchronous planar tranche output fails closed when its NPY cannot be written",
          "[core][output_writer][tranche][fail-closed]")
{
    const auto dir = make_tmp_dir("tranche_write_failure");
    const std::string missing_step = dir + "/missing/step_000000";
    const std::vector<RegisteredDiagnosticProduct> products = {
        {
            "cape",
            [](int theta_index, std::vector<float>& values)
            {
                values = {
                    static_cast<float>(theta_index + 1),
                    static_cast<float>(theta_index + 2),
                };
                return true;
            },
        },
    };
    const std::unordered_set<std::string> active_fields = {"cape"};
    ExportSnapshot snapshot;
    std::string error_message;

    const bool ok = write_or_snapshot_column_tranche_fields(
        products,
        active_fields,
        [](const char*, std::vector<float>&, int) { return true; },
        error_message,
        snapshot,
        false,
        2,
        2,
        7,
        missing_step);

    REQUIRE_FALSE(ok);
    REQUIRE(error_message.find("failed to write") != std::string::npos);
    REQUIRE(error_message.find("cape") != std::string::npos);
    REQUIRE_FALSE(std::filesystem::exists(
        std::filesystem::path(missing_step) / "cape.npy"));
    cleanup_dir(dir);
}

TEST_CASE("qualification diagnostics require root and comparison timesteps to reach cadence",
          "[core][output_writer][qualification][cadence]")
{
    REQUIRE(qualification_diagnostics_cadence_reachable(30.0, 1.5));
    REQUIRE(qualification_diagnostics_cadence_reachable(30.0, 6.0));
    REQUIRE_FALSE(qualification_diagnostics_cadence_reachable(30.0, 12.0));
    REQUIRE_FALSE(qualification_diagnostics_cadence_reachable(30.0, 0.0));
}

TEST_CASE("qualification diagnostics reject contradictory child ownership",
          "[core][output_writer][qualification][ownership][fail-closed]")
{
    REQUIRE(qualification_diagnostics_child_ownership_consistent(false, false));
    REQUIRE(qualification_diagnostics_child_ownership_consistent(true, true));
    REQUIRE_FALSE(qualification_diagnostics_child_ownership_consistent(true, false));
    REQUIRE_FALSE(qualification_diagnostics_child_ownership_consistent(false, true));
}

TEST_CASE("Writer converts filesystem exceptions to a terminal output error",
          "[core][output_writer][exception]")
{
    for (bool asynchronous : {false, true})
    {
        const auto dir = make_tmp_dir(asynchronous ? "throw_async" : "throw_sync");
        std::filesystem::create_symlink("history_index.pending.json",
            std::filesystem::path(dir) / "history_index.pending.json");
        OutputConfig config;
        config.async_io = asynchronous;
        AsyncOutputWriter writer(config);
        auto snapshot = make_snapshot(dir + "/step_000000", 0, 4, 4, 4);
        snapshot.manifest_json = indexed_manifest("0123456789abcdef0123456789abcdef", 0);
        if (asynchronous) REQUIRE(writer.submit(std::move(snapshot)));
        else REQUIRE_FALSE(writer.submit(std::move(snapshot)));
        REQUIRE_FALSE(writer.flush());
        REQUIRE(writer.has_error());
        REQUIRE_FALSE(writer.error_message().empty());
        REQUIRE_FALSE(writer.wait_until_ready());
        REQUIRE_FALSE(writer.submit(make_snapshot(dir + "/step_000001", 1, 4, 4, 4)));
        cleanup_dir(dir);
    }
}

TEST_CASE("History index rejects oversized sparse artifacts before reading",
          "[core][history_index][memory]")
{
    const auto dir = make_tmp_dir("oversized_index");
    const auto path = std::filesystem::path(dir) / "history_index.json";
    std::ofstream(path).close();
    std::filesystem::resize_file(path, (1ULL << 30) + 1);
    std::string error;
    REQUIRE_FALSE(nimbus::output::publish_history_index_after_manifest(dir,
        indexed_manifest("0123456789abcdef0123456789abcdef", 0), error));
    REQUIRE(error.find("invalid history index artifact") != std::string::npos);
    cleanup_dir(dir);
}

TEST_CASE("parcel exports stop CIN at LFC and CAPE at EL and publish the owned skin",
          "[core][output][parcel][skin][science]")
{
    SimulationStorageState storage{{4, 4, 8, 1000, 1000, 0.5, 1000}};
    storage.grid.geometry.initialize(4, 4, 8, 1000, 1000, 1000,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& f = storage.fields;
    for (auto* field : {&f.rho, &f.p, &f.theta, &f.qv, &f.qc, &f.qr,
                       &f.qi, &f.qs, &f.qg, &f.qh, &f.u, &f.v, &f.w, &f.tke})
        field->resize(4, 4, 8, 0.0f);
    auto& d = storage.diagnostics;
    for (auto* field : {&d.radar_reflectivity, &d.vorticity_r,
                       &d.vorticity_theta, &d.vorticity_z})
        field->resize(4, 4, 8, 0.0f);
    f.rho.fill(1.0f);
    storage.base.theta0.assign(8, 300.0);
    storage.base.qv0.assign(8, 0.0);
    storage.base.rho0.assign(8, 1.0);
    storage.base.p0.assign(8, 100000.0);
    const double environment[] = {300, 300, 280, 290, 280, 280, 280, 280};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 8; ++k)
            {
                const double pressure = 100000.0 * std::exp(-0.1 * k);
                f.p(i, j, k) = static_cast<float>(pressure);
                f.theta(i, j, k) = static_cast<float>(environment[k]
                    * std::pow(100000.0 / pressure, 287.0 / 1004.0));
                f.qv(i, j, k) = tmv::derived::saturation_mixing_ratio_from_temp_pressure(300, 100000);
            }
    auto state = make_simulation_state_view(storage);
    std::vector<float> precipitation(16), rain(16), residual(16), wind(8);
    double rain_time = 0.0;
    std::unordered_set<std::string> selected{"cape", "cin", "lfc", "el", "skin_temperature"};
    tmv::diagnostics::TrancheProductRegistryContext context{
        32, -30, 120, precipitation, rain, residual, rain_time, wind, wind,
        selected, state, {1, 3, 1, 3, 0, 8}, f.u, f.v, f.w};
    const auto evaluate = [&]()
    {
        std::unordered_map<std::string, float> result;
        std::string error;
        REQUIRE(tmv::diagnostics::with_diagnostic_tranche_registry(context, error,
            [&](const auto& products)
            {
                for (const auto& product : products)
                    if (selected.count(product.field_id))
                    {
                        std::vector<float> values;
                        if (!product.compute_slice(1, values)) return false;
                        result[product.field_id] = values[1];
                    }
                return true;
            }));
        return result;
    };
    const auto baseline = evaluate();
    REQUIRE(baseline.at("lfc") > 0.0f);
    REQUIRE(baseline.at("el") > baseline.at("lfc"));
    REQUIRE(baseline.at("cape") > 0.0f);
    REQUIRE(baseline.at("cin") < 0.0f);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 5; k < 8; ++k)
                f.theta(i, j, k) += static_cast<float>(10.0
                    * std::pow(100000.0 / f.p(i, j, k), 287.0 / 1004.0));
    REQUIRE(evaluate() == baseline);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 5; k < 8; ++k)
                f.theta(i, j, k) = static_cast<float>(200.0
                    * std::pow(100000.0 / f.p(i, j, k), 287.0 / 1004.0));
    REQUIRE(evaluate() == baseline);
    auto& pbl = storage.runtime.boundary_layer;
    pbl.surface_config.Tsfc = 330.0;
    auto prescribed = evaluate();
    REQUIRE(prescribed.at("skin_temperature") == 330.0f);
    REQUIRE(prescribed.at("cape") == baseline.at("cape"));
    REQUIRE(prescribed.at("cin") == baseline.at("cin"));
    pbl.surface_config.Tsfc = -1.0;
    pbl.skin_theta_2d.assign(16, 310.0f);
    pbl.skin_captured = true;
    REQUIRE(evaluate().at("skin_temperature") == Approx(310.0));
    selected = {"lifted_index", "showalter_index", "k_index", "total_totals"};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 8; ++k)
                f.p(i,j,k) = 100000.0f - 100.0f*k;
    for (const auto& entry : evaluate()) REQUIRE(std::isnan(entry.second));
    const double pressure_levels[] = {100000, 92500, 80000, 75000, 65000, 55000, 45000, 35000};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 8; ++k)
            {
                const double pressure = pressure_levels[k];
                const double temperature = 300.0 + 40.0 * std::log(pressure / 100000.0);
                f.p(i,j,k) = static_cast<float>(pressure);
                f.theta(i,j,k) = static_cast<float>(temperature * std::pow(100000.0/pressure, 287.0/1004.0));
                f.qv(i,j,k) = 0.008f;
            }
    const auto supported = evaluate();
    for (const auto& entry : supported) REQUIRE(std::isfinite(entry.second));
    const double fraction = std::log(55000.0/50000.0) / std::log(55000.0/45000.0);
    const double level_height = 5500.0 + 1000.0 * fraction;
    const double lcl_depth = 125.0 * (300.0 - tmv::derived::dewpoint_from_temp_qv_pressure(300,0.008f,100000));
    const double parcel_at_500 = 300.0 - 0.0098 * std::min(level_height - 500.0, lcl_depth)
        - 0.006 * std::max(0.0, level_height - 500.0 - lcl_depth);
    REQUIRE(supported.at("lifted_index") == Approx(300.0 + 40.0*std::log(0.5) - parcel_at_500).margin(1.0e-4));

    auto& terrain = storage.runtime.terrain;
    terrain.config.scheme_id = "bell";
    terrain.metrics.z.resize(4,4,8,0.0);
    terrain.metrics.z_face.resize(4,4,9,0.0);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
        {
            for (int k = 0; k <= 8; ++k) terrain.metrics.z_face(i,j,k) = 7960.0 + 5.0*k;
            for (int k = 0; k < 8; ++k)
            {
                terrain.metrics.z(i,j,k) = 7962.5 + 5.0*k;
                f.u(i,j,k) = 2.5f + 5.0f*k;
                f.qc(i,j,k) = k == 3 ? 0.001f : 0.0f;
            }
        }
    selected = {"u10", "cloud_top_height", "cloud_base_height"};
    const auto mapped = evaluate();
    REQUIRE(mapped.at("u10") == Approx(10.0));
    REQUIRE(mapped.at("cloud_base_height") == Approx(17.5));
    REQUIRE(mapped.at("cloud_top_height") == Approx(17.5));
}

TEST_CASE("pressure-level missing values remain lossless in compressed histories",
          "[core][output_writer][zfp][missing]")
{
    const auto dir = make_tmp_dir("pressure_missing");
    OutputConfig config;
    config.format = OutputFormat::zfp;
    config.async_io = false;
    AsyncOutputWriter writer(config);
    auto snapshot = make_snapshot(dir + "/step_000000", 0, 2, 2, 2);
    snapshot.fields.clear();
    snapshot.fields.push_back({"lifted_index", {1.25f,
        std::numeric_limits<float>::quiet_NaN(), 2.5f, -3.0f}, 2, 2, 0, false});
    REQUIRE(writer.submit(std::move(snapshot)));
    REQUIRE(writer.flush());
    const auto path = std::filesystem::path(dir) / "step_000000" / "lifted_index.npy";
    REQUIRE(std::filesystem::is_regular_file(path));
    std::ifstream input(path, std::ios::binary);
    input.seekg(-static_cast<std::streamoff>(4 * sizeof(float)), std::ios::end);
    float values[4];
    input.read(reinterpret_cast<char*>(values), sizeof(values));
    REQUIRE(values[0] == 1.25f);
    REQUIRE(std::isnan(values[1]));
    REQUIRE(values[2] == 2.5f);
    REQUIRE(values[3] == -3.0f);
    cleanup_dir(dir);
}

TEST_CASE("ZFP missing-value storage is explicit and malformed shapes publish nothing",
          "[core][output_writer][zfp][missing][atomic]")
{
    for (const auto* policy : {"serial", "field_parallel"})
    {
        const auto dir = make_tmp_dir(std::string("missing_atomic_") + policy);
        OutputConfig config;
        config.format = OutputFormat::zfp;
        config.async_io = false;
        config.zfp_field_execution = policy;
        config.zfp_field_workers = std::string(policy) == "serial" ? 1 : 2;
        config.zfp_field_scratch_mib = 64;
        AsyncOutputWriter writer(config);
        auto valid = make_snapshot(dir + "/step_000000", 0, 2, 2, 2);
        valid.fields.front().name = "u";
        FieldSnapshotEntry missing;
        missing.name = "k_index";
        missing.data = {1.0f, std::numeric_limits<float>::quiet_NaN(),
                        2.0f, 3.0f};
        missing.dim0 = 2;
        missing.dim1 = 2;
        missing.is_3d = false;
        REQUIRE(nimbus::output::classify_zfp_field_storage(missing) ==
                nimbus::output::ZfpFieldStorage::LosslessNpy);
        valid.fields.push_back(missing);
        REQUIRE(writer.submit(std::move(valid)));
        REQUIRE(writer.flush());
        REQUIRE(std::filesystem::is_regular_file(
            std::filesystem::path(dir) / "step_000000/u.zfp3d"));
        REQUIRE(std::filesystem::is_regular_file(
            std::filesystem::path(dir) / "step_000000/k_index.npy"));
        REQUIRE_FALSE(std::filesystem::exists(
            std::filesystem::path(dir) / "step_000000/k_index.zfp3d"));

        OutputConfig invalid_config = config;
        AsyncOutputWriter invalid_writer(invalid_config);
        auto invalid = make_snapshot(dir + "/invalid/step_000001", 1, 2, 2, 2);
        invalid.fields.front().name = "u";
        missing.data.pop_back();
        REQUIRE(nimbus::output::classify_zfp_field_storage(missing) ==
                nimbus::output::ZfpFieldStorage::InvalidShape);
        invalid.fields.push_back(std::move(missing));
        REQUIRE_FALSE(invalid_writer.submit(std::move(invalid)));
        REQUIRE_FALSE(std::filesystem::exists(
            std::filesystem::path(dir) / "invalid/step_000001"));
        REQUIRE_FALSE(std::filesystem::exists(
            std::filesystem::path(dir) / "invalid/history_index.json"));
        cleanup_dir(dir);
    }
}

TEST_CASE("ZFP manifest declares lossless missing-value fallback per published field",
          "[core][output_writer][zfp][missing][manifest]")
{
    OutputStepManifestContext context;
    context.storage_format = "zfp";
    context.step_dir = "step_000000";
    context.publication_generation = "0123456789abcdef0123456789abcdef";
    context.published_fields = {"k_index", "u"};
    context.grid.nr = 2;
    context.grid.nth = 2;
    context.grid.nz = 2;
    OutputStepManifestCompression compression;
    compression.field_id = "u";
    compression.mode = "accuracy";
    compression.mode_parameter = 1.0e-4;
    compression.mode_parameter_units = "absolute_field_units";
    compression.error_contract = "zfp_absolute_accuracy_before_prefilters";
    compression.frame_role = "keyframe";
    context.compression.push_back(compression);
    const auto manifest = build_output_step_manifest_json(context);
    REQUIRE(manifest.find("\"lossy_compression_contract\": \"nimbus.zfp.per_field.v1\"") !=
            std::string::npos);
    const auto k_index = manifest.find("\"field_id\": \"k_index\"");
    REQUIRE(k_index != std::string::npos);
    const auto next_field = manifest.find("\n    }", k_index);
    REQUIRE(next_field != std::string::npos);
    const auto fallback = manifest.substr(k_index, next_field - k_index);
    REQUIRE(fallback.find("\"file_pattern\": \"k_index.npy\"") !=
            std::string::npos);
    REQUIRE(fallback.find("\"storage_format\": \"npy_2d\"") !=
            std::string::npos);
    REQUIRE(fallback.find("\"missing_value\": \"NaN\"") !=
            std::string::npos);
    REQUIRE(fallback.find("\"compression\"") == std::string::npos);
    const auto u = manifest.find("\"field_id\": \"u\"");
    REQUIRE(u != std::string::npos);
    const auto compressed = manifest.substr(u, manifest.find("\n    }", u) - u);
    REQUIRE(compressed.find("\"file_pattern\": \"u.zfp3d\"") !=
            std::string::npos);
    REQUIRE(compressed.find("\"compression\": {") != std::string::npos);
}

TEST_CASE("owned output shapes reject before any field publication", "[core][output_writer][owned][atomic]")
{
    for (auto format : {OutputFormat::npy_3d,OutputFormat::csv,OutputFormat::netcdf,OutputFormat::zfp})
        for (bool asynchronous : {false,true})
        {
            const auto dir=make_tmp_dir("owned_shape_rejected");
            OutputConfig config;config.format=format;config.async_io=asynchronous;
            AsyncOutputWriter writer(config);
            auto snapshot=make_snapshot(dir+"/step_000000",0,4,4,4);
            auto invalid=snapshot.fields.front();
            invalid.name="invalid_shape";
            invalid.dim0=5;
            snapshot.fields.push_back(std::move(invalid));
            if (asynchronous) REQUIRE(writer.submit(std::move(snapshot)));
            else REQUIRE_FALSE(writer.submit(std::move(snapshot)));
            REQUIRE_FALSE(writer.flush());
            REQUIRE(writer.has_error());
            CHECK(writer.error_message().find("invalid_shape")!=std::string::npos);
            CHECK(writer.error_message().find("inconsistent dimensions")!=std::string::npos);
            CHECK_FALSE(std::filesystem::exists(std::filesystem::path(dir)/"step_000000"));
            cleanup_dir(dir);
        }
}
