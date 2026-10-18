/**
 * @file test_zfp_roundtrip.cpp
 * @brief Comprehensive roundtrip tests for ZFP compression + delta encoding.
 *
 * Verifies:
 *   - Keyframe write/read roundtrip within tolerance
 *   - Simple delta sequence reconstruction
 *   - Predictive delta sequence reconstruction
 *   - Per-field tolerance is respected
 *   - Header parsing for v1, v2, and seekable-brick v3 formats
 *   - ZfpFrameReader stateful reconstruction
 *   - Error handling (missing keyframe, corrupted header)
 *   - Scratch buffer reuse (no per-frame allocation)
 */
#include "catch2/catch.hpp"
#include "core/output/zfp_reader.hpp"
#include "core/output/output_writer.hpp"
#include "core/output/output_config.hpp"
#include "core/output/step_manifest.hpp"
#include "core/field/field_snapshot.hpp"
#include "core/infra/allocation_telemetry.hpp"
#include "core/infra/physical_constants.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>

namespace
{

TEST_CASE("Allocation telemetry preserves over-aligned allocation ownership", "[core][allocation]")
{
    constexpr std::size_t bytes     = 137;
    constexpr auto        alignment = std::align_val_t{128};
    for (bool array : {false, true})
    {
        for (bool sized : {false, true})
        {
            tmv::allocation_telemetry::begin_measurement();
            void*      pointer     = array ? ::operator new[](bytes, alignment) : ::operator new(bytes, alignment);
            const auto allocations = tmv::allocation_telemetry::end_measurement();
            CHECK(reinterpret_cast<std::uintptr_t>(pointer) % 128 == 0);
            CHECK(allocations.heap_allocations == 1);
            CHECK(allocations.heap_requested_bytes == bytes);
            std::memset(pointer, 0x5a, bytes);
            if (array && sized)
                ::operator delete[](pointer, bytes, alignment);
            else if (array)
                ::operator delete[](pointer, alignment);
            else if (sized)
                ::operator delete(pointer, bytes, alignment);
            else
                ::operator delete(pointer, alignment);
        }
    }
}

std::filesystem::path make_tmp_dir(const std::string& name)
{
    auto dir = std::filesystem::temp_directory_path() / ("tmv_test_zfp_" + name);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

void cleanup_dir(const std::filesystem::path& dir);
std::vector<unsigned char> read_file_bytes(const std::filesystem::path& path);
std::vector<float> extract_window(const std::vector<float>& full,
                                  const std::array<std::size_t, 3>& shape,
                                  const std::array<std::size_t, 3>& origin,
                                  const std::array<std::size_t, 3>& extent);
std::vector<float> make_smooth_field(int dim0, int dim1, int dim2,
                                     float base, float amplitude);

TEST_CASE("ZFP v4 reversible writer preserves exact bricks and rejects insufficient scratch",
          "[core][zfp][v4][writer]")
{
    const auto dir = make_tmp_dir("v4_reversible_writer");
    constexpr int d0 = 65;
    constexpr int d1 = 3;
    constexpr int d2 = 33;
    std::vector<float> source(static_cast<std::size_t>(d0) * d1 * d2);
    for (std::size_t i = 0; i < source.size(); ++i)
        source[i] = static_cast<float>((i * 37) % 997) / 17.0f;
    source[0] = -0.0f;
    std::size_t bytes = 123;
    std::string error;
    const auto rejected = (dir / "rejected.zfp3d").string();
    REQUIRE_FALSE(write_zfp_3d_reversible(source.data(), d0, d1, d2,
                                         rejected, 64, bytes, error));
    REQUIRE_FALSE(std::filesystem::exists(rejected));
    REQUIRE(bytes == 123);
    const float one_value = 1.0f;
    const auto overflow_path = (dir / "overflow.zfp3d").string();
    REQUIRE_FALSE(write_zfp_3d_reversible(
        &one_value, std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
        overflow_path, 2U * 1024U * 1024U, bytes, error));
    REQUIRE(error.find("overflow") != std::string::npos);
    REQUIRE_FALSE(std::filesystem::exists(overflow_path));
    const auto path = (dir / "valid.zfp3d").string();
    error.clear();
    REQUIRE(write_zfp_3d_reversible(source.data(), d0, d1, d2,
                                   path, 2U * 1024U * 1024U, bytes, error));
    REQUIRE(bytes == std::filesystem::file_size(path));
    Zfp3dHeader header;
    std::vector<float> decoded;
    REQUIRE(read_zfp_3d(path, header, decoded, error));
    REQUIRE(header.version == 4);
    REQUIRE(header.mode == 3);
    REQUIRE(header.mode_param == 0.0);
    REQUIRE(header.flags == 0);
    REQUIRE(decoded.size() == source.size());
    REQUIRE(std::memcmp(decoded.data(), source.data(), source.size() * sizeof(float)) == 0);
    const auto window_origin = std::array<std::size_t, 3>{63, 1, 30};
    const auto window_extent = std::array<std::size_t, 3>{2, 2, 3};
    std::vector<float> window;
    REQUIRE(read_zfp_3d_window(path, window_origin, window_extent, header, window, error));
    REQUIRE(window == extract_window(source, {d0, d1, d2}, window_origin, window_extent));
    std::vector<float> brick;
    REQUIRE(read_zfp_3d_brick(path, {1, 0, 1}, header, brick, error));
    REQUIRE(brick == extract_window(source, {d0, d1, d2}, {64, 0, 32}, {1, 3, 1}));
    ZfpFrameReader frame_reader;
    decoded.clear();
    REQUIRE(frame_reader.read_frame("reflectivity_dbz", path, decoded, error));
    REQUIRE_FALSE(frame_reader.has_history("reflectivity_dbz"));
    REQUIRE(std::memcmp(decoded.data(), source.data(), source.size() * sizeof(float)) == 0);
    REQUIRE_FALSE(write_zfp_3d_reversible(source.data(), d0, d1, d2,
                                          path, 2U * 1024U * 1024U, bytes, error));
    source.back() = std::numeric_limits<float>::quiet_NaN();
    const auto late_rejected = (dir / "late_rejected.zfp3d").string();
    REQUIRE_FALSE(write_zfp_3d_reversible(source.data(), d0, d1, d2,
                                          late_rejected, 2U * 1024U * 1024U,
                                          bytes, error));
    REQUIRE_FALSE(std::filesystem::exists(late_rejected));
    for (const auto& entry : std::filesystem::directory_iterator(dir))
        REQUIRE(entry.path().filename().string().find(".v4.partial.") == std::string::npos);
    cleanup_dir(dir);
}

TEST_CASE("ZFP v4 sealed radar vectors and corruptions obey bitwise and atomic reads",
          "[core][zfp][v4][vectors]")
{
    const char* root_env = std::getenv("NIMBUS_TEST_ZFP_V4_VECTOR_ROOT");
    if (!root_env)
        return;
    const std::filesystem::path root(root_env);
    const std::array<const char*, 2> steps{"step_000000", "step_000001"};
    const std::array<const char*, 5> fields{
        "reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"};
    const auto dir = make_tmp_dir("v4_sealed_vectors");
    for (const auto* step : steps)
        for (const auto* field : fields)
        {
            const auto path = (root / step / (std::string(field) + ".zfp3d")).string();
            Zfp3dHeader header;
            std::vector<float> decoded;
            std::string error;
            INFO(path);
            REQUIRE(read_zfp_3d(path, header, decoded, error));
            REQUIRE(header.version == 4);
            REQUIRE(header.mode == 3);
            REQUIRE(header.flags == 0);
            std::vector<float> brick;
            REQUIRE(read_zfp_3d_brick(path, {0, 0, 0}, header, brick, error));
            REQUIRE(brick == extract_window(decoded,
                {static_cast<std::size_t>(header.dim0),
                 static_cast<std::size_t>(header.dim1),
                 static_cast<std::size_t>(header.dim2)},
                {0, 0, 0},
                {std::min<std::size_t>(64, header.dim0),
                 std::min<std::size_t>(64, header.dim1),
                 std::min<std::size_t>(32, header.dim2)}));
            const auto rewritten = (dir / (std::string(step) + "_" + field + ".zfp3d")).string();
            std::size_t bytes = 0;
            REQUIRE(write_zfp_3d_reversible(decoded.data(), header.dim0, header.dim1,
                                            header.dim2, rewritten,
                                            2U * 1024U * 1024U, bytes, error));
            REQUIRE(read_file_bytes(path) == read_file_bytes(rewritten));
        }
    const std::array<const char*, 14> corruptions{
        "bad_version", "bad_mode", "noncanonical_parameter", "delta_flag",
        "reserved", "brick_magic", "truncated_header", "truncated_directory",
        "payload_total_mismatch", "noncanonical_offset", "duplicate_offset",
        "digest_flip", "payload_flip", "oversized_brick"};
    for (const auto* name : corruptions)
    {
        const auto path = (root / "corruptions" / (std::string(name) + ".zfp3d")).string();
        INFO(path);
        Zfp3dHeader header;
        header.version = 99;
        std::vector<float> output{31.0f, -0.0f};
        std::string error;
        REQUIRE_FALSE(read_zfp_3d(path, header, output, error));
        REQUIRE(header.version == 99);
        REQUIRE(output.size() == 2);
        REQUIRE(output[0] == 31.0f);
        REQUIRE(std::signbit(output[1]));
        REQUIRE_FALSE(error.empty());
        Zfp3dHeader window_header;
        window_header.version = 98;
        std::vector<float> window_output{77.0f};
        REQUIRE_FALSE(read_zfp_3d_window(path, {0, 0, 0}, {1, 1, 1},
                                         window_header, window_output, error));
        REQUIRE(window_header.version == 98);
        REQUIRE(window_output == std::vector<float>{77.0f});
    }
    cleanup_dir(dir);
}

TEST_CASE("ZFP v4 test-only mixed frames keep radar reversible and non-radar v3",
          "[core][zfp][v4][mixed]")
{
    constexpr int d0 = 12;
    constexpr int d1 = 12;
    constexpr int d2 = 48;
    const std::array<const char*, 5> radar_fields{
        "reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"};
    const auto serial_dir = make_tmp_dir("v4_mixed_serial");
    const auto parallel_dir = make_tmp_dir("v4_mixed_parallel");
    const auto legacy_dir = make_tmp_dir("v4_mixed_default_off");
    OutputConfig serial_config;
    serial_config.format = OutputFormat::zfp;
    serial_config.async_io = false;
    serial_config.zfp_mode = "accuracy";
    serial_config.zfp_tolerance = 1.0e-3;
    serial_config.zfp_keyframe_interval = 4;
    serial_config.zfp_predictive_delta = true;
    serial_config.zfp_field_scratch_mib = 64;
    serial_config.zfp_reversible_radar_v4 = true;
    OutputConfig parallel_config = serial_config;
    parallel_config.zfp_field_execution = "field_parallel";
    parallel_config.zfp_resolved_field_workers = 4;
    OutputConfig legacy_config = serial_config;
    legacy_config.zfp_reversible_radar_v4 = false;
    AsyncOutputWriter serial(serial_config);
    AsyncOutputWriter parallel(parallel_config);
    AsyncOutputWriter legacy(legacy_config);
    const char* root_env = std::getenv("NIMBUS_TEST_ZFP_V4_VECTOR_ROOT");
    const auto source = [&](const char* field, int frame)
    {
        if (root_env)
        {
            const auto path = std::filesystem::path(root_env) /
                (frame == 0 ? "step_000000" : "step_000001") /
                (std::string(field) + ".zfp3d");
            Zfp3dHeader header;
            std::vector<float> decoded;
            std::string error;
            REQUIRE(read_zfp_3d(path.string(), header, decoded, error));
            return decoded;
        }
        const std::string id(field);
        const float base = id == "reflectivity_dbz" ? 20.0f :
            id == "rhohv" ? 0.9f : id == "kdp" ? 0.5f : 1.0f;
        const float amplitude = id == "rhohv" ? 0.01f : 0.5f;
        return make_smooth_field(d0, d1, d2, base, amplitude);
    };
    for (int frame = 0; frame < 2; ++frame)
    {
        const auto make_snapshot = [&](const std::filesystem::path& root)
        {
            ExportSnapshot snapshot;
            snapshot.export_index = frame;
            snapshot.step_dir = root / ("step_" + std::to_string(frame));
            for (const auto* field : radar_fields)
            {
                FieldSnapshotEntry entry;
                entry.name = field;
                entry.dim0 = d0;
                entry.dim1 = d1;
                entry.dim2 = d2;
                entry.is_3d = true;
                entry.data = source(field, frame);
                snapshot.fields.push_back(std::move(entry));
            }
            FieldSnapshotEntry theta;
            theta.name = "theta";
            theta.dim0 = d0;
            theta.dim1 = d1;
            theta.dim2 = d2;
            theta.is_3d = true;
            theta.data = make_smooth_field(d0, d1, d2, 300.0f + frame, 4.0f);
            snapshot.fields.push_back(std::move(theta));
            return snapshot;
        };
        REQUIRE(serial.submit(make_snapshot(serial_dir)));
        REQUIRE(parallel.submit(make_snapshot(parallel_dir)));
        if (frame == 0)
            REQUIRE(legacy.submit(make_snapshot(legacy_dir)));
        for (const auto* field : radar_fields)
        {
            const auto filename = std::string(field) + ".zfp3d";
            const auto serial_path = serial_dir / ("step_" + std::to_string(frame)) / filename;
            const auto parallel_path = parallel_dir / ("step_" + std::to_string(frame)) / filename;
            REQUIRE(read_file_bytes(serial_path) == read_file_bytes(parallel_path));
            if (root_env)
            {
                const auto vector_path = std::filesystem::path(root_env) /
                    (frame == 0 ? "step_000000" : "step_000001") / filename;
                REQUIRE(read_file_bytes(serial_path) == read_file_bytes(vector_path));
            }
            Zfp3dHeader header;
            std::vector<float> decoded;
            std::string error;
            REQUIRE(read_zfp_3d(serial_path.string(), header, decoded, error));
            REQUIRE(header.version == 4);
            REQUIRE(header.mode == 3);
            REQUIRE(header.flags == 0);
            const auto expected = source(field, frame);
            REQUIRE(std::memcmp(decoded.data(), expected.data(),
                                expected.size() * sizeof(float)) == 0);
        }
        Zfp3dHeader theta_header;
        std::vector<float> theta_data;
        std::string error;
        const auto serial_theta = serial_dir / ("step_" + std::to_string(frame)) / "theta.zfp3d";
        const auto parallel_theta = parallel_dir / ("step_" + std::to_string(frame)) / "theta.zfp3d";
        REQUIRE(read_file_bytes(serial_theta) == read_file_bytes(parallel_theta));
        REQUIRE(read_zfp_3d(serial_theta.string(), theta_header, theta_data, error));
        REQUIRE(theta_header.version == 3);
        REQUIRE(theta_header.mode == 0);
    }
    Zfp3dHeader legacy_header;
    std::string error;
    REQUIRE(read_zfp_3d_header((legacy_dir / "step_0" / "reflectivity_dbz.zfp3d").string(),
                               legacy_header, error));
    REQUIRE(legacy_header.version == 3);
    cleanup_dir(serial_dir);
    cleanup_dir(parallel_dir);
    cleanup_dir(legacy_dir);
}

TEST_CASE("ZFP v4 typed mixed manifest record is exact and fail closed",
          "[core][zfp][v4][manifest]")
{
    REQUIRE_THROWS_AS(make_reversible_zfp_manifest_compression("theta"),
                      std::invalid_argument);
    OutputStepManifestContext context;
    context.step_dir = "step_000000";
    context.storage_format = "zfp";
    context.storage_includes_model_halos = false;
    context.grid.nr = 12;
    context.grid.nth = 12;
    context.grid.nz = 48;
    context.published_fields = {"reflectivity_dbz", "theta"};
    context.radar.enabled = true;
    context.radar.products = {"reflectivity_dbz"};
    context.compression.push_back(
        make_reversible_zfp_manifest_compression("reflectivity_dbz"));
    OutputStepManifestCompression theta;
    theta.field_id = "theta";
    theta.mode = "accuracy";
    theta.mode_parameter = 1.0e-3;
    theta.mode_parameter_units = "absolute_field_units";
    theta.error_contract = "zfp_absolute_accuracy_before_prefilters";
    theta.frame_role = "keyframe";
    context.compression.push_back(theta);
    const auto mixed = build_output_step_manifest_json(context);
    REQUIRE(mixed.find("\"compression_contract\": \"nimbus.zfp.per_field.v2\"") !=
            std::string::npos);
    REQUIRE(mixed.find("\"contract_version\": 2") != std::string::npos);
    REQUIRE(mixed.find("\"mode\": \"reversible\"") != std::string::npos);
    REQUIRE(mixed.find("\"container_version\": 4") != std::string::npos);
    REQUIRE(mixed.find("\"container_version\": 3") != std::string::npos);
    REQUIRE(mixed.find("\"lossy_compression_contract\"") == std::string::npos);
    context.compression[0].frame_role = "simple_delta";
    REQUIRE_THROWS_AS(build_output_step_manifest_json(context), std::invalid_argument);
    context.compression[0] = make_reversible_zfp_manifest_compression("reflectivity_dbz");
    context.compression[0].mode_parameter = -0.0;
    REQUIRE_THROWS_AS(build_output_step_manifest_json(context), std::invalid_argument);
    context.compression[0] = make_reversible_zfp_manifest_compression("reflectivity_dbz");
    context.radar.products = {"rhohv"};
    REQUIRE_THROWS_AS(build_output_step_manifest_json(context), std::invalid_argument);
    context.radar.products = {"reflectivity_dbz"};
    context.compression.erase(context.compression.begin());
    const auto legacy = build_output_step_manifest_json(context);
    REQUIRE(legacy.find("\"lossy_compression_contract\": \"nimbus.zfp.per_field.v1\"") !=
            std::string::npos);
    REQUIRE(legacy.find("\"contract_version\"") == std::string::npos);
}

TEST_CASE("ZFP v4 candidate rejects invalid radar values and scratch without publication",
          "[core][zfp][v4][failure]")
{
    const auto dir = make_tmp_dir("v4_candidate_failure");
    for (const auto* policy : {"serial", "field_parallel"})
    {
        OutputConfig config;
        config.format = OutputFormat::zfp;
        config.async_io = false;
        config.zfp_mode = "accuracy";
        config.zfp_reversible_radar_v4 = true;
        config.zfp_field_execution = policy;
        config.zfp_resolved_field_workers = 2;
        config.zfp_field_scratch_mib = 64;
        AsyncOutputWriter writer(config);
        ExportSnapshot snapshot;
        snapshot.step_dir = dir / (std::string(policy) + "_invalid");
        for (const auto* field : {"reflectivity_dbz", "rhohv", "theta"})
        {
            FieldSnapshotEntry entry;
            entry.name = field;
            entry.dim0 = 2;
            entry.dim1 = 2;
            entry.dim2 = 2;
            entry.is_3d = true;
            entry.data.assign(8, std::string(field) == "rhohv" ? 1.2f :
                                 std::string(field) == "theta" ? 300.0f : 20.0f);
            snapshot.fields.push_back(std::move(entry));
        }
        REQUIRE_FALSE(writer.submit(std::move(snapshot)));
        REQUIRE(writer.error_message().find("outside the field contract") != std::string::npos);
        REQUIRE_FALSE(std::filesystem::exists(dir / (std::string(policy) + "_invalid")));
        REQUIRE(writer.snapshots_written() == 0);
    }
    OutputConfig limited;
    limited.format = OutputFormat::zfp;
    limited.async_io = false;
    limited.zfp_mode = "accuracy";
    limited.zfp_reversible_radar_v4 = true;
    limited.zfp_field_scratch_mib = 1;
    AsyncOutputWriter writer(limited);
    ExportSnapshot snapshot;
    snapshot.step_dir = dir / "scratch_rejected";
    for (const auto* field : {"reflectivity_dbz", "theta"})
    {
        FieldSnapshotEntry entry;
        entry.name = field;
        entry.dim0 = 64;
        entry.dim1 = 64;
        entry.dim2 = 32;
        entry.is_3d = true;
        entry.data.assign(64U * 64U * 32U,
                          std::string(field) == "theta" ? 300.0f : 20.0f);
        snapshot.fields.push_back(std::move(entry));
    }
    REQUIRE_FALSE(writer.submit(std::move(snapshot)));
    REQUIRE_FALSE(std::filesystem::exists(dir / "scratch_rejected" / "manifest.json"));
    REQUIRE_FALSE(std::filesystem::exists(dir / "scratch_rejected" / "reflectivity_dbz.zfp3d"));
    REQUIRE_FALSE(std::filesystem::exists(dir / "history_index.json"));
    REQUIRE(writer.snapshots_written() == 0);
    cleanup_dir(dir);
}

void cleanup_dir(const std::filesystem::path& dir)
{
    std::filesystem::remove_all(dir);
}

std::vector<unsigned char> read_file_bytes(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

/// Create a 3D field with a smooth, physically-motivated pattern.
/// value(i,j,k) = base + amplitude * sin(pi*i/dim0) * cos(pi*j/dim1) * (k/dim2)
/// This gives a smooth field that compresses well and produces small deltas.
std::vector<float> make_smooth_field(int dim0, int dim1, int dim2, float base, float amplitude)
{
    const std::size_t  n = static_cast<std::size_t>(dim0) * dim1 * dim2;
    std::vector<float> data(n);
    for (int i = 0; i < dim0; ++i)
    {
        const float si = std::sin(static_cast<float>(physical_constants::pi) * i / dim0);
        for (int j = 0; j < dim1; ++j)
        {
            const float cj = std::cos(static_cast<float>(physical_constants::pi) * j / dim1);
            for (int k = 0; k < dim2; ++k)
            {
                const float       zfrac = static_cast<float>(k) / dim2;
                const std::size_t idx   = i * dim1 * dim2 + j * dim2 + k;
                data[idx]               = base + amplitude * si * cj * zfrac;
            }
        }
    }
    return data;
}

/// Create a time-evolved version of the smooth field.
/// Adds a small perturbation that grows linearly with time_step,
/// simulating a slowly evolving field (ideal for delta encoding).
std::vector<float> make_evolved_field(int dim0, int dim1, int dim2, float base, float amplitude, int time_step)
{
    auto        data = make_smooth_field(dim0, dim1, dim2, base, amplitude);
    const float dt   = 0.01f * time_step;
    for (std::size_t i = 0; i < data.size(); ++i)
    {
        // Small linear drift + tiny oscillation
        data[i] += dt * (0.1f + 0.05f * std::sin(static_cast<float>(i) * 0.01f));
    }
    return data;
}

/// Compute max absolute error between two buffers.
float max_abs_error(const std::vector<float>& a, const std::vector<float>& b)
{
    REQUIRE(a.size() == b.size());
    float max_err = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
    }
    return max_err;
}

std::vector<float> extract_window(const std::vector<float>&         full,
                                  const std::array<std::size_t, 3>& shape,
                                  const std::array<std::size_t, 3>& origin,
                                  const std::array<std::size_t, 3>& extent)
{
    std::vector<float> window(extent[0] * extent[1] * extent[2]);
    for (std::size_t i = 0; i < extent[0]; ++i)
        for (std::size_t j = 0; j < extent[1]; ++j)
            for (std::size_t k = 0; k < extent[2]; ++k)
                window[(i * extent[1] + j) * extent[2] + k] =
                    full[((origin[0] + i) * shape[1] + origin[1] + j) * shape[2] + origin[2] + k];
    return window;
}

/// Build an ExportSnapshot with a single named field.
ExportSnapshot make_single_field_snapshot(const std::filesystem::path& step_dir,
                                          int                          index,
                                          const std::string&           field_name,
                                          const std::vector<float>&    data,
                                          int                          dim0,
                                          int                          dim1,
                                          int                          dim2)
{
    ExportSnapshot snap;
    snap.export_index      = index;
    snap.simulation_time_s = index * 5.0;
    snap.step_dir          = step_dir;

    FieldSnapshotEntry entry;
    entry.name  = field_name;
    entry.dim0  = dim0;
    entry.dim1  = dim1;
    entry.dim2  = dim2;
    entry.is_3d = true;
    entry.data  = data;
    snap.fields.push_back(std::move(entry));

    return snap;
}

struct QualifiedFieldPattern
{
    const char* name;
    float       base;
    float       amplitude;
};

constexpr std::array<QualifiedFieldPattern, 8> kQualifiedFieldPatterns = {{
    {"u", 5.0f, 25.0f},
    {"v", -3.0f, 20.0f},
    {"w", 0.0f, 10.0f},
    {"theta", 300.0f, 20.0f},
    {"p", 85000.0f, 10000.0f},
    {"rho", 1.0f, 0.2f},
    {"qv", 0.008f, 0.004f},
    {"qr", 0.0004f, 0.0003f},
}};

std::vector<float>
make_qualified_field(const QualifiedFieldPattern& pattern, int dim0, int dim1, int dim2, int frame, double tolerance)
{
    auto        values    = make_smooth_field(dim0, dim1, dim2, pattern.base, pattern.amplitude);
    const float evolution = pattern.amplitude * 2.0e-3f * frame;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        values[index] += evolution * std::sin(0.013f * static_cast<float>(index));
    }

    // Exercise physically meaningful classifications close to zero. A value
    // separated from zero by four codec bounds must retain its sign; exact
    // zero moisture is allowed to move only within the advertised bound.
    const float separated = static_cast<float>(4.0 * tolerance);
    if (std::string(pattern.name) == "qv" || std::string(pattern.name) == "qr")
    {
        values[0] = 0.0f;
        values[1] = separated;
    }
    else if (std::string(pattern.name) == "u" || std::string(pattern.name) == "v" || std::string(pattern.name) == "w")
    {
        values[0] = -separated;
        values[1] = separated;
    }
    return values;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Keyframe roundtrip
// ---------------------------------------------------------------------------

TEST_CASE("ZFP keyframe roundtrip preserves data within tolerance", "[core][zfp][roundtrip][analytical]")
{
    const auto   dir  = make_tmp_dir("keyframe_rt");
    const int    dim0 = 8, dim1 = 12, dim2 = 16;
    const double tolerance = 1.0e-4;

    OutputConfig cfg;
    cfg.format                = OutputFormat::zfp;
    cfg.async_io              = false;
    cfg.zfp_mode              = "accuracy";
    cfg.zfp_tolerance         = tolerance;
    cfg.zfp_keyframe_interval = 0; // all keyframes

    AsyncOutputWriter writer(cfg);

    auto original = make_smooth_field(dim0, dim1, dim2, 300.0f, 50.0f);
    auto step_dir = dir / "step_000";
    auto snap     = make_single_field_snapshot(step_dir, 0, "theta", original, dim0, dim1, dim2);
    REQUIRE(writer.submit(std::move(snap)));

    // Read back
    Zfp3dHeader        header;
    std::vector<float> readback;
    std::string        error;
    auto               zfp_path = (step_dir / "theta.zfp3d").string();
    REQUIRE(read_zfp_3d(zfp_path, header, readback, error));

    // Verify header
    CHECK(header.version == 3);
    CHECK(header.is_bricked());
    CHECK(header.brick_dim0 == 64);
    CHECK(header.brick_dim1 == 64);
    CHECK(header.brick_dim2 == 32);
    CHECK(header.dim0 == dim0);
    CHECK(header.dim1 == dim1);
    CHECK(header.dim2 == dim2);
    CHECK(header.is_keyframe());
    CHECK(header.mode == 0); // accuracy

    // Verify data within tolerance
    REQUIRE(readback.size() == original.size());
    float err = max_abs_error(original, readback);
    CHECK(err <= static_cast<float>(tolerance));

    // Verify compressed size is smaller than raw
    CHECK(header.compressed_size < original.size() * sizeof(float));

    cleanup_dir(dir);
}

// ---------------------------------------------------------------------------
// Simple delta sequence roundtrip
// ---------------------------------------------------------------------------

TEST_CASE("ZFP simple delta sequence roundtrip reconstructs correctly", "[core][zfp][roundtrip][delta][analytical]")
{
    const auto   dir  = make_tmp_dir("simple_delta_rt");
    const int    dim0 = 8, dim1 = 12, dim2 = 16;
    const double tolerance         = 1.0e-4;
    const int    num_frames        = 32;
    const int    keyframe_interval = 32; // long chain with one keyframe at frame 0

    OutputConfig cfg;
    cfg.format                = OutputFormat::zfp;
    cfg.async_io              = false;
    cfg.zfp_mode              = "accuracy";
    cfg.zfp_tolerance         = tolerance;
    cfg.zfp_keyframe_interval = keyframe_interval;
    cfg.zfp_predictive_delta  = false;

    AsyncOutputWriter writer(cfg);

    // Write a sequence of evolving frames
    std::vector<std::vector<float>> originals;
    for (int t = 0; t < num_frames; ++t)
    {
        auto data = make_evolved_field(dim0, dim1, dim2, 300.0f, 50.0f, t);
        originals.push_back(data);

        auto step_dir = dir / ("step_" + std::to_string(t));
        auto snap     = make_single_field_snapshot(step_dir, t, "theta", data, dim0, dim1, dim2);
        REQUIRE(writer.submit(std::move(snap)));
    }

    // Verify frame types via header inspection
    {
        Zfp3dHeader h;
        std::string err;
        // Frame 0 should be keyframe
        REQUIRE(read_zfp_3d_header((dir / "step_0" / "theta.zfp3d").string(), h, err));
        CHECK(h.is_keyframe());

        // Frames 1-4 should be simple deltas
        for (int t = 1; t < num_frames; ++t)
        {
            REQUIRE(read_zfp_3d_header((dir / ("step_" + std::to_string(t)) / "theta.zfp3d").string(), h, err));
            CHECK(h.is_simple_delta());
        }
    }

    // Read back with ZfpFrameReader and verify reconstruction
    ZfpFrameReader reader;
    for (int t = 0; t < num_frames; ++t)
    {
        auto               path = (dir / ("step_" + std::to_string(t)) / "theta.zfp3d").string();
        std::vector<float> reconstructed;
        std::string        error;
        REQUIRE(reader.read_frame("theta", path, reconstructed, error));

        // The writer predicts from decoded references, exactly as the reader
        // does. Prior-frame error therefore cancels from the residual and the
        // current original remains bounded by this frame's ZFP accuracy.
        float err   = max_abs_error(originals[t], reconstructed);
        float bound = static_cast<float>(tolerance) * 1.01f;
        CHECK(err <= bound);
    }

    // Verify delta frames compress better than keyframes (smaller files)
    auto keyframe_size = std::filesystem::file_size(dir / "step_0" / "theta.zfp3d");
    auto delta_size    = std::filesystem::file_size(dir / "step_1" / "theta.zfp3d");
    CHECK(delta_size < keyframe_size);

    cleanup_dir(dir);
}

// ---------------------------------------------------------------------------
// Predictive delta sequence roundtrip
// ---------------------------------------------------------------------------

TEST_CASE("ZFP predictive delta sequence roundtrip reconstructs correctly",
          "[core][zfp][roundtrip][delta][predictive][analytical]")
{
    const auto   dir  = make_tmp_dir("pred_delta_rt");
    const int    dim0 = 8, dim1 = 12, dim2 = 16;
    const double tolerance         = 1.0e-4;
    const int    num_frames        = 32;
    const int    keyframe_interval = 32; // long chain with one keyframe at frame 0

    OutputConfig cfg;
    cfg.format                = OutputFormat::zfp;
    cfg.async_io              = false;
    cfg.zfp_mode              = "accuracy";
    cfg.zfp_tolerance         = tolerance;
    cfg.zfp_keyframe_interval = keyframe_interval;
    cfg.zfp_predictive_delta  = true;

    AsyncOutputWriter writer(cfg);

    // Write a linearly-evolving sequence (predictive delta excels here)
    std::vector<std::vector<float>> originals;
    for (int t = 0; t < num_frames; ++t)
    {
        auto data = make_evolved_field(dim0, dim1, dim2, 300.0f, 50.0f, t);
        originals.push_back(data);

        auto step_dir = dir / ("step_" + std::to_string(t));
        auto snap     = make_single_field_snapshot(step_dir, t, "theta", data, dim0, dim1, dim2);
        REQUIRE(writer.submit(std::move(snap)));
    }

    // Frame 0: keyframe
    // Frame 1: simple delta (only one previous frame available)
    // Frames 2+: predictive delta (two previous frames available)
    {
        Zfp3dHeader h;
        std::string err;
        REQUIRE(read_zfp_3d_header((dir / "step_0" / "theta.zfp3d").string(), h, err));
        CHECK(h.is_keyframe());

        REQUIRE(read_zfp_3d_header((dir / "step_1" / "theta.zfp3d").string(), h, err));
        CHECK(h.is_simple_delta());

        for (int t = 2; t < num_frames; ++t)
        {
            REQUIRE(read_zfp_3d_header((dir / ("step_" + std::to_string(t)) / "theta.zfp3d").string(), h, err));
            CHECK(h.is_predictive_delta());
        }
    }

    // Reconstruct and verify
    ZfpFrameReader reader;
    for (int t = 0; t < num_frames; ++t)
    {
        auto               path = (dir / ("step_" + std::to_string(t)) / "theta.zfp3d").string();
        std::vector<float> reconstructed;
        std::string        error;
        REQUIRE(reader.read_frame("theta", path, reconstructed, error));

        float err   = max_abs_error(originals[t], reconstructed);
        float bound = static_cast<float>(tolerance) * 1.01f;
        CHECK(err <= bound);
    }

    // Predictive delta on linearly evolving data should compress better
    // than simple delta (smaller residuals)
    if (num_frames >= 4)
    {
        auto simple_size = std::filesystem::file_size(dir / "step_1" / "theta.zfp3d");
        auto pred_size   = std::filesystem::file_size(dir / "step_3" / "theta.zfp3d");
        CHECK(pred_size <= simple_size);
    }

    cleanup_dir(dir);
}

TEST_CASE("ZFP predictive history reuses full-field storage after warm-up", "[core][zfp][roundtrip][delta][allocation]")
{
    const auto            dir  = make_tmp_dir("predictive_history_reuse");
    constexpr int         dim0 = 8, dim1 = 8, dim2 = 8;
    constexpr std::size_t raw_field_bytes = static_cast<std::size_t>(dim0) * dim1 * dim2 * sizeof(float);

    OutputConfig cfg;
    cfg.format                = OutputFormat::zfp;
    cfg.async_io              = false;
    cfg.zfp_mode              = "accuracy";
    cfg.zfp_tolerance         = 1.0e-4;
    cfg.zfp_keyframe_interval = 32;
    cfg.zfp_predictive_delta  = true;

    AsyncOutputWriter writer(cfg);
    for (int frame = 0; frame < 3; ++frame)
    {
        auto values = make_evolved_field(dim0, dim1, dim2, 300.0f, 50.0f, frame);
        REQUIRE(writer.submit(make_single_field_snapshot(
            dir / ("step_" + std::to_string(frame)), frame, "theta", values, dim0, dim1, dim2)));
    }

    auto measured_values   = make_evolved_field(dim0, dim1, dim2, 300.0f, 50.0f, 3);
    auto measured_snapshot = make_single_field_snapshot(dir / "step_3", 3, "theta", measured_values, dim0, dim1, dim2);

    tmv::allocation_telemetry::begin_measurement();
    REQUIRE(writer.submit(std::move(measured_snapshot)));
    const auto allocations = tmv::allocation_telemetry::end_measurement();

    // The compressed byte buffer and filesystem machinery may allocate, but
    // predictive reference rotation must not allocate another raw field after
    // scratch/current/previous history capacity is established.
    CHECK(tmv::allocation_telemetry::heap_size_count(tmv::allocation_telemetry::Component::Unattributed,
                                                     raw_field_bytes) == 0);
    CHECK(allocations.physics_field_allocations == 0);

    ZfpFrameReader     reader;
    std::vector<float> reconstructed;
    std::string        error;
    for (int frame = 0; frame <= 3; ++frame)
    {
        REQUIRE(reader.read_frame(
            "theta", (dir / ("step_" + std::to_string(frame)) / "theta.zfp3d").string(), reconstructed, error));
    }
    CHECK(max_abs_error(measured_values, reconstructed) <= static_cast<float>(cfg.zfp_tolerance) * 1.01f);
    cleanup_dir(dir);
}

TEST_CASE("ZFP v3 writer exposes bricks as zero-staging strided views",
          "[core][zfp][bricks][writer][allocation][workspace]")
{
    const auto    dir  = make_tmp_dir("bricked_writer_zero_staging");
    constexpr int dim0 = 64, dim1 = 64, dim2 = 48;
    const auto    values = make_smooth_field(dim0, dim1, dim2, 300.0f, 20.0f);

    OutputConfig cfg;
    cfg.format                = OutputFormat::zfp;
    cfg.async_io              = false;
    cfg.zfp_mode              = "accuracy";
    cfg.zfp_tolerance         = 1.0e-4;
    cfg.zfp_keyframe_interval = 8;
    cfg.zfp_predictive_delta  = true;

    AsyncOutputWriter writer(cfg);
    REQUIRE(writer.submit(make_single_field_snapshot(dir / "step_0", 0, "theta", values, dim0, dim1, dim2)));
    CHECK(writer.zfp_brick_staging_capacity_bytes() == 0);
    CHECK(writer.zfp_codec_workspace_initializations() == 1);
    CHECK(writer.zfp_bitstream_bindings() == 1);
    const std::size_t high_water_capacity = writer.zfp_bitstream_capacity_bytes();
    REQUIRE(high_water_capacity > 0);

    auto evolved = values;
    for (std::size_t index = 0; index < evolved.size(); ++index)
        evolved[index] += 1.0e-3f * std::sin(0.01f * index);
    REQUIRE(writer.submit(make_single_field_snapshot(dir / "step_1", 1, "theta", evolved, dim0, dim1, dim2)));
    CHECK(writer.zfp_codec_workspace_initializations() == 1);
    CHECK(writer.zfp_bitstream_bindings() == 1);
    CHECK(writer.zfp_bitstream_capacity_bytes() == high_water_capacity);

    ZfpFrameReader     reader;
    std::vector<float> reconstructed;
    std::string        error;
    REQUIRE(reader.read_frame("theta", (dir / "step_0" / "theta.zfp3d").string(), reconstructed, error));
    CHECK(max_abs_error(values, reconstructed) <= static_cast<float>(cfg.zfp_tolerance) * 1.01f);
    cleanup_dir(dir);
}

TEST_CASE("ZFP codec workspace grows once and retains its high-water binding", "[core][zfp][bricks][writer][workspace]")
{
    const auto dir = make_tmp_dir("workspace_high_water");

    OutputConfig cfg;
    cfg.format                = OutputFormat::zfp;
    cfg.async_io              = false;
    cfg.zfp_mode              = "accuracy";
    cfg.zfp_tolerance         = 1.0e-4;
    cfg.zfp_keyframe_interval = 0;

    AsyncOutputWriter writer(cfg);
    CHECK(writer.zfp_codec_workspace_initializations() == 0);
    CHECK(writer.zfp_bitstream_bindings() == 0);
    CHECK(writer.zfp_bitstream_capacity_bytes() == 0);
    CHECK(writer.zfp_execution_policy_id() == -1);

    const auto small = make_smooth_field(8, 8, 8, 300.0f, 10.0f);
    REQUIRE(writer.submit(make_single_field_snapshot(dir / "step_0", 0, "theta", small, 8, 8, 8)));
    REQUIRE(writer.zfp_codec_workspace_initializations() == 1);
    REQUIRE(writer.zfp_bitstream_bindings() == 1);
    CHECK(writer.zfp_execution_policy_id() == 0);
    CHECK_FALSE(writer.zfp_codec_version().empty());
    CHECK(writer.zfp_codec_version() != "unknown");
    const std::size_t small_capacity = writer.zfp_bitstream_capacity_bytes();
    REQUIRE(small_capacity > 0);

    const auto large = make_smooth_field(64, 64, 32, 300.0f, 10.0f);
    REQUIRE(writer.submit(make_single_field_snapshot(dir / "step_1", 1, "theta", large, 64, 64, 32)));
    CHECK(writer.zfp_codec_workspace_initializations() == 1);
    REQUIRE(writer.zfp_bitstream_bindings() == 2);
    const std::size_t large_capacity = writer.zfp_bitstream_capacity_bytes();
    REQUIRE(large_capacity > small_capacity);

    const auto smaller = make_smooth_field(12, 10, 9, 300.0f, 10.0f);
    REQUIRE(writer.submit(make_single_field_snapshot(dir / "step_2", 2, "theta", smaller, 12, 10, 9)));
    CHECK(writer.zfp_codec_workspace_initializations() == 1);
    CHECK(writer.zfp_bitstream_bindings() == 2);
    CHECK(writer.zfp_bitstream_capacity_bytes() == large_capacity);

    const std::array<std::vector<float>, 3> originals = {small, large, smaller};
    ZfpFrameReader                          reader;
    std::string                             error;
    for (std::size_t frame = 0; frame < originals.size(); ++frame)
    {
        std::vector<float> decoded;
        REQUIRE(reader.read_frame(
            "theta", (dir / ("step_" + std::to_string(frame)) / "theta.zfp3d").string(), decoded, error));
        CHECK(max_abs_error(originals[frame], decoded) <= static_cast<float>(cfg.zfp_tolerance) * 1.01f);
    }
    cleanup_dir(dir);
}

TEST_CASE("ZFP persistent workspace preserves deterministic bytes in every codec mode",
          "[core][zfp][writer][workspace][modes]")
{
    struct ModeCase
    {
        const char* mode;
        uint16_t    header_mode;
        double      header_parameter;
    };
    constexpr std::array<ModeCase, 3> modes  = {{{"accuracy", 0, 1.0e-4}, {"precision", 1, 24.0}, {"rate", 2, 32.0}}};
    const auto                        values = make_smooth_field(33, 35, 17, 300.0f, 25.0f);

    for (const auto& mode : modes)
    {
        CAPTURE(mode.mode);
        const auto dir = make_tmp_dir("workspace_mode_" + std::string(mode.mode));

        OutputConfig cfg;
        cfg.format                = OutputFormat::zfp;
        cfg.async_io              = false;
        cfg.zfp_mode              = mode.mode;
        cfg.zfp_tolerance         = mode.header_parameter;
        cfg.zfp_precision_bits    = 24;
        cfg.zfp_rate_bps          = 32;
        cfg.zfp_keyframe_interval = 0;

        AsyncOutputWriter writer(cfg);
        REQUIRE(writer.submit(make_single_field_snapshot(dir / "step_0", 0, "theta", values, 33, 35, 17)));
        REQUIRE(writer.submit(make_single_field_snapshot(dir / "step_1", 1, "theta", values, 33, 35, 17)));
        CHECK(writer.zfp_codec_workspace_initializations() == 1);
        CHECK(writer.zfp_bitstream_bindings() == 1);

        const auto first_path  = dir / "step_0" / "theta.zfp3d";
        const auto second_path = dir / "step_1" / "theta.zfp3d";
        CHECK(read_file_bytes(first_path) == read_file_bytes(second_path));

        Zfp3dHeader        first_header;
        Zfp3dHeader        second_header;
        std::vector<float> first_decoded;
        std::vector<float> second_decoded;
        std::string        error;
        REQUIRE(read_zfp_3d(first_path.string(), first_header, first_decoded, error));
        REQUIRE(read_zfp_3d(second_path.string(), second_header, second_decoded, error));
        CHECK(first_header.mode == mode.header_mode);
        CHECK(first_header.mode_param == Approx(mode.header_parameter));
        CHECK(second_header.mode == first_header.mode);
        CHECK(second_header.mode_param == first_header.mode_param);
        CHECK(first_decoded == second_decoded);
        CHECK(
            std::all_of(first_decoded.begin(), first_decoded.end(), [](float value) { return std::isfinite(value); }));
        // Precision/rate are not part of the qualified absolute-error
        // contract, but this fixed smooth physical fixture must remain a
        // useful deterministic smoke rather than merely decode to finites.
        CHECK(max_abs_error(values, first_decoded) <= 1.0e-2f);
        cleanup_dir(dir);
    }
}

TEST_CASE("ZFP async writer owns and drains one persistent codec workspace", "[core][zfp][writer][workspace][async]")
{
    const auto    dir  = make_tmp_dir("workspace_async");
    constexpr int dim0 = 64, dim1 = 32, dim2 = 40;

    OutputConfig cfg;
    cfg.format                = OutputFormat::zfp;
    cfg.async_io              = true;
    cfg.zfp_mode              = "accuracy";
    cfg.zfp_tolerance         = 1.0e-4;
    cfg.zfp_keyframe_interval = 8;
    cfg.zfp_predictive_delta  = true;

    AsyncOutputWriter                 writer(cfg);
    std::array<std::vector<float>, 3> originals;
    for (int frame = 0; frame < 3; ++frame)
    {
        originals[frame] = make_evolved_field(dim0, dim1, dim2, 300.0f, 20.0f, frame);
        REQUIRE(writer.wait_until_ready());
        REQUIRE(writer.submit(make_single_field_snapshot(
            dir / ("step_" + std::to_string(frame)), frame, "theta", originals[frame], dim0, dim1, dim2)));
    }
    REQUIRE(writer.flush());
    CHECK(writer.snapshots_written() == originals.size());
    CHECK(writer.zfp_codec_workspace_initializations() == 1);
    CHECK(writer.zfp_bitstream_bindings() == 1);
    CHECK(writer.zfp_bitstream_capacity_bytes() > 0);

    ZfpFrameReader reader;
    std::string    error;
    for (std::size_t frame = 0; frame < originals.size(); ++frame)
    {
        std::vector<float> decoded;
        REQUIRE(reader.read_frame(
            "theta", (dir / ("step_" + std::to_string(frame)) / "theta.zfp3d").string(), decoded, error));
        CHECK(max_abs_error(originals[frame], decoded) <= static_cast<float>(cfg.zfp_tolerance) * 1.01f);
    }
    cleanup_dir(dir);
}

TEST_CASE("ZFP v3 window decodes only intersecting bricks exactly", "[core][zfp][bricks][window]")
{
    const auto                       dir  = make_tmp_dir("bricked_window");
    constexpr int                    dim0 = 70, dim1 = 67, dim2 = 35;
    const std::array<std::size_t, 3> shape{dim0, dim1, dim2};
    const std::array<std::size_t, 3> origin{61, 60, 29};
    const std::array<std::size_t, 3> extent{9, 7, 5};
    const auto                       values = make_smooth_field(dim0, dim1, dim2, 300.0f, 40.0f);

    OutputConfig cfg;
    cfg.format        = OutputFormat::zfp;
    cfg.async_io      = false;
    cfg.zfp_mode      = "accuracy";
    cfg.zfp_tolerance = 1.0e-4;
    AsyncOutputWriter writer(cfg);
    REQUIRE(writer.submit(make_single_field_snapshot(dir / "step_0", 0, "theta", values, dim0, dim1, dim2)));

    const auto         path = (dir / "step_0" / "theta.zfp3d").string();
    Zfp3dHeader        full_header;
    Zfp3dHeader        window_header;
    std::vector<float> full;
    std::vector<float> window;
    std::string        error;
    REQUIRE(read_zfp_3d(path, full_header, full, error));
    REQUIRE(read_zfp_3d_window(path, origin, extent, window_header, window, error));
    CHECK(window_header.version == 3);
    CHECK(window_header.brick_count0 == 2);
    CHECK(window_header.brick_count1 == 2);
    CHECK(window_header.brick_count2 == 2);
    CHECK(window == extract_window(full, shape, origin, extent));
    CHECK(max_abs_error(extract_window(values, shape, origin, extent), window) <=
          static_cast<float>(cfg.zfp_tolerance) * 1.01f);
    cleanup_dir(dir);
}

TEST_CASE("ZFP v3 predictive windows replay the identical local chain", "[core][zfp][bricks][window][delta]")
{
    const auto                       dir  = make_tmp_dir("bricked_predictive_window");
    constexpr int                    dim0 = 70, dim1 = 67, dim2 = 35;
    const std::array<std::size_t, 3> shape{dim0, dim1, dim2};
    const std::array<std::size_t, 3> origin{61, 58, 29};
    const std::array<std::size_t, 3> extent{8, 9, 6};

    OutputConfig cfg;
    cfg.format                = OutputFormat::zfp;
    cfg.async_io              = false;
    cfg.zfp_mode              = "accuracy";
    cfg.zfp_tolerance         = 1.0e-4;
    cfg.zfp_keyframe_interval = 16;
    cfg.zfp_predictive_delta  = true;
    AsyncOutputWriter               writer(cfg);
    std::vector<std::vector<float>> originals;
    for (int frame = 0; frame < 4; ++frame)
    {
        originals.push_back(make_evolved_field(dim0, dim1, dim2, 300.0f, 40.0f, frame));
        REQUIRE(writer.submit(make_single_field_snapshot(
            dir / ("step_" + std::to_string(frame)), frame, "theta", originals.back(), dim0, dim1, dim2)));
    }

    ZfpFrameReader     full_reader;
    ZfpFrameReader     window_reader;
    std::vector<float> full;
    std::vector<float> window;
    std::string        error;
    for (int frame = 0; frame < 4; ++frame)
    {
        const auto  path = (dir / ("step_" + std::to_string(frame)) / "theta.zfp3d").string();
        Zfp3dHeader full_header;
        Zfp3dHeader window_header;
        REQUIRE(full_reader.read_frame("theta-full", path, full_header, full, error));
        REQUIRE(window_reader.read_frame_window("theta-window", path, origin, extent, window_header, window, error));
        CHECK(window == extract_window(full, shape, origin, extent));
        CHECK(max_abs_error(extract_window(originals[frame], shape, origin, extent), window) <=
              static_cast<float>(cfg.zfp_tolerance) * 1.01f);
    }
    cleanup_dir(dir);
}

TEST_CASE("ZFP v3 small window avoids full-field allocation", "[core][zfp][bricks][window][allocation]")
{
    const auto        dir  = make_tmp_dir("bricked_window_allocation");
    constexpr int     dim0 = 128, dim1 = 128, dim2 = 64;
    const auto        values           = make_smooth_field(dim0, dim1, dim2, 300.0f, 20.0f);
    const std::size_t full_field_bytes = values.size() * sizeof(float);

    OutputConfig cfg;
    cfg.format        = OutputFormat::zfp;
    cfg.async_io      = false;
    cfg.zfp_mode      = "accuracy";
    cfg.zfp_tolerance = 1.0e-4;
    AsyncOutputWriter writer(cfg);
    REQUIRE(writer.submit(make_single_field_snapshot(dir / "step_0", 0, "theta", values, dim0, dim1, dim2)));

    Zfp3dHeader        header;
    std::vector<float> window;
    std::string        error;
    tmv::allocation_telemetry::begin_measurement();
    REQUIRE(
        read_zfp_3d_window((dir / "step_0" / "theta.zfp3d").string(), {61, 61, 29}, {4, 4, 4}, header, window, error));
    const auto allocations = tmv::allocation_telemetry::end_measurement();
    CHECK(window.size() == 64);
    CHECK(tmv::allocation_telemetry::heap_size_count(tmv::allocation_telemetry::Component::Unattributed,
                                                     full_field_bytes) == 0);
    CHECK(allocations.physics_field_allocations == 0);
    cleanup_dir(dir);
}

TEST_CASE("ZFP v3 rejects a corrupt directory without mutating output", "[core][zfp][bricks][corruption]")
{
    const auto    dir  = make_tmp_dir("bricked_directory_corruption");
    constexpr int dim0 = 40, dim1 = 37, dim2 = 21;
    const auto    values = make_smooth_field(dim0, dim1, dim2, 300.0f, 20.0f);

    OutputConfig cfg;
    cfg.format        = OutputFormat::zfp;
    cfg.async_io      = false;
    cfg.zfp_mode      = "accuracy";
    cfg.zfp_tolerance = 1.0e-4;
    AsyncOutputWriter writer(cfg);
    REQUIRE(writer.submit(make_single_field_snapshot(dir / "step_0", 0, "theta", values, dim0, dim1, dim2)));

    const auto  path = dir / "step_0" / "theta.zfp3d";
    Zfp3dHeader header;
    std::string error;
    REQUIRE(read_zfp_3d_header(path.string(), header, error));
    {
        std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
        REQUIRE(file.is_open());
        const std::streamoff first_entry_offset = 44 + static_cast<std::streamoff>(header.compressed_size);
        const uint64_t       invalid_offset     = 1;
        file.seekp(first_entry_offset);
        file.write(reinterpret_cast<const char*>(&invalid_offset), sizeof(invalid_offset));
        REQUIRE(file.good());
    }

    std::vector<float> output{123.0f, 456.0f};
    const auto         sentinel = output;
    CHECK_FALSE(read_zfp_3d_window(path.string(), {0, 0, 0}, {4, 4, 4}, header, output, error));
    CHECK(error.find("brick directory entry") != std::string::npos);
    CHECK(output == sentinel);
    cleanup_dir(dir);
}

// ---------------------------------------------------------------------------
// Keyframe interval resets delta state
// ---------------------------------------------------------------------------

TEST_CASE("ZFP keyframe interval correctly resets delta state", "[core][zfp][roundtrip][delta][analytical]")
{
    const auto   dir  = make_tmp_dir("keyframe_reset");
    const int    dim0 = 8, dim1 = 8, dim2 = 8;
    const double tolerance         = 1.0e-3;
    const int    keyframe_interval = 3;
    const int    num_frames        = 7;

    OutputConfig cfg;
    cfg.format                = OutputFormat::zfp;
    cfg.async_io              = false;
    cfg.zfp_mode              = "accuracy";
    cfg.zfp_tolerance         = tolerance;
    cfg.zfp_keyframe_interval = keyframe_interval;
    cfg.zfp_predictive_delta  = false;

    AsyncOutputWriter writer(cfg);

    std::vector<std::vector<float>> originals;
    for (int t = 0; t < num_frames; ++t)
    {
        auto data = make_evolved_field(dim0, dim1, dim2, 100.0f, 20.0f, t);
        originals.push_back(data);

        auto step_dir = dir / ("step_" + std::to_string(t));
        auto snap     = make_single_field_snapshot(step_dir, t, "theta", data, dim0, dim1, dim2);
        REQUIRE(writer.submit(std::move(snap)));
    }

    // Expected pattern with interval=3:
    // frame 0: keyframe (counter 0 % 3 == 0)
    // frame 1: delta
    // frame 2: delta
    // frame 3: keyframe (counter 3 % 3 == 0)
    // frame 4: delta
    // frame 5: delta
    // frame 6: keyframe (counter 6 % 3 == 0)
    Zfp3dHeader h;
    std::string err;
    for (int t = 0; t < num_frames; ++t)
    {
        auto path = (dir / ("step_" + std::to_string(t)) / "theta.zfp3d").string();
        REQUIRE(read_zfp_3d_header(path, h, err));
        if (t % keyframe_interval == 0)
        {
            CHECK(h.is_keyframe());
        }
        else
        {
            CHECK(h.is_simple_delta());
        }
    }

    // Full reconstruction should work across keyframe boundaries
    ZfpFrameReader reader;
    for (int t = 0; t < num_frames; ++t)
    {
        auto               path = (dir / ("step_" + std::to_string(t)) / "theta.zfp3d").string();
        std::vector<float> reconstructed;
        REQUIRE(reader.read_frame("theta", path, reconstructed, err));

        float error = max_abs_error(originals[t], reconstructed);
        float bound = static_cast<float>(tolerance) * 1.01f;
        CHECK(error <= bound);
    }

    cleanup_dir(dir);
}

// ---------------------------------------------------------------------------
// Per-field tolerance
// ---------------------------------------------------------------------------

TEST_CASE("ZFP global visualization tolerance cannot erase Ertel PV",
          "[core][zfp][roundtrip][potential-vorticity][science]")
{
    const auto dir = make_tmp_dir("pv_science_tolerance");
    constexpr int dim0 = 8;
    constexpr int dim1 = 8;
    constexpr int dim2 = 8;

    OutputConfig cfg;
    cfg.format = OutputFormat::zfp;
    cfg.async_io = false;
    cfg.zfp_mode = "accuracy";
    cfg.zfp_tolerance = 1.0e-3;
    cfg.zfp_keyframe_interval = 0;
    AsyncOutputWriter writer(cfg);

    std::vector<float> original(
        static_cast<std::size_t>(dim0 * dim1 * dim2));
    for (std::size_t n = 0; n < original.size(); ++n)
        original[n] = static_cast<float>(
            -4.0e-6 + 8.0e-6 * static_cast<double>(n)
                / static_cast<double>(original.size() - 1));

    ExportSnapshot snapshot;
    snapshot.export_index = 0;
    snapshot.step_dir = dir / "step_0";
    FieldSnapshotEntry field;
    field.name = "potential_vorticity";
    field.dim0 = dim0;
    field.dim1 = dim1;
    field.dim2 = dim2;
    field.is_3d = true;
    field.data = original;
    snapshot.fields.push_back(std::move(field));
    REQUIRE(writer.submit(std::move(snapshot)));
    REQUIRE(writer.flush());

    ZfpFrameReader reader;
    std::vector<float> decoded;
    std::string error;
    REQUIRE(reader.read_frame(
        "potential_vorticity",
        (dir / "step_0" / "potential_vorticity.zfp3d").string(),
        decoded,
        error));
    REQUIRE(decoded.size() == original.size());
    CHECK(max_abs_error(original, decoded) <= 1.01e-8f);
    CHECK(*std::max_element(decoded.begin(), decoded.end()) > 1.0e-6f);
    CHECK(*std::min_element(decoded.begin(), decoded.end()) < -1.0e-6f);
    cleanup_dir(dir);
}

TEST_CASE("ZFP per-field tolerance produces field-appropriate compression",
          "[core][zfp][roundtrip][tolerance][analytical]")
{
    const auto   dir  = make_tmp_dir("per_field_tol");
    const int    dim0 = 8, dim1 = 8, dim2 = 16;
    const double tight_tol = 1.0e-5;
    const double loose_tol = 1.0e-2;

    OutputConfig cfg;
    cfg.format                                   = OutputFormat::zfp;
    cfg.async_io                                 = false;
    cfg.zfp_mode                                 = "accuracy";
    cfg.zfp_tolerance                            = 1.0e-3; // global fallback
    cfg.zfp_keyframe_interval                    = 0;
    cfg.zfp_per_field_tolerances                 = true;
    cfg.zfp_field_tolerances["u"]                = tight_tol;
    cfg.zfp_field_tolerances["reflectivity_dbz"] = loose_tol;

    AsyncOutputWriter writer(cfg);

    auto u_data   = make_smooth_field(dim0, dim1, dim2, 0.0f, 30.0f);
    auto dbz_data = make_smooth_field(dim0, dim1, dim2, 20.0f, 60.0f);

    ExportSnapshot snap;
    snap.export_index      = 0;
    snap.simulation_time_s = 0.0;
    snap.step_dir          = dir / "step_0";

    FieldSnapshotEntry u_entry;
    u_entry.name  = "u";
    u_entry.dim0  = dim0;
    u_entry.dim1  = dim1;
    u_entry.dim2  = dim2;
    u_entry.is_3d = true;
    u_entry.data  = u_data;
    snap.fields.push_back(std::move(u_entry));

    FieldSnapshotEntry dbz_entry;
    dbz_entry.name  = "reflectivity_dbz";
    dbz_entry.dim0  = dim0;
    dbz_entry.dim1  = dim1;
    dbz_entry.dim2  = dim2;
    dbz_entry.is_3d = true;
    dbz_entry.data  = dbz_data;
    snap.fields.push_back(std::move(dbz_entry));

    REQUIRE(writer.submit(std::move(snap)));

    // Read back both fields
    Zfp3dHeader        h;
    std::vector<float> readback;
    std::string        error;

    REQUIRE(read_zfp_3d((dir / "step_0" / "u.zfp3d").string(), h, readback, error));
    float u_err = max_abs_error(u_data, readback);
    CHECK(u_err <= static_cast<float>(tight_tol));

    REQUIRE(read_zfp_3d((dir / "step_0" / "reflectivity_dbz.zfp3d").string(), h, readback, error));
    float dbz_err = max_abs_error(dbz_data, readback);
    CHECK(dbz_err <= static_cast<float>(loose_tol));

    // Loose tolerance should produce smaller files
    auto u_size   = std::filesystem::file_size(dir / "step_0" / "u.zfp3d");
    auto dbz_size = std::filesystem::file_size(dir / "step_0" / "reflectivity_dbz.zfp3d");
    CHECK(dbz_size < u_size);

    cleanup_dir(dir);
}

TEST_CASE("qualified ZFP core history obeys unit-aware decoded error budgets",
          "[core][zfp][roundtrip][qualified][science]")
{
    const auto  dir  = make_tmp_dir("qualified_core_history");
    const int   dim0 = 12, dim1 = 10, dim2 = 16;
    const int   frame_count = 3;
    const auto& budgets     = qualified_zfp_field_budgets();

    REQUIRE(budgets.size() >= kQualifiedFieldPatterns.size());

    OutputConfig cfg;
    cfg.format                   = OutputFormat::zfp;
    cfg.async_io                 = false;
    cfg.zfp_mode                 = "accuracy";
    cfg.zfp_keyframe_interval    = 4;
    cfg.zfp_predictive_delta     = true;
    cfg.zfp_sparse_threshold     = 0.0f;
    cfg.zfp_float16_prefilter    = false;
    cfg.zfp_per_field_tolerances = true;
    cfg.keep_last_frames = 3;
    cfg.minimum_free_space_mib = 4096;
    cfg.maximum_retained_history_files = 256;
    for (const auto& pattern : kQualifiedFieldPatterns)
    {
        REQUIRE(budgets.count(pattern.name) == 1);
        cfg.resolved_fields.insert(pattern.name);
        cfg.zfp_field_tolerances[pattern.name] = budgets.at(pattern.name).max_configured_absolute_error;
    }
    REQUIRE(validate_qualified_zfp_history(cfg).empty());

    std::array<std::unordered_map<std::string, std::vector<float>>, frame_count> originals;
    AsyncOutputWriter                                                            writer(cfg);
    const std::size_t                                                            raw_frame_bytes =
        static_cast<std::size_t>(dim0) * dim1 * dim2 * sizeof(float) * kQualifiedFieldPatterns.size();

    for (int frame = 0; frame < frame_count; ++frame)
    {
        ExportSnapshot snapshot;
        snapshot.export_index      = frame;
        snapshot.simulation_time_s = frame * 10.0;
        snapshot.step_dir          = dir / ("step_" + std::to_string(frame));

        for (const auto& pattern : kQualifiedFieldPatterns)
        {
            const double tolerance         = budgets.at(pattern.name).max_configured_absolute_error;
            auto         values            = make_qualified_field(pattern, dim0, dim1, dim2, frame, tolerance);
            originals[frame][pattern.name] = values;

            FieldSnapshotEntry entry;
            entry.name  = pattern.name;
            entry.dim0  = dim0;
            entry.dim1  = dim1;
            entry.dim2  = dim2;
            entry.is_3d = true;
            entry.data  = std::move(values);
            snapshot.fields.push_back(std::move(entry));
        }
        REQUIRE(writer.submit(std::move(snapshot)));

        std::size_t compressed_frame_bytes = 0;
        for (const auto& pattern : kQualifiedFieldPatterns)
        {
            compressed_frame_bytes += std::filesystem::file_size(dir / ("step_" + std::to_string(frame)) /
                                                                 (std::string(pattern.name) + ".zfp3d"));
        }
        CHECK(compressed_frame_bytes < raw_frame_bytes);
    }

    ZfpFrameReader reader;
    for (int frame = 0; frame < frame_count; ++frame)
    {
        for (const auto& pattern : kQualifiedFieldPatterns)
        {
            const auto   path      = dir / ("step_" + std::to_string(frame)) / (std::string(pattern.name) + ".zfp3d");
            const auto&  original  = originals[frame].at(pattern.name);
            const double tolerance = budgets.at(pattern.name).max_configured_absolute_error;
            const float  bound     = static_cast<float>(tolerance) * 1.01f;

            Zfp3dHeader        header;
            std::vector<float> decoded;
            std::string        error;
            REQUIRE(reader.read_frame(pattern.name, path.string(), header, decoded, error));
            REQUIRE(decoded.size() == original.size());
            CHECK(header.mode == 0);
            CHECK(header.mode_param == Approx(tolerance));
            CHECK(max_abs_error(original, decoded) <= bound);

            double inventory_error = 0.0;
            for (std::size_t index = 0; index < decoded.size(); ++index)
            {
                inventory_error += static_cast<double>(decoded[index]) - static_cast<double>(original[index]);
            }
            inventory_error = std::abs(inventory_error / decoded.size());
            CHECK(inventory_error <= bound);

            for (std::size_t index = 1; index < decoded.size(); ++index)
            {
                const float decoded_increment  = decoded[index] - decoded[index - 1];
                const float original_increment = original[index] - original[index - 1];
                CHECK(std::abs(decoded_increment - original_increment) <= 2.0f * bound);
            }

            const std::string field(pattern.name);
            if (field == "qv" || field == "qr")
            {
                CHECK(std::abs(decoded[0]) <= bound);
                CHECK(decoded[1] > 0.0f);
            }
            else if (field == "u" || field == "v" || field == "w")
            {
                CHECK(decoded[0] < 0.0f);
                CHECK(decoded[1] > 0.0f);
            }
        }
    }

    cleanup_dir(dir);
}

TEST_CASE("field-parallel ZFP matches serial bytes and decoded error deterministically",
          "[core][zfp][parallel][qualified][determinism][memory]")
{
    constexpr int dim0 = 18, dim1 = 16, dim2 = 20;
    constexpr int frame_count = 3;
    const auto& budgets = qualified_zfp_field_budgets();
    const auto serial_dir = make_tmp_dir("field_parallel_serial");
    const auto one_dir = make_tmp_dir("field_parallel_one");
    const auto four_a_dir = make_tmp_dir("field_parallel_four_a");
    const auto four_b_dir = make_tmp_dir("field_parallel_four_b");

    const auto config = [&](const std::string& policy, int workers)
    {
        OutputConfig cfg;
        cfg.format = OutputFormat::zfp;
        cfg.async_io = false;
        cfg.zfp_mode = "accuracy";
        cfg.zfp_keyframe_interval = 4;
        cfg.zfp_predictive_delta = true;
        cfg.zfp_per_field_tolerances = true;
        cfg.zfp_field_execution = policy;
        cfg.zfp_field_workers = workers;
        cfg.zfp_resolved_field_workers = std::max(1, workers);
        cfg.zfp_field_scratch_mib = 64;
        for (const auto& pattern : kQualifiedFieldPatterns)
        {
            cfg.resolved_fields.insert(pattern.name);
            cfg.zfp_field_tolerances[pattern.name] =
                budgets.at(pattern.name).max_configured_absolute_error;
        }
        return cfg;
    };
    const auto snapshot = [&](const std::filesystem::path& root, int frame)
    {
        ExportSnapshot value;
        value.export_index = frame;
        value.simulation_time_s = frame;
        value.step_dir = root / ("step_" + std::to_string(frame));
        for (const auto& pattern : kQualifiedFieldPatterns)
        {
            FieldSnapshotEntry entry;
            entry.name = pattern.name;
            entry.dim0 = dim0;
            entry.dim1 = dim1;
            entry.dim2 = dim2;
            entry.is_3d = true;
            entry.data = make_qualified_field(
                pattern, dim0, dim1, dim2, frame,
                budgets.at(pattern.name).max_configured_absolute_error);
            value.fields.push_back(std::move(entry));
        }
        return value;
    };

    AsyncOutputWriter serial(config("serial", 1));
    AsyncOutputWriter one(config("field_parallel", 1));
    AsyncOutputWriter four_a(config("field_parallel", 4));
    AsyncOutputWriter four_b(config("field_parallel", 4));
    for (int frame = 0; frame < frame_count; ++frame)
    {
        REQUIRE(serial.submit(snapshot(serial_dir, frame)));
        REQUIRE(one.submit(snapshot(one_dir, frame)));
        REQUIRE(four_a.submit(snapshot(four_a_dir, frame)));
        REQUIRE(four_b.submit(snapshot(four_b_dir, frame)));
    }

    ZfpFrameReader reader;
    for (int frame = 0; frame < frame_count; ++frame)
        for (const auto& pattern : kQualifiedFieldPatterns)
        {
            const std::string filename = std::string(pattern.name) + ".zfp3d";
            const auto serial_path = serial_dir / ("step_" + std::to_string(frame)) / filename;
            const auto one_path = one_dir / ("step_" + std::to_string(frame)) / filename;
            const auto four_a_path = four_a_dir / ("step_" + std::to_string(frame)) / filename;
            const auto four_b_path = four_b_dir / ("step_" + std::to_string(frame)) / filename;
            const auto expected = read_file_bytes(serial_path);
            CHECK(read_file_bytes(one_path) == expected);
            CHECK(read_file_bytes(four_a_path) == expected);
            CHECK(read_file_bytes(four_b_path) == expected);

            std::vector<float> decoded;
            std::string error;
            REQUIRE(reader.read_frame(pattern.name, four_a_path.string(), decoded, error));
            const auto original = make_qualified_field(
                pattern, dim0, dim1, dim2, frame,
                budgets.at(pattern.name).max_configured_absolute_error);
            CHECK(max_abs_error(original, decoded) <=
                  static_cast<float>(budgets.at(pattern.name).max_configured_absolute_error) * 1.01f);
        }

    CHECK(one.zfp_resolved_field_workers() == 1);
    CHECK(four_a.zfp_resolved_field_workers() == 4);
    CHECK(four_a.zfp_execution_policy_id() == 0);
    CHECK(four_a.zfp_parallel_tasks_completed() ==
          frame_count * kQualifiedFieldPatterns.size());
    CHECK(four_a.zfp_parallel_scratch_high_water_bytes() > 0);
    CHECK(four_a.zfp_parallel_scratch_high_water_bytes() <= 64ULL * 1024ULL * 1024ULL);
    cleanup_dir(serial_dir);
    cleanup_dir(one_dir);
    cleanup_dir(four_a_dir);
    cleanup_dir(four_b_dir);
}

TEST_CASE("field-parallel ZFP rolls back an ordered commit failure",
          "[core][zfp][parallel][failure-atomicity]")
{
    const auto dir = make_tmp_dir("field_parallel_rollback");
    const auto step = dir / "step_0";
    std::filesystem::create_directories(step);
    const std::vector<unsigned char> sentinel = {1, 2, 3, 4};
    {
        std::ofstream existing(step / "v.zfp3d", std::ios::binary);
        existing.write(reinterpret_cast<const char*>(sentinel.data()),
                       static_cast<std::streamsize>(sentinel.size()));
    }
    OutputConfig cfg;
    cfg.format = OutputFormat::zfp;
    cfg.async_io = false;
    cfg.zfp_field_execution = "field_parallel";
    cfg.zfp_field_workers = 2;
    cfg.zfp_resolved_field_workers = 2;
    cfg.zfp_field_scratch_mib = 16;
    AsyncOutputWriter writer(cfg);
    ExportSnapshot snapshot;
    snapshot.step_dir = step;
    for (const std::string name : {"u", "v"})
    {
        FieldSnapshotEntry entry;
        entry.name = name;
        entry.dim0 = 8;
        entry.dim1 = 8;
        entry.dim2 = 8;
        entry.is_3d = true;
        entry.data = make_smooth_field(8, 8, 8, 0.0f, 1.0f);
        snapshot.fields.push_back(std::move(entry));
    }
    REQUIRE_FALSE(writer.submit(std::move(snapshot)));
    CHECK(writer.has_error());
    CHECK_FALSE(std::filesystem::exists(step / "u.zfp3d"));
    CHECK(read_file_bytes(step / "v.zfp3d") == sentinel);
    CHECK_FALSE(std::filesystem::exists(step / "u.zfp3d.parallel.tmp"));
    CHECK_FALSE(std::filesystem::exists(step / "v.zfp3d.parallel.tmp"));
    CHECK_FALSE(std::filesystem::exists(step / "manifest.json"));
    cleanup_dir(dir);
}

// ---------------------------------------------------------------------------
// Header-only read
// ---------------------------------------------------------------------------

TEST_CASE("ZFP precision headers reject fractional and out-of-range parameters",
          "[core][zfp][reader][invalid_precision]")
{
    const auto dir = make_tmp_dir("invalid_precision");
    OutputConfig config;
    config.format = OutputFormat::zfp;
    config.async_io = false;
    AsyncOutputWriter writer(config);
    const auto path = dir / "step_0" / "p.zfp3d";
    REQUIRE(writer.submit(make_single_field_snapshot(dir / "step_0", 0, "p",
        make_smooth_field(4, 4, 4, 1.0f, 1.0f), 4, 4, 4)));
    for (double precision : {1.5, 33.0, 1.0e300, 0.0, -1.0,
            std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        CAPTURE(precision);
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        const std::uint16_t mode = 1;
        file.seekp(6);
        file.write(reinterpret_cast<const char*>(&mode), sizeof(mode));
        file.seekp(20);
        file.write(reinterpret_cast<const char*>(&precision), sizeof(precision));
        file.close();
        Zfp3dHeader header;
        std::string error;
        REQUIRE_FALSE(read_zfp_3d_header(path.string(), header, error));
        std::vector<float> output{17.0f};
        REQUIRE_FALSE(read_zfp_3d(path.string(), header, output, error));
        REQUIRE(output == std::vector<float>{17.0f});
    }
    cleanup_dir(dir);
}

TEST_CASE("ZFP header-only read returns correct metadata", "[core][zfp][reader][analytical]")
{
    const auto   dir  = make_tmp_dir("header_only");
    const int    dim0 = 4, dim1 = 6, dim2 = 10;
    const double tolerance = 5.0e-3;

    OutputConfig cfg;
    cfg.format                = OutputFormat::zfp;
    cfg.async_io              = false;
    cfg.zfp_mode              = "accuracy";
    cfg.zfp_tolerance         = tolerance;
    cfg.zfp_keyframe_interval = 0;

    AsyncOutputWriter writer(cfg);
    auto              data     = make_smooth_field(dim0, dim1, dim2, 1.0f, 1.0f);
    auto              step_dir = dir / "step_0";
    auto              snap     = make_single_field_snapshot(step_dir, 0, "p", data, dim0, dim1, dim2);
    REQUIRE(writer.submit(std::move(snap)));

    Zfp3dHeader h;
    std::string error;
    REQUIRE(read_zfp_3d_header((step_dir / "p.zfp3d").string(), h, error));

    CHECK(h.version == 3);
    CHECK(h.is_bricked());
    CHECK(h.mode == 0);
    CHECK(h.dim0 == dim0);
    CHECK(h.dim1 == dim1);
    CHECK(h.dim2 == dim2);
    CHECK(h.mode_param == Approx(tolerance));
    CHECK(h.compressed_size > 0);
    CHECK(h.is_keyframe());
    CHECK(h.total_elements() == static_cast<std::size_t>(dim0 * dim1 * dim2));

    cleanup_dir(dir);
}

// ---------------------------------------------------------------------------
// Error: delta frame without keyframe
// ---------------------------------------------------------------------------

TEST_CASE("ZfpFrameReader rejects delta frame without preceding keyframe", "[core][zfp][reader][error]")
{
    const auto dir  = make_tmp_dir("no_keyframe");
    const int  dim0 = 4, dim1 = 4, dim2 = 4;

    // Write a keyframe + delta sequence
    OutputConfig cfg;
    cfg.format                = OutputFormat::zfp;
    cfg.async_io              = false;
    cfg.zfp_tolerance         = 1.0e-3;
    cfg.zfp_keyframe_interval = 5;
    cfg.zfp_predictive_delta  = false;

    AsyncOutputWriter writer(cfg);
    for (int t = 0; t < 2; ++t)
    {
        auto data     = make_evolved_field(dim0, dim1, dim2, 1.0f, 1.0f, t);
        auto step_dir = dir / ("step_" + std::to_string(t));
        auto snap     = make_single_field_snapshot(step_dir, t, "f", data, dim0, dim1, dim2);
        REQUIRE(writer.submit(std::move(snap)));
    }

    // Try to read frame 1 (delta) without first reading frame 0 (keyframe)
    ZfpFrameReader     reader;
    std::vector<float> data;
    std::string        error;
    auto               delta_path = (dir / "step_1" / "f.zfp3d").string();
    CHECK_FALSE(reader.read_frame("f", delta_path, data, error));
    CHECK(error.find("no preceding keyframe") != std::string::npos);

    cleanup_dir(dir);
}

// ---------------------------------------------------------------------------
// Multi-field interleaved reconstruction
// ---------------------------------------------------------------------------

TEST_CASE("ZfpFrameReader handles multiple fields independently", "[core][zfp][roundtrip][multi_field][analytical]")
{
    const auto   dir  = make_tmp_dir("multi_field");
    const int    dim0 = 6, dim1 = 8, dim2 = 10;
    const double tolerance  = 1.0e-3;
    const int    num_frames = 4;

    OutputConfig cfg;
    cfg.format                = OutputFormat::zfp;
    cfg.async_io              = false;
    cfg.zfp_tolerance         = tolerance;
    cfg.zfp_keyframe_interval = 4;
    cfg.zfp_predictive_delta  = false;

    AsyncOutputWriter writer(cfg);

    // Write 2 fields per frame with different evolution rates
    std::vector<std::vector<float>> u_originals, theta_originals;
    for (int t = 0; t < num_frames; ++t)
    {
        auto u_data     = make_evolved_field(dim0, dim1, dim2, 0.0f, 20.0f, t);
        auto theta_data = make_evolved_field(dim0, dim1, dim2, 300.0f, 50.0f, t * 2);
        u_originals.push_back(u_data);
        theta_originals.push_back(theta_data);

        ExportSnapshot snap;
        snap.export_index      = t;
        snap.simulation_time_s = t * 5.0;
        snap.step_dir          = dir / ("step_" + std::to_string(t));

        FieldSnapshotEntry u_entry;
        u_entry.name  = "u";
        u_entry.dim0  = dim0;
        u_entry.dim1  = dim1;
        u_entry.dim2  = dim2;
        u_entry.is_3d = true;
        u_entry.data  = u_data;
        snap.fields.push_back(std::move(u_entry));

        FieldSnapshotEntry th_entry;
        th_entry.name  = "theta";
        th_entry.dim0  = dim0;
        th_entry.dim1  = dim1;
        th_entry.dim2  = dim2;
        th_entry.is_3d = true;
        th_entry.data  = theta_data;
        snap.fields.push_back(std::move(th_entry));

        REQUIRE(writer.submit(std::move(snap)));
    }

    // Reconstruct both fields and verify independence
    ZfpFrameReader reader;
    for (int t = 0; t < num_frames; ++t)
    {
        auto               step_dir = dir / ("step_" + std::to_string(t));
        std::vector<float> u_recon, theta_recon;
        std::string        error;

        REQUIRE(reader.read_frame("u", (step_dir / "u.zfp3d").string(), u_recon, error));
        REQUIRE(reader.read_frame("theta", (step_dir / "theta.zfp3d").string(), theta_recon, error));

        float u_err  = max_abs_error(u_originals[t], u_recon);
        float th_err = max_abs_error(theta_originals[t], theta_recon);

        float bound = static_cast<float>(tolerance) * 1.01f;
        CHECK(u_err <= bound);
        CHECK(th_err <= bound);
    }

    // Verify field histories are independent
    CHECK(reader.has_history("u"));
    CHECK(reader.has_history("theta"));
    reader.reset_field("u");
    CHECK_FALSE(reader.has_history("u"));
    CHECK(reader.has_history("theta"));

    cleanup_dir(dir);
}

// ---------------------------------------------------------------------------
// ZfpFrameReader reset
// ---------------------------------------------------------------------------

TEST_CASE("ZfpFrameReader reset clears all field history", "[core][zfp][reader][analytical]")
{
    const auto dir  = make_tmp_dir("reset");
    const int  dim0 = 4, dim1 = 4, dim2 = 4;

    OutputConfig cfg;
    cfg.format                = OutputFormat::zfp;
    cfg.async_io              = false;
    cfg.zfp_tolerance         = 1.0e-3;
    cfg.zfp_keyframe_interval = 0;

    AsyncOutputWriter writer(cfg);
    auto              data     = make_smooth_field(dim0, dim1, dim2, 1.0f, 1.0f);
    auto              step_dir = dir / "step_0";
    auto              snap     = make_single_field_snapshot(step_dir, 0, "f", data, dim0, dim1, dim2);
    REQUIRE(writer.submit(std::move(snap)));

    ZfpFrameReader     reader;
    std::vector<float> readback;
    std::string        error;
    REQUIRE(reader.read_frame("f", (step_dir / "f.zfp3d").string(), readback, error));
    CHECK(reader.has_history("f"));

    reader.reset();
    CHECK_FALSE(reader.has_history("f"));

    cleanup_dir(dir);
}

// ---------------------------------------------------------------------------
// Compression ratio: delta < keyframe for slowly evolving data
// ---------------------------------------------------------------------------

TEST_CASE("ZFP delta frames are smaller than keyframes for slow evolution", "[core][zfp][compression][analytical]")
{
    const auto   dir  = make_tmp_dir("compression_ratio");
    const int    dim0 = 16, dim1 = 16, dim2 = 32;
    const double tolerance = 1.0e-3;

    OutputConfig cfg;
    cfg.format                = OutputFormat::zfp;
    cfg.async_io              = false;
    cfg.zfp_tolerance         = tolerance;
    cfg.zfp_keyframe_interval = 10;
    cfg.zfp_predictive_delta  = false;

    AsyncOutputWriter writer(cfg);

    for (int t = 0; t < 3; ++t)
    {
        auto data     = make_evolved_field(dim0, dim1, dim2, 300.0f, 50.0f, t);
        auto step_dir = dir / ("step_" + std::to_string(t));
        auto snap     = make_single_field_snapshot(step_dir, t, "theta", data, dim0, dim1, dim2);
        REQUIRE(writer.submit(std::move(snap)));
    }

    auto keyframe_bytes = std::filesystem::file_size(dir / "step_0" / "theta.zfp3d");
    auto delta_bytes    = std::filesystem::file_size(dir / "step_1" / "theta.zfp3d");
    auto raw_bytes      = static_cast<std::size_t>(dim0) * dim1 * dim2 * sizeof(float);

    // Delta should be smaller than keyframe
    CHECK(delta_bytes < keyframe_bytes);

    // Both should be smaller than raw
    CHECK(keyframe_bytes < raw_bytes);

    // Report compression ratios for information
    float keyframe_ratio = static_cast<float>(raw_bytes) / keyframe_bytes;
    float delta_ratio    = static_cast<float>(raw_bytes) / delta_bytes;
    CAPTURE(keyframe_ratio, delta_ratio);

    // Minimum expectations: ZFP should achieve at least 2x compression
    CHECK(keyframe_ratio >= 2.0f);
    CHECK(delta_ratio >= 2.0f);

    cleanup_dir(dir);
}

TEST_CASE("Metal ZFP checkpoint continuation preserves decoded history and science ledgers",
          "[core][zfp][metal_zfp_restart][.real_device]")
{
    const char* continuous_env = std::getenv("NIMBUS_TEST_METAL_ZFP_CONTINUOUS");
    const char* resumed_env = std::getenv("NIMBUS_TEST_METAL_ZFP_RESUMED");
    REQUIRE(continuous_env != nullptr);
    REQUIRE(resumed_env != nullptr);
    const std::filesystem::path continuous(continuous_env);
    const std::filesystem::path resumed(resumed_env);
    const std::array<std::string, 8> fields = {
        "u", "v", "w", "theta", "p", "rho", "qv", "qr"};
    const std::unordered_map<std::string, double> tolerance = {
        {"u", 1.0e-4}, {"v", 1.0e-4}, {"w", 1.0e-4},
        {"theta", 1.0e-3}, {"p", 5.0e-2}, {"rho", 1.0e-6},
        {"qv", 1.0e-8}, {"qr", 1.0e-8}};
    const std::array<std::string, 3> continuous_steps = {
        "step_000004", "step_000005", "step_000006"};
    const std::array<std::string, 3> resumed_steps = {
        "step_000000", "step_000001", "step_000002"};

    ZfpFrameReader continuous_reader;
    ZfpFrameReader resumed_reader;
    for (std::size_t step = 0; step < continuous_steps.size(); ++step)
    {
        for (const std::string& field : fields)
        {
            const auto continuous_path = continuous / continuous_steps[step] /
                (field + ".zfp3d");
            const auto resumed_path = resumed / resumed_steps[step] /
                (field + ".zfp3d");
            Zfp3dHeader continuous_header;
            Zfp3dHeader resumed_header;
            std::vector<float> continuous_values;
            std::vector<float> resumed_values;
            std::string error;
            REQUIRE(continuous_reader.read_frame(field, continuous_path.string(),
                                                 continuous_header, continuous_values, error));
            REQUIRE(resumed_reader.read_frame(field, resumed_path.string(),
                                              resumed_header, resumed_values, error));
            REQUIRE(continuous_header.is_bricked());
            REQUIRE(resumed_header.is_bricked());
            REQUIRE(continuous_header.total_elements() == 18U * 18U * 18U);
            REQUIRE(resumed_header.total_elements() == continuous_header.total_elements());
            REQUIRE(continuous_values.size() == resumed_values.size());
            CHECK(continuous_header.is_keyframe() == (step == 0));
            CHECK(resumed_header.is_keyframe() == (step == 0));
            CHECK(continuous_header.is_simple_delta() == (step == 1));
            CHECK(resumed_header.is_simple_delta() == (step == 1));
            CHECK(continuous_header.is_predictive_delta() == (step == 2));
            CHECK(resumed_header.is_predictive_delta() == (step == 2));
            double maximum = 0.0;
            double scale = 0.0;
            for (std::size_t cell = 0; cell < continuous_values.size(); ++cell)
            {
                REQUIRE(std::isfinite(continuous_values[cell]));
                REQUIRE(std::isfinite(resumed_values[cell]));
                maximum = std::max(maximum, static_cast<double>(
                    std::abs(continuous_values[cell] - resumed_values[cell])));
                scale = std::max(scale, static_cast<double>(
                    std::max(std::abs(continuous_values[cell]),
                             std::abs(resumed_values[cell]))));
            }
            const double ulp = scale == 0.0 ?
                static_cast<double>(std::numeric_limits<float>::denorm_min()) :
                static_cast<double>(std::nextafter(static_cast<float>(scale),
                    std::numeric_limits<float>::infinity()) - static_cast<float>(scale));
            CAPTURE(field, step, maximum, ulp);
            CHECK(maximum <= 2.02 * tolerance.at(field) + 2.0 * ulp);
        }
    }

    for (const char* name : {"budget.csv", "energy_budget.csv", "moisture_budget.csv",
                              "momentum_budget.csv", "radiation_budget.csv",
                              "thermodynamic_source_budget.csv"})
    {
        std::ifstream continuous_file(continuous / name);
        std::ifstream resumed_file(resumed / name);
        REQUIRE(continuous_file.good());
        REQUIRE(resumed_file.good());
        std::vector<std::string> continuous_lines;
        std::vector<std::string> resumed_lines;
        std::string line;
        while (std::getline(continuous_file, line))
        {
            continuous_lines.push_back(line);
        }
        while (std::getline(resumed_file, line))
        {
            resumed_lines.push_back(line);
        }
        REQUIRE(resumed_lines.size() > 1);
        REQUIRE(continuous_lines.size() >= resumed_lines.size());
        CHECK(continuous_lines.front() == resumed_lines.front());
        CHECK(std::equal(resumed_lines.begin() + 1, resumed_lines.end(),
                         continuous_lines.end() - (resumed_lines.size() - 1)));
    }
}

TEST_CASE("Metal and CPU ZFP candidate histories preserve bounded decoded-field parity",
          "[core][zfp][metal_zfp_backend_parity][.real_device]")
{
    const char* cpu_env = std::getenv("NIMBUS_TEST_METAL_ZFP_CPU_RUN");
    const char* metal_env = std::getenv("NIMBUS_TEST_METAL_ZFP_METAL_RUN");
    REQUIRE(cpu_env != nullptr);
    REQUIRE(metal_env != nullptr);
    const std::filesystem::path cpu(cpu_env);
    const std::filesystem::path metal(metal_env);
    const std::unordered_map<std::string, double> tolerance = {
        {"u", 1.0e-4}, {"v", 1.0e-4}, {"w", 1.0e-4},
        {"theta", 1.0e-3}, {"p", 5.0e-2}, {"rho", 1.0e-6},
        {"qv", 1.0e-8}, {"qr", 1.0e-8}};
    const std::array<std::string, 2> steps = {"step_000000", "step_000001"};
    ZfpFrameReader cpu_reader;
    ZfpFrameReader metal_reader;
    for (std::size_t step = 0; step < steps.size(); ++step)
    {
        for (const auto& [field, codec_bound] : tolerance)
        {
            const auto cpu_path = cpu / steps[step] / (field + ".zfp3d");
            const auto metal_path = metal / steps[step] / (field + ".zfp3d");
            Zfp3dHeader cpu_header;
            Zfp3dHeader metal_header;
            std::vector<float> cpu_values;
            std::vector<float> metal_values;
            std::string error;
            REQUIRE(cpu_reader.read_frame(field, cpu_path.string(), cpu_header,
                                          cpu_values, error));
            REQUIRE(metal_reader.read_frame(field, metal_path.string(), metal_header,
                                            metal_values, error));
            REQUIRE(cpu_values.size() == metal_values.size());
            REQUIRE(cpu_values.size() == 18U * 18U * 18U);
            CHECK(cpu_header.is_keyframe() == (step == 0));
            CHECK(metal_header.is_keyframe() == (step == 0));
            CHECK(cpu_header.is_simple_delta() == (step == 1));
            CHECK(metal_header.is_simple_delta() == (step == 1));
            double maximum = 0.0;
            double scale = 0.0;
            for (std::size_t cell = 0; cell < cpu_values.size(); ++cell)
            {
                REQUIRE(std::isfinite(cpu_values[cell]));
                REQUIRE(std::isfinite(metal_values[cell]));
                maximum = std::max(maximum, static_cast<double>(
                    std::abs(cpu_values[cell] - metal_values[cell])));
                scale = std::max(scale, static_cast<double>(
                    std::max(std::abs(cpu_values[cell]), std::abs(metal_values[cell]))));
            }
            const double ulp = scale == 0.0 ?
                static_cast<double>(std::numeric_limits<float>::denorm_min()) :
                static_cast<double>(std::nextafter(static_cast<float>(scale),
                    std::numeric_limits<float>::infinity()) - static_cast<float>(scale));
            const double native_limit = std::max(320.0 * ulp,
                field == "u" || field == "v" || field == "w" ? 5.0e-4 : 0.0);
            const double bound = 2.02 * codec_bound + native_limit;
            CAPTURE(field, step, maximum, bound);
            CHECK(maximum <= bound);
            std::cout << "ZFP_PARITY field=" << field << " step=" << step
                      << " error=" << maximum << " limit=" << bound << '\n';
        }
    }
}
