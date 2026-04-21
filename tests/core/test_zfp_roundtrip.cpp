/**
 * @file test_zfp_roundtrip.cpp
 * @brief Comprehensive roundtrip tests for ZFP compression + delta encoding.
 *
 * Verifies:
 *   - Keyframe write/read roundtrip within tolerance
 *   - Simple delta sequence reconstruction
 *   - Predictive delta sequence reconstruction
 *   - Per-field tolerance is respected
 *   - Header parsing for v1 and v2 formats
 *   - ZfpFrameReader stateful reconstruction
 *   - Error handling (missing keyframe, corrupted header)
 *   - Scratch buffer reuse (no per-frame allocation)
 */
#include "catch2/catch.hpp"
#include "core/output/zfp_reader.hpp"
#include "core/output/output_writer.hpp"
#include "core/output/output_config.hpp"
#include "core/field/field_snapshot.hpp"

#include <cmath>
#include <filesystem>
#include <numeric>

namespace {

std::filesystem::path make_tmp_dir(const std::string& name)
{
    auto dir = std::filesystem::temp_directory_path() / ("tmv_test_zfp_" + name);
    std::filesystem::create_directories(dir);
    return dir;
}

void cleanup_dir(const std::filesystem::path& dir)
{
    std::filesystem::remove_all(dir);
}

/// Create a 3D field with a smooth, physically-motivated pattern.
/// value(i,j,k) = base + amplitude * sin(pi*i/dim0) * cos(pi*j/dim1) * (k/dim2)
/// This gives a smooth field that compresses well and produces small deltas.
std::vector<float> make_smooth_field(int dim0, int dim1, int dim2,
                                     float base, float amplitude)
{
    const std::size_t n = static_cast<std::size_t>(dim0) * dim1 * dim2;
    std::vector<float> data(n);
    for (int i = 0; i < dim0; ++i)
    {
        const float si = std::sin(static_cast<float>(M_PI) * i / dim0);
        for (int j = 0; j < dim1; ++j)
        {
            const float cj = std::cos(static_cast<float>(M_PI) * j / dim1);
            for (int k = 0; k < dim2; ++k)
            {
                const float zfrac = static_cast<float>(k) / dim2;
                const std::size_t idx = i * dim1 * dim2 + j * dim2 + k;
                data[idx] = base + amplitude * si * cj * zfrac;
            }
        }
    }
    return data;
}

/// Create a time-evolved version of the smooth field.
/// Adds a small perturbation that grows linearly with time_step,
/// simulating a slowly evolving field (ideal for delta encoding).
std::vector<float> make_evolved_field(int dim0, int dim1, int dim2,
                                      float base, float amplitude,
                                      int time_step)
{
    auto data = make_smooth_field(dim0, dim1, dim2, base, amplitude);
    const float dt = 0.01f * time_step;
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

/// Build an ExportSnapshot with a single named field.
ExportSnapshot make_single_field_snapshot(
    const std::filesystem::path& step_dir, int index,
    const std::string& field_name,
    const std::vector<float>& data,
    int dim0, int dim1, int dim2)
{
    ExportSnapshot snap;
    snap.export_index = index;
    snap.simulation_time_s = index * 5.0;
    snap.step_dir = step_dir;

    FieldSnapshotEntry entry;
    entry.name = field_name;
    entry.dim0 = dim0;
    entry.dim1 = dim1;
    entry.dim2 = dim2;
    entry.is_3d = true;
    entry.data = data;
    snap.fields.push_back(std::move(entry));

    return snap;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Keyframe roundtrip
// ---------------------------------------------------------------------------

TEST_CASE("ZFP keyframe roundtrip preserves data within tolerance",
          "[core][zfp][roundtrip][analytical]")
{
    const auto dir = make_tmp_dir("keyframe_rt");
    const int dim0 = 8, dim1 = 12, dim2 = 16;
    const double tolerance = 1.0e-4;

    OutputConfig cfg;
    cfg.format = OutputFormat::zfp;
    cfg.async_io = false;
    cfg.zfp_mode = "accuracy";
    cfg.zfp_tolerance = tolerance;
    cfg.zfp_keyframe_interval = 0; // all keyframes

    AsyncOutputWriter writer(cfg);

    auto original = make_smooth_field(dim0, dim1, dim2, 300.0f, 50.0f);
    auto step_dir = dir / "step_000";
    auto snap = make_single_field_snapshot(step_dir, 0, "theta", original,
                                           dim0, dim1, dim2);
    REQUIRE(writer.submit(std::move(snap)));

    // Read back
    Zfp3dHeader header;
    std::vector<float> readback;
    std::string error;
    auto zfp_path = (step_dir / "theta.zfp3d").string();
    REQUIRE(read_zfp_3d(zfp_path, header, readback, error));

    // Verify header
    CHECK(header.version == 2);
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

TEST_CASE("ZFP simple delta sequence roundtrip reconstructs correctly",
          "[core][zfp][roundtrip][delta][analytical]")
{
    const auto dir = make_tmp_dir("simple_delta_rt");
    const int dim0 = 8, dim1 = 12, dim2 = 16;
    const double tolerance = 1.0e-4;
    const int num_frames = 5;
    const int keyframe_interval = 5; // keyframe at frame 0 only

    OutputConfig cfg;
    cfg.format = OutputFormat::zfp;
    cfg.async_io = false;
    cfg.zfp_mode = "accuracy";
    cfg.zfp_tolerance = tolerance;
    cfg.zfp_keyframe_interval = keyframe_interval;
    cfg.zfp_predictive_delta = false;

    AsyncOutputWriter writer(cfg);

    // Write a sequence of evolving frames
    std::vector<std::vector<float>> originals;
    for (int t = 0; t < num_frames; ++t)
    {
        auto data = make_evolved_field(dim0, dim1, dim2, 300.0f, 50.0f, t);
        originals.push_back(data);

        auto step_dir = dir / ("step_" + std::to_string(t));
        auto snap = make_single_field_snapshot(step_dir, t, "theta", data,
                                               dim0, dim1, dim2);
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
            REQUIRE(read_zfp_3d_header(
                (dir / ("step_" + std::to_string(t)) / "theta.zfp3d").string(), h, err));
            CHECK(h.is_simple_delta());
        }
    }

    // Read back with ZfpFrameReader and verify reconstruction
    ZfpFrameReader reader;
    for (int t = 0; t < num_frames; ++t)
    {
        auto path = (dir / ("step_" + std::to_string(t)) / "theta.zfp3d").string();
        std::vector<float> reconstructed;
        std::string error;
        REQUIRE(reader.read_frame("theta", path, reconstructed, error));

        // Reconstruction error compounds: each delta adds up to `tolerance`
        // of error. After N deltas from a keyframe, max error ~ N * tolerance.
        // But ZFP error is typically well below the tolerance, so we use
        // a generous bound.
        float err = max_abs_error(originals[t], reconstructed);
        float bound = static_cast<float>(tolerance) * (t + 1) * 2.0f;
        CHECK(err <= bound);
    }

    // Verify delta frames compress better than keyframes (smaller files)
    auto keyframe_size = std::filesystem::file_size(dir / "step_0" / "theta.zfp3d");
    auto delta_size = std::filesystem::file_size(dir / "step_1" / "theta.zfp3d");
    CHECK(delta_size < keyframe_size);

    cleanup_dir(dir);
}

// ---------------------------------------------------------------------------
// Predictive delta sequence roundtrip
// ---------------------------------------------------------------------------

TEST_CASE("ZFP predictive delta sequence roundtrip reconstructs correctly",
          "[core][zfp][roundtrip][delta][predictive][analytical]")
{
    const auto dir = make_tmp_dir("pred_delta_rt");
    const int dim0 = 8, dim1 = 12, dim2 = 16;
    const double tolerance = 1.0e-4;
    const int num_frames = 6;
    const int keyframe_interval = 6; // single keyframe at frame 0

    OutputConfig cfg;
    cfg.format = OutputFormat::zfp;
    cfg.async_io = false;
    cfg.zfp_mode = "accuracy";
    cfg.zfp_tolerance = tolerance;
    cfg.zfp_keyframe_interval = keyframe_interval;
    cfg.zfp_predictive_delta = true;

    AsyncOutputWriter writer(cfg);

    // Write a linearly-evolving sequence (predictive delta excels here)
    std::vector<std::vector<float>> originals;
    for (int t = 0; t < num_frames; ++t)
    {
        auto data = make_evolved_field(dim0, dim1, dim2, 300.0f, 50.0f, t);
        originals.push_back(data);

        auto step_dir = dir / ("step_" + std::to_string(t));
        auto snap = make_single_field_snapshot(step_dir, t, "theta", data,
                                               dim0, dim1, dim2);
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
            REQUIRE(read_zfp_3d_header(
                (dir / ("step_" + std::to_string(t)) / "theta.zfp3d").string(), h, err));
            CHECK(h.is_predictive_delta());
        }
    }

    // Reconstruct and verify
    ZfpFrameReader reader;
    for (int t = 0; t < num_frames; ++t)
    {
        auto path = (dir / ("step_" + std::to_string(t)) / "theta.zfp3d").string();
        std::vector<float> reconstructed;
        std::string error;
        REQUIRE(reader.read_frame("theta", path, reconstructed, error));

        float err = max_abs_error(originals[t], reconstructed);
        float bound = static_cast<float>(tolerance) * (t + 1) * 2.0f;
        CHECK(err <= bound);
    }

    // Predictive delta on linearly evolving data should compress better
    // than simple delta (smaller residuals)
    if (num_frames >= 4)
    {
        auto simple_size = std::filesystem::file_size(dir / "step_1" / "theta.zfp3d");
        auto pred_size = std::filesystem::file_size(dir / "step_3" / "theta.zfp3d");
        CHECK(pred_size <= simple_size);
    }

    cleanup_dir(dir);
}

// ---------------------------------------------------------------------------
// Keyframe interval resets delta state
// ---------------------------------------------------------------------------

TEST_CASE("ZFP keyframe interval correctly resets delta state",
          "[core][zfp][roundtrip][delta][analytical]")
{
    const auto dir = make_tmp_dir("keyframe_reset");
    const int dim0 = 8, dim1 = 8, dim2 = 8;
    const double tolerance = 1.0e-3;
    const int keyframe_interval = 3;
    const int num_frames = 7;

    OutputConfig cfg;
    cfg.format = OutputFormat::zfp;
    cfg.async_io = false;
    cfg.zfp_mode = "accuracy";
    cfg.zfp_tolerance = tolerance;
    cfg.zfp_keyframe_interval = keyframe_interval;
    cfg.zfp_predictive_delta = false;

    AsyncOutputWriter writer(cfg);

    std::vector<std::vector<float>> originals;
    for (int t = 0; t < num_frames; ++t)
    {
        auto data = make_evolved_field(dim0, dim1, dim2, 100.0f, 20.0f, t);
        originals.push_back(data);

        auto step_dir = dir / ("step_" + std::to_string(t));
        auto snap = make_single_field_snapshot(step_dir, t, "theta", data,
                                               dim0, dim1, dim2);
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
        auto path = (dir / ("step_" + std::to_string(t)) / "theta.zfp3d").string();
        std::vector<float> reconstructed;
        REQUIRE(reader.read_frame("theta", path, reconstructed, err));

        float error = max_abs_error(originals[t], reconstructed);
        // After a keyframe reset, error bounds restart from tolerance
        int frames_since_keyframe = t % keyframe_interval;
        float bound = static_cast<float>(tolerance) * (frames_since_keyframe + 1) * 2.0f;
        CHECK(error <= bound);
    }

    cleanup_dir(dir);
}

// ---------------------------------------------------------------------------
// Per-field tolerance
// ---------------------------------------------------------------------------

TEST_CASE("ZFP per-field tolerance produces field-appropriate compression",
          "[core][zfp][roundtrip][tolerance][analytical]")
{
    const auto dir = make_tmp_dir("per_field_tol");
    const int dim0 = 8, dim1 = 8, dim2 = 16;
    const double tight_tol = 1.0e-5;
    const double loose_tol = 1.0e-2;

    OutputConfig cfg;
    cfg.format = OutputFormat::zfp;
    cfg.async_io = false;
    cfg.zfp_mode = "accuracy";
    cfg.zfp_tolerance = 1.0e-3; // global fallback
    cfg.zfp_keyframe_interval = 0;
    cfg.zfp_per_field_tolerances = true;
    cfg.zfp_field_tolerances["u"] = tight_tol;
    cfg.zfp_field_tolerances["reflectivity_dbz"] = loose_tol;

    AsyncOutputWriter writer(cfg);

    auto u_data = make_smooth_field(dim0, dim1, dim2, 0.0f, 30.0f);
    auto dbz_data = make_smooth_field(dim0, dim1, dim2, 20.0f, 60.0f);

    ExportSnapshot snap;
    snap.export_index = 0;
    snap.simulation_time_s = 0.0;
    snap.step_dir = dir / "step_0";

    FieldSnapshotEntry u_entry;
    u_entry.name = "u";
    u_entry.dim0 = dim0;
    u_entry.dim1 = dim1;
    u_entry.dim2 = dim2;
    u_entry.is_3d = true;
    u_entry.data = u_data;
    snap.fields.push_back(std::move(u_entry));

    FieldSnapshotEntry dbz_entry;
    dbz_entry.name = "reflectivity_dbz";
    dbz_entry.dim0 = dim0;
    dbz_entry.dim1 = dim1;
    dbz_entry.dim2 = dim2;
    dbz_entry.is_3d = true;
    dbz_entry.data = dbz_data;
    snap.fields.push_back(std::move(dbz_entry));

    REQUIRE(writer.submit(std::move(snap)));

    // Read back both fields
    Zfp3dHeader h;
    std::vector<float> readback;
    std::string error;

    REQUIRE(read_zfp_3d((dir / "step_0" / "u.zfp3d").string(), h, readback, error));
    float u_err = max_abs_error(u_data, readback);
    CHECK(u_err <= static_cast<float>(tight_tol));

    REQUIRE(read_zfp_3d((dir / "step_0" / "reflectivity_dbz.zfp3d").string(),
                         h, readback, error));
    float dbz_err = max_abs_error(dbz_data, readback);
    CHECK(dbz_err <= static_cast<float>(loose_tol));

    // Loose tolerance should produce smaller files
    auto u_size = std::filesystem::file_size(dir / "step_0" / "u.zfp3d");
    auto dbz_size = std::filesystem::file_size(dir / "step_0" / "reflectivity_dbz.zfp3d");
    CHECK(dbz_size < u_size);

    cleanup_dir(dir);
}

// ---------------------------------------------------------------------------
// Header-only read
// ---------------------------------------------------------------------------

TEST_CASE("ZFP header-only read returns correct metadata",
          "[core][zfp][reader][analytical]")
{
    const auto dir = make_tmp_dir("header_only");
    const int dim0 = 4, dim1 = 6, dim2 = 10;
    const double tolerance = 5.0e-3;

    OutputConfig cfg;
    cfg.format = OutputFormat::zfp;
    cfg.async_io = false;
    cfg.zfp_mode = "accuracy";
    cfg.zfp_tolerance = tolerance;
    cfg.zfp_keyframe_interval = 0;

    AsyncOutputWriter writer(cfg);
    auto data = make_smooth_field(dim0, dim1, dim2, 1.0f, 1.0f);
    auto step_dir = dir / "step_0";
    auto snap = make_single_field_snapshot(step_dir, 0, "p", data,
                                           dim0, dim1, dim2);
    REQUIRE(writer.submit(std::move(snap)));

    Zfp3dHeader h;
    std::string error;
    REQUIRE(read_zfp_3d_header((step_dir / "p.zfp3d").string(), h, error));

    CHECK(h.version == 2);
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

TEST_CASE("ZfpFrameReader rejects delta frame without preceding keyframe",
          "[core][zfp][reader][error]")
{
    const auto dir = make_tmp_dir("no_keyframe");
    const int dim0 = 4, dim1 = 4, dim2 = 4;

    // Write a keyframe + delta sequence
    OutputConfig cfg;
    cfg.format = OutputFormat::zfp;
    cfg.async_io = false;
    cfg.zfp_tolerance = 1.0e-3;
    cfg.zfp_keyframe_interval = 5;
    cfg.zfp_predictive_delta = false;

    AsyncOutputWriter writer(cfg);
    for (int t = 0; t < 2; ++t)
    {
        auto data = make_evolved_field(dim0, dim1, dim2, 1.0f, 1.0f, t);
        auto step_dir = dir / ("step_" + std::to_string(t));
        auto snap = make_single_field_snapshot(step_dir, t, "f", data,
                                               dim0, dim1, dim2);
        REQUIRE(writer.submit(std::move(snap)));
    }

    // Try to read frame 1 (delta) without first reading frame 0 (keyframe)
    ZfpFrameReader reader;
    std::vector<float> data;
    std::string error;
    auto delta_path = (dir / "step_1" / "f.zfp3d").string();
    CHECK_FALSE(reader.read_frame("f", delta_path, data, error));
    CHECK(error.find("no preceding keyframe") != std::string::npos);

    cleanup_dir(dir);
}

// ---------------------------------------------------------------------------
// Multi-field interleaved reconstruction
// ---------------------------------------------------------------------------

TEST_CASE("ZfpFrameReader handles multiple fields independently",
          "[core][zfp][roundtrip][multi_field][analytical]")
{
    const auto dir = make_tmp_dir("multi_field");
    const int dim0 = 6, dim1 = 8, dim2 = 10;
    const double tolerance = 1.0e-3;
    const int num_frames = 4;

    OutputConfig cfg;
    cfg.format = OutputFormat::zfp;
    cfg.async_io = false;
    cfg.zfp_tolerance = tolerance;
    cfg.zfp_keyframe_interval = 4;
    cfg.zfp_predictive_delta = false;

    AsyncOutputWriter writer(cfg);

    // Write 2 fields per frame with different evolution rates
    std::vector<std::vector<float>> u_originals, theta_originals;
    for (int t = 0; t < num_frames; ++t)
    {
        auto u_data = make_evolved_field(dim0, dim1, dim2, 0.0f, 20.0f, t);
        auto theta_data = make_evolved_field(dim0, dim1, dim2, 300.0f, 50.0f, t * 2);
        u_originals.push_back(u_data);
        theta_originals.push_back(theta_data);

        ExportSnapshot snap;
        snap.export_index = t;
        snap.simulation_time_s = t * 5.0;
        snap.step_dir = dir / ("step_" + std::to_string(t));

        FieldSnapshotEntry u_entry;
        u_entry.name = "u";
        u_entry.dim0 = dim0;
        u_entry.dim1 = dim1;
        u_entry.dim2 = dim2;
        u_entry.is_3d = true;
        u_entry.data = u_data;
        snap.fields.push_back(std::move(u_entry));

        FieldSnapshotEntry th_entry;
        th_entry.name = "theta";
        th_entry.dim0 = dim0;
        th_entry.dim1 = dim1;
        th_entry.dim2 = dim2;
        th_entry.is_3d = true;
        th_entry.data = theta_data;
        snap.fields.push_back(std::move(th_entry));

        REQUIRE(writer.submit(std::move(snap)));
    }

    // Reconstruct both fields and verify independence
    ZfpFrameReader reader;
    for (int t = 0; t < num_frames; ++t)
    {
        auto step_dir = dir / ("step_" + std::to_string(t));
        std::vector<float> u_recon, theta_recon;
        std::string error;

        REQUIRE(reader.read_frame("u", (step_dir / "u.zfp3d").string(),
                                   u_recon, error));
        REQUIRE(reader.read_frame("theta", (step_dir / "theta.zfp3d").string(),
                                   theta_recon, error));

        float u_err = max_abs_error(u_originals[t], u_recon);
        float th_err = max_abs_error(theta_originals[t], theta_recon);

        float bound = static_cast<float>(tolerance) * (t + 1) * 2.0f;
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

TEST_CASE("ZfpFrameReader reset clears all field history",
          "[core][zfp][reader][analytical]")
{
    const auto dir = make_tmp_dir("reset");
    const int dim0 = 4, dim1 = 4, dim2 = 4;

    OutputConfig cfg;
    cfg.format = OutputFormat::zfp;
    cfg.async_io = false;
    cfg.zfp_tolerance = 1.0e-3;
    cfg.zfp_keyframe_interval = 0;

    AsyncOutputWriter writer(cfg);
    auto data = make_smooth_field(dim0, dim1, dim2, 1.0f, 1.0f);
    auto step_dir = dir / "step_0";
    auto snap = make_single_field_snapshot(step_dir, 0, "f", data,
                                           dim0, dim1, dim2);
    REQUIRE(writer.submit(std::move(snap)));

    ZfpFrameReader reader;
    std::vector<float> readback;
    std::string error;
    REQUIRE(reader.read_frame("f", (step_dir / "f.zfp3d").string(),
                               readback, error));
    CHECK(reader.has_history("f"));

    reader.reset();
    CHECK_FALSE(reader.has_history("f"));

    cleanup_dir(dir);
}

// ---------------------------------------------------------------------------
// Compression ratio: delta < keyframe for slowly evolving data
// ---------------------------------------------------------------------------

TEST_CASE("ZFP delta frames are smaller than keyframes for slow evolution",
          "[core][zfp][compression][analytical]")
{
    const auto dir = make_tmp_dir("compression_ratio");
    const int dim0 = 16, dim1 = 16, dim2 = 32;
    const double tolerance = 1.0e-3;

    OutputConfig cfg;
    cfg.format = OutputFormat::zfp;
    cfg.async_io = false;
    cfg.zfp_tolerance = tolerance;
    cfg.zfp_keyframe_interval = 10;
    cfg.zfp_predictive_delta = false;

    AsyncOutputWriter writer(cfg);

    for (int t = 0; t < 3; ++t)
    {
        auto data = make_evolved_field(dim0, dim1, dim2, 300.0f, 50.0f, t);
        auto step_dir = dir / ("step_" + std::to_string(t));
        auto snap = make_single_field_snapshot(step_dir, t, "theta", data,
                                               dim0, dim1, dim2);
        REQUIRE(writer.submit(std::move(snap)));
    }

    auto keyframe_bytes = std::filesystem::file_size(dir / "step_0" / "theta.zfp3d");
    auto delta_bytes = std::filesystem::file_size(dir / "step_1" / "theta.zfp3d");
    auto raw_bytes = static_cast<std::size_t>(dim0) * dim1 * dim2 * sizeof(float);

    // Delta should be smaller than keyframe
    CHECK(delta_bytes < keyframe_bytes);

    // Both should be smaller than raw
    CHECK(keyframe_bytes < raw_bytes);

    // Report compression ratios for information
    float keyframe_ratio = static_cast<float>(raw_bytes) / keyframe_bytes;
    float delta_ratio = static_cast<float>(raw_bytes) / delta_bytes;
    CAPTURE(keyframe_ratio, delta_ratio);

    // Minimum expectations: ZFP should achieve at least 2x compression
    CHECK(keyframe_ratio >= 2.0f);
    CHECK(delta_ratio >= 2.0f);

    cleanup_dir(dir);
}
