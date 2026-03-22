/**
 * @file export_dataset.cpp
 * @brief Disk scanning and frame loading for exported tornado scalar fields.
 *
 * Validates per-step slice completeness, enforces shape consistency across time,
 * and converts raw float fields into robustly normalized volume payloads. Includes
 * defensive parsing to avoid crashes on malformed filenames and NPY metadata.
 */

#include "data/export_dataset.hpp"

#include "data/npy_reader.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <unordered_map>

namespace oglcpp 
{
namespace 
{

/** @brief Escape regex metacharacters in a literal token. */
std::string regex_escape(const std::string& input) 
{
    std::string out;
    out.reserve(input.size() * 2);
    const std::string meta = R"(\.^$|()[]*+?{})";

    for (char c : input) 
    {
        if (meta.find(c) != std::string::npos) 
        {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    return out;
}

/** @brief Return `true` for finite floating-point values. */
bool is_finite(float x) 
{
    return std::isfinite(static_cast<double>(x));
}

/** @brief Parse a non-negative decimal integer with range validation. */
bool parse_nonnegative_int(const std::string& text, int& out, std::string& error)
{
    errno = 0;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0' || errno == ERANGE ||
        parsed > static_cast<unsigned long>(std::numeric_limits<int>::max()))
    {
        error = "invalid non-negative integer: '" + text + "'";
        return false;
    }

    out = static_cast<int>(parsed);
    return true;
}

/** @brief Compute `nx*ny*nz` with overflow checks. */
bool checked_volume_size(int nx, int ny, int nz, std::size_t& out, std::string& error)
{
    if (nx <= 0 || ny <= 0 || nz <= 0)
    {
        error = "non-positive dataset dimensions";
        return false;
    }

    const std::size_t sx = static_cast<std::size_t>(nx);
    const std::size_t sy = static_cast<std::size_t>(ny);
    const std::size_t sz = static_cast<std::size_t>(nz);

    if (sx > std::numeric_limits<std::size_t>::max() / sy)
    {
        error = "dataset dimensions overflow size_t (nx*ny)";
        return false;
    }
    const std::size_t xy = sx * sy;
    if (xy > std::numeric_limits<std::size_t>::max() / sz)
    {
        error = "dataset dimensions overflow size_t (nx*ny*nz)";
        return false;
    }

    out = xy * sz;
    return true;
}

struct StepShape
{
    int nx = 0;
    int ny = 0;
    int nz = 0;
    bool is_3d_volume = false; ///< True if the step uses 3D volume files
};

/** @brief Validate theta slices and extract 3D dimensions for one step directory. */
bool inspect_step_shape(const std::filesystem::path& step_dir,
                        const std::string& field_name,
                        StepShape& out,
                        std::string& error)
{
    // Check for 3D volume file first: <field_name>.npy with 3D shape
    const auto volume_path = step_dir / (field_name + ".npy");
    if (std::filesystem::exists(volume_path))
    {
        NpyArray3D vol;
        std::string load_error;
        if (load_npy_float32_3d(volume_path, vol, load_error))
        {
            // 3D volume: shape is (NR, NTH, NZ) in row-major Field3D layout
            out.nx = vol.dim0;  // NR (radial)
            out.ny = vol.dim1;  // NTH (azimuthal)
            out.nz = vol.dim2;  // NZ (vertical)
            out.is_3d_volume = true;
            if (out.nx <= 0 || out.ny <= 0 || out.nz <= 0)
            {
                error = "non-positive shape in 3D volume " + volume_path.string();
                return false;
            }
            return true;
        }
        // If 3D load failed, fall through to 2D slice detection
    }

    // Fall back to per-theta 2D slice detection
    std::regex slice_re("^th([0-9]+)_" + regex_escape(field_name) + "\\.npy$");
    std::unordered_map<int, std::filesystem::path> slices_by_theta;
    int max_theta = -1;

    std::error_code iter_error;
    std::filesystem::directory_iterator it(
        step_dir,
        std::filesystem::directory_options::skip_permission_denied,
        iter_error);
    if (iter_error)
    {
        error = "failed to iterate '" + step_dir.string() + "': " + iter_error.message();
        return false;
    }

    const std::filesystem::directory_iterator end;
    for (; it != end; it.increment(iter_error))
    {
        if (iter_error)
        {
            error = "failed while iterating '" + step_dir.string() + "': " + iter_error.message();
            return false;
        }

        const std::filesystem::directory_entry& entry = *it;
        std::error_code type_error;
        if (!entry.is_regular_file(type_error))
        {
            continue;
        }

        const std::string name = entry.path().filename().string();
        std::smatch match;
        if (!std::regex_match(name, match, slice_re) || match.size() < 2)
        {
            continue;
        }

        int theta_idx = -1;
        std::string parse_error;
        if (!parse_nonnegative_int(match[1].str(), theta_idx, parse_error))
        {
            error = "invalid theta slice index in '" + name + "': " + parse_error;
            return false;
        }
        slices_by_theta[theta_idx] = entry.path();
        max_theta = std::max(max_theta, theta_idx);
    }

    if (max_theta < 0)
    {
        error = "no theta slices found for field '" + field_name + "'";
        return false;
    }

    out.ny = max_theta + 1;
    for (int th = 0; th < out.ny; ++th)
    {
        if (slices_by_theta.find(th) == slices_by_theta.end())
        {
            std::ostringstream oss;
            oss << "missing theta slice th" << th << "_" << field_name << ".npy";
            error = oss.str();
            return false;
        }
    }

    NpyArray2DShape first_shape;
    std::string load_error;
    if (!load_npy_float32_2d_shape(slices_by_theta[0], first_shape, load_error))
    {
        error = "failed to read th0 shape: " + load_error;
        return false;
    }

    out.nz = first_shape.rows;
    out.nx = first_shape.cols;
    if (out.nx <= 0 || out.ny <= 0 || out.nz <= 0)
    {
        error = "non-positive shape detected";
        return false;
    }

    for (int th = 1; th < out.ny; ++th)
    {
        NpyArray2DShape shape;
        if (!load_npy_float32_2d_shape(slices_by_theta[th], shape, load_error))
        {
            std::ostringstream oss;
            oss << "failed to read th" << th << " shape: " << load_error;
            error = oss.str();
            return false;
        }

        if (shape.rows != out.nz || shape.cols != out.nx)
        {
            std::ostringstream oss;
            oss << "slice shape mismatch at th" << th
                << ": expected (" << out.nz << ", " << out.nx
                << "), got (" << shape.rows << ", " << shape.cols << ")";
            error = oss.str();
            return false;
        }
    }

    return true;
}

} // namespace

/** @brief Store dataset root and field token for later scanning/loading. */
ExportDataset::ExportDataset(std::filesystem::path root_dir, std::string field_name)
    : root_dir_(std::move(root_dir)), field_name_(std::move(field_name)) {}

/** @brief Clamp scalar values into the unit interval. */
float ExportDataset::clamp01(float v) 
{
    if (v < 0.0f) {return 0.0f;}

    if (v > 1.0f) {return 1.0f;}
    return v;
}

/** @brief Normalize source data with percentile clipping and non-finite sanitization. */
void ExportDataset::normalize_volume(const std::vector<float>& src,
                                     std::vector<float>& dst,
                                     float& raw_min,
                                     float& raw_max,
                                     float& norm_low,
                                     float& norm_high,
                                     std::size_t& nan_count,
                                     std::size_t& inf_count,
                                     std::size_t& sanitized_nonfinite_count) 
{
    raw_min = std::numeric_limits<float>::infinity();
    raw_max = -std::numeric_limits<float>::infinity();
    nan_count = 0;
    inf_count = 0;
    sanitized_nonfinite_count = 0;
    std::size_t finite_count = 0;

    for (float v : src) 
    {
        if (!is_finite(v)) 
        {
            if (std::isnan(static_cast<double>(v))){++nan_count;} else {++inf_count;}
            continue;
        }
        raw_min = std::min(raw_min, v);
        raw_max = std::max(raw_max, v);
        ++finite_count;
    }

    dst.resize(src.size(), 0.0f);

    if (finite_count == 0 || !(raw_max > raw_min)) 
    {
        if (!std::isfinite(raw_min)) 
        {
            raw_min = 0.0f;
            raw_max = 0.0f;
        }
        norm_low = raw_min;
        norm_high = raw_max;
        sanitized_nonfinite_count = nan_count + inf_count;
        return;
    }

    constexpr int bins = 2048;
    std::vector<std::size_t> hist(bins, 0);
    const float span = raw_max - raw_min;

    for (float v : src) 
    {
        if (!is_finite(v)) 
        {
            continue;
        }
        const float t = (v - raw_min) / span;
        int idx = static_cast<int>(t * static_cast<float>(bins - 1));
        idx = std::max(0, std::min(bins - 1, idx));
        ++hist[static_cast<std::size_t>(idx)];
    }

    const std::size_t low_target = static_cast<std::size_t>(0.02 * static_cast<double>(finite_count));
    const std::size_t high_target = static_cast<std::size_t>(0.98 * static_cast<double>(finite_count));

    std::size_t cumulative = 0;
    int low_bin = 0;
    int high_bin = bins - 1;

    for (int i = 0; i < bins; ++i) 
    {
        cumulative += hist[static_cast<std::size_t>(i)];
        if (cumulative >= low_target) 
        {
            low_bin = i;
            break;
        }
    }

    cumulative = 0;
    for (int i = 0; i < bins; ++i) 
    {
        cumulative += hist[static_cast<std::size_t>(i)];
        if (cumulative >= high_target) 
        {
            high_bin = i;
            break;
        }
    }

    norm_low = raw_min + (static_cast<float>(low_bin) / static_cast<float>(bins - 1)) * span;
    norm_high = raw_min + (static_cast<float>(high_bin) / static_cast<float>(bins - 1)) * span;

    if (!(norm_high > norm_low)) 
    {
        norm_low = raw_min;
        norm_high = raw_max;
    }

    const float norm_span = std::max(norm_high - norm_low, 1e-12f);
    for (std::size_t i = 0; i < src.size(); ++i) 
    {
        const float v = src[i];
        if (!is_finite(v)) 
        {
            dst[i] = 0.0f;
            ++sanitized_nonfinite_count;
            continue;
        }
        dst[i] = clamp01((v - norm_low) / norm_span);
    }
}

/** @brief Discover step directories and validate shape consistency across frames. */
bool ExportDataset::scan(std::string& error) 
{
    step_dirs_.clear();
    nx_ = ny_ = nz_ = 0;

    std::error_code fs_error;
    const bool exists = std::filesystem::exists(root_dir_, fs_error);
    if (fs_error)
    {
        error = "Failed to access input path '" + root_dir_.string() + "': " + fs_error.message();
        return false;
    }

    if (!exists) 
    {
        error = "Input directory does not exist: " + root_dir_.string();
        return false;
    }
    fs_error.clear();
    const bool is_dir = std::filesystem::is_directory(root_dir_, fs_error);
    if (fs_error)
    {
        error = "Failed to inspect input path '" + root_dir_.string() + "': " + fs_error.message();
        return false;
    }
    if (!is_dir)
    {
        error = "Input path is not a directory: " + root_dir_.string();
        return false;
    }

    std::vector<std::pair<int, std::filesystem::path>> indexed_steps;
    std::regex step_re("^step_([0-9]+)$");

    std::error_code iter_error;
    std::filesystem::directory_iterator it(
        root_dir_,
        std::filesystem::directory_options::skip_permission_denied,
        iter_error);
    if (iter_error)
    {
        error = "Failed to iterate input directory '" + root_dir_.string() + "': " + iter_error.message();
        return false;
    }

    const std::filesystem::directory_iterator end;
    for (; it != end; it.increment(iter_error)) 
    {
        if (iter_error)
        {
            error = "Directory iteration failed in '" + root_dir_.string() + "': " + iter_error.message();
            return false;
        }

        const std::filesystem::directory_entry& entry = *it;
        std::error_code type_error;
        if (!entry.is_directory(type_error)) 
        {
            continue;
        }

        const std::string name = entry.path().filename().string();
        std::smatch match;
        if (!std::regex_match(name, match, step_re) || match.size() < 2) 
        {
            continue;
        }

        int step_idx = -1;
        std::string parse_error;
        if (!parse_nonnegative_int(match[1].str(), step_idx, parse_error))
        {
            std::cerr << "[vulkan][dataset] Skipping " << entry.path().string()
                      << " (invalid step index: " << parse_error << ")\n";
            continue;
        }

        indexed_steps.emplace_back(step_idx, entry.path());
    }

    if (indexed_steps.empty()) 
    {
        error = "No step_XXXXXX directories found in " + root_dir_.string();
        return false;
    }

    std::sort(indexed_steps.begin(), indexed_steps.end(), [](const auto& a, const auto& b) 
    {
        return a.first < b.first;
    });

    bool have_reference_shape = false;
    StepShape reference_shape{};

    for (const auto& step : indexed_steps)
    {
        StepShape shape{};
        std::string shape_error;
        if (!inspect_step_shape(step.second, field_name_, shape, shape_error))
        {
            std::cerr << "[vulkan][dataset] Skipping " << step.second.string()
                      << " (" << shape_error << ")\n";
            continue;
        }

        if (!have_reference_shape)
        {
            reference_shape = shape;
            have_reference_shape = true;
            step_dirs_.push_back(step.second);
            continue;
        }

        if (shape.nx != reference_shape.nx ||
            shape.ny != reference_shape.ny ||
            shape.nz != reference_shape.nz)
        {
            std::cerr << "[vulkan][dataset] Skipping " << step.second.string()
                      << " (shape mismatch: expected "
                      << reference_shape.nx << "x" << reference_shape.ny << "x" << reference_shape.nz
                      << ", got "
                      << shape.nx << "x" << shape.ny << "x" << shape.nz << ")\n";
            continue;
        }

        step_dirs_.push_back(step.second);
    }

    if (step_dirs_.empty())
    {
        error = "No valid step directories with consistent shape found in " + root_dir_.string();
        return false;
    }

    nx_ = reference_shape.nx;
    ny_ = reference_shape.ny;
    nz_ = reference_shape.nz;
    is_3d_volume_ = reference_shape.is_3d_volume;
    return true;
}

/** @brief Load, validate, and normalize one time-step volume frame. */
bool ExportDataset::load_frame(std::size_t frame_idx, VolumeFrame& out, std::string& error) const 
{
    if (frame_idx >= step_dirs_.size()) 
    {
        std::ostringstream oss;
        oss << "Frame index " << frame_idx << " out of range [0, " << (step_dirs_.empty() ? 0 : step_dirs_.size() - 1)
            << "]";
        error = oss.str();
        return false;
    }

    if (nx_ <= 0 || ny_ <= 0 || nz_ <= 0) 
    {
        error = "Dataset must be scanned before load_frame()";
        return false;
    }

    std::size_t raw_size = 0;
    if (!checked_volume_size(nx_, ny_, nz_, raw_size, error))
    {
        return false;
    }
    std::vector<float> raw(raw_size, 0.0f);

    std::size_t expected_slice_size = 0;
    if (!checked_volume_size(nx_, 1, nz_, expected_slice_size, error))
    {
        return false;
    }

    const auto& step_dir = step_dirs_[frame_idx];

    if (is_3d_volume_)
    {
        // --- 3D volume path: load single file and transpose ---
        const auto volume_path = step_dir / (field_name_ + ".npy");
        NpyArray3D vol;
        std::string load_error;
        if (!load_npy_float32_3d(volume_path, vol, load_error))
        {
            error = "Failed loading 3D volume " + volume_path.string() + ": " + load_error;
            return false;
        }
        if (vol.dim0 != nx_ || vol.dim1 != ny_ || vol.dim2 != nz_)
        {
            std::ostringstream oss;
            oss << "3D volume shape mismatch in " << volume_path.string()
                << ": expected (" << nx_ << ", " << ny_ << ", " << nz_
                << "), got (" << vol.dim0 << ", " << vol.dim1 << ", " << vol.dim2 << ")";
            error = oss.str();
            return false;
        }

        // Transpose from Field3D layout [NR][NTH][NZ] = [i*NTH*NZ + j*NZ + k]
        // to viewer layout [Z][TH][X] = [z*NY*NX + th*NX + x]
        for (int z = 0; z < nz_; ++z)
        {
            for (int th = 0; th < ny_; ++th)
            {
                for (int x = 0; x < nx_; ++x)
                {
                    const std::size_t src_idx =
                        static_cast<std::size_t>(x) * static_cast<std::size_t>(ny_) *
                            static_cast<std::size_t>(nz_) +
                        static_cast<std::size_t>(th) * static_cast<std::size_t>(nz_) +
                        static_cast<std::size_t>(z);
                    const std::size_t dst_idx =
                        (static_cast<std::size_t>(z) * static_cast<std::size_t>(ny_) +
                         static_cast<std::size_t>(th)) *
                            static_cast<std::size_t>(nx_) +
                        static_cast<std::size_t>(x);
                    raw[dst_idx] = vol.data[src_idx];
                }
            }
        }
    }
    else
    {
        // --- 2D slice path: load per-theta slices and assemble ---
        for (int th = 0; th < ny_; ++th)
        {
            const std::filesystem::path path = step_dir / ("th" + std::to_string(th) + "_" + field_name_ + ".npy");

            NpyArray2D slice;
            std::string load_error;

            if (!load_npy_float32_2d(path, slice, load_error))
            {
                error = "Failed loading " + path.string() + ": " + load_error;
                return false;
            }

            if (slice.rows != nz_ || slice.cols != nx_)
            {
                std::ostringstream oss;
                oss << "Slice shape mismatch in " << path.string() << ": expected (" << nz_ << ", " << nx_
                    << "), got (" << slice.rows << ", " << slice.cols << ")";
                error = oss.str();
                return false;
            }
            if (slice.data.size() != expected_slice_size)
            {
                std::ostringstream oss;
                oss << "Slice payload mismatch in " << path.string() << ": expected "
                    << expected_slice_size << " float samples, got " << slice.data.size();
                error = oss.str();
                return false;
            }

            for (int z = 0; z < nz_; ++z)
            {
                const std::size_t row_base = static_cast<std::size_t>(z) * static_cast<std::size_t>(nx_);

                for (int x = 0; x < nx_; ++x)
                {
                    const float value = slice.data[row_base + static_cast<std::size_t>(x)];
                    const std::size_t idx = ((static_cast<std::size_t>(z) * static_cast<std::size_t>(ny_)) +
                                             static_cast<std::size_t>(th)) *
                                                static_cast<std::size_t>(nx_) +
                                            static_cast<std::size_t>(x);
                    if (idx >= raw.size())
                    {
                        error = "internal index overflow while packing volume frame";
                        return false;
                    }
                    raw[idx] = value;
                }
            }
        }
    }

    out.nx = nx_;
    out.ny = ny_;
    out.nz = nz_;
    normalize_volume(raw,
                     out.normalized,
                     out.raw_min,
                     out.raw_max,
                     out.norm_low,
                     out.norm_high,
                     out.nan_count,
                     out.inf_count,
                     out.sanitized_nonfinite_count);
    return true;
}

} // namespace oglcpp
