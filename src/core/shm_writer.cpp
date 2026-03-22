/**
 * @file shm_writer.cpp
 * @brief Simulation-side shared memory writer implementation.
 */

#include "core/shm_writer.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>

#if defined(__APPLE__) || defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define TMV_HAS_POSIX_SHM 1
#endif

ShmWriter::~ShmWriter()
{
    close();
}

bool ShmWriter::open(int nx, int ny, int nz,
                     const std::vector<std::string>& field_names,
                     const char* shm_name)
{
#ifndef TMV_HAS_POSIX_SHM
    (void)nx; (void)ny; (void)nz; (void)field_names; (void)shm_name;
    std::cerr << "[SHM] POSIX shared memory not available on this platform\n";
    return false;
#else
    if (field_names.empty() || nx <= 0 || ny <= 0 || nz <= 0)
    {
        std::cerr << "[SHM] Invalid parameters for shared memory writer\n";
        return false;
    }

    const int field_count = std::min(static_cast<int>(field_names.size()),
                                     tmv_shm::kMaxFields);

    nx_ = nx;
    ny_ = ny;
    nz_ = nz;
    shm_name_ = shm_name;
    shm_size_ = tmv_shm::total_shm_size(nx, ny, nz, field_count);

    // Unlink any stale region from a previous run
    shm_unlink(shm_name);

    shm_fd_ = shm_open(shm_name, O_CREAT | O_RDWR, 0644);
    if (shm_fd_ < 0)
    {
        std::cerr << "[SHM] Failed to create shared memory region '" << shm_name << "'\n";
        return false;
    }

    if (ftruncate(shm_fd_, static_cast<off_t>(shm_size_)) != 0)
    {
        std::cerr << "[SHM] Failed to resize shared memory region\n";
        ::close(shm_fd_);
        shm_unlink(shm_name);
        shm_fd_ = -1;
        return false;
    }

    shm_base_ = mmap(nullptr, shm_size_, PROT_READ | PROT_WRITE,
                      MAP_SHARED, shm_fd_, 0);
    if (shm_base_ == MAP_FAILED)
    {
        std::cerr << "[SHM] Failed to map shared memory region\n";
        shm_base_ = nullptr;
        ::close(shm_fd_);
        shm_unlink(shm_name);
        shm_fd_ = -1;
        return false;
    }

    // Zero the entire region
    std::memset(shm_base_, 0, shm_size_);

    // Write header
    auto* header = reinterpret_cast<tmv_shm::ShmHeader*>(shm_base_);
    header->magic = tmv_shm::kShmMagic;
    header->version = tmv_shm::kShmVersion;
    header->nx = static_cast<int32_t>(nx);
    header->ny = static_cast<int32_t>(ny);
    header->nz = static_cast<int32_t>(nz);
    header->field_count = static_cast<int32_t>(field_count);
    header->simulation_time = 0.0;
    header->sequence_number = 0;

    // Write field names
    for (int i = 0; i < field_count; ++i)
    {
        tmv_shm::write_field_name(shm_base_, i, field_names[i].c_str());
    }

    std::cout << "[SHM] Live transport active: " << shm_name
              << " (" << nx << "x" << ny << "x" << nz
              << ", " << field_count << " fields"
              << ", " << (shm_size_ / (1024 * 1024)) << " MB)\n";
    return true;
#endif
}

void ShmWriter::write_field(int field_index, const Field3D& field)
{
    if (!shm_base_)
    {
        return;
    }

    const auto* header = reinterpret_cast<const tmv_shm::ShmHeader*>(shm_base_);
    if (field_index < 0 || field_index >= header->field_count)
    {
        return;
    }

    const int nr = field.size_r();
    const int nth = field.size_th();
    const int nz = field.size_z();

    float* dst = tmv_shm::field_data_ptr(shm_base_, field_index, nx_, ny_, nz_);
    const std::size_t voxels = static_cast<std::size_t>(nx_) *
                               static_cast<std::size_t>(ny_) *
                               static_cast<std::size_t>(nz_);

    // Transpose from Field3D [NR][NTH][NZ] → viewer layout [Z][TH][X]
    // and compute min/max for normalization in one pass
    float fmin = std::numeric_limits<float>::max();
    float fmax = std::numeric_limits<float>::lowest();

    for (int k = 0; k < nz && k < nz_; ++k)
    {
        for (int j = 0; j < nth && j < ny_; ++j)
        {
            for (int i = 0; i < nr && i < nx_; ++i)
            {
                float val = field(i, j, k);
                const std::size_t viewer_idx =
                    static_cast<std::size_t>(k) * static_cast<std::size_t>(ny_) *
                    static_cast<std::size_t>(nx_) +
                    static_cast<std::size_t>(j) * static_cast<std::size_t>(nx_) +
                    static_cast<std::size_t>(i);

                if (std::isfinite(val))
                {
                    fmin = std::min(fmin, val);
                    fmax = std::max(fmax, val);
                    dst[viewer_idx] = val;
                }
                else
                {
                    dst[viewer_idx] = 0.0f;
                }
            }
        }
    }

    // Normalize to [0,1] using robust percentile clipping (2%–98%)
    // Build a 256-bin histogram for fast percentile estimation
    if (fmin >= fmax)
    {
        // Constant field or all non-finite → fill with 0.5
        for (std::size_t idx = 0; idx < voxels; ++idx)
        {
            dst[idx] = 0.5f;
        }
        return;
    }

    constexpr int kBins = 256;
    int histogram[kBins] = {};
    const float inv_range = static_cast<float>(kBins - 1) / (fmax - fmin);
    int finite_count = 0;

    for (std::size_t idx = 0; idx < voxels; ++idx)
    {
        float val = dst[idx];
        if (val != 0.0f || (fmin <= 0.0f && fmax >= 0.0f))
        {
            int bin = static_cast<int>((val - fmin) * inv_range);
            bin = std::max(0, std::min(kBins - 1, bin));
            histogram[bin]++;
            finite_count++;
        }
    }

    // Find 2% and 98% percentile values
    const int low_target = std::max(1, finite_count * 2 / 100);
    const int high_target = std::max(1, finite_count * 98 / 100);
    float norm_low = fmin;
    float norm_high = fmax;
    int cumulative = 0;

    for (int b = 0; b < kBins; ++b)
    {
        cumulative += histogram[b];
        if (cumulative >= low_target && norm_low == fmin)
        {
            norm_low = fmin + static_cast<float>(b) / inv_range;
        }
        if (cumulative >= high_target)
        {
            norm_high = fmin + static_cast<float>(b) / inv_range;
            break;
        }
    }

    if (norm_high <= norm_low)
    {
        norm_high = norm_low + 1.0e-6f;
    }

    // Apply normalization in-place
    const float scale = 1.0f / (norm_high - norm_low);
    for (std::size_t idx = 0; idx < voxels; ++idx)
    {
        float val = (dst[idx] - norm_low) * scale;
        dst[idx] = std::max(0.0f, std::min(1.0f, val));
    }
}

void ShmWriter::commit(double simulation_time)
{
    if (!shm_base_)
    {
        return;
    }

    auto* header = reinterpret_cast<tmv_shm::ShmHeader*>(shm_base_);
    header->simulation_time = simulation_time;

    // Release fence: ensure all field writes are visible before bumping sequence
    std::atomic_thread_fence(std::memory_order_release);
    header->sequence_number++;
}

void ShmWriter::close()
{
#ifdef TMV_HAS_POSIX_SHM
    if (shm_base_ != nullptr)
    {
        munmap(shm_base_, shm_size_);
        shm_base_ = nullptr;
    }
    if (shm_fd_ >= 0)
    {
        ::close(shm_fd_);
        shm_fd_ = -1;
    }
    if (!shm_name_.empty())
    {
        shm_unlink(shm_name_.c_str());
        std::cout << "[SHM] Live transport closed: " << shm_name_ << "\n";
        shm_name_.clear();
    }
#endif
}
