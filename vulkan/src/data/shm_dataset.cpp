/**
 * @file shm_dataset.cpp
 * @brief Viewer-side shared memory dataset reader implementation.
 */

#include "data/shm_dataset.hpp"
#include "core/output/shm_transport.hpp"

#include <atomic>
#include <cstring>
#include <iostream>

#if defined(__APPLE__) || defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define TMV_HAS_POSIX_SHM 1
#endif

namespace oglcpp
{

ShmDataset::ShmDataset(std::string field_name, const char* shm_name)
    : field_name_(std::move(field_name))
    , shm_name_(shm_name)
{
}

ShmDataset::~ShmDataset()
{
#ifdef TMV_HAS_POSIX_SHM
    if (shm_base_ != nullptr && shm_base_ != MAP_FAILED)
    {
        munmap(const_cast<void*>(shm_base_), shm_size_);
    }
    if (shm_fd_ >= 0)
    {
        ::close(shm_fd_);
    }
    // Reader does NOT unlink — only the writer owns the region
#endif
}

ShmDataset::ShmDataset(ShmDataset&& other) noexcept
    : field_name_(std::move(other.field_name_))
    , shm_name_(std::move(other.shm_name_))
    , shm_base_(other.shm_base_)
    , shm_size_(other.shm_size_)
    , shm_fd_(other.shm_fd_)
    , nx_(other.nx_)
    , ny_(other.ny_)
    , nz_(other.nz_)
    , field_index_(other.field_index_)
    , last_sequence_(other.last_sequence_)
{
    other.shm_base_ = nullptr;
    other.shm_fd_ = -1;
}

ShmDataset& ShmDataset::operator=(ShmDataset&& other) noexcept
{
    if (this != &other)
    {
#ifdef TMV_HAS_POSIX_SHM
        if (shm_base_ != nullptr && shm_base_ != MAP_FAILED)
        {
            munmap(const_cast<void*>(shm_base_), shm_size_);
        }
        if (shm_fd_ >= 0)
        {
            ::close(shm_fd_);
        }
#endif
        field_name_ = std::move(other.field_name_);
        shm_name_ = std::move(other.shm_name_);
        shm_base_ = other.shm_base_;
        shm_size_ = other.shm_size_;
        shm_fd_ = other.shm_fd_;
        nx_ = other.nx_;
        ny_ = other.ny_;
        nz_ = other.nz_;
        field_index_ = other.field_index_;
        last_sequence_ = other.last_sequence_;
        other.shm_base_ = nullptr;
        other.shm_fd_ = -1;
    }
    return *this;
}

bool ShmDataset::scan(std::string& error)
{
#ifndef TMV_HAS_POSIX_SHM
    error = "POSIX shared memory not available on this platform";
    return false;
#else
    shm_fd_ = shm_open(shm_name_.c_str(), O_RDONLY, 0);
    if (shm_fd_ < 0)
    {
        error = "no live simulation found (SHM '" + shm_name_ + "' not available)";
        return false;
    }

    // Get the actual region size
    struct stat st{};
    if (fstat(shm_fd_, &st) != 0 ||
        st.st_size < static_cast<off_t>(sizeof(tmv_shm::ShmHeader)))
    {
        error = "SHM region too small for header";
        ::close(shm_fd_);
        shm_fd_ = -1;
        return false;
    }
    shm_size_ = static_cast<std::size_t>(st.st_size);

    shm_base_ = mmap(nullptr, shm_size_, PROT_READ, MAP_SHARED, shm_fd_, 0);
    if (shm_base_ == MAP_FAILED)
    {
        error = "failed to map SHM region";
        shm_base_ = nullptr;
        ::close(shm_fd_);
        shm_fd_ = -1;
        return false;
    }

    // Validate header
    const auto* header = reinterpret_cast<const tmv_shm::ShmHeader*>(shm_base_);
    if (header->magic != tmv_shm::kShmMagic)
    {
        error = "SHM magic mismatch — region is not a tornado model transport";
        munmap(const_cast<void*>(shm_base_), shm_size_);
        shm_base_ = nullptr;
        ::close(shm_fd_);
        shm_fd_ = -1;
        return false;
    }
    if (header->version != tmv_shm::kShmVersion)
    {
        error = "SHM version mismatch (expected " +
                std::to_string(tmv_shm::kShmVersion) +
                ", got " + std::to_string(header->version) + ")";
        munmap(const_cast<void*>(shm_base_), shm_size_);
        shm_base_ = nullptr;
        ::close(shm_fd_);
        shm_fd_ = -1;
        return false;
    }

    nx_ = header->nx;
    ny_ = header->ny;
    nz_ = header->nz;

    // Find our field in the name table
    field_index_ = -1;
    for (int i = 0; i < header->field_count; ++i)
    {
        std::string name = tmv_shm::read_field_name(shm_base_, i);
        if (name == field_name_)
        {
            field_index_ = i;
            break;
        }
    }

    if (field_index_ < 0)
    {
        error = "field '" + field_name_ + "' not found in SHM transport (available: ";
        for (int i = 0; i < header->field_count; ++i)
        {
            if (i > 0) error += ", ";
            error += tmv_shm::read_field_name(shm_base_, i);
        }
        error += ")";
        munmap(const_cast<void*>(shm_base_), shm_size_);
        shm_base_ = nullptr;
        ::close(shm_fd_);
        shm_fd_ = -1;
        return false;
    }

    // Verify the region is large enough for this field's data
    const std::size_t required = tmv_shm::field_data_offset(field_index_ + 1, nx_, ny_, nz_);
    if (shm_size_ < required)
    {
        error = "SHM region too small for field data";
        munmap(const_cast<void*>(shm_base_), shm_size_);
        shm_base_ = nullptr;
        ::close(shm_fd_);
        shm_fd_ = -1;
        return false;
    }

    std::cout << "[SHM] Attached to live transport '" << shm_name_
              << "', field '" << field_name_
              << "' (index " << field_index_
              << ", " << nx_ << "x" << ny_ << "x" << nz_ << ")\n";
    return true;
#endif
}

bool ShmDataset::load_frame(std::size_t /*frame_idx*/,
                            VolumeFrame& out, std::string& error) const
{
    if (shm_base_ == nullptr || field_index_ < 0)
    {
        error = "SHM not attached";
        return false;
    }

    const auto* header = reinterpret_cast<const tmv_shm::ShmHeader*>(shm_base_);

    // Acquire fence: ensure we see the writer's latest data
    std::atomic_thread_fence(std::memory_order_acquire);

    const std::size_t voxels = static_cast<std::size_t>(nx_) *
                               static_cast<std::size_t>(ny_) *
                               static_cast<std::size_t>(nz_);

    out.nx = nx_;
    out.ny = ny_;
    out.nz = nz_;
    out.normalized.resize(voxels);

    // Data is already normalized [0,1] and in viewer layout by the writer
    const float* src = tmv_shm::field_data_ptr(shm_base_, field_index_, nx_, ny_, nz_);
    std::memcpy(out.normalized.data(), src, voxels * sizeof(float));

    // Stats are approximate since data is already normalized
    out.raw_min = 0.0f;
    out.raw_max = 1.0f;
    out.norm_low = 0.0f;
    out.norm_high = 1.0f;
    out.nan_count = 0;
    out.inf_count = 0;
    out.sanitized_nonfinite_count = 0;

    last_sequence_ = header->sequence_number;
    return true;
}

bool ShmDataset::has_new_frame() const
{
    if (shm_base_ == nullptr)
    {
        return false;
    }
    const auto* header = reinterpret_cast<const tmv_shm::ShmHeader*>(shm_base_);
    std::atomic_thread_fence(std::memory_order_acquire);
    return header->sequence_number != last_sequence_;
}

} // namespace oglcpp
