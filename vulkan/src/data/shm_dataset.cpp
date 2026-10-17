/**
 * @file shm_dataset.cpp
 * @brief Viewer-side shared memory dataset reader implementation.
 */

#include "data/shm_dataset.hpp"
#include "core/output/shm_transport.hpp"

#include <atomic>
#include <cstring>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define TMV_HAS_POSIX_SHM 1
#define TMV_HAS_NATIVE_SHM 1
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define TMV_HAS_NATIVE_SHM 1
#endif

namespace oglcpp
{
namespace
{

bool checked_region_size(const tmv_shm::ShmHeader& header, std::size_t& required)
{
    const std::size_t nx = static_cast<std::size_t>(header.nx);
    const std::size_t ny = static_cast<std::size_t>(header.ny);
    const std::size_t nz = static_cast<std::size_t>(header.nz);
    const std::size_t max = std::numeric_limits<std::size_t>::max();
    if (nx > max / ny || nx * ny > max / nz)
    {
        return false;
    }

    const std::size_t voxels = nx * ny * nz;
    const std::size_t prefix = tmv_shm::field_data_offset(0, 1, 1, 1);
    const std::size_t bytes_per_voxel =
        sizeof(float) * static_cast<std::size_t>(header.field_count);
    if (voxels > (max - prefix) / bytes_per_voxel)
    {
        return false;
    }
    required = prefix + voxels * bytes_per_voxel;
    return true;
}

} // namespace

ShmDataset::ShmDataset(std::string field_name, const char* shm_name)
    : field_name_(std::move(field_name))
    , shm_name_(shm_name)
{
}

ShmDataset::~ShmDataset()
{
    release_mapping();
}

void ShmDataset::release_mapping() noexcept
{
#if defined(TMV_HAS_POSIX_SHM)
    if (shm_base_ != nullptr && shm_base_ != MAP_FAILED)
    {
        munmap(const_cast<void*>(shm_base_), shm_size_);
    }
    if (shm_fd_ >= 0)
    {
        ::close(shm_fd_);
    }
#elif defined(_WIN32)
    if (shm_base_ != nullptr)
    {
        UnmapViewOfFile(shm_base_);
    }
    if (shm_handle_ != nullptr)
    {
        CloseHandle(static_cast<HANDLE>(shm_handle_));
    }
#endif
    // Reader never destroys/unlinks the named region; the writer owns it.
    shm_base_ = nullptr;
    shm_size_ = 0;
    shm_fd_ = -1;
    publication_generation_.clear();
#if defined(_WIN32)
    shm_handle_ = nullptr;
#endif
}

ShmDataset::ShmDataset(ShmDataset&& other) noexcept
    : field_name_(std::move(other.field_name_))
    , shm_name_(std::move(other.shm_name_))
    , shm_base_(other.shm_base_)
    , shm_size_(other.shm_size_)
    , shm_fd_(other.shm_fd_)
#if defined(_WIN32)
    , shm_handle_(other.shm_handle_)
#endif
    , nx_(other.nx_)
    , ny_(other.ny_)
    , nz_(other.nz_)
    , source_nx_(other.source_nx_)
    , source_ny_(other.source_ny_)
    , source_nz_(other.source_nz_)
    , physical_domain_(other.physical_domain_)
    , grid_spacing_(other.grid_spacing_)
    , field_index_(other.field_index_)
    , field_metadata_(other.field_metadata_)
    , publication_generation_(std::move(other.publication_generation_))
    , source_snapshot_(std::move(other.source_snapshot_))
    , raw_snapshot_(std::move(other.raw_snapshot_))
    , last_sequence_(other.last_sequence_)
    , last_simulation_time_(other.last_simulation_time_)
    , last_horizontal_frame_(other.last_horizontal_frame_)
{
    other.shm_base_ = nullptr;
    other.shm_fd_ = -1;
#if defined(_WIN32)
    other.shm_handle_ = nullptr;
#endif
}

ShmDataset& ShmDataset::operator=(ShmDataset&& other) noexcept
{
    if (this != &other)
    {
        release_mapping();
        field_name_ = std::move(other.field_name_);
        shm_name_ = std::move(other.shm_name_);
        shm_base_ = other.shm_base_;
        shm_size_ = other.shm_size_;
        shm_fd_ = other.shm_fd_;
#if defined(_WIN32)
        shm_handle_ = other.shm_handle_;
#endif
        nx_ = other.nx_;
        ny_ = other.ny_;
        nz_ = other.nz_;
        source_nx_ = other.source_nx_;
        source_ny_ = other.source_ny_;
        source_nz_ = other.source_nz_;
        physical_domain_ = other.physical_domain_;
        grid_spacing_ = other.grid_spacing_;
        field_index_ = other.field_index_;
        field_metadata_ = other.field_metadata_;
        publication_generation_ =
            std::move(other.publication_generation_);
        source_snapshot_ = std::move(other.source_snapshot_);
        raw_snapshot_ = std::move(other.raw_snapshot_);
        last_sequence_ = other.last_sequence_;
        last_simulation_time_ = other.last_simulation_time_;
        last_horizontal_frame_ = other.last_horizontal_frame_;
        other.shm_base_ = nullptr;
        other.shm_fd_ = -1;
#if defined(_WIN32)
        other.shm_handle_ = nullptr;
#endif
    }
    return *this;
}

bool ShmDataset::scan(std::string& error)
{
#ifndef TMV_HAS_NATIVE_SHM
    error = "native shared memory not available on this platform";
    return false;
#else
    release_mapping();
#if defined(TMV_HAS_POSIX_SHM)
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
#elif defined(_WIN32)
    const std::wstring mapping_name = tmv_shm::windows_mapping_name(shm_name_);
    HANDLE mapping = OpenFileMappingW(FILE_MAP_READ, FALSE, mapping_name.c_str());
    if (mapping == nullptr)
    {
        error = "no live simulation found (shared mapping '" + shm_name_ +
                "' not available)";
        return false;
    }
    shm_handle_ = mapping;
    shm_base_ = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (shm_base_ == nullptr)
    {
        error = "failed to map live shared-memory region";
        release_mapping();
        return false;
    }
    MEMORY_BASIC_INFORMATION region{};
    if (VirtualQuery(shm_base_, &region, sizeof(region)) == 0 ||
        region.RegionSize < sizeof(tmv_shm::ShmHeader))
    {
        error = "shared-memory region too small for header";
        release_mapping();
        return false;
    }
    shm_size_ = region.RegionSize;
#endif

    // Validate header
    const auto* header = reinterpret_cast<const tmv_shm::ShmHeader*>(shm_base_);
    const uint64_t attached_sequence =
        header->sequence_number.load(std::memory_order_acquire);
    if (attached_sequence == 1U)
    {
        error = "SHM transport is still initializing or publishing a frame";
        release_mapping();
        return false;
    }
    if (header->magic != tmv_shm::kShmMagic)
    {
        error = "SHM magic mismatch — region is not a tornado model transport";
        release_mapping();
        return false;
    }
    if (header->version != tmv_shm::kShmVersion)
    {
        error = "SHM version mismatch (expected " +
                std::to_string(tmv_shm::kShmVersion) +
                ", got " + std::to_string(header->version) + ")";
        release_mapping();
        return false;
    }
    if (header->header_size != sizeof(tmv_shm::ShmHeader))
    {
        error = "SHM header ABI size mismatch";
        release_mapping();
        return false;
    }
    if (header->field_count <= 0 || header->field_count > tmv_shm::kMaxFields ||
        header->nx <= 0 || header->ny <= 0 || header->nz <= 0 ||
        !tmv_shm::physical_domain_valid(
            header->physical_domain, header->nx, header->ny, header->nz) ||
        !tmv_shm::grid_spacing_valid(header->grid_spacing_m) ||
        !tmv_shm::horizontal_frame_valid(header->horizontal_frame) ||
        header->publication_generation[
            tmv_shm::kPublicationGenerationLength - 1] != '\0' ||
        !tmv_shm::publication_generation_valid(std::string_view(
            header->publication_generation,
            tmv_shm::kPublicationGenerationLength - 1)))
    {
        error = "SHM header has invalid dimensions or field count";
        release_mapping();
        return false;
    }
    std::size_t required_region_size = 0;
    if (!checked_region_size(*header, required_region_size) ||
        shm_size_ < required_region_size)
    {
        error = "SHM region size does not match its declared payload";
        release_mapping();
        return false;
    }
    source_nx_ = header->nx;
    source_ny_ = header->ny;
    source_nz_ = header->nz;
    physical_domain_ = header->physical_domain;
    grid_spacing_ = {header->grid_spacing_m.x_m,
                     header->grid_spacing_m.y_m,
                     header->grid_spacing_m.z_m};
    publication_generation_ = header->publication_generation;
    last_horizontal_frame_ = header->horizontal_frame;
    nx_ = physical_domain_.upper[0] - physical_domain_.lower[0] + 1;
    ny_ = physical_domain_.upper[1] - physical_domain_.lower[1] + 1;
    nz_ = physical_domain_.upper[2] - physical_domain_.lower[2] + 1;

    // Find our field in the name table
    field_index_ = -1;
    for (int i = 0; i < header->field_count; ++i)
    {
        if (!tmv_shm::field_name_is_terminated(shm_base_, i))
        {
            error = "SHM field-name table contains an unterminated entry";
            release_mapping();
            return false;
        }
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
        release_mapping();
        return false;
    }

    field_metadata_ = *tmv_shm::field_metadata_ptr(shm_base_, field_index_);
    if (field_metadata_.struct_size != sizeof(tmv_shm::ShmFieldMetadata) ||
        field_metadata_.encoding != tmv_shm::ShmValueEncoding::RawFloat32 ||
        (field_metadata_.polarity != tmv_shm::ShmFieldPolarity::Unipolar &&
         field_metadata_.polarity != tmv_shm::ShmFieldPolarity::Bipolar) ||
        field_metadata_.nonfinite_policy !=
            tmv_shm::ShmNonfinitePolicy::PreserveAndSanitizeInReader ||
        field_metadata_.grid_location != tmv_shm::ShmGridLocation::CellCenter ||
        field_metadata_.display_scaling != tmv_shm::ShmDisplayScaling::RobustPerFrame ||
        std::memchr(field_metadata_.units, '\0', tmv_shm::kMaxUnitLen) == nullptr)
    {
        error = "unsupported or malformed SHM field metadata";
        release_mapping();
        return false;
    }

    // Verify the region is large enough for this field's data
    const std::size_t required = tmv_shm::field_data_offset(
        field_index_ + 1, source_nx_, source_ny_, source_nz_);
    if (shm_size_ < required)
    {
        error = "SHM region too small for field data";
        release_mapping();
        return false;
    }

    std::cout << "[SHM] Attached to live transport '" << shm_name_
              << "', field '" << field_name_
              << "' (index " << field_index_
              << ", " << nx_ << "x" << ny_ << "x" << nz_
              << " physical in " << source_nx_ << "x" << source_ny_ << "x" << source_nz_
              << ", spacing=" << grid_spacing_.x_m << "x"
              << grid_spacing_.y_m << "x" << grid_spacing_.z_m << " m"
              << ", raw float32, units=" << field_metadata_.units
              << ", polarity="
              << (is_bipolar() ? "bipolar" : "unipolar") << ")\n";
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

    const std::size_t source_voxels = static_cast<std::size_t>(source_nx_) *
                                      static_cast<std::size_t>(source_ny_) *
                                      static_cast<std::size_t>(source_nz_);
    const std::size_t physical_voxels = static_cast<std::size_t>(nx_) *
                                        static_cast<std::size_t>(ny_) *
                                        static_cast<std::size_t>(nz_);
    const float* src = tmv_shm::field_data_ptr(
        shm_base_, field_index_, source_nx_, source_ny_, source_nz_);
    source_snapshot_.resize(source_voxels);
    raw_snapshot_.resize(physical_voxels);

    // Copy only a coherent version. A full fence before the trailing sequence
    // load prevents the raw memcpy from moving outside its seqlock bracket.
    constexpr int kMaxSnapshotAttempts = 64;
    for (int attempt = 0; attempt < kMaxSnapshotAttempts; ++attempt)
    {
        const uint64_t before =
            header->sequence_number.load(std::memory_order_acquire);
        if ((before & 1U) != 0U)
        {
            std::this_thread::yield();
            continue;
        }

        const double snapshot_time = header->simulation_time;
        const tmv_shm::ShmHorizontalFrame snapshot_frame =
            header->horizontal_frame;
        std::memcpy(source_snapshot_.data(), src, source_voxels * sizeof(float));
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const uint64_t after =
            header->sequence_number.load(std::memory_order_acquire);

        if (before != after || (after & 1U) != 0U)
        {
            std::this_thread::yield();
            continue;
        }

        for (int z = 0; z < nz_; ++z)
        {
            const int source_z = physical_domain_.lower[2] + z;
            for (int y = 0; y < ny_; ++y)
            {
                const int source_y = physical_domain_.lower[1] + y;
                for (int x = 0; x < nx_; ++x)
                {
                    const int source_x = physical_domain_.lower[0] + x;
                    const std::size_t source_index =
                        (static_cast<std::size_t>(source_z) *
                             static_cast<std::size_t>(source_ny_) +
                         static_cast<std::size_t>(source_y)) *
                            static_cast<std::size_t>(source_nx_) +
                        static_cast<std::size_t>(source_x);
                    const std::size_t physical_index =
                        (static_cast<std::size_t>(z) * static_cast<std::size_t>(ny_) +
                         static_cast<std::size_t>(y)) * static_cast<std::size_t>(nx_) +
                        static_cast<std::size_t>(x);
                    raw_snapshot_[physical_index] = source_snapshot_[source_index];
                }
            }
        }

        out.nx = nx_;
        out.ny = ny_;
        out.nz = nz_;
        ExportDataset::normalize_volume(
            raw_snapshot_,
            out.normalized,
            out.raw_min,
            out.raw_max,
            out.norm_low,
            out.norm_high,
            out.nan_count,
            out.inf_count,
            out.sanitized_nonfinite_count,
            0.0f,
            field_metadata_.polarity == tmv_shm::ShmFieldPolarity::Bipolar);

        last_sequence_ = after;
        last_simulation_time_ = snapshot_time;
        last_horizontal_frame_ = snapshot_frame;
        return true;
    }

    error = "SHM snapshot changed while field data was being copied";
    return false;
}

bool ShmDataset::has_new_frame() const
{
    if (shm_base_ == nullptr)
    {
        return false;
    }
    const auto* header = reinterpret_cast<const tmv_shm::ShmHeader*>(shm_base_);
    const uint64_t sequence =
        header->sequence_number.load(std::memory_order_acquire);
    return (sequence & 1U) == 0U && sequence != last_sequence_;
}

} // namespace oglcpp
