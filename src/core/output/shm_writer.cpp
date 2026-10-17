/**
 * @file shm_writer.cpp
 * @brief Simulation-side shared memory writer implementation.
 */

#include "core/output/shm_writer.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <new>
#include <unordered_set>

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

ShmWriter::~ShmWriter()
{
    close();
}

bool ShmWriter::open(int nx, int ny, int nz,
                     const std::vector<std::string>& field_names,
                     const char* shm_name)
{
    return open(nx, ny, nz, field_names,
                tmv_shm::full_physical_domain(nx, ny, nz),
                tmv_shm::ShmGridSpacing{1.0, 1.0, 1.0}, shm_name);
}

bool ShmWriter::open(int nx, int ny, int nz,
                     const std::vector<std::string>& field_names,
                     const tmv_shm::ShmPhysicalDomain& physical_domain,
                     const char* shm_name)
{
    return open(nx, ny, nz, field_names, physical_domain,
                tmv_shm::ShmGridSpacing{1.0, 1.0, 1.0},
                tmv_shm::ShmHorizontalFrame{}, shm_name);
}

bool ShmWriter::open(int nx, int ny, int nz,
                     const std::vector<std::string>& field_names,
                     const tmv_shm::ShmPhysicalDomain& physical_domain,
                     const tmv_shm::ShmGridSpacing& grid_spacing_m,
                     const char* shm_name)
{
    return open(nx, ny, nz, field_names, physical_domain, grid_spacing_m,
                tmv_shm::ShmHorizontalFrame{}, shm_name);
}

bool ShmWriter::open(int nx, int ny, int nz,
                     const std::vector<std::string>& field_names,
                     const tmv_shm::ShmPhysicalDomain& physical_domain,
                     const tmv_shm::ShmGridSpacing& grid_spacing_m,
                     const tmv_shm::ShmHorizontalFrame& horizontal_frame,
                     const char* shm_name,
                     const std::string& publication_generation)
{
#ifndef TMV_HAS_NATIVE_SHM
    (void)nx; (void)ny; (void)nz; (void)field_names; (void)physical_domain;
    (void)grid_spacing_m; (void)horizontal_frame; (void)shm_name;
    tmv::log_error("[SHM] native shared memory is unavailable on this platform");
    return false;
#else
    if (field_names.empty() || field_names.size() > tmv_shm::kMaxFields ||
        nx <= 0 || ny <= 0 || nz <= 0 ||
        shm_name == nullptr || !tmv_shm::valid_logical_shm_name(shm_name) ||
        !tmv_shm::physical_domain_valid(physical_domain, nx, ny, nz) ||
        !tmv_shm::grid_spacing_valid(grid_spacing_m) ||
        !tmv_shm::horizontal_frame_valid(horizontal_frame) ||
        !tmv_shm::publication_generation_valid(publication_generation))
    {
        tmv::log_error("[SHM] invalid shared-memory writer parameters");
        return false;
    }

    std::unordered_set<std::string> unique_names;
    for (const std::string& name : field_names)
    {
        if (name.empty() || name.size() >= tmv_shm::kMaxFieldNameLen ||
            name.find('\0') != std::string::npos ||
            !unique_names.insert(name).second)
        {
            tmv::log_error("[SHM] field names must be unique, non-empty, and fit ",
                           (tmv_shm::kMaxFieldNameLen - 1), " bytes");
            return false;
        }
    }

    const int field_count = static_cast<int>(field_names.size());

    nx_ = nx;
    ny_ = ny;
    nz_ = nz;
    field_count_ = field_count;
    fields_written_.assign(static_cast<std::size_t>(field_count_), false);
    shm_name_ = shm_name;
    shm_size_ = tmv_shm::total_shm_size(nx, ny, nz, field_count);

    // POSIX names survive until unlinked. Windows named mappings are destroyed
    // when their last handle closes and can be safely reopened while a reader
    // still holds the same mapping, provided the requested view fits.
#if defined(TMV_HAS_POSIX_SHM)
    shm_unlink(shm_name);

    shm_fd_ = shm_open(shm_name, O_CREAT | O_RDWR, 0644);
    if (shm_fd_ < 0)
    {
        tmv::log_error("[SHM] failed to create shared-memory region '", shm_name,
                       "'");
        return false;
    }
#elif defined(_WIN32)
    const std::wstring mapping_name = tmv_shm::windows_mapping_name(shm_name_);
    const auto bytes = static_cast<std::uint64_t>(shm_size_);
    const DWORD size_high = static_cast<DWORD>(bytes >> 32U);
    const DWORD size_low = static_cast<DWORD>(bytes & 0xffffffffULL);
    HANDLE mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr,
                                        PAGE_READWRITE, size_high, size_low,
                                        mapping_name.c_str());
    if (mapping == nullptr)
    {
        tmv::log_error("[SHM] failed to create Windows shared-memory mapping '",
                       shm_name, "' (error ", GetLastError(), ")");
        return false;
    }
    shm_handle_ = mapping;
    shm_base_ = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, shm_size_);
    if (shm_base_ == nullptr)
    {
        tmv::log_error("[SHM] failed to map Windows shared-memory region (error ",
                       GetLastError(), ")");
        CloseHandle(mapping);
        shm_handle_ = nullptr;
        return false;
    }
#endif

#if defined(TMV_HAS_POSIX_SHM)
    if (ftruncate(shm_fd_, static_cast<off_t>(shm_size_)) != 0)
    {
        tmv::log_error("[SHM] failed to resize shared-memory region");
        ::close(shm_fd_);
        shm_unlink(shm_name);
        shm_fd_ = -1;
        return false;
    }

    shm_base_ = mmap(nullptr, shm_size_, PROT_READ | PROT_WRITE,
                      MAP_SHARED, shm_fd_, 0);
    if (shm_base_ == MAP_FAILED)
    {
        tmv::log_error("[SHM] failed to map shared-memory region");
        shm_base_ = nullptr;
        ::close(shm_fd_);
        shm_unlink(shm_name);
        shm_fd_ = -1;
        return false;
    }
#endif

    // Zero the entire region
    std::memset(shm_base_, 0, shm_size_);

    // Construct an initializing (odd) header. magic remains zero until the
    // immutable name/metadata tables are ready, so early attach attempts fail
    // closed instead of interpreting a partially initialized ABI.
    auto* header = ::new (shm_base_) tmv_shm::ShmHeader{};
    header->sequence_number.store(1, std::memory_order_relaxed);
    header->magic = 0;
    header->version = tmv_shm::kShmVersion;
    header->nx = static_cast<int32_t>(nx);
    header->ny = static_cast<int32_t>(ny);
    header->nz = static_cast<int32_t>(nz);
    header->field_count = static_cast<int32_t>(field_count);
    header->header_size = static_cast<uint32_t>(sizeof(tmv_shm::ShmHeader));
    header->reserved = 0;
    std::memcpy(header->publication_generation,
                publication_generation.data(), publication_generation.size());
    header->publication_generation[32] = '\0';
    header->physical_domain = physical_domain;
    header->grid_spacing_m = grid_spacing_m;
    header->horizontal_frame = horizontal_frame;
    header->horizontal_frame.grid_origin_earth_m[0] =
        horizontal_frame.grid_origin_t0_earth_m[0];
    header->horizontal_frame.grid_origin_earth_m[1] =
        horizontal_frame.grid_origin_t0_earth_m[1];
    header->simulation_time = 0.0;

    // Write immutable field names and raw-value descriptors.
    for (int i = 0; i < field_count; ++i)
    {
        tmv_shm::write_field_name(shm_base_, i, field_names[i].c_str());
        auto* metadata = tmv_shm::field_metadata_ptr(shm_base_, i);
        ::new (metadata) tmv_shm::ShmFieldMetadata(
            tmv_shm::make_field_metadata(field_names[i]));
    }

    // Keep initialization as the first open transaction. No reader may treat
    // allocation-zeroed payloads as a scientific initial state; the first
    // complete field bundle publishes magic and closes sequence 1 to 2.
    frame_open_ = true;

    tmv::log_info("[SHM] live transport active: ", shm_name,
                  " (", nx, "x", ny, "x", nz,
                  ", ", field_count, " fields, ",
                  (shm_size_ / (1024 * 1024)), " MiB)");
    return true;
#endif
}

void ShmWriter::begin_frame()
{
    if (!shm_base_ || frame_open_)
    {
        return;
    }

    auto* header = reinterpret_cast<tmv_shm::ShmHeader*>(shm_base_);
    const uint64_t previous =
        header->sequence_number.fetch_add(1, std::memory_order_acq_rel);
    if ((previous & 1U) != 0U)
    {
        // This indicates an internal single-writer protocol violation. Restore
        // an odd value so readers continue to reject the incomplete payload.
        header->sequence_number.fetch_add(1, std::memory_order_acq_rel);
        tmv::log_error("[SHM] refusing nested frame publication");
        return;
    }
    frame_open_ = true;
    std::fill(fields_written_.begin(), fields_written_.end(), false);
}

bool ShmWriter::write_field(int field_index, const Field3D& field)
{
    if (!shm_base_)
    {
        return false;
    }

    const auto* header = reinterpret_cast<const tmv_shm::ShmHeader*>(shm_base_);
    if (field_index < 0 || field_index >= header->field_count)
    {
        return false;
    }

    const int nr = field.size_r();
    const int nth = field.size_th();
    const int nz = field.size_z();
    if (nr != nx_ || nth != ny_ || nz != nz_)
    {
        tmv::log_error(
            "[SHM] field dimensions do not match the transport region");
        return false;
    }

    begin_frame();
    if (!frame_open_)
    {
        return false;
    }

    float* dst = tmv_shm::field_data_ptr(shm_base_, field_index, nx_, ny_, nz_);

    // Transpose from Field3D [NR][NTH][NZ] -> viewer layout [Z][TH][X].
    // No anomaly transform, absolute value, clipping, or normalization belongs
    // in the scientific transport.
    for (int k = 0; k < nz_; ++k)
    {
        for (int j = 0; j < ny_; ++j)
        {
            for (int i = 0; i < nx_; ++i)
            {
                const std::size_t viewer_idx =
                    static_cast<std::size_t>(k) * static_cast<std::size_t>(ny_) *
                    static_cast<std::size_t>(nx_) +
                    static_cast<std::size_t>(j) * static_cast<std::size_t>(nx_) +
                    static_cast<std::size_t>(i);

                dst[viewer_idx] = field(i, j, k);
            }
        }
    }
    fields_written_[static_cast<std::size_t>(field_index)] = true;
    return true;
}

bool ShmWriter::commit(double simulation_time)
{
    if (!shm_base_)
        return false;
    const auto* header =
        reinterpret_cast<const tmv_shm::ShmHeader*>(shm_base_);
    return commit(simulation_time, header->horizontal_frame);
}

bool ShmWriter::commit(
    double simulation_time,
    const tmv_shm::ShmHorizontalFrame& horizontal_frame)
{
    if (!shm_base_)
    {
        return false;
    }
    if (!std::isfinite(simulation_time) ||
        !tmv_shm::horizontal_frame_valid(horizontal_frame))
        return false;

    auto* header = reinterpret_cast<tmv_shm::ShmHeader*>(shm_base_);
    begin_frame();
    if (!frame_open_)
    {
        return false;
    }
    if (fields_written_.size() != static_cast<std::size_t>(field_count_) ||
        std::find(fields_written_.begin(), fields_written_.end(), false) !=
            fields_written_.end())
    {
        // Do not expose a new time with stale/zero fields. The sequence stays
        // odd, and the next attempted frame must rewrite every declared slot
        // before it can restore an even, readable publication.
        std::fill(fields_written_.begin(), fields_written_.end(), false);
        tmv::log_error("[SHM] refusing incomplete field-bundle publication");
        return false;
    }
    header->simulation_time = simulation_time;
    header->horizontal_frame = horizontal_frame;
    header->horizontal_frame.grid_origin_earth_m[0] =
        header->horizontal_frame.grid_origin_t0_earth_m[0] +
        header->horizontal_frame.frame_velocity_earth_ms[0] * simulation_time;
    header->horizontal_frame.grid_origin_earth_m[1] =
        header->horizontal_frame.grid_origin_t0_earth_m[1] +
        header->horizontal_frame.frame_velocity_earth_ms[1] * simulation_time;
    header->magic = tmv_shm::kShmMagic;

    // The release operation publishes every payload store and simulation_time
    // under the next even sequence.
    header->sequence_number.fetch_add(1, std::memory_order_release);
    frame_open_ = false;
    return true;
}

std::uint64_t ShmWriter::sequence_number() const
{
    if (!shm_base_)
        return 0;
    const auto* header =
        reinterpret_cast<const tmv_shm::ShmHeader*>(shm_base_);
    return header->sequence_number.load(std::memory_order_acquire);
}

void ShmWriter::close()
{
#if defined(TMV_HAS_POSIX_SHM)
    if (shm_base_ != nullptr)
    {
        munmap(shm_base_, shm_size_);
        shm_base_ = nullptr;
        frame_open_ = false;
        fields_written_.clear();
        field_count_ = 0;
    }
    if (shm_fd_ >= 0)
    {
        ::close(shm_fd_);
        shm_fd_ = -1;
    }
    if (!shm_name_.empty())
    {
        shm_unlink(shm_name_.c_str());
        tmv::log_info("[SHM] live transport closed: ", shm_name_);
        shm_name_.clear();
    }
#elif defined(_WIN32)
    if (shm_base_ != nullptr)
    {
        UnmapViewOfFile(shm_base_);
        shm_base_ = nullptr;
    }
    if (shm_handle_ != nullptr)
    {
        CloseHandle(static_cast<HANDLE>(shm_handle_));
        shm_handle_ = nullptr;
    }
    if (!shm_name_.empty())
    {
        tmv::log_info("[SHM] live transport closed: ", shm_name_);
        shm_name_.clear();
    }
    frame_open_ = false;
    fields_written_.clear();
    field_count_ = 0;
#endif
}
