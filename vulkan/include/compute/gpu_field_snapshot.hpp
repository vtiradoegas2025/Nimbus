#pragma once

/**
 * @file gpu_field_snapshot.hpp
 * @brief GPU-resident simulation state for cross-timestep persistence.
 *
 * On discrete GPUs, each kernel dispatch currently requires H2D (host-to-device)
 * and D2H (device-to-host) transfers. When multiple kernels chain within a
 * timestep (e.g., microphysics → diffusion → advection), these round-trips
 * add ~5-15µs each on PCIe.
 *
 * GPUFieldSnapshot eliminates this by keeping simulation fields resident on
 * GPU memory across kernel dispatches within a timestep. Fields are uploaded
 * once at timestep start and downloaded only when needed for I/O or diagnostics.
 *
 * On unified memory (Apple Silicon / integrated GPUs), this is a no-op wrapper
 * since both CPU and GPU share physical RAM — no copies are needed regardless.
 *
 * Design contract:
 *   1. At timestep start:  snapshot.upload(fields)
 *   2. Between kernels:    snapshot.device_ptr("u"), etc. — zero-copy GPU access
 *   3. At timestep end:    snapshot.download(fields) — only if I/O is needed
 *   4. At shutdown:        snapshot.release()
 *
 * Implementation status: INTERFACE ONLY — not yet wired into the dispatch path.
 * The buffer pool (gpu_buffer_pool.hpp) provides the underlying allocation;
 * this layer adds field-name mapping and lifecycle management.
 */

#if defined(__has_include)
#if __has_include(<vulkan/vulkan.h>)
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#define TMV_GPU_FIELD_SNAPSHOT_HAS_VULKAN 1
#endif
#endif

#if defined(TMV_GPU_FIELD_SNAPSHOT_HAS_VULKAN)

#include "gpu_buffer_pool.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tmv_vulkan
{

/**
 * @brief Tracks a single named field on the GPU.
 */
struct GPUFieldEntry
{
    std::string name;
    int pool_slot = -1;          ///< Index into GpuBufferPool
    VkDeviceSize byte_size = 0;
    bool dirty_on_gpu = false;   ///< True if GPU has newer data than CPU
    bool dirty_on_cpu = false;   ///< True if CPU has newer data than GPU
};

/**
 * @brief GPU-resident snapshot of all simulation fields for one timestep.
 *
 * Manages the lifecycle of field data on the GPU across multiple
 * kernel dispatches within a single timestep.
 */
class GPUFieldSnapshot
{
public:
    explicit GPUFieldSnapshot(GpuBufferPool& pool, bool unified_memory)
        : pool_(pool), unified_memory_(unified_memory) {}

    ~GPUFieldSnapshot() { release(); }

    // Non-copyable, movable
    GPUFieldSnapshot(const GPUFieldSnapshot&) = delete;
    GPUFieldSnapshot& operator=(const GPUFieldSnapshot&) = delete;

    /**
     * @brief Registers a field by name and acquires a GPU buffer for it.
     *
     * @param name       Field identifier (e.g., "u", "theta", "qv").
     * @param byte_size  Required buffer size in bytes.
     * @return True if the buffer was successfully acquired.
     */
    bool register_field(const std::string& name, VkDeviceSize byte_size)
    {
        if (fields_.count(name)) return true; // already registered

        int slot = pool_.acquire(byte_size);
        if (slot < 0) return false;

        GPUFieldEntry entry;
        entry.name = name;
        entry.pool_slot = slot;
        entry.byte_size = byte_size;
        fields_[name] = entry;
        return true;
    }

    /**
     * @brief Uploads host data to the GPU buffer for the named field.
     *
     * On unified memory, this is a no-op (CPU writes go directly to GPU-visible
     * memory). On discrete GPUs, this copies via the staging buffer.
     *
     * @param name     Field name (must be previously registered).
     * @param src      Host pointer to source data.
     * @param bytes    Number of bytes to copy.
     * @return True on success.
     */
    bool upload(const std::string& name, const void* src, VkDeviceSize bytes)
    {
        auto it = fields_.find(name);
        if (it == fields_.end()) return false;

        auto& entry = it->second;
        if (bytes > entry.byte_size) return false;

        if (unified_memory_)
        {
            // On unified memory, the device buffer is host-visible.
            // Copy directly to the mapped pointer.
            void* mapped = pool_.device(entry.pool_slot).mapped;
            if (mapped)
            {
                std::memcpy(mapped, src, bytes);
            }
        }
        else
        {
            // Discrete GPU: copy to staging, then issue a transfer command.
            // The actual vkCmdCopyBuffer must be issued by the dispatch caller
            // since we don't own the command buffer here.
            void* staging_mapped = pool_.staging(entry.pool_slot).mapped;
            if (staging_mapped)
            {
                std::memcpy(staging_mapped, src, bytes);
            }
            entry.dirty_on_cpu = false;
            entry.dirty_on_gpu = false; // will be true after the transfer command
        }
        return true;
    }

    /**
     * @brief Downloads GPU data back to host for the named field.
     *
     * @param name     Field name.
     * @param dst      Host pointer to destination buffer.
     * @param bytes    Number of bytes to read.
     * @return True on success.
     */
    bool download(const std::string& name, void* dst, VkDeviceSize bytes) const
    {
        auto it = fields_.find(name);
        if (it == fields_.end()) return false;

        const auto& entry = it->second;
        if (bytes > entry.byte_size) return false;

        if (unified_memory_)
        {
            const void* mapped = pool_.device(entry.pool_slot).mapped;
            if (mapped)
            {
                std::memcpy(dst, mapped, bytes);
            }
        }
        else
        {
            // Discrete: read from staging (assumes a D2H transfer was issued)
            const void* staging_mapped = pool_.staging(entry.pool_slot).mapped;
            if (staging_mapped)
            {
                std::memcpy(dst, staging_mapped, bytes);
            }
        }
        return true;
    }

    /**
     * @brief Returns the device buffer for dispatch binding.
     *
     * Kernels bind this buffer as a storage buffer for compute dispatch.
     * On unified memory, this is the same buffer the CPU writes to.
     */
    VkBuffer device_buffer(const std::string& name) const
    {
        auto it = fields_.find(name);
        if (it == fields_.end()) return VK_NULL_HANDLE;
        return pool_.device(it->second.pool_slot).buffer;
    }

    /**
     * @brief Returns the staging buffer for transfer commands.
     *
     * On unified memory, returns VK_NULL_HANDLE (no staging needed).
     */
    VkBuffer staging_buffer(const std::string& name) const
    {
        if (unified_memory_) return VK_NULL_HANDLE;
        auto it = fields_.find(name);
        if (it == fields_.end()) return VK_NULL_HANDLE;
        return pool_.staging(it->second.pool_slot).buffer;
    }

    /** @brief Returns true if the field is registered. */
    bool has_field(const std::string& name) const
    {
        return fields_.count(name) > 0;
    }

    /** @brief Returns the number of registered fields. */
    std::size_t field_count() const { return fields_.size(); }

    /** @brief Returns true if running on unified memory (no staging needed). */
    bool is_unified() const { return unified_memory_; }

    /**
     * @brief Releases all pool slots back to the buffer pool.
     */
    void release()
    {
        for (auto& [name, entry] : fields_)
        {
            if (entry.pool_slot >= 0)
            {
                pool_.release(entry.pool_slot);
                entry.pool_slot = -1;
            }
        }
        fields_.clear();
    }

    /**
     * @brief Returns the names of all fields needing H2D transfer.
     *
     * On unified memory, returns empty (no transfers needed).
     * On discrete, returns fields where dirty_on_cpu is true.
     */
    std::vector<std::string> pending_uploads() const
    {
        std::vector<std::string> result;
        if (unified_memory_) return result;
        for (const auto& [name, entry] : fields_)
        {
            if (entry.dirty_on_cpu) result.push_back(name);
        }
        return result;
    }

    /**
     * @brief Returns the names of all fields needing D2H transfer.
     */
    std::vector<std::string> pending_downloads() const
    {
        std::vector<std::string> result;
        if (unified_memory_) return result;
        for (const auto& [name, entry] : fields_)
        {
            if (entry.dirty_on_gpu) result.push_back(name);
        }
        return result;
    }

private:
    GpuBufferPool& pool_;
    bool unified_memory_;
    std::unordered_map<std::string, GPUFieldEntry> fields_;
};

} // namespace tmv_vulkan

#endif // TMV_GPU_FIELD_SNAPSHOT_HAS_VULKAN
