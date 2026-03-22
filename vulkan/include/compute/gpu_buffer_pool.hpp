#pragma once

/**
 * @file gpu_buffer_pool.hpp
 * @brief Persistent GPU buffer pool with acquire/release semantics.
 *
 * Manages staging+device buffer pairs for Vulkan compute dispatch.
 * Slots grow on demand and are never freed during a run — only at shutdown.
 * This avoids per-dispatch allocation overhead while keeping memory bounded
 * to the high-water mark of concurrent buffer usage.
 *
 * Internal to the Vulkan compute backend; not part of the public API.
 */

#if defined(__has_include)
#if __has_include(<vulkan/vulkan.h>)
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#define TMV_GPU_BUFFER_POOL_HAS_VULKAN 1
#endif
#endif

#if defined(TMV_GPU_BUFFER_POOL_HAS_VULKAN)

#include <cstdint>
#include <vector>

namespace tmv_vulkan
{

struct GpuBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
};

/**
 * @brief Pool of staging+device GPU buffer pairs.
 *
 * Usage:
 *   int slot = pool.acquire(field_bytes);
 *   // use pool.staging(slot) and pool.device(slot) for dispatch
 *   pool.release(slot);
 *
 * The pool does NOT own Vulkan resource creation/destruction directly.
 * The owning backend must call set_allocator() before first acquire,
 * and destroy_all() at shutdown.
 */
class GpuBufferPool
{
public:
    /// Function type for creating a GpuBuffer (staging or device).
    /// Returns true on success, filling out_buf.
    using CreateBufferFn = bool (*)(void* user_data, VkDeviceSize size,
                                    bool is_staging, GpuBuffer& out_buf);

    /// Function type for destroying a GpuBuffer.
    using DestroyBufferFn = void (*)(void* user_data, GpuBuffer& buf);

    /**
     * @brief Registers allocator callbacks. Must be called before acquire().
     * @param create_fn  Creates a buffer (staging or device).
     * @param destroy_fn Destroys a buffer.
     * @param user_data  Opaque pointer passed to callbacks (typically the backend).
     */
    void set_allocator(CreateBufferFn create_fn, DestroyBufferFn destroy_fn,
                       void* user_data)
    {
        create_fn_ = create_fn;
        destroy_fn_ = destroy_fn;
        user_data_ = user_data;
    }

    /**
     * @brief Acquires a buffer slot with at least min_size bytes.
     *
     * If a free slot with sufficient capacity exists, it is reused.
     * Otherwise a new slot is allocated (growing the pool).
     * If the best free slot is too small, it is destroyed and reallocated.
     *
     * @param min_size Minimum byte size for both staging and device buffers.
     * @return Slot index (>= 0) on success, -1 on failure.
     */
    int acquire(VkDeviceSize min_size)
    {
        if (create_fn_ == nullptr || destroy_fn_ == nullptr)
        {
            return -1;
        }

        // First pass: find a free slot with sufficient capacity
        for (std::size_t i = 0; i < slots_.size(); ++i)
        {
            if (!slots_[i].in_use && slots_[i].capacity >= min_size)
            {
                slots_[i].in_use = true;
                return static_cast<int>(i);
            }
        }

        // Second pass: find any free slot (will need reallocation)
        for (std::size_t i = 0; i < slots_.size(); ++i)
        {
            if (!slots_[i].in_use)
            {
                // Destroy old buffers and reallocate
                destroy_fn_(user_data_, slots_[i].staging);
                destroy_fn_(user_data_, slots_[i].device);
                slots_[i].staging = {};
                slots_[i].device = {};
                slots_[i].capacity = 0;

                if (!create_fn_(user_data_, min_size, true, slots_[i].staging) ||
                    !create_fn_(user_data_, min_size, false, slots_[i].device))
                {
                    destroy_fn_(user_data_, slots_[i].staging);
                    destroy_fn_(user_data_, slots_[i].device);
                    slots_[i].staging = {};
                    slots_[i].device = {};
                    return -1;
                }
                slots_[i].capacity = min_size;
                slots_[i].in_use = true;
                return static_cast<int>(i);
            }
        }

        // No free slots — grow the pool
        Slot new_slot{};
        if (!create_fn_(user_data_, min_size, true, new_slot.staging) ||
            !create_fn_(user_data_, min_size, false, new_slot.device))
        {
            destroy_fn_(user_data_, new_slot.staging);
            destroy_fn_(user_data_, new_slot.device);
            return -1;
        }
        new_slot.capacity = min_size;
        new_slot.in_use = true;
        slots_.push_back(new_slot);
        return static_cast<int>(slots_.size() - 1);
    }

    /**
     * @brief Releases a slot back to the pool for reuse.
     * @param slot_index Index returned by acquire().
     */
    void release(int slot_index)
    {
        if (slot_index >= 0 && static_cast<std::size_t>(slot_index) < slots_.size())
        {
            slots_[static_cast<std::size_t>(slot_index)].in_use = false;
        }
    }

    /** @brief Releases all slots (marks as free, does not destroy). */
    void release_all()
    {
        for (auto& slot : slots_)
        {
            slot.in_use = false;
        }
    }

    GpuBuffer& staging(int slot_index)
    {
        return slots_[static_cast<std::size_t>(slot_index)].staging;
    }

    GpuBuffer& device(int slot_index)
    {
        return slots_[static_cast<std::size_t>(slot_index)].device;
    }

    const GpuBuffer& staging(int slot_index) const
    {
        return slots_[static_cast<std::size_t>(slot_index)].staging;
    }

    const GpuBuffer& device(int slot_index) const
    {
        return slots_[static_cast<std::size_t>(slot_index)].device;
    }

    std::size_t slot_count() const { return slots_.size(); }
    std::size_t active_count() const
    {
        std::size_t count = 0;
        for (const auto& s : slots_) { if (s.in_use) ++count; }
        return count;
    }

    /**
     * @brief Destroys all buffer resources and clears the pool.
     * Must be called at shutdown.
     */
    void destroy_all()
    {
        if (destroy_fn_ == nullptr) return;
        for (auto& slot : slots_)
        {
            destroy_fn_(user_data_, slot.staging);
            destroy_fn_(user_data_, slot.device);
        }
        slots_.clear();
    }

private:
    struct Slot
    {
        GpuBuffer staging;
        GpuBuffer device;
        VkDeviceSize capacity = 0;
        bool in_use = false;
    };

    std::vector<Slot> slots_;
    CreateBufferFn create_fn_ = nullptr;
    DestroyBufferFn destroy_fn_ = nullptr;
    void* user_data_ = nullptr;
};

} // namespace tmv_vulkan

#endif // TMV_GPU_BUFFER_POOL_HAS_VULKAN
