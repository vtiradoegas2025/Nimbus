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

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace tmv_vulkan
{

struct GpuBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize allocation_size = 0;
    VkDeviceSize memory_offset = 0;
    void* mapped = nullptr;
    void* shared_allocation = nullptr;
    VkMemoryPropertyFlags memory_properties = 0;
    bool device_allocation = false;
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

    /// Function type for creating exact-sized staging/device buffer batches.
    /// Each output array has count elements. Implementations must leave every
    /// element safely destroyable when returning false.
    using CreateBufferBatchFn = bool (*)(
        void* user_data, const VkDeviceSize* sizes, std::size_t count,
        GpuBuffer* staging_buffers, GpuBuffer* device_buffers);

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
     * @brief Registers an optional batch allocator.
     *
     * acquire_many() uses this callback for every set of slots that actually
     * needs storage. Backends can therefore suballocate a whole dispatch from
     * one exact-sized memory block without changing slot ownership.
     */
    void set_batch_allocator(CreateBufferBatchFn create_batch_fn)
    {
        create_batch_fn_ = create_batch_fn;
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
        int slot = -1;
        return acquire_many(min_size, 1, &slot) ? slot : -1;
    }

    /**
     * @brief Acquires count slots transactionally.
     *
     * Sufficient free slots are reused first. All remaining replacements and
     * new slots are passed to the batch allocator together, allowing their
     * Vulkan buffers to share one aligned memory allocation. On failure, no
     * slot remains in use.
     */
    bool acquire_many(VkDeviceSize min_size, std::size_t count,
                      int* slot_indices)
    {
        scalar_size_scratch_.assign(count, min_size);
        return acquire_many(
            scalar_size_scratch_.data(), count, slot_indices);
    }

    /**
     * @brief Acquires a mixed-size slot batch transactionally.
     *
     * Slot indices correspond one-for-one with min_sizes. This is used for
     * kernels that mix 3-D fields with compact vertical profiles.
     */
    bool acquire_many(const VkDeviceSize* min_sizes, std::size_t count,
                      int* slot_indices)
    {
        if (create_fn_ == nullptr || destroy_fn_ == nullptr ||
            count == 0 || min_sizes == nullptr || slot_indices == nullptr)
        {
            return false;
        }

        for (std::size_t i = 0; i < count; ++i)
        {
            if (min_sizes[i] == 0) return false;
            slot_indices[i] = -1;
        }

        request_order_scratch_.resize(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            request_order_scratch_[i] = i;
        }
        std::sort(
            request_order_scratch_.begin(), request_order_scratch_.end(),
            [&](std::size_t a, std::size_t b)
            {
                return min_sizes[a] > min_sizes[b];
            });

        std::size_t acquired = 0;
        for (std::size_t request : request_order_scratch_)
        {
            std::size_t best_slot = slots_.size();
            for (std::size_t slot = 0; slot < slots_.size(); ++slot)
            {
                if (!slots_[slot].in_use &&
                    slots_[slot].capacity >= min_sizes[request] &&
                    (best_slot == slots_.size() ||
                     slots_[slot].capacity < slots_[best_slot].capacity))
                {
                    best_slot = slot;
                }
            }
            if (best_slot != slots_.size())
            {
                slots_[best_slot].in_use = true;
                slot_indices[request] = static_cast<int>(best_slot);
                ++acquired;
            }
        }
        if (acquired == count) return true;

        const std::size_t allocation_count = count - acquired;
        allocation_requests_scratch_.clear();
        allocation_requests_scratch_.reserve(allocation_count);
        for (std::size_t request = 0; request < count; ++request)
        {
            if (slot_indices[request] < 0)
            {
                allocation_requests_scratch_.push_back(request);
            }
        }
        replacement_indices_scratch_.clear();
        replacement_indices_scratch_.reserve(allocation_count);
        for (std::size_t i = 0;
             i < slots_.size() &&
                 replacement_indices_scratch_.size() < allocation_count;
             ++i)
        {
            if (!slots_[i].in_use)
            {
                destroy_fn_(user_data_, slots_[i].staging);
                destroy_fn_(user_data_, slots_[i].device);
                slots_[i] = {};
                replacement_indices_scratch_.push_back(static_cast<int>(i));
            }
        }

        allocation_sizes_scratch_.resize(allocation_count);
        for (std::size_t i = 0; i < allocation_count; ++i)
        {
            allocation_sizes_scratch_[i] =
                min_sizes[allocation_requests_scratch_[i]];
        }
        staging_scratch_.assign(allocation_count, {});
        device_scratch_.assign(allocation_count, {});
        bool created = false;
        if (create_batch_fn_ != nullptr)
        {
            created = create_batch_fn_(
                user_data_, allocation_sizes_scratch_.data(), allocation_count,
                staging_scratch_.data(), device_scratch_.data());
        }
        else
        {
            created = true;
            for (std::size_t i = 0; i < allocation_count; ++i)
            {
                if (!create_fn_(
                        user_data_, allocation_sizes_scratch_[i], true,
                        staging_scratch_[i]) ||
                    !create_fn_(
                        user_data_, allocation_sizes_scratch_[i], false,
                        device_scratch_[i]))
                {
                    created = false;
                    break;
                }
            }
        }
        if (!created)
        {
            for (std::size_t i = 0; i < allocation_count; ++i)
            {
                destroy_fn_(user_data_, staging_scratch_[i]);
                destroy_fn_(user_data_, device_scratch_[i]);
            }
            for (std::size_t request = 0; request < count; ++request)
            {
                if (slot_indices[request] >= 0)
                {
                    slots_[static_cast<std::size_t>(
                        slot_indices[request])].in_use = false;
                    slot_indices[request] = -1;
                }
            }
            return false;
        }

        for (std::size_t i = 0; i < allocation_count; ++i)
        {
            Slot slot{};
            slot.staging = staging_scratch_[i];
            slot.device = device_scratch_[i];
            slot.capacity = allocation_sizes_scratch_[i];
            slot.in_use = true;
            int assigned_index = -1;
            if (i < replacement_indices_scratch_.size())
            {
                assigned_index = replacement_indices_scratch_[i];
                slots_[static_cast<std::size_t>(assigned_index)] = slot;
            }
            else
            {
                slots_.push_back(slot);
                assigned_index = static_cast<int>(slots_.size() - 1);
            }
            slot_indices[allocation_requests_scratch_[i]] = assigned_index;
            staging_scratch_[i] = {};
            device_scratch_[i] = {};
        }
        return true;
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
        staging_scratch_.clear();
        device_scratch_.clear();
    }

    /**
     * @brief Forgets handles without invoking Vulkan destruction callbacks.
     *
     * Use only when the owning VkDevice is being destroyed as the terminal
     * cleanup boundary (for example after VK_ERROR_DEVICE_LOST).  Calling
     * child destroy functions in that state can itself be invalid or unsafe.
     */
    void abandon_all() noexcept
    {
        slots_.clear();
        staging_scratch_.clear();
        device_scratch_.clear();
        create_fn_ = nullptr;
        create_batch_fn_ = nullptr;
        destroy_fn_ = nullptr;
        user_data_ = nullptr;
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
    std::vector<VkDeviceSize> scalar_size_scratch_;
    std::vector<std::size_t> request_order_scratch_;
    std::vector<std::size_t> allocation_requests_scratch_;
    std::vector<int> replacement_indices_scratch_;
    std::vector<VkDeviceSize> allocation_sizes_scratch_;
    std::vector<GpuBuffer> staging_scratch_;
    std::vector<GpuBuffer> device_scratch_;
    CreateBufferFn create_fn_ = nullptr;
    CreateBufferBatchFn create_batch_fn_ = nullptr;
    DestroyBufferFn destroy_fn_ = nullptr;
    void* user_data_ = nullptr;
};

} // namespace tmv_vulkan

#endif // TMV_GPU_BUFFER_POOL_HAS_VULKAN
