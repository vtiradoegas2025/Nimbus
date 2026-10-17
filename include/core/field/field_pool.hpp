#pragma once

/**
 * @file field_pool.hpp
 * @brief Thread-safe pool of reusable Field3D buffers.
 *
 * Eliminates per-timestep heap allocations for temporary fields.
 * Callers acquire a buffer (either recycled or freshly allocated),
 * use it, and release it back to the pool when done. The pool
 * keeps released buffers alive so the next acquire reuses the
 * existing allocation instead of hitting the heap.
 *
 * Usage:
 *   auto& pool = FieldPool::instance();
 *   Field3D buf = pool.acquire(NR, NTH, NZ);
 *   // ... use buf ...
 *   pool.release(std::move(buf));
 *
 * Or use the RAII guard:
 *   auto guard = pool.scoped_acquire(NR, NTH, NZ);
 *   Field3D& buf = guard.field;
 *   // ... buf is released when guard goes out of scope ...
 */

#include "core/field/field3d.hpp"

#include <atomic>
#include <cstddef>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

class FieldPool
{
    struct Shape
    {
        int nr = 0;
        int nth = 0;
        int nz = 0;

        bool operator==(const Shape& other) const noexcept
        {
            return nr == other.nr && nth == other.nth && nz == other.nz;
        }
    };

    struct ShapeHash
    {
        std::size_t operator()(const Shape& shape) const noexcept
        {
            std::size_t seed = static_cast<std::size_t>(shape.nr);
            seed ^= static_cast<std::size_t>(shape.nth)
                + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
            seed ^= static_cast<std::size_t>(shape.nz)
                + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
            return seed;
        }
    };

    using Buckets = std::unordered_map<Shape, std::vector<Field3D>, ShapeHash>;

public:
    /// Returns the global singleton pool.
    static FieldPool& instance()
    {
        static FieldPool pool;
        return pool;
    }

    /**
     * @brief Acquire a field buffer with the given dimensions.
     *
     * If the pool has a buffer with matching dimensions, it is recycled
     * (zero-filled). Otherwise a new Field3D is allocated.
     */
    Field3D acquire(int nr, int nth, int nz)
    {
        Field3D buf = take_recycled(nr, nth, nz);
        if (buf.empty())
        {
            return Field3D(nr, nth, nz);
        }
        // Full-volume work happens after releasing the pool mutex.
        buf.fill(0.0f);
        return buf;
    }

    /**
     * @brief Acquire storage whose complete contents will be overwritten.
     *
     * Matching recycled storage is returned without the normal zero fill.
     * The caller must write every element before any read. Newly allocated
     * Field3D storage is still value-initialized by the container.
     */
    Field3D acquire_for_overwrite(int nr, int nth, int nz)
    {
        Field3D buf = take_recycled(nr, nth, nz);
        if (buf.empty())
        {
            buf = Field3D(nr, nth, nz);
        }
        if (poison_overwrite_.load(std::memory_order_relaxed))
        {
            buf.fill(std::numeric_limits<float>::quiet_NaN());
        }
        return buf;
    }

    /**
     * @brief Acquire a field initialized as a copy of another field.
     *
     * Attempts to recycle a buffer with matching dimensions and copies
     * the source data into it. Falls back to copy-construction if no
     * matching buffer is available.
     */
    Field3D acquire_copy(const Field3D& src)
    {
        Field3D buf = take_recycled(
            src.size_r(), src.size_th(), src.size_z());
        if (buf.empty())
        {
            return Field3D(src);
        }
        // Copy after releasing the pool mutex so unrelated shapes can be
        // acquired/released concurrently.
        std::memcpy(buf.data(), src.data(), src.size() * sizeof(float));
        return buf;
    }

    /**
     * @brief Return a buffer to the pool for future reuse.
     *
     * The buffer's memory is kept alive. Pass by value (use std::move).
     */
    void release(Field3D buf) noexcept
    {
        if (buf.empty())
        {
            return;
        }
        try
        {
            std::lock_guard<std::mutex> lock(mutex_);
            free_[{buf.size_r(), buf.size_th(), buf.size_z()}].push_back(
                std::move(buf));
            ++free_count_;
        }
        catch (...)
        {
            // Caching is optional, including during exception unwinding.
        }
    }

    /// Number of buffers currently in the free list.
    std::size_t free_count() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return free_count_;
    }

    /// Logical payload bytes retained by currently free buffers.
    std::size_t retained_payload_bytes() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t bytes = 0;
        for (const auto& [shape, bucket] : free_)
        {
            (void)shape;
            for (const Field3D& field : bucket)
            {
                bytes += field.size() * sizeof(float);
            }
        }
        return bytes;
    }

    /// Number of shape buckets whose vector capacity is retained for reuse.
    std::size_t retained_shape_count() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return free_.size();
    }

    /**
     * @brief Enables NaN poisoning for overwrite acquisitions.
     *
     * Strict/validation runs can enable this to expose a caller that reads or
     * publishes a cell it promised to overwrite. Performance runs leave it
     * disabled and avoid the fill entirely.
     */
    void set_overwrite_poisoning(bool enabled) noexcept
    {
        poison_overwrite_.store(enabled, std::memory_order_relaxed);
    }

    bool overwrite_poisoning_enabled() const noexcept
    {
        return poison_overwrite_.load(std::memory_order_relaxed);
    }

    /// Release all pooled memory.
    void clear()
    {
        Buckets released;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released.swap(free_);
            free_count_ = 0;
        }
        // Destroy potentially large allocations after releasing the mutex.
    }

    /**
     * @brief RAII guard that releases a field back to the pool on destruction.
     */
    struct ScopedField
    {
        Field3D field;
        FieldPool& pool;

        ScopedField(Field3D f, FieldPool& p) : field(std::move(f)), pool(p) {}
        ~ScopedField() { pool.release(std::move(field)); }

        // Non-copyable
        ScopedField(const ScopedField&) = delete;
        ScopedField& operator=(const ScopedField&) = delete;

        // Movable
        ScopedField(ScopedField&& other) noexcept
            : field(std::move(other.field)), pool(other.pool) {}
    };

    /// Acquire a scoped field that auto-releases when it goes out of scope.
    ScopedField scoped_acquire(int nr, int nth, int nz)
    {
        return ScopedField(acquire(nr, nth, nz), *this);
    }

    /// Acquire scoped storage that the caller promises to overwrite fully.
    ScopedField scoped_acquire_for_overwrite(int nr, int nth, int nz)
    {
        return ScopedField(acquire_for_overwrite(nr, nth, nz), *this);
    }

    /// Acquire a scoped copy of an existing field.
    ScopedField scoped_acquire_copy(const Field3D& src)
    {
        return ScopedField(acquire_copy(src), *this);
    }

private:
    FieldPool() = default;
    ~FieldPool() = default;

    Field3D take_recycled(int nr, int nth, int nz)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = free_.find({nr, nth, nz});
        if (it == free_.end() || it->second.empty())
        {
            return {};
        }
        Field3D field = std::move(it->second.back());
        it->second.pop_back();
        --free_count_;
        // Keep the empty bucket and its vector capacity. Erasing it here
        // rebuilt the unordered-map node plus the vector's 1/2/4 growth every
        // time a caller checked out all same-shaped fields, defeating the
        // pool's steady-state allocation contract.
        return field;
    }

    mutable std::mutex mutex_;
    Buckets free_;
    std::size_t free_count_ = 0;
    std::atomic<bool> poison_overwrite_{false};
};
