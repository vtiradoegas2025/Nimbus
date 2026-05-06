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

#include <mutex>
#include <vector>

class FieldPool
{
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
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = free_.begin(); it != free_.end(); ++it)
        {
            if (it->size_r() == nr && it->size_th() == nth && it->size_z() == nz)
            {
                Field3D buf = std::move(*it);
                free_.erase(it);
                buf.fill(0.0f);
                return buf;
            }
        }
        return Field3D(nr, nth, nz);
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
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = free_.begin(); it != free_.end(); ++it)
        {
            if (it->size_r() == src.size_r() &&
                it->size_th() == src.size_th() &&
                it->size_z() == src.size_z())
            {
                Field3D buf = std::move(*it);
                free_.erase(it);
                std::memcpy(buf.data(), src.data(), src.size() * sizeof(float));
                return buf;
            }
        }
        return Field3D(src);
    }

    /**
     * @brief Return a buffer to the pool for future reuse.
     *
     * The buffer's memory is kept alive. Pass by value (use std::move).
     */
    void release(Field3D buf)
    {
        if (buf.empty())
        {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        free_.push_back(std::move(buf));
    }

    /// Number of buffers currently in the free list.
    std::size_t free_count() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return free_.size();
    }

    /// Release all pooled memory.
    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        free_.clear();
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

    /// Acquire a scoped copy of an existing field.
    ScopedField scoped_acquire_copy(const Field3D& src)
    {
        return ScopedField(acquire_copy(src), *this);
    }

private:
    FieldPool() = default;
    ~FieldPool() = default;

    mutable std::mutex mutex_;
    std::vector<Field3D> free_;
};
