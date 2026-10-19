#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#include <malloc.h>
#endif

/**
 * @file field3d.hpp
 * @brief Cache-friendly contiguous 3D float field container.
 *
 * Provides row-major storage with both `(i,j,k)` and `[i][j][k]` access.
 * Includes size checking for allocations and overflow-safe dimension math.
 * Used broadly across dynamics and physics modules.
 */

/**
 * @brief Allocator giving Field3D storage a page-aligned base address AND
 *        an allocation size rounded up to the same alignment.
 *
 * Both properties are requirements for importing field memory into Vulkan
 * via VK_EXT_external_memory_host (GPU.md Phase 4.6 zero-copy:
 * minImportedHostPointerAlignment is 16 KB on Apple Silicon and typically
 * 4 KB on discrete drivers - 16 KB satisfies both). CPU code sees no
 * behavioral change; the alignment only widens the allocation (at most
 * 16 KB of slack per field).
 */
template <typename T>
struct PageAlignedAllocator
{
    using value_type = T;

    /// 16 KB: Apple Silicon's minImportedHostPointerAlignment, a multiple
    /// of the 4 KB seen on discrete drivers.
    static constexpr std::size_t kAlignmentBytes = 16384;

    PageAlignedAllocator() noexcept = default;
    template <typename U>
    PageAlignedAllocator(const PageAlignedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n)
    {
        if (n > (std::numeric_limits<std::size_t>::max() - (kAlignmentBytes - 1)) / sizeof(T))
        {
            throw std::bad_array_new_length();
        }
        const std::size_t raw = n * sizeof(T);
        const std::size_t bytes =
            ((raw + kAlignmentBytes - 1) / kAlignmentBytes) * kAlignmentBytes;
#ifdef _WIN32
        void* p = _aligned_malloc(bytes > 0 ? bytes : kAlignmentBytes, kAlignmentBytes);
#else
        void* p = std::aligned_alloc(kAlignmentBytes, bytes > 0 ? bytes : kAlignmentBytes);
#endif
        if (p == nullptr)
        {
            throw std::bad_alloc();
        }
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t) noexcept
    {
#ifdef _WIN32
        _aligned_free(p);
#else
        std::free(p);
#endif
    }

    template <typename U>
    bool operator==(const PageAlignedAllocator<U>&) const noexcept
    {
        return true;
    }
    template <typename U>
    bool operator!=(const PageAlignedAllocator<U>&) const noexcept
    {
        return false;
    }
};

class Field3D
{
public:
    /**
     * @brief Constructs an empty field.
     */
    Field3D() : NR_(0), NTH_(0), NZ_(0) {}

    /**
     * @brief Constructs a zero-initialized field.
     * @param nr Radial dimension.
     * @param nth Azimuthal dimension.
     * @param nz Vertical dimension.
     */
    Field3D(int nr, int nth, int nz) : NR_(nr), NTH_(nth), NZ_(nz)
    {
        data_.resize(checked_size(nr, nth, nz), 0.0f);
    }

    /**
     * @brief Constructs a field initialized with a constant value.
     * @param nr Radial dimension.
     * @param nth Azimuthal dimension.
     * @param nz Vertical dimension.
     * @param init_value Fill value.
     */
    Field3D(int nr, int nth, int nz, float init_value) : NR_(nr), NTH_(nth), NZ_(nz)
    {
        data_.resize(checked_size(nr, nth, nz), init_value);
    }

    /**
     * @brief Copy constructor.
     */
    Field3D(const Field3D& other) : NR_(other.NR_), NTH_(other.NTH_), NZ_(other.NZ_), data_(other.data_) {}

    /**
     * @brief Move constructor.
     */
    Field3D(Field3D&& other) noexcept
        : NR_(other.NR_), NTH_(other.NTH_), NZ_(other.NZ_), data_(std::move(other.data_))
    {
        other.NR_ = other.NTH_ = other.NZ_ = 0;
    }

    /**
     * @brief Copy assignment.
     */
    Field3D& operator=(const Field3D& other)
    {
        if (this != &other)
        {
            if (other.data_.size() > data_.capacity())
            {
                decltype(data_) candidate(other.data_);
                data_.swap(candidate);
            }
            else
                data_ = other.data_;
            NR_ = other.NR_;
            NTH_ = other.NTH_;
            NZ_ = other.NZ_;
        }
        return *this;
    }

    /**
     * @brief Move assignment.
     */
    Field3D& operator=(Field3D&& other) noexcept
    {
        if (this != &other)
        {
            NR_ = other.NR_;
            NTH_ = other.NTH_;
            NZ_ = other.NZ_;
            data_ = std::move(other.data_);
            other.NR_ = other.NTH_ = other.NZ_ = 0;
        }
        return *this;
    }

    /**
     * @brief Resizes and zero-initializes field storage.
     * @param nr Radial dimension.
     * @param nth Azimuthal dimension.
     * @param nz Vertical dimension.
     */
    void resize(int nr, int nth, int nz)
    {
        const size_t new_size = checked_size(nr, nth, nz);
        data_.resize(new_size, 0.0f);
        NR_ = nr;
        NTH_ = nth;
        NZ_ = nz;
    }

    /**
     * @brief Resizes and fills field storage with a constant value.
     * @param nr Radial dimension.
     * @param nth Azimuthal dimension.
     * @param nz Vertical dimension.
     * @param init_value Fill value.
     */
    void resize(int nr, int nth, int nz, float init_value)
    {
        const size_t new_size = checked_size(nr, nth, nz);
        if (new_size > data_.capacity())
        {
            decltype(data_) candidate(new_size, init_value);
            data_.swap(candidate);
        }
        else if (data_.size() != new_size)
        {
            data_.assign(new_size, init_value);
        }
        else
        {
            std::fill(data_.begin(), data_.end(), init_value);
        }
        NR_ = nr;
        NTH_ = nth;
        NZ_ = nz;
    }

    /**
     * @brief Releases all owned storage and returns the field to its empty state.
     *
     * Unlike resizing to a zero-sized shape, this explicitly drops retained
     * capacity. Runtime-owned lazy fields use it when a new problem starts so
     * a diagnostic volume from an earlier run cannot remain resident.
     */
    void release_storage() noexcept
    {
        decltype(data_) empty;
        data_.swap(empty);
        NR_ = 0;
        NTH_ = 0;
        NZ_ = 0;
    }

    /**
     * @brief Fills all elements with a constant value.
     * @param value Fill value.
     */
    void fill(float value) { std::fill(data_.begin(), data_.end(), value); }

    /**
     * @brief Assigns data from a nested vector representation.
     * @param nr Radial dimension.
     * @param nth Azimuthal dimension.
     * @param nz Vertical dimension.
     * @param nested Input nested vector data.
     */
    void assign(int nr, int nth, int nz, const std::vector<std::vector<std::vector<float>>>& nested)
    {
        if (static_cast<int>(nested.size()) != nr)
        {
            throw std::invalid_argument("Field3D::assign nested size mismatch along r dimension");
        }

        for (int i = 0; i < nr; ++i)
        {
            if (static_cast<int>(nested[i].size()) != nth)
            {
                throw std::invalid_argument("Field3D::assign nested size mismatch along theta dimension");
            }

            for (int j = 0; j < nth; ++j)
            {
                if (static_cast<int>(nested[i][j].size()) != nz)
                {
                    throw std::invalid_argument("Field3D::assign nested size mismatch along z dimension");
                }
            }
        }

        resize(nr, nth, nz);

        for (int i = 0; i < nr; ++i)
        {
            for (int j = 0; j < nth; ++j)
            {
                for (int k = 0; k < nz; ++k)
                {
                    (*this)[i][j][k] = nested[i][j][k];
                }
            }
        }
    }

    /**
     * @brief Returns radial dimension.
     */
    int size_r() const { return NR_; }

    /**
     * @brief Returns azimuthal dimension.
     */
    int size_th() const { return NTH_; }

    /**
     * @brief Returns vertical dimension.
     */
    int size_z() const { return NZ_; }

    /**
     * @brief Returns total flattened element count.
     */
    size_t size() const { return data_.size(); }

    /**
     * @brief Returns retained flattened-element capacity.
     *
     * This is primarily useful for allocation-ownership qualification; normal
     * numerical code should reason from size and shape.
     */
    size_t capacity() const { return data_.capacity(); }

    /**
     * @brief Reports whether storage is empty.
     */
    bool empty() const { return data_.empty(); }

    /**
     * @brief Returns mutable pointer to contiguous storage.
     */
    float* data() { return data_.data(); }

    /**
     * @brief Returns const pointer to contiguous storage.
     */
    const float* data() const { return data_.data(); }

    /**
     * @brief Mutable element access using `(i,j,k)` indexing.
     */
    float& operator()(int i, int j, int k) { return data_[flatten_index(i, j, k)]; }

    /**
     * @brief Const element access using `(i,j,k)` indexing.
     */
    const float& operator()(int i, int j, int k) const { return data_[flatten_index(i, j, k)]; }

private:
    class Slice2D
    {
        Field3D* parent_;
        int i_;

    public:
        Slice2D(Field3D* parent, int i) : parent_(parent), i_(i) {}

        class Slice1D
        {
            Field3D* parent_;
            int i_;
            int j_;

        public:
            Slice1D(Field3D* parent, int i, int j) : parent_(parent), i_(i), j_(j) {}

            class Slice0D
            {
                Field3D* parent_;
                int i_;
                int j_;
                int k_;

            public:
                Slice0D(Field3D* parent, int i, int j, int k)
                    : parent_(parent), i_(i), j_(j), k_(k)
                {
                }

                operator float&() { return (*parent_)(i_, j_, k_); }

                float& operator=(float value) { return (*parent_)(i_, j_, k_) = value; }

                // CRITICAL: Without an explicit copy-assignment operator that
                // copies the *underlying float value*, the compiler synthesizes
                // a default copy assignment that copies the proxy's parent_/i_/
                // j_/k_ members and silently ignores the value. That makes
                // expressions like `field[a][b][c] = field[d][e][f]` no-ops
                // throughout the codebase. This was the root cause of the
                // "boundary conditions never apply" bug — see Journey.md
                // Phase 2 "Bug 5: Field3D proxy silent no-op assignment".
                Slice0D& operator=(const Slice0D& other)
                {
                    (*parent_)(i_, j_, k_) = static_cast<float>(other);
                    return *this;
                }

                template <typename ConstSlice>
                Slice0D& operator=(const ConstSlice& other)
                {
                    (*parent_)(i_, j_, k_) = static_cast<float>(other);
                    return *this;
                }

                float& operator+=(float value) { return (*parent_)(i_, j_, k_) += value; }

                float& operator+=(double value) { return (*parent_)(i_, j_, k_) += static_cast<float>(value); }

                float& operator-=(float value) { return (*parent_)(i_, j_, k_) -= value; }

                float& operator-=(double value) { return (*parent_)(i_, j_, k_) -= static_cast<float>(value); }

                float& operator*=(float value) { return (*parent_)(i_, j_, k_) *= value; }

                float& operator/=(float value) { return (*parent_)(i_, j_, k_) /= value; }

                bool operator==(float value) const { return (*parent_)(i_, j_, k_) == value; }

                bool operator!=(float value) const { return (*parent_)(i_, j_, k_) != value; }

                operator float() const { return (*parent_)(i_, j_, k_); }
            };

            Slice0D operator[](int k) { return Slice0D(parent_, i_, j_, k); }

            class ConstSlice0D
            {
                const Field3D* parent_;
                int i_;
                int j_;
                int k_;

            public:
                ConstSlice0D(const Field3D* parent, int i, int j, int k)
                    : parent_(parent), i_(i), j_(j), k_(k)
                {
                }

                operator float() const { return (*parent_)(i_, j_, k_); }

                double to_double() const { return static_cast<double>((*parent_)(i_, j_, k_)); }
            };

            ConstSlice0D operator[](int k) const { return ConstSlice0D(parent_, i_, j_, k); }
        };

        Slice1D operator[](int j) { return Slice1D(parent_, i_, j); }

        class ConstSlice1D
        {
            const Field3D* parent_;
            int i_;
            int j_;

        public:
            ConstSlice1D(const Field3D* parent, int i, int j) : parent_(parent), i_(i), j_(j) {}

            class ConstSlice0D
            {
                const Field3D* parent_;
                int i_;
                int j_;
                int k_;

            public:
                ConstSlice0D(const Field3D* parent, int i, int j, int k)
                    : parent_(parent), i_(i), j_(j), k_(k)
                {
                }

                operator float() const { return (*parent_)(i_, j_, k_); }

                double to_double() const { return static_cast<double>((*parent_)(i_, j_, k_)); }
            };

            ConstSlice0D operator[](int k) const { return ConstSlice0D(parent_, i_, j_, k); }
        };

        ConstSlice1D operator[](int j) const { return ConstSlice1D(parent_, i_, j); }
    };

    class ConstSlice2D
    {
        const Field3D* parent_;
        int i_;

    public:
        ConstSlice2D(const Field3D* parent, int i) : parent_(parent), i_(i) {}

        class ConstSlice1D
        {
            const Field3D* parent_;
            int i_;
            int j_;

        public:
            ConstSlice1D(const Field3D* parent, int i, int j) : parent_(parent), i_(i), j_(j) {}

            class ConstSlice0D
            {
                const Field3D* parent_;
                int i_;
                int j_;
                int k_;

            public:
                ConstSlice0D(const Field3D* parent, int i, int j, int k)
                    : parent_(parent), i_(i), j_(j), k_(k)
                {
                }

                operator float() const { return (*parent_)(i_, j_, k_); }

                double to_double() const { return static_cast<double>((*parent_)(i_, j_, k_)); }
            };

            ConstSlice0D operator[](int k) const { return ConstSlice0D(parent_, i_, j_, k); }
        };

        ConstSlice1D operator[](int j) const { return ConstSlice1D(parent_, i_, j); }
    };

public:
    /**
     * @brief Provides mutable chained indexing `[i][j][k]`.
     */
    Slice2D operator[](int i) { return Slice2D(this, i); }

    /**
     * @brief Provides const chained indexing `[i][j][k]`.
     */
    ConstSlice2D operator[](int i) const { return ConstSlice2D(this, i); }

    /**
     * @brief Returns a mutable pointer to the start of a contiguous k-column.
     *
     * Enables SIMD and auto-vectorized inner loops by bypassing the proxy
     * object chain of operator[]. The returned pointer addresses NZ_
     * contiguous floats: row_ptr(i,j)[0] through row_ptr(i,j)[NZ_-1].
     */
    float* row_ptr(int i, int j)
    {
        return &data_[static_cast<size_t>(i) * static_cast<size_t>(NTH_) * static_cast<size_t>(NZ_) +
                       static_cast<size_t>(j) * static_cast<size_t>(NZ_)];
    }

    /**
     * @brief Returns a const pointer to the start of a contiguous k-column.
     */
    const float* row_ptr(int i, int j) const
    {
        return &data_[static_cast<size_t>(i) * static_cast<size_t>(NTH_) * static_cast<size_t>(NZ_) +
                       static_cast<size_t>(j) * static_cast<size_t>(NZ_)];
    }

private:
    size_t flatten_index(int i, int j, int k) const
    {
        assert(i >= 0 && i < NR_ && j >= 0 && j < NTH_ && k >= 0 && k < NZ_);
        // Row-major: idx = i*NTH*NZ + j*NZ + k.
        return static_cast<size_t>(i) * static_cast<size_t>(NTH_) * static_cast<size_t>(NZ_) +
               static_cast<size_t>(j) * static_cast<size_t>(NZ_) +
               static_cast<size_t>(k);
    }

    static size_t checked_size(int nr, int nth, int nz)
    {
        if (nr < 0 || nth < 0 || nz < 0)
        {
            throw std::invalid_argument("Field3D dimensions must be non-negative");
        }

        const size_t nr_sz = static_cast<size_t>(nr);
        const size_t nth_sz = static_cast<size_t>(nth);
        const size_t nz_sz = static_cast<size_t>(nz);

        if (nr_sz != 0 && nth_sz > std::numeric_limits<size_t>::max() / nr_sz)
        {
            throw std::overflow_error("Field3D size overflow on nr*nth");
        }

        const size_t nr_nth = nr_sz * nth_sz;
        if (nr_nth != 0 && nz_sz > std::numeric_limits<size_t>::max() / nr_nth)
        {
            throw std::overflow_error("Field3D size overflow on nr*nth*nz");
        }

        return nr_nth * nz_sz;
    }

    int NR_;
    int NTH_;
    int NZ_;
    std::vector<float, PageAlignedAllocator<float>> data_;
};
