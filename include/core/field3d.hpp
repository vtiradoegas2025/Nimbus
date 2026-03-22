#pragma once

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

/**
 * @file field3d.hpp
 * @brief Cache-friendly contiguous 3D float field container.
 *
 * Provides row-major storage with both `(i,j,k)` and `[i][j][k]` access.
 * Includes size checking for allocations and overflow-safe dimension math.
 * Used broadly across dynamics and physics modules.
 */

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
            NR_ = other.NR_;
            NTH_ = other.NTH_;
            NZ_ = other.NZ_;
            data_ = other.data_;
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
        NR_ = nr;
        NTH_ = nth;
        NZ_ = nz;
        data_.resize(new_size, 0.0f);
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
        NR_ = nr;
        NTH_ = nth;
        NZ_ = nz;

        if (data_.size() != new_size)
        {
            data_.assign(new_size, init_value);
        }
        else
        {
            std::fill(data_.begin(), data_.end(), init_value);
        }
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
    std::vector<float> data_;
};
