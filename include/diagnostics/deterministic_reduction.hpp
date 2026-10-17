/**
 * @file deterministic_reduction.hpp
 * @brief Thread-count-independent helpers for scientific diagnostic sums.
 */

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tmv::diagnostics
{

inline bool g_deterministic_reductions = false;

inline void set_deterministic_reductions(bool enabled)
{
    g_deterministic_reductions = enabled;
}

inline bool deterministic_reductions_enabled()
{
    return g_deterministic_reductions;
}

class CompensatedSum
{
public:
    void add(double value)
    {
        const double next = sum_ + value;
        if (std::abs(sum_) >= std::abs(value))
        {
            correction_ += (sum_ - next) + value;
        }
        else
        {
            correction_ += (value - next) + sum_;
        }
        sum_ = next;
    }

    double value() const
    {
        return sum_ + correction_;
    }

private:
    double sum_ = 0.0;
    double correction_ = 0.0;
};

template <std::size_t N>
std::vector<std::array<double, N>>& fixed_slab_scratch()
{
    static thread_local std::vector<std::array<double, N>> scratch;
    return scratch;
}

/**
 * @brief Reduce fixed horizontal slabs independently, then merge by slab index.
 *
 * The callback receives an absolute i index and a zero-initialized accumulator
 * array. It must visit the cells within that slab in a deterministic order.
 * OpenMP may assign slabs to workers in any order without affecting a slab's
 * partial. The final compensated merge always follows increasing i, making the
 * result independent of thread count and OpenMP schedule on one build target.
 */
template <std::size_t N, typename AccumulateSlab>
std::array<double, N> fixed_slab_reduce(int i_begin,
                                        int i_end,
                                        AccumulateSlab accumulate_slab)
{
    const int slab_count = i_end > i_begin ? i_end - i_begin : 0;
    // Each call site retains its horizontal scratch on the invoking thread.
    // Scientific cadence calls these reductions every step; reusing capacity
    // avoids turning deterministic diagnostics into a steady-state allocator.
    auto& partials = fixed_slab_scratch<N>();
    partials.assign(
        static_cast<std::size_t>(slab_count), std::array<double, N>{});

    #pragma omp parallel for schedule(static)
    for (int slab = 0; slab < slab_count; ++slab)
    {
        accumulate_slab(i_begin + slab,
                        partials[static_cast<std::size_t>(slab)]);
    }

    std::array<CompensatedSum, N> merged{};
    for (const auto& partial : partials)
    {
        for (std::size_t n = 0; n < N; ++n)
        {
            merged[n].add(partial[n]);
        }
    }

    std::array<double, N> result{};
    for (std::size_t n = 0; n < N; ++n)
    {
        result[n] = merged[n].value();
    }
    return result;
}

}  // namespace tmv::diagnostics
