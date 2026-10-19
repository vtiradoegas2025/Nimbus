#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace tmv
{
struct ByteCopyRange
{
    void* destination;
    const void* source;
};

// Destinations must be disjoint from other ranges, including other sources.
// An exact source/destination identity is allowed and leaves that range alone.
template<std::size_t N>
inline void copy_byte_ranges(const std::array<ByteCopyRange, N>& ranges, std::size_t bytes)
{
    if (bytes == 0 || N == 0)
        return;
#ifdef _OPENMP
    if (bytes >= 8 * 1024 * 1024 && !omp_in_parallel())
    {
        const auto available = static_cast<std::size_t>(omp_get_max_threads());
        const int workers = static_cast<int>(std::min({available, std::size_t(32), bytes / (256 * 1024)}));
        if (workers > 1)
        {
            #pragma omp parallel num_threads(workers)
            {
                const auto team = static_cast<std::size_t>(omp_get_num_threads());
                const auto worker = static_cast<std::size_t>(omp_get_thread_num());
                const auto width = (bytes / team / 64) * 64;
                const auto offset = worker * width;
                const auto count = worker + 1 == team ? bytes - offset : width;
                for (const auto& range : ranges)
                    if (range.destination != range.source)
                        std::memcpy(static_cast<unsigned char*>(range.destination) + offset,
                                    static_cast<const unsigned char*>(range.source) + offset, count);
            }
            return;
        }
    }
#endif
    for (const auto& range : ranges)
        if (range.destination != range.source)
            std::memcpy(range.destination, range.source, bytes);
}

// As with memcpy, the source and destination ranges must not overlap.
inline void copy_bytes(void* destination, const void* source, std::size_t bytes)
{
    if (bytes == 0 || destination == source)
        return;
#ifdef _OPENMP
    if (bytes >= 8 * 1024 * 1024 && !omp_in_parallel())
    {
        const auto available = static_cast<std::size_t>(omp_get_max_threads());
        const int workers = static_cast<int>(std::min({available, std::size_t(32), bytes / (256 * 1024)}));
        if (workers > 1)
        {
            #pragma omp parallel num_threads(workers)
            {
                const auto team = static_cast<std::size_t>(omp_get_num_threads());
                const auto worker = static_cast<std::size_t>(omp_get_thread_num());
                const auto width = (bytes / team / 64) * 64;
                const auto offset = worker * width;
                const auto count = worker + 1 == team ? bytes - offset : width;
                std::memcpy(static_cast<unsigned char*>(destination) + offset,
                            static_cast<const unsigned char*>(source) + offset, count);
            }
            return;
        }
    }
#endif
    std::memcpy(destination, source, bytes);
}
}
