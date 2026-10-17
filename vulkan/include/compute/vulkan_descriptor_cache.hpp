#pragma once

/**
 * @file vulkan_descriptor_cache.hpp
 * @brief Pure cache key for stable pooled descriptor bindings.
 */

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tmv::vulkan_compute
{

class DescriptorBindingCache
{
public:
    bool matches_or_update(const std::uintptr_t* buffers,
                           const std::size_t* ranges,
                           std::size_t count)
    {
        bool matches = buffers_.size() == count && ranges_.size() == count;
        for (std::size_t i = 0; matches && i < count; ++i)
        {
            matches = buffers_[i] == buffers[i] && ranges_[i] == ranges[i];
        }
        if (matches) return true;

        buffers_.assign(buffers, buffers + count);
        ranges_.assign(ranges, ranges + count);
        return false;
    }

    void clear()
    {
        buffers_.clear();
        ranges_.clear();
    }

private:
    std::vector<std::uintptr_t> buffers_;
    std::vector<std::size_t> ranges_;
};

} // namespace tmv::vulkan_compute
