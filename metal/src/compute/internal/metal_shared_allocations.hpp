#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace nimbus::metal::detail
{

// Tracks buffers exposed through the generic backend allocation service.
class MetalSharedAllocations
{
public:
    void initialize()
    {
        buffers_ = [[NSMutableDictionary alloc] init];
        sizes_   = [[NSMutableDictionary alloc] init];
    }

    std::uint64_t allocate(id<MTLDevice> device, std::size_t bytes)
    {
        if (device == nil || buffers_ == nil || sizes_ == nil || bytes == 0 ||
            next_ == std::numeric_limits<std::uint64_t>::max())
        {
            return 0;
        }

        id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (buffer == nil)
            return 0;

        const std::uint64_t handle = next_++;
        NSNumber*           key    = [NSNumber numberWithUnsignedLongLong:handle];
        buffers_[key]              = buffer;
        sizes_[key]                = [NSNumber numberWithUnsignedLongLong:bytes];
        live_bytes_ += bytes;
        return handle;
    }

    void release(std::uint64_t allocation) noexcept
    {
        if (allocation == 0 || buffers_ == nil || sizes_ == nil)
            return;

        NSNumber* key  = [NSNumber numberWithUnsignedLongLong:allocation];
        NSNumber* size = sizes_[key];
        if (size == nil)
            return;

        const std::size_t bytes = static_cast<std::size_t>(size.unsignedLongLongValue);
        live_bytes_             = bytes <= live_bytes_ ? live_bytes_ - bytes : 0;
        [buffers_ removeObjectForKey:key];
        [sizes_ removeObjectForKey:key];
    }

    std::size_t count() const noexcept
    {
        return buffers_ == nil ? 0 : static_cast<std::size_t>(buffers_.count);
    }

    std::size_t bytes() const noexcept
    {
        return live_bytes_;
    }

    void clear() noexcept
    {
        [buffers_ removeAllObjects];
        [sizes_ removeAllObjects];
        buffers_    = nil;
        sizes_      = nil;
        next_       = 1;
        live_bytes_ = 0;
    }

private:
    NSMutableDictionary<NSNumber*, id<MTLBuffer>>* buffers_    = nil;
    NSMutableDictionary<NSNumber*, NSNumber*>*     sizes_      = nil;
    std::uint64_t                                  next_       = 1;
    std::size_t                                    live_bytes_ = 0;
};

} // namespace nimbus::metal::detail
