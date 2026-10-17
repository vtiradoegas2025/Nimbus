#pragma once

#import <Metal/Metal.h>

#include "compute/resident_field_store.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nimbus::metal::detail
{

// Adapts the backend-neutral residency state machine to shared Metal buffers.
class MetalResidentState
{
public:
    using Store      = nimbus::compute::ResidentFieldStore;
    using Allocation = Store::Allocation;

    struct Slot
    {
        id<MTLBuffer> current   = nil;
        id<MTLBuffer> candidate = nil;
        std::size_t   bytes     = 0;
    };

    MetalResidentState()
    {
        fields.set_allocator(&MetalResidentState::acquire_one, &MetalResidentState::release_one, this);
        fields.set_batch_allocator(&MetalResidentState::acquire_many);
    }

    void bind(id<MTLDevice> device) noexcept
    {
        device_ = device;
    }

    Slot* slot(Allocation allocation) noexcept
    {
        const auto found = slots_.find(allocation);
        return found == slots_.end() ? nullptr : &found->second;
    }

    std::size_t size() const noexcept
    {
        return slots_.size();
    }

    void clear() noexcept
    {
        fields.clear(true);
        slots_.clear();
        next_   = 1;
        device_ = nil;
    }

    Store fields;

private:
    static Allocation acquire_one(void* user_data, std::size_t bytes)
    {
        auto* self = static_cast<MetalResidentState*>(user_data);
        if (self == nullptr || self->device_ == nil || bytes == 0 ||
            self->next_ == std::numeric_limits<Allocation>::max())
        {
            return Store::invalid_allocation;
        }

        id<MTLBuffer> current   = [self->device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> candidate = [self->device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (current == nil || candidate == nil)
            return Store::invalid_allocation;

        const Allocation allocation = self->next_++;
        self->slots_.emplace(allocation, Slot{current, candidate, bytes});
        return allocation;
    }

    static bool acquire_many(void* user_data, const std::size_t* bytes, std::size_t count, Allocation* allocations)
    {
        auto* self = static_cast<MetalResidentState*>(user_data);
        if (self == nullptr || self->device_ == nil || bytes == nullptr || allocations == nullptr || count == 0 ||
            count > 64 || self->next_ <= 0 ||
            static_cast<std::uint64_t>(self->next_) + count >
                static_cast<std::uint64_t>(std::numeric_limits<Allocation>::max()))
        {
            return false;
        }

        std::vector<Slot> candidates;
        candidates.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
        {
            if (bytes[index] == 0)
                return false;
            id<MTLBuffer> current =
                [self->device_ newBufferWithLength:bytes[index] options:MTLResourceStorageModeShared];
            id<MTLBuffer> candidate =
                [self->device_ newBufferWithLength:bytes[index] options:MTLResourceStorageModeShared];
            if (current == nil || candidate == nil)
                return false;
            candidates.push_back({current, candidate, bytes[index]});
        }

        for (std::size_t index = 0; index < count; ++index)
        {
            const Allocation allocation = self->next_++;
            self->slots_.emplace(allocation, std::move(candidates[index]));
            allocations[index] = allocation;
        }
        return true;
    }

    static void release_one(void* user_data, Allocation allocation)
    {
        auto* self = static_cast<MetalResidentState*>(user_data);
        if (self != nullptr)
            self->slots_.erase(allocation);
    }

    id<MTLDevice>                        device_ = nil;
    std::unordered_map<Allocation, Slot> slots_;
    Allocation                           next_ = 1;
};

} // namespace nimbus::metal::detail
