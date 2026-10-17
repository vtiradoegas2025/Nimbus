#pragma once

#import <Metal/Metal.h>

#include "metal_telemetry.hpp"

#include <cstddef>
#include <chrono>
#include <string>

namespace nimbus::metal::detail
{

// Owns the native objects and identity that describe one Metal device session.
class MetalDeviceContext
{
public:
    id<MTLDevice>       device = nil;
    id<MTLCommandQueue> queue  = nil;

    std::string name;
    std::string operating_system;
    bool        unified_memory          = false;
    int         configured_device_index = -1;

    bool valid() const noexcept
    {
        return device != nil && queue != nil;
    }

    std::size_t max_buffer_bytes() const noexcept
    {
        return device == nil ? 0 : static_cast<std::size_t>(device.maxBufferLength);
    }

    id<MTLCommandBuffer> command_buffer() const noexcept
    {
        return queue == nil ? nil : [queue commandBuffer];
    }

    bool finish(id<MTLCommandBuffer> command_buffer, MetalTelemetry& telemetry) const noexcept
    {
        telemetry.last_completed = false;
        if (command_buffer == nil)
            return false;

        const auto submitted = std::chrono::steady_clock::now();
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted)
            return false;

        telemetry.last_completed = true;
        ++telemetry.completed_command_buffers;
        telemetry.total_submit_wait_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - submitted).count();
        telemetry.last_device_ms   = 0.0;
        const NSTimeInterval start = command_buffer.GPUStartTime;
        const NSTimeInterval end   = command_buffer.GPUEndTime;
        if (end >= start && start > 0.0)
        {
            telemetry.timestamp_support = true;
            ++telemetry.timestamped_command_buffers;
            telemetry.last_device_ms    = static_cast<double>((end - start) * 1000.0);
        }
        telemetry.total_device_ms += telemetry.last_device_ms;
        return true;
    }

    void reset() noexcept
    {
        queue  = nil;
        device = nil;
        name.clear();
        operating_system.clear();
        unified_memory = false;
    }
};

} // namespace nimbus::metal::detail
