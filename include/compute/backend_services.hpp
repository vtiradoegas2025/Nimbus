#pragma once

/**
 * @file backend_services.hpp
 * @brief Narrow service contracts implemented separately by each GPU API.
 */

#include "compute/dispatch_graph.hpp"

#include <cstddef>
#include <cstdint>

namespace nimbus::compute
{

using BackendAllocation = std::uint64_t;
using BackendPipeline = std::uint64_t;

class DeviceContext
{
public:
    virtual ~DeviceContext() = default;
    virtual const char* api_name() const noexcept = 0;
    virtual const char* device_name() const noexcept = 0;
    virtual bool available() const noexcept = 0;
};

class BufferAllocator
{
public:
    virtual ~BufferAllocator() = default;
    virtual BackendAllocation allocate(std::size_t bytes) = 0;
    virtual void release(BackendAllocation allocation) = 0;
};

class PipelineCache
{
public:
    virtual ~PipelineCache() = default;
    virtual BackendPipeline pipeline(KernelId kernel) const noexcept = 0;
    virtual bool ready(KernelId kernel) const noexcept = 0;
};

class CommandSubmission
{
public:
    virtual ~CommandSubmission() = default;
    // DispatchGraph describes ordering hazards only. A non-empty graph is not
    // executable unless the backend has an independent binding for every
    // node's resources, dimensions, and constants; implementations must fail
    // closed when those bindings are absent.
    virtual bool submit(const DispatchGraph& graph) = 0;
};

class TimestampProvider
{
public:
    virtual ~TimestampProvider() = default;
    virtual bool supported() const noexcept = 0;
    virtual double last_device_milliseconds() const noexcept = 0;
};

} // namespace nimbus::compute
