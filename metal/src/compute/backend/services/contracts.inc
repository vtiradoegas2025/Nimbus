// Generic backend services, allocation ownership, and residency counters.
const char* MetalComputeBackend::api_name() const noexcept
{
    return "metal";
}

const char* MetalComputeBackend::device_name() const noexcept
{
    return impl_->hardware.name.c_str();
}

bool MetalComputeBackend::available() const noexcept
{
    return impl_->hardware.valid() &&
           impl_->pipelines.ready(nimbus::compute::KernelId::DomainRecyclingHomogeneousFieldShift);
}

nimbus::compute::BackendAllocation MetalComputeBackend::allocate(std::size_t bytes)
{
    @autoreleasepool
    {
        return available() ? impl_->shared_allocations.allocate(impl_->hardware.device, bytes) : 0;
    }
}

void MetalComputeBackend::release(nimbus::compute::BackendAllocation allocation)
{
    @autoreleasepool
    {
        impl_->shared_allocations.release(allocation);
    }
}

nimbus::compute::BackendPipeline MetalComputeBackend::pipeline(nimbus::compute::KernelId kernel) const noexcept
{
    return available() ? impl_->pipelines.handle(kernel) : 0;
}

bool MetalComputeBackend::ready(nimbus::compute::KernelId kernel) const noexcept
{
    return available() && impl_->pipelines.ready(kernel);
}

bool MetalComputeBackend::submit(const nimbus::compute::DispatchGraph& graph)
{
    @autoreleasepool
    {
        if (!available() || !graph.nodes().empty())
            return false;
        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        return impl_->finish(command_buffer);
    }
}

bool MetalComputeBackend::supported() const noexcept
{
    return available() && impl_->telemetry.timestamp_support;
}

double MetalComputeBackend::last_device_milliseconds() const noexcept
{
    return impl_->telemetry.last_device_ms;
}

bool MetalComputeBackend::last_command_buffer_completed() const noexcept
{
    return impl_->telemetry.last_completed;
}

std::uint64_t MetalComputeBackend::completed_command_buffer_count() const noexcept
{
    return impl_->telemetry.completed_command_buffers;
}

std::size_t MetalComputeBackend::live_shared_allocation_count() const noexcept
{
    return impl_->shared_allocations.count();
}

std::size_t MetalComputeBackend::live_shared_allocation_bytes() const noexcept
{
    return impl_->shared_allocations.bytes();
}

std::size_t MetalComputeBackend::resident_field_count() const noexcept
{
    return impl_->resident.size();
}

std::uint64_t MetalComputeBackend::resident_upload_count() const noexcept
{
    return impl_->telemetry.resident_uploads;
}

std::uint64_t MetalComputeBackend::resident_reuse_count() const noexcept
{
    return impl_->telemetry.resident_reuses;
}

std::uint64_t MetalComputeBackend::completed_shift_batch_count() const noexcept
{
    return impl_->telemetry.completed_shift_batches;
}
