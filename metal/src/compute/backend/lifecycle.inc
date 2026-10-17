// Device selection, catalog-driven runtime MSL compilation, and teardown.
MetalComputeBackend::MetalComputeBackend() : impl_(std::make_unique<Impl>())
{
}

MetalComputeBackend::~MetalComputeBackend()
{
    shutdown();
}

std::string MetalComputeBackend::name() const
{
    return "metal";
}

void MetalComputeBackend::configure(const ComputeBackendConfig& config)
{
    impl_->hardware.configured_device_index = config.device_index;
}

bool MetalComputeBackend::initialize(std::string& error)
{
    @autoreleasepool
    {
        shutdown();
        error.clear();

        if (impl_->hardware.configured_device_index < -1)
        {
            error = "Metal device_index must be -1 or a non-negative device ordinal";
            return false;
        }
        if (impl_->hardware.configured_device_index == -1)
        {
            impl_->hardware.device = MTLCreateSystemDefaultDevice();
        }
        else
        {
            NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
            const NSUInteger index = static_cast<NSUInteger>(impl_->hardware.configured_device_index);
            if (index >= devices.count)
            {
                std::ostringstream message;
                message << "Metal device_index " << impl_->hardware.configured_device_index
                        << " is out of range for " << devices.count << " visible device(s)";
                error = message.str();
                return false;
            }
            impl_->hardware.device = devices[index];
        }
        if (impl_->hardware.device == nil)
        {
            error = "Metal system default device is unavailable";
            return false;
        }

        impl_->hardware.name             = ns_string(impl_->hardware.device.name);
        impl_->hardware.operating_system = ns_string(NSProcessInfo.processInfo.operatingSystemVersionString);
        impl_->hardware.unified_memory    = impl_->hardware.device.hasUnifiedMemory;
        impl_->telemetry.timestamp_support = false;

        impl_->hardware.queue = [impl_->hardware.device newCommandQueue];
        if (impl_->hardware.queue == nil)
        {
            error = "Metal command queue creation failed";
            shutdown();
            return false;
        }

        for (const detail::MetalLibrarySpec& library : detail::metal_library_catalog())
        {
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            if (library.precise_math)
            {
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
                if (@available(macOS 15.0, *))
                {
                    options.mathMode                   = MTLMathModeSafe;
                    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
                }
                else
#endif
                {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                    options.fastMathEnabled = NO;
#pragma clang diagnostic pop
                }
            }
            if (!impl_->pipelines.compile_library(impl_->hardware.device,
                                                  options,
                                                  library.label,
                                                  library.source,
                                                  library.source_hash.c_str(),
                                                  library.kernels.data(),
                                                  library.kernel_count,
                                                  error))
            {
                shutdown();
                return false;
            }
        }
        for (const detail::MetalPipelineAlias& alias : detail::metal_pipeline_aliases())
        {
            if (!impl_->pipelines.alias(alias.alias, alias.target))
            {
                error = "Metal pipeline alias creation failed";
                shutdown();
                return false;
            }
        }

        impl_->shared_allocations.initialize();
        impl_->resident.bind(impl_->hardware.device);
        return true;
    }
}

void MetalComputeBackend::shutdown()
{
    @autoreleasepool
    {
        const char* timing = std::getenv("NIMBUS_GPU_KERNEL_TIMING");
        const auto& stats = impl_->telemetry;
        if (timing != nullptr && timing[0] == '1' && stats.completed_command_buffers > 0)
            std::cout << "[METAL TELEMETRY] scope=backend_generation command_buffers="
                      << stats.completed_command_buffers
                      << " timestamped_command_buffers=" << stats.timestamped_command_buffers
                      << " gpu_total_ms=" << stats.total_device_ms
                      << " submit_wait_wall_ms=" << stats.total_submit_wait_ms
                      << " resident_uploads=" << stats.resident_uploads
                      << " resident_reuses=" << stats.resident_reuses << '\n';
        impl_->clear_generation();
    }
}

void MetalComputeBackend::populate_hardware_info(HardwareInfo& info) const
{
    if (!available())
        return;
    info.gpu_available   = true;
    info.gpu_device_name = impl_->hardware.name;
}

ComputeMemoryProperties MetalComputeBackend::memory_properties() const
{
    ComputeMemoryProperties result;
    if (!available())
        return result;
    result.accelerator                = true;
    result.unified_host_device_memory = impl_->hardware.unified_memory;
    result.max_storage_buffer_bytes   = static_cast<std::uint64_t>(impl_->max_buffer_bytes());
    result.device_local_heap_bytes    = static_cast<std::uint64_t>(impl_->hardware.device.recommendedMaxWorkingSetSize);
    return result;
}

std::uint64_t MetalComputeBackend::successful_dispatch_count() const
{
    return impl_->telemetry.completed_command_buffers;
}

std::uint64_t MetalComputeBackend::successful_kernel_dispatch_count(nimbus::compute::KernelId kernel) const
{
    return impl_->telemetry.kernel_dispatches(kernel);
}
