// Metal storage-mode benchmark implementation.
bool MetalComputeBackend::run_domain_shift_storage_benchmark(const MetalStorageBenchmarkConfig& config,
                                                             MetalStorageBenchmarkResult&       result)
{
    @autoreleasepool
    {
        result                  = {};
        std::size_t cells       = 0;
        std::size_t field_bytes = 0;
        if (!supports_homogeneous_domain_field_shift() || config.nr < 2 || config.nth < 2 || config.nz < 1 ||
            config.iterations < 1 || config.iterations > 10000 || config.warmups < 0 || config.warmups > 20 ||
            config.samples < 3 || config.samples > 100 || config.threadgroup_width < 1 ||
            config.threadgroup_width > 1024 || !checked_bytes(config.nr, config.nth, config.nz, cells, field_bytes) ||
            field_bytes > impl_->max_buffer_bytes())
        {
            return false;
        }
        const std::size_t profile_bytes = static_cast<std::size_t>(config.nz) * sizeof(float);
        if (profile_bytes > impl_->max_buffer_bytes())
            return false;

        std::vector<float> source(cells);
        std::vector<float> profile(static_cast<std::size_t>(config.nz));
        std::vector<float> expected(cells);
        for (int k = 0; k < config.nz; ++k)
            profile[static_cast<std::size_t>(k)] = -512.0f - static_cast<float>(k) * 0.5f;
        for (int i = 0; i < config.nr; ++i)
        {
            for (int j = 0; j < config.nth; ++j)
            {
                for (int k = 0; k < config.nz; ++k)
                {
                    const std::size_t index =
                        (static_cast<std::size_t>(i) * config.nth + static_cast<std::size_t>(j)) * config.nz +
                        static_cast<std::size_t>(k);
                    source[index] = static_cast<float>(static_cast<int>((index * 37u) % 4096u) - 2048) * 0.125f;
                }
            }
        }
        for (int i = 0; i < config.nr; ++i)
        {
            for (int j = 0; j < config.nth; ++j)
            {
                for (int k = 0; k < config.nz; ++k)
                {
                    const std::size_t index =
                        (static_cast<std::size_t>(i) * config.nth + static_cast<std::size_t>(j)) * config.nz +
                        static_cast<std::size_t>(k);
                    if (i < config.nr - 1 && j >= 1)
                    {
                        const std::size_t source_index =
                            (static_cast<std::size_t>(i + 1) * config.nth + static_cast<std::size_t>(j - 1)) *
                                config.nz +
                            static_cast<std::size_t>(k);
                        expected[index] = source[source_index];
                    }
                    else
                    {
                        expected[index] = profile[static_cast<std::size_t>(k)];
                    }
                }
            }
        }

        PushConstants constants{};
        constants.nr                     = config.nr;
        constants.nth                    = config.nth;
        constants.nz                     = config.nz;
        constants.shift_i_cells          = 1;
        constants.shift_j_cells          = -1;
        constants.destination_i_begin    = 0;
        constants.destination_i_end      = config.nr - 1;
        constants.destination_j_begin    = 1;
        constants.destination_j_end      = config.nth;
        constants.leading_profile_offset = 0;

        const NSUInteger maximum =
            impl_->pipeline(KernelId::DomainRecyclingHomogeneousFieldShift).maxTotalThreadsPerThreadgroup;
        if (maximum == 0 || static_cast<NSUInteger>(config.threadgroup_width) > maximum)
            return false;
        const MTLSize grid  = MTLSizeMake(static_cast<NSUInteger>(cells), 1, 1);
        const MTLSize group = MTLSizeMake(static_cast<NSUInteger>(config.threadgroup_width), 1, 1);

        struct Measurement
        {
            double             device_ms  = 0.0;
            double             wall_ms    = 0.0;
            std::size_t        heap_bytes = 0;
            std::vector<float> output;
        };

        const auto execute = [&](bool private_heap, Measurement& measurement) -> bool
        {
            id<MTLBuffer> source_buffer       = nil;
            id<MTLBuffer> profile_buffer      = nil;
            id<MTLBuffer> destination_buffer  = nil;
            id<MTLBuffer> source_staging      = nil;
            id<MTLBuffer> profile_staging     = nil;
            id<MTLBuffer> destination_staging = nil;
            id<MTLHeap>   heap                = nil;

            if (private_heap)
            {
                source_staging  = [impl_->hardware.device newBufferWithBytes:source.data()
                                                                      length:field_bytes
                                                                     options:MTLResourceStorageModeShared];
                profile_staging = [impl_->hardware.device newBufferWithBytes:profile.data()
                                                                      length:profile_bytes
                                                                     options:MTLResourceStorageModeShared];
                destination_staging =
                    [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared];
                if (source_staging == nil || profile_staging == nil || destination_staging == nil)
                    return false;

                const MTLResourceOptions options = MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked;
                const MTLSizeAndAlign    source_size =
                    [impl_->hardware.device heapBufferSizeAndAlignWithLength:field_bytes options:options];
                const MTLSizeAndAlign profile_size =
                    [impl_->hardware.device heapBufferSizeAndAlignWithLength:profile_bytes options:options];
                const auto append_heap_resource =
                    [](NSUInteger current, MTLSizeAndAlign resource, NSUInteger& next) -> bool
                {
                    if (resource.align == 0)
                        return false;
                    const NSUInteger remainder = current % resource.align;
                    const NSUInteger padding   = remainder == 0 ? 0 : resource.align - remainder;
                    if (current > std::numeric_limits<NSUInteger>::max() - padding ||
                        current + padding > std::numeric_limits<NSUInteger>::max() - resource.size)
                        return false;
                    next = current + padding + resource.size;
                    return true;
                };
                NSUInteger heap_bytes = 0;
                if (!append_heap_resource(heap_bytes, source_size, heap_bytes) ||
                    !append_heap_resource(heap_bytes, profile_size, heap_bytes) ||
                    !append_heap_resource(heap_bytes, source_size, heap_bytes))
                    return false;

                MTLHeapDescriptor* descriptor = [[MTLHeapDescriptor alloc] init];
                descriptor.size               = heap_bytes;
                descriptor.storageMode        = MTLStorageModePrivate;
                descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
                heap                          = [impl_->hardware.device newHeapWithDescriptor:descriptor];
                if (heap == nil)
                    return false;
                source_buffer      = [heap newBufferWithLength:field_bytes options:options];
                profile_buffer     = [heap newBufferWithLength:profile_bytes options:options];
                destination_buffer = [heap newBufferWithLength:field_bytes options:options];
                if (source_buffer == nil || profile_buffer == nil || destination_buffer == nil)
                    return false;
                measurement.heap_bytes = static_cast<std::size_t>(heap_bytes);
            }
            else
            {
                source_buffer  = [impl_->hardware.device newBufferWithBytes:source.data()
                                                                     length:field_bytes
                                                                    options:MTLResourceStorageModeShared];
                profile_buffer = [impl_->hardware.device newBufferWithBytes:profile.data()
                                                                     length:profile_bytes
                                                                    options:MTLResourceStorageModeShared];
                destination_buffer =
                    [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared];
                if (source_buffer == nil || profile_buffer == nil || destination_buffer == nil)
                    return false;
            }

            const auto           wall_start = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> command    = impl_->command_buffer();
            if (command == nil)
                return false;
            if (private_heap)
            {
                id<MTLBlitCommandEncoder> upload = [command blitCommandEncoder];
                if (upload == nil)
                    return false;
                [upload copyFromBuffer:source_staging
                          sourceOffset:0
                              toBuffer:source_buffer
                     destinationOffset:0
                                  size:field_bytes];
                [upload copyFromBuffer:profile_staging
                          sourceOffset:0
                              toBuffer:profile_buffer
                     destinationOffset:0
                                  size:profile_bytes];
                [upload endEncoding];
            }

            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (encoder == nil)
                return false;
            [encoder setComputePipelineState:impl_->pipeline(KernelId::DomainRecyclingHomogeneousFieldShift)];
            [encoder setBuffer:source_buffer offset:0 atIndex:0];
            [encoder setBuffer:profile_buffer offset:0 atIndex:1];
            [encoder setBuffer:destination_buffer offset:0 atIndex:2];
            [encoder setBytes:&constants length:sizeof(constants) atIndex:3];
            for (int iteration = 0; iteration < config.iterations; ++iteration)
                [encoder dispatchThreads:grid threadsPerThreadgroup:group];
            [encoder endEncoding];

            if (private_heap)
            {
                id<MTLBlitCommandEncoder> download = [command blitCommandEncoder];
                if (download == nil)
                    return false;
                [download copyFromBuffer:destination_buffer
                            sourceOffset:0
                                toBuffer:destination_staging
                       destinationOffset:0
                                    size:field_bytes];
                [download endEncoding];
            }
            [command commit];
            [command waitUntilCompleted];
            const auto wall_end = std::chrono::steady_clock::now();
            if (command.status != MTLCommandBufferStatusCompleted)
                return false;
            const NSTimeInterval gpu_start = command.GPUStartTime;
            const NSTimeInterval gpu_end   = command.GPUEndTime;
            if (gpu_start <= 0.0 || gpu_end < gpu_start)
                return false;
            measurement.device_ms = static_cast<double>((gpu_end - gpu_start) * 1000.0);
            measurement.wall_ms   = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
            const void* output    = private_heap ? destination_staging.contents : destination_buffer.contents;
            if (output == nullptr)
                return false;
            measurement.output.resize(cells);
            std::memcpy(measurement.output.data(), output, field_bytes);
            return std::memcmp(measurement.output.data(), expected.data(), field_bytes) == 0;
        };

        for (int warmup = 0; warmup < config.warmups; ++warmup)
        {
            Measurement shared_warmup;
            Measurement private_warmup;
            if (!execute(false, shared_warmup) || !execute(true, private_warmup))
                return false;
        }

        std::vector<double> shared_device;
        std::vector<double> shared_wall;
        std::vector<double> private_device;
        std::vector<double> private_wall;
        shared_device.reserve(static_cast<std::size_t>(config.samples));
        shared_wall.reserve(static_cast<std::size_t>(config.samples));
        private_device.reserve(static_cast<std::size_t>(config.samples));
        private_wall.reserve(static_cast<std::size_t>(config.samples));
        std::size_t private_heap_bytes = 0;
        for (int sample = 0; sample < config.samples; ++sample)
        {
            Measurement shared_measurement;
            Measurement private_measurement;
            const bool  shared_first = sample % 2 == 0;
            if (shared_first)
            {
                if (!execute(false, shared_measurement) || !execute(true, private_measurement))
                    return false;
            }
            else if (!execute(true, private_measurement) || !execute(false, shared_measurement))
            {
                return false;
            }
            if (private_heap_bytes != 0 && private_heap_bytes != private_measurement.heap_bytes)
                return false;
            private_heap_bytes = private_measurement.heap_bytes;
            shared_device.push_back(shared_measurement.device_ms);
            shared_wall.push_back(shared_measurement.wall_ms);
            private_device.push_back(private_measurement.device_ms);
            private_wall.push_back(private_measurement.wall_ms);
        }

        const auto percentile = [](std::vector<double> values, double quantile)
        {
            std::sort(values.begin(), values.end());
            const double      rank  = std::ceil(quantile * static_cast<double>(values.size()));
            const std::size_t index = static_cast<std::size_t>(std::max(1.0, rank)) - 1;
            return values[std::min(index, values.size() - 1)];
        };
        result.field_bytes                             = field_bytes;
        result.private_heap_bytes                      = private_heap_bytes;
        result.dispatches_per_sample                   = static_cast<std::uint64_t>(config.iterations);
        result.threadgroup_width                       = static_cast<std::uint64_t>(config.threadgroup_width);
        result.shared_bitwise_correct                  = true;
        result.private_bitwise_correct                 = true;
        result.shared.median_device_milliseconds       = percentile(shared_device, 0.5);
        result.shared.p95_device_milliseconds          = percentile(shared_device, 0.95);
        result.shared.median_wall_milliseconds         = percentile(shared_wall, 0.5);
        result.shared.p95_wall_milliseconds            = percentile(shared_wall, 0.95);
        result.shared.command_buffers                  = static_cast<std::uint64_t>(config.samples);
        result.private_heap.median_device_milliseconds = percentile(private_device, 0.5);
        result.private_heap.p95_device_milliseconds    = percentile(private_device, 0.95);
        result.private_heap.median_wall_milliseconds   = percentile(private_wall, 0.5);
        result.private_heap.p95_wall_milliseconds      = percentile(private_wall, 0.95);
        result.private_heap.gpu_upload_bytes           = static_cast<std::uint64_t>(field_bytes + profile_bytes);
        result.private_heap.gpu_download_bytes         = static_cast<std::uint64_t>(field_bytes);
        result.private_heap.command_buffers            = static_cast<std::uint64_t>(config.samples);
        return true;
    }
}
