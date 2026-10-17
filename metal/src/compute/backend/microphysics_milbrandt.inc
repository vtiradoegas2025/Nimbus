bool MetalComputeBackend::dispatch_milbrandt_sedimentation(const MilbrandtSedimentationDispatch& d)
{
    @autoreleasepool
    {
        namespace contract = nimbus::compute::milbrandt_sedimentation;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::MilbrandtSedimentation) || !contract::validate(d, cells) ||
            cells * sizeof(float) > impl_->max_buffer_bytes())
        {
            return false;
        }
        const auto         constants = contract::constants(d);
        const auto         table     = contract::coefficients(d);
        std::vector<float> scratch(12 * cells);
        const auto         inputs = contract::inputs(d, scratch.data(), table.data());
        id<MTLBuffer>      buffers[12];
        for (std::size_t f = 0; f < inputs.size(); ++f)
        {
            const std::size_t count = f == 7 ? scratch.size() : f == 8 ? table.size() : cells;
            if (count * sizeof(float) > impl_->max_buffer_bytes())
            {
                return false;
            }
            buffers[f] = [impl_->hardware.device newBufferWithBytes:inputs[f]
                                                             length:count * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        }
        for (std::size_t f = inputs.size(); f < 12; ++f)
        {
            buffers[f] =
                [impl_->hardware.device newBufferWithLength:cells * sizeof(float) options:MTLResourceStorageModeShared];
        }
        for (auto buffer : buffers)
        {
            if (buffer == nil)
            {
                return false;
            }
        }
        auto pipeline = impl_->pipeline(KernelId::MilbrandtSedimentation);
        if (pipeline == nil || pipeline.maxTotalThreadsPerThreadgroup == 0)
        {
            return false;
        }
        id<MTLCommandBuffer> command = impl_->command_buffer();
        if (command == nil)
        {
            return false;
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil)
        {
            return false;
        }
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger f = 0; f < 12; ++f)
        {
            [encoder setBuffer:buffers[f] offset:0 atIndex:f];
        }
        [encoder setBytes:&constants length:sizeof(constants) atIndex:12];
        [encoder dispatchThreads:MTLSizeMake(std::size_t(d.nx) * d.ny, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command))
        {
            return false;
        }
        ++impl_->telemetry.completed_milbrandt_sedimentation_dispatches;
        std::array<const float*, 3> views;
        for (std::size_t f = 0; f < views.size(); ++f)
        {
            views[f] = static_cast<const float*>(buffers[9 + f].contents);
            if (!views[f])
            {
                return false;
            }
        }
        if (!contract::valid_candidate(d, views, cells))
        {
            return false;
        }
        const auto outputs = d.output;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(views[f], cells, outputs[f]);
        }
        return true;
    }
}

bool MetalComputeBackend::dispatch_milbrandt_cold(const MilbrandtColdDispatch& d)
{
    @autoreleasepool
    {
        namespace contract = nimbus::compute::milbrandt_cold;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::MilbrandtCold) || !contract::validate(d, cells) ||
            cells * sizeof(float) > impl_->max_buffer_bytes())
        {
            return false;
        }
        const auto                        constants = contract::constants(d);
        const auto                        packed    = contract::pack(d, cells);
        const std::array<const float*, 2> inputs{packed[0].data(), packed[1].data()};
        id<MTLBuffer>                     buffers[16];
        for (std::size_t f = 0; f < inputs.size(); ++f)
        {
            const std::size_t count = packed[f].size();
            if (count * sizeof(float) > impl_->max_buffer_bytes())
            {
                return false;
            }
            buffers[f] = [impl_->hardware.device newBufferWithBytes:inputs[f]
                                                             length:count * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        }
        for (std::size_t f = inputs.size(); f < 16; ++f)
        {
            buffers[f] =
                [impl_->hardware.device newBufferWithLength:cells * sizeof(float) options:MTLResourceStorageModeShared];
        }
        for (auto buffer : buffers)
        {
            if (buffer == nil)
            {
                return false;
            }
        }
        auto pipeline = impl_->pipeline(KernelId::MilbrandtCold);
        if (pipeline == nil || pipeline.maxTotalThreadsPerThreadgroup == 0)
        {
            return false;
        }
        id<MTLCommandBuffer> command = impl_->command_buffer();
        if (command == nil)
        {
            return false;
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil)
        {
            return false;
        }
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger f = 0; f < 16; ++f)
        {
            [encoder setBuffer:buffers[f] offset:0 atIndex:f];
        }
        [encoder setBytes:&constants length:sizeof(constants) atIndex:16];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command))
        {
            return false;
        }
        ++impl_->telemetry.completed_milbrandt_cold_dispatches;
        std::array<const float*, 14> views;
        for (std::size_t f = 0; f < views.size(); ++f)
        {
            views[f] = static_cast<const float*>(buffers[2 + f].contents);
            if (!views[f])
            {
                return false;
            }
        }
        if (!contract::valid_candidate(d, views, cells))
        {
            return false;
        }
        const auto outputs = d.output;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(views[f], cells, outputs[f]);
        }
        return true;
    }
}

bool MetalComputeBackend::dispatch_milbrandt_melting(const MilbrandtMeltingDispatch& d)
{
    @autoreleasepool
    {
        namespace contract = nimbus::compute::milbrandt_melting;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::MilbrandtMelting) || !contract::validate(d, cells) ||
            cells * sizeof(float) > impl_->max_buffer_bytes())
        {
            return false;
        }
        const auto    constants = contract::constants(d);
        const auto    table     = contract::coefficients(d);
        const auto    inputs    = contract::inputs(d, table.data());
        id<MTLBuffer> buffers[24];
        for (std::size_t f = 0; f < inputs.size(); ++f)
        {
            const std::size_t count = f == 12 ? table.size() : cells;
            if (count * sizeof(float) > impl_->max_buffer_bytes())
            {
                return false;
            }
            buffers[f] = [impl_->hardware.device newBufferWithBytes:inputs[f]
                                                             length:count * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        }
        for (std::size_t f = inputs.size(); f < 24; ++f)
        {
            buffers[f] =
                [impl_->hardware.device newBufferWithLength:cells * sizeof(float) options:MTLResourceStorageModeShared];
        }
        for (auto buffer : buffers)
        {
            if (buffer == nil)
            {
                return false;
            }
        }
        auto pipeline = impl_->pipeline(KernelId::MilbrandtMelting);
        if (pipeline == nil || pipeline.maxTotalThreadsPerThreadgroup == 0)
        {
            return false;
        }
        id<MTLCommandBuffer> command = impl_->command_buffer();
        if (command == nil)
        {
            return false;
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil)
        {
            return false;
        }
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger f = 0; f < 24; ++f)
        {
            [encoder setBuffer:buffers[f] offset:0 atIndex:f];
        }
        [encoder setBytes:&constants length:sizeof(constants) atIndex:24];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command))
        {
            return false;
        }
        ++impl_->telemetry.completed_milbrandt_melting_dispatches;
        std::array<const float*, 11> views;
        for (std::size_t f = 0; f < views.size(); ++f)
        {
            views[f] = static_cast<const float*>(buffers[13 + f].contents);
            if (!views[f])
            {
                return false;
            }
        }
        if (!contract::valid_candidate(d, views, cells))
        {
            return false;
        }
        const auto outputs = d.output;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(views[f], cells, outputs[f]);
        }
        return true;
    }
}

bool MetalComputeBackend::dispatch_milbrandt_cloud_adjustment(const MilbrandtCloudAdjustmentDispatch& d)
{
    @autoreleasepool
    {
        namespace contract = nimbus::compute::milbrandt_cloud;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::MilbrandtCloudAdjustment) || !contract::validate(d, cells) ||
            contract::velocity_value_count(d, cells) > impl_->max_buffer_bytes() / sizeof(float))
        {
            return false;
        }
        const auto         constants = contract::constants(d);
        std::vector<float> velocity;
        const auto         inputs = contract::inputs(d, cells, velocity);
        id<MTLBuffer>      buffers[14];
        for (std::size_t f = 0; f < inputs.size(); ++f)
        {
            const std::size_t count = f == 6 ? contract::velocity_value_count(d, cells) : cells;
            if (count * sizeof(float) > impl_->max_buffer_bytes())
            {
                return false;
            }
            buffers[f] = [impl_->hardware.device newBufferWithBytes:inputs[f]
                                                             length:count * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        }
        for (std::size_t f = inputs.size(); f < 14; ++f)
        {
            buffers[f] =
                [impl_->hardware.device newBufferWithLength:cells * sizeof(float) options:MTLResourceStorageModeShared];
        }
        for (auto buffer : buffers)
        {
            if (buffer == nil)
            {
                return false;
            }
        }
        auto pipeline = impl_->pipeline(KernelId::MilbrandtCloudAdjustment);
        if (pipeline == nil || pipeline.maxTotalThreadsPerThreadgroup == 0)
        {
            return false;
        }
        id<MTLCommandBuffer> command = impl_->command_buffer();
        if (command == nil)
        {
            return false;
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil)
        {
            return false;
        }
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger f = 0; f < 14; ++f)
        {
            [encoder setBuffer:buffers[f] offset:0 atIndex:f];
        }
        [encoder setBytes:&constants length:sizeof(constants) atIndex:14];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command))
        {
            return false;
        }
        ++impl_->telemetry.completed_milbrandt_cloud_adjustment_dispatches;
        std::array<const float*, 7> views;
        for (std::size_t f = 0; f < views.size(); ++f)
        {
            views[f] = static_cast<const float*>(buffers[7 + f].contents);
            if (!views[f])
            {
                return false;
            }
        }
        if (!contract::valid_candidate(d, views, cells))
        {
            return false;
        }
        const auto outputs = d.output;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(views[f], cells, outputs[f]);
        }
        return true;
    }
}

bool MetalComputeBackend::dispatch_milbrandt_warm_rain(const MilbrandtWarmRainDispatch& d)
{
    @autoreleasepool
    {
        namespace contract = nimbus::compute::milbrandt_warm;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::MilbrandtWarmRain) || !contract::validate(d, cells) ||
            cells * sizeof(float) > impl_->max_buffer_bytes())
        {
            return false;
        }
        const auto    constants    = contract::constants(d);
        const auto    coefficients = contract::coefficients(d);
        const auto    inputs       = contract::inputs(d, coefficients.data());
        id<MTLBuffer> buffers[21];
        for (std::size_t f = 0; f < inputs.size(); ++f)
        {
            const std::size_t count = f < 14 ? cells : coefficients.size();
            if (count * sizeof(float) > impl_->max_buffer_bytes())
            {
                return false;
            }
            buffers[f] = [impl_->hardware.device newBufferWithBytes:inputs[f]
                                                             length:count * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        }
        for (std::size_t f = inputs.size(); f < 21; ++f)
        {
            buffers[f] =
                [impl_->hardware.device newBufferWithLength:cells * sizeof(float) options:MTLResourceStorageModeShared];
        }
        for (auto buffer : buffers)
        {
            if (buffer == nil)
            {
                return false;
            }
        }
        auto pipeline = impl_->pipeline(KernelId::MilbrandtWarmRain);
        if (pipeline == nil || pipeline.maxTotalThreadsPerThreadgroup == 0)
        {
            return false;
        }
        id<MTLCommandBuffer> command = impl_->command_buffer();
        if (command == nil)
        {
            return false;
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil)
        {
            return false;
        }
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger f = 0; f < 21; ++f)
        {
            [encoder setBuffer:buffers[f] offset:0 atIndex:f];
        }
        [encoder setBytes:&constants length:sizeof(constants) atIndex:21];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command))
        {
            return false;
        }
        ++impl_->telemetry.completed_milbrandt_warm_rain_dispatches;
        std::array<const float*, 6> views;
        for (std::size_t f = 0; f < views.size(); ++f)
        {
            views[f] = static_cast<const float*>(buffers[15 + f].contents);
            if (!views[f])
            {
                return false;
            }
        }
        if (!contract::valid_candidate(d, views, cells))
        {
            return false;
        }
        const auto outputs = d.output;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(views[f], cells, outputs[f]);
        }
        return true;
    }
}
