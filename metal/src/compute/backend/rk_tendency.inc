bool MetalComputeBackend::supports_rk_tendency() const
{
    return ready(nimbus::compute::KernelId::RkTendency);
}

RkCombinationStatus MetalComputeBackend::dispatch_rk_tendency(const RkTendencyDispatch& d)
{
    @autoreleasepool
    {
        namespace contract = nimbus::compute::rk_tendency;
        const auto cells   = contract::cells(d);
        if (!contract::validate(d))
            return RkCombinationStatus::InvalidInput;
        if (!supports_rk_tendency() || cells * sizeof(float) > impl_->max_buffer_bytes())
            return RkCombinationStatus::Unavailable;
        const auto    constants = contract::constants(d);
        const auto    inputs    = d.input;
        id<MTLBuffer> buffers[12];
        for (std::size_t f = 0; f < inputs.size(); ++f)
        {
            const std::size_t count = cells;
            if (count * sizeof(float) > impl_->max_buffer_bytes())
            {
                return RkCombinationStatus::Unavailable;
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
                return RkCombinationStatus::Unavailable;
            }
        }
        auto pipeline = impl_->pipeline(KernelId::RkTendency);
        if (pipeline == nil || pipeline.maxTotalThreadsPerThreadgroup == 0)
        {
            return RkCombinationStatus::Unavailable;
        }
        id<MTLCommandBuffer> command = impl_->command_buffer();
        if (command == nil)
        {
            return RkCombinationStatus::Unavailable;
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil)
        {
            return RkCombinationStatus::Unavailable;
        }
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger f = 0; f < 12; ++f)
        {
            [encoder setBuffer:buffers[f] offset:0 atIndex:f];
        }
        [encoder setBytes:&constants length:sizeof(constants) atIndex:12];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command))
        {
            return RkCombinationStatus::Unavailable;
        }
        ++impl_->telemetry.completed_rk_tendency_dispatches;
        std::array<const float*, 5> views;
        for (std::size_t f = 0; f < views.size(); ++f)
        {
            views[f] = static_cast<const float*>(buffers[7 + f].contents);
            if (!views[f])
            {
                return RkCombinationStatus::Unavailable;
            }
        }
        if (!contract::finite_candidate(views, cells))
        {
            return RkCombinationStatus::NonFinite;
        }
        const auto outputs = d.output;
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(views[f], cells, outputs[f]);
        }
        return RkCombinationStatus::Completed;
    }
}
