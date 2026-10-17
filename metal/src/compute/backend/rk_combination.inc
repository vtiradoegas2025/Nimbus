bool MetalComputeBackend::supports_rk_combination() const
{
    return ready(nimbus::compute::KernelId::RkCombination);
}

RkCombinationStatus MetalComputeBackend::dispatch_rk_combination(const RkCombinationDispatch& d)
{
    @autoreleasepool
    {
        namespace contract = nimbus::compute::rk_combination;
        const auto cells   = d.cells;
        if (!contract::validate(d))
            return RkCombinationStatus::InvalidInput;
        if (!supports_rk_combination() || cells * sizeof(float) > impl_->max_buffer_bytes())
            return RkCombinationStatus::Unavailable;
        const auto    constants = contract::constants(d);
        const auto    inputs    = contract::inputs(d);
        id<MTLBuffer> buffers[13];
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
        for (std::size_t f = inputs.size(); f < 13; ++f)
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
        auto pipeline = impl_->pipeline(KernelId::RkCombination);
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
        for (NSUInteger f = 0; f < 13; ++f)
        {
            [encoder setBuffer:buffers[f] offset:0 atIndex:f];
        }
        [encoder setBytes:&constants length:sizeof(constants) atIndex:13];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command))
        {
            return RkCombinationStatus::Unavailable;
        }
        ++impl_->telemetry.completed_rk_combination_dispatches;
        std::array<const float*, 4> views;
        for (std::size_t f = 0; f < views.size(); ++f)
        {
            views[f] = static_cast<const float*>(buffers[9 + f].contents);
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
