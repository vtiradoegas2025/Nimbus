bool MetalComputeBackend::dispatch_collocated_sgs(const CollocatedSgsDispatch& d)
{
    @autoreleasepool
    {
        namespace contract = nimbus::compute::collocated_sgs;
        const auto cells   = contract::cells(d);
        if (!contract::validate(d))
            return false;
        if (!supports(TurbulenceKernelOp::CollocatedSgs) || cells * sizeof(float) > impl_->max_buffer_bytes())
            return false;
        const auto    constants = contract::constants(d);
        const auto    inputs    = d.input;
        id<MTLBuffer> buffers[23];
        for (std::size_t f = 0; f < inputs.size(); ++f)
        {
            const std::size_t count = cells;
            if (count * sizeof(float) > impl_->max_buffer_bytes())
            {
                return false;
            }
            buffers[f] = [impl_->hardware.device newBufferWithBytes:inputs[f]
                                                             length:count * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        }
        for (std::size_t f = inputs.size(); f < 23; ++f)
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
        auto pipeline = impl_->pipeline(KernelId::CollocatedSgs);
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
        for (NSUInteger f = 0; f < 23; ++f)
        {
            [encoder setBuffer:buffers[f] offset:0 atIndex:f];
        }
        [encoder setBytes:&constants length:sizeof(constants) atIndex:23];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command))
        {
            return false;
        }
        ++impl_->telemetry.completed_collocated_sgs_dispatches;
        std::array<const float*, 8> views;
        for (std::size_t f = 0; f < views.size(); ++f)
        {
            views[f] = static_cast<const float*>(buffers[15 + f].contents);
            if (!views[f])
            {
                return false;
            }
        }
        if (!contract::finite_candidate(views, cells))
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
