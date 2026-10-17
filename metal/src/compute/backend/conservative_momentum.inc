bool MetalComputeBackend::dispatch_conservative_momentum(const ConservativeMomentumDispatch& d)
{
    @autoreleasepool
    {
        namespace contract = nimbus::compute::conservative_momentum;
        if (!contract::validate(d) || !ready(KernelId::ConservativeMomentum))
            return false;
        const auto cells        = contract::cells(d);
        const auto input_bytes  = cells * ConservativeMomentumDispatch::input_stride * sizeof(float);
        const auto output_bytes = contract::output_values(d) * sizeof(float);
        if (input_bytes > impl_->max_buffer_bytes() || output_bytes > impl_->max_buffer_bytes())
            return false;
        id<MTLBuffer> input =
            [impl_->hardware.device newBufferWithBytes:d.input length:input_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> output =
            [impl_->hardware.device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
        auto pipeline = impl_->pipeline(KernelId::ConservativeMomentum);
        if (input == nil || output == nil || pipeline == nil || pipeline.maxTotalThreadsPerThreadgroup == 0)
            return false;
        id<MTLCommandBuffer> command = impl_->command_buffer();
        if (command == nil)
            return false;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil)
            return false;
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:output offset:0 atIndex:1];
        [encoder setBytes:&d.constants length:sizeof(d.constants) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command))
            return false;
        ++impl_->telemetry.completed_conservative_momentum_dispatches;
        const auto* values = static_cast<const float*>(output.contents);
        if (!values || !contract::finite_candidate(values, d))
            return false;
        std::copy_n(values, contract::output_values(d), d.output);
        return true;
    }
}
