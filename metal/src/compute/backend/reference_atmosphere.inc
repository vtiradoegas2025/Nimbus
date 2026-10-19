bool MetalComputeBackend::supports_reference_atmosphere() const
{
    return ready(KernelId::ReferenceAtmosphere);
}
bool MetalComputeBackend::dispatch_reference_atmosphere(const ReferenceAtmosphereDispatch& d)
{
    @autoreleasepool
    {
        namespace contract = nimbus::compute::reference_atmosphere;
        if (!contract::validate(d) || !supports_reference_atmosphere())
            return false;
        const auto cells        = std::size_t(d.constants.count);
        const auto input_bytes  = contract::input_count(d) * sizeof(float);
        const auto output_bytes = cells * ReferenceAtmosphereDispatch::output_stride * sizeof(float);
        if (input_bytes > impl_->max_buffer_bytes() || output_bytes > impl_->max_buffer_bytes())
            return false;
        id<MTLBuffer> input =
            [impl_->hardware.device newBufferWithBytes:d.input length:input_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> output =
            [impl_->hardware.device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
        auto pipeline = impl_->pipeline(KernelId::ReferenceAtmosphere);
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
        ++impl_->telemetry.completed_reference_atmosphere_dispatches;
        const auto* values = static_cast<const float*>(output.contents);
        if (!values || !contract::finite_candidate(values, d))
            return false;
        std::copy_n(values, cells * ReferenceAtmosphereDispatch::output_stride, d.output);
        return true;
    }
}
