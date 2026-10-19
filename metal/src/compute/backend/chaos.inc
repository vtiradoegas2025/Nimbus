bool MetalComputeBackend::supports_earth_control_evolution() const
{
    return ready(KernelId::EarthControlEvolution);
}

bool MetalComputeBackend::dispatch_earth_control_evolution(
    const EarthControlEvolutionDispatch& dispatch)
{
    @autoreleasepool
    {
        if (!supports_earth_control_evolution() || dispatch.previous == nullptr ||
            dispatch.destination == nullptr || dispatch.control_i_count <= 0 ||
            dispatch.control_j_count <= 0 || dispatch.nz <= 0 ||
            !std::isfinite(dispatch.rho) || dispatch.rho < 0.0f || dispatch.rho > 1.0f ||
            static_cast<std::uint32_t>(dispatch.mode) >
                static_cast<std::uint32_t>(EarthControlEvolutionMode::Reconstruct))
            return false;

        const std::uint64_t cells64 =
            static_cast<std::uint64_t>(dispatch.control_i_count) *
            static_cast<std::uint64_t>(dispatch.control_j_count) *
            static_cast<std::uint64_t>(dispatch.nz);
        if (cells64 == 0 || cells64 > std::numeric_limits<std::size_t>::max() / sizeof(float))
            return false;
        const std::size_t cells = static_cast<std::size_t>(cells64);
        const std::size_t bytes = cells * sizeof(float);
        if (bytes > impl_->max_buffer_bytes())
            return false;

        id<MTLBuffer> previous = [impl_->hardware.device newBufferWithBytes:dispatch.previous
                                                                      length:bytes
                                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> destination = [impl_->hardware.device newBufferWithLength:bytes
                                                                        options:MTLResourceStorageModeShared];
        if (previous == nil || destination == nil)
            return false;

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil)
            return false;

        nimbus::compute::abi::EarthControlEvolutionPushConstants constants{};
        constants.seed = dispatch.seed;
        std::memcpy(&constants.member_bits, &dispatch.member_id, sizeof(constants.member_bits));
        constants.initial_process_key = dispatch.initial_process_key;
        constants.innovation_process_key = dispatch.innovation_process_key;
        constants.innovation_epoch = dispatch.innovation_epoch;
        constants.control_i_min = dispatch.control_i_min;
        constants.control_j_min = dispatch.control_j_min;
        constants.control_i_count = dispatch.control_i_count;
        constants.control_j_count = dispatch.control_j_count;
        constants.nz = dispatch.nz;
        constants.rho = static_cast<float>(dispatch.rho);
        constants.rho_low = static_cast<float>(dispatch.rho - constants.rho);
        constants.mode = static_cast<std::uint32_t>(dispatch.mode);

        id<MTLComputePipelineState> pipeline = impl_->pipeline(KernelId::EarthControlEvolution);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:previous offset:0 atIndex:0];
        [encoder setBuffer:destination offset:0 atIndex:1];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:2];
        const NSUInteger maximum = pipeline.maxTotalThreadsPerThreadgroup;
        if (maximum == 0)
        {
            [encoder endEncoding];
            return false;
        }
        const NSUInteger width = std::min<NSUInteger>(64, maximum);
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command_buffer))
            return false;

        std::memcpy(dispatch.destination, destination.contents, bytes);
        ++impl_->telemetry.completed_earth_control_evolution_dispatches;
        return true;
    }
}
