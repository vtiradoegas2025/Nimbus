bool MetalComputeBackend::dispatch_cgrid_boundary(const CgridBoundaryDispatch& d)
{
    @autoreleasepool
    {
        namespace contract = nimbus::compute::cgrid_boundary;
        if (!contract::validate(d) || !ready(KernelId::AcousticBcFaceCartesianCgrid))
            return false;
        const auto count = contract::cells(d), waves = contract::waves(d);
        if (std::max(count, waves) * sizeof(float) > impl_->max_buffer_bytes())
            return false;
        std::array<id<MTLBuffer>, 6> buffers{};
        for (std::size_t f = 0; f < 6; ++f)
        {
            buffers[f] = [impl_->hardware.device newBufferWithBytes:f == 5 ? d.wave : d.fields[f]
                                                             length:(f == 5 ? waves : count) * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
            if (buffers[f] == nil)
                return false;
        }
        auto pipeline = impl_->pipeline(KernelId::AcousticBcFaceCartesianCgrid);
        if (pipeline == nil || pipeline.maxTotalThreadsPerThreadgroup == 0)
            return false;
        id<MTLCommandBuffer> command = impl_->command_buffer();
        if (command == nil)
            return false;
        auto constants = d.constants;
        for (int face = 0; face < 5; ++face)
        {
            constants.face                       = face;
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (encoder == nil)
                return false;
            [encoder setComputePipelineState:pipeline];
            for (NSUInteger f = 0; f < 6; ++f)
                [encoder setBuffer:buffers[f] offset:0 atIndex:f];
            [encoder setBytes:&constants length:sizeof(constants) atIndex:6];
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(
                                          std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
            [encoder endEncoding];
        }
        if (!impl_->finish(command))
            return false;
        std::array<const float*, 6> result;
        for (std::size_t f = 0; f < 6; ++f)
        {
            result[f] = static_cast<const float*>(buffers[f].contents);
            if (!result[f])
                return false;
        }
        if (!contract::finite_candidate(result, d))
            return false;
        for (std::size_t f = 0; f < 5; ++f)
            std::copy_n(result[f], count, d.output[f]);
        std::copy_n(result[5], waves, d.wave_output);
        impl_->telemetry.completed_acoustic_boundary_face_dispatches += 5;
        return true;
    }
}
