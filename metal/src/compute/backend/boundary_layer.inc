BoundaryLayerComputeKernels* MetalComputeBackend::boundary_layer_kernels()
{
    return ready(KernelId::MynnLevel25Stability) ? this : nullptr;
}

const BoundaryLayerComputeKernels* MetalComputeBackend::boundary_layer_kernels() const
{
    return ready(KernelId::MynnLevel25Stability) ? this : nullptr;
}

bool MetalComputeBackend::supports(BoundaryLayerKernelOp op) const
{
    switch (op)
    {
        case BoundaryLayerKernelOp::SlabColumn:
            return ready(KernelId::SlabColumn);
        case BoundaryLayerKernelOp::MynnLevel25Stability:
            return ready(KernelId::MynnLevel25Stability);
    }
    return false;
}

bool MetalComputeBackend::dispatch_mynn_level25_stability(
    const MynnLevel25StabilityDispatch& dispatch)
{
    @autoreleasepool
    {
        std::size_t cells = 0;
        std::size_t field_bytes = 0;
        if (!supports(BoundaryLayerKernelOp::MynnLevel25Stability) ||
            dispatch.gm_s2 == nullptr || dispatch.gh_s2 == nullptr ||
            dispatch.mixing_length_m == nullptr || dispatch.q_squared_m2s2 == nullptr ||
            dispatch.sm == nullptr || dispatch.sh == nullptr ||
            dispatch.gradient_richardson == nullptr || dispatch.flux_richardson == nullptr ||
            dispatch.equilibrium_limiter == nullptr ||
            !checked_bytes(dispatch.count, 1, 1, cells, field_bytes) ||
            field_bytes > impl_->max_buffer_bytes())
        {
            return false;
        }

        const void* inputs[4] = {dispatch.gm_s2, dispatch.gh_s2,
                                 dispatch.mixing_length_m, dispatch.q_squared_m2s2};
        void* outputs[5] = {dispatch.sm, dispatch.sh, dispatch.gradient_richardson,
                            dispatch.flux_richardson, dispatch.equilibrium_limiter};
        for (int output = 0; output < 5; ++output)
        {
            for (int prior = 0; prior < output; ++prior)
                if (outputs[output] == outputs[prior]) return false;
            for (const void* input : inputs)
                if (outputs[output] == input) return false;
        }
        for (std::size_t index = 0; index < cells; ++index)
        {
            if (!std::isfinite(dispatch.gm_s2[index]) || dispatch.gm_s2[index] < 0.0f ||
                !std::isfinite(dispatch.gh_s2[index]) ||
                !std::isfinite(dispatch.mixing_length_m[index]) ||
                dispatch.mixing_length_m[index] <= 0.0f ||
                !std::isfinite(dispatch.q_squared_m2s2[index]) ||
                dispatch.q_squared_m2s2[index] <= 0.0f)
            {
                return false;
            }
        }

        id<MTLBuffer> input_buffers[4] = {
            [impl_->hardware.device newBufferWithBytes:dispatch.gm_s2 length:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithBytes:dispatch.gh_s2 length:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithBytes:dispatch.mixing_length_m length:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithBytes:dispatch.q_squared_m2s2 length:field_bytes options:MTLResourceStorageModeShared],
        };
        id<MTLBuffer> output_buffers[5] = {
            [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared],
        };
        for (id<MTLBuffer> buffer : input_buffers) if (buffer == nil) return false;
        for (id<MTLBuffer> buffer : output_buffers) if (buffer == nil) return false;

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil) return false;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) return false;

        MynnLevel25StabilityConstants constants{};
        constants.count = dispatch.count;
        [encoder setComputePipelineState:impl_->pipeline(KernelId::MynnLevel25Stability)];
        for (NSUInteger index = 0; index < 4; ++index)
            [encoder setBuffer:input_buffers[index] offset:0 atIndex:index];
        for (NSUInteger index = 0; index < 5; ++index)
            [encoder setBuffer:output_buffers[index] offset:0 atIndex:index + 4];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:9];

        const NSUInteger maximum =
            impl_->pipeline(KernelId::MynnLevel25Stability).maxTotalThreadsPerThreadgroup;
        if (maximum == 0)
        {
            [encoder endEncoding];
            return false;
        }
        const NSUInteger width = std::min<NSUInteger>(64, maximum);
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command_buffer)) return false;

        for (std::size_t index = 0; index < cells; ++index)
        {
            const float sm_value = static_cast<const float*>(output_buffers[0].contents)[index];
            const float sh_value = static_cast<const float*>(output_buffers[1].contents)[index];
            const float ri_value = static_cast<const float*>(output_buffers[2].contents)[index];
            const float rf_value = static_cast<const float*>(output_buffers[3].contents)[index];
            const float qdiv_value = static_cast<const float*>(output_buffers[4].contents)[index];
            if (!std::isfinite(sm_value) || sm_value < 0.0f ||
                !std::isfinite(sh_value) || sh_value < 0.0f ||
                !std::isfinite(ri_value) || !std::isfinite(rf_value) ||
                !std::isfinite(qdiv_value) || qdiv_value <= 0.0f || qdiv_value > 1.0f)
            {
                return false;
            }
        }
        for (int output = 0; output < 5; ++output)
            std::memcpy(outputs[output], output_buffers[output].contents, field_bytes);
        ++impl_->telemetry.completed_mynn_level25_stability_dispatches;
        return true;
    }
}
