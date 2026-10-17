bool MetalComputeBackend::dispatch_thompson_moment_sedimentation(const ThompsonMomentSedimentationDispatch& d)
{
    @autoreleasepool
    {
        namespace contract = nimbus::compute::thompson_sedimentation;
        std::size_t cells = 0, columns = 0;
        if (!supports(MicrophysicsKernelOp::ThompsonMomentSedimentation) || !contract::validate(d, cells, columns) ||
            cells * sizeof(float) > impl_->max_buffer_bytes())
        {
            return false;
        }
        const auto    constants = contract::constants(d);
        const auto    inputs    = contract::inputs(d);
        id<MTLBuffer> buffers[14];
        for (std::size_t f = 0; f < inputs.size(); ++f)
        {
            buffers[f] = [impl_->hardware.device newBufferWithBytes:inputs[f]
                                                             length:cells * sizeof(float)
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
        auto pipeline = impl_->pipeline(KernelId::ThompsonMomentSedimentation);
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
        [encoder dispatchThreads:MTLSizeMake(columns, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command))
        {
            return false;
        }
        ++impl_->telemetry.completed_thompson_moment_sedimentation_dispatches;
        const auto* mass    = static_cast<const float*>(buffers[7].contents);
        const auto* number  = static_cast<const float*>(buffers[8].contents);
        const auto* surface = static_cast<const float*>(buffers[9].contents);
        if (mass == nullptr || number == nullptr || surface == nullptr ||
            !contract::valid_candidate(mass, number, surface, cells, columns))
        {
            return false;
        }
        std::copy_n(mass, cells, d.mass_tendency);
        std::copy_n(number, cells, d.number_tendency);
        std::copy_n(surface, columns, d.surface_mass);
        return true;
    }
}

bool MetalComputeBackend::dispatch_thompson_cloud_adjustment(const ThompsonCloudAdjustmentDispatch& d)
{
    @autoreleasepool
    {
        namespace contract = nimbus::compute::thompson_cloud;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::ThompsonCloudAdjustment) || !contract::validate(d, cells) ||
            cells * sizeof(float) > impl_->max_buffer_bytes())
        {
            return false;
        }
        const auto    constants = contract::constants(d);
        const auto    inputs    = contract::inputs(d);
        id<MTLBuffer> buffers[18];
        for (std::size_t f = 0; f < inputs.size(); ++f)
        {
            const std::size_t count = (f < 8 || f == 10) ? cells
                                      : f == 8 ? d.tables->activation().size()
                                               : d.tables->evaporation().size();
            if (count * sizeof(float) > impl_->max_buffer_bytes())
            {
                return false;
            }
            buffers[f] = [impl_->hardware.device newBufferWithBytes:inputs[f]
                                                             length:count * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        }
        for (std::size_t f = inputs.size(); f < 18; ++f)
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
        auto pipeline = impl_->pipeline(KernelId::ThompsonCloudAdjustment);
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
        for (NSUInteger f = 0; f < 18; ++f)
        {
            [encoder setBuffer:buffers[f] offset:0 atIndex:f];
        }
        [encoder setBytes:&constants length:sizeof(constants) atIndex:18];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command))
        {
            return false;
        }
        ++impl_->telemetry.completed_thompson_cloud_adjustment_dispatches;
        std::array<const float*, 7> views;
        for (std::size_t f = 0; f < views.size(); ++f)
        {
            views[f] = static_cast<const float*>(buffers[11 + f].contents);
            if (!views[f])
            {
                return false;
            }
        }
        if (!contract::valid_candidate(views, cells))
        {
            return false;
        }
        const auto outputs = contract::outputs(d);
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(views[f], cells, outputs[f]);
        }
        return true;
    }
}

bool MetalComputeBackend::dispatch_thompson_warm_rain(const ThompsonWarmRainDispatch& d)
{
    @autoreleasepool
    {
        namespace contract = nimbus::compute::thompson_warm;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::ThompsonWarmRain) || !contract::validate(d, cells) ||
            cells * sizeof(float) > impl_->max_buffer_bytes())
        {
            return false;
        }
        const auto    constants = contract::constants(d);
        const auto    inputs    = contract::inputs(d);
        id<MTLBuffer> buffers[24];
        for (std::size_t f = 0; f < inputs.size(); ++f)
        {
            const std::size_t count = f < 13 ? cells : d.coefficients->size();
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
        auto pipeline = impl_->pipeline(KernelId::ThompsonWarmRain);
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
        ++impl_->telemetry.completed_thompson_warm_rain_dispatches;
        std::array<const float*, 10> views;
        for (std::size_t f = 0; f < views.size(); ++f)
        {
            views[f] = static_cast<const float*>(buffers[14 + f].contents);
            if (!views[f])
            {
                return false;
            }
        }
        if (!contract::valid_candidate(d, views, cells))
        {
            return false;
        }
        const auto outputs = contract::outputs(d);
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(views[f], cells, outputs[f]);
        }
        return true;
    }
}

bool MetalComputeBackend::dispatch_thompson_cloud_phase(const ThompsonCloudPhaseDispatch& d)
{
    @autoreleasepool
    {
        namespace contract = nimbus::compute::thompson_phase;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::ThompsonCloudPhase) || !contract::validate(d, cells) ||
            cells * sizeof(float) > impl_->max_buffer_bytes())
        {
            return false;
        }
        const auto    constants = contract::constants(d);
        const auto    inputs    = contract::inputs(d);
        id<MTLBuffer> buffers[18];
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
        for (std::size_t f = inputs.size(); f < 18; ++f)
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
        auto pipeline = impl_->pipeline(KernelId::ThompsonCloudPhase);
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
        for (NSUInteger f = 0; f < 18; ++f)
        {
            [encoder setBuffer:buffers[f] offset:0 atIndex:f];
        }
        [encoder setBytes:&constants length:sizeof(constants) atIndex:18];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command))
        {
            return false;
        }
        ++impl_->telemetry.completed_thompson_cloud_phase_dispatches;
        std::array<const float*, 6> views;
        for (std::size_t f = 0; f < views.size(); ++f)
        {
            views[f] = static_cast<const float*>(buffers[12 + f].contents);
            if (!views[f])
            {
                return false;
            }
        }
        if (!contract::valid_candidate(d, views, cells))
        {
            return false;
        }
        const auto outputs = contract::outputs(d);
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(views[f], cells, outputs[f]);
        }
        return true;
    }
}

bool MetalComputeBackend::dispatch_thompson_freezing(const ThompsonFreezingDispatch& d)
{
    @autoreleasepool
    {
        namespace contract = nimbus::compute::thompson_freezing;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::ThompsonFreezing) || !contract::validate(d, cells) ||
            cells * sizeof(float) > impl_->max_buffer_bytes())
        {
            return false;
        }
        const auto    constants = contract::constants(d);
        const auto    inputs    = contract::inputs(d);
        id<MTLBuffer> buffers[29];
        for (std::size_t f = 0; f < inputs.size(); ++f)
        {
            const std::size_t count = f < 15 ? cells : d.tables->values().size();
            if (count * sizeof(float) > impl_->max_buffer_bytes())
            {
                return false;
            }
            buffers[f] = [impl_->hardware.device newBufferWithBytes:inputs[f]
                                                             length:count * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        }
        for (std::size_t f = inputs.size(); f < 29; ++f)
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
        auto pipeline = impl_->pipeline(KernelId::ThompsonFreezing);
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
        for (NSUInteger f = 0; f < 29; ++f)
        {
            [encoder setBuffer:buffers[f] offset:0 atIndex:f];
        }
        [encoder setBytes:&constants length:sizeof(constants) atIndex:29];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command))
        {
            return false;
        }
        ++impl_->telemetry.completed_thompson_freezing_dispatches;
        std::array<const float*, 13> views;
        for (std::size_t f = 0; f < views.size(); ++f)
        {
            views[f] = static_cast<const float*>(buffers[16 + f].contents);
            if (!views[f])
            {
                return false;
            }
        }
        if (!contract::valid_candidate(d, views, cells))
        {
            return false;
        }
        const auto outputs = contract::outputs(d);
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(views[f], cells, outputs[f]);
        }
        return true;
    }
}

bool MetalComputeBackend::dispatch_thompson_deposition(const ThompsonDepositionDispatch& d)
{
    @autoreleasepool
    {
        namespace contract = nimbus::compute::thompson_deposition;
        std::size_t cells  = 0;
        if (!supports(MicrophysicsKernelOp::ThompsonDeposition) || !contract::validate(d, cells) ||
            cells * sizeof(float) > impl_->max_buffer_bytes())
        {
            return false;
        }
        const auto    constants = contract::constants(d);
        const auto    inputs    = contract::inputs(d);
        id<MTLBuffer> buffers[26];
        for (std::size_t f = 0; f < inputs.size(); ++f)
        {
            const std::size_t count = f < 12 ? cells : d.tables->values().size();
            if (count * sizeof(float) > impl_->max_buffer_bytes())
            {
                return false;
            }
            buffers[f] = [impl_->hardware.device newBufferWithBytes:inputs[f]
                                                             length:count * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        }
        for (std::size_t f = inputs.size(); f < 26; ++f)
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
        auto pipeline = impl_->pipeline(KernelId::ThompsonDeposition);
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
        for (NSUInteger f = 0; f < 26; ++f)
        {
            [encoder setBuffer:buffers[f] offset:0 atIndex:f];
        }
        [encoder setBytes:&constants length:sizeof(constants) atIndex:26];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command))
        {
            return false;
        }
        ++impl_->telemetry.completed_thompson_deposition_dispatches;
        std::array<const float*, 13> views;
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
        const auto outputs = contract::outputs(d);
        for (std::size_t f = 0; f < outputs.size(); ++f)
        {
            std::copy_n(views[f], cells, outputs[f]);
        }
        return true;
    }
}

bool MetalComputeBackend::dispatch_thompson_completion(const ThompsonCompletionDispatch& d)
{
    @autoreleasepool
    {
        namespace contract = nimbus::compute::thompson_completion;
        if (!supports(MicrophysicsKernelOp::ThompsonCompletion) || !contract::validate(d) ||
            d.tables->values().size() * sizeof(float) > impl_->max_buffer_bytes())
            return false;
        if (impl_->thompson_completion_tables != d.tables || impl_->thompson_completion_coefficients == nil)
        {
            id<MTLBuffer> table = [impl_->hardware.device newBufferWithBytes:d.tables->values().data()
                                                                      length:d.tables->values().size() * sizeof(float)
                                                                     options:MTLResourceStorageModeShared];
            if (table == nil)
                return false;
            impl_->thompson_completion_coefficients = table;
            impl_->thompson_completion_tables       = d.tables;
        }
        id<MTLBuffer> input    = [impl_->hardware.device
            newBufferWithBytes:d.input
                        length:d.cells * nimbus::compute::thompson_completion_input_stride * sizeof(float)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> output   = [impl_->hardware.device newBufferWithLength:d.cells * 18 * sizeof(float)
                                                                     options:MTLResourceStorageModeShared];
        auto          pipeline = impl_->pipeline(KernelId::ThompsonCompletion);
        if (input == nil || output == nil || pipeline == nil)
            return false;
        id<MTLCommandBuffer> command = impl_->command_buffer();
        if (command == nil)
            return false;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil)
            return false;
        const auto pc = contract::constants(d);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:impl_->thompson_completion_coefficients offset:0 atIndex:1];
        [encoder setBuffer:output offset:0 atIndex:2];
        [encoder setBytes:&pc length:sizeof(pc) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(d.cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command))
            return false;
        const auto* values = static_cast<const float*>(output.contents);
        if (values == nullptr || !contract::valid_candidate(d, values))
            return false;
        std::copy_n(values, d.cells * 18, d.output);
        ++impl_->telemetry.completed_thompson_completion_dispatches;
        return true;
    }
}
