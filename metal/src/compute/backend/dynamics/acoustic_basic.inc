// Scalar acoustic pressure/momentum dispatches and their batched form.
bool MetalComputeBackend::dispatch_acoustic_pressure(const AcousticPressureDispatch& d)
{
    @autoreleasepool
    {
        std::size_t  cells = 0, bytes = 0;
        const float* inputs[5] = {d.u_data, d.v_data, d.w_data, d.rho_in, d.p_in};
        if (!supports(DynamicsKernelOp::AcousticPressure) || d.nr < 3 || d.nth < 3 || d.nz < 3 || !d.rho_out ||
            !d.p_out || d.rho_out == d.p_out || !checked_bytes(d.nr, d.nth, d.nz, cells, bytes) ||
            !std::isfinite(d.dr) || d.dr <= 0 || !std::isfinite(d.dtheta) || d.dtheta <= 0 || !std::isfinite(d.dz) ||
            d.dz <= 0 || !std::isfinite(d.gamma_val) || d.gamma_val <= 0 || !std::isfinite(d.dt_small) ||
            d.dt_small < 0 || !std::isfinite(d.rho_floor) || d.rho_floor <= 0 || !std::isfinite(d.p_floor) ||
            d.p_floor <= 0)
            return false;
        for (const float* input : inputs)
        {
            if (!input || d.rho_out == input || d.p_out == input)
                return false;
        }
        id<MTLBuffer> in[5] = {};
        for (int i = 0; i < 5; ++i)
        {
            in[i] =
                [impl_->hardware.device newBufferWithBytes:inputs[i] length:bytes options:MTLResourceStorageModeShared];
            if (!in[i])
                return false;
        }
        id<MTLBuffer> out[2] = {
            [impl_->hardware.device newBufferWithLength:bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithLength:bytes options:MTLResourceStorageModeShared]};
        if (!out[0] || !out[1])
            return false;
        AcousticAgridPressureConstants c{};
        c.nr                                 = d.nr;
        c.nth                                = d.nth;
        c.nz                                 = d.nz;
        c.dr                                 = d.dr;
        c.dtheta                             = d.dtheta;
        c.dz_val                             = d.dz;
        c.gamma_val                          = d.gamma_val;
        c.dt_small                           = d.dt_small;
        c.rho_floor                          = d.rho_floor;
        c.p_floor                            = d.p_floor;
        id<MTLCommandBuffer>         command = impl_->command_buffer();
        id<MTLComputeCommandEncoder> e       = command ? [command computeCommandEncoder] : nil;
        if (!e)
            return false;
        NSUInteger maximum = impl_->pipeline(KernelId::AcousticPressureCartesian).maxTotalThreadsPerThreadgroup;
        if (!maximum)
        {
            [e endEncoding];
            return false;
        }
        [e setComputePipelineState:impl_->pipeline(KernelId::AcousticPressureCartesian)];
        for (NSUInteger i = 0; i < 5; ++i)
            [e setBuffer:in[i] offset:0 atIndex:i];
        [e setBuffer:out[0] offset:0 atIndex:5];
        [e setBuffer:out[1] offset:0 atIndex:6];
        [e setBytes:&c length:sizeof(c) atIndex:7];
        [e dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [e endEncoding];
        if (!impl_->finish(command))
            return false;
        std::memcpy(d.rho_out, out[0].contents, bytes);
        std::memcpy(d.p_out, out[1].contents, bytes);
        ++impl_->telemetry.completed_acoustic_agrid_pressure_dispatches;
        return true;
    }
}

bool MetalComputeBackend::dispatch_acoustic_momentum(const AcousticMomentumDispatch& d)
{
    @autoreleasepool
    {
        std::size_t  cells = 0, bytes = 0;
        const float* inputs[5]  = {d.rho_data, d.p_data, d.u_in, d.v_in, d.w_in};
        float*       outputs[3] = {d.u_out, d.v_out, d.w_out};
        if (!supports(DynamicsKernelOp::AcousticMomentum) || d.nr < 3 || d.nth < 3 || d.nz < 3 ||
            !checked_bytes(d.nr, d.nth, d.nz, cells, bytes) || !std::isfinite(d.dr) || d.dr <= 0 ||
            !std::isfinite(d.dtheta) || d.dtheta <= 0 || !std::isfinite(d.dz) || d.dz <= 0 ||
            !std::isfinite(d.dt_small) || d.dt_small < 0 || !std::isfinite(d.wind_clamp_h) || d.wind_clamp_h < 0 ||
            !std::isfinite(d.wind_clamp_v) || d.wind_clamp_v < 0)
            return false;
        for (const float* input : inputs)
            if (!input)
                return false;
        for (int o = 0; o < 3; ++o)
        {
            if (!outputs[o])
                return false;
            for (int p = 0; p < o; ++p)
                if (outputs[o] == outputs[p])
                    return false;
            for (const float* input : inputs)
                if (outputs[o] == input)
                    return false;
        }
        id<MTLBuffer> in[5]  = {};
        id<MTLBuffer> out[3] = {};
        for (int i = 0; i < 5; ++i)
        {
            in[i] =
                [impl_->hardware.device newBufferWithBytes:inputs[i] length:bytes options:MTLResourceStorageModeShared];
            if (!in[i])
                return false;
        }
        for (int i = 0; i < 3; ++i)
        {
            out[i] = [impl_->hardware.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!out[i])
                return false;
        }
        AcousticAgridMomentumConstants c{};
        c.nr                                 = d.nr;
        c.nth                                = d.nth;
        c.nz                                 = d.nz;
        c.dr                                 = d.dr;
        c.dtheta                             = d.dtheta;
        c.dz_val                             = d.dz;
        c.dt_small                           = d.dt_small;
        c.wind_clamp_h                       = d.wind_clamp_h;
        c.wind_clamp_v                       = d.wind_clamp_v;
        id<MTLCommandBuffer>         command = impl_->command_buffer();
        id<MTLComputeCommandEncoder> e       = command ? [command computeCommandEncoder] : nil;
        if (!e)
            return false;
        NSUInteger maximum = impl_->pipeline(KernelId::AcousticMomentumCartesian).maxTotalThreadsPerThreadgroup;
        if (!maximum)
        {
            [e endEncoding];
            return false;
        }
        [e setComputePipelineState:impl_->pipeline(KernelId::AcousticMomentumCartesian)];
        for (NSUInteger i = 0; i < 5; ++i)
            [e setBuffer:in[i] offset:0 atIndex:i];
        for (NSUInteger i = 0; i < 3; ++i)
            [e setBuffer:out[i] offset:0 atIndex:i + 5];
        [e setBytes:&c length:sizeof(c) atIndex:8];
        [e dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [e endEncoding];
        if (!impl_->finish(command))
            return false;
        for (int i = 0; i < 3; ++i)
            std::memcpy(outputs[i], out[i].contents, bytes);
        ++impl_->telemetry.completed_acoustic_agrid_momentum_dispatches;
        return true;
    }
}

bool MetalComputeBackend::dispatch_acoustic_substeps_batched(const AcousticSubstepsBatchedDispatch& d)
{
    @autoreleasepool
    {
        std::size_t cells = 0, bytes = 0;
        float*      host[5] = {d.u, d.v, d.w, d.rho, d.p};
        if (!supports(DynamicsKernelOp::AcousticSubstepsBatched) || d.n_substeps <= 0 || d.n_substeps > 100000 ||
            d.nr < 3 || d.nth < 3 || d.nz < 3 || !checked_bytes(d.nr, d.nth, d.nz, cells, bytes) ||
            !std::isfinite(d.dr) || d.dr <= 0 || !std::isfinite(d.dtheta) || d.dtheta <= 0 || !std::isfinite(d.dz) ||
            d.dz <= 0 || !std::isfinite(d.gamma_val) || d.gamma_val <= 0 || !std::isfinite(d.dt_small) ||
            d.dt_small < 0 || !std::isfinite(d.rho_floor) || d.rho_floor <= 0 || !std::isfinite(d.p_floor) ||
            d.p_floor <= 0 || !std::isfinite(d.wind_clamp_h) || d.wind_clamp_h < 0 || !std::isfinite(d.wind_clamp_v) ||
            d.wind_clamp_v < 0)
            return false;
        for (int i = 0; i < 5; ++i)
        {
            if (!host[i])
                return false;
            for (int p = 0; p < i; ++p)
                if (host[i] == host[p])
                    return false;
        }
        id<MTLBuffer> current[5] = {}, candidate[5] = {};
        for (int i = 0; i < 5; ++i)
        {
            current[i] =
                [impl_->hardware.device newBufferWithBytes:host[i] length:bytes options:MTLResourceStorageModeShared];
            candidate[i] = [impl_->hardware.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!current[i] || !candidate[i])
                return false;
        }
        AcousticAgridPressureConstants pc{};
        pc.nr        = d.nr;
        pc.nth       = d.nth;
        pc.nz        = d.nz;
        pc.dr        = d.dr;
        pc.dtheta    = d.dtheta;
        pc.dz_val    = d.dz;
        pc.gamma_val = d.gamma_val;
        pc.dt_small  = d.dt_small;
        pc.rho_floor = d.rho_floor;
        pc.p_floor   = d.p_floor;
        AcousticAgridMomentumConstants mc{};
        mc.nr           = d.nr;
        mc.nth          = d.nth;
        mc.nz           = d.nz;
        mc.dr           = d.dr;
        mc.dtheta       = d.dtheta;
        mc.dz_val       = d.dz;
        mc.dt_small     = d.dt_small;
        mc.wind_clamp_h = d.wind_clamp_h;
        mc.wind_clamp_v = d.wind_clamp_v;
        NSUInteger pmax = impl_->pipeline(KernelId::AcousticPressureCartesian).maxTotalThreadsPerThreadgroup,
                   mmax = impl_->pipeline(KernelId::AcousticMomentumCartesian).maxTotalThreadsPerThreadgroup;
        if (!pmax || !mmax)
            return false;
        id<MTLCommandBuffer> command = impl_->command_buffer();
        if (!command)
            return false;
        for (int step = 0; step < d.n_substeps; ++step)
        {
            id<MTLComputeCommandEncoder> pe = [command computeCommandEncoder];
            if (!pe)
                return false;
            [pe setComputePipelineState:impl_->pipeline(KernelId::AcousticPressureCartesian)];
            for (NSUInteger i = 0; i < 5; ++i)
                [pe setBuffer:current[i] offset:0 atIndex:i];
            [pe setBuffer:candidate[3] offset:0 atIndex:5];
            [pe setBuffer:candidate[4] offset:0 atIndex:6];
            [pe setBytes:&pc length:sizeof(pc) atIndex:7];
            [pe dispatchThreads:MTLSizeMake(cells, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, pmax), 1, 1)];
            [pe endEncoding];
            id<MTLComputeCommandEncoder> me = [command computeCommandEncoder];
            if (!me)
                return false;
            [me setComputePipelineState:impl_->pipeline(KernelId::AcousticMomentumCartesian)];
            [me setBuffer:candidate[3] offset:0 atIndex:0];
            [me setBuffer:candidate[4] offset:0 atIndex:1];
            for (NSUInteger i = 0; i < 3; ++i)
                [me setBuffer:current[i] offset:0 atIndex:i + 2];
            for (NSUInteger i = 0; i < 3; ++i)
                [me setBuffer:candidate[i] offset:0 atIndex:i + 5];
            [me setBytes:&mc length:sizeof(mc) atIndex:8];
            [me dispatchThreads:MTLSizeMake(cells, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, mmax), 1, 1)];
            [me endEncoding];
            for (int i = 0; i < 5; ++i)
                std::swap(current[i], candidate[i]);
        }
        if (!impl_->finish(command))
            return false;
        for (int i = 0; i < 5; ++i)
            std::memcpy(host[i], current[i].contents, bytes);
        impl_->telemetry.completed_acoustic_agrid_pressure_dispatches += static_cast<std::uint64_t>(d.n_substeps);
        impl_->telemetry.completed_acoustic_agrid_momentum_dispatches += static_cast<std::uint64_t>(d.n_substeps);
        ++impl_->telemetry.completed_acoustic_agrid_batched_dispatches;
        return true;
    }
}

bool MetalComputeBackend::dispatch_acoustic_substep_fused(const AcousticSubstepFusedDispatch& d)
{
    AcousticSubstepsBatchedDispatch batch{d.u,
                                          d.v,
                                          d.w,
                                          d.rho,
                                          d.p,
                                          d.nr,
                                          d.nth,
                                          d.nz,
                                          d.dr,
                                          d.dtheta,
                                          d.dz,
                                          d.gamma_val,
                                          d.dt_small,
                                          1,
                                          d.rho_floor,
                                          d.p_floor,
                                          d.wind_clamp_h,
                                          d.wind_clamp_v};
    if (!dispatch_acoustic_substeps_batched(batch))
        return false;
    --impl_->telemetry.completed_acoustic_agrid_batched_dispatches;
    ++impl_->telemetry.completed_acoustic_agrid_fused_dispatches;
    return true;
}
